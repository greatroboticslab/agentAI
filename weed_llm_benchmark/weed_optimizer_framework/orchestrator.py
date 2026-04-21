"""
Orchestrator — Two modes of operation.

Mode 1 (Strategy): Brain proposes full strategy → rigid pipeline executes
Mode 2 (Agent): Brain decides one action at a time → sees result → decides next

Agent mode is the real tool-calling pattern: Brain controls the flow,
not a fixed pipeline. It can inspect labels, run VLM inference, adjust
parameters, and decide when to stop — all within a single round.
"""

import json
import os
import time
import logging
from datetime import datetime
from .config import Config
from .memory import Memory
from .monitor import QualityMonitor
from .brain import SuperBrain
from .tools import ToolRegistry
from .tools.label_gen import generate_consensus_labels
from .tools.yolo_trainer import train_yolo
from .tools.evaluator import evaluate_full
from .tools.vlm_pool import VLMPool
from .tools.web_identifier import WebIdentifier
from .tools.dataset_discovery import DatasetDiscovery
from .tools.model_discovery import ModelDiscovery

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main optimization loop with two modes: strategy and agent."""

    def __init__(self, brain_model_id=None, max_rounds=3,
                 max_no_improve=2, memory_path=None, mode="agent"):
        self.max_rounds = max_rounds
        self.max_no_improve = max_no_improve
        self.mode = mode  # "strategy" or "agent"

        # Initialize components
        Config.ensure_dirs()
        self.memory = Memory(path=memory_path)
        self.brain = SuperBrain(model_id=brain_model_id)
        self.monitor = QualityMonitor()
        self.vlm_pool = VLMPool()
        self.web_identifier = WebIdentifier()
        self.model_discovery = ModelDiscovery()
        self.dataset_discovery = DatasetDiscovery()
        self.tools = ToolRegistry()

        # Register tools
        self.tools.register("generate_labels", self._tool_generate_labels,
                            "Generate consensus pseudo-labels from VLM pool", requires_gpu=True)
        self.tools.register("train_yolo", self._tool_train_yolo,
                            "Train YOLO with pseudo-labels and replay buffer", requires_gpu=True)
        self.tools.register("evaluate", self._tool_evaluate,
                            "Evaluate YOLO on old + new species", requires_gpu=True)

        self.run_log = []
        self._current_label_dir = None  # track labels within agent round
        self._current_model_path = None  # track model within agent round

    # --- Tool wrappers ---

    def _tool_generate_labels(self, strategy, iteration):
        return generate_consensus_labels(strategy, iteration)

    def _tool_train_yolo(self, strategy, label_dir, iteration):
        return train_yolo(strategy, label_dir, iteration)

    def _tool_evaluate(self, model_path):
        return evaluate_full(model_path)

    # =========================================================
    # MAIN ENTRY POINT
    # =========================================================

    def run(self):
        """Execute the optimization loop."""
        self._print_header()

        if not self.memory.baseline:
            self._establish_baseline()

        if len(self.memory.experiments) == 0:
            self._seed_known_best()

        if self.mode == "agent":
            self._run_agent_mode()
        else:
            self._run_strategy_mode()

        self._print_summary()
        self._save_run_log()
        self._write_continuation_flag()

    # =========================================================
    # MODE 2: AGENT MODE — Brain decides each step
    # =========================================================

    def _run_agent_mode(self):
        """Agent mode: Brain decides one action at a time."""
        no_improve_count = 0

        for round_num in range(self.max_rounds):
            logger.info(f"\n{'='*70}")
            logger.info(f"AGENT ROUND {round_num + 1}/{self.max_rounds}")
            logger.info(f"{'='*70}")

            result = self._execute_agent_round(round_num)

            if result:
                assessment = self.monitor.assess_result(
                    result, self.memory.baseline, self.memory.current_best)
                if assessment["is_new_best"]:
                    no_improve_count = 0
                    logger.info("NEW BEST found!")
                else:
                    no_improve_count += 1
                    logger.info(f"No improvement ({no_improve_count}/{self.max_no_improve})")
            else:
                no_improve_count += 1

            if no_improve_count >= self.max_no_improve:
                logger.info(f"Stopping: no improvement for {self.max_no_improve} consecutive rounds")
                break

    def _execute_agent_round(self, round_num):
        """Execute a single agent round: Brain decides actions step by step.

        Returns the final evaluation result dict, or None if no evaluation was done.
        """
        iteration = len(self.memory.experiments)
        max_actions = 10  # prevent infinite loops
        self._current_label_dir = None
        self._current_model_path = None

        # v3.0 gating: compute current labeled-bbox image count.
        # v3.0.11: include yolo_autolabel (OWLv2-pseudo-bbox on classification sets)
        # because mega_trainer trains on them.
        def _current_bbox_count():
            total = 0
            for info in self.dataset_discovery.registry.get("datasets", {}).values():
                if info.get("status") not in ("downloaded", "used_for_training"):
                    continue
                if info.get("annotation") not in (
                    "bbox", "bbox+segmentation", "yolo", "yolo_autolabel"
                ):
                    continue
                total += info.get("local_images", 0)
            return total

        bbox_count = _current_bbox_count()
        MEGA_THRESHOLD = getattr(Config, "MEGA_TRAIN_MIN_IMAGES", 1000)

        # Count how many bbox-datasets total vs. user's internet-collection goal
        bbox_datasets = sum(1 for info in self.dataset_discovery.registry.get("datasets", {}).values()
                            if info.get("annotation") in ("bbox", "bbox+segmentation", "yolo")
                            and info.get("status") in ("downloaded", "used_for_training"))

        # Build initial context for the Brain
        data_status = (
            f"Cumulative bbox-labeled images: {bbox_count} across {bbox_datasets} datasets\n"
            f"Min for train_yolo_mega: {MEGA_THRESHOLD}\n"
            f"Goal: keep harvesting +5 new datasets every round (internet-scale)\n"
            f"Mega gate: {'READY' if bbox_count >= MEGA_THRESHOLD else 'INSUFFICIENT — harvest first'}"
        )
        context_history = [
            {"role": "system", "content": self.memory.get_summary_for_brain()},
            {"role": "system", "content": f"VLM Pool:\n{self.vlm_pool.get_summary_for_brain()}"},
            {"role": "system", "content": f"External:\n{self.web_identifier.get_summary_for_brain()}"},
            {"role": "system", "content": f"Models:\n{self.model_discovery.get_summary_for_brain()}"},
            {"role": "system", "content": f"Data:\n{self.dataset_discovery.get_summary_for_brain()}"},
            {"role": "system", "content": f"DATA GATE: {data_status}"},
            {"role": "system", "content": f"Round {round_num + 1}/{self.max_rounds}. "
                                          f"Iteration {iteration}. Choose your first action."},
        ]

        final_result = None
        actions_taken = []
        action_counts = {}  # track repeated actions

        for step in range(max_actions):
            logger.info(f"\n--- Step {step + 1}/{max_actions} ---")

            # Brain decides next action (step_num for smart fallback)
            action = self.brain.decide_next_action(context_history, step_num=step)
            action_name = action.get("action", "done")

            # Prevent infinite loops: if same action repeated 2+ times, force progression.
            # v3.0-aware: if Brain loops on harvest/search (common when HF pool exhausted), force mega
            # training on the accumulated real-labeled data instead of falling back to v2.x consensus.
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            repeat_limit = 1 if action_name in ("harvest_new_datasets", "search_datasets") else 2
            if action_counts[action_name] > repeat_limit and action_name not in ("done", "evaluate"):
                logger.warning(f"Action '{action_name}' repeated {action_counts[action_name]} times, forcing progression")
                bbox_now = _current_bbox_count()
                if bbox_now >= MEGA_THRESHOLD and not self._current_model_path:
                    # v3.0 path: enough real data, go straight to mega training
                    action = {"action": "train_yolo_mega",
                              "params": {"epochs": 100, "imgsz": 640, "force": True},
                              "reasoning": f"Forced: harvest/search exhausted, {bbox_now} bbox imgs ready for mega"}
                elif not self._current_label_dir and not self._current_model_path:
                    action = {"action": "generate_consensus",
                              "params": {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3},
                              "reasoning": "Forced: insufficient real data and no labels yet, falling back to v2.x consensus"}
                elif not self._current_model_path:
                    action = {"action": "train_yolo",
                              "params": {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3},
                              "reasoning": "Forced: labels ready, moving to training"}
                else:
                    action = {"action": "evaluate", "params": {},
                              "reasoning": "Forced: model ready, moving to evaluation"}
                action_name = action["action"]
            params = action.get("params", {})
            reasoning = action.get("reasoning", "")

            logger.info(f"Action: {action_name}")
            logger.info(f"Reasoning: {reasoning}")
            actions_taken.append(action)

            # Execute the action
            try:
                if action_name == "done":
                    reason = params.get("reason", "Brain decided to stop")
                    context_history.append(
                        {"role": "observation", "content": f"Round ended: {reason}"})
                    break

                elif action_name == "inspect_labels":
                    vlm_key = params.get("vlm_key", "florence2_base")
                    sample = params.get("sample_size", 20)
                    stats = self.vlm_pool.inspect_label_quality(vlm_key, sample)
                    obs = f"Label inspection for {vlm_key}: {json.dumps(stats, indent=2)}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "run_vlm_inference":
                    vlm_key = params.get("vlm_key", "florence2_base")
                    max_imgs = params.get("max_images", 50)
                    holdout_dir = os.path.join(Config.HOLDOUT_DIR, "train", "images")
                    detections = self.vlm_pool.infer_batch(vlm_key, holdout_dir, max_imgs)
                    # Save detections as labels
                    out_dir = os.path.join(Config.FRAMEWORK_DIR, f"live_{vlm_key}_iter{iteration}")
                    count = self.vlm_pool.save_detections_as_labels(detections, out_dir)
                    self.vlm_pool.unload_all()  # free GPU for next step
                    obs = (f"VLM {vlm_key} live inference: {len(detections)} images, "
                           f"{count} detections saved to {out_dir}")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "generate_consensus":
                    vlm_models = params.get("vlm_models", ["florence2_base", "owlv2"])
                    min_votes = params.get("min_votes", 2)
                    consensus_iou = params.get("consensus_iou", 0.3)
                    strategy = {
                        "vlm_models": vlm_models, "min_votes": min_votes,
                        "consensus_iou": consensus_iou, "use_yolo_old": True,
                    }
                    label_dir, stats = generate_consensus_labels(strategy, iteration)
                    self._current_label_dir = label_dir
                    obs = (f"Consensus labels: {stats['images']} images, "
                           f"{stats['consensus_boxes']} consensus, "
                           f"{stats['yolo_old_boxes']} YOLO old, "
                           f"{stats['rejected_boxes']} rejected")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "train_yolo":
                    if not self._current_label_dir:
                        obs = "ERROR: No labels generated yet. Call generate_consensus first."
                        context_history.append({"role": "observation", "content": obs})
                        logger.warning(obs)
                        continue

                    lr = params.get("lr", 0.001)
                    epochs = params.get("epochs", 50)
                    replay = params.get("replay_ratio", 0.3)
                    strategy = {
                        "lr": lr, "epochs": epochs, "replay_ratio": replay,
                        "freeze_layers": 0, "batch_size": -1, "patience": 15,
                    }

                    # Validate
                    valid, violations, adjusted = self.monitor.validate_strategy(
                        strategy, self.memory)
                    if not valid:
                        strategy = adjusted
                        obs = f"Strategy adjusted: {violations}"
                        context_history.append({"role": "observation", "content": obs})

                    model_path = train_yolo(strategy, self._current_label_dir, iteration)
                    self._current_model_path = model_path
                    obs = f"YOLO training complete: {model_path}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "train_yolo_mega":
                    # v3.0.12: if any dataset is in needs_autolabel state, reroute.
                    # v3.0.17: BUT if autolabel_pending already ran this round, let
                    # mega proceed — v3.0.15's per-round cap intentionally defers
                    # remaining datasets to the NEXT round, so seeing needs_autolabel
                    # entries after autolabel ran is expected, not a bug. Without this
                    # skip, guardrail + round cap formed an infinite loop (Job
                    # 40114079: autolabel → mega → reroute → autolabel → walltime).
                    autolabel_already_ran = any(
                        a.get("action") == "autolabel_pending" for a in actions_taken
                    )
                    pending_autolabel = [
                        slug for slug, info in self.dataset_discovery.registry["datasets"].items()
                        if info.get("annotation") == "needs_autolabel"
                        and info.get("status") == "downloaded"
                        and info.get("local_path")
                    ]
                    if pending_autolabel and not params.get("force") and not autolabel_already_ran:
                        obs = (
                            f"GUARDRAIL REROUTE: {len(pending_autolabel)} dataset(s) have "
                            f"annotation=needs_autolabel ({pending_autolabel[:3]}...). "
                            f"Running autolabel_pending automatically before mega to avoid "
                            f"training on stale pool while these sit unused. Call "
                            f"train_yolo_mega again after autolabel completes."
                        )
                        logger.warning(obs)
                        context_history.append({"role": "observation", "content": obs})
                        # Synthesize an autolabel_pending action
                        action = {
                            "action": "autolabel_pending",
                            "params": {"conf_threshold": 0.12},
                            "reasoning": "Guardrail reroute: needs_autolabel data exists",
                        }
                        action_name = "autolabel_pending"
                        params = action["params"]
                        # Fall through to the autolabel_pending handler below
                        # by re-entering the dispatch via a synthesized action
                        from .tools.autolabel import autolabel_dataset
                        registry_cb = {
                            "get": lambda s: self.dataset_discovery.registry["datasets"].get(s),
                            "update": lambda s, u: (
                                self.dataset_discovery.registry["datasets"][s].update(u),
                                self.dataset_discovery._save_registry(),
                            ),
                        }
                        total_owl = total_fb = total_empty = 0
                        summary = [f"Autolabel: {len(pending_autolabel)} pending"]
                        for slug in pending_autolabel:
                            try:
                                r = autolabel_dataset(slug, registry_cb,
                                                      conf_threshold=0.12)
                                if r.get("status") == "ok":
                                    total_owl += r.get("labeled_with_owl", 0)
                                    total_fb += r.get("labeled_with_fallback", 0)
                                    total_empty += r.get("empty", 0)
                                    summary.append(
                                        f"  {slug}: owl={r.get('labeled_with_owl',0)} "
                                        f"fb={r.get('labeled_with_fallback',0)} "
                                        f"empty={r.get('empty',0)}"
                                    )
                                else:
                                    summary.append(f"  {slug}: {r.get('status')}")
                            except Exception as e:
                                summary.append(f"  {slug}: {type(e).__name__}: {str(e)[:100]}")
                        summary.append(
                            f"TOTAL: owl={total_owl} fb={total_fb} empty={total_empty}. "
                            f"Registry flipped to yolo_autolabel. Now call train_yolo_mega."
                        )
                        obs = "\n".join(summary)
                        context_history.append({"role": "observation", "content": obs})
                        logger.info(obs)
                        actions_taken[-1] = action  # record the reroute
                        continue  # let Brain choose again (should pick mega now)

                    # v3.0 cumulative gate: small threshold + grows with every harvest
                    current_bbox = _current_bbox_count()
                    threshold = MEGA_THRESHOLD
                    force = bool(params.get("force"))
                    # v3.0.10: auto-release when harvest couldn't GROW the bbox pool,
                    # not just when downloaded==0. v3.0.9 Kaggle pulled 380K
                    # classification images — downloaded>0 but bbox delta was 0 —
                    # and we blocked anyway, regressing to v2.x pseudo-label path.
                    bbox_deltas = [
                        a.get("_harvest_result", {}).get("bbox_delta")
                        for a in actions_taken
                        if a.get("action") == "harvest_new_datasets"
                    ]
                    harvest_ran = len(bbox_deltas) > 0
                    harvest_no_bbox_growth = harvest_ran and max(bbox_deltas or [0]) == 0
                    if harvest_no_bbox_growth and current_bbox < threshold and not force:
                        logger.info(f"[Gate] auto-releasing (harvest added no bbox): training on "
                                    f"{current_bbox} bbox imgs (<{threshold})")
                        force = True
                    if current_bbox < threshold and not force:
                        obs = (
                            f"BLOCKED: train_yolo_mega needs at least {threshold} bbox images, "
                            f"only {current_bbox} present. Call harvest_new_datasets first to "
                            f"grow the pool. Or pass force=true to train on what's there."
                        )
                        context_history.append({"role": "observation", "content": obs})
                        logger.warning(obs)
                    else:
                        from .tools.mega_trainer import train_yolo_mega
                        mega_strategy = dict(params)
                        try:
                            model_path, summary = train_yolo_mega(mega_strategy, iteration)
                            self._current_model_path = model_path
                            obs = (f"Mega training complete:\n"
                                   f"  base={summary['base_model']}, imgs={summary['merged_images']}, "
                                   f"classes={summary['num_classes']}\n"
                                   f"  datasets={summary['datasets_used']}\n"
                                   f"  best.pt={model_path}")
                        except Exception as e:
                            obs = f"Mega training failed: {e}. Check dataset_discovery.download_dataset results."
                        context_history.append({"role": "observation", "content": obs})
                        logger.info(obs)

                elif action_name == "evaluate":
                    if not self._current_model_path:
                        # Evaluate baseline
                        self._current_model_path = Config.YOLO_8SP_WEIGHTS

                    result = evaluate_full(self._current_model_path)
                    final_result = result
                    assessment = self.monitor.assess_result(
                        result, self.memory.baseline, self.memory.current_best)
                    obs = (f"Evaluation result:\n"
                           f"  Old: F1={result['old_f1']}, mAP50={result['old_map50']}, "
                           f"mAP50-95={result['old_map50_95']}\n"
                           f"  New: F1={result['new_f1']}, mAP50={result['new_map50']}, "
                           f"mAP50-95={result['new_map50_95']}\n"
                           f"  Forgetting: {result['forgetting']}\n"
                           f"  {'NEW BEST!' if assessment['is_new_best'] else 'No improvement'}")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "identify_weed":
                    # Use plant.id or local identification
                    max_imgs = params.get("max_images", 5)
                    holdout_dir = os.path.join(Config.HOLDOUT_DIR, "test", "images")
                    results = self.web_identifier.identify_batch(
                        holdout_dir, max_images=max_imgs, use_api=True)
                    n_weeds = sum(1 for r in results.values() if r.get("is_weed"))
                    species = set(r.get("species", "?") for r in results.values() if r.get("is_weed"))
                    obs = (f"Web identification: {len(results)} images, {n_weeds} contain weeds. "
                           f"Species found: {species}. "
                           f"API usage: {self.web_identifier.get_usage_info()}")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "search_datasets":
                    # Search for weed datasets (deduped + tracks discovered ones)
                    query = params.get("query", "weed detection")
                    known = self.dataset_discovery.list_all()
                    hf_results = self.dataset_discovery.search_huggingface(query)
                    new_hf = [r for r in hf_results if not r.get("already_known")]
                    obs = f"Registry ({len(known)} datasets):\n"
                    for d in known:
                        marker = "TRAINED" if d["used"] else ("DL" if d["status"] == "downloaded" else "—")
                        obs += f"  [{marker}] {d['name']}: {d['images']} imgs, {d['annotation']}\n"
                    obs += f"\nHuggingFace search '{query}': {len(hf_results)} results, {len(new_hf)} NEW\n"
                    for r in new_hf[:8]:
                        obs += f"  • {r.get('hf_id')} (dl={r.get('downloads', 0)})\n"
                    obs += f"\n{self.dataset_discovery.get_summary_for_brain()}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "download_dataset":
                    # Download a weed dataset
                    ds_name = params.get("name", "weedsense")
                    max_imgs = params.get("max_images")
                    try:
                        path, stats = self.dataset_discovery.download_dataset(ds_name, max_imgs)
                        obs = f"Dataset '{ds_name}': {json.dumps(stats)}\nPath: {path}"
                    except Exception as e:
                        obs = f"Download failed for '{ds_name}': {e}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "autolabel_pending":
                    # v3.0.11: run OWLv2 on every dataset whose annotation ==
                    # needs_autolabel, converting classification images into
                    # YOLO bbox labels. Mega trainer then picks them up via
                    # annotation == "yolo_autolabel".
                    # v3.0.15: add per-round image cap so mega+evaluate get
                    # walltime. Without cap, v3.0.14 burned all 8h on autolabel
                    # and never trained. OWLv2-large at ~1.7 img/sec means
                    # 20K imgs ~3h — leaves 4-5h for mega+eval.
                    from .tools.autolabel import autolabel_dataset
                    conf = params.get("conf_threshold", 0.12)
                    max_imgs = params.get("max_images_per_ds", 15000)
                    max_ds = params.get("max_datasets")
                    max_total = params.get("max_total_images", 20000)
                    pending = [
                        slug for slug, v in self.dataset_discovery.registry["datasets"].items()
                        if v.get("annotation") == "needs_autolabel"
                        and v.get("status") == "downloaded"
                        and v.get("local_path")
                    ]
                    if max_ds:
                        pending = pending[:max_ds]
                    if not pending:
                        obs = ("No datasets with annotation=needs_autolabel. "
                               "Either no classification sets were harvested this run, "
                               "or they were already auto-labeled. Proceed to train_yolo_mega.")
                        context_history.append({"role": "observation", "content": obs})
                        logger.info(obs)
                    else:
                        registry_cb = {
                            "get": lambda s: self.dataset_discovery.registry["datasets"].get(s),
                            "update": lambda s, u: (
                                self.dataset_discovery.registry["datasets"][s].update(u),
                                self.dataset_discovery._save_registry(),
                            ),
                        }
                        summary_lines = [f"Autolabel: {len(pending)} dataset(s) pending "
                                         f"(caps: per_ds={max_imgs} total={max_total})"]
                        total_with_owl = 0
                        total_fallback = 0
                        total_empty = 0
                        total_processed_this_round = 0
                        for slug in pending:
                            if total_processed_this_round >= max_total:
                                summary_lines.append(
                                    f"  {slug}: SKIPPED (round cap reached: "
                                    f"{total_processed_this_round}/{max_total})"
                                )
                                continue
                            remaining = max_total - total_processed_this_round
                            per_ds_cap = min(max_imgs, remaining)
                            try:
                                r = autolabel_dataset(
                                    slug, registry_cb,
                                    conf_threshold=conf,
                                    max_images=per_ds_cap,
                                )
                                if r.get("status") == "ok":
                                    ds_total = r.get("labeled_with_owl", 0) + r.get("labeled_with_fallback", 0)
                                    total_with_owl += r.get("labeled_with_owl", 0)
                                    total_fallback += r.get("labeled_with_fallback", 0)
                                    total_empty += r.get("empty", 0)
                                    total_processed_this_round += ds_total
                                    summary_lines.append(
                                        f"  {slug}: owl={r.get('labeled_with_owl',0)} "
                                        f"fb={r.get('labeled_with_fallback',0)} "
                                        f"empty={r.get('empty',0)} "
                                        f"prompt={r.get('prompt','?')!r}"
                                    )
                                else:
                                    summary_lines.append(f"  {slug}: {r.get('status','error')}")
                            except Exception as e:
                                summary_lines.append(f"  {slug}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")
                        summary_lines.append(
                            f"TOTAL: owl={total_with_owl} fallback={total_fallback} "
                            f"empty={total_empty} processed={total_processed_this_round}/{max_total} — "
                            f"registry flipped to yolo_autolabel; call train_yolo_mega next."
                        )
                        obs = "\n".join(summary_lines)
                        context_history.append({"role": "observation", "content": obs})
                        logger.info(obs)

                elif action_name == "harvest_new_datasets":
                    # Find and download up to N NEW bbox datasets this run
                    max_new = params.get("max_new", 5)
                    max_imgs = params.get("max_images_per_ds", 30000)
                    queries = params.get("queries")
                    bbox_before = _current_bbox_count()
                    try:
                        result = self.dataset_discovery.harvest_new_datasets(
                            max_new=max_new,
                            max_images_per_ds=max_imgs,
                            queries=queries,
                        )
                        ok = sum(1 for r in result.get("results", [])
                                 if r["stats"].get("status") == "downloaded")
                        new_imgs = sum(r["stats"].get("images", 0)
                                       for r in result.get("results", []))
                        new_labeled = sum(r["stats"].get("labeled", 0)
                                          for r in result.get("results", []))
                        bbox_after = _current_bbox_count()
                        bbox_delta = bbox_after - bbox_before
                        # v3.0.10: bbox_delta drives gate auto-release (not `ok`).
                        # downloaded>0 doesn't mean bbox-labeled>0.
                        action["_harvest_result"] = {
                            "downloaded": ok, "new_imgs": new_imgs,
                            "bbox_delta": bbox_delta,
                            "bbox_before": bbox_before, "bbox_after": bbox_after,
                        }
                        obs = (f"Harvest: downloaded {ok}/{max_new} new datasets. "
                               f"+{new_imgs} images ({new_labeled} with bboxes).\n")
                        for r in result.get("results", []):
                            s = r["stats"]
                            obs += (f"  {r['hf_id']}: {s.get('status')} "
                                    f"imgs={s.get('images', '?')} "
                                    f"labeled={s.get('labeled', '?')} "
                                    f"kind={s.get('annotation_kind', '?')}\n")
                        if ok == 0:
                            obs += ("HARVEST COMPLETE: 0 new datasets this round. HF pool "
                                    "exhausted for current queries. DO NOT CALL harvest_new_datasets "
                                    "AGAIN THIS ROUND — proceed to train_yolo_mega on accumulated data, "
                                    "or call evaluate/done to end the round.\n")
                        else:
                            obs += ("HARVEST COMPLETE: added datasets to registry. "
                                    "Next action should be train_yolo_mega (not another harvest).\n")
                    except Exception as e:
                        obs = f"Harvest failed: {type(e).__name__}: {e}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "search_models":
                    # Search HuggingFace for new models
                    query = params.get("query", "weed detection")
                    results = self.model_discovery.search_huggingface(query)
                    obs = f"HuggingFace search '{query}': {len(results)} models found.\n"
                    for r in results[:5]:
                        obs += f"  - {r.get('model_id', '?')} (downloads: {r.get('downloads', '?')})\n"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "run_external_model":
                    # Run an external model from HuggingFace
                    model_key = params.get("model_key", "detr_weed")
                    max_imgs = params.get("max_images", 50)
                    holdout_dir = os.path.join(Config.HOLDOUT_DIR, "train", "images")
                    detections = self.model_discovery.infer_batch(model_key, holdout_dir, max_imgs)
                    out_dir = os.path.join(Config.FRAMEWORK_DIR, f"ext_{model_key}_iter{iteration}")
                    count = self.model_discovery.save_detections_as_labels(detections, out_dir)
                    self.model_discovery.unload_all()
                    obs = (f"External model {model_key}: {len(detections)} images, "
                           f"{count} detections saved to {out_dir}")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "analyze_failure":
                    # Brain analyzes why the last experiment failed
                    focus = params.get("focus", "forgetting")
                    # Gather analysis context
                    last_exp = self.memory.experiments[-1] if self.memory.experiments else {}
                    last_result = last_exp.get("result", {})
                    lessons = [l["lesson"] for l in self.memory.get_all_lessons()[-5:]]

                    analysis_prompt = (
                        f"Analyzing failure (focus: {focus}):\n"
                        f"  Last result: old_f1={last_result.get('old_f1', '?')}, "
                        f"new_f1={last_result.get('new_f1', '?')}, "
                        f"forgetting={last_result.get('forgetting', '?')}\n"
                        f"  Baseline: old_f1={self.memory.baseline.get('old_f1', '?')}, "
                        f"new_f1={self.memory.baseline.get('new_f1', '?')}\n"
                        f"  Recent lessons: {lessons}\n"
                        f"  Known: VLM pseudo-labels have 27.4% false positive rate\n"
                        f"  Known: 2-model consensus is best, but noise still too high\n"
                    )

                    # Ask Brain to analyze (if Ollama available)
                    try:
                        import ollama
                        r = ollama.chat(
                            model=self.brain.model_id,
                            messages=[{"role": "user", "content":
                                f"You are analyzing why a weed detection experiment failed.\n\n"
                                f"{analysis_prompt}\n"
                                f"What is the root cause? What specific action should be taken next? "
                                f"Be concrete and actionable. 3-5 sentences."}]
                        )
                        analysis = r.message.content or "No analysis generated"
                    except Exception:
                        analysis = (
                            f"Root cause: label noise (27.4% FP). VLM pseudo-labels contain false "
                            f"positives that teach YOLO wrong patterns, causing forgetting. "
                            f"Recommended: use filter_labels to remove low-confidence detections "
                            f"before training. Two-pass approach: train once, filter with YOLO's "
                            f"own predictions at conf>0.7, retrain on cleaned labels."
                        )

                    obs = f"FAILURE ANALYSIS:\n{analysis}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "freeze_train":
                    # Wang 2025: backbone freeze for catastrophic forgetting prevention
                    if not self._current_label_dir:
                        obs = "ERROR: No labels generated yet."
                        context_history.append({"role": "observation", "content": obs})
                        continue

                    freeze = params.get("freeze_layers", 10)
                    lr = params.get("lr", 0.001)
                    epochs = params.get("epochs", 50)
                    replay = params.get("replay_ratio", 0.3)

                    strategy = {
                        "freeze_layers": freeze, "lr": lr, "epochs": epochs,
                        "replay_ratio": replay, "batch_size": -1, "patience": 15,
                    }
                    valid, violations, adjusted = self.monitor.validate_strategy(strategy, self.memory)
                    if not valid:
                        strategy = adjusted

                    logger.info(f"FREEZE TRAIN: freezing layers 0-{freeze} (Wang 2025)")
                    model_path = train_yolo(strategy, self._current_label_dir, iteration)
                    self._current_model_path = model_path
                    obs = f"Freeze training complete (freeze={freeze}): {model_path}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "two_pass_train":
                    # Two-pass self-training: train → filter with own predictions → retrain
                    if not self._current_label_dir:
                        obs = "ERROR: No labels generated yet."
                        context_history.append({"role": "observation", "content": obs})
                        continue

                    lr = params.get("lr", 0.001)
                    epochs = params.get("epochs", 30)
                    filter_conf = params.get("filter_conf", 0.8)

                    # Pass 1: Train on noisy labels
                    logger.info("[TWO-PASS] Pass 1: Training on noisy labels...")
                    strategy_p1 = {"lr": lr, "epochs": epochs, "replay_ratio": 0.3,
                                   "freeze_layers": 10, "batch_size": -1, "patience": 10}
                    valid, _, adjusted = self.monitor.validate_strategy(strategy_p1, self.memory)
                    if not valid: strategy_p1 = adjusted
                    model_p1 = train_yolo(strategy_p1, self._current_label_dir, iteration)

                    # Filter: use pass-1 model to clean labels
                    logger.info(f"[TWO-PASS] Filtering with conf>{filter_conf}...")
                    from .tools.label_filter import filter_labels_with_yolo
                    filtered_dir, fstats = filter_labels_with_yolo(
                        model_path=model_p1,
                        label_dir=self._current_label_dir,
                        image_dir=os.path.join(Config.HOLDOUT_DIR, "train", "images"),
                        conf_threshold=filter_conf,
                        iteration=iteration + 100,  # avoid dir collision
                    )

                    # Pass 2: Retrain on cleaned labels with LoRA
                    logger.info("[TWO-PASS] Pass 2: Retraining on filtered labels with hybrid LoRA...")
                    from .tools.lora_yolo import train_yolo_with_lora
                    lora_strategy = {"lora_rank": 64, "lora_alpha": 128.0, "lr": 0.0005,
                                     "epochs": epochs, "replay_ratio": 0.3, "batch_size": -1,
                                     "patience": 10, "lora_mode": "hybrid"}
                    try:
                        model_p2 = train_yolo_with_lora(lora_strategy, filtered_dir, iteration + 200)
                    except Exception:
                        # Fallback: standard freeze train on filtered labels
                        strategy_p2 = {"lr": 0.0005, "epochs": epochs, "replay_ratio": 0.3,
                                       "freeze_layers": 10, "batch_size": -1, "patience": 10}
                        model_p2 = train_yolo(strategy_p2, filtered_dir, iteration + 200)

                    self._current_model_path = model_p2
                    self._current_label_dir = filtered_dir
                    obs = (f"Two-pass training complete:\n"
                           f"  Pass 1: trained on noisy labels\n"
                           f"  Filter: {fstats['original']}→{fstats['kept']} kept, "
                           f"{fstats['removed']} removed ({fstats['removal_rate']:.1%})\n"
                           f"  Pass 2: retrained on filtered labels with hybrid LoRA\n"
                           f"  Model: {model_p2}")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "lora_train":
                    # LoRA: inject adapters into head Conv2d, freeze backbone+neck
                    if not self._current_label_dir:
                        obs = "ERROR: No labels generated yet."
                        context_history.append({"role": "observation", "content": obs})
                        continue

                    from .tools.lora_yolo import train_yolo_with_lora

                    lora_strategy = {
                        "lora_rank": params.get("lora_rank", 16),
                        "lora_alpha": params.get("lora_alpha", 32.0),
                        "lr": params.get("lr", 0.0005),
                        "epochs": params.get("epochs", 50),
                        "replay_ratio": 0.3,
                        "batch_size": -1,
                        "patience": 15,
                    }
                    logger.info(f"LORA TRAIN: rank={lora_strategy['lora_rank']}, "
                                f"alpha={lora_strategy['lora_alpha']}, lr={lora_strategy['lr']}")
                    try:
                        model_path = train_yolo_with_lora(lora_strategy, self._current_label_dir, iteration)
                        self._current_model_path = model_path
                        obs = f"LoRA training complete: {model_path}"
                    except Exception as e:
                        obs = f"LoRA training failed: {e}. Tried to inject LoRA adapters into YOLO head."
                        logger.error(obs)
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "distill_train":
                    # Self-distillation training (Teach YOLO to Remember)
                    if not self._current_label_dir:
                        obs = "ERROR: No labels generated yet."
                        context_history.append({"role": "observation", "content": obs})
                        continue

                    # For now, distillation is implemented as: train with frozen teacher predictions
                    # Falls back to standard train with low LR + freeze (approximation)
                    distill_alpha = params.get("distill_alpha", 0.5)
                    lr = params.get("lr", 0.0005)
                    epochs = params.get("epochs", 50)

                    strategy = {
                        "freeze_layers": 5,  # partial freeze + low LR ≈ distillation effect
                        "lr": lr, "epochs": epochs, "replay_ratio": 0.4,
                        "batch_size": -1, "patience": 15,
                    }
                    valid, _, adjusted = self.monitor.validate_strategy(strategy, self.memory)
                    if not valid:
                        strategy = adjusted

                    logger.info(f"DISTILL TRAIN: distill_alpha={distill_alpha}, lr={lr} (Teach YOLO to Remember)")
                    model_path = train_yolo(strategy, self._current_label_dir, iteration)
                    self._current_model_path = model_path
                    obs = f"Distill training complete: {model_path}"
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                elif action_name == "filter_labels":
                    # YOLO self-training filter: use high-confidence predictions to clean labels
                    conf_thresh = params.get("confidence_threshold", 0.7)

                    if not self._current_label_dir:
                        obs = "ERROR: No labels to filter. Generate consensus labels first."
                        context_history.append({"role": "observation", "content": obs})
                        continue

                    # Need a trained YOLO to filter with
                    filter_model = self._current_model_path or Config.YOLO_8SP_WEIGHTS
                    if not os.path.exists(filter_model):
                        obs = "ERROR: No YOLO model available for filtering."
                        context_history.append({"role": "observation", "content": obs})
                        continue

                    from .tools.label_filter import filter_labels_with_yolo
                    filtered_dir, filter_stats = filter_labels_with_yolo(
                        model_path=filter_model,
                        label_dir=self._current_label_dir,
                        image_dir=os.path.join(Config.HOLDOUT_DIR, "train", "images"),
                        conf_threshold=conf_thresh,
                        iteration=iteration,
                    )
                    self._current_label_dir = filtered_dir  # use filtered labels for next train
                    obs = (f"Label filtering (conf>{conf_thresh}): "
                           f"{filter_stats['original']} original → {filter_stats['kept']} kept, "
                           f"{filter_stats['removed']} removed ({filter_stats['removal_rate']:.1%} noise removed)")
                    context_history.append({"role": "observation", "content": obs})
                    logger.info(obs)

                else:
                    obs = f"Unknown action: {action_name}"
                    context_history.append({"role": "observation", "content": obs})

            except Exception as e:
                obs = f"ERROR executing {action_name}: {str(e)}"
                context_history.append({"role": "observation", "content": obs})
                logger.error(obs)
                import traceback; traceback.print_exc()

        # Record experiment
        strategy_summary = {
            "name": f"agent_round_{round_num + 1}",
            "mode": "agent",
            "actions_taken": [a.get("action") for a in actions_taken],
            "num_actions": len(actions_taken),
        }
        if final_result:
            # Brain reflects
            lesson = self.brain.reflect(strategy_summary, final_result, self.memory)
            if lesson:
                self.memory.add_lesson(lesson, f"Agent round {round_num + 1}")
            self.memory.add_experiment(strategy_summary, final_result,
                                       json.dumps([a.get("reasoning", "") for a in actions_taken]))
        else:
            self.memory.add_experiment(strategy_summary,
                                       {"old_f1": 0, "new_f1": 0, "forgetting": True,
                                        "error": "No evaluation performed"},
                                       "Agent round completed without evaluation")

        # Cleanup
        self.vlm_pool.unload_all()

        self.run_log.append({
            "round": round_num + 1,
            "mode": "agent",
            "actions": actions_taken,
            "result": final_result,
        })

        return final_result

    # =========================================================
    # MODE 1: STRATEGY MODE — Original rigid pipeline
    # =========================================================

    def _run_strategy_mode(self):
        """Strategy mode: Brain proposes full strategy → rigid execution."""
        no_improve_count = 0
        for round_num in range(self.max_rounds):
            round_start = time.time()
            logger.info(f"\n{'='*70}")
            logger.info(f"STRATEGY ROUND {round_num + 1}/{self.max_rounds}")
            logger.info(f"{'='*70}")

            success = self._execute_strategy_round(round_num)

            if success:
                latest = self.memory.experiments[-1]
                assessment = self.monitor.assess_result(
                    latest["result"], self.memory.baseline, self.memory.current_best)
                if assessment["is_new_best"]:
                    no_improve_count = 0
                    logger.info("NEW BEST found!")
                else:
                    no_improve_count += 1
                    logger.info(f"No improvement ({no_improve_count}/{self.max_no_improve})")
            else:
                no_improve_count += 1

            elapsed = time.time() - round_start
            logger.info(f"Round {round_num + 1} completed in {elapsed:.0f}s")

            if no_improve_count >= self.max_no_improve:
                logger.info(f"Stopping: no improvement for {self.max_no_improve} rounds")
                break

    def _execute_strategy_round(self, round_num):
        """Execute a single strategy round (original rigid pipeline)."""
        iteration = len(self.memory.experiments)
        try:
            logger.info("[Step 1] Brain analyzing and proposing strategy...")
            strategy, reasoning = self.brain.analyze_and_propose(self.memory)
            strategy["iteration"] = iteration

            logger.info("[Step 2] Validating strategy...")
            is_valid, violations, adjusted = self.monitor.validate_strategy(strategy, self.memory)
            if not is_valid:
                logger.warning(f"Adjusted: {violations}")
                strategy = adjusted

            logger.info(f"Strategy: {strategy.get('name')}")
            logger.info(f"  VLMs: {strategy.get('vlm_models')}, votes>={strategy.get('min_votes')}")

            logger.info("[Step 3] Generating labels...")
            label_dir, label_stats = self.tools.call(
                "generate_labels", strategy=strategy, iteration=iteration)

            logger.info("[Step 4] Training YOLO...")
            model_path = self.tools.call(
                "train_yolo", strategy=strategy, label_dir=label_dir, iteration=iteration)

            logger.info("[Step 5] Evaluating...")
            result = self.tools.call("evaluate", model_path=model_path)

            assessment = self.monitor.assess_result(
                result, self.memory.baseline, self.memory.current_best)
            logger.info(self.monitor.format_assessment(assessment, result))

            logger.info("[Step 6] Brain reflecting...")
            lesson = self.brain.reflect(strategy, result, self.memory)
            if lesson:
                severity = "high" if assessment["forgetting"] else "info"
                self.memory.add_lesson(lesson, f"Round {round_num+1}: {strategy.get('name')}", severity)

            self.memory.add_experiment(strategy, result, reasoning[:300])
            self.run_log.append({"round": round_num + 1, "mode": "strategy",
                                 "strategy": strategy.get("name"), "result": result})
            return True

        except Exception as e:
            logger.error(f"Round {round_num + 1} failed: {e}")
            import traceback; traceback.print_exc()
            self.memory.add_experiment(
                {"name": "error"}, {"old_f1": 0, "new_f1": 0, "forgetting": True, "error": str(e)},
                f"Error: {e}")
            return False

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def _establish_baseline(self):
        logger.info("Establishing baseline...")
        if not os.path.exists(Config.YOLO_8SP_WEIGHTS):
            raise FileNotFoundError(f"Base YOLO weights not found: {Config.YOLO_8SP_WEIGHTS}")

        baseline = evaluate_full(Config.YOLO_8SP_WEIGHTS)
        self.memory.baseline = {
            "old_f1": baseline["old_f1"], "old_map50": baseline["old_map50"],
            "old_map50_95": baseline["old_map50_95"],
            "new_f1": baseline["new_f1"], "new_map50": baseline["new_map50"],
            "new_map50_95": baseline["new_map50_95"],
        }
        self.memory.current_best = dict(self.memory.baseline)
        self.memory.current_best["strategy"] = "baseline_yolo_8sp"
        self.memory.current_best["iteration"] = -1
        self.memory.save()
        logger.info(f"Baseline: old_F1={baseline['old_f1']}, new_F1={baseline['new_f1']}")

    def _seed_known_best(self):
        logger.info("Seeding with Phase 3E best result")
        self.memory.add_experiment(
            {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
             "consensus_iou": 0.3, "use_yolo_old": True, "lr": 0.001, "epochs": 50,
             "freeze_layers": 0, "replay_ratio": 0.3, "name": "seed_best_known_phase3e"},
            {"old_f1": 0.897, "new_f1": 0.622, "old_map50": 0.851, "new_map50": 0.559,
             "old_map50_95": 0.810, "new_map50_95": 0.493, "forgetting": False},
            "Seed: Phase 3E Florence+OWLv2 consensus")

    # =========================================================
    # OUTPUT
    # =========================================================

    def _print_header(self):
        header = f"""
{'='*70}
  WEED OPTIMIZER FRAMEWORK v1.0.0
  Mode: {self.mode.upper()}
  Brain: {self.brain.model_id}
  Max rounds: {self.max_rounds}
  VLMs: {self.vlm_pool.get_available_vlms()}
{'='*70}"""
        logger.info(header)
        print(header)

    def _print_summary(self):
        print(f"\n{'='*70}")
        print(f"FRAMEWORK RUN COMPLETE (mode={self.mode})")
        print(f"{'='*70}")

        b = self.memory.baseline
        print(f"\nBaseline: old_F1={b.get('old_f1')}, new_F1={b.get('new_f1')}")
        cb = self.memory.current_best
        print(f"Best:     old_F1={cb.get('old_f1')}, new_F1={cb.get('new_f1')}")

        print(f"\nAll experiments ({len(self.memory.experiments)}):")
        for e in self.memory.experiments:
            r = e["result"]
            name = e["strategy"].get("name", "?")
            status = "FORGET" if r.get("forgetting") else "ok"
            mode = e["strategy"].get("mode", "strategy")
            print(f"  {e['iteration']}: [{mode}] {name:<35s} "
                  f"old_f1={r.get('old_f1', '?'):<6} new_f1={r.get('new_f1', '?'):<6} [{status}]")

        print(f"\nLessons learned ({len(self.memory.lessons)}):")
        for l in self.memory.lessons:
            print(f"  - {l['lesson']}")

        print(f"\nTool usage:")
        for name, stats in self.tools.get_stats().items():
            print(f"  {name}: {stats['calls']} calls, {stats['total_time']}s total")
        print(f"\nMemory saved to: {self.memory.path}")

    def _save_run_log(self):
        log_path = os.path.join(Config.FRAMEWORK_DIR, "run_log.json")
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "brain_model": self.brain.model_id,
            "max_rounds": self.max_rounds,
            "baseline": self.memory.baseline,
            "current_best": self.memory.current_best,
            "rounds": self.run_log,
            "tool_stats": self.tools.get_stats(),
            "total_experiments": len(self.memory.experiments),
            "total_lessons": len(self.memory.lessons),
        }
        tmp = log_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        os.replace(tmp, log_path)
        logger.info(f"Run log saved to {log_path}")

    def _write_continuation_flag(self):
        """Write should_continue.txt if more optimization rounds could help.

        This enables SLURM job chaining: the current job's script checks
        for this file and auto-submits the next job.
        """
        flag_path = os.path.join(Config.FRAMEWORK_DIR, "should_continue.txt")

        # Continue if: we found improvements and haven't hit diminishing returns
        should_continue = False
        if len(self.memory.experiments) >= 2:
            latest = self.memory.experiments[-1]["result"]
            if latest.get("new_f1", 0) > self.memory.baseline.get("new_f1", 0):
                should_continue = True  # still improving
            if not latest.get("forgetting", True):
                should_continue = True  # no forgetting, worth trying more

        if should_continue:
            with open(flag_path, "w") as f:
                f.write(f"continue\ntotal_experiments={len(self.memory.experiments)}\n")
            logger.info(f"Continuation flag written: {flag_path}")
        else:
            # Remove flag if exists
            if os.path.exists(flag_path):
                os.remove(flag_path)
            logger.info("No continuation needed")
