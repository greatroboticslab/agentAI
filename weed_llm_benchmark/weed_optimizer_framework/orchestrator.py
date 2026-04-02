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

        # Build initial context for the Brain
        context_history = [
            {"role": "system", "content": self.memory.get_summary_for_brain()},
            {"role": "system", "content": f"VLM Pool:\n{self.vlm_pool.get_summary_for_brain()}"},
            {"role": "system", "content": f"Round {round_num + 1}/{self.max_rounds}. "
                                          f"Iteration {iteration}. Choose your first action."},
        ]

        final_result = None
        actions_taken = []

        for step in range(max_actions):
            logger.info(f"\n--- Step {step + 1}/{max_actions} ---")

            # Brain decides next action (step_num for smart fallback)
            action = self.brain.decide_next_action(context_history, step_num=step)
            action_name = action.get("action", "done")
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

                else:
                    obs = f"Unknown action: {action_name}. Available: {list(action.keys())}"
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
