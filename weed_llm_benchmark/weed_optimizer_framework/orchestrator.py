"""
Orchestrator — The main while loop (Brain → Tools → Evaluate → Brain).

A simple loop where the Brain decides what to do, calls tools,
sees results, and iterates.

The loop runs for max_rounds or until no improvement is found for
consecutive_no_improve rounds.
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
    """Main optimization loop.

    Pattern:
        while not converged:
            strategy = brain.analyze_and_propose(memory)
            strategy = monitor.validate(strategy)
            labels = tools.call("generate_labels", strategy)
            model = tools.call("train_yolo", labels)
            result = tools.call("evaluate", model)
            memory.record(strategy, result)
            brain.reflect(result)
    """

    def __init__(self, brain_model_id=None, max_rounds=3,
                 max_no_improve=2, memory_path=None):
        self.max_rounds = max_rounds
        self.max_no_improve = max_no_improve

        # Initialize components
        Config.ensure_dirs()
        self.memory = Memory(path=memory_path)
        self.brain = SuperBrain(model_id=brain_model_id)
        self.monitor = QualityMonitor()
        self.vlm_pool = VLMPool()
        self.tools = ToolRegistry()

        # Register tools
        self.tools.register(
            "generate_labels", self._tool_generate_labels,
            "Generate consensus pseudo-labels from VLM pool",
            requires_gpu=True
        )
        self.tools.register(
            "train_yolo", self._tool_train_yolo,
            "Train YOLO with pseudo-labels and replay buffer",
            requires_gpu=True
        )
        self.tools.register(
            "evaluate", self._tool_evaluate,
            "Evaluate YOLO on old + new species (mAP@0.5, mAP@0.5:0.95, F1)",
            requires_gpu=True
        )

        self.run_log = []  # log of this run

    # --- Tool wrappers ---

    def _tool_generate_labels(self, strategy, iteration):
        return generate_consensus_labels(strategy, iteration)

    def _tool_train_yolo(self, strategy, label_dir, iteration):
        return train_yolo(strategy, label_dir, iteration)

    def _tool_evaluate(self, model_path):
        return evaluate_full(model_path)

    # --- Main loop ---

    def run(self):
        """Execute the main optimization loop."""
        self._print_header()

        # Step 0: Establish baseline if not set
        if not self.memory.baseline:
            self._establish_baseline()

        # Step 0.5: Seed with known best if first run
        if len(self.memory.experiments) == 0:
            self._seed_known_best()

        # Main loop
        no_improve_count = 0
        for round_num in range(self.max_rounds):
            round_start = time.time()
            logger.info(f"\n{'='*70}")
            logger.info(f"ROUND {round_num + 1}/{self.max_rounds}")
            logger.info(f"{'='*70}")

            success = self._execute_round(round_num)

            if success:
                # Check if this round improved
                latest = self.memory.experiments[-1]
                assessment = self.monitor.assess_result(
                    latest["result"], self.memory.baseline, self.memory.current_best
                )
                if assessment["is_new_best"]:
                    no_improve_count = 0
                    logger.info(f"NEW BEST found!")
                else:
                    no_improve_count += 1
                    logger.info(f"No improvement ({no_improve_count}/{self.max_no_improve})")
            else:
                no_improve_count += 1

            elapsed = time.time() - round_start
            logger.info(f"Round {round_num + 1} completed in {elapsed:.0f}s")

            # Early stop if no improvement for too long
            if no_improve_count >= self.max_no_improve:
                logger.info(f"Stopping: no improvement for {self.max_no_improve} consecutive rounds")
                break

        self._print_summary()
        self._save_run_log()

    def _execute_round(self, round_num):
        """Execute a single optimization round.

        Returns True if the round completed successfully, False if error.
        """
        iteration = len(self.memory.experiments)

        try:
            # 1. Brain analyzes and proposes strategy
            logger.info("[Step 1] Brain analyzing history and proposing strategy...")
            strategy, reasoning = self.brain.analyze_and_propose(self.memory)
            strategy["iteration"] = iteration

            # 2. Monitor validates strategy
            logger.info("[Step 2] Validating strategy against lessons...")
            is_valid, violations, adjusted = self.monitor.validate_strategy(
                strategy, self.memory
            )
            if not is_valid:
                logger.warning(f"Strategy adjusted: {violations}")
                strategy = adjusted

            logger.info(f"Strategy: {strategy.get('name')}")
            logger.info(f"  VLMs: {strategy.get('vlm_models')}, votes>={strategy.get('min_votes')}")
            logger.info(f"  lr={strategy.get('lr')}, epochs={strategy.get('epochs')}, "
                        f"replay={strategy.get('replay_ratio')}")

            # 3. Generate labels
            logger.info("[Step 3] Generating consensus pseudo-labels...")
            label_dir, label_stats = self.tools.call(
                "generate_labels", strategy=strategy, iteration=iteration
            )

            # 4. Train YOLO
            logger.info("[Step 4] Training YOLO...")
            model_path = self.tools.call(
                "train_yolo", strategy=strategy, label_dir=label_dir, iteration=iteration
            )

            # 5. Evaluate
            logger.info("[Step 5] Evaluating on old + new species...")
            result = self.tools.call("evaluate", model_path=model_path)

            # 6. Assess result
            assessment = self.monitor.assess_result(
                result, self.memory.baseline, self.memory.current_best
            )
            logger.info(self.monitor.format_assessment(assessment, result))

            # 7. Brain reflects
            logger.info("[Step 6] Brain reflecting on result...")
            lesson = self.brain.reflect(strategy, result, self.memory)
            if lesson:
                severity = "high" if assessment["forgetting"] else "info"
                self.memory.add_lesson(lesson, f"Round {round_num+1}: {strategy.get('name')}", severity)

            # 8. Record in memory
            self.memory.add_experiment(strategy, result, reasoning[:300])

            # Log
            self.run_log.append({
                "round": round_num + 1,
                "strategy": strategy.get("name"),
                "result": result,
                "assessment": assessment,
                "lesson": lesson,
            })

            return True

        except Exception as e:
            logger.error(f"Round {round_num + 1} failed: {e}")
            import traceback
            traceback.print_exc()

            # Record failure
            self.memory.add_experiment(
                strategy if 'strategy' in dir() else {"name": "error"},
                {"old_f1": 0, "new_f1": 0, "forgetting": True, "error": str(e)},
                f"Error: {e}"
            )
            return False

    # --- Initialization ---

    def _establish_baseline(self):
        """Evaluate the base YOLO model to establish baseline metrics."""
        logger.info("Establishing baseline...")
        if not os.path.exists(Config.YOLO_8SP_WEIGHTS):
            raise FileNotFoundError(
                f"Base YOLO weights not found: {Config.YOLO_8SP_WEIGHTS}\n"
                "Run the leave-4-out experiment first to generate these weights."
            )

        baseline = evaluate_full(Config.YOLO_8SP_WEIGHTS)
        self.memory.baseline = {
            "old_f1": baseline["old_f1"],
            "old_map50": baseline["old_map50"],
            "old_map50_95": baseline["old_map50_95"],
            "new_f1": baseline["new_f1"],
            "new_map50": baseline["new_map50"],
            "new_map50_95": baseline["new_map50_95"],
        }
        self.memory.current_best = dict(self.memory.baseline)
        self.memory.current_best["strategy"] = "baseline_yolo_8sp"
        self.memory.current_best["iteration"] = -1
        self.memory.save()

        logger.info(f"Baseline established:")
        logger.info(f"  Old: F1={baseline['old_f1']}, mAP50={baseline['old_map50']}, "
                    f"mAP50-95={baseline['old_map50_95']}")
        logger.info(f"  New: F1={baseline['new_f1']}, mAP50={baseline['new_map50']}, "
                    f"mAP50-95={baseline['new_map50_95']}")

    def _seed_known_best(self):
        """Seed memory with known best result from Phase 3E."""
        logger.info("Seeding with known best result (Phase 3E: Florence+OWLv2 consensus)")
        self.memory.add_experiment(
            {
                "vlm_models": ["florence2_base", "owlv2"],
                "min_votes": 2,
                "consensus_iou": 0.3,
                "use_yolo_old": True,
                "lr": 0.001,
                "epochs": 50,
                "freeze_layers": 0,
                "replay_ratio": 0.3,
                "name": "seed_best_known_phase3e",
            },
            {
                "old_f1": 0.897,
                "new_f1": 0.622,
                "old_map50": 0.851,
                "new_map50": 0.559,
                "old_map50_95": 0.810,
                "new_map50_95": 0.493,
                "forgetting": False,
            },
            "Seed: best result from Phase 3E agent optimizer (Florence+OWLv2 2-vote consensus)"
        )

    # --- Output ---

    def _print_header(self):
        """Print framework header."""
        header = f"""
{'='*70}
  WEED OPTIMIZER FRAMEWORK v{Config.__dict__.get('VERSION', '1.0.0')}
  Brain: {self.brain.model_id}
  Max rounds: {self.max_rounds}
  VLMs available: {self.vlm_pool.get_available_vlms()}
{'='*70}"""
        logger.info(header)
        print(header)

    def _print_summary(self):
        """Print final summary of all experiments."""
        print(f"\n{'='*70}")
        print("FRAMEWORK RUN COMPLETE")
        print(f"{'='*70}")

        b = self.memory.baseline
        print(f"\nBaseline: old_F1={b.get('old_f1')}, new_F1={b.get('new_f1')}, "
              f"old_mAP50={b.get('old_map50')}, new_mAP50={b.get('new_map50')}")

        cb = self.memory.current_best
        print(f"Best:     old_F1={cb.get('old_f1')}, new_F1={cb.get('new_f1')}, "
              f"old_mAP50={cb.get('old_map50')}, new_mAP50={cb.get('new_map50')}")

        print(f"\nAll experiments ({len(self.memory.experiments)}):")
        for e in self.memory.experiments:
            r = e["result"]
            name = e["strategy"].get("name", "?")
            status = "FORGET" if r.get("forgetting") else "ok"
            print(f"  {e['iteration']}: {name:<40s} "
                  f"old_f1={r.get('old_f1', '?'):<6} new_f1={r.get('new_f1', '?'):<6} "
                  f"old_map50={r.get('old_map50', '?'):<6} new_map50={r.get('new_map50', '?'):<6} "
                  f"[{status}]")

        print(f"\nLessons learned ({len(self.memory.lessons)}):")
        for l in self.memory.lessons:
            print(f"  - [{l.get('severity', 'info')}] {l['lesson']}")

        print(f"\nTool usage:")
        for name, stats in self.tools.get_stats().items():
            print(f"  {name}: {stats['calls']} calls, {stats['total_time']}s total")

        print(f"\nMemory saved to: {self.memory.path}")

    def _save_run_log(self):
        """Save detailed run log to JSON."""
        log_path = os.path.join(Config.FRAMEWORK_DIR, "run_log.json")
        log_data = {
            "timestamp": datetime.now().isoformat(),
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
