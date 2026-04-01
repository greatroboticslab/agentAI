"""
Quality Monitor — Detects forgetting, validates strategies, tracks per-class drift.

Acts as the safety net: no strategy executes without passing validation,
and no result is accepted without quality assessment.
"""

import logging
from .config import Config

logger = logging.getLogger(__name__)


class QualityMonitor:
    """Monitors detection quality and catches problems before they cause damage."""

    def __init__(self):
        self.forgetting_threshold = Config.FORGETTING_THRESHOLD

    # --- Strategy validation (PRE-execution) ---

    def validate_strategy(self, strategy, memory):
        """Check if a proposed strategy violates any known lessons.

        Returns (is_valid, violations, adjusted_strategy).
        If invalid, returns an adjusted strategy that respects constraints.
        """
        violations = []
        adjusted = dict(strategy)
        constraints = memory.get_critical_constraints()

        # HL01: No excessive freezing
        freeze = adjusted.get("freeze_layers", 0)
        max_freeze = constraints.get("HL01", {}).get("max", 3)
        if freeze > max_freeze:
            violations.append(f"HL01: freeze_layers={freeze} > max={max_freeze}")
            adjusted["freeze_layers"] = max_freeze

        # HL02: No excessive replay
        replay = adjusted.get("replay_ratio", 0.3)
        max_replay = constraints.get("HL02", {}).get("max", 0.5)
        if replay > max_replay:
            violations.append(f"HL02: replay_ratio={replay} > max={max_replay}")
            adjusted["replay_ratio"] = max_replay

        # HL05: Not too many VLMs
        vlms = adjusted.get("vlm_models", [])
        max_vlms = constraints.get("HL05", {}).get("max_count", 4)
        if len(vlms) > max_vlms:
            violations.append(f"HL05: {len(vlms)} VLMs > max={max_vlms}")
            # Keep only the highest-precision ones
            scored = [(v, Config.get_vlm_precision(v)) for v in vlms]
            scored.sort(key=lambda x: x[1], reverse=True)
            adjusted["vlm_models"] = [v for v, _ in scored[:max_vlms]]

        # HL03: No SAM caption
        if adjusted.get("use_sam_caption", False):
            violations.append("HL03: SAM caption classification is too noisy")
            adjusted["use_sam_caption"] = False

        # HL04: No VLM fine-tuning
        if adjusted.get("finetune_vlm", False):
            violations.append("HL04: Fine-tuning VLMs degrades zero-shot ability")
            adjusted["finetune_vlm"] = False

        # Check for obviously bad lr
        lr = adjusted.get("lr", 0.001)
        if lr > 0.01:
            violations.append(f"lr={lr} too high, capping at 0.01")
            adjusted["lr"] = 0.01
        if lr < 1e-6:
            violations.append(f"lr={lr} too low, setting to 1e-5")
            adjusted["lr"] = 1e-5

        # Epochs sanity
        epochs = adjusted.get("epochs", 50)
        if epochs > 200:
            violations.append(f"epochs={epochs} too high, capping at 150")
            adjusted["epochs"] = 150
        if epochs < 10:
            violations.append(f"epochs={epochs} too low, setting to 20")
            adjusted["epochs"] = 20

        is_valid = len(violations) == 0
        if not is_valid:
            logger.warning(f"Strategy validation found {len(violations)} issues:")
            for v in violations:
                logger.warning(f"  - {v}")

        return is_valid, violations, adjusted

    # --- Result assessment (POST-execution) ---

    def check_forgetting(self, old_f1):
        """Check if old species performance dropped below threshold."""
        return old_f1 < self.forgetting_threshold

    def assess_result(self, result, baseline, current_best):
        """Comprehensive assessment of an experiment result.

        Returns a dict describing whether the result is an improvement,
        and in which dimensions.
        """
        assessment = {
            # Core checks
            "forgetting": self.check_forgetting(result.get("old_f1", 0)),
            "improved_new_f1": result.get("new_f1", 0) > current_best.get("new_f1", 0),
            "improved_old_f1": result.get("old_f1", 0) >= current_best.get("old_f1", 0),

            # mAP checks
            "improved_new_map50": result.get("new_map50", 0) > current_best.get("new_map50", 0),
            "improved_new_map50_95": result.get("new_map50_95", 0) > current_best.get("new_map50_95", 0),

            # Vs baseline
            "better_than_baseline_new": result.get("new_f1", 0) > baseline.get("new_f1", 0),
            "better_than_baseline_old": result.get("old_f1", 0) >= baseline.get("old_f1", 0) - 0.03,

            # Deltas
            "delta_new_f1": round(result.get("new_f1", 0) - baseline.get("new_f1", 0), 4),
            "delta_old_f1": round(result.get("old_f1", 0) - baseline.get("old_f1", 0), 4),
            "delta_new_map50": round(result.get("new_map50", 0) - baseline.get("new_map50", 0), 4),
        }

        # Overall: improved new AND no forgetting
        assessment["is_improvement"] = (
            assessment["better_than_baseline_new"]
            and not assessment["forgetting"]
        )

        # New best: strictly better than current best AND no forgetting
        assessment["is_new_best"] = (
            assessment["improved_new_f1"]
            and not assessment["forgetting"]
        )

        return assessment

    def format_assessment(self, assessment, result):
        """Format assessment as human-readable string."""
        lines = []
        lines.append(f"  Old F1={result.get('old_f1', '?')}, "
                     f"New F1={result.get('new_f1', '?')}")
        lines.append(f"  Old mAP50={result.get('old_map50', '?')}, "
                     f"New mAP50={result.get('new_map50', '?')}")
        lines.append(f"  Old mAP50-95={result.get('old_map50_95', '?')}, "
                     f"New mAP50-95={result.get('new_map50_95', '?')}")

        if assessment["forgetting"]:
            lines.append(f"  !! FORGETTING DETECTED (old F1 < {self.forgetting_threshold})")
        if assessment["is_new_best"]:
            lines.append(f"  ** NEW BEST RESULT **")
        elif assessment["is_improvement"]:
            lines.append(f"  + Improvement over baseline (delta new F1: {assessment['delta_new_f1']:+.4f})")
        else:
            lines.append(f"  - No improvement (delta new F1: {assessment['delta_new_f1']:+.4f})")

        return "\n".join(lines)

    # --- Per-class analysis ---

    def analyze_per_class(self, per_class_results, species_type="old"):
        """Analyze per-class results to find weak spots.

        Returns list of (class_name, f1, status) sorted by F1.
        """
        analysis = []
        for cls_name, metrics in per_class_results.items():
            f1 = metrics.get("f1", 0)
            if f1 < 0.5:
                status = "critical"
            elif f1 < 0.7:
                status = "weak"
            elif f1 < 0.85:
                status = "moderate"
            else:
                status = "good"
            analysis.append((cls_name, f1, status))

        analysis.sort(key=lambda x: x[1])
        return analysis

    def detect_distribution_shift(self, current_per_class, baseline_per_class):
        """Detect which classes have shifted significantly from baseline.

        Returns list of (class_name, baseline_f1, current_f1, delta).
        """
        shifts = []
        for cls_name in current_per_class:
            if cls_name not in baseline_per_class:
                continue
            curr_f1 = current_per_class[cls_name].get("f1", 0)
            base_f1 = baseline_per_class[cls_name].get("f1", 0)
            delta = curr_f1 - base_f1
            if abs(delta) > 0.05:  # 5% shift is significant
                shifts.append((cls_name, base_f1, curr_f1, delta))

        shifts.sort(key=lambda x: x[3])  # worst first
        return shifts
