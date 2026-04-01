"""
Memory — Persistent experiment history, lessons learned, and baselines.

Stores knowledge that persists across sessions and prevents repeating known mistakes.

Storage: JSON file with atomic writes (.tmp → os.replace) to prevent corruption.
"""

import json
import os
import logging
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)


class Memory:
    """Persistent memory for the optimizer framework.

    Three types of memory:
    1. Hard lessons — NEVER violated, from 18 sessions of experiments
    2. Learned lessons — discovered by Brain during optimization
    3. Experiment history — full record of strategies and results
    """

    def __init__(self, path=None):
        from .config import Config
        self.path = path or os.path.join(Config.FRAMEWORK_DIR, "memory.json")
        self.experiments = []
        self.lessons = []
        self.baseline = {}
        self.current_best = {}
        self.meta = {
            "created": datetime.now().isoformat(),
            "total_iterations": 0,
            "framework_version": "1.0.0",
        }

        self._init_hard_lessons()

        if os.path.exists(self.path):
            self.load()

    def _init_hard_lessons(self):
        """10 lessons from 18 sessions. These are NEVER overridden."""
        self.hard_lessons = [
            {
                "id": "HL01",
                "lesson": "NEVER freeze backbone layers — causes catastrophic failure (F1=0.155)",
                "evidence": "Phase 3C: frozen backbone test",
                "severity": "critical",
                "constraint": {"field": "freeze_layers", "max": 3},
            },
            {
                "id": "HL02",
                "lesson": "Replay buffer >50% makes forgetting WORSE (-3.0% vs -2.4%)",
                "evidence": "Phase 3C: replay50 test",
                "severity": "high",
                "constraint": {"field": "replay_ratio", "max": 0.5},
            },
            {
                "id": "HL03",
                "lesson": "SAM + caption keyword classification is too noisy (-11% new species)",
                "evidence": "Phase 3D: SAM-enhanced test",
                "severity": "high",
                "constraint": {"field": "use_sam_caption", "forbidden": True},
            },
            {
                "id": "HL04",
                "lesson": "Fine-tuning VLMs (Florence-2) degrades their zero-shot ability (-11.3% mAP)",
                "evidence": "Phase 3F: Florence-2 fine-tune test",
                "severity": "high",
                "constraint": {"field": "finetune_vlm", "forbidden": True},
            },
            {
                "id": "HL05",
                "lesson": "2 complementary models (high-prec + high-recall) beat 7 models voting",
                "evidence": "Phase 3E: agent optimizer 5-strategy test",
                "severity": "high",
                "constraint": {"field": "vlm_models", "max_count": 4},
            },
            {
                "id": "HL06",
                "lesson": "Florence-2-base (prec=0.789) + OWLv2 (rec=0.918) is the best VLM pair",
                "evidence": "Phase 3E: consensus_flo+owl achieved +0.016 F1",
                "severity": "info",
            },
            {
                "id": "HL07",
                "lesson": "Florence-2 <OD> mode does NOT output calibrated confidence — threshold filtering doesn't work",
                "evidence": "Phase 3E: conf 0.3/0.5/0.7 all gave identical results",
                "severity": "high",
            },
            {
                "id": "HL08",
                "lesson": "Old species F1 must stay above 0.90 — reject any training causing more forgetting",
                "evidence": "All phases: forgetting is the primary constraint",
                "severity": "critical",
                "constraint": {"field": "old_f1_threshold", "min": 0.90},
            },
            {
                "id": "HL09",
                "lesson": "Label noise (27.4% FP from Florence-2) is the ROOT CAUSE of all failures",
                "evidence": "All pseudo-label approaches degrade YOLO because of noise accumulation",
                "severity": "critical",
            },
            {
                "id": "HL10",
                "lesson": "YOLO trained on 8/12 species drops 27% F1 on unseen 4 species",
                "evidence": "Phase 3B: Leave-4-Out experiment",
                "severity": "info",
            },
        ]

    # --- Experiment recording ---

    def add_experiment(self, strategy, result, brain_reasoning=""):
        """Record an experiment with its strategy, result, and reasoning."""
        entry = {
            "iteration": len(self.experiments),
            "timestamp": datetime.now().isoformat(),
            "strategy": deepcopy(strategy),
            "result": deepcopy(result),
            "brain_reasoning": str(brain_reasoning)[:500],
        }
        self.experiments.append(entry)
        self.meta["total_iterations"] = len(self.experiments)

        # Update current best if improved AND no forgetting
        if (not result.get("forgetting", True)
                and result.get("new_f1", 0) > self.current_best.get("new_f1", 0)):
            self.current_best = {
                "iteration": entry["iteration"],
                "strategy": deepcopy(strategy),
                "old_f1": result.get("old_f1", 0),
                "new_f1": result.get("new_f1", 0),
                "old_map50": result.get("old_map50", 0),
                "new_map50": result.get("new_map50", 0),
                "old_map50_95": result.get("old_map50_95", 0),
                "new_map50_95": result.get("new_map50_95", 0),
            }
            logger.info(f"New best! iter={entry['iteration']}, new_f1={result['new_f1']}")

        self.save()
        return entry

    # --- Lesson management ---

    def add_lesson(self, lesson, evidence, severity="info"):
        """Add a lesson learned by the Brain."""
        # Avoid duplicates
        for existing in self.lessons:
            if existing["lesson"] == lesson:
                return existing
        entry = {
            "id": f"LL{len(self.lessons):02d}",
            "lesson": lesson,
            "evidence": evidence,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }
        self.lessons.append(entry)
        self.save()
        logger.info(f"Lesson learned: {lesson}")
        return entry

    def get_all_lessons(self):
        """Get all lessons (hard-coded + learned)."""
        return self.hard_lessons + self.lessons

    def get_critical_constraints(self):
        """Extract machine-readable constraints from hard lessons."""
        constraints = {}
        for lesson in self.hard_lessons:
            c = lesson.get("constraint")
            if c:
                constraints[lesson["id"]] = c
        return constraints

    # --- Context generation for Brain ---

    def get_summary_for_brain(self):
        """Generate a text summary for the Brain to read and reason about.

        This is the 'context window' that the Brain sees before proposing a strategy.
        Structured to give the Brain maximum relevant information concisely.
        """
        lines = []

        # Baseline and current best
        lines.append("=== PERFORMANCE BASELINE ===")
        lines.append(f"YOLO (8-species, no augmentation):")
        lines.append(f"  Old species: F1={self.baseline.get('old_f1', '?')}, "
                     f"mAP50={self.baseline.get('old_map50', '?')}, "
                     f"mAP50-95={self.baseline.get('old_map50_95', '?')}")
        lines.append(f"  New species: F1={self.baseline.get('new_f1', '?')}, "
                     f"mAP50={self.baseline.get('new_map50', '?')}, "
                     f"mAP50-95={self.baseline.get('new_map50_95', '?')}")

        if self.current_best:
            lines.append(f"\nCurrent best strategy (iter {self.current_best.get('iteration', '?')}):")
            lines.append(f"  Old: F1={self.current_best.get('old_f1', '?')}, "
                         f"mAP50={self.current_best.get('old_map50', '?')}")
            lines.append(f"  New: F1={self.current_best.get('new_f1', '?')}, "
                         f"mAP50={self.current_best.get('new_map50', '?')}")

        # Hard lessons
        lines.append("\n=== HARD LESSONS (NEVER violate these) ===")
        for l in self.hard_lessons:
            if l["severity"] in ("critical", "high"):
                lines.append(f"  [{l['severity'].upper()}] {l['lesson']}")

        # Learned lessons
        if self.lessons:
            lines.append(f"\n=== LEARNED LESSONS ({len(self.lessons)} total, showing last 5) ===")
            for l in self.lessons[-5:]:
                lines.append(f"  [{l['id']}] {l['lesson']}")

        # Recent experiments
        if self.experiments:
            lines.append(f"\n=== EXPERIMENT HISTORY ({len(self.experiments)} total, showing last 8) ===")
            for e in self.experiments[-8:]:
                r = e["result"]
                name = e["strategy"].get("name", "?")
                status = "FORGETTING" if r.get("forgetting") else "OK"
                lines.append(
                    f"  iter{e['iteration']}: {name:<40s} "
                    f"old_f1={r.get('old_f1', '?'):<6} new_f1={r.get('new_f1', '?'):<6} "
                    f"[{status}]"
                )

        lines.append(f"\n=== AVAILABLE VLM TOOLS ===")
        lines.append("florence2_base (prec=0.789), owlv2 (rec=0.943), florence2_large (prec=0.692)")
        lines.append("internvl2_8b (prec=0.545), minicpm_v45 (prec=0.407), qwen25_vl_3b, qwen25_vl_7b")

        return "\n".join(lines)

    # --- Persistence ---

    def save(self):
        """Save memory to disk with atomic write."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "meta": self.meta,
            "baseline": self.baseline,
            "current_best": self.current_best,
            "hard_lessons": self.hard_lessons,
            "learned_lessons": self.lessons,
            "experiments": self.experiments,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def load(self):
        """Load memory from disk."""
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.meta = data.get("meta", self.meta)
            self.baseline = data.get("baseline", {})
            self.current_best = data.get("current_best", {})
            self.lessons = data.get("learned_lessons", [])
            self.experiments = data.get("experiments", [])
            logger.info(f"Memory loaded: {len(self.experiments)} experiments, "
                        f"{len(self.lessons)} lessons")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Memory load failed ({e}), starting fresh")
