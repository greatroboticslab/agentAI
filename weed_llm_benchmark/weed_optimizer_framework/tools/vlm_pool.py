"""
VLM Pool — Read-only pool of Vision Language Models for weed detection.

These models are NEVER fine-tuned. They serve as "eyes" for the SuperBrain,
providing pseudo-labels that YOLO trains on.

Current approach: VLM labels are pre-generated (from Phase 2 benchmark runs)
and stored as YOLO-format .txt files. This module provides access to those
pre-generated labels and metadata about each VLM's strengths.

Future: could add live inference mode (load VLM → run on new images in real-time).
"""

import os
import logging
from ..config import Config

logger = logging.getLogger(__name__)


class VLMPool:
    """Pool of read-only VLM assistants.

    Provides:
    - VLM metadata (precision, recall, mAP, strengths)
    - Access to pre-generated labels
    - Recommendations for VLM selection based on strategy needs
    """

    def __init__(self):
        self.registry = Config.VLM_REGISTRY
        self._validate_label_dirs()

    def _validate_label_dirs(self):
        """Check which VLMs have pre-generated labels available."""
        self.available = {}
        self.unavailable = {}
        for key, info in self.registry.items():
            label_dir = Config.get_vlm_label_dir(key)
            if os.path.isdir(label_dir):
                n_files = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
                self.available[key] = n_files
            else:
                self.unavailable[key] = label_dir

        if self.available:
            logger.info(f"VLM labels available: {list(self.available.keys())}")
        if self.unavailable:
            logger.warning(f"VLM labels missing: {list(self.unavailable.keys())}")

    def get_available_vlms(self):
        """Get list of VLM keys that have pre-generated labels."""
        return list(self.available.keys())

    def get_vlm_info(self, key):
        """Get detailed info about a VLM."""
        return self.registry.get(key, {})

    def get_precision(self, key):
        """Get known precision for a VLM."""
        return self.registry.get(key, {}).get("precision", 0.0)

    def get_recall(self, key):
        """Get known recall for a VLM."""
        return self.registry.get(key, {}).get("recall", 0.0)

    def recommend_pair(self):
        """Recommend the best VLM pair for consensus (HL06).

        Strategy: pick highest-precision + highest-recall for complementarity.
        """
        available = self.get_available_vlms()
        if len(available) < 2:
            return available

        # Sort by precision
        by_prec = sorted(available, key=lambda k: self.get_precision(k), reverse=True)
        # Sort by recall
        by_rec = sorted(available, key=lambda k: self.get_recall(k), reverse=True)

        # Best pair: highest precision model + highest recall model
        prec_best = by_prec[0]
        rec_best = by_rec[0]

        if prec_best == rec_best and len(by_rec) > 1:
            rec_best = by_rec[1]

        return [prec_best, rec_best]

    def get_summary_for_brain(self):
        """Generate text summary of VLM pool for Brain context."""
        lines = ["Available VLMs (read-only, pre-generated labels):"]
        for key in sorted(self.available.keys()):
            info = self.registry[key]
            lines.append(
                f"  {key}: prec={info.get('precision', '?')}, "
                f"rec={info.get('recall', '?')}, mAP50={info.get('mAP50', '?')}, "
                f"labels={self.available[key]} files — {info.get('description', '')}"
            )
        if self.unavailable:
            lines.append(f"\nUnavailable (labels not on disk): {list(self.unavailable.keys())}")
        return "\n".join(lines)

    def get_label_count(self, key, stem):
        """Count how many boxes a VLM detected for a given image stem."""
        label_dir = Config.get_vlm_label_dir(key)
        label_file = os.path.join(label_dir, f"{stem}.txt")
        if not os.path.exists(label_file):
            return 0
        count = 0
        with open(label_file) as f:
            for line in f:
                if len(line.strip().split()) >= 5:
                    count += 1
        return count

    def compute_agreement_rate(self, vlm_keys, sample_stems=None, iou_threshold=0.3):
        """Compute pairwise agreement rate between VLMs on a set of images.

        This helps the Brain understand which VLMs agree and which disagree.
        """
        from .label_gen import _load_vlm_boxes, _compute_iou

        if sample_stems is None:
            # Use first VLM's available labels as reference
            ref_dir = Config.get_vlm_label_dir(vlm_keys[0])
            if not os.path.isdir(ref_dir):
                return {}
            sample_stems = [f.replace(".txt", "") for f in os.listdir(ref_dir)
                            if f.endswith(".txt")][:50]  # sample 50

        agreements = {}
        for i, v1 in enumerate(vlm_keys):
            for v2 in vlm_keys[i + 1:]:
                pair_key = f"{v1}+{v2}"
                match_count = 0
                total_count = 0

                for stem in sample_stems:
                    boxes1 = _load_vlm_boxes(v1, stem)
                    boxes2 = _load_vlm_boxes(v2, stem)
                    total_count += len(boxes1) + len(boxes2)

                    for b1 in boxes1:
                        for b2 in boxes2:
                            if _compute_iou(b1[:4], b2[:4]) >= iou_threshold:
                                match_count += 2  # both matched
                                break

                rate = match_count / total_count if total_count > 0 else 0
                agreements[pair_key] = round(rate, 3)

        return agreements
