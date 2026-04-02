"""
VLM Pool — Read-only pool of Vision Language Models for weed detection.

These models are NEVER fine-tuned. They serve as "eyes" for the SuperBrain.

Two modes:
1. Pre-generated labels: read from Phase 2 benchmark label files (fast)
2. Live inference: load VLM onto GPU, run on images, return detections (flexible)

Live inference is the key upgrade: Brain can now ask VLMs to detect on ANY image,
not just images that were pre-labeled. This makes the framework truly adaptive.

Supported live models:
- Florence-2-base (~0.9GB VRAM, best precision 0.789)
- OWLv2-large (~1.75GB VRAM, best recall 0.943)
- Both can coexist on V100-32GB alongside YOLO (5.5GB)
"""

import os
import gc
import json
import logging
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)


class VLMPool:
    """Pool of read-only VLM assistants with live inference capability."""

    def __init__(self):
        self.registry = Config.VLM_REGISTRY
        self._loaded_models = {}  # key -> (model, processor)
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

    # =========================================================
    # LIVE INFERENCE — Load VLM, run on images, return boxes
    # =========================================================

    def load_model(self, vlm_key):
        """Load a VLM onto GPU for live inference."""
        if vlm_key in self._loaded_models:
            return

        import torch

        info = self.registry.get(vlm_key)
        if not info:
            raise ValueError(f"Unknown VLM: {vlm_key}")

        hf_id = info["hf_id"]
        cache_dir = os.path.join(Config.HF_CACHE, "hub")
        logger.info(f"[VLM] Loading {vlm_key} ({hf_id})...")

        if vlm_key in ("florence2_base", "florence2_large"):
            from transformers import AutoModelForCausalLM, AutoProcessor
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.float16, trust_remote_code=True,
                cache_dir=cache_dir)
            model = model.cuda().eval()
            processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=True, cache_dir=cache_dir)
            self._loaded_models[vlm_key] = (model, processor)

        elif vlm_key == "owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            model = Owlv2ForObjectDetection.from_pretrained(
                hf_id, cache_dir=cache_dir)
            model = model.cuda().eval()
            processor = Owlv2Processor.from_pretrained(
                hf_id, cache_dir=cache_dir)
            self._loaded_models[vlm_key] = (model, processor)

        else:
            raise ValueError(f"Live inference not implemented for {vlm_key}. "
                             f"Supported: florence2_base, florence2_large, owlv2")

        logger.info(f"[VLM] {vlm_key} loaded")

    def unload_model(self, vlm_key):
        """Unload a VLM from GPU."""
        if vlm_key in self._loaded_models:
            del self._loaded_models[vlm_key]
            import torch
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"[VLM] {vlm_key} unloaded")

    def unload_all(self):
        """Unload all VLMs from GPU."""
        keys = list(self._loaded_models.keys())
        for key in keys:
            self.unload_model(key)

    def infer(self, vlm_key, image_path):
        """Run live inference on a single image.

        Returns list of detections: [{"box": [cx,cy,w,h], "confidence": float, "label": str}]
        All boxes are in YOLO normalized format [0,1].
        """
        if vlm_key not in self._loaded_models:
            self.load_model(vlm_key)

        model, processor = self._loaded_models[vlm_key]

        if vlm_key in ("florence2_base", "florence2_large"):
            return self._infer_florence(model, processor, image_path)
        elif vlm_key == "owlv2":
            return self._infer_owlv2(model, processor, image_path)
        else:
            raise ValueError(f"No inference function for {vlm_key}")

    def infer_batch(self, vlm_key, image_dir, max_images=None):
        """Run live inference on a directory of images.

        Returns dict: {stem: [detections]} for each image.
        """
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if max_images:
            image_files = image_files[:max_images]

        results = {}
        for img_file in image_files:
            stem = Path(img_file).stem
            img_path = os.path.join(image_dir, img_file)
            try:
                detections = self.infer(vlm_key, img_path)
                results[stem] = detections
            except Exception as e:
                logger.warning(f"Inference failed for {img_file}: {e}")
                results[stem] = []

        logger.info(f"[VLM] {vlm_key} batch: {len(results)} images, "
                    f"{sum(len(d) for d in results.values())} total detections")
        return results

    def save_detections_as_labels(self, detections_dict, output_dir, class_id=8):
        """Save detection results as YOLO-format label files.

        Args:
            detections_dict: {stem: [{"box": [cx,cy,w,h], ...}]}
            output_dir: where to write .txt files
            class_id: class ID to assign (default 8 = novel_weed)
        """
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for stem, dets in detections_dict.items():
            lines = []
            for det in dets:
                b = det["box"]
                lines.append(f"{class_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
            with open(os.path.join(output_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
            count += len(lines)
        logger.info(f"Saved {count} detections to {output_dir}")
        return count

    # --- Florence-2 inference ---

    def _infer_florence(self, model, processor, image_path):
        """Florence-2 <OD> mode → normalized [cx,cy,w,h] boxes."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        inputs = processor(text="<OD>", images=image, return_tensors="pt")
        inputs = {k: v.to(model.device, torch.float16) if v.dtype == torch.float32
                  else v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        text = processor.batch_decode(output, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(text, task="<OD>", image_size=(w, h))

        detections = []
        if "<OD>" in parsed and "bboxes" in parsed["<OD>"]:
            bboxes = parsed["<OD>"]["bboxes"]
            labels = parsed["<OD>"].get("labels", ["weed"] * len(bboxes))
            for bbox, label in zip(bboxes, labels):
                # Florence returns [x1,y1,x2,y2] in pixel coords
                x1, y1, x2, y2 = bbox
                # Convert to YOLO normalized [cx,cy,w,h]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if 0 < bw < 1 and 0 < bh < 1:
                    detections.append({
                        "box": [cx, cy, bw, bh],
                        "confidence": 0.5,  # Florence doesn't output calibrated conf
                        "label": label,
                    })

        return detections

    # --- OWLv2 inference ---

    def _infer_owlv2(self, model, processor, image_path):
        """OWLv2 open-vocabulary detection → normalized [cx,cy,w,h] boxes."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        texts = [["weed"]]
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(model.device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.1
        )

        detections = []
        if results:
            r = results[0]
            for score, box in zip(r["scores"], r["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                # Normalize to [0,1]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if 0 < bw < 1 and 0 < bh < 1:
                    detections.append({
                        "box": [cx, cy, bw, bh],
                        "confidence": float(score),
                        "label": "weed",
                    })

        return detections

    # =========================================================
    # PRE-GENERATED LABELS — Read from disk (existing)
    # =========================================================

    def get_available_vlms(self):
        """Get list of VLM keys that have pre-generated labels."""
        return list(self.available.keys())

    def get_vlm_info(self, key):
        return self.registry.get(key, {})

    def get_precision(self, key):
        return self.registry.get(key, {}).get("precision", 0.0)

    def get_recall(self, key):
        return self.registry.get(key, {}).get("recall", 0.0)

    def recommend_pair(self):
        """Recommend the best VLM pair (HL06: high-prec + high-recall)."""
        available = self.get_available_vlms()
        if len(available) < 2:
            return available
        by_prec = sorted(available, key=lambda k: self.get_precision(k), reverse=True)
        by_rec = sorted(available, key=lambda k: self.get_recall(k), reverse=True)
        prec_best = by_prec[0]
        rec_best = by_rec[0]
        if prec_best == rec_best and len(by_rec) > 1:
            rec_best = by_rec[1]
        return [prec_best, rec_best]

    def get_summary_for_brain(self):
        """Generate text summary of VLM pool for Brain context."""
        lines = ["Available VLMs (read-only):"]
        for key in sorted(self.available.keys()):
            info = self.registry[key]
            lines.append(
                f"  {key}: prec={info.get('precision', '?')}, "
                f"rec={info.get('recall', '?')}, mAP50={info.get('mAP50', '?')}, "
                f"labels={self.available[key]} files — {info.get('description', '')}"
            )
        lines.append("\nLive inference supported: florence2_base, florence2_large, owlv2")
        loaded = list(self._loaded_models.keys())
        if loaded:
            lines.append(f"Currently loaded on GPU: {loaded}")
        return "\n".join(lines)

    def get_label_count(self, key, stem):
        """Count boxes in a VLM's pre-generated label file."""
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

    def inspect_label_quality(self, vlm_key, sample_size=20):
        """Inspect label quality for a VLM — useful for Brain to decide trust level.

        Returns stats about detection density, box sizes, consistency.
        """
        label_dir = Config.get_vlm_label_dir(vlm_key)
        if not os.path.isdir(label_dir):
            return {"error": f"No labels for {vlm_key}"}

        files = [f for f in os.listdir(label_dir) if f.endswith('.txt')][:sample_size]
        box_counts = []
        box_sizes = []

        for f in files:
            with open(os.path.join(label_dir, f)) as fh:
                lines = [l.strip().split() for l in fh if len(l.strip().split()) >= 5]
                box_counts.append(len(lines))
                for parts in lines:
                    w, h = float(parts[3]), float(parts[4])
                    box_sizes.append(w * h)

        import numpy as np
        return {
            "vlm": vlm_key,
            "sample_size": len(files),
            "avg_boxes_per_image": round(np.mean(box_counts), 1) if box_counts else 0,
            "max_boxes_per_image": max(box_counts) if box_counts else 0,
            "images_with_no_detection": sum(1 for c in box_counts if c == 0),
            "avg_box_area": round(np.mean(box_sizes), 4) if box_sizes else 0,
            "median_box_area": round(np.median(box_sizes), 4) if box_sizes else 0,
        }
