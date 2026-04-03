"""
Model Discovery — Find, download, and run external weed detection models.

Sources:
1. HuggingFace Hub — search for weed detection models, download, run inference
2. GitHub — search repos with weed detection, clone, extract models

This tool lets the Brain autonomously discover new models beyond our pre-defined
VLM pool. Found models become additional "tools" for label generation.

Key HuggingFace models (pre-researched):
- machinelearningzuu/detr-resnet-50_finetuned-weed-detection (DETR, ready to use)
- MuayThaiLegz/WeedDetection-YOLOv8s (YOLOv8, ready to use)
- Vigneshtrill/Agribot_Weed_Detection (YOLO, Nov 2025)

Key GitHub repos:
- AlexOlsen/DeepWeeds (ResNet50, 17.5K images, 248 stars)
- cropandweed/cropandweed-dataset (largest dataset, 113 stars)
- Daraan/CropAndWeedDetection (YOLOv7)
"""

import os
import gc
import json
import logging
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)

# Pre-researched models that are known to work
KNOWN_HF_MODELS = {
    "detr_weed": {
        "hf_id": "machinelearningzuu/detr-resnet-50_finetuned-weed-detection",
        "type": "detr",
        "description": "DETR ResNet-50 fine-tuned on weed detection",
        "framework": "transformers",
    },
    "detr_deformable_weed": {
        "hf_id": "machinelearningzuu/deformable-detr-box-finetuned-weed-detection",
        "type": "deformable_detr",
        "description": "Deformable DETR fine-tuned on weed detection",
        "framework": "transformers",
    },
    "yolov8s_weed": {
        "hf_id": "MuayThaiLegz/WeedDetection-YOLOv8s",
        "type": "yolov8",
        "description": "YOLOv8s fine-tuned on weed detection (recent)",
        "framework": "ultralytics",
    },
}

KNOWN_GITHUB_REPOS = [
    {
        "url": "https://github.com/AlexOlsen/DeepWeeds",
        "name": "DeepWeeds",
        "stars": 248,
        "type": "classification",
        "description": "ResNet50 + InceptionV3, 17.5K images, 8 Australian weed species",
    },
    {
        "url": "https://github.com/cropandweed/cropandweed-dataset",
        "name": "CropAndWeed",
        "stars": 113,
        "type": "detection",
        "description": "Multi-modal annotations, WACV 2023, largest weed dataset",
    },
    {
        "url": "https://github.com/Daraan/CropAndWeedDetection",
        "name": "CropAndWeedDetection",
        "stars": 21,
        "type": "detection",
        "description": "YOLOv7 fine-tuned on CropAndWeed dataset",
    },
]


class ModelDiscovery:
    """Discover and use external weed detection models."""

    def __init__(self):
        self.cache_dir = os.path.join(Config.HF_CACHE, "hub")
        self._loaded_models = {}
        self.discovered_models = list(KNOWN_HF_MODELS.keys())

    # =========================================================
    # SEARCH — Find models on HuggingFace / GitHub
    # =========================================================

    def search_huggingface(self, query="weed detection", max_results=10):
        """Search HuggingFace Hub for weed detection models.

        Returns list of model info dicts.
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(
                search=query,
                sort="downloads",
                direction=-1,
                limit=max_results,
            )
            results = []
            for m in models:
                results.append({
                    "model_id": m.modelId,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "pipeline_tag": getattr(m, "pipeline_tag", "unknown"),
                    "last_modified": str(getattr(m, "lastModified", "")),
                })
            logger.info(f"[Discovery] Found {len(results)} models on HuggingFace for '{query}'")
            return results
        except Exception as e:
            logger.warning(f"[Discovery] HuggingFace search failed: {e}")
            # Return pre-researched list
            return [{"model_id": v["hf_id"], "type": v["type"],
                     "description": v["description"]}
                    for v in KNOWN_HF_MODELS.values()]

    def list_known_models(self):
        """List all pre-researched models."""
        models = []
        for key, info in KNOWN_HF_MODELS.items():
            models.append({
                "key": key,
                "hf_id": info["hf_id"],
                "type": info["type"],
                "description": info["description"],
                "loaded": key in self._loaded_models,
            })
        return models

    def list_known_repos(self):
        """List known GitHub repositories."""
        return KNOWN_GITHUB_REPOS

    # =========================================================
    # LOAD + INFER — Download model and run detection
    # =========================================================

    def load_model(self, model_key):
        """Load a pre-researched model by key."""
        if model_key in self._loaded_models:
            return

        if model_key not in KNOWN_HF_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Known: {list(KNOWN_HF_MODELS.keys())}")

        info = KNOWN_HF_MODELS[model_key]
        hf_id = info["hf_id"]
        model_type = info["type"]

        logger.info(f"[Discovery] Loading {model_key} ({hf_id})...")

        if model_type in ("detr", "deformable_detr"):
            self._load_detr(model_key, hf_id)
        elif model_type == "yolov8":
            self._load_yolo(model_key, hf_id)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"[Discovery] {model_key} loaded")

    def _load_detr(self, key, hf_id):
        """Load a DETR model from HuggingFace."""
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor = AutoImageProcessor.from_pretrained(hf_id, cache_dir=self.cache_dir)
        model = AutoModelForObjectDetection.from_pretrained(
            hf_id, cache_dir=self.cache_dir)
        model = model.cuda().eval()
        self._loaded_models[key] = {"model": model, "processor": processor, "type": "detr"}

    def _load_yolo(self, key, hf_id):
        """Load a YOLO model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=hf_id, filename="best.pt",
                                         cache_dir=self.cache_dir)
        except Exception:
            # Try alternative filenames
            from huggingface_hub import hf_hub_download
            try:
                model_path = hf_hub_download(repo_id=hf_id, filename="yolov8s.pt",
                                             cache_dir=self.cache_dir)
            except Exception:
                logger.error(f"Could not find YOLO weights in {hf_id}")
                return

        from ultralytics import YOLO
        model = YOLO(model_path)
        self._loaded_models[key] = {"model": model, "processor": None, "type": "yolo"}

    def unload_model(self, key):
        """Unload a model from GPU."""
        if key in self._loaded_models:
            del self._loaded_models[key]
            import torch
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"[Discovery] {key} unloaded")

    def unload_all(self):
        """Unload all models."""
        for key in list(self._loaded_models.keys()):
            self.unload_model(key)

    def infer(self, model_key, image_path):
        """Run inference on a single image.

        Returns list of detections: [{"box": [cx,cy,w,h], "confidence": float, "label": str}]
        """
        if model_key not in self._loaded_models:
            self.load_model(model_key)

        info = self._loaded_models[model_key]
        model_type = info["type"]

        if model_type == "detr":
            return self._infer_detr(info, image_path)
        elif model_type == "yolo":
            return self._infer_yolo(info, image_path)
        else:
            return []

    def _infer_detr(self, info, image_path):
        """DETR inference → normalized [cx,cy,w,h] boxes."""
        import torch
        from PIL import Image

        model = info["model"]
        processor = info["processor"]

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(model.device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )

        detections = []
        if results:
            r = results[0]
            for score, label_id, box in zip(r["scores"], r["labels"], r["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if 0 < bw < 1 and 0 < bh < 1:
                    label = model.config.id2label.get(label_id.item(), "weed")
                    detections.append({
                        "box": [cx, cy, bw, bh],
                        "confidence": float(score),
                        "label": label,
                    })

        return detections

    def _infer_yolo(self, info, image_path):
        """YOLO inference → normalized [cx,cy,w,h] boxes."""
        model = info["model"]
        results = model.predict(image_path, conf=0.25, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cx, cy, bw, bh = box.xywhn[0].tolist()
                detections.append({
                    "box": [cx, cy, bw, bh],
                    "confidence": float(box.conf[0]),
                    "label": r.names.get(int(box.cls[0]), "weed"),
                })

        return detections

    def infer_batch(self, model_key, image_dir, max_images=None):
        """Run inference on a directory of images.

        Returns dict: {stem: [detections]}
        """
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
                dets = self.infer(model_key, img_path)
                results[stem] = dets
            except Exception as e:
                logger.warning(f"Inference failed for {img_file}: {e}")
                results[stem] = []

        total_dets = sum(len(d) for d in results.values())
        logger.info(f"[Discovery] {model_key} batch: {len(results)} images, {total_dets} detections")
        return results

    def save_detections_as_labels(self, detections_dict, output_dir, class_id=8):
        """Save detection results as YOLO-format label files."""
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
        logger.info(f"[Discovery] Saved {count} detections to {output_dir}")
        return count

    def get_summary_for_brain(self):
        """Summary for Brain context."""
        lines = ["External model discovery:"]
        lines.append(f"  Known HF models: {list(KNOWN_HF_MODELS.keys())}")
        loaded = list(self._loaded_models.keys())
        if loaded:
            lines.append(f"  Currently loaded: {loaded}")
        lines.append("  Actions: search_huggingface, load_model, infer_batch")
        lines.append("  DETR weed detector available (detr_weed) — transformer-based, different from our VLMs")
        return "\n".join(lines)
