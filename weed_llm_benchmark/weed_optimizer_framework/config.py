"""
Configuration — all paths, constants, model registry, and cluster settings.

Centralized here so no other file has hardcoded paths or magic numbers.
"""

import os
from pathlib import Path


class Config:
    """Global configuration for the Weed Optimizer Framework."""

    # --- Base paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMEWORK_DIR = os.path.join(BASE_DIR, "results", "framework")
    HF_CACHE = os.environ.get("HF_HOME", "/ocean/projects/cis240145p/byler/hf_cache")

    # --- Detection model (v3.0: pursue accuracy limit, real-time speed not required) ---
    # Default = yolo26x (verified URL in ultralytics 8.4+ GitHub assets, 55.7M params).
    # mega_trainer walks DETECTION_MODEL_FALLBACKS in order — first one that loads wins.
    # Override via env: WEED_DETECTION_MODEL=<name.pt>
    DETECTION_MODEL = os.environ.get("WEED_DETECTION_MODEL", "yolo26x.pt")
    DETECTION_MODEL_FALLBACKS = [
        "yolo26x.pt",    # latest (Apr 2026), mAP50-95=57.5, 55.7M
        "yolo12x.pt",    # mAP50-95=55.2, 59.1M (if ultralytics 8.4+)
        "yolo11x.pt",    # mAP50-95=54.7, 56.9M (stable)
        "yolov10x.pt",   # mAP50-95=54.4, 31.8M (confirmed loads on cluster)
        "yolo11l.pt",    # mAP50-95=53.4, 25.3M (smaller fallback)
    ]
    DETECTION_MODEL_VARIANTS = {
        "yolo26x": {"params": "55.7M", "mAP50_95": 57.5, "description": "Latest, best accuracy (Apr 2026)"},
        "yolo12x": {"params": "59.1M", "mAP50_95": 55.2, "description": "YOLO12 (ultralytics 8.4+)"},
        "yolo11x": {"params": "56.9M", "mAP50_95": 54.7, "description": "YOLO11 largest, stable"},
        "yolov10x": {"params": "31.8M", "mAP50_95": 54.4, "description": "Confirmed loads on cluster"},
        "rt-detr-x": {"params": "65M", "mAP50_95": 54.8, "description": "Transformer detector"},
    }

    # --- Leave-4-Out dataset paths ---
    L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
    SP8_DIR = os.path.join(L4O_DIR, "dataset_8species")
    HOLDOUT_DIR = os.path.join(L4O_DIR, "dataset_holdout")
    YOLO_8SP_WEIGHTS = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")
    LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")

    # --- Species ---
    ALL_CLASSES = {
        0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
        4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
        8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge",
    }
    TRAIN_SPECIES_IDS = {0, 1, 6, 7, 8, 9, 10, 11}  # 8 species YOLO was trained on
    HOLDOUT_SPECIES_IDS = {2, 3, 4, 5}                 # 4 unseen species
    NOVEL_CLASS_ID = 8  # class id used for novel weed detections in pseudo-labels

    # --- VLM Model Registry (read-only, never fine-tuned) ---
    VLM_REGISTRY = {
        "florence2_base": {
            "hf_id": "microsoft/Florence-2-base",
            "precision": 0.789,
            "recall": 0.519,
            "mAP50": 0.434,
            "label_dir": "florence2_base_cottonweeddet12",
            "conda_env": "compat",  # needs transformers 4.46
            "description": "Best VLM. Highest precision (0.789). 0.23B params.",
        },
        "florence2_large": {
            "hf_id": "microsoft/Florence-2-large",
            "precision": 0.692,
            "recall": 0.431,
            "mAP50": 0.329,
            "label_dir": "florence2_cottonweeddet12",
            "conda_env": "compat",
            "description": "Larger Florence-2. Lower precision than base. 0.77B.",
        },
        "owlv2": {
            "hf_id": "google/owlv2-large-patch14-ensemble",
            "precision": 0.194,
            "recall": 0.943,
            "mAP50": 0.184,
            "label_dir": "owlv2_cottonweeddet12",
            "conda_env": "bench",
            "description": "Highest recall (0.943). Open-vocabulary detector.",
        },
        "internvl2_8b": {
            "hf_id": "OpenGVLab/InternVL2-8B",
            "precision": 0.545,
            "recall": 0.354,
            "mAP50": 0.208,
            "label_dir": "internvl2_cottonweeddet12",
            "conda_env": "compat",
            "description": "Good precision. 8B VLM.",
        },
        "qwen25_vl_3b": {
            "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "precision": 0.333,
            "recall": 0.249,
            "mAP50": 0.196,
            "label_dir": "qwen3b_cottonweeddet12",
            "conda_env": "bench",
            "description": "Native bbox JSON. 3B.",
        },
        "minicpm_v45": {
            "hf_id": "openbmb/MiniCPM-V-4o-5",
            "precision": 0.407,
            "recall": 0.340,
            "mAP50": 0.192,
            "label_dir": "minicpm_v45_cottonweeddet12",
            "conda_env": "bench",
            "description": "Feb 2026 model. 4.5B.",
        },
        "qwen25_vl_7b": {
            "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "precision": 0.334,
            "recall": 0.214,
            "mAP50": 0.176,
            "label_dir": "qwen7b_cottonweeddet12",
            "conda_env": "bench",
            "description": "Native bbox JSON. 7B.",
        },
    }

    # --- Brain model options (swappable) ---
    BRAIN_MODELS = {
        "qwen25_7b": {
            "hf_id": "Qwen/Qwen2.5-7B-Instruct",
            "vram_gb": 14,
            "description": "Legacy default.",
        },
        "deepseek_r1_7b": {
            "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "vram_gb": 14,
            "description": "Good reasoning but no native tool calling.",
        },
        "gemma4_26b": {
            "hf_id": "gemma4:26b",
            "vram_gb": 18,
            "description": "Gemma 4 MoE (26B/3.8B active). Native tool calling. Best quality-per-VRAM.",
        },
        "qwen3_14b": {
            "hf_id": "qwen3:14b",
            "vram_gb": 10,
            "description": "Qwen3 14B. Most reliable tool calling. Fits easily on V100.",
        },
    }

    # --- Training defaults ---
    DEFAULT_STRATEGY = {
        "vlm_models": ["florence2_base", "owlv2"],
        "min_votes": 2,
        "consensus_iou": 0.3,
        "use_yolo_old": True,
        "lr": 0.001,
        "epochs": 50,
        "freeze_layers": 0,
        "replay_ratio": 0.3,
        "batch_size": -1,  # auto
        "patience": 15,
        "name": "default_consensus",
    }

    # --- Quality thresholds ---
    FORGETTING_THRESHOLD = 0.90    # old F1 must stay above this
    MIN_CONSENSUS_BOXES = 5        # min pseudo-labels to proceed with training

    # --- v3.0 mega training gate ---
    # Brain must download at least this many bbox-labeled images before train_yolo_mega fires.
    # Rationale: user wants "tens of thousands to hundreds of thousands" — 5648 is not enough.
    # Override via env: WEED_MEGA_MIN_IMAGES=<n>
    MEGA_TRAIN_MIN_IMAGES = int(os.environ.get("WEED_MEGA_MIN_IMAGES", "50000"))
    CONFIDENCE_THRESHOLD = 0.25    # YOLO inference for label generation / detection
    EVAL_CONFIDENCE = 0.001        # Low conf for mAP evaluation (full PR curve, standard practice)
    IOU_MATCH_THRESHOLD = 0.5      # IoU for TP/FP matching in evaluation

    # --- mAP evaluation IoU thresholds ---
    MAP50_THRESHOLDS = [0.5]
    MAP50_95_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    @classmethod
    def get_vlm_label_dir(cls, vlm_key):
        """Get the label directory path for a VLM."""
        info = cls.VLM_REGISTRY.get(vlm_key, {})
        dirname = info.get("label_dir", "")
        return os.path.join(cls.LABELED_DIR, dirname, "detected", "labels")

    @classmethod
    def get_vlm_precision(cls, vlm_key):
        """Get the known precision for a VLM."""
        return cls.VLM_REGISTRY.get(vlm_key, {}).get("precision", 0.0)

    @classmethod
    def ensure_dirs(cls):
        """Create required directories."""
        os.makedirs(cls.FRAMEWORK_DIR, exist_ok=True)

    @classmethod
    def get_species_names(cls, class_ids=None):
        """Get species name list for YOLO data.yaml."""
        if class_ids is None:
            class_ids = sorted(cls.TRAIN_SPECIES_IDS)
        return [cls.ALL_CLASSES[i] for i in class_ids] + ["novel_weed"]
