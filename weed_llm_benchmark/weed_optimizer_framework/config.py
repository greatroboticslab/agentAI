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
            "description": "Current default. Adequate for strategy generation.",
        },
        "deepseek_r1_7b": {
            "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "vram_gb": 14,
            "description": "Stronger reasoning. Good for discovering novel strategies.",
        },
        "qwen25_72b_awq": {
            "hf_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "vram_gb": 20,
            "description": "Most intelligent. May push V100-32GB limits.",
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
    CONFIDENCE_THRESHOLD = 0.25    # YOLO inference confidence
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
