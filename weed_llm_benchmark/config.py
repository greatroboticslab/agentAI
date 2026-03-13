"""
Configuration for weed detection LLM benchmark.
Defines all models to test and shared settings.
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "images")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
HF_CACHE = "/ocean/projects/cis240145p/byler/hf_cache"
OLLAMA_BIN = "/ocean/projects/cis240145p/byler/ollama/bin/ollama"
OLLAMA_MODELS = "/ocean/projects/cis240145p/byler/ollama/models"

# ============================================================
# Prompt (shared across all models)
# ============================================================
WEED_DETECTION_PROMPT = """Analyze this agricultural field image carefully.

Your task:
1. Identify ALL visible plants in the image, distinguishing between crops and weeds.
2. For each detected plant/weed, provide:
   - "label": the type (e.g., "weed", "crop", "grass", or specific species name if identifiable)
   - "confidence": your confidence level ("high", "medium", "low")
   - "bbox": approximate bounding box as [x_min, y_min, x_max, y_max] in percentage of image dimensions (0-100)
   - "description": brief description of the plant appearance

Return ONLY valid JSON in this exact format:
{
  "detections": [
    {
      "label": "weed",
      "species": "dandelion",
      "confidence": "high",
      "bbox": [10, 20, 25, 40],
      "description": "Yellow flowering weed with jagged leaves"
    }
  ],
  "scene_description": "Brief description of the overall field scene",
  "weed_severity": "none|low|medium|high",
  "crop_type": "identified crop type or unknown"
}
"""

# Qwen2.5-VL specific grounding prompt (uses native bbox_2d format)
QWEN_GROUNDING_PROMPT = """Detect all plants in this agricultural field image.
For each plant, classify it as either "weed" or "crop".
If you can identify the species, include that too.
Return the result as a JSON list with bounding boxes.

Output format:
{"detections": [{"bbox_2d": [x1, y1, x2, y2], "label": "weed", "species": "dandelion", "confidence": "high"}], "weed_severity": "none|low|medium|high", "crop_type": "unknown"}"""

# Simpler prompt for models that struggle with complex instructions
WEED_DETECTION_PROMPT_SIMPLE = """Look at this agricultural field image.
Identify all weeds and crops visible.
For each plant, estimate its location as a bounding box [x_min, y_min, x_max, y_max] in percentage (0-100) of the image.
Return the result as JSON with format: {"detections": [{"label": "weed/crop", "bbox": [x1,y1,x2,y2], "confidence": "high/medium/low"}]}"""

# ============================================================
# Model Definitions
# ============================================================

# --- Ollama models (easiest to set up) ---
# Ranked by weed detection suitability based on grounding/bbox capability
OLLAMA_MODELS_LIST = [
    # TIER 1: Native bounding box / grounding support
    {
        "name": "qwen2.5vl:7b",
        "description": "Qwen2.5-VL 7B - BEST: native bbox JSON, grounding support",
        "size": "~6GB",
        "tier": 1,
        "has_grounding": True,
        "use_grounding_prompt": True,
    },
    {
        "name": "qwen2.5vl:3b",
        "description": "Qwen2.5-VL 3B - native bbox, lighter, may be more reliable for grounding",
        "size": "~3.2GB",
        "tier": 1,
        "has_grounding": True,
        "use_grounding_prompt": True,
    },
    {
        "name": "moondream",
        "description": "Moondream2 1.8B - dedicated detect() API, fast second-opinion",
        "size": "~1.7GB",
        "tier": 1,
        "has_grounding": True,
        "use_grounding_prompt": False,  # Uses detect() API instead
    },
    # TIER 2: Good vision but no native grounding
    {
        "name": "llama3.2-vision:11b",
        "description": "Meta Llama 3.2 Vision 11B - strong general vision, no bbox",
        "size": "~7GB",
        "tier": 2,
        "has_grounding": False,
    },
    {
        "name": "minicpm-v",
        "description": "MiniCPM-V - efficient, strong OCR/vision, no bbox",
        "size": "~5GB",
        "tier": 2,
        "has_grounding": False,
    },
    {
        "name": "llava:13b",
        "description": "LLaVA 1.5 13B - classic vision-language, classifier only",
        "size": "~8GB",
        "tier": 3,
        "has_grounding": False,
    },
    {
        "name": "llava:34b",
        "description": "LLaVA 1.6 34B - largest LLaVA, classifier only",
        "size": "~20GB",
        "tier": 3,
        "has_grounding": False,
    },
]

# --- HuggingFace / Transformers models (more flexible, better quality) ---
HF_MODELS_LIST = [
    {
        "name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "description": "Qwen2.5 VL 7B - top-tier open vision model, native bbox support",
        "conda_env": "qwen",
        "backend": "transformers",
        "tier": 1,
        "already_downloaded": True,
    },
    {
        "name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "Qwen2.5 VL 3B - smaller but still strong",
        "conda_env": "qwen",
        "backend": "transformers",
        "tier": 2,
        "already_downloaded": True,
    },
    {
        "name": "openbmb/MiniCPM-V-2_6",
        "description": "MiniCPM-V 2.6 - efficient, good detail recognition",
        "conda_env": "minicpm",
        "backend": "transformers",
        "tier": 1,
        "already_downloaded": False,
    },
    {
        "name": "BLIP3o/BLIP3o-Model-8B",
        "description": "BLIP3o 8B - Salesforce's latest multimodal",
        "conda_env": None,  # may need new env
        "backend": "transformers",
        "tier": 2,
        "already_downloaded": True,
    },
    {
        "name": "microsoft/Florence-2-large",
        "description": "Florence-2 - specialized in visual grounding & detection",
        "conda_env": None,
        "backend": "transformers",
        "tier": 1,
        "already_downloaded": False,
        "notes": "Best for bbox/grounding tasks. Needs separate setup.",
    },
    {
        "name": "OpenGVLab/InternVL2-8B",
        "description": "InternVL2 8B - strong Chinese/English vision model",
        "conda_env": None,
        "backend": "transformers",
        "tier": 1,
        "already_downloaded": False,
    },
]
