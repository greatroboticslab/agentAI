#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roboflow Bridge: Download images -> LLM weed detection -> Upload labeled results back.

This lets you:
1. Download unlabeled/labeled images from Roboflow
2. Run LLM (Qwen2.5-VL) weed detection on the cluster
3. Upload the LLM-labeled results back to Roboflow for visual inspection

Usage (on GPU node):
    conda activate qwen
    python roboflow_bridge.py

Or non-interactive:
    python roboflow_bridge.py --download --project weed1-rmxbe --version 1
    python roboflow_bridge.py --detect --source downloads/weed1-rmxbe
    python roboflow_bridge.py --upload --source llm_labeled/detected --project weed1-llm-labeled
    python roboflow_bridge.py --all --project weed1-rmxbe --version 1 --upload-project weed1-llm-labeled
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# ============================================================
# Config
# ============================================================
WORKSPACE_ID = "mtsu-2h73y"
HF_CACHE = "/ocean/projects/cis240145p/byler/hf_cache"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")
API_KEY_FILE = os.path.join(BASE_DIR, ".roboflow_key")

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")


def get_api_key():
    """Load API key from file, or ask user and save it."""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    print("[!] No Roboflow API key found.")
    print("    Go to: https://app.roboflow.com/settings/api")
    key = input("    Paste your API key here: ").strip()
    if key:
        with open(API_KEY_FILE, 'w') as f:
            f.write(key)
        os.chmod(API_KEY_FILE, 0o600)  # only owner can read
        print(f"    [+] Key saved to {API_KEY_FILE}")
    return key


API_KEY = None  # loaded lazily


def ensure_api_key():
    global API_KEY
    if API_KEY is None:
        API_KEY = get_api_key()
    return API_KEY


def get_input(prompt, default=None):
    if default:
        val = input(f"{prompt} (default: {default}): ").strip()
        return val if val else default
    return input(f"{prompt}: ").strip()


def list_roboflow_projects():
    """List all projects in the workspace and let user pick one."""
    from roboflow import Roboflow

    key = ensure_api_key()
    print("\n[*] Fetching your Roboflow projects...")
    try:
        rf = Roboflow(api_key=key)
        ws = rf.workspace(WORKSPACE_ID)
        projects = ws.project_list

        if not projects:
            print("[!] No projects found in workspace.")
            return None, None

        print(f"\n  Your Projects ({WORKSPACE_ID}):")
        print(f"  {'#':<4} {'Project ID':<30} {'Type':<20} {'Images':<8}")
        print(f"  {'-'*65}")
        for i, p in enumerate(projects):
            pid = p.get("id", "?")
            ptype = p.get("type", "?")
            # Try to get image count
            num_images = p.get("annotation_group", {}).get("num_images", "?") if isinstance(p.get("annotation_group"), dict) else "?"
            if num_images == "?":
                num_images = p.get("num_images", "?")
            print(f"  {i+1:<4} {pid:<30} {ptype:<20} {num_images}")

        print()
        choice = get_input(f"Select project number (1-{len(projects)}), or type project ID")

        # If user typed a number, map to project
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(projects):
                selected = projects[idx]
                project_id = selected.get("id", "?")
                # Strip workspace prefix (e.g. "mtsu-2h73y/weed2okok" -> "weed2okok")
                if "/" in project_id:
                    project_id = project_id.split("/")[-1]
                print(f"  -> Selected: {project_id}")

                # List versions
                try:
                    proj = ws.project(project_id)
                    versions = proj.versions()
                    if versions:
                        print(f"\n  Versions for '{project_id}':")
                        for v in versions:
                            vid = v.version if hasattr(v, 'version') else '?'
                            print(f"    v{vid}")
                except Exception:
                    pass

                return project_id, selected
        except ValueError:
            # User typed a project ID directly - strip prefix if included
            if "/" in choice:
                choice = choice.split("/")[-1]
            return choice, None

        return None, None

    except Exception as e:
        print(f"[!] Failed to list projects: {e}")
        if "401" in str(e) or "API key" in str(e):
            print("[!] API key is invalid. Deleting saved key...")
            if os.path.exists(API_KEY_FILE):
                os.remove(API_KEY_FILE)
            API_KEY = None
            return list_roboflow_projects()  # retry with new key
        return None, None


# ============================================================
# 1. Download from Roboflow
# ============================================================
def download_from_roboflow(project_id, version_num="1", fmt="yolov8", download_path=None):
    from roboflow import Roboflow

    key = ensure_api_key()
    if download_path is None:
        download_path = os.path.join(DOWNLOAD_DIR, project_id)

    print(f"\n{'='*50}")
    print(f"DOWNLOADING FROM ROBOFLOW")
    print(f"{'='*50}")
    print(f"  Workspace: {WORKSPACE_ID}")
    print(f"  Project:   {project_id}")
    print(f"  Version:   {version_num}")
    print(f"  Format:    {fmt}")
    print(f"  Save to:   {download_path}")

    try:
        rf = Roboflow(api_key=key)
        project = rf.workspace(WORKSPACE_ID).project(project_id)
        dataset = project.version(int(version_num)).download(fmt, location=download_path)
        print(f"\n[+] Download complete: {download_path}")

        # Count images
        img_count = 0
        for split in ["train", "valid", "test", ""]:
            img_dir = os.path.join(download_path, split, "images") if split else os.path.join(download_path, "images")
            if os.path.isdir(img_dir):
                count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if count > 0:
                    print(f"    {split or 'root'}/images: {count} images")
                    img_count += count

        print(f"  Total images: {img_count}")
        return download_path
    except Exception as e:
        print(f"[!] Download failed: {e}")
        return None


# ============================================================
# 2. LLM Detection (Multi-model support)
# ============================================================

# Registry of all supported models
MODEL_REGISTRY = {
    "qwen7b": {
        "name": "Qwen2.5-VL-7B",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "type": "qwen",
        "size": "~14GB",
        "description": "Best overall - native bbox, strong weed recognition",
    },
    "qwen3b": {
        "name": "Qwen2.5-VL-3B",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "qwen",
        "size": "~6GB",
        "description": "Lighter, faster, still good grounding",
    },
    "minicpm": {
        "name": "MiniCPM-V-2.6",
        "model_id": "openbmb/MiniCPM-V-2_6",
        "type": "minicpm",
        "size": "~16GB",
        "description": "Strong vision, efficient, good detail recognition",
    },
    "internvl2": {
        "name": "InternVL2-8B",
        "model_id": "OpenGVLab/InternVL2-8B",
        "type": "internvl",
        "size": "~16GB",
        "description": "Strong grounding, good plant recognition",
    },
    "florence2": {
        "name": "Florence-2-large",
        "model_id": "microsoft/Florence-2-large",
        "type": "florence",
        "size": "~1.5GB",
        "description": "Microsoft's lightweight vision model with native object detection",
    },
    # ---- New models added 2026-03-16 (Phase 2 expansion) ----
    "qwen3_8b": {
        "name": "Qwen3-VL-8B",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "type": "qwen3",
        "size": "~16GB",
        "description": "Latest Qwen VL with enhanced grounding (Jan 2026)",
    },
    "grounding_dino": {
        "name": "Grounding-DINO-base",
        "model_id": "IDEA-Research/grounding-dino-base",
        "type": "grounding_dino",
        "size": "~1GB",
        "description": "Open-set object detector with text prompts (ECCV 2024)",
    },
    "paligemma2": {
        "name": "PaliGemma2-3B-mix",
        "model_id": "google/paligemma2-3b-mix-448",
        "type": "paligemma",
        "size": "~6GB",
        "description": "Google detection model with native <loc> coordinate tokens",
    },
    "yolo_world": {
        "name": "YOLO-World-v2-L",
        "model_id": "yolov8l-worldv2",
        "type": "yolo_world",
        "size": "~200MB",
        "description": "Open-vocabulary YOLO with text-prompted detection",
    },
    "minicpm_v45": {
        "name": "MiniCPM-V-4.5",
        "model_id": "openbmb/MiniCPM-V-4_5",
        "type": "minicpm_v45",
        "size": "~16GB",
        "description": "Strong 8B VLM with native detect mode (Feb 2026)",
    },
    "molmo2": {
        "name": "Molmo-7B-D",
        "model_id": "allenai/Molmo-7B-D-0924",
        "type": "molmo",
        "size": "~14GB",
        "description": "Allen AI model with precise pixel coordinate output",
    },
    "deepseek_vl2": {
        "name": "DeepSeek-VL2-Small",
        "model_id": "deepseek-ai/deepseek-vl2-small",
        "type": "deepseek_vl",
        "size": "~8GB",
        "description": "MoE VLM with grounding tokens (Dec 2024)",
    },
    # ---- New models added 2026-03-17 (Phase 2 expansion v2) ----
    "florence2_base": {
        "name": "Florence-2-base",
        "model_id": "microsoft/Florence-2-base",
        "type": "florence",
        "size": "~0.9GB",
        "description": "Florence-2 base (0.23B) — smaller detection baseline",
    },
    "owlv2": {
        "name": "OWLv2-large",
        "model_id": "google/owlv2-large-patch14-ensemble",
        "type": "owlv2",
        "size": "~1.75GB",
        "description": "Google zero-shot object detector with text queries",
    },
    "omdet_turbo": {
        "name": "OmDet-Turbo",
        "model_id": "omlab/omdet-turbo-swin-tiny-hf",
        "type": "omdet",
        "size": "~0.9GB",
        "description": "Fast zero-shot detector (100 FPS on COCO)",
    },
    "internvl2_4b": {
        "name": "InternVL2-4B",
        "model_id": "OpenGVLab/InternVL2-4B",
        "type": "internvl",
        "size": "~8GB",
        "description": "Mid-size InternVL2 for scaling analysis",
    },
    "internvl2_2b": {
        "name": "InternVL2-2B",
        "model_id": "OpenGVLab/InternVL2-2B",
        "type": "internvl",
        "size": "~4GB",
        "description": "Smallest InternVL2 for scaling analysis",
    },
    "internvl2_5_8b": {
        "name": "InternVL2.5-8B",
        "model_id": "OpenGVLab/InternVL2_5-8B",
        "type": "internvl",
        "size": "~16GB",
        "description": "Improved InternVL2 (Dec 2024)",
    },
    "mm_gdino": {
        "name": "MM-Grounding-DINO-L",
        "model_id": "ShilongLiu/GroundingDINO",
        "type": "grounding_dino",
        "size": "~1.5GB",
        "description": "Improved Grounding DINO with better zero-shot AP",
    },
}


def select_model_interactive():
    """Interactive model selection menu."""
    print(f"\n  Available Models:")
    print(f"  {'#':<4} {'Key':<12} {'Name':<22} {'Size':<8} {'Description'}")
    print(f"  {'-'*80}")
    keys = list(MODEL_REGISTRY.keys())
    for i, key in enumerate(keys):
        m = MODEL_REGISTRY[key]
        print(f"  {i+1:<4} {key:<12} {m['name']:<22} {m['size']:<8} {m['description']}")
    print()
    choice = get_input(f"Select model (1-{len(keys)}) or type key", "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return keys[idx]
    except ValueError:
        if choice in MODEL_REGISTRY:
            return choice
    return "qwen7b"


def load_model(model_key="qwen7b"):
    """Load any supported model by key."""
    info = MODEL_REGISTRY[model_key]
    model_id = info["model_id"]
    model_type = info["type"]

    import torch

    if model_type == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"[*] Loading {model_id}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        # Limit pixels to prevent OOM on V100-32GB with high-res images
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "minicpm":
        from transformers import AutoModel, AutoTokenizer
        print(f"[*] Loading {model_id}...")
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto",
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        model.eval()
        print(f"[+] {info['name']} loaded.")
        return model, tokenizer, model_type

    elif model_type == "internvl":
        from transformers import AutoModel, AutoTokenizer
        print(f"[*] Loading {model_id}...")
        # Requires compat env (transformers 4.46) — 4.57+ breaks .generate()
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto",
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        model.eval()
        print(f"[+] {info['name']} loaded.")
        return model, tokenizer, model_type

    elif model_type == "florence":
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"[*] Loading {model_id}...")
        # Florence-2 custom code lacks _no_split_modules so device_map="auto"
        # is unsupported. Load to CPU then .cuda().
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        model.eval()
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print(f"[*] Loading {model_id}...")
        # Use .cuda() — device_map="auto" hangs on Qwen3-VL
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        print(f"[*] Model on GPU, loading processor...")
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=os.path.join(HF_CACHE, "hub"),
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "grounding_dino":
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"[*] Loading {model_id}...")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        print(f"[*] Loading {model_id}...")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "yolo_world":
        from ultralytics import YOLOWorld
        print(f"[*] Loading {model_id}...")
        model = YOLOWorld(model_id)
        model.set_classes(["weed", "plant", "grass"])
        print(f"[+] {info['name']} loaded.")
        return model, None, model_type

    elif model_type == "minicpm_v45":
        from transformers import AutoModel, AutoTokenizer
        print(f"[*] Loading {model_id}...")
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        model.eval()
        print(f"[+] {info['name']} loaded.")
        return model, tokenizer, model_type

    elif model_type == "molmo":
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"[*] Loading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "deepseek_vl":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[*] Loading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        model.eval()
        print(f"[+] {info['name']} loaded.")
        return model, tokenizer, model_type

    elif model_type == "owlv2":
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        print(f"[*] Loading {model_id}...")
        model = Owlv2ForObjectDetection.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        processor = Owlv2Processor.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    elif model_type == "omdet":
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        print(f"[*] Loading {model_id}...")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        ).cuda()
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=os.path.join(HF_CACHE, "hub"),
        )
        print(f"[+] {info['name']} loaded.")
        return model, processor, model_type

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Keep backward compatibility
def load_qwen_model(model_size="7b"):
    key = f"qwen{model_size}"
    model, processor, _ = load_model(key)
    return model, processor


WEED_PROMPT = """Detect all plants in this agricultural field image.
For each plant, classify it as "weed" or "crop".
If you can identify the species, include it.

Return ONLY valid JSON:
{"detections": [{"label": "weed", "species": "dandelion", "confidence": "high", "bbox": [x_min, y_min, x_max, y_max]}], "weed_severity": "low"}

The bbox should be in pixel coordinates of the original image."""


def run_inference(model, processor, image_path, model_type="qwen"):
    """Run inference with any supported model type."""
    dispatch = {
        "qwen": _infer_qwen,
        "qwen3": _infer_qwen,  # same chat interface as qwen2.5
        "minicpm": _infer_minicpm,
        "minicpm_v45": _infer_minicpm,  # same .chat() API
        "internvl": _infer_internvl,
        "florence": _infer_florence,
        "grounding_dino": _infer_grounding_dino,
        "paligemma": _infer_paligemma,
        "yolo_world": _infer_yolo_world,
        "molmo": _infer_molmo,
        "deepseek_vl": _infer_deepseek_vl,
        "owlv2": _infer_owlv2,
        "omdet": _infer_omdet,
    }
    if model_type not in dispatch:
        raise ValueError(f"Unknown model type: {model_type}")
    return dispatch[model_type](model, processor, image_path)


def _infer_qwen(model, processor, image_path):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
        {"type": "text", "text": WEED_PROMPT},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)

    generated = out[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return response


def _infer_minicpm(model, tokenizer, image_path):
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, WEED_PROMPT]}]

    with torch.no_grad():
        response = model.chat(
            image=None, msgs=msgs, tokenizer=tokenizer,
            max_new_tokens=2048, temperature=0.1,
        )
    return response


def _infer_internvl(model, tokenizer, image_path):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    image = Image.open(image_path)
    pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)

    gen_config = {"max_new_tokens": 2048, "temperature": 0.1, "do_sample": True}
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, WEED_PROMPT, gen_config)
    return response


def _infer_florence(model, processor, image_path):
    """Florence-2 inference using native object detection task."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    task = "<OD>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(model.device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result, task=task,
                                                image_size=(image.width, image.height))

    # Convert Florence-2 OD output to our standard JSON format
    od_result = parsed.get("<OD>", {})
    bboxes = od_result.get("bboxes", [])
    labels = od_result.get("labels", [])

    detections = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        # Normalize to [0, 1] using original image dims to avoid
        # ambiguity in convert_bbox_to_yolo's heuristic
        detections.append({
            "label": label.lower() if label else "object",
            "bbox": [x1 / image.width, y1 / image.height,
                     x2 / image.width, y2 / image.height],
            "confidence": "medium",
        })

    response = json.dumps({"detections": detections})
    return response


def _infer_grounding_dino(model, processor, image_path):
    """Grounding DINO: text-prompted open-set object detection."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text = "weed"  # single query works better than multi-class for GDINO

    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=0.25, text_threshold=0.25,
        target_sizes=[(image.height, image.width)]
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "label": label if isinstance(label, str) else "weed",
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": round(score.item(), 3),
        })
    return json.dumps({"detections": detections})


def _infer_owlv2(model, processor, image_path):
    """OWLv2: zero-shot object detection with text queries."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    # Single query to avoid duplicate detections of the same object
    texts = [["weed"]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(model.device)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1
    )[0]

    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        # Normalize to [0, 1] for consistent coordinate handling
        detections.append({
            "label": "weed",
            "bbox": [x1 / image.width, y1 / image.height,
                     x2 / image.width, y2 / image.height],
            "confidence": round(score.item(), 3),
        })
    return json.dumps({"detections": detections})


def _infer_omdet(model, processor, image_path):
    """OmDet-Turbo: fast zero-shot object detection."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text_labels = [["weed", "plant"]]
    inputs = processor(image, text=text_labels, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, text_labels,
        threshold=0.2,
        target_sizes=[image.size[::-1]]
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "label": label if isinstance(label, str) else "weed",
            "bbox": [x1 / image.width, y1 / image.height,
                     x2 / image.width, y2 / image.height],
            "confidence": round(score.item(), 3),
        })
    return json.dumps({"detections": detections})


def _infer_paligemma(model, processor, image_path):
    """PaliGemma 2: detection via <loc> coordinate tokens."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    prompt = "detect weed"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)

    decoded = processor.decode(output[0], skip_special_tokens=False)
    # PaliGemma outputs: <loc{Y1}><loc{X1}><loc{Y2}><loc{X2}> label
    loc_pattern = r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*(\w+)'
    matches = re.findall(loc_pattern, decoded)

    detections = []
    for y1, x1, y2, x2, label in matches:
        detections.append({
            "label": label.lower(),
            "bbox": [
                round(int(x1) / 1024 * image.width, 1),
                round(int(y1) / 1024 * image.height, 1),
                round(int(x2) / 1024 * image.width, 1),
                round(int(y2) / 1024 * image.height, 1),
            ],
            "confidence": "medium",
        })
    return json.dumps({"detections": detections})


def _infer_yolo_world(model, _processor, image_path):
    """YOLO-World: open-vocabulary YOLO detection."""
    results = model.predict(image_path, conf=0.25, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "label": model.names.get(cls, "object"),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(conf, 3),
            })
    return json.dumps({"detections": detections})


def _infer_molmo(model, processor, image_path):
    """Molmo: Allen AI model with pixel coordinate output."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor.process(images=[image], text=WEED_PROMPT)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()
              if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        output = model.generate_from_batch(inputs, max_new_tokens=2048,
                                           tokenizer=processor.tokenizer)
    response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def _infer_deepseek_vl(model, tokenizer, image_path):
    """DeepSeek-VL2: MoE VLM with <ref>/<det> grounding tokens."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    prompt = (
        "Detect all weeds in this image. For each detection, output bounding box "
        "coordinates as JSON: {\"detections\": [{\"label\": \"weed\", "
        "\"bbox\": [x_min, y_min, x_max, y_max]}]}"
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    # DeepSeek-VL2 uses a chat interface similar to other VLMs
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                           add_generation_prompt=True)
    if isinstance(inputs, dict):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = inputs.to(model.device)

    with torch.no_grad():
        if isinstance(inputs, dict):
            out = model.generate(**inputs, max_new_tokens=2048)
        else:
            out = model.generate(inputs, max_new_tokens=2048)

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    return response


def extract_json(text):
    """Extract JSON from model response."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{[\s\S]*\}"]:
        match = re.search(pattern, str(text), re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(candidate)
            except (json.JSONDecodeError, IndexError):
                continue
    return None


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """Convert [x1, y1, x2, y2] to YOLO format [cx, cy, w, h] normalized [0,1].

    Handles multiple coordinate systems:
      - [0, 1]:    Already normalized fractions
      - [0, 100]:  Percentage coordinates
      - [0, 1000]: Qwen2.5-VL / grounding-model normalized coords
      - Otherwise:  Absolute pixel coordinates → normalize by img dims
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    max_val = max(x1, y1, x2, y2)

    if max_val <= 1.0:
        # Already [0, 1] normalized
        pass
    elif max_val <= 100 and all(0 <= v <= 100 for v in [x1, y1, x2, y2]):
        # Percentage [0, 100]
        x1, y1, x2, y2 = x1 / 100, y1 / 100, x2 / 100, y2 / 100
    elif max_val <= 1000 and max(img_w, img_h) > 1000:
        # Qwen2.5-VL style [0, 1000] normalized coords
        # (model outputs in [0,999] range, image is much larger)
        x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    else:
        # Absolute pixel coordinates
        x1, y1, x2, y2 = x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h

    # Clamp to [0, 1]
    x1, y1 = max(0, min(x1, 1)), max(0, min(y1, 1))
    x2, y2 = max(0, min(x2, 1)), max(0, min(y2, 1))

    # Convert to YOLO center format
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return cx, cy, w, h


def detect_images(source_dir, model_size="7b", output_dir=None, model_key=None):
    """Run LLM detection on all images in source_dir."""
    from PIL import Image
    import cv2

    # Resolve model_key from model_size for backward compatibility
    if model_key is None:
        model_key = f"qwen{model_size}"
    if model_key not in MODEL_REGISTRY:
        print(f"[!] Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
        return None

    model_info = MODEL_REGISTRY[model_key]
    model_short_name = model_info["name"].lower().replace(" ", "-").replace(".", "")

    if output_dir is None:
        output_dir = os.path.join(LABELED_DIR, model_short_name)

    # Find all images (check common dataset structures)
    image_files = []
    for search_dir in [source_dir,
                       os.path.join(source_dir, "images"),
                       os.path.join(source_dir, "train", "images"),
                       os.path.join(source_dir, "valid", "images"),
                       os.path.join(source_dir, "test", "images")]:
        if os.path.isdir(search_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(glob.glob(os.path.join(search_dir, ext)))

    image_files = sorted(set(image_files))
    if not image_files:
        print(f"[!] No images found in {source_dir}")
        return None

    print(f"\n{'='*50}")
    print(f"LLM WEED DETECTION")
    print(f"{'='*50}")
    print(f"  Source:  {source_dir}")
    print(f"  Images:  {len(image_files)}")
    print(f"  Model:   {model_info['name']} ({model_info['model_id']})")
    print(f"  Output:  {output_dir}")

    # Create output dirs
    det_images = os.path.join(output_dir, "detected", "images")
    det_labels = os.path.join(output_dir, "detected", "labels")
    nodet_images = os.path.join(output_dir, "no_detection", "images")
    nodet_labels = os.path.join(output_dir, "no_detection", "labels")
    vis_dir = os.path.join(output_dir, "visualized")
    for d in [det_images, det_labels, nodet_images, nodet_labels, vis_dir]:
        os.makedirs(d, exist_ok=True)

    # Load model
    model, processor, model_type = load_model(model_key)

    detected_count = 0
    nodet_count = 0
    all_results = []

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(img_name)[0]

        print(f"\n[{i+1}/{len(image_files)}] {img_name}")

        try:
            # Get image dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [!] Cannot read image, skipping")
                continue
            img_h, img_w = img.shape[:2]

            # Run LLM
            start = time.time()
            response = run_inference(model, processor, img_path, model_type)
            elapsed = time.time() - start
            print(f"  LLM inference: {elapsed:.1f}s")

            # Parse response
            parsed = extract_json(response)
            detections = []
            if parsed and "detections" in parsed:
                detections = parsed["detections"]

            result_entry = {
                "image": img_name,
                "raw_response": response[:2000],
                "num_detections": len(detections),
                "time_s": elapsed,
            }

            if detections:
                detected_count += 1
                # Save image
                shutil.copy2(img_path, os.path.join(det_images, img_name))

                # Convert to YOLO format labels
                yolo_lines = []
                for det in detections:
                    bbox = det.get("bbox") or det.get("bbox_2d")
                    if not bbox or len(bbox) != 4:
                        continue

                    # Class 0 = weed (matching your YOLO model)
                    label = det.get("label", "weed").lower()
                    class_id = 0 if "weed" in label else 1

                    cx, cy, w, h = convert_bbox_to_yolo(bbox, img_w, img_h)

                    # Sanity check
                    if w > 0 and h > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
                        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                # Write label file
                label_path = os.path.join(det_labels, f"{name_no_ext}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines) + '\n')

                print(f"  Found {len(detections)} plants, {len(yolo_lines)} valid bboxes")

                # Draw visualization
                vis_img = img.copy()
                for det in detections:
                    bbox = det.get("bbox") or det.get("bbox_2d")
                    if not bbox or len(bbox) != 4:
                        continue
                    bx1, by1, bx2, by2 = [float(v) for v in bbox]
                    max_bval = max(bx1, by1, bx2, by2)
                    # Convert to pixel coords for drawing
                    if max_bval <= 1.0:
                        x1, y1 = int(bx1 * img_w), int(by1 * img_h)
                        x2, y2 = int(bx2 * img_w), int(by2 * img_h)
                    elif max_bval <= 100 and all(0 <= v <= 100 for v in [bx1, by1, bx2, by2]):
                        x1, y1 = int(bx1 / 100 * img_w), int(by1 / 100 * img_h)
                        x2, y2 = int(bx2 / 100 * img_w), int(by2 / 100 * img_h)
                    elif max_bval <= 1000 and max(img_w, img_h) > 1000:
                        x1, y1 = int(bx1 / 1000 * img_w), int(by1 / 1000 * img_h)
                        x2, y2 = int(bx2 / 1000 * img_w), int(by2 / 1000 * img_h)
                    else:
                        x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)

                    label = det.get("label", "?")
                    species = det.get("species", "")
                    color = (0, 0, 255) if "weed" in label.lower() else (0, 255, 0)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {species}" if species else label
                    cv2.putText(vis_img, text, (x1, max(y1-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                vis_path = os.path.join(vis_dir, img_name)
                cv2.imwrite(vis_path, vis_img)

                result_entry["detections"] = detections
                result_entry["yolo_labels"] = yolo_lines
            else:
                nodet_count += 1
                shutil.copy2(img_path, os.path.join(nodet_images, img_name))
                label_path = os.path.join(nodet_labels, f"{name_no_ext}.txt")
                with open(label_path, 'w') as f:
                    f.write("")
                print(f"  No detections")

            all_results.append(result_entry)

        except Exception as e:
            print(f"  [!] Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save full results JSON
    results_path = os.path.join(output_dir, "detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"DETECTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Total processed: {detected_count + nodet_count}")
    print(f"  With detections: {detected_count}")
    print(f"  No detections:   {nodet_count}")
    print(f"  Results JSON:    {results_path}")
    print(f"  Visualizations:  {vis_dir}")
    print(f"  YOLO labels:     {det_labels}")

    return output_dir


# ============================================================
# 3. Upload to Roboflow
# ============================================================
def upload_to_roboflow(dataset_path, project_name):
    """Upload labeled dataset back to Roboflow."""
    from roboflow import Roboflow

    key = ensure_api_key()

    print(f"\n{'='*50}")
    print(f"UPLOADING TO ROBOFLOW")
    print(f"{'='*50}")

    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
        print(f"[!] Expected structure: {dataset_path}/images/ and {dataset_path}/labels/")
        return False

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))

    label_files = glob.glob(os.path.join(labels_path, "*.txt"))

    print(f"  Project:  {project_name}")
    print(f"  Images:   {len(image_files)}")
    print(f"  Labels:   {len(label_files)}")

    try:
        rf = Roboflow(api_key=key)
        workspace = rf.workspace(WORKSPACE_ID)

        workspace.upload_dataset(
            dataset_path,
            project_name,
            num_workers=10,
            project_type="object-detection",
            batch_name="llm-weed-detection",
            num_retries=3,
        )

        print(f"\n[+] Upload successful!")
        print(f"    View at: https://app.roboflow.com/{WORKSPACE_ID}/{project_name}")
        return True
    except Exception as e:
        print(f"[!] Upload failed: {e}")
        print("\nTrying alternative upload method (image by image)...")
        return upload_images_individually(images_path, labels_path, project_name)


def upload_images_individually(images_path, labels_path, project_name):
    """Fallback: upload images one by one via API."""
    from roboflow import Roboflow

    key = ensure_api_key()
    try:
        rf = Roboflow(api_key=key)
        project = rf.workspace(WORKSPACE_ID).project(project_name)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(images_path, ext)))

        uploaded = 0
        failed = 0
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_path, f"{name_no_ext}.txt")

            try:
                if os.path.exists(label_path):
                    project.upload(img_path, annotation_path=label_path)
                else:
                    project.upload(img_path)
                uploaded += 1
                if uploaded % 10 == 0:
                    print(f"  Uploaded {uploaded}/{len(image_files)}")
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  [!] Failed to upload {img_name}: {e}")

        print(f"\n[+] Uploaded {uploaded}/{len(image_files)} images ({failed} failed)")
        return uploaded > 0
    except Exception as e:
        print(f"[!] Individual upload also failed: {e}")
        return False


# ============================================================
# Interactive Menu
# ============================================================
def interactive_menu():
    print("\n" + "=" * 60)
    print("  Roboflow <-> LLM Weed Detection Bridge")
    print("=" * 60)
    print()
    print("1. Download dataset from Roboflow")
    print("2. Run LLM weed detection on images")
    print("3. Upload LLM-labeled results to Roboflow")
    print("4. Full pipeline (download -> detect -> upload)")
    print("5. Exit")
    print()
    return get_input("Select option (1-5)")


def pick_project(action_label="download from"):
    """Let user pick a project from the list."""
    project_id, _ = list_roboflow_projects()
    if not project_id:
        project_id = get_input("Enter project ID manually")
    version = get_input("Version", "1")
    return project_id, version


def pick_model():
    """Let user select a model from the registry."""
    model_key = select_model_interactive()
    model_info = MODEL_REGISTRY[model_key]
    print(f"\n  Model: {model_info['name']} ({model_info['model_id']})")
    return model_key


def run_interactive():
    while True:
        choice = interactive_menu()

        if choice == "1":
            project_id, version = pick_project("download from")
            fmt = get_input("Format (yolov8/coco/voc)", "yolov8")
            download_from_roboflow(project_id, version, fmt)
            input("\nPress Enter to continue...")

        elif choice == "2":
            # List downloaded datasets
            if os.path.isdir(DOWNLOAD_DIR):
                downloaded = [d for d in os.listdir(DOWNLOAD_DIR)
                              if os.path.isdir(os.path.join(DOWNLOAD_DIR, d))]
                if downloaded:
                    print("\n  Downloaded datasets:")
                    for i, d in enumerate(downloaded):
                        print(f"    {i+1}. {d}")
                    pick = get_input(f"Select (1-{len(downloaded)}) or type path", "1")
                    try:
                        idx = int(pick) - 1
                        source = os.path.join(DOWNLOAD_DIR, downloaded[idx])
                    except (ValueError, IndexError):
                        source = pick
                else:
                    source = get_input("Image source directory")
            else:
                source = get_input("Image source directory")

            # Model selection
            model_key = pick_model()
            detect_images(source, model_key=model_key)
            input("\nPress Enter to continue...")

        elif choice == "3":
            # List available labeled results
            if os.path.isdir(LABELED_DIR):
                labeled = [d for d in os.listdir(LABELED_DIR)
                           if os.path.isdir(os.path.join(LABELED_DIR, d))]
                if labeled:
                    print("\n  Labeled results:")
                    for i, d in enumerate(labeled):
                        det_path = os.path.join(LABELED_DIR, d, "detected", "images")
                        count = len(glob.glob(os.path.join(det_path, "*"))) if os.path.isdir(det_path) else 0
                        print(f"    {i+1}. {d}  ({count} images)")
                    pick = get_input(f"Select (1-{len(labeled)}) or type path", "1")
                    try:
                        idx = int(pick) - 1
                        source = os.path.join(LABELED_DIR, labeled[idx], "detected")
                    except (ValueError, IndexError):
                        source = pick
                else:
                    source = get_input("Labeled dataset path")
            else:
                source = get_input("Labeled dataset path")

            print("\nSelect upload target project:")
            project_id, _ = list_roboflow_projects()
            if not project_id:
                project_id = get_input("Enter project name to upload to")
            upload_to_roboflow(source, project_id)
            input("\nPress Enter to continue...")

        elif choice == "4":
            print("\n--- Select SOURCE project to download from ---")
            project_id, version = pick_project("download from")

            # Model selection
            model_key = pick_model()
            model_name = MODEL_REGISTRY[model_key]["name"].lower().replace(" ", "-").replace(".", "")

            # Auto-generate upload project name with model info
            default_upload = f"{project_id}-{model_name}"
            upload_project = get_input("Upload project name", default_upload)

            # Step 1: Download
            dl_path = download_from_roboflow(project_id, version)
            if not dl_path:
                print("[!] Download failed, aborting")
                continue

            # Step 2: Detect
            out_path = detect_images(dl_path, model_key=model_key)
            if not out_path:
                print("[!] Detection failed, aborting")
                continue

            # Step 3: Upload
            detected_path = os.path.join(out_path, "detected")
            upload_to_roboflow(detected_path, upload_project)

            print(f"\n{'='*50}")
            print("FULL PIPELINE COMPLETE")
            print(f"{'='*50}")
            print(f"  Downloaded from: {project_id}")
            print(f"  Model used:      {MODEL_REGISTRY[model_key]['name']}")
            print(f"  LLM labeled:     {out_path}")
            print(f"  Uploaded to:      {upload_project}")
            print(f"  Visualizations:   {os.path.join(out_path, 'visualized')}")
            print(f"\n  Check results at: https://app.roboflow.com/{WORKSPACE_ID}/{upload_project}")
            input("\nPress Enter to continue...")

        elif choice == "5":
            print("Bye!")
            break


def _run_evaluation(dataset_dir, output_dir, model_key):
    """Run evaluation comparing detections against ground truth."""
    try:
        from evaluate import evaluate_dataset, load_predictions_from_json, load_yolo_labels, print_evaluation

        # Find ground truth labels
        gt_dir = None
        for split in ["test", "valid"]:
            candidate = os.path.join(dataset_dir, split, "labels")
            if os.path.isdir(candidate):
                gt_dir = candidate
                break

        if not gt_dir:
            print("[!] No ground truth labels found for evaluation")
            return

        result_json = os.path.join(output_dir, "detection_results.json")
        if not os.path.exists(result_json):
            print("[!] No detection results JSON found for evaluation")
            return

        gt = load_yolo_labels(gt_dir)
        predictions = load_predictions_from_json(result_json)
        summary = evaluate_dataset(gt, predictions, iou_thresholds=[0.25, 0.5, 0.75])
        print_evaluation(summary, model_name=MODEL_REGISTRY.get(model_key, {}).get("name", model_key))

        # Save evaluation results
        eval_path = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Evaluation saved to {eval_path}")

    except ImportError:
        print("[!] evaluate module not found. Run: pip install numpy")
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Roboflow <-> LLM Weed Detection Bridge")
    parser.add_argument("--download", action="store_true", help="Download from Roboflow")
    parser.add_argument("--detect", action="store_true", help="Run LLM detection")
    parser.add_argument("--upload", action="store_true", help="Upload to Roboflow")
    parser.add_argument("--all", action="store_true", help="Full pipeline")
    parser.add_argument("--project", type=str, default="weed1-rmxbe")
    parser.add_argument("--version", type=str, default="1")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--upload-project", type=str, default=None)
    parser.add_argument("--model-key", type=str, default="qwen7b",
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Model to use: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--model-size", type=str, default=None, help="(deprecated) Use --model-key instead")
    parser.add_argument("--format", type=str, default="yolov8")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after detection (compare against ground truth)")
    args = parser.parse_args()

    # Handle backward compat: --model-size -> --model-key
    model_key = args.model_key
    if args.model_size:
        model_key = f"qwen{args.model_size}"

    # Non-interactive mode
    if args.download or args.detect or args.upload or args.all:
        if args.all:
            model_name = MODEL_REGISTRY[model_key]["name"].lower().replace(" ", "-").replace(".", "")
            dl_path = download_from_roboflow(args.project, args.version, args.format)
            if dl_path:
                out_path = detect_images(dl_path, model_key=model_key)
                if out_path:
                    up_proj = args.upload_project or f"{args.project}-{model_name}"
                    upload_to_roboflow(os.path.join(out_path, "detected"), up_proj)
                    if args.evaluate:
                        _run_evaluation(dl_path, out_path, model_key)
        elif args.download:
            download_from_roboflow(args.project, args.version, args.format)
        elif args.detect:
            source = args.source or os.path.join(DOWNLOAD_DIR, args.project)
            out_path = detect_images(source, model_key=model_key)
            if args.evaluate and out_path:
                _run_evaluation(source, out_path, model_key)
        elif args.upload:
            source = args.source or os.path.join(LABELED_DIR, "detected")
            up_proj = args.upload_project or f"{args.project}-llm-labeled"
            upload_to_roboflow(source, up_proj)
    else:
        # Interactive mode
        run_interactive()


if __name__ == "__main__":
    main()
