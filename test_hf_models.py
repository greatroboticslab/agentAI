"""
Test weed detection using HuggingFace vision models (transformers backend).
Supports: Qwen2.5-VL, MiniCPM-V, Florence-2, InternVL2, BLIP3o

Usage:
    python test_hf_models.py --image images/weed1.jpg --model qwen7b
    python test_hf_models.py --image images/weed1.jpg --model all
    python test_hf_models.py --image-dir images/ --model qwen7b
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from config import (
    HF_CACHE,
    HF_MODELS_LIST,
    IMAGE_DIR,
    RESULT_DIR,
    WEED_DETECTION_PROMPT,
    WEED_DETECTION_PROMPT_SIMPLE,
)

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")


# ============================================================
# Model Loaders
# ============================================================

def load_qwen25vl(model_name):
    """Load Qwen2.5-VL model."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    print(f"[*] Loading {model_name}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    print(f"[+] {model_name} loaded.")
    return model, processor


def infer_qwen25vl(model, processor, image_path, prompt):
    """Run inference with Qwen2.5-VL."""
    from qwen_vl_utils import process_vision_info
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
        )

    # Decode only the generated tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def load_minicpmv(model_name="openbmb/MiniCPM-V-2_6"):
    """Load MiniCPM-V model."""
    from transformers import AutoModel, AutoTokenizer
    import torch

    print(f"[*] Loading {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    model.eval()
    print(f"[+] {model_name} loaded.")
    return model, tokenizer


def infer_minicpmv(model, tokenizer, image_path, prompt):
    """Run inference with MiniCPM-V."""
    import torch

    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, prompt]}]

    with torch.no_grad():
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.1,
        )
    return response


def load_florence2(model_name="microsoft/Florence-2-large"):
    """Load Florence-2 model - specialized for grounding."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    print(f"[*] Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    print(f"[+] {model_name} loaded.")
    return model, processor


def infer_florence2(model, processor, image_path, prompt=None):
    """
    Run inference with Florence-2.
    Florence-2 uses task-specific prompts, not free-form text.
    We'll use multiple tasks for comprehensive detection.
    """
    import torch

    image = Image.open(image_path).convert("RGB")
    results = {}

    # Task 1: Open vocabulary detection - find weeds
    tasks = [
        ("<OPEN_VOCABULARY_DETECTION>", "weed, crop, grass, plant, dandelion, thistle, clover"),
        ("<CAPTION_TO_PHRASE_GROUNDING>", "weeds growing in the agricultural field"),
        ("<DETAILED_CAPTION>", None),
        ("<OD>", None),  # General object detection
    ]

    for task_prompt, text_input in tasks:
        try:
            if text_input:
                full_prompt = task_prompt + text_input
            else:
                full_prompt = task_prompt

            inputs = processor(text=full_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=3,
                )
            response = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                response,
                task=task_prompt,
                image_size=(image.width, image.height),
            )
            results[task_prompt] = parsed
        except Exception as e:
            results[task_prompt] = {"error": str(e)}

    # Convert Florence results to our standard format
    return format_florence_results(results, image.width, image.height)


def format_florence_results(results, img_w, img_h):
    """Convert Florence-2 output to standard weed detection JSON format."""
    detections = []

    # Process open vocabulary detection results
    for task_key in ["<OPEN_VOCABULARY_DETECTION>", "<OD>", "<CAPTION_TO_PHRASE_GROUNDING>"]:
        if task_key not in results:
            continue
        task_result = results[task_key]
        if isinstance(task_result, dict) and "error" not in task_result:
            for key, value in task_result.items():
                if "bboxes" in str(value) or isinstance(value, dict):
                    bbox_data = value if isinstance(value, dict) else task_result
                    bboxes = bbox_data.get("bboxes", [])
                    labels = bbox_data.get("labels", [])
                    for i, bbox in enumerate(bboxes):
                        label = labels[i] if i < len(labels) else "plant"
                        # Convert pixel coords to percentage
                        x1 = round(bbox[0] / img_w * 100, 1)
                        y1 = round(bbox[1] / img_h * 100, 1)
                        x2 = round(bbox[2] / img_w * 100, 1)
                        y2 = round(bbox[3] / img_h * 100, 1)
                        is_weed = any(w in label.lower() for w in ["weed", "dandelion", "thistle", "clover", "grass"])
                        detections.append({
                            "label": "weed" if is_weed else label,
                            "species": label,
                            "confidence": "medium",
                            "bbox": [x1, y1, x2, y2],
                            "source_task": task_key,
                        })

    # Get scene description
    caption = ""
    if "<DETAILED_CAPTION>" in results:
        cap_result = results["<DETAILED_CAPTION>"]
        if isinstance(cap_result, dict):
            caption = cap_result.get("<DETAILED_CAPTION>", str(cap_result))
        else:
            caption = str(cap_result)

    output = {
        "detections": detections,
        "scene_description": caption,
        "weed_severity": "unknown",
        "crop_type": "unknown",
        "raw_florence_results": {k: str(v)[:500] for k, v in results.items()},
    }
    return json.dumps(output, indent=2)


def load_internvl2(model_name="OpenGVLab/InternVL2-8B"):
    """Load InternVL2 model."""
    from transformers import AutoModel, AutoTokenizer
    import torch

    print(f"[*] Loading {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.path.join(HF_CACHE, "hub"),
    )
    model.eval()
    print(f"[+] {model_name} loaded.")
    return model, tokenizer


def infer_internvl2(model, tokenizer, image_path, prompt):
    """Run inference with InternVL2."""
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    image = Image.open(image_path)
    transform = build_transform(448)
    pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)

    generation_config = {"max_new_tokens": 2048, "temperature": 0.1, "do_sample": True}
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response


# ============================================================
# Model Registry
# ============================================================

MODEL_SHORTCUTS = {
    "qwen7b": {
        "full_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "loader": load_qwen25vl,
        "inferencer": infer_qwen25vl,
    },
    "qwen3b": {
        "full_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "loader": load_qwen25vl,
        "inferencer": infer_qwen25vl,
    },
    "minicpm": {
        "full_name": "openbmb/MiniCPM-V-2_6",
        "loader": load_minicpmv,
        "inferencer": infer_minicpmv,
    },
    "florence2": {
        "full_name": "microsoft/Florence-2-large",
        "loader": load_florence2,
        "inferencer": infer_florence2,
    },
    "internvl2": {
        "full_name": "OpenGVLab/InternVL2-8B",
        "loader": load_internvl2,
        "inferencer": infer_internvl2,
    },
}


def extract_json(text):
    """Try to extract JSON from model response text."""
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text), re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(candidate)
            except (json.JSONDecodeError, IndexError):
                continue
    return None


def test_model(model_key, image_paths, use_simple_prompt=False):
    """Test a single HF model on given images."""
    if model_key not in MODEL_SHORTCUTS:
        print(f"[!] Unknown model: {model_key}")
        print(f"    Available: {list(MODEL_SHORTCUTS.keys())}")
        sys.exit(1)

    info = MODEL_SHORTCUTS[model_key]
    prompt = WEED_DETECTION_PROMPT_SIMPLE if use_simple_prompt else WEED_DETECTION_PROMPT

    # Load model
    loader = info["loader"]
    if model_key == "florence2":
        model, processor = loader(info["full_name"])
    else:
        model, processor = loader(info["full_name"])

    results = []
    for image_path in image_paths:
        img_name = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"Model: {info['full_name']} | Image: {img_name}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            inferencer = info["inferencer"]
            response = inferencer(model, processor, image_path, prompt)
            elapsed = time.time() - start_time

            print(f"\n--- Raw Response ({elapsed:.1f}s) ---")
            resp_str = str(response)
            print(resp_str[:3000])
            if len(resp_str) > 3000:
                print(f"... (truncated, total {len(resp_str)} chars)")

            # Parse JSON
            parsed = extract_json(response)
            result = {
                "success": True,
                "model": info["full_name"],
                "model_key": model_key,
                "image": img_name,
                "raw_response": resp_str[:5000],
                "total_duration_s": elapsed,
            }

            if parsed:
                print(f"\n--- Parsed JSON ---")
                print(json.dumps(parsed, indent=2)[:2000])
                result["parsed_json"] = parsed
                result["json_valid"] = True
                detections = parsed.get("detections", [])
                weeds = [d for d in detections if "weed" in str(d.get("label", "")).lower()]
                result["num_detections"] = len(detections)
                result["num_weeds"] = len(weeds)
                result["has_bbox"] = any("bbox" in d for d in detections)
                print(f"\nDetections: {len(detections)} total, {len(weeds)} weeds")
                print(f"Has bounding boxes: {result['has_bbox']}")
            else:
                print("\n[!] Could not parse JSON from response")
                result["json_valid"] = False
                # Retry with simple prompt
                if not use_simple_prompt:
                    print("[*] Retrying with simpler prompt...")
                    start2 = time.time()
                    response2 = inferencer(model, processor, image_path, WEED_DETECTION_PROMPT_SIMPLE)
                    elapsed2 = time.time() - start2
                    parsed2 = extract_json(response2)
                    if parsed2:
                        result["parsed_json"] = parsed2
                        result["json_valid"] = True
                        result["used_simple_prompt"] = True
                        result["total_duration_s"] = elapsed + elapsed2

            results.append(result)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[!] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "success": False,
                "model": info["full_name"],
                "model_key": model_key,
                "image": img_name,
                "error": str(e),
                "total_duration_s": elapsed,
            })

    return results


def print_summary(results):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'Image':<15} {'JSON?':<6} {'Weeds':<6} {'BBox?':<6} {'Time(s)':<8}")
    print(f"{'-'*80}")
    for r in results:
        if r.get("success"):
            print(
                f"{r.get('model_key','?'):<35} "
                f"{r.get('image','?'):<15} "
                f"{'YES' if r.get('json_valid') else 'NO':<6} "
                f"{r.get('num_weeds', '?'):<6} "
                f"{'YES' if r.get('has_bbox') else 'NO':<6} "
                f"{r.get('total_duration_s', 0):<8.1f}"
            )
        else:
            print(f"{r.get('model_key','?'):<35} {r.get('image','?'):<15} FAILED: {r.get('error','')[:30]}")


def main():
    parser = argparse.ArgumentParser(description="Test weed detection with HuggingFace vision models")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen7b",
        help=f"Model shortcut or 'all'. Options: {list(MODEL_SHORTCUTS.keys())}",
    )
    parser.add_argument("--simple-prompt", action="store_true", help="Use simpler prompt")
    args = parser.parse_args()

    # Gather images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(Path(args.image_dir).glob(ext))
        image_paths = [str(p) for p in sorted(image_paths)]
    else:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(Path(IMAGE_DIR).glob(ext))
        image_paths = [str(p) for p in sorted(image_paths)]

    if not image_paths:
        print("[!] No images found. Put weed photos in images/ directory or use --image flag.")
        sys.exit(1)

    print(f"[*] Found {len(image_paths)} image(s)")

    # Run tests
    all_results = []
    if args.model == "all":
        model_keys = list(MODEL_SHORTCUTS.keys())
    else:
        model_keys = [args.model]

    for mk in model_keys:
        results = test_model(mk, image_paths, args.simple_prompt)
        all_results.extend(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULT_DIR, f"hf_benchmark_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[+] Results saved to {output_file}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
