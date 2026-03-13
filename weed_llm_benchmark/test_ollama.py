"""
Test weed detection using Ollama vision models.
Usage:
    python test_ollama.py --image images/weed1.jpg
    python test_ollama.py --image images/weed1.jpg --model llava:13b
    python test_ollama.py --image-dir images/ --model all
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from config import (
    IMAGE_DIR,
    OLLAMA_BIN,
    OLLAMA_MODELS,
    OLLAMA_MODELS_LIST,
    QWEN_GROUNDING_PROMPT,
    RESULT_DIR,
    WEED_DETECTION_PROMPT,
    WEED_DETECTION_PROMPT_SIMPLE,
)

OLLAMA_API = "http://localhost:11434"


def check_ollama_running():
    """Check if ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def start_ollama_server():
    """Start ollama server in background."""
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = OLLAMA_MODELS
    env["OLLAMA_HOST"] = "0.0.0.0:11434"
    print("[*] Starting ollama server...")
    proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to start
    for i in range(30):
        if check_ollama_running():
            print("[+] Ollama server is running.")
            return proc
        time.sleep(1)
    print("[!] Failed to start ollama server.")
    sys.exit(1)


def pull_model(model_name):
    """Pull a model if not already available."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=10)
        available = [m["name"] for m in r.json().get("models", [])]
        # Check if model is already pulled (handle tag variants)
        for a in available:
            if model_name in a or a in model_name:
                print(f"[+] Model {model_name} already available as {a}")
                return True
    except Exception:
        pass

    print(f"[*] Pulling model {model_name}... (this may take a while)")
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = OLLAMA_MODELS
    result = subprocess.run(
        [OLLAMA_BIN, "pull", model_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout for large models
    )
    if result.returncode != 0:
        print(f"[!] Failed to pull {model_name}: {result.stderr}")
        return False
    print(f"[+] Successfully pulled {model_name}")
    return True


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(model_name, image_path, prompt=None, timeout=120):
    """Send image + prompt to ollama and get response."""
    if prompt is None:
        prompt = WEED_DETECTION_PROMPT

    img_b64 = encode_image(image_path)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temp for more deterministic output
            "num_predict": 2048,
        },
    }

    start_time = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_API}/api/generate",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - start_time
        if r.status_code == 200:
            result = r.json()
            return {
                "success": True,
                "response": result.get("response", ""),
                "total_duration_s": elapsed,
                "model": model_name,
                "eval_count": result.get("eval_count", 0),
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {r.status_code}: {r.text}",
                "model": model_name,
            }
    except requests.Timeout:
        return {
            "success": False,
            "error": f"Timeout after {timeout}s",
            "model": model_name,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model_name,
        }


def extract_json(text):
    """Try to extract JSON from model response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code fences
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def get_prompt_for_model(model_name, use_simple=False):
    """Select the best prompt for each model."""
    # Qwen2.5-VL models use native grounding prompt
    if "qwen" in model_name.lower():
        return QWEN_GROUNDING_PROMPT
    if use_simple:
        return WEED_DETECTION_PROMPT_SIMPLE
    return WEED_DETECTION_PROMPT


def test_single_image(model_name, image_path, use_simple_prompt=False):
    """Test a single model on a single image."""
    prompt = get_prompt_for_model(model_name, use_simple_prompt)
    img_name = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Image: {img_name}")
    print(f"{'='*60}")

    result = query_ollama(model_name, image_path, prompt)

    if not result["success"]:
        print(f"[!] Error: {result['error']}")
        return result

    raw_response = result["response"]
    print(f"\n--- Raw Response ({result['total_duration_s']:.1f}s) ---")
    print(raw_response[:2000])  # Truncate very long responses
    if len(raw_response) > 2000:
        print(f"... (truncated, total {len(raw_response)} chars)")

    # Try to parse JSON
    parsed = extract_json(raw_response)
    if parsed:
        print(f"\n--- Parsed JSON ---")
        print(json.dumps(parsed, indent=2))
        result["parsed_json"] = parsed
        result["json_valid"] = True

        # Count detections
        detections = parsed.get("detections", [])
        weeds = [d for d in detections if "weed" in d.get("label", "").lower()]
        crops = [d for d in detections if "crop" in d.get("label", "").lower()]
        print(f"\nDetections: {len(detections)} total, {len(weeds)} weeds, {len(crops)} crops")

        # Check if bboxes are present (bbox or bbox_2d for Qwen format)
        has_bbox = any("bbox" in d or "bbox_2d" in d for d in detections)
        # Normalize bbox_2d -> bbox for consistency
        for d in detections:
            if "bbox_2d" in d and "bbox" not in d:
                d["bbox"] = d["bbox_2d"]
        print(f"Has bounding boxes: {has_bbox}")
        result["num_detections"] = len(detections)
        result["num_weeds"] = len(weeds)
        result["has_bbox"] = has_bbox
    else:
        print("\n[!] Could not parse JSON from response")
        result["json_valid"] = False
        result["parsed_json"] = None

        # Retry with simple prompt if complex failed
        if not use_simple_prompt:
            print("[*] Retrying with simpler prompt...")
            return test_single_image(model_name, image_path, use_simple_prompt=True)

    return result


def run_benchmark(models, image_paths):
    """Run all models on all images and collect results."""
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for image_path in image_paths:
        for model_info in models:
            model_name = model_info["name"]

            # Pull model if needed
            if not pull_model(model_name):
                print(f"[!] Skipping {model_name} - could not pull")
                continue

            result = test_single_image(model_name, image_path)
            result["image"] = os.path.basename(image_path)
            result["tier"] = model_info.get("tier", 0)
            result["description"] = model_info.get("description", "")
            all_results.append(result)

    # Save results
    output_file = os.path.join(RESULT_DIR, f"ollama_benchmark_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[+] Results saved to {output_file}")

    # Print summary
    print_summary(all_results)
    return all_results


def print_summary(results):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Image':<15} {'JSON?':<6} {'Weeds':<6} {'BBox?':<6} {'Time(s)':<8}")
    print(f"{'-'*80}")
    for r in results:
        if r.get("success"):
            print(
                f"{r['model']:<30} "
                f"{r.get('image','?'):<15} "
                f"{'YES' if r.get('json_valid') else 'NO':<6} "
                f"{r.get('num_weeds', '?'):<6} "
                f"{'YES' if r.get('has_bbox') else 'NO':<6} "
                f"{r.get('total_duration_s', 0):<8.1f}"
            )
        else:
            print(f"{r['model']:<30} {r.get('image','?'):<15} FAILED: {r.get('error','')[:30]}")


def main():
    parser = argparse.ArgumentParser(description="Test weed detection with Ollama vision models")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images to test")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name or 'all' to test all models",
    )
    parser.add_argument("--simple-prompt", action="store_true", help="Use simpler prompt")
    parser.add_argument("--no-pull", action="store_true", help="Don't pull models, only use available ones")
    args = parser.parse_args()

    # Determine images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        img_dir = args.image_dir
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(Path(img_dir).glob(ext))
        image_paths = [str(p) for p in sorted(image_paths)]
    else:
        # Default: look in images/ directory
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(Path(IMAGE_DIR).glob(ext))
        image_paths = [str(p) for p in sorted(image_paths)]

    if not image_paths:
        print("[!] No images found. Put weed photos in images/ directory or use --image flag.")
        print(f"    Image directory: {IMAGE_DIR}")
        sys.exit(1)

    print(f"[*] Found {len(image_paths)} image(s) to test")

    # Start ollama if needed
    server_proc = None
    if not check_ollama_running():
        server_proc = start_ollama_server()

    # Determine models
    if args.model == "all":
        models = OLLAMA_MODELS_LIST
    else:
        models = [{"name": args.model, "tier": 0, "description": "user-specified"}]

    try:
        run_benchmark(models, image_paths)
    finally:
        if server_proc:
            print("[*] Stopping ollama server...")
            server_proc.terminate()
            server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
