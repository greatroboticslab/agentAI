"""
Quick single-model test script. Minimal setup, just test one image fast.

Usage (on a GPU node):
    conda activate qwen
    python quick_test.py --image YOUR_WEED_PHOTO.jpg
    python quick_test.py --image YOUR_WEED_PHOTO.jpg --model qwen3b   # lighter model
"""

import argparse
import json
import os
import sys
import time

os.environ["HF_HOME"] = "/ocean/projects/cis240145p/byler/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis240145p/byler/hf_cache/hub"

PROMPT = """Analyze this agricultural field image carefully.
Identify ALL visible plants, distinguishing between crops and weeds.

For each detection provide:
- "label": "weed" or "crop"
- "species": specific name if identifiable
- "confidence": "high"/"medium"/"low"
- "bbox": [x_min, y_min, x_max, y_max] as percentage (0-100)
- "description": brief visual description

Return ONLY valid JSON:
{"detections": [...], "weed_severity": "none|low|medium|high", "crop_type": "..."}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to weed image")
    parser.add_argument("--model", default="qwen7b", choices=["qwen7b", "qwen3b"])
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct" if args.model == "qwen7b" else "Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"Loading {model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir="/ocean/projects/cis240145p/byler/hf_cache/hub",
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir="/ocean/projects/cis240145p/byler/hf_cache/hub",
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(args.image)}"},
        {"type": "text", "text": PROMPT},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    print(f"Running inference on {args.image}...")
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
    elapsed = time.time() - start

    generated = out[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    print(f"\n{'='*60}")
    print(f"Model: {model_id} | Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}")

    # Try to save parsed result
    import re
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{[\s\S]*\}"]:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if "```" in pattern else match.group(0)
                parsed = json.loads(candidate)
                out_file = f"result_{os.path.splitext(os.path.basename(args.image))[0]}.json"
                with open(out_file, "w") as f:
                    json.dump(parsed, f, indent=2)
                print(f"\nParsed JSON saved to: {out_file}")
                dets = parsed.get("detections", [])
                weeds = [d for d in dets if "weed" in str(d.get("label", "")).lower()]
                print(f"Found {len(dets)} detections, {len(weeds)} weeds")
                break
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    main()
