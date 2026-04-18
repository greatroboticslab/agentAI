"""
Auto-label classification datasets with OWLv2 → convert to YOLO bbox format.

Why: most "weed detection" datasets discovered by Kaggle/HF search are actually
classification (plant disease, plantvillage, etc.). Rejecting them caps us at
~11K bbox images. Auto-labeling turns 380K+ classification images into bbox
training data — a much cleaner signal than blind VLM consensus because the
class IS known to be present (GT from classification), so OWLv2 only has to
localize, not identify.

Pipeline:
  1. Pick OWLv2 with a text prompt per dataset (derived from dataset description
     or a safe default like "plant weed leaf").
  2. For each image:
     - Run OWLv2 → list of (box, score)
     - Keep boxes with score >= conf_threshold
     - If no box passes, write an empty label (skip in training) or fall back
       to whole-image box (weak but lossless).
  3. Write YOLO labels to the dataset's `labels/` dir alongside `images/`.
  4. Flip registry annotation from `needs_autolabel` → `yolo_autolabel` so
     mega_trainer picks them up.

mega_trainer expects paths like `images/x.jpg` + `labels/x.txt`. Our Kaggle
downloads are copied from kagglehub cache with arbitrary internal layouts, so
we walk the dataset dir for images and place labels as siblings.
"""

import os
import gc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


def _pick_prompt(dataset_info):
    """Infer a reasonable OWLv2 text prompt from the dataset's metadata."""
    desc = (dataset_info.get("description") or "").lower()
    name = (dataset_info.get("kaggle_ref") or dataset_info.get("hf_id")
            or dataset_info.get("github_name") or "").lower()
    hay = f"{desc} {name}"
    # Order matters: most specific first
    if "weed" in hay:
        return "a weed"
    if "disease" in hay or "blight" in hay or "rust" in hay or "mildew" in hay:
        return "a plant leaf"
    if "fruit" in hay or "tomato" in hay or "apple" in hay:
        return "a fruit"
    if "wheat" in hay or "rice" in hay or "corn" in hay or "maize" in hay:
        return "a crop plant"
    if "seedling" in hay:
        return "a seedling"
    if "pest" in hay or "insect" in hay:
        return "an insect"
    if "plant" in hay or "leaf" in hay:
        return "a plant leaf"
    return "a plant"


def _find_images(root):
    """Recursively find image files under root."""
    images = []
    for ext in IMG_EXTS:
        images.extend(Path(root).rglob(f"*{ext}"))
    # Drop any inside an existing labels/ dir
    return [p for p in images if "labels" not in p.parts]


def _label_path_for(img_path):
    """Sibling label path: {parent}/labels/{stem}.txt (mega_trainer-compatible)."""
    p = Path(img_path)
    # If image is under .../images/..., map to .../labels/...
    if "images" in p.parts:
        parts = list(p.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
    else:
        lbl = p.parent / "labels" / (p.stem + ".txt")
    return lbl


def autolabel_dataset(slug, registry_cb, conf_threshold=0.12, max_images=None,
                      prompt=None, fallback_whole_image=True, class_id=0):
    """Generate YOLO-format bbox labels for an unlabeled dataset using OWLv2.

    Args:
      slug: dataset slug (key into registry)
      registry_cb: dict {"get": fn(slug)->info, "update": fn(slug, updates)}
      conf_threshold: OWLv2 score filter (0.10-0.15 works for class-known images)
      max_images: cap per-dataset for fast iteration; None = all
      prompt: override the inferred text prompt
      fallback_whole_image: if OWLv2 returns nothing, emit a whole-image bbox
      class_id: YOLO class id to assign (default 0 = single-class generic weed)

    Returns:
      dict with keys: status, images, labeled_with_owl, labeled_with_fallback,
                     empty, prompt, avg_boxes_per_image
    """
    info = registry_cb["get"](slug)
    if not info:
        return {"status": "unknown_dataset"}
    local_path = info.get("local_path")
    if not local_path or not os.path.isdir(local_path):
        return {"status": "no_local_path"}

    text_prompt = prompt or _pick_prompt(info)
    logger.info(f"[Autolabel] {slug}: prompt={text_prompt!r}, conf>={conf_threshold}")

    images = _find_images(local_path)
    if max_images:
        images = images[:max_images]
    if not images:
        return {"status": "no_images"}
    logger.info(f"[Autolabel] {slug}: {len(images)} images to process")

    # Load OWLv2 once
    try:
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        from PIL import Image
    except Exception as e:
        return {"status": "owlv2_import_failed", "error": str(e)[:200]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Autolabel] loading OWLv2 on {device}...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-large-patch14-ensemble"
    ).to(device)
    model.eval()

    texts = [[text_prompt]]
    stats = {
        "status": "ok", "images": len(images),
        "labeled_with_owl": 0, "labeled_with_fallback": 0, "empty": 0,
        "total_boxes": 0, "prompt": text_prompt,
    }

    for idx, img_path in enumerate(images):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.debug(f"[Autolabel] skip unreadable {img_path}: {e}")
            stats["empty"] += 1
            continue
        w, h = image.size

        lines = []
        try:
            inputs = processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([[h, w]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )
            if results and len(results[0]["scores"]) > 0:
                for score, box in zip(results[0]["scores"], results[0]["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    if 0.01 < bw < 0.98 and 0.01 < bh < 0.98:
                        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        except Exception as e:
            logger.debug(f"[Autolabel] OWL err on {img_path.name}: {str(e)[:80]}")

        if lines:
            stats["labeled_with_owl"] += 1
        elif fallback_whole_image:
            # Whole image as single bbox — weak but preserves the image in training
            lines = [f"{class_id} 0.5 0.5 1.0 1.0"]
            stats["labeled_with_fallback"] += 1
        else:
            stats["empty"] += 1
            continue

        stats["total_boxes"] += len(lines)
        lbl_path = _label_path_for(img_path)
        os.makedirs(lbl_path.parent, exist_ok=True)
        lbl_path.write_text("\n".join(lines) + "\n")

        if (idx + 1) % 500 == 0:
            logger.info(f"[Autolabel] {slug}: {idx+1}/{len(images)} — "
                        f"owl={stats['labeled_with_owl']} "
                        f"fb={stats['labeled_with_fallback']}")

    # Cleanup GPU
    del model, processor
    import gc as _gc; _gc.collect()
    try:
        import torch as _t; _t.cuda.empty_cache()
    except Exception:
        pass

    stats["avg_boxes_per_image"] = round(stats["total_boxes"] / max(stats["images"], 1), 2)

    # Flip registry annotation
    total_labeled = stats["labeled_with_owl"] + stats["labeled_with_fallback"]
    registry_cb["update"](slug, {
        "annotation": "yolo_autolabel",
        "autolabel_prompt": text_prompt,
        "autolabel_conf": conf_threshold,
        "autolabel_stats": {
            "with_owl": stats["labeled_with_owl"],
            "with_fallback": stats["labeled_with_fallback"],
            "empty": stats["empty"],
            "total_boxes": stats["total_boxes"],
        },
        "local_labeled": total_labeled,
    })
    logger.info(f"[Autolabel] {slug}: COMPLETE. owl={stats['labeled_with_owl']} "
                f"fallback={stats['labeled_with_fallback']} empty={stats['empty']} "
                f"avg_boxes={stats['avg_boxes_per_image']}")
    return stats
