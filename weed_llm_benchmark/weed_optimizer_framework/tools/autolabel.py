"""
Auto-label classification datasets with OWLv2 → convert to YOLO bbox format.

v3.0.13: batched inference + resume + per-dataset cap. v3.0.12 single-image
OWLv2 on V100 ran at ~1 img/sec — 380K images needed 100h, won't fit 8h
walltime. Batching to 16 images/forward + fp16 gets 15-20 img/sec (~10x).

Why auto-label at all: most "weed detection" matches on Kaggle/HF are actually
classification (plantvillage, plant-disease, etc.). Rejecting them caps us at
~11K bbox images. Autolabeling turns 380K+ into usable bbox training data —
cleaner signal than blind VLM consensus because the class IS known to be
present.

Pipeline:
  1. For each image, check if label .txt already exists (resume).
  2. Batch N images, run OWLv2 with text prompt inferred from dataset description.
  3. Per image: keep boxes with score >= conf_threshold, else fallback
     whole-image bbox.
  4. Write YOLO labels as `{parent}/labels/{stem}.txt`.
  5. Flip registry annotation: needs_autolabel → yolo_autolabel.
  6. Save registry every 500 images so walltime-cancelled runs preserve progress.
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
    images = []
    for ext in IMG_EXTS:
        images.extend(Path(root).rglob(f"*{ext}"))
    return [p for p in images if "labels" not in p.parts]


def _label_path_for(img_path):
    p = Path(img_path)
    if "images" in p.parts:
        parts = list(p.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
    else:
        lbl = p.parent / "labels" / (p.stem + ".txt")
    return lbl


def autolabel_dataset(slug, registry_cb, conf_threshold=0.12, max_images=30000,
                      prompt=None, fallback_whole_image=True, class_id=0,
                      batch_size=16, save_every=500):
    """Generate YOLO-format bbox labels for an unlabeled dataset using OWLv2.

    v3.0.13 changes:
      * Batched inference (batch_size=16 default) — ~10-20x speedup on V100
      * fp16 model weights for memory + throughput
      * Resume: skip images whose label .txt already exists
      * max_images default 30000 (was None) to fit 8h walltime across 3 datasets
      * Periodic registry save every `save_every` images

    Args:
      slug: dataset slug (key into registry)
      registry_cb: dict {"get": fn(slug)->info, "update": fn(slug, updates)}
      conf_threshold: OWLv2 score filter (0.10-0.15 works)
      max_images: per-dataset cap (30000 = ~30 min on V100 with batch=16)
      prompt: override the inferred text prompt
      fallback_whole_image: if OWLv2 returns nothing, emit whole-image bbox
      class_id: YOLO class id (default 0 = single-class generic)
      batch_size: OWLv2 forward-pass batch size
      save_every: save registry after processing this many images
    """
    info = registry_cb["get"](slug)
    if not info:
        return {"status": "unknown_dataset"}
    local_path = info.get("local_path")
    if not local_path or not os.path.isdir(local_path):
        return {"status": "no_local_path"}

    text_prompt = prompt or _pick_prompt(info)
    images_all = _find_images(local_path)
    if max_images:
        images_all = images_all[:max_images]
    if not images_all:
        return {"status": "no_images"}

    # Resume: drop images whose label already exists
    images = []
    resumed = 0
    for p in images_all:
        lbl = _label_path_for(p)
        if lbl.exists() and lbl.stat().st_size > 0:
            resumed += 1
            continue
        images.append(p)

    logger.info(f"[Autolabel] {slug}: prompt={text_prompt!r} conf>={conf_threshold} "
                f"total={len(images_all)} resumed={resumed} todo={len(images)} "
                f"batch={batch_size}")

    if not images:
        # All done already — just flip the registry annotation
        stats = {"status": "ok", "images": len(images_all), "resumed": resumed,
                 "labeled_with_owl": 0, "labeled_with_fallback": 0,
                 "empty": 0, "total_boxes": 0, "prompt": text_prompt}
        _flip_registry(registry_cb, slug, stats, text_prompt, conf_threshold, resumed)
        return stats

    try:
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        from PIL import Image
    except Exception as e:
        return {"status": "owlv2_import_failed", "error": str(e)[:200]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 on CUDA for 2x memory + speed; fp32 on CPU for correctness
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"[Autolabel] loading OWLv2 on {device} ({dtype})...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-large-patch14-ensemble", torch_dtype=dtype,
    ).to(device)
    model.eval()

    texts_per_image = [[text_prompt]]
    stats = {
        "status": "ok", "images": len(images_all), "resumed": resumed,
        "labeled_with_owl": 0, "labeled_with_fallback": 0, "empty": 0,
        "total_boxes": 0, "prompt": text_prompt,
    }

    processed = 0
    skipped_unreadable = 0

    for batch_start in range(0, len(images), batch_size):
        batch_paths = images[batch_start : batch_start + batch_size]
        batch_pil = []
        batch_sizes = []
        batch_valid_paths = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
                batch_pil.append(im)
                batch_sizes.append(im.size)  # (w, h)
                batch_valid_paths.append(p)
            except Exception:
                skipped_unreadable += 1

        if not batch_pil:
            processed += len(batch_paths)
            continue

        # OWLv2 expects parallel lists: texts[i] applies to image[i]
        texts_batch = [[text_prompt]] * len(batch_pil)

        try:
            inputs = processor(text=texts_batch, images=batch_pil, return_tensors="pt",
                               padding=True)
            # Move + cast
            for k, v in inputs.items():
                if v.dtype in (torch.float32, torch.float16):
                    inputs[k] = v.to(device=device, dtype=dtype)
                else:
                    inputs[k] = v.to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # target_sizes as (H, W) for post_process
            target_sizes = torch.tensor([[h, w] for (w, h) in batch_sizes]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=conf_threshold
            )
        except Exception as e:
            logger.warning(f"[Autolabel] batch err ({len(batch_pil)} imgs): "
                           f"{type(e).__name__}: {str(e)[:120]} — falling back whole-image")
            results = [{"scores": [], "boxes": []} for _ in batch_pil]

        # Write labels per image
        for (img_path, (w, h), r) in zip(batch_valid_paths, batch_sizes, results):
            lines = []
            scores = r["scores"] if hasattr(r["scores"], "__iter__") else []
            boxes = r["boxes"] if hasattr(r["boxes"], "__iter__") else []
            for score, box in zip(scores, boxes):
                try:
                    x1, y1, x2, y2 = box.tolist()
                except Exception:
                    x1, y1, x2, y2 = list(box)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if 0.01 < bw < 0.98 and 0.01 < bh < 0.98:
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if lines:
                stats["labeled_with_owl"] += 1
            elif fallback_whole_image:
                lines = [f"{class_id} 0.5 0.5 1.0 1.0"]
                stats["labeled_with_fallback"] += 1
            else:
                stats["empty"] += 1
                processed += 1
                continue

            stats["total_boxes"] += len(lines)
            lbl_path = _label_path_for(img_path)
            os.makedirs(lbl_path.parent, exist_ok=True)
            lbl_path.write_text("\n".join(lines) + "\n")
            processed += 1

        # Account for unreadable images in this batch
        processed += (len(batch_paths) - len(batch_valid_paths))

        if processed % save_every < batch_size:
            logger.info(f"[Autolabel] {slug}: {processed}/{len(images)} (resumed+{resumed}) "
                        f"owl={stats['labeled_with_owl']} fb={stats['labeled_with_fallback']} "
                        f"unread={skipped_unreadable}")
            # Incremental registry save so walltime-cancel preserves progress
            _flip_registry(registry_cb, slug, stats, text_prompt, conf_threshold,
                           resumed, in_progress=True)

    # Final save
    stats["empty"] += skipped_unreadable
    _flip_registry(registry_cb, slug, stats, text_prompt, conf_threshold, resumed)

    del model, processor
    gc.collect()
    try:
        import torch as _t; _t.cuda.empty_cache()
    except Exception:
        pass

    stats["avg_boxes_per_image"] = round(
        stats["total_boxes"] / max(stats["labeled_with_owl"] + stats["labeled_with_fallback"], 1), 2
    )
    logger.info(f"[Autolabel] {slug}: COMPLETE. owl={stats['labeled_with_owl']} "
                f"fallback={stats['labeled_with_fallback']} empty={stats['empty']} "
                f"resumed={resumed} avg_boxes={stats.get('avg_boxes_per_image', 0)}")
    return stats


def _flip_registry(registry_cb, slug, stats, prompt, conf, resumed, in_progress=False):
    """Update registry entry with autolabel progress."""
    total_labeled = stats["labeled_with_owl"] + stats["labeled_with_fallback"] + resumed
    registry_cb["update"](slug, {
        "annotation": "yolo_autolabel",
        "autolabel_prompt": prompt,
        "autolabel_conf": conf,
        "autolabel_in_progress": in_progress,
        "autolabel_stats": {
            "with_owl": stats["labeled_with_owl"],
            "with_fallback": stats["labeled_with_fallback"],
            "empty": stats["empty"],
            "resumed": resumed,
            "total_boxes": stats["total_boxes"],
        },
        "local_labeled": total_labeled,
    })
