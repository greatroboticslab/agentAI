"""
dinov2_route.py — DINOv2 embedding + cosine-similarity routing.

Use cases (see memory/project_roboflow_pipeline_plan.md, Phase D2):

1. **Bucket C router**: assign a CWD12 species to bucket-C unknown
   images via nearest-neighbor vs a per-species exemplar bank.

2. **Weed-vs-not-weed quality gate**: at harvest time, embed each
   downloaded image and flag those whose nearest exemplar is not a CWD12
   species (or below similarity threshold). Useful for off-goal slug
   detection (disease/crop/junk slipping through name-filter).

3. **Near-dup dedup across slugs**: embed every harvested image; flag
   pairs with cosine > 0.97 as near-duplicates regardless of source slug.
   Saves training compute and prevents data-leak across train/eval.

Approach:
- Backbone: facebook/dinov2-base (small, ~86M params) or dinov2-large
  for quality. Frozen, eval mode. Run on GPU if available, CPU fine for
  small batches.
- Exemplar bank: directory tree
      <bank>/<species>/<filename>.jpg
  where <species> ∈ CWD12 (or `not_weed` for negative class). The
  existing object_bank/ at results/framework/synth_cutpaste/object_bank/
  is the natural source — each CWD12 species has 50-400 cut-paste crops.
- For each target image: compute embedding, cosine vs every exemplar,
  return top-K hits + the dominant species label.

Outputs to --out JSON:
  {
    "exemplar_species": [...],
    "results": [
      {"image": "/abs/path", "top": [{"species": "Goosegrass",
                                       "exemplar": "g0042.jpg",
                                       "cosine": 0.84},
                                      ...K hits],
                  "best_species": "Goosegrass", "best_cosine": 0.84,
                  "is_cwd12_match": true},
      ...
    ]
  }

Threshold (--reject-below) defaults 0.5; below → "not_weed" label.

CLI:
  python -m weed_optimizer_framework.tools.dinov2_route \\
      --target-dir <imgs> \\
      --exemplar-root results/framework/synth_cutpaste/object_bank \\
      --out results/framework/dinov2_routing.json \\
      --top-k 5 --reject-below 0.5 --max-images 200 [--dry-run]

GPU recommended (10× faster). Sbatch wrapper to follow next iter
(run_v3_0_52_dinov2_route.sh, planned).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))

CWD12 = (
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
)


def _load_dinov2(model_id: str = "facebook/dinov2-base"):
    """Lazy load DINOv2 backbone + processor."""
    print(f"[dinov2] loading {model_id} ...")
    import torch
    from transformers import AutoModel, AutoImageProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[dinov2] WARNING no GPU — will be slow", file=sys.stderr)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return processor, model, device


def _embed_images(image_paths, processor, model, device, batch_size: int = 16):
    """Yield (path, embedding_tensor) for each image. Embedding is
    L2-normalized CLS token (so cosine = dot product)."""
    import torch
    from PIL import Image
    batch_imgs = []; batch_paths = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[dinov2] skip {p}: {e}", file=sys.stderr)
            continue
        batch_imgs.append(img); batch_paths.append(p)
        if len(batch_imgs) == batch_size:
            yield from _process_batch(batch_imgs, batch_paths,
                                       processor, model, device)
            batch_imgs = []; batch_paths = []
    if batch_imgs:
        yield from _process_batch(batch_imgs, batch_paths,
                                   processor, model, device)


def _process_batch(imgs, paths, processor, model, device):
    import torch
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # CLS token is the first position
        emb = outputs.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, dim=-1)
    for path, e in zip(paths, emb.cpu()):
        yield path, e


def build_exemplar_bank(exemplar_root: Path, max_per_species: int = 50):
    """Walk exemplar tree and return [(species, path), ...]."""
    bank: list = []
    if not exemplar_root.is_dir():
        return bank
    for sp_dir in sorted(exemplar_root.iterdir()):
        if not sp_dir.is_dir():
            continue
        species = sp_dir.name
        imgs = []
        for p in sp_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(p)
            if len(imgs) >= max_per_species:
                break
        for p in imgs:
            bank.append((species, p))
    return bank


def route(args):
    target_dir = Path(args.target_dir)
    exemplar_root = Path(args.exemplar_root)
    out_path = Path(args.out)

    if not target_dir.is_dir():
        print(f"FATAL: target-dir not found: {target_dir}", file=sys.stderr)
        sys.exit(2)
    if not exemplar_root.is_dir():
        print(f"FATAL: exemplar-root not found: {exemplar_root}", file=sys.stderr)
        sys.exit(2)

    bank = build_exemplar_bank(exemplar_root, args.max_per_species)
    bank_species = sorted({sp for sp, _ in bank})
    print(f"[bank] {len(bank)} exemplars across {len(bank_species)} species")
    if not bank:
        print("FATAL: empty exemplar bank", file=sys.stderr)
        sys.exit(2)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    target_imgs = sorted(p for p in target_dir.iterdir() if p.suffix in exts)
    if args.max_images:
        target_imgs = target_imgs[: args.max_images]
    print(f"[targets] {len(target_imgs)} images in {target_dir}")

    print(f"\n=== DINOv2 routing ===")
    print(f"  model:        {args.model_id}")
    print(f"  exemplars:    {len(bank)}")
    print(f"  targets:      {len(target_imgs)}")
    print(f"  top-K:        {args.top_k}")
    print(f"  reject-below: {args.reject_below}")
    if args.dry_run:
        print("[dry-run] not loading model. exiting.")
        return

    import torch
    processor, model, device = _load_dinov2(args.model_id)

    # Embed bank
    print("[bank] embedding exemplars ...")
    bank_emb = []
    bank_paths = [p for _, p in bank]
    for path, e in _embed_images(bank_paths, processor, model, device,
                                   batch_size=16):
        bank_emb.append(e)
    bank_emb_t = torch.stack(bank_emb)  # [N, D]
    print(f"[bank] embeddings ready: {tuple(bank_emb_t.shape)}")

    # Embed targets + nearest neighbor
    print("[targets] embedding + routing ...")
    results = []
    t0 = time.time()
    cwd12_set = set(CWD12)
    for img_path, te in _embed_images(target_imgs, processor, model, device,
                                       batch_size=16):
        sims = (bank_emb_t @ te).tolist()  # cosine since L2-normalized
        # top-K
        idxs = sorted(range(len(sims)), key=lambda i: -sims[i])[: args.top_k]
        top = [{"species": bank[i][0],
                "exemplar": bank[i][1].name,
                "cosine": round(sims[i], 4)} for i in idxs]
        best_sp = top[0]["species"] if top else "unknown"
        best_cos = top[0]["cosine"] if top else 0.0
        if best_cos < args.reject_below:
            best_sp = "not_weed"
        is_cwd12 = best_sp in cwd12_set
        results.append({
            "image": str(img_path),
            "top": top,
            "best_species": best_sp,
            "best_cosine": best_cos,
            "is_cwd12_match": is_cwd12,
        })
    elapsed = time.time() - t0
    print(f"[done] {len(results)} targets routed in {elapsed:.0f}s "
          f"({len(results)/max(1,elapsed):.1f} imgs/s)")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": args.model_id,
        "exemplar_root": str(exemplar_root),
        "exemplar_species": bank_species,
        "n_exemplars": len(bank),
        "n_targets": len(results),
        "elapsed_sec": round(elapsed, 1),
        "reject_below": args.reject_below,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp, out_path)
    print(f"\nWROTE: {out_path} ({out_path.stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--exemplar-root",
                    default=str(REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"))
    ap.add_argument("--out",
                    default=str(REPO / "results" / "framework" / "dinov2_routing.json"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--reject-below", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-per-species", type=int, default=50,
                    help="cap exemplars per species (for speed)")
    ap.add_argument("--model-id", default="facebook/dinov2-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    route(args)


if __name__ == "__main__":
    main()
