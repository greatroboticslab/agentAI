"""
roboflow_sync.py — push curated weed datasets to a Roboflow project for
bbox ground-truth labeling, and pull verified annotations back.

Part of the Roboflow-integrated active-learning labeling pipeline
(see memory project_roboflow_pipeline_plan). Division of labor:
  our site = high-level dataset SELECTION  →  Roboflow = bbox labeling
  → ground truth  →  cluster = training.

SECURITY: the API key is read from env ROBOFLOW_API_KEY only. NEVER hard-
code it, never print it, never commit it. (Same rule as the GitHub PAT.)

Project structure (user decision 2026-05-28): ONE multi-class object-
detection project `cwd12-weeds` with the 12 CottonWeedDet12 classes.
Seed = cottonweeddet12/train (gold, all 12 species). NEVER upload the
valid/test split (eval contamination).

Usage (run on cluster where the images live; key in env):
  export ROBOFLOW_API_KEY=...        # do NOT echo
  python -m weed_optimizer_framework.tools.roboflow_sync whoami
  python -m weed_optimizer_framework.tools.roboflow_sync create-project
  python -m weed_optimizer_framework.tools.roboflow_sync upload \
       --images downloads/cottonweeddet12/train/images \
       --labels downloads/cottonweeddet12/train/labels \
       --split train --batch green --limit 5      # test first!
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CWD12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]
PROJECT_NAME = "cwd12-weeds"


def _key() -> str:
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not k:
        print("FATAL: ROBOFLOW_API_KEY not set in env", file=sys.stderr)
        sys.exit(2)
    return k


def _rf():
    from roboflow import Roboflow
    return Roboflow(api_key=_key())


def _workspace():
    return _rf().workspace()


def cmd_whoami(args):
    ws = _workspace()
    # ws.url / ws.project_list — print non-secret identifiers only
    print("workspace:", getattr(ws, "url", "?"))
    try:
        projs = ws.projects()
        print("projects:", projs)
    except Exception as e:
        print("projects(): ", type(e).__name__, e)


def cmd_create_project(args):
    ws = _workspace()
    try:
        proj = ws.create_project(
            project_name=PROJECT_NAME,
            project_type="object-detection",
            project_license="MIT",
            annotation=PROJECT_NAME,
        )
        print(f"CREATED project {PROJECT_NAME}: {proj}")
    except Exception as e:
        # Already exists is fine — report and continue
        print(f"create_project: {type(e).__name__}: {e}")
        print("(if it already exists, that's OK — proceed to upload)")


def cmd_upload(args):
    """Upload images + YOLO .txt annotations to the project.
    `batch` tags provenance: green=human/gold-trusted, red=model-proposed."""
    images = Path(args.images)
    labels = Path(args.labels)
    if not images.is_dir():
        print(f"FATAL: images dir not found: {images}", file=sys.stderr)
        sys.exit(2)
    ws = _workspace()
    proj = ws.project(PROJECT_NAME)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    imgs = sorted(p for p in images.iterdir() if p.suffix in exts)
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"uploading {len(imgs)} images  split={args.split} batch={args.batch}")

    ok = fail = noann = 0
    t0 = time.time()
    for i, img in enumerate(imgs, 1):
        lbl = labels / (img.stem + ".txt") if labels else None
        has_ann = lbl is not None and lbl.is_file()
        if not has_ann:
            noann += 1
        try:
            kw = dict(
                image_path=str(img),
                split=args.split,
                batch_name=f"{args.batch}-{args.split}",
                tag_names=[args.batch],
                num_retry_uploads=2,
            )
            if has_ann:
                kw["annotation_path"] = str(lbl)
                # YOLO txt needs the class index→name map
                kw["annotation_labelmap"] = {i: n for i, n in enumerate(CWD12)}
            proj.single_upload(**kw)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  [{i}] FAIL {img.name}: {type(e).__name__}: {e}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(imgs)} ok={ok} fail={fail} "
                  f"({time.time()-t0:.0f}s)")
    print(f"DONE upload: ok={ok} fail={fail} no_annotation={noann} "
          f"in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    sub.add_parser("create-project")
    up = sub.add_parser("upload")
    up.add_argument("--images", required=True)
    up.add_argument("--labels", default="")
    up.add_argument("--split", default="train", choices=["train", "valid", "test"])
    up.add_argument("--batch", default="green",
                    help="provenance tag: green=human/gold, red=model-proposed")
    up.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    {"whoami": cmd_whoami,
     "create-project": cmd_create_project,
     "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
