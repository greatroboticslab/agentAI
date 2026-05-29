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


def _primary_species(lbl: Path):
    """Read a YOLO .txt and return the CWD12 name of its most-frequent class.
    None if no valid boxes."""
    try:
        lines = lbl.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    from collections import Counter
    c = Counter()
    for ln in lines:
        p = ln.split()
        if p and p[0].lstrip("-").isdigit():
            cid = int(p[0])
            if 0 <= cid < len(CWD12):
                c[cid] += 1
    if not c:
        return None
    return CWD12[c.most_common(1)[0][0]]


def cmd_bulk_upload(args):
    """Upload images grouped BY SPECIES into per-species Roboflow batches
    (user request: 'different classes in different places'), in parallel.

    One project (cwd12-weeds), but batch_name = green-<species> so the
    Annotate view groups/filters by species. Parallel workers measure
    whether 57s/img was per-call overhead (parallel helps) or global
    free-tier rate limiting (parallel won't help → wait for paid tier)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    images = Path(args.images)
    labels = Path(args.labels)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_imgs = sorted(p for p in images.iterdir() if p.suffix in exts)

    # group by primary species; cap per species for testing via --per-species
    by_sp: dict = {}
    for img in all_imgs:
        lbl = labels / (img.stem + ".txt")
        sp = _primary_species(lbl) if lbl.is_file() else None
        sp = sp or "Unlabeled"
        by_sp.setdefault(sp, []).append((img, lbl if lbl.is_file() else None))
    # build work list with per-species cap
    work = []
    for sp, items in sorted(by_sp.items()):
        take = items[: args.per_species] if args.per_species else items
        for img, lbl in take:
            work.append((sp, img, lbl))
    print(f"species present: { {k: len(v) for k, v in sorted(by_sp.items())} }")
    print(f"uploading {len(work)} images across {len(by_sp)} species, "
          f"workers={args.workers}")

    ws = _workspace()
    proj = ws.project(PROJECT_NAME)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}
    labelmap = {i: n for i, n in enumerate(CWD12)}

    def _one(task):
        sp, img, lbl = task
        t0 = time.time()
        try:
            kw = dict(image_path=str(img), split=args.split,
                      batch_name=f"{args.batch}-{sp}",
                      tag_names=[args.batch, sp], num_retry_uploads=1)
            if lbl is not None:
                kw["annotation_path"] = str(lbl)
                kw["annotation_labelmap"] = labelmap
            proj.single_upload(**kw)
            with lock:
                counters["ok"] += 1
            return (sp, img.name, time.time() - t0, None)
        except Exception as e:
            with lock:
                counters["fail"] += 1
            return (sp, img.name, time.time() - t0, f"{type(e).__name__}: {e}")

    t0 = time.time()
    per_times = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, t) for t in work]
        for i, f in enumerate(as_completed(futs), 1):
            sp, name, dt, err = f.result()
            per_times.append(dt)
            if err:
                print(f"  FAIL [{sp}] {name}: {err}")
            if i % 10 == 0:
                print(f"  ... {i}/{len(work)} ok={counters['ok']} "
                      f"fail={counters['fail']} ({time.time()-t0:.0f}s)")
    wall = time.time() - t0
    avg = sum(per_times) / len(per_times) if per_times else 0
    print(f"DONE bulk: ok={counters['ok']} fail={counters['fail']} "
          f"wall={wall:.0f}s  per-img(in-thread)avg={avg:.1f}s  "
          f"effective={wall/max(1,len(work)):.1f}s/img")


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
    bu = sub.add_parser("bulk-upload")
    bu.add_argument("--images", required=True)
    bu.add_argument("--labels", required=True)
    bu.add_argument("--split", default="train", choices=["train", "valid", "test"])
    bu.add_argument("--batch", default="green")
    bu.add_argument("--workers", type=int, default=8)
    bu.add_argument("--per-species", type=int, default=0,
                    help="cap images per species (0=all). For testing.")
    args = ap.parse_args()

    {"whoami": cmd_whoami,
     "create-project": cmd_create_project,
     "upload": cmd_upload,
     "bulk-upload": cmd_bulk_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
