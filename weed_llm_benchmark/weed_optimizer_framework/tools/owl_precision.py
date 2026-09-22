"""
owl_precision.py — Master-plan P2: OWL auto-label precision metric.

OWL pre-annotation writes single-species YOLO proposals (cid=0) for a target
species onto an image set; this measures how good those proposals are against the
hand-labeled holdout ground truth. This is the paper methodology number
("auto-label precision") the OWL chain was always meant to produce.

precision = #proposal boxes that match a GT box of the target species (IoU≥thr)
            / #proposal boxes
recall    = #GT boxes of the target species matched / #GT boxes

Default: SpottedSpurge proposals vs cwd12 holdout valid GT.

    python -m weed_optimizer_framework.tools.owl_precision --species SpottedSpurge
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()
if not REPO.exists():
    REPO = Path(__file__).resolve().parents[2]

from .cwd12_species import CWD12_SPECIES, cli_species, species_of

# cwd12 holdout GT uses the ORIGINAL cottonweeddet12 class ids.
# v3.60.0: the GT id is the species' cwd12 id (CWD12_SPECIES); it was the
# index of the legacy label, so "Goosegrass" scored against SpottedSpurge GT.


def _read_yolo(path: Path, only_cid=None):
    boxes = []
    try:
        for line in path.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            cid = int(float(t[0]))
            if only_cid is not None and cid != only_cid:
                continue
            cx, cy, w, h = map(float, t[1:5])
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    except Exception:
        pass
    return boxes


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def evaluate(species: str, prop_dir: Path, gt_dir: Path, iou_thr: float = 0.5) -> dict:
    sp = species_of(species)
    if sp is None:
        raise ValueError(f"{species!r} is not a cwd12 species "
                         f"({', '.join(CWD12_SPECIES)})")
    species = sp
    gt_cid = CWD12_SPECIES.index(sp)
    n_prop = n_prop_matched = n_gt = n_gt_matched = 0
    files_with_props = files_with_gt = 0
    n_files = 0
    # "on-GT" view: proposals only on images that actually contain the species.
    # Decouples the metric from the dataset mismatch where most sampled images
    # don't contain the target (which floors global precision artificially).
    n_prop_on_gt = n_prop_matched_on_gt = 0
    for pf in sorted(prop_dir.glob("*.txt")):
        n_files += 1
        props = _read_yolo(pf, only_cid=None)   # OWL writes cid=0 single-species
        gt = _read_yolo(gt_dir / pf.name, only_cid=gt_cid)
        if props:
            files_with_props += 1
        if gt:
            files_with_gt += 1
            n_prop_on_gt += len(props)
        n_prop += len(props)
        n_gt += len(gt)
        gt_used = [False] * len(gt)
        for pb in props:
            best, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if gt_used[j]:
                    continue
                v = _iou(pb, gb)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr and bj >= 0:
                gt_used[bj] = True
                n_prop_matched += 1
                if gt:
                    n_prop_matched_on_gt += 1
        n_gt_matched += sum(gt_used)
    precision = n_prop_matched / n_prop if n_prop else 0.0
    recall = n_gt_matched / n_gt if n_gt else 0.0
    precision_on_gt = n_prop_matched_on_gt / n_prop_on_gt if n_prop_on_gt else 0.0
    return {
        "species": species, "gt_cid": gt_cid, "iou_thr": iou_thr,
        "n_proposal_files": n_files,
        "files_with_proposals": files_with_props, "files_with_gt": files_with_gt,
        "n_proposal_boxes": n_prop, "n_gt_boxes": n_gt,
        "n_proposal_matched": n_prop_matched, "n_gt_matched": n_gt_matched,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "n_proposal_boxes_on_gt_images": n_prop_on_gt,
        "precision_on_gt_images": round(precision_on_gt, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OWL auto-label precision vs holdout GT")
    ap.add_argument("--species", default="SpottedSpurge",
                    help="cwd12 species (v3.60.0: species, not legacy label)")
    ap.add_argument("--prop-dir", default=None,
                    help="default results/framework/owl_red_proposals_species/<species>")
    ap.add_argument("--gt-dir", default=None,
                    help="default downloads/cottonweeddet12/valid/labels "
                         "(matches owl_preannotate default target valid/images)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    if cli_species(args.species) is None:   # v3.60.0: no legacy-only label
        ap.error(f"--species {args.species!r} is not a cwd12 species "
                 f"({', '.join(CWD12_SPECIES)})")
    args.species = cli_species(args.species)
    # v3.60.0: species-era proposals live under owl_red_proposals_species/.
    from .owl_preannotate import default_proposals_dir
    prop_dir = Path(args.prop_dir) if args.prop_dir else \
        default_proposals_dir(args.species)
    gt_dir = Path(args.gt_dir) if args.gt_dir else \
        REPO / "downloads" / "cottonweeddet12" / "valid" / "labels"
    print(f"[owl-precision] species={args.species}")
    print(f"  prop_dir: {prop_dir}  exists={prop_dir.is_dir()}")
    print(f"  gt_dir  : {gt_dir}  exists={gt_dir.is_dir()}")
    if not prop_dir.is_dir() or not gt_dir.is_dir():
        print("  ERROR: missing prop or gt dir")
        return
    res = evaluate(args.species, prop_dir, gt_dir, args.iou)
    import json
    print(json.dumps(res, indent=2))
    print(f"\n>>> OWL {args.species} auto-label PRECISION={res['precision']} "
          f"RECALL={res['recall']} (IoU≥{args.iou}, "
          f"{res['n_proposal_boxes']} props vs {res['n_gt_boxes']} GT)")


if __name__ == "__main__":
    main()
