"""
v3.0.30.2 — build/refresh rich training+eval metadata next to each pycoco summary.

For each `*_pycoco_summary.json`, write `*_meta.json` alongside it with:
  - training: dict of run config (slugs used, image count, classes, epochs, etc.)
  - per_class_AP: dict {class_name: {AP50, AP50_95, n_gt}}
  - per_class breakdown computed by re-running COCOeval with categoryIds=[c]
  - quality_flags: counts of user-flagged garbage / good / unsure at run time

For the v3.0.29 SAFETY result we backfill from known facts.
For future eval scripts, this metadata is computed/written by the eval itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]


def per_class_ap_from_pycoco(gt_path: Path, pred_path: Path,
                              cat_ids: list[int]) -> dict:
    """Run COCOeval once per category to get per-class AP50, AP50_95."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    try:
        coco_dt = coco_gt.loadRes(str(pred_path))
    except Exception as e:
        print(f"[meta] loadRes failed: {e}")
        return {}

    per_class = {}
    for cid in cat_ids:
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.catIds = [cid]
        ev.evaluate()
        ev.accumulate()
        # capture without summarize() noise
        # stats[0]=mAP50-95, stats[1]=mAP50
        # Be defensive: catch all-empty case
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()
        s = ev.stats
        gt_count = sum(1 for a in coco_gt.dataset["annotations"]
                       if a["category_id"] == cid)
        name = CANONICAL_12[cid] if cid < len(CANONICAL_12) else f"cls{cid}"
        per_class[name] = {
            "AP50_95": float(s[0]) if s[0] >= 0 else 0.0,
            "AP50":    float(s[1]) if s[1] >= 0 else 0.0,
            "AP75":    float(s[2]) if s[2] >= 0 else 0.0,
            "n_gt":    gt_count,
        }
    return per_class


def known_training_meta_v3_0_29() -> dict:
    """Backfill what we know about v3.0.29 SAFETY (safety best.pt run)."""
    return {
        "label": "v3.0.29 SAFETY (yolo26x cwd12-only)",
        "research_version": "v3.0.29",
        "model_arch": "yolo26x.pt (COCO pretrained)",
        "training_data": {
            "primary_source": "cwd12 train portion only (stem-filtered, 0 leak)",
            "n_slugs_used": 1,
            "n_images_train": 3671,
            "n_classes": 12,
            "class_names": CANONICAL_12,
            "include_autolabel": False,
            "uses_brain_harvested": False,
        },
        "validation_data": {
            "source": "cwd12 test (848) + valid (1129) = 1977 NEVER_TRAIN holdout",
            "n_images": 1977,
            "n_annotations": 3257,
        },
        "run_config": {
            "epochs_planned": 200,
            "epochs_completed": 62,
            "exit_reason": "TIMEOUT at 12h walltime",
            "imgsz": 1024,
            "batch": 8,
            "lr0": 0.001,
            "lr_final": 0.01,
            "patience": 40,
            "optimizer": "auto (SGD-like)",
            "augmentation": {
                "mosaic": 1.0,
                "mixup": 0.1,
                "close_mosaic": 15,
                "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                "degrees": 10.0, "translate": 0.1, "scale": 0.5,
                "fliplr": 0.5,
            },
            "slurm_job_id": "40612856",
            "wallclock_h": 12.0,
        },
        "notes": (
            "Baseline 'safety net' run after v3.0.27 contamination retraction. "
            "Trains on cwd12 train alone (no Brain-harvested data) to get a "
            "clean baseline. v3.0.6 published baseline on same task was 0.865 "
            "ultralytics (~0.71 pyco)."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="dir containing *_pycoco_summary.json files")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    summaries = sorted(results_dir.glob("*pycoco*summary.json"))
    print(f"[meta] found {len(summaries)} summaries in {results_dir}")

    cat_ids_12 = list(range(12))

    for summ_path in summaries:
        # Detect what version this is
        stem = summ_path.stem
        meta_path = summ_path.with_name(stem.replace("_summary", "_meta") + ".json")
        print(f"\n[meta] processing {summ_path.name}")
        summ = json.loads(summ_path.read_text())

        # Find sibling gt + pred JSONs (named "*_gt.json" and "*_pred.json")
        gt_path = summ_path.with_name(stem.replace("_summary", "_gt") + ".json")
        pred_path = summ_path.with_name(stem.replace("_summary", "_pred") + ".json")
        if not gt_path.exists():
            # try generic v3_0_29_pycoco_gt.json sibling
            cands = sorted(results_dir.glob("v3_0_29_pycoco_gt.json"))
            if cands: gt_path = cands[0]
        if not pred_path.exists():
            cands = sorted(results_dir.glob("v3_0_29_pycoco_pred.json"))
            if cands: pred_path = cands[0]

        per_class = {}
        if gt_path.exists() and pred_path.exists():
            print(f"[meta]   computing per-class AP from "
                  f"{gt_path.name} + {pred_path.name}...")
            try:
                per_class = per_class_ap_from_pycoco(gt_path, pred_path, cat_ids_12)
                print(f"[meta]   per-class done: "
                      f"{[(k, round(v['AP50_95'],3)) for k,v in per_class.items()]}")
            except Exception as e:
                print(f"[meta]   per-class FAILED: {e}")
        else:
            print(f"[meta]   no sibling GT/pred; skipping per-class AP")

        # Pick training metadata: hardcoded for known runs
        if "v3_0_29" in stem and "rfdetr" not in stem:
            training_meta = known_training_meta_v3_0_29()
        else:
            training_meta = {
                "label": stem,
                "notes": "training metadata not yet captured for this run",
            }

        out = {
            **training_meta,
            "summary": summ,
            "per_class_AP": per_class,
        }

        with open(meta_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[meta]   wrote {meta_path.name}")


if __name__ == "__main__":
    main()
