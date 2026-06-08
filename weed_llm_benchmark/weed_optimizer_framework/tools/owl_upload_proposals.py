"""owl_upload_proposals.py — precision-GATED upload of OWL red proposals.

The OWL pre-annotation chain can massively over-propose (precision was 0.002 in
the v3.0.96 test). Uploading those raw proposals to Roboflow pollutes the human
labeling surface. This wrapper measures auto-label precision against the holdout
GT FIRST and refuses to upload unless precision clears a bar (or --force).

Flow:
  1. owl_precision.evaluate(species, prop_dir, gt_dir)
  2. gate on precision (global) or precision_on_gt_images (--use-on-gt)
  3. if pass OR --force → exec roboflow_sync bulk-upload (the real upload)
     else → print a clear refusal + exit 2 (button shows "failed", nothing uploaded)

Env overrides (so the dashboard button stays zero-arg but tunable):
  OWL_UPLOAD_MIN_PRECISION (default 0.30)
  OWL_UPLOAD_USE_ON_GT     (1 → gate on precision_on_gt_images)
  OWL_UPLOAD_FORCE         (1 → skip the gate)
  OWL_SPECIES (default Goosegrass)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from weed_optimizer_framework.tools import owl_precision as _op

REPO = Path(os.environ.get(
    "REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))


def main() -> int:
    ap = argparse.ArgumentParser(description="precision-gated OWL proposal upload")
    ap.add_argument("--species", default=os.environ.get("OWL_SPECIES", "Goosegrass"))
    ap.add_argument("--prop-dir", default=None,
                    help="OWL red proposals dir (default results/framework/owl_red_proposals/<species>)")
    ap.add_argument("--gt-dir", default=None,
                    help="holdout GT labels (default downloads/cottonweeddet12/valid/labels)")
    ap.add_argument("--images", default="results/leave4out/dataset_holdout/test/images")
    ap.add_argument("--project", default="weed-crop-agent-dataset")
    ap.add_argument("--per-species", default="50")
    ap.add_argument("--workers", default="4")
    ap.add_argument("--min-precision", type=float,
                    default=float(os.environ.get("OWL_UPLOAD_MIN_PRECISION", "0.30")))
    ap.add_argument("--use-on-gt", action="store_true",
                    default=os.environ.get("OWL_UPLOAD_USE_ON_GT", "0") == "1",
                    help="gate on precision_on_gt_images (only imgs that contain the species)")
    ap.add_argument("--force", action="store_true",
                    default=os.environ.get("OWL_UPLOAD_FORCE", "0") == "1",
                    help="upload regardless of precision")
    args = ap.parse_args()

    prop_dir = Path(args.prop_dir) if args.prop_dir else \
        REPO / "results" / "framework" / "owl_red_proposals" / args.species
    gt_dir = Path(args.gt_dir) if args.gt_dir else \
        REPO / "downloads" / "cottonweeddet12" / "valid" / "labels"

    print(f"[owl-upload-gate] species={args.species} prop_dir={prop_dir}")
    if not prop_dir.is_dir():
        print(f"FATAL: proposals dir not found: {prop_dir} — run owl_preannotate first")
        return 2

    res = _op.evaluate(args.species, prop_dir, gt_dir)
    key = "precision_on_gt_images" if args.use_on_gt else "precision"
    prec = res.get(key, 0.0)
    print(f"[owl-upload-gate] {key}={prec}  (global precision={res.get('precision')}, "
          f"on_gt={res.get('precision_on_gt_images')}, recall={res.get('recall')}, "
          f"props={res.get('n_proposal_boxes')}, gt={res.get('n_gt_boxes')})")
    print(f"[owl-upload-gate] gate: need {key} >= {args.min_precision}  force={args.force}")

    if not args.force and prec < args.min_precision:
        print(f"REFUSED: {key}={prec} < {args.min_precision}. NOT uploading "
              f"(would pollute Roboflow with low-quality boxes).")
        print("  → improve OWL (lower --top-k, raise --conf-threshold, species-matched "
              "target images) and re-measure, or set OWL_UPLOAD_FORCE=1 to override.")
        return 2

    argv = [
        "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
        "bulk-upload",
        "--images", args.images,
        "--labels", str(prop_dir),
        "--split", "train", "--batch", "red",
        "--workers", args.workers,
        "--per-species", args.per_species,
        "--project", args.project,
    ]
    print(f"[owl-upload-gate] PASS → uploading: {' '.join(argv)}")
    return subprocess.call(argv, cwd=str(REPO))


if __name__ == "__main__":
    sys.exit(main())
