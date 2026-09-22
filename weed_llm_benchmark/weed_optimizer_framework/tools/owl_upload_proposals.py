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
  OWL_SPECIES (default SpottedSpurge)

v3.60.0: --species is a cwd12 species. The proposal files hold it as class
id 0, so bulk-upload gets --single-species and uploads id 0 under that
species (in the target project's vocabulary); the 12-class map had uploaded
every OWL box as cwd12 id 0's name. A proposals dir must carry the
_owl_proposals.json that owl_preannotate writes: dirs from earlier runs are
named by a legacy label ("Goosegrass" held SpottedSpurge proposals).
--photos tells bulk-upload whether --images are cwd12 photographs (a cwd12
image dir is, see CWD12_IMAGE_DIRS); for any other --images it must be given, because boxes on
other photographs are written as real names, which download-merge reads back.
v3.60.0: the default --prop-dir is owl_red_proposals_species/<species> (the
legacy root reuses names such as Goosegrass for another plant), and the
default --images is the target_dir the manifest records: bulk-upload pairs
<images>/<stem> with <prop-dir>/<stem>.txt, and the old fixed default
(dataset_holdout/test) shares no stem with owl_preannotate_one's default
target (cottonweeddet12/valid), so every image went up as Unlabeled. An
upload whose --images share no stem with the proposals is refused.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from weed_optimizer_framework.tools import owl_precision as _op
from weed_optimizer_framework.tools.cwd12_species import CWD12_SPECIES, cli_species
from weed_optimizer_framework.tools.owl_preannotate import (
    PROPOSALS_MANIFEST, default_proposals_dir, read_manifest)

REPO = Path(os.environ.get(
    "REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))
# Image dirs that hold cwd12 photographs (relative to REPO).
CWD12_IMAGE_DIRS = (
    "results/leave4out/dataset_holdout/test/images",
    "downloads/cottonweeddet12/train/images",
    "downloads/cottonweeddet12/valid/images",
    "downloads/cottonweeddet12/test/images",
)
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _is_cwd12_image_dir(path):
    try:
        rp = Path(path).resolve()
    except Exception:
        return False
    return any(rp == (REPO / d).resolve() for d in CWD12_IMAGE_DIRS)


def _stem_overlap(images, prop_dir):
    """(images with a non-empty proposal file, proposal files)."""
    try:
        stems = {p.stem for p in Path(images).iterdir() if p.suffix in _IMG_EXTS}
    except OSError:
        stems = set()
    props = {p.stem for p in Path(prop_dir).iterdir()
             if p.suffix == ".txt" and p.stat().st_size > 0}
    return len(stems & props), len(props)


def main() -> int:
    ap = argparse.ArgumentParser(description="precision-gated OWL proposal upload")
    ap.add_argument("--species", default=os.environ.get("OWL_SPECIES", "SpottedSpurge"),
                    help="cwd12 species, e.g. SpottedSpurge")
    ap.add_argument("--prop-dir", default=None,
                    help="OWL red proposals dir (default results/framework/owl_red_proposals_species/<species>)")
    ap.add_argument("--gt-dir", default=None,
                    help="GT labels for the gate (default: the labels/ beside --images, "
                         "else downloads/cottonweeddet12/valid/labels)")
    ap.add_argument("--images", default=None,
                    help="images the proposals were made on (default: the "
                         "manifest's target_dir)")
    ap.add_argument("--photos", choices=["cwd12", "other"], default=None,
                    help="are --images cwd12 photographs? (default: cwd12 for a "
                         "cwd12 image dir; required otherwise)")
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
    species = cli_species(args.species)   # v3.60.0: no legacy-only label
    if species is None:
        print(f"FATAL: --species {args.species!r} is not a cwd12 species "
              f"({', '.join(CWD12_SPECIES)})")
        return 2
    args.species = species

    prop_dir = Path(args.prop_dir) if args.prop_dir else \
        default_proposals_dir(args.species)

    print(f"[owl-upload-gate] species={args.species} prop_dir={prop_dir}")
    if not prop_dir.is_dir():
        print(f"FATAL: proposals dir not found: {prop_dir} — run owl_preannotate first")
        return 2

    manifest = read_manifest(prop_dir)
    if manifest.get("vocabulary") != "species" or manifest.get("species") != species:
        print(f"FATAL: {prop_dir} has no {PROPOSALS_MANIFEST} naming species "
              f"{species} (found {manifest.get('species')!r}). Proposal dirs from "
              f"before v3.60.0 are named by legacy labels; re-run owl_preannotate "
              f"--species {species}.")
        return 2
    if args.images is None:
        args.images = manifest.get("target_dir")
        if not args.images:
            print(f"FATAL: {PROPOSALS_MANIFEST} records no target_dir; pass --images")
            return 2
    if args.photos is None:
        if not _is_cwd12_image_dir(args.images):
            print(f"FATAL: --images {args.images} is not a cwd12 image dir; "
                  f"pass --photos cwd12 or --photos other")
            return 2
        args.photos = "cwd12"
    if not Path(args.images).is_dir():
        print(f"FATAL: --images dir not found: {args.images}")
        return 2
    # v3.60.0: score against the labels of the images the proposals were
    # made on when --gt-dir is not given (cwd12 valid for the default run)
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        sib = Path(args.images).parent / "labels"
        gt_dir = sib if (Path(args.images).name == "images" and sib.is_dir()) else \
            REPO / "downloads" / "cottonweeddet12" / "valid" / "labels"
    n_paired, n_props = _stem_overlap(args.images, prop_dir)
    print(f"[owl-upload-gate] images={args.images}: {n_paired} of {n_props} "
          f"proposal files pair with an image")
    if n_paired == 0:
        print(f"FATAL: no image in {args.images} has a proposal file in {prop_dir}; "
              f"every image would go up as Unlabeled. Pass the --images the "
              f"proposals were made on.")
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
        "--single-species", species,
        "--photos", args.photos,
    ]
    print(f"[owl-upload-gate] PASS → uploading: {' '.join(argv)}")
    return subprocess.call(argv, cwd=str(REPO))


if __name__ == "__main__":
    sys.exit(main())
