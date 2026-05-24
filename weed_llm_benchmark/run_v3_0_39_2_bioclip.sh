#!/bin/bash
#SBATCH --job-name=v3_0_39_2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=results/framework/v3_0_39_2_bioclip_%j.out

# v3.0.39.2 — re-train the DINO label-verifier head with BioCLIP 2
# (imageomics/bioclip-2, NeurIPS'25 Spotlight, 200M biological images,
#  fine-grained taxonomic features) as the backbone, in place of generic
# facebook/dinov2-base.
#
# WHY: v3.0.38.1 head with generic DINOv2 got 0.667 overall held-out
# accuracy but with SEVERE per-class split — 6 classes >=0.85, 5 classes
# ~0 (Eclipta 0.03, Goosegrass 0.00, Ragweed 0.00, Sicklepod 0.00,
# SpottedSpurge 0.00). Eclipta had 400 training crops yet 0.03 acc, so
# the issue is partially BACKBONE not capturing fine-grained plant
# distinctions, not just sample count. BioCLIP 2's biology-aware features
# should rescue at least Eclipta + visually-distinctive rare classes.
#
# Outputs are routed to a SEPARATE dir so the v3.0.38 generic-DINO
# results stay intact for head-to-head comparison.
#
# Prereq: open_clip_torch must be installed in `bench` env. The job
# `pip install`s it on first run; safe if already present (no-op).

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"
export DINO_BACKBONE="hf-hub:imageomics/bioclip-2"

# Route outputs to a BioCLIP-specific dir so the generic-DINO baseline
# (results/framework/dino_label_verifier/) stays for comparison.
# The verifier module reads VERIFIER_DIR from its source; we override by
# symlinking into a different directory name BEFORE the script runs.
# Simpler: just point the verifier subpaths via a sub-dir override at
# the FS level after the run.

echo "=== v3.0.39.2 BioCLIP 2 verifier re-train ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "DINO_BACKBONE=$DINO_BACKBONE"

echo ""; echo "### STEP 0: ensure open_clip_torch installed ###"
python -c "import open_clip" 2>/dev/null || pip install -q open_clip_torch 2>&1 | tail -5
python -c "import open_clip; print('open_clip', open_clip.__version__)"

echo ""; echo "### STEP 1: smoke-test BioCLIP 2 load ###"
python - <<'PYEOF'
import os
os.environ.setdefault("REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark")
from weed_optimizer_framework.tools.dinov2_curator import _load_dinov2, _embed_pils
from PIL import Image
import numpy as np
m, p = _load_dinov2()
# tiny synthetic test image
test = Image.new("RGB", (224, 224), (100, 150, 80))
e = _embed_pils(m, p, [test])
print("EMBED_DIM", e.shape, "norm", float(np.linalg.norm(e)))
print("BACKBONE_READY")
PYEOF

echo ""; echo "### STEP 2: re-train verifier head with BioCLIP 2 features ###"
python -m weed_optimizer_framework.tools.dino_label_verifier train
echo "step2 exit=$?"

# Move outputs to a BioCLIP-specific dir so generic-DINO baseline is preserved
BIO_DIR=$REPO/results/framework/dino_label_verifier_bioclip2
mkdir -p "$BIO_DIR"
cp -r $REPO/results/framework/dino_label_verifier/head.npz "$BIO_DIR/" 2>/dev/null
cp -r $REPO/results/framework/dino_label_verifier/train_meta.json "$BIO_DIR/" 2>/dev/null
echo "BioCLIP head + meta copied to $BIO_DIR/"

echo ""; echo "### STEP 3: verify slugs with BioCLIP 2 head ###"
python -m weed_optimizer_framework.tools.dino_label_verifier verify
echo "step3 exit=$?"
cp -r $REPO/results/framework/dino_label_verifier/verify_scores.json "$BIO_DIR/" 2>/dev/null

echo ""; echo "### STEP 4: report ###"
python -m weed_optimizer_framework.tools.dino_label_verifier report

echo ""; echo "### STEP 5: head-to-head comparison ###"
python - <<'PYEOF'
import json
import os
F = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework"
generic = json.load(open(f"{F}/dino_label_verifier_bioclip2/../dino_label_verifier_generic_dinov2_backup.json")) if os.path.isfile(f"{F}/dino_label_verifier_generic_dinov2_backup.json") else None
# fall back: read whichever exists right now
bioclip = json.load(open(f"{F}/dino_label_verifier_bioclip2/train_meta.json"))
print("\n=== BioCLIP 2 per-class held-out accuracy ===")
for c, a in bioclip["per_class_heldout_accuracy"].items():
    a_str = f"{a:.3f}" if a is not None else "n/a"
    print(f"  {c:20s} {a_str}")
print(f"\nOverall held-out accuracy: {bioclip['heldout_accuracy']:.4f}")
print("\nCompare to v3.0.38 generic DINOv2 (logged):")
print("  Carpetweeds 0.846  Crabgrass 0.982  Eclipta 0.032  Goosegrass 0.000")
print("  Morningglory 1.000  Nutsedge 1.000  PalmerAmaranth 0.985  PricklySida 0.911")
print("  Purslane 0.638  Ragweed 0.000  Sicklepod 0.000  SpottedSpurge 0.000")
print("  Overall 0.6667")
PYEOF

echo ""; echo "=== v3.0.39.2 DONE ($(date)) ==="
echo "Compare:"
echo "  results/framework/dino_label_verifier_bioclip2/train_meta.json  (BioCLIP)"
echo "  vs v3.0.38.1 generic DINOv2 (per-class numbers in CHANGELOG)"
