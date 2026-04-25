#!/bin/bash
#SBATCH --job-name=v3_0_24_clean
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_24_clean_%j.out

# v3.0.24: One-shot fresh training, NOT in the auto-chain.
# Skips yolo_autolabel data (class_id=0 contamination bug from v3.0.23 audit)
# and uses 100 epochs + imgsz=1024 to match the v3.0.6 YOLO11n baseline that
# achieved mAP50-95=0.865 on cottonweeddet12. Goal: prove the bug fix lands
# us back at or above baseline performance, then iterate to add autolabel
# data back in with proper per-dataset class assignment in v3.0.25+.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: conda activate failed" >&2
    exit 2
fi
set +e

cd /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/ocean/projects/cis240145p/byler/hf_cache

echo "=== v3.0.24 clean fresh training ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Force fresh start (don't load v3.0.23 best.pt — it has the contamination)
# Use the mega_trainer directly, not the agent loop, so we get one clean run.
python - <<'PYEOF'
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from weed_optimizer_framework.tools.mega_trainer import train_yolo_mega

strategy = {
    "include_autolabel": False,  # v3.0.24 critical fix
    "epochs": 100,
    "imgsz": 1024,
    "batch_size": 5,             # V100-32GB at 1024 fits ~5
    "lr": 0.001,
    "patience": 50,
    "workers": 4,
    "fresh_start": True,         # don't load v3.0.23 best.pt (contaminated)
}
best_pt, summary = train_yolo_mega(strategy, iteration="v3_0_24_clean")
print("\n=== TRAIN COMPLETE ===")
print("best_pt:", best_pt)
import json
print(json.dumps(summary, indent=2))
PYEOF

EXIT_CODE=$?
echo "=== Done (exit=$EXIT_CODE) ==="
echo "Date: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Auto-running clean eval on cottonweeddet12 test+valid..."
    # Update eval_v3_0_23.py to point at the new best.pt
    BEST_PT=$(ls -t results/framework/mega_iterv3_0_24_clean/train*/weights/best.pt 2>/dev/null | head -1)
    echo "Eval target: $BEST_PT"
    if [ -n "$BEST_PT" ] && [ -f "$BEST_PT" ]; then
        BEST_PT="$BEST_PT" python - <<'PYEOF'
import os, json
from pathlib import Path
import yaml
from ultralytics import YOLO

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
BEST_PT = os.environ["BEST_PT"]
OUT = f"{REPO}/results/v3_0_24_eval"
os.makedirs(OUT, exist_ok=True)

V3_NAMES = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
            "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
            "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]

# Reuse the eval staging from v3.0.23 if it exists; otherwise rebuild.
def stage(name, src_yaml, split):
    src = yaml.safe_load(open(src_yaml))
    raw_names = src["names"]
    if isinstance(raw_names, dict):
        src_names = [raw_names[i] for i in sorted(raw_names.keys())]
    else:
        src_names = list(raw_names)
    mapping = {i: V3_NAMES.index(n) for i, n in enumerate(src_names) if n in V3_NAMES}
    src_dir = Path(src_yaml).parent / split
    out = Path(OUT) / name
    img_d = out / "images"; lbl_d = out / "labels"
    if out.exists():
        import shutil; shutil.rmtree(out)
    img_d.mkdir(parents=True); lbl_d.mkdir(parents=True)
    imgs = sorted(list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")))
    n_lbl = 0
    for img in imgs:
        try: os.symlink(img.resolve(), img_d / img.name)
        except FileExistsError: pass
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            with open(lbl) as f: lines = f.readlines()
            with open(lbl_d / (img.stem + ".txt"), "w") as g:
                for line in lines:
                    p = line.strip().split()
                    if not p: continue
                    try: orig = int(p[0])
                    except ValueError: continue
                    if orig in mapping:
                        g.write(f"{mapping[orig]} {' '.join(p[1:])}\n")
            n_lbl += 1
        else:
            (lbl_d / (img.stem + ".txt")).touch()
    yaml_out = out / "data.yaml"
    yaml.safe_dump({"train": str(img_d), "val": str(img_d),
                    "nc": len(V3_NAMES), "names": V3_NAMES}, open(yaml_out, "w"))
    print(f"  staged {name}: {len(imgs)} imgs, {n_lbl} labels")
    return str(yaml_out)

model = YOLO(BEST_PT)
print("Loaded:", BEST_PT)
out = {}
for name, src_split in [("cwd12_test", "test"), ("cwd12_valid", "valid")]:
    y = stage(name, f"{REPO}/downloads/cottonweeddet12/data.yaml", src_split)
    res = model.val(data=y, split="val", device=0, save=False, save_json=False,
                    plots=False, verbose=True, project=OUT, name=name+"_run", exist_ok=True)
    out[name] = {
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "per_class_mAP50_95": {V3_NAMES[i]: float(res.box.maps[i])
                                for i in range(len(V3_NAMES)) if i < len(res.box.maps)},
    }
    print(f"  {name}: mAP50={out[name]['mAP50']:.4f} mAP50-95={out[name]['mAP50_95']:.4f}")

json.dump(out, open(f"{OUT}/v3_0_24_eval.json", "w"), indent=2)
print("Saved:", f"{OUT}/v3_0_24_eval.json")
PYEOF
    fi
fi

echo "=== ALL DONE ==="
echo "Date: $(date)"
