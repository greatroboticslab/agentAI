#!/bin/bash
#SBATCH --job-name=v3_0_24_eval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/framework/v3_0_24_eval_%j.out

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

echo "=== v3.0.24 clean eval ==="
echo "Date: $(date)"

# Override BEST_PT for eval_v3_0_23.py to point at v3.0.24 weights
BEST_PT=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/mega_iterv3_0_24_clean/train/weights/best.pt
OUT_DIR=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/v3_0_24_eval

python - <<PYEOF
import os, json, shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
BEST_PT = "$BEST_PT"
OUT = "$OUT_DIR"
os.makedirs(OUT, exist_ok=True)

V3_NAMES = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
            "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
            "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]

def stage(name, src_yaml, split):
    src = yaml.safe_load(open(src_yaml))
    raw = src["names"]
    src_names = [raw[i] for i in sorted(raw.keys())] if isinstance(raw, dict) else list(raw)
    mapping = {i: V3_NAMES.index(n) for i, n in enumerate(src_names) if n in V3_NAMES}
    src_dir = Path(src_yaml).parent / split
    out = Path(OUT) / name
    if out.exists():
        shutil.rmtree(out)
    img_d = out / "images"; lbl_d = out / "labels"
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

print("Loading:", BEST_PT)
model = YOLO(BEST_PT)
out = {}
for nm, sp in [("cwd12_test", "test"), ("cwd12_valid", "valid")]:
    y = stage(nm, f"{REPO}/downloads/cottonweeddet12/data.yaml", sp)
    res = model.val(data=y, split="val", device=0, save=False, save_json=False,
                    plots=False, verbose=True, project=OUT, name=nm + "_run", exist_ok=True)
    out[nm] = {
        "n_images": len(list(Path(y).parent.glob("images/*"))),
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "per_class_mAP50_95": {V3_NAMES[i]: float(res.box.maps[i])
                                for i in range(len(V3_NAMES)) if i < len(res.box.maps)},
    }
    print(f"  {nm}: mAP50={out[nm]['mAP50']:.4f} mAP50-95={out[nm]['mAP50_95']:.4f}")

json.dump(out, open(f"{OUT}/v3_0_24_eval.json", "w"), indent=2)
print("\nSaved:", f"{OUT}/v3_0_24_eval.json")
PYEOF

echo "=== Done (exit=$?) ==="
echo "Date: $(date)"
