#!/bin/bash
#SBATCH --job-name=s3_bmeval
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_bmeval_%j.out
#
# S3's last gate item — the best-model card needs more than a headline number: the
# per-species breakdown across all three seeds, so the card can state where the model
# is weak as precisely as where it is strong. Evaluates the three sealed YOLO11n
# checkpoints on the same 1,977-image holdout they were validated against.
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
nvidia-smi --query-gpu=name --format=csv,noheader
python -u - <<'PY'
import json, os, statistics, time
from ultralytics import YOLO
REPO="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"
NAMES=['Carpetweeds','Crabgrass','Eclipta','Goosegrass','Morningglory','Nutsedge',
       'PalmerAmaranth','PricklySida','Purslane','Ragweed','Sicklepod','SpottedSpurge']
per_seed={}
for seed in (101,102,103):
    w=f"{REPO}/results/framework/s3_yolo11n/s{seed}/weights/best.pt"
    if not os.path.exists(w):
        print("[bm] missing", w); continue
    r=YOLO(w).val(data=REPO+"/cwd12_sealed.yaml", imgsz=640, device=0,
                  workers=4, verbose=False, plots=False,
                  project=REPO+"/results/framework/s3_bmeval", name=f"s{seed}")
    ap=list(getattr(r.box, "maps", []) or [])
    per_seed[seed]={"map50_95":round(float(r.box.map),4),"map50":round(float(r.box.map50),4),
                    "per_class":{NAMES[i]: round(float(v),4) for i,v in enumerate(ap) if i<len(NAMES)}}
    print("[bm] seed %d mAP50-95=%.4f mAP50=%.4f" % (seed, r.box.map, r.box.map50))
cls={}
for n in NAMES:
    vals=[s["per_class"].get(n) for s in per_seed.values() if s["per_class"].get(n) is not None]
    if vals:
        cls[n]={"mean":round(statistics.mean(vals),4),
                "std":round(statistics.stdev(vals),4) if len(vals)>1 else 0.0,"n":len(vals)}
overall=[s["map50_95"] for s in per_seed.values()]
out={"model":"YOLO11n (COCO-pretrained), sealed cwd12 protocol",
     "checkpoints":[f"results/framework/s3_yolo11n/s{s}/weights/best.pt" for s in per_seed],
     "holdout":"cwd12 test+valid, 1,977 images, never trained on",
     "map50_95":{"mean":round(statistics.mean(overall),4),
                 "std":round(statistics.stdev(overall),4) if len(overall)>1 else 0.0,
                 "n":len(overall),"per_seed":overall},
     "per_species_map50_95":cls,
     "evaluated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
     "job":os.environ.get("SLURM_JOB_ID")}
json.dump(out, open(REPO+"/results/framework/s3_best_model_eval.json","w"), indent=1)
print("[bm] weakest:", sorted(cls.items(), key=lambda kv: kv[1]["mean"])[:3])
print("[bm] strongest:", sorted(cls.items(), key=lambda kv: -kv[1]["mean"])[:3])
PY
