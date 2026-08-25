#!/bin/bash
#SBATCH --job-name=s3_tier
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=14:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_tier_%a_%j.out
#
# S3 tier ladder — the campaign's signature figure, and the one question a
# platform user actually has: *how much harvested data can I add to my clean
# in-domain data before it starts hurting?*
#
# Built entirely from the already-canonicalised M1 merge (nc=100, dedup and
# holdout-guard already applied), so no class remapping is involved and every
# tier sits on identical footing with the published full-merge point:
#   core        = the in-domain cottonweed_* portion of that corpus
#   core + N    = plus N randomly sampled harvested images (seeded, reproducible)
#   full        = 55,690 -> 0.6032 +/- 0.0046 (already measured, jobs 44234060)
# val is the same canonical-mapped 1,977-image sealed holdout M1 used.
#
# Submit: sbatch --array=0-3 run_s3_tier_ladder.sh      (0=core, 1=+5k, 2=+15k, 3=+40k)
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
SRC=$REPO/results/framework/merged_iterm1_raw_s101
ADDS=(0 5000 15000 40000); ADD="${ADDS[${SLURM_ARRAY_TASK_ID:-0}]}"
TDIR=$REPO/results/framework/s3_tiers/tier_$ADD
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
echo "=== tier core+$ADD  $(date) ==="; nvidia-smi --query-gpu=name --format=csv,noheader
python -u - <<PY
import csv, json, os, random, shutil, time
SRC="$SRC"; TDIR="$TDIR"; ADD=int("$ADD"); REPO="$REPO"
src_img=os.path.join(SRC,"train","images"); src_lbl=os.path.join(SRC,"train","labels")
names=sorted(os.listdir(src_img))
core=[n for n in names if n.startswith("cottonweed_")]
rest=[n for n in names if not n.startswith("cottonweed_")]
random.Random(1234).shuffle(rest)                  # seeded: the same tier is the same images
pick=core+rest[:ADD]
print("[tier] core=%d harvested_added=%d total=%d (pool of %d harvested)"
      % (len(core), min(ADD,len(rest)), len(pick), len(rest)))
for sub in ("images","labels"):
    d=os.path.join(TDIR,"train",sub)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
for n in pick:
    stem=os.path.splitext(n)[0]
    os.symlink(os.path.join(src_img,n), os.path.join(TDIR,"train","images",n))
    lp=os.path.join(src_lbl, stem+".txt")
    if os.path.exists(lp):
        os.symlink(lp, os.path.join(TDIR,"train","labels",stem+".txt"))
# reuse the merge's own class list and its staged sealed holdout as val
import re
src_yaml=open(os.path.join(SRC,"data.yaml")).read()
m=re.search(r"names:\s*(\[.*?\]|\n(?:\s*-\s*.*\n)+)", src_yaml, re.S)
open(os.path.join(TDIR,"data.yaml"),"w").write(
    "path: %s\ntrain: train/images\nval: %s\n%s\n" % (TDIR, os.path.join(SRC,"cwd12_holdout","images"),
                                                      "names: "+m.group(1).strip() if m else ""))
print("[tier] yaml written")
from ultralytics import YOLO
t0=time.time()
mdl=YOLO(REPO+"/yolo11n.pt")                       # pretrained: the practical setting a user is in
mdl.train(data=os.path.join(TDIR,"data.yaml"), epochs=60, imgsz=640, batch=-1,
          seed=101, deterministic=True, patience=20, workers=4, device=0,
          project=REPO+"/results/framework/s3_tiers", name="run_%d"%ADD,
          verbose=False, cos_lr=True)
d=REPO+"/results/framework/s3_tiers/run_%d"%ADD
cands=[p for p in [d]+[d+str(i) for i in range(2,6)] if os.path.exists(os.path.join(p,"results.csv"))]
rows=list(csv.DictReader(open(os.path.join(cands[-1],"results.csv")))) if cands else []
col=[k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
vals=[float(r[col[0]]) for r in rows if col and r.get(col[0])]
out={"tier_added":ADD,"train_images":len(pick),"core_images":len(core),
     "model":"YOLO11n (COCO-pretrained)","epochs_cap":60,"seed":101,
     "epochs_ran":len(rows),"best_map50_95":round(max(vals),4) if vals else None,
     "ok":bool(vals),"hours":round((time.time()-t0)/3600,2),
     "job":os.environ.get("SLURM_JOB_ID"),"ended":time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(out, open(REPO+"/results/framework/s3_tier_%d.json"%ADD,"w"), indent=1)
print("[tier]", out)
PY
