#!/bin/bash
#SBATCH --job-name=s3_tierb
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45G
#SBATCH --time=14:00:00
#SBATCH --output=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/s3_tierb_%a_%j.out
#
# S3 tier ladder, arm B — the SAME ladder with ONE variable changed: what class
# id a harvested box carries.
#
# Arm A (run_s3_tier_ladder.sh, jobs 44397807_[0-3], already measured) gives each
# harvested dataset a class id from `md5(slug) %% 88`, i.e. the label a harvested
# box carries is a hash of the website it came from. Two photos of the same weed
# from two sources get two different classes; five of the 22 merged datasets --
# 14,902 images, 33%% of the harvested corpus -- collide onto a shared id.
#
# Arm B puts every harvested box in ONE class (12). nc stays 100 so the head size
# is held constant and this measures the class space, not the head. The core, the
# add-on images (same seed-1234 shuffle), the holdout, the model, the epochs and
# the seed are byte-identical to arm A.
#
# The question is not "does adding data raise mAP" -- arm A already answered that
# with a flat-to-negative ladder. It is whether the ladder's SLOPE changes when
# the harvested labels stop being noise. A flatter or rising arm B is the finding.
#
#   arm A   +0 0.8636  +5k 0.8599  +15k 0.8614  +40k 0.8436   (measured)
#   arm B   +0 ?       +5k ?       +15k ?       +40k ?
#
# Submit: sbatch --array=0-3 run_s3_tier_ladder_b.sh
#
# ---- original header of arm A, unchanged, below ----
# S3 tier ladder — the campaign's signature figure, and the one question a
# platform user actually has: *how much harvested data can I add to my clean
# in-domain data before it starts hurting?*
#
# Built entirely from the already-canonicalised M1 merge (nc=100, dedup and
# holdout-guard already applied), so no class remapping is involved and every
# tier sits on identical footing with the published full-merge point:
#   core        = the FULL cwd12 train split, remapped to canonical ids
#   core + N    = plus N randomly sampled harvested images (seeded, reproducible)
#   full        = 55,690 -> 0.6032 +/- 0.0046 (already measured, jobs 44234060)
# val is the same canonical-mapped 1,977-image sealed holdout M1 used.
#
# v3.23.1 CORRECTION: the first ladder took its core from the merged corpus's own
# cottonweed_* files and that core is CRIPPLED — 2,918 images / 4,175 instances with
# Goosegrass at 44 and SpottedSpurge at 59, because the merge's dedup and holdout-stem
# filter strip the rare classes hardest. It scored 0.5601 with ZERO harvested images
# added, which is why the "harvested data costs 0.27" reading was wrong: the loss was
# already there before any harvested image arrived. The core is now the real cwd12
# train split (3,671 images / 6,131 instances), remapped from cwd12's original class
# order to the canonical one, so the ladder measures what it claims to.
#
# Submit: sbatch --array=0-3 run_s3_tier_ladder.sh      (0=core, 1=+5k, 2=+15k, 3=+40k)
set -uo pipefail
REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
SRC=$REPO/results/framework/merged_iterm1_raw_s101
ADDS=(0 5000 15000 40000); ADD="${ADDS[${SLURM_ARRAY_TASK_ID:-0}]}"
TDIR=$REPO/results/framework/s3_tiers/v2b_tier_$ADD
source /jet/home/byler/miniconda3/etc/profile.d/conda.sh
conda activate bench
echo "=== tier ARM B (single aux class) core+$ADD  $(date) ==="; nvidia-smi --query-gpu=name --format=csv,noheader
python -u - <<PY
import csv, json, os, random, shutil, time
SRC="$SRC"; TDIR="$TDIR"; ADD=int("$ADD"); REPO="$REPO"
src_img=os.path.join(SRC,"train","images"); src_lbl=os.path.join(SRC,"train","labels")
names=sorted(os.listdir(src_img))
rest=[n for n in names if not n.startswith("cottonweed_")]   # harvested add-on pool
random.Random(1234).shuffle(rest)                  # seeded: the same tier is the same images
for sub in ("images","labels"):
    d=os.path.join(TDIR,"train",sub)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

# --- core: the real cwd12 train split, class ids remapped orig -> canonical ---
V3=["Carpetweeds","Crabgrass","PalmerAmaranth","PricklySida","Purslane","Ragweed",
    "Sicklepod","SpottedSpurge","Eclipta","Goosegrass","Morningglory","Nutsedge"]
ORIG=["Carpetweeds","Crabgrass","Eclipta","Goosegrass","Morningglory","Nutsedge",
      "PalmerAmaranth","PricklySida","Purslane","Ragweed","Sicklepod","SpottedSpurge"]
O2C={i: V3.index(n) for i, n in enumerate(ORIG)}
cw=os.path.join(REPO,"downloads","cottonweeddet12","train")
ncore=0
for f in sorted(os.listdir(os.path.join(cw,"images"))):
    stem,ext=os.path.splitext(f)
    lp=os.path.join(cw,"labels",stem+".txt")
    if not os.path.exists(lp): continue
    os.symlink(os.path.join(cw,"images",f), os.path.join(TDIR,"train","images","cwd12_"+f))
    out=[]
    for line in open(lp):
        p=line.split()
        if len(p)>=5 and int(float(p[0])) in O2C:
            out.append("%d %s" % (O2C[int(float(p[0]))], " ".join(p[1:5])))
    open(os.path.join(TDIR,"train","labels","cwd12_"+stem+".txt"),"w").write("\n".join(out)+"\n")
    ncore+=1
# --- add-on: N harvested images from the canonicalised merge ---
# ARM B: the only line that differs from arm A. Arm A symlinks the harvested
# label file untouched, so every box keeps the md5(slug)-derived class id the
# merge gave it. Here every harvested box is rewritten to a single shared class,
# so the model is asked "is this a weed" instead of "which website is this from".
SINGLE_AUX = 12
n_rewritten = 0
n_boxes = 0
for n in rest[:ADD]:
    stem=os.path.splitext(n)[0]
    os.symlink(os.path.join(src_img,n), os.path.join(TDIR,"train","images",n))
    lp=os.path.join(src_lbl, stem+".txt")
    if os.path.exists(lp):
        out=[]
        for line in open(lp):
            p=line.split()
            if len(p)>=5:
                out.append("%d %s" % (SINGLE_AUX, " ".join(p[1:5])))
        open(os.path.join(TDIR,"train","labels",stem+".txt"),"w").write("\n".join(out)+"\n")
        n_rewritten += 1
        n_boxes += len(out)
print("[tier][armB] rewrote %d harvested label files, %d boxes -> class %d"
      % (n_rewritten, n_boxes, SINGLE_AUX))
core=[1]*ncore; pick=[1]*(ncore+min(ADD,len(rest)))
print("[tier] core=%d (real cwd12 train, remapped) harvested_added=%d total=%d"
      % (ncore, min(ADD,len(rest)), len(pick)))
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
mdl.train(data=os.path.join(TDIR,"data.yaml"), epochs=100, imgsz=640, batch=-1,
          seed=101, deterministic=True, patience=20, workers=4, device=0,
          project=REPO+"/results/framework/s3_tiers", name="v2b_run_%d"%ADD,
          verbose=False, cos_lr=True)
d=REPO+"/results/framework/s3_tiers/v2b_run_%d"%ADD
cands=[p for p in [d]+[d+str(i) for i in range(2,6)] if os.path.exists(os.path.join(p,"results.csv"))]
rows=list(csv.DictReader(open(os.path.join(cands[-1],"results.csv")))) if cands else []
col=[k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
vals=[float(r[col[0]]) for r in rows if col and r.get(col[0])]
out={"arm":"B_single_aux_class","tier_added":ADD,"train_images":len(pick),"core_images":len(core),
     "model":"YOLO11n (COCO-pretrained)","harvested_label_files":n_rewritten,"harvested_boxes":n_boxes,"aux_class":SINGLE_AUX,"epochs_cap":100,"seed":101,
     "epochs_ran":len(rows),"best_map50_95":round(max(vals),4) if vals else None,
     "ok":bool(vals),"hours":round((time.time()-t0)/3600,2),
     "job":os.environ.get("SLURM_JOB_ID"),"ended":time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(out, open(REPO+"/results/framework/s3_tier_v2b_%d.json"%ADD,"w"), indent=1)
print("[tier]", out)
PY
