#!/bin/bash
#SBATCH --job-name=ctlchainS
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/framework/ctl_chainS_%x_a%a_%j.out
#
# Does the round loop's warm-start chain cost accuracy, and does truncating the
# cosine schedule cost accuracy? Two factors, held against round 15's own
# measured cell.
#
#   arm            start                       schedule                      cell
#   A  cold+complete   yolo26x.pt (pretrained)   epochs=30 patience=30 no time= new
#   B  warm+complete   round 14's best.pt        epochs=30 patience=30 no time= new
#   -  (round 15)      round 14's best.pt        epochs=60 patience=20 time=10.0 0.5607
#
# The round recipe's `time=10.0` is not a harmless safety cap: ultralytics
# rewrites self.epochs from it after every epoch and rebuilds the scheduler, so
# the cosine horizon is re-planned against a clock and patience=20 ends the run
# before it. Both arms drop it.
#
# Everything else is held: the SAME merged dataset directory round 15 trained on
# (not a re-merge -- a re-merge could differ if the registry moved), the same
# sealed holdout, seed 101, imgsz 640, batch -1, cos_lr, mosaic 1.0, mixup 0.1.
# optimizer stays "auto" on purpose: every round ran at auto's lr=0.01, so
# forcing lr0 would add a third factor. That the saved args.yaml records the
# discarded lr0=0.001 is a record-keeping defect, reported separately.
#
# This script NEVER writes registry["last_mega_weights"]. The campaign's chain
# must not pick up a control run as its next base.
set -uo pipefail

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO" || exit 1

ARM_INDEX="${SLURM_ARRAY_TASK_ID:-1}"
# Both arms were single runs, so the +0.0287 cold-vs-warm gap is quoted against a
# seed std measured on a DIFFERENT recipe. CTL_SEED repeats each arm so the gap
# gets an error bar of its own.
CTL_SEED="${CTL_SEED:-101}"
case "$ARM_INDEX" in
  1) ARM=A; BASE="$REPO/yolo26x.pt";                                                              EPOCHS=30; PATIENCE=30 ;;
  2) ARM=B; BASE="$REPO/results/framework/mega_iterrnd14_train/job45592739/weights/best.pt";       EPOCHS=30; PATIENCE=30 ;;
  *) echo "FATAL: unknown arm index $ARM_INDEX" >&2; exit 1 ;;
esac
ARM="${ARM}s${CTL_SEED}"

DATA="$REPO/results/framework/merged_iterrnd15_train/data.yaml"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nojob}}"

[ -f "$DATA" ] || { echo "FATAL: data yaml missing: $DATA" >&2; exit 1; }
[ -f "$BASE" ] || { echo "FATAL: base weights missing: $BASE" >&2; exit 1; }

echo "=== ctl_chain arm=$ARM job=${SLURM_JOB_ID:-none} key=$JOB_KEY $(date) ==="
echo "[cfg] base=$BASE"
echo "[cfg] data=$DATA epochs=$EPOCHS patience=$PATIENCE imgsz=640 seed=101"

if [ -d "$REPO/weed_llm_benchmark/weed_optimizer_framework" ]; then
    rsync -a --delete \
        "$REPO/weed_llm_benchmark/weed_optimizer_framework/" \
        "$REPO/weed_optimizer_framework/" \
        && echo "[sync] outer package refreshed from the tracked nested copy"
fi

source /jet/home/byler/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate bench || { echo "FATAL: conda activate bench failed" >&2; exit 1; }
echo "python: $(which python)  $(python -V 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

ARM="$ARM" CTL_SEED="$CTL_SEED" BASE="$BASE" DATA="$DATA" EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" \
JOB_KEY="$JOB_KEY" REPO="$REPO" python -u - <<'PYEOF'
import csv, json, os, time, traceback
from pathlib import Path

ARM = os.environ["ARM"]; SEED = int(os.environ["CTL_SEED"]); BASE = os.environ["BASE"]; DATA = os.environ["DATA"]
EPOCHS = int(os.environ["EPOCHS"]); PATIENCE = int(os.environ["PATIENCE"])
JOB_KEY = os.environ["JOB_KEY"]; REPO = os.environ["REPO"]
FW = Path(REPO) / "results" / "framework"
PROJ = FW / "ctl_chain"
dest = FW / ("ctl_chain_arm%s_%s.json" % (ARM, JOB_KEY))

rec = {"arm": ARM, "base_weights": BASE, "data_yaml": DATA,
       "epochs": EPOCHS, "patience": PATIENCE, "seed": SEED, "imgsz": 640,
       "job_key": JOB_KEY, "task_job_id": os.environ.get("SLURM_JOB_ID"),
       "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running",
       "question": ("cold vs warm start and complete vs truncated cosine, "
                    "against round 15's measured 0.5607 (warm + truncated)")}
FW.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(rec, indent=1))
print("[ctl] wrote started record:", dest)

from ultralytics import YOLO

try:
    model = YOLO(BASE)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=-1,
        imgsz=640,
        device=0,
        project=str(PROJ),
        name="arm%s_%s" % (ARM, JOB_KEY),
        patience=PATIENCE,
        lr0=0.001,          # recorded; optimizer="auto" is what actually decides
        workers=4,
        verbose=False,
        save_period=1,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,
        seed=SEED,
        deterministic=True,
        # NO time= HERE, deliberately. ultralytics/engine/trainer.py L542-544:
        #   if self.args.time:
        #       self.epochs = self.args.epochs = ceil(self.args.time*3600/mean_epoch_time)
        # A wall-clock cap rewrites the epoch count after every epoch and rebuilds
        # the cosine with it, which is the exact defect this control exists to
        # remove -- the campaign's own runs carry time=10.0 and never anneal.
        # SLURM's 12 h walltime is the backstop and save_period=1 leaves last.pt.
    )
    save_dir = str(model.trainer.save_dir)
    rec["save_dir"] = save_dir
    rows = list(csv.DictReader(open(os.path.join(save_dir, "results.csv"))))
    keys = [k.strip() for k in (rows[0].keys() if rows else [])]

    def col(name):
        for k in keys:
            if k.strip() == name:
                return k
        for k in keys:
            if name in k.replace(" ", ""):
                return k
        return None

    km = col("metrics/mAP50-95(B)"); k5 = col("metrics/mAP50(B)")
    kl = col("lr/pg0"); kt = col("train/box_loss"); kv = col("val/box_loss")

    def g(r, k):
        try:
            return round(float(r[k].strip()), 5)
        except Exception:
            return None

    curve = [{"e": i + 1, "map": g(r, km), "map50": g(r, k5), "lr": g(r, kl),
              "tbox": g(r, kt), "vbox": g(r, kv)} for i, r in enumerate(rows)]
    rec["curve"] = curve
    scored = [c for c in curve if c["map"] is not None]
    best = max(scored, key=lambda c: c["map"]) if scored else None
    rec["epochs_ran"] = len(curve)
    rec["best_epoch"] = (best or {}).get("e")
    rec["best_map50_95"] = (best or {}).get("map")
    rec["last_map50_95"] = (scored[-1]["map"] if scored else None)
    rec["cosine_completed"] = bool(len(curve) >= EPOCHS)
    rec["status"] = "done"
    rec["ok"] = rec["best_map50_95"] is not None
except Exception as exc:
    rec["status"] = "failed"
    rec["ok"] = False
    rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    rec["traceback"] = traceback.format_exc()[-3000:]
    print("[ctl] FAILED:", rec["error"])

rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
dest.write_text(json.dumps(rec, indent=1))
print("[ctl] arm %s: best=%s at epoch %s over %s epochs (cosine_completed=%s)"
      % (ARM, rec.get("best_map50_95"), rec.get("best_epoch"),
         rec.get("epochs_ran"), rec.get("cosine_completed")))
print("[ctl] wrote", dest)
PYEOF

echo "=== ctl_chain arm=$ARM finished rc=$? $(date) ==="
