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
import importlib.util, json, os, statistics, time
from ultralytics import YOLO
REPO="/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"

def load_cwd12_species():
    # Loaded by file (stdlib only): the git-tracked nested copy first, then the
    # outer one, without importing the package's heavy __init__.
    for p in (REPO+"/weed_llm_benchmark/weed_optimizer_framework/tools/cwd12_species.py",
              REPO+"/weed_optimizer_framework/tools/cwd12_species.py"):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("cwd12_species", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("[bm] FATAL: cwd12_species.py not found")

cs = load_cwd12_species()
# v3.60.0: the sealed checkpoints and cwd12_sealed.yaml are in cwd12 label-id
# order, so per-class results are named by the species of each id. The list here
# used to be the legacy labels, which were wrong for every id but PricklySida.
NAMES = list(cs.CWD12_SPECIES)

def head_species(names):
    seq = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    return list(cs.species_names_for(seq))

per_seed={}
for seed in (101,102,103):
    w=f"{REPO}/results/framework/s3_yolo11n/s{seed}/weights/best.pt"
    if not os.path.exists(w):
        print("[bm] missing", w); continue
    model=YOLO(w)
    if head_species(model.names)[:12] != NAMES:
        print("[bm] WARNING: %s does not name the cwd12 species in id order (%s); "
              "per-class keys below assume it does" % (w, dict(model.names)))
    r=model.val(data=REPO+"/cwd12_sealed.yaml", imgsz=640, device=0,
                  workers=4, verbose=False, plots=False,
                  project=REPO+"/results/framework/s3_bmeval", name=f"s{seed}")
    # `maps` is a numpy array; `x or []` evaluates its truth value and raises
    # "truth value of an array ... is ambiguous". Convert explicitly.
    maps = getattr(r.box, "maps", None)
    ap = [] if maps is None else [float(v) for v in list(maps)]
    per_seed[seed]={"map50_95":round(float(r.box.map),4),"map50":round(float(r.box.map50),4),
                    "n_classes_reported":len(ap),
                    "per_class":{NAMES[i]: round(v,4) for i,v in enumerate(ap) if i<len(NAMES)}}
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
     "per_species_keys":"species by cwd12 id (cwd12_species.CWD12_SPECIES)",
     "evaluated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
     "job":os.environ.get("SLURM_JOB_ID")}
def output_path(path):
    # v3.60.0: a file without per_species_keys is the legacy-keyed artifact the
    # model card cites; keep it and write the species result beside it.
    try:
        old = json.load(open(path))
    except (OSError, ValueError):
        return path
    if isinstance(old, dict) and "per_species_keys" not in old:
        return path[:-len(".json")] + "_species.json"
    return path

out_path = output_path(REPO+"/results/framework/s3_best_model_eval.json")
json.dump(out, open(out_path,"w"), indent=1)
print("[bm] wrote", out_path)
print("[bm] weakest:", sorted(cls.items(), key=lambda kv: kv[1]["mean"])[:3])
print("[bm] strongest:", sorted(cls.items(), key=lambda kv: -kv[1]["mean"])[:3])
PY
