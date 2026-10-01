"""Protocol v3: the recipe table, the capacity arms and their cost estimates
(docs/CONTINUOUS_LOOP.md §5.1, §2.6 L-4, L-6; docs/INCREMENTAL_PROTOCOL.md,
"Protocol v3").

Nothing here trains, scores or imports Ultralytics, so a builder, the executor,
the autopilot and the tests can all import it on any machine.

The table. Every recipe is written out, key for key, in exp.json's form (every
inc.driver RECIPE_KEYS key but the per-run seed):

  cold   base and union runs: 100 epochs, SGD, lr0 0.01, warmup 3 epochs
         (warmup_bias_lr 0.1, Ultralytics' default, as the v1 builders write
         it), cosine to lrf 0.01;
  r0     the current full rehearsal: 30 epochs, lr0 0.002, warmup 1 epoch,
         cosine to lrf 0.01 (inc.pilot's 'full' recipe, unchanged);
  x1a    LR re-warm: 30 epochs, warmup 3 epochs, peak lr0 0.005, cosine to lrf
         0.01;
  x1b    LR re-warm: 50 epochs, peak lr0 0.01, warmup 3 epochs, cosine to lrf
         0.01.
Every recipe shares SGD, batch 32 (unless the arm sets its own, below),
momentum 0.937, weight decay 0.0005, close_mosaic 10, deterministic and
trainer 'full'. An incremental recipe's
warmup_bias_lr equals its lr0, the runner doc's convention for incremental
runs (without it the bias LR would start at Ultralytics' 0.1, far above a
fine-tune's peak). imgsz is the arm's (below). Freeze and LoRA are not in the
table: L-6 keeps them out of stream version 1 (pilot_v3: 3/7 agreement each,
against the 5/7 the survival rule needs), so a production run with either is a
deviation and is refused.

deviations(kind, recipe, arm) lists how a recipe departs from the table for a
run of that kind; a production run with any departure is refused by
inc2.train. Every key is compared except seed (set per run by the driver),
cache and workers (operational: they change where pixels come from, not what
is learned).

The arms (L-4). An arm is a detector and an input size: the cold runs start
from the arm's COCO checkpoint and every recipe of the experiment trains at the
arm's imgsz. The arm is recorded in exp.json ("arm": the record resolve_arm
returns, with the checkpoint's sha256) and pinned there, since an experiment
is defined once; inc2.train checks every cold run's init weights against it.

  n640   yolo11n.pt at 640 px   (continuity: every INC experiment so far)
  s640   yolo11s.pt at 640 px
  m640   yolo11m.pt at 640 px
  m832   yolo11m.pt at 832 px, batch 16   (measurement arm, 2026-09-30)
  s1024  yolo11s.pt at 1024 px             (measurement arm, 2026-09-30)
  l640     yolo11l.pt at 640 px            (measurement arm, 2026-10-01)
  y26m640  yolo26m.pt at 640 px            (measurement arm, 2026-10-01)
  y26l640  yolo26l.pt at 640 px            (measurement arm, 2026-10-01)

L-4's grid (GRID_ARMS: n640, s640, m640) varies capacity only: every arm
trains at 640 px and is scored by the locked scorer, which infers at 640 px
(inc/scorer.py IMGSZ; another size is a TEST- score). capacity-verdict chooses
among these alone.

The measurement arms (MEASURE_ARMS: m832, s1024) train at a larger input,
because on base_v2 the gap to 0.90 sits in the small prostrate weeds
(YOLO11m at 640: Carpetweed 0.736, SpottedSpurge 0.810, Purslane 0.828 test
AP50-95, R0's milestone read). They are built on base_v2 like the capacity
arms (3 seeds) but with finals dev and imageweeds only: they are built after
R0, at no milestone read, so they never read test (P10; inc2.baseline refuses
it). capacity-verdict --record lists them, and they are never a candidate of
the decision or a stream's arm. Their production scores are
still the locked scorer's, at 640 px: they measure what training at the
larger size gives under 640 px inference. Inferring at the arm's size needs
a scorer version whose imgsz is the arm's (R4); until then
inc2.scorer_native reads their finals at their own size, record only, by a
pre-registered rule against m640 at 640 (inc2.baseline rescore-native,
native-verdict; docs/CONTINUOUS_LOOP.md, group B, amendment 2026-10-01).

The box-quality arms (l640, y26m640, y26l640; 2026-10-01, docs/CONTINUOUS_LOOP.md
group B, amendment 'box-quality measurement arms'). On test, m640's
class-agnostic mAP50-95 is 0.8901 against its 12-class 0.8786: a perfect
species call on m640's boxes would still score about 0.89, so data that only
fixes species (s001 shrank dev's 12-class/agnostic gap from 0.017 to 0.007
and left agnostic dev where it was) cannot reach 0.90 on m640. Agnostic test
grew with capacity alone (n 0.8751, s 0.8842, m 0.8901). These arms ask
whether a larger (YOLO11l) or newer (YOLO26, NMS-free, small-target-aware
assignment) detector places better boxes. They train and score at 640 px
like the grid and are measurement arms all the same: built after R0, finals
dev and imageweeds, never a candidate of the capacity decision. They are
judged by the same pre-registered rule (native-verdict, read at their own
imgsz, which is 640), on dev only. Batch 32: the measured activations per
image (saved tensors of one training forward with the loss, fp32, CPU,
Ultralytics 8.4.37, nc 13) are 949 MiB for yolo11m, 1,232 for yolo11l
(1.30 x), 1,100 for yolo26m (1.16 x) and 1,374 for yolo26l (1.45 x); m640
peaked at 15.8 GB on a V100-32GB at batch 32 (Ultralytics GPU_mem, b_v2_m640
seed 0), so the largest is about 23-26 GB. Time, from m640's 2.68 h per base
run scaled by FLOPs (GPU-bound): l640 3.4 h, y26m640 2.9-3.2 h, y26l640
3.7-4.0 h (YOLO26 assigns for two heads in training), all under D26's 6.4 h
line. yolo11x and yolo26x (195.5 and 208.7 GFLOPs) would take about 8 h per
base run on one V100 and are left out.

Their estimates (2026-09-30), from the measured 640 px runs on the cluster
(one V100-32GB, cache ram, base_v2 6,811 images, 100 epochs; one base run
with its dev score): n640 about 1.2 h, s640 1.58 h (4.73 GPU-h for 3), m640
2.68 h (8.05 GPU-h for 3), i.e. 6.3, 8.3 and 14.2 ms per image-epoch. YOLO11m
at 640 is GPU-bound, so its time scales with the pixels:
  m832   14.2 ms x 1.69 = 24 ms: about 4.5 h per base run, 5.0 h with 10 % for
         batch 16; 3 seeds with finals 14-16 GPU-h;
  s1024  8.3 ms x 2.56 = 21 ms: at most about 4.0 h per base run (YOLO11s at
         640 is partly loader-bound, so the pixel ratio over-states it); 3
         seeds with finals 10-13 GPU-h.
Both are well under the 8 h cold walltime (inc.driver COLD_TIME_LIMIT) and
D26's 6.4 h line. GPU memory: the activations saved for the backward pass,
per image, measured with torch.autograd.graph.saved_tensors_hooks on
yolo11{s,m}.yaml at nc 13 in train mode with the loss (fp32, CPU, Ultralytics
8.4.22), are 392 MiB for s640, 833 MiB for m640, 1,014 MiB for s1024 and
1,410 MiB for m832. At batch 32 that is 26 GiB for m640 (which trains on one
V100-32GB under AMP), 32 GiB for s1024 (1.22 x m640: fits) and 44 GiB for
m832 (1.69 x m640: about 27-30 GB under AMP with the workspace, too close to
32 GB). m832 therefore trains at batch 16 (22 GiB, 0.85 x m640). Ultralytics
accumulates gradients to its nominal batch of 64 (accumulate = round(64 /
batch)) and scales the weight decay by batch x accumulate / 64, so batch 16
and batch 32 both take one optimizer step per 64 images with the same decay;
only BatchNorm's batch statistics differ. RAM cache (inc2.train.choose_cache,
0.6 x the job's 45G): about 1.35 GB per 1,000 images at 640 with Ultralytics'
50 % margin (docs/CONTINUOUS_LOOP.md 5.6), so base_v2 needs about 15.5 GB at
832 and 23.5 GB at 1024, both under 27 GB.

Cost (estimates, V100, 1 SU per GPU-hour). The measured rates are YOLO11n at
640 px (docs/CONTINUOUS_LOOP.md §5.6): cold 6.0-7.0 ms per image-epoch
(realloop_v1 base 1.974 h / 3 / 3,927 / 100 = 6.0; b0_v1 base 1.768 h / 3 /
3,049 / 100 = 7.0), incremental 7.4 (realloop_v1: 9.05 h over 36 runs), and
scoring 54.5 ms per exam image (b0_v1 finals: 0.4318 h / 3 runs / 9,501
images). Another arm's rate is bracketed:
  high = the n640 rate x the arm's FLOPs ratio (the contract's basis). YOLO11n
         at 640 is not compute-bound on a V100 (the data loader is), so this
         over-states a larger model's time;
  low  = the n640 rate x the arm's pixel ratio (imgsz / 640)^2: a larger model
         is never faster than the loader, whose work grows with the pixels.
FLOPs are GFLOPs of one forward pass at nc 13, from Ultralytics'
torch_utils.get_flops (thop) on yolo11{n,s,m}.yaml, 8.4.22, at 640: n 6.454,
s 21.574, m 68.240 (gflops_at_640). Scoring runs at 640 for every arm, so
the score's high bracket scales by gflops_at_640. Training runs at the arm's
imgsz: gflops is the 640 figure x (imgsz / 640)^2, and the pixel ratio is 1
for the grid (the low bracket is the n640 rate itself) and (imgsz / 640)^2
for a measurement arm. The high bracket of a measurement arm (m832 x17.9,
s1024 x8.6 of n640) over-states it several times: the measured m640 rate is
2.2 x n640's, not 10.6 x. Every figure
these functions return says est. with its basis; a measured rate
(measured_rates) replaces them once an arm has run.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

PROTOCOL = "v3"
PROTOCOL_PACKAGE = "inc2"
SPLITS_VERSION = "v2"

# ------------------------------------------------------------------ recipes
TRAINER = "full"
COMMON = {"trainer": TRAINER, "optimizer": "SGD", "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
          "cos_lr": True, "freeze": None, "lora": None, "batch": 32, "cache": "ram", "workers": 5,
          "close_mosaic": 10, "deterministic": True}
COLD = {"epochs": 100, "lr0": 0.01, "warmup_epochs": 3, "warmup_bias_lr": 0.1}
INCREMENTAL = {
    "r0": {"epochs": 30, "lr0": 0.002, "warmup_epochs": 1, "warmup_bias_lr": 0.002},
    "x1a": {"epochs": 30, "lr0": 0.005, "warmup_epochs": 3, "warmup_bias_lr": 0.005},
    "x1b": {"epochs": 50, "lr0": 0.01, "warmup_epochs": 3, "warmup_bias_lr": 0.01},
}
INCREMENTAL_NAMES = tuple(INCREMENTAL)          # r0, x1a, x1b
COLD_NAME = "cold"
COLD_KINDS = ("base", "union")
INC_KINDS = ("cand", "null")
TRAIN_KINDS = COLD_KINDS + INC_KINDS
# Keys a deviation check skips: the per-run seed and the two operational settings.
UNCHECKED_KEYS = ("seed", "cache", "workers")
RECIPE_KEYS = ("trainer", "epochs", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
               "warmup_epochs", "warmup_bias_lr", "cos_lr", "freeze", "lora", "imgsz", "batch",
               "cache", "workers", "close_mosaic", "deterministic")      # inc.driver RECIPE_KEYS minus seed
# L-6: excluded from stream version 1 by the survival rule applied to pilot_v3's chains.
EXCLUDED_TRAINERS = {"freeze": "L-6: pilot_v3 freeze chain agreed with truth on 3/7 steps (5/7 needed)",
                     "lora": "L-6: pilot_v3 lora chain agreed with truth on 3/7 steps (5/7 needed)"}

# -------------------------------------------------------------------- arms
ARMS = {
    "n640": {"model": "yolo11n.pt", "imgsz": 640, "gflops": 6.454, "gflops_at_640": 6.454},
    "s640": {"model": "yolo11s.pt", "imgsz": 640, "gflops": 21.574, "gflops_at_640": 21.574},
    "m640": {"model": "yolo11m.pt", "imgsz": 640, "gflops": 68.240, "gflops_at_640": 68.240},
    # measurement arms (module docstring): gflops = gflops_at_640 x (imgsz / 640)^2; an arm's "batch"
    # replaces COMMON's in every recipe of the arm (m832: batch 32 would not fit one V100-32GB)
    "m832": {"model": "yolo11m.pt", "imgsz": 832, "gflops": 115.326, "gflops_at_640": 68.240, "batch": 16},
    "s1024": {"model": "yolo11s.pt", "imgsz": 1024, "gflops": 55.229, "gflops_at_640": 21.574},
    # box-quality measurement arms (2026-10-01): larger or newer detectors at 640 px, batch 32 like the grid
    "l640": {"model": "yolo11l.pt", "imgsz": 640, "gflops": 87.325, "gflops_at_640": 87.325},
    "y26m640": {"model": "yolo26m.pt", "imgsz": 640, "gflops": 74.821, "gflops_at_640": 74.821},
    "y26l640": {"model": "yolo26l.pt", "imgsz": 640, "gflops": 93.221, "gflops_at_640": 93.221},
}
ARM_IDS = tuple(ARMS)
GRID_ARMS = ("n640", "s640", "m640")            # L-4's capacity grid: the arms a decision may choose
MEASURE_ARMS = tuple(a for a in ARM_IDS if a not in GRID_ARMS)
DEFAULT_ARM = "n640"
REFERENCE_ARM = "n640"                          # the arm the measured rates are of
FLOPS_BASIS = ("GFLOPs of one forward pass at nc 13, ultralytics.utils.torch_utils.get_flops (thop) on "
               "yolo11{n,s,m}.yaml, Ultralytics 8.4.22; yolo11l.yaml, yolo26m.yaml and yolo26l.yaml, Ultralytics "
               "8.4.37 (yolo11m.yaml gives the same 68.240 there)")
ARM_RECORD_KEYS = ("id", "model", "imgsz", "gflops", "gflops_at_640", "weights_sha256")

# ------------------------------------------------------------------- rates
COLD_MS = (6.0, 7.0)          # per image-epoch, YOLO11n at 640, V100
INC_MS = (7.4, 7.4)
SCORE_MS = 54.5               # per exam image scored, YOLO11n, V100
RATE_BASIS = {
    "cold": "realloop_v1 base 1.974 GPU-h / 3 runs / 3,927 images / 100 epochs = 6.0 ms; b0_v1 base 1.768 GPU-h "
            "/ 3 / 3,049 / 100 = 7.0 ms (YOLO11n 640, V100; docs/CONTINUOUS_LOOP.md 5.6)",
    "incremental": "realloop_v1: 9.05 GPU-h over 36 incremental runs = 7.4 ms per image-epoch (5.6)",
    "score": "b0_v1 finals: 0.4318 GPU-h / 3 runs / 9,501 exam images = 54.5 ms per image",
}
V2_FINAL_EXAM_IMAGES = {"dev": 617, "imageweeds": 3208, "test": 1977}     # splits v1 summary.json (v2 byte copies)
TRUTH_STEP_CAP_GPU_H = 25.0   # L-4: a step with truth above this runs truth every ceil(cost/25)-th step


class RecipeError(ValueError):
    """A recipe, arm or cost request outside Protocol v3."""


# ------------------------------------------------------------------ helpers
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_arm_id(arm):
    if arm not in ARMS:
        raise RecipeError("arm %r is not one of %s (docs/CONTINUOUS_LOOP.md L-4)" % (arm, list(ARM_IDS)))
    return arm


def arm_id(arm):
    """The id of an arm given as an id or as a record."""
    if isinstance(arm, dict):
        return check_arm_id(arm.get("id"))
    return check_arm_id(arm if arm is not None else DEFAULT_ARM)


def arm_from_arch(arch, imgsz):
    """The arm id of a detector name and a training imgsz, as the autopilot's
    L23B argv names an arm (--arch yolo11s --imgsz 640 -> s640). Both must
    be given and must be one row of the table; anything else raises."""
    if arch is None or imgsz is None:
        raise RecipeError("an arm named by --arch needs --imgsz too (and the reverse), got %r, %r" % (arch, imgsz))
    try:
        size = int(imgsz)
    except (TypeError, ValueError):
        raise RecipeError("imgsz %r is not an integer" % (imgsz,))
    name = str(arch)
    name = name[:-3] if name.endswith(".pt") else name
    hits = [a for a, v in ARMS.items() if v["model"][:-3] == name and v["imgsz"] == size]
    if len(hits) != 1:
        raise RecipeError("no Protocol v3 arm is %s at %d px; the arms are %s (docs/CONTINUOUS_LOOP.md L-4)"
                          % (name, size, ", ".join("%s = %s at %d" % (a, v["model"][:-3], v["imgsz"])
                                                   for a, v in ARMS.items())))
    return hits[0]


def _arm_keys(a):
    """The recipe keys an arm sets: its imgsz, and its batch when it has one."""
    return {"imgsz": a["imgsz"], "batch": a.get("batch", COMMON["batch"])}


def cold(arm=DEFAULT_ARM):
    """The cold recipe (base, union) of an arm, in exp.json's form (no seed)."""
    a = ARMS[arm_id(arm)]
    return dict(COMMON, **COLD, **_arm_keys(a))


def incremental(name, arm=DEFAULT_ARM):
    """An incremental recipe (r0, x1a, x1b) of an arm, in exp.json's form."""
    if name not in INCREMENTAL:
        why = EXCLUDED_TRAINERS.get(name)
        raise RecipeError("recipe %r is not in the Protocol v3 table %s%s"
                          % (name, list(INCREMENTAL_NAMES), " (%s)" % why if why else ""))
    a = ARMS[arm_id(arm)]
    return dict(COMMON, **INCREMENTAL[name], **_arm_keys(a))


def table(arm=DEFAULT_ARM):
    """{'cold': ..., 'r0': ..., 'x1a': ..., 'x1b': ...} for an arm."""
    out = {COLD_NAME: cold(arm)}
    out.update((n, incremental(n, arm)) for n in INCREMENTAL_NAMES)
    return out


def _diff(recipe, want):
    out = []
    for k in sorted(set(want) | set(recipe)):
        if k in UNCHECKED_KEYS:
            continue
        if k not in want:
            out.append("%s %r (not a recipe key)" % (k, recipe.get(k)))
            continue
        v, got = want[k], recipe.get(k, "<missing>")
        same = (got is v) if isinstance(v, bool) or isinstance(got, bool) else got == v
        if not same:
            out.append("%s %r (protocol v3 %r)" % (k, got, v))
    return out


def match(kind, recipe, arm=DEFAULT_ARM):
    """The table name recipe equals for a run of this kind ('cold', 'r0',
    'x1a' or 'x1b'), or None."""
    if not isinstance(recipe, dict):
        return None
    if kind in COLD_KINDS:
        return COLD_NAME if not _diff(recipe, cold(arm)) else None
    if kind in INC_KINDS:
        for n in INCREMENTAL_NAMES:
            if not _diff(recipe, incremental(n, arm)):
                return n
    return None


def deviations(kind, recipe, arm=DEFAULT_ARM):
    """How recipe departs from Protocol v3 for a run of kind (base / union:
    the arm's cold recipe; cand / null: one of r0, x1a, x1b at the arm's
    imgsz); [] when it does not. For an incremental run the departures are
    listed against the nearest table recipe (fewest differing keys, then the
    table's order), named in the first entry."""
    if kind not in TRAIN_KINDS:
        raise RecipeError("kind %r trains nothing; only %s have a recipe" % (kind, list(TRAIN_KINDS)))
    if not isinstance(recipe, dict):
        return ["recipe is not an object"]
    extra = []
    t = recipe.get("trainer")
    if t in EXCLUDED_TRAINERS:
        extra.append("trainer %r is out of stream version 1 (%s)" % (t, EXCLUDED_TRAINERS[t]))
    if kind in COLD_KINDS:
        return extra + _diff(recipe, cold(arm))
    best = None
    for n in INCREMENTAL_NAMES:
        d = _diff(recipe, incremental(n, arm))
        if not d:
            return extra
        if best is None or len(d) < len(best[1]):
            best = (n, d)
    return extra + ["nearest protocol v3 recipe %r:" % best[0]] + best[1]


# -------------------------------------------------------------------- arms
def resolve_arm(arm=DEFAULT_ARM, repo=None, require_weights=True, sha256_file=None):
    """The arm's exp.json record: {id, model, imgsz, gflops, gflops_at_640,
    weights_sha256}. The checkpoint is REPO/<model> (the executor resolves a
    relative init against REPO and downloads nothing): its sha256 is pinned
    into the record. A missing checkpoint raises when require_weights (every
    production build), else the record says weights_sha256 None."""
    aid = arm_id(arm)
    a = ARMS[aid]
    rec = {"id": aid, "model": a["model"], "imgsz": a["imgsz"], "gflops": a["gflops"],
           "gflops_at_640": a["gflops_at_640"], "weights_sha256": None}
    if repo is not None:
        p = Path(repo) / a["model"]
        if p.is_file():
            if sha256_file is None:
                from ..inc.common import sha256_file as sha256_file_
                sha256_file = sha256_file_
            rec["weights_sha256"] = sha256_file(p)
        elif require_weights:
            raise RecipeError("arm %s needs its COCO checkpoint at %s (nothing is downloaded: place a complete "
                              "copy there first)" % (aid, p))
    elif require_weights:
        raise RecipeError("arm %s: no REPO to find %s in" % (aid, a["model"]))
    return rec


def check_arm_record(rec):
    """Raise unless rec is an arm record as resolve_arm writes it (the id's
    fixed fields unchanged). Returns the id."""
    if not isinstance(rec, dict):
        raise RecipeError("arm record must be an object, got %r" % (rec,))
    unknown = sorted(set(rec) - set(ARM_RECORD_KEYS))
    if unknown:
        raise RecipeError("arm record has unknown key(s) %s" % unknown)
    aid = arm_id(rec)
    a = ARMS[aid]
    for k in ("model", "imgsz", "gflops", "gflops_at_640"):
        if rec.get(k) != a[k]:
            raise RecipeError("arm record %s says %s %r; the table says %r" % (aid, k, rec.get(k), a[k]))
    sha = rec.get("weights_sha256")
    if sha is not None and (not isinstance(sha, str) or len(sha) != 64):
        raise RecipeError("arm record %s: weights_sha256 %r is not a sha256" % (aid, sha))
    return aid


def stamp(arm_record):
    """The keys every Protocol v3 exp.json carries: merge into the definition
    (inc.driver ignores keys it does not know, and pins the whole definition
    at init)."""
    check_arm_record(arm_record)
    return {"protocol": PROTOCOL, "protocol_package": PROTOCOL_PACKAGE, "splits_version": SPLITS_VERSION,
            "arm": copy.deepcopy(arm_record), "init_weights": arm_record["model"]}


# -------------------------------------------------------------------- cost
def flops_factor(arm, at_640=False):
    """The arm's FLOPs over n640's (at its imgsz, or at 640 for scoring)."""
    a, ref = ARMS[arm_id(arm)], ARMS[REFERENCE_ARM]
    key = "gflops_at_640" if at_640 else "gflops"
    return a[key] / ref[key]


def pixel_factor(arm):
    return (ARMS[arm_id(arm)]["imgsz"] / float(ARMS[REFERENCE_ARM]["imgsz"])) ** 2


def rates(arm=DEFAULT_ARM):
    """{'cold', 'incremental', 'score'}: (low, high) ms per image-epoch (per
    image scored for 'score'), est., with the basis (module docstring)."""
    aid = arm_id(arm)
    ff, pf, sf = flops_factor(aid), pixel_factor(aid), flops_factor(aid, at_640=True)

    def bracket(lo_hi):
        lo, hi = lo_hi
        return (round(lo * min(pf, ff), 4), round(hi * max(ff, pf), 4))

    return {"cold": bracket(COLD_MS), "incremental": bracket(INC_MS),
            "score": (round(SCORE_MS, 4), round(SCORE_MS * max(1.0, sf), 4)),
            "estimate": True, "arm": aid, "flops_factor": round(ff, 4), "pixel_factor": round(pf, 4),
            "score_flops_factor": round(sf, 4),
            "basis": dict(RATE_BASIS, flops=FLOPS_BASIS,
                          bracket="low = n640 rate x pixel ratio (the loader bounds a larger model from below); "
                                  "high = n640 rate x FLOPs ratio (contract L-4 basis; YOLO11n is loader-bound "
                                  "on a V100, so this over-states larger models)")}


def _hours(ms, image_epochs):
    return ms * image_epochs / 3.6e6


def run_hours(kind, n_images, epochs, arm=DEFAULT_ARM, rate=None):
    """(low, high) GPU-h of one training run, est."""
    r = rate or rates(arm)
    key = "cold" if kind in COLD_KINDS else "incremental"
    lo, hi = r[key]
    return (_hours(lo, n_images * epochs), _hours(hi, n_images * epochs))


def score_hours(n_images, arm=DEFAULT_ARM, rate=None):
    r = rate or rates(arm)
    lo, hi = r["score"]
    return (_hours(lo, n_images), _hours(hi, n_images))


def baseline_cost(n_images, seeds, final_exams, arm=DEFAULT_ARM, exam_images=None, walltime_h=8.0):
    """Estimated GPU-h of a baseline experiment: one cold run per seed (with
    its dev score) and one final run per seed on final_exams. Also the
    projected longest run against the cold walltime (inc.driver
    COLD_TIME_LIMIT, 8 h) at 0.8 of it (D26's line)."""
    exam_images = dict(V2_FINAL_EXAM_IMAGES, **(exam_images or {}))
    rate = rates(arm)
    rec = cold(arm)
    n_seeds = len(list(seeds))
    tr = run_hours("base", n_images, rec["epochs"], rate=rate)
    dev = score_hours(exam_images["dev"], rate=rate)
    per_run = (tr[0] + dev[0], tr[1] + dev[1])
    fin_images = sum(exam_images[e] for e in final_exams)
    fin = score_hours(fin_images, rate=rate)
    total = (n_seeds * (per_run[0] + fin[0]), n_seeds * (per_run[1] + fin[1]))
    return {"estimate": True, "arm": arm_id(arm), "n_images": int(n_images), "seeds": n_seeds,
            "epochs": rec["epochs"], "imgsz": rec["imgsz"],
            "per_run_gpu_h": [round(per_run[0], 3), round(per_run[1], 3)],
            "final_run_gpu_h": [round(fin[0], 3), round(fin[1], 3)],
            "total_gpu_h": [round(total[0], 2), round(total[1], 2)],
            "walltime": {"limit_h": walltime_h, "d26_line_h": round(0.8 * walltime_h, 2),
                         "longest_run_h": [round(per_run[0], 2), round(per_run[1], 2)],
                         "over_d26_line": [per_run[0] >= 0.8 * walltime_h, per_run[1] >= 0.8 * walltime_h]},
            "rates_ms": {k: list(rate[k]) for k in ("cold", "incremental", "score")},
            "flops_factor": rate["flops_factor"], "pixel_factor": rate["pixel_factor"],
            "basis": rate["basis"],
            "note": "est.: rates measured for YOLO11n at 640 on V100, bracketed for this arm (low: pixel ratio, "
                    "high: FLOPs ratio); a measured rate of this arm replaces them"}


def step_cost(n_pool, m, arm=DEFAULT_ARM, recipe="r0", seeds=3, truth=True, cold_ms=None, inc_ms=None):
    """GPU-h of one chain step (seeds cand on N + M and seeds null on N at the
    recipe's epochs) plus, with truth, seeds cold union runs on N + M. With
    cold_ms / inc_ms (measured, ms per image-epoch) the figure is (x, x);
    otherwise the est. bracket of rates(arm)."""
    rate = rates(arm)
    ep_inc = incremental(recipe, arm)["epochs"]
    ep_cold = cold(arm)["epochs"]
    ci = (cold_ms, cold_ms) if cold_ms is not None else rate["cold"]
    ii = (inc_ms, inc_ms) if inc_ms is not None else rate["incremental"]
    out = []
    for j in (0, 1):
        chain = _hours(ii[j], seeds * ((n_pool + m) + n_pool) * ep_inc)
        tr = _hours(ci[j], seeds * (n_pool + m) * ep_cold) if truth else 0.0
        out.append(chain + tr)
    return {"gpu_h": [round(out[0], 3), round(out[1], 3)], "estimate": cold_ms is None or inc_ms is None,
            "n_pool": int(n_pool), "m": int(m), "recipe": recipe, "truth": bool(truth), "seeds": seeds}


def truth_every(step_gpu_h, cap=TRUTH_STEP_CAP_GPU_H):
    """L-4: 1 when a step with its truth arm costs <= cap GPU-h, else
    ceil(cost / cap)."""
    if not _is_num(step_gpu_h) or step_gpu_h < 0:
        raise RecipeError("step cost %r is not a non-negative number" % (step_gpu_h,))
    return 1 if step_gpu_h <= cap else int(math.ceil(step_gpu_h / cap))


def measured_rates(run_records):
    """Measured ms per image-epoch from finished run.json records of one kind
    ({'train_seconds', 'n_train_images', 'spec': {'recipe': {'epochs'}}}):
    (median, n) or (None, 0)."""
    vals = []
    for rj in run_records:
        try:
            secs = float(rj["train_seconds"])
            n = int(rj["n_train_images"])
            ep = int(rj["spec"]["recipe"]["epochs"])
        except (KeyError, TypeError, ValueError):
            continue
        if secs > 0 and n > 0 and ep > 0:
            vals.append(1000.0 * secs / (n * ep))
    if not vals:
        return None, 0
    vals.sort()
    k = len(vals)
    med = vals[k // 2] if k % 2 else 0.5 * (vals[k // 2 - 1] + vals[k // 2])
    return med, k


# ------------------------------------------------------------ Stage B rule
P_RECIPE_FLAG = 0.25          # gate.GateConfig().p_recipe_flag
STAGE_B_RULE = ("per step, delta_min = max(0, inc - mean(null) - 2 sd(null)) (the data effect a helpful increment "
                "needs to clear the regression guard); the chosen chain has 1. the smallest median delta_min, then "
                "2. fewer recipe flags (P_recipe <= 0.25), then 3. the cheaper recipe (fewer epochs), and 4. a tie "
                "goes to r0 (docs/CONTINUOUS_LOOP.md 5.1, Stage B)")


def delta_min(inc, null_values, sd_mult=2.0):
    """max(0, inc - mean(null) - sd_mult x sd(null)), sample sd (ddof 1)."""
    vals = [float(v) for v in null_values]
    if not vals or not all(math.isfinite(v) for v in vals) or not _is_num(inc):
        raise RecipeError("delta_min needs a finite incumbent and null values, got %r, %r" % (inc, null_values))
    m = math.fsum(vals) / len(vals)
    sd = math.sqrt(math.fsum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) >= 2 else 0.0
    return max(0.0, float(inc) - m - sd_mult * sd)


def _median(xs):
    xs = sorted(xs)
    k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])


def stage_b_choice(gate_entries, chains, arm=DEFAULT_ARM, p_recipe_flag=P_RECIPE_FLAG):
    """Segment 1's chain by the pre-registered rule (STAGE_B_RULE), from the
    pinned driver's ledger gate entries of those chains. chains maps each
    chain name to its table recipe name (e.g. {"r0": "r0", "x1a": "x1a"}).
    Returns {"chosen", "table", "rule"}; test is never an input."""
    table = {}
    for c, rname in chains.items():
        steps = [e for e in gate_entries if e.get("type") == "gate" and e.get("chain") == c]
        if not steps:
            raise RecipeError("chain %s has no decided step" % c)
        dm = [delta_min(e["decision"]["inc"], e["decision"]["null_values"]) for e in steps]
        flags = sum(1 for e in steps if e["decision"]["p_recipe"] <= p_recipe_flag + 1e-12)
        table[c] = {"recipe": rname, "steps": len(steps), "delta_min": dm, "median_delta_min": _median(dm),
                    "recipe_flags": flags, "epochs": incremental(rname, arm)["epochs"]}
    order = sorted(table, key=lambda c: (table[c]["median_delta_min"], table[c]["recipe_flags"],
                                         table[c]["epochs"], 0 if table[c]["recipe"] == "r0" else 1, c))
    return {"chosen": order[0], "order": order, "table": table, "rule": STAGE_B_RULE}
