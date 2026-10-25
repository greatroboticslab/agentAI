"""V3F D6: RF-DETR-L 12-class on base A under the protocol (docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered
as one family)", V3F.5 D6; decisions 12 and 27).

    sbatch run_inc2_detr.sh inc2.v3fam_detr smoke            (L23K, TRAIN lane: staging, one epoch, the checks)
    sbatch run_inc2_detr.sh inc2.v3fam_detr train --seed K   (L23K, three seed jobs, K = 0, 1, 2)
    python -m weed_optimizer_framework.tools.inc2.v3fam_detr score          (= inc2.v3fam score --arm D6, L23J)
    python -m weed_optimizer_framework.tools.inc2.v3fam_detr rehearse-exif --out DIR   (read-only, a person's)
    python -m weed_optimizer_framework.tools.inc2.v3fam_detr sources        (what D6's reference read scores)
    python -m weed_optimizer_framework.tools.inc2.v3fam_detr check-env      (D6's environment: the import check)
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

Data (written once, v3fam/v1/d6/stage/ with stage.json). train/ holds all of
base_v2 (LOCK v2, 6,811 rows): images by symlink named <key><ext>, labels as
copies of their verified bytes (13 classes in INC id order, OtherPlant 12);
valid/ holds the round(0.02 x 6,811) = 136 rows with the smallest
stable_int("inc2/v3fam/d6/valid/<key>"), which are also in train/ (the
training data is exactly base_v2; their per-epoch numbers are never used);
data.yaml names the 13 INC classes. No test/ folder: a directory named test
anywhere in the tree refuses, and so does any image of dev, test or
ImageWeeds (the exams D6 is scored or read on: by key, image sha256 or
resolved path) and any staged file resolving into a materialised exam
(INC_DIR/exams), the probe's root or the zoo's root (Revision 29: ood22's
and ood23's images and the probe's are base_v2's own rows, which train/
holds by V3F.5 D6; how many base_v2 rows they share is recorded). D0-S's
confirmed copies leave train/ (V3F.3.7 item 4): staging requires D0-S's
complete record and drops the base_v2 rows on its removal list (the family
data summary's d6_staging_removals, when it exists, must agree), recording
them; valid/ is drawn from what is left. EXIF: rfdetr 1.6.5.post0 reads YOLO folders through supervision and
OpenCV; the read-only rehearsal (V3F.8 item 3, rehearse-exif) decides
EXIF_STAGING: "symlink" (OpenCV yields the EXIF-transposed frame the labels
are in; checked again on the staged tagged images, or the staging refuses)
or "transpose" (the images tagged 3, 6 or 8 are written once, transposed, as
JPEG quality 100 with 4:4:4 sampling, the rest symlinked). Until the
rehearsal sets it, staging refuses.

Recipe (fixed): RFDETRLarge from REPO/rf-detr-large-2026.pth (135,954,129
bytes; rfdetr's MD5 checked, sha256 recorded), resolution 704, 50 epochs,
batch_size 2 x grad_accum_steps 8, lr 1e-4, lr_encoder 1.5e-4, weight_decay
1e-4, use_ema (ema_decay 0.993, ema_tau 100), early_stopping false, run_test
false, checkpoint_interval 1000 (> 50 epochs: the archive ModelCheckpoint
never fires), eval_interval 1 on valid/, num_workers 4, tensorboard false,
seed through rfdetr's TrainConfig.seed; every other field at its default
(recorded). The final weights are the EMA weights after the last epoch (the
EMA callback's state after fit, which rfdetr also leaves in the module),
exported once as v3fam/v1/d6/s<k>/final_ema.pth in rfdetr's stripped
{model, args, epoch} format and reloaded to check they equal the in-process
weights exactly. No checkpoint_best_* file is ever loaded. Each attempt
trains into a fresh directory (s<k>/attempt_<n>/); after the export, and on
any failure, the attempt removes the files it wrote there itself (last.ckpt,
checkpoint_best_{regular,ema,total}.pth) and refuses to remove anything else.

The smoke, each seed's training and the scoring run inside
v3fam.e_trees_unchanged (V3F.3.7: E3's tree and capacity/e{1,2,3}_* hashed
before and after; the job fails if either changed).

Smoke (first): one epoch on the full staging (v3fam/v1/d6/smoke/), the same
checks, no archive checkpoint and no test folder, the export equal to the
in-process EMA weights and its predictions on 20 training images equal to
the in-process model's; it records the epoch time. The seed jobs run only if
epoch time x 50 + 0.5 h <= 0.8 x 36 h = 28.8 h; otherwise smoke.json records
"not run (walltime)" and train refuses. The epoch time is the epoch's own
(deviation 5, 2026-10-24: EPOCH_TIME_DEF, from the first training epoch's
start to the end of fit, timed by a callback that keeps no state); the
one-time work before it (the dataset construction, the sanity validation) is
recorded apart as setup_seconds. The smoke of 2026-10-21 took the whole fit
(2,405.6 s, about 905 s of it before the epoch) and is moved aside by the
next smoke's own job (inc2.v3fam_supersede, group d6).

Scoring (L23J, --arm D6): each seed's final EMA weights predict every exam
image (EXIF-transposed, at 704 px, threshold 0.001, the top 300 queries x
classes), the predictions convert to rows in original pixels with the class
index mapped to the INC id by name (rfdetr's background index 13 dropped and
counted; any other index refused), and the rows replace the predictions of
b_v2_m640 seed 0 (the scaffold) in the family inject_rows pass (the
injection reproduction of V3F.3.2 item 4 and the GT round trip guard it).
Dev decides (P5, Tier C); ImageWeeds (in this process) and ooddev_v1 (in the
zoo's root, a subprocess) are reported. D6 has one reference read
(read_sources).

The RF-DETR calls are behind a backend (RfdetrBackend; tests pass a fake):
train, save, load_state, states_equal, load_model, predict, read_yolo.

Environment (deviation 2, 2026-10-12): rfdetr 1.6.5.post0 needs supervision's DetectionDataset and
Detections.metadata, which the bench env's supervision 0.6.0 lacks (groundingdino-py pins it there). D6's
processes run in a virtual environment over the bench env that adds only the pins of v3fam_detr_env.txt
(supervision 0.27.0.post2, defusedxml 0.7.1); run_inc2_detr.sh, and run_inc2_build.sh for D6's score and
score-test, put it first on PATH, and `check-env` (env_problems) refuses any other environment. rfdetr has no
__version__: its version is read from its installed metadata.

A refusal raises inc2.v3fam.V3famRefused; the CLI prints "[inc2.v3fam_detr]
ERROR: ..." and exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import common as C2
from . import recipes as RC
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F
from . import v3fam_systems as SYS

# ------------------------------------------------------------------ constants (pre-registered, V3F.5 D6)
ARM = "D6"
DIR = "D6"
EPOCHS = 50
SMOKE_EPOCHS = 1
RESOLUTION = 704
RECIPE = {"batch_size": 2, "grad_accum_steps": 8, "lr": 1e-4, "lr_encoder": 1.5e-4, "weight_decay": 1e-4,
          "use_ema": True, "ema_decay": 0.993, "ema_tau": 100, "early_stopping": False, "run_test": False,
          "checkpoint_interval": 1000, "eval_interval": 1, "num_workers": 4, "tensorboard": False}
PRETRAIN = "rf-detr-large-2026.pth"          # in REPO
PRETRAIN_BYTES = 135954129
PRETRAIN_MD5 = "5cb72153541cbcb9aa6efa26222acc75"   # rfdetr 1.6.5.post0's registry entry for this file
RFDETR_VERSION = "1.6.5.post0"
ENV_FILE = Path(__file__).resolve().with_name("v3fam_detr_env.txt")   # D6's environment pins (deviation 2)
ENV_IMPORTS = ("torch", "rfdetr", "supervision", "pytorch_lightning")
VALID_SHARE = 0.02
VALID_SEED_TEXT = "inc2/v3fam/d6/valid/%s"
NUM_CLASSES = C.NC                           # 13: the twelve species and OtherPlant
CLASS_NAMES = tuple(C.CLASS_NAMES)
BACKGROUND_ID = NUM_CLASSES                  # rfdetr's num_classes + 1 logits: index 13 is no-object
THRESHOLD = 0.001
NUM_SELECT = 300
PRED_BATCH = 8
SMOKE_PRED_IMAGES = 20
PRED_XYXY_TOL = 1e-2                         # px: the export's predictions against the in-process model's
PRED_CONF_TOL = 1e-4
WALLTIME_H = 36.0
WALLTIME_SHARE = 0.8                         # the seeds run only if the projection is <= 28.8 h
OVERHEAD_H = 0.5
NOT_RUN_WALLTIME = "not run (walltime)"
# The read-only rehearsal's outcome (V3F.8 item 3): "symlink" or "transpose"; staging refuses while it is None.
# Recorded 2026-10-12 (deviation 2; docs/CONTINUOUS_LOOP.md, "Deviation (2026-10-12)"): rfdetr's reader in D6's
# environment (supervision 0.27.0.post2) yields exif_transpose's pixels for base_v2's tagged images (MAD 0) but
# scales the label boxes by the stored, unrotated width and height, so every image tagged 6 or 8 failed the check
# (boxes about 1,000 px off): the tagged images are staged transposed.
EXIF_STAGING = "transpose"
EXIF_MODES = ("symlink", "transpose")
ROTATED_TAGS = (3, 6, 8)
TRANSPOSED_JPEG = {"format": "JPEG", "quality": 100, "subsampling": 0}
EXIF_MAD_TOL = F.EXIF_MAD_TOL                # grey levels
EXIF_BOX_TOL = 1.0                           # px: the reader's label boxes against the transposed frame's
HOLDOUT_EXAMS = ("dev", "test", "imageweeds")     # the exams D6 is scored or read on (Revision 29)
SHARED_EXAMS = ("ood22", "ood23")             # their images are base_v2's tsw22/tsw23 rows: staged, counted
OWN_FILES = ("last.ckpt", "checkpoint_best_regular.pth", "checkpoint_best_ema.pth", "checkpoint_best_total.pth")
ARCHIVE_RE = re.compile(r"\Acheckpoint_\d+(\.\w+)?\Z")      # the interval archive (checkpoint_{epoch})
FINAL = "final_ema.pth"
SPLITS = ("train", "valid")
FMT_STAGE = "inc2-v3fam-d6-stage/1"
FMT_SMOKE = "inc2-v3fam-d6-smoke/1"
FMT_TRAIN = "inc2-v3fam-d6-train/1"
FMT_REHEARSAL = "inc2-v3fam-d6-exif-rehearsal/1"
REPORTED_EXAMS = ("imageweeds", SYS.OODDEV)


def log(msg):
    print("[inc2.v3fam_detr] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths
def d6_dir():
    return F.root() / "d6"


def stage_dir():
    return d6_dir() / "stage"


def stage_record_path():
    return d6_dir() / "stage.json"


def smoke_dir():
    return d6_dir() / "smoke"


def seed_dir(seed):
    return d6_dir() / ("s%d" % int(seed))


def final_path(seed):
    return seed_dir(seed) / FINAL


def pretrain_path():
    return C.REPO / PRETRAIN


def _utc():
    return TS._utc()


def _read_json(path):
    return TS._read_json(path)


def _sha(path):
    return TS._sha(path)


def walltime_cap_h():
    return WALLTIME_SHARE * WALLTIME_H


def projection_h(epoch_seconds, epochs=EPOCHS):
    """The projected training time of one seed job: epoch time x 50 + 0.5 h."""
    return float(epoch_seconds) * epochs / 3600.0 + OVERHEAD_H


def valid_count(n):
    """round(0.02 x n), half up, at least one image (136 for base_v2's 6,811)."""
    return max(1, int(RC.round_half_up(VALID_SHARE * n)))


def valid_keys(rows):
    """The valid/ sample: the round(0.02 x n) keys with the smallest stable_int('inc2/v3fam/d6/valid/<key>')."""
    ranked = sorted(rows, key=lambda r: (C.stable_int(VALID_SEED_TEXT % r["key"]), r["key"]))
    return sorted(r["key"] for r in ranked[:valid_count(len(rows))])


def class_map(names):
    """The INC id of each class index of a model or a data.yaml (by name); refuses anything but the 13 INC names
    in INC id order (the map is the identity, and this checks it)."""
    names = [str(n) for n in names]
    if names != list(CLASS_NAMES):
        raise F.V3famRefused("class names %s are not the INC classes %s in INC id order" % (names, list(CLASS_NAMES)))
    import numpy as np
    return np.asarray([CLASS_NAMES.index(n) for n in names], dtype=np.int64)


# ------------------------------------------------------------------ the holdout index (never staged)
def foreign_roots():
    """The directories no staged file may resolve into: every materialised exam (INC_DIR/exams), the probe's root
    and the zoo's root (resolved)."""
    return sorted({os.path.realpath(str(p)) for p in (C.INC_DIR / "exams", F.root_of("probe"),
                                                       F.root_of("ooddev_v1"))})


def _under(path, roots):
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def holdout_index():
    """{keys, sha256s, paths (resolved), roots, counts} of every image of dev, test and ImageWeeds (LOCK v1's
    manifests; the exams D6 is scored or read on) and the directories no staged file may resolve into
    (foreign_roots). ood22's and ood23's images and the probe's are base_v2's own rows (Revision 29), which train/
    holds: not in the index; shared_with_base counts them for the record. Only the key, image and sha256 fields are
    read; no label is opened."""
    keys, shas, paths, n = set(), set(), set(), {}
    for exam in HOLDOUT_EXAMS:
        p = C.manifest_path(exam)
        if not p.is_file():
            raise F.V3famRefused("the holdout manifest of %s is missing (%s): staging cannot be checked" % (exam, p))
        rows = C.read_manifest(p)
        n[exam] = len(rows)
        for r in rows:
            keys.add(str(r["key"]))
            if r.get("sha256"):
                shas.add(str(r["sha256"]))
            paths.add(os.path.realpath(str(r["image"])))
    return {"keys": keys, "sha256": shas, "paths": paths, "roots": foreign_roots(), "counts": n}


def shared_with_base(rows):
    """{exam: base rows sharing a key, image sha256 or resolved path with it} for ood22, ood23 (LOCK v1) and the
    probe (its root's manifest, when it exists): base_v2's own rows, staged (Revision 29), recorded."""
    out = {}
    mans = [(e, C.manifest_path(e)) for e in SHARED_EXAMS]
    mans.append(("probe", F.root_of("probe") / "splits" / "v1" / "probe.jsonl"))
    for exam, p in mans:
        if not p.is_file():
            out[exam] = None
            continue
        ks, ss, ps = set(), set(), set()
        for r in C.read_manifest(p):
            ks.add(str(r["key"]))
            if r.get("sha256"):
                ss.add(str(r["sha256"]))
            ps.add(os.path.realpath(str(r["image"])))
        out[exam] = sum(1 for r in rows if str(r["key"]) in ks or str(r.get("sha256")) in ss
                        or os.path.realpath(str(r["image"])) in ps)
    return out


def holdout_hits(rows, hold):
    """Rows that are holdout images (by key, image sha256 or resolved path) or whose image resolves into a foreign
    root."""
    out = []
    for r in rows:
        rp = os.path.realpath(str(r["image"]))
        if str(r["key"]) in hold["keys"] or str(r.get("sha256")) in hold["sha256"] or rp in hold["paths"] \
                or _under(rp, hold["roots"]):
            out.append(r["key"])
    return out


def check_tree(root_, hold):
    """Refuse a staging or output tree holding a test folder, or a file or link resolving to a holdout image (by
    its resolved path, or a staged name whose stem is a holdout key) or into a foreign root. Returns the number of
    entries seen."""
    root_ = Path(root_)
    seen = 0
    for dp, dns, fns in os.walk(str(root_)):
        for dn in dns:
            if dn.lower() == "test":
                raise F.V3famRefused("%s holds a test/ folder (%s): D6 stages no test split"
                                     % (root_, os.path.join(dp, dn)))
        for fn in fns:
            seen += 1
            p = os.path.join(dp, fn)
            stem = os.path.splitext(fn)[0]
            rp = os.path.realpath(p)
            if stem in hold["keys"] or rp in hold["paths"] or _under(rp, hold["roots"]):
                raise F.V3famRefused("%s is a holdout image or resolves into an exam's root (%s): never staged"
                                     % (p, rp))
    return seen


def d0s_staging_removals(base_keys):
    """(the base_v2 keys D0-S's confirmed copies remove from train/, the record) (V3F.3.7 item 4): D0-S's complete
    record is required (inc2.v3fam_sweep.removal_keys); the removal list's base_v2 rows. When the family data
    summary exists its d6_staging_removals must be the same list."""
    from . import v3fam_data as VD
    from . import v3fam_sweep as SW
    try:
        keys, sha = SW.removal_keys()
    except SW.SweepRefused as e:
        raise F.V3famRefused("D6 stages base_v2 minus D0-S's confirmed copies: %s" % e)
    drop = sorted(set(keys) & set(base_keys))
    summ = VD.read_summary()
    if summ is not None:
        want = sorted((summ.get("d0s") or {}).get("d6_staging_removals") or [])
        if want != drop:
            raise F.V3famRefused("the family data summary's d6_staging_removals (%d) differ from D0-S's removals "
                                 "among base_v2 (%d)" % (len(want), len(drop)))
    rec = SW.read_record() or {}
    return drop, {"record_sha256": _sha(SW.sweep_dir() / "record.json"), "removal_sha256": sha,
                  "removal_rows": len(keys), "dropped": drop, "n_dropped": len(drop),
                  "dropped_sha256": C.sha256_text("\n".join(drop)), "h": len(rec.get("H") or [])}


# ------------------------------------------------------------------ the environment (deviation 2, 2026-10-12)
def env_pins(path=None):
    """{distribution: version} of v3fam_detr_env.txt's requirement lines (name==version --hash=...)."""
    out = {}
    for ln in Path(path or ENV_FILE).read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)(\s+--hash=sha256:[0-9a-f]{64})+\Z", ln)
        if not m:
            raise F.V3famRefused("%s: %r is not 'name==version --hash=sha256:...'" % (path or ENV_FILE, ln))
        out[m.group(1).lower()] = m.group(2)
    if "supervision" not in out:
        raise F.V3famRefused("%s pins no supervision" % (path or ENV_FILE))
    return out


def env_problems(version=None, importer=None, prefix=None, pins=None):
    """Why this interpreter is not D6's environment ([] when it is): rfdetr at RFDETR_VERSION and every pin of
    v3fam_detr_env.txt installed as pinned, supervision imported from this environment's own site-packages (the
    virtual environment over the bench env, not the bench env's 0.6.0) with DetectionDataset.from_yolo, and torch,
    rfdetr, supervision and pytorch_lightning importable. version, importer and prefix are
    importlib.metadata.version, importlib.import_module and sys.prefix (tests pass others)."""
    import importlib
    import importlib.metadata as md
    version = version or md.version
    importer = importer or importlib.import_module
    prefix = os.path.realpath(prefix or sys.prefix)
    out = []
    for name, want in sorted(dict(pins if pins is not None else env_pins(), rfdetr=RFDETR_VERSION).items()):
        try:
            got = version(name)
        except Exception:                    # noqa: BLE001 - not installed (PackageNotFoundError) is a problem
            got = None
        if got != want:
            out.append("%s is %s, not %s" % (name, got or "not installed", want))
    for mod in ENV_IMPORTS:
        try:
            m = importer(mod)
        except Exception as e:               # noqa: BLE001 - any import failure is reported
            out.append("%s does not import (%s: %s)" % (mod, type(e).__name__, str(e)[:200]))
            continue
        if mod == "supervision":
            f = os.path.realpath(str(getattr(m, "__file__", "") or ""))
            if not f.startswith(prefix + os.sep):
                out.append("supervision is imported from %s, outside this environment (%s)" % (f, prefix))
            if not callable(getattr(getattr(m, "DetectionDataset", None), "from_yolo", None)):
                out.append("supervision has no DetectionDataset.from_yolo")
    return out


def check_env():
    """`check-env`: the job scripts' import check and the login node's (deviation 2); refuses unless env_problems
    is empty. Returns the versions it saw."""
    bad = env_problems()
    if bad:
        raise F.V3famRefused("not D6's environment (%s; %s says how it is made): %s"
                             % (sys.prefix, ENV_FILE.name, "; ".join(bad)))
    vers = RfdetrBackend().versions()
    log("D6's environment: %s" % json.dumps(vers, sort_keys=True))
    return vers


# ------------------------------------------------------------------ the epoch clock (deviation 5)
EPOCH_TIME_DEF = ("from the first training epoch's start (pytorch_lightning on_train_epoch_start, after the dataset "
                  "construction and the sanity validation) to the end of fit: the epoch's training, its validation, "
                  "its checkpoints and everything after it until fit returns")


def _epoch_clock():
    """A pytorch_lightning Callback that records the wall time each training epoch starts (never during the sanity
    validation), and nothing else: no state, no effect on the run. The smoke's epoch time is the span from the
    first epoch's start to the end of fit (EPOCH_TIME_DEF): V3F.5 D6's rule (epoch time x 50 + 0.5 h <= 28.8 h)
    multiplies the epoch's own time, and the one-time work before it (rfdetr builds its training dataset by decoding
    every image once, about 870 s on 2026-10-21, then the sanity validation) happens once per job, inside the
    0.5 h. On 2026-10-21 the whole one-epoch fit (2,405.6 s) was taken as the epoch time (deviation 5)."""
    import pytorch_lightning as pl

    class EpochClock(pl.Callback):
        def __init__(self):
            super().__init__()
            self.starts = []

        def on_train_epoch_start(self, trainer, pl_module):
            if not getattr(trainer, "sanity_checking", False):
                self.starts.append(time.time())
    return EpochClock()


def _clock_record(starts, t0, t1):
    """{epochs_started, setup_seconds, epoch_seconds, definition}: setup is fit's start to the first epoch's start,
    epoch_seconds the first epoch's start to the end of fit (None when no epoch started)."""
    starts = [float(x) for x in starts or []]
    first = starts[0] if starts else None
    return {"epochs_started": len(starts), "definition": EPOCH_TIME_DEF,
            "setup_seconds": round(first - t0, 3) if first is not None else None,
            "epoch_seconds": round(t1 - first, 3) if first is not None else None,
            "epoch_start_offsets_s": [round(x - t0, 3) for x in starts[:60]]}


# ------------------------------------------------------------------ the backend (rfdetr 1.6.5.post0)
class RfdetrBackend(object):
    """The RF-DETR calls, imported only here (rfdetr 1.6.5.post0, supervision, torch: D6's environment)."""
    name = "rfdetr"
    epoch_clock = True                       # train reports the epoch clock (_epoch_clock); the smoke requires it

    def versions(self):
        import importlib.metadata as md
        import pytorch_lightning
        import supervision
        import torch
        return {"rfdetr": md.version("rfdetr"), "supervision": supervision.__version__, "torch": torch.__version__,
                "pytorch_lightning": pytorch_lightning.__version__, "defusedxml": md.version("defusedxml"),
                "env": sys.prefix}

    def train(self, stage, out, seed, epochs, pretrain):
        """Train RFDETRLarge on the staging with RECIPE; returns {ema_state, inprocess_state (cpu tensors), args,
        epoch, max_epochs, interrupted, callbacks, seconds, model, class_names, seed_mechanism}."""
        import importlib.metadata as md
        import rfdetr.training as RT
        from rfdetr import RFDETRLarge
        have = md.version("rfdetr")          # the package has no __version__ (deviation 2)
        if have != RFDETR_VERSION:
            raise F.V3famRefused("rfdetr %s: D6 is pre-registered on %s" % (have, RFDETR_VERSION))
        captured = {}
        orig = RT.build_trainer
        clock = _epoch_clock()

        def build(*a, **k):
            tr = orig(*a, **k)
            captured["trainer"], captured["train_config"] = tr, a[0]
            tr.callbacks.insert(0, clock)    # times only: no state, no effect on the run (deviation 5)
            return tr
        RT.build_trainer = build
        try:
            model = RFDETRLarge(pretrain_weights=str(pretrain))
            if getattr(model, "is_deprecated", False):
                raise F.V3famRefused("RFDETRLarge fell back to its deprecated configuration on %s" % pretrain)
            if int(model.model_config.resolution) != RESOLUTION:
                raise F.V3famRefused("RFDETRLarge's resolution is %s, not %d" % (model.model_config.resolution,
                                                                                RESOLUTION))
            t0 = time.time()
            model.train(dataset_dir=str(stage), output_dir=str(out), epochs=int(epochs), seed=int(seed),
                        **RECIPE)
            t1 = time.time()
            secs = t1 - t0
        finally:
            RT.build_trainer = orig
        tr = captured.get("trainer")
        if tr is None:
            raise F.V3famRefused("rfdetr trained without its trainer factory: port inc2/v3fam_detr.py")
        if clock not in list(tr.callbacks):
            raise F.V3famRefused("the epoch clock is not among the trainer's callbacks: port inc2/v3fam_detr.py")
        emas = [c for c in tr.callbacks if callable(getattr(c, "get_ema_model_state_dict", None))]
        best = [c for c in tr.callbacks if type(c).__name__ == "BestModelCallback"]
        arch = [c for c in tr.callbacks if type(c).__name__ == "ModelCheckpoint"
                and getattr(c, "filename", None) == "checkpoint_{epoch}"]
        if len(emas) != 1 or len(best) != 1 or len(arch) != 1:
            raise F.V3famRefused("the trainer holds %d EMA, %d best-model and %d archive callbacks, not one each"
                                 % (len(emas), len(best), len(arch)))
        ema = emas[0].get_ema_model_state_dict()
        if ema is None:
            raise F.V3famRefused("the EMA callback holds no averaged model after fit")
        live = model.model.model.state_dict()
        tc = captured["train_config"]
        args = tc.model_dump() if hasattr(tc, "model_dump") else dict(vars(tc))
        names = list(model.model.class_names or [])
        args["class_names"] = names
        return {"ema_state": {k: v.detach().cpu().clone() for k, v in ema.items()},
                "inprocess_state": {k: v.detach().cpu().clone() for k, v in live.items()},
                "args": args, "epoch": int(tr.current_epoch), "max_epochs": int(tr.max_epochs),
                "interrupted": bool(getattr(tr, "interrupted", False)),
                "callbacks": {"run_test": bool(getattr(best[0], "_run_test", True)),
                              "archive_every_n_epochs": int(getattr(arch[0], "_every_n_epochs", 0) or 0),
                              "ema": True},
                "seconds": secs, "model": model, "class_names": names,
                "epoch_clock": _clock_record(clock.starts, t0, t1),
                "seed_mechanism": "rfdetr TrainConfig.seed: seed_everything(seed + global_rank, workers=True) at "
                                  "fit start"}

    def save(self, payload, path):
        import torch
        torch.save(payload, str(path))

    def load_state(self, path):
        import torch
        return torch.load(str(path), map_location="cpu", weights_only=False)

    def states_equal(self, a, b):
        import torch
        return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)

    def load_model(self, path):
        from rfdetr import RFDETRLarge
        m = RFDETRLarge(pretrain_weights=str(path), num_classes=NUM_CLASSES)
        if getattr(m, "is_deprecated", False):
            raise F.V3famRefused("RFDETRLarge fell back to its deprecated configuration on %s" % path)
        m.model.model.eval()
        return m

    def model_class_names(self, model):
        return list(model.model.class_names or [])

    def predict(self, model, pils, threshold):
        """[(xyxy float [n, 4] in the image's pixels, confidence [n], class index [n])] per image."""
        import numpy as np
        model.model.model.eval()
        dets = model.predict(list(pils), threshold=threshold)
        if not isinstance(dets, list):
            dets = [dets]
        return [(np.asarray(d.xyxy, dtype=np.float64).reshape(-1, 4), np.asarray(d.confidence, dtype=np.float64),
                 np.asarray(d.class_id)) for d in dets]

    def read_yolo(self, split_dir, data_yaml):
        """What rfdetr's YOLO reader yields: supervision.DetectionDataset.from_yolo (OpenCV) over split_dir's
        images/ and labels/; (image path, RGB array, xyxy, class ids) per image, one image at a time (deviation 2:
        a list of the 606 tagged base_v2 frames at full resolution is about 22 GB; the single-process rehearsal
        that built it was killed on the login node after 28 min at about 12 GB)."""
        import numpy as np
        import supervision as sv
        ds = sv.DetectionDataset.from_yolo(images_directory_path=str(Path(split_dir) / "images"),
                                           annotations_directory_path=str(Path(split_dir) / "labels"),
                                           data_yaml_path=str(data_yaml))
        for i in range(len(ds)):
            path, img, det = ds[i]
            yield (str(path), np.ascontiguousarray(np.asarray(img)[:, :, ::-1]),
                   np.asarray(det.xyxy, dtype=np.float64).reshape(-1, 4), np.asarray(det.class_id))


def default_backend(backend=None):
    return backend if backend is not None else RfdetrBackend()


# ------------------------------------------------------------------ EXIF (the rehearsal and the staging check)
def transposed_rgb(path):
    """The image as PIL's exif_transpose sees it (the labels' frame), RGB uint8."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im0:
        return np.asarray(ImageOps.exif_transpose(im0).convert("RGB"))


def label_boxes_px(label_path, w, h):
    """[(cls, x1, y1, x2, y2)] of a YOLO label in a w x h frame."""
    out = []
    for c, cx, cy, bw, bh in C.read_yolo(label_path):
        out.append((int(c), (cx - bw / 2.0) * w, (cy - bh / 2.0) * h, (cx + bw / 2.0) * w, (cy + bh / 2.0) * h))
    return out


def reader_check(backend, rows, tmp):
    """For each row (tagged images), whether rfdetr's reader (backend.read_yolo on a YOLO folder of these images
    alone, under tmp) yields the EXIF-transposed frame the labels are in: the array within EXIF_MAD_TOL grey levels
    of exif_transpose's, the same shape, and the label boxes within EXIF_BOX_TOL px of the transposed frame's.
    Returns {key: {tag, ok, mad, shape, box_diff}}."""
    import numpy as np
    tmp = Path(tmp)
    d = tmp / "train"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    by_name = {}
    for r in rows:
        ext = os.path.splitext(r["image"])[1].lower() or ".jpg"
        os.symlink(os.path.realpath(r["image"]), str(d / "images" / (r["key"] + ext)))
        (d / "labels" / (r["key"] + ".txt")).write_bytes(_verified_label(r))
        by_name[r["key"]] = r
    yml = write_data_yaml(tmp / "data.yaml", tmp)
    out = {}
    for path, rgb, xyxy, cls in backend.read_yolo(d, yml):
        key = os.path.splitext(os.path.basename(path))[0]
        r = by_name.get(key)
        if r is None:
            continue
        b = transposed_rgb(r["image"])
        rec = {"tag": TS.header(r["image"])[0], "reader_shape": list(rgb.shape[:2]), "transposed_shape":
               list(b.shape[:2])}
        if rgb.shape != b.shape:
            out[key] = dict(rec, ok=False, mad=None, box_diff=None)
            continue
        mad = float(np.abs(rgb.astype(np.int16) - b.astype(np.int16)).mean())
        h, w = b.shape[:2]
        lab = sorted(label_boxes_px(r["label"], w, h))
        got = sorted(zip([int(c) for c in cls], *[list(map(float, xyxy[:, j])) for j in range(4)]))
        if len(lab) != len(got) or [x[0] for x in lab] != [x[0] for x in got]:
            bd = None
        else:
            bd = max([max(abs(a - g) for a, g in zip(x[1:], y[1:])) for x, y in zip(lab, got)] or [0.0])
        out[key] = dict(rec, mad=mad, box_diff=bd, ok=bool(mad <= EXIF_MAD_TOL and bd is not None
                                                           and bd <= EXIF_BOX_TOL))
    missing = sorted(set(by_name) - set(out))
    for k in missing:
        out[k] = {"tag": TS.header(by_name[k]["image"])[0], "ok": False, "why": "the reader did not yield it"}
    return out


def rehearse_exif(out_dir, backend=None, rows=None):
    """The read-only rehearsal (V3F.8 item 3): rfdetr's reader on base_v2's images tagged 3, 6 or 8, and a
    montage of their label boxes on the reader's arrays for the visual check. Writes into out_dir only (which
    must lie outside INC_DIR); never under INC_DIR."""
    out_dir = Path(out_dir).resolve()
    if C.INC_DIR.resolve() == out_dir or C.INC_DIR.resolve() in out_dir.parents:
        raise F.V3famRefused("the rehearsal writes nothing under INC_DIR (%s)" % C.INC_DIR)
    backend = default_backend(backend)
    if rows is None:
        rows = C.read_manifest(C2.v2_manifest_path("base_v2"))
    tagged = [r for r in rows if TS.header(r["image"])[0] in ROTATED_TAGS]
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3fam_d6_rehearsal_") as tmp:
        res = reader_check(backend, tagged, tmp)
    ok = all(v.get("ok") for v in res.values())
    doc = {"format": FMT_REHEARSAL, "tagged": len(tagged), "by_tag": _count_tags(tagged), "all_ok": ok,
           "failed": sorted(k for k, v in res.items() if not v.get("ok")), "per_image": res,
           "suggests": "symlink" if ok else "transpose",
           "note": "the visual half (label boxes on the plants) is a person's; EXIF_STAGING records the outcome",
           "created_utc": _utc()}
    (out_dir / "d6_exif_rehearsal.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    log("rehearsal: %d tagged images, reader frame = labels' frame on %d; suggests %s"
        % (len(tagged), sum(1 for v in res.values() if v.get("ok")), doc["suggests"]))
    return doc


def _count_tags(rows):
    out = {}
    for r in rows:
        t = str(TS.header(r["image"])[0])
        out[t] = out.get(t, 0) + 1
    return out


# ------------------------------------------------------------------ staging (written once)
def _verified_label(r):
    """A row's label bytes, refused unless they hash to its label_sha256 and every class is an INC id."""
    b = Path(r["label"]).read_bytes()
    if hashlib.sha256(b).hexdigest() != r.get("label_sha256"):
        raise F.V3famRefused("the label of %s does not hash to its manifest's label_sha256" % r["key"])
    for ln in b.decode("utf-8").splitlines():
        t = ln.split()
        if not t:
            continue
        c = float(t[0])
        if c != int(c) or not 0 <= int(c) < NUM_CLASSES or len(t) != 5:
            raise F.V3famRefused("the label of %s holds the row %r (classes 0..%d, five fields)"
                                 % (r["key"], ln, NUM_CLASSES - 1))
    return b


def write_data_yaml(path, root_):
    path = Path(path)
    lines = ["path: %s" % Path(root_).resolve(), "train: train/images", "val: valid/images",
             "nc: %d" % NUM_CLASSES, "names:"] + ["  - %s" % n for n in CLASS_NAMES]
    path.write_text("\n".join(lines) + "\n")
    return path


def stage(exif=None, rows=None, backend=None):
    """Write v3fam/v1/d6/stage and stage.json once (module docstring), or verify them when they exist. rows:
    the base rows (tests); production reads LOCK v2's base_v2, checked against the lock."""
    if stage_record_path().exists():
        return verify_stage()
    exif = exif if exif is not None else EXIF_STAGING
    if exif not in EXIF_MODES:
        raise F.V3famRefused("EXIF_STAGING is %r: the read-only rehearsal (V3F.8 item 3, rehearse-exif) decides "
                             "%s and is recorded in inc2/v3fam_detr.py before D6 stages anything"
                             % (exif, " or ".join(EXIF_MODES)))
    if rows is None:
        try:
            msha = C2.verify_manifest_against_lock_v2("base_v2")
        except C2.Inc2Error as e:
            raise F.V3famRefused("LOCK v2's base_v2: %s" % e)
        rows = C.read_manifest(C2.v2_manifest_path("base_v2"))
        source = {"manifest": "splits/v2/base_v2.jsonl", "sha256": msha, "locked": "LOCK v2"}
    else:
        source = {"manifest": "given rows (tests)", "sha256": None, "locked": None}
    rows = sorted(rows, key=lambda r: r["key"])
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise F.V3famRefused("the base holds duplicate keys")
    drop, d0s = d0s_staging_removals(keys)
    if drop:
        rows = [r for r in rows if r["key"] not in set(drop)]
        keys = [r["key"] for r in rows]
        log("D0-S's confirmed copies: %d base_v2 row(s) left out of train/ (V3F.3.7 item 4)" % len(drop))
    hold = holdout_index()
    hits = holdout_hits(rows, hold)
    if hits:
        raise F.V3famRefused("%d base row(s) are holdout images (e.g. %s): never staged" % (len(hits), hits[:3]))
    shared = shared_with_base(rows)
    vk = set(valid_keys(rows))
    tags = {r["key"]: TS.header(r["image"])[0] for r in rows}
    tagged = [r for r in rows if tags[r["key"]] in ROTATED_TAGS]
    final = stage_dir()
    if final.exists() or final.is_symlink():
        raise F.V3famRefused("%s exists without stage.json (a crashed staging): move it aside" % final)
    tmp = final.with_name(".stage.%d.%d.tmp" % (os.getpid(), time.monotonic_ns()))
    transposed = {}
    try:
        for split in SPLITS:
            (tmp / split / "images").mkdir(parents=True)
            (tmp / split / "labels").mkdir(parents=True)
        if exif == "transpose" and tagged:
            from PIL import Image, ImageOps
            (tmp / "transposed").mkdir()
            for r in tagged:
                p = tmp / "transposed" / ("%s.jpg" % r["key"])
                with Image.open(r["image"]) as im0:
                    ImageOps.exif_transpose(im0).convert("RGB").save(str(p), **TRANSPOSED_JPEG)
                transposed[r["key"]] = C.sha256_file(p)
        label_shas = []
        for split in SPLITS:
            for r in rows:
                if split == "valid" and r["key"] not in vk:
                    continue
                if r["key"] in transposed:
                    name = "%s.jpg" % r["key"]
                    target = os.path.relpath(str(tmp / "transposed" / name), str(tmp / split / "images"))
                else:
                    name = r["key"] + (os.path.splitext(r["image"])[1].lower() or ".jpg")
                    target = os.path.realpath(str(r["image"]))
                os.symlink(target, str(tmp / split / "images" / name))
                b = _verified_label(r)
                (tmp / split / "labels" / ("%s.txt" % r["key"])).write_bytes(b)
                if split == "train":
                    label_shas.append(r["label_sha256"])
        write_data_yaml(tmp / "data.yaml", final)
        check_tree(tmp, hold)
        exif_rec = {"mode": exif, "tagged": len(tagged), "by_tag": _count_tags(tagged),
                    "transposed": transposed, "transposed_jpeg": dict(TRANSPOSED_JPEG)}
        if exif == "symlink" and tagged:
            # the staged tagged images through rfdetr's reader: the labels' frame, or the staging refuses
            with tempfile.TemporaryDirectory(prefix="v3fam_d6_reader_") as td:
                chk = reader_check(default_backend(backend), tagged, td)
            bad = sorted(k for k, v in chk.items() if not v.get("ok"))
            exif_rec["reader_check"] = {"checked": len(chk), "failed": bad}
            if bad:
                raise F.V3famRefused("EXIF_STAGING is symlink but rfdetr's reader does not yield the labels' frame "
                                     "for %d tagged image(s) (e.g. %s): the rehearsal's outcome does not hold here"
                                     % (len(bad), bad[:3]))
        os.rename(str(tmp), str(final))
    except BaseException:
        shutil.rmtree(str(tmp), ignore_errors=True)
        raise
    rec = {"format": FMT_STAGE, "source": source, "n_train": len(rows),
           "keys_sha256": C.sha256_text("\n".join(keys)), "labels_sha256": C.sha256_text("\n".join(label_shas)),
           "valid": {"n": len(vk), "share": VALID_SHARE, "seed_text": VALID_SEED_TEXT % "<key>",
                     "keys": sorted(vk), "keys_sha256": C.sha256_text("\n".join(sorted(vk))),
                     "subset_of_train": vk <= set(keys)},
           "exif": exif_rec, "data_yaml_sha256": C.sha256_file(final / "data.yaml"),
           "holdout_check": {"exams": dict(hold["counts"]), "hits": 0, "roots": hold["roots"],
                             "base_v2_rows_shared_with": shared,
                             "rule": "dev, test and ImageWeeds by key, sha256 and resolved path; nothing resolving into "
                                     "INC_DIR/exams, the probe's root or the zoo's root; ood22's, ood23's and the "
                                     "probe's images are base_v2's own rows and are staged (Revision 29)"},
           "d0s": d0s,
           "test_folder": False, "class_names": list(CLASS_NAMES), "decided_by": F.DECIDED_BY,
           "code": F.code_shas(), "created_utc": _utc()}
    F.write_once(stage_record_path(), rec)
    log("staged base_v2 for RF-DETR: %d train, %d valid (in train), EXIF %s (%d tagged)"
        % (len(rows), len(vk), exif, len(tagged)))
    return verify_stage()


def verify_stage():
    """The staging as stage.json records it: every train/ and valid/ entry, the labels' sha256, valid in train,
    no test/ folder and no holdout image anywhere (rechecked). Returns {record, sha256, entries}."""
    rec = _read_json(stage_record_path())
    if not isinstance(rec, dict) or rec.get("format") != FMT_STAGE:
        raise F.V3famRefused("%s is missing: D6's staging has not run (smoke stages it)" % stage_record_path())
    d = stage_dir()
    if not (d / "data.yaml").is_file() or C.sha256_file(d / "data.yaml") != rec["data_yaml_sha256"]:
        raise F.V3famRefused("%s does not hash as stage.json records" % (d / "data.yaml"))
    hold = holdout_index()
    n = check_tree(d, hold)
    keys = {}
    for split in SPLITS:
        imgs = sorted(os.listdir(str(d / split / "images")))
        labs = sorted(os.listdir(str(d / split / "labels")))
        keys[split] = sorted(os.path.splitext(f)[0] for f in imgs)
        if keys[split] != sorted(os.path.splitext(f)[0] for f in labs):
            raise F.V3famRefused("%s/%s's images and labels differ" % (d, split))
        for f in imgs:
            p = d / split / "images" / f
            if not p.is_symlink() or not p.exists():
                raise F.V3famRefused("%s is not a symlink to an existing image" % p)
    if C.sha256_text("\n".join(keys["train"])) != rec["keys_sha256"]:
        raise F.V3famRefused("train/ holds other keys than stage.json records")
    if keys["valid"] != rec["valid"]["keys"] or not set(keys["valid"]) <= set(keys["train"]):
        raise F.V3famRefused("valid/ is not the recorded sample of train/")
    lab = [hashlib.sha256((d / "train" / "labels" / ("%s.txt" % k)).read_bytes()).hexdigest() for k in keys["train"]]
    if C.sha256_text("\n".join(lab)) != rec["labels_sha256"]:
        raise F.V3famRefused("train/labels do not hash as stage.json records")
    for k, sha in (rec["exif"].get("transposed") or {}).items():
        if C.sha256_file(d / "transposed" / ("%s.jpg" % k)) != sha:
            raise F.V3famRefused("the transposed copy of %s does not hash as recorded" % k)
    return {"record": rec, "sha256": _sha(stage_record_path()), "entries": n}


# ------------------------------------------------------------------ the pretrained weights
def pretrain_check():
    """REPO/rf-detr-large-2026.pth: its size and rfdetr's registry MD5; its sha256 recorded."""
    p = pretrain_path()
    if p.is_symlink() or not p.is_file():
        raise F.V3famRefused("%s is missing (or a symlink): D6 trains from that file, offline" % p)
    size = p.stat().st_size
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            md5.update(b)
            sha.update(b)
    if size != PRETRAIN_BYTES or md5.hexdigest() != PRETRAIN_MD5:
        raise F.V3famRefused("%s is %d bytes, MD5 %s: not rfdetr's %s (%d bytes, MD5 %s)"
                             % (p, size, md5.hexdigest(), PRETRAIN, PRETRAIN_BYTES, PRETRAIN_MD5))
    return {"file": PRETRAIN, "bytes": size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


# ------------------------------------------------------------------ one training attempt
class Attempt(object):
    """A fresh output directory for one training attempt; the attempt removes only the files it wrote there
    itself (OWN_FILES, regular files written after it started) and refuses anything else."""

    def __init__(self, parent):
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        n = 1
        while (parent / ("attempt_%d" % n)).exists():
            n += 1
        self.dir = parent / ("attempt_%d" % n)
        self.started = time.time()
        self.dir.mkdir()                     # exist_ok False: every file in it is this attempt's
        self.removed = []

    def owned(self, p):
        p = Path(p)
        try:
            st = p.lstat()
        except OSError:
            return False
        return (p.parent.resolve() == self.dir.resolve() and p.name in OWN_FILES and not p.is_symlink()
                and p.is_file() and st.st_mtime >= self.started - 1.0)

    def remove(self, p):
        """Remove one file the attempt wrote; refuse (V3famRefused) anything else."""
        p = Path(p)
        if not self.owned(p):
            raise F.V3famRefused("%s was not written by this attempt (or is not one of %s): not removed"
                                 % (p, list(OWN_FILES)))
        size = p.stat().st_size
        p.unlink()
        self.removed.append({"file": p.name, "bytes": size})

    def clean(self):
        """Remove every OWN_FILES file present in the attempt's directory (each checked by remove)."""
        for name in OWN_FILES:
            p = self.dir / name
            if p.exists() or p.is_symlink():
                self.remove(p)
        return list(self.removed)


class _Term(object):
    """SIGTERM (scancel, the walltime) raises SystemExit in the training process, so its clean-up runs."""

    def __enter__(self):
        self.old = None
        try:
            self.old = signal.signal(signal.SIGTERM, self._raise)
        except ValueError:                   # not the main thread (tests): nothing to install
            self.old = None
        return self

    @staticmethod
    def _raise(signum, frame):
        raise SystemExit(143)

    def __exit__(self, *exc):
        if self.old is not None:
            signal.signal(signal.SIGTERM, self.old)
        return False


def _check_trained(res, epochs, out):
    """The trainer's facts the protocol fixes: run_test false, the archive interval above the epochs, every epoch
    completed, not interrupted, no archive checkpoint written."""
    cb = res.get("callbacks") or {}
    probs = []
    if cb.get("run_test") is not False:
        probs.append("run_test is %r" % cb.get("run_test"))
    if not int(cb.get("archive_every_n_epochs") or 0) > epochs:
        probs.append("the archive checkpoint interval %r does not exceed %d epochs" % (
            cb.get("archive_every_n_epochs"), epochs))
    if int(res.get("epoch", -1)) != epochs or res.get("interrupted"):
        probs.append("the trainer ended at epoch %r of %d (interrupted %r)" % (res.get("epoch"), epochs,
                                                                             res.get("interrupted")))
    arch = sorted(f for f in os.listdir(str(out)) if ARCHIVE_RE.match(f))
    if arch:
        probs.append("archive checkpoints were written: %s" % arch)
    if list(res.get("class_names") or []) != list(CLASS_NAMES):
        probs.append("the model's classes are %s" % res.get("class_names"))
    if probs:
        raise F.V3famRefused("the RF-DETR run is not D6's pre-registered one: %s" % "; ".join(probs))
    return {"run_test": False, "archive_every_n_epochs": int(cb["archive_every_n_epochs"]), "epochs": epochs,
            "archive_checkpoints": [], "ema": bool(cb.get("ema"))}


def _export(backend, res, dest):
    """The final EMA weights in rfdetr's stripped {model, args, epoch} format, written once at dest (a staged
    file hard-linked into place), reloaded and checked equal to the EMA callback's state and the in-process
    module's, exactly. Never reads a checkpoint_best_* file. Returns its record."""
    dest = Path(dest)
    if dest.exists() or dest.is_symlink():
        raise F.V3famRefused("%s exists: the final weights are written once" % dest)
    if not backend.states_equal(res["ema_state"], res["inprocess_state"]):
        raise F.V3famRefused("the in-process module does not hold the EMA callback's weights after fit")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = dest.with_name(".%s.%d.%d.staged" % (dest.name, os.getpid(), time.monotonic_ns()))
    try:
        backend.save({"model": res["ema_state"], "args": res["args"], "epoch": int(res["epoch"])}, staged)
        back = backend.load_state(staged)
        if set(back) < {"model", "args", "epoch"} or not backend.states_equal(back["model"], res["ema_state"]):
            raise F.V3famRefused("the exported weights do not reload as the in-process EMA weights exactly")
        sha = C.sha256_file(staged)
        os.link(str(staged), str(dest))
    finally:
        if staged.exists():
            staged.unlink()
    return {"file": dest.name, "sha256": sha, "bytes": dest.stat().st_size, "format": "{model, args, epoch}",
            "epoch": int(res["epoch"]), "equals_inprocess_ema": True}


def _pils(rows):
    from PIL import Image, ImageOps
    out = []
    for r in rows:
        with Image.open(r["image"]) as im0:
            out.append(ImageOps.exif_transpose(im0).convert("RGB"))
    return out


def _compare_predictions(a, b):
    """Max differences of two models' predictions on the same images (same rows, classes, boxes, scores)."""
    import numpy as np
    dx = dc = 0.0
    for (xa, ca, ka), (xb, cb, kb) in zip(a, b):
        if len(ca) != len(cb) or not np.array_equal(np.asarray(ka), np.asarray(kb)):
            return {"same_rows": False}
        if len(ca):
            dx = max(dx, float(np.abs(np.asarray(xa) - np.asarray(xb)).max()))
            dc = max(dc, float(np.abs(np.asarray(ca) - np.asarray(cb)).max()))
    return {"same_rows": True, "xyxy_max_abs_diff_px": dx, "conf_max_abs_diff": dc,
            "ok": bool(dx <= PRED_XYXY_TOL and dc <= PRED_CONF_TOL)}


def _train_attempt(backend, parent, seed, epochs, dest, smoke):
    """One training attempt (fresh directory), its checks, the export, the clean-up (always). Returns its record."""
    srec = verify_stage()
    pre = pretrain_check()
    att = Attempt(parent)
    rec = {"attempt": att.dir.name, "seed": int(seed), "epochs": epochs, "stage_sha256": srec["sha256"],
           "pretrain": pre, "recipe": dict(RECIPE, epochs=epochs, resolution=RESOLUTION), "started_utc": _utc()}
    exported = None
    try:
        with _Term():
            res = backend.train(stage_dir(), att.dir, seed, epochs, pretrain_path())
        rec["trainer"] = _check_trained(res, epochs, att.dir)
        rec["fit_seconds"] = round(float(res["seconds"]), 3)
        if isinstance(res.get("epoch_clock"), dict):
            rec["epoch_clock"] = res["epoch_clock"]
        rec["seed_mechanism"] = res.get("seed_mechanism")
        rec["train_config"] = {k: v for k, v in sorted((res.get("args") or {}).items())
                               if isinstance(v, (int, float, str, bool, type(None), list))}
        exported = _export(backend, res, dest)
        rec["export"] = exported
        if smoke:
            rows = sorted(C.read_manifest(C2.v2_manifest_path("base_v2")) if srec["record"]["source"]["sha256"]
                          else _stage_rows(), key=lambda r: r["key"])[:SMOKE_PRED_IMAGES]
            pils = _pils(rows)
            mine = backend.predict(res["model"], pils, THRESHOLD)
            fresh = backend.predict(backend.load_model(dest), pils, THRESHOLD)
            cmp_ = _compare_predictions(mine, fresh)
            rec["export_predictions"] = dict(cmp_, images=len(rows))
            if not cmp_.get("ok"):
                raise F.V3famRefused("the exported weights do not predict as the in-process model on %d training "
                                     "images (%s)" % (len(rows), cmp_))
        check_tree(att.dir, holdout_index())
    except BaseException as e:
        rec["failed"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        if exported is not None and dest.exists():
            dest.unlink()                    # written by this attempt only, never recorded
        raise
    finally:
        try:
            rec["cleanup"] = {"removed": att.clean(), "kept": sorted(os.listdir(str(att.dir)))}
        except F.V3famRefused as e:
            rec["cleanup"] = {"refused": str(e)}
            log("WARNING: clean-up refused: %s" % e)
        rec["finished_utc"] = _utc()
        if rec.get("failed"):
            TS._write_json(att.dir / "attempt.json", rec)
    return rec, res


def _stage_rows():
    """The staged train rows (tests: stage.json records no manifest): key and image from the staged links."""
    d = stage_dir() / "train" / "images"
    return [{"key": os.path.splitext(f)[0], "image": os.path.realpath(str(d / f))} for f in sorted(os.listdir(str(d)))]


# ------------------------------------------------------------------ smoke and train (L23K)
def smoke_record():
    return _read_json(smoke_dir() / "smoke.json")


def smoke(backend=None, exif=None, rows=None):
    """L23K's first job (module docstring): the staging (once), one epoch, the checks, the projection. Writes
    v3fam/v1/d6/smoke/smoke.json once; status complete, or 'not run (walltime)' when the projection exceeds
    28.8 h."""
    F.check_frozen()
    supersede_smoke()                        # deviation 5: the smoke of 2026-10-21 moved aside first
    if (smoke_dir() / "smoke.json").exists():
        raise F.V3famRefused("%s exists: the smoke runs once" % (smoke_dir() / "smoke.json"))
    backend = default_backend(backend)
    with F.e_trees_unchanged() as before:
        stage(exif=exif, rows=rows, backend=backend)
        dest = smoke_dir() / FINAL
        rec, _res = _train_attempt(backend, smoke_dir(), 0, SMOKE_EPOCHS, dest, smoke=True)
        secs, measure = smoke_epoch_seconds(rec, backend)
        proj = projection_h(secs)
        ok = proj <= walltime_cap_h() + 1e-9
        exp = rec["export"]
        os.unlink(str(dest))                 # the smoke's export served its checks; written by this job
        doc = {"format": FMT_SMOKE, "status": "complete" if ok else NOT_RUN_WALLTIME, "epoch_seconds": secs,
               "epoch_time": measure, "fit_seconds": rec["fit_seconds"],
               "setup_seconds": (rec.get("epoch_clock") or {}).get("setup_seconds"),
               "projection_h": proj, "cap_h": walltime_cap_h(), "walltime_h": WALLTIME_H, "walltime_ok": ok,
               "rule": "epoch time x %d + %.1f h <= %.1f x %.0f h" % (EPOCHS, OVERHEAD_H, WALLTIME_SHARE, WALLTIME_H),
               "attempt": rec, "export_removed": {"file": exp["file"], "sha256": exp["sha256"]},
               "e_trees": {"files": len(before), "sha256": C.sha256_text(json.dumps(before, sort_keys=True))},
               "versions": _versions(backend), "decided_by": F.DECIDED_BY, "code": F.code_shas(),
               "created_utc": _utc()}
    F.write_once(smoke_dir() / "smoke.json", doc)     # only once the E-series trees were found unchanged
    log("smoke: one epoch in %.0fs (%s; the whole fit %.0fs), projection %.1f h (cap %.1f h): %s" % (
        secs, measure, float(rec["fit_seconds"]), proj, walltime_cap_h(), doc["status"]))
    return doc


def supersede_smoke():
    """inc2.v3fam_supersede for D6's smoke record only (group d6): the smoke of 2026-10-21 (job 49542653), which took
    the whole one-epoch fit as the epoch time, moved aside byte for byte into d6/smoke/superseded_2026-10-24/ with
    its manifest before this smoke runs (deviation 5). The score jobs move only their own group, so the two never
    move the same file at once. A move that would overwrite a file refuses the job."""
    from . import v3fam_supersede as SUP
    try:
        return SUP.apply(C.INC_DIR, repo=C.REPO, log=log, groups=(SUP.GROUP_D6,))
    except SUP.SupersedeRefused as e:
        raise F.V3famRefused(str(e))
    except OSError as e:                     # a full disk, a lost mount: the platform's, never the arm's
        raise F.V3famRefused("a superseded record could not be moved aside (%s: %s): it is left in place and "
                             "nothing is moved over another file" % (type(e).__name__, e))


def smoke_epoch_seconds(rec, backend):
    """(epoch seconds, how measured) for the walltime rule: the epoch clock's span from the first training epoch's
    start to the end of fit (EPOCH_TIME_DEF; deviation 5); a backend that keeps no clock (the tests' fakes) gives
    the whole fit, which can only be longer. The real backend's clock must have seen the epoch, or the smoke refuses
    (never the whole fit again, as on 2026-10-21)."""
    ck = rec.get("epoch_clock") or {}
    if ck.get("epoch_seconds") is not None and int(ck.get("epochs_started") or 0) >= 1:
        return float(ck["epoch_seconds"]), "epoch clock: " + EPOCH_TIME_DEF
    if getattr(backend, "epoch_clock", False):
        raise F.V3famRefused("the epoch clock saw no training epoch (%s): the smoke records no epoch time" % ck)
    return float(rec["fit_seconds"]), "the whole fit (the backend keeps no epoch clock)"


def _versions(backend):
    try:
        return backend.versions()
    except Exception as e:                   # noqa: BLE001 - recorded, never fatal
        return {"error": "%s: %s" % (type(e).__name__, e)}


def train(seed, backend=None):
    """One of L23K's seed jobs: refused unless the smoke completed within the walltime rule and on the same
    staging and pretrained file; trains 50 epochs into a fresh attempt directory, exports s<k>/final_ema.pth once
    and writes s<k>/train.json once."""
    F.check_frozen()
    seed = int(seed)
    if seed not in F.SEEDS:
        raise F.V3famRefused("D6 trains seeds %s, not %d" % (list(F.SEEDS), seed))
    sm = smoke_record()
    if not isinstance(sm, dict) or sm.get("format") != FMT_SMOKE:
        raise F.V3famRefused("the smoke has not run: L23K's seeds follow it")
    if sm.get("status") != "complete" or sm.get("walltime_ok") is not True:
        raise F.V3famRefused("the smoke projected %.1f h against the %.1f h rule: D6 is %s"
                             % (float(sm.get("projection_h") or 0), walltime_cap_h(), NOT_RUN_WALLTIME))
    tj = seed_dir(seed) / "train.json"
    if tj.exists() or final_path(seed).exists():
        raise F.V3famRefused("seed %d is trained already (%s): written once" % (seed, tj))
    srec = verify_stage()
    if srec["sha256"] != sm["attempt"]["stage_sha256"]:
        raise F.V3famRefused("the staging changed since the smoke")
    if pretrain_check()["sha256"] != sm["attempt"]["pretrain"]["sha256"]:
        raise F.V3famRefused("the pretrained file changed since the smoke")
    backend = default_backend(backend)
    with F.e_trees_unchanged() as before:
        rec, _res = _train_attempt(backend, seed_dir(seed), seed, EPOCHS, final_path(seed), smoke=False)
        doc = {"format": FMT_TRAIN, "arm": ARM, "seed": seed, "status": "done", "final": rec["export"],
               "attempt": rec, "smoke_sha256": _sha(smoke_dir() / "smoke.json"), "versions": _versions(backend),
               "e_trees": {"files": len(before), "sha256": C.sha256_text(json.dumps(before, sort_keys=True))},
               "decided_by": F.DECIDED_BY, "code": F.code_shas(), "created_utc": _utc()}
    F.write_once(tj, doc)                    # only once the E-series trees were found unchanged
    log("D6 seed %d trained: %s (%s), %.1f h" % (seed, FINAL, rec["export"]["sha256"][:12],
                                                 float(rec["fit_seconds"]) / 3600.0))
    return doc


def final_record(seed):
    """{path, sha256, train_json_sha256} of a trained seed, its final weights hashing as train.json records."""
    tj = seed_dir(seed) / "train.json"
    d = _read_json(tj)
    if not isinstance(d, dict) or d.get("format") != FMT_TRAIN or d.get("status") != "done":
        raise F.V3famRefused("D6 seed %d has no train.json: not trained (or %s)" % (seed, NOT_RUN_WALLTIME))
    p = final_path(seed)
    if p.is_symlink() or not p.is_file() or C.sha256_file(p) != d["final"]["sha256"]:
        raise F.V3famRefused("%s is missing or does not hash as train.json records" % p)
    return {"seed": int(seed), "path": str(p), "file": SYS.rel(p), "sha256": d["final"]["sha256"],
            "train_json_sha256": _sha(tj)}


def read_sources():
    """What D6's reference read scores (V3F.3.6): the three final EMA files and the scaffold."""
    scaffold = SYS.e3_source("M", 0)
    return {"mode": "inject_rows", "scaffold": SYS.source_record(scaffold),
            "seeds": [{k: v for k, v in final_record(s).items() if k != "path"} for s in F.SEEDS],
            "inference": {"resolution": RESOLUTION, "threshold": THRESHOLD, "num_select": NUM_SELECT}}


# ------------------------------------------------------------------ predictions -> rows (the converter)
def convert(xyxy, conf, class_id, names=CLASS_NAMES):
    """One image's predictions as inject_rows rows in original pixels: the class index mapped to the INC id
    (class_map: by name), rfdetr's background index dropped and counted, any other index refused. Returns
    ({xyxy float64 [n, 4], conf float32 [n], cls float64 [n]}, {background})."""
    import numpy as np
    cmap = class_map(names)
    xy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    cf = np.asarray(conf, dtype=np.float64).reshape(-1)
    ci = np.asarray(class_id, dtype=np.float64).reshape(-1)
    if not (len(xy) == len(cf) == len(ci)):
        raise F.V3famRefused("%d boxes, %d scores and %d classes" % (len(xy), len(cf), len(ci)))
    if not (np.isfinite(xy).all() and np.isfinite(cf).all() and np.isfinite(ci).all()):
        raise F.V3famRefused("a prediction holds a non-finite value")
    if np.any(ci != np.rint(ci)) or np.any(ci < 0) or np.any(ci > BACKGROUND_ID):
        raise F.V3famRefused("a prediction's class index lies outside 0..%d (background %d): %s"
                             % (NUM_CLASSES - 1, BACKGROUND_ID, sorted(set(ci.tolist()))[:5]))
    if np.any(cf < 0) or np.any(cf > 1):
        raise F.V3famRefused("a prediction's score lies outside [0, 1]")
    ci = ci.astype(np.int64)
    keep = ci != BACKGROUND_ID
    return ({"xyxy": xy[keep], "conf": cf[keep].astype(np.float32), "cls": cmap[ci[keep]].astype(np.float64)},
            {"background": int((~keep).sum())})


def predict_exam(backend, model, exam):
    """{key: rows} for every image of exam (the manifest's own images, EXIF-transposed) and the counts."""
    rows = SYS.exam_rows(exam)
    names = backend.model_class_names(model)
    out = {}
    st = {"images": len(rows), "rows": 0, "background": 0, "max_rows_per_image": 0, "outside_image_1px": 0}
    for i in range(0, len(rows), PRED_BATCH):
        chunk = rows[i:i + PRED_BATCH]
        pils = _pils(chunk)
        preds = backend.predict(model, pils, THRESHOLD)
        if len(preds) != len(chunk):
            raise F.V3famRefused("%d predictions for %d images" % (len(preds), len(chunk)))
        for r, im, (xy, cf, ci) in zip(chunk, pils, preds):
            if len(cf) > NUM_SELECT:
                raise F.V3famRefused("%d predictions for %s, more than num_select %d" % (len(cf), r["key"],
                                                                                       NUM_SELECT))
            got, c = convert(xy, cf, ci, names)
            w, h = im.size
            b = got["xyxy"]
            if len(b):
                st["outside_image_1px"] += int(((b[:, 0] < -1) | (b[:, 1] < -1) | (b[:, 2] > w + 1)
                                                | (b[:, 3] > h + 1)).sum())
            out[r["key"]] = got
            st["rows"] += int(len(got["conf"]))
            st["background"] += c["background"]
            st["max_rows_per_image"] = max(st["max_rows_per_image"], int(len(got["conf"])))
    return out, st


# ------------------------------------------------------------------ the score job (L23J --arm D6)
def score_d6(lock_check=True, batch=S.BATCH, device=None, reported=True, backend=None):
    """D6's scoring job: each seed's final EMA weights predict dev (deciding: P5, Tier C), ImageWeeds and
    ooddev_v1 (reported); the rows go through the family inject_rows pass on b_v2_m640 seed 0's scaffold."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        names = F.equivalence_names("dev")
        eqs = F.require_equivalence(names)
        finals = {s: final_record(s) for s in F.SEEDS}
        scaffold = SYS.e3_source("M", 0)
        backend = default_backend(backend)
        dev_files, passes = [], []
        for s in F.SEEDS:
            model = None
            for exam in ("dev",) + (REPORTED_EXAMS if reported else ()):
                item = {"dir": DIR, "seed": s, "exam": exam}
                got = SYS.existing(DIR, s, exam)
                if got is not None:
                    if exam == "dev":
                        dev_files.append(got)
                    else:
                        passes.append(dict(item, file=got["file"], sha256=got["sha256"], status="kept"))
                    continue

                def one(s=s, exam=exam):
                    nonlocal model
                    if model is None:
                        model = backend.load_model(finals[s]["path"])
                        class_map(backend.model_class_names(model))
                    rows, st = predict_exam(backend, model, exam)
                    cfg = {"mode": "inject_rows", "nms": "locked", "inject": rows}
                    ex_names = F.equivalence_names(exam)
                    if exam in F.ROOT_EXAMS:
                        F.require_equivalence(ex_names)
                        ps = SYS.in_root(scaffold["weights"], exam, cfg, ARM, kw)
                    else:
                        if exam != "dev":
                            F.require_equivalence(ex_names)
                        ps = F._family_pass(scaffold["weights"], exam, cfg, ARM, **kw)
                    x = {"detr": {"final": {k: v for k, v in finals[s].items() if k != "path"},
                                  "inference": {"resolution": RESOLUTION, "threshold": THRESHOLD,
                                                "num_select": NUM_SELECT, "batch": PRED_BATCH,
                                                "frame": "EXIF-transposed original pixels"},
                                  "converter": {"class_map": "index -> INC id by name (identity)",
                                                "background_dropped": st["background"]},
                                  "predictions": st},
                         "scaffold": SYS.source_record(scaffold)}
                    return SYS.commit(ARM, DIR, s, exam, ps, cfg, ex_names, extra=x)
                if exam == "dev":
                    dev_files.append(one())
                    log("D6 seed %d on dev: species %s" % (s, dev_files[-1].get("species_map50_95")))
                else:
                    SYS.reported_pass(passes, item, one)
            model = None
        rec = SYS.score_record(ARM, dev_files, eqs, before=before,
                               reported=SYS.write_reported(ARM, passes) if reported else {"status": "not run"},
                               extra={"finals": [{k: v for k, v in finals[s].items() if k != "path"}
                                                 for s in F.SEEDS]})
    log("D6 scored (%.0fs)" % (time.time() - t0))
    return rec


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F D6: RF-DETR-L 12-class on base A (staging, smoke, train, score).")
    ap.add_argument("command", choices=("smoke", "train", "score", "stage", "rehearse-exif", "sources", "check-env"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="rehearse-exif: a directory outside INC_DIR")
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.v3fam_detr] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    try:
        if a.command == "smoke":
            smoke()
        elif a.command == "train":
            if a.seed is None:
                raise F.V3famRefused("train needs --seed 0|1|2")
            train(a.seed)
        elif a.command == "stage":
            print(json.dumps({k: v for k, v in stage()["record"].items() if k != "valid"}, indent=1,
                             sort_keys=True))
        elif a.command == "rehearse-exif":
            if not a.out:
                raise F.V3famRefused("rehearse-exif needs --out DIR (outside INC_DIR)")
            rehearse_exif(a.out)
        elif a.command == "sources":
            print(json.dumps(read_sources(), indent=1, sort_keys=True))
        elif a.command == "check-env":
            check_env()
        else:
            score_d6(lock_check=not a.no_lock_check_for_tests, batch=a.batch or S.BATCH, device=a.device)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, SN.NativeRefused, V.VerifyError,
            C2.Inc2Error, ValueError) as e:
        print("[inc2.v3fam_detr] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
