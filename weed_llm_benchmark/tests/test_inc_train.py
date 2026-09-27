#!/usr/bin/env python3
"""The INC executor, end to end on the CPU (docs/INCREMENTAL_PROTOCOL_RUNNER.md,
"Executor").

A fake split set is built in a temporary INC_DIR, consistent with inc.common
and the scorer's test mode: a dev and a test exam (manifests, materialised exam
dirs), LOCK.json holding their hashes and scorer.py's, a complete never-train
index over both, a train manifest, and an experiment whose exp.json says
"testing": true, as the driver writes it (the scorer then keeps the protocol's
imgsz 640 / batch 32 and runs on the CPU); a second experiment gives explicit
scorer settings (imgsz 64, batch 8). INC_SCORER_TESTING=1 is set, so every
score is production=false with a TEST- stamp; the executor never sets it.

What is pinned:
- full, freeze and lora runs each leave weights/final.pt, run.json (every
  field the runner doc lists) and scores/dev.json, and nothing else of size:
  no materialised data (nor a .trash-* copy of it), no last.pt / best.pt /
  last_merged.pt, no owner file;
- the recipe's lr0, optimizer, seed and epochs reach Model.train (recorded by
  a wrapper around it) and train_lora, with val=False, plots=False,
  project=<run dir>, name='train', exist_ok=True;
- the weights copied are the trainer's own save_dir's: a run whose trainer
  saves elsewhere, next to a decoy last.pt at the usual path, gets the
  trainer's weights;
- the guard refuses a manifest holding a dev image or a re-encoded copy of
  one, and the refusal is a failed run.json; a re-run is attempt 2;
- a spec listing 'test' is refused unless kind is final, as are unknown spec
  or recipe keys, a missing recipe key, optimizer 'auto', a label Ultralytics
  would drop, a testing experiment without INC_SCORER_TESTING; a production
  run with a recipe other than the protocol's (the pilot's own recipes are
  the protocol's), and (without CUDA) a production run with the protocol's;
- with CUDA simulated: a production final run scores through a stub scorer
  that sees no INC_SCORER_TESTING and no test setting; a production score
  that is not production=true, or a testing score that is, fails the run;
  a production training run refuses while the checkpoint Ultralytics' AMP
  check loads is absent (it would be downloaded in place);
- code drift: a nested git-tracked copy that differs fails the run at stage
  code; INC_ALLOW_DRIFT=1 runs it with a warning;
- a soup of two runs is their elementwise mean in fp32 and loads with plain
  YOLO(); members of different architectures, or the same weights twice, are
  refused; a soup whose scoring failed re-scores its final.pt on the next
  attempt instead of averaging again;
- a final run scores its init (test included) without training;
- re-runs (a cand): a done run is a no-op; a changed score file is re-scored,
  not retrained; a re-score that fails, then an attempt that fails before
  scoring, still keep final.pt, and the next attempt re-scores it; a missing
  final.pt, or a train manifest whose bytes changed, retrains; --force
  re-runs a done run as attempt 2;
- a JPEG with bytes after its EOI marker is copied into data/images (not
  linked), and the source is untouched by Ultralytics' re-save; exact
  duplicate images and dHash-0 pairs are counted in run.json;
- a <10 px image that Ultralytics drops fails the run before any epoch; the
  data dir is renamed aside, run.json is written, and only then is it
  deleted;
- a skipped save (weights not from the final epoch) fails the run as
  diverged, for full and lora; cache 'ram' beyond 60% of the job's memory
  falls back to no cache, recorded;
- the run-dir lock: a busy flock or a live owner file (after waiting) exits 3
  touching nothing; a stale owner file (old heartbeat, or an exited process
  on this host) is taken over and recorded; flock unsupported on the mount
  leaves the owner file in charge; any other flock error is a failed run.json
  at stage lock (never over a done one); an attempt whose owner file was
  taken over writes and removes nothing and exits 3;
- units: flock coherence from mountinfo lines, the cgroup / Slurm memory
  limit, the RAM-cache estimate, the AMP-check checkpoint lookup, the
  hashing pool cancelled on an exception, labels.cache counts;
- SIGTERM during training still writes a failed run.json;
- run_inc_job.sh runs line SLURM_ARRAY_TASK_ID+1 of its list, advances the
  driver when INC_JOB_ADVANCE=1, or (auto) only when --flock-check passes,
  never when 0, and exits with the executor's status.

Run:  python3 tests/test_inc_train.py
"""
import contextlib
import errno
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import zipfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_inc_train_"))
# common reads INC_DIR and REPO at import: this test never sees the machine's real ones.
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["INC_SCORER_TESTING"] = "1"
for _k in ("INC_ALLOW_DRIFT", "INC_JOB_ADVANCE", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
           "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
    os.environ.pop(_k, None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import train as T  # noqa: E402

FAILURES = []
EXP = "t_exp"
IMGSZ, BATCH = 64, 8
EXP_DICT = "t_dict"
EXP_PROD = "t_prod"
TESTING_DICT = {"imgsz": IMGSZ, "batch": BATCH, "device": "cpu"}
RECIPE = {"trainer": "full", "epochs": 1, "optimizer": "SGD", "lr0": 0.0123, "lrf": 0.01,
          "momentum": 0.9, "weight_decay": 0.0005, "warmup_epochs": 1, "warmup_bias_lr": 0.01,
          "cos_lr": True, "freeze": None, "lora": None, "imgsz": IMGSZ, "batch": BATCH, "seed": 0,
          "cache": False, "workers": 0, "close_mosaic": 0, "deterministic": True}
DOC_FIELDS = ("status", "error", "attempt", "seconds", "n_train_images", "n_train_boxes",
              "trainable_params", "total_params", "guard", "weights_sha256", "ultralytics_version",
              "hostname", "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id")
N_TRAIN = 24


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def sha(path):
    return C.sha256_file(path)


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def env(**values):
    """Set (a str) or unset (None) environment variables for the block."""
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------- fixtures
def make_image(path, seed):
    """96x96 JPEG: a random 9x8 low-frequency background (so every image has
    its own dHash) and 1-2 coloured rectangles; returns YOLO boxes."""
    rng = np.random.RandomState(seed)
    low = rng.randint(30, 220, (8, 9)).astype(np.uint8)
    im = Image.fromarray(low).resize((96, 96), Image.BICUBIC).convert("RGB")
    draw = ImageDraw.Draw(im)
    boxes = []
    for _ in range(1 + seed % 2):
        c = int(rng.randint(0, C.NC))
        w, h = rng.randint(20, 40, 2)
        x, y = rng.randint(0, 96 - w), rng.randint(0, 96 - h)
        draw.rectangle([x, y, x + w, y + h], fill=tuple(int(v) for v in rng.randint(0, 255, 3)))
        boxes.append((c, (x + w / 2) / 96, (y + h / 2) / 96, w / 96, h / 96))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return boxes


def make_rows(prefix, n, seed0, source, session=True):
    rows = []
    for i in range(n):
        key = "%s_%03d" % (prefix, i)
        img = TMP / "src" / prefix / (key + ".jpg")
        lab = TMP / "src" / (prefix + "_labels") / (key + ".txt")
        C.write_yolo(lab, make_image(img, seed0 + i))
        rows.append({"image": str(img), "label": str(lab), "sha256": sha(img), "label_sha256": sha(lab),
                     "source": source, "session": ("s%d" % (i % 3)) if session else "", "key": key})
    return rows


def extra_row(key, image, boxes, source="extra"):
    lab = TMP / "src" / "extra_labels" / (key + ".txt")
    C.write_yolo(lab, boxes)
    return {"image": str(image), "label": str(lab), "sha256": sha(image), "label_sha256": sha(lab),
            "source": source, "session": "", "key": key}


def write_rows_manifest(name, rows):
    path = C.INC_DIR / EXP / "manifests" / ("%s.jsonl" % name)
    C.write_manifest(path, rows)
    return path


def cold_checkpoint(path, nc=C.NC, seed=0):
    """yolo11n.yaml at nc classes, its class biases raised to -1 (sigmoid 0.27)
    so that one epoch at imgsz 64 still leaves a model that predicts boxes: the
    scorer's collapsed_ap raises on an exam with no prediction at all (see
    test_scorer_zero_predictions)."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(seed)
    net = DetectionModel("yolo11n.yaml", nc=nc, verbose=False)
    det = net.model[-1]
    with torch.no_grad():
        for level in range(det.nl):
            det.cv3[level][-1].bias.fill_(-1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net, "train_args": {}}, path)
    return path


def build_fixture():
    """The fake split set; returns the train manifest and the rows."""
    dev = make_rows("dv", 8, 100, "dev", session=False)
    test = make_rows("te", 6, 200, "test", session=False)
    manifests = {}
    for split, rows in (("dev", dev), ("test", test)):
        manifests[split] = C.write_manifest(C.manifest_path(split), rows)
        C.materialise(rows, C.EXAMS_DIR / split)
    C.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.LOCK_PATH.write_text(json.dumps({"manifests": manifests,
                                       "scorer_sha256": sha(pathlib.Path(S.__file__).resolve())}))
    entries = [[C.dhash(r["image"]), split, r["key"]] for split, rows in (("dev", dev), ("test", test))
               for r in rows]
    C.NEVER_TRAIN_INDEX.write_text(json.dumps({"entries": entries, "bits": C.HOLDOUT_NEAR_DUP_BITS,
                                               "complete": True, "min_expected": len(entries)}))
    train = make_rows("tr", N_TRAIN, 0, "synthetic_train")
    (C.INC_DIR / EXP).mkdir(parents=True, exist_ok=True)
    (C.INC_DIR / EXP / "exp.json").write_text(json.dumps({"testing": True}))
    (C.INC_DIR / EXP_DICT).mkdir(parents=True, exist_ok=True)
    (C.INC_DIR / EXP_DICT / "exp.json").write_text(json.dumps({"testing": TESTING_DICT}))
    cold_checkpoint(C.REPO / "yolo11n.pt")
    return write_rows_manifest("train_a", train), train, dev


def spec(run_id, kind, exams=("dev",), exp=EXP, **kw):
    out = C.INC_DIR / exp / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    s = {"exp": exp, "run_id": run_id, "kind": kind, "exams": list(exams), "out_dir": str(out)}
    s.update(kw)
    path = out / "spec.json"
    path.write_text(json.dumps(s, indent=1))
    return path


def recipe(**over):
    r = dict(RECIPE)
    r.update(over)
    return r


def protocol_recipes():
    """The pilot builder's cold and incremental recipes, seed set as the driver does."""
    from weed_optimizer_framework.tools.inc import pilot as P
    return dict(P.cold_recipe(), seed=0), {k: dict(v, seed=1) for k, v in P.inc_recipes().items()}


def run(path, *extra):
    return T.main(["--spec", str(path)] + list(extra))


def run_json(path):
    p = pathlib.Path(path).parent / T.RUN_JSON
    return json.loads(p.read_text()) if p.exists() else {}


def err(rj, n=400):
    return (rj.get("error") or "")[-n:]


def sizeable(out_dir):
    """Files a finished run keeps apart from spec/run.json/attempt/lock,
    final.pt, scores/ and the train logs: must be none."""
    left = []
    for dp, dns, fns in os.walk(out_dir):
        for n in fns:
            p = pathlib.Path(dp) / n
            rel = p.relative_to(out_dir)
            if (rel.parts[0] == "data" or rel.parts[0].startswith(T.TRASH_PREFIX)
                    or rel.parts[0].startswith(T.OWNER_NAME)
                    or (p.suffix == ".pt" and rel != pathlib.Path("weights/final.pt"))):
                left.append(str(rel))
    if (out_dir / "data").exists():
        left.append("data/")
    left += [p.name + "/" for p in out_dir.glob(T.TRASH_PREFIX + "*")]
    return left


@contextlib.contextmanager
def recording_train(redirect=None, decoy=None, inspect=None):
    """Wrap Ultralytics' Model.train: record every call's kwargs and the sha256
    of the trainer's last.pt; optionally make the trainer save under another
    name, with a decoy last.pt planted at <project>/train/weights/last.pt, and
    call inspect(kwargs) before training."""
    from ultralytics.engine.model import Model
    orig = Model.train
    calls = []

    def train(self, trainer=None, **kw):
        calls.append(dict(kw))
        if inspect:
            inspect(kw)
        if redirect:
            if decoy:
                d = pathlib.Path(kw["project"]) / "train" / "weights"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(decoy, d / "last.pt")
            kw = dict(kw, name=redirect)
        out = orig(self, trainer=trainer, **kw)
        calls[-1]["_last_sha256"] = sha(self.trainer.last) if pathlib.Path(self.trainer.last).exists() else None
        calls[-1]["_save_dir"] = str(self.trainer.save_dir)
        return out

    Model.train = train
    try:
        yield calls
    finally:
        Model.train = orig


@contextlib.contextmanager
def recording_lora():
    from weed_optimizer_framework.tools.inc import lora as L
    orig = L.train_lora
    calls = []

    def train_lora(*a, **kw):
        calls.append(dict(kw, _args=a))
        return orig(*a, **kw)

    L.train_lora = train_lora
    try:
        yield calls
    finally:
        L.train_lora = orig


# ------------------------------------------------------------------- tests
def test_scorer_zero_predictions():
    """Not the executor's contract, so reported, not failed: a run whose model
    predicts nothing on the whole exam should score 0, not crash the scorer
    (the executor would record a failed run at stage 'score')."""
    try:
        S.collapsed_ap(np.zeros((0, 10), dtype=bool), np.zeros(0), 5)
        print("  ok   (scorer) no predictions on an exam scores instead of raising")
    except ValueError as e:
        print("  NOTE scorer.collapsed_ap raises on an exam with no prediction at all (%s); "
              "such a run fails at stage 'score'. The fixtures here predict boxes." % e)


def test_units(train_rows):
    print("units: flock coherence, the protocol recipe, memory, AMP weights, hashing, labels.cache")
    mi = "\n".join([
        "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw",
        "40 22 0:40 / /ocean rw,relatime shared:20 - lustre 10.1.1.1@o2ib:/ocean rw,flock,lazystatfs",
        "41 22 0:41 / /jet/home rw,relatime shared:21 - nfs4 srv:/home rw,vers=4.1,local_lock=none",
        "42 22 0:42 / /scratch rw - lustre x@o2ib:/scratch rw,localflock",
        "43 22 0:43 / /old rw - lustre x@o2ib:/old rw,lazystatfs",
        "44 22 0:44 / /nfsl rw - nfs srv:/x rw,local_lock=flock",
        "45 22 0:45 / /my\\040dir rw - fuse.sshfs a:/ rw",
        "46 40 0:46 / /ocean/tmp rw - tmpfs tmpfs rw",
    ])
    cases = [("/ocean/projects/x/inc/pilot_v1", "/ocean", "lustre", True),
             ("/ocean", "/ocean", "lustre", True),
             ("/oceanic/x", "/", "ext4", True),
             ("/ocean/tmp/a", "/ocean/tmp", "tmpfs", True),
             ("/jet/home/u", "/jet/home", "nfs4", True),
             ("/scratch/a", "/scratch", "lustre", False),
             ("/old/a", "/old", "lustre", False),
             ("/nfsl/a", "/nfsl", "nfs", False),
             ("/my dir/a", "/my dir", "fuse.sshfs", None)]
    bad = []
    for path, mnt, fs, want in cases:
        m = T.mount_of(path, mi)
        v = T.flock_verdict(m)[0]
        if m is None or (m[0], m[1]) != (mnt, fs) or v is not want:
            bad.append((path, m and m[:2], v))
    check("mountinfo: the longest mount point holds a path; flock is coherent across nodes on Lustre "
          "'flock', NFS, node-local disks, not on 'localflock', noflock or local_lock=flock, unknown on "
          "fuse", not bad, bad)
    v = T.flock_coherence(TMP)
    rc = T.main(["--flock-check", str(TMP)])
    check("--flock-check exits 0 only when flock is coherent across nodes (here %s: %s)"
          % (v["cross_node"], v["reason"]), rc == (0 if v["cross_node"] is True else 1), rc)

    cold, incs = protocol_recipes()
    devs = {("base", "cold"): T.protocol_deviations("base", cold), ("union", "cold"): T.protocol_deviations("union", cold)}
    for name, r in incs.items():
        for kind in ("cand", "null"):
            devs[(kind, name)] = T.protocol_deviations(kind, r)
    check("the pilot's cold recipe (base, union) and its full / freeze / lora recipes (cand, null) are "
          "the protocol's", not any(devs.values()), {k: v for k, v in devs.items() if v})
    keys = lambda d: {x.split()[0] for x in d}  # noqa: E731
    check("the protocol table catches a cold recipe on a cand run, an incremental one on a base run, "
          "and the test recipe",
          keys(T.protocol_deviations("cand", cold)) == {"epochs", "lr0", "warmup_epochs", "warmup_bias_lr"}
          and keys(T.protocol_deviations("base", incs["lora"])) == {"trainer", "epochs", "warmup_epochs"}
          and keys(T.protocol_deviations("base", incs["full"])) == {"epochs", "lr0", "warmup_epochs"}
          and keys(T.protocol_deviations("cand", dict(incs["freeze"], freeze=10))) == {"freeze"}
          and keys(T.protocol_deviations("cand", dict(incs["lora"], lora={"rank": 8, "alpha": 32}))) == {"lora"}
          and {"imgsz", "batch", "epochs", "lr0"} <= keys(T.protocol_deviations("base", RECIPE)),
          T.protocol_deviations("base", incs["lora"]))

    root = TMP / "cg2"
    (root / "job" / "step").mkdir(parents=True)
    (root / "job" / "memory.max").write_text("1073741824\n")
    (root / "job" / "step" / "memory.max").write_text("max\n")
    (TMP / "cg2_self").write_text("0::/job/step\n")
    v1 = TMP / "cg1"
    (v1 / "memory" / "slurm" / "uid_1" / "job_2").mkdir(parents=True)
    (v1 / "memory" / "slurm" / "uid_1" / "job_2" / "memory.limit_in_bytes").write_text(str(45 << 30))
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(2 ** 63 - 4096))
    (TMP / "cg1_self").write_text("12:pids:/slurm/uid_1/job_2\n5:memory:/slurm/uid_1/job_2\n")
    check("cgroup memory limits: v2 (the job's memory.max above the step's 'max'), v1 (Slurm's job cgroup)",
          T._cgroup_memory_limit(str(TMP / "cg2_self"), str(root)) == 1 << 30
          and T._cgroup_memory_limit(str(TMP / "cg1_self"), str(v1)) == 45 << 30
          and T._cgroup_memory_limit(str(TMP / "none"), str(root)) is None)
    with patched(T, "_cgroup_memory_limit", lambda *a, **k: None):
        with env(SLURM_MEM_PER_NODE="46080"):
            a = T.job_memory_limit()
        with env(SLURM_MEM_PER_CPU="4000", SLURM_CPUS_ON_NODE="5"):
            b = T.job_memory_limit()
        c = T.job_memory_limit()
    check("job memory: --mem, --mem-per-cpu x CPUs, unknown", a == (46080 << 20, "SLURM_MEM_PER_NODE")
          and b[0] == 20000 << 20 and c == (None, None), (a, b, c))

    rows = train_rows
    want = 640 * 640 * 3 * len(rows) * 1.5
    est = T.ram_cache_estimate(rows, 640)
    ram = recipe(cache="ram", imgsz=640)
    out = {}
    for name, limit in (("small", (50 << 20, "test")), ("big", (45 << 30, "test")), ("unknown", (None, None))):
        rec = {"warnings": []}
        with patched(T, "job_memory_limit", lambda limit=limit: limit):
            out[name] = (T.choose_cache(ram, rows, rec), rec)
    check("the RAM-cache estimate is Ultralytics' (h*w*3 at imgsz, +50%%): %.1f MB" % (est / 1e6),
          abs(est - want) <= 2, (est, want))
    check("cache 'ram' past 60% of a 50 MB job falls back to no cache, with a warning and the numbers",
          out["small"][0] is False and out["small"][1]["cache"]["used"] is False
          and out["small"][1]["cache"]["requested"] == "ram" and out["small"][1]["cache"]["limit_bytes"] == 50 << 20
          and len(out["small"][1]["warnings"]) == 1, out["small"])
    check("... and stays 'ram' in a 45 GB job, or when the limit is unknown",
          out["big"][0] == "ram" and not out["big"][1]["warnings"] and out["unknown"][0] == "ram")
    rec = {"warnings": []}
    check("cache false stays false, without an estimate", T.choose_cache(RECIPE, rows, rec) is False
          and rec["cache"] == {"requested": False, "used": False})

    cwd, wd = TMP / "amp_cwd", TMP / "amp_wd"
    cwd.mkdir()
    wd.mkdir()
    name, path, _ = T.amp_check_weights(cwd, wd)
    check("the installed Ultralytics' check_amp names one checkpoint (%s)" % name,
          isinstance(name, str) and name.endswith(".pt") and path is None)

    def refusal():
        try:
            T.require_amp_weights(cwd, wd)
        except T.RunError as e:
            return e
        return None

    e1 = refusal()
    (cwd / name).write_bytes(os.urandom(200000))
    e2 = refusal()
    (cwd / name).unlink()
    with zipfile.ZipFile(wd / name, "w") as z:
        z.writestr("archive/data.pkl", os.urandom(200000))
    got_wd = T.require_amp_weights(cwd, wd)
    shutil.move(str(wd / name), str(cwd / name))
    got_cwd = T.require_amp_weights(cwd, wd)
    check("AMP-check checkpoint: absent -> refused at stage device (it would be downloaded); a partial "
          "file -> refused as incomplete; a whole one in weights_dir or the working dir -> recorded",
          e1 is not None and e1.stage == "device" and "downloaded" in str(e1)
          and e2 is not None and "incomplete" in str(e2)
          and got_wd["path"] == str((wd / name).resolve()) and got_cwd["path"] == str((cwd / name).resolve())
          and got_cwd["sha256"] == sha(cwd / name), (e1, e2, got_wd, got_cwd))

    lock, n = threading.Lock(), [0]

    def slow(p):
        with lock:
            n[0] += 1
            k = n[0]
        if k == 3:
            raise KeyboardInterrupt("simulated SIGTERM while hashing")
        time.sleep(0.05)
        return "x", 1

    t0 = time.time()
    raised = False
    with patched(T, "_hash_image", slow):
        try:
            T.hash_images(["p%d" % i for i in range(400)])
        except KeyboardInterrupt:
            raised = True
    el = time.time() - t0
    check("an exception while hashing cancels the queued hashes instead of awaiting them "
          "(%.2f s, %d of 400 hashed)" % (el, n[0]), raised and el < 1.0 and n[0] < 100)

    d = TMP / "cache_chk"
    d.mkdir()
    np.save(str(d / "labels.cache"), {"results": (24, 0, 0, 1, 25), "labels": [{}] * 24,
                                      "msgs": ["x: ignoring corrupt image/label"]}, allow_pickle=True)
    (d / "labels.cache.npy").rename(d / "labels.cache")
    try:
        T.check_dataset_cache(d, 25)
        e = None
    except T.RunError as x:
        e = x
    check("labels.cache counting 24 of 25 images (1 corrupt) fails the run at stage train",
          e is not None and e.stage == "train" and "24 of the manifest's 25" in str(e), e)


def test_refusals(train_manifest):
    print("refusals before any training")
    base = dict(init="yolo11n.pt", train_manifest=str(train_manifest))
    cases = [
        ("unknown spec key", spec("r_key", "base", note="x", recipe=recipe(), **base), "unknown key"),
        ("unknown recipe key", spec("r_rkey", "base", recipe=recipe(lr=0.1), **base), "unknown recipe key"),
        ("missing recipe key", spec("r_miss", "base", recipe={k: v for k, v in RECIPE.items() if k != "lrf"},
                                    **base), "missing ['lrf']"),
        ("'test' in a cand spec", spec("r_test", "cand", exams=("dev", "test"), recipe=recipe(), **base),
         "sealed test"),
        ("'test' in a soup spec", spec("r_test2", "soup", exams=("test",), soup_of=["a.pt", "b.pt"]),
         "sealed test"),
        ("optimizer 'auto'", spec("r_auto", "base", recipe=recipe(optimizer="auto"), **base), "auto"),
        ("trainer lora without recipe.lora", spec("r_lora", "cand", recipe=recipe(trainer="lora"), **base),
         "iff recipe.lora"),
        ("freeze set on a full trainer", spec("r_frz", "cand", recipe=recipe(freeze=10), **base),
         "iff recipe.freeze"),
        ("soup with a train_manifest", spec("r_soup", "soup", soup_of=["a.pt", "b.pt"],
                                            train_manifest=str(train_manifest)), "takes no"),
        ("final without init", spec("r_final", "final"), "needs ['init']"),
    ]
    p = spec("r_out", "base", recipe=recipe(), **base)
    s = json.loads(p.read_text())
    s["out_dir"] = str(TMP / "elsewhere")
    p.write_text(json.dumps(s))
    cases.append(("out_dir not the spec's dir", p, "not the spec's own directory"))
    for name, path, needle in cases:
        rc = run(path)
        rj = run_json(path)
        check("%s is refused (exit 1, failed run.json at stage spec)" % name,
              rc == 1 and rj.get("status") == "failed" and rj.get("stage") == "spec"
              and needle in (rj.get("error") or "") and not (path.parent / "train").exists(),
              (rc, rj.get("stage"), err(rj, 300)))

    bad_lab = TMP / "src" / "bad_labels" / "tr_000.txt"
    bad_lab.parent.mkdir(parents=True)
    bad_lab.write_text("13 0.5 0.5 0.2 0.2\n")
    rows = C.read_manifest(train_manifest)
    rows[0] = dict(rows[0], label=str(bad_lab), label_sha256=sha(bad_lab))
    p = spec("r_label", "base", init="yolo11n.pt", train_manifest=str(write_rows_manifest("bad_label", rows)),
             recipe=recipe())
    rc, rj = run(p), run_json(p)
    check("a label with class 13 (Ultralytics would drop the image) is refused at stage manifest",
          rc == 1 and rj.get("stage") == "manifest" and "outside 0..12" in (rj.get("error") or ""),
          (rc, rj.get("stage"), err(rj, 300)))
    rows = C.read_manifest(train_manifest)
    rows[1] = dict(rows[1], sha256="0" * 64)
    p = spec("r_sha", "base", init="yolo11n.pt", train_manifest=str(write_rows_manifest("bad_sha", rows)),
             recipe=recipe())
    rc, rj = run(p), run_json(p)
    check("an image that differs from its manifest sha256 is refused at stage manifest",
          rc == 1 and rj.get("stage") == "manifest" and "sha256" in (rj.get("error") or ""),
          (rc, rj.get("stage")))

    with env(**{S.TEST_ENV: None}):
        p = spec("r_env", "base", recipe=recipe(), **base)
        rc, rj = run(p), run_json(p)
    check("a testing experiment without %s is refused" % S.TEST_ENV,
          rc == 1 and rj.get("stage") == "testing" and S.TEST_ENV in (rj.get("error") or ""),
          (rc, rj.get("stage"), err(rj, 300)))

    p = spec("r_prod", "base", exp=EXP_PROD, recipe=recipe(), **base)
    rc, rj = run(p), run_json(p)
    check("a production run with a recipe other than the protocol's is refused at stage recipe, naming "
          "the departures", rc == 1 and rj.get("stage") == "recipe" and rj.get("testing") is False
          and rj.get("protocol_recipe") is False and "imgsz 64 (protocol 640)" in (rj.get("error") or ""),
          (rc, rj.get("stage"), err(rj, 300)))
    import torch
    if not torch.cuda.is_available():
        cold, _ = protocol_recipes()
        p = spec("r_prod2", "base", exp=EXP_PROD, recipe=cold, **base)
        rc, rj = run(p), run_json(p)
        check("a production run with the protocol's recipe on a machine without CUDA is refused at "
              "stage device, testing=false", rc == 1 and rj.get("stage") == "device"
              and rj.get("testing") is False and rj.get("protocol_recipe") is True,
              (rc, rj.get("stage"), err(rj, 300)))
    else:
        print("  skip production-without-CUDA refusal (this machine has CUDA)")


def test_guard(train_rows, dev_rows):
    print("never-train guard")
    leak = [dict(r) for r in train_rows]
    d0 = dev_rows[0]
    leak.append(dict(d0, key="leak_exact", source="leak"))
    copy = TMP / "src" / "leak" / "reencoded.jpg"
    copy.parent.mkdir(parents=True)
    Image.open(d0["image"]).save(copy, quality=60)
    leak.append(dict(d0, key="leak_reencoded", image=str(copy), sha256=sha(copy), source="leak"))
    guard = C.NeverTrainGuard.load()
    hits, _ = guard.check([copy])
    check("fixture: the re-encoded dev photo is within %d bits of it" % C.HOLDOUT_NEAR_DUP_BITS, len(hits) == 1)
    p = spec("leak", "cand", init="yolo11n.pt", train_manifest=str(write_rows_manifest("leak", leak)),
             recipe=recipe())
    rc, rj = run(p), run_json(p)
    g = rj.get("guard") or {}
    check("a manifest holding a dev image and a near copy of one is refused: exit 1, failed run.json",
          rc == 1 and rj.get("status") == "failed" and rj.get("stage") == "guard", (rc, rj.get("stage")))
    check("run.json records the guard: %s checked, %s hits" % (g.get("checked"), g.get("hits")),
          g.get("checked") == N_TRAIN + 2 and g.get("hits") == 2 and g.get("unhashable") == 0
          and {h[1] for h in g.get("first_hits", [])} == {"dev"}, g)
    check("nothing was materialised or trained",
          not (p.parent / "data").exists() and not (p.parent / "train").exists()
          and not (p.parent / "weights").exists() and not sizeable(p.parent))
    check("every documented run.json field is present on a failure too",
          all(k in rj for k in DOC_FIELDS) and rj.get("attempt") == 1 and "Traceback" in (rj.get("error") or ""),
          [k for k in DOC_FIELDS if k not in rj])
    rc = run(p)
    rj = run_json(p)
    att = json.loads((p.parent / T.ATTEMPT_JSON).read_text())
    check("a failed run may be re-run: attempt 2, the first attempt kept in attempt.json's history",
          rc == 1 and rj.get("attempt") == 2 and [h["attempt"] for h in att["history"]] == [1],
          (rj.get("attempt"), att))


def test_full(train_manifest):
    print("full run (base, cold from REPO/yolo11n.pt)")
    from ultralytics import YOLO
    p = spec("full_s0", "base", init="yolo11n.pt", train_manifest=str(train_manifest), recipe=recipe())
    with recording_train() as calls:
        rc = run(p)
    rj = run_json(p)
    out = p.parent
    final = out / "weights" / "final.pt"
    check("exit 0, status done, attempt 1, marked testing",
          rc == 0 and rj.get("status") == "done" and rj.get("attempt") == 1 and rj.get("testing") is True
          and rj.get("error") is None, (rc, rj.get("stage"), err(rj, 500)))
    check("every documented run.json field is present",
          all(k in rj for k in DOC_FIELDS), [k for k in DOC_FIELDS if k not in rj])
    kw = calls[0] if len(calls) == 1 else {}
    check("recipe lr0 / optimizer / seed / epochs reach Model.train",
          (kw.get("lr0"), kw.get("optimizer"), kw.get("seed"), kw.get("epochs"))
          == (RECIPE["lr0"], RECIPE["optimizer"], RECIPE["seed"], RECIPE["epochs"]), kw)
    check("... with val=False, plots=False, project=<run dir>, name='train', exist_ok, no freeze, "
          "and every other recipe setting",
          kw.get("val") is False and kw.get("plots") is False
          and pathlib.Path(kw.get("project", "")).resolve() == out.resolve()
          and kw.get("name") == "train" and kw.get("exist_ok") is True and kw.get("freeze") is None
          and kw.get("cache") is False
          and all(kw.get(k) == RECIPE[k] for k in ("lrf", "momentum", "weight_decay", "warmup_epochs",
                                                    "warmup_bias_lr", "cos_lr", "imgsz", "batch",
                                                    "workers", "close_mosaic", "deterministic")), kw)
    check("final.pt is the trainer's last.pt, and run.json has its sha256",
          final.is_file() and sha(final) == kw.get("_last_sha256") == rj.get("weights_sha256"))
    score = json.loads((out / "scores" / "dev.json").read_text()) if (out / "scores" / "dev.json").exists() else {}
    check("scores/dev.json: those weights on dev, a TEST score (production=false)",
          score.get("weights_sha256") == rj.get("weights_sha256") and score.get("exam") == "dev"
          and score.get("production") is False and score.get("scorer_sha256", "").startswith(S.TEST_PREFIX)
          and rj["scores"]["dev"]["sha256"] == sha(out / "scores" / "dev.json"), score.get("deviations"))
    st = score.get("settings") or {}
    check("'testing': true keeps the protocol's imgsz 640 / batch 32 and the lock check, and scores on "
          "the CPU, the one deviation", (st.get("imgsz"), st.get("batch"), score.get("lock_checked"),
                                          score.get("device")) == (640, 32, True, "cpu")
          and len(score.get("deviations") or []) >= 1, (st, score.get("deviations")))
    check("run.json holds no metric (the driver reads score.json only)",
          not any(k in json.dumps(rj["scores"]) for k in ("map50", "per_class")))
    labels = sum(len(C.read_yolo(r["label"])) for r in C.read_manifest(train_manifest))
    check("counts: %s images, %s boxes; guard checked all, 0 hits; no duplicate images"
          % (rj.get("n_train_images"), rj.get("n_train_boxes")),
          rj.get("n_train_images") == N_TRAIN and rj.get("n_train_boxes") == labels
          and rj["guard"]["checked"] == N_TRAIN and rj["guard"]["hits"] == 0
          and rj.get("dataset_check", {}).get("images") == N_TRAIN
          and rj["duplicate_images"]["extra_rows"] == 0 and rj["dhash0_collisions"]["extra_rows"] == 0)
    check("a testing run records how its recipe departs from the protocol's, and the weights' epoch",
          rj.get("protocol_recipe") is False and any("imgsz" in d for d in rj.get("recipe_deviations", []))
          and rj.get("weights_epoch") == 0 and rj.get("cache") == {"requested": False, "used": False})
    check("full: every parameter but the fixed DFL conv trains (%s / %s)"
          % (rj.get("trainable_params"), rj.get("total_params")),
          rj.get("total_params") and rj["total_params"] - rj["trainable_params"] == 16)
    check("cleanup: no data dir, no last.pt / best.pt, no owner file; train logs kept; the lock recorded",
          not sizeable(out) and (out / "train" / "args.yaml").is_file()
          and (out / "train" / "results.csv").is_file() and rj["lock"]["flock"] == "held"
          and rj["lock"]["took_over"] == [], sizeable(out))
    m = YOLO(str(final))
    check("final.pt loads with plain YOLO() in the INC class space", list(m.names.values()) == C.CLASS_NAMES)

    before = (out / T.RUN_JSON).read_bytes()
    trash = out / (T.TRASH_PREFIX + "data-1-left")
    (trash / "images").mkdir(parents=True)
    (trash / "images" / "x.jpg").write_bytes(b"x")
    t0 = time.time()
    rc = run(p)
    check("re-running a done run is a no-op (exit 0, run.json untouched, %.1fs); a leftover .trash-* "
          "dir is removed" % (time.time() - t0),
          rc == 0 and (out / T.RUN_JSON).read_bytes() == before and final.is_file() and not trash.exists())
    s = json.loads(p.read_text())
    p.write_text(json.dumps(dict(reversed(list(s.items()))), indent=4))
    rc = run(p)
    check("... also when the spec file is rewritten with other key order and indentation",
          rc == 0 and (out / T.RUN_JSON).read_bytes() == before)
    import fcntl
    with open(out / T.LOCK_NAME, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = run(p, "--force")
    check("while another executor holds the run dir, a second one exits 3 and touches nothing",
          rc == T.EXIT_BUSY and (out / T.RUN_JSON).read_bytes() == before and final.is_file()
          and not (out / T.OWNER_NAME).exists())
    return final


def test_save_dir(train_manifest):
    print("the trainer's save_dir, not a hard-coded path")
    decoy = cold_checkpoint(TMP / "decoy.pt", seed=7)
    p = spec("full_s1", "base", init="yolo11n.pt", train_manifest=str(train_manifest), recipe=recipe(seed=1))
    with recording_train(redirect="train_moved", decoy=decoy) as calls:
        rc = run(p)
    rj = run_json(p)
    final = p.parent / "weights" / "final.pt"
    check("a trainer that saved to train_moved/ (decoy last.pt planted in train/): final.pt is the "
          "trainer's weights",
          rc == 0 and calls and sha(final) == calls[0]["_last_sha256"] != sha(decoy)
          and rj.get("train_dir") == str((p.parent / "train_moved").resolve()),
          (rc, rj.get("train_dir"), err(rj)))
    check("its weights were cleaned up too", not sizeable(p.parent), sizeable(p.parent))
    return final


def test_freeze(train_manifest, init):
    print("freeze run (cand from the full run)")
    p = spec("freeze_s0", "cand", init=str(init), train_manifest=str(train_manifest),
             recipe=recipe(trainer="freeze", freeze=11, lr0=0.002))
    with recording_train() as calls:
        rc = run(p)
    rj = run_json(p)
    kw = calls[0] if calls else {}
    check("exit 0, done, final.pt + scores/dev.json", rc == 0 and rj.get("status") == "done"
          and (p.parent / "weights" / "final.pt").is_file() and (p.parent / "scores" / "dev.json").is_file(),
          (rc, rj.get("stage"), err(rj, 500)))
    check("freeze=11 and lr0 0.002 reach Model.train", kw.get("freeze") == 11 and kw.get("lr0") == 0.002, kw)
    check("layers 0-10 frozen: trainable %s of %s" % (rj.get("trainable_params"), rj.get("total_params")),
          0 < (rj.get("trainable_params") or 0) < 0.8 * (rj.get("total_params") or 0))
    check("init recorded with its sha256", rj.get("init") == str(init.resolve()) and rj.get("init_sha256") == sha(init))
    check("cleanup", not sizeable(p.parent), sizeable(p.parent))


def test_lora(train_manifest, init):
    print("lora run (cand from the full run)")
    from ultralytics import YOLO
    p = spec("lora_s0", "cand", init=str(init), train_manifest=str(train_manifest),
             recipe=recipe(trainer="lora", lora={"rank": 4, "alpha": 8}, lr0=0.01, seed=2))
    with recording_lora() as calls:
        rc = run(p)
    rj = run_json(p)
    kw = calls[0] if calls else {}
    out = p.parent
    check("exit 0, done, final.pt + scores/dev.json", rc == 0 and rj.get("status") == "done"
          and (out / "weights" / "final.pt").is_file() and (out / "scores" / "dev.json").is_file(),
          (rc, rj.get("stage"), err(rj, 500)))
    check("rank / alpha / lr0 / optimizer / seed / epochs reach train_lora",
          (kw.get("rank"), kw.get("alpha"), kw.get("lr0"), kw.get("optimizer"), kw.get("seed"), kw.get("epochs"))
          == (4, 8, 0.01, "SGD", 2, 1) and kw.get("val") is False
          and pathlib.Path(kw.get("project", "")).resolve() == out.resolve()
          and kw.get("name") == "train", kw)
    lj = json.loads((out / "train" / "lora.json").read_text()) if (out / "train" / "lora.json").exists() else {}
    check("final.pt is train_lora's merged checkpoint (sha in lora.json's epoch; adapters recorded)",
          lj.get("rank") == 4 and (rj.get("lora") or {}).get("n_wrapped") == lj.get("n_wrapped") > 0
          and rj.get("weights_source", "").endswith("last_merged.pt") and rj.get("weights_epoch") == 0,
          rj.get("lora"))
    check("lora: trainable %s of %s parameters" % (rj.get("trainable_params"), rj.get("total_params")),
          0 < (rj.get("trainable_params") or 0) < (rj.get("total_params") or 0)
          and rj.get("trainable_params") == lj.get("trainable_params"))
    m = YOLO(str(out / "weights" / "final.pt"))
    check("final.pt loads with plain YOLO(), no adapters left",
          list(m.names.values()) == C.CLASS_NAMES
          and not [x for x in m.model.modules() if type(x).__name__ == "ConvLoRA"])
    check("cleanup: last.pt and last_merged.pt gone, lora.json kept", not sizeable(out)
          and not (out / "train" / "weights").exists(), sizeable(out))


def test_lora_diverged(train_manifest, init):
    print("lora: a merged checkpoint from an earlier epoch (Ultralytics skipped a save)")
    from weed_optimizer_framework.tools.inc import lora as L
    p = spec("lora_div", "cand", init=str(init), train_manifest=str(train_manifest),
             recipe=recipe(trainer="lora", lora={"rank": 4, "alpha": 8}, lr0=0.01, seed=5, epochs=3))

    def fake_train_lora(init_weights, data_yaml, project, name, **kw):
        sd = pathlib.Path(project) / name
        (sd / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(init_weights, sd / "weights" / L.MERGED_NAME)
        (sd / L.LORA_JSON).write_text(json.dumps({"merged_from_epoch": 1, "skipped_saves": [2], "rank": 4,
                                                  "alpha": 8, "trainable_params": 1, "total_params": 2}))
        return str(sd / "weights" / L.MERGED_NAME)

    with patched(L, "train_lora", fake_train_lora):
        rc = run(p)
    rj = run_json(p)
    check("last_merged.pt from epoch 1 of 3: the run fails at stage train as diverged, recorded",
          rc == 1 and rj.get("stage") == "train" and "diverged" in (rj.get("error") or "")
          and rj.get("diverged") == {"weights_epoch": 1, "final_epoch": 2, "epochs": 3, "skipped_saves": [2]},
          (rc, rj.get("stage"), rj.get("diverged"), err(rj, 300)))
    check("... no final.pt, and the merged checkpoint was cleaned up",
          not (p.parent / "weights" / "final.pt").exists() and not sizeable(p.parent), sizeable(p.parent))


def test_diverged_and_cache(train_manifest):
    print("full: last.pt from an earlier epoch; cache 'ram' beyond the job's memory")
    p = spec("diverged", "base", init="yolo11n.pt", train_manifest=str(train_manifest),
             recipe=recipe(seed=7, epochs=2, cache="ram"))
    with patched(T, "_checkpoint_epoch", lambda ck: 0), patched(T, "job_memory_limit", lambda: (1000, "test")), \
            recording_train() as calls:
        rc = run(p)
    rj = run_json(p)
    kw = calls[0] if calls else {}
    check("cache 'ram' estimated past 60% of the job's memory: Model.train gets cache=False, recorded",
          kw.get("cache") is False and (rj.get("cache") or {}).get("used") is False
          and (rj.get("cache") or {}).get("requested") == "ram"
          and any("cache 'ram' needs" in w for w in rj.get("warnings", [])), (kw.get("cache"), rj.get("cache")))
    check("last.pt holding epoch 0 of 2: the run fails at stage train as diverged, recorded",
          rc == 1 and rj.get("stage") == "train" and "diverged" in (rj.get("error") or "")
          and rj.get("diverged") == {"weights_epoch": 0, "final_epoch": 1, "epochs": 2}
          and rj.get("trained") is False, (rc, rj.get("stage"), rj.get("diverged"), err(rj, 300)))
    check("... no final.pt; data and train weights cleaned up",
          not (p.parent / "weights" / "final.pt").exists() and not sizeable(p.parent), sizeable(p.parent))


def test_corrupt_image(train_rows):
    print("an image Ultralytics drops; the record is written before the slow cleanup")
    tiny = TMP / "src" / "tiny" / "tiny.png"
    tiny.parent.mkdir(parents=True)
    Image.fromarray(np.random.RandomState(9).randint(0, 255, (6, 6, 3)).astype(np.uint8)).save(tiny)
    rows = [dict(r) for r in train_rows] + [extra_row("y_tiny", tiny, [(1, 0.5, 0.5, 0.5, 0.5)])]
    hits, unhashable = C.NeverTrainGuard.load().check([tiny])
    check("fixture: the 6x6 image passes the guard", not hits and not unhashable, (hits, unhashable))
    p = spec("tiny", "base", init="yolo11n.pt", train_manifest=str(write_rows_manifest("train_tiny", rows)),
             recipe=recipe(seed=6))
    order = []
    orig = T.purge_dir

    def purge(path):
        path = pathlib.Path(path)
        order.append((path.name, (p.parent / T.RUN_JSON).is_file(), (path / "images").is_dir()))
        orig(path)

    with patched(T, "purge_dir", purge):
        rc = run(p)
    rj = run_json(p)
    check("a 6x6 image: Ultralytics loads %d of %d, and the run fails at stage train before any epoch"
          % (N_TRAIN, N_TRAIN + 1),
          rc == 1 and rj.get("stage") == "train"
          and "loaded %d of the manifest's %d" % (N_TRAIN, N_TRAIN + 1) in (rj.get("error") or "")
          and not (p.parent / "train" / "results.csv").exists(), (rc, rj.get("stage"), err(rj, 300)))
    check("data/ was renamed aside, run.json written, and only then was the renamed dir deleted",
          len(order) == 1 and order[0][0].startswith(T.TRASH_PREFIX + "data-") and order[0][1] and order[0][2]
          and not sizeable(p.parent) and str(p.parent.resolve() / "data") + "/" in rj.get("cleanup_removed", []),
          (order, sizeable(p.parent)))


def test_rerun(train_rows):
    print("re-runs of a cand: no-op, re-score, keep final.pt, retrain; odd images in the manifest")
    rows = [dict(r) for r in train_rows]
    src0 = pathlib.Path(train_rows[0]["image"])
    tail = TMP / "src" / "extra" / "tail.jpg"
    boxes = make_image(tail, 500)
    with open(tail, "ab") as fh:
        fh.write(b"\x00\x00")            # bytes after the EOI marker; PIL still decodes it
    tail_sha = sha(tail)
    rows.append(extra_row("x_tail", tail, boxes))
    dup = TMP / "src" / "extra" / "dup.jpg"
    shutil.copyfile(src0, dup)
    rows.append(dict(train_rows[0], image=str(dup), key="x_dup", source="extra"))
    png = TMP / "src" / "extra" / "png.png"
    Image.open(src0).save(png)
    rows.append(dict(train_rows[0], image=str(png), sha256=sha(png), key="x_png", source="extra"))
    hits, unhashable = C.NeverTrainGuard.load().check([tail, dup, png])
    check("fixture: the three extra images pass the guard, the PNG has the JPEG's dHash",
          not hits and not unhashable and C.dhash(png) == C.dhash(src0) and T._jpeg_without_eoi(tail),
          (hits, unhashable))
    man = write_rows_manifest("train_rr", rows)
    p = spec("rr", "cand", init="yolo11n.pt", train_manifest=str(man), recipe=recipe(seed=4))
    out = p.parent
    final = out / "weights" / "final.pt"
    seen = {}

    def look(kw):
        imgs = pathlib.Path(kw["data"]).parent / "images"
        seen["tail"] = (imgs / "x_tail.jpg").is_file() and not (imgs / "x_tail.jpg").is_symlink()
        seen["dup"] = (imgs / "x_dup.jpg").is_symlink()

    with recording_train(inspect=look):
        rc = run(p)
    rj = run_json(p)
    check("attempt 1 done", rc == 0 and rj.get("status") == "done", (rc, rj.get("stage"), err(rj)))
    check("the JPEG with bytes after its EOI was copied into data/images (others linked); its source is "
          "unchanged after Ultralytics' re-save", seen == {"tail": True, "dup": True} and sha(tail) == tail_sha
          and rj.get("materialised", {}).get("jpegs_copied_not_linked") == 1, (seen, rj.get("materialised")))
    di, dh = rj.get("duplicate_images") or {}, rj.get("dhash0_collisions") or {}
    check("run.json counts one exact duplicate (across sources) and one dHash-0 pair, with warnings",
          (di.get("extra_rows"), di.get("across_source_session"), di.get("first")) == (1, 1, [["tr_000", "x_dup"]])
          and (dh.get("extra_rows"), dh.get("first")) == (1, [["tr_000", "x_png"]])
          and sum(w.startswith(("duplicate_images", "dhash0_collisions")) for w in rj.get("warnings", [])) == 2,
          (di, dh))
    mtime, wsha = final.stat().st_mtime_ns, sha(final)

    exam_dir = C.EXAMS_DIR / "dev"
    with open(out / "scores" / "dev.json", "a") as fh:
        fh.write(" ")
    exam_dir.rename(exam_dir.with_name("dev.away"))
    try:
        rc = run(p)
    finally:
        exam_dir.with_name("dev.away").rename(exam_dir)
    rj = run_json(p)
    check("a changed score file is re-scored, not retrained: attempt 2 resumes attempt 1; with the dev "
          "exam gone it fails at stage score and keeps final.pt",
          rc == 1 and rj.get("attempt") == 2 and rj.get("resumed_from_attempt") == 1
          and rj.get("stage") == "score" and rj.get("trained") is True and final.stat().st_mtime_ns == mtime,
          (rc, rj.get("attempt"), rj.get("resumed_from_attempt"), rj.get("stage")))
    with env(**{S.TEST_ENV: None}):
        rc = run(p)
    rj = run_json(p)
    check("attempt 3 fails before scoring (stage testing): final.pt kept, run.json still names it",
          rc == 1 and rj.get("attempt") == 3 and rj.get("stage") == "testing" and final.is_file()
          and final.stat().st_mtime_ns == mtime and rj.get("trained") is True
          and rj.get("weights_sha256") == wsha and rj.get("resumed_from_attempt") == 2,
          (rc, rj.get("attempt"), rj.get("stage"), final.exists(), rj.get("cleanup_removed")))
    rc = run(p)
    rj = run_json(p)
    check("attempt 4 re-scores that final.pt: done, resumed from 3, file untouched, n_train_images carried",
          rc == 0 and rj.get("attempt") == 4 and rj.get("resumed_from_attempt") == 3
          and final.stat().st_mtime_ns == mtime and rj.get("weights_sha256") == wsha
          and rj.get("n_train_images") == N_TRAIN + 3
          and rj["scores"]["dev"]["sha256"] == sha(out / "scores" / "dev.json"),
          (rc, rj.get("attempt"), rj.get("resumed_from_attempt"), err(rj)))

    final.unlink()
    rc = run(p)
    rj = run_json(p)
    check("final.pt deleted after a done run: not a no-op; attempt 5 retrains",
          rc == 0 and rj.get("attempt") == 5 and rj.get("resumed_from_attempt") is None and final.is_file()
          and rj.get("weights_sha256") == sha(final), (rc, rj.get("attempt"), err(rj)))
    lines = man.read_text().splitlines()
    man.write_text("\n".join(reversed(lines)) + "\n")
    new_sha = sha(man)
    rc = run(p)
    rj = run_json(p)
    check("the train manifest's bytes changed behind the same spec: attempt 6 retrains, the new sha recorded",
          rc == 0 and rj.get("attempt") == 6 and rj.get("resumed_from_attempt") is None
          and rj.get("train_manifest_sha256") == new_sha, (rc, rj.get("attempt"), err(rj)))
    before = (out / T.RUN_JSON).read_bytes()
    rc = run(p)
    check("... after which the run is a no-op again", rc == 0 and (out / T.RUN_JSON).read_bytes() == before)


def test_soup(a, b):
    print("soup")
    import torch
    from ultralytics import YOLO
    # the driver writes cand[0]'s weights as a soup's init; it is accepted and not used
    p = spec("soup_ab", "soup", init=str(a), soup_of=[str(a), str(b)])
    exam_dir = C.EXAMS_DIR / "dev"
    exam_dir.rename(exam_dir.with_name("dev.away"))
    try:
        rc = run(p)
    finally:
        exam_dir.with_name("dev.away").rename(exam_dir)
    rj = run_json(p)
    final = p.parent / "weights" / "final.pt"
    check("with the dev exam missing, the soup fails at stage score (scorer refused) but keeps final.pt",
          rc == 1 and rj.get("stage") == "score" and rj.get("trained") is True and final.is_file()
          and "refused" in (rj.get("error") or ""), (rc, rj.get("stage"), err(rj, 300)))
    mtime, first_sha = final.stat().st_mtime_ns, sha(final)
    rc = run(p)
    rj = run_json(p)
    check("the next attempt re-scores that final.pt: done, attempt 2, resumed from 1, file untouched",
          rc == 0 and rj.get("status") == "done" and rj.get("attempt") == 2
          and rj.get("resumed_from_attempt") == 1 and final.stat().st_mtime_ns == mtime
          and rj.get("weights_sha256") == first_sha
          and rj.get("soup_of") == [str(a.resolve()), str(b.resolve())],
          (rc, rj.get("stage"), err(rj, 300)))
    check("scores/dev.json written", (p.parent / "scores" / "dev.json").is_file())

    def net(path):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return ck["ema"] if ck.get("ema") is not None else ck["model"]

    sa, sb, ss = net(a).state_dict(), net(b).state_dict(), net(final).state_dict()
    fl = [k for k in sa if sa[k].is_floating_point()]
    exact = all(torch.equal(ss[k], (sa[k].float() + sb[k].float()) / 2) for k in fl)
    others = all(torch.equal(ss[k], sa[k]) for k in sa if k not in fl)
    moved = sum(not torch.equal(sa[k], sb[k]) for k in fl)
    check("soup = elementwise fp32 mean of the two (%d float tensors, %d differ between members); "
          "non-float buffers from the first" % (len(fl), moved),
          exact and others and moved > 0 and ss[fl[0]].dtype == torch.float32 and list(ss) == list(sa))
    m = YOLO(str(final))
    check("the soup loads with plain YOLO() in the INC class space", list(m.names.values()) == C.CLASS_NAMES)

    other = cold_checkpoint(TMP / "nc12.pt", nc=12)
    for rid, name, members, needle in (("soup_bad_arch", "members of different architectures", [a, other],
                                        "different"),
                                       ("soup_bad_dup", "the same weights twice", [a, a], "same weights twice")):
        q = spec(rid, "soup", soup_of=[str(x) for x in members])
        rc, rj = run(q), run_json(q)
        check("a soup of %s is refused" % name,
              rc == 1 and rj.get("stage") == "soup" and needle in (rj.get("error") or "")
              and not (q.parent / "weights" / "final.pt").exists(),
              (rc, rj.get("stage"), err(rj, 300)))
    return final


def test_final(init):
    print("final (scoring only)")
    p = spec("final_soup", "final", exams=("dev", "test"), init=str(init))
    rc = run(p)
    rj = run_json(p)
    final = p.parent / "weights" / "final.pt"
    check("exit 0, done; final.pt is a symlink to init; scored on dev and test",
          rc == 0 and rj.get("status") == "done" and final.is_symlink()
          and final.resolve() == init.resolve() and rj.get("weights_sha256") == sha(init)
          and sorted(rj.get("scores", {})) == ["dev", "test"]
          and all((p.parent / "scores" / ("%s.json" % e)).is_file() for e in ("dev", "test")),
          (rc, rj.get("stage"), err(rj)))
    check("no training: no train dir, no data, no parameter counts",
          not (p.parent / "train").exists() and not (p.parent / "data").exists()
          and rj.get("trainable_params") is None and rj.get("n_train_images") is None)
    rc = run(p, "--force")
    rj = run_json(p)
    check("--force re-runs a done run: attempt 2, done, init untouched",
          rc == 0 and rj.get("attempt") == 2 and rj.get("forced") is True and init.is_file()
          and final.is_symlink(), (rc, rj.get("attempt")))
    q = spec("final_dict", "final", exp=EXP_DICT, init=str(init))
    rc = run(q)
    rj = run_json(q)
    score = json.loads((q.parent / "scores" / "dev.json").read_text()) if rj.get("status") == "done" else {}
    check("an exp.json 'testing' object's imgsz / batch / device reach the scorer",
          rc == 0 and (score.get("settings", {}).get("imgsz"), score.get("settings", {}).get("batch"),
                       score.get("device")) == (IMGSZ, BATCH, "cpu")
          and rj.get("testing_settings") == TESTING_DICT, (rc, rj.get("stage"), score.get("settings")))
    return p


def test_lock(init):
    print("the run-dir lock: owner file, staleness, flock errors, a lost lock")

    def fspec(rid):
        return spec(rid, "final", exp=EXP_DICT, init=str(init))

    def plant(path, **info):
        o = path.parent / T.OWNER_NAME
        o.write_text(json.dumps(dict({"host": "other-node", "pid": 12345, "token": "t0"}, **info)))
        return o

    def enolck(fh, op):
        raise OSError(errno.ENOLCK, "No locks available")

    def eio(fh, op):
        raise OSError(errno.EIO, "Input/output error")

    with patched(T, "LOCK_WAIT_SECONDS", 0.6), patched(T, "LOCK_POLL_SECONDS", 0.1):
        p = fspec("lk_node")
        o = plant(p)
        t0 = time.time()
        rc = run(p)
        el = time.time() - t0
        check("a fresh owner file of another node: waits LOCK_WAIT_SECONDS (%.1f s), then exits 3 touching "
              "nothing" % el, rc == 3 and not (p.parent / T.RUN_JSON).exists()
              and not (p.parent / T.ATTEMPT_JSON).exists() and json.loads(o.read_text())["token"] == "t0"
              and el >= 0.5, (rc, el))
        old = time.time() - T.LOCK_STALE_SECONDS - 60
        os.utime(o, (old, old))
        rc, rj = run(p), run_json(p)
        took = (rj.get("lock") or {}).get("took_over") or [{}]
        check("once its heartbeat is older than LOCK_STALE_SECONDS it is taken over (recorded) and the run "
              "is done; no owner file is left",
              rc == 0 and rj.get("status") == "done" and len(took) == 1 and "heartbeat" in took[0].get("reason", "")
              and (took[0].get("owner") or {}).get("token") == "t0" and not sizeable(p.parent)
              and not list(p.parent.glob(T.OWNER_NAME + "*")), (rc, rj.get("lock"), err(rj)))

        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        q = fspec("lk_dead")
        plant(q, host=socket.gethostname(), pid=dead.pid)
        rc, rj = run(q), run_json(q)
        took = (rj.get("lock") or {}).get("took_over") or [{}]
        check("a fresh owner file of an exited process on this host is taken over at once",
              rc == 0 and "has exited" in took[0].get("reason", ""), (rc, rj.get("lock")))

        r = fspec("lk_enolck")
        with patched(T, "_flock", enolck):
            rc, rj = run(r), run_json(r)
        check("flock unsupported on the mount (ENOLCK): the owner file alone guards the dir; done, recorded",
              rc == 0 and rj.get("status") == "done" and "ENOLCK" in ((rj.get("lock") or {}).get("flock") or ""),
              (rc, rj.get("lock"), err(rj)))
        o = plant(r)
        before = (r.parent / T.RUN_JSON).read_bytes()
        with patched(T, "_flock", enolck):
            rc = run(r, "--force")
        check("... and there a live owner file still makes a second executor exit 3",
              rc == 3 and (r.parent / T.RUN_JSON).read_bytes() == before)
        o.unlink()

        s = fspec("lk_eio")
        with patched(T, "_flock", eio):
            rc, rj = run(s), run_json(s)
        check("any other flock error: exit 1 and a failed run.json at stage lock naming it (all documented "
              "fields)", rc == 1 and rj.get("status") == "failed" and rj.get("stage") == "lock"
              and "Input/output error" in (rj.get("error") or "") and all(k in rj for k in DOC_FIELDS),
              (rc, rj.get("stage"), err(rj)))
        with patched(T, "_flock", eio):
            rc = run(r)
        check("... but a done run.json is left as it is", rc == 1 and (r.parent / T.RUN_JSON).read_bytes() == before)

        u = fspec("lk_lost")
        orig = T.score_exams

        def steal(*a, **k):
            own = u.parent / T.OWNER_NAME
            own.unlink()                  # judged stale by another executor, which took the dir
            own.write_text(json.dumps({"host": "other-node", "pid": 1, "token": "thief"}))
            time.sleep(0.3)
            return orig(*a, **k)

        with patched(T, "LOCK_HEARTBEAT_SECONDS", 0.05), patched(T, "score_exams", steal):
            rc = run(u)
        own = u.parent / T.OWNER_NAME
        check("an attempt whose owner file was taken over exits 3 and writes / removes nothing (the new owner "
              "keeps the dir and its owner file)",
              rc == 3 and not (u.parent / T.RUN_JSON).exists() and json.loads(own.read_text())["token"] == "thief"
              and (u.parent / "weights" / "final.pt").is_symlink(), rc)
        own.unlink()


def test_drift(init):
    print("code drift from the git-tracked nested copy")
    nested = C.REPO / "weed_llm_benchmark" / "weed_optimizer_framework"
    for m in T.CODE_MODULES:
        (nested / m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(T.package_dir() / m, nested / m)
    try:
        p = spec("drift_same", "final", exp=EXP_DICT, init=str(init))
        rc, rj = run(p), run_json(p)
        check("a nested copy identical to the running one: compared, no drift, done",
              rc == 0 and (rj.get("code") or {}).get("nested_dir") == str(nested) and rj.get("code_drift") == [],
              (rc, rj.get("code_drift"), err(rj)))
        with open(nested / "tools/inc/scorer.py", "a") as fh:
            fh.write("\n# edited\n")
        q = spec("drift_edit", "final", exp=EXP_DICT, init=str(init))
        rc, rj = run(q), run_json(q)
        check("one module edited in the nested copy: the run fails at stage code naming it",
              rc == 1 and rj.get("stage") == "code" and rj.get("code_drift") == ["tools/inc/scorer.py"],
              (rc, rj.get("stage"), rj.get("code_drift")))
        with env(**{T.ALLOW_DRIFT_ENV: "1"}):
            rc, rj = run(q), run_json(q)
        check("%s=1 runs it anyway, with a warning naming the module" % T.ALLOW_DRIFT_ENV,
              rc == 0 and rj.get("code_drift") == ["tools/inc/scorer.py"]
              and any("tools/inc/scorer.py" in w for w in rj.get("warnings", [])), (rc, err(rj)))
    finally:
        shutil.rmtree(C.REPO / "weed_llm_benchmark", ignore_errors=True)


def test_production(init, train_manifest):
    print("production runs with CUDA simulated (a stub scorer)")
    import torch
    stub = TMP / "stub_scorer.py"
    stub.write_text(textwrap.dedent('''
        import hashlib, json, os, sys
        a = sys.argv[1:]
        get = lambda k: a[a.index(k) + 1]
        prod = os.environ.get("STUB_PRODUCTION") == "1"
        with open(get("--weights"), "rb") as fh:
            w = fh.read()
        out = {"exam": get("--exam"), "weights_sha256": hashlib.sha256(w).hexdigest(), "production": prod,
               "scorer_sha256": "stub", "deviations": [] if prod else ["stub"], "argv": a,
               "saw_test_env": os.environ.get("INC_SCORER_TESTING")}
        with open(get("--out"), "w") as fh:
            json.dump(out, fh)
    '''))
    orig_cmd, orig_amp = T.scorer_command, T.amp_check_weights

    def cmd(*a):
        c = orig_cmd(*a)
        return [c[0], str(stub)] + c[4:]

    with patched(torch.cuda, "is_available", lambda: True), patched(T, "scorer_command", cmd):
        with env(STUB_PRODUCTION="1"):
            p = spec("prod_final", "final", exp=EXP_PROD, init=str(init))
            rc, rj = run(p), run_json(p)
            s = json.loads((p.parent / "scores" / "dev.json").read_text()) if rc == 0 else {}
            check("a production final run: done, testing=false, device 0; the scorer ran without %s and "
                  "without any test setting" % S.TEST_ENV,
                  rc == 0 and rj.get("testing") is False and rj.get("device") == "0"
                  and rj["scores"]["dev"]["production"] is True and s.get("saw_test_env") is None
                  and not {"--device", "--imgsz", "--batch", "--no-lock-check-for-tests"} & set(s.get("argv", [])),
                  (rc, rj.get("stage"), err(rj), s))
            q = spec("test_final_prod", "final", exp=EXP, init=str(init))
            rc, rj = run(q), run_json(q)
            check("a testing run whose score comes back production=true fails at stage score",
                  rc == 1 and rj.get("stage") == "score" and "is a production score" in (rj.get("error") or ""),
                  (rc, rj.get("stage"), err(rj)))
        with env(STUB_PRODUCTION="0"):
            r = spec("prod_final_test", "final", exp=EXP_PROD, init=str(init))
            rc, rj = run(r), run_json(r)
            check("a production run whose score is not production=true fails at stage score",
                  rc == 1 and rj.get("stage") == "score" and "not a production score" in (rj.get("error") or ""),
                  (rc, rj.get("stage"), err(rj)))
        cold, _ = protocol_recipes()
        with patched(T, "amp_check_weights", lambda cwd=None, weights_dir=None: orig_amp(TMP / "no_cwd", TMP / "no_wd")):
            u = spec("prod_amp", "base", exp=EXP_PROD, init="yolo11n.pt", train_manifest=str(train_manifest),
                     recipe=cold)
            rc, rj = run(u), run_json(u)
        check("a production training run on CUDA without the AMP-check checkpoint in place is refused at "
              "stage device, before anything is materialised",
              rc == 1 and rj.get("stage") == "device" and "downloaded" in (rj.get("error") or "")
              and not (u.parent / "data").exists() and not sizeable(u.parent), (rc, rj.get("stage"), err(rj)))


def test_sigterm(train_manifest):
    print("SIGTERM during training")
    p = spec("sigterm", "base", init="yolo11n.pt", train_manifest=str(train_manifest),
             recipe=recipe(epochs=50, seed=3))
    proc = subprocess.Popen([sys.executable, "-m", "weed_optimizer_framework.tools.inc.train", "--spec", str(p)],
                            cwd=str(ROOT), env=dict(os.environ), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    while time.time() < deadline and not (p.parent / "train" / "args.yaml").exists():
        time.sleep(0.2)
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    rc = proc.wait(timeout=300)
    rj = run_json(p)
    check("killed while training: exit 1 and a failed run.json at stage train naming the signal",
          rc == 1 and rj.get("status") == "failed" and rj.get("stage") == "train"
          and "signal %d" % signal.SIGTERM in (rj.get("error") or ""),
          (rc, rj.get("stage"), err(rj, 300)))
    check("... and its data and weights were cleaned up, the owner file released",
          not sizeable(p.parent), sizeable(p.parent))


def test_job_script(done_spec, bad_spec):
    print("run_inc_job.sh")
    src = ROOT / "run_inc_job.sh"
    r = subprocess.run(["bash", "-n", str(src)], capture_output=True, text=True)
    check("bash -n passes", r.returncode == 0, r.stderr)
    text = src.read_text()
    check("sbatch header: GPU-shared, v100-32:1, 5 CPUs, 45G, 3h, log to results/framework/inc/logs/%x_%A_%a.out",
          all(s in text for s in ("--partition=GPU-shared", "--gres=gpu:v100-32:1", "--cpus-per-task=5",
                                  "--mem=45G", "--time=03:00:00",
                                  "weed_llm_benchmark/results/framework/inc/logs/%x_%A_%a.out")))
    stub = TMP / "job"
    (stub / "bin").mkdir(parents=True)
    (stub / "conda.sh").write_text("conda() { :; }\n")
    calls = stub / "calls.txt"
    py = stub / "bin" / "python"
    py.write_text("#!/bin/bash\nfor a in \"$@\"; do\n"
                  "  if [ \"$a\" = weed_optimizer_framework.tools.inc.driver ]; then echo \"driver $*\" >> %s; exit 7; fi\n"
                  "  if [ \"$a\" = --flock-check ]; then echo \"flock $*\" >> %s; exit ${STUB_FLOCK_RC:-1}; fi\n"
                  "done\nexec %s \"$@\"\n" % (calls, calls, sys.executable))
    py.chmod(0o755)
    lines = []
    for ln in text.splitlines():
        if ln.startswith("REPO="):
            ln = "REPO=%s" % ROOT
        elif ln.startswith("CONDA_SH="):
            ln = "CONDA_SH=%s" % (stub / "conda.sh")
        lines.append(ln)
    job = stub / "run_inc_job.sh"
    job.write_text("\n".join(lines) + "\n")
    lst = stub / "list.txt"
    lst.write_text("%s\n  %s  \n" % (done_spec, bad_spec))
    base_env = dict(os.environ, PATH="%s:%s" % (stub / "bin", os.environ["PATH"]))

    def job_run(task, *args, **extra):
        if calls.exists():
            calls.unlink()
        e = dict(base_env, SLURM_ARRAY_TASK_ID=str(task), **extra)
        r = subprocess.run(["bash", str(job), str(lst)] + list(args), env=e, capture_output=True,
                           text=True, timeout=600)
        return r, (calls.read_text().splitlines() if calls.exists() else [])

    driver = "driver -u -m weed_optimizer_framework.tools.inc.driver advance --exp %s --quiet" % EXP
    flock = "flock -u -m weed_optimizer_framework.tools.inc.train --flock-check %s" % (C.INC_DIR / EXP)
    r0, c0 = job_run(0, EXP, INC_JOB_ADVANCE="1")
    r1, c1 = job_run(1, INC_JOB_ADVANCE="1")
    r2, c2 = job_run(0, EXP, STUB_FLOCK_RC="0")
    r3, c3 = job_run(0, EXP, STUB_FLOCK_RC="1")
    r4, c4 = job_run(0, EXP, INC_JOB_ADVANCE="0")
    r5, c5 = job_run(0, EXP, INC_JOB_ADVANCE="maybe")
    r9, c9 = job_run(9, EXP)
    check("task 0 runs line 1 (a done run: no-op) and exits 0 although the driver failed",
          r0.returncode == 0 and "already done" in r0.stdout, (r0.returncode, r0.stdout[-400:], r0.stderr[-400:]))
    check("task 1 runs line 2 (a refused spec) and exits 1, the executor's status",
          r1.returncode == 1 and "FAILED" in r1.stdout, (r1.returncode, r1.stdout[-400:], r1.stderr[-400:]))
    check("INC_JOB_ADVANCE=1: the driver is advanced after each run, exp from the argument or from the spec",
          c0 == [driver] and c1 == [driver], (c0, c1))
    check("auto (the default): --flock-check on INC_DIR/<exp> first; the driver only when it passes, "
          "else a warning", c2 == [flock, driver] and c3 == [flock] and "not advanced from this node" in r3.stderr
          and r2.returncode == r3.returncode == 0, (c2, c3, r3.stderr[-300:]))
    check("INC_JOB_ADVANCE=0 never advances; an unknown value warns and does not either",
          c4 == [] and "in-job advance off" in r4.stdout and c5 == [] and "not auto, 1 or 0" in r5.stderr
          and r4.returncode == r5.returncode == 0, (c4, c5, r5.stderr[-300:]))
    check("a task past the end of the list exits 2 without running anything",
          r9.returncode == 2 and "no spec on line 10" in r9.stderr and c9 == [], (r9.returncode, r9.stderr[-300:]))


def main():
    t0 = time.time()
    try:
        check("the test runs on its own INC_DIR and REPO",
              C.INC_DIR == TMP / "inc" and C.REPO == TMP / "repo", (C.INC_DIR, C.REPO))
        train_manifest, train_rows, dev_rows = build_fixture()
        hits, unhashable = C.NeverTrainGuard.load().check([r["image"] for r in train_rows])
        check("fixture: no training image is within %d bits of an exam image" % C.HOLDOUT_NEAR_DUP_BITS,
              not hits and not unhashable, hits[:3])
        test_scorer_zero_predictions()
        test_units(train_rows)
        test_refusals(train_manifest)
        test_guard(train_rows, dev_rows)
        full0 = test_full(train_manifest)
        full1 = test_save_dir(train_manifest)
        test_freeze(train_manifest, full0)
        test_lora(train_manifest, full0)
        test_lora_diverged(train_manifest, full0)
        test_diverged_and_cache(train_manifest)
        test_corrupt_image(train_rows)
        test_rerun(train_rows)
        soup = test_soup(full0, full1)
        done = test_final(soup)
        test_lock(soup)
        test_drift(soup)
        test_production(soup, train_manifest)
        test_sigterm(train_manifest)
        test_job_script(done, C.INC_DIR / EXP / "runs" / "r_key" / "spec.json")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
