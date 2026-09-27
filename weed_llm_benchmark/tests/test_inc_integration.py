#!/usr/bin/env python3
"""The INC runner end to end on the CPU: pilot build -> driver -> job script ->
executor -> scorer -> gate -> ledger -> final runs -> report, with nothing faked
but the cluster (docs/INCREMENTAL_PROTOCOL_RUNNER.md).

A fake split set lives in a temporary INC_DIR and REPO laid out like the
cluster checkout: REPO/weed_optimizer_framework (the outer copy the job
imports; here a symlink to this package), REPO/weed_llm_benchmark/ (the nested
copy and run_inc_job.sh, with only its REPO= and CONDA_SH= lines rewritten),
REPO/yolo11n.pt, the two Breal sources and their registry entry. LOCK.json
holds every manifest and scorer.py; the never-train index holds every
evaluation image.

The fixture is built so that the gate has a real effect to find on a CPU in a
few optimizer steps (a from-scratch yolo11n learns nothing measurable there):
  * every image is a 3x3 grid of 24-px textured squares on a textured
    background, one colour per class; at imgsz 64 each square is a 16-px box
    centred on a stride-8 (P3) anchor, a lattice that is the same in training
    and in the scorer's rect validation (whose canvas adds 16 px);
  * yolo11n.pt is yolo11n.yaml (nc 13) with its BatchNorm statistics
    calibrated on generated images that are in no split (at random init they
    do not match the activations, and eval-mode features vanish), a box head
    that predicts exactly one stride per side (a single DFL bin, so a few
    bias updates cannot move it), and a class head in which only P3 predicts:
    OtherPlant at bias -1, the 12 species at bias -7.0 with zero weights, just
    below the scorer's conf 0.001, so a model predicts species only after
    species positives have lifted those biases;
  * train_core has 8 sessions: the two largest (P0) are labelled OtherPlant,
    the six small ones (the bins) with their species. A null run replays P0
    only, so its species AP stays exactly 0; a cand run (D_k plus replay)
    has species positives, so its species AP is > 0, and the first step is an
    ACCEPT in both chains: a soup, a moved incumbent, and the next steps
    trained from it;
  * the recipes stay inside Ultralytics' warmup (at least 100 iterations
    whenever warmup_epochs > 0; these runs have a few), where weights ramp up
    from lr 0 while biases take warmup_bias_lr and every iteration is an
    optimizer step even at batch 8 (below nbs 64 Ultralytics otherwise
    accumulates 8 iterations per step);
  * dev holds at least 30 boxes of every species, as the gate's species guard
    requires (GateConfig.min_species_gt).

The experiment is built by inc.pilot build, testing, with the executor's test
settings (imgsz 64, batch 8, CPU); only three things are narrowed for CPU time:
the recipes (inc.pilot.cold_recipe / inc_recipes: 2 cold and 5 incremental
epochs at imgsz 64, batch 8, mosaic off, no RAM cache; LoRA rank 4), the
sequence (I1, Bswap, I2
instead of seven steps: a clean step, a planted bad one, and a clean one
whose truth-arm 'without' skips it) and the chains (full and lora). Builder,
driver, executor, scorer, gate and report are the real ones.

The backend is a LocalBackend: submit() is what sbatch would do with
SlurmBackend's own argv, i.e. array task i runs the job script
(bash run_inc_job.sh <list file> <exp>, the argv's last three) with
SLURM_ARRAY_TASK_ID=i, the environment the driver passed (submission_env), the
job's SLURM_* ids, a python stub for the conda env, and INC_JOB_ADVANCE=0:
the job's own advance calls the driver CLI, whose SlurmBackend would need
sbatch, so the test advances in-process (LocalBackend) after every batch of
finished tasks, as the job-end advance would. Tasks run up to PARALLEL at a
time, like an array.

What is pinned:
  * the pilot's production definition, initialised with a backend that runs
    nothing: every base and union spec the driver wrote passes the executor's
    validate_spec, and every spec's recipe (the chains' too, seed set as the
    driver sets it) is the protocol's for its kind (protocol_deviations []),
    so a production experiment's runs are not refused at stage recipe;
  * the same for pilot_v2's production definition (replay mode full), with
    base seed 0 marked done by hand so the driver also writes every chain's
    step-1 cand and null specs: they pass validate_spec with the protocol's
    recipe (the manifests grow, the recipe does not); cand trains P0 + I1,
    null P0;
  * the testing experiment reaches done with nothing blocked and no run
    failed, and at least one step is an ACCEPT (so the soup path runs): base 3, truth 3 per step, cand + null 6 per chain and step, a
    soup per ACCEPT, and the final runs (each chain's incumbent, the base
    seeds, T_final) on every exam;
  * every run.json is done, testing, of this spec (spec_sha256), with every
    score production=false and TEST- stamped; test is scored by final runs
    only; weights/final.pt hashes to run.json's weights_sha256; a final
    run's final.pt is a link to its init;
  * the ledger: a gate entry per chain and step and a truth entry per step,
    with input sha256s equal to the score files on disk; a soup entry per
    ACCEPT; the gate verdicts and their inputs re-derive from the score files
    with gate.decide;
  * the chain follows its decisions: each step's cand and null runs start
    from the incumbent (the spec's init and run.json's init_sha256), which
    moves only after an ACCEPT, to the soup or cand s0 as choose_soup says;
    the soup averages that step's three cand weights; D_k joins the accepted
    pool only then; a LoRA chain's next step trains from merged (or souped)
    weights;
  * the truth arm's 'without' for a step after the planted bad step is the
    last clean step's 'with', and T_final holds the clean data only;
  * the report agrees with the ledger (verdicts, truth, agreement), fills the
    final table for every exam (one incumbent per chain, three seeds of the
    base and of T_final), says TESTING, and its GPU-hours are the run.json
    seconds; the report and status CLIs run;
  * the job script's in-job advance (driver CLI) runs cleanly on the done
    experiment; one more advance submits nothing and writes nothing.

Run:  python3 tests/test_inc_integration.py      (about 5-10 minutes on a laptop CPU)
"""
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_inc_integration_"))
# common reads INC_DIR and REPO at import: this test never sees the machine's real ones.
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["INC_SCORER_TESTING"] = "1"            # tests only: the experiment here is a testing one
for _k in list(os.environ):
    if _k.startswith(("SLURM_", "SBATCH_")) or _k in ("INC_ALLOW_DRIFT", "INC_JOB_ADVANCE", "INC_JOB_SCRIPT"):
        os.environ.pop(_k, None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc import report as R  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import train as T  # noqa: E402

FAILURES = []
EXP = "integ_v1"
EXP_PROD = "integ_prod"
EXP_PROD_FULL = "integ_prod_full"
TESTING = {"imgsz": 64, "batch": 8, "device": "cpu"}
SEQUENCE = ("I1", "Bswap", "I2")
CHAINS = ("full", "lora")
PARALLEL = 4
THREADS_PER_TASK = "2"
MAX_ROUNDS = 200
IMG = 96                      # source images; 64 px at the test imgsz
CELL = 24                     # grid cell and square size (16 px at imgsz 64: one P3 stride per side)
GRID = 3
ORIGIN = 6                    # square centres at 18, 42, 66 px = 12, 28, 44 at imgsz 64: P3 anchors (4 + 8k)
SPECIES_BIAS = -7.0           # just below logit(0.001) = -6.907, the scorer's conf threshold
SLUG_A, SLUG_B = P.BREAL_SLUGS
# One colour per class, so that a box's class can be seen.
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
           (220, 190, 255), (128, 128, 128)]


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def sha(path):
    return C.sha256_file(path)


def read(path):
    with open(path) as fh:
        return json.load(fh)


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# ------------------------------------------------------------------ recipes
def tiny_cold():
    """The cold recipe, narrowed for a CPU: every key the protocol's has. The
    whole run is inside Ultralytics' warmup (module docstring); close_mosaic =
    epochs turns mosaic off from the first epoch."""
    return {"trainer": "full", "epochs": 2, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
            "momentum": 0.9, "weight_decay": 0.0005, "warmup_epochs": 1, "warmup_bias_lr": 0.5,
            "cos_lr": True, "freeze": None, "lora": None, "imgsz": 64, "batch": 8,
            "cache": False, "workers": 0, "close_mosaic": 2, "deterministic": True}


def tiny_inc():
    full = dict(tiny_cold(), epochs=5, close_mosaic=5, warmup_bias_lr=1.0)
    lora = dict(full, trainer="lora", warmup_bias_lr=0.5, lora={"rank": 4, "alpha": 8})
    return {"full": full, "lora": lora}


# ------------------------------------------------------------------ fixture
def _lowfreq(rng, lo, hi):
    """IMG x IMG float array: a random 9x8 grid upsampled (every image its own dHash)."""
    small = rng.integers(lo, hi, (8, 9)).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((IMG, IMG), Image.BICUBIC), dtype=np.float32)


def make_image(path, rng, classes, draw_as=None):
    """IMG x IMG JPEG: a textured background and, in reading order over the
    3x3 grid, one CELL x CELL textured square per entry of classes, coloured
    by draw_as (default: the class itself). Returns the YOLO boxes."""
    im = np.repeat(_lowfreq(rng, 40, 216)[:, :, None], 3, 2)
    tex = _lowfreq(rng, 110, 256)[:, :, None] / 255.0
    boxes = []
    for i, c in enumerate(classes):
        cx, cy = ORIGIN + CELL * (i % GRID) + CELL // 2, ORIGIN + CELL * (i // GRID) + CELL // 2
        x, y = cx - CELL // 2, cy - CELL // 2
        colour = np.asarray(PALETTE[draw_as[i] if draw_as is not None else c], np.float32)
        im[y:y + CELL, x:x + CELL] = colour[None, None, :] * tex[y:y + CELL, x:x + CELL]
        boxes.append((int(c), cx / float(IMG), cy / float(IMG), CELL / float(IMG), CELL / float(IMG)))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(path, quality=95)
    return boxes


def row(img, lab, source, session, key):
    return {"image": str(img), "label": str(lab), "sha256": sha(img), "label_sha256": sha(lab),
            "source": source, "session": session, "key": key}


def build_world():
    """train_core, the exams, LOCK.json, the never-train index, REPO's layout,
    yolo11n.pt and the Breal sources. Returns train_core's rows."""
    rng = np.random.default_rng(20260926)
    repo = C.REPO
    # train_core: two large sessions (P0: OtherPlant labels on species-coloured boxes),
    # six small ones (the bins: species labels).
    sessions = [("20210801_camA", 12, True), ("20210802_camA", 12, True)]
    sessions += [("202108%02d_camB" % (10 + i), 4, False) for i in range(6)]
    core = []
    for sess, n, other in sessions:
        for j in range(n):
            stem = "%s_%d" % (sess, j + 1)
            img = repo / "downloads" / "cwd12" / "images" / (stem + ".jpg")
            lab = repo / "downloads" / "cwd12" / "labels" / (stem + ".txt")
            species = [int(x) for x in rng.integers(0, 12, GRID * GRID)]
            boxes = make_image(img, rng, species)
            if other:
                boxes = [(C.OTHER_PLANT,) + b[1:] for b in boxes]
            C.write_yolo(lab, boxes)
            core.append(row(img, lab, "cottonweeddet12/train", sess, "train_core__" + stem))
    manifests = {"train_core": C.write_manifest(C.manifest_path("train_core"), core)}

    # the exams: dev with >= 30 boxes of every species (the gate's species guard), no OtherPlant
    n_dev = -(-12 * G.GateConfig().min_species_gt // (GRID * GRID))
    exams = {"dev": n_dev, "test": 4, "ood22": 4, "ood23": 4, "imageweeds": 4}
    entries = []
    for split, n in exams.items():
        rows = []
        for j in range(n):
            key = "%s__img%03d" % (split, j)
            img = TMP / "exams_src" / split / (key + ".jpg")
            lab = TMP / "exams_src" / split / (key + ".txt")
            classes = ([(GRID * GRID * j + i) % 12 for i in range(GRID * GRID)] if split == "dev"
                       else [int(x) for x in rng.integers(0, 13, GRID * GRID)])
            C.write_yolo(lab, make_image(img, rng, classes))
            rows.append(row(img, lab, split, "", key))
            entries.append([C.dhash(img), split, key])
        manifests[split] = C.write_manifest(C.manifest_path(split), rows)
        C.materialise(rows, C.EXAMS_DIR / split)
    C.LOCK_PATH.write_text(json.dumps({"manifests": manifests,
                                       "scorer_sha256": sha(pathlib.Path(S.__file__).resolve())}))
    C.NEVER_TRAIN_INDEX.write_text(json.dumps({"entries": entries, "bits": C.HOLDOUT_NEAR_DUP_BITS,
                                               "complete": True, "min_expected": len(entries)}))

    # REPO as on the cluster: the outer copy the job imports, the nested git-tracked copy,
    # the job script (only REPO= and CONDA_SH= rewritten) and a stub conda env.
    (repo / "weed_llm_benchmark").mkdir(parents=True, exist_ok=True)
    os.symlink(str(ROOT / "weed_optimizer_framework"), str(repo / "weed_optimizer_framework"))
    os.symlink(str(ROOT / "weed_optimizer_framework"), str(repo / "weed_llm_benchmark" / "weed_optimizer_framework"))
    stub = TMP / "stub"
    (stub / "bin").mkdir(parents=True)
    (stub / "conda.sh").write_text("conda() { :; }\n")
    py = stub / "bin" / "python"
    py.write_text("#!/bin/bash\nexec %s \"$@\"\n" % sys.executable)
    py.chmod(0o755)
    lines = []
    for ln in (ROOT / "run_inc_job.sh").read_text().splitlines():
        if ln.startswith("REPO="):
            ln = "REPO=%s" % repo
        elif ln.startswith("CONDA_SH="):
            ln = "CONDA_SH=%s" % (stub / "conda.sh")
        lines.append(ln)
    (repo / "weed_llm_benchmark" / "run_inc_job.sh").write_text("\n".join(lines) + "\n")
    cold_checkpoint(repo / D.COLD_INIT, rng)

    # Breal: two sources and their registry entry (names joined through species_of)
    registry = {}
    for slug, names, sub, n in ((SLUG_A, ["Ragweed", "corn", "Waterhemp"], "", 6),
                                (SLUG_B, {"0": "weed"}, "train", 5)):
        d = repo / "datasets" / slug / sub if sub else repo / "datasets" / slug
        n_cls = len(names)
        for j in range(n):
            img = d / "images" / ("src_%03d.jpg" % j)
            cls = [int(x) for x in rng.integers(0, n_cls, GRID * GRID)]
            boxes = make_image(img, rng, cls, draw_as=[[5, 12, 0][c] if n_cls == 3 else 12 for c in cls])
            C.write_yolo(d / "labels" / ("src_%03d.txt" % j), boxes)
        registry[slug] = {"class_names": names, "annotation": "bbox"}
    reg = P.registry_path()
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps({"datasets": registry}))
    return core


def cold_checkpoint(path, rng, seed=0):
    """REPO/yolo11n.pt for this fixture (module docstring): yolo11n.yaml at nc
    13, BatchNorm statistics calibrated on 32 generated images in no split,
    every level's box head at one stride per side, only P3 predicting classes
    (OtherPlant at bias -1, species at SPECIES_BIAS with zero weights)."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(seed)
    net = DetectionModel("yolo11n.yaml", nc=C.NC, verbose=False)
    det = net.model[-1]
    with torch.no_grad():
        for level in range(det.nl):
            cls = det.cv3[level][-1]
            cls.bias.fill_(-10.0)
            if level == 0:
                cls.bias[:C.OTHER_PLANT] = SPECIES_BIAS
                cls.bias[C.OTHER_PLANT] = -1.0
            cls.weight[:C.OTHER_PLANT].zero_()
            box = det.cv2[level][-1]
            box.weight.mul_(0.01)
            bins = torch.full((det.reg_max,), -10.0)
            bins[1] = 0.0
            box.bias.copy_(bins.repeat(4))
    calib = TMP / "calibration"
    arrays = []
    for j in range(32):
        img = calib / ("c%02d.jpg" % j)
        make_image(img, rng, [int(x) for x in rng.integers(0, C.NC, GRID * GRID)])
        arrays.append(np.asarray(Image.open(img).convert("RGB").resize((64, 64))))
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                    # a cumulative average over the calibration batch
    net.train()
    with torch.no_grad():
        net(x)
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    net.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net, "train_args": {}}, path)
    return path


# ------------------------------------------------------------------ backend
class LocalBackend(D.Backend):
    """sbatch, squeue and sacct for one machine: submit() queues the array that
    SlurmBackend.sbatch_argv describes; run_pending() runs its tasks through the
    job script, PARALLEL at a time."""

    def __init__(self):
        self.subs = []
        self.tasks = {}
        self.order = []
        self._next, self._raw = 7000, 90000
        self.log = []

    def submit(self, list_file, n, exp, job_name, log_dir, env=None, time_limit=None):
        argv = D.SlurmBackend().sbatch_argv(list_file, n, exp, job_name, log_dir, time_limit)
        script, lst, e = argv[-3:]
        with open(lst) as fh:
            specs = [ln.strip() for ln in fh if ln.strip()]
        if len(specs) != n or not argv[2].startswith("--array=0-%d%%" % (n - 1)):
            raise D.SubmitNotStarted("the list file holds %d specs for an array of %d (%s)" % (len(specs), n, argv[2]))
        jid = str(self._next)
        self._next += 1
        self.subs.append({"job_id": jid, "argv": argv, "specs": specs, "env": dict(env or {}),
                          "time_limit": time_limit, "exp": e})
        for i, s in enumerate(specs):
            t = "%s_%d" % (jid, i)
            self.tasks[t] = {"state": "PENDING", "raw": str(self._raw), "index": i, "job_id": jid,
                             "script": script, "list": lst, "exp": e, "spec": s}
            self.order.append(t)
            self._raw += 1
        return jid

    def queued(self):
        out = set()
        for t, v in self.tasks.items():
            if v["state"] in ("PENDING", "RUNNING"):
                out.update((t, v["raw"]))
        return out

    def task_states(self, job_ids):
        ids, out = set(job_ids), {}
        for t, v in self.tasks.items():
            if v["job_id"] in ids or t in ids or v["raw"] in ids:
                out[t] = out[v["raw"]] = v["state"]
        return out

    def lookup(self, job_name, since_ts=None):
        return D.ABSENT, None

    def pending(self):
        return [t for t in self.order if self.tasks[t]["state"] == "PENDING"]

    def _env(self, t):
        v = self.tasks[t]
        sub = [s for s in self.subs if s["job_id"] == v["job_id"]][0]
        env = dict(sub["env"])
        env.update(SLURM_JOB_ID=v["raw"], SLURM_ARRAY_JOB_ID=v["job_id"], SLURM_ARRAY_TASK_ID=str(v["index"]),
                   PATH="%s:%s" % (TMP / "stub" / "bin", env.get("PATH", os.environ["PATH"])),
                   INC_JOB_ADVANCE="0", OMP_NUM_THREADS=THREADS_PER_TASK, MKL_NUM_THREADS=THREADS_PER_TASK)
        return env

    def run_pending(self, limit=PARALLEL):
        """Start up to `limit` pending tasks, wait for them; returns how many ran."""
        batch = self.pending()[:limit]
        procs = []
        for t in batch:
            v = self.tasks[t]
            v["state"] = "RUNNING"
            out = TMP / "tasklogs" / ("%s.log" % t)
            out.parent.mkdir(parents=True, exist_ok=True)
            fh = open(out, "w")
            p = subprocess.Popen(["bash", v["script"], v["list"], v["exp"]], env=self._env(t),
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(TMP))
            procs.append((t, p, fh, out, time.time()))
        for t, p, fh, out, t0 in procs:
            rc = p.wait()
            fh.close()
            v = self.tasks[t]
            v.update(state="COMPLETED" if rc == 0 else "FAILED", rc=rc, seconds=time.time() - t0, logfile=str(out))
            self.log.append((t, rc, pathlib.Path(v["spec"]).parent.name, round(time.time() - t0, 1)))
            if rc != 0:
                print("       task %s (%s) exited %d; log tail:\n%s"
                      % (t, pathlib.Path(v["spec"]).parent.name, rc, "\n".join(
                          out.read_text().splitlines()[-25:])))
        return len(batch)


def state_of(exp):
    return read(D.Paths(exp).state)


def ledger_of(exp):
    p = D.Paths(exp).ledger
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def drive(exp, backend):
    """advance, run a batch of tasks, advance, ... until done or nothing moves."""
    t0 = time.time()
    for i in range(MAX_ROUNDS):
        D.Driver(exp, backend=backend, quiet=True).advance()
        st = state_of(exp)
        if st["done"]:
            break
        if not backend.run_pending():
            break
        if i % 5 == 0:
            n = len(st["runs"])
            c = sum(1 for r in st["runs"].values() if r["status"] == "complete")
            print("       ... %.0fs: %d/%d runs complete, %d task(s) run" % (time.time() - t0, c, n, len(backend.log)))
    return state_of(exp)


# ------------------------------------------------------------------ tests
def test_production_specs(core):
    """The pilot's production definition, initialised without running anything:
    the specs the driver writes are ones the executor accepts, with the
    protocol's recipe for their kind."""
    print("production pilot: driver specs vs the executor's spec rules and protocol recipe")
    with patched(P, "SEQUENCE", SEQUENCE):
        backend = LocalBackend()            # nothing is run: only the specs and the submission
        # A production build and advance must not see the test-mode variable.
        old = os.environ.pop("INC_SCORER_TESTING")
        try:
            summary, defn, res = P.build_pilot(EXP_PROD, testing=False, backend=backend, quiet=True)
        finally:
            os.environ["INC_SCORER_TESTING"] = old
    paths = D.Paths(EXP_PROD)
    specs = {d.name: read(d / "spec.json") for d in sorted(paths.runs.iterdir())}
    kinds = sorted({s["kind"] for s in specs.values()})
    bad = []
    for rid, s in specs.items():
        try:
            T.validate_spec(s, paths.spec(rid))
        except T.RunError as e:
            bad.append((rid, str(e)))
    check("init writes the base and truth specs (%d), each accepted by the executor's validate_spec"
          % len(specs), kinds == ["base", "union"] and len(specs) == 3 + 3 * len(SEQUENCE) and not bad,
          (kinds, len(specs), bad[:2]))
    devs = {rid: T.protocol_deviations(s["kind"], s["recipe"]) for rid, s in specs.items()}
    devs.update({"chain %s" % r: T.protocol_deviations("cand", dict(rec, seed=0))
                 for r, rec in defn["recipes"].items()})
    check("every production recipe is the protocol's for its kind (base, union, and cand/null of %s)"
          % sorted(defn["recipes"]), not any(devs.values()), {k: v for k, v in devs.items() if v})
    sub = backend.subs[0] if backend.subs else {}
    check("the production submission is one array holding every run, with the cold time limit, and no "
          "test-mode variable in its environment",
          len(backend.subs) == 1 and len(sub["specs"]) == len(specs) and sub["time_limit"] == D.COLD_TIME_LIMIT
          and S.TEST_ENV not in sub["env"], [(s["time_limit"], len(s["specs"])) for s in backend.subs])
    argv = sub.get("argv") or [None] * 3
    check("the job the driver submits is REPO/weed_llm_benchmark/run_inc_job.sh <list> <exp>",
          argv[-3] == str(C.REPO / "weed_llm_benchmark" / "run_inc_job.sh") and argv[-1] == EXP_PROD
          and pathlib.Path(str(argv[-2])).is_file(), argv)
    check("the production experiment is not a testing one", state_of(EXP_PROD)["testing"] is False
          and read(paths.exp_json)["testing"] is False)


def test_production_specs_full(core):
    """pilot_v2's production definition (replay mode full), initialised without
    training: base seed 0 is marked done by hand so the driver writes every
    chain's step-1 cand and null specs; those, and the init specs, are the ones
    the executor accepts, with the protocol's recipe for their kind."""
    print("production pilot, replay mode full: cand / null specs vs the executor's spec rules and protocol recipe")
    with patched(P, "SEQUENCE", SEQUENCE):
        backend = LocalBackend()            # nothing is run
        old = os.environ.pop("INC_SCORER_TESTING")
        try:
            summary, defn, res = P.build_pilot(EXP_PROD_FULL, testing=False, backend=backend, quiet=True,
                                               replay_mode="full")
            paths = D.Paths(EXP_PROD_FULL)
            rid = "base__s0"                # as the executor leaves a finished run
            (paths.run_dir(rid) / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(C.REPO / D.COLD_INIT, paths.weights(rid))
            (paths.run_dir(rid) / "scores").mkdir(parents=True, exist_ok=True)
            paths.score(rid, "dev").write_text(json.dumps({"exam": "dev", "production": True}))
            paths.run_json(rid).write_text(json.dumps({"status": "done", "seconds": 1.0,
                                                       "weights_sha256": sha(paths.weights(rid))}))
            D.Driver(EXP_PROD_FULL, backend=backend, quiet=True).advance()
        finally:
            os.environ["INC_SCORER_TESTING"] = old
    st = state_of(EXP_PROD_FULL)
    specs = {d.name: read(d / "spec.json") for d in sorted(paths.runs.iterdir())}
    chain_specs = {r: s for r, s in specs.items() if s["kind"] in ("cand", "null")}
    bad = []
    for rid, s in specs.items():
        try:
            T.validate_spec(s, paths.spec(rid))
        except T.RunError as e:
            bad.append((rid, str(e)))
    check("the build defines replay mode full, production, with the sample build's recipes",
          defn["replay_mode"] == "full" and summary["replay_mode"] == "full" and defn["testing"] is False
          and defn["recipes"] == P.inc_recipes() and defn["truth_recipe"] == P.cold_recipe())
    check("with base seed 0 done, every chain's step-1 cand and null specs are written (%d), and the executor's "
          "validate_spec accepts all %d specs" % (len(chain_specs), len(specs)),
          len(chain_specs) == 6 * len(defn["recipes"]) and not bad and not st["blocked"], (bad[:2], st["blocked"]))
    devs = {rid: T.protocol_deviations(s["kind"], s["recipe"]) for rid, s in specs.items()}
    check("every production recipe of pilot_v2 is the protocol's for its kind: base, union, and the chains' "
          "cand / null as the driver wrote them (the manifests grow, the recipe does not)",
          not any(devs.values()) and all(s["recipe"] == dict(defn["recipes"][rid.split("__")[0]],
                                                              seed=s["recipe"]["seed"])
                                         for rid, s in chain_specs.items()),
          {k: v for k, v in devs.items() if v})
    p0 = {r["key"] for r in C.read_manifest(defn["base"]["manifest"])}
    i1 = {r["key"] for r in C.read_manifest(defn["steps"][0]["manifest"])}
    sets_ok = all({r["key"] for r in C.read_manifest(s["train_manifest"])} == (p0 | i1 if s["kind"] == "cand" else p0)
                  for s in chain_specs.values())
    check("step 1: cand trains P0 + I1, null P0, from base seed 0's weights", sets_ok and all(
        s["init"] == str(paths.weights("base__s0")) for s in chain_specs.values()))
    sub = backend.subs[-1] if backend.subs else {}
    check("the chain array keeps the job script's time limit and has no test-mode variable",
          len(backend.subs) == 2 and sub["time_limit"] is None and S.TEST_ENV not in sub["env"]
          and len(sub["specs"]) == len(chain_specs), [(s["time_limit"], len(s["specs"])) for s in backend.subs])


def test_run(core):
    print("testing pilot: build, then drive to done through the job script and the real executor")
    backend = LocalBackend()
    t0 = time.time()
    with patched(P, "SEQUENCE", SEQUENCE), patched(P, "cold_recipe", tiny_cold), \
            patched(P, "inc_recipes", tiny_inc):
        summary, defn, res = P.build_pilot(EXP, testing=dict(TESTING), backend=backend, quiet=True)
    check("the build defines the narrowed pilot: steps %s, chains %s, truth on, testing settings"
          % (list(SEQUENCE), list(CHAINS)),
          [s["name"] for s in defn["steps"]] == list(SEQUENCE) and sorted(defn["recipes"]) == sorted(CHAINS)
          and defn["truth"] is True and defn["testing"] == TESTING
          and [s["clean"] for s in defn["steps"]] == [True, False, True], defn.get("steps"))
    st = drive(EXP, backend)
    minutes = (time.time() - t0) / 60.0
    print("       drove %s in %.1f min: %d task(s)" % (EXP, minutes, len(backend.log)))
    blocked = st.get("blocked")
    check("the experiment reaches done, nothing blocked", st["done"] and not blocked,
          (st["done"], blocked, [ln for ln in D.Driver(EXP).status()]))
    failed = [(rid, r["status"], r["history"]) for rid, r in st["runs"].items()
              if r["status"] != "complete" or r["history"]]
    check("every run completed at its first attempt", not failed, failed[:3])
    check("every job-script task exited 0", all(rc == 0 for _, rc, _, _ in backend.log),
          [x for x in backend.log if x[1] != 0][:5])
    return st, defn, backend


def test_runs(st, defn, backend):
    print("runs: specs, run.json, scores, weights")
    paths = D.Paths(EXP)
    ledger = ledger_of(EXP)
    n_accept = sum(1 for e in ledger if e["type"] == "gate" and e["decision"]["verdict"] == G.ACCEPT)
    kinds = {}
    for rid, r in st["runs"].items():
        kinds.setdefault(r["kind"], []).append(rid)
    n_final = len(CHAINS) + 3 + 3
    want = {"base": 3, "union": 3 * len(SEQUENCE), "cand": 3 * len(SEQUENCE) * len(CHAINS),
            "null": 3 * len(SEQUENCE) * len(CHAINS), "final": n_final}
    if n_accept:
        want["soup"] = n_accept
    got = {k: len(v) for k, v in kinds.items()}
    check("runs per kind %s (a soup per ACCEPT)" % want, got == want, got)
    bad, test_seen = [], []
    for rid in st["runs"]:
        spec = read(paths.spec(rid))
        rj = read(paths.run_json(rid))
        if "test" in spec["exams"]:
            test_seen.append(spec["kind"])
        w = paths.weights(rid)
        problems = []
        if rj.get("status") != "done" or rj.get("testing") is not True or rj.get("exp") != EXP:
            problems.append("status/testing/exp %s/%s/%s" % (rj.get("status"), rj.get("testing"), rj.get("exp")))
        if rj.get("spec_sha256") != T.spec_digest(spec, paths.spec(rid)):
            problems.append("spec_sha256")
        if sorted(rj.get("scores") or {}) != sorted(spec["exams"]):
            problems.append("scores %s != exams %s" % (sorted(rj.get("scores") or {}), spec["exams"]))
        for e in spec["exams"]:
            s = read(paths.score(rid, e))
            if s.get("production") is not False or not s["scorer_sha256"].startswith(S.TEST_PREFIX) \
                    or s["exam"] != e or s["weights_sha256"] != rj.get("weights_sha256"):
                problems.append("score %s" % e)
            if (rj["scores"].get(e) or {}).get("sha256") != sha(paths.score(rid, e)):
                problems.append("run.json score sha %s" % e)
        if not w.exists() or sha(w) != rj.get("weights_sha256"):
            problems.append("weights/final.pt")
        if spec["kind"] == "final" and not (w.is_symlink() and os.path.realpath(w) == os.path.realpath(spec["init"])):
            problems.append("final.pt is not a link to init")
        if spec["kind"] in ("base", "union", "cand", "null"):
            if not rj.get("n_train_images") or (rj.get("guard") or {}).get("hits") != 0:
                problems.append("train record %s %s" % (rj.get("n_train_images"), rj.get("guard")))
            if rj.get("init_sha256") != sha(spec["init"] if os.path.isabs(spec["init"]) else C.REPO / spec["init"]):
                problems.append("init_sha256")
        if (paths.run_dir(rid) / "data").exists():
            problems.append("data/ left behind")
        if problems:
            bad.append((rid, problems))
    check("every run.json is done, testing, of its spec; every score TEST- stamped and of these weights; "
          "final.pt as recorded (a link to init for a final run); no data/ left", not bad, bad[:4])
    check("test is scored by final runs only, and by every one of them",
          set(test_seen) == {"final"} and len(test_seen) == n_final, test_seen)
    fin = [read(paths.spec(rid)) for rid in kinds["final"]]
    check("final runs score every exam %s" % list(D.FINAL_EXAMS),
          all(s["exams"] == list(D.FINAL_EXAMS) for s in fin))
    recs = [read(paths.run_json(rid)) for rid in st["runs"]]
    check("run.json records the job's Slurm ids and the testing settings",
          all(r.get("slurm_job_id") and r.get("slurm_array_task_id") is not None for r in recs)
          and all(r.get("testing_settings") == TESTING for r in recs), recs[0].get("slurm_job_id"))
    return n_accept


def test_ledger_and_chains(st, defn):
    print("ledger and chains: decisions, inputs, incumbents")
    paths = D.Paths(EXP)
    ledger = ledger_of(EXP)
    by_id = {e["id"]: e for e in ledger}
    missing = [i for i in (["gate/%s/%d" % (r, n) for r in CHAINS for n in range(1, len(SEQUENCE) + 1)]
                           + ["truth/%d" % n for n in range(1, len(SEQUENCE) + 1)]) if i not in by_id]
    check("a gate entry per chain and step, a truth entry per step", not missing, missing)
    stale = []
    for e in ledger:
        for arm in (e.get("inputs") or {}).values():
            for x in (arm if isinstance(arm, list) else [arm]):
                if sha(x["path"]) != x["sha256"]:
                    stale.append((e["id"], x["path"]))
    check("every ledger input sha256 is the score file on disk", not stale, stale[:3])
    check("every ledger entry says testing", all(e.get("testing") is True for e in ledger))

    cfg = G.GateConfig(require_production=False)
    redo = []
    for e in (x for x in ledger if x["type"] == "gate"):
        inp = e["inputs"]
        d = G.decide(read(inp["inc"]["path"]), [read(x["path"]) for x in inp["cand"]],
                     [read(x["path"]) for x in inp["null"]], cfg)
        if d.to_dict() != e["decision"]:
            redo.append(e["id"])
    check("every gate decision re-derives from its score files with gate.decide", not redo, redo)

    problems = []
    verdicts = {}
    for r in CHAINS:
        ch = st["chains"][r]
        inc = {"run_id": "base__s0", "weights": str(paths.weights("base__s0"))}
        accepted = []
        for n in range(1, len(SEQUENCE) + 1):
            stp = ch["steps"][str(n)]
            g = by_id["gate/%s/%d" % (r, n)]
            verdicts[(r, n)] = g["decision"]["verdict"]
            for rid in stp["cand"] + stp["null"]:
                spec = read(paths.spec(rid))
                rj = read(paths.run_json(rid))
                if spec["init"] != inc["weights"] or rj.get("init_sha256") != sha(inc["weights"]):
                    problems.append("%s: init %s, incumbent %s" % (rid, spec["init"], inc["weights"]))
                if spec["recipe"] != dict(defn["recipes"][r], seed=spec["recipe"]["seed"]):
                    problems.append("%s: recipe" % rid)
            if g["incumbent_before"]["run_id"] != inc["run_id"]:
                problems.append("gate/%s/%d incumbent_before %s" % (r, n, g["incumbent_before"]["run_id"]))
            # replay samples: disjoint, from the accepted pool, |D_k| each
            pool = set()
            for name in ["P0"] + accepted:
                src = defn["base"] if name == "P0" else [s for s in defn["steps"] if s["name"] == name][0]
                pool |= {x["key"] for x in C.read_manifest(src["manifest"])}
            d_keys = {x["key"] for x in C.read_manifest(defn["steps"][n - 1]["manifest"])}
            r1 = {x["key"] for x in C.read_manifest(stp["manifests"]["R1"]["path"])}
            r2 = {x["key"] for x in C.read_manifest(stp["manifests"]["R2"]["path"])}
            if r1 & r2 or not (r1 | r2) <= pool or len(r1) != len(d_keys) or len(r2) != len(d_keys):
                problems.append("%s step %d replay" % (r, n))
            cand_keys = {x["key"] for x in C.read_manifest(stp["manifests"]["cand"]["path"])}
            null_keys = {x["key"] for x in C.read_manifest(stp["manifests"]["null"]["path"])}
            if cand_keys != d_keys | r1 or null_keys != r1 | r2:
                problems.append("%s step %d cand/null manifests" % (r, n))
            if g["decision"]["verdict"] == G.ACCEPT:
                se = by_id.get("soup/%s/%d" % (r, n))
                soup = read(paths.run_json(stp["soup"]))
                cand_w = [str(paths.weights(x)) for x in stp["cand"]]
                real = [os.path.realpath(x) for x in cand_w]
                if se is None or [os.path.realpath(x) for x in soup.get("soup_of") or []] != real \
                        or read(paths.spec(stp["soup"]))["soup_of"] != cand_w:
                    problems.append("%s step %d soup" % (r, n))
                else:
                    want = G.choose_soup(read(paths.score(stp["soup"], "dev")),
                                         [read(paths.score(x, "dev")) for x in stp["cand"]], cfg)
                    new = stp["soup"] if want == "soup" else stp["cand"][0]
                    if se["choice"] != want or se["incumbent_after"]["run_id"] != new:
                        problems.append("%s step %d soup choice %s != %s" % (r, n, se["choice"], want))
                    inc = {"run_id": new, "weights": str(paths.weights(new))}
                accepted.append(SEQUENCE[n - 1])
            elif stp.get("soup"):
                problems.append("%s step %d: a soup without ACCEPT" % (r, n))
            if stp["incumbent_after"]["run_id"] != inc["run_id"]:
                problems.append("%s step %d incumbent_after %s != %s"
                                % (r, n, stp["incumbent_after"]["run_id"], inc["run_id"]))
        if ch["accepted"] != accepted or ch["incumbent"]["run_id"] != inc["run_id"]:
            problems.append("%s: accepted %s / incumbent %s, expected %s / %s"
                            % (r, ch["accepted"], ch["incumbent"]["run_id"], accepted, inc["run_id"]))
        fin = read(paths.spec("final__%s__incumbent" % r))
        if fin["init"] != inc["weights"]:
            problems.append("%s: final incumbent %s != %s" % (r, fin["init"], inc["weights"]))
    check("each chain follows its decisions: cand/null from the incumbent, replay disjoint from the "
          "accepted pool, a soup of the three cands per ACCEPT, the incumbent moved as choose_soup says, "
          "the final run on the last incumbent", not problems, problems[:5])
    print("       verdicts: %s" % ", ".join("%s/%s %s" % (r, SEQUENCE[n - 1], v) for (r, n), v in sorted(verdicts.items())))

    tr = st["truth"]["steps"]
    want_without = {1: ["base__s%d" % s for s in (0, 1, 2)],
                    2: ["truth__s01_I1__union__s%d" % s for s in (0, 1, 2)],
                    3: ["truth__s01_I1__union__s%d" % s for s in (0, 1, 2)]}
    got = {n: tr[str(n)]["without"] for n in (1, 2, 3)}
    t_final = C.read_manifest(tr["3"]["manifest"]["path"])
    clean_keys = set()
    for name in ("P0", "I1", "I2"):
        src = defn["base"] if name == "P0" else [s for s in defn["steps"] if s["name"] == name][0]
        clean_keys |= {x["key"] for x in C.read_manifest(src["manifest"])}
    fins = [read(paths.spec("final__Tfinal__s%d" % s))["init"] for s in (0, 1, 2)]
    check("truth arm: after the planted Bswap, 'without' is I1's 'with'; T_final is P0+I1+I2 and its "
          "runs get the final runs", got == want_without and {x["key"] for x in t_final} == clean_keys
          and fins == [str(paths.weights("truth__s03_I2__union__s%d" % s)) for s in (0, 1, 2)], (got, fins))
    return verdicts


def test_report(st, defn, n_accept):
    print("report")
    paths = D.Paths(EXP)
    r = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc.report", "--exp", EXP],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    check("the report CLI exits 0 and says TESTING", r.returncode == 0 and "(TESTING)" in r.stdout,
          (r.returncode, r.stdout[-300:], r.stderr[-800:]))
    rep = read(paths.root / "report.json")
    md = (paths.root / "report.md").read_text()
    ledger = {e["id"]: e for e in ledger_of(EXP)}
    bad = []
    for srow in rep["steps"]:
        n = srow["k"]
        t = ledger["truth/%d" % n]["detail"]
        if srow["truth"]["verdict"] != t["verdict"]:
            bad.append("truth %d" % n)
        for c in CHAINS:
            g = ledger["gate/%s/%d" % (c, n)]["decision"]
            x = srow["chains"][c]
            if x is None or x["verdict"] != g["verdict"] or x["p_data"] != g["p_data"] \
                    or x["agree"] != (D.VERDICT_TO_TRUTH[g["verdict"]] == t["verdict"]):
                bad.append("%s %d" % (c, n))
    check("report steps: every chain's verdict, P_data and agreement, and the truth verdict, as in the ledger",
          len(rep["steps"]) == len(SEQUENCE) and not bad, bad)
    fin = {row["model"]: row for row in rep["final"]}
    want_n = {("chain %s: final incumbent" % c): 1 for c in CHAINS}
    want_n.update({"base P0": 3, "T_final (union of clean data)": 3})
    holes = []
    for model, n in want_n.items():
        row = fin.get(model)
        if row is None:
            holes.append((model, "missing"))
            continue
        for e in R.REPORT_EXAMS:
            for k in ("twelve", "agnostic"):
                m = row["exams"][e][k]
                if m["n"] != n or m["mean"] is None:
                    holes.append((model, e, k, m))
        if row["production"] != [False]:
            holes.append((model, "production", row["production"]))
    check("final table: every model on every exam (12-class and agnostic), n = 1 per chain and 3 for the "
          "base and T_final, all test-mode", not holes and len(fin) == len(want_n), holes[:4])
    secs = sum(read(paths.run_json(rid))["seconds"] for rid in st["runs"])
    check("GPU-hours are the run.json seconds (%.1f h)" % (secs / 3600.0),
          abs(rep["gpu_hours_total"] * 3600.0 - secs) < 1e-6
          and sum(v["runs"] for v in rep["gpu_hours"].values()) == len(st["runs"]), (rep["gpu_hours_total"], secs))
    check("report.md: TESTING banner, decisions, final table and GPU-hours",
          "TESTING experiment" in md and "## Decisions per step" in md and "## Final quality" in md
          and "## GPU-hours" in md and rep["testing"] is True and rep["done"] is True)
    if n_accept:
        check("an ACCEPT's soup choice reaches the report", any(
            srow["chains"][c] and srow["chains"][c]["soup_choice"] for srow in rep["steps"] for c in CHAINS))


def test_cli_and_idempotence(backend):
    print("CLIs, the job's in-job advance, idempotence")
    paths = D.Paths(EXP)
    r = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc.driver", "status", "--exp", EXP],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    check("status CLI: done, [TESTING]", r.returncode == 0 and "[TESTING] done: yes" in r.stdout, r.stdout[-600:])
    before = (paths.state.read_bytes(), paths.ledger.read_bytes())
    n_subs = len(backend.subs)
    D.Driver(EXP, backend=backend, quiet=True).advance()
    check("one more advance submits nothing and writes nothing",
          len(backend.subs) == n_subs and (paths.state.read_bytes(), paths.ledger.read_bytes()) == before)
    # The job script with its own advance on: the driver CLI, on a done experiment, with the job's env.
    t = backend.order[-1]
    v = backend.tasks[t]
    env = dict(backend._env(t), INC_JOB_ADVANCE="1")
    r = subprocess.run(["bash", v["script"], v["list"], v["exp"]], env=env, capture_output=True, text=True,
                       timeout=600, cwd=str(TMP))
    check("the job script re-run on a done spec: the executor no-ops, the in-job driver advance runs "
          "cleanly, exit 0", r.returncode == 0 and "already done" in r.stdout and "WARNING" not in r.stderr
          and (paths.state.read_bytes(), paths.ledger.read_bytes()) == before,
          (r.returncode, r.stdout[-500:], r.stderr[-800:]))


def main():
    t0 = time.time()
    try:
        check("the test runs on its own INC_DIR and REPO",
              C.INC_DIR == TMP / "inc" and C.REPO == TMP / "repo", (C.INC_DIR, C.REPO))
        core = build_world()
        hits, unhashable = C.NeverTrainGuard.load().check([r["image"] for r in core])
        check("fixture: no train_core image is within %d bits of an exam image" % C.HOLDOUT_NEAR_DUP_BITS,
              not hits and not unhashable, hits[:3])
        test_production_specs(core)
        test_production_specs_full(core)
        st, defn, backend = test_run(core)
        if st["done"]:
            n_accept = test_runs(st, defn, backend)
            verdicts = test_ledger_and_chains(st, defn)
            check("at least one step is an ACCEPT, so the soup path ran end to end (the fixture makes "
                  "step 1 one in both chains: null runs have no species positives)", n_accept > 0, verdicts)
            test_report(st, defn, n_accept)
            test_cli_and_idempotence(backend)
    finally:
        if os.environ.get("INC_KEEP_TMP") == "1":
            print("kept %s" % TMP)
        else:
            shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
