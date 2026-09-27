#!/usr/bin/env python3
"""The INC driver runs the protocol's experiments to the end by rule, and the
pilot build plants exactly the known answers the pilot is scored against.

No cluster and no training: FakeBackend 'runs' each spec through a synthetic
executor that writes run.json, scores/<exam>.json and weights/final.pt the way
inc/train.py does. Its dev scores follow a known model, so every decision is
known in advance:
  * a cand run scores its parent incumbent + the increment's effect (clean
    increments +0.02, Bswap -0.06 on the 12-class score only, Breal 0) + a seed
    offset; a null run scores parent + seed offset;
  * a cold run scores 0.40 + 0.02 per clean increment it holds - 0.06 if it
    holds Bswap, + seed offset; a soup scores the mean of its cands + 0.001
    (- 0.004 in the freeze chain, so choose_soup keeps cand s0 there).
So the gate must ACCEPT I1..I5, REJECT Bswap (blamed on data, attributed to
labels: the agnostic score holds) and HOLD Breal (cand and null tie seed for
seed), and the truth arm must say helps / hurts / neutral.

Pinned:
  * pilot build: P0 is the smallest set of largest sessions reaching 50%;
    sessions are never split; 6 greedy bins sorted by smallest session, bin 3
    the Bswap source; Bswap changes exactly 40% (round half up) of its boxes,
    each to another species, only the class token, reproducibly; Breal joins
    names through species_of (others -> OtherPlant), drops near-copies of
    evaluation images (real dHash, real never-train index) and of train_core
    images, and unlabelled or malformed images, and samples the median clean
    increment size; recipes and truth manifests are the runner doc's; the
    effective warmup (Ultralytics' 100-iteration floor) is recorded; a second
    build refuses; a production build refuses unlocked splits;
  * a full pilot-shaped run (3 chains, 7 steps, truth arm) reaches done: the
    verdicts above; REJECT and HOLD leave the incumbent unchanged; ACCEPT
    runs a soup and moves the incumbent; the next step trains from it; the
    replay samples are disjoint, drawn from the accepted pool and
    reproducible; the ledger's input sha256s match the score files; one
    sbatch array per call; test appears only in final specs, and final runs
    exist only once every chain and the truth arm are complete;
  * replay mode: a definition without replay_mode (pilot_v1's) runs byte for
    byte as replay_mode 'sample' (state, ledger, replay samples, manifests,
    specs, runs), and sample mode draws the R1 / R2 samples and makes the
    runs and submissions the driver made before replay_mode existed (pinned
    key-list digests); a default build writes 'sample'. With --replay-mode full
    the build records the mode and the effective warmup at the mode's cand
    and null sizes (pool_min / pool_max); cand trains on the whole accepted
    pool + D_k and null on the pool, D_k never in it; accepted increments
    join the pool for later steps, rejected and held ones never do; having
    accepted exactly the clean increments, cand and null are the truth arm's
    'with' and 'without' sets; the step state and every gate entry record
    the mode and sizes; no replay sample is drawn (a pool under 2|D_k|, which
    blocks a sample-mode chain, is fine); every spec passes inc/train.py
    with the protocol's recipe; status and the report show the mode and the
    cand / null sizes; an unknown mode, or one on a baseline, is refused;
    with chains that decide differently, each chain's cand / null are its
    own accepted pool (+ D_k), computed from the planted verdicts;
  * gate block (exp.json "gate", protocol v2's net flips option): a
    definition without it, with {} and with {"flips_mode": "negative"} write
    byte for byte the same state, ledger, manifests and specs, and their gate
    decisions, truth details and soup records equal the digests the driver and
    gate gave before the block existed (2 chains x 3 steps with planted
    per-image flips: step A's cands fix more images than they break, step B's
    break more than they fix); with {"flips_mode": "net"} step A is accepted
    where v1 rejects it on flips alone, step B is still rejected, every gate
    entry records the net config and every count, the truth arm is untouched,
    status and the report say so; a gate block's metric reaches the gate,
    soup and truth calls (their values are the score files' map50); init
    refuses an unknown key, require_production, a bad mode, type or range, a
    block that is not an object, a block on a baseline, and min_seeds above
    the seeds, writing nothing; the pilot builder writes the block into
    exp.json and build_summary.json (default negative, net on request) and
    its CLI refuses the flag on build-b0 and build-baseline;
  * retry: a failed run is resubmitted once (the same spec in the same run
    dir), a second failure blocks its chain and only its chain; a task that
    ended without run.json fails at once; a vanished task fails only after
    8 h; every failure branch is counted once however many advances run
    before the retry; a task that ended while attempt.json names another live
    job (a duplicate's exit 3) re-tracks that job instead of failing; a run
    marked failed whose done run.json appears later is recovered and its
    block lifted; a run.json that appears between the first look and squeue
    is collected, not failed;
  * submissions: 'submitting' is saved before sbatch; a pass that dies after
    sbatch, or an sbatch that times out or answers without a job id after
    queueing, is resolved by job name (found -> tracked, Slurm unreachable ->
    left, absent -> requeued only after the grace period); an sbatch that
    never started requeues at once; never two arrays for one run;
  * exclusion: advance is idempotent (a second call submits nothing and
    writes nothing) and returns at once when the flock or the lease is held;
    stale leases (expired, dead pid on this host, unreadable and old) are
    broken; a pass stops without saving or submitting when another pass
    saved state or took its lease; a mount without flock works on the lease;
  * definitions: an increment overlapping the base or another step (key,
    path or image sha256), an edited manifest, fewer than 3 seeds, or a truth
    recipe other than the base's is refused at init, before anything is
    written; the runtime disjointness checks block the chain and the truth
    arm with no run created; an edited manifest blocks at use;
  * code: every ledger entry carries the sha256 of the pinned modules; a pass
    refuses when the code differs from the pins (until 'repin', recorded) or
    from the git-tracked nested copy (unless INC_ALLOW_DRIFT=1, recorded);
  * operations: an unreadable score file is retried, not blocked, until
    TRANSIENT_LIMIT passes; 'unblock' resets the failed runs (ledger);
    'watch' advances to done; a partial last ledger line is repaired by
    advance and skipped by the report; the CLI reports OSError / ValueError
    without a traceback;
  * a production experiment cannot be decided on test-mode scores, and a
    testing experiment refuses to advance without INC_SCORER_TESTING=1;
  * Slurm output parsing and the sbatch command line (%40 concurrency);
  * the report renders decisions, agreement, the final table and GPU-hours;
  * B0 (build-b0) runs base seeds, then final runs on every exam.

Run:  python3 tests/test_inc_driver.py
"""
import collections
import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import pathlib
import shutil
import socket
import statistics
import sys
import tempfile
import time

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_driver_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"            # tests only: the experiments here are testing ones
os.environ.pop("INC_JOB_SCRIPT", None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc import report as R  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc=Exception, contains=None):
    try:
        fn()
    except exc as e:
        if contains and contains not in str(e):
            print("       raised %s without %r: %s" % (type(e).__name__, contains, e))
            return False
        return True
    return False


def keys_of(path):
    return [r["key"] for r in C.read_manifest(path)]


# ------------------------------------------------------------------ the world
SESSION_SIZES = [30, 26, 22, 18, 15, 14, 12, 11, 10, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2]
SLUG_A, SLUG_B = P.BREAL_SLUGS
NAMES_A = ["Ragweed", "corn", "Waterhemp"]          # -> 5, 12, 0
NAMES_B = {"0": "weed"}                             # dict form -> 12
JOIN_A = {0: 5, 1: 12, 2: 0}


def _png(path, rng, size=24):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(path)


def _boxes(rng, n, n_cls):
    return [(int(rng.integers(0, n_cls)), float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8)),
             float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.05, 0.3))) for _ in range(n)]


def _write_label(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for b in boxes:
            fh.write("%d %.7f %.7f %.7f %.7f\n" % b)


def make_world():
    """train_core (24 sessions, real small images), the two Breal sources with
    a registry, and a never-train index holding three planted Breal copies."""
    rng = np.random.default_rng(7)
    root = TMP / "repo"
    order = rng.permutation(len(SESSION_SIZES))     # names do not follow sizes
    rows = []
    for i, n in enumerate(SESSION_SIZES):
        sess = "2021%04d_cam%d" % (int(order[i]) * 7 + 11, i % 3)
        for j in range(n):
            stem = "%s_%d" % (sess, j + 1)
            img = root / "downloads" / "cwd12" / "images" / (stem + ".png")
            lab = root / "downloads" / "cwd12" / "labels" / (stem + ".txt")
            _png(img, rng)
            _write_label(lab, _boxes(rng, int(rng.integers(1, 4)), 12))
            rows.append({"image": str(img), "label": str(lab), "sha256": C.sha256_file(img),
                         "label_sha256": C.sha256_file(lab), "source": "cottonweeddet12/train",
                         "session": sess, "key": "train_core__%s" % stem})
    C.write_manifest(C.manifest_path("train_core"), rows)

    breal = {}
    a_dir = root / "datasets" / SLUG_A
    for j in range(30):
        img = a_dir / "images" / ("wcd_%03d.png" % j)
        _png(img, rng)
        if j == 29:
            continue                                # no label file
        lab = a_dir / "labels" / ("wcd_%03d.txt" % j)
        if j == 28:
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("0 0.5 0.5 0.2\n")       # 4 columns
            continue
        _write_label(lab, _boxes(rng, int(rng.integers(1, 3)), 3))
        breal[str(img)] = lab
    # a byte-identical copy of a train_core photograph under another name
    dup = a_dir / "images" / "wcd_dup.png"
    shutil.copyfile(rows[5]["image"], dup)
    (a_dir / "labels" / "wcd_dup.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    b_dir = root / "datasets" / SLUG_B
    for j in range(20):
        img = b_dir / "train" / "images" / ("aer_%03d.png" % j)
        _png(img, rng)
        lab = b_dir / "train" / "labels" / ("aer_%03d.txt" % j)
        _write_label(lab, _boxes(rng, 1, 1))
        breal[str(img)] = lab
    reg = root / "results" / "framework" / "dataset_registry.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps({"datasets": {SLUG_A: {"class_names": NAMES_A, "annotation": "bbox"},
                                            SLUG_B: {"class_names": NAMES_B, "annotation": "bbox"}}}))

    planted = [str(a_dir / "images" / "wcd_000.png"), str(b_dir / "train" / "images" / "aer_005.png"),
               str(a_dir / "images" / "wcd_010.png")]
    entries = [[C.dhash(planted[0]), "dev", "dev__x0"], [C.dhash(planted[1]), "ood22", "ood22__x1"],
               [C.dhash(planted[2]) ^ 0b1001, "test", "test__x2"]]      # 2 bits away
    entries += [[int.from_bytes(rng.bytes(8), "big"), "dev", "dev__r%d" % i] for i in range(30)]
    C.NEVER_TRAIN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(C.NEVER_TRAIN_INDEX, "w") as fh:
        json.dump({"entries": entries, "min_expected": len(entries), "complete": True}, fh)
    return {"rows": rows, "breal_labels": breal, "planted": planted, "core_copy": str(dup)}


# ------------------------------------------------------ synthetic executor
CLEAN = P.CLEAN
EFFECT = {"I1": 0.02, "I2": 0.02, "I3": 0.02, "I4": 0.02, "I5": 0.02, "Bswap": -0.06, "Breal": 0.0}
NOISE = {0: 0.0, 1: 0.002, 2: -0.002}
FACTOR = {"dev": 1.0, "ood22": 0.7, "ood23": 0.6, "imageweeds": 0.5, "test": 0.95}
SECONDS = {"base": 7200.0, "union": 7200.0, "cand": 1800.0, "null": 1800.0, "soup": 60.0,
           "final": 300.0}
SPECIES = G.SPECIES


class Executor:
    """What inc/train.py writes, with dev scores from the known model above."""

    def __init__(self, inc_keys=None):
        self.inc_keys = inc_keys or {}
        self.attempts = collections.Counter()
        self.fail_once, self.fail_always, self.lose_once, self.end_failed_once = set(), set(), set(), set()
        self.soup_penalty = {"freeze": -0.004}
        self.seen = []

    def __call__(self, spec):
        rid = spec["run_id"]
        out = pathlib.Path(spec["out_dir"])
        self.attempts[rid] += 1
        n = self.attempts[rid]
        self.seen.append(spec)
        if (out / "run.json").exists():               # inc/train.py: removed when an attempt starts
            (out / "run.json").unlink()
        if rid in self.lose_once and n == 1:
            return "LOST"
        if rid in self.end_failed_once and n == 1:
            return "FAILED"                           # killed before writing run.json
        if rid in self.fail_always or (rid in self.fail_once and n == 1):
            self._run_json(out, {"status": "failed", "attempt": n, "seconds": 5.0,
                                 "error": "Traceback (most recent call last): synthetic failure of %s" % rid})
            return "FAILED"
        w = out / "weights" / "final.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "final":
            if w.is_symlink() or w.exists():
                w.unlink()
            os.symlink(spec["init"], w)
        else:
            w.write_bytes(("weights of %s attempt %d" % (rid, n)).encode())
        for exam in spec["exams"]:
            v, a = self.values(spec, exam)
            self._score(out / "scores" / ("%s.json" % exam), exam, v, a, w)
        self._run_json(out, {"status": "done", "attempt": n, "seconds": SECONDS[spec["kind"]],
                             "error": None, "weights_sha256": C.sha256_file(w)})
        return "COMPLETED"

    @staticmethod
    def _run_json(out, obj):
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps(obj))

    @staticmethod
    def parent(weights):
        s = json.loads((pathlib.Path(weights).parent.parent / "scores" / "dev.json").read_text())
        return s["map50_95"], s["agnostic_map50_95"]

    def values(self, spec, exam):
        kind, rid = spec["kind"], spec["run_id"]
        if kind in ("base", "union"):
            keys = set(keys_of(spec["train_manifest"]))
            n_clean = sum(1 for i in CLEAN if i in self.inc_keys and self.inc_keys[i] <= keys)
            bswap = any(k.startswith("Bswap__") for k in keys)
            s = NOISE[spec["recipe"]["seed"]]
            return 0.40 + 0.02 * n_clean - 0.06 * bswap + s, 0.60 + 0.02 * n_clean + s
        if kind in ("cand", "null"):
            pv, pa = self.parent(spec["init"])
            name = rid.split("__")[1].split("_", 1)[1]
            eff = EFFECT.get(name, 0.02) if kind == "cand" else 0.0
            s = NOISE[spec["recipe"]["seed"]]
            return pv + eff + s, pa + max(eff, 0.0) + s
        if kind == "soup":
            vals = [self.parent(w) for w in spec["soup_of"]]
            pen = self.soup_penalty.get(rid.split("__")[0], 0.001)
            return (statistics.fmean(v for v, _ in vals) + pen,
                    statistics.fmean(a for _, a in vals) + pen)
        pv, pa = self.parent(spec["init"])            # final
        return pv * FACTOR[exam], pa * FACTOR[exam]

    @staticmethod
    def _score(path, exam, v, a, weights):
        other = 0 if exam in ("dev", "test") else 25
        n_gt = {s: 40 for s in SPECIES}
        n_gt["OtherPlant"] = other
        per_class = {s: v for s in SPECIES}
        if other:
            per_class["OtherPlant"] = 0.5 * v
        s = {"exam": exam, "scorer_sha256": "TEST-" + "5" * 64, "production": False,
             "deviations": ["LOCK.json not checked"],
             "manifest_sha256": hashlib.sha256(("m/" + exam).encode()).hexdigest(),
             "key_order_sha256": hashlib.sha256(("k/" + exam).encode()).hexdigest(),
             "weights_sha256": C.sha256_file(weights), "n_images": 20,
             "map50_95": v, "map50": min(1.0, v + 0.2), "agnostic_map50_95": a,
             "agnostic_map50": min(1.0, a + 0.2), "per_class": per_class, "n_gt": n_gt,
             "image_correct": "1" * 20, "species_map50_95": v, "species_map50": v + 0.2}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s))


class Clock:
    def __init__(self):
        self.t = 1.8e9

    def __call__(self):
        return self.t


def state_of(exp):
    return json.loads(D.Paths(exp).state.read_text())


def ledger_of(exp):
    p = D.Paths(exp).ledger
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def all_specs(exp):
    runs = D.Paths(exp).runs
    return [json.loads((runs / d / "spec.json").read_text()) for d in sorted(os.listdir(runs))]


def small_defn(exp, rows, step_names=("A", "B"), recipes=("full",), truth=False, testing=True,
               base_n=60, step_n=10):
    """A small chain definition over train_core rows (base, then steps)."""
    mdir = TMP / "small" / exp
    base_rows = rows[:base_n]
    entries = []
    for i, name in enumerate(step_names):
        rs = rows[base_n + i * step_n: base_n + (i + 1) * step_n]
        p = mdir / ("%s.jsonl" % name)
        entries.append({"name": name, "manifest": str(p), "manifest_sha256": C.write_manifest(p, rs),
                        "n_images": len(rs), "clean": True})
    bp = mdir / "base.jsonl"
    return {"exp": exp, "type": "chain", "testing": testing, "seeds": [0, 1, 2],
            "init_weights": "yolo11n.pt", "decision_exam": "dev", "final_exams": list(D.FINAL_EXAMS),
            "base": {"name": "base", "manifest": str(bp), "manifest_sha256": C.write_manifest(bp, base_rows),
                     "n_images": len(base_rows), "recipe": P.cold_recipe()},
            "steps": entries, "recipes": {r: P.inc_recipes()[r] for r in recipes},
            "truth": truth, "truth_recipe": P.cold_recipe()}


def drive(exp, fb, clock, executor, max_iter=200, on_state=None, hold=None):
    """Run pending tasks (except held ones), advance, repeat until done or
    nothing moves."""
    calls = []
    for _ in range(max_iter):
        ran = fb.run_pending(executor, hold=hold)
        before = len(fb.submissions)
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        calls.append(len(fb.submissions) - before)
        st = state_of(exp)
        if on_state:
            on_state(st)
        if st["done"] or (ran == 0 and len(fb.submissions) == before):
            break
    return calls


# ------------------------------------------------------------------ tests
def test_units():
    print("unit: replay samples, Slurm parsing, spec rules")
    rows = [{"key": "k%03d" % i, "image": "/i/%d.png" % i} for i in range(50)]
    r1, r2 = D.replay_samples(rows, 10, "e/full/3")
    k1, k2 = {r["key"] for r in r1}, {r["key"] for r in r2}
    check("replay: two samples of n, disjoint", len(k1) == 10 and len(k2) == 10 and not k1 & k2)
    a1, a2 = D.replay_samples(list(reversed(rows)), 10, "e/full/3")
    check("replay: deterministic and independent of the pool's order",
          [r["key"] for r in a1] == [r["key"] for r in r1] and [r["key"] for r in a2] == [r["key"] for r in r2])
    b1, _ = D.replay_samples(rows, 10, "e/full/4")
    check("replay: another step draws another sample", {r["key"] for r in b1} != k1)
    check("replay: a pool under 2n raises", raises(lambda: D.replay_samples(rows, 26, "x"), D.DriverError))
    check("replay: seeded by stable_int, as documented",
          [r["key"] for r in r1] == [sorted(rows, key=lambda r: r["key"])[i]["key"] for i in
                                     sorted(int(x) for x in np.random.default_rng(
                                         C.stable_int("e/full/3")).permutation(50)[:10])])

    check("sbatch --parsable with a cluster name", D.parse_parsable("12345;bridges2\n") == "12345")
    check("sbatch output that is not a job id raises",
          raises(lambda: D.parse_parsable("Submitted batch job 1"), D.DriverError))
    check("array ranges expand", D.expand_task_ids("77_[4-6,9%40]") == {"77_4", "77_5", "77_6", "77_9"})
    check("squeue -r lines", D.parse_squeue("77_1\n77_[3-4]\n\n78\n") == {"77_1", "77_3", "77_4", "78"})
    check("sacct: single array tasks with their state, pending ranges skipped",
          D.parse_sacct("77_4|FAILED\n77_5|CANCELLED by 42\n77_[6-9]|PENDING\n77_7|RUNNING\n77_8|COMPLETED\n")
          == {"77_4": "FAILED", "77_5": "CANCELLED", "77_7": "RUNNING", "77_8": "COMPLETED"})
    sb = D.SlurmBackend(script="/x/run_inc_job.sh", user="u")
    argv = sb.sbatch_argv("/l/0003.txt", 10, "pilot_v1", "inc_pilot_v1_0003", "/logs")
    check("sbatch: one array with %40 concurrency, the list file and the exp",
          argv[:3] == ["sbatch", "--parsable", "--array=0-9%40"] and argv[-3:] == ["/x/run_inc_job.sh",
                                                                                 "/l/0003.txt", "pilot_v1"]
          and not any(a.startswith("--time") for a in argv), argv)
    argv = sb.sbatch_argv("/l/0003.txt", 10, "pilot_v1", "inc_pilot_v1_0003", "/logs", "08:00:00")
    check("sbatch: --time only when asked for", "--time=08:00:00" in argv and argv[-3] == "/x/run_inc_job.sh")
    check("spec: a soup takes no init, every other kind needs one",
          raises(lambda: D.validate_spec({"exp": "e", "run_id": "s", "kind": "soup", "init": "/a",
                                          "soup_of": ["/a", "/b"], "exams": ["dev"], "out_dir": "/o"}),
                 D.DriverError, "init")
          and raises(lambda: D.validate_spec({"exp": "e", "run_id": "f", "kind": "final",
                                              "exams": ["dev"], "out_dir": "/o"}), D.DriverError, "init"))
    saved = dict(os.environ)
    try:
        os.environ.update(SLURM_JOB_ID="5", SLURM_MEM_PER_NODE="45G", SBATCH_PARTITION="x",
                          SLURM_CONF="/etc/slurm.conf")
        env_p, env_t = D.submission_env(False), D.submission_env(True)
        check("sbatch env: an enclosing job's SLURM_/SBATCH_ vars are dropped, SLURM_CONF kept",
              "SLURM_JOB_ID" not in env_p and "SLURM_MEM_PER_NODE" not in env_p
              and "SBATCH_PARTITION" not in env_p and env_p.get("SLURM_CONF") == "/etc/slurm.conf")
        check("sbatch env: INC_SCORER_TESTING reaches jobs of a testing experiment only",
              "INC_SCORER_TESTING" not in env_p and env_t.get("INC_SCORER_TESTING") == "1")
    finally:
        os.environ.clear()
        os.environ.update(saved)

    spec = {"exp": "e", "run_id": "full__s01_A__cand__s0", "kind": "cand", "init": "/w.pt",
            "train_manifest": "/m.jsonl", "recipe": dict(P.inc_recipes()["full"], seed=0),
            "exams": ["dev"], "out_dir": "/o"}
    check("spec: a cand spec validates", D.validate_spec(dict(spec)) is not None)
    check("spec: test outside a final spec is refused",
          raises(lambda: D.validate_spec(dict(spec, exams=["dev", "test"])), D.DriverError, "final"))
    check("spec: unknown keys are refused",
          raises(lambda: D.validate_spec(dict(spec, attempt=2)), D.DriverError, "unknown"))
    check("spec: optimizer auto is refused",
          raises(lambda: D.validate_spec(dict(spec, recipe=dict(spec["recipe"], optimizer="auto"))),
                 D.DriverError, "explicit"))
    rec = P.inc_recipes()
    check("recipes: the runner doc's full / freeze / lora",
          rec["full"]["epochs"] == 30 and rec["full"]["lr0"] == 0.002 and rec["full"]["warmup_bias_lr"] == 0.002
          and rec["full"]["optimizer"] == "SGD" and rec["full"]["warmup_epochs"] == 1
          and rec["full"]["lrf"] == 0.01 and rec["full"]["cos_lr"] is True
          and rec["freeze"]["freeze"] == 11 and rec["freeze"]["trainer"] == "freeze"
          and rec["lora"]["lr0"] == 0.01 and rec["lora"]["warmup_bias_lr"] == 0.01
          and rec["lora"]["lora"] == {"rank": 16, "alpha": 32} and rec["lora"]["trainer"] == "lora"
          and rec["lora"]["freeze"] is None)
    cold = P.cold_recipe()
    check("recipes: the protocol's cold run",
          cold["epochs"] == 100 and cold["optimizer"] == "SGD" and cold["lr0"] == 0.01
          and cold["lrf"] == 0.01 and cold["warmup_epochs"] == 3 and cold["cos_lr"] is True
          and cold["imgsz"] == 640 and cold["batch"] == 32 and cold["deterministic"] is True)


def greedy_reference(sizes, sessions, n_bins=6):
    bins, tot = [[] for _ in range(n_bins)], [0] * n_bins
    for s in sorted(sessions, key=lambda s: (-sizes[s], s)):
        i = min(range(n_bins), key=lambda j: (tot[j], j))
        bins[i].append(s)
        tot[i] += sizes[s]
    return sorted((sorted(b) for b in bins), key=lambda b: b[0])


def test_pilot_build(world):
    print("pilot build")
    exp = "pilot_t"
    fb = FakeBackendRecorder()
    summary, defn, _ = P.build_pilot(exp, testing=True, backend=fb, quiet=True)
    paths = D.Paths(exp)
    rows = world["rows"]
    sizes = collections.Counter(r["session"] for r in rows)
    total = len(rows)

    p0 = summary["p0"]["sessions"]
    order = sorted(sizes, key=lambda s: (-sizes[s], s))
    n_p0 = sum(sizes[s] for s in p0)
    check("P0: the largest sessions, in order", p0 == order[:len(p0)], p0)
    check("choose_p0: exactly 50% is enough; ties by name",
          P.choose_p0({"b": 5, "a": 3, "c": 2}) == ["b"]
          and P.choose_p0({"y": 4, "x": 4, "z": 2}) == ["x", "y"]
          and P.choose_p0({"y": 4, "x": 4, "z": 1}) == ["x", "y"])
    check("P0: >= 50% of train_core, and the last session was needed",
          2 * n_p0 >= total and 2 * (n_p0 - sizes[p0[-1]]) < total, (n_p0, total))
    check("P0 manifest = its sessions' rows", set(keys_of(paths.manifests / "P0.jsonl"))
          == {r["key"] for r in rows if r["session"] in set(p0)})

    rest = [s for s in sizes if s not in set(p0)]
    bins = [b["sessions"] for b in summary["bins"]]
    check("bins: 6, greedy (independent re-implementation), sorted by smallest session",
          bins == greedy_reference(sizes, rest) and len(bins) == 6
          and [b[0] for b in bins] == sorted(b[0] for b in bins))
    check("bins: every remaining session in exactly one bin",
          sorted(s for b in bins for s in b) == sorted(rest))
    roles = [b["role"] for b in summary["bins"]]
    check("bin 3 is the Bswap source, the others I1..I5 in order",
          roles == ["I1", "I2", "I3", "Bswap_source", "I4", "I5"], roles)
    mean = sum(b["images"] for b in summary["bins"]) / 6.0
    check("bins: sizes recorded and flagged exactly outside +-35%",
          all(b["flagged"] == (abs(b["images"] - mean) > 0.35 * mean) for b in summary["bins"]))
    for i, name in ((0, "I1"), (4, "I4")):
        check("%s = bin %d's sessions, whole" % (name, i),
              set(keys_of(paths.manifests / ("%s.jsonl" % name)))
              == {r["key"] for r in rows if r["session"] in set(bins[i])})
    session_home = collections.defaultdict(set)
    for name in ("P0",) + CLEAN:
        for r in C.read_manifest(paths.manifests / ("%s.jsonl" % name)):
            session_home[r["session"]].add(name)
    check("no session is split between parts", all(len(v) == 1 for v in session_home.values()))

    # Bswap
    src = {r["key"]: r for r in rows if r["session"] in set(bins[3])}
    bs_rows = C.read_manifest(paths.manifests / "Bswap.jsonl")
    n_boxes = sum(len(C.read_yolo(r["label"])) for r in src.values())
    want = int(math.floor(0.4 * n_boxes + 0.5))
    changed, other_diff, wrong_to, byte_changes = 0, 0, 0, 0
    for r in bs_rows:
        old = open(src[r["key"][len("Bswap__"):]]["label"]).read().splitlines()
        new = open(r["label"]).read().splitlines()
        check_len = len(old) == len(new)
        for a, b in zip(old, new):
            if a != b:
                byte_changes += 1
                ta, tb = a.split(), b.split()
                if ta[1:] != tb[1:]:
                    other_diff += 1
                if int(tb[0]) == int(ta[0]) or not 0 <= int(tb[0]) < 12:
                    wrong_to += 1
                changed += 1
        if not check_len:
            other_diff += 1
    ch = json.loads((paths.root / "labels" / "Bswap_changes.json").read_text())
    check("Bswap: exactly 40%% of the boxes changed (%d of %d)" % (want, n_boxes),
          changed == want == summary["bswap"]["changed"] == len(ch["changes"]), (changed, want))
    check("Bswap: every change to a different species (0-11), only the class token",
          wrong_to == 0 and other_diff == 0 and all(c["to"] != c["from"] for c in ch["changes"]))
    check("Bswap: labels written under INC_DIR/<exp>/labels/Bswap, hashed in the manifest",
          all(pathlib.Path(r["label"]).parent == paths.root / "labels" / "Bswap"
              and C.sha256_file(r["label"]) == r["label_sha256"] and r["source"] == "Bswap"
              for r in bs_rows) and len(bs_rows) == len(src))
    again, info = P.make_bswap(exp, list(src.values()), TMP / "bswap_again")
    other, info2 = P.make_bswap("another_exp", list(src.values()), TMP / "bswap_other")
    check("Bswap: reproducible from the exp name, different for another name",
          info["changes"] == ch["changes"] and info2["changes"] != ch["changes"])

    # Breal
    br = C.read_manifest(paths.manifests / "Breal.jsonl")
    clean_sizes = [len(keys_of(paths.manifests / ("%s.jsonl" % n))) for n in CLEAN]
    check("Breal: sampled to the median clean increment size",
          len(br) == int(statistics.median(clean_sizes)) == summary["breal"]["n"], (len(br), clean_sizes))
    brs = summary["breal"]
    check("Breal: the three planted near-copies of evaluation images are dropped and counted",
          brs["dropped_near_eval"] == 3 and not set(world["planted"]) & {r["image"] for r in br}
          and brs["after_guard"] == brs["candidates"] - 4, brs["dropped_near_eval"])
    check("Breal: a byte copy of a train_core photograph is dropped and counted (near train_core)",
          brs["dropped_near_train_core"] == 1 and brs["slugs"][SLUG_A]["dropped"].get("near_train_core") == 1
          and world["core_copy"] not in {r["image"] for r in br}
          and brs["near_train_core_examples"][0]["train_core_key"] == world["rows"][5]["key"]
          and brs["near_train_core_examples"][0]["bits"] == 0, brs.get("near_train_core_examples"))
    check("Breal: unlabelled and malformed images dropped and counted",
          brs["slugs"][SLUG_A]["dropped"].get("no_label") == 1
          and brs["slugs"][SLUG_A]["dropped"].get("bad_label_columns") == 1)
    check("Breal: class-order assumption recorded as unverified",
          brs["class_order_verified"] is False and "unverified" in brs["class_order_assumption"])
    ok_join, n_a = True, 0
    for r in br:
        orig = C.read_yolo(world["breal_labels"][r["image"]])
        got = C.read_yolo(r["label"])
        if r["source"] == SLUG_A:
            n_a += 1
            want_boxes = [(JOIN_A[b[0]],) + b[1:] for b in orig]
        else:
            want_boxes = [(12,) + b[1:] for b in orig]
        if len(got) != len(want_boxes) or any(
                g[0] != w[0] or max(abs(x - y) for x, y in zip(g[1:], w[1:])) > 1e-6
                for g, w in zip(got, want_boxes)):
            ok_join = False
    check("Breal: names joined through species_of (Ragweed 5, Waterhemp 0, others OtherPlant)",
          ok_join and n_a > 0 and brs["slugs"][SLUG_A]["join"] == {"Ragweed": "Ragweed",
                                                                  "corn": "OtherPlant",
                                                                  "Waterhemp": "Waterhemp"})
    check("Breal: rows hash their images and labels",
          all(C.sha256_file(r["image"]) == r["sha256"] and C.sha256_file(r["label"]) == r["label_sha256"]
              for r in br))

    # the definition and the init
    ex = json.loads(paths.exp_json.read_text())
    check("exp.json: sequence, clean flags, recipes, testing",
          [s["name"] for s in ex["steps"]] == ["I1", "I2", "Bswap", "I3", "Breal", "I4", "I5"]
          and [s["clean"] for s in ex["steps"]] == [True, True, False, True, False, True, True]
          and ex["recipes"] == P.inc_recipes() and ex["base"]["recipe"] == P.cold_recipe()
          and ex["truth_recipe"] == P.cold_recipe() and ex["testing"] is True and ex["truth"] is True)
    wu = summary["warmup"]
    n_i1 = len(keys_of(paths.manifests / "I1.jsonl"))
    nb = math.ceil(2 * n_i1 / 32.0)
    want_wu = {"images": 2 * n_i1, "iterations_per_epoch": nb, "warmup_iterations": max(round(nb), 100),
               "warmup_epochs_nominal": 1, "warmup_epochs_effective": round(min(max(round(nb), 100), 30 * nb)
                                                                           / float(nb), 2),
               "epochs": 30, "covers_whole_run": max(round(nb), 100) >= 30 * nb}
    real = P.effective_warmup(P.inc_recipes()["full"], 472)
    check("warmup: the effective warmup (100-iteration floor) is recorded per run type, in exp.json too",
          wu["incremental"]["full"]["s01_I1"] == want_wu and ex["effective_warmup"] == wu
          and wu["cold"]["base"]["images"] == n_p0 and len(wu["cold"]["truth"]) == 7
          and real["iterations_per_epoch"] == 15 and real["warmup_iterations"] == 100
          and real["warmup_epochs_effective"] == 6.67, (wu["incremental"]["full"]["s01_I1"], want_wu, real))
    check("attribution steps 4-5 recorded as out of scope in exp.json",
          set(ex["attribution_scope"]["not_run"]) == {"4", "5"})
    check("build summary: sizes per increment and per species",
          set(summary["increments"]) == set(P.SEQUENCE)
          and all(set(v["boxes"]) == set(C.CLASS_NAMES) for v in summary["increments"].values())
          and (paths.root / "build_summary.json").is_file())
    st = state_of(exp)
    check("init: base (3) and the whole truth arm (7 x 3) submitted as one array",
          len(fb.submissions) == 1 and fb.submissions[0]["n"] == 24
          and sum(1 for r in st["runs"].values() if r["owner"] == "truth") == 21
          and all(r["status"] == "submitted" for r in st["runs"].values()), len(fb.submissions))
    tw = D.Paths(exp).manifests / "truth"
    t_i3 = set(keys_of(tw / "s04_I3_with.jsonl"))
    want_i3 = set().union(*(set(keys_of(paths.manifests / ("%s.jsonl" % n))) for n in ("P0", "I1", "I2", "I3")))
    t_i4 = set(keys_of(tw / "s06_I4_with.jsonl"))
    want_i4 = want_i3 | set(keys_of(paths.manifests / "I4.jsonl"))
    t_bs = set(keys_of(tw / "s03_Bswap_with.jsonl"))
    check("truth: T grows only by clean increments (I3 without Bswap, I4 without Breal)",
          t_i3 == want_i3 and t_i4 == want_i4
          and t_bs == (want_i3 - set(keys_of(paths.manifests / "I3.jsonl"))) | set(keys_of(paths.manifests / "Bswap.jsonl")))
    check("truth: 'without' is the last clean step's 'with', or the base runs",
          st["truth"]["steps"]["1"]["without"] == ["base__s0", "base__s1", "base__s2"]
          and st["truth"]["steps"]["4"]["without"] == st["truth"]["steps"]["2"]["with"]
          and st["truth"]["steps"]["6"]["without"] == st["truth"]["steps"]["4"]["with"])
    check("a second build under the same name refuses",
          raises(lambda: P.build_pilot(exp, testing=True, backend=fb, quiet=True), P.PilotError, "already built"))
    saved = os.environ.pop("INC_SCORER_TESTING")
    try:
        check("--testing without INC_SCORER_TESTING=1 refuses",
              raises(lambda: P.build_pilot("pilot_t2", testing=True, backend=fb, quiet=True), P.PilotError))
    finally:
        os.environ["INC_SCORER_TESTING"] = saved
    check("a production build refuses unlocked splits and writes no definition",
          raises(lambda: P.build_pilot("pilot_prod", testing=False, backend=fb, quiet=True), P.PilotError,
                 "LOCK.json")
          and raises(lambda: P.build_b0("b0_prod", testing=False, backend=fb, quiet=True), P.PilotError,
                     "LOCK.json")
          and not D.Paths("pilot_prod").exp_json.exists() and not D.Paths("b0_prod").exp_json.exists())
    # a build that stops before driver init leaves no definition and may be rebuilt
    s2, _, none = P.build_pilot("pilot_t3", testing=True, backend=fb, quiet=True, init=False)
    first = (D.Paths("pilot_t3").manifests / "Bswap.jsonl").read_bytes()
    s3, _, _ = P.build_pilot("pilot_t3", testing=True, backend=fb, quiet=True, init=False)
    check("an interrupted build (no exp.json) is rebuilt identically",
          none is None and not D.Paths("pilot_t3").exp_json.exists()
          and (D.Paths("pilot_t3").manifests / "Bswap.jsonl").read_bytes() == first
          and s2["increments"] == s3["increments"] and len(fb.submissions) == 1)
    inc_keys = {n: set(keys_of(paths.manifests / ("%s.jsonl" % n))) for n in CLEAN}
    return exp, fb, inc_keys


class FakeBackendRecorder(D.FakeBackend):
    pass


def test_full_pilot(exp, fb, inc_keys):
    print("full pilot-shaped run to done")
    ex = Executor(inc_keys)
    ex.fail_once.add("lora__s01_I1__null__s2")
    clock = Clock()
    violations = []

    def invariant(st):
        finals = [r for r in st["runs"].values() if r["kind"] == "final"]
        if finals:
            chains_done = all(c["phase"] == "done" for c in st["chains"].values())
            truth_done = all(s["decision"] for s in st["truth"]["steps"].values())
            if not (chains_done and truth_done):
                violations.append("final runs before completion")

    # the last truth step lags behind every chain: final runs must wait for it
    calls = drive(exp, fb, clock, ex, on_state=invariant,
                  hold=lambda spec: spec["run_id"].startswith("truth__s07_"))
    st = state_of(exp)
    check("with the truth arm unfinished, every chain can finish but no final run is created",
          all(c["phase"] == "done" for c in st["chains"].values())
          and st["truth"]["steps"]["7"]["decision"] is None and not st["final"]["created"]
          and not [r for r in st["runs"].values() if r["kind"] == "final"])
    calls += drive(exp, fb, clock, ex, on_state=invariant)
    st = state_of(exp)
    check("the experiment reaches done", st["done"] is True, D.Driver(exp).status())
    check("never more than one sbatch array per advance call", max(calls) <= 1, calls)
    check("final runs exist only after every chain and the truth arm are complete", not violations, violations)
    specs = all_specs(exp)
    check("test is listed only by final specs",
          all(("test" in s["exams"]) == (s["kind"] == "final") for s in specs)
          and any(s["kind"] == "final" for s in specs))
    check("the executor never ran a non-final spec on test",
          not [s for s in ex.seen if "test" in s["exams"] and s["kind"] != "final"])

    want = {"I1": "ACCEPT", "I2": "ACCEPT", "Bswap": "REJECT", "I3": "ACCEPT", "Breal": "HOLD",
            "I4": "ACCEPT", "I5": "ACCEPT"}
    for r, chn in st["chains"].items():
        got = {s["name"]: s["decision"]["verdict"] for s in chn["steps"].values()}
        check("chain %s: verdicts ACCEPT x5, REJECT Bswap, HOLD Breal" % r, got == want, got)
        check("chain %s: pools (accepted / neutral / quarantined)" % r,
              chn["accepted"] == list(CLEAN) and chn["neutral"] == ["Breal"] and chn["quarantined"] == ["Bswap"])
        s3, s5 = chn["steps"]["3"], chn["steps"]["5"]
        check("chain %s: REJECT and HOLD keep the incumbent" % r,
              s3["incumbent_after"] == s3["incumbent_before"] == chn["steps"]["2"]["incumbent_after"]
              and s5["incumbent_after"] == s5["incumbent_before"] and s3["soup"] is None and s5["soup"] is None)
        check("chain %s: Bswap blamed on data and attributed to labels" % r,
              s3["decision"]["blame"] == "data" and s3["decision"]["class_vs_loc"] == "labels")
        want_inc = "%s__s01_I1__%s" % (r, "cand__s0" if r == "freeze" else "soup")
        check("chain %s: ACCEPT -> soup -> choose_soup -> new incumbent (%s)" % (r, want_inc),
              chn["steps"]["1"]["incumbent_after"]["run_id"] == want_inc
              and chn["steps"]["1"]["incumbent_before"]["run_id"] == "base__s0"
              and chn["steps"]["1"]["soup"] == "%s__s01_I1__soup" % r)
        spec2 = json.loads(D.Paths(exp).spec("%s__s02_I2__cand__s1" % r).read_text())
        check("chain %s: step 2 trains from the new incumbent's weights" % r,
              spec2["init"] == chn["steps"]["1"]["incumbent_after"]["weights"]
              and spec2["recipe"]["seed"] == 1 and spec2["recipe"]["trainer"] == P.inc_recipes()[r]["trainer"])
        # replay samples of step 4 (I3): pool = P0 + I1 + I2 (Bswap rejected)
        s4 = chn["steps"]["4"]
        k1 = set(keys_of(s4["manifests"]["R1"]["path"]))
        k2 = set(keys_of(s4["manifests"]["R2"]["path"]))
        pool_rows = (C.read_manifest(D.Paths(exp).manifests / "P0.jsonl")
                     + C.read_manifest(D.Paths(exp).manifests / "I1.jsonl")
                     + C.read_manifest(D.Paths(exp).manifests / "I2.jsonl"))
        pool = {x["key"] for x in pool_rows}
        e1, e2 = D.replay_samples(pool_rows, len(inc_keys["I3"]), "%s/%s/4" % (exp, r))
        check("chain %s: replay R1, R2 disjoint, |D_k| each, from the accepted pool, reproducible" % r,
              not k1 & k2 and len(k1) == len(k2) == len(inc_keys["I3"]) and k1 | k2 <= pool
              and k1 == {x["key"] for x in e1} and k2 == {x["key"] for x in e2})
        check("chain %s: cand = D_k + R1, null = R1 + R2" % r,
              set(keys_of(s4["manifests"]["cand"]["path"])) == inc_keys["I3"] | k1
              and set(keys_of(s4["manifests"]["null"]["path"])) == k1 | k2)
    check("the chains draw different replay samples (seeded by recipe)",
          keys_of(st["chains"]["full"]["steps"]["4"]["manifests"]["R1"]["path"])
          != keys_of(st["chains"]["lora"]["steps"]["4"]["manifests"]["R1"]["path"]))

    truth = {s["name"]: s["decision"]["verdict"] for s in st["truth"]["steps"].values()}
    check("truth arm: helps x5, hurts Bswap, neutral Breal",
          truth == {"I1": "helps", "I2": "helps", "Bswap": "hurts", "I3": "helps", "Breal": "neutral",
                    "I4": "helps", "I5": "helps"}, truth)

    r = st["runs"]["lora__s01_I1__null__s2"]
    check("a failed run is resubmitted once and then completes (attempt 2)",
          r["status"] == "complete" and r["attempt"] == 2 and len(r["history"]) == 1
          and "synthetic failure" in r["history"][0]["error"] and ex.attempts["lora__s01_I1__null__s2"] == 2)
    check("the retry is the same spec in the same run dir; the failure's run.json sha256 and "
          "seconds are kept in state",
          [x["out_dir"] for x in ex.seen if x["run_id"] == "lora__s01_I1__null__s2"]
          == [str(D.Paths(exp).run_dir("lora__s01_I1__null__s2"))] * 2
          and r["history"][0]["seconds"] == 5.0 and len(r["history"][0]["run_json_sha256"]) == 64)
    soups = [s for s in specs if s["kind"] == "soup"]
    check("soup specs carry soup_of (the three cand weights) and no init, as inc/train.py requires",
          len(soups) == 15 and all("init" not in s and len(s["soup_of"]) == 3 and "recipe" not in s
                                   for s in soups))
    check("an array holding cold runs gets the longer time limit, a cheap-only array the script's",
          fb.submissions[0]["time_limit"] == D.COLD_TIME_LIMIT
          and any(sub["time_limit"] is None for sub in fb.submissions))
    try:
        from weed_optimizer_framework.tools.inc import train as TR
    except ImportError as e:
        TR = None
        print("  skip inc/train.py cross-checks: %s" % e)
    if TR is not None:
        bad = []
        for s in specs:
            try:
                TR.validate_spec(s, D.Paths(exp).spec(s["run_id"]))
            except Exception as e:  # noqa: BLE001 -- any refusal is a failure here
                bad.append("%s: %s" % (s["run_id"], e))
        check("inc/train.py accepts every spec the driver wrote (%d specs)" % len(specs), not bad, bad[:3])
        settings, _ = TR.testing_settings(exp)
        check("inc/train.py reads the experiment as testing from exp.json", settings == {})

    full_ledger = ledger_of(exp)
    ledger = [e for e in full_ledger if e["type"] in ("gate", "soup", "truth")]
    types = collections.Counter(e["type"] for e in full_ledger)
    check("ledger: 21 gate, 15 soup and 7 truth entries (and the code and gate pins), no duplicates",
          types == {"gate": 21, "soup": 15, "truth": 7, "code_pin": 1, "gate_pin": 1}
          and len({e["id"] for e in full_ledger}) == len(full_ledger), types)
    sha_ok = all(C.sha256_file(i["path"]) == i["sha256"]
                 for e in ledger for grp in e["inputs"].values()
                 for i in (grp if isinstance(grp, list) else [grp]))
    check("ledger: every input score path is recorded with its sha256", sha_ok)
    check("ledger: every entry says testing", all(e["testing"] is True for e in full_ledger))
    gate_sha = C.sha256_file(pathlib.Path(D.__file__).parent / "gate.py")
    drv_sha = C.sha256_file(D.__file__)
    check("ledger: every entry records the sha256 of the gate and the driver that decided it",
          all(e["code"]["modules"]["tools/inc/gate.py"] == gate_sha
              and e["code"]["modules"]["tools/inc/driver.py"] == drv_sha
              and set(e["code"]["modules"]) == set(D.PINNED_MODULES) and e["code"]["drift"] == []
              for e in full_ledger)
          and st["code"]["modules"]["tools/inc/gate.py"] == gate_sha)
    breal_gate = [e for e in ledger if e["id"] == "gate/full/5"][0]
    check("ledger: attribution steps 4-5 listed as not run; leave-one-source-out for Breal's two sources",
          set(breal_gate["attribution_not_run"]) == {"label_audit", "leave_one_source_out"}
          and breal_gate["sources"] == sorted(P.BREAL_SLUGS)
          and set([e for e in ledger if e["id"] == "gate/full/3"][0]["attribution_not_run"]) == {"label_audit"})
    g = [e for e in ledger if e["id"] == "gate/full/3"][0]
    check("ledger: the gate entry stores the full Decision dict",
          G.Decision.from_dict(g["decision"]).verdict == "REJECT" and g["inputs"]["inc"]["run_id"]
          == st["chains"]["full"]["steps"]["3"]["incumbent_before"]["run_id"])

    subs = st["submissions"]
    ok_sub = True
    for sub in subs:
        lines = pathlib.Path(sub["list_file"]).read_text().split()
        for i, rid in enumerate(sub["tasks"]):
            if lines[i] != str(D.Paths(exp).spec(rid)):
                ok_sub = False
    last_job = {rid: rr["job"] for rid, rr in st["runs"].items()}
    ok_jobs = all(last_job[rid] == "%s_%d" % (sub["job_id"], i)
                  for sub in subs for i, rid in enumerate(sub["tasks"])
                  if st["runs"][rid]["submission"] == sub["n"])
    check("state records every submission: job id and array index -> run_id",
          ok_sub and ok_jobs and all(s["status"] == "submitted" and s["job_id"] for s in subs))
    check("the testing experiment's jobs get INC_SCORER_TESTING=1",
          all(s["testing_env"] == "1" for s in fb.submissions))

    n_sub = len(fb.submissions)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("advance after done submits nothing", len(fb.submissions) == n_sub)
    lines = D.Driver(exp).status()
    check("status: a line per chain and arm, marked TESTING",
          sum(1 for ln in lines if " chain " in ln) == 3 and any(" truth:" in ln for ln in lines)
          and any("done: yes" in ln for ln in lines) and all("[TESTING]" in ln for ln in lines), lines)

    rep = R.build(exp)
    md = (D.Paths(exp).root / "report.md").read_text()
    rj = json.loads((D.Paths(exp).root / "report.json").read_text())
    check("report: every step, every chain agrees with the truth arm",
          len(rj["steps"]) == 7 and all(v["agree"] == 7 and v["compared"] == 7 for v in rj["agreement"].values()))
    check("report: Bswap attributed to labels in every chain",
          all(c["bswap"]["attributed_to_labels"] for c in rj["chains"].values()))
    fin = {row["model"]: row for row in rj["final"]}
    base_row = fin["base P0"]
    t_row = fin["T_final (union of clean data)"]
    full_row = fin["chain full: final incumbent"]
    base_dev = [json.loads(D.Paths(exp).score("base__s%d" % s, "dev").read_text())["map50_95"] for s in range(3)]
    check("report: final table, mean +- sd over seeds (test included), one value per chain",
          base_row["exams"]["test"]["twelve"]["n"] == 3 and t_row["exams"]["dev"]["twelve"]["n"] == 3
          and full_row["exams"]["test"]["twelve"]["n"] == 1 and full_row["exams"]["test"]["twelve"]["sd"] is None
          and abs(base_row["exams"]["test"]["twelve"]["mean"] - 0.95 * statistics.fmean(base_dev)) < 1e-12
          and abs(base_row["exams"]["dev"]["twelve"]["sd"] - statistics.stdev(base_dev)) < 1e-12)
    want_full = (7 * 6 * 1800.0 + 5 * 60.0) / 3600.0
    want_lora = want_full + 5.0 / 3600.0
    check("report: GPU-hours per chain from run.json seconds (failed attempts included)",
          abs(rj["gpu_hours"]["chain:full"]["hours"] - want_full) < 1e-9
          and abs(rj["gpu_hours"]["chain:lora"]["hours"] - want_lora) < 1e-9
          and abs(rj["gpu_hours"]["truth"]["hours"] - 21 * 2.0) < 1e-9, rj["gpu_hours"])
    check("report.md renders the tables and says TESTING",
          "[TESTING]" in md and "| s03_Bswap |" in md and "## Final quality" in md
          and "## GPU-hours" in md and rep["testing"] is True)


def _tree(root, replace):
    """{relative path: bytes} of every file under root but exp.json and the
    lock, lease and request files, with the text `replace` masked."""
    out = {}
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.is_file() and p.name not in ("exp.json", "advance.lease", "state.json.lock", "advance.request"):
            out[str(p.relative_to(root))] = p.read_bytes().replace(str(replace).encode(), b"<INC_DIR>")
    return out


def test_replay_default(rows):
    print("replay mode: a definition without replay_mode (pilot_v1's) runs exactly as replay_mode 'sample'")
    exp = "v1_shape"
    runs = {}
    for label, key in (("absent", None), ("sample", "sample")):
        inc_dir = TMP / ("inc_replay_%s" % label)
        old = C.INC_DIR
        C.INC_DIR = inc_dir
        try:
            d = small_defn(exp, rows, step_names=("A", "B", "C"), recipes=("full", "lora"), truth=True)
            if key is not None:
                d["replay_mode"] = key
            fb = D.FakeBackend()
            clock = Clock()
            D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
            drive(exp, fb, clock, Executor())
            drv = D.Driver(exp)
            drv.status()
            ex = json.loads(D.Paths(exp).exp_json.read_text())
            runs[label] = {"tree": _tree(D.Paths(exp).root, inc_dir), "mode": drv.replay_mode, "exp": ex,
                           "state": state_of(exp), "subs": [s["n"] for s in fb.submissions],
                           "run_ids": sorted(s["run_id"] for s in all_specs(exp))}
        finally:
            C.INC_DIR = old
    a, b = runs["absent"], runs["sample"]
    check("(setup) both reach done", a["state"]["done"] and b["state"]["done"])
    check("a definition without replay_mode loads as 'sample', and its exp.json stays without the key",
          a["mode"] == "sample" and "replay_mode" not in a["exp"] and b["exp"]["replay_mode"] == "sample"
          and {k: v for k, v in b["exp"].items() if k != "replay_mode"} == a["exp"])
    diff = sorted(set(a["tree"]) ^ set(b["tree"])) + sorted(
        k for k in set(a["tree"]) & set(b["tree"]) if a["tree"][k] != b["tree"][k])
    check("... and writes byte for byte the same state, ledger, replay samples, manifests, specs and runs "
          "(%d files; INC_DIR masked)" % len(a["tree"]), not diff and a["subs"] == b["subs"], diff[:5])
    s1 = a["state"]["chains"]["full"]["steps"]["1"]
    check("a sample-mode step records what it always did: R1, R2, cand, null, the seed text, no replay_mode key",
          set(s1["manifests"]) == {"R1", "R2", "cand", "null"} and s1["replay_seed_text"] == "%s/full/1" % exp
          and "replay_mode" not in s1 and "pool_accepted" not in s1)

    # Pinned against the driver before replay_mode existed: the same definition
    # driven by that code drew these samples (sha256[:16] of each manifest's key
    # list, in file order; keys do not depend on TMP), made these runs and
    # these submissions. A sample-mode change that the two runs above share
    # fails here.
    def kh(keys):
        return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
    pinned = {("full", 1): ("d1de8836cc0ee032", "05b28a0e871b10fd", "a4892fc820f8a593", "e325e3501714918d"),
              ("full", 2): ("b650d3dc73a631df", "4144a9f20002bbae", "b34f6b652fc6a369", "7ccedf78d3457e2a"),
              ("full", 3): ("835545ad2f325ef0", "246fd404d965e1fa", "fe0e6eeb652e7a9f", "852b597bee673420"),
              ("lora", 1): ("8c250230e7938ba4", "26a3207c5bf15789", "0fe5acfd7b3e4d66", "0b9136b06a1da892"),
              ("lora", 2): ("e896e518c1a2b5a7", "16514672b8d05518", "9fc9eb10324f7d28", "aed5f92c7fd786a5"),
              ("lora", 3): ("df755ce476729294", "d5f9ec57a62094ab", "67c7fda6687aaec0", "87611efa802c5f67")}
    got = {}
    for (r, k) in pinned:
        s = a["state"]["chains"][r]["steps"][str(k)]
        got[(r, k)] = tuple(kh(keys_of(s["manifests"][m]["path"])) for m in ("R1", "R2", "cand", "null"))
        got[(r, k, "rest")] = (s["replay_seed_text"], s["pool_images"], s["d_images"], s["decision"]["verdict"])
    want = dict(pinned)
    want.update({(r, k, "rest"): ("%s/%s/%d" % (exp, r, k), 50 + 10 * k, 10, "ACCEPT") for (r, k) in pinned})
    check("sample mode draws the pre-change driver's R1 / R2 and trains its cand / null (pinned key-list "
          "digests, seed texts, pool sizes, verdicts; 2 chains x 3 steps)", got == want,
          {k: v for k, v in got.items() if want.get(k) != v})
    check("... and makes the pre-change driver's 62 runs in its 8 submissions of 12, 12, 2, 12, 2, 12, 2, 8 tasks",
          len(a["run_ids"]) == 62 and kh(a["run_ids"]) == "ec6f0ff2bfee5925"
          and a["subs"] == [12, 12, 2, 12, 2, 12, 2, 8], (len(a["run_ids"]), kh(a["run_ids"]), a["subs"]))


def test_full_rehearsal(world, sample_exp):
    print("replay mode full (full rehearsal): pilot build, every chain to done, ledger, specs, report")
    rows = world["rows"]
    fb = D.FakeBackend(first_job_id=3000)
    check("the builder refuses an unknown replay mode, writing nothing",
          raises(lambda: P.build_pilot("pilot_badmode", testing=True, backend=fb, quiet=True, replay_mode="half"),
                 P.PilotError, "replay mode") and not D.Paths("pilot_badmode").root.exists())
    exp = "pilotfull_t"
    summary, defn, _ = P.build_pilot(exp, testing=True, backend=fb, quiet=True, replay_mode="full")
    paths = D.Paths(exp)
    ex = json.loads(paths.exp_json.read_text())
    bsum = json.loads((paths.root / "build_summary.json").read_text())
    sample_ex = json.loads(D.Paths(sample_exp).exp_json.read_text())
    check("exp.json and build_summary.json say replay_mode full; the recipes, sequence, seeds and truth arm are "
          "the sample build's",
          ex["replay_mode"] == "full" and bsum["replay_mode"] == "full" and summary["replay_mode"] == "full"
          and all(ex[k] == sample_ex[k] for k in ("recipes", "truth_recipe", "truth", "seeds", "final_exams",
                                                   "init_weights", "type"))
          and [s["name"] for s in ex["steps"]] == [s["name"] for s in sample_ex["steps"]]
          and [s["clean"] for s in ex["steps"]] == [s["clean"] for s in sample_ex["steps"]]
          and ex["base"]["recipe"] == sample_ex["base"]["recipe"])
    check("a default build writes replay_mode sample into exp.json and the build summary",
          sample_ex["replay_mode"] == "sample"
          and json.loads((D.Paths(sample_exp).root / "build_summary.json").read_text())["replay_mode"] == "sample")

    # the effective warmup for the mode's actual cand / null sizes
    n = {s["name"]: s["n_images"] for s in ex["steps"]}
    p0 = ex["base"]["n_images"]
    ew = P.effective_warmup
    wu = ex["effective_warmup"]
    ok_wu = wu == bsum["warmup"] and wu["replay_mode"] == "full"
    for r, rec in P.inc_recipes().items():
        before = 0
        for i, name in enumerate(P.SEQUENCE, 1):
            want = {"cand": {"pool_min": ew(rec, p0 + n[name]), "pool_max": ew(rec, p0 + before + n[name])},
                    "null": {"pool_min": ew(rec, p0), "pool_max": ew(rec, p0 + before)}}
            ok_wu = ok_wu and wu["incremental"][r]["s%02d_%s" % (i, name)] == want
            before += n[name]
    sample_wu = P.warmup_table(P.inc_recipes(), P.cold_recipe(), p0,
                               [(nm, n[nm], nm in CLEAN) for nm in P.SEQUENCE])
    eff = [v["warmup_epochs_effective"] for rv in wu["incremental"].values() for sv in rv.values()
           for arm in sv.values() for v in arm.values()]
    check("warmup (full): cand = pool + D_k and null = pool, each at pool_min (P0) and pool_max (P0 + every "
          "earlier increment); cold runs as in a sample build",
          ok_wu and wu["cold"] == sample_wu["cold"]
          and wu["incremental_effective_epochs"] == {"min": min(eff), "max": max(eff)}, wu["incremental"]["full"])
    real = P.inc_recipes()["full"]
    check("warmup (full) at pilot_v1's sizes: null on P0 (1,540) warms up 2.04 of 30 epochs, cand on P0 + 238 "
          "1.79, and a pool of ~3,000 about 1 (sample mode: 6.67)",
          ew(real, 1540)["warmup_epochs_effective"] == 2.04 and ew(real, 1540 + 238)["warmup_epochs_effective"] == 1.79
          and ew(real, 2968)["warmup_epochs_effective"] == 1.08 and ew(real, 476)["warmup_epochs_effective"] == 6.67)

    inc_keys = {nm: set(keys_of(paths.manifests / ("%s.jsonl" % nm))) for nm in CLEAN}
    clock = Clock()
    exe = Executor(inc_keys)
    drive(exp, fb, clock, exe)
    st = state_of(exp)
    check("the full-rehearsal pilot reaches done", st["done"] is True, D.Driver(exp).status())
    keys = {nm: set(keys_of(paths.manifests / ("%s.jsonl" % nm))) for nm in ("P0",) + P.SEQUENCE}
    want_v = {"I1": "ACCEPT", "I2": "ACCEPT", "Bswap": "REJECT", "I3": "ACCEPT", "Breal": "HOLD",
              "I4": "ACCEPT", "I5": "ACCEPT"}
    tr = st["truth"]["steps"]
    for r, chn in st["chains"].items():
        got = {s["name"]: s["decision"]["verdict"] for s in chn["steps"].values()}
        check("chain %s (full): verdicts ACCEPT x5, REJECT Bswap, HOLD Breal" % r, got == want_v, got)
        bad, accepted, pools = [], [], []
        for k, name in enumerate(P.SEQUENCE, 1):
            s = chn["steps"][str(k)]
            pool = keys["P0"].union(*(keys[a] for a in accepted))
            cand = set(keys_of(s["manifests"]["cand"]["path"]))
            null = set(keys_of(s["manifests"]["null"]["path"]))
            if cand != pool | keys[name] or null != pool or keys[name] & pool:
                bad.append((name, "sets"))
            if (s["pool_images"], s["d_images"], s["manifests"]["cand"]["n_images"], s["manifests"]["null"]["n_images"]) \
                    != (len(pool), len(keys[name]), len(pool) + len(keys[name]), len(pool)):
                bad.append((name, "sizes"))
            if set(s["manifests"]) != {"cand", "null"} or s["replay_seed_text"] is not None \
                    or s["replay_mode"] != "full" or s["pool_accepted"] != accepted:
                bad.append((name, "record"))
            pools.append(len(pool))
            if s["decision"]["verdict"] == "ACCEPT":
                accepted.append(name)
        check("chain %s (full): cand = the whole accepted pool + D_k, null = the pool, D_k never in the pool; "
              "sizes, mode and the pool's increments recorded; no R1 / R2" % r, not bad, bad)
        want_pools = [p0]
        for name in P.SEQUENCE[:-1]:
            want_pools.append(want_pools[-1] + (len(keys[name]) if want_v[name] == "ACCEPT" else 0))
        check("chain %s (full): accepted increments join the pool for the later steps, the rejected (Bswap) and "
              "held (Breal) ones never do" % r,
              pools == want_pools and chn["accepted"] == list(CLEAN)
              and not keys["Bswap"] & set(keys_of(chn["steps"]["7"]["manifests"]["cand"]["path"]))
              and not keys["Breal"] & set(keys_of(chn["steps"]["7"]["manifests"]["cand"]["path"])), pools)
        same = all(chn["steps"][str(k)]["manifests"]["cand"]["sha256"] == tr[str(k)]["manifest"]["sha256"]
                   for k in range(1, 8))
        nulls = True
        for k in range(1, 8):
            j = max([0] + [i for i in range(1, k) if P.SEQUENCE[i - 1] in CLEAN])     # last clean step before k
            want_sha = tr[str(j)]["manifest"]["sha256"] if j else ex["base"]["manifest_sha256"]
            nulls = nulls and chn["steps"][str(k)]["manifests"]["null"]["sha256"] == want_sha
        check("chain %s (full): having accepted exactly the clean increments, each step's cand set is the truth "
              "arm's 'with' set and its null set the 'without' set (same manifest bytes)" % r, same and nulls)
        spec = json.loads(paths.spec("%s__s04_I3__null__s2" % r).read_text())
        check("chain %s (full): null s2 of step 4 trains the pool manifest from the incumbent with the chain's "
              "recipe" % r,
              spec["train_manifest"] == chn["steps"]["4"]["manifests"]["null"]["path"]
              and spec["init"] == chn["steps"]["4"]["incumbent_before"]["weights"]
              and spec["recipe"] == dict(P.inc_recipes()[r], seed=2))
    check("no replay sample is written in full mode", not list(paths.manifests.glob("*/*_R1.jsonl"))
          and not list(paths.manifests.glob("*/*_R2.jsonl")))

    gates = [e for e in ledger_of(exp) if e["type"] == "gate"]
    check("ledger: every gate entry records replay_mode full, the pool, |D_k|, the pool's increments and the "
          "manifests' sizes (cand = pool + D_k, null = pool)",
          len(gates) == 21 and all(
              e["replay_mode"] == "full" and e["replay_seed_text"] is None and set(e["manifests"]) == {"cand", "null"}
              and e["manifests"]["cand"]["n_images"] == e["pool_images"] + e["d_images"]
              and e["manifests"]["null"]["n_images"] == e["pool_images"]
              and e["pool_accepted"] == st["chains"][e["chain"]]["steps"][str(e["k"])]["pool_accepted"]
              for e in gates))
    specs = all_specs(exp)
    try:
        from weed_optimizer_framework.tools.inc import train as TR
    except ImportError as e:
        TR = None
        print("  skip inc/train.py cross-checks: %s" % e)
    if TR is not None:
        bad = []
        for s in specs:
            try:
                TR.validate_spec(s, paths.spec(s["run_id"]))
            except Exception as e:  # noqa: BLE001 -- any refusal is a failure here
                bad.append("%s: %s" % (s["run_id"], e))
        devs = {s["run_id"]: TR.protocol_deviations(s["kind"], s["recipe"]) for s in specs
                if s["kind"] in D.TRAIN_KINDS}
        check("inc/train.py accepts every full-mode spec (%d), and every training recipe is the protocol's "
              "for its kind" % len(specs), not bad and devs and not any(devs.values()),
              (bad[:3], {k: v for k, v in devs.items() if v}))
    lines = D.Driver(exp).status()
    check("status: every chain line says replay full",
          sum(1 for ln in lines if " chain " in ln and "(replay full)" in ln) == 3, lines)

    R.build(exp)
    rj = json.loads((paths.root / "report.json").read_text())
    md = (paths.root / "report.md").read_text()
    s4 = st["chains"]["full"]["steps"]["4"]
    rows_ok = all(c["replay_mode"] == "full"
                  and c["train_images"] == {"cand": st["chains"][r]["steps"][str(s["k"])]["manifests"]["cand"]["n_images"],
                                            "null": st["chains"][r]["steps"][str(s["k"])]["manifests"]["null"]["n_images"]}
                  and c["pool_images"] == st["chains"][r]["steps"][str(s["k"])]["pool_images"]
                  and c["warmup_epochs_effective"]["null"]
                  == ew(P.inc_recipes()[r], c["train_images"]["null"])["warmup_epochs_effective"]
                  for s in rj["steps"] for r, c in s["chains"].items())
    check("report: replay mode full, and per chain and step the cand / null train sizes, pool and warmup",
          rj["replay_mode"] == "full" and rows_ok and "- Replay mode: full rehearsal" in md
          and "images cand/null" in md
          and "| %d/%d |" % (s4["manifests"]["cand"]["n_images"], s4["manifests"]["null"]["n_images"]) in md
          and "From the decided steps' train sizes" in md and "pool_min (P0)" in md)
    check("report: every full-mode chain agrees with the truth arm",
          all(v["agree"] == 7 and v["compared"] == 7 for v in rj["agreement"].values()))
    rs = json.loads((D.Paths(sample_exp).root / "report.json").read_text())
    rs_md = (D.Paths(sample_exp).root / "report.md").read_text()
    check("report (sample build): replay mode sample, cand and null each 2|D_k| images",
          rs["replay_mode"] == "sample" and "- Replay mode: sample" in rs_md and all(
              c["replay_mode"] == "sample" and c["train_images"]["cand"] == c["train_images"]["null"]
              == 2 * c["d_images"] for s in rs["steps"] for c in s["chains"].values()))

    # full rehearsal needs no pool of 2|D_k|: the pool is not sampled
    clock = Clock()
    fb2 = D.FakeBackend(first_job_id=4000)
    d = small_defn("small_sample", rows, step_names=("A", "B"), base_n=15, step_n=10)
    D.Driver("small_sample", backend=fb2, clock=clock, quiet=True).init(d)
    drive("small_sample", fb2, clock, Executor())
    ss = state_of("small_sample")
    d = small_defn("small_full", rows, step_names=("A", "B"), base_n=15, step_n=10)
    d["replay_mode"] = "full"
    D.Driver("small_full", backend=fb2, clock=clock, quiet=True).init(d)
    drive("small_full", fb2, clock, Executor())
    sf = state_of("small_full")
    ch = sf["chains"]["full"]
    check("a pool smaller than 2|D_k| blocks a sample-mode chain but is rehearsed in full: cand 25 / null 15, "
          "then 35 / 25 once A is accepted",
          "too small" in ss["blocked"]["chain:full"]["error"] and sf["done"] is True and not sf["blocked"]
          and [(ch["steps"][k]["manifests"]["cand"]["n_images"], ch["steps"][k]["manifests"]["null"]["n_images"])
               for k in ("1", "2")] == [(25, 15), (35, 25)])

    def refused(e, dd, contains):
        return raises(lambda: D.Driver(e, backend=fb2, clock=clock, quiet=True).init(dd), D.DriverError, contains) \
            and not D.Paths(e).exp_json.exists() and not D.Paths(e).state.exists()
    check("init refuses an unknown replay_mode, writing nothing",
          refused("mode_bad", dict(small_defn("mode_bad", rows), replay_mode="half"), "replay_mode"))
    base = {k: v for k, v in small_defn("mode_base", rows).items()
            if k not in ("steps", "recipes", "truth", "truth_recipe")}
    check("init refuses replay_mode on a baseline experiment, writing nothing",
          refused("mode_base", dict(base, type="baseline", replay_mode="sample"), "chain"))
    saved_err = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        rc = P.main(["build-b0", "--exp", "b0_mode", "--testing", "--replay-mode", "full"])
    finally:
        sys.stderr.close()
        sys.stderr = saved_err
    check("pilot CLI: --replay-mode on build-b0 is refused", rc == 1 and not D.Paths("b0_mode").exp_json.exists())


# per chain, per step: the cand effect; + ACCEPT, - REJECT, 0 HOLD (cand and null tie seed for seed)
DIVERGE = {"full": {"A": 0.02, "B": -0.06, "C": 0.02, "D": 0.0, "E": 0.02},
           "lora": {"A": -0.06, "B": 0.02, "C": 0.0, "D": 0.02, "E": 0.02},
           "freeze": {"A": 0.0, "B": 0.0, "C": -0.06, "D": 0.02, "E": 0.02}}
DIVERGE_VERDICT = {0.02: "ACCEPT", -0.06: "REJECT", 0.0: "HOLD"}


class DivergeExecutor(Executor):
    """The synthetic executor with the cand effect keyed on (chain, step), so
    the three chains accept different increments and their pools differ."""

    def values(self, spec, exam):
        if spec["kind"] not in ("cand", "null"):
            return super().values(spec, exam)
        chain, tag = spec["run_id"].split("__")[:2]
        pv, pa = self.parent(spec["init"])
        eff = DIVERGE[chain][tag.split("_", 1)[1]] if spec["kind"] == "cand" else 0.0
        s = NOISE[spec["recipe"]["seed"]]
        return pv + eff + s, pa + max(eff, 0.0) + s


def test_full_rehearsal_divergent(rows):
    print("replay mode full, chains that decide differently: each chain rehearses its own accepted pool")
    exp = "full_diverge"
    steps = ("A", "B", "C", "D", "E")
    d = small_defn(exp, rows, step_names=steps, recipes=tuple(DIVERGE))
    d["replay_mode"] = "full"
    fb = D.FakeBackend(first_job_id=6000)
    clock = Clock()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
    drive(exp, fb, clock, DivergeExecutor())
    st = state_of(exp)
    check("(setup) the divergent full-mode experiment reaches done", st["done"] is True, D.Driver(exp).status())
    keys = {"base": set(keys_of(d["base"]["manifest"]))}
    keys.update((s["name"], set(keys_of(s["manifest"]))) for s in d["steps"])
    # what each chain must have accepted before each step, from the planted effects alone
    before = {}
    for r, plan in DIVERGE.items():
        acc = []
        for name in steps:
            before[(r, name)] = list(acc)
            if DIVERGE_VERDICT[plan[name]] == "ACCEPT":
                acc.append(name)
        before[(r, None)] = acc
    check("(setup) the planted plans give the three chains different pools at some step",
          len({tuple(tuple(before[(r, n)]) for n in steps) for r in DIVERGE}) == 3
          and any(len({tuple(before[(r, n)]) for r in DIVERGE}) == 3 for n in steps))
    gates = {e["id"]: e for e in ledger_of(exp) if e["type"] == "gate"}
    for r, plan in DIVERGE.items():
        ch = st["chains"][r]
        bad = []
        for k, name in enumerate(steps, 1):
            s = ch["steps"].get(str(k))
            if not s or not s.get("decision"):
                bad.append((name, "not decided"))
                continue
            acc = before[(r, name)]
            pool = keys["base"].union(*(keys[a] for a in acc))
            if s["decision"]["verdict"] != DIVERGE_VERDICT[plan[name]]:
                bad.append((name, "verdict", s["decision"]["verdict"]))
            cand = keys_of(s["manifests"]["cand"]["path"])
            null = keys_of(s["manifests"]["null"]["path"])
            if set(cand) != pool | keys[name] or set(null) != pool or keys[name] & pool \
                    or len(cand) != len(set(cand)) or len(null) != len(set(null)):
                bad.append((name, "sets"))
            if (s["pool_accepted"], s["pool_images"], s["d_images"], s["manifests"]["cand"]["n_images"],
                    s["manifests"]["null"]["n_images"]) != (acc, len(pool), len(keys[name]),
                                                           len(pool) + len(keys[name]), len(pool)):
                bad.append((name, "record", s["pool_accepted"], s["pool_images"]))
            g = gates.get("gate/%s/%d" % (r, k)) or {}
            if g.get("pool_accepted") != acc or g.get("pool_images") != len(pool) \
                    or g.get("manifests") != s["manifests"]:
                bad.append((name, "ledger"))
            for kind in ("cand", "null"):
                for sd in d["seeds"]:
                    sp = json.loads(D.Paths(exp).spec("%s__%s__%s__s%d" % (r, s["tag"], kind, sd)).read_text())
                    if sp["train_manifest"] != s["manifests"][kind]["path"] \
                            or sp["init"] != s["incumbent_before"]["weights"]:
                        bad.append((name, "spec", kind, sd))
        check("chain %s: planted verdicts; cand = base + this chain's accepted increments %s + D_k, null = that "
              "pool, D_k never in it; state, ledger and specs record it (expected pools from the plan, not the "
              "state)" % (r, before[(r, None)]), not bad and ch["accepted"] == before[(r, None)],
              (bad, ch["accepted"]))
    others = True
    for r in DIVERGE:
        mine = set(before[(r, None)])
        s = st["chains"][r]["steps"].get(str(len(steps)))
        last = set(keys_of(s["manifests"]["null"]["path"])) if s else set()
        others = others and bool(s)
        for o in DIVERGE:
            for name in set(before[(o, None)]) - mine:
                others = others and not keys[name] & last
    check("no chain's pool holds an increment only another chain accepted", others)


# ------------------------------------------------------------------ gate block
FLIP_BITS = "1" * 12 + "0" * 8                 # every run but a churned cand: 12 of 20 images correct
FLIP_PLAN = {"A": (6, 8), "B": (6, 1)}         # step -> (negative, positive) flips of every cand run


class FlipsExecutor(Executor):
    """The synthetic executor with per-image correctness: every run scores
    FLIP_BITS except the cand runs of a FLIP_PLAN step, which lose `negative`
    of its correct images and gain `positive` of its wrong ones. Step A is a
    clearly better model that also changes many images the other way (net -2),
    step B a model that breaks more than it fixes (net +5); every step's
    12-class effect is +0.02, so P_data = 1 throughout."""

    def values(self, spec, exam):
        self._cur = spec
        return super().values(spec, exam)

    def _score(self, path, exam, v, a, weights):
        Executor._score(path, exam, v, a, weights)
        spec = self._cur
        bits = FLIP_BITS
        if spec["kind"] == "cand" and exam == "dev":
            step = spec["run_id"].split("__")[1].split("_", 1)[1]
            if step in FLIP_PLAN:
                neg, pos = FLIP_PLAN[step]
                b = list(FLIP_BITS)
                for i in range(neg):
                    b[i] = "0"
                for i in range(pos):
                    b[12 + i] = "1"
                bits = "".join(b)
        s = json.loads(path.read_text())
        s["image_correct"] = bits
        path.write_text(json.dumps(s))


class SplitSoupExecutor(Executor):
    """The synthetic executor, except that every soup scores 0.01 below its
    cands' mean on map50_95 and 0.01 above it on map50: the soup wins on map50
    and loses on map50_95."""

    def _score(self, path, exam, v, a, weights):
        Executor._score(path, exam, v, a, weights)
        if "__soup" not in path.parent.parent.name:
            return
        s = json.loads(path.read_text())
        s["map50"] = min(1.0, v + 0.2 - 0.001 + 0.01)       # the cands' mean map50 is v - 0.001 + 0.2
        s["map50_95"] = v - 0.001 - 0.01
        path.write_text(json.dumps(s))


def decision_digests(exp):
    """sha256[:16] of the gate decisions, truth details and soup records of
    exp's ledger (sort_keys JSON; no path, clock or code hash, so the digest
    does not depend on TMP or on the code's own sha256)."""
    out = {}
    for e in ledger_of(exp):
        if e["type"] == "gate":
            body = [e["decision"], [x["sha256"] for arm in ("cand", "null") for x in e["inputs"][arm]]]
        elif e["type"] == "truth":
            body = e["detail"]
        elif e["type"] == "soup":
            body = [e["choice"], e["soup_value"], e["cand_values"], e["metric"]]
        else:
            continue
        out[e["id"]] = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    return out


# Pinned against the driver and gate before exp.json's gate block and
# GateConfig.flips_mode existed (2026-09-27): the "flips_v1" definition of
# _flips_run (2 chains x 3 steps, truth arm, FlipsExecutor) without a gate block
# gave these decision digests (decision_digests). A v1 change that the absent,
# {} and 'negative' runs share fails here.
PINNED_V1_DECISIONS = {
    "gate/full/1": "fe6e15389b4af07a", "gate/full/2": "4224bfea4f5eb7b2", "gate/full/3": "56ef49a92c75da32",
    "gate/lora/1": "f85f407425fdcf52", "gate/lora/2": "55414eb65bf6b82c", "gate/lora/3": "09188103630d2cd3",
    "soup/full/3": "d5030db33f86ff1b", "soup/lora/3": "d5030db33f86ff1b",
    "truth/1": "3fc6092fdd8b38d3", "truth/2": "3fc6092fdd8b38d3", "truth/3": "3fc6092fdd8b38d3"}


def _flips_run(label, rows, gate=None):
    """The flips_v1 definition, with `gate` as its gate block (None: no key),
    driven to done in its own INC_DIR; the tree is taken before the report."""
    exp = "flips_v1"
    inc_dir = TMP / ("inc_gate_%s" % label)
    old = C.INC_DIR
    C.INC_DIR = inc_dir
    try:
        d = small_defn(exp, rows, step_names=("A", "B", "C"), recipes=("full", "lora"), truth=True)
        if gate is not None:
            d["gate"] = gate
        fb = D.FakeBackend()
        clock = Clock()
        D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
        drive(exp, fb, clock, FlipsExecutor())
        out = {"tree": _tree(D.Paths(exp).root, inc_dir), "state": state_of(exp), "ledger": ledger_of(exp),
               "digests": decision_digests(exp), "subs": [x["n"] for x in fb.submissions],
               "exp": json.loads(D.Paths(exp).exp_json.read_text()), "lines": D.Driver(exp).status()}
        R.build(exp)
        out["report_md"] = (D.Paths(exp).root / "report.md").read_text()
        out["report"] = json.loads((D.Paths(exp).root / "report.json").read_text())
        return out
    finally:
        C.INC_DIR = old


def test_gate_block(rows):
    print("gate block: absent = {} = flips_mode negative (v1, byte for byte), net (v2), validation")
    runs = {label: _flips_run(label, rows, gate) for label, gate in
            (("absent", None), ("empty", {}), ("negative", {"flips_mode": "negative"}),
             ("net", {"flips_mode": "net"}))}
    check("(setup) all four reach done", all(r["state"]["done"] for r in runs.values()),
          {k: r["lines"] for k, r in runs.items() if not r["state"]["done"]})
    a = runs["absent"]
    check("(setup) exp.json keeps each gate block as defined, and nothing else differs",
          "gate" not in a["exp"] and runs["empty"]["exp"]["gate"] == {}
          and runs["negative"]["exp"]["gate"] == {"flips_mode": "negative"}
          and all({k: v for k, v in r["exp"].items() if k != "gate"} == a["exp"] for r in runs.values()))
    check("no gate block: state.json has no gate pin and the ledger no gate_pin entry (the state and ledger a "
          "definition without the block always had)",
          D.GATE_PIN not in a["state"] and not any(e["type"] == "gate_pin" for e in a["ledger"]))
    v1_pin = dataclasses.asdict(G.GateConfig(require_production=False))
    for label in ("empty", "negative"):
        b = runs[label]
        pin_lines = [ln for ln in b["tree"]["ledger.jsonl"].splitlines(keepends=True)
                     if json.loads(ln)["type"] == "gate_pin"]
        rest = {k: v for k, v in b["tree"].items() if k not in ("state.json", "ledger.jsonl")}
        diff = sorted(set(a["tree"]) ^ set(b["tree"])) + sorted(
            k for k in set(a["tree"]) & set(rest) if a["tree"][k] != rest[k])
        check("gate %s writes byte for byte the manifests, specs and runs of a definition without a gate block "
              "(%d files; INC_DIR masked), and the same submissions" % (json.dumps(b["exp"]["gate"]), len(a["tree"])),
              not diff and a["subs"] == b["subs"], diff[:5])
        st_b = json.loads(b["tree"]["state.json"])
        pin = st_b.pop(D.GATE_PIN, None)
        check("gate %s: state.json is the no-block state plus the gate pin, which spells the whole v1 config out "
              "(flips_mode 'negative' included)" % json.dumps(b["exp"]["gate"]),
              st_b == json.loads(a["tree"]["state.json"]) and pin == {"block": b["exp"]["gate"], "config": v1_pin}
              and v1_pin["flips_mode"] == "negative", pin)
        check("gate %s: the ledger is the no-block ledger plus one gate_pin entry after the code pin, holding the "
              "same config" % json.dumps(b["exp"]["gate"]),
              len(pin_lines) == 1 and b["tree"]["ledger.jsonl"].replace(pin_lines[0], b"") == a["tree"]["ledger.jsonl"]
              and [e["id"] for e in b["ledger"]][:2] == ["code_pin/0", "gate_pin/0"]
              and json.loads(pin_lines[0])["config"] == v1_pin
              and json.loads(pin_lines[0])["block"] == b["exp"]["gate"])
        check("gate %s: the gate decisions, truth details and soup records are the no-block run's"
              % json.dumps(b["exp"]["gate"]), b["digests"] == a["digests"])
    check("without a gate block the gate decisions, truth details and soup records are the pre-change driver's "
          "(pinned digests, 2 chains x 3 steps, planted flips)", a["digests"] == PINNED_V1_DECISIONS,
          {k: v for k, v in a["digests"].items() if PINNED_V1_DECISIONS.get(k) != v})
    gates = {e["id"]: e["decision"] for e in a["ledger"] if e["type"] == "gate"}
    check("v1: step A (net -2 per cand) is REJECTed by the flips guard alone at P_data 1, B too, C ACCEPTed; "
          "no decision records a flips_mode",
          all(gates["gate/%s/1" % r]["verdict"] == "REJECT" and gates["gate/%s/1" % r]["p_data"] == 1.0
              and gates["gate/%s/1" % r]["guards"]["regression"]["passed"]
              and gates["gate/%s/1" % r]["guards"]["species"]["passed"]
              and not gates["gate/%s/1" % r]["guards"]["flips"]["passed"]
              and gates["gate/%s/2" % r]["verdict"] == "REJECT" and gates["gate/%s/3" % r]["verdict"] == "ACCEPT"
              and a["state"]["chains"][r]["accepted"] == ["C"] for r in ("full", "lora"))
          and not any("flips_mode" in d["config"] for d in gates.values()))

    n = runs["net"]
    ng = {e["id"]: e["decision"] for e in n["ledger"] if e["type"] == "gate"}
    ok = True
    for r in ("full", "lora"):
        g1, g2 = ng["gate/%s/1" % r]["guards"]["flips"], ng["gate/%s/2" % r]["guards"]["flips"]
        ok = ok and (ng["gate/%s/1" % r]["verdict"], ng["gate/%s/2" % r]["verdict"],
                     ng["gate/%s/3" % r]["verdict"]) == ("ACCEPT", "REJECT", "ACCEPT")
        ok = ok and n["state"]["chains"][r]["accepted"] == ["A", "C"]
        ok = ok and g1["passed"] and g1["mode"] == "net" and g1["cand_negative_flips"] == [6, 6, 6] \
            and g1["cand_positive_flips"] == [8, 8, 8] and g1["cand_net_flips"] == [-2, -2, -2] \
            and g1["null_net_flips"] == [0, 0, 0] and g1["excess"] == -2.0 and g1["threshold"] == 3.0
        ok = ok and not g2["passed"] and g2["cand_net_flips"] == [5, 5, 5] and g2["excess"] == 5.0 \
            and "flips (net +5.0 > 3.0 images)" in ng["gate/%s/2" % r]["reason"]
    check("net: step A is ACCEPTed (it fixes more than it breaks), B still REJECTed on net flips (+5 > 3), "
          "C ACCEPTed; the chains accept A and C", ok, {k: d["verdict"] for k, d in ng.items()})
    check("net: every gate entry's decision records config flips_mode 'net' and the net guard with every count",
          all(d["config"]["flips_mode"] == "net" and d["guards"]["flips"]["mode"] == "net"
              and {"cand_negative_flips", "cand_positive_flips", "null_negative_flips",
                   "null_positive_flips"} <= set(d["guards"]["flips"]) for d in ng.values()))
    check("net: the truth arm is untouched (its details equal the v1 run's)",
          {k: v for k, v in n["digests"].items() if k.startswith("truth/")}
          == {k: v for k, v in a["digests"].items() if k.startswith("truth/")})
    check("status: every chain line of the net experiment says (gate flips net); a v1 one never mentions it",
          sum(1 for ln in n["lines"] if " chain " in ln and "(gate flips net)" in ln) == 2
          and not any("gate flips" in ln for ln in a["lines"]), n["lines"])
    net_pin = dataclasses.asdict(G.GateConfig(flips_mode="net", require_production=False))
    check("net: state.json pins the block and the whole resolved config; the ledger's second entry is that pin",
          n["state"][D.GATE_PIN] == {"block": {"flips_mode": "net"}, "config": net_pin}
          and n["ledger"][1]["id"] == "gate_pin/0" and n["ledger"][1]["config"] == net_pin)
    check("report: the header shows the pinned gate config (net: protocol v2, pinned from the block; v1: no block)",
          "- Gate: the flips guard counts net flips (protocol v2" in n["report_md"]
          and 'pinned at init (state.json gate_pin) from exp.json gate block {"flips_mode": "net"}' in n["report_md"]
          and "flips_mode=net" in n["report_md"]
          and n["report"]["gate"]["flips_mode"] == "net" and n["report"]["gate"]["block"] == {"flips_mode": "net"}
          and n["report"]["gate"]["pinned"] is True and n["report"]["gate"]["config"] == net_pin
          and "- Gate: the flips guard counts negative flips (protocol v1" in a["report_md"]
          and "no gate block in exp.json at init" in a["report_md"] and a["report"]["gate"]["block"] is None
          and a["report"]["gate"]["pinned"] is False
          and a["report"]["gate"]["config"] == dataclasses.asdict(G.GateConfig(require_production=False)))
    check("report: the pin is cross-checked against every gate, soup and truth entry (and the gate_pin entry); "
          "none differs and exp.json still matches",
          all(r["report"]["gate"]["ledger_mismatches"] == [] and r["report"]["gate"]["exp_json_changes"] == {}
              for r in runs.values())
          and n["report"]["gate"]["ledger_checked"] == {"gate": 6, "soup": 4, "truth": 3, "gate_pin": 1}
          and a["report"]["gate"]["ledger_checked"] == {"gate": 6, "soup": 2, "truth": 3}
          and "- Gate config check: the ledger's 6 gate, 4 soup and 3 truth entries and its gate_pin entry record "
              "the pinned config" in n["report_md"]
          and "GATE CONFIG" not in n["report_md"] and "GATE CONFIG" not in a["report_md"],
          [r["report"]["gate"] for r in runs.values()])
    rows_n = {(r, s["step"]): s["chains"][r] for s in n["report"]["steps"] for r in ("full", "lora")}
    # the synthetic union runs carry no increment effect here, so the truth arm is neutral on every step
    check("report (net): each step carries v1's verdict on the same runs: A REJECT (flips alone), B REJECT, "
          "C ACCEPT, where v2 gives ACCEPT, REJECT, ACCEPT",
          all([(rows_n[(r, x)]["verdict"], rows_n[(r, x)]["v1_counterfactual"]) for x in "ABC"]
              == [("ACCEPT", "REJECT"), ("REJECT", "REJECT"), ("ACCEPT", "ACCEPT")]
              and [rows_n[(r, x)]["v1_agree"] for x in "ABC"] == [False, False, False]
              and [rows_n[(r, x)]["agree"] for x in "ABC"] == [False, False, False] for r in ("full", "lora"))
          and all(s["truth"]["verdict"] == "neutral" for s in n["report"]["steps"]),
          {k: (v["verdict"], v.get("v1_counterfactual")) for k, v in rows_n.items()})
    chk = n["report"].get("v2_check") or {}
    check("report (net): the pre-registered v2 check: 6 (chain, step) pairs, neither rule agrees with the "
          "(neutral) truth arm, A is the discordant step in both chains: inconclusive (final)",
          chk.get("outcome") == "inconclusive" and chk.get("final") is True and chk.get("compared") == 6
          and chk.get("agree_v2") == 0 and chk.get("agree_v1_counterfactual") == 0
          and [(x["chain"], x["step"], x["v2"], x["v1_counterfactual"], x["truth"]) for x in chk["discordant"]]
          == [("full", "A", "ACCEPT", "REJECT", "neutral"), ("lora", "A", "ACCEPT", "REJECT", "neutral")]
          and chk.get("non_clean_accepted") == []
          and n["report"]["agreement"]["full"]["v1_counterfactual"] == {"agree": 0, "compared": 3}
          and "## Protocol v2 check (pre-registered)" in n["report_md"]
          and "- Outcome: inconclusive" in n["report_md"]
          and "v1 on the same runs would on 0 of 3" in n["report_md"]
          and "full A: v2 ACCEPT, v1 REJECT, truth neutral" in n["report_md"], chk)
    test_v2_check_rule()
    check("report (v1 experiments): no v1 counterfactual and no v2 check",
          all("v2_check" not in runs[x]["report"] and "Protocol v2 check" not in runs[x]["report_md"]
              and not any("v1_counterfactual" in (c or {}) for s in runs[x]["report"]["steps"]
                          for c in s["chains"].values())
              for x in ("absent", "empty", "negative")))

    # the gate block's config reaches every gate call: decide, choose_soup and truth_detail. Every soup
    # scores above its cands' mean on map50 and below it on map50_95, so choose_soup's pick itself shows
    # which metric it was given (the soup ledger entry's metric fields are written apart from that call).
    exp = "gate_metric"
    fb = D.FakeBackend()
    clock = Clock()
    d = small_defn(exp, rows, step_names=("A", "B"), recipes=("full",), truth=True)
    d["gate"] = {"metric": "map50"}
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
    drive(exp, fb, clock, SplitSoupExecutor())
    led = ledger_of(exp)

    def m50(x):
        return json.loads(pathlib.Path(x["path"]).read_text())["map50"]

    def m5095(x):
        return json.loads(pathlib.Path(x["path"]).read_text())["map50_95"]
    g = [e for e in led if e["type"] == "gate"]
    t = [e for e in led if e["type"] == "truth"]
    sp = [e for e in led if e["type"] == "soup"]
    check("a gate block's metric reaches decide, choose_soup and truth_detail: every entry is on map50, "
          "with the score files' map50 values",
          state_of(exp)["done"] and len(g) == 2 and len(t) == 2 and len(sp) == 2
          and all(e["decision"]["metric"] == "map50" and e["decision"]["config"]["metric"] == "map50"
                  and e["decision"]["cand_values"] == [m50(x) for x in e["inputs"]["cand"]]
                  and e["decision"]["inc"] == m50(e["inputs"]["inc"]) for e in g)
          and all(e["detail"]["metric"] == "map50"
                  and e["detail"]["with_values"] == [m50(x) for x in e["inputs"]["with"]] for e in t)
          and all(e["metric"] == "map50" and e["soup_value"] == m50(e["inputs"]["soup"]) for e in sp),
          [(e["id"], e.get("metric")) for e in sp])
    check("choose_soup decides on the block's metric: every soup is above its cands' mean on map50 and below it "
          "on map50_95, and every pick is the soup, which becomes the incumbent",
          len(sp) == 2 and all(
              m50(e["inputs"]["soup"]) > statistics.fmean(m50(x) for x in e["inputs"]["cand"])
              and m5095(e["inputs"]["soup"]) < statistics.fmean(m5095(x) for x in e["inputs"]["cand"])
              and e["choice"] == "soup" and e["incumbent_after"]["run_id"].endswith("__soup") for e in sp)
          and state_of(exp)["chains"]["full"]["incumbent"]["run_id"] == sp[-1]["incumbent_after"]["run_id"],
          [(e["id"], e["choice"]) for e in sp])

    check("gate_config: absent = GateConfig() (production) or GateConfig(require_production=False) (testing); "
          "an integer threshold is read as the float field it is",
          D.gate_config({"testing": False}) == G.GateConfig()
          and D.gate_config({"testing": True}) == G.GateConfig(require_production=False)
          and D.gate_config({"gate": {"flips_mode": "net"}}) == G.GateConfig(flips_mode="net")
          and D.gate_config({"gate": {"flips_slack_images": 3}}) == G.GateConfig()
          and G.config_record(D.gate_config({"gate": {"flips_slack_images": 3}})) == G.config_record(G.GateConfig())
          and D.GATE_KEYS == tuple(f.name for f in dataclasses.fields(G.GateConfig) if f.name != "require_production"))

    def refused(e, gate, contains, typ="chain"):
        dd = small_defn(e, rows)
        if typ == "baseline":
            dd = dict({k: v for k, v in dd.items() if k not in ("steps", "recipes", "truth", "truth_recipe")},
                      type="baseline")
        dd["gate"] = gate
        return raises(lambda: D.Driver(e, backend=fb, clock=clock, quiet=True).init(dd), D.DriverError, contains) \
            and not D.Paths(e).exp_json.exists() and not D.Paths(e).state.exists()
    for i, (gate, contains) in enumerate((
            ({"flips_mod": "net"}, "unknown key"),
            ({"require_production": False}, "require_production"),
            ({"flips_mode": "both"}, "flips_mode"),
            ({"flips_mode": None}, "flips_mode"),
            ({"flips_slack_images": "3"}, "flips_slack_images"),
            ({"flips_sd_mult": True}, "flips_sd_mult"),
            ({"flips_slack_images": -1}, "non-negative"),
            ({"flips_sd_mult": float("nan")}, "finite"),
            ({"p_accept": 1.5}, "probability"),
            ({"p_reject": 0.8}, "below p_accept"),
            ({"metric": "agnostic_map50_95"}, "metric"),
            ({"min_seeds": 0}, "positive integer"),
            ({"min_seeds": 4}, "at least 4 seeds"),
            ("net", "object"),
            (["flips_mode"], "object"))):
        check("init refuses the gate block %s, writing nothing" % json.dumps(gate),
              refused("gate_bad%d" % i, gate, contains))
    check("init refuses the gate block {\"flips_slack_images\": 10**400} (too large for a float) with a "
          "DriverError, writing nothing", refused("gate_bad_big", {"flips_slack_images": 10 ** 400}, "too large"))
    check("init refuses a gate block on a baseline experiment, writing nothing",
          refused("gate_base", {"flips_mode": "net"}, "chain", typ="baseline"))


def test_gate_builders(world, sample_exp):
    print("gate block: the pilot builder writes it, its CLI refuses it where no gate decides")
    sample_ex = json.loads(D.Paths(sample_exp).exp_json.read_text())
    sample_bs = json.loads((D.Paths(sample_exp).root / "build_summary.json").read_text())
    check("a default pilot build writes gate {flips_mode: negative} into exp.json and build_summary.json",
          sample_ex["gate"] == {"flips_mode": "negative"} and sample_bs["gate"] == {"flips_mode": "negative"})
    fb = D.FakeBackend(first_job_id=7000)
    check("the builder refuses an unknown gate flips mode, writing nothing",
          raises(lambda: P.build_pilot("pilot_badgate", testing=True, backend=fb, quiet=True,
                                       gate_flips_mode="both"), P.PilotError, "gate flips mode")
          and not D.Paths("pilot_badgate").root.exists())
    exp = "pilotnet_t"
    summary, defn, _ = P.build_pilot(exp, testing=True, backend=fb, quiet=True, gate_flips_mode="net")
    ex = json.loads(D.Paths(exp).exp_json.read_text())
    bs = json.loads((D.Paths(exp).root / "build_summary.json").read_text())
    check("--gate-flips-mode net: exp.json and build_summary.json hold gate {flips_mode: net}; the recipes, "
          "sequence, seeds, replay mode and truth arm are the default build's",
          ex["gate"] == {"flips_mode": "net"} and bs["gate"] == {"flips_mode": "net"}
          and summary["gate"] == {"flips_mode": "net"} and defn["gate"] == {"flips_mode": "net"}
          and all(ex[k] == sample_ex[k] for k in ("recipes", "truth_recipe", "truth", "seeds", "final_exams",
                                                   "init_weights", "type", "replay_mode"))
          and [s["name"] for s in ex["steps"]] == [s["name"] for s in sample_ex["steps"]]
          and D.gate_config(ex) == G.GateConfig(flips_mode="net", require_production=False))
    lines = D.Driver(exp).status()
    check("... and every chain's status line says (gate flips net)",
          sum(1 for ln in lines if " chain " in ln and "(gate flips net)" in ln) == 3, lines)
    fb2 = D.FakeBackend(first_job_id=8000)
    saved_sb = D.SlurmBackend
    D.SlurmBackend = lambda *a, **k: fb2
    saved_err = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        rc_net = P.main(["build", "--exp", "pilot_cli_net", "--testing", "--gate-flips-mode", "net", "--quiet"])
        rc_b0 = P.main(["build-b0", "--exp", "b0_gate", "--testing", "--gate-flips-mode", "net"])
        rc_bl = P.main(["build-baseline", "--exp", "bl_gate", "--manifest", str(C.manifest_path("train_core")),
                        "--testing", "--gate-flips-mode", "net"])
    finally:
        sys.stderr.close()
        sys.stderr = saved_err
        D.SlurmBackend = saved_sb
    cli = json.loads(D.Paths("pilot_cli_net").exp_json.read_text()) if rc_net == 0 else {}
    check("pilot CLI: build --gate-flips-mode net writes the block (exit 0); build-b0 and build-baseline refuse "
          "the flag (exit 1) and build nothing",
          rc_net == 0 and cli.get("gate") == {"flips_mode": "net"} and rc_b0 == 1 and rc_bl == 1
          and not D.Paths("b0_gate").exp_json.exists() and not D.Paths("bl_gate").exp_json.exists())


def test_gate_pin(rows):
    print("gate pin: the gate config is fixed at init; an edited gate block or testing flag stops the experiment")

    def to_first_decision(exp, fb, clock, ex):
        for _ in range(200):
            fb.run_pending(ex)
            D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
            if any(e["type"] == "gate" for e in ledger_of(exp)):
                return True
        return False

    def edit(exp, **changes):
        p = D.Paths(exp).exp_json
        j = json.loads(p.read_text())
        for k, v in changes.items():
            if v is None:
                j.pop(k, None)
            else:
                j[k] = v
        p.write_text(json.dumps(j))

    def frozen(exp):
        paths = D.Paths(exp)
        return paths.state.read_bytes(), paths.ledger.read_bytes()

    # a definition without a block: its pin is the protocol's defaults (nothing written)
    exp = "pin_none"
    fb, clock = D.FakeBackend(), Clock()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(
        small_defn(exp, rows, step_names=("A", "B", "C"), recipes=("full",)))
    ex = FlipsExecutor()
    check("(setup) the no-block chain reaches its first decision, a v1 REJECT on flips alone",
          to_first_decision(exp, fb, clock, ex)
          and [e["decision"]["verdict"] for e in ledger_of(exp) if e["type"] == "gate"] == ["REJECT"]
          and D.GATE_PIN not in state_of(exp))
    original = D.Paths(exp).exp_json.read_text()
    before, n_subs = frozen(exp), len(fb.submissions)
    edit(exp, gate={"flips_mode": "net", "flips_slack_images": 50})
    fb.run_pending(ex)
    check("a gate block added after init: advance refuses, naming the changed fields, and writes nothing",
          raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.DriverError,
                 "gate config differs from the one experiment pin_none was initialised with")
          and frozen(exp) == before and len(fb.submissions) == n_subs)
    try:
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        msg = ""
    except D.DriverError as e:
        msg = str(e)
    check("... the refusal names each changed field as [pinned, now] and is not a code-pin refusal (the "
          "autopilot's classifier must not read it as one)",
          "'flips_slack_images': [3.0, 50.0]" in msg and "'flips_mode': ['negative', 'net']" in msg
          and "initialised without a gate block" in msg and "code changed since experiment" not in msg, msg)
    lines = D.Driver(exp).status()
    check("... status says GATE CONFIG CHANGED and still tags the chain with the pinned (default) config",
          any("GATE CONFIG CHANGED" in ln and "flips_mode" in ln for ln in lines)
          and not any(" chain " in ln and "(gate" in ln for ln in lines), lines)
    R.build(exp)
    md = (D.Paths(exp).root / "report.md").read_text()
    rj = json.loads((D.Paths(exp).root / "report.json").read_text())
    check("... the report shows the pinned v1 config, not exp.json's, and flags the change",
          "- Gate: the flips guard counts negative flips (protocol v1" in md
          and "no gate block in exp.json at init" in md and "- GATE CONFIG CHANGED:" in md
          and rj["gate"]["flips_mode"] == "negative" and rj["gate"]["config"]["flips_slack_images"] == 3.0
          and set(rj["gate"]["exp_json_changes"]) == {"flips_mode", "flips_slack_images"}, rj["gate"])
    for label, change in (("removing testing (require_production)", {"testing": None}),
                          ("a block that changes one threshold (p_accept 0.8)", {"gate": {"p_accept": 0.8}})):
        D.Paths(exp).exp_json.write_text(original)
        edit(exp, **change)
        check("%s is refused too" % label,
              raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.DriverError,
                     "gate config differs") and frozen(exp) == before)
    D.Paths(exp).exp_json.write_text(original)
    drive(exp, fb, clock, ex)
    gates = [e for e in ledger_of(exp) if e["type"] == "gate"]
    check("restoring exp.json resumes the chain; every decision is a v1 one at slack 3",
          state_of(exp)["done"] and len(gates) == 3
          and all("flips_mode" not in e["decision"]["config"] and e["decision"]["config"]["flips_slack_images"] == 3.0
                  for e in gates)
          and [e["decision"]["verdict"] for e in gates] == ["REJECT", "REJECT", "ACCEPT"],
          [e["decision"]["verdict"] for e in gates])

    # a definition with a block: the pin is state.json's, the ledger's gate_pin/0 says it too
    exp = "pin_net"
    fb, clock = D.FakeBackend(), Clock()
    d = small_defn(exp, rows, step_names=("A", "B"), recipes=("full",))
    d["gate"] = {"flips_mode": "net"}
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
    check("(setup) the net chain reaches its first decision, a net ACCEPT",
          to_first_decision(exp, fb, clock, ex)
          and [e["decision"]["verdict"] for e in ledger_of(exp) if e["type"] == "gate"] == ["ACCEPT"])
    original = D.Paths(exp).exp_json.read_text()
    before = frozen(exp)
    for label, change in (("removing the net block", {"gate": None}),
                          ("switching it to negative", {"gate": {"flips_mode": "negative"}}),
                          ("relaxing its slack", {"gate": {"flips_mode": "net", "flips_slack_images": 50}})):
        D.Paths(exp).exp_json.write_text(original)
        edit(exp, **change)
        check("net experiment: %s after init is refused, nothing written" % label,
              raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.DriverError,
                     "(state.json gate_pin)") and frozen(exp) == before)
    D.Paths(exp).exp_json.write_text(original)
    edit(exp, gate={"flips_mode": "net", "flips_slack_images": 3})
    fb.run_pending(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("a block that resolves to the pinned config (a default written out) is the same rule and runs",
          frozen(exp) != before)
    D.Paths(exp).exp_json.write_text(original)
    drive(exp, fb, clock, ex)
    led = ledger_of(exp)
    check("net experiment: done, every decision on net flips, one gate_pin entry",
          state_of(exp)["done"] and all(e["decision"]["config"]["flips_mode"] == "net"
                                        for e in led if e["type"] == "gate")
          and sum(1 for e in led if e["type"] == "gate_pin") == 1)

    # the report cross-checks the pin against the ledger's decisions, not exp.json
    defn = json.loads(D.Paths(exp).exp_json.read_text())
    st = state_of(exp)
    ledger = {e["id"]: e for e in led}
    g = R.gate_record(defn, st, ledger)
    check("report: a consistent experiment has no mismatch",
          g["ledger_mismatches"] == [] and g["exp_json_changes"] == {} and g["pinned"] is True)
    forged = json.loads(json.dumps(st))
    forged[D.GATE_PIN]["config"]["flips_mode"] = "negative"
    g = R.gate_record(defn, forged, ledger)
    check("report: a pin the decisions do not match lists every gate entry and the gate_pin entry as mismatches",
          sorted(m["id"] for m in g["ledger_mismatches"]) == sorted(
              [e["id"] for e in led if e["type"] in ("gate", "gate_pin")])
          and g["flips_mode"] == "negative" and g["exp_json_changes"] == {"flips_mode": ["negative", "net"]}, g)
    forged = json.loads(json.dumps(led))
    for e in forged:
        if e["type"] == "soup":
            e["metric"] = "map50"
    g = R.gate_record(defn, st, {e["id"]: e for e in forged})
    check("report: a soup entry decided on another metric is a mismatch",
          [m["id"] for m in g["ledger_mismatches"]] == [e["id"] for e in led if e["type"] == "soup"]
          and all(m["recorded"] == {"metric": "map50"} for m in g["ledger_mismatches"]), g["ledger_mismatches"])

    # status tags and the edges of the block's validation
    check("status tag: defaults (production or testing) none; net 'flips net' first; any other non-default "
          "threshold by name",
          D.gate_tag(G.GateConfig()) == "" and D.gate_tag(G.GateConfig(require_production=False)) == ""
          and D.gate_tag(G.GateConfig(flips_mode="net")) == " (gate flips net)"
          and D.gate_tag(G.GateConfig(flips_slack_images=50.0)) == " (gate flips_slack_images=50.0)"
          and D.gate_tag(G.GateConfig(flips_mode="net", min_seeds=1, flips_slack_images=50.0))
          == " (gate flips net, flips_slack_images=50.0, min_seeds=1)")
    check("validate_gate: an integer too large for a float is a DriverError, not an OverflowError",
          raises(lambda: D.validate_gate({"flips_sd_mult": 10 ** 400}), D.DriverError, "too large")
          and raises(lambda: D.validate_gate({"p_accept": 10 ** 400}), D.DriverError, "too large"))
    check("pinned_gate_config: no pin = the defaults by state's testing; a pin that is not a config raises",
          D.pinned_gate_config({"testing": False}) == G.GateConfig()
          and D.pinned_gate_config({"testing": True}) == G.GateConfig(require_production=False)
          and D.pinned_gate_config({D.GATE_PIN: {"block": {}, "config": net_pin_of(True)}})
          == G.GateConfig(flips_mode="net", require_production=False)
          and raises(lambda: D.pinned_gate_config({D.GATE_PIN: {"config": {"flips_mod": "net"}}}), D.DriverError,
                     "not a GateConfig"))


def test_v2_check_rule():
    """report.v2_check's outcome rule on hand-built step rows."""
    net = {"flips_mode": "net"}

    def step(name, clean, truth, **chains):
        t = {"verdict": truth} if truth else None
        eq = {"ACCEPT": "helps", "HOLD": "neutral", "REJECT": "hurts"}
        return {"step": name, "clean": clean, "truth": t,
                "chains": {r: {"verdict": v2, "v1_counterfactual": v1,
                               "agree": (eq[v2] == truth) if truth else None,
                               "v1_agree": (eq[v1] == truth) if truth else None}
                           for r, (v2, v1) in chains.items()}}
    done, running = {"done": True}, {"done": False}
    truth = {"truth": True}
    cases = [
        ("v2 right where v1 is wrong", truth, done,
         [step("I1", True, "helps", full=("ACCEPT", "REJECT")), step("I2", True, "hurts", full=("REJECT", "REJECT"))],
         ("supported", 2, 1, 1, 0)),
        ("v2 wrong where v1 is right", truth, done,
         [step("I1", True, "hurts", full=("ACCEPT", "REJECT")), step("I2", True, "helps", full=("ACCEPT", "ACCEPT"))],
         ("refuted", 1, 2, 1, 0)),
        ("a non-clean step v2 accepts refutes it even when v2 agrees more (and v1 accepts it too)", truth, done,
         [step("I1", True, "helps", full=("ACCEPT", "REJECT")), step("Breal", False, "helps", full=("ACCEPT", "ACCEPT"))],
         ("refuted", 2, 1, 1, 1)),
        ("a tie, one discordant step each way", truth, done,
         [step("I1", True, "helps", full=("ACCEPT", "REJECT")), step("I2", True, "hurts", full=("ACCEPT", "REJECT"))],
         ("inconclusive", 1, 1, 2, 0)),
        ("no step on which the rules differ", truth, done,
         [step("I1", True, "helps", full=("ACCEPT", "ACCEPT"), lora=("HOLD", "HOLD"))],
         ("inconclusive", 1, 1, 0, 0)),
        ("no truth arm: nothing compared; a non-clean acceptance still refutes", {"truth": False}, done,
         [step("I1", True, None, full=("ACCEPT", "REJECT")), step("UNVERIFIED", False, None, full=("ACCEPT", "REJECT"))],
         ("refuted", 0, 0, 2, 1)),
    ]
    for label, defn, st, steps, (outcome, a2, a1, n_disc, n_bad) in cases:
        c = R.v2_check(defn, st, steps, net)
        check("v2 check: %s -> %s" % (label, outcome),
              (c["outcome"], c["agree_v2"], c["agree_v1_counterfactual"], len(c["discordant"]),
               len(c["non_clean_accepted"])) == (outcome, a2, a1, n_disc, n_bad) and c["final"] is True, c)
    c = R.v2_check(truth, running, cases[0][3], net)
    check("v2 check: provisional (final false) until the experiment is done", c["final"] is False
          and c["outcome"] == "supported")
    check("v2 check: none for a v1 experiment or a baseline",
          R.v2_check(truth, done, cases[0][3], {"flips_mode": "negative"}) is None
          and R.v2_check(truth, done, cases[0][3], None) is None)


def net_pin_of(testing):
    return dataclasses.asdict(G.GateConfig(flips_mode="net", require_production=not testing))


def test_idempotent_and_lock(rows):
    print("idempotent advance, lock")
    exp = "idem_t"
    fb = D.FakeBackend()
    clock = Clock()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows))
    check("init submits the base runs once", len(fb.submissions) == 1 and fb.submissions[0]["n"] == 3)
    before = state_of(exp)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    after = state_of(exp)
    check("a second advance with nothing finished submits nothing and writes nothing",
          len(fb.submissions) == 1 and before == after)
    check("init again with the same definition is a no-op",
          D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows))["submitted"] == 0
          and len(fb.submissions) == 1)
    other = small_defn(exp, rows, step_names=("A",))
    check("init with a different definition refuses",
          raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(other), D.DriverError, "different"))

    fb.run_pending(Executor())
    paths = D.Paths(exp)
    fh = open(paths.lock, "a+")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        st_before = paths.state.read_bytes()
        t0 = time.time()
        res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        dt = time.time() - t0
        check("lock held: advance returns at once, touching nothing",
              res["locked"] is True and res["passes"] == 0 and dt < 2.0
              and paths.state.read_bytes() == st_before and len(fb.submissions) == 1, (res, dt))
        check("lock held: the call leaves a request for the holder", paths.request.exists())
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    fh = open(paths.lock, "a+")
    fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
    try:
        res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        check("the lock is exclusive: a shared holder also makes advance return at once",
              res["locked"] is True and len(fb.submissions) == 1)
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("after the lock is released the next call collects and moves on",
          res["passes"] == 1 and not paths.request.exists()
          and st["chains"]["full"]["phase"] == "step" and len(fb.submissions) == 2
          and fb.submissions[1]["n"] == 6)


def test_crash_recovery(rows):
    print("a pass that dies after the ledger append and before saving state")
    exp = "crash_t"
    fb = D.FakeBackend()
    clock = Clock()
    ex = Executor()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    fb.run_pending(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()     # step A's six runs go out
    fb.run_pending(ex)
    state_before = D.Paths(exp).state.read_bytes()

    class Crash(Exception):
        pass

    drv = D.Driver(exp, backend=fb, clock=clock, quiet=True)

    def die():
        raise Crash()
    drv._save_state = die
    check("the pass dies after deciding", raises(drv.advance, Crash))
    check("... having appended the gate entry but saved no state",
          [e["id"] for e in ledger_of(exp)] == ["code_pin/0", "gate/full/1"]
          and D.Paths(exp).state.read_bytes() == state_before)
    n_sub = len(fb.submissions)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    soups = [s for sub in fb.submissions for s in sub["specs"] if s.endswith("__soup/spec.json")]
    check("the next pass re-derives the same decision without a second ledger entry",
          [e["id"] for e in ledger_of(exp)] == ["code_pin/0", "gate/full/1"]
          and st["chains"]["full"]["steps"]["1"]["decision"]["verdict"] == "ACCEPT")
    check("... and reuses the soup spec the dead pass wrote, submitting it once",
          len(soups) == 1 and len(fb.submissions) == n_sub + 1 and st["chains"]["full"]["phase"] == "soup")
    drive(exp, fb, clock, ex)
    check("... and the experiment still finishes", state_of(exp)["done"] is True)


def test_retry_block(rows):
    print("retry once, then blocked; ended and vanished tasks")
    exp = "block_t"
    fb = D.FakeBackend()
    clock = Clock()
    ex = Executor()
    ex.lose_once.add("base__s1")
    ex.end_failed_once.add("base__s2")
    ex.fail_always.add("full__s01_A__cand__s1")
    defn = small_defn(exp, rows, recipes=("full", "lora"))
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(defn)
    fb.run_pending(ex)
    clock.t += 60
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    s1, s2 = st["runs"]["base__s1"], st["runs"]["base__s2"]
    check("a task that ended (sacct FAILED) without run.json fails at once and is resubmitted",
          s2["attempt"] == 2 and s2["status"] == "submitted" and "ended FAILED" in s2["history"][0]["error"])
    check("a vanished task is not failed before 8 h", s1["attempt"] == 1 and s1["status"] == "submitted")
    check("the retry and the new step runs go out as one array",
          len(fb.submissions) == 2 and fb.submissions[1]["n"] == 1 + 12
          and "base__s2" in fb.submissions[1]["specs"][0])
    clock.t += 8 * 3600 + 1
    fb.run_pending(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    s1, c1 = st["runs"]["base__s1"], st["runs"]["full__s01_A__cand__s1"]
    check("after 8 h a vanished task (no run.json, not in squeue) is failed and resubmitted",
          s1["attempt"] == 2 and s1["status"] == "submitted" and "not in squeue" in s1["history"][0]["error"])
    check("a run.json failure is resubmitted once", c1["attempt"] == 2 and c1["status"] == "submitted")
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    c1 = state_of(exp)["runs"]["full__s01_A__cand__s1"]
    check("while the retry waits, the failed attempt's run.json (still on disk) is not counted again",
          c1["attempt"] == 2 and c1["status"] == "submitted" and len(c1["history"]) == 1
          and D.Paths(exp).run_json("full__s01_A__cand__s1").is_file())
    fb.run_pending(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    c1 = st["runs"]["full__s01_A__cand__s1"]
    check("a second failure marks the run failed and blocks its chain with the error",
          c1["status"] == "failed" and st["chains"]["full"]["phase"] == "blocked"
          and "synthetic failure" in st["blocked"]["chain:full"]["error"])
    drive(exp, fb, clock, ex)
    st = state_of(exp)
    check("the other chain is not blocked and finishes every step",
          st["chains"]["lora"]["phase"] == "done" and "chain:lora" not in st["blocked"]
          and [s["decision"]["verdict"] for s in st["chains"]["lora"]["steps"].values()] == ["ACCEPT", "ACCEPT"])
    check("a blocked chain keeps the experiment from its final runs",
          st["done"] is False and not st["final"]["created"])
    lines = D.Driver(exp).status()
    check("status shows the block", any("chain full" in ln and "BLOCKED" in ln for ln in lines), lines)
    n_sub = len(fb.submissions)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("advancing a blocked experiment submits nothing", len(fb.submissions) == n_sub)


class BlindQueue(D.FakeBackend):
    """squeue returns nothing (a controller hiccup); sacct still answers."""

    def queued(self):
        return set()


def test_squeue_glitch(rows):
    print("squeue glitch: sacct says the task is live")
    exp = "glitch_t"
    fb = BlindQueue()
    clock = Clock()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    t = sorted(fb.tasks)[0]
    fb.tasks[t]["state"] = "RUNNING"
    clock.t += 9 * 3600
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("a task absent from squeue for > 8 h is not failed while sacct says it is pending or running",
          all(r["status"] == "submitted" and r["attempt"] == 1 for r in st["runs"].values())
          and len(fb.submissions) == 1)


def test_production_and_env(rows):
    print("production gate, testing flag")
    exp = "prod_t"
    fb = D.FakeBackend()
    clock = Clock()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",), testing=False))
    check("a production experiment's jobs do not get INC_SCORER_TESTING",
          fb.submissions[0]["testing_env"] is None)
    drive(exp, fb, clock, Executor())
    st = state_of(exp)
    check("a production experiment is never decided on test-mode scores: the chain blocks",
          st["chains"]["full"]["phase"] == "blocked"
          and "not a production score" in st["blocked"]["chain:full"]["error"]
          and st["chains"]["full"]["steps"]["1"]["decision"] is None
          and [e["type"] for e in ledger_of(exp)] == ["code_pin"])

    exp3 = "settings_t"
    fb3 = D.FakeBackend()
    D.Driver(exp3, backend=fb3, clock=clock, quiet=True).init(
        small_defn(exp3, rows, step_names=("A",), testing={"imgsz": 64, "batch": 2, "device": "cpu"}))
    drive(exp3, fb3, clock, Executor())
    st3 = state_of(exp3)
    check("exp.json testing may be inc/train.py's settings object; it counts as testing",
          st3["done"] is True and st3["testing"] is True and fb3.submissions[0]["testing_env"] == "1"
          and all(e["testing"] is True for e in ledger_of(exp3)))
    bad_t = small_defn("bad_t", rows, step_names=("A",), testing={"epochs": 1})
    check("a testing object with keys inc/train.py does not know is refused",
          raises(lambda: D.Driver("bad_t", backend=fb3, clock=clock, quiet=True).init(bad_t),
                 D.DriverError, "testing"))

    exp2 = "env_t"
    fb2 = D.FakeBackend()
    D.Driver(exp2, backend=fb2, clock=clock, quiet=True).init(small_defn(exp2, rows, step_names=("A",)))
    saved = os.environ.pop("INC_SCORER_TESTING")
    try:
        check("a testing experiment refuses to advance without INC_SCORER_TESTING=1",
              raises(lambda: D.Driver(exp2, backend=fb2, clock=clock, quiet=True).advance(),
                     D.DriverError, "INC_SCORER_TESTING"))
    finally:
        os.environ["INC_SCORER_TESTING"] = saved


def test_b0():
    print("B0 (baseline)")
    exp = "b0_t"
    fb = D.FakeBackend()
    clock = Clock()
    summary, defn, _ = P.build_b0(exp, testing=True, backend=fb, quiet=True)
    paths = D.Paths(exp)
    check("B0: train_core copied byte for byte",
          C.sha256_file(paths.manifests / "train_core.jsonl") == C.sha256_file(C.manifest_path("train_core"))
          and defn["type"] == "baseline")
    check("B0: 3 cold base runs, dev only",
          len(fb.submissions) == 1 and fb.submissions[0]["n"] == 3
          and all(json.loads(pathlib.Path(s).read_text())["exams"] == ["dev"]
                  and json.loads(pathlib.Path(s).read_text())["recipe"] == dict(P.cold_recipe(), seed=i)
                  for i, s in enumerate(fb.submissions[0]["specs"])))
    ex = Executor()
    drive(exp, fb, clock, ex)
    st = state_of(exp)
    finals = [s for s in all_specs(exp) if s["kind"] == "final"]
    check("B0: final runs of every seed on every exam, test included, then done",
          st["done"] and len(finals) == 3 and all(s["exams"] == list(D.FINAL_EXAMS) for s in finals))
    s2, d2, _ = P.build_b0("b0_set", testing={"imgsz": 64, "device": "cpu"}, backend=D.FakeBackend(), quiet=True)
    check("B0 with test-mode settings: exp.json carries inc/train.py's settings object",
          json.loads(D.Paths("b0_set").exp_json.read_text())["testing"] == {"imgsz": 64, "device": "cpu"}
          and s2["testing"] is True)
    check("builders refuse test-mode settings inc/train.py does not know",
          raises(lambda: P.build_b0("b0_bad", testing={"epochs": 1}, backend=D.FakeBackend(), quiet=True),
                 P.PilotError, "not among"))
    rj = R.build(exp)
    check("B0: the report's base row has test over 3 seeds",
          rj["final"][0]["exams"]["test"]["twelve"]["n"] == 3 and rj["final"][0]["model"] == "base train_core")

# ------------------------------------------------------------- new coverage
def _write(path, text):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def test_slurm_backend():
    print("SlurmBackend against fake sbatch / squeue / sacct")
    b = TMP / "bin"
    mode = TMP / "bin_mode"
    mode.mkdir(parents=True, exist_ok=True)
    now = time.time()
    loc = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))  # noqa: E731
    _write(b / "sbatch", "#!/bin/bash\nm=$(cat %s/sbatch)\n"
                         "case $m in\n"
                         " ok) echo 'sbatch: warning: memory per cpu raised'; echo '4242;bridges2';;\n"
                         " socket) echo 'sbatch: error: Batch job submission failed: Socket timed out on "
                         "send/recv operation' >&2; exit 1;;\n"
                         " words) echo 'Submitted batch job 4242';;\n"
                         " slow) sleep 5; echo 1;;\n"
                         "esac\n" % mode)
    _write(b / "squeue", "#!/bin/bash\nm=$(cat %s/squeue)\n"
                         "case $m in\n down) exit 1;;\n empty) ;;\n has) echo '555_[0-2%%40]|%s';;\n"
                         " old) echo '333_[0-2%%40]|%s';;\nesac\n" % (mode, loc(now + 5), loc(now - 7200)))
    _write(b / "sacct", "#!/bin/bash\nm=$(cat %s/sacct)\n"
                        "case $m in\n down) exit 1;;\n empty) ;;\n old) echo '444_0|%s';;\n"
                        " new) echo '556_1|%s'; echo '556_2|%s';;\nesac\n" % (mode, loc(now - 7200),
                                                                            loc(now + 5), loc(now + 5)))
    script = TMP / "job.sh"
    _write(script, "#!/bin/bash\n")
    saved = os.environ["PATH"]
    os.environ["PATH"] = "%s:%s" % (b, saved)
    try:
        sb = D.SlurmBackend(script=script, user="u", timeout=1)
        args = ("/l/0000.txt", 3, "e", "inc_e_0000", TMP / "logs")

        def submit(m):
            (mode / "sbatch").write_text(m)
            return sb.submit(*args, env=D.submission_env(False))
        check("sbatch: the job id is read from any stdout line (a warning came first)", submit("ok") == "4242")
        check("sbatch: a non-zero exit (e.g. socket timeout after queueing) leaves the outcome open",
              raises(lambda: submit("socket"), D.SubmitUncertain, "Socket timed out")
              and not issubclass(D.SubmitUncertain, D.SubmitNotStarted))
        check("sbatch: rc 0 without a job id leaves the outcome open",
              raises(lambda: submit("words"), D.SubmitUncertain, "not a job id"))
        check("sbatch: a client timeout leaves the outcome open",
              raises(lambda: submit("slow"), D.SubmitUncertain, "no answer"))
        check("sbatch: a missing job script means nothing was submitted",
              raises(lambda: D.SlurmBackend(script=TMP / "nope.sh").submit(*args), D.SubmitNotStarted))

        def lookup(sq, sa, since=now - 60):
            (mode / "squeue").write_text(sq)
            (mode / "sacct").write_text(sa)
            return sb.lookup("inc_e_0000", since)
        check("lookup: squeue has it -> found", lookup("has", "down") == (D.FOUND, "555"))
        check("lookup: squeue unreachable -> unknown, never absent", lookup("down", "empty")[0] == D.UNKNOWN)
        check("lookup: squeue empty, sacct unreachable -> unknown", lookup("empty", "down")[0] == D.UNKNOWN)
        check("lookup: sacct has it after the submission -> found", lookup("empty", "new") == (D.FOUND, "556"))
        check("lookup: an older job with the same name (a deleted, rebuilt experiment) is not adopted",
              lookup("old", "old") == (D.ABSENT, None))
        check("lookup: both answer and neither has it -> absent", lookup("empty", "empty") == (D.ABSENT, None))
    finally:
        os.environ["PATH"] = saved
    check("parse_sacct: JobID, JobIDRaw, State rows map both ids; plain jobs kept",
          D.parse_sacct("77_4|81|FAILED\n77_[6-9]|77|PENDING\n90|90|RUNNING\n")
          == {"77_4": "FAILED", "81": "FAILED", "90": "RUNNING"})
    check("parse_squeue: '%i %A' lines give array ids and raw ids",
          D.parse_squeue("77_4 81\n77_5 82\n") == {"77_4", "81", "77_5", "82"})
    check("parse_parsable: warnings after the id too", D.parse_parsable("123\nsbatch: warning: x\n") == "123")


class Die(BaseException):
    """The process dies (e.g. killed at its time limit): nothing after this runs."""


class DieAfterSbatch(D.FakeBackend):
    """sbatch queues the array (or not), then the pass dies before saving."""

    def __init__(self, queue=True, lookup_result=None):
        super().__init__()
        self.die_next, self.queue, self.lookup_result = True, queue, lookup_result

    def submit(self, *a, **k):
        if self.die_next:
            self.die_next = False
            if self.queue:
                super().submit(*a, **k)
            raise Die()
        return super().submit(*a, **k)

    def lookup(self, job_name, since_ts=None):
        if self.lookup_result is not None:
            return self.lookup_result
        return super().lookup(job_name, since_ts)


def test_submission_recovery(rows):
    print("submissions whose outcome is uncertain")
    clock = Clock()
    # (a) dies after sbatch queued the array: found by name, tracked, never submitted twice
    exp = "subm_found"
    fb = DieAfterSbatch()
    check("a pass that dies right after sbatch leaves the submission 'submitting' on disk",
          raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows)), Die)
          and state_of(exp)["submissions"][0]["status"] == "submitting"
          and all(r["status"] == "submitting" for r in state_of(exp)["runs"].values()))
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("... the next advance finds the job by name and tracks it; one array only",
          len(fb.submissions) == 1 and st["submissions"][0]["status"] == "submitted"
          and st["submissions"][0]["job_id"] == "1000"
          and [r["job"] for r in st["runs"].values()] == ["1000_0", "1000_1", "1000_2"])
    drive(exp, fb, clock, Executor())
    check("... and the experiment finishes with one array per run", state_of(exp)["done"] is True
          and all(len([t for t in fb.tasks.values() if t["spec"].endswith("/%s/spec.json" % rid)]) == 1
                  for rid in state_of(exp)["runs"]))

    # (b) Slurm unreachable: the submission is left, however long it takes
    exp = "subm_unknown"
    fb = DieAfterSbatch(lookup_result=(D.UNKNOWN, "sacct exited 1: slurmdbd down"))
    raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows)), Die)
    for dt in (60, 3 * 3600):
        clock.t += dt
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("sacct / squeue unreachable: never marked lost, never resubmitted (even hours later)",
          len(fb.submissions) == 1 and st["submissions"][0]["status"] == "submitting"
          and st["submissions"][0]["last_lookup"]["result"] == D.UNKNOWN
          and any("sbatch outcome not known" in ln for ln in D.Driver(exp).status()))
    fb.lookup_result = None
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("... and resolved once Slurm answers", state_of(exp)["submissions"][0]["status"] == "submitted"
          and len(fb.submissions) == 1)

    # (c) died before Slurm got it: absent, requeued only after the grace period
    exp = "subm_absent"
    fb = DieAfterSbatch(queue=False)
    raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows)), Die)
    clock.t += 60
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("absent within the grace period (slurmdbd may lag): left 'submitting', nothing resubmitted",
          len(fb.submissions) == 0 and st["submissions"][0]["status"] == "submitting")
    clock.t += D.SUBMIT_GRACE_SECONDS + 1
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("absent after the grace period: 'lost', its runs go out once in a new array",
          len(fb.submissions) == 1 and st["submissions"][0]["status"] == "lost"
          and st["submissions"][1]["status"] == "submitted" and fb.submissions[0]["n"] == 3)

    # (d) sbatch times out after queueing (the review's probe_dup): no duplicate
    class TimeoutAfterQueue(D.FakeBackend):
        fail_next = True

        def submit(self, *a, **k):
            jid = super().submit(*a, **k)
            if self.fail_next:
                self.fail_next = False
                raise D.SubmitUncertain("sbatch gave no answer within 180 s")
            return jid
    exp = "subm_timeout"
    fb = TimeoutAfterQueue()
    check("an sbatch timeout is reported, and the runs stay 'submitting'",
          raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(
              small_defn(exp, rows, step_names=("A",))), D.DriverError, "looks it up by name")
          and all(r["status"] == "submitting" for r in state_of(exp)["runs"].values()))
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("... the next advance adopts the queued array: one array, tracked",
          [(x["job_id"], x["n"]) for x in fb.submissions] == [("1000", 3)]
          and [(x["n"], x["status"], x["job_id"]) for x in st["submissions"]] == [(0, "submitted", "1000")])

    # (e) sbatch never started: requeued at once
    class NotStarted(D.FakeBackend):
        fail_next = True

        def submit(self, *a, **k):
            if self.fail_next:
                self.fail_next = False
                raise D.SubmitNotStarted("sbatch could not be started: FileNotFoundError")
            return super().submit(*a, **k)
    exp = "subm_notstarted"
    fb = NotStarted()
    raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows)), D.DriverError)
    st = state_of(exp)
    check("an sbatch that never started: the submission is an error and its runs are queued again",
          st["submissions"][0]["status"] == "error" and all(r["status"] == "queued" for r in st["runs"].values()))
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("... and the next advance submits them", len(fb.submissions) == 1 and fb.submissions[0]["n"] == 3
          and all(r["status"] == "submitted" for r in state_of(exp)["runs"].values()))

    # (f) sbatch refused every time (e.g. a missing job script): not an endless loop
    class NeverStarts(D.FakeBackend):
        def submit(self, *a, **k):
            raise D.SubmitNotStarted("job script /x/run_inc_job.sh not found (set INC_JOB_SCRIPT)")
    exp = "subm_never"
    fb = NeverStarts()
    for _ in range(D.MAX_SUBMIT_FAILURES + 2):
        call = (lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows))) \
            if not D.Paths(exp).state.exists() else (lambda: D.Driver(exp, backend=fb, clock=clock,
                                                                         quiet=True).advance())
        raises(call, D.DriverError)
    st = state_of(exp)
    check("sbatch refused %d times: the runs fail with the sbatch error and block their unit, no more "
          "submissions" % D.MAX_SUBMIT_FAILURES,
          all(r["status"] == "failed" and "job script" in r["error"] for r in st["runs"].values())
          and "base" in st["blocked"] and "job script" in st["blocked"]["base"]["error"]
          and len(st["submissions"]) == D.MAX_SUBMIT_FAILURES, (len(st["submissions"]), st["blocked"]))


class FinishesDuringSqueue(D.FakeBackend):
    """Every pending task finishes while squeue is being asked."""

    def __init__(self, ex):
        super().__init__()
        self.ex, self.armed = ex, False

    def queued(self):
        if self.armed:
            self.armed = False
            for t in self.pending():
                self.tasks[t]["state"] = self.ex(_spec(self.tasks[t]["spec"]))
        return super().queued()


def _spec(path):
    return json.loads(pathlib.Path(path).read_text())


class BadOnce(Executor):
    """The first attempt of a run in self.bad leaves a given broken output."""

    def __init__(self, bad):
        super().__init__()
        self.bad = dict(bad)

    def __call__(self, spec):
        rid = spec["run_id"]
        mode = self.bad.pop(rid, None)
        if mode is None:
            return super().__call__(spec)
        state = super().__call__(spec)
        self.attempts[rid] = 1
        out = pathlib.Path(spec["out_dir"])
        if mode == "unreadable":
            (out / "run.json").write_text("{not json")
        elif mode == "scores":
            shutil.rmtree(out / "scores")
        elif mode == "weights":
            (out / "weights" / "final.pt").unlink()
        return state


def test_failure_counted_once(rows):
    print("every failure branch is counted once, however many advances before the retry")
    clock = Clock()
    for mode in ("failed", "unreadable", "scores", "weights", "ended", "vanished"):
        exp = "cnt_%s" % mode
        fb = D.FakeBackend()
        if mode == "failed":
            ex = Executor()
            ex.fail_once.add("base__s1")
        elif mode == "ended":
            ex = Executor()
            ex.end_failed_once.add("base__s1")
        elif mode == "vanished":
            ex = Executor()
            ex.lose_once.add("base__s1")
        else:
            ex = BadOnce({"base__s1": mode})
        D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
        fb.run_pending(ex)
        if mode == "vanished":
            clock.t += D.STALE_SECONDS + 1
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        r1 = state_of(exp)["runs"]["base__s1"]
        for _ in range(2):                           # the retry is still pending
            clock.t += 60
            D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        r = state_of(exp)["runs"]["base__s1"]
        check("%s: counted once (attempt 2, one history entry) across two more advances, retry pending" % mode,
              r1["attempt"] == 2 and r["attempt"] == 2 and len(r["history"]) == 1 and r["status"] == "submitted"
              and "base" not in state_of(exp)["blocked"], (r["attempt"], len(r["history"]), r["status"]))
        fb.run_pending(ex)
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        r = state_of(exp)["runs"]["base__s1"]
        check("%s: the retry completes" % mode, r["status"] == "complete" and r["attempt"] == 2, r["status"])

    exp = "race_t"
    ex = Executor()
    fb = FinishesDuringSqueue(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    fb.armed = True
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("a run.json that appears between the first look and squeue is collected, not failed",
          all(st["runs"][x]["status"] == "complete" and not st["runs"][x]["history"]
              for x in ("base__s0", "base__s1", "base__s2")))


def test_duplicate_task(rows):
    print("a duplicate task that exits busy; a late done run.json")
    clock = Clock()
    exp = "busy_t"
    fb = D.FakeBackend()
    ex = Executor()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    paths = D.Paths(exp)
    # an untracked job (77777) holds base__s0's run dir; the tracked task exits 3 without run.json
    fb.tasks["999_0"] = {"spec": str(paths.spec("base__s0")), "state": "RUNNING", "job_name": "x", "raw": "77777"}
    paths.run_dir("base__s0").mkdir(parents=True, exist_ok=True)
    paths.attempt_json("base__s0").write_text(json.dumps({"attempt": 1, "slurm_job_id": "77777",
                                                          "started_utc": D._utc(clock())}))
    fb.tasks["1000_0"]["state"] = "FAILED"
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    r = state_of(exp)["runs"]["base__s0"]
    check("a task that ended without run.json while attempt.json names a live job: re-tracked, not counted",
          r["status"] == "submitted" and r["attempt"] == 1 and not r["history"] and r["job"] == "77777"
          and r["events"][0]["event"] == "retracked" and len(fb.submissions) == 1, r)
    fb.tasks["999_0"]["state"] = ex(_spec(paths.spec("base__s0")))
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    r = state_of(exp)["runs"]["base__s0"]
    check("... and the holder's done run.json completes the run", r["status"] == "complete" and r["attempt"] == 1)

    exp = "busy_dead"
    fb = D.FakeBackend()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    paths = D.Paths(exp)
    fb.tasks["999_0"] = {"spec": str(paths.spec("base__s0")), "state": "COMPLETED", "job_name": "x", "raw": "77777"}
    paths.run_dir("base__s0").mkdir(parents=True, exist_ok=True)
    paths.attempt_json("base__s0").write_text(json.dumps({"attempt": 1, "slurm_job_id": "77777"}))
    fb.tasks["1000_0"]["state"] = "FAILED"
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    r = state_of(exp)["runs"]["base__s0"]
    check("attempt.json naming a job that has ended: the failure is counted as before",
          r["attempt"] == 2 and len(r["history"]) == 1 and "ended FAILED" in r["history"][0]["error"])

    # the executor's owner file (inc/train.py: heartbeat every 30 s, stale after 300 s)
    for case in ("slurm", "outside", "stale"):
        exp = "owner_%s" % case
        fb = D.FakeBackend()
        D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
        paths = D.Paths(exp)
        paths.run_dir("base__s1").mkdir(parents=True, exist_ok=True)
        if case == "slurm":
            fb.tasks["998_0"] = {"spec": str(paths.spec("base__s1")), "state": "RUNNING", "job_name": "x",
                                 "raw": "88888"}
        owner = {"host": "gpu-node-7", "pid": 4242, "token": "abc", "started_utc": D._utc(),
                 "slurm_job_id": "88888" if case == "slurm" else None}
        paths.owner("base__s1").write_text(json.dumps(owner))
        if case == "stale":
            old = time.time() - D.OWNER_FRESH_SECONDS - 30
            os.utime(paths.owner("base__s1"), (old, old))
        fb.tasks["1000_1"]["state"] = "FAILED"
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        r = state_of(exp)["runs"]["base__s1"]
        if case == "slurm":
            check("a fresh owner file naming a live job: re-tracked to it, not counted",
                  r["job"] == "88888" and r["attempt"] == 1 and not r["history"])
        elif case == "outside":
            D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
            r = state_of(exp)["runs"]["base__s1"]
            check("a fresh owner file outside Slurm: waited for (noted once), not counted",
                  r["status"] == "submitted" and r["attempt"] == 1 and not r["history"]
                  and [e["event"] for e in r["events"]] == ["held"])
        else:
            check("a stale owner file (no heartbeat): the failure is counted",
                  r["attempt"] == 2 and len(r["history"]) == 1)
    try:
        from weed_optimizer_framework.tools.inc import train as TR
        check("the owner file's name and staleness match inc/train.py's",
              TR.OWNER_NAME == D.OWNER_FILE and TR.LOCK_STALE_SECONDS == D.OWNER_FRESH_SECONDS)
    except ImportError as e:
        print("  skip inc/train.py owner-file cross-check: %s" % e)

    exp = "late_t"
    fb = D.FakeBackend()
    ex = Executor()
    rid = "full__s01_A__cand__s1"
    ex.fail_always.add(rid)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    for _ in range(4):
        fb.run_pending(ex)
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    check("(setup) two failures block the chain", st["runs"][rid]["status"] == "failed"
          and "chain:full" in st["blocked"] and st["blocked"]["chain:full"]["cause"]["run_id"] == rid)
    ex.fail_always.discard(rid)
    ex(_spec(D.Paths(exp).spec(rid)))              # a duplicate finished it after all
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    st = state_of(exp)
    unb = [e for e in ledger_of(exp) if e["type"] == "unblock"]
    check("a later done run.json recovers the failed run and lifts the block it caused (ledger, auto)",
          st["runs"][rid]["status"] == "complete" and "chain:full" not in st["blocked"]
          and st["chains"]["full"]["phase"] in ("step", "soup") and len(unb) == 1 and unb[0]["auto"] is True
          and unb[0]["unit"] == "chain:full", (st["runs"][rid]["status"], st["blocked"]))
    drive(exp, fb, clock, ex)
    check("... and the experiment finishes", state_of(exp)["done"] is True)


class BumpsGeneration(D.FakeBackend):
    """Another pass saves state while this one asks squeue."""
    exp = None
    armed = False

    def queued(self):
        if self.armed:
            self.armed = False
            p = D.Paths(self.exp).state
            st = json.loads(p.read_text())
            st["generation"] = st.get("generation", 0) + 1
            p.write_text(json.dumps(st))
        return super().queued()


class StealsLease(D.FakeBackend):
    exp = None
    armed = False

    def queued(self):
        if self.armed:
            self.armed = False
            D.Paths(self.exp).lease.write_text(json.dumps({"token": "someone-else", "host": "other-node",
                                                           "pid": 1, "expires_ts": time.time() + 600}))
        return super().queued()


class NestedAdvance(D.FakeBackend):
    """While this pass runs, another advance starts (a task ended on another node)."""
    exp = None
    armed = False
    inner = None

    def queued(self):
        if self.armed:
            self.armed = False
            self.inner = D.Driver(self.exp, backend=self, clock=Clock(), quiet=True).advance()
        return super().queued()


def test_exclusion(rows):
    print("exclusion: the lease, the generation fence, a mount without flock")
    clock = Clock()
    exp = "lease_t"
    fb = D.FakeBackend()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    fb.run_pending(Executor())
    paths = D.Paths(exp)
    check("no lease is left behind by a finished advance", not paths.lease.exists())
    before = paths.state.read_bytes()
    paths.lease.write_text(json.dumps({"token": "t1", "host": "some-other-node", "pid": 1,
                                       "expires_ts": time.time() + 600}))
    res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("a live lease held on another node: advance returns at once, touching nothing",
          res["locked"] is True and res["passes"] == 0 and paths.state.read_bytes() == before
          and len(fb.submissions) == 1 and paths.request.exists())
    paths.lease.write_text(json.dumps({"token": "t1", "host": "some-other-node", "pid": 1,
                                       "expires_ts": time.time() - 1}))
    res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("an expired lease is broken and the pass runs", res["passes"] == 1 and len(fb.submissions) == 2
          and not paths.lease.exists())
    import subprocess
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    paths.lease.write_text(json.dumps({"token": "t2", "host": socket.gethostname(), "pid": dead.pid,
                                       "expires_ts": time.time() + 600}))
    res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("a lease whose holder pid on this host is dead is broken at once", res["passes"] == 1)
    paths.lease.write_text("{partial")
    res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("an unreadable fresh lease (its writer may be mid-write) counts as held", res["locked"] is True)
    old = time.time() - D.LEASE_SECONDS - 5
    os.utime(paths.lease, (old, old))
    res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("an unreadable lease older than the lease period is broken", res["passes"] == 1)

    for cls, what, contains in ((BumpsGeneration, "another pass saved state", "generation"),
                                (StealsLease, "another pass took the lease", "lost the advance lease")):
        exp = "fence_%s" % cls.__name__
        fb = cls()
        fb.exp = exp
        D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
        fb.run_pending(Executor(), hold=lambda spec: spec["run_id"] == "base__s2")   # squeue gets asked
        on_disk = json.loads(D.Paths(exp).state.read_text())
        fb.armed = True
        ok = raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.ConcurrentPass, contains)
        after = json.loads(D.Paths(exp).state.read_text())
        check("%s mid-pass: this pass stops without saving or submitting" % what,
              ok and len(fb.submissions) == 1 and after["runs"] == on_disk["runs"]
              and not [e for e in ledger_of(exp) if e["type"] == "gate"])
        if cls is StealsLease:
            check("... and leaves the other holder's lease in place",
                  json.loads(D.Paths(exp).lease.read_text())["token"] == "someone-else")
            D.Paths(exp).lease.unlink()

    exp = "noflock_t"
    fb = NestedAdvance()
    fb.exp = exp
    real_flock = D.fcntl.flock

    def no_flock(fd, op):
        raise OSError(errno.ENOSYS, "Function not implemented")
    D.fcntl.flock = no_flock
    try:
        D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
        fb.run_pending(Executor(), hold=lambda spec: spec["run_id"] == "base__s2")
        fb.armed = True
        res = D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    finally:
        D.fcntl.flock = real_flock
    st = state_of(exp)
    step_arrays = [x for x in fb.submissions if any("__s01_A__" in sp for sp in x["specs"])]
    check("a mount without flock (ENOSYS): the lease alone excludes a concurrent advance; one array",
          res["passes"] >= 1 and fb.inner["locked"] is True and len(step_arrays) == 1
          and step_arrays[0]["n"] == 6 and len(st["submissions"]) == 2
          and all(x["status"] == "submitted" for x in st["submissions"]))


def test_definitions(rows):
    print("definitions: disjointness, manifests, seeds, truth recipe")
    fb = D.FakeBackend()
    clock = Clock()

    def refused(exp, defn, contains):
        ok = raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).init(defn), D.DriverError, contains)
        return ok and not D.Paths(exp).exp_json.exists() and not D.Paths(exp).state.exists()

    d = small_defn("ov_base", rows)
    over = rows[55:65]                               # 5 base rows + 5 step rows
    p = TMP / "small" / "ov_base" / "A.jsonl"
    d["steps"][0].update(manifest_sha256=C.write_manifest(p, over), n_images=len(over))
    check("init refuses an increment that shares images with the base, writing nothing",
          refused("ov_base", d, "shares 5 image(s) with the base"))
    d = small_defn("ov_step", rows)
    p = TMP / "small" / "ov_step" / "B.jsonl"
    rs = rows[65:75]
    d["steps"][1].update(manifest_sha256=C.write_manifest(p, rs), n_images=len(rs))
    check("init refuses two steps that share images", refused("ov_step", d, "with step A"))
    d = small_defn("ov_sha", rows)
    copy = dict(rows[3], key="copy_of_base_3", image=str(TMP / "copy3.png"))
    shutil.copyfile(rows[3]["image"], copy["image"])
    p = TMP / "small" / "ov_sha" / "A.jsonl"
    rs = rows[60:69] + [copy]
    d["steps"][0].update(manifest_sha256=C.write_manifest(p, rs), n_images=len(rs))
    check("init refuses a byte copy of a base image under another key and path (image sha256)",
          refused("ov_sha", d, "shares 1 image(s)"))
    d = small_defn("bad_sha", rows)
    d["steps"][1]["manifest_sha256"] = "0" * 64
    check("init refuses a manifest that does not hash as recorded, writing nothing",
          refused("bad_sha", d, "hashes to"))
    check("a chain with fewer than 3 seeds is refused", refused("two_seeds", dict(small_defn("two_seeds", rows),
                                                                                seeds=[0, 1]), "at least 3"))
    d = small_defn("tr_rec", rows, truth=True)
    d["truth_recipe"] = dict(d["truth_recipe"], epochs=50)
    check("a truth recipe other than the base recipe is refused", refused("tr_rec", d, "truth_recipe"))

    # the runtime checks, with the init check switched off (a definition from before it existed)
    real = D.check_definition_data
    D.check_definition_data = lambda defn: None
    try:
        exp = "rt_disjoint"
        d = small_defn(exp, rows, truth=True)
        p = TMP / "small" / exp / "B.jsonl"
        rs = [dict(r, key="again__" + r["key"]) for r in rows[20:30]]   # base images under new keys
        d["steps"][1].update(manifest_sha256=C.write_manifest(p, rs), n_images=len(rs))
        D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
    finally:
        D.check_definition_data = real
    st = state_of(exp)
    check("runtime: an overlapping increment blocks the truth arm with no truth run created at all",
          "truth" in st["blocked"] and "shares" in st["blocked"]["truth"]["error"]
          and not [x for x in st["runs"] if x.startswith("truth__")] and not st["truth"].get("created"))
    ex = Executor()
    drive(exp, fb, clock, ex)
    st = state_of(exp)
    check("runtime: an increment overlapping the chain's accepted pool blocks the chain, no run created",
          st["chains"]["full"]["phase"] == "blocked" and "accepted pool" in st["blocked"]["chain:full"]["error"]
          and st["chains"]["full"]["steps"]["1"]["decision"]["verdict"] == "ACCEPT"
          and not [x for x in st["runs"] if "__s02_B__" in x])

    exp = "truth_atomic"
    d = small_defn(exp, rows, truth=True)
    junk = D.Paths(exp).run_dir("truth__s02_B__union__s1")
    junk.mkdir(parents=True)
    (junk / "stray.txt").write_text("left by someone else")
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
    st = state_of(exp)
    check("the truth arm is created all or none: a failure at step 2 leaves no step-1 truth run behind",
          "truth" in st["blocked"] and "refusing to reuse" in st["blocked"]["truth"]["error"]
          and not [x for x in st["runs"] if x.startswith("truth__")]
          and not [x for sub in fb.submissions for x in sub["specs"] if "/truth__" in x and "/%s/" % exp in x])

    exp = "edited_t"
    d = small_defn(exp, rows)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(d)
    pb = pathlib.Path(d["steps"][1]["manifest"])
    pb.write_text("".join(pb.read_text().splitlines(True)[:-1]))
    drive(exp, fb, clock, Executor())
    st = state_of(exp)
    check("a manifest edited after init blocks the chain when it is used, no run created",
          "changed since the experiment was defined" in st["blocked"]["chain:full"]["error"]
          and not [x for x in st["runs"] if "__s02_B__" in x])


def test_code_pin(rows):
    print("code pinning")
    clock = Clock()
    exp = "code_t"
    fb = D.FakeBackend()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    paths = D.Paths(exp)
    st = json.loads(paths.state.read_text())
    real_gate = st["code"]["modules"]["tools/inc/gate.py"]
    st["code"]["modules"]["tools/inc/gate.py"] = "0" * 64      # the code the experiment was pinned to
    paths.state.write_text(json.dumps(st))
    fb.run_pending(Executor())
    before = paths.state.read_bytes()
    check("a pass refuses when the gate differs from the pinned hash, changing nothing",
          raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.DriverError,
                 "changed since") and paths.state.read_bytes() == before and len(fb.submissions) == 1)
    check("repin needs a reason", raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).repin(""),
                                         D.DriverError, "reason"))
    D.Driver(exp, backend=fb, clock=clock, quiet=True).repin("gate.py fix reviewed")
    rp = [e for e in ledger_of(exp) if e["type"] == "code_repin"]
    check("repin records old and new hashes and the reason in the ledger",
          len(rp) == 1 and rp[0]["changed"] == ["tools/inc/gate.py"] and rp[0]["old"]["tools/inc/gate.py"] == "0" * 64
          and rp[0]["new"]["tools/inc/gate.py"] == real_gate and rp[0]["reason"] == "gate.py fix reviewed")
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("... after which the experiment advances", len(fb.submissions) == 2)

    nested = C.REPO / "weed_llm_benchmark" / "weed_optimizer_framework"
    for m in D.PINNED_MODULES:
        dst = nested / m
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(D.package_dir() / m, dst)
    with open(nested / "tools/inc/gate.py", "a") as fh:
        fh.write("\n# edited in the git-tracked copy only\n")
    try:
        fb.run_pending(Executor())
        check("a pass refuses when the running copy differs from the git-tracked nested copy",
              raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.DriverError,
                     "git-tracked copy"))
        os.environ["INC_ALLOW_DRIFT"] = "1"
        try:
            D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        finally:
            os.environ.pop("INC_ALLOW_DRIFT")
        g = [e for e in ledger_of(exp) if e["type"] == "gate"]
        check("INC_ALLOW_DRIFT=1 lets it through, and the ledger entry records the drift",
              len(g) == 1 and g[0]["code"]["drift"] == ["tools/inc/gate.py"])
    finally:
        shutil.rmtree(C.REPO / "weed_llm_benchmark")


def test_operations(rows):
    print("operations: transient reads, unblock, watch, a partial ledger line, CLI errors")
    clock = Clock()
    exp = "transient_t"
    fb = D.FakeBackend()
    ex = Executor()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    fb.run_pending(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    fb.run_pending(ex)
    real = D.Driver._score_inputs
    calls = {"n": 0}

    def flaky(pairs):
        if calls["n"] < 2:
            calls["n"] += 1
            raise OSError(errno.EIO, "Input/output error (simulated)")
        return real(pairs)
    D.Driver._score_inputs = staticmethod(flaky)
    try:
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        st1 = state_of(exp)
        lines = D.Driver(exp).status()
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    finally:
        D.Driver._score_inputs = staticmethod(real)
    st = state_of(exp)
    check("an unreadable score file is retried on the next pass, not blocked (shown in status)",
          "chain:full" not in st1["blocked"] and st1["transient"]["chain:full"]["count"] == 1
          and any("retrying" in ln for ln in lines)
          and st["chains"]["full"]["steps"]["1"]["decision"]["verdict"] == "ACCEPT"
          and "chain:full" not in st["transient"] and not st["blocked"])
    exp = "transient_block"
    fb = D.FakeBackend()
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    fb.run_pending(ex)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    fb.run_pending(ex)

    def broken(pairs):
        raise OSError(errno.EIO, "Input/output error (simulated)")
    D.Driver._score_inputs = staticmethod(broken)
    try:
        for _ in range(D.TRANSIENT_LIMIT):
            D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    finally:
        D.Driver._score_inputs = staticmethod(real)
    st = state_of(exp)
    check("... and blocked only after TRANSIENT_LIMIT passes in a row",
          "chain:full" in st["blocked"] and "passes in a row" in st["blocked"]["chain:full"]["error"])

    exp = "unblock_t"
    fb = D.FakeBackend()
    ex = Executor()
    rid = "full__s01_A__null__s2"
    ex.fail_always.add(rid)
    D.Driver(exp, backend=fb, clock=clock, quiet=True).init(small_defn(exp, rows, step_names=("A",)))
    for _ in range(4):
        fb.run_pending(ex)
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
    check("(setup) the chain is blocked", "chain:full" in state_of(exp)["blocked"])
    check("status: WAITING when nothing runs and a unit is blocked",
          any("WAITING" in ln and "unblock" in ln for ln in D.Driver(exp).status()))
    drv = lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True)  # noqa: E731
    check("unblock needs a reason and a blocked unit",
          raises(lambda: drv().unblock(["chain:full"], ""), D.DriverError, "reason")
          and raises(lambda: drv().unblock(["truth"], "x"), D.DriverError, "not blocked"))
    ex.fail_always.discard(rid)
    n_sub = len(fb.submissions)
    drv().unblock(["chain:full"], "the synthetic failure was fixed")
    st = state_of(exp)
    unb = [e for e in ledger_of(exp) if e["type"] == "unblock"]
    check("unblock: the block is lifted, the failed run resubmitted with fresh attempts, recorded",
          not st["blocked"] and st["chains"]["full"]["phase"] == "step" and st["runs"][rid]["attempt"] == 3
          and st["runs"][rid]["status"] == "submitted" and len(fb.submissions) == n_sub + 1
          and len(unb) == 1 and unb[0]["auto"] is False and unb[0]["reset_runs"] == [rid]
          and unb[0]["reason"] == "the synthetic failure was fixed")

    out = []
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock.t += s
        fb.run_pending(ex)
    code = D.watch(exp, interval=60, max_hours=2, backend=fb, sleep=sleep, clock=clock, out=out.append)
    check("watch: advances every interval until done", code == 0 and state_of(exp)["done"] is True
          and sleeps and set(sleeps) == {60} and any("done: yes" in ln for ln in out))
    rep = R.build(exp)
    md = (D.Paths(exp).root / "report.md").read_text()
    check("report: interventions and the pinned code are shown",
          "## Interventions" in md and "unblock chain:full (by hand)" in md and "Decision code pinned" in md
          and len(rep["interventions"]) == 1)

    paths = D.Paths(exp)
    good = paths.ledger.read_bytes()
    with open(paths.ledger, "ab") as fh:
        fh.write(b'{"id": "gate/full/9", "type": "gate", "decision": {"verd')
    rep = R.build(exp)
    check("report: a partial last ledger line is skipped with a note",
          any("partial" in n for n in rep["notes"]) and "partial" in (paths.root / "report.md").read_text())
    exp2 = "partial_t"
    fb2 = D.FakeBackend()
    D.Driver(exp2, backend=fb2, clock=clock, quiet=True).init(small_defn(exp2, rows, step_names=("A",)))
    fb2.run_pending(ex)
    p2 = D.Paths(exp2)
    before = p2.ledger.read_bytes()
    with open(p2.ledger, "ab") as fh:
        fh.write(b'{"id": "gate/full/1", "type": "gate", "inpu')
    D.Driver(exp2, backend=fb2, clock=clock, quiet=True).advance()
    frags = list(p2.root.glob("ledger.partial.*.txt"))
    check("advance repairs a partial last ledger line (fragment kept) and goes on",
          p2.ledger.read_bytes().startswith(before) and p2.ledger.read_bytes().endswith(b"\n")
          and len(frags) == 1 and frags[0].read_bytes() == b'{"id": "gate/full/1", "type": "gate", "inpu'
          and len(fb2.submissions) == 2)
    paths.ledger.write_bytes(b"{broken middle line\n" + good)
    check("a corrupt line that is not the last one stops advance with a clear error",
          raises(lambda: D.Driver(exp, backend=fb, clock=clock, quiet=True).advance(), D.DriverError, "not JSON"))
    paths.state.write_text("{not json")
    saved_err, saved_out = sys.stderr, sys.stdout
    sys.stderr = sys.stdout = open(os.devnull, "w")
    try:
        rc = D.main(["advance", "--exp", exp])
        rc_unblock = D.main(["unblock", "--exp", exp2, "--all", "--reason", "r"])
        rc_repin = D.main(["repin", "--exp", exp2, "--reason", "r"])
        rc_status = D.main(["status", "--exp", exp2])
    finally:
        sys.stderr.close()
        sys.stderr, sys.stdout = saved_err, saved_out
    check("the CLI reports a ValueError (a corrupt state.json) as an error, exit 1, no traceback", rc == 1)
    check("the CLI: unblock with nothing blocked fails cleanly, repin with unchanged code is a no-op",
          rc_unblock == 1 and rc_repin == 0 and rc_status == 0
          and not [e for e in ledger_of(exp2) if e["type"] in ("unblock", "code_repin")])


def main():
    world = make_world()
    test_units()
    exp, fb, inc_keys = test_pilot_build(world)
    test_full_pilot(exp, fb, inc_keys)
    test_replay_default(world["rows"])
    test_full_rehearsal(world, exp)
    test_full_rehearsal_divergent(world["rows"])
    test_gate_block(world["rows"])
    test_gate_builders(world, exp)
    test_gate_pin(world["rows"])
    test_idempotent_and_lock(world["rows"])
    test_crash_recovery(world["rows"])
    test_retry_block(world["rows"])
    test_squeue_glitch(world["rows"])
    test_production_and_env(world["rows"])
    test_b0()
    test_slurm_backend()
    test_submission_recovery(world["rows"])
    test_failure_counted_once(world["rows"])
    test_duplicate_task(world["rows"])
    test_exclusion(world["rows"])
    test_definitions(world["rows"])
    test_code_pin(world["rows"])
    test_operations(world["rows"])


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
