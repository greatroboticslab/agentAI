#!/usr/bin/env python3
"""The INC autopilot's cluster verbs (inc_autopilot/remote.py) and the build
job script (run_inc_build.sh): docs/INC_AUTOPILOT.md, components 4 and 5.

No cluster: a temporary INC_DIR, the driver on FakeBackend with a synthetic
executor (the pattern of tests/test_inc_driver.py), fake sbatch / squeue
executables, and a stub package tree for the job script.

Pinned:
  * the marker: every verb prints exactly one "INCAP <json>" line on stdout,
    whatever happens (bad verb, bad arguments, a crash), exit 0 iff ok;
    parse_marked skips login noise, parse_one wants exactly one record;
  * test blindness: a snapshot's "decision" section holds no key named after
    a non-dev exam; report.json's final table goes to display_only whole and
    only its dev column (rows of exactly model, runs, exams.dev) into
    decision; a sentinel planted in a final run's scores/test.json reaches
    display_only, never decision; perturbing every leaf under a test / ood22
    / ood23 / imageweeds key (means, sd, n) and flipping the final rows'
    production flags (pilot_v1's report), and every numeric field and the
    production flag of a real driver experiment's non-dev score files
    (re-reported, so report.py derives production from test too), leaves
    the decision bytes identical; snapshot never opens runs/ and never ships
    report.md;
  * snapshot: exp, state (runs summarised), report, build summary, ledger with
    line numbers, --ledger-from with the prefix sha256, through_sha256 (the
    next read's prefix), N:SHA256 re-read from 0 on a rewritten prefix, the
    per-call ledger cap, a partial last line, the per-file cap, provenance,
    audit outputs, pilot_v1's real hand-run audit (INC_DIR/audit/
    pilot_v1_audit.json) shipped as pilot_v1/audit/label_audit.json and read
    by evidence, a not-yet-built experiment;
  * Step 1: select_clusters.csv aggregated per source and set through the two
    select manifests (status counts, uncertified, boxes, unmatched keys), the
    select_summary sha256 cross-check, the cache (miss, hit, invalidated),
    the big files listed but never shipped; the fixture writer; the pulled
    step1 replay fixture (skipped until it is pulled from the cluster);
  * advance / report / status / unblock on a driver experiment: results as
    data, DriverError kinds (not_built, code_pin); unblock only of a block
    whose live cause is on thresholds.json D5.transient_cause_kinds (a
    failed_run block is refused; a real transient block, made by failing the
    score reads TRANSIENT_LIMIT passes, is lifted), at most
    D5.max_auto_unblocks_per_unit per unit (reason prefix or the driver's
    auto flag), thresholds.json read with no fallback;
  * submit: the builder grammar (pilot build, pilot build-baseline, realloop
    build, relevance build, audit), lever argv forms, every refusal (flags,
    names, values, paths outside INC_DIR, symlinks, already built, queued
    duplicate under its own or the script's default name, an audit that
    already exists, squeue down, no menu, no policy row, bounds, provenance
    fields), a flag left out checked at the builder's default (pilot build
    without --replay-mode is sample replay, which L1 refuses; the real
    levers.json too), a lever's "fixed" values, any-of semantics over
    levers, a policy row's bounds, the sbatch command line and environment
    (no test-mode scoring, no drift override, INCAP_*), sbatch failure;
  * cancel: the abandonment marker (written before scancel, never on a dry
    run, with who and which approval), the jobs an in-job advance submits
    meanwhile, and its effect: advance and unblock refuse, campaign-snapshot
    skips the advance, status and snapshot show it; _actions.jsonl;
  * sync-outer: refused while a build / audit / relevance job is in flight
    (an experiment's run array is not), on any change to a protected module
    (driver.PINNED_MODULES and train.CODE_MODULES, e.g. mega_trainer.py,
    near_dup.py) while an experiment is unfinished, a change back to a pin
    allowed, an abandoned experiment holds nothing; _actions.jsonl;
  * outer-copy drift as production has it: remote.py run from a copy of the
    package placed at REPO/weed_llm_benchmark (where the driver's own drift
    check never compares), the outer copy changed in mega_trainer.py:
    advance refuses as code_drift, status names the module;
  * campaign-snapshot batches advance, report (auto), snapshot, Step 1 and
    status in one record;
  * run_inc_build.sh: the sbatch header, argument refusals, provenance of a
    good build, a builder refusal, outer/nested drift (refused, or allowed
    and recorded), a failed advance; its module list is the builders'
    import closure; the per-experiment build lock (held by a live job:
    refused, nothing written; stale: taken over; released on every exit);
    SIGTERM records status killed and the phase.

Run:  python3 tests/test_inc_ap_remote.py
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_remote_"))
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"            # tests only: the driver experiments here are testing ones
os.environ.pop("INC_JOB_SCRIPT", None)
BIN = TMP / "bin"
os.environ["INCAP_SQUEUE"] = str(BIN / "squeue")
os.environ["INCAP_SBATCH"] = str(BIN / "sbatch")
os.environ["INCAP_LEVERS_JSON"] = str(TMP / "levers.json")
os.environ["INCAP_SCRIPT_DIR"] = str(PKG_ROOT)
os.environ["INCAP_CANCEL_SETTLE"] = "0"
for k in ("INCAP_MAX_FILE_BYTES", "INCAP_MAX_LEDGER_BYTES"):
    os.environ.pop(k, None)
sys.path.insert(0, str(PKG_ROOT))

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as RM  # noqa: E402

INC = pathlib.Path(C.INC_DIR)
LOCAL_INC = PKG_ROOT / "results" / "framework" / "inc"
REPLAY_FIXTURES = PKG_ROOT / "tests" / "fixtures" / "inc_replay"
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def skip(name, why):
    print("  skip %s (%s)" % (name, why))


def dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def refused(rec, contains=None):
    return (rec.get("ok") is False and (contains is None or contains in str(rec.get("error"))))


# ------------------------------------------------------------- fake Slurm
def make_bin():
    BIN.mkdir(parents=True, exist_ok=True)
    (BIN / "squeue").write_text(
        "#!/bin/bash\n"
        "[ -f %(t)s/squeue_fail ] && { echo 'slurm_load_jobs error: Socket timed out' >&2; exit 1; }\n"
        "[ -f %(t)s/squeue.txt ] && cat %(t)s/squeue.txt\n"
        "[ -f %(t)s/squeue_next.txt ] && mv %(t)s/squeue_next.txt %(t)s/squeue.txt\n"
        "exit 0\n" % {"t": TMP})
    (BIN / "sbatch").write_text(
        "#!/bin/bash\n"
        "%(py)s - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "with open(%(calls)r, 'a') as fh:\n"
        "    fh.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': {k: v for k, v in os.environ.items()\n"
        "              if k.startswith(('INCAP_', 'INC_', 'SLURM_'))}}) + '\\n')\n"
        "PY\n"
        "[ -f %(t)s/sbatch_fail ] && { echo 'sbatch: error: Batch job submission failed: Invalid qos' >&2; exit 1; }\n"
        "echo 'sbatch: warning: a login-node note'\n"
        "echo 4242\n" % {"t": TMP, "py": sys.executable, "calls": str(TMP / "sbatch_calls.jsonl")})
    for p in (BIN / "squeue", BIN / "sbatch"):
        p.chmod(0o755)


def set_queue(lines, then=None):
    """The fake squeue's output; 'then' replaces it after the next call."""
    p = TMP / "squeue.txt"
    (TMP / "squeue_next.txt").unlink(missing_ok=True)
    if then is not None:
        (TMP / "squeue_next.txt").write_text("".join(ln + "\n" for ln in then))
    if lines is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text("".join(ln + "\n" for ln in lines))


def sbatch_calls():
    p = TMP / "sbatch_calls.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------- a driver experiment on FakeBackend
NOISE = {0: 0.0, 1: 0.002, 2: -0.002}
FACTOR = {"dev": 1.0, "ood22": 0.7, "ood23": 0.6, "imageweeds": 0.5, "test": 0.95}
SECONDS = {"base": 7200.0, "union": 7200.0, "cand": 1800.0, "null": 1800.0, "soup": 60.0, "final": 300.0}


class Executor:
    """What inc/train.py writes (as in tests/test_inc_driver.py): a cold run
    scores 0.40 + seed offset, a cand its parent + 0.02, a null its parent, a
    soup the mean of its cands + 0.001, a final run its parent x the exam
    factor. fail_if(spec): true for a run that fails every attempt."""

    def __init__(self, fail_if=None):
        self.fail_if = fail_if

    def __call__(self, spec):
        rid, out = spec["run_id"], pathlib.Path(spec["out_dir"])
        if (out / "run.json").exists():
            (out / "run.json").unlink()
        if self.fail_if is not None and self.fail_if(spec):
            self._run_json(out, {"status": "failed", "attempt": 1, "seconds": 5.0,
                                 "error": "synthetic failure of %s" % rid})
            return "FAILED"
        w = out / "weights" / "final.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "final":
            if w.is_symlink() or w.exists():
                w.unlink()
            os.symlink(spec["init"], w)
        else:
            w.write_bytes(("weights of %s" % rid).encode())
        for exam in spec["exams"]:
            v, a = self.values(spec, exam)
            self._score(out / "scores" / ("%s.json" % exam), exam, v, a, w)
        self._run_json(out, {"status": "done", "attempt": 1, "seconds": SECONDS[spec["kind"]], "error": None,
                             "weights_sha256": C.sha256_file(w)})
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
        kind = spec["kind"]
        if kind in ("base", "union"):
            s = NOISE[spec["recipe"]["seed"]]
            return 0.40 + s, 0.60 + s
        if kind in ("cand", "null"):
            pv, pa = self.parent(spec["init"])
            eff = 0.02 if kind == "cand" else 0.0
            s = NOISE[spec["recipe"]["seed"]]
            return pv + eff + s, pa + eff + s
        if kind == "soup":
            vals = [self.parent(w) for w in spec["soup_of"]]
            return statistics.fmean(v for v, _ in vals) + 0.001, statistics.fmean(a for _, a in vals) + 0.001
        pv, pa = self.parent(spec["init"])
        return pv * FACTOR[exam], pa * FACTOR[exam]

    @staticmethod
    def _score(path, exam, v, a, weights):
        other = 0 if exam in ("dev", "test") else 25
        n_gt = {s: 40 for s in G.SPECIES}
        n_gt["OtherPlant"] = other
        per_class = {s: v for s in G.SPECIES}
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


def make_rows(n, tag):
    rows = []
    for i in range(n):
        img = TMP / "img" / ("%s_%03d.jpg" % (tag, i))
        rows.append({"image": str(img), "label": str(img.with_suffix(".txt")),
                     "sha256": hashlib.sha256(("img/%s/%d" % (tag, i)).encode()).hexdigest(),
                     "label_sha256": hashlib.sha256(("lab/%s/%d" % (tag, i)).encode()).hexdigest(),
                     "source": "src_%s" % tag, "session": "", "key": "%s__%03d" % (tag, i)})
    return rows


def small_defn(exp):
    mdir = TMP / "small" / exp
    bp, ap = mdir / "base.jsonl", mdir / "A.jsonl"
    base_rows, a_rows = make_rows(40, exp + "b"), make_rows(8, exp + "a")
    return {"exp": exp, "type": "chain", "testing": True, "seeds": [0, 1, 2], "init_weights": "yolo11n.pt",
            "decision_exam": "dev", "final_exams": list(D.FINAL_EXAMS),
            "base": {"name": "base", "manifest": str(bp), "manifest_sha256": C.write_manifest(bp, base_rows),
                     "n_images": len(base_rows), "recipe": P.cold_recipe()},
            "steps": [{"name": "A", "manifest": str(ap), "manifest_sha256": C.write_manifest(ap, a_rows),
                       "n_images": len(a_rows), "clean": True}],
            "recipes": {"full": P.inc_recipes()["full"]}, "truth": False, "truth_recipe": P.cold_recipe()}


class Clock:
    def __init__(self):
        self.t = 1.8e9

    def __call__(self):
        return self.t


def state_of(exp):
    return json.loads(D.Paths(exp).state.read_text())


def drive(exp, fb, clock, executor, max_iter=100):
    for _ in range(max_iter):
        ran = fb.run_pending(executor)
        before = len(fb.submissions)
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        st = state_of(exp)
        if st["done"] or (ran == 0 and len(fb.submissions) == before):
            return st
    return state_of(exp)


# ------------------------------------------------------------------ tests
def test_markers():
    rec = {"verb": "status", "ok": True, "note": "two\nlines and a \"quote\"", "n": [1, 2.5, None]}
    import io
    buf = io.StringIO()
    RM.emit(rec, buf)
    out = buf.getvalue()
    check("emit writes one line", out.count("\n") == 1 and out.startswith(M.REMOTE_MARK + " "), repr(out[:80]))
    noisy = ("Last login: Sat Sep 27 from 10.0.0.1\r\n*** Bridges-2 ***\n  %s\nINCAPABLE of anything\n"
             "echo INCAP {\"x\": 1}\n" % out.strip())
    check("parse_marked skips banners and near-miss lines", RM.parse_marked(noisy) == [rec])
    check("parse_one returns the record", RM.parse_one(noisy, verb="status") == rec)
    for text, why in (("no marker here\n", "zero"), (out + out, "two"), ("INCAP [1, 2]\n", "not an object"),
                      ("INCAP {not json\n", "malformed")):
        try:
            RM.parse_one(text)
            check("parse_one raises on %s" % why, False)
        except ValueError:
            check("parse_one raises on %s" % why, True)
    try:
        RM.parse_one(out, verb="snapshot")
        check("parse_one raises on another verb's record", False)
    except ValueError:
        check("parse_one raises on another verb's record", True)


def run_cli(*args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    e["PYTHONPATH"] = str(PKG_ROOT)
    p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc_autopilot.remote"] + list(args),
                       capture_output=True, text=True, cwd=str(PKG_ROOT), env=e, timeout=300)
    return p


def test_cli_one_line():
    cases = [(("status",), True), (("snapshot", "--exp", "pilot_v1"), True),
             (("snapshot", "--exp", "../x"), False), (("snapshot",), False), (("advance", "--exp", "nope"), False),
             (("report", "--exp", "nope"), False), (("bogus",), False), ((), False), (("status", "--help"), False),
             (("submit", "build", "--dry-run", "--", "pilot", "build", "--exp", "pilot_v9",
               "--replay-mode", "full"), True),
             (("submit", "build", "--", "pilot", "build", "--exp", "pilot_v9", "--testing"), False),
             (("campaign-snapshot", "--exp", "pilot_v1", "--ledger-from", "pilot_v1=bad"), False),
             (("unblock", "--exp", "pilot_v1", "--unit", "base", "--reason", "by hand"), False),
             (("cancel", "--exp", "pilot_v1", "--dry-run"), True), (("sync-outer", "--dry-run"), False),
             (("submit", "pilot", "build", "--exp", "pilot_v9", "--replay-mode", "full", "--dry-run"), False)]
    for args, ok in cases:
        p = run_cli(*args)
        lines = p.stdout.splitlines()
        good = len(lines) == 1 and lines[0].startswith(M.REMOTE_MARK + " ")
        rec = RM.parse_one(p.stdout) if good else {}
        check("cli %s: exactly one INCAP line, ok=%s, exit %d" % (" ".join(args) or "(no verb)", ok, 0 if ok else 1),
              good and rec.get("ok") is ok and p.returncode == (0 if ok else 1),
              (p.returncode, p.stdout[:300], p.stderr[-300:]))


def copy_local(exp, names):
    dst = INC / exp
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copyfile(LOCAL_INC / exp / n, dst / n)


def perturb_leaves(obj, how):
    """obj with every number replaced by how(number), every bool flipped,
    None made a number (sd of a single run is null) and every string changed."""
    if isinstance(obj, dict):
        return {k: perturb_leaves(v, how) for k, v in obj.items()}
    if isinstance(obj, list):
        return [perturb_leaves(v, how) for v in obj]
    if isinstance(obj, bool):
        return not obj
    if isinstance(obj, (int, float)):
        return how(obj)
    if obj is None:
        return how(0.0)
    return str(obj) + "~"


def perturb_non_dev(final, how):
    """Every leaf under a non-dev exam key of a report's final table (mean,
    sd, n), and each row's production flags, which report.py derives from
    every exam's score files, test included."""
    n = 0
    for row in final:
        for exam in list(row["exams"]):
            if exam in RM.NON_DEV_EXAMS:
                row["exams"][exam] = perturb_leaves(row["exams"][exam], how)
                n += 1
        prod = row.get("production") or []
        row["production"] = [not prod[0]] if len(prod) == 1 else [True]
    return n


def diagnose_d3_levers(snap):
    """(D3's levers, its detail) on the evidence of a snapshot record, or None
    when the sibling modules do not import (their own tests cover that)."""
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
    except Exception:
        return None
    d = [x for x in DG.detect(E.from_snapshot(snap, "pilot_v1"), only=["D3"]) if x["id"] == "D3"]
    return (d[0]["levers"], d[0].get("detail") or {}) if d else ([], {})


def test_snapshot_pilot_v1():
    if not (LOCAL_INC / "pilot_v1" / "report.json").is_file():
        skip("snapshot of pilot_v1", "no local pilot_v1")
        return
    copy_local("pilot_v1", ["exp.json", "report.json", "report.md", "ledger.jsonl", "build_summary.json"])
    copy_local("b0_v1", ["exp.json", "ledger.jsonl"])
    sentinel = 0.987654321
    runs = INC / "pilot_v1" / "runs" / "final__base__s0" / "scores"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "test.json").write_text(json.dumps({"map50_95": sentinel}))

    rec = RM.snapshot("pilot_v1", step1=False)
    dec = rec.get("decision") or {}
    art = dec.get("artifacts") or {}
    rep_final = json.loads((LOCAL_INC / "pilot_v1" / "report.json").read_text())["final"]
    check("pilot_v1 snapshot ok and built", rec["ok"] and rec["built"], rec.get("error"))
    check("decision carries exp, report (no final) and build summary",
          set(art) == {"pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/build_summary.json"}
          and "final" not in art["pilot_v1/report.json"] and "agreement" in art["pilot_v1/report.json"], sorted(art))
    check("no key named after a non-dev exam anywhere in decision", RM.non_dev_keys(dec) == [],
          RM.non_dev_keys(dec)[:5])
    check("report's final table is display_only, whole", rec["display_only"].get("pilot_v1/report.json#/final")
          == rep_final)
    fd = dec["derived"]["report_final_dev"]
    check("decision holds only the final table's dev column: rows of exactly model, runs, exams.dev",
          len(fd) == len(rep_final) and all(set(r) == {"model", "runs", "exams"} for r in fd)
          and all(set(r["exams"]) == {"dev"} for r in fd)
          and fd[0]["exams"]["dev"] == rep_final[0]["exams"]["dev"], [sorted(r) for r in fd][:2])
    test_val = repr(rep_final[0]["exams"]["test"]["twelve"]["mean"])
    check("a test value of pilot_v1 is not in decision", test_val not in dumps(dec) and test_val in dumps(rec))
    check("snapshot never opens runs/ (the planted scores/test.json sentinel is nowhere)",
          repr(sentinel) not in dumps(rec) and not any("runs/" in k for k in rec["files"]))
    check("report.md (it holds the final table) is not shipped",
          "pilot_v1/report.md" not in rec["files"] and "pilot_v1/report.md" not in dumps(dec))
    led = dec["ledger"]
    check("the whole ledger with line numbers", led["n_lines"] == 29 and led["complete"] and not led["partial_tail"]
          and [e["line"] for e in led["entries"]] == list(range(1, 30))
          and led["entries"][0]["entry"]["id"] == "code_pin/0", (led["n_lines"], led["complete"]))
    real = (LOCAL_INC / "pilot_v1" / "report.json").read_bytes()
    check("files carry sha256 (outside decision)", rec["files"]["pilot_v1/report.json"]["sha256"]
          == hashlib.sha256(real).hexdigest() and "files" not in dec)
    check("absent state.json is listed missing", "pilot_v1/state.json" in rec["missing"])

    # metamorphic: every leaf under a non-dev exam of the final table changed, and
    # every row's production flags flipped -> decision bytes identical
    rp = INC / "pilot_v1" / "report.json"
    rj = json.loads(rp.read_text())
    n_blocks = perturb_non_dev(rj["final"], lambda v: v * 0.5 + 0.1234567)
    changed = [r for r0, r in zip(rep_final, rj["final"]) for e in RM.NON_DEV_EXAMS
               if r["exams"][e]["twelve"]["sd"] != r0["exams"][e]["twelve"]["sd"]
               and r["exams"][e]["twelve"]["n"] != r0["exams"][e]["twelve"]["n"]]
    check("(setup) every non-dev block perturbed, sd and n included, and production flipped",
          n_blocks == 4 * len(rep_final) and len(changed) == n_blocks
          and all(r["production"] != r0["production"] for r0, r in zip(rep_final, rj["final"])))
    rp.write_text(json.dumps(rj, indent=1))
    rec2 = RM.snapshot("pilot_v1", step1=False)
    check("test-blind: perturbed test/ood/imageweeds values and production flags leave decision identical",
          dumps(rec2["decision"]) == dumps(dec))
    check("... while display_only and the file hash change",
          dumps(rec2["display_only"]) != dumps(rec["display_only"])
          and rec2["files"]["pilot_v1/report.json"]["sha256"] != rec["files"]["pilot_v1/report.json"]["sha256"])

    # ledger windows
    r = RM.snapshot("pilot_v1", ledger_from=20, step1=False)["decision"]["ledger"]
    raw = (INC / "pilot_v1" / "ledger.jsonl").read_bytes().split(b"\n")
    check("--ledger-from 20 ships lines 21..29 and the sha256 of lines 1..20",
          [e["line"] for e in r["entries"]] == list(range(21, 30)) and r["next_line"] == 29
          and r["prefix_sha256"] == hashlib.sha256(b"".join(x + b"\n" for x in raw[:20])).hexdigest())
    sha = lambda k: hashlib.sha256(b"".join(x + b"\n" for x in raw[:k])).hexdigest()  # noqa: E731
    r20 = RM.snapshot("pilot_v1", ledger_from=0, step1=False)["decision"]["ledger"]
    check("a read's through_sha256 is the sha256 of lines 1..next_line: the next read's prefix_sha256",
          r20["through_sha256"] == sha(29) and r["through_sha256"] == sha(29)
          and RM.snapshot("pilot_v1", ledger_from=29, step1=False)["decision"]["ledger"]["prefix_sha256"] == sha(29))
    r = RM.snapshot("pilot_v1", ledger_from=20, step1=False, ledger_sha256=sha(20))["decision"]["ledger"]
    check("--ledger-from 20:<the right sha256>: lines 21..29, no mismatch",
          [e["line"] for e in r["entries"]] == list(range(21, 30)) and "prefix_mismatch" not in r)
    r = RM.snapshot("pilot_v1", ledger_from=20, step1=False, ledger_sha256="0" * 64)["decision"]["ledger"]
    check("--ledger-from 20:<another sha256> (the prefix was rewritten): re-read from line 0, flagged",
          r["from_line"] == 0 and [e["line"] for e in r["entries"]] == list(range(1, 30))
          and r["prefix_mismatch"] == {"from_line": 20, "expected": "0" * 64, "found": sha(20)}
          and r["prefix_sha256"] == sha(0), {k: r.get(k) for k in ("from_line", "prefix_mismatch")})
    p = run_cli("snapshot", "--exp", "pilot_v1", "--no-step1", "--ledger-from", "20:%s" % ("0" * 64))
    rr = RM.parse_one(p.stdout, verb="snapshot") if p.returncode == 0 else {}
    check("cli --ledger-from N:SHA256", (rr.get("decision") or {}).get("ledger", {}).get("prefix_mismatch"),
          (p.returncode, p.stdout[-300:]))
    for bad in ("20:xyz", "-1", "20:" + "A" * 64):
        p = run_cli("snapshot", "--exp", "pilot_v1", "--ledger-from", bad)
        check("cli --ledger-from %s: usage error" % bad[:12],
              p.returncode == 1 and RM.parse_one(p.stdout)["error_kind"] == "usage")
    r = RM.snapshot("pilot_v1", ledger_from=99, step1=False)["decision"]["ledger"]
    check("--ledger-from past the end says re-read", not r["complete"] and "re-read" in r["error"] and not r["entries"])
    os.environ["INCAP_MAX_LEDGER_BYTES"] = "20000"
    r = RM.snapshot("pilot_v1", step1=False)["decision"]["ledger"]
    os.environ.pop("INCAP_MAX_LEDGER_BYTES")
    check("the per-call ledger cap stops early and says where to go on",
          not r["complete"] and 1 <= len(r["entries"]) < 29 and r["next_line"] == len(r["entries"]),
          (len(r["entries"]), r["next_line"]))
    lp = INC / "pilot_v1" / "ledger.jsonl"
    lp.write_bytes(lp.read_bytes() + b'{"id": "gate/full/8", "partial')
    r = RM.snapshot("pilot_v1", step1=False)["decision"]["ledger"]
    check("a partial last line is flagged and not shipped", r["partial_tail"] and r["n_lines"] == 29
          and len(r["entries"]) == 29)
    shutil.copyfile(LOCAL_INC / "pilot_v1" / "ledger.jsonl", lp)

    os.environ["INCAP_MAX_FILE_BYTES"] = "20000"
    r = RM.snapshot("pilot_v1", step1=False)
    os.environ.pop("INCAP_MAX_FILE_BYTES")
    f = r["files"]["pilot_v1/report.json"]
    check("a file over the cap is hashed, not shipped",
          f["shipped"] is False and "larger than" in f["why"] and f["sha256"]
          and "pilot_v1/report.json" not in r["decision"]["artifacts"] and not r["display_only"])

    # pilot_v1's real label audit, run by hand to INC_DIR/audit/pilot_v1_audit.json
    legacy_src = LOCAL_INC / "audit" / "pilot_v1_audit.json"
    if legacy_src.is_file():
        d3_levers = diagnose_d3_levers(RM.snapshot("pilot_v1", step1=False))
        legacy = INC / "audit" / "pilot_v1_audit.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy_src, legacy)
        r = RM.snapshot("pilot_v1", step1=False)
        a = r["decision"]["artifacts"].get("pilot_v1/audit/label_audit.json") or {}
        real = json.loads(legacy_src.read_text())
        check("the hand-run audit INC_DIR/audit/pilot_v1_audit.json ships as pilot_v1/audit/label_audit.json, "
              "its real path in files",
              a.get("built_utc") == real["built_utc"] and set(a.get("audits", {})) == set(real["audits"])
              and r["files"]["audit/pilot_v1_audit.json"]["as"] == "pilot_v1/audit/label_audit.json"
              and r["files"]["audit/pilot_v1_audit.json"]["sha256"] == C.sha256_file(legacy_src)
              and RM.non_dev_keys(a) == [], sorted(r["files"]))
        d3_after = diagnose_d3_levers(r)
        if d3_levers is None:
            skip("diagnose D3 with the hand-run audit", "evidence / diagnose do not import")
        else:
            check("diagnose D3 reads it: L4 (run the audit) before, not after",
                  "L4" in d3_levers[0] and "L4" not in d3_after[0] and "audit_separates" in d3_after[1],
                  (d3_levers, d3_after))
        legacy.unlink()
    else:
        skip("the hand-run pilot_v1 audit", "no local results/framework/inc/audit/pilot_v1_audit.json")

    # provenance and audit outputs join the snapshot
    prov = INC / "_campaign" / "provenance" / "pilot_v1.json"
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text(json.dumps({"exp": "pilot_v1", "attempts": [{"status": "advanced", "trigger": ["D1"]}]}))
    au = INC / "pilot_v1" / "audit"
    au.mkdir(exist_ok=True)
    (au / "label_audit.json").write_text(json.dumps({"audits": {"Bswap": {"species": {"above_baseline": True}}}}))
    r = RM.snapshot("pilot_v1", step1=False)["decision"]["artifacts"]
    check("provenance and audit outputs are decision artifacts",
          r["_campaign/provenance/pilot_v1.json"]["attempts"][0]["trigger"] == ["D1"]
          and r["pilot_v1/audit/label_audit.json"]["audits"]["Bswap"]["species"]["above_baseline"] is True)

    if (LOCAL_INC / "audit" / "pilot_v1_audit.json").is_file():
        (INC / "audit").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL_INC / "audit" / "pilot_v1_audit.json", INC / "audit" / "pilot_v1_audit.json")
    r = RM.snapshot("pilot_v1", step1=False)
    check("an audit inside the experiment wins over the hand-run one",
          r["decision"]["artifacts"]["pilot_v1/audit/label_audit.json"]["audits"] == {
              "Bswap": {"species": {"above_baseline": True}}})
    (INC / "audit" / "pilot_v1_audit.json").unlink(missing_ok=True)
    (au / "label_audit.json").unlink()                  # submit's audit cases need pilot_v1 unaudited

    r = RM.snapshot("not_built_yet", step1=False)
    check("an experiment not built yet: ok, built false, everything missing",
          r["ok"] and r["built"] is False and "not_built_yet/exp.json" in r["missing"])
    r = RM.snapshot("../etc", step1=False)
    check("a bad experiment name is refused", refused(r) and r["error_kind"] == "bad_name")
    shutil.copyfile(LOCAL_INC / "pilot_v1" / "report.json", rp)


def test_driver_experiment():
    exp = "rx_v1"
    fb, clock, ex = D.FakeBackend(), Clock(), Executor()
    D.init(small_defn(exp), backend=fb, quiet=True, clock=clock)
    st = drive(exp, fb, clock, ex)
    check("synthetic chain experiment reaches done", st["done"], st.get("blocked"))

    r = RM.report(exp)
    root = INC / exp
    check("report verb: ok, hashes only",
          r["ok"] and r["done"] and set(r["files"]) == {exp + "/report.json", exp + "/report.md"}
          and "final" not in dumps(r), r.get("error"))
    snap = RM.snapshot(exp, step1=False)
    dec = snap["decision"]
    sr = dec["derived"]["state_runs"]
    check("state without runs, run counts derived",
          "runs" not in dec["artifacts"][exp + "/state.json"] and sr["n_runs"] == len(st["runs"])
          and all(set(c) == {"complete"} for c in sr["by_owner"].values()) and not sr["failed"], sr)
    check("state's submissions keep everything but the task lists",
          all("tasks" not in s and s.get("job_name", "").startswith("inc_%s_" % exp)
              for s in dec["artifacts"][exp + "/state.json"]["submissions"]))
    gates = [e for e in dec["ledger"]["entries"] if e["entry"].get("type") == "gate"]
    check("the gate entry is in the ledger", len(gates) == 1 and gates[0]["entry"]["decision"]["verdict"] == "ACCEPT")
    check("no non-dev key in a real experiment's decision", RM.non_dev_keys(dec) == [])
    os.environ["INCAP_MAX_FILE_BYTES"] = "2000"
    small = RM.snapshot(exp, step1=False)
    os.environ.pop("INCAP_MAX_FILE_BYTES")
    check("state.json over the per-file cap still ships (compacted); other files over it do not",
          small["files"][exp + "/state.json"]["bytes"] > 2000 and exp + "/state.json" in small["decision"]["artifacts"]
          and small["files"][exp + "/report.json"]["shipped"] is False, small["files"].get(exp + "/state.json"))

    sentinel = 0.123456789
    tj = root / "runs" / "final__full__incumbent" / "scores" / "test.json"      # one run: its mean is the value
    t = json.loads(tj.read_text())
    t["map50_95"] = t["species_map50_95"] = sentinel
    tj.write_text(json.dumps(t))
    RM.report(exp)
    s1 = RM.snapshot(exp, step1=False)
    check("a sentinel in a final run's scores/test.json reaches display_only, never decision",
          repr(sentinel) in dumps(s1["display_only"]) and repr(sentinel) not in dumps(s1["decision"]))

    def norm(d):
        d = json.loads(dumps(d))
        d["artifacts"][exp + "/report.json"].pop("generated_utc", None)   # a timestamp, not a value
        return dumps(d)

    n_files = 0
    for rid in os.listdir(root / "runs"):
        for exam in RM.NON_DEV_EXAMS:
            p = root / "runs" / rid / "scores" / ("%s.json" % exam)
            if p.exists():
                s = json.loads(p.read_text())
                # every number and flag the scorer writes (production too: report.py
                # derives each final row's production from every exam's files)
                for k, v in list(s.items()):
                    if isinstance(v, bool):
                        s[k] = not v
                    elif isinstance(v, (int, float)):
                        s[k] = v * 0.3 + 0.05
                    elif isinstance(v, dict):
                        s[k] = {kk: (vv * 0.3 + 0.05 if isinstance(vv, (int, float)) else vv)
                                for kk, vv in v.items()}
                p.write_text(json.dumps(s))
                n_files += 1
    RM.report(exp)
    s2 = RM.snapshot(exp, step1=False)
    fin1, fin2 = (x["display_only"][exp + "/report.json#/final"] for x in (s1, s2))
    check("(setup) the re-report sees the flipped production flags of the non-dev score files",
          n_files and all(r1["production"] == [False] and r2["production"] == [False, True]
                          for r1, r2 in zip(fin1, fin2)), [(r1["production"], r2["production"])
                                                           for r1, r2 in zip(fin1, fin2)][:3])
    check("test-blind end to end: every non-dev score and flag re-reported, decision identical (but the timestamp)",
          norm(s2["decision"]) == norm(s1["decision"]) and dumps(s2["display_only"]) != dumps(s1["display_only"]))
    check("... and report_final_dev rows are exactly model, runs, exams.dev",
          all(set(r) == {"model", "runs", "exams"} and set(r["exams"]) == {"dev"}
              for r in s2["decision"]["derived"]["report_final_dev"]))

    a = RM.advance(exp, backend=fb)
    check("advance on a done experiment: ok, result as data",
          a["ok"] and a["result"]["done"] is True and a["result"]["exp"] == exp and "lines" in a["result"], a)
    a = RM.advance("never_built")
    check("advance before the build: not_built", refused(a) and a["error_kind"] == "not_built", a)
    sp = D.Paths(exp).state

    set_queue(["123_[0-5%40]|inc_" + exp + "_0007|PENDING|0:00|2026-09-27T01:00:00",
               "124|other_job|RUNNING|1:00|2026-09-27T01:00:00"])
    s = RM.status()
    e = {x["exp"]: x for x in s["experiments"]}
    check("status lists the experiment, done, report current, and only inc_ jobs",
          s["ok"] and e[exp]["done"] and e[exp]["report"] == "current" and e[exp]["chains"]["full"]["phase"] == "done"
          and [j["name"] for j in s["squeue"]["jobs"]] == ["inc_%s_0007" % exp], (e.get(exp), s["squeue"]))
    set_queue(None)
    (TMP / "squeue_fail").write_text("x")
    s = RM.status()
    (TMP / "squeue_fail").unlink()
    check("status says when squeue is down", s["ok"] and s["squeue"]["ok"] is False and "exited 1" in s["squeue"]["error"])

    # unblock (L7): a cand run that fails every attempt blocks 'chain:full'
    exp2 = "rx_block"
    fb2 = D.FakeBackend(first_job_id=5000)
    ex2 = Executor(fail_if=lambda sp: sp["kind"] == "cand" and sp["recipe"]["seed"] == 1)
    D.init(small_defn(exp2), backend=fb2, quiet=True, clock=clock)
    st2 = drive(exp2, fb2, clock, ex2)
    unit = "chain:full"
    check("a failing cand run blocks 'chain:full'", unit in st2["blocked"], st2["blocked"])
    sp2 = D.Paths(exp2).state
    saved = sp2.read_text()
    st = json.loads(saved)
    st["code"]["modules"]["tools/inc/gate.py"] = "0" * 64
    sp2.write_text(json.dumps(st))
    a = RM.advance(exp2, backend=fb2)
    check("advance on changed pinned code: code_pin", refused(a) and a["error_kind"] == "code_pin", a)
    sp2.write_text(saved)
    check("unblock refuses a reason that is not 'auto: ...'",
          refused(RM.unblock(exp2, unit, "because", backend=fb2), "auto:"))
    check("unblock refuses a unit that is not blocked",
          refused(RM.unblock(exp2, "base", "auto: x", backend=fb2), "not blocked"))
    check("unblock refuses a malformed unit", refused(RM.unblock(exp2, "base; rm", "auto: x", backend=fb2), "unit"))
    check("rx_block's block is a failed_run one (driver.py _block_on_failed)",
          st2["blocked"][unit]["cause"]["kind"] == "failed_run", st2["blocked"][unit].get("cause"))
    u = RM.unblock(exp2, unit, "auto: transient", backend=fb2)
    st2b = state_of(exp2)
    check("unblock refuses a failed_run block as 'transient': the live cause decides, nothing changes",
          refused(u, "not a transient one") and u["error_kind"] == "refused" and u["cause"]["kind"] == "failed_run"
          and unit in st2b["blocked"] and not st2b.get("unblocks"), u.get("error"))

    # a real transient block: the chain's score reads fail TRANSIENT_LIMIT passes in a row
    exp3 = "rx_transient"
    fb3 = D.FakeBackend(first_job_id=7000)
    D.init(small_defn(exp3), backend=fb3, quiet=True, clock=clock)
    for _ in range(2):
        fb3.run_pending(ex)
        D.Driver(exp3, backend=fb3, clock=clock, quiet=True).advance()
    fb3.run_pending(ex)
    real_reads = D.Driver._score_inputs

    def broken(pairs):
        raise OSError(5, "Input/output error (simulated)")

    def transient_block(passes):
        D.Driver._score_inputs = staticmethod(broken)
        try:
            for _ in range(passes):
                D.Driver(exp3, backend=fb3, clock=clock, quiet=True).advance()
        finally:
            D.Driver._score_inputs = staticmethod(real_reads)
        return state_of(exp3)

    st3 = transient_block(D.TRANSIENT_LIMIT)
    check("(setup) unreadable scores for TRANSIENT_LIMIT passes block 'chain:full' with cause transient",
          (st3["blocked"].get(unit) or {}).get("cause") == {"kind": "transient"}, st3["blocked"])
    D.Driver._score_inputs = staticmethod(broken)             # the unblock's own advance still cannot read
    try:
        u = RM.unblock(exp3, unit, "auto: transient", backend=fb3)
    finally:
        D.Driver._score_inputs = staticmethod(real_reads)
    st3 = state_of(exp3)
    led = [json.loads(x) for x in D.Paths(exp3).ledger.read_text().splitlines()]
    check("unblock lifts a transient block through the driver (ledger, state)",
          u["ok"] and u["cause"] == {"kind": "transient"} and st3["unblocks"][-1]["reason"] == "auto: transient"
          and unit not in st3["blocked"] and [e for e in led if e["type"] == "unblock"], u.get("error"))
    st3 = transient_block(D.TRANSIENT_LIMIT - 1)
    check("the unit blocks again, transient again", (st3["blocked"].get(unit) or {}).get("cause") == {"kind": "transient"},
          st3["blocked"])
    check("a second automatic unblock of the same unit is refused (D5.max_auto_unblocks_per_unit = 1)",
          refused(RM.unblock(exp3, unit, "auto: again", backend=fb3), "at most 1 automatic unblock"))
    sp3 = D.Paths(exp3).state
    st = json.loads(sp3.read_text())
    st["unblocks"][-1].update(reason="the driver's late recovery", auto=True)
    sp3.write_text(json.dumps(st))
    check("... also when the earlier one is marked auto by the driver without the reason prefix",
          refused(RM.unblock(exp3, unit, "auto: again", backend=fb3), "at most 1 automatic unblock"))
    st["unblocks"][-1].update(reason="by hand", auto=False)
    sp3.write_text(json.dumps(st))
    r = RM.unblock(exp3, unit, "auto: after a person's unblock", backend=fb3)
    check("an earlier unblock by hand does not count against the autopilot's one",
          r["ok"] and unit not in state_of(exp3)["blocked"], r.get("error"))
    shutil.rmtree(D.Paths(exp3).root)                 # its edited state is no experiment the later cases know

    rules = RM.unblock_rules()
    check("thresholds.json D5 is what unblock reads (and AUTO_PREFIX agrees with it)",
          rules == (("transient",), 1, "auto:") and RM.AUTO_PREFIX == rules[2], rules)
    bad = TMP / "thresholds_bad.json"
    for text, why in (("{}", "no D5"), (json.dumps({"D5": {"transient_cause_kinds": {"value": "transient"},
                                                           "max_auto_unblocks_per_unit": {"value": 1},
                                                           "auto_reason_prefix": {"value": "auto:"}}}), "malformed")):
        bad.write_text(text)
        try:
            RM.unblock_rules(bad)
            check("thresholds.json %s: no automatic unblock" % why, False)
        except RM.Refused:
            check("thresholds.json %s: no automatic unblock" % why, True)
    try:
        RM.unblock_rules(TMP / "nope.json")
        check("thresholds.json missing: no automatic unblock", False)
    except RM.Refused:
        check("thresholds.json missing: no automatic unblock", True)

    # campaign-snapshot batches everything in one record
    rp = root / "report.json"
    os.utime(rp, (rp.stat().st_atime, sp.stat().st_mtime - 100))       # report older than state
    c = RM.campaign_snapshot([exp, exp2, "not_built_yet"], do_advance=True, report_mode="auto",
                             ledger_from={exp: 1}, step1=True, backend=fb)
    ex_ = c["experiments"]
    check("campaign-snapshot: advance, report (stale -> rebuilt), snapshot per experiment; step1; status",
          ex_[exp]["advance"]["ok"] and ex_[exp]["report"]["ok"] and ex_[exp]["snapshot"]["ok"]
          and ex_[exp]["snapshot"]["decision"]["ledger"]["from_line"] == 1
          and "report" not in ex_[exp2] and ex_["not_built_yet"]["advance"].get("skipped") == "not built"
          and c["step1"]["verb"] == "step1" and c["status"]["verb"] == "status"
          and "step1/select_summary.json" not in ex_[exp]["snapshot"]["files"], json.dumps(c)[:600])
    check("campaign-snapshot keeps test values out of every decision section",
          all(RM.non_dev_keys(s["snapshot"]["decision"]) == [] for s in ex_.values()))
    return exp


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_step1():
    s1 = INC / "step1"
    s1.mkdir(parents=True, exist_ok=True)
    sel = [{"key": "a%d" % i, "source": "srcA"} for i in range(3)]
    pool = ([{"key": "a%d" % i, "source": "srcA"} for i in range(3, 5)]
            + [{"key": "g%d" % i, "source": "fvossel__csgo_player_detection"} for i in range(4)])
    write_jsonl(s1 / "base_selected.jsonl", sel)
    write_jsonl(s1 / "increment_pool.jsonl", pool)
    rows = ([("a0", "selected", 2, 0), ("a1", "selected", 1, 1), ("a2", "refilled", 3, 0),
             ("a3", "pool", 1, 0), ("a4", "below_gate", 0, 2)]
            + [("g%d" % i, st, 0, 3) for i, st in enumerate(["no_evidence", "no_evidence", "no_feature",
                                                             "below_gate"])]
            + [("orphan", "pool", 1, 1)])
    with open(s1 / "select_clusters.csv", "w") as fh:
        fh.write("key,dup_group,l1,l2,score_kind,cosine,score,typicality,species_boxes,other_boxes,rank,status\n")
        for k, st, sp, ot in rows:
            fh.write("%s,0,0,0,species,0.5,0.4,0.3,%d,%d,0,%s\n" % (k, sp, ot, st))
    csv_sha = C.sha256_file(s1 / "select_clusters.csv")
    (s1 / "select_summary.json").write_text(json.dumps({
        "sizes": {"verified": 10, "increment_pool": 6, "no_evidence": 2},
        "sources": {"increment_pool": {"srcA": 2, "fvossel__csgo_player_detection": 4}},
        "never_train": {"per_split": {"dev": 5, "test": 7}},
        "outputs": {"select_clusters.csv": {"sha256": csv_sha}}}))
    (s1 / "admit_summary.json").write_text(json.dumps({"images": {"admitted": 10}, "conflict_boxes": 3,
                                                       "per_slug": {"srcA": {"boxes": {"verified": 7}}}}))
    (s1 / "relevance.json").write_text(json.dumps({"increment_pool": {"sources": {
        "fvossel__csgo_player_detection": {"status": "fail", "top_set_share": {"leaf_disease": 0.0}}}}}))
    return s1


def test_step1():
    for p in (INC / "_campaign" / "cache").glob("*.json") if (INC / "_campaign" / "cache").is_dir() else []:
        p.unlink()
    s1 = make_step1()
    r = RM.step1_snapshot()
    agg = r["decision"]["derived"]["select_clusters_by_source"]
    src = agg["sources"]
    check("clusters aggregated per source and set through the select manifests",
          src["srcA"]["selected"]["images"] == 3 and src["srcA"]["increment_pool"]["images"] == 2
          and src["srcA"]["selected"]["status"] == {"refilled": 1, "selected": 2}
          and src["srcA"]["increment_pool"]["uncertified"] == 1
          and src["fvossel__csgo_player_detection"]["increment_pool"] ==
          {"images": 4, "status": {"below_gate": 1, "no_evidence": 2, "no_feature": 1}, "uncertified": 4,
           "species_boxes": 0, "other_boxes": 12}, json.dumps(src)[:400])
    check("an unmatched key is counted, not dropped",
          agg["rows"] == 10 and agg["unmatched_keys"] == 1 and src[RM.UNMATCHED]["unmatched"]["images"] == 1)
    check("the CSV hash matches select_summary's record", agg["sha256_matches_select_summary"] is True)
    check("first aggregation is a cache miss", r["cache"]["select_clusters_by_source"] == "miss")
    r2 = RM.step1_snapshot()
    check("second is a cache hit with the same decision bytes",
          r2["cache"]["select_clusters_by_source"] == "hit" and dumps(r2["decision"]) == dumps(r["decision"]))
    check("the big Step 1 files are listed, never shipped",
          all(r["files"]["step1/" + n]["shipped"] is False for n in RM.STEP1_BIG)
          and r["files"]["step1/select_clusters.csv"]["sha256_recorded_by_select"] == agg["sha256"]
          and not any(n.endswith((".csv", ".jsonl")) for n in r["decision"]["artifacts"]))
    check("select_summary, admit_summary and relevance are decision artifacts",
          {"step1/select_summary.json", "step1/admit_summary.json", "step1/relevance.json"}
          <= set(r["decision"]["artifacts"]))
    check("a split-named key inside Step 1 is redacted and its pointer listed",
          "/artifacts/step1~1select_summary.json/never_train/per_split/test" in r["redacted"]
          and "test" not in r["decision"]["artifacts"]["step1/select_summary.json"]["never_train"]["per_split"])
    with open(s1 / "select_clusters.csv", "a") as fh:
        fh.write("g0,0,0,0,species,0.5,0.4,0.3,0,1,0,no_evidence\n")
    r3 = RM.step1_snapshot()
    check("a changed CSV invalidates the cache",
          r3["cache"]["select_clusters_by_source"] == "miss"
          and r3["decision"]["derived"]["select_clusters_by_source"]["rows"] == 11
          and r3["decision"]["derived"]["select_clusters_by_source"]["sha256_matches_select_summary"] is False)
    check("snapshot --no-step1 leaves Step 1 out; the default takes it",
          "step1/select_summary.json" not in RM.snapshot("pilot_v1", step1=False)["files"]
          and "step1/select_summary.json" in RM.snapshot("pilot_v1")["files"])

    # fixture writer, through the CLI's INCAP line
    p = run_cli("snapshot", "--exp", "pilot_v1")
    src_file = TMP / "snap.txt"
    src_file.write_text("banner\n" + p.stdout)
    out = TMP / "fixture_out"
    f = run_cli("fixture", "--from", str(src_file), "--out", str(out))
    fr = RM.parse_one(f.stdout, verb="fixture")
    man = json.loads((out / "MANIFEST.json").read_text()) if (out / "MANIFEST.json").exists() else {}
    agg_f = json.loads((out / "step1" / "select_clusters_by_source.json").read_text()) \
        if (out / "step1" / "select_clusters_by_source.json").exists() else {}
    check("fixture: artifacts, the whole ledger and the per-source table as files, hashed in MANIFEST.json",
          fr["ok"] and (out / "step1" / "select_summary.json").is_file() and (out / "pilot_v1" / "ledger.jsonl").is_file()
          and len((out / "pilot_v1" / "ledger.jsonl").read_text().splitlines()) == 29
          and agg_f.get("rows") == 11 and set(man.get("files", {})) == set(fr["written"])
          and all(hashlib.sha256((out / n).read_bytes()).hexdigest() == h for n, h in man["files"].items()),
          (fr.get("error"), sorted(fr.get("written", {}))))
    check("fixture files hold no key named after a non-dev exam",
          all(RM.non_dev_keys(json.loads((out / n).read_text())) == [] for n in man.get("files", {})
              if n.endswith(".json")))
    rep_f = json.loads((out / "pilot_v1" / "report.json").read_text())
    check("the fixture's report.json carries the final table's dev column only (D4's tie-break stays readable)",
          rep_f["final"] and all(set(r["exams"]) == {"dev"} for r in rep_f["final"]))
    try:
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
    except Exception as e:                      # the evidence module's own test covers its import
        skip("evidence from the record == evidence from the fixture", "evidence does not import: %s" % e)
    else:
        snap_rec = RM.parse_one(p.stdout)

        def canon(ev):
            d = json.loads(ev.canonical())
            d.pop("loader", None)               # which files were touched, in what order, and notes
            return dumps(d)

        check("evidence.from_snapshot(record) and evidence.load_dir(fixture) read the same values",
              canon(E.from_snapshot(snap_rec, "pilot_v1")) == canon(E.load_dir(out, "pilot_v1", exps=["pilot_v1"])))
    shutil.rmtree(s1)


def test_step1_replay_fixture():
    d = REPLAY_FIXTURES / "step1"
    agg_p, sel_p = d / "select_clusters_by_source.json", d / "select_summary.json"
    if not (agg_p.is_file() and sel_p.is_file()):
        skip("step1 replay fixture", "not pulled yet: 'remote snapshot --exp <exp> > snap.txt' on the cluster, "
                                      "'remote fixture --from snap.txt --out DIR' on the lab, then DIR/step1/* into "
                                      "tests/fixtures/inc_replay/step1/, pinned in its MANIFEST.json")
        return
    agg, sel = json.loads(agg_p.read_text()), json.loads(sel_p.read_text())
    pool = (sel.get("sources") or {}).get("increment_pool") or {}
    got = {s: v.get("increment_pool", {}).get("images", 0) for s, v in agg["sources"].items()}
    check("pulled Step 1: per-source pool images agree with select_summary.sources.increment_pool",
          all(got.get(s, 0) == n for s, n in pool.items()), [(s, n, got.get(s)) for s, n in pool.items()
                                                               if got.get(s, 0) != n][:5])
    for s in ("fvossel__csgo_player_detection", "rf_bishwarup-halder__crop-health-advisor"):
        check("pulled Step 1: %s is in the increment pool" % s, got.get(s, 0) > 0, got.get(s))
    check("pulled Step 1: no unmatched keys", agg.get("unmatched_keys") == 0, agg.get("unmatched_keys"))


LEVERS = {
    "_meta": {"note": "test menu"},
    "levers": {
        "L1": {"policy_action": "inc_build_pilot",
               "param_bounds": {"exp": {"type": "str", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
                                "replay_mode": {"type": "enum", "value_type": "str", "values": ["full"]}}},
        "L2": {"policy_action": "inc_build_realloop",
               "param_bounds": {"exp": {"type": "str", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
                                "replay_mode": {"type": "enum", "value_type": "str", "values": ["sample", "full"]},
                                "recipes": {"type": "enum", "value_type": "str",
                                            "values": ["full", "freeze", "lora", "full,freeze", "full,lora"]},
                                "size": {"type": "int", "min": 50, "max": 5000},
                                "n_verified": {"type": "int", "min": 1, "max": 12},
                                "no_truth": {"type": "int", "min": 0, "max": 1}}},
        "L5": {"policy_action": "inc_build_realloop",
               "param_bounds": {"exp": {"type": "str", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
                                "replay_mode": {"type": "enum", "value_type": "str", "values": ["sample", "full"]},
                                "recipes": {"type": "enum", "value_type": "str", "values": ["full"]},
                                "size": {"type": "int", "min": 50, "max": 20000}}},
        "L3": {"policy_action": "inc_relevance_build",
               "param_bounds": {"sample": {"type": "int", "min": 100, "max": 2000},
                                "seed": {"type": "int", "min": 0, "max": 99}}},
        "L4": {"policy_action": "inc_label_audit", "param_bounds": {"nshards": {"type": "int", "min": 1, "max": 64}}},
        "L8": {"policy_action": "inc_build_baseline",
               "param_bounds": {"exp": {"type": "str", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"},
                                "seeds": {"type": "enum", "value_type": "str", "values": ["0,1,2"]}}},
        "L7": {"policy_action": "inc_unblock_transient", "param_bounds": {}},
    }}
OCEAN_INC = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/"


def write_levers(obj):
    (TMP / "levers.json").write_text(json.dumps(obj))


class LocalPolicy:
    """policy.describe with policy_actions.json's rows as they are, except the
    cluster's INC_DIR in each path pattern is this test's INC_DIR; 'rows'
    replaces whole rows ({action: row or None for an absent row})."""

    def __init__(self, rows=None):
        self.rows = rows or {}

    def __enter__(self):
        pol = RM._policy()
        self.pol, self.real = pol, pol.describe
        local = re.escape(str(INC) + "/")

        def fix(o):
            if isinstance(o, dict):
                return {k: (v.replace(OCEAN_INC, local) if k == "pattern" and isinstance(v, str) else fix(v))
                        for k, v in o.items()}
            return o

        def describe(action):
            if action in self.rows:
                row = self.rows[action]
                return ({"action": action, "known": False} if row is None
                        else dict(row, action=action, known=True))
            return fix(self.real(action))

        pol.describe = describe
        return self

    def __exit__(self, *exc):
        self.pol.describe = self.real
        return False


def lever_ids(rec):
    return [x["lever"] for x in (rec.get("menu") or {}).get("levers", [])]


def submit_fixtures():
    step1 = INC / "step1"
    step1.mkdir(parents=True, exist_ok=True)
    base_b = step1 / "base_B.jsonl"
    base_b.write_text("{}\n")
    rel = step1 / "relevance.json"
    rel.write_text("{}\n")
    m = INC / "pilot_v1" / "manifests"
    m.mkdir(parents=True, exist_ok=True)
    for n in ("P0", "I1", "Bswap"):
        (m / ("%s.jsonl" % n)).write_text("{}\n")
    return step1, base_b, rel, m


def test_submit():
    write_levers(LEVERS)
    set_queue([])
    check("builder defaults: pilot build's replay mode and flips mode and build-baseline's seeds are the driver's",
          RM.builder_defaults("inc_build_pilot") == {"replay_mode": D.DEFAULT_REPLAY_MODE,
                                                     "gate_flips_mode": D.DEFAULT_FLIPS_MODE}
          and RM.builder_defaults("inc_build_realloop")["gate_flips_mode"] == D.DEFAULT_FLIPS_MODE
          and RM.builder_defaults("inc_build_baseline") == {"seeds": ",".join(map(str, D.SEEDS))})
    try:
        from weed_optimizer_framework.tools.inc import realloop as RL
        from weed_optimizer_framework.tools.inc import relevance as RV
    except Exception as e:                      # modules another change is editing; their own tests cover them
        skip("builder defaults mirror realloop / relevance", "import failed: %s" % e)
    else:
        src = pathlib.Path(RV.__file__).read_text()
        check("builder defaults mirror realloop.N_VERIFIED, relevance.SAMPLE and relevance build --seed",
              RM.REALLOOP_N_VERIFIED == RL.N_VERIFIED and RM.RELEVANCE_SAMPLE == RV.SAMPLE
              and re.search(r'add_argument\("--seed", type=int, default=%d\)' % RM.RELEVANCE_SEED, src),
              (RL.N_VERIFIED, RV.SAMPLE))
        check("--increment-sources: the enum is select.SOURCE_MODES, the default select.SOURCES_RELEVANCE (realloop's "
              "argparse default), 'evidence' select.SOURCES_EVIDENCE",
              RM.INCREMENT_SOURCES == tuple(RL.S.SOURCE_MODES) and RM.INCREMENT_SOURCES_DEFAULT == RL.S.SOURCES_RELEVANCE
              and RM.INCREMENT_SOURCES_EVIDENCE == RL.S.SOURCES_EVIDENCE
              and RM.builder_defaults("inc_build_realloop")["increment_sources"] == RL.S.SOURCES_RELEVANCE,
              (RL.S.SOURCE_MODES, RL.S.SOURCES_RELEVANCE))
    step1, base_b, rel, m = submit_fixtures()
    script = str(PKG_ROOT / "run_inc_build.sh")
    with LocalPolicy():
        _test_submit(step1, base_b, rel, m, script)


def _test_submit(step1, base_b, rel, m, script):
    want = ["pilot", "build", "--exp", "pilot_v2", "--replay-mode", "full"]
    for builder, form in (("build", want),
                          ("build", ["inc.pilot", "build", "--exp", "pilot_v2", "--replay-mode", "full"]),
                          ("build", ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build", "--exp",
                                     "pilot_v2", "--replay-mode", "full"]),
                          ("build", ["sbatch", "run_inc_build.sh", "pilot", "build", "--exp=pilot_v2",
                                     "--replay-mode", "full"]),
                          ("pilot", want[1:])):
        r = RM.submit(builder, form, dry_run=True)
        tail = ["pilot", "build", "--exp=pilot_v2", "--replay-mode", "full"] if "--exp=pilot_v2" in form else want
        check("dry run of L1 as 'submit %s %s'" % (builder, " ".join(form[:3])),
              r["ok"] and r["action"] == "inc_build_pilot" and r["job_name"] == "inc_build_pilot_v2"
              and r["script"] == "run_inc_build.sh"
              and r["sbatch_argv"] == [str(BIN / "sbatch"), "--parsable", "--job-name=inc_build_pilot_v2", script] + tail
              and lever_ids(r) == ["L1"] and r["menu"]["policy_row"] == "present", r.get("error"))
    check("nothing is submitted on a dry run", sbatch_calls() == [])

    r = RM.submit("build", ["realloop", "build", "--exp", "real_v1", "--replay-mode", "full", "--recipes",
                            "full,freeze", "--base", str(base_b), "--n-verified", "6", "--size", "300",
                            "--no-truth", "--relevance", str(rel)], dry_run=True)
    check("realloop build with every flag: lever bounds over the policy row's (paths from the policy row)",
          r["ok"] and r["params"]["no_truth"] is True and r["params"]["size"] == 300 and lever_ids(r) == ["L2"]
          and r["menu"]["levers"][0]["declared_by_policy"] == ["base", "relevance"], r.get("error"))
    r = RM.submit("realloop", ["build", "--exp", "real_v1", "--replay-mode", "sample", "--recipes", "full",
                               "--size", "10000"], dry_run=True)
    check("a size only L5 admits: any lever of the action may admit", r["ok"] and lever_ids(r) == ["L5"],
          r.get("error"))
    r = RM.submit("build", ["pilot", "build-baseline", "--exp", "base_b_v1", "--manifest", str(base_b),
                            "--seeds", "0,1,2"], dry_run=True)
    check("pilot build-baseline (L8)", r["ok"] and r["action"] == "inc_build_baseline", r.get("error"))
    r = RM.submit("relevance", ["build", "--sample", "300", "--seed", "0"], dry_run=True)
    check("relevance build (L3)", r["ok"] and r["script_args"] == ["build", "--sample", "300", "--seed", "0"]
          and r["job_name"] == "inc_relevance", r.get("error"))
    audit_args = ["--trusted", str(m / "P0.jsonl"), "--audit", "I1=%s" % (m / "I1.jsonl"),
                  "Bswap=%s" % (m / "Bswap.jsonl"), "--out", str(INC / "pilot_v1" / "audit" / "label_audit.json")]
    r = RM.submit("audit", audit_args, dry_run=True)
    check("audit (L4) as run_inc_audit.sh takes it; job named after the experiment; paths from the policy row",
          r["ok"] and r["script_args"] == audit_args and r["job_name"] == "inc_audit_pilot_v1"
          and len(r["params"]["audit"]) == 2
          and r["menu"]["levers"][0]["declared_by_policy"] == ["audit", "out", "trusted"], r.get("error"))

    outside = TMP / "outside.jsonl"
    outside.write_text("{}\n")
    link = INC / "step1" / "link.jsonl"
    if not link.exists():
        os.symlink(outside, link)
    rl = ["realloop", "build", "--exp", "x", "--replay-mode", "full", "--recipes", "full"]
    bad = [
        ("train", ["pilot", "build", "--exp", "x"], "builder"),
        ("build", ["--exp", "x"], "start with the module"),
        ("build", ["driver", "advance", "--exp", "x"], "start with the module"),
        ("build", ["pilot", "build-b0", "--exp", "x"], "does not run"),
        ("build", ["relevance", "build"], "start with the module"),
        ("pilot", ["build-b0", "--exp", "x"], "does not run"),
        ("build", ["pilot", "build", "--exp", "x", "--force"], "not accepted"),
        ("build", ["pilot", "build", "--exp", "x", "--testing"], "not accepted"),
        ("build", ["pilot", "build", "--exp", "x", "--testing-settings=1"], "not accepted"),
        ("build", ["pilot", "build", "--exp", "x", "--testing-settings={}"], "metacharacter"),
        ("build", ["pilot", "build", "--exp", "../x"], "not a name"),
        ("build", ["pilot", "build", "--exp", "a b"], "whitespace"),
        ("build", ["pilot", "build", "--exp", "x" * 65], "not a name"),
        ("build", ["pilot", "build", "--exp", "x;rm"], "metacharacter"),
        ("build", ["pilot", "build", "--exp", "x", "--exp", "y"], "twice"),
        ("build", ["pilot", "build", "--exp", "x", "extra"], "unexpected argument"),
        ("build", ["pilot", "build", "--replay-mode", "full"], "needs --exp"),
        ("build", ["pilot", "build", "--exp"], "needs a value"),
        ("build", ["pilot", "build", "--exp", "x", "--replay-mode", "partial"], "not one of"),
        ("build", ["pilot", "build", "--exp", "x", "--replay-mode", "sample"], "no inc_build_pilot lever admits"),
        ("build", ["pilot", "build", "--exp", "x"],
         "not given, checked at the builder's default: gate_flips_mode=negative, replay_mode=sample"),
        ("build", ["pilot", "build", "--exp", "x", "--replay-mode", "full", "--gate-flips-mode", "both"], "not one of"),
        ("build", ["pilot", "build", "--exp", "x", "--replay-mode", "full", "--gate-flips-mode"], "needs a value"),
        ("build", ["pilot", "build-baseline", "--exp", "b", "--manifest", str(base_b), "--gate-flips-mode", "net"],
         "not accepted"),
        ("pilot", ["build", "--exp=x"], "'replay_mode'='sample' is not one of ['full']"),
        ("build", ["realloop", "build", "--exp", "x", "--replay-mode", "full", "--recipes", "full,full"], "subset"),
        ("build", ["realloop", "build", "--exp", "x", "--replay-mode", "full", "--recipes", "sgd"], "subset"),
        ("build", ["realloop", "build", "--exp", "x", "--replay-mode", "full", "--recipes", "freeze,lora"],
         "no inc_build_realloop lever admits"),
        ("build", ["realloop", "build", "--exp", "x", "--replay-mode", "full", "--recipes", "freeze,full"],
         "no inc_build_realloop lever admits"),
        ("build", rl + ["--size", "0"], "positive integer"),
        ("build", rl + ["--size", "-5"], "positive integer"),
        ("build", rl + ["--size", "99999"], "outside"),
        ("build", rl + ["--no-truth=1"], "takes no value"),
        ("build", rl + ["--relevance", "/etc/hosts"], "outside INC_DIR"),
        ("build", rl + ["--relevance", "step1/relevance.json"], "absolute"),
        ("build", rl + ["--relevance", str(INC / "step1" / ".." / "step1" / "relevance.json")], "'..'"),
        ("build", rl + ["--relevance", str(INC / "step1" / "nope.json")], "no such file"),
        ("build", rl + ["--increment-sources", "both"], "not one of"),
        ("build", rl + ["--increment-sources"], "needs a value"),
        ("build", rl + ["--increment-sources", "evidence", "--relevance", str(rel)], "does not go with it"),
        ("build", rl + ["--increment-sources", "evidence", "--min-evidence", "2"], "not accepted"),
        ("build", rl + ["--base", str(link)], "outside INC_DIR"),
        ("build", ["pilot", "build-baseline", "--exp", "b", "--manifest", str(base_b), "--seeds", "0,0,1"], "distinct"),
        ("build", ["pilot", "build-baseline", "--exp", "b", "--manifest", str(base_b), "--seeds", "0,1"],
         "no inc_build_baseline lever admits"),
        ("build", ["pilot", "build", "--exp", "pilot_v1", "--replay-mode", "full"], "already built"),
        ("relevance", ["build", "--force"], "not accepted"),
        ("relevance", ["build", "--sample", "5"], "outside"),
        ("relevance", ["build", "--out", str(step1 / "relevance_s1.json")], "no declared bound"),
        ("relevance", ["pilot", "build"], "runs inc.relevance"),
        ("audit", audit_args[:-2] + ["--out", str(INC / "pilot_v1" / "label_audit.json")], "required pattern"),
        ("audit", audit_args[:-2] + ["--out", str(INC / "pilot_v1" / "audit" / "label_audit.md")], ".json"),
        ("audit", audit_args[:-2] + ["--out", str(INC / "nope_exp" / "audit" / "a.json")], "built exp"),
        ("audit", ["--trusted", str(m / "P0.jsonl"), "--audit", "I1=%s" % (m / "I1.jsonl"),
                   "I1=%s" % (m / "Bswap.jsonl"), "--out", audit_args[-1]], "given twice"),
        ("audit", ["--trusted", str(m / "P0.jsonl"), "--audit", "=%s" % (m / "I1.jsonl"), "--out", audit_args[-1]],
         "NAME=MANIFEST"),
        ("audit", ["--trusted", str(m / "P0.jsonl"), "--audit", "I.1=%s" % (m / "I1.jsonl"), "--out",
                   audit_args[-1]], "required pattern"),
        ("audit", ["--trusted", str(m / "P0.jsonl"), "--out", audit_args[-1]], "needs --audit"),
        ("audit", ["--trusted", str(m / "P0.jsonl"), "--audit", "--out", audit_args[-1]], "NAME=MANIFEST values"),
    ]
    for builder, args, why in bad:
        r = RM.submit(builder, args, dry_run=True)
        check("refused: %s %s (%s)" % (builder, " ".join(a[:40] for a in args)[:90], why),
              refused(r, why) and r["error_kind"] == "refused", r.get("error"))
    for meta, why in (({"approval_id": "a b"}, "--approval-id"), ({"trigger": "D1;x"}, "--trigger"),
                      ({"trigger": ""}, "--trigger"), ({"decided_by": "human:a b"}, "--decided-by"),
                      ({"parent_exp": "../p"}, "--parent-exp")):
        r = RM.submit("build", want, meta=meta, dry_run=True)
        check("refused provenance field %s" % why, refused(r, why), r.get("error"))

    # the menu itself
    (TMP / "levers.json").unlink()
    check("no levers.json: nothing is submitted", refused(RM.submit("build", want, dry_run=True), "levers.json"))
    write_levers({"levers": [{"id": "L3", "policy_action": "inc_relevance_build"}]})
    check("a list-form levers.json without a row for the action: not on the menu",
          refused(RM.submit("build", want, dry_run=True), "not on the menu"))
    with LocalPolicy({"inc_relevance_build": None}):
        r0 = RM.submit("relevance", ["build"], dry_run=True)
        r1 = RM.submit("relevance", ["build", "--sample", "300"], dry_run=True)
    check("no (valid) policy row: refused, even a parameter-free command a lever without bounds would admit",
          refused(r0, "no valid row for inc_relevance_build") and refused(r1, "the policy table is the authority"),
          (r0.get("error"), r1.get("error")))
    with LocalPolicy({"inc_relevance_build": {"param_bounds": {"sample": {"type": "int", "min": 200, "max": 400}}}}):
        r1 = RM.submit("relevance", ["build", "--sample", "300"], dry_run=True)
        r2 = RM.submit("relevance", ["build", "--sample", "500"], dry_run=True)
    check("a lever row without bounds takes its policy row's",
          r1["ok"] and r1["menu"]["levers"] == [{"lever": "L3", "declared_by_lever": [],
                                                 "declared_by_policy": ["sample"]}], r1.get("error"))
    check("the policy row's bounds refuse what they refuse", refused(r2, "outside"), r2.get("error"))
    write_levers({"levers": {"L3": {"policy_action": "inc_relevance_build",
                                    "param_bounds": {"sample": {"type": "int", "min": 50, "max": 5000}}}}})
    with LocalPolicy({"inc_relevance_build": {"param_bounds": {"sample": {"type": "int", "min": 200, "max": 400}}}}):
        r3 = RM.submit("relevance", ["build", "--sample", "4000"], dry_run=True)
    check("a lever cannot widen its policy row", refused(r3, "policy_actions.json inc_relevance_build refuses"),
          r3.get("error"))
    with LocalPolicy({"inc_relevance_build": {"param_bounds": {"sample": {"type": "int", "min": 50, "max": 5000},
                                                               "seed": {"type": "int", "min": 5, "max": 9}}}}):
        r4 = RM.submit("relevance", ["build", "--sample", "400"], dry_run=True)
        r5 = RM.submit("relevance", ["build", "--sample", "400", "--seed", "7"], dry_run=True)
    check("a flag left out is checked at the builder's default by the policy row too (relevance --seed 0)",
          refused(r4, "seed=0") and r5["ok"] and r5["menu"]["defaults_checked"] == {}, (r4.get("error"),
                                                                                      r5.get("error")))
    # a lever's "fixed" values (levers.json L1 fixes replay_mode) bind even when its bounds are wider
    write_levers({"levers": {"L1": {"policy_action": "inc_build_pilot", "fixed": {"replay_mode": "full"},
                                    "param_bounds": {"exp": LEVERS["levers"]["L1"]["param_bounds"]["exp"],
                                                     "replay_mode": {"type": "enum", "value_type": "str",
                                                                     "values": ["sample", "full"]}}}}})
    r6 = RM.submit("build", ["pilot", "build", "--exp", "x", "--replay-mode", "sample"], dry_run=True)
    r7 = RM.submit("build", ["pilot", "build", "--exp", "x"], dry_run=True)
    r8 = RM.submit("build", ["pilot", "build", "--exp", "x", "--replay-mode", "full"], dry_run=True)
    check("a lever's fixed value binds: stated otherwise or left to a default that differs, refused",
          refused(r6, "'replay_mode' must be 'full' (the lever fixes it)") and refused(r7, "must be 'full'")
          and refused(r7, "replay_mode=sample") and r8["ok"], (r6.get("error"), r7.get("error"), r8.get("error")))
    write_levers(LEVERS)
    r = RM.submit("build", ["realloop", "build", "--exp", "real_v1", "--replay-mode", "full", "--recipes", "full"],
                  dry_run=True)
    check("the defaults a bound declares are reported as checked (realloop --n-verified 6, --no-truth absent, "
          "--gate-flips-mode negative, --increment-sources relevance)",
          r["ok"] and r["menu"]["defaults_checked"] == {"gate_flips_mode": "negative", "n_verified": 6,
                                                        "no_truth": 0, "increment_sources": "relevance"},
          r.get("menu"))
    r = RM.submit("build", ["realloop", "build", "--exp", "real_v1", "--replay-mode", "full", "--recipes", "full",
                            "--increment-sources", "evidence", "--gate-flips-mode", "net"], dry_run=True)
    check("a real loop on source-level species evidence (--increment-sources evidence, no relevance file) is "
          "admitted: the policy row declares the enum",
          r["ok"] and r["params"]["increment_sources"] == "evidence" and "relevance" not in r["params"]
          and "increment_sources" not in r["menu"]["defaults_checked"], (r.get("error"), r.get("menu")))
    r = RM.submit("build", ["pilot", "build", "--exp", "pilot_v3", "--replay-mode", "full", "--gate-flips-mode", "net"],
                  dry_run=True)
    check("pilot build ... --gate-flips-mode net is admitted (this test menu's L1; the policy row declares the flag)",
          r["ok"] and any(x["lever"] == "L1" and "gate_flips_mode" in x["declared_by_policy"]
                          for x in r["menu"]["levers"]), (r.get("error"), r.get("menu")))
    r = RM.submit("build", ["realloop", "build", "--exp", "real_v1", "--replay-mode", "full", "--recipes", "full",
                            "--gate-flips-mode", "net"], dry_run=True)
    check("a real loop on a v2 pilot's gate (--gate-flips-mode net) is admitted",
          r["ok"] and "gate_flips_mode" not in r["menu"]["defaults_checked"], (r.get("error"), r.get("menu")))

    # the real sbatch path, with a fake sbatch
    os.environ["INC_ALLOW_DRIFT"] = "1"
    os.environ["INC_BUILD_ALLOW_DRIFT"] = "1"
    os.environ["SLURM_JOB_ID"] = "999"
    shutil.rmtree(INC / "logs", ignore_errors=True)
    try:
        r = RM.submit("build", want, meta={"parent_exp": "pilot_v1", "trigger": "D1,D4", "approval_id": "ab12cd",
                                           "decided_by": "human:harry@example.org"})
    finally:
        for k in ("INC_ALLOW_DRIFT", "INC_BUILD_ALLOW_DRIFT", "SLURM_JOB_ID"):
            os.environ.pop(k)
    calls = sbatch_calls()
    env = calls[-1]["env"] if calls else {}
    check("submit: job id from sbatch --parsable (a warning line before it)", r["ok"] and r["job_id"] == "4242",
          r.get("error"))
    check("sbatch got the validated command line",
          calls and calls[-1]["argv"] == ["--parsable", "--job-name=inc_build_pilot_v2", script] + want)
    check("the job's environment: INCAP_* provenance, no test-mode scoring, no drift override, no enclosing job",
          env.get("INCAP_PARENT_EXP") == "pilot_v1" and env.get("INCAP_TRIGGER") == "D1,D4"
          and env.get("INCAP_APPROVAL_ID") == "ab12cd" and env.get("INCAP_DECIDED_BY") == "human:harry@example.org"
          and env.get("INCAP_REQUESTED_UTC") and "INC_SCORER_TESTING" not in env and "INC_ALLOW_DRIFT" not in env
          and "INC_BUILD_ALLOW_DRIFT" not in env and "SLURM_JOB_ID" not in env, sorted(env))
    check("the log directories exist before sbatch", (INC / "logs").is_dir() and (INC / "step1" / "logs").is_dir())
    set_queue(["777|inc_build_pilot_v2|PENDING|0:00|2026-09-27T01:00:00"])
    r = RM.submit("build", want)
    check("a second build of the same experiment while one is queued: refused",
          refused(r, "already queued") and len(sbatch_calls()) == 1, r.get("error"))
    set_queue(["778|inc_build|RUNNING|0:10|2026-09-27T01:00:00"])
    r = RM.submit("build", want)
    check("a build submitted by hand (the script's own job name, experiment unknown): refused",
          refused(r, "under the script's own name") and len(sbatch_calls()) == 1, r.get("error"))
    set_queue(["779|inc_audit|PENDING|0:00|2026-09-27T01:00:00"])
    r = RM.submit("audit", audit_args)
    check("... the same for an audit submitted by hand", refused(r, "inc_audit is queued or running as job 779"),
          r.get("error"))
    set_queue(["779|inc_audit_pilot_v1|PENDING|0:00|2026-09-27T01:00:00"])
    r = RM.submit("audit", audit_args)
    check("... and for the autopilot's own audit of the experiment", refused(r, "already queued or running"),
          r.get("error"))
    set_queue([])
    for where in (INC / "audit" / "pilot_v1_audit.json", INC / "pilot_v1" / "audit" / "label_audit.json"):
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text("{}\n")
        r = RM.submit("audit", audit_args, dry_run=True)
        check("an audit of pilot_v1 already at %s: the autopilot does not run it again"
              % where.relative_to(INC), refused(r, "already exists") and str(where) in r["error"], r.get("error"))
        where.unlink()
    (TMP / "squeue_fail").write_text("x")
    r = RM.submit("build", want)
    (TMP / "squeue_fail").unlink()
    check("squeue down: refused (a duplicate cannot be ruled out)", refused(r, "squeue unavailable"))
    (TMP / "sbatch_fail").write_text("x")
    r = RM.submit("build", want)
    (TMP / "sbatch_fail").unlink()
    check("sbatch failing: ok false, kind submit, its stderr kept",
          r["ok"] is False and r["error_kind"] == "submit" and "Invalid qos" in r["error"], r.get("error"))
    p = run_cli("submit", "build", "--approval-id", "x1", "--trigger=D1", "--dry-run", "--", *want)
    rec = RM.parse_one(p.stdout, verb="submit")
    check("cli submit with options before '--'",
          rec["ok"] and rec["dry_run"] and rec["provenance"] == {"approval_id": "x1", "trigger": ["D1"]},
          rec.get("error"))
    p = run_cli("submit", "build", "--bogus", "1", "--", *want)
    check("cli submit refuses an unknown option before '--'",
          RM.parse_one(p.stdout)["error_kind"] == "usage" and p.returncode == 1)


def render_lever_argv(argv, values):
    """A levers.json argv template with values: '{k}' tokens (inside a token too),
    '{*k}' expands a list, {"if": k, "tokens": [...]} only when k has a value."""
    out = []
    for tok in argv:
        if isinstance(tok, dict):
            if values.get(tok.get("if")) is not None:
                out += render_lever_argv(tok.get("tokens") or [], values)
            continue
        m = re.fullmatch(r"\{\*([a-z_]+)\}", tok)
        if m:
            out += [str(v) for v in values[m.group(1)]]
            continue
        out.append(re.sub(r"\{([a-z_]+)\}", lambda mm: str(values[mm.group(1)]), tok))
    return out


def lever_values(bounds, fixed):
    vals = dict(fixed)
    for k, b in (bounds or {}).items():
        if k in vals or not isinstance(b, dict):
            continue
        if b.get("type") == "enum":
            vals[k] = b["values"][0]
        elif b.get("type") == "int":
            vals[k] = min(b["max"], max(b["min"], 300))
    return vals


def test_menu_integration():
    """The package's own levers.json and executor.render against remote's
    grammar and menu (skipped while those files are not there)."""
    path = pathlib.Path(RM.__file__).with_name("levers.json")
    if not path.is_file():
        skip("the package's levers.json against submit", "not written yet (component 3)")
        return
    try:
        _, items = RM.load_levers(path)
    except RM.Refused as e:
        check("the package's levers.json loads", False, e)
        return
    check("the package's levers.json loads", True)
    step1, base_b, rel, m = submit_fixtures()
    set_queue([])
    paths = {"exp": "child_v1", "base": str(base_b), "relevance": str(rel), "manifest": str(base_b),
             "trusted": str(m / "P0.jsonl"), "audits": ["I1=%s" % (m / "I1.jsonl"), "Bswap=%s" % (m / "Bswap.jsonl")],
             "out": str(INC / "pilot_v1" / "audit" / "label_audit.json")}
    os.environ["INCAP_LEVERS_JSON"] = str(path)
    try:
        with LocalPolicy():
            for lid, row in items:
                act = row.get("policy_action")
                if act not in RM.FORMS or not isinstance(row.get("argv"), list):
                    continue
                via = row.get("via") or row.get("script") or ""
                builder = {"run_inc_build.sh": "build", "run_inc_relevance.sh": "relevance",
                           "run_inc_audit.sh": "audit"}.get(via)
                vals = lever_values(row.get("param_bounds"), dict(paths, **(row.get("fixed") or {})))
                argv = render_lever_argv(row["argv"], vals)
                r = RM.submit(builder, argv, dry_run=True)
                check("levers.json %s (%s): its argv, filled in, is admitted by %s itself"
                      % (lid, act, lid), r["ok"] and lid in lever_ids(r), (argv, r.get("error")))
            r = RM.submit("build", ["pilot", "build", "--exp", "pilot_v9"], dry_run=True)
            r2 = RM.submit("build", ["pilot", "build", "--exp", "pilot_v9", "--replay-mode", "sample"], dry_run=True)
            check("the package's menu: pilot build without --replay-mode (sample replay by default) is refused "
                  "like the stated sample", refused(r, "replay_mode") and refused(r2, "replay_mode"),
                  (r.get("error"), r2.get("error")))
            r = RM.submit("build", ["pilot", "build", "--exp", "pilot_v3", "--replay-mode", "full",
                                    "--gate-flips-mode", "net"], dry_run=True)
            check("the package's menu: L9's command is admitted by L9 (and L1, which carries a parent's gate)",
                  r["ok"] and {"L1", "L9"} <= set(lever_ids(r)), (r.get("error"), lever_ids(r) if r["ok"] else None))
            r = RM.submit("build", ["pilot", "build", "--exp", "pilot_v3", "--replay-mode", "sample"], dry_run=True)
            check("the package's menu: a sample pilot left at the default gate is not L9 (L9 fixes net)",
                  not r["ok"] and "'gate_flips_mode' must be 'net' (the lever fixes it), got 'negative'"
                  in (r.get("error") or ""), r.get("error"))
            try:
                from weed_optimizer_framework.tools.inc_autopilot import executor as EX
            except Exception as e:              # a sibling module that does not import is its own test's failure
                skip("executor.render against submit", "executor does not import: %s" % e)
                return
            params = {"inc_build_pilot": {"exp": "child_v1", "replay_mode": "full"},
                      "inc_build_baseline": {"exp": "child_v1", "manifest": str(base_b)},
                      "inc_build_realloop": {"exp": "child_v1", "replay_mode": "full", "recipes": "full",
                                             "relevance": str(rel), "size": 300, "n_verified": 6, "no_truth": 1},
                      "inc_relevance_build": {},
                      "inc_label_audit": {"trusted": paths["trusted"], "audit": ",".join(paths["audits"]),
                                          "out": paths["out"]}}
            for act, prm in params.items():
                try:
                    rendered = EX.render(act, prm)
                except Exception as e:
                    check("executor.render(%s) renders" % act, False, e)
                    continue
                remote = rendered.get("remote") or []
                ok = bool(remote) and remote[0] == "submit"
                r = {}
                if ok:
                    builder, meta, _, args = RM._split_submit(remote[1:])
                    r = RM.submit(builder, args, meta=meta, dry_run=True)
                check("executor.render(%s) gives a remote line submit accepts" % act,
                      ok and r.get("ok") and r.get("action") == act, (remote, r.get("error")))
            ev_prm = {"exp": "child_v1", "base": str(base_b), "replay_mode": "full", "recipes": "full",
                      "increment_sources": "evidence", "gate_flips_mode": "net"}
            rendered = EX.render("inc_build_realloop", ev_prm)
            builder, meta, _, args = RM._split_submit(rendered["remote"][1:])
            r = RM.submit(builder, args, meta=meta, dry_run=True)
            check("executor.render of an evidence real loop (R7's L2: --increment-sources evidence before "
                  "--gate-flips-mode, no --relevance) is admitted by the package's L2",
                  rendered["builder"][-4:] == ["--increment-sources", "evidence", "--gate-flips-mode", "net"]
                  and r.get("ok") and "L2" in lever_ids(r) and r["params"].get("increment_sources") == "evidence",
                  (rendered.get("builder"), r.get("error")))
            sized = dict(ev_prm, size=287, n_verified=4)
            rendered = EX.render("inc_build_realloop", sized)
            builder, meta, _, args = RM._split_submit(rendered["remote"][1:])
            r = RM.submit(builder, args, meta=meta, dry_run=True)
            check("executor.render of a sized evidence real loop (R8's L2: --increment-sources evidence --size 287 "
                  "--n-verified 4 --gate-flips-mode net) is admitted by the package's L2, both values as given",
                  rendered["builder"][-8:] == ["--increment-sources", "evidence", "--size", "287", "--n-verified", "4",
                                               "--gate-flips-mode", "net"]
                  and r.get("ok") and "L2" in lever_ids(r)
                  and (r["params"].get("size"), r["params"].get("n_verified")) == (287, 4),
                  (rendered.get("builder"), r.get("error"), r.get("params")))
            for bad, why in (({"n_verified": 13}, "n_verified"), ({"size": 20001}, "size")):
                rendered = EX.render("inc_build_realloop", dict(sized, **bad))
                builder, meta, _, args = RM._split_submit(rendered["remote"][1:])
                r = RM.submit(builder, args, meta=meta, dry_run=True)
                check("... and one with %s outside L2's bounds is refused by submit" % bad,
                      not r.get("ok") and why in (r.get("error") or ""), r.get("error"))
            for act, prm, verb in (("inc_unblock_transient", {"exp": "x", "unit": "chain:full", "cause": "transient"},
                                    "unblock"), ("inc_cancel_exp", {"exp": "x"}, "cancel"),
                                   ("inc_sync_outer", {}, "sync-outer"), ("inc_snapshot", {"exp": "x"}, "snapshot"),
                                   ("inc_advance", {"exp": "x"}, "advance"), ("inc_report", {"exp": "x"}, "report")):
                try:
                    remote = EX.render(act, prm).get("remote") or []
                except Exception as e:
                    if verb in ("cancel", "sync-outer") and "no '%s' verb" % verb in str(e):
                        skip("executor.render(%s)" % act, "executor.py still lists %r as a pending remote verb "
                                                         "(PENDING_REMOTE_VERBS); remote.py has it now" % verb)
                    else:
                        check("executor.render(%s) renders" % act, False, e)
                    continue
                parses = True
                try:
                    ap = RM._Parser()
                    if verb == "unblock":
                        ap.add_argument("--exp", required=True)
                        ap.add_argument("--unit", required=True)
                        ap.add_argument("--reason", required=True)
                        a = ap.parse_args(remote[1:])
                        parses = a.reason.startswith(RM.AUTO_PREFIX)
                    elif verb in ("sync-outer", "cancel"):
                        if verb == "cancel":
                            ap.add_argument("--exp", required=True)
                        ap.add_argument("--approval-id")
                        ap.add_argument("--decided-by")
                        ap.add_argument("--dry-run", action="store_true")
                        ap.parse_args(remote[1:])
                    else:
                        ap.add_argument("--exp", required=True)
                        ap.parse_args(remote[1:])
                except RM._ArgError:
                    parses = False
                check("executor.render(%s) names the remote verb %s with arguments it takes" % (act, verb),
                      remote[:1] == [verb] and parses, remote)
    finally:
        os.environ["INCAP_LEVERS_JSON"] = str(TMP / "levers.json")


def actions_log():
    p = INC / "_campaign" / "provenance" / "_actions.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_cancel():
    queue = ["123_[0-5%40]|inc_rx_v1_0007|PENDING|0:00|x", "123_6|inc_rx_v1_0007|RUNNING|1:00|x",
             "130|inc_build_rx_v1|RUNNING|0:10|x", "131|inc_rx_v1_extra|RUNNING|0:10|x",
             "140|inc_audit_rx_v1|PENDING|0:00|x", "150|inc_rx_v10_0001|PENDING|0:00|x",
             "160|other|RUNNING|0:00|x"]
    set_queue(queue)
    calls = TMP / "scancel_calls.txt"
    sc = BIN / "scancel"
    sc.write_text("#!/bin/bash\necho \"$@\" >> %s\n[ -f %s/scancel_fail ] && { echo 'scancel: error' >&2; exit 1; }\n"
                  "exit 0\n" % (calls, TMP))
    sc.chmod(0o755)
    os.environ["INCAP_SCANCEL"] = str(sc)
    marker = INC / "_campaign" / "abandoned" / "rx_v1.json"
    meta = {"approval_id": "ap1", "decided_by": "human:harry@example.org"}
    try:
        r = RM.cancel("rx_v1", dry_run=True, meta=meta)
        check("cancel --dry-run: the experiment's arrays, build and audit, not a longer name or another exp; "
              "no scancel, no marker, no log",
              r["ok"] and r["job_ids"] == ["123", "130", "140"] and not calls.exists() and not marker.exists()
              and not actions_log(), r.get("job_ids"))
        # an in-job advance submits array 0008 while the first scancel runs: squeue shows it on the re-read
        set_queue(queue, then=queue + ["125|inc_rx_v1_0008|PENDING|0:00|x"])
        n_log = len(actions_log())
        r = RM.cancel("rx_v1", meta=meta)
        m = json.loads(marker.read_text()) if marker.exists() else {}
        log = actions_log()
        check("cancel scancels exactly those ids, then the array submitted meanwhile",
              r["ok"] and calls.read_text().split() == ["123", "130", "140", "125"]
              and r["late_job_ids"] == ["125"], (r.get("error"), calls.read_text() if calls.exists() else None))
        check("cancel writes the abandonment marker: who, which approval, which jobs",
              m.get("exp") == "rx_v1" and m.get("approval_id") == "ap1"
              and m.get("decided_by") == "human:harry@example.org" and m["cancels"][-1]["job_ids"] == ["123", "130",
                                                                                                        "140"]
              and r["abandoned"] == str(marker), m)
        check("... and one _actions.jsonl record of who authorised it",
              len(log) == n_log + 1 and log[-1]["verb"] == "cancel" and log[-1]["exp"] == "rx_v1"
              and log[-1]["approval_id"] == "ap1" and log[-1]["decided_by"] == "human:harry@example.org"
              and log[-1]["ok"] is True and log[-1]["detail"]["late_job_ids"] == ["125"], log[-1:])

        a = RM.advance("rx_v1", backend=D.FakeBackend())
        check("an abandoned experiment: advance refused (abandoned), before the driver runs",
              refused(a, "abandoned") and a["error_kind"] == "abandoned", a.get("error"))
        u = RM.unblock("rx_v1", "chain:full", "auto: transient", backend=D.FakeBackend())
        check("... unblock refused", refused(u, "abandoned") and u["error_kind"] == "abandoned", u.get("error"))
        set_queue([])
        c = RM.campaign_snapshot(["rx_v1"], do_advance=True, step1=False, backend=D.FakeBackend())
        check("... campaign-snapshot --advance skips it (the batch stays ok) and its snapshot shows the marker",
              c["ok"] and c["experiments"]["rx_v1"]["advance"].get("skipped") == "abandoned"
              and c["experiments"]["rx_v1"]["snapshot"]["abandoned"]["approval_id"] == "ap1",
              c["experiments"]["rx_v1"].get("advance"))
        e = {x["exp"]: x for x in RM.status()["experiments"]}
        check("... and status names it", (e["rx_v1"].get("abandoned") or {}).get("approval_id") == "ap1",
              e.get("rx_v1"))

        set_queue(queue)
        (TMP / "scancel_fail").write_text("x")
        r = RM.cancel("rx_v1")
        (TMP / "scancel_fail").unlink()
        check("scancel failing: ok false, kind cancel; the marker keeps both cancels; the failure is logged",
              r["ok"] is False and r["error_kind"] == "cancel" and len(json.loads(marker.read_text())["cancels"]) == 2
              and actions_log()[-1]["ok"] is False)
        set_queue([])
        r = RM.cancel("rx_v1")
        check("nothing queued: ok, nothing cancelled", r["ok"] and r["job_ids"] == [])
        (TMP / "squeue_fail").write_text("x")
        r = RM.cancel("rx_v1")
        (TMP / "squeue_fail").unlink()
        check("squeue down: refused", r["ok"] is False and r["error_kind"] == "squeue")
        check("cancel refuses a bad name", RM.cancel("a;b")["error_kind"] == "bad_name")
        r = RM.cancel("rx_other", meta={"approval_id": "a b"})
        check("cancel refuses a bad approval id before anything is written",
              refused(r, "--approval-id") and not (INC / "_campaign" / "abandoned" / "rx_other.json").exists())
        p = run_cli("cancel", "--exp", "rx_v1", "--approval-id", "x1", "--decided-by", "human:a@b.org", "--dry-run")
        rec = RM.parse_one(p.stdout, verb="cancel")
        check("cli cancel takes --approval-id and --decided-by",
              rec["ok"] and rec["provenance"] == {"approval_id": "x1", "decided_by": "human:a@b.org"}, rec.get("error"))

        marker.unlink()                                  # a person resumes it
        a = RM.advance("rx_v1", backend=D.FakeBackend())
        check("the marker removed: advance runs again", a["ok"], a.get("error"))
    finally:
        os.environ.pop("INCAP_SCANCEL")
        set_queue([])


def test_sync_outer():
    """Runs last: it creates package copies under REPO, which the driver then
    compares its running code with."""
    repo = pathlib.Path(C.REPO)
    nested, outer = repo / "weed_llm_benchmark" / "weed_optimizer_framework", repo / "weed_optimizer_framework"
    set_queue([])
    r = RM.sync_outer(dry_run=True)
    check("sync-outer without both copies: refused", refused(r, "need both copies"))
    st = state_of("rx_block")
    pinned = st["code"]["modules"]
    real_gate = (PKG_ROOT / "weed_optimizer_framework" / "tools" / "inc" / "gate.py").read_bytes()
    check("rx_block is unfinished and pins the real gate.py",
          not st["done"] and pinned["tools/inc/gate.py"] == hashlib.sha256(real_gate).hexdigest())
    from weed_optimizer_framework.tools.inc import train as T
    check("the protected modules are driver.PINNED_MODULES and train.CODE_MODULES, read from them",
          set(RM.protected_modules()) == set(D.PINNED_MODULES) | set(T.CODE_MODULES)
          and {"tools/mega_trainer.py", "tools/near_dup.py", "tools/inc/train.py"} <= set(RM.protected_modules()))
    for root in (nested, outer):
        (root / "tools" / "inc" / "__pycache__").mkdir(parents=True, exist_ok=True)
        (root / "a.py").write_text("a = 1\n")
        (root / "tools" / "inc" / "gate.py").write_text("# edited gate\n")
        (root / "tools" / "inc" / "train.py").write_text("# train\n")
        (root / "tools" / "mega_trainer.py").write_text("# mega_trainer\n")
        (root / "tools" / "near_dup.py").write_text("# near_dup\n")
    (nested / "a.py").write_text("a = 2\n")
    (nested / "new.py").write_text("new = 1\n")
    (nested / ".hidden").write_text("x\n")
    (nested / "tools" / "inc" / "__pycache__" / "gate.cpython-312.pyc").write_bytes(b"\0")
    (outer / "keep.py").write_text("outer only\n")
    r = RM.sync_outer(dry_run=True)
    check("dry run lists the changed files only (no __pycache__, no dotfile)",
          r["ok"] and [c["module"] for c in r["changes"]] == ["a.py", "new.py"] and r["running"] == ["rx_block"]
          and (outer / "a.py").read_text() == "a = 1\n", (r.get("changes"), r.get("error")))
    check("a dry run writes no _actions.jsonl record", not [x for x in actions_log() if x["verb"] == "sync-outer"])
    r = RM.sync_outer(meta={"approval_id": "ap2", "decided_by": "human:harry@example.org"})
    check("sync writes them; deletes nothing; skips caches",
          r["ok"] and r["written"] == ["a.py", "new.py"] and (outer / "a.py").read_text() == "a = 2\n"
          and (outer / "new.py").is_file() and (outer / "keep.py").is_file() and not (outer / ".hidden").exists()
          and not (outer / "tools" / "inc" / "__pycache__" / "gate.cpython-312.pyc").exists(), r.get("error"))
    log = [x for x in actions_log() if x["verb"] == "sync-outer"]
    check("... and records who authorised it in _actions.jsonl",
          len(log) == 1 and log[0]["approval_id"] == "ap2" and log[0]["decided_by"] == "human:harry@example.org"
          and log[0]["detail"]["written"] == ["a.py", "new.py"] and log[0]["ok"] is True, log)

    # jobs in flight
    (nested / "a.py").write_text("a = 4\n")                  # a change no experiment depends on
    set_queue(["901|inc_rx_block_0003|RUNNING|0:10|x"])
    r = RM.sync_outer(dry_run=True)
    check("an experiment's run array in squeue is not in flight for the sync (its modules are checked instead)",
          r["ok"] and r["in_flight"] == [], r.get("error"))
    for name in ("inc_build_pilot_v9", "inc_build", "inc_audit_pilot_v1", "inc_relevance", "inc_verify", "inc_plan",
                 "inc_rx_block_extra"):
        set_queue(["900|%s|RUNNING|0:10|x" % name, "901|inc_rx_block_0003|RUNNING|0:10|x"])
        r = RM.sync_outer(dry_run=True)
        check("%s queued or running: refused (it imports modules mid-job)" % name,
              refused(r, "other than experiment run arrays") and [j["name"] for j in r["in_flight"]] == [name],
              r.get("error"))
    set_queue([])
    (TMP / "squeue_fail").write_text("x")
    r = RM.sync_outer()
    (TMP / "squeue_fail").unlink()
    check("squeue down: refused, nothing written", r["ok"] is False and r["error_kind"] == "squeue"
          and (outer / "a.py").read_text() == "a = 2\n")

    (nested / "tools" / "inc" / "gate.py").write_text("# another gate\n")
    (nested / "a.py").write_text("a = 3\n")
    r = RM.sync_outer()
    check("a pinned module of an unfinished experiment would change: refused, nothing written",
          refused(r, "rx_block:tools/inc/gate.py") and (outer / "a.py").read_text() == "a = 2\n"
          and (outer / "tools" / "inc" / "gate.py").read_text() == "# edited gate\n", r.get("error"))
    check("... and the refusal is recorded too", actions_log()[-1]["verb"] == "sync-outer"
          and actions_log()[-1]["ok"] is False)
    (nested / "tools" / "inc" / "gate.py").write_bytes(real_gate)
    r = RM.sync_outer(dry_run=True)
    check("a change back to the pinned hash is allowed", r["ok"] and "tools/inc/gate.py" in
          [c["module"] for c in r["changes"]], r.get("error"))
    (nested / "tools" / "inc" / "train.py").write_text("# new recipe\n")
    r = RM.sync_outer(dry_run=True)
    check("inc/train.py would change while an experiment is unfinished: refused",
          refused(r, "rx_block:tools/inc/train.py"), r.get("error"))
    (nested / "tools" / "inc" / "train.py").write_text("# train\n")
    (nested / "tools" / "mega_trainer.py").write_text("# mega_trainer, the INC guards changed\n")
    (nested / "tools" / "near_dup.py").write_text("# near_dup changed\n")
    r = RM.sync_outer(dry_run=True)
    check("mega_trainer.py and near_dup.py (train.CODE_MODULES; common.py imports both) would change: refused",
          refused(r, "rx_block:tools/mega_trainer.py") and refused(r, "rx_block:tools/near_dup.py"), r.get("error"))

    marker = INC / "_campaign" / "abandoned" / "rx_block.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"exp": "rx_block"}))
    r = RM.sync_outer(dry_run=True)
    check("an abandoned experiment with no run array in squeue holds nothing",
          r["ok"] and r["abandoned"] == ["rx_block"] and r["running"] == [], r.get("error"))
    set_queue(["901|inc_rx_block_0003|RUNNING|0:10|x"])
    r = RM.sync_outer(dry_run=True)
    check("... but one whose array still runs does", refused(r, "rx_block:tools/mega_trainer.py"), r.get("error"))
    set_queue([])
    marker.unlink()

    a = RM.advance("rx_block")
    check("advance while the outer copy differs from the nested one: code_drift, worded as D7 matches",
          refused(a, "differs from the git-tracked copy") and a["error_kind"] == "code_drift"
          and "tools/mega_trainer.py" in a["outer_drift"]["drift"], a.get("error"))
    u = RM.unblock("rx_block", "chain:full", "auto: transient")
    check("unblock refuses before the drift check on a non-transient cause", refused(u, "not a transient one"))
    p = run_cli("sync-outer", "--dry-run")
    check("cli sync-outer: one line", len(p.stdout.splitlines()) == 1 and RM.parse_one(p.stdout)["verb"] == "sync-outer")
    shutil.rmtree(repo / "weed_llm_benchmark")
    shutil.rmtree(outer)

    # as in production: remote.py runs from the nested copy at REPO/weed_llm_benchmark, where the
    # driver's own drift check (driver_code: compare only when the running copy is not the nested
    # one) never compares; the jobs import the outer copy
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".*")
    shutil.copytree(PKG_ROOT / "weed_optimizer_framework", nested, ignore=ign)
    shutil.copytree(nested, outer, ignore=ign)
    d0 = RM.outer_drift()
    check("(setup) two identical copies of the package: checked, no drift", d0["checked"] and d0["drift"] == [], d0)
    with open(outer / "tools" / "mega_trainer.py", "a") as fh:
        fh.write("\n# an edit made to the outer copy only\n")
    env = dict(os.environ, PYTHONPATH=str(repo / "weed_llm_benchmark"))
    base = [sys.executable, "-m", "weed_optimizer_framework.tools.inc_autopilot.remote"]
    p = subprocess.run(base + ["advance", "--exp", "rx_block"], capture_output=True, text=True,
                       cwd=str(repo / "weed_llm_benchmark"), env=env, timeout=300)
    rec = RM.parse_one(p.stdout, verb="advance") if p.stdout.strip() else {}
    check("production layout: advance run from the nested copy refuses as code_drift on an outer-only edit",
          rec.get("code", {}).get("nested") is True and rec.get("error_kind") == "code_drift"
          and rec.get("outer_drift", {}).get("drift") == ["tools/mega_trainer.py"] and p.returncode == 1,
          (p.returncode, p.stdout[-400:], p.stderr[-400:]))
    p = subprocess.run(base + ["status"], capture_output=True, text=True, cwd=str(repo / "weed_llm_benchmark"),
                       env=env, timeout=300)
    rec = RM.parse_one(p.stdout, verb="status") if p.stdout.strip() else {}
    check("production layout: status names the drifted module",
          rec.get("ok") and rec["outer_drift"]["checked"] and rec["outer_drift"]["drift"] == ["tools/mega_trainer.py"],
          (p.stdout[-400:], p.stderr[-300:]))
    shutil.rmtree(repo / "weed_llm_benchmark")
    shutil.rmtree(outer)


BUILD_MODULES = None


def script_modules():
    text = (PKG_ROOT / "run_inc_build.sh").read_text()
    m = re.search(r"^MODULES=\((.*?)\)", text, re.S | re.M)
    return m.group(1).split()


def import_closure(starts):
    """tools/ modules reachable from the given tools/inc modules through their
    relative imports (lazy ones too), not descending out of tools/inc."""
    tools = PKG_ROOT / "weed_optimizer_framework" / "tools"
    seen, todo = set(), list(starts)
    rx = re.compile(r"^\s*from (\.+)([A-Za-z0-9_.]*) import \(?([A-Za-z0-9_, ]*)", re.M)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        if not rel.startswith("tools/inc/"):
            continue
        for dots, mod, names in rx.findall((tools.parent / rel).read_text()):
            base = "tools/inc" if dots == "." else "tools" if dots == ".." else None
            if base is None:
                continue
            cands = ([base + "/" + mod.replace(".", "/") + ".py"] if mod
                     else [base + "/" + n.split()[0] + ".py" for n in names.split(",") if n.split()])
            for c in cands:
                if (tools.parent / c).is_file() and c not in seen:
                    todo.append(c)
    return seen


def test_run_inc_build_sh():
    text = (PKG_ROOT / "run_inc_build.sh").read_text()
    hdr = dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", text, re.M))
    check("sbatch header: GPU-shared, v100-32:1, 5 cpus, 40G, 4 h, log in inc/logs",
          hdr.get("partition") == "GPU-shared" and hdr.get("gres") == "gpu:v100-32:1"
          and hdr.get("cpus-per-task") == "5" and hdr.get("mem") == "40G" and hdr.get("time") == "04:00:00"
          and hdr.get("output", "").endswith("results/framework/inc/logs/%x_%j.out"), hdr)
    check("bash -n", subprocess.run(["bash", "-n", str(PKG_ROOT / "run_inc_build.sh")]).returncode == 0)
    mods = script_modules()
    closure = import_closure(["tools/inc/pilot.py", "tools/inc/realloop.py"])
    expected = closure - {"tools/inc/lora.py",            # training only (inc/train.py imports it to train LoRA)
                          "tools/semisup_labeler.py"}     # embedding only (verify embed / relevance build)
    check("the drift list is the builders' import closure (+ inc/__init__.py)",
          set(mods) == expected | {"tools/inc/__init__.py"}, (sorted(set(mods) ^ (expected | {"tools/inc/__init__.py"}))))

    repo = TMP / "buildrepo"
    inc = TMP / "build_inc"
    shims = TMP / "shims"
    shims.mkdir(exist_ok=True)
    (shims / "python").write_text("#!/bin/bash\nexec %s \"$@\"\n" % sys.executable)
    (shims / "squeue").write_text("#!/bin/bash\n[ -n \"${STUB_SQUEUE_OUT:-}\" ] && echo \"$STUB_SQUEUE_OUT\"\n"
                                  "exit ${STUB_SQUEUE_RC:-0}\n")
    if not shutil.which("sha256sum"):
        (shims / "sha256sum").write_text("#!/bin/bash\nexec %s -c 'import hashlib,sys; p=sys.argv[1]; "
                                         "print(hashlib.sha256(open(p,\"rb\").read()).hexdigest(), p)' \"$@\"\n"
                                         % sys.executable)
    for p in shims.iterdir():
        p.chmod(0o755)
    conda = TMP / "conda.sh"
    conda.write_text("conda() { return 0; }\n")
    stub_builder = (
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "exp = a[a.index('--exp') + 1] if '--exp' in a else [x[6:] for x in a if x.startswith('--exp=')][0]\n"
        "print('[stub] %%s' %% ' '.join(a), flush=True)\n"
        "if exp.startswith('slow'):\n"
        "    import time\n"
        "    open(os.path.join(os.environ['INC_DIR'], exp + '.started'), 'w').close()\n"
        "    time.sleep(120)\n"
        "if exp.startswith('refuse'):\n"
        "    print('[inc.%(m)s] ERROR: a production build refuses without step1/relevance.json', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "d = os.path.join(os.environ['INC_DIR'], exp)\n"
        "os.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'exp.json'), 'w').write(json.dumps({'exp': exp, 'argv': a}))\n")
    stub_driver = ("import os, sys\nprint('[stub driver] %s' % ' '.join(sys.argv[1:]))\n"
                   "sys.exit(int(os.environ.get('STUB_ADVANCE_RC', '0')))\n")
    for root in (repo / "weed_optimizer_framework", repo / "weed_llm_benchmark" / "weed_optimizer_framework"):
        for d in ("", "tools"):
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / "__init__.py").write_text("")
        for mname in mods:
            p = root / mname
            p.parent.mkdir(parents=True, exist_ok=True)
            base = os.path.basename(mname)
            p.write_text(stub_builder % {"m": base[:-3]} if base in ("pilot.py", "realloop.py")
                         else stub_driver if base == "driver.py" else "# stub %s\n" % mname)

    def run(args, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("INCAP_", "INC_", "SLURM_"))}
        env.update({"PATH": "%s:%s" % (shims, os.environ.get("PATH", "")), "INC_BUILD_REPO": str(repo),
                    "INC_BUILD_CONDA_SH": str(conda), "INC_DIR": str(inc), "SLURM_JOB_ID": "777",
                    "INCAP_PARENT_EXP": "pilot_v1", "INCAP_TRIGGER": "D1,D4", "INCAP_APPROVAL_ID": "ab12",
                    "INCAP_DECIDED_BY": "round-scheduler:inc-autopilot", "INCAP_REQUESTED_UTC": "2026-09-27T01:00:00Z",
                    "TMPDIR": str(TMP)})
        env.update(extra or {})
        return subprocess.run(["bash", str(PKG_ROOT / "run_inc_build.sh")] + args, capture_output=True, text=True,
                              env=env, timeout=120)

    def prov(exp):
        p = inc / "_campaign" / "provenance" / ("%s.json" % exp)
        return json.loads(p.read_text()) if p.exists() else None

    p = run(["pilot", "build", "--exp", "pilot_v2", "--replay-mode", "full"])
    pr = prov("pilot_v2")
    a = (pr or {}).get("attempts", [{}])[-1]
    check("a good build: exit 0, the builder and then the advance ran",
          p.returncode == 0 and "[stub] build --exp pilot_v2 --replay-mode full" in p.stdout
          and "[stub driver] advance --exp pilot_v2" in p.stdout and (inc / "pilot_v2" / "exp.json").is_file(),
          (p.returncode, p.stdout[-400:], p.stderr[-400:]))
    check("provenance written outside the experiment, one attempt with every field",
          pr and pr["exp"] == "pilot_v2" and len(pr["attempts"]) == 1 and a["job_id"] == "777"
          and a["argv"] == ["pilot", "build", "--exp", "pilot_v2", "--replay-mode", "full"]
          and a["parent_exp"] == "pilot_v1" and a["trigger"] == ["D1", "D4"] and a["approval_id"] == "ab12"
          and a["decided_by"] == "round-scheduler:inc-autopilot" and a["requested_utc"] == "2026-09-27T01:00:00Z"
          and a["status"] == "advanced" and a["build_rc"] == 0 and a["advance_rc"] == 0 and a["drift"] == []
          and len(a["modules"]) == len(mods) and a.get("finished_utc")
          and not (inc / "pilot_v2" / "provenance").exists(), a)

    p = run(["realloop", "build", "--exp=refuse_real", "--replay-mode", "full", "--recipes", "full"])
    a = (prov("refuse_real") or {}).get("attempts", [{}])[-1]
    check("a builder refusal: its exit status, status build_failed, the ERROR line kept, no advance",
          p.returncode == 1 and a.get("status") == "build_failed" and a.get("build_rc") == 1
          and "[inc.realloop] ERROR: a production build refuses without step1/relevance.json" in a.get("refusal", "")
          and "[stub driver]" not in p.stdout, (p.returncode, a))

    for tok, exp in (("inc.pilot", "dotted_a"), ("weed_optimizer_framework.tools.inc.pilot", "dotted_b")):
        p = run([tok, "build", "--exp", exp, "--replay-mode", "full"])
        a = (prov(exp) or {}).get("attempts", [{}])[-1]
        check("the policy table's module form %s runs inc.pilot; provenance keeps the argv as given" % tok,
              p.returncode == 0 and "[stub] build --exp %s" % exp in p.stdout and a.get("argv", [None])[0] == tok
              and a.get("status") == "advanced", (p.returncode, p.stderr[-300:]))
    p = run(["pilot", "build-baseline", "--exp", "base_b_v1", "--manifest", "/x/base_B.jsonl"],
            {"STUB_ADVANCE_RC": "3"})
    a = (prov("base_b_v1") or {}).get("attempts", [{}])[-1]
    check("a failed advance after a good build: exit 0, recorded",
          p.returncode == 0 and a.get("status") == "built_advance_failed" and a.get("advance_rc") == 3, a)

    gate = repo / "weed_optimizer_framework" / "tools" / "inc" / "gate.py"
    gate.write_text("# outer copy edited\n")
    p = run(["pilot", "build", "--exp", "pilot_v3", "--replay-mode", "full"])
    a = (prov("pilot_v3") or {}).get("attempts", [{}])[-1]
    check("outer/nested drift: refused before the builder, recorded",
          p.returncode == 1 and "FATAL: outer modules differ" in p.stderr and "[stub]" not in p.stdout
          and a.get("status") == "refused_drift" and a.get("drift") == ["tools/inc/gate.py"]
          and not (inc / "pilot_v3").exists(), (p.returncode, p.stderr[-300:], a))
    p = run(["pilot", "build", "--exp", "pilot_v3", "--replay-mode", "full"], {"INC_BUILD_ALLOW_DRIFT": "1"})
    pr = prov("pilot_v3") or {}
    a = pr.get("attempts", [{}])[-1]
    check("drift allowed by hand: runs, recorded as allowed; the refused attempt stays",
          p.returncode == 0 and a.get("drift_allowed") is True and a.get("status") == "advanced"
          and len(pr.get("attempts", [])) == 2 and pr["attempts"][0]["status"] == "refused_drift", a)
    gate.write_text("# stub tools/inc/gate.py\n")

    locks = inc / "_campaign" / "locks"
    check("the build lock is released after a good build, a builder refusal and a failed advance",
          locks.is_dir() and not list(locks.iterdir()), sorted(os.listdir(locks)) if locks.is_dir() else None)
    lk = locks / "lock_a.build"
    for out, rc, state in (("RUNNING", "0", "alive"), ("PENDING", "0", "alive"),
                           ("slurm_load_jobs error: Socket timed out", "1", "unknown")):
        lk.write_text("4321 r001 2026-09-27T01:00:00Z\n")
        p = run(["pilot", "build", "--exp", "lock_a", "--replay-mode", "full"],
                {"STUB_SQUEUE_OUT": out, "STUB_SQUEUE_RC": rc})
        check("the lock held by job 4321 (squeue: %s): refused, nothing written, the lock kept" % out[:20],
              p.returncode == 1 and "another build of lock_a holds" in p.stderr and "its job is %s" % state in p.stderr
              and "[stub]" not in p.stdout and prov("lock_a") is None and not (inc / "lock_a").exists()
              and lk.read_text().startswith("4321 "), (p.returncode, p.stderr[-300:]))
    lk.write_text("none somehost 2026-09-27T01:00:00Z\n")
    p = run(["pilot", "build", "--exp", "lock_a", "--replay-mode", "full"])
    check("a lock without a job id (a run by hand): refused, a person removes it",
          p.returncode == 1 and "its job is unknown" in p.stderr and prov("lock_a") is None, p.stderr[-300:])
    for out, rc in (("slurm_load_jobs error: Invalid job id specified", "1"), ("COMPLETED", "0")):
        lk.write_text("4321 r001 2026-09-27T01:00:00Z\n")
        shutil.rmtree(inc / "lock_a", ignore_errors=True)
        p = run(["pilot", "build", "--exp", "lock_a", "--replay-mode", "full"],
                {"STUB_SQUEUE_OUT": out, "STUB_SQUEUE_RC": rc})
        check("a stale lock (squeue: %s): taken over, the build runs, the lock released" % out[-24:],
              p.returncode == 0 and "took over the stale build lock" in p.stdout and "[stub] build" in p.stdout
              and not lk.exists(), (p.returncode, p.stdout[-300:], p.stderr[-300:]))
    lk.write_text("777 r001 2026-09-27T01:00:00Z\n")          # this job's own id: a requeued job
    shutil.rmtree(inc / "lock_a", ignore_errors=True)
    p = run(["pilot", "build", "--exp", "lock_a", "--replay-mode", "full"], {"STUB_SQUEUE_OUT": "RUNNING"})
    check("a lock of this job's own id (requeued) is taken over", p.returncode == 0 and not lk.exists(),
          (p.returncode, p.stderr[-300:]))

    # SIGTERM (scancel, the time limit) mid-build: recorded, lock released
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INCAP_", "INC_", "SLURM_"))}
    env.update({"PATH": "%s:%s" % (shims, os.environ.get("PATH", "")), "INC_BUILD_REPO": str(repo),
                "INC_BUILD_CONDA_SH": str(conda), "INC_DIR": str(inc), "SLURM_JOB_ID": "778", "TMPDIR": str(TMP)})
    pr = subprocess.Popen(["bash", str(PKG_ROOT / "run_inc_build.sh"), "pilot", "build", "--exp", "slow_a",
                           "--replay-mode", "full"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          env=env, start_new_session=True)
    started = inc / "slow_a.started"
    for _ in range(600):
        if started.exists() or pr.poll() is not None:
            break
        time.sleep(0.1)
    held = (locks / "slow_a.build").exists()
    os.killpg(pr.pid, signal.SIGTERM)
    try:
        out, err = pr.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(pr.pid, signal.SIGKILL)
        out, err = pr.communicate()
    a = (prov("slow_a") or {}).get("attempts", [{}])[-1]
    check("SIGTERM during the build: exit 143, provenance says killed during the build, lock and temp file gone",
          started.exists() and held and pr.returncode == 143 and a.get("status") == "killed"
          and a.get("killed_during") == "build" and a.get("finished_utc") and not (locks / "slow_a.build").exists()
          and not list(TMP.glob("inc_build_slow_a.*")), (pr.returncode, a, err[-300:]))

    for args, why in ((["driver", "advance", "--exp", "x"], "not a builder"), (["pilot", "build"], "no --exp"),
                      (["pilot", "build", "--exp", "../x"], "unsafe --exp"), (["pilot"], "no command"),
                      ([], "nothing")):
        p = run(args)
        check("usage error (%s): exit 2, nothing written" % why,
              p.returncode == 2 and "usage:" in p.stderr and not (inc / "x").exists(), (p.returncode, p.stderr[-200:]))


def test_job_script_for():
    """remote.advance / unblock point the pinned driver at the experiment's own executor (2026-09-29: an advance
    from the login node submitted pilot_v4's runs to run_inc_job.sh, which refused the Protocol v3 recipes)."""
    v2, v1 = "js_v2_exp", "js_v1_exp"
    for e, d in ((v2, {"type": "pilot", "protocol_package": "inc2"}), (v1, {"type": "pilot"})):
        (INC / e).mkdir(parents=True, exist_ok=True)
        (INC / e / "exp.json").write_text(json.dumps(d))
    script = pathlib.Path(C.REPO) / "weed_llm_benchmark" / "run_inc2_job.sh"
    old = os.environ.get("INC_JOB_SCRIPT")
    try:
        os.environ.pop("INC_JOB_SCRIPT", None)
        missing = None
        if not script.exists():
            try:
                RM._job_script_for(v2)
            except RM.Refused as e:
                missing = str(e)
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("#!/bin/bash\n")
        check("a v2 experiment whose job script is missing is refused, before anything is submitted",
              missing is not None and "run_inc2_job.sh" in missing, missing)
        got = RM._job_script_for(v2)
        check("a v2 experiment (protocol_package inc2) sets INC_JOB_SCRIPT to run_inc2_job.sh",
              got == str(script) and os.environ.get("INC_JOB_SCRIPT") == str(script), got)
        got1 = RM._job_script_for(v1)
        check("a v1 experiment removes it again (the driver's default, run_inc_job.sh)",
              got1 is None and "INC_JOB_SCRIPT" not in os.environ)
        from weed_optimizer_framework.tools.inc import driver as DRV
        RM._job_script_for(v2)
        check("  and the pinned driver then submits the v2 script", str(DRV.job_script()) == str(script))
    finally:
        if old is None:
            os.environ.pop("INC_JOB_SCRIPT", None)
        else:
            os.environ["INC_JOB_SCRIPT"] = old


def main():
    make_bin()
    print("the job script per experiment (advance, unblock)")
    test_job_script_for()
    print("markers")
    test_markers()
    print("snapshot of pilot_v1 (local files)")
    test_snapshot_pilot_v1()
    print("a driver experiment: report, snapshot, advance, status, unblock, campaign-snapshot")
    test_driver_experiment()
    print("Step 1 aggregates and fixtures")
    test_step1()
    test_step1_replay_fixture()
    print("submit")
    test_submit()
    print("the package's levers.json and executor.render against submit")
    test_menu_integration()
    print("cancel")
    test_cancel()
    print("the one-line CLI")
    test_cli_one_line()
    print("run_inc_build.sh")
    test_run_inc_build_sh()
    print("sync-outer (last: it creates a nested copy under REPO)")
    test_sync_outer()
    print("\n%d failure(s)" % len(FAILURES))
    if not FAILURES:
        shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
