#!/usr/bin/env python3
r"""WP5 SU-ledger tests -- no cluster, no Mongo, no model.

The gate line this file exists to prove: "SU ledger for the last 20 jobs
matches sacct within 1%" (docs/TIERED_SUPERVISION_EXECUTION.md, WP5). Three
properties are load-bearing and each gets its own section below:

  * **The two rate constants are the hardware's, not a guess.** H100 = 2 SU
    per GPU-hour, V100 = 1, as briefed 2026-09-04 against the campaign's
    11,005 SU balance. Both live in su_rates.json, not compiled into
    su_ledger.py, so a drifted number is a one-file diff, not a code change.
  * **A job's SU is counted once, from its parent allocation.** `sacct`
    without `-X` prints a `.batch` and `.extern` row alongside the parent for
    every job, and an array task's own sub-steps carry its task suffix
    (`<id>_<task>.batch`). Summing all of them would overcount every job in
    the ledger by roughly 3x; `parse_sacct` must fold them back to one row per
    job and the fixtures below are built with realistic sub-step rows to
    prove it does.
  * **Nothing is silently free.** An elapsed time sacct reports as literal
    `Unknown` (a job still being accounted) must not read as 0 SU, and a GPU
    family `su_rates.json` does not know must not read as free compute --
    both must be flagged so the gap is visible instead of absorbed.

Fixtures are given in both sacct layouts this module claims to read: `-P`
(pipe-delimited) and the fixed-width default, built column-true from real
values via `fixed_width()` below rather than hand-padded, because a
hand-padded fixed-width fixture is exactly the kind of test that passes for
the wrong reason.

Run:  python3 tests/test_su_ledger.py
 (or) python -m pytest tests/test_su_ledger.py -q
"""
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import su_ledger as sl  # noqa: E402
from weed_optimizer_framework.tools.brain import corpus as _corpus_mod  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


def ck_eq(name, got, want, tol=1e-6):
    if isinstance(want, float) and isinstance(got, (int, float)):
        cond = abs(float(got) - want) <= tol
    else:
        cond = got == want
    print(("  ok   " if cond else "  FAIL ") + name
          + ("" if cond else "  (got %r, want %r)" % (got, want)))
    if not cond:
        _fails.append(name)


_tmp = tempfile.mkdtemp(prefix="su_ledger_test_")


def newdir():
    """A fresh, empty ledger root -- one per scenario so jobs recorded in one
    test do not show up as `extra_jobs` in another's reconcile()."""
    d = tempfile.mkdtemp(dir=_tmp)
    return d


def fixed_width(header, rows):
    """A column-true fixed-width table: sacct's default layout, right-
    justified fields sized to their own content and header, dash ruler under
    the header. Built from values rather than hand-padded so column
    boundaries are always correct."""
    widths = [max(len(header[i]), max((len(r[i]) for r in rows), default=0))
              for i in range(len(header))]

    def fmt(vals):
        return " ".join(v.rjust(widths[i]) for i, v in enumerate(vals))

    lines = [fmt(header), " ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def entry(job, step="train", actor="round-scheduler", gpu_count=8,
          gpu_type="h100-80", elapsed_s=3600, su=None, round_=1, domain="weed",
          ts="2026-09-06T00:00:00Z"):
    return {"job": job, "domain": domain, "round": round_, "step": step,
            "actor": actor, "gpu_count": gpu_count, "gpu_type": gpu_type,
            "elapsed_s": elapsed_s, "su": su, "sacct_row": None, "ts": ts}


def pipe_sacct(rows, cols=("JobID", "State", "Elapsed", "AllocTRES")):
    """`-P` pipe-delimited sacct text from [(jobid, state, elapsed, tres), ...]."""
    lines = ["|".join(cols)]
    lines += ["|".join(r) for r in rows]
    return "\n".join(lines)


# ============================================================ rate constants
print("-- the two rate constants --")
_r = sl.rates()
ck("rates.json is readable with no errors", _r["errors"] == [])
ck_eq("H100 = 2 SU per GPU-hour", _r["values"]["rates.h100.su_per_gpu_hour"], 2.0)
ck_eq("V100 = 1 SU per GPU-hour", _r["values"]["rates.v100.su_per_gpu_hour"], 1.0)
ck("every rate carries a stated reason",
   all(_r["why"].get(k) for k in _r["values"]))
ck_eq("su_for(h100, 8 GPUs, 1h) = 16 SU (8 x 1h x 2 SU/GPU-h)",
      sl.su_for("h100-80", 8, 3600)["value"], 16.0)
ck_eq("su_for(v100, 4 GPUs, 0.5h) = 2 SU (4 x 0.5h x 1 SU/GPU-h)",
      sl.su_for("v100-32", 4, 1800)["value"], 2.0)
_h100 = sl.su_for("h100", 8, 3600)
ck("a measured, known-rate allocation is not flagged unknown in any way",
   _h100["unknown_rate"] is False and _h100["unknown_elapsed"] is False)
ck_eq("a GPU-free step (gpu_count=0) costs 0 SU, not a gap",
      sl.su_for(None, 0, 3600)["value"], 0.0)
ck("...and that 0 is not itself flagged unknown_rate",
   sl.su_for(None, 0, 3600)["unknown_rate"] is False)

# ==================================================== array-task / substep dedup
print("\n-- array-task and .batch/.extern de-duplication --")
# A plain job with a parent row plus .batch and .extern sub-steps, -P format.
# Each sub-step, read on its own, carries a GPU allocation too (as real sacct
# .extern rows often do) -- if summed, this job would be over-billed 3x.
P_DEDUPE = pipe_sacct([
    ("700", "COMPLETED", "01:00:00", "billing=8,cpu=32,gres/gpu:h100-80=8,mem=200G"),
    ("700.batch", "COMPLETED", "01:00:00", "cpu=32,mem=200G"),
    ("700.extern", "COMPLETED", "01:00:00", "billing=8,gres/gpu:h100-80=8"),
])
_jobs = sl.parse_sacct(P_DEDUPE)
ck_eq("one job record for a parent + .batch + .extern trio", len(_jobs), 1)
ck_eq("its id is the parent's, not a sub-step's", _jobs[0]["jobid"], "700")
ck("its source is explicitly the parent allocation row",
   _jobs[0]["source"] == "parent allocation row")
ck_eq("its GPU count is the parent row's own (8), not summed across sub-steps",
      _jobs[0]["gpu_count"], 8)
ck_eq("its SU is computed once from the parent row: 8 x 1h x 2 = 16, not 32 or 48",
      sl.su_for(_jobs[0]["gpu_type"], _jobs[0]["gpu_count"],
               _jobs[0]["elapsed_s"])["value"], 16.0)
ck_eq("both sub-steps are recorded as sub-steps of the one job",
      sorted(_jobs[0]["substep_ids"]), ["700.batch", "700.extern"])

# The same trio, fixed-width layout, to prove both parsers agree.
FW_DEDUPE = fixed_width(
    ["JobID", "State", "Elapsed", "AllocTRES"],
    [["700", "COMPLETED", "01:00:00", "billing=8,cpu=32,gres/gpu:h100-80=8,mem=200G"],
     ["700.batch", "COMPLETED", "01:00:00", "cpu=32,mem=200G"],
     ["700.extern", "COMPLETED", "01:00:00", "billing=8,gres/gpu:h100-80=8"]])
_jobs_fw = sl.parse_sacct(FW_DEDUPE)
ck_eq("fixed-width: one job record for the same trio", len(_jobs_fw), 1)
ck_eq("fixed-width: same SU as the -P parse",
      sl.su_for(_jobs_fw[0]["gpu_type"], _jobs_fw[0]["gpu_count"],
               _jobs_fw[0]["elapsed_s"])["value"], 16.0)

# An array job: two tasks, each with its own .batch -- must NOT fold together.
P_ARRAY = pipe_sacct([
    ("800_1", "COMPLETED", "00:10:00", "gres/gpu:h100-80=1"),
    ("800_1.batch", "COMPLETED", "00:10:00", ""),
    ("800_2", "COMPLETED", "00:20:00", "gres/gpu:h100-80=1"),
    ("800_2.extern", "COMPLETED", "00:20:00", "gres/gpu:h100-80=1"),
])
_arr = {j["jobid"]: j for j in sl.parse_sacct(P_ARRAY)}
ck_eq("two array tasks stay two separate job records", sorted(_arr), ["800_1", "800_2"])
ck_eq("task 1 keeps its own elapsed time (10 min)", _arr["800_1"]["elapsed_s"], 600.0)
ck_eq("task 2 keeps its own, different elapsed time (20 min)",
      _arr["800_2"]["elapsed_s"], 1200.0)
ck_eq("task 1's own .batch sub-step is attributed to task 1, not task 2",
      _arr["800_1"]["substep_ids"], ["800_1.batch"])

# ============================================================ unknown GPU type
print("\n-- unknown GPU type: a gap, never a free ride --")
_unk = sl.su_for("a100-40", 2, 3600)
ck("an unresolved GPU family is flagged unknown_rate", _unk["unknown_rate"] is True)
ck("...and is NOT reported as a 0-SU allocation",
   _unk["value"] is not None and _unk["value"] > 0.0)
ck_eq("it is charged at the fallback ceiling (2 SU/GPU-h): 2 x 1h x 2 = 4",
      _unk["value"], 4.0)
ck("its reason names the unresolved type", "a100-40" in _unk["reason"])

P_UNKNOWN_GPU = pipe_sacct([("900", "COMPLETED", "01:00:00", "gres/gpu:a100-40=2")])
_unk_job = sl.parse_sacct(P_UNKNOWN_GPU)[0]
_unk_su = sl.su_for(_unk_job["gpu_type"], _unk_job["gpu_count"], _unk_job["elapsed_s"])
ck("an unknown-GPU job parsed from sacct is flagged the same way",
   _unk_su["unknown_rate"] is True and _unk_su["value"] == 4.0)

# ============================================================= unknown elapsed
print("\n-- an Unknown-elapsed pending job contributes 'unknown', never 0 --")
_pending = sl.su_for("h100-80", 8, "Unknown")
ck("unknown elapsed flags unknown_elapsed", _pending["unknown_elapsed"] is True)
ck("...and its value is None, not 0", _pending["value"] is None)

d_pending = newdir()
sl.record(entry("950", elapsed_s="Unknown", su=None), base_dir=d_pending)
_stored = sl._read_deduped("weed", base_dir=d_pending)[0]
ck_eq("the stored entry's elapsed_s is the literal string 'unknown'",
      _stored["elapsed_s"], "unknown")
ck("the stored entry's su.value is None, not 0", _stored["su"]["value"] is None)
_tot_pending = sl.total("weed", base_dir=d_pending)
ck_eq("total() excludes it from the numeric sum", _tot_pending["su"], 0.0)
ck_eq("...and names it under unknown_su_jobs instead of silently costing 0",
      _tot_pending["unknown_su_jobs"], ["950"])
ck_eq("n_unknown counts it", _tot_pending["n_unknown"], 1)

_sacct_still_unknown = pipe_sacct([("950", "RUNNING", "Unknown", "gres/gpu:h100-80=8")])
_rec_pending = sl.reconcile("weed", _sacct_still_unknown, base_dir=d_pending)
ck("reconcile() refuses to certify a job it cannot measure",
   _rec_pending["ok"] is False)
ck_eq("...and names it in unknown_elapsed_jobs",
      [j["job"] for j in _rec_pending["unknown_elapsed_jobs"]], ["950"])

# =============================================================== idempotency
print("\n-- idempotent re-record --")
d_idem = newdir()
r1 = sl.record(entry("960", elapsed_s=1800, su=None), base_dir=d_idem)
ck("first record is not an update", r1["updated"] is False)
ck_eq("first record's SU: 8 x 0.5h x 2 = 8", r1["entry"]["su"]["value"], 8.0)
r2 = sl.record(entry("960", elapsed_s=3600, su=None), base_dir=d_idem)
ck("re-recording the same job+step reports itself as an update",
   r2["updated"] is True)
ck_eq("...and shows the value it superseded",
      r2["previous"]["su"]["value"], 8.0)
ck_eq("...and the new value: 8 x 1h x 2 = 16", r2["entry"]["su"]["value"], 16.0)
_lines = open(str(sl._ledger_path("weed", d_idem))).read().strip().splitlines()
ck_eq("the ledger is append-only: both lines are still on disk", len(_lines), 2)
_tot_idem = sl.total("weed", base_dir=d_idem)
ck_eq("total() reflects the update once, not both writes summed (16, not 24)",
      _tot_idem["su"], 16.0)
ck_eq("...over exactly one entry", _tot_idem["n_entries"], 1)

# =============================================================== reconcile()
print("\n-- reconcile(): within tolerance --")
d_ok = newdir()
sl.record(entry("970", elapsed_s=3600, su=None), base_dir=d_ok)   # true SU = 16
r_ok = sl.reconcile(
    "weed", pipe_sacct([("970", "COMPLETED", "01:00:00", "gres/gpu:h100-80=8")]),
    base_dir=d_ok)
ck("ledger matching sacct exactly reconciles ok", r_ok["ok"] is True)
ck_eq("delta is zero", r_ok["delta"], 0.0)
ck("no missing/extra/unknown jobs", not r_ok["missing_jobs"]
   and not r_ok["extra_jobs"] and not r_ok["unknown_rate_jobs"]
   and not r_ok["unknown_elapsed_jobs"])

print("\n-- reconcile(): outside tolerance --")
d_bad = newdir()
# Ledger says 20 SU; sacct's own truth for this allocation is 16 -- a 25% miss.
sl.record(entry("971", elapsed_s=3600, su=20.0), base_dir=d_bad)
r_bad = sl.reconcile(
    "weed", pipe_sacct([("971", "COMPLETED", "01:00:00", "gres/gpu:h100-80=8")]),
    base_dir=d_bad)
ck("a 25% miss on one job fails reconciliation", r_bad["ok"] is False)
_row_bad = r_bad["jobs"][0]
ck_eq("the job is named with both numbers", _row_bad["job"], "971")
ck_eq("ledger_su as recorded", _row_bad["ledger_su"], 20.0)
ck_eq("sacct_su as measured", _row_bad["sacct_su"], 16.0)
ck("its delta_pct is over the 1% tolerance", abs(_row_bad["delta_pct"]) > 1.0)
ck("it is marked not ok", _row_bad["ok"] is False)

print("\n-- reconcile(): two errors that cancel to a correct-looking total --")
d_cancel = newdir()
# True SU for each is 16. One ledger entry is +2 high, the other -2 low: the
# aggregate ledger total (32) equals the aggregate sacct total (32) exactly,
# so a total-only check would pass this. Neither job is individually correct.
sl.record(entry("972", elapsed_s=3600, su=18.0), base_dir=d_cancel)
sl.record(entry("973", elapsed_s=3600, su=14.0), base_dir=d_cancel)
r_cancel = sl.reconcile(
    "weed", pipe_sacct([("972", "COMPLETED", "01:00:00", "gres/gpu:h100-80=8"),
                        ("973", "COMPLETED", "01:00:00", "gres/gpu:h100-80=8")]),
    base_dir=d_cancel)
ck_eq("the aggregate totals do cancel out exactly", r_cancel["ledger_su"],
      r_cancel["sacct_su"])
ck_eq("...so the aggregate delta is zero", r_cancel["delta"], 0.0)
ck("BUT reconcile() still reports ok:false", r_cancel["ok"] is False)
_by_job_cancel = {j["job"]: j for j in r_cancel["jobs"]}
ck("job 972 is named and marked not ok", "972" in _by_job_cancel
   and _by_job_cancel["972"]["ok"] is False)
ck("job 973 is named and marked not ok", "973" in _by_job_cancel
   and _by_job_cancel["973"]["ok"] is False)
ck_eq("job 972's own delta is +2, not hidden by the cancellation",
      _by_job_cancel["972"]["delta"], 2.0)
ck_eq("job 973's own delta is -2, not hidden by the cancellation",
      _by_job_cancel["973"]["delta"], -2.0)

print("\n-- reconcile(): missing and extra jobs --")
d_miss = newdir()
sl.record(entry("980", elapsed_s=3600, su=None), base_dir=d_miss)  # never in sacct below
r_miss = sl.reconcile(
    "weed", pipe_sacct([("981", "COMPLETED", "01:00:00", "gres/gpu:h100-80=8")]),
    base_dir=d_miss)
ck("a job in sacct but never recorded fails reconciliation", r_miss["ok"] is False)
ck_eq("it is named in missing_jobs", [j["job"] for j in r_miss["missing_jobs"]],
      ["981"])
ck_eq("a job recorded but outside the sacct window is named in extra_jobs",
      [j["job"] for j in r_miss["extra_jobs"]], ["980"])

# ================================================================ aggregates
print("\n-- by_actor / by_step / remaining --")
d_agg = newdir()
sl.record(entry("990", step="collect", actor="round-scheduler", gpu_count=0,
                gpu_type=None, elapsed_s=600, su=None), base_dir=d_agg)
sl.record(entry("991", step="train", actor="tier1:qwen2.5-coder:7b",
                elapsed_s=3600, su=None), base_dir=d_agg)
sl.record(entry("992", step="train", actor="round-scheduler",
                elapsed_s=1800, su=None), base_dir=d_agg)
_by_actor = sl.by_actor("weed", base_dir=d_agg)
ck_eq("round-scheduler's total: 0 (collect) + 8 (half-hour train) = 8",
      _by_actor["round-scheduler"]["su"], 8.0)
ck_eq("tier1's own total: 16 (one-hour train)",
      _by_actor["tier1:qwen2.5-coder:7b"]["su"], 16.0)
_by_step = sl.by_step("weed", base_dir=d_agg)
ck_eq("collect step costs 0 SU (no GPU)", _by_step["collect"]["su"], 0.0)
ck_eq("train step totals 24 SU across both train jobs", _by_step["train"]["su"], 24.0)
_rem = sl.remaining("weed", {"su_envelope": 100, "per_round_cap": 10}, base_dir=d_agg)
ck_eq("remaining SU against a 100 SU envelope: 100 - 24 = 76",
      _rem["remaining_su"], 76.0)
ck_eq("pct_used: 24/100", _rem["pct_used"], 0.24)
_rem_noenv = sl.remaining("weed", {}, base_dir=d_agg)
ck("remaining() with no envelope reports unknown, not the full amount",
   _rem_noenv["remaining_su"] is None and _rem_noenv["reason"])

# ===================================================================== misc
print("\n-- entry identity and validation --")
try:
    sl.record({"job": "", "domain": "weed", "step": "train"}, base_dir=newdir())
    ck("record() rejects an entry with no job id", False)
except ValueError:
    ck("record() rejects an entry with no job id", True)
try:
    sl.record({"job": "1", "domain": "", "step": "train"}, base_dir=newdir())
    ck("record() rejects an entry with no domain", False)
except ValueError:
    ck("record() rejects an entry with no domain", True)

print("\n-- the CLI --")
import contextlib  # noqa: E402
import io  # noqa: E402

d_cli = newdir()
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _code = sl.main(["--base-dir", d_cli, "record",
                     json.dumps(entry("999", elapsed_s=3600, su=None))])
ck_eq("CLI record exits 0", _code, 0)
ck("CLI record prints the stored entry as JSON",
   json.loads(_buf.getvalue())["entry"]["job"] == "999")

_buf2 = io.StringIO()
with contextlib.redirect_stdout(_buf2):
    _code2 = sl.main(["--base-dir", d_cli, "total", "weed"])
ck_eq("CLI total exits 0", _code2, 0)
ck_eq("CLI total reports the job just recorded", json.loads(_buf2.getvalue())["su"], 16.0)

_buf3 = io.StringIO()
with contextlib.redirect_stdout(_buf3):
    _code3 = sl.main(["rates"])
ck_eq("CLI rates exits 0", _code3, 0)
ck_eq("CLI rates prints the same H100 constant",
      json.loads(_buf3.getvalue())["values"]["rates.h100.su_per_gpu_hour"], 2.0)

_sacct_path = pathlib.Path(d_cli) / "sacct.txt"
_sacct_path.write_text(
    pipe_sacct([("999", "COMPLETED", "01:00:00", "gres/gpu:h100-80=8")]),
    encoding="utf-8")
_buf4 = io.StringIO()
with contextlib.redirect_stdout(_buf4):
    _code4 = sl.main(["--base-dir", d_cli, "reconcile", "weed",
                      "--sacct-file", str(_sacct_path)])
ck_eq("CLI reconcile exits 0 when the ledger matches sacct", _code4, 0)
ck("CLI reconcile reports ok:true as JSON",
   json.loads(_buf4.getvalue())["ok"] is True)

shutil.rmtree(_tmp, ignore_errors=True)


# ---- one sacct parser, not two (v3.29.0) -----------------------------------
# This module carried its own copy of the two sacct layouts until the shared
# parser was published. The v3.27.1 repair was caused by exactly that: two
# readers disagreeing about a stored sacct section, so every job-state check
# answered `unknown` on a bundle that held the TIMEOUT row all along. These
# checks fail if a second copy comes back.
_SRC = pathlib.Path(sl.__file__).read_text()
ck("su_ledger declares no ruler parser of its own", "_ruler_spans" not in _SRC)
ck("su_ledger declares no header finder of its own", "def _find_header" not in _SRC)
ck("su_ledger delegates to the shared parser", "parse_sacct_text" in _SRC)

_TEXT = ("$ sacct -j 44727703\n"
         "JobID|JobName|State|Elapsed|Timelimit\n"
         "44727703|rndtrain|TIMEOUT|12:00:18|12:00:00\n"
         "44727703.batch|batch|CANCELLED|12:00:19|\n")
_from_corpus = _corpus_mod.parse_sacct_text(_TEXT)[0]
_from_ledger = sl.parse_sacct(_TEXT)
ck("the shared parser reads both printed rows", len(_from_corpus) == 2)
ck("the ledger folds the substep into one job", len(_from_ledger) == 1)
ck("the ledger reads the parent state through the shared parser",
   _from_ledger[0]["state"] == "TIMEOUT")
ck("the ledger reads the parent elapsed through the shared parser",
   _from_ledger[0]["elapsed_raw"] == "12:00:18")
ck("the command echo is not parsed as a row",
   all("sacct -j" not in str(r.get("raw", "")) for r in _from_corpus))


if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
