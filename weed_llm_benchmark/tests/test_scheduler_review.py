#!/usr/bin/env python3
"""WP1 review-state-machine tests — the whole path from a submit to a pause.

One fake context for the whole story: an in-memory ledger with db.py's real
merge/ownership semantics, a fake SLURM whose job states and command output the
test sets, and a temporary config file. No Mongo, no cluster, no network, no
sleeping.

What the story protects, in the words of the failure it comes from:
  * On 2026-08-29 two 12 h train jobs hit their walltime at epoch 24 and 16 of
    60. The second was submitted, unchanged, on the tick after the first was
    recorded failed; the pair then tripped the stop-loss and the domain sat
    paused for six days with nothing saying so. v3.25.0 holds a failed step for
    review, corrects it deterministically when no verdict arrives, and pauses
    loudly.
  * The correction is a permanent config write, so it is conditioned on the
    failure's shape and recorded: halving the epochs of a step that died of
    something other than its walltime corrects nothing, and an unrecorded write
    moves the campaign off the recipe the noise floors were measured on.
  * The guards that make this safe must not switch themselves off: the
    correction must be applied once per review across a restart, a supersede
    must not run while a tick holds the lock, an UNKNOWN job must still hit the
    step deadline, a projection that cannot run must say so, and a heartbeat
    written while every ledger write is lost must not read as green.

Run:  python -m pytest tests/test_scheduler_review.py
 (or) python tests/test_scheduler_review.py
"""
import copy
import json
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db  # noqa: E402
from weed_optimizer_framework.tools import round_scheduler as rs  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


TODAY = time.strftime("%Y-%m-%d")
REVIEW_TIMED_OUT_S = (rs.REVIEW_TIMEOUT_MIN + 1) * 60


# ---------------------------------------------------------------- doubles --
class Log:
    def __init__(self):
        self.lines = []

    def _add(self, level, msg, *a):
        self.lines.append((level, (msg % a) if a else str(msg)))

    def info(self, m, *a):
        self._add("info", m, *a)

    def warning(self, m, *a):
        self._add("warning", m, *a)

    def error(self, m, *a):
        self._add("error", m, *a)

    def count(self, text, level=None):
        return sum(1 for lv, m in self.lines
                   if text in m and (level is None or lv == level))


class FakeLedger:
    """The round doc the scheduler writes to, with db.py's real merge semantics."""

    ROUND_STEPS = list(db.ROUND_STEPS)

    def __init__(self, doc=None, config=None):
        self.doc = doc or {"domain": "weed", "round_num": 1, "steps": {},
                           "metrics": {}}
        self.config = config or copy.deepcopy(db.DEFAULT_DOMAIN_CONFIG)
        self.writes = []
        self.config_writes = []
        self.down = False            # Mongo unreachable: writes return None

    # -- domain config
    def get_domain_config(self, domain):
        return copy.deepcopy(self.config)

    def set_domain_config(self, domain, patch, actor="user"):
        if self.down:
            return None
        self.config = db._deep_merge(self.config, patch)
        self.config_writes.append((domain, copy.deepcopy(patch), actor))
        return copy.deepcopy(self.config)

    render_step_command = staticmethod(db.render_step_command)

    # -- rounds
    def get_current_round(self, domain):
        return copy.deepcopy(self.doc)

    def start_round(self, domain, actor="user"):
        self.doc = {"domain": domain, "round_num": self.doc["round_num"] + 1,
                    "steps": {}, "metrics": {}}
        return copy.deepcopy(self.doc)

    def record_round_step(self, domain, step, status, detail=None, job=None,
                          actor="user", round_num=None, metrics=None,
                          params=None, decided_by=None, review=None, su=None):
        if step not in self.ROUND_STEPS or self.down:
            return None
        entry = db._round_step_entry(status, detail, job, actor, "T",
                                     params=params, decided_by=decided_by,
                                     review=review, su=su)
        head = self.doc["steps"].get(step)
        self.doc["steps"][step] = db.merge_step_entry(head, entry, actor)
        for k, v in (metrics or {}).items():
            self.doc["metrics"][k] = v
        self.writes.append({"step": step, "status": status, "actor": actor,
                            "detail": detail, "job": job, "review": review,
                            "params": params, "metrics": metrics})
        return copy.deepcopy(self.doc)

    def step(self, name):
        return self.doc["steps"].get(name) or {}

    def corrections(self):
        c = self.config.get("round_params_corrections")
        return c if isinstance(c, list) else []

    def epochs(self):
        return (self.config.get("round_params") or {}).get("epochs")


class FakeCluster:
    """squeue / sacct / scancel / one-shot shell, all scripted by the test."""

    def __init__(self):
        self.states = {}          # jobid -> squeue/sacct state
        self.starts = {}          # jobid -> sacct Start string
        self.sh_out = {}          # substring of the command -> stdout
        self.argv = []
        self.submitted = []
        self.cancelled = []
        self.next_job = 7000

    def slurm(self, argv, timeout=40):
        self.argv.append(list(argv))
        jid = argv[2] if len(argv) > 2 else ""
        if argv[0] == "squeue":
            return {"ok": True, "stdout": self.states.get(jid, ""), "stderr": ""}
        if argv[0] == "sacct":
            if argv[-1] == "Start":
                return {"ok": True, "stdout": self.starts.get(jid, ""), "stderr": ""}
            return {"ok": True, "stdout": self.states.get(jid, ""), "stderr": ""}
        if argv[0] == "scancel":
            self.cancelled.append(argv[1])
            return {"ok": True, "stdout": "", "stderr": ""}
        return {"ok": True, "stdout": "", "stderr": ""}

    def sh(self, cmd, timeout=60):
        if cmd.strip().startswith("sbatch"):
            self.next_job += 1
            self.submitted.append(cmd)
            return {"ok": True, "stdout": "Submitted batch job %d" % self.next_job,
                    "stderr": ""}
        for key, out in self.sh_out.items():
            if key in cmd:
                return {"ok": True, "stdout": out, "stderr": ""}
        return {"ok": True, "stdout": "", "stderr": ""}


_TMP = tempfile.mkdtemp(prefix="rs_review_")


def setup(steps=None, enabled=True, doc_round=4, state=None):
    """Fresh config file, fresh ledger, fresh cluster, empty in-process state."""
    rs._STATE.clear()
    rs._FALLBACK_LOGGED.clear()
    rs._LEDGER.update(ok=True, last_error_ts=None)
    log, ledger, cluster = Log(), FakeLedger(), FakeCluster()
    ledger.doc = {"domain": "weed", "round_num": doc_round,
                  "steps": copy.deepcopy(steps or {}), "metrics": {}}
    cfgfile = tempfile.mktemp(dir=_TMP, suffix=".json")
    dcfg = {"enabled": enabled, "max_rounds_per_day": 2}
    if state:
        dcfg["state"] = dict(state)
    pathlib.Path(cfgfile).write_text(json.dumps({"domains": {"weed": dcfg}}))
    rs._CFG_FILE = cfgfile
    actions = []
    rs._CTX.clear()
    rs._CTX.update({"log": log, "db": ledger, "record_step": ledger.record_round_step,
                    "slurm": cluster.slurm, "slurm_sh": cluster.sh,
                    "repo": _TMP, "log_action": lambda a, r: actions.append((a, r))})
    return log, ledger, cluster, dcfg, actions


def done(status="done", job=None, detail=None):
    e = {"status": status, "actor": "round-scheduler", "at": "T"}
    if job:
        e["job"] = job
    if detail is not None:
        e["detail"] = detail
    return e


def S():
    """The live state dict — re-read after every simulated restart."""
    return rs._STATE["weed"]


def on_disk():
    return json.loads(pathlib.Path(rs._CFG_FILE).read_text())["domains"]["weed"]


def no_sleep(fn, *a, **kw):
    """Run one tick with the sacct-lag backoff stubbed out (not part of any pin)."""
    real = time.sleep
    time.sleep = lambda _s: None
    try:
        return fn(*a, **kw)
    finally:
        time.sleep = real


# ===========================================================================
# One continuous story: submit -> TIMEOUT -> review -> correction -> pause
# ===========================================================================
print("\n[1] the train step is submitted")
# collect/filter/label are already behind this round: this file pins the review
# machine, which only the train step reaches.
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped")})
rs._advance("weed", dcfg)
job1 = S()["job"]

ck("a train job is submitted", job1 is not None and S()["step"] == "train")
ck("the ledger head is running with that job",
   ledger.step("train").get("status") == "running"
   and ledger.step("train").get("job") == job1)
ck("the submission counts against the resubmit cap",
   (S().get("submits") or {}).get("4:train") == 1)
ck("the round starts with the sealed recipe", ledger.epochs() == 60)


print("\n[2] the job ends TIMEOUT: failed, held for review, not resubmitted")
cluster.states[job1] = "TIMEOUT"
rs._advance("weed", dcfg)
entry = ledger.step("train")
review_id = (entry.get("review") or {}).get("review_id")

ck("the TIMEOUT is recorded as a failed step", entry.get("status") == "failed")
ck("the failure names the job state", "TIMEOUT" in str(entry.get("detail")))
ck("the step is put in review", (entry.get("review") or {}).get("status") == "awaiting")
ck("the review is timestamped",
   float((entry.get("review") or {}).get("queued_at") or 0) > 0)
ck("one failure counts toward the stop-loss", S()["fails"] == 1)
ck("one failure does not pause the domain", dcfg.get("enabled") is True)
ck("how the job ended is remembered for the correction",
   (S().get("last_terminal") or {}).get("train", {}).get("state") == "TIMEOUT")
ck("the in-flight state is cleared", S()["job"] is None and S()["step"] is None)

n_sub = len(cluster.submitted)
rs._advance("weed", dcfg)
ck("the next tick does not resubmit the identical job",
   len(cluster.submitted) == n_sub)
ck("the hold is visible on the state",
   (S().get("review") or {}).get("status") == "awaiting")
ck("nothing was corrected while the review is still open", ledger.epochs() == 60)


print("\n[3] the review times out: the correction is applied and recorded")
rev = dict(ledger.step("train")["review"])
rev["queued_at"] = time.time() - REVIEW_TIMED_OUT_S
ledger.doc["steps"]["train"]["review"] = rev
rs._advance("weed", dcfg)
job2 = S()["job"]
corr = ledger.corrections()

ck("the epochs are halved", ledger.epochs() == 30)
ck("the time cap is left alone",
   (ledger.config.get("round_params") or {}).get("train_time_cap_h") == 10.8)
ck("exactly one correction record is written", len(corr) == 1)
c0 = corr[0] if corr else {}
for key in ("at", "by", "key", "old", "new", "reason", "from_round", "review_id"):
    ck("the correction record carries %s" % key, key in c0)
ck("the correction record keeps the value that was replaced", c0.get("old") == 60)
ck("the correction record names the value written", c0.get("new") == 30)
ck("the correction record names the actor", c0.get("by") == "round-scheduler")
ck("the correction record names the round it came from", c0.get("from_round") == 4)
ck("the correction record ties back to the review", c0.get("review_id") == review_id)
ck("the correction record is a timestamped string", isinstance(c0.get("at"), str)
   and c0.get("at", "").startswith(time.strftime("%Y")))
ck("the correction is written in the same config write as the value",
   any("round_params" in p and "round_params_corrections" in p
       for _d, p, _a in ledger.config_writes))
ck("the ledger detail names the previous value so it can be reverted",
   "previous value 60" in str(ledger.step("train").get("detail") or "")
   or any("previous value 60" in str(w.get("detail") or "") for w in ledger.writes))
ck("the corrected step is resubmitted", job2 is not None and job2 != job1)
ck("the resubmitted line carries the corrected epochs",
   "TRAIN_EPOCHS=30" in cluster.submitted[-1])
ck("the resubmitted line keeps the time cap",
   "TRAIN_TIME_H=10.8" in cluster.submitted[-1])


print("\n[4] the correction survives a restart and is not applied twice")
saved = on_disk().get("state") or {}
ck("the resubmit cap is persisted", (saved.get("submits") or {}).get("4:train") == 2)
ck("the applied-review memo is persisted", bool(saved.get("review_applied")))

# Dashboard restart: the in-process state is rebuilt from the config file only.
memo_before = S()["review_applied"]
rs._STATE.clear()
dcfg = (rs._cfg().get("domains") or {})["weed"]
rs._state("weed", dcfg)
ck("the resubmit cap comes back after the restart",
   (S().get("submits") or {}).get("4:train") == 2)
ck("the applied-review memo comes back after the restart",
   S().get("review_applied") == memo_before)
ck("what the failure was comes back after the restart",
   (S().get("last_terminal") or {}).get("train", {}).get("state") == "TIMEOUT")

# The head can still read `awaiting` on the next tick (a tier-1 or human entry
# is not overwritten by the scheduler, so the `applied` write lands in attempts).
# Without the memo the epochs would be halved again, every tick.
ledger.doc["steps"]["train"] = done("failed", job=job1, detail="job state TIMEOUT")
ledger.doc["steps"]["train"]["review"] = rev
S().update(job=None, step=None)
n_sub = len(cluster.submitted)
rs._advance("weed", dcfg)
job3 = S()["job"]

ck("a second pass over the same review does not halve the epochs again",
   ledger.epochs() == 30)
ck("no second correction record is written", len(ledger.corrections()) == 1)
ck("the step is still allowed to run", len(cluster.submitted) == n_sub + 1)
ck("the re-run carries the corrected epochs, not a second correction",
   "TRAIN_EPOCHS=30" in cluster.submitted[-1])


print("\n[5] a non-walltime failure is resubmitted unchanged")
# The corrected run finishes, so the stop-loss counter is clean for the next round.
cluster.states[job3] = "COMPLETED"
cluster.sh_out["m1_"] = "TRAINMETRIC " + json.dumps(
    {"artifact": "results/framework/m1_curated_seed101_%s.json" % job3,
     "job_id": job3, "csv": "results/framework/mega_iterrnd4_train/results.csv",
     "mtime": time.time(), "best": 0.5912, "rows": 30})
rs._advance("weed", dcfg)
ck("a completed run clears the stop-loss counter", S()["fails"] == 0)
ck("the metric of the corrected run lands on the round",
   ledger.doc["metrics"].get("map50_95") == 0.5912)

rs._advance("weed", dcfg)                       # every step done -> a new round
ck("a new round is opened", ledger.doc.get("round_num") == 5)
ledger.doc["steps"] = {"collect": done(), "filter": done(), "label": done("skipped")}
rs._advance("weed", dcfg)                       # round 5's train goes out
job4 = S()["job"]
ck("round 5 submits its train step", job4 is not None and S()["step"] == "train")

cluster.states[job4] = "FAILED"                 # an OOM / a bad merge, not the wall
rs._advance("weed", dcfg)
ck("the non-walltime failure is recorded failed",
   ledger.step("train").get("status") == "failed")
ck("it is held for review like any other failure",
   (ledger.step("train").get("review") or {}).get("status") == "awaiting")
ck("its shape is recorded as it happened",
   (S().get("last_terminal") or {}).get("train", {}).get("state") == "FAILED")
ck("no walltime warning is claimed for it",
   (S().get("last_terminal") or {}).get("train", {}).get("walltime_warned") is False)
ck("it counts toward the stop-loss", S()["fails"] == 1)

rev5 = dict(ledger.step("train")["review"])
# A distinct queue second: the applied-review memo is keyed on the step and the
# review's queue time, and this whole story runs inside one second of real time.
rev5["queued_at"] = time.time() - (REVIEW_TIMED_OUT_S + 137)
ledger.doc["steps"]["train"]["review"] = rev5
n_sub = len(cluster.submitted)
rs._advance("weed", dcfg)
job5 = S()["job"]

ck("a non-walltime failure does not halve the epochs", ledger.epochs() == 30)
ck("no correction record is written for it", len(ledger.corrections()) == 1)
ck("the ledger says the step was resubmitted unchanged",
   any("not walltime shaped" in str(w.get("detail") or "")
       and "resubmitting unchanged" in str(w.get("detail") or "")
       for w in ledger.writes))
ck("the step is resubmitted", job5 is not None and len(cluster.submitted) == n_sub + 1)
ck("the resubmission is the same recipe", "TRAIN_EPOCHS=30" in cluster.submitted[-1])


print("\n[6] two consecutive failures pause the domain, on disk, with a reason")
cluster.states[job5] = "FAILED"
rs._advance("weed", dcfg)
saved = on_disk()

ck("the second consecutive failure counts", S()["fails"] == 2)
ck("the domain is disabled in the live config", dcfg.get("enabled") is False)
ck("the domain is disabled on disk", saved.get("enabled") is False)
ck("the pause records why", "stop-loss" in str(saved.get("paused_reason")))
ck("the failure count is on disk with the pause",
   (saved.get("state") or {}).get("fails") == 2)


print("\n[7] a supersede is refused while a tick holds the lock")


class _HeldLock:
    """Stands in for the scheduler lock held by a slow tick."""

    def __init__(self):
        self.released = 0

    def acquire(self, timeout=None):
        return False

    def release(self):
        self.released += 1


held, real_lock = _HeldLock(), rs._LOCK
rs._LOCK = held
n_writes, n_cancel = len(ledger.writes), len(cluster.cancelled)
busy = rs.supersede_step("weed", "train", reason="recipe corrected by hand",
                         actor="human:agronomist@example.org")
rs._LOCK = real_lock

ck("a supersede that cannot take the lock reports busy",
   busy.get("ok") is False and busy.get("busy") is True)
ck("it touches no ledger state", len(ledger.writes) == n_writes)
ck("it cancels no job", len(cluster.cancelled) == n_cancel)
ck("it releases no lock it never took", held.released == 0)
ck("the busy answer is mirrored into the action log",
   any(a == "rounds_supersede" and r.get("busy") for a, r in acts))


print("\n[8] a superseded step consumes no stop-loss life and is re-run")
# A person clears the pause (as the admin route does) and supersedes the failed
# step because the recipe was corrected by hand; one failure stays on the count.
dcfg["enabled"] = True
dcfg.pop("paused_reason", None)
S()["fails"] = 1
fails_before = S()["fails"]
res = rs.supersede_step("weed", "train", reason="recipe corrected by hand",
                        actor="human:agronomist@example.org")

ck("supersede reports ok", res.get("ok") is True and res.get("superseded") is True)
ck("the job is cancelled", job5 in cluster.cancelled)
ck("the outcome is a status the ledger accepts",
   ledger.step("train").get("status") in db._ROUND_STATUSES)
ck("the outcome is marked superseded", rs._is_superseded(ledger.step("train")) is True)
ck("the reason is kept", "corrected by hand" in str(ledger.step("train").get("detail")))
ck("the actor is the one that superseded it",
   ledger.step("train").get("actor") == "human:agronomist@example.org")
ck("supersede does not increment fails", S()["fails"] == fails_before)
ck("supersede does not pause the domain", dcfg.get("enabled") is True)

n_sub = len(cluster.submitted)
rs._advance("weed", dcfg)
ck("the superseded step is re-run", len(cluster.submitted) == n_sub + 1)
ck("the re-run is a train submission",
   "run_m1_merged_seeds.sh" in cluster.submitted[-1])
ck("the re-run does not count as a failure", S()["fails"] == fails_before)
ck("the domain is still enabled after the re-run", dcfg.get("enabled") is True)


# ===========================================================================
# Guards that must not switch themselves off silently
# ===========================================================================
print("\n[9] an UNKNOWN job is counted and still hits the step deadline")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="7501")})
rs._STATE["weed"] = rs._blank_state()
S().update(job="7501", step="train", started=time.time() - 3600)
# neither squeue nor sacct answers for this job: _job_state reports UNKNOWN
no_sleep(rs._advance, "weed", dcfg)
ck("an UNKNOWN job stays in flight", S()["job"] == "7501")
ck("the UNKNOWN tick is counted", S().get("unknown_ticks") == 1)
no_sleep(rs._advance, "weed", dcfg)
ck("consecutive UNKNOWN ticks accumulate", S().get("unknown_ticks") == 2)
ck("no verdict is invented for it", ledger.step("train").get("status") == "running")

S()["started"] = time.time() - (rs.STEP_TIMEOUT_H * 3600 + 60)
no_sleep(rs._advance, "weed", dcfg)
ck("an UNKNOWN job past the deadline is declared lost",
   ledger.step("train").get("status") == "failed")
ck("the lost step names the deadline",
   "step timeout" in str(ledger.step("train").get("detail")))
ck("the lost job is cancelled", cluster.cancelled == ["7501"])
ck("the lost step counts toward the stop-loss", S()["fails"] == 1)
ck("the in-flight state is cleared", S()["job"] is None)
ck("the UNKNOWN counter is reset once it is resolved", S().get("unknown_ticks") == 0)


print("\n[10] a walltime projection that cannot run says so, once per job")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="7601")})
rs._STATE["weed"] = rs._blank_state()
S().update(job="7601", step="train", started=time.time() - 3600)
cluster.states["7601"] = "RUNNING"
# a trace with epochs but no walltime: there is nothing to project against
cluster.sh_out["_brain"] = "TRACE " + json.dumps(
    {"kind": "epoch", "epoch": 6, "elapsed_s": 6000.0, "eta_total_s": 60000.0,
     "epoch_records": 6})
rs._advance("weed", dcfg)
ck("the disabled projection is recorded against the job it belongs to",
   S().get("walltime_off") == "7601")
ck("it is reported at warning level",
   log.count("walltime projection unavailable", level="warning") == 1)
ck("it does not fail the step", ledger.step("train").get("status") == "running"
   and S()["fails"] == 0)

rs._advance("weed", dcfg)
ck("it is reported once per job, not once per tick",
   log.count("walltime projection unavailable") == 1)

S().update(job="7602", walltime_off="7601")
cluster.states["7602"] = "RUNNING"
cluster.sh_out.clear()                      # no trace file at all for this job
rs._advance("weed", dcfg)
ck("a new job re-arms the report", log.count("walltime projection unavailable") == 2)

# a complete trace still projects, and is never reported as unavailable
S().update(job="7603", walltime_off="")
cluster.states["7603"] = "RUNNING"
cluster.sh_out["_brain"] = "TRACE " + json.dumps(
    {"kind": "epoch", "epoch": 12, "elapsed_s": 6000.0, "eta_total_s": 104400.0,
     "walltime_s": 43200.0, "epoch_records": 12})
rs._advance("weed", dcfg)
ck("a usable trace projects instead of reporting itself unavailable",
   S().get("walltime_off") == "" and S().get("walltime_warned") == "7603")
ck("the projection is recorded on the running step",
   "walltime-bound projection" in str(ledger.step("train").get("detail")))
ck("a job with a usable trace is not reported as unavailable",
   log.count("walltime projection unavailable") == 2)


print("\n[11] the heartbeat reports whether the ledger is reachable")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done("running", job="7701")})
hb_path = pathlib.Path(_TMP) / "results" / "framework" / "scheduler_status.json"

rs._record("weed", "collect", "running", job="7701")
rs._heartbeat(rs._cfg(), 1.0)
hb = json.loads(hb_path.read_text())
ck("a reachable ledger reads ok", hb.get("mongo_ok") is True)
ck("with no error timestamp", hb.get("mongo_last_error_ts") is None)

ledger.down = True                          # Mongo down: the write returns None
rs._record("weed", "collect", "done", job="7701")
rs._heartbeat(rs._cfg(), 1.0)
hb = json.loads(hb_path.read_text())
ck("a lost ledger write is visible in the heartbeat", hb.get("mongo_ok") is False)
ck("the outage is dated", isinstance(hb.get("mongo_last_error_ts"), float)
   and hb["mongo_last_error_ts"] > 0)
ck("the tick itself is still reported alive", float(hb.get("ts") or 0) > 0)

ledger.down = False
rs._record("weed", "collect", "done", job="7701")
rs._heartbeat(rs._cfg(), 1.0)
hb2 = json.loads(hb_path.read_text())
ck("a recovered ledger reads ok again", hb2.get("mongo_ok") is True)
ck("the last outage stays readable after recovery",
   hb2.get("mongo_last_error_ts") == hb.get("mongo_last_error_ts"))


print("\n[12] the correction stops at the configured floor")


def awaiting(job, queued_at, review_id="r-floor"):
    e = done("failed", job=job, detail="job state TIMEOUT")
    e["review"] = {"status": "awaiting", "review_id": review_id,
                   "queued_at": queued_at}
    return e


log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": awaiting("7801", time.time() - REVIEW_TIMED_OUT_S)})
ledger.config["round_params_floor"] = {"epochs": 40}
rs._STATE["weed"] = rs._blank_state()
S()["last_terminal"] = {"train": {"state": "TIMEOUT", "job": "7801",
                                  "walltime_warned": False}}
rs._advance("weed", dcfg)

ck("the correction stops at the floor instead of halving past it",
   ledger.epochs() == 40)
ck("the record shows what was replaced and by what",
   ledger.corrections()[-1].get("old") == 60
   and ledger.corrections()[-1].get("new") == 40)
ck("the refusal to go below the floor is logged",
   log.count("refusing to write below the floor", level="warning") == 1)
ck("the corrected step is submitted with the floor value",
   "TRAIN_EPOCHS=40" in cluster.submitted[-1])

# already at the floor: nothing is written and the step runs unchanged
S().update(job=None, step=None)
ledger.doc["steps"]["train"] = awaiting(
    "7802", time.time() - (REVIEW_TIMED_OUT_S + 211), review_id="r-floor-2")
S()["last_terminal"] = {"train": {"state": "TIMEOUT", "job": "7802",
                                  "walltime_warned": False}}
n_corr, n_sub = len(ledger.corrections()), len(cluster.submitted)
rs._advance("weed", dcfg)

ck("a value already at the floor is left alone", ledger.epochs() == 40)
ck("no correction record is written for a no-op", len(ledger.corrections()) == n_corr)
ck("the ledger says why nothing was corrected",
   any("already at or below the" in str(w.get("detail") or "")
       for w in ledger.writes))
ck("the step still runs", len(cluster.submitted) == n_sub + 1)


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
