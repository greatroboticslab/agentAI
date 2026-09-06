#!/usr/bin/env python3
"""WP1 scheduler state tests — restart drill, review gate, supersede, heartbeat.

Everything here runs against doubles: an in-memory ledger that implements the
v3.25.0 round-doc contract (attempts + ownership), a fake SLURM whose job states
and command output the test sets, and a temporary config file. No Mongo, no
cluster, no network, no sleeping.

What each case protects, in the words of the failure it comes from:
  * restart drill — a dashboard restart used to rebuild the scheduler's state
    with fails=0, rounds_today=1 and started=now, so the stop-loss history was
    erased, the per-day cap was misreported and the step-timeout deadline moved
    forward by the downtime. Counters now come back from the config file and the
    step's start time from `sacct`.
  * review gate — on 2026-08-29 a train step that hit its 12 h walltime was
    resubmitted, unchanged, on the next tick; the second job burnt another 12 h
    and the pair paused the domain. A failed step now waits for a verdict.
  * review timeout — with nobody to answer, the wait must still end in a
    deterministic correction (halve the epochs, keep the time cap) rather than in
    a stalled loop.
  * supersede — a step cancelled on purpose because its parameters were
    corrected is not evidence that the domain is broken, so it must not consume a
    stop-loss life, and the step must run again rather than count as done.

Run:  python -m pytest tests/test_scheduler_state.py  (or) python tests/test_scheduler_state.py
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

    def has(self, text, level=None):
        return any(text in m and (level is None or lv == level)
                   for lv, m in self.lines)


class FakeLedger:
    """The round doc the scheduler writes to, with db.py's real merge semantics."""

    ROUND_STEPS = list(db.ROUND_STEPS)

    def __init__(self, doc=None, config=None):
        self.doc = doc or {"domain": "weed", "round_num": 1, "steps": {},
                           "metrics": {}}
        self.config = config or copy.deepcopy(db.DEFAULT_DOMAIN_CONFIG)
        self.writes = []
        self.config_writes = []

    # -- domain config
    def get_domain_config(self, domain):
        return copy.deepcopy(self.config)

    def set_domain_config(self, domain, patch, actor="user"):
        self.config = db._deep_merge(self.config, patch)
        self.config_writes.append((domain, patch, actor))
        return copy.deepcopy(self.config)

    render_step_command = staticmethod(db.render_step_command)
    # The stub must expose whatever the scheduler asks a database for. The
    # policy gate authorises the parameters a step renders, and a stub without
    # this helper makes the gate fall back to the whole round dict and refuse
    # every step -- a stale stub reading as a policy failure.
    step_fields = staticmethod(db.step_fields)
    validate_step_command = staticmethod(db.validate_step_command)

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
        if step not in self.ROUND_STEPS:
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


class FakeCluster:
    """squeue / sacct / scancel / one-shot shell, all scripted by the test."""

    def __init__(self):
        self.states = {}          # jobid -> squeue/sacct state
        self.starts = {}          # jobid -> sacct Start string
        self.sh_out = {}          # substring of the command -> stdout
        self.argv = []
        self.submitted = []
        self.cancelled = []
        self.next_job = 5000

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


_TMP = tempfile.mkdtemp(prefix="rs_state_")


def setup(state=None, steps=None, enabled=True, doc_round=4):
    """Fresh config file, fresh ledger, fresh cluster, empty in-process state."""
    rs._STATE.clear()
    rs._FALLBACK_LOGGED.clear()
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


# -------------------------------------------------- 1. restart drill -------
print("\n[1] restart drill: counters and the step's real start time survive")
started_at = time.time() - 3600
log, ledger, cluster, dcfg, _acts = setup(
    state={"job": "44727703", "step": "train", "fails": 1, "day": TODAY,
           "rounds_today": 2, "started_sacct": 0},
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44727703")})
cluster.starts["44727703"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(started_at))
rs._recover_inflight()
st = rs._STATE["weed"]
ck("stop-loss counter survives the restart", st["fails"] == 1)
ck("rounds_today survives the restart", st["rounds_today"] == 2)
ck("day survives the restart", st["day"] == TODAY)
ck("in-flight job re-adopted from the ledger",
   st["job"] == "44727703" and st["step"] == "train")
ck("step start comes from sacct, not from the restart moment",
   abs(st["started"] - started_at) < 90)
ck("recovered state is written back to the config file",
   (json.loads(pathlib.Path(rs._CFG_FILE).read_text())["domains"]["weed"]["state"]
    ["fails"]) == 1)

# an unreadable persisted state must not take the scheduler down
log2, ledger2, cluster2, dcfg2, _a2 = setup(
    state={"fails": "not-a-number", "rounds_today": 2})
rs._recover_inflight()
ck("a corrupt persisted counter falls back to zero, with a warning",
   rs._STATE["weed"]["fails"] == 0 and log2.has("unreadable persisted state"))


# ------------------------------------------- 2. failed train -> review -----
print("\n[2] a failed train is held for review, not resubmitted")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44727703")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44727703", step="train", started=time.time() - 3600)
cluster.states["44727703"] = "TIMEOUT"
rs._advance("weed", dcfg)

entry = ledger.step("train")
ck("the TIMEOUT is recorded as a failed step", entry.get("status") == "failed")
ck("the failure names the job state", "TIMEOUT" in str(entry.get("detail")))
ck("the step is put in review", (entry.get("review") or {}).get("status") == "awaiting")
ck("the review is timestamped", float((entry.get("review") or {}).get("queued_at", 0)) > 0)
ck("one failure counts toward the stop-loss", rs._STATE["weed"]["fails"] == 1)
ck("one failure does not pause the domain", dcfg.get("enabled") is True)
ck("the previous head survives as an attempt",
   any(a.get("status") == "running" for a in entry.get("attempts") or []))

rs._advance("weed", dcfg)          # the tick that used to resubmit blind
ck("no resubmission while the review is awaiting", cluster.submitted == [])
ck("the hold is logged", log.has("held for review"))
ck("the in-flight state stays clear", rs._STATE["weed"]["job"] is None)


# --------------------------------------- 3. review timeout -> fallback -----
print("\n[3] the review times out and the deterministic correction is applied")
old_review = dict(ledger.step("train")["review"])
old_review["queued_at"] = time.time() - (91 * 60)
ledger.doc["steps"]["train"]["review"] = old_review
before = ledger.get_domain_config("weed")["round_params"]["epochs"]
rs._advance("weed", dcfg)

after = ledger.get_domain_config("weed")["round_params"]["epochs"]
ck("epochs are halved by the fallback (%s -> %s)" % (before, after),
   before == 60 and after == 30)
ck("the time cap is left alone",
   ledger.get_domain_config("weed")["round_params"]["train_time_cap_h"] == 10.8)
ck("the correction is recorded on the ledger with the timeout named",
   any("review timed out" in str(w.get("detail")) for w in ledger.writes))
ck("the correction is written by the scheduler, not attributed elsewhere",
   any(w["actor"] == "round-scheduler" for w in ledger.writes
       if "review timed out" in str(w.get("detail"))))
ck("the step is resubmitted after the correction", len(cluster.submitted) == 1)
ck("the resubmitted line carries the corrected epochs",
   "TRAIN_EPOCHS=30" in (cluster.submitted[0] if cluster.submitted else ""))
ck("the resubmitted line keeps the time cap",
   "TRAIN_TIME_H=10.8" in (cluster.submitted[0] if cluster.submitted else ""))
ck("the submission goes through the action log",
   any(a == "rounds_train" for a, _r in acts))
ck("the ledger records what the job was rendered with",
   (ledger.step("train").get("params") or {}).get("epochs") == 30)

n_before = len(cluster.submitted)
rs._STATE["weed"].update(job=None, step=None)
ledger.doc["steps"]["train"] = done("failed", job="44727703",
                                    detail="job state TIMEOUT")
ledger.doc["steps"]["train"]["review"] = old_review
rs._advance("weed", dcfg)
ck("a second timeout pass does not halve the epochs again",
   ledger.get_domain_config("weed")["round_params"]["epochs"] == 30)
ck("the step is still allowed to run", len(cluster.submitted) == n_before + 1)


# a review entry with no queue time gets the full wait, not an instant timeout
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": dict(done("failed", job="44767709", detail="job state TIMEOUT"),
                         review={"status": "awaiting", "review_id": "r1"})})
rs._STATE["weed"] = rs._blank_state()
before_epochs = ledger.get_domain_config("weed")["round_params"]["epochs"]
rs._advance("weed", dcfg)
ck("a review with no queued_at holds the step instead of expiring at once",
   cluster.submitted == [] and log.has("held for review"))
ck("a review with no queued_at does not trigger the correction",
   ledger.get_domain_config("weed")["round_params"]["epochs"] == before_epochs)


print("\n[3b] two reviewed failures still pause the domain, on disk")
# the stop-loss still fires, and now says so on disk
log, ledger, cluster, dcfg, acts = setup(
    state={"fails": 1, "day": TODAY, "rounds_today": 1},
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44767709")})
rs._recover_inflight()
rs._STATE["weed"].update(job="44767709", step="train", started=time.time() - 3600)
cluster.states["44767709"] = "TIMEOUT"
rs._advance("weed", dcfg)
saved = json.loads(pathlib.Path(rs._CFG_FILE).read_text())["domains"]["weed"]
ck("a second consecutive failure pauses the domain", saved.get("enabled") is False)
ck("the pause records why", "stop-loss" in str(saved.get("paused_reason")))
ck("the pause is logged as an error", log.has("PAUSED by stop-loss", level="error"))
ck("the failure count is on disk with the pause",
   (saved.get("state") or {}).get("fails") == 2)


# ------------------------------------------------- 4. supersede ------------
print("\n[4] a superseded step is terminal but not a failure")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44767709")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44767709", step="train", started=time.time() - 600,
                         fails=1)
res = rs.supersede_step("weed", "train", reason="epochs corrected to 30",
                        actor="tier1:glm-4.7-flash")

ck("supersede reports ok", res.get("ok") is True and res.get("superseded") is True)
ck("the job is cancelled", cluster.cancelled == ["44767709"])
ck("the outcome is recorded with a status the ledger accepts",
   ledger.step("train").get("status") in db._ROUND_STATUSES)
ck("the outcome names the supersede",
   str(ledger.step("train").get("detail")).startswith(rs.SUPERSEDE_PREFIX))
ck("the reason is kept", "epochs corrected" in str(ledger.step("train").get("detail")))
ck("the actor is the one that superseded it",
   ledger.step("train").get("actor") == "tier1:glm-4.7-flash")
ck("supersede does not increment fails", rs._STATE["weed"]["fails"] == 1)
ck("supersede does not pause the domain", dcfg.get("enabled") is True)
ck("supersede clears the in-flight step", rs._STATE["weed"]["job"] is None)
ck("supersede is mirrored into the action log",
   any(a == "rounds_supersede" for a, _r in acts))

ck("a superseded step does not count as done",
   rs._is_superseded(ledger.step("train")) is True)
rs._advance("weed", dcfg)
ck("the next tick re-runs the superseded step", len(cluster.submitted) == 1)
ck("the re-run is a train submission",
   "run_m1_merged_seeds.sh" in (cluster.submitted[0] if cluster.submitted else ""))
ck("a plain skipped step still counts as done",
   rs._is_superseded(done("skipped", detail="autolabel-in-collect")) is False)


# ----------------------------------- 5. in-flight walltime projection ------
print("\n[5] a running train is projected against its own walltime")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44727703")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44727703", step="train", started=time.time() - 3600)
cluster.states["44727703"] = "RUNNING"
cluster.sh_out["_brain"] = 'TRACE %s' % json.dumps(
    {"kind": "epoch", "epoch": 12, "elapsed_s": 6000.0, "eta_total_s": 104400.0,
     "walltime_s": 43200.0, "epoch_records": 12})
rs._advance("weed", dcfg)

ck("the projection is logged as a warning", log.has("walltime-bound projection",
                                                    level="warning"))
ck("the projection is recorded on the running step",
   "walltime-bound projection" in str(ledger.step("train").get("detail")))
ck("the step stays in flight (WP1 observes, it does not cancel)",
   rs._STATE["weed"]["job"] == "44727703" and cluster.cancelled == [])
ck("the projection does not count as a failure", rs._STATE["weed"]["fails"] == 0)

n_writes = len(ledger.writes)
rs._advance("weed", dcfg)
ck("the projection is written to the ledger once per job, not once per tick",
   len(ledger.writes) == n_writes)
ck("but it is still logged every tick",
   sum(1 for lv, m in log.lines if "walltime-bound projection" in m) == 2)

# a healthy run must stay silent
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44900001")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44900001", step="train", started=time.time() - 3600)
cluster.states["44900001"] = "RUNNING"
cluster.sh_out["_brain"] = 'TRACE %s' % json.dumps(
    {"kind": "epoch", "epoch": 12, "elapsed_s": 6000.0, "eta_total_s": 30000.0,
     "walltime_s": 43200.0, "epoch_records": 12})
rs._advance("weed", dcfg)
ck("a run that fits its walltime raises nothing",
   not log.has("walltime-bound projection"))

# two epochs are not a projection
rs._STATE["weed"]["walltime_warned"] = False
cluster.sh_out["_brain"] = 'TRACE %s' % json.dumps(
    {"kind": "epoch", "epoch": 2, "elapsed_s": 6000.0, "eta_total_s": 104400.0,
     "walltime_s": 43200.0, "epoch_records": 2})
rs._advance("weed", dcfg)
ck("fewer than three epoch records is not enough to project",
   not log.has("walltime-bound projection"))


# ------------------------------------------- 6. job-scoped train metric ----
print("\n[6] the train metric comes from this job's own artifact")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44727703")})
rs._STATE["weed"] = rs._blank_state()
started = time.time() - 7200
rs._STATE["weed"].update(job="44727703", step="train", started=started)
cluster.states["44727703"] = "COMPLETED"
cluster.sh_out["m1_"] = 'TRAINMETRIC %s' % json.dumps(
    {"artifact": "results/framework/m1_curated_seed101_44727703.json",
     "job_id": "44727703",
     "csv": "results/framework/mega_iterrnd4_train/rndtrain_44727703/results.csv",
     "mtime": time.time() - 60, "best": 0.60193, "rows": 60})
rs._advance("weed", dcfg)
ck("the metric lands on the round", ledger.doc["metrics"].get("map50_95") == 0.6019)
ck("the metric records which job produced it",
   ledger.doc["metrics"].get("job") == "44727703")
ck("the metric's source is that job's results.csv",
   "44727703" in str(ledger.doc["metrics"].get("source")))
ck("eval is attached to the same run", ledger.step("eval").get("status") == "done")

# a foreign run finishing during the round must not be attached
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44727703")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44727703", step="train", started=started)
cluster.states["44727703"] = "COMPLETED"
cluster.sh_out["m1_"] = 'TRAINMETRIC %s' % json.dumps(
    {"artifact": "results/framework/m1_curated_seed101_44727703.json",
     "job_id": "44900999", "csv": "x/results.csv", "mtime": time.time(),
     "best": 0.88, "rows": 60})
rs._advance("weed", dcfg)
ck("an artifact belonging to another job is refused",
   ledger.doc["metrics"].get("map50_95") is None and log.has("foreign run"))

# an artifact older than the step is stale, not this round's result
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done(), "filter": done(), "label": done("skipped"),
           "train": done("running", job="44727703")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44727703", step="train", started=started)
cluster.states["44727703"] = "COMPLETED"
cluster.sh_out["m1_"] = 'TRAINMETRIC %s' % json.dumps(
    {"artifact": "a.json", "job_id": "44727703", "csv": "x/results.csv",
     "mtime": started - 86400, "best": 0.88, "rows": 60})
rs._advance("weed", dcfg)
ck("a results.csv older than the step is refused",
   ledger.doc["metrics"].get("map50_95") is None and log.has("stale metric"))


# --------------------------------------------------- 7. heartbeat ----------
print("\n[7] every tick leaves proof the scheduler ran")
log, ledger, cluster, dcfg, acts = setup(
    steps={"collect": done("running", job="44322382")})
rs._STATE["weed"] = rs._blank_state()
rs._STATE["weed"].update(job="44322382", step="collect", fails=1, rounds_today=2,
                         review={"status": "awaiting", "step": "train"})
cfg = rs._cfg()
cfg["domains"]["weed"]["paused_reason"] = "stop-loss: 2 consecutive step failures"
rs._heartbeat(cfg, 12.5)

hb_path = pathlib.Path(_TMP) / "results" / "framework" / "scheduler_status.json"
ck("the heartbeat file is written where the health check reads it", hb_path.is_file())
hb = json.loads(hb_path.read_text()) if hb_path.is_file() else {}
ck("heartbeat carries ts / tick_s / tick_duration_s",
   isinstance(hb.get("ts"), float) and hb.get("tick_s") == rs.TICK_S
   and hb.get("tick_duration_s") == 12.5)
d = (hb.get("domains") or {}).get("weed") or {}
for key in ("enabled", "paused_reason", "job", "step", "fails", "review",
            "rounds_today"):
    ck("heartbeat domain block carries %s" % key, key in d)
ck("a paused domain is visible in the heartbeat",
   d.get("paused_reason").startswith("stop-loss"))
ck("the in-flight job is visible in the heartbeat",
   d.get("job") == "44322382" and d.get("step") == "collect")
ck("an awaiting review is visible in the heartbeat",
   (d.get("review") or {}).get("status") == "awaiting")
ck("the heartbeat leaves no temporary file behind",
   not (hb_path.parent / (hb_path.name + ".tmp")).exists())


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
