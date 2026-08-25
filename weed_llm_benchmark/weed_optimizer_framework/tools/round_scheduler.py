"""Unattended round scheduler — the double-agent loop's continuous mode (S4).

For each ENABLED project, a background thread advances rounds through the ledger's
canonical steps (collect → filter → label → train → eval): it fires the same
cluster actions the project's agent cards fire, records every step on the rounds
ledger with its real job id, polls SLURM until the job terminates, attaches
metrics on success, and moves on. The project page's compounding chart reads the
same ledger, so a project left alone accumulates visible, honest rounds.

Guard rails (SUPERWEED_PLAN §1):
  * disabled by default, per-domain enable via admin API; config survives restarts
  * per-day round cap; stop-loss: 2 consecutive failed rounds pause the domain
  * every step is log-verified — a job that ends un-COMPLETED is recorded failed,
    never quietly skipped; metrics come from artifacts, not assumptions
  * one step in flight per domain, ever; scheduler ticks are cheap (one squeue)

v1 step wiring (weed-domain pipeline; other domains can map their own actions):
  collect  -> brain_harvest cluster action (harvest + autolabel inside the job)
  filter   -> dinov2 scoring refresh (run_s2_dino_scores.sh)
  label    -> recorded as "autolabel-in-collect" (no separate human step here;
              human labeling remains available through the Labeler card)
  train    -> one curated-tier seed of run_m1_merged_seeds.sh (sealed holdout)
  eval     -> holdout metric parsed from the train artifact (best epoch), so eval
              is attached to the same run rather than a second GPU job
"""
import json
import os
import re
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
_CTX = {}
_CFG_FILE = os.path.expanduser("~/.round_scheduler.json")
_LOCK = threading.Lock()
_STATE = {}          # domain -> {"job": id, "step": name, "round": n, "fails": k}

TICK_S = 120
STEP_TIMEOUT_H = 30          # a step's job may run this long before declared lost
DEFAULT_CFG = {"domains": {}}    # {"weed": {"enabled": False, "max_rounds_per_day": 2}}

# step -> (submit command argv ON the cluster repo root, jobname prefix)
_WEED_STEPS = {
    # v3.22.21: the script's own #SBATCH is 4 h, which was enough at 45 datasets
    # and is not at 61 — round 3's harvest ran 3 h 10 m and round 4 hit the wall.
    # Override on the command line so the script stays untouched for manual use.
    # v3.23.2: harvest is now capped at 3 new datasets per round, on evidence.
    # The corrected tier ladder shows added web-harvested data is worth ~0.00-0.02
    # against a clean core, while a 10 h harvest managed only three Roboflow
    # datasets (each zip is downloaded whole before being capped to 5,000 images)
    # and still timed out. Collecting more is no longer buying accuracy, so the
    # loop spends less on it rather than pretending the volume matters.
    "collect": ("sbatch --time=10:00:00 --export=ALL,BRAIN_MAX_NEW=3 "
                "run_v3_0_43_brain_harvest_oneshot.sh", "brain"),
    "filter": ("sbatch run_s2_dino_scores.sh", "s2_dino"),
    "train": ("sbatch --array=1-1 --job-name=rndtrain --gres=gpu:h100-80:1 "
              "--time=12:00:00 --export=ALL,TIER=curated,MIN_DINO_SCORE=0.50 "
              "run_m1_merged_seeds.sh", "rndtrain"),
}


def _log():
    return _CTX["log"]


def _cfg() -> dict:
    try:
        return json.loads(Path(_CFG_FILE).read_text())
    except Exception:
        return json.loads(json.dumps(DEFAULT_CFG))


def _save_cfg(c: dict):
    p = Path(_CFG_FILE)
    p.write_text(json.dumps(c, indent=1))
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


def _slurm(cmd_list, timeout=40):
    return _CTX["slurm"](cmd_list, timeout)


def _submit(shell_cmd: str):
    """Submit on the cluster via the dashboard's ssh path; return job id or None."""
    res = _CTX["slurm_sh"](shell_cmd)
    out = (res.get("stdout") or "") + (res.get("stderr") or "")
    m = re.search(r"Submitted batch job (\d+)", out)
    return (m.group(1) if m else None), out.strip()[:300]


def _job_state(jobid: str) -> str:
    """SLURM state for a job, or "UNKNOWN" only after sacct has had time to catch up.

    v3.22.21: a job that has just left the queue is briefly in neither `squeue`
    nor `sacct`, and the first version treated that gap as a failure — job
    44322382 COMPLETED normally and was recorded as a failed step, which then
    counted toward the stop-loss. A success miscounted as a failure is the same
    class of defect as a failure miscounted as a success. Poll a few times before
    concluding anything.
    """
    for attempt in range(4):
        q = _slurm(["squeue", "-j", jobid, "-h", "-o", "%T"], timeout=30)
        st = (q.get("stdout") or "").strip()
        if st:
            return st.splitlines()[0]
        a = _slurm(["sacct", "-j", jobid, "-X", "-n", "-o", "State"], timeout=30)
        st = ((a.get("stdout") or "").strip().splitlines() or [""])[0].strip()
        if st:
            return st.split()[0]
        if attempt < 3:
            time.sleep(20)                    # sacct lag, not a verdict
    _log().warning("[rounds] job %s in neither squeue nor sacct after 4 polls "
                   "— reporting UNKNOWN" % jobid)
    return "UNKNOWN"


def _train_metric(started_ts: float = 0.0) -> dict:
    """Holdout mAP50-95 of THIS round's train run.

    Deliberately strict about which artifact it reads. Ultralytics increments its
    run directory (train, train2, …), so a repeated recipe leaves older runs in
    place: on 2026-08-23 the round's own result lived in `train2` while `train`
    still held the previous day's M1 result. Globbing the plain `train` path would
    have attached a stale number to a fresh round — a fabricated metric, which is
    worse than none. So: take the NEWEST results.csv, and only accept it if it was
    written after this step started. Otherwise return no metric.
    """
    res = _CTX["slurm_sh"](
        "F=$(ls -t results/framework/mega_iter*/train*/results.csv 2>/dev/null | head -1); "
        "[ -z \"$F\" ] && exit 0; "
        "echo \"$F\"; stat -c %Y \"$F\"; "
        "python3 - \"$F\" <<'PY'\n"
        "import csv, sys\n"
        "rows = list(csv.DictReader(open(sys.argv[1])))\n"
        "col = [k for k in (rows[0] if rows else {}) if 'mAP50-95' in k]\n"
        "vals = [float(r[col[0]]) for r in rows if col and r.get(col[0])]\n"
        "print(max(vals) if vals else '')\n"
        "print(len(rows))\n"
        "PY", timeout=90)
    lines = [x.strip() for x in (res.get("stdout") or "").strip().splitlines() if x.strip()]
    if len(lines) < 4:
        return {}
    path, mtime, best, epochs = lines[0], lines[1], lines[2], lines[3]
    try:
        mtime_f, best_f = float(mtime), float(best)
    except ValueError:
        return {}
    if started_ts and mtime_f < started_ts:
        _log().warning("[rounds] newest results.csv (%s) predates this step — "
                       "refusing to attach a stale metric" % path)
        return {}
    return {"map50_95": round(best_f, 4), "epochs": int(epochs or 0), "source": path}


def _record(domain, step, status, detail=None, job=None, metrics=None):
    try:
        _CTX["record_step"](domain, step, status, detail=detail, job=job,
                            actor="round-scheduler", metrics=metrics)
    except Exception as e:
        _log().warning(f"[rounds] ledger write failed {domain}/{step}: {e}")


def _advance(domain: str, dcfg: dict):
    """One tick for one domain: poll the in-flight step or launch the next one."""
    st = _STATE.setdefault(domain, {"job": None, "step": None, "fails": 0,
                                    "started": 0, "day": "", "rounds_today": 0})
    db = _CTX["db"]

    if st["job"]:                                   # a step is in flight — poll it
        state = _job_state(st["job"])
        if state == "UNKNOWN":
            # still ambiguous after retries: leave it in flight rather than
            # inventing a verdict; the next tick asks again.
            return
        if state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            if time.time() - st["started"] > STEP_TIMEOUT_H * 3600:
                _record(domain, st["step"], "failed", detail="step timeout",
                        job=st["job"])
                _slurm(["scancel", st["job"]])
                st.update(job=None, step=None)
                st["fails"] += 1
            return
        if state == "COMPLETED":
            if st["step"] == "collect":
                # v3.22.12: capture licenses for whatever the harvest registered.
                # Per-registration-site hooks are whack-a-mole (HF builder datasets
                # register through a different path), so the scheduler closes the
                # gap once, after the step, for every source.
                _CTX["slurm_sh"](
                    "python -m weed_optimizer_framework.tools.license_audit backfill "
                    "2>&1 | tail -3", timeout=300)
            metrics = _train_metric(st.get("started", 0)) if st["step"] == "train" else None
            _record(domain, st["step"], "done", job=st["job"], metrics=metrics)
            if st["step"] == "train" and metrics:
                _record(domain, "eval", "done", job=st["job"], metrics=metrics,
                        detail="holdout metric from the train run's results.csv")
            _log().info(f"[rounds] {domain}: step {st['step']} done (job {st['job']})")
            st.update(job=None, step=None)
            st["fails"] = 0
        else:
            _record(domain, st["step"], "failed", detail="job state %s" % state,
                    job=st["job"])
            _log().warning(f"[rounds] {domain}: step {st['step']} FAILED ({state})")
            st.update(job=None, step=None)
            st["fails"] += 1
            if st["fails"] >= 2:
                dcfg["enabled"] = False
                dcfg["paused_reason"] = "stop-loss: 2 consecutive step failures"
                c = _cfg(); c["domains"][domain] = dcfg; _save_cfg(c)
                _log().error(f"[rounds] {domain} PAUSED by stop-loss")
        return

    # nothing in flight — figure out the next step
    cur = db.get_current_round(domain)
    steps_done = {s for s, e in (cur or {}).get("steps", {}).items()
                  if isinstance(e, dict) and e.get("status") in ("done", "skipped")}
    order = [s for s in db.ROUND_STEPS]
    nxt = next((s for s in order if s not in steps_done), None)

    if cur is None or nxt is None:                  # open a new round
        today = time.strftime("%Y-%m-%d")
        if st["day"] != today:
            st["day"], st["rounds_today"] = today, 0
        if st["rounds_today"] >= int(dcfg.get("max_rounds_per_day", 2)):
            return
        doc = db.start_round(domain, actor="round-scheduler")
        if not doc:
            return
        st["rounds_today"] += 1
        _log().info(f"[rounds] {domain}: opened round {doc.get('round_num')}")
        return

    if nxt == "label":                              # autolabel happens inside collect
        _record(domain, "label", "skipped",
                detail="autolabel-in-collect; human labeling stays on the Labeler card")
        return
    if nxt == "eval":                               # attached to train when it lands
        _record(domain, "eval", "skipped", detail="no train metric this round")
        return

    cmd = _WEED_STEPS.get(nxt)
    if not cmd:
        _record(domain, nxt, "skipped", detail="no action wired for this domain")
        return
    jobid, out = _submit(cmd[0])
    if jobid:
        _record(domain, nxt, "running", job=jobid, detail=out[:120])
        st.update(job=jobid, step=nxt, started=time.time())
        _log().info(f"[rounds] {domain}: step {nxt} submitted (job {jobid})")
    else:
        _record(domain, nxt, "failed", detail="submit failed: " + out)
        st["fails"] += 1


def _loop():
    while True:
        time.sleep(TICK_S)
        try:
            c = _cfg()
            for domain, dcfg in (c.get("domains") or {}).items():
                if dcfg.get("enabled"):
                    with _LOCK:
                        _advance(domain, dcfg)
        except Exception as e:
            try:
                _log().warning(f"[rounds] scheduler tick error: {e}")
            except Exception:
                pass


@router.get("/api/rounds/scheduler")
def scheduler_status(request: Request):
    c = _cfg()
    return JSONResponse({"ok": True, "config": c,
                         "inflight": {d: {k: v for k, v in s.items()}
                                      for d, s in _STATE.items()}})


@router.post("/api/rounds/scheduler")
async def scheduler_set(request: Request):
    actor = _CTX["actor"](request)
    if not _CTX["is_admin"](actor):
        return JSONResponse({"ok": False, "error": "administrators only"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain = re.sub(r"[^a-z0-9_]", "", str(body.get("domain") or "").lower())
    if not domain:
        return JSONResponse({"ok": False, "error": "domain required"}, status_code=400)
    c = _cfg()
    d = c["domains"].setdefault(domain, {"enabled": False, "max_rounds_per_day": 2})
    if "enabled" in body:
        d["enabled"] = bool(body["enabled"])
        d.pop("paused_reason", None)
    if "max_rounds_per_day" in body:
        d["max_rounds_per_day"] = max(1, min(6, int(body["max_rounds_per_day"])))
    _save_cfg(c)
    _log().info(f"[rounds] scheduler config by {actor}: {domain} -> {d}")
    return JSONResponse({"ok": True, "domain": domain, "config": d})


def _recover_inflight():
    """Re-adopt steps still marked `running` on the ledger after a restart.

    `_STATE` is in-process, so a dashboard restart mid-round would otherwise leave
    a live SLURM job untracked and the scheduler free to submit the same step
    again. The ledger already holds the truth (step status + real job id), so read
    it back on boot. Verified need: a restart at 17:5x happened while round 1's
    train job was 2h40m in.
    """
    try:
        db = _CTX["db"]
        for domain, dcfg in (_cfg().get("domains") or {}).items():
            if not dcfg.get("enabled"):
                continue
            cur = db.get_current_round(domain) or {}
            for step, entry in (cur.get("steps") or {}).items():
                if isinstance(entry, dict) and entry.get("status") == "running" \
                        and entry.get("job"):
                    _STATE[domain] = {"job": str(entry["job"]), "step": step,
                                      "fails": 0, "started": time.time(),
                                      "day": time.strftime("%Y-%m-%d"),
                                      "rounds_today": 1}
                    _log().info("[rounds] %s: re-adopted in-flight step %s (job %s) "
                                "from the ledger after restart"
                                % (domain, step, entry["job"]))
    except Exception as e:
        _log().warning("[rounds] in-flight recovery failed: %s" % e)


def mount(app, ctx: dict):
    _CTX.update(ctx)
    _recover_inflight()
    threading.Thread(target=_loop, daemon=True).start()
    app.include_router(router)
