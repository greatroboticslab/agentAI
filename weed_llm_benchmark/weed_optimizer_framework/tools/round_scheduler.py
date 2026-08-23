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
    "collect": ("sbatch run_v3_0_43_brain_harvest_oneshot.sh", "brain"),
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
    q = _slurm(["squeue", "-j", jobid, "-h", "-o", "%T"], timeout=30)
    st = (q.get("stdout") or "").strip()
    if st:
        return st.splitlines()[0]
    a = _slurm(["sacct", "-j", jobid, "-X", "-n", "-o", "State"], timeout=30)
    st = ((a.get("stdout") or "").strip().splitlines() or ["UNKNOWN"])[0].strip()
    return st.split()[0] if st else "UNKNOWN"


def _train_metric() -> dict:
    """Best holdout mAP50-95 from the newest round-train artifact (results.csv)."""
    res = _CTX["slurm_sh"](
        "D=$(ls -td results/framework/mega_iterm1_curated_s101/train "
        "results/framework/mega_iter*/train 2>/dev/null | head -1); "
        "tail -n +2 \"$D/results.csv\" 2>/dev/null | "
        "awk -F, 'BEGIN{b=0} {for(i=1;i<=NF;i++) if($i>b && $i<1 && i>=8 && i<=13) b=$i} END{print b}'; "
        "echo \"$D\"")
    lines = (res.get("stdout") or "").strip().splitlines()
    try:
        return {"map50_95": float(lines[0]), "source": lines[1] if len(lines) > 1 else "?"}
    except Exception:
        return {}


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
            metrics = _train_metric() if st["step"] == "train" else None
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


def mount(app, ctx: dict):
    _CTX.update(ctx)
    threading.Thread(target=_loop, daemon=True).start()
    app.include_router(router)
