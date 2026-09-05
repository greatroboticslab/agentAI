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

v3.25.0 — repairs for the failure class that stopped the loop on 2026-08-29
(two train steps hit the 12 h walltime at epoch 24/60 and 16/60, the domain
paused itself, and nothing said so for six days):
  * the train metric is read from the job-scoped artifact of the job THIS step
    submitted, never from the newest results.csv on disk
  * a running train's own epoch trace is projected against its walltime every
    tick, so "this job cannot finish" is observable at a fraction of the
    walltime instead of only at the wall
  * a failed step is held for review: the identical job is not resubmitted
    before a verdict lands or the review times out, whose deterministic fallback
    halves the epochs
  * a cancel that replaces a step is recorded `superseded` — terminal, but not a
    failure, so it never trips the stop-loss
  * fails / rounds_today / day / the step start time live in the config file, so
    a dashboard restart no longer re-arms a domain that had already failed once
  * step commands are rendered from the domain config; the literals below stay
    as the documented fallback
  * every submission is mirrored into the cluster-action history, and every tick
    writes a heartbeat, so a paused or dead scheduler is never silent again
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
# v3.25.0: outcome of the last ledger write, mirrored into the heartbeat. A loop
# that keeps ticking while every transition is lost to a down Mongo looks exactly
# like a healthy one from outside, which is the silence of 2026-08-29 again.
_LEDGER = {"ok": True, "last_error_ts": None}

TICK_S = 120
STEP_TIMEOUT_H = 30          # a step's job may run this long before declared lost
DEFAULT_CFG = {"domains": {}}    # {"weed": {"enabled": False, "max_rounds_per_day": 2}}

# v3.25.0 knobs. REVIEW_TIMEOUT_MIN is only the floor used when the domain config
# carries no brain block; TICK_WARN_S is the point at which a tick is eating its
# own cadence (the loop sleeps TICK_S between ticks, so a tick near TICK_S means
# the remote calls, not the schedule, decide how often a step is polled).
REVIEW_TIMEOUT_MIN = 90
TICK_WARN_S = 90
# A superseded step is recorded with a ledger status the ledger already accepts
# (db._ROUND_STATUSES has no "superseded") plus this detail prefix; _advance
# reads the prefix back so a superseded step is re-run rather than counted done.
SUPERSEDE_PREFIX = "superseded: "

# step -> (submit command argv ON the cluster repo root, jobname prefix)
# v3.25.0: this table is now the FALLBACK. The live commands are rendered from
# the domain config (steps templates + round_params) so a correction can change
# the next submitted line without a code change; these literals are what a domain
# whose config carries no steps block still gets, and they are the byte-identical
# seed the config templates render to with their default round_params.
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

_FALLBACK_LOGGED = set()     # (domain, step) already reported as template-less


def _log():
    return _CTX["log"]


def _repo_root() -> str:
    """Lab-side repo root — where the heartbeat lives (same resolution as sync_health)."""
    return str(_CTX.get("repo") or os.environ.get("REPO_ROOT")
               or os.path.expanduser("~/weed_llm_benchmark"))


def _cfg() -> dict:
    try:
        return json.loads(Path(_CFG_FILE).read_text())
    except Exception:
        return json.loads(json.dumps(DEFAULT_CFG))


def _save_cfg(c: dict):
    # Written from the scheduler thread and from the admin route with no lock
    # between them. A plain write_text leaves a window in which _cfg() reads a
    # truncated file, falls back to DEFAULT_CFG and reports "no domain is
    # enabled" — the alarm goes green while a domain is paused. Replace is
    # atomic, so a reader sees either the old file or the new one.
    p = Path(_CFG_FILE)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(c, indent=1))
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(str(tmp), str(p))


def _slurm(cmd_list, timeout=40):
    return _CTX["slurm"](cmd_list, timeout)


def _sh(shell_cmd: str, timeout=90) -> dict:
    """One batched shell command on the cluster repo root. Never raises."""
    try:
        return _CTX["slurm_sh"](shell_cmd, timeout) or {}
    except TypeError:            # older ctx hook without a timeout parameter
        try:
            return _CTX["slurm_sh"](shell_cmd) or {}
        except Exception as e:
            _log().warning("[rounds] remote command failed: %s" % e)
            return {}
    except Exception as e:
        _log().warning("[rounds] remote command failed: %s" % e)
        return {}


def _submit(shell_cmd: str):
    """Submit on the cluster via the dashboard's ssh path; return job id or None."""
    res = _CTX["slurm_sh"](shell_cmd)
    out = (res.get("stdout") or "") + (res.get("stderr") or "")
    m = re.search(r"Submitted batch job (\d+)", out)
    return (m.group(1) if m else None), out.strip()[:300]


def _log_action(action: str, result: dict):
    """Mirror a scheduler submission into the dashboard's cluster-action history.

    v3.25.0: the loop was the only thing on the platform that could start a GPU
    job without leaving a row in cluster_actions.jsonl, so the history panel and
    every audit built on it under-reported the scheduler's spend entirely.
    """
    fn = _CTX.get("log_action")
    if not callable(fn):
        return
    try:
        fn(action, result)
    except Exception as e:
        _log().warning("[rounds] action log failed (%s): %s" % (action, e))


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


def _sacct_start(jobid: str) -> float:
    """A job's real Start time as epoch seconds, or 0.0 when sacct cannot say.

    v3.25.0: after a dashboard restart the in-flight step's start time used to be
    the restart moment, which pushed the STEP_TIMEOUT_H deadline forward by the
    downtime and made the stale-artifact check compare against the wrong instant.
    A start time that cannot be true (clock or timezone skew between the lab and
    the cluster) is discarded rather than trusted, because a start in the future
    silently disarms both checks.
    """
    try:
        r = _slurm(["sacct", "-j", str(jobid), "-X", "-n", "-o", "Start"], timeout=30)
        raw = ((r.get("stdout") or "").strip().splitlines() or [""])[0].strip()
        if not raw or raw.lower() in ("unknown", "none"):
            return 0.0
        t = time.mktime(time.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0.0
    now = time.time()
    if t > now + 3600 or t < now - 30 * 86400:
        _log().warning("[rounds] sacct Start for job %s (%s) is outside the "
                       "plausible window — ignoring it" % (jobid, raw))
        return 0.0
    return t


# --------------------------------------------------------------------------
# per-domain state, persisted (v3.25.0)
# --------------------------------------------------------------------------
def _blank_state() -> dict:
    # walltime_warned holds the job id the in-flight projection has already
    # reported (v3.25.0: a bool could not tell one job's warning from the next
    # one's); walltime_off the job whose projection is unavailable; last_terminal
    # step -> how that step's job actually ended, which is what decides whether a
    # deterministic correction applies; unknown_ticks how many consecutive polls
    # answered neither squeue nor sacct.
    return {"job": None, "step": None, "fails": 0, "started": 0,
            "day": "", "rounds_today": 0, "review": None,
            "walltime_warned": False, "review_applied": "", "review_seen": {},
            "submits": {}, "walltime_off": "", "last_terminal": {},
            "unknown_ticks": 0}


def _state_snapshot(st: dict) -> dict:
    """The part of a domain's state that must outlive this process."""
    # v3.25.0: `submits` (the resubmit-loop cap) and `review_applied` (the memo
    # that stops the deterministic correction being applied twice for the same
    # review) were process-local, so a dashboard restart re-armed both and the
    # epochs could be halved a second time for one review.
    submits = {}
    for k, v in (st.get("submits") or {}).items():
        try:
            submits[str(k)] = int(v)
        except (TypeError, ValueError):
            continue                 # a value that is not a count is not a cap
    # last_terminal travels with them: a review outlives a restart, and a
    # correction that cannot tell what the failure was resubmits unchanged —
    # which for a walltime failure is the identical 12 h job of 2026-08-29.
    last = {}
    for k, v in (st.get("last_terminal") or {}).items():
        if isinstance(v, dict):
            last[str(k)] = {"state": str(v.get("state") or ""),
                            "job": (str(v.get("job")) if v.get("job") else None),
                            "walltime_warned": bool(v.get("walltime_warned"))}
    return {"job": (str(st.get("job")) if st.get("job") else None),
            "step": st.get("step") or None,
            "fails": int(st.get("fails") or 0),
            "day": str(st.get("day") or ""),
            "rounds_today": int(st.get("rounds_today") or 0),
            "started_sacct": float(st.get("started") or 0),
            "submits": submits,
            "review_applied": str(st.get("review_applied") or ""),
            "last_terminal": last}


def _state(domain: str, dcfg: dict = None) -> dict:
    """In-process state for a domain, seeded once from the on-disk snapshot.

    v3.25.0: `fails`, `rounds_today` and `day` were process-local, so every
    dashboard restart reset the stop-loss counter to zero and the per-day cap
    with it — a domain that had already burned one failure came back armed as if
    it were clean. The file is the memory across restarts; this dict stays the
    live copy within one process.
    """
    st = _STATE.get(domain)
    if st is not None:
        return st
    st = _blank_state()
    saved = (dcfg or {}).get("state") or {}
    try:
        st["fails"] = int(saved.get("fails") or 0)
        st["day"] = str(saved.get("day") or "")
        st["rounds_today"] = int(saved.get("rounds_today") or 0)
        st["started"] = float(saved.get("started_sacct") or 0)
        sv = saved.get("submits")
        for k, v in (sv.items() if isinstance(sv, dict) else ()):
            try:
                st["submits"][str(k)] = int(v)
            except (TypeError, ValueError):
                continue     # one unreadable counter must not cost the rest
        st["review_applied"] = str(saved.get("review_applied") or "")
        lt = saved.get("last_terminal")
        if isinstance(lt, dict):
            st["last_terminal"] = {str(k): dict(v) for k, v in lt.items()
                                   if isinstance(v, dict)}
        if saved.get("job"):
            st["job"], st["step"] = str(saved["job"]), (saved.get("step") or None)
    except Exception as e:
        _log().warning("[rounds] %s: unreadable persisted state (%s) — starting "
                       "from zero counters" % (domain, e))
    _STATE[domain] = st
    return st


def _persist_state(domain: str, st: dict, dcfg: dict = None):
    """Write a domain's counters back to the config file. Never raises."""
    snap = _state_snapshot(st)
    if isinstance(dcfg, dict):
        dcfg["state"] = snap
    try:
        c = _cfg()
        c.setdefault("domains", {}).setdefault(domain, {})["state"] = snap
        _save_cfg(c)
    except Exception as e:
        _log().warning("[rounds] %s: state not persisted (%s)" % (domain, e))


def _pause(domain: str, dcfg: dict, st: dict, reason: str):
    """Stop-loss: disable the domain, on disk, with the reason it stopped."""
    dcfg["enabled"] = False
    dcfg["paused_reason"] = reason
    dcfg["state"] = _state_snapshot(st)
    try:
        c = _cfg()
        c.setdefault("domains", {})[domain] = dcfg
        _save_cfg(c)
    except Exception as e:
        _log().warning("[rounds] %s: pause not persisted (%s)" % (domain, e))
    _log().error("[rounds] %s PAUSED by stop-loss" % domain)


# --------------------------------------------------------------------------
# step commands: rendered from the domain config (v3.25.0)
# --------------------------------------------------------------------------
def _domain_cfg(domain: str) -> dict:
    try:
        return _CTX["db"].get_domain_config(domain) or {}
    except Exception as e:
        _log().warning("[rounds] %s: domain config unreadable (%s)" % (domain, e))
        return {}


def _render_via_db(domain: str, cfg: dict, step: str, extra: dict) -> str:
    """Render through db.render_step_command: config + step + the per-submission
    fields (round, domain, iter_name, trace). `step` is that helper's own
    positional argument, so it is never passed inside the field dict."""
    fn = getattr(_CTX.get("db"), "render_step_command", None)
    if not callable(fn):
        return ""
    try:
        cmd = fn(cfg, step, **extra)
    except Exception as e:
        _log().warning("[rounds] %s/%s: step command did not render (%s)"
                       % (domain, step, e))
        return ""
    return cmd if isinstance(cmd, str) and cmd.strip() else ""


def _render_local(domain: str, step: str, tmpl: str, fields: dict) -> str:
    """Render a step template with str.format. A template that does not render is
    reported and dropped — a half-rendered sbatch line must never be submitted.

    v3.25.0: the rendered line is validated here too. `steps` templates moved out
    of code and into the per-domain config, which is writable over HTTP, and the
    result is handed to a shell on the shared cluster account. `db.render_step_command`
    validates what it renders, but this local path exists for a template that
    reached the config before that check did (or by a write straight to the
    collection), so it must not be the way around the check.
    """
    try:
        cmd = str(tmpl).format(**fields)
    except Exception as e:
        _log().warning("[rounds] %s/%s: step template does not render (%s)"
                       % (domain, step, e))
        return ""
    check = getattr(_CTX.get("db"), "validate_step_command", None)
    if callable(check):
        try:
            ok, why = check(cmd)
        except Exception as e:                       # a broken validator refuses
            ok, why = False, "validator raised: %s" % e
        if not ok:
            _log().error("[rounds] %s/%s: step command REFUSED (%s)"
                         % (domain, step, why))
            return ""
    return cmd


def _step_command(domain: str, step: str, round_num):
    """(command, params, used_fallback) for one step of one round.

    The rendered fields are recorded on the ledger with the step, so the exact
    parameters a job ran with are readable afterwards instead of inferred from
    the sbatch line.
    """
    cfg = _domain_cfg(domain)
    steps = cfg.get("steps") if isinstance(cfg.get("steps"), dict) else {}
    params = dict(cfg.get("round_params") or {})
    # A per-round merge/project name: a resubmit inside the same round reuses the
    # same merged corpus directory (the merge symlinks ~100k images), while a new
    # round gets its own, so a stale mega_iter can never be mistaken for this
    # round's run. The trace path is left to the job, which is the only place
    # that knows its own job id (BRAIN_TRACE empty => the script derives it).
    iter_name = "rnd%s_%s" % (round_num, step)
    extra = {"round": round_num, "domain": domain, "iter_name": iter_name,
             "trace": ""}
    fields = dict(params)
    fields.update(extra)
    fields["step"] = step

    tmpl = steps.get(step)
    if tmpl:
        cmd = _render_via_db(domain, cfg, step, extra) \
            or _render_local(domain, step, tmpl, fields)
        if cmd:
            return cmd, fields, False

    key = (domain, step)
    if key not in _FALLBACK_LOGGED:
        _FALLBACK_LOGGED.add(key)
        _log().info("[rounds] %s/%s: no usable step template in the domain config "
                    "— using the built-in fallback command (v3.25.0)"
                    % (domain, step))
    return (_WEED_STEPS.get(step) or (None, ""))[0], fields, True


# --------------------------------------------------------------------------
# metrics and in-flight checks
# --------------------------------------------------------------------------
# One batched command: find the job-scoped artifact, read the save_dir it
# recorded, and summarise that run's own results.csv. Printed as a single marked
# JSON line so a login profile that writes to stdout cannot shift the fields.
_TRAIN_METRIC_SH = r"""J=__JOBID__
A=$(ls -t results/framework/m1_*_"$J".json 2>/dev/null | head -1)
if [ -z "$A" ]; then echo 'TRAINMETRIC {"error": "no job-scoped artifact"}'; exit 0; fi
python3 - "$A" <<'PY'
import csv, json, os, sys
art = sys.argv[1]
r = {"artifact": art, "job_id": "", "csv": "", "mtime": 0.0, "best": None, "rows": 0}
try:
    d = json.load(open(art))
except Exception as e:
    d, r["error"] = {}, "artifact unreadable: %s" % str(e)[:120]
r["job_id"] = str(d.get("job_id") or "")
s = d.get("summary") if isinstance(d.get("summary"), dict) else {}
sd = s.get("save_dir") or d.get("save_dir") or ""
if sd:
    c = os.path.join(sd, "results.csv")
    r["csv"] = c
    if os.path.isfile(c):
        r["mtime"] = os.path.getmtime(c)
        rows = list(csv.DictReader(open(c)))
        r["rows"] = len(rows)
        col = [k for k in (rows[0] if rows else {}) if "mAP50-95" in k]
        vals = []
        for x in (rows if col else []):
            try:
                vals.append(float(x.get(col[0])))
            except (TypeError, ValueError):
                pass          # one unparseable row must not lose the whole run
        if vals:
            r["best"] = max(vals)
    else:
        r["error"] = "results.csv missing under the recorded save_dir"
elif "error" not in r:
    r["error"] = "artifact records no save_dir"
print("TRAINMETRIC " + json.dumps(r))
PY
"""


def _marked(res: dict, marker: str) -> dict:
    """Parse the last `marker {json}` line out of a remote command's output."""
    payload = ""
    for line in (res.get("stdout") or "").splitlines():
        line = line.strip()
        if line.startswith(marker + " "):
            payload = line[len(marker) + 1:]
    if not payload:
        return {}
    try:
        d = json.loads(payload)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _train_metric(started_ts: float = 0.0, jobid: str = "") -> dict:
    """Holdout mAP50-95 of THIS round's train run, read from THIS job's artifact.

    Deliberately strict about which artifact it reads. Ultralytics increments its
    run directory (train, train2, …), so a repeated recipe leaves older runs in
    place: on 2026-08-23 the round's own result lived in `train2` while `train`
    still held the previous day's M1 result. Globbing the plain `train` path would
    have attached a stale number to a fresh round — a fabricated metric, which is
    worse than none. So: take the NEWEST results.csv, and only accept it if it was
    written after this step started. Otherwise return no metric.

    v3.25.0: "newest on disk" is still not the same run as "this step's job" —
    any manual M1 seed finishing during a round would have been attached to the
    round. With a job id, read `results/framework/m1_*_<jobid>.json` instead, take
    the save_dir that job recorded, and refuse the metric when the artifact is
    missing, when the job_id inside it is not this job, or when the csv predates
    the step. Without a job id the pre-v3.25.0 path is used unchanged.
    """
    jid = re.sub(r"[^0-9]", "", str(jobid or ""))
    if not jid:
        return _train_metric_newest(started_ts)

    res = _sh(_TRAIN_METRIC_SH.replace("__JOBID__", jid), timeout=90)
    d = _marked(res, "TRAINMETRIC")
    if not d:
        _log().warning("[rounds] no train-metric answer for job %s — no metric "
                       "attached" % jid)
        return {}
    if d.get("job_id") and str(d["job_id"]) != jid:
        _log().warning("[rounds] artifact %s records job %s, not %s — refusing "
                       "to attach a foreign run's metric"
                       % (d.get("artifact"), d["job_id"], jid))
        return {}
    if d.get("best") is None:
        _log().warning("[rounds] job %s: %s — no metric attached"
                       % (jid, d.get("error") or "no mAP50-95 value in results.csv"))
        return {}
    try:
        mtime_f = float(d.get("mtime") or 0)
        best_f = float(d["best"])
    except (TypeError, ValueError):
        _log().warning("[rounds] job %s: unparseable metric %r — no metric "
                       "attached" % (jid, d.get("best")))
        return {}
    if started_ts and mtime_f and mtime_f < started_ts:
        _log().warning("[rounds] %s predates this step (%.0f < %.0f) — refusing "
                       "to attach a stale metric"
                       % (d.get("csv"), mtime_f, started_ts))
        return {}
    return {"map50_95": round(best_f, 4), "epochs": int(d.get("rows") or 0),
            "source": d.get("csv") or d.get("artifact"), "job": jid}


def _train_metric_newest(started_ts: float = 0.0) -> dict:
    """Pre-v3.25.0 reader: newest results.csv anywhere, accepted only if it was
    written after this step started. Kept for callers with no job id."""
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


# The running step's own epoch trace: last epoch record plus how many there are.
# Not folded into the state poll because that one is argv-based and retries on
# sacct lag; this is one extra shell command per tick while a train runs.
_TRACE_TAIL_SH = r"""J=__JOBID__
T=$(ls -t results/framework/_brain/__DOMAIN__/trace/*_"$J".jsonl 2>/dev/null | head -1)
if [ -z "$T" ]; then echo 'TRACE {"error": "no trace file"}'; exit 0; fi
python3 - "$T" <<'PY'
import json, sys
last, n = None, 0
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    if isinstance(rec, dict) and rec.get("kind") == "epoch":
        n, last = n + 1, rec
out = dict(last or {})
out["epoch_records"] = n
print("TRACE " + json.dumps(out, default=str))
PY
"""


def _projection_unavailable(domain: str, st: dict, jid: str, why: str):
    """Report once per job that the walltime projection cannot run, and why.

    v3.25.0: every early return in _walltime_check was silent, so a job whose
    trace carried no walltime left the projection switched off while the loop
    still looked instrumented — a detector that turns itself off unannounced is
    the failure class this check exists to close. Once per job, not per tick: at
    a 120 s cadence a 12 h train would otherwise log this 360 times.
    """
    if st.get("walltime_off") == jid:
        return
    st["walltime_off"] = jid
    _log().warning("[rounds] %s: walltime projection unavailable for job %s (%s) "
                   "— this job is not being watched for the 2026-08-29 failure"
                   % (domain, jid, why))


def _walltime_check(domain: str, st: dict):
    """Project a running train against its own walltime from its epoch trace.

    The 2026-08-29 pair of failures was only visible at the wall: the job ran for
    12 h and then reported TIMEOUT. The epoch trace carries elapsed time and the
    projected total from epoch 1, so the same verdict is available within the
    first hour. WP1 only observes it — nothing is cancelled here.
    """
    jid = re.sub(r"[^0-9]", "", str(st.get("job") or ""))
    dom = re.sub(r"[^a-z0-9_]", "", str(domain).lower())
    if not jid or not dom:
        return
    res = _sh(_TRACE_TAIL_SH.replace("__JOBID__", jid).replace("__DOMAIN__", dom),
              timeout=60)
    rec = _marked(res, "TRACE")
    if not rec or rec.get("error"):
        _projection_unavailable(domain, st, jid,
                                str((rec or {}).get("error") or "no trace answer"))
        return
    try:
        n = int(rec.get("epoch_records") or 0)
        eta = float(rec.get("eta_total_s") or 0)
        wall = float(rec.get("walltime_s") or 0)
    except (TypeError, ValueError):
        _projection_unavailable(domain, st, jid, "unreadable trace record")
        return
    if n <= 0:
        _projection_unavailable(domain, st, jid, "the trace carries no epoch record")
        return
    if wall <= 0:
        _projection_unavailable(domain, st, jid, "the trace carries no walltime_s, "
                                                 "so there is nothing to project against")
        return
    if n < 3 or eta <= 0 or eta <= 0.95 * wall:
        return
    detail = ("walltime-bound projection: epoch %s after %.1f h projects %.1f h "
              "total against a %.1f h walltime (%d epoch records, job %s)"
              % (rec.get("epoch"), float(rec.get("elapsed_s") or 0) / 3600.0,
                 eta / 3600.0, wall / 3600.0, n, jid))
    _log().warning("[rounds] %s: %s" % (domain, detail))
    # Ledger once per job: the head entry would otherwise be rewritten every
    # tick and push the step's real history out of its capped attempts list.
    # The job id (not a flag) is kept, so the deterministic correction can later
    # tell whether the warning belonged to the job that actually failed.
    if st.get("walltime_warned") != jid:
        st["walltime_warned"] = jid
        _record(domain, st.get("step") or "train", "running", detail=detail,
                job=st.get("job"))


# --------------------------------------------------------------------------
# ledger writes and the review state machine
# --------------------------------------------------------------------------
def _ledger_outcome(ok: bool):
    """Remember whether the last ledger write landed. Never raises.

    The heartbeat carries this so an alarm can tell "the loop is running" from
    "the loop is running and recording nothing"; the error timestamp is sticky so
    a recovered Mongo still shows when it was last down.
    """
    try:
        _LEDGER["ok"] = bool(ok)
        if not ok:
            _LEDGER["last_error_ts"] = time.time()
    except Exception:
        pass


def _record(domain, step, status, detail=None, job=None, metrics=None,
            review=None, params=None, actor="round-scheduler"):
    kw = {"detail": detail, "job": job, "actor": actor, "metrics": metrics}
    if review is not None:
        kw["review"] = review
    if params is not None:
        kw["params"] = params
    doc = None
    try:
        doc = _CTX["record_step"](domain, step, status, **kw)
    except TypeError:
        # A ledger without the v3.25.0 review/params fields must still get the
        # transition; losing the whole record would hide the step entirely.
        kw.pop("review", None)
        kw.pop("params", None)
        try:
            doc = _CTX["record_step"](domain, step, status, **kw)
        except Exception as e:
            _ledger_outcome(False)
            _log().warning(f"[rounds] ledger write failed {domain}/{step}: {e}")
            return
    except Exception as e:
        _ledger_outcome(False)
        _log().warning(f"[rounds] ledger write failed {domain}/{step}: {e}")
        return
    if doc is None:
        # record_round_step returns None rather than raising when Mongo is down.
        # Without this the loop keeps submitting GPU jobs whose transitions never
        # reach the ledger, and the round reads as if the step never ran.
        _ledger_outcome(False)
        _log().warning("[rounds] ledger refused %s/%s -> %s (Mongo down or the "
                       "step is not a ledger step) — this transition is lost"
                       % (domain, step, status))
        return
    _ledger_outcome(True)


def _step_entry(cur: dict, step: str) -> dict:
    e = ((cur or {}).get("steps") or {}).get(step)
    return e if isinstance(e, dict) else {}


def _is_superseded(entry) -> bool:
    return (isinstance(entry, dict) and entry.get("status") == "skipped"
            and str(entry.get("detail") or "").startswith(SUPERSEDE_PREFIX))


def _review_timeout_s(domain: str) -> float:
    try:
        brain = _domain_cfg(domain).get("brain") or {}
        return max(60.0, float(brain.get("review_timeout_min",
                                         REVIEW_TIMEOUT_MIN)) * 60.0)
    except Exception:
        return REVIEW_TIMEOUT_MIN * 60.0


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _walltime_shaped(st: dict, step: str) -> bool:
    """True when the recorded failure of `step` is the 2026-08-29 shape.

    Either sacct reported TIMEOUT for that job, or the in-flight projection had
    already warned that this job could not finish inside its walltime. Anything
    else (a CUDA OOM, a bad merge, an sbatch the scheduler rejected) is a
    different fault, and shortening the run does not address it.
    """
    rec = (st or {}).get("last_terminal") or {}
    rec = rec.get(step) if isinstance(rec, dict) else None
    if not isinstance(rec, dict):
        return False
    if str(rec.get("state") or "").upper().startswith("TIMEOUT"):
        return True
    return bool(rec.get("walltime_warned"))


def _epochs_floor(cfg: dict) -> int:
    """The lowest epochs value a correction may write, from the domain config."""
    block = cfg.get("round_params_floor")
    block = block if isinstance(block, dict) else {}
    try:
        return max(1, int(block.get("epochs", 20)))
    except (TypeError, ValueError):
        return 20


def _fallback_correction(domain: str, step: str, waited_s: float, st: dict = None,
                         from_round=None, review_id: str = "") -> str:
    """The deterministic correction applied when no verdict arrives in time.

    Halving the epochs is the one correction the loop can make on its own that
    addresses the observed failure (a job that cannot finish inside its walltime)
    without changing what is being measured: the time cap stays where it is, so
    the run still ends with a valid checkpoint. The floor keeps the run
    comparable to the sealed recipes.

    v3.25.0: the correction is conditioned on the failure's shape and leaves a
    record. Halving the epochs of a step that died of something else corrects
    nothing, and an unrecorded permanent config write moves the campaign off the
    recipe the noise floors were measured on with no way back.
    """
    mins = int(waited_s // 60)
    if step != "train":
        return ("review timed out after %d min; no deterministic parameter "
                "correction is defined for step %s, resubmitting unchanged"
                % (mins, step))
    if not _walltime_shaped(st or {}, step):
        rec = ((st or {}).get("last_terminal") or {}).get(step) or {}
        return ("review timed out after %d min; the failure (%s) is not walltime "
                "shaped, so no parameter correction applies — resubmitting "
                "unchanged" % (mins, rec.get("state") or "cause not recorded"))
    cfg = _domain_cfg(domain)
    params = dict(cfg.get("round_params") or {})
    try:
        old = int(params.get("epochs", 60) or 60)
    except (TypeError, ValueError):
        old = 60
    floor = _epochs_floor(cfg)
    half = old // 2
    # Never below the floor and never upward: a config already under the floor is
    # left where it is rather than raised by something called a correction.
    new = min(old, max(floor, half))
    if new > half:
        _log().warning("[rounds] %s: epochs %d halve to %d, under the "
                       "round_params_floor.epochs of %d — refusing to write below "
                       "the floor, keeping %d" % (domain, old, half, floor, new))
    cap = params.get("train_time_cap_h")
    if new == old:
        return ("review timed out after %d min — epochs %d is already at or below "
                "the %d floor, resubmitting unchanged" % (mins, old, floor))
    detail = ("review timed out after %d min — deterministic fallback applied: "
              "epochs %d -> %d (previous value %d; revert with "
              "round_params.epochs=%d), train_time_cap_h %s unchanged"
              % (mins, old, new, old, old, cap))
    # The correction record travels in the same config write as the value it
    # explains, so a config can never hold a corrected epochs count with no
    # provenance for it.
    record = {"at": _iso_now(), "by": "round-scheduler", "key": "epochs",
              "old": old, "new": new,
              "reason": ("review timed out after %d min on a walltime-shaped "
                         "failure" % mins),
              "from_round": from_round, "review_id": str(review_id or "")}
    prior = cfg.get("round_params_corrections")
    prior = [x for x in prior if isinstance(x, dict)] if isinstance(prior, list) else []
    try:
        ok = _CTX["db"].set_domain_config(
            domain,
            {"round_params": {"epochs": new},
             "round_params_corrections": (prior + [record])[-20:]},
            actor="round-scheduler")
        if not ok:
            detail += " (config write refused — the next job keeps the old epochs)"
    except Exception as e:
        detail += " (config write failed: %s)" % str(e)[:80]
    return detail


def _review_gate(domain: str, cur: dict, step: str, st: dict) -> bool:
    """True when `step` may be (re)submitted now.

    v3.25.0: a failed step used to be resubmitted on the very next tick with the
    identical command. On 2026-08-29 that meant a second 12 h job with the same
    recipe that had just proved it could not finish, and the stop-loss then paused
    the domain on the pair. A failure now waits for a verdict, or for the review
    timeout and its deterministic correction.
    """
    entry = _step_entry(cur, step)
    rev = entry.get("review")
    if not isinstance(rev, dict) or rev.get("status") != "awaiting":
        st["review"] = None
        (st.get("review_seen") or {}).pop(step, None)
        return True
    try:
        queued = float(rev.get("queued_at") or 0)
    except (TypeError, ValueError):
        queued = 0.0
    if not queued:
        # A review entry with no queue time still gets the full wait, timed from
        # the first tick that saw it, rather than being read as already expired.
        queued = st.setdefault("review_seen", {}).setdefault(step, time.time())
    waited = time.time() - queued
    timeout = _review_timeout_s(domain)
    if waited < timeout:
        st["review"] = {"status": "awaiting", "step": step, "queued_at": queued}
        _log().info("[rounds] %s: step %s held for review (%d of %d min) — not "
                    "resubmitting" % (domain, step, waited // 60, timeout // 60))
        return False

    # A head entry owned by a supervisor or a person is not overwritten by this
    # actor (db.merge_step_entry), so the `applied` write below can land in
    # `attempts` and leave the head still `awaiting`. Without this memo the
    # correction would then be re-applied on every tick, halving the epochs
    # again and again.
    memo = "%s:%s" % (step, int(queued))
    if st.get("review_applied") == memo:
        st["review"] = None
        return True
    st["review_applied"] = memo
    (st.get("review_seen") or {}).pop(step, None)

    detail = _fallback_correction(domain, step, waited, st,
                                  from_round=(cur or {}).get("round_num"),
                                  review_id=rev.get("review_id") or "")
    _log().warning("[rounds] %s: %s" % (domain, detail))
    _record(domain, step, "failed", detail=detail, job=entry.get("job"),
            review={"status": "applied", "review_id": rev.get("review_id") or "",
                    "queued_at": queued})
    st["review"] = None
    return True


def supersede_step(domain: str, step: str, reason: str = "",
                   actor: str = "round-scheduler") -> dict:
    """Cancel a step's job and record it as superseded — terminal, not failed.

    v3.25.0: the loop had exactly one terminal outcome for a step that did not
    complete, `failed`, and two of those pause the domain. A step cancelled on
    purpose because its parameters were corrected is not evidence that the domain
    is broken, so it must not consume a stop-loss life. The ledger only accepts
    db._ROUND_STATUSES, which has no `superseded`, so the outcome is written as
    `skipped` with a detail that names it; _advance reads that prefix back and
    re-runs the step instead of counting it done.
    """
    # v3.25.0: the lock used to be advisory here — a failed acquire fell through
    # and ran anyway, so a supersede arriving during a slow tick could clear the
    # state dict while _advance was mid-flight and cancel a job the tick was
    # still reasoning about. A busy answer is the caller's to retry.
    if not _LOCK.acquire(timeout=30):
        res = {"ok": False, "busy": True, "domain": domain, "step": step,
               "error": "a scheduler tick is in flight — supersede not attempted"}
        try:
            _log().warning("[rounds] supersede %s/%s not attempted: the scheduler "
                           "lock was held for 30s" % (domain, step))
        except Exception:
            pass
        _log_action("rounds_supersede", res)
        return res
    try:
        st = _state(domain, (_cfg().get("domains") or {}).get(domain))
        job = str(st.get("job") or "") if (st.get("step") == step) else ""
        if not job:
            job = str(_step_entry(_CTX["db"].get_current_round(domain) or {},
                                  step).get("job") or "")
        cancelled = ""
        if job:
            r = _slurm(["scancel", job]) or {}
            cancelled = ((r.get("stdout") or "") + (r.get("stderr") or "")).strip()[:200]
        detail = SUPERSEDE_PREFIX + (str(reason or "no reason given"))[:400]
        _record(domain, step, "skipped", detail=detail, job=(job or None),
                actor=actor)
        if st.get("step") == step:
            st.update(job=None, step=None, walltime_warned=False)
            _persist_state(domain, st)
        res = {"ok": True, "domain": domain, "step": step, "job": job or None,
               "status": "skipped", "superseded": True, "reason": reason,
               "actor": actor, "scancel": cancelled}
    except Exception as e:
        res = {"ok": False, "domain": domain, "step": step, "error": str(e)[:200]}
        try:
            _log().warning("[rounds] supersede %s/%s failed: %s" % (domain, step, e))
        except Exception:
            pass
    finally:
        _LOCK.release()
    _log_action("rounds_supersede", res)
    return res


# --------------------------------------------------------------------------
# the tick
# --------------------------------------------------------------------------
def _terminal(st: dict, step: str, state: str):
    """Remember how a step's job ended — the input to the correction's decision."""
    if not step:
        return
    st.setdefault("last_terminal", {})[step] = {
        "state": str(state), "job": (str(st.get("job")) if st.get("job") else None),
        "walltime_warned": (bool(st.get("walltime_warned"))
                            and str(st.get("walltime_warned")) == str(st.get("job")))}


def _step_deadline(domain: str, st: dict, dcfg: dict, state: str) -> bool:
    """Declare an in-flight step lost once STEP_TIMEOUT_H has passed; True if it did.

    v3.25.0: this deadline lived inside the RUNNING/PENDING branch, so a job that
    neither squeue nor sacct would answer for (UNKNOWN) was never timed out and
    never alarmed: the step stayed in flight indefinitely while the heartbeat
    still reported the domain enabled with no paused_reason.
    """
    if not st.get("started"):
        return False
    if time.time() - st["started"] <= STEP_TIMEOUT_H * 3600:
        return False
    _record(domain, st["step"], "failed",
            detail="step timeout: %dh in state %s" % (STEP_TIMEOUT_H, state),
            job=st["job"])
    _slurm(["scancel", st["job"]])
    _terminal(st, st["step"], "step timeout (%s)" % state)
    st.update(job=None, step=None, walltime_warned=False)
    st["unknown_ticks"] = 0
    st["fails"] += 1
    _persist_state(domain, st, dcfg)
    return True


def _advance(domain: str, dcfg: dict):
    """One tick for one domain: poll the in-flight step or launch the next one."""
    st = _state(domain, dcfg)
    db = _CTX["db"]

    if st["job"]:                                   # a step is in flight — poll it
        state = _job_state(st["job"])
        if state == "UNKNOWN":
            # still ambiguous after retries: leave it in flight rather than
            # inventing a verdict; the next tick asks again. The step deadline
            # still applies, and the tick is counted so a signal can see how long
            # this has been going on.
            st["unknown_ticks"] = int(st.get("unknown_ticks") or 0) + 1
            _step_deadline(domain, st, dcfg, state)
            return
        st["unknown_ticks"] = 0
        if state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            if state == "RUNNING" and st["step"] == "train":
                _walltime_check(domain, st)
            _step_deadline(domain, st, dcfg, state)
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
            metrics = (_train_metric(st.get("started", 0), st.get("job"))
                       if st["step"] == "train" else None)
            _record(domain, st["step"], "done", job=st["job"], metrics=metrics)
            if st["step"] == "train" and metrics:
                _record(domain, "eval", "done", job=st["job"], metrics=metrics,
                        detail="holdout metric from the train run's results.csv")
            _log().info(f"[rounds] {domain}: step {st['step']} done (job {st['job']})")
            # A step that completed has no failure shape any more; leaving the
            # old one would let a later review be corrected on a stale cause.
            (st.get("last_terminal") or {}).pop(st["step"], None)
            st.update(job=None, step=None, walltime_warned=False)
            st["fails"] = 0
            _persist_state(domain, st, dcfg)
        else:
            review = {"status": "awaiting",
                      "review_id": "auto-%s-%d" % (st["job"], int(time.time())),
                      "queued_at": time.time()}
            # Before the in-flight fields are cleared: what the correction reads.
            _terminal(st, st["step"], state)
            _record(domain, st["step"], "failed", detail="job state %s" % state,
                    job=st["job"], review=review)
            _log().warning(f"[rounds] {domain}: step {st['step']} FAILED ({state})")
            _log().info("[rounds] %s: step %s awaiting review — no resubmit for up "
                        "to %d min" % (domain, st["step"],
                                       _review_timeout_s(domain) // 60))
            st["review"] = {"status": "awaiting", "step": st["step"],
                            "queued_at": review["queued_at"]}
            st.update(job=None, step=None, walltime_warned=False)
            st["fails"] += 1
            _persist_state(domain, st, dcfg)
            if st["fails"] >= 2:
                _pause(domain, dcfg, st, "stop-loss: 2 consecutive step failures")
        return

    # nothing in flight — figure out the next step
    cur = db.get_current_round(domain)
    steps_done = {s for s, e in (cur or {}).get("steps", {}).items()
                  if isinstance(e, dict) and e.get("status") in ("done", "skipped")
                  and not _is_superseded(e)}
    order = [s for s in db.ROUND_STEPS]
    nxt = next((s for s in order if s not in steps_done), None)

    if cur is None or nxt is None:                  # open a new round
        today = time.strftime("%Y-%m-%d")
        if st["day"] != today:
            st["day"], st["rounds_today"] = today, 0
            _persist_state(domain, st, dcfg)
        if st["rounds_today"] >= int(dcfg.get("max_rounds_per_day", 2)):
            return
        doc = db.start_round(domain, actor="round-scheduler")
        if not doc:
            return
        st["rounds_today"] += 1
        _persist_state(domain, st, dcfg)
        _log().info(f"[rounds] {domain}: opened round {doc.get('round_num')}")
        return

    if nxt == "label":                              # autolabel happens inside collect
        _record(domain, "label", "skipped",
                detail="autolabel-in-collect; human labeling stays on the Labeler card")
        return
    if nxt == "eval":                               # attached to train when it lands
        _record(domain, "eval", "skipped", detail="no train metric this round")
        return

    if not _review_gate(domain, cur, nxt, st):
        return

    # A step whose head entry stops advancing (a supervisor-owned entry the
    # scheduler may not overwrite, a ledger that refuses writes) would otherwise
    # be resubmitted for as long as the domain is enabled, one GPU job per tick.
    # Two failures already pause a domain, so a third submission of the same step
    # in the same round is a loop, not a retry.
    key = "%s:%s" % (cur.get("round_num"), nxt)
    tries = int((st.get("submits") or {}).get(key, 0))
    if tries >= 3:
        _pause(domain, dcfg, st, "resubmit loop: step %s of round %s submitted "
                                 "%d times without leaving the ledger head"
                                 % (nxt, cur.get("round_num"), tries))
        return

    cmd, params, _fallback = _step_command(domain, nxt, cur.get("round_num"))
    if not cmd:
        _record(domain, nxt, "skipped", detail="no action wired for this domain")
        return
    jobid, out = _submit(cmd)
    _log_action("rounds_" + nxt, {"ok": bool(jobid), "domain": domain,
                                  "round": cur.get("round_num"), "step": nxt,
                                  "jobid": jobid, "cmd": cmd, "msg": out})
    if jobid:
        st.setdefault("submits", {})[key] = tries + 1
        _record(domain, nxt, "running", job=jobid, detail=out[:120], params=params)
        st.update(job=jobid, step=nxt, started=time.time(), walltime_warned=False)
        st["review"] = None
        _persist_state(domain, st, dcfg)
        _log().info(f"[rounds] {domain}: step {nxt} submitted (job {jobid})")
    else:
        queued_at = time.time()
        # A rejected sbatch is not a walltime failure — recorded so the review's
        # deterministic fallback resubmits it unchanged instead of shortening a
        # run that never started.
        _terminal(st, nxt, "submit failed")
        _record(domain, nxt, "failed", detail="submit failed: " + out,
                review={"status": "awaiting",
                        "review_id": "auto-submit-%d" % int(queued_at),
                        "queued_at": queued_at})
        # The queue time travels with the heartbeat: scheduler_health can only
        # raise its "review overdue" warn when it can date the wait.
        st["review"] = {"status": "awaiting", "step": nxt, "queued_at": queued_at}
        st["fails"] += 1
        _persist_state(domain, st, dcfg)


def _heartbeat(cfg: dict, tick_duration_s: float):
    """Positive proof the scheduler is alive, per tick.

    Same inverted semantics as the sync alarm: an outage cannot be expected to
    report itself, so the loop keeps proving it ran and the absence of that proof
    is the alarm. Every configured domain is listed, enabled or not, because a
    domain paused by the stop-loss is exactly the state that went unnoticed for
    six days after 2026-08-29.
    """
    try:
        domains = {}
        for domain, dcfg in (cfg.get("domains") or {}).items():
            dcfg = dcfg if isinstance(dcfg, dict) else {}
            st = _STATE.get(domain)
            if st is None:
                saved = dcfg.get("state") or {}
                st = {"job": saved.get("job"), "step": saved.get("step"),
                      "fails": saved.get("fails"), "review": None,
                      "rounds_today": saved.get("rounds_today")}
            domains[domain] = {
                "enabled": bool(dcfg.get("enabled")),
                "paused_reason": str(dcfg.get("paused_reason") or ""),
                "job": (str(st.get("job")) if st.get("job") else None),
                "step": st.get("step") or None,
                "fails": int(st.get("fails") or 0),
                "review": st.get("review") or None,
                "rounds_today": int(st.get("rounds_today") or 0),
                "unknown_ticks": int(st.get("unknown_ticks") or 0),
            }
        # mongo_ok: a tick that runs while every ledger write is lost advances
        # nothing, and without this the heartbeat reports it as a healthy loop.
        try:
            last_err = (float(_LEDGER.get("last_error_ts"))
                        if _LEDGER.get("last_error_ts") else None)
        except (TypeError, ValueError):
            last_err = None
        payload = {"ts": time.time(), "tick_s": TICK_S,
                   "tick_duration_s": round(float(tick_duration_s), 3),
                   "mongo_ok": bool(_LEDGER.get("ok", True)),
                   "mongo_last_error_ts": last_err,
                   "domains": domains}
        p = Path(_repo_root()) / "results" / "framework" / "scheduler_status.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(str(tmp), str(p))     # a half-written heartbeat reads as crit
    except Exception as e:
        try:
            _log().warning("[rounds] heartbeat not written: %s" % e)
        except Exception:
            pass


def _loop():
    while True:
        time.sleep(TICK_S)
        t0 = time.time()
        c = {}
        try:
            c = _cfg()
            for domain, dcfg in (c.get("domains") or {}).items():
                # A hand-edited config whose domain block is not an object would
                # otherwise raise here and skip every other domain's tick.
                if not isinstance(dcfg, dict):
                    _log().warning("[rounds] %s: config block is not an object "
                                   "— skipping this domain" % domain)
                    continue
                if dcfg.get("enabled"):
                    with _LOCK:
                        _advance(domain, dcfg)
        except Exception as e:
            try:
                _log().warning(f"[rounds] scheduler tick error: {e}")
            except Exception:
                pass
        dur = time.time() - t0
        try:
            if dur > TICK_WARN_S:
                _log().warning("[rounds] tick took %.0fs (over %ds) — remote calls, "
                               "not the schedule, are setting the poll cadence"
                               % (dur, TICK_WARN_S))
        except Exception:
            pass
        _heartbeat(c if c else _cfg(), dur)


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

    v3.25.0: the counters come back too. The first version rebuilt the state with
    `fails: 0`, `rounds_today: 1` and `started: now`, so a restart cleared the
    stop-loss history, misreported the per-day cap, and moved the step-timeout
    deadline forward by the downtime. The persisted snapshot carries the counters
    and `sacct` carries the job's real start time.
    """
    try:
        db = _CTX["db"]
        for domain, dcfg in (_cfg().get("domains") or {}).items():
            dcfg = dcfg if isinstance(dcfg, dict) else {}
            _STATE.pop(domain, None)
            st = _state(domain, dcfg)
            if not dcfg.get("enabled"):
                continue
            cur = db.get_current_round(domain) or {}
            for step, entry in (cur.get("steps") or {}).items():
                if isinstance(entry, dict) and entry.get("status") == "running" \
                        and entry.get("job"):
                    st["job"], st["step"] = str(entry["job"]), step
                    st["started"] = (_sacct_start(st["job"]) or st.get("started")
                                     or time.time())
                    if not st["day"]:
                        st["day"] = time.strftime("%Y-%m-%d")
                        st["rounds_today"] = max(1, int(st["rounds_today"] or 0))
                    _log().info("[rounds] %s: re-adopted in-flight step %s (job %s) "
                                "from the ledger after restart (fails=%d, "
                                "rounds_today=%d)"
                                % (domain, step, entry["job"], st["fails"],
                                   st["rounds_today"]))
            _persist_state(domain, st, dcfg)
    except Exception as e:
        _log().warning("[rounds] in-flight recovery failed: %s" % e)


def mount(app, ctx: dict):
    _CTX.update(ctx)
    _recover_inflight()
    # The loop sleeps TICK_S before its first heartbeat, so without this every
    # restart would show the "rounds are not advancing" banner for two minutes.
    _heartbeat(_cfg(), 0.0)
    threading.Thread(target=_loop, daemon=True).start()
    app.include_router(router)
