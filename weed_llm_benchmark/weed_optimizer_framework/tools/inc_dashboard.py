"""The INC campaign surface on the lab dashboard (docs/INC_AUTOPILOT.md, section e).

What it serves
--------------
Read routes, all lab-local. They read what the campaign ticker
(inc_autopilot/campaign.py, run by the round scheduler) and the executor
already wrote, through campaign.py's own paths and readers, and the approval
queue. No GET route reaches the cluster: an INC read over ssh per page view
would add logins the Bridges-2 login node throttles, and the ticker already
makes the one batched call per tick.

    GET  /api/inc/campaign               campaigns as the ticker holds them, budget, heartbeat
    GET  /api/inc/lineage                experiments and the edges between them
    GET  /api/inc/{exp}/snapshot         the latest cached record, as a step x chain grid
    GET  /api/inc/{exp}/diagnoses        the ticker's recorded diagnoses (or a labelled lab preview)
    GET  /api/inc/{exp}/ledger?line=N    lines of the ticker's lab copy of the ledger
    GET  /api/inc/proposals              the item in flight, filed items, R4 cards, brain state
    GET  /api/inc/track                  track record per lever and per proposer
    GET  /api/inc/replay                 the last replay result and the prospective records
    GET  /api/health/inc                 ok | warn | crit (unauthenticated, like the other alarms)
    POST /api/inc/campaign               admin only: create, enable, pause, goal, envelope, autonomy
    POST /api/inc/action/{verb}          through inc_autopilot.executor as human:<person>
    GET  /inc                            the page

What the ticker writes, and what is read here (campaign.py, Paths):
  * ~/.round_scheduler.json "campaigns": {name: config} (the scheduler's
    helpers and lock);
  * campaigns/<name>/state.json: phase, current experiment ("exp"), its
    experiments, the item in flight, the pause of a stop-loss, cards,
    health, ledger positions, last tick and snapshot times;
  * campaigns/<name>/latest_snapshot.json: the whole campaign-snapshot payload
    of the last tick that observed (full status: squeue, builds);
  * campaigns/<name>/diagnoses.json: the last full diagnosis {utc, exp, diagnoses};
  * snapshots/<exp>/<stamp>.json, written when the experiment's record
    changed: {"record": <campaign-snapshot record of that experiment>,
    "campaign", "tick_utc", "context"} (the ticker's earlier form, {"record":
    {snapshot, advance, report}, "status": ...}, and a bare snapshot record
    are read too); snapshots/step1/<stamp>.json, the Step 1 record;
  * snapshots/<exp>/frozen.json: the decision part of a finished experiment;
  * snapshots/<exp>/ledger.jsonl: the lab's copy of the experiment ledger,
    extended only on a read that continues it (through_sha256) and rewritten
    on a prefix_mismatch (campaign._Run._mirror_ledgers);
  * campaign_status.json: the per-tick heartbeat;
  * inc_campaign.jsonl: the campaign ledger.

Writes. The campaign POST goes through campaign.configure / pause / set_goal,
which change the config under the scheduler's lock, the ticker's state and
the campaign ledger together. The action POST is the executor's: it
authorises at execution time as the person, and nothing here runs a command
itself. A manual snapshot is kept in the ticker's snapshot history format.

Test blindness. The diagnoses shown are the ticker's record, or a preview
computed through evidence.from_snapshot on the ticker's own inputs, which
never reads a snapshot's display_only part. The report's final table (every
exam, test included) is read in exactly one place, `_display_final`, whose
output goes to the page under a "display only" label and nowhere else.
"""
from __future__ import annotations

import collections
import copy
import datetime
import json
import os
import re
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .brain import approvals as AP
from .inc_autopilot import budget as BU
from .inc_autopilot import campaign as C
from .inc_autopilot import diagnose as DG
from .inc_autopilot import evidence as E
from .inc_autopilot import executor as EX
from .inc_autopilot import levers as LV
from .inc_autopilot import model as M
from .inc_autopilot import outcome as OC
from .inc_autopilot import remote as RM
from .inc_dashboard_page import PAGE as _PAGE

router = APIRouter()
_CTX = {}

DOMAIN = M.DOMAIN
NAME_RE = C.NAME_RE
# INC_DIR entries that are not experiments (evidence.RESERVED_DIRS), plus the
# campaign's own directory.
RESERVED = tuple(E.RESERVED_DIRS) + ("_campaign",)
DISPLAY_ONLY_LABEL = "display only - never used by decisions"
# inc/report.py:13: ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts.
VERDICT_TO_TRUTH = {"ACCEPT": "helps", "HOLD": "neutral", "REJECT": "hurts"}
RECIPE_ORDER = ("full", "freeze", "lora")
# The ticker runs every campaign.EVERY_N_TICKS-th round-scheduler tick of 120 s
# (600 s, the driver's WATCH_INTERVAL). Three missed ticks is an alarm.
TICKER_EXPECTED_S = C.EVERY_N_TICKS * 120
STALE_AFTER_S = 3 * TICKER_EXPECTED_S
MAX_ROWS = 200
LEDGER_CONTEXT_MAX = 100
HISTORY_SCAN = 20
# snapshots/<exp>/<stamp>.json as campaign._Run._persist_snapshot names them
# (20260927T093136Z.json), and a manual snapshot kept by this module (.manual).
STAMP_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z(\.manual)?\.json\Z")
# A campaign goal is decision input ("goal met -> COMPLETE"), so it may not
# name an exam other than dev (contract (d)); campaign.check_goal checks its form.
NON_DEV_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(test|ood22|ood23|imageweeds)(?![A-Za-z0-9])")
DIAG_CACHE_S = 30.0
REPLAY_CACHE_S = 60.0
# /api/health/inc is served without a login (an external monitor must reach
# it), so a person's address in any text it returns is masked.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
FILE_CACHE_MAX = 48
ACK_MAX = 300
# Ticker cards that wait on a person (campaign._alarm counts them as warn).
PERSON_CARDS = ("approval", "escalation", "cluster", "config")

# Page verbs -> policy actions. Every one goes through executor.submit; a
# policy action id is accepted as its own verb.
VERBS = {
    "snapshot": "inc_snapshot", "report": "inc_report", "advance": "inc_advance",
    "unblock": "inc_unblock_transient", "cancel": "inc_cancel_exp", "sync-outer": "inc_sync_outer",
    "relevance": "inc_relevance_build", "audit": "inc_label_audit",
    "build-pilot": "inc_build_pilot", "build-realloop": "inc_build_realloop",
    "build-baseline": "inc_build_baseline",
}
EXECUTE_APPROVED = "execute-approved"
REQUEST_KEYS = ("lever", "trigger", "cites", "argv", "parent_exp", "child_exp", "est_gpu_hours")
CONFIG_KEYS = ("enabled", "paused_reason", "goal", "autonomy", "autonomy_granted_by", "envelope_su",
               "daily_cap_su", "current_exp", "exps")

_CACHE_LOCK = threading.Lock()
_FILE_CACHE = collections.OrderedDict()   # path -> ((size, mtime_ns), parsed JSON)
_MIRRORS = collections.OrderedDict()      # path -> ((size, mtime_ns), {line: entry}, torn lines)
_DIAG_CACHE = {}                           # (exp, campaign) -> (t, key, result)
_REPLAY = {}                               # "v" -> (t, sig, status)


# ------------------------------------------------------------------ plumbing
def _log():
    lg = _CTX.get("log")
    return lg if lg is not None else _NullLog()


class _NullLog(object):
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _lab_repo_arg():
    """The lab_repo campaign.Paths and executor.Context take: None (model.py's
    tree, the one the round scheduler's ticker writes, since it passes none)
    when the dashboard's tree is that tree, else the dashboard's (tests)."""
    r = _CTX.get("repo")
    if not r:
        return None
    try:
        if Path(str(r)).resolve() == Path(M.LAB_REPO).resolve():
            return None
    except OSError:
        pass
    return str(r)


def _repo():
    return Path(_lab_repo_arg() or M.LAB_REPO)


def _paths():
    return C.Paths(_lab_repo_arg())


def _tree_mismatch():
    """Why the page cannot see the ticker's files, or "". The round scheduler's
    ticker writes under model.LAB_REPO; the page reads the dashboard's tree."""
    r = _CTX.get("repo")
    if not r or _CTX.get("ticker_tree") is False:
        return ""
    try:
        a, b = Path(str(r)).resolve(), Path(M.LAB_REPO).resolve()
    except OSError:
        return ""
    if a == b:
        return ""
    return ("the ticker writes under %s but this dashboard reads %s (REPO_ROOT and LAB_REPO "
            "differ), so the page cannot see what the ticker records" % (b, a))


def _unavailable(reason, **extra):
    out = {"available": False, "reason": reason}
    out.update(extra)
    return out


def _utc(ts=None):
    t = time.time() if ts is None else ts
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_seconds(stamp):
    try:
        return datetime.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _age(stamp, now=None):
    t = _utc_seconds(stamp)
    return None if t is None else (now or time.time()) - t


def _exp_ok(exp):
    return isinstance(exp, str) and bool(NAME_RE.match(exp)) and exp not in RESERVED


def _bad_exp(exp):
    return JSONResponse({"available": False, "reason": "%r is not an experiment name" % (exp,)},
                        status_code=400)


def _jsonl(path, limit=None):
    """Records of a JSONL file, oldest first; a torn line is skipped. None when absent."""
    try:
        with open(str(path), "rb") as fh:
            raw = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[-limit:] if limit else out


def _sig(p):
    try:
        st = Path(p).stat()
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def _read_json(p, cache=True):
    """Parsed JSON of a file (shared from a small cache: never mutate it), or None."""
    p = Path(p)
    sig = _sig(p)
    if sig is None:
        return None
    key = str(p)
    if cache:
        with _CACHE_LOCK:
            hit = _FILE_CACHE.get(key)
            if hit is not None and hit[0] == sig:
                _FILE_CACHE.move_to_end(key)
                return hit[1]
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        obj = None
    if cache:
        with _CACHE_LOCK:
            _FILE_CACHE[key] = (sig, obj)
            while len(_FILE_CACHE) > FILE_CACHE_MAX:
                _FILE_CACHE.popitem(last=False)
    return obj


def _short(v, n=160):
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= n else s[:n] + "..."


def _mask(s):
    return EMAIL_RE.sub("<person>", str(s)) if s is not None else None


# ------------------------------------------------------------ config and state
def _hooks():
    """(load, save, lock) of the scheduler config: campaign.py's, so a write here
    takes the scheduler's lock and re-reads the file inside it."""
    h = _CTX.get("cfg_hooks")
    return tuple(h) if h else C.default_cfg_hooks()


def _cfg_read():
    """(config, None) or (None, why it could not be read)."""
    try:
        c = _hooks()[0]()
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, _short(str(e), 300))
    if not isinstance(c, dict):
        return None, "the configuration is not an object"
    return c, None


def _campaigns(cfg):
    """Every campaign in the config, defaults filled in as the ticker reads them."""
    raw = (cfg or {}).get(C.CONFIG_KEY) if isinstance(cfg, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    return [C.campaign_config(raw[n], n) for n in sorted(raw) if NAME_RE.match(str(n))]


def _pick(cfg, name=None):
    cs = _campaigns(cfg)
    if name:
        return next((c for c in cs if c["name"] == name), None)
    enabled = [c for c in cs if c.get("enabled")]
    return (enabled or cs or [None])[0]


def _state(name):
    """The ticker's state of a campaign, or None before its first tick."""
    st = _read_json(_paths().state(name))
    if not isinstance(st, dict) or st.get("format") != C.STATE_FORMAT:
        return None
    return st


def _state_names():
    d = _paths().campaigns
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and NAME_RE.match(p.name))


def _states():
    """[(name, state)] of every campaign the ticker has a state for."""
    out = []
    for n in _state_names():
        st = _state(n)
        if st is not None:
            out.append((n, st))
    return out


def _paused(cfg, st):
    """The reason a campaign is paused as the ticker reads it (config
    paused_reason, else the state's stop-loss), or None. A resume the ticker
    has not seen yet (config resumed_utc after the state's pause) ends the
    pause, as campaign._Run._resumed does on the next tick."""
    if (cfg or {}).get("paused_reason"):
        return cfg["paused_reason"]
    p = (st or {}).get("paused") or {}
    if not p:
        return None
    r = (cfg or {}).get("resumed_utc")
    if (cfg or {}).get("enabled") and r and str(r) >= str(p.get("utc") or ""):
        return None
    return p.get("reason") or "paused"


def _current(cfg, st):
    """(current experiment, where it comes from): the ticker's state; the
    configured starting point only before the ticker's first tick."""
    if st and _exp_ok(st.get("exp")):
        return st["exp"], "ticker state"
    cur = (cfg or {}).get("current_exp")
    if _exp_ok(cur):
        return cur, "configuration (the ticker has not run yet)"
    return None, None


def _exec_campaign(cfg, st=None):
    """The campaign as the executor reads it (the ticker's _Run.camp)."""
    if not cfg:
        return None
    return {"name": cfg["name"], "autonomy": cfg.get("autonomy") or "off",
            "autonomy_granted_by": cfg.get("autonomy_granted_by"),
            "envelope_su": cfg.get("envelope_su"), "daily_cap_su": cfg.get("daily_cap_su"),
            "paused_reason": _paused(cfg, st)}


def _exctx(write=False):
    """The executor's Context on this lab. Reads get no slurm_sh at all, so a
    read cannot reach the cluster through the executor either."""
    return EX.Context(slurm_sh=_CTX.get("slurm_sh") if write else None,
                      resources=_resources, lab_repo=_lab_repo_arg())


def _resources():
    """`mongo_ok` from the dashboard's db module; unknown when it cannot say."""
    db = _CTX.get("db")
    out = {}
    if db is not None:
        try:
            out["mongo_ok"] = bool(db.available())
        except Exception:
            pass
    return out


def _replay():
    """executor.replay_status, kept for REPLAY_CACHE_S (it hashes the package)."""
    ctx = _exctx()
    sig = _sig(ctx.replay_result)
    now = time.time()
    with _CACHE_LOCK:
        hit = _REPLAY.get("v")
        if hit and hit[1] == sig and now - hit[0] < REPLAY_CACHE_S:
            return hit[2]
    try:
        st = EX.replay_status(ctx)
    except Exception as e:
        st = {"passed": False, "reason": "replay status unavailable: %s" % e, "recorded": None}
    with _CACHE_LOCK:
        _REPLAY["v"] = (now, sig, st)
    return st


def _run_for(cfg, st):
    """A campaign._Run holding the ticker's config and (a copy of) its state,
    with an ssh budget of zero: its readers (_composite, _prefix, _context,
    lineage) give the ticker's own evidence inputs, and nothing through it can
    reach the cluster (a call raises)."""
    run = C._Run(cfg["name"], cfg, _paths(), C._SshBudget(None, limit=0), time.time, _log(),
                 _hooks(), _resources, None, None, None)
    run.st = copy.deepcopy(st)
    return run


# ------------------------------------------------------------ cached records
def _payload(name):
    """The campaign's last campaign-snapshot payload, or None."""
    obj = _read_json(_paths().latest(name))
    return obj if isinstance(obj, dict) and obj.get("verb") == "campaign-snapshot" else None


def _status_entry(status, exp):
    for x in (status or {}).get("experiments") or []:
        if isinstance(x, dict) and x.get("exp") == exp:
            return x
    return None


def _history_files(exp):
    d = _paths().snapshots / exp
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and STAMP_RE.match(p.name))


def _usable(snap):
    return isinstance(snap, dict) and snap.get("verb") == "snapshot" and snap.get("ok") is not False


def _history_parts(obj, exp):
    """{snapshot, advance, report, status, exp_status} of one history file of
    exp, in any form the ticker has written it, or None."""
    if not isinstance(obj, dict):
        return None
    rec = obj.get("record") if isinstance(obj.get("record"), dict) else obj
    st = obj.get("status") if isinstance(obj.get("status"), dict) else None
    if rec.get("verb") == "campaign-snapshot":
        sub = (rec.get("experiments") or {}).get(exp)
        sub = sub if isinstance(sub, dict) else {}
        status = rec.get("status") if isinstance(rec.get("status"), dict) else None
        return {"snapshot": sub.get("snapshot"), "advance": sub.get("advance"),
                "report": sub.get("report"), "status": status, "exp_status": _status_entry(status, exp)}
    if rec.get("verb") == "snapshot":
        return {"snapshot": rec, "advance": None, "report": None, "status": None,
                "exp_status": st if st and "experiments" not in st else None}
    if isinstance(rec.get("snapshot"), dict):
        full = st if st and "experiments" in st else None
        return {"snapshot": rec["snapshot"], "advance": rec.get("advance"), "report": rec.get("report"),
                "status": full, "exp_status": _status_entry(full, exp) if full else st}
    return None


def _candidates(exp):
    """(candidates, skipped): the cached records of exp the ticker (or a manual
    snapshot) left, each {kind, source, snapshot, ...}."""
    out, skipped = [], []
    for name in _state_names():
        pl = _payload(name)
        sub = ((pl or {}).get("experiments") or {}).get(exp)
        if not isinstance(sub, dict):
            continue
        snap = sub.get("snapshot")
        src = "campaigns/%s/latest_snapshot.json" % name
        if not _usable(snap):
            skipped.append({"file": src, "why": "the snapshot failed: %s"
                            % _short((snap or {}).get("error"), 200)})
            continue
        status = pl.get("status") if isinstance(pl.get("status"), dict) else None
        out.append({"kind": "latest", "rank": 2, "source": src, "campaign": name, "snapshot": snap,
                    "advance": sub.get("advance"), "report": sub.get("report"), "status": status,
                    "exp_status": _status_entry(status, exp),
                    "step1": pl.get("step1") if isinstance(pl.get("step1"), dict) else None})
    for p in list(reversed(_history_files(exp)))[:HISTORY_SCAN]:
        obj = _read_json(p)
        hp = _history_parts(obj, exp)
        snap = (hp or {}).get("snapshot")
        src = "snapshots/%s/%s" % (exp, p.name)
        if not _usable(snap):
            skipped.append({"file": src, "why": "unreadable" if obj is None else
                            "not a snapshot record" if not isinstance(snap, dict) else
                            "the snapshot failed: %s" % _short(snap.get("error"), 200)})
            continue
        out.append(dict(hp, kind="history", rank=1, source=src, campaign=obj.get("campaign"),
                        step1=None, by=obj.get("by")))
        break
    fz = _read_json(_paths().frozen(exp))
    if isinstance(fz, dict) and _usable(fz.get("snapshot")):
        out.append({"kind": "frozen", "rank": 0, "source": "snapshots/%s/frozen.json" % exp,
                    "campaign": None, "snapshot": fz["snapshot"], "advance": None, "report": None,
                    "status": None, "exp_status": None, "step1": None,
                    "frozen_utc": fz.get("frozen_utc")})
    return out, skipped


def _latest(exp, now=None):
    """The newest cached record of exp (by the record's own utc; on a tie the
    ticker's latest payload, which has the full status, before a history file,
    before frozen.json), or None."""
    cands, skipped = _candidates(exp)
    if not cands:
        return None
    best = max(cands, key=lambda c: (str(c["snapshot"].get("utc") or ""), c["rank"]))
    best = dict(best)
    best["utc"] = best["snapshot"].get("utc")
    best["age_s"] = _age(best["utc"], now)
    best["skipped"] = skipped[:5]
    return best


def _display_final(exp, latest):
    """The report's final table, every exam: DISPLAY ONLY. Read nowhere else.
    frozen.json keeps only the decision part, so a finished experiment's table
    comes from its newest history file."""
    key = "%s/report.json#/final" % exp
    snaps = [latest["snapshot"]] if latest else []
    for p in list(reversed(_history_files(exp)))[:HISTORY_SCAN]:
        snap = (_history_parts(_read_json(p), exp) or {}).get("snapshot")
        if isinstance(snap, dict):
            snaps.append(snap)
    for s in snaps:
        rows = (s.get("display_only") or {}).get(key)
        if isinstance(rows, list):
            return {"label": DISPLAY_ONLY_LABEL, "rows": rows, "from_utc": s.get("utc")}
    return None


def _latest_step1():
    """The newest Step 1 record: a campaign's last payload, or the ticker's
    Step 1 history (snapshots/step1/<stamp>.json)."""
    cands = [(_payload(n) or {}).get("step1") for n in _state_names()]
    for p in list(reversed(_history_files("step1")))[:1]:
        cands.append(_read_json(p))
    best = None
    for s1 in cands:
        if isinstance(s1, dict) and s1.get("verb") == "step1" and s1.get("ok") is not False:
            if best is None or str(s1.get("utc") or "") > str(best.get("utc") or ""):
                best = s1
    return best


def _known_exps():
    names = set()
    d = _paths().snapshots
    if d.is_dir():
        names |= {p.name for p in d.iterdir() if p.is_dir() and _exp_ok(p.name)}
    for n, st in _states():
        names |= {e for e in st.get("exps") or [] if _exp_ok(e)}
        names |= {e for e in ((_payload(n) or {}).get("experiments") or {}) if _exp_ok(e)}
    return sorted(names)


def _owner(exp, name=None):
    """(config, state) of the campaign that holds exp: `name` when its state
    lists exp, else the first campaign whose state does, else (None, None)."""
    cfg, _err = _cfg_read()
    camps = {c["name"]: c for c in _campaigns(cfg)} if cfg else {}
    order = ([name] if name else []) + [n for n in sorted(camps) if n != name]
    for n in order:
        st = _state(n)
        c = camps.get(n)
        if c and st and (exp == st.get("exp") or exp in (st.get("exps") or [])):
            return c, st
    return None, None


# ------------------------------------------------------------ the ledger copy
def _mirror(exp):
    """({line: entry}, torn lines) of the ticker's copy snapshots/<exp>/ledger.jsonl, or (None, 0)."""
    p = _paths().mirror(exp)
    sig = _sig(p)
    if sig is None:
        return None, 0
    key = str(p)
    with _CACHE_LOCK:
        hit = _MIRRORS.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1], hit[2]
    rows, torn = {}, 0
    try:
        with open(key, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows[int(r["line"])] = r.get("entry")
                except (ValueError, KeyError, TypeError):
                    torn += 1
    except OSError:
        return None, 0
    with _CACHE_LOCK:
        _MIRRORS[key] = (sig, rows, torn)
        while len(_MIRRORS) > 16:
            _MIRRORS.popitem(last=False)
    return rows, torn


def _ledger_pos(exp):
    """The ticker's position in exp's ledger (state "ledger"), the furthest one."""
    best = None
    for _n, st in _states():
        pos = (st.get("ledger") or {}).get(exp)
        if isinstance(pos, dict) and (best is None or int(pos.get("next_line") or 0)
                                      > int(best.get("next_line") or 0)):
            best = pos
    return best


def _ledger_state(exp, latest=None):
    """The ledger of exp as the lab holds it: the ticker's copy, which it
    extends only with a read that continues it (the read carries the copy's
    through_sha256) and rewrites from line 0 on a prefix_mismatch. With no copy,
    the lines of a cached read from line 0. Nothing is spliced here."""
    rows, torn = _mirror(exp)
    pos = _ledger_pos(exp)
    notes = []
    if rows is not None:
        held = max(rows) if rows else 0
        nxt = int((pos or {}).get("next_line") or 0) if pos else None
        consistent = pos is not None and held == nxt and sorted(rows) == list(range(1, held + 1))
        if pos is None:
            notes.append("no campaign state records a position in this ledger")
        elif not consistent:
            notes.append("the ticker's copy (through line %d) and its recorded position (line %s) "
                         "disagree; the ticker reads the ledger again from line 0 when that happens"
                         % (held, nxt))
        if torn:
            notes.append("%d unreadable line(s) in the lab copy" % torn)
        complete = bool(pos and pos.get("complete", True) and (pos.get("n_lines") is None
                                                               or pos.get("n_lines") == nxt))
        return {"rows": rows, "next": held, "n_lines": (pos or {}).get("n_lines"),
                "through_sha256": (pos or {}).get("through_sha256"), "verified": consistent,
                "complete": complete, "source": "the ticker's copy (snapshots/%s/ledger.jsonl)" % exp,
                "notes": notes}
    led = ((latest or {}).get("snapshot") or {}).get("decision", {}).get("ledger") if latest else None
    if isinstance(led, dict) and led.get("from_line") == 0 and not led.get("error"):
        rows = {it["line"]: it.get("entry") for it in led.get("entries") or []
                if isinstance(it, dict) and isinstance(it.get("line"), int)}
        return {"rows": rows, "next": max(rows) if rows else 0, "n_lines": led.get("n_lines"),
                "through_sha256": led.get("through_sha256"), "verified": bool(led.get("complete", True)),
                "complete": bool(led.get("complete", True)),
                "source": "the read from line 0 in %s (no ticker copy)" % latest["source"],
                "notes": [] if led.get("complete", True) else
                ["the read stopped at the byte cap: lines after %s are not held" % led.get("next_line")]}
    return {"rows": {}, "next": 0, "n_lines": None, "through_sha256": None, "verified": False,
            "complete": False, "source": None,
            "notes": ["the ticker holds no copy of this ledger and no cached read starts at line 0"]}


# ------------------------------------------------------------ evidence
def _decision_only(snap):
    return C._Run._decision_only(snap)


def _evidence(exp, name=None):
    """(Evidence, info), or (None, reason). With a campaign that holds exp, the
    ticker's own inputs: its last payload's decision part and its frozen
    experiments (campaign._Run._composite), its ledger copies (_prefix) and its
    context (_context: lineage from this campaign's executions, refusals,
    outcomes, budget with its projection). Otherwise the cached records alone,
    with a context that says what it lacks."""
    latest = _latest(exp)
    if latest is None:
        return None, "no usable snapshot of %s has been cached on the lab yet" % exp
    cfg, st = _owner(exp, name)
    if st is not None:
        run = _run_for(cfg, st)
        payload = _payload(cfg["name"]) or {"verb": "campaign-snapshot", "utc": latest["utc"],
                                            "experiments": {}}
        record = run._composite(payload)
        for e in [exp] + list(st.get("exps") or []):
            if e not in record["experiments"] and _exp_ok(e):
                le = latest if e == exp else _latest(e)
                if le is not None:
                    record["experiments"][e] = {"snapshot": _decision_only(le["snapshot"])}
        status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
        ctx = run._context(status, payload)
        if exp != st.get("exp"):
            # history and advance describe the campaign's current experiment
            ctx.pop("history", None)
            ctx.pop("advance", None)
        prefix = run._prefix(record)
        source = "ticker"
        note = ("the ticker's inputs for campaign %s (its payload of %s, its ledger copies, its "
                "context)" % (cfg["name"], payload.get("utc")))
    else:
        record = {"verb": "campaign-snapshot", "utc": latest["utc"], "experiments": {}}
        for e in [exp] + [x for x in _known_exps() if x != exp]:
            le = latest if e == exp else _latest(e)
            if le is not None:
                record["experiments"][e] = {"snapshot": _decision_only(le["snapshot"])}
        ctx = {"now_utc": _utc()}
        prefix = {}
        for e in record["experiments"]:
            rows, _t = _mirror(e)
            if rows:
                prefix[e] = sorted(rows.items())
        source = "partial"
        note = ("partial: no campaign holds %s, so the context has no lineage, refusals, outcomes "
                "or budget" % exp)
    step1 = latest.get("step1") or _latest_step1()
    if isinstance(step1, dict) and "step1" not in record:
        record["step1"] = _decision_only(step1)
    ev = E.from_snapshot(record, exp, context=ctx, ledger_prefix=prefix)
    return ev, {"latest": latest, "campaign": (cfg or {}).get("name"), "source": source,
                "note": note, "context_keys": sorted(ctx)}


# ------------------------------------------------------------ diagnoses
def _recorded(exp, name=None):
    """(record, campaign) of the newest campaigns/<name>/diagnoses.json about exp."""
    best = None
    for n in _state_names():
        rec = _read_json(_paths().diagnoses(n))
        if not isinstance(rec, dict) or rec.get("exp") != exp or not isinstance(rec.get("diagnoses"), list):
            continue
        k = (n == name, str(rec.get("utc") or ""))
        if best is None or k > best[0]:
            best = (k, rec, n)
    return (best[1], best[2]) if best else (None, None)


def _diagnoses(exp, name=None):
    """{"diagnoses", "source", ...} for exp, cached for DIAG_CACHE_S."""
    p = _paths()
    key = tuple(_sig(x) for x in [p.frozen(exp), p.mirror(exp), p.ledger,
                                  _exctx().exec_log] + [f for f in _history_files(exp)[-1:]]) + \
        tuple((n, _sig(p.state(n)), _sig(p.latest(n)), _sig(p.diagnoses(n))) for n in _state_names())
    ck = (exp, name)
    now = time.time()
    with _CACHE_LOCK:
        hit = _DIAG_CACHE.get(ck)
        if hit and hit[1] == key and now - hit[0] < DIAG_CACHE_S:
            return hit[2]
    out = _compute_diagnoses(exp, name)
    with _CACHE_LOCK:
        _DIAG_CACHE[ck] = (now, key, out)
    return out


def _compute_diagnoses(exp, name):
    latest = _latest(exp)
    rec, rec_camp = _recorded(exp, name)
    if latest is None and rec is None:
        return _unavailable("no usable snapshot of %s has been cached on the lab yet" % exp, exp=exp)
    ev, info = None, None
    if latest is not None:
        try:
            ev, info = _evidence(exp, name or rec_camp)
        except Exception as e:
            info = {"error": "%s: %s" % (type(e).__name__, _short(str(e), 300))}
    out = {"available": True, "exp": exp, "snapshot_file": (latest or {}).get("source"),
           "snapshot_utc": (latest or {}).get("utc")}
    if rec is not None:
        out.update(source="ticker", campaign=rec_camp, recorded_utc=rec.get("utc"),
                   recorded_in="campaigns/%s/diagnoses.json (%s)" % (rec_camp, rec.get("utc")),
                   diagnoses=rec["diagnoses"])
    elif ev is not None:
        out.update(source="lab", diagnoses=DG.detect(ev),
                   note="a preview computed on the lab from %s; the ticker diagnoses an experiment "
                        "itself once it is done and acts only on its own record, so this is shown "
                        "and never alarmed on" % info["note"])
    else:
        return _unavailable("the cached snapshot could not be read as evidence: %s"
                            % (info if isinstance(info, str) else (info or {}).get("error")), exp=exp)
    _cfg, st = _owner(exp, name or rec_camp)
    if st is not None and st.get("exp") == exp and st.get("health"):
        out["health"] = {"utc": st.get("last_snapshot_utc"),
                         "fired": [d for d in st["health"] if isinstance(d, dict)]}
    diags = [d for d in out["diagnoses"] if isinstance(d, dict)]
    out["diagnoses"] = diags
    out["n_fired"] = sum(1 for d in diags if d.get("fired"))
    out["n_unknown"] = sum(1 for d in diags if str(d.get("summary", "")).startswith("unknown"))
    if ev is not None:
        try:
            p = LV.propose(diags, ev)
            out["computed"] = {k: p.get(k) or [] for k in ("proposals", "cards", "operations",
                                                            "deferred", "refused")}
        except Exception as e:
            out["computed"] = {"error": "%s: %s" % (type(e).__name__, _short(str(e), 300))}
    out["_ev"] = ev
    return out


def _public(d):
    return {k: v for k, v in (d or {}).items() if not str(k).startswith("_")}


def _link_cites(cites, exp):
    """Each cite with, for a ledger line, the lab route that shows it."""
    out = []
    for c in cites or []:
        if not isinstance(c, dict):
            continue
        c = dict(c)
        art = str(c.get("artifact") or "")
        e = E.exp_of(art) if art.endswith("/ledger.jsonl") else None
        if e and isinstance(c.get("line"), int):
            c["ledger_url"] = "/api/inc/%s/ledger?line=%d" % (e, c["line"])
        out.append(c)
    return out


# ------------------------------------------------------------ views
def _grid(ev, exp):
    """The step x chain table of verdict against truth, with the address of each cell."""
    defn = ev.json("%s/exp.json" % exp) or {}
    steps = defn.get("steps") if isinstance(defn.get("steps"), list) else []
    rows = DG.gate_rows(ev, exp)
    truth = DG.truth_rows(ev, exp)
    chains = set(r.chain for r in rows if r.chain)
    rec = defn.get("recipes")
    chains |= set(rec) if isinstance(rec, dict) else set(
        x.get("name") if isinstance(x, dict) else x for x in (rec or []) if x)
    chains = [c for c in RECIPE_ORDER if c in chains] + sorted(c for c in chains if c not in RECIPE_ORDER)
    ks = set(range(1, len(steps) + 1)) | set(k for k in truth if isinstance(k, int)) \
        | set(r.k for r in rows if isinstance(r.k, int))
    cells = {(r.chain, r.k): r for r in rows}
    out = []
    for k in sorted(ks):
        s = steps[k - 1] if 1 <= k <= len(steps) and isinstance(steps[k - 1], dict) else {}
        tv, tc = truth.get(k, (None, None))
        row = {"k": k, "step": s.get("name"), "clean": s.get("clean"), "planted": s.get("planted"),
               "truth": tv, "truth_cite": (_link_cites([tc], exp) or [None])[0] if tc else None,
               "chains": {}}
        for ch in chains:
            r = cells.get((ch, k))
            if r is None:
                row["chains"][ch] = None
                continue
            v = r["verdict"]
            try:
                cite = (_link_cites([r.cite("verdict")], exp) or [None])[0]
            except (KeyError, TypeError, ValueError, IndexError):
                cite = None
            row["chains"][ch] = {"verdict": v, "p_data": r["p_data"], "p_recipe": r["p_recipe"],
                                 "class_vs_loc": r["class_vs_loc"],
                                 "agree": (VERDICT_TO_TRUTH.get(v) == tv) if tv and v else None,
                                 "cite": cite}
        out.append(row)
    agreement = {}
    for ch in chains:
        cmp_ = [r["chains"][ch]["agree"] for r in out
                if r["chains"].get(ch) and r["chains"][ch]["agree"] is not None]
        agreement[ch] = {"agree": sum(1 for x in cmp_ if x), "compared": len(cmp_)}
    return {"chains": chains, "rows": out, "agreement": agreement,
            "source": rows[0].src if rows else ("report" if ev.json("%s/report.json" % exp) else None)}


def _squeue():
    """(the newest squeue the ticker shipped, its utc, its campaign) or (None, None, None)."""
    best = (None, None, None)
    for n in _state_names():
        pl = _payload(n)
        sq = ((pl or {}).get("status") or {}).get("squeue") if isinstance((pl or {}).get("status"), dict) else None
        if isinstance(sq, dict) and sq.get("ok") and isinstance(sq.get("jobs"), list):
            if best[1] is None or str(pl.get("utc") or "") > str(best[1]):
                best = (sq, pl.get("utc"), n)
    return best


def _jobs(exp, record):
    """This experiment's jobs, from the newest squeue a tick shipped."""
    sq, utc, _n = _squeue()
    rx = RM.exp_job_re(exp)
    if sq is not None:
        jobs = [dict(j, base_id=str(j.get("id") or "").split("_")[0]) for j in sq["jobs"]
                if isinstance(j, dict) and rx.match(str(j.get("name") or ""))]
        return jobs, "squeue of the ticker's snapshot at %s" % utc
    derived = ((record or {}).get("decision") or {}).get("derived") or {}
    live = (derived.get("state_runs") or {}).get("live_job_ids")
    if isinstance(live, list):
        return [{"id": j, "base_id": str(j), "name": None, "state": "submitted (per state.json)"}
                for j in live], "state.json live job ids (no squeue cached)"
    return [], "no job list cached"


def _done(artifacts, exp):
    """state.json's done, else report.json's (a snapshot may carry either), else None."""
    for name in ("%s/state.json" % exp, "%s/report.json" % exp):
        a = artifacts.get(name)
        if isinstance(a, dict) and "done" in a:
            return bool(a["done"])
    return None


def _state_view(ev, exp):
    st = ev.json("%s/state.json" % exp) or {}
    chains = {}
    for r, c in sorted((st.get("chains") or {}).items()):
        if isinstance(c, dict):
            chains[r] = {k: c.get(k) for k in ("phase", "k", "incumbent", "accepted", "neutral",
                                               "quarantined")}
    blocked = {}
    for u, b in sorted((st.get("blocked") or {}).items()):
        b = b if isinstance(b, dict) else {}
        cause = b.get("cause") if isinstance(b.get("cause"), dict) else {"kind": b.get("cause")}
        blocked[u] = {"cause": cause.get("kind"), "error": _short(b.get("error") or "", 400),
                      "phase_before": b.get("phase_before")}
    if not st:
        return {}
    out = {k: st.get(k) for k in ("type", "generation", "done", "done_utc", "updated_utc", "testing")}
    out.update(chains=chains, blocked=blocked, transient=sorted(st.get("transient") or {}))
    return out


def _budget(cfg, st):
    try:
        return EX.budget_now(_exec_campaign(cfg, st), _exctx())
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------ GET routes
def _campaign_row(c, st, now):
    s = C.summary(c, st)
    cur, cur_src = _current(c, st)
    paused = _paused(c, st)
    latest = _latest(cur, now) if cur else None
    observed = None
    if latest:
        art = (latest["snapshot"].get("decision") or {}).get("artifacts") or {}
        sst = art.get("%s/state.json" % cur) or {}
        observed = {"built": latest["snapshot"].get("built"),
                    "abandoned": latest["snapshot"].get("abandoned"),
                    "generation": sst.get("generation"), "done": _done(art, cur),
                    "n_blocked": len(sst.get("blocked") or {})}
    card = s.get("card") if isinstance(s.get("card"), dict) else None
    row = {"name": c["name"], "enabled": bool(c.get("enabled")), "paused_reason": paused,
           "goal": c.get("goal"), "phase": s.get("phase"), "exp": cur, "current_exp": cur,
           "current_source": cur_src, "configured_exp": c.get("current_exp"),
           "exps": [e for e in (s.get("exps") or c.get("exps") or []) if _exp_ok(e)],
           "autonomy": str(c.get("autonomy") or "off"),
           "autonomy_granted_by": c.get("autonomy_granted_by"),
           "envelope_su": c.get("envelope_su"), "daily_cap_su": c.get("daily_cap_su"),
           "brain_enabled": bool((c.get("brain") or {}).get("enabled")),
           "updated_by": c.get("updated_by"), "updated_utc": c.get("updated_utc"),
           "budget": _budget(c, st), "ticked": st is not None,
           "item": s.get("item"), "card": card, "cards": s.get("cards") or [],
           "fails": s.get("fails"), "errors": s.get("errors"),
           "last_error": ({k: st["last_error"].get(k) for k in ("utc", "error")}
                          if isinstance((st or {}).get("last_error"), dict) else None),
           "health": s.get("health"), "diagnoses": s.get("diagnoses"),
           "building": s.get("building"), "wait_jobs": s.get("wait_jobs"), "brain": s.get("brain"),
           "prospective": s.get("prospective"), "ticks": s.get("ticks"),
           "last_tick_utc": s.get("last_tick_utc"), "last_tick_age_s": _age(s.get("last_tick_utc"), now),
           "last_snapshot_utc": s.get("last_snapshot_utc"),
           "last_snapshot_age_s": _age(s.get("last_snapshot_utc"), now),
           "snapshot_failures": (st or {}).get("snapshot_failures") or 0,
           "latest_record": ({"exp": cur, "source": latest["source"], "utc": latest["utc"],
                              "age_s": latest["age_s"]} if latest else None),
           "observed": observed}
    return row


def _heartbeat(now=None):
    hb = _read_json(_paths().status)
    if not isinstance(hb, dict):
        return None
    ts = hb.get("ts") if isinstance(hb.get("ts"), (int, float)) else _utc_seconds(hb.get("utc"))
    return {"utc": hb.get("utc"), "ts": ts, "age_s": ((now or time.time()) - ts) if ts else None}


def _goal_options():
    out = [{"id": d, "name": n} for d, n in sorted(DG.NAMES.items())]
    out.append({"id": "D4", "name": DG.D4_NOT_READY})
    return out


@router.get("/api/inc/campaign")
def api_inc_campaign(campaign: str = ""):
    cfg, err = _cfg_read()
    if err:
        return _unavailable("the campaign configuration is unreadable: %s" % err)
    cs = _campaigns(cfg)
    if campaign and not any(c["name"] == campaign for c in cs):
        return _unavailable("no campaign named %r" % campaign, names=[c["name"] for c in cs])
    rp = _replay()
    now = time.time()
    rows = [_campaign_row(c, _state(c["name"]), now) for c in cs if not campaign or c["name"] == campaign]
    return {"available": True, "campaigns": rows, "known_exps": _known_exps(),
            "replay": {k: rp.get(k) for k in ("passed", "reason")}, "heartbeat": _heartbeat(now),
            "health": verdict(now), "reason": "" if cs else "no INC campaign is configured yet",
            "ticker_expected_s": TICKER_EXPECTED_S, "goal_options": _goal_options(),
            "tree_mismatch": _tree_mismatch() or None}


LINEAGE_EVENTS = ("executed", "recovered", "built", "build_failed")


@router.get("/api/inc/lineage")
def api_inc_lineage(campaign: str = ""):
    """Experiments and the edges between them; an edge carries the lever, the
    trigger diagnoses and the approval id that made it."""
    try:
        execs = EX.executions(_exctx())
    except Exception as e:
        _log().warning("[inc] execution log unreadable: %s" % e)
        execs = []
    if campaign:
        execs = [r for r in execs if r.get("campaign") in (campaign, None)]
    items = _approvals()
    cancelled = {str((r.get("params") or {}).get("exp")) for r in execs
                 if r.get("action") == "inc_cancel_exp" and r.get("status") == "executed"}
    nodes, edges = {}, {}

    def node(e):
        if e and _exp_ok(str(e)):
            nodes.setdefault(str(e), {"exp": str(e)})

    def edge(parent, child, src, **kw):
        if not child:
            return
        node(child)
        node(parent)
        k = (parent or "", child)
        cur = edges.setdefault(k, {"parent": parent or None, "child": child, "sources": [],
                                   "trigger": [], "approval_ids": [], "job_ids": []})
        if src not in cur["sources"]:
            cur["sources"].append(src)
        for t in kw.pop("trigger", None) or []:
            t = t.get("id") if isinstance(t, dict) else t
            if t and t not in cur["trigger"]:
                cur["trigger"].append(t)
        aid = kw.pop("approval_id", None)
        if aid and aid not in cur["approval_ids"]:
            cur["approval_ids"].append(aid)
        for j in kw.pop("job_ids", None) or []:
            if str(j) not in cur["job_ids"]:
                cur["job_ids"].append(str(j))
        for key, v in kw.items():
            if v not in (None, "", []):
                cur[key] = v

    for r in execs:
        if not str(r.get("action") or "").startswith("inc_build_"):
            continue
        child = r.get("child_exp") or (r.get("params") or {}).get("exp")
        st = r.get("status")
        item = items.get(r.get("approval_id")) if r.get("approval_id") else None
        if st == "filed" and item is not None:
            st = {"pending": "awaiting approval", "denied": "denied"}.get(item.get("status"), st)
        edge(r.get("parent_exp"), child, "executions", lever=r.get("lever"), trigger=r.get("trigger"),
             approval_id=r.get("approval_id"), job_ids=r.get("job_ids"), status=st,
             decided_by=r.get("decided_by"), basis=r.get("basis"), action=r.get("action"), utc=r.get("ts"))
    for rec in _jsonl(_paths().ledger) or []:
        if rec.get("event") not in LINEAGE_EVENTS or not rec.get("child_exp"):
            continue
        if campaign and rec.get("campaign") not in (campaign, None):
            continue
        extra = {"status": rec["event"]} if rec["event"] in ("built", "build_failed") else {}
        edge(rec.get("parent_exp"), rec.get("child_exp"), "campaign ledger", lever=rec.get("lever"),
             trigger=rec.get("trigger"), approval_id=rec.get("approval_id"),
             job_ids=rec.get("job_ids"), decided_by=rec.get("decided_by"), **extra)
    for e in _known_exps():
        node(e)
        latest = _latest(e)
        if latest is None:
            continue
        snap = latest["snapshot"]
        art = (snap.get("decision") or {}).get("artifacts") or {}
        defn = art.get("%s/exp.json" % e) or {}
        nodes[e].update({"builder": defn.get("builder"), "type": defn.get("type"),
                         "replay_mode": defn.get("replay_mode"),
                         "initialised_utc": defn.get("initialised_utc"), "done": _done(art, e),
                         "built": snap.get("built"), "abandoned": bool(snap.get("abandoned")),
                         "snapshot_utc": latest["utc"]})
        prov = art.get("_campaign/provenance/%s.json" % e)
        att = (prov or {}).get("attempts") if isinstance(prov, dict) else None
        if isinstance(att, list) and att and isinstance(att[-1], dict) and att[-1].get("parent_exp"):
            a = att[-1]
            edge(a.get("parent_exp"), e, "cluster provenance", trigger=a.get("trigger"),
                 approval_id=a.get("approval_id"), decided_by=a.get("decided_by"),
                 job_ids=[a.get("job_id")] if a.get("job_id") else [])
    for e in cancelled:
        if e in nodes:
            nodes[e]["cancelled"] = True
    children = {k[1] for k in edges}
    roots = sorted(n for n in nodes if n not in children)
    return {"available": True, "nodes": sorted(nodes.values(), key=lambda n: (
        str(n.get("initialised_utc") or "~"), n["exp"])),
            "edges": sorted(edges.values(), key=lambda x: (str(x.get("utc") or ""), x["child"])),
            "roots": roots,
            "reason": "" if nodes else "no experiment is known on the lab yet"}


@router.get("/api/inc/{exp}/snapshot")
def api_inc_snapshot(exp: str, campaign: str = ""):
    if not _exp_ok(exp):
        return _bad_exp(exp)
    latest = _latest(exp)
    if latest is None:
        _c, skipped = _candidates(exp)
        return _unavailable("no usable snapshot of %s has been cached on the lab yet" % exp
                            + (" (%d record(s), none usable)" % len(skipped) if skipped else ""),
                            exp=exp, skipped=skipped[:5])
    snap = latest["snapshot"]
    try:
        ev, info = _evidence(exp, campaign or None)
    except Exception as e:
        return _unavailable("the cached snapshot could not be read as evidence: %s: %s"
                            % (type(e).__name__, _short(str(e), 300)), exp=exp, file=latest["source"])
    if ev is None:
        return _unavailable(info, exp=exp)
    led = _ledger_state(exp, latest)
    jobs, jobs_src = _jobs(exp, snap)
    rep = ev.json("%s/report.json" % exp)
    defn = ev.json("%s/exp.json" % exp) or {}
    adv = latest.get("advance")
    return {"available": True, "exp": exp, "file": latest["source"], "kind": latest["kind"],
            "utc": latest["utc"], "age_s": latest["age_s"], "skipped": latest["skipped"],
            "by": latest.get("by"), "built": snap.get("built"), "abandoned": snap.get("abandoned"),
            "definition": {k: defn.get(k) for k in ("builder", "type", "replay_mode", "initialised_utc",
                                                     "decision_exam", "seeds")},
            "state": _state_view(ev, exp), "runs": ev.json("%s/%s" % (exp, E.DERIVED_STATE_RUNS)),
            "grid": _grid(ev, exp),
            "report": ({"present": True, "done": rep.get("done"), "agreement": rep.get("agreement"),
                        "gpu_hours_total": rep.get("gpu_hours_total"),
                        "generated_utc": rep.get("generated_utc")} if isinstance(rep, dict)
                       else {"present": False}),
            "jobs": jobs, "jobs_source": jobs_src,
            "advance": ({k: adv.get(k) for k in ("ok", "skipped", "error", "error_kind", "submitted",
                                                 "job_ids", "done")} if isinstance(adv, dict) else None),
            "ledger": {"lines_held": len(led["rows"]), "through_line": led["next"],
                       "file_lines": led["n_lines"], "verified": led["verified"],
                       "complete": led["complete"], "source": led["source"], "notes": led["notes"]},
            "evidence": {"source": info["source"], "note": info["note"], "campaign": info["campaign"]},
            "missing": snap.get("missing"), "evidence_notes": ev.notes[-10:],
            "display_only": _display_final(exp, latest)}


@router.get("/api/inc/{exp}/diagnoses")
def api_inc_diagnoses(exp: str, campaign: str = ""):
    if not _exp_ok(exp):
        return _bad_exp(exp)
    out = _public(_diagnoses(exp, campaign or None))
    if out.get("available"):
        out["diagnoses"] = [dict(d, cites=_link_cites(d.get("cites"), exp)) for d in out["diagnoses"]]
        out["diagnoses"].sort(key=lambda d: (not d.get("fired"),
                                             {"crit": 0, "warn": 1, "info": 2}.get(d.get("severity"), 3)))
        if out.get("health"):
            out["health"] = dict(out["health"], fired=[dict(d, cites=_link_cites(d.get("cites"), exp))
                                                       for d in out["health"]["fired"]])
    return out


@router.get("/api/inc/{exp}/ledger")
def api_inc_ledger(exp: str, line: int = 0, context: int = 5, tail: int = 20):
    """Ledger lines of exp as the lab holds them: `line` +- `context`, or the
    last `tail` lines. Lines the lab never received are listed as missing,
    not skipped over."""
    if not _exp_ok(exp):
        return _bad_exp(exp)
    led = _ledger_state(exp, _latest(exp))
    rows = led["rows"]
    if not rows:
        return _unavailable("no ledger line of %s has reached the lab yet" % exp, exp=exp,
                            notes=led["notes"])
    context = max(0, min(int(context or 0), LEDGER_CONTEXT_MAX))
    last = max(rows)
    if line and line > 0:
        lo, hi = max(1, line - context), line + context
    else:
        n = max(1, min(int(tail or 20), LEDGER_CONTEXT_MAX))
        lo, hi = max(1, last - n + 1), last
    out = [{"line": i, "entry": rows[i]} for i in range(lo, hi + 1) if i in rows]
    missing = [i for i in range(lo, min(hi, last) + 1) if i not in rows]
    res = {"available": True, "exp": exp, "line": line or None, "first": lo, "last": hi,
           "rows": out, "missing": missing, "held_through": led["next"], "file_lines": led["n_lines"],
           "verified": led["verified"], "complete": led["complete"], "source": led["source"],
           "notes": led["notes"]}
    if line and line not in rows:
        res["reason"] = ("line %d is not held on the lab (held through line %s)" % (line, led["next"]))
    return res


def _approvals():
    try:
        return AP.state(DOMAIN, root=str(_exctx().approvals_root))
    except Exception as e:
        _log().warning("[inc] approval queue unreadable: %s" % e)
        return {}


def _item_view(it):
    cx = it.get("context") if isinstance(it.get("context"), dict) else {}
    by = cx.get("proposed_by") or it.get("requested_by")
    kind = ("deterministic" if by == M.AUTOPILOT_ACTOR else
            "brain" if str(by or "").startswith("tier2:") else "other")
    return {"id": it.get("id"), "ts": it.get("ts"), "status": it.get("status"), "action": it.get("action"),
            "risk": it.get("risk"), "params": it.get("params"), "requested_by": it.get("requested_by"),
            "proposed_by": by, "kind": kind, "campaign": cx.get("campaign"), "lever": cx.get("lever"),
            "trigger": cx.get("trigger"), "cites": cx.get("cites"), "argv": cx.get("argv"),
            "parent_exp": cx.get("parent_exp"), "child_exp": cx.get("child_exp"),
            "est_su": it.get("est_su"), "reason": it.get("reason"), "decided_by": it.get("decided_by"),
            "decision_reason": it.get("decision_reason"), "decision_basis": it.get("decision_basis"),
            "execution": it.get("execution"), "brain_notes": cx.get("brain_notes")}


def _plan_digests(campaign):
    """The brain digests the ticker staged (plans/<campaign>/<n>.input.json), newest first."""
    base = _paths().plans
    dirs = [base / campaign] if campaign else ([p for p in base.iterdir() if p.is_dir()]
                                               if base.is_dir() else [])
    out = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.glob("*.input.json"):
            obj = _read_json(p)
            if isinstance(obj, dict):
                out.append({"campaign": d.name, "n": obj.get("n"), "exp": obj.get("exp"),
                            "sha256": obj.get("sha256"), "created_utc": obj.get("created_utc"),
                            "tokens_estimated": obj.get("tokens_estimated")})
    return sorted(out, key=lambda s: (s["campaign"], s["n"] if isinstance(s["n"], int) else -1),
                  reverse=True)[:20]


def _card_rows(st, exp, computed):
    """R4 cards: the ticker's recorded ones (deterministic and brain), completed
    from the lever menu's card rows, then the rules' preview on exp."""
    menu = {}
    try:
        menu = LV.load_menu().get("cards") or {}
    except Exception:
        pass
    cards = []
    for c in reversed((st or {}).get("cards") or []):
        if isinstance(c, dict):
            row = dict(menu.get(c.get("lever")) or {})
            row.update({k: v for k, v in c.items() if v not in (None, "", [])})
            row["source"] = "ticker (%s, %s)" % (c.get("exp"), c.get("utc"))
            cards.append(row)
    for c in (computed or {}).get("cards") or []:
        if isinstance(c, dict):
            cards.append(dict(c, source="rules on %s (preview)" % exp, cites=_link_cites(c.get("cites"), exp)))
    seen, uniq = set(), []
    for c in cards:
        k = (c.get("lever"), c.get("exp") or exp, str(c.get("proposed_by") or ""))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


@router.get("/api/inc/proposals")
def api_inc_proposals(campaign: str = "", exp: str = ""):
    """The ticker's item in flight, what waits on a person (deterministic next
    to brain), the R4 cards and the brain's state."""
    if exp and not _exp_ok(exp):
        return _bad_exp(exp)
    items = [_item_view(it) for it in _approvals().values()
             if str(it.get("action") or "").startswith("inc_")]
    if campaign:
        items = [it for it in items if it["campaign"] in (campaign, None)]
    items.sort(key=lambda it: -(it.get("ts") or 0))
    cfg, _err = _cfg_read()
    camp = _pick(cfg, campaign or None) if cfg else None
    st = _state(camp["name"]) if camp else None
    exp = exp or (_current(camp, st)[0] if camp else None)
    computed = None
    if exp:
        d = _diagnoses(exp, (camp or {}).get("name"))
        if d.get("available"):
            computed = d.get("computed")
    summ = C.summary(camp, st) if camp else {}
    return {"available": True, "campaign": (camp or {}).get("name"), "exp": exp,
            "in_flight": summ.get("item"), "card": summ.get("card"),
            "filed": {"deterministic": [i for i in items if i["kind"] == "deterministic"][:MAX_ROWS],
                      "brain": [i for i in items if i["kind"] == "brain"][:MAX_ROWS],
                      "other": [i for i in items if i["kind"] == "other"][:MAX_ROWS]},
            "n_pending": sum(1 for i in items if i["status"] == "pending"),
            "computed": computed, "cards": _card_rows(st, exp, computed),
            "brain": summ.get("brain"), "brain_enabled": bool(((camp or {}).get("brain") or {}).get("enabled")),
            "plans": _plan_digests((camp or {}).get("name") or ""),
            "approve_route": "/api/brain/%s/approvals/{id}" % DOMAIN}


@router.get("/api/inc/track")
def api_inc_track():
    p = _paths()
    summary = _read_json(p.track_summary)
    if isinstance(summary, dict) and ("levers" in summary or "proposers" in summary):
        return {"available": True, "source": "track_record.json", **summary}
    events = _jsonl(p.track_events)
    if events is None:
        return _unavailable("no outcome or plan validation has been recorded yet")
    return {"available": True, "source": "track_record.jsonl", **OC.fold(events)}


@router.get("/api/inc/replay")
def api_inc_replay():
    st = _replay()
    rec = st.get("recorded") if isinstance(st.get("recorded"), dict) else None
    replay = {"passed": st.get("passed"), "reason": st.get("reason"),
              "code_hash_now": st.get("code_hash_now"),
              "recorded": ({k: rec.get(k) for k in ("status", "code_hash", "recorded_utc", "cases")}
                           if rec else None)}
    prospective = []
    rdir = _paths().replay
    ver = DG.rules_version()
    if rdir.is_dir():
        for p in sorted(rdir.glob("prospective*.json")):
            obj = _read_json(p)
            if isinstance(obj, dict):
                row = {k: obj.get(k) for k in ("case", "exp", "pilot", "outcome", "ready", "blocked_by",
                                               "replay_mode", "recipes", "levers", "decided_utc",
                                               "report_sha256", "rules_version")}
                row["file"] = p.name
                # Only a record of the current rules version is one a guard reads,
                # and the one compared with the real loop built after it; the others
                # (older rules, the unversioned name) stay for history, superseded.
                row["current_rules"] = obj.get("rules_version") == ver
                row["superseded"] = not row["current_rules"]
                row["compare"] = _compare_prospective(obj) if row["current_rules"] else None
                prospective.append(row)
    return {"available": True, "replay": replay, "prospective": prospective, "rules_version": ver,
            "note": "R4b (the prospective case) is the one test of generalisation: its record is "
                    "written before a person chooses the real-loop recipe, and compared after."}


def _definition(e):
    latest = _latest(e)
    if latest is None:
        return None, None
    art = (latest["snapshot"].get("decision") or {}).get("artifacts") or {}
    return art.get("%s/exp.json" % e) or {}, art


def _compare_prospective(rec):
    """The prospective D4 record against the first real loop built after it, if one is cached."""
    t0 = str(rec.get("decided_utc") or "")
    for e in _known_exps():
        defn, _art = _definition(e)
        if not defn or defn.get("builder") != "inc.realloop build" \
                or str(defn.get("initialised_utc") or "") < t0:
            continue
        r = defn.get("recipes")
        names = sorted(r) if isinstance(r, dict) else [x.get("name") if isinstance(x, dict) else x
                                                         for x in (r or [])]
        gate = defn.get("gate") if isinstance(defn.get("gate"), dict) else {}
        out = DG.compare_prospective(rec, defn.get("replay_mode"), [n for n in names if n],
                                     gate.get("flips_mode") if isinstance(gate.get("flips_mode"), str) else None)
        out["realloop"] = e
        out["realloop_initialised_utc"] = defn.get("initialised_utc")
        return out
    return None


# ------------------------------------------------------------ health
def _recorded_crit(cfg, st):
    """Fired crit diagnoses the ticker recorded on the campaign's current
    experiment: its health diagnoses of the last observation, and its last full
    diagnosis when that was about the current experiment. A record not newer
    than a person's resume (config resumed_utc; one-second stamps, so the same
    second counts as before it, as campaign._Run._resumed orders a pause) was
    answered by that resume; the ticker's next observation says whether it
    still holds."""
    out = []
    since = str((cfg or {}).get("resumed_utc") or "")
    if str((st or {}).get("last_snapshot_utc") or "") > since:
        for d in (st or {}).get("health") or []:
            if isinstance(d, dict) and d.get("fired") and d.get("severity") == "crit":
                out.append("%s %s (health, %s)" % (d.get("id"), d.get("name"), st.get("last_snapshot_utc")))
    rec = _read_json(_paths().diagnoses(cfg["name"]))
    if isinstance(rec, dict) and rec.get("exp") == (st or {}).get("exp") and str(rec.get("utc") or "") > since:
        for d in rec.get("diagnoses") or []:
            if isinstance(d, dict) and d.get("fired") and d.get("severity") == "crit":
                out.append("%s %s (diagnosed %s)" % (d.get("id"), d.get("name"), rec.get("utc")))
    return out


def verdict(now=None):
    """ok | warn | crit over every configured campaign, from the ticker's own
    records (never from a lab-computed diagnosis). Never raises.

    crit: the config cannot be read; the page cannot see the ticker's tree
    while a campaign is enabled; a campaign paused (a stop-loss, in the config
    or only in the ticker's state, or a person); an enabled campaign the
    ticker has not ticked for STALE_AFTER_S (or never, that long after it was
    enabled); a fired crit diagnosis the ticker recorded on the current
    experiment. warn: the ticker raising; failed snapshots; a card waiting on
    a person (approval, escalation, cluster, config) or a COMPLETE campaign
    (campaign._alarm's warn); an enabled campaign not ticked yet (for
    STALE_AFTER_S after it was enabled or configured); autonomy on
    without a replay pass on this code; this check failing itself. ok
    otherwise, including "no campaign configured"."""
    now = now or time.time()
    try:
        return _verdict(now)
    except Exception as e:
        return {"level": "warn", "ok": False, "campaigns": [], "checked_ts": now,
                "reason": "INC health self-check failed (%s)" % _short(_mask(str(e)), 120)}


def _verdict(now):
    cfg, err = _cfg_read()
    if err:
        return {"level": "crit", "ok": False, "campaigns": [], "checked_ts": now,
                "reason": "the campaign configuration cannot be read, so whether a campaign is running "
                          "is unknown"}
    rows, crit, warn = [], [], []
    replay_ok = bool(_replay().get("passed"))
    hb = _heartbeat(now)
    camps = _campaigns(cfg)
    tree = _tree_mismatch()
    if tree and camps:
        (crit if any(c.get("enabled") for c in camps) else warn).append(tree)
    for c in camps:
        name = c["name"]
        st = _state(name)
        cur, _src = _current(c, st)
        paused = _paused(c, st)
        last = _utc_seconds((st or {}).get("last_tick_utc"))
        since = max([t for t in (_utc_seconds(c.get("resumed_utc")), _utc_seconds(c.get("updated_utc")))
                     if t] or [0]) or None
        row = {"campaign": name, "enabled": bool(c.get("enabled")), "paused_reason": _mask(paused),
               "phase": (st or {}).get("phase"), "current_exp": cur,
               "last_tick_age_s": (now - last) if last else None, "crit": [], "warn": []}
        if paused:
            row["crit"].append("paused: %s" % _short(_mask(paused), 200))
        elif c.get("enabled"):
            if last is None:
                waited = (now - since) if since else None
                if waited is None or waited > STALE_AFTER_S:
                    row["crit"].append("the ticker has never ticked this enabled campaign")
                else:
                    row["warn"].append("not ticked yet (enabled %d min ago; one tick every %d min)"
                                       % (waited // 60, TICKER_EXPECTED_S // 60))
            elif now - last > STALE_AFTER_S:
                # the ticker ticks a paused or disabled campaign that has a state
                # too, so a fresh last tick needs no grace after a resume
                row["crit"].append("last ticked %d min ago (one tick is expected every %d min)"
                                   % ((now - last) // 60, TICKER_EXPECTED_S // 60))
            if st:
                if st.get("errors"):
                    row["warn"].append("the ticker raised on %d tick(s) in a row: %s"
                                       % (st["errors"], _short((st.get("last_error") or {}).get("error"), 160)))
                if st.get("snapshot_failures"):
                    row["warn"].append("%d campaign snapshot(s) in a row failed" % st["snapshot_failures"])
                card = st.get("card") or {}
                if st.get("phase") == "COMPLETE":
                    row["warn"].append("complete: %s; a person decides the next step"
                                       % _short(card.get("title") or "", 160))
                elif card.get("kind") in PERSON_CARDS:
                    row["warn"].append("%s: %s" % (card["kind"], _short(card.get("title") or "", 160)))
                row["crit"] += _recorded_crit(c, st)
        if str(c.get("autonomy") or "off") == "envelope" and not replay_ok:
            row["warn"].append("autonomy is on but no replay pass is recorded on this code; R3 builds "
                               "wait for a person")
        row["crit"] = [_mask(x) for x in row["crit"]]
        row["warn"] = [_mask(x) for x in row["warn"]]
        crit += ["%s: %s" % (name, x) for x in row["crit"]]
        warn += ["%s: %s" % (name, x) for x in row["warn"]]
        rows.append(row)
    if crit:
        level, why = "crit", "; ".join(crit)
    elif warn:
        level, why = "warn", "; ".join(warn)
    elif not rows:
        level, why = "ok", "no INC campaign is configured"
    else:
        level, why = "ok", "%d campaign(s), none alarming" % len(rows)
    return {"level": level, "ok": level == "ok", "reason": why, "campaigns": rows,
            "heartbeat": {"utc": (hb or {}).get("utc"), "age_s": (hb or {}).get("age_s")},
            "stale_after_s": STALE_AFTER_S, "checked_ts": now}


@router.get("/api/health/inc")
def health_inc(request: Request):
    v = verdict()
    return JSONResponse(v, status_code=200 if v["ok"] else 503)


# ------------------------------------------------------------ POST routes
def _who(request):
    """(actor, None) or (None, JSONResponse refusal). The actor comes from the
    dashboard's identity hook (session, API key or Basic login), never from
    the body."""
    fn = _CTX.get("actor_of")
    if not callable(fn):
        return None, JSONResponse({"ok": False, "reason": "no identity hook is wired; refusing a write "
                                                          "with no author"}, status_code=403)
    try:
        actor = str(fn(request) or "").strip()
    except Exception as e:
        return None, JSONResponse({"ok": False, "reason": "could not identify the caller: %s" % e},
                                  status_code=403)
    if not actor:
        return None, JSONResponse({"ok": False, "reason": "could not identify the caller"},
                                  status_code=401)
    return actor, None


def _allowed(hook, actor):
    fn = _CTX.get(hook)
    if not callable(fn):
        return False
    try:
        return bool(fn(actor))
    except Exception:
        return False


async def _body(request):
    try:
        b = await request.json()
    except Exception:
        b = {}
    return b if isinstance(b, dict) else {}


def _num_in(v, lo, hi, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ValueError("%s must be a number" % what)
    if v < lo or (hi is not None and v > hi):
        raise ValueError("%s must be in [%s, %s], got %s" % (what, lo, hi if hi is not None else "-", v))
    return float(v)


def _goal(v):
    """The goal as campaign.check_goal normalises it (None clears it), refused
    when it names an exam other than dev."""
    g = C.check_goal(v)
    if g is not None and NON_DEV_RE.search(json.dumps(g)):
        raise ValueError("a goal is decision input and may name no exam but %s (it mentions %s)"
                         % (M.DECISION_EXAM, NON_DEV_RE.search(json.dumps(g)).group(1)))
    return g


def _plan_changes(body, actor):
    """The campaign.py calls a POST body asks for, validated, in the order
    they run: {"configure": kwargs, "goal": (goal,) or None, "pause": reason or None}."""
    conf = {}
    resume = body.get("resume") is True or body.get("enabled") is True
    if resume and "pause" in body:
        raise ValueError("a body cannot both pause and resume the campaign")
    if resume:
        conf["enable"] = True
    elif body.get("enabled") is False:
        conf["enable"] = False
    dom, _src = BU.domain_budget(None)
    if "envelope_su" in body:
        cap = BU._num(dom.get("su_envelope", dom.get("envelope")))
        conf["envelope_su"] = _num_in(body["envelope_su"], 0, cap,
                                      "envelope_su (capped by the domain envelope)")
    if "daily_cap_su" in body:
        cap = BU._num(dom.get("daily_cap"))
        conf["daily_cap_su"] = _num_in(body["daily_cap_su"], 0, cap,
                                       "daily_cap_su (capped by the domain cap)")
    if "autonomy" in body:
        mode = str(body.get("autonomy") or "")
        if mode not in ("off", "envelope"):
            raise ValueError("autonomy must be 'off' or 'envelope'")
        if mode == "envelope" and "@" not in actor:
            raise ValueError("autonomy is granted in a person's name, and %r is not a person's "
                             "account (sign in with your own account)" % actor)
        conf["autonomy"] = mode
    goal = None
    if "goal" in body:
        goal = (_goal(body.get("goal")),)
    pause = None
    if "pause" in body:
        pause = str(body.get("pause") or "").strip()[:300]
        if not pause:
            raise ValueError("a pause needs a reason")
    if not conf and goal is None and pause is None and body.get("create") is not True:
        raise ValueError("nothing to change (enabled, resume, pause, goal, envelope_su, daily_cap_su "
                         "or autonomy)")
    return {"configure": conf, "goal": goal, "pause": pause}


def _apply_campaign(name, plan, by, create_exp):
    """Runs on a worker thread: campaign.configure / set_goal / pause, each a
    read-modify-write of this campaign's block under the scheduler's lock, a
    state update and a campaign-ledger entry. Returns (created, applied)."""
    hooks = _hooks()
    lab = _lab_repo_arg()
    applied = []
    conf = dict(plan["configure"])
    if create_exp:
        conf.update(exps=[create_exp], current=create_exp)
    if conf:
        C.configure(name, by, cfg_hooks=hooks, lab_repo=lab, **conf)
        applied.append("configure(%s)" % ", ".join(sorted(conf)))
    if plan["goal"] is not None:
        C.set_goal(name, plan["goal"][0], by, cfg_hooks=hooks, lab_repo=lab)
        applied.append("set_goal")
    if plan["pause"] is not None:
        C.pause(name, plan["pause"], by, cfg_hooks=hooks, lab_repo=lab)
        applied.append("pause")
    return applied


@router.post("/api/inc/campaign")
async def api_inc_campaign_set(request: Request):
    """Administrators only: create a campaign, enable / disable it, pause /
    resume it, set its goal, its envelope and daily cap, and its autonomy.
    Every change goes through campaign.py, as the person."""
    actor, refusal = _who(request)
    if refusal is not None:
        return refusal
    if not _allowed("is_admin", actor):
        return JSONResponse({"ok": False, "reason": "administrators only"}, status_code=403)
    body = await _body(request)
    name = str(body.get("campaign") or "")
    if not NAME_RE.match(name):
        return JSONResponse({"ok": False, "reason": "campaign must be a name matching %s"
                                                    % NAME_RE.pattern}, status_code=400)
    by = "human:" + actor
    if not C.HUMAN_RE.match(by):
        return JSONResponse({"ok": False, "reason": "%r cannot be recorded as a person (human:<email>); "
                                                    "sign in with your account" % by}, status_code=403)
    cfg, err = _cfg_read()
    if err:
        return JSONResponse({"ok": False, "reason": "the configuration is unreadable (%s); refusing "
                                                    "to overwrite it" % err}, status_code=409)
    before = next((c for c in _campaigns(cfg) if c["name"] == name), None)
    create_exp = None
    if before is None:
        if body.get("create") is not True:
            return JSONResponse({"ok": False, "reason": "no campaign named %r (send create: true "
                                                        "with the experiment it starts from)" % name},
                                status_code=404)
        create_exp = body.get("exp")
        if not _exp_ok(create_exp):
            return JSONResponse({"ok": False, "reason": "a new campaign needs exp, the experiment it "
                                                        "follows first"}, status_code=400)
    try:
        plan = _plan_changes(body, actor)
    except ValueError as e:
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=400)
    try:
        applied = await run_in_threadpool(_apply_campaign, name, plan, by, create_exp)
    except ValueError as e:
        return JSONResponse({"ok": False, "reason": "not written: %s" % e}, status_code=409)
    except OSError as e:
        return JSONResponse({"ok": False, "reason": "not written: %s" % e}, status_code=500)
    cfg2, _err = _cfg_read()
    after = next((c for c in _campaigns(cfg2) if c["name"] == name), None) or {}
    changes = {k: [(before or {}).get(k), after.get(k)] for k in CONFIG_KEYS
               if (before or {}).get(k) != after.get(k)}
    la = _CTX.get("log_action")
    if callable(la):
        try:
            la("inc_campaign_config", {"ok": True, "actor": actor, "campaign": name,
                                       "msg": ("; ".join(applied) + " " + json.dumps(changes, default=str)
                                               + (" reason: " + str(body.get("reason"))[:200]
                                                  if body.get("reason") else ""))[:500]})
        except Exception:
            pass
    _log().info("[inc] campaign %s changed by %s: %s" % (name, actor, applied))
    out = {"ok": True, "campaign": name, "created": before is None, "applied": applied,
           "changes": changes, "config": after}
    if after.get("autonomy") == "envelope":
        st = _replay()
        if not st.get("passed"):
            out["warning"] = ("autonomy is on, but the envelope rule grants nothing until a replay "
                              "pass is recorded on this code: %s" % st.get("reason"))
    return out


def _latest_pilot():
    """The newest finished pilot cached on the lab (inc.pilot build, a chain,
    done), by its initialised_utc, or None."""
    best = None
    for e in _known_exps():
        defn, art = _definition(e)
        if not defn or defn.get("builder") != DG.PILOT_BUILDER or defn.get("type") != "chain":
            continue
        if _done(art, e) is not True:
            continue
        k = str(defn.get("initialised_utc") or "")
        if best is None or k > best[0]:
            best = (k, e)
    return best[1] if best else None


def _realloop_request(verb, body):
    """(params, parent) of the real-loop build this request runs, or (None,
    None) when it runs none: the build-realloop verb's params, or the
    params of the inc_build_realloop approval item execute-approved names."""
    if verb == EXECUTE_APPROVED:
        try:
            item = AP.state(DOMAIN, root=_exctx().approvals_root).get(str(body.get("approval_id") or ""))
        except Exception:
            item = None
        if not isinstance(item, dict) or item.get("action") != "inc_build_realloop":
            return None, None
        cx = item.get("context") if isinstance(item.get("context"), dict) else {}
        return (item.get("params") if isinstance(item.get("params"), dict) else {}), cx.get("parent_exp")
    if VERBS.get(verb, verb) != "inc_build_realloop":
        return None, None
    return (body.get("params") if isinstance(body.get("params"), dict) else {}), body.get("parent_exp")


def _prospective_guard(verb, body, st=None):
    """"" when this request may build a real loop, else why not. The first
    real loop after a pilot is the one test of generalisation (R4b) only
    when D4's decision on that pilot was frozen before it and the loop
    follows it: the ticker writes replay/prospective_d4_<pilot>__<rules
    version>.json, and every L2/L6 path applies diagnose.prospective_guard
    (a READY record under the current rules version, the ticker's sha256 of
    it when its state has one, and a build whose replay mode, recipes and
    gate flips mode are the record's). This route applies it to the
    build-realloop verb and to execute-approved of an inc_build_realloop
    item. The pilot is the build's parent (a rebuild of a real loop follows
    no pilot decision, as the ticker's own guard), else the latest finished
    pilot. A person may still build with an acknowledgement, which the
    execution log records."""
    params, parent = _realloop_request(verb, body)
    if params is None:
        return ""
    ack = str(body.get("acknowledge_non_prospective") or "").strip()
    if parent:
        defn, _art = _definition(parent)
        if (defn or {}).get("builder") == DG.REALLOOP_BUILDER:
            return ""
        pilot = parent
    else:
        pilot = _latest_pilot()
    if pilot is not None:
        ver = DG.rules_version()
        rec = ((st or {}).get("prospective") or {}).get(pilot) or {}
        sha = rec.get("sha256") if rec.get("rules_version") == ver else None
        why, _differs = DG.prospective_guard(_paths().replay, pilot, params, sha256=sha, version=ver)
        if not why:
            return ""
    if ack:
        return ""
    if pilot is None:
        return ("no finished pilot is cached on the lab, so whether D4's decision is frozen in a "
                "prospective record cannot be checked; wait for the ticker, or send "
                "acknowledge_non_prospective with your reason")
    return ("%s: a real loop built now would make R4b non-prospective, or would not follow D4's frozen "
            "decision. Wait for the ticker to diagnose %s under the current rules, or send "
            "acknowledge_non_prospective with your reason" % (why, pilot))


def _keep_manual_snapshot(res, exp, camp_name, who):
    """A manual snapshot kept where the ticker keeps its history, in its format
    (snapshots/<exp>/<stamp>.manual.json); the ticker never reads it back."""
    payload = (res.get("remote") or {}).get("payload")
    if not _usable(payload) or payload.get("exp", exp) != exp:
        return None
    utc = _utc()
    p = _paths().snapshots / exp / ("%s.manual.json" % utc.replace(":", "").replace("-", ""))
    rec = {"verb": "campaign-snapshot", "ok": True, "utc": payload.get("utc"),
           "experiments": {exp: {"snapshot": payload}}, "status": None}
    try:
        C._write_json(p, {"record": rec, "campaign": camp_name, "tick_utc": utc, "by": who,
                          "context": None})
    except OSError as e:
        _log().warning("[inc] manual snapshot not kept: %s" % e)
        return None
    return str(p.name)


def _run_action(verb, body, who, cfg, st):
    """Runs on a worker thread: the executor call (one ssh at most)."""
    ctx = _exctx(write=True)
    camp = _exec_campaign(cfg, st)
    if verb == EXECUTE_APPROVED:
        ack = str(body.get("acknowledge_non_prospective") or "").strip()[:ACK_MAX]
        kw = {}
        if ack and _realloop_request(verb, body)[0] is not None:
            kw["note"] = "acknowledged non-prospective by %s: %s" % (who, ack)
        res = EX.execute_approved(str(body.get("approval_id")), camp, ctx, invoked_by=who, **kw)
        return res, res.get("action") if isinstance(res, dict) else None
    action = VERBS.get(verb, verb)
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    reason = str(body.get("reason") or "manual %s from the INC page" % verb)[:500]
    ack = str(body.get("acknowledge_non_prospective") or "").strip()[:ACK_MAX]
    if action == "inc_build_realloop" and ack:
        reason = ("%s [acknowledged non-prospective by %s: %s]" % (reason, who, ack))[:900]
    req = {"policy_action": action, "params": params, "reason": reason}
    for k in REQUEST_KEYS:
        if body.get(k) not in (None, "", []):
            req[k] = body[k]
    res = EX.submit(req, actor=who, campaign=camp, ctx=ctx)
    if isinstance(res, dict) and action == "inc_snapshot" and res.get("status") == "executed":
        res["kept_as"] = _keep_manual_snapshot(res, str(params.get("exp")), (cfg or {}).get("name"), who)
    return res, action


@router.post("/api/inc/action/{verb}")
async def api_inc_action(verb: str, request: Request):
    """One executor request as the signed-in person. The executor re-checks
    everything: the policy row, the parameters, the campaign envelope and the
    resources; this route only identifies the person and passes the request on."""
    actor, refusal = _who(request)
    if refusal is not None:
        return refusal
    if not _allowed("can_use_cluster", actor):
        return JSONResponse({"ok": False, "reason": "you do not have cluster access"}, status_code=403)
    body = await _body(request)
    if verb != EXECUTE_APPROVED and verb not in VERBS and verb not in VERBS.values():
        return JSONResponse({"ok": False, "reason": "unknown INC action %r (one of %s)"
                                                    % (verb, sorted(VERBS) + [EXECUTE_APPROVED])},
                            status_code=400)
    if verb == EXECUTE_APPROVED and not re.match(r"^[A-Za-z0-9_-]{1,64}$", str(body.get("approval_id") or "")):
        return JSONResponse({"ok": False, "reason": "approval_id is required"}, status_code=400)
    cname = str(body.get("campaign") or "")
    cfg_all, err = _cfg_read()
    if err:
        return JSONResponse({"ok": False, "reason": "the configuration is unreadable: %s" % err},
                            status_code=500)
    camp = _pick(cfg_all, cname or None)
    if cname and camp is None:
        return JSONResponse({"ok": False, "reason": "no campaign named %r" % cname}, status_code=404)
    why = _prospective_guard(verb, body, _state(camp["name"]) if camp else None)
    if why:
        return JSONResponse({"ok": False, "status": "refused", "reasons": [why]}, status_code=409)
    who = "human:" + actor
    res, action = await run_in_threadpool(_run_action, verb, body, who, camp,
                                          _state(camp["name"]) if camp else None)
    res = res if isinstance(res, dict) else {"status": "refused", "ok": False,
                                              "reasons": ["the executor returned nothing"]}
    la = _CTX.get("log_action")
    if callable(la):
        try:
            jobs = res.get("job_ids") or []
            la("inc_" + verb.replace("-", "_") if not verb.startswith("inc_") else verb,
               {"ok": bool(res.get("ok")), "actor": actor, "status": res.get("status"),
                "campaign": (camp or {}).get("name"), "approval_id": res.get("approval_id"),
                "msg": "; ".join([str(x) for x in res.get("reasons") or []][:3])[:400]
                + ("; Submitted batch job %s" % jobs[0] if jobs else "")})
        except Exception:
            pass
    status = res.get("status")
    code = 200 if status in ("executed", "filed") else 403 if status == "refused" else 502
    out = {k: v for k, v in res.items() if not str(k).startswith("_")}
    if action == "inc_snapshot" and isinstance(out.get("remote"), dict):
        # the record is kept on the lab (kept_as); the page reads it from there
        out["remote"] = {k: v for k, v in out["remote"].items() if k != "payload"}
    out["verb"], out["policy_action"] = verb, action
    return JSONResponse(json.loads(json.dumps(out, default=str)), status_code=code)


# ------------------------------------------------------------ the page
@router.get("/inc")
def page_inc():
    """The INC campaign page: English, one column on a phone, read from the
    routes above; the page itself holds no state."""
    return HTMLResponse(_PAGE)


def mount(app, ctx: dict):
    _CTX.update(ctx)
    app.include_router(router)
    why = _tree_mismatch()
    if why:
        _log().warning("[inc] %s" % why)
