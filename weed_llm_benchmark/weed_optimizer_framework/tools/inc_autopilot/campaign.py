"""INC campaign ticker: the lab side that moves a campaign along (docs/INC_AUTOPILOT.md,
component 6 and step (g)8).

The round scheduler starts `tick()` every EVERY_N_TICKS-th tick of its loop
(120 s x 5 = 600 s, the driver's WATCH_INTERVAL), on a daemon thread of its
own: one campaign ssh can take minutes (the snapshot verb's timeout is 600 s),
more than the scheduler heartbeat's 360 s staleness alarm allows a loop tick,
so the rounds loop never waits on it; a tick still running makes the next one
skip, and an in-process lock plus a lock file keep ticks (the thread's and a
hand-run CLI tick) from overlapping. Each enabled campaign moves one step
through

    RUN --(state done)--> REPORT --> DIAGNOSE --> PROPOSE --> VALIDATE --> GATE
     ^                                   |                                  |
     |          goal met, or nothing to  +--> COMPLETE (a human card)       |
     |          propose and no brain proposal                               v
     +------------------------------------------------------------------ EXECUTE

with one item in flight per campaign: the experiment that runs, or the one
proposal waiting at GATE (an unblock of a transient block waits there too
and then returns to RUN), or the label audit / relevance job it launched
(WAIT_JOB). A GATE with no item (the item ended without a phase of its own)
moves to DIAGNOSE, which observes and diagnoses again: the campaign never
idles silently. PROPOSE and VALIDATE run inside one tick; BRAIN_WAIT holds a
campaign whose deterministic layer has nothing to propose while its brain
plan is still due; brain proposals left for a person wait in COMPLETE, which
adopts an approved item on any later tick.

Outcomes of a call (executor.never_ran / executor.uncertain):
  * it never reached the cluster (ssh never connected): nothing ran or is
    charged, an approval it claimed is released, the item stays as it is and
    is driven again; not a failed step (a card after SNAPSHOT_FAILURES_CARD);
  * its outcome is unknown (a timeout, a dropped connection, a `started`
    record after a restart): charged, never run again under that proposal. A
    build is followed -- RUN on its child, and the snapshots decide: the job
    in squeue (by id or by its name inc_build_<exp>) or its provenance
    record confirms it, neither for BUILD_LOST_SNAPSHOTS snapshots means it
    did not happen and the lever is proposed again under a new id. Anything
    else pauses for a person, in a phase that observes once re-enabled.

Configuration: `~/.round_scheduler.json`, key "campaigns", written with the
scheduler's own atomic helpers (round_scheduler._cfg / _save_cfg) under its
`_LOCK`, the lock its per-domain state writes hold, so neither write drops the
other's change:

    {"campaigns": {"<name>": {"enabled": false, "paused_reason": null,
        "goal": null | {"kind": "diagnosis", "id": "D4", "name": "decision_slot_ready"}
                     | {"kind": "exp_done", "exp": "<exp>"},
        "exps": ["pilot_v1", ...], "current_exp": "pilot_v1",
        "autonomy": "off" | "envelope", "autonomy_granted_by": "human:<email>",
        "envelope_su": 300, "daily_cap_su": 120,
        "brain": {"enabled": false, "model": null},
        "updated_by": "human:<email>", "updated_utc": "...",
        "resumed_utc": "...", "switch": {"exp", "utc", "by"},
        "state": {"phase", "exp", "exps", "item", "card", "paused", "updated_utc"}}}}

A person's changes go through `configure` / `pause` / `set_goal` (the admin
route and the CLI). Two of them are requests the ticker applies once, so a
tick that was running at that moment cannot undo them: `resumed_utc` (an
enable clears a pause the ticker had set earlier) and `switch` (a new current
experiment). The ticker writes back only its view: "state", and current_exp
and exps as the campaign moves to a child.

Disabled by default: nothing is done for a campaign until a person enables
it (`configure`, the `enable` CLI verb). `autonomy: "envelope"` is the owner's
2026-09-27 rule, enforced by executor.py: an R3 build of L1, L2, L5, L8 or L9 runs
without a per-item approval when the replay tests passed on the current code
and every other rule holds; otherwise it waits for a person.

State (lab, owned by the ticker; the file is the memory across restarts):
    <CAMPAIGN_DIR>/campaigns/<name>/state.json     phase, current experiment, the item
                                                   in flight, counters, ledger positions
    <CAMPAIGN_DIR>/campaigns/<name>/latest_snapshot.json   the last campaign-snapshot record
    <CAMPAIGN_DIR>/campaigns/<name>/diagnoses.json the last full diagnosis
    <SNAPSHOT_DIR>/<exp>/<utc>.json                snapshot history, written when an
                                                   experiment's record changed: {"record":
                                                   <campaign-snapshot record of that exp>,
                                                   "context": <the ticker's context>}
    <SNAPSHOT_DIR>/step1/<utc>.json                the Step 1 record, when it changed
    <SNAPSHOT_DIR>/<exp>/ledger.jsonl              the lab's copy of the experiment ledger
    <SNAPSHOT_DIR>/<exp>/frozen.json               the decision part of a finished
                                                   experiment, reused instead of re-read
    <CAMPAIGN_LEDGER>                              append-only: every entry carries
        decided_by, trigger (diagnosis ids with their cites), parent_exp, child_exp,
        approval_id and job_ids
    <CAMPAIGN_DIR>/campaign_status.json            one status line per campaign, every tick

One ssh per tick. Bridges-2 throttles repeated logins, so a tick makes at most
one call through the injected `slurm_sh`, across every campaign:
  * first the item in flight is driven through executor.submit (filing is
    lab-only; running it is the tick's ssh), or an approved item a person or
    the brain filed is executed, or a staged brain plan is submitted;
  * only when that used no ssh, one executor.campaign_snapshot call advances
    (RUN only), reports and snapshots the live experiments, reads Step 1 and
    squeue, and pulls a due brain reply, all in one verb.
A wrapper refuses a second call in the same tick as a backstop; the order
above never reaches it.

Every tick, from the snapshot: the health diagnoses (D5, D6, D7, D8, D10,
D14). Stop-losses (contract (d)) pause the campaign (config enabled false and
paused_reason, state paused, a human card): two consecutive failed steps; any
D7, D10 or D14; a blocked unit that is not transient; a lever proposed a
fourth time in the campaign; D6 on two consecutive snapshots; a submission
other than a build whose outcome is unknown; an experiment a person
cancelled; the ticker raising three ticks in a row. A paused campaign does
nothing at all until a person enables it again. A build that fails with a
refusal levers.json marks 'retry': false (one the identical build meets
again: realloop's evidenced-pool capacity, select's draw shortfall) is not
left to the stop-loss: its request is declined at once (_build_failed), so
it is never submitted a second time, and DREF escalates to a person.

Approved items another filer queued (a brain proposal a person approved, an
item a person filed) are adopted only after the checks the ticker's own
proposals pass: a lever already applied to that parent (the campaign
lineage, and levers.applied on the last snapshot) is never run, with a card;
an L2/L6 taking a pilot's D4 decision waits for its prospective record. A
goal the ticker cannot check (check_goal) turns autonomy off and raises a
card; configure refuses autonomy 'envelope' while it stands.

Lineage: the context `lineage` records come from the executor's execution log
(every lever that ran or may have run in the campaign, a build still in
flight included; a build that failed does not count), so levers.applied never
proposes a lever twice on one parent. Remote ledger reads carry the
`through_sha256` of the previous read; a `prefix_mismatch` rewrites the lab's
copy from line 0, and an incomplete read postpones every decision.

Prospective D4 (contract (f), R4b): when D4 looks at a finished pilot, its
decision is written to REPLAY_DIR/prospective_d4_<pilot>__<rules version>.json
(diagnose.prospective_name; once per pilot and rules version) before any
proposal is made, and its sha256 goes into the campaign ledger with the
sha256 of every earlier record of that pilot it supersedes (another rules
version, or the unversioned name written before versions existed; none is
ever modified or deleted). A real loop built from a pilot's decision (L2 or
L6 whose parent is not a real loop, whatever triggered it: D4, D11, the
brain, an approved item) is never proposed, filed or run without a READY
record of the current rules version whose sha256 is the one this ticker
wrote (diagnose.current_prospective). A pilot decided on protocol v2 gets
its record only once its report's v2_check is final (diagnose.prospective_d4
returns 'pending' and writes nothing before; the ledger notes it once).

Rules changes: every DIAGNOSE records the rules version it ran under; a
campaign waiting in COMPLETE or BRAIN_WAIT with nothing in flight whose
version differs from the current one moves to DIAGNOSE (ledger
'rules_changed'), which re-diagnoses the current experiment and writes the
new record first. RUN, REPORT, WAIT_JOB (an R2 item such as L3 or L4) and
GATE reach DIAGNOSE on their own. Nothing already executed runs again: every
lever still passes levers.applied (the lineage) and the declined keys.

The brain (component 8), when the campaign config enables it: a dev-only
digest is staged locally at DIAGNOSE, submitted through the executor
(inc_plan_submit, an R2 job on the cluster), pulled back with the snapshot
(inc_plan_pull), collected against its deadline (brain_plan.collect),
validated item by item (validate.py) and merged; every brain proposal is filed
as tier2 for a person's approval. The deterministic item never waits for it.
When nothing deterministic is left, the campaign waits for a plan still in
time, then completes with a human card.

Test blindness: evidence comes only from the snapshot's `decision` part
(evidence.from_snapshot; `display_only` is dropped before), the context holds
no exam value, and budget and outcome code read dev only. Nothing here opens a
score file.

CLI (runs on the lab; `tick` makes one ssh through CLUSTER_SSH):
    python -m weed_optimizer_framework.tools.inc_autopilot.campaign status [--name N]
    python -m weed_optimizer_framework.tools.inc_autopilot.campaign tick [--name N ...]
    python -m weed_optimizer_framework.tools.inc_autopilot.campaign enable --name N --by human:<email>
        [--exp E ...] [--current E] [--autonomy off|envelope] [--envelope-su SU]
        [--daily-cap-su SU] [--brain on|off] [--brain-model M]
    python -m weed_optimizer_framework.tools.inc_autopilot.campaign pause --name N --by human:<email> --reason R
    python -m weed_optimizer_framework.tools.inc_autopilot.campaign set-goal --name N --by human:<email>
        (--diagnosis D4:decision_slot_ready | --exp-done E | --none)
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import brain_plan as BP
from . import budget as B
from . import diagnose as DG
from . import evidence as E
from . import executor as X
from . import levers as LV
from . import model as M
from . import outcome as OC
from . import validate as V
from ..brain import approvals as AP
from ..brain import policy as POL

try:
    import fcntl
except ImportError:                  # not on the lab or the cluster
    fcntl = None

STATE_FORMAT = "inc-autopilot/campaign-state/1"
STATUS_FORMAT = "inc-autopilot/campaign-status/1"
CONFIG_KEY = "campaigns"
EVERY_N_TICKS = 5                    # x round_scheduler.TICK_S 120 s = 600 s (driver WATCH_INTERVAL)
AUTO = M.AUTOPILOT_ACTOR
PHASES = ("RUN", "REPORT", "DIAGNOSE", "PROPOSE", "VALIDATE", "GATE", "EXECUTE", "WAIT_JOB",
          "BRAIN_WAIT", "COMPLETE")
OBSERVING = ("RUN", "REPORT", "DIAGNOSE", "WAIT_JOB", "BRAIN_WAIT")
ADOPTING = ("GATE", "BRAIN_WAIT", "DIAGNOSE", "COMPLETE")
MAX_FAILED_STEPS = 2                 # contract (d): 2 consecutive failed campaign steps
STALE_TO_PAUSE = 2                   # D6 on this many consecutive snapshots ("still stale")
ERRORS_TO_PAUSE = 3                  # the ticker raising on this many ticks in a row
SNAPSHOT_FAILURES_CARD = 6           # an hour of failed snapshots raises a card
BUILD_LOST_SNAPSHOTS = 3             # a build job gone from squeue with no provenance, this many snapshots
HISTORY_KEEP = 48                    # (utc, generation) per experiment for D6: 8 h at 600 s
REFUSALS_KEEP = 5
CARDS_KEEP = 50
STOP_IDS = ("D7", "D10", "D14")
DA_IN_FLIGHT = ("staged", "submitted")    # a devil's-advocate pass not yet ended
BUILD_FAILED = ("build_failed", "refused_drift", "refused_missing_module", "env_failed", "killed")
# The R4 decisions of the funnel audit (docs/FUNNEL_AUDIT.md 13), logged once
# per campaign with decided_by human-delegated (runner 5.5.4).
FUNNEL_DECISIONS = ("DEC-1", "DEC-2", "DEC-3", "DEC-4", "DEC-5", "DEC-6", "DEC-7", "DEC-8", "DEC-9", "DEC-10")
# A callable taking the _Run and returning its executor's lab hooks (tests).
LAB_HOOKS = None
# The params that tell one funnel step from another in the lineage.
FUNNEL_KEY_PARAMS = ("verb", "rl", "part", "what", "policy")
TRANSIENT_REFUSALS = ("the cluster is not reachable", "Mongo's health", "the execution log",
                      "no slurm_sh hook", "could not be locked", "collides with another request",
                      "could not be filed", "the approval log could not be written")
DEFAULT_CONFIG = {"enabled": False, "paused_reason": None, "goal": None, "exps": [],
                  "current_exp": None, "autonomy": "off", "autonomy_granted_by": None,
                  "envelope_su": None, "daily_cap_su": None,
                  "brain": {"enabled": False, "model": None}, "funnel": True}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
HUMAN_RE = re.compile(r"^human:[A-Za-z0-9][A-Za-z0-9@._+-]{0,126}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}$")

_TICK_LOCK = threading.Lock()


# --- small helpers ------------------------------------------------------------------
def _utc(t):
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _secs(stamp):
    try:
        return datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(obj):
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _read_json(path):
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _short(text, n=300):
    text = str(text)
    return text if len(text) <= n else text[:n] + "..."


class _Log(object):
    """Fallback logger (stderr) when the caller gives none."""

    def _w(self, lvl, msg):
        print("[inc-campaign] %s %s" % (lvl, msg), file=sys.stderr)

    def info(self, msg):
        self._w("INFO", msg)

    def warning(self, msg):
        self._w("WARNING", msg)

    def error(self, msg):
        self._w("ERROR", msg)


def lab_repo_arg(repo):
    """The `lab_repo` to give Paths and executor.Context for the dashboard's
    repo root `repo` (its REPO_ROOT): None when it is model.LAB_REPO's tree
    (model.py's paths, and the SU-ledger directory override), else `repo`, so
    the ticker writes its state, approvals and status where the dashboard,
    its approval queue and the /inc page read them."""
    if not repo:
        return None
    try:
        if Path(str(repo)).resolve() == Path(M.LAB_REPO).resolve():
            return None
    except OSError:
        pass
    return str(repo)


class Paths(object):
    """Every lab file the ticker reads or writes. `lab_repo` None gives model.py's
    paths (the dashboard's tree); a directory moves all of them (tests)."""

    def __init__(self, lab_repo=None):
        if lab_repo is None:
            self.brain_dir, self.campaign_dir = M.BRAIN_DIR, M.CAMPAIGN_DIR
        else:
            self.brain_dir = Path(lab_repo) / "results" / "framework" / "_brain"
            self.campaign_dir = self.brain_dir / M.DOMAIN / "inc"
        self.lab_repo = lab_repo
        self.ledger = self.campaign_dir / "inc_campaign.jsonl"         # model.CAMPAIGN_LEDGER
        self.snapshots = self.campaign_dir / "snapshots"                # model.SNAPSHOT_DIR
        self.plans = self.campaign_dir / "plans"                        # model.PLAN_DIR
        self.replay = self.campaign_dir / "replay"                      # model.REPLAY_DIR
        self.campaigns = self.campaign_dir / "campaigns"
        self.status = self.campaign_dir / "campaign_status.json"
        self.track_events = self.campaign_dir / "track_record.jsonl"    # outcome.TRACK_EVENTS
        self.track_summary = self.campaign_dir / "track_record.json"
        self.experiments = self.brain_dir / M.DOMAIN / "experiments.jsonl"
        self.lock = self.campaigns / ".tick.lock"
        # The funnel audit (docs/FUNNEL_AUDIT.md 8.3, 8.6): the claims register,
        # the lab's INC tree (its funnel/ holds prospective_da.json and the
        # fetched files the sync pushes), the contract and the R14 fixture (the
        # DA's blind markers), and a person's verify queue (L14).
        self.claims = self.campaign_dir / "claims.json"
        root = Path(lab_repo) if lab_repo is not None else M.LAB_REPO
        self.lab_inc = root / "results" / "framework" / "inc"
        self.contract = root.parent / "docs" / "FUNNEL_AUDIT.md"
        self.r14_fixtures = [root / "tests" / "fixtures" / "inc_replay" / "funnel" / "da" / name
                             for name in ("da_positive.json", "da_sycophantic.json")]
        self.verify_queue = self.brain_dir / M.DOMAIN / "known_truth" / "human_verify_queue.jsonl"

    def state(self, name):
        return self.campaigns / name / "state.json"

    def latest(self, name):
        return self.campaigns / name / "latest_snapshot.json"

    def diagnoses(self, name):
        return self.campaigns / name / "diagnoses.json"

    def mirror(self, exp):
        return self.snapshots / exp / "ledger.jsonl"

    def frozen(self, exp):
        return self.snapshots / exp / "frozen.json"

    def plan_input(self, name, n):
        return self.plans / name / ("%d.input.json" % int(n))


class _FileLock(object):
    """Non-blocking exclusive lock: one tick at a time across processes (the
    dashboard's thread and a hand-run CLI tick)."""

    def __init__(self, path):
        self.path, self.fh, self.got = Path(path), None, False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(str(self.path), "a")
        if fcntl is None:
            self.got = True
            return True
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.got = True
        except OSError:
            self.got = False
        return self.got

    def __exit__(self, *exc):
        try:
            if self.got and fcntl is not None:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
        return False


class _SshBudget(object):
    """slurm_sh allowed once per tick (Bridges-2 login throttling). A second call
    is refused without touching the network (executor.NotSent: the executor
    records it as a call that never reached the cluster); the ticker's order
    never makes one."""

    def __init__(self, fn, limit=1):
        self.fn, self.limit, self.calls, self.refused = fn, limit, 0, 0

    def left(self):
        return callable(self.fn) and self.calls < self.limit

    def __call__(self, script, timeout=60):
        if self.calls >= self.limit:
            self.refused += 1
            raise X.NotSent("one ssh per tick: this tick already made its call")
        self.calls += 1
        try:
            return self.fn(script, timeout)
        except TypeError:        # an older hook without a timeout parameter
            return self.fn(script)


# --- configuration ------------------------------------------------------------------
def _local_cfg_hooks(path):
    """Load and atomic save of a scheduler config file, as round_scheduler does."""
    path = Path(os.path.expanduser(str(path)))

    def load():
        if not path.exists():
            return {"domains": {}}
        c = json.loads(path.read_text())
        if not isinstance(c, dict):
            raise ValueError("%s is not a JSON object" % path)
        return c

    def save(c):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(c, indent=1))
        try:
            os.chmod(str(tmp), 0o600)
        except OSError:
            pass
        os.replace(str(tmp), str(path))
    return load, save


def default_cfg_hooks(path=None):
    """(load, save, lock) of ~/.round_scheduler.json: the scheduler's own `_cfg`,
    `_save_cfg` and `_LOCK` when it can be imported (the lab, where the
    dashboard runs), else the same file handled the same way (no lock). The
    lock is held around every read-modify-write of a campaign's block, so a
    campaign write and the scheduler's per-domain state write (made under the
    same lock) cannot drop each other's change. A load that fell back because
    the file is unreadable raises instead of returning the scheduler's default,
    so a campaign write can never replace a config it could not read."""
    if path is None:
        try:
            from .. import round_scheduler as RS
        except Exception:
            return _local_cfg_hooks("~/.round_scheduler.json")

        def load():
            c = RS._cfg()
            if not RS._CFG_READ.get("ok", True):
                raise ValueError("the scheduler config is unreadable: %s" % RS._CFG_READ.get("error"))
            return c
        return load, RS._save_cfg, RS._LOCK
    return _local_cfg_hooks(path)


def campaign_config(raw, name):
    """A campaign's config with every default filled in, and its name."""
    c = copy.deepcopy(DEFAULT_CONFIG)
    raw = raw if isinstance(raw, dict) else {}
    for k, v in raw.items():
        c[k] = copy.deepcopy(v)
    brain = dict(DEFAULT_CONFIG["brain"])
    if isinstance(raw.get("brain"), dict):
        brain.update(raw["brain"])
    c["brain"] = brain
    c["name"] = name
    return c


def check_goal(goal):
    """The normalised goal, or ValueError. None is 'no goal': the campaign runs
    until nothing is left to propose."""
    if goal is None:
        return None
    if not isinstance(goal, dict):
        raise ValueError("a goal is an object, got %r" % (goal,))
    kind = goal.get("kind")
    if kind == "diagnosis":
        did, dname = goal.get("id"), goal.get("name")
        if did not in DG.NAMES:
            raise ValueError("diagnosis %r is not one diagnose.py knows" % (did,))
        if not isinstance(dname, str) or not re.match(r"^[a-z_]{1,64}$", dname):
            raise ValueError("a diagnosis goal names the diagnosis outcome, e.g. decision_slot_ready")
        return {"kind": "diagnosis", "id": did, "name": dname}
    if kind == "exp_done":
        exp = goal.get("exp")
        if not NAME_RE.match(str(exp or "")):
            raise ValueError("exp_done needs an experiment name, got %r" % (exp,))
        return {"kind": "exp_done", "exp": exp}
    raise ValueError("goal kind must be 'diagnosis' or 'exp_done', got %r" % (kind,))


def _blank_state(name, cfg):
    exps = [e for e in (cfg.get("exps") or []) if NAME_RE.match(str(e))]
    cur = cfg.get("current_exp") or (exps[-1] if exps else None)
    if cur and cur not in exps:
        exps.append(cur)
    return {"format": STATE_FORMAT, "name": name, "phase": "RUN", "exp": cur, "exps": exps,
            "ticks": 0, "item": None, "fails": 0, "errors": 0, "paused": None, "card": None,
            "cards": [], "ledger": {}, "history": {}, "frozen": {}, "snap_hash": {}, "stale": {},
            "diagnoses": [], "health": [], "diagnosed_hash": None, "health_hash": None,
            "prospective": {}, "rules_version": None, "reported": {}, "launched": {}, "building": None,
            "build_failed": {}, "wait_jobs": [], "refusals": [], "declined": [], "attempts": {},
            "adopt_skip": [], "superseded": [], "brain": None, "brain_n": 0, "da": None,
            "funnel_sync_due": False,
            "snapshot_failures": 0, "transport_failures": 0,
            "notes": {}, "last_snapshot_utc": None, "last_tick_utc": None, "updated_utc": None}


def load_state(paths, name):
    st = _read_json(paths.state(name))
    if not isinstance(st, dict) or st.get("format") != STATE_FORMAT:
        return None
    return st


def save_state(paths, name, st):
    _write_json(paths.state(name), st)


def _append_ledger(paths, rec):
    paths.ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(str(paths.ledger), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def ledger_entry(name, event, utc, phase=None, exp=None, decided_by=AUTO, trigger=None,
                 parent_exp=None, child_exp=None, approval_id=None, job_ids=None, **detail):
    """One campaign-ledger record: every entry carries the provenance fields."""
    rec = {"utc": utc, "campaign": name, "event": event, "phase": phase, "exp": exp,
           "decided_by": decided_by, "trigger": list(trigger or []), "parent_exp": parent_exp,
           "child_exp": child_exp, "approval_id": approval_id, "job_ids": list(job_ids or [])}
    rec.update(detail)
    return rec


CONFIG_LOCK_WAIT_S = 120


def _update_config(cfg_hooks, name, fn):
    """Read-modify-write one campaign's block, under the scheduler's lock when
    the hooks carry one. Raises ValueError when the file cannot be read or the
    lock is not free within CONFIG_LOCK_WAIT_S (nothing is written)."""
    load, save = cfg_hooks[0], cfg_hooks[1]
    lock = cfg_hooks[2] if len(cfg_hooks) > 2 else None
    if lock is not None and not lock.acquire(timeout=CONFIG_LOCK_WAIT_S):
        raise ValueError("the scheduler config lock was busy for %d s; nothing was written"
                         % CONFIG_LOCK_WAIT_S)
    try:
        c = load()
        if not isinstance(c, dict):
            raise ValueError("the scheduler config is not an object")
        c.setdefault("domains", {})
        camps = c.setdefault(CONFIG_KEY, {})
        if not isinstance(camps, dict):
            raise ValueError("the config's %r block is not an object" % CONFIG_KEY)
        cur = camps.get(name) if isinstance(camps.get(name), dict) else {}
        new = fn(copy.deepcopy(cur))
        camps[name] = new
        save(c)
        return new
    finally:
        if lock is not None:
            lock.release()


def configure(name, by, enable=None, exps=None, current=None, autonomy=None, envelope_su=None,
              daily_cap_su=None, brain=None, brain_model=None, cfg_hooks=None, lab_repo=None,
              clock=None):
    """Create or change a campaign as person `by` (the admin route and the CLI).

    `enable=True` enables it and clears a pause (config and state); a campaign
    that had completed goes back to RUN on its current experiment. `autonomy`
    'envelope' records `by` as the person whose envelope the autopilot's R3
    builds draw on (executor: autonomy_granted_by). Returns the config."""
    if not NAME_RE.match(str(name or "")):
        raise ValueError("campaign name %r does not match %s" % (name, NAME_RE.pattern))
    if not HUMAN_RE.match(str(by or "")):
        raise ValueError("a campaign is configured by a person (human:<email>), not %r" % (by,))
    if exps is not None:
        bad = [e for e in exps if not NAME_RE.match(str(e))]
        if bad:
            raise ValueError("experiment names %r are not valid" % bad)
    if current is not None and not NAME_RE.match(str(current)):
        raise ValueError("experiment name %r is not valid" % (current,))
    if autonomy not in (None, "off", "envelope"):
        raise ValueError("autonomy is 'off' or 'envelope', not %r" % (autonomy,))
    for label, v in (("envelope_su", envelope_su), ("daily_cap_su", daily_cap_su)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            raise ValueError("%s must be a number >= 0, got %r" % (label, v))
    if brain_model is not None and not MODEL_RE.match(str(brain_model)):
        raise ValueError("brain model %r is not a model tag" % (brain_model,))
    cfg_hooks = cfg_hooks or default_cfg_hooks()
    now = (clock or time.time)()
    utc = _utc(now)

    def change(c):
        if exps is not None:
            c["exps"] = list(dict.fromkeys(str(e) for e in exps))
        if current is not None:
            c["current_exp"] = str(current)
            # A request the ticker applies once (see _Run._switched), so a tick
            # running at this moment cannot write the old experiment back.
            c["switch"] = {"exp": str(current), "utc": utc, "by": by}
            lst = c.setdefault("exps", [])
            if current not in lst:
                lst.append(str(current))
        if autonomy is not None:
            if autonomy == "envelope":
                try:
                    check_goal(c.get("goal"))
                except ValueError as e:
                    # The goal is where the campaign stops; one the ticker cannot
                    # check would let envelope builds run past it.
                    raise ValueError("autonomy 'envelope' is refused while the campaign's goal is "
                                     "one the ticker cannot check (%s); set a goal first" % e)
            c["autonomy"] = autonomy
            if autonomy == "envelope":
                c["autonomy_granted_by"] = by
        if envelope_su is not None:
            c["envelope_su"] = float(envelope_su)
        if daily_cap_su is not None:
            c["daily_cap_su"] = float(daily_cap_su)
        if brain is not None or brain_model is not None:
            b = dict(c.get("brain") or {})
            if brain is not None:
                b["enabled"] = bool(brain)
            if brain_model is not None:
                b["model"] = brain_model
            c["brain"] = b
        if enable is not None:
            c["enabled"] = bool(enable)
            if enable:
                c["paused_reason"] = None
                c["resumed_utc"] = utc
        c.setdefault("enabled", False)
        c["updated_by"], c["updated_utc"] = by, utc
        return c
    new = _update_config(cfg_hooks, name, change)
    paths = Paths(lab_repo)
    full = campaign_config(new, name)
    st = load_state(paths, name) or _blank_state(name, full)
    if exps is not None:
        for e in full.get("exps") or []:
            if e not in st["exps"]:
                st["exps"].append(e)
    if current is not None:
        if current != st.get("exp"):
            if current not in st["exps"]:
                st["exps"].append(current)
            st.update(exp=current, phase="RUN", item=None, building=None, card=None)
        st["switch_applied"] = (new.get("switch") or {}).get("utc")
    if enable:
        st["paused"] = None
        st["fails"], st["errors"] = 0, 0
        st["frozen"] = {}
        if st.get("phase") == "COMPLETE":
            st.update(phase="RUN", card=None)
        elif (st.get("card") or {}).get("kind") == "paused":
            st["card"] = None
    st["updated_utc"] = utc
    save_state(paths, name, st)
    _append_ledger(paths, ledger_entry(
        name, "configured", utc, st.get("phase"), st.get("exp"), decided_by=by,
        config={k: full.get(k) for k in ("enabled", "goal", "exps", "current_exp", "autonomy",
                                         "autonomy_granted_by", "envelope_su", "daily_cap_su",
                                         "brain")}))
    return full


def pause(name, reason, by, cfg_hooks=None, lab_repo=None, clock=None):
    """Pause a campaign as person `by`: nothing runs until it is enabled again."""
    if not NAME_RE.match(str(name or "")):
        raise ValueError("campaign name %r is not valid" % (name,))
    if not HUMAN_RE.match(str(by or "")):
        raise ValueError("a campaign is paused by a person (human:<email>), not %r" % (by,))
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("a pause needs a reason")
    cfg_hooks = cfg_hooks or default_cfg_hooks()
    utc = _utc((clock or time.time)())
    why = "paused by %s: %s" % (by, reason)

    def change(c):
        c.update(enabled=False, paused_reason=why, updated_by=by, updated_utc=utc)
        return c
    new = _update_config(cfg_hooks, name, change)
    paths = Paths(lab_repo)
    st = load_state(paths, name) or _blank_state(name, campaign_config(new, name))
    st["paused"] = {"reason": why, "utc": utc, "by": by, "config_written": True}
    st["card"] = {"kind": "paused", "title": "Paused by a person", "detail": why, "utc": utc}
    save_state(paths, name, st)
    _append_ledger(paths, ledger_entry(name, "paused", utc, st.get("phase"), st.get("exp"),
                                       decided_by=by, reason=why))
    return campaign_config(new, name)


def set_goal(name, goal, by, cfg_hooks=None, lab_repo=None, clock=None):
    """Set (or clear, goal None) the campaign's goal as person `by`."""
    if not NAME_RE.match(str(name or "")):
        raise ValueError("campaign name %r is not valid" % (name,))
    if not HUMAN_RE.match(str(by or "")):
        raise ValueError("a goal is set by a person (human:<email>), not %r" % (by,))
    goal = check_goal(goal)
    cfg_hooks = cfg_hooks or default_cfg_hooks()
    utc = _utc((clock or time.time)())

    def change(c):
        c.update(goal=goal, updated_by=by, updated_utc=utc)
        c.setdefault("enabled", False)
        return c
    new = _update_config(cfg_hooks, name, change)
    paths = Paths(lab_repo)
    st = load_state(paths, name)
    _append_ledger(paths, ledger_entry(name, "goal_set", utc, (st or {}).get("phase"),
                                       (st or {}).get("exp"), decided_by=by, goal=goal))
    return campaign_config(new, name)


# --- the tick -------------------------------------------------------------------------
def _resources_fn(resources, db):
    if resources is not None:
        return resources

    def probe():
        mod = db
        if mod is None:
            try:
                from .. import db as mod
            except Exception:
                mod = None
        ok = None
        if mod is not None:
            try:
                ok = bool(mod.available())
            except Exception:
                ok = None
        return {"mongo_ok": ok} if ok is not None else {}
    return probe


def tick(slurm_sh=None, cfg_hooks=None, log=None, db=None, log_action=None, clock=None,
         lab_repo=None, names=None, resources=None, domain_budget=None, preamble=None):
    """One pass over every enabled campaign: at most one ssh through `slurm_sh`.

    Never raises: a campaign whose step raised is recorded (and paused after
    ERRORS_TO_PAUSE in a row), and a failure outside any campaign is returned.
    `resources` (dict or callable) overrides the Mongo probe through `db`;
    `names` limits the pass to those campaigns."""
    log = log or _Log()
    if not _TICK_LOCK.acquire(False):
        return {"ok": False, "skipped": "another campaign tick is running in this process"}
    try:
        paths = Paths(lab_repo)
        with _FileLock(paths.lock) as got:
            if not got:
                return {"ok": False, "skipped": "another process holds the campaign tick lock"}
            return _tick_all(paths, slurm_sh, cfg_hooks or default_cfg_hooks(), log, db,
                             log_action, clock or time.time, names, resources, domain_budget,
                             preamble)
    except Exception as e:
        try:
            log.warning("[inc-campaign] tick failed: %s: %s" % (type(e).__name__, e))
        except Exception:
            pass
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, _short(e))}
    finally:
        _TICK_LOCK.release()


def _tick_all(paths, slurm_sh, cfg_hooks, log, db, log_action, clock, names, resources,
              domain_budget, preamble):
    cfg = cfg_hooks[0]()
    camps = cfg.get(CONFIG_KEY) if isinstance(cfg, dict) else None
    camps = camps if isinstance(camps, dict) else {}
    order = sorted(n for n in camps if NAME_RE.match(str(n)))
    if names:
        order = [n for n in order if n in set(names)]
    if len(order) > 1:               # the ssh slot goes round: rotate the first campaign
        k = int(clock() // (EVERY_N_TICKS * 120)) % len(order)
        order = order[k:] + order[:k]
    ssh = _SshBudget(slurm_sh)
    res_fn = _resources_fn(resources, db)
    out = {}
    for name in order:
        run = _Run(name, camps.get(name), paths, ssh, clock, log, cfg_hooks, res_fn,
                   domain_budget, log_action, preamble)
        out[name] = run.go()
    try:
        write_status(paths, cfg_hooks, clock)
    except Exception as e:
        log.warning("[inc-campaign] status not written: %s" % e)
    return {"ok": True, "campaigns": out, "ssh_calls": ssh.calls, "ssh_refused": ssh.refused}


class _Run(object):
    """One campaign's step in one tick."""

    def __init__(self, name, raw, paths, ssh, clock, log, cfg_hooks, resources, domain_budget,
                 log_action, preamble):
        self.name, self.paths, self.ssh, self.clock, self.log = name, paths, ssh, clock, log
        self.cfg_hooks, self.log_action = cfg_hooks, log_action
        self.cfg = campaign_config(raw, name)
        self.now = clock()
        self.utc = _utc(self.now)
        self.st = None
        self.ev = None
        self.status = {}
        self.partial = False
        self.xctx = X.Context(slurm_sh=ssh, resources=resources, domain_budget=domain_budget,
                              clock=clock, lab_repo=paths.lab_repo, preamble=preamble,
                              diagnoses=lambda _name: self._hook_diagnoses())
        self.xctx.local_hooks.update(self._lab_hooks())

    # ---- plumbing
    def _goal_error(self):
        """"" when the configured goal is one check_goal accepts (or none), else why not."""
        try:
            check_goal(self.cfg.get("goal"))
        except ValueError as e:
            return str(e)
        return ""

    @property
    def camp(self):
        """The campaign as the executor reads it. Autonomy is off while the goal
        is one the ticker cannot check: the goal is where the campaign stops,
        so envelope builds would run past a stop it cannot see."""
        c = self.cfg
        autonomy = c.get("autonomy") or "off"
        if autonomy == "envelope" and self._goal_error():
            autonomy = "off"
        return {"name": self.name, "autonomy": autonomy,
                "autonomy_granted_by": c.get("autonomy_granted_by"),
                "envelope_su": c.get("envelope_su"), "daily_cap_su": c.get("daily_cap_su"),
                "paused_reason": c.get("paused_reason") or ((self.st or {}).get("paused") or {})
                .get("reason")}

    def _hook_diagnoses(self):
        """The executor's `diagnoses` hook: the fired diagnoses the item was proposed
        from, with the latest health diagnoses over them (a stop-loss that fired
        since stops a self-approval)."""
        st = self.st or {}
        base = list(((st.get("item") or {}).get("diagnoses")) or st.get("diagnoses") or [])
        health = {d.get("id"): d for d in st.get("health") or []}
        out = [d for d in base if d.get("id") not in health] + list(health.values())
        return [d for d in out if isinstance(d, dict) and d.get("fired")]

    def _ledger(self, event, **kw):
        st = self.st or {}
        kw.setdefault("phase", st.get("phase"))
        kw.setdefault("exp", st.get("exp"))
        rec = ledger_entry(self.name, event, self.utc, **kw)
        rec["tick"] = st.get("ticks")
        try:
            _append_ledger(self.paths, rec)
        except OSError as e:
            self.log.warning("[inc-campaign] %s: ledger not written: %s" % (self.name, e))
        return rec

    def _once(self, key, value, event, **kw):
        """A ledger entry only when `value` differs from the last one noted under `key`."""
        notes = self.st.setdefault("notes", {})
        h = _sha(value)
        if notes.get(key) == h:
            return None
        notes[key] = h
        return self._ledger(event, **kw)

    def _trigger(self, p, diags=None):
        """[{id, name, exp, cites}] of the proposal's trigger diagnoses."""
        by = {d.get("id"): d for d in (diags or []) if isinstance(d, dict)}
        out = []
        for t in (p or {}).get("trigger") or []:
            d = by.get(t) or {}
            out.append({"id": t, "name": d.get("name"), "exp": d.get("exp"),
                        "cites": list(d.get("cites") or [])})
        return out

    def _card(self, kind, title, detail=""):
        self.st["card"] = {"kind": kind, "title": title, "detail": detail, "utc": self.utc}

    def _paused_reason(self):
        if not self.cfg.get("enabled"):
            return self.cfg.get("paused_reason") or "not enabled"
        return self.cfg.get("paused_reason") or ((self.st.get("paused") or {}).get("reason"))

    def _switched(self):
        """A person set the current experiment (config "switch"): applied once, here
        or by configure() itself, whichever comes first."""
        st = self.st
        sw = self.cfg.get("switch") if isinstance(self.cfg.get("switch"), dict) else None
        if not sw or not NAME_RE.match(str(sw.get("exp") or "")) or st.get("switch_applied") == sw.get("utc"):
            return
        st["switch_applied"] = sw.get("utc")
        if sw["exp"] == st.get("exp"):
            return
        if sw["exp"] not in st["exps"]:
            st["exps"].append(sw["exp"])
        old = st.get("exp")
        st.update(exp=sw["exp"], phase="RUN", item=None, building=None, card=None)
        self._ledger("switched", decided_by=sw.get("by") or "human", parent_exp=old,
                     reasons=["a person set the current experiment to %s at %s" % (sw["exp"], sw.get("utc"))])

    def _resumed(self):
        """A person enabled the campaign after the state's pause was set: the pause
        is over, even when a tick running at that moment wrote its state back.

        Read off the config: enabled with no paused_reason, and either a
        `resumed_utc` (configure) not before the pause, or a pause that was
        written into the config (`config_written`) -- the ticker never writes
        enabled back to true, so an enabled config after such a pause is a
        person's resume however it was written. A pause the config never
        received (its write failed) is lifted only by a resumed_utc."""
        st = self.st
        p = st.get("paused") or {}
        r = self.cfg.get("resumed_utc")
        if not p or not self.cfg.get("enabled") or self.cfg.get("paused_reason"):
            return
        by_stamp = bool(r) and str(r) >= str(p.get("utc") or "")
        if not (by_stamp or p.get("config_written")):
            return
        st["paused"] = None
        if (st.get("card") or {}).get("kind") == "paused":
            st["card"] = None
        self._ledger("resumed", decided_by=self.cfg.get("updated_by") or "human",
                     reasons=["enabled at %s after the pause of %s" % (r, p.get("utc")) if by_stamp else
                              "the config was enabled again (paused_reason cleared) after the pause of "
                              "%s was written to it" % p.get("utc")])

    def _pause(self, reason, card_detail=""):
        """Stop-loss: the campaign stops, on disk (config and state), with a card."""
        st = self.st
        st["paused"] = {"reason": reason, "utc": self.utc, "by": AUTO, "config_written": False}
        self._card("paused", "Paused: " + _short(reason, 160), card_detail or reason)

        def change(c):
            c.update(enabled=False, paused_reason=reason)
            return c
        try:
            _update_config(self.cfg_hooks, self.name, change)
            self.cfg.update(enabled=False, paused_reason=reason)
            st["paused"]["config_written"] = True
        except Exception as e:
            self.log.warning("[inc-campaign] %s: pause not written to the config (%s); the "
                             "state file holds it" % (self.name, e))
        self._ledger("paused", reason=reason)
        self.log.error("[inc-campaign] %s PAUSED: %s" % (self.name, reason))

    def _stop_loss_fails(self):
        if self.st.get("fails", 0) >= MAX_FAILED_STEPS:
            self._pause("stop-loss: %d consecutive failed campaign steps (last: %s)"
                        % (self.st["fails"], _short((self.st.get("refusals") or [{}])[-1]
                                                   .get("message") or "see the ledger", 200)))

    def _refresh_waiting_card(self):
        """COMPLETE waiting on approvals (card approval_ids): once a person has
        decided every one of them and none is left to adopt, the card says so
        instead of asking for approvals that no longer wait."""
        st = self.st
        card = st.get("card") or {}
        ids = card.get("approval_ids")
        if st.get("phase") != "COMPLETE" or card.get("kind") != "approval" or not ids:
            return
        items = AP.state(M.DOMAIN, root=self.xctx.approvals_root)
        if any((items.get(i) or {}).get("status") == "pending" for i in ids):
            return
        if any((items.get(i) or {}).get("status") == "approved"
               and (items.get(i) or {}).get("execution") is None
               and i not in (st.get("adopt_skip") or []) for i in ids):
            return
        self._complete("residual", "Nothing left the autopilot or the brain may propose",
                       "a person decided the brain's proposals %s; a person decides the next step"
                       % ", ".join(ids))

    def _complete(self, kind, title, detail):
        """COMPLETE: a human card, never a silent idle. A negative claim still
        open or challenged is named as not concluded (contract 8.6)."""
        st = self.st
        detail = str(detail or "") + self._complete_claims_note()
        st["phase"] = "COMPLETE"
        st["item"] = None
        self._card(kind, title, detail)
        self._ledger("complete", kind=kind, title=title, detail=detail)

    # ---- entry
    def go(self):
        paths = self.paths
        st0 = load_state(paths, self.name)
        if st0 is None:
            if not self.cfg.get("enabled"):
                return {"enabled": False}
            st0 = _blank_state(self.name, self.cfg)
        work = copy.deepcopy(st0)
        self.st = work
        try:
            out = self._step()
            work["errors"] = 0
            work["last_error"] = None
            work["last_tick_utc"] = self.utc
            work["updated_utc"] = self.utc
            save_state(paths, self.name, work)
            self._mirror_config()
            return out
        except Exception as e:
            self.st = st0
            st0["errors"] = int(st0.get("errors") or 0) + 1
            st0["last_error"] = {"utc": self.utc, "error": "%s: %s" % (type(e).__name__, _short(e)),
                                 "trace": _short(traceback.format_exc(), 3000)}
            st0["last_tick_utc"] = self.utc
            self._ledger("error", error=st0["last_error"]["error"])
            if st0["errors"] >= ERRORS_TO_PAUSE:
                self._pause("the ticker raised on %d ticks in a row (last: %s)"
                            % (st0["errors"], st0["last_error"]["error"]))
            try:
                save_state(paths, self.name, st0)
            except Exception:
                pass
            self.log.warning("[inc-campaign] %s: tick raised %s" % (self.name, st0["last_error"]["error"]))
            return {"error": st0["last_error"]["error"], "phase": st0.get("phase")}

    def _mirror_config(self):
        """The live phase, experiment and experiments, written into the campaign's
        config block (key "state", and current_exp / exps) when they changed, so a
        reader of the config alone (the page) sees where the campaign is."""
        st = self.st
        item = st.get("item") or {}
        view = {"phase": st.get("phase"), "exp": st.get("exp"), "exps": list(st.get("exps") or []),
                "item": ({"lever": (item.get("proposal") or {}).get("lever"),
                          "status": item.get("status"), "approval_id": item.get("approval_id")}
                         if item else None),
                "card": ({"kind": st["card"].get("kind"), "title": st["card"].get("title")}
                         if st.get("card") else None),
                "paused": (st.get("paused") or {}).get("reason")}
        cur = self.cfg.get("state") if isinstance(self.cfg.get("state"), dict) else {}
        if {k: cur.get(k) for k in view} == view and self.cfg.get("current_exp") == st.get("exp") \
                and list(self.cfg.get("exps") or []) == view["exps"]:
            return

        def change(c):
            c["state"] = dict(view, updated_utc=self.utc)
            c["current_exp"] = st.get("exp")
            c["exps"] = list(view["exps"])
            return c
        try:
            _update_config(self.cfg_hooks, self.name, change)
        except Exception as e:
            self.log.warning("[inc-campaign] %s: config view not written: %s" % (self.name, e))

    def _step(self):
        st = self.st
        st["ticks"] = int(st.get("ticks") or 0) + 1
        self._switched()
        self._resumed()
        why = self._paused_reason()
        if why:
            return {"phase": st.get("phase"), "paused": why, "ssh": False}
        if not st.get("exp"):
            self._card("config", "No current experiment",
                       "Set the campaign's current experiment (enable --current EXP).")
            return {"phase": st.get("phase"), "error": "no current experiment"}
        gerr = self._goal_error()
        if gerr:
            self._once("goal_invalid", gerr, "goal_invalid", error=gerr,
                       reasons=["the goal is not one the ticker can check, so the campaign cannot "
                                "complete on it; autonomy is off until a person sets a goal"])
            if not st.get("card") or (st.get("card") or {}).get("kind") == "config":
                self._card("config", "The campaign goal cannot be checked",
                           "%s. Set the goal again (a diagnosis that fires, or an experiment that "
                           "is done); until then every R3 build waits for a person." % gerr)
        elif (st.get("card") or {}).get("title") == "The campaign goal cannot be checked":
            st["card"] = None
        self._rules_changed()
        before = self.ssh.calls
        if st.get("phase") == "COMPLETE":
            self._adopt_approved()
            self._refresh_waiting_card()
            return {"phase": st.get("phase"), "exp": st.get("exp"),
                    "ssh": self.ssh.calls > before}
        self._act()
        if self.ssh.calls == before and self.ssh.left() and self._wants_observation() \
                and not self._paused_reason():
            self._observe()
        return {"phase": st.get("phase"), "exp": st.get("exp"), "ssh": self.ssh.calls > before,
                "item": ((st.get("item") or {}).get("proposal") or {}).get("lever")}

    def _rules_changed(self):
        """A campaign that is waiting (COMPLETE, BRAIN_WAIT) is diagnosed again when
        the rules version (diagnose.rules_version) differs from the one its last
        DIAGNOSE ran under: DIAGNOSE observes, re-diagnoses the current
        experiment and writes the prospective record of the new rules before
        any proposal. Nothing already executed runs again: every lever passes
        levers.applied (the context lineage) and the declined keys, as on any
        DIAGNOSE. A campaign with an item in flight, a build or a job reaches
        DIAGNOSE on its own (RUN, REPORT, WAIT_JOB, GATE all end there), and
        _diagnose records the version then; an L2/L6 waiting at GATE (for a
        person, the envelope or the budget) is dropped as soon as the rules
        change (_drive_item, _stale_rules), and is checked against the
        current rules' record again right before it runs."""
        st = self.st
        ver = DG.rules_version()
        last = st.get("rules_version")
        if last == ver or st.get("phase") not in ("COMPLETE", "BRAIN_WAIT"):
            return
        if st.get("item") is not None or st.get("building") or st.get("wait_jobs"):
            return
        self._ledger("rules_changed", rules_version=ver, previous_rules_version=last,
                     reasons=["the rules version is %s, not %s, the one the last DIAGNOSE of %s ran under: "
                              "DIAGNOSE again (nothing executed runs twice; the prospective record of the new "
                              "rules is written before any proposal)" % (ver, last or "unrecorded", st.get("exp"))])
        if (st.get("card") or {}).get("kind") in ("residual", "goal"):
            st["card"] = None
        st["phase"] = "DIAGNOSE"

    def _wants_observation(self):
        st = self.st
        b = st.get("brain") or {}
        return st.get("phase") in OBSERVING or b.get("status") == "submitted" \
            or (st.get("da") or {}).get("status") == "submitted"

    # ---- part A: the item, an approved item, the brain submission
    def _act(self):
        st = self.st
        before = self.ssh.calls
        if st.get("item") is not None:
            if st.get("phase") != "GATE":
                st["phase"] = "GATE"
            self._drive_item()
            if self.ssh.calls > before or self._paused_reason():
                return
            item = st.get("item")
            if item is not None and item.get("status") == "filed":
                self._adopt_approved()
        elif st.get("phase") in ADOPTING:
            self._adopt_approved()
        if st.get("item") is None and st.get("phase") == "GATE" and self.ssh.calls == before \
                and not self._paused_reason():
            # GATE holds one item in flight; with none (an item that ended with no
            # phase of its own, a superseded one) the campaign looks again rather
            # than idling silently: DIAGNOSE observes and re-diagnoses.
            st["phase"] = "DIAGNOSE"
            self._ledger("gate_empty", reasons=["no item at GATE and none adopted: DIAGNOSE observes "
                                                "and diagnoses again"])
        if self.ssh.calls == before and self.ssh.left() and not self._paused_reason():
            self._brain_expire()
            b = st.get("brain") or {}
            if b.get("status") == "staged":
                self._submit_brain()
        if self.ssh.calls == before and self.ssh.left() and not self._paused_reason():
            da = st.get("da") or {}
            if da.get("status") == "staged":
                self._submit_da()
        if self.ssh.calls == before and self.ssh.left() and not self._paused_reason():
            self._funnel_sync()

    def _prior_execution(self, pid):
        """The last execution record of proposal `pid` that ran, may have run, or
        failed after it started (a restart between the executor and the state
        write is recovered from here, never run twice). A call that never
        reached the cluster (executor.never_ran) is not one: the same proposal
        is driven again."""
        if not pid:
            return None
        hit = None
        for r in X.executions(self.xctx):
            if r.get("proposal_id") == pid and r.get("status") in ("executed", "started", "failed") \
                    and not X.never_ran(r):
                hit = r
        return hit

    def _applied_by_lineage(self, p, exclude_pid=None):
        """The campaign lineage record (self.lineage()) showing lever p.lever was
        already applied to p.parent_exp, as levers.applied reads the context
        lineage (L2 and L6 together; failed, refused and cancelled records do
        not count), else None. L7 is per unit and is not checked here (the
        executor allows one automatic unblock per unit)."""
        lever = p.get("lever")
        if not lever or lever == "L7":
            return None
        lids = ("L2", "L6") if lever in ("L2", "L6") else (lever,)
        for r in self.lineage():
            if r.get("status") in LV.LINEAGE_IGNORED or r.get("lever") not in lids:
                continue
            if exclude_pid and r.get("proposal_id") == exclude_pid:
                continue
            if lever in LV.FUNNEL_LEVERS:
                # a funnel step is one per (lever, verb / part / what / policy)
                mine = {k: (p.get("params") or {}).get(k) for k in FUNNEL_KEY_PARAMS}
                if {k: (r.get("params") or {}).get(k) for k in FUNNEL_KEY_PARAMS} == mine:
                    return r
                continue
            if r.get("parent_exp") == p.get("parent_exp"):
                return r
        return None

    def _drive_item(self):
        item = self.st["item"]
        p = item["proposal"]
        prior = self._prior_execution(p.get("id"))
        if prior is not None:
            self._ledger("recovered", approval_id=prior.get("approval_id"),
                         parent_exp=p.get("parent_exp"), child_exp=p.get("child_exp"),
                         job_ids=prior.get("job_ids"), lever=p.get("lever"),
                         status=prior.get("status"), proposal_id=p.get("id"))
            return self._on_result(item, prior)
        stale = self._stale_rules(p)
        if stale:
            # The rules changed while the L2/L6 waited (for a person, the
            # envelope or the budget): its decision is frozen again first.
            return self._drop_unfrozen(item, stale)
        if not self.ssh.left():
            return None
        if item.get("status") == "filed" and item.get("approval_id") and self.camp["autonomy"] != "envelope":
            ap = AP.state(M.DOMAIN, root=self.xctx.approvals_root).get(item["approval_id"]) or {}
            if ap.get("status") == "pending":
                return None          # only a person's decision changes anything; nothing to log
        dup = self._applied_by_lineage(p, exclude_pid=p.get("id"))
        if dup is not None:
            # Applied meanwhile by another path (an adopted item, a person's
            # action on the page): a lever is applied to a parent once.
            why = ("%s was already applied to %s in this campaign (%s, approval %s, %s); the item is "
                   "dropped, not run a second time" % (p.get("lever"), p.get("parent_exp"),
                                                       dup.get("child_exp") or "no child",
                                                       dup.get("approval_id"), dup.get("status")))
            self._ledger("not_taken", lever=p.get("lever"), parent_exp=p.get("parent_exp"),
                         child_exp=p.get("child_exp"), approval_id=item.get("approval_id"),
                         proposal_id=p.get("id"), reasons=[why])
            self._card("approval", "Not run: %s on %s was already applied" % (p.get("lever"),
                                                                             p.get("parent_exp")), why)
            self.st["item"] = None
            self.st["phase"] = item.get("resume") or "DIAGNOSE"
            return None
        if p.get("lever") in ("L2", "L6"):
            # Checked again right before it runs, as an adopted item is: a
            # READY record under the current rules, the ticker's, followed.
            held = self._d4_unfrozen(p, self.ev if self.ev is not None else self._cached_evidence())
            if held:
                return self._drop_unfrozen(item, held)
        res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
        return self._on_result(item, res)

    def _on_result(self, item, res):
        s = res.get("status")
        if s == "executed":
            return self._on_executed(item, res)
        if s == "started":
            return self._on_uncertain(item, res)
        if s == "failed":
            return self._on_failed(item, res)
        if s == "filed":
            return self._on_filed(item, res)
        return self._on_refused(item, res)

    def _mirror_action(self, p, res):
        fn = self.log_action
        if not callable(fn):
            return
        try:
            jobs = list(res.get("job_ids") or [])
            fn("inc_autopilot:%s" % p.get("policy_action"),
               {"ok": res.get("status") == "executed", "jobid": jobs[0] if jobs else None,
                "job_ids": jobs, "campaign": self.name, "lever": p.get("lever"),
                "approval_id": res.get("approval_id"), "decided_by": res.get("decided_by"),
                "cmd": " ".join(p.get("argv") or []), "msg": "; ".join(res.get("reasons") or [])})
        except Exception as e:
            self.log.warning("[inc-campaign] action log failed: %s" % e)

    def _on_executed(self, item, res):
        st = self.st
        p = item["proposal"]
        action = p.get("policy_action") or ""
        child = (res.get("params") or {}).get("exp") if action.startswith("inc_build_") else None
        child = child or p.get("child_exp")
        self._ledger("executed", decided_by=res.get("decided_by") or res.get("authorized_as"),
                     trigger=self._trigger(p, item.get("diagnoses")), parent_exp=p.get("parent_exp"),
                     child_exp=child, approval_id=res.get("approval_id"), job_ids=res.get("job_ids"),
                     lever=p.get("lever"), action=action, authorized_as=res.get("authorized_as"),
                     basis=res.get("basis"), est_su=res.get("est_su"), proposal_id=p.get("id"),
                     argv=p.get("argv"), proposed_by=p.get("proposed_by"))
        self._mirror_action(p, res)
        st["fails"] = 0
        st["transport_failures"] = 0
        st["item"] = None                  # an adopted item that ran supersedes the own one
        if action.startswith("inc_build_") and child:
            self._follow_build(item, res, child)
        elif action == "inc_unblock_transient":
            st["phase"] = item.get("resume") or "RUN"
        else:
            st["wait_jobs"] = list(res.get("job_ids") or [])
            st["wait_since_utc"] = self.utc
            st["card"] = None
            st["phase"] = "WAIT_JOB"

    def _follow_build(self, item, res, child, uncertain=False):
        """RUN on the experiment a build makes: its building record is what the
        next snapshots check (squeue by job id or by its job name
        inc_build_<exp>, and its provenance record), whether the submission is
        known to have happened or its outcome is unknown (`uncertain`: no job
        id; the build is confirmed by the snapshot, or found lost)."""
        st = self.st
        p = item["proposal"]
        if child not in st["exps"]:
            st["exps"].append(child)
        st["launched"][child] = {k: copy.deepcopy(p.get(k)) for k in
                                 ("id", "lever", "policy_action", "params", "argv", "predicted",
                                  "control", "success", "proposed_by", "parent_exp",
                                  "child_exp", "trigger")}
        st["launched"][child].update(approval_id=res.get("approval_id"), key=item.get("key"),
                                     attempt=int(item.get("attempt") or 0), uncertain=bool(uncertain),
                                     since_utc=self.utc)
        st["building"] = {"exp": child, "parent": p.get("parent_exp"),
                          "job_ids": list(res.get("job_ids") or []), "since_utc": self.utc,
                          "snapshots": 0, "lost": 0, "uncertain": bool(uncertain)}
        st["build_failed"].pop(child, None)
        st["exp"] = child
        st["refusals"] = []
        st["item"] = None
        st["phase"] = "RUN"
        if uncertain:
            self._card("cluster", "The build of %s may have been submitted" % child,
                       "the submission of %s (%s) ended with no answer from the cluster (%s); the "
                       "next snapshots decide: the job in the queue or its provenance record "
                       "confirms it, none for %d snapshots means it did not happen"
                       % (child, p.get("lever"), _short("; ".join(res.get("reasons") or []), 300),
                          BUILD_LOST_SNAPSHOTS))
        else:
            st["card"] = None

    def _on_uncertain(self, item, res):
        """An outcome that is unknown: the call may have run on the cluster (it
        started and printed no outcome, the ssh timed out or dropped, or only a
        `started` record survived a restart). It is charged, and never run
        again under this proposal. A build is followed: RUN on its child, and
        the snapshots decide whether it exists. Anything else pauses for a
        person, in a phase that observes again once the campaign is enabled."""
        st = self.st
        p = item["proposal"]
        action = p.get("policy_action") or res.get("action") or ""
        child = ((res.get("params") or {}).get("exp") if action.startswith("inc_build_") else None) \
            or p.get("child_exp")
        build = action.startswith("inc_build_") and bool(child) and bool(NAME_RE.match(str(child)))
        self._ledger("uncertain", lever=p.get("lever"), approval_id=res.get("approval_id"),
                     parent_exp=p.get("parent_exp"), child_exp=child, proposal_id=p.get("id"),
                     reasons=res.get("reasons"), charged=True,
                     next=("RUN on %s: the snapshots decide" % child) if build else "paused")
        st["transport_failures"] = 0
        st["item"] = None                  # it may have run: the own item is superseded
        if build:
            return self._follow_build(item, res, child, uncertain=True)
        st["phase"] = item.get("resume") or "DIAGNOSE"
        self._pause("the outcome of %s (%s) is unknown: the call may have run on the cluster and "
                    "wrote no outcome; a person checks the cluster, then enables the campaign"
                    % (p.get("lever"), p.get("policy_action")))

    def _on_transport(self, item, res):
        """A call that never reached the cluster (ssh never connected; the
        executor's never_ran): nothing ran, nothing is charged, and an approval
        it had claimed was released. The item stays as it is, under the same
        proposal id, and is driven again on a later tick; it is not a failed
        campaign step (a card after SNAPSHOT_FAILURES_CARD in a row)."""
        st = self.st
        p = item["proposal"]
        n = st["transport_failures"] = int(st.get("transport_failures") or 0) + 1
        if n == 1 or n == SNAPSHOT_FAILURES_CARD:
            self._ledger("transport_failed", lever=p.get("lever"), approval_id=res.get("approval_id"),
                         parent_exp=p.get("parent_exp"), child_exp=p.get("child_exp"),
                         proposal_id=p.get("id"), reasons=res.get("reasons"), failures=n,
                         released=res.get("released"))
        if n >= SNAPSHOT_FAILURES_CARD:
            self._card("cluster", "The cluster was not reached for %d calls" % n,
                       "; ".join(res.get("reasons") or []))
        if item.get("adopted") and res.get("released") is False:
            # The claim could not be given back: the approval reads as used.
            st["adopt_skip"] = list(st.get("adopt_skip") or []) + [item.get("approval_id")]

    def _on_failed(self, item, res):
        st = self.st
        p = item["proposal"]
        if X.never_ran(res):
            return self._on_transport(item, res)
        why = "; ".join(res.get("reasons") or []) or "the remote verb failed"
        self._ledger("failed", decided_by=res.get("decided_by") or AUTO, lever=p.get("lever"),
                     trigger=self._trigger(p, item.get("diagnoses")), parent_exp=p.get("parent_exp"),
                     child_exp=p.get("child_exp"), approval_id=res.get("approval_id"),
                     job_ids=res.get("job_ids"), proposal_id=p.get("id"), reasons=res.get("reasons"),
                     charged=bool(res.get("charged")))
        self._mirror_action(p, res)
        if X.uncertain(res):
            return self._on_uncertain(item, res)
        st["transport_failures"] = 0
        st["attempts"][item["key"]] = int(item.get("attempt") or 0) + 1
        st["refusals"] = (list(st.get("refusals") or []) +
                          [{"builder": p.get("policy_action"), "message": _short(why, 1000)}])[-REFUSALS_KEEP:]
        st["fails"] = int(st.get("fails") or 0) + 1
        if item.get("adopted") and st.get("item") is not None:
            st["phase"] = "GATE"           # the adopted item failed; the own item is still in flight
        else:
            st["item"] = None
            st["phase"] = item.get("resume") or "DIAGNOSE"
        self._stop_loss_fails()

    def _on_filed(self, item, res):
        st = self.st
        p = item["proposal"]
        if item.get("status") != "filed" or item.get("approval_id") != res.get("approval_id"):
            item.update(status="filed", approval_id=res.get("approval_id"), filed_utc=self.utc)
            self._ledger("filed", decided_by=AUTO, lever=p.get("lever"),
                         trigger=self._trigger(p, item.get("diagnoses")),
                         parent_exp=p.get("parent_exp"), child_exp=p.get("child_exp"),
                         approval_id=res.get("approval_id"), proposal_id=p.get("id"),
                         reasons=res.get("reasons"), est_su=res.get("est_su"))
            self._card("approval", "Approval needed: %s %s" % (p.get("lever"), p.get("title") or ""),
                       "approval %s: %s" % (res.get("approval_id"),
                                            "; ".join(res.get("reasons") or [])))
        else:
            self._once("filed:%s" % item.get("approval_id"), res.get("reasons"), "waiting",
                       approval_id=res.get("approval_id"), reasons=res.get("reasons"))
        item["last_reasons"] = list(res.get("reasons") or [])
        st["phase"] = "GATE"

    def _decider(self, approval_id):
        it = AP.state(M.DOMAIN, root=self.xctx.approvals_root).get(approval_id) if approval_id else None
        return (it or {}).get("decided_by")

    def _on_refused(self, item, res):
        st = self.st
        p = item["proposal"]
        why = "; ".join(res.get("reasons") or []) or "refused"
        if "denied by" in why:
            self._ledger("denied", decided_by=self._decider(res.get("approval_id")) or "human",
                         lever=p.get("lever"), parent_exp=p.get("parent_exp"),
                         child_exp=p.get("child_exp"), approval_id=res.get("approval_id"),
                         proposal_id=p.get("id"), reasons=res.get("reasons"))
            st["declined"] = list(st.get("declined") or []) + [item["key"]]
            st["item"] = None
            st["card"] = None
            st["phase"] = item.get("resume") or "DIAGNOSE"
            return None
        if "already executed" in why or "already ran" in why:
            rec = self._prior_execution(p.get("id"))
            if rec is None and res.get("approval_id"):
                for r in X.executions(self.xctx):
                    if r.get("approval_id") == res.get("approval_id") and \
                            r.get("status") in ("executed", "started", "failed") and not X.never_ran(r):
                        rec = r
            if rec is not None:
                return self._on_result(item, rec)
        if any(t in why for t in TRANSIENT_REFUSALS):
            self._once("refused:%s" % p.get("id"), why, "waiting", lever=p.get("lever"),
                       approval_id=res.get("approval_id"), reasons=res.get("reasons"))
            return None
        if "the campaign is paused" in why:
            return None
        if "budget:" in why:
            self._ledger("refused", lever=p.get("lever"), approval_id=res.get("approval_id"),
                         proposal_id=p.get("id"), reasons=res.get("reasons"))
            return self._pause("budget: %s cannot be charged to the campaign envelope (%s); a "
                               "person raises the envelope or drops the proposal"
                               % (p.get("lever"), _short(why, 300)))
        self._ledger("refused", lever=p.get("lever"), parent_exp=p.get("parent_exp"),
                     child_exp=p.get("child_exp"), approval_id=res.get("approval_id"),
                     proposal_id=p.get("id"), reasons=res.get("reasons"))
        if item.get("adopted"):
            st["adopt_skip"] = list(st.get("adopt_skip") or []) + [item.get("approval_id")]
            return None
        st["declined"] = list(st.get("declined") or []) + [item["key"]]
        st["item"] = None
        st["fails"] = int(st.get("fails") or 0) + 1
        st["refusals"] = (list(st.get("refusals") or []) +
                          [{"builder": p.get("policy_action"), "message": _short(why, 1000)}])[-REFUSALS_KEEP:]
        st["phase"] = item.get("resume") or "DIAGNOSE"
        self._stop_loss_fails()
        return None

    def _adopt_approved(self):
        """Execute an item a person approved for this campaign (a brain proposal,
        or one a person filed), when nothing else is in flight.

        Checked first, since an approval can be stale (a card approved after
        another item applied the same lever, or a renamed duplicate): an item
        whose lever was already applied to its parent (the campaign lineage,
        and levers.applied on the last snapshot's evidence) is refused for
        good, with a ledger entry and a card. An L2/L6 built from a pilot's D4
        decision waits until that decision is frozen in a prospective record.
        The campaign's own item is superseded only by an adopted item that ran
        or may have run; it is recorded under `superseded` (its approval, if
        still pending in the queue, is never executed through the ticker
        without these checks)."""
        st = self.st
        if not self.ssh.left():
            return False
        mine = (st.get("item") or {}).get("approval_id")
        skip = set(st.get("adopt_skip") or [])
        items = [i for i in AP.awaiting_execution(M.DOMAIN, root=self.xctx.approvals_root)
                 if ((i.get("context") or {}).get("campaign") == self.name
                     and i.get("id") != mine and i.get("id") not in skip)]
        ev = self.ev if self.ev is not None else (self._cached_evidence() if items else None)
        for it in items:
            cx = it.get("context") or {}
            p = {"id": cx.get("proposal_id"), "lever": cx.get("lever"), "policy_action": it.get("action"),
                 "params": it.get("params"), "argv": cx.get("argv"), "parent_exp": cx.get("parent_exp"),
                 "child_exp": cx.get("child_exp"), "trigger": list(cx.get("trigger") or []),
                 "cites": list(cx.get("cites") or []), "risk": it.get("risk"),
                 "proposed_by": cx.get("proposed_by") or it.get("requested_by")}
            refuse, wait = self._adoption_check(p, ev)
            if refuse:
                st["adopt_skip"] = list(st.get("adopt_skip") or []) + [it.get("id")]
                self._ledger("adopt_refused", lever=p.get("lever"), parent_exp=p.get("parent_exp"),
                             child_exp=p.get("child_exp"), approval_id=it.get("id"),
                             proposal_id=p.get("id"), decided_by=it.get("decided_by") or "human",
                             reasons=[refuse])
                self._card("approval", "Approved item not run: %s on %s"
                           % (p.get("lever"), p.get("parent_exp")),
                           "approval %s: %s" % (it.get("id"), refuse))
                continue
            if wait:
                self._once("adopt_wait:%s" % it.get("id"), wait, "waiting", lever=p.get("lever"),
                           approval_id=it.get("id"), reasons=[wait])
                continue
            item = {"proposal": p, "key": self._key(p), "attempt": 0, "status": "approved",
                    "approval_id": it.get("id"), "diagnoses": list(st.get("diagnoses") or []),
                    "resume": None, "adopted": True}
            before = self.ssh.calls
            res = X.execute_approved(it["id"], self.camp, self.xctx, invoked_by=AUTO, quiet_repeat=True)
            old = st.get("item")
            if old is not None and (res.get("status") == "executed" or X.uncertain(res)):
                self._ledger("superseded", approval_id=old.get("approval_id"),
                             lever=(old.get("proposal") or {}).get("lever"),
                             parent_exp=(old.get("proposal") or {}).get("parent_exp"),
                             proposal_id=(old.get("proposal") or {}).get("id"), by_approval=it.get("id"),
                             reasons=["a person approved %s instead; the superseded item is dropped, "
                                      "and re-proposed only if a diagnosis still calls for it"
                                      % it.get("id")])
                st["superseded"] = (list(st.get("superseded") or []) +
                                    [{"approval_id": old.get("approval_id"),
                                      "proposal_id": (old.get("proposal") or {}).get("id"),
                                      "by_approval": it.get("id"), "utc": self.utc}])[-CARDS_KEEP:]
            self._on_result(item, res)
            return self.ssh.calls > before
        return False

    def _adoption_check(self, p, ev):
        """(refusal, wait) of an approved item before the ticker executes it:
        refusal "" or why it is never run; wait "" or why not yet. An L2/L6
        that does not follow D4's decision frozen under the current rules is
        refused (a person may still build it from the page, with an
        acknowledgement); one whose decision is not frozen yet waits."""
        dup = self._applied_by_lineage(p)
        if dup is not None:
            return ("%s was already applied to %s in this campaign (%s, approval %s, %s); a lever is "
                    "applied to a parent once, and a person decides whether to repeat it"
                    % (p.get("lever"), p.get("parent_exp"), dup.get("child_exp") or "no child",
                       dup.get("approval_id"), dup.get("status"))), ""
        held, differs = self._d4_guard(p, ev)
        if held and differs:
            return held, ""
        if held:
            return "", held
        if ev is not None and p.get("lever") in ("L1", "L2", "L5", "L6", "L8", "L9"):
            try:
                child, _cite = LV.applied(ev, p["lever"], p.get("parent_exp"),
                                          **self._applied_args(p, ev))
            except Exception as e:
                return "", "levers.applied could not check it (%s: %s)" % (type(e).__name__, _short(e, 200))
            if child is not None:
                return ("%s was already applied to %s: %s (levers.applied on the last snapshot); a "
                        "person decides whether to repeat it" % (p["lever"], p.get("parent_exp"), child)), ""
        return "", ""

    @staticmethod
    def _applied_args(p, ev):
        """levers.applied's detail / params for an approved item's build flags."""
        params = p.get("params") or {}
        lever = p.get("lever")
        if lever in ("L2", "L6"):
            if p.get("parent_exp") in LV._loops(ev):
                return {"params": {"replay_mode": params.get("replay_mode")}}
            recipes = params.get("recipes")
            recipes = [r for r in str(recipes).split(",") if r] if recipes else []
            return {"detail": {"replay_mode": params.get("replay_mode"), "recipes": recipes}}
        if lever == "L5":
            return {"params": {"replay_mode": params.get("replay_mode")}}
        return {}

    def _d4_unfrozen(self, p, ev, pilot=None):
        """"" or why an L2/L6 built from a pilot's D4 decision may not be proposed
        or run yet (the reason of _d4_guard, whatever its kind)."""
        return self._d4_guard(p, ev, pilot)[0]

    def _d4_guard(self, p, ev, pilot=None):
        """(why, differs) of an L2/L6 built from a pilot's D4 decision: why is ""
        when it may be proposed or run -- that decision is frozen in a READY
        prospective record under the current rules version, written by this
        ticker (its state's sha256 is the file's), and the build follows it
        (replay mode, recipes, gate flips mode) -- else why not
        (diagnose.prospective_guard, the check every L2/L6 path applies;
        contract (f), R4b). differs is True only when a READY record of the
        current rules exists and the build does not follow it: waiting does
        not change that, so an approved item of that kind is refused, not
        held. Keyed on what the build is -- a real loop whose parent is not
        a real loop takes its replay mode and recipes from a pilot's decision
        -- not on the trigger that proposed it, so an L6 from D11, a brain
        proposal and an approved item are held to it too. `pilot`: the pilot
        to look up when the proposal names no parent."""
        if p.get("lever") not in ("L2", "L6"):
            return "", False
        parent = p.get("parent_exp") or pilot
        if ev is None:
            return "no snapshot evidence yet to tell whether %s is a pilot or a real loop" % parent, False
        if parent and parent in LV._loops(ev):
            return "", False               # L2 from D1: a rebuild of a real loop, no pilot decision
        if not parent:
            return ("D4's decision on its pilot is not frozen in a prospective record yet; no real loop is "
                    "proposed or run before it is"), False
        ver = DG.rules_version()
        rec = (self.st.get("prospective") or {}).get(parent) or {}
        if not rec.get("sha256") or rec.get("rules_version") != ver:
            return ("D4's decision on %s is not frozen in a prospective record under the current rules version "
                    "%s yet%s; no real loop is proposed or run before it is"
                    % (parent, ver, " (the ticker's record is of rules version %s)" % rec.get("rules_version")
                       if rec.get("sha256") else "")), False
        # A proposal with no params at all names no build yet (the executor
        # refuses an inc_build_realloop without replay_mode and recipes).
        why, differs = DG.prospective_guard(self.paths.replay, parent, p.get("params"),
                                            sha256=rec.get("sha256"), version=ver)
        if why:
            return "%s; no real loop is proposed or run from it" % why, differs
        return "", False

    def _stale_rules(self, p):
        """"" or why an L2/L6 in flight was proposed under other rules: its
        pilot's prospective record in the state is of another rules version
        than the current one. Needs no evidence (a real loop parent has no
        record, so a rebuild of one is never stale)."""
        if p.get("lever") not in ("L2", "L6"):
            return ""
        rec = (self.st.get("prospective") or {}).get(p.get("parent_exp")) or {}
        ver = DG.rules_version()
        if rec and rec.get("rules_version") != ver:
            return ("%s on %s was proposed under rules version %s; the rules version is now %s, so D4's decision "
                    "on %s is frozen again under the new rules before a real loop is proposed or run"
                    % (p.get("lever"), p.get("parent_exp"), rec.get("rules_version") or "unrecorded", ver,
                       p.get("parent_exp")))
        return ""

    def _drop_unfrozen(self, item, why):
        """Drops the item in flight: an L2/L6 that no longer passes _d4_guard
        (the rules changed while it waited, its record is not the ticker's,
        or the build does not follow the frozen decision). DIAGNOSE observes,
        writes the current rules' record first, and proposes again only what
        the diagnoses still call for. The item's approval is never executed
        through the ticker without these checks: re-proposed unchanged, it
        is the same proposal (the same id and approval); otherwise
        _adopt_approved checks it like any approved item."""
        p = item["proposal"]
        self._ledger("not_taken", lever=p.get("lever"), parent_exp=p.get("parent_exp"),
                     child_exp=p.get("child_exp"), approval_id=item.get("approval_id"),
                     proposal_id=p.get("id"), reasons=[why])
        self._card("approval", "Not run: %s on %s waits for D4's decision under the current rules"
                   % (p.get("lever"), p.get("parent_exp")), why)
        self.st["item"] = None
        self.st["phase"] = "DIAGNOSE"
        return None

    def _cached_evidence(self):
        """Evidence from the last snapshot this campaign kept (latest_snapshot.json
        and the frozen experiments), for a check made before this tick observes;
        None when there is none. The state is left as it was."""
        payload = _read_json(self.paths.latest(self.name))
        if not isinstance(payload, dict) or payload.get("verb") != "campaign-snapshot":
            return None
        saved = (self.partial, copy.deepcopy(self.st.get("frozen")))
        try:
            status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
            ctx = self._context(status, payload)
            rec = self._composite(payload)
            return E.from_snapshot(rec, self.st["exp"], context=ctx, ledger_prefix=self._prefix(rec))
        except Exception:
            return None
        finally:
            self.partial, self.st["frozen"] = saved

    def _submit_brain(self):
        st = self.st
        b = st["brain"]
        params = {"campaign": self.name, "n": int(b["n"]), "digest_sha256": b["digest_sha256"]}
        if b.get("model"):
            params["model"] = b["model"]
        req = {"id": _sha([self.name, "plan", b["n"], b["digest_sha256"]])[:32],
               "policy_action": "inc_plan_submit", "params": params,
               "reason": "INC research brain: plan %d of %s for %s" % (b["n"], self.name, b.get("exp"))}
        res = X.submit(req, actor=AUTO, campaign=self.camp, ctx=self.xctx)
        why = "; ".join(res.get("reasons") or [])
        if res.get("status") == "executed":
            jobs = list(res.get("job_ids") or [])
            b.update(status="submitted", submitted_utc=self.utc, job_id=jobs[0] if jobs else None)
            self._ledger("brain_submitted", job_ids=jobs, n=b["n"], digest_sha256=b["digest_sha256"],
                         model=b.get("model"), est_su=res.get("est_su"))
        elif X.uncertain(res):
            # It may have been submitted: pulled like a submitted plan until its
            # deadline (the cluster side refuses a second submission of it).
            b.update(status="submitted", submitted_utc=self.utc, job_id=None, uncertain=True)
            self._ledger("brain_submitted", n=b["n"], digest_sha256=b["digest_sha256"],
                         model=b.get("model"), est_su=res.get("est_su"), uncertain=True,
                         reasons=res.get("reasons"))
        elif X.never_ran(res) or (res.get("status") == "refused"
                                  and any(t in why for t in TRANSIENT_REFUSALS)):
            self._once("brain_submit:%s" % b["n"], why, "brain_waiting", reasons=res.get("reasons"))
        else:
            b.update(status="failed", reason=_short(why or res.get("status"), 500))
            self._ledger("brain_failed", n=b["n"], reasons=res.get("reasons"), status=res.get("status"))

    # ---- part B: one snapshot, then the phase machine
    def _live_exps(self):
        st = self.st
        frozen = st.get("frozen") or {}
        failed = st.get("build_failed") or {}
        live = [e for e in st["exps"] if e not in frozen and e not in failed]
        if st["exp"] not in live:
            live.append(st["exp"])
        return live

    def _observe(self):
        st = self.st
        exp = st["exp"]
        live = self._live_exps()
        b = st.get("brain") or {}
        da = st.get("da") or {}
        pulls = []
        if b.get("status") == "submitted":
            pulls.append(("brain", {"campaign": self.name, "n": int(b["n"])}))
        if da.get("status") == "submitted":
            pulls.append(("da", {"campaign": self.name, "n": int(da["n"])}))
        pull = pulls[0][1] if len(pulls) == 1 else ([p for _k, p in pulls] if pulls else None)
        lf = {}
        for e in live:
            pos = (st.get("ledger") or {}).get(e)
            if pos and pos.get("next_line"):
                lf[e] = (int(pos["next_line"]), pos.get("through_sha256"))
        res = X.campaign_snapshot(live, advance=st.get("phase") == "RUN", report="auto",
                                  ledger_from=lf, actor=AUTO, campaign=self.camp, ctx=self.xctx,
                                  plan_pull=pull, funnel=bool(self.cfg.get("funnel")))
        if res.get("status") == "refused":
            self._once("snapshot_refused", res.get("reasons"), "snapshot_refused",
                       reasons=res.get("reasons"))
            return
        payload = (res.get("remote") or {}).get("payload")
        if not isinstance(payload, dict) or payload.get("verb") != "campaign-snapshot":
            st["snapshot_failures"] = int(st.get("snapshot_failures") or 0) + 1
            n = st["snapshot_failures"]
            if n == 1 or n == SNAPSHOT_FAILURES_CARD:
                self._ledger("snapshot_failed", reasons=res.get("reasons"), failures=n)
            if n >= SNAPSHOT_FAILURES_CARD:
                self._card("cluster", "No campaign snapshot for %d ticks" % n,
                           "; ".join(res.get("reasons") or []))
            return
        if st.get("snapshot_failures"):
            self._ledger("snapshot_recovered", failures=st["snapshot_failures"])
            if (st.get("card") or {}).get("kind") == "cluster":
                st["card"] = None
        st["snapshot_failures"] = 0
        if st.get("transport_failures"):
            st["transport_failures"] = 0
            if str((st.get("card") or {}).get("title") or "").startswith("The cluster was not reached"):
                st["card"] = None
        st["last_snapshot_utc"] = self.utc
        status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
        self.status = status
        self._mirror_ledgers(payload)
        self._note_history(status)
        ctx = self._context(status, payload)
        self._persist_snapshot(payload, ctx)
        record = self._composite(payload)
        try:
            ev = E.from_snapshot(record, exp, context=ctx, ledger_prefix=self._prefix(record))
        except (E.EvidenceError, KeyError, TypeError, ValueError) as e:
            self._once("evidence_error", str(e), "evidence_error", error=_short(e, 500))
            return
        self.ev = ev
        if self.partial:
            self._once("ledger_partial", sorted(st.get("ledger") or {}), "ledger_partial",
                       reasons=["a ledger was read only in part this tick; no decision is made on it"])
            return
        if pulls:
            got = res.get("plan_pulls") if isinstance(pull, list) else [res.get("plan_pull")]
            for (kind, _p), r in zip(pulls, got or []):
                if kind == "brain":
                    self._brain_pulled(r)
                else:
                    self._da_pulled(r)
        if st.get("phase") == "GATE" or self._paused_reason():
            return
        self._phase_machine(ev, payload, status)

    def _phase_machine(self, ev, payload, status):
        st = self.st
        self._freeze(payload, status)
        health = DG.detect(ev, only=DG.HEALTH)
        st["health"] = [d for d in health if d.get("fired")]
        self._once("health:%s" % st["exp"], [(d["id"], d.get("name"), d.get("summary"))
                                             for d in st["health"]],
                   "health", fired=[{"id": d["id"], "name": d.get("name"), "severity": d.get("severity"),
                                     "summary": d.get("summary"), "cites": d.get("cites")}
                                    for d in st["health"]], diagnoses=health)
        if self._health_stops(ev, health):
            return
        if st["phase"] == "RUN":
            prog = self._run_progress(payload, status)
            if prog != "done":
                return
            self._ledger("finished", parent_exp=(st.get("launched") or {}).get(st["exp"], {})
                         .get("parent_exp"))
            st["phase"] = "REPORT"
        if st["phase"] == "WAIT_JOB":
            if self._jobs_queued(status, st.get("wait_jobs") or []):
                return
            self._ledger("job_finished", job_ids=st.get("wait_jobs"))
            st["wait_jobs"] = []
            st["phase"] = "DIAGNOSE"
        if st["phase"] == "REPORT":
            if not self._report_ready(ev, status, payload):
                return
            self._on_report(ev)
            st["phase"] = "DIAGNOSE"
        if st["phase"] == "DIAGNOSE":
            if not self._exp_done(status):
                st["phase"] = "RUN"
                return
            self._freeze(payload, status, current_too=True)
            self._diagnose(ev)
        if st["phase"] == "BRAIN_WAIT":
            self._brain_wait_step()

    # ---- ledgers, snapshots, evidence
    def _read_mirror(self, exp):
        rows = []
        try:
            with open(str(self.paths.mirror(exp)), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    rows.append((int(r["line"]), r.get("entry")))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return rows

    def _write_mirror(self, exp, rows):
        p = self.paths.mirror(exp)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(".%s.%d.tmp" % (p.name, os.getpid()))
        with open(str(tmp), "w", encoding="utf-8") as fh:
            for ln, e in rows:
                fh.write(json.dumps({"line": ln, "entry": e}, sort_keys=True) + "\n")
        os.replace(str(tmp), str(p))

    def _mirror_ledgers(self, payload):
        """The lab's copy of each ledger, extended by the lines this read shipped.

        The read was asked from the copy's next line with the sha256 the last read
        ended on; a prefix the file no longer has comes back from line 0 flagged
        prefix_mismatch and replaces the copy. A read that is not the continuation
        of the copy, or that stopped at the byte cap, makes this tick decide
        nothing (the next read continues or starts over)."""
        st = self.st
        pos_all = st.setdefault("ledger", {})
        for exp, sub in sorted((payload.get("experiments") or {}).items()):
            snap = (sub or {}).get("snapshot") or {}
            led = ((snap.get("decision") or {}).get("ledger"))
            if not isinstance(led, dict):
                continue
            pos = pos_all.get(exp) or {"next_line": 0, "through_sha256": None}
            if led.get("error"):
                self._write_mirror(exp, [])
                pos_all[exp] = {"next_line": 0, "through_sha256": None}
                self.partial = True
                continue
            frm = int(led.get("from_line") or 0)
            rows = self._read_mirror(exp) if frm else []
            if led.get("prefix_mismatch"):
                self._ledger("ledger_rewritten", exp=exp, detail=led.get("prefix_mismatch"))
                rows = []
            elif frm and (rows is None or frm != int(pos.get("next_line") or 0)
                          or len(rows) != frm or (rows and rows[-1][0] != frm)):
                self._write_mirror(exp, [])
                pos_all[exp] = {"next_line": 0, "through_sha256": None}
                self.partial = True
                continue
            rows = [r for r in (rows or []) if r[0] <= frm]
            rows += [(int(it["line"]), it.get("entry")) for it in led.get("entries") or []
                     if isinstance(it, dict) and "line" in it]
            self._write_mirror(exp, rows)
            pos_all[exp] = {"next_line": int(led.get("next_line") or len(rows)),
                            "through_sha256": led.get("through_sha256"),
                            "n_lines": led.get("n_lines"), "complete": bool(led.get("complete", True))}
            if not led.get("complete", True):
                self.partial = True

    def _persist_snapshot(self, payload, ctx):
        """The campaign's last record, and the snapshot history the page reads:
        SNAPSHOT_DIR/<exp>/<utc>.json, a {"record": <campaign-snapshot record of that
        experiment>, "context": <the ticker's context>, ...} wrapper, written when the
        experiment's record changed in substance (every read that shipped ledger
        lines, a rewritten ledger, done, a block, a new report or definition), and
        SNAPSHOT_DIR/step1/<utc>.json when Step 1 changed."""
        st = self.st
        _write_json(self.paths.latest(self.name), payload)
        seen = st.setdefault("snap_hash", {})
        stamp = self.utc.replace(":", "").replace("-", "")
        for exp, sub in sorted((payload.get("experiments") or {}).items()):
            snap = (sub or {}).get("snapshot") or {}
            dec = snap.get("decision") or {}
            state = (dec.get("artifacts") or {}).get("%s/state.json" % exp) or {}
            led = dec.get("ledger") or {}
            files = snap.get("files") or {}
            body = {"built": snap.get("built"), "abandoned": bool(snap.get("abandoned")),
                    "ledger": [led.get("next_line"), led.get("through_sha256"),
                               bool(led.get("prefix_mismatch"))],
                    "done": state.get("done"), "blocked": sorted((state.get("blocked") or {})),
                    "advance_error": ((sub or {}).get("advance") or {}).get("error_kind"),
                    "files": {k: (v or {}).get("sha256") for k, v in sorted(files.items())
                              if not k.endswith("/state.json") and not k.endswith("/ledger.jsonl")}}
            h = _sha(body)
            if seen.get(exp) == h:
                continue
            seen[exp] = h
            rec = {"verb": "campaign-snapshot", "ok": payload.get("ok"), "utc": payload.get("utc"),
                   "experiments": {exp: sub}, "status": payload.get("status")}
            _write_json(self.paths.snapshots / exp / ("%s.json" % stamp),
                        {"record": rec, "campaign": self.name, "tick_utc": self.utc,
                         "context": ctx if exp == st["exp"] else None})
        s1 = payload.get("step1")
        if isinstance(s1, dict):
            h = _sha({"decision": s1.get("decision"), "files": s1.get("files")})
            if seen.get("step1") != h:
                seen["step1"] = h
                _write_json(self.paths.snapshots / "step1" / ("%s.json" % stamp), s1)

    def _note_history(self, status):
        st = self.st
        hist = st.setdefault("history", {})
        for x in status.get("experiments") or []:
            if not isinstance(x, dict) or x.get("exp") not in st["exps"]:
                continue
            if x.get("generation") is None:
                continue
            h = list(hist.get(x["exp"]) or [])
            h.append({"utc": self.utc, "generation": x.get("generation")})
            hist[x["exp"]] = h[-HISTORY_KEEP:]

    @staticmethod
    def _decision_only(snap):
        """A snapshot record without its display_only part (every exam of the final
        table): what evidence.from_snapshot is given."""
        if not isinstance(snap, dict):
            return snap
        return {k: v for k, v in snap.items() if k != "display_only"}

    def _composite(self, payload):
        st = self.st
        exps = {}
        for exp, sub in (payload.get("experiments") or {}).items():
            snap = (sub or {}).get("snapshot")
            if isinstance(snap, dict) and snap.get("built", True) is not False:
                exps[exp] = {"snapshot": self._decision_only(snap)}
        for exp in sorted(st.get("frozen") or {}):
            if exp in exps:
                continue
            fz = _read_json(self.paths.frozen(exp))
            if isinstance(fz, dict) and isinstance(fz.get("snapshot"), dict):
                exps[exp] = {"snapshot": self._decision_only(fz["snapshot"])}
            else:
                st["frozen"].pop(exp, None)          # read it again next tick
                self.partial = True
        rec = {"verb": "campaign-snapshot", "utc": payload.get("utc"), "experiments": exps}
        if isinstance(payload.get("step1"), dict):
            rec["step1"] = self._decision_only(payload["step1"])
        if isinstance(payload.get("funnel"), dict) and payload["funnel"].get("verb") == "funnel-summary":
            rec["funnel"] = self._decision_only(payload["funnel"])
        return rec

    def _prefix(self, record):
        out = {}
        for exp in (record.get("experiments") or {}):
            rows = self._read_mirror(exp)
            if rows:
                out[exp] = rows
        return out

    def _budget_now(self):
        try:
            return X.budget_now(self.camp, self.xctx)
        except Exception:
            return None

    def _outcomes(self):
        out = []
        for ev in OC.read_events(self.paths.track_events):
            if ev.get("kind") == "outcome" and ev.get("child_exp") in self.st["exps"]:
                out.append({"lever": ev.get("lever"), "child_exp": ev.get("child_exp"),
                            "predicted": ev.get("predicted"), "verdict": ev.get("verdict")})
        return out

    def lineage(self):
        """Open issue 1: the campaign's lever executions as evidence.py 'lineage'
        records, from the executor's execution log. A lever that ran, or started
        and may have run, counts (levers.applied never proposes it again on that
        parent); a build still building is 'in_flight'; a run that certainly did
        not happen, or a build whose job failed, is 'failed' (levers.py ignores it)."""
        st = self.st
        building = (st.get("building") or {}).get("exp")
        failed = st.get("build_failed") or {}
        launched = st.get("launched") or {}
        out = []
        for r in X.executions(self.xctx):
            if r.get("campaign") != self.name or not r.get("lever"):
                continue
            s = r.get("status")
            if s == "executed":
                ls = "executed"
            elif s == "started":
                ls = "in_flight"
            elif s == "failed":
                ls = "uncertain" if r.get("charged") else "failed"
            else:
                continue                      # filed or refused: nothing ran
            child = r.get("child_exp")
            if str(r.get("action") or "").startswith("inc_build_"):
                child = (r.get("params") or {}).get("exp") or child
            if ls in ("executed", "in_flight", "uncertain") and child and child in failed:
                ls = "failed"                 # its build failed, or never happened (lost)
            elif ls in ("executed", "uncertain", "in_flight") and child and child == building:
                ls = "in_flight"
            elif ls == "uncertain" and child and (launched.get(child) or {}).get("built_utc"):
                ls = "executed"               # the snapshot confirmed the build
            rec = {"lever": r.get("lever"), "action": r.get("action"),
                   "parent_exp": r.get("parent_exp"), "child_exp": child, "status": ls,
                   "approval_id": r.get("approval_id"), "proposal_id": r.get("proposal_id"),
                   "ts": r.get("ts")}
            if r.get("lever") in LV.FUNNEL_LEVERS:
                rec["params"] = {k: (r.get("params") or {}).get(k) for k in FUNNEL_KEY_PARAMS
                                 if (r.get("params") or {}).get(k) is not None}
            out.append(rec)
        return out

    def _context(self, status, payload):
        """The ticker's context for evidence.py (campaign/context.json). No exam value."""
        st = self.st
        exp = st["exp"]
        ctx = {"now_utc": self.utc, "lineage": self.lineage(),
               "refusals": list(st.get("refusals") or []), "outcomes": self._outcomes()}
        sq = status.get("squeue") if isinstance(status.get("squeue"), dict) else {}
        if sq.get("ok"):
            ctx["squeue"] = [j.get("name") for j in sq.get("jobs") or [] if isinstance(j, dict)]
        if (st.get("history") or {}).get(exp):
            ctx["history"] = list(st["history"][exp])
        adv = ((payload.get("experiments") or {}).get(exp) or {}).get("advance")
        if isinstance(adv, dict) and adv.get("ok") is False and adv.get("error"):
            ctx["advance"] = {"error": _short(adv.get("error"), 2000),
                              "error_kind": adv.get("error_kind")}
        if self.cfg.get("funnel"):
            lab = self._funnel_lab_files()
            if lab:
                ctx["funnel_lab"] = lab
            try:
                ctx["funnel_lab_pull"] = X.funnel_pull_local(self.paths.lab_inc)
            except OSError as e:
                self._once("funnel_lab_pull", str(e), "funnel_lab_unreadable", reasons=[_short(e, 300)])
        reg = self._claims()
        if reg is not None:
            ctx["claims"] = reg              # evidence keeps it as campaign/claims.json
        bud = self._budget_now()
        if isinstance(bud, dict) and bud.get("envelope_su") is not None:
            ctx["budget"] = {"envelope_su": bud.get("envelope_su"),
                             "spent_su": bud.get("spent_su") or 0.0,
                             "projected_su": bud.get("committed_su") or 0.0}
        return ctx

    # ---- health and progress
    def _health_stops(self, ev, health):
        """Stop-losses read off the health diagnoses; True when the tick stops here."""
        st = self.st
        fired = {d["id"]: d for d in health if d.get("fired")}
        for did, what in (("D14", "halt: a decision path touched a non-dev exam"),
                          ("D10", "budget projection over the envelope"),
                          ("D7", "code drift: the driver refused to advance on code")):
            if did in fired:
                self._pause("%s (%s): %s" % (what, did, _short(fired[did].get("summary"), 400)))
                if did == "D7":
                    self._cards_from(["X5"], fired[did])
                return True
        if st["phase"] != "RUN":
            return False
        d5 = fired.get("D5")
        if d5 is not None:
            if "OP_PAUSE" in (d5.get("levers") or []):
                self._pause("a blocked unit is not transient (D5): %s" % _short(d5.get("summary"), 400))
                return True
            if "L7" in (d5.get("levers") or []) and st.get("item") is None:
                props = LV.propose([d5], ev).get("proposals") or []
                cand, stop = self._filter(props)
                if stop:
                    self._pause(stop)
                    return True
                if cand:
                    self._new_item(cand[0], [d5] + [d for d in st["health"] if d["id"] != "D5"],
                                   resume="RUN")
                    return True
        d6 = fired.get("D6")
        stale = st.setdefault("stale", {})
        if d6 is not None:
            stale[st["exp"]] = int(stale.get(st["exp"]) or 0) + 1
            if stale[st["exp"]] >= STALE_TO_PAUSE:
                self._pause("stale advance (D6) on %d consecutive snapshots: %s"
                            % (stale[st["exp"]], _short(d6.get("summary"), 300)))
                return True
        else:
            stale[st["exp"]] = 0
        return False

    def _status_of(self, status, exp):
        for x in status.get("experiments") or []:
            if isinstance(x, dict) and x.get("exp") == exp:
                return x
        return None

    def _exp_done(self, status):
        x = self._status_of(status, self.st["exp"])
        return bool(x and x.get("done"))

    def _jobs_queued(self, status, job_ids):
        sq = status.get("squeue") if isinstance(status.get("squeue"), dict) else {}
        if not sq.get("ok"):
            return True                    # cannot tell: keep waiting
        ids = {str(j).split("_")[0] for j in job_ids}
        return any(str(j.get("id", "")).split("_")[0] in ids for j in sq.get("jobs") or []
                   if isinstance(j, dict))

    def _run_progress(self, payload, status):
        """'building' | 'failed' | 'running' | 'done' | 'not built' | 'abandoned'."""
        st = self.st
        exp = st["exp"]
        sub = (payload.get("experiments") or {}).get(exp) or {}
        snap = sub.get("snapshot") or {}
        if snap.get("abandoned"):
            self._pause("experiment %s was cancelled (%s): a person resumes the campaign"
                        % (exp, _short(snap.get("abandoned"), 200)))
            return "abandoned"
        building = st.get("building") or {}
        if building.get("exp") == exp:
            prov = (status.get("builds") or {}).get(exp) or {}
            if not self._prov_current(prov, building):
                prov = {}                  # an earlier attempt's record (a build that failed before)
            building["snapshots"] = int(building.get("snapshots") or 0) + 1
            if snap.get("built"):
                self._ledger("built", child_exp=exp, parent_exp=building.get("parent"),
                             job_ids=building.get("job_ids"), provenance=prov,
                             uncertain=bool(building.get("uncertain")))
                st["building"] = None
                st["fails"] = 0
                if isinstance((st.get("launched") or {}).get(exp), dict):
                    st["launched"][exp]["built_utc"] = self.utc
                if building.get("uncertain") and (st.get("card") or {}).get("kind") == "cluster":
                    st["card"] = None
            elif prov.get("status") in BUILD_FAILED:
                return self._build_failed(exp, building, prov.get("status"),
                                          prov.get("refusal") or prov.get("status"))
            else:
                jobs = set(str(j).split("_")[0] for j in building.get("job_ids") or [])
                sq = status.get("squeue") if isinstance(status.get("squeue"), dict) else {}
                queued = any(str(j.get("id", "")).split("_")[0] in jobs
                             or j.get("name") == "inc_build_%s" % exp
                             for j in (sq.get("jobs") or []) if isinstance(j, dict))
                if not queued and not prov and sq.get("ok"):
                    building["lost"] = int(building.get("lost") or 0) + 1
                    if building["lost"] >= BUILD_LOST_SNAPSHOTS:
                        return self._build_failed(
                            exp, building, "lost",
                            ("no inc_build_%s job was queued and none wrote a provenance record in %d "
                             "snapshots: the submission whose outcome was unknown did not happen"
                             % (exp, BUILD_LOST_SNAPSHOTS)) if building.get("uncertain") else
                            "the build job left the queue and wrote no provenance record")
                return "building"
        if not snap.get("built"):
            self._once("not_built:%s" % exp, exp, "waiting",
                       reasons=["%s has no exp.json on the cluster yet" % exp])
            return "not built"
        x = self._status_of(status, exp)
        return "done" if x and x.get("done") else "running"

    @staticmethod
    def _prov_current(prov, building):
        """True when a provenance record (status.builds[exp]) is this build's
        attempt: its job id is the submitted one, or, with no job id to match
        (an uncertain submission, a record without one), it started after the
        submission. A record of an earlier attempt under the same name (a build
        that failed before) says nothing about this one."""
        if not prov:
            return False
        ids = {str(j).split("_")[0] for j in building.get("job_ids") or []}
        jid = str(prov.get("job_id") or "").split("_")[0]
        if ids and jid:
            return jid in ids
        return str(prov.get("started_utc") or "") >= str(building.get("since_utc") or "")

    def _build_failed(self, exp, building, kind, message):
        st = self.st
        parent = building.get("parent")
        self._ledger("build_failed", child_exp=exp, parent_exp=parent,
                     job_ids=building.get("job_ids"), kind=kind, reasons=[_short(message, 1000)])
        st["build_failed"][exp] = kind
        st["building"] = None
        launched = (st.get("launched") or {}).pop(exp, None) or {}
        if launched.get("key"):
            st["attempts"][launched["key"]] = int(launched.get("attempt") or 0) + 1
        if exp in st["exps"]:
            st["exps"].remove(exp)
        st["exp"] = parent or st["exp"]
        st["phase"] = "DIAGNOSE"
        if building.get("uncertain") and kind == "lost":
            # A submission whose outcome was unknown turned out never to have
            # happened: a transport problem, not a builder refusal or a failed
            # step. Its estimate stays charged (the executor's record may have
            # run); the lever is proposed again under a new id.
            if (st.get("card") or {}).get("kind") == "cluster":
                st["card"] = None
            return "failed"
        st["refusals"] = (list(st.get("refusals") or []) +
                          [{"builder": launched.get("policy_action") or "inc_build",
                            "message": _short(message, 1000)}])[-REFUSALS_KEEP:]
        st["fails"] = int(st.get("fails") or 0) + 1
        ref = LV.match_refusal(message)
        if ref is not None and ref.get("retry") is False and launched.get("key"):
            # A refusal the identical build meets again (levers.json 'retry': false,
            # e.g. realloop's evidenced-pool capacity or select's draw): the same
            # request is declined, never submitted a second time; DREF escalates
            # and a person chooses what to build instead.
            st["declined"] = list(st.get("declined") or []) + [launched["key"]]
            self._ledger("not_taken", lever=launched.get("lever"), parent_exp=parent, child_exp=exp,
                         reasons=["the build refused with a refusal the identical build meets again (%s): "
                                  "this request is declined for the campaign; a person decides the next build"
                                  % _short(ref.get("why") or ref.get("pattern"), 300)])
        self._stop_loss_fails()
        return "failed"

    def _report_ready(self, ev, status, payload):
        exp = self.st["exp"]
        x = self._status_of(status, exp) or {}
        rep = ev.json("%s/report.json" % exp)
        if x.get("report") == "current" and isinstance(rep, dict) and rep.get("done") is True:
            return True
        sub = ((payload.get("experiments") or {}).get(exp) or {}).get("report")
        err = (sub or {}).get("error") if isinstance(sub, dict) else None
        self._once("report_wait:%s" % exp, [x.get("report"), err], "waiting",
                   reasons=["the report of %s is %s%s" % (exp, x.get("report") or "missing",
                                                            ("; report failed: %s" % _short(err, 300))
                                                            if err else "")])
        return False

    def _on_report(self, ev):
        """A finished experiment: its spend goes into the SU ledger, and when the
        campaign launched it, its outcome is scored against the prediction."""
        st = self.st
        exp = st["exp"]
        rep = ev.json("%s/report.json" % exp) or {}
        prov = (ev.provenance.get("%s/report.json" % exp) or {})
        launched = (st.get("launched") or {}).get(exp)
        if exp not in (st.get("reported") or {}):
            if launched:
                r = B.record_report_spend(rep, self.name, base_dir=self.xctx.su_base_dir)
                spend = {k: r.get(k) for k in ("ok", "su", "reason", "unknown_units")}
                if str(launched.get("policy_action") or "").startswith("inc_build_"):
                    spend["build_job"] = self._record_build_spend(exp, rep)
            else:
                r, spend = {}, {"ok": None, "su": None,
                                "reason": "not launched by this campaign; its spend is not "
                                          "charged to the campaign envelope"}
            st["reported"][exp] = {"utc": self.utc, "ok": spend["ok"], "su": spend["su"]}
            self._ledger("reported", report_sha256=prov.get("sha256"), spend=spend,
                         approval_id=(launched or {}).get("approval_id"),
                         parent_exp=(launched or {}).get("parent_exp"))
        if launched and not launched.get("scored"):
            parent = launched.get("parent_exp")
            prep = ev.json("%s/report.json" % parent) if parent else None
            b0 = ev.json("b0_v1/report.json")
            pred = launched.get("predicted")
            if not pred:
                try:
                    pred = LV.row(launched.get("lever")).get("predicted")
                except Exception:
                    pred = None
            try:
                out = OC.record(dict(launched, predicted=pred), rep, prep, b0, child_exp=exp,
                                parent_exp=parent, events_path=self.paths.track_events,
                                summary_path=self.paths.track_summary,
                                experiments_path=self.paths.experiments)
                self._ledger("outcome", child_exp=exp, parent_exp=parent,
                             approval_id=launched.get("approval_id"), lever=launched.get("lever"),
                             metric=out.get("metric"), verdict=out.get("verdict"),
                             correct=out.get("correct"), contradicted=out.get("contradicted"),
                             delta=out.get("delta"))
            except Exception as e:
                self._ledger("outcome_failed", child_exp=exp, parent_exp=parent,
                             error="%s: %s" % (type(e).__name__, _short(e)))
            launched["scored"] = True
        st["fails"] = 0

    def _record_build_spend(self, exp, rep):
        """The SU of the job that built `exp` (budget.record_build_spend): its
        elapsed time from this build's provenance record (started_utc to the
        updated_utc of a finished attempt), capped at the job's walltime; else
        that walltime (levers.build_job_hours), an upper bound."""
        prov = (self.status.get("builds") or {}).get(exp) or {}
        try:
            cap, detail = LV.build_job_hours()
        except LV.Defer as e:
            cap, detail = None, {"error": str(e)}
        t0, t1 = _secs(prov.get("started_utc")), _secs(prov.get("updated_utc"))
        if prov.get("status") in ("advanced", "built_advance_failed") and t0 is not None \
                and t1 is not None and t1 >= t0:
            hours = (t1 - t0) / 3600.0
            if cap is not None:
                hours = min(hours, cap)
            source, measured = ("provenance of job %s, %s to %s" % (prov.get("job_id"), prov.get("started_utc"),
                                                                    prov.get("updated_utc"))), True
        elif cap is not None:
            hours, measured = cap, False
            source = "walltime %s of %s" % (detail.get("time"), detail.get("script"))
        else:
            return {"ok": False, "reason": "neither a finished provenance record nor the build job's "
                                           "walltime is known (%s)" % detail.get("error")}
        try:
            r = B.record_build_spend(exp, self.name, hours, source, ts=rep.get("done_utc"),
                                     measured=measured, base_dir=self.xctx.su_base_dir)
        except Exception as e:
            return {"ok": False, "reason": "%s: %s" % (type(e).__name__, _short(e, 300))}
        return {k: r.get(k) for k in ("ok", "su", "hours", "measured", "reason")}

    def _freeze(self, payload, status, current_too=False):
        """Keep a finished experiment's decision part on the lab and stop re-reading
        it: done, its report current, its ledger read to the end. The current
        experiment is frozen only once its report was handled (DIAGNOSE); it is
        read live while it is current either way."""
        st = self.st
        frozen = st.setdefault("frozen", {})
        for exp, sub in sorted((payload.get("experiments") or {}).items()):
            if exp in frozen or (exp == st["exp"] and not current_too):
                continue
            x = self._status_of(status, exp) or {}
            pos = (st.get("ledger") or {}).get(exp) or {}
            snap = (sub or {}).get("snapshot")
            if not (x.get("done") and x.get("report") == "current" and isinstance(snap, dict)
                    and snap.get("built")):
                continue
            if pos and (not pos.get("complete", True) or (pos.get("n_lines") is not None
                                                         and pos.get("next_line") != pos.get("n_lines"))):
                continue
            _write_json(self.paths.frozen(exp), {"exp": exp, "frozen_utc": self.utc,
                                                 "snapshot": self._decision_only(snap)})
            frozen[exp] = {"utc": self.utc}

    # ---- diagnose, prospective, propose, validate
    def _cards_from(self, levers, d):
        menu = LV.load_menu()
        for lid in levers:
            r = (menu.get("cards") or {}).get(lid)
            if r is None:
                continue
            c = M.card(lid, r.get("title", lid), [d.get("id")], d.get("cites") or [],
                       r.get("hypothesis", ""), r.get("why_menu_insufficient", ""),
                       r.get("required_change", ""), r.get("cheapest_test", ""), r.get("control", ""),
                       r.get("success_criterion", ""))
            self._record_cards([c])

    def _record_cards(self, cards, source=AUTO):
        st = self.st
        have = {c.get("key") for c in st.get("cards") or []}
        for c in cards or []:
            key = "%s:%s:%s" % (c.get("lever"), st["exp"], ",".join(sorted(c.get("trigger") or [])))
            if key in have:
                continue
            have.add(key)
            rec = {"key": key, "lever": c.get("lever"), "title": c.get("title"), "risk": "R4",
                   "exp": st["exp"], "trigger": c.get("trigger"), "utc": self.utc,
                   "proposed_by": c.get("proposed_by") or source,
                   "required_change": c.get("required_change"), "hypothesis": c.get("hypothesis")}
            st["cards"] = (list(st.get("cards") or []) + [rec])[-CARDS_KEEP:]
            self._ledger("card", decided_by=rec["proposed_by"], lever=c.get("lever"),
                         trigger=[{"id": t, "cites": c.get("cites")} for t in c.get("trigger") or []],
                         title=c.get("title"), risk="R4")

    def _goal(self, diags, status):
        goal = self.cfg.get("goal")
        try:
            goal = check_goal(goal)
        except ValueError as e:
            self._once("goal_invalid", str(e), "goal_invalid", error=str(e))
            return False, ""
        if goal is None:
            return False, ""
        if goal["kind"] == "diagnosis":
            for d in diags:
                if d.get("id") == goal["id"] and d.get("fired") and d.get("name") == goal["name"] \
                        and not (d.get("detail") or {}).get("blocked_by"):
                    return True, "%s %s fired: %s" % (d["id"], d["name"], _short(d.get("summary"), 300))
            return False, ""
        x = self._status_of(status, goal["exp"]) if status else None
        if x and x.get("done"):
            return True, "experiment %s is done" % goal["exp"]
        return False, ""

    def _prospective(self, ev, diags):
        """R4b: D4's decision on a finished pilot, written once per (pilot, rules
        version) before any proposal (diagnose.prospective_name). An earlier
        record of the pilot -- another rules version, or the unversioned name
        written before records were versioned -- is never modified or deleted:
        the new record's ledger entry names each one it supersedes, with its
        sha256, and the state keeps them under `history`."""
        st = self.st
        ver = DG.rules_version()
        by = {d["id"]: d for d in diags}
        d4 = by.get("D4") or {}
        pilot = (d4.get("detail") or {}).get("pilot") if d4.get("fired") else None
        if pilot is None and st["exp"] in DG.complete_pilots(ev):
            pilot = st["exp"]
        if pilot is None:
            return
        st.setdefault("prospective", {})
        prev = st["prospective"].get(pilot) or {}
        if prev.get("rules_version") == ver and (prev.get("sha256") or prev.get("conflict")):
            return
        rep = ev.json("%s/report.json" % pilot)
        if not isinstance(rep, dict):
            return
        out = self.paths.replay / DG.prospective_name(pilot, ver)
        history = [h for h in (prev.get("history") or []) if isinstance(h, dict)]
        if prev.get("sha256") or prev.get("conflict"):
            history.append({k: prev.get(k) for k in ("path", "sha256", "ready", "utc", "rules_version",
                                                     "conflict") if prev.get(k) is not None})
        try:
            rec = DG.prospective_d4(rep, out_path=out, defn=ev.json("%s/exp.json" % pilot),
                                    now=self.utc, version=ver)
        except ValueError as e:
            st["prospective"][pilot] = {"conflict": _short(e, 500), "utc": self.utc, "rules_version": ver,
                                        "history": history[-CARDS_KEEP:]}
            self._ledger("prospective_conflict", exp=pilot, error=_short(e, 500), rules_version=ver)
            return
        except OSError as e:
            self._once("prospective_failed:%s" % pilot, str(e), "prospective_failed", exp=pilot,
                       error=_short(e, 300))
            return
        if rec.get("pending"):
            # A v2 pilot whose check is not final has no decision yet: nothing is
            # frozen, and a later DIAGNOSE tries again (no L2 before it, _d4_unfrozen).
            self._once("prospective_pending:%s" % pilot, rec["pending"], "prospective_pending", exp=pilot,
                       reasons=["%s was decided on protocol v2 and its v2_check is %s: D4's decision is recorded "
                                "once the check is final" % (pilot, rec["pending"])])
            return
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        superseded = []
        for path, pver, old in DG.prospective_records(self.paths.replay, pilot):
            if pver == ver:
                continue
            try:
                osha = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                osha = None
            superseded.append({"path": str(path), "sha256": osha,
                               "rules_version": (old or {}).get("rules_version") or pver,
                               "ready": (old or {}).get("ready"), "outcome": (old or {}).get("outcome"),
                               "blocked_by": (old or {}).get("blocked_by")})
        on_disk = {x["path"] for x in superseded}
        for h in history:
            if h.get("path") and h["path"] not in on_disk and h.get("rules_version") != ver:
                superseded.append({"path": h["path"], "sha256": h.get("sha256"),
                                   "rules_version": h.get("rules_version"), "ready": h.get("ready"),
                                   "missing": True})
        st["prospective"][pilot] = {"path": str(out), "sha256": sha, "ready": rec.get("ready"),
                                    "utc": self.utc, "rules_version": ver,
                                    "replay_mode": rec.get("replay_mode"), "recipes": rec.get("recipes"),
                                    "gate_flips_mode": rec.get("gate_flips_mode"),
                                    "blocked_by": rec.get("blocked_by"), "history": history[-CARDS_KEEP:]}
        self._ledger("prospective_d4", exp=pilot, path=str(out), sha256=sha,
                     outcome=rec.get("outcome"), ready=rec.get("ready"),
                     replay_mode=rec.get("replay_mode"), recipes=rec.get("recipes"),
                     gate_flips_mode=rec.get("gate_flips_mode"), blocked_by=rec.get("blocked_by"),
                     rules_version=ver, rules_files=rec.get("rules_files"), superseded=superseded,
                     report_file_sha256=(ev.provenance.get("%s/report.json" % pilot) or {}).get("sha256"),
                     trigger=[{"id": "D4", "name": d4.get("name"), "exp": pilot,
                               "cites": d4.get("cites") or []}] if d4.get("fired") else [])

    def _funnel_step(self, ev, diags, prop):
        """The funnel audit's part of DIAGNOSE: DEC-1..DEC-10 logged once, D18's
        claim transitions applied, the prospective DA record scored once a
        valid audit of this Step 1 lands (outcome.record_da: H11 and the DA's
        predictions, the adversary's track record), the DA pass staged when
        due, the funnel sync queued when the lab holds files the cluster does
        not."""
        if ev.json(E.FUNNEL_LEDGER) is not None:
            self._log_decisions()
        self._apply_d18(diags)
        self._score_da(ev)
        why, ids = self._da_wanted(ev, prop)
        if why:
            self._stage_da(ev, why, ids, prop)
        if any(o.get("op") == "OP_FUNNEL_SYNC" for o in (prop or {}).get("operations") or []) \
                or (self.cfg.get("funnel") and LV.funnel_sync_needed(ev)):
            # Data movement (R0, runner 6.3) is due whenever one side holds a
            # listed file the other lacks, whichever diagnosis fires: the
            # sheets and the audit come back after D17-D19 have gone quiet.
            self.st["funnel_sync_due"] = True

    def _key(self, p):
        return _canon([p.get("lever"), p.get("parent_exp"), list(BP.proposal_key(p))])

    def _invalid(self, p):
        """"" when the executor can render the proposal as it will run it and its
        parameters are inside the policy row's bounds; else why not."""
        row = POL.describe(p.get("policy_action"))
        if not row.get("known"):
            return row.get("reason") or "no policy row"
        try:
            pol, _meta = X.resolve_params(p["policy_action"], row, p.get("params"), p.get("argv"),
                                          p.get("est_gpu_hours"))
            ok, why = X.argv_check(X.render(p["policy_action"], pol), p.get("argv"))
        except X.ExecError as e:
            return str(e)
        if not ok:
            return why
        ok, bad = POL._check_params(pol, row["param_bounds"])
        return "" if ok else "; ".join(bad)

    def _filter(self, proposals, diags=None):
        """(candidates, stop-loss reason or None), in the order levers.propose ranked
        them (by the diagnoses that triggered them, D1 first)."""
        st = self.st
        by = {d["id"]: d for d in diags or []}
        out = []
        self._declined_now = []
        for p in proposals or []:
            key = self._key(p)
            if key in (st.get("declined") or []):
                self._once("declined:%s" % key, key, "not_taken", lever=p.get("lever"),
                           reasons=["declined or refused earlier in this campaign"])
                self._declined_now.append(p)
                continue
            n = X._lever_count(self.xctx, self.name, p.get("lever"))
            if n >= X.MAX_LEVER_SUBMISSIONS:
                return [], ("stop-loss: lever %s already ran %d times in this campaign; a %s would "
                            "be more than %d" % (p.get("lever"), n, "further submission",
                                                 X.MAX_LEVER_SUBMISSIONS))
            if p.get("waits_for"):
                w = p["waits_for"]
                self._once("waits:%s:%s" % (p.get("lever"), " ".join(p.get("argv") or [])), w, "not_taken",
                           lever=p.get("lever"), reasons=["waits for %s: %s is not on the cluster (%s)"
                                                          % (w.get("lever"), w.get("cluster_file"), w.get("why"))])
                continue
            bad = self._invalid(p)
            if bad:
                self._ledger("invalid_proposal", lever=p.get("lever"), reasons=[bad],
                             parent_exp=p.get("parent_exp"), child_exp=p.get("child_exp"))
                continue
            unfrozen = self._d4_unfrozen(p, self.ev,
                                         pilot=((by.get("D4") or {}).get("detail") or {}).get("pilot"))
            if unfrozen:
                self._ledger("not_taken", lever=p.get("lever"), parent_exp=p.get("parent_exp"),
                             child_exp=p.get("child_exp"), reasons=[unfrozen])
                continue
            out.append(p)
        return out, None

    def _new_item(self, p, diags, resume=None):
        st = self.st
        key = self._key(p)
        attempt = int((st.get("attempts") or {}).get(key) or 0)
        p = copy.deepcopy(p)
        p["id"] = _sha([self.name, p.get("lever"), p.get("policy_action"), p.get("argv"),
                        p.get("params"), p.get("parent_exp"), attempt])[:32]
        p["created_utc"] = self.utc
        fired = [d for d in diags if isinstance(d, dict) and d.get("fired")]
        st["item"] = {"proposal": p, "key": key, "attempt": attempt, "status": "proposed",
                      "approval_id": None, "diagnoses": fired, "resume": resume,
                      "proposed_utc": self.utc}
        st["phase"] = "GATE"
        self._ledger("proposed", lever=p.get("lever"), action=p.get("policy_action"),
                     risk=p.get("risk"), argv=p.get("argv"), est_gpu_hours=p.get("est_gpu_hours"),
                     trigger=self._trigger(p, fired), parent_exp=p.get("parent_exp"),
                     child_exp=p.get("child_exp"), proposal_id=p["id"], attempt=attempt,
                     estimate=p.get("estimate"))

    def _diagnose(self, ev):
        st = self.st
        diags = DG.detect(ev)
        fired = [d for d in diags if d.get("fired")]
        st["diagnoses"] = fired
        st["rules_version"] = DG.rules_version()
        _write_json(self.paths.diagnoses(self.name), {"utc": self.utc, "exp": st["exp"],
                                                     "diagnoses": diags})
        self._once("diagnosed:%s" % st["exp"], [(d["id"], d.get("name"), d.get("summary"))
                                                for d in fired],
                   "diagnosed", trigger=[{"id": d["id"], "name": d.get("name"), "exp": d.get("exp"),
                                          "cites": d.get("cites")} for d in fired],
                   summaries={d["id"]: d.get("summary") for d in fired}, diagnoses=diags)
        self._prospective(ev, diags)
        met, why = self._goal(diags, self.status)
        if met:
            return self._complete("goal", "Goal met", why)
        prop = LV.propose(diags, ev)
        self._record_cards(prop.get("cards") or [])
        ops = prop.get("operations") or []
        stop = [o for o in ops if o.get("op") in ("OP_PAUSE", "OP_HALT")]
        if stop:
            o = stop[0]
            why = "; ".join(d.get("summary") or "" for d in fired if d["id"] in (o.get("trigger") or []))
            return self._pause("%s by %s on %s: %s" % (o.get("op"), ", ".join(o.get("trigger") or []),
                                                       o.get("exp"), _short(why, 400)))
        for o in ops:
            if o.get("op") == "OP_ESCALATE":
                trig = list(o.get("trigger") or [])
                why = "; ".join(d.get("summary") or "" for d in fired if d["id"] in trig)
                d13 = "D13" in trig
                self._once("escalate:%s" % st["exp"], o.get("trigger"), "escalated",
                           trigger=[{"id": t} for t in trig],
                           reasons=["a prediction was contradicted (D13): the brain and a person look"] if d13
                           else ["%s escalated to a person: %s" % (", ".join(trig), _short(why, 400))])
                self._card("escalation", "A lever's prediction was contradicted" if d13
                           else "Escalated to a person by %s" % ", ".join(trig),
                           "; ".join(d.get("summary") or "" for d in fired if d["id"] == "D13") if d13 else why)
        self._funnel_step(ev, diags, prop)
        cands, stop_reason = self._filter(prop.get("proposals") or [], diags)
        if stop_reason:
            return self._pause(stop_reason)
        if self.cfg["brain"].get("enabled") and ((st.get("brain") or {}).get("exp") != st["exp"]):
            self._stage_brain(ev, diags, cands, prop)
        if cands:
            self._new_item(cands[0], fired)
            for other in cands[1:]:
                self._ledger("not_taken", lever=other.get("lever"), parent_exp=other.get("parent_exp"),
                             child_exp=other.get("child_exp"),
                             trigger=self._trigger(other, fired),
                             reasons=["one item in flight per campaign; ranked below %s"
                                      % cands[0].get("lever")])
            return None
        b = st.get("brain") or {}
        if b.get("exp") == st["exp"] and b.get("status") in ("staged", "submitted"):
            st["phase"] = "BRAIN_WAIT"
            self._ledger("brain_wait", reasons=["nothing deterministic to propose; the brain plan "
                                                "%d is still due" % b.get("n")])
            return None
        da = st.get("da") or {}
        if da.get("status") in DA_IN_FLIGHT:
            # Contract 8.6: the DA pass runs before a COMPLETE card that carries
            # a negative claim. COMPLETE neither submits nor pulls, so the
            # campaign waits here (BRAIN_WAIT observes and submits) and
            # diagnoses again once the pass has ended.
            st["phase"] = "BRAIN_WAIT"
            st["da_wait"] = True
            self._ledger("da_wait", reasons=["nothing deterministic to propose; the devil's-advocate pass %s "
                                             "(%s) is due before the COMPLETE card" % (da.get("n"),
                                                                                      da.get("claim_ids"))])
            return None
        return self._complete_residual(prop, fired)

    def _complete_residual(self, prop, fired):
        deferred = ["%s (%s): %s" % (x.get("lever"), x.get("trigger"), _short(x.get("reason"), 200))
                    for x in prop.get("deferred") or []]
        declined = ["%s on %s" % (p.get("lever"), p.get("parent_exp"))
                    for p in getattr(self, "_declined_now", None) or []]
        detail = ("no lever can be proposed on %s. Fired: %s. Deferred: %s.%s A person decides the "
                  "next step (a new experiment, a change of goal, or an off-menu card)."
                  % (self.st["exp"], ", ".join("%s %s" % (d["id"], d.get("name")) for d in fired)
                     or "none", "; ".join(deferred) or "none",
                     (" Declined or refused earlier in this campaign, not proposed again: %s."
                      % "; ".join(declined)) if declined else ""))
        return self._complete("residual", "Nothing left the autopilot may propose", detail)

    # ---- the funnel audit: claims, decisions and the devil's advocate
    # docs/FUNNEL_AUDIT.md 8.3 (claims), 8.6 (the DA pass), 13 (DEC-1..DEC-10);
    # runner 5.5.4. The claims register is the lab's campaign/claims.json
    # (Paths.claims); every transition is a campaign ledger entry with its
    # decided_by. The DA pass is staged when D17 fires (OP_DA), when a negative
    # claim appears in the register, and before a COMPLETE card while a
    # negative claim is open; it runs as the plan job with PLAN_ROLE=adversary,
    # is pulled with the snapshot, validated (validate.validate_da), recorded
    # in the lab's funnel/prospective_da.json and pushed by the funnel sync.
    def _claims(self):
        """The claims register (funnel.claims.load), or None when there is none;
        an invalid register is noted once and read as none."""
        from ..funnel import claims as FC
        if not self.paths.claims.is_file():
            return None
        try:
            return FC.load(self.paths.claims)
        except Exception as e:
            self._once("claims_invalid", str(e), "claims_invalid", error=_short(e, 400))
            return None

    def _save_claims(self, reg):
        from ..funnel import claims as FC
        FC.save(self.paths.claims, reg)

    def _log_decisions(self):
        """DEC-1..DEC-10 (contract 13), logged once per campaign with decided_by
        human-delegated, the first time the funnel ledger is in the evidence."""
        st = self.st
        if st.get("decisions_logged"):
            return
        st["decisions_logged"] = self.utc
        self._ledger("decisions_recorded", decided_by="human-delegated",
                     decisions=list(FUNNEL_DECISIONS), source="docs/FUNNEL_AUDIT.md 13",
                     reasons=["the R4 decisions of the funnel audit, made under the project owner's standing "
                              "delegation before any platform code, sample draw or reference label; X12 (the final "
                              "claim status) is not delegated"])

    def _transition(self, reg, claim_id, to, by, reason, cites, actor_kind, proof=None):
        """One claim transition through funnel.claims (its rules refuse what is not
        allowed), saved and entered in the campaign ledger. Returns why not, or ""."""
        from ..funnel import claims as FC
        try:
            FC.transition(reg, claim_id, to, by, reason, cites, actor_kind, proof)
            self._save_claims(reg)
        except Exception as e:
            self._once("claim_refused:%s:%s" % (claim_id, to), str(e), "claim_transition_refused",
                       decided_by=by, claim_id=claim_id, to=to, reasons=[_short(e, 400)])
            return str(e)
        self._ledger("claim_transition", decided_by=by, claim_id=claim_id, to=to, actor_kind=actor_kind,
                     reasons=[reason], cites=cites)
        return ""

    def _apply_d18(self, diags):
        """D18 silent on a valid audit whose fingerprint matches proposes each
        challenged negative claim's move to tested_survives (actor autopilot);
        the cite carries the audit's sha256 (funnel.claims)."""
        d18 = next((d for d in diags if d.get("id") == "D18"), None)
        trans = ((d18 or {}).get("detail") or {}).get("claim_transitions") or []
        if not trans:
            return
        reg = self._claims()
        if reg is None:
            return
        for t in trans:
            sha = t.get("audit_sha256")
            cite = M.cite("funnel/audit_v1.json", sha, pointer="")
            proof = {"audit_sha256": sha, "valid": True, "fingerprint_match": True,
                     "d18_fired": bool((d18 or {}).get("fired") and "L13" in ((d18 or {}).get("levers") or []))}
            self._transition(reg, t.get("claim_id"), t.get("to"), AUTO,
                             "D18 is silent on the valid audit of this Step 1 (fingerprint %s...): the filters' "
                             "false negatives stay below the pre-registered bounds" % str(t.get("ledger_fingerprint"))[:12],
                             [cite], "autopilot", proof)

    def _da_claims_done(self):
        """The claims a DA pass was staged for in this campaign (answered,
        invalid or failed alike: a pass is recorded, never retried until it
        passes; contract 10 F2)."""
        st = self.st
        return set(st.get("da_claims") or []) | set((st.get("da") or {}).get("claim_ids") or [])

    def _da_wanted(self, ev, prop):
        """(why or "", [claim id]): the DA pass is due when D17 asked for it, or a
        negative claim is open that no DA pass has answered yet. One claim per
        pass: the reply (inc-da-reply/1) answers one claim_id, so a digest of
        several claims would leave the others unanswered, and open for good
        (open -> tested_survives is not a transition)."""
        reg = self._claims()
        from ..funnel import claims as FC
        neg = [c for c in FC.open_negative(reg)] if reg else []
        if not neg:
            return "", []
        done = self._da_claims_done()
        pending = sorted((c for c in neg if c["id"] not in done), key=lambda c: (len(c["id"]), c["id"]))
        if not pending:
            return "", []
        ops = [o.get("op") for o in (prop or {}).get("operations") or []]
        if "OP_DA" in ops:
            return "D17 fired on an unaudited negative conclusion", [pending[0]["id"]]
        new = [c["id"] for c in pending if c.get("status") == "open"]
        if new:
            return ("negative claim %s is open and no devil's-advocate pass has answered it" % new[0]), [new[0]]
        return "", []

    def _stage_da(self, ev, why, claim_ids, prop):
        """Build and stage the DA digest (brain_plan.build_da_digest): blind to
        the contract, the pre-registration, the R14 fixture and D17's proposals."""
        st = self.st
        da = st.get("da") or {}
        if da.get("status") in ("staged", "submitted"):
            return
        led = ev.json(E.FUNNEL_LEDGER)
        if not isinstance(led, dict):
            self._once("da_no_ledger", claim_ids, "da_waiting", reasons=["the DA pass needs the funnel ledger in the "
                                                                          "evidence (remote funnel summary)"])
            return
        n = int(st.get("brain_n") or 0) + 1
        st["brain_n"] = n
        st["da_claims"] = sorted(self._da_claims_done() | set(claim_ids))
        d17 = [p for p in (prop or {}).get("proposals") or [] if "D17" in (p.get("trigger") or [])]
        loop = DG._latest_loop(ev)[0]
        try:
            reg = self._claims() or {}
            only = dict(reg, claims=[c for c in reg.get("claims") or [] if c.get("id") in claim_ids])
            marks = BP.blind_markers(self.paths.lab_inc / "funnel" / "prereg_v1.json",
                                     self.paths.contract, self.paths.r14_fixtures, d17)
            digest = BP.build_da_digest(ev, only, led, BP.load_menu(), campaign=self.name, n=n,
                                        markers=marks, loop=loop, created_utc=self.utc)
            _write_json(self.paths.plan_input(self.name, n), digest)
        except Exception as e:
            st["da"] = {"n": n, "status": "failed", "claim_ids": claim_ids,
                        "reason": "%s: %s" % (type(e).__name__, _short(e, 300))}
            self._ledger("da_failed", n=n, reasons=[st["da"]["reason"]])
            return
        st["da"] = {"n": n, "status": "staged", "digest_sha256": digest["sha256"], "claim_ids": claim_ids,
                    "fingerprint": led.get("fingerprint"), "staged_utc": self.utc, "why": why,
                    "markers": len(marks), "blind_markers": digest.get("blind_markers"), "exp": loop}
        self._ledger("da_staged", n=n, digest_sha256=digest["sha256"], claim_ids=claim_ids, reasons=[why],
                     tokens_estimated=digest.get("tokens_estimated"), num_ctx=digest.get("num_ctx"),
                     blind_markers_checked=len(marks), dropped_artifacts=digest.get("dropped_artifacts"))

    def _submit_da(self):
        st = self.st
        da = st["da"]
        params = {"campaign": self.name, "n": int(da["n"]), "digest_sha256": da["digest_sha256"]}
        req = {"id": _sha([self.name, "da", da["n"], da["digest_sha256"]])[:32],
               "policy_action": "inc_plan_submit", "params": params,
               "reason": "devil's-advocate pass %d of %s against %s" % (da["n"], self.name, da.get("claim_ids"))}
        res = X.submit(req, actor=AUTO, campaign=self.camp, ctx=self.xctx)
        why = "; ".join(res.get("reasons") or [])
        if res.get("status") == "executed" or X.uncertain(res):
            jobs = list(res.get("job_ids") or [])
            da.update(status="submitted", submitted_utc=self.utc, job_id=jobs[0] if jobs else None,
                      uncertain=not res.get("status") == "executed")
            self._ledger("da_submitted", job_ids=jobs, n=da["n"], digest_sha256=da["digest_sha256"],
                         est_su=res.get("est_su"))
        elif X.never_ran(res) or (res.get("status") == "refused" and any(t in why for t in TRANSIENT_REFUSALS)):
            self._once("da_submit:%s" % da["n"], why, "da_waiting", reasons=res.get("reasons"))
        else:
            da.update(status="failed", reason=_short(why or res.get("status"), 500))
            self._ledger("da_failed", n=da["n"], reasons=res.get("reasons"), status=res.get("status"))

    def _planner_model(self):
        """The model the planner's last reply was resolved to (the brain plan
        this campaign merged), else the planner role's resolution now."""
        from .. import model_router as MR
        got = (self.st.get("brain") or {}).get("model_resolved")
        return got or str(MR.resolve("planner").get("model") or "")

    def _da_pulled(self, pull_res):
        st = self.st
        da = st.get("da") or {}
        if da.get("status") != "submitted":
            return
        payload = ((pull_res or {}).get("remote") or {}).get("payload") if pull_res else None
        reply, exists = None, False
        if isinstance(payload, dict) and payload.get("verb") == "plan-pull":
            exists = bool(payload.get("exists"))
            reply = payload.get("reply")
        col = BP.collect(reply if exists else None, da.get("submitted_utc"), now_utc=self.utc,
                         digest_sha256=da.get("digest_sha256"), pulled_utc=self.utc, role="adversary")
        if col["status"] == "pending":
            return
        if col["status"] != "ready":
            da.update(status=col["status"], reason=_short(col.get("reason"), 500))
            self._ledger("da_%s" % col["status"], n=da.get("n"), reasons=[col.get("reason")],
                         recorded=_short(json.dumps(reply, sort_keys=True), 2000) if reply else None)
            return
        self._merge_da(col["reply"])

    def _merge_da(self, reply):
        """Validate the DA reply, record it (funnel/prospective_da.json on the
        lab, pushed by the funnel sync), move each answered claim to
        challenged when surviving counter-arguments came from a model of
        another family, and file what they propose (tier2:adversary levers,
        R4 cards). An invalid reply is recorded as invalid, not retried."""
        from ..funnel import claims as FC
        st = self.st
        da = st["da"]
        ev = self.ev
        reg = self._claims() or FC.new_register()
        diags = DG.detect(ev)
        adv = str(reply.get("model_used") or reply.get("model") or "")
        resolved = {"adversary": adv, "planner": self._planner_model()}
        try:
            from . import corpus as CO
            corp = CO.Corpus()
        except Exception:
            corp = None
        staged = dict(reg, claims=[c for c in reg.get("claims") or [] if c.get("id") in (da.get("claim_ids") or [])])
        val = V.validate_da(reply, ev, ev.json(E.FUNNEL_LEDGER), BP.load_menu(), resolved, diagnoses=diags,
                            corpus=corp, claims=staged)
        path = self.paths.lab_inc / "funnel" / "prospective_da.json"
        if path.exists():
            self._ledger("da_not_recorded", n=da.get("n"), reasons=["%s exists: the prospective DA record is "
                                                                     "written once" % path])
        else:
            try:
                rec = dict(self._prospective_header(da), claim_ids=da.get("claim_ids"),
                           digest_sha256=da.get("digest_sha256"), blind_check=self._blind_check(da),
                           model=val["model"], reply=V.da_reply_of(reply), validation=val,
                           stage_forecast=val.get("stage_forecast"), committed_utc=self.utc,
                           campaign=self.name, n=da.get("n"))
                _write_json(path, rec)
            except Exception as e:
                self._ledger("da_not_recorded", n=da.get("n"),
                             reasons=["the prospective DA record could not be written: %s: %s"
                                      % (type(e).__name__, _short(e, 400))])
        surviving = sum(1 for c in val.get("counter_arguments") or [] if c.get("kept") and c.get("counted"))
        da.update(status="merged" if val.get("ok") else "invalid", merged_utc=self.utc,
                  surviving=surviving, moves_claims=val.get("moves_claims"),
                  invalid_reason=val.get("invalid_reason"))
        self._ledger("da_merged" if val.get("ok") else "da_invalid", decided_by=BP.actor_for("adversary/%s" % adv),
                     n=da.get("n"), counts=val.get("counts"), same_family=val["model"].get("same_family"),
                     moves_claims=val.get("moves_claims"), invalid_reason=val.get("invalid_reason"),
                     path=str(path))
        if not val.get("ok") or not val.get("moves_claims"):
            return
        cites = [c for ca in val["counter_arguments"] if ca.get("counted") for c in ca.get("evidence_cites") or []]
        proof = {"valid": True, "surviving": surviving, "model": adv, "planner_model": resolved["planner"],
                 "same_family": val["model"]["same_family"]}
        actor = BP.actor_for("adversary/%s" % adv)
        for cid in [val.get("claim_id")]:
            c = FC.get(reg, cid) if cid else None
            if c and c.get("status") in ("open", "tested_survives"):
                self._transition(reg, cid, "challenged", actor,
                                 "%d grounded counter-argument(s) of the devil's advocate survived validation"
                                 % surviving, cites[:20], "adversary", proof)
        for f in V.da_filings(val, BP.load_menu()):
            if "card" in f:
                r = (LV.load_menu().get("cards") or {}).get(f["card"]) or {}
                self._record_cards([M.card(f["card"], r.get("title", f["card"]), ["D17"], cites[:5],
                                           r.get("hypothesis", ""), r.get("why_menu_insufficient", ""),
                                           r.get("required_change", ""), r.get("cheapest_test", ""),
                                           r.get("control", ""), r.get("success_criterion", ""),
                                           proposed_by=actor)], source=actor)
            else:
                self._ledger("da_test_proposed", decided_by=actor, lever=f["lever"], params=f.get("params"),
                             counter_argument=f["counter_argument"],
                             reasons=["a surviving counter-argument names %s as its cheapest test; the "
                                      "deterministic funnel levers run it in their order" % f["lever"]])

    def _prospective_header(self, da):
        """The runner 1.2 header of funnel/prospective_da.json: the sha256 of
        the pre-registration and of the contract (contract, header: every later
        audit artifact records both), the domain config, the code and the
        staged digest and claims register it answers. Refuses (ValueError) when
        the lab's contract is not the one the pre-registration names."""
        from .. import funnel as FUN
        from ..funnel import domain as FD
        pre_path = self.paths.lab_inc / "funnel" / "prereg_v1.json"
        raw = pre_path.read_bytes()
        pre = json.loads(raw.decode("utf-8"))
        csha = hashlib.sha256(self.paths.contract.read_bytes()).hexdigest()
        if (pre.get("contract") or {}).get("sha256") != csha:
            raise ValueError("the contract %s (sha256 %s) is not the one %s names (%s)"
                             % (self.paths.contract, csha[:12], pre_path, str((pre.get("contract") or {})
                                                                               .get("sha256"))[:12]))
        prereg = {"path": str(pre_path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "core_sha256": FD.prereg_core_sha256(pre),
                  "contract": {"path": pre["contract"].get("path") or str(self.paths.contract), "sha256": csha}}
        inputs = {"digest": {"path": str(self.paths.plan_input(self.name, int(da.get("n") or 0))),
                             "sha256": da.get("digest_sha256")}}
        if self.paths.claims.is_file():
            inputs["claims"] = FUN.file_record(self.paths.claims)
        return FUN.header("prospective_da", FD.load(pre.get("domain") or M.DOMAIN), prereg, inputs,
                          modules=(V, BP))

    def _blind_check(self, da):
        """{"markers": the staged digest's hashed markers, "found": the markers
        its bytes hold now (none, or check_staged would have refused it)}."""
        marks = [list(m) for m in da.get("blind_markers") or []]
        found = []
        dg = _read_json(self.paths.plan_input(self.name, int(da.get("n") or 0)))
        if isinstance(dg, dict) and marks:
            found = BP.blind_found(BP._dump({k: v for k, v in dg.items() if k != "blind_markers"}),
                                   [(int(a), str(b)) for a, b in marks])
        return {"markers": marks, "found": found}

    def _score_da(self, ev):
        """outcome.record_da once per (prospective record, audit): when the
        lab's funnel/prospective_da.json exists and the evidence holds a valid
        audit of this Step 1 (diagnose._valid_audit), H11 and the DA's
        predictions enter the adversary's track record (contract 6 H11, 8.6)."""
        path = self.paths.lab_inc / "funnel" / "prospective_da.json"
        led = ev.json(E.FUNNEL_LEDGER)
        if not path.is_file() or not isinstance(led, dict):
            return
        audit, _c, _why = DG._valid_audit(ev, led)
        if audit is None:
            return
        raw = path.read_bytes()
        key = [hashlib.sha256(raw).hexdigest(), (ev.provenance.get(E.FUNNEL_AUDIT) or {}).get("sha256")
               or (ev.provenance.get(E.FUNNEL_AUDIT) or {}).get("json_sha256")]
        if self.st.get("da_scored") == key:
            return
        try:
            from ..funnel import ledger as FL
            pda = json.loads(raw.decode("utf-8"))
            out = OC.record_da(pda, audit, events_path=self.paths.track_events,
                               summary_path=self.paths.track_summary, recoverable_stages=FL.recoverable_stages(led))
        except Exception as e:
            self._once("da_score_failed:%s" % key[0][:12], str(e), "da_score_failed",
                       reasons=["%s: %s" % (type(e).__name__, _short(e, 400))])
            return
        self.st["da_scored"] = key
        self._ledger("da_scored", decided_by=out.get("proposed_by"), h11=(out.get("h11") or {}).get("verdict"),
                     scored=out.get("scored"), correct=out.get("correct"), prospective_sha256=key[0],
                     audit_sha256=key[1])

    def _complete_claims_note(self):
        """The COMPLETE card's line on the negative claims: a claim still open or
        challenged is never 'concluded' (contract 8.6)."""
        reg = self._claims()
        if not reg:
            return ""
        from ..funnel import claims as FC
        neg = FC.open_negative(reg)
        if not neg:
            return ""
        return (" Negative claim(s) not concluded: %s (a challenged claim is not concluded while its "
                "counter-arguments stand; a person signs a claim's final status, card X12)."
                % ", ".join("%s %s" % (c.get("id"), c.get("status")) for c in neg))

    def _funnel_lab_files(self):
        """{INC_DIR-relative path: sha256} of the lab's funnel files the sync
        pushes (executor.FUNNEL_SYNC_FILES), sha256 cached by size and mtime."""
        cache = self.st.setdefault("funnel_lab_cache", {})
        out, keep = {}, {}
        for rel in X.funnel_sync_list(self.paths.lab_inc):
            p = self.paths.lab_inc / rel
            try:
                stt = p.stat()
            except OSError:
                continue
            key = "%s|%d|%d" % (rel, stt.st_size, stt.st_mtime_ns)
            sha = cache.get(key) or hashlib.sha256(p.read_bytes()).hexdigest()
            keep[key] = sha
            out[rel] = sha
        self.st["funnel_lab_cache"] = keep
        return out

    def _funnel_sync(self):
        """The funnel sync (inc_funnel_sync, R0) when DIAGNOSE asked for it; it is
        this tick's one connection to the cluster."""
        st = self.st
        if not st.get("funnel_sync_due") or not self.ssh.left():
            return
        self.ssh.calls += 1                       # the sync opens its own connection: the tick's one
        res = X.submit({"id": _sha([self.name, "funnel_sync", self.utc])[:32], "policy_action": "inc_funnel_sync",
                        "params": {}, "reason": "push the lab's funnel files to the cluster"},
                       actor=AUTO, campaign=self.camp, ctx=self.xctx)
        st["funnel_sync_due"] = False
        self._ledger("funnel_sync", status=res.get("status"), reasons=res.get("reasons"),
                     pushed=((res.get("remote") or {}).get("payload") or {}).get("pushed"))

    def _lab_hooks(self):
        """The lab hooks this campaign's executor calls: the funnel fetch (L11a,
        L12), the verify queue (L14) and the funnel sync (LAB_HOOKS replaces
        them, tests)."""
        if LAB_HOOKS is not None:
            return LAB_HOOKS(self)
        return {"inc_funnel_fetch": X.funnel_fetch_hook(),
                "inc_funnel_sync": X.funnel_sync_hook(self.paths.lab_inc, os.environ.get("CLUSTER_SSH", "")),
                "inc_verify_queue": lambda params: X.write_verify_queue(
                    LV.verify_queue_rows(self.ev) if self.ev is not None else [], self.paths.verify_queue)}


    # ---- the brain
    def _stage_brain(self, ev, diags, cands, prop):
        st = self.st
        old = st.get("brain") or {}
        if old.get("status") in ("staged", "submitted"):
            self._ledger("brain_superseded", n=old.get("n"),
                         reasons=["plan %s for %s was still %s when %s was diagnosed"
                                  % (old.get("n"), old.get("exp"), old.get("status"), st["exp"])])
        n = int(st.get("brain_n") or 0) + 1
        st["brain_n"] = n
        exp = st["exp"]
        try:
            try:
                from . import corpus as C
                corp = C.Corpus()
            except Exception:
                corp = None
            parents = [e for e in st["exps"] if e != exp and e in ev.exps()][-3:]
            bud = self._budget_now() or {}
            residuals = ["%s deferred: %s" % (x.get("lever"), _short(x.get("reason"), 200))
                         for x in prop.get("deferred") or []]
            digest = BP.build_digest(ev, exp, diags, BP.load_menu(), campaign=self.name, n=n,
                                     track=OC.fold(OC.read_events(self.paths.track_events)),
                                     lineage=self.lineage(),
                                     budget={k: bud.get(k) for k in ("envelope_su", "spent_su",
                                                                     "committed_su", "remaining_su")},
                                     residuals=residuals, deterministic=cands, corpus=corp,
                                     parents=parents, created_utc=self.utc)
            _write_json(self.paths.plan_input(self.name, n), digest)
        except Exception as e:
            st["brain"] = {"n": n, "exp": exp, "status": "failed", "reason": "%s: %s"
                           % (type(e).__name__, _short(e, 300))}
            self._ledger("brain_failed", n=n, reasons=[st["brain"]["reason"]])
            return
        st["brain"] = {"n": n, "exp": exp, "status": "staged", "digest_sha256": digest["sha256"],
                       "staged_utc": self.utc, "model": self.cfg["brain"].get("model"),
                       "deterministic": [{k: p.get(k) for k in ("lever", "policy_action", "argv",
                                                                "child_exp", "params", "id")}
                                         for p in cands]}
        # The digest chose its num_ctx from its size and, if it had to, trimmed
        # itself to fit (brain_plan TRIM_ORDER); both are in the ledger, not
        # only in the staged file.
        self._ledger("brain_staged", n=n, digest_sha256=digest["sha256"],
                     tokens_estimated=digest.get("tokens_estimated"),
                     num_ctx=digest.get("num_ctx"),
                     trimmed=[c.get("cut") for c in (digest.get("trimmed") or {}).get("cuts") or []])

    def _brain_pulled(self, pull_res):
        st = self.st
        b = st.get("brain") or {}
        if b.get("status") != "submitted":
            return
        payload = ((pull_res or {}).get("remote") or {}).get("payload") if pull_res else None
        reply, exists = None, False
        if isinstance(payload, dict) and payload.get("verb") == "plan-pull":
            exists = bool(payload.get("exists"))
            reply = payload.get("reply")
            if exists and not isinstance(reply, dict):
                b.update(status="failed", reason=_short(payload.get("error") or "unreadable reply"))
                self._ledger("brain_failed", n=b.get("n"), reasons=[b["reason"]])
                return
        col = BP.collect(reply if exists else None, b.get("submitted_utc"), now_utc=self.utc,
                         digest_sha256=b.get("digest_sha256"), pulled_utc=self.utc)
        if col["status"] == "pending":
            return
        if col["status"] != "ready":
            b.update(status=col["status"], reason=_short(col.get("reason"), 500))
            self._ledger("brain_%s" % col["status"], n=b.get("n"), reasons=[col.get("reason")])
            return
        self._merge_brain(col)

    def _merge_brain(self, col):
        st = self.st
        b = st["brain"]
        reply = col["reply"]
        digest = _read_json(self.paths.plan_input(self.name, b["n"])) or {}
        fired = (((digest.get("sections") or {}).get("diagnoses") or {}).get("fired")) or []
        try:
            from . import corpus as C
            corp = C.Corpus()
        except Exception:
            corp = None
        val = V.validate(reply, self.ev, BP.load_menu(), corp, model=reply.get("model"),
                         diagnoses=fired, exp=b.get("exp"))
        try:
            OC.record_validation(val, events_path=self.paths.track_events,
                                 summary_path=self.paths.track_summary)
        except OSError:
            pass
        b["model_resolved"] = str(reply.get("model_used") or reply.get("model") or "") or None
        merged = BP.merge(b.get("deterministic") or [], col, val)
        filed = []
        for p in merged.get("proposals") or []:
            by = p.get("proposed_by")
            if by in (None, AUTO) or p.get("risk") not in ("R2", "R3"):
                continue
            dup = self._applied_by_lineage(p)
            held = (("%s was already applied to %s in this campaign (%s)"
                     % (p.get("lever"), p.get("parent_exp"), dup.get("child_exp") or dup.get("status")))
                    if dup is not None else self._d4_unfrozen(p, self.ev))
            if held:
                self._ledger("not_taken", decided_by=by, lever=p.get("lever"),
                             parent_exp=p.get("parent_exp"), child_exp=p.get("child_exp"),
                             reasons=["brain proposal not filed: %s" % held])
                filed.append({"lever": p.get("lever"), "status": "not_taken", "approval_id": None,
                              "reasons": [held]})
                continue
            r = X.submit(p, actor=by, campaign=self.camp, ctx=self.xctx)
            filed.append({"lever": p.get("lever"), "status": r.get("status"),
                          "approval_id": r.get("approval_id"), "reasons": (r.get("reasons") or [])[:3]})
        self._record_cards(merged.get("cards") or [], source=val.get("proposed_by"))
        b.update(status="merged", merged_utc=self.utc, filed=filed,
                 counts={k: (val.get("counts") or {}).get(k) for k in ("items", "valid", "dropped",
                                                                      "cards", "leaks")})
        self._ledger("brain_merged", decided_by=val.get("proposed_by") or "tier2",
                     n=b.get("n"), filed=filed, counts=b["counts"],
                     approval_ids=[f["approval_id"] for f in filed if f.get("approval_id")])

    def _brain_expire(self):
        """A plan that was never submitted in time is given up (the deterministic
        path never waits longer than the plan's own deadline)."""
        b = self.st.get("brain") or {}
        t0 = _secs(b.get("staged_utc"))
        if b.get("status") == "staged" and t0 is not None and self.now - t0 > BP.PLAN_TIMEOUT_S:
            b.update(status="timeout", reason="staged at %s and not submitted within %d s"
                     % (b.get("staged_utc"), BP.PLAN_TIMEOUT_S))
            self._ledger("brain_timeout", n=b.get("n"), reasons=[b["reason"]])

    def _da_expire(self):
        """A DA pass staged and never submitted in time is given up, as a brain
        plan is (a submitted one times out in brain_plan.collect)."""
        da = self.st.get("da") or {}
        t0 = _secs(da.get("staged_utc"))
        if da.get("status") == "staged" and t0 is not None and self.now - t0 > BP.PLAN_TIMEOUT_S:
            da.update(status="timeout", reason="staged at %s and not submitted within %d s"
                      % (da.get("staged_utc"), BP.PLAN_TIMEOUT_S))
            self._ledger("da_timeout", n=da.get("n"), reasons=[da["reason"]])

    def _brain_wait_step(self):
        st = self.st
        self._brain_expire()
        self._da_expire()
        b = st.get("brain") or {}
        if b.get("status") in ("staged", "submitted"):
            return
        if (st.get("da") or {}).get("status") in DA_IN_FLIGHT:
            return
        if st.pop("da_wait", False) and not (b.get("exp") == st.get("exp") and b.get("filed")):
            # The wait was for the DA pass alone: DIAGNOSE again, so the next
            # claim's pass is staged or the COMPLETE card states the claims as
            # the pass left them.
            st["phase"] = "DIAGNOSE"
            self._ledger("da_wait_over", n=(st.get("da") or {}).get("n"),
                         reasons=["the devil's-advocate pass ended %s: DIAGNOSE again"
                                  % (st.get("da") or {}).get("status")])
            return
        filed = [f for f in b.get("filed") or [] if f.get("status") == "filed"]
        if filed:
            # Nothing is in flight: the campaign waits on a person. COMPLETE
            # adopts an approved item on any later tick (GATE is for an item
            # in flight, and a GATE with none observes again).
            ids = [f["approval_id"] for f in filed]
            st["phase"] = "COMPLETE"
            st["item"] = None
            self._card("approval", "Brain proposals await a person", "approvals %s" % ", ".join(ids))
            st["card"]["approval_ids"] = ids
            self._ledger("brain_proposals_filed", approval_ids=ids)
            return
        self._complete("residual", "Nothing left the autopilot or the brain may propose",
                       "the brain plan %s ended %s (%s) with no proposal a person could approve; "
                       "a person decides the next step" % (b.get("n"), b.get("status"),
                                                          _short(b.get("reason") or "", 200)))


# --- status -----------------------------------------------------------------------------
def _alarm(cfg, st):
    if cfg.get("paused_reason") or ((st or {}).get("paused") or {}).get("reason"):
        return "crit"
    if not cfg.get("enabled"):
        return "off"
    if (st or {}).get("errors"):
        return "warn"
    if (st or {}).get("phase") == "COMPLETE" or ((st or {}).get("card") or {}).get("kind") in (
            "approval", "escalation", "cluster"):
        return "warn"
    return "ok"


def summary(cfg, st):
    """A compact, page-ready view of one campaign."""
    st = st or {}
    item = st.get("item") or {}
    p = item.get("proposal") or {}
    b = st.get("brain") or {}
    return {"enabled": bool(cfg.get("enabled")), "alarm": _alarm(cfg, st),
            "paused_reason": cfg.get("paused_reason") or (st.get("paused") or {}).get("reason"),
            "goal": cfg.get("goal"), "autonomy": cfg.get("autonomy"), "phase": st.get("phase"),
            "exp": st.get("exp"), "exps": st.get("exps"),
            "item": ({"lever": p.get("lever"), "action": p.get("policy_action"),
                      "argv": p.get("argv"), "status": item.get("status"),
                      "approval_id": item.get("approval_id"), "est_gpu_hours": p.get("est_gpu_hours"),
                      "trigger": p.get("trigger"), "parent_exp": p.get("parent_exp"),
                      "child_exp": p.get("child_exp")} if p else None),
            "card": st.get("card"), "cards": (st.get("cards") or [])[-10:],
            "fails": st.get("fails"), "errors": st.get("errors"),
            "health": [d.get("id") for d in st.get("health") or []],
            "diagnoses": [d.get("id") for d in st.get("diagnoses") or []],
            "building": st.get("building"), "wait_jobs": st.get("wait_jobs"),
            "brain": {k: b.get(k) for k in ("n", "exp", "status", "job_id", "submitted_utc",
                                            "reason", "counts")} if b else None,
            "prospective": st.get("prospective"), "rules_version": st.get("rules_version"),
            "last_tick_utc": st.get("last_tick_utc"),
            "last_snapshot_utc": st.get("last_snapshot_utc"), "ticks": st.get("ticks")}


def status(name=None, cfg_hooks=None, lab_repo=None):
    """{name: {"config", "summary"}} of every configured campaign (or one)."""
    cfg_hooks = cfg_hooks or default_cfg_hooks()
    paths = Paths(lab_repo)
    cfg = cfg_hooks[0]()
    camps = (cfg or {}).get(CONFIG_KEY) or {}
    out = {}
    for n in sorted(camps):
        if name and n != name:
            continue
        c = campaign_config(camps.get(n), n)
        out[n] = {"config": c, "summary": summary(c, load_state(paths, n))}
    return out


def write_status(paths, cfg_hooks, clock=time.time):
    """The per-tick heartbeat the dashboard's /api/health/inc reads."""
    cfg = cfg_hooks[0]()
    camps = (cfg or {}).get(CONFIG_KEY) or {}
    out = {"format": STATUS_FORMAT, "utc": _utc(clock()), "ts": clock(), "campaigns": {}}
    for n in sorted(camps):
        if not NAME_RE.match(str(n)):
            continue
        c = campaign_config(camps.get(n), n)
        out["campaigns"][n] = summary(c, load_state(paths, n))
    _write_json(paths.status, out)
    return out


# --- CLI --------------------------------------------------------------------------------
def cli_slurm_sh(target=None, repo=None):
    """The dashboard's cluster call, for a hand-run tick: `bash -lc SCRIPT` from the
    cluster repo over the same ControlMaster ssh (CLUSTER_SSH), or locally when
    CLUSTER_SSH is unset (on the cluster itself)."""
    target = target if target is not None else os.environ.get("CLUSTER_SSH", "")
    repo = repo or os.environ.get("CLUSTER_REPO", M.CLUSTER_REPO)

    def run(script, timeout=60):
        if target:
            cm = os.path.expanduser("~/.ssh/cm-%r@%h:%p")
            remote = "cd %s 2>/dev/null; %s" % (shlex.quote(repo), " ".join(
                shlex.quote(a) for a in ("bash", "-lc", script)))
            cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=25",
                   "-o", "ControlMaster=auto", "-o", "ControlPath=%s" % cm,
                   "-o", "ControlPersist=12h", target, remote]
            limit = max(int(timeout) + 20, 35)
        else:
            cmd, limit = ["bash", "-lc", script], int(timeout)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "TIMEOUT", "returncode": -1}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "returncode": -2}
        return {"ok": p.returncode == 0, "stdout": p.stdout, "stderr": p.stderr,
                "returncode": p.returncode}
    return run


def _parse_goal(a):
    if a.none:
        return None
    if a.diagnosis:
        did, sep, dname = a.diagnosis.partition(":")
        if not sep:
            raise ValueError("--diagnosis takes ID:NAME, e.g. D4:decision_slot_ready")
        return {"kind": "diagnosis", "id": did, "name": dname}
    if a.exp_done:
        return {"kind": "exp_done", "exp": a.exp_done}
    raise ValueError("set-goal needs --diagnosis ID:NAME, --exp-done EXP or --none")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inc_autopilot.campaign",
                                 description="INC campaign ticker (docs/INC_AUTOPILOT.md, component 6)")
    ap.add_argument("--config", default=None, help="scheduler config file (default ~/.round_scheduler.json)")
    ap.add_argument("--lab-repo", default=None, help="lab tree holding results/framework/_brain")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("status")
    s.add_argument("--name", default=None)
    t = sub.add_parser("tick")
    t.add_argument("--name", action="append", default=[])
    e = sub.add_parser("enable")
    e.add_argument("--name", required=True)
    e.add_argument("--by", required=True)
    e.add_argument("--exp", action="append", default=None)
    e.add_argument("--current", default=None)
    e.add_argument("--autonomy", choices=("off", "envelope"), default=None)
    e.add_argument("--envelope-su", type=float, default=None)
    e.add_argument("--daily-cap-su", type=float, default=None)
    e.add_argument("--brain", choices=("on", "off"), default=None)
    e.add_argument("--brain-model", default=None)
    p = sub.add_parser("pause")
    p.add_argument("--name", required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    g = sub.add_parser("set-goal")
    g.add_argument("--name", required=True)
    g.add_argument("--by", required=True)
    g.add_argument("--diagnosis", default=None)
    g.add_argument("--exp-done", default=None)
    g.add_argument("--none", action="store_true")
    a = ap.parse_args(argv)
    hooks = default_cfg_hooks(a.config)
    try:
        if a.cmd == "status":
            out = status(a.name, hooks, a.lab_repo)
        elif a.cmd == "tick":
            out = tick(slurm_sh=cli_slurm_sh(), cfg_hooks=hooks, lab_repo=a.lab_repo,
                       names=a.name or None)
        elif a.cmd == "enable":
            out = configure(a.name, a.by, enable=True, exps=a.exp, current=a.current,
                            autonomy=a.autonomy, envelope_su=a.envelope_su,
                            daily_cap_su=a.daily_cap_su,
                            brain=None if a.brain is None else a.brain == "on",
                            brain_model=a.brain_model, cfg_hooks=hooks, lab_repo=a.lab_repo)
        elif a.cmd == "pause":
            out = pause(a.name, a.reason, a.by, hooks, a.lab_repo)
        elif a.cmd == "set-goal":
            out = set_goal(a.name, _parse_goal(a), a.by, hooks, a.lab_repo)
        else:
            ap.print_help(sys.stderr)
            return 2
    except ValueError as err:
        print(json.dumps({"ok": False, "error": str(err)}))
        return 1
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0 if not (isinstance(out, dict) and out.get("ok") is False) else 1


if __name__ == "__main__":
    sys.exit(main())
