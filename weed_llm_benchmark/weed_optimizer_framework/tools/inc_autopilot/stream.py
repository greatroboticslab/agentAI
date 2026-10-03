"""Stream mode: the lane ticker of a continuous campaign (docs/CONTINUOUS_LOOP.md 6).

campaign.tick hands a campaign whose config says `mode: "stream"` to
StreamRun; experiment-mode campaigns keep campaign._Run, byte for byte.

Three lanes, one item in flight per lane:

    DATA   IDLE -> DISCOVER (L15, lab, detached) | NAMES (L26) | PROBE (LP)
                -> COLLECT (L16 job, or L16L on the lab + L16S sync) -> INTAKE (L16I)
                -> ADMIT (L17) -> IDLE; WAIT_DATA when collection is exhausted (D29)
    TRAIN  IDLE -> SEGMENT (L18, the experiment runs) -> COMMIT (L19) -> IDLE; L22 forks
    MAINT  IDLE -> R0 (L23 splits, L23B baselines, LV verdicts, L25 Stage A, LI init,
                LA choose-arm, L28 Stage C) | NATIVE (L23N, a done measurement
                arm's native-resolution rescore) | MILESTONE (L20) | COMPARE (LC)
                | ROLLBACK (L21) | BISECT (L27) | AUDIT (L4) -> IDLE

Each tick, in order (6.2):
  1. lab work, no ssh: the last snapshot and the lab's files are folded into
     the evidence; the diagnoses (diagnose_stream D20-D33 and the lanes' own
     items) run on it; the stop-losses and holds apply; each idle lane takes the
     first item its diagnoses call for; lab items (L15, L26, L16L, L16RL, L16S)
     start as detached processes, and the ones running are polled;
  2. the one ssh of the tick: the ready items of every lane go to the cluster
     in one executor.submit_many batch, in priority order (TRAIN, DATA, MAINT;
     an R3 item goes alone, since its envelope grant runs it in its own call);
     with none ready, one stream snapshot (executor.stream_snapshot) observes
     every lane;
  3. the health checks on that snapshot (D5-D7 and D14 on every live
     experiment, D10S, D26, D27) and the stream ledger's hash chain.

Never COMPLETE by itself (P7, 6.7): exhaustion moves the DATA lane to
WAIT_DATA with discovery backing off 7 -> 14 -> 30 days and a card; a person
may mark the campaign complete when every condition of 6.7 holds. Stop-losses
pause with a card: 2 consecutive failed steps hold a lane; 3 consecutive
zero-yield sources hold DATA; 2 held lanes, D7, D10S, D14, D27, a broken
stream ledger chain or 3 raising ticks pause the campaign.

Governance (6.5) is the executor's: R2 data levers run directly only with
data_autonomy on (a person's flag), a stream replay pass and the caps; R3
builds run within the envelope (autonomy 'envelope', granted by a person) or
wait for a person; R4 is a card. Module drift between the lab and the
cluster (S23) refuses every stream submission until they agree.

Domain-free (S13): the domain's facts come from its stream-domain config
(stream_domains/<domain>.json) and its funnel domain config; this module names
no class, source, exam or lab.

CLI (lab):
    python -m weed_optimizer_framework.tools.inc_autopilot.stream enable --name N --by human:<email>
        --domain D [--autonomy off|envelope] [--data-autonomy off|on] [--envelope-su SU]
        [--window-cap-su SU] [--daily-cap-su SU] [--alloc-reserve-su SU] [--collect-gb-envelope GB]
        [--collect-gb-daily GB] [--envelope-end-utc YYYY-MM-DDTHH:MM:SSZ] [--protocol-v3-accepted]
    python -m weed_optimizer_framework.tools.inc_autopilot.stream status [--name N]
    python -m weed_optimizer_framework.tools.inc_autopilot.stream release --name N --by human:<email>
        (a person's release of every lane a stop-loss held: 2 consecutive failed steps, a step past
        stop_loss step_retries, the DATA lane's zero-yield run; the next tick frees each lane held
        before the release and clears its steps' failure counts. `enable` does the same.)
    python -m weed_optimizer_framework.tools.inc_autopilot.stream lab-run --spec FILE   (the detached runner)
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
import time
import traceback
from pathlib import Path

from . import budget as B
from . import diagnose as DG
from . import diagnose_stream as DS
from . import evidence as E
from . import executor as X
from . import levers_stream as LS
from . import model as M
from ..brain import approvals as AP

STATE_FORMAT = "inc-autopilot/stream-state/1"
LANES = ("DATA", "TRAIN", "MAINT")
# A quarantine (L24) is a stop, not a collection step: it has a lane of its
# own so a busy DATA lane never delays it, and it goes to the cluster first
# (6.2 step 2: stop/cancel, then TRAIN, then DATA, then MAINT).
ALL_LANES = ("STOP",) + LANES
SUBMIT_ORDER = ("STOP", "TRAIN", "DATA", "MAINT")
AUTO = M.AUTOPILOT_ACTOR
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
HUMAN_RE = re.compile(r"^human:[A-Za-z0-9][A-Za-z0-9@._+-]{0,126}$")
HEALTH_EXP = ("D5", "D6", "D7", "D14")
CARDS_KEEP = 80
REFUSALS_KEEP = 5
FAILED_IDS_KEEP = 200                # proposal ids of lane items that ended failed (never proposed again)
# The collection levers, whose attempts are bounded per source (7.5: 3 failed
# attempts close a source; a 4th attempt pauses, S15): the step bound
# (stop_loss step_retries) leaves them to that rule.
SOURCE_BOUND = ("L16", "L16L", "L16R", "L16RL", "L16S", "L16I", "L17")
# The fetch levers (a source's attempt), the ones that fetch on the lab (folded
# from the lab process, then synced: L16S), and a source review's two forms
# (L16R on the cluster, L16RL on the lab, chosen by the candidate's placement).
FETCH_LEVERS = ("L16", "L16L", "L16R", "L16RL")
LAB_FETCH_LEVERS = ("L16L", "L16RL")
REVIEW_LEVERS = ("L16R", "L16RL")
STALE_TO_PAUSE = 2
BUILD_LOST_SNAPSHOTS = 3
WAIT_REFUSALS = ("today's cap", "this month's window", "of the domain's", "the cluster is not reachable",
                 "Mongo's health", "the execution log", "no slurm_sh hook", "could not be locked",
                 "collides with another request", "could not be filed", "one ssh per tick")
# L-2 (docs/CONTINUOUS_LOOP.md 2.6): the stream campaign's defaults.
STREAM_DEFAULTS = {"enabled": False, "paused_reason": None, "mode": "stream", "domain": None,
                   "protocol_package": None, "stream": {}, "goal": {"kind": "continuous"},
                   "autonomy": "off", "autonomy_granted_by": None, "data_autonomy": "off",
                   "envelope_su": 1000.0, "envelope_end_utc": "2026-12-31T23:59:59Z", "window": "month",
                   "window_cap_su": 350.0, "daily_cap_su": 120.0, "alloc_reserve_su": None,
                   "collect_gb_envelope": 200.0, "collect_gb_daily": 50.0, "protocol_v3_accepted_by": None,
                   "brain": {"enabled": False, "model": None}}
LANE_OF = {"L15": "DATA", "L26": "DATA", "LP": "DATA", "L16": "DATA", "L16L": "DATA", "L16R": "DATA",
           "L16RL": "DATA", "L16I": "DATA", "L16S": "DATA", "L17": "DATA", "L24": "STOP", "LH": "DATA",
           "L18": "TRAIN", "L19": "TRAIN", "L22": "TRAIN",
           "L20": "MAINT", "L21": "MAINT", "L23": "MAINT", "L23B": "MAINT", "L23N": "MAINT", "L25": "MAINT",
           "L27": "MAINT", "L28": "MAINT", "L4": "MAINT", "LV": "MAINT", "LI": "MAINT", "LA": "MAINT", "LC": "MAINT"}
# Record-only levers (a measurement arm's native-resolution rescore, L23N): a
# failure is a person's card, never a failed step of its lane (no stop-loss
# counts it), and the item is not proposed again (_failed); a submission whose
# outcome is unknown is followed by its job name and its record, never a pause
# (_follow_record_only). A qos refusal is a platform defect and holds the lane
# as for every lever (S21).
RECORD_ONLY_LEVERS = ("L23N",)
# A callable taking the StreamRun and returning its lab runner (tests).
LAB_RUNNER = None
# Refusals the identical request meets again (retry: false): a Step 1 batch
# refused on its pins (the frozen verifier, the v2 LOCK), or a cutter that
# cannot fill; a person decides (X11, D22).
RETRY_FALSE_RE = re.compile(r"verifier|LOCK|pinned|pins? (mismatch|differ)", re.I)
# R3 items only a person decides that must not hold their lane while they
# wait: filed "parked" and adopted into the lane once approved (a funnel_F9
# release at its deadline would otherwise stop collection until a person acts).
PERSON_LEVERS = ("LH",)


# ------------------------------------------------------------------ helpers
def step_key(lever, policy_params):
    """The key of one step of a lane (the attempts, the step's failed runs):
    the lever and its policy params (levers_stream.policy_params) without the
    price (est_gpu_hours), which may change between two proposals of the same
    step. The proposal side and the failure side compute it the same way, so a
    failed step is proposed again with the next attempt, under a new id."""
    return "%s:%s" % (lever, _sha({k: v for k, v in (policy_params or {}).items() if k != "est_gpu_hours"})[:12])


def _utc(t):
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _secs(stamp):
    try:
        return datetime.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


def _short(text, n=300):
    text = str(text)
    return text if len(text) <= n else text[:n] + "..."


def _record_only(d):
    """A measurement arm's D5 without OP_PAUSE: the arm is record only and no
    lane waits for it, so a block that is not transient is a person's card,
    never a pause, and never the firing stop-loss that refuses every envelope
    grant (executor._trigger_check). L7 for its transient units stays."""
    if "OP_PAUSE" not in (d.get("levers") or []):
        return d
    return dict(d, levers=[x for x in d["levers"] if x != "OP_PAUSE"], severity="warn",
                summary="%s (a measurement arm, record only: a card, not a pause)" % d.get("summary"),
                detail=dict(d.get("detail") or {}, record_only=True))


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


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fetch_facts(rows):
    """{"sources": {source: [[epoch, bytes]]}, "open": {source: [epoch]}} of a
    machine's source ledger (stream_remote.fetch_facts on each source row:
    its 'fetched' events, and its fetch_started events no closing event
    follows), which the byte limits count (executor.fetched_bytes); None
    when a row carries none (a fold written before they were kept, or of a
    ledger not read whole) or one is unreadable."""
    out = {"sources": {}, "open": {}}
    for src, row in rows.items():
        ev = row.get("fetched_events") if isinstance(row, dict) else None
        op = row.get("open_fetches") if isinstance(row, dict) else None
        if not isinstance(ev, list) or not isinstance(op, list):
            return None
        got = []
        for e in ev:
            if not isinstance(e, (list, tuple)) or len(e) != 2 or _num(e[1]) is None:
                return None
            got.append([_secs(e[0]), _num(e[1])])
        out["sources"][str(src)] = got
        if op:
            out["open"][str(src)] = [_secs(t) for t in op]
    return out


# ------------------------------------------------------------------ paths
class StreamPaths(object):
    """The lab files of one domain's stream campaigns (model.domain_campaign_dir)."""

    def __init__(self, lab_repo=None, domain=None):
        self.domain = domain or M.DOMAIN
        self.lab_repo = lab_repo
        self.campaign_dir = M.domain_campaign_dir(self.domain, lab_repo)
        self.ledger = self.campaign_dir / "inc_campaign.jsonl"
        self.replay = self.campaign_dir / "replay"
        self.campaigns = self.campaign_dir / "campaigns"
        root = Path(lab_repo) if lab_repo is not None else M.LAB_REPO
        self.lab_inc = root / "results" / "framework" / "inc"

    def state(self, name):
        return self.campaigns / name / "state.json"

    def latest(self, name):
        return self.campaigns / name / "latest_stream_snapshot.json"

    def diagnoses(self, name):
        return self.campaigns / name / "diagnoses.json"

    def frozen(self, name, exp):
        return self.campaigns / name / "frozen" / ("%s.json" % exp)

    def lab_jobs(self, name):
        return self.campaigns / name / "lab_jobs"

    def candidates(self):
        return self.lab_inc / "collect" / "candidates.json"

    def staging(self):
        return self.lab_inc / "intake" / "staging"

    def names_dir(self):
        return self.lab_inc / "intake" / "names"

    def cand_rel(self, source):
        """INC_DIR-relative path of one discovered source's candidate record, as
        synced to the cluster for its fetch there (collect fetch --candidates)."""
        return "intake/candidates_sync/%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", str(source))[:160]


# ------------------------------------------------------------------ config
def stream_config(raw, name):
    c = copy.deepcopy(STREAM_DEFAULTS)
    for k, v in (raw if isinstance(raw, dict) else {}).items():
        c[k] = copy.deepcopy(v)
    c["stream"] = dict(c.get("stream") or {})
    c["name"] = name
    return c


def check_stream_config(cfg):
    """[problems] of a stream campaign's config; [] when the ticker can run it."""
    out = []
    if cfg.get("mode") != "stream":
        out.append("mode is %r, not 'stream'" % cfg.get("mode"))
    if not isinstance(cfg.get("domain"), str) or not re.match(r"^[a-z][a-z0-9_]{0,39}$", cfg["domain"]):
        out.append("a stream campaign names its domain (domain: %r)" % cfg.get("domain"))
    if (cfg.get("goal") or {}).get("kind") != "continuous":
        out.append("a stream campaign's goal is {'kind': 'continuous'}, not %r" % cfg.get("goal"))
    for k in ("autonomy",):
        if cfg.get(k) not in ("off", "envelope"):
            out.append("%s is 'off' or 'envelope', not %r" % (k, cfg.get(k)))
    if cfg.get("data_autonomy") not in ("off", "on"):
        out.append("data_autonomy is 'off' or 'on', not %r" % cfg.get("data_autonomy"))
    return out


def configure_stream(name, by, domain=None, enable=None, autonomy=None, data_autonomy=None, envelope_su=None,
                     window_cap_su=None, daily_cap_su=None, alloc_reserve_su=None, collect_gb_envelope=None,
                     protocol_v3_accepted=None, stream_domain=None, collect_config=None, cfg_hooks=None,
                     lab_repo=None, clock=None, complete=None, envelope_end_utc=None, collect_gb_daily=None):
    """Create or change a stream campaign as person `by`. data_autonomy,
    autonomy 'envelope', the acceptance of Protocol v3 and a completion are a
    person's flags (6.5, 10): the ticker never writes them."""
    from . import campaign as C
    if not NAME_RE.match(str(name or "")):
        raise ValueError("campaign name %r is not valid" % (name,))
    if not HUMAN_RE.match(str(by or "")):
        raise ValueError("a stream campaign is configured by a person (human:<email>), not %r" % (by,))
    for label, v in (("envelope_su", envelope_su), ("window_cap_su", window_cap_su), ("daily_cap_su", daily_cap_su),
                     ("alloc_reserve_su", alloc_reserve_su), ("collect_gb_envelope", collect_gb_envelope),
                     ("collect_gb_daily", collect_gb_daily)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            raise ValueError("%s must be a number >= 0, got %r" % (label, v))
    if envelope_end_utc is not None and _secs(envelope_end_utc) is None:
        # the envelope's end pauses the campaign (envelope_ended): a person
        # sets the new end here, never by hand-editing the config
        raise ValueError("envelope_end_utc must read YYYY-MM-DDTHH:MM:SSZ, got %r" % (envelope_end_utc,))
    cfg_hooks = cfg_hooks or C.default_cfg_hooks()
    now = (clock or time.time)()
    utc = _utc(now)

    def change(c):
        c.setdefault("mode", "stream")
        if c.get("mode") != "stream":
            raise ValueError("campaign %s is an experiment-mode campaign; a stream campaign is a new one" % name)
        c.setdefault("goal", {"kind": "continuous"})
        if domain is not None:
            c["domain"] = domain
        dom = LS.load_domain(stream_domain or c.get("stream", {}).get("stream_domain") or c.get("domain"))
        st = dict(c.get("stream") or {})
        st.setdefault("sid", dom["sid"])
        st.setdefault("stream_domain", stream_domain or dom["domain"])
        st.setdefault("collect_config", collect_config or dom.get("collect_config"))
        st.setdefault("K_max", (dom.get("increment") or {}).get("K_max"))
        c["stream"] = st
        c.setdefault("protocol_package", dom.get("protocol_package"))
        if autonomy is not None:
            if autonomy not in ("off", "envelope"):
                raise ValueError("autonomy is 'off' or 'envelope'")
            c["autonomy"] = autonomy
            if autonomy == "envelope":
                c["autonomy_granted_by"] = by
        if data_autonomy is not None:
            if data_autonomy not in ("off", "on"):
                raise ValueError("data_autonomy is 'off' or 'on'")
            c["data_autonomy"] = data_autonomy
            c["data_autonomy_by"] = by
            c["data_autonomy_utc"] = utc
        for k, v in (("envelope_su", envelope_su), ("window_cap_su", window_cap_su), ("daily_cap_su", daily_cap_su),
                     ("alloc_reserve_su", alloc_reserve_su), ("collect_gb_envelope", collect_gb_envelope),
                     ("collect_gb_daily", collect_gb_daily)):
            if v is not None:
                c[k] = float(v)
        if envelope_end_utc is not None:
            c["envelope_end_utc"] = str(envelope_end_utc)
        if protocol_v3_accepted:
            c["protocol_v3_accepted_by"] = by
        if complete:
            c["completed_by"], c["completed_utc"] = by, utc
        if enable is not None:
            c["enabled"] = bool(enable)
            if enable:
                c["paused_reason"] = None
                c["resumed_utc"] = utc
        c.setdefault("enabled", False)
        c["updated_by"], c["updated_utc"] = by, utc
        return c
    new = C._update_config(cfg_hooks, name, change)
    full = stream_config(new, name)
    bad = check_stream_config(full)
    if bad:
        raise ValueError("; ".join(bad))
    paths = StreamPaths(C.lab_repo_arg(lab_repo) if lab_repo else None, full["domain"])
    _append(paths.ledger, {"utc": utc, "campaign": name, "event": "configured", "decided_by": by,
                           "config": {k: full.get(k) for k in ("enabled", "domain", "autonomy", "data_autonomy",
                                                               "envelope_su", "envelope_end_utc", "window_cap_su",
                                                               "daily_cap_su", "alloc_reserve_su", "collect_gb_envelope",
                                                               "collect_gb_daily", "protocol_v3_accepted_by",
                                                               "stream")}})
    return full


def _append(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def blank_state(name, cfg):
    return {"format": STATE_FORMAT, "name": name, "ticks": 0, "errors": 0, "paused": None,
            "lanes": {ln: {"phase": "IDLE", "item": None, "fails": 0, "hold": None, "diag_hold": None,
                           "until_utc": None} for ln in ALL_LANES},
            "sources": {}, "zero_run": [], "segments": [], "milestones": [], "exps": [], "frozen": {},
            "stage": {"r0": {}}, "discover": {}, "cards": [], "card": None, "notes": {},
            "refusals": [], "declined": [], "source_reviews": {}, "person_items": {}, "history": {}, "stale": {},
            "d33_history": [],
            "attempts": {}, "rollbacks": [], "failed_ids": [], "step_failures": {},
            "doublings": 0, "drift": None, "decisions_logged": False, "prospective": None, "bisected": None,
            "last_snapshot_utc": None, "last_tick_utc": None, "snapshot_failures": 0, "updated_utc": None}


def load_state(paths, name):
    st = _read_json(paths.state(name))
    return st if isinstance(st, dict) and st.get("format") == STATE_FORMAT else None


# ------------------------------------------------------------------ the lab runner
LAB_TIMEOUT_S = 6 * 3600                  # a lab job's wall clock unless it moves bytes (below)
LAB_FETCH_MIN_RATE = 1.0e6                # bytes/s a lab fetch is still worth waiting for
LAB_FETCH_TIMEOUT_MAX_S = 48 * 3600


def lab_job_timeout(kind, max_bytes):
    """The lab runner's wall clock for one job. A fetch is sized from the bytes
    it may move: 2 h plus max_bytes at 1 MB/s, between 6 h and 48 h. The flat
    6 h cut a 49.7 GB single-file fetch (zenodo_15808623, about 5.8 h at the
    2.4 MB/s the lab measured on 2026-10-02) at the limit, and a killed fetch
    restarts from zero: the collector resumes only within its own process.
    The collector's own deadline (base_timeout_s + size / min_rate_bytes_per_s)
    still decides a stalled download well before this."""
    if kind != "fetch":
        return LAB_TIMEOUT_S
    try:
        b = float(max_bytes)
    except (TypeError, ValueError):
        return LAB_TIMEOUT_S
    if b <= 0:
        return LAB_TIMEOUT_S
    return int(min(LAB_FETCH_TIMEOUT_MAX_S, max(LAB_TIMEOUT_S, 2 * 3600 + b / LAB_FETCH_MIN_RATE)))


class LabRunner(object):
    """Detached lab processes (S22: a lab fetch never blocks the tick). launch()
    starts `python -m ...stream lab-run --spec FILE` in its own session and
    returns at once; the child runs the argv and writes FILE's result path."""

    def __init__(self, root, cwd=None, python=None):
        self.root, self.cwd, self.python = Path(root), cwd or str(LS.CODE_ROOT), python or sys.executable

    def launch(self, job, argv, timeout=LAB_TIMEOUT_S):
        self.root.mkdir(parents=True, exist_ok=True)
        spec = {"job": job, "argv": [str(a) for a in argv], "result": str(self.root / ("%s.result.json" % job)),
                "timeout": int(timeout), "cwd": self.cwd}
        sp = self.root / ("%s.spec.json" % job)
        _write_json(sp, spec)
        env = dict(os.environ)
        env["PYTHONPATH"] = self.cwd + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.Popen([self.python, "-m", "weed_optimizer_framework.tools.inc_autopilot.stream", "lab-run",
                          "--spec", str(sp)], cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return {"job": job, "spec": str(sp)}

    def poll(self, job):
        return _read_json(self.root / ("%s.result.json" % job))


def lab_run(spec_path):
    """The detached child: run the spec's argv, write its result atomically."""
    spec = _read_json(spec_path) or {}
    t0 = time.time()
    try:
        p = subprocess.run(spec["argv"], cwd=spec.get("cwd"), capture_output=True, text=True,
                           timeout=int(spec.get("timeout") or 3600))
        res = {"ok": p.returncode == 0, "rc": p.returncode, "tail": (p.stdout or "")[-2000:],
               "stderr_tail": (p.stderr or "")[-800:]}
    except (OSError, subprocess.TimeoutExpired, KeyError) as e:
        res = {"ok": False, "rc": None, "error": "%s: %s" % (type(e).__name__, e)}
    res.update(job=spec.get("job"), seconds=round(time.time() - t0, 3), finished_utc=M.utc_now())
    _write_json(spec.get("result") or (str(spec_path) + ".result.json"), res)
    return 0 if res.get("ok") else 1


def sync_argv(lab_inc, source, target, data_target, file=None, names=False):
    """The lab -> cluster push of one source's staging (L16S), or of one
    discovered source's candidate record (`file`, under intake/candidates_sync/),
    or of the collector's names layer for it (`names`), as the funnel's sync:
    rsync of the listed files, then a sha256 check of each on arrival."""
    return [sys.executable, "-m", "weed_optimizer_framework.tools.inc_autopilot.stream", "lab-sync",
            "--lab-inc", str(lab_inc), "--source", str(source), "--target", str(target or ""),
            "--data-target", str(data_target or "")] + (["--file", str(file)] if file else []) + \
        (["--names"] if names else [])


CAND_SYNC_DIR = "intake/candidates_sync"
# The collector's names layer (collect.names: INC_DIR/intake/names/, written by
# L26 on the lab only, since the authority needs the network). Every collector
# job on the cluster reads it offline, so a source whose names L26 resolved is
# refused there as names-pending until the layer is pushed.
NAMES_DIR = "intake/names"


def names_files(root, source):
    """The names-layer files of one source that exist under the lab INC_DIR
    `root`: the layer itself and the source's resolution record (the file name
    as collect.names writes it, collect.safe_name)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source)).strip("_") or "x"
    rels = ["%s/names_cache.json" % NAMES_DIR, "%s/names_%s.json" % (NAMES_DIR, safe)]
    return [r for r in rels if (Path(root) / r).is_file()]


def lab_sync(lab_inc, source, target, data_target=None, runner=None, cluster_inc=None, repo=None, file=None,
             names=False):
    """Push INC_DIR/intake/staging/<source>/ (or the one file `file`, under
    intake/candidates_sync/; or, with `names`, the names layer and the
    source's names record under intake/names/) to the cluster and verify every
    file on arrival (sha256), refusing any path outside it. A staging push
    carries the source's names files too, so the cluster's intake reads the
    names L26 resolved on the lab. {"ok", "pushed", "error"}."""
    run = runner or subprocess.run
    root = Path(lab_inc)
    if names:
        files = names_files(root, source)
        if not files:
            return {"ok": False, "pushed": [], "error": "no names layer under %s/%s" % (root, NAMES_DIR)}
    elif file:
        rel = Path(str(file))
        if rel.is_absolute() or ".." in rel.parts or str(rel.parent) != CAND_SYNC_DIR or not (root / rel).is_file():
            return {"ok": False, "pushed": [], "error": "%s is not a candidate record under %s" % (file, CAND_SYNC_DIR)}
        files = [str(rel)]
    else:
        base = root / "intake" / "staging" / str(source)
        if not base.is_dir():
            return {"ok": False, "pushed": [], "error": "no staging for %s" % source}
        files = []
        for d, _dirs, fs in os.walk(str(base)):
            for f in fs:
                files.append(str((Path(d) / f).relative_to(root)))
        files = sorted(files) + names_files(root, source)
    if any(".." in Path(f).parts for f in files):
        return {"ok": False, "pushed": [], "error": "a path outside the staging"}
    if not target:
        return {"ok": False, "pushed": [], "error": "no cluster ssh target (CLUSTER_SSH)"}
    shas = {rel: LS.sha256_file(root / rel) for rel in files}     # streamed: staging blobs run to GBs
    cluster_inc = cluster_inc or M.CLUSTER_INC_DIR
    p = run(["rsync", "-a", "--files-from=-", "--", str(root) + "/", "%s:%s/" % (data_target or target, cluster_inc)],
            input="\n".join(files) + "\n", capture_output=True, text=True, timeout=6 * 3600)
    if p.returncode != 0:
        return {"ok": False, "pushed": [], "error": "rsync exited %d: %s" % (p.returncode, (p.stderr or "")[-300:])}
    c = run(["ssh", target, X._remote_py(repo or M.CLUSTER_REPO, X._SYNC_CHECK_PY, [cluster_inc, "0"])],
            input=json.dumps(shas, sort_keys=True), capture_output=True, text=True, timeout=600)
    return {"ok": c.returncode == 0, "pushed": files, "sha256": shas,
            "error": "" if c.returncode == 0 else "the arrival check failed: %s" % ((c.stdout or "") + (c.stderr or ""))[-300:]}


# ------------------------------------------------------------------ the ticker
class StreamRun(object):
    """One stream campaign's step in one tick (campaign._tick_all dispatches here)."""

    def __init__(self, name, raw, paths, ssh, clock, log, cfg_hooks, resources, domain_budget, log_action,
                 preamble):
        self.name, self.ssh, self.clock, self.log = name, ssh, clock, log
        self.cfg_hooks, self.log_action, self.resources = cfg_hooks, log_action, resources
        self.cfg = stream_config(raw, name)
        self.now = clock()
        self.utc = _utc(self.now)
        self.domain = M.campaign_domain(self.cfg)
        self.paths = StreamPaths(paths.lab_repo, self.domain)
        self.st = None
        self.ev = None
        self.diags = []
        self.dom = None
        self.th = None
        self.prior = None
        self.domain_budget = domain_budget
        self.preamble = preamble
        self.xctx = X.Context(slurm_sh=ssh, resources=resources, domain_budget=domain_budget, clock=clock,
                              lab_repo=paths.lab_repo, preamble=preamble, domain=self.domain,
                              diagnoses=lambda _n: self._hook_diagnoses())
        self.runner = LAB_RUNNER(self) if LAB_RUNNER is not None else LabRunner(self.paths.lab_jobs(name))
        self.xctx.local_hooks.update(self._lab_hooks())

    # ---- plumbing
    @property
    def sid(self):
        return str((self.cfg.get("stream") or {}).get("sid") or (self.dom or {}).get("sid"))

    @property
    def pkg(self):
        return self.cfg.get("protocol_package") or (self.dom or {}).get("protocol_package")

    def _fired(self):
        return [d for d in self.diags if isinstance(d, dict) and d.get("fired")]

    def _hook_diagnoses(self):
        """The executor's diagnoses hook (as campaign._Run's): the fired
        diagnoses of this tick, and those each lane item was proposed from --
        a lane-gated diagnosis (D22, D24) goes silent once its item holds the
        lane, and the envelope still checks the item against the diagnosis
        that proposed it. Stop-loss and health diagnoses are this tick's."""
        now = {d.get("id"): d for d in self._fired()}
        out = list(now.values())
        for ln in ALL_LANES:
            it = ((self.st or {}).get("lanes") or {}).get(ln, {}).get("item") or {}
            for d in it.get("diagnoses") or []:
                if isinstance(d, dict) and d.get("fired") and d.get("id") not in now:
                    now[d["id"]] = d
                    out.append(d)
        return out

    def _ledger(self, event, **kw):
        rec = {"utc": self.utc, "campaign": self.name, "event": event, "mode": "stream",
               "tick": (self.st or {}).get("ticks"), "decided_by": kw.pop("decided_by", AUTO)}
        rec.update(kw)
        try:
            _append(self.paths.ledger, rec)
        except OSError as e:
            self.log.warning("[inc-stream] %s: ledger not written: %s" % (self.name, e))
        return rec

    def _once(self, key, value, event, **kw):
        notes = self.st.setdefault("notes", {})
        h = _sha(value)
        if notes.get(key) == h:
            return None
        notes[key] = h
        return self._ledger(event, **kw)

    def _card(self, kind, title, detail="", lever=None, trigger=None):
        st = self.st
        key = "%s:%s:%s" % (kind, lever or "", _sha([title, detail])[:12])
        if any(c.get("key") == key for c in st.get("cards") or []):
            st["card"] = {"kind": kind, "title": title, "detail": detail, "utc": self.utc}
            return
        rec = {"key": key, "kind": kind, "lever": lever, "title": title, "detail": _short(detail, 2000),
               "risk": "R4" if lever and str(lever).startswith("X") else None, "utc": self.utc,
               "trigger": list(trigger or [])}
        st["cards"] = (list(st.get("cards") or []) + [rec])[-CARDS_KEEP:]
        st["card"] = {"kind": kind, "title": title, "detail": _short(detail, 2000), "utc": self.utc}
        self._ledger("card", kind=kind, lever=lever, title=title, detail=_short(detail, 1000),
                     trigger=list(trigger or []))

    def _pause(self, reason, detail=""):
        from . import campaign as C
        st = self.st
        st["paused"] = {"reason": reason, "utc": self.utc, "by": AUTO, "config_written": False}
        self._card("paused", "Paused: " + _short(reason, 160), detail or reason)

        def change(c):
            c.update(enabled=False, paused_reason=reason)
            return c
        try:
            C._update_config(self.cfg_hooks, self.name, change)
            self.cfg.update(enabled=False, paused_reason=reason)
            st["paused"]["config_written"] = True
        except Exception as e:
            self.log.warning("[inc-stream] %s: pause not written to the config (%s)" % (self.name, e))
        self._ledger("paused", reason=reason)
        self.log.error("[inc-stream] %s PAUSED: %s" % (self.name, reason))

    def _paused_reason(self):
        if not self.cfg.get("enabled"):
            return self.cfg.get("paused_reason") or "not enabled"
        return self.cfg.get("paused_reason") or ((self.st.get("paused") or {}).get("reason"))

    def _resumed(self):
        st = self.st
        p = st.get("paused") or {}
        r = self.cfg.get("resumed_utc")
        if not p and r and self.cfg.get("enabled") and not self.cfg.get("paused_reason"):
            # a lane held by a stop-loss while the campaign ran on: a person's
            # resume (`stream enable`) after the hold releases it; without this
            # a single held lane could only be released by a pause
            for ln in LANES:
                lane = st["lanes"][ln]
                if lane.get("hold") and lane.get("hold_utc") and str(r) >= str(lane["hold_utc"]):
                    self._ledger("lane_released", lane=ln, hold=lane["hold"],
                                 decided_by=self.cfg.get("updated_by") or "human")
                    lane["hold"], lane["fails"], lane["hold_utc"] = None, 0, None
                    self._release_steps(ln)
                    if ln == "DATA":
                        st["zero_run"] = []
        if not p or not self.cfg.get("enabled") or self.cfg.get("paused_reason"):
            return
        if not ((r and str(r) >= str(p.get("utc") or "")) or p.get("config_written")):
            return
        st["paused"] = None
        st["errors"] = 0
        for ln in LANES:                             # a person's resume clears the lanes' stop-loss holds
            lane = st["lanes"][ln]
            if lane.get("hold"):
                self._ledger("lane_released", lane=ln, hold=lane["hold"])
                self._release_steps(ln)
            lane["hold"], lane["fails"] = None, 0
        st["zero_run"] = []
        if (st.get("card") or {}).get("kind") == "paused":
            st["card"] = None
        self._ledger("resumed", decided_by=self.cfg.get("updated_by") or "human")

    @property
    def camp(self):
        """The campaign as the executor reads it (executor._campaign)."""
        c, st = self.cfg, self.st or {}
        lanes = st.get("lanes") or {}
        in_flight = {}
        for ln in ALL_LANES:
            it = (lanes.get(ln) or {}).get("item")
            if it and it.get("status") in ("executed", "running"):
                fam = LS.family(it["lever"]) if it.get("lever") in LANE_OF else it.get("lever")
                in_flight[fam] = in_flight.get(fam, 0) + 1
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) if self.st else None
        pend = [x for x in ((q or {}).get("rollback_pending") or []) if isinstance(x, dict) and x.get("to_pool")]
        return {"name": self.name, "mode": "stream", "autonomy": c.get("autonomy") or "off",
                "autonomy_granted_by": c.get("autonomy_granted_by"), "envelope_su": c.get("envelope_su"),
                "daily_cap_su": c.get("daily_cap_su"), "window_cap_su": c.get("window_cap_su"),
                "data_autonomy": c.get("data_autonomy") or "off",
                "paused_reason": c.get("paused_reason") or (st.get("paused") or {}).get("reason"),
                "last_milestone_pool": pend[-1]["to_pool"] if pend else None,
                "in_flight": in_flight, "limit_counts": self._limit_counts(),
                "collect_gb_envelope": c.get("collect_gb_envelope"), "collect_gb_daily": c.get("collect_gb_daily"),
                "fetched": self._fetched()}

    def _limit_counts(self):
        st = self.st or {}
        q = (self._artifact("stream/%s/queue_summary.json" % self.sid) if self.st else None) or {}
        pend = [x for x in q.get("rollback_pending") or [] if isinstance(x, dict)]
        rb = st.get("rollbacks") or []
        out = {"L21": {"per_milestone": sum(1 for r in rb if pend and r.get("milestone") == pend[-1].get("milestone"))},
               "L27": {"per_rollback": 1 if rb and st.get("bisected") == rb[-1].get("utc") else 0},
               "L28": {"per_stream_version": 1 if ((st.get("stage") or {}).get("r0") or {}).get("stage_c_built")
                       else 0}}
        return out

    def _fetched(self):
        """What the executor's byte limits count fetched bytes from
        (executor.fetched_bytes, 6.6): the fetch attempts whose end the lane
        saw, the cluster collector's fetched facts (_fetch_facts) as of the
        last snapshot that folded its ledger, and the lab collector's own."""
        st = self.st or {}
        return {"ended": dict(self._fetch_ends()), "cluster": copy.deepcopy(st.get("cluster_fetched") or {}),
                "lab": self._lab_fetched()}

    def _fetch_ends(self):
        """{proposal id: utc} of this campaign's fetch attempts (FETCH_LEVERS)
        whose end the lane observed (state fetch_ends, written by _done and
        _failed). A state written before the record existed takes them once
        from the campaign ledger's item_done and failed entries, so attempts
        that ended before it count their fetched bytes too."""
        st = self.st
        if not isinstance(st, dict):
            return {}
        if not isinstance(st.get("fetch_ends"), dict):
            ends = {}
            try:
                with open(str(self.paths.ledger), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if '"item_done"' not in line and '"failed"' not in line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("campaign") == self.name and rec.get("event") in ("item_done", "failed") \
                                and rec.get("lever") in FETCH_LEVERS and rec.get("proposal_id"):
                            ends[str(rec["proposal_id"])] = rec.get("utc")
            except OSError:
                pass
            st["fetch_ends"] = ends
        return st["fetch_ends"]

    def _lab_fetched(self):
        """The fetched facts (_fetch_facts) of the lab collector's own ledger
        (lab INC_DIR/intake/sources.jsonl: L16L and L16RL append there, and no
        snapshot folds it; a fetch that failed after finishing some files
        records those), or None when it cannot be read whole (missing,
        unreadable, a line that is not JSON): the lab's ended fetches then
        count their requested max_bytes."""
        from . import stream_remote as SR
        rows = []
        try:
            with open(str(self.paths.lab_inc / "intake" / "sources.jsonl"), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if isinstance(r, dict) and r.get("source"):
                        rows.append(r)
        except (OSError, ValueError):
            return None
        rows.sort(key=lambda r: (str(r.get("ts") or ""), str(r.get("source") or "")))
        ff = SR.fetch_facts(rows)
        return None if ff is None else _fetch_facts(ff)

    # ---- entry
    def go(self):
        st0 = load_state(self.paths, self.name)
        if st0 is None:
            if not self.cfg.get("enabled"):
                return {"enabled": False, "mode": "stream"}
            st0 = blank_state(self.name, self.cfg)
        self.st = copy.deepcopy(st0)
        for ln in ALL_LANES:                         # a state written before a lane existed
            self.st["lanes"].setdefault(ln, {"phase": "IDLE", "item": None, "fails": 0, "hold": None,
                                             "diag_hold": None, "until_utc": None})
        try:
            out = self._step()
            self.st["errors"] = 0
            self.st["last_error"] = None
            self.st["last_tick_utc"] = self.st["updated_utc"] = self.utc
            _write_json(self.paths.state(self.name), self.st)
            self._mirror_config()
            return out
        except Exception as e:
            self.st = st0
            st0["errors"] = int(st0.get("errors") or 0) + 1
            st0["last_error"] = {"utc": self.utc, "error": "%s: %s" % (type(e).__name__, _short(e)),
                                 "trace": _short(traceback.format_exc(), 3000)}
            st0["last_tick_utc"] = self.utc
            self._ledger("error", error=st0["last_error"]["error"])
            if st0["errors"] >= int(LS.t(self.th or LS.load_thresholds(), "stop_loss", "errors_to_pause")):
                self._pause("the ticker raised on %d ticks in a row (last: %s)" % (st0["errors"],
                                                                                   st0["last_error"]["error"]))
            try:
                _write_json(self.paths.state(self.name), st0)
            except Exception:
                pass
            self.log.warning("[inc-stream] %s: tick raised %s" % (self.name, st0["last_error"]["error"]))
            return {"error": st0["last_error"]["error"], "mode": "stream"}

    def _mirror_config(self):
        from . import campaign as C
        st = self.st
        view = {"mode": "stream", "lanes": {ln: {"phase": st["lanes"][ln]["phase"],
                                                  "item": (st["lanes"][ln].get("item") or {}).get("lever"),
                                                  "hold": st["lanes"][ln].get("hold") or st["lanes"][ln].get("diag_hold")}
                                             for ln in ALL_LANES},
                "card": ({"kind": st["card"].get("kind"), "title": st["card"].get("title")} if st.get("card") else None),
                "paused": (st.get("paused") or {}).get("reason")}
        cur = self.cfg.get("state") if isinstance(self.cfg.get("state"), dict) else {}
        if {k: cur.get(k) for k in view} == view:
            return

        def change(c):
            c["state"] = dict(view, updated_utc=self.utc)
            return c
        try:
            C._update_config(self.cfg_hooks, self.name, change)
        except Exception as e:
            self.log.warning("[inc-stream] %s: config view not written: %s" % (self.name, e))

    def _step(self):
        st = self.st
        st["ticks"] = int(st.get("ticks") or 0) + 1
        self._resumed()
        why = self._paused_reason()
        if why:
            return {"mode": "stream", "paused": why, "ssh": False}
        bad = check_stream_config(self.cfg)
        if bad:
            self._card("config", "The stream campaign's config cannot run", "; ".join(bad))
            return {"mode": "stream", "error": "; ".join(bad)}
        self.dom = LS.load_domain((self.cfg.get("stream") or {}).get("stream_domain") or self.domain)
        self.th = LS.load_thresholds()
        self.prior = DS.prior_evidence(self.dom)
        self._log_decisions()
        env_end = _secs(self.cfg.get("envelope_end_utc"))
        if env_end is not None and self.now > env_end:
            self._pause("envelope_ended: the campaign envelope ran to %s; a person sets a new one (card X14)"
                        % self.cfg.get("envelope_end_utc"))
            return {"mode": "stream", "paused": "envelope_ended", "ssh": False}
        before = self.ssh.calls
        if not self.paths.latest(self.name).is_file() or self.st.get("await_snapshot"):
            # nothing observed yet (or only the stream a fork replaced): this
            # tick's one call is the snapshot
            self._poll_lab()
            self._observe()
            return {"mode": "stream", "ssh": self.ssh.calls > before, "first": True}
        # 1. lab work on the cached evidence (no ssh)
        self._poll_lab()
        self._build_evidence()
        self._diagnose()
        if self._paused_reason():
            return {"mode": "stream", "paused": self._paused_reason(), "ssh": self.ssh.calls > before}
        self._materialise()
        self._run_lab_items()
        # 2. the one ssh: the ready items, else the snapshot
        if self.ssh.left() and not self._paused_reason():
            if not self._submit_ready():
                self._observe()
        return {"mode": "stream", "ssh": self.ssh.calls > before,
                "lanes": {ln: {"phase": st["lanes"][ln]["phase"], "item": (st["lanes"][ln].get("item") or {}).get("lever"),
                               "status": (st["lanes"][ln].get("item") or {}).get("status")} for ln in ALL_LANES}}

    def _log_decisions(self):
        """DEC-A..DEC-D (human) and L-1..L-7 (human-delegated), logged once per campaign."""
        st = self.st
        if st.get("decisions_logged"):
            return
        for d in self.dom.get("decisions") or []:
            self._ledger("decision", decided_by=d.get("decided_by"), id=d.get("id"), text=d.get("text"),
                         decided_utc=d.get("utc"), domain_config_sha256=self.dom.get("_sha256"))
        st["decisions_logged"] = True

    # ---- the lab's own files
    def _collect_config(self):
        rel = (self.cfg.get("stream") or {}).get("collect_config") or self.dom.get("collect_config")
        if not rel:
            return None, None
        p = Path(rel)
        if not p.is_absolute():
            p = LS.TOOLS_DIR / rel
        return p, _read_json(p)

    def _never_train(self):
        return LS.never_train_slugs(self.dom)

    def _candidate_row(self, source):
        """The raw row of a source in the lab's candidates.json, or None."""
        raw = _read_json(self.paths.candidates())
        rows = raw.get("candidates") if isinstance(raw, dict) else raw
        for r in rows or []:
            if isinstance(r, dict) and str(r.get("source_id") or r.get("id") or "") == str(source):
                return r
        return None

    def _candidates(self):
        """The lab's candidates (L15's candidates.json, collect-candidates/1) and
        the known items of the collect config, normalised, with each source's
        state merged in. The collector's own pre-check (`precheck`: failures
        with their action and the governance class of the decision they ask
        for) is honoured: a close or refuse action refuses the source, an R3
        hold files it for a person, another hold waits."""
        raw = _read_json(self.paths.candidates())
        rows = raw.get("candidates") if isinstance(raw, dict) else raw
        rows = [r for r in rows or [] if isinstance(r, dict)]
        _p, cc = self._collect_config()
        known = [dict(r, known_item=True) for r in ((cc or {}).get("known_items") or []) if isinstance(r, dict)]
        placement = ((self._artifact("intake/placement.json") or {}).get("providers")) or {}
        lab_only = set((((cc or {}).get("placement") or {}).get("lab_only")) or [])
        overrides = (cc or {}).get("licence_overrides")
        overrides = overrides if isinstance(overrides, dict) else {}
        nt, nt_err = self._never_train()
        by = {}
        for r in known + rows:
            cid = str(r.get("source_id") or r.get("id") or r.get("source") or r.get("slug") or "")
            if not cid:
                continue
            cur = by.get(cid, {})
            cur.update({k: v for k, v in r.items() if v is not None})
            cur["id"] = cid
            cur["known_item"] = bool(cur.get("known_item") or r.get("known_item"))
            by[cid] = cur
        out = []
        for cid in sorted(by):
            r = by[cid]
            classes = r.get("target_classes")
            if classes is None:
                classes = [c.get("name") for c in r.get("classes_mapped") or []
                           if isinstance(c, dict) and c.get("status") in ("target", "target_synonym")]
            lic = r.get("licence", r.get("license"))
            lic_ok = r.get("licence_ok")
            lcls = lic.get("class") if isinstance(lic, dict) else None
            if isinstance(lic, dict):
                if lic_ok is None and lic.get("class") in ("refused",):
                    lic_ok = False
                lic = lic.get("id") if lic.get("class") not in (None, "unresolved") else None
            if lic is None and lic_ok is True and r.get("licence_override"):
                lic = r.get("licence_id")        # a person's licence override, as collect.plan records it
            # a person's licence override the collect config records, read here as _fold reads it: a candidates
            # file plan wrote before the override still says unresolved, and its pre-check still holds
            # licence_unresolved, until the next L15; an unresolved licence only, never a refused one
            ov = overrides.get(cid)
            by_ov = isinstance(ov, dict) and bool(ov.get("id")) and lcls == "unresolved" and lic_ok is not False
            if by_ov and lic is None:
                lic, lic_ok = ov.get("id"), True
            lifted = ("licence_unresolved",) if by_ov else ()
            s = (self.st.get("sources") or {}).get(cid) or {}
            prov = r.get("provider")
            pr = (placement.get(prov) or {}) if isinstance(placement, dict) else {}
            on_cluster = pr.get("placement") == "cluster" if "placement" in pr else pr.get("status") == "pass"
            pl = "lab" if prov in lab_only or pr.get("lab_only") or not on_cluster else "cluster"
            pc = r.get("precheck") if isinstance(r.get("precheck"), dict) else {}
            fails = [f for f in pc.get("failures") or [] if isinstance(f, dict)]
            names_pending = bool(r.get("names_pending")) or (r.get("decision") or {}).get("status") == "pending_names" \
                or any(f.get("code") == "names_pending" for f in fails)
            out.append({"id": cid, "provider": prov, "licence": lic, "licence_ok": lic_ok,
                        "target_classes": sorted(set(c for c in classes or [] if c)),
                        "bytes": r.get("bytes"), "images": r.get("images"),
                        "expected_target_boxes": r.get("expected_target_boxes"), "target_boxes": r.get("target_boxes"),
                        "lab_group": r.get("lab_group"), "evaluation_lab": bool(r.get("evaluation_lab")),
                        "copy_scan_done": bool(r.get("copy_scan_done")),
                        "credentials": r.get("credentials_ok", r.get("credentials")),
                        "image_level": bool(r.get("image_level") or r.get("annotation_type") in ("image-level",
                                                                                                "image_level")
                                            or r.get("annotation") == "image_level"),
                        "never_train": (nt is None) or cid in (nt or set()), "known_item": r["known_item"],
                        "status": s.get("status"), "partial": bool(s.get("partial")), "placement": pl,
                        "attempts": int(s.get("attempts") or 0),
                        "names_unresolved": (bool(r.get("names_unresolved")) or names_pending)
                        and not s.get("names_resolved"),
                        "names_resolved": bool(s.get("names_resolved")),
                        "collector_precheck": {"refuse": sorted(f.get("code") for f in fails
                                                                if f.get("action") in ("close", "refuse")
                                                                and f.get("code") != "names_pending"),
                                               "review": sorted(f.get("code") for f in fails
                                                                if f.get("action") == "hold" and f.get("risk") == "R3"
                                                                and f.get("code") not in lifted),
                                               "wait": sorted(f.get("code") for f in fails
                                                              if f.get("action") == "hold" and f.get("risk") != "R3"
                                                              and f.get("code") != "names_pending"
                                                              and f.get("code") not in lifted)}})
        if nt is None:
            self._once("never_train_unreadable", nt_err, "never_train_unreadable", reasons=[nt_err])
            self._card("data", "The never-train list cannot be read: no source is collected", nt_err)
        return out

    # ---- evidence
    def _artifact(self, name):
        p = _read_json(self.paths.latest(self.name)) or {}
        arts = (((p.get("stream") or {}).get("decision") or {}).get("artifacts")) or {}
        return arts.get(name)

    def _record(self, payload):
        """The campaign-snapshot-shaped record evidence.from_snapshot reads: the
        live experiments' decision parts, the frozen ones, and the stream summary
        with the cached dev scores merged in. display_only never enters it."""
        exps = {}
        for exp, sub in ((payload or {}).get("experiments") or {}).items():
            snap = (sub or {}).get("snapshot")
            if isinstance(snap, dict) and snap.get("built", True) is not False:
                exps[exp] = {"snapshot": {k: v for k, v in snap.items() if k != "display_only"}}
        for exp in sorted(self.st.get("frozen") or {}):
            if exp in exps:
                continue
            fz = _read_json(self.paths.frozen(self.name, exp))
            if isinstance(fz, dict) and isinstance(fz.get("snapshot"), dict):
                exps[exp] = {"snapshot": fz["snapshot"]}
        stream = copy.deepcopy((payload or {}).get("stream") or {"verb": "stream-summary", "decision": {"artifacts": {}}})
        stream["verb"] = "stream-summary"
        stream.setdefault("decision", {}).setdefault("artifacts", {})
        # the tick's clock at which the snapshot was taken, not the cluster's wall
        # clock: the evidence's bytes (and the digest) must not change with it
        return {"verb": "campaign-snapshot", "utc": self.st.get("last_snapshot_utc"), "experiments": exps,
                "stream": stream}

    def _build_evidence(self):
        payload = _read_json(self.paths.latest(self.name)) or {}
        ctx = self._context(payload)
        try:
            self.ev = E.from_snapshot(self._record(payload), self.sid, context=ctx,
                                      domain=LS.funnel_ref(self.dom))
        except (E.EvidenceError, KeyError, TypeError, ValueError) as e:
            self._once("evidence_error", str(e), "evidence_error", error=_short(e, 500))
            self.ev = E.from_texts({}, self.sid, context=ctx, domain=LS.funnel_ref(self.dom))

    def _stage(self, payload):
        """The rollout's state as the diagnoses read it (context /stage): the
        lock, every tracked experiment's status, the baselines, Stage A and C,
        the verdicts the platform has run, Step 1's one-time jobs."""
        st = self.st
        r0 = (st.get("stage") or {}).get("r0") or {}
        lock = self._artifact("splits/v2/lock_status.json") or {}
        status = {x.get("exp"): x for x in (((payload or {}).get("status") or {}).get("experiments") or [])
                  if isinstance(x, dict)}
        exp_status = {e: ("done" if x.get("done") else "built" if x.get("built") else "missing")
                      for e, x in sorted(status.items()) if e}
        base = {}
        for b in (self.dom.get("baselines") or {}).get("items") or []:
            x = exp_status.get(b["exp"])
            base[b["id"]] = x if x in ("done", "built") else (
                "building" if r0.get("baseline_%s" % b["id"]) == "building" else "missing")
        sa_exp = (self.dom.get("stage_a") or {}).get("exp")
        sc_exp = "%s_c001" % self.sid
        s1 = (self._artifact("step1_stream/status.json") or {}).get("one_time") or {}
        # a measurement arm's native-resolution rescore (L23N): done once its record says complete, else what
        # the platform ran (running, done, failed), else missing
        native = {}
        for b in (self.dom.get("baselines") or {}).get("items") or []:
            if b.get("measure"):
                rec = self._artifact("%s/native_rescore.json" % b["exp"]) or {}
                native[b["id"]] = "done" if rec.get("status") == "complete" else (
                    r0.get("native_%s" % b["id"]) or "missing")

        def state_of(exp, key):
            x = exp_status.get(exp)
            return x if x in ("done", "built") else ("building" if r0.get(key) == "building" else "missing")
        return {"lock": bool(lock.get("locked")), "splits_built": bool(r0.get("splits_built")),
                "train_manifests": lock.get("train_manifests") or [], "baselines": base, "native": native,
                "exp_status": exp_status, "verdicts": dict(r0.get("verdicts") or {}),
                "protocol_v3_accepted": bool(self.cfg.get("protocol_v3_accepted_by")),
                "stage_a": {"exp": sa_exp, "status": state_of(sa_exp, "stage_a")},
                "stage_a_ready": bool((st.get("stage_a") or {}).get("ready")) or (self._artifact("%s/%s" % (
                    sa_exp, (self.dom.get("stage_a") or {}).get("record") or "stage_a.json")) or {}).get("status")
                == "READY",
                "stage_c": {"exp": sc_exp, "status": state_of(sc_exp, "stage_c")},
                "stage_c_submitted": bool(r0.get("stage_c_built")),
                "stage_c_decided": bool(st.get("stage_c")), "capacity": (st.get("capacity") or {}).get("chosen"),
                "placement": bool(self._artifact("intake/placement.json")), "probe_ran": bool(r0.get("probe")),
                "step1_stream": {k: bool(s1.get(k)) for k in ("bootstrap", "knowntruth", "backfill")},
                "prospective": bool(st.get("prospective"))}

    def _context(self, payload):
        st = self.st
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        Mv = int(q.get("M") or (self.dom.get("increment") or {}).get("M") or 0)
        pool = q.get("pool") or {}
        bud = None
        try:
            bud = X.budget_now(self.camp, self.xctx)
        except Exception:
            bud = None
        al = {}
        prj = (payload or {}).get("projects") or {}
        pat = re.compile(str((self.dom.get("allocation") or {}).get("resource_pattern") or "GPU"), re.I)
        blk = [r for r in prj.get("resources") or [] if isinstance(r, dict) and pat.search(str(r.get("resource")))]
        al["balance_su"] = blk[0].get("balance_su") if blk else None
        al["end_date"] = (blk[0].get("end_date") if blk and blk[0].get("end_date") else
                          (self.dom.get("allocation") or {}).get("end_date"))
        al["reserve_su"] = self.cfg.get("alloc_reserve_su")
        # what every campaign has committed and not yet spent: the balance the
        # `projects` output reports already has the spent SU taken off, so
        # spent SU is not subtracted a second time (D27)
        al["committed_all_su"] = (bud or {}).get("domain_committed_su")
        dl = {}
        try:
            from ..brain import su_ledger as SL
            r = SL.remaining(self.domain, B.domain_budget(self.domain_budget)[0], base_dir=self.xctx.su_base_dir)
            dl = {k: r.get(k) for k in ("campaign_su", "envelope", "remaining_su")}
        except Exception as e:                       # unknown is never "enough": reported, not assumed
            dl = {"error": "%s: %s" % (type(e).__name__, _short(e, 200))}
        qu = dict((payload or {}).get("quota") or {})
        qu["staging_gb"] = sum((_num(s.get("bytes")) or 0.0) for s in (st.get("sources") or {}).values()
                               if s.get("status") in ("fetched", "intaken")) / 1e9
        lanes = {}
        for ln in LANES:
            lane = st["lanes"][ln]
            lanes[ln] = {"phase": lane["phase"], "busy": lane.get("item") is not None,
                         "hold": lane.get("hold") or lane.get("diag_hold"), "half_cadence": bool(lane.get("half"))}
        arm = self._arm()
        builds = [_num(v) for v in (st.get("build_hours") or [])]
        return {"now_utc": self.utc, "sid": self.sid, "M": Mv, "K_max": LS.t(self.th, "D22", "K_max"),
                "lanes": lanes, "queue": {"Q": 0}, "candidates": self._candidates(),
                "discover": dict(st.get("discover") or {}), "sources": copy.deepcopy(st.get("sources") or {}),
                "refusals": list(st.get("refusals") or []),
                "budget": {k: (bud or {}).get(k) for k in ("envelope_su", "spent_su", "committed_su", "remaining_su",
                                                          "window_remaining_su", "domain_remaining_su")},
                "allocation": al, "quota": qu, "collect_gb_envelope": self.cfg.get("collect_gb_envelope"),
                "domain_ledger": dl,
                "stage": self._stage(payload), "segments": copy.deepcopy(st.get("segments") or []),
                "milestones": copy.deepcopy(st.get("milestones") or []),
                "milestone0": self._milestone0(),
                "pool": {"images": pool.get("images") or (self.dom.get("increment") or {}).get("base_images"),
                         "id": pool.get("current")},
                "arm": arm, "recipes": [r for r in self._recipes().split(",") if r],
                "build_hours_max": max([b for b in builds if b is not None] or [None])
                if [b for b in builds if b is not None] else None,
                "d33_history": list(st.get("d33_history") or []),
                "d33_species": sorted({s for h in st.get("d33_history") or [] if h.get("fired")
                                       for s in h.get("species") or []}),
                "doublings": int(st.get("doublings") or 0), "bisected": st.get("bisected"),
                "rollbacks": copy.deepcopy(st.get("rollbacks") or []),
                "limits": {"gb_per_source": (LS.limits("L16") or {}).get("gb_per_source"),
                           "attempts_per_source": (LS.limits("L16") or {}).get("attempts_per_source")}}

    def _milestone0(self):
        """Milestone 0: the stream's own (choose-arm adopts the capacity
        decision's chosen experiment), else the default arm's baseline."""
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        rec = (((q.get("milestones") or {}).get("records") or {}).get("0")) or {}
        if rec.get("exp"):
            return rec["exp"]
        cap = self.dom.get("capacity") or {}
        return ((cap.get("arms") or {}).get(cap.get("default_arm")) or {}).get("exp")

    def _arm(self):
        """The capacity arm the stream runs ({name, cost_factor, exp}): the
        capacity decision's chosen arm once recorded, else the default."""
        cap = self.dom.get("capacity") or {}
        name = (self.st.get("capacity") or {}).get("chosen") or cap.get("default_arm")
        return dict(((cap.get("arms") or {}).get(name) or {}), name=name)

    # ---- diagnoses, stop-losses, holds
    def _diagnose(self):
        st = self.st
        diags = DS.detect(self.ev, self.dom, self.th, prior=self.prior)
        self.diags = diags + list(st.get("health") or [])
        fired = [d for d in diags if d.get("fired")]
        _write_json(self.paths.diagnoses(self.name), {"utc": self.utc, "sid": self.sid, "diagnoses": diags,
                                                     "rules_version": DS.rules_version()})
        self._once("diagnosed", [(d["id"], d.get("summary")) for d in fired], "diagnosed",
                   fired=[{"id": d["id"], "name": d.get("name"), "summary": d.get("summary"),
                           "cites": d.get("cites")} for d in fired], rules_version=DS.rules_version(),
                   digest_sha256=self.digest())
        by = DS.by_id(diags)
        # stop-losses that pause
        for did in ("D10S", "D27"):
            d = by.get(did) or {}
            if d.get("fired") and "OP_PAUSE" in (d.get("levers") or []):
                if "X14" in (d.get("levers") or []):
                    self._card("budget", "Budget or allocation: %s" % _short(d.get("summary"), 120),
                               d.get("summary"), lever="X14", trigger=[did])
                return self._pause("%s (%s): %s" % ((d.get("detail") or {}).get("reason") or did, did,
                                                    _short(d.get("summary"), 400)))
        # cards and holds
        for d in fired:
            for lv in d.get("levers") or []:
                if str(lv).startswith("X"):
                    card = LS.card(lv) or {}
                    title = card.get("title") or lv
                    if not card:
                        from . import levers as LV
                        title = ((LV.load_menu().get("cards") or {}).get(lv) or {}).get("title") or lv
                    self._card("research", "%s: %s" % (lv, title), d.get("summary"), lever=lv, trigger=[d["id"]])
            if "OP_ESCALATE" in (d.get("levers") or []) or "OP_CARD" in (d.get("levers") or []):
                self._card("escalation", "%s %s" % (d["id"], d.get("name")), d.get("summary"), trigger=[d["id"]])
        holds = {ln: None for ln in LANES}
        for did, hold_of in (("D26", None), ("D30", "TRAIN"), ("D32", "TRAIN"), ("D25", None), ("D33", None),
                             ("D22", None), ("D27", None), ("DCAN", "TRAIN"), ("D28", None)):
            d = by.get(did) or {}
            if not d.get("fired"):
                continue
            det = d.get("detail") or {}
            h = det.get("hold") if hold_of is None else hold_of
            if did == "D22" and det.get("refused"):
                h = "TRAIN"
            for ln in ([h] if isinstance(h, str) else list(h or [])):
                if ln in holds and holds[ln] is None:
                    holds[ln] = "%s %s" % (did, d.get("name"))
        # D33's history (consecutive segments) and D21's closures
        d33 = by.get("D33") or {}
        if d33.get("fired") and d33.get("exp") and not (d33.get("detail") or {}).get("prospective"):
            hist = [h for h in st.get("d33_history") or [] if h.get("exp") != d33["exp"]]
            hist.append({"exp": d33["exp"], "fired": True, "species": (d33.get("detail") or {}).get("species")})
            st["d33_history"] = hist[-12:]
        d28 = by.get("D28") or {}
        for h in ((d28.get("detail") or {}).get("leaks") or []) if d28.get("fired") else []:
            s = st["sources"].setdefault(str(h.get("source")), {})
            if s.get("status") not in ("closed", "quarantined", "held"):
                s.update(status="held", held_reason="source leak (D28): quarantine pending")
                self._ledger("source_held", source=h.get("source"), reasons=["D28 leak"], trigger=["D28"])
        d21 = by.get("D21") or {}
        for c in ((d21.get("detail") or {}).get("close") or []) if d21.get("fired") else []:
            s = st["sources"].setdefault(c["source"], {})
            if s.get("status") not in ("closed", "quarantined"):
                s.update(status="closed", closed_reason=c.get("why"), closed_utc=self.utc)
                self._ledger("source_closed", source=c["source"], reasons=[c.get("why")], trigger=["D21"])
        d31 = by.get("D31") or {}
        for src, n in (((d31.get("detail") or {}).get("per_source") or {}).items() if d31.get("fired") else []):
            st["sources"].setdefault(src, {})["blames"] = int(n)
        d27 = by.get("D27") or {}
        st["want_largest"] = bool(d27.get("fired") and (d27.get("detail") or {}).get("hold") == "DATA")
        if st["want_largest"] and (d27.get("detail") or {}).get("largest"):
            self._card("quota", "/ocean is short of space for collection: a person frees space",
                       "largest directories: %s" % ", ".join("%s %.0f GB" % (x.get("dir"), x.get("gb") or 0)
                                                            for x in (d27["detail"]["largest"] or [])[:10]),
                       trigger=["D27"])
        d29 = by.get("D29") or {}
        lane = st["lanes"]["DATA"]
        if d29.get("fired") and lane.get("item") is None and lane["phase"] != "WAIT_DATA":
            days = int((d29.get("detail") or {}).get("wait_days") or 7)
            lane.update(phase="WAIT_DATA", until_utc=_utc(self.now + days * 86400))
            self._card("wait_data", "Collection exhausted: waiting %d days for new sources" % days, d29.get("summary"),
                       trigger=["D29"])
            self._ledger("wait_data", days=days, until_utc=lane["until_utc"], trigger=["D29"])
        d32 = by.get("D32") or {}
        st["lanes"]["DATA"]["half"] = bool(d32.get("fired"))
        for ln in LANES:
            old = st["lanes"][ln].get("diag_hold")
            st["lanes"][ln]["diag_hold"] = holds[ln]
            if holds[ln] != old:
                self._ledger("lane_hold" if holds[ln] else "lane_hold_cleared", lane=ln, hold=holds[ln] or old)
        # R0 records: Stage A, Stage C, the capacity decision
        dsa = by.get("DSA") or {}
        if dsa.get("fired") and not (st.get("stage_a") or {}).get("ready"):
            self._record_stage_a(dsa)
        dsc = by.get("DSC") or {}
        if dsc.get("fired") and not st.get("stage_c"):
            st["stage_c"] = {k: v for k, v in (dsc.get("detail") or {}).items() if k != "cites"}
            self._ledger("stage_c", verdict=st["stage_c"], trigger=["DSC"])
        dcap = by.get("DCAP") or {}
        if dcap.get("fired") and not st.get("capacity"):
            st["capacity"] = dict(dcap.get("detail") or {})
            self._ledger("capacity_decision", decision=st["capacity"], trigger=["DCAP"],
                         reasons=["L-4, as inc2.baseline capacity-verdict recorded it (dev only): an arm qualifies "
                                  "when its dev mean beats n640's by more than 2 pooled sd"])
        held = [ln for ln in LANES if st["lanes"][ln].get("hold")]
        if len(held) >= int(LS.t(self.th, "stop_loss", "held_lanes_to_pause")):
            return self._pause("stop-loss: %d lanes are held (%s)" % (len(held), ", ".join(
                "%s: %s" % (ln, st["lanes"][ln]["hold"]) for ln in held)))

    def digest(self):
        """The sha256 of everything a decision read (the evidence's canonical
        bytes); test blindness (S12) keeps it identical when every test and
        non-decision exam value changes."""
        return hashlib.sha256(self.ev.canonical()).hexdigest() if self.ev is not None else None

    def _record_stage_a(self, dsa):
        st = self.st
        rec = {"format": "inc-autopilot/prospective-stage-a/1", "sid": self.sid, "ready": True,
               "detail": dsa.get("detail"), "rules_version": DS.rules_version(), "utc": self.utc,
               "cites": dsa.get("cites")}
        path = self.paths.replay / ("prospective_stage_a_%s__%s.json" % (self.sid, DS.rules_version()))
        _write_json(path, rec)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        st["stage_a"] = {"ready": True, "recipes": (dsa.get("detail") or {}).get("recipes"), "path": str(path),
                         "sha256": sha, "chosen": (dsa.get("detail") or {}).get("chosen")}
        self._ledger("stage_a_ready", path=str(path), sha256=sha, recipes=st["stage_a"]["recipes"],
                     survivors=(dsa.get("detail") or {}).get("survivors"), trigger=["DSA"])

    # ---- lanes: from diagnoses to items
    def _train_ready(self):
        """"" when the TRAIN lane may cut a segment (R0 READY, R0b and R1), else why not."""
        stg = self._stage(_read_json(self.paths.latest(self.name)) or {})
        entries = [e for e in (self._artifact("stream/%s/ledger.jsonl" % self.sid) or []) if isinstance(e, dict)]
        led = [e.get("event") for e in entries]
        miss = []
        # a commit (L19) shows in the evidence only at the next snapshot, and a
        # tick that submits takes no snapshot: the next segment is not cut
        # until the diagnoses that read the last commit (D8S, D30, D32, D33,
        # D31) have read it, or their hold would come one segment late
        seen = {e.get("exp") for e in entries if e.get("event") == "commit"}
        unseen = [s["exp"] for s in self.st.get("segments") or [] if s.get("committed") and s["exp"] not in seen]
        if unseen:
            miss.append("the commit of %s is not yet in the evidence (the next snapshot shows it)" % ", ".join(unseen))
        if not stg["lock"]:
            miss.append("splits v2 not locked")
        m0 = self._milestone0()
        if m0 and stg["exp_status"].get(m0) != "done":
            miss.append("milestone 0 (%s) not done" % m0)
        can = next((b for b in (self.dom.get("baselines") or {}).get("items") or []
                    if b.get("verdict") == "canary-verdict"), None)
        if can and (self._artifact("%s/canary.json" % can["exp"]) or {}).get("passed") is not True:
            miss.append("the canary has not passed")
        if not stg["stage_a_ready"]:
            miss.append("Stage A not READY")
        chosen = (self.st.get("capacity") or {}).get("chosen")
        if not chosen:
            miss.append("no capacity decision (L-4)")
        elif self.ev is not None and ((self.ev.json(E.CONTEXT) or {}).get("arm") or {}).get("name") != chosen:
            # recorded this tick, after the evidence was built: the walltime and
            # price rules (D26) read the arm from the evidence, so the first cut
            # waits one tick for evidence that carries the chosen arm
            miss.append("the capacity decision is newer than this tick's evidence")
        if "init" not in led:
            miss.append("the stream is not created")
        elif "arm" not in led:
            miss.append("the stream has not adopted the capacity decision")
        if not self.st.get("stage_c"):
            miss.append("Stage C not decided")
        if not stg["step1_stream"].get("bootstrap"):
            miss.append("step1_stream not bootstrapped")
        return "; ".join(miss)

    def _prospective_guard(self):
        """The prospective stream record of the current stream rules version
        (written once, before the first L18); "" when it holds."""
        st = self.st
        ver = DS.rules_version()
        rec = st.get("prospective") or {}
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        # one record per rules version AND per stream version: a fork (L22)
        # doubles M under a new sid, and its first L18 needs its own record
        if rec.get("rules_version") == ver and rec.get("sha256") and rec.get("sid", self.sid) == self.sid \
                and rec.get("M", q.get("M")) == q.get("M"):
            return ""
        body = DS.prospective_record(self.dom, self.th, self.cfg, {"stage_a": st.get("stage_a"),
                                                                  "capacity": st.get("capacity"),
                                                                  "M": q.get("M")}, self.utc)
        path = self.paths.replay / ("prospective_stream_%s__%s.json" % (self.sid, ver))
        if not path.is_file():
            _write_json(path, body)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        st["prospective"] = {"path": str(path), "sha256": sha, "rules_version": ver, "record_sha256": body["sha256"],
                             "sid": self.sid, "M": q.get("M")}
        self._ledger("prospective_stream", path=str(path), sha256=sha, rules_version=ver,
                     record_sha256=body["sha256"], M=body["M"], K_max=body["K_max"])
        return ""

    def _intents(self):
        """[(lane, lever, params, trigger id, extra)] from the fired diagnoses, in
        detect's order (stops and TRAIN first, then DATA, then MAINT's R0)."""
        out = []
        for d in self._fired():
            det = d.get("detail") or {}
            did = d["id"]
            props = det.get("propose")
            props = props if isinstance(props, list) else ([props] if isinstance(props, dict) else [])
            if did == "DR0":
                for ln, x in sorted((det.get("due") or {}).items()):
                    props.append(dict(x))
            if did == "DHOLD":
                for h in det.get("holds") or []:
                    props.append({"lever": h["lever"], "hold": h["hold"],
                                  **({"verb": "scan-holds"} if h["lever"] == "L17" else {})})
            for pr in props:
                lever = pr.get("lever")
                if lever in LANE_OF:
                    out.append((LANE_OF[lever], lever, pr, did, d))
        return out

    def _materialise(self):
        st = self.st
        if st.get("drift"):
            return
        self._adopt_person_items()
        for lane_name, lever, pr, did, d in self._intents():
            if lever in PERSON_LEVERS:
                self._file_person(lane_name, lever, pr, did, d)
                continue
            lane = st["lanes"][lane_name]
            if lane.get("item") is not None or lane.get("hold") or lane.get("diag_hold"):
                continue
            if lane_name == "DATA" and lane.get("half") and st["ticks"] % 2:
                continue
            if lane_name == "DATA" and lane["phase"] == "WAIT_DATA":
                if lever == "L15" and self.now < (_secs(lane.get("until_utc")) or 0):
                    continue
                if lever != "L15":
                    lane["phase"] = "IDLE"
            if lane_name == "TRAIN" and lever == "L18":
                why = self._train_ready()
                if why:
                    self._once("train_wait", why, "not_taken", lever="L18", reasons=["R0 not READY: " + why])
                    continue
                self._prospective_guard()
            try:
                p = self._proposal(lever, pr, did, d)
            except (LS.LeverError, ValueError, KeyError) as e:
                self._once("render:%s:%s" % (lever, did), str(e), "not_taken", lever=lever,
                           reasons=["cannot render %s: %s" % (lever, _short(e, 300))])
                continue
            if p is None:
                continue
            if p["id"] in (st.get("declined") or []):
                continue
            if self._step_exhausted(lane_name, lane, p):
                continue
            done_at = (st.get("done_keys") or {}).get(_sha([p.get("lever") or lever, p.get("params") or {}])[:16])
            # done at or after the snapshot the evidence came from (an item the
            # fold of that snapshot finished may not show its effect in it)
            if done_at and done_at >= str(st.get("last_snapshot_utc") or ""):
                # done since the evidence was taken: the next snapshot shows its effect first
                self._once("stale:%s" % p["id"], done_at, "not_taken", lever=lever,
                           reasons=["%s ran at %s, after the last snapshot: waiting for a snapshot that shows it"
                                    % (lever, done_at)])
                continue
            lever = p.get("lever") or lever           # the sub-lever the proposal took (L16 -> L16L on the lab)
            lane["item"] = {"proposal": p, "lever": lever, "status": "proposed", "since_utc": self.utc,
                            "attempt": p.get("attempt", 0), "trigger": did, "diagnoses": [d]}
            lane["phase"] = self._phase_of(lever)
            self._ledger("proposed", lane=lane_name, lever=lever, action=p["policy_action"], risk=p["risk"],
                         argv=p["argv"], est_gpu_hours=p.get("est_gpu_hours"), trigger=[did],
                         cites=p.get("cites"), proposal_id=p["id"], child_exp=p.get("child_exp"))
        self._file_reviews()

    @staticmethod
    def _phase_of(lever):
        return {"L15": "DISCOVER", "L26": "NAMES", "LP": "PROBE", "L16": "COLLECT", "L16L": "COLLECT",
                "L16R": "COLLECT", "L16RL": "COLLECT", "L16S": "SYNC", "L16I": "INTAKE", "L17": "ADMIT",
                "L24": "QUARANTINE", "LH": "HOLD_RELEASE", "L18": "SEGMENT", "L19": "COMMIT", "L22": "FORK",
                "L20": "MILESTONE", "L21": "ROLLBACK", "L23": "SPLITS", "L23B": "BASELINE", "L23N": "NATIVE",
                "L25": "STAGE_A", "L27": "BISECT",
                "L28": "STAGE_C", "L4": "AUDIT", "LV": "VERDICT", "LI": "INIT", "LA": "ARM",
                "LC": "COMPARE"}.get(lever, "ITEM")

    def _failed_ids(self):
        """Proposal ids of this campaign's lane items that ended failed (state
        failed_ids). A state written before the list existed takes them once
        from the campaign ledger's 'failed' entries, so a step that failed
        before this record existed is not proposed under its old id either."""
        st = self.st
        if not isinstance(st.get("failed_ids"), list):
            ids = []
            try:
                with open(str(self.paths.ledger), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if '"failed"' not in line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("campaign") == self.name and rec.get("event") == "failed" \
                                and rec.get("proposal_id") and rec["proposal_id"] not in ids:
                            ids.append(rec["proposal_id"])
            except OSError:
                pass
            st["failed_ids"] = ids[-FAILED_IDS_KEEP:]
        return list(st["failed_ids"])

    def _step_exhausted(self, ln, lane, p):
        """True, with lane `ln` held for a person, when the step of proposal p
        (step_key: its lever and policy params) has failed more than
        stop_loss step_retries times: it is proposed again at most that many
        times after a failure. A person's release (stream release, or
        enable) frees the lane and clears the lane's step counts. The
        collection levers are bounded per source instead (SOURCE_BOUND)."""
        if p.get("lever") in SOURCE_BOUND:
            return False
        key = step_key(p.get("lever"), p.get("params"))
        rec = (self.st.get("step_failures") or {}).get(key) or {}
        n = int(rec.get("n") or 0)
        retries = int(LS.t(self.th, "stop_loss", "step_retries"))
        if n <= retries:
            return False
        lane["hold"] = ("stop-loss: %s failed %d times (last: %s); a step is proposed again at most %d times after a "
                        "failure, so a person decides whether to run it again: stream release --name %s --by "
                        "human:<email>" % (p.get("lever"), n, _short(rec.get("last"), 200), retries, self.name))
        lane["hold_utc"] = self.utc
        self._card("stop_loss", "Lane %s held: %s failed %d times" % (ln, p.get("lever"), n), lane["hold"],
                   lever=p.get("lever"))
        self._ledger("lane_held", lane=ln, hold=lane["hold"], lever=p.get("lever"), step=key,
                     proposal_id=rec.get("proposal_id"))
        return True

    def _release_steps(self, ln):
        """A person released lane `ln`: its steps' failure counts start again."""
        sf = self.st.get("step_failures") or {}
        for k in [k for k, v in sf.items() if (v or {}).get("lane") == ln]:
            sf.pop(k, None)

    def _next_name(self, kind):
        taken = set(self.st.get("exps") or [])
        n = 1
        while "%s_%s%03d" % (self.sid, kind, n) in taken:
            n += 1
        return "%s_%s%03d" % (self.sid, kind, n)

    def _proposal(self, lever, pr, did, d):
        """The lane item's proposal for lever with the diagnosis's params."""
        st, dom = self.st, self.dom
        cites = list(d.get("cites") or [])
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        arm = self._arm()
        info = {"pool_images": (self.ev.json(E.CONTEXT) or {}).get("pool", {}).get("images"), "M": q.get("M"),
                "arm": arm.get("name")}
        params, child, parent, lid = {}, None, None, lever
        pkg = self.pkg
        lab = str(self.paths.lab_inc)
        if lever == "L15":
            cp, _cc = self._collect_config()
            params = {"config": str(cp), "classes": ",".join(pr.get("classes") or []),
                      "out": str(self.paths.candidates())}
        elif lever == "L16":
            src = pr["source"]
            s = st["sources"].get(src) or {}
            if s.get("status") in ("closed", "quarantined", "held"):
                return None                       # closed this tick (D21, D28, a retry-false refusal)
            if int(s.get("attempts") or 0) >= int((LS.limits("L16") or {}).get("attempts_per_source") or 3):
                # a 4th attempt on one source is a loop, not a retry (S15)
                self._pause("stop-loss: a 4th collection attempt on source %s was about to be proposed" % src)
                return None
            cap = int(float((LS.limits("L16") or {}).get("gb_per_source") or 50) * 1e9)
            want = int(_num(pr.get("bytes")) or (dom.get("collect") or {}).get("max_bytes_default") or cap)
            params = {"source": src, "max_bytes": max(1, min(want, cap))}
            if pr.get("placement") == "lab":
                lid = "L16L"
                params["out"] = str(self.paths.staging()) + "/"
            elif s.get("names_resolved") and not s.get("names_synced"):
                # its names were resolved on the lab (L26): the cluster's
                # collector reads them offline, so the layer is pushed first
                lid = "L16S"
                params = {"source": src, "names": 1}
            elif not pr.get("known_item"):
                # a discovered source fetched on the cluster: the collector there
                # reads its candidate record from the file the lab synced first
                if not s.get("candidate_synced"):
                    lid = "L16S"
                    params = {"source": src, "candidate": 1}
                else:
                    params["candidates"] = s["candidate_synced"]
        elif lever == "L26":
            params = {"source": pr["source"], "out": str(self.paths.names_dir()) + "/"}
        elif lever == "L16S":
            params = {"source": pr["source"]}
        elif lever == "L16I":
            params = {"source": pr["source"]}
        elif lever == "L17":
            params = {"verb": pr.get("verb") or "admit"}
            if pr.get("intake"):
                params["intake"] = pr["intake"]
                imgs = ((self.ev.json("intake/%s/summary.json" % pr["intake"]) or {}).get("images"))
                info["images"] = imgs
            if pr.get("hold"):
                params["hold"] = pr["hold"]
            files = (((_read_json(self.paths.latest(self.name)) or {}).get("stream") or {}).get("files")) or {}
            ver = (self.ev.json("step1_stream/status.json") or {}).get("versions") or {}
            pins = {"verifier": ver.get("verifier"),
                    "lock_sha256": ver.get("splits_lock_sha256") or (files.get("splits/v2/lock_status.json") or {})
                    .get("sha256")}
            first = st.setdefault("pins", {})
            for k, v in pins.items():
                if v and first.get(k) and first[k] != v:
                    self._card("research", "X11: %s changed (%s -> %s)" % (k, first[k][:12], v[:12]),
                               "a verifier refit or a LOCK change is a versioned event (R4): no batch is admitted "
                               "under it until a person decides", lever="X11", trigger=[did])
                    return None
                first.setdefault(k, v)
            info["pins"] = pins
        elif lever == "L24":
            params = {"pkg": pkg, "source": pr["source"], "cite": pr.get("cite") or did}
            src_st = st["sources"].get(pr["source"]) or {}
            if src_st.get("status") == "quarantined" or src_st.get("leak_kept_by"):
                return None
        elif lever == "LH":
            params = {"pkg": pkg, "stream": self.sid, "hold": pr.get("hold") or "funnel_F9"}
        elif lever == "L18":
            child = pr.get("exp") or q.get("next_segment") or self._next_name("s")
            params = {"pkg": pkg, "stream": self.sid, "k": int(pr.get("k") or 1), "exp": child}
            recipes = self._recipes()
            M_ = int(q.get("M") or dom["increment"]["M"])
            every = LS.truth_every(dom, self.th, int(info["pool_images"] or dom["increment"]["base_images"]), M_,
                                   tuple(recipes.split(",")), float(arm.get("cost_factor") or 1.0))
            # the stream's own truth cadence (L-4, per segment) decides; this is the price's
            info.update(recipes=recipes, truth_every=every)
        elif lever == "L19":
            params = {"pkg": pkg, "exp": pr["exp"]}
        elif lever == "L22":
            params = {"pkg": pkg, "stream": self.sid, "m": int(pr["m"])}
        elif lever == "L20":
            child = pr.get("exp") or ((q.get("milestones") or {}).get("next")) or self._next_name("m")
            params = {"pkg": pkg, "stream": self.sid}
        elif lever == "L21":
            params = {"pkg": pkg, "stream": self.sid, "to": pr.get("to") or "P_0"}
        elif lever == "L27":
            child = self._next_bisect_arm()
            params = {"pkg": pkg, "stream": self.sid, "from_pool": pr.get("from_pool") or "P_0"}
            info["n_suspect"] = pr.get("n_suspect")
        elif lever == "L28":
            if not q.get("M"):
                return None                       # the stream (and its M) does not exist yet
            child = "%s_c001" % self.sid
            params = {"pkg": pkg, "stream": self.sid, "holdout": (dom.get("stage_c") or {}).get("holdout"),
                      "m": int(q["M"])}
            info["recipes"] = ",".join(q.get("stage_b") or ["r0"])
        elif lever == "L23":
            params = {"pkg": pkg, "verb": pr.get("verb") or "build"}
        elif lever == "L23B":
            b = next(x for x in (dom.get("baselines") or {}).get("items") or [] if x["id"] == pr["baseline"])
            params = {"pkg": pkg, "exp": b["exp"], "seeds": b["seeds"], "arm": b.get("arm") or "n640",
                      "role": b["role"]}
            if b.get("union"):
                params["union"] = ",".join(LS.inc_path(x) for x in b["union"])
            else:
                params["manifest"] = LS.inc_path(b["manifest"])
            child = b["exp"]
            info["images"] = b.get("images") or (dom.get("increment") or {}).get("base_images")
            info["arm"] = b.get("arm")
        elif lever == "L23N":
            b = next(x for x in (dom.get("baselines") or {}).get("items") or []
                     if x["id"] == pr["baseline"] and x.get("measure"))
            ref = ((dom.get("capacity") or {}).get("native") or {})["reference_exp"]
            params = {"pkg": pkg, "exp": b["exp"], "reference": ref}
            parent = b["exp"]                     # it builds nothing: the job is followed by its id
            seeds = [x for x in str(b.get("seeds") or "").split(",") if x != ""]
            info["runs"] = 2 * len(seeds) if seeds else None
        elif lever == "L25":
            sa = dom.get("stage_a") or {}
            params = {"pkg": pkg, "exp": sa["exp"], "from_exp": sa["from_exp"], "recipes": sa["recipes"]}
            child, parent = sa["exp"], sa["from_exp"]
        elif lever == "LV":
            params = {"pkg": pkg, "module": pr["module"], "verb": pr["verb"]}
            if pr.get("exp"):
                params["exp"] = pr["exp"]
            info["verdict_key"] = pr.get("key")
        elif lever == "LI":
            params = {"pkg": pkg, "stream": self.sid, "stage_b": pr["stage_b"]}
        elif lever == "LA":
            params = {"pkg": pkg, "stream": self.sid}
        elif lever == "LC":
            params = {"pkg": pkg, "exp": pr["exp"]}
            parent = pr["exp"]
        elif lever == "LP":
            params = {}
        elif lever == "L4":
            return self._audit_proposal(pr, did, d)
        else:
            return None
        r = LS.row(lid)
        est = None
        if r.get("estimator") not in ("zero",):
            est, estimate = LS.price(lid, params, dom, info)
        else:
            estimate = {"estimator": "zero"}
        akey = step_key(lid, LS.policy_params(lid, params))
        attempt = int((st.get("attempts") or {}).get(akey) or 0)
        failed_ids = set(self._failed_ids())
        for _ in range(64):
            p = LS.proposal(self.name, lid, params, trigger=[did], cites=cites, lane=LANE_OF.get(lever),
                            parent_exp=parent, child_exp=child, est=est, estimate=estimate, attempt=attempt)
            if p["id"] not in failed_ids:
                break
            # the id of a lane item that ended failed is never proposed again: the
            # executor runs an id once and refuses a changed proposal under an id
            # it filed (seen live on 2026-09-28: L23 after its job failed)
            attempt += 1
        p["lever"] = lid
        if info.get("pins"):
            p["pins"] = info["pins"]
        if info.get("verdict_key"):
            p["verdict_key"] = info["verdict_key"]
        if lever == "L18":
            p["base_pool"] = {k: (q.get("pool") or {}).get(k) for k in ("current", "sha256", "images")}
        return p

    def _recipes(self):
        """The recipes the next segment runs (pricing only; inc2.stream
        decides): the chosen recipe once Stage B chose one, else the stream's
        Stage B arms (init's --stage-b, Stage A's recorded recipes)."""
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        if q.get("chosen_recipe"):
            return str(q["chosen_recipe"])
        return ",".join(q.get("stage_b") or []) or str((self.st.get("stage_a") or {}).get("recipes") or "r0")

    def _next_bisect_arm(self):
        """The first arm the next bisect build makes: <sid>_bNNN, numbered
        across rollbacks after every arm the stream ledger records."""
        n = 0
        for e in self._artifact("stream/%s/ledger.jsonl" % self.sid) or []:
            if isinstance(e, dict) and e.get("event") == "bisect" and e.get("phase") == "build":
                n += len(e.get("arms") or {})
        return "%s_b%03d" % (self.sid, n + 1)

    def _audit_proposal(self, pr, did, d):
        """L4 (the existing label audit) on a segment's data-blamed increments."""
        from . import levers as LV
        exp = pr["exp"]
        try:
            args, cites = LV.audit_args(self.ev, exp)
        except Exception as e:
            self._once("l4:%s" % exp, str(e), "not_taken", lever="L4", reasons=[_short(e, 300)])
            return None
        steps = set(pr.get("steps") or [])
        audits = [a for a in args["audits"] if a.split("=", 1)[0] in steps] or args["audits"]
        params = {"trusted": args["trusted"], "audit": ",".join(audits), "out": args["out"]}
        argv = ["sbatch", "run_inc_audit.sh", "--trusted", args["trusted"], "--audit"] + audits + ["--out", args["out"]]
        pid = _sha([self.name, "L4", argv])[:32]
        return {"id": pid, "lever": "L4", "family": "L4", "argv": argv, "params": params,
                "policy_action": "inc_label_audit", "risk": "R2", "trigger": [did],
                "cites": list(d.get("cites") or []) + cites, "lit": [], "control": "", "success": "", "falsifier": "",
                "est_gpu_hours": 0.0, "proposed_by": AUTO, "lane": "MAINT", "follow": "job",
                "parent_exp": exp, "child_exp": exp, "attempt": 0}

    def _context_candidate(self, src):
        """The source's candidate row in the evidence context (_candidates), or {}."""
        cands = ((self.ev.json(E.CONTEXT) if self.ev is not None else None) or {}).get("candidates") or []
        return next((c for c in cands if isinstance(c, dict) and c.get("id") == src), {})

    def _file_reviews(self):
        """D20's candidates that failed a pre-check: an R3 item for a person
        each, filed once, never taking a lane (6.3 L16). Its form follows the
        candidate's placement, as D20's L16 -> L16L does: L16R fetches on the
        cluster, L16RL on the lab, where the cluster's collector refuses a
        provider placement.json does not place on compute nodes
        (not_placed_on_cluster: job 47302914 ran an approved L16R of a
        lab-placed source). A review superseded because its cluster form met a
        lab-placed source (_misplaced_review), or because its cluster form
        already ran for a source now placed on the lab (_spent_cluster_review),
        is filed again, in the form its placement calls for."""
        st = self.st
        d20 = DS.by_id(self.diags).get("D20") or {}
        for r in ((d20.get("detail") or {}).get("review") or [])[:3]:
            src = r.get("source")
            if not src:
                continue
            prev = (st.get("source_reviews") or {}).get(src)
            cand = self._context_candidate(src)
            self._spent_cluster_review(src, prev, cand)
            superseded = isinstance(prev, dict) and prev.get("status") == "superseded"
            if prev is not None and not superseded:
                continue
            cap = int(float((LS.limits("L16") or {}).get("gb_per_source") or 50) * 1e9)
            params = {"source": src, "max_bytes": max(1, min(int(_num(cand.get("bytes")) or cap), cap))}
            lid = "L16R"
            if cand.get("placement") == "lab":
                lid = "L16RL"
                params["out"] = str(self.paths.staging()) + "/"
            try:
                est, estimate = LS.price(lid, params, self.dom)
                p = LS.proposal(self.name, lid, params, trigger=["D20"], cites=d20.get("cites") or [],
                                est=est, estimate=estimate)
            except LS.LeverError as e:
                self._once("review:%s" % src, str(e), "not_taken", lever=lid, reasons=[str(e)])
                continue
            p["reason"] = "pre-check failed for %s: %s; a person decides" % (src, "; ".join(r.get("reasons") or []))
            res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
            rec = {"approval_id": res.get("approval_id"), "status": res.get("status"), "reasons": r.get("reasons"),
                   "utc": self.utc, "lever": lid}
            if superseded:
                rec["superseded"] = list(prev.get("superseded") or []) + [
                    {"approval_id": prev.get("approval_id"), "lever": prev.get("lever") or "L16R",
                     "utc": prev.get("superseded_utc"), "reason": prev.get("superseded_reason")}]
            st.setdefault("source_reviews", {})[src] = rec
            if res.get("status") == "filed" and res.get("approval_id"):
                # an approval by a person is adopted into the DATA lane and run
                self._park("DATA", p, "D20", d20, res.get("approval_id"))
            self._ledger("review_filed", lever=lid, source=src, approval_id=res.get("approval_id"),
                         status=res.get("status"), reasons=r.get("reasons"))
            if any("credentials" in x for x in r.get("reasons") or []):
                self._card("research", "X16: credentials for %s" % src, "; ".join(r["reasons"]), lever="X16",
                           trigger=["D20"])

    def _spent_cluster_review(self, src, prev, cand):
        """A source's filed review in its cluster form (L16R) whose approval
        already ran, for a source whose candidate is now placed on the lab, is
        marked superseded, so that _file_reviews files its lab form (L16RL)
        when D20 lists the source again: the cluster's collector refuses such a
        provider (not_placed_on_cluster), and one approval runs once. The live
        record: approval ap-1790700040-42144765 ran as job 47302914 before the
        lab form existed, and its review stayed 'filed'. D20 lists no source
        that is being fetched, so a run still in progress is never superseded."""
        if not isinstance(prev, dict) or prev.get("status") != "filed" or (prev.get("lever") or "L16R") != "L16R" \
                or cand.get("placement") != "lab":
            return
        a = AP.state(self.domain, root=self.xctx.approvals_root).get(prev.get("approval_id")) or {}
        ex = a.get("execution") or {}
        if ex.get("phase") not in ("done", "failed"):
            return
        why = ("its cluster form (L16R, approval %s) already ran (%s by %s, outcome %s), and its provider %s is "
               "placed on the lab: the cluster's collector refuses it (not_placed_on_cluster), so the review is "
               "filed again in its lab form (L16RL)"
               % (prev.get("approval_id"), ex.get("phase"), ex.get("executed_by"),
                  (ex.get("outcome") or {}).get("status"), cand.get("provider")))
        prev.update(status="superseded", superseded_utc=self.utc, superseded_reason=_short(why, 500))
        self._ledger("review_superseded", lever="L16R", source=src, approval_id=prev.get("approval_id"),
                     reasons=[why])

    # ---- lab items (no ssh)
    def _lab_hooks(self):
        run = self.runner
        target = os.environ.get("CLUSTER_SSH", "")

        def hook(kind):
            def h(params):
                job = "%s_%s_%s" % (kind, re.sub(r"[^A-Za-z0-9_]", "_", str(params.get("source") or "")),
                                    _sha([params, self.utc])[:10])
                if kind == "sync" and params.get("candidate"):
                    rel = self.paths.cand_rel(params["source"])
                    row = self._candidate_row(params["source"])
                    if row is None:
                        return {"ok": False, "error": "no candidate record of %s in the lab's candidates.json"
                                                      % params["source"]}
                    _write_json(self.paths.lab_inc / rel, {"format": "collect-candidates/1", "candidates": [row],
                                                           "from": str(self.paths.candidates()),
                                                           "written_utc": self.utc})
                    argv = sync_argv(self.paths.lab_inc, params["source"], target, M.CLUSTER_DATA_SSH, file=rel)
                elif kind == "sync":
                    argv = sync_argv(self.paths.lab_inc, params["source"], target, M.CLUSTER_DATA_SSH,
                                     names=bool(params.get("names")))
                else:
                    verb = {"discover": "plan", "names": "names", "fetch": "fetch"}[kind]
                    # the lab's INC tree, named: the ticker's environment sets no INC_DIR, and inc.common's
                    # default is the cluster's /ocean path (2026-09-29: two L15 plans failed at intake_lock
                    # with PermissionError '/ocean' after five minutes of discovery)
                    argv = [sys.executable, "-m", "weed_optimizer_framework.tools.collect", verb,
                            "--inc-dir", str(self.paths.lab_inc)]
                    for k, flag in (("config", "--config"), ("classes", "--classes"), ("source", "--source"),
                                    ("max_bytes", "--max-bytes"), ("out", "--out")):
                        if params.get(k) not in (None, ""):
                            argv += [flag, str(params[k])]
                try:
                    got = run.launch(job, argv, timeout=lab_job_timeout(kind, params.get("max_bytes")))
                except Exception as e:
                    return {"ok": False, "error": "%s: %s" % (type(e).__name__, _short(e, 300))}
                return {"ok": True, "detached": True, "lab_job": got.get("job", job), "argv": argv}
            return h
        return {"inc_stream_discover": hook("discover"), "inc_stream_names": hook("names"),
                "inc_stream_collect_lab": hook("fetch"), "inc_stream_collect_review_lab": hook("fetch"),
                "inc_stream_sync": hook("sync")}

    def _run_lab_items(self):
        """Start every ready lab item (a detached process; no ssh): a proposed
        one, and a filed one as _ready treats a filed cluster item (approved:
        run it; run elsewhere: follow that run; denied: clear it; a gated data
        lever under data_autonomy on: re-check its gates). _ready skips lab
        actions, and this loop once took proposed items only, so an approved
        lab fetch (L16L) waited for ever: the DATA lane stood still from
        2026-09-29 to 2026-09-30 with its approval recorded."""
        ap = None
        for ln in ALL_LANES:
            it = self.st["lanes"][ln].get("item")
            if not it:
                continue
            p = it["proposal"]
            if p["policy_action"] not in X.LAB_ACTIONS:
                continue
            if it.get("status") == "proposed":
                res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
            elif it.get("status") == "filed" and it.get("approval_id"):
                if ap is None:
                    ap = AP.state(self.domain, root=self.xctx.approvals_root)
                a = ap.get(it["approval_id"]) or {}
                if a.get("status") == "approved" and a.get("execution") is None:
                    res = X.execute_approved(it["approval_id"], self.camp, self.xctx, invoked_by=AUTO,
                                             quiet_repeat=True)
                elif a.get("status") == "approved":
                    self._executed_elsewhere(ln, it, a)
                    continue
                elif a.get("status") == "denied":
                    self._ledger("denied", lane=ln, lever=it["lever"], approval_id=it["approval_id"],
                                 decided_by=a.get("decided_by") or "human")
                    self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
                    self._person_denied(it, a)
                    self._clear(ln)
                    continue
                elif p["policy_action"] in X.GATED_R2_ACTIONS and self.cfg.get("data_autonomy") == "on":
                    res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
                else:
                    continue
            else:
                continue
            self._on_result(ln, it, res)

    def _poll_lab(self):
        for ln in ALL_LANES:
            it = self.st["lanes"][ln].get("item")
            if not it or it.get("status") != "running" or not it.get("lab_job"):
                continue
            res = self.runner.poll(it["lab_job"])
            if res is None:
                continue
            self._ledger("lab_job_finished", lane=ln, lever=it["lever"], lab_job=it["lab_job"], ok=res.get("ok"),
                         rc=res.get("rc"), seconds=res.get("seconds"))
            if res.get("ok"):
                self._done(ln, it, res)
            else:
                self._failed(ln, it, "the lab process %s ended %s: %s" % (it["lab_job"], res.get("rc"),
                                                                           _short(res.get("stderr_tail") or res.get("error")
                                                                                  or "", 300)))

    # ---- the one ssh: ready items
    def _ready(self):
        out = []
        ap = AP.state(self.domain, root=self.xctx.approvals_root)
        for ln in SUBMIT_ORDER:
            it = self.st["lanes"][ln].get("item")
            if not it:
                continue
            p = it["proposal"]
            if p["policy_action"] in X.LAB_ACTIONS:
                continue
            if it.get("status") == "proposed":
                out.append((ln, it, None))
            elif it.get("status") == "filed" and it.get("approval_id"):
                a = ap.get(it["approval_id"]) or {}
                if a.get("status") == "approved" and a.get("execution") is None:
                    why = self._misplaced_review(it)
                    if why:
                        self._refuse_misplaced_review(ln, it, why)
                    else:
                        out.append((ln, it, it["approval_id"]))
                elif a.get("status") == "approved":
                    # run by someone else (a person from the INC page): the
                    # lane follows that run instead of waiting for ever
                    self._executed_elsewhere(ln, it, a)
                elif a.get("status") == "denied":
                    self._ledger("denied", lane=ln, lever=it["lever"], approval_id=it["approval_id"],
                                 decided_by=a.get("decided_by") or "human")
                    self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
                    self._person_denied(it, a)
                    self._clear(ln)
                elif self.cfg.get("autonomy") == "envelope" and p.get("risk") == "R3" \
                        and it["lever"] in LS.envelope_levers():
                    out.append((ln, it, None))          # re-checked against the envelope each tick
                elif p["policy_action"] in X.GATED_R2_ACTIONS and self.cfg.get("data_autonomy") == "on":
                    # a gated data lever filed for a person (shadow mode, a
                    # placeholder floor, a limit): re-checked against its gates
                    # each tick once a person has set data_autonomy on, so the
                    # lane does not wait for ever on an item filed before
                    out.append((ln, it, None))
        return out

    def _misplaced_review(self, it):
        """Why an approved source review in its cluster form (L16R: sbatch
        run_inc_collect.sh fetch) must not be submitted, or "": its source's
        candidate is placed on the lab, whose provider the cluster's collector
        refuses (not_placed_on_cluster). Such a review was filed before the
        lab form (L16RL) existed (mediatum_1717366: approval
        ap-1790700040-42144765, job 47302914). Fail closed: a source with no
        candidate row this tick (a discovered source the last plan, L15, no
        longer lists) is not submitted either, since its placement cannot be
        read, and an L16R carries no --candidates, so the cluster's collector
        could not load a discovered source's record anyway."""
        if it.get("lever") != "L16R":
            return ""
        src = ((it.get("proposal") or {}).get("params") or {}).get("source")
        cand = self._context_candidate(src) if src else {}
        if not cand:
            return ("L16R (approval %s) fetches source %s on the cluster, and the source has no candidate row in this "
                    "tick's candidates (the last plan, L15, no longer lists it, and the collect config has no known "
                    "item of that id): its provider's placement cannot be read, and the cluster's collector could not "
                    "load its record (an L16R carries no --candidates), so the approved cluster fetch is not "
                    "submitted. If the source is listed again, D20 fetches it (L16 or L16L, by its placement) or files "
                    "its review again in the form its placement calls for" % (it.get("approval_id"), src))
        if cand.get("placement") != "lab":
            return ""
        return ("L16R (approval %s) fetches source %s on the cluster, and its provider %s is placed on the lab "
                "(intake/placement.json, the collect config's lab_only): the cluster's collector refuses it "
                "(not_placed_on_cluster), so the approved cluster fetch is not submitted. The source is fetched on "
                "the lab instead: D20's L16 -> L16L once its pre-check passes, else its review filed again in its lab "
                "form (L16RL) for a person" % (it.get("approval_id"), src, cand.get("provider")))

    def _refuse_misplaced_review(self, ln, it, why):
        """Not submitted (_misplaced_review): recorded, declined, the lane
        cleared, nothing charged and no attempt or failure counted; the
        source's review is superseded so that _file_reviews files it again in
        the form its placement calls for. The approval is closed in the
        approvals log (_close_unsubmitted), so neither the ticker nor a
        person's Run now on the INC page (executor.execute_approved) can still
        send it to sbatch, where the cluster's collector would refuse it."""
        p = it["proposal"]
        src = (p.get("params") or {}).get("source")
        aid = it.get("approval_id")
        closed, cwhy = self._close_unsubmitted(aid, why)
        self._ledger("refused", lane=ln, lever=it["lever"], approval_id=aid, proposal_id=p["id"],
                     source=src, reasons=[why], charged=False, approval_closed=closed,
                     close_error=cwhy or None)
        self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
        rv = (self.st.get("source_reviews") or {}).get(src)
        if isinstance(rv, dict):
            rv.update(status="superseded", superseded_utc=self.utc, superseded_reason=_short(why, 500))
        note = ("approval %s is closed (not_submitted, no job), so it cannot be run again" % aid if closed else
                "approval %s could not be closed (%s): do not run it from the INC page, the cluster's collector "
                "refuses it" % (aid, cwhy))
        self._card("data", "Approved review of %s not submitted (approval %s)" % (src, aid), "%s. %s" % (why, note),
                   lever=it["lever"])
        self._clear(ln)

    def _close_unsubmitted(self, aid, why):
        """(closed, why not) for an approval the lane will not submit: its one
        run is recorded as started, then failed with the outcome not_submitted
        and no job, in the approvals log only (approvals.record_executed), so
        approvals.awaiting_execution no longer lists it and
        executor.execute_approved refuses it as already executed. The execution
        log, the budget and the source's attempts are not touched."""
        now = self.xctx.clock()
        got = AP.record_executed(self.domain, aid, "started", AUTO, now, root=self.xctx.approvals_root)
        if not got.get("ok"):
            return False, got.get("reason") or "the approval could not be claimed"
        got = AP.record_executed(self.domain, aid, "failed", AUTO, now, root=self.xctx.approvals_root,
                                 outcome={"status": "not_submitted", "job_ids": [], "error": _short(why, 1000)})
        if not got.get("ok"):
            # claimed, so it cannot run, but its outcome is not recorded
            return False, got.get("reason") or "the approval's outcome could not be written"
        return True, ""

    def _executed_elsewhere(self, ln, it, a):
        """A lane item whose approval another executor ran (a person from the
        INC page): the lane takes that run's outcome."""
        ex = a.get("execution") or {}
        oc = ex.get("outcome") or {}
        if ex.get("phase") == "done" and oc.get("status") == "executed":
            self._ledger("executed_elsewhere", lane=ln, lever=it["lever"], approval_id=it["approval_id"],
                         executed_by=ex.get("executed_by"), job_ids=oc.get("job_ids"))
            self._on_result(ln, it, {"status": "executed", "job_ids": list(oc.get("job_ids") or []),
                                     "approval_id": it["approval_id"], "decided_by": a.get("decided_by"),
                                     "basis": "approval executed by %s" % ex.get("executed_by")})
        elif ex.get("phase") in ("done", "failed"):
            self._failed(ln, it, "approval %s, run by %s, ended %s: %s" % (
                it["approval_id"], ex.get("executed_by"), ex.get("phase"), _short(oc.get("error") or "", 300)),
                charged=True)
        else:
            self._once("elsewhere:%s" % it["approval_id"], ex, "waiting", lane=ln, lever=it["lever"],
                       approval_id=it["approval_id"],
                       reasons=["approval %s is being run by %s (started %s); the lane waits for its outcome"
                                % (it["approval_id"], ex.get("executed_by"), ex.get("started_at"))])

    def _person_denied(self, it, a):
        """What a person's denial means beyond the declined proposal: a denied
        leak quarantine (L24 on D28) keeps the source, which lifts D28's TRAIN
        hold for it; a denied source review is recorded against the source."""
        p = it.get("proposal") or {}
        params = p.get("params") or {}
        src = params.get("source")
        if it.get("lever") == "L24" and src and params.get("cite") == "D28":
            self.st["sources"].setdefault(str(src), {})["leak_kept_by"] = a.get("decided_by") or "human"
            self._ledger("leak_kept", source=src, decided_by=a.get("decided_by") or "human",
                         approval_id=it.get("approval_id"))
        if it.get("lever") in REVIEW_LEVERS and src:
            rv = (self.st.get("source_reviews") or {}).get(src)
            if isinstance(rv, dict):
                rv["status"] = "denied"

    # ---- items a person decides, filed without holding their lane
    def _file_person(self, lane_name, lever, pr, did, d):
        """An R3 item only a person may decide (LH, a funnel_F9 release at its
        deadline) is filed without taking its lane, so the lane keeps working
        while it waits; once approved it is adopted into the lane and run."""
        st = self.st
        try:
            p = self._proposal(lever, pr, did, d)
        except (LS.LeverError, ValueError, KeyError) as e:
            self._once("render:%s:%s" % (lever, did), str(e), "not_taken", lever=lever,
                       reasons=["cannot render %s: %s" % (lever, _short(e, 300))])
            return
        if p is None or p["id"] in (st.get("declined") or []) or p["id"] in (st.get("person_items") or {}):
            return
        done_at = (st.get("done_keys") or {}).get(_sha([p.get("lever") or lever, p.get("params") or {}])[:16])
        if done_at and done_at >= str(st.get("last_snapshot_utc") or ""):
            return                                # ran since the evidence was taken
        res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
        if res.get("status") != "filed" or not res.get("approval_id"):
            self._once("person:%s" % p["id"], res.get("reasons"), "not_taken", lever=p.get("lever") or lever,
                       reasons=res.get("reasons") or [str(res.get("status"))])
            return
        self._park(lane_name, p, did, d, res.get("approval_id"))
        self._ledger("filed", lane=lane_name, lever=p.get("lever") or lever, approval_id=res.get("approval_id"),
                     reasons=res.get("reasons"), proposal_id=p["id"], est_su=res.get("est_su"), parked=True)

    def _park(self, lane_name, p, did, d, approval_id):
        self.st.setdefault("person_items", {})[p["id"]] = {
            "lane": lane_name, "lever": p.get("lever"), "proposal": p, "approval_id": approval_id,
            "trigger": did, "diagnoses": [d] if d else [], "filed_utc": self.utc}

    def _adopt_person_items(self):
        """Parked items a person has decided: an approved one takes its lane
        when the lane is free (and runs through the approval); a denied one is
        declined; one run elsewhere is dropped."""
        st = self.st
        items = st.get("person_items") or {}
        if not items:
            return
        ap = AP.state(self.domain, root=self.xctx.approvals_root)
        for pid in sorted(items, key=lambda k: str(items[k].get("filed_utc"))):
            rec = items[pid]
            a = ap.get(rec.get("approval_id")) or {}
            if a.get("status") == "denied":
                it = {"proposal": rec["proposal"], "lever": rec.get("lever"), "approval_id": rec.get("approval_id")}
                self._ledger("denied", lane=rec.get("lane"), lever=rec.get("lever"), approval_id=rec.get("approval_id"),
                             decided_by=a.get("decided_by") or "human")
                st["declined"] = (list(st.get("declined") or []) + [pid])[-200:]
                self._person_denied(it, a)
                items.pop(pid, None)
            elif a.get("status") == "approved" and a.get("execution") is not None:
                self._ledger("executed_elsewhere", lane=rec.get("lane"), lever=rec.get("lever"),
                             approval_id=rec.get("approval_id"), executed_by=(a.get("execution") or {}).get("executed_by"))
                items.pop(pid, None)
            elif a.get("status") == "approved":
                lane = st["lanes"].get(rec.get("lane")) or {}
                if lane.get("item") is not None or lane.get("hold"):
                    continue
                lane["item"] = {"proposal": rec["proposal"], "lever": rec.get("lever"), "status": "filed",
                                "approval_id": rec.get("approval_id"), "since_utc": self.utc, "attempt": 0,
                                "trigger": rec.get("trigger"), "diagnoses": rec.get("diagnoses") or []}
                lane["phase"] = self._phase_of(rec.get("lever"))
                self._ledger("adopted", lane=rec.get("lane"), lever=rec.get("lever"),
                             approval_id=rec.get("approval_id"), proposal_id=pid,
                             decided_by=a.get("decided_by") or "human")
                items.pop(pid, None)

    def _submit_ready(self):
        """True when this tick's ssh went to submissions."""
        if self.st.get("drift"):
            return False
        ready = self._ready()
        if not ready:
            return False
        before = self.ssh.calls
        own = [x for x in ready if x[1]["proposal"].get("risk") == "R3" or x[2]]
        direct = [x for x in ready if x not in own]

        def run_direct():
            if direct and self.ssh.calls == before and self.ssh.left() and not self._paused_reason():
                res_all = X.submit_many([it["proposal"] for _ln, it, _a in direct], actor=AUTO, campaign=self.camp,
                                        ctx=self.xctx)
                for (ln, it, _a), res in zip(direct, res_all):
                    self._on_result(ln, it, res)
        # 6.2 step 2: stop/cancel first. A ready STOP item (a quarantine, an
        # unblock) goes before any build: an R3 build tried first would take
        # the tick's one ssh and cut from a source about to be quarantined.
        stop_first = any(ln == "STOP" for ln, _it, _a in direct)
        if stop_first:
            run_direct()
        # An R3 item (or an approved one) runs in its own call when its grant
        # or approval lets it run, and is filed without any call otherwise:
        # each is tried in priority order until one uses the tick's ssh.
        for ln, it, appr in own:
            if self.ssh.calls > before or not self.ssh.left():
                break
            if appr:
                res = X.execute_approved(appr, self.camp, self.xctx, invoked_by=AUTO, quiet_repeat=True)
            else:
                res = X.submit(it["proposal"], actor=AUTO, campaign=self.camp, ctx=self.xctx)
            self._on_result(ln, it, res)
        if not stop_first:
            run_direct()
        return self.ssh.calls > before

    def _on_result(self, ln, it, res):
        st = self.st
        s = res.get("status")
        p = it["proposal"]
        payload = ((res.get("remote") or {}).get("payload")) or {}
        if s == "executed":
            it.update(status="running", job_ids=list(res.get("job_ids") or []), approval_id=res.get("approval_id")
                      or it.get("approval_id"), executed_utc=self.utc, lab_job=payload.get("lab_job"))
            self._ledger("executed", lane=ln, lever=it["lever"], action=p["policy_action"], job_ids=res.get("job_ids"),
                         approval_id=res.get("approval_id"), decided_by=res.get("decided_by") or res.get("authorized_as"),
                         basis=res.get("basis"), est_su=res.get("est_su"), proposal_id=p["id"],
                         child_exp=p.get("child_exp"), trigger=p.get("trigger"))
            self._mirror_action(p, res)
            self._on_started(ln, it)
            if p.get("follow") == "login" or p["policy_action"] in ("inc_stream_commit", "inc_stream_rollback",
                                                                     "inc_stream_quarantine", "inc_stream_release",
                                                                     "inc_unblock_transient"):
                self._done(ln, it, payload)
            return
        if s == "filed":
            if it.get("status") != "filed" or it.get("approval_id") != res.get("approval_id"):
                it.update(status="filed", approval_id=res.get("approval_id"), filed_utc=self.utc)
                self._ledger("filed", lane=ln, lever=it["lever"], approval_id=res.get("approval_id"),
                             reasons=res.get("reasons"), proposal_id=p["id"], est_su=res.get("est_su"))
            else:
                self._once("filed:%s" % p["id"], res.get("reasons"), "waiting", lane=ln, lever=it["lever"],
                           approval_id=res.get("approval_id"), reasons=res.get("reasons"))
            return
        why = "; ".join(res.get("reasons") or []) or str(s)
        if s == "failed":
            if X.never_ran(res):
                self._once("transport:%s" % p["id"], why, "transport_failed", lane=ln, lever=it["lever"],
                           reasons=res.get("reasons"))
                return
            if payload.get("error_kind") == "busy":
                # inc2.stream's lease is held by another writer (exit 3): a
                # transient, never a failed step; the same proposal runs next tick
                self._once("busy:%s" % p["id"], why, "waiting", lane=ln, lever=it["lever"],
                           reasons=["another writer holds stream.lease: %s" % _short(why, 200)])
                if it.get("status") != "filed":
                    it["status"] = "proposed"
                return
            if payload.get("error_kind") == "qos":
                # S21: a qos rejection is a platform defect, not an attempt of the source
                self._ledger("platform_defect", lane=ln, lever=it["lever"], reasons=res.get("reasons"),
                             error_kind="qos")
                self._card("platform", "sbatch refused on qos: a platform defect", why, trigger=list(p.get("trigger") or []))
                st["lanes"][ln]["hold"] = "platform defect: qos refusal (%s); a person checks the partition" % p["policy_action"]
                st["lanes"][ln]["hold_utc"] = self.utc
                self._clear(ln, keep_phase=False)
                return
            if X.uncertain(res):
                if p.get("follow") in ("build", "experiment") and p.get("child_exp"):
                    it.update(status="running", uncertain=True, job_ids=[])
                    self._ledger("uncertain", lane=ln, lever=it["lever"], child_exp=p.get("child_exp"),
                                 reasons=res.get("reasons"), next="the snapshots decide")
                    self._on_started(ln, it)
                    return
                if it["lever"] in RECORD_ONLY_LEVERS:
                    # record only (L23N): its job has a name of its own (a second is refused while it is queued)
                    # and writes once, so it is followed by that name and its record; never a pause
                    it.update(status="running", uncertain=True, job_ids=[])
                    self._ledger("uncertain", lane=ln, lever=it["lever"], reasons=res.get("reasons"),
                                 next="followed by its job name; its record decides")
                    self._on_started(ln, it)
                    return
                self._ledger("uncertain", lane=ln, lever=it["lever"], reasons=res.get("reasons"), next="paused")
                return self._pause("the outcome of %s (%s) is unknown: it may have run on the cluster; a person checks"
                                   % (it["lever"], p["policy_action"]))
            return self._failed(ln, it, why, charged=bool(res.get("charged")))
        # refused
        if any(t in why for t in WAIT_REFUSALS) or "the campaign is paused" in why:
            self._once("wait:%s" % p["id"], why, "waiting", lane=ln, lever=it["lever"], reasons=res.get("reasons"))
            return
        if "already ran" in why or "already executed" in why:
            self._ledger("recovered", lane=ln, lever=it["lever"], reasons=res.get("reasons"))
            if p.get("follow") in ("build", "experiment") and p.get("child_exp"):
                it["status"] = "running"          # its experiment is followed by name
                self._on_started(ln, it)
                return
            # a job or verb that already ran has no job id to follow here: its
            # effect shows in the snapshots; following it would fail it as a
            # job that "ended unknown" and count toward the lane's stop-loss
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            self._clear(ln)
            return
        if "left in the campaign envelope" in why:
            self._ledger("refused", lane=ln, lever=it["lever"], reasons=res.get("reasons"))
            return self._pause("budget: %s cannot be charged to the campaign envelope (%s)" % (it["lever"], _short(why)))
        self._ledger("refused", lane=ln, lever=it["lever"], reasons=res.get("reasons"), proposal_id=p["id"])
        self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
        self._failed(ln, it, why)

    def _mirror_action(self, p, res):
        fn = self.log_action
        if not callable(fn):
            return
        try:
            jobs = list(res.get("job_ids") or [])
            fn("inc_autopilot:%s" % p.get("policy_action"),
               {"ok": res.get("status") == "executed", "jobid": jobs[0] if jobs else None, "job_ids": jobs,
                "campaign": self.name, "lever": p.get("lever"), "approval_id": res.get("approval_id"),
                "decided_by": res.get("decided_by"), "cmd": " ".join(p.get("argv") or []),
                "msg": "; ".join(res.get("reasons") or [])})
        except Exception as e:
            self.log.warning("[inc-stream] action log failed: %s" % e)

    def _on_started(self, ln, it):
        """Bookkeeping when an item starts: a source's attempt, an experiment the
        stream now owns, the R0 progress."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        if lever in FETCH_LEVERS:
            s = st["sources"].setdefault(params["source"], {})
            s["attempts"] = int(s.get("attempts") or 0) + 1
            s.update(status="fetching", placement="lab" if lever in LAB_FETCH_LEVERS else "cluster",
                     max_bytes=params.get("max_bytes"))
        child = p.get("child_exp")
        if child and child not in st["exps"] and p.get("follow") in ("build", "experiment"):
            st["exps"].append(child)
            if lever == "L18":
                st["segments"].append({"exp": child, "n": len(st["segments"]) + 1, "done": False,
                                       "committed": False, "built_utc": None})
            elif lever == "L20":
                done = [s for s in st["segments"] if s.get("committed")]
                st["milestones"].append({"exp": child, "n": len(st["milestones"]) + 1, "done": False,
                                         "after_segment": len(done), "pool": None})
        r0 = st["stage"].setdefault("r0", {})
        if lever == "L23B":
            b = next((x for x in (self.dom.get("baselines") or {}).get("items") or [] if x["exp"] == child), None)
            if b:
                r0["baseline_%s" % b["id"]] = "building"
        elif lever == "L23N":
            b = self._native_item(params)
            if b:
                r0["native_%s" % b["id"]] = "running"
        elif lever == "L25":
            r0["stage_a"] = "building"
        elif lever == "L28":
            r0["stage_c"], r0["stage_c_built"] = "building", True
        elif lever == "LP":
            r0["probe"] = self.utc
        for k in ("pilot",):
            ref = (self.dom.get("stage_a") or {}).get("from_exp")
            if lever == "L25" and ref and ref not in st["exps"]:
                st["exps"].append(ref)
        if lever == "L15":
            st["discover"]["before"] = sorted(c.get("id") for c in (self.ev.json(E.CONTEXT) or {}).get("candidates") or [])

    def _clear(self, ln, keep_phase=False):
        lane = self.st["lanes"][ln]
        lane["item"] = None
        if not keep_phase:
            lane["phase"] = "IDLE"

    def _done(self, ln, it, payload=None):
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        lane = st["lanes"][ln]
        lane["fails"] = 0
        self._ledger("item_done", lane=ln, lever=lever, proposal_id=p["id"], child_exp=p.get("child_exp"))
        # what a later tick diagnoses before the next snapshot still shows the
        # state before this item: the same work is not proposed again on it
        dk = st.setdefault("done_keys", {})
        dk[_sha([lever, p.get("params") or {}])[:16]] = self.utc
        if len(dk) > 64:
            for k in sorted(dk, key=lambda k: dk[k])[:len(dk) - 64]:
                dk.pop(k, None)
        if lever == "L15":
            before = set(st["discover"].pop("before", []) or [])
            after = set(c.get("id") for c in self._candidates())
            new = len(after - before)
            d = st["discover"]
            d.update(last_utc=self.utc, found_new=new, runs=int(d.get("runs") or 0) + 1,
                     empty_runs=(int(d.get("empty_runs") or 0) + 1) if new == 0 else 0)
            self._check_recall()
        elif lever in FETCH_LEVERS:
            self._fetch_ends()[p["id"]] = self.utc
            s = st["sources"].setdefault(params["source"], {})
            s.update(status="fetched", fetched_utc=self.utc)
            if lever in LAB_FETCH_LEVERS:
                # a lab fetch appends to the lab's own sources.jsonl, which no
                # snapshot folds: its completeness comes from its closing line
                got = self._collect_result(payload)
                self._fetch_completeness(params["source"], got.get("complete"), got.get("remaining"), "lab fetch")
        elif lever == "L16S" and params.get("names"):
            st["sources"].setdefault(params["source"], {})["names_synced"] = self.utc
        elif lever == "L16S" and params.get("candidate"):
            st["sources"].setdefault(params["source"], {})["candidate_synced"] = LS.inc_path(
                self.paths.cand_rel(params["source"]))
        elif lever == "L16S":
            st["sources"].setdefault(params["source"], {})["synced"] = True
        elif lever == "L16I":
            s = st["sources"].setdefault(params["source"], {})
            s["status"] = "intaken"
        elif lever == "L17" and params.get("verb") == "admit":
            src = next((k for k, v in st["sources"].items() if v.get("batch") == params.get("intake")), None)
            if src:
                st["sources"][src]["status"] = "admitted"
                st["sources"][src]["admit_done_utc"] = self.utc
        elif lever == "L17":
            st["stage"]["r0"]["step1_%s" % params.get("verb")] = self.utc
        elif lever == "L19":
            for s in st["segments"]:
                if s["exp"] == params.get("exp"):
                    s["committed"] = True
                    s["committed_utc"] = self.utc
        elif lever == "L21":
            rb = st.setdefault("rollbacks", [])
            q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
            pend = [x for x in q.get("rollback_pending") or [] if isinstance(x, dict)
                    and x.get("to_pool") == params.get("to")]
            rb.append({"utc": self.utc, "to": params.get("to"), "milestone": pend[-1].get("milestone") if pend else None})
            self._card("research", "X4: leave-one-source-out after the rollback to %s" % params.get("to"),
                       "the platform bisects the suspect increments (L27); X4 stays for what it cannot separate",
                       lever="X4", trigger=[p.get("trigger", [""])[0] if p.get("trigger") else "D25"])
        elif lever == "L24":
            s = st["sources"].setdefault(params["source"], {})
            s.update(status="quarantined", quarantined_utc=self.utc, cite=params.get("cite"))
            self._card("leak" if params.get("cite") == "D28" else "data",
                       "Source %s quarantined (%s)" % (params["source"], params.get("cite")),
                       "L24 ran on a firing %s; reversible by a person" % params.get("cite"),
                       trigger=[params.get("cite")])
        elif lever == "L23":
            if params.get("verb") == "build":
                st["stage"]["r0"]["splits_built"] = True
        elif lever == "L23N":
            b = self._native_item(params)
            if b:
                st["stage"].setdefault("r0", {})["native_%s" % b["id"]] = "done"
        elif lever == "LV":
            key = p.get("verdict_key") or params.get("exp") or params.get("verb")
            st["stage"]["r0"].setdefault("verdicts", {})[key] = self.utc
        elif lever == "L26":
            st["sources"].setdefault(params["source"], {})["names_resolved"] = self.utc
        elif lever == "L22":
            st["doublings"] = int(st.get("doublings") or 0) + 1
            st["fork_pending"] = self.utc
        elif lever == "L27":
            rb = st.get("rollbacks") or []
            st["bisected"] = rb[-1]["utc"] if rb else self.utc
        self._clear(ln)

    def _failed(self, ln, it, why, charged=False, retry=True):
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        lane = st["lanes"][ln]
        if lever in FETCH_LEVERS:
            # ended: the byte limits count what it fetched, not its max_bytes
            self._fetch_ends()[p["id"]] = self.utc
        if lever in RECORD_ONLY_LEVERS:
            # record only (a measurement arm's native-resolution rescore): a person's card; the lane's
            # failure count, its step count and its stop-loss are not touched, and the item stays failed,
            # so DR0 does not propose it again (a person reruns it, or leaves it)
            b = self._native_item(params)
            if b:
                st["stage"].setdefault("r0", {})["native_%s" % b["id"]] = "failed"
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                         charged=charged, record_only=True)
            self._card("escalation", "Native-resolution rescore of %s failed (%s)" % (params.get("exp"), lever),
                       "%s. Record only: the stream runs on and the rescore is not proposed again; a person reruns "
                       "inc2.baseline rescore-native --exp %s or leaves it" % (_short(why, 400), params.get("exp")),
                       lever=lever, trigger=list(p.get("trigger") or []))
            self._clear(ln)
            return
        if not retry:
            # a refusal the identical request meets again (a verifier or LOCK
            # mismatch, S3): declined for good, its source held, a person decides
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            src = params.get("source") or next((k for k, v in st["sources"].items()
                                                if v.get("batch") == params.get("intake")), None)
            if src:
                st["sources"].setdefault(src, {}).update(status="held", held_reason=_short(why, 300))
            self._card("research", "X11: the verifier or the LOCK changed under %s" % lever, why, lever="X11",
                       trigger=list(p.get("trigger") or []))
            self._ledger("not_taken", lane=ln, lever=lever, proposal_id=p["id"], retry=False,
                         reasons=["refused with a refusal the identical request meets again: %s" % _short(why, 300)])
            self._clear(ln)
            return
        lane["fails"] = int(lane.get("fails") or 0) + 1
        key = step_key(lever, params)             # the proposal's params: its policy params and price
        att = st.setdefault("attempts", {})
        att[key] = int(att.get(key) or 0) + 1
        st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        sf = st.setdefault("step_failures", {})
        sf[key] = {"n": int((sf.get(key) or {}).get("n") or 0) + 1, "lever": lever, "lane": ln,
                   "last": _short(why, 300), "utc": self.utc, "proposal_id": p["id"]}
        st["refusals"] = (list(st.get("refusals") or []) + [{"builder": p.get("policy_action"),
                                                              "message": _short(why, 1000)}])[-REFUSALS_KEEP:]
        self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                     charged=charged, fails=lane["fails"])
        if lever in FETCH_LEVERS + ("L16I", "L17") and params.get("source") or lever in FETCH_LEVERS:
            src = params.get("source") or next((k for k, v in st["sources"].items()
                                                if v.get("batch") == params.get("intake")), None)
            if src:
                s = st["sources"].setdefault(src, {})
                s["failures"] = int(s.get("failures") or 0) + 1
                s["status"] = "candidate" if s["failures"] < 3 else "closed"
                if s["status"] == "closed":
                    s["closed_reason"] = "3 failed attempts (7.5)"
                    self._ledger("source_closed", source=src, reasons=["3 failed attempts"])
                    self._zero_yield(src, "failed")
        self._clear(ln)
        if lane["fails"] >= int(LS.t(self.th, "stop_loss", "lane_fails")):
            lane["hold"] = "stop-loss: %d consecutive failed steps (last: %s)" % (lane["fails"], _short(why, 200))
            lane["hold_utc"] = self.utc
            self._card("stop_loss", "Lane %s held" % ln, lane["hold"])
            self._ledger("lane_held", lane=ln, hold=lane["hold"])

    def _zero_yield(self, src, how, boxes=None):
        """The DATA lane's zero-yield stop-loss (6.6, S20): 3 consecutive sources
        ending failed or with zero admitted target boxes hold the lane, with a
        card listing each source's decision reasons."""
        st = self.st
        if how == "yield" and boxes:
            st["zero_run"] = []
            return
        run = [r for r in st.get("zero_run") or [] if r.get("source") != src]
        run.append({"source": src, "how": how, "reasons": self._decision_reasons(src), "utc": self.utc})
        st["zero_run"] = run
        n = int(LS.t(self.th, "stop_loss", "zero_yield_sources"))
        if len(run) >= n:
            lane = st["lanes"]["DATA"]
            lane["hold"] = "stop-loss: %d consecutive sources ended failed or with zero admitted target boxes" % len(run)
            lane["hold_utc"] = self.utc
            self._card("stop_loss", "DATA lane held: %d zero-yield sources" % len(run), "; ".join(
                "%s (%s): %s" % (r["source"], r["how"], ", ".join("%s %s" % (k, v) for k, v in
                                                                  sorted((r.get("reasons") or {}).items())) or "no reasons recorded")
                for r in run))
            self._ledger("lane_held", lane="DATA", hold=lane["hold"], sources=run)

    def _fetch_completeness(self, src, complete, remaining=None, basis=""):
        """A source whose last fetch stopped at a byte cap (collect.fetch:
        complete false, the files left recorded as remaining) is `partial`:
        once its admitted shard is observed, D20 may collect its next shard
        (the collector continues from what staging holds), within L16's
        attempt and byte limits. A complete fetch clears it."""
        if not isinstance(complete, bool):
            return
        s = self.st["sources"].setdefault(str(src), {})
        was = bool(s.get("partial"))
        s["partial"] = not complete
        if remaining is not None:
            s["fetch_remaining"] = remaining
        if s["partial"] != was:
            self._ledger("source_partial" if s["partial"] else "source_complete", source=src,
                         remaining=remaining, basis=basis)

    @staticmethod
    def _collect_result(res):
        """The collector's closing JSON ({...} after '[collect] <verb>: ') in a
        lab process's stdout tail, or {}."""
        for line in reversed(str((res or {}).get("tail") or "").splitlines()):
            m = re.match(r"^\[collect\] [a-z]+: (\{.*\})\s*$", line.strip())
            if m:
                try:
                    out = json.loads(m.group(1))
                    return out if isinstance(out, dict) else {}
                except ValueError:
                    return {}
        return {}

    def _decision_reasons(self, src):
        """{reason: count} of a source's intake decisions (its batches' summaries,
        from the snapshot being folded when there is one)."""
        out = {}
        arts = getattr(self, "_arts", None)
        if arts is None:
            arts = {n: self.ev.json(n) for n in (self.ev.artifacts if self.ev is not None else [])}
        for name in sorted(arts):
            if name.startswith("intake/") and name.endswith("/summary.json"):
                s = arts.get(name) or {}
                if s.get("source") != src:
                    continue
                # collect.intake's summary: its decisions by reason (the guard's and the
                # normaliser's rejections), else the yield's rejected counts
                dec = dict(((s.get("decisions") or {}).get("by_reason")) or {})
                dec.pop("kept", None)
                for k, v in (dec or (s.get("yield") or {}).get("rejected") or s.get("zero_yield_reasons") or {}).items():
                    out[k] = out.get(k, 0) + int(v or 0)
        return out

    def _check_recall(self):
        """S1b: discovery's recall of the owner's known items (collect config);
        a miss is a discovery defect and raises a card; the item stays fetchable.
        The collector's own recall (candidates.json `recall`: a known item is
        found when a search result matched its match rules) is read when
        present; else a known item counts as found when a searched row names it."""
        raw = _read_json(self.paths.candidates())
        rec_c = raw.get("recall") if isinstance(raw, dict) and isinstance(raw.get("recall"), dict) else None
        _p, cc = self._collect_config()
        known = [str(r.get("id") or r.get("source")) for r in (cc or {}).get("known_items") or []
                 if isinstance(r, dict) and r.get("recall", True) is not False]
        if rec_c is not None:
            missed = sorted(str(m.get("id")) for m in rec_c.get("missed") or [] if isinstance(m, dict))
            n = int(rec_c.get("known_items") or len(known))
            basis = "collector"
        else:
            rows = raw.get("candidates") if isinstance(raw, dict) else raw
            found = {str(r.get("source_id") or r.get("id") or r.get("source") or "") for r in rows or []
                     if isinstance(r, dict) and not r.get("known_item")}
            missed = sorted(k for k in known if k not in found)
            n = len(known)
            basis = "rows"
        rec = {"known": n, "found": n - len(missed), "recall": (n - len(missed)) / float(n) if n else None,
               "missed": missed, "basis": basis}
        self.st["discover"]["recall"] = rec
        self._ledger("discovery_recall", **rec)
        if missed:
            self._card("discovery", "Discovery missed %d known item(s)" % len(missed),
                       "a discovery defect (S1b): %s; each stays fetchable by L16" % ", ".join(missed))

    # ---- observe: the one snapshot
    def _r0_exps(self):
        """The rollout's experiments the stream reads whoever built them: the
        baselines, Stage A and the pilot it is judged against, Stage C."""
        dom = self.dom or {}
        out = [b["exp"] for b in (dom.get("baselines") or {}).get("items") or []]
        sa = dom.get("stage_a") or {}
        out += [x for x in (sa.get("exp"), sa.get("from_exp")) if x]
        out.append("%s_c001" % self.sid)
        return out

    def _live_exps(self):
        st = self.st
        names = list(dict.fromkeys(list(st.get("exps") or []) + self._r0_exps()))
        return [e for e in names if e not in (st.get("frozen") or {})]

    def _observe(self):
        st = self.st
        payload_old = _read_json(self.paths.latest(self.name)) or {}
        status_old = {x.get("exp"): x for x in ((payload_old.get("status") or {}).get("experiments") or [])
                      if isinstance(x, dict)}
        live = self._live_exps()
        ref = (self.dom.get("stage_a") or {}).get("from_exp")
        advance = [e for e in live if e != ref and e in (st.get("exps") or [])
                   and (status_old.get(e) or {}).get("built") and not (status_old.get(e) or {}).get("done")]
        jobs = []
        for ln in ALL_LANES:
            it = st["lanes"][ln].get("item") or {}
            if it.get("status") == "running":
                jobs += [str(j) for j in it.get("job_ids") or []]
        res = X.stream_snapshot(self.sid, exps=live, advance=advance, sacct=jobs,
                                largest=bool(st.get("want_largest")), campaign=self.camp, ctx=self.xctx)
        if res.get("status") == "refused":
            self._once("snapshot_refused", res.get("reasons"), "snapshot_refused", reasons=res.get("reasons"))
            return
        payload = (res.get("remote") or {}).get("payload")
        if not isinstance(payload, dict) or payload.get("verb") != "stream-snapshot":
            st["snapshot_failures"] = int(st.get("snapshot_failures") or 0) + 1
            if st["snapshot_failures"] in (1, 6):
                self._ledger("snapshot_failed", reasons=res.get("reasons"), failures=st["snapshot_failures"])
            if st["snapshot_failures"] >= 6:
                self._card("cluster", "No stream snapshot for %d ticks" % st["snapshot_failures"],
                           "; ".join(res.get("reasons") or []))
            return
        st["snapshot_failures"] = 0
        st["last_snapshot_utc"] = self.utc
        st.pop("await_snapshot", None)
        _write_json(self.paths.latest(self.name), payload)
        self._fold(payload)

    def _fold(self, payload):
        """The snapshot into the lanes: code drift (S23), experiments built and
        done, jobs ended (sacct), sources' intake and admission, holds, health."""
        st = self.st
        # S23: the cluster's stream modules against the lab's
        from . import stream_remote as SR
        lab = SR.module_hashes(LS.CODE_ROOT / "weed_optimizer_framework")
        cl = ((payload.get("code") or {}).get("modules")) or {}
        diff = sorted(k for k in set(lab) | set(cl) if lab.get(k) != cl.get(k))
        if diff:
            if st.get("drift") != diff:
                self._card("drift", "Lab and cluster stream code differ: every stream submission is refused",
                           "modules: %s" % ", ".join(diff[:12]))
                self._ledger("code_drift", modules=diff)
            st["drift"] = diff
        elif st.get("drift"):
            self._ledger("code_drift_cleared", modules=st["drift"])
            st["drift"] = None
        q = ((((payload.get("stream") or {}).get("decision") or {}).get("artifacts")) or {}).get(
            "stream/%s/queue_summary.json" % self.sid) or {}
        self._adopt_fork(q)
        chain = ((payload.get("stream") or {}).get("ledger_chain")) or {}
        if chain.get("ok") is False:
            return self._pause("the stream ledger's hash chain does not verify (%s)" % chain.get("why"))
        status = {x.get("exp"): x for x in ((payload.get("status") or {}).get("experiments") or [])
                  if isinstance(x, dict)}
        sq = (payload.get("status") or {}).get("squeue") or {}
        queued = {str(j.get("id", "")).split("_")[0] for j in sq.get("jobs") or [] if isinstance(j, dict)} \
            if sq.get("ok") else None
        names = {j.get("name") for j in sq.get("jobs") or [] if isinstance(j, dict)} if sq.get("ok") else set()
        sacct = ((payload.get("sacct") or {}).get("jobs")) or {}
        arts = (((payload.get("stream") or {}).get("decision") or {}).get("artifacts")) or {}
        self._arts = arts
        # the experiments the stream built (segments, milestones, Stage C, every
        # bisect arm): tracked, so each is advanced and its status observed
        for e in arts.get("stream/%s/ledger.jsonl" % self.sid) or []:
            if not isinstance(e, dict):
                continue
            built = []                            # (never `names`: the queued job names, read by the items below)
            if e.get("event") in ("build", "milestone", "feasibility") and e.get("phase") in (None, "build"):
                built.append(e.get("exp"))
            if e.get("event") == "bisect" and e.get("phase") == "build":
                built += list((e.get("arms") or {}).values())
            for n_ in built:
                if n_ and NAME_RE.match(str(n_)) and n_ not in st["exps"]:
                    st["exps"].append(str(n_))
        # sources: intake batches (collect.intake summary.json) and the
        # collector's fold of intake/sources.jsonl
        for name, s in sorted(arts.items()):
            if name.startswith("intake/") and name.endswith("/summary.json") and isinstance(s, dict) and s.get("source"):
                src = st["sources"].setdefault(str(s["source"]), {})
                batch = str(s.get("batch") or name.split("/")[1])
                if batch not in (src.get("batches") or []):
                    src.setdefault("batches", []).append(batch)
                    src["batch"] = batch
                src["images"] = s.get("rows")
                src["target_boxes_intake"] = (s.get("yield") or {}).get("target_boxes")
                if s.get("zero_yield"):
                    src["zero_yield_reasons"] = s.get("zero_yield_reasons")
        srows = arts.get("intake/sources.json") or {}
        if isinstance(arts.get("intake/sources.json"), dict):
            # the cluster collector's fetched facts as of this snapshot, which the
            # byte limits count (executor.fetched_bytes): None when the fold
            # carries none; otherwise a source absent from the ledger fetched
            # nothing on the cluster
            st["cluster_fetched"] = dict(_fetch_facts(srows) or {"sources": None, "open": None}, utc=self.utc)
        _cp, cc = self._collect_config()
        overrides = (cc or {}).get("licence_overrides")
        overrides = overrides if isinstance(overrides, dict) else {}
        for src, row in srows.items():
            if not isinstance(row, dict):
                continue
            s = st["sources"].setdefault(src, {})
            for k in ("bytes", "su", "total_bytes", "failed_attempts"):
                if row.get(k) is not None:
                    s[k] = row[k]
            self._fetch_completeness(src, row.get("fetch_complete"), row.get("fetch_remaining"), "collector ledger")
            rst = row.get("status")
            ov = None
            if rst == "held" and isinstance(overrides.get(src), dict) and (
                    "licence_unresolved" in (row.get("holds") or []) or row.get("reason") == "licence_unresolved"):
                # a person's licence override (the collect config's licence_overrides) lifts the collector's
                # licence hold: collect.plan records that release in the lab's ledger, which no snapshot folds,
                # so a hold in this ledger is read as released here (the next fetch checks the source anew)
                rst, ov = "candidate", overrides[src]
            if rst in ("held", "closed", "quarantined") and s.get("status") not in ("closed", "quarantined"):
                if s.get("status") != rst:
                    self._ledger("source_status", source=src, status=rst, by="collector",
                                 reasons=[_short(row.get("reason") or "", 300)])
                s["status"] = rst
                s["held_by"] = "collector"
                if row.get("reason"):
                    s["held_reason"] = _short(row.get("reason"), 300)
            elif s.get("status") == "held" and s.get("held_by") == "collector" \
                    and rst in ("candidate", "fetched", "intaken"):
                # the collector released its hold (licence, credentials, the copy scan): collectable again
                s.update(status=rst, held_by=None, held_reason=None)
                if ov is not None:
                    # the release is the person's decision the collect config records, not the collector's
                    self._ledger("source_released", source=src, status=s["status"], by="licence_overrides",
                                 decided_by=ov.get("decided_by"), decided_utc=ov.get("decided_utc"),
                                 licence=ov.get("id"), research_only=ov.get("research_only"),
                                 collect_config_sha256=LS.sha256_file(_cp) if _cp and Path(_cp).is_file() else None)
                else:
                    self._ledger("source_released", source=src, status=s["status"], by="collector")
            elif rst in ("fetched", "intaken") and s.get("status") in (None, "candidate"):
                # fetched or intaken outside this campaign's own items (a person's run of the collector, a lab
                # fetch, a restarted ticker): the collector's ledger is the record, so DPIPE takes the next step
                # (L16I, then L17 admit) instead of the source waiting with no status for ever
                s["status"] = row["status"]
                self._ledger("source_status", source=src, status=row["status"], by="collector",
                             reasons=["adopted from the collector's ledger"])
        s1 = (arts.get("step1_stream/status.json") or {}).get("per_source") or {}
        for src, row in s1.items():
            s = st["sources"].setdefault(src, {})
            s["admitted_target_boxes"] = (row or {}).get("target_boxes_admitted", s.get("admitted_target_boxes"))
            if s.get("status") == "admitted" and not s.get("yield_recorded") and s.get("admit_done_utc"):
                s["yield_recorded"] = True
                boxes = int(s.get("admitted_target_boxes") or 0)
                self._zero_yield(src, "yield" if boxes else "zero", boxes)
        # items
        for ln in ALL_LANES:
            it = st["lanes"][ln].get("item")
            if not it or it.get("status") != "running" or it.get("lab_job"):
                continue
            p = it["proposal"]
            follow = p.get("follow")
            child = p.get("child_exp")
            if follow in ("build", "experiment") and child:
                self._settle_build_job(it, queued, sacct)
                x = status.get(child) or {}
                if x.get("built"):
                    for s in st["segments"]:
                        if s["exp"] == child and not s.get("built_utc"):
                            s["built_utc"] = self.utc
                    if follow == "build" or (follow == "experiment" and x.get("done") and x.get("report") == "current"):
                        self._on_exp_done(child, x) if x.get("done") else None
                        self._done(ln, it)
                        continue
                    if follow == "experiment":
                        continue
                ended = queued is not None and not any(j.split("_")[0] in queued for j in it.get("job_ids") or []) \
                    and ("inc_build_%s" % child) not in names
                if ended:
                    it["lost"] = int(it.get("lost") or 0) + 1
                    if it["lost"] >= BUILD_LOST_SNAPSHOTS or any((sacct.get(j) or {}).get("state") not in
                                                                (None, "RUNNING", "PENDING", "COMPLETED")
                                                                for j in it.get("job_ids") or []):
                        prov = ((payload.get("status") or {}).get("builds") or {}).get(child) or {}
                        self._failed(ln, it, "the build of %s ended without the experiment%s" % (
                            child, (": %s" % prov.get("refusal")) if prov.get("refusal") else ""))
                continue
            if follow == "job":
                ids = [str(j) for j in it.get("job_ids") or []]
                if not ids and it.get("uncertain") and it["lever"] in RECORD_ONLY_LEVERS:
                    self._follow_record_only(ln, it, queued, names, arts)
                    continue
                if queued is None or any(j.split("_")[0] in queued for j in ids):
                    continue
                states = [((sacct.get(j) or {}).get("state") or "") for j in ids]
                if ids and not all(states):
                    it["lost"] = int(it.get("lost") or 0) + 1
                    if it["lost"] < BUILD_LOST_SNAPSHOTS:
                        continue
                ok = bool(ids) and all(s.startswith("COMPLETED") for s in states)
                # the refusal line of the job's log (stream_remote.sacct: data jobs only)
                refusal = "" if ok else "; ".join(str((sacct.get(j) or {}).get("refusal")) for j in ids
                                                  if (sacct.get(j) or {}).get("refusal"))
                if p["policy_action"] not in ("inc_label_audit",):
                    for j in ids:
                        if sacct.get(j):
                            B.record_job_spend(j, self.name, sacct[j], p["policy_action"], domain=self.domain,
                                               base_dir=self.xctx.su_base_dir, ts=self.utc)
                if ok:
                    self._done(ln, it)
                else:
                    self._failed(ln, it, "job(s) %s ended %s%s" % (",".join(ids), ",".join(states) or "unknown",
                                                                   ("; refused: %s" % refusal) if refusal else ""),
                                 charged=True, retry=not (p["policy_action"] == "inc_stream_admit"
                                                          and RETRY_FALSE_RE.search(refusal)))
        # a fork (L22) whose job this very snapshot shows ended: its summary
        # already names the new stream, so it is adopted now, not one tick
        # later on evidence that still reads the old stream's last commit
        self._adopt_fork(q)
        # experiments the stream reads: done -> frozen (spend recorded for its own)
        for exp in list(dict.fromkeys(list(st.get("exps") or []) + self._r0_exps())):
            x = status.get(exp) or {}
            if x.get("done") and x.get("report") == "current":
                self._on_exp_done(exp, x)
                sub = (payload.get("experiments") or {}).get(exp) or {}
                snap = sub.get("snapshot")
                if isinstance(snap, dict) and snap.get("built") and exp not in (st.get("frozen") or {}):
                    _write_json(self.paths.frozen(self.name, exp),
                                {"exp": exp, "frozen_utc": self.utc,
                                 "snapshot": {k: v for k, v in snap.items() if k != "display_only"}})
                    st.setdefault("frozen", {})[exp] = {"utc": self.utc}
        self._health(payload)

    def _follow_record_only(self, ln, it, queued, names, arts):
        """A record-only item (L23N) whose submission's outcome is unknown: it
        runs while a job of its name is queued; then it is done once its record
        (<exp>/native_rescore.json) says complete, and failed (record only: a
        card) after BUILD_LOST_SNAPSHOTS snapshots with neither."""
        from . import stream_remote as SR
        exp = ((it["proposal"].get("params") or {}).get("exp"))
        if queued is None or (SR.NATIVE_JOB_NAME % exp) in names:
            return
        if ((arts.get("%s/native_rescore.json" % exp) or {}).get("status")) == "complete":
            self._done(ln, it)
            return
        it["lost"] = int(it.get("lost") or 0) + 1
        if it["lost"] >= BUILD_LOST_SNAPSHOTS:
            self._failed(ln, it, "its submission's outcome was unknown, and in %d snapshots no job named %s was queued "
                                 "and %s/native_rescore.json was not complete" % (it["lost"], SR.NATIVE_JOB_NAME % exp,
                                                                                  exp), charged=True)

    def _settle_build_job(self, it, queued, sacct):
        """6.6: the build job itself (run_inc2_build.sh holds one V100 on
        GPU-shared while it builds) is settled from sacct once it has ended;
        the experiment's runs are settled by its report spend."""
        if it.get("build_settled") or queued is None:
            return
        ids = [str(j) for j in it.get("job_ids") or []]
        if not ids or any(j.split("_")[0] in queued for j in ids):
            return
        rows = {j: sacct.get(j) for j in ids}
        if not all(r and r.get("state") and not str(r["state"]).startswith(("RUNNING", "PENDING"))
                   for r in rows.values()):
            return
        for j, r in rows.items():
            B.record_job_spend(j, self.name, r, it["proposal"].get("policy_action"), domain=self.domain,
                               base_dir=self.xctx.su_base_dir, ts=self.utc)
        it["build_settled"] = self.utc

    def _adopt_fork(self, q):
        """The new stream version a finished fork (L22) created: the campaign's
        sid moves to it (its summary names it as forked_to)."""
        st = self.st
        if not (st.get("fork_pending") and NAME_RE.match(str((q or {}).get("forked_to") or ""))):
            return
        new_sid = str(q["forked_to"])
        old_sid = self.sid
        from . import campaign as C

        def change(c):
            c.setdefault("stream", {})["sid"] = new_sid
            return c
        try:
            C._update_config(self.cfg_hooks, self.name, change)
            self.cfg.setdefault("stream", {})["sid"] = new_sid
            # items proposed or filed for the old stream (it builds nothing
            # more) are withdrawn, not left to fail against it
            for ln in ALL_LANES:
                it = st["lanes"][ln].get("item") or {}
                prm = (it.get("proposal") or {}).get("params") or {}
                if it and it.get("status") in ("proposed", "filed") and (
                        prm.get("stream") == old_sid or str(prm.get("exp") or "").startswith(old_sid + "_")):
                    self._ledger("withdrawn", lane=ln, lever=it.get("lever"), approval_id=it.get("approval_id"),
                                 reasons=["the stream %s was forked to %s" % (old_sid, new_sid)])
                    st["declined"] = (list(st.get("declined") or []) + [it["proposal"]["id"]])[-200:]
                    self._clear(ln)
            self._ledger("fork_adopted", sid=new_sid, previous_sid=q.get("sid"), trigger=["D8S"],
                         reasons=["L22 doubled M: a new stream version adopts the pool and the quarantine"])
            st["fork_pending"] = None
            st["segments"], st["milestones"] = [], []
            # the evidence in hand is the old stream's: nothing is proposed on
            # it for the new one (its init would read as missing, and LI would
            # be proposed); the next tick's one call is the new stream's snapshot
            st["await_snapshot"] = new_sid
        except Exception as e:
            self.log.warning("[inc-stream] %s: the fork's sid not written: %s" % (self.name, e))

    def _on_exp_done(self, exp, x):
        st = self.st
        for s in st["segments"]:
            if s["exp"] == exp and not s.get("done"):
                s["done"] = True
                s["done_utc"] = self.utc
                self._ledger("segment_done", exp=exp)
        for m in st["milestones"]:
            if m["exp"] == exp and not m.get("done"):
                m["done"] = True
                q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
                m["pool"] = (q.get("pool") or {}).get("current") or m.get("pool")
                self._ledger("milestone_done", exp=exp, pool=m["pool"])
        reported = st.setdefault("reported", {})
        if exp in reported or exp not in (st.get("exps") or []):
            return
        rep = self.ev.json("%s/report.json" % exp) if self.ev is not None else None
        if isinstance(rep, dict) and rep.get("done"):
            r = B.record_report_spend(rep, self.name, domain=self.domain, base_dir=self.xctx.su_base_dir)
            reported[exp] = {"utc": self.utc, "su": r.get("su"), "ok": r.get("ok")}
            self._ledger("reported", exp=exp, spend={k: r.get(k) for k in ("ok", "su", "reason")})

    def _health(self, payload):
        """D5-D7 and D14 on every live experiment of the snapshot (the pinned
        driver's experiments: the autopilot's own health rules, diagnose.py),
        and the stops they call for. A measurement arm's (a baseline marked
        measure: record only, no lane waits for it) D5 and D6 never stop the
        stream: they are a person's card (_record_only)."""
        st = self.st
        out = []
        measure = self._measure_exps()
        sq = (payload.get("status") or {}).get("squeue") or {}
        names = [j.get("name") for j in sq.get("jobs") or [] if isinstance(j, dict)] if sq.get("ok") else None
        for exp, sub in sorted((payload.get("experiments") or {}).items()):
            snap = (sub or {}).get("snapshot") or {}
            if not snap.get("built"):
                continue
            hist = st.setdefault("history", {}).setdefault(exp, [])
            gen = (((snap.get("decision") or {}).get("artifacts") or {}).get("%s/state.json" % exp) or {}).get("generation")
            if gen is not None:
                hist.append({"utc": self.utc, "generation": gen})
                st["history"][exp] = hist[-48:]
            ctx = {"now_utc": self.utc, "history": list(st["history"][exp])}
            if names is not None:
                ctx["squeue"] = names
            adv = (sub or {}).get("advance")
            if isinstance(adv, dict) and adv.get("ok") is False and adv.get("error"):
                ctx["advance"] = {"error": _short(adv.get("error"), 2000), "error_kind": adv.get("error_kind")}
            rec = {"verb": "campaign-snapshot", "experiments": {exp: {"snapshot": {k: v for k, v in snap.items()
                                                                                    if k != "display_only"}}}}
            try:
                ev = E.from_snapshot(rec, exp, context=ctx, domain=LS.funnel_ref(self.dom))
            except Exception as e:
                self._once("health_ev:%s" % exp, str(e), "evidence_error", exp=exp, error=_short(e, 300))
                continue
            for d in DG.detect(ev, only=HEALTH_EXP):
                if d.get("fired"):
                    out.append(_record_only(d) if exp in measure and d["id"] == "D5" else d)
        st["health"] = out
        by = {}
        for d in out:
            by.setdefault(d["id"], []).append(d)
        for did, what in (("D14", "halt: a decision path touched a non-dev exam"),
                          ("D7", "code drift: the driver refused to advance on code")):
            if did in by:
                if did == "D7":
                    self._card("research", "X5: repin or sync the outer copy", by[did][0].get("summary"), lever="X5",
                               trigger=[did])
                return self._pause("%s (%s): %s" % (what, did, _short(by[did][0].get("summary"), 400)))
        for d in by.get("D5") or []:
            if "OP_PAUSE" in (d.get("levers") or []):
                return self._pause("a blocked unit of %s is not transient (D5): %s" % (d.get("exp"),
                                                                                   _short(d.get("summary"), 300)))
            if (d.get("detail") or {}).get("record_only"):
                self._measure_card(d)
            stop = st["lanes"]["STOP"]
            units = [u for u in (d.get("detail") or {}).get("units") or [] if u.get("transient")]
            if units and "L7" in (d.get("levers") or []) and stop.get("item") is None:
                u = units[0]
                argv = ["python", "-m", "weed_optimizer_framework.tools.inc.driver", "unblock", "--exp", d.get("exp"),
                        "--unit", u["unit"], "--reason", "auto: %s" % u["cause"]]
                p = {"id": _sha([self.name, "L7", argv])[:32], "lever": "L7", "family": "L7", "argv": argv,
                     "params": {"exp": d.get("exp"), "unit": u["unit"], "cause": u["cause"]},
                     "policy_action": "inc_unblock_transient", "risk": "R2", "trigger": ["D5"],
                     "cites": list(d.get("cites") or []), "lit": [], "control": "", "success": "", "falsifier": "",
                     "est_gpu_hours": 0.0, "proposed_by": AUTO, "lane": "STOP", "follow": "login",
                     "parent_exp": d.get("exp"), "child_exp": None, "attempt": 0}
                if p["id"] not in (st.get("declined") or []):
                    stop["item"] = {"proposal": p, "lever": "L7", "status": "proposed", "since_utc": self.utc,
                                    "attempt": 0, "trigger": "D5", "diagnoses": [d]}
                    self._ledger("proposed", lane="STOP", lever="L7", action="inc_unblock_transient", risk="R2",
                                 argv=argv, trigger=["D5"], proposal_id=p["id"])
        stale = st.setdefault("stale", {})
        for d in by.get("D6") or []:
            stale[d["exp"]] = int(stale.get(d["exp"]) or 0) + 1
            if stale[d["exp"]] >= STALE_TO_PAUSE:
                if d["exp"] in measure:
                    self._measure_card(d)
                    continue
                return self._pause("stale advance (D6) of %s on %d consecutive snapshots" % (d["exp"], stale[d["exp"]]))
        for exp in list(stale):
            if not any(d.get("exp") == exp for d in by.get("D6") or []):
                stale[exp] = 0

    def _measure_exps(self):
        """The measurement arms' experiments (baselines marked measure)."""
        return {b["exp"] for b in ((self.dom or {}).get("baselines") or {}).get("items") or [] if b.get("measure")}

    def _native_item(self, params):
        """The measurement baseline an L23N item rescores (by its --exp), or None."""
        return next((b for b in ((self.dom or {}).get("baselines") or {}).get("items") or []
                     if b.get("measure") and b.get("exp") == (params or {}).get("exp")), None)

    def _measure_card(self, d):
        """A measurement arm's D5 (a block that is not transient) or D6 (a
        stale advance): a person's card, once per block and generation, and
        the stream runs on; the experiment stays built, so DR0 does not
        propose it again."""
        exp = d.get("exp")
        gen = (((self.st.get("history") or {}).get(exp) or [{}])[-1]).get("generation")
        if self._once("measure_health:%s:%s" % (d["id"], exp), [(d.get("detail") or {}).get("units"), gen],
                      "measure_health", exp=exp, trigger=[d["id"]], summary=_short(d.get("summary"), 300)):
            self._card("escalation", "Measurement arm %s: %s" % (exp, d["id"]),
                       "%s. The arm is record only and the stream runs on; a person unblocks it (inc.driver "
                       "unblock) or leaves it" % _short(d.get("summary"), 400), trigger=[d["id"]])


# ------------------------------------------------------------------ status
def summary(cfg, st):
    """A compact, page-ready view of one stream campaign."""
    st = st or {}
    lanes = st.get("lanes") or {}
    alarm = "crit" if (cfg.get("paused_reason") or (st.get("paused") or {}).get("reason")) else \
        "off" if not cfg.get("enabled") else "warn" if (st.get("errors") or any(
            (lanes.get(ln) or {}).get("hold") for ln in LANES) or (st.get("card") or {}).get("kind") in
            ("approval", "escalation", "cluster", "drift", "stop_loss", "platform")) else "ok"
    segs = st.get("segments") or []
    return {"mode": "stream", "enabled": bool(cfg.get("enabled")), "alarm": alarm,
            "paused_reason": cfg.get("paused_reason") or (st.get("paused") or {}).get("reason"),
            "goal": cfg.get("goal"), "autonomy": cfg.get("autonomy"), "data_autonomy": cfg.get("data_autonomy"),
            "domain": cfg.get("domain"), "sid": (cfg.get("stream") or {}).get("sid"),
            "lanes": {ln: {"phase": (lanes.get(ln) or {}).get("phase"),
                           "item": ((lanes.get(ln) or {}).get("item") or {}).get("lever"),
                           "status": ((lanes.get(ln) or {}).get("item") or {}).get("status"),
                           "hold": (lanes.get(ln) or {}).get("hold") or (lanes.get(ln) or {}).get("diag_hold")}
                      for ln in ALL_LANES if ln in lanes or ln in LANES},
            "segments": len(segs), "committed": sum(1 for s in segs if s.get("committed")),
            "milestones": len(st.get("milestones") or []), "card": st.get("card"),
            "cards": (st.get("cards") or [])[-10:], "errors": st.get("errors"), "drift": st.get("drift"),
            "stage_a": st.get("stage_a"), "stage_c": st.get("stage_c"), "capacity": (st.get("capacity") or {}).get("chosen"),
            "complete_available": complete_available(st), "completed_by": cfg.get("completed_by"),
            "last_tick_utc": st.get("last_tick_utc"), "last_snapshot_utc": st.get("last_snapshot_utc"),
            "ticks": st.get("ticks")}


def complete_available(st):
    """6.7: COMPLETE is available to a person (never taken by the ticker) when
    D29 has moved DATA to WAIT_DATA, nothing is in flight in any lane, and every
    committed segment's increments have been through a milestone."""
    st = st or {}
    lanes = st.get("lanes") or {}
    if (lanes.get("DATA") or {}).get("phase") != "WAIT_DATA":
        return False
    if any((lanes.get(ln) or {}).get("item") for ln in ALL_LANES):
        return False
    segs = [s for s in st.get("segments") or [] if s.get("committed")]
    ms = [m for m in st.get("milestones") or [] if m.get("done")]
    last = max([m.get("after_segment") or 0 for m in ms] or [0])
    return len(segs) <= last


def release(name, by, cfg_hooks=None, lab_repo=None, clock=None):
    """A person releases the lanes of stream campaign `name` that a stop-loss
    held (6.6): the campaign is enabled again with a resume stamp
    (configure_stream enable), and the next tick frees every lane held before
    that stamp (StreamRun._resumed: 'lane_released' in the campaign ledger)
    and clears its steps' failure counts. Returns the lanes it will free."""
    from . import campaign as C
    cfg_hooks = cfg_hooks or C.default_cfg_hooks()
    raw = ((cfg_hooks[0]() or {}).get(C.CONFIG_KEY) or {}).get(name)
    if not isinstance(raw, dict) or raw.get("mode") != "stream":
        raise ValueError("%r is not a stream campaign" % (name,))
    c = stream_config(raw, name)
    st = load_state(StreamPaths(lab_repo, M.campaign_domain(c)), name) or {}
    held = {ln: (st.get("lanes") or {}).get(ln, {}).get("hold") for ln in LANES
            if ((st.get("lanes") or {}).get(ln) or {}).get("hold")}
    out = configure_stream(name, by, enable=True, cfg_hooks=cfg_hooks, lab_repo=lab_repo, clock=clock)
    return {"ok": True, "campaign": name, "released_by": by, "resumed_utc": out.get("resumed_utc"),
            "held_lanes": held}


def status(name=None, cfg_hooks=None, lab_repo=None):
    from . import campaign as C
    cfg_hooks = cfg_hooks or C.default_cfg_hooks()
    camps = (cfg_hooks[0]() or {}).get(C.CONFIG_KEY) or {}
    out = {}
    for n in sorted(camps):
        raw = camps.get(n)
        if (name and n != name) or not isinstance(raw, dict) or raw.get("mode") != "stream":
            continue
        c = stream_config(raw, n)
        paths = StreamPaths(lab_repo, M.campaign_domain(c))
        out[n] = {"config": c, "summary": summary(c, load_state(paths, n))}
    return out


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="inc_autopilot.stream", description="stream-mode campaigns "
                                 "(docs/CONTINUOUS_LOOP.md 6)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--lab-repo", default=None)
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("enable")
    e.add_argument("--name", required=True)
    e.add_argument("--by", required=True)
    e.add_argument("--domain", default=None)
    e.add_argument("--autonomy", choices=("off", "envelope"), default=None)
    e.add_argument("--data-autonomy", choices=("off", "on"), default=None)
    for f in ("--envelope-su", "--window-cap-su", "--daily-cap-su", "--alloc-reserve-su", "--collect-gb-envelope",
              "--collect-gb-daily"):
        e.add_argument(f, type=float, default=None)
    e.add_argument("--envelope-end-utc", default=None)
    e.add_argument("--protocol-v3-accepted", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--name", default=None)
    rl = sub.add_parser("release")
    rl.add_argument("--name", required=True)
    rl.add_argument("--by", required=True)
    c = sub.add_parser("complete")
    c.add_argument("--name", required=True)
    c.add_argument("--by", required=True)
    r = sub.add_parser("lab-run")
    r.add_argument("--spec", required=True)
    y = sub.add_parser("lab-sync")
    y.add_argument("--lab-inc", required=True)
    y.add_argument("--source", required=True)
    y.add_argument("--file", default=None)
    y.add_argument("--target", default="")
    y.add_argument("--data-target", default="")
    y.add_argument("--names", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "lab-run":
        return lab_run(a.spec)
    if a.cmd == "lab-sync":
        res = lab_sync(a.lab_inc, a.source, a.target, a.data_target or None, file=a.file, names=a.names)
        print(json.dumps(res, sort_keys=True))
        return 0 if res.get("ok") else 1
    from . import campaign as C
    hooks = C.default_cfg_hooks(a.config)
    try:
        if a.cmd == "enable":
            out = configure_stream(a.name, a.by, domain=a.domain, enable=True, autonomy=a.autonomy,
                                   data_autonomy=a.data_autonomy, envelope_su=a.envelope_su,
                                   window_cap_su=a.window_cap_su, daily_cap_su=a.daily_cap_su,
                                   alloc_reserve_su=a.alloc_reserve_su, collect_gb_envelope=a.collect_gb_envelope,
                                   collect_gb_daily=a.collect_gb_daily, envelope_end_utc=a.envelope_end_utc,
                                   protocol_v3_accepted=a.protocol_v3_accepted or None, cfg_hooks=hooks,
                                   lab_repo=a.lab_repo)
        elif a.cmd == "release":
            out = release(a.name, a.by, cfg_hooks=hooks, lab_repo=a.lab_repo)
        elif a.cmd == "status":
            out = status(a.name, hooks, a.lab_repo)
        elif a.cmd == "complete":
            st = status(a.name, hooks, a.lab_repo).get(a.name) or {}
            if not (st.get("summary") or {}).get("complete_available"):
                raise ValueError("COMPLETE is not available: 6.7's conditions do not all hold")
            out = configure_stream(a.name, a.by, complete=True, enable=False, cfg_hooks=hooks, lab_repo=a.lab_repo)
        else:
            ap.print_help(sys.stderr)
            return 2
    except ValueError as err:
        print(json.dumps({"ok": False, "error": str(err)}))
        return 1
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
