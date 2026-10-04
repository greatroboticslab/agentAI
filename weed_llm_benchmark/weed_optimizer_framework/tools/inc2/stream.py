"""The stream builder of the continuous loop: fixed-size increments cut from the
Step 1 queue, segments run by the pinned driver, the guard-based commit,
rollback and bisection, milestones and the hash-chained stream ledger
(docs/CONTINUOUS_LOOP.md §3.4-3.8, §5, decisions D-D and L-2..L-6).

    python -m weed_optimizer_framework.tools.inc2.stream init --stream SID [--base P] [--lock L] [--m M]
        [--k-max 4] [--arm n640] [--milestone0 EXP] [--stage-b r0[,x1a]] [--testing] [--decided-by WHO]
    ... choose-arm  --stream SID [--capacity INC_DIR/capacity/capacity_v1.json]
    ... cut         --stream SID                     (dry run: the next increment, or why there is none)
    ... build       --stream SID --k K [--recipes r0,x1a] [--exp SID_sNNN] [--no-truth --decided-by human:X]
                    [--arch yolo11n --imgsz 640] [--truth-every N]   (the arm must be the stream's; N can only
                                                                        make the truth arm more frequent)
    ... commit      --exp SID_sNNN
    ... milestone   --stream SID                     (builds SID_mNNN; once it is done, compares it)
    ... compare     --exp SID_mNNN | SID_c001 | SID_bNNN
    ... rollback    --stream SID --to P_c [--decided-by human:X]
    ... bisect      --stream SID --from P_c          (builds the arms; once they are done, decides)
    ... feasibility --stream SID --holdout tsw22 --m M [--recipes r0,x1a]   (M must be the stream's M)
    ... fork        --stream SID --m 2M [--to NEW_SID]
    ... quarantine  --source SLUG --cite D28 [--stream SID]
    ... unquarantine --source SLUG --stream SID --decided-by human:X
    ... release     --stream SID --hold KIND [--source S | --keys FILE] --decided-by human:X [--reason TEXT]
                    [--licence TEXT [--not-research-only]]
                    (KIND licence, join_conflict or funnel_F9; h6_scan is lifted only by the copy scan; the rows
                    a licence release lifts are research_only unless it records --licence and --not-research-only)
    ... withdraw    --exp SID_sNNN --reason TEXT
    ... summary     --stream SID                     (writes queue_summary.json)
    ... verify      --stream SID                     (the hash chain and every file it names)
    ... status      --stream SID

Every experiment the stream builds is named <sid>_[smcb]NNN: segments sNNN,
milestones mNNN, Stage C c001, bisect arms bNNN. --decided-by defaults to
'platform'; a person's approval the autopilot carries in INCAP_DECIDED_BY
('human:...') is taken when the flag is not given.

Everything lives in INC_DIR/stream/<sid>/ (this module is its one writer; a
writing verb holds stream.lease, inc.driver's O_EXCL lease):
    stream.json            the stream's definition, written once by init
    ledger.jsonl           append-only, hash-chained: every line carries seq and
                           prev_sha256 = sha256 of the file's bytes before it.
                           It is the only source of truth: state.json,
                           consumed.jsonl and quarantine_dhash.json are derived
                           from it after every operation (a crash between the
                           ledger append and the derived files is repaired by
                           the next operation)
    state.json             the folded ledger, for people
    consumed.jsonl         append-only (key, increment, disposition), owned here
    quarantine_dhash.json  every quarantined image's dHashes (data or neutral)
    queue_summary.json     what the autopilot reads (no test or ImageWeeds value)
    increments/<inc>.jsonl the increment's manifest (inc.common.MANIFEST_KEYS)
      <inc>.rows.jsonl     per row: the queue fields, the dHashes, provenance
      <inc>.meta.json      sources, species boxes, admission kinds, seed, pins
    pool/P_<s>.<sha16>.jsonl  the pool after segment s (P_0 = base_v2's bytes)
      P_<s>.dhash.json     key -> the row's dHashes; P_<s>.meta.json research_only
    milestones/mNNN/       incumbent.json (which chain incumbent inc2.baseline
                           secondary scores, in the milestone experiment's
                           runs/secondary__incumbent), comparison.json (the 5 v 5
                           dev decision) and research_log_entry.md (the entry
                           for RESEARCH_LOG, test included: for people only)
    bisect/, feasibility/  manifests of the L27 and L28 experiments
    report.{md,json}       inc2.stream_report

The cut (§3.4). Eligible: a queue row as step1_stream.load_queue folds it
(queue/events.jsonl: holds released by serve-holds, refusals, supersessions)
that is not refused, evidenced, with no hold left (a person's recorded
release, `release`, also lifts one), >= 1 kept verified target box (P4:
OtherPlant-only rows stay queued), not consumed, not of a quarantined source,
none of its dHashes within QUARANTINE_BITS of a quarantined image, none within
POOL_BITS of the current pool, of an increment in flight or of a suspect
increment, and not already in the pool by key, path or sha256. Draw units are
the transitive groups of eligible rows within UNIT_BITS (6), joined with
step1_stream's near-duplicate groups, so that increments of one pool are
pairwise disjoint at 6 bits as commit checks; a group larger than M (a video
or a tray chains its frames) is split into its 3-bit near-duplicate groups,
and no two of them go to different increments of one cut. A unit that still
exceeds M, or mixes verifier versions, is reported as uncuttable and is not
counted as supply. One increment is exactly M images, from one verifier
version. Order: the oldest source holding >= M eligible images
first (one source first), then the other sources oldest-first; within a
source, split-remainder capture groups first, then capture groups by their
oldest admission batch; within a capture group, select.balanced_order (select
.cluster_lists over l1 / l2), seeded by stable_int("<sid>/<segment>/<step>").
select.draw_parts([order], sizes, 1, M) takes units in that order, passing over
one that would overshoot: a capture group that does not fit is split by unit
and its remainder is cut first into the next increment. With two or more
sources eligible, no species may hold more than SPECIES_CAP (60 %) of the
increment's target boxes: the last-picked unit of the over-cap species is
swapped for later units of the same total size; what cannot be met is
recorded. Drawn rows then pass the image sha256 check and GuardV2 on the
dHash and its 8 flips and rotations of the training image and of the unmasked
original (fail closed), and the same 8 variants against the quarantine and
against the pool, the increments in flight and the suspect ones (a mirrored or
rotated re-upload); a hit excludes the whole unit and the draw is redone.
A cut that cannot fill M writes cut_refusal.json with the reason (D22), never
a silent wait.

A segment (§3.5) is an ordinary pinned-driver chain experiment <sid>_sNNN:
base = the current pool (cold, the arm's cold recipe), K increments as
clean: false steps, replay_mode full, gate {"flips_mode": "net"}, seeds 0-2,
the truth arm by the truth policy, final exams dev and imageweeds, plus
inc2.recipes.stamp(arm). Segment 1 of a stream (the first committed one) is
Stage B: R0 and the Stage A survivor; later segments run the chosen recipe.

The commit (§3.5) reads each step of each chain under Protocol v3's gate
(inc2.gate3.decide_experiment, L-3: the species guard with tol_s = max(0.03,
1.96 SE_s) from the incumbent's scorer sidecar; the driver's decision is
re-derived from the score files it names first), picks the chain by the
Stage B rule (inc2.recipes.stage_b_choice: smallest median delta_min, then
fewer recipe flags, then the cheaper recipe, a tie to r0) and disposes of
every increment by the guard-based rule (§3.5 [review]; never by
attribution.blame):
    accepted     the v3 commit verdict is ACCEPT: the increment joins P_s
    data         P_data <= p_reject unless the step's truth arm says helps:
                 quarantined (every dHash), never cut again
    species      the species guard failed \\
    flips        the flips guard failed    > return to the queue once; a second
    recipe       only the regression guard /  non-accept makes them 'neutral'
                 failed (not counted while D30 fires on the segment)
    hold         HOLD (guards pass, P_data between the thresholds)
    truth_helps  REJECT on P_data alone with the truth arm saying helps
P_s = P_{s-1} + the accepted increments, checked pairwise disjoint by key,
path, sha256 and 6-bit dHash. A segment whose base is no longer the current
pool (a rollback happened while it ran) returns its images uncounted.

A milestone (§3.6, §5.5) is inc2.baseline's 5-seed baseline <sid>_mNNN on the
current pool (finals dev, imageweeds and test) plus the chain incumbent scored
on the same exams. It is compared with the last good milestone on dev only:
gate.truth_detail and the one-sided 5 v 5 permutation test; only p <= 0.025
with mean(new) < mean(old) is "hurts" and recommends L21 to that milestone's
pool; a species-guard-only "hurts" is recorded for D33 / X17. Test values
never enter the ledger or queue_summary.json (test blindness, S12).

Nothing here trains, scores or imports Ultralytics.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ..inc import common as C
from ..inc import driver as D
from ..inc import gate as G
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NEAR_DUP_BITS, NearHashIndex

# ------------------------------------------------------------------ constants
FORMAT = "inc2-stream/1"
SUMMARY_FORMAT = "inc2-stream/queue-summary/1"
STATE_FORMAT = "inc2-stream-state/1"
QUARANTINE_FORMAT = "inc2-stream-quarantine/1"
STREAM_VERSION = 1
# weed_optimizer_framework.tools, from __package__: under `python -m ...inc2.stream` __name__ is "__main__",
# and a PKG derived from it made every _inc2() import fail as "not installed" (stream init, job 47276839)
PKG = (__package__ or "weed_optimizer_framework.tools.inc2").rsplit(".", 1)[0]
BUILDER = "inc2.stream build"
M_FRAC_NUM, M_FRAC_DEN = 1, 10                    # M = ceil(0.10 x |base_v2|), fixed per stream version (P2)
K_MAX = 4
SPECIES_CAP = 0.6                                 # §3.4 [review]
UNIT_BITS = HOLDOUT_NEAR_DUP_BITS                 # draw units: transitive 6-bit groups
SUB_UNIT_BITS = NEAR_DUP_BITS                     # a 6-bit group larger than M splits into its 3-bit groups (§3.4)
POOL_BITS = HOLDOUT_NEAR_DUP_BITS                 # commit's pairwise disjointness radius
QUARANTINE_BITS = NEAR_DUP_BITS                   # a re-harvested photograph of a quarantined image
CRASH_GRACE_S = 2 * D.LEASE_SECONDS               # a ledger line whose external step never followed, this old, is repaired
QUARANTINE_CITES = ("D28", "D31")                 # L24 (§6.3): a firing D28 or D31; any other cite is a person's
# A person may release these holds (§6.7: funnel_F9 at its deadline; licence and join conflicts row by row or by
# source). h6_scan is never released by hand: the copy scan is mandatory (D-C, P9) and only step1_stream's
# serve-holds (the scan itself) lifts it.
RELEASABLE_HOLDS = ("licence", "join_conflict", "funnel_F9")
STALE_DAYS = 7                                    # D22: Q >= M and the oldest eligible row this old
CHAIN_SEEDS = (0, 1, 2)
MILESTONE_SEEDS = (0, 1, 2, 3, 4)
BISECT_SEEDS = (0, 1, 2)
SECONDARY_RUN = "secondary__incumbent"            # inc2.baseline secondary's default run id
PERM_ALPHA = 0.025                                # §3.6: one-sided 5 v 5 permutation test
BOUNDARY_SD_MULT = 2.0                            # §5.5.1
DEFAULT_ARM = "n640"
DEFAULT_STAGE_B = ("r0",)
R0 = "r0"
SEGMENT_EXAMS = ("dev", "imageweeds")
MILESTONE_EXAMS = ("dev", "imageweeds", "test")
BISECT_EXAMS = ("dev",)                           # a bisect arm never reads test (P10)
GATE_BLOCK = {"flips_mode": "net"}
MILESTONE_TRIGGERS = {"accepted_increments": 4, "segments": 3, "days_with_accepted": 30}
# The budget a new stream's definition records (stream.json "budget"; the
# report reads it). L-2's monthly window (350 SU) and daily cap (120 SU) were
# removed on 2026-10-04 by the owner (docs/CONTINUOUS_LOOP.md 6.6): they only
# delayed healthy work (about 9 h of an idle cluster that day). A stream
# defined before then still records them in its definition; what is enforced
# is the campaign's config (inc_autopilot.budget), never this record.
BUDGET = {"envelope_su": 1000, "until": "2026-12-31",
          "decided": "L-2 (2026-09-28); its monthly window and daily cap removed 2026-10-04 (owner)"}
HOLD_KINDS = ("h6_scan", "licence", "join_conflict", "funnel_F9")
# GuardV2 reasons that are harmless for a queue row at cut time: the Step 1
# seen / intake indexes (the row itself, admitted earlier). Anything else
# refuses the row's unit, including a reason this module does not know.
HARMLESS_GUARD_REASONS = ("exact_dup", "near_dup", "near_dup_intake", "near_consumed", "near_intake")
ACCEPTED, DATA, SPECIES, FLIPS, RECIPE, HOLD_D, TRUTH_HELPS = (
    "accepted", "data", "species", "flips", "recipe", "hold", "truth_helps")
RETURN_DISPOSITIONS = (SPECIES, FLIPS, RECIPE, HOLD_D, TRUTH_HELPS)
DISPOSITIONS = (ACCEPTED, DATA) + RETURN_DISPOSITIONS
DISPOSITION_RULE = (
    "Protocol v3 (docs/CONTINUOUS_LOOP.md §3.5 [review]), per increment of the chosen chain, the first that "
    "applies: ACCEPT (v3 gate) -> accepted; P_data <= p_reject unless the step's truth arm says helps -> data "
    "(quarantined by dHash); the v3 species guard failed -> species; the flips guard failed -> flips; only the "
    "regression guard failed -> recipe (not counted while D30 fires on the segment); HOLD -> hold; REJECT on "
    "P_data alone with truth helps -> truth_helps. Every non-data non-accept returns the images once; a second "
    "counted non-accept makes them neutral (quarantined). attribution.blame is recorded, never read.")
STAGE_B_RULE = ("Stage B (§5.1), inc2.recipes.stage_b_choice on the pinned ledger's gate entries: per recipe, "
                "delta_min = max(0, inc - mean(null) - 2 sd(null)) per step; the smallest median delta_min, then "
                "fewer recipe flags (P_recipe <= p_recipe_flag), then the cheaper recipe (fewer epochs), a tie to r0. "
                "Test is never read.")
# Status of a key in the stream (the fold): what makes it eligible again.
ELIGIBLE_STATUSES = (None, "returned", "released", "returned_bisect")
EVENTS = ("init", "prospective", "code_change", "arm", "cut", "build", "commit", "return", "quarantine",
          "unquarantine", "withdraw", "rollback", "milestone", "bisect", "feasibility", "fork", "release")
# Per-row reasons of the variant check at the cut (all 8 flips and rotations of the drawn images against the
# quarantine and against the pool, the increments in flight and the suspect ones).
VARIANT_QUARANTINE, VARIANT_POOL = "variant:quarantined_dhash", "variant:near_pool_or_in_flight"
EXIT_OK, EXIT_REFUSED, EXIT_BUSY = 0, 1, 3


class StreamError(RuntimeError):
    """A condition under which the stream must not go on (fail closed)."""


class LeaseBusy(StreamError):
    """Another writer holds stream.lease."""


class CutRefused(StreamError):
    """The cutter cannot fill exactly M images; .reason says why (D22)."""

    def __init__(self, msg, reason=None):
        super().__init__(msg)
        self.reason = reason or {"why": msg}


def log(msg):
    print("[inc2.stream] %s" % msg, flush=True)


# ----------------------------------------------------------------------- io
def _utc(ts=None):
    return D._utc(ts)


def _ts(utc):
    return D._ts(utc)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_obj(obj):
    return _sha_bytes(_canon(obj).encode("utf-8"))


def _write_json(path, obj):
    D._write_json(path, obj)
    return C.sha256_file(path)


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return C.sha256_file(path)


def _write_jsonl(path, rows):
    return _write_text(path, "".join(_canon(r) + "\n" for r in rows))


def _read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as e:
        raise StreamError("unreadable JSON %s (%s)" % (path, e))


def _read_jsonl(path, tolerate_partial=True):
    """Rows of a JSON-lines file; a partial last line (an append in progress
    or killed) is skipped, any other bad line refuses."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        return []
    lines = data.split(b"\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            if tolerate_partial and i == len(lines) - 1:
                continue
            raise StreamError("%s line %d is not JSON" % (path, i + 1))
    return out


def _file_rec(path):
    path = Path(path)
    return {"path": str(path), "sha256": C.sha256_file(path)}


def _check_rec(rec, what):
    p = Path(rec["path"])
    if not p.is_file():
        raise StreamError("%s %s is missing" % (what, p))
    got = C.sha256_file(p)
    if got != rec["sha256"]:
        raise StreamError("%s %s hashes to %s, the ledger recorded %s" % (what, p, got[:12], rec["sha256"][:12]))
    return p


def _as_hash(v):
    """An int dHash from an int or a decimal / hex string; None otherwise."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 0 <= v < 2 ** 64 else None
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            h = int(s, 16) if s.startswith("0x") else int(s)
        except ValueError:
            return None
        return h if 0 <= h < 2 ** 64 else None
    return None


def _mean(xs):
    xs = [float(x) for x in xs]
    return math.fsum(xs) / len(xs) if xs else None


def _sd(xs):
    xs = [float(x) for x in xs]
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _median(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])


def species_names():
    return list(C.CLASS_NAMES[:C.OTHER_PLANT])


def _species_vec(row):
    """The 12 kept verified target boxes of a queue row, as ints (a dict by
    name or a list in class-id order)."""
    v = row.get("species_boxes")
    names = species_names()
    if isinstance(v, dict):
        vec = [int(v.get(n, 0) or 0) for n in names]
    elif isinstance(v, (list, tuple)) and len(v) == len(names):
        vec = [int(x or 0) for x in v]
    else:
        return None
    return vec if all(x >= 0 for x in vec) else None


def _check_actor(who):
    who = str(who or "platform").strip()
    if not re.match(r"^(platform|human|human-delegated)(:[^\s]+)?$", who):
        raise StreamError("--decided-by %r: 'platform', 'human:<id>' or 'human-delegated:<id>'" % who)
    return who


def _human(who):
    return str(who).startswith(("human:", "human-delegated:"))


def _grid_arms(RC):
    """The arms a stream may run: L-4's capacity grid (inc2.recipes.GRID_ARMS;
    every arm of a recipes module that names no grid). A measurement arm
    (inc2.recipes.MEASURE_ARMS) is recorded, never a stream's arm."""
    return tuple(getattr(RC, "GRID_ARMS", None) or RC.ARMS)


def permutation_p(new, old):
    """One-sided exact permutation p of mean(new) - mean(old) (small = new
    below old): the share of all relabellings whose difference is <= the
    observed one (ties count)."""
    allv = [float(x) for x in new] + [float(x) for x in old]
    n = len(new)
    if n < 1 or len(old) < 1:
        raise StreamError("a permutation test needs both groups")
    obs = _mean(new) - _mean(old)
    total = hit = 0
    for idx in itertools.combinations(range(len(allv)), n):
        s = set(idx)
        a = [allv[i] for i in idx]
        b = [allv[i] for i in range(len(allv)) if i not in s]
        total += 1
        if _mean(a) - _mean(b) <= obs + 1e-12:
            hit += 1
    return hit / float(total)


# ------------------------------------------------------------------- paths
def stream_root():
    return Path(C.INC_DIR) / "stream"


class StreamPaths:
    def __init__(self, sid):
        self.sid = D.check_name(sid, "stream")
        self.root = stream_root() / sid
        r = self.root
        self.stream_json = r / "stream.json"
        self.ledger = r / "ledger.jsonl"
        self.state = r / "state.json"
        self.lease = r / "stream.lease"
        self.consumed = r / "consumed.jsonl"
        self.quarantine = r / "quarantine_dhash.json"
        self.summary = r / "queue_summary.json"
        self.cut_refusal = r / "cut_refusal.json"
        self.increments = r / "increments"
        self.pool = r / "pool"
        self.milestones = r / "milestones"
        self.bisect = r / "bisect"
        self.feasibility = r / "feasibility"
        self.dhash_cache = r / "dhash_cache.jsonl"
        self.report_json = r / "report.json"
        self.report_md = r / "report.md"

    def inc_manifest(self, inc):
        return self.increments / ("%s.jsonl" % inc)

    def inc_rows(self, inc):
        return self.increments / ("%s.rows.jsonl" % inc)

    def inc_meta(self, inc):
        return self.increments / ("%s.meta.json" % inc)

    def milestone_dir(self, n):
        return self.milestones / ("m%03d" % n)


def segment_exp(sid, n):
    return "%s_s%03d" % (sid, n)


def milestone_exp(sid, n):
    return "%s_m%03d" % (sid, n)


def feasibility_exp(sid, n=1):
    return "%s_c%03d" % (sid, n)


def bisect_exp(sid, n):
    """The n-th bisect arm of the stream (one per suspect increment, numbered
    across rollbacks; the ledger maps each to its increment)."""
    return "%s_b%03d" % (sid, n)


def inc_name(n):
    return "inc%04d" % n


def pool_name(s):
    return "P_%d" % s


def step1_root():
    return Path(C.INC_DIR) / "step1_stream"


# ------------------------------------------------------------------ ledger
class Ledger:
    """ledger.jsonl: seq and prev_sha256 (of the file's bytes before the line)
    on every line. read() verifies the chain; append() refuses when the file
    changed since this object read it (one writer, fenced)."""

    def __init__(self, path):
        self.path = Path(path)
        self.n = 0
        self.head = _sha_bytes(b"")

    def read(self, repair=False):
        if not self.path.is_file():
            self.n, self.head = 0, _sha_bytes(b"")
            return []
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            if repair:
                keep = self.path.with_name("ledger.partial.%s.txt" % _utc().replace(":", "").replace("-", ""))
                keep.write_bytes(data[cut:])
                with open(self.path, "r+b") as fh:
                    fh.truncate(cut)
                    fh.flush()
                    os.fsync(fh.fileno())
                log("WARNING: %s ended in a partial line (%d bytes, an append killed mid-write); cut back, "
                    "fragment kept in %s" % (self.path, len(data) - cut, keep))
            data = data[:cut]
        h = hashlib.sha256()
        out = []
        for i, line in enumerate(data.split(b"\n")[:-1]):
            prev = h.hexdigest()
            try:
                e = json.loads(line)
            except ValueError:
                raise StreamError("%s line %d is not JSON: the ledger is broken" % (self.path, i + 1))
            if not isinstance(e, dict) or e.get("seq") != i or e.get("prev_sha256") != prev:
                raise StreamError("%s line %d: seq %r / prev_sha256 %s do not match the file before it (%d, %s): "
                                  "the ledger chain is broken, refusing" % (self.path, i + 1,
                                                                           (e or {}).get("seq") if isinstance(e, dict) else None,
                                                                           str((e or {}).get("prev_sha256") if isinstance(e, dict) else None)[:12],
                                                                           i, prev[:12]))
            h.update(line + b"\n")
            out.append(e)
        self.n, self.head = len(out), h.hexdigest()
        return out

    def append(self, entry):
        data = self.path.read_bytes() if self.path.is_file() else b""
        if _sha_bytes(data) != self.head:
            raise StreamError("%s changed since this operation read it: another writer, refusing" % self.path)
        e = dict(entry, seq=self.n, prev_sha256=self.head)
        line = (json.dumps(e, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self.head = _sha_bytes(data + line)
        self.n += 1
        return e


def verify_ledger(sid):
    """The stream's ledger entries after checking the chain (StreamError on a break)."""
    return Ledger(StreamPaths(sid).ledger).read(repair=False)


# ---------------------------------------------------------- dependencies
def _inc2(name):
    """inc2.<name> (another group's module), or None when it is not installed."""
    try:
        return importlib.import_module("%s.inc2.%s" % (PKG, name))
    except ImportError:
        return None


class Deps:
    """Everything the stream reaches outside itself; tests pass doubles.

    guard            an object with check(dhash, variants) -> (reason, match)
                     (inc2.guard.GuardV2.load(LOCK v2) by default)
    hasher           path -> (dhash, {variant: dHash}) of the stored pixels
                     (inc2.guard.image_hashes by default)
    gate3            Protocol v3's gate (inc2.gate3; see Gate3)
    recipes          inc2.recipes (table(arm), resolve_arm, stamp, step_cost,
                     truth_every, ARMS)
    backend          inc.driver's backend (SlurmBackend by default)
    baseline_runner  argv -> exit status of `python -m ...inc2.baseline <argv>`
    secondary_runner (milestone exp, weights, source) -> the sbatch argv that
                     inc2.baseline secondary writes for the incumbent's scores
    submitter        argv -> job id (sbatch --parsable)"""

    def __init__(self, guard=None, hasher=None, gate3=None, recipes=None, backend=None, baseline_runner=None,
                 secondary_runner=None, submitter=None, repo=None):
        self.guard = guard
        self.hasher = hasher
        self._gate3 = gate3
        self._recipes = recipes
        self.backend = backend
        self.baseline_runner = baseline_runner or _run_baseline_cli
        self.secondary_runner = secondary_runner or _run_secondary_cli
        self.submitter = submitter or _submit
        self.repo = repo

    def get_guard(self, lock_path, testing=False):
        if self.guard is None:
            g = _inc2("guard")
            if g is None:
                raise StreamError("inc2.guard (GuardV2, splits v2) is not installed: the stream cannot clear a row "
                                  "against the never-train v2 index (fail closed)")
            try:
                self.guard = g.GuardV2.load(lock_path)
            except Exception as e:  # noqa: BLE001 - any failure to load is a refusal
                raise StreamError("GuardV2.load(%s) refused: %s: %s" % (lock_path, type(e).__name__, e))
        return self.guard

    def get_hasher(self):
        if self.hasher is None:
            g = _inc2("guard")
            if g is None:
                raise StreamError("inc2.guard is not installed: no dHash variants (fail closed)")
            self.hasher = g.image_hashes
        return self.hasher

    def gate3(self):
        return Gate3(self._gate3)

    def recipes(self):
        if self._recipes is None:
            self._recipes = _inc2("recipes")
            if self._recipes is None:
                raise StreamError("inc2.recipes (Protocol v3's recipe table) is not installed")
        return self._recipes


def _run_baseline_cli(argv):
    cmd = [sys.executable, "-u", "-m", "%s.inc2.baseline" % PKG] + list(argv)
    log("running %s" % " ".join(cmd))
    return subprocess.run(cmd).returncode


def _run_secondary_cli(exp, weights, source):
    cmd = [sys.executable, "-u", "-m", "%s.inc2.baseline" % PKG, "secondary", "--exp", exp, "--weights", str(weights),
           "--source", str(source)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise StreamError("inc2.baseline secondary exited %d: %s" % (p.returncode, (p.stderr or "")[-400:]))
    for ln in reversed(p.stdout.splitlines()):
        try:
            argv = json.loads(ln)
        except ValueError:
            continue
        if isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv):
            return argv
    raise StreamError("inc2.baseline secondary printed no argv")


def _submit(argv):
    if not argv or argv[0] != "sbatch":
        raise StreamError("not an sbatch argv: %s" % argv)
    p = subprocess.run(list(argv), capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise StreamError("sbatch exited %d: %s" % (p.returncode, (p.stderr or "")[-300:]))
    return D.parse_parsable(p.stdout)


# ------------------------------------------------------------------- gate3
GUARD_NAMES = ("regression", "species", "flips")


class Gate3:
    """Protocol v3's reading of a finished chain experiment (L-3), through
    group B's inc2.gate3.decide_experiment(exp): every gate and truth entry of
    the pinned driver's ledger re-decided with tol_s = max(0.03, 1.96 SE_s)
    from the incumbent's scorer sidecar, the score files checked against the
    sha256 the ledger recorded and the pinned decision reproduced first. It
    writes INC_DIR/<exp>/gate3.json; the stream records that file's sha256.
    A step inc2.gate3 cannot decide ('unavailable': no sidecar, a refused
    input) keeps its pinned verdict as commit_verdict (gate3's rule: nothing
    is accepted on a rule that could not be applied); the stream then reads
    the pinned guards and records the step as v3-unavailable. Without
    inc2.gate3 installed a segment is not read at all (fail closed)."""

    def __init__(self, module=None):
        self._m = module

    def module(self):
        if self._m is None:
            self._m = _inc2("gate3")
        if self._m is None:
            raise StreamError("inc2.gate3 (Protocol v3's gate, L-3) is not installed: a segment cannot be read "
                              "(fail closed)")
        return self._m

    def read(self, exp):
        """(document, {(chain, k): step record}, {k: truth record}, file record)."""
        m = self.module()
        try:
            doc = m.decide_experiment(exp)
        except Exception as e:  # noqa: BLE001 - gate3's refusal is the stream's refusal
            raise StreamError("inc2.gate3 refused %s: %s: %s" % (exp, type(e).__name__, e))
        steps = {(s.get("chain"), int(s.get("k"))): s for s in doc.get("steps", [])}
        truth = {int(t.get("k")): t for t in doc.get("truth", [])}
        out = doc.get("out")
        rec = _file_rec(out) if out and Path(out).is_file() else {"path": None, "sha256": _sha_obj(doc)}
        return doc, steps, truth, rec


def normalise(step3, entry):
    """One step as the stream records it: the v3 record's commit verdict and
    failed guards (the pinned ones when v3 was unavailable), with the pinned
    decision's statistics."""
    d = entry["decision"]
    applied = bool(step3 and step3.get("v3_applied"))
    if applied:
        failed = list(step3.get("failed_guards") or [])
        sp_failed = list(((step3.get("guards") or {}).get("species") or {}).get("failed") or [])
        tol = {s: (step3.get("species_tolerance") or {}).get(s) for s in sp_failed}
        se = {s: (step3.get("species_se") or {}).get(s) for s in sp_failed}
        blame = (step3.get("attribution") or {}).get("blame")
    else:
        failed = [g for g in GUARD_NAMES if not d["guards"][g]["passed"]]
        sp_failed = list(d["guards"]["species"]["failed"])
        tol, se = {}, {}
        blame = d["attribution"].get("blame")
    verdict = (step3 or {}).get("commit_verdict") or d["verdict"]
    return {"verdict": verdict, "v3_applied": applied,
            "v3_unavailable_reason": None if applied else (step3 or {}).get("reason", "no gate3 record"),
            "p_data": float(d["p_data"]), "p_recipe": float(d["p_recipe"]), "inc": float(d["inc"]),
            "cand_mean": float(d["cand_mean"]), "null_mean": float(d["null_mean"]), "null_sd": float(d["null_sd"]),
            "guards": {g: g not in failed for g in GUARD_NAMES}, "species_failed": sp_failed,
            "species_tolerance": tol, "species_se": se, "blame": blame,
            "recipe_flag": d["attribution"].get("recipe_flag")}


def dispose(v3, truth, p_reject):
    """The guard-based disposition of one non-accepted (or accepted) step
    (module docstring; §3.5 [review]). truth is the step's truth verdict or
    None when the segment ran without the truth arm."""
    if v3["verdict"] == G.ACCEPT:
        return ACCEPTED
    if v3["p_data"] <= p_reject + 1e-12 and truth != G.HELPS:
        return DATA
    g = v3["guards"]
    if not g["species"]:
        return SPECIES
    if not g["flips"]:
        return FLIPS
    if not g["regression"]:
        return RECIPE
    if v3["verdict"] == G.HOLD:
        return HOLD_D
    return TRUTH_HELPS


# ------------------------------------------------------------------ queue
class QueueView:
    """The Step 1 queue as the cutter sees it, read-only: step1_stream's own
    reader (inc2.step1_stream.load_queue: queue/queue.jsonl with
    queue/events.jsonl folded in: hold releases and additions, refusals
    (superseded rows included), near-dup groups merged, 'evidenced' from the
    current evidence index), so the queue's semantics are defined once, by
    its writer. Without that module the queue is not read at all (a second
    fold here would disagree with the writer's). The bytes of both files are
    hashed so a cut records what it saw."""

    def __init__(self, root=None):
        self.root = Path(root or step1_root())
        self.queue_path = self.root / "queue" / "queue.jsonl"
        self.events_path = self.root / "queue" / "events.jsonl"
        self.rows = collections.OrderedDict()
        self.version_order = []
        self.sha256 = None
        self.events_sha256 = None
        self.reader = None
        self._batch_times = {}

    @staticmethod
    def _bytes(path):
        try:
            data = Path(path).read_bytes()
        except FileNotFoundError:
            return b""
        if data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]           # an append in progress
        return data

    def load(self):
        self.sha256 = _sha_bytes(self._bytes(self.queue_path))
        self.events_sha256 = _sha_bytes(self._bytes(self.events_path))
        S1 = _inc2("step1_stream")
        if S1 is None or not hasattr(S1, "load_queue") or not hasattr(S1, "Layout"):
            # no silent second reader: the writer's fold also recomputes 'evidenced', merges near-duplicate groups
            # and applies a licence release's research_only; a private fold would disagree (fail closed)
            raise StreamError("inc2.step1_stream (the queue's one reader, load_queue) cannot be imported: the "
                              "queue is not read (fail closed)")
        try:
            rows = S1.load_queue(S1.Layout(root=self.root, inc_dir=self.root.parent))
        except Exception as e:  # noqa: BLE001 - the writer's refusal is the cutter's
            raise StreamError("step1_stream.load_queue refused %s: %s: %s" % (self.root, type(e).__name__, e))
        self.reader = "inc2.step1_stream.load_queue"
        for r in rows:
            self.rows[r["key"]] = r
            v = str(r.get("verifier") or "unknown")
            if v not in self.version_order:
                self.version_order.append(v)
        return self

    def batch_time(self, batch):
        """The UTC time a batch was committed (its batch.json), or None."""
        if batch not in self._batch_times:
            bj = _read_json(self.root / "batches" / str(batch) / "batch.json", None) or {}
            t = None
            for k in ("committed_utc", "finished_utc", "created_utc", "utc"):
                if bj.get(k):
                    t = bj[k]
                    break
            self._batch_times[batch] = t
        return self._batch_times[batch]

    def admitted_utc(self, row):
        return row.get("admitted_utc") or self.batch_time(row.get("batch"))


def _hold_rank(h):
    return HOLD_KINDS.index(h) if h in HOLD_KINDS else len(HOLD_KINDS)


def holds_of(row):
    """The holds a queue row still carries: 'holds' when the row has the
    list (step1_stream's folded rows), else its 'hold_until'."""
    if isinstance(row.get("holds"), list):
        return sorted({h for h in row["holds"] if h}, key=_hold_rank)
    return [row["hold_until"]] if row.get("hold_until") else []


def is_masked(row):
    """A masked admission (D-B): it trains on the masked PNG ('image'), and
    its photograph is 'unmasked_image'."""
    return row.get("admission") == "masked" or bool(
        row.get("unmasked_image") and row.get("unmasked_image") != row.get("image"))


def manifest_fields(row):
    """The inc.common.MANIFEST_KEYS of a queue row (the cutter's manifest
    row), or None when the row does not carry them."""
    if not all(row.get(k) for k in ("image", "label", "sha256", "label_sha256", "key")):
        return None
    return {"image": str(row["image"]), "label": str(row["label"]), "sha256": str(row["sha256"]),
            "label_sha256": str(row["label_sha256"]), "source": str(row.get("source") or ""),
            "session": str(row.get("session") or row.get("capture_group") or ""), "key": str(row["key"])}


# ------------------------------------------------------------------- fold
class Fold:
    """The stream's state, folded from its ledger (one function for replay and
    for the running operation, so the two can never disagree)."""

    def __init__(self, paths):
        self.p = paths
        self.defn = None
        self.sid = paths.sid
        self.pools = collections.OrderedDict()
        self.pool = None
        self.segments = collections.OrderedDict()      # n -> record
        self.increments = collections.OrderedDict()    # id -> record
        self.keys = {}                                 # key -> {"status", "increment", "returns"}
        self.transitions = []                          # consumed.jsonl rows, in ledger order
        self.quarantine = []                           # {"dhash", "key", "increment", "reason", "seq"}
        self.q_sources = {}
        self.releases = []
        self.milestones = collections.OrderedDict()    # n -> record (0 = milestone 0)
        self.rollbacks = []
        self.bisects = {}
        self.feasibility = None
        self.arm = None
        self.stage_b = list(DEFAULT_STAGE_B)
        self.chosen_recipe = None
        self.code = None
        self.forked_to = None
        self.next_inc = 1
        self.next_seg = 1
        self.next_ms = 1
        self.events = 0
        self.head = None
        self.prospective = None
        self._rows_cache = {}

    # ---- reading what the ledger names
    def inc_rows(self, inc):
        """{key: row sidecar} of an increment (sha256 checked)."""
        if inc not in self._rows_cache:
            rec = self.increments[inc]
            _check_rec(rec["rows_file"], "increment rows")
            self._rows_cache[inc] = collections.OrderedDict(
                (r["key"], r) for r in _read_jsonl(rec["rows_file"]["path"], tolerate_partial=False))
        return self._rows_cache[inc]

    def pool_rows(self, name):
        rec = self.pools[name]
        _check_rec(rec, "pool %s" % name)
        return C.read_manifest(rec["path"])

    def pool_hashes(self, name):
        rec = self.pools[name]
        _check_rec(rec["dhash"], "pool %s dHash map" % name)
        return {k: [int(h) for h in v] for k, v in (_read_json(rec["dhash"]["path"]) or {}).get("hashes", {}).items()}

    # ---- the fold
    def apply(self, e):
        fn = getattr(self, "_ev_" + e["event"], None)
        if fn is None:
            raise StreamError("ledger line %s: unknown event %r" % (e.get("seq"), e.get("event")))
        fn(e)
        self.events = e["seq"] + 1

    def _set_key(self, e, key, status, inc, disposition=None, counted=None):
        st = self.keys.setdefault(key, {"status": None, "increment": None, "returns": 0})
        st["status"] = status
        st["increment"] = inc
        if disposition is not None:
            row = {"key": key, "increment": inc, "disposition": disposition, "seq": e["seq"]}
            if counted is not None:
                row["counted"] = bool(counted)
            self.transitions.append(row)

    def _quarantine_inc(self, e, inc, keys, reason):
        rows = self.inc_rows(inc)
        for k in keys:
            for h in rows[k].get("hashes", {}).values():
                if h is not None:
                    self.quarantine.append({"dhash": int(h), "key": k, "increment": inc, "reason": reason,
                                            "seq": e["seq"]})

    def _ev_init(self, e):
        self.defn = _read_json(self.p.stream_json)
        if self.defn is None or C.sha256_file(self.p.stream_json) != e["stream_json_sha256"]:
            raise StreamError("%s is missing or differs from the one init recorded" % self.p.stream_json)
        pool = dict(e["pool"])
        self.pools[pool["name"]] = pool
        self.pool = pool["name"]
        self.arm = e.get("arm")
        self.stage_b = list(e.get("stage_b") or DEFAULT_STAGE_B)
        self.code = e.get("code")
        if e.get("milestone0"):
            self.milestones[0] = {"n": 0, "exp": e["milestone0"], "pool": pool["name"], "state": "external",
                                  "verdict": "baseline", "good": True}
        inh = e.get("inherited")
        if inh:
            for r in _read_jsonl(_check_rec(inh["keys"], "inherited key states")):
                self.keys[r["key"]] = {"status": r["status"], "increment": r.get("increment"),
                                       "returns": int(r.get("returns", 0))}
            q = _read_json(_check_rec(inh["quarantine"], "inherited quarantine"))
            for x in q.get("entries", []):
                self.quarantine.append(dict(x, seq=e["seq"], inherited=True))
            self.q_sources = dict(inh.get("quarantined_sources") or {})
            self.chosen_recipe = inh.get("chosen_recipe")
            # a person's releases stay released in the new stream version (a fork is the platform's L22, which
            # must not undo a person's decision)
            for r in inh.get("releases") or []:
                self.releases.append(dict(r, seq=e["seq"], inherited=True))

    def _ev_prospective(self, e):
        self.prospective = e["record_sha256"]

    def _ev_code_change(self, e):
        self.code = e["new"]

    def _ev_arm(self, e):
        self.arm = e["arm"]
        if e.get("milestone0"):
            self.milestones[0] = {"n": 0, "exp": e["milestone0"], "pool": self.defn_pool0(), "state": "external",
                                  "verdict": "baseline", "good": True}

    def defn_pool0(self):
        return next(iter(self.pools))

    def _ev_cut(self, e):
        inc = e["increment"]
        self.increments[inc] = {"id": inc, "segment": e["segment"], "step": e["step"], "manifest": e["manifest"],
                                "rows_file": e["rows"], "meta": e["meta"], "status": "in_segment",
                                "sources": e.get("sources"), "species_boxes": e.get("species_boxes"),
                                "n_images": e["manifest"]["n_images"], "split_capture_groups":
                                e.get("split_capture_groups", []), "disposition": None, "cut_seq": e["seq"],
                                "utc": e["utc"]}
        for k in self.inc_rows(inc):
            self._set_key(e, k, "in_increment", inc)
        self.next_inc = max(self.next_inc, int(inc[3:]) + 1)

    def _ev_build(self, e):
        n = e["segment"]
        self.segments[n] = {"n": n, "exp": e["exp"], "state": "built", "base_pool": e["base_pool"],
                            "increments": list(e["increments"]), "recipes": list(e["recipes"]),
                            "truth": e["truth"], "truth_policy": e.get("truth_policy"), "k": len(e["increments"]),
                            "k_requested": e.get("k_requested"), "built_utc": e["utc"],
                            "definition_sha256": e.get("definition_sha256"), "commit": None}
        self.next_seg = max(self.next_seg, n + 1)
        if not self.chosen_recipe:
            self.stage_b = list(e["recipes"])

    def _ev_commit(self, e):
        seg = self.segments[e["segment"]]
        seg["state"] = "committed"
        seg["commit"] = {k: e.get(k) for k in ("chosen", "dispositions", "counted", "stale_base", "d30", "d33",
                                                "pool", "steps", "choice", "utc", "discordant", "research_only")}
        seg["commit"]["utc"] = e["utc"]
        if e.get("chosen") and not self.chosen_recipe and not e.get("stale_base"):
            self.chosen_recipe = e["chosen"]
        for inc, disp in (e.get("dispositions") or {}).items():
            rec = self.increments[inc]
            rec["disposition"] = disp
            rec["status"] = disp
            counted = (e.get("counted") or {}).get(inc)
            keys = list(self.inc_rows(inc))
            if disp == ACCEPTED:
                for k in keys:
                    self._set_key(e, k, "accepted", inc, ACCEPTED)
            elif disp == DATA:
                for k in keys:
                    self._set_key(e, k, "quarantined", inc, DATA)
                self._quarantine_inc(e, inc, keys, "data")
            elif disp == "stale":
                for k in keys:
                    self._set_key(e, k, "released", inc, "released_stale_base", counted=False)
            else:
                neutral = []
                for k in keys:
                    st = self.keys[k]
                    if counted and st.get("returns", 0) >= 1:
                        self._set_key(e, k, "neutral", inc, "neutral", counted=True)
                        neutral.append(k)
                    else:
                        self._set_key(e, k, "returned", inc, "returned:%s" % disp, counted=counted)
                        if counted:
                            st["returns"] = st.get("returns", 0) + 1
                if neutral:
                    self._quarantine_inc(e, inc, neutral, "neutral")
        pool = e.get("pool")
        if pool:
            self.pools[pool["name"]] = dict(pool)
            self.pool = pool["name"]

    def _ev_return(self, e):
        pass                                     # informational: the commit event carries the transition

    def _ev_quarantine(self, e):
        if e.get("scope") == "source":
            self.q_sources[e["source"]] = {"cite": e.get("cite"), "seq": e["seq"], "utc": e["utc"]}

    def _ev_unquarantine(self, e):
        self.q_sources.pop(e["source"], None)

    def _ev_withdraw(self, e):
        if e.get("orphan_increments"):
            # cut lines whose build line never followed (the build was killed between them): no segment
            # exists, so only the increments are withdrawn and their images released, uncounted
            for inc in e["orphan_increments"]:
                self.increments[inc]["status"] = "withdrawn"
                for k in self.inc_rows(inc):
                    if self.keys.get(k, {}).get("increment") == inc and self.keys[k]["status"] == "in_increment":
                        self._set_key(e, k, "released", inc, "released_withdrawn", counted=False)
            return
        seg = self.segments[e["segment"]]
        seg["state"] = "withdrawn"
        seg["withdraw_reason"] = e.get("reason")
        for inc in seg["increments"]:
            self.increments[inc]["status"] = "withdrawn"
            for k in self.inc_rows(inc):
                if self.keys.get(k, {}).get("increment") == inc and self.keys[k]["status"] == "in_increment":
                    self._set_key(e, k, "released", inc, "released_withdrawn", counted=False)

    def _ev_rollback(self, e):
        self.rollbacks.append({"n": len(self.rollbacks) + 1, "to": e["to"], "from": e["from"],
                               "suspect": list(e["suspect"]), "milestone": e.get("milestone"), "seq": e["seq"],
                               "utc": e["utc"], "decided_by": e.get("by")})
        for inc in e["suspect"]:
            self.increments[inc]["status"] = "suspect"
            for k in self.inc_rows(inc):
                if self.keys.get(k, {}).get("status") == "accepted":
                    self._set_key(e, k, "suspect", inc, "suspect")
        self.pool = e["to"]

    def _ev_milestone(self, e):
        n = e["n"]
        if e["phase"] == "build":
            self.milestones[n] = {"n": n, "exp": e["exp"], "pool": e["pool"], "state": "built", "verdict": None,
                                  "good": None, "built_utc": e["utc"]}
            self.next_ms = max(self.next_ms, n + 1)
        elif e["phase"] == "build_failed":
            self.milestones[n]["state"] = "failed"
        elif e["phase"] == "compare":
            m = self.milestones[n]
            m.update(state="compared", verdict=e["verdict"], good=e["verdict"] != "hurts",
                     compared_with=e.get("compared_with"), rollback_recommended=e.get("rollback_recommended"),
                     to_pool=e.get("to_pool"), compared_utc=e["utc"], perm_p=e.get("perm_p"),
                     species_failed=e.get("species_failed"))

    def _ev_bisect(self, e):
        rb = e["rollback"]
        b = self.bisects.setdefault(rb, {"arms": {}, "decisions": {}})
        if e["phase"] == "build":
            b["arms"].update(e["arms"])
            b["from"] = e["from"]
        else:
            for inc, v in e["decisions"].items():
                b["decisions"][inc] = v
                rec = self.increments[inc]
                rec["status"] = "bisect_%s" % v
                keys = list(self.inc_rows(inc))
                if v == G.HELPS:
                    for k in keys:
                        self._set_key(e, k, "returned_bisect", inc, "returned_bisect_helps", counted=False)
                elif v == G.HURTS:
                    for k in keys:
                        self._set_key(e, k, "quarantined", inc, DATA)
                    self._quarantine_inc(e, inc, keys, "data")
                else:
                    for k in keys:
                        self._set_key(e, k, "neutral", inc, "neutral")
                    self._quarantine_inc(e, inc, keys, "neutral")

    def _ev_feasibility(self, e):
        f = self.feasibility or {}
        f.update({k: v for k, v in e.items() if k not in ("seq", "prev_sha256", "event", "sid")})
        self.feasibility = f

    def _ev_fork(self, e):
        self.forked_to = e["to"]

    def _ev_release(self, e):
        self.releases.append({k: e.get(k) for k in ("hold", "source", "keys_file", "by", "reason", "seq", "scope",
                                                     "licence", "research_only")})

    # ---- derived views
    def current_pool(self):
        return self.pools[self.pool]

    def pool_lineage(self, name=None):
        out, n = [], name or self.pool
        while n is not None:
            out.append(n)
            n = self.pools[n].get("parent")
        return out

    def in_flight(self):
        return [s for s in self.segments.values() if s["state"] == "built"]

    def last_good_milestone(self, exclude=None):
        good = [m for n, m in self.milestones.items() if m.get("good") and n != exclude
                and m["pool"] in self.pool_lineage()]
        return good[-1] if good else None

    def milestone_in_flight(self):
        for m in self.milestones.values():
            if m["state"] == "built":
                return m
        return None

    def _releases(self):
        """The recorded releases of holds a person may release (an h6_scan
        release, whatever wrote it, is never honoured: the copy scan is
        mandatory)."""
        return [r for r in self.releases if r.get("hold") in RELEASABLE_HOLDS]

    def released_keys(self):
        out = {}
        for r in self._releases():
            if r.get("keys_file"):
                for x in _read_jsonl(_check_rec(r["keys_file"], "released keys")):
                    out.setdefault(x["key"] if isinstance(x, dict) else str(x), set()).add(r["hold"])
        return out

    def released_sources(self):
        out = collections.defaultdict(set)
        for r in self._releases():
            if r.get("source"):
                out[r["source"]].add(r["hold"])
        return out

    def released_all(self):
        return {r["hold"] for r in self._releases() if r.get("scope") == "all" and r["hold"] == "funnel_F9"}

    def licence_releases(self):
        """({key: release}, {source: release}): the latest release of the
        licence hold per key and per source, which the cutter applies to the
        rows it lifts (_row_licence)."""
        by_key, by_src = {}, {}
        for r in self._releases():
            if r["hold"] != "licence":
                continue
            if r.get("keys_file"):
                for x in _read_jsonl(_check_rec(r["keys_file"], "released keys")):
                    by_key[x["key"] if isinstance(x, dict) else str(x)] = r
            if r.get("source"):
                by_src[r["source"]] = r
        return by_key, by_src


# ----------------------------------------------------------------- stream
class Stream:
    """One stream version: INC_DIR/stream/<sid>/."""

    def __init__(self, sid, deps=None, clock=time.time, quiet=False):
        self.p = StreamPaths(sid)
        self.sid = self.p.sid
        self.deps = deps or Deps()
        self.clock = clock
        self.quiet = quiet
        self.ledger = Ledger(self.p.ledger)
        self.fold = None
        self._lease = None
        self._hcache = None

    def _log(self, msg):
        if not self.quiet:
            log("%s: %s" % (self.sid, msg))

    def _now(self):
        return self.clock()

    # ---- loading and writing
    def load(self, repair=False):
        entries = self.ledger.read(repair=repair)
        if not entries:
            raise StreamError("stream %s has no ledger at %s: run init first" % (self.sid, self.p.ledger))
        f = Fold(self.p)
        for e in entries:
            if e.get("sid") != self.sid:
                raise StreamError("ledger line %d names stream %r" % (e["seq"], e.get("sid")))
            f.apply(e)
        f.head = self.ledger.head
        self.fold = f
        return f

    @contextlib.contextmanager
    def writing(self, new=False):
        self.p.root.mkdir(parents=True, exist_ok=True)
        lease = D.Lease(self.p.lease)
        if not lease.acquire():
            held = (D.Lease.read(self.p.lease) or {}).get("body") or {}
            raise LeaseBusy("stream %s is being written by another operation (%s pid %s); retry later"
                            % (self.sid, held.get("host"), held.get("pid")))
        self._lease = lease
        try:
            if not new:
                self.load(repair=True)
                self._note_code()
                self._repair()
            yield self
            if self.fold is not None:
                self.sync()
        finally:
            lease.release()
            self._lease = None

    def event(self, event, by="platform", **payload):
        if event not in EVENTS:
            raise StreamError("internal: unknown event %r" % event)
        if self._lease is not None:
            self._lease.check()
        entry = dict(payload, event=event, sid=self.sid, utc=_utc(self._now()), by=_check_actor(by))
        e = self.ledger.append(entry)
        if self.fold is None:
            self.fold = Fold(self.p)
        self.fold.apply(e)
        self.fold.head = self.ledger.head
        return e

    def _stale_line(self, utc):
        """True when a ledger line of this time is older than CRASH_GRACE_S: the
        operation that wrote it held stream.lease, so a writer that now holds
        the lease is past it (dead, or its lease expired twice over)."""
        t = _ts(utc) if utc else None
        return t is not None and self._now() - t > CRASH_GRACE_S

    def _repair(self):
        """Complete what a killed operation left half done (the ledger line is
        written before the external step, so the next writer can see it):
          * cut lines without their build line: the increments are withdrawn
            and their images released, uncounted (otherwise they would stay
            'in_increment' for good, and supply would shrink without a trace);
          * a built segment whose driver init never wrote state.json (nothing
            was submitted): withdrawn, images released, uncounted;
          * a milestone whose inc2.baseline build never wrote state.json:
            recorded build_failed, so the next L20 builds again (it would
            otherwise wait on it for ever);
          * the same for Stage C (feasibility may then be built again).
        The last three only CRASH_GRACE_S after their line."""
        f = self.fold
        orphans = collections.OrderedDict()
        for inc, r in f.increments.items():
            if r["status"] == "in_segment" and r["segment"] not in f.segments:
                orphans.setdefault(r["segment"], []).append(inc)
        for n, incs in orphans.items():
            self.event("withdraw", segment=n, exp=segment_exp(self.sid, n), orphan_increments=incs,
                       reason="repair: the build of %s never reached the ledger after its cut lines (killed); "
                              "its images are released, uncounted" % segment_exp(self.sid, n))
            log("WARNING: %s: released the orphaned increments %s of %s" % (self.sid, incs, segment_exp(self.sid, n)))
        for s in list(f.in_flight()):
            if not D.Paths(s["exp"]).state.exists() and self._stale_line(s.get("built_utc")):
                self.event("withdraw", segment=s["n"], exp=s["exp"],
                           reason="repair: driver init of %s never wrote state.json (killed before any "
                                  "submission); its images are released, uncounted" % s["exp"])
                log("WARNING: %s: withdrew %s (driver init never completed)" % (self.sid, s["exp"]))
        m = f.milestone_in_flight()
        if m is not None and not D.Paths(m["exp"]).state.exists() and self._stale_line(m.get("built_utc")):
            self.event("milestone", phase="build_failed", n=m["n"], exp=m["exp"], rc=None,
                       reason="repair: inc2.baseline build of %s never wrote state.json" % m["exp"])
            log("WARNING: %s: milestone %s recorded build_failed (its build never completed)" % (self.sid, m["exp"]))
        fz = f.feasibility or {}
        if (fz.get("exp") and fz.get("phase") == "build" and not D.Paths(fz["exp"]).state.exists()
                and self._stale_line(fz.get("utc"))):
            self.event("feasibility", phase="build_failed", exp=fz["exp"],
                       reason="repair: driver init of %s never wrote state.json" % fz["exp"])

    def _note_code(self):
        now = code_hashes()
        if self.fold.code != now:
            old = self.fold.code or {}
            changed = sorted(m for m in set(old) | set(now) if old.get(m) != now.get(m))
            self.event("code_change", old=old, new=now, changed=changed)

    def sync(self):
        """Write every derived file from the fold: consumed.jsonl (appended),
        quarantine_dhash.json, state.json, queue_summary.json."""
        f = self.fold
        have = _read_jsonl(self.p.consumed)
        want = f.transitions
        if len(have) > len(want) or any(_canon(a) != _canon(b) for a, b in zip(have, want)):
            raise StreamError("%s does not match the ledger's transitions (%d rows, %d expected): it was edited; "
                              "refusing" % (self.p.consumed, len(have), len(want)))
        if len(want) > len(have):
            with open(self.p.consumed, "ab") as fh:
                fh.write("".join(_canon(r) + "\n" for r in want[len(have):]).encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
        q = {"format": QUARANTINE_FORMAT, "sid": self.sid, "bits": QUARANTINE_BITS, "ledger_seq": f.events,
             "entries": f.quarantine, "sources": f.q_sources}
        if _read_json(self.p.quarantine) != json.loads(json.dumps(q)):
            _write_json(self.p.quarantine, q)
        _write_json(self.p.state, self.state_record())
        self.write_summary()

    def state_record(self):
        f = self.fold
        counts = collections.Counter(v["status"] for v in f.keys.values())
        return {"format": STATE_FORMAT, "sid": self.sid, "ledger": {"events": f.events, "head_sha256": f.head},
                "M": f.defn["M"], "K_max": f.defn["K_max"], "testing": bool(f.defn.get("testing")),
                "pool": f.pool, "pools": f.pools, "segments": {str(k): v for k, v in f.segments.items()},
                "increments": {k: {x: v.get(x) for x in ("segment", "step", "status", "disposition", "n_images",
                                                         "sources", "species_boxes")}
                               for k, v in f.increments.items()},
                "key_status": dict(counts), "quarantined_images": len({x["key"] for x in f.quarantine}),
                "quarantined_sources": f.q_sources,
                "milestones": {str(k): v for k, v in f.milestones.items()}, "rollbacks": f.rollbacks,
                "bisects": {str(k): v for k, v in f.bisects.items()}, "feasibility": f.feasibility,
                "arm": f.arm, "stage_b": f.stage_b, "chosen_recipe": f.chosen_recipe, "forked_to": f.forked_to,
                "next": {"increment": inc_name(f.next_inc), "segment": segment_exp(self.sid, f.next_seg),
                         "milestone": milestone_exp(self.sid, f.next_ms)},
                "code": f.code}

    # ---------------------------------------------------------------- init
    def init(self, base=None, lock=None, m=None, k_max=K_MAX, arm=DEFAULT_ARM, milestone0=None, stage_b=None,
             testing=False, decided_by="platform", step1_dir=None, _fork=None):
        """Create the stream: stream.json, P_0 (base_v2's bytes, content
        addressed, with its dHashes), the init and prospective ledger lines."""
        if self.p.ledger.exists() and self.p.ledger.stat().st_size:
            raise StreamError("stream %s already exists at %s; a stream is defined once" % (self.sid, self.p.root))
        if testing and os.environ.get("INC_SCORER_TESTING") != "1":
            raise StreamError("a testing stream builds testing experiments, which need INC_SCORER_TESTING=1")
        RC = self.deps.recipes()
        if arm not in _grid_arms(RC):
            raise StreamError("arm %r is not one of L-4's grid %s (a measurement arm is never a stream's arm)"
                              % (arm, list(_grid_arms(RC))))
        stage_b = self._check_stage_b(stage_b or list(DEFAULT_STAGE_B))
        k_max = int(k_max)
        if not 1 <= k_max <= K_MAX:
            raise StreamError("--k-max must be 1..%d" % K_MAX)
        lock = Path(lock) if lock else Path(C.INC_DIR) / "splits" / "v2" / "LOCK.json"
        if _fork is None:
            base = Path(os.path.abspath(str(base))) if base else Path(C.INC_DIR) / "splits" / "v2" / "base_v2.jsonl"
            if not base.is_file():
                raise StreamError("the base manifest %s does not exist (splits v2 first, L23)" % base)
        with self.writing(new=True):
            base_rows = C.read_manifest(base) if _fork is None else _fork["rows"]
            base_sha = C.sha256_file(base) if _fork is None else _fork["sha256"]
            lock_rec = None
            if lock.is_file():
                lk = _read_json(lock)
                if lk.get("splits_version") != "v2":
                    raise StreamError("%s is not a v2 LOCK" % lock)
                want = (lk.get("manifests") or {}).get("base_v2")
                if _fork is None and want != base_sha:
                    raise StreamError("the base %s (%s) is not the base_v2 LOCK v2 records (%s)"
                                      % (base, base_sha[:12], str(want)[:12]))
                lock_rec = _file_rec(lock)
            elif not testing:
                raise StreamError("LOCK v2 %s is missing: a production stream starts from locked splits" % lock)
            if not base_rows:
                raise StreamError("the base manifest is empty")
            M = int(m) if m is not None else -(-len(base_rows) * M_FRAC_NUM // M_FRAC_DEN)
            if M < 1:
                raise StreamError("M must be >= 1")
            self.p.pool.mkdir(parents=True, exist_ok=True)
            if _fork is None:
                hashes, ro = self._base_hashes(base_rows, lock)
            else:
                hashes, ro = _fork["hashes"], _fork["research_only"]
            pool = self._write_pool(0, base_rows, hashes, parent=None,
                                    parts=["base_v2"] if _fork is None else ["%s:%s" % (_fork["from"]["sid"],
                                                                                      _fork["from"]["pool"])],
                                    research_only=ro, src_path=_fork["path"] if _fork else base)
            if pool["sha256"] != base_sha:
                raise StreamError("P_0 does not hash like its source (%s != %s)" % (pool["sha256"][:12], base_sha[:12]))
            arm_rec = {"id": arm, "cost_multiplier": 1.0, "cost_measured": arm == DEFAULT_ARM, "source": "init"}
            defn = {"format": FORMAT, "sid": self.sid, "stream_version": STREAM_VERSION, "M": M, "K_max": k_max,
                    "testing": testing if isinstance(testing, dict) else bool(testing),
                    "base": {"path": str(base) if _fork is None else _fork["path"], "sha256": base_sha,
                             "n_images": len(base_rows)},
                    "lock": lock_rec, "lock_path": str(lock), "gate": GATE_BLOCK, "chain_seeds": list(CHAIN_SEEDS),
                    "milestone_seeds": list(MILESTONE_SEEDS), "segment_exams": list(SEGMENT_EXAMS),
                    "milestone_exams": list(MILESTONE_EXAMS), "budget": BUDGET,
                    "step1_stream": str(Path(step1_dir) if step1_dir else step1_root()),
                    "rules": self._rules(), "created_utc": _utc(self._now()), "decided_by": _check_actor(decided_by),
                    "forked_from": (_fork or {}).get("from")}
            _write_json(self.p.stream_json, defn)
            payload = {"stream_json_sha256": C.sha256_file(self.p.stream_json), "M": M, "K_max": k_max,
                       "pool": pool, "arm": arm_rec, "stage_b": stage_b, "milestone0": milestone0,
                       "code": code_hashes()}
            if _fork is not None:
                payload["inherited"] = _fork["inherited"]
                payload["milestone0"] = _fork.get("milestone0") or milestone0
            self.event("init", by=decided_by, **payload)
            rec = self.prospective_record()
            self.event("prospective", by=decided_by, record=rec, record_sha256=_sha_obj(rec))
            self._log("stream defined: M = %d, K <= %d, P_0 = %d images (%s), arm %s, Stage B recipes %s"
                      % (M, k_max, len(base_rows), pool["sha256"][:12], arm, stage_b))
        return self.fold

    def _rules(self):
        return {"species_cap": SPECIES_CAP, "unit_bits": UNIT_BITS, "pool_bits": POOL_BITS,
                "quarantine_bits": QUARANTINE_BITS, "stale_days": STALE_DAYS, "perm_alpha": PERM_ALPHA,
                "boundary_sd_mult": BOUNDARY_SD_MULT,
                "milestone_triggers": MILESTONE_TRIGGERS, "disposition": DISPOSITION_RULE, "stage_b": STAGE_B_RULE,
                "harmless_guard_reasons": list(HARMLESS_GUARD_REASONS)}

    def prospective_record(self):
        """The pre-registered record (§6.8 last row): the sha256 of (M, K, the
        truth policy, the thresholds, the recipe rule, the gate block) goes to
        the ledger before the first L18."""
        f = self.fold
        return {"M": f.defn["M"], "K_max": f.defn["K_max"],
                "truth_policy": "on for every step (P3) unless a step with its truth arm exceeds 25 GPU-h (L-4): "
                                "then on every ceil(cost/25)-th segment; off only by a person (--no-truth "
                                "--decided-by human:...)",
                "thresholds": self._rules(), "recipe_rule": STAGE_B_RULE, "stage_b": f.stage_b,
                "gate_block": GATE_BLOCK, "gate_version": "Protocol v3 (L-3): species tolerance max(0.03, 1.96 SE_s), "
                                                          "inc2.gate3, read at commit",
                "disposition_rule": DISPOSITION_RULE}

    @staticmethod
    def _check_stage_b(recipes):
        recipes = [r.strip() for r in (recipes.split(",") if isinstance(recipes, str) else recipes) if r.strip()]
        if not recipes or recipes[0] != R0 or len(set(recipes)) != len(recipes) or len(recipes) > 2:
            raise StreamError("Stage B runs r0 and at most one Stage A survivor, r0 first (§5.1); got %s" % recipes)
        for r in recipes[1:]:
            if r not in ("x1a", "x1b"):
                raise StreamError("%r is not a Stage A candidate (x1a, x1b); freeze and LoRA are out of stream "
                                  "version 1 (L-6)" % r)
        return recipes

    def _base_hashes(self, rows, lock):
        """{key: [dHash]} of the base rows: splits v2's provenance file when it
        lists exactly these rows (its sha256 checked against LOCK v2), else
        computed; research_only rows counted from the same file."""
        prov_path = Path(lock).parent / "base_v2_provenance.jsonl"
        lk = _read_json(lock) if Path(lock).is_file() else {}
        want = lk.get("provenance_sha256") or ((lk.get("provenance") or {}) if isinstance(lk.get("provenance"), dict)
                                                else {}).get("sha256")
        prov = {}
        if prov_path.is_file() and (want is None or C.sha256_file(prov_path) == want):
            prov = {r["key"]: r for r in _read_jsonl(prov_path, tolerate_partial=False)}
        out, ro, known = {}, 0, bool(prov)
        hasher = None
        for r in rows:
            p = prov.get(r["key"])
            if p is not None and p.get("sha256") == r["sha256"] and p.get("dhash") is not None:
                out[r["key"]] = [int(p["dhash"])]
                ro += 1 if p.get("research_only") else 0
                continue
            known = False
            hasher = hasher or self.deps.get_hasher()
            h, _v = hasher(r["image"])
            if h is None:
                raise StreamError("base image %s cannot be hashed: the pool's 6-bit disjointness cannot be "
                                  "checked (fail closed)" % r["image"])
            out[r["key"]] = [int(h)]
        return out, {"rows": ro, "known": known}

    def _write_pool(self, s, rows, hashes, parent, parts, research_only, src_path=None):
        """pool/P_<s>.<sha16>.jsonl (+ dHash map, meta); returns its record."""
        tmp = self.p.pool / (".P_%d.%d.tmp.jsonl" % (s, os.getpid()))
        if src_path is not None:
            with open(src_path, "rb") as fh:
                data = fh.read()
            sha = _sha_bytes(data)
            tmp.write_bytes(data)
        else:
            sha = C.write_manifest(tmp, rows)
        path = self.p.pool / ("P_%d.%s.jsonl" % (s, sha[:16]))
        os.replace(tmp, path)
        if C.sha256_file(path) != sha:
            raise StreamError("pool %s does not hash as written" % path)
        dpath = self.p.pool / ("P_%d.%s.dhash.json" % (s, sha[:16]))
        _write_json(dpath, {"pool": pool_name(s), "pool_sha256": sha, "hashes": {k: hashes[k] for k in sorted(hashes)}})
        return {"name": pool_name(s), "segment": s, "path": str(path), "sha256": sha, "n_images": len(rows),
                "dhash": _file_rec(dpath), "parent": parent, "parts": parts,
                "research_only_rows": int(research_only.get("rows", 0)),
                "research_only_known": bool(research_only.get("known")),
                "research_only": (True if research_only.get("rows") else
                                  (False if research_only.get("known") else "unknown"))}

    # ---------------------------------------------------------- choose-arm
    def choose_arm(self, capacity=None, decided_by="platform"):
        """L-4: adopt the capacity grid's decision, made by inc2.baseline
        capacity-verdict on dev only (INC_DIR/capacity/capacity_v1.json): its
        chosen arm, milestone 0 = the chosen arm's R0 baseline, and the
        measured cost of the arm relative to inc2.recipes' estimate. Allowed
        before the first build only (another arm later is a new stream
        version)."""
        with self.writing():
            f = self.fold
            if f.segments:
                raise StreamError("the arm is fixed once a segment is built (a new arm is a new stream version)")
            RC = self.deps.recipes()
            path = Path(capacity) if capacity else Path(C.INC_DIR) / "capacity" / "capacity_v1.json"
            doc = _read_json(path)
            if not isinstance(doc, dict) or doc.get("format") != "inc2-capacity/1":
                raise StreamError("%s is not inc2.baseline's capacity decision (inc2-capacity/1)" % path)
            arm = doc.get("chosen_arm")
            if arm not in _grid_arms(RC):
                raise StreamError("the capacity decision chose %r, not an arm of L-4's grid %s (a measurement arm "
                                  "is never a stream's arm)" % (arm, list(_grid_arms(RC))))
            sc = doc.get("step_cost") or {}
            measured = sc.get("estimate") is False
            est = RC.step_cost(int(doc["n_images"]), int(doc["m"]), arm=arm, recipe=R0, truth=True)["gpu_h"][1]
            meas = (sc.get("gpu_h") or [None, None])[1]
            mult = (float(meas) / est) if (measured and meas and est) else 1.0
            arms = {v.get("arm"): {"exp": e, "mean": v.get("mean"), "sd": v.get("sd"), "seeds": v.get("seeds"),
                                   "qualifies": v.get("qualifies"), "diff_vs_n": v.get("diff_vs_n"),
                                   "pooled_sd": v.get("pooled_sd")}
                    for e, v in (doc.get("arms") or {}).items()}
            arm_rec = {"id": arm, "cost_multiplier": mult, "cost_measured": measured, "source": "capacity decision",
                       "truth_every_at_base": doc.get("truth_every"), "record": doc.get("chosen_arm_record")}
            self.event("arm", by=decided_by, arm=arm_rec, capacity=_file_rec(path), arms=arms,
                       qualifying=doc.get("qualifying"), milestone0=doc.get("chosen_exp"), rule=doc.get("rule"))
            self._log("capacity grid: chosen %s (%s); qualifying %s; cost x%.3f of the estimate"
                      % (arm, doc.get("chosen_exp"), doc.get("qualifying"), mult))
            return arm_rec

    def _base_dev(self, exp, metric="map50_95"):
        """Dev values of a baseline's base runs (every seed, which must be complete)."""
        paths = D.Paths(exp)
        defn = _read_json(paths.exp_json)
        if defn is None:
            raise StreamError("experiment %s does not exist" % exp)
        out = []
        for s in defn["seeds"]:
            sc = _read_json(paths.score("base__s%d" % s, "dev"))
            if sc is None:
                raise StreamError("%s: base__s%d has no dev score yet" % (exp, s))
            out.append(float(sc[metric]))
        return out

    def _base_scores(self, exp, inputs=None):
        """The dev score dicts of a baseline's base runs (every seed); with
        inputs (a list), each file's {run_id, path, sha256} is appended."""
        paths = D.Paths(exp)
        defn = _read_json(paths.exp_json)
        if defn is None:
            raise StreamError("experiment %s does not exist" % exp)
        out = []
        for s in defn["seeds"]:
            p = paths.score("base__s%d" % s, "dev")
            try:
                data = p.read_bytes()
            except FileNotFoundError:
                raise StreamError("%s: base__s%d has no dev score yet" % (exp, s))
            out.append(json.loads(data.decode("utf-8")))
            if inputs is not None:
                inputs.append({"run_id": "base__s%d" % s, "path": str(p), "sha256": _sha_bytes(data)})
        return out

    # ------------------------------------------------------------ the cut
    def cut_plan(self, k, seg_n=None, compute_hashes=True, guard=True):
        """Up to k increments of exactly M, as they would be cut now (nothing
        is written). Returns (plans, refusal or None, analysis)."""
        f = self.fold
        M = f.defn["M"]
        seg_n = seg_n or f.next_seg
        view = self.queue()
        ana = self.eligibility(view, compute_hashes=compute_hashes)
        plans, refusal = [], None
        taken_units = set()
        remainders = self._remainders()
        excluded_units = set()
        guard_log = collections.Counter()
        for step in range(1, k + 1):
            try:
                plan = self._cut_one(ana, M, seg_n, step, taken_units, excluded_units, remainders, guard, guard_log)
            except CutRefused as e:
                refusal = dict(e.reason, step=step, segment=seg_n)
                break
            plans.append(plan)
            units = ana["units"]
            pids = {units[u]["parent"] for u in plan["units"]}
            # the rest of a split near-duplicate group waits for a later cut: in another increment of this one it
            # could be accepted beside its near-copy (commit's 6-bit disjointness)
            taken_units |= {u for u in range(len(units)) if units[u]["parent"] in pids}
            for r in plan["rows"]:
                for h in r["hashes"].values():
                    if h is not None:
                        ana["_near"].add(int(h), ("cut", r["key"]), max_bits=POOL_BITS)
            remainders = set(plan["split_capture_groups_keys"])
        ana["guard_excluded"] = dict(guard_log)
        return plans, refusal, ana

    def _remainders(self):
        f = self.fold
        if not f.increments:
            return set()
        last = list(f.increments.values())[-1]
        return {tuple(x) for x in last.get("split_capture_groups") or []}

    def queue(self):
        root = Path(self.fold.defn.get("step1_stream") or step1_root())
        return QueueView(root).load()

    def _hash_cache(self):
        """dhash_cache.jsonl, (image, sha256) -> hashes. It is a cache: a line
        that does not parse (a torn append) is skipped and recomputed, never a
        reason to refuse the stream."""
        if self._hcache is None:
            self._hcache = {}
            try:
                data = self.p.dhash_cache.read_bytes()
            except FileNotFoundError:
                data = b""
            bad = 0
            for ln in data.split(b"\n"):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                    self._hcache[(r["image"], r["sha256"])] = r
                except (ValueError, KeyError, TypeError):
                    bad += 1
            if bad:
                log("WARNING: %s: %d unreadable line(s) skipped (recomputed when needed)" % (self.p.dhash_cache, bad))
        return self._hcache

    def _cache_add(self, recs):
        if not recs:
            return
        cache = self._hash_cache()
        for r in recs:
            cache[(r["image"], r["sha256"])] = r
        if self._lease is None:
            return                          # only the lease holder writes the stream directory (a dry run does not)
        with open(self.p.dhash_cache, "ab") as fh:
            fh.write("".join(_canon(r) + "\n" for r in recs).encode("utf-8"))

    def row_hashes(self, row, compute=True, full=False):
        """{"train": dHash of the training image, "unmasked": of the original
        photograph (masked rows only)} and, with full, {role: 8 variants}.
        step1_stream's queue row carries dhash (the unmasked photograph) and,
        for a masked row, dhash_masked (the masked PNG it trains on); those
        are used as they are, and a missing one is computed from the file
        (cached by path and sha256). full recomputes both from the files."""
        masked = is_masked(row)
        out = {"train": _as_hash(row.get("dhash_masked") if masked else row.get("dhash")),
               "unmasked": _as_hash(row.get("dhash")) if masked else None}
        variants = {}
        if not compute:
            return out, variants
        cache = self._hash_cache()
        new = []
        for role, img, sha in (("train", row.get("image"), row.get("sha256")),
                               ("unmasked", row.get("unmasked_image"), row.get("unmasked_sha256"))):
            if not img or (role == "unmasked" and not masked):
                continue
            if not full and out[role] is not None:
                continue
            ck = (str(img), str(sha))
            c = cache.get(ck)
            if c is None or (full and c.get("variants") is None):
                h, v = self.deps.get_hasher()(img)
                c = {"image": str(img), "sha256": str(sha), "dhash": None if h is None else int(h),
                     "variants": ({kk: int(vv) for kk, vv in v.items()} if isinstance(v, dict) else None)}
                new.append(c)
            if c["dhash"] is not None:
                out[role] = c["dhash"]
            variants[role] = c.get("variants")
        self._cache_add(new)
        return out, variants

    def eligibility(self, view, compute_hashes=True):
        """The eligible rows and why every other row is not (module docstring)."""
        f = self.fold
        pool_name_ = f.pool
        pool_rows = f.pool_rows(pool_name_)
        pool_keys = {r["key"] for r in pool_rows}
        pool_images = {r["image"] for r in pool_rows}
        pool_shas = {r["sha256"] for r in pool_rows}
        near = NearHashIndex()
        for k, hs in f.pool_hashes(pool_name_).items():
            for h in hs:
                near.add(int(h), ("pool", k), max_bits=POOL_BITS)
        for inc, rec in f.increments.items():
            # accepted increments are in the pool's hashes; in-flight and suspect ones are not
            if rec["status"] in ("in_segment", "suspect"):
                for k, r in f.inc_rows(inc).items():
                    for h in (r.get("hashes") or {}).values():
                        if h is not None:
                            near.add(int(h), (rec["status"], k), max_bits=POOL_BITS)
        quarantine = NearHashIndex()
        for q in f.quarantine:
            quarantine.add(int(q["dhash"]), ("quarantine", q["key"]), max_bits=QUARANTINE_BITS)
        rel_keys = f.released_keys()
        rel_src = f.released_sources()
        rel_all = f.released_all()
        lic_keys, lic_src = f.licence_releases()
        now = self._now()
        reasons = collections.Counter()
        held = collections.Counter()
        past_deadline = collections.Counter()
        eligible = []
        other = 0
        for key, row in view.rows.items():
            if row.get("refused"):
                reasons["refused_in_queue"] += 1
                continue
            if row.get("test_v1"):
                # a main-test list holds it (step1_stream.test_v1_rows): never trained
                reasons["test_v1"] += 1
                continue
            vec = _species_vec(row)
            if vec is None:
                reasons["bad_species_boxes"] += 1
                continue
            if sum(vec) < 1 or row.get("kind") == "other":
                other += 1
                reasons["kind_other"] += 1
                continue
            if row.get("evidenced") is not True:
                reasons["not_evidenced"] += 1
                continue
            left = [h for h in holds_of(row) if h not in rel_all
                    and h not in rel_keys.get(key, set()) and h not in rel_src.get(row.get("source"), set())]
            if left:
                for h in left:
                    held[h] += 1
                dl = row.get("hold_deadline")
                for h in left:
                    d = dl.get(h) if isinstance(dl, dict) else dl
                    t = _deadline_ts(d) if d else None
                    if t is not None and t < now:
                        past_deadline[h] += 1
                reasons["held"] += 1
                continue
            st = f.keys.get(key, {}).get("status")
            if st not in ELIGIBLE_STATUSES:
                reasons["consumed:%s" % st] += 1
                continue
            if row.get("source") in f.q_sources:
                reasons["quarantined_source"] += 1
                continue
            man = manifest_fields(row)
            if man is None:
                reasons["no_manifest_fields"] += 1
                continue
            if key in pool_keys or man["image"] in pool_images or man["sha256"] in pool_shas:
                reasons["in_pool"] += 1
                continue
            hashes, _v = self.row_hashes(row, compute=compute_hashes)
            known = [h for h in hashes.values() if h is not None]
            if compute_hashes and hashes.get("train") is None:
                reasons["unhashable"] += 1
                continue
            if any(quarantine.find(h) is not None for h in known):
                reasons["quarantined_dhash"] += 1
                continue
            if any(near.find(h) is not None for h in known):
                reasons["near_pool_or_in_flight"] += 1
                continue
            # a licence hold a person lifted: the release's licence and research_only apply (_row_licence)
            lic_rel = (lic_keys.get(key) or lic_src.get(row.get("source"))) if "licence" in holds_of(row) else None
            eligible.append({"key": key, "row": row, "man": man, "species": vec, "tb": sum(vec),
                             "hashes": hashes, "hash_known": bool(known), "licence_release": lic_rel,
                             "source": str(row.get("source") or ""), "batch": str(row.get("batch") or ""),
                             "capture_group": str(row.get("capture_group") or ("batch:%s" % row.get("batch"))),
                             "group": row.get("group"),
                             "verifier": str(row.get("verifier") or "unknown"),
                             "l1": int(row["l1"]) if isinstance(row.get("l1"), int) else -1,
                             "l2": int(row["l2"]) if isinstance(row.get("l2"), int) else 0,
                             "admitted_utc": view.admitted_utc(row)})
        M = f.defn["M"]
        units = self._units(eligible, M)
        # what can never be cut as it stands: a near-duplicate group larger than M, or a unit whose rows carry
        # two verifier versions (it waits for a re-judge); not counted as supply (D20 / D22 read Q)
        stuck = [u for u in units if u["size"] > M or u["verifier"] is None]
        stuck_rows = {i for u in stuck for i in u["members"]}
        cuttable = [e for i, e in enumerate(eligible) if i not in stuck_rows]
        oldest = [e["admitted_utc"] for e in cuttable if e.get("admitted_utc")]
        return {"view": view, "eligible": eligible, "units": units, "reasons": dict(reasons), "held": dict(held),
                "held_past_deadline": dict(past_deadline), "other_kind": other,
                "oldest_eligible_utc": min(oldest) if oldest else None,
                "cuttable": cuttable,
                "uncuttable": {"images": len(stuck_rows),
                               "units_larger_than_M": sum(1 for u in stuck if u["size"] > M),
                               "mixed_verifier_units": sum(1 for u in stuck if u["verifier"] is None)},
                "hashes_checked": compute_hashes,
                "rows_without_hash": sum(1 for e in eligible if not e["hash_known"]),
                "_near": near, "_quarantine": quarantine}

    @staticmethod
    def _units(eligible, M=None):
        """The draw units of the eligible rows. A parent is a transitive
        UNIT_BITS group (over every dHash a row carries, the training image's
        and the unmasked original's), joined with step1_stream's near-duplicate
        group of its rows; two rows within UNIT_BITS are always in one parent.
        A parent of at most M images is one unit. A larger one would never fit
        an increment (a video or a greenhouse tray chains its frames within 6
        bits), so it is split into its SUB_UNIT_BITS near-duplicate groups
        (§3.4 [review]: the draw unit is the near-duplicate group): each is a
        unit that carries its 'parent', and the cut never places two units of
        one parent in different increments of one cut, which keeps the pool's
        6-bit pairwise disjointness (commit) whatever is accepted."""
        n = len(eligible)

        def linked(bits):
            parent = list(range(n))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(i, j):
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)

            idx = NearHashIndex()
            owners = collections.defaultdict(list)
            by_group = {}
            for i, e in enumerate(eligible):
                for h in sorted(set(h for h in e["hashes"].values() if h is not None)):
                    for (who, _bits) in idx.matches(h):
                        for j in owners[who]:
                            union(i, j)
                    owners[h].append(i)
                    idx.add(h, h, max_bits=bits)
                g = e.get("group")
                if g is not None:
                    key = json.dumps(g, sort_keys=True, default=str)
                    if key in by_group:
                        union(i, by_group[key])
                    else:
                        by_group[key] = i
            return find

        def unit(members, pid, parent_size):
            members = sorted(members, key=lambda i: (eligible[i]["batch"], eligible[i]["key"]))
            first = eligible[members[0]]
            vers = {eligible[i]["verifier"] for i in members}
            feat = [eligible[i] for i in members if eligible[i]["l1"] >= 0]
            return {"members": members, "size": len(members), "source": first["source"],
                    "capture_group": first["capture_group"], "batch": min(eligible[i]["batch"] for i in members),
                    "verifier": first["verifier"] if len(vers) == 1 else None,
                    "l1": feat[0]["l1"] if feat else -1, "l2": max(0, feat[0]["l2"]) if feat else 0,
                    "species": [sum(eligible[i]["species"][s] for i in members) for s in range(C.OTHER_PLANT)],
                    "sources": sorted({eligible[i]["source"] for i in members}),
                    "keys": [eligible[i]["key"] for i in members], "parent": pid,
                    "parent_size": parent_size, "split": parent_size is not None}

        find6 = linked(UNIT_BITS)
        groups = collections.OrderedDict()
        for i in range(n):
            groups.setdefault(find6(i), []).append(i)
        find3 = None
        units = []
        for pid, members in enumerate(groups.values()):
            if M is None or len(members) <= M:
                units.append(unit(members, pid, None))
                continue
            find3 = find3 or linked(SUB_UNIT_BITS)
            subs = collections.OrderedDict()
            for i in sorted(members, key=lambda i: (eligible[i]["batch"], eligible[i]["key"])):
                subs.setdefault(find3(i), []).append(i)
            for sm in subs.values():
                units.append(unit(sm, pid, len(members)))
        return units

    def _cut_one(self, ana, M, seg_n, step, taken, excluded, remainders, guard, guard_log):
        """One increment of exactly M (module docstring)."""
        import numpy as np
        from ..inc import select as S
        units = ana["units"]
        avail = [u for u in range(len(units)) if u not in taken and u not in excluded]
        q = sum(units[u]["size"] for u in avail)
        if q < M:
            raise CutRefused("the queue holds %d eligible target images against M = %d" % (q, M),
                             {"why": "short", "eligible_images": q, "M": M})
        big = [u for u in avail if units[u]["size"] > M]
        mixed = [u for u in avail if units[u]["verifier"] is None]
        # one verifier version per increment: the newest version with >= M images, else the largest
        order_v = ana["view"].version_order or ["unknown"]
        per_v = collections.Counter()
        for u in avail:
            if units[u]["verifier"] is not None and units[u]["size"] <= M:
                per_v[units[u]["verifier"]] += units[u]["size"]
        full = [v for v in order_v if per_v.get(v, 0) >= M]
        if full:
            version = full[-1]
        elif per_v:
            version = max(per_v, key=lambda v: (per_v[v], order_v.index(v) if v in order_v else -1))
        else:
            version = None
        cand = [u for u in avail if units[u]["verifier"] == version and units[u]["size"] <= M]
        seed_text = "%s/%03d/%d" % (self.sid, seg_n, step)
        seed = C.stable_int(seed_text)
        sources_eligible = sorted({units[u]["source"] for u in cand})
        attempt = 0
        while True:
            attempt += 1
            order, first_source = self._order(cand, units, M, remainders, seed)
            sizes = np.array([units[u]["size"] for u in range(len(units))], dtype=np.int64)
            try:
                (part,), _left = S.draw_parts([np.asarray(order, dtype=np.int64)], sizes, 1, M)
            except S.SelectError as e:
                raise CutRefused("no exact fill of M = %d from %d eligible images in %d units: %s"
                                 % (M, sum(units[u]["size"] for u in cand), len(cand), e),
                                 {"why": "no_exact_fill", "eligible_images": q, "M": M, "units": len(cand),
                                  "units_larger_than_M": len(big), "mixed_verifier_units": len(mixed),
                                  "images_in_units_larger_than_M": sum(units[u]["size"] for u in big),
                                  "verifier": version, "select_error": str(e)})
            picked = [int(u) for u in part]
            pos = {u: i for i, u in enumerate(order)}
            picked.sort(key=lambda u: pos[u])
            cap = {"applies": len(sources_eligible) >= 2, "cap": SPECIES_CAP, "swaps": 0}
            if cap["applies"]:
                picked, cap = self._species_cap(picked, order, units, cap)
            share, top = _max_share(picked, units)
            cap.update(max_share=share, max_species=top, met=(share is None or share <= SPECIES_CAP + 1e-12))
            if not guard:
                break
            bad = self._guard_units(picked, units, ana, guard_log)
            if not bad:
                break
            for u in bad:
                excluded.add(u)
            cand = [u for u in cand if u not in bad]
            if sum(units[u]["size"] for u in cand) < M:
                raise CutRefused("after the guard excluded %d unit(s), %d eligible images remain against M = %d"
                                 % (len(bad), sum(units[u]["size"] for u in cand), M),
                                 {"why": "short_after_guard", "eligible_images": q, "M": M,
                                  "guard_excluded": dict(guard_log)})
        ecount = sum(units[u]["size"] for u in picked)
        if ecount != M:
            raise StreamError("internal: the draw holds %d images, not %d" % (ecount, M))
        cgs_picked = collections.Counter((units[u]["source"], units[u]["capture_group"]) for u in picked)
        cg_total = collections.Counter((units[u]["source"], units[u]["capture_group"]) for u in cand)
        split = sorted(cg for cg in cgs_picked if cg_total[cg] > cgs_picked[cg])
        split_groups = collections.Counter(units[u]["parent"] for u in picked if units[u]["split"])
        split_rec = [{"parent_size": next(units[u]["parent_size"] for u in picked if units[u]["parent"] == p),
                      "taken": sum(units[u]["size"] for u in picked if units[u]["parent"] == p),
                      "units": c} for p, c in sorted(split_groups.items())]
        return {"units": picked, "seed_text": seed_text, "seed": seed, "first_source": first_source,
                "verifier": version, "species_cap": cap, "split_capture_groups_keys": split,
                "split_near_dup_groups": split_rec,
                "capture_groups": len(cgs_picked), "attempts": attempt,
                "rows": [ana["eligible"][i] for u in picked for i in units[u]["members"]]}

    @staticmethod
    def _order(cand, units, M, remainders, seed):
        """The draw order of the candidate units (module docstring)."""
        import numpy as np
        from ..inc import select as S
        by_src = collections.defaultdict(list)
        for u in cand:
            by_src[units[u]["source"]].append(u)
        size_src = {s: sum(units[u]["size"] for u in us) for s, us in by_src.items()}
        oldest = {s: min(units[u]["batch"] for u in us) for s, us in by_src.items()}
        full = sorted((s for s in by_src if size_src[s] >= M), key=lambda s: (oldest[s], s))
        first = full[0] if full else None
        srcs = ([first] if first else []) + sorted((s for s in by_src if s != first), key=lambda s: (oldest[s], s))
        rng = np.random.default_rng(seed)
        n = len(units)
        ul1 = np.array([units[u]["l1"] for u in range(n)], dtype=np.int64)
        ul2 = np.array([units[u]["l2"] for u in range(n)], dtype=np.int64)
        order = []
        for s in srcs:
            by_cg = collections.defaultdict(list)
            for u in by_src[s]:
                by_cg[units[u]["capture_group"]].append(u)
            cg_old = {cg: min(units[u]["batch"] for u in us) for cg, us in by_cg.items()}
            for cg in sorted(by_cg, key=lambda cg: (0 if (s, cg) in remainders else 1, cg_old[cg], cg)):
                idx = np.array(sorted(by_cg[cg]), dtype=np.int64)
                order.extend(int(x) for x in S.balanced_order(idx, ul1, ul2, rng.permutation, rng))
        return order, first

    @staticmethod
    def _species_cap(picked, order, units, cap):
        """Swap the last-picked unit of the over-cap species for later units of
        the same total size whose share of it is within the cap."""
        removed = set()
        cur = list(picked)
        while True:
            share, top = _max_share(cur, units)
            if share is None or share <= SPECIES_CAP + 1e-12:
                break
            x = species_names().index(top)
            in_cur = set(cur)
            swapped = False
            for u in reversed([v for v in cur if units[v]["species"][x] > 0]):
                room = units[u]["size"]
                add, got = [], 0
                for v in order:
                    if v in in_cur or v in removed or got == room:
                        continue
                    tb = sum(units[v]["species"])
                    if tb and units[v]["species"][x] / float(tb) > SPECIES_CAP:
                        continue
                    if got + units[v]["size"] <= room:
                        add.append(v)
                        got += units[v]["size"]
                if got == room:
                    new = [v for v in cur if v != u] + add
                    s2, _t = _max_share(new, units)
                    if s2 is not None and s2 < share - 1e-12:
                        cur = new
                        removed.add(u)
                        cap["swaps"] += 1
                        swapped = True
                        break
                removed.add(u)
            if not swapped:
                break
        return cur, cap

    def guard(self):
        """GuardV2 over the LOCK v2 this stream was defined on: a production
        stream refuses when that LOCK changed since init (another splits
        version means another stream version, §3.3)."""
        f = self.fold
        rec = f.defn.get("lock")
        if rec:
            _check_rec(rec, "LOCK v2 (as the stream's init recorded it)")
        elif not f.defn.get("testing"):
            raise StreamError("the stream records no LOCK v2: it cannot clear a row against the never-train index")
        return self.deps.get_guard(f.defn.get("lock_path"), testing=bool(f.defn.get("testing")))

    def _guard_units(self, picked, units, ana, guard_log):
        """Units with a row that fails the image sha256 check, GuardV2 (on the
        training image and the unmasked original, 8 variants each), or whose
        flips and rotations come within QUARANTINE_BITS of a quarantined image
        or within POOL_BITS of the pool, an increment in flight, a suspect one
        or an earlier increment of this cut (a mirrored or rotated re-upload
        of such an image is the same photograph)."""
        g = self.guard()
        near, quar = ana.get("_near"), ana.get("_quarantine")
        bad = []
        for u in picked:
            refuse = None
            for i in units[u]["members"]:
                e = ana["eligible"][i]
                man = e["man"]
                try:
                    got = C.sha256_file(man["image"])
                except OSError:
                    got = None
                if got != man["sha256"]:
                    refuse = "image_sha256"
                    break
                hashes, variants = self.row_hashes(e["row"], compute=True, full=True)
                for role in ("train", "unmasked"):
                    if role == "unmasked" and not is_masked(e["row"]):
                        continue
                    reason, _m = g.check(hashes.get(role), variants.get(role))
                    if reason is not None and reason not in HARMLESS_GUARD_REASONS:
                        refuse = "%s:%s" % (role, reason)
                        break
                    vs = [int(v) for v in (variants.get(role) or {}).values() if v is not None]
                    if quar is not None and any(quar.find(v) is not None for v in vs):
                        refuse = "%s:%s" % (role, VARIANT_QUARANTINE)
                        break
                    if near is not None and any(near.find(v) is not None for v in vs):
                        refuse = "%s:%s" % (role, VARIANT_POOL)
                        break
                if refuse:
                    break
                e["hashes"] = hashes
            if refuse:
                guard_log[refuse] += units[u]["size"]
                bad.append(u)
        return bad

    # ---------------------------------------------------------------- build
    def recipes_for(self, seg_n, recipes=None):
        f = self.fold
        committed = [s for s in f.segments.values() if s["state"] == "committed" and not (s["commit"] or {}).get("stale_base")]
        if f.chosen_recipe:
            if recipes and list(recipes) != [f.chosen_recipe]:
                raise StreamError("Stage B chose %s at the first committed segment; a segment runs it alone (no arm "
                                  "is added after results are seen without a new pre-registration)" % f.chosen_recipe)
            return [f.chosen_recipe]
        if committed:
            raise StreamError("internal: a committed segment without a chosen recipe")
        return self._check_stage_b(recipes or f.stage_b)

    def truth_policy(self, seg_ordinal, recipes, n_pool, M, no_truth=False, decided_by="platform",
                     requested_every=None):
        """L-4: a step with its truth arm (one chain per recipe + the cold
        union runs) above 25 GPU-h runs the truth arm on every
        ceil(cost/25)-th step. The pinned driver's truth arm is per
        experiment, so the rule applies per segment: the truth arm is on for
        segment ordinals 0, every, 2 every, ... (the first segment always).
        The cost is inc2.recipes.step_cost's high bound for the arm at the
        current pool size, times the measured / estimated ratio the capacity
        grid recorded (choose-arm) when there is one. A cadence the autopilot
        computed (L18's --truth-every) can only make the truth arm more
        frequent: the smaller of the two is used and both are recorded (the
        platform never drops the truth arm on its own cost model, P3)."""
        f = self.fold
        RC = self.deps.recipes()
        arm = (f.arm or {}).get("id", DEFAULT_ARM)
        measured = bool((f.arm or {}).get("cost_measured"))
        mult = float((f.arm or {}).get("cost_multiplier") or 1.0) if measured else 1.0

        def hi(recipe, truth):
            return RC.step_cost(n_pool, M, arm=arm, recipe=recipe, truth=truth)["gpu_h"][1] * mult

        truth_part = hi(R0, True) - hi(R0, False)
        cost = truth_part + sum(hi(r, False) for r in recipes)
        every_stream = RC.truth_every(cost)
        every = every_stream
        if requested_every is not None:
            if int(requested_every) < 1:
                raise StreamError("--truth-every must be >= 1")
            every = min(every_stream, int(requested_every))
        on = seg_ordinal % every == 0
        rec = {"step_cost_gpu_h": round(cost, 3), "every": every, "every_stream": every_stream,
               "every_requested": None if requested_every is None else int(requested_every),
               "segment_ordinal": seg_ordinal, "on": on,
               "granularity": "segment (the pinned driver's truth arm is per experiment)", "arm": arm,
               "cost_multiplier": mult, "cost_basis": "inc2.recipes.step_cost (high bound) for the arm x the measured "
                                                      "multiplier of the capacity grid" if measured else
               "inc2.recipes.step_cost est. (high bound) for the arm", "estimate": not measured}
        if no_truth:
            if not _human(decided_by):
                raise StreamError("the truth arm is switched off only by a person (P3): --decided-by human:<id>")
            rec.update(on=False, switched_off_by=decided_by)
        return rec

    def check_arm_flags(self, arch=None, imgsz=None):
        """L18's --arch / --imgsz (the autopilot's view of the capacity arm)
        must name the arm this stream runs: the arm is fixed by choose-arm
        (L-4), never switched by a build flag (fail closed)."""
        if arch is None and imgsz is None:
            return
        f = self.fold
        RC = self.deps.recipes()
        arm_id = (f.arm or {}).get("id", DEFAULT_ARM)
        a = RC.ARMS[arm_id]
        model = str(a["model"])
        names = {model, model[:-3] if model.endswith(".pt") else model}
        if (arch is not None and str(arch) not in names) or (imgsz is not None and int(imgsz) != int(a["imgsz"])):
            raise StreamError("--arch %s --imgsz %s is not this stream's arm %s (%s at %d px): the arm is fixed by "
                              "`choose-arm` (L-4 capacity decision), never by a build flag"
                              % (arch, imgsz, arm_id, model, int(a["imgsz"])))

    def build(self, k, recipes=None, exp=None, no_truth=False, decided_by="platform", arch=None, imgsz=None,
              truth_every=None):
        """L18: cut up to k increments and build the segment, then driver init."""
        with self.writing():
            f = self.fold
            if f.forked_to:
                raise StreamError("stream %s was forked to %s; it builds nothing more" % (self.sid, f.forked_to))
            if f.in_flight():
                raise StreamError("segment %s is in flight: one segment at a time (L18 <= 1 in flight; a second one "
                                  "on the same base would only be returned stale at commit): commit or withdraw it "
                                  "first" % [s["exp"] for s in f.in_flight()])
            k = int(k)
            if not 1 <= k <= f.defn["K_max"]:
                raise StreamError("--k must be 1..%d" % f.defn["K_max"])
            seg_n = f.next_seg
            name = segment_exp(self.sid, seg_n)
            if exp is not None and exp != name:
                raise StreamError("the next segment of %s is %s, not %s" % (self.sid, name, exp))
            paths = D.Paths(name)
            if paths.exp_json.exists() or paths.state.exists():
                raise StreamError("%s already exists but the stream ledger has no build of it; withdraw or inspect "
                                  "it by hand" % paths.root)
            self._check_job_script()
            self.check_arm_flags(arch, imgsz)
            recs = self.recipes_for(seg_n, recipes)
            RC = self.deps.recipes()
            arm_id = (f.arm or {}).get("id", DEFAULT_ARM)
            table = RC.table(arm_id)
            pool = f.current_pool()
            ordinal = sum(1 for s in f.segments.values() if s["state"] != "withdrawn")
            truth = self.truth_policy(ordinal, recs, pool["n_images"], f.defn["M"], no_truth, decided_by,
                                      requested_every=truth_every)
            plans, refusal, ana = self.cut_plan(k, seg_n)
            self._write_refusal(refusal, ana, k)
            if not plans:
                raise StreamError("nothing cut for %s: %s" % (name, (refusal or {}).get("why")))
            written = [self._write_increment(plan, f.next_inc + i, seg_n, i + 1, ana) for i, plan in enumerate(plans)]
            testing = f.defn.get("testing") or False
            arm_rec = RC.resolve_arm(arm_id, repo=self.deps.repo or C.REPO, require_weights=not testing)
            cold = table["cold"]
            defn = {"exp": name, "type": "chain", "builder": BUILDER, "testing": testing, "replay_mode": "full",
                    "gate": dict(GATE_BLOCK), "seeds": list(CHAIN_SEEDS), "decision_exam": D.DECISION_EXAM,
                    "final_exams": list(SEGMENT_EXAMS),
                    "base": {"name": pool["name"], "manifest": pool["path"], "manifest_sha256": pool["sha256"],
                             "n_images": pool["n_images"], "recipe": cold},
                    "steps": [{"name": w["id"], "manifest": w["manifest"]["path"],
                               "manifest_sha256": w["manifest"]["sha256"], "n_images": w["manifest"]["n_images"],
                               "clean": False, "kind": "stream", "sources": w["sources"],
                               "meta_sha256": w["meta"]["sha256"]} for w in written],
                    "recipes": {r: table[r] for r in recs}, "truth": bool(truth["on"]), "truth_recipe": cold,
                    "increment_images": f.defn["M"],
                    "stream": {"sid": self.sid, "segment": seg_n, "stream_json_sha256": C.sha256_file(self.p.stream_json),
                               "pool": pool["name"], "truth_policy": truth, "k_requested": k, "k_built": len(written),
                               "stage_b": f.chosen_recipe is None, "protocol_gate": "v3 (L-3), read at commit",
                               "research_only": _research_only(pool, written)},
                    "attribution_scope": {"read_at_commit": DISPOSITION_RULE}}
            defn.update(RC.stamp(arm_rec))
            D.validate_definition(defn)
            D.check_definition_data(defn)
            for w in written:
                self.event("cut", by=decided_by, **w["event"])
            self.event("build", by=decided_by, segment=seg_n, exp=name, base_pool=pool["name"],
                       increments=[w["id"] for w in written], recipes=recs, truth=bool(truth["on"]),
                       truth_policy=truth, k_requested=k, exp_json=None, arm=arm_rec,
                       definition_sha256=_sha_obj(D._comparable(defn)), refusal=refusal)
            try:
                D.Driver(name, backend=self.deps.backend, clock=self.clock, quiet=self.quiet).init(defn)
            except Exception as e:  # noqa: BLE001 - any failure leaves the segment withdrawn, recorded
                self.event("withdraw", by="platform", segment=seg_n, exp=name,
                           reason="driver init failed: %s: %s" % (type(e).__name__, e))
                raise StreamError("driver init of %s failed (the segment is withdrawn, its images released): %s"
                                  % (name, e))
            log("built experiment %s" % name)
            self._log("%s: %d increment(s) of %d on %s (%d images), recipes %s, truth %s"
                      % (name, len(written), f.defn["M"], pool["name"], pool["n_images"], recs,
                         "on" if truth["on"] else "off"))
            return name

    def _check_job_script(self):
        """The segment's runs must execute inc2.train: INC_JOB_SCRIPT names
        run_inc2_job.sh (set here when unset); the v1 script is refused."""
        if self.deps.backend is not None:
            return
        js = os.environ.get("INC_JOB_SCRIPT")
        if not js:
            js = str(Path(C.REPO) / "weed_llm_benchmark" / "run_inc2_job.sh")
            os.environ["INC_JOB_SCRIPT"] = js
        if Path(js).name != "run_inc2_job.sh":
            raise StreamError("INC_JOB_SCRIPT is %s: a v2 segment must submit run_inc2_job.sh (the v1 executor "
                              "refuses every tsw image)" % js)

    def _write_refusal(self, refusal, ana, k):
        rec = {"utc": _utc(self._now()), "k_requested": k, "refusal": refusal,
               "eligible_images": sum(u["size"] for u in ana["units"]), "M": self.fold.defn["M"],
               "reasons": ana["reasons"], "guard_excluded": ana.get("guard_excluded", {})}
        _write_json(self.p.cut_refusal, rec)

    def _write_increment(self, plan, n, seg_n, step, ana):
        inc = inc_name(n)
        rows = plan["rows"]
        man_rows = [r["man"] for r in rows]
        mpath = self.p.inc_manifest(inc)
        msha = C.write_manifest(mpath, man_rows)
        side = []
        for r in sorted(rows, key=lambda r: r["key"]):
            q = r["row"]
            lic, ro = _row_licence(r)
            rel = r.get("licence_release")
            side.append({"key": r["key"], "source": r["source"], "batch": r["batch"], "capture_group": r["capture_group"],
                         "verifier": r["verifier"], "reference": q.get("reference"), "species_boxes": r["species"],
                         "other_boxes": q.get("other_boxes"), "admission": q.get("admission"),
                         "n_masked": q.get("n_masked"), "masked_area_frac": q.get("masked_area_frac"),
                         "lab_group": q.get("lab_group"), "licence": lic, "research_only": ro,
                         "licence_release": None if rel is None else {k: rel.get(k) for k in (
                             "by", "reason", "seq", "scope", "licence", "research_only")},
                         "prior": q.get("prior"),
                         "stream_pins_sha": q.get("stream_pins_sha"), "image": r["man"]["image"],
                         "sha256": r["man"]["sha256"], "unmasked_image": q.get("unmasked_image"),
                         "hashes": {kk: vv for kk, vv in r["hashes"].items() if vv is not None},
                         "queue_row_sha256": _sha_obj({kk: vv for kk, vv in q.items() if not kk.startswith("_")})})
        rpath = self.p.inc_rows(inc)
        rsha = _write_jsonl(rpath, side)
        sp = [0] * C.OTHER_PLANT
        for r in rows:
            for i, v in enumerate(r["species"]):
                sp[i] += v
        names = species_names()
        sources = dict(collections.Counter(r["source"] for r in rows))
        kinds = dict(collections.Counter(str(r["row"].get("admission") or "whole") for r in rows))
        maf = [float(r["row"]["masked_area_frac"]) for r in rows
               if isinstance(r["row"].get("masked_area_frac"), (int, float))]
        meta = {"increment": inc, "segment": seg_n, "step": step, "images": len(rows), "sources": sources,
                "first_source": plan["first_source"], "target_boxes": {names[i]: sp[i] for i in range(len(names))},
                "admission": kinds, "n_masked": sum(int(r["row"].get("n_masked") or 0) for r in rows),
                "masked_area_frac": ({"mean": _mean(maf), "max": max(maf)} if maf else None),
                "capture_groups": plan["capture_groups"],
                "split_capture_groups": [list(x) for x in plan["split_capture_groups_keys"]],
                "split_near_dup_groups": plan.get("split_near_dup_groups") or [],
                "species_cap": plan["species_cap"], "verifier": plan["verifier"],
                "reference": sorted({str(r["row"].get("reference")) for r in rows}),
                "stream_pins_sha": sorted({str(r["row"].get("stream_pins_sha")) for r in rows}),
                "queue": {"path": str(ana["view"].queue_path), "sha256": ana["view"].sha256,
                          "events_sha256": ana["view"].events_sha256},
                "seed_text": plan["seed_text"], "seed": plan["seed"], "attempts": plan["attempts"],
                "research_only_rows": sum(1 for r in rows if _row_licence(r)[1]),
                "lab_groups": dict(collections.Counter(str(r["row"].get("lab_group")) for r in rows)),
                "licences": dict(collections.Counter(str(_row_licence(r)[0]) for r in rows)),
                "prior": dict(collections.Counter(str(r["row"].get("prior")) for r in rows if r["row"].get("prior"))),
                "manifest": {"path": str(mpath), "sha256": msha}, "rows": {"path": str(rpath), "sha256": rsha},
                "draw": "select.balanced_order + select.draw_parts([order], sizes, 1, M), unchanged"}
        mepath = self.p.inc_meta(inc)
        mesha = _write_json(mepath, meta)
        event = {"increment": inc, "segment": seg_n, "step": step,
                 "manifest": {"path": str(mpath), "sha256": msha, "n_images": len(rows)},
                 "rows": {"path": str(rpath), "sha256": rsha}, "meta": {"path": str(mepath), "sha256": mesha},
                 "seed_text": plan["seed_text"], "seed": plan["seed"], "sources": sources,
                 "species_boxes": meta["target_boxes"], "first_source": plan["first_source"],
                 "split_capture_groups": meta["split_capture_groups"], "species_cap": plan["species_cap"],
                 "verifier": plan["verifier"], "queue_sha256": ana["view"].sha256}
        return {"id": inc, "manifest": event["manifest"], "meta": event["meta"], "sources": sorted(sources),
                "research_only_rows": meta["research_only_rows"], "event": event}

    # --------------------------------------------------------------- commit
    def commit(self, exp):
        """L19: dispose of a finished segment's increments (module docstring)."""
        with self.writing():
            f = self.fold
            seg = next((s for s in f.segments.values() if s["exp"] == exp), None)
            if seg is None:
                raise StreamError("%s is not a segment of stream %s" % (exp, self.sid))
            if seg["state"] != "built":
                raise StreamError("segment %s is %s; only a built segment is committed" % (exp, seg["state"]))
            paths = D.Paths(exp)
            st = _read_json(paths.state)
            if not st or not st.get("done"):
                raise StreamError("segment %s is not finished (driver state done=%s): an unfinished segment is "
                                  "never committed" % (exp, (st or {}).get("done")))
            defn = _read_json(paths.exp_json)
            if seg.get("definition_sha256") and _sha_obj(D._comparable(defn)) != seg["definition_sha256"]:
                raise StreamError("%s is not the definition the build recorded (sha256 of its content)" % paths.exp_json)
            led = _driver_ledger(paths)
            cfg = D.pinned_gate_config(st)
            _doc, g3_steps, g3_truth, g3_rec = self.deps.gate3().read(exp)
            steps_out = {}
            for r in defn["recipes"]:
                steps_out[r] = []
                for n, stp in enumerate(defn["steps"], 1):
                    g = led.get("gate/%s/%d" % (r, n))
                    if g is None:
                        raise StreamError("%s has no gate decision for %s step %d although it is done" % (exp, r, n))
                    v3 = normalise(g3_steps.get((r, n)), g)
                    truth = _truth_verdict(led, g3_truth, n) if defn.get("truth") else None
                    disp = dispose(v3, truth, cfg.p_reject)
                    pinned = g["decision"]
                    steps_out[r].append({"k": n, "increment": stp["name"], "v3": v3, "truth": truth,
                                         "disposition": disp,
                                         "pinned": {"verdict": pinned["verdict"], "p_data": pinned["p_data"],
                                                    "p_recipe": pinned["p_recipe"],
                                                    "guards": {kk: bool(vv["passed"]) for kk, vv in pinned["guards"].items()},
                                                    "species_failed": pinned["guards"]["species"]["failed"],
                                                    "blame": pinned["attribution"].get("blame")}})
            if len(defn["recipes"]) == 1:
                chosen = list(defn["recipes"])[0]
                choice = {"rule": "one recipe", "table": None}
            else:
                RC = self.deps.recipes()
                gates = [e for e in led.values() if e.get("type") == "gate"]
                arm_id = (defn.get("arm") or {}).get("id", DEFAULT_ARM)
                choice = RC.stage_b_choice(gates, {r: r for r in defn["recipes"]}, arm=arm_id,
                                           p_recipe_flag=cfg.p_recipe_flag)
                chosen = choice["chosen"]
            steps = steps_out[chosen]
            rejects = [s for s in steps if s["v3"]["verdict"] == G.REJECT]
            recipe_rejects = [s for s in rejects if s["disposition"] == RECIPE]
            d30 = {"rejects": len(rejects), "recipe_caused": len(recipe_rejects),
                   "fires": bool(rejects) and 2 * len(recipe_rejects) >= len(rejects)}
            sp_count = collections.Counter(x for s in rejects for x in s["v3"]["species_failed"])
            d33 = {"rejects": len(rejects), "species_fail_counts": dict(sp_count),
                   "species": sorted(x for x, c in sp_count.items() if rejects and 2 * c >= len(rejects))}
            discordant = [{"k": s["k"], "increment": s["increment"], "pinned": s["pinned"]["verdict"],
                           "v3": s["v3"]["verdict"]} for s in steps if s["pinned"]["verdict"] != s["v3"]["verdict"]]
            stale = seg["base_pool"] != f.pool
            dispositions, counted = {}, {}
            for s in steps:
                inc = s["increment"]
                if stale:
                    dispositions[inc] = "stale"
                    counted[inc] = False
                else:
                    dispositions[inc] = s["disposition"]
                    counted[inc] = s["disposition"] in RETURN_DISPOSITIONS and not (
                        s["disposition"] == RECIPE and d30["fires"])
            pool_rec = None
            accepted = [inc for inc, d in dispositions.items() if d == ACCEPTED]
            if not stale:
                pool_rec = self._commit_pool(seg["n"], accepted)
            unavailable = [{"chain": r, "k": s["k"], "reason": s["v3"]["v3_unavailable_reason"]}
                           for r, ss in steps_out.items() for s in ss if not s["v3"]["v3_applied"]]
            self.event("commit", segment=seg["n"], exp=exp, exp_json_sha256=C.sha256_file(paths.exp_json),
                       driver_ledger_sha256=C.sha256_file(paths.ledger), chosen=chosen, recipe=chosen,
                       accepted=[i for i, d in dispositions.items() if d == ACCEPTED], choice=choice,
                       steps={r: v for r, v in steps_out.items()}, dispositions=dispositions, counted=counted,
                       stale_base=stale, pool=pool_rec, d30=d30, d33=d33, discordant=discordant,
                       research_only=(pool_rec or {}).get("research_only"),
                       gate={"config": G.config_record(cfg), "version": "v3 (inc2.gate3)", "gate3": g3_rec,
                             "v3_unavailable": unavailable})
            for inc, disp in dispositions.items():
                n_img = f.increments[inc]["n_images"]
                if disp == DATA:
                    self.event("quarantine", scope="increment", increment=inc, reason="data", segment=seg["n"],
                               n_images=n_img, sources=f.increments[inc]["sources"])
                elif disp in RETURN_DISPOSITIONS or disp == "stale":
                    nn = sum(1 for k in f.inc_rows(inc) if f.keys[k]["status"] == "neutral"
                             and f.keys[k]["increment"] == inc)
                    self.event("return", increment=inc, segment=seg["n"], disposition=disp, counted=counted[inc],
                               n_images=n_img - nn, n_neutral=nn)
                    if nn:
                        self.event("quarantine", scope="increment", increment=inc, reason="neutral",
                                   segment=seg["n"], n_images=nn, sources=f.increments[inc]["sources"])
            self._segment_report(exp)
            self._log("committed %s: chain %s; %s%s" % (exp, chosen, dict(collections.Counter(dispositions.values())),
                                                         "; stale base (returned uncounted)" if stale else
                                                         "; P_%d = %d images" % (seg["n"], pool_rec["n_images"])))
            return {"chosen": chosen, "dispositions": dispositions, "pool": pool_rec, "d30": d30, "d33": d33}

    def _commit_pool(self, s, accepted):
        """P_s = P_{s-1} + the accepted increments, pairwise disjoint by key,
        path, sha256 and POOL_BITS dHash."""
        f = self.fold
        prev = f.current_pool()
        parts = [(prev["name"], f.pool_rows(prev["name"]), f.pool_hashes(prev["name"]))]
        for inc in accepted:
            rows = C.read_manifest(_check_rec(f.increments[inc]["manifest"], "increment manifest"))
            side = f.inc_rows(inc)
            parts.append((inc, rows, {k: [int(h) for h in side[k].get("hashes", {}).values() if h is not None]
                                      for k in side}))
        keys, images, shas = {}, {}, {}
        idx = NearHashIndex()
        clashes = []
        for name, rows, hs in parts:
            for r in rows:
                for field, seen in (("key", keys), ("image", images), ("sha256", shas)):
                    if r[field] in seen and seen[r[field]] != name:
                        clashes.append((field, r["key"], seen[r[field]], name))
                    seen.setdefault(r[field], name)
            for k, hlist in hs.items():
                for h in hlist:
                    for (owner, bits) in idx.matches(h):
                        if owner[0] != name:
                            clashes.append(("dhash<=%d" % POOL_BITS, k, owner, name))
            for k, hlist in hs.items():
                for h in hlist:
                    idx.add(h, (name, k), max_bits=POOL_BITS)
        if clashes:
            raise StreamError("P_%d would not be pairwise disjoint: %d clash(es), e.g. %s" % (s, len(clashes), clashes[:3]))
        rows_all = [r for _n, rows, _h in parts for r in rows]
        hashes = {}
        for _n, _rows, hs in parts:
            hashes.update(hs)
        ro_rows = prev.get("research_only_rows", 0) + sum(
            1 for inc in accepted for r in f.inc_rows(inc).values() if r.get("research_only"))
        ro = {"rows": ro_rows, "known": prev.get("research_only_known", False)}
        return self._write_pool(s, rows_all, hashes, parent=prev["name"], parts=[prev["name"]] + list(accepted),
                                research_only=ro)

    def _segment_report(self, exp):
        try:
            from . import stream_report as SR
            SR.segment_report(exp, stream=self)
        except Exception as e:  # noqa: BLE001 - the report never blocks a commit
            log("WARNING: the segment report of %s was not written: %s: %s" % (exp, type(e).__name__, e))
        self._stream_report()

    def _stream_report(self):
        """stream/<sid>/report.{md,json} (§3.7), refreshed at every commit and
        milestone comparison so that the owner's report is never older than the
        last decision (it carries test: for people, outside the evidence
        allow-list). It never blocks the operation."""
        try:
            from . import stream_report as SR
            SR.build(self.sid, stream=self)
        except Exception as e:  # noqa: BLE001
            log("WARNING: the stream report of %s was not written: %s: %s" % (self.sid, type(e).__name__, e))

    # ------------------------------------------------------------ withdraw
    def withdraw(self, exp, reason, decided_by="platform"):
        if not str(reason or "").strip():
            raise StreamError("withdraw needs a --reason")
        with self.writing():
            seg = next((s for s in self.fold.segments.values() if s["exp"] == exp), None)
            if seg is None or seg["state"] != "built":
                raise StreamError("%s is not a built, uncommitted segment of %s" % (exp, self.sid))
            self.event("withdraw", by=decided_by, segment=seg["n"], exp=exp, reason=str(reason).strip())
            self._log("withdrew %s: its images are back in the queue, uncounted" % exp)

    # -------------------------------------------------------------- milestone
    def milestone(self, force=False, decided_by="platform"):
        """L20: build the next milestone on the current pool; when the one in
        flight is done, compare it instead."""
        with self.writing():
            f = self.fold
            m = f.milestone_in_flight()
            if m is not None:
                st = _read_json(D.Paths(m["exp"]).state) or {}
                if st.get("done"):
                    return self._compare_milestone(m)
                self._log("milestone %s is running (not done); nothing to do" % m["exp"])
                return {"waiting": m["exp"]}
            pool = f.current_pool()
            last = f.last_good_milestone()
            if (last is not None and (f.pools.get(last["pool"]) or {}).get("sha256") == pool["sha256"]
                    and not force):
                raise StreamError("the pool %s holds the same images as the last milestone's (%s): nothing to "
                                  "consolidate" % (pool["name"], last["exp"]))
            n = f.next_ms
            exp = milestone_exp(self.sid, n)
            if D.Paths(exp).exp_json.exists():
                raise StreamError("%s exists without a milestone record" % exp)
            arm = (f.arm or {}).get("id", DEFAULT_ARM)
            argv = ["build", "--exp", exp, "--manifest", pool["path"],
                    "--seeds", ",".join(str(s) for s in MILESTONE_SEEDS), "--arm", arm, "--role", "milestone",
                    "--final-exams", ",".join(MILESTONE_EXAMS)]
            if f.defn.get("testing"):
                argv += (["--testing-settings", json.dumps(f.defn["testing"])] if isinstance(f.defn["testing"], dict)
                         else ["--testing"])
            self.event("milestone", by=decided_by, phase="build", n=n, exp=exp, pool=pool["name"],
                       pool_sha256=pool["sha256"], seeds=list(MILESTONE_SEEDS), argv=argv, arm=arm)
            rc = self.deps.baseline_runner(argv)
            if rc != 0 or not D.Paths(exp).exp_json.exists():
                self.event("milestone", phase="build_failed", n=n, exp=exp, rc=rc)
                raise StreamError("inc2.baseline build of %s exited %s" % (exp, rc))
            log("built experiment %s" % exp)
            inc_rec = self._score_incumbent(n, exp)
            self._log("milestone %s built on %s (%d images, 5 seeds); incumbent: %s"
                      % (exp, pool["name"], pool["n_images"], inc_rec.get("status")))
            return {"built": exp}

    def _score_incumbent(self, n, exp):
        """§3.6's secondary number: the chain incumbent of the last committed
        segment on the current pool's lineage, scored on the milestone's exams
        by inc2.baseline secondary (a kind-final spec in the milestone
        experiment, run by run_inc2_job.sh; the driver does not track it),
        submitted here. Its scores stay in INC_DIR/<exp>/runs/
        secondary__incumbent/, never in the stream ledger."""
        f = self.fold
        segs = [s for s in f.segments.values() if s["state"] == "committed" and (s["commit"] or {}).get("pool")
                and s["commit"]["pool"]["name"] in f.pool_lineage()]
        rec = {"status": "none", "exams": list(MILESTONE_EXAMS), "run_id": SECONDARY_RUN}
        if segs:
            seg = segs[-1]
            st = _read_json(D.Paths(seg["exp"]).state) or {}
            ch = (st.get("chains") or {}).get(seg["commit"]["chosen"]) or {}
            w = (ch.get("incumbent") or {}).get("weights")
            if w and Path(w).exists():
                src = "%s chain %s incumbent %s" % (seg["exp"], seg["commit"]["chosen"], ch["incumbent"]["run_id"])
                rec.update(segment=seg["exp"], chain=seg["commit"]["chosen"], incumbent=ch["incumbent"]["run_id"],
                           weights=w, source=src)
                try:
                    argv = self.deps.secondary_runner(exp, w, src)
                    rec["argv"] = argv
                    rec["job_id"] = self.deps.submitter(argv)
                    rec["status"] = "submitted"
                except Exception as e:  # noqa: BLE001 - the secondary number never blocks a milestone
                    rec["status"] = "failed"
                    rec["error"] = "%s: %s" % (type(e).__name__, e)
            else:
                rec["status"] = "no weights"
        _write_json(self.p.milestone_dir(n) / "incumbent.json", rec)
        return rec

    def compare(self, exp):
        with self.writing():
            f = self.fold
            m = next((x for x in f.milestones.values() if x["exp"] == exp and x["n"] > 0), None)
            if m is not None:
                if m["state"] != "built":
                    raise StreamError("milestone %s is %s" % (exp, m["state"]))
                return self._compare_milestone(m)
            if f.feasibility and f.feasibility.get("exp") == exp:
                return self._read_feasibility(exp)
            b = self._find_bisect_arm(exp)
            if b is not None:
                return self._decide_bisect(b)
            raise StreamError("%s is neither a milestone, the feasibility chain nor a bisect arm of %s" % (exp, self.sid))

    def _compare_milestone(self, m):
        f = self.fold
        st = _read_json(D.Paths(m["exp"]).state) or {}
        if not st.get("done"):
            raise StreamError("milestone %s is not finished" % m["exp"])
        prev = f.last_good_milestone(exclude=m["n"])
        if prev is None:
            raise StreamError("no earlier milestone to compare %s with (milestone 0 is init's --milestone0)" % m["exp"])
        ins_new, ins_old = [], []
        new_s, old_s = self._base_scores(m["exp"], ins_new), self._base_scores(prev["exp"], ins_old)
        cfg = G.GateConfig(require_production=not f.defn.get("testing"))
        td = G.truth_detail(new_s, old_s, cfg)
        metric = cfg.metric
        nv, ov = [float(s[metric]) for s in new_s], [float(s[metric]) for s in old_s]
        pp = permutation_p(nv, ov)
        hurts = pp <= PERM_ALPHA + 1e-12 and _mean(nv) < _mean(ov)
        if hurts:
            verdict = "hurts"
        elif td["verdict"] == G.HURTS and td["species"]["failed"] and td["p"] > cfg.p_reject:
            verdict = "species_only"
        elif td["verdict"] == G.HURTS:
            verdict = "hurts_not_significant"
        else:
            verdict = td["verdict"]
        to_pool = prev["pool"] if hurts else None
        cmp = {"milestone": m["exp"], "compared_with": prev["exp"], "metric": metric, "exam": "dev",
               "new": nv, "old": ov, "new_mean": _mean(nv), "new_sd": _sd(nv), "old_mean": _mean(ov),
               "old_sd": _sd(ov), "perm_p": pp, "perm_alpha": PERM_ALPHA, "truth_p": td["p"],
               "truth_verdict": td["verdict"], "species_failed": td["species"]["failed"], "verdict": verdict,
               "rollback_recommended": hurts, "to_pool": to_pool, "inputs": {"new": ins_new, "old": ins_old},
               "rule": "only the one-sided 5 v 5 permutation test (p <= %.3f, mean(new) < mean(old)) rolls back; a "
                       "species-guard-only 'hurts' raises D33 / X17 (§3.6 [review])" % PERM_ALPHA}
        _write_json(self.p.milestone_dir(m["n"]) / "comparison.json", cmp)
        self.event("milestone", phase="compare", n=m["n"], exp=m["exp"], compared_with=prev["exp"],
                   verdict=verdict, perm_p=pp, truth_p=td["p"], new_mean_dev=_mean(nv), old_mean_dev=_mean(ov),
                   species_failed=td["species"]["failed"], rollback_recommended=hurts, to_pool=to_pool,
                   inputs={"new": ins_new, "old": ins_old})
        try:
            from . import stream_report as SR
            SR.milestone_entry(self, m["n"])
        except Exception as e:  # noqa: BLE001
            log("WARNING: the milestone entry of %s was not written: %s: %s" % (m["exp"], type(e).__name__, e))
        self._stream_report()
        self._log("milestone %s vs %s on dev: %s (perm p %.4f, truth P %.3f)" % (m["exp"], prev["exp"], verdict, pp, td["p"]))
        return cmp

    # --------------------------------------------------------------- rollback
    def rollback(self, to, decided_by="platform"):
        """L21: the pool pointer returns to P_c; the increments accepted since
        are suspect (card X4; bisect runs L27)."""
        with self.writing():
            f = self.fold
            if to not in f.pools:
                raise StreamError("%s is not a pool of %s (%s)" % (to, self.sid, list(f.pools)))
            lineage = f.pool_lineage()
            if to not in lineage or to == f.pool:
                raise StreamError("%s is not an ancestor of the current pool %s" % (to, f.pool))
            last = [x for x in f.milestones.values() if x.get("verdict") == "hurts" and x.get("rollback_recommended")]
            rec = last[-1] if last else None
            envelope = rec is not None and rec.get("to_pool") == to and not any(
                r.get("milestone") == rec["exp"] for r in f.rollbacks)
            if not envelope and not _human(decided_by):
                raise StreamError("a rollback to %s is not the recommended one of the last 'hurts' milestone (at most "
                                  "one per milestone): it needs a person (--decided-by human:<id>)" % to)
            suspect = []
            for name in lineage[:lineage.index(to)]:
                suspect += [x for x in f.pools[name].get("parts", [])[1:] if x in f.increments]
            suspect = sorted(set(suspect), key=lambda x: int(x[3:]))
            self.event("rollback", by=decided_by, to=to, **{"from": f.pool}, suspect=suspect,
                       milestone=rec["exp"] if rec else None, envelope=envelope)
            self._log("rolled back to %s; suspect increments %s (card X4; bisect with L27)" % (to, suspect))
            return suspect

    # ----------------------------------------------------------------- bisect
    def bisect(self, from_pool, decided_by="platform"):
        """L27: one cold 3-seed arm P_c + {increment} per suspect increment,
        compared with P_c's milestone seeds by gate.truth_detail."""
        with self.writing():
            f = self.fold
            rbs = [r for r in f.rollbacks if r["to"] == from_pool]
            if not rbs:
                raise StreamError("no rollback to %s to bisect" % from_pool)
            rb = rbs[-1]
            b = f.bisects.get(rb["n"])
            if b and b["arms"]:
                return self._decide_bisect(rb["n"])
            if not rb["suspect"]:
                raise StreamError("the rollback to %s suspended no increment" % from_pool)
            if not [m for m in f.milestones.values() if m["pool"] == from_pool and m.get("good")]:
                # the arms are decided against P_c's milestone seeds: without them they could never be decided,
                # and their GPU-hours would buy nothing (X4 stays with a person)
                raise StreamError("no good milestone on %s to compare bisect arms with: the bisection could never "
                                  "decide (card X4 stays with a person)" % from_pool)
            self._check_job_script()
            RC = self.deps.recipes()
            arm_id = (f.arm or {}).get("id", DEFAULT_ARM)
            testing = f.defn.get("testing") or False
            cold = RC.table(arm_id)["cold"]
            arm_rec = RC.resolve_arm(arm_id, repo=self.deps.repo or C.REPO, require_weights=not testing)
            base_rows = f.pool_rows(from_pool)
            g = self.guard()
            arms, defns = {}, []
            n_arm = sum(len(x["arms"]) for x in f.bisects.values())
            for inc in rb["suspect"]:
                rows = C.read_manifest(_check_rec(f.increments[inc]["manifest"], "increment manifest"))
                for r in rows:
                    h, v = self.deps.get_hasher()(r["image"])
                    reason, _m = g.check(h, v)
                    if reason is not None and reason not in HARMLESS_GUARD_REASONS:
                        raise StreamError("bisect: %s of %s fails the never-train v2 guard (%s)" % (r["key"], inc, reason))
                n_arm += 1
                exp = bisect_exp(self.sid, n_arm)
                mp = self.p.bisect / ("rb%02d" % rb["n"]) / ("%s.jsonl" % inc)
                sha = C.write_manifest(mp, base_rows + rows)
                # §8: the arm trains on P_c plus the increment, so its models are research-only when either holds one
                ro_inc = sum(1 for r in f.inc_rows(inc).values() if r.get("research_only"))
                defn = {"exp": exp, "type": "baseline", "builder": "inc2.stream bisect", "testing": testing,
                        "seeds": list(BISECT_SEEDS), "decision_exam": D.DECISION_EXAM, "final_exams": list(BISECT_EXAMS),
                        "base": {"name": "%s_plus_%s" % (from_pool.replace("P_", "P"), inc), "manifest": str(mp),
                                 "manifest_sha256": sha, "n_images": len(base_rows) + len(rows), "recipe": cold},
                        "stream": {"sid": self.sid, "rollback": rb["n"], "from": from_pool, "increment": inc,
                                   "research_only": _research_only(f.pools[from_pool],
                                                                   [{"research_only_rows": ro_inc}])}}
                defn.update(RC.stamp(arm_rec))
                D.validate_definition(defn)
                D.check_definition_data(defn)
                arms[inc] = exp
                defns.append(defn)
            self.event("bisect", by=decided_by, phase="build", rollback=rb["n"], rollback_utc=rb["utc"], arms=arms,
                       **{"from": from_pool})
            for defn in defns:
                D.Driver(defn["exp"], backend=self.deps.backend, clock=self.clock, quiet=self.quiet).init(defn)
                log("built experiment %s" % defn["exp"])
            self._log("bisect of the rollback to %s: %d arm(s) built" % (from_pool, len(defns)))
            return arms

    def _find_bisect_arm(self, exp):
        for rb, b in self.fold.bisects.items():
            if exp in b["arms"].values():
                return rb
        return None

    def _decide_bisect(self, rb_n):
        f = self.fold
        b = f.bisects[rb_n]
        rb = f.rollbacks[rb_n - 1]
        base_ms = [m for m in f.milestones.values() if m["pool"] == rb["to"] and m.get("good")]
        if not base_ms:
            raise StreamError("no milestone on %s to compare the bisect arms with" % rb["to"])
        ins_without = []
        without = self._base_scores(base_ms[-1]["exp"], ins_without)
        cfg = G.GateConfig(require_production=not f.defn.get("testing"))
        decisions, details = {}, {}
        for inc, exp in b["arms"].items():
            if inc in b["decisions"]:
                continue
            st = _read_json(D.Paths(exp).state) or {}
            if not st.get("done"):
                continue
            ins_with = []
            td = G.truth_detail(self._base_scores(exp, ins_with), without, cfg)
            decisions[inc] = td["verdict"]
            details[inc] = {"exp": exp, "p": td["p"], "species_failed": td["species"]["failed"],
                            "with_mean": td["with_mean"], "without_mean": td["without_mean"],
                            "inputs": {"with": ins_with, "without": ins_without}}
        if decisions:
            self.event("bisect", phase="decide", rollback=rb_n, rollback_utc=rb["utc"], decisions=decisions,
                       details=details, compared_with=base_ms[-1]["exp"])
        pending = [inc for inc in b["arms"] if inc not in f.bisects[rb_n]["decisions"]]
        self._log("bisect %d: decided %s; waiting on %s" % (rb_n, decisions, pending))
        return {"decided": decisions, "pending": pending}

    # ------------------------------------------------------------ feasibility
    def feasibility(self, holdout="tsw22", m=None, decided_by="platform", recipes=None):
        """L28 Stage C: base = P_0 minus one increment D of exactly M target
        images of the holdout part, drawn as whole sessions (the last one
        split), one clean: false step D, the Stage B recipes (--recipes: r0
        and the Stage A survivor), truth on. Stage C asks whether the gate
        can accept anything at this stream's M, so --m must be that M (a
        person may ask for another); a Stage C whose build never completed
        (repaired as build_failed) may be built again."""
        with self.writing():
            f = self.fold
            if f.feasibility and f.feasibility.get("exp") and f.feasibility.get("phase") != "build_failed":
                raise StreamError("Stage C runs once per stream version (%s exists)" % f.feasibility["exp"])
            if m is not None and int(m) != int(f.defn["M"]) and not _human(decided_by):
                raise StreamError("--m %s is not this stream's M = %d: Stage C measures the gate at the stream's own "
                                  "M (another M needs a person, --decided-by human:<id>)" % (m, f.defn["M"]))
            self._check_job_script()
            M = int(m or f.defn["M"])
            p0 = list(f.pools)[0]
            rows = f.pool_rows(p0)
            prefix = holdout + "__"
            tgt = [r for r in rows if r["key"].startswith(prefix)
                   and any(b[0] < C.OTHER_PLANT for b in C.read_yolo(r["label"]))]
            by_s = collections.defaultdict(list)
            for r in tgt:
                by_s[r.get("session") or r["key"]].append(r)
            import numpy as np
            rng = np.random.default_rng(C.stable_int("%s/feasibility/%s" % (self.sid, holdout)))
            sessions = sorted(by_s)
            order = [sessions[i] for i in rng.permutation(len(sessions))]
            D_rows, split = [], None
            for s in order:
                rs = sorted(by_s[s], key=lambda r: r["key"])
                if len(D_rows) + len(rs) <= M:
                    D_rows += rs
                elif len(D_rows) < M:
                    need = M - len(D_rows)
                    D_rows += rs[:need]
                    split = {"session": s, "taken": need, "of": len(rs)}
                if len(D_rows) == M:
                    break
            if len(D_rows) != M:
                raise StreamError("the holdout %s holds %d target images, fewer than M = %d" % (holdout, len(tgt), M))
            dkeys = {r["key"] for r in D_rows}
            base = [r for r in rows if r["key"] not in dkeys]
            fdir = self.p.feasibility
            dp, bp = fdir / ("D_%s.jsonl" % holdout), fdir / ("base_minus_D_%s.jsonl" % holdout)
            dsha, bsha = C.write_manifest(dp, D_rows), C.write_manifest(bp, base)
            RC = self.deps.recipes()
            arm_id = (f.arm or {}).get("id", DEFAULT_ARM)
            testing = f.defn.get("testing") or False
            table = RC.table(arm_id)
            if isinstance(recipes, str):
                recipes = [r for r in recipes.split(",") if r.strip()]
            recs = self.recipes_for(1, recipes or None)
            exp = feasibility_exp(self.sid)
            defn = {"exp": exp, "type": "chain", "builder": "inc2.stream feasibility", "testing": testing,
                    "replay_mode": "full", "gate": dict(GATE_BLOCK), "seeds": list(CHAIN_SEEDS),
                    "decision_exam": D.DECISION_EXAM, "final_exams": list(SEGMENT_EXAMS),
                    "base": {"name": "P0_minus_D", "manifest": str(bp), "manifest_sha256": bsha, "n_images": len(base),
                             "recipe": table["cold"]},
                    "steps": [{"name": "D_%s" % holdout, "manifest": str(dp), "manifest_sha256": dsha,
                               "n_images": len(D_rows), "clean": False, "kind": "stage_c"}],
                    "recipes": {r: table[r] for r in recs}, "truth": True, "truth_recipe": table["cold"],
                    "increment_images": M,
                    "stream": {"sid": self.sid, "stage": "C", "holdout": holdout, "split_session": split,
                               # §8: base and D are both drawn from P_0, so P_0's flag holds for the models
                               "research_only": _research_only(f.pools[p0], [])}}
            defn.update(RC.stamp(RC.resolve_arm(arm_id, repo=self.deps.repo or C.REPO, require_weights=not testing)))
            D.validate_definition(defn)
            D.check_definition_data(defn)
            self.event("feasibility", by=decided_by, phase="build", exp=exp, holdout=holdout, m=M,
                       D={"path": str(dp), "sha256": dsha, "n_images": len(D_rows)},
                       base={"path": str(bp), "sha256": bsha, "n_images": len(base)}, recipes=recs,
                       split_session=split, sessions=len({r.get("session") for r in D_rows}))
            D.Driver(exp, backend=self.deps.backend, clock=self.clock, quiet=self.quiet).init(defn)
            log("built experiment %s" % exp)
            self._log("Stage C %s built: base %d images, D %d %s images" % (exp, len(base), len(D_rows), holdout))
            return exp

    def _read_feasibility(self, exp):
        paths = D.Paths(exp)
        st = _read_json(paths.state) or {}
        if not st.get("done"):
            raise StreamError("Stage C %s is not finished" % exp)
        defn = _read_json(paths.exp_json)
        led = _driver_ledger(paths)
        cfg = D.pinned_gate_config(st)
        _doc, g3_steps, g3_truth, g3_rec = self.deps.gate3().read(exp)
        truth = _truth_verdict(led, g3_truth, 1)
        out = {}
        for r in defn["recipes"]:
            v3 = normalise(g3_steps.get((r, 1)), led["gate/%s/1" % r])
            g = v3["guards"]
            species_only = (v3["verdict"] == G.REJECT and not g["species"] and g["regression"] and g["flips"]
                            and v3["p_data"] > cfg.p_reject)
            out[r] = {"verdict": v3["verdict"], "disposition": dispose(v3, truth, cfg.p_reject),
                      "species_failed": v3["species_failed"], "species_only_reject": species_only,
                      "p_data": v3["p_data"], "p_recipe": v3["p_recipe"], "v3_applied": v3["v3_applied"]}
        res = {"per_recipe": out, "truth": truth, "gate3": g3_rec,
               "m_feasible": any(v["verdict"] == G.ACCEPT for v in out.values()),
               "species_only_reject": any(v["species_only_reject"] for v in out.values()),
               "d33_prospective": sorted({s for v in out.values() if v["species_only_reject"] for s in v["species_failed"]})}
        self.event("feasibility", phase="read", exp=exp, result=res)
        self._log("Stage C %s: %s" % (exp, {r: v["verdict"] for r, v in out.items()}))
        return res

    # ------------------------------------------------------------------ fork
    def fork(self, m, to=None, decided_by="platform"):
        """L22: a new stream version with M doubled; it adopts the current pool
        and the quarantine."""
        with self.writing():
            f = self.fold
            if f.forked_to:
                raise StreamError("%s was already forked to %s" % (self.sid, f.forked_to))
            if f.in_flight():
                raise StreamError("segments in flight %s: commit or withdraw them first" % [s["exp"] for s in f.in_flight()])
            m = int(m)
            doublings = int((f.defn.get("forked_from") or {}).get("doublings", 0))
            if m != 2 * f.defn["M"] and not _human(decided_by):
                raise StreamError("L22 doubles M (%d -> %d); another M is card X17 (a person)" % (f.defn["M"], 2 * f.defn["M"]))
            if doublings >= 2 and not _human(decided_by):
                raise StreamError("M has been doubled twice already: card X17 (a person)")
            # not "<sid>_m<M>": with a 3-digit M that is a milestone experiment's name (<sid>_mNNN)
            to = to or "%s_fork%d" % (self.sid, m)
            if re.match(r"^.+_[smcb]\d{3}$", to):
                raise StreamError("%r reads as a stream experiment name (<sid>_[smcb]NNN); name the new stream "
                                  "otherwise" % to)
            new = Stream(to, deps=self.deps, clock=self.clock, quiet=self.quiet)
            if new.p.root.exists() and any(new.p.root.iterdir()):
                raise StreamError("stream %s exists" % to)
            pool = f.current_pool()
            new.p.root.mkdir(parents=True, exist_ok=True)
            inh_keys = [{"key": k, "status": v["status"], "increment": v.get("increment"), "returns": v.get("returns", 0)}
                        for k, v in sorted(f.keys.items()) if v["status"] not in ELIGIBLE_STATUSES or v.get("returns")]
            kp = new.p.root / "inherited_keys.jsonl"
            _write_jsonl(kp, inh_keys)
            qp = new.p.root / "inherited_quarantine.json"
            _write_json(qp, {"from": self.sid, "entries": [{k: v for k, v in x.items() if k != "seq"} for x in f.quarantine]})
            last_ms = f.last_good_milestone()
            fork = {"rows": f.pool_rows(pool["name"]), "sha256": pool["sha256"], "path": pool["path"],
                    "hashes": f.pool_hashes(pool["name"]),
                    "research_only": {"rows": pool.get("research_only_rows", 0), "known": pool.get("research_only_known")},
                    "from": {"sid": self.sid, "pool": pool["name"], "ledger_head": f.head, "M": f.defn["M"],
                             "doublings": doublings + (1 if m == 2 * f.defn["M"] else 0)},
                    "inherited": {"keys": _file_rec(kp), "quarantine": _file_rec(qp),
                                  "quarantined_sources": f.q_sources, "chosen_recipe": f.chosen_recipe,
                                  "releases": [{k: v for k, v in r.items() if k not in ("seq", "inherited")}
                                               for r in f.releases]},
                    "milestone0": last_ms["exp"] if last_ms else None}
            new.init(lock=f.defn.get("lock_path"), m=m, k_max=f.defn["K_max"], arm=(f.arm or {}).get("id", DEFAULT_ARM),
                     stage_b=f.stage_b, testing=f.defn.get("testing") or False, decided_by=decided_by,
                     step1_dir=f.defn.get("step1_stream"), _fork=fork)
            self.event("fork", by=decided_by, to=to, m_new=m, pool=pool["name"],
                       new_ledger_head=new.ledger.head)
            self._log("forked to %s with M = %d on %s" % (to, m, pool["name"]))
            return to

    # --------------------------------------------------- quarantine, release
    def quarantine_source(self, source, cite, decided_by="platform"):
        if not re.match(r"^D\d+$", str(cite or "")):
            raise StreamError("L24 needs a firing diagnosis to cite (D28 or D31), got %r" % cite)
        if cite not in QUARANTINE_CITES and not _human(decided_by):
            raise StreamError("L24 quarantines a source on a firing D28 (leak) or D31 (data blamed) only (§6.3); "
                              "%s needs a person (--decided-by human:<id>)" % cite)
        with self.writing():
            if source in self.fold.q_sources:
                self._log("source %s is already quarantined" % source)
                return
            self.event("quarantine", by=decided_by, scope="source", source=source, cite=cite)

    def unquarantine_source(self, source, decided_by):
        if not _human(decided_by):
            raise StreamError("lifting a source quarantine is a person's decision (--decided-by human:<id>)")
        with self.writing():
            if source not in self.fold.q_sources:
                raise StreamError("source %s is not quarantined" % source)
            self.event("unquarantine", by=decided_by, source=source)

    def release(self, hold, reason, decided_by, source=None, keys_file=None, licence=None, research_only=True):
        """A person's release of a hold for the cutter (P8: the owner may
        release the funnel_F9 rows early; an R3 item at a hold's deadline).
        Without --source or --keys it covers every row of that hold kind,
        which only funnel_F9 allows: a licence and a join conflict are
        released row by row or source by source. The copy scan (h6_scan) is
        never released here: it is mandatory for every source that is not
        provenance-cleared (D-C, P9, §3.2), a copy dHash misses would inflate
        the success measure, and only the scan itself (step1_stream
        serve-holds, L17 scan-holds at the deadline) lifts it. A licence
        release fails closed: the rows it lifts are research_only unless the
        person records the licence text (--licence TEXT) and says they are
        not (--not-research-only); both are recorded and the cutter applies
        them (_row_licence). --not-research-only is refused for a text that
        names no known licence or restricts use, read as step1_stream reads
        a person's override (intake_licence_state)."""
        if hold not in HOLD_KINDS:
            raise StreamError("--hold must be one of %s" % (HOLD_KINDS,))
        if hold not in RELEASABLE_HOLDS:
            raise StreamError("%s is not released by hand: the embedding copy scan is mandatory (D-C, P9); it is "
                              "served by step1_stream serve-holds (L17 scan-holds), which releases what it clears"
                              % hold)
        if not _human(decided_by):
            raise StreamError("a hold is released by the funnel, the copy scan or a person (--decided-by human:<id>)")
        if source and keys_file:
            raise StreamError("name --source or --keys, not both")
        if not source and not keys_file and hold != "funnel_F9":
            raise StreamError("a stream-wide release is for funnel_F9 only; %s is released by --source or --keys"
                              % hold)
        licence = str(licence).strip() if licence is not None else None
        if hold != "licence" and (licence or research_only is not True):
            raise StreamError("--licence and --not-research-only belong to a licence release (--hold licence)")
        if research_only is not True and not licence:
            raise StreamError("--not-research-only needs the person's licence text (--licence TEXT): without it the "
                              "released rows stay research_only (fail closed, P6)")
        if research_only is not True:
            # the text is read as step1_stream reads a person's override (its one reader): an unknown licence, or one
            # that restricts use, keeps the rows research_only, so --not-research-only is refused for it
            S1 = _inc2("step1_stream")
            if S1 is None or not hasattr(S1, "intake_licence_state"):
                raise StreamError("inc2.step1_stream (the reader of a person's licence text) cannot be imported: "
                                  "--not-research-only is refused (fail closed, P6)")
            _l, ro = S1.intake_licence_state({"source": source}, {"licence": {"override": {
                "id": licence, "research_only": False}}})
            if ro:
                raise StreamError("--not-research-only: %r names no known licence that allows use beyond research "
                                  "(unknown, non-commercial, no-derivatives or restricted): the released rows stay "
                                  "research_only (fail closed, P6)" % licence)
        lic_rec = {"licence": licence or None, "research_only": research_only is not False} if hold == "licence" \
            else {}
        reason = str(reason or "R3 approval of a person")
        with self.writing():
            rec = None
            if keys_file:
                keys = [ln.strip() for ln in Path(keys_file).read_text().splitlines() if ln.strip()]
                kp = self.p.root / "releases" / ("r%03d.jsonl" % (len(self.fold.releases) + 1))
                _write_jsonl(kp, [{"key": k} for k in keys])
                rec = _file_rec(kp)
            self.event("release", by=decided_by, hold=hold, source=source, keys_file=rec, reason=reason,
                       scope="keys" if keys_file else ("source" if source else "all"), **lic_rec)

    # ---------------------------------------------------------------- summary
    def write_summary(self):
        s = self.summary()
        _write_json(self.p.summary, s)
        return s

    def summary(self):
        """queue_summary.json: what the autopilot's diagnoses read (§6.4). Dev
        numbers only; no test or ImageWeeds value."""
        f = self.fold
        M, K = f.defn["M"], f.defn["K_max"]
        try:
            ana = self.eligibility(self.queue(), compute_hashes=False)
            q_err = None
        except StreamError as e:
            ana, q_err = None, str(e)
        # Q: the eligible target images the cutter can actually cut (a near-duplicate group larger than M or a
        # mixed-verifier unit is reported apart, never counted as supply: D20 would not collect and D22 would build
        # against images no increment can take)
        Q = sum(e["tb"] > 0 for e in ana["cuttable"]) if ana else 0
        per_species = collections.Counter()
        per_source = collections.Counter()
        for e in (ana["cuttable"] if ana else []):
            per_source[e["source"]] += 1
            for i, v in enumerate(e["species"]):
                per_species[species_names()[i]] += v
        oldest = ana["oldest_eligible_utc"] if ana else None
        age = None
        if oldest and (_ts(oldest) or _date_ts(oldest)):
            age = (self._now() - (_ts(oldest) or _date_ts(oldest))) / 86400.0
        in_flight = f.in_flight()
        probe = None
        if ana and Q >= M:
            try:
                self._cut_one(ana, M, f.next_seg, 1, set(), set(), self._remainders(), False, collections.Counter())
                probe = {"exact_fill": True, "approximate": ana["rows_without_hash"] > 0}
            except CutRefused as e:
                probe = dict(e.reason, exact_fill=False, approximate=ana["rows_without_hash"] > 0)
        ready = (Q >= K * M or (Q >= M and age is not None and age >= STALE_DAYS))
        segs = []
        for s in f.segments.values():
            done = bool((_read_json(D.Paths(s["exp"]).state) or {}).get("done")) if s["state"] == "built" else None
            c = s.get("commit") or {}
            segs.append({"n": s["n"], "exp": s["exp"], "state": s["state"], "done": done, "base_pool": s["base_pool"],
                         "increments": s["increments"], "recipes": s["recipes"], "truth": s["truth"],
                         "built_utc": s["built_utc"], "committed_utc": c.get("utc"), "chosen": c.get("chosen"),
                         "dispositions": c.get("dispositions"), "d30": c.get("d30"), "d33": c.get("d33"),
                         "stale_base": c.get("stale_base")})
        committed = [s for s in f.segments.values() if s["state"] == "committed"]
        last = committed[-1] if committed else None
        last_ms = f.last_good_milestone()
        covered = set(f.pool_lineage(last_ms["pool"])) if last_ms else set()
        since = [s for s in committed if (s["commit"] or {}).get("pool")
                 and s["commit"]["pool"]["name"] in f.pool_lineage() and s["commit"]["pool"]["name"] not in covered]
        acc_since = sum(1 for s in since for d in (s["commit"].get("dispositions") or {}).values() if d == ACCEPTED)
        first_acc = [s["commit"]["utc"] for s in since if any(d == ACCEPTED for d in (s["commit"].get("dispositions") or {}).values())]
        days = ((self._now() - _ts(first_acc[0])) / 86400.0) if first_acc else None
        due = []
        if acc_since >= MILESTONE_TRIGGERS["accepted_increments"]:
            due.append("accepted_increments")
        if len(since) >= MILESTONE_TRIGGERS["segments"]:
            due.append("segments")
        if days is not None and days >= MILESTONE_TRIGGERS["days_with_accepted"]:
            due.append("days_with_accepted")
        boundary = self._boundary()
        if boundary and boundary.get("fires"):
            due.append("boundary_check")
        last2 = committed[-2:]
        consumed2 = collections.Counter()
        for s in last2:
            for inc in s["increments"]:
                for n_, v in (f.increments[inc].get("species_boxes") or {}).items():
                    consumed2[n_] += v
        deficit = sorted(n_ for n_ in consumed2 if per_species.get(n_, 0) < consumed2[n_])
        mrec = {str(n): {k: v.get(k) for k in ("exp", "pool", "state", "verdict", "rollback_recommended", "to_pool",
                                                 "compared_with", "perm_p", "species_failed")}
                for n, v in f.milestones.items()}
        pending_rb = [m for m in f.milestones.values() if m.get("rollback_recommended")
                      and not any(r.get("milestone") == m["exp"] for r in f.rollbacks)]
        rb = f.rollbacks[-1] if f.rollbacks else None
        x4 = None
        if rb:
            b = f.bisects.get(rb["n"], {"arms": {}, "decisions": {}})
            undecided = [i for i in rb["suspect"] if i not in b["decisions"]]
            sep = [i for i, v in b["decisions"].items() if v == G.HURTS]
            x4 = {"raised": bool(undecided) or (not sep and bool(b["decisions"])), "rollback": rb["n"],
                  "suspect": rb["suspect"], "bisect_built": bool(b["arms"]), "decisions": b["decisions"],
                  "unresolved": undecided,
                  "note": (None if sep or undecided else
                           "no single suspect increment hurts: the effect may need two increments together")}
        disp_counts = collections.Counter(i.get("disposition") for i in f.increments.values() if i.get("disposition"))
        src_blame = collections.Counter()
        for inc in f.increments.values():
            if inc.get("disposition") == DATA:
                for s_ in (inc.get("sources") or {}):
                    src_blame[s_] += 1
        pool = f.current_pool()
        return {
            "format": SUMMARY_FORMAT, "sid": self.sid, "stream_version": STREAM_VERSION,
            "generated_utc": _utc(self._now()), "testing": bool(f.defn.get("testing")),
            "M": M, "K_max": K, "arm": f.arm, "stage_b": f.stage_b, "chosen_recipe": f.chosen_recipe,
            "forked_to": f.forked_to,
            "pool": dict({k: pool.get(k) for k in ("name", "path", "sha256", "n_images", "research_only",
                                                    "research_only_rows", "parent")},
                         current=pool["name"], images=pool["n_images"]),
            "eligible": {"images": Q, "target_boxes": dict(per_species), "oldest_utc": oldest,
                         "oldest_age_days": age, "by_source": dict(per_source),
                         "uncuttable": dict(ana["uncuttable"]) if ana else None},
            # {hold: {"rows", "past_deadline"}}: the shape the autopilot's DHOLD reads (S26)
            "held": ({h: {"rows": int(n), "past_deadline": int(ana["held_past_deadline"].get(h, 0))}
                      for h, n in sorted(ana["held"].items())} if ana else {}),
            "consumed_last": dict(consumed2),
            "m_over_pool": M / float(pool["n_images"]) if pool["n_images"] else None,
            "queue": ({"path": str(ana["view"].queue_path), "sha256": ana["view"].sha256,
                       "events_sha256": ana["view"].events_sha256, "rows": len(ana["view"].rows),
                       "eligible_images": Q, "eligible_target_boxes": dict(per_species),
                       "eligible_by_source": dict(per_source), "oldest_eligible_utc": oldest,
                       "oldest_eligible_age_days": age, "not_eligible": ana["reasons"], "held": ana["held"],
                       "held_past_deadline": ana["held_past_deadline"], "other_kind": ana["other_kind"],
                       "uncuttable": ana["uncuttable"],
                       "hashes_checked": False, "rows_without_hash": ana["rows_without_hash"]}
                      if ana else {"error": q_err}),
            "cut": {"ready": bool(ready and not in_flight), "q_ge_km": Q >= K * M,
                    "q_ge_m_stale": bool(Q >= M and age is not None and age >= STALE_DAYS),
                    "k": max(0, min(K, Q // M)) if M else 0, "train_idle": not in_flight, "probe": probe,
                    "last_refusal": _read_json(self.p.cut_refusal)},
            "segments": segs,
            "in_flight": [s["exp"] for s in in_flight],
            "uncommitted_done": [s["exp"] for s in segs if s["state"] == "built" and s["done"]],
            "next_segment": segment_exp(self.sid, f.next_seg), "next_increment": inc_name(f.next_inc),
            "last_segment": ({"exp": last["exp"], "d30": last["commit"].get("d30"), "d33": last["commit"].get("d33"),
                              "dispositions": last["commit"].get("dispositions"),
                              "discordant": last["commit"].get("discordant")} if last else None),
            "dispositions": dict(disp_counts), "data_blamed_sources": dict(src_blame),
            "consumed_target_boxes_last2": dict(consumed2), "species_deficit": deficit,
            "milestones": {"records": mrec, "in_flight": (f.milestone_in_flight() or {}).get("exp"),
                           "last_good": (last_ms or {}).get("exp"), "accepted_since": acc_since,
                           "segments_since": len(since), "days_since_first_accepted": days, "due": bool(due),
                           "due_reasons": due, "next": milestone_exp(self.sid, f.next_ms)},
            "boundary_check": boundary,
            "rollback_pending": [{"milestone": m["exp"], "to_pool": m["to_pool"]} for m in pending_rb],
            "x4": x4, "suspect_increments": [i for i, r in f.increments.items() if r["status"] == "suspect"],
            "quarantined_sources": f.q_sources,
            "quarantined_images": len({x["key"] for x in f.quarantine}),
            "feasibility": ({k: f.feasibility.get(k) for k in ("exp", "phase", "result", "m", "holdout")}
                            if f.feasibility else None),
            "ledger": {"events": f.events, "head_sha256": f.head}, "code": f.code,
        }

    def _boundary(self):
        """§5.5.1: the newest segment whose base runs are complete against the
        previous segment's base runs (dev), 3 v 3; an alarm, not a claim."""
        f = self.fold
        segs = [s for s in f.segments.values() if s["state"] != "withdrawn"]
        for i in range(len(segs) - 1, 0, -1):
            new, old = segs[i], segs[i - 1]
            if new["base_pool"] == old["base_pool"]:
                continue
            try:
                nv, ov = self._base_dev(new["exp"]), self._base_dev(old["exp"])
            except StreamError:
                continue
            fires = _mean(nv) < _mean(ov) - BOUNDARY_SD_MULT * _sd(ov)
            return {"segment": new["exp"], "previous": old["exp"], "new_mean": _mean(nv), "old_mean": _mean(ov),
                    "old_sd": _sd(ov), "fires": bool(fires)}
        return None

    # --------------------------------------------------------------- verify
    def verify(self):
        """The ledger chain and every file it names (sha256)."""
        f = self.load()
        problems = []
        for name, rec in f.pools.items():
            for what, r in (("pool %s" % name, rec), ("pool %s dHash map" % name, rec["dhash"])):
                try:
                    _check_rec(r, what)
                except StreamError as e:
                    problems.append(str(e))
        for inc, rec in f.increments.items():
            for what in ("manifest", "rows_file", "meta"):
                try:
                    _check_rec(rec[what], "%s %s" % (inc, what))
                except StreamError as e:
                    problems.append(str(e))
        return {"events": f.events, "head_sha256": f.head, "problems": problems}


# --------------------------------------------------------------- helpers
def _deadline_ts(text):
    """The moment a hold deadline has passed: a full timestamp, or the end of
    a YYYY-MM-DD day (step1_stream: 'today > deadline')."""
    t = _ts(text)
    if t is not None:
        return t
    d = _date_ts(text)
    return None if d is None else d + 86400.0


def _date_ts(text):
    try:
        return float(datetime.datetime.strptime(str(text)[:10], "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc).timestamp())
    except (TypeError, ValueError):
        return None


def _max_share(units_ids, units):
    tot = [0] * C.OTHER_PLANT
    for u in units_ids:
        for i, v in enumerate(units[u]["species"]):
            tot[i] += v
    s = sum(tot)
    if not s:
        return None, None
    i = max(range(len(tot)), key=lambda j: (tot[j], -j))
    return tot[i] / float(s), species_names()[i]


def _row_licence(e):
    """(licence, research_only) of a cut row: the queue row's, unless a
    person's release lifted its licence hold. Then the release decides, fail
    closed: its licence text (--licence) when it records one, and
    research_only true unless it records both the person's licence text and
    --not-research-only; a row the queue marks research_only stays so."""
    q = e["row"]
    rel = e.get("licence_release")
    if rel is None:
        return q.get("licence"), q.get("research_only")
    ro = bool(q.get("research_only")) or not (rel.get("licence") and rel.get("research_only") is False)
    return rel.get("licence") or q.get("licence"), ro


def _research_only(pool, written):
    """§8: a model trained on any research_only image is research-only. The
    segment's cand, truth and final models train on the pool plus its
    increments; 'unknown' when the pool's licences are not all recorded."""
    ro = pool.get("research_only")
    inc_rows = sum(int(w.get("research_only_rows") or 0) for w in written)
    models = True if (ro is True or inc_rows) else ("unknown" if ro == "unknown" else False)
    return {"pool": ro, "pool_research_only_rows": pool.get("research_only_rows"),
            "increments_research_only_rows": inc_rows, "models": models}


def _driver_ledger(paths):
    """{entry id: entry} of an experiment's ledger (inc.report.read_ledger)."""
    from ..inc import report as R
    return R.read_ledger(paths)


def _truth_verdict(led, g3_truth, n):
    """Step n's truth verdict: inc2.gate3's truth3 commit verdict (the v3
    species guard between the arms), else the pinned ledger's; None when the
    step has no truth decision."""
    t = g3_truth.get(n)
    if t is not None and t.get("commit_verdict"):
        return t["commit_verdict"]
    te = led.get("truth/%d" % n)
    return te["detail"]["verdict"] if te else None


def code_hashes():
    """sha256 of the modules whose version decides a stream's outputs."""
    root = D.package_dir()
    mods = ["tools/inc2/%s" % n for n in sorted(os.listdir(root / "tools" / "inc2")) if n.endswith(".py")]
    mods += list(D.PINNED_MODULES) + ["tools/inc/select.py", "tools/inc/report.py", "tools/inc/scorer.py",
                                      "tools/near_dup.py"]
    out = {}
    for m in sorted(set(mods)):
        p = root / m
        out[m] = C.sha256_file(p) if p.is_file() else None
    return out


def only_stream():
    d = stream_root()
    sids = sorted(n for n in os.listdir(d) if (d / n / "ledger.jsonl").is_file()) if d.is_dir() else []
    live = [s for s in sids if not (_read_json(d / s / "state.json") or {}).get("forked_to")]
    if len(live) != 1:
        raise StreamError("name the stream with --stream (streams: %s)" % sids)
    return live[0]


def sid_of_exp(exp):
    m = re.match(r"^(.+)_([smcb]\d{3})$", str(exp))
    if not m:
        raise StreamError("%r is not a stream experiment name (<sid>_sNNN, _mNNN, _cNNN, _bNNN)" % exp)
    return m.group(1)


# -------------------------------------------------------------------- CLI
def build_parser():
    """The CLI's argparse parser (the autopilot's replay reads its L18-L28 argv back with it)."""
    ap = argparse.ArgumentParser(prog="python -m %s.inc2.stream" % PKG, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, stream=True, stream_required=True):
        p = sub.add_parser(name)
        if stream:
            p.add_argument("--stream", required=stream_required)
        p.add_argument("--decided-by", default="platform")
        p.add_argument("--quiet", action="store_true")
        return p

    p = add("init")
    p.add_argument("--base")
    p.add_argument("--lock")
    p.add_argument("--m", type=int)
    p.add_argument("--k-max", type=int, default=K_MAX)
    p.add_argument("--arm", default=DEFAULT_ARM)
    p.add_argument("--milestone0")
    p.add_argument("--stage-b")
    p.add_argument("--step1-dir")
    p.add_argument("--testing", action="store_true")
    p.add_argument("--testing-settings")
    p = add("choose-arm")
    p.add_argument("--capacity", help="inc2.baseline capacity-verdict's decision (default INC_DIR/capacity/"
                                      "capacity_v1.json)")
    add("cut")
    p = add("build")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--recipes")
    p.add_argument("--exp")
    p.add_argument("--no-truth", action="store_true")
    # the autopilot's L18 argv (stream_levers.json): its view of the arm, checked against the stream's own
    # (never a switch), and its truth cadence, which can only make the truth arm more frequent
    p.add_argument("--arch")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--truth-every", type=int)
    p = add("commit", stream=False)
    p.add_argument("--exp", required=True)
    p = add("milestone")
    p.add_argument("--force", action="store_true")
    p = add("compare", stream=False)
    p.add_argument("--exp", required=True)
    p = add("rollback")
    p.add_argument("--to", required=True)
    p = add("bisect")
    p.add_argument("--from", dest="from_pool", required=True)
    p = add("feasibility")
    p.add_argument("--holdout", default="tsw22")
    p.add_argument("--m", type=int)
    p.add_argument("--recipes", help="the Stage B candidates (r0 and the Stage A survivor), as L28 passes them")
    p = add("fork")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--to")
    p = add("quarantine", stream_required=False)
    p.add_argument("--source", required=True)
    p.add_argument("--cite", required=True)
    p = add("unquarantine")
    p.add_argument("--source", required=True)
    p = add("release")
    p.add_argument("--hold", required=True)
    p.add_argument("--source")
    p.add_argument("--keys")
    p.add_argument("--reason")
    p.add_argument("--licence", help="--hold licence: the licence text the person records for the released rows")
    p.add_argument("--not-research-only", action="store_true",
                   help="--hold licence with --licence: the released rows are not research_only (else they are)")
    p = add("withdraw", stream=False)
    p.add_argument("--exp", required=True)
    p.add_argument("--reason", required=True)
    add("summary")
    add("verify")
    add("status")
    return ap


def main(argv=None, deps=None, clock=time.time):
    a = build_parser().parse_args(argv)
    try:
        sid = getattr(a, "stream", None)
        if a.cmd in ("commit", "compare", "withdraw"):
            sid = sid_of_exp(a.exp)
        elif a.cmd == "quarantine" and not sid:
            sid = only_stream()
        st = Stream(sid, deps=deps, clock=clock, quiet=a.quiet)
        who = a.decided_by
        env_who = os.environ.get("INCAP_DECIDED_BY", "")
        if who == "platform" and _human(env_who):
            who = env_who                               # a person's approval the autopilot carries (remote.py)
        if a.cmd == "init":
            testing = a.testing
            if a.testing_settings:
                testing = json.loads(a.testing_settings)
            st.init(base=a.base, lock=a.lock, m=a.m, k_max=a.k_max, arm=a.arm, milestone0=a.milestone0,
                    stage_b=a.stage_b, testing=testing, decided_by=who, step1_dir=a.step1_dir)
        elif a.cmd == "choose-arm":
            st.choose_arm(a.capacity, decided_by=who)
        elif a.cmd == "cut":
            st.load()
            plans, refusal, ana = st.cut_plan(1, compute_hashes=True, guard=True)
            print(json.dumps({"would_cut": [{"images": len(p["rows"]), "first_source": p["first_source"],
                                              "sources": dict(collections.Counter(r["source"] for r in p["rows"])),
                                              "species_cap": p["species_cap"], "verifier": p["verifier"],
                                              "split_capture_groups": p["split_capture_groups_keys"]} for p in plans],
                              "refusal": refusal, "not_eligible": ana["reasons"], "held": ana["held"]},
                             indent=1, sort_keys=True, default=str))
            return EXIT_OK if plans else EXIT_REFUSED
        elif a.cmd == "build":
            st.build(a.k, recipes=a.recipes.split(",") if a.recipes else None, exp=a.exp, no_truth=a.no_truth,
                     decided_by=who, arch=a.arch, imgsz=a.imgsz, truth_every=a.truth_every)
        elif a.cmd == "commit":
            st.commit(a.exp)
        elif a.cmd == "milestone":
            st.milestone(force=a.force, decided_by=who)
        elif a.cmd == "compare":
            st.compare(a.exp)
        elif a.cmd == "rollback":
            st.rollback(a.to, decided_by=who)
        elif a.cmd == "bisect":
            st.bisect(a.from_pool, decided_by=who)
        elif a.cmd == "feasibility":
            st.feasibility(holdout=a.holdout, m=a.m, decided_by=who, recipes=a.recipes)
        elif a.cmd == "fork":
            st.fork(a.m, to=a.to, decided_by=who)
        elif a.cmd == "quarantine":
            st.quarantine_source(a.source, a.cite, decided_by=who)
        elif a.cmd == "unquarantine":
            st.unquarantine_source(a.source, decided_by=who)
        elif a.cmd == "release":
            st.release(a.hold, a.reason, who, source=a.source, keys_file=a.keys, licence=a.licence,
                       research_only=not a.not_research_only)
        elif a.cmd == "withdraw":
            st.withdraw(a.exp, a.reason, decided_by=who)
        elif a.cmd == "summary":
            with st.writing():
                pass
            print(json.dumps({k: v for k, v in _read_json(st.p.summary).items() if k in ("cut", "queue", "milestones")},
                             indent=1, sort_keys=True, default=str))
        elif a.cmd == "verify":
            res = st.verify()
            print(json.dumps(res, indent=1, sort_keys=True))
            return EXIT_OK if not res["problems"] else EXIT_REFUSED
        elif a.cmd == "status":
            st.load()
            print(json.dumps(st.state_record(), indent=1, sort_keys=True, default=str)[:20000])
    except LeaseBusy as e:
        print("[inc2.stream] BUSY: %s" % e, file=sys.stderr)
        return EXIT_BUSY
    except (StreamError, D.DriverError, OSError, ValueError, KeyError) as e:
        print("[inc2.stream] ERROR: %s" % e, file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
