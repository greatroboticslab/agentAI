"""Incremental Step 1 with per-box admission (docs/CONTINUOUS_LOOP.md §3.3,
§3.9, §6.7, §8; decisions D-B, P8, P9, L-5).

    python -m weed_optimizer_framework.tools.inc2.step1_stream bootstrap [--lock PATH]
    python -m weed_optimizer_framework.tools.inc2.step1_stream admit --intake BATCH
    python -m weed_optimizer_framework.tools.inc2.step1_stream admit --registry [--slugs a,b]
    python -m weed_optimizer_framework.tools.inc2.step1_stream backfill            # batch b0000
    python -m weed_optimizer_framework.tools.inc2.step1_stream knowntruth [--sets tsw22,tsw23]
    python -m weed_optimizer_framework.tools.inc2.step1_stream rejoin --slug SLUG
    python -m weed_optimizer_framework.tools.inc2.step1_stream serve-holds [--hold h6_scan|licence|funnel_F9]
    python -m weed_optimizer_framework.tools.inc2.step1_stream scan-holds --hold KIND    # the autopilot's name
    python -m weed_optimizer_framework.tools.inc2.step1_stream status
    python -m weed_optimizer_framework.tools.inc2.step1_stream verify

Every output lives under INC_DIR/step1_stream/ (this module is its one writer;
a writing verb takes step1_stream/.lock). The pinned v1 Step 1 (inc/verify.py,
inc/select.py) cannot append: its paths are v1 constants, its pool refuses
3SeasonWeedDet10, any pool change voids its embeddings and verifier, and its
crop ids start at 0. This module uses their functions as a library, unchanged
(the class join, the key maker, label files, the probe of an image, the crop
table reader, the BioCLIP-2 embedder and _embed_images, the frozen Verifier,
verdicts and image_verdict, box matching and metrics; select's
species_reference, image_features, hierarchical_kmeans, dup_groups), and
funnel.leak's detector as a library for the embedding copy check.

Layout of INC_DIR/step1_stream/:
  STREAM.json              the pins (bootstrap; its sha256 is every queue row's
                           stream_pins_sha)
  verifier/v1/             the frozen v1 verifier, copied (probe, npz, thresholds,
                           fit_info) plus oof_keys.json: the (key, box) of the
                           OtherPlant crops it was trained on, judged out of fold
  reference_v1.npz         species prototypes, the leave-one-out percentile scale,
                           OtherPlant prototypes, l1/l2 centres
  canary.npz               64 train_core crops and their v1 features and verdicts
  index/                   seen (exact dHash), keys, groups (3-bit near-dup groups,
                           union-find), processed (per input), evidence (verified
                           target boxes per source), base_expert (6-bit index of
                           the expert-labelled base rows), overrides (rejoin)
  batches/bNNNN/           plan, ingest, crops.csv, emb.npz, verdicts.npz,
                           admission.jsonl, queue_rows.jsonl, human.jsonl,
                           events.jsonl, index_update.json, batch.json (last)
  labels/<source>/         INC labels, named by content (<key>.<sha16>.txt)
  images_masked/<source>/  masked copies (inc2.mask.mask_except)
  ledger/batches.jsonl     append-only, hash-chained (prev_sha256 of the file)
  queue/queue.jsonl        append-only queue rows (format below)
  queue/events.jsonl       append-only events: hold releases and refusals,
                           added holds, near-dup group merges, supersessions
  human/queue.jsonl        append-only human queue (refused overlaps, conflicts)
  leak/                    the evaluation descriptor cache of the copy detector
  status.json              read by the autopilot

A batch's stages each resume: a killed job reruns the same verb and continues
from the last finished stage; a committed batch is a no-op. Crop ids are local
to a batch (0..n-1, so verify.Crops reads them) plus a global offset recorded
at commit: v1 holds [0, n_v1), and each committed batch takes the next range,
so global ids are contiguous and disjoint.

The admission (stage 6) is inc2.mask.decide (D-B); --rule image restores the
v1 image rule (whole or not admitted), which is how the equivalence test
compares this module with verify pool + crops + admit.

D28-v2 (docs/CONTINUOUS_LOOP.md, amendment 2026-10-03): a row GuardV2 refuses
as a dHash copy of an evaluation image is also weighed by its pair cosine
with the evaluation image it matched, from the copy scanner's own descriptors
(score_eval_hits, inc2.eval_hits). The row keeps "pair_cos", batch.json the
batch's record ("eval_hits"), and status.json folds it per source
(eval_hits_scored, eval_hit_pair_cos, eval_hit_copy_threshold) for the
autopilot's D28.

Holds (a row is eligible for a cut only with none left; queue/events.jsonl
releases them): h6_scan (the embedding copy scan has not cleared the row: a
source that is not provenance-cleared and was not scanned, or a registry slug
the v1 pool caught copying (lab_evidence, P9)), licence (no licence recorded),
funnel_F9 (the b0000 masked rows, P8), join_conflict (an exact duplicate whose
labels disagree with its twin: both are held and a rejoin is queued).

Nothing here names a domain's species; class ids are the INC class space.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..inc import common as C
from ..inc import select as S
from ..inc import verify as V
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NEAR_DUP_BITS, NearHashIndex
from . import eval_hits as EH
from . import mask as MK

# ------------------------------------------------------------------ constants
FORMAT = "inc2-step1-stream/1"
STATUS_FORMAT = "inc2-step1-stream-status/1"
QUEUE_FORMAT = "inc2-stream-queue/1"
STREAM_VERSION = 1
VERIFIER_VERSION = "v1"
REFERENCE_VERSION = "v1"
CONTRACT_EMBEDDER = "hf-hub:imageomics/bioclip-2+fp16"
CONTRACT_DIM = 768
CONTRACT_VERIFIER_CROPS_SHA = "d8188b16fad47c34b6a771fb806ab2c1b607d6b99126fc134ada34f299164485"
CONTRACT_V1_CROPS = 564686          # recorded beside the value read from crops_info.json, not asserted
CONTRACT_VETO_IMAGES = 457          # the contract's post-hoc count; recorded, not asserted (census_v1 decides)
BATCH_CAP = 50000                   # images per batch (contract §3.3: about 23 min of embedding)
CHUNK_IMAGES = 2000                 # embedding resumes per chunk of this many images
CANARY_N = 64
CANARY_MIN_COS = 0.999
CANARY_SEED = "inc2/step1_stream/canary"
REFERENCE_SEED = 0                  # select.build's seed, so the l1/l2 centres are base B's clusters
OTHER_PROTO_SAMPLE = 25000          # select.typicality's per-fold sample
OTHER_PROTO_K_MAX = 64
GROUP_BITS = NEAR_DUP_BITS          # near-duplicate groups: unmasked dHash at 3 bits
NEAR_CONSUMED_BITS = NEAR_DUP_BITS  # the near_consumed guard
BASE_COPY_BITS = HOLDOUT_NEAR_DUP_BITS
HOLD_DEADLINE_DAYS = 21             # §6.7: holds that wait on the funnel carry a deadline
HOLD_ORDER = ("h6_scan", "licence", "join_conflict", "funnel_F9")
DEADLINE_HOLDS = ("h6_scan", "funnel_F9")
EXPERT_SETS = ("train_core", "tsw22", "tsw23")
EVAL_LABS = ("LuLab",)              # the lab of dev and test: a same-lab source waits for the H6 scan (P9)
L5_DROPPED_SOURCES = ("rf_karthikeya-c8pvy__weed-detection-cwp10", "rf_zig-zag-lnodr__weed-detection-vanpe")
V2_EVAL_SPLITS = ("dev", "test", "imageweeds")
WILSON_Z = 1.96
NEVER_TRAIN_REASONS = ("near_eval_v2", "near_eval_variant", "near_eval_embed")
REFIT_PRECISION_LB = 0.99           # proposed trigger (§3.3), for card X11; thresholds.json pre-registers it
REFIT_MIN_MATCHED_VERIFIED = 30
REFIT_SPECIES_NEW_BOXES = 100
REFIT_SPECIES_UNKNOWN_SHARE = 0.5
FUNNEL_DOMAIN_DEV = Path("step1_r1") / "domain_dev.jsonl"
NC_TOKENS = ("nc", "noncommercial", "non-commercial", "non commercial")
UNRESOLVED_LICENCES = ("", "unresolved", "unknown", "none", "null")
PINNED_MODULES = ("tools/inc/verify.py", "tools/inc/select.py", "tools/funnel/recover.py", "tools/funnel/leak.py",
                  "tools/cwd12_species.py", "tools/near_dup.py", "tools/semisup_labeler.py", "tools/mega_trainer.py")
OWN_MODULES = ("tools/inc2/step1_stream.py", "tools/inc2/mask.py")
PKG_DIR = Path(__file__).resolve().parents[2]     # weed_optimizer_framework/

# box verdicts: verify's six plus the unmapped id (13), which is never embedded
VERDICT_CODES = tuple(V.VERDICT_CODES) + (MK.UNMAPPED,)
KIND_REGISTRY, KIND_INTAKE, KIND_BACKFILL, KIND_KNOWNTRUTH, KIND_REJOIN = (
    "registry", "intake", "backfill", "knowntruth", "rejoin")
QUEUE_KEYS = ("key", "batch", "source", "group", "capture_group", "l1", "l2", "kind", "score", "species_boxes",
              "other_boxes", "admission", "n_masked", "evidenced", "lab_group", "licence", "research_only", "prior",
              "hold_until", "holds", "hold_deadline", "verifier", "stream_pins_sha",
              # the manifest fields the cutter writes (inc.common.MANIFEST_KEYS), plus provenance
              "image", "label", "sha256", "label_sha256", "session", "unmasked_image", "unmasked_sha256",
              "dhash", "dhash_masked", "masked_area_frac", "input", "reference", "typicality", "supersedes",
              "admitted_utc", "scanned", "h6_reason", "format")


class StreamError(RuntimeError):
    """A condition under which a step1_stream verb must not write its outputs."""


def log(msg):
    print("[inc2.step1_stream %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


CLOCK = _utcnow                     # tests replace it to move time forward


def _utc():
    return CLOCK().strftime("%Y-%m-%dT%H:%M:%SZ")


def _date(offset_days=0):
    return (CLOCK() + datetime.timedelta(days=offset_days)).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------- io
def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    return C.sha256_file(path)


def _write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return _sha_bytes(data)


def _write_json(path, obj):
    return _write_bytes(path, (json.dumps(obj, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8"))


def _read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as e:
        raise StreamError("unreadable JSON %s (%s)" % (path, e))


def _write_jsonl(path, rows):
    return _write_bytes(path, "".join(_canon(r) + "\n" for r in rows).encode("utf-8"))


def _read_jsonl(path):
    """The complete lines of a JSON-lines file; a last line without its
    newline is an append still in progress (or killed) and is not read."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
    rows = []
    for i, line in enumerate(data.decode("utf-8").split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            raise StreamError("%s line %d is not JSON" % (path, i))
    return rows


def _repair_tail(path):
    """Drop a trailing partial line (an append a killed job did not finish)."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with open(path, "r+b") as fh:
            fh.truncate(cut)
        log("WARNING: %s ended in a partial line (%d bytes); it was dropped" % (path, len(data) - cut))


def _append_jsonl(path, rows):
    """Append rows in one write (after repairing a partial last line)."""
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    with open(path, "ab") as fh:
        fh.write("".join(_canon(r) + "\n" for r in rows).encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())


def ledger_append(path, entry):
    """Append one hash-chained line: prev_sha256 is the sha256 of the file's
    bytes before this line (of b"" for the first line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    prev = _sha_bytes(path.read_bytes()) if path.exists() else _sha_bytes(b"")
    entry = dict(entry, prev_sha256=prev)
    with open(path, "ab") as fh:
        fh.write((_canon(entry) + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    return entry


def verify_ledger(path):
    """The ledger's entries after checking the chain; StreamError on a break."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    out, pos = [], 0
    for n, line in enumerate(data.split(b"\n")[:-1], 1):
        want = _sha_bytes(data[:pos])
        try:
            e = json.loads(line)
        except ValueError:
            raise StreamError("%s line %d is not JSON: the ledger is broken" % (path, n))
        if e.get("prev_sha256") != want:
            raise StreamError("%s line %d: prev_sha256 %s does not match the file before it (%s): "
                              "the ledger was edited" % (path, n, str(e.get("prev_sha256"))[:12], want[:12]))
        out.append(e)
        pos += len(line) + 1
    if pos != len(data):
        raise StreamError("%s ends in a partial line" % path)
    return out


def _module_hashes():
    out = {}
    for m in PINNED_MODULES + OWN_MODULES + ("tools/inc2/guard.py", "tools/inc2/common.py",
                                             "tools/inc2/eval_hits.py"):
        p = PKG_DIR / m
        out[m] = _sha_file(p) if p.is_file() else None
    return out


def wilson_lb(k, n, z=WILSON_Z):
    if not n:
        return None
    p = k / float(n)
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    r = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return round((c - r) / d, 6)


# ------------------------------------------------------------------- layout
class Layout:
    """Every path of one step1_stream state directory."""

    def __init__(self, root=None, inc_dir=None):
        self.inc_dir = Path(inc_dir or C.INC_DIR)
        self.root = Path(root or self.inc_dir / "step1_stream")
        r = self.root
        self.stream_json = r / "STREAM.json"
        self.verifier_dir = r / "verifier" / VERIFIER_VERSION
        self.reference = r / ("reference_%s.npz" % REFERENCE_VERSION)
        self.canary = r / "canary.npz"
        self.index_dir = r / "index"
        self.batches = r / "batches"
        self.ledger = r / "ledger" / "batches.jsonl"
        self.queue = r / "queue" / "queue.jsonl"
        self.events = r / "queue" / "events.jsonl"
        self.human = r / "human" / "queue.jsonl"
        self.status = r / "status.json"
        self.lock = r / ".lock"
        self.labels = r / "labels"
        self.masked = r / "images_masked"
        self.leak_dir = r / "leak"
        self.cache = r / "cache"

    def batch_dir(self, bid):
        return self.batches / bid

    def index(self, name):
        return self.index_dir / ("%s.json" % name)


OWNER_STALE_SECONDS = 9 * 3600      # longer than the job walltime (8 h): an older holder was ended by Slurm
OWNER_WRITE_GRACE_SECONDS = 60      # an owner file still being written is not taken for a dead one


def _owner_path(layout):
    return layout.root / ".lock.owner"


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError, TypeError):
        return True
    return True


def _slurm_job_alive(job):
    """True / False from squeue for a job id; None when squeue cannot tell
    (not installed, timed out, or an answer this function does not know)."""
    import shutil
    import subprocess
    exe = shutil.which("squeue")
    if not exe or not re.fullmatch(r"[0-9]+(_[0-9]+)?", str(job)):
        return None
    try:
        p = subprocess.run([exe, "-h", "-j", str(job), "-o", "%T"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode == 0:
        states = [s.strip() for s in p.stdout.splitlines() if s.strip()]
        return bool(states) and any(s in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED")
                                    for s in states)
    if "invalid job id" in (p.stderr + p.stdout).lower():
        return False
    return None


def _owner_stale(path, old):
    """None while the owner file names a live writer, else why it is stale."""
    import socket
    if not isinstance(old, dict):
        try:
            age = time.time() - os.stat(path).st_mtime
        except FileNotFoundError:
            return "gone"
        return None if age < OWNER_WRITE_GRACE_SECONDS else "unreadable and %.0f s old" % age
    if old.get("host") == socket.gethostname() and not _pid_alive(old.get("pid")):
        return "its process %s on this host has exited" % old.get("pid")
    if old.get("job") and _slurm_job_alive(old["job"]) is False:
        return "its Slurm job %s is no longer running" % old["job"]
    age = time.time() - float(old.get("t") or 0)
    if age > OWNER_STALE_SECONDS:
        return "%.1f h old, longer than any step1_stream job runs" % (age / 3600.0)
    return None


def _take_owner(path):
    """Create the owner file with O_CREAT|O_EXCL (atomic on Lustre and NFS,
    whatever the mount's flock mode); a stale one is taken over (logged).
    Returns this writer's token; StreamError while another writer lives."""
    import socket
    doc = {"token": "%s:%d:%s" % (socket.gethostname(), os.getpid(), os.urandom(8).hex()),
           "host": socket.gethostname(), "pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID"),
           "t": time.time(), "utc": _utc()}
    for _attempt in range(3):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                old = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                old = None
            why = _owner_stale(path, old)
            if why is None:
                raise StreamError("%s names a live writer (%s); step1_stream has one writer at a time"
                                  % (path, {k: (old or {}).get(k) for k in ("host", "pid", "job", "utc")}))
            log("WARNING: taking over %s: %s" % (path, why))
            with contextlib.suppress(FileNotFoundError):
                os.unlink(str(path))
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(doc, sort_keys=True))
            fh.flush()
            os.fsync(fh.fileno())
        return doc["token"]
    raise StreamError("%s could not be taken" % path)


def _drop_owner(path, token):
    try:
        if json.loads(Path(path).read_text()).get("token") == token:
            os.unlink(str(path))
    except (OSError, ValueError):
        pass


@contextlib.contextmanager
def writer_lock(layout):
    """One writer at a time over step1_stream/: a non-blocking fcntl.flock on
    .lock (it excludes writers on one node) and the owner file .lock.owner,
    created with O_EXCL (it excludes writers across nodes: on Lustre, flock is
    coherent across nodes only on a mount with the 'flock' option, and a mount
    without flock support answers ENOLCK or ENOSYS, which is tolerated with a
    warning). The owner file of a dead writer (its pid on this host gone, or
    older than any job runs) is taken over."""
    import errno
    import fcntl
    layout.root.mkdir(parents=True, exist_ok=True)
    fh = open(layout.lock, "a+")
    token = None
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise StreamError("%s is held by another writer; step1_stream has one writer at a time" % layout.lock)
            if e.errno not in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL,
                               getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)):
                raise StreamError("fcntl.flock on %s failed (%s)" % (layout.lock, e))
            log("WARNING: flock is unavailable on this mount (%s): the owner file alone keeps one writer"
                % errno.errorcode.get(e.errno, e.errno))
        token = _take_owner(_owner_path(layout))
        fh.seek(0)
        fh.truncate()
        fh.write("%d %s\n" % (os.getpid(), _utc()))
        fh.flush()
        yield
    finally:
        if token is not None:
            _drop_owner(_owner_path(layout), token)
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


# ------------------------------------------------------------------ indexes
class Index:
    """A JSON index file with the ids of the batches already applied to it, so
    a commit that is re-run after a kill applies each batch once."""

    def __init__(self, layout, name, default):
        self.path = layout.index(name)
        doc = _read_json(self.path, None)
        self.doc = doc if doc is not None else {"applied": [], "data": default}

    @property
    def data(self):
        return self.doc["data"]

    def applied(self, bid):
        return bid in self.doc["applied"]

    def save(self, bid=None):
        if bid is not None and bid not in self.doc["applied"]:
            self.doc["applied"].append(bid)
        _write_json(self.path, self.doc)


def _classes(boxes):
    return sorted(int(b[0]) for b in boxes)


class Groups:
    """Near-duplicate groups over queued images (unmasked dHash, GROUP_BITS):
    a union-find persisted as {dhash: gid} plus {gid: parent}. A new image that
    reaches two groups merges them (the smaller id wins, recorded as an event)."""

    def __init__(self, data):
        self.hash_gid = {int(h): int(g) for h, g in (data.get("hash_gid") or {}).items()}
        self.parent = {int(g): int(p) for g, p in (data.get("parent") or {}).items()}
        self.next = int(data.get("next", 0))
        self.index = NearHashIndex()
        for h in self.hash_gid:
            self.index.add(h, h, max_bits=GROUP_BITS)

    def find(self, g):
        g = int(g)
        root = g
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        return root

    def assign(self, h):
        """(gid, merged gids) for a new hash, added to the index."""
        h = int(h)
        if h in self.hash_gid:
            return self.find(self.hash_gid[h]), []
        roots = sorted({self.find(self.hash_gid[o]) for o, _b in self.index.matches(h)})
        if roots:
            g = roots[0]
            for r in roots[1:]:
                self.parent[r] = g
        else:
            g = self.next
            self.next += 1
            self.parent[g] = g
        self.hash_gid[h] = g
        self.index.add(h, h, max_bits=GROUP_BITS)
        return g, roots[1:]

    def to_data(self):
        return {"hash_gid": {str(h): g for h, g in sorted(self.hash_gid.items())},
                "parent": {str(g): p for g, p in sorted(self.parent.items())}, "next": self.next}


# ------------------------------------------------------------------- guards
class Guards:
    """GuardV2 (inc2.guard, group A) plus the checks it leaves to Step 1.

    check(path) runs, first hit wins: unhashable (no dHash, or its variants
    cannot be computed), then GuardV2.check(dhash, variants) (never-train v2,
    its flip and rotation variants, base copies, and whatever else GuardV2
    holds), and returns (reason | None, match, dhash). Any error inside
    GuardV2 is a refusal of the whole batch (fail closed), never a pass.
    near_consumed and exact_dup need the stream's own state and live in the
    batch code."""

    def __init__(self, guard, variants_fn, hash_fn=None, record=None):
        self.guard = guard
        self.variants_fn = variants_fn
        self.hash_fn = hash_fn or C.dhash
        self.record = record or {}

    def check(self, path, h=None):
        if h is None:
            h = self.hash_fn(path)
        if h is None:
            return "unhashable", None, None
        return self._decide(path, h, _variants_task((self.variants_fn, path)))

    def check_many(self, items, procs=1, what="guard"):
        """[(reason, match, dhash)] for items [(path, dhash or None)], in
        order; the variants are computed in worker processes (the images are
        read there), the decisions here."""
        items = list(items)
        hs = [h if h is not None else self.hash_fn(p) for p, h in items]
        todo = [(self.variants_fn, p) for (p, _h), h in zip(items, hs) if h is not None]
        with V._Workers(procs) as w:
            got = iter(list(w.imap(_variants_task, todo, chunksize=16)))
        out = []
        for n, ((p, _h), h) in enumerate(zip(items, hs), 1):
            out.append(("unhashable", None, None) if h is None else self._decide(p, h, next(got)))
            if n % 20000 == 0:
                log("  %s: %d/%d images checked" % (what, n, len(items)))
        return out

    def _decide(self, path, h, variants):
        if variants is None:
            return "unhashable", None, h
        try:
            reason, match = self.guard.check(int(h), variants)
        except Exception as e:
            raise StreamError("GuardV2.check failed on %s (%s: %s): the stream refuses rather than pass it"
                              % (path, type(e).__name__, e))
        return (str(reason) if reason else None), _jsonable(match), int(h)


def _variants_task(task):
    """Worker: the variant dHashes of one image, or None when they cannot be computed."""
    fn, path = task
    try:
        return fn(path)
    except Exception:
        return None


def _jsonable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return str(x)


def load_guards(lock_path):
    """Guards over inc2.guard.GuardV2.load(lock_path). Refuses (fail closed)
    when inc2.guard cannot be imported or the lock cannot be loaded."""
    try:
        from . import guard as G
    except ImportError as e:
        raise StreamError("inc2.guard is not importable (%s): step1_stream refuses to judge anything without "
                          "GuardV2" % e)
    if not Path(lock_path).is_file():
        raise StreamError("the splits v2 LOCK %s does not exist; run inc2.splits lock first" % lock_path)
    try:
        guard = G.GuardV2.load(lock_path)
    except Exception as e:
        raise StreamError("GuardV2.load(%s) failed (%s: %s)" % (lock_path, type(e).__name__, e))
    return Guards(guard, G.dhash_variants, record={"lock": str(lock_path), "lock_sha256": _sha_file(lock_path)})


# ------------------------------------------------------------ splits v2 lock
def lock_v2_default(inc_dir):
    return Path(inc_dir) / "splits" / "v2" / "LOCK.json"


def read_lock_v2(lock_path):
    """(lock doc, {name: path}) with the never-train and base-copy indexes and
    the manifests it names, each re-hashed against the lock."""
    lock_path = Path(lock_path)
    lock = _read_json(lock_path, None)
    if not isinstance(lock, dict):
        raise StreamError("the splits v2 LOCK %s is missing or unreadable" % lock_path)
    if str(lock.get("splits_version")) != "v2":
        raise StreamError("%s is not a v2 lock (splits_version %r)" % (lock_path, lock.get("splits_version")))
    d = lock_path.parent
    files = {"nevertrain": d / "nevertrain_dhash.json", "base_copies": d / "base_copies_dhash.json"}
    for name, key in (("nevertrain", "nevertrain_sha256"), ("base_copies", "base_copies_sha256")):
        want = lock.get(key)
        if not want or not files[name].is_file():
            raise StreamError("%s: %s is missing or the lock records no %s" % (lock_path, files[name].name, key))
        got = _sha_file(files[name])
        if got != want:
            raise StreamError("%s changed since the v2 lock (%s != %s)" % (files[name], got[:12], want[:12]))
    for name, want in sorted((lock.get("manifests") or {}).items()):
        p = d / ("%s.jsonl" % name)
        files["manifest:" + name] = p
        if p.is_file() and _sha_file(p) != want:
            raise StreamError("manifest %s changed since the v2 lock" % p)
    return lock, files


# ------------------------------------------------------- licences and labs
def licence_table(inc_dir, domain_path=None):
    """{source: licence text} from the funnel's recorded cards (read only:
    funnel/cards/index.json through funnel.recover.fetched_licences) and the
    funnel domain config's sources.licences, which wins, as in recover."""
    from ..funnel import recover as R
    fdir = Path(inc_dir) / "funnel"
    out = dict(R.fetched_licences(fdir))
    dp = Path(domain_path) if domain_path else PKG_DIR / "tools" / "funnel" / "domains" / "weed.json"
    dom = _read_json(dp, {}) or {}
    out.update(((dom.get("sources") or {}).get("licences")) or {})
    return out, {"cards_index": _file_rec(fdir / "cards" / "index.json"), "domain_config": _file_rec(dp)}


def _file_rec(p):
    p = Path(p)
    return {"path": str(p), "sha256": _sha_file(p) if p.is_file() else None}


def licence_id(text):
    """The licence family a licence text names, or None when it names none
    that is known to allow research use: fail closed. The provenance suffix
    funnel.recover.fetched_licences appends (" (fetched by L11a from <url>,
    <date>)") is cut first, so a word of the card URL is never read as a
    licence. "unknown", "Other (specified in description)", "Private", "All
    rights reserved" and anything unrecognised give None (held, P6)."""
    t = str(text or "").strip()
    t = re.sub(r"\s*\(fetched by L11a from .*$", "", t, flags=re.S | re.I).strip().lower()
    if t in UNRESOLVED_LICENCES:
        return None
    if re.search(r"\b(other|unknown|private|proprietary|all rights reserved|not specified|see description|"
                 r"noassertion|custom)\b", t):
        return None
    words = set(re.findall(r"[a-z0-9]+", t.replace("_", "-")))
    if "creativecommons.org/publicdomain" in t or "cc0" in words or "public domain" in t or "pddl" in words:
        return "cc0"
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)", t)
    if m or t.startswith("cc-by") or ("cc" in words and "by" in words) or "attribution" in words \
            or "creative commons" in t:
        kind = m.group(1) if m else t
        flags = set(re.findall(r"[a-z]+", kind.replace("-", " ")))
        nc = "nc" in flags or "noncommercial" in words or "non-commercial" in t or "non commercial" in t
        nd = "nd" in flags or "noderivatives" in words or "noderivs" in words or "no derivatives" in t
        sa = "sa" in flags or "sharealike" in words or "share-alike" in t or "share alike" in t
        return "cc-by" + ("-nc" if nc else "") + ("-sa" if sa else "") + ("-nd" if nd else "")
    for fam, pat in (("odbl", r"\bodbl\b|open database"), ("odc-by", r"\bodc[- ]by\b"), ("cdla", r"\bcdla\b"),
                     ("mit", r"\bmit\b"), ("apache", r"\bapache\b"), ("bsd", r"\bbsd\b"), ("gpl", r"\b[al]?gpl"),
                     ("unlicense", r"\bunlicense\b")):
        if re.search(pat, t):
            return fam
    return None


def licence_state(text):
    """(licence or None, research_only): None when unresolved (held, P6): a
    text that names no known licence family (licence_id) is unresolved,
    whatever it says. A non-commercial or no-derivatives licence is
    research-only (P6, as the collector's policy reads them)."""
    lid = licence_id(text)
    if lid is None:
        return None, None
    return str(text).strip(), ("-nc" in lid or "-nd" in lid)


def _restricted(text):
    """True when a licence text restricts use (non-commercial in any
    spelling, research, academic, educational, non-profit, personal or
    evaluation use): the collector's own rule (collect.licence.restricted),
    so a person's override reads as the collector reads it."""
    from ..collect import licence as CL
    return bool(CL.restricted(text))


def intake_licence_state(row, srec):
    """(licence or None, research_only) of an intake row. The collector's own
    verdict decides (collect/licence.py): a person's override recorded in the
    source's licence record resolves it; otherwise the row's licence_class
    must be permissive or research_only (unresolved or refused is held); a row
    without a class (an older intake) falls back to licence_state. An
    override is a licence text or the collector's record of a person's
    decision (a dict with the licence text as id); anything else, or a record
    without a licence text, refuses (StreamError). The override is
    research-only unless its record says false, when its text restricts use
    (collect.licence.restricted: non-commercial, research, academic ...),
    names a non-commercial or no-derivatives licence or names no known
    licence (licence_id None: fail closed)."""
    lrec = (srec or {}).get("licence") if isinstance((srec or {}).get("licence"), dict) else {}
    ro_row = bool(row.get("research_only"))
    ov = lrec.get("override")
    if ov:
        if not isinstance(ov, (str, dict)):
            raise StreamError("the licence override of source %s is %s, neither a licence text nor a person's "
                              "decision record: the row is not admitted (fail closed)"
                              % (row.get("source"), type(ov).__name__))
        text = ov if isinstance(ov, str) else (ov.get("id") or ov.get("licence") or ov.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise StreamError("the licence override of source %s records no licence text (id): the row is not "
                              "admitted (fail closed)" % row.get("source"))
        lid, ro = licence_state(text)
        ov_ro = ov.get("research_only") is not False if isinstance(ov, dict) else False
        return "%s (person override)" % text, bool(ro_row or ro or ov_ro or lid is None or lrec.get("research_only")
                                                   or _restricted(text))
    cls = row.get("licence_class", lrec.get("class"))
    if cls is not None:
        if cls not in ("permissive", "research_only") or not row.get("licence"):
            return None, None
        return str(row["licence"]), bool(ro_row or cls == "research_only")
    lic, ro = licence_state(row.get("licence"))
    return lic, (bool(ro_row or ro) if lic is not None else None)


def funnel_lab_groups(domain_path=None):
    dp = Path(domain_path) if domain_path else PKG_DIR / "tools" / "funnel" / "domains" / "weed.json"
    dom = _read_json(dp, {}) or {}
    out = {}
    for lab, members in sorted(((dom.get("sources") or {}).get("lab_groups") or {}).items()):
        for m in members or []:
            out.setdefault(str(m), str(lab))
    return out


def v1_lab_evidence(pool_summary):
    """{slug: {"eval_copies": n, "cwd12_copies": n, "near_eval_by_split": {...}}} of the v1 slugs the v1
    pool caught copying the evaluation splits or train_core (contract §3.3 Inputs)."""
    out = {}
    for s, st in sorted(((pool_summary or {}).get("per_slug") or {}).items()):
        ne = {k: int(v) for k, v in (st.get("near_eval_by_split") or {}).items()}
        cc = int(st.get("cwd12_copies", 0) or 0)
        if sum(ne.values()) or cc:
            out[s] = {"near_eval_by_split": ne, "cwd12_copies": cc,
                      "eval_copies": sum(v for k, v in ne.items() if k in V2_EVAL_SPLITS)}
    return out


# ------------------------------------------------------------- copy scanner
class CopyScanner:
    """The embedding copy detector (GuardV2's seventh check, near_eval_embed):
    funnel.leak.detect against dev + test + ImageWeeds, under a passed
    calibration. `own` is the stream's own calibration (INC_DIR/intake/leak),
    `funnel` the funnel's leak_v1.json (reused by sha256 when it passed).

    usable(lab_evidence, deadline_passed): a same-lab row (P9) is released by
    the funnel's calibration, or by the stream's own once its hold deadline
    passed (§6.7); any other row by either."""

    def __init__(self, index=None, own=None, funnel=None, rejected=None):
        self.index = index
        self.cal = {"own": own, "funnel": funnel}
        self.rejected = dict(rejected or {})     # {path: why} of calibrations that were found and not used

    @property
    def available(self):
        return self.index is not None and any(self.cal.values())

    def which(self, lab_evidence, deadline_passed):
        if self.index is None:
            return None
        if self.cal["funnel"] is not None:
            return "funnel"
        if self.cal["own"] is not None and (not lab_evidence or deadline_passed):
            return "own"
        return None

    def record(self):
        out = {k: (None if v is None else {"path": v["path"], "sha256": v["sha256"],
                                          "cos_threshold": v["threshold"]})
               for k, v in self.cal.items()}
        if self.rejected:
            out["rejected"] = dict(sorted(self.rejected.items()))
        return out

    def scan(self, items, which):
        """{key: hit or None} for items [{"key", "path"}]. An image the
        detector cannot describe is refused on its own, never cleared and
        never allowed to fail the whole batch: its hit is
        {"unscannable": true, ...} (reason unhashable_embed)."""
        from ..funnel import embed as E
        from ..funnel import leak as L
        if not items:
            return {}
        cal = self.cal[which]
        rows = [{"key": it["key"], "image": str(it["path"])} for it in items]
        res = E.embed_images(rows, None, self.index.embedder, prepare=L.prepare_hashed, n_hashes=len(L.VARIANTS))
        X = np.asarray(res["X"], dtype=np.float32)
        H = np.asarray(res["H"], dtype=np.uint64)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(X).all(axis=1) & np.asarray(res["hash_ok"], dtype=bool) \
                & (np.nan_to_num(np.linalg.norm(X, axis=1)) > 0)
        images = [{"key": r["key"], "path": r["image"], "desc": X[i], "hashes": [int(h) for h in H[i]]}
                  for i, r in enumerate(rows) if ok[i]]
        hits = L.detect(images, self.index, cal["record"]) if images else []
        out = {it["key"]: None for it in items}
        for i, r in enumerate(rows):
            if not ok[i]:
                out[r["key"]] = {"unscannable": True, "calibration": which,
                                 "why": "the copy detector cannot describe this image"}
        for h in hits:
            if out.get(h["key"]) is None:
                out[h["key"]] = {"eval_split": h["eval_split"], "eval_key": h["eval_key"], "cos": h["cos"],
                                 "bits": h["bits"], "variant": h["variant"], "calibration": which}
        return out


def scan_reason(hit, masked=False):
    """The refusal reason of a copy-scan hit."""
    r = "unhashable_embed" if (hit or {}).get("unscannable") else "near_eval_embed"
    return ("masked_" + r) if masked else r


V2_CAL_FORMAT = "inc2-embed-calibration/2"     # inc2.embed_calibration.FORMAT (EC.state's "format")


def copy_threshold(scanner):
    """(threshold, its calibration record) that D28-v2 compares a dHash hit's
    pair cosine with: the v2 calibration's cos_threshold (decision L-9(c),
    read through inc2.embed_calibration by load_scanner) when the scanner
    holds it, else the lowest threshold it holds (a per-pair threshold is
    lower, so never less strict); (None, None) without a calibration."""
    cals = [c for c in ((scanner.cal if scanner is not None else None) or {}).values() if c]
    if not cals:
        return None, None
    v2 = [c for c in cals if c.get("format") == V2_CAL_FORMAT]
    c = min(v2 or cals, key=lambda x: float(x["threshold"]))
    return float(c["threshold"]), {"path": c.get("path"), "sha256": c.get("sha256"), "format": c.get("format")}


def score_eval_hits(rows, scanner, procs=1, guards=None):
    """D28-v2's evidence for a batch (inc2.eval_hits; docs/CONTINUOUS_LOOP.md,
    amendment 2026-10-03): the pair cosine of every row GuardV2 refused as a
    dHash copy of an evaluation image (near_eval_v2, near_eval_variant), with
    the evaluation image its match names, from the copy scanner's own
    evaluation descriptors (its EvalIndex) and embedder: the scan's view of
    both images, on the scale of the calibration's threshold. Sets each such
    row's "pair_cos" (None and "pair_cos_why" when it cannot be scored) and
    returns the batch record D28 reads (eval_hits.record, kept in batch.json
    and folded per source into status.json). Without a scanner index nothing
    is scored and the record says why; D28 then reads those hits by its
    one-hit rule (fail closed). Never raises: the guard has already refused
    the rows, this only weighs them. With guards, each hit is weighed against
    the guard's match and every other evaluation image within the
    never-train radius of it (inc2.eval_hits.eval_matches), and keeps the
    highest pair cosine."""
    hits = [r for r in rows if r.get("decision") in EH.DHASH_HIT_REASONS]
    thr, cal = copy_threshold(scanner)
    index = getattr(scanner, "index", None) if scanner is not None else None
    name = getattr(getattr(index, "embedder", None), "name", None)
    items = []
    for r in hits:
        m = r.get("guard_match") if isinstance(r.get("guard_match"), dict) else {}
        also = None
        if guards is not None and index is not None:
            try:
                also = EH.eval_matches(guards, guards.variants_fn(r.get("image")))
            except Exception:  # noqa: BLE001 - without the list the guard's own match is weighed alone
                also = None
        items.append({"key": "%s|%s" % (r.get("input"), r.get("item")), "source": r.get("source"),
                      "image": r.get("image"), "split": m.get("split"), "eval_key": m.get("key"), "also": also})
    why, scored = None, {}
    if not items:
        pass
    elif index is None or getattr(index, "embedder", None) is None:
        why = "no copy scanner index (no passed calibration loaded): the hits are not weighed"
    else:
        pos = {(str(s), str(k)): j for j, (s, k) in enumerate(zip(index.split, index.eval_key))}

        def eval_desc(split, key):
            j = pos.get((split, key))
            return None if j is None else index.Xn[j]
        scored = EH.pair_cosines(items, index.embedder, eval_desc=eval_desc, procs=procs)
    for r, it in zip(hits, items):
        got = scored.get(it["key"]) or {}
        r["pair_cos"] = got.get("pair_cos")
        if r["pair_cos"] is None:
            r["pair_cos_why"] = got.get("why") or why
        elif (got.get("weighed") or 0) > 1:
            r["pair_cos_weighed"], r["pair_cos_best"] = got["weighed"], got.get("best")
    rec = EH.record(items, scored, embedder_name=name, copy_threshold=thr, calibration=cal, why=why)
    if items:
        log("  dHash hits on evaluation images: %d, %d with a pair cosine (max %s)%s"
            % (rec["hits"], rec["scored"], max((v["max_pair_cos"] for v in rec["per_source"].values()
                                                if v["max_pair_cos"] is not None), default="n/a"),
               "; %s" % rec["why"] if rec["why"] else ""))
    return rec


def calibration_state(path):
    """(passed calibration record or None, why not). "ok": true is not taken
    on trust: the record must also show it passed (inc2.guard's
    calibration_problems: a cosine threshold in [-1, 1], the never-train dHash
    radius, gates at least as strict as the pre-registered H6 ones, every
    augmentation family and negative set within them) and name the embedder
    its threshold belongs to. A detector whose threshold can never fire would
    release every h6_scan hold it is asked about."""
    p = Path(path)
    if not p.is_file():
        return None, "absent"
    try:
        doc = _read_json(p, None)
    except StreamError as e:
        return None, str(e)
    if not isinstance(doc, dict):
        return None, "not a JSON object"
    cal = doc.get("calibration")
    if not isinstance(cal, dict):
        return None, "no calibration record"
    if cal.get("ok") is not True:
        return None, "the calibration did not pass (%s)" % ("; ".join(cal.get("why") or []) or "ok is not true")
    try:
        from . import guard as G
    except ImportError as e:
        return None, "inc2.guard is not importable (%s): the record cannot be checked" % e
    probs = G.calibration_problems(cal)
    if probs:
        return None, "says ok, but its record does not show a passed calibration: %s" % "; ".join(probs[:5])
    emb = ((doc.get("detector") or {}).get("descriptor") or {}).get("embedder")
    if not emb:
        return None, "does not name the embedder its threshold belongs to"
    return {"path": str(p), "sha256": _sha_file(p), "record": cal, "threshold": float(cal["cos_threshold"]),
            "embedder": emb}, None


def load_calibration(path):
    """A passed calibration record (calibration_state), else None."""
    return calibration_state(path)[0]


def OWN_CALIBRATIONS(inc_dir):
    """Where the stream's own detector calibration may be: the one inc2.splits
    runs before lock (splits/v2/leak/leak_calibration.json, inc2.guard.calibrate)
    and the contract's intake/leak/leak_v1.json; the first that passed is used."""
    inc_dir = Path(inc_dir)
    return [inc_dir / "splits" / "v2" / "leak" / "leak_calibration.json", inc_dir / "intake" / "leak" / "leak_v1.json"]


def V2_CALIBRATION(inc_dir):
    """The v2 embedding calibration (decision L-9(c)) splits v2 writes beside
    LOCK v2: per-image false positives on hard same-domain negatives, never
    below its base's threshold."""
    return Path(inc_dir) / "splits" / "v2" / "embed_calibration_v2.json"


def v2_calibration_state(path, lock_path=None):
    """(record in calibration_state's form plus "role", or None, why) of the
    v2 calibration; with a LOCK v2 that exists, the file must hash to what it
    records (inc2.embed_calibration.state)."""
    try:
        from . import embed_calibration as EC
    except ImportError as e:
        return None, "inc2.embed_calibration is not importable (%s)" % e
    return EC.state(path, lock_path)


class _EvalAdapter:
    def __init__(self, rows):
        self._rows = rows

    def eval_rows(self):
        return {s: [dict(r) for r in rs] for s, rs in self._rows.items()}


def eval_rows_v1():
    """dev, test and ImageWeeds: the v1 manifests (the v2 evaluation splits are
    byte copies of them, §4.1)."""
    out = {}
    for s in V2_EVAL_SPLITS:
        p = C.manifest_path(s)
        if not p.is_file():
            raise StreamError("the evaluation manifest %s is missing" % p)
        out[s] = sorted(C.read_manifest(p), key=lambda r: r["key"])
    return out


def load_scanner(layout, own_path=None, funnel_path=None, embedder=None, eval_rows=None, procs=1, v2_path=None):
    """A CopyScanner from the calibrations on disk. Without a passed
    calibration it holds no index (rows needing the scan are then held).

    Decision L-9(c): when splits v2's embed_calibration_v2.json loads (and
    hashes as LOCK v2 records, once there is one), its threshold judges every
    row: it takes its base's role ("funnel" for the funnel's leak_v1.json,
    "own" for the stream's) and the role of any other calibration found, so
    a same-lab row is still released only as P9 says, but never at the
    per-pair threshold the v2 calibration replaces. A v2 file that does not
    load is recorded as rejected and the others are used as before."""
    from ..funnel import embed as E
    from ..funnel import leak as L
    own, rejected = None, {}
    for p in ([own_path] if own_path else OWN_CALIBRATIONS(layout.inc_dir)):
        own, why = calibration_state(p)
        if own is not None:
            break
        if why != "absent":
            rejected[str(p)] = why
    fp = funnel_path or layout.inc_dir / "funnel" / "leak_v1.json"
    fun, why = calibration_state(fp)
    if fun is None and why != "absent":
        rejected[str(fp)] = why
    v2p = Path(v2_path) if v2_path else V2_CALIBRATION(layout.inc_dir)
    v2, why = v2_calibration_state(v2p, Path(layout.inc_dir) / "splits" / "v2" / "LOCK.json")
    if v2 is not None:
        if v2["role"] == "funnel" or fun is not None:
            fun = dict(v2)
        if v2["role"] != "funnel" or own is not None:
            own = dict(v2)
    elif why != "absent":
        rejected[str(v2p)] = why
    for p, w in sorted(rejected.items()):
        log("WARNING: calibration %s is not used: %s (rows needing the copy scan stay held)" % (p, w))
    if own is None and fun is None:
        return CopyScanner(rejected=rejected)
    names = {c["embedder"] for c in (own, fun) if c is not None and c.get("embedder")}
    if embedder is None:
        if len(names) != 1:
            raise StreamError("the calibrations name %s descriptor embedders; the copy scan needs one"
                              % (sorted(names) or "no"))
        model, _, pooling = names.pop().rpartition(":")
        embedder = E.LazyEmbedder(model, pooling or "cls")
    else:
        bad = sorted(n for n in names if n != embedder.name)
        if bad:
            raise StreamError("the copy scan's embedder %s is not the calibrated one (%s)" % (embedder.name, bad))
    rows = eval_rows or eval_rows_v1()
    cache = layout.leak_dir / ("eval_desc_%s.npz" % _sha_bytes(embedder.name.encode())[:12])
    index = L.eval_index(_EvalAdapter(rows), embedder, cache, procs=procs)
    return CopyScanner(index, own, fun, rejected=rejected)


# ------------------------------------------------------------- v1 readers
def v1_files():
    """The v1 Step 1 files, read through verify's own constants (correct for v1)."""
    st = V.STEP1
    return {"pool": V.POOL, "pool_meta": V.POOL_META, "pool_summary": V.POOL_SUMMARY, "copies": V.COPIES,
            "crops": V.CROPS, "crops_skipped": V.CROPS_SKIPPED, "crops_info": V.CROPS_INFO,
            "verified": V.VERIFIED_MANIFEST, "admit_summary": V.ADMIT_SUMMARY, "pool_verdicts": V.POOL_VERDICTS,
            "verifier_dir": V.VERIFIER_DIR, "emb_dir": V.EMB_DIR, "cache_dir": V.CACHE_DIR,
            "increment_pool": st / S.POOL, "base_selected": st / S.BASE_SELECTED, "clusters": st / S.CLUSTERS,
            "select_summary": st / S.SUMMARY}


def _require(path, what):
    if not Path(path).exists():
        raise StreamError("%s is missing (%s)" % (path, what))


def _finite(X, idx):
    idx = np.asarray(idx, dtype=np.int64)
    if not len(idx):
        return idx
    return idx[np.isfinite(np.asarray(X[idx], dtype=np.float32)).all(axis=1)]


def _unit(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def load_v1_embeddings(crops, nshards=None):
    """(X, info) of the v1 shard set admit judged with."""
    adm = _read_json(V.ADMIT_SUMMARY, {}) or {}
    n = nshards or (adm.get("embeddings") or {}).get("nshards")
    try:
        return V.load_embeddings(crops, n)
    except V.VerifyError as e:
        raise StreamError("v1 embeddings: %s" % e)


# ------------------------------------------------------------ reference pack
def build_reference(crops, X, verified_keys, pool_verdicts=None, seed=REFERENCE_SEED, clusters_csv=None):
    """The frozen reference pack (contract §3.3 stage 8): select's species
    prototypes and leave-one-out percentile scale over train_core crops; the
    l1/l2 centres of select's two-level k-means over the admitted v1 images
    (seed 0, as select build, so they are base B's clusters); OtherPlant
    prototypes (k-means centres of other_ok pool crops). Returns (arrays, meta)."""
    lab = crops.label
    core_j = _finite(X, crops.where("core"))
    P, ref, _loo = S.species_reference(X, core_j, lab)
    keys = sorted(verified_keys)
    pidx = {k: i for i, k in enumerate(keys)}
    pool_all = crops.where("pool")
    img = np.fromiter((pidx.get(crops.key[i], -1) for i in pool_all), dtype=np.int64, count=len(pool_all))
    keep = img >= 0
    pj, pim = pool_all[keep].astype(np.int64), img[keep]
    fin = np.isfinite(np.asarray(X[pj], dtype=np.float32)).all(axis=1) if len(pj) else np.zeros(0, bool)
    F, has = S.image_features(len(keys), X, pj[fin], pim[fin])
    l1 = np.full(len(keys), -1, dtype=np.int64)
    l2 = np.full(len(keys), -1, dtype=np.int64)
    if has.any():
        a, b = S.hierarchical_kmeans(F[has], seed)
        l1[has], l2[has] = a, b
    k1 = int(l1.max()) + 1 if has.any() else 0
    c1 = np.zeros((k1, X.shape[1]), dtype=np.float32)
    for k in range(k1):
        c1[k] = _unit(F[l1 == k].mean(axis=0, keepdims=True))[0]
    pairs = sorted({(int(x), int(y)) for x, y in zip(l1[has], l2[has])})
    c2 = np.zeros((len(pairs), X.shape[1]), dtype=np.float32)
    for n, (x, y) in enumerate(pairs):
        c2[n] = _unit(F[(l1 == x) & (l2 == y)].mean(axis=0, keepdims=True))[0]
    other = np.zeros((0, X.shape[1]), dtype=np.float32)
    n_other = 0
    if pool_verdicts is not None:
        oj = pool_all[np.asarray(pool_verdicts)[pool_all] == V.VERDICT_CODES.index(V.OTHER_OK)]
        oj = _finite(X, oj)
        n_other = int(len(oj))
        if len(oj) > OTHER_PROTO_SAMPLE:
            oj = np.sort(np.random.default_rng(S.PROTO_SEED).choice(oj, OTHER_PROTO_SAMPLE, replace=False))
        if len(oj):
            k = int(min(OTHER_PROTO_K_MAX, max(1, round(math.sqrt(len(oj) / S.K1_PER)))))
            other = _unit(S._kmeans_centres(_unit(X[oj]), k, S.PROTO_SEED))
    agree = None
    if clusters_csv is not None and Path(clusters_csv).is_file():
        rec = {k: (int(r["l1"]), int(r["l2"])) for k, r in S.read_clusters(clusters_csv).items()}
        comp = [(rec.get(k), (int(l1[i]), int(l2[i]))) for i, k in enumerate(keys) if k in rec]
        agree = {"images": len(comp), "same_l1_l2": int(sum(a == b for a, b in comp))}
    arrays = {"P": np.asarray(P, dtype=np.float64), "ref": np.concatenate(ref) if ref else np.zeros(0),
              "ref_off": np.cumsum([0] + [len(r) for r in ref]).astype(np.int64), "c1": c1, "c2": c2,
              "c2_ids": np.array(pairs, dtype=np.int64).reshape(-1, 2), "other": np.asarray(other, np.float32)}
    meta = {"version": REFERENCE_VERSION, "seed": int(seed), "images": len(keys), "with_feature": int(has.sum()),
            "k1": k1, "sub_clusters": len(pairs), "other_ok_crops": n_other, "other_prototypes": int(len(other)),
            "train_core_crops": int(len(core_j)), "select_clusters_agreement": agree,
            "rule": "l1 = argmax cosine to the l1 centres; l2 = argmax cosine to that l1's sub-cluster centres; "
                    "an image without a box feature has l1 = l2 = -1"}
    return arrays, meta


class Reference:
    """The reference pack, loaded (placement and scores)."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.meta = json.loads(str(d["meta"]))
            self.P = d["P"].astype(np.float64)
            ref, off = d["ref"], d["ref_off"]
            self.ref = [ref[off[k]:off[k + 1]] for k in range(len(off) - 1)]
            self.c1, self.c2, self.c2_ids = d["c1"], d["c2"], d["c2_ids"]
            self.other = d["other"]
        self.sha256 = _sha_file(path)

    def place(self, F, has):
        n = len(F)
        l1 = np.full(n, -1, dtype=np.int64)
        l2 = np.full(n, -1, dtype=np.int64)
        idx = np.flatnonzero(has)
        if len(idx) and len(self.c1):
            a = np.argmax(_unit(F[idx]) @ self.c1.T, axis=1)
            l1[idx] = a
            for i, c in zip(idx, a):
                m = np.flatnonzero(self.c2_ids[:, 0] == c)
                if len(m):
                    l2[i] = int(self.c2_ids[m[int(np.argmax(self.c2[m] @ _unit(F[i:i + 1])[0]))], 1])
        return l1, l2

    def species_percentile(self, Xu, labels):
        labels = np.asarray(labels, dtype=np.int64)
        if not len(labels):
            return np.zeros(0)
        cos = (np.asarray(Xu, dtype=np.float64) * self.P[labels]).sum(axis=1)
        return S.percentile(self.ref, cos, labels)

    def typicality(self, Xu):
        if not len(self.other) or not len(Xu):
            return np.full(len(Xu), np.nan)
        return (np.asarray(Xu, dtype=np.float32) @ self.other.T).max(axis=1)


# ------------------------------------------------------------------ canary
class _RowCrops:
    """The part of verify.Crops that _embed_images reads (row, images)."""

    def __init__(self, rows):
        self.rows = rows

    def row(self, i):
        r = self.rows[i]
        return {"crop_id": int(i), "cx": float(r["cx"]), "cy": float(r["cy"]), "w": float(r["w"]),
                "h": float(r["h"]), "W": int(r["W"]), "H": int(r["H"])}

    def images(self):
        out, idx = [], {}
        for i, r in enumerate(self.rows):
            j = idx.get(r["image"])
            if j is None:
                idx[r["image"]] = len(out)
                out.append((r["image"], [i]))
            else:
                out[j][1].append(i)
        return out


def build_canary(crops, X, ver, n=CANARY_N):
    core = _finite(X, crops.where("core"))
    if not len(core):
        raise StreamError("no embedded train_core crop for the canary")
    pick = np.sort(np.random.default_rng(C.stable_int(CANARY_SEED)).choice(core, min(n, len(core)), replace=False))
    rows = [{"crop_id": int(i), "image": crops.image[i], "key": crops.key[i], "box": int(crops.box[i]),
             "cx": float(crops.cx[i]), "cy": float(crops.cy[i]), "w": float(crops.w[i]), "h": float(crops.h[i]),
             "W": int(crops.W[i]), "H": int(crops.H[i]), "label": int(crops.label[i])} for i in pick]
    Xc = np.asarray(X[pick], dtype=np.float16)
    v = ver.judge_features(np.array([r["label"] for r in rows]), Xc)[0]
    return {"X": Xc}, {"rows": rows, "verdicts": [str(x) for x in v], "seed_text": CANARY_SEED,
                       "min_cos": CANARY_MIN_COS}


def check_canary(path, embedder, ver, batch=V.BATCH):
    """The canary crops re-embedded now: every cosine to the v1 row >= the pin
    and identical verdicts. Returns the record; raises StreamError on drift."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        X0 = d["X"].astype(np.float32)
    rows = meta["rows"]
    mini = _RowCrops(rows)
    with V._Workers(1) as w:
        ids, X1, st = V._embed_images(mini.images(), mini, embedder, w, batch)
    X1 = X1[np.argsort(ids)].astype(np.float16)          # the precision every shard stores and is judged at
    if not np.isfinite(X1.astype(np.float32)).all():
        raise StreamError("canary drift: %d canary crop(s) could not be embedded now"
                          % int((~np.isfinite(X1).all(axis=1)).sum()))
    cos = (_unit(X0) * _unit(X1.astype(np.float32))).sum(axis=1)
    v1 = ver.judge_features(np.array([r["label"] for r in rows]), X1)[0]
    same = [a == b for a, b in zip(v1.tolist(), meta["verdicts"])]
    rec = {"n": len(rows), "min_cos": round(float(cos.min()), 6), "verdicts_same": int(sum(same)),
           "pin_min_cos": float(meta.get("min_cos", CANARY_MIN_COS))}
    if cos.min() < rec["pin_min_cos"] or not all(same):
        raise StreamError("canary drift: min cosine %.6f (pin %.3f), %d of %d verdicts unchanged: the embedder "
                          "or its environment changed; refusing before any write"
                          % (cos.min(), rec["pin_min_cos"], sum(same), len(rows)))
    return rec


# --------------------------------------------------------------- bootstrap
VERIFIER_FILES = ("probe.joblib", "verifier.npz", "thresholds.json", "fit_info.json")


def _v1_records():
    f = v1_files()
    out = {}
    for name in ("pool", "pool_meta", "pool_summary", "crops", "crops_skipped", "crops_info", "verified",
                 "admit_summary", "pool_verdicts"):
        out[name] = _file_rec(f[name])
    return out


def _probe_paths(paths, procs, what):
    """{path: (dhash, sha256, W, H)} through verify's own probe (no cache)."""
    with V._Workers(procs) as w:
        return V._probe_all([(p, p) for p in paths], w, {}, what)


def build_base_expert(files, procs=1):
    """{str(dhash): [set, key, label path]} over the expert-labelled base rows
    (train_core, tsw22, tsw23 of the v2 lock), and the per-set counts."""
    out, counts, missing = {}, {}, []
    for s in EXPERT_SETS:
        p = files.get("manifest:" + s)
        if p is None or not Path(p).is_file():
            missing.append(s)
            continue
        rows = sorted(C.read_manifest(p), key=lambda r: r["key"])
        probe = _probe_paths([r["image"] for r in rows], procs, "base expert %s" % s)
        n = 0
        for r in rows:
            h = probe[r["image"]][0]
            if h is None:
                raise StreamError("base image %s cannot be hashed; the known-truth index would miss its copies"
                                  % r["image"])
            out.setdefault(str(int(h)), [s, r["key"], r["label"]])
            n += 1
        counts[s] = n
    return out, {"per_set": counts, "missing_sets": missing}


def bootstrap(layout, lock_path=None, testing=False, seed_v1=True, procs=V.PROCS, force=False):
    """Pins, the frozen verifier copy, the reference pack, the canary, the
    base-expert index and the seen / key / processed / evidence indexes
    seeded from v1 (seed_v1=False starts them empty: the equivalence test's
    "ingest from empty"). Refuses to run twice (a new stream version is a new
    state directory)."""
    if layout.stream_json.exists() and not force:
        raise StreamError("%s exists: bootstrap runs once per stream version" % layout.stream_json)
    if force and layout.ledger.exists() and verify_ledger(layout.ledger):
        raise StreamError("batches are committed under %s; a new bootstrap needs a new stream version"
                          % layout.root)
    lock_path = Path(lock_path or lock_v2_default(layout.inc_dir))
    lock, files = read_lock_v2(lock_path)
    vf = v1_files()
    for name in ("pool", "pool_meta", "pool_summary", "crops", "crops_skipped", "crops_info", "verified",
                 "admit_summary", "pool_verdicts"):
        _require(vf[name], "v1 Step 1 output")
    for name in VERIFIER_FILES[:2]:
        _require(Path(vf["verifier_dir"]) / name, "v1 verifier")
    crops = V.Crops(vf["crops"])
    info = _read_json(vf["crops_info"], {})
    if info.get("crops_sha256") != crops.sha:
        raise StreamError("v1 crops_info.json does not describe crops.csv")
    ver0 = V.Verifier.load(vf["verifier_dir"])
    if ver0.meta.get("crops_sha256") != crops.sha:
        raise StreamError("the v1 verifier was fitted on another crops.csv")
    fit_emb = (ver0.info.get("embeddings") or {})
    if not testing and (fit_emb.get("embedder") != CONTRACT_EMBEDDER or int(fit_emb.get("dim") or 0) != CONTRACT_DIM):
        raise StreamError("the v1 verifier was fitted on %s (dim %s) features, not the contract's %s (%d); "
                          "pass --testing only for a synthetic world" % (fit_emb.get("embedder"), fit_emb.get("dim"),
                                                                        CONTRACT_EMBEDDER, CONTRACT_DIM))
    t0 = time.time()
    with writer_lock(layout):
        # 1. the frozen verifier
        layout.verifier_dir.mkdir(parents=True, exist_ok=True)
        vfiles = {}
        for name in VERIFIER_FILES:
            src = Path(vf["verifier_dir"]) / name
            if src.is_file():
                _write_bytes(layout.verifier_dir / name, src.read_bytes())
                vfiles[name] = _sha_file(layout.verifier_dir / name)
        other_ids = ver0.arrays.get("other_ids", np.zeros(0, np.int64)).astype(np.int64)
        oof_keys = [[crops.key[int(i)], int(crops.box[int(i)])] for i in other_ids]
        _write_json(layout.verifier_dir / "oof_keys.json", {"crops_sha256": crops.sha, "keys": oof_keys})
        vfiles["oof_keys.json"] = _sha_file(layout.verifier_dir / "oof_keys.json")
        ver = V.Verifier.load(layout.verifier_dir)
        # 2. reference pack and canary (v1 embeddings)
        X, emb = load_v1_embeddings(crops)
        pv, codes = S.read_pool_verdicts(vf["pool_verdicts"], crops)
        if list(codes) != list(V.VERDICT_CODES):
            raise StreamError("pool_verdicts.npz verdict codes %s are not verify's" % codes)
        verified = [r["key"] for r in C.read_manifest(vf["verified"])]
        arrays, rmeta = build_reference(crops, X, verified, pv, clusters_csv=vf["clusters"])
        V._save_npz(layout.reference, rmeta, **arrays)
        carr, cmeta = build_canary(crops, X, ver)
        V._save_npz(layout.canary, cmeta, **carr)
        del X
        # 3. indexes
        base_expert, be_info = build_base_expert(files, procs)
        idx = {"base_expert": base_expert, "seen": {}, "keys": [], "processed": {}, "evidence": {},
               "groups": Groups({}).to_data(), "overrides": {}, "listing": {}}
        if seed_v1:
            meta_rows = V._read_jsonl(vf["pool_meta"])
            idx["seen"] = {str(int(m["dhash"])): [m["key"], _classes(m["boxes"])] for m in meta_rows}
            idx["keys"] = sorted({m["key"] for m in meta_rows} | {r["key"] for r in V._read_jsonl(vf["copies"])})
            ps = _read_json(vf["pool_summary"], {})
            for slug in sorted(ps.get("per_slug") or {}):
                cache = _read_json(Path(vf["cache_dir"]) / "dhash" / ("%s.json" % V._sanitise(slug)), {}) or {}
                idx["processed"]["registry:%s" % slug] = sorted(cache)
            adm = _read_json(vf["admit_summary"], {})
            idx["evidence"] = {s: int(((d or {}).get("boxes") or {}).get(V.VERIFIED, 0))
                               for s, d in sorted((adm.get("per_slug") or {}).items())}
        for name, data in idx.items():
            _write_json(layout.index(name), {"applied": [], "data": data})
        # 4. pins
        pins = {"format": FORMAT, "stream_version": STREAM_VERSION, "built_utc": _utc(), "testing": bool(testing),
                "seeded_from_v1": bool(seed_v1),
                "splits": {"lock": str(lock_path), "lock_sha256": _sha_file(lock_path),
                           "nevertrain_sha256": lock["nevertrain_sha256"],
                           "base_copies_sha256": lock["base_copies_sha256"],
                           "expert_sets": be_info},
                "verifier": {"version": VERIFIER_VERSION, "dir": str(layout.verifier_dir), "files": vfiles,
                             "crops_sha256": crops.sha, "contract_crops_sha256": CONTRACT_VERIFIER_CROPS_SHA,
                             "source_dir": str(vf["verifier_dir"])},
                "embedder": {"name": emb["embedder"], "dim": int(emb["dim"]), "contract": CONTRACT_EMBEDDER,
                             "contract_dim": CONTRACT_DIM},
                "canary": {"file": str(layout.canary), "sha256": _sha_file(layout.canary), "n": len(cmeta["rows"]),
                           "min_cos": CANARY_MIN_COS, "seed_text": CANARY_SEED},
                "reference": {"version": REFERENCE_VERSION, "file": str(layout.reference),
                              "sha256": _sha_file(layout.reference), "meta": rmeta},
                "v1": {"inputs": _v1_records(), "crops": int(crops.n), "crops_contract": CONTRACT_V1_CROPS,
                       "embeddings": emb},
                "crop_offset": int(crops.n),
                "hold_deadline_days": HOLD_DEADLINE_DAYS, "batch_cap": BATCH_CAP,
                "modules": _module_hashes()}
        _write_json(layout.stream_json, pins)
        write_status(layout)
    log("bootstrap: pins %s; verifier %s; reference k1=%d; canary %d crops; base expert %s; seeded from v1: %s "
        "(%.0fs)" % (layout.stream_json, vfiles.get("verifier.npz", "")[:12], rmeta["k1"], len(cmeta["rows"]),
                     be_info["per_set"], seed_v1, time.time() - t0))
    return pins


def read_pins(layout):
    pins = _read_json(layout.stream_json, None)
    if not isinstance(pins, dict) or pins.get("format") != FORMAT:
        raise StreamError("%s is missing: run step1_stream bootstrap first" % layout.stream_json)
    return pins


def check_pins(layout, embedder=None, canary=True, lock_path=None):
    """(pins, stream_pins_sha) after checking every pin; raises StreamError
    before anything is written. With embedder None only the file pins are
    checked (verbs that embed nothing)."""
    pins = read_pins(layout)
    sp = pins["splits"]
    lp = Path(lock_path or sp["lock"])
    if str(lp) != sp["lock"]:
        raise StreamError("the lock %s is not the pinned %s" % (lp, sp["lock"]))
    if not lp.is_file() or _sha_file(lp) != sp["lock_sha256"]:
        raise StreamError("the splits v2 LOCK %s changed since bootstrap (pinned %s): a new splits version "
                          "means a new stream version" % (lp, sp["lock_sha256"][:12]))
    lock, _files = read_lock_v2(lp)
    for k in ("nevertrain_sha256", "base_copies_sha256"):
        if lock.get(k) != sp[k]:
            raise StreamError("the v2 %s differs from the pinned one" % k)
    for name, want in sorted(pins["verifier"]["files"].items()):
        p = layout.verifier_dir / name
        if not p.is_file() or _sha_file(p) != want:
            raise StreamError("the frozen verifier file %s changed (pinned %s)" % (p, want[:12]))
    for what in ("reference", "canary"):
        p = Path(pins[what]["file"])
        if not p.is_file() or _sha_file(p) != pins[what]["sha256"]:
            raise StreamError("%s changed since bootstrap" % p)
    rec = None
    if embedder is not None:
        if embedder.name != pins["embedder"]["name"] or int(embedder.dim) != int(pins["embedder"]["dim"]):
            raise StreamError("the embedder %s (dim %s) is not the pinned %s (dim %s)"
                              % (embedder.name, embedder.dim, pins["embedder"]["name"], pins["embedder"]["dim"]))
        if canary:
            rec = check_canary(Path(pins["canary"]["file"]), embedder, V.Verifier.load(layout.verifier_dir))
    return pins, _sha_file(layout.stream_json), rec


# --------------------------------------------------------------- listing
def _join_record(to_inc, src_names, wildcard):
    if wildcard:
        return {"*": [None, C.CLASS_NAMES[C.OTHER_PLANT]]}
    return {str(i): [src_names.get(i, ""), C.CLASS_NAMES[c]] for i, c in sorted(to_inc.items())}


def _norm_join(j):
    return json.loads(json.dumps(j, sort_keys=True))


def _read_source_boxes(path, to_inc, wildcard, max_id=C.OTHER_PLANT):
    """(boxes [[inc id, cx, cy, w, h]], src [[src id, name]], drop reason)."""
    try:
        src, _bad, _clipped = V.read_source_label(path)
    except (OSError, UnicodeDecodeError):
        return None, None, "unreadable_label"
    if not src:
        return None, None, "no_boxes"
    if to_inc is None:                                    # intake: the ids are INC ids already
        inc = [b[0] if 0 <= b[0] <= max_id else None for b in src]
    elif wildcard:
        inc = [C.OTHER_PLANT] * len(src)
    else:
        inc = [to_inc.get(b[0]) for b in src]
    if any(c is None for c in inc):
        return None, None, "unmapped_class"
    return [[int(c)] + [float(x) for x in b[1:]] for c, b in zip(inc, src)], [[b[0], ""] for b in src], None


def list_registry(layout, slugs=None, cap=BATCH_CAP, domain_path=None):
    """New paths of the v1 registry slugs: (items, refusals, listing). Only
    slugs of the v1 pool are read (contract §3.3 Inputs): any other registry
    slug is refused and must come through `collect intake`; a slug whose class
    join changed since v1 is refused (a `rejoin` is needed)."""
    from .. import mega_trainer as MT
    _check_join()
    ps = _read_json(V.POOL_SUMMARY, None)
    if not isinstance(ps, dict):
        raise StreamError("the v1 pool_summary.json is missing")
    per_slug = ps.get("per_slug") or {}
    registry = V._load_registry()
    flags = V._load_flags()
    processed = Index(layout, "processed", {}).data
    overrides = Index(layout, "overrides", {}).data
    evidence = v1_lab_evidence(ps)
    lic, lic_rec = licence_table(layout.inc_dir, domain_path)
    labs = funnel_lab_groups(domain_path)
    want = sorted(set(slugs)) if slugs else sorted(registry)
    items, refusals, listing = [], {}, {}
    taken = 0
    for slug in want:
        if slug not in registry:
            refusals[slug] = "not_in_registry"
            continue
        info = registry[slug]
        st = per_slug.get(slug)
        if st is None:
            refusals[slug] = "not_in_v1_pool"
            continue
        if st.get("calibration_only"):
            refusals[slug] = "calibration_only"
            continue
        why = V._skip_reason(slug, info, flags, MT)
        root = V._resolve_dir(info) if why is None else None
        if why is None and root is None:
            why = "missing_dir"
        parts = V._layout(root) if why is None else None
        if why is None and parts is None:
            why = "unknown_layout"
        if why is not None:
            refusals[slug] = why
            continue
        to_inc, src_names, wildcard = V.class_join(slug, info)
        if _norm_join(_join_record(to_inc, src_names, wildcard)) != _norm_join(st.get("join") or {}):
            refusals[slug] = "join_changed"
            continue
        ov = (overrides.get(slug) or {}).get("map") or {}
        if ov:
            to_inc = dict(to_inc)
            to_inc.update({int(k): int(v) for k, v in ov.items()})
        done = set(processed.get("registry:%s" % slug, []))
        ev = evidence.get(slug)
        licence, research_only = licence_state(lic.get(slug))
        lab = "LuLab" if ev else labs.get(slug)
        n_listed = n_new = n_taken = 0
        for split, idir, ldir in parts:
            labels = set(os.listdir(ldir))
            for name in sorted(os.listdir(idir)):
                if not V._is_image(name):
                    continue
                n_listed += 1
                stem = os.path.splitext(name)[0]
                rel = "%s/images/%s" % (split, name) if split else "images/%s" % name
                if rel in done:
                    continue
                if taken >= cap:
                    n_new += 1
                    continue
                it = {"input": "registry:%s" % slug, "item": rel, "source": slug, "split": split, "stem": stem,
                      "rel": rel, "image": str(Path(idir) / name), "boxes": None, "src": None, "drop": None,
                      "licence": licence, "research_only": research_only, "lab_group": lab,
                      "lab_evidence": bool(ev), "eval_copies": (ev or {}).get("eval_copies", 0),
                      "provenance_cleared": False, "capture_group": "", "session": "",
                      "declared_sha256": None, "declared_label_sha256": None, "intake_key": None,
                      "join_version": (overrides.get(slug) or {}).get("version", 0)}
                if stem + ".txt" not in labels:
                    it["drop"] = "no_label"
                else:
                    boxes, src, drop = _read_source_boxes(Path(ldir) / (stem + ".txt"), to_inc, wildcard)
                    it["drop"] = drop
                    if drop is None:
                        it["boxes"] = boxes
                        it["src"] = [[s[0], "" if wildcard else src_names.get(s[0], "")] for s in src]
                        taken += 1
                items.append(it)
                n_new += 1
                n_taken += 1
        listing[slug] = {"listed": n_listed, "new": n_new, "taken": n_taken, "pending": n_new - n_taken,
                         "utc": _utc()}
    return items, refusals, {"slugs": listing, "licences": lic_rec, "pending": sum(
        v["new"] for v in listing.values()) - len(items)}


def _check_join():
    """verify's own check: mega_trainer's class join must be the v3.60.0
    species join (a stale package copy would join by legacy labels)."""
    try:
        V._check_join_version()
    except V.VerifyError as e:
        raise StreamError(str(e))


def _intake_sources(doc):
    """{source id: source record} of an intake batch's sources.json: the
    collector's one-source form ({"source_id": ..., "licence": ..., ...}), a
    {"sources": {...} | [...]} form, or a bare {id: record} map."""
    if isinstance(doc, dict) and doc.get("source_id"):
        return {str(doc["source_id"]): doc}
    if isinstance(doc, dict) and isinstance(doc.get("sources"), (dict, list)):
        doc = doc["sources"]
    if isinstance(doc, list):
        return {str(d.get("source") or d.get("id")): d for d in doc if isinstance(d, dict)}
    return {str(k): v for k, v in (doc or {}).items() if isinstance(v, dict)}


def list_intake(layout, name, cap=BATCH_CAP):
    """Rows of a collect intake batch (INC_DIR/intake/<name>/) not yet
    processed. Their labels are INC ids 0-13 (13 = unmapped, masked)."""
    d = layout.inc_dir / "intake" / name
    for f in ("summary.json", "manifest.jsonl"):
        if not (d / f).is_file():
            raise StreamError("intake batch %s has no %s: it is not complete" % (d, f))
    sources = _intake_sources(_read_json(d / "sources.json", {}))
    processed = set(Index(layout, "processed", {}).data.get("intake:%s" % name, []))
    rows = sorted(C.read_manifest(d / "manifest.jsonl"), key=lambda r: r["key"])
    items, taken = [], 0
    for r in rows:
        if r["key"] in processed:
            continue
        if taken >= cap:
            break
        src = str(r.get("source") or "")
        srec = sources.get(src) or {}
        lab = r.get("lab_group") or srec.get("lab_group")
        licence, ro = intake_licence_state(r, srec)
        # cleared only when the row and its source both say so, the lab is not the evaluation lab, and the
        # collector did not hold the row for the copy scan itself
        cleared = bool(r.get("provenance_cleared", srec.get("provenance_cleared"))) \
            and srec.get("provenance_cleared", True) is not False and lab not in EVAL_LABS \
            and r.get("hold_until") != "h6_scan" and "h6_scan" not in (r.get("holds") or [])
        it = {"input": "intake:%s" % name, "item": r["key"], "source": src, "split": "", "stem": Path(r["image"]).stem,
              "rel": None, "image": r["image"], "boxes": None, "src": None, "drop": None,
              "licence": licence, "research_only": bool(r.get("research_only")) or bool(ro),
              "lab_group": lab, "lab_evidence": lab in EVAL_LABS, "eval_copies": 0,
              "provenance_cleared": cleared,
              "capture_group": str(r.get("capture_group") or ""), "session": str(r.get("capture_group") or ""),
              "declared_sha256": r.get("sha256"), "declared_label_sha256": r.get("label_sha256"),
              "intake_key": r["key"], "join_version": 0}
        if not r.get("label") or not Path(r["label"]).is_file():
            it["drop"] = "no_label"
        elif r.get("label_sha256") and _sha_file(r["label"]) != r["label_sha256"]:
            it["drop"] = "label_sha_mismatch"
        else:
            boxes, srcb, drop = _read_source_boxes(r["label"], None, False, max_id=MK.UNMAPPED_ID)
            it["drop"] = drop
            if drop is None:
                it["boxes"], it["src"] = boxes, srcb
                taken += 1
        items.append(it)
    return items, {}, {"intake": name, "summary": _file_rec(d / "summary.json"),
                       "manifest": _file_rec(d / "manifest.jsonl"), "sources": _file_rec(d / "sources.json"),
                       "pending": sum(1 for r in rows if r["key"] not in processed) - len(items)}


# ----------------------------------------------------------------- ingest
def consumed_hashes(layout, queue_rows=None):
    """dHash of every image a stream has consumed (INC_DIR/stream/*/consumed.jsonl
    keys, looked up in the queue): the near_consumed guard's index."""
    keys = set()
    sd = layout.inc_dir / "stream"
    if sd.is_dir():
        for sid in sorted(os.listdir(sd)):
            for r in _read_jsonl(sd / sid / "consumed.jsonl"):
                if r.get("key"):
                    keys.add(r["key"])
    idx = NearHashIndex()
    if keys:
        for r in (queue_rows if queue_rows is not None else _read_jsonl(layout.queue)):
            if r.get("key") in keys and r.get("dhash") is not None:
                idx.add(int(r["dhash"]), r["key"], max_bits=NEAR_CONSUMED_BITS)
    return idx, len(keys)


class _KeyMaker(V._Keys):
    def __init__(self, used):
        super().__init__()
        self.used = set(used)

    def intake(self, source, key, image):
        k = V._sanitise(key)
        if k in self.used:
            k = "%s__%s" % (k, C.sha256_text("%s/%s" % (source, image))[:8])
        if k in self.used:
            raise StreamError("key collision for intake row %s/%s" % (source, key))
        self.used.add(k)
        return k


def _probe_items(layout, items, procs):
    """{image: (dhash, sha256, W, H)} with a per-input cache under cache/dhash."""
    by_input = collections.defaultdict(list)
    for it in items:
        by_input[it["input"]].append(it)
    out = {}
    with V._Workers(procs) as w:
        for inp in sorted(by_input):
            cp = layout.cache / "dhash" / ("%s.json" % V._sanitise(inp))
            cache = _read_json(cp, {}) or {}
            got = V._probe_all([(it["item"], it["image"]) for it in by_input[inp]], w, cache, inp)
            _write_json(cp, cache)
            out.update(got)
    return out


def ingest(layout, items, guards, scanner, procs=1, bid=None):
    """Stages 2-4: hash, guard (GuardV2, exact_dup / held_join_conflict,
    near_consumed, near_eval_embed), keys. Returns (rows, summary); rows keep
    the item order, one per item, with "decision" (pool, knowntruth or a
    refusal reason)."""
    idx_keys = Index(layout, "keys", [])
    idx_seen = Index(layout, "seen", {})
    base_expert = Index(layout, "base_expert", {}).data
    be_index = NearHashIndex()
    for h, v in base_expert.items():
        be_index.add(int(h), tuple(v), max_bits=BASE_COPY_BITS)
    cons_index, n_consumed = consumed_hashes(layout)
    keys = _KeyMaker(idx_keys.data)
    seen = {int(h): (v[0], list(v[1])) for h, v in idx_seen.data.items()}
    local_seen = {}
    todo = [it for it in items if it["drop"] is None]
    probe = _probe_items(layout, todo, procs)
    gitems = [it for it in todo if probe[it["image"]][0] is not None and probe[it["image"]][2]
              and not (it["declared_sha256"] and probe[it["image"]][1] != it["declared_sha256"])]
    gres = dict(zip([it["image"] for it in gitems],
                    guards.check_many([(it["image"], probe[it["image"]][0]) for it in gitems], procs, "ingest guard")))
    rows, counts = [], collections.Counter()
    for it in items:
        reason = None
        r = dict(it, key=None, dhash=None, sha256=None, W=0, H=0, decision=None, guard_match=None, twin=None,
                 kt=None, label_sha256_full=None, scanned=None)
        if it["drop"] is not None:
            r["decision"] = it["drop"]
            rows.append(r)
            continue
        h, sha, W, H = probe[it["image"]]
        r.update(dhash=h, sha256=sha, W=int(W or 0), H=int(H or 0))
        r["label_sha256_full"] = C.sha256_text(V._yolo_text(it["boxes"]))
        if h is None or not W or not H:
            r["decision"] = "unhashable"
        elif it["declared_sha256"] and sha != it["declared_sha256"]:
            r["decision"] = "sha_mismatch"
        else:
            reason, match, _h = gres[it["image"]]
            if reason in ("unhashable",):
                r["decision"] = reason
            elif reason and reason != "base_copy":
                r["decision"], r["guard_match"] = reason, match
        if r["decision"] is None:
            if it["intake_key"] is not None:
                r["key"] = keys.intake(it["source"], it["intake_key"], it["image"])
            else:
                r["key"] = keys.make(it["source"], it["split"], it["stem"], it["rel"])
            if reason == "base_copy":
                r["decision"], r["guard_match"] = "base_copy", match
                m = be_index.find(int(h))
                if m is not None:
                    (bset, bkey, blabel), bits = m
                    r["decision"] = "knowntruth"
                    r["kt"] = {"set": bset, "base_key": bkey, "base_label": blabel, "bits": int(bits)}
            else:
                cls = _classes(it["boxes"])
                twin = local_seen.get(int(h)) or seen.get(int(h))
                if twin is not None:
                    r["twin"] = twin[0]
                    r["decision"] = "exact_dup" if list(twin[1]) == cls else "held_join_conflict"
                elif cons_index.find(int(h)) is not None:
                    r["decision"] = "near_consumed"
                    r["guard_match"] = list(cons_index.find(int(h)))
                else:
                    r["decision"] = "pool"
                    local_seen[int(h)] = (r["key"], cls)
        rows.append(r)
    # a twin inside this batch that is in conflict is held too
    conflict_twins = {r["twin"] for r in rows if r["decision"] == "held_join_conflict"}
    # the embedding copy check on the unmasked images of the pool rows that need it
    need = [r for r in rows if r["decision"] == "pool" and not r["provenance_cleared"]]
    by_which = collections.defaultdict(list)
    for r in need:
        w = scanner.which(r["lab_evidence"], False) if scanner is not None else None
        if w is not None:
            by_which[w].append(r)
    for w, rs in sorted(by_which.items()):
        res = scanner.scan([{"key": r["key"], "path": r["image"]} for r in rs], w)
        for r in rs:
            r["scanned"] = w
            if res.get(r["key"]) is not None:
                r["decision"], r["guard_match"] = scan_reason(res[r["key"]]), res[r["key"]]
    # D28-v2: the pair cosine of every dHash copy of an evaluation image (the scanner holds both descriptors)
    eval_rec = score_eval_hits(rows, scanner, procs, guards)
    for r in rows:
        counts[r["decision"]] += 1
        r["hold_join_conflict"] = r["key"] in conflict_twins
    summary = {"items": len(items), "decisions": dict(sorted(counts.items())), "consumed_keys": n_consumed,
               "guard": guards.record, "scanner": scanner.record() if scanner is not None else None,
               "scanned": {w: len(rs) for w, rs in sorted(by_which.items())}, "eval_hits": eval_rec}
    return rows, summary


# ------------------------------------------------------------------- crops
def write_crops(bdir, rows):
    """crops.csv (verify.CROP_FIELDS; local ids 0..n-1; "pool" for pool rows,
    "copy" for known-truth base copies) and crops_skipped.csv (small, no_size,
    unmapped). Returns the verify.Crops of it."""
    from ..semisup_labeler import MIN_BOX_PX
    body, left = [], []
    for r in sorted((r for r in rows if r["decision"] in ("pool", "knowntruth")), key=lambda r: r["key"]):
        set_ = "pool" if r["decision"] == "pool" else "copy"
        W, H = int(r["W"]), int(r["H"])
        for b, box in enumerate(r["boxes"]):
            cls, cx, cy, w, h = box
            if int(cls) == MK.UNMAPPED_ID:
                left.append([set_, r["key"], b, "unmapped"])
                continue
            if not W or not H:
                left.append([set_, r["key"], b, "no_size"])
                continue
            if w * W < MIN_BOX_PX or h * H < MIN_BOX_PX:
                left.append([set_, r["key"], b, "small"])
                continue
            body.append([len(body), set_, r["key"], r["image"], r["source"], r["source"], b, "%.6f" % cx,
                         "%.6f" % cy, "%.6f" % w, "%.6f" % h, W, H, int(cls), (r["src"][b][1] if r["src"] else "")])
    for name, header, rs in (("crops_skipped.csv", V.SKIPPED_FIELDS, left), ("crops.csv", V.CROP_FIELDS, body)):
        buf = io.StringIO()
        wr = csv.writer(buf, lineterminator="\r\n")
        wr.writerow(header)
        wr.writerows(rs)
        _write_bytes(bdir / name, buf.getvalue().encode("utf-8"))
    crops = V.Crops(bdir / "crops.csv")
    _write_json(bdir / "crops_info.json", {"crops": crops.n, "crops_sha256": crops.sha,
                                           "skipped_sha256": _sha_file(bdir / "crops_skipped.csv"),
                                           "ingest_sha256": _sha_file(bdir / "ingest.jsonl"),
                                           "min_box_px": MIN_BOX_PX})
    return crops


def read_skipped(bdir):
    out = {}
    with open(bdir / "crops_skipped.csv", newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for _set, key, box, reason in rd:
            out[(key, int(box))] = reason
    return out


def embed_batch(bdir, crops, embedder, procs=1, batch=V.BATCH, chunk=CHUNK_IMAGES):
    """(X float16 [crops.n, dim], meta): BioCLIP-2 features of every crop,
    resumable per chunk of images (emb/partNNNNN.npz), then emb.npz. A crop
    that cannot be cut or embedded is a NaN row (it is judged failed)."""
    final = bdir / "emb.npz"
    meta = V._npz_meta(final) if final.exists() else None
    if meta and meta.get("crops_sha256") == crops.sha:
        with np.load(final) as d:
            return d["X"], meta
    images = crops.images()
    chunk = max(1, int(chunk))
    n_parts = (len(images) + chunk - 1) // chunk
    parts = []
    t0 = time.time()
    with V._Workers(procs) as w:
        for j in range(n_parts):
            part = bdir / "emb" / ("part%05d.npz" % j)
            parts.append(part)
            pm = V._npz_meta(part) if part.exists() else None
            if pm and pm.get("crops_sha256") == crops.sha and pm.get("chunk_images") == chunk:
                continue
            t1 = time.time()
            ids, X, st = V._embed_images(images[j * chunk:(j + 1) * chunk], crops, embedder, w, int(batch))
            V._save_npz(part, {"crops_sha256": crops.sha, "chunk_images": chunk, "embedder": embedder.name,
                               "dim": int(embedder.dim), "stats": dict(st), "seconds": round(time.time() - t1, 1)},
                        crop_ids=ids, X=X.astype(np.float16))
            log("  embed chunk %d/%d: %d crops" % (j + 1, n_parts, len(ids)))
    names, dims, stats = set(), set(), collections.Counter()
    dim = int(embedder.dim) if not parts else None
    ids_all, X_parts = [], []
    for part in parts:
        with np.load(part) as d:
            pm = json.loads(str(d["meta"]))
            ids_all.append(d["crop_ids"])
            X_parts.append(d["X"])
        names.add(pm["embedder"])
        dims.add(int(pm["dim"]))
        stats.update(pm.get("stats", {}))
    if len(names) > 1 or len(dims) > 1:
        raise StreamError("%s: chunks embedded by different models %s" % (bdir, sorted(names)))
    if dims:
        dim = dims.pop()
    X = np.full((crops.n, dim), np.nan, dtype=np.float16)
    seen = np.zeros(crops.n, dtype=np.int64)
    for ids, Xp in zip(ids_all, X_parts):
        if len(ids):
            X[ids] = Xp
            np.add.at(seen, ids, 1)
    if not (seen == 1).all():
        raise StreamError("%s: the embedding chunks do not cover every crop exactly once" % bdir)
    meta = {"crops_sha256": crops.sha, "embedder": names.pop() if names else embedder.name, "dim": int(dim),
            "crops": int(crops.n), "stats": dict(stats), "seconds": round(time.time() - t0, 1)}
    V._save_npz(final, meta, crop_ids=np.arange(crops.n, dtype=np.int64), X=X)
    for part in parts:
        with contextlib.suppress(OSError):
            part.unlink()
    return X, meta


def judge_batch(layout, bdir, crops, X):
    """verdicts.npz: the frozen verifier's verdict of every crop. A crop that
    is one of the OtherPlant crops the verifier was trained on (same key and
    box as in v1) is judged by its out-of-fold prediction, as verify admit
    does."""
    ver = V.Verifier.load(layout.verifier_dir)
    v, j, pj, cj = ver.judge_features(crops.label, X)
    oof = (_read_json(layout.verifier_dir / "oof_keys.json", {}) or {}).get("keys") or []
    pos = {(k, int(b)): n for n, (k, b) in enumerate(oof)}
    hit = [(i, pos[(crops.key[i], int(crops.box[i]))]) for i in crops.where("pool")
           if (crops.key[i], int(crops.box[i])) in pos]
    if hit:
        rows = np.array([i for i, _n in hit], dtype=np.int64)
        on = np.array([n for _i, n in hit], dtype=np.int64)
        oP, oC, oU = ver.arrays["other_oof_P"][on], ver.arrays["other_oof_cos"][on], ver.arrays["other_oof_usable"][on]
        vo, jo, pjo, cjo = V.verdicts(np.full(len(on), V.OTHER), oP, oC, ver.tau_p, ver.sigma)
        vo[~oU.astype(bool)] = V.UNKNOWN
        v[rows], j[rows], pj[rows], cj[rows] = vo, jo, pjo, cjo
    codes = np.array([VERDICT_CODES.index(x) for x in v], dtype=np.int8)
    V._save_npz(bdir / "verdicts.npz", {"crops_sha256": crops.sha, "verdict_codes": list(VERDICT_CODES),
                                        "verifier": VERIFIER_VERSION, "oof_judged": len(hit)},
                crop_id=np.arange(crops.n, dtype=np.int64), verdict=codes, pred=j, p=pj, cosine=cj)
    return codes


def _load_verdicts(bdir, crops):
    with np.load(bdir / "verdicts.npz") as d:
        meta = json.loads(str(d["meta"]))
        if meta.get("crops_sha256") != crops.sha:
            return None
        return d["verdict"].copy(), d["pred"].copy(), d["p"].copy()


# -------------------------------------------------------------- admission
def _label_file(layout, source, key, boxes):
    text = V._yolo_text(boxes)
    path, sha = V._label_file(layout.labels, source, key, text)
    V._write_label(path, text)
    return str(path), sha


def _box_verdicts(r, crop_of, skipped, codes):
    out = []
    for b, box in enumerate(r["boxes"]):
        if int(box[0]) == MK.UNMAPPED_ID:
            out.append(MK.UNMAPPED)
            continue
        i = crop_of.get((r["key"], b))
        if i is not None:
            out.append(VERDICT_CODES[int(codes[i])])
            continue
        why = skipped.get((r["key"], b))
        if why is None:
            raise StreamError("box %s/%d has no crop row and was not left out: crops.csv is not this batch's"
                              % (r["key"], b))
        out.append(V.SMALL if why == "small" else V.FAILED)
    return out


def image_feature(Xu_rows):
    """The select image feature: the unit mean of unit box embeddings."""
    if not len(Xu_rows):
        return None
    return _unit(np.asarray(Xu_rows, dtype=np.float32).mean(axis=0, keepdims=True))[0]


def _place_rows(ref, feats, labels_kept, v1_scores=None):
    """Per admitted row: l1, l2, kind, score, typicality from the kept boxes'
    unit features (feats: [(Xu [n_kept_with_features, D], labels of those)])."""
    D = ref.c1.shape[1] if len(ref.c1) else (ref.P.shape[1])
    F = np.zeros((len(feats), D), dtype=np.float32)
    has = np.zeros(len(feats), dtype=bool)
    for i, (Xu, _lab) in enumerate(feats):
        f = image_feature(Xu)
        if f is not None:
            F[i], has[i] = f, True
    l1, l2 = ref.place(F, has)
    out = []
    for i, ((Xu, lab), kept) in enumerate(zip(feats, labels_kept)):
        lab = np.asarray(lab, dtype=np.int64)
        sp = lab < V.OTHER
        kind = "cwd12" if any(int(c) < V.OTHER for c in kept) else "other"
        score, typ = None, None
        if kind == "cwd12" and sp.any():
            score = round(float(ref.species_percentile(Xu[sp], lab[sp]).mean()), 6)
        if (~sp).any():
            t = ref.typicality(Xu[~sp])
            typ = round(float(np.nanmean(t)), 6) if np.isfinite(t).any() else None
        out.append({"l1": int(l1[i]), "l2": int(l2[i]), "kind": kind, "score": score, "typicality": typ})
    return out


def row_holds(r, extra=(), days=HOLD_DEADLINE_DAYS):
    """(holds, deadlines) of a new queue row."""
    holds, dl = [], {}
    if not r.get("provenance_cleared") and not r.get("scanned"):
        holds.append("h6_scan")
    if r.get("licence") is None:
        holds.append("licence")
    if r.get("hold_join_conflict"):
        holds.append("join_conflict")
    holds.extend(h for h in extra if h not in holds)
    for h in holds:
        if h in DEADLINE_HOLDS:
            dl[h] = _date(days)
    return [h for h in HOLD_ORDER if h in holds] + [h for h in holds if h not in HOLD_ORDER], dl


def _hold_until(holds):
    for h in HOLD_ORDER:
        if h in holds:
            return h
    return holds[0] if holds else None


def _mask_or_refuse(rec, image, mask_boxes, keep_boxes, out_dir, key):
    """inc2.mask.mask_except, or None with the record turned into a refusal
    (mask_failed) when the image cannot be opened to mask it: one unreadable
    image refuses itself, never the whole batch (a batch that fails on every
    rerun would block every other input behind it)."""
    try:
        return MK.mask_except(image, mask_boxes, keep_boxes, out_dir, key=key)
    except MK.MaskError as e:
        rec["admission"], rec["refusal"] = MK.NOT_ADMITTED, "mask_failed"
        rec["guard_match"] = {"why": str(e)[:300]}
        return None


def admit_rows(layout, bdir, rows, crops, codes, X, ref, pins_sha, bid, guards, scanner, rule="box",
               v1_scores=None, extra_holds=None, evidence=None, prior=None):
    """Stage 6 and 8: the admission of every pool row, the masks, the masked
    copy's guards, placement and the queue rows (group ids come at commit).
    Returns (admission records, queue rows, human rows)."""
    crop_of = {(crops.key[i], int(crops.box[i])): int(i) for i in range(crops.n)}
    skipped = read_skipped(bdir)
    recs, qrows, human = [], [], []
    todo = []
    for r in sorted((r for r in rows if r["decision"] == "pool"), key=lambda r: r["key"]):
        vs = _box_verdicts(r, crop_of, skipped, codes)
        labels = [int(b[0]) for b in r["boxes"]]
        coords = [b[1:] for b in r["boxes"]]
        if rule == "image":
            iv = V.image_verdict(labels, vs)
            d = {"admission": MK.WHOLE if iv == V.ADMITTED else MK.NOT_ADMITTED, "image_verdict": iv,
                 "keep": list(range(len(labels))) if iv == V.ADMITTED else [], "mask": [], "overlaps": []}
        else:
            d = MK.decide(labels, vs, coords)
        rec = {"key": r["key"], "source": r["source"], "verdicts": vs, "labels": labels, **d, "refusal": None,
               "mask_record": None}
        mrec = None
        if d["admission"] == MK.MASKED:
            mrec = _mask_or_refuse(rec, r["image"], [r["boxes"][i] for i in d["mask"]],
                                   [r["boxes"][i] for i in d["keep"]], layout.masked / V._sanitise(r["source"]), r["key"])
        if mrec is not None:
            rec["mask_record"] = mrec
            reason, match, hm = guards.check(mrec["path"])
            rec["dhash_masked"] = hm
            if reason:
                rec["admission"], rec["refusal"] = MK.NOT_ADMITTED, "masked_%s" % reason
                rec["guard_match"] = match
            elif r.get("scanned"):
                todo.append((rec, r))
        recs.append((rec, r))
    # the masked copies of scanned rows get the embedding check too
    by_which = collections.defaultdict(list)
    for rec, r in todo:
        by_which[r["scanned"]].append((rec, r))
    for w, lst in sorted(by_which.items()):
        res = scanner.scan([{"key": rec["key"], "path": rec["mask_record"]["path"]} for rec, _r in lst], w)
        for rec, _r in lst:
            if res.get(rec["key"]) is not None:
                rec["admission"], rec["refusal"], rec["guard_match"] = (MK.NOT_ADMITTED,
                                                                        scan_reason(res[rec["key"]], masked=True),
                                                                        res[rec["key"]])
    feats, kept_labels, admitted = [], [], []
    for rec, r in recs:
        if rec["admission"] in (MK.WHOLE, MK.MASKED):
            ids = [crop_of[(r["key"], b)] for b in rec["keep"] if (r["key"], b) in crop_of]
            ids = [i for i in ids if np.isfinite(np.asarray(X[i], dtype=np.float32)).all()]
            Xu = _unit(np.asarray(X[ids], dtype=np.float32)) if ids else np.zeros((0, X.shape[1]), np.float32)
            feats.append((Xu, [int(crops.label[i]) for i in ids]))
            kept_labels.append([rec["labels"][b] for b in rec["keep"]])
            admitted.append((rec, r))
        elif rec["admission"] == MK.REFUSED_OVERLAP or rec["image_verdict"] == V.CONFLICT:
            human.append({"key": r["key"], "batch": bid, "source": r["source"], "image": r["image"],
                          "reason": "refused_overlap" if rec["admission"] == MK.REFUSED_OVERLAP else "conflict",
                          "labels": rec["labels"], "verdicts": rec["verdicts"], "overlaps": rec["overlaps"],
                          "boxes": r["boxes"], "utc": _utc()})
    places = _place_rows(ref, feats, kept_labels) if admitted else []
    for (rec, r), pl in zip(admitted, places):
        kept = [r["boxes"][b] for b in rec["keep"]]
        if rec["admission"] == MK.WHOLE:
            label, lsha = _label_file(layout, r["source"], r["key"], r["boxes"])
            image, sha = r["image"], r["sha256"]
            dm, frac = None, 0.0
        else:
            label, lsha = _label_file(layout, r["source"], r["key"], kept)
            image, sha = rec["mask_record"]["path"], rec["mask_record"]["sha256"]
            dm, frac = rec.get("dhash_masked"), rec["mask_record"]["masked_area_frac"]
        sp = [0] * V.OTHER
        for b in kept:
            if int(b[0]) < V.OTHER:
                sp[int(b[0])] += 1
        extra = list((extra_holds or {}).get(r["key"], []))
        holds, dl = row_holds(r, extra)
        score = pl["score"]
        if pl["kind"] == "other" and v1_scores is not None:
            score = v1_scores.get(r["source"])
        q = {"format": QUEUE_FORMAT, "key": r["key"], "batch": bid, "source": r["source"], "group": None,
             "capture_group": r.get("capture_group") or "", "l1": pl["l1"], "l2": pl["l2"], "kind": pl["kind"],
             "score": score, "typicality": pl["typicality"], "species_boxes": sp,
             "other_boxes": sum(1 for b in kept if int(b[0]) == V.OTHER), "admission": rec["admission"],
             "n_masked": len(rec["mask"]), "evidenced": bool((evidence or {}).get(r["source"], 0) >= 1),
             "lab_group": r.get("lab_group"), "licence": r.get("licence"),
             "research_only": bool(r.get("research_only")), "prior": (prior or {}).get(r["key"]),
             "hold_until": _hold_until(holds), "holds": holds, "hold_deadline": dl,
             "verifier": VERIFIER_VERSION, "reference": REFERENCE_VERSION, "admitted_utc": _utc(),
             "stream_pins_sha": pins_sha,
             "image": image, "label": label, "sha256": sha, "label_sha256": lsha,
             "session": r.get("session") or "", "unmasked_image": r["image"], "unmasked_sha256": r["sha256"],
             "dhash": int(r["dhash"]), "dhash_masked": dm, "masked_area_frac": frac, "input": r["input"],
             "supersedes": r.get("supersedes"), "scanned": r.get("scanned"),
             "h6_reason": ("lab_evidence" if r.get("lab_evidence") else "not_scanned")
             if "h6_scan" in holds else None}
        qrows.append(q)
    out_recs = [rec for rec, _r in recs]
    return out_recs, qrows, human


# ------------------------------------------------------------ known truth
def known_truth(rows, crops, codes):
    """Stage 7: base copies box-matched to their expert labels (train_core,
    tsw22, tsw23): verify._metrics per set and overall, with the Wilson lower
    bound of verified precision."""
    crop_of = {(crops.key[i], int(crops.box[i])): int(i) for i in range(crops.n)}
    per = collections.defaultdict(lambda: ([], []))
    unmatched = collections.Counter()
    for r in rows:
        if r["decision"] != "knowntruth":
            continue
        truth = C.read_yolo(r["kt"]["base_label"])
        for b, box in enumerate(r["boxes"]):
            if int(box[0]) == MK.UNMAPPED_ID:
                continue
            t = V._match_box(tuple(box), truth)
            if t is None:
                unmatched[r["kt"]["set"]] += 1
                continue
            i = crop_of.get((r["key"], b))
            v = VERDICT_CODES[int(codes[i])] if i is not None else V.SMALL
            per[r["kt"]["set"]][0].append(int(box[0]) == int(t[0]))
            per[r["kt"]["set"]][1].append(v)
    out = {"per_set": {}, "unmatched_boxes": dict(unmatched)}
    allc, allv = [], []
    for s in sorted(per):
        c, v = per[s]
        out["per_set"][s] = _kt_metrics(c, v)
        allc += c
        allv += v
    out["overall"] = _kt_metrics(allc, allv)
    return out


def _kt_metrics(correct, verdict):
    m = V._metrics(np.asarray(correct, dtype=bool), np.asarray(verdict, dtype=object))
    ver = np.asarray(verdict, dtype=object) == V.VERIFIED
    k = int((ver & np.asarray(correct, dtype=bool)).sum()) if len(correct) else 0
    n = int(ver.sum()) if len(correct) else 0
    m.update({"matched_verified": n, "verified_correct": k, "verified_precision_wilson_lb": wilson_lb(k, n)})
    return m


# ------------------------------------------------------------------ commit
def _batch_ids(layout):
    if not layout.batches.is_dir():
        return []
    return sorted(n for n in os.listdir(layout.batches) if re.fullmatch(r"b\d{4}", n))


def _next_bid(layout):
    """The next batch id; b0000 is the backfill's, so batches start at b0001."""
    return "b%04d" % (max([0] + [int(b[1:]) for b in _batch_ids(layout)]) + 1)


def _ledger_ids(layout):
    return [e["batch"] for e in verify_ledger(layout.ledger)]


def _append_missing(path, rows, bid, key="key"):
    """Append the rows of batch bid that the file does not hold yet (a commit
    re-run after a kill appends each row once)."""
    have = {(r.get("batch"), r.get(key), r.get("event")) for r in _read_jsonl(path) if r.get("batch") == bid}
    _append_jsonl(path, [r for r in rows if (r.get("batch"), r.get(key), r.get("event")) not in have])


def prepare_commit(layout, bid, bdir, doc, queue_rows, human_rows, events, index_changes, crops_n):
    """Everything a commit writes, fixed before batch.json: group ids, the
    index update, the batch record. batch.json is written last."""
    ledger = verify_ledger(layout.ledger)
    if bid in [e["batch"] for e in ledger]:
        return _read_json(bdir / "batch.json")
    pins = read_pins(layout)
    offset = int(pins["crop_offset"]) + sum(int(e.get("crops", 0)) for e in ledger)
    groups = Groups(Index(layout, "groups", {}).data)
    events = list(events)
    for q in sorted(queue_rows, key=lambda q: q["key"]):
        g, merged = groups.assign(q["dhash"])
        q["group"] = g
        if merged:
            events.append({"event": "merge_groups", "batch": bid, "key": q["key"], "into": g, "from": merged,
                           "utc": _utc()})
    index_changes = dict(index_changes, groups=groups.to_data())
    _write_jsonl(bdir / "queue_rows.jsonl", sorted(queue_rows, key=lambda q: q["key"]))
    _write_jsonl(bdir / "human.jsonl", human_rows)
    _write_jsonl(bdir / "events.jsonl", events)
    _write_json(bdir / "index_update.json", index_changes)
    doc = dict(doc, batch=bid, crop_offset=offset, crops=int(crops_n), global_crop_ids=[offset, offset + int(crops_n)],
               queue_rows=len(queue_rows), queue_rows_sha256=_sha_file(bdir / "queue_rows.jsonl"),
               human_rows=len(human_rows), events=len(events),
               index_update_sha256=_sha_file(bdir / "index_update.json"),
               stream_pins_sha=_sha_file(layout.stream_json), committed_utc=_utc(), modules=_module_hashes())
    _write_json(bdir / "batch.json", doc)
    return doc


def finish_commit(layout, bid):
    """Apply a written batch.json: indexes, queue, events, human queue, then the
    ledger line. Idempotent: a batch already in the ledger is a no-op."""
    bdir = layout.batch_dir(bid)
    doc = _read_json(bdir / "batch.json", None)
    if doc is None:
        raise StreamError("%s has no batch.json: nothing to commit" % bdir)
    if bid in _ledger_ids(layout):
        return doc
    for name, sha in (("queue_rows.jsonl", doc["queue_rows_sha256"]),
                      ("index_update.json", doc["index_update_sha256"])):
        if _sha_file(bdir / name) != sha:
            raise StreamError("%s/%s changed after batch.json was written" % (bdir, name))
    upd = _read_json(bdir / "index_update.json", {})
    for name, change in sorted(upd.items()):
        ix = Index(layout, name, {} if name not in ("keys",) else [])
        if ix.applied(bid):
            continue
        if name == "groups":
            ix.doc["data"] = change
        elif name == "keys":
            ix.doc["data"] = sorted(set(ix.data) | set(change))
        elif name == "processed":
            for inp, items in change.items():
                ix.data[inp] = sorted(set(ix.data.get(inp, [])) | set(items))
        elif name == "evidence":
            for s, n in change.items():
                ix.data[s] = int(ix.data.get(s, 0)) + int(n)
        elif name in ("seen", "listing", "overrides"):
            ix.data.update(change)
        else:
            raise StreamError("unknown index %s in %s" % (name, bdir))
        ix.save(bid)
    _append_missing(layout.queue, _read_jsonl(bdir / "queue_rows.jsonl"), bid)
    _append_missing(layout.events, _read_jsonl(bdir / "events.jsonl"), bid)
    _append_missing(layout.human, _read_jsonl(bdir / "human.jsonl"), bid)
    ledger_append(layout.ledger, {"batch": bid, "kind": doc.get("kind"), "input": doc.get("input"),
                                  "crop_offset": doc["crop_offset"], "crops": doc["crops"],
                                  "queue_rows": doc["queue_rows"], "queue_rows_sha256": doc["queue_rows_sha256"],
                                  "batch_json_sha256": _sha_file(bdir / "batch.json"), "utc": _utc()})
    write_status(layout)
    log("committed %s: %d queue rows, crops %s" % (bid, doc["queue_rows"], doc["global_crop_ids"]))
    return doc


def _finish_pending(layout):
    """Finish every batch whose batch.json exists but is not in the ledger."""
    done = set(_ledger_ids(layout))
    for bid in _batch_ids(layout):
        if bid not in done and (layout.batch_dir(bid) / "batch.json").exists():
            finish_commit(layout, bid)


def _in_progress(layout):
    """The batch a writer must finish first: planned (plan.json written), not
    committed and without batch.json. A directory without plan.json holds
    nothing a rerun could resume (a job killed before it planned) and blocks
    nothing; its id is never reused. b0000 is the backfill's own: it resumes
    only through `backfill`, and a backfill that does not reconcile (a census
    mismatch waits for a person) must not stop every admit behind it."""
    done = set(_ledger_ids(layout))
    for bid in _batch_ids(layout):
        d = layout.batch_dir(bid)
        if bid == "b0000" or bid in done or (d / "batch.json").exists() or not (d / "plan.json").exists():
            continue
        return bid
    return None


# --------------------------------------------------------------- one batch
def _stage_current(path, key, want):
    doc = _read_json(path, None)
    return isinstance(doc, dict) and doc.get(key) == want


def run_batch(layout, spec, list_fn, embedder, guards, scanner=None, rule="box", procs=1, batch=V.BATCH,
              chunk=CHUNK_IMAGES, kind=KIND_REGISTRY, canary=True):
    """One batch, stages 1-9 (contract §3.3). spec names the input (the same
    argv resumes a killed batch). Returns the batch record, or None when there
    is nothing new."""
    if rule not in ("box", "image"):
        raise StreamError("--rule must be box or image")
    scanner = scanner or CopyScanner()
    with writer_lock(layout):
        pins, pins_sha, canary_rec = check_pins(layout, embedder, canary=canary)
        _finish_pending(layout)
        bid = _in_progress(layout)
        if bid is not None:
            bdir = layout.batch_dir(bid)
            plan = _read_json(bdir / "plan.json", {})
            if plan.get("spec") != spec or plan.get("rule") != rule:
                raise StreamError("batch %s (%s, rule %s) is in progress: rerun its own argv to finish it"
                                  % (bid, plan.get("spec"), plan.get("rule")))
            items = _read_jsonl(bdir / "plan_items.jsonl")
            log("resuming %s (%s)" % (bid, spec))
        else:
            items, refusals, listing = list_fn()
            if not items:
                ix = Index(layout, "listing", {})
                ix.data.update({k: v for k, v in (listing.get("slugs") or {}).items()})
                ix.data.update({k: {"refused": v, "utc": _utc()} for k, v in (refusals or {}).items()})
                ix.save()
                if kind == KIND_KNOWNTRUTH:
                    # a known-truth run that finds nothing to measure has still run: status says so, or the
                    # platform would submit it again and again (one_time.knowntruth)
                    ot = Index(layout, "one_time", {})
                    ot.data.setdefault("knowntruth", {"utc": _utc(), "spec": spec, "batch": None})
                    ot.save()
                write_status(layout)
                log("%s: nothing new (refused: %s)" % (spec, refusals or "none"))
                return None
            bid = _next_bid(layout)
            bdir = layout.batch_dir(bid)
            bdir.mkdir(parents=True)
            _write_jsonl(bdir / "plan_items.jsonl", items)
            _write_json(bdir / "plan.json", {"batch": bid, "kind": kind, "spec": spec, "rule": rule,
                                             "items": len(items), "items_sha256": _sha_file(bdir / "plan_items.jsonl"),
                                             "refusals": refusals, "listing": listing, "stream_pins_sha": pins_sha,
                                             "created_utc": _utc()})
            log("%s: planned %s with %d item(s)" % (spec, bid, len(items)))
        plan = _read_json(bdir / "plan.json")
        t0 = time.time()
        # stages 2-4
        if not _stage_current(bdir / "ingest.json", "items_sha256", plan["items_sha256"]):
            rows, isum = ingest(layout, items, guards, scanner, procs, bid)
            if kind == KIND_KNOWNTRUTH:
                for r in rows:                      # a known-truth batch never queues anything
                    if r["decision"] == "pool":
                        r["decision"] = "not_base_copy"
            _write_jsonl(bdir / "ingest.jsonl", rows)
            _write_json(bdir / "ingest.json", dict(isum, items_sha256=plan["items_sha256"],
                                                   ingest_sha256=_sha_file(bdir / "ingest.jsonl")))
        rows = _read_jsonl(bdir / "ingest.jsonl")
        isum = _read_json(bdir / "ingest.json")
        # stage 5
        if not _stage_current(bdir / "crops_info.json", "ingest_sha256", isum["ingest_sha256"]):
            crops = write_crops(bdir, rows)
        else:
            crops = V.Crops(bdir / "crops.csv")
        X, emb = embed_batch(bdir, crops, embedder, procs, batch, chunk)
        got = _load_verdicts(bdir, crops) if (bdir / "verdicts.npz").exists() else None
        codes = got[0] if got is not None else judge_batch(layout, bdir, crops, X)
        # stages 6-8
        ref = Reference(layout.reference)
        ev_idx = Index(layout, "evidence", {}).data
        ev_batch = collections.Counter()
        crop_of = {(crops.key[i], int(crops.box[i])): int(i) for i in range(crops.n)}
        for r in rows:
            if r["decision"] == "pool":
                for b, box in enumerate(r["boxes"]):
                    i = crop_of.get((r["key"], b))
                    if i is not None and int(box[0]) < V.OTHER and VERDICT_CODES[int(codes[i])] == V.VERIFIED:
                        ev_batch[r["source"]] += 1
        evidence = {s: int(ev_idx.get(s, 0)) + int(ev_batch.get(s, 0)) for s in set(ev_idx) | set(ev_batch)}
        v1s = _v1_source_scores()
        recs, qrows, human = admit_rows(layout, bdir, rows, crops, codes, X, ref, pins_sha, bid, guards, scanner,
                                        rule=rule, v1_scores=v1s, evidence=evidence)
        kt = known_truth(rows, crops, codes)
        _write_jsonl(bdir / "admission.jsonl", recs)
        # stage 9
        events = []
        for r in rows:
            if r["decision"] == "held_join_conflict":
                events.append({"event": "hold", "batch": bid, "key": r["twin"], "hold": "join_conflict",
                               "because": r["key"], "utc": _utc()})
                events.append({"event": "rejoin_needed", "batch": bid, "key": r["key"], "twin": r["twin"],
                               "source": r["source"], "utc": _utc()})
        in_batch = {q["key"] for q in qrows}
        events = [e for e in events if not (e["event"] == "hold" and e["key"] in in_batch)]
        processed = collections.defaultdict(list)
        for it in items:
            processed[it["input"]].append(it["item"])
        index_changes = {
            "processed": dict(processed),
            "keys": sorted({r["key"] for r in rows if r.get("key")}),
            "seen": {str(int(r["dhash"])): [r["key"], _classes(r["boxes"])] for r in rows if r["decision"] == "pool"},
            "evidence": dict(ev_batch),
            "listing": dict({k: v for k, v in ((plan.get("listing") or {}).get("slugs") or {}).items()},
                            **{k: {"refused": v, "utc": plan.get("created_utc")}
                               for k, v in (plan.get("refusals") or {}).items()})}
        adm = collections.Counter(rec["admission"] for rec in recs)
        per_src = collections.defaultdict(collections.Counter)
        for r in rows:
            per_src[r["source"]][r["decision"]] += 1
        per_ref = collections.defaultdict(collections.Counter)
        for rec in recs:
            if rec["refusal"]:
                per_ref[rec["source"]][rec["refusal"]] += 1
        sp_verdicts = collections.defaultdict(collections.Counter)
        for r in rows:
            if r["decision"] == "pool":
                for b, box in enumerate(r["boxes"]):
                    if int(box[0]) < V.OTHER:
                        i = crop_of.get((r["key"], b))
                        sp_verdicts[C.CLASS_NAMES[int(box[0])]][VERDICT_CODES[int(codes[i])] if i is not None
                                                                else "not_embedded"] += 1
        doc = {"format": FORMAT, "kind": kind, "input": spec, "rule": rule,
               "plan_sha256": _sha_file(bdir / "plan.json"),
               "items": len(items), "decisions": isum["decisions"], "admission": dict(sorted(adm.items())),
               "refusals_after_admission": dict(collections.Counter(r["refusal"] for r in recs if r["refusal"])),
               "knowntruth": kt, "canary": canary_rec,
               "per_source_decisions": {k: dict(v) for k, v in sorted(per_src.items())},
               "per_source_refusals": {k: dict(v) for k, v in sorted(per_ref.items())},
               "species_verdicts": {k: dict(v) for k, v in sorted(sp_verdicts.items())}, "embeddings": emb,
               "guard": isum.get("guard"),
               "scanner": isum.get("scanner"), "scanned": isum.get("scanned"), "eval_hits": isum.get("eval_hits"),
               "inputs": {"plan_items": _file_rec(bdir / "plan_items.jsonl"),
                          "ingest": _file_rec(bdir / "ingest.jsonl"),
                          "crops": _file_rec(bdir / "crops.csv"), "verdicts": _file_rec(bdir / "verdicts.npz"),
                          "emb": _file_rec(bdir / "emb.npz"), "reference": _file_rec(layout.reference)},
               "seconds": round(time.time() - t0, 1)}
        prepare_commit(layout, bid, bdir, doc, qrows, human, events, index_changes, crops.n)
        return finish_commit(layout, bid)


def _v1_source_scores():
    """select's recorded source score (the median percentile of a source's
    species boxes), the score of an OtherPlant-only image (recorded only)."""
    ss = _read_json(v1_files()["select_summary"], {}) or {}
    return {s: v.get("median") for s, v in ((ss.get("retrieval") or {}).get("source_evidence") or {}).items()}


# ---------------------------------------------------------- b0000 backfill
BACKFILL_ID = "b0000"


def realloop_priors(inc_dir, exp="realloop_v1"):
    """{key: "realloop_v1:<step>:<verdict>(<blame>)"} for every image a
    realloop experiment drew, from its exp.json manifests (sha256-checked) and
    report.json verdicts. {} without the experiment."""
    d = Path(inc_dir) / exp
    ej, rj = _read_json(d / "exp.json", None), _read_json(d / "report.json", None)
    if ej is None or rj is None:
        return {}, {"exp": exp, "found": False}
    verdicts = {}
    for st in rj.get("steps") or []:
        tags = []
        for ch in sorted((st.get("chains") or {})):
            c = st["chains"][ch] or {}
            tags.append("%s(%s)" % (c.get("verdict"), (c.get("attribution") or {}).get("blame")))
        verdicts[st.get("step")] = ";".join(tags)
    out, used = {}, []
    for st in ej.get("steps") or []:
        p = Path(st.get("manifest") or "")
        if not p.is_file():
            p = d / "manifests" / Path(st.get("manifest") or st.get("select_manifest") or "").name
        if not p.is_file():
            raise StreamError("%s step %s: manifest %s is missing" % (exp, st.get("name"), p))
        if st.get("manifest_sha256") and _sha_file(p) != st["manifest_sha256"]:
            raise StreamError("%s step %s: %s does not hash to exp.json's manifest_sha256" % (exp, st.get("name"), p))
        used.append(_file_rec(p))
        for r in C.read_manifest(p):
            out.setdefault(r["key"], "%s:%s:%s" % (exp, st.get("name"), verdicts.get(st.get("name"), "undecided")))
    return out, {"exp": exp, "found": True, "manifests": used, "exp_json": _file_rec(d / "exp.json"),
                 "report_json": _file_rec(d / "report.json")}


def _census(path):
    doc = _read_json(path, None) if path else None
    if not isinstance(doc, dict) or not isinstance(doc.get("veto"), dict):
        return None
    ve = doc["veto"]
    per = {k: int(v.get("verified", 0)) - int(v.get("admitted", 0)) for k, v in (ve.get("per_class") or {}).items()}
    return {"path": str(path), "sha256": _sha_file(path), "images": int(ve.get("images", -1)),
            "lost_boxes": int(ve.get("lost_boxes", -1)), "lost_per_class": per}


def backfill(layout, guards, scanner=None, census_path=None, require_census=True, realloop_exp="realloop_v1",
             procs=1, domain_path=None, v1_embeddings=None):
    """Batch b0000: D-B over the existing v1 pool (contract §3.3 One-time jobs).
    Queue rows: the v1 increment pool (whole) and the masked rows of the
    non-admitted images holding a verified target box (held until the
    funnel's F9, P8); base B's images are marked as base (the ones in base_v2
    and the ones L-5 dropped), never queued. Every row passes GuardV2 on the
    unmasked image (and the masked copy), the same lab-group and licence
    holds as registry slugs apply, and the candidates reconcile with the
    funnel census (census_v1 veto), or nothing is committed."""
    scanner = scanner or CopyScanner()
    bid = BACKFILL_ID
    with writer_lock(layout):
        pins, pins_sha, _c = check_pins(layout, None)
        _finish_pending(layout)
        if bid in _ledger_ids(layout):
            log("%s is committed already: nothing to do" % bid)
            return _read_json(layout.batch_dir(bid) / "batch.json")
        ip = _in_progress(layout)
        if ip is not None and ip != bid:
            raise StreamError("batch %s is in progress; finish it before the backfill" % ip)
        bdir = layout.batch_dir(bid)
        bdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        vf = v1_files()
        for name in ("pool", "pool_meta", "pool_summary", "crops", "crops_skipped", "verified", "admit_summary",
                     "pool_verdicts", "increment_pool", "base_selected", "select_summary", "clusters"):
            _require(vf[name], "v1 Step 1 / select output")
        crops = V.Crops(vf["crops"])
        if crops.sha != pins["verifier"]["crops_sha256"]:
            raise StreamError("the v1 crops.csv is not the one the frozen verifier was fitted on")
        pv, codes = S.read_pool_verdicts(vf["pool_verdicts"], crops)
        lookup = V._box_verdict_lookup(crops, "pool", lambda i: V.VERDICT_CODES[int(pv[i])])
        pool = {r["key"]: r for r in C.read_manifest(vf["pool"])}
        meta = {m["key"]: m for m in V._read_jsonl(vf["pool_meta"])}
        verified = {r["key"] for r in C.read_manifest(vf["verified"])}
        inc_rows = C.read_manifest(vf["increment_pool"])
        base_keys = {r["key"] for r in C.read_manifest(vf["base_selected"])}
        if {r["key"] for r in inc_rows} | base_keys != verified or {r["key"] for r in inc_rows} & base_keys:
            raise StreamError("increment_pool.jsonl and base_selected.jsonl do not partition verified.jsonl")
        verdicts, admitted = {}, set()
        for key in sorted(pool):
            m = meta[key]
            vs = [lookup(key, b)[0] for b in range(len(m["boxes"]))]
            verdicts[key] = vs
            if V.image_verdict([int(b[0]) for b in m["boxes"]], vs) == V.ADMITTED:
                admitted.add(key)
        if admitted != verified:
            raise StreamError("the image rule over pool_verdicts.npz admits %d images, verified.jsonl holds %d: "
                              "the v1 files do not agree" % (len(admitted), len(verified)))
        cand = sorted(k for k in pool if k not in admitted and any(
            int(b[0]) < V.OTHER and v == V.VERIFIED for b, v in zip(meta[k]["boxes"], verdicts[k])))
        # holds and labels of the v1 slugs
        ps = _read_json(vf["pool_summary"], {})
        evid = v1_lab_evidence(ps)
        lic, lic_rec = licence_table(layout.inc_dir, domain_path)
        labs = funnel_lab_groups(domain_path)
        priors, prior_rec = realloop_priors(layout.inc_dir, realloop_exp)
        # base_v2 membership of base B's images
        lock, lfiles = read_lock_v2(pins["splits"]["lock"])
        bv2 = lfiles.get("manifest:base_v2")
        base_v2_keys = {r["key"] for r in C.read_manifest(bv2)} if bv2 is not None and Path(bv2).is_file() else None
        l5p = Path(pins["splits"]["lock"]).parent / "l5_excluded.jsonl"
        if l5p.is_file() and lock.get("l5_excluded_sha256") and _sha_file(l5p) != lock["l5_excluded_sha256"]:
            raise StreamError("%s changed since the v2 lock" % l5p)
        l5_keys = ({r["key"] for r in _read_jsonl(l5p)} if l5p.is_file()
                   else {k for k in base_keys if pool[k]["source"] in L5_DROPPED_SOURCES})
        if l5_keys - base_keys:
            raise StreamError("%d L-5 excluded key(s) are not base B images: %s is not this v1 pool's"
                              % (len(l5_keys - base_keys), l5p))
        base_rec = {"base_b": len(base_keys),
                    "in_base_v2": (len(base_keys & base_v2_keys) if base_v2_keys is not None
                                   else len(base_keys - l5_keys)),
                    "dropped_l5": len(l5_keys),
                    "l5_excluded": _file_rec(l5p) if l5p.is_file() else {"rule": "sources %s" % (L5_DROPPED_SOURCES,)},
                    "base_v2_manifest": _file_rec(bv2) if bv2 is not None else None,
                    "rule": "base B's images are never queued: those in base_v2 are base, the L-5 ones are "
                            "excluded outright"}

        def item(key):
            r, m = pool[key], meta[key]
            ev = evid.get(r["source"])
            licence, ro = licence_state(lic.get(r["source"]))
            return {"input": "v1", "item": key, "key": key, "source": r["source"], "image": r["image"],
                    "sha256": r["sha256"], "dhash": int(m["dhash"]), "W": m["W"], "H": m["H"],
                    "boxes": [list(b) for b in m["boxes"]], "src": m["src"], "licence": licence,
                    "research_only": ro, "lab_group": "LuLab" if ev else labs.get(r["source"]),
                    "lab_evidence": bool(ev), "provenance_cleared": False, "capture_group": "", "session": "",
                    "decision": "pool", "scanned": None, "hold_join_conflict": False}
        rows = [item(r["key"]) for r in sorted(inc_rows, key=lambda r: r["key"])] + [item(k) for k in cand]
        # guards on the unmasked images (all eight variants) and near_consumed
        cons_index, n_cons = consumed_hashes(layout)
        refused = collections.Counter()
        res = guards.check_many([(r["image"], r["dhash"]) for r in rows], procs, "b0000 guard")
        for r, (reason, match, _h) in zip(rows, res):
            if reason is None and cons_index.find(r["dhash"]) is not None:
                reason, match = "near_consumed", list(cons_index.find(r["dhash"]))
            if reason:
                r["decision"], r["guard_match"] = reason, match
                refused[reason] += 1
        log("  b0000 guards: %d images, refused %s" % (len(rows), dict(refused)))
        by_which = collections.defaultdict(list)
        for r in rows:
            if r["decision"] == "pool":
                w = scanner.which(r["lab_evidence"], False)
                if w is not None:
                    by_which[w].append(r)
        for w, rs in sorted(by_which.items()):
            res = scanner.scan([{"key": r["key"], "path": r["image"]} for r in rs], w)
            for r in rs:
                r["scanned"] = w
                if res.get(r["key"]) is not None:
                    r["decision"], r["guard_match"] = scan_reason(res[r["key"]]), res[r["key"]]
                    refused[r["decision"]] += 1
        # D28-v2: the pair cosine of every dHash copy of an evaluation image
        eval_rec = score_eval_hits(rows, scanner, procs, guards)
        # admission: whole (increment pool) and D-B (candidates)
        cset = set(cand)
        X, emb = v1_embeddings if v1_embeddings is not None else load_v1_embeddings(crops)
        crop_of = {(crops.key[i], int(crops.box[i])): int(i) for i in crops.where("pool")}
        recs, extra, todo = [], {}, []
        for r in rows:
            if r["key"] in cset:
                d = MK.decide([int(b[0]) for b in r["boxes"]], verdicts[r["key"]], [b[1:] for b in r["boxes"]])
            else:
                d = {"admission": MK.WHOLE, "image_verdict": V.ADMITTED, "keep": list(range(len(r["boxes"]))),
                     "mask": [], "overlaps": []}
            rec = {"key": r["key"], "source": r["source"], "verdicts": verdicts[r["key"]],
                   "candidate": r["key"] in cset,
                   "labels": [int(b[0]) for b in r["boxes"]], **d, "refusal": None, "mask_record": None}
            if r["decision"] != "pool":
                rec["refusal"] = r["decision"]
                if d["admission"] in (MK.WHOLE, MK.MASKED):
                    rec["admission"] = MK.NOT_ADMITTED
            elif d["admission"] == MK.MASKED:
                mrec = _mask_or_refuse(rec, r["image"], [r["boxes"][i] for i in d["mask"]],
                                       [r["boxes"][i] for i in d["keep"]], layout.masked / V._sanitise(r["source"]),
                                       r["key"])
                if mrec is not None:
                    rec["mask_record"] = mrec
                    reason, match, hm = guards.check(mrec["path"])
                    rec["dhash_masked"] = hm
                    if reason:
                        rec["admission"], rec["refusal"], rec["guard_match"] = (MK.NOT_ADMITTED, "masked_%s" % reason,
                                                                                match)
                    elif r["scanned"]:
                        todo.append((rec, r))
                    extra[r["key"]] = ["funnel_F9"]
            recs.append((rec, r))
        by_w = collections.defaultdict(list)
        for rec, r in todo:
            by_w[r["scanned"]].append((rec, r))
        for w, lst in sorted(by_w.items()):
            res = scanner.scan([{"key": rec["key"], "path": rec["mask_record"]["path"]} for rec, _r in lst], w)
            for rec, _r in lst:
                if res.get(rec["key"]) is not None:
                    rec["admission"], rec["refusal"] = MK.NOT_ADMITTED, scan_reason(res[rec["key"]], masked=True)
        # placement over the v1 features of the kept boxes
        ref = Reference(layout.reference)
        clusters = S.read_clusters(vf["clusters"])
        ev_idx = Index(layout, "evidence", {}).data
        feats, kept_l, adm = [], [], []
        human = []
        for rec, r in recs:
            if rec["admission"] in (MK.WHOLE, MK.MASKED):
                ids = [crop_of[(r["key"], b)] for b in rec["keep"] if (r["key"], b) in crop_of]
                ids = [i for i in ids if np.isfinite(np.asarray(X[i], dtype=np.float32)).all()]
                Xu = _unit(np.asarray(X[ids], dtype=np.float32)) if ids else np.zeros((0, X.shape[1]), np.float32)
                feats.append((Xu, [int(crops.label[i]) for i in ids]))
                kept_l.append([rec["labels"][b] for b in rec["keep"]])
                adm.append((rec, r))
            elif rec["admission"] == MK.REFUSED_OVERLAP:
                human.append({"key": r["key"], "batch": bid, "source": r["source"], "image": r["image"],
                              "reason": "refused_overlap", "labels": rec["labels"], "verdicts": rec["verdicts"],
                              "overlaps": rec["overlaps"], "boxes": r["boxes"], "utc": _utc()})
        places = _place_rows(ref, feats, kept_l) if adm else []
        qrows = []
        for (rec, r), pl in zip(adm, places):
            kept = [r["boxes"][b] for b in rec["keep"]]
            if rec["admission"] == MK.WHOLE:
                label, lsha = pool[r["key"]]["label"], pool[r["key"]]["label_sha256"]
                image, sha, dm, frac = r["image"], r["sha256"], None, 0.0
                c = clusters.get(r["key"]) or {}
                score = float(c["score"]) if c.get("score") not in (None, "") else None
            else:
                label, lsha = _label_file(layout, r["source"], r["key"], kept)
                image, sha = rec["mask_record"]["path"], rec["mask_record"]["sha256"]
                dm, frac = rec.get("dhash_masked"), rec["mask_record"]["masked_area_frac"]
                score = pl["score"]
            sp = [0] * V.OTHER
            for b in kept:
                if int(b[0]) < V.OTHER:
                    sp[int(b[0])] += 1
            holds, dl = row_holds(r, extra.get(r["key"], ()))
            qrows.append({"format": QUEUE_FORMAT, "key": r["key"], "batch": bid, "source": r["source"], "group": None,
                          "capture_group": "", "l1": pl["l1"], "l2": pl["l2"], "kind": pl["kind"], "score": score,
                          "typicality": pl["typicality"], "species_boxes": sp,
                          "other_boxes": sum(1 for b in kept if int(b[0]) == V.OTHER), "admission": rec["admission"],
                          "n_masked": len(rec["mask"]), "evidenced": int(ev_idx.get(r["source"], 0)) >= 1,
                          "lab_group": r["lab_group"], "licence": r["licence"],
                          "research_only": bool(r["research_only"]),
                          "prior": priors.get(r["key"]), "hold_until": _hold_until(holds), "holds": holds,
                          "hold_deadline": dl, "verifier": VERIFIER_VERSION, "reference": REFERENCE_VERSION,
                          "admitted_utc": _utc(),
                          "stream_pins_sha": pins_sha, "image": image, "label": label, "sha256": sha,
                          "label_sha256": lsha, "session": "", "unmasked_image": r["image"],
                          "unmasked_sha256": r["sha256"], "dhash": int(r["dhash"]), "dhash_masked": dm,
                          "masked_area_frac": frac, "input": "v1", "supersedes": None, "scanned": r["scanned"],
                          "h6_reason": ("lab_evidence" if r["lab_evidence"] else "not_scanned")
                          if "h6_scan" in holds else None})
        # reconciliation with the funnel census
        crec = [rec for rec, _r in recs if rec["candidate"]]
        n_adm = collections.Counter(rec["admission"] for rec in crec)
        lost = collections.Counter()
        recovered = collections.Counter()
        for rec in crec:
            for b, (l, v) in enumerate(zip(rec["labels"], rec["verdicts"])):
                if l < V.OTHER and v == V.VERIFIED:
                    lost[C.CLASS_NAMES[l]] += 1
                    if rec["admission"] == MK.MASKED and b in rec["keep"]:
                        recovered[C.CLASS_NAMES[l]] += 1
        census = _census(census_path or layout.inc_dir / "funnel" / "census_v1.json")
        if census is None and require_census:
            raise StreamError("census_v1.json (the funnel's F3 census) is missing or has no veto section: b0000 "
                              "must reconcile with it (pass --no-census to record the check as not done)")
        checks = {"accounted": sum(n_adm.values()) == len(cand) and all(
                      a in MK.ADMISSIONS for a in n_adm) and n_adm.get(MK.WHOLE, 0) == 0,
                  "recovered_le_lost": all(recovered[k] <= lost[k] for k in recovered)}
        if census is not None:
            checks["images_match_census"] = len(cand) == census["images"]
            checks["boxes_match_census"] = sum(lost.values()) == census["lost_boxes"]
            checks["recovered_le_census"] = all(recovered[k] <= census["lost_per_class"].get(k, 0) for k in recovered)
            checks["lost_per_class_match_census"] = all(
                lost.get(k, 0) == v for k, v in census["lost_per_class"].items()) and all(
                census["lost_per_class"].get(k, 0) == v for k, v in lost.items())
        recon = {"candidates": len(cand), "whole": n_adm.get(MK.WHOLE, 0), "masked": n_adm.get(MK.MASKED, 0),
                 "refused_overlap": n_adm.get(MK.REFUSED_OVERLAP, 0), "not_admitted": n_adm.get(MK.NOT_ADMITTED, 0),
                 "not_admitted_reasons": dict(collections.Counter(rec["refusal"] or rec["image_verdict"]
                                                                  for rec in crec
                                                                  if rec["admission"] == MK.NOT_ADMITTED)),
                 "lost_boxes": sum(lost.values()), "lost_per_class": dict(sorted(lost.items())),
                 "recovered_boxes": sum(recovered.values()), "recovered_per_class": dict(sorted(recovered.items())),
                 "census": census, "checks": checks,
                 "contract_images": {"contract": CONTRACT_VETO_IMAGES, "got": len(cand), "recorded_not_asserted": True},
                 "ok": all(checks.values())}
        _write_json(bdir / "reconciliation.json", recon)
        if not recon["ok"]:
            raise StreamError("b0000 does not reconcile (%s); see %s. Nothing was committed"
                              % ({k: v for k, v in checks.items() if not v}, bdir / "reconciliation.json"))
        _write_jsonl(bdir / "admission.jsonl", [rec for rec, _r in recs])
        adm_all = collections.Counter(rec["admission"] for rec, _r in recs)
        per_src, per_ref = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        for r in rows:
            per_src[r["source"]][r["decision"]] += 1
        for rec, _r in recs:
            if rec["refusal"] and not rec["refusal"] == _r["decision"]:
                per_ref[rec["source"]][rec["refusal"]] += 1
        doc = {"format": FORMAT, "kind": KIND_BACKFILL, "input": "v1", "rule": "box",
               "per_source_decisions": {k: dict(v) for k, v in sorted(per_src.items())},
               "per_source_refusals": {k: dict(v) for k, v in sorted(per_ref.items())},
               "items": len(rows), "decisions": dict(sorted(collections.Counter(r["decision"] for r in rows).items())),
               "admission": dict(sorted(adm_all.items())), "reconciliation": recon, "base": base_rec,
               "priors": prior_rec, "licences": lic_rec, "consumed_keys": n_cons, "guard": guards.record,
               "scanner": scanner.record(), "scanned": {w: len(rs) for w, rs in sorted(by_which.items())},
               "eval_hits": eval_rec, "embeddings": emb if isinstance(emb, dict) else None,
               # b0000 writes no ingest.jsonl: each hit's match and pair cosine are kept here
               "eval_hit_pairs": [{"key": r["key"], "source": r["source"], "decision": r["decision"],
                                   "match": r.get("guard_match"), "pair_cos": r.get("pair_cos"),
                                   "pair_cos_why": r.get("pair_cos_why")}
                                  for r in rows if r["decision"] in EH.DHASH_HIT_REASONS],
               "inputs": dict(_v1_records(), increment_pool=_file_rec(vf["increment_pool"]),
                              base_selected=_file_rec(vf["base_selected"]), clusters=_file_rec(vf["clusters"]),
                              select_summary=_file_rec(vf["select_summary"]), reference=_file_rec(layout.reference)),
               "seconds": round(time.time() - t0, 1)}
        prepare_commit(layout, bid, bdir, doc, qrows, human, [], {}, 0)
        return finish_commit(layout, bid)


# -------------------------------------------------------------- knowntruth
def list_knowntruth(layout, sets=("tsw22", "tsw23"), cap=BATCH_CAP):
    """The v1 registry images that are copies of expert-labelled base rows
    (from the v1 dHash caches, within 6 bits of an index entry of `sets`):
    for 3SeasonWeedDet10, the 1,055 former near-eval images (copies of
    ood22/ood23, now base copies). Not yet measured ones only."""
    _check_join()
    ps = _read_json(V.POOL_SUMMARY, None) or {}
    registry = V._load_registry()
    base = Index(layout, "base_expert", {}).data
    idx = NearHashIndex()
    for h, v in base.items():
        if v[0] in sets:
            idx.add(int(h), tuple(v), max_bits=BASE_COPY_BITS)
    processed = Index(layout, "processed", {}).data
    items, refusals, taken = [], {}, 0
    for slug in sorted(ps.get("per_slug") or {}):
        info = registry.get(slug)
        root = V._resolve_dir(info) if isinstance(info, dict) else None
        parts = V._layout(root) if root is not None else None
        if parts is None:
            refusals[slug] = "missing_dir"
            continue
        to_inc, src_names, wildcard = V.class_join(slug, info)
        cache = _read_json(V.CACHE_DIR / "dhash" / ("%s.json" % V._sanitise(slug)), {}) or {}
        done = set(processed.get("knowntruth:%s" % slug, []))
        for split, idir, ldir in parts:
            for name in sorted(os.listdir(idir)):
                if not V._is_image(name) or taken >= cap:
                    continue
                stem = os.path.splitext(name)[0]
                rel = "%s/images/%s" % (split, name) if split else "images/%s" % name
                c = cache.get(rel)
                if rel in done or not c or c[2] is None or idx.find(int(c[2])) is None:
                    continue
                lp = Path(ldir) / (stem + ".txt")
                if not lp.is_file():
                    continue
                boxes, src, drop = _read_source_boxes(lp, to_inc, wildcard)
                if drop is not None:
                    continue
                items.append({"input": "knowntruth:%s" % slug, "item": rel, "source": slug, "split": split,
                              "stem": stem, "rel": rel, "image": str(Path(idir) / name), "boxes": boxes,
                              "src": [[s[0], "" if wildcard else src_names.get(s[0], "")] for s in src],
                              "drop": None, "licence": None, "research_only": None, "lab_group": None,
                              "lab_evidence": False, "eval_copies": 0, "provenance_cleared": True,
                              "capture_group": "", "session": "", "declared_sha256": None,
                              "declared_label_sha256": None, "intake_key": None, "join_version": 0})
                taken += 1
    return items, refusals, {"sets": list(sets)}


def run_knowntruth(layout, embedder, guards, sets=("tsw22", "tsw23"), procs=1, batch=V.BATCH,
                   chunk=CHUNK_IMAGES, canary=True):
    """R1 `knowntruth`: a batch that queues nothing and measures the frozen
    verifier's precision on base copies box-matched to expert labels, with its
    Wilson lower bound (the first reading outside the cwd12 capture domain)."""
    spec = "knowntruth:%s" % ",".join(sets)
    return run_batch(layout, spec, lambda: list_knowntruth(layout, sets), embedder, guards, None, "box", procs,
                     batch, chunk, kind=KIND_KNOWNTRUTH, canary=canary)


# ------------------------------------------------------------------ rejoin
def resolutions(layout, slug, domain_path=None):
    """({src id: INC id}, provenance) from the funnel's recorded resolutions
    only: the accepted class maps recover recorded (step1_r1/recovery.json
    class_maps, accepted: card + geometry, H3a supported, class gate) and the
    name_status_v2.json target synonyms recover itself applies (via scientific
    or override). Two different targets for one id refuse."""
    from ..funnel import domain as D
    from ..funnel import recover as R
    dom = D.load(str(domain_path) if domain_path else "weed")
    out, via = {}, {}
    rp = layout.inc_dir / FUNNEL_DOMAIN_DEV.parent / "recovery.json"
    rec = _read_json(rp, {}) or {}
    for m in rec.get("class_maps") or []:
        if m.get("source") == slug and m.get("accepted") and m.get("map_to"):
            out[str(m["src_id"])] = int(dom.class_id(m["map_to"]))
            via[str(m["src_id"])] = "accepted_map"
    np_ = layout.inc_dir / "funnel" / "name_status_v2.json"
    ns = _read_json(np_, {}) or {}
    for n in ns.get("names") or []:
        if n.get("source") != slug or n.get("status_v2") != "target_synonym" or n.get("via") not in ("scientific",
                                                                                                   "override"):
            continue
        t = R._target_of_taxon(n.get("taxon"), dom)
        if t is None:
            continue
        sid = str(n["src_id"])
        if sid in out and out[sid] != int(t):
            raise StreamError("%s id %s: the recorded resolutions disagree (%s vs %s)" % (slug, sid, out[sid], t))
        out.setdefault(sid, int(t))
        via.setdefault(sid, "target_synonym")
    return out, {"recovery": _file_rec(rp), "name_status_v2": _file_rec(np_), "via": via,
                 "domain_config": {"path": str(dom.path), "sha256": dom.sha256}}


def rejoin(layout, slug, guards, scanner=None, domain_path=None, v1_embeddings=None, procs=1):
    """The versioned `rejoin --slug` event (contract §3.3): relabel the slug's
    boxes whose source id has a recorded resolution, re-judge those boxes from
    their stored embeddings (no new embedding), re-admit per box, and queue the
    result as <key>__rj<version> rows that supersede the old ones. Later
    registry batches of the slug use the override table.

    A relabelled image is a new queue row, so it passes what every new row
    passes: base B's images (base_v2 and the L-5 drops) are never queued;
    GuardV2 (never-train v2 with the 8 variants, base copies) on the unmasked
    image, and on the masked copy; near_consumed; an image the stream refused
    for any reason but a supersession stays refused. Every live row of the
    image's chain (the original and earlier rejoins) is superseded, so one
    image is never queued twice. Holds carry over (funnel_F9 and any other
    the old row still has; h6_scan and licence are decided anew, join_conflict
    is what a rejoin resolves), and every row relabelled from the v1 pool
    holds funnel_F9 (P8: the funnel's recovery arms use these relabels; F9's
    domain_dev list refuses or releases them in serve-holds)."""
    scanner = scanner or CopyScanner()
    with writer_lock(layout):
        pins, pins_sha, _c = check_pins(layout, None)
        _finish_pending(layout)
        if pins.get("seeded_from_v1") and BACKFILL_ID not in _ledger_ids(layout):
            raise StreamError("rejoin relabels queued v1 rows: batch %s (the backfill) must be committed first"
                              % BACKFILL_ID)
        spec = "rejoin:%s" % slug
        ip = _in_progress(layout)
        if ip is not None and (_read_json(layout.batch_dir(ip) / "plan.json", {}) or {}).get("spec") != spec:
            raise StreamError("batch %s is in progress; finish it before a rejoin" % ip)
        maps, prov = resolutions(layout, slug, domain_path)
        if not maps:
            raise StreamError("%s has no recorded resolution (accepted map or target synonym): nothing to rejoin"
                              % slug)
        ov = Index(layout, "overrides", {})
        cur = ov.data.get(slug) or {}
        if cur.get("map") == {k: int(v) for k, v in sorted(maps.items())}:
            log("%s: rejoin version %s already applies this map" % (slug, cur.get("version")))
            return None
        version = int(cur.get("version", 0)) + 1
        bid = ip if ip is not None else _next_bid(layout)
        bdir = layout.batch_dir(bid)
        bdir.mkdir(parents=True, exist_ok=True)
        # planned first: a rejoin killed half-way is resumed (redone) by the same argv, never left blocking
        _write_json(bdir / "plan.json", {"batch": bid, "kind": KIND_REJOIN, "spec": spec, "rule": "box",
                                         "version": version, "created_utc": _utc()})
        if ip is not None:
            log("resuming %s (%s)" % (bid, spec))
        t0 = time.time()
        ver = V.Verifier.load(layout.verifier_dir)
        oof = (_read_json(layout.verifier_dir / "oof_keys.json", {}) or {}).get("keys") or []
        oof_pos = {(k, int(b)): n for n, (k, b) in enumerate(oof)}
        units = []                       # (row, [stored verdicts], crop features by box, crop label ids)
        # (a) the v1 rows of the slug; base B's images (base_v2 and L-5) are never queued
        vf = v1_files()
        crops = V.Crops(vf["crops"])
        pv, _codes = S.read_pool_verdicts(vf["pool_verdicts"], crops)
        lookup = V._box_verdict_lookup(crops, "pool", lambda i: V.VERDICT_CODES[int(pv[i])])
        pool = {r["key"]: r for r in C.read_manifest(vf["pool"])}
        base_b = {r["key"] for r in C.read_manifest(vf["base_selected"])} if Path(vf["base_selected"]).is_file() \
            else None
        if base_b is None and pins.get("seeded_from_v1"):
            raise StreamError("%s is missing: base B's images could not be kept out of the queue" % vf["base_selected"])
        base_b = base_b or set()
        metas = [m for m in V._read_jsonl(vf["pool_meta"]) if m["source"] == slug]
        n_base = sum(1 for m in metas if m["key"] in base_b)
        metas = [m for m in metas if m["key"] not in base_b]
        X1 = None
        if metas:
            X1, _emb = v1_embeddings if v1_embeddings is not None else load_v1_embeddings(crops)
        for m in metas:
            vs, fx = [], {}
            for b in range(len(m["boxes"])):
                v, i = lookup(m["key"], b)
                vs.append(v)
                if i is not None:
                    fx[b] = ("v1", int(i))
            r = pool[m["key"]]
            units.append(({"key": m["key"], "source": slug, "image": r["image"], "sha256": r["sha256"],
                           "dhash": int(m["dhash"]), "boxes": [list(b) for b in m["boxes"]], "src": m["src"],
                           "input": "v1"}, vs, fx))
        # (b) the stream's own rows of the slug
        for b_id in _ledger_ids(layout):
            bd = layout.batch_dir(b_id)
            if not (bd / "ingest.jsonl").exists() or not (bd / "crops.csv").exists():
                continue
            brows = [r for r in _read_jsonl(bd / "ingest.jsonl") if r["source"] == slug and r["decision"] == "pool"]
            if not brows:
                continue
            bc = V.Crops(bd / "crops.csv")
            got = _load_verdicts(bd, bc)
            if got is None:
                raise StreamError("%s: verdicts.npz does not describe crops.csv" % bd)
            skipped = read_skipped(bd)
            cof = {(bc.key[i], int(bc.box[i])): int(i) for i in range(bc.n)}
            for r in brows:
                vs = _box_verdicts(r, cof, skipped, got[0])
                fx = {b: (b_id, cof[(r["key"], b)]) for b in range(len(r["boxes"])) if (r["key"], b) in cof}
                units.append((dict(r, input="%s:%s" % (b_id, r["input"])), vs, fx))
        feats_cache = {}

        def feature(src, i):
            if src == "v1":
                return np.asarray(X1[i], dtype=np.float32)
            if src not in feats_cache:
                with np.load(layout.batch_dir(src) / "emb.npz") as d:
                    feats_cache[src] = d["X"]
            return np.asarray(feats_cache[src][i], dtype=np.float32)

        qrows_all = load_queue(layout)
        chains = collections.defaultdict(list)        # original key -> its queue rows (itself and its rejoins)
        for q in qrows_all:
            chains[q.get("supersedes") or q["key"]].append(q)
        cons_index, _n = consumed_hashes(layout)
        lic, lic_rec = licence_table(layout.inc_dir, domain_path)
        evid = v1_lab_evidence(_read_json(vf["pool_summary"], {}))
        labs = funnel_lab_groups(domain_path)
        ev_idx = Index(layout, "evidence", {}).data
        ref = Reference(layout.reference)
        # the relabelled images, and GuardV2 on each one's unmasked image (all 8 variants, base copies)
        todo = []
        for row, vs, fx in units:
            new = [list(b) for b in row["boxes"]]
            changed = []
            for b, (s_id, _name) in enumerate(row["src"] or []):
                t = maps.get(str(s_id))
                if t is not None and int(new[b][0]) != t:
                    new[b][0] = t
                    changed.append(b)
            if changed:
                todo.append((row, vs, fx, new, changed))
        gres = guards.check_many([(row["image"], row.get("dhash")) for row, _v, _f, _n2, _c2 in todo], procs,
                                 "rejoin guard")
        recs, adm, feats, kept_l, events, human = [], [], [], [], [], []
        for (row, vs, fx, new, changed), (g_reason, g_match, _gh) in zip(todo, gres):
            vs = list(vs)
            for b in changed:
                if b not in fx:
                    continue                        # small / failed: its verdict does not depend on the label
                src, i = fx[b]
                if src == "v1" and (crops.key[i], int(crops.box[i])) in oof_pos:
                    n = oof_pos[(crops.key[i], int(crops.box[i]))]
                    vv = V.verdicts(np.array([new[b][0]]), ver.arrays["other_oof_P"][n:n + 1],
                                    ver.arrays["other_oof_cos"][n:n + 1], ver.tau_p, ver.sigma)[0][0]
                    if not bool(ver.arrays["other_oof_usable"][n]):
                        vv = V.UNKNOWN
                else:
                    vv = ver.judge_features(np.array([new[b][0]]), feature(src, i)[None, :])[0][0]
                vs[b] = str(vv)
            d = MK.decide([int(b[0]) for b in new], vs, [b[1:] for b in new])
            nkey = "%s__rj%d" % (row["key"], version)
            rec = {"key": nkey, "supersedes": row["key"], "source": slug, "changed_boxes": changed, "verdicts": vs,
                   "labels": [int(b[0]) for b in new], **d, "refusal": None, "mask_record": None}
            chain = chains.get(row["key"]) or []
            live = [q for q in chain if not q.get("refused")]
            old_refused = [q["refused"] for q in chain if q.get("refused")
                           and not str(q["refused"]).startswith("superseded:")]
            if g_reason:
                rec["admission"], rec["refusal"], rec["guard_match"] = MK.NOT_ADMITTED, g_reason, g_match
            elif old_refused:
                rec["admission"], rec["refusal"] = MK.NOT_ADMITTED, "old_refused:%s" % old_refused[0]
            elif cons_index.find(int(row["dhash"])) is not None:
                rec["admission"], rec["refusal"] = MK.NOT_ADMITTED, "near_consumed"
            mrec = None
            if rec["admission"] == MK.MASKED:
                mrec = _mask_or_refuse(rec, row["image"], [new[i] for i in d["mask"]], [new[i] for i in d["keep"]],
                                       layout.masked / V._sanitise(slug), nkey)
            if mrec is not None:
                rec["mask_record"] = mrec
                reason, match, hm = guards.check(mrec["path"])
                rec["dhash_masked"] = hm
                if reason:
                    rec["admission"], rec["refusal"], rec["guard_match"] = MK.NOT_ADMITTED, "masked_%s" % reason, match
            for q in live:
                events.append({"event": "refuse", "batch": bid, "key": q["key"],
                               "reason": "superseded:rejoin_v%d" % version, "by": nkey, "utc": _utc()})
            if rec["admission"] in (MK.WHOLE, MK.MASKED):
                pairs = [(b, fx[b]) for b in d["keep"] if b in fx]
                xs = [feature(s_, i_) for _b, (s_, i_) in pairs]
                ok = [bool(np.isfinite(x).all()) for x in xs]
                Xu = (_unit(np.stack([x for x, o in zip(xs, ok) if o])) if any(ok)
                      else np.zeros((0, ref.P.shape[1]), np.float32))
                feats.append((Xu, [int(new[b][0]) for (b, _f), o in zip(pairs, ok) if o]))
                kept_l.append([int(new[b][0]) for b in d["keep"]])
                adm.append((rec, dict(row, boxes=new), chain))
            elif rec["admission"] == MK.REFUSED_OVERLAP:
                human.append({"key": nkey, "batch": bid, "source": slug, "image": row["image"],
                              "reason": "refused_overlap", "labels": rec["labels"], "verdicts": vs,
                              "overlaps": rec["overlaps"], "boxes": new, "utc": _utc()})
            recs.append(rec)
        places = _place_rows(ref, feats, kept_l) if adm else []
        qrows = []
        for (rec, row, chain), pl in zip(adm, places):
            kept = [row["boxes"][b] for b in rec["keep"]]
            label, lsha = _label_file(layout, slug, rec["key"], kept)
            if rec["admission"] == MK.WHOLE:
                image, sha, dm, frac = row["image"], row["sha256"], None, 0.0
            else:
                image, sha = rec["mask_record"]["path"], rec["mask_record"]["sha256"]
                dm, frac = rec.get("dhash_masked"), rec["mask_record"]["masked_area_frac"]
            old = chain[-1] if chain else {}
            ev = evid.get(slug)
            if old:
                licence, ro = old.get("licence"), old.get("research_only")
                if licence is None:
                    licence, ro = licence_state(lic.get(slug))
            elif row.get("input") != "v1" and "licence" in row:
                licence, ro = row.get("licence"), row.get("research_only")
            else:
                licence, ro = licence_state(lic.get(slug))
            lab_ev = bool(row.get("lab_evidence")) or bool(ev) or any(q.get("h6_reason") == "lab_evidence"
                                                                      for q in chain)
            r2 = {"provenance_cleared": bool(row.get("provenance_cleared")) and not lab_ev, "scanned": None,
                  "licence": licence, "lab_evidence": lab_ev}
            carry = [h for q in chain if not q.get("refused") or str(q["refused"]).startswith("superseded:")
                     for h in (q.get("holds") or []) if h not in ("h6_scan", "licence", "join_conflict")]
            if row.get("input") == "v1":
                carry.append("funnel_F9")
            holds, dl = row_holds(r2, sorted(set(carry)))
            for h in dl:
                old_dl = (old.get("hold_deadline") or {}).get(h) if old else None
                if old_dl and h in (old.get("holds") or []):
                    dl[h] = old_dl                      # a carried hold keeps its deadline
            sp = [0] * V.OTHER
            for b in kept:
                if int(b[0]) < V.OTHER:
                    sp[int(b[0])] += 1
            qrows.append({"format": QUEUE_FORMAT, "key": rec["key"], "batch": bid, "source": slug, "group": None,
                          "capture_group": old.get("capture_group") or row.get("capture_group") or "",
                          "l1": pl["l1"], "l2": pl["l2"],
                          "kind": pl["kind"], "score": pl["score"], "typicality": pl["typicality"],
                          "species_boxes": sp, "other_boxes": sum(1 for b in kept if int(b[0]) == V.OTHER),
                          "admission": rec["admission"], "n_masked": len(rec["mask"]),
                          "evidenced": int(ev_idx.get(slug, 0)) >= 1,
                          "lab_group": old.get("lab_group") or row.get("lab_group")
                          or ("LuLab" if ev else labs.get(slug)),
                          "licence": licence, "research_only": bool(ro), "prior": old.get("prior"),
                          "hold_until": _hold_until(holds), "holds": holds, "hold_deadline": dl,
                          "verifier": VERIFIER_VERSION, "reference": REFERENCE_VERSION, "admitted_utc": _utc(),
                          "stream_pins_sha": pins_sha,
                          "image": image, "label": label, "sha256": sha, "label_sha256": lsha,
                          "session": old.get("session") or row.get("session") or "", "unmasked_image": row["image"],
                          "unmasked_sha256": row["sha256"], "dhash": int(row["dhash"]), "dhash_masked": dm,
                          "masked_area_frac": frac, "input": "rejoin_v%d:%s" % (version, row["input"]),
                          "supersedes": row["key"], "scanned": None,
                          "h6_reason": ("lab_evidence" if r2["lab_evidence"] else "not_scanned")
                          if "h6_scan" in holds else None})
        _write_jsonl(bdir / "admission.jsonl", recs)
        per_src = collections.Counter(("refused:%s" % r["refusal"]) if r["refusal"] else r["admission"] for r in recs)
        doc = {"format": FORMAT, "kind": KIND_REJOIN, "input": "rejoin:%s:v%d" % (slug, version), "rule": "box",
               "slug": slug, "version": version, "map": {k: int(v) for k, v in sorted(maps.items())},
               "provenance": prov, "units": len(units), "relabelled": len(recs), "base_b_left_out": n_base,
               "admission": dict(collections.Counter(r["admission"] for r in recs)),
               "refusals_after_admission": dict(collections.Counter(r["refusal"] for r in recs if r["refusal"])),
               "per_source_rejoin": {slug: dict(per_src)}, "guard": guards.record, "licences": lic_rec,
               "seconds": round(time.time() - t0, 1)}
        changes = {"overrides": {slug: {"version": version, "map": doc["map"], "provenance": prov}}}
        prepare_commit(layout, bid, bdir, doc, qrows, human, events, changes, 0)
        return finish_commit(layout, bid)


# -------------------------------------------------------- the queue, read
def eligible_step1(row):
    """Step 1's part of a queue row's eligibility (contract §3.4): not refused,
    evidenced, no hold left, and at least one kept verified target box. The
    cutter adds consumption, quarantine and the verifier version."""
    return (not row.get("refused") and bool(row.get("evidenced")) and not row.get("holds")
            and sum(row.get("species_boxes") or []) >= 1)


def load_queue(layout):
    """The queue as the cutter reads it: queue.jsonl rows with the events
    folded in (holds released or added, refusals, near-dup groups merged into
    their current root), evidenced from the current evidence index, and
    eligible_step1. Rows keep the file order."""
    rows = _read_jsonl(layout.queue)
    by = {}
    for r in rows:
        if r["key"] in by:
            raise StreamError("queue.jsonl holds key %s twice" % r["key"])
        by[r["key"]] = dict(r, holds=list(r.get("holds") or []), refused=None, released={})
    for e in _read_jsonl(layout.events):
        row = by.get(e.get("key"))
        if row is None:
            continue
        ev = e.get("event")
        if ev == "release" and e.get("hold") in row["holds"]:
            row["holds"].remove(e["hold"])
            row["released"][e["hold"]] = {k: e.get(k) for k in ("batch", "utc", "calibration", "reason")}
            if e.get("hold") == "licence" and e.get("licence"):
                row["licence"], row["research_only"] = e["licence"], bool(e.get("research_only"))
        elif ev == "hold" and e.get("hold") not in row["holds"]:
            row["holds"].append(e["hold"])
        elif ev == "refuse" and not row["refused"]:
            row["refused"] = e.get("reason")
    groups = Groups(Index(layout, "groups", {}).data)
    evidence = Index(layout, "evidence", {}).data
    out = []
    for r in rows:
        row = by[r["key"]]
        row["holds"] = [h for h in HOLD_ORDER if h in row["holds"]] + [h for h in row["holds"] if h not in HOLD_ORDER]
        row["hold_until"] = _hold_until(row["holds"])
        if row.get("group") is not None:
            row["group"] = groups.find(row["group"])
        row["evidenced"] = int(evidence.get(row["source"], 0)) >= 1
        row["eligible_step1"] = eligible_step1(row)
        out.append(row)
    return out


# -------------------------------------------------------------- serve holds
def serve_holds(layout, scanner=None, domain_path=None, domain_dev_path=None, kinds=None):
    """Release or refuse held queue rows (contract §3.9, §6.7):
      licence    released once the funnel's cards or config record a licence;
      funnel_F9  when F9 wrote step1_r1/domain_dev.jsonl, every row whose image
                 is a domain-dev image or within 3 bits of one is refused for
                 good (the H10d hold-out), and the rest are released;
      h6_scan    the copy detector scans the unmasked and the masked image: a
                 hit refuses the row, a clean scan releases it. A same-lab row
                 (P9) is served by the funnel's calibration, or by the stream's
                 own once its hold deadline passed.
    A funnel_F9 hold past its deadline is listed for a person (R3), never
    released here. kinds (the job's --hold) limits which holds are served;
    a domain-dev image is refused whatever kinds says (a refusal is always
    safe). Appends events and a hash-chained line to ledger/holds.jsonl."""
    scanner = scanner or CopyScanner()
    kinds = set(kinds) if kinds else set(DEADLINE_HOLDS) | {"licence"}
    bad = sorted(kinds - {"h6_scan", "licence", "funnel_F9"})
    if bad:
        raise StreamError("serve-holds serves h6_scan, licence and funnel_F9, not %s" % bad)
    with writer_lock(layout):
        check_pins(layout, None)
        run = "serve-%s" % _utc()
        q = [r for r in load_queue(layout) if not r["refused"]]
        lic, lic_rec = licence_table(layout.inc_dir, domain_path)
        ddp = Path(domain_dev_path or layout.inc_dir / FUNNEL_DOMAIN_DEV)
        dd_keys, dd_index = None, NearHashIndex()
        if ddp.is_file():
            dd_keys = {r["key"] for r in C.read_manifest(ddp)}
            meta = {m["key"]: m for m in V._read_jsonl(V.POOL_META)} if Path(V.POOL_META).is_file() else {}
            for k in dd_keys:
                h = (meta.get(k) or {}).get("dhash")
                if h is not None:
                    dd_index.add(int(h), k, max_bits=GROUP_BITS)
        today = _date(0)
        events, scan_todo, past = [], collections.defaultdict(list), collections.Counter()

        def ev(kind, row, **kw):
            events.append(dict({"event": kind, "batch": run, "key": row["key"], "utc": _utc()}, **kw))
        for row in q:
            base_key = row.get("supersedes") or row["key"]
            if dd_keys is not None and (base_key in dd_keys or row["key"] in dd_keys
                                        or dd_index.find(int(row["dhash"])) is not None):
                ev("refuse", row, reason="domain_dev", evidence=_file_rec(ddp))
                continue
            if "funnel_F9" in row["holds"] and "funnel_F9" in kinds:
                if dd_keys is not None:
                    ev("release", row, hold="funnel_F9", reason="F9 wrote domain_dev.jsonl", evidence=_file_rec(ddp))
                elif (row.get("hold_deadline") or {}).get("funnel_F9") and today > row["hold_deadline"]["funnel_F9"]:
                    past["funnel_F9"] += 1
            if "licence" in row["holds"] and "licence" in kinds:
                l, ro = licence_state(lic.get(row["source"]))
                if l is not None:
                    ev("release", row, hold="licence", licence=l, research_only=bool(ro), reason="licence recorded")
            if "h6_scan" in row["holds"] and "h6_scan" in kinds:
                dl = (row.get("hold_deadline") or {}).get("h6_scan")
                passed = dl is not None and today > dl
                w = scanner.which(row.get("h6_reason") == "lab_evidence", passed)
                if w is not None:
                    scan_todo[w].append(row)
                elif passed:
                    past["h6_scan"] += 1
        scanned = {}
        for w, rows in sorted(scan_todo.items()):
            items = []
            for row in rows:
                items.append({"key": row["key"], "path": row["unmasked_image"]})
                if row["image"] != row["unmasked_image"]:
                    items.append({"key": row["key"] + "#masked", "path": row["image"]})
            res = scanner.scan(items, w)
            for row in rows:
                hit = res.get(row["key"]) or res.get(row["key"] + "#masked")
                if hit is not None:
                    ev("refuse", row, reason=scan_reason(hit), hit=hit)
                else:
                    ev("release", row, hold="h6_scan", calibration=(scanner.record() or {}).get(w),
                       reason="copy scan clean")
            scanned[w] = len(rows)
        _append_jsonl(layout.events, events)
        summary = {"run": run, "kinds": sorted(kinds),
                   "released": dict(collections.Counter(e["hold"] for e in events if e["event"] == "release")),
                   "refused": dict(collections.Counter(e["reason"] for e in events if e["event"] == "refuse")),
                   "scanned": scanned, "past_deadline": dict(past), "domain_dev": _file_rec(ddp),
                   "licences": lic_rec, "scanner": scanner.record(), "utc": _utc()}
        ledger_append(layout.root / "ledger" / "holds.jsonl", summary)
        write_status(layout)
    log("serve-holds: released %s, refused %s, past deadline %s" % (summary["released"], summary["refused"],
                                                                     summary["past_deadline"]))
    return summary


# ------------------------------------------------------------------ status
STATUS_KEYS = {"format": str, "built_utc": str, "stream_version": int, "versions": dict, "pending": dict,
               "batches": dict, "crop_ids": dict, "queue": dict, "admission": dict, "holds": dict,
               "holds_past_deadline": dict, "refused": dict, "knowntruth": dict, "refit_triggers": dict,
               "human_queue": dict, "per_source": dict, "one_time": dict}


def check_status(doc):
    """[] or the problems of a status.json (its schema)."""
    probs = []
    for k, t in STATUS_KEYS.items():
        if k not in doc:
            probs.append("missing %s" % k)
        elif not isinstance(doc[k], t):
            probs.append("%s is %s, not %s" % (k, type(doc[k]).__name__, t.__name__))
    if not probs:
        if doc["format"] != STATUS_FORMAT:
            probs.append("format %r" % doc["format"])
        for k in ("committed", "in_progress", "by_kind"):
            if k not in doc["batches"]:
                probs.append("batches.%s missing" % k)
        for k in ("v1", "next"):
            if not isinstance(doc["crop_ids"].get(k), int):
                probs.append("crop_ids.%s is not an int" % k)
        for k in ("rows", "eligible_target_images", "admitted_images", "per_species_boxes", "kind"):
            if k not in doc["queue"]:
                probs.append("queue.%s missing" % k)
        for k in ("masked", "refused_overlap", "whole", "not_admitted"):
            if not isinstance(doc["admission"].get(k), int):
                probs.append("admission.%s is not an int" % k)
        for k in ("bootstrap", "backfill", "knowntruth"):
            if k not in doc["one_time"]:
                probs.append("one_time.%s missing" % k)
        for src, row in doc["per_source"].items():
            for k in ("images_seen", "near_eval_embed", "target_boxes_admitted"):
                if not isinstance((row or {}).get(k, 0), int):
                    probs.append("per_source.%s.%s is not an int" % (src, k))
    return probs


# ----------------------------------------------------- D28-v2 sidecars
EVAL_HITS_DIR = "eval_hits"          # step1_stream/eval_hits/<batch>.json


def eval_hits_sidecar_path(layout, bid):
    return layout.root / EVAL_HITS_DIR / ("%s.json" % bid)


def dhash_counts(doc):
    """{source: dHash hits on evaluation images} a batch record counted (its
    per_source_decisions near_eval_v2 + near_eval_variant)."""
    out = {}
    for src, dec in ((doc or {}).get("per_source_decisions") or {}).items():
        n = sum(int((dec or {}).get(k) or 0) for k in EH.DHASH_HIT_REASONS)
        if n > 0:
            out[str(src)] = n
    return out


def _weighs_all(rec, counts):
    """True when an eval_hits record holds a pair cosine for every dHash hit
    the batch counted, source by source."""
    per = (rec or {}).get("per_source") if isinstance((rec or {}).get("per_source"), dict) else {}
    return all(len([c for c in ((per.get(s) or {}).get("pair_cos") or []) if c is not None]) >= n
               for s, n in counts.items())


def _sidecar_rows(layout, bid, doc, guards):
    """(rows in ingest's form for the dHash hits of a committed batch, {row key:
    why it cannot be weighed}, the files read). b0000 (no ingest.jsonl): the
    rows admission.jsonl records refused near_eval_v2 or near_eval_variant,
    each image and dHash from the v1 pool manifest and dHash cache (the image
    checked against the manifest's sha256), the guard's match decided again by
    GuardV2 on them (the same pinned LOCK, so the same decision: one that
    differs is recorded, never weighed). A later batch: its ingest.jsonl
    (checked against the sha256 its batch.json records), each image checked
    against the row's sha256 and decided again the same way."""
    bdir = layout.batch_dir(bid)
    rows, failed, inputs = [], {}, {}

    def again(r, image, dhash, sha):
        k = "%s|%s" % (r["input"], r["item"])
        if not image or not Path(image).is_file():
            failed[k] = "the image %s is gone" % image
        elif sha and _sha_file(image) != sha:
            failed[k] = "the image no longer hashes to the recorded sha256"
        else:
            reason, match, _h = guards._decide(image, int(dhash), guards.variants_fn(image))
            mm, rm = (match if isinstance(match, dict) else {}), (r.get("guard_match")
                                                                 if isinstance(r.get("guard_match"), dict) else None)
            if reason != r["decision"] or (rm is not None and (mm.get("split"), mm.get("key"))
                                           != (rm.get("split"), rm.get("key"))):
                failed[k] = "GuardV2 now decides %s (%s:%s), the batch recorded %s" % (
                    reason, mm.get("split"), mm.get("key"), r["decision"])
            else:
                r["guard_match"] = match
        rows.append(r)

    if doc.get("kind") == KIND_BACKFILL:
        vf = v1_files()
        adm = bdir / "admission.jsonl"
        inputs = {"admission": _file_rec(adm), "pool": _file_rec(vf["pool"]), "pool_meta": _file_rec(vf["pool_meta"])}
        hit = [a for a in _read_jsonl(adm) if a.get("refusal") in EH.DHASH_HIT_REASONS]
        keys = {a["key"] for a in hit}
        pool = {r["key"]: r for r in C.read_manifest(vf["pool"]) if r["key"] in keys}
        meta = {m["key"]: m for m in V._read_jsonl(vf["pool_meta"]) if m.get("key") in keys}
        for a in hit:
            pr, m = pool.get(a["key"]) or {}, meta.get(a["key"]) or {}
            r = {"input": "v1", "item": a["key"], "key": a["key"], "source": a.get("source"), "image": pr.get("image"),
                 "decision": a["refusal"], "guard_match": None}
            if m.get("dhash") is None:
                failed["v1|%s" % a["key"]] = "the v1 pool manifest or dHash cache lacks it"
                rows.append(r)
                continue
            again(r, pr.get("image"), m["dhash"], pr.get("sha256"))
        return rows, failed, inputs
    ing = bdir / "ingest.jsonl"
    want = ((doc.get("inputs") or {}).get("ingest") or {}).get("sha256")
    inputs = {"ingest": _file_rec(ing)}
    if not ing.is_file() or (want and _sha_file(ing) != want):
        return rows, {"*": "%s is missing or does not hash to the sha256 batch.json records" % ing}, inputs
    for x in _read_jsonl(ing):
        if x.get("decision") in EH.DHASH_HIT_REASONS:
            r = {"input": x.get("input"), "item": x.get("item"), "key": x.get("key"), "source": x.get("source"),
                 "image": x.get("image"), "decision": x["decision"], "guard_match": x.get("guard_match")}
            if x.get("dhash") is None:
                failed["%s|%s" % (r["input"], r["item"])] = "its ingest row holds no dHash"
                rows.append(r)
                continue
            again(r, x.get("image"), x["dhash"], x.get("sha256"))
    return rows, failed, inputs


def step1_eval_hits(layout, bid, doc, sha, guards, scanner, procs=1):
    """D28-v2's sidecar of one committed Step 1 batch whose batch.json does not
    weigh its dHash hits (committed before the amendment):
    step1_stream/eval_hits/<batch>.json, the batch's hits re-derived
    (_sidecar_rows) and weighed by the copy scanner's embedder and evaluation
    descriptors (score_eval_hits, with every evaluation image within the
    radius). doc: the batch.json read; sha: the sha256 of the bytes it was
    parsed from (recorded as batch_json_sha256, which write_status compares
    with the ledger's). batch.json is never rewritten (the stream ledger
    hash-locks it): write_status folds the sidecar in its place. Returns the
    sidecar."""
    rows, failed, inputs = _sidecar_rows(layout, bid, doc, guards)
    why_all = failed.pop("*", None)          # the batch's rows cannot be read at all: nothing is weighed
    ok = [r for r in rows if "%s|%s" % (r["input"], r["item"]) not in failed]
    rec0 = score_eval_hits(ok, scanner, procs, guards) if ok else {}
    thr, cal = copy_threshold(scanner)
    items, merged = [], {}
    for r in rows:
        k = "%s|%s" % (r["input"], r["item"])
        items.append({"key": k, "source": r.get("source"), "reason": r["decision"], "match": r.get("guard_match")})
        if k in failed:
            merged[k] = {"pair_cos": None, "why": failed[k]}
        elif r.get("pair_cos") is not None:
            m = r.get("guard_match") if isinstance(r.get("guard_match"), dict) else {}
            merged[k] = {"pair_cos": r["pair_cos"], "why": None, "weighed": r.get("pair_cos_weighed") or 1,
                         "best": r.get("pair_cos_best") or [m.get("split"), m.get("key")]}
        else:
            merged[k] = {"pair_cos": None, "why": r.get("pair_cos_why") or rec0.get("why") or "not weighed"}
    name = rec0.get("embedder") or getattr(getattr(getattr(scanner, "index", None), "embedder", None), "name", None)
    rec = EH.record(items, merged, embedder_name=name, copy_threshold=thr, calibration=cal,
                    why=why_all or (None if ok else (rec0.get("why") if rec0 else None)))
    side = EH.sidecar(bid, rec, EH.sidecar_pairs(items, merged), "step1", batch_json_sha256=sha, built_utc=_utc(),
                      batch_kind=doc.get("kind"), counted=dhash_counts(doc), inputs=inputs,
                      scanner=scanner.record() if scanner is not None else None, guard=guards.record,
                      modules=_module_hashes())
    _write_json(eval_hits_sidecar_path(layout, bid), side)
    log("  %s: %d dHash hit(s) weighed again into %s, %d with a pair cosine (max %s)"
        % (bid, rec["hits"], eval_hits_sidecar_path(layout, bid), rec["scored"],
           max((c for v in rec["per_source"].values() for c in v["pair_cos"]), default="n/a")))
    return side


def _collect_config():
    """The collector's domain config (collect.__main__.default_config), which
    collect.intake.rescore_eval_hits reads a fetch's format options from."""
    from ..collect import config as CF
    from ..collect.__main__ import default_config
    return CF.load(default_config())


def eval_hits(layout, guards, scanner, bids=None, intakes=None, procs=1, force=False, lock_path=None, cfg=None):
    """step1_stream eval-hits (D28-v2; docs/CONTINUOUS_LOOP.md, amendment
    2026-10-03; lever L17 verb eval-hits): weigh again, into sidecars, the
    dHash hits on evaluation images of every committed batch whose own
    record does not weigh them, so that D28 judges its source by the
    amendment's rule instead of the one-hit fallback:
      * each Step 1 batch (b0000 and later; known-truth batches are not a
        source's supply) -> step1_stream/eval_hits/<batch>.json
        (step1_eval_hits), folded per source into status.json by
        write_status;
      * each intake batch -> intake/<batch>/eval_hits.json
        (collect.intake.rescore_eval_hits, given this job's embedder,
        evaluation descriptors and GuardV2), which the snapshot ships and D28
        reads.
    bids / intakes: only these (None: every batch that needs it; bids []:
    no Step 1 batch; intakes False: no intake batch). A batch that already
    has a sidecar is skipped unless force. Without a copy scanner index no
    Step 1 sidecar is written (it would weigh nothing), nor for a batch.json
    that no longer hashes to the sha256 the ledger commits (it would never be
    folded): those batches are "failed". Writes the
    one_time index's "eval_hits" (what ran, when) and status.json. Returns
    {"step1": {bid: result}, "intake": {batch: result}}; the CLI exits 2
    when any batch is "failed"."""
    out = {"step1": {}, "intake": {}}
    index = getattr(scanner, "index", None) if scanner is not None else None
    with writer_lock(layout):
        for e in verify_ledger(layout.ledger):
            bid = e["batch"]
            if bids is not None and bid not in bids:
                continue
            # the sidecar names the batch.json it was made from by the sha256 of the very bytes read here
            # (write_status folds it only when that is the sha256 the ledger commits)
            bj = layout.batch_dir(bid) / "batch.json"
            try:
                raw = bj.read_bytes()
                doc = (json.loads(raw.decode("utf-8")) if raw.strip() else {}) or {}
            except FileNotFoundError:
                raw, doc = None, {}
            except (OSError, ValueError) as ex:
                raise StreamError("unreadable JSON %s (%s)" % (bj, ex))
            counts = dhash_counts(doc)
            if doc.get("kind") == KIND_KNOWNTRUTH or not counts or _weighs_all(doc.get("eval_hits"), counts):
                out["step1"][bid] = {"status": "not_needed"}
                continue
            if eval_hits_sidecar_path(layout, bid).is_file() and not force:
                out["step1"][bid] = {"status": "exists"}
                continue
            if index is None or getattr(index, "embedder", None) is None:
                # a sidecar that weighs nothing would use up the batch's attempt: none is written, the job fails
                out["step1"][bid] = {"status": "failed", "why": "no copy scanner index (no passed calibration "
                                                                "loaded): nothing can be weighed"}
                continue
            sha = _sha_bytes(raw)
            if sha != e["batch_json_sha256"]:
                # a sidecar of a batch.json that changed since commit would never be folded: none is written
                out["step1"][bid] = {"status": "failed", "why": "batch.json hashes to %s, not to the sha256 the "
                                                                "ledger commits (%s): changed since commit"
                                                                % (sha[:12], str(e["batch_json_sha256"])[:12])}
                continue
            side = step1_eval_hits(layout, bid, doc, sha, guards, scanner, procs)
            out["step1"][bid] = {"status": "written", "hits": side["eval_hits"]["hits"],
                                 "scored": side["eval_hits"]["scored"]}
        if intakes is not False:
            d = layout.inc_dir / "intake"
            names = sorted(n for n in (os.listdir(d) if d.is_dir() else []) if (d / n / "summary.json").is_file())
            todo = [n for n in names if intakes is None or n in intakes]
            try:
                from ..collect import intake as CI
                need = [n for n in todo if force or (CI.eval_hits_needed(_read_json(d / n / "summary.json", {}) or {})
                                                     and not (d / n / CI.EVAL_HITS_NAME).is_file())]
                cfg = cfg if cfg is not None or not need else _collect_config()
            except Exception as e:  # noqa: BLE001 - the collector cannot be loaded: its batches wait (fail closed)
                need = []
                for n in todo:
                    out["intake"][n] = {"status": "failed", "why": "the collector does not load (%s: %s)"
                                                                 % (type(e).__name__, str(e)[:200])}
            pos = {(str(sp), str(k)): j for j, (sp, k) in enumerate(zip(index.split, index.eval_key))} \
                if index is not None else {}

            def eval_desc(split, key):
                j = pos.get((split, key))
                return None if j is None else index.Xn[j]
            for n in todo:
                if n in out["intake"]:
                    continue
                if n not in need:
                    out["intake"][n] = {"status": "not_needed"}
                    continue
                try:
                    out["intake"][n] = CI.rescore_eval_hits(
                        cfg, n, inc=layout.inc_dir, lock_path=lock_path, guard=guards.guard,
                        embedder=getattr(index, "embedder", None), eval_desc=eval_desc if index is not None else None,
                        force=force)
                except Exception as e:  # noqa: BLE001 - one batch that cannot be weighed never stops the others
                    out["intake"][n] = {"status": "failed", "why": "%s: %s" % (type(e).__name__, str(e)[:300])}
                log("  intake %s: %s" % (n, out["intake"][n]))
        ot = Index(layout, "one_time", {})
        ot.data["eval_hits"] = {"utc": _utc(), "step1": {k: v.get("status") for k, v in out["step1"].items()},
                                "intake": {k: v.get("status") for k, v in out["intake"].items()}}
        ot.save()
        write_status(layout)
    return out


def _intake_pending(layout):
    out = {}
    d = layout.inc_dir / "intake"
    if not d.is_dir():
        return out
    processed = Index(layout, "processed", {}).data
    for name in sorted(os.listdir(d)):
        p = d / name
        if not (p / "summary.json").is_file() or not (p / "manifest.jsonl").is_file():
            continue
        n = sum(1 for line in open(p / "manifest.jsonl") if line.strip())
        left = n - len(processed.get("intake:%s" % name, []))
        if left > 0:
            out[name] = left
    return out


def write_status(layout):
    """status.json (read by the autopilot). Everything is derived from the
    ledger, the batch records, the queue and the indexes."""
    pins = _read_json(layout.stream_json, {}) or {}
    ledger = verify_ledger(layout.ledger)
    ids = [e["batch"] for e in ledger]
    docs = {b: (_read_json(layout.batch_dir(b) / "batch.json", {}) or {}) for b in ids}
    q = load_queue(layout) if layout.queue.exists() else []
    live = [r for r in q if not r["refused"]]
    per_species = collections.Counter()
    per_source = collections.defaultdict(lambda: collections.Counter())
    for r in live:
        for k, n in enumerate(r.get("species_boxes") or []):
            per_species[C.CLASS_NAMES[k]] += int(n)
        s = per_source[r["source"]]
        s["images"] += 1
        s["target_boxes"] += sum(r.get("species_boxes") or [])
        s["target_boxes_admitted"] += sum(r.get("species_boxes") or [])
        s[r["admission"]] += 1
        s["eligible"] += int(r["eligible_step1"])
    # what Step 1 saw and refused per source (D28 reads images_seen and near_eval_embed): every judged item
    # of every batch, the refusals after admission (masked copies), and the queue rows refused later
    # (serve-holds' copy scan, domain dev, supersession)
    # D28-v2 reads, per source, the pair cosines of its dHash hits on evaluation images (a batch's eval_hits
    # record, or, for a batch committed before the record existed, its sidecar step1_stream/eval_hits/<batch>.json
    # when that weighed exactly the hits the batch counted and was made from the committed batch.json); a hit
    # counted in decision:near_eval_* without one is read by the one-hit rule (fail closed). "eval_hits" lists
    # the batches still due for a sidecar (step1_stream eval-hits, L17) and how each sidecar was read.
    hit_cos, hit_thr = collections.defaultdict(list), {}
    eh_status = {"due": [], "sidecars": {}}
    led_sha = {e["batch"]: e.get("batch_json_sha256") for e in ledger}
    for b, d in docs.items():
        if d.get("kind") == KIND_KNOWNTRUTH:
            continue                         # base copies measured, not a source's supply
        eh = d.get("eval_hits") if isinstance(d.get("eval_hits"), dict) else {}
        counts = dhash_counts(d)
        if counts and not _weighs_all(eh, counts):
            sp = eval_hits_sidecar_path(layout, b)
            side = _read_json(sp, None) if sp.is_file() else None
            if side is None:
                eh_status["due"].append(b)
            else:
                rec, why = EH.usable_sidecar(side, b, counts)
                if rec is not None and side.get("batch_json_sha256") != led_sha.get(b):
                    rec, why = None, "it was made from another batch.json than the one the ledger commits"
                srec = side.get("eval_hits") if isinstance(side.get("eval_hits"), dict) else {}
                eh_status["sidecars"][b] = {"used": rec is not None, "why": why, "hits": srec.get("hits"),
                                            "scored": srec.get("scored"), "built_utc": side.get("built_utc")}
                if rec is not None:
                    eh = rec
        for src, rec in ((eh.get("per_source") or {}) if isinstance(eh.get("per_source"), dict) else {}).items():
            per_source[src]["eval_hits_scored"] += int((rec or {}).get("scored") or 0)
            hit_cos[src].extend(float(c) for c in ((rec or {}).get("pair_cos") or []))
            if eh.get("copy_threshold") is not None:
                t = float(eh["copy_threshold"])
                hit_thr[src] = min(hit_thr.get(src, t), t)
        for src, dec in (d.get("per_source_decisions") or {}).items():
            for k, n in dec.items():
                per_source[src]["decision:%s" % k] += int(n)
                per_source[src]["images_seen"] += int(n)
                if k in NEVER_TRAIN_REASONS:
                    per_source[src]["never_train_refused"] += int(n)
                if k == "near_eval_embed":
                    per_source[src]["near_eval_embed"] += int(n)
        for src, dec in (d.get("per_source_refusals") or {}).items():
            for k, n in dec.items():
                per_source[src]["refused:%s" % k] += int(n)
                if k == "masked_near_eval_embed":
                    per_source[src]["near_eval_embed"] += int(n)
    for r in q:
        if r["refused"]:
            per_source[r["source"]]["refused:%s" % r["refused"]] += 1
            if r["refused"] == "near_eval_embed":
                per_source[r["source"]]["near_eval_embed"] += 1
                per_source[r["source"]]["never_train_refused"] += 1
    adm = collections.Counter()
    for d in docs.values():
        for k, n in (d.get("admission") or {}).items():
            adm[k] += int(n)
    holds = collections.Counter(h for r in live for h in r["holds"])
    today = _date(0)
    past = collections.Counter(h for r in live for h, dl in (r.get("hold_deadline") or {}).items()
                               if h in r["holds"] and dl and today > dl)
    kt = {b: d["knowntruth"]["overall"] for b, d in docs.items()
          if isinstance(d.get("knowntruth"), dict) and d["knowntruth"].get("overall", {}).get("boxes")}
    trig_prec = [b for b, m in kt.items() if (m.get("matched_verified") or 0) >= REFIT_MIN_MATCHED_VERIFIED
                 and m.get("verified_precision_wilson_lb") is not None
                 and m["verified_precision_wilson_lb"] < REFIT_PRECISION_LB]
    sp_new, sp_unknown = collections.Counter(), collections.Counter()
    for d in docs.values():
        for sp, vv in (d.get("species_verdicts") or {}).items():
            sp_new[sp] += sum(vv.values())
            sp_unknown[sp] += int(vv.get(V.UNKNOWN, 0))
    trig_sp = sorted(s for s in sp_new if sp_new[s] >= REFIT_SPECIES_NEW_BOXES
                     and sp_unknown[s] >= REFIT_SPECIES_UNKNOWN_SHARE * sp_new[s])
    listing = Index(layout, "listing", {}).data if layout.index("listing").exists() else {}
    first = {}
    for e in ledger:
        first.setdefault(e.get("kind"), e.get("utc"))
    ot = Index(layout, "one_time", {}).data if layout.index("one_time").exists() else {}
    one_time = {"bootstrap": pins.get("built_utc") if pins else None, "backfill": first.get(KIND_BACKFILL),
                "knowntruth": first.get(KIND_KNOWNTRUTH) or (ot.get("knowntruth") or {}).get("utc"),
                "eval_hits": (ot.get("eval_hits") or {}).get("utc")}
    doc = {"format": STATUS_FORMAT, "built_utc": _utc(), "one_time": one_time,
           "stream_version": int(pins.get("stream_version", STREAM_VERSION)),
           "versions": {"verifier": VERIFIER_VERSION, "reference": REFERENCE_VERSION,
                        "stream_pins_sha256": _sha_file(layout.stream_json) if layout.stream_json.exists() else None,
                        "splits_lock_sha256": (pins.get("splits") or {}).get("lock_sha256"),
                        "embedder": (pins.get("embedder") or {}).get("name")},
           "pending": {"registry_new_paths": {s: int(v.get("pending", 0)) for s, v in sorted(listing.items())
                                              if "refused" not in v},
                       "registry_listed_utc": {s: v.get("utc") for s, v in sorted(listing.items())},
                       "registry_refused": {s: v["refused"] for s, v in sorted(listing.items()) if "refused" in v},
                       "rejoin_needed": sorted({e.get("source") or e.get("key") for e in _read_jsonl(layout.events)
                                                if e.get("event") == "rejoin_needed"}
                                               | {s for s, v in listing.items() if v.get("refused") == "join_changed"}),
                       "intake_batches": _intake_pending(layout)},
           "batches": {"committed": ids, "in_progress": [b for b in _batch_ids(layout) if b not in ids],
                       "by_kind": dict(collections.Counter(e.get("kind") for e in ledger))},
           "crop_ids": {"v1": int(pins.get("crop_offset", 0)),
                        "next": int(pins.get("crop_offset", 0)) + sum(int(e.get("crops", 0)) for e in ledger)},
           "queue": {"rows": len(q), "live_rows": len(live),
                     "eligible_target_images": sum(1 for r in live if r["eligible_step1"]),
                     "eligible_target_boxes": sum(sum(r["species_boxes"]) for r in live if r["eligible_step1"]),
                     "admitted_images": dict(collections.Counter(r["admission"] for r in live)),
                     "per_species_boxes": {n: int(per_species.get(n, 0)) for n in C.CLASS_NAMES[:V.OTHER]},
                     "kind": dict(collections.Counter(r["kind"] for r in live))},
           "admission": {k: int(adm.get(k, 0)) for k in MK.ADMISSIONS},
           "holds": dict(sorted(holds.items())), "holds_past_deadline": dict(sorted(past.items())),
           "refused": dict(collections.Counter(r["refused"] for r in q if r["refused"])),
           "knowntruth": kt,
           "refit_triggers": {"precision_lb_below": trig_prec, "species_unknown_share": trig_sp,
                              "rule": "Wilson lb of known-truth verified precision < %.2f on >= %d matched verified "
                                      "boxes; or a species with >= %d new target-labelled boxes, unknown share >= "
                                      "%.1f (proposed, card X11)" % (REFIT_PRECISION_LB, REFIT_MIN_MATCHED_VERIFIED,
                                                                     REFIT_SPECIES_NEW_BOXES,
                                                                     REFIT_SPECIES_UNKNOWN_SHARE),
                              "fired": bool(trig_prec or trig_sp)},
           "human_queue": {"rows": len(_read_jsonl(layout.human))}, "eval_hits": eh_status,
           "per_source": {s: dict({"images_seen": 0, "near_eval_embed": 0, "target_boxes_admitted": 0}, **v)
                          for s, v in sorted(per_source.items())}}
    for s, row in doc["per_source"].items():
        if s in hit_cos or "eval_hits_scored" in row:
            row["eval_hit_pair_cos"] = sorted(hit_cos.get(s, []), reverse=True)
            row["eval_hit_copy_threshold"] = hit_thr.get(s)
    _write_json(layout.status, doc)
    return doc


def verify_state(layout):
    """[] or the problems of the state: the ledger chain, contiguous and
    disjoint global crop ids, batch records unchanged, unique queue keys and
    rows per batch as committed."""
    probs = []
    try:
        ledger = verify_ledger(layout.ledger)
    except StreamError as e:
        return [str(e)]
    pins = read_pins(layout)
    nxt = int(pins["crop_offset"])
    for e in ledger:
        if int(e["crop_offset"]) != nxt:
            probs.append("%s: crop ids start at %d, expected %d" % (e["batch"], e["crop_offset"], nxt))
        nxt = int(e["crop_offset"]) + int(e["crops"])
        bj = layout.batch_dir(e["batch"]) / "batch.json"
        if not bj.is_file() or _sha_file(bj) != e["batch_json_sha256"]:
            probs.append("%s: batch.json changed since commit" % e["batch"])
    rows = _read_jsonl(layout.queue)
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        probs.append("queue.jsonl holds %d duplicate key(s)" % (len(keys) - len(set(keys))))
    per = collections.Counter(r["batch"] for r in rows)
    for e in ledger:
        if per.get(e["batch"], 0) != int(e["queue_rows"]):
            probs.append("%s: %d queue rows, the ledger says %d" % (e["batch"], per.get(e["batch"], 0), e["queue_rows"]))
    return probs


# --------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc2.step1_stream",
                                 description="Incremental Step 1 with per-box admission (docs/CONTINUOUS_LOOP.md §3.3)")
    ap.add_argument("verb", choices=("bootstrap", "admit", "backfill", "knowntruth", "rejoin", "serve-holds",
                                     "scan-holds", "eval-hits", "status", "verify"),
                    help="scan-holds is serve-holds under the name the autopilot submits (L17 --hold); eval-hits "
                         "weighs again, into sidecars, the dHash hits of batches committed before D28-v2 (L17)")
    ap.add_argument("--state-dir", default=None, help="default INC_DIR/step1_stream")
    ap.add_argument("--lock", default=None, help="the splits v2 LOCK.json (default INC_DIR/splits/v2/LOCK.json)")
    ap.add_argument("--intake", default=None, help="admit: the collect intake batch name; eval-hits: only this "
                                                    "intake batch (with --batch-id: those Step 1 batches too)")
    ap.add_argument("--batch-id", action="append", default=None,
                    help="eval-hits: only this committed Step 1 batch (repeatable; default every batch that needs it)")
    ap.add_argument("--force", action="store_true", help="eval-hits: write a sidecar again where one exists")
    ap.add_argument("--registry", action="store_true", help="admit: new paths of the v1 registry slugs")
    ap.add_argument("--slugs", default=None, help="admit --registry: only these slugs (comma-separated)")
    ap.add_argument("--rule", choices=("box", "image"), default="box", help="admission rule (D-B box, v1 image)")
    ap.add_argument("--slug", default=None, help="rejoin: the slug")
    ap.add_argument("--sets", default="tsw22,tsw23", help="knowntruth: expert base sets")
    ap.add_argument("--census", default=None, help="backfill: census_v1.json (default INC_DIR/funnel/census_v1.json)")
    ap.add_argument("--no-census", action="store_true", help="backfill: record the census check as not done")
    ap.add_argument("--realloop-exp", default="realloop_v1", help="backfill: the experiment whose draws are priors")
    ap.add_argument("--leak", default=None, help="the stream's own detector calibration (INC_DIR/intake/leak/leak_v1.json)")
    ap.add_argument("--funnel-leak", default=None, help="the funnel's leak_v1.json (INC_DIR/funnel/leak_v1.json)")
    ap.add_argument("--domain-dev", default=None, help="serve-holds: the funnel F9 domain_dev.jsonl")
    ap.add_argument("--hold", action="append", default=None, choices=("h6_scan", "licence", "funnel_F9"),
                    help="serve-holds / scan-holds: serve only this hold (repeatable; default all three)")
    ap.add_argument("--no-seed-v1", action="store_true", help="bootstrap: start the indexes empty")
    ap.add_argument("--testing", action="store_true", help="bootstrap: accept a non-contract embedder")
    ap.add_argument("--procs", type=int, default=V.PROCS)
    ap.add_argument("--batch", type=int, default=V.BATCH, help="crops per forward pass")
    ap.add_argument("--chunk-images", type=int, default=CHUNK_IMAGES)
    ap.add_argument("--cap", type=int, default=BATCH_CAP, help="images per batch")
    ap.add_argument("--no-amp", action="store_true")
    return ap


def eval_hits_scope(a):
    """(bids, intakes) for eval_hits from the CLI's flags: neither --batch-id
    nor --intake -> every batch that needs a sidecar (None, None); --batch-id
    only -> those Step 1 batches and no intake batch; --intake only -> that
    intake batch and no Step 1 batch ([]); both -> both."""
    if not (a.batch_id or a.intake):
        return None, None
    return list(a.batch_id or []), ([a.intake] if a.intake else False)


def main(argv=None):
    a = build_parser().parse_args(argv)
    layout = Layout(a.state_dir)
    try:
        if a.verb == "bootstrap":
            bootstrap(layout, a.lock, testing=a.testing, seed_v1=not a.no_seed_v1, procs=a.procs)
        elif a.verb == "status":
            try:
                with writer_lock(layout):
                    doc = write_status(layout)
            except StreamError as e:
                log("%s; status.json is read, not rewritten" % e)
                doc = _read_json(layout.status, {}) or {}
            print(json.dumps({k: doc.get(k) for k in ("batches", "queue", "admission", "holds")}, indent=1))
        elif a.verb == "verify":
            probs = verify_state(layout)
            for p in probs:
                log("PROBLEM: %s" % p)
            if probs:
                return 2
            log("state verifies: ledger chain, crop ids, batch records, queue")
        else:
            pins = read_pins(layout)
            if a.lock and Path(a.lock).resolve() != Path(pins["splits"]["lock"]).resolve():
                raise StreamError("--lock %s is not the pinned splits v2 LOCK %s: the guard must be the one the "
                                  "stream was bootstrapped with (a new LOCK is a new stream version)"
                                  % (a.lock, pins["splits"]["lock"]))
            guards = load_guards(pins["splits"]["lock"])
            scanner = load_scanner(layout, a.leak, a.funnel_leak, procs=a.procs)
            if a.verb == "admit":
                if bool(a.intake) == bool(a.registry):
                    raise StreamError("admit needs exactly one of --intake NAME or --registry")
                emb = V.BioclipEmbedder(amp=not a.no_amp)
                if a.intake:
                    spec, fn, kind = "intake:%s" % a.intake, (lambda: list_intake(layout, a.intake, a.cap)), KIND_INTAKE
                else:
                    slugs = [s for s in (a.slugs or "").split(",") if s]
                    spec = "registry" + (":" + ",".join(sorted(slugs)) if slugs else "")
                    fn, kind = (lambda: list_registry(layout, slugs or None, a.cap)), KIND_REGISTRY
                run_batch(layout, spec, fn, emb, guards, scanner, a.rule, a.procs, a.batch, a.chunk_images, kind=kind)
            elif a.verb == "backfill":
                backfill(layout, guards, scanner, a.census, not a.no_census, a.realloop_exp, a.procs)
            elif a.verb == "knowntruth":
                emb = V.BioclipEmbedder(amp=not a.no_amp)
                run_knowntruth(layout, emb, guards, tuple(s for s in a.sets.split(",") if s), a.procs, a.batch,
                               a.chunk_images)
            elif a.verb == "rejoin":
                if not a.slug:
                    raise StreamError("rejoin needs --slug")
                rejoin(layout, a.slug, guards, scanner, procs=a.procs)
            elif a.verb in ("serve-holds", "scan-holds"):
                serve_holds(layout, scanner, domain_dev_path=a.domain_dev, kinds=a.hold)
            elif a.verb == "eval-hits":
                bids, intakes = eval_hits_scope(a)
                out = eval_hits(layout, guards, scanner, bids=bids, intakes=intakes, procs=a.procs,
                                force=a.force, lock_path=pins["splits"]["lock"])
                bad = sorted("%s %s" % (k, n) for k in ("step1", "intake") for n, v in out[k].items()
                             if v.get("status") == "failed")
                print(json.dumps(out, indent=1, default=str))
                if bad:
                    # a batch left without a sidecar is proposed again: a failed job, not a silent loop
                    raise StreamError("no sidecar for %s" % ", ".join(bad))
    except StreamError as e:
        log("FAILED: %s" % e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
