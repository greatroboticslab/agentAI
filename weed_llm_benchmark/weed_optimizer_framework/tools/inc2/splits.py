"""Splits v2: the grown base, its never-train index and LOCK v2
(docs/CONTINUOUS_LOOP.md §4.1-4.2; owner decision D-A; decision L-5).

    python -m weed_optimizer_framework.tools.inc2.splits build   [--testing] [--licences F] [--tsw-record F]
                                                                  [--skip-scan] [--calibration F]
    python -m weed_optimizer_framework.tools.inc2.splits lock    [--testing]
    python -m weed_optimizer_framework.tools.inc2.splits scan    [--calibration F] [--procs N] [--force]
    python -m weed_optimizer_framework.tools.inc2.splits verify
    python -m weed_optimizer_framework.tools.inc2.splits summary

Only the training side changes (§4.1). dev, test and imageweeds are byte
copies of the v1 manifests, each equal to the v1 LOCK's sha256, so every v2
model is scored by the unchanged v1 scorer on the v1 exams and B0's, B's and
realloop_v1's numbers stay comparable. train_core is v1's train_core minus
the rows decision L-8 drops (below): every other line byte-identical, in
order, and a byte copy of v1 when nothing is dropped. The v1 files are
inputs only; nothing under splits/v1 is written.

build (one GPU-shared job; the order of §4.2):
  1. precondition: inc.splits.verify() is [] (a full re-hash of v1), the
     scorer is the one the v1 LOCK records;
  2. the v1 manifests read, each refused on any mismatch with the v1 LOCK;
  3. tsw22 / tsw23 from the v1 ood22 / ood23 rows: keys tsw2x__<stem> (no
     training row carries an exam key), source 3seasonweeddet10/data202x,
     session = the stem minus the frame number, labels byte-copied from the
     v1 converted files to v2/labels/tsw2x/ (label_sha256 unchanged), the
     Zenodo record's licence recorded;
  4. drops, each recorded with its reason: within 6 bits of dev, test or
     imageweeds by dHash or by any of the 8 flips and rotations; a copy
     found by the embedding scan under the row's copy rule (L-9, below;
     unhashable_embed for an image the scan could not describe); a row of a
     dev or test capture session (L-9(a)); a row within 6 bits of an image of
     an earlier part of the base (train_core, tsw22, tsw23, base B's part, in
     that order: the one ood23-ood22 pair keeps its tsw22 row). Rows sharing
     a session with train_core are kept and counted;
  5. base B's 878 (step1/base_selected.jsonl, checked against the sha256 that
     select_summary.json records): L-5 drops every image of cwp10 and vanpe
     outright (listed in l5_excluded.jsonl); the rest get the same 8-variant
     check against dev, test and imageweeds, the embedding scan and, once
     the funnel's leak_v1.json is complete, its list of base B copies (L-9(b):
     only matches against a v2 evaluation split that are copies at the v2
     threshold), and a hit refuses the build (an R4 incident: it would void
     B's and realloop_v1's results, funnel H6(b)).
     The L-5 part is checked too, and a hit there is recorded as that
     incident without refusing (its images are in no v2 manifest). Licences
     come per source from the funnel's card index, the funnel config's
     licence table, the Zenodo record or an owner table (--licences);
     unresolved -> research_only true, licence "unresolved", the base's
     owner-accepted exemption, never silently. train_core gets the 8-variant
     check too (v1 compared the stored dHash only). Decision L-8: a row
     within 6 bits of an evaluation image under a flip or rotation
     (near_eval_variant) is dropped from base v2 and from train_core.jsonl,
     listed in train_core_variant_drops.jsonl with its match (sha256 in LOCK
     v2) and recorded in summary.json as an incident; B0's, B's and
     realloop_v1's training sets held it and are not re-run. More drops
     than variant_drop_cap(|train_core|) = max(1, floor(0.5 % of
     train_core)), or a hit on the stored dHash itself (which v1 checked, so
     v1 and this build disagree), still refuses the build (R4). A near copy
     of a dropped image in tsw or base B is dropped as near_train_core: the
     de-duplication index keeps every v1 train_core image;
  6. nevertrain_dhash.json: dev + test + imageweeds (the v1 index's entries
     for those splits) at 6 bits, written incomplete (refused by every
     loader) until lock;
  7. base_copies_dhash.json: every base v2 image at 6 bits, likewise;
  8. base_v2.jsonl = train_core + tsw22 + tsw23 + base B's kept part, checked
     pairwise disjoint across parts by key, path, sha256 and 6-bit dHash
     (with variants); base_v2_provenance.jsonl holds per row its part, dHash
     and 8 variants, licence, research_only and lab group;
  9. summary.json, written last.

The embedding scan (§4.2 step 5 [review], R0b): the embedding copy detector of
§3.2 (funnel.leak as a library, via inc2.guard) over every candidate a build
could keep besides train_core (all v1 ood22 / ood23 rows under their tsw
keys, all of base B's 878) against dev, test and imageweeds. It reads only v1,
base_selected.jsonl and the funnel config, so its result does not depend on
any build (a scan never forgets a row an earlier build dropped). The
threshold comes from a passed funnel leak_v1.json when there is one, else
from the stream's own calibration under splits/v2/leak/ (seed prefix
stream/v1/leak, the funnel prereg's H6 recall and false-positive gates).
Writes embed_scan.json. build runs it itself (after its worker processes
have ended) whenever no embed_scan.json covers the current candidates, the
v1 manifests, base_selected.jsonl and the funnel config; the `scan` verb runs
it alone. So the platform's sequence is build, then lock (L23's two verbs);
build --skip-scan applies only an existing scan, and lock refuses without one.

Decision L-9 (2026-09-29; job 47259471 refused on the threshold's scene
similarity, not on copies): the scan's threshold (the funnel's 0.8256) is
recalibrated for images without capture provenance (inc2.embed_calibration:
per-image false positives on hard same-domain negatives, never below the
scan's threshold) into embed_calibration_v2.json, which the scan verb and a
build (scan_mode auto) write after the scan. build applies embed_scan.json
through it, and records per row which rule applied (copy_rules.jsonl; the
provenance file's copy_rule):
  tsw rows (the md5-verified Zenodo original of the lab's own dataset):
    a row whose capture session (the date_camera prefix of its name) is a
    dev or test session is dropped (dev_session, test_session);
    tsw_provenance: every other row with a capture session is exempt from
    the embedding threshold and stays under the 6-bit dHash + 8-variant
    check; tsw_embed_v2: a row without one is judged like base B;
    an image the scan could not describe is dropped (unhashable_embed);
  base B (base_b_embed_v2): a hit at or above the v2 threshold (or within
    6 dHash bits) in the kept part refuses the build (H6(b), R4); a hit
    only at the scan's threshold is recorded, not applied; the L-5 part is
    recorded as an incident as before;
  the funnel's leak_v1 base_B list: an entry is applied only when its
    evaluation split is a v2 one (dev, test, imageweeds) and it is a copy
    at the v2 threshold; an entry against ood22 / ood23 (exams in v1,
    training in v2) is recorded as not_v2_split, one below the v2
    threshold as below_v2_threshold.
Without a current v2 calibration (build --skip-scan), every scan hit and
every v2-split funnel entry applies (the scan's own threshold), and lock
refuses until one exists.

lock: refuses unless v1 still verifies, the build is intact (every manifest,
image and label re-hashed), embed_scan.json covers every tsw and harvested
row of base_v2 by key and sha256 with no copy among them under each row's
copy rule (re-derived; the provenance file's copy_rule must agree), the v2
calibration is the one the build applied and was made from the scan's
calibration and evaluation descriptors, copy_rules.jsonl is the build's, no
base v2 row is
refused by the never-train guard (from the recorded dHash and 8 variants),
the L-8 list hashes as the build recorded, is within the cap, names only v1
train_core rows (same bytes) that the never-train guard refuses as
near_eval_variant on their re-computed hashes, and v2's train_core is v1's
minus exactly those rows,
the funnel's leak_v1.json (when complete) lists no base v2 image as a copy
of a v2 evaluation split at the v2 threshold (L-9(b)),
the byte copies still equal v1 and the scorer is v1's, and the build is not
a --testing one (unless lock --testing). Marks both indexes complete, writes
LOCK.json (with the funnel's H6 status of base B's part: pending, or the
leak_v1 result; the L-8 list's sha256; the v2 calibration's, its negatives
file's and copy_rules.jsonl's sha256; derived_from: the byte copies and
train_core's derivation) and lock_log.jsonl, then makes every file under splits/v2
read-only. A second lock or a build after lock refuses: a change is a new
splits version (R4).

--testing lifts the pins on the real data (the v1 LOCK shas, base_selected's
sha256 and the contract's counts) for synthetic worlds; the build records it,
lock refuses such a build without its own --testing, and run_inc2_splits.sh
refuses both.

One writer at a time: build, scan and lock hold INC_DIR/splits/.v2.writer.lock.
Directories are listed once each, never rglob'd (Lustre).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import sys
import time
from pathlib import Path

from . import common as C2
from . import embed_calibration as EC
from . import guard as G
from ..inc import common as C1
from ..inc import splits as S1
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NearHashIndex

BITS = HOLDOUT_NEAR_DUP_BITS
TSW = {"tsw22": "ood22", "tsw23": "ood23"}
TSW_SOURCE = {"tsw22": "3seasonweeddet10/data2022", "tsw23": "3seasonweeddet10/data2023"}
TSW_ZENODO_RECORD = "14861516"
# AgML's copy of the same 3SeasonWeedDet10 release; the funnel config names its lab group.
TSW_LAB_ALIASES = ("project_agml__three_season_weed_detection",)
BASE_B = "base_b"
# The funnel's own H6 scan (leak_v1.json, pre-registered) listing a base B image as a copy.
FUNNEL_REASON = "funnel_leak_v1_copy"
COPY_REASONS = ("near_eval_v2", "near_eval_variant", "near_eval_embed", FUNNEL_REASON)
PARTS = ("train_core", "tsw22", "tsw23", BASE_B)
L5_SOURCES = ("rf_karthikeya-c8pvy__weed-detection-cwp10", "rf_zig-zag-lnodr__weed-detection-vanpe")
L5_DECISION = "L-5, docs/CONTINUOUS_LOOP.md 2.6 (human-delegated, 2026-09-28)"
# L-8: train_core rows a flip or rotation puts within 6 bits of an evaluation image are dropped,
# up to the cap; beyond it the build refuses (an R4 incident for a person).
L8_DECISION = "L-8, docs/CONTINUOUS_LOOP.md 2.6 (human-delegated, 2026-09-29)"
VARIANT_DROP_REASON = "near_eval_variant"
VARIANT_DROP_CAP_FRAC = 0.005
VARIANT_DROP_CAP_MIN = 1
V1_RESULTS_NOTE = ("B0's, B's and realloop_v1's training sets held these train_core images (v1 compared the stored "
                   "dHash only). The effect is negligible (%d of %d training images) and those results are not "
                   "re-run (L-8).")
# L-9: which copy rule judges a candidate row (copy_rules.jsonl, the provenance file's copy_rule)
L9_DECISION = "L-9, docs/CONTINUOUS_LOOP.md 2.6 (human-delegated, 2026-09-29)"
RULE_TRAIN_CORE = "train_core"                  # v1's train_core, the 8-variant dHash check (L-8)
RULE_TSW_PROVENANCE = "tsw_provenance"          # L-9(a): exempt from the embedding threshold, dHash + 8 variants
RULE_TSW_EVAL_SESSION = "tsw_eval_session"      # L-9(a): a dev or test capture session, dropped
RULE_TSW_EMBED = "tsw_embed_v2"                 # a tsw row without a capture session: the v2 threshold
RULE_BASE_B = "base_b_embed_v2"                 # L-9(c): the v2 threshold
RULE_L5 = "l5_excluded"
COPY_RULES_NAME = "copy_rules.jsonl"
FUNNEL_NOT_V2 = "not_v2_split"                  # L-9(b): a funnel match against ood22 / ood23
FUNNEL_BELOW_V2 = "below_v2_threshold"          # a funnel match below the v2 threshold and > 6 dHash bits
FUNNEL_APPLIED = "applied"
UNRESOLVED = "unresolved"
LICENCE_EXEMPTION = ("An unresolved licence of a base v2 row is recorded as licence 'unresolved' with "
                     "research_only true: the owner-accepted exemption for the base (D-A named these images; "
                     "docs/CONTINUOUS_LOOP.md 4.2 step 5). A model trained on them is research-only.")

# The real data, checked unless --testing: the v1 LOCK (sha256 prefixes of the
# files this build reads; §4.1 names the four byte copies), base B's harvested
# part (§4.2 step 8) and the contract's counts (§4.2, L-5).
REAL_PINS = {
    "v1_manifests": {"dev": "a5c904cd", "test": "31ba7650", "imageweeds": "357936e3", "train_core": "242ef6b9",
                     "ood22": "e5a6aff7", "ood23": "8be59afb"},
    "v1_scorer": "18c00837",
    "base_selected": "7e47d374",
    "rows": {"dev": 617, "test": 1977, "imageweeds": 3208, "train_core": 3049, "ood22": 1915, "ood23": 1784,
             "base_selected": 878},
    "nevertrain_entries": 5802,
    "l5_excluded": 812,
}

SUMMARY_NAME = "summary.json"
EMBED_SCAN_NAME = "embed_scan.json"
LOCK_LOG_NAME = "lock_log.jsonl"
LEAK_DIR_NAME = "leak"
LABELS_NAME = "labels"
STAGING_NAME = ".staging"
SUMMARY_FORMAT = "inc2-splits-summary/1"
SCAN_FORMAT = "inc2-embed-scan/1"
PROGRESS_EVERY = 1000
PROCS = 5


class SplitError(C2.Inc2Error):
    """A condition under which splits v2 must not be built, scanned or locked."""


def log(msg):
    print("[inc2.splits] %s" % msg, flush=True)


# ------------------------------------------------------------------- paths
def step1_dir():
    return C1.INC_DIR / "step1"


def funnel_dir():
    return C1.INC_DIR / "funnel"


def base_selected_path():
    return step1_dir() / "base_selected.jsonl"


def select_summary_path():
    return step1_dir() / "select_summary.json"


def pool_summary_path():
    return step1_dir() / "pool_summary.json"


def pool_path():
    return step1_dir() / "pool.jsonl"


def cards_index_path():
    return funnel_dir() / "cards" / "index.json"


def funnel_leak_path():
    return funnel_dir() / "leak_v1.json"


def funnel_prereg_path():
    return funnel_dir() / "prereg_v1.json"


def default_domain_config():
    return Path(__file__).resolve().parents[1] / "funnel" / "domains" / "weed.json"


def default_tsw_record():
    return C1.REPO / "downloads" / "3seasonweeddet10" / ("zenodo_%s.json" % TSW_ZENODO_RECORD)


def summary_path():
    return C2.SPLITS_DIR / SUMMARY_NAME


def embed_scan_path():
    return C2.SPLITS_DIR / EMBED_SCAN_NAME


def embed_calibration_path():
    return C2.SPLITS_DIR / EC.NAME


def copy_rules_path():
    return C2.SPLITS_DIR / COPY_RULES_NAME


def leak_dir():
    return C2.SPLITS_DIR / LEAK_DIR_NAME


def label_dir(split):
    return C2.SPLITS_DIR / LABELS_NAME / split


def writer_lock_path():
    return C1.INC_DIR / "splits" / ".v2.writer.lock"


# ------------------------------------------------------------------ helpers
SLURM_ALIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUE", "RESIZING", "SIGNALING",
               "STAGE_OUT")


def slurm_job_state(job_id, squeue=("squeue",)):
    """"alive", "dead" or "unknown" for a Slurm job id, as run_inc_build.sh
    reads squeue: "Invalid job id" or a finished state is dead; no squeue, a
    timeout or any other error is unknown."""
    import subprocess
    if not str(job_id).isdigit():
        return "unknown"
    if str(job_id) == os.environ.get("SLURM_JOB_ID"):
        return "dead"                                   # this job, requeued
    try:
        r = subprocess.run(list(squeue) + ["-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        return "dead" if "Invalid job id" in out else "unknown"
    return "alive" if any(s in out for s in SLURM_ALIVE) else "dead"


def _holder_stale(parts):
    """True when the writer lock's holder ("pid host utc slurm_job_id") no
    longer runs: its process on this host is gone, or its Slurm job has ended
    (a job killed at its time limit on another node leaves the lock behind:
    SIGTERM does not run the release)."""
    if len(parts) >= 2 and parts[1] == socket.gethostname() and parts[0].isdigit():
        try:
            os.kill(int(parts[0]), 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False
    if len(parts) >= 4 and parts[3].isdigit():
        return slurm_job_state(parts[3]) == "dead"
    return False


@contextlib.contextmanager
def writer():
    """One build / scan / lock at a time (O_EXCL lock file). A lock whose
    holder no longer runs (_holder_stale: its process on this host, or its
    Slurm job anywhere) is taken over; any other holder refuses."""
    path = writer_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    me = "%d %s %s %s\n" % (os.getpid(), socket.gethostname(), C2.utc(), os.environ.get("SLURM_JOB_ID", "none"))
    for attempt in (0, 1):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                held = path.read_text()
            except OSError:
                held = ""
            stale = attempt == 0 and _holder_stale(held.split())
            if stale:
                log("WARNING: took over the stale writer lock %s (%s)" % (path, held.strip()))
                path.unlink()
                continue
            raise SplitError("another splits v2 writer holds %s (%s); remove it by hand only when none runs"
                             % (path, held.strip() or "unreadable"))
        with os.fdopen(fd, "w") as fh:
            fh.write(me)
        break
    try:
        yield
    finally:
        try:
            if path.read_text() == me:
                path.unlink()
        except OSError:
            pass


def variant_drop_cap(n_train_core):
    """The most train_core rows decision L-8 lets a build drop by itself:
    max(1, floor(0.5 % of train_core)); 15 of the real 3,049. A single
    image is always a stray; more than the cap is an R4 incident."""
    return max(VARIANT_DROP_CAP_MIN, int(VARIANT_DROP_CAP_FRAC * int(n_train_core)))


def _pin(testing, what, got, want, errors):
    """Record a real-data pin; a mismatch is an error unless testing."""
    ok = str(got).startswith(str(want)) if isinstance(want, str) else got == want
    if not ok and not testing:
        errors.append("%s is %s, the contract's is %s" % (what, str(got)[:16], want))
    return ok


def _hash_task(path):
    h, v = G.image_hashes(path)
    return path, h, v


def _sha_task(path):
    try:
        return path, C1.sha256_file(path)
    except OSError:
        return path, None


def _pmap(fn, items, procs, what):
    """[fn(item)] in order, over worker processes when procs > 1."""
    items = list(items)
    n = len(items)
    out = []
    if procs > 1 and n > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp
        with ctx.Pool(procs) as pool:
            for i, r in enumerate(pool.imap(fn, items, chunksize=8), 1):
                out.append(r)
                if i % PROGRESS_EVERY == 0 or i == n:
                    log("  %s %d/%d" % (what, i, n))
    else:
        for i, it in enumerate(items, 1):
            out.append(fn(it))
            if i % PROGRESS_EVERY == 0 or i == n:
                log("  %s %d/%d" % (what, i, n))
    return out


def hash_rows(rows, what, procs=1):
    """Set r["dhash"] and r["variants"] (8 ints, VARIANTS order, or None) on
    every row, one decode pass per distinct image path."""
    todo = sorted({r["image"] for r in rows if "dhash" not in r})
    got = {p: (h, v) for p, h, v in _pmap(_hash_task, todo, procs, "dHash %s" % what)}
    for r in rows:
        if "dhash" not in r:
            h, v = got[r["image"]]
            r["dhash"] = h
            vl = G.variant_list(v)
            r["variants"] = [x for _n, x in vl] if vl else None


def _session_of_image(path):
    return S1.session_of(os.path.splitext(os.path.basename(str(path)))[0])


def tsw_key(split, v1_key):
    pre = TSW[split] + "__"
    if not v1_key.startswith(pre):
        raise SplitError("%s row key %r lacks the prefix %r" % (TSW[split], v1_key, pre))
    return "%s__%s" % (split, v1_key[len(pre):])


def research_only(licence):
    """P6: True unless the licence is a recognised permissive one; an
    unresolved or non-commercial licence, one restricted to research,
    academic, educational or non-profit use, or one this rule does not know,
    is research-only. A restriction wins over a permissive name in the same
    text ("CC BY Non Commercial", "CC BY 4.0, research use only")."""
    if not licence or str(licence).strip().lower() == UNRESOLVED:
        return True
    t = str(licence).lower()
    letters = re.sub(r"[^a-z]", "", t)
    if (re.search(r"(^|[^a-z])nc([^a-z]|$)", t)
            or any(w in letters for w in ("noncommercial", "notforcommercial", "nocommercial", "researchonly",
                                          "researchuseonly", "researchpurposesonly", "academic", "educational",
                                          "nonprofit", "personaluse", "evaluationonly"))):
        return True
    if re.search(r"cc[-_ ]?by|cc0|cc[-_ ]?zero|public[-_ ]domain|pddl|(^|[^a-z])mit([^a-z]|$)|apache|"
                 r"(^|[^a-z])bsd([^a-z]|$)|odc[-_ ]?by", t):
        return False
    return True


class Licences:
    """Per-source licence lookup, in order: the funnel's card index, the funnel
    config's sources.licences, the Zenodo record of 3SeasonWeedDet10 (tsw
    sources), an owner table ({"sources": {source: {"licence", "evidence"}}})."""

    def __init__(self, cards=None, config=None, tsw_record=None, table=None):
        self.cards = (cards or {}).get("cards") or {}
        self.config = ((config or {}).get("sources") or {}).get("licences") or {}
        lic = (((tsw_record or {}).get("metadata") or {}).get("license") or {})
        self.tsw = lic.get("id") if isinstance(lic, dict) else (lic or None)
        self.table = (table or {}).get("sources") or {}

    def lookup(self, source):
        names = [source, source.split("/")[0]]
        for n in names:
            for e in self.cards.get(n) or ():
                if e.get("licence"):
                    return str(e["licence"]), "funnel cards index (%s, %s)" % (n, e.get("url") or e.get("file"))
        for n in names:
            if self.config.get(n):
                return str(self.config[n]), "funnel domain config sources.licences (%s)" % n
        if source in TSW_SOURCE.values() and self.tsw:
            return str(self.tsw), "Zenodo record %s metadata.license.id" % TSW_ZENODO_RECORD
        for n in names:
            e = self.table.get(n)
            if isinstance(e, dict) and e.get("licence"):
                return str(e["licence"]), "owner table (%s): %s" % (n, e.get("evidence") or "no evidence given")
        return UNRESOLVED, "no licence record found"


def lab_group(groups, source, part):
    """The lab group of a row, from the funnel config's sources.lab_groups."""
    names = {source, source.split("/")[0], part}
    if part in TSW:
        names |= set(TSW_LAB_ALIASES)
    hits = sorted(g for g, members in (groups or {}).items() if names & set(members or ()))
    return hits[0] if len(hits) == 1 else None


def _read_json(path, what):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise SplitError("%s missing: %s" % (what, path))
    except (OSError, ValueError) as e:
        raise SplitError("%s unreadable: %s (%s)" % (what, path, e))


def _opt_json(path):
    path = Path(path)
    if not path.is_file():
        return None, None
    return _read_json(path, str(path)), C2.file_record(path)


def _code_record():
    mods = {"inc2/common.py": C2.__file__, "inc2/guard.py": G.__file__, "inc2/splits.py": __file__,
            "inc/common.py": C1.__file__, "inc/splits.py": S1.__file__}
    return {k: C1.sha256_file(v) for k, v in sorted(mods.items())}


def _counts(rows):
    counts = [0] * C1.NC
    for r in rows:
        for b in C1.read_yolo(r["label"]):
            if 0 <= b[0] < C1.NC:
                counts[b[0]] += 1
    return dict(zip(C1.CLASS_NAMES, counts))


def read_summary():
    return _read_json(summary_path(), "splits v2 summary.json (run build first)")


def _manifest_row(r):
    return {k: r[k] for k in C1.MANIFEST_KEYS}


# -------------------------------------------------------------------- build
SCAN_MODES = ("auto", "skip")


def build(testing=False, scorer_path=None, licences_path=None, tsw_record_path=None, domain_config=None,
          procs=PROCS, scan_mode="auto", embedder=None, calibration_path=None):
    """Build every v2 file but LOCK.json (module docstring). Returns the
    summary. None of the build's own files is written before every check has
    passed. scan_mode "auto" runs the embedding copy scan first when
    embed_scan.json does not cover the current candidates (so build then lock
    is the whole sequence, as the platform's L23 runs it); "skip" applies
    only a scan that already exists (lock still refuses without one)."""
    if scan_mode not in SCAN_MODES:
        raise SplitError("scan_mode %r is not one of %s" % (scan_mode, SCAN_MODES))
    with writer():
        try:
            return _build(testing, scorer_path, licences_path, tsw_record_path, domain_config, procs, scan_mode,
                          embedder, calibration_path)
        finally:
            shutil.rmtree(C2.SPLITS_DIR / STAGING_NAME, ignore_errors=True)


def _v1_state(testing, scorer_path, errors):
    """(v1 LOCK, v1 summary, v1 index, scorer path, inputs): after a full v1 verify."""
    scorer = Path(scorer_path or S1.default_scorer_path())
    log("precondition: inc.splits.verify() (a full re-hash of splits v1)")
    probs = S1.verify(scorer_path=scorer)
    if probs:
        raise SplitError("splits v1 do not verify (%d problem(s), e.g. %s); v2 is built only on an intact v1"
                         % (len(probs), probs[:3]))
    lock1 = C1.read_lock()
    if C1.sha256_file(scorer) != lock1["scorer_sha256"]:
        raise SplitError("scorer %s is not the one the v1 LOCK records" % scorer)
    _pin(testing, "the v1 scorer sha256", lock1["scorer_sha256"], REAL_PINS["v1_scorer"], errors)
    for split, want in REAL_PINS["v1_manifests"].items():
        _pin(testing, "the v1 LOCK sha256 of %s" % split, lock1["manifests"].get(split, ""), want, errors)
    summ1 = _read_json(C1.SPLITS_DIR / S1.SUMMARY_NAME, "v1 summary.json")
    idx1 = _read_json(C1.NEVER_TRAIN_INDEX, "the v1 never-train index")
    if idx1.get("complete") is not True:
        raise SplitError("the v1 never-train index is not complete; lock v1 first")
    inputs = {"v1_lock": C2.file_record(C1.LOCK_PATH), "v1_summary": C2.file_record(C1.SPLITS_DIR / S1.SUMMARY_NAME),
              "v1_nevertrain": C2.file_record(C1.NEVER_TRAIN_INDEX), "scorer": C2.file_record(scorer)}
    return lock1, summ1, idx1, scorer, inputs


def _build(testing, scorer_path, licences_path, tsw_record_path, domain_config, procs, scan_mode="auto",
           embedder=None, calibration_path=None):
    t0 = time.time()
    if C2.LOCK_PATH.exists():
        raise SplitError("%s exists: splits v2 are locked; a change is a new splits version (R4)" % C2.LOCK_PATH)
    errors = []
    lock1, summ1, idx1, scorer, inputs = _v1_state(testing, scorer_path, errors)

    # 2. the v1 manifests: the byte copies (dev, test, imageweeds), train_core (copied minus the L-8 drops), ood
    v1rows, copy_bytes = {}, {}
    for split in C2.BYTE_COPIES + C2.FILTERED_COPIES + ("ood22", "ood23"):
        p = C1.manifest_path(split)
        with open(p, "rb") as fh:
            data = fh.read()
        got = hashlib.sha256(data).hexdigest()
        if got != lock1["manifests"].get(split):
            raise SplitError("v1 manifest %s hashes to %s, the v1 LOCK says %s" % (split, got[:12],
                                                                                  str(lock1["manifests"].get(split))[:12]))
        if split in C2.BYTE_COPIES + C2.FILTERED_COPIES:
            copy_bytes[split] = data
        v1rows[split] = C1.read_manifest(p)
        _pin(testing, "the v1 row count of %s" % split, len(v1rows[split]), REAL_PINS["rows"][split], errors)

    # 3. the v2 never-train index: the v1 entries of dev, test and imageweeds
    entries = [[int(h), s, k] for h, s, k in idx1["entries"] if s in C2.EVAL_SPLITS]
    have = {(s, k) for _h, s, k in entries}
    want = {(s, r["key"]) for s in C2.EVAL_SPLITS for r in v1rows[s]}
    if have != want:
        raise SplitError("the v1 never-train index does not hold exactly the dev, test and imageweeds rows "
                         "(%d missing, %d extra)" % (len(want - have), len(have - want)))
    _pin(testing, "the v2 never-train entries", len(entries), REAL_PINS["nevertrain_entries"], errors)
    if errors:
        raise SplitError("the real-data pins do not hold (pass --testing only for a synthetic world): %s"
                         % "; ".join(errors))
    eval_guard = G.GuardV2(entries)

    # base B's part and its inputs
    sel_sum, sel_rec = _opt_json(select_summary_path())
    if sel_sum is None:
        raise SplitError("%s missing: base_selected.jsonl cannot be checked against its recorded sha256"
                         % select_summary_path())
    bs_path = base_selected_path()
    if not bs_path.is_file():
        raise SplitError("%s missing" % bs_path)
    bs_sha = C1.sha256_file(bs_path)
    rec_sha = ((sel_sum.get("outputs") or {}).get("base_selected.jsonl") or {}).get("sha256")
    if bs_sha != rec_sha:
        raise SplitError("%s hashes to %s, select_summary.json records %s" % (bs_path, bs_sha[:12], str(rec_sha)[:12]))
    _pin(testing, "base_selected.jsonl's sha256", bs_sha, REAL_PINS["base_selected"], errors)
    base_rows = C1.read_manifest(bs_path)
    _pin(testing, "the base_selected row count", len(base_rows), REAL_PINS["rows"]["base_selected"], errors)
    if errors:
        raise SplitError("the real-data pins do not hold (pass --testing only for a synthetic world): %s"
                         % "; ".join(errors))
    inputs.update(base_selected=C2.file_record(bs_path), select_summary=sel_rec)
    pool_sum, rec = _opt_json(pool_summary_path())
    if rec:
        inputs["pool_summary"] = rec

    # licences and lab groups
    dom_path = Path(domain_config or default_domain_config())
    domain_raw = _read_json(dom_path, "the funnel domain config")
    inputs["funnel_domain"] = C2.file_record(dom_path)
    cards, rec = _opt_json(cards_index_path())
    if rec:
        inputs["cards_index"] = rec
    tsw_rec, rec = _opt_json(tsw_record_path or default_tsw_record())
    if rec:
        inputs["tsw_zenodo_record"] = rec
    table = None
    if licences_path:
        table = _read_json(licences_path, "the licence table")
        inputs["licences"] = C2.file_record(licences_path)
    lic = Licences(cards, domain_raw, tsw_rec, table)
    groups = (domain_raw.get("sources") or {}).get("lab_groups") or {}

    # dHashes
    core = sorted(v1rows["train_core"], key=lambda r: r["key"])
    for r in base_rows:
        r.setdefault("session", "")
    tsw_rows = {}
    for split, v1split in TSW.items():
        rows = []
        for r in sorted(v1rows[v1split], key=lambda r: r["key"]):
            key = tsw_key(split, r["key"])
            rows.append({"image": r["image"], "label": str(label_dir(split) / (key + ".txt")),
                         "v1_label": r["label"], "sha256": r["sha256"], "label_sha256": r["label_sha256"],
                         "source": TSW_SOURCE[split], "session": _session_of_image(r["image"]), "key": key,
                         "v1_key": r["key"], "v1_source": r.get("source")})
        tsw_rows[split] = rows
    all_rows = core + tsw_rows["tsw22"] + tsw_rows["tsw23"] + base_rows
    hash_rows(all_rows, "base v2 candidates", procs)
    bad_core = [r["key"] for r in core if r["dhash"] is None or r["variants"] is None]
    if bad_core:
        raise SplitError("%d train_core image(s) cannot be hashed (e.g. %s); v1 hashed them all"
                         % (len(bad_core), bad_core[:3]))

    # train_core against the eight variants (docs/CONTINUOUS_LOOP.md 8, never-train v2): v1 checked the stored
    # dHash only, so a flipped or rotated evaluation copy may sit in train_core. Decision L-8: such a row is
    # dropped from base v2 and from train_core.jsonl through a recorded list, up to the cap; a hit on the stored
    # dHash itself means v1 and this build disagree, and it refuses (R4)
    core_drops, core_refused = [], []
    for r in core:
        reason, match = eval_guard.check(r["dhash"], r["variants"])
        if reason == VARIANT_DROP_REASON:
            core_drops.append((r, match))
        elif reason:
            core_refused.append({"key": r["key"], "reason": reason, "match": match})
    if core_refused:
        raise SplitError(
            "R4 INCIDENT: %d train_core image(s) are refused by the never-train guard on their stored dHash (e.g. "
            "%s). v1 compared exactly that, so v1 and this build disagree; decision L-8 covers flips and rotations "
            "only. The build is refused and the owner decides. Nothing was written." % (len(core_refused),
                                                                                       core_refused[:5]))
    cap = variant_drop_cap(len(core))
    drop_rows = [{"key": r["key"], "reason": VARIANT_DROP_REASON, "match": m} for r, m in core_drops]
    if len(core_drops) > cap:
        raise SplitError(
            "R4 INCIDENT: %d train_core image(s) are within %d bits of an evaluation image under a flip or rotation, "
            "more than the %d decision L-8 lets a build drop by itself (max(%d, %.1f %% of %d); e.g. %s). The build "
            "is refused and the owner decides. Nothing was written."
            % (len(core_drops), BITS, cap, VARIANT_DROP_CAP_MIN, 100 * VARIANT_DROP_CAP_FRAC, len(core),
               drop_rows[:5]))
    if core_drops:
        log("INCIDENT (decision L-8): %d train_core image(s) are within %d bits of an evaluation image under a flip "
            "or rotation (%s). They are dropped from base v2 and train_core.jsonl and listed in %s. %s"
            % (len(core_drops), BITS, drop_rows, C2.VARIANT_DROPS_NAME,
               V1_RESULTS_NOTE % (len(core_drops), len(core))))
    drop_keys = {r["key"] for r, _m in core_drops}
    drop_shas = {r["sha256"] for r, _m in core_drops}
    core_kept = [r for r in core if r["key"] not in drop_keys]
    if sum(1 for r in core if r["sha256"] in drop_shas) != len(core_drops):
        raise SplitError("a dropped train_core image shares its bytes with a kept row; the L-8 list is by image "
                         "sha256, so the drop would not be exact")

    # base B: the recorded bytes, L-5, the H6(b) check
    shas = dict(_pmap(_sha_task, sorted({r["image"] for r in base_rows} | {r["label"] for r in base_rows}),
                      procs, "sha256 base B"))
    changed = [r["key"] for r in base_rows
               if shas.get(r["image"]) != r["sha256"] or shas.get(r["label"]) != r["label_sha256"]]
    if changed:
        raise SplitError("%d base_selected row(s) no longer hash as recorded (e.g. %s)" % (len(changed), changed[:3]))

    # the embedding scan (4.2 step 5): run here when no scan covers the current candidates (scan_mode
    # "auto"), after every worker process above has ended, so the embedder is loaded last
    scan = _load_scan()
    want_scan = {(t, tsw_key(t, r["key"]), r["sha256"]) for t, v in TSW.items() for r in v1rows[v]}
    want_scan |= {(BASE_B, r["key"], r["sha256"]) for r in base_rows}
    current = scan is not None and _scan_current(scan, want_scan, lock1, bs_sha, inputs["funnel_domain"]["sha256"])
    if scan_mode == "auto" and not current:
        log("the embedding copy scan %s: running it now (scan_mode auto)"
            % ("does not cover the current candidates" if scan is not None else "has not run"))
        scan = _scan(embedder, calibration_path, procs, False, dom_path, testing)
        current = _scan_current(scan, want_scan, lock1, bs_sha, inputs["funnel_domain"]["sha256"])
        if not current:
            raise SplitError("the embedding scan just written does not cover the build's candidates")
    scanned, hits, unscannable = _scan_index(scan)
    ecal = None
    if scan is not None:
        inputs["embed_scan"] = C2.file_record(embed_scan_path())
        # L-9(c): the threshold for images without capture provenance (made by the scan, or here when the
        # scan is current but the calibration is not; --skip-scan only applies a current one)
        ecal = _calibrate_v2(scan, embedder, dom_path, domain_raw, procs, testing, compute=(scan_mode == "auto"))
        if ecal is not None:
            inputs["embed_calibration_v2"] = C2.file_record(embed_calibration_path())
        else:
            log("WARNING: no current v2 embedding calibration (build --skip-scan): every scan hit and every v2 "
                "funnel entry applies at the scan's own threshold, and lock refuses until `scan` or `build` "
                "writes %s" % EC.NAME)
    t2 = ecal["cos_threshold"] if ecal is not None else None
    t_strict = ecal["strict_threshold"] if ecal is not None else None
    funnel_matches, funnel_rec = funnel_base_b_matches()
    if funnel_rec.get("path"):
        inputs["funnel_leak_v1"] = {"path": funnel_rec["path"], "sha256": funnel_rec["sha256"]}

    # sessions
    dev_sessions = set(summ1.get("dev", {}).get("sessions") or ())
    if dev_sessions != {r["session"] for r in v1rows["dev"]}:
        raise SplitError("the v1 summary's dev sessions differ from the dev manifest's")
    sess = {"dev": dev_sessions, "test": {r["session"] for r in v1rows["test"]} - {""},
            "train_core": {r["session"] for r in core_kept} - {""}}

    # 4-5. decisions, part by part, in base order; every candidate's copy rule recorded (L-9)
    drops = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    overlap = {}
    rules = []
    earlier = NearHashIndex()
    for r in core:                      # every v1 train_core image, the L-8 drops too: a near copy of one is dropped
        earlier.add(r["dhash"], ("train_core", r["key"]), max_bits=BITS)
    kept = {"train_core": core_kept}

    def drop(part, r, reason, detail=None):
        drops[part][reason] += 1
        if len(examples[part]) < 50:
            examples[part].append({"key": r["key"], "reason": reason, "match": detail or {}})

    for split in ("tsw22", "tsw23"):
        keep, ov = [], {"dev": 0, "test": 0, "train_core": 0, "sessions": collections.Counter()}
        for r in tsw_rows[split]:
            for where in ("dev", "test", "train_core"):
                if r["session"] in sess[where]:
                    ov[where] += 1
                    ov["sessions"]["%s|%s" % (where, r["session"])] += 1
            ev = embed_evidence(hits, unscannable, (split, r["key"], r["sha256"]), t2, t_strict)
            rule = tsw_rule(r["session"], sess["dev"], sess["test"])
            reason, match = eval_guard.check(r["dhash"], r["variants"])
            if not reason and rule == RULE_TSW_EVAL_SESSION:
                where = "dev" if r["session"] in sess["dev"] else "test"
                reason, match = "%s_session" % where, {"session": r["session"]}
            if not reason and ev["unscannable"]:
                reason, match = "unhashable_embed", {"why": "the image cannot be described"}
            if not reason and rule == RULE_TSW_EMBED and ev["applied_hit"]:
                reason, match = "near_eval_embed", ev["match"]
            if not reason:
                hit = _near_earlier(earlier, r)
                if hit:
                    reason, match = "near_%s" % hit[0][0], {"with": hit[0][1], "bits": hit[1], "variant": hit[2]}
            rules.append(_rule_row(split, r, rule, reason, match, ev, None, "dropped" if reason else "kept"))
            if reason:
                drop(split, r, reason, match)
                continue
            keep.append(r)
        for r in keep:
            earlier.add(r["dhash"], (split, r["key"]), max_bits=BITS)
        ov["sessions"] = dict(sorted(ov["sessions"].items()))
        overlap[split] = ov
        kept[split] = keep

    incident, unchecked, refused, l5_rows, keep = [], [], [], [], []
    for r in sorted(base_rows, key=lambda r: r["key"]):
        excluded = r["source"] in L5_SOURCES
        rule = RULE_L5 if excluded else RULE_BASE_B
        ev = embed_evidence(hits, unscannable, (BASE_B, r["key"], r["sha256"]), t2, t_strict)
        fun = funnel_verdict((funnel_matches.get(r["key"]) or []) + (funnel_matches.get(str(r["image"])) or []), t2)
        reason, match = eval_guard.check(r["dhash"], r["variants"])
        if not reason and ev["unscannable"]:
            reason, match = "unhashable_embed", {"why": "the image cannot be described"}
        if not reason and ev["applied_hit"]:
            reason, match = "near_eval_embed", ev["match"]
        if not reason and fun["applied"]:
            reason, match = FUNNEL_REASON, fun["applied"]
        what = {"key": r["key"], "source": r["source"], "reason": reason, "match": match or {}}
        if excluded:
            l5_rows.append(r)
            rules.append(_rule_row(BASE_B, r, rule, reason, match, ev, fun, "excluded"))
            if reason in COPY_REASONS:
                incident.append(what)
            elif reason:
                unchecked.append(what)
            continue
        if reason:
            refused.append(what)          # a copy, or an image that cannot be compared (fail closed)
            continue
        hit = _near_earlier(earlier, r)
        if hit:
            reason, match = "near_%s" % hit[0][0], {"with": hit[0][1], "bits": hit[1], "variant": hit[2]}
            rules.append(_rule_row(BASE_B, r, rule, reason, match, ev, fun, "dropped"))
            drop(BASE_B, r, reason, match)
            continue
        rules.append(_rule_row(BASE_B, r, rule, None, None, ev, fun, "kept"))
        keep.append(r)
    kept[BASE_B] = keep
    if refused:
        copies = [w for w in refused if w["reason"] in COPY_REASONS]
        raise SplitError(
            "R4 INCIDENT (funnel H6(b)): %d image(s) of base B's part that base v2 would keep are copies of an "
            "evaluation image, %d cannot be compared with one (%s). A copy voids B's and realloop_v1's results; "
            "the build is refused and the owner decides (docs/CONTINUOUS_LOOP.md 4.2 step 5). Judged at %s. "
            "Nothing was written."
            % (len(copies), len(refused) - len(copies), refused[:5],
               "the v2 threshold %.6f (L-9(c))" % t2 if t2 is not None
               else "the scan's own threshold (no v2 calibration)"))
    if incident:
        log("R4 INCIDENT (funnel H6(b)) in base B's L-5 part: %d image(s) copy an evaluation image (%s). They "
            "are in no v2 manifest, so the build goes on; B's and realloop_v1's results are void (recorded in "
            "summary.json and LOCK.json)." % (len(incident), incident[:3]))
    l5_counts = collections.Counter(r["source"] for r in l5_rows)
    _pin(testing, "the L-5 excluded image count", len(l5_rows), REAL_PINS["l5_excluded"], errors)
    if errors:
        raise SplitError("the real-data pins do not hold: %s" % "; ".join(errors))

    # 8. base v2 and its disjointness
    base_v2 = []
    for part in PARTS:
        for r in kept[part]:
            base_v2.append(dict(r, part=part))
    problems = disjoint_problems(base_v2)
    if problems:
        raise SplitError("base v2 is not pairwise disjoint across parts: %s" % problems[:5])

    # provenance (with the copy rule that judged each row, L-9)
    rule_of = {x["key"]: x["rule"] for x in rules}
    per_source = {}
    prov = []
    for r in sorted(base_v2, key=lambda r: r["key"]):
        if r["source"] not in per_source:
            text, basis = lic.lookup(r["source"])
            per_source[r["source"]] = {"licence": text, "basis": basis, "research_only": research_only(text),
                                       "lab_group": lab_group(groups, r["source"], r["part"]), "rows": 0}
        ps = per_source[r["source"]]
        ps["rows"] += 1
        prov.append({"key": r["key"], "part": r["part"], "source": r["source"], "session": r["session"],
                     "image": r["image"], "sha256": r["sha256"], "label_sha256": r["label_sha256"],
                     "dhash": int(r["dhash"]), "variants": [int(x) for x in r["variants"]],
                     "licence": ps["licence"], "licence_basis": ps["basis"], "research_only": ps["research_only"],
                     "lab_group": ps["lab_group"], "origin_key": r.get("v1_key") or r["key"],
                     "copy_rule": RULE_TRAIN_CORE if r["part"] == "train_core" else rule_of[r["key"]]})
    evidence = {}
    for s in sorted({r["source"] for r in base_rows}):
        v = ((pool_sum or {}).get("per_slug") or {}).get(s)
        if v is not None:
            evidence[s] = {k: v.get(k) for k in ("kept", "cwd12_copies", "near_eval_by_split", "dropped")}

    # train_core v2 (L-8): v1's bytes minus the dropped rows, every other line byte-identical
    core_bytes, filtered = C2.filter_manifest_bytes(copy_bytes["train_core"], drop_shas)
    left = sorted(json.loads(ln)["key"] for ln in core_bytes.splitlines() if ln.strip())
    if {r["key"] for r in filtered} != drop_keys or left != sorted(r["key"] for r in core_kept):
        raise SplitError("v1 train_core minus the L-8 rows is not the kept train_core")
    variant_drops = [dict({k: r[k] for k in C1.MANIFEST_KEYS}, dhash=int(r["dhash"]),
                          variants=[int(x) for x in r["variants"]], reason=VARIANT_DROP_REASON, match=m,
                          decided_by=L8_DECISION)
                     for r, m in sorted(core_drops, key=lambda t: t[0]["key"])]

    # ---- every check has passed: write
    C2.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    staged = C2.SPLITS_DIR / STAGING_NAME
    shutil.rmtree(staged, ignore_errors=True)
    for split in TSW:
        for r in kept[split]:
            dst = staged / split / (r["key"] + ".txt")
            if C2.atomic_copy(r["v1_label"], dst) != r["label_sha256"]:
                raise SplitError("label %s changed while it was copied" % r["v1_label"])
    for split in TSW:
        src = staged / split
        src.mkdir(parents=True, exist_ok=True)
        S1._swap_in(src, label_dir(split))
    shas = {}
    for split in C2.BYTE_COPIES:
        shas[split] = C2.write_bytes_atomic(C2.v2_manifest_path(split), copy_bytes[split])
        if shas[split] != lock1["manifests"][split]:
            raise SplitError("the byte copy of %s does not hash as v1" % split)
    shas["train_core"] = C2.write_bytes_atomic(C2.v2_manifest_path("train_core"), core_bytes)
    if (shas["train_core"] == lock1["manifests"]["train_core"]) != (not core_drops):
        raise SplitError("train_core v2 does not hash as v1 minus the L-8 rows")
    drops_sha = C2.write_jsonl_atomic(C2.TRAIN_CORE_VARIANT_DROPS, variant_drops)
    for split in TSW:
        shas[split] = C1.write_manifest(C2.v2_manifest_path(split), [_manifest_row(r) for r in kept[split]])
    shas[C2.BASE_MANIFEST] = C1.write_manifest(C2.v2_manifest_path(C2.BASE_MANIFEST),
                                               [_manifest_row(r) for r in base_v2])
    nt = {"entries": entries, "bits": BITS, "complete": False, "missing": [], "count_warnings": {},
          "min_expected": len(entries) + 1, "splits": list(C2.EVAL_SPLITS),
          "derived_from": {"v1_index_sha256": inputs["v1_nevertrain"]["sha256"]},
          "note": "INCOMPLETE until inc2.splits lock: every loader refuses it (min_expected > entries)"}
    nt_sha = C2.write_json_atomic(C2.NEVER_TRAIN_INDEX, nt)
    bc_entries = [[int(r["dhash"]), r["part"], r["key"]] for r in base_v2]
    bc = {"entries": bc_entries, "bits": BITS, "complete": False, "min_expected": len(bc_entries) + 1,
          "parts": list(PARTS), "note": "INCOMPLETE until inc2.splits lock"}
    bc_sha = C2.write_json_atomic(C2.BASE_COPIES_INDEX, bc)
    prov_sha = C2.write_jsonl_atomic(C2.BASE_PROVENANCE, prov)
    l5 = [{"key": r["key"], "source": r["source"], "image": r["image"], "sha256": r["sha256"],
           "dhash": r["dhash"], "variants": r["variants"], "decided_by": L5_DECISION}
          for r in sorted(l5_rows, key=lambda r: r["key"])]
    l5_sha = C2.write_jsonl_atomic(C2.L5_EXCLUDED, l5)
    rules.sort(key=lambda x: (PARTS.index(x["part"]), x["key"]))
    rules_sha = C2.write_jsonl_atomic(copy_rules_path(), rules)

    unresolved = sorted(s for s, v in per_source.items() if v["licence"] == UNRESOLVED)
    ro_rows = sum(v["rows"] for v in per_source.values() if v["research_only"])
    parts_n = {p: len(kept[p]) for p in PARTS}
    boxes = _counts(base_v2)
    summary = {
        "format": SUMMARY_FORMAT, "splits_version": C2.SPLITS_VERSION, "testing": bool(testing),
        "built_utc": C2.utc(), "build_seconds": round(time.time() - t0, 1),
        "splits_dir": str(C2.SPLITS_DIR), "inputs": inputs, "code": _code_record(),
        "manifests": {s: {"sha256": shas[s], "rows": n} for s, n in
                      [(s, len(v1rows[s])) for s in C2.BYTE_COPIES] + [("train_core", len(core_kept))]
                      + [(s, len(kept[s])) for s in TSW] + [(C2.BASE_MANIFEST, len(base_v2))]},
        "byte_copies": {s: {"sha256": shas[s], "v1_lock_sha256": lock1["manifests"][s], "identical": True}
                        for s in C2.BYTE_COPIES},
        "train_core": {"v1_rows": len(core), "rows": len(core_kept), "sha256": shas["train_core"],
                       "v1_lock_sha256": lock1["manifests"]["train_core"], "identical": not core_drops,
                       "derivation": "v1 train_core.jsonl minus the rows whose image sha256 %s lists (L-8); every "
                                     "other line byte-identical, in order" % C2.VARIANT_DROPS_NAME},
        "train_core_variant_drops": {
            "decided_by": L8_DECISION, "incident": bool(core_drops), "count": len(core_drops), "cap": cap,
            "cap_rule": "max(%d, floor(%.1f %% of the %d v1 train_core rows)); more refuses the build (R4)"
                        % (VARIANT_DROP_CAP_MIN, 100 * VARIANT_DROP_CAP_FRAC, len(core)),
            "keys": sorted(drop_keys), "matches": drop_rows, "file": str(C2.TRAIN_CORE_VARIANT_DROPS),
            "sha256": drops_sha, "excluded_from": ["train_core.jsonl", "base_v2.jsonl", "base_copies_dhash.json"],
            "v1_results": V1_RESULTS_NOTE % (len(core_drops), len(core)) if core_drops else None},
        "base_v2": {"images": len(base_v2), "parts": parts_n, "boxes": boxes, "boxes_total": sum(boxes.values()),
                    "otherplant_share": round(boxes["OtherPlant"] / max(1, sum(boxes.values())), 6),
                    "disjoint": "key, path, sha256 and 6-bit dHash (with the 8 variants) across parts"},
        "tsw": {s: {"v1_rows": len(tsw_rows[s]), "kept": len(kept[s]), "dropped": dict(drops[s]),
                    "examples": examples[s], "session_overlap": overlap[s]} for s in TSW},
        "v1_excluded": {"ood22": {"cwd12_twins": ((summ1.get("exam_cwd12_copies") or {}).get("ood22") or {})
                                  .get("dropped"),
                                  "dropped": ((summ1.get("dropped") or {}).get("ood22") or {}).get("images")},
                        "ood23": {"cwd12_twins": ((summ1.get("exam_cwd12_copies") or {}).get("ood23") or {})
                                  .get("dropped"),
                                  "dropped": ((summ1.get("dropped") or {}).get("ood23") or {}).get("images")},
                        "eval_cross_near_dups": summ1.get("eval_cross_near_dups")},
        "base_b": {"rows": len(base_rows), "kept": len(kept[BASE_B]), "dropped": dict(drops[BASE_B]),
                   "examples": examples[BASE_B], "sources": dict(collections.Counter(r["source"] for r in base_rows)),
                   "kept_sources": dict(collections.Counter(r["source"] for r in kept[BASE_B])),
                   "l5": {"decided_by": L5_DECISION, "sources": list(L5_SOURCES), "excluded": len(l5_rows),
                          "per_source": dict(l5_counts), "file": str(C2.L5_EXCLUDED), "sha256": l5_sha},
                   "h6b": {"refused_included": 0, "incident_l5_part": bool(incident), "incident": incident[:200],
                           "unchecked_l5_part": unchecked[:200],
                           "checks": ["8-variant dHash against dev, test and imageweeds",
                                      "embed scan" if scan is not None else "embed scan: not yet run",
                                      "funnel leak_v1 base_B copies: %s" % funnel_rec["state"]]},
                   "v1_pool_evidence": evidence},
        "funnel_leak_v1": funnel_rec,
        "nevertrain": {"entries": len(entries), "per_split": dict(collections.Counter(s for _h, s, _k in entries)),
                       "bits": BITS, "complete": False, "sha256": nt_sha},
        "base_copies": {"entries": len(bc_entries), "bits": BITS, "complete": False, "sha256": bc_sha},
        "provenance": {"file": str(C2.BASE_PROVENANCE), "sha256": prov_sha, "rows": len(prov)},
        "licences": {"per_source": per_source, "unresolved_sources": unresolved, "research_only_rows": ro_rows,
                     "exemption": LICENCE_EXEMPTION},
        "embed_scan": ({"applied": True, "sha256": inputs["embed_scan"]["sha256"],
                        "dropped": {s: drops[s].get("near_eval_embed", 0) + drops[s].get("unhashable_embed", 0)
                                    for s in TSW}}
                       if scan is not None else {"applied": False}),
        "dev_sessions": sorted(dev_sessions),
        "test_sessions": sorted(sess["test"]),
        "l9": _l9_summary(rules, rules_sha, ecal, scan),
        "embed_v2": _embed_v2_summary(ecal, scan, rules),
    }
    C2.write_json_atomic(summary_path(), summary)
    log("built in %.0fs: base_v2 %d images (%s); tsw drops %s; L-5 excluded %d; L-8 train_core drops %d; "
        "never-train %d entries; copy rules %s; v2 threshold %s%s"
        % (time.time() - t0, len(base_v2), parts_n, {s: dict(drops[s]) for s in TSW}, len(l5_rows), len(core_drops),
           len(entries), dict(collections.Counter(x["rule"] for x in rules)), t2, "; TESTING world" if testing else ""))
    if scan is None:
        log("next: `scan` (the embedding copy scan), then `build` again (it applies the scan), then `lock`")
    return summary


def _l9_summary(rules, rules_sha, ecal, scan):
    """summary.json's l9 block: per tsw split and for base B, the copy rules
    applied, what the scan flagged at its own threshold and what was applied,
    the funnel entries by verdict (L-9(b))."""
    out = {"decided_by": L9_DECISION,
           "copy_rules": {"file": str(copy_rules_path()), "sha256": rules_sha, "rows": len(rules)},
           "scan_threshold": (scan or {}).get("detector", {}).get("cos_threshold") if scan else None,
           "v2_threshold": ecal["cos_threshold"] if ecal else None}
    for part in list(TSW) + [BASE_B]:
        rs = [x for x in rules if x["part"] == part]
        flagged = [x for x in rs if x["embed"]["base_hit"]]
        exempt = [x for x in flagged if not x["embed_threshold_applies"]]
        cs = sorted(float(x["embed"]["base_hit"]["cos"]) for x in exempt)
        rec = {"rules": dict(collections.Counter(x["rule"] for x in rs)),
               "decisions": dict(collections.Counter("%s|%s" % (x["rule"], x["decision"]) for x in rs)),
               "flagged_at_scan_threshold": len(flagged),
               "flagged_where_the_threshold_applies": len(flagged) - len(exempt),
               "copies_at_v2_threshold_where_it_applies": sum(1 for x in rs if x["embed_threshold_applies"]
                                                              and x["embed"]["applied_hit"]),
               "flagged_exempt": len(exempt), "flagged_exempt_kept": sum(1 for x in exempt if x["decision"] == "kept"),
               "flagged_exempt_at_v2_threshold": sum(1 for x in exempt if x["embed"]["applied_hit"]),
               "flagged_exempt_cos": ({"min": cs[0], "median": cs[len(cs) // 2], "max": cs[-1]} if cs else None),
               "flagged_exempt_eval_splits": dict(collections.Counter(x["embed"]["base_hit"]["eval_split"]
                                                                      for x in exempt))}
        fun = collections.Counter(e["verdict"] for x in rs for e in x.get("funnel") or ())
        if fun:
            rec["funnel_entries"] = dict(fun)
            rec["funnel_images"] = {v: sum(1 for x in rs if any(e["verdict"] == v for e in x.get("funnel") or ()))
                                    for v in (FUNNEL_APPLIED, FUNNEL_NOT_V2, FUNNEL_BELOW_V2)}
        out[part] = rec
    return out


def _embed_v2_summary(ecal, scan, rules):
    """summary.json's embed_v2 block: the v2 calibration applied, and per
    source the embedding hits against the count its per-image
    false-positive rate predicts (EC.source_verdict; informational: a
    source's removal from base v2 is the owner's decision, R4)."""
    if ecal is None:
        return {"applied": False, "why": "no scan yet" if scan is None else "no current v2 calibration "
                                                                            "(build --skip-scan)"}
    cal = ecal["calibration"]
    per = collections.defaultdict(lambda: {"images": 0, "hits": 0, "strict": 0, "dhash": 0})
    for x in rules:
        if x["embed"]["unscannable"]:
            continue
        v = per["%s|%s" % (x["part"], x["source"])]
        v["images"] += 1
        v["hits"] += int(x["embed"]["applied_hit"])
        v["strict"] += int(x["embed"]["strict_hit"])
        v["dhash"] += int(x["reason"] in ("near_eval_v2", "near_eval_variant") or x["embed"]["dhash_hit"])
    per_source = {k: EC.source_verdict(v["images"], v["hits"], v["strict"], v["dhash"], ecal["p_false"])
                  for k, v in sorted(per.items())}
    return {"applied": True, "decided_by": EC.DECISION, "file": ecal["file"], "cos_threshold": ecal["cos_threshold"],
            "strict_threshold": ecal["strict_threshold"], "base": cal.get("base"), "p_false": ecal["p_false"],
            "known_limits": ecal["known_limits"],
            "negatives": {t: {k: v.get(k) for k in ("n", "false_hits", "fpr", "ub", "at_base", "constraining")}
                          for t, v in (cal.get("negatives") or {}).items()},
            "recall": {f: {k: v.get(k) for k in ("n", "hits", "recall", "lb", "at_base")}
                       for f, v in (cal.get("positives") or {}).items()},
            "per_source": per_source,
            "per_source_note": "informational: a flagged source is the owner's decision (R4); single images are "
                               "judged by the copy rules"}


def _near_earlier(index, r):
    """((part, key), bits, variant) of the nearest image of an earlier part
    within 6 bits of the row's dHash or one of its variants, else None."""
    vs = list(zip(G.VARIANTS, r["variants"]))
    best = None
    for name, h in vs:
        m = index.find(h)
        if m is not None and (best is None or m[1] < best[1]):
            best = (m[0], m[1], name)
    return best


def disjoint_problems(rows):
    """Clashes between rows of different parts: the same key, image path or
    sha256, or a dHash (or variant) within 6 bits of another part's dHash."""
    probs = []
    for field in ("key", "image", "sha256"):
        seen = {}
        for r in rows:
            v = r[field]
            if v in seen and seen[v][0] != r["part"]:
                probs.append("%s %s in %s and %s" % (field, v, seen[v][0], r["part"]))
            elif field == "key" and v in seen:
                probs.append("key %s twice in %s" % (v, r["part"]))
            seen.setdefault(v, (r["part"], r["key"]))
    by_part = collections.defaultdict(NearHashIndex)
    for r in rows:
        by_part[r["part"]].add(int(r["dhash"]), r["key"], max_bits=BITS)
    for r in rows:
        for part, idx in by_part.items():
            if part == r["part"]:
                continue
            for h in r["variants"]:
                m = idx.find(int(h))
                if m is not None:
                    probs.append("%s (%s) within %d bits of %s (%s)" % (r["key"], r["part"], m[1], m[0], part))
                    break
    return probs


# --------------------------------------------------------------- the scan
def _load_scan():
    p = embed_scan_path()
    if not p.is_file():
        return None
    doc = _read_json(p, "embed_scan.json")
    if doc.get("format") != SCAN_FORMAT or doc.get("status") != "complete":
        raise SplitError("%s is not a complete %s record" % (p, SCAN_FORMAT))
    return doc


def _scan_index(scan):
    """({(set, key, sha256)} scanned, {(set, key, sha256): [every hit]},
    {(set, key, sha256)} that could not be described) of a scan. Each hit is
    {eval_split, eval_key, cos, bits, variant} (evaluation keys only)."""
    if scan is None:
        return set(), {}, set()
    scanned = set()
    sha = {}
    for s, rows in (scan.get("scanned") or {}).items():
        for key, sha256 in rows:
            scanned.add((s, key, sha256))
            sha[(s, key)] = sha256
    hits = collections.defaultdict(list)
    for h in scan.get("hits") or ():
        k = (h["set"], h["key"], sha.get((h["set"], h["key"])))
        hits[k].append({kk: h[kk] for kk in ("eval_split", "eval_key", "cos", "bits", "variant")})
    uns = {(u["set"], u["key"], sha.get((u["set"], u["key"]))) for u in scan.get("unscannable") or ()}
    return scanned, dict(hits), uns


def _cos(e):
    try:
        return float(e.get("cos"))
    except (TypeError, ValueError):
        return float("-inf")


def _bits(e):
    try:
        return int(e.get("bits"))
    except (TypeError, ValueError):
        return 65


def embed_evidence(hits, unscannable, k, threshold, strict=None):
    """What the scan says about one candidate (set, key, sha256), judged at
    `threshold` (the v2 one; None: the scan's own, every hit applies): the
    best hit at the scan's threshold, whether a hit is a copy at `threshold`
    (cos >= it or within 6 dHash bits under a variant) and its best entry,
    a hit at or above the strict threshold, a dHash hit, and whether the
    image could not be described."""
    entries = list(hits.get(k) or ())
    best = max(entries, key=lambda e: (_cos(e), -_bits(e))) if entries else None
    applied = [e for e in entries if threshold is None or _bits(e) <= BITS or _cos(e) >= threshold]
    top = max(applied, key=lambda e: (_cos(e), -_bits(e))) if applied else None
    return {"unscannable": k in unscannable, "base_hit": dict(best) if best else None, "n_base_hits": len(entries),
            "applied_hit": bool(applied), "match": dict(top, n_copies=len(applied)) if top else None,
            "strict_hit": strict is not None and any(_cos(e) >= strict for e in entries),
            "dhash_hit": any(_bits(e) <= BITS for e in entries), "threshold": threshold}


def tsw_rule(session, dev_sessions, test_sessions):
    """L-9(a): the copy rule of a tsw row by its capture session."""
    if session and (session in dev_sessions or session in test_sessions):
        return RULE_TSW_EVAL_SESSION
    if EC.capture_session(session):
        return RULE_TSW_PROVENANCE
    return RULE_TSW_EMBED


def funnel_verdict(entries, threshold):
    """L-9(b): {"applied": the best entry that is a v2 copy or None,
    "entries": every distinct entry with its verdict}. An entry against an
    evaluation split outside v2 (ood22, ood23) is not_v2_split; one against
    a v2 split is applied when it is within 6 dHash bits or at or above
    `threshold` (None: always, the scan's own threshold), else
    below_v2_threshold. An entry whose numbers cannot be read is applied
    (fail closed)."""
    out, seen = [], set()
    for e in entries or ():
        ident = (e.get("eval_split"), e.get("eval_key"))
        if ident in seen:
            continue
        seen.add(ident)
        if e.get("eval_split") not in C2.EVAL_SPLITS:
            verdict = FUNNEL_NOT_V2
        elif threshold is None or _bits(e) <= BITS or _cos(e) >= threshold or not math.isfinite(_cos(e)):
            verdict = FUNNEL_APPLIED
        else:
            verdict = FUNNEL_BELOW_V2
        out.append({"eval_split": e.get("eval_split"), "eval_key": e.get("eval_key"), "cos": e.get("cos"),
                    "bits": e.get("bits"), "variant": e.get("variant"), "verdict": verdict})
    applied = [e for e in out if e["verdict"] == FUNNEL_APPLIED]
    best = max(applied, key=lambda e: (_cos(e), -_bits(e))) if applied else None
    return {"applied": dict({k: best[k] for k in ("eval_split", "eval_key", "cos", "bits", "variant")},
                            source="funnel leak_v1.json", n_copies=len(applied)) if best else None,
            "entries": out}


def _rule_row(part, r, rule, reason, match, ev, fun, decision):
    """One row of copy_rules.jsonl: the candidate, the copy rule that judged
    it, its decision and the evidence (evaluation keys only)."""
    row = {"part": part, "key": r["key"], "sha256": r["sha256"], "source": r["source"],
           "session": r.get("session") or "", "rule": rule, "decision": decision, "reason": reason,
           "match": match or None, "decided_by": L9_DECISION,
           "embed_threshold_applies": rule in (RULE_TSW_EMBED, RULE_BASE_B, RULE_L5),
           "embed": {k: ev[k] for k in ("unscannable", "base_hit", "n_base_hits", "applied_hit", "strict_hit",
                                        "dhash_hit", "threshold")}}
    if fun is not None and fun["entries"]:
        row["funnel"] = fun["entries"]
    return row


def _scan_current(scan, want, lock1, bs_sha, domain_sha):
    """True when embed_scan.json was made from the current candidates: the
    (set, key, sha256) of every v1 ood22 / ood23 row under its tsw key and of
    every base_selected row, the v1 LOCK's evaluation and ood manifests,
    base_selected.jsonl and the funnel domain config (which names the
    embedder, the augmentation families and the negative groups)."""
    if scan is None:
        return False
    inp = scan.get("inputs") or {}
    mans = inp.get("manifests") or {}
    if any(mans.get(s) != lock1["manifests"].get(s) for s in ("ood22", "ood23") + C2.EVAL_SPLITS):
        return False
    if (inp.get("base_selected") or {}).get("sha256") != bs_sha:
        return False
    if (inp.get("funnel_domain") or {}).get("sha256") != domain_sha:
        return False
    if _scan_inputs_changed(scan):
        return False
    scanned, _hits, _uns = _scan_index(scan)
    return scanned == want


def _scan_inputs_changed(scan):
    """Why the calibration file or the evaluation descriptors the scan used
    no longer hash as it recorded (a funnel leak rerun, a deleted cache), or
    None. Such a scan is stale: its hits were cut at another threshold, and
    the v2 calibration (L-9(c)) is built on exactly those two files."""
    det = scan.get("detector") or {}
    for what, rec in (("calibration", det.get("calibration") or {}),
                      ("evaluation descriptors", det.get("eval_descriptors") or {})):
        p = rec.get("path")
        if not p or not Path(p).is_file() or C1.sha256_file(p) != rec.get("sha256"):
            return "the %s the scan used (%s) is missing or changed since the scan" % (what, p)
    return None


def funnel_base_b_matches(path=None):
    """({key or image path: [every match entry]}, record) of the base B
    images the funnel's pre-registered H6 scan (leak_v1.json, status
    complete, passed calibration) lists as copies of an evaluation image,
    whatever the evaluation split: which of them apply is funnel_verdict's
    (L-9(b)). The listing in leak_v1.json is capped (funnel.leak.LISTED_MAX
    pairs); when it may be truncated (as long as the cap, or fewer images
    than the count), every base_B copy pair is read from the pairs file it
    records, checked by sha256, and a pairs file that cannot be read refuses
    (fail closed): a match against a v2 split beyond the cap must not be
    missed. No file, or a run that is not complete: ({}, state "pending"),
    which the LOCK records."""
    from ..funnel import leak as L
    p = Path(path or funnel_leak_path())
    if not p.is_file():
        return {}, {"state": "pending", "path": None, "sha256": None}
    doc = _read_json(p, "leak_v1.json")
    rec = {"path": str(p), "sha256": C1.sha256_file(p), "status": doc.get("status")}
    if doc.get("status") != "complete" or not (doc.get("calibration") or {}).get("ok"):
        return {}, dict(rec, state="not complete (%s)" % doc.get("status"))
    b = (doc.get("scans") or {}).get("base_B")
    if not isinstance(b, dict):
        raise SplitError("%s is complete but holds no base_B scan; base B's H6(b) result cannot be read" % p)
    listed = [e for e in b.get("listed") or [] if isinstance(e, dict)]
    n = int(b.get("copies") or 0)
    entries = list(listed)
    truncated = len(listed) >= L.LISTED_MAX or n > len({e.get("key") for e in listed})
    if truncated:
        pc = doc.get("pairs_csv") or {}
        pp = Path(pc.get("path") or (p.parent / "leak_pairs_v1.csv"))
        if not pp.is_file() or (pc.get("sha256") and C1.sha256_file(pp) != pc["sha256"]):
            raise SplitError("%s lists %d of base B's %d copies and its pairs file %s is missing or changed; the "
                             "rest cannot be read (fail closed)" % (p, len(listed), n, pp))
        import csv
        with open(pp, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("set") == "base_B" and row.get("kind") == "copy":
                    entries.append({"key": row["key"], "eval_split": row["eval_split"], "eval_key": row["eval_key"],
                                    "cos": float(row["cos"]), "bits": int(row["bits"]), "variant": row["variant"]})
    out = collections.defaultdict(list)
    for e in entries:
        m = {k: e.get(k) for k in ("eval_split", "eval_key", "cos", "bits", "variant")}
        m["source"] = "funnel leak_v1.json"
        for k in {e.get("key"), e.get("image")}:
            if k:
                out[str(k)].append(m)
    splits_hit = collections.Counter(e.get("eval_split") for e in entries)
    return dict(out), dict(rec, state="complete", base_b_copies=n, pairs_read=bool(truncated),
                           base_copy=bool((doc.get("h6b") or {}).get("base_copy")),
                           entries_by_split=dict(sorted((str(k), v) for k, v in splits_hit.items())))


def _embedder_name(domain_raw):
    from ..funnel import embed as E
    model, pooling = E.features_config(domain_raw)
    return E.embedder_name(model, pooling)


def _calibrate_v2(scan, embedder, dom_path, domain_raw, procs, testing, force=False, compute=True):
    """The v2 embedding calibration (L-9(c), inc2.embed_calibration) for the
    calibration and evaluation descriptors this scan used, written to
    splits/v2/embed_calibration_v2.json: loaded when current, else computed
    (compute True) or None (compute False: build --skip-scan). The base is
    the scan's calibration, whose file must still hash as the scan recorded;
    the evaluation descriptors are the scan's cache, likewise. Reads v1's
    train_core, dev, test and imageweeds (each checked against the v1
    LOCK), Step 1's pool.jsonl (the provenance-disjoint and easy negatives)
    and the funnel config. The embedder is made only when something must be
    described."""
    det = scan.get("detector") or {}
    stale = _scan_inputs_changed(scan)
    if stale:
        if not compute:
            log("WARNING: %s; the v2 calibration is not applied (build --skip-scan)" % stale)
            return None
        raise SplitError("%s: run `scan` (or build) again" % stale)
    bp = (det.get("calibration") or {}).get("path")
    ev = det.get("eval_descriptors") or {}
    cache = ev.get("path")
    name = embedder.name if embedder is not None else _embedder_name(domain_raw)
    try:
        base = EC.base_info(bp, det.get("calibration_source"), name)
    except C2.Inc2Error as e:
        raise SplitError("the scan's calibration cannot be the v2 calibration's base: %s" % e)
    _sets, eval_rows, inputs = scan_candidates()
    lock1 = C1.read_lock()
    core_path = C1.manifest_path("train_core")
    if C1.sha256_file(core_path) != lock1["manifests"]["train_core"]:
        raise SplitError("v1 train_core does not match the v1 LOCK")
    core_rows = C1.read_manifest(core_path)
    pool_rows = C1.read_manifest(pool_path()) if pool_path().is_file() else []
    recall_min, fpr_max, gates = _h6_gates()
    summ1 = _read_json(C1.SPLITS_DIR / S1.SUMMARY_NAME, "v1 summary.json")
    holder = {"emb": embedder}

    def get_embedder():
        if holder["emb"] is None:
            holder["emb"] = _make_embedder(domain_raw)
        return holder["emb"]

    def get_index():
        return G.eval_index(eval_rows, get_embedder(), cache, procs=procs)
    inputs.update(train_core=C2.file_record(core_path), funnel_domain=C2.file_record(dom_path), gates=gates,
                  embed_scan=C2.file_record(embed_scan_path()), eval_descriptors=dict(ev),
                  pool=C2.file_record(pool_path()) if pool_path().is_file() else None)
    try:
        return EC.calibrate(embed_calibration_path(), base, name, get_embedder, get_index, ev["sha256"], eval_rows,
                            core_rows, pool_rows, domain_raw, leak_dir(), recall_min, fpr_max,
                            min_negatives=EC.MIN_NEGATIVES_TESTING if testing else EC.MIN_NEGATIVES,
                            dev_sessions=(summ1.get("dev") or {}).get("sessions") or (), procs=procs,
                            testing=testing, force=force, inputs=inputs, domain_sha=C1.sha256_file(dom_path),
                            compute=compute)
    except C2.Inc2Error as e:
        raise SplitError("the v2 embedding calibration (L-9(c)): %s" % e)


def _make_embedder(domain_raw):
    """The domain config's descriptor embedder (funnel.embed.LazyEmbedder),
    offline: compute nodes have no internet, and a model missing from the
    Hugging Face cache must fail at once rather than wait on the network."""
    from ..funnel import embed as E
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model, pooling = E.features_config(domain_raw)
    return E.LazyEmbedder(model, pooling)


def _h6_gates():
    """(recall_min, fpr_max, basis): the funnel prereg's H6 gates, else the contract's."""
    from ..funnel import leak as L
    pre, rec = _opt_json(funnel_prereg_path())
    h6 = (((pre or {}).get("hypotheses") or {}).get("H6") or {}).get("calibration") or {}
    if "recall_min_per_family" in h6 and "fpr_max" in h6:
        return float(h6["recall_min_per_family"]), float(h6["fpr_max"]), {"prereg": rec}
    return L.RECALL_MIN, L.FPR_MAX, {"contract": "funnel.leak.RECALL_MIN / FPR_MAX"}


def scan(embedder=None, calibration_path=None, procs=PROCS, force=False, domain_config=None, testing=False):
    """The embedding copy scan (module docstring). Returns embed_scan.json's content."""
    with writer():
        return _scan(embedder, calibration_path, procs, force, domain_config, testing)


def scan_candidates():
    """({set: rows}, {split: eval rows}, inputs): every row a build could put
    into base v2 besides train_core (all v1 ood22 / ood23 rows under their tsw
    keys, all of base_selected.jsonl), and the evaluation rows (the v1 files,
    which the v2 byte copies equal). Independent of any build, so a scan never
    forgets a row an earlier build dropped. Each file is checked against the
    sha256 its producer recorded."""
    lock1 = C1.read_lock()
    inputs = {"v1_lock": C2.file_record(C1.LOCK_PATH), "manifests": {}}
    rows = {}
    for split in ("ood22", "ood23") + C2.EVAL_SPLITS:
        p = C1.manifest_path(split)
        got = C1.sha256_file(p)
        if got != lock1["manifests"].get(split):
            raise SplitError("v1 manifest %s does not match the v1 LOCK" % split)
        inputs["manifests"][split] = got
        rows[split] = C1.read_manifest(p)
    sel_sum, sel_rec = _opt_json(select_summary_path())
    bs = base_selected_path()
    want = ((sel_sum or {}).get("outputs") or {}).get("base_selected.jsonl", {}).get("sha256")
    if not bs.is_file() or not want or C1.sha256_file(bs) != want:
        raise SplitError("%s is missing or does not hash as select_summary.json records" % bs)
    inputs.update(base_selected=C2.file_record(bs), select_summary=sel_rec)
    sets = {t: sorted(({"key": tsw_key(t, r["key"]), "image": r["image"], "sha256": r["sha256"]}
                       for r in rows[v]), key=lambda r: r["key"]) for t, v in TSW.items()}
    sets[BASE_B] = sorted(C1.read_manifest(bs), key=lambda r: r["key"])
    return sets, {s: rows[s] for s in C2.EVAL_SPLITS}, inputs


def _scan(embedder, calibration_path, procs, force, domain_config, testing):
    t0 = time.time()
    if C2.LOCK_PATH.exists():
        raise SplitError("%s exists: splits v2 are locked" % C2.LOCK_PATH)
    sets, eval_rows, inputs = scan_candidates()
    dom_path = Path(domain_config or default_domain_config())
    domain_raw = _read_json(dom_path, "the funnel domain config")
    inputs["funnel_domain"] = C2.file_record(dom_path)
    if embedder is None:
        embedder = _make_embedder(domain_raw)
    ldir = leak_dir()
    ldir.mkdir(parents=True, exist_ok=True)
    cal, source, gates = None, None, None
    if calibration_path:
        cal, source = G.load_calibration(calibration_path, embedder.name), "given"
    elif funnel_leak_path().is_file():
        try:
            cal, source = G.load_calibration(funnel_leak_path(), embedder.name), "funnel_leak_v1"
        except G.GuardError as e:
            log("the funnel's leak_v1.json is not reused: %s" % e)
    if cal is None:
        ref = (domain_raw.get("sources") or {}).get("reference")
        pairs = ((domain_raw.get("leak") or {}).get("negative_source_pairs")) or []
        need = {s for p in pairs for s in p} - {ref}
        if ref != "train_core":
            raise SplitError("the funnel config's reference split is %r; the v2 calibration uses train_core" % ref)
        if not pool_path().is_file():
            raise SplitError("%s missing: the calibration's negative group (%s) is read from it"
                             % (pool_path(), sorted(need)))
        lock1 = C1.read_lock()
        core_path = C1.manifest_path("train_core")
        if C1.sha256_file(core_path) != lock1["manifests"]["train_core"]:
            raise SplitError("v1 train_core does not match the v1 LOCK")
        ref_rows = C1.read_manifest(core_path)
        pool_rows = [r for r in C1.read_manifest(pool_path()) if r.get("source") in need]
        recall_min, fpr_max, gates = _h6_gates()
        log("calibrating the copy detector: %d reference images, %d negative-group images, recall >= %.2f, "
            "false positives <= %.3f" % (len(ref_rows), len(pool_rows), recall_min, fpr_max))
        try:
            cal = G.calibrate(ldir, domain_raw, embedder, ref_rows, pool_rows, recall_min, fpr_max,
                              seed_prefix=G.SEED_PREFIX, procs=procs, testing=testing, force=force,
                              inputs={"funnel_domain": inputs["funnel_domain"], "pool": C2.file_record(pool_path()),
                                      "train_core": C2.file_record(core_path), "gates": gates})
        except G.GuardError as e:
            raise SplitError(str(e))
        source = "inc2"
    try:
        index = G.eval_index(eval_rows, embedder, ldir / "eval_desc_v2.npz", procs=procs, force=force)
        scanner = G.EmbedScanner(index, cal)
    except (G.GuardError, C2.Inc2Error, RuntimeError) as e:
        raise SplitError("the evaluation index cannot be built: %s" % e)
    rows, where = [], {}
    for s in (list(TSW) + [BASE_B]):
        for r in sets[s]:
            if r["key"] in where:
                raise SplitError("key %s is in both %s and %s" % (r["key"], where[r["key"]], s))
            where[r["key"]] = s
            rows.append(r)
    log("scanning %d images (%s) against %d evaluation images, cos >= %.4f (%s)"
        % (len(rows), {s: len(v) for s, v in sets.items()}, index.n, scanner.threshold, source))
    copies, unscannable = scanner.scan(rows, desc_path=ldir / "scan_desc_v2.npz", procs=procs)
    hits = []
    for key in sorted(copies):
        for e in copies[key]:
            hits.append(dict(e, set=where[key]))
    counts = {s: {"scanned": len(sets[s]), "copies": len({h["key"] for h in hits if h["set"] == s}),
                  "unscannable": sum(1 for k in unscannable if where[k] == s)} for s in sets}
    doc = {"format": SCAN_FORMAT, "splits_version": C2.SPLITS_VERSION, "built_utc": C2.utc(),
           "testing": bool(testing), "seconds": round(time.time() - t0, 1), "inputs": inputs,
           "detector": dict(scanner.record(), calibration_source=source, seed_prefix=G.SEED_PREFIX,
                            rule="copy iff cos >= cos_threshold or min over the 8 variants of dHash bits <= "
                                 "dhash_bits_max"),
           "scanned": {s: [[r["key"], r["sha256"]] for r in sets[s]] for s in sets},
           "hits": hits, "unscannable": [{"set": where[k], "key": k} for k in sorted(unscannable)],
           "counts": counts, "code": _code_record(), "status": "complete"}
    C2.write_json_atomic(embed_scan_path(), doc)
    log("scan done in %.0fs: %s" % (time.time() - t0, counts))
    # L-9(c): the threshold for images without capture provenance, from the descriptors the scan's
    # calibration recorded (computed here only where they are missing)
    t1 = time.time()
    ecal = _calibrate_v2(doc, embedder, dom_path, domain_raw, procs, testing, force=force, compute=True)
    log("v2 calibration in %.0fs: cos >= %.6f (the scan's %.6f), strict %.6f, known limits %s. Next: `build` (it "
        "applies the scan through it), then `lock`" % (time.time() - t1, ecal["cos_threshold"], scanner.threshold,
                                                       ecal["strict_threshold"],
                                                       [x["family"] for x in ecal["known_limits"]]))
    return doc


# --------------------------------------------------------------- lock/verify
def file_shas(splits, procs=1):
    """{path: sha256 or None} of every image and label the given v2
    manifests name, each distinct file hashed once (worker processes when
    procs > 1)."""
    paths = set()
    for split in splits:
        p = C2.v2_manifest_path(split)
        if p.is_file():
            for r in C1.read_manifest(p):
                paths.add(r["image"])
                paths.add(r["label"])
    return dict(_pmap(_sha_task, sorted(paths), procs, "sha256 v2 files"))


def manifest_problems(split, shas=None):
    """Full re-hash of one v2 manifest's images and labels against its rows
    (shas: a file_shas() result to reuse)."""
    path = C2.v2_manifest_path(split)
    if not path.exists():
        return ["%s: manifest %s missing" % (split, path)]
    rows = C1.read_manifest(path)
    probs = []
    for r in rows:
        if split != C2.BASE_MANIFEST and not r["key"].startswith(split + "__"):
            probs.append("%s: key %s lacks the split prefix" % (split, r["key"]))
        for field, want in (("image", r["sha256"]), ("label", r["label_sha256"])):
            got = shas.get(r[field]) if shas is not None and r[field] in shas else _sha_task(r[field])[1]
            if got is None:
                probs.append("%s: %s %s is missing" % (split, field, r[field]))
            elif got != want:
                probs.append("%s: %s %s changed (%s != %s)" % (split, field, r[field], got[:12], want[:12]))
    return probs


def _base_parts_problems(summary):
    """base_v2 against its parts and the provenance file."""
    probs = []
    base = C1.read_manifest(C2.v2_manifest_path(C2.BASE_MANIFEST))
    prov = C1.read_manifest(C2.BASE_PROVENANCE) if C2.BASE_PROVENANCE.is_file() else []
    by_key = {p["key"]: p for p in prov}
    if set(by_key) != {r["key"] for r in base} or len(prov) != len(base):
        probs.append("base_v2_provenance.jsonl does not list exactly the base_v2 rows")
        return probs, base, prov
    for s in C2.TRAIN_SPLITS:
        want = {r["key"] for r in C1.read_manifest(C2.v2_manifest_path(s))}
        got = {k for k, p in by_key.items() if p["part"] == s}
        if want != got:
            probs.append("base_v2's %s part differs from %s.jsonl (%d vs %d rows)" % (s, s, len(got), len(want)))
    for r in base:
        p = by_key[r["key"]]
        if (p["image"], p["sha256"], p["label_sha256"], p["source"]) != (r["image"], r["sha256"], r["label_sha256"],
                                                                         r["source"]):
            probs.append("provenance of %s differs from base_v2" % r["key"])
            break
    rows = [dict(p) for p in prov]
    probs += disjoint_problems(rows)
    return probs, base, prov


def _never_train_problems(prov, eval_entries):
    """base v2 rows (their recorded dHash and 8 variants) within 6 bits of an
    evaluation image: the build refused them already; lock checks again from
    the provenance file, whose rows lock has matched to base_v2.jsonl."""
    guard = G.GuardV2(eval_entries)
    bad = []
    for p in prov:
        reason, match = guard.check(p.get("dhash"), p.get("variants"))
        if reason:
            bad.append((p["key"], reason, match))
    if bad:
        return ["%d base v2 row(s) are refused by the never-train guard (e.g. %s)" % (len(bad), bad[:3])]
    return []


def _variant_drops_problems(summary, lock1, eval_entries, v2_core_sha):
    """(problems, rows, sha256) of the L-8 list (train_core_variant_drops.jsonl)
    at lock: it hashes as the build recorded; it holds at most the cap; each
    row is a v1 train_core row (same key, image, label and bytes) whose image
    still hashes as recorded, whose re-computed dHash and 8 variants are the
    recorded ones, and which the never-train guard refuses as
    near_eval_variant; and v2's train_core is exactly v1's minus the listed
    rows. A list that fails any of these was not written by this build's
    rule (a tampered list), so lock refuses."""
    path = C2.TRAIN_CORE_VARIANT_DROPS
    rec = summary.get("train_core_variant_drops") or {}
    try:
        sha = C1.sha256_file(path)
    except OSError:
        return ["%s missing: build again" % path], [], None
    probs = []
    if sha != rec.get("sha256"):
        probs.append("%s changed since the build (%s != %s)" % (path.name, sha[:12], str(rec.get("sha256"))[:12]))
    try:
        rows = C1.read_manifest(path)
    except (OSError, ValueError) as e:
        return probs + ["%s unreadable: %s" % (path.name, e)], [], sha
    v1_path = C1.manifest_path("train_core")
    with open(v1_path, "rb") as fh:
        v1_bytes = fh.read()
    if hashlib.sha256(v1_bytes).hexdigest() != lock1["manifests"]["train_core"]:
        return probs + ["v1 train_core does not match the v1 LOCK"], rows, sha
    v1_by_key = {r["key"]: r for r in C1.read_manifest(v1_path)}
    cap = variant_drop_cap(len(v1_by_key))
    if len(rows) > cap:
        probs.append("%s lists %d rows, more than the L-8 cap %d" % (path.name, len(rows), cap))
    if sorted(r.get("key") for r in rows) != sorted(rec.get("keys") or []) or len(rows) != rec.get("count"):
        probs.append("%s does not list the rows summary.json records" % path.name)
    guard = G.GuardV2(eval_entries)
    for r in rows:
        v1 = v1_by_key.get(r.get("key"))
        if v1 is None or any(r.get(k) != v1[k] for k in C1.MANIFEST_KEYS):
            probs.append("%s: %s is not a v1 train_core row with the same bytes" % (path.name, r.get("key")))
            continue
        if _sha_task(r["image"])[1] != r["sha256"]:
            probs.append("%s: the image of %s no longer hashes as recorded" % (path.name, r["key"]))
            continue
        h, v = G.image_hashes(r["image"])
        vl = G.variant_list(v)
        if h is None or vl is None or r.get("dhash") != int(h) or [x for _n, x in vl] != r.get("variants"):
            probs.append("%s: the recorded dHash or variants of %s are not its image's" % (path.name, r["key"]))
            continue
        reason, _m = guard.check(r["dhash"], r["variants"])
        if reason != VARIANT_DROP_REASON or r.get("reason") != VARIANT_DROP_REASON:
            probs.append("%s: %s is not refused as %s (the guard says %s); L-8 drops only those"
                         % (path.name, r["key"], VARIANT_DROP_REASON, reason))
    reduced, dropped = C2.filter_manifest_bytes(v1_bytes, {r.get("sha256") for r in rows})
    if hashlib.sha256(reduced).hexdigest() != v2_core_sha or len(dropped) != len(rows):
        probs.append("v2 train_core is not v1's train_core minus the rows %s lists" % path.name)
    return probs, rows, sha


def _index_problems(path, want_entries, what):
    try:
        data = _read_json(path, what)
    except SplitError as e:
        return [str(e)], None
    probs = []
    if data.get("bits") != BITS:
        probs.append("%s: bits %s != %d" % (what, data.get("bits"), BITS))
    have = sorted((int(h), s, k) for h, s, k in data.get("entries") or ())
    if have != sorted(want_entries):
        probs.append("%s: entries differ from the manifests (%d held, %d expected)"
                     % (what, len(have), len(want_entries)))
    return probs, data


def _embed_calibration_problems(scan, summary):
    """(the v2 calibration record or None, problems) at lock: it exists and
    shows it passed, it was made from the calibration and evaluation
    descriptors this scan used, its negatives file hashes as it records, and
    it is the one the build applied (summary.json embed_v2)."""
    path = embed_calibration_path()
    if not path.is_file():
        return None, ["%s missing: run `build` (or `scan`) again; the v2 embedding calibration (L-9(c)) judges "
                      "base B and every tsw row without a capture session" % EC.NAME]
    try:
        ecal = EC.load(path)
        doc = _read_json(path, EC.NAME)
    except (C2.Inc2Error, OSError, ValueError) as e:
        return None, ["the v2 embedding calibration cannot be used: %s" % e]
    probs = []
    det = scan.get("detector") or {}
    if ((ecal.get("base") or {}).get("file") or {}).get("sha256") != (det.get("calibration") or {}).get("sha256"):
        probs.append("%s was made from another calibration than the one embed_scan.json used: run `build` again"
                     % EC.NAME)
    if (doc.get("identity") or {}).get("eval_descriptors_sha256") != (det.get("eval_descriptors") or {}).get("sha256"):
        probs.append("%s was made from other evaluation descriptors than embed_scan.json's" % EC.NAME)
    nc = doc.get("negatives_csv") or {}
    npath = C2.SPLITS_DIR / EC.NEGATIVES_NAME
    if not npath.is_file() or C1.sha256_file(npath) != nc.get("sha256"):
        probs.append("%s is missing or does not hash as %s records" % (EC.NEGATIVES_NAME, EC.NAME))
    if ((summary.get("embed_v2") or {}).get("file") or {}).get("sha256") != ecal["file"]["sha256"]:
        probs.append("the build applied %s: run `build` again (it applies the scan through %s)"
                     % ("no v2 calibration" if not (summary.get("embed_v2") or {}).get("applied")
                        else "another v2 calibration", EC.NAME))
    if ecal.get("testing") and not summary.get("testing"):
        probs.append("%s is a testing calibration and the build is not" % EC.NAME)
    return ecal, probs


def _scan_problems(base_prov, summary):
    """Coverage of the embedding scan over base v2's tsw and harvested rows,
    and each row judged again by its copy rule (L-9): a tsw row of a dev or
    test capture session; an image the scan could not describe; a copy at
    the v2 threshold (within 6 dHash bits, or cos at or above it) of a row
    the threshold applies to (base B, a tsw row without a capture session).
    The provenance file's copy_rule must be the rule re-derived here.
    Returns (problems, scan, v2 calibration record or None)."""
    try:
        scan = _load_scan()
    except SplitError as e:
        return [str(e)], None, None
    if scan is None:
        return ["embed_scan.json missing: run `scan` (the embedding copy scan of 4.2 step 5) before lock"], None, None
    probs = []
    for s in C2.EVAL_SPLITS:
        if (scan.get("inputs") or {}).get("manifests", {}).get(s) != summary["manifests"][s]["sha256"]:
            probs.append("embed_scan.json was made against another %s manifest" % s)
    ecal, eprobs = _embed_calibration_problems(scan, summary)
    probs += eprobs
    t2 = ecal["cos_threshold"] if ecal is not None else None
    scanned, hits, uns = _scan_index(scan)
    dev_s = set(summary.get("dev_sessions") or ())
    test_s = {r["session"] for r in C1.read_manifest(C2.v2_manifest_path("test"))} - {""}
    uncovered, bad, wrong_rule = [], [], []
    for p in base_prov:
        if p["part"] == "train_core":
            if p.get("copy_rule") != RULE_TRAIN_CORE:
                wrong_rule.append(p["key"])
            continue
        s = p["part"]
        k = (s, p["key"], p["sha256"])
        rule = tsw_rule(p.get("session") or "", dev_s, test_s) if s in TSW else RULE_BASE_B
        if p.get("copy_rule") != rule:
            wrong_rule.append(p["key"])
        if k not in scanned:
            uncovered.append(p["key"])
            continue
        ev = embed_evidence(hits, uns, k, t2)
        if rule == RULE_TSW_EVAL_SESSION:
            bad.append((p["key"], "a dev or test capture session (L-9(a))"))
        elif ev["unscannable"]:
            bad.append((p["key"], "unhashable_embed"))
        elif rule != RULE_TSW_PROVENANCE and ev["applied_hit"]:
            bad.append((p["key"], "near_eval_embed"))
    if uncovered:
        probs.append("%d base v2 row(s) were not scanned (e.g. %s): run `scan` again" % (len(uncovered), uncovered[:3]))
    if wrong_rule:
        probs.append("%d provenance row(s) do not carry the copy rule L-9 gives them (e.g. %s)"
                     % (len(wrong_rule), wrong_rule[:3]))
    if bad:
        probs.append("%d base v2 row(s) are flagged by the embedding scan under their copy rule (e.g. %s): run "
                     "`build` again (it drops such tsw rows; such a row of base B's part is an R4 incident and "
                     "refuses)" % (len(bad), bad[:3]))
    return probs, scan, ecal


def funnel_h6_status():
    """The funnel's H6 result for base B's part: "pending", or leak_v1.json's."""
    p = funnel_leak_path()
    if not p.is_file():
        return "pending"
    try:
        doc = _read_json(p, "leak_v1.json")
    except SplitError as e:
        return {"path": str(p), "unreadable": str(e)}
    return {"path": str(p), "sha256": C1.sha256_file(p), "status": doc.get("status"),
            "calibration_ok": bool((doc.get("calibration") or {}).get("ok")),
            "base_copy": (doc.get("h6b") or {}).get("base_copy"), "incident": (doc.get("h6b") or {}).get("incident")}


def lock(scorer_path=None, procs=PROCS, testing=False):
    """Write LOCK v2 after the checks of the module docstring. A build made
    with --testing (real-data pins lifted) is locked only with testing=True,
    which run_inc2_splits.sh never passes: the platform's argument-less lock
    cannot seal a synthetic world into the real INC_DIR."""
    with writer():
        return _lock(scorer_path, procs, testing)


def _lock(scorer_path, procs, testing=False):
    if C2.LOCK_PATH.exists():
        raise SplitError("%s exists; a change to a locked splits version is a new version (R4)" % C2.LOCK_PATH)
    summary = read_summary()
    if summary.get("testing") and not testing:
        raise SplitError("summary.json is a --testing build (the real-data pins were lifted); it is locked only "
                         "with lock --testing, never by the platform. Build again without --testing.")
    scorer = Path(scorer_path or S1.default_scorer_path())
    log("lock: v1 verify, then a full re-hash of v2")
    probs1 = S1.verify(scorer_path=scorer)
    if probs1:
        raise SplitError("splits v1 do not verify (%s)" % probs1[:3])
    lock1 = C1.read_lock()
    if C1.sha256_file(C1.LOCK_PATH) != summary["inputs"]["v1_lock"]["sha256"]:
        raise SplitError("the v1 LOCK changed since the build")
    if C1.sha256_file(scorer) != lock1["scorer_sha256"]:
        raise SplitError("scorer %s is not the v1 LOCK's" % scorer)
    probs = []
    shas = {}
    files = file_shas(C2.V2_MANIFESTS, procs)
    for s in C2.V2_MANIFESTS:
        p = C2.v2_manifest_path(s)
        if not p.is_file():
            probs.append("manifest %s missing" % p)
            continue
        shas[s] = C1.sha256_file(p)
        if shas[s] != summary["manifests"][s]["sha256"]:
            probs.append("manifest %s changed since the build" % s)
        probs += manifest_problems(s, files)
    for s in C2.BYTE_COPIES:
        if shas.get(s) != lock1["manifests"][s]:
            probs.append("%s is no longer a byte copy of v1" % s)
    if probs:
        raise SplitError("cannot lock, %d problem(s): %s" % (len(probs), probs[:10]))
    bprobs, base, prov = _base_parts_problems(summary)
    eval_entries = []
    idx1 = _read_json(C1.NEVER_TRAIN_INDEX, "the v1 never-train index")
    want_eval = {(s, r["key"]) for s in C2.EVAL_SPLITS for r in C1.read_manifest(C2.v2_manifest_path(s))}
    for h, s, k in idx1["entries"]:
        if (s, k) in want_eval:
            eval_entries.append((int(h), s, k))
    ip, nt = _index_problems(C2.NEVER_TRAIN_INDEX, eval_entries, "the v2 never-train index")
    bp, bc = _index_problems(C2.BASE_COPIES_INDEX, [(int(p["dhash"]), p["part"], p["key"]) for p in prov],
                             "the base-copy index")
    sp, scan, ecal = _scan_problems(prov, summary)
    dp, drops, drops_sha = _variant_drops_problems(summary, lock1, eval_entries, shas.get("train_core"))
    probs = bprobs + ip + bp + sp + dp
    if not bprobs:
        probs += _never_train_problems(prov, eval_entries)
    funnel_matches, funnel_rec = funnel_base_b_matches()
    t2 = ecal["cos_threshold"] if ecal is not None else None
    hit = sorted(p["key"] for p in prov if p["part"] == BASE_B
                 and funnel_verdict((funnel_matches.get(p["key"]) or []) + (funnel_matches.get(str(p["image"])) or []),
                                    t2)["applied"])
    if hit:
        probs.append("R4 INCIDENT (funnel H6(b)): the funnel's leak_v1.json lists %d image(s) of base v2's base B "
                     "part as copies of a v2 evaluation image at the v2 threshold (L-9(b)) (e.g. %s); the owner "
                     "decides" % (len(hit), hit[:3]))
    try:
        rules_sha = C1.sha256_file(copy_rules_path())
    except OSError:
        rules_sha = None
    if rules_sha is None or rules_sha != ((summary.get("l9") or {}).get("copy_rules") or {}).get("sha256"):
        probs.append("%s is missing or changed since the build" % COPY_RULES_NAME)
    if probs:
        raise SplitError("cannot lock, %d problem(s): %s" % (len(probs), probs[:10]))
    nt.pop("note", None)
    nt.update(complete=True, min_expected=len(nt["entries"]), locked_utc=C2.utc())
    bc.pop("note", None)
    bc.update(complete=True, min_expected=len(bc["entries"]), locked_utc=C2.utc())
    nt_sha = C2.write_json_atomic(C2.NEVER_TRAIN_INDEX, nt)
    bc_sha = C2.write_json_atomic(C2.BASE_COPIES_INDEX, bc)
    scan_counts = {}
    for s in (list(TSW) + [BASE_B]):
        scan_counts[s] = scan["counts"].get(s)
    base_b = summary["base_b"]
    new = {
        "splits_version": C2.SPLITS_VERSION,
        "manifests": {s: shas[s] for s in C2.V2_MANIFESTS},
        "eval_splits": list(C2.EVAL_SPLITS), "train_splits": list(C2.TRAIN_SPLITS),
        "base_manifest": C2.BASE_MANIFEST, "final_exams": list(C2.FINAL_EXAMS),
        "nevertrain_sha256": nt_sha, "nevertrain_entries": len(nt["entries"]),
        "base_copies_sha256": bc_sha, "base_copies_entries": len(bc["entries"]),
        "provenance_sha256": C1.sha256_file(C2.BASE_PROVENANCE),
        "provenance": {"file": C2.BASE_PROVENANCE.name, "sha256": C1.sha256_file(C2.BASE_PROVENANCE),
                       "rows": len(prov)},
        "l5_excluded_sha256": C1.sha256_file(C2.L5_EXCLUDED),
        C2.VARIANT_DROPS_LOCK_KEY: drops_sha,
        "train_core_variant_drops": {"file": C2.VARIANT_DROPS_NAME, "sha256": drops_sha, "rows": len(drops),
                                     "keys": sorted(r["key"] for r in drops), "decided_by": L8_DECISION,
                                     "cap": variant_drop_cap(len(C1.read_manifest(C1.manifest_path("train_core")))),
                                     "incident": bool(drops)},
        "summary_sha256": C1.sha256_file(summary_path()),
        "embed_scan_sha256": C1.sha256_file(embed_scan_path()),
        EC.LOCK_KEY: ecal["file"]["sha256"],
        "embed_calibration_v2": {"file": EC.NAME, "sha256": ecal["file"]["sha256"], "decided_by": EC.DECISION,
                                 "cos_threshold": ecal["cos_threshold"], "strict_threshold": ecal["strict_threshold"],
                                 "p_false": ecal["p_false"], "testing": ecal["testing"],
                                 "base": {k: (ecal["base"] or {}).get(k) for k in ("source", "role", "cos_threshold",
                                                                                   "file")},
                                 "known_limits": [x["family"] for x in ecal["known_limits"]],
                                 "negatives_csv": EC.NEGATIVES_NAME,
                                 "negatives_csv_sha256": C1.sha256_file(C2.SPLITS_DIR / EC.NEGATIVES_NAME)},
        "embed_calibration_v2_negatives_sha256": C1.sha256_file(C2.SPLITS_DIR / EC.NEGATIVES_NAME),
        "copy_rules_sha256": rules_sha,
        "l9": {"decided_by": L9_DECISION, "copy_rules": COPY_RULES_NAME,
               "rules": {part: (summary.get("l9") or {}).get(part, {}).get("rules") for part in list(TSW) + [BASE_B]},
               "funnel_base_b_entries": ((summary.get("l9") or {}).get(BASE_B) or {}).get("funnel_entries")},
        "scorer_sha256": C1.sha256_file(scorer),
        "derived_from": {"v1_lock_sha256": C1.sha256_file(C1.LOCK_PATH), "v1_lock_path": str(C1.LOCK_PATH),
                         "identical": list(C2.BYTE_COPIES) + ([] if drops else list(C2.FILTERED_COPIES)),
                         "train_core": {"v1_sha256": lock1["manifests"]["train_core"], "sha256": shas["train_core"],
                                        "minus": C2.VARIANT_DROPS_NAME, "minus_sha256": drops_sha,
                                        "dropped": len(drops),
                                        "rule": "v1 train_core.jsonl without the rows whose image sha256 the list "
                                                "names; every other line byte-identical, in order (L-8)"}},
        "h6": {"base_b": {"dhash_8_variant_check": "passed for the kept part",
                          "embed_scan": {"calibration_source": scan["detector"].get("calibration_source"),
                                         "cos_threshold": scan["detector"].get("cos_threshold"),
                                         "cos_threshold_v2": ecal["cos_threshold"],
                                         "embedder": scan["detector"].get("embedder"),
                                         "copies_in_base_v2": 0, "counts": scan_counts},
                          "l5_excluded": base_b["l5"]["excluded"],
                          "incident_in_l5_part": base_b["h6b"]["incident_l5_part"],
                          "funnel_leak_v1": funnel_h6_status(),
                          "funnel_base_b_copies_in_base_v2": 0, "funnel_state": funnel_rec["state"]}},
        "h6_status": {"base_v2_copies": 0, "embed_scan": scan["detector"].get("calibration_source"),
                      "embed_v2_threshold": ecal["cos_threshold"],
                      "funnel_leak_v1": funnel_rec["state"],
                      "incident_in_l5_part": base_b["h6b"]["incident_l5_part"]},
        "research_only_rows": summary["licences"]["research_only_rows"],
        "testing": bool(summary.get("testing")),
        "created_utc": C2.utc(),
    }
    C2.write_json_atomic(C2.LOCK_PATH, new)
    with open(C2.SPLITS_DIR / LOCK_LOG_NAME, "a") as fh:
        fh.write(json.dumps({"utc": new["created_utc"], "lock": new, "lock_sha256": C1.sha256_file(C2.LOCK_PATH)},
                            sort_keys=True) + "\n")
    n = _make_read_only(C2.SPLITS_DIR)
    log("locked splits v2: base_v2 %d images, %d never-train entries, %d base-copy entries, %d L-8 train_core "
        "drop(s); %d files read-only" % (len(base), len(nt["entries"]), len(bc["entries"]), len(drops), n))
    return new


def _walk_files(root):
    out, stack = [], [str(root)]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                else:
                    out.append(e.path)
    return sorted(out)


def _make_read_only(root):
    files = _walk_files(root)
    for f in files:
        os.chmod(f, 0o444)
    return len(files)


def verify(scorer_path=None, check_v1=True, procs=PROCS):
    """Every problem a full re-hash of v2 (and, with check_v1, of v1) finds
    against LOCK v2."""
    probs = []
    try:
        lk = C2.read_lock_v2()
    except C2.Inc2Error as e:
        return [str(e)]
    files = file_shas(C2.V2_MANIFESTS, procs)
    for s in C2.V2_MANIFESTS:
        try:
            C2.verify_manifest_against_lock_v2(s, lk)
        except (C2.Inc2Error, OSError) as e:
            probs.append(str(e))
            continue
        probs += manifest_problems(s, files)
    for name, key in ((C2.NEVER_TRAIN_INDEX, "nevertrain_sha256"), (C2.BASE_COPIES_INDEX, "base_copies_sha256"),
                      (C2.BASE_PROVENANCE, "provenance_sha256"), (C2.L5_EXCLUDED, "l5_excluded_sha256"),
                      (C2.TRAIN_CORE_VARIANT_DROPS, C2.VARIANT_DROPS_LOCK_KEY),
                      (summary_path(), "summary_sha256"), (embed_scan_path(), "embed_scan_sha256"),
                      (embed_calibration_path(), EC.LOCK_KEY),
                      (C2.SPLITS_DIR / EC.NEGATIVES_NAME, "embed_calibration_v2_negatives_sha256"),
                      (copy_rules_path(), "copy_rules_sha256")):
        try:
            if C1.sha256_file(name) != lk.get(key):
                probs.append("%s changed since it was locked" % name)
        except OSError:
            probs.append("%s missing" % name)
    try:                                  # train_core v2 = v1's minus the L-8 rows (derived_from.train_core)
        drops, _rec = C2.read_variant_drops(lk, C2.LOCK_PATH, production=True)
        with open(C1.manifest_path("train_core"), "rb") as fh:
            reduced, dropped = C2.filter_manifest_bytes(fh.read(), {r["sha256"] for r in drops})
        der = (lk.get("derived_from") or {}).get("train_core") or {}
        if (hashlib.sha256(reduced).hexdigest() != lk["manifests"].get("train_core") or len(dropped) != len(drops)
                or der.get("sha256") != lk["manifests"].get("train_core")):
            probs.append("train_core v2 is not v1's train_core minus the rows %s lists" % C2.VARIANT_DROPS_NAME)
    except (C2.Inc2Error, OSError, ValueError, KeyError) as e:
        probs.append("train_core's L-8 derivation cannot be checked: %s" % e)
    try:
        G.GuardV2.load(C2.LOCK_PATH)
    except C2.Inc2Error as e:
        probs.append("GuardV2 refuses the LOCK: %s" % e)
    try:
        EC.load(embed_calibration_path(), lock=lk)
    except (C2.Inc2Error, OSError) as e:
        probs.append("the v2 embedding calibration does not load against LOCK v2: %s" % e)
    scorer = Path(scorer_path or S1.default_scorer_path())
    try:
        if C1.sha256_file(scorer) != lk["scorer_sha256"]:
            probs.append("scorer %s changed since it was locked" % scorer)
    except OSError:
        probs.append("scorer %s missing" % scorer)
    try:
        lock1 = C1.read_lock()
        for s in lk["derived_from"]["identical"]:
            if lock1["manifests"].get(s) != lk["manifests"].get(s):
                probs.append("%s is no longer identical to the v1 LOCK's" % s)
        if C1.sha256_file(C1.LOCK_PATH) != lk["derived_from"]["v1_lock_sha256"]:
            probs.append("the v1 LOCK changed since v2 was locked")
    except (OSError, KeyError, ValueError) as e:
        probs.append("the v1 LOCK cannot be compared: %s" % e)
    for f in _walk_files(C2.SPLITS_DIR):
        if os.stat(f).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            probs.append("%s is writable (every file is 0444 after lock)" % f)
    if check_v1:
        probs += ["v1: %s" % p for p in S1.verify(scorer_path=scorer)]
    return probs


# ------------------------------------------------------------------- CLI
def print_summary():
    s = read_summary()
    log("splits v2 built %s (%ss)%s" % (s["built_utc"], s["build_seconds"], "; TESTING" if s.get("testing") else ""))
    for split, v in s["manifests"].items():
        log("  %-11s %6d rows  %s" % (split, v["rows"], v["sha256"][:12]))
    log("base_v2: %s" % s["base_v2"]["parts"])
    for t in TSW:
        log("%s: kept %d of %d, dropped %s, session overlap dev %d test %d train_core %d"
            % (t, s["tsw"][t]["kept"], s["tsw"][t]["v1_rows"], s["tsw"][t]["dropped"],
               s["tsw"][t]["session_overlap"]["dev"], s["tsw"][t]["session_overlap"]["test"],
               s["tsw"][t]["session_overlap"]["train_core"]))
    log("base B: kept %d of %d; L-5 excluded %d %s; H6(b) incident in the L-5 part: %s"
        % (s["base_b"]["kept"], s["base_b"]["rows"], s["base_b"]["l5"]["excluded"], s["base_b"]["l5"]["per_source"],
           s["base_b"]["h6b"]["incident_l5_part"]))
    vd = s.get("train_core_variant_drops") or {}
    log("train_core: %s of %s v1 rows kept; L-8 drops %s (cap %s) %s"
        % (s["train_core"]["rows"], s["train_core"]["v1_rows"], vd.get("count"), vd.get("cap"), vd.get("matches")))
    log("licences: unresolved %s; research_only rows %d" % (s["licences"]["unresolved_sources"],
                                                           s["licences"]["research_only_rows"]))
    log("embed scan applied: %s" % s["embed_scan"]["applied"])
    e2 = s.get("embed_v2") or {}
    l9 = s.get("l9") or {}
    log("L-9: v2 threshold %s (scan's %s), strict %s, known limits %s; copy rules %s"
        % (e2.get("cos_threshold"), l9.get("scan_threshold"), e2.get("strict_threshold"),
           [x.get("family") for x in e2.get("known_limits") or []],
           {p: (l9.get(p) or {}).get("rules") for p in list(TSW) + [BASE_B]}))
    for p in list(TSW) + [BASE_B]:
        v = l9.get(p) or {}
        log("  %s: flagged at the scan's threshold %s (exempt %s, kept %s; cos %s); copies at v2 where it applies %s; "
            "funnel %s" % (p, v.get("flagged_at_scan_threshold"), v.get("flagged_exempt"), v.get("flagged_exempt_kept"),
                           v.get("flagged_exempt_cos"), v.get("copies_at_v2_threshold_where_it_applies"),
                           v.get("funnel_images")))
    log("LOCK v2: %s" % ("present" if C2.LOCK_PATH.exists() else "absent"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc2.splits",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the v2 manifests and indexes (not LOCK.json)")
    b.add_argument("--testing", action="store_true",
                   help="a synthetic world: lift the real-data pins (recorded; the job script refuses it)")
    b.add_argument("--licences", default=None, help="an owner licence table (JSON {sources: {name: {licence, "
                                                    "evidence}}})")
    b.add_argument("--tsw-record", default=None, help="the saved Zenodo record of 3SeasonWeedDet10 (default %s)"
                                                      % default_tsw_record())
    b.add_argument("--domain-config", default=None, help="the funnel domain config (read only; default %s)"
                                                         % default_domain_config())
    b.add_argument("--scorer", default=None, help="the scorer the v1 LOCK records (default inc/scorer.py)")
    b.add_argument("--procs", type=int, default=PROCS)
    b.add_argument("--skip-scan", action="store_true",
                   help="do not run the embedding copy scan (a GPU job) when none covers the candidates; lock "
                        "refuses until one does")
    b.add_argument("--calibration", default=None,
                   help="a passed calibration for the scan (inc2 or the funnel's leak_v1.json)")
    sc = sub.add_parser("scan", help="the embedding copy scan (a GPU job)")
    sc.add_argument("--calibration", default=None, help="a passed calibration (inc2 or the funnel's leak_v1.json)")
    sc.add_argument("--domain-config", default=None)
    sc.add_argument("--procs", type=int, default=PROCS)
    sc.add_argument("--force", action="store_true", help="recompute cached descriptors and the calibration")
    sc.add_argument("--testing", action="store_true", help="a synthetic world (recorded; the job script refuses it)")
    lk = sub.add_parser("lock", help="write LOCK v2 and make splits/v2 read-only")
    lk.add_argument("--scorer", default=None)
    lk.add_argument("--procs", type=int, default=PROCS)
    lk.add_argument("--testing", action="store_true",
                    help="lock a --testing build (a synthetic world; the job script refuses it)")
    vf = sub.add_parser("verify", help="re-hash everything against LOCK v2 (and v1)")
    vf.add_argument("--scorer", default=None)
    vf.add_argument("--procs", type=int, default=PROCS)
    vf.add_argument("--skip-v1", action="store_true", help="do not re-hash splits v1")
    sub.add_parser("summary", help="print summary.json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            build(testing=args.testing, scorer_path=args.scorer, licences_path=args.licences,
                  tsw_record_path=args.tsw_record, domain_config=args.domain_config, procs=args.procs,
                  scan_mode="skip" if args.skip_scan else "auto", calibration_path=args.calibration)
        elif args.cmd == "scan":
            scan(calibration_path=args.calibration, procs=args.procs, force=args.force,
                 domain_config=args.domain_config, testing=args.testing)
        elif args.cmd == "lock":
            lock(scorer_path=args.scorer, procs=args.procs, testing=args.testing)
        elif args.cmd == "verify":
            probs = verify(scorer_path=args.scorer, check_v1=not args.skip_v1, procs=args.procs)
            for p in probs[:200]:
                log("MISMATCH %s" % p)
            if probs:
                log("verify FAILED: %d problem(s)" % len(probs))
                return 1
            log("verify OK: every v2 manifest, image, label, index, the scorer and the v1 derivation match LOCK v2")
        elif args.cmd == "summary":
            print_summary()
    except C2.Inc2Error as e:
        log("ERROR: %s" % e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
