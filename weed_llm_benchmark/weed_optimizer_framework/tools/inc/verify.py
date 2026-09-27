"""INC Step 1: the box verifier (labeler Phase B). Contract: docs/INCREMENTAL_PROTOCOL.md, Step 1.

    python -m weed_optimizer_framework.tools.inc.verify pool        # species-joined harvested pool
    python -m weed_optimizer_framework.tools.inc.verify crops       # one row per box
    python -m weed_optimizer_framework.tools.inc.verify embed [--shard i --nshards n]
    python -m weed_optimizer_framework.tools.inc.verify fit         # probe + per-class thresholds
    python -m weed_optimizer_framework.tools.inc.verify calibrate   # precision on known truth
    python -m weed_optimizer_framework.tools.inc.verify admit       # verified.jsonl + review queue
    python -m weed_optimizer_framework.tools.inc.verify all

Outputs go under INC_DIR/step1/.

The harvested registry holds boxes whose species come from string joins of each
source's class list (cwd12_species). A join can be wrong in ways a name cannot
show: a class list in the wrong order, a site's "Ragweed" that is giant ragweed,
a copy of cwd12 exported with the legacy labels. This module reads every box
back from the pixels before any of it can enter a high-precision base.

pool       The registry's species-joined boxes (mega_trainer's own class map,
           trainer slot -> INC id through cwd12_species), minus NEVER_TRAIN
           slugs, user-flagged and autolabelled slugs, images within 6 dHash
           bits of an evaluation image (never-train index; an image that
           cannot be hashed is dropped), images within 6 bits of a train_core
           image (kept apart as cwd12_copies.jsonl: their true labels are
           known) and exact duplicates. The two leave-4-out cwd12 copies
           (cottonweed_sp8, cottonweed_holdout) are read for their train_core
           photographs only, as calibration copies; nothing of theirs enters
           the pool. Labels are rewritten in INC ids under step1/labels, one
           file per content (its sha256 is in the name, written atomically,
           never rewritten), so a manifest's label never changes under it;
           source labels are never touched. Each copy also records the labels
           the pre-v3.60.0 join gave its boxes (old_join).
crops      One row per box of train_core (label = cwd12 id, group = session),
           of the cwd12 copies and of the pool (label = INC id, group = slug).
           No crop image is written; a crop is re-cut from (image, box). Boxes
           under MIN_BOX_PX go to crops_skipped.csv. crops_info.json binds
           crops.csv to the sha256 of the pool, copies and train_core files it
           was made from; embed, fit, calibrate and admit refuse when any of
           them changed since.
embed      BioCLIP-2 features of every crop, sharded by image and resumable per
           chunk. A crop that cannot be cut or embedded is a NaN row: rows are
           addressed by crop_id, so a failure never shifts a neighbour's row.
fit        A logistic-regression probe on L2-normalised features: the 12 cwd12
           species from train_core crops, plus OtherPlant (12) from a seeded
           sample of pool boxes whose source named a plant that is neither a
           cwd12 species nor a relative or variant of one (genus, common-name
           token). Per-class thresholds by 5-fold GroupKFold over capture
           sessions, set jointly: tau_p[k] and sigma[k] sit at the same rank of
           the held-out true-k crops the probe ranks first, the highest rank at
           which a held-out true-k crop is verified 95 % of the time (argmax,
           probability and cosine together; recall per class).
calibrate  The verifier's precision where the truth is known: (a) the cwd12
           copies, box-matched to their train_core original, scored with
           verifiers that never saw that photograph's session, under the
           current join and under the pre-v3.60.0 join (old_wrong_joins);
           (b) seeded 20 % label swaps on held-out train_core folds.
admit      An image is admitted when every species box is verified and no box
           is a conflict. Conflict images go to a verify-only human queue
           (conflicts.csv plus a crop sheet) and are never auto-trained. The
           OtherPlant boxes the probe was trained on are judged by its
           out-of-fold predictions, never by the probe that saw them.

One verdict function (`verdicts`) serves fit, calibrate and admit.

Directories are listed once each (one os.listdir per known image or label
directory), never walked: several harvested slugs are very large and live on
Lustre. Hashing and cropping run in worker processes; every stage prints its
progress and writes its outputs atomically; hashes and embeddings are cached so
a killed job resumes where it stopped.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from ..cwd12_species import (CWD12_ID_TO_SLOT, CWD12_LEGACY_LABELS, CWD12_SPECIES,
                             TRAINER_SLOT_LEGACY, name_key, species_of)
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NearHashIndex

# ------------------------------------------------------------------ constants
EMBEDDER_NAME = "hf-hub:imageomics/bioclip-2"
N_SPECIES = len(CWD12_SPECIES)
OTHER = C.OTHER_PLANT
N_FOLDS = 5
RECALL = 0.95                 # per-class recall the thresholds are set at
MAX_COND_PASS = 0.99          # cap on the pass rate asked of the crops the probe ranks first
MIN_CAL = 5                   # held-out crops a class needs for its thresholds
# A harvested photograph this close to a train_core photograph is that
# photograph (cwd12_copies.jsonl). train_core photographs are natural field
# photos, where near_dup.py finds the holdout radius costs almost nothing; a
# Roboflow re-export of a cwd12 photograph can sit 4-5 bits from it.
CORE_COPY_BITS = HOLDOUT_NEAR_DUP_BITS
N_OTHER = 5000                # OtherPlant training sample
SWAP_FRAC = 0.20              # calibrate (b)
MATCH_TOL = 0.02              # calibrate (a): box coordinate match, as in mega_trainer
SHEET_MAX = 200
PROBE_C = 1.0
PROBE_MAX_ITER = 3000
PROCS = 5
BATCH = 128
CHUNK_IMAGES = 2000           # embed resumes per chunk of this many images
PROGRESS_EVERY = 2000
KEY_MAX = 180                 # bytes; key + ".txt" stays well under NAME_MAX

# mega_trainer._merge_datasets' valid_annotations (without include_autolabel)
VALID_ANNOTATIONS = ("bbox", "bbox+segmentation", "yolo")
SPLIT_DIRS = ("train", "valid", "val", "test")

# Box verdicts. SMALL: under MIN_BOX_PX, never embedded. FAILED: no features.
VERIFIED, CONFLICT, UNKNOWN, OTHER_OK, SMALL, FAILED = (
    "verified", "conflict", "unknown", "other_ok", "small", "failed")
VERDICT_CODES = (VERIFIED, CONFLICT, UNKNOWN, OTHER_OK, SMALL, FAILED)
ADMITTED = "admitted"         # image verdicts: ADMITTED, CONFLICT, UNKNOWN
NO_HELD_OUT = "no_held_out_model"   # calibrate (a): no verifier held that session out

# Trainer slot (mega_trainer, cwd12_species.CWD12_ID_TO_SLOT) -> INC id.
SLOT_TO_INC = {slot: i for i, slot in CWD12_ID_TO_SLOT.items()}
assert sorted(SLOT_TO_INC) == list(range(N_SPECIES))

# Source names that do not say which plant a box holds. A generic "weed" box
# in a cotton field may well be Palmer amaranth, so these never enter the
# OtherPlant training sample (they still enter the pool as OtherPlant boxes,
# where the verifier can call them a conflict).
GENERIC_NAME_KEYS = frozenset("""
    weed weeds weedplant weedplants weedspecies weedsp weedspp plant plants
    object objects vegetation veg unknown other others misc broadleaf broadleaves
    broadleafweed broadleafweeds grass grasses grassweed grassweeds narrowleaf
    narrowleafweed dicot dicots monocot monocots sedge sedges seedling seedlings
    leaf leaves green greenery background bg na none null unlabeled unlabelled
    label target thing item mixed sp spp species herb herbs invasive
    invasiveplant volunteer wildplant car bug mite worm rot rust mold stone rock
    soil tag pot tray marker pest pests
""".split())
NON_PLANT_WORDS = ("pest", "insect", "disease", "blight", "mildew", "lesion", "virus",
                   "bacteri", "fungus", "fungal", "deficien", "damage", "aphid", "beetle",
                   "caterpillar", "larva", "moth", "person", "human", "tractor", "robot",
                   "shadow", "novel", "unknown")
_GENERIC_RE = re.compile(r"(class|cls|label|obj|object|item|id|weed|weeds|plant|plants|"
                         r"crop|crops|species|sp|type)?\d+")
# species_of is strict on purpose (a loose match joins the wrong slot), so a
# miss there does not mean the plant is not a cwd12 species: "Amaranthus sp.",
# "pigweed", "palmer amaranth seedling", "morning glory sp", "spurge". A name
# holding any of these tokens (genus, common name, or the name of a species'
# close relatives) never enters the OtherPlant training sample; its boxes
# still enter the pool as OtherPlant, where the verifier can call them a
# conflict. Every cwd12 alias key holds one of them.
CWD12_RELATED_TOKENS = (
    "waterhemp", "amaranth", "pigweed", "palmer",                       # Amaranthus
    "morningglory", "ipomoea",                                          # MorningGlory
    "purslane", "portulaca",                                            # Purslane
    "spurge", "euphorbia",                                              # SpottedSpurge
    "carpetweed", "mollugo",                                            # Carpetweed
    "ragweed", "ambrosia",                                              # Ragweed
    "eclipta", "falsedaisy",                                            # Eclipta
    "sida",                                                             # PricklySida
    "sicklepod", "senna", "cassia",                                     # Sicklepod
    "goosegrass", "eleusine",                                           # Goosegrass
    "groundcherry", "physalis",                                         # CutleafGroundcherry
)
# Distinct species that hold a token above and that a probe must learn apart
# from the cwd12 one (a site's "Ragweed" can be giant ragweed).
OTHER_ALLOWED_KEYS = frozenset({"giantragweed", "ambrosiatrifida"})

# ---------------------------------------------------------------------- paths
STEP1 = C.INC_DIR / "step1"
LABELS_DIR = STEP1 / "labels"
COPY_LABELS_DIR = STEP1 / "labels_cwd12_copies"
CACHE_DIR = STEP1 / "cache"
POOL = STEP1 / "pool.jsonl"
POOL_META = STEP1 / "pool_meta.jsonl"
POOL_SUMMARY = STEP1 / "pool_summary.json"
COPIES = STEP1 / "cwd12_copies.jsonl"
CORE_PROBE = CACHE_DIR / "train_core_probe.json"
CROPS = STEP1 / "crops.csv"
CROPS_SKIPPED = STEP1 / "crops_skipped.csv"
CROPS_INFO = STEP1 / "crops_info.json"
EMB_DIR = STEP1 / "emb"
VERIFIER_DIR = STEP1 / "verifier"
CALIBRATION = STEP1 / "calibration.json"
VERIFIED_MANIFEST = STEP1 / "verified.jsonl"
CONFLICTS = STEP1 / "conflicts.csv"
SHEET = STEP1 / "conflicts_sheet.png"
ADMIT_SUMMARY = STEP1 / "admit_summary.json"
POOL_VERDICTS = STEP1 / "pool_verdicts.npz"
REGISTRY = C.REPO / "results" / "framework" / "dataset_registry.json"
FLAGS = C.REPO / "results" / "framework" / "dataset_flags.json"

CROP_FIELDS = ("crop_id", "set", "key", "image", "source", "group", "box",
               "cx", "cy", "w", "h", "W", "H", "label", "src_name")
SKIPPED_FIELDS = ("set", "key", "box", "reason")      # reason: small | no_size
SETS = ("core", "copy", "pool")


class VerifyError(RuntimeError):
    """A condition under which a Step 1 stage must not produce its outputs."""


def log(msg):
    print("[inc.verify] %s" % msg, flush=True)


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path):
    return C.read_manifest(path)


def _require(path, what):
    if not Path(path).exists():
        raise VerifyError("%s not found at %s; run `%s` first" % (path.name, path, what))


def _sanitise(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "x"


def _is_image(name):
    return not name.startswith(".") and os.path.splitext(name)[1].lower() in C.IMG_EXTS


def _counter(d):
    return {str(k): int(v) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}


# ---------------------------------------------------------------- workers
class _Workers:
    """map() over worker processes, or in this process when procs <= 1."""

    def __init__(self, procs):
        self.procs = max(1, int(procs or 1))
        self._pool = None

    def __enter__(self):
        if self.procs > 1:
            import multiprocessing as mp
            self._pool = mp.Pool(self.procs)
        return self

    def imap(self, fn, items, chunksize=8):
        if self._pool is None:
            return map(fn, items)
        return self._pool.imap(fn, items, chunksize)

    def __exit__(self, exc_type, exc, tb):
        if self._pool is not None:
            if exc_type is None:
                self._pool.close()
            else:
                self._pool.terminate()
            self._pool.join()
            self._pool = None
        return False


def _probe_image(path):
    """(path, dhash, sha256, W, H, error), reading the file once.

    W, H are the size the trainer sees (a transposing EXIF orientation swaps
    them, as cv2 applies it). dhash is the project's one dHash, on the file as
    stored, as the never-train index was built."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        return path, None, None, 0, 0, "read: %s" % e
    sha = hashlib.sha256(data).hexdigest()
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            W, H = im.size
            try:
                orient = im.getexif().get(0x0112)
            except Exception:
                orient = None
        if orient in (5, 6, 7, 8):
            W, H = H, W
    except Exception as e:
        return path, None, sha, 0, 0, "open: %s" % e
    h = C.dhash(io.BytesIO(data))
    return path, h, sha, int(W), int(H), (None if h is not None else "dhash failed")


def _probe_all(items, workers, cache, what):
    """{abs path: (dhash, sha256, W, H)} for items [(cache key, abs path)].

    cache {key: [size, mtime_ns, dhash, sha256, W, H]} is reused when size and
    mtime still match and updated in place. A failure is not cached, so a
    transient read error is retried next run; its entry is (None, None, 0, 0)."""
    out, todo = {}, []
    for ck, p in items:
        try:
            st = os.stat(p)
        except OSError:
            out[p] = (None, None, 0, 0)
            continue
        c = cache.get(ck)
        if c and c[0] == st.st_size and c[1] == st.st_mtime_ns and c[2] is not None:
            out[p] = (c[2], c[3], c[4], c[5])
        else:
            todo.append((ck, p, st.st_size, st.st_mtime_ns))
    if todo:
        log("  %s: hashing %d image(s) (%d cached)" % (what, len(todo), len(items) - len(todo)))
    t0 = time.time()
    by_path = {p: (ck, size, mt) for ck, p, size, mt in todo}
    for i, (p, h, sha, W, H, err) in enumerate(
            workers.imap(_probe_image, [t[1] for t in todo], chunksize=16), 1):
        ck, size, mt = by_path[p]
        out[p] = (h, sha, W, H)
        if h is not None:
            cache[ck] = [size, mt, h, sha, W, H]
        if i % PROGRESS_EVERY == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-6)
            log("  %s: hashed %d/%d (%.0f img/s)" % (what, i, len(todo), rate))
    return out


def _load_cache(path):
    try:
        return _read_json(path)
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------ class join
def _names_list(names):
    """A registry class_names value as a list indexed by source id."""
    if not names:
        return []
    if isinstance(names, dict):
        try:
            ids = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return [str(names[k]) for k in sorted(names)]
        out = [""] * (max(ids) + 1 if ids else 0)
        for i, v in ids.items():
            if i >= 0:
                out[i] = v
        return out
    return [str(n) for n in names]


def inc_id_of_slot(slot):
    """Trainer slot -> INC id: slots 0-11 by the inverse of CWD12_ID_TO_SLOT,
    auxiliary slots -> OtherPlant."""
    slot = int(slot)
    return SLOT_TO_INC[slot] if 0 <= slot < N_SPECIES else OTHER


def class_join(slug, info):
    """(src id -> INC id, src id -> source name, wildcard) for one registry
    entry, through mega_trainer's own class map (the v3.60.0 species join).
    wildcard: the entry names no classes; every box becomes OtherPlant."""
    from .. import mega_trainer as MT
    names = _names_list(info.get("class_names"))
    ds_map, _ = MT._build_canonical_class_map(slug, dict(info, class_names=names))
    if "__wildcard__" in ds_map:
        return {}, {}, True
    to_inc = {int(s): inc_id_of_slot(slot) for s, slot in ds_map.items()}
    return to_inc, {i: n for i, n in enumerate(names)}, False


def other_name_status(name):
    """None when a source class name may feed the OtherPlant training sample
    (it names a plant that is not a cwd12 species), else why not: 'no_name',
    'cwd12_species', 'cwd12_related' (a genus, common-name token or relative
    of one, CWD12_RELATED_TOKENS, unless OTHER_ALLOWED_KEYS), 'generic'
    ("weed", "plant", "class3") or 'not_a_plant' (pest, disease, object)."""
    k = name_key(name or "")
    if not k:
        return "no_name"
    if species_of(name) is not None:
        return "cwd12_species"
    if k not in OTHER_ALLOWED_KEYS and any(t in k for t in CWD12_RELATED_TOKENS):
        return "cwd12_related"
    if k in GENERIC_NAME_KEYS or _GENERIC_RE.fullmatch(k):
        return "generic"
    if any(w in k for w in NON_PLANT_WORDS):
        return "not_a_plant"
    return None


def named_other_plant(name):
    """True when a source class name names a plant that is not a cwd12 species
    and not a relative or variant of one (other_name_status)."""
    return other_name_status(name) is None


# ------------------------------------------------- the pre-v3.60.0 join
# What mega_trainer did before v3.60.0 (commit 420a449: _build_canonical_class_map
# and the merge's strict remap). A class name joined a trainer slot only when
# it was, as a string, one of the slot LABELS (TRAINER_SLOT_LEGACY). A dataset
# with at least four such names (or a leave-4-out copy) was a "cottonweed
# dataset": its other names were deleted. Any other dataset's other names went
# to an auxiliary slot (OtherPlant here); no names: every box to one auxiliary
# slot. A source id with no entry was deleted. The slot then holds the species
# SLOT_TO_INC says, which is how 3SeasonWeedDet10's Purslane became Palmer
# amaranth. The two leave-4-out copies are read through the lists
# dataset_discovery registered for them before v3.60.0 (legacy labels of
# Config.TRAIN_SPECIES_IDS and HOLDOUT_SPECIES_IDS), whatever the registry
# holds now: cottonweed_holdout's four names over twelve-id files.
DELETED = None
OLD_REGISTRY_NAMES = {
    "cottonweed_sp8": [CWD12_LEGACY_LABELS[i] for i in (0, 1, 6, 7, 8, 9, 10, 11)],
    "cottonweed_holdout": [CWD12_LEGACY_LABELS[i] for i in (2, 3, 4, 5)],
}
assert OLD_REGISTRY_NAMES["cottonweed_sp8"] == TRAINER_SLOT_LEGACY[:8]


def old_join(slug, class_names):
    """(src id -> INC id or DELETED, default for other src ids) under the
    pre-v3.60.0 join, with class_names read as that code read them (a dict
    iterates its keys)."""
    raw = OLD_REGISTRY_NAMES.get(slug, class_names) or []
    names = [str(n) for n in raw]
    legacy = list(TRAINER_SLOT_LEGACY)
    hits = sum(1 for n in names if n in legacy)
    if slug in OLD_REGISTRY_NAMES or hits >= 4:
        if not names:
            return {i: SLOT_TO_INC[s] for i, s in CWD12_ID_TO_SLOT.items()}, DELETED
        return {i: SLOT_TO_INC[legacy.index(n)] if n in legacy else DELETED
                for i, n in enumerate(names)}, DELETED
    if names:
        return {i: SLOT_TO_INC[legacy.index(n)] if n in legacy else OTHER
                for i, n in enumerate(names)}, DELETED
    return {}, OTHER


def _check_join_version():
    """Refuse to build the pool through a pre-v3.60.0 mega_trainer (a stale
    package copy): the current join must read real names by species."""
    got, _names, _w = class_join("__join_check__", {"class_names": ["Purslane", "Ragweed", "Crabgrass"],
                                                    "annotation": "bbox"})
    want = {0: CWD12_SPECIES.index("Purslane"), 1: CWD12_SPECIES.index("Ragweed"), 2: OTHER}
    if got != want:
        raise VerifyError("mega_trainer's class join is not the v3.60.0 species join (Purslane, "
                          "Ragweed, Crabgrass -> %s, want %s): a stale package copy?" % (got, want))


def read_source_label(path):
    """([(src_cls, cx, cy, w, h)], bad lines, clipped boxes) of a harvested
    YOLO label file. A polygon line (YOLO-seg) becomes its bounding box; boxes
    are clipped to the frame; a degenerate or malformed line is counted and
    left out."""
    boxes, bad, clipped = [], 0, 0
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if not t:
                continue
            try:
                c = float(t[0])
                v = [float(x) for x in t[1:]]
            except ValueError:
                bad += 1
                continue
            if c != int(c) or c < 0 or not all(np.isfinite(v)):
                bad += 1
                continue
            if len(v) == 4:
                cx, cy, w, h = v
                x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
            elif len(v) >= 6 and len(v) % 2 == 0:
                x0, x1, y0, y1 = min(v[0::2]), max(v[0::2]), min(v[1::2]), max(v[1::2])
            else:
                bad += 1
                continue
            cx0, cx1, cy0, cy1 = max(0.0, x0), min(1.0, x1), max(0.0, y0), min(1.0, y1)
            if (cx0, cx1, cy0, cy1) != (x0, x1, y0, y1):
                clipped += 1
            if cx1 <= cx0 or cy1 <= cy0:
                bad += 1
                continue
            boxes.append((int(c), (cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
    return boxes, bad, clipped


def _yolo_text(boxes):
    """What common.write_yolo writes, so label_sha256 needs no re-read."""
    return "".join("%d %.6f %.6f %.6f %.6f\n" % (int(b[0]), b[1], b[2], b[3], b[4])
                   for b in boxes)


def _layout(root):
    """[(split, image dir, label dir)] of a harvested dataset: images/ + labels/
    and/or <split>/images + <split>/labels. None when neither is there."""
    root = Path(root)
    parts = []
    if (root / "images").is_dir() and (root / "labels").is_dir():
        parts.append(("", root / "images", root / "labels"))
    for s in SPLIT_DIRS:
        if (root / s / "images").is_dir() and (root / s / "labels").is_dir():
            parts.append((s, root / s / "images", root / s / "labels"))
    return parts or None


def _load_registry():
    _require(REGISTRY, "dataset discovery")
    reg = _read_json(REGISTRY)
    ds = reg.get("datasets") if isinstance(reg.get("datasets"), dict) else reg
    return {str(k): v for k, v in ds.items()}


def _load_flags():
    try:
        flags = _read_json(FLAGS)
    except (OSError, ValueError):
        return {}
    return flags if isinstance(flags, dict) else {}


def _skip_reason(slug, info, flags, MT):
    if not isinstance(info, dict):
        return "bad_registry_entry"
    if slug in MT.NEVER_TRAIN_SLUGS:
        return "never_train"
    fl = flags.get(slug)
    if isinstance(fl, dict) and fl.get("flag") == "garbage":
        return "user_flag_garbage"
    if str(info.get("status")) == "quarantined":
        return "quarantined"
    ann = info.get("annotation")
    if ann == "yolo_autolabel":
        return "autolabel"
    if ann not in VALID_ANNOTATIONS:
        return "annotation_not_bbox"
    return None


def _resolve_dir(info):
    lp = info.get("local_path")
    if not lp:
        return None
    p = Path(lp)
    if not p.is_absolute():
        p = C.REPO / p
    return p if p.is_dir() else None


class _Keys:
    """Unique, path-safe, length-capped manifest keys, the same on every run."""

    def __init__(self):
        self.used = set()

    def make(self, slug, split, stem, rel):
        key = "%s__%s%s" % (_sanitise(slug), (split + "__") if split else "", _sanitise(stem))
        tag = C.sha256_text("%s/%s" % (slug, rel))[:8]
        if len(key.encode("utf-8")) > KEY_MAX:
            key = key.encode("utf-8")[:KEY_MAX - 10].decode("utf-8", "ignore") + "__" + tag
        if key in self.used:
            key = "%s__%s" % (key, tag)
        if key in self.used:
            raise VerifyError("key collision for %s/%s" % (slug, rel))
        self.used.add(key)
        return key


# ------------------------------------------------------------------- pool
def _core_rows():
    _require(C.manifest_path("train_core"), "inc.splits build")
    return sorted(C.read_manifest(C.manifest_path("train_core")), key=lambda r: r["key"])


def _lock_check():
    """sha256 of the train_core manifest and of the never-train index, checked
    against LOCK.json when the splits are locked (the pool's eval guard is only
    as good as the index it reads)."""
    nt_sha = C.sha256_file(C.NEVER_TRAIN_INDEX)
    if not C.LOCK_PATH.exists():
        log("WARNING: %s does not exist; train_core and the never-train index are used unlocked"
            % C.LOCK_PATH)
        return {"train_core_sha256": C.sha256_file(C.manifest_path("train_core")),
                "nevertrain_sha256": nt_sha, "locked": False, "nevertrain_locked": False}
    lock = C.read_lock()
    out = {"train_core_sha256": C.verify_manifest_against_lock("train_core", lock),
           "nevertrain_sha256": nt_sha, "locked": True}
    want = lock.get("nevertrain_sha256")
    if want is None:
        log("WARNING: %s records no nevertrain_sha256; the never-train index is used unchecked"
            % C.LOCK_PATH)
        out["nevertrain_locked"] = False
        return out
    if nt_sha != want:
        raise VerifyError("never-train index %s changed since it was locked (%s != %s)"
                          % (C.NEVER_TRAIN_INDEX, nt_sha[:12], str(want)[:12]))
    out["nevertrain_locked"] = True
    return out


def _label_file(base, slug, key, text):
    """Where a converted label with content `text` lives, and its sha256. The
    content's hash is in the name, so a path always holds the same bytes: a
    manifest that names it (verified.jsonl, select's base and increments) can
    never see its label change under it, whatever a later `pool` run makes."""
    sha = C.sha256_text(text)
    return Path(base) / _sanitise(slug) / ("%s.%s.txt" % (key, sha[:16])), sha


def _write_label(path, text):
    """Write a label file atomically unless it is there already (same name,
    same content). Returns True when it was written."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return True


def _check_label_format():
    """_yolo_text must write what common.write_yolo writes (the INC label
    format), so label_sha256 means the same everywhere."""
    boxes = [(0, 0.5, 0.25, 0.125, 0.0625), (OTHER, 0.1234567, 0.9, 0.3, 0.33333333)]
    tmp = CACHE_DIR / ("format_check.%d.txt" % os.getpid())
    try:
        C.write_yolo(tmp, boxes)
        same = C.sha256_file(tmp) == C.sha256_text(_yolo_text(boxes))
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if not same:
        raise VerifyError("common.write_yolo's format changed; _yolo_text must follow")


def cmd_pool(args):
    """Build pool.jsonl, pool_meta.jsonl, cwd12_copies.jsonl, pool_summary.json."""
    from .. import mega_trainer as MT
    t0 = time.time()
    _check_join_version()
    registry = _load_registry()
    flags = _load_flags()
    guard = C.NeverTrainGuard.load()
    lock = _lock_check()
    core = _core_rows()
    calib_only = set(MT.CWD12_COPY_ID_MAPS)
    log("pool: %d registry entries, never-train index %d images, train_core %d images"
        % (len(registry), guard.n, len(core)))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _check_label_format()

    with _Workers(args.procs) as workers:
        core_cache = _load_cache(CORE_PROBE)
        core_probe = _probe_all([(r["image"], r["image"]) for r in core], workers,
                                core_cache, "train_core")
        _write_json(CORE_PROBE, core_cache)
        core_index = NearHashIndex()
        core_by_key = {}
        core_unhashable = 0
        for r in core:
            h = core_probe[r["image"]][0]
            if h is None:
                core_unhashable += 1
                continue
            core_index.add(h, r["key"], max_bits=CORE_COPY_BITS)
            core_by_key[r["key"]] = r
        if core_unhashable:
            log("WARNING: %d train_core image(s) cannot be hashed; copies of them go undetected"
                % core_unhashable)

        keys = _Keys()
        skipped = collections.defaultdict(list)
        per_slug = {}
        seen_exact = {}
        pool_rows, meta_rows, copy_rows = [], [], []
        dropped_total = collections.Counter()
        class_boxes = collections.Counter()
        labels_written = 0
        slugs = sorted(registry)
        for si, slug in enumerate(slugs, 1):
            info = registry[slug]
            reason = _skip_reason(slug, info, flags, MT)
            root = None
            if reason is None:
                root = _resolve_dir(info)
                if root is None:
                    reason = "missing_dir"
            parts = None
            if reason is None:
                parts = _layout(root)
                if parts is None:
                    reason = "unknown_layout"
            if reason is not None:
                skipped[reason].append(slug)
                continue
            only_copies = slug in calib_only
            to_inc, src_names, wildcard = class_join(slug, info)
            old_map, old_default = old_join(slug, info.get("class_names"))
            st = {"layout": [p[0] or "." for p in parts], "images_listed": 0, "kept": 0,
                  "calibration_only": only_copies,
                  "dropped": collections.Counter(), "bad_label_lines": 0, "clipped_boxes": 0,
                  "boxes": collections.Counter(), "near_eval_by_split": collections.Counter(),
                  "cwd12_copies": 0,
                  "join": ({"*": [None, C.CLASS_NAMES[OTHER]]} if wildcard else
                           {str(i): [src_names.get(i, ""), C.CLASS_NAMES[c]]
                            for i, c in sorted(to_inc.items())}),
                  "old_join": dict({str(i): (C.CLASS_NAMES[c] if c is not None else "deleted")
                                    for i, c in sorted(old_map.items())},
                                   **{"*": C.CLASS_NAMES[old_default] if old_default is not None
                                      else "deleted"})}
            # 1. list once per directory; read labels; join classes
            items = []
            for split, idir, ldir in parts:
                labels = set(os.listdir(ldir))
                for name in sorted(os.listdir(idir)):
                    if not _is_image(name):
                        continue
                    st["images_listed"] += 1
                    stem = os.path.splitext(name)[0]
                    rel = "%s/images/%s" % (split, name) if split else "images/%s" % name
                    if stem + ".txt" not in labels:
                        st["dropped"]["no_label"] += 1
                        continue
                    try:
                        src, bad, clipped = read_source_label(ldir / (stem + ".txt"))
                    except (OSError, UnicodeDecodeError):
                        st["dropped"]["unreadable_label"] += 1
                        continue
                    st["bad_label_lines"] += bad
                    st["clipped_boxes"] += clipped
                    if not src:
                        st["dropped"]["no_boxes"] += 1
                        continue
                    if wildcard:
                        inc = [OTHER] * len(src)
                    else:
                        inc = [to_inc.get(b[0]) for b in src]
                        if any(c is None for c in inc):
                            # a class id the registry's list does not have: the
                            # list does not describe these files
                            st["dropped"]["unmapped_class"] += 1
                            continue
                    boxes = [(c,) + tuple(b[1:]) for c, b in zip(inc, src)]
                    meta_src = [[b[0], "" if wildcard else src_names.get(b[0], "")] for b in src]
                    old = [old_map.get(b[0], old_default) for b in src]
                    items.append({"split": split, "stem": stem, "rel": rel,
                                  "image": str(Path(idir) / name), "boxes": boxes, "src": meta_src,
                                  "old": old})
            # 2. hash (cached per slug)
            cache_path = CACHE_DIR / "dhash" / ("%s.json" % _sanitise(slug))
            cache = _load_cache(cache_path)
            probe = _probe_all([(it["rel"], it["image"]) for it in items], workers, cache,
                               "%s [%d/%d]" % (slug, si, len(slugs)))
            _write_json(cache_path, cache)
            hashable = [it["image"] for it in items if probe[it["image"]][0] is not None]
            hits, _ = guard.check(hashable, hash_fn=lambda p: probe[p][0])
            hit_by_image = {h[0]: h for h in hits}
            # 3. decide, in file order
            for it in items:
                h, sha, W, H = probe[it["image"]]
                if h is None or not W or not H:
                    st["dropped"]["unhashable"] += 1
                    continue
                hit = hit_by_image.get(it["image"])
                if hit is not None:
                    st["dropped"]["near_eval"] += 1
                    st["near_eval_by_split"][hit[1]] += 1
                    continue
                m = core_index.find(h)
                if m is None and only_copies:
                    # a leave-4-out copy's photograph that is not a train_core
                    # one: no known truth here, and never pool data
                    st["dropped"]["calibration_only_not_train_core"] += 1
                    continue
                key = keys.make(slug, it["split"], it["stem"], it["rel"])
                text = _yolo_text(it["boxes"])
                if m is not None:
                    orig = core_by_key[m[0]]
                    lbl, lsha = _label_file(COPY_LABELS_DIR, slug, key, text)
                    labels_written += _write_label(lbl, text)
                    copy_rows.append({
                        "image": it["image"], "label": str(lbl), "sha256": sha,
                        "label_sha256": lsha, "source": slug, "session": "",
                        "key": key, "W": W, "H": H, "dhash": h, "bits": int(m[1]), "src": it["src"],
                        "old_join": it["old"], "calibration_only": only_copies,
                        "train_core_key": orig["key"], "train_core_image": orig["image"],
                        "train_core_label": orig["label"], "train_core_session": orig["session"]})
                    st["dropped"]["cwd12_copy"] += 1
                    st["cwd12_copies"] += 1
                    continue
                if h in seen_exact:
                    st["dropped"]["exact_dup"] += 1
                    continue
                seen_exact[h] = key
                lbl, lsha = _label_file(LABELS_DIR, slug, key, text)
                labels_written += _write_label(lbl, text)
                pool_rows.append({"image": it["image"], "label": str(lbl), "sha256": sha,
                                  "label_sha256": lsha, "source": slug,
                                  "session": "", "key": key})
                meta_rows.append({"key": key, "source": slug, "W": W, "H": H, "dhash": h,
                                  "boxes": [list(b) for b in it["boxes"]], "src": it["src"]})
                st["kept"] += 1
                for b in it["boxes"]:
                    st["boxes"][C.CLASS_NAMES[b[0]]] += 1
            for k, v in st["dropped"].items():
                dropped_total[k] += v
            class_boxes.update(st["boxes"])
            for k in ("dropped", "boxes", "near_eval_by_split"):
                st[k] = _counter(st[k])
            per_slug[slug] = st
            log("[%d/%d] %s%s: %d listed, %d kept, %d cwd12 copies, dropped %s"
                % (si, len(slugs), slug, " (calibration copies only)" if only_copies else "",
                   st["images_listed"], st["kept"], st["cwd12_copies"], st["dropped"] or "none"))

    manifest_sha = C.write_manifest(POOL, pool_rows)
    _write_jsonl(POOL_META, sorted(meta_rows, key=lambda r: r["key"]))
    _write_jsonl(COPIES, sorted(copy_rows, key=lambda r: r["key"]))
    summary = {
        "built_utc": _utc(), "seconds": round(time.time() - t0, 1),
        "registry": str(REGISTRY), "flags": str(FLAGS), "lock": lock,
        "never_train_index": {"entries": guard.n, "path": str(C.NEVER_TRAIN_INDEX),
                              "sha256": lock["nevertrain_sha256"],
                              "matches_lock": lock.get("nevertrain_locked", False)},
        "train_core": {"images": len(core), "unhashable": core_unhashable},
        "near_dup_bits_train_core": CORE_COPY_BITS,
        "slugs_total": len(registry), "slugs_used": len(per_slug),
        "calibration_only_slugs": sorted(s for s in per_slug if s in calib_only),
        "skipped": {k: sorted(v) for k, v in sorted(skipped.items())},
        "skipped_counts": {k: len(v) for k, v in sorted(skipped.items())},
        "unsure_flagged_used": sorted(s for s in per_slug
                                      if (flags.get(s) or {}).get("flag") == "unsure"),
        "images": len(pool_rows), "boxes": sum(class_boxes.values()),
        "boxes_per_class": {n: int(class_boxes.get(n, 0)) for n in C.CLASS_NAMES},
        "dropped": _counter(dropped_total),
        "cwd12_copies": {"images": len(copy_rows),
                         "per_slug": _counter(collections.Counter(r["source"] for r in copy_rows))},
        "label_files_written": labels_written,
        "pool_sha256": manifest_sha, "pool_meta_sha256": C.sha256_file(POOL_META),
        "copies_sha256": C.sha256_file(COPIES),
        "per_slug": per_slug,
    }
    _write_json(POOL_SUMMARY, summary)
    log("pool: %d images, %d boxes from %d slugs; %d cwd12 copies; %d new label files; dropped %s; "
        "skipped slugs %s (%.0fs)"
        % (len(pool_rows), summary["boxes"], len(per_slug), len(copy_rows), labels_written,
           summary["dropped"], summary["skipped_counts"], time.time() - t0))
    return summary


# ------------------------------------------------------------------ crops
def _input_hashes():
    """sha256 of every file crops.csv is made from."""
    return {"pool_sha256": C.sha256_file(POOL), "pool_meta_sha256": C.sha256_file(POOL_META),
            "copies_sha256": C.sha256_file(COPIES),
            "train_core_sha256": C.sha256_file(C.manifest_path("train_core"))}


def cmd_crops(args):
    """crops.csv: one row per box (min side MIN_BOX_PX) of train_core, the
    cwd12 copies and the pool; crops_skipped.csv: every box left out, and why.
    Reads no image: sizes come from the pool stage."""
    from ..semisup_labeler import MIN_BOX_PX
    for p in (POOL, POOL_META, COPIES, CORE_PROBE):
        _require(p, "verify pool")
    inputs = _input_hashes()
    core = _core_rows()
    core_cache = _read_json(CORE_PROBE)
    pool_image = {r["key"]: r["image"] for r in C.read_manifest(POOL)}
    rows, left_out = [], []
    small = collections.Counter()
    no_size = collections.Counter()

    def add(set_, key, image, source, group, boxes, W, H, names):
        if not W or not H:
            no_size[set_] += 1
            left_out.extend([set_, key, b, "no_size"] for b in range(len(boxes)))
            return
        for b, box in enumerate(boxes):
            cls, cx, cy, w, h = box
            if w * W < MIN_BOX_PX or h * H < MIN_BOX_PX:
                small[set_] += 1
                left_out.append([set_, key, b, "small"])
                continue
            rows.append([len(rows), set_, key, image, source, group, b,
                         "%.6f" % cx, "%.6f" % cy, "%.6f" % w, "%.6f" % h, W, H, int(cls), names[b]])

    for r in core:
        c = core_cache.get(r["image"]) or [0, 0, None, None, 0, 0]
        boxes = [b for b in C.read_yolo(r["label"])]
        if any(not 0 <= b[0] < N_SPECIES for b in boxes):
            raise VerifyError("train_core label %s has a class outside 0..11" % r["label"])
        add("core", r["key"], r["image"], "train_core", r["session"], boxes, c[4], c[5],
            [C.CLASS_NAMES[b[0]] for b in boxes])
    for r in _read_jsonl(COPIES):
        boxes = C.read_yolo(r["label"])
        if C.sha256_file(r["label"]) != r["label_sha256"]:
            raise VerifyError("copy label %s does not hash to its label_sha256" % r["label"])
        add("copy", r["key"], r["image"], r["source"], r["source"], boxes, r["W"], r["H"],
            [s[1] for s in r["src"]])
    for m in _read_jsonl(POOL_META):
        boxes = [tuple(b) for b in m["boxes"]]
        add("pool", m["key"], pool_image[m["key"]], m["source"], m["source"], boxes,
            m["W"], m["H"], [s[1] for s in m["src"]])

    CROPS.parent.mkdir(parents=True, exist_ok=True)
    for path, header, body in ((CROPS_SKIPPED, SKIPPED_FIELDS, left_out), (CROPS, CROP_FIELDS, rows)):
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(header)
            wr.writerows(body)
        os.replace(tmp, path)
    if _input_hashes() != inputs:
        raise VerifyError("the pool changed while `verify crops` ran; rerun it")
    per = collections.Counter((r[1], r[13]) for r in rows)
    info = {"built_utc": _utc(), "crops": len(rows), "crops_sha256": C.sha256_file(CROPS),
            "skipped_sha256": C.sha256_file(CROPS_SKIPPED), "inputs": inputs,
            "per_set": {s: sum(v for (ss, _c), v in per.items() if ss == s) for s in SETS},
            "per_set_class": {s: {C.CLASS_NAMES[c]: int(per.get((s, c), 0)) for c in range(C.NC)}
                              for s in SETS},
            "images": {s: len({r[2] for r in rows if r[1] == s}) for s in SETS},
            "dropped_small_boxes": _counter(small), "images_without_size": _counter(no_size),
            "min_box_px": MIN_BOX_PX}
    _write_json(CROPS_INFO, info)
    log("crops: %d (%s); small boxes left out %s" % (len(rows), info["per_set"], dict(small)))
    return info


def check_fresh(crops):
    """crops_info.json, after checking that it describes this crops.csv and that
    nothing crops.csv was made from (pool.jsonl, pool_meta.jsonl,
    cwd12_copies.jsonl, the train_core manifest, crops_skipped.csv) changed
    since. Everything downstream of crops.csv (shards, verifier, verdicts) is
    keyed by crops.csv's sha256, so this one check covers the chain."""
    _require(CROPS_INFO, "verify crops")
    info = _read_json(CROPS_INFO)
    if info.get("crops_sha256") != crops.sha:
        raise VerifyError("%s does not describe this crops.csv; rerun `verify crops`" % CROPS_INFO.name)
    want = info.get("inputs") or {}
    got = _input_hashes()
    changed = sorted(k[:-len("_sha256")] for k in got if got[k] != want.get(k))
    if changed:
        raise VerifyError("%s changed after `verify crops` built crops.csv from it: rerun crops, "
                          "embed, fit, calibrate and admit" % ", ".join(changed))
    if C.sha256_file(CROPS_SKIPPED) != info.get("skipped_sha256"):
        raise VerifyError("%s changed after `verify crops`; rerun it" % CROPS_SKIPPED.name)
    return info


def read_skipped(set_name):
    """{(key, box): reason} of the boxes `verify crops` left out of set_name."""
    out = {}
    with open(CROPS_SKIPPED, newline="") as fh:
        rd = csv.reader(fh)
        if tuple(next(rd)) != SKIPPED_FIELDS:
            raise VerifyError("%s: unexpected columns" % CROPS_SKIPPED)
        for st, key, box, reason in rd:
            if st == set_name:
                out[(key, int(box))] = reason
    return out


def _box_verdict_lookup(crops, set_name, verdict_of_crop):
    """A function (key, box) -> verdict for set_name's boxes: the judged crop's
    verdict, SMALL / FAILED for a box `verify crops` left out as small / without
    a size. A box with neither is a crops.csv that was not made from these
    inputs, and raises."""
    row = {(crops.key[i], int(crops.box[i])): int(i) for i in crops.where(set_name)}
    skipped = read_skipped(set_name)

    def lookup(key, box):
        i = row.get((key, box))
        if i is not None:
            return verdict_of_crop(i), i
        why = skipped.get((key, box))
        if why is None:
            raise VerifyError("%s box %s/%d has no crop row and was not left out by `verify crops`: "
                              "crops.csv was made from other inputs" % (set_name, key, box))
        return (SMALL if why == "small" else FAILED), None
    return lookup


class Crops:
    """crops.csv, column-wise (a million rows as dicts would not fit well)."""

    def __init__(self, path=CROPS):
        _require(Path(path), "verify crops")
        self.path = Path(path)
        self.sha = C.sha256_file(path)
        cols = {f: [] for f in CROP_FIELDS}
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            if tuple(header) != CROP_FIELDS:
                raise VerifyError("%s: unexpected columns %s" % (path, header))
            for row in rd:
                for f, v in zip(CROP_FIELDS, row):
                    cols[f].append(v)
        self.n = len(cols["crop_id"])
        if [int(x) for x in cols["crop_id"]] != list(range(self.n)):
            raise VerifyError("%s: crop_id is not 0..n-1 in order" % path)
        self.set = np.array([SETS.index(s) for s in cols["set"]], dtype=np.int8)
        self.key, self.image, self.source = cols["key"], cols["image"], cols["source"]
        self.group, self.src_name = cols["group"], cols["src_name"]
        self.box = np.array(cols["box"], dtype=np.int64)
        for f in ("cx", "cy", "w", "h"):
            setattr(self, f, np.array(cols[f], dtype=np.float64))
        self.W = np.array(cols["W"], dtype=np.int64)
        self.H = np.array(cols["H"], dtype=np.int64)
        self.label = np.array(cols["label"], dtype=np.int64)

    def where(self, set_name):
        return np.flatnonzero(self.set == SETS.index(set_name))

    def row(self, i):
        return {"crop_id": int(i), "cx": float(self.cx[i]), "cy": float(self.cy[i]),
                "w": float(self.w[i]), "h": float(self.h[i]),
                "W": int(self.W[i]), "H": int(self.H[i])}

    def images(self):
        """[(image, [crop ids])] in crop order (crops of an image are contiguous)."""
        out, idx = [], {}
        for i, img in enumerate(self.image):
            j = idx.get(img)
            if j is None:
                idx[img] = len(out)
                out.append((img, [i]))
            else:
                out[j][1].append(i)
        return out


# ------------------------------------------------------------------ embed
class BioclipEmbedder:
    """BioCLIP-2 through semisup_labeler's loader and batch embedder (one
    definition of how a crop becomes a feature). fp16 autocast on CUDA."""

    def __init__(self, amp=True):
        from .. import semisup_labeler as SL
        from PIL import Image
        self._SL = SL
        self.device = SL._device()
        self.amp = bool(amp) and self.device == "cuda"
        self.name = EMBEDDER_NAME + ("+fp16" if self.amp else "")
        log("loading %s on %s" % (EMBEDDER_NAME, self.device))
        self.model, self.proc = SL._load_backbone(EMBEDDER_NAME, self.device)
        grey = Image.new("RGB", (SL.CROP_PX, SL.CROP_PX), (124, 124, 124))
        self.dim = int(self([grey]).shape[1])

    def __call__(self, pils):
        import contextlib
        import torch
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.amp
               else contextlib.nullcontext())
        with ctx:
            return self._SL._embed_batch(self.model, self.proc, pils, ("clip",), self.device)["clip"]


def _cut_task(task):
    """Worker: (image, crop ids, [uint8 HxWx3 crops] or None, error)."""
    image, rows = task
    ids = [r["crop_id"] for r in rows]
    try:
        from PIL import Image, ImageOps
        from ..semisup_labeler import _cut
        with Image.open(image) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")
        W, H = im.size
        return image, ids, [np.asarray(_cut(im, dict(r, W=W, H=H))) for r in rows], None
    except Exception as e:
        return image, ids, None, "%s: %s" % (type(e).__name__, e)


def _embed_safely(embedder, pils, stats):
    """(len(pils), dim) float32; a batch that fails is retried crop by crop
    and a crop that still fails, or gives a non-finite row, is NaN."""
    dim = int(embedder.dim)
    try:
        F = np.asarray(embedder(pils), dtype=np.float32)
        if F.shape != (len(pils), dim):
            raise VerifyError("embedder returned %s for %d crops" % (F.shape, len(pils)))
    except Exception as e:
        stats["batch_retries"] += 1
        if stats["batch_retries"] <= 3:
            log("  WARNING: batch of %d failed (%s: %s); retrying crop by crop"
                % (len(pils), type(e).__name__, e))
        F = np.full((len(pils), dim), np.nan, dtype=np.float32)
        for i, p in enumerate(pils):
            try:
                v = np.asarray(embedder([p]), dtype=np.float32)
                if v.shape == (1, dim):
                    F[i] = v[0]
            except Exception:
                pass
    bad = ~np.isfinite(F).all(axis=1)
    F[bad] = np.nan
    stats["failed_crops"] += int(bad.sum())
    return F


def _embed_images(images, crops, embedder, workers, batch):
    """Features for every crop of `images` [(image, crop ids)]: (crop ids, X
    float32, stats). Rows follow `crop ids`; failures are NaN rows."""
    from PIL import Image
    ids = [i for _img, cs in images for i in cs]
    pos = {c: k for k, c in enumerate(ids)}
    X = np.full((len(ids), int(embedder.dim)), np.nan, dtype=np.float32)
    stats = collections.Counter()
    buf, buf_ids = [], []

    def flush():
        F = _embed_safely(embedder, buf, stats)
        for c, v in zip(buf_ids, F):
            X[pos[c]] = v

    tasks = [(img, [crops.row(i) for i in cs]) for img, cs in images]
    for image, cids, arrays, err in workers.imap(_cut_task, tasks, chunksize=4):
        if err is not None:
            stats["failed_images"] += 1
            stats["failed_crops"] += len(cids)
            if stats["failed_images"] <= 5:
                log("  WARNING: cannot cut %s: %s" % (image, err))
            continue
        for c, a in zip(cids, arrays):
            buf.append(Image.fromarray(a))
            buf_ids.append(c)
            if len(buf) >= batch:
                flush()
                buf, buf_ids = [], []
    if buf:
        flush()
    return np.array(ids, dtype=np.int64), X, stats


def _shard_name(s, n, part=None):
    base = "emb_s%03d_of_%03d" % (s, n)
    return base + (".part%05d.npz" % part if part is not None else ".npz")


def _save_npz(path, meta, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    os.replace(tmp, path)


def _npz_meta(path):
    try:
        with np.load(path) as d:
            return json.loads(str(d["meta"]))
    except Exception:
        return None


def _shard_done(s, n, crops, force=False):
    """The finished shard's meta when it is current, else None."""
    final = EMB_DIR / _shard_name(s, n)
    meta = _npz_meta(final) if final.exists() else None
    return meta if (meta and meta.get("crops_sha256") == crops.sha and not force) else None


def _embed_shard(s, n, crops, images, args, get_embedder, workers):
    final = EMB_DIR / _shard_name(s, n)
    meta = _npz_meta(final) if final.exists() else None
    if meta and meta.get("crops_sha256") == crops.sha and not args.force:
        log("embed shard %d/%d: done already (%s)" % (s, n, final.name))
        return meta
    if meta:
        log("embed shard %d/%d: %s was made from another crops.csv; recomputing" % (s, n, final.name))
    lo, hi = s * len(images) // n, (s + 1) * len(images) // n
    mine = images[lo:hi]
    want = np.array([c for _img, cs in mine for c in cs], dtype=np.int64)
    chunk = max(1, int(args.chunk_images))
    n_parts = (len(mine) + chunk - 1) // chunk
    log("embed shard %d/%d: %d images, %d crops, %d chunk(s) of %d images"
        % (s, n, len(mine), len(want), n_parts, chunk))
    t0 = time.time()
    done_crops = 0
    parts = []
    for j in range(n_parts):
        part = EMB_DIR / _shard_name(s, n, j)
        parts.append(part)
        pm = _npz_meta(part) if part.exists() else None
        if (pm and not args.force and pm.get("crops_sha256") == crops.sha
                and pm.get("chunk_images") == chunk):
            continue
        embedder = get_embedder()
        t1 = time.time()
        ids, X, st = _embed_images(mine[j * chunk:(j + 1) * chunk], crops, embedder,
                                   workers, int(args.batch))
        _save_npz(part, {"crops_sha256": crops.sha, "chunk_images": chunk, "embedder": embedder.name,
                         "dim": int(embedder.dim), "stats": dict(st),
                         "seconds": round(time.time() - t1, 1)},
                  crop_ids=ids, X=X.astype(np.float16))
        done_crops += len(ids)
        rate = done_crops / max(time.time() - t0, 1e-6)
        log("  shard %d/%d chunk %d/%d: %d crops (%d failed), %.1f crops/s"
            % (s, n, j + 1, n_parts, len(ids), st.get("failed_crops", 0), rate))
    ids, Xs, names, dims, stats, secs = [], [], set(), set(), collections.Counter(), 0.0
    for part in parts:
        with np.load(part) as d:
            pm = json.loads(str(d["meta"]))
            ids.append(d["crop_ids"])
            Xs.append(d["X"])
        names.add(pm["embedder"])
        dims.add(pm["dim"])
        stats.update(pm.get("stats", {}))
        secs += pm.get("seconds", 0.0)
    if len(names) > 1 or len(dims) > 1:
        raise VerifyError("shard %d/%d: chunks embedded by different models %s; rerun with --force"
                          % (s, n, sorted(names)))
    ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    if not np.array_equal(np.sort(ids), np.sort(want)) or len(set(ids.tolist())) != len(ids):
        raise VerifyError("shard %d/%d: chunks do not cover the shard's crops exactly once" % (s, n))
    dim = dims.pop() if dims else 0
    X = np.concatenate(Xs) if Xs else np.zeros((0, dim), dtype=np.float16)
    meta = {"crops_sha256": crops.sha, "shard": s, "nshards": n, "images": len(mine),
            "crops": int(len(ids)), "embedder": names.pop() if names else None, "dim": int(dim),
            "stats": dict(stats), "seconds": round(secs, 1), "built_utc": _utc()}
    _save_npz(final, meta, crop_ids=ids, X=X)
    for part in parts:
        part.unlink()
    log("embed shard %d/%d: %d crops, %d failed -> %s" % (s, n, len(ids), stats.get("failed_crops", 0),
                                                           final.name))
    return meta


def cmd_embed(args, embedder=None):
    """BioCLIP-2 features for every crop; --shard i --nshards n does one shard."""
    crops = Crops()
    check_fresh(crops)
    images = crops.images()
    n = int(args.nshards or 1)
    shards = [int(args.shard)] if args.shard is not None else list(range(n))
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    holder = {"e": embedder}

    def get_embedder():
        if holder["e"] is None:
            holder["e"] = BioclipEmbedder(amp=not args.no_amp)
        return holder["e"]

    todo = [s for s in shards if _shard_done(s, n, crops, args.force) is None]
    out = [_shard_done(s, n, crops) for s in shards if s not in todo]
    for s in shards:
        if s not in todo:
            log("embed shard %d/%d: done already" % (s, n))
    if not todo:
        return out
    # workers first: fork before the model is on the GPU
    with _Workers(args.procs) as workers:
        for s in todo:
            out.append(_embed_shard(s, n, crops, images, args, get_embedder, workers))
    return out


def load_embeddings(crops, nshards=None):
    """(X float16 [crops.n, dim], info): a complete, current set of shards,
    every crop id exactly once; NaN rows are crops that failed."""
    files = collections.defaultdict(dict)
    if EMB_DIR.is_dir():
        for name in os.listdir(EMB_DIR):
            m = re.fullmatch(r"emb_s(\d{3})_of_(\d{3})\.npz", name)
            if m:
                files[int(m.group(2))][int(m.group(1))] = EMB_DIR / name
    problems = []
    for n in ([int(nshards)] if nshards else sorted(files)):
        have = files.get(n, {})
        missing = [s for s in range(n) if s not in have]
        metas = {s: _npz_meta(p) for s, p in have.items()}
        stale = [s for s, m in metas.items() if not m or m.get("crops_sha256") != crops.sha]
        if missing or stale:
            problems.append("nshards=%d: missing shards %s, stale shards %s" % (n, missing, stale))
            continue
        filled = [m for m in metas.values() if m.get("crops", 0) > 0]   # a shard can be empty
        names = {m["embedder"] for m in filled}
        dims = {m["dim"] for m in filled}
        if len(names) != 1 or len(dims) != 1:
            problems.append("nshards=%d: shards from different embedders %s" % (n, sorted(names)))
            continue
        dim = dims.pop()
        X = np.full((crops.n, dim), np.nan, dtype=np.float16)
        seen = np.zeros(crops.n, dtype=np.int64)
        failed = 0
        for s in range(n):
            with np.load(have[s]) as d:
                ids = d["crop_ids"]
                if len(ids):
                    X[ids] = d["X"]
                    np.add.at(seen, ids, 1)
            failed += int(metas[s].get("stats", {}).get("failed_crops", 0))
        if not (seen == 1).all():
            problems.append("nshards=%d: %d crop(s) not covered exactly once"
                            % (n, int((seen != 1).sum())))
            continue
        info = {"nshards": n, "embedder": names.pop(), "dim": int(dim), "failed_crops": failed,
                "nan_rows": int((~np.isfinite(X).all(axis=1)).sum())}
        return X, info
    raise VerifyError("no complete, current set of embedding shards in %s (%s); run `verify embed`"
                      % (EMB_DIR, "; ".join(problems) or "none found"))


# ------------------------------------------------------------ the verifier
def _norm(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def verdicts(labels, P, cos, tau_p, sigma):
    """Box verdicts from probe probabilities P [n, 13] and prototype cosines
    cos [n, 13]; the one rule calibrate and admit share.

    j = argmax P. confident(j) = P_j >= tau_p[j] and (j == OtherPlant or
    cos_j >= sigma[j]). A species label l: verified if j == l and confident,
    conflict if j != l and confident, else unknown. An OtherPlant label:
    conflict if j is a species and confident, else other_ok. A row whose
    probabilities are not finite is failed. Returns (verdict, j, p_j, cos_j)."""
    labels = np.asarray(labels, dtype=np.int64)
    P = np.asarray(P, dtype=np.float64)
    cos = np.asarray(cos, dtype=np.float64)
    n = len(labels)
    if n == 0:
        return (np.array([], dtype=object), np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
    if labels.min() < 0 or labels.max() >= C.NC:
        raise ValueError("labels outside 0..%d" % (C.NC - 1))
    finite = np.isfinite(P).all(axis=1)
    j = np.where(finite, np.nan_to_num(P, nan=-1.0).argmax(axis=1), 0)
    rows = np.arange(n)
    pj, cj = P[rows, j], cos[rows, j]
    tau = np.asarray(tau_p, dtype=np.float64)[j]
    sig = np.asarray(sigma, dtype=np.float64)[j]
    with np.errstate(invalid="ignore"):
        conf = finite & (pj >= tau) & ((j == OTHER) | (cj >= sig))
    out = np.full(n, UNKNOWN, dtype=object)
    sp = labels < OTHER
    out[sp & (j == labels) & conf] = VERIFIED
    out[sp & (j != labels) & conf] = CONFLICT
    ot = labels == OTHER
    out[ot] = OTHER_OK
    out[ot & (j < OTHER) & conf] = CONFLICT
    out[~finite] = FAILED
    return out, j, pj, cj


def image_verdict(labels, box_verdicts):
    """admitted: every species box verified, every OtherPlant box other_ok
    (or too small to embed), no conflict. conflict: any box a conflict.
    unknown: anything else (a species box not confidently confirmed, a box
    whose features failed)."""
    vs = list(box_verdicts)
    if CONFLICT in vs:
        return CONFLICT
    ok = all((v == VERIFIED) if int(l) < OTHER else (v in (OTHER_OK, SMALL))
             for l, v in zip(labels, vs))
    return ADMITTED if ok and vs else UNKNOWN


def _assign_folds(groups, n_folds, seed, random_ok):
    """Fold id per sample: GroupKFold over groups; with fewer groups than folds,
    one fold per group, or (random_ok) a seeded random split by sample."""
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    u = np.unique(groups)
    if len(u) >= n_folds:
        k = n_folds
    elif random_ok:
        rng = np.random.default_rng(seed)
        return rng.permutation(n) % n_folds
    else:
        k = len(u)
    if k < 2:
        raise VerifyError("need at least 2 groups to cross-validate, got %d" % len(u))
    folds = np.empty(n, dtype=np.int64)
    for f, (_tr, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(n), groups=groups)):
        folds[te] = f
    return folds


def _train_probe(X, y):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER,
                                  class_weight="balanced").fit(X, y)


def _prototypes(X, y):
    """[13, dim] unit class means; NaN rows for classes without samples."""
    P = np.full((C.NC, X.shape[1]), np.nan, dtype=np.float32)
    for k in range(C.NC):
        m = y == k
        if m.any():
            v = X[m].mean(axis=0)
            P[k] = v / max(float(np.linalg.norm(v)), 1e-8)
    return P


class Verifier:
    """Probe + prototypes + per-class thresholds. X passed in must be finite
    and L2-normalised (judge_features does both)."""

    def __init__(self, probe, protos, tau_p, sigma, info=None):
        self.probe = probe
        self.protos = np.asarray(protos, dtype=np.float32)
        self.tau_p = np.asarray(tau_p, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.info = info or {}

    def scores(self, Xn):
        P = np.zeros((len(Xn), C.NC))
        if len(Xn):
            P[:, self.probe.classes_] = self.probe.predict_proba(Xn)
        with np.errstate(invalid="ignore"):
            cos = Xn @ self.protos.T
        return P, cos

    def judge(self, labels, Xn):
        P, cos = self.scores(Xn)
        return verdicts(labels, P, cos, self.tau_p, self.sigma)

    def judge_features(self, labels, X, chunk=50000):
        """Verdicts for raw features X (any float dtype; NaN rows -> failed),
        in chunks so a million crops need not be normalised at once."""
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        v = np.full(n, FAILED, dtype=object)
        j = np.full(n, -1, dtype=np.int64)
        pj = np.full(n, np.nan)
        cj = np.full(n, np.nan)
        for a in range(0, n, chunk):
            Xa = np.asarray(X[a:a + chunk], dtype=np.float32)
            ok = np.isfinite(Xa).all(axis=1)
            if ok.any():
                idx = a + np.flatnonzero(ok)
                v[idx], j[idx], pj[idx], cj[idx] = self.judge(labels[idx], _norm(Xa[ok]))
        return v, j, pj, cj

    def save(self, d, extra=None):
        import joblib
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "probe.joblib.tmp"
        joblib.dump(self.probe, tmp)
        os.replace(tmp, d / "probe.joblib")
        extra = dict(extra or {})
        arrays = {k: extra.pop(k) for k in list(extra) if isinstance(extra[k], np.ndarray)}
        _save_npz(d / "verifier.npz", {"info": self.info, **extra}, protos=self.protos,
                  tau_p=self.tau_p, sigma=self.sigma, **arrays)
        pc = self.info.get("per_class") or {}
        _write_json(d / "thresholds.json", {
            "recall_target": RECALL, "class_names": C.CLASS_NAMES,
            "rule": self.info.get("threshold_rule"),
            "tau_p": {C.CLASS_NAMES[k]: _f(self.tau_p[k]) for k in range(C.NC)},
            "sigma": {C.CLASS_NAMES[k]: _f(self.sigma[k]) for k in range(C.NC)},
            "recall_cv": {n: (pc.get(n) or {}).get("cv_confirmed") for n in C.CLASS_NAMES},
            "note": "null = no calibration data: the class is never confidently predicted. "
                    "recall_cv: share of held-out true-class crops verified (OtherPlant: not "
                    "called a conflict)"})

    @classmethod
    def load(cls, d=VERIFIER_DIR):
        import joblib
        d = Path(d)
        _require(d / "verifier.npz", "verify fit")
        with np.load(d / "verifier.npz") as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k not in ("meta",)}
        ver = cls(joblib.load(d / "probe.joblib"), arrays["protos"], arrays["tau_p"],
                  arrays["sigma"], meta.get("info"))
        ver.meta, ver.arrays = meta, arrays
        return ver


def _f(x):
    x = float(x)
    return round(x, 6) if np.isfinite(x) else None


def joint_thresholds(p, cos, top, k, recall=RECALL, max_cond=MAX_COND_PASS):
    """(tau, sigma, info) for class k from held-out true-k crops: p and cos are
    their probability and prototype cosine for k, top marks the crops the probe
    ranked k first.

    The two thresholds and the argmax act together (verdicts), so they are set
    together: both at the same rank r of the sorted p and cos of the top crops,
    the highest r at which a held-out true-k crop is still verified with rate
    `recall`. That needs a pass rate of recall * n / n_top among the top crops;
    it is capped at max_cond, so a class the probe ranks first too rarely is
    not loosened to the least typical of its top crops (its recall is then
    below target and says so). OtherPlant has no cosine test. A class with
    fewer than MIN_CAL top crops gets infinite thresholds (never confident)."""
    p = np.asarray(p, dtype=np.float64)
    cos = np.asarray(cos, dtype=np.float64)
    top = np.asarray(top, dtype=bool)
    n, n_top = int(len(p)), int(top.sum())
    info = {"n": n, "n_top": n_top}
    if n_top < MIN_CAL:
        info.update(rule="too_few", target_reachable=False, recall_cv=None)
        return np.inf, np.inf, info
    cond = min(recall * n / n_top, max_cond)
    pt, ct = p[top], cos[top]
    sp, sc = np.sort(pt), np.sort(ct)
    use_cos = k != OTHER

    def pass_rate(r):
        ok = pt >= sp[r]
        if use_cos:
            ok &= ct >= sc[r]
        return ok.mean()

    lo, hi = 0, n_top - 1          # pass_rate(0) == 1 >= cond; non-increasing in r
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pass_rate(mid) >= cond:
            lo = mid
        else:
            hi = mid - 1
    tau, sigma = float(sp[lo]), float(sc[lo])
    got = pt >= tau
    if use_cos:
        got &= ct >= sigma
    info.update(rule="joint", rank=int(lo), cond_pass_target=round(float(cond), 6),
                target_reachable=bool(recall * n <= max_cond * n_top),
                recall_cv=round(float(got.sum()) / n, 6))
    return tau, sigma, info


def fit_verifier(Xc, yc, gc, Xo=None, go=None, n_folds=N_FOLDS, seed=0):
    """Fit the verifier: probe on species crops Xc/yc (+ OtherPlant crops Xo),
    thresholds by n_folds-fold cross-validation grouped by gc (capture session)
    and go (source slug; random folds when there are too few slugs).

    Inputs must be finite and L2-normalised. A class held out of a fold's
    training set (all its sessions in one fold) has no prediction to calibrate
    on there; those crops are left out of its thresholds. Thresholds per class:
    joint_thresholds. The out-of-fold probabilities and cosines of the Xo rows
    are kept (verifier.oof_other) so the crops the final probe was trained on
    can be judged by a model that did not see them."""
    Xc = np.asarray(Xc, dtype=np.float32)
    yc = np.asarray(yc, dtype=np.int64)
    Xo = np.zeros((0, Xc.shape[1]), dtype=np.float32) if Xo is None else np.asarray(Xo, np.float32)
    go = np.zeros(0, dtype=object) if go is None else np.asarray(go)
    fc = _assign_folds(gc, n_folds, seed, random_ok=False)
    nf = int(fc.max()) + 1
    fo = _assign_folds(go, nf, seed, random_ok=True) if len(Xo) else np.zeros(0, np.int64)
    X_all = np.concatenate([Xc, Xo])
    y_all = np.concatenate([yc, np.full(len(Xo), OTHER, dtype=np.int64)])
    f_all = np.concatenate([fc, fo])
    Ph = np.full((len(X_all), C.NC), np.nan)
    Ch = np.full((len(X_all), C.NC), np.nan)
    usable = np.zeros(len(X_all), dtype=bool)   # its class was in its fold's training set
    for f in range(nf):
        tr, te = f_all != f, f_all == f
        if not te.any():
            continue
        if len(np.unique(y_all[tr])) < 2:
            raise VerifyError("fold %d: fewer than 2 classes to train on" % f)
        probe = _train_probe(X_all[tr], y_all[tr])
        protos = _prototypes(X_all[tr], y_all[tr])
        v = Verifier(probe, protos, np.zeros(C.NC), np.zeros(C.NC))
        Ph[te], Ch[te] = v.scores(X_all[te])
        usable[te] = np.isin(y_all[te], probe.classes_)
    tau_p = np.full(C.NC, np.inf)
    sigma = np.full(C.NC, np.inf)
    per_class = {}
    held = usable & np.isfinite(Ph).all(axis=1)
    top_all = np.zeros(len(X_all), dtype=bool)
    top_all[held] = Ph[held].argmax(axis=1) == y_all[held]
    for k in range(C.NC):
        m = held & (y_all == k)
        tau_p[k], sigma[k], th = joint_thresholds(Ph[m, k], Ch[m, k], top_all[m], k)
        per_class[C.CLASS_NAMES[k]] = {"train": int((y_all == k).sum()), "calibration": int(m.sum()),
                                       "held_out_unseen": int(((y_all == k) & ~usable).sum()),
                                       "thresholds": th}
    # in-sample description of the thresholds on the held-out predictions
    v, j, _pj, _cj = verdicts(y_all[held], Ph[held], Ch[held], tau_p, sigma)
    yh = y_all[held]
    for k in range(C.NC):
        m = yh == k
        pc = per_class[C.CLASS_NAMES[k]]
        pc["tau_p"], pc["sigma"] = _f(tau_p[k]), _f(sigma[k])
        pc["cv_top1"] = _f((j[m] == k).mean()) if m.any() else None
        pc["cv_confirmed"] = (_f(np.isin(v[m], (VERIFIED, OTHER_OK)).mean()) if m.any() else None)
        pc["cv_conflict"] = _f((v[m] == CONFLICT).mean()) if m.any() else None
        if k < OTHER and not np.isfinite(tau_p[k]):
            log("WARNING: %s has %d top-ranked calibration crop(s) (< %d): never verified"
                % (C.CLASS_NAMES[k], pc["thresholds"]["n_top"], MIN_CAL))
        # a class with no training sample (the caller fitted without it; cmd_fit
        # warns when it has no OtherPlant sample) has no recall to report
        elif pc["train"] and not pc["thresholds"]["target_reachable"]:
            log("WARNING: %s: cv top-1 %.3f is too low for %.0f %% recall; recall %.3f"
                % (C.CLASS_NAMES[k], pc["cv_top1"] or 0.0, 100 * RECALL,
                   pc["thresholds"]["recall_cv"] or 0.0))
    spm = yh < OTHER
    sp_rec = [per_class[C.CLASS_NAMES[k]]["cv_confirmed"] for k in range(N_SPECIES)]
    sp_rec = [x for x in sp_rec if x is not None]
    info = {"folds": nf, "recall_target": RECALL, "max_cond_pass": MAX_COND_PASS,
            "threshold_rule": "joint: tau_p and sigma at one rank of the held-out top-1 true-k "
                              "crops, the highest with verified rate >= recall_target",
            "n_species_crops": int(len(Xc)), "n_other_crops": int(len(Xo)),
            "cv_top1_species": _f((j[spm] == yh[spm]).mean()) if spm.any() else None,
            "cv_recall_species_min": _f(min(sp_rec)) if sp_rec else None,
            "cv_recall_species_mean": _f(np.mean(sp_rec)) if sp_rec else None,
            "per_class": per_class}
    probe = _train_probe(X_all, y_all)
    ver = Verifier(probe, _prototypes(X_all, y_all), tau_p, sigma, info)
    n_c = len(Xc)
    ver.oof_other = {"P": Ph[n_c:], "cos": Ch[n_c:], "usable": usable[n_c:]}
    return ver


def _species_training(crops, X):
    """Indices, normalised features, labels and sessions of the usable
    train_core crops."""
    idx = crops.where("core")
    idx = idx[np.isfinite(np.asarray(X[idx], dtype=np.float32)).all(axis=1)]
    return idx, _norm(X[idx]), crops.label[idx], np.array([crops.group[i] for i in idx])


def _other_sample(crops, X, n_other, seed):
    """Seeded sample of pool OtherPlant crops whose source named a plant that
    is not a cwd12 species nor a relative of one (other_name_status). Returns
    (indices, eligible name counts, excluded name counts, excluded name counts
    by reason)."""
    status = {}
    elig, names_in, names_out = [], collections.Counter(), collections.Counter()
    by_reason = collections.defaultdict(collections.Counter)
    for i in crops.where("pool"):
        if crops.label[i] != OTHER:
            continue
        nm = crops.src_name[i]
        if nm not in status:
            status[nm] = other_name_status(nm)
        why = status[nm]
        if why is None:
            elig.append(i)
            names_in[nm] += 1
        else:
            names_out[nm or "(no name)"] += 1
            by_reason[why][nm or "(no name)"] += 1
    elig = np.array(elig, dtype=np.int64)
    if len(elig):
        finite = np.isfinite(np.asarray(X[elig], dtype=np.float32)).all(axis=1)
        elig = elig[finite]
    if len(elig) > n_other:
        rng = np.random.default_rng(seed)
        elig = np.sort(rng.choice(elig, n_other, replace=False))
    return elig, names_in, names_out, by_reason


def cmd_fit(args):
    t0 = time.time()
    crops = Crops()
    fresh = check_fresh(crops)
    X, emb = load_embeddings(crops, args.nshards)
    core_idx, Xc, yc, gc = _species_training(crops, X)
    if not len(core_idx):
        raise VerifyError("no train_core crops with features")
    seed = C.stable_int("inc/step1/otherplant")
    other_idx, names_in, names_out, by_reason = _other_sample(crops, X, int(args.n_other), seed)
    Xo = _norm(X[other_idx]) if len(other_idx) else None
    go = np.array([crops.source[i] for i in other_idx]) if len(other_idx) else None
    if not len(other_idx):
        log("WARNING: no named OtherPlant crops in the pool: the probe has 12 classes and "
            "never predicts OtherPlant")
    log("fit: %d species crops from %d sessions, %d OtherPlant crops from %d sources"
        % (len(core_idx), len(set(gc)), len(other_idx), len(set(go)) if go is not None else 0))
    ver = fit_verifier(Xc, yc, gc, Xo, go, seed=seed)
    ver.info.update({"crops_sha256": crops.sha, "inputs": fresh["inputs"], "embeddings": emb,
                     "seconds": round(time.time() - t0, 1), "built_utc": _utc()})
    oof = ver.oof_other
    ver.save(VERIFIER_DIR, {"crops_sha256": crops.sha, "other_ids": other_idx,
                            "core_ids": core_idx, "other_oof_P": oof["P"],
                            "other_oof_cos": oof["cos"], "other_oof_usable": oof["usable"]})
    fit_info = dict(ver.info, other_sample={
        "requested": int(args.n_other), "taken": int(len(other_idx)), "seed": seed,
        "sources": _counter(collections.Counter(go.tolist())) if go is not None else {},
        "eligible_names": dict(names_in.most_common(200)),
        "excluded_names": dict(names_out.most_common(200)),
        "excluded_names_by_reason": {r: dict(c.most_common(200)) for r, c in sorted(by_reason.items())}})
    _write_json(VERIFIER_DIR / "fit_info.json", fit_info)
    log("fit: cv top-1 on species %.4f, cv recall per species min %.4f mean %.4f; thresholds -> %s "
        "(%.0fs)" % (ver.info["cv_top1_species"] or float("nan"),
                     ver.info["cv_recall_species_min"] or float("nan"),
                     ver.info["cv_recall_species_mean"] or float("nan"),
                     VERIFIER_DIR / "thresholds.json", time.time() - t0))
    return fit_info


def _load_fitted(crops, nshards=None):
    """(verifier, X, embedding info) for calibrate and admit: crops.csv is
    current (check_fresh), the verifier was fitted on it, and the embeddings
    are the shard set the verifier was fitted on (by default that one; a
    different --nshards, embedder or dimension is refused)."""
    fresh = check_fresh(crops)
    ver = Verifier.load()
    if ver.meta.get("crops_sha256") != crops.sha:
        raise VerifyError("the verifier was fitted on another crops.csv; rerun `verify fit`")
    if (ver.info.get("inputs") or {}) != fresh["inputs"]:
        raise VerifyError("the verifier was fitted on other pool / train_core files; rerun `verify fit`")
    fit_emb = ver.info.get("embeddings") or {}
    X, emb = load_embeddings(crops, nshards or fit_emb.get("nshards"))
    diff = {k: (fit_emb.get(k), emb.get(k)) for k in ("embedder", "dim", "nshards")
            if fit_emb.get(k) != emb.get(k)}
    if diff:
        raise VerifyError("these embeddings are not the ones the verifier was fitted on "
                          "(fitted, now): %s; rerun `verify fit` or pass its --nshards" % diff)
    return ver, X, emb


# ---------------------------------------------------------------- calibrate
def _metrics(correct, verdict):
    """Verifier quality on boxes whose label correctness is known."""
    correct = np.asarray(correct, dtype=bool)
    verdict = np.asarray(verdict, dtype=object)
    judged = np.isin(verdict, (VERIFIED, CONFLICT, UNKNOWN, OTHER_OK))
    conf, ver = verdict == CONFLICT, verdict == VERIFIED
    wrong = ~correct

    def rate(a, b):
        b = int(b.sum())
        return round(float(a.sum()) / b, 4) if b else None

    return {"boxes": int(len(correct)), "judged": int(judged.sum()),
            "label_correct_rate": rate(correct, np.ones_like(correct)),
            "wrong_judged": int((wrong & judged).sum()), "correct_judged": int((correct & judged).sum()),
            "conflict_recall_on_wrong": rate(conf & wrong, wrong & judged),
            "false_conflict_rate_on_correct": rate(conf & correct, correct & judged),
            "verified_precision": rate(ver & correct, ver),
            "verified_recall_on_correct": rate(ver & correct, correct & judged),
            "verdicts": _counter(collections.Counter(verdict.tolist()))}


def _match_box(b, ref, tol=MATCH_TOL):
    """The ref box whose (cx, cy, w, h) all lie within tol of b's (closest)."""
    best, best_d = None, None
    for t in ref:
        d = max(abs(x - y) for x, y in zip(b[1:5], t[1:5]))
        if d < tol and (best_d is None or d < best_d):
            best, best_d = t, d
    return best


def _swap_labels(yc, te_idx, f):
    """Calibrate (b): the seeded swap of fold f. Returns (swapped positions,
    their new labels); every new label differs from the true one."""
    rng = np.random.default_rng(C.stable_int("inc/step1/swaps/%d" % f))
    n_sw = int(round(SWAP_FRAC * len(te_idx)))
    if not n_sw:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    sw = rng.choice(te_idx, n_sw, replace=False)
    return sw, (yc[sw] + rng.integers(1, N_SPECIES, size=n_sw)) % N_SPECIES


def cmd_calibrate(args):
    """calibration.json: (a) cwd12 copies with known truth, under the current
    and the pre-v3.60.0 join; (b) seeded swaps."""
    t0 = time.time()
    crops = Crops()
    final, X, emb = _load_fitted(crops, args.nshards)
    other_idx = final.arrays.get("other_ids", np.zeros(0, np.int64)).astype(np.int64)
    core_idx, Xc, yc, gc = _species_training(crops, X)
    Xo = _norm(X[other_idx]) if len(other_idx) else None
    go = np.array([crops.source[i] for i in other_idx]) if len(other_idx) else None

    # outer folds by session; each fold's verifier never saw its sessions
    outer = _assign_folds(gc, N_FOLDS, 0, random_ok=False)
    nf = int(outer.max()) + 1
    fold_of_session = {g: int(f) for g, f in zip(gc, outer)}
    models, swap_folds = [], []
    all_correct, all_verdict = [], []
    true_sp, new_sp = collections.Counter(), collections.Counter()
    for f in range(nf):
        tr, te = outer != f, outer == f
        ver_f = fit_verifier(Xc[tr], yc[tr], gc[tr], Xo, go, seed=f)
        models.append(ver_f)
        te_idx = np.flatnonzero(te)
        sw, new = _swap_labels(yc, te_idx, f)
        lab = yc.copy()
        lab[sw] = new
        v, _j, _pj, _cj = ver_f.judge(lab[te], Xc[te])
        correct = lab[te] == yc[te]
        m = _metrics(correct, v)
        pairs = sorted(zip(core_idx[sw].tolist(), new.tolist()))
        ts = collections.Counter(C.CLASS_NAMES[c] for c in yc[sw].tolist())
        ns = collections.Counter(C.CLASS_NAMES[c] for c in new.tolist())
        true_sp.update(ts)
        new_sp.update(ns)
        m.update({"fold": f, "sessions": sorted(set(gc[te].tolist())), "swapped": int((~correct).sum()),
                  "swapped_true_species": _counter(ts), "swapped_to_species": _counter(ns),
                  "swap_sha256": C.sha256_text(json.dumps(pairs))})
        swap_folds.append(m)
        all_correct.append(correct)
        all_verdict.append(v)
        log("calibrate (b) fold %d/%d: conflict recall on swapped %s, false conflict %s, "
            "verified precision %s" % (f + 1, nf, m["conflict_recall_on_wrong"],
                                        m["false_conflict_rate_on_correct"], m["verified_precision"]))
    swaps = {"frac": SWAP_FRAC, "folds": swap_folds,
             "swapped_true_species": _counter(true_sp), "swapped_to_species": _counter(new_sp),
             "overall": _metrics(np.concatenate(all_correct), np.concatenate(all_verdict))}

    copies = _calibrate_copies(crops, X, final, models, fold_of_session)
    cal = {"built_utc": _utc(), "seconds": round(time.time() - t0, 1), "crops_sha256": crops.sha,
           "embeddings": emb, "thresholds": {"tau_p": [_f(x) for x in final.tau_p],
                                             "sigma": [_f(x) for x in final.sigma],
                                             "class_names": C.CLASS_NAMES},
           "cwd12_copies": copies, "swaps": swaps,
           "notes": ["(a) truth = the train_core box at the same coordinates (tol %.2f); the verdicts "
                     "come from verifiers fitted without the original's capture session. "
                     "current_join labels the copies as the pool labels everything now; "
                     "old_wrong_joins labels the same boxes as mega_trainer did before v3.60.0 "
                     "(legacy-label string match, cottonweed_holdout through its four-name list), "
                     "and counts the boxes that join deleted. final_model_optimistic uses the "
                     "verifier admit uses, which was trained on the original photographs." % MATCH_TOL,
                     "(b) %d%% of each held-out fold's species labels reassigned to a random other "
                     "species (seeded); the fold's verifier is fitted, thresholds included, on the "
                     "other folds only." % int(100 * SWAP_FRAC),
                     "verified_precision = share of verified boxes whose label is correct; "
                     "conflict_recall_on_wrong and false_conflict_rate_on_correct are over boxes "
                     "with a verdict (not small, not failed)."]}
    _write_json(CALIBRATION, cal)
    keys = ("boxes", "label_correct_rate", "conflict_recall_on_wrong", "verified_precision")
    log("calibrate: (a) current join %s; old wrong joins %s, %d deleted; (b) overall %s (%.0fs)"
        % ({k: copies["current_join"]["overall"].get(k) for k in keys},
           {k: copies["old_wrong_joins"]["overall"].get(k) for k in keys},
           sum(copies["old_wrong_joins"]["deleted_boxes"].values()),
           {k: swaps["overall"].get(k) for k in ("conflict_recall_on_wrong",
                                                 "false_conflict_rate_on_correct",
                                                 "verified_precision")}, time.time() - t0))
    return cal


def _calibrate_copies(crops, X, final, models, fold_of_session):
    """Calibrate (a). Every box of every cwd12 copy is box-matched to its
    train_core original (the truth) and judged under two labellings: the
    current join (the label file) and the pre-v3.60.0 join (old_join, where a
    deleted box has no label and is only counted)."""
    copies = _read_jsonl(COPIES) if COPIES.exists() else []
    crop_of = {(crops.key[i], int(crops.box[i])): int(i) for i in crops.where("copy")}
    left_out = read_skipped("copy")
    recs = []            # (slug, truth, current label, old label or DELETED, crop id or -1, fold or -1, why)
    unmatched = collections.Counter()
    no_fold = 0
    for r in copies:
        mine = C.read_yolo(r["label"])
        old = r.get("old_join")
        if old is None or len(old) != len(mine):
            raise VerifyError("%s: copy %s has no pre-v3.60.0 labels for its %d boxes; rerun `verify pool`"
                              % (COPIES.name, r["key"], len(mine)))
        ref = C.read_yolo(r["train_core_label"])
        f = fold_of_session.get(r["train_core_session"])
        for k, b in enumerate(mine):
            t = _match_box(b, ref)
            if t is None:
                unmatched[r["source"]] += 1
                continue
            if f is None:
                no_fold += 1
            i = crop_of.get((r["key"], k))
            why = left_out.get((r["key"], k))
            if i is None and why is None:
                raise VerifyError("copy box %s/%d has no crop row and was not left out by `verify crops`"
                                  % (r["key"], k))
            recs.append((r["source"], int(t[0]), int(b[0]), old[k], -1 if i is None else i,
                         -1 if f is None else f, why))
    slugs = np.array([x[0] for x in recs], dtype=object)
    truth = np.array([x[1] for x in recs], dtype=np.int64)
    ci = np.array([x[4] for x in recs], dtype=np.int64)
    folds = np.array([x[5] for x in recs], dtype=np.int64)
    no_size = np.array([x[6] == "no_size" for x in recs], dtype=bool)

    def judge(labels, m):
        """(held-out verdicts, final-model verdicts) of the recs in mask m."""
        n = int(m.sum())
        v_held = np.full(n, SMALL, dtype=object)
        v_final = np.full(n, SMALL, dtype=object)
        v_held[no_size[m]] = FAILED
        v_final[no_size[m]] = FAILED
        c, fo, lab = ci[m], folds[m], labels[m]
        has = c >= 0
        if has.any():
            v_final[has] = final.judge_features(lab[has], X[c[has]])[0]
            for f, ver in enumerate(models):
                mf = has & (fo == f)
                if mf.any():
                    v_held[mf] = ver.judge_features(lab[mf], X[c[mf]])[0]
            v_held[has & (fo < 0)] = NO_HELD_OUT
        return v_held, v_final

    def block(labels, m):
        correct = labels[m] == truth[m]
        v_held, v_final = judge(labels, m)
        sl = slugs[m]
        out = {"overall": _metrics(correct, v_held), "per_slug": {},
               "final_model_optimistic": {"overall": _metrics(correct, v_final), "per_slug": {}}}
        for s in sorted(set(sl.tolist())):
            ms = sl == s
            out["per_slug"][s] = _metrics(correct[ms], v_held[ms])
            out["final_model_optimistic"]["per_slug"][s] = _metrics(correct[ms], v_final[ms])
        return out

    everything = np.ones(len(recs), dtype=bool)
    cur = np.array([x[2] for x in recs], dtype=np.int64)
    kept = np.array([x[3] is not DELETED for x in recs], dtype=bool)
    old = np.array([OTHER if x[3] is DELETED else int(x[3]) for x in recs], dtype=np.int64)
    old_block = block(old, kept)
    rwd = {}
    for s in sorted(set(slugs.tolist())):
        ms = slugs == s
        rwd[s] = [int((ms & kept & (old == truth)).sum()), int((ms & kept & (old != truth)).sum()),
                  int((ms & ~kept).sum())]
    old_block.update({
        "deleted_boxes": {s: v[2] for s, v in rwd.items()},
        "right_wrong_deleted": rwd,
        "right_wrong_deleted_total": [sum(v[i] for v in rwd.values()) for i in range(3)]})
    per_slug_images = collections.Counter(r["source"] for r in copies)
    out = {"copies": len(copies), "copies_per_slug": _counter(per_slug_images),
           "calibration_only_slugs": sorted({r["source"] for r in copies if r.get("calibration_only")}),
           "matched_boxes": len(recs), "unmatched_boxes": _counter(unmatched),
           "matched_without_fold": no_fold,
           "current_join": block(cur, everything), "old_wrong_joins": old_block}
    if not copies:
        out["note"] = "no cwd12 copies in the pool: (a) measured nothing"
    return out


# ------------------------------------------------------------------- admit
def cmd_admit(args):
    """verified.jsonl (admitted images), conflicts.csv + crop sheet, admit_summary.json."""
    t0 = time.time()
    crops = Crops()
    ver, X, emb = _load_fitted(crops, args.nshards)
    pool = {r["key"]: r for r in C.read_manifest(POOL)}
    meta = {m["key"]: m for m in _read_jsonl(POOL_META)}
    pidx = crops.where("pool")
    pos = {int(c): n for n, c in enumerate(pidx)}
    for i in pidx:
        m = meta.get(crops.key[i])
        b = int(crops.box[i])
        if (m is None or b >= len(m["boxes"]) or int(m["boxes"][b][0]) != int(crops.label[i])
                or crops.image[i] != pool[m["key"]]["image"]):
            raise VerifyError("pool crop %d (%s/%d) does not match pool_meta.jsonl: crops.csv was "
                              "made from another pool" % (i, crops.key[i], b))
    log("admit: judging %d pool crops" % len(pidx))
    v, j, pj, cj = ver.judge_features(crops.label[pidx], X[pidx])

    # the OtherPlant crops the final probe was trained on: out-of-fold verdicts
    other_ids = ver.arrays.get("other_ids", np.zeros(0, np.int64)).astype(np.int64)
    n_oof = 0
    if len(other_ids):
        oP, oC = ver.arrays.get("other_oof_P"), ver.arrays.get("other_oof_cos")
        oU = ver.arrays.get("other_oof_usable")
        if oP is None or oC is None or oU is None or len(oP) != len(other_ids):
            raise VerifyError("the verifier has no out-of-fold predictions for its OtherPlant "
                              "training crops; rerun `verify fit`")
        rows = [pos.get(int(c)) for c in other_ids]
        if any(r is None for r in rows):
            raise VerifyError("an OtherPlant training crop is not a pool crop; rerun `verify fit`")
        rows = np.array(rows, dtype=np.int64)
        vo, jo, pjo, cjo = verdicts(np.full(len(rows), OTHER), oP, oC, ver.tau_p, ver.sigma)
        vo[~oU.astype(bool)] = UNKNOWN          # no fold model had OtherPlant to predict with
        v[rows], j[rows], pj[rows], cj[rows] = vo, jo, pjo, cjo
        n_oof = len(rows)
    lookup = _box_verdict_lookup(crops, "pool", lambda i: v[pos[i]])

    img_verdict = {}
    by_slug = collections.defaultdict(lambda: {"images": collections.Counter(),
                                               "boxes": collections.Counter()})
    by_species = collections.defaultdict(lambda: {"images": collections.Counter(),
                                                  "boxes": collections.Counter()})
    conflicts = []
    blocked_small = 0
    small_boxes = 0
    for key in sorted(pool):
        m = meta[key]
        labels = [int(b[0]) for b in m["boxes"]]
        bv = []
        for k, l in enumerate(labels):
            x, i = lookup(key, k)
            bv.append(x)
            small_boxes += x == SMALL
            if x == CONFLICT:
                n = pos[i]
                conflicts.append({"source": m["source"], "image": pool[key]["image"], "key": key,
                                  "box": k, "label": l, "label_name": C.CLASS_NAMES[l],
                                  "pred": int(j[n]), "pred_name": C.CLASS_NAMES[int(j[n])],
                                  "p": round(float(pj[n]), 4),
                                  "cosine": round(float(cj[n]), 4) if np.isfinite(cj[n]) else "",
                                  "crop_id": int(i)})
        iv = image_verdict(labels, bv)
        img_verdict[key] = iv
        if iv == UNKNOWN and all((x == VERIFIED or x == SMALL) if l < OTHER else x in (OTHER_OK, SMALL)
                                 for l, x in zip(labels, bv)):
            blocked_small += 1
        s = by_slug[m["source"]]
        s["images"][iv] += 1
        s["boxes"].update(bv)
        for l in set(labels):
            by_species[C.CLASS_NAMES[l]]["images"][iv] += 1
        for l, x in zip(labels, bv):
            by_species[C.CLASS_NAMES[l]]["boxes"][x] += 1

    admitted = [pool[k] for k in sorted(pool) if img_verdict[k] == ADMITTED]
    vsha = C.write_manifest(VERIFIED_MANIFEST, admitted)

    sheet_items = _sheet_selection(conflicts, int(args.sheet_max))
    for n, c in enumerate(sheet_items):
        c["sheet_index"] = n
    fields = ["source", "image", "key", "box", "label", "label_name", "pred", "pred_name", "p",
              "cosine", "crop_id", "sheet_index"]
    tmp = CONFLICTS.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for c in conflicts:
            wr.writerow({k: c.get(k, "") for k in fields})
    os.replace(tmp, CONFLICTS)
    sheet = _contact_sheet(sheet_items, crops, SHEET) if sheet_items else None

    _save_npz(POOL_VERDICTS, {"crops_sha256": crops.sha, "verdict_codes": list(VERDICT_CODES)},
              crop_id=pidx, verdict=np.array([VERDICT_CODES.index(x) for x in v], dtype=np.int8),
              pred=j, p=pj, cosine=cj)

    img_tot = collections.Counter(img_verdict.values())
    box_tot = collections.Counter(v.tolist())
    summary = {"built_utc": _utc(), "seconds": round(time.time() - t0, 1),
               "crops_sha256": crops.sha, "embeddings": emb, "verified_sha256": vsha,
               "inputs": ver.info.get("inputs"),
               "rule": "admitted = every species box verified and every OtherPlant box other_ok "
                       "(or too small to embed); conflict = any box a conflict; else unknown",
               "images": _counter(img_tot), "boxes": _counter(box_tot),
               "boxes_small_not_embedded": int(small_boxes),
               "images_unknown_only_for_small_species_boxes": blocked_small,
               "admitted_boxes_per_class": _box_counts(meta[r["key"]] for r in admitted),
               "otherplant_training_boxes": int(len(other_ids)),
               "otherplant_training_boxes_judged_out_of_fold": int(n_oof),
               "conflict_boxes": len(conflicts), "sheet": str(sheet) if sheet else None,
               "sheet_boxes": len(sheet_items),
               "per_slug": {s: {"images": _counter(d["images"]), "boxes": _counter(d["boxes"])}
                            for s, d in sorted(by_slug.items())},
               "per_species": {s: {"images": _counter(d["images"]), "boxes": _counter(d["boxes"])}
                               for s, d in sorted(by_species.items())}}
    _write_json(ADMIT_SUMMARY, summary)
    log("admit: images %s; %d conflict boxes -> %s (%.0fs)"
        % (summary["images"], len(conflicts), CONFLICTS, time.time() - t0))
    return summary


def _box_counts(metas):
    counts = collections.Counter(int(b[0]) for m in metas for b in m["boxes"])
    return {C.CLASS_NAMES[c]: int(counts.get(c, 0)) for c in range(C.NC)}


def _sheet_selection(conflicts, limit):
    """Up to `limit` conflicts, round-robin over sources, most confident first."""
    by = collections.defaultdict(list)
    for c in sorted(conflicts, key=lambda c: (-c["p"], c["key"], c["box"])):
        by[c["source"]].append(c)
    queues = [by[s] for s in sorted(by)]
    out = []
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


def _contact_sheet(items, crops, path, tile=128, cols=10, caption=30):
    """PNG of conflict crops: '#n', the label and the probe's call under each."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from ..semisup_labeler import _cut
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + caption)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    by_image = collections.defaultdict(list)
    for n, c in enumerate(items):
        by_image[c["image"]].append((n, c))
    for image, lst in by_image.items():
        try:
            with Image.open(image) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
        except Exception:
            im = None
        for n, c in lst:
            x, y = (n % cols) * tile, (n // cols) * (tile + caption)
            if im is not None:
                r = crops.row(c["crop_id"])
                r.update(W=im.size[0], H=im.size[1])
                sheet.paste(_cut(im, r).resize((tile, tile)), (x, y))
            else:
                draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=(200, 200, 200))
            draw.text((x + 2, y + tile + 1), "#%d L:%s" % (n, c["label_name"][:13]), fill=(0, 0, 0),
                      font=font)
            draw.text((x + 2, y + tile + 14), "P:%s %.2f" % (c["pred_name"][:12], c["p"]),
                      fill=(170, 0, 0), font=font)
    tmp = Path(path).with_name(Path(path).name + ".tmp.png")
    sheet.save(tmp)
    os.replace(tmp, path)
    return Path(path)


# --------------------------------------------------------------------- CLI
def cmd_all(args, embedder=None):
    cmd_pool(args)
    cmd_crops(args)
    a = argparse.Namespace(**vars(args))
    a.shard = None
    cmd_embed(a, embedder=embedder)
    cmd_fit(args)
    cmd_calibrate(args)
    return cmd_admit(args)


COMMANDS = {"pool": cmd_pool, "crops": cmd_crops, "embed": cmd_embed, "fit": cmd_fit,
            "calibrate": cmd_calibrate, "admit": cmd_admit, "all": cmd_all}


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc.verify",
                                 description="INC Step 1 box verifier (labeler Phase B)")
    ap.add_argument("cmd", choices=list(COMMANDS))
    ap.add_argument("--shard", type=int, default=None, help="embed: this shard only (needs --nshards)")
    ap.add_argument("--nshards", type=int, default=None,
                    help="embed: number of shards (default 1); later stages: which shard set to read")
    ap.add_argument("--procs", type=int, default=PROCS, help="worker processes for hashing / cropping")
    ap.add_argument("--batch", type=int, default=BATCH, help="embed: crops per forward pass")
    ap.add_argument("--chunk-images", type=int, default=CHUNK_IMAGES,
                    help="embed: images per resumable chunk")
    ap.add_argument("--n-other", type=int, default=N_OTHER, help="fit: OtherPlant sample size")
    ap.add_argument("--sheet-max", type=int, default=SHEET_MAX, help="admit: crops on the sheet")
    ap.add_argument("--no-amp", action="store_true", help="embed: fp32 on CUDA")
    ap.add_argument("--force", action="store_true", help="embed: recompute finished shards")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.shard is not None:
        if a.nshards is None:
            ap.error("--shard needs --nshards")
        if not 0 <= a.shard < a.nshards:
            ap.error("--shard must be in 0..%d" % (a.nshards - 1))
        if a.cmd not in ("embed",):
            ap.error("--shard applies to embed only")
    if a.nshards is not None and a.nshards < 1:
        ap.error("--nshards must be >= 1")
    return a


def main(argv=None):
    args = parse_args(argv)
    try:
        COMMANDS[args.cmd](args)
    except VerifyError as e:
        log("FAILED: %s" % e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
