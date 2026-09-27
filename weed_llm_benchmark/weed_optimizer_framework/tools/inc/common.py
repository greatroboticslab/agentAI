"""Shared definitions for the INC protocol (docs/INCREMENTAL_PROTOCOL.md).

Everything another INC module needs to agree on lives here: where results go,
the class space, the manifest format, the lock file and the never-train guard.

Class space: 13 ids. 0-11 are the cwd12 species in cwd12 id order (the order of
cottonweeddet12's own label files, cwd12_species.CWD12_SPECIES); 12 is
OtherPlant, for any plant that is not one of them. Trainer-slot ids
(cwd12_species.CWD12_ID_TO_SLOT) are never used inside INC.

A manifest is a JSON-lines file, one image per line:
    {"image": abs path, "label": abs path of a YOLO .txt in the INC class space,
     "sha256": image sha256, "label_sha256": label sha256,
     "source": dataset / split name, "session": capture session or "",
     "key": unique short name used when the image is materialised}
Label files referenced by a manifest are never edited in place; a relabelled
copy (e.g. species-joined, or a planted corruption) is written to a new file.

No third-party imports at module load (numpy/PIL are imported where used), so
the gate, driver and tests can import this on any machine.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..cwd12_species import CWD12_SPECIES
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NearHashIndex

REPO = Path(os.environ.get("REPO", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))
INC_DIR = Path(os.environ.get("INC_DIR", str(REPO / "results" / "framework" / "inc")))
SPLITS_VERSION = "v1"
SPLITS_DIR = INC_DIR / "splits" / SPLITS_VERSION
EXAMS_DIR = INC_DIR / "exams" / SPLITS_VERSION      # materialised, read-only exam dirs
LOCK_PATH = SPLITS_DIR / "LOCK.json"
NEVER_TRAIN_INDEX = SPLITS_DIR / "nevertrain_dhash.json"

OTHER_PLANT = 12
CLASS_NAMES = list(CWD12_SPECIES) + ["OtherPlant"]
NC = len(CLASS_NAMES)
assert NC == 13 and CLASS_NAMES[5] == "Ragweed" and CLASS_NAMES[0] == "Waterhemp"

# Splits that must never be trained on; every image in them goes into the
# never-train index at HOLDOUT_NEAR_DUP_BITS. "test" is the historical sealed
# cwd12 holdout (valid + test, 1,977 images).
EVAL_SPLITS = ("dev", "test", "ood22", "ood23", "imageweeds")
TRAIN_SPLITS = ("train_core",)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------------ hashing
def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_int(text, mod=2 ** 31 - 1):
    """A seed derived from a string, identical on every machine and run."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % mod


def dhash(path):
    """The project's one dHash (mega_trainer._dhash), so INC guards and the merge
    guard agree on what 'the same photograph' means."""
    from ..mega_trainer import _dhash
    return _dhash(path)


# ---------------------------------------------------------------- manifests
MANIFEST_KEYS = ("image", "label", "sha256", "label_sha256", "source", "session", "key")


def write_manifest(path, rows):
    """Write rows (dicts with MANIFEST_KEYS) as JSON lines, sorted by key, and
    return the manifest file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate manifest keys in %s" % path)
    for r in rows:
        missing = [k for k in MANIFEST_KEYS if k not in r]
        if missing:
            raise ValueError("manifest row missing %s: %r" % (missing, r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in sorted(rows, key=lambda r: r["key"]):
            fh.write(json.dumps({k: r[k] for k in MANIFEST_KEYS}, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return sha256_file(path)


def read_manifest(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_path(split):
    return SPLITS_DIR / ("%s.jsonl" % split)


def read_yolo(path):
    """[(cls, cx, cy, w, h)] from a YOLO label file; [] if missing/empty."""
    out = []
    try:
        with open(path) as fh:
            for ln in fh:
                t = ln.split()
                if len(t) >= 5:
                    out.append((int(float(t[0])),) + tuple(float(x) for x in t[1:5]))
    except OSError:
        pass
    return out


def write_yolo(path, boxes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for b in boxes:
            c = int(b[0])
            if not 0 <= c < NC:
                raise ValueError("class id %d outside the INC class space (0..%d)" % (c, NC - 1))
            fh.write("%d %.6f %.6f %.6f %.6f\n" % (c, b[1], b[2], b[3], b[4]))


def class_counts(rows):
    """Boxes per INC class over a manifest's rows."""
    counts = [0] * NC
    for r in rows:
        for b in read_yolo(r["label"]):
            counts[b[0]] += 1
    return {CLASS_NAMES[i]: counts[i] for i in range(NC)}


# ------------------------------------------------------------ materialising
def materialise(rows, out_dir, data_yaml_split="val"):
    """Build a YOLO dataset dir from manifest rows: images/ holds symlinks named
    <key><ext>, labels/ holds copies of the label files. Writes data.yaml with
    both train and val pointing at this dir (the caller chooses how to use it).
    Returns the data.yaml path. out_dir must not exist or must be empty."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to materialise into non-empty %s" % out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    for r in rows:
        ext = os.path.splitext(r["image"])[1].lower() or ".jpg"
        os.symlink(r["image"], out_dir / "images" / (r["key"] + ext))
        with open(r["label"]) as src, open(out_dir / "labels" / (r["key"] + ".txt"), "w") as dst:
            dst.write(src.read())
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as fh:
        fh.write("path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n" % (out_dir, NC))
        for n in CLASS_NAMES:
            fh.write("  - %s\n" % n)
    return yaml_path


# ------------------------------------------------------------------- lock
def read_lock():
    with open(LOCK_PATH) as fh:
        return json.load(fh)


def verify_manifest_against_lock(split, lock=None):
    """Raise unless the split's manifest file hashes to what LOCK.json recorded."""
    lock = lock or read_lock()
    want = lock["manifests"].get(split)
    if want is None:
        raise RuntimeError("split %r is not in LOCK.json" % split)
    got = sha256_file(manifest_path(split))
    if got != want:
        raise RuntimeError("manifest %s changed since it was locked (%s != %s)"
                           % (split, got[:12], want[:12]))
    return got


# ------------------------------------------------------- never-train guard
class NeverTrainGuard:
    """Every dev / test / exam image, by dHash, at HOLDOUT_NEAR_DUP_BITS.

    check(paths) returns [(path, split, image, bits)] for every path within
    range of an evaluation image; assert_trainable raises if there is any, and
    also if an image cannot be hashed (fail closed: an image we cannot compare
    is an image we cannot clear)."""

    def __init__(self, entries):
        self.index = NearHashIndex()
        self.n = 0
        for h, split, image in entries:
            self.index.add(int(h), (split, image), max_bits=HOLDOUT_NEAR_DUP_BITS)
            self.n += 1

    @classmethod
    def load(cls, path=None):
        path = Path(path or NEVER_TRAIN_INDEX)
        with open(path) as fh:
            data = json.load(fh)
        guard = cls(data["entries"])
        if guard.n < data.get("min_expected", 0):
            raise RuntimeError("never-train index holds %d images, expected >= %d"
                               % (guard.n, data["min_expected"]))
        return guard

    def check(self, paths, hash_fn=None):
        hash_fn = hash_fn or dhash
        hits, unhashable = [], []
        for p in paths:
            h = hash_fn(p)
            if h is None:
                unhashable.append(str(p))
                continue
            m = self.index.find(h)
            if m is not None:
                (split, image), bits = m
                hits.append((str(p), split, image, bits))
        return hits, unhashable

    def assert_trainable(self, paths, hash_fn=None):
        hits, unhashable = self.check(paths, hash_fn=hash_fn)
        if hits or unhashable:
            raise RuntimeError(
                "never-train guard: %d image(s) within %d bits of an evaluation image, "
                "%d unhashable; first: %s %s"
                % (len(hits), HOLDOUT_NEAR_DUP_BITS, len(unhashable), hits[:3], unhashable[:3]))
        return True
