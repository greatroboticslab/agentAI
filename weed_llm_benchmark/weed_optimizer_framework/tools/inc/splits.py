"""INC Step 0.1: the splits, the never-train index and LOCK.json.

    python -m weed_optimizer_framework.tools.inc.splits build    # manifests, index, exams
    python -m weed_optimizer_framework.tools.inc.splits lock     # freeze them (--relock)
    python -m weed_optimizer_framework.tools.inc.splits verify   # full re-hash vs LOCK
    python -m weed_optimizer_framework.tools.inc.splits summary

Why each piece exists (docs/INCREMENTAL_PROTOCOL.md, "Splits"):

* dev is made of WHOLE capture sessions of cwd12 train. Frames of one session
  are burst neighbours; a dev that shares sessions with train_core would score
  memorised scenes, and every accept/reject decision is taken on dev.
* train_core also loses every train image within HOLDOUT_NEAR_DUP_BITS dHash
  bits of any dev, test or exam image, so no evaluation photograph is trained
  on under another name.
* An exam keeps only photographs cwd12 does not hold: an exam image within
  HOLDOUT_NEAR_DUP_BITS of any cwd12 image leaves the exam (data2022 shares a
  capture session, 20220129_CanonEOS4000D_EO, with cwd12 train).
* The exam labels (3SeasonWeedDet10 2022/2023, ImageWeeds) are converted into
  the INC class space and written under SPLITS_DIR/labels, never next to the
  source, so a later relabel can never edit what a locked manifest hashed.
  They are staged first and swapped in only after every check has passed, so
  a rebuild that fails leaves the locked label files as they were.
* An exam must not quietly shrink: every dropped image and box is counted
  and capped, and lock compares the images KEPT with the protocol's count.
* Every evaluation image goes into the never-train index; every manifest, the
  index and scorer.py are hashed into LOCK.json, and `verify` re-reads every
  byte against it. An index built while an exam is missing, or whose counts
  lock has not yet accepted, is written so that NeverTrainGuard.load refuses
  it: a partial index must never pass for the whole guard.

The build is deterministic: dev comes from an exact search with a fixed
tie-break (no random numbers, so no dependence on a library's RNG stream), and
every file it writes is sorted. Directories are listed once each, never
rglob'd (Lustre).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from . import common as C
from ..cwd12_species import CWD12_SPECIES, name_key, species_of
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NearHashIndex

ALL_SPLITS = C.TRAIN_SPLITS + C.EVAL_SPLITS
N_SPECIES = len(CWD12_SPECIES)

# The dev search. The protocol (docs/INCREMENTAL_PROTOCOL.md, Splits) asks for
# whole sessions, "about 15%" of the train images and every species >= 30
# boxes in dev. "About 15%" is read here as a 13-18% window in which the
# subset closest to 15% wins. A rule that each species keeps a share of its
# boxes in train is NOT in the protocol: it exists (--dev-min-keep) but is off
# by default. On cwd12 the defaults give 8 sessions, 617 of 3,671 images
# (16.8%), every species >= 31 boxes.
DEV_TARGET_FRAC = 0.15
DEV_FRAC_RANGE = (0.13, 0.18)
DEV_MIN_BOXES = 30
DEV_MIN_KEEP_FRAC = 0.0
DEV_MAX_NODES = 50_000_000
# Tried, and reported, when the requested constraints admit no dev:
# (min boxes per species, min share kept in train, cap rare species). The
# first is the protocol as written; the rest give up dev boxes, or add a keep
# rule for whoever asked for one.
DEV_RELAXATIONS = ((DEV_MIN_BOXES, 0.0, False), (25, 0.0, False), (20, 0.0, False),
                   (30, 0.70, True), (25, 0.70, True), (20, 0.70, False),
                   (30, 0.65, True), (30, 0.60, False))

# Names in the exams that are plants but no cwd12 species.
DEFAULT_OTHER_PLANT = ("Lambsquarters",)

# ImageWeeds label files ship without a yaml; this order is from the HF
# dataset card (crossdataset_eval.py, checked 2026-08-26), not from the data.
IMAGEWEEDS_CLASSES = ["horseweed", "kochia", "corn", "ragweed", "redrootpigweed"]
IMAGEWEEDS_SLUG = "project_agml__imageweeds_weed_detection"

# Sizes stated in the protocol, compared with the images each exam KEEPS. A
# different count is what a download still being unpacked, or a conversion
# that dropped images, looks like, so it blocks lock unless accepted.
EXPECTED_IMAGES = {"test": 1977, "ood22": 1948, "ood23": 1748, "imageweeds": 3208}

# Per exam: images dropped for any reason (no annotation, annotated on another
# frame, every box degenerate) plus images kept as background only because
# they have no label file may number at most
# max(MAX_DROPS_ALWAYS_OK, max_drop_share * images found); degenerate boxes
# likewise against boxes read. More means the layout is not what this
# converter assumes, or the download is incomplete, and the build stops instead
# of quietly changing the exam. --max-drop-share raises the share once the
# drops have been looked at.
MAX_DROP_SHARE = 0.01
MAX_DROPS_ALWAYS_OK = 5

STAGING_NAME = ".staging"

PROGRESS_EVERY = 500
SUMMARY_NAME = "summary.json"
LOCK_LOG_NAME = "lock_log.jsonl"


class SplitError(RuntimeError):
    """A condition under which the splits must not be built or locked."""


def log(msg):
    print("[inc.splits] %s" % msg, flush=True)


# ----------------------------------------------------------------- sources
def cwd12_dir():
    return C.REPO / "downloads" / "cottonweeddet12"


def three_season_dir(split):
    return C.REPO / "downloads" / "3seasonweeddet10" / {"ood22": "data2022", "ood23": "data2023"}[split]


def imageweeds_dir():
    return C.REPO / "datasets" / IMAGEWEEDS_SLUG


def summary_path():
    return C.SPLITS_DIR / SUMMARY_NAME


def converted_label_dir(split):
    return C.SPLITS_DIR / "labels" / split


def staging_dir():
    """Where build writes converted labels until every check has passed."""
    return C.SPLITS_DIR / STAGING_NAME


def other_plant_names(extra=()):
    """DEFAULT_OTHER_PLANT plus extra names, once each: --other-plant adds to
    the default, so naming a new class never un-declares Lambsquarters."""
    out, seen = [], set()
    for n in tuple(DEFAULT_OTHER_PLANT) + tuple(extra or ()):
        if name_key(n) not in seen:
            seen.add(name_key(n))
            out.append(n)
    return tuple(out)


def default_scorer_path():
    return Path(__file__).resolve().with_name("scorer.py")


def session_of(stem):
    """Capture session of a cwd12 image: the stem minus its frame number
    (20210812_iPhoneSE_YL_123 -> 20210812_iPhoneSE_YL)."""
    return re.sub(r"_\d+$", "", stem)


def sanitise(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "x"


def _is_image(name):
    return os.path.splitext(name)[1].lower() in C.IMG_EXTS and not name.startswith(".")


def _walk_files(root):
    """[(dir, sorted file names)] for root and every directory under it, one
    scandir per directory. Hidden entries and __MACOSX (zip residue) skipped."""
    out, stack = [], [str(root)]
    while stack:
        d = stack.pop()
        files, subdirs = [], []
        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith(".") or e.name == "__MACOSX":
                    continue
                (subdirs if e.is_dir() else files).append(e)
        out.append((d, sorted(e.name for e in files)))
        stack.extend(sorted((e.path for e in subdirs), reverse=True))
    return out


def _progress(i, n, what):
    if i % PROGRESS_EVERY == 0 or i == n:
        log("  %s %d/%d" % (what, i, n))


# ------------------------------------------------------------- cwd12 splits
def _strict_yolo(path, max_cls, check_range=True):
    """Boxes of a YOLO label file, refusing anything the trainer would drop or
    misread: not 5 columns, a class outside 0..max_cls, a value outside [0, 1]
    (unless check_range is False: a converted exam clips instead)."""
    boxes = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            t = line.split()
            if not t:
                continue
            if len(t) != 5:
                raise SplitError("%s:%d: %d columns, expected 5" % (path, n, len(t)))
            c = float(t[0])
            if c != int(c) or not 0 <= int(c) <= max_cls:
                raise SplitError("%s:%d: class %s outside 0..%d" % (path, n, t[0], max_cls))
            v = [float(x) for x in t[1:]]
            if check_range and (min(v) < 0 or max(v) > 1):
                raise SplitError("%s:%d: coordinates outside [0, 1]: %s" % (path, n, line.strip()))
            boxes.append((int(c),) + tuple(v))
    return boxes


def list_cwd12(split):
    """Rows of one cwd12 split; the label is cwd12's own file (already INC ids
    0-11, checked here)."""
    root = cwd12_dir() / split
    img_dir, lbl_dir = root / "images", root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SplitError("cwd12 %s not found at %s" % (split, root))
    labels = set(os.listdir(lbl_dir))
    rows, unlabelled = [], []
    for name in sorted(os.listdir(img_dir)):
        if not _is_image(name):
            continue
        stem = os.path.splitext(name)[0]
        if stem + ".txt" not in labels:
            unlabelled.append(name)
            continue
        lbl = lbl_dir / (stem + ".txt")
        rows.append({"image": str(img_dir / name), "label": str(lbl),
                     "source": "cottonweeddet12/%s" % split, "session": session_of(stem),
                     "stem": stem, "rel": "%s/images/%s" % (split, name),
                     "boxes": _strict_yolo(lbl, N_SPECIES - 1)})
    if unlabelled:
        raise SplitError("cwd12 %s: %d image(s) without a label file, e.g. %s"
                         % (split, len(unlabelled), unlabelled[:5]))
    return rows


def species_counts(boxes):
    counts = [0] * N_SPECIES
    for b in boxes:
        if b[0] < N_SPECIES:
            counts[b[0]] += 1
    return counts


# ---------------------------------------------------------------- dev search
def _dev_bounds(totals, min_dev_boxes, min_keep_frac, cap_rare):
    """Per-species (floor, upper) on dev boxes. upper keeps min_keep_frac of the
    species in train; a species too rare for both rules is a contradiction,
    unless cap_rare lowers its floor to its upper."""
    upper = [t - int(math.ceil(min_keep_frac * t - 1e-9)) for t in totals]
    floor = [min(min_dev_boxes, u) if cap_rare else min_dev_boxes for u in upper]
    return floor, upper


def _dev_search(names, sizes, boxes, floor, upper, lo_n, hi_n, target_n, max_nodes):
    """Exact branch and bound over session subsets.

    Subsets are visited in lexicographic order of their sorted name tuples (a
    set, then each extension by a later name in ascending order), so the first
    subset reaching a deviation |n - target_n| is also the tie-break winner and
    any branch that cannot beat the best deviation is cut. Bounds: image and
    per-species upper limits, species floors still reachable from the
    remaining sessions, and the nearest image total still reachable (subset
    sums of the remaining sessions, as a bitset)."""
    n_s, k = len(names), len(floor)
    reach = [0] * (n_s + 1)
    reach[n_s] = 1
    rem = [[0] * k for _ in range(n_s + 1)]
    for i in range(n_s - 1, -1, -1):
        reach[i] = reach[i + 1] | (reach[i + 1] << sizes[i])
        rem[i] = [rem[i + 1][c] + boxes[i][c] for c in range(k)]
    best, nodes = [None], [0]

    def nearest(n, j):
        lo_s, hi_s = max(0, lo_n - n), hi_n - n
        if hi_s < lo_s:
            return None
        o = target_n - n
        c = min(max(int(round(o)), lo_s), hi_s)
        mask = reach[j]
        for d in range(hi_s - lo_s + 1):
            hit = [abs(x - o) for x in (c - d, c + d) if lo_s <= x <= hi_s and (mask >> x) & 1]
            if hit:
                return min(hit)
            if c - d < lo_s and c + d > hi_s:
                return None
        return None

    def visit(i, n, bx, chosen):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise SplitError("dev search exceeded %d nodes; tighten or relax the constraints" % max_nodes)
        if lo_n <= n <= hi_n and all(bx[c] >= floor[c] for c in range(k)):
            d = abs(n - target_n)
            if best[0] is None or d < best[0][0]:
                best[0] = (d, list(chosen), n, list(bx))
        for j in range(i, n_s):
            n2 = n + sizes[j]
            if n2 > hi_n:
                continue
            bx2 = [bx[c] + boxes[j][c] for c in range(k)]
            if any(bx2[c] > upper[c] or bx2[c] + rem[j + 1][c] < floor[c] for c in range(k)):
                continue
            lb = nearest(n2, j + 1)
            if lb is None or (best[0] is not None and lb >= best[0][0]):
                continue
            chosen.append(names[j])
            visit(j + 1, n2, bx2, chosen)
            chosen.pop()

    visit(0, 0, [0] * k, [])
    return best[0], nodes[0]


def choose_dev(session_images, session_boxes, target_frac=DEV_TARGET_FRAC,
               frac_range=DEV_FRAC_RANGE, min_dev_boxes=DEV_MIN_BOXES,
               min_keep_frac=DEV_MIN_KEEP_FRAC, cap_rare=False, max_nodes=DEV_MAX_NODES,
               explain=True):
    """dev as a set of whole sessions: the subset with an image share in
    frac_range, every species >= min_dev_boxes boxes in dev and (only when
    min_keep_frac > 0, which the protocol does not ask for) >= min_keep_frac of
    its boxes left in train, that is closest to target_frac; ties go to the
    smallest sorted tuple of session names. Exact, so the same data always
    gives the same dev on any machine.

    session_images: {session: n images}; session_boxes: {session: [boxes per
    species id 0..N_SPECIES-1]}. When nothing is feasible the error lists the
    per-species bounds and which relaxations are feasible (explain=True)."""
    names = sorted(session_images)
    sizes = [session_images[s] for s in names]
    boxes = [list(session_boxes[s]) for s in names]
    total = sum(sizes)
    totals = [sum(b[c] for b in boxes) for c in range(N_SPECIES)]
    floor, upper = _dev_bounds(totals, min_dev_boxes, min_keep_frac, cap_rare)
    lo_n = int(math.ceil(frac_range[0] * total - 1e-9))
    hi_n = int(math.floor(frac_range[1] * total + 1e-9))
    target_n = target_frac * total
    best, nodes = _dev_search(names, sizes, boxes, floor, upper, lo_n, hi_n, target_n, max_nodes)
    search = {"method": "exact branch and bound", "nodes": nodes, "target_frac": target_frac,
              "frac_range": list(frac_range), "min_dev_boxes": min_dev_boxes,
              "min_keep_frac": min_keep_frac, "cap_rare": cap_rare,
              "n_sessions": len(names), "n_images": total,
              "floor": dict(zip(CWD12_SPECIES, floor)), "upper": dict(zip(CWD12_SPECIES, upper)),
              "capped": {CWD12_SPECIES[c]: floor[c] for c in range(N_SPECIES) if floor[c] < min_dev_boxes}}
    if best is None:
        contradictions = {CWD12_SPECIES[c]: {"train_boxes": totals[c], "dev_max": upper[c]}
                          for c in range(N_SPECIES) if floor[c] > upper[c]}
        alts = []
        if explain:
            for md, kf, cap in DEV_RELAXATIONS:
                if (md, kf, cap) == (min_dev_boxes, min_keep_frac, cap_rare):
                    continue
                f2, u2 = _dev_bounds(totals, md, kf, cap)
                try:
                    b2, _ = _dev_search(names, sizes, boxes, f2, u2, lo_n, hi_n, target_n, max_nodes)
                    verdict = "infeasible" if b2 is None else \
                        "feasible, %.1f%% of images, min species boxes %d%s" % (
                            100 * b2[2] / total, min(b2[3]),
                            " (below the protocol's %d)" % DEV_MIN_BOXES if min(b2[3]) < DEV_MIN_BOXES else "")
                except SplitError:
                    verdict = "search exceeded %d nodes" % max_nodes
                literal = (md, kf, cap) == (DEV_MIN_BOXES, 0.0, False)
                alts.append("--dev-min-boxes %d --dev-min-keep %.2f%s%s: %s"
                            % (md, kf, " --dev-cap-rare" if cap else "",
                               " (the protocol as written)" if literal else "", verdict))
        raise SplitError(
            "no feasible dev: no set of whole sessions holds %.0f-%.0f%% of the %d train "
            "images with every species >= %d boxes in dev%s. Train boxes per species: %s. %s%s"
            % (100 * frac_range[0], 100 * frac_range[1], total, min_dev_boxes,
               (" and >= %.0f%% of its boxes left in train" % (100 * min_keep_frac))
               if min_keep_frac > 0 else "",
               dict(zip(CWD12_SPECIES, totals)),
               ("Contradictory for %s (the keep rule caps dev below the floor). "
                % contradictions) if contradictions else "",
               ("Relaxations (exact search): " + "; ".join(alts)) if alts else ""))
    _, sessions, n, bx = best
    return {"sessions": sessions, "images": n, "frac": n / total,
            "boxes": dict(zip(CWD12_SPECIES, bx)),
            "keep_frac": {CWD12_SPECIES[c]: (1 - bx[c] / totals[c]) if totals[c] else 1.0
                          for c in range(N_SPECIES)},
            "search": search}


# ------------------------------------------------------- exam annotations
def _image_frame(path):
    """(width, height, EXIF orientation) of the pixels the trainer sees: cv2
    applies the EXIF orientation when it loads a JPEG, so a transposing
    orientation swaps the stored size. Missing EXIF is orientation None."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        orient = None
        try:
            orient = im.getexif().get(0x0112)
        except Exception:
            orient = "unreadable"
    if orient in (5, 6, 7, 8):
        w, h = h, w
    return w, h, orient


def _identity_orientation(orient):
    """True when the decoder shows the stored pixels as they are: no EXIF
    orientation, 1, or a value outside 2..8 (ignored by cv2 and PIL)."""
    return orient is None or (isinstance(orient, int) and not 2 <= orient <= 8)


def _frame_check(orient, size, w, h):
    """None when the annotation was made on the frame the trainer sees (w x h,
    after EXIF), else the reason to drop the image.

    A size that is the transposed frame means the boxes were drawn on the
    stored pixels of a rotated image. Without a size, or on a square image, a
    transposing orientation (5-8) cannot be checked at all. Orientations 2-4
    keep the size, so a size cannot tell; convert_three_season decides those
    per split."""
    if size is not None and size != (w, h):
        return "annotation_frame_rotated" if size == (h, w) else "annotation_size_mismatch"
    if _identity_orientation(orient):
        return None
    if size is not None and (orient in (2, 3, 4) or (orient in (5, 6, 7, 8) and w != h)):
        return None
    return "annotation_frame_unverifiable"


def _xml_boxes(path):
    """([(name, x1, y1, x2, y2)], (w, h) or None) from a Pascal VOC file."""
    root = ET.parse(path).getroot()
    size = None
    s = root.find("size")
    if s is not None:
        try:
            w, h = int(float(s.findtext("width"))), int(float(s.findtext("height")))
            size = (w, h) if w > 0 and h > 0 else None
        except (TypeError, ValueError):
            size = None
    out = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            raise SplitError("%s: object %r has no bndbox" % (path, name))
        try:
            xy = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            raise SplitError("%s: object %r has an incomplete bndbox" % (path, name))
        out.append((name,) + tuple(xy))
    return out, size


def _region_name(attrs, path):
    """The class of a VGG region: {"Group": {"Name": true}} or {"class": "Name"}."""
    names = []
    for v in (attrs or {}).values():
        if isinstance(v, dict):
            names += [k for k, flag in v.items() if flag]
        elif isinstance(v, str) and v.strip():
            names.append(v.strip())
    if len(names) != 1:
        raise SplitError("%s: region class is not one name: %r" % (path, attrs))
    return names[0]


def _rect(shape, path):
    if "bbox" in shape:
        x, y, w, h = [float(v) for v in shape["bbox"]]
        return x, y, x + w, y + h
    if "all_points_x" in shape:
        xs, ys = shape["all_points_x"], shape["all_points_y"]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
    try:
        x, y = float(shape["x"]), float(shape["y"])
        return x, y, x + float(shape["width"]), y + float(shape["height"])
    except (KeyError, TypeError, ValueError):
        raise SplitError("%s: unreadable region shape %r" % (path, shape))


def _json_boxes(path, image_name):
    """([(name, x1, y1, x2, y2)], (w, h) or None) from a VGG-annotator json
    (regions with x, y, width, height) or a COCO json ([x, y, w, h] bboxes)."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "annotations" in data and "categories" in data:
        cats = {c["id"]: c["name"] for c in data["categories"]}
        images = data.get("images") or []
        ids = [im["id"] for im in images if os.path.basename(str(im.get("file_name", ""))) == image_name]
        if not ids and len(images) == 1:
            ids = [images[0]["id"]]
        if not ids and len(images) > 1:
            # without a match every annotation in the file would land on this image
            raise SplitError("%s: COCO json lists %d images and none is %s"
                             % (path, len(images), image_name))
        size = None
        for im in images:
            if im.get("id") in ids and im.get("width") and im.get("height"):
                size = (int(im["width"]), int(im["height"]))
        anns = [a for a in data["annotations"] if not ids or a.get("image_id") in ids]
        out = []
        for a in anns:
            x, y, w, h = [float(v) for v in a["bbox"]]
            out.append((str(cats[a["category_id"]]).strip(), x, y, x + w, y + h))
        return out, size
    if isinstance(data, dict) and "regions" in data:
        entries = [data]
    elif isinstance(data, dict):
        entries = [v for v in data.values() if isinstance(v, dict) and "regions" in v]
        named = [e for e in entries if e.get("filename") == image_name]
        entries = named or entries
    else:
        entries = []
    if len(entries) != 1:
        raise SplitError("%s: not a recognised VGG or COCO annotation (%d VGG entries; "
                         "top-level keys %s)" % (path, len(entries),
                                                 sorted(data)[:8] if isinstance(data, dict) else type(data).__name__))
    regions = entries[0]["regions"]
    if isinstance(regions, dict):
        regions = [regions] if "shape_attributes" in regions else list(regions.values())
    out = []
    for reg in regions:
        shape = reg.get("shape_attributes") or {}
        if "bbox" in reg:
            shape = dict(shape, bbox=reg["bbox"])
        out.append((_region_name(reg.get("region_attributes"), path),) + _rect(shape, path))
    return out, None


def exam_class(name, other_plant_names):
    """INC id of an exam class name, or None if it cannot be placed."""
    sp = species_of(name)
    if sp is not None:
        return CWD12_SPECIES.index(sp)
    k = name_key(name)
    others = {name_key(n) for n in other_plant_names}
    if k in others or (k.endswith("s") and k[:-1] in others) or k + "s" in others:
        return C.OTHER_PLANT
    return None


def _to_yolo_box(cls, x1, y1, x2, y2, w, h, stats):
    """Clip a pixel box to the image and normalise; None if degenerate."""
    if min(x1, y1) < 0 or x2 > w or y2 > h:
        stats["clipped_boxes"] += 1
    x1, x2 = max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))
    y1, y2 = max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def _new_stats():
    return {"images_found": 0, "boxes_read": 0, "dropped_images": collections.Counter(),
            "background_images": collections.Counter(),
            "dropped_boxes": collections.Counter(), "clipped_boxes": 0,
            "exif_orientation": collections.Counter(), "annotation": collections.Counter(),
            "examples": collections.defaultdict(list)}


def _drop(stats, reason, what):
    stats["dropped_images"][reason] += 1
    if len(stats["examples"][reason]) < 10:
        stats["examples"][reason].append(what)


def _check_drops(split, stats, max_drop_share=MAX_DROP_SHARE):
    """Stop the build when an exam lost (or kept without labels) more images,
    or dropped more boxes, than a converter should ever need to."""
    n, nb = stats["images_found"], stats["boxes_read"]
    irregular = sum(stats["dropped_images"].values()) + sum(stats["background_images"].values())
    degenerate = stats["dropped_boxes"]["degenerate"]
    over = []
    if irregular > max(MAX_DROPS_ALWAYS_OK, max_drop_share * n):
        over.append("%d of %d images dropped or kept without labels (%s%s)"
                    % (irregular, n, dict(stats["dropped_images"]),
                       ", background: %s" % dict(stats["background_images"])
                       if stats["background_images"] else ""))
    if degenerate > max(MAX_DROPS_ALWAYS_OK, max_drop_share * nb):
        over.append("%d of %d boxes degenerate" % (degenerate, nb))
    if over:
        raise SplitError(
            "%s: %s; the limit is max(%d, %.1f%%). The layout is not what this converter "
            "assumes, or the download is incomplete (e.g. %s). Inspect it; if the drops are "
            "genuine, rebuild with a larger --max-drop-share."
            % (split, "; ".join(over), MAX_DROPS_ALWAYS_OK, 100 * max_drop_share,
               {k: v[:3] for k, v in stats["examples"].items()}))


def convert_three_season(src_root, split, other_plant_names=DEFAULT_OTHER_PLANT,
                         max_drop_share=MAX_DROP_SHARE):
    """Rows of a 3SeasonWeedDet10 season (Pascal VOC .xml next to each image,
    the .json as a fallback), boxes in INC ids normalised by the real image
    size. Labels are not written here (the key decides the file name). Raises
    on any class name that is neither a cwd12 species nor in other_plant_names,
    and when more images or boxes are dropped than max_drop_share allows.

    The format ships an annotation for every image, so an image without one
    is dropped (not kept as background). Once any image of the split turns out
    to be annotated on its stored pixels despite an EXIF rotation, every image
    with an EXIF orientation is unverifiable and dropped: for orientations 2-4
    the sizes agree either way, and boxes drawn on the stored pixels would land
    mirrored or 180 degrees away from the plant the trainer sees."""
    src_root = Path(src_root)
    stats = _new_stats()
    rows, unknown = [], collections.defaultdict(list)
    for d, files in _walk_files(src_root):
        lower = {f.lower(): f for f in files}
        for name in files:
            if not _is_image(name):
                continue
            stats["images_found"] += 1
            if stats["images_found"] % PROGRESS_EVERY == 0:
                log("  converting %s: %d images" % (split, stats["images_found"]))
            stem = os.path.splitext(name)[0]
            img = os.path.join(d, name)
            rel = os.path.relpath(img, src_root)
            xml, js = lower.get((stem + ".xml").lower()), lower.get((stem + ".json").lower())
            ann, size, kind = None, None, None
            if xml:
                try:
                    ann, size = _xml_boxes(os.path.join(d, xml))
                    kind = "xml"
                except ET.ParseError:
                    ann = None
            if ann is None and js:
                try:
                    ann, size = _json_boxes(os.path.join(d, js), name)
                except (ValueError, KeyError, TypeError, IndexError) as e:
                    raise SplitError("%s: unreadable json annotation (%s: %s)"
                                     % (os.path.join(d, js), type(e).__name__, e))
                kind = "json" if not xml else "json_after_bad_xml"
            if ann is None:
                if xml:
                    raise SplitError("%s: the XML does not parse and there is no json" % img)
                _drop(stats, "no_annotation", rel)
                continue
            stats["annotation"][kind] += 1
            try:
                w, h, orient = _image_frame(img)
            except Exception as e:
                raise SplitError("%s: image cannot be read (%s); a partial download?" % (img, e))
            stats["exif_orientation"][str(orient)] += 1
            reason = _frame_check(orient, size, w, h)
            if reason:
                _drop(stats, reason, "%s ann %s image %s orientation %s" % (rel, size, (w, h), orient))
                continue
            stats["boxes_read"] += len(ann)
            boxes = []
            for bname, x1, y1, x2, y2 in ann:
                cls = exam_class(bname, other_plant_names)
                if cls is None:
                    if len(unknown[bname]) < 3:
                        unknown[bname].append(rel)
                    continue
                if x2 <= x1 or y2 <= y1:
                    stats["dropped_boxes"]["degenerate"] += 1
                    continue
                b = _to_yolo_box(cls, x1, y1, x2, y2, w, h, stats)
                if b is None:
                    stats["dropped_boxes"]["degenerate"] += 1
                    continue
                boxes.append(b)
            if ann and not boxes:
                _drop(stats, "all_boxes_degenerate", rel)
                continue
            rows.append({"image": img, "source": "3seasonweeddet10/%s" % src_root.name,
                         "session": "", "stem": stem, "rel": rel, "boxes": boxes,
                         "orientation": orient})
    if unknown:
        raise SplitError(
            "%s: class name(s) that are neither a cwd12 species nor OtherPlant (%s): %s"
            % (split, list(other_plant_names),
               "; ".join("%r e.g. %s" % (k, v) for k, v in sorted(unknown.items()))))
    if stats["dropped_images"]["annotation_frame_rotated"]:
        kept = []
        for r in rows:
            if _identity_orientation(r["orientation"]) or not r["boxes"]:
                kept.append(r)
            else:
                _drop(stats, "annotation_frame_unverifiable",
                      "%s orientation %s, in a split with rotated-frame annotations"
                      % (r["rel"], r["orientation"]))
        rows = kept
    for r in rows:
        del r["orientation"]
    _check_drops(split, stats, max_drop_share)
    return rows, stats


def _imageweeds_names_file(root):
    """Class names shipped with the dataset, if any: (file, [names]) or None."""
    listing = set(os.listdir(root))
    for fname in ("data.yaml", "dataset.yaml", "classes.txt", "obj.names"):
        if fname not in listing:
            continue
        path = root / fname
        if fname.endswith(".yaml"):
            import yaml
            with open(path) as fh:
                names = (yaml.safe_load(fh) or {}).get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
        else:
            with open(path) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            return fname, [str(n) for n in names]
    return None


def imageweeds_class_map(names=IMAGEWEEDS_CLASSES):
    """ImageWeeds id -> INC id: a cwd12 species keeps its id (ragweed -> 5),
    everything else is OtherPlant."""
    out = {}
    for i, n in enumerate(names):
        sp = species_of(n)
        out[i] = CWD12_SPECIES.index(sp) if sp else C.OTHER_PLANT
    return out


def convert_imageweeds(root, max_drop_share=MAX_DROP_SHARE):
    """Rows of ImageWeeds (images/, YOLO labels/ mirrored), boxes in INC ids.

    An image without a label file is a background image, as Ultralytics reads a
    YOLO dataset (and as the earlier ImageWeeds scores counted all 3,208), so it
    is kept with an empty label; it is counted, and capped with the drops,
    because a labels/ dir still being unpacked looks exactly like this. YOLO
    labels carry no frame size, so an image with boxes and an EXIF orientation
    cannot be checked and is dropped."""
    root = Path(root)
    stats = _new_stats()
    shipped = _imageweeds_names_file(root)
    if shipped is not None and [name_key(n) for n in shipped[1]] != [name_key(n) for n in IMAGEWEEDS_CLASSES]:
        raise SplitError("ImageWeeds ships %s with classes %s, not the assumed %s"
                         % (shipped[0], shipped[1], IMAGEWEEDS_CLASSES))
    order = {"names": list(IMAGEWEEDS_CLASSES),
             "inc_ids": [imageweeds_class_map()[i] for i in range(len(IMAGEWEEDS_CLASSES))],
             "status": ("verified against %s" % shipped[0]) if shipped else
                       ("UNVERIFIED ASSUMPTION: the label files ship without a class list; the "
                        "order is from the HF dataset card (crossdataset_eval.py, 2026-08-26)")}
    cmap = imageweeds_class_map()
    img_root, lbl_root = root / "images", root / "labels"
    if not img_root.is_dir() or not lbl_root.is_dir():
        raise SplitError("ImageWeeds at %s has no images/ or labels/ dir" % root)
    labels = set()
    for d, files in _walk_files(lbl_root):
        rel_d = os.path.relpath(d, lbl_root)
        labels.update(os.path.normpath(os.path.join(rel_d, f)) for f in files)
    rows = []
    for d, files in _walk_files(img_root):
        for name in files:
            if not _is_image(name):
                continue
            stats["images_found"] += 1
            if stats["images_found"] % PROGRESS_EVERY == 0:
                log("  converting imageweeds: %d images" % stats["images_found"])
            img = os.path.join(d, name)
            rel = os.path.relpath(img, img_root)
            stem = os.path.splitext(name)[0]
            lrel = os.path.normpath(os.path.splitext(rel)[0] + ".txt")
            if lrel in labels:
                src = _strict_yolo(lbl_root / lrel, len(IMAGEWEEDS_CLASSES) - 1, check_range=False)
            else:
                src = []
                stats["background_images"]["no_label_file"] += 1
                if len(stats["examples"]["background_no_label_file"]) < 10:
                    stats["examples"]["background_no_label_file"].append(rel)
            try:
                w, h, orient = _image_frame(img)
            except Exception as e:
                raise SplitError("%s: image cannot be read (%s)" % (img, e))
            stats["exif_orientation"][str(orient)] += 1
            reason = _frame_check(orient, None, w, h) if src else None
            if reason:
                _drop(stats, reason, "%s orientation %s" % (rel, orient))
                continue
            stats["boxes_read"] += len(src)
            boxes = []
            for c, cx, cy, bw, bh in src:
                b = _to_yolo_box(cmap[c], (cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h, w, h, stats)
                if b is None:
                    stats["dropped_boxes"]["degenerate"] += 1
                    continue
                boxes.append(b)
            if src and not boxes:
                _drop(stats, "all_boxes_degenerate", rel)
                continue
            rows.append({"image": img, "source": IMAGEWEEDS_SLUG, "session": "",
                         "stem": stem, "rel": rel, "boxes": boxes})
    _check_drops("imageweeds", stats, max_drop_share)
    return rows, stats, order


# ----------------------------------------------------------------- keys
def assign_keys(split, rows):
    """key = <split>__<sanitised stem>; stems that collide within the split
    get a suffix from their source-relative path, so every key is unique and
    the same on every rebuild."""
    base = ["%s__%s" % (split, sanitise(r["stem"])) for r in rows]
    counts = collections.Counter(base)
    for r, b in zip(rows, base):
        r["key"] = b if counts[b] == 1 else "%s__%s" % (b, C.sha256_text(r["rel"])[:8])
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise SplitError("%s: duplicate keys after disambiguation" % split)


# ------------------------------------------------------------- exam dirs
def _expected_data_yaml(out_dir):
    """The data.yaml common.materialise writes into out_dir."""
    text = "path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n" % (out_dir, C.NC)
    return text + "".join("  - %s\n" % n for n in C.CLASS_NAMES)


def exam_problems(split, rows, out_dir=None):
    """Differences between a materialised exam dir and the manifest rows."""
    out_dir = Path(out_dir or C.EXAMS_DIR / split)
    if not out_dir.is_dir():
        return ["%s: exam dir %s is missing" % (split, out_dir)]
    probs = []
    want = {r["key"]: r for r in rows}
    extra = sorted(n for n in set(os.listdir(out_dir)) - {"images", "labels", "data.yaml"}
                   if not n.endswith(".cache"))
    if extra:
        probs.append("%s: unexpected entries in the exam dir: %s" % (split, extra))
    try:
        images = os.listdir(out_dir / "images")
    except OSError:
        images = []
    try:
        labels = os.listdir(out_dir / "labels")
    except OSError:
        labels = []
    seen_img, seen_lbl = set(), set()
    for n in images:
        key, ext = os.path.splitext(n)
        r = want.get(key)
        p = out_dir / "images" / n
        if r is None:
            probs.append("%s: exam image %s is not in the manifest" % (split, n))
            continue
        seen_img.add(key)
        if not os.path.islink(p) or os.readlink(p) != r["image"] or \
                ext != (os.path.splitext(r["image"])[1].lower() or ".jpg"):
            probs.append("%s: exam image %s does not link to %s" % (split, n, r["image"]))
    for n in labels:
        key = os.path.splitext(n)[0]
        r = want.get(key)
        if r is None or not n.endswith(".txt"):
            probs.append("%s: exam label %s is not in the manifest" % (split, n))
            continue
        seen_lbl.add(key)
        if C.sha256_file(out_dir / "labels" / n) != r["label_sha256"]:
            probs.append("%s: exam label %s differs from the manifest" % (split, n))
    for key in sorted(set(want) - seen_img)[:10]:
        probs.append("%s: exam image for %s missing" % (split, key))
    for key in sorted(set(want) - seen_lbl)[:10]:
        probs.append("%s: exam label for %s missing" % (split, key))
    try:
        with open(out_dir / "data.yaml") as fh:
            if fh.read() != _expected_data_yaml(out_dir):
                probs.append("%s: exam data.yaml differs" % split)
    except OSError:
        probs.append("%s: exam data.yaml missing" % split)
    return probs


def materialise_exam(split, rows):
    """EXAMS_DIR/<split>, built once and made read-only; an existing dir must
    already match the manifest (it is never rebuilt silently)."""
    out_dir = C.EXAMS_DIR / split
    if out_dir.exists():
        probs = exam_problems(split, rows, out_dir)
        if probs:
            raise SplitError("exam dir %s differs from the new %s manifest (%d problem(s): %s). "
                             "Remove it by hand if the change is intended."
                             % (out_dir, split, len(probs), probs[:5]))
        return "kept"
    try:
        yaml_path = C.materialise(rows, out_dir)
        for n in os.listdir(out_dir / "labels"):
            os.chmod(out_dir / "labels" / n, 0o444)
        os.chmod(yaml_path, 0o444)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return "created"


# ------------------------------------------------------------------ build
def _hash_rows(rows, what):
    """sha256 of every image and label. A converted label is hashed where it
    is staged, which is byte for byte what is swapped in."""
    n = len(rows)
    for i, r in enumerate(rows, 1):
        r["sha256"] = C.sha256_file(r["image"])
        r["label_sha256"] = C.sha256_file(r.get("staged_label") or r["label"])
        _progress(i, n, "sha256 %s" % what)


def _dhash_rows(rows, what):
    """dHash every row once (a row hashed by an earlier step keeps its hash)."""
    n = len(rows)
    for i, r in enumerate(rows, 1):
        if "dhash" not in r:
            r["dhash"] = C.dhash(r["image"])
        _progress(i, n, "dhash %s" % what)


def drop_cwd12_copies(exam_rows, cwd12_rows, what):
    """(kept, dropped) exam rows: an exam image within HOLDOUT_NEAR_DUP_BITS of
    any cwd12 photograph (train, valid or test) is cwd12 data, not out of domain.
    3SeasonWeedDet10's data2022 shares a capture session with cwd12 train
    (20220129_CanonEOS4000D_EO). dropped = [(row, cwd12 image, bits)]. Rows that
    cannot be hashed are kept here; the index step refuses them."""
    _dhash_rows(cwd12_rows, "cwd12")
    index = NearHashIndex()
    for r in cwd12_rows:
        if r["dhash"] is not None:
            index.add(r["dhash"], r["image"], max_bits=HOLDOUT_NEAR_DUP_BITS)
    _dhash_rows(exam_rows, what)
    kept, dropped = [], []
    for r in exam_rows:
        m = index.find(r["dhash"]) if r["dhash"] is not None else None
        if m is None:
            kept.append(r)
        else:
            dropped.append((r, m[0], m[1]))
    return kept, dropped


def _counts(rows):
    counts = [0] * C.NC
    for r in rows:
        for b in r["boxes"]:
            counts[b[0]] += 1
    return counts


def _plain(stats):
    return {k: (dict(v) if isinstance(v, (collections.Counter, collections.defaultdict)) else v)
            for k, v in stats.items()}


def _swap_in(staged, final):
    """Replace a converted-label dir by its staged copy (renames within
    SPLITS_DIR, so on one file system)."""
    final.parent.mkdir(parents=True, exist_ok=True)
    old = final.with_name(final.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        os.rename(final, old)
    os.rename(staged, final)
    shutil.rmtree(old, ignore_errors=True)


def index_record(entries, missing, count_warnings, n_dev):
    """The never-train index file's content.

    It is complete only when every evaluation split was built and every exam
    kept the protocol's image count. Otherwise min_expected is the protocol's
    total (dev plus EXPECTED_IMAGES), and always more than the entries held, so
    NeverTrainGuard.load, and with it every merge guard that loads the index,
    refuses the file until the missing exams are built and `lock` has accepted
    the counts. A partial index must never pass for the whole guard."""
    complete = not missing and not count_warnings
    rec = {"entries": entries, "bits": HOLDOUT_NEAR_DUP_BITS, "complete": complete,
           "missing": list(missing), "count_warnings": count_warnings,
           "min_expected": len(entries)}
    if not complete:
        rec["min_expected"] = max(len(entries) + 1, n_dev + sum(EXPECTED_IMAGES.values()))
        rec["note"] = ("INCOMPLETE: NeverTrainGuard.load refuses this index (%d entries, "
                       "min_expected %d); missing %s; exams whose image count differs from the "
                       "protocol: %s" % (len(entries), rec["min_expected"], list(missing) or "none",
                                          sorted(count_warnings) or "none"))
    return rec


def build(target_frac=DEV_TARGET_FRAC, frac_range=DEV_FRAC_RANGE, min_dev_boxes=DEV_MIN_BOXES,
          min_keep_frac=DEV_MIN_KEEP_FRAC, cap_rare=False,
          other_plant_names=DEFAULT_OTHER_PLANT, rebuild_locked=False,
          max_drop_share=MAX_DROP_SHARE):
    """Build every split's manifest, the converted exam labels, the never-train
    index, the exam dirs and summary.json. Returns the summary. Whatever it
    fails on, nothing written by an earlier build has been changed."""
    try:
        return _build(target_frac, frac_range, min_dev_boxes, min_keep_frac, cap_rare,
                      other_plant_names, rebuild_locked, max_drop_share)
    finally:
        shutil.rmtree(staging_dir(), ignore_errors=True)


def _build(target_frac, frac_range, min_dev_boxes, min_keep_frac, cap_rare,
           other_plant_names, rebuild_locked, max_drop_share):
    t0 = time.time()
    if C.LOCK_PATH.exists() and not rebuild_locked:
        raise SplitError("%s exists: the splits are locked. A rebuild changes locked "
                         "manifests; pass --rebuild-locked, then `lock --relock`." % C.LOCK_PATH)
    for p in (C.REPO, C.INC_DIR):
        if not Path(p).is_absolute():
            raise SplitError("%s must be an absolute path" % p)

    # 1. dev = whole sessions of cwd12 train
    train = list_cwd12("train")
    by_session = collections.defaultdict(list)
    for r in train:
        by_session[r["session"]].append(r)
    dev_choice = choose_dev({s: len(v) for s, v in by_session.items()},
                            {s: species_counts(b for r in v for b in r["boxes"])
                             for s, v in by_session.items()},
                            target_frac=target_frac, frac_range=frac_range,
                            min_dev_boxes=min_dev_boxes, min_keep_frac=min_keep_frac,
                            cap_rare=cap_rare)
    dev_sessions = set(dev_choice["sessions"])
    log("train: %d images in %d sessions; dev = %d sessions, %d images (%.1f%%)"
        % (len(train), len(by_session), len(dev_sessions), dev_choice["images"],
           100 * dev_choice["frac"]))
    dev_deviations = []
    low = {k: v for k, v in dev_choice["boxes"].items() if v < DEV_MIN_BOXES}
    if low:
        dev_deviations.append("species under the protocol's %d dev boxes: %s" % (DEV_MIN_BOXES, low))
    if abs(target_frac - DEV_TARGET_FRAC) > 1e-9:
        dev_deviations.append("target share %.3f instead of the protocol's %.2f"
                              % (target_frac, DEV_TARGET_FRAC))
    if min_keep_frac > 0:
        dev_deviations.append("keep rule %.2f applied (not a protocol rule)" % min_keep_frac)
    for d in dev_deviations:
        log("WARNING: dev: %s" % d)

    rows = {"dev": [r for r in train if r["session"] in dev_sessions],
            "test": list_cwd12("valid") + list_cwd12("test")}
    candidates = [r for r in train if r["session"] not in dev_sessions]
    stats = {}
    missing = []

    # 2. exams, converted into the INC class space
    for split in ("ood22", "ood23"):
        src = three_season_dir(split)
        if not src.is_dir():
            log("WARNING: %s source %s does not exist; %s skipped (lock will refuse)" % (split, src, split))
            missing.append(split)
            continue
        rows[split], stats[split] = convert_three_season(src, split, other_plant_names, max_drop_share)
    iw_order = None
    if not imageweeds_dir().is_dir():
        log("WARNING: imageweeds source %s does not exist; skipped (lock will refuse)" % imageweeds_dir())
        missing.append("imageweeds")
    else:
        rows["imageweeds"], stats["imageweeds"], iw_order = convert_imageweeds(imageweeds_dir(),
                                                                               max_drop_share)

    for split in ("ood22", "ood23", "imageweeds"):
        if split in rows and not rows[split]:
            log("WARNING: %s source holds no usable image; skipped (lock will refuse)" % split)
            del rows[split]
            missing.append(split)
    for split in missing:
        if C.manifest_path(split).exists():
            raise SplitError("%s: its source is gone but %s exists from an earlier build"
                             % (split, C.manifest_path(split)))

    # 2b. an exam keeps only photographs cwd12 does not hold. Counted, not capped
    #     by max_drop_share: this defines the exam, it is not a conversion loss.
    exam_cwd12 = {}
    for split in ("ood22", "ood23", "imageweeds"):
        if split not in rows:
            continue
        rows[split], dropped = drop_cwd12_copies(rows[split], train + rows["test"], split)
        exam_cwd12[split] = {"dropped": len(dropped),
                             "examples": [{"exam": r["image"], "cwd12": img, "bits": bits}
                                          for r, img, bits in dropped[:20]]}
        if dropped:
            log("%s: %d image(s) are cwd12 photographs (<= %d dHash bits) and leave the exam"
                % (split, len(dropped), HOLDOUT_NEAR_DUP_BITS))
        if not rows[split]:
            raise SplitError("%s: every image is a cwd12 photograph" % split)

    # 3. keys; converted labels are staged and replace SPLITS_DIR/labels/<split>
    #    only after every check below has passed, so a failed rebuild leaves the
    #    label files a locked manifest hashed as they were
    for split, rs in rows.items():
        assign_keys(split, rs)
    assign_keys("train_core", candidates)
    staged = staging_dir()
    shutil.rmtree(staged, ignore_errors=True)
    exam_sources = [s for s in ("ood22", "ood23", "imageweeds") if s in rows]
    for split in exam_sources:
        for r in rows[split]:
            r["label"] = str(converted_label_dir(split) / (r["key"] + ".txt"))
            r["staged_label"] = str(staged / split / (r["key"] + ".txt"))
            C.write_yolo(r["staged_label"], r["boxes"])

    # 4. never-train index over every eval image; train_core = the rest of train
    #    minus anything within range of it
    eval_splits = [s for s in C.EVAL_SPLITS if s in rows]
    for split in eval_splits:
        _dhash_rows(rows[split], split)
        bad = [r["image"] for r in rows[split] if r["dhash"] is None]
        if bad:
            raise SplitError("%s: %d evaluation image(s) cannot be hashed, e.g. %s"
                             % (split, len(bad), bad[:5]))
    entries, cross = [], collections.Counter()
    probe = NearHashIndex()
    for split in eval_splits:
        for r in sorted(rows[split], key=lambda r: r["key"]):
            m = probe.find(r["dhash"])
            if m is not None and m[0][0] != split:
                cross["%s~%s" % (split, m[0][0])] += 1
            probe.add(r["dhash"], (split, r["key"]), max_bits=HOLDOUT_NEAR_DUP_BITS)
            entries.append([r["dhash"], split, r["key"]])
    guard = C.NeverTrainGuard(entries)

    _dhash_rows(candidates, "train")
    cache = {r["image"]: r["dhash"] for r in candidates}
    hits, _ = guard.check([r["image"] for r in candidates], hash_fn=cache.get)
    hit_by_image = {h[0]: h for h in hits}
    train_drop, train_examples = collections.Counter(), collections.defaultdict(list)
    core = []
    for r in candidates:
        if r["dhash"] is None:
            train_drop["unhashable"] += 1
            continue
        h = hit_by_image.get(r["image"])
        if h is not None:
            reason = "near_%s" % h[1]
            train_drop[reason] += 1
            if len(train_examples[reason]) < 20:
                train_examples[reason].append({"train": r["key"], "eval": h[2], "bits": h[3]})
            continue
        core.append(r)
    rows["train_core"] = core
    log("train_core: %d images; dropped %s" % (len(core), dict(train_drop) or "none"))

    train_totals = species_counts(b for r in train for b in r["boxes"])
    core_counts = species_counts(b for r in core for b in r["boxes"])
    short = {CWD12_SPECIES[c]: round(core_counts[c] / train_totals[c], 4)
             for c in range(N_SPECIES) if core_counts[c] < min_keep_frac * train_totals[c]}
    if short:
        raise SplitError("after near-copy removal, train_core keeps under %.0f%% of the "
                         "boxes of %s; dropped: %s" % (100 * min_keep_frac, short, dict(train_drop)))

    # 5. image counts against the protocol, on the images each split KEEPS; the
    #    hashes; the existing exam dirs. Nothing on disk has changed yet.
    found = {s: st["images_found"] for s, st in stats.items()}
    found["test"] = len(rows["test"])
    count_warnings = {s: {"expected": n, "found": found[s], "kept": len(rows[s])}
                      for s, n in EXPECTED_IMAGES.items()
                      if s in rows and len(rows[s]) != n}
    for split, rs in rows.items():
        _hash_rows(rs, split)
    for split in eval_splits:
        if (C.EXAMS_DIR / split).exists():
            probs = exam_problems(split, rows[split])
            if probs:
                raise SplitError("exam dir %s differs from the new %s manifest (%d problem(s): %s). "
                                 "Remove it by hand if the change is intended."
                                 % (C.EXAMS_DIR / split, split, len(probs), probs[:5]))

    # 6. converted labels swapped in; manifests, index, exam dirs, summary
    for split in exam_sources:
        _swap_in(staged / split, converted_label_dir(split))
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_sha = {}
    for split in ALL_SPLITS:
        if split in rows:
            manifest_sha[split] = C.write_manifest(C.manifest_path(split), rows[split])
    index = index_record(entries, missing, count_warnings, len(rows["dev"]))
    _write_json(C.NEVER_TRAIN_INDEX, index)
    if not index["complete"]:
        log("WARNING: never-train index written INCOMPLETE; NeverTrainGuard.load (every "
            "merge guard) refuses it until the missing exams are built and `lock` accepts "
            "the counts (%d entries, min_expected %d)" % (len(entries), index["min_expected"]))
    exams = {split: materialise_exam(split, rows[split]) for split in eval_splits}

    per_split = {}
    for split in ALL_SPLITS:
        if split not in rows:
            continue
        counts = _counts(rows[split])
        per_split[split] = {"images": len(rows[split]), "boxes_total": sum(counts),
                            "boxes": dict(zip(C.CLASS_NAMES, counts)),
                            "manifest_sha256": manifest_sha[split]}
    summary = {
        "splits_version": C.SPLITS_VERSION,
        "built_utc": _utc(), "build_seconds": round(time.time() - t0, 1),
        "repo": str(C.REPO), "splits_dir": str(C.SPLITS_DIR), "exams_dir": str(C.EXAMS_DIR),
        "class_names": C.CLASS_NAMES,
        "missing": missing,
        "splits": per_split,
        "dev": {"sessions": dev_choice["sessions"], "images": dev_choice["images"],
                "frac": round(dev_choice["frac"], 5), "boxes": dev_choice["boxes"],
                "keep_frac": {k: round(v, 4) for k, v in dev_choice["keep_frac"].items()},
                "protocol_deviations": dev_deviations,
                "search": dev_choice["search"]},
        "train_sessions": {s: len(v) for s, v in sorted(by_session.items())},
        "dropped": {"train_core": dict(train_drop),
                    **{s: {"images": dict(st["dropped_images"]), "boxes": dict(st["dropped_boxes"]),
                           "kept_as_background": dict(st["background_images"])}
                       for s, st in stats.items()}},
        "drop_examples": {"train_core": dict(train_examples),
                          **{s: dict(st["examples"]) for s, st in stats.items()}},
        "conversion": {s: {k: v for k, v in _plain(st).items() if k != "examples"}
                       for s, st in stats.items()},
        "max_drop_share": max_drop_share,
        "other_plant_names": list(other_plant_names),
        "imageweeds_class_order": iw_order,
        "eval_cross_near_dups": dict(cross),
        "exam_cwd12_copies": exam_cwd12,
        "count_warnings": count_warnings,
        "nevertrain": {"entries": len(entries), "bits": HOLDOUT_NEAR_DUP_BITS,
                       "complete": index["complete"], "min_expected": index["min_expected"]},
        "exams": exams,
    }
    _write_json(summary_path(), summary)
    if count_warnings:
        log("WARNING: kept image counts differ from the protocol: %s" % count_warnings)
    log("built in %.0fs: %s; missing %s"
        % (time.time() - t0, {s: v["images"] for s, v in per_split.items()}, missing or "none"))
    return summary


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_summary():
    try:
        with open(summary_path()) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise SplitError("%s missing: run `build` first" % summary_path())


# ------------------------------------------------------------ lock/verify
def manifest_problems(split):
    """Full re-hash of one manifest's images and labels against its rows."""
    path = C.manifest_path(split)
    if not path.exists():
        return ["%s: manifest %s missing" % (split, path)]
    rows = C.read_manifest(path)
    probs = []
    n = len(rows)
    for i, r in enumerate(rows, 1):
        if not r["key"].startswith(split + "__"):
            probs.append("%s: key %s lacks the split prefix" % (split, r["key"]))
        for field, want in (("image", r["sha256"]), ("label", r["label_sha256"])):
            try:
                got = C.sha256_file(r[field])
            except OSError:
                probs.append("%s: %s %s is missing" % (split, field, r[field]))
                continue
            if got != want:
                probs.append("%s: %s %s changed (%s != %s)" % (split, field, r[field], got[:12], want[:12]))
        _progress(i, n, "verify %s" % split)
    return probs


def nevertrain_problems(require_complete=True):
    """The never-train index against the eval manifests. require_complete also
    demands the index be final (what lock writes and verify expects); lock
    checks the entries first, before it finalises."""
    try:
        with open(C.NEVER_TRAIN_INDEX) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        return ["never-train index unreadable: %s" % e]
    have = {(s, k) for _h, s, k in data["entries"]}
    want = set()
    for split in C.EVAL_SPLITS:
        if C.manifest_path(split).exists():
            want |= {(split, r["key"]) for r in C.read_manifest(C.manifest_path(split))}
    probs = []
    if have != want:
        probs.append("never-train index: %d eval image(s) not indexed, %d stale entries"
                     % (len(want - have), len(have - want)))
    if data.get("bits") != HOLDOUT_NEAR_DUP_BITS:
        probs.append("never-train index: bits %s != %d" % (data.get("bits"), HOLDOUT_NEAR_DUP_BITS))
    if require_complete:
        if data.get("complete") is not True:
            probs.append("never-train index is not complete (missing %s, count differences %s)"
                         % (data.get("missing"), sorted(data.get("count_warnings") or {})))
        elif data.get("min_expected") != len(want):
            probs.append("never-train index: min_expected %s != %d eval images"
                         % (data.get("min_expected"), len(want)))
    return probs


def _finalise_index(summary):
    """Mark the never-train index complete once lock has accepted the build's
    counts; from then on NeverTrainGuard.load accepts it. Returns whether the
    file changed."""
    with open(C.NEVER_TRAIN_INDEX) as fh:
        data = json.load(fh)
    if data.get("complete") is True:
        return False
    if data.get("missing") or (data.get("count_warnings") or {}) != (summary.get("count_warnings") or {}):
        raise SplitError("the never-train index (missing %s, count differences %s) does not match "
                         "summary.json (count differences %s); rebuild"
                         % (data.get("missing"), data.get("count_warnings"), summary.get("count_warnings")))
    data.pop("note", None)
    data.update(complete=True, min_expected=len(data["entries"]),
                accepted_count_warnings=data.get("count_warnings") or {})
    _write_json(C.NEVER_TRAIN_INDEX, data)
    return True


def _exam_dir_problems(split):
    if not C.manifest_path(split).exists():
        return []
    return exam_problems(split, C.read_manifest(C.manifest_path(split)))


def lock(relock=False, scorer_path=None, accept_counts=False):
    """Write LOCK.json after a full check that the build is complete and intact.

    An exam that keeps a different number of images than the protocol states
    is usually a download still being unpacked, or images the conversion
    dropped, so it blocks the lock unless accept_counts says the difference was
    looked at. Locking is also what marks the never-train index complete."""
    if C.LOCK_PATH.exists() and not relock:
        raise SplitError("%s exists; pass --relock to replace it" % C.LOCK_PATH)
    summary = read_summary()
    absent = [s for s in ALL_SPLITS if s in summary.get("missing", []) or not C.manifest_path(s).exists()]
    if absent:
        raise SplitError("cannot lock while split(s) %s are missing; rebuild once their "
                         "data is in place" % absent)
    if summary.get("count_warnings") and not accept_counts:
        raise SplitError("image counts differ from the protocol: %s. If the sources are "
                         "complete, lock with --accept-counts" % summary["count_warnings"])
    scorer = Path(scorer_path or default_scorer_path())
    if not scorer.is_file():
        raise SplitError("scorer %s not found; LOCK.json must record it" % scorer)
    probs = []
    for split in ALL_SPLITS:
        probs += manifest_problems(split)
    probs += nevertrain_problems(require_complete=False)
    for split in C.EVAL_SPLITS:
        probs += _exam_dir_problems(split)
    if probs:
        raise SplitError("cannot lock, %d problem(s): %s" % (len(probs), probs[:10]))
    if _finalise_index(summary):
        log("never-train index marked complete (count differences accepted: %s)"
            % (summary.get("count_warnings") or "none"))
    probs = nevertrain_problems()
    if probs:
        raise SplitError("cannot lock: %s" % probs)
    new = {"splits_version": C.SPLITS_VERSION,
           "manifests": {s: C.sha256_file(C.manifest_path(s)) for s in ALL_SPLITS},
           "nevertrain_sha256": C.sha256_file(C.NEVER_TRAIN_INDEX),
           "scorer_sha256": C.sha256_file(scorer),
           "created_utc": _utc()}
    entry = {"utc": new["created_utc"], "lock": new,
             "accepted_count_warnings": summary.get("count_warnings") or {}}
    if C.LOCK_PATH.exists():
        old = C.read_lock()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = C.SPLITS_DIR / ("LOCK.%s.json" % stamp)
        i = 1
        while archive.exists():
            archive = C.SPLITS_DIR / ("LOCK.%s-%d.json" % (stamp, i))
            i += 1
        shutil.copy2(C.LOCK_PATH, archive)
        diff = lock_diff(old, new)
        entry.update(previous=archive.name, diff=diff)
        log("relock: previous lock kept as %s" % archive.name)
        for line in diff or ["no hash changed"]:
            log("  %s" % line)
    _write_json(C.LOCK_PATH, new)
    with open(C.SPLITS_DIR / LOCK_LOG_NAME, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    for p in [C.manifest_path(s) for s in ALL_SPLITS] + [C.NEVER_TRAIN_INDEX, C.LOCK_PATH]:
        os.chmod(p, 0o444)
    log("locked %d manifests, never-train index and %s" % (len(new["manifests"]), scorer.name))
    return new


def lock_diff(old, new):
    out = []
    om, nm = old.get("manifests", {}), new.get("manifests", {})
    for s in sorted(set(om) | set(nm)):
        if om.get(s) != nm.get(s):
            out.append("manifest %s: %s -> %s" % (s, (om.get(s) or "-")[:12], (nm.get(s) or "-")[:12]))
    for k in ("nevertrain_sha256", "scorer_sha256", "splits_version"):
        if old.get(k) != new.get(k):
            out.append("%s: %s -> %s" % (k, str(old.get(k))[:12], str(new.get(k))[:12]))
    return out


def verify(scorer_path=None):
    """Every problem found by a full re-hash against the manifests and LOCK."""
    probs = []
    try:
        lk = C.read_lock()
    except FileNotFoundError:
        return ["%s missing: the splits are not locked" % C.LOCK_PATH]
    for split in sorted(set(lk["manifests"]) | set(ALL_SPLITS)):
        if split not in lk["manifests"]:
            probs.append("%s: not in LOCK.json" % split)
            continue
        try:
            C.verify_manifest_against_lock(split, lk)
        except (RuntimeError, OSError) as e:
            probs.append(str(e))
        probs += manifest_problems(split)
    try:
        if C.sha256_file(C.NEVER_TRAIN_INDEX) != lk["nevertrain_sha256"]:
            probs.append("never-train index changed since it was locked")
    except OSError:
        probs.append("never-train index %s missing" % C.NEVER_TRAIN_INDEX)
    probs += nevertrain_problems()
    scorer = Path(scorer_path or default_scorer_path())
    try:
        if C.sha256_file(scorer) != lk["scorer_sha256"]:
            probs.append("scorer %s changed since it was locked" % scorer)
    except OSError:
        probs.append("scorer %s missing" % scorer)
    for split in C.EVAL_SPLITS:
        probs += _exam_dir_problems(split)
    return probs


# ------------------------------------------------------------------- CLI
def print_summary():
    s = read_summary()
    log("splits %s built %s (%ss)" % (s["splits_version"], s["built_utc"], s["build_seconds"]))
    for split, v in s["splits"].items():
        log("  %-11s %6d images %7d boxes" % (split, v["images"], v["boxes_total"]))
    log("  missing: %s" % (s["missing"] or "none"))
    d = s["dev"]
    log("dev: %d sessions, %d images (%.1f%%): %s" % (len(d["sessions"]), d["images"],
                                                      100 * d["frac"], ", ".join(d["sessions"])))
    log("dev boxes per species: %s" % d["boxes"])
    log("dropped: %s" % s["dropped"])
    if s.get("imageweeds_class_order"):
        log("imageweeds class order: %s" % s["imageweeds_class_order"]["status"])
    if s["dev"].get("protocol_deviations"):
        log("dev deviates from the protocol: %s" % s["dev"]["protocol_deviations"])
    if s.get("count_warnings"):
        log("count warnings: %s" % s["count_warnings"])
    try:
        with open(C.NEVER_TRAIN_INDEX) as fh:
            nt = json.load(fh)
        log("never-train index: %d entries, %s"
            % (len(nt["entries"]), "complete" if nt.get("complete") is True else
               "INCOMPLETE (missing %s, count differences %s): refused until the exams are "
               "built and `lock` accepts the counts" % (nt.get("missing"), sorted(nt.get("count_warnings") or {}))))
    except (OSError, ValueError, KeyError) as e:
        log("never-train index: unreadable (%s)" % e)
    if s.get("eval_cross_near_dups"):
        log("eval images near an earlier eval split: %s" % s["eval_cross_near_dups"])
    log("LOCK.json: %s" % ("present, created %s" % C.read_lock()["created_utc"]
                           if C.LOCK_PATH.exists() else "absent"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc.splits",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build manifests, never-train index and exam dirs")
    b.add_argument("--other-plant", action="append", default=None, metavar="NAME",
                   help="an exam class name that is a plant but no cwd12 species; repeatable, "
                        "added to the default (%s)" % ", ".join(DEFAULT_OTHER_PLANT))
    b.add_argument("--dev-min-boxes", type=int, default=DEV_MIN_BOXES,
                   help="boxes every species needs in dev (protocol: %(default)s)")
    b.add_argument("--dev-min-keep", type=float, default=DEV_MIN_KEEP_FRAC,
                   help="share of each species' boxes that must stay in train; not a "
                        "protocol rule, off by default (%(default)s)")
    b.add_argument("--dev-cap-rare", action="store_true",
                   help="a species too rare for both rules gets the largest dev share "
                        "the keep rule allows instead of failing")
    b.add_argument("--max-drop-share", type=float, default=MAX_DROP_SHARE,
                   help="share of an exam's images that may be dropped (or kept as background "
                        "for want of a label file) before build stops (default %(default)s)")
    b.add_argument("--rebuild-locked", action="store_true",
                   help="rebuild although LOCK.json exists (then run `lock --relock`)")
    lk = sub.add_parser("lock", help="write LOCK.json")
    lk.add_argument("--relock", action="store_true", help="replace an existing LOCK.json")
    lk.add_argument("--accept-counts", action="store_true",
                    help="lock although an exam's image count differs from the protocol")
    sub.add_parser("verify", help="re-hash everything against the manifests and LOCK.json")
    sub.add_parser("summary", help="print summary.json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            build(min_dev_boxes=args.dev_min_boxes, min_keep_frac=args.dev_min_keep,
                  cap_rare=args.dev_cap_rare,
                  other_plant_names=other_plant_names(args.other_plant),
                  rebuild_locked=args.rebuild_locked, max_drop_share=args.max_drop_share)
        elif args.cmd == "lock":
            lock(relock=args.relock, accept_counts=args.accept_counts)
        elif args.cmd == "verify":
            probs = verify()
            for p in probs[:200]:
                log("MISMATCH %s" % p)
            if probs:
                log("verify FAILED: %d problem(s)" % len(probs))
                return 1
            log("verify OK: every manifest, image, label, the never-train index, the scorer "
                "and the exam dirs match LOCK.json")
        elif args.cmd == "summary":
            print_summary()
    except SplitError as e:
        log("ERROR: %s" % e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
