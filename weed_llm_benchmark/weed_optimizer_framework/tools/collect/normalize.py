"""Annotation formats -> one form (docs/CONTINUOUS_LOOP.md §3.2 "Normalise").

A source tree (the fetched files, archives extracted) is listed once
(Tree: one os.walk, every directory read once, no rglob), then read by the
first format that recognises it (or the format a known item pins):

  box_csv    a CSV with one row per box: a label column, a file column and
             xmin/ymin/xmax/ymax (or x/y/w/h) in pixels, optionally a group
             column (e.g. a tray or plot id: the capture group). Labels may be
             EPPO codes; the class map resolves them.
  coco       JSON with images, annotations and categories (bbox [x, y, w, h]
             in pixels; a segmentation-only annotation gives its polygon's
             box); several split files are merged by category name.
  weedcoco   coco whose categories carry a taxon (a "taxon" field, or names
             "<role>: <taxon> (<qualifier>)"): the taxon is the class map's hint.
  via        VGG Image Annotator JSON (rect or polygon regions; the class is
             the region attribute named by the options, else the first
             string attribute).
  voc        Pascal VOC XML (object/name, bndbox, size).
  yolo       label .txt files beside or parallel to the images (images/ ->
             labels/), classes from data.yaml names, classes.txt, obj.names
             or _darknet.labels; polygon lines give their bounding box
             (inc.verify.read_source_label, used as a library).
  hf_parquet parquet shards of image bytes and objects {bbox, category}
             (pyarrow; refused with a reason when it is not installed).

The result: the class list in source-id order ({"id", "name", "hints"}), and
one item per annotated image: its path in the tree, boxes as (source class id,
cx, cy, w, h) normalised to the stored pixel grid and clipped to it, the
counts of malformed and clipped boxes, and its capture group (capture_group():
a group column, the options' regex, a video or a capture date in the name,
else the image alone). Images without a label file and label files without an
image are listed, never silently dropped; box rows of a table that name images
the tree does not hold (a partial fetch) are counted. A box that can be placed
but names no listed class (a COCO category id missing from its categories, a
VOC object or VIA region without a name) keeps its place with the source class
id NO_CLASS: intake gives it the unmapped id, so per-box admission masks the
plant instead of leaving it unlabelled in a kept image (§3.2). Nothing here
names a domain.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from . import NormaliseError

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SKIP_DIRS = ("__MACOSX",)
FORMATS = ("box_csv", "weedcoco", "coco", "via", "voc", "yolo", "hf_parquet")
NO_CLASS = -1          # a placed box whose class the source does not list (module docstring)
CLASS_FILES = ("classes.txt", "obj.names", "_darknet.labels", "labels.txt", "classes.names")
_VIDEO = re.compile(r"(?P<clip>.+?\.(mp4|avi|mov|mkv))[_\-.]?\d*$", re.I)
_DATE = re.compile(r"(?<![0-9])(20\d\d)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?![0-9])|(20\d\d)Y(\d\d)M(\d\d)D")
_RF_TAIL = re.compile(r"(_(jpg|jpeg|png))?\.rf\.[0-9a-f]{8,}$", re.I)


# ------------------------------------------------------------------ the tree
class Tree(object):
    """Every file under root, from one walk."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=True):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            for f in sorted(filenames):
                if f.startswith("._") or f == ".DS_Store":
                    continue
                self.files.append(f if rel_dir == "." else "%s/%s" % (rel_dir, f))
        self.files.sort()
        self.images = [f for f in self.files if os.path.splitext(f)[1].lower() in IMG_EXTS]
        self.by_stem = {}
        for f in self.images:
            self.by_stem.setdefault(os.path.splitext(f.split("/")[-1])[0], []).append(f)
        self.image_set = set(self.images)
        self.no_ext = {os.path.splitext(f)[0]: f for f in self.images}

    def path(self, rel):
        return self.root / rel

    def ext(self, *exts):
        exts = tuple(e.lower() for e in exts)
        return [f for f in self.files if f.lower().endswith(exts)]

    def named(self, *names):
        names = set(n.lower() for n in names)
        return [f for f in self.files if f.split("/")[-1].lower() in names]

    def find_image(self, ref, near=None):
        """The image a reference names: an exact relative path (with or
        without its extension), else by base name, preferring one whose path
        ends like the reference, then one under `near`."""
        ref = str(ref or "").replace("\\", "/").lstrip("./")
        if ref in self.image_set:
            return ref
        noext = os.path.splitext(ref)[0] if os.path.splitext(ref)[1].lower() in IMG_EXTS else ref
        if noext in self.no_ext:
            return self.no_ext[noext]
        if near:
            cand = "%s/%s" % (near.rstrip("/"), ref) if near not in (".", "") else ref
            if cand in self.image_set:
                return cand
        stem = os.path.splitext(ref.split("/")[-1])[0] if os.path.splitext(ref)[1].lower() in IMG_EXTS \
            else ref.split("/")[-1]
        hits = self.by_stem.get(stem) or []
        if len(hits) == 1:
            return hits[0]
        if hits:
            tail = [h for h in hits if os.path.splitext(h)[0].endswith(noext)]
            if len(tail) == 1:
                return tail[0]
            if near:
                under = [h for h in hits if h.startswith(near.rstrip("/") + "/")]
                if len(under) == 1:
                    return under[0]
        return None


class _Sizes(object):
    def __init__(self, tree):
        self.tree = tree
        self.cache = {}

    def get(self, rel):
        if rel not in self.cache:
            try:
                from PIL import Image
                with Image.open(self.tree.path(rel)) as im:
                    self.cache[rel] = im.size
            except Exception:  # noqa: BLE001 - an unreadable image has no size
                self.cache[rel] = None
        return self.cache[rel]


def _clip_xyxy(x0, y0, x1, y1, W, H):
    """((cx, cy, w, h) normalised and clipped, clipped?) or (None, _) when degenerate."""
    if not (W and H) or W <= 0 or H <= 0:
        return None, False
    vals = [x0, y0, x1, y1]
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return None, False
    a0, a1 = sorted((x0 / W, x1 / W))
    b0, b1 = sorted((y0 / H, y1 / H))
    c0, c1, d0, d1 = max(0.0, a0), min(1.0, a1), max(0.0, b0), min(1.0, b1)
    clipped = (c0, c1, d0, d1) != (a0, a1, b0, b1)
    if c1 <= c0 or d1 <= d0:
        return None, clipped
    return ((c0 + c1) / 2, (d0 + d1) / 2, c1 - c0, d1 - d0), clipped


def capture_group(rel, opts=None, given=None):
    """(group, basis): a given group column value; else the options' regex;
    else a video in the name (frames "<clip>.mp4_<n>"); else a capture date in
    the name (20210728, 2021-07-28, 2021Y07M28D) with the directory; else the
    image itself (a singleton group, which constrains nothing)."""
    if given not in (None, ""):
        return str(given), "column"
    rx = (opts or {}).get("capture_group_regex")
    if rx:
        m = re.search(rx, rel)
        if m:
            return (m.group("group") if "group" in m.groupdict() else m.group(0)), "regex"
    d, _, name = rel.rpartition("/")
    stem = _RF_TAIL.sub("", os.path.splitext(name)[0])
    m = _VIDEO.search(stem)
    if m:
        return ("%s/%s" % (d, m.group("clip")) if d else m.group("clip")), "video"
    m = _DATE.search(stem)
    if m:
        day = re.sub(r"[^0-9]", "", m.group(0))[:8]
        return ("%s/%s" % (d, day) if d else day), "date"
    return rel, "image"


class Result(object):
    def __init__(self, fmt):
        self.format = fmt
        self.classes = []
        self.items = []
        self.images_without_labels = []
        self.labels_without_images = []
        self.problems = []
        self.ignored_formats = []
        self.rows_without_images = 0

    def summary(self):
        return {"format": self.format, "classes": len(self.classes), "items": len(self.items),
                "boxes": sum(len(i["boxes"]) for i in self.items),
                "bad_boxes": sum(i["bad"] for i in self.items), "clipped": sum(i["clipped"] for i in self.items),
                "images_without_labels": len(self.images_without_labels),
                "labels_without_images": len(self.labels_without_images),
                "rows_without_images": self.rows_without_images, "problems": self.problems[:50],
                "ignored_formats": self.ignored_formats}


def _add_item(res, tree, rel, boxes, bad, clipped, opts, group=None, label_rel=None, wh=None):
    g, basis = capture_group(rel, opts, group)
    res.items.append({"rel": rel, "path": str(tree.path(rel)), "boxes": boxes, "bad": bad, "clipped": clipped,
                      "group": g, "group_basis": basis, "label_rel": label_rel, "wh": wh})


# ------------------------------------------------------------------ box_csv
_COLS = {
    "label": ("label_id", "label", "class", "class_name", "category", "species", "name"),
    "file": ("filename", "file", "file_name", "image", "image_name", "image_id", "path", "img"),
    "xmin": ("xmin", "x_min", "x1", "left"), "ymin": ("ymin", "y_min", "y1", "top"),
    "xmax": ("xmax", "x_max", "x2", "right"), "ymax": ("ymax", "y_max", "y2", "bottom"),
    "x": ("x", "bbox_x"), "y": ("y", "bbox_y"), "w": ("w", "width", "bbox_w", "bbox_width"),
    "h": ("h", "height", "bbox_h", "bbox_height"),
    "group": ("group", "session", "tray_id", "plot", "plot_id", "sequence", "video"),
}


def _csv_cols(header, opts):
    low = {h.strip().lower(): h for h in header}
    out = {}
    for k, names in _COLS.items():
        forced = (opts or {}).get("%s_column" % k)
        if forced and forced in header:
            out[k] = forced
            continue
        for n in names:
            if n in low:
                out[k] = low[n]
                break
    if "label" in out and "file" in out and (all(k in out for k in ("xmin", "ymin", "xmax", "ymax"))
                                             or all(k in out for k in ("x", "y", "w", "h"))):
        return out
    return None


def _box_csv_files(tree, opts):
    out = []
    for f in tree.ext(".csv"):
        try:
            with open(tree.path(f), newline="", encoding="utf-8", errors="replace") as fh:
                header = next(csv.reader(fh), None)
        except OSError:
            continue
        if header and _csv_cols(header, opts):
            out.append(f)
    want = (opts or {}).get("csv")
    if want:
        out = [f for f in out if f == want or f.split("/")[-1] == want]
    return out


def read_box_csv(tree, opts=None):
    files = _box_csv_files(tree, opts)
    if not files:
        raise NormaliseError("no box CSV in the tree")
    res = Result("box_csv")
    sizes = _Sizes(tree)
    per = {}
    names = []
    missing = set()
    for f in files:
        near = f.rpartition("/")[0]
        with open(tree.path(f), newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh)
            cols = _csv_cols(rd.fieldnames or [], opts)
            for row in rd:
                lab = str(row.get(cols["label"]) or "").strip()
                ref = str(row.get(cols["file"]) or "").strip()
                rel = tree.find_image(ref, near)
                if rel is None:
                    missing.add(ref)
                    continue
                if lab not in names:
                    names.append(lab)
                ent = per.setdefault(rel, {"rows": [], "group": row.get(cols["group"]) if "group" in cols else None})
                try:
                    if "xmin" in cols:
                        x0, y0 = float(row[cols["xmin"]]), float(row[cols["ymin"]])
                        x1, y1 = float(row[cols["xmax"]]), float(row[cols["ymax"]])
                    else:
                        x0, y0 = float(row[cols["x"]]), float(row[cols["y"]])
                        x1, y1 = x0 + float(row[cols["w"]]), y0 + float(row[cols["h"]])
                    ent["rows"].append((lab, x0, y0, x1, y1))
                except (TypeError, ValueError):
                    ent["rows"].append((lab, None, None, None, None))
    order = sorted(names)
    res.classes = [{"id": i, "name": n, "hints": []} for i, n in enumerate(order)]
    idx = {n: i for i, n in enumerate(order)}
    for rel in sorted(per):
        wh = sizes.get(rel)
        boxes, bad, clipped = [], 0, 0
        for lab, x0, y0, x1, y1 in per[rel]["rows"]:
            if x0 is None or wh is None:
                bad += 1
                continue
            b, c = _clip_xyxy(x0, y0, x1, y1, wh[0], wh[1])
            clipped += int(c)
            if b is None:
                bad += 1
                continue
            boxes.append((idx[lab],) + b)
        _add_item(res, tree, rel, boxes, bad, clipped, opts, group=per[rel]["group"], label_rel=None, wh=wh)
    res.rows_without_images = len(missing)
    if missing:
        res.problems.append("%d box-table file names are not in the tree (a partial fetch keeps only some groups)"
                            % len(missing))
    listed = set(per)
    res.images_without_labels = [f for f in tree.images if f not in listed]
    return res


# ------------------------------------------------------------------ coco
def _coco_files(tree):
    out = []
    for f in tree.ext(".json"):
        try:
            with open(tree.path(f), encoding="utf-8") as fh:
                head = fh.read(4096)
        except OSError:
            continue
        if '"images"' in head or '"annotations"' in head or '"categories"' in head or len(head) < 4096:
            try:
                with open(tree.path(f), encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and all(isinstance(d.get(k), list) for k in ("images", "annotations",
                                                                                  "categories")):
                out.append(f)
    return out


def _is_weedcoco(docs):
    for d in docs:
        if d.get("agcontexts") is not None:
            return True
        for c in d.get("categories") or []:
            if isinstance(c, dict) and (c.get("taxon") or c.get("role") or re.match(r"^[a-z ]{2,20}:\s*\S",
                                                                                    str(c.get("name") or ""))):
                return True
    return False


def read_coco(tree, opts=None):
    from .classmap import role_taxon_hint
    files = _coco_files(tree)
    if not files:
        raise NormaliseError("no COCO annotation file in the tree")
    docs = []
    for f in files:
        with open(tree.path(f), encoding="utf-8") as fh:
            docs.append((f, json.load(fh)))
    fmt = "weedcoco" if _is_weedcoco([d for _f, d in docs]) else "coco"
    res = Result(fmt)
    sizes = _Sizes(tree)
    names, hints = [], {}
    for _f, d in docs:
        for c in sorted(d["categories"], key=lambda c: (c.get("id") is None, c.get("id"))):
            n = str(c.get("name"))
            if n not in names:
                names.append(n)
            hs = hints.setdefault(n, [])
            for h in (c.get("taxon"), role_taxon_hint(n), c.get("eppo_taxon_code"), c.get("species")):
                if h and str(h) not in hs:
                    hs.append(str(h))
    if len(docs) == 1:
        order = [str(c.get("name")) for c in sorted(docs[0][1]["categories"], key=lambda c: c.get("id") or 0)]
        order = list(dict.fromkeys(order))
    else:
        order = sorted(names)
    res.classes = [{"id": i, "name": n, "hints": hints.get(n, [])} for i, n in enumerate(order)]
    idx = {n: i for i, n in enumerate(order)}
    per = {}
    listed = set()
    for f, d in docs:
        near = f.rpartition("/")[0]
        cat = {c.get("id"): str(c.get("name")) for c in d["categories"]}
        imgs = {}
        for im in d["images"]:
            rel = tree.find_image(im.get("file_name") or im.get("path") or "", near)
            if rel is None:
                res.labels_without_images.append(str(im.get("file_name")))
                continue
            imgs[im.get("id")] = (rel, im.get("width"), im.get("height"))
            listed.add(rel)
            per.setdefault(rel, {"boxes": [], "bad": 0, "clipped": 0, "wh": None})
        for a in d["annotations"]:
            if a.get("image_id") not in imgs:
                continue
            rel, W, H = imgs[a["image_id"]]
            ent = per[rel]
            if not (W and H):
                sz = sizes.get(rel)
                W, H = sz if sz else (None, None)
            ent["wh"] = (W, H) if W and H else None
            bb = a.get("bbox")
            if (not bb or len(bb) != 4) and a.get("segmentation"):
                seg = a["segmentation"]
                pts = [p for poly in (seg if isinstance(seg, list) else []) for p in (poly if isinstance(poly, list) else [])]
                if len(pts) >= 6:
                    xs, ys = pts[0::2], pts[1::2]
                    bb = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            name = cat.get(a.get("category_id"))
            try:
                x, y, w, h = [float(v) for v in bb]
            except (TypeError, ValueError):
                ent["bad"] += 1
                continue
            b, c = _clip_xyxy(x, y, x + w, y + h, W, H)
            ent["clipped"] += int(c)
            if b is None:
                ent["bad"] += 1
                continue
            ent["boxes"].append((idx[name] if name is not None else NO_CLASS,) + b)
    for rel in sorted(per):
        e = per[rel]
        _add_item(res, tree, rel, e["boxes"], e["bad"], e["clipped"], opts, label_rel=None, wh=e["wh"])
    res.images_without_labels = [f for f in tree.images if f not in listed]
    return res


# ------------------------------------------------------------------ via
def _via_files(tree):
    out = []
    for f in tree.ext(".json"):
        try:
            with open(tree.path(f), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("_via_img_metadata"), dict):
            out.append((f, d["_via_img_metadata"]))
        elif isinstance(d, dict) and d and all(isinstance(v, dict) and "filename" in v and "regions" in v
                                               for v in d.values()):
            out.append((f, d))
    return out


def read_via(tree, opts=None):
    files = _via_files(tree)
    if not files:
        raise NormaliseError("no VIA project in the tree")
    res = Result("via")
    sizes = _Sizes(tree)
    attr = (opts or {}).get("via_class_attribute")
    per, names = {}, []
    for f, meta in files:
        near = f.rpartition("/")[0]
        for _k, e in sorted(meta.items()):
            rel = tree.find_image(e.get("filename"), near)
            if rel is None:
                res.labels_without_images.append(str(e.get("filename")))
                continue
            regs = e.get("regions") or []
            regs = list(regs.values()) if isinstance(regs, dict) else regs
            ent = per.setdefault(rel, [])
            for r in regs:
                ra = r.get("region_attributes") or {}
                lab = ra.get(attr) if attr else next((v for v in ra.values() if isinstance(v, str) and v.strip()), None)
                if isinstance(lab, dict):
                    lab = next((k for k, v in lab.items() if v), None)
                sa = r.get("shape_attributes") or {}
                ent.append((None if lab is None else str(lab).strip(), sa))
                if lab is not None and str(lab).strip() not in names:
                    names.append(str(lab).strip())
    order = sorted(names)
    res.classes = [{"id": i, "name": n, "hints": []} for i, n in enumerate(order)]
    idx = {n: i for i, n in enumerate(order)}
    for rel in sorted(per):
        wh = sizes.get(rel)
        boxes, bad, clipped = [], 0, 0
        for lab, sa in per[rel]:
            if wh is None:
                bad += 1
                continue
            try:
                if sa.get("name") == "rect":
                    x0, y0 = float(sa["x"]), float(sa["y"])
                    x1, y1 = x0 + float(sa["width"]), y0 + float(sa["height"])
                elif sa.get("all_points_x"):
                    xs, ys = [float(v) for v in sa["all_points_x"]], [float(v) for v in sa["all_points_y"]]
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                elif sa.get("name") in ("circle", "ellipse"):
                    rx = float(sa.get("r") or sa.get("rx"))
                    ry = float(sa.get("r") or sa.get("ry"))
                    x0, x1 = float(sa["cx"]) - rx, float(sa["cx"]) + rx
                    y0, y1 = float(sa["cy"]) - ry, float(sa["cy"]) + ry
                else:
                    bad += 1
                    continue
            except (KeyError, TypeError, ValueError):
                bad += 1
                continue
            b, c = _clip_xyxy(x0, y0, x1, y1, wh[0], wh[1])
            clipped += int(c)
            if b is None:
                bad += 1
                continue
            boxes.append((idx[lab] if lab else NO_CLASS,) + b)
        _add_item(res, tree, rel, boxes, bad, clipped, opts, wh=wh)
    listed = set(per)
    res.images_without_labels = [f for f in tree.images if f not in listed]
    return res


# ------------------------------------------------------------------ voc
def _voc_files(tree):
    out = []
    for f in tree.ext(".xml"):
        try:
            root = ET.parse(tree.path(f)).getroot()
        except (ET.ParseError, OSError):
            continue
        if root.tag == "annotation":
            out.append((f, root))
    return out


def read_voc(tree, opts=None):
    files = _voc_files(tree)
    if not files:
        raise NormaliseError("no Pascal VOC file in the tree")
    res = Result("voc")
    sizes = _Sizes(tree)
    per, names = {}, []
    for f, root in files:
        near = f.rpartition("/")[0]
        fname = (root.findtext("filename") or "").strip()
        rel = tree.find_image(fname, near) if fname else None
        if rel is None:
            stem = os.path.splitext(f.split("/")[-1])[0]
            rel = tree.find_image(stem, near)
        if rel is None:
            res.labels_without_images.append(f)
            continue
        W = H = None
        try:
            W = float(root.findtext("size/width") or 0) or None
            H = float(root.findtext("size/height") or 0) or None
        except ValueError:
            pass
        if not (W and H):
            sz = sizes.get(rel)
            W, H = sz if sz else (None, None)
        ent = per.setdefault(rel, {"objs": [], "wh": (W, H) if W and H else None, "label": f})
        for ob in root.findall("object"):
            n = (ob.findtext("name") or "").strip()
            bb = ob.find("bndbox")
            try:
                x0, y0 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
                x1, y1 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            except (AttributeError, TypeError, ValueError):
                ent["objs"].append((n, None))
                continue
            ent["objs"].append((n, (x0, y0, x1, y1)))
            if n and n not in names:
                names.append(n)
    order = sorted(names)
    res.classes = [{"id": i, "name": n, "hints": []} for i, n in enumerate(order)]
    idx = {n: i for i, n in enumerate(order)}
    for rel in sorted(per):
        e = per[rel]
        boxes, bad, clipped = [], 0, 0
        for n, bb in e["objs"]:
            if bb is None or e["wh"] is None:
                bad += 1
                continue
            b, c = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], e["wh"][0], e["wh"][1])
            clipped += int(c)
            if b is None:
                bad += 1
                continue
            boxes.append((idx[n] if n else NO_CLASS,) + b)
        _add_item(res, tree, rel, boxes, bad, clipped, opts, label_rel=e["label"], wh=e["wh"])
    listed = set(per)
    res.images_without_labels = [f for f in tree.images if f not in listed]
    return res


# ------------------------------------------------------------------ yolo
def _yaml_names(path):
    """The class names of a data.yaml (list or {id: name}), else None."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        import yaml
        d = yaml.safe_load(text)
        names = (d or {}).get("names") if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001 - no PyYAML, or not YAML: the small parser below
        names = None
        m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            names = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        else:
            m = re.search(r"^names\s*:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
            if m:
                block = m.group(1).splitlines()
                if all(re.match(r"^\s*-\s", ln) for ln in block if ln.strip()):
                    names = [re.sub(r"^\s*-\s*", "", ln).strip().strip("'\"") for ln in block if ln.strip()]
                else:
                    names = {}
                    for ln in block:
                        mm = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", ln)
                        if mm:
                            names[int(mm.group(1))] = mm.group(2).strip("'\"")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def _yolo_label_for(img_rel, label_set):
    d, _, name = img_rel.rpartition("/")
    stem = os.path.splitext(name)[0]
    cands = []
    parts = img_rel.split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cands.append("/".join(parts[:i] + ["labels"] + parts[i + 1:-1] + [stem + ".txt"]))
    cands.append(("%s/%s.txt" % (d, stem)) if d else "%s.txt" % stem)
    for c in cands:
        if c in label_set:
            return c
    return None


def read_yolo(tree, opts=None):
    from ..inc import verify as V
    txts = [f for f in tree.ext(".txt") if f.split("/")[-1].lower() not in CLASS_FILES
            and not f.split("/")[-1].lower().startswith(("readme", "license", "licence"))]
    label_set = set(txts)
    pairs = []
    used = set()
    for img in tree.images:
        lab = _yolo_label_for(img, label_set)
        if lab is not None:
            pairs.append((img, lab))
            used.add(lab)
    if not pairs:
        raise NormaliseError("no YOLO label file beside or parallel to an image")
    names = None
    ys = sorted(tree.named("data.yaml", "dataset.yaml", "data.yml"), key=lambda f: (f.count("/"), f))
    for y in ys:
        names = _yaml_names(tree.path(y))
        if names:
            break
    if not names:
        for cf in sorted(tree.named(*CLASS_FILES), key=lambda f: (f.count("/"), f)):
            lines = [ln.strip() for ln in tree.path(cf).read_text(encoding="utf-8", errors="replace").splitlines()]
            names = [ln for ln in lines if ln]
            if names:
                break
    res = Result("yolo")
    res.classes = [{"id": i, "name": n, "hints": []} for i, n in enumerate(names or [])]
    seen_ids = set()
    for img, lab in pairs:
        boxes, bad, clipped = V.read_source_label(tree.path(lab))
        seen_ids.update(b[0] for b in boxes)
        _add_item(res, tree, img, [tuple(b) for b in boxes], bad, clipped, opts, label_rel=lab)
    if not names:
        res.classes = [{"id": i, "name": str(i), "hints": []} for i in range(max(seen_ids) + 1)] if seen_ids else []
        res.problems.append("no class names file: ids are their own names (numeric, unmapped without a card)")
    else:
        extra = sorted(i for i in seen_ids if i >= len(names))
        if extra:
            res.problems.append("label ids %s are beyond the %d class names" % (extra[:10], len(names)))
            res.classes += [{"id": i, "name": str(i), "hints": []} for i in range(len(names), max(extra) + 1)]
    paired = {p[0] for p in pairs}
    res.images_without_labels = [f for f in tree.images if f not in paired]
    res.labels_without_images = sorted(label_set - used)
    return res


# ------------------------------------------------------------------ hf parquet
def read_hf_parquet(tree, opts=None, out_images=None):
    """Parquet shards with an image column ({bytes, path}) and objects
    ({bbox, category}); bbox_format "xywh" (default) or "xyxy", in pixels.
    Images are written under out_images (a directory the caller owns)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise NormaliseError("hf_parquet needs pyarrow, which is not installed here")
    files = tree.ext(".parquet")
    if not files:
        raise NormaliseError("no parquet shard in the tree")
    if out_images is None:
        raise NormaliseError("hf_parquet needs a directory for the images it writes")
    out_images = Path(out_images)
    fmt = (opts or {}).get("bbox_format", "xywh")
    names = list((opts or {}).get("class_names") or [])
    res = Result("hf_parquet")
    items = []
    seen = set()
    for f in files:
        t = pq.read_table(tree.path(f))
        cols = t.column_names
        icol = "image" if "image" in cols else next((c for c in cols if "image" in c), None)
        ocol = "objects" if "objects" in cols else None
        if icol is None or ocol is None:
            res.problems.append("%s: no image/objects columns (%s)" % (f, cols))
            continue
        for i, row in enumerate(t.to_pylist()):
            im = row.get(icol) or {}
            data = im.get("bytes") if isinstance(im, dict) else None
            if not data:
                continue
            ext = os.path.splitext(str((im or {}).get("path") or ""))[1].lower() or ".jpg"
            rel = "%s/%06d%s" % (os.path.splitext(f)[0].replace("/", "__"), i, ext if ext in IMG_EXTS else ".jpg")
            p = out_images / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            try:
                from PIL import Image
                with Image.open(io.BytesIO(data)) as pim:
                    W, H = pim.size
            except Exception:  # noqa: BLE001
                W = H = None
            ob = row.get(ocol) or {}
            cats = ob.get("categories") or ob.get("category") or []
            boxes, bad, clipped = [], 0, 0
            for bb, c in zip(ob.get("bbox") or [], cats):
                try:
                    x, y, a, b = [float(v) for v in bb]
                except (TypeError, ValueError):
                    bad += 1
                    continue
                x1, y1 = (a, b) if fmt == "xyxy" else (x + a, y + b)
                bx, cl = _clip_xyxy(x, y, x1, y1, W, H)
                clipped += int(cl)
                if bx is None:
                    bad += 1
                    continue
                seen.add(int(c))
                boxes.append((int(c),) + bx)
            items.append((rel, boxes, bad, clipped, (W, H) if W else None))
    sub = Tree(out_images)
    for rel, boxes, bad, clipped, wh in items:
        _add_item(res, sub, rel, boxes, bad, clipped, opts, wh=wh)
    n = max(len(names), (max(seen) + 1) if seen else 0)
    res.classes = [{"id": i, "name": names[i] if i < len(names) else str(i), "hints": []} for i in range(n)]
    return res


# ------------------------------------------------------------------ dispatch
_DETECT = (
    ("box_csv", lambda t, o: bool(_box_csv_files(t, o))),
    ("coco", lambda t, o: bool(_coco_files(t))),
    ("via", lambda t, o: bool(_via_files(t))),
    ("voc", lambda t, o: bool(_voc_files(t))),
    ("yolo", lambda t, o: bool(t.images) and any(_yolo_label_for(i, set(t.ext(".txt"))) for i in t.images[:200])),
    ("hf_parquet", lambda t, o: bool(t.ext(".parquet"))),
)


def detect(tree, opts=None):
    """The formats the tree holds, in dispatch order."""
    out = []
    for name, fn in _DETECT:
        try:
            if fn(tree, opts):
                out.append(name)
        except Exception:  # noqa: BLE001 - a probe that fails means "not this format"
            continue
    return out


def read(root, opts=None, out_images=None):
    """(Tree, Result): the tree under root read by the format opts["format"]
    names, else the first format detect() finds."""
    tree = Tree(root)
    opts = dict(opts or {})
    found = detect(tree, opts)
    fmt = opts.get("format")
    if fmt in ("weedcoco",):
        fmt = "coco"
    if fmt is None:
        if not found:
            raise NormaliseError("no known annotation format under %s (%d files, %d images)"
                                 % (root, len(tree.files), len(tree.images)))
        fmt = found[0]
    if fmt not in ("box_csv", "coco", "via", "voc", "yolo", "hf_parquet"):
        raise NormaliseError("unknown format %r (known: %s)" % (fmt, FORMATS))
    if fmt == "box_csv":
        res = read_box_csv(tree, opts)
    elif fmt == "coco":
        res = read_coco(tree, opts)
    elif fmt == "via":
        res = read_via(tree, opts)
    elif fmt == "voc":
        res = read_voc(tree, opts)
    elif fmt == "yolo":
        res = read_yolo(tree, opts)
    else:
        res = read_hf_parquet(tree, opts, out_images=out_images)
    res.ignored_formats = [f for f in found if f != fmt]
    return tree, res
