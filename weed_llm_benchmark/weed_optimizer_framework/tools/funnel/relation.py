"""Label spaces resolved against evidence (contract §6 H0b, H0c, H1-pre, H3a;
lever L11, §8.5; runner §4.14, §5.2.4). Verb `map`:

  --part geometry  (F4) Each source whose card resolver names upstream
                   annotations (fetched and hashed by `fetch --what cards`) is
                   matched to the pool's own label files by box geometry. The
                   matched boxes pair a pool class id with an upstream class id
                   (or name). For a source with a card class table the id
                   alignment is chosen among identity, +1, -1 and the "drop
                   class d" maps on a seeded half of the matched images and
                   confirmed on the other half (H3a). For a source whose card
                   is authoritative the pool's class names are compared with the
                   upstream class names box by box (H1-pre: an off-by-one
                   class order shows up here, before any reference label).
                   Writes relation_geometry_v1.json and the class-map proposals
                   class_maps.json (immutable after F4; applying a map is a
                   recovery decision, recorded in recovery.json).
  --part relation  (F5) The relation audit (Missing Link, UniDet): a class's
                   relation to class k is the mean judge probability of k over
                   the class's crops, with names hidden. H0(b) runs it on copies
                   of reference photographs with a known truth (a pipeline
                   check), H0(c) on a shifted named source (a named relative
                   must not map to a target), and the no-information classes
                   get visual proposals for the person queue (L14).
                   Writes relation_audit_v1.json.

Which sources play which role comes from the domain config
(sources.card_resolvers, sources.relation_checks, known_truth); the pool's
labels come from the adapter. Nothing here names a domain.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..inc import common as C
from . import (RelationError, header, read_json, strip_volatile, write_json_atomic)
from . import domain as D
from .names import key as name_key

TOL_PX = 2
MAP_MIN = 0.5
FLAG_MAX = 0.2
MIN_BOXES = 20
# name statuses (names.py) whose name says "not a target"; the resolved ones
# name a taxon, and H0(c) fails when such a class is mapped to a target
RESOLVED_NON_TARGET = ("target_related", "taxon_resolved")
NAMED_NON_TARGET = RESOLVED_NON_TARGET + ("role",)
STEP = 0.01                       # centre grid of the image-pairing index (normalised units)


# ------------------------------------------------------------------ upstream
class Upstream(dict):
    """{file key: [(id or name, x0, y0, x1, y1, W, H)]}. Pixel coordinates when
    W and H are known, else normalised (W = H = None). .quantum[file] is the
    normalised rounding step of the file's coordinates (0 for exact pixels),
    .names[id] the class names a classes file gives, .frames[file] the frame
    name the annotation records."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.quantum = {}
        self.names = {}
        self.frames = {}


class _Tree(object):
    """Files of a zip archive or of a directory tree, by relative path."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.zip = None
            self.files = sorted(p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file())
        elif zipfile.is_zipfile(self.path):
            self.zip = zipfile.ZipFile(self.path)
            self.files = sorted(n for n in self.zip.namelist() if not n.endswith("/"))
        else:
            raise RelationError("%s is neither a directory nor a zip archive" % path)

    def read(self, rel):
        if self.zip is not None:
            return self.zip.read(rel)
        with open(self.path / rel, "rb") as fh:
            return fh.read()


def _decimals(tok):
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def _stem(rel):
    return os.path.splitext(rel.split("/")[-1])[0]


def _parse_voc(data):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise RelationError("unreadable VOC annotation (%s)" % e)
    size = root.find("size")
    W = int(float(size.findtext("width"))) if size is not None else None
    H = int(float(size.findtext("height"))) if size is not None else None
    boxes = []
    for obj in root.findall("object"):
        nm = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0, y0, x1, y1 = (float(bb.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
        if W and H:
            x0, x1 = max(0.0, min(x0, W)), max(0.0, min(x1, W))
            y0, y1 = max(0.0, min(y0, H)), max(0.0, min(y1, H))
        boxes.append((nm, x0, y0, x1, y1, W, H))
    return boxes, root.findtext("filename")


def _parse_yolo(text):
    """(boxes, rounding step): the step is half a unit of the finest decimal the
    file writes (a writer that drops trailing zeros writes 0.2 for 0.20)."""
    boxes, dec = [], 0
    for ln in text.splitlines():
        t = ln.split()
        if len(t) < 5:
            continue
        try:
            c = int(float(t[0]))
            cx, cy, w, h = (float(x) for x in t[1:5])
        except ValueError:
            continue
        dec = max(dec, max(_decimals(x) for x in t[1:5]))
        x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
        y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
        boxes.append((c, x0, y0, x1, y1, None, None))
    return boxes, (0.5 * 10 ** (-dec) if boxes else 0.0)


def read_upstream(archive_path, kind, classes_file=None, voc_dir=None, yolo_dir=None, frame_name_regex=None):
    """The upstream annotations of an archive (zip) or a directory of fetched
    files. kind: "voc" (names, pixel boxes), "yolo" (ids, normalised boxes;
    classes_file names the ids), "voc+yolo" (pixel boxes from VOC, each box's
    id from the YOLO file of the same stem: the nearest box centre)."""
    tree = _Tree(archive_path)
    up = Upstream()
    voc = [f for f in tree.files if f.lower().endswith(".xml") and (voc_dir is None or ("/%s/" % voc_dir) in "/" + f)]
    yol = [f for f in tree.files if f.lower().endswith(".txt") and (yolo_dir is None or ("/%s/" % yolo_dir) in "/" + f)
           and (classes_file is None or f.split("/")[-1] != classes_file)]
    if classes_file:
        cf = [f for f in tree.files if f.split("/")[-1] == classes_file]
        tables = {}
        for f in cf:
            names = [ln.strip() for ln in tree.read(f).decode("utf-8", "replace").splitlines() if ln.strip()]
            tables.setdefault(tuple(names), []).append(f)
        if len(tables) > 1:
            # one id -> name table cannot describe folders that number their classes differently
            raise RelationError("%s: the %s files disagree (%s); the ids cannot be read with one class table"
                                % (archive_path, classes_file, sorted(v[0] for v in tables.values())))
        if tables:
            names = list(next(iter(tables)))
            up.names = {i: n for i, n in enumerate(names)}
    rx = re.compile(frame_name_regex) if frame_name_regex else None
    if kind == "voc":
        for f in voc:
            boxes, fn = _parse_voc(tree.read(f))
            up[_stem(f)] = boxes
            up.quantum[_stem(f)] = 0.0
            up.frames[_stem(f)] = fn
    elif kind == "yolo":
        for f in yol:
            boxes, q = _parse_yolo(tree.read(f).decode("utf-8", "replace"))
            k = _stem(f)
            if k in up:
                raise RelationError("two upstream label files share the stem %r" % k)
            up[k] = boxes
            up.quantum[k] = q
    elif kind == "voc+yolo":
        ystem = {_stem(f): f for f in yol}
        for f in voc:
            k = _stem(f)
            boxes, fn = _parse_voc(tree.read(f))
            up.frames[k] = fn
            if k not in ystem:
                continue
            yb, _q = _parse_yolo(tree.read(ystem[k]).decode("utf-8", "replace"))
            out = []
            for (nm, x0, y0, x1, y1, W, H) in boxes:
                if not W or not H or not yb:
                    continue
                cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                best = min(yb, key=lambda b: abs((b[1] + b[3]) / 2 - cx) + abs((b[2] + b[4]) / 2 - cy))
                out.append((int(best[0]), x0, y0, x1, y1, W, H))
                up.names.setdefault(int(best[0]), collections.Counter())[nm] += 1
            up[k] = out
            up.quantum[k] = 0.0
        up.names = {i: c.most_common(1)[0][0] for i, c in up.names.items()}
    else:
        raise RelationError("upstream kind %r is not voc, yolo or voc+yolo" % kind)
    if rx is not None:
        up.stems = {k: (rx.match(fn or "").group("stem") if fn and rx.match(fn) else None)
                    for k, fn in up.frames.items()}
    return up


# ------------------------------------------------------------------ geometry
def _frame(pool_box, up_W, up_H):
    """A pool box (cls, cx, cy, w, h, W, H) in the frame upstream boxes use."""
    cls, cx, cy, w, h, W, H = pool_box
    if up_W and up_H:
        return (cls, (cx - w / 2) * up_W, (cy - h / 2) * up_H, (cx + w / 2) * up_W, (cy + h / 2) * up_H)
    return (cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _match_boxes(pboxes, uboxes, tol_px, quantum):
    """Greedy one-to-one match, nearest first. Returns [(pool i, up j)]."""
    if not pboxes or not uboxes:
        return []
    upW, upH = uboxes[0][5], uboxes[0][6]
    if upW and upH:
        tolx = toly = tol_px
        conv_u = [(u[1], u[2], u[3], u[4]) for u in uboxes]
        conv_p = [_frame(p, upW, upH)[1:] for p in pboxes]
    else:
        W, H = pboxes[0][5], pboxes[0][6]
        tolx = tol_px / float(W) + quantum
        toly = tol_px / float(H) + quantum
        conv_u = [(u[1], u[2], u[3], u[4]) for u in uboxes]
        conv_p = [_frame(p, None, None)[1:] for p in pboxes]
    cand = []
    for i, pb in enumerate(conv_p):
        for j, ub in enumerate(conv_u):
            dx = max(abs(pb[0] - ub[0]), abs(pb[2] - ub[2]))
            dy = max(abs(pb[1] - ub[1]), abs(pb[3] - ub[3]))
            if dx <= tolx + 1e-9 and dy <= toly + 1e-9:
                cand.append((dx / max(tolx, 1e-12) + dy / max(toly, 1e-12), i, j))
    cand.sort()
    used_i, used_j, out = set(), set(), []
    for _d, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return sorted(out)


def geometry_match(pool_labels, upstream, tol_px=TOL_PX):
    """Pair pool images with upstream files and their boxes.

    pool_labels: {image key: [(cls, cx, cy, w, h, W, H)]} (normalised YOLO
    boxes, the pool image size). upstream: {file: [(id, x0, y0, x1, y1, W, H)]}
    (read_upstream). A pool image pairs with the upstream file whose boxes
    match all of its boxes (within tol_px in the upstream frame, plus the
    upstream's own rounding step); the upstream may hold more boxes (a class
    the pool's export dropped). Candidates come from a grid index of box
    centres; ties go to the lower file key and are counted as ambiguous."""
    quantum = getattr(upstream, "quantum", {}) or {}
    index = collections.defaultdict(set)
    for f, boxes in upstream.items():
        for b in boxes:
            W, H = b[5], b[6]
            cx = ((b[1] + b[3]) / 2 / W) if W else (b[1] + b[3]) / 2
            cy = ((b[2] + b[4]) / 2 / H) if H else (b[2] + b[4]) / 2
            index[(int(round(cx / STEP)), int(round(cy / STEP)))].add(f)
    image_pairs, box_pairs, ambiguous, unmatched = {}, [], [], []
    for key in sorted(pool_labels):
        pb = pool_labels[key]
        if not pb:
            continue
        big = max(pb, key=lambda b: b[3] * b[4])
        gx, gy = int(round(big[1] / STEP)), int(round(big[2] / STEP))
        cands = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cands |= index.get((gx + dx, gy + dy), set())
        best, tie, pairs = None, False, []
        for f in sorted(cands):
            m = _match_boxes(pb, upstream[f], tol_px, quantum.get(f, 0.0))
            if len(m) == len(pb):
                if best is None or len(upstream[f]) < len(upstream[best]):
                    tie = best is not None and len(upstream[f]) == len(upstream[best])
                    best, pairs = f, m
                elif len(upstream[f]) == len(upstream[best]):
                    tie = True
        if best is None:
            unmatched.append(key)
            continue
        if tie:
            ambiguous.append(key)
        image_pairs[key] = best
        for i, j in pairs:
            box_pairs.append({"pool_key": key, "pool_box": i, "pool_id": pb[i][0], "file": best, "up_box": j,
                              "up_id": upstream[best][j][0]})
    return {"image_pairs": image_pairs, "box_pairs": box_pairs, "ambiguous": ambiguous, "unmatched": unmatched}


def alignments(n_upstream_classes):
    """{name: {upstream id: pool id | None}}: identity, plus1, minus1 and
    drop<d> (the pool's export dropped upstream class d: ids above d move down
    by one) for every d in 0..n-1."""
    n = int(n_upstream_classes)
    out = {"identity": {u: u for u in range(n)},
           "plus1": {u: u + 1 for u in range(n)},
           "minus1": {u: (u - 1 if u > 0 else None) for u in range(n)}}
    for d in range(n):
        out["drop%d" % d] = {u: (u if u < d else (None if u == d else u - 1)) for u in range(n)}
    return out


def _order(names):
    base = {"identity": 0, "plus1": 1, "minus1": 2}
    return sorted(names, key=lambda n: (base.get(n, 3), int(n[4:]) if n.startswith("drop") else 0))


def choose_alignment(box_pairs, aligns, seed_text, n_pool_classes=None, min_agreement=None):
    """The alignment that best explains the matched boxes: chosen on a seeded
    half of the matched images (share of box pairs whose pool id equals the
    alignment of the upstream id), confirmed when it is also the best on the
    other half. Ties (alignments the matched ids cannot tell apart) go to one
    whose kept classes number n_pool_classes, then to the one that moves the
    fewest ids, then to the order identity, plus1, minus1, drop0.. ."""
    keys = sorted({bp["pool_key"] for bp in box_pairs})
    import numpy as np
    perm = np.random.default_rng(C.stable_int(seed_text)).permutation(len(keys))
    half_a = {keys[i] for i in perm[:(len(keys) + 1) // 2]}
    pa = [bp for bp in box_pairs if bp["pool_key"] in half_a]
    pb = [bp for bp in box_pairs if bp["pool_key"] not in half_a]

    def agree(amap, pairs):
        if not pairs:
            return 0.0
        ok = sum(1 for bp in pairs if isinstance(bp["up_id"], int) and amap.get(bp["up_id"]) == bp["pool_id"])
        return ok / float(len(pairs))

    def kept(amap):
        return len({v for v in amap.values() if v is not None})

    def moved(amap):
        return sum(1 for u, v in amap.items() if v is not None and v != u)
    rows = []
    for name in _order(aligns):
        rows.append({"name": name, "agreement_a": round(agree(aligns[name], pa), 6),
                     "agreement_b": round(agree(aligns[name], pb), 6), "n_a": len(pa), "n_b": len(pb),
                     "kept_classes": kept(aligns[name]), "moved_ids": moved(aligns[name])})
    rank = {n: i for i, n in enumerate(_order(aligns))}

    def best(field):
        return sorted(rows, key=lambda r: (-r[field], 0 if (n_pool_classes is not None and
                                                              r["kept_classes"] == n_pool_classes) else 1,
                                           r["moved_ids"], rank[r["name"]]))[0]
    ca, cb = best("agreement_a"), best("agreement_b")
    confirmed = ca["agreement_b"] >= cb["agreement_b"] - 1e-12
    if min_agreement is not None:
        confirmed = confirmed and ca["agreement_b"] >= min_agreement
    hkeys = lambda ks: hashlib.sha256("\n".join(sorted(ks)).encode("utf-8")).hexdigest()
    return {"alignments": rows, "chosen": ca["name"], "confirmed": bool(confirmed),
            "halves": {"seed_text": seed_text, "a": hkeys(half_a), "b": hkeys(set(keys) - half_a),
                       "n_a": len(half_a), "n_b": len(keys) - len(half_a)}}


# ------------------------------------------------------------------ relation audit
def relation_scores(unit_crop_ids, P, labels, row_of=None):
    """{unit: {class: mean P over the unit's crops}}. P is indexed by crop id,
    or by row_of[crop id] when given (a judge score file's own rows)."""
    import numpy as np
    out = {}
    for unit, ids in sorted(unit_crop_ids.items()):
        rows = [row_of[c] if row_of is not None else c for c in ids if row_of is None or c in row_of]
        if not rows:
            out[unit] = {}
            continue
        M = np.asarray(P[rows], dtype=np.float64)
        M = M[np.isfinite(M).all(axis=1)]
        out[unit] = {lab: round(float(M[:, k].mean()), 6) for k, lab in enumerate(labels)} if len(M) else {}
    return out


def relation_audit(units, scores, joins, truth=None, map_min=MAP_MIN, flag_max=FLAG_MAX, min_boxes=MIN_BOXES):
    """Map and flag units (Missing Link rule): u maps to k when k is its argmax
    relation, r(u, k) >= map_min and u has >= min_boxes boxes; u's join is
    flagged when r(u, join) < flag_max and another class has r >= map_min.
    units: {unit: boxes}; joins: {unit: joined class}; truth: {unit: class}."""
    out = []
    for u in sorted(units):
        r = scores.get(u) or {}
        mapped = None
        if r and units[u] >= min_boxes:
            k, v = sorted(r.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if v >= map_min:
                mapped = k
        j = joins.get(u)
        flagged = bool(r) and r.get(j, 0.0) < flag_max and any(v >= map_min for c, v in r.items() if c != j)
        out.append({"unit": u, "boxes": int(units[u]), "truth": (truth or {}).get(u), "r": r, "mapped_to": mapped,
                    "join": j, "join_flagged": flagged})
    return out


# ------------------------------------------------------------------ proposals
def _target_of_taxon(domain, taxon, resolver=None):
    if not taxon:
        return None
    if resolver is not None:
        e = resolver.resolve(taxon, scientific_only=True)
        for t in domain.targets:
            if e is not None and resolver.is_target_synonym(e, t):
                return t["name"]
        return None
    for t in domain.targets:
        if taxon == t.get("taxon"):
            return t["name"]
        if t.get("rank") == "genus" and taxon.split(" ")[0] == t.get("taxon"):
            return t["name"]
    return None


def card_name_check(table, up_names):
    """The card class table against the class names the upstream annotations
    carry, id by id. A card's ids are read as upstream ids, so a card whose
    names differ from the upstream's own is not describing these files, and
    no map may be read from it (contract §8.5 L11: a map is accepted only
    where the card and the geometry match agree)."""
    if not up_names:
        return {"checked": False, "agree": [], "disagree": [],
                "why": "the upstream annotations carry no class names to compare with the card"}
    agree, disagree = [], []
    for u in sorted(up_names, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x))):
        row = table.get(str(u)) if isinstance(table, dict) else None
        card = (row or {}).get("name") if isinstance(row, dict) else None
        if card is not None and name_key(card) == name_key(up_names[u]):
            agree.append(u)
        else:
            disagree.append({"up_id": u, "upstream": up_names[u], "card": card})
    return {"checked": True, "agree": agree, "disagree": disagree}


def _card_names_ok(mt):
    cn = (mt or {}).get("card_names") or {}
    return bool(cn.get("checked")) and not cn.get("disagree")


def card_proposals(domain, geometry, pool_summary, resolver=None, name_status=None):
    """Class-map proposals (class_maps.json): card + geometry (a card class
    table read through the chosen, confirmed id alignment), and taxonomy (a
    name the authority resolves as a target synonym, from name_status_v2).
    A source whose class count, under the chosen alignment, differs from the
    card table's goes to the person queue (to_L14)."""
    props = []
    res_all = (domain.raw["sources"].get("card_resolvers") or {})
    for slug in sorted(res_all):
        res = res_all[slug] or {}
        table = res.get("class_table") or {}
        mt = ((geometry or {}).get("matches") or {}).get(slug)
        st = (pool_summary.get("per_slug") or {}).get(slug)
        if not table or not mt or st is None or mt.get("mode") != "ids":
            continue
        join = st.get("join") or {}
        amap = alignments(len(table)).get(mt.get("chosen")) or {}
        inv = {v: u for u, v in amap.items() if v is not None}
        kept = len(inv)
        count_ok = kept == len([k for k in join if k != "*"])
        for sid in sorted(join, key=lambda s: (len(s), s)):
            if sid == "*":
                continue
            u = inv.get(int(sid)) if sid.isdigit() else None
            row = table.get(str(u)) if u is not None else None
            target = _target_of_taxon(domain, row.get("taxon") if row else None, resolver)
            status, reason = "proposed", "card class %s (%s) through alignment %s" % (
                u, row.get("taxon") if row else None, mt.get("chosen"))
            if not mt.get("confirmed"):
                status, reason = "to_L14", "the id alignment was not confirmed on the second half"
            elif not count_ok:
                status, reason = "to_L14", ("the card table keeps %d classes under %s, the source has %d"
                                            % (kept, mt.get("chosen"), len(join)))
            elif not _card_names_ok(mt):
                cn = mt.get("card_names") or {}
                status, reason = "to_L14", (
                    "the upstream class names disagree with the card table: %s" % cn.get("disagree")[:5]
                    if cn.get("checked") else "the card table was not compared with the upstream class names"
                    " (%s)" % cn.get("why", "no record"))
            props.append({"source": slug, "src_id": sid, "src_name": join[sid][0], "map_to": target,
                          "via": "card+geometry",
                          "card": {"path": mt.get("card_path"), "sha256": mt.get("card_sha256"),
                                   "table_source": res.get("table_source")},
                          "geometry": {"alignment": mt.get("chosen"), "agreement": mt.get("agreement_b")},
                          "status": status, "reason": reason})
    for n in (name_status or {}).get("names", []):
        if n.get("status_v2") != "target_synonym":
            continue
        target = _target_of_taxon(domain, n.get("taxon"), resolver)
        props.append({"source": n["source"], "src_id": n["src_id"], "src_name": n["name"], "map_to": target,
                      "via": "taxonomy", "card": None, "geometry": None,
                      "status": "proposed" if target else "to_L14",
                      "reason": "the authority resolves %r to %s (%s)" % (n["name"], n.get("taxon"), n.get("via"))})
    props.sort(key=lambda p: (p["source"], len(p["src_id"]), p["src_id"], p["via"]))
    return props


# ------------------------------------------------------------------ runs
def _load(prereg, domain):
    pre = prereg if hasattr(prereg, "core_sha256") else D.load_prereg(prereg)
    dom = D.load(domain if domain is not None else pre.domain_name)
    D.check_prereg_domain(pre, dom)
    return pre, dom


def _cards_index(funnel_dir):
    p = Path(funnel_dir) / "cards" / "index.json"
    if not p.exists():
        raise RelationError("no cards/index.json in %s: run fetch --what cards (lever L11a)" % funnel_dir)
    return read_json(p)


def _upstream_file(funnel_dir, cards, slug):
    ents = [e for e in (cards.get("cards") or {}).get(slug, []) if e.get("what") == "annotations"]
    if not ents:
        return None
    if len(ents) == 1 and ents[0].get("file"):
        p = Path(funnel_dir) / "cards" / ents[0]["file"]
        if not p.exists():
            raise RelationError("fetched annotations %s are missing" % p)
        if C.sha256_file(p) != ents[0]["sha256"]:
            raise RelationError("fetched annotations %s changed since fetch" % p)
        return p, ents[0]["sha256"]
    # many files (one per upstream label file): the slug's annotations directory
    d = Path(funnel_dir) / "cards" / slug / "annotations"
    for e in ents:
        p = Path(funnel_dir) / "cards" / e["file"]
        if not p.exists() or C.sha256_file(p) != e["sha256"]:
            raise RelationError("fetched annotation %s is missing or changed" % p)
    digest = hashlib.sha256("".join(sorted(e["sha256"] for e in ents)).encode("utf-8")).hexdigest()
    return d, digest


def run_geometry(prereg, domain, funnel_dir, adapter, force=False, testing=False):
    """F4 `map --part geometry`: relation_geometry_v1.json and class_maps.json."""
    pre, dom = _load(prereg, domain)
    fd = Path(funnel_dir)
    cards = _cards_index(fd)
    # the taxonomy proposals come from the frozen name status (census, F3); class_maps.json
    # is immutable once written, so it is never written without them
    nsp = fd / "name_status_v2.json"
    if not nsp.exists():
        raise RelationError("no name_status_v2.json in %s: run census (F3) before map --part geometry" % fd)
    ns = read_json(nsp)
    tc = (ns.get("taxonomy_cache") or {}).get("path")
    if not tc:
        raise RelationError("%s names no taxonomy cache" % nsp)
    from . import taxonomy as T
    resolver = T.Resolver(T.load_cache(tc, dom), dom)
    h3a = pre.hypothesis("H3a")
    gmin, imin = float(h3a["geometry_match_min"]), float(h3a["id_agreement_min"])
    ps_path = C.INC_DIR / "step1" / "pool_summary.json"
    ps = read_json(ps_path)
    res_all = dom.raw["sources"].get("card_resolvers") or {}
    auth = set(dom.authoritative_sources())
    kt6_src = set(dom.kt("KT6").get("sources", [])) if "KT6" in dom.kt_ids() else set()
    pm_path = C.INC_DIR / "step1" / "pool_meta.jsonl"              # the pool's own labels (source_labels)
    matches, inputs, h1pre = {}, {"pool_summary": ps_path}, {}
    if pm_path.exists():
        inputs["pool_meta"] = pm_path
    for slug in sorted(res_all):
        spec = (res_all[slug] or {}).get("upstream_annotations")
        if not spec:
            continue
        got = _upstream_file(fd, cards, slug)
        if got is None:
            if slug in auth or slug in kt6_src:
                raise RelationError("no fetched upstream annotations for %s: run fetch --what cards" % slug)
            continue
        path, sha = got
        up = read_upstream(path, spec.get("format", "yolo"), classes_file=spec.get("classes_file"),
                           voc_dir=spec.get("voc_dir"), yolo_dir=spec.get("yolo_dir"),
                           frame_name_regex=spec.get("frame_name_regex"))
        pool = adapter.source_labels(slug)
        gm = geometry_match(pool, up, TOL_PX)
        n_pool = sum(len(v) for v in pool.values())
        n_match = len(gm["box_pairs"])
        rec = {"upstream": {"path": str(path), "sha256": sha}, "pool_boxes": n_pool, "matched_boxes": n_match,
               "matched_share": round(n_match / n_pool, 6) if n_pool else 0.0, "tolerance_px": TOL_PX,
               "tolerance_rule": "per box corner, in the upstream frame; plus half the last decimal of a "
                                 "normalised upstream coordinate",
               "matched_images": len(gm["image_pairs"]), "pool_images": len(pool),
               "paired_upstream": sorted(set(gm["image_pairs"].values())), "upstream_files": len(up),
               "ambiguous_images": len(gm["ambiguous"]), "card_path": str(path), "card_sha256": sha}
        stems = getattr(up, "stems", None)
        rec["stems"] = ({"distinct": len({s for s in stems.values() if s}), "files": len(stems),
                         "rule": spec.get("frame_name_regex")} if stems is not None else None)
        table = (res_all[slug] or {}).get("class_table") or {}
        join = ((ps.get("per_slug") or {}).get(slug) or {}).get("join") or {}
        if table and all(isinstance(bp["up_id"], int) for bp in gm["box_pairs"]):
            ch = choose_alignment(gm["box_pairs"], alignments(len(table)), "funnel/v1/h3a/half",
                                  n_pool_classes=len([k for k in join if k != "*"]), min_agreement=imin)
            rec.update(mode="ids", alignments=ch["alignments"], chosen=ch["chosen"], confirmed=ch["confirmed"],
                       halves=ch["halves"],
                       agreement_b=[a for a in ch["alignments"] if a["name"] == ch["chosen"]][0]["agreement_b"])
            rec["card_names"] = card_name_check(table, getattr(up, "names", {}) or {})
        else:
            rec.update(mode="names", alignments=[], chosen=None, confirmed=False, halves=None, agreement_b=None)
        if slug in auth:
            upn = getattr(up, "names", {}) or {}
            agree, tot, conf = 0, 0, collections.Counter()
            for bp in gm["box_pairs"]:
                pn = (join.get(str(bp["pool_id"])) or [""])[0]
                un = upn.get(bp["up_id"], bp["up_id"]) if isinstance(bp["up_id"], int) else bp["up_id"]
                tot += 1
                agree += int(name_key(pn) == name_key(un))
                conf["%s -> %s" % (pn, un)] += 1
            share = agree / float(tot) if tot else 0.0
            ok = tot > 0 and share >= imin and rec["matched_share"] >= gmin
            rec["names"] = {"agreement": round(share, 6), "n": tot,
                            "pairs": dict(sorted(conf.items(), key=lambda kv: (-kv[1], kv[0]))[:40])}
            h1pre[slug] = {"pass": bool(ok),
                           "why": ("pool class names agree with the upstream names on %.4f of %d matched boxes; "
                                   "%.4f of pool boxes matched (need %.2f and %.2f)"
                                   % (share, tot, rec["matched_share"], imin, gmin))}
        matches[slug] = rec
        if Path(path).is_file():
            inputs["upstream:%s" % slug] = {"path": str(path), "sha256": sha}
        else:
            # a directory of upstream label files: its digest of the files' sha256 values is recorded
            # without a path (check_records re-hashes files only), and the fetched cards index that
            # pins every one of those files is recorded as a file below
            inputs["upstream:%s" % slug] = {"path": None, "dir": str(path), "sha256": sha,
                                            "what": "sha256 of the sorted sha256 values of the fetched files"}
            inputs["cards_index"] = Path(fd) / "cards" / "index.json"
    h3 = {"pass": False, "why": "no card-resolved source with upstream annotations"}
    for slug in sorted(kt6_src):
        rec = matches.get(slug)
        if rec is None:
            h3 = {"pass": False, "why": "%s has no geometry match" % slug}
            continue
        ok = rec["mode"] == "ids" and rec["matched_share"] >= gmin and rec["confirmed"] and \
            (rec["agreement_b"] or 0) >= imin
        h3 = {"pass": bool(ok),
              "why": "%s: matched share %.4f (need %.2f); alignment %s agreement %.4f on the second half "
                     "(need %.2f), confirmed %s" % (slug, rec["matched_share"], gmin, rec["chosen"],
                                                    rec["agreement_b"] or 0.0, imin, rec["confirmed"])}
    geo = header("relation_geometry", dom, pre, inputs, seeds={"halves": "funnel/v1/h3a/half"},
                 modules=(_self(),), testing=testing)
    geo.update({"matches": matches, "h3a_exact": h3, "h1_pre": h1pre})
    geo = json.loads(json.dumps(geo))                   # the form it has on disk
    props = card_proposals(dom, geo, ps, resolver=resolver, name_status=ns)
    geo_path, cm_path = fd / "relation_geometry_v1.json", fd / "class_maps.json"
    old_cm = read_json(cm_path) if cm_path.exists() else None
    if old_cm is not None and old_cm.get("proposals") != props and not force:
        # refuse before anything is written: the geometry file KT6 and H3a read stays as it was
        raise RelationError("class_maps.json holds other proposals and is immutable after F4; rerun with "
                            "--force only before any map is applied")
    old_geo = read_json(geo_path) if geo_path.exists() else None
    geo_same = old_geo is not None and strip_volatile(old_geo) == strip_volatile(geo)
    if geo_same and old_cm is not None and old_cm.get("proposals") == props:
        return old_geo                                  # same inputs: a no-op (runner §1.2)
    if geo_same:
        geo = old_geo
    else:
        write_json_atomic(geo_path, geo)
    new = header("class_maps", dom, pre, {"relation_geometry": geo_path, "pool_summary": ps_path,
                                          "name_status_v2": nsp, "taxonomy_cache": Path(tc)},
                 modules=(_self(),), testing=testing)
    new["proposals"] = props
    write_json_atomic(cm_path, new)
    return geo


def run_relation(prereg, domain, funnel_dir, adapter, testing=False):
    """F5 `map --part relation`: relation_audit_v1.json (H0b, H0c, visual
    proposals), from the judge score files and the adapter's class units."""
    import numpy as np
    pre, dom = _load(prereg, domain)
    fd = Path(funnel_dir)
    checks = dom.raw["sources"].get("relation_checks") or {}
    need = ("h0b_map_sources", "h0b_flag_old_join", "h0b_keep_current_join", "h0c_source")
    miss = [k for k in need if k not in checks]
    if miss:
        raise RelationError("domain %s: sources.relation_checks lacks %s" % (dom.name, miss))
    jq_path = fd / "judge_qualification.json"
    if not jq_path.exists():
        raise RelationError("no judge_qualification.json in %s: run qualify first (F5)" % fd)
    jq = read_json(jq_path)
    h0b_judge = checks.get("h0b_judge", "J-knn1")
    best, best_lb = None, None
    for jid, rec in sorted((jq.get("judges") or {}).items()):
        bt = (rec.get("by_type") or {}).get("other_noinfo") or {}
        lb = ((bt.get("precision_at_half") or {}).get("lb"))
        if bt.get("qualified") and lb is not None and (best_lb is None or lb > best_lb):
            best, best_lb = jid, lb
    labels_cache = {}

    def scores_for(judge):
        p = fd / "judges" / ("%s__crops.npz" % judge)
        if not p.exists():
            raise RelationError("judge score file %s is missing" % p)
        with np.load(p, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"]))
            ui, P = d["unit_index"].astype(np.int64), d["P"].astype(np.float32)
        row_of = {int(c): i for i, c in enumerate(ui)}
        labels_cache[judge] = (meta.get("labels") or [], P, row_of, C.sha256_file(p))
        return labels_cache[judge]
    units = adapter.relation_units(dom, sorted(set(checks["h0b_map_sources"]) | set(checks["h0b_flag_old_join"])
                                               | set(checks["h0b_keep_current_join"])))
    labs, P, row_of, sha_b = scores_for(h0b_judge)
    sc = relation_scores({u: v["crop_ids"] for u, v in units.items()}, P, labs, row_of)
    rows, checks_out = [], []
    for joinkind in ("current", "old"):
        us = {u: v["boxes"] for u, v in units.items()}
        js = {u: v["join" if joinkind == "current" else "old_join"] for u, v in units.items()}
        tr = {u: v["truth"] for u, v in units.items()}
        for r in relation_audit(us, sc, js, tr):
            r["join_kind"] = joinkind
            r["source"] = units[r["unit"]]["source"]
            rows.append(r)
    # every check needs units to judge: a source with no unit of >= MIN_BOXES
    # boxes has not been tested, and an untested check fails (H0 gates the audit)
    mapped = [r for r in rows if r["join_kind"] == "current" and r["source"] in checks["h0b_map_sources"]
              and r["boxes"] >= MIN_BOXES]
    map_untested = sorted(set(checks["h0b_map_sources"]) - {r["source"] for r in mapped})
    ok_map = not map_untested and all(r["mapped_to"] == r["truth"] for r in mapped)
    flagged_old = [r for r in rows if r["join_kind"] == "old" and r["source"] in checks["h0b_flag_old_join"]
                   and r["join"] != r["truth"] and r["boxes"] >= MIN_BOXES]
    ok_flag = bool(flagged_old) and all(r["join_flagged"] for r in flagged_old)
    kept = [r for r in rows if r["join_kind"] == "current" and r["source"] in checks["h0b_keep_current_join"]]
    keep_untested = sorted(set(checks["h0b_keep_current_join"])
                           - {r["source"] for r in kept if r["boxes"] >= MIN_BOXES})
    ok_keep = not keep_untested and not any(r["join_flagged"] for r in kept)
    checks_out = [{"name": "maps every class to its twin's species", "pass": ok_map, "n": len(mapped),
                   "untested_sources": map_untested},
                  {"name": "flags the old wrong join", "pass": ok_flag, "n": len(flagged_old)},
                  {"name": "does not flag the current join", "pass": ok_keep, "n": len(kept),
                   "untested_sources": keep_untested}]
    h0b = {"units": rows, "pass": bool(ok_map and ok_flag and ok_keep), "checks": checks_out}
    h0c = {"source": checks["h0c_source"], "units": [], "class_accuracy": None, "maps_relative_to_target": [],
           "pass": None, "judge": best}
    proposals = []
    if best is not None:
        labs2, P2, row2, sha_c = scores_for(best)
        pu = adapter.pool_class_units(dom, fd)
        mine = {u: v for u, v in pu.items() if v["source"] == checks["h0c_source"]}
        sc2 = relation_scores({u: v["crop_ids"] for u, v in mine.items()}, P2, labs2, row2)
        res = relation_audit({u: v["boxes"] for u, v in mine.items()}, sc2, {u: v["join"] for u, v in mine.items()})
        other_name = dom.other["name"]
        right, n = 0, 0
        rel, no_truth = [], []
        for r in res:
            v = mine[r["unit"]]
            if r["boxes"] < MIN_BOXES:
                continue
            st = v["status_v2"]
            if st == "target":
                n += 1
                right += int(r["mapped_to"] == v["join"])
            elif st in NAMED_NON_TARGET:
                # the name says: not a target. A resolved name (a relative, or any other
                # named taxon, e.g. a known attractor) mapped to a target fails H0(c)
                n += 1
                right += int(r["mapped_to"] in (None, other_name, "other"))
                if st in RESOLVED_NON_TARGET and r["mapped_to"] in dom.target_names:
                    rel.append(r["unit"])
            else:
                no_truth.append(r["unit"])          # no-information, excluded or synonym names
        h0c.update(units=res, class_accuracy=round(right / float(n), 6) if n else None,
                   class_accuracy_n=n, maps_relative_to_target=rel, units_without_name_truth=no_truth)
        # untested is not passed: a source with no named class of >= MIN_BOXES boxes
        h0c["pass"] = bool(n) and not rel
        frames_noinfo = set((dom.raw.get("names") or {}).get("frames", {}).get("noinfo", []))
        noinfo = {u: v for u, v in pu.items() if v["status_v2"] in frames_noinfo}
        sc3 = relation_scores({u: v["crop_ids"] for u, v in noinfo.items()}, P2, labs2, row2)
        for r in relation_audit({u: v["boxes"] for u, v in noinfo.items()}, sc3,
                                {u: v["join"] for u, v in noinfo.items()}):
            if r["mapped_to"] in dom.target_names:
                proposals.append({"unit": r["unit"], "map_to": r["mapped_to"], "r": r["r"],
                                  "status": "proposal_L14"})
    inputs = {"judge_qualification": jq_path, "h0b_scores": {"path": str(fd / "judges" / ("%s__crops.npz" % h0b_judge)),
                                                              "sha256": sha_b}}
    if best is not None:
        # the H0(c) and proposal judge's scores and the ledger its class units came from
        inputs["h0c_scores"] = {"path": str(fd / "judges" / ("%s__crops.npz" % best)), "sha256": sha_c}
        if (fd / "ledger.jsonl").exists():
            inputs["ledger"] = fd / "ledger.jsonl"
    doc = header("relation_audit", dom, pre, inputs, modules=(_self(),), testing=testing)
    doc.update({"judge": h0b_judge, "h0b": h0b, "h0c": h0c, "visual_proposals": proposals,
                "rule": {"map_min": MAP_MIN, "flag_max": FLAG_MAX, "min_boxes": MIN_BOXES}})
    write_json_atomic(fd / "relation_audit_v1.json", doc)
    return doc


def _self():
    import sys
    return sys.modules[__name__]
