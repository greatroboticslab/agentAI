"""Blind multiple-choice sheets for the reference labeller (contract
docs/FUNNEL_AUDIT.md §4.3, §5.2, DEC-1, DEC-2; runner
docs/FUNNEL_AUDIT_RUNNER.md §4.10, §5.4.1, §7.1).

    python -m <package>.tools.funnel sheets --prereg PATH       (F7a, cluster)

Inputs: the locked sample (sample_v1.csv, sample_v1_key.jsonl, frames_v1.json;
their sha256 values must equal the prereg's sample-lock amendment), the
adapter's known truth and crop tables, and the pixels.

Outputs:
  sheets_v1/          pool sheets (may be shipped to the lab); boards;
                      instructions.txt; index.json (contains_eval_pixels false)
  sheets_v1_cluster/  pair sheets that show evaluation pixels (cluster-only)
  sheets_v1_key/      key.jsonl (what every position holds) and packing.json
                      (cluster-only)

Blinding. A sheet shows an enlarged crop, a context thumbnail with the box in
red and a reference board; it never holds a source, stratum, verdict, score,
proposal or unit id. Every pool item gets the same option list (all targets,
all attractors, the tail), so the options cannot reveal the stratum. The
sheet JSON is checked for leaks before anything is written.

Boards. An item goes on the first board (config order) whose material shares
nothing with it (qualify.shares): a set that is not independent contributes
all of its items' keys (so an item of the reference lab group, or a copy of a
reference photograph, never sits on a board of reference exemplars); an
independent set contributes its drawn exemplars, per observation.

Composition (per board). Items are queued per group in item-id order; sheets
are filled round-robin over G0, G1, G2, G2v, G2a, G3, identity, G4 with
`sheet.items - sheet.sentinels` non-sentinel slots and at most `max_low_prior`
G4 items. A sheet needs (sentinel targets + sum of its items' expected target
shares) / size >= min_prevalence; when it is short with its minimum of target
sentinels, a G4 item is swapped for the next G1 or G2 item. Target sentinels
per sheet follow a seeded schedule (1-3, more where prevalence needs them).
Sentinels that fit several boards go where sentinels are short. Surplus
sentinels fill empty slots and then sentinel-only sheets; a shortfall is
recorded in packing.json. Positions are a seeded permutation per sheet.

Anchoring sentinels (a wrong proposal drawn from the configured attractor
pairs) exist only on a person's verify-only items (person_items); machine
sheets carry no proposal.

Nothing here names a domain.
"""
from __future__ import annotations

import collections
import csv
import io
import math
import sys
from pathlib import Path

from ..inc import common as C
from ..inc import verify as V
from . import (FUNNEL_DIR, DisjointnessError, FunnelError, SheetError, StaleInput, canonical_json,
               file_record, header, read_csv, read_json, read_jsonl, rng, sha256_bytes,
               write_json_atomic, write_jsonl_atomic, _atomic_write_bytes)
from . import qualify as Q

SHEETS_DIR = "sheets_v1"
CLUSTER_DIR = "sheets_v1_cluster"
KEY_DIR = "sheets_v1_key"
TILE = 224
HEADER_PX = 20
PANEL_W, PANEL_H = 2 * TILE, HEADER_PX + TILE
COLS, ROWS = 3, 5
BOARD_TILE = 112
BOARD_GAP = 4
BOARD_TEXT_W = 520
BOARD_ROW_H = BOARD_TILE + 8
GREY = (124, 124, 124)
RED = (255, 0, 0)
JPEG = {"format": "JPEG", "quality": 92, "subsampling": 0, "optimize": False}
GROUP_ORDER = ("G0", "G1", "G2", "G2v", "G2a", "G3", "identity", "G4")
LOW_PRIOR_GROUP = "G4"
SWAP_FROM = ("G1", "G2")
SENTINEL_GROUP = "sentinel"
SENTINEL_GROUPS = ("sentinel", "pair_sentinel")
IDENTITY_GROUP = "identity"
EVAL_CLASS = "eval"
POOL_CLASS = "pool"
SHEET_KEYS = ("format", "sheet_id", "kind", "image", "board", "options", "question", "items")
ITEM_KEYS = ("item_id", "position", "panel", "tiles")
KEY_FIELDS = ("item_id", "sheet_id", "position", "unit_id", "group", "stratum", "is_sentinel", "kt", "truth",
              "truth_taxon", "truth_kind", "pair_truth", "board_id")


def log(msg):
    print("[funnel.sheets] %s" % msg, flush=True)


def _rl_cfg(domain):
    rl = domain.raw.get("reference_labeller")
    if not isinstance(rl, dict):
        raise SheetError("domain %s has no reference_labeller section" % domain.name)
    for k in ("prompt", "sheet_prompt", "pair_prompt", "pair_sheet_prompt", "boards", "sheet"):
        if k not in rl:
            raise SheetError("reference_labeller.%s is not set in domain %s" % (k, domain.name))
    sh = rl["sheet"]
    for k in ("items", "sentinels", "min_prevalence", "max_low_prior"):
        if k not in sh:
            raise SheetError("reference_labeller.sheet.%s is not set" % k)
    if int(sh["items"]) != COLS * ROWS:
        raise SheetError("reference_labeller.sheet.items is %s; the layout holds %d panels"
                         % (sh["items"], COLS * ROWS))
    return rl


def options_text(options):
    return "\n".join("%d. %s" % (o["n"], o["text"]) for o in options)


def fill_prompt(template, options):
    """The config's prompt with its {options} placeholder filled (plain
    replacement: a prompt may hold other braces)."""
    if "{options}" not in template:
        raise SheetError("a labeller prompt has no {options} placeholder")
    return template.replace("{options}", options_text(options))


# ------------------------------------------------------------------ pixels
def _open_rgb(path):
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as im0:
            return ImageOps.exif_transpose(im0).convert("RGB")
    except Exception as e:
        raise SheetError("cannot open image %s (%s: %s)" % (path, type(e).__name__, e))


def crop_tile(image, box):
    """The crop exactly as verify cuts it (verify._cut_task: EXIF-transposed,
    square, grey padding, 224 px), as a PIL image."""
    from PIL import Image
    row = {"crop_id": 0, "cx": float(box["cx"]), "cy": float(box["cy"]), "w": float(box["w"]),
           "h": float(box["h"]), "W": 0, "H": 0}
    _img, _ids, arrs, err = V._cut_task((str(image), [row]))
    if err:
        raise SheetError("cannot cut a crop from %s (%s)" % (image, err))
    return Image.fromarray(arrs[0]).convert("RGB")


def letterbox(im, size=TILE, box=None):
    """The whole image letterboxed to size x size on grey; box (cx, cy, w, h
    normalised) drawn as a 2 px red rectangle."""
    from PIL import Image, ImageDraw
    W, H = im.size
    scale = size / float(max(W, H))
    w, h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    canvas = Image.new("RGB", (size, size), GREY)
    ox, oy = (size - w) // 2, (size - h) // 2
    canvas.paste(im.resize((w, h), Image.BICUBIC), (ox, oy))
    if box is not None:
        x0 = ox + (box["cx"] - box["w"] / 2.0) * W * scale
        x1 = ox + (box["cx"] + box["w"] / 2.0) * W * scale
        y0 = oy + (box["cy"] - box["h"] / 2.0) * H * scale
        y1 = oy + (box["cy"] + box["h"] / 2.0) * H * scale
        d = ImageDraw.Draw(canvas)
        d.rectangle([int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)) - 1, int(math.ceil(y1)) - 1],
                    outline=RED, width=2)
    return canvas


def _font(size=14):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _jpeg_bytes(im):
    buf = io.BytesIO()
    im.save(buf, **JPEG)
    return buf.getvalue()


def _png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _wrap(text, width=62):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]


# ------------------------------------------------------------ crop locator
class Locator(object):
    """Where the pixels of a unit are: {"image", "box": {cx, cy, w, h}} for a
    box unit, {"a", "b"} for a pair. Crop-table rows come from the adapter;
    independent photos from kt7-style tables named in the funnel directory."""

    PREFIX_SET = {"b": "pool", "t1": "core", "t2": "copy"}

    def __init__(self, adapter, funnel_dir, reference=None):
        self.adapter = adapter
        self.reference = reference
        self.funnel_dir = Path(funnel_dir)
        self._crops = None
        self._by_key = None
        self._photos = None
        self._manifests = {}
        self._registry = None

    def _table(self):
        if self._crops is None:
            self._crops = self.adapter.crop_table()
            t = self._crops
            sets = t.set
            names = [V.SETS[int(x)] for x in sets] if hasattr(sets, "dtype") else [str(x) for x in sets]
            self._by_key = {}
            for i in range(t.n):
                self._by_key[(names[i], t.key[i], int(t.box[i]))] = i
        return self._crops

    def _photo_table(self):
        if self._photos is None:
            self._photos = {}
            p = self.funnel_dir / "kt7" / "crops_kt7.csv"
            if p.is_file():
                with open(p, newline="", encoding="utf-8") as fh:
                    rd = csv.DictReader(fh)
                    for r in rd:
                        self._photos[Path(r["image"]).stem] = r
                        self._photos["key:%s" % r["key"]] = r
                        self._photos["crop:%s" % r["crop_id"]] = r
        return self._photos

    def box(self, unit_id, crop_id=None):
        u = str(unit_id)
        pre, _, rest = u.partition(":")
        if pre in self.PREFIX_SET:
            key, _, b = rest.rpartition("#")
            t = self._table()
            i = self._by_key.get((self.PREFIX_SET[pre], key, int(b)))
            if i is None:
                raise SheetError("unit %s has no row in the crop table" % u)
            if crop_id not in (None, "") and int(crop_id) != int(i):
                raise SheetError("unit %s: the sample says crop %s, the crop table %d" % (u, crop_id, i))
            return {"image": t.image[i], "box": {"cx": float(t.cx[i]), "cy": float(t.cy[i]),
                                                 "w": float(t.w[i]), "h": float(t.h[i])}}
        if "/" in rest:
            obs, _, photo = rest.rpartition("/")
            tab = self._photo_table()
            r = tab.get("key:%s" % u) or tab.get("%s_%s" % (obs, photo))
            if r is None and crop_id not in (None, ""):
                r = tab.get("crop:%s" % crop_id)
            if r is None:
                raise SheetError("unit %s has no row in kt7/crops_kt7.csv" % u)
            img = r["image"]
            if not Path(img).is_absolute():
                img = str(self.funnel_dir / "kt7" / img) if (self.funnel_dir / "kt7" / img).is_file() \
                    else str(self.funnel_dir / img)
            return {"image": img, "box": {"cx": float(r["cx"]), "cy": float(r["cy"]),
                                          "w": float(r["w"]), "h": float(r["h"])}}
        raise SheetError("unit %s is not a box or photo unit" % u)

    def _manifest(self, split):
        if split not in self._manifests:
            p = C.manifest_path(split)
            if not p.is_file():
                raise SheetError("no manifest for split %s at %s" % (split, p))
            self._manifests[split] = {r["key"]: r["image"] for r in C.read_manifest(p)}
        return self._manifests[split]

    def _pool(self):
        if "pool" not in self._manifests:
            self._manifests["pool"] = {r["key"]: r["image"] for r in self.adapter.pool_rows()}
        return self._manifests["pool"]

    def _copies(self):
        if "copies" not in self._manifests:
            self._manifests["copies"] = ({r["key"]: r["image"] for r in C.read_manifest(V.COPIES)}
                                         if Path(V.COPIES).is_file() else {})
        return self._manifests["copies"]

    def _source_image(self, slug, rel):
        if self._registry is None:
            self._registry = V._load_registry()
        info = self._registry.get(slug)
        root = V._resolve_dir(info) if isinstance(info, dict) else None
        if root is None:
            raise SheetError("source %s has no local directory for %s" % (slug, rel))
        return str(Path(root) / rel)

    def _image_a(self, slug, x):
        """Image A of a pair: a reference image by key, a copy by key, a pool
        image by key, else a source file by its relative path."""
        ref = self.reference
        if ref and slug == ref:
            img = self._manifest(ref).get(x)
            if img is not None:
                return img
        for table in (self._copies(), self._pool()):
            if x in table:
                return table[x]
        if "/" in x:
            return self._source_image(slug, x)
        raise SheetError("pair image %s|%s is not a reference, copy or pool key, nor a path" % (slug, x))

    def _image_b(self, split, y):
        from ..inc import common as CC
        if split in CC.EVAL_SPLITS or split in CC.TRAIN_SPLITS:
            img = self._manifest(split).get(y)
        elif split == "dup":
            img = self._pool().get(y)
        else:
            img = self._pool().get(y) or self._copies().get(y)
        if img is None:
            raise SheetError("pair image %s|%s is not in that split, the pool or the copies" % (split, y))
        return img

    def pair(self, unit_id):
        """p:<source>|<x>|<split>|<y> (read from the right, so the source part
        of a calibration pair, "calibration:<a>|<b>", keeps its "|"): A is <x>
        (a reference, copy or pool key, or a relative path under the source's
        directory), B is <y> of <split> (an evaluation or reference split,
        "dup" for the kept pool twin, else a pool or copy key)."""
        u = str(unit_id)
        if not u.startswith("p:"):
            raise SheetError("unit %s is not a pair" % u)
        parts = u[2:].rsplit("|", 3)            # the source part may itself hold "|" (calibration pairs)
        if len(parts) != 4:
            raise SheetError("pair unit %s does not have four parts" % u)
        slug, x, split, y = parts
        return {"a": self._image_a(slug, x), "b": self._image_b(split, y)}


# ------------------------------------------------------------------ boards
def _kt_items(known_truth):
    return {k: [dict(it, kt=k) for it in v] for k, v in known_truth.items()}


def _exemplars_for(domain, kt_items, board, option):
    """The exemplar items of one board row (seeded, runner §4.10)."""
    kt_id = board["targets_from"] if option["kind"] == "target" else board["attractors_from"]
    per = int(board.get("per_option", 6))
    items = kt_items.get(kt_id, [])
    if option["kind"] == "target":
        pool = [it for it in items if it.get("truth_kind") == "target" and it.get("truth") is not None
                and int(it["truth"]) == int(option["class"])]
    else:
        pool = [it for it in items if it.get("truth_kind") == "attractor" and it.get("truth_taxon") == option["taxon"]]
    independent = kt_id in domain.independent_sets()
    if independent:
        pool = [it for it in pool if it.get("role") == "exemplar"]
    pool = sorted(pool, key=lambda it: it["id"])
    if not pool:
        return [], None
    r = rng("funnel/v1/board/%s/%d" % (board["id"], option["n"]))
    sessions = None
    if not independent and any(it.get("session") for it in pool):
        all_s = sorted({it.get("session") for it in pool if it.get("session")})
        pick = r.permutation(len(all_s))[:2]
        sessions = sorted(all_s[i] for i in pick)
        pool = [it for it in pool if it.get("session") in sessions]
    order = r.permutation(len(pool))
    return [pool[i] for i in order[:per]], sessions


def board_plans(domain, known_truth):
    """{board id: {"id", "options", "exemplars": {n: [items]}, "sessions": {n:
    [..]}, "material": key lists}} (no pixels)."""
    rl = _rl_cfg(domain)
    kt_items = _kt_items(known_truth)
    options = domain.options()
    out = {}
    for b in rl["boards"]:
        ex, sess = {}, {}
        mat_items = []
        for o in options:
            if o["kind"] == "tail":
                continue
            chosen, sessions = _exemplars_for(domain, kt_items, b, o)
            ex[o["n"]] = chosen
            if sessions:
                sess[o["n"]] = sessions
        # a non-independent set counts whole; an independent one by its exemplars
        for kt_id in sorted({b["targets_from"], b["attractors_from"]}):
            if kt_id in domain.independent_sets():
                mat_items.extend(Q.item_keys(it, independent=True)
                                 for items in ex.values() for it in items if it["kt"] == kt_id)
            else:
                mat_items.extend(Q.item_keys(it, independent=False) for it in kt_items.get(kt_id, []))
        out[b["id"]] = {"id": b["id"], "board": dict(b), "options": options, "exemplars": ex,
                        "sessions": sess, "material": Q.material_of(mat_items)}
    return out


def boards(domain, adapter, out_dirs, known_truth=None, funnel_dir=None):
    """Board plans with their images rendered into each of out_dirs
    (board_<id>.jpg and board_<id>.json). Returns {board id: plan + files}."""
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    kt = known_truth if known_truth is not None else adapter.known_truth(domain, funnel_dir)
    plans = board_plans(domain, kt)
    loc = Locator(adapter, funnel_dir, reference=domain.reference_source)
    for bid, plan in sorted(plans.items()):
        img = render_board(plan, loc)
        data = _jpeg_bytes(img)
        plan["image_bytes"] = data
        plan["files"] = {}
        for d in out_dirs:
            plan["files"][str(d)] = write_board(plan, img, data, Path(d))
    return plans


def render_board(plan, locator):
    from PIL import Image, ImageDraw
    options = plan["options"]
    per = int(plan["board"].get("per_option", 6))
    width = BOARD_TEXT_W + per * (BOARD_TILE + BOARD_GAP) + BOARD_GAP
    height = len(options) * BOARD_ROW_H + 8
    im = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(im)
    font = _font(14)
    for r, o in enumerate(options):
        y = 4 + r * BOARD_ROW_H
        for li, line in enumerate(_wrap("%d. %s" % (o["n"], o["text"]))[:5]):
            d.text((6, y + 4 + 18 * li), line, fill=(0, 0, 0), font=font)
        for j, it in enumerate(plan["exemplars"].get(o["n"], [])[:per]):
            px = locator.box(it["id"], it.get("crop_id"))
            tile = crop_tile(px["image"], px["box"]).resize((BOARD_TILE, BOARD_TILE), Image.BICUBIC)
            im.paste(tile, (BOARD_TEXT_W + BOARD_GAP + j * (BOARD_TILE + BOARD_GAP), y))
    return im


def write_board(plan, img, data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jpg = out_dir / ("board_%s.jpg" % plan["id"])
    _atomic_write_bytes(jpg, data)
    doc = {"format": "funnel-board/1", "board_id": plan["id"],
           "image": {"file": jpg.name, "sha256": sha256_bytes(data), "w": img.size[0], "h": img.size[1]},
           "options": [{"n": o["n"], "text": o["text"]} for o in plan["options"]],
           "exemplars": {str(n): [it["id"] for it in items] for n, items in sorted(plan["exemplars"].items())}}
    js = out_dir / ("board_%s.json" % plan["id"])
    sha = write_json_atomic(js, doc)
    return {"image": jpg.name, "image_sha256": doc["image"]["sha256"], "json": js.name, "json_sha256": sha}


def fitting_boards(item_keys, plans):
    """The boards (config order) whose material shares nothing with the item."""
    return [bid for bid, p in plans.items() if Q.disjoint(item_keys, p["material"])]


def board_for(item_keys, domain, boards_):
    """The first board whose material shares nothing with the item;
    DisjointnessError when none does."""
    order = [b["id"] for b in _rl_cfg(domain)["boards"]]
    plans = {bid: boards_[bid] for bid in order if bid in boards_}
    fit = fitting_boards(item_keys, plans)
    if not fit:
        raise DisjointnessError("no board shares nothing with item keys %s" % (item_keys,))
    return fit[0]


# ------------------------------------------------------------- composition
def _pred_class_id(pred, domain, where):
    """The class id of a frame row's pred: the class name strata.py writes, or
    a decimal id. Anything else is refused, never guessed."""
    text = str(pred).strip()
    if text in domain.class_names:
        return domain.class_id(text)
    try:
        return domain.class_id(domain.class_name(int(text)))
    except (ValueError, FunnelError):
        raise SheetError("%s: pred %r is neither a class name nor a class id of domain %s"
                         % (where, pred, domain.name))


def stratum_priors(funnel_dir, domain):
    """{stratum: expected target share}: the share of the stratum's frame
    units whose step-1 prediction is a target (frames_v1/<group>.csv pred,
    a class name as strata.py writes it)."""
    strata = collections.defaultdict(lambda: [0, 0])
    fj = Path(funnel_dir) / "frames_v1.json"
    frames = read_json(fj)
    for g, info in sorted((frames.get("groups") or {}).items()):
        rec = info.get("file") or {}
        p = Path(rec.get("path") or (Path(funnel_dir) / "frames_v1" / ("%s.csv" % g)))
        if not p.is_file():
            p = Path(funnel_dir) / "frames_v1" / ("%s.csv" % g)
        if rec.get("sha256") and file_record(p)["sha256"] != rec["sha256"]:
            raise StaleInput("frame file %s does not hash to what frames_v1.json records" % p)
        _h, rows = read_csv(p)
        for r in rows:
            s = strata[r["stratum"]]
            pred = r.get("pred")
            if pred in (None, ""):
                continue
            s[1] += 1
            s[0] += int(domain.is_target(_pred_class_id(pred, domain, "%s unit %s" % (p.name, r.get("unit_id")))))
    return {s: (k / n if n else 0.0) for s, (k, n) in strata.items()}


def _expected(item, priors, planted_share):
    if item["is_sentinel"] or item["group"] == IDENTITY_GROUP:
        return 1.0 if item.get("truth_kind") == "target" else 0.0
    if item["group"] == "G0":
        return float(planted_share or 0.0)
    return float(priors.get(item["stratum"], 0.0))


def _need_targets(sum_e, size, min_prev, n_sent=3):
    """The fewest target sentinels (1..n_sent) that bring a sheet of `size`
    whose other items sum to sum_e expected targets up to min_prev."""
    for t in range(1, n_sent + 1):
        if (t + sum_e) / float(max(size, 1)) >= min_prev:
            return t
    return n_sent


def _fill_items(queues, slots, max_low):
    items = []
    n_low = 0
    progress = True
    while len(items) < slots and progress:
        progress = False
        for g in GROUP_ORDER:
            if len(items) >= slots:
                break
            q = queues.get(g)
            if not q:
                continue
            if g == LOW_PRIOR_GROUP and n_low >= max_low:
                continue
            items.append(q.pop(0))
            n_low += int(g == LOW_PRIOR_GROUP)
            progress = True
    return items


def compose(sample_rows, key_rows, frames, domain, board_id=None, planted_share=0.0, seed_prefix=None):
    """Sheet plans for one board's items (runner §4.10). sample_rows are the
    items (dicts with item_id, group, stratum, is_sentinel, truth_kind);
    key_rows maps item_id -> key row; frames maps stratum -> expected target
    share. Returns [{"board_id", "items": [...], "sentinels": [...],
    "expected_prevalence", "below_min_prevalence", "n_target_sentinels"}]."""
    rl = _rl_cfg(domain)
    sh = rl["sheet"]
    size, n_sent = int(sh["items"]), int(sh["sentinels"])
    slots = size - n_sent
    min_prev, max_low = float(sh["min_prevalence"]), int(sh["max_low_prior"])
    items, sentinels = [], []
    for r in sample_rows:
        k = key_rows.get(r["item_id"]) or {}
        it = dict(r)
        for f in ("truth", "truth_kind", "truth_taxon", "pair_truth"):
            if f not in it or it[f] in (None, ""):
                it[f] = k.get(f)
        it["e"] = _expected(it, frames, planted_share)
        (sentinels if it["is_sentinel"] else items).append(it)
    unknown = sorted({it["group"] for it in items} - set(GROUP_ORDER))
    if unknown:
        raise SheetError("groups %s have no place in the sheet order %s" % (unknown, GROUP_ORDER))
    queues = {g: sorted([it for it in items if it["group"] == g], key=lambda x: x["item_id"])
              for g in GROUP_ORDER}
    sheets = []
    while any(queues.values()):
        chosen = _fill_items(queues, slots, max_low)
        if not chosen:
            break
        size_now = len(chosen) + n_sent
        # swap G4 items for the next G1/G2 items while the sheet stays short
        # even with every sentinel slot a target
        while (n_sent + sum(i["e"] for i in chosen)) / float(size_now) < min_prev:
            low = [i for i in chosen if i["group"] == LOW_PRIOR_GROUP]
            nxt = next((g for g in SWAP_FROM if queues.get(g)), None)
            if not low or nxt is None:
                break
            out = low[-1]
            chosen.remove(out)
            queues[LOW_PRIOR_GROUP].insert(0, out)
            chosen.append(queues[nxt].pop(0))
        sheets.append({"board_id": board_id, "items": chosen, "sentinels": []})
    # target sentinels: each sheet's need first, then the rest spread in a
    # seeded order, 1..n_sent per sheet, in proportion to the supply
    targets = sorted([s for s in sentinels if s.get("truth_kind") == "target"], key=lambda x: x["item_id"])
    others = sorted([s for s in sentinels if s.get("truth_kind") != "target"], key=lambda x: x["item_id"])
    order = list(range(len(sheets)))
    if sheets:
        perm = rng("%s/sentinels" % (seed_prefix or "funnel/v1/sheets")).permutation(len(sheets))
        order = [int(i) for i in perm]
    need = {i: _need_targets(sum(x["e"] for x in s["items"]), len(s["items"]) + n_sent, min_prev, n_sent)
            for i, s in enumerate(sheets)}
    supply = len(targets) + len(others)
    t_reg = min(len(targets), n_sent * len(sheets),
                max(sum(need.values()), int(round(n_sent * len(sheets) * len(targets) / float(max(supply, 1))))))
    quota = {i: 0 for i in range(len(sheets))}
    placed = 0
    for rnd in range(1, n_sent + 1):
        for i in order:
            if placed < t_reg and quota[i] < min(need[i], rnd):
                quota[i] += 1
                placed += 1
    for rnd in range(1, n_sent + 1):
        for i in order:
            if placed < t_reg and quota[i] < rnd:
                quota[i] += 1
                placed += 1
    for i in order:
        for _ in range(quota[i]):
            sheets[i]["sentinels"].append(targets.pop(0))
    for i in order:
        while len(sheets[i]["sentinels"]) < n_sent and (others or targets):
            sheets[i]["sentinels"].append(others.pop(0) if others else targets.pop(0))
    surplus = targets + others
    for i in range(len(sheets)):
        while surplus and len(sheets[i]["items"]) + len(sheets[i]["sentinels"]) < size:
            sheets[i]["sentinels"].append(surplus.pop(0))
    while surplus:
        sheets.append({"board_id": board_id, "items": [], "sentinels": surplus[:size]})
        surplus = surplus[size:]
    for s in sheets:
        tot = len(s["items"]) + len(s["sentinels"])
        s["n_target_sentinels"] = sum(1 for x in s["sentinels"] if x.get("truth_kind") == "target")
        s["expected_prevalence"] = ((s["n_target_sentinels"] + sum(x["e"] for x in s["items"])) / float(tot)
                                    if tot else 0.0)
        s["below_min_prevalence"] = s["expected_prevalence"] < min_prev
        s["sentinel_shortfall"] = max(0, n_sent - len(s["sentinels"])) if s["items"] else 0
        s["n_low_prior"] = sum(1 for x in s["items"] if x["group"] == LOW_PRIOR_GROUP)
    return sheets


def compose_pairs(items, domain):
    """Pair sheets: 12 pair items and 3 pair sentinels per sheet, no board."""
    rl = _rl_cfg(domain)
    size, n_sent = int(rl["sheet"]["items"]), int(rl["sheet"]["sentinels"])
    body = sorted([i for i in items if not i["is_sentinel"]], key=lambda x: x["item_id"])
    sent = sorted([i for i in items if i["is_sentinel"]], key=lambda x: x["item_id"])
    sheets = []
    while body:
        sheets.append({"board_id": None, "items": body[:size - n_sent], "sentinels": []})
        body = body[size - n_sent:]
    for i in range(len(sheets)):
        while sent and len(sheets[i]["sentinels"]) < n_sent:
            sheets[i]["sentinels"].append(sent.pop(0))
    for i in range(len(sheets)):
        while sent and len(sheets[i]["items"]) + len(sheets[i]["sentinels"]) < size:
            sheets[i]["sentinels"].append(sent.pop(0))
    while sent:
        sheets.append({"board_id": None, "items": [], "sentinels": sent[:size]})
        sent = sent[size:]
    for s in sheets:
        s["sentinel_shortfall"] = max(0, n_sent - len(s["sentinels"])) if s["items"] else 0
        s["expected_prevalence"] = None
        s["below_min_prevalence"] = False
        s["n_target_sentinels"] = None
        s["n_low_prior"] = 0
    return sheets


def _positions(sheet_id, n):
    perm = rng("funnel/v1/sheets/%s" % sheet_id).permutation(COLS * ROWS)
    return [int(p) + 1 for p in perm[:n]]


def _panel_rect(position):
    col, row = (position - 1) % COLS, (position - 1) // COLS
    return [col * PANEL_W, row * PANEL_H, PANEL_W, PANEL_H]


# --------------------------------------------------------------- rendering
def render_sheet(plan, out_dir, adapter=None, locator=None, options=None, question=""):
    """sheet_<nnnn>.jpg and .json for a multiple-choice plan whose items carry
    "position" and "pixels" ({"image", "box"}). Returns the sheet JSON."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    font = _font(14)
    im = Image.new("RGB", (COLS * PANEL_W, ROWS * PANEL_H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    entries = []
    for it in sorted(plan["placed"], key=lambda x: x["position"]):
        x, y, w, h = _panel_rect(it["position"])
        px = it["pixels"]
        d.text((x + 4, y + 2), "#%d" % it["position"], fill=(0, 0, 0), font=font)
        im.paste(crop_tile(px["image"], px["box"]), (x, y + HEADER_PX))
        im.paste(letterbox(_open_rgb(px["image"]), TILE, px["box"]), (x + TILE, y + HEADER_PX))
        entries.append({"item_id": it["item_id"], "position": it["position"], "panel": [x, y, w, h],
                        "tiles": {"crop": [x, y + HEADER_PX, TILE, TILE],
                                  "context": [x + TILE, y + HEADER_PX, TILE, TILE]}})
    return _write_sheet(plan, im, entries, out_dir, "mc", options, question)


def render_pair_sheet(plan, out_dir, adapter=None, locator=None, options=None, question=""):
    """A pair sheet: each panel shows image A (left) and image B (right),
    both letterboxed, with no box. Cluster-only directory."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    if out_dir.name != CLUSTER_DIR:
        raise SheetError("pair sheets show evaluation pixels and are written to %s only, not %s"
                         % (CLUSTER_DIR, out_dir))
    font = _font(14)
    im = Image.new("RGB", (COLS * PANEL_W, ROWS * PANEL_H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    entries = []
    for it in sorted(plan["placed"], key=lambda x: x["position"]):
        x, y, w, h = _panel_rect(it["position"])
        px = it["pixels"]
        d.text((x + 4, y + 2), "#%d" % it["position"], fill=(0, 0, 0), font=font)
        im.paste(letterbox(_open_rgb(px["a"]), TILE), (x, y + HEADER_PX))
        im.paste(letterbox(_open_rgb(px["b"]), TILE), (x + TILE, y + HEADER_PX))
        entries.append({"item_id": it["item_id"], "position": it["position"], "panel": [x, y, w, h],
                        "tiles": {"a": [x, y + HEADER_PX, TILE, TILE],
                                  "b": [x + TILE, y + HEADER_PX, TILE, TILE]}})
    return _write_sheet(plan, im, entries, out_dir, "pair", options, question)


def _write_sheet(plan, im, entries, out_dir, kind, options, question):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _jpeg_bytes(im)
    jpg = out_dir / ("%s.jpg" % plan["sheet_id"])
    _atomic_write_bytes(jpg, data)
    board = plan.get("board")
    doc = {"format": "funnel-sheet/1", "sheet_id": plan["sheet_id"], "kind": kind,
           "image": {"file": jpg.name, "sha256": sha256_bytes(data), "w": im.size[0], "h": im.size[1]},
           "board": board, "options": [{"n": o["n"], "text": o["text"]} for o in options or []],
           "question": question, "items": entries}
    return doc


def blind_problems(sheet_doc, forbidden):
    """Leaks in a sheet JSON: a key outside the pinned layout, or any
    forbidden string (sources, strata, unit ids, class names, taxa) anywhere
    but the option list and the question, which are the same on every sheet."""
    probs = []
    if tuple(sorted(sheet_doc)) != tuple(sorted(SHEET_KEYS)):
        probs.append("sheet keys %s are not %s" % (sorted(sheet_doc), sorted(SHEET_KEYS)))
    for it in sheet_doc.get("items", []):
        if tuple(sorted(it)) != tuple(sorted(ITEM_KEYS)):
            probs.append("item keys %s are not %s" % (sorted(it), sorted(ITEM_KEYS)))
    body = {k: v for k, v in sheet_doc.items() if k not in ("options", "question")}
    text = canonical_json(body).lower()
    for f in sorted({str(x) for x in forbidden if x not in (None, "") and len(str(x)) >= 3}):
        if f.lower() in text:
            probs.append("sheet %s holds %r" % (sheet_doc.get("sheet_id"), f))
    return probs


# ------------------------------------------------------------------- run
def _load_locked(prereg, funnel_dir):
    lock = prereg.sample_lock
    if lock is None:
        raise SheetError("the prereg holds no sample lock: run draw (F6) first")
    paths = {"sample": funnel_dir / "sample_v1.csv", "key": funnel_dir / "sample_v1_key.jsonl",
             "frames": funnel_dir / "frames_v1.json"}
    want = {"sample": lock.get("sample_sha256"), "key": lock.get("key_sha256"),
            "frames": lock.get("frames_sha256")}
    recs = {}
    for k, p in paths.items():
        recs[k] = file_record(p)
        if want[k] is not None and recs[k]["sha256"] != want[k]:
            raise StaleInput("%s does not hash to the sample lock's %s_sha256" % (p, k))
        if want[k] is None and k in ("sample", "key"):
            raise SheetError("the sample lock records no %s_sha256" % k)
    _h, sample = read_csv(paths["sample"])
    key_rows = read_jsonl(paths["key"])
    planted = next((r["planted"] for r in key_rows if "planted" in r), None)
    key = {r["item_id"]: r for r in key_rows if "item_id" in r}
    return sample, key, planted, recs


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes")


def _items(sample, key, adapter, domain, kt_ids_of):
    """Items for packing: sample row + real unit id + disjointness keys."""
    out = []
    missing = []
    for r in sample:
        k = key.get(r["item_id"])
        if k is None:
            missing.append(r["item_id"])
            continue
        unit = k.get("unit_id") or r.get("unit_id")
        is_sent = r.get("group") in SENTINEL_GROUPS or _truthy(r.get("is_sentinel"))
        keys = {f: r.get(f) for f in Q.KEYS}
        out.append({"item_id": r["item_id"], "unit_id": unit, "group": r.get("group"),
                    "stratum": r.get("stratum"), "sheet_class": r.get("sheet_class") or POOL_CLASS,
                    "crop_id": r.get("crop_id"), "is_sentinel": is_sent,
                    "kt": r.get("kt") or ";".join(kt_ids_of.get(unit, [])), "keys": keys,
                    "truth": k.get("truth"), "truth_kind": k.get("truth_kind"),
                    "truth_taxon": k.get("truth_taxon"), "pair_truth": k.get("pair_truth"),
                    "source": r.get("source")})
    if missing:
        raise SheetError("%d sample item(s) are missing from the key, e.g. %s" % (len(missing), missing[:3]))
    hidden = [it["unit_id"] for it in out if not it["keys"].get("source") and it["sheet_class"] == POOL_CLASS]
    if hidden:
        ks = adapter.unit_keys(hidden)
        for it in out:
            if it["unit_id"] in ks:
                it["keys"] = {f: ks[it["unit_id"]].get(f) for f in Q.KEYS}
    # an item without its four keys would share with no board and could sit beside its own exemplars
    keyless = sorted(it["unit_id"] for it in out
                     if it["sheet_class"] == POOL_CLASS and not all(it["keys"].get(f) for f in Q.KEYS))
    if keyless:
        raise DisjointnessError("%d sheet item(s) lack disjointness keys (source, near_dup3, provenance, lab), "
                                "e.g. %s: their board cannot be chosen" % (len(keyless), keyless[:3]))
    for it in out:
        kts = [x for x in str(it["kt"] or "").replace("+", ";").split(";") if x]
        ind = any(x in domain.independent_sets() for x in kts)
        it["cmp_keys"] = Q.item_keys(dict(it["keys"], id=it["unit_id"]), independent=ind)
    return out


def check_not_exemplars(items, plans):
    """SheetError when a sheet item is one of the boards' exemplars (the
    labeller would see its answer on the board)."""
    exemplar_ids = {it["id"] for p in plans.values() for xs in p["exemplars"].values() for it in xs}
    shown = sorted(it["unit_id"] for it in items if it["unit_id"] in exemplar_ids)
    if shown:
        raise SheetError("%d sheet item(s) are board exemplars, e.g. %s" % (len(shown), shown[:3]))


def _assign_boards(pool_items, plans, order):
    """{board id: [items]}: non-sentinels on their first fitting board;
    sentinels fixed where only one board fits, flexible ones to the board
    whose sentinel need is largest."""
    fits = {it["item_id"]: [b for b in order if Q.disjoint(it["cmp_keys"], plans[b]["material"])]
            for it in pool_items}
    bad = sorted(i for i, f in fits.items() if not f)
    if bad:
        raise DisjointnessError("%d item(s) share material with every board, e.g. %s" % (len(bad), bad[:3]))
    by = {b: [] for b in order}
    flex = []
    for it in sorted(pool_items, key=lambda x: x["item_id"]):
        f = fits[it["item_id"]]
        if not it["is_sentinel"] or len(f) == 1:
            by[f[0]].append(it)
        else:
            flex.append(it)
    return by, flex, fits


def run(prereg, domain, funnel_dir=None, adapter=None, force=False, testing=False):
    """sheets_v1/, sheets_v1_cluster/ and sheets_v1_key/ from the locked sample."""
    from . import adapters as A
    prereg, domain = Q._load(prereg, domain)
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    if adapter is None:
        adapter = A.load(domain.adapter)
    rl = _rl_cfg(domain)
    size, n_sent = int(rl["sheet"]["items"]), int(rl["sheet"]["sentinels"])
    sample, key, planted, recs = _load_locked(prereg, funnel_dir)
    pool_dir, eval_dir, key_dir = funnel_dir / SHEETS_DIR, funnel_dir / CLUSTER_DIR, funnel_dir / KEY_DIR
    inputs = {"sample": recs["sample"], "key": recs["key"], "frames": recs["frames"]}
    existing = _existing_index(pool_dir)
    if existing is not None and not force:
        want_in = {k: v["sha256"] for k, v in inputs.items()}
        got_in = {k: (existing.get("inputs", {}).get(k) or {}).get("sha256") for k in want_in}
        if got_in == want_in and existing.get("prereg_core_sha256") == prereg.core_sha256:
            log("sheets: outputs exist for these inputs; nothing to do")
            return existing
        raise SheetError("%s exists and was made from other inputs; rerun with --force" % pool_dir)
    answers = funnel_dir / "rl_answers"
    if force and answers.is_dir() and any(answers.rglob("*.json")):
        raise SheetError("answers exist under %s; sheets they answered are never re-rendered" % answers)
    kt = adapter.known_truth(domain, funnel_dir)
    kt_ids_of = collections.defaultdict(list)
    for k, items in kt.items():
        for it in items:
            kt_ids_of[it["id"]].append(k)
    items = _items(sample, key, adapter, domain, kt_ids_of)
    options = domain.options()
    pair_options = domain.pair_options()
    loc = Locator(adapter, funnel_dir, reference=domain.reference_source)
    plans = board_plans(domain, kt)
    order = [b["id"] for b in rl["boards"]]
    check_not_exemplars(items, plans)
    pool_items = [it for it in items if it["sheet_class"] != EVAL_CLASS]
    eval_items = [it for it in items if it["sheet_class"] == EVAL_CLASS]
    by_board, flex, fits = _assign_boards(pool_items, plans, order)
    priors = stratum_priors(funnel_dir, domain)
    share = (planted or {}).get("share", 0.0)
    # sentinel need per board, then flexible sentinels where the need is largest
    slots = size - n_sent
    need = {}
    for b in order:
        nb = len([i for i in by_board[b] if not i["is_sentinel"]])
        have = len([i for i in by_board[b] if i["is_sentinel"]])
        need[b] = n_sent * int(math.ceil(nb / float(slots))) - have
    for it in flex:
        f = fits[it["item_id"]]
        b = max(f, key=lambda x: (need[x], -order.index(x)))
        by_board[b].append(it)
        need[b] -= 1
    sheets = []
    for b in order:
        if not by_board[b]:
            continue
        for s in compose(by_board[b], key, priors, domain, board_id=b, planted_share=share,
                         seed_prefix="funnel/v1/sheets/%s" % b):
            sheets.append(s)
    pair_sheets = compose_pairs(eval_items, domain) if eval_items else []
    # ids, positions, pixels
    n = 0
    for s in sheets + pair_sheets:
        n += 1
        s["sheet_id"] = "sheet_%04d" % n
        members = s["items"] + s["sentinels"]
        pos = _positions(s["sheet_id"], len(members))
        s["placed"] = []
        for it, p in zip(sorted(members, key=lambda x: x["item_id"]), pos):
            px = loc.pair(it["unit_id"]) if s in pair_sheets else loc.box(it["unit_id"], it.get("crop_id"))
            s["placed"].append(dict(it, position=p, pixels=px))
    used_boards = sorted({s["board_id"] for s in sheets if s["board_id"]})
    board_files = {}
    for bid in used_boards:
        img = render_board(plans[bid], loc)
        data = _jpeg_bytes(img)
        board_files[bid] = write_board(plans[bid], img, data, pool_dir)
    forbidden = set(domain.class_names)
    for it in items:
        forbidden.update([it["unit_id"], it["stratum"], it.get("source"), it["keys"].get("source"),
                          it.get("truth_taxon")])
    mc_q = fill_prompt(rl["sheet_prompt"], options)
    pair_q = fill_prompt(rl["pair_sheet_prompt"], pair_options)
    key_out, index_pool, index_eval, packing = [], [], [], []
    for s in sheets + pair_sheets:
        is_pair = s in pair_sheets
        if is_pair:
            if any(it["sheet_class"] != EVAL_CLASS for it in s["placed"]):
                raise SheetError("a pair sheet holds a pool item")
            doc = render_pair_sheet(s, eval_dir, options=pair_options, question=pair_q)
        else:
            if any(it["sheet_class"] == EVAL_CLASS for it in s["placed"]):  # never evaluation pixels here
                raise SheetError("an item with sheet_class eval would be written to %s" % SHEETS_DIR)
            bf = board_files[s["board_id"]]
            s["board"] = {"id": s["board_id"], "file": bf["image"], "sha256": bf["image_sha256"]}
            doc = render_sheet(s, pool_dir, options=options, question=mc_q)
        probs = blind_problems(doc, forbidden)
        if probs:
            raise SheetError("blinding: %s" % "; ".join(probs[:5]))
        d = eval_dir if is_pair else pool_dir
        js_sha = write_json_atomic(d / ("%s.json" % s["sheet_id"]), doc)
        (index_eval if is_pair else index_pool).append(
            {"sheet_id": s["sheet_id"], "json_sha256": js_sha, "image_sha256": doc["image"]["sha256"],
             "board_id": s.get("board_id"), "n_items": len(doc["items"])})
        for it in s["placed"]:
            key_out.append({"item_id": it["item_id"], "sheet_id": s["sheet_id"], "position": it["position"],
                            "unit_id": it["unit_id"], "group": it["group"], "stratum": it["stratum"],
                            "is_sentinel": bool(it["is_sentinel"]), "kt": it["kt"] or None,
                            "truth": it.get("truth"), "truth_taxon": it.get("truth_taxon"),
                            "truth_kind": it.get("truth_kind"), "pair_truth": it.get("pair_truth"),
                            "board_id": s.get("board_id")})
        packing.append({"sheet_id": s["sheet_id"], "board_id": s.get("board_id"), "kind": "pair" if is_pair else "mc",
                        "items": len(s["items"]), "sentinels": len(s["sentinels"]),
                        "target_sentinels": s.get("n_target_sentinels"), "low_prior": s.get("n_low_prior"),
                        "expected_prevalence": s.get("expected_prevalence"),
                        "below_min_prevalence": s.get("below_min_prevalence"),
                        "sentinel_shortfall": s.get("sentinel_shortfall")})
    for d, text in ((pool_dir, mc_q), (eval_dir, pair_q)):
        if (d is pool_dir and index_pool) or (d is eval_dir and index_eval):
            _atomic_write_bytes(d / "instructions.txt", (text + "\n").encode("utf-8"))
    key_out.sort(key=lambda r: (r["sheet_id"], r["position"]))
    key_sha = write_jsonl_atomic(key_dir / "key.jsonl", key_out)
    boards_meta = [{"board_id": b, "image_sha256": board_files[b]["image_sha256"],
                    "json_sha256": board_files[b]["json_sha256"]} for b in used_boards]
    pack_doc = {"format": "funnel-sheets-packing/1", "sheets": packing,
                "flexible_sentinels": len(flex), "planted_share_used": bool(planted),
                "boards": {b: {"sessions": {str(k): v for k, v in plans[b]["sessions"].items()},
                               "exemplars": {str(k): [x["id"] for x in v]
                                             for k, v in sorted(plans[b]["exemplars"].items())}}
                           for b in order}}
    write_json_atomic(key_dir / "packing.json", pack_doc)
    seeds = {"positions": "funnel/v1/sheets/<sheet_id>", "sentinels": "funnel/v1/sheets/<board>/sentinels",
             "boards": "funnel/v1/board/<board>/<option>"}
    out = {}
    for d, index, eval_pixels in ((pool_dir, index_pool, False), (eval_dir, index_eval, True)):
        if not index and d is eval_dir:
            continue
        doc = dict(header("sheets", domain, prereg, inputs, seeds=seeds, modules=(sys.modules[__name__],),
                          testing=testing),
                   contains_eval_pixels=eval_pixels, sheets=index,
                   boards=[] if eval_pixels else boards_meta, key_sha256=key_sha,
                   sample_sha256=recs["sample"]["sha256"], prereg_core_sha256=prereg.core_sha256)
        write_json_atomic(d / "index.json", doc)
        out[d.name] = doc
    log("sheets: %d pool sheet(s) on boards %s, %d pair sheet(s); key %s"
        % (len(index_pool), used_boards, len(index_eval), key_sha[:12]))
    return out.get(SHEETS_DIR) or out.get(CLUSTER_DIR)


def _existing_index(d):
    p = Path(d) / "index.json"
    if not p.is_file():
        return None
    try:
        return read_json(p)
    except FunnelError:
        return None


# ------------------------------------------------------ person (L14) items
def person_items(items, proposals, anchoring_share=0.25, blind_share=0.2, seed_text="funnel/v1/l14/person",
                 domain=None):
    """Verify-only items for a person (contract §4.3, runner §9 item 7).

    items: [{"item_id", "is_sentinel", "truth", "truth_kind", "truth_taxon"}];
    proposals: {item_id: option number} for non-sentinel items (the proposal
    shown, J1's call). A seeded blind_share of all items is asked as blind
    multiple choice (no proposal). A seeded anchoring_share of the sentinels
    (among the non-blind ones that have a configured pair) carries a wrong
    proposal drawn from the attractor pairs: a target's siblings and the
    attractors confused with it, or the targets an attractor is confused
    with. Returns rows with the visible fields (item_id, mode, proposal,
    proposal_text) and the key fields (anchor, proposal_from); person_view()
    keeps the visible ones only."""
    if domain is None:
        raise SheetError("person_items needs the domain config (its options and attractor pairs)")
    opts = domain.options()
    by_class = {o["class"]: o for o in opts if o["kind"] == "target"}
    by_taxon = {o["taxon"]: o for o in opts if o["kind"] == "attractor"}
    items = sorted(items, key=lambda x: x["item_id"])
    r = rng(seed_text)
    n_blind = int(round(blind_share * len(items)))
    blind = set(items[int(i)]["item_id"] for i in r.permutation(len(items))[:n_blind])
    sentinels = [it for it in items if it.get("is_sentinel")]

    def wrong_options(it):
        if it.get("truth_kind") == "target":
            t = domain.target(int(it["truth"]))
            out = [by_class[domain.target(s)["id"]] for s in t.get("siblings") or []]
            out += [by_taxon[a["taxon"]] for a in domain.attractors if t["name"] in (a.get("confused_with") or [])]
            return out
        if it.get("truth_kind") == "attractor":
            a = next((a for a in domain.attractors if a["taxon"] == it.get("truth_taxon")), None)
            return [by_class[domain.target(nm)["id"]] for nm in (a.get("confused_with") or [])] if a else []
        return []
    eligible = [it for it in sentinels if it["item_id"] not in blind and wrong_options(it)]
    n_anchor = min(len(eligible), int(round(anchoring_share * len(sentinels))))
    anchors = {}
    for i in r.permutation(len(eligible))[:n_anchor]:
        it = eligible[int(i)]
        cands = sorted(wrong_options(it), key=lambda o: o["n"])
        anchors[it["item_id"]] = cands[int(r.integers(0, len(cands)))]
    out = []
    for it in items:
        iid = it["item_id"]
        if iid in blind:
            out.append({"item_id": iid, "mode": "blind", "proposal": None, "proposal_text": None,
                        "anchor": False, "proposal_from": "none"})
            continue
        if iid in anchors:
            o = anchors[iid]
            out.append({"item_id": iid, "mode": "verify", "proposal": o["n"], "proposal_text": o["text"],
                        "anchor": True, "proposal_from": "attractor_pair"})
            continue
        if it.get("is_sentinel"):
            o = _truth_option(it, by_class, by_taxon, opts)
            out.append({"item_id": iid, "mode": "verify", "proposal": o["n"] if o else None,
                        "proposal_text": o["text"] if o else None, "anchor": False, "proposal_from": "truth"})
            continue
        n = proposals.get(iid)
        o = next((x for x in opts if x["n"] == n), None) if n is not None else None
        out.append({"item_id": iid, "mode": "verify" if o else "blind", "proposal": o["n"] if o else None,
                    "proposal_text": o["text"] if o else None, "anchor": False,
                    "proposal_from": "prediction" if o else "none"})
    return out


def _truth_option(it, by_class, by_taxon, opts):
    if it.get("truth_kind") == "target":
        return by_class.get(int(it["truth"]))
    if it.get("truth_kind") == "attractor":
        return by_taxon.get(it.get("truth_taxon"))
    ans = {"other": "other", "non_object": "non_object"}.get(it.get("truth_kind"))
    return next((o for o in opts if o["kind"] == "tail" and o["answer"] == ans), None) if ans else None


def person_view(rows):
    """What a person sees of person_items rows."""
    return [{"item_id": r["item_id"], "mode": r["mode"], "proposal": r["proposal"],
             "proposal_text": r["proposal_text"]} for r in rows]
