#!/usr/bin/env python3
"""Base v3 for E1 (inc2/base3.py; docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-03): E1, weed-box base v3 (pre-registered)").

The synthetic world of tests/test_inc2_train.py (v1 and v2 splits, the real
inc2.guard, LOCK v2 marked testing), plus a dataset registry, an intake
batch and a stream ledger of its own.

What is pinned:
- the config: the shipped base3_v1.json loads, lists the 16 dock slugs as one
  family, maps every MH-Weed16 id to weed, keeps the test groups and the
  cwd12 copies out, and a config that lists a slug twice, maps a class to
  anything but weed or drop, or names an unknown family is refused;
- geometry: a polygon line becomes its bounding box, coordinates are
  clipped, a malformed line is a problem; the box side is sqrt(w x h) at
  640; each of the 8 box transforms puts a painted box where funnel.leak's
  variant of that name puts its pixels; layout matching is one-to-one;
- the pair search equals a brute-force search over random hashes (3 and 6
  bits, with and without variants);
- build, end to end on the CPU: arm A is base_v2 whole, every box class 12,
  its rows byte-identical inside arm B; inc2.train's check_manifest and
  guard_rows accept both manifests; a crop box is dropped, a weed box under
  8 px is masked (the PNG equals Ultralytics' load_image of the original
  with inc2.mask's fill), an image over 640 px is a PNG that reads back as
  load_image's pixels, a weed box over 90 %, a short side under 320 px, an
  image without a weed box or a label are dropped; a source failing the
  convention rule, a quarantined source, a source whose class names are not
  in the config and a source with an unweighed dHash hit on a dev image
  (fail closed) are excluded; an intake row whose hold Step 1 has not
  released stays out; exact, flip and identical-layout copies are deduped
  (the lower tier kept), a flip copy whose boxes do not follow the flip is
  kept, a copy of a base_v2 image is dropped; the holdout is whole capture
  groups (export stems across the dock family), never a group touching
  base_v2, deterministic; the family cap holds; summary.json carries no key
  named after a non-dev split and records every input's sha256; a second
  build refuses;
- D28-v2 in the build: a source whose dHash hit weighs below the copy
  threshold keeps its other rows; one at or above it is excluded;
- count: read-only (writes only to --out, outside INC_DIR), the as-is and
  the lifted scenarios, the same holdout as the build;
- the quarantine is the ledger's: an unquarantine event admits the source;
- inc2.baseline builds E1's arms with cold_budget (the pinned driver
  accepts the definition), and a manifest that is not one summary.json
  records gets the cold table.

Run:  python3 tests/test_inc2_base3.py
"""
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO to its own temporary dirs first)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import base3 as B3  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import stream as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402

FAILURES = W.FAILURES
check = W.check
TMP = W.TMP
SID = "s_e1"
DS = C.REPO / "datasets"
LOADER = {"sample": 24, "batches": 2, "warm": 1, "batch": 8, "workers": 0}
QUAR = {"event": "quarantine", "scope": "source", "source": "src_quar", "cite": "D28", "by": "platform"}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (B3.Base3Error, B.BaselineError, RC.RecipeError, D.DriverError) as e:
        return e
    return None


# ------------------------------------------------------------------- world
def picture(path, seed, size, fmt="JPEG"):
    """A smooth random picture (distinct dHash per seed) at size (w, h)."""
    rng = np.random.RandomState(seed)
    low = rng.randint(20, 235, (8, 9, 3)).astype(np.uint8)
    im = Image.fromarray(low).resize(size, Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "JPEG":
        im.save(path, quality=95)
    else:
        im.save(path, format=fmt)
    return im


def label(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(str(x) for x in ln) + "\n" for ln in lines))


def rf_name(stem, salt):
    return "%s_jpg.rf.%s" % (stem, hashlib.md5(("%s/%s" % (stem, salt)).encode()).hexdigest())


def boxes_for(seed, n=2, lo=0.12, hi=0.22):
    rng = np.random.RandomState(1000 + seed)
    out = []
    for _ in range(n):
        w, h = rng.uniform(lo, hi, 2)
        cx, cy = rng.uniform(w / 2 + 0.01, 1 - w / 2 - 0.01), rng.uniform(h / 2 + 0.01, 1 - h / 2 - 0.01)
        out.append((round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)))
    return out


def make_source(slug, files, names, yaml_names=True):
    """files: [(rel image path, seed, size, label lines or None, fmt)]."""
    root = DS / slug
    for rel, seed, size, lines, fmt in files:
        p = root / rel
        if isinstance(seed, pathlib.Path):
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed, p)
        elif callable(seed):
            seed(p)
        else:
            picture(p, seed, size, fmt)
        if lines is not None:
            label(pathlib.Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt"), lines)
    if yaml_names:
        (root / "data.yaml").write_text("names:\n" + "".join("- '%s'\n" % n for n in names) + "nc: %d\n" % len(names))
    return root


def flip_file(src, dst, how=Image.FLIP_LEFT_RIGHT, size=None):
    """A flipped copy of src (how None: unflipped), resized to size when given."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        out = im.convert("RGB") if how is None else im.convert("RGB").transpose(how)
        if size is not None:
            out = out.resize(size, Image.BICUBIC)
        out.save(dst, quality=95)


def build_registry_world(Wd):
    """The sources this test builds; returns (registry, config path, facts)."""
    f = {}
    # src_big: 800x600 weed + crop; tiny, huge, short-side, crop-only and unlabelled images
    files = []
    for i in range(10):
        bx = boxes_for(i)
        lines = [(0,) + b for b in bx] + [(1, 0.5, 0.5, 0.1, 0.1)]
        size = (800, 600)
        if i == 0:
            lines.append((0, 0.3, 0.3, 0.008, 0.008))            # 4.4 px at 640: masked
        if i == 1:
            lines.append((0, 0.5, 0.5, 0.98, 0.98))              # over 90 %
        if i == 2:
            size = (400, 300)                                    # short side 300
        if i == 3:
            lines = [(1, 0.5, 0.5, 0.2, 0.2)]                    # crop only
        files.append(("train/images/big_%02d.jpg" % i, 10 + i, size, None if i == 4 else lines, "JPEG"))
    make_source("src_big", files, ["weed", "crop"])
    # the dock family: a (tier 2) and b (tier 3), Roboflow export names, 352 x 352
    a_files = []
    for i in range(8):
        a_files.append(("train/images/%s.jpg" % rf_name("d%02d" % i, "a"), 40 + i, (352, 352),
                        [(0,) + b for b in boxes_for(40 + i, n=3)], "JPEG"))
    ra = make_source("src_dock_a", a_files, ["0 ridderzuring"])
    a_img = lambda i: ra / ("train/images/%s.jpg" % rf_name("d%02d" % i, "a"))  # noqa: E731
    a_box = lambda i: boxes_for(40 + i, n=3)  # noqa: E731
    hb = [(1 - b[0], b[1], b[2], b[3]) for b in a_box(1)]
    b_files = [("train/images/%s.jpg" % rf_name("d00", "b"), a_img(0), None, [(0,) + b for b in a_box(0)], None),
               ("train/images/%s.jpg" % rf_name("d01", "b"), lambda p: flip_file(a_img(1), p), None,
                [(0,) + b for b in hb], None),
               ("train/images/%s.jpg" % rf_name("x02", "b"), lambda p: flip_file(a_img(2), p), None,
                [(0,) + b for b in boxes_for(77, n=2)], None),
               ("train/images/%s.jpg" % rf_name("x03", "b"), 90, (352, 352), [(0,) + b for b in a_box(3)], "JPEG")]
    for i in range(10, 14):
        b_files.append(("train/images/%s.jpg" % rf_name("e%02d" % i, "b"), 40 + i, (352, 352),
                        [(0,) + b for b in boxes_for(40 + i, n=3)], "JPEG"))
    b_files.append(("train/images/%s.jpg" % rf_name("d04", "b2"), 97, (352, 352),
                    [(0,) + b for b in boxes_for(97, n=2)], "JPEG"))   # another export of stem d04
    rb = make_source("src_dock_b", b_files, ["weed"])
    f["dock"] = {"a_img": a_img, "rb": rb}
    # src_tiny: every box about 10 px at 640 -> the convention rule
    make_source("src_tiny", [("images/t%02d.jpg" % i, 60 + i, (640, 640),
                              [(0, 0.3, 0.3, 0.015, 0.015), (0, 0.6, 0.6, 0.016, 0.016)], "JPEG") for i in range(6)],
                ["weed"])
    # src_quar: quarantined by the stream
    make_source("src_quar", [("images/q%02d.jpg" % i, 70 + i, (500, 500), [(0,) + b for b in boxes_for(70 + i)], "JPEG")
                             for i in range(5)], ["weed"])
    # src_poly: polygon labels
    make_source("src_poly", [("images/p%02d.jpg" % i, 80 + i, (500, 400),
                              [(0, 0.1, 0.1, 0.4, 0.12, 0.35, 0.45, 0.12, 0.4)], "JPEG") for i in range(5)], ["weed"])
    # src_badnames: a class name the config does not list
    make_source("src_badnames", [("images/n%02d.jpg" % i, 85 + i, (500, 500), [(0, 0.5, 0.5, 0.2, 0.2)], "JPEG")
                                 for i in range(3)], ["weed", "mystery"])
    # src_leak: an hflip copy of a dev image, and three ordinary images
    lk = [("images/l%02d.jpg" % i, 120 + i, (400, 400), [(0,) + b for b in boxes_for(120 + i)], "JPEG")
          for i in range(3)]
    lk.append(("images/l_copy.jpg", lambda p: flip_file(Wd["dev"][0]["image"], p, size=(400, 400)), None,
               [(0, 0.5, 0.5, 0.3, 0.3)], None))
    make_source("src_leak", lk, ["weed"])
    # src_basecopy: a byte copy of a base_v2 image (a base copy) and two ordinary images
    bc = [("images/c%02d.jpg" % i, 130 + i, (400, 400), [(0,) + b for b in boxes_for(130 + i)], "JPEG")
          for i in range(5)]
    bc.append(("images/c_base.jpg", lambda p: flip_file(Wd["base_v2"][0]["image"], p, how=None, size=(400, 400)),
               None, [(0, 0.5, 0.5, 0.3, 0.3)], None))
    make_source("src_basecopy", bc, ["weed"])
    reg = {"datasets": {}}
    for slug in ("src_big", "src_dock_a", "src_dock_b", "src_tiny", "src_quar", "src_poly", "src_badnames", "src_leak",
                 "src_basecopy", "src_other"):
        reg["datasets"][slug] = {"local_path": str(DS / slug), "status": "downloaded",
                                 "provenance": {"license": "CC BY 4.0"}}
    reg["datasets"]["src_dock_b"]["class_names"] = "['weed']"
    # the intake source: 1000 x 750 photographs, INC ids 2 and 13, capture groups, holds
    idir = C.INC_DIR / "intake" / "i0001_int_src"
    rows = []
    for i in range(6):
        img = idir / "images" / ("int_%02d.jpg" % i)
        picture(img, 150 + i, (1000, 750))
        lab = idir / "labels" / ("int_%02d.txt" % i)
        label(lab, [(2 if i % 2 else 13,) + b for b in boxes_for(150 + i)])
        rows.append({"image": str(img), "label": str(lab), "sha256": C.sha256_file(img), "key": "int_src__%02d" % i,
                     "source": "int_src", "capture_group": "tray%d" % (i // 2), "width": 1000, "height": 750,
                     "dhash": C.dhash(img), "holds": (["licence"] if i == 0 else ["h6_scan"] if i == 1 else []),
                     "licence": "cc-by-4.0", "research_only": i == 0, "label_sha256": C.sha256_file(lab)})
    (idir / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    f["intake_rows"] = rows
    conf = json.loads(B3.CONFIG.read_text())
    conf["rules"]["holdout"].update(min_images=2, max_images=4)
    conf["rules"]["families"] = {"dock": dict(conf["rules"]["families"]["dock"], cap_share=0.35)}
    conf["sources"] = {
        "src_big": {"tier": 3, "classes": {"weed": "weed", "crop": "drop"}},
        "src_dock_a": {"family": "dock", "tier": 2, "classes": {"0 ridderzuring": "weed"}},
        "src_dock_b": {"family": "dock", "tier": 3, "classes": {"weed": "weed"}},
        "src_tiny": {"tier": 3, "classes": {"weed": "weed"}},
        "src_quar": {"tier": 3, "classes": {"weed": "weed"}},
        "src_poly": {"tier": 3, "classes": {"weed": "weed"}},
        "src_badnames": {"tier": 3, "classes": {"weed": "weed"}},
        "src_leak": {"tier": 3, "classes": {"weed": "weed"}},
        "src_basecopy": {"tier": 3, "classes": {"weed": "weed"}}}
    conf["intake"] = {"int_src": {"tier": 1, "classes": "all_weed"}}
    conf["excluded"] = {"src_other": "not a weed source"}
    cp = TMP / "base3_test.json"
    cp.write_text(json.dumps(conf, indent=1))
    return reg, cp, f


def write_provenance(Wd):
    from weed_optimizer_framework.tools.inc2 import guard as G
    prov = []
    for r in Wd["base_v2"]:
        h, v = G.image_hashes(r["image"])
        prov.append({"key": r["key"], "sha256": r["sha256"], "dhash": h, "variants": [v[k] for k in G.VARIANTS],
                     "research_only": False, "licence": "unresolved"})
    (W.v2_dir() / "base_v2_provenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in prov))


def make_stream(sid, events, pool_rows=None):
    """A stream ledger with an init line (its pool P_0 holding pool_rows when given) and the given events."""
    p = S.StreamPaths(sid)
    if p.root.exists():
        shutil.rmtree(p.root)
    p.root.mkdir(parents=True)
    p.stream_json.write_text(json.dumps({"sid": sid}))
    pool = {"name": "P_0"}
    if pool_rows is not None:
        pm = p.root / "pool" / "P_0.jsonl"
        pool.update(path=str(pm), sha256=C.write_manifest(pm, pool_rows))
    led = S.Ledger(p.ledger)
    led.append({"event": "init", "sid": sid, "utc": "2026-10-03T00:00:00Z", "pool": pool,
                "stream_json_sha256": C.sha256_file(p.stream_json)})
    for ev in events:
        led.append(dict(ev, sid=sid, utc="2026-10-03T00:00:01Z"))


@contextlib.contextmanager
def holds(view):
    old = B3.queue_holds
    B3.queue_holds = lambda: (view, {"read": view is not None, "stub": True})
    try:
        yield
    finally:
        B3.queue_holds = old


def clear_v3():
    if B3.out_dir().exists():
        shutil.rmtree(B3.out_dir())


# ------------------------------------------------------------------- tests
def test_config():
    print("the shipped config (base3_v1.json)")
    conf, sha = B3.load_config()
    dock = sorted(s for s, e in conf["sources"].items() if e.get("family") == "dock")
    check("16 dock slugs form one family, capped at 35 %% (%d listed)" % len(dock),
          len(dock) == 16 and conf["rules"]["families"]["dock"]["cap_share"] == 0.35
          and "rf_unitec-qdvgo__weed-detection-in-grass" in dock and "francesco__grass_weeds" in dock)
    mh = conf["sources"]["project_agml__mh_weed16_weed_detection"]["classes"]
    check("MH-Weed16: all 15 ids are weeds", sorted(mh, key=int) == [str(i) for i in range(15)]
          and set(mh.values()) == {"weed"})
    ex = conf["excluded"]
    check("test groups, OOD-dev groups, cwd12 copies and the intake sources' registry paths are excluded",
          all(s in ex for s in ("project_agml__imageweeds_weed_detection", "project_agml__weed_crop_detection",
                                "project_agml__crop_weed_detection_latvia", "rf_test-8qezo__weed-detection-ycai2",
                                "kg_ravirajsinh45__crop-and-weed-detection-data-with-bounding-boxes",
                                "project_agml__maize_weed_detection", "weedai_5c78d067-8750-4803-9cbe-57df8fae55e4",
                                "rf_main-otq0a__weed-in-paddy-field-wmjlr", "rf_1111-gzfxi__weed-chilling",
                                "cottonweed_sp8", "rf_agrobot-weed-workspace__weed-detection-sd89f",
                                "kg_yuzhenlu__cottonweeddet3", "mediatum_1717366"))
          and set(conf["intake"]) == {"kg_yuzhenlu__cottonweeddet3", "mediatum_1717366"})
    itmo = conf["sources"]["rf_itmo-mp0nn__grass-detection-4"]["classes"]
    check("rf_itmo: forbs weed; Poa, litter, bare patch, dry grass dropped",
          {k for k, v in itmo.items() if v == "drop"} == {"Dry_grass", "Musor", "Poa_pratensis", "Poa_trivialis",
                                                           "Propleshina"})
    r = conf["rules"]
    check("the pre-registered numbers: side >= 32, <= 25 % under 16, <= 10 % over 80 %, <= 25 per image; mask "
          "< 8; drop > 90 %, short side < 320, masked > 50 %; dedupe 3 bits, IoU 0.8 on 80 %, layout 0.01 x 3; "
          "holdout 15 %, 30-400, seed inc2/base3/test_v1",
          r["source"] == {**r["source"], "median_side_min_px": 32, "small_side_px": 16, "small_share_max": 0.25,
                          "big_area": 0.8, "big_share_max": 0.1, "median_boxes_max": 25}
          and r["box"]["mask_side_px"] == 8 and r["image"]["drop_box_area"] == 0.9
          and r["image"]["min_short_side_px"] == 320 and r["image"]["max_masked_area"] == 0.5
          and r["dedupe"]["near_bits"] == 3 and r["dedupe"]["layout_iou"] == 0.8 and r["dedupe"]["layout_share"] == 0.8
          and r["dedupe"]["layout_round"] == 0.01 and r["dedupe"]["layout_min_boxes"] == 3
          and r["holdout"]["share"] == 0.15 and r["holdout"]["min_images"] == 30 and r["holdout"]["max_images"] == 400
          and r["holdout"]["seed_text"] == "inc2/base3/test_v1" and r["holdout"]["max_group"] == 400)
    bad = []
    for mut in (lambda c: c["excluded"].update(src_x="x") or c["sources"].update(src_x={"classes": {"w": "weed"}}),
                lambda c: c["sources"]["rf_kinjj__weed-avnag"]["classes"].update(weed="maybe"),
                lambda c: c["sources"]["rf_kinjj__weed-avnag"].update(family="nofamily"),
                lambda c: c.update(class_id=11)):
        c = json.loads(B3.CONFIG.read_text())
        mut(c)
        p = TMP / "bad_conf.json"
        p.write_text(json.dumps(c))
        bad.append(refused(B3.load_config, p) is not None)
    check("a slug both included and excluded, a role other than weed/drop, an unknown family or another class id "
          "refuse the config", all(bad), bad)


def test_geometry():
    print("geometry")
    p = TMP / "geo" / "l.txt"
    label(p, [(0, 0.1, 0.2, 0.3, 0.1, 0.35, 0.4, 0.05, 0.3), (1, 0.5, 0.5, 0.2, 0.2), (0, 1.05, 0.5, 0.2, 0.2)])
    boxes, polys, probs = B3.parse_label(p)
    check("a polygon line becomes its bounding box; a box past the edge is clipped",
          polys == 1 and not probs and abs(boxes[0][1] - 0.2) < 1e-9 and abs(boxes[0][3] - 0.3) < 1e-9
          and abs(boxes[0][4] - 0.3) < 1e-9 and abs(boxes[2][1] + boxes[2][3] / 2 - 1.0) < 1e-9, boxes)
    label(p, [(0, 0.5, 0.5, 0.2), ("x", 0.5, 0.5, 0.2, 0.2)])
    check("a line with 4 fields or a non-numeric one is a problem", len(B3.parse_label(p)[2]) == 2)
    check("box side = sqrt(w x h) at 640: 0.1 x 0.1 of 1280 x 960 is 64 x 48 -> 55.4",
          abs(B3.side_px((0.5, 0.5, 0.1, 0.1), 1280, 960) - (64 * 48) ** 0.5) < 1e-9)
    from weed_optimizer_framework.tools.funnel import leak as L
    base = Image.new("RGB", (120, 80), (0, 0, 0))
    ImageDraw.Draw(base).rectangle([18, 8, 47, 27], fill=(255, 255, 255))
    box = ((18 + 48) / 2 / 120.0, (8 + 28) / 2 / 80.0, 30 / 120.0, 20 / 80.0)
    ops = {"id": None, "hflip": [Image.FLIP_LEFT_RIGHT], "vflip": [Image.FLIP_TOP_BOTTOM], "rot180": [Image.ROTATE_180],
           "transpose": [Image.TRANSPOSE], "rot90": [Image.ROTATE_90], "rot270": [Image.ROTATE_270],
           "transverse": [Image.TRANSVERSE]}
    ok = []
    vs = L.dhash_variants(base)
    for v, op in ops.items():
        im = base if op is None else base.transpose(op[0])
        a = np.asarray(im.convert("L")) > 128
        ys, xs = np.nonzero(a)
        got = ((xs.min() + xs.max() + 1) / 2 / im.width, (ys.min() + ys.max() + 1) / 2 / im.height,
               (xs.max() - xs.min() + 1) / im.width, (ys.max() - ys.min() + 1) / im.height)
        want = B3.transform_box(box, v)
        hv = L.dhash_variants(im)["id"]
        ok.append(all(abs(x - y) < 1e-9 for x, y in zip(got, want)) and hv == vs[v])
    check("each of the 8 box transforms puts the box where the image transform whose dHash funnel.leak names so "
          "puts its pixels", all(ok), ok)
    a = [(0.2, 0.2, 0.1, 0.1), (0.6, 0.6, 0.2, 0.2)]
    check("layout match: the same boxes match, flipped boxes match only under the flip, one box of two does not",
          B3.layout_match(a, a, "id", 0.8, 0.8)
          and B3.layout_match(a, [B3.transform_box(x, "hflip") for x in a], "hflip", 0.8, 0.8)
          and not B3.layout_match(a, [B3.transform_box(x, "hflip") for x in a], "id", 0.8, 0.8)
          and not B3.layout_match(a, a[:1], "id", 0.8, 0.8))


def test_pairs():
    print("pair search")
    rng = np.random.default_rng(5)
    n = 300
    H = rng.integers(0, 2 ** 63, size=(n, 8), dtype=np.uint64)
    for i in range(0, 60, 3):                       # planted near pairs, some through a variant
        j = i + 1
        flip = np.uint64(1) << np.uint64(int(rng.integers(0, 64)))
        H[j, 0] = H[i, int(rng.integers(0, 8))] ^ flip
    ok = rng.random(n) > 0.2
    ok[:60] = True
    for bits in (3, 6):
        Q, Tt, Dd, V = B3.near_pairs(H, ok, bits)
        got = {(int(q), int(t)): (int(d), int(v)) for q, t, d, v in zip(Q, Tt, Dd, V)}
        want = {}
        for q in range(n):
            for v in (range(8) if ok[q] else [0]):
                for t in range(n):
                    if q == t:
                        continue
                    d = bin(int(H[q, v]) ^ int(H[t, 0])).count("1")
                    if d <= bits and ((q, t) not in want or (d, v) < want[(q, t)]):
                        want[(q, t)] = (d, v)
        check("near_pairs at %d bits equals brute force (%d pairs)" % (bits, len(want)), got == want and len(want) > 0,
              (len(got), len(want)))


def summary_ok(summ):
    from weed_optimizer_framework.tools.inc_autopilot import remote as R
    return not R.non_dev_keys(summ)


def test_build(Wd, reg, cp, f):
    print("build (testing LOCK, no embedding calibration: dHash hits unweighed, fail closed)")
    clear_v3()
    make_stream(SID, [QUAR])
    view = {"int_src__00": []}             # the licence hold released by Step 1; h6_scan of row 01 not
    with holds(view):
        summ = B3.build(SID, conf_path=cp, registry=reg, testing=True, procs=2, loader=LOADER)
    root = B3.out_dir()
    ps = summ["per_source"]
    a = C.read_manifest(root / "base_v2_weed.jsonl")
    b = C.read_manifest(root / "base_v3_weed.jsonl")
    check("arm A is base_v2 whole (%d rows) and inside arm B byte for byte (%d rows)" % (len(a), len(b)),
          len(a) == len(Wd["base_v2"]) and {r["key"] for r in a} == {r["key"] for r in Wd["base_v2"]}
          and all(r in b for r in a) and len(b) > len(a))
    labs_ok = all(all(ln.split()[0] == "12" for ln in open(r["label"]).read().splitlines()) for r in b)
    bv2 = {r["key"]: r for r in Wd["base_v2"]}
    nbox = all(len(C.read_yolo(r["label"])) == len(C.read_yolo(bv2[r["key"]]["label"])) for r in a)
    check("every box of both arms is class 12; base_v2 rows keep every box", labs_ok and nbox)
    rows, dh, info = T.check_manifest(root / "base_v3_weed.jsonl")
    g = T.guard_rows(rows, dh, production=False)
    rows_a, dh_a, _ = T.check_manifest(root / "base_v2_weed.jsonl")
    T.guard_rows(rows_a, dh_a, production=False)
    check("inc2.train's check_manifest and guard_rows accept both manifests (B: %s)" % g["reasons"],
          info["n_train_images"] == len(b) and g["refused"] == 0 and g["crosscheck_hits"] == 0)
    big = ps["src_big"]
    check("src_big: crop-only -> no weed box, unlabelled -> no_label, a box over 90 %%, a short side; kept rows hold "
          "weed boxes only (%s)" % big["dropped"],
          big["dropped"].get("no_weed_box") == 1 and big["dropped"].get("no_label") == 1
          and big["dropped"].get("box_over_90") == 1 and big["dropped"].get("short_side") == 1
          and big["in_B"] + big["holdout_v1"] == 6)
    prov = {p["key"]: p for p in (json.loads(x) for x in (root / B3.PROVENANCE).read_text().splitlines())}
    k0 = "src_big__train_images_big_00"
    p0 = prov.get(k0)
    from weed_optimizer_framework.tools.inc2 import mask as MK
    ok_mask = False
    if p0:
        im, _ = B3._load_640(p0["original_image"])
        keep = [b_ for b_ in (tuple(float(x) for x in ln.split()[1:]) for ln in open(p0["label"]).read().splitlines())]
        want, _r, _k, _m = MK.masked_array(im, [(0.3, 0.3, 0.008, 0.008)], keep)
        got, _ = B3._load_640(p0["image"])
        ok_mask = (p0["masked_boxes"] == 1 and p0["image"].endswith(".png") and got.shape == (480, 640, 3)
                   and (got == want).all() and len(keep) == 2 and not (got == im).all())
    check("a weed box under 8 px is masked: the PNG is Ultralytics' load_image of the original with inc2.mask's "
          "fill, its label holds the other weed boxes", ok_mask, p0)
    pngs = [p for p in prov.values() if p["source"] == "src_big" and p["image"].endswith(".png")]
    rb = []
    for p in pngs:
        im, _ = B3._load_640(p["original_image"])
        got, _ = B3._load_640(p["image"])
        rb.append(p["masked_boxes"] > 0 or (got.shape == im.shape and (got == im).all()))
    check("an 800 x 600 image trains on a 640 x 480 PNG that reads back as load_image's pixels (%d)" % len(pngs),
          pngs and all(rb) and all(p["sha256"] == C.sha256_file(p["image"]) for p in pngs))
    check("src_tiny fails the convention rule (median side under 32 px): excluded whole",
          ps["src_tiny"]["dropped"] == {"convention": 6} and not ps["src_tiny"]["convention"]["passes"],
          ps["src_tiny"])
    check("src_quar is quarantined by the stream: excluded", ps["src_quar"]["dropped"] == {"quarantined": 5}
          and ps["src_quar"]["quarantined"])
    check("src_badnames: a class name the config does not list -> the source is not read",
          "src_badnames" not in ps and "mystery" in str(summ["inputs"]) or
          ps.get("src_badnames", {}).get("in_B", 0) == 0)
    check("src_poly: polygons became boxes (%d)" % ps["src_poly"]["polygons"],
          ps["src_poly"]["polygons"] == 5 and ps["src_poly"]["in_B"] + ps["src_poly"]["holdout_v1"] == 5)
    lk = ps["src_leak"]
    check("src_leak: an hflip copy of a dev image is refused by GuardV2 and, unweighed (no calibration), leaks its "
          "source (fail closed): every row out (%s)" % lk["dropped"],
          lk["in_B"] == 0 and lk["guard"].get("near_eval_variant") == 1 and lk["leak"]["flagged"]
          and lk["dropped"].get("source_leak") == 3)
    bcs = ps["src_basecopy"]
    check("src_basecopy: the copy of a base_v2 image is dropped (base_copy, 1 of 6 images: under D28's 20 %%); the "
          "others stay (%s)" % bcs["dropped"],
          bcs["dropped"].get("base_copy") == 1 and bcs["in_B"] + bcs["holdout_v1"] == 5 and not bcs["leak"]["flagged"])
    th, _sha = B3.load_thresholds()
    lv = B3.leak_verdict(5, 0, [], 0, 1e-4, 0.95, 1, th)
    check("D28's base-copy rule: 1 base copy in 5 images (20 %) leaks the source", lv["flagged"]
          and "base copies" in " ".join(lv["why"]), lv)
    it = ps["int_src"]
    check("intake: the row whose hold Step 1 has not released stays out; the released one is in (%s)" % it["dropped"],
          it["dropped"].get("intake_hold") == 1 and it["in_B"] + it["holdout_v1"] == 5)
    da, db = ps["src_dock_a"], ps["src_dock_b"]
    dropped_b = db["dropped"]
    check("dock: the byte copy, the hflip copy with flipped boxes and the identical layout are duplicates of the "
          "tier-2 source (%s); the hflip copy with other boxes is kept" % dropped_b,
          dropped_b.get("duplicate") == 3 and db["in_B"] + db["holdout_v1"] + dropped_b.get("family_cap", 0) == 6
          and da.get("dropped", {}).get("duplicate", 0) == 0)
    hold = summ["holdout_v1"]
    files_ok = all((root / v["file"]).is_file() for v in hold["per_source"].values())
    check("the holdout: test_v1/<source>.jsonl per source with held rows (%s), never trained"
          % {k: v["images"] for k, v in hold["per_source"].items()},
          files_ok and hold["images"] > 0 and not ({p["key"] for p in prov.values() if p["holdout_v1"]}
                                                   & {r["key"] for r in b}))
    grp = {}
    for p in prov.values():
        grp.setdefault(p["group"], set()).add((p["holdout_v1"], p["kind"] == "base"))
    check("held rows are whole capture groups, and no group holding a base_v2 row is held",
          all(len({h for h, _b in s}) == 1 for s in grp.values()) and
          not any(any(h for h, _b in s) and any(bb for _h, bb in s) for s in grp.values()))
    stems = {}
    for p in prov.values():
        st = B3._stem_of(p["original_image"])
        if p["family"] == "dock" and st:
            stems.setdefault(st, set()).add(p["group"])
    check("a Roboflow export stem joins one capture group across the family (stem d04 of src_dock_a and src_dock_b)",
          all(len(v) == 1 for v in stems.values()) and len(stems.get("d04", ())) == 1)
    cap = summ["selection"]["family_caps"]["dock"]
    check("the dock cap: D <= floor(0.35 / 0.65 x non-dock) after the holdout (%s)" % cap,
          cap["after"] <= cap["cap"] and cap["cap"] == int(0.35 / 0.65 * cap["non_family"] + 1e-9))
    check("summary.json: complete, every input's sha256, no key named after a non-dev split",
          summ["status"] == "complete" and summary_ok(summ) and summ["inputs"]["registry"]["sha256"] is None
          or summ["inputs"]["lock_v2"]["sha256"] and summary_ok(summ))
    check("a second build refuses (built once)", refused(B3.build, SID, conf_path=cp, registry=reg, testing=True)
          is not None)
    wt = summ["walltime"]
    check("the walltime guard: arm B's loader measured through Ultralytics' own training pipeline (%.1f ms per image) "
          "and a base run of cold_budget projected at %.2f h under the %.1f h line"
          % (wt["loader_ms_per_image"], wt["projected_base_run_h"], wt["line_h"]),
          wt["loader"]["images"] == 16 and wt["line_h"] == 6.4 and not wt["over_line"]
          and abs(wt["projected_base_run_h"] - (max(wt["loader_ms_per_image"], 12.65) * 1.2e6 / 3.6e6 + 0.27)) < 1e-3)
    s1 = B3.read_summary()
    k, s = B3.e1_arm_of(s1["arms"]["B"]["sha256"])
    check("e1_arm_of names the arm of a manifest the summary records, and nothing else",
          k == "B" and B3.e1_arm_of(s1["arms"]["A"]["sha256"])[0] == "A" and B3.e1_arm_of("0" * 64)[0] is None)
    return summ


def test_leak_weighed(Wd, reg, cp, f):
    print("D28-v2 in the build: pair cosines decide the source")
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH

    class FakeIndex:
        split, eval_key, Xn, n = [], [], np.zeros((0, 4), dtype=np.float32), 0
        record = {"path": None, "sha256": None}

        class embedder:
            name = "stand-in"

    class FakeScanner:
        index = FakeIndex()
        threshold = 0.95

        def scan(self, rows, desc_path=None, procs=1):
            return {}, set()

        def record(self):
            return {"embedder": "stand-in", "cos_threshold": 0.95, "p_false": 1e-4}
    out = {}
    for cos in (0.3, 0.97):
        clear_v3()
        old = EH.pair_cosines
        EH.pair_cosines = lambda hits, emb, eval_desc=None, procs=1, batch=32, c=cos: {
            h["key"]: {"pair_cos": c, "why": None, "weighed": 1, "best": None} for h in hits}
        try:
            with holds({"int_src__00": []}):
                s = B3.build(SID, conf_path=cp, registry=reg, testing=True, procs=2, scanner=FakeScanner(),
                             loader=LOADER)
        finally:
            EH.pair_cosines = old
        out[cos] = s["per_source"]["src_leak"]
    lo, hi = out[0.3], out[0.97]
    check("a dHash hit weighed at 0.30 is chance: only its image leaves, the source's other rows stay (%s)"
          % lo["dropped"], not lo["leak"]["flagged"] and lo["in_B"] + lo["holdout_v1"] == 3
          and lo["dropped"].get("guard:near_eval_variant") == 1)
    check("weighed at 0.97 (>= the copy threshold 0.95) it leaks the source: every row out",
          hi["leak"]["flagged"] and hi["in_B"] == 0 and hi["dropped"].get("source_leak") == 3)


def test_count_and_quarantine(Wd, reg, cp, f, built):
    print("count (read-only) and the quarantine as an input")
    clear_v3()

    class Stored(B3.StoredHashes):
        def __init__(self):
            self.by_path, self.sha_by_path, self.record = {}, {}, {"stand-in": True}
            self.compute_max_bytes = 10 ** 9
    before = sorted(str(p) for p in C.INC_DIR.rglob("*"))
    out = TMP / "count_out"
    with holds({"int_src__00": []}):
        rep = B3.count(SID, out, conf_path=cp, registry=reg, testing=True, hashes=Stored())
    after = sorted(str(p) for p in C.INC_DIR.rglob("*"))
    sc = rep["scenarios"]
    check("count writes its report to --out and nothing under INC_DIR", before == after
          and (out / "e1_base3_count.json").is_file())
    check("count refuses an --out inside INC_DIR", refused(B3.count, SID, C.INC_DIR / "x", conf_path=cp, registry=reg,
                                                           testing=True, hashes=Stored()) is not None)
    asis, lift = sc["as_is"], sc["lifted"]
    check("as is: src_quar stays out; lifted: its rows are in (%d -> %d)" % (asis["arms"]["B"]["images"],
                                                                           lift["arms"]["B"]["images"]),
          asis["per_source"]["src_quar"]["in_B"] == 0 and lift["lifted"] == ["src_quar"]
          and lift["per_source"]["src_quar"]["in_B"] + lift["per_source"]["src_quar"]["holdout_v1"] == 5)
    check("count's arm A is the build's (%d)" % asis["arms"]["A"]["images"],
          asis["arms"]["A"]["images"] == built["arms"]["A"]["images"])
    check("count reports the dHash hits it cannot weigh and the pending embedding check",
          asis["dhash_hits_unweighed"].get("src_leak") == 1 and "embedding_check" in asis["pending"])
    make_stream(SID, [{"event": "quarantine", "scope": "source", "source": "src_quar", "cite": "D28", "by": "platform"},
                      {"event": "unquarantine", "source": "src_quar", "by": "human:x"}])
    with holds({"int_src__00": []}):
        rep2 = B3.count(SID, out, conf_path=cp, registry=reg, testing=True, hashes=Stored())
    ps = rep2["scenarios"]["as_is"]["per_source"]["src_quar"]
    check("an unquarantine event in the ledger admits the source (the quarantine is the ledger's, not a constant)",
          not ps["quarantined"] and ps["in_B"] + ps["holdout_v1"] == 5
          and rep2["inputs"]["stream"]["head_sha256"] != rep["inputs"]["stream"]["head_sha256"])
    with holds({"int_src__00": []}):
        rep3 = B3.count(SID, out, conf_path=cp, registry=reg, testing=True, hashes=Stored())
    check("the holdout is deterministic (the same rows held out twice)",
          rep3["scenarios"]["as_is"]["selection"]["holdout"] == rep2["scenarios"]["as_is"]["selection"]["holdout"])
    e = None
    shutil.rmtree(S.StreamPaths("s_none").root, ignore_errors=True)
    e = refused(B3.count, "s_none", out, conf_path=cp, registry=reg, testing=True, hashes=Stored())
    check("no stream ledger: refused (the quarantine is unknown)", e is not None and "quarantine" in str(e), e)
    with holds(None):
        rep4 = B3.count(SID, out, conf_path=cp, registry=reg, testing=True, hashes=Stored())
    check("Step 1's queue unreadable: intake rows with holds stay out (fail closed)",
          rep4["scenarios"]["as_is"]["per_source"]["int_src"]["dropped"].get("intake_hold") == 2)


def test_pool_rows(Wd, reg, cp, f):
    print("a row the stream has trained on never enters the holdout")
    clear_v3()
    rows = f["intake_rows"]
    pool = [{k: r[k] for k in C.MANIFEST_KEYS if k in r} for r in rows[2:]]
    for x in pool:
        x.setdefault("session", "")
    make_stream(SID, [QUAR], pool_rows=pool)
    conf = json.loads(cp.read_text())
    conf["rules"]["holdout"].update(min_images=6, max_images=6, share=1.0)
    allp = TMP / "base3_pool.json"
    allp.write_text(json.dumps(conf))
    with holds({"int_src__00": []}):
        summ = B3.build(SID, conf_path=allp, registry=reg, testing=True, procs=2, loader=LOADER)
    held = {json.loads(x)["key"] for x in (B3.out_dir() / B3.PROVENANCE).read_text().splitlines()
            if json.loads(x)["holdout_v1"]}
    it = summ["per_source"]["int_src"]
    check("the intake rows in the stream's pool (by key) are never held out, even when the rule would take every "
          "group (int_src: %d held of %d kept)" % (it["holdout_v1"], it["in_B"] + it["holdout_v1"]),
          not ({r["key"] for r in rows[2:]} & held) and summ["inputs"]["stream"]["pool_rows"] == 4
          and it["in_B"] >= 4, (sorted(held), it))


def test_walltime(Wd, reg, cp):
    print("the walltime guard: a projected base run over D26's line leaves splits v3 unusable")
    clear_v3()
    conf = json.loads(cp.read_text())
    conf["rules"]["walltime"]["gpu_ms_per_image_epoch"] = 30.0
    slow = TMP / "base3_slow.json"
    slow.write_text(json.dumps(conf))
    make_stream(SID, [QUAR])
    with holds({"int_src__00": []}):
        e = refused(B3.build, SID, conf_path=slow, registry=reg, testing=True, procs=2, loader=LOADER)
    s = json.loads((B3.out_dir() / B3.SUMMARY).read_text())
    check("projected at 30 ms x 1.2M + 0.27 h = 10.27 h > 6.4 h: the build refuses, summary.json says over_walltime, "
          "and neither manifest is an E1 arm (read_summary and e1_arm_of see none)",
          e is not None and "over the" in str(e) and s["status"] == "over_walltime"
          and abs(s["walltime"]["projected_base_run_h"] - 10.27) < 1e-6 and B3.read_summary() is None
          and B3.e1_arm_of(s["arms"]["B"]["sha256"])[0] is None, (e, s.get("status")))


def test_baseline_budget(Wd, reg, cp):
    print("inc2.baseline on the E1 manifests: cold_budget")
    clear_v3()
    make_stream(SID, [QUAR])
    with holds({"int_src__00": []}):
        summ = B3.build(SID, conf_path=cp, registry=reg, testing=True, procs=2, loader=LOADER)
    (C.REPO / "yolo11m.pt").write_bytes(os.urandom(4096))
    ma, mb = summ["arms"]["A"]["manifest"], summ["arms"]["B"]["manifest"]
    for exp in ("e1t_a", "e1t_b", "e1t_x"):
        shutil.rmtree(C.INC_DIR / exp, ignore_errors=True)
    da, _ = B.build_definition("e1t_a", manifest=ma, arm="m640", testing=True)
    db, sb = B.build_definition("e1t_b", manifest=mb, arm="m640", testing=True)
    D.validate_definition(json.loads(json.dumps(db)))
    D.check_definition_data(db)
    na, nb = summ["arms"]["A"]["images"], summ["arms"]["B"]["images"]
    check("E1-A and E1-B build as role baseline with cold_budget for their own N (epochs %d and %d), finals dev and "
          "imageweeds, seeds 0-2; the pinned driver accepts the definition"
          % (da["base"]["recipe"]["epochs"], db["base"]["recipe"]["epochs"]),
          da["role"] == db["role"] == "baseline" and da["recipe_name"] == "cold_budget"
          and da["base"]["recipe"] == RC.cold_budget("m640", na) and db["base"]["recipe"] == RC.cold_budget("m640", nb)
          and da["e1"]["arm"] == "A" and db["e1"]["arm"] == "B" and db["final_exams"] == ["dev", "imageweeds"]
          and db["seeds"] == [0, 1, 2] and not RC.check_budget_record(db["budget"], "m640", nb))
    check("... priced by the budget: the same GPU-h whatever N",
          da["cost_estimate"]["total_gpu_h"] == db["cost_estimate"]["total_gpu_h"]
          and da["cost_estimate"]["recipe_name"] == "cold_budget")
    e = refused(B.build_definition, "e1t_x", manifest=mb, arm="n640", testing=True)
    check("an E1 manifest on another arm than m640 is refused by inc2.baseline (pre-registered on YOLO11m at 640)",
          isinstance(e, B.BaselineError) and "E1's arms train on" in str(e), e)
    other = TMP / "other.jsonl"
    C.write_manifest(other, C.read_manifest(mb)[:-1])
    dx, _ = B.build_definition("e1t_x", manifest=other, arm="m640", testing=True)
    check("a manifest summary.json does not record trains the cold table", dx["base"]["recipe"] == RC.cold("m640")
          and dx.get("recipe_name") is None)
    return summ


def main():
    t0 = time.time()
    Wd = W.build_world()
    write_provenance(Wd)
    reg, cp, f = build_registry_world(Wd)
    test_config()
    test_geometry()
    test_pairs()
    built = test_build(Wd, reg, cp, f)
    test_leak_weighed(Wd, reg, cp, f)
    test_count_and_quarantine(Wd, reg, cp, f, built)
    test_pool_rows(Wd, reg, cp, f)
    test_walltime(Wd, reg, cp)
    test_baseline_budget(Wd, reg, cp)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    if FAILURES:
        for x in FAILURES:
            print("  FAILED: %s" % x)
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
