#!/usr/bin/env python3
"""Base v3 for E1 (inc2/base3.py; docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-03): E1, weed-box base v3 (pre-registered)").

The synthetic world of tests/test_inc2_train.py (v1 and v2 splits, the real
inc2.guard, LOCK v2 marked testing), plus a dataset registry, an intake
batch and a stream ledger of its own.

What is pinned:
- the config: the shipped config (base3_v2.json, revision 3) loads, lists
  the 16 dock slugs as one family, maps every MH-Weed16 id to weed, keeps
  the test groups and the cwd12 copies out, groups SIU's frames by video
  (group_regex on real frame names), and a config that lists a slug twice,
  maps a class to anything but weed or drop, names an unknown family, or
  has a group_regex that captures no group (registry or intake) is refused;
- geometry: a polygon line becomes its bounding box, coordinates are
  clipped, a malformed line is a problem; the box side is sqrt(w x h) at
  640; each of the 8 box transforms puts a painted box where funnel.leak's
  variant of that name puts its pixels; layout matching is one-to-one;
- the pair search equals a brute-force search over random hashes (3 and 6
  bits, with and without variants), within one set and between two (both
  directions);
- each rule on its own (synthetic rows): the source rule's big-box share,
  boxes per image and small-box share each fail a source alone; a masked
  area over 50 % drops an image; a class id beyond the name list refuses
  it; an index cross-check hit alone drops a row; an embedding refusal drops
  its row; the walltime guard takes the slower of loader and GPU;
- selection (synthetic rows): 4-6 bit pairs are kept but grouped, a pair
  near only under a variant is grouped, the higher resolution of two byte
  copies is kept, two base_v2 copies are both kept, the intake capture group,
  the file-name session and a family's export stem each join a group, an
  identical layout is a copy only within 10 bits, group names come from
  content; the holdout takes exactly round_half_up(15 %) of singleton rows,
  never passes max_images for any source (rows held in other sources'
  groups count against that source's max and its target), never holds a
  group over max_group; the quarantine, a new source of its own photos and
  the rows' order leave test v1 unchanged; a quarantined copy's best
  non-quarantined copy enters arm B; rows an earlier test list holds are
  held first or never trained;
- the evaluation-group guard: within 6 bits in either direction dropped, 7
  bits kept, base_v2 counted not dropped;
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
  the lifted scenarios with one test v1, the same holdout as the build;
- earlier test lists (splits/*/test_v1): a rebuild after the quarantine is
  lifted holds the same test v1 and trains none of it; a rebuild with a new
  row holds every earlier test row again or never trains it; a partial
  build's test lists block a rebuild;
- the evaluation groups end to end: an hflip copy of a src_big image in an
  evaluation group's slug drops it (near_eval_group), summary.json counts it
  and names the slug that is not in the registry;
- the quarantine is the ledger's: an unquarantine event admits the source;
- the rows of an increment in flight count as pool rows: an increment cut
  and not committed (in_segment), or accepted and rolled back (suspect), or
  of a status the builder does not know, is never held out to test v1, and
  the build records their count; a withdrawn, 'data' or returned increment's
  rows are not pool rows;
- an intake source's group_regex (SIU, revision 3): rows named like the real
  intake (weed_dataset/Dataset/images/<split>/<EPPO>_week_<n>_IMG_<id>_
  frame_<k>, the intake capture group the single image) get the video as
  their session, so a video's frames are one capture group: the holdout and
  the siu family cap take whole videos, where without it a video's frames
  were split between test v1 and arm B;
- inc2.baseline builds E1's arms with cold_budget (the pinned driver
  accepts the definition), a manifest no base v3 summary records and that
  lies outside splits/v3 gets the cold table, and a base v3 manifest that is
  not a complete summary's E1 arm (over_walltime, edited, another arm, a
  union, an unrecorded file under splits/v3) is refused.

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


def near_copy(src, dst, lo=5, hi=9):
    """A copy of src with a painted rectangle whose dHash lies lo..hi bits from src's under every one of the 8
    variants (both directions): beyond dedupe's 3 bits, within the layout rule's 10."""
    from weed_optimizer_framework.tools.inc2 import guard as G
    hs, vs = G.image_hashes(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for frac in (0.12, 0.16, 0.2, 0.25, 0.3, 0.35):
        for pos in ((0.1, 0.1), (0.5, 0.2), (0.2, 0.6), (0.6, 0.6), (0.35, 0.35)):
            for gray in (0, 255, 128):
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    x0, y0 = int(pos[0] * w), int(pos[1] * h)
                    ImageDraw.Draw(im).rectangle([x0, y0, x0 + int(frac * w), y0 + int(frac * h)], fill=(gray,) * 3)
                    im.save(dst, quality=95)
                hd, vd = G.image_hashes(dst)
                d = min([bin(vs[v] ^ hd).count("1") for v in vs] + [bin(vd[v] ^ hs).count("1") for v in vd])
                if lo <= d <= hi:
                    return d
    raise RuntimeError("no near copy of %s within %d-%d bits" % (src, lo, hi))


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
               ("train/images/%s.jpg" % rf_name("x03", "b"), 90, (352, 352), [(0,) + b for b in a_box(3)], "JPEG"),
               ("train/images/%s.jpg" % rf_name("x05", "b"), lambda p: f.__setitem__("x05_bits", near_copy(a_img(5), p)),
                None, [(0,) + b for b in a_box(5)], None)]
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
    # src_session: DeepWeeds-like capture times in the file names (group_regex): s1_* and s2_* are two sessions
    make_source("src_session", [("images/s%d_%02d.jpg" % (i // 3 + 1, i), 160 + i, (400, 400),
                                 [(0,) + b for b in boxes_for(160 + i)], "JPEG") for i in range(6)], ["weed"])
    # src_evalgrp: an evaluation group's slug holding an hflip copy of src_big's big_06 (excluded by name; its
    # images guard every candidate row)
    make_source("src_evalgrp", [("images/e00.jpg", lambda p: flip_file(DS / "src_big" / "train/images/big_06.jpg", p,
                                                                         size=(640, 480)), None, None, None),
                                ("images/e01.jpg", 170, (400, 400), None, "JPEG")], ["weed"])
    reg = {"datasets": {}}
    for slug in ("src_big", "src_dock_a", "src_dock_b", "src_tiny", "src_quar", "src_poly", "src_badnames", "src_leak",
                 "src_basecopy", "src_other", "src_session", "src_evalgrp"):
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
        "src_basecopy": {"tier": 3, "classes": {"weed": "weed"}},
        "src_session": {"tier": 3, "classes": {"weed": "weed"}, "group_regex": "^(s\\d)_"}}
    conf["intake"] = {"int_src": {"tier": 1, "classes": "all_weed"}}
    conf["excluded"] = {"src_other": "not a weed source", "src_evalgrp": "evaluation group G1",
                        "src_unregistered": "evaluation group G1 (not in the registry)"}
    conf["evaluation_groups"] = dict(conf["evaluation_groups"], groups={"G1": ["src_evalgrp", "src_unregistered"]})
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
    print("the shipped config (base3_v2.json)")
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
                                "kg_yuzhenlu__cottonweeddet3", "mediatum_1717366", "zenodo_15808623"))
          and set(conf["intake"]) == {"kg_yuzhenlu__cottonweeddet3", "mediatum_1717366", "zenodo_15808623"})
    check("v2 (revision 2, before any build): SIU is an intake source of family siu, capped at 35 % like the dock "
          "family", conf["version"].startswith("v2") and conf["intake"]["zenodo_15808623"]["family"] == "siu"
          and conf["rules"]["families"]["siu"]["cap_share"] == 0.35, conf["intake"]["zenodo_15808623"])
    import re
    rx = re.compile(conf["intake"]["zenodo_15808623"].get("group_regex") or "^$")
    names = {"ABUTH_week_10_IMG_1656_frame_0049.jpeg": "ABUTH_week_10_IMG_1656",
             "ABUTH_week_10_IMG_1656_frame_0088.jpeg": "ABUTH_week_10_IMG_1656",
             "AMAPA_week_1_IMG_0007_frame_0230.jpeg": "AMAPA_week_1_IMG_0007",
             "SORVU_week_11_IMG_2210_frame_0001.jpg": "SORVU_week_11_IMG_2210"}
    got = {n: (rx.search(n).group(1) if rx.search(n) else None) for n in names}
    check("revision 3 (before any build): SIU's group_regex takes the video from a frame's file name "
          "(<EPPO>_week_<n>_IMG_<id>_frame_<k>), and the config says revision 3 and why",
          conf["version"] == "v2 revision 3" and got == names and rx.search("IMG_1656.jpeg") is None
          and "revision 3 (before any build)" in conf["decided_by"] and "dependent" in conf["decided_by"]
          and "in flight" in conf["rules"]["holdout"]["why"], (conf["version"], got))
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
          and r["holdout"]["seed_text"] == "inc2/base3/test_v1" and r["holdout"]["max_group"] == 400
          and r["dedupe"]["layout_max_bits"] == 10)
    eg = conf["evaluation_groups"]
    check("the evaluation groups (test v1's NDSU, Latvia, sesame, maize, PAGS8 and the OOD-dev paddy and chilli) guard "
          "at 6 bits, every slug of them excluded by name",
          eg["bits"] == 6 and sorted(eg["groups"]) == ["Latvia", "NDSU", "PAGS8", "chilli", "maize", "paddy", "sesame"]
          and all(x in ex for v in eg["groups"].values() for x in v)
          and "rf_tuf__weed-3434e" not in str(eg) and len(eg["groups"]["Latvia"]) == 7
          and "kg_ravirajsinh45__crop-and-weed-detection-data-with-bounding-boxes" in eg["groups"]["sesame"])
    bad = []
    for mut in (lambda c: c["excluded"].update(src_x="x") or c["sources"].update(src_x={"classes": {"w": "weed"}}),
                lambda c: c["sources"]["rf_kinjj__weed-avnag"]["classes"].update(weed="maybe"),
                lambda c: c["sources"]["rf_kinjj__weed-avnag"].update(family="nofamily"),
                lambda c: c.update(class_id=11),
                lambda c: c["evaluation_groups"]["groups"]["NDSU"].append("rf_kinjj__weed-avnag"),
                lambda c: c.pop("evaluation_groups"),
                lambda c: c["rules"]["dedupe"].pop("layout_max_bits"),
                lambda c: c["intake"]["zenodo_15808623"].update(group_regex="[A-Z]{5}_week_\\d+_frame_"),
                lambda c: c["intake"]["zenodo_15808623"].update(group_regex="(unclosed"),
                lambda c: c["sources"]["rf_kinjj__weed-avnag"].update(group_regex="IMG_\\d+")):
        c = json.loads(B3.CONFIG.read_text())
        mut(c)
        p = TMP / "bad_conf.json"
        p.write_text(json.dumps(c))
        bad.append(refused(B3.load_config, p) is not None)
    check("a slug both included and excluded, a role other than weed/drop, an unknown family, another class id, an "
          "evaluation group listing an included slug, no evaluation groups, no layout_max_bits, or a group_regex that "
          "captures no group or does not compile (intake or registry) refuse the config", all(bad), bad)


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
    # between two sets, and the nearest row of B in both directions
    A, B = H[:150].copy(), H[150:].copy()
    okA, okB = ok[:150], ok[150:]
    for i in range(0, 40, 4):                      # planted: A's variant near B's dHash, and B's variant near A's
        B[i, 0] = A[i, int(rng.integers(1, 8))] ^ (np.uint64(1) << np.uint64(3))
        A[i + 1, 0] = B[i + 1, int(rng.integers(1, 8))] ^ (np.uint64(3) << np.uint64(10))
        okA[i], okB[i + 1] = True, True
    for bits in (3, 6):
        Q, Tt, Dd, V = B3.near_pairs_between(A, okA, B, bits)
        got = {(int(q), int(t)): (int(d), int(v)) for q, t, d, v in zip(Q, Tt, Dd, V)}
        want = {}
        for q in range(len(A)):
            for v in (range(8) if okA[q] else [0]):
                for t in range(len(B)):
                    d = bin(int(A[q, v]) ^ int(B[t, 0])).count("1")
                    if d <= bits and ((q, t) not in want or (d, v) < want[(q, t)]):
                        want[(q, t)] = (d, v)
        dist, arg = B3.cross_nearest(A, okA, B, okB, bits)
        bf = []
        for q in range(len(A)):
            best = bits + 1
            for t in range(len(B)):
                for v in range(8):
                    if v == 0 or okA[q]:
                        best = min(best, bin(int(A[q, v]) ^ int(B[t, 0])).count("1"))
                    if v == 0 or okB[t]:
                        best = min(best, bin(int(B[t, v]) ^ int(A[q, 0])).count("1"))
            bf.append(best if best <= bits else bits + 1)
        check("near_pairs_between and cross_nearest (both directions) at %d bits equal brute force (%d pairs, %d rows "
              "near)" % (bits, len(want), sum(1 for x in bf if x <= bits)),
              got == want and len(want) > 0 and dist.tolist() == bf and sum(1 for x in bf if x <= bits) >= 20
              and all((a_ >= 0) == (d_ <= bits) for a_, d_ in zip(arg.tolist(), dist.tolist())), (len(got), len(want)))


# ------------------------------------------------------------ unit worlds
def h64(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def srow(key, source="s_a", kind="registry", tier=3, sha=None, h=None, var=None, boxes=None, W=640, H=480, stem=None,
         capture=None, group_key=None, family=None, in_pool=False):
    """A synthetic candidate row: random far-apart hashes unless given (h: the dHash; var: the 8 variants)."""
    r = B3._new_row(source, kind, None, "/synthetic/%s.jpg" % key, None, key, family=family, tier=tier)
    if var is None:
        var = [h64("%s/%d" % (key, v)) for v in range(8)]
        if h is not None:
            var[0] = h
    r.update(sha256=sha or hashlib.sha256(key.encode("utf-8")).hexdigest(), W=W, H=H, dhash=int(var[0]),
             variants=[int(x) for x in var], stem=stem, capture=capture, group_key=group_key, in_pool=in_pool,
             boxes=list(boxes if boxes is not None else [(0.3, 0.3, 0.2, 0.2), (0.7, 0.6, 0.2, 0.3)]))
    r["n_weed"] = len(r["boxes"])
    return r


def unit_conf(**holdout):
    conf = json.loads(B3.CONFIG.read_text())
    conf["rules"]["holdout"].update(holdout)
    conf["rules"]["families"] = {}
    return conf


def by_key(rows):
    return {r["key"]: r for r in rows}


def test_rules_units():
    print("the source, image and box rules on their own")
    rules = json.loads(B3.CONFIG.read_text())["rules"]
    good = [(0.5, 0.5, 0.2, 0.2)]

    def rows_with(boxes_list, W=640, H=480):
        return [dict(srow("c%d" % i, boxes=b), W=W, H=H) for i, b in enumerate(boxes_list)]
    base = rows_with([good] * 10)
    big = rows_with([[(0.5, 0.5, 0.95, 0.9)]] * 2 + [good] * 8)
    many = rows_with([[(0.1 + 0.03 * (k % 26), 0.1 + 0.3 * (k // 26), 0.1, 0.1) for k in range(30)]] * 5)
    small = rows_with([[(0.5, 0.5, 0.2, 0.2)] * 7 + [(0.2, 0.2, 0.02, 0.02)] * 3] * 4)
    r0, rb, rm, rs = (B3.source_convention(x, rules["source"]) for x in (base, big, many, small))
    check("the source rule: a clean source passes; 20 %% of boxes over 80 %% of the image fails on that rule alone (%s)"
          % rb["fails"], r0["passes"] and not rb["passes"] and len(rb["fails"]) == 1 and "over 80" in rb["fails"][0],
          (r0, rb))
    check("  a median of 30 boxes per image fails on that rule alone (%s)" % rm["fails"],
          not rm["passes"] and len(rm["fails"]) == 1 and "boxes per image" in rm["fails"][0], rm)
    check("  30 %% of boxes under 16 px with a median side of %.0f px fails on that rule alone (%s)"
          % (rs["median_side_px"], rs["fails"]),
          not rs["passes"] and len(rs["fails"]) == 1 and "under 16" in rs["fails"][0] and rs["median_side_px"] >= 32, rs)
    # the masked-area rule: thin boxes under 8 px (sqrt(w x h)) filling 62 % of the image's rows
    thin = [(0.5, (k + 0.5) / 480.0, 1.0, 0.0001) for k in range(300)]
    r = dict(srow("m1", boxes=[(0.5, 0.5, 0.2, 0.2)] + thin), W=640, H=480)
    B3.image_rules(r, rules)
    r2 = dict(srow("m2", boxes=[(0.5, 0.5, 0.2, 0.2)] + thin[:150]), W=640, H=480)
    B3.image_rules(r2, rules)
    check("a masked area over 50 %% drops the image (%.3f); %.3f keeps it" % (r["masked_area"], r2["masked_area"]),
          r["drop"] == "masked_over_50" and r["masked_area"] > 0.5 and r2["drop"] is None and 0 < r2["masked_area"] < 0.5,
          (r["drop"], r["masked_area"], r2["drop"], r2["masked_area"]))
    lab = TMP / "units" / "two.txt"
    label(lab, [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.2, 0.2)])
    conf = {"sources": {"s_a": {"classes": {"weed": "weed"}}}}
    r = B3._new_row("s_a", B3.REGISTRY_KIND, None, "/x.jpg", lab, "s_a__x")
    r["names"] = ["weed"]
    B3.map_boxes(r, conf)
    r2 = B3._new_row("s_a", B3.REGISTRY_KIND, None, "/x.jpg", lab, "s_a__y")
    r2["names"] = ["weed", "crop"]
    B3.map_boxes(r2, {"sources": {"s_a": {"classes": {"weed": "weed", "crop": "drop"}}}})
    check("a class id beyond the name list refuses the image (unmapped_class); with both names mapped one weed box "
          "stays and the crop box is dropped", r["drop"] == "unmapped_class" and r2["drop"] is None
          and r2["n_weed"] == 1 and r2["n_drop"] == 1, (r["drop"], r2))
    g = [dict(srow("g%d" % i), drop=None) for i in range(4)]
    g[0]["guard"] = {"reasons": [], "refused": False, "refused_by": [], "crosscheck": True}
    g[1]["guard"] = {"reasons": ["near_eval_v2"], "refused": True, "refused_by": ["near_eval_v2"], "crosscheck": False}
    g[2]["guard"] = {"reasons": ["base_copy"], "refused": False, "refused_by": [], "crosscheck": False}
    g[3]["guard"] = {"pending": True, "reasons": [], "refused": False, "refused_by": [], "crosscheck": False}
    B3._guard_drops(g)
    check("the guard's verdicts: an index cross-check hit alone drops the row; a refusal drops it; a base copy drops a "
          "new row; an unjudged row is not judged", [x["drop"] for x in g] == ["guard:crosscheck", "guard:near_eval_v2",
                                                                              "base_copy", None], [x["drop"] for x in g])
    e = [dict(srow("e%d" % i, source="s_l" if i < 3 else "s_ok"), drop=None) for i in range(5)]
    e[0]["embed"], e[1]["embed"], e[3]["embed"] = "near_eval_embed", "unhashable", "near_eval_embed"
    B3.apply_leak_drops(e, {"s_l": {"flagged": True}, "s_ok": {"flagged": False}})
    check("an embedding refusal (near_eval_embed, unhashable) drops its row before D28-v2's source verdict drops the "
          "rest of a flagged source", [x["drop"] for x in e] == ["embed:near_eval_embed", "embed:unhashable",
                                                                "source_leak", "embed:near_eval_embed", None],
          [x["drop"] for x in e])
    w = rules["walltime"]
    lo, hi = B3.walltime(5.0, rules), B3.walltime(20.0, rules)
    check("the walltime guard takes the slower of the loader and the GPU: a 5 ms loader projects the GPU's %.2f h; a "
          "20 ms loader projects %.2f h, over the 6.4 h line" % (lo["projected_base_run_h"], hi["projected_base_run_h"]),
          abs(lo["projected_base_run_h"] - (w["gpu_ms_per_image_epoch"] * 1.2e6 / 3.6e6 + 0.27)) < 1e-3
          and not lo["over_line"] and abs(hi["projected_base_run_h"] - (20.0 * 1.2e6 / 3.6e6 + 0.27)) < 1e-3
          and hi["over_line"], (lo, hi))


def held_of(rows):
    return sorted(r["key"] for r in rows if r["drop"] == "holdout_v1")


def test_select_units():
    import copy
    print("selection on synthetic rows: dedupe, capture groups, the holdout, the quarantine")
    conf = unit_conf(share=0.0, min_images=0, max_images=10)
    H0, H1, H2 = h64("x0"), h64("x1"), h64("x2")
    c_var = [h64("c/%d" % v) for v in range(8)]
    lay = [(0.2, 0.2, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1), (0.8, 0.7, 0.1, 0.1)]
    rows = [srow("a1", h=H0), srow("a2", h=H0 ^ 0b11111),                   # 5 bits, the same boxes
            srow("c1", var=c_var), srow("c2", h=c_var[1] ^ 0b11, boxes=[(0.6, 0.2, 0.3, 0.1)]),  # near only by hflip
            srow("e1", sha="e" * 64, W=320, H=240), srow("e2", sha="e" * 64, W=1280, H=960),     # bytes, other sizes
            srow("g1", kind="base", source="b", sha="9" * 64, tier=0), srow("g2", kind="base", source="b", sha="9" * 64,
                                                                             tier=0),
            srow("g3", sha="9" * 64, tier=1),                                       # an external copy of a base row
            srow("h1", capture="tray|1"), srow("h2", capture="tray|1"),
            srow("i1", group_key="s|20170101-120000"), srow("i2", group_key="s|20170101-120000"),
            srow("j1", source="d_a", family="dock", stem="IMG_7"), srow("j2", source="d_b", family="dock", stem="IMG_7"),
            srow("k1", h=H1, boxes=lay), srow("k2", h=H1 ^ 0xFF, boxes=lay),         # identical layout, 8 bits
            srow("l1", h=H2, boxes=lay), srow("l2", boxes=lay)]                      # identical layout, far apart
    for r in rows:
        if r["kind"] == "base":
            r["in_pool"] = True
    rec = B3.select(rows, conf)
    R = by_key(rows)
    same = lambda a, b: R[a]["group"] == R[b]["group"]  # noqa: E731
    check("dHash 5 bits apart with the same boxes: both kept (dedupe is 3 bits), one capture group (6 bits)",
          R["a1"]["drop"] is None and R["a2"]["drop"] is None and same("a1", "a2"), (R["a1"]["drop"], R["a2"]["drop"]))
    check("near only under a variant (hflip, 2 bits) with other boxes: both kept, one capture group",
          R["c1"]["drop"] is None and R["c2"]["drop"] is None and same("c1", "c2"))
    check("the same bytes at two sizes: the higher resolution is kept", R["e2"]["drop"] is None
          and R["e1"]["drop"] == "duplicate" and R["e1"]["dup_of"] == "e2")
    check("two base_v2 rows with the same bytes are both kept; an external copy of them is dup_of_base",
          R["g1"]["drop"] is None and R["g2"]["drop"] is None and R["g3"]["drop"] == "dup_of_base")
    check("the intake capture group, the file-name session and the export stem across a family each join one group",
          same("h1", "h2") and same("i1", "i2") and same("j1", "j2") and not same("h1", "i1") and not same("a1", "h1"))
    check("an identical layout within 10 bits is a copy (%s); on far-apart photos it is not (%d pair too far)"
          % (R["k2"]["drop"], rec["layout_pairs_too_far"]),
          {R["k1"]["drop"], R["k2"]["drop"]} == {None, "duplicate"} and R["l1"]["drop"] is None
          and R["l2"]["drop"] is None and rec["layout_pairs_too_far"] >= 1)
    names = {r["key"]: r["group"] for r in rows}
    rows2 = [copy.deepcopy(r) for r in rows[::-1]]
    for r in rows2:
        r["drop"] = None
        r.pop("group", None)
        r.pop("dup_of", None)
    B3.select(rows2, conf)
    check("group names come from content: the rows in reverse order get the same names",
          {r["key"]: r["group"] for r in rows2} == names)

    # ---- the holdout's counts
    conf = unit_conf(share=0.15, min_images=2, max_images=10, max_group=11)
    rows = [srow("s%02d" % i, source="src_s") for i in range(40)]
    rows += [srow("t%02d_%d" % (i, k), source="src_t", capture="t|%d" % i) for i in range(34) for k in range(3)]
    rows += [srow("big%02d" % i, source="src_s", capture="one|big") for i in range(12)]
    rec = B3.select(rows, conf)
    held = held_of(rows)
    check("a capture group over max_group (12 rows > 11) is never held out, and summary.json counts its rows by "
          "reason", not [k for k in held if k.startswith("big")]
          and rec["holdout"]["src_s"]["ineligible_rows"] == {"over_max_group": 12}, rec["holdout"]["src_s"])
    held = [k for k in held if not k.startswith("big")]
    hs = [k for k in held if k.startswith("s")]
    ht = [k for k in held if k.startswith("t")]
    check("15 %% of 52 rows (40 singletons, 12 in one group too big to hold): exactly 8 held (round half up, min 2, "
          "max 10); got %d" % len(hs), len(hs) == 8)
    check("groups of 3 under max 10: 9 held, never a group that would pass 10 (got %d)" % len(ht), len(ht) == 9)
    # a source's rows inside other sources' groups count against its own max and its own target
    rows = []
    for o in ("v", "y", "z"):          # src_o owns 30 groups of 2 own rows + 1 row of src_u
        rows += [srow("%s%02d_%d" % (o, i, k), source="src_%s" % o, capture="%s|%d" % (o, i))
                 for i in range(30) for k in range(2)]
        rows += [srow("u%s%02d" % (o, i), source="src_u", capture="%s|%d" % (o, i)) for i in range(30)]
    rows += [srow("w%02d" % i, source="src_w", capture="w|%d" % i) for i in range(10)]
    rows += [srow("x%02d" % i, source="src_x", capture="w|%d" % i) for i in range(10)]
    rows += [srow("x_own%02d" % i, source="src_x") for i in range(18)]
    rec = B3.select(rows, unit_conf(share=0.15, min_images=2, max_images=6, max_group=400))
    held = held_of(rows)
    n = {c: sum(1 for k in held if k.startswith(c)) for c in "uvyzwx"}
    ho = rec["holdout"]
    check("src_u's rows sit in src_v's, src_y's and src_z's groups: src_u stops at max 6 (%s), so src_z's groups are "
          "never held (%d skipped)" % (n, ho["src_z"]["skipped_over_max"]),
          n["u"] == 6 and n["v"] == 6 and n["y"] == 6 and n["z"] == 0 and ho["src_u"]["images"] == 6
          and ho["src_z"]["skipped_over_max"] == 30 and all(v["images"] <= 6 for v in ho.values()), ho)
    check("src_x's target (%d) counts its rows held in src_w's groups (%d): it holds %d of its own, %d in all"
          % (ho["src_x"]["target"], n["w"], n["x"] - n["w"], n["x"]),
          ho["src_x"]["target"] == 4 and n["w"] == 2 and n["x"] == 4 and ho["src_x"]["images"] == 4, ho)

    # ---- the quarantine does not move test v1; nor does a new source; nor the rows' order
    def world():
        out = [srow("p%02d" % i, source="src_p") for i in range(40)]
        out += [srow("q%02d" % i, source="src_q") for i in range(20)]
        out += [srow("q_copy", source="src_q", tier=1, sha="c" * 64), srow("p_copy", source="src_p", sha="c" * 64,
                                                                            in_pool=True)]
        return out
    conf = unit_conf(share=0.15, min_images=2, max_images=10, max_group=400)
    a, b = world(), world()
    ra = B3.select(a, conf, quarantine={"src_q"})
    B3.select(b, conf, quarantine=set())
    A, Bk = by_key(a), by_key(b)
    hq = [k for k in held_of(a) if k.startswith("q")]
    check("lifting src_q's quarantine leaves every source's test rows unchanged (%d held, %d of src_q's)"
          % (len(held_of(a)), len(hq)), held_of(a) == held_of(b) and len(hq) == 3)
    check("quarantined: src_q's other rows are out of arm B, its held rows stay in test v1",
          all(A[k]["drop"] == "quarantined" for k in A if k.startswith("q") and k not in hq and k != "q_copy")
          and all(Bk[k]["drop"] is None for k in Bk if k.startswith("q") and k not in hq) and ra["quarantined"] > 0)
    check("a copy group whose kept row is quarantined keeps its copy from a source that is not (p_copy in B as is; a "
          "duplicate of q_copy when lifted)", A["q_copy"]["drop"] == "quarantined" and A["p_copy"]["drop"] is None
          and Bk["q_copy"]["drop"] is None and Bk["p_copy"]["drop"] == "duplicate" and ra["promoted_copies"] == 1,
          (A["q_copy"]["drop"], A["p_copy"]["drop"], Bk["p_copy"]["drop"]))
    c = world() + [srow("n%02d" % i, source="src_n") for i in range(30)]
    B3.select(c, conf, quarantine={"src_q"})
    check("a new source of its own photos leaves every other source's test rows unchanged",
          [k for k in held_of(c) if not k.startswith("n")] == held_of(a))
    d = world()[::-1]
    B3.select(d, conf, quarantine={"src_q"})
    check("the rows read in reverse order: the same test rows", held_of(d) == held_of(a))

    # ---- an earlier build's test list
    rows = world()
    target = sorted(k for k in by_key(rows) if k.startswith("p") and k not in held_of(a))[:2]
    R = by_key(rows)
    R["p_copy"]["in_pool"] = True
    prior = [{"key": "elsewhere", "original_sha256": R[target[0]]["sha256"]},
             {"key": "elsewhere2", "dhash": R[target[1]]["dhash"] ^ 0b1111, "variants": None},
             {"key": "elsewhere3", "capture_keys": [["capture", "tray|9"]]},
             {"key": "elsewhere4", "original_sha256": "c" * 64}]
    rows.append(srow("p_tray", source="src_p", capture="tray|9", in_pool=True))
    n = B3.mark_prior(rows, prior, 6)
    rec = B3.select(rows, conf, quarantine={"src_q"})
    R = by_key(rows)
    check("rows an earlier test list holds (by bytes, by a dHash 4 bits away, by a capture relation: %d marked) never "
          "train: in an eligible group they are held first (%s), in one touching the pools they are dropped "
          "(prior_test_v1)" % (n, [R[k]["drop"] for k in target]),
          n == 5 and all(R[k]["drop"] == "holdout_v1" for k in target) and R["p_tray"]["drop"] == "prior_test_v1"
          and R["q_copy"]["drop"] == "prior_test_v1" and R["p_copy"]["drop"] == "duplicate"
          and rec["prior_test_dropped"] == 2 and rec["promoted_copies"] == 0,
          {k: R[k]["drop"] for k in target + ["p_tray", "p_copy", "q_copy"]})


def test_eval_guard():
    print("the evaluation-group guard")
    import numpy as np
    g = np.asarray([[h64("ev%d/%d" % (i, v)) for v in range(8)] for i in range(5)], dtype=np.uint64)
    rows = [srow("r_near_var", var=[int(g[0, 3]) ^ 0b11111] + [h64("rv/%d" % v) for v in range(1, 8)]),
            srow("r_near_mine", var=[h64("rm/0")] + [int(g[1, 0]) ^ 0b111111] + [h64("rm/%d" % v) for v in range(2, 8)]),
            srow("r_far7", h=int(g[2, 0]) ^ 0b1111111),
            srow("r_base", kind="base", source="b", h=int(g[3, 0]) ^ 0b1),
            dict(srow("r_nohash"), dhash=None, variants=None)]
    rec = B3.eval_guard(rows, {"G": (g, np.ones(len(g), dtype=bool))}, 6)
    R = by_key(rows)
    check("a row within 6 bits of an evaluation image under its variant, or the image's variant within 6 bits of the "
          "row, is dropped (near_eval_group); 7 bits away it stays; a base_v2 row is counted, never dropped",
          R["r_near_var"]["drop"] == "near_eval_group" and R["r_near_mine"]["drop"] == "near_eval_group"
          and R["r_far7"]["drop"] is None and R["r_base"]["drop"] is None and R["r_base"]["eval_group"] == {"G": 1}
          and rec["groups"]["G"]["by_source"] == {"base_v2": {"within": 1, "within_3": 1},
                                                  "s_a": {"within": 2, "within_3": 0}}
          and rec["dropped"] == 2 and rec["unjudged"] == 1, rec)


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
    check("src_big: crop-only -> no weed box, unlabelled -> no_label, a box over 90 %%, a short side, an hflip copy in "
          "an evaluation group -> near_eval_group; kept rows hold weed boxes only (%s)" % big["dropped"],
          big["dropped"].get("no_weed_box") == 1 and big["dropped"].get("no_label") == 1
          and big["dropped"].get("box_over_90") == 1 and big["dropped"].get("short_side") == 1
          and big["dropped"].get("near_eval_group") == 1 and big["eval_groups"] == {"G1": 1}
          and big["in_B"] + big["holdout_v1"] == 5)
    eg = summ["evaluation_groups"]
    check("the evaluation groups in summary.json: G1's images hashed (%s), the slug not in the registry named, src_big's "
          "row within 6 bits counted" % eg["groups"]["G1"]["slugs"],
          eg["groups"]["G1"]["hashes"] == 2 and eg["slugs_not_read"] == {"src_unregistered": "not in the registry"}
          and eg["groups"]["G1"]["by_source"] == {"src_big": {"within": 1, "within_3": 1}} and eg["dropped"] == 1
          and eg["bits"] == 6, eg)
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
    q = ps["src_quar"]
    check("src_quar is quarantined by the stream: none of its rows in arm B; its held rows (the holdout ignores the "
          "quarantine: %d) stay in test v1, marked" % q["holdout_v1"],
          q["in_B"] == 0 and q["holdout_v1"] == 2 and q["dropped"] == {"quarantined": 3} and q["quarantined"]
          and summ["holdout_v1"]["per_source"]["src_quar"]["quarantined_source"] is True
          and summ["quarantine"] == ["src_quar"], q)
    check("src_badnames: a class name the config does not list -> the source is not read (summary.json names why)",
          "src_badnames" not in ps and "mystery" in summ["sources_not_read"].get("src_badnames", ""),
          summ.get("sources_not_read"))
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
    check("dock: the byte copy, the hflip copy with flipped boxes and the identical layout %d bits away are duplicates "
          "of the tier-2 source (%s); the hflip copy with other boxes, and the identical layout on another photo, are "
          "kept" % (f["x05_bits"], dropped_b),
          dropped_b.get("duplicate") == 3 and db["in_B"] + db["holdout_v1"] + dropped_b.get("family_cap", 0) == 7
          and da.get("dropped", {}).get("duplicate", 0) == 0 and 4 <= f["x05_bits"] <= 10)
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
    ins = summ["inputs"]
    shas = [summ["config"]["sha256"], ins["registry"]["sha256"], ins["base_v2"]["sha256"],
            ins["base_v2"]["provenance_sha256"], ins["lock_v2"]["sha256"], ins["stream"]["head_sha256"],
            summ["leak_thresholds"]["sha256"]] + [v["manifest_sha256"] for v in ins["intake"].values()]
    check("summary.json: complete, every input's sha256 (%d), no key named after a non-dev split" % len(shas),
          summ["status"] == "complete" and summary_ok(summ) and len(shas) == 8
          and all(isinstance(x, str) and len(x) == 64 for x in shas), shas)
    grp_of = {p["key"]: p["group"] for p in prov.values()}
    for x in (json.loads(ln) for ln in (root / B3.DROPPED).read_text().splitlines()):
        if x.get("group"):
            grp_of[x["key"]] = x["group"]
    k_d02 = next(k for k in grp_of if k.startswith("src_dock_a__") and "_d02_jpg" in k)
    k_x02 = next(k for k in grp_of if k.startswith("src_dock_b__") and "_x02_jpg" in k)
    check("an hflip copy whose boxes do not follow the flip (src_dock_b x02) joins its original's capture group "
          "(src_dock_a d02) through the hflip variant", grp_of[k_d02] == grp_of[k_x02], (k_d02, k_x02))
    trays = [grp_of.get("int_src__%02d" % i) for i in (2, 3, 4, 5)]
    check("intake rows of one capture group (tray1: 02, 03; tray2: 04, 05) share a capture group; two trays do not",
          None not in trays and trays[0] == trays[1] and trays[2] == trays[3] and trays[0] != trays[2], trays)
    ses = {k: g for k, g in grp_of.items() if k.startswith("src_session__")}
    s1 = {g for k, g in ses.items() if "_s1_" in k}
    s2 = {g for k, g in ses.items() if "_s2_" in k}
    check("src_session's file-name session (group_regex ^(s\\d)_) joins s1_* into one group and s2_* into another",
          len(ses) == 6 and len(s1) == 1 and len(s2) == 1 and s1 != s2, ses)
    hrow = next(json.loads(ln) for f in sorted((root / B3.HOLDOUT_DIR).glob("*.jsonl"))
                for ln in f.read_text().splitlines())
    check("a test list row carries its capture relations and whether its source is quarantined",
          "capture_keys" in hrow and "quarantined_source" in hrow and hrow["original_sha256"], hrow.keys())
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
    summ["_held_keys"] = sorted(p["key"] for p in prov.values() if p["holdout_v1"])
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
            # the embedding detector finds one copy: src_poly's p01 (used as it is: no PNG)
            return {r["key"]: [{"cos": 0.99, "bits": 12, "split": "dev", "eval_key": "x"}] for r in rows
                    if r["image"].endswith("/src_poly/images/p01.jpg")}, set()

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
        out["poly_%s" % cos] = s["per_source"]["src_poly"]
    lo, hi = out[0.3], out[0.97]
    po = out["poly_0.3"]
    check("the embedding detector's copy is dropped (embed:near_eval_embed) and, one copy in 5 images, D28-v2's "
          "embedding rule takes the rest of its source (%s)" % po["dropped"],
          po["dropped"].get("embed:near_eval_embed") == 1 and po["leak"]["flagged"]
          and po["dropped"].get("source_leak") == 4 and po["in_B"] + po["holdout_v1"] == 0, po)
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
    check("as is: src_quar's rows stay out of arm B; lifted: its rows are in (%d -> %d)"
          % (asis["arms"]["B"]["images"], lift["arms"]["B"]["images"]),
          asis["per_source"]["src_quar"]["in_B"] == 0 and lift["lifted"] == ["src_quar"]
          and lift["per_source"]["src_quar"]["in_B"] + lift["per_source"]["src_quar"]["holdout_v1"] == 5
          and lift["arms"]["B"]["images"] == asis["arms"]["B"]["images"] + 3)
    check("lifting src_quar leaves test v1 unchanged, every source's held rows included (%d images, keys %s)"
          % (asis["holdout_v1"]["images"], asis["holdout_v1"]["sha256_of_keys"][:12]),
          rep["holdout_same_in_every_scenario"] is True and asis["holdout_v1"] == lift["holdout_v1"]
          and all(asis["per_source"][s_]["holdout_v1"] == lift["per_source"][s_]["holdout_v1"]
                  for s_ in asis["per_source"]) and asis["per_source"]["src_quar"]["holdout_v1"] == 2)
    check("count hashes and judges every candidate row, the quarantined ones included (none left unjudged)",
          asis["pending"]["guard_unjudged_rows"] == 0 and lift["pending"]["guard_unjudged_rows"] == 0
          and sum(lift["per_source"]["src_quar"]["guard"].values()) == 0
          and rep["evaluation_groups"]["unjudged"] == 0)
    same = {s_: (v, built["selection"]["holdout"].get(s_)) for s_, v in asis["selection"]["holdout"].items()
            if s_ != "src_leak"}
    ck = {k for k in asis["holdout_v1"]["keys"] if not k.startswith("src_leak__")}
    bk = {k for k in built["_held_keys"] if not k.startswith("src_leak__")}
    check("count's holdout is the build's for every source but src_leak (count cannot weigh its dHash hit, so it "
          "does not drop the source): the same records and the same %d test rows" % len(bk),
          all(a_ == b_ for a_, b_ in same.values()) and len(same) >= 8 and ck == bk and len(bk) >= 15,
          ({k: v for k, v in same.items() if v[0] != v[1]}, sorted(ck ^ bk)))
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
    big05 = DS / "src_big" / "train" / "images" / "big_05.jpg"
    pool.append(dict(pool[0], key="elsewhere__big05", image=str(big05), sha256=C.sha256_file(big05)))
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
          not ({r["key"] for r in rows[2:]} & held) and summ["inputs"]["stream"]["pool_rows"] == 5
          and it["in_B"] >= 4, (sorted(held), it))
    big = summ["per_source"]["src_big"]
    check("a registry row whose bytes a pool row holds under another key is never held out either (src_big: %d of %d "
          "held; big_05 in arm B)" % (big["holdout_v1"], big["in_B"] + big["holdout_v1"]),
          "src_big__train_images_big_05" not in held and big["holdout_v1"] == 4 and big["in_B"] == 1, big)


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
    (C.REPO / "yolo11m.pt").write_bytes(os.urandom(4096))
    shutil.rmtree(C.INC_DIR / "e1t_w", ignore_errors=True)
    e = refused(B.build_definition, "e1t_w", manifest=s["arms"]["B"]["manifest"], arm="m640", testing=True)
    other = TMP / "copy_of_b.jsonl"
    shutil.copyfile(s["arms"]["B"]["manifest"], other)
    e2 = refused(B.build_definition, "e1t_w", manifest=other, arm="m640", role="baseline", testing=True)
    check("inc2.baseline refuses arm B of an over_walltime summary, and a copy of it elsewhere, instead of training "
          "the 100-epoch cold table (%s)" % e, isinstance(e, B.BaselineError) and "base v3 manifest" in str(e)
          and isinstance(e2, B.BaselineError) and "records it as arm B (status over_walltime)" in str(e2)
          and not (C.INC_DIR / "e1t_w" / "exp.json").exists(), (e, e2))


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
    inside = B3.out_dir() / "extra.jsonl"
    C.write_manifest(inside, C.read_manifest(mb)[:-2])
    for exp in ("e1t_y", "e1t_z", "e1t_u"):
        shutil.rmtree(C.INC_DIR / exp, ignore_errors=True)
    e1 = refused(B.build_definition, "e1t_y", manifest=inside, arm="m640", testing=True)
    e2 = refused(B.build_definition, "e1t_u", union=[str(ma), str(other)], arm="m640", testing=True)
    e3 = refused(B.build_definition, "e1t_z", manifest=ma, arm="n640", role="baseline", testing=True)
    sp = B3.out_dir() / B3.SUMMARY
    keep = sp.read_bytes()
    sp.write_text(json.dumps(dict(json.loads(keep), status="edited")))
    copy_b = TMP / "copy_b2.jsonl"
    shutil.copyfile(mb, copy_b)
    e4 = refused(B.build_definition, "e1t_z", manifest=copy_b, arm="m640", testing=True)
    sp.write_bytes(keep)
    inside.unlink()
    check("inc2.baseline refuses a manifest under splits/v3 that no summary records, a union with an E1 manifest, an "
          "E1 manifest on another arm, and an E1 manifest once summary.json is no longer complete",
          all(isinstance(x, B.BaselineError) for x in (e1, e2, e3, e4)) and "lies under" in str(e1)
          and "never in a union" in str(e2) and "status edited" in str(e4), (e1, e2, e3, e4))
    return summ


def test_prior_lists(Wd, reg, cp):
    print("earlier test lists: never trained in a later build, held again first; a partial build blocks a rebuild")
    clear_v3()
    splits = B3.out_dir().parent

    def keys_of(root):
        held = {json.loads(ln)["key"] for f in (root / B3.HOLDOUT_DIR).glob("*.jsonl")
                for ln in f.read_text().splitlines()}
        return held, {r["key"] for r in C.read_manifest(root / ("%s.jsonl" % B3.ARM_B))}
    make_stream(SID, [QUAR])
    with holds({"int_src__00": []}):
        B3.build(SID, conf_path=cp, registry=reg, testing=True, procs=2, loader=LOADER)
    held1, b1 = keys_of(B3.out_dir())
    os.rename(B3.out_dir(), splits / "v3_prev1")
    make_stream(SID, [QUAR, {"event": "unquarantine", "source": "src_quar", "by": "human:x"}])
    with holds({"int_src__00": []}):
        s2 = B3.build(SID, conf_path=cp, registry=reg, testing=True, procs=2, loader=LOADER)
    held2, b2 = keys_of(B3.out_dir())
    check("rebuilt with src_quar's quarantine lifted, the first build's test lists read as never-train (%s): the same "
          "test v1 (%d), none of it in arm B, arm B gains src_quar's other 3 rows"
          % ([f["file"].split("/splits/")[-1] for f in s2["prior_test"]["files"]], len(held2)),
          held2 == held1 and not (held1 & b2) and b2 - b1 == {k for k in b2 if k.startswith("src_quar__")}
          and len(b2 - b1) == 3 and s2["prior_test"]["rows"] == len(held1) and s2["prior_test"]["marked"] >= len(held1),
          (sorted(held1 ^ held2), s2["prior_test"]))
    os.rename(B3.out_dir(), splits / "v3_prev2")
    with holds({"int_src__00": [], "int_src__01": []}):
        s3 = B3.build(SID, conf_path=cp, registry=reg, testing=True, procs=2, loader=LOADER)
    held3, b3 = keys_of(B3.out_dir())
    check("rebuilt once more with a new row (int_src__01's hold released, joining tray0): every earlier test row is "
          "held again or never trained (%d earlier lists)" % len(s3["prior_test"]["files"]),
          held1 <= held3 and not (held1 & b3) and "int_src__01" in (held3 | b3)
          and len(s3["prior_test"]["files"]) == 2 * len(s2["holdout_v1"]["per_source"]),
          (sorted(held1 - held3), s3["selection"]["holdout"].get("int_src"),
           [json.loads(x) for x in (B3.out_dir() / B3.DROPPED).read_text().splitlines() if "int_src" in x]))
    os.rename(B3.out_dir(), splits / "v3_prev3")
    (B3.out_dir() / B3.HOLDOUT_DIR).mkdir(parents=True)
    e = refused(B3.build, SID, conf_path=cp, registry=reg, testing=True, procs=2, loader=LOADER)
    check("splits/v3/test_v1 without a summary (a build that did not finish): a rebuild refuses until it is moved "
          "aside", e is not None and "did not finish" in str(e), e)
    clear_v3()
    for d in splits.glob("v3_prev*"):
        shutil.rmtree(d)


def test_intake_family(f):
    print("an intake source's family: its rows carry it (the family cap applies), an unknown one refuses")
    conf = json.loads(B3.CONFIG.read_text())
    conf["intake"] = {"int_src": {"tier": 1, "classes": "all_weed", "family": "siu"}}
    rows, _rec = B3.intake_rows(conf, holds_view={})
    check("every row of an intake source listed with family siu is of family siu",
          rows and all(r["family"] == "siu" for r in rows), [r["family"] for r in rows])
    conf["intake"]["int_src"]["family"] = "nope"
    tmp = TMP / "conf_family.json"
    tmp.write_text(json.dumps(conf))
    check("  an intake source whose family has no rule refuses the config", refused(B3.load_config, tmp))


def _cut(sid, inc, seg, rows):
    """A cut line of inc2.stream's ledger for increment `inc` of segment `seg`; its rows sidecar is written
    outside the stream's directory, which make_stream recreates."""
    p = S.StreamPaths(sid)
    side = TMP / "sidecars" / sid / ("%s.rows.jsonl" % inc)
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text("".join(json.dumps({"key": r["key"], "sha256": r["sha256"], "source": r["source"],
                                        "hashes": {}}, sort_keys=True) + "\n" for r in rows))
    return {"event": "cut", "increment": inc, "segment": seg, "step": 1,
            "manifest": {"path": str(p.inc_manifest(inc)), "sha256": "0" * 64, "n_images": len(rows)},
            "rows": {"path": str(side), "sha256": C.sha256_file(side)}, "meta": {"path": str(p.inc_meta(inc))},
            "sources": sorted({r["source"] for r in rows}), "by": "platform"}


def test_in_flight(Wd, reg, cp, f):
    print("the rows of an increment in flight count as pool rows for the holdout")
    check("released statuses: data, stale, withdrawn, every return disposition and a bisect's verdicts; in_segment, "
          "suspect, accepted and an unknown status are not released",
          all(B3.released_status(x) for x in ("data", "stale", "withdrawn", "species", "flips", "recipe", "hold",
                                              "truth_helps", "bisect_helps", "bisect_hurts", "bisect_neutral"))
          and not any(B3.released_status(x) for x in ("in_segment", "suspect", "accepted", "brand_new", None)))
    rows = f["intake_rows"]
    big = [{"key": "src_big__train_images_big_%02d" % i, "sha256": "f%063d" % i, "source": "src_big"} for i in range(2)]
    # inc0001 cut (in_segment); inc0002 cut, then its build was killed (withdrawn); inc0003 committed as 'data';
    # inc0004 accepted then rolled back (suspect); inc0005 of a status this builder does not know
    events = [_cut(SID, "inc0001", 2, rows[2:]), _cut(SID, "inc0002", 3, big[:1]),
              {"event": "withdraw", "segment": 3, "orphan_increments": ["inc0002"], "reason": "test", "by": "platform"},
              _cut(SID, "inc0003", 1, big[1:]), _cut(SID, "inc0004", 1, rows[1:2]),
              {"event": "build", "segment": 1, "exp": "%s_s001" % SID, "base_pool": "P_0",
               "increments": ["inc0003", "inc0004"], "recipes": ["r0"], "truth": False, "by": "platform"},
              {"event": "commit", "segment": 1, "dispositions": {"inc0003": "data", "inc0004": "accepted"},
               "counted": {}, "by": "platform"},
              {"event": "rollback", "to": "P_0", "from": "P_1", "suspect": ["inc0004"], "by": "platform"},
              QUAR]
    make_stream(SID, events)
    q, rec, (keys, shas) = B3.quarantined_sources(SID)
    fl = rec["in_flight"]
    check("quarantined_sources: in flight are inc0001 (in_segment, 4 rows) and inc0004 (suspect, 1 row); the "
          "withdrawn inc0002 and the 'data' inc0003 are not; the record counts 5 rows",
          sorted(fl["increments"]) == ["inc0001", "inc0004"] and fl["rows"] == 5
          and fl["increments"]["inc0001"] == {"status": "in_segment", "segment": 2, "rows": 4}
          and fl["increments"]["inc0004"]["status"] == "suspect"
          and {r["key"] for r in rows[1:]} <= keys and not ({b["key"] for b in big} & keys)
          and rec["pool_rows"] == 0 and list(q) == ["src_quar"], (fl, rec["pool_rows"]))
    orig = S.Fold._ev_commit

    def odd(self, e):
        orig(self, e)
        self.increments["inc0004"]["status"] = "brand_new"
    S.Fold._ev_commit = odd
    try:
        _q, rec2, (keys2, _s) = B3.quarantined_sources(SID)
    finally:
        S.Fold._ev_commit = orig
    check("  an increment of a status the builder does not know counts as in flight (fail closed)",
          "inc0004" in rec2["in_flight"]["increments"] and rows[1]["key"] in keys2, rec2["in_flight"])
    side = TMP / "sidecars" / SID / "inc0001.rows.jsonl"
    side.write_text(side.read_text() + "\n")
    check("  an increment's rows sidecar that no longer hashes as its cut line says refuses the build (the rows "
          "it trains on are unknown)", refused(B3.quarantined_sources, SID) is not None)
    # end to end: the rule would hold every group, yet no row of an increment in flight is held out
    clear_v3()
    make_stream(SID, [_cut(SID, "inc0001", 2, rows[2:]), QUAR])
    conf = json.loads(cp.read_text())
    conf["rules"]["holdout"].update(min_images=6, max_images=6, share=1.0)
    allp = TMP / "base3_flight.json"
    allp.write_text(json.dumps(conf))
    with holds({"int_src__00": []}):
        summ = B3.build(SID, conf_path=allp, registry=reg, testing=True, procs=2, loader=LOADER)
    held = {json.loads(x)["key"] for x in (B3.out_dir() / B3.PROVENANCE).read_text().splitlines()
            if json.loads(x)["holdout_v1"]}
    it = summ["per_source"]["int_src"]
    stv = summ["inputs"]["stream"]
    check("an increment cut before the build (in_segment): none of its 4 intake rows is held out to test v1, though "
          "the rule would take every group (int_src: %d held of %d kept); summary.json records the in-flight rows "
          "(%s) and the candidate rows marked (%s)" % (it["holdout_v1"], it["in_B"] + it["holdout_v1"],
                                                      stv.get("in_flight", {}).get("rows"),
                                                      stv.get("pool_marked_rows")),
          not ({r["key"] for r in rows[2:]} & held) and it["in_B"] >= 4
          and stv["in_flight"]["rows"] == 4 and stv["in_flight"]["increments"]["inc0001"]["status"] == "in_segment"
          and stv["pool_rows"] == 0 and stv["pool_marked_rows"] >= 4, (sorted(held), it, stv.get("in_flight")))
    clear_v3()
    make_stream(SID, [QUAR])


def _vid_rows(src, batch, frames, rel_dir="weed_dataset/Dataset/images/test"):
    """An intake batch named as zenodo_15808623's i0004 is (key <src>__weed_dataset__Dataset__images__<split>__<name>,
    the intake's capture group and session the image's own path); frames: [(name, split or None)]."""
    idir = C.INC_DIR / "intake" / batch
    (idir / "images").mkdir(parents=True, exist_ok=True)
    out = []
    for name, split in frames:
        rel = "%s/%s" % (rel_dir if split is None else rel_dir.rsplit("/", 1)[0] + "/" + split, name)
        key = "%s__%s" % (src, rel[:-len(".jpeg")].replace("/", "__"))
        img = idir / "images" / (key + ".jpeg")
        out.append({"key": key, "image": str(img), "rel": rel, "sha256": hashlib.sha256(key.encode()).hexdigest(),
                    "label": str(idir / "labels" / (key + ".txt")), "label_sha256": "0" * 64, "source": src,
                    "capture_group": rel, "capture_group_basis": "image", "session": rel, "width": 720,
                    "height": 960, "dhash": h64(key), "holds": [], "licence": "cc-by-nc-sa-4.0",
                    "research_only": True})
    (idir / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    return idir, out


def test_intake_video_groups():
    print("an intake source's group_regex (SIU, revision 3): a video's frames are one capture group")
    shipped, _sha = B3.load_config()
    zrx = shipped["intake"]["zenodo_15808623"]["group_regex"]
    vids = ["ABUTH_week_10_IMG_1656", "AMAPA_week_3_IMG_0420", "SIDSP_week_7_IMG_0999", "ECHCG_week_2_IMG_1200"]
    frames = [("%s_frame_%04d.jpeg" % (v, k), None) for v in vids for k in (1, 49, 88, 120)]
    frames.append(("%s_frame_%04d.jpeg" % (vids[0], 200), "train"))     # the same video under the source's train/
    frames.append(("IMG_0001.jpeg", None))                               # a name the pattern does not match
    idir, man = _vid_rows("vid_src", "i0009_vid_src", frames)
    try:
        conf = unit_conf(share=0.25, min_images=1, max_images=6, max_group=400)
        conf["intake"] = {"vid_src": {"tier": 1, "classes": "all_weed", "family": "siu", "group_regex": zrx}}
        rows, _rec = B3.intake_rows(conf, holds_view={})
        R = by_key(rows)
        gk = {r["rel"].rsplit("/", 1)[1]: R[r["key"]]["group_key"] for r in man}
        check("rows named like the real intake (the capture group the single image) get the video as their session "
              "(group_key '<source>|<video>'), from the original file name, whatever folder holds it; an unmatched "
              "name gets none",
              all(gk["%s_frame_%04d.jpeg" % (v, k)] == "vid_src|%s" % v for v in vids for k in (1, 49, 88, 120))
              and gk["%s_frame_0200.jpeg" % vids[0]] == "vid_src|%s" % vids[0] and gk["IMG_0001.jpeg"] is None
              and all(R[r["key"]]["capture"] == "vid_src|%s" % r["rel"] for r in man), gk)

        def synth(rws):
            out = []
            for r in rws:
                x = srow(r["key"], source=r["source"], kind=r["kind"], tier=r["tier"], family=r["family"],
                         capture=r["capture"], group_key=r["group_key"])
                out.append(x)
            return out
        sel = synth(rows)
        B3.select(sel, conf)
        S_ = by_key(sel)
        vid_of = {r["key"]: (r["group_key"] or r["key"]) for r in rows}
        groups = {}
        for k, x in S_.items():
            groups.setdefault(vid_of[k], set()).add(x["group"])
        held = held_of(sel)
        split = [v for v in groups if v.startswith("vid_src|") and 0 < sum(1 for k in held if vid_of[k] == v)
                 < sum(1 for k in S_ if vid_of[k] == v)]
        check("each video is one capture group (%d videos), and the holdout takes whole videos: %d frames held, no "
              "video split between test v1 and arm B" % (len(vids), len(held)),
              all(len(g) == 1 for g in groups.values()) and len({next(iter(g)) for v, g in groups.items()
                                                                  if v.startswith("vid_src|")}) == len(vids)
              and held and not split, (sorted(held), split))
        conf0 = json.loads(json.dumps(conf))
        conf0["intake"]["vid_src"].pop("group_regex")
        rows0, _r = B3.intake_rows(conf0, holds_view={})
        sel0 = synth(rows0)
        B3.select(sel0, conf0)
        held0 = held_of(sel0)
        split0 = [v for v in vids if 0 < sum(1 for k in held0 if vid_of[k] == "vid_src|" + v)
                  < sum(1 for k in vid_of if vid_of[k] == "vid_src|" + v)]
        check("  without the group_regex (revision 2) the intake capture group is the single frame, and frames of "
              "one video land in both test v1 and arm B (%d videos split)" % len(split0), len(split0) >= 1,
              sorted(held0))
        # the siu family cap drops whole videos
        conf2 = unit_conf(share=0.0, min_images=0, max_images=0, max_group=400)
        conf2["rules"]["families"] = {"siu": {"cap_share": 0.35}}
        conf2["intake"] = conf["intake"]
        sel2 = synth(rows) + [srow("other%02d" % i, source="src_other") for i in range(12)]
        rec2 = B3.select(sel2, conf2)
        S2_ = by_key(sel2)
        part = [v for v in vids if len({S2_[k]["drop"] for k in vid_of if vid_of[k] == "vid_src|" + v}) > 1]
        cap = rec2["family_caps"]["siu"]
        check("the siu cap (35 %%: at most %d of %d frames beside 12 other rows) drops whole videos (%d frames "
              "dropped), never part of one" % (cap["cap"], cap["before"], cap["dropped"]),
              cap["cap"] == 6 and cap["after"] <= 6 and cap["dropped"] >= 4 and not part, (cap, part))
    finally:
        shutil.rmtree(str(idir), ignore_errors=True)


def main():
    t0 = time.time()
    Wd = W.build_world()
    write_provenance(Wd)
    reg, cp, f = build_registry_world(Wd)
    test_config()
    test_geometry()
    test_pairs()
    test_rules_units()
    test_select_units()
    test_eval_guard()
    built = test_build(Wd, reg, cp, f)
    test_leak_weighed(Wd, reg, cp, f)
    test_count_and_quarantine(Wd, reg, cp, f, built)
    test_pool_rows(Wd, reg, cp, f)
    test_in_flight(Wd, reg, cp, f)
    test_walltime(Wd, reg, cp)
    test_baseline_budget(Wd, reg, cp)
    test_prior_lists(Wd, reg, cp)
    test_intake_family(f)
    test_intake_video_groups()
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    if FAILURES:
        for x in FAILURES:
            print("  FAILED: %s" % x)
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
