#!/usr/bin/env python3
"""DS1's data builds (inc2/ds1_data.py, lever L30; docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-29): DS1", DS.0
items 2-6 and 14-19, DS.5, DS.9.5).

A synthetic world in a temporary INC_DIR: base B's provenance (base_v2 rows with six train_core and two NDSU rows,
external Field, Turf and Indoor rows, SIU frames of the videos the new frames come from), D1's B', base A; new
candidates of five sources (a field source with capture-group pairs, a re-upload carrying cwd12 capture names, a
source whose OtherPlant boxes are crops, a source holding one copy of a cwd12 dev image, SIU's video frames of target,
non-target and crop species); the exams' stored hashes (dev, ood23 with a planted rot90 copy, a test v1 list, H10d);
a stub DINOv2 scanner that finds one ood23 copy dHash misses; the decided sheets. Image files are real (small
PNGs), hashes are a stub hasher's (planted distances).

Units: eligibility (each refusal and hold that keeps a row out; a licence hold and a species-level non-admission
that do not); the capture-name scan; names to roles (weed class 12, crops dropped, unresolved held); base3's box and
image rules; thin(10) on a synthetic video seeded with S0's frames and its seeded order; the family cap dropping
whole videos in seeded order counted before repeats; the cwd12 floor's repeat count and train_core's image-epochs
equal to S0's; S1c-R's size, composition and kept train_core rows; KD, KT and KF's quotas and equal N_K; content-named
capture groups; the draws (5 % / 10 % / 30 %, a group over 500 never, a new source leaving the others' draws
unchanged, an enlargement keeping the order); both guard directions on planted copies (dHash rot90, DINOv2 only,
an H10d record, an MD row's capture relation); the EXIF refusal after the guard; write-once.

The world end to end (screen, pools, md_s0 drawn, its power, frozen, draw with MT-W v2 frozen after the check,
manifests): MD-S0 frozen before B'o; B'o without MD-S0's rows, the NDSU rows and the guard's hits (a base_v2 row's
dev hit kept and listed); S0 in S1c in S1u, row bytes unchanged; the repeats; SIU capped in S1c, not in S1u; test v1
companions and SIU never in MD or MT-W v2; every exam row in DS1's never-train list, which the stream cutter's
queue reader marks (inc2.step1_stream.test_v1_rows); plan_state's view for the platform.

Run:  python3 tests/test_inc2_ds1_data.py
"""
import collections
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import traceback

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_data_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import base3 as B3  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_data as DD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402

FAILURES = []
SIU = DD.SIU
DD.PROJECT_LOADER = False


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except D.Ds1Refused as e:
        return e
    return None


# ------------------------------------------------------------------ hashes with planted distances
def h64(seed):
    return int(np.random.default_rng(seed).integers(0, 2 ** 63)) | (1 << 63)


def flip(h, bits, start=0):
    for b in range(start, start + bits):
        h ^= 1 << b
    return h


def variants(h, seed):
    return [int(h)] + [h64(seed * 100 + i) for i in range(1, 8)]


HASHES = {}        # image path -> (dhash, variants)


def make_img(path, seed, size=(360, 360)):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return str(path)


def make_label(path, boxes):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes))
    return str(path)


def hasher(path):
    h = HASHES.get(str(path))
    if h is None:
        return None, None
    return h[0], {k: v for k, v in zip(B3.VARIANTS, h[1])}


class StubScanner(object):
    """A DINOv2 detector stub: {image path: (exam, exam key)} it calls copies."""

    def __init__(self, flagged):
        self.flagged = dict(flagged)

    def scan(self, rows, desc_path=None, procs=1):
        out = {}
        for r in rows:
            hit = self.flagged.get(str(r["image"]))
            if hit:
                out[r["key"]] = [{"eval_split": hit[0], "eval_key": hit[1], "cos": 0.97}]
        return out, set()


# ------------------------------------------------------------------ units
def t_eligibility():
    print("eligibility (DS.5.1 item 1, DS.0 item 2)")
    ctx = {"base_keys": {"bk"}, "base_shas": {"bsha"}, "eval_sources": {"evg"}, "quarantined": {"q28"},
           "never": {"nf"}, "consumed": {"used"}}
    base = {"key": "k", "source": "s", "sha256": "x", "holds": [], "refused": None, "admission": "whole"}
    cases = [("role C (a D28 re-upload)", dict(base, source=DD.ROLE_C[2]), "role_c"),
             ("2SeasonWeedDet8 (never fetched)", dict(base, source="zenodo_10762138"), "role_c"),
             ("a never-train source", dict(base, source="nf"), "never_train_source"),
             ("a D28-quarantined source", dict(base, source="q28"), "quarantined_d28"),
             ("an evaluation group by name", dict(base, source="evg"), "evaluation_group"),
             ("refused for a copy", dict(base, refused="near_eval_embed"), "refused:near_eval_embed"),
             ("refused for EXIF at intake", dict(base, refused="exif_orientation"), "refused:exif_orientation"),
             ("held funnel_F9", dict(base, holds=["licence", "funnel_F9"]), "held:funnel_F9"),
             ("held join_conflict", dict(base, holds=["join_conflict"]), "held:join_conflict"),
             ("held h6_scan", dict(base, holds=["h6_scan"]), "held:h6_scan"),
             ("a test v1 row", dict(base, test_v1="bytes"), "test_v1"),
             ("an overlap refusal at admission", dict(base, admission="refused_overlap"), "admission:refused_overlap"),
             ("a row base B holds (key)", dict(base, key="bk"), "in_base_b"),
             ("a row base B holds (bytes)", dict(base, sha256="bsha"), "in_base_b"),
             ("a row a stream consumed", dict(base, key="used"), "stream_consumed")]
    for name, row, want in cases:
        check("out: %s -> %s" % (name, want), DD.eligibility(row, ctx) == want, DD.eligibility(row, ctx))
    check("in: a licence hold (research-only until a person decides)", DD.eligibility(dict(base, holds=["licence"]),
                                                                                      ctx) is None)
    check("in: a row only not admitted at species level", DD.eligibility(dict(base, admission="not_admitted"), ctx)
          is None)
    hits = DD.capture_name_hits(["IphoneSE_YL_Morningglory_039_JPG.rf.0a1b", "20210628_iPhone11Pro_BT_63.jpg",
                                 "NIKOND3300_YL_carpetweed_12.png", "field_plot_0007.jpg", "IMG_4521.JPG"])
    check("the capture-name scan: cwd12 and 3SeasonWeedDet10 stems found, ordinary names not",
          [h[0][:10] for h in hits] == ["IphoneSE_Y", "20210628_i", "NIKOND3300"], hits)


def t_roles_rules():
    print("names to roles (DS.5.1 item 2) and base3's rules (item 3)")
    row = {"key": "zenodo_15808623__ABUTH_week_3_IMG_0001_frame_0004", "source": SIU,
           "rel": "ABUTH_week_3_IMG_0001_frame_0004.jpg", "boxes": [(12, 0.5, 0.5, 0.2, 0.2), (12, 0.2, 0.2, 0.1, 0.1)]}
    check("SIU's unit is its video's EPPO code", DD.unit_of(row, 12) == "eppo:ABUTH" and
          DD.siu_video(row["rel"]) == ("ABUTH_week_3_IMG_0001", "ABUTH"))
    keep, sp, dropped, unres = DD.weed_boxes(row, {"eppo:ABUTH": "weed"})
    check("a weed unit's boxes become class 12, species kept per box", len(keep) == 2 and sp == ["eppo:ABUTH"] * 2
          and not unres)
    keep, sp, dropped, unres = DD.weed_boxes(dict(row, rel="SORVU_week_1_IMG_9_frame_1.jpg"), {"eppo:SORVU": "crop"})
    check("a crop unit's boxes are dropped as background", not keep and dropped == {"crop": 2})
    r2 = {"key": "k", "source": "x", "boxes": [(4, 0.5, 0.5, 0.2, 0.2), (13, 0.3, 0.3, 0.1, 0.1)]}
    keep, sp, dropped, unres = DD.weed_boxes(r2, {"inc:4": "weed"})
    check("an undecided unit (the unmapped id) is unresolved: the image is held", unres == ["inc:13"])
    check("the sheet's suggestions: targets weed, OtherPlant and unmapped unresolved, SORVU crop",
          DD.suggest_role("x", "inc:3") == "weed" and DD.suggest_role("x", "inc:12") == "unresolved" and
          DD.suggest_role(SIU, "eppo:SORVU") == "crop" and DD.suggest_role(SIU, "eppo:AMAPA") == "weed")
    rules, _conf = DD.load_rules()
    small = [{"key": "s%d" % i, "source": "tiny", "W": 640, "H": 640, "boxes": [(0.5, 0.5, 0.02, 0.02)], "drop": None}
             for i in range(5)]
    ok = [{"key": "o%d" % i, "source": "fine", "W": 640, "H": 640, "boxes": [(0.5, 0.5, 0.2, 0.2), (0.3, 0.3, 0.2,
                                                                                                   0.2)],
           "drop": None} for i in range(8)]
    ok[0]["boxes"] = [(0.5, 0.5, 0.2, 0.2), (0.2, 0.2, 0.005, 0.005)]
    ok.append({"key": "short", "source": "fine", "W": 640, "H": 300, "boxes": [(0.5, 0.5, 0.2, 0.2)], "drop": None})
    ok.append({"key": "huge", "source": "fine", "W": 640, "H": 640, "boxes": [(0.5, 0.5, 0.99, 0.99),
                                                                              (0.5, 0.5, 0.2, 0.2)], "drop": None})
    conv = DD.apply_rules(small + ok, rules)
    check("a source whose median weed box is under 32 px is excluded whole (convention)",
          not conv["tiny"]["passes"] and all(r["drop"] == "source_convention" for r in small))
    o0 = ok[0]
    check("a box under 8 px is masked (kept boxes stay)", o0["drop"] is None and len(o0["mask"]) == 1 and
          len(o0["boxes"]) == 1)
    check("an image under 320 px on its short side and one with a box over 90 % are dropped",
          ok[8]["drop"] == "short_side" and ok[9]["drop"] == "box_over_90", [(o["key"], o["drop"]) for o in ok])


def t_thin_cap_floor():
    print("thin(10), the family cap and the cwd12 floor (DS.0 items 3 and 15)")
    base = h64(5)
    frames = [{"key": "v_f%02d" % i, "dhash": flip(base, (i % 4) * 4, 0) if i < 8 else h64(100 + i)}
              for i in range(12)]
    kept, dropped = DD.thin(frames, "vid1")
    check("frames within 10 bits of a kept frame are dropped, the rest kept (the series' seeded order)",
          len(kept) + len(dropped) == 12 and len(kept) >= 4 and all(d is None or d <= 10 for d in dropped.values()),
          (kept, dropped))
    check("  the order is the series' seed's (stable)", DD.series_order("vid1", [f["key"] for f in frames]) ==
          DD.series_order("vid1", [f["key"] for f in reversed(frames)]))
    kept2, dropped2 = DD.thin(frames, "vid1", seeded_hashes=[base])
    check("  S0's frames of the series count as kept: a frame near one of them is dropped",
          all(f["key"] not in kept2 for f in frames if f["dhash"] == base), (kept2, dropped2))
    rows = [{"key": "v%d_f%d" % (v, f), "series": "v%d" % v} for v in range(6) for f in range(10)]
    kept, dropped, rec = DD.family_cap(rows, base_count=10, other_count=80, family="siu")
    allowed = int(0.35 / 0.65 * 80) - 10
    order = sorted({r["series"] for r in rows}, key=lambda v: (C.stable_int("inc2/ds1/cap/siu/%s" % v), v))
    check("the cap holds the family at 35 %% of the rows before repeats (%d new allowed), dropping whole videos "
          "in the seeded order" % allowed, len(kept) <= allowed and all(len([r for r in kept if r["series"] == v]) in
                                                                         (0, 10) for v in order) and
          rec["allowed_new"] == allowed, rec)
    check("  the videos dropped are the last in the order that do not fit", dropped == [v for v in order if v not in
                                                                                         {r["series"] for r in kept}])
    n_s0, c = 39500, 3005
    r = DD.floor_repeats(60000, n_s0, c)
    check("the floor's repeat count r = round_half_up((N - C) / (N_S0 - C)), N the rows before repeats: 2 for "
          "60,000, 1 at S0's size, 3 for 116,000", r == 2 and DD.floor_repeats(n_s0, n_s0, c) == 1 and
          DD.floor_repeats(116000, n_s0, c) == 3)
    n2 = 2 * n_s0 - c                         # (N - C) / (N_S0 - C) = 2 exactly
    r2 = DD.floor_repeats(n2, n_s0, c)
    ep0, ep2 = RC.budget_epochs(n_s0), RC.budget_epochs(n2 + (r2 - 1) * c)
    check("  where the ratio is whole, train_core's image-epochs equal S0's (30 epochs a row at 1.2 M)",
          r2 == 2 and ep0 == 30 and ep2 * r2 == ep0, (ep0, ep2, r2))
    reps = DD.repeat_rows([{"key": "tc_a", "image": "/a.png"}], 3)
    check("  repeat rows name the same files under <key>__r<j>", [x["key"] for x in reps] == ["tc_a__r1", "tc_a__r2"]
          and all(x["image"] == "/a.png" for x in reps))
    s1c = [{"key": "tc%d" % i} for i in range(5)] + [{"key": "x%03d" % i} for i in range(60)]
    s1cr = DD.s1cr_draw(s1c, 30, {"tc%d" % i for i in range(5)})
    check("S1c-R: exactly S0's size, every train_core row kept, the rest by stable_int(inc2/ds1/r/<key>)",
          len(s1cr) == 30 and sum(1 for x in s1cr if x["key"].startswith("tc")) == 5 and
          [x["key"] for x in s1cr[5:]] == [x["key"] for x in sorted(s1c[5:], key=lambda r: (C.stable_int(
              "inc2/ds1/r/%s" % r["key"]), r["key"]))][:25])


def t_k_arms():
    print("the K arms (DS.5.3): KD, KT and KF at equal N_K")
    siu = []
    for v, eppo in enumerate(["AMAPA", "AMAPA", "AMATA", "ABUTH", "ABUTH", "CHEAL", "SORVU"]):
        for f in range(12):
            siu.append({"key": "%s_v%d_f%02d" % (eppo, v, f), "series": "%s_v%d" % (eppo, v), "eppo": eppo,
                        "thin": f % 2 == 0})
    kf = [{"key": "kf%03d" % i, "thin": True} for i in range(100)]
    pools, t, n_k = DD.k_pools(siu, kf, crop_eppo=("SORVU",))
    check("T is the thinned target-species frames (AMAPA, AMATA, SIDSP); crops out; N_K the smallest supply",
          t == 18 and n_k == 36 and pools["supplies"] == {"KD": 72, "KT": 36, "KF": 100}, (t, n_k, pools["supplies"]))
    kd, kt, kfr = DD.kd_draw(pools, t, n_k), DD.kt_draw(pools, t, n_k), DD.kf_draw(pools, n_k)
    check("each K arm adds exactly N_K rows", len(kd) == len(kt) == len(kfr) == n_k)
    check("KD takes T target frames then non-target ones (no crop video)", sum(1 for r in kd if r["eppo"] in
                                                                                DD.SIU_TARGET_EPPO) == t and
          not any(r["eppo"] == "SORVU" for r in kd))
    check("KT takes T thinned target frames round-robin, then thinned non-target frames", all(r["thin"] for r in kt)
          and sum(1 for r in kt if r["eppo"] in DD.SIU_TARGET_EPPO) == t and len({r["series"] for r in kt if r[
              "eppo"] in DD.SIU_TARGET_EPPO}) == 3)
    check("KF's rows are thinned base B rows in the seeded order", all(r["thin"] for r in kfr))
    check("a K pool under N_K refuses its draw", refused(DD.kf_draw, {"kf": kf[:10]}, n_k))


def t_groups_draws():
    print("capture groups and the exam draws (DS.5.4)")
    rows = [{"key": "a%d" % i, "sha256": "sa%d" % i, "dhash": h64(i), "variants": None, "source": "s",
             "capture": "s|g%d" % (i // 2)} for i in range(10)]
    g, dg = DD.capture_groups(rows)
    check("the same intake capture group joins its images", g["a0"] == g["a1"] and g["a0"] != g["a2"])
    want = "g%s" % hashlib.sha256("\n".join(sorted(["sha:sa0", "sha:sa1"])).encode()).hexdigest()[:16]
    check("a group is named by the sha256 of its members' sorted sha256s", g["a0"] == want)
    rows.append({"key": "a10", "sha256": "sa10", "dhash": flip(h64(4), 3), "variants": None, "source": "s"})
    g2, _ = DD.capture_groups(rows)
    check("a 6-bit dHash relation joins groups", g2["a10"] == g2["a4"])
    g3, _ = DD.capture_groups(rows, links=[("a2", "a8")])
    check("a DINOv2 link joins groups", g3["a2"] == g3["a8"])
    by = {"src_a": [{"key": "a%03d" % i} for i in range(400)], "src_b": [{"key": "b%03d" % i} for i in range(200)]}
    groups = {r["key"]: "G%s" % r["key"][:-1] for s, rs in by.items() for r in rs}
    digests = {gg: hashlib.sha256(gg.encode()).hexdigest() for gg in set(groups.values())}
    d1 = DD.draw_exams(by, groups, digests, {"src_a": 400, "src_b": 200}, new_sources={"src_a", "src_b"})
    check("MD about 5 % of a source (at least 15), then MT-W v2 about 10 % more (at least 30)",
          d1["src_a"]["md_images"] >= 20 and d1["src_a"]["mtw_images"] >= 40 and d1["src_b"]["md_images"] >= 15 and
          d1["src_b"]["mtw_images"] >= 30, d1)
    check("  together at most 30 % of the source", all(d["md_images"] + d["mtw_images"] <= 0.3 * d["admitted"]
                                                      for d in d1.values()))
    by2 = dict(by, src_c=[{"key": "c%03d" % i} for i in range(100)])
    groups2 = dict(groups, **{r["key"]: "G%s" % r["key"][:-1] for r in by2["src_c"]})
    digests2 = dict(digests, **{gg: hashlib.sha256(gg.encode()).hexdigest() for gg in set(groups2.values())})
    d2 = DD.draw_exams(by2, groups2, digests2, {"src_a": 400, "src_b": 200, "src_c": 100}, new_sources={
        "src_a", "src_b", "src_c"})
    check("a new source leaves the other sources' draws unchanged", d2["src_a"] == d1["src_a"] and
          d2["src_b"] == d1["src_b"])
    t = DD.enlarge_targets(d1, 2.0)
    d3 = DD.draw_exams(by, groups, digests, {"src_a": 400, "src_b": 200}, new_sources=(), md_target=t)
    check("an enlargement keeps the order: the earlier MD groups first, more after them",
          d3["src_a"]["md"][:len(d1["src_a"]["md"])] == d1["src_a"]["md"] and len(d3["src_a"]["md"]) >
          len(d1["src_a"]["md"]))
    big = {"src_d": [{"key": "d%04d" % i} for i in range(1200)]}
    gbig = {r["key"]: ("BIG" if i < 600 else "g%d" % (i // 10)) for i, r in enumerate(big["src_d"])}
    dbig = {gg: hashlib.sha256(gg.encode()).hexdigest() for gg in set(gbig.values())}
    d4 = DD.draw_exams(big, gbig, dbig, {"src_d": 1200}, new_sources={"src_d"})
    check("a group of more than 500 images is never drawn", "BIG" not in d4["src_d"]["md"] + d4["src_d"]["mtw"])


def t_guard():
    print("the leakage guard both directions (DS.5.1 item 7) and EXIF after it (item 8)")
    ex_h = h64(77)
    ex = {"ood23": {"keys": ["ood23__x_1"], "H": np.asarray([variants(ex_h, 77)], dtype=np.uint64),
                    "ok": np.asarray([True])}}
    rot = variants(h64(78), 78)
    rot[3] = flip(ex_h, 4)                      # the row's rot90 variant 4 bits from the exam image
    rows = [{"key": "near", "dhash": rot[0], "variants": rot}, {"key": "far", "dhash": h64(79),
                                                               "variants": variants(h64(79), 79)}]
    hits = DD.dhash_hits(rows, ex)
    check("a copy found under one of the 8 variants (rot90) within 6 bits is a hit; an unrelated row is not",
          "near" in hits and "far" not in hits and hits["near"][0]["exam"] == "ood23", hits)
    ex_rows = [{"key": "e1", "dhash": rot[0], "variants": [rot[0]] + [h64(i) for i in range(7)]}]
    ex2 = {"md": {"keys": ["md|e1"], "H": np.asarray([ex_rows[0]["variants"]], dtype=np.uint64),
                  "ok": np.asarray([True])}}
    rows2 = [{"key": "r1", "dhash": h64(81), "variants": [h64(81)] + [flip(rot[0], 2)] + [h64(i + 50) for i in
                                                                                         range(6)]}]
    check("  the other direction: the exam image's id within 6 bits of the row's hflip", "r1" in DD.dhash_hits(
        rows2, ex2))
    sc = StubScanner({"/img/dino_only.png": ("ood23", "ood23__y_2")})
    eh = DD.embed_hits([{"key": "dino", "image": "/img/dino_only.png"}, {"key": "clean", "image": "/img/c.png"}], sc)
    check("a copy only the DINOv2 detector finds is a hit", list(eh) == ["dino"] and eh["dino"][0]["how"] == "embed")
    exam_rows = [{"exam": "md_s0", "key": "m1", "group": "gX", "capture_keys": [["capture", "src|tray7"]],
                  "sha256": "sm1"}]
    gh = DD.group_hits([{"key": "t1", "source": "src", "capture": "src|tray7"}, {"key": "t2", "source": "src"},
                        {"key": "t3", "source": "src", "sha256": "sm1"}], exam_rows)
    check("a row sharing a capture relation (or the bytes) with an MD row is a hit", sorted(gh) == ["t1", "t3"], gh)
    check("EXIF orientation 2-8 refused, none, 0 and 1 not", DD.exif_refusal({"image": "a"}, lambda p: 6) ==
          "exif_orientation" and DD.exif_refusal({"image": "a"}, lambda p: 1) is None and
          DD.exif_refusal({"image": "a"}, lambda p: None) is None)


# ------------------------------------------------------------------ the world
def build_world():
    """Base B, D1's B', base A, the candidates, the exams, the decisions (module docstring)."""
    src = TMP / "src"
    base_b, bprime, base_a, base_man = {}, {}, {}, {}
    setting = {}

    def add_base(key, source, kind, seed, setting_=None, rel=None):
        img = make_img(src / "base" / ("%s.png" % key), seed)
        lab = make_label(src / "base" / ("%s.txt" % key), [(12, 0.5, 0.5, 0.25, 0.25)])
        h = h64(seed)
        HASHES[img] = (h, variants(h, seed))
        sha = C.sha256_file(img)
        man = {"image": img, "label": lab, "sha256": sha, "label_sha256": C.sha256_file(lab), "source": source,
               "session": "", "key": key}
        prov = dict(man, kind=kind, arms=["A", "B"] if kind == B3.BASE_KIND else ["B"], original_sha256=sha,
                    original_image=img, dhash=h, variants=variants(h, seed), group="gb_%s" % key,
                    capture_keys=[], _boxes=[(0.5, 0.5, 0.25, 0.25)], holdout_v1=False)
        if rel:
            prov["original_image"] = rel
        base_b[key], bprime[key], base_man[key] = prov, man, man
        if kind == B3.BASE_KIND:
            base_a[key] = man
        setting[key] = setting_
    for i in range(6):
        add_base("train_core__20210901_x_%d" % i, "cottonweeddet12/train", B3.BASE_KIND, 1000 + i, "Field")
    for i in range(2):
        add_base("ndsu_%d" % i, "project_agml__imageweeds_weed_detection", B3.BASE_KIND, 1100 + i, "Field")
    for i in range(120):
        add_base("rf_field_a__img%03d" % i, "rf_field_a", "external", 1200 + i, "Field")
    for i in range(80):
        add_base("rf_turf_b__img%03d" % i, "rf_turf_b", "external", 1400 + i, "Turf")
    for i in range(60):
        add_base("mediatum__tray%d_%d" % (i // 5, i), "mediatum_1717366", "external", 1600 + i, "Indoor")
    # SIU frames base B already holds (they seed thin(10) of their videos)
    siu_seed_h = {}
    for v in ("AMAPA_week_1_IMG_0001", "ABUTH_week_1_IMG_0002"):
        key = "%s__%s_frame_0000" % (SIU, v)
        add_base(key, SIU, "external", 1800 + len(siu_seed_h), "Indoor", rel="%s_frame_0000.jpg" % v)
        siu_seed_h[v] = base_b[key]["dhash"]
    # a B' row 6 bits from an ood23 image: the guard drops it from B'o; a base_v2 row with a dev hit: kept, listed
    ood_h = h64(9001)
    base_b["rf_field_a__img005"]["variants"][2] = flip(ood_h, 5)
    dev_h = h64(9002)
    base_b["train_core__20210901_x_0"]["variants"][1] = flip(dev_h, 2)
    cands = []

    def cand(key, source, seed, boxes, rel=None, series=None, eppo=None, capture=None, holds=(), refused=None,
             admission="whole", h=None, size=(360, 360)):
        img = make_img(src / "cand" / source / ("%s.png" % key.replace("/", "_")), seed, size)
        lab = make_label(src / "cand" / source / ("%s.txt" % key.replace("/", "_")), boxes)
        hh = h if h is not None else h64(seed)
        HASHES[img] = (hh, variants(hh, seed))
        cands.append({"key": key, "source": source, "pool": "p3", "image": img, "sha256": C.sha256_file(img),
                      "label": lab, "boxes": [tuple(b) for b in boxes], "W": size[0], "H": size[1],
                      "rel": rel or key, "series": series, "eppo": eppo, "capture": capture, "stem": None,
                      "session": "", "dhash": hh, "variants": None, "licence": "cc-by", "research_only": False,
                      "holds": list(holds), "refused": refused, "admission": admission, "test_v1": None,
                      "batch": "b0009"})
    for i in range(200):
        cand("src_field__p%03d" % i, "src_field", 2000 + i, [(12, 0.5, 0.5, 0.2, 0.2), (12, 0.25, 0.3, 0.1, 0.12)],
             capture="src_field|pair%03d" % (i // 2))
    cand("src_field__held", "src_field", 2300, [(12, 0.5, 0.5, 0.2, 0.2)], holds=["funnel_F9"])
    cand("src_field__lic", "src_field", 2301, [(12, 0.5, 0.5, 0.2, 0.2)], holds=["licence"])
    cand("src_field__ood_dino", "src_field", 2302, [(12, 0.5, 0.5, 0.2, 0.2)])
    cand("src_field__exif", "src_field", 2303, [(12, 0.5, 0.5, 0.2, 0.2)])
    cand("src_field__dup", "src_field", 2304, [(12, 0.5, 0.5, 0.2, 0.2), (12, 0.25, 0.3, 0.1, 0.12)],
         h=flip(h64(2010), 2))
    cand("src_field__h10d", "src_field", 2305, [(12, 0.5, 0.5, 0.2, 0.2)], h=flip(h64(9003), 3))
    for i in range(30):
        cand("src_reup__IphoneSE_YL_Morningglory_%03d_JPG.rf.ab" % i, "src_reup", 2400 + i, [(4, 0.5, 0.5, 0.2, 0.2)],
             rel="IphoneSE_YL_Morningglory_%03d_JPG.rf.ab.jpg" % i)
    for i in range(30):
        cand("src_crop__%03d" % i, "src_crop", 2500 + i, [(12, 0.5, 0.5, 0.2, 0.2)])
    for i in range(30):
        cand("src_devcopy__%03d" % i, "src_devcopy", 2600 + i, [(12, 0.5, 0.5, 0.2, 0.2)],
             h=dev_h if i == 0 else None)
    for vi, (v, eppo) in enumerate([("AMAPA_week_1_IMG_0001", "AMAPA"), ("AMAPA_week_2_IMG_0003", "AMAPA"),
                                    ("AMATA_week_1_IMG_0004", "AMATA"), ("ABUTH_week_1_IMG_0002", "ABUTH"),
                                    ("ABUTH_week_2_IMG_0005", "ABUTH"), ("CHEAL_week_1_IMG_0006", "CHEAL"),
                                    ("SORVU_week_1_IMG_0007", "SORVU")]):
        base_h = siu_seed_h.get(v, h64(3000 + vi))
        for f in range(1, 13):
            hh = flip(base_h, 4) if f in (1, 2) else h64(3100 + vi * 50 + f)
            cand("%s__%s_frame_%04d" % (SIU, v, f), SIU, 3200 + vi * 50 + f, [(12, 0.5, 0.5, 0.2, 0.2)],
                 rel="%s_frame_%04d.jpg" % (v, f), series=v, eppo=eppo, admission="not_admitted" if f % 3 == 0
                 else "whole", h=hh)
    for c in cands:
        if c["source"] == SIU:
            c["pool"] = "p1"
    exams = {}

    def ex(name, keys, hs):
        exams[name] = {"keys": keys, "H": np.asarray([variants(h, 50000 + i) for i, h in enumerate(hs)],
                                                     dtype=np.uint64), "ok": np.ones(len(keys), dtype=bool)}
    ex("dev", ["dev__a", "dev__b"], [dev_h, h64(9004)])
    ex("ood23", ["ood23__20230801_iPhone12_WY_1"], [ood_h])
    ex("h10d", ["h10d__k1"], [h64(9003)])
    scanners = {"ood": StubScanner({c["image"]: ("ood23", "ood23__20230801_iPhone12_WY_1") for c in cands
                                    if c["key"] == "src_field__ood_dino"})}
    decisions = {
        "src_field": {"decision": "accept", "setting": "Field", "roles": {"inc:12": "weed"}, "audit_required": True,
                      "audited": True},
        "src_crop": {"decision": "accept", "setting": "Field", "roles": {"inc:12": "crop"}},
        "src_reup": {"decision": "reject", "setting": "Field", "roles": {}},
        "src_devcopy": {"decision": "accept", "setting": "Field", "roles": {"inc:12": "weed"}},
        SIU: {"decision": "accept_training_only", "setting": "Indoor", "audit_required": True, "audited": True,
              "roles": {"eppo:AMAPA": "weed", "eppo:AMATA": "weed", "eppo:ABUTH": "weed", "eppo:CHEAL": "weed",
                        "eppo:SORVU": "crop"}}}
    orient = {c["image"]: 6 for c in cands if c["key"] == "src_field__exif"}
    return DD.Inputs(base_b=base_b, bprime=bprime, base_a=base_a, base_b_man=base_man, setting=setting,
                     tc_keys={k for k in base_b if k.startswith("train_core__")},
                     ndsu_keys={k for k in base_b if k.startswith("ndsu_")}, candidates=cands, decisions=decisions,
                     exams=exams, scanners=scanners, eval_sources={"project_agml__crop_weed_detection_latvia"},
                     families={SIU: "siu", "mediatum_1717366": "mfwd"}, hasher=hasher,
                     read_orientation=lambda p: orient.get(str(p)), production=False)


def t_world():
    print("the world end to end: screen, pools, MD-S0, draw, manifests")
    inp = build_world()
    pr = TMP / "proofs"
    pr.mkdir(parents=True, exist_ok=True)
    (pr / "a.ots").write_bytes(b"a")
    (pr / "b.ots").write_bytes(b"b")
    D.timestamp_record(*[str(pathlib.Path(__file__).resolve().parents[2] / "docs" / "prereg" / n) for n in (
        "DS1_2026-10-29.md.ots", "DS1_2026-10-29.stamp.txt.ots")])
    scr = DD.screen(1, inp)
    srcs = scr["sources"]
    check("the screen records every candidate source with its units and counts", set(srcs) >= {
        "src_field", "src_reup", "src_crop", "src_devcopy", SIU} and srcs["src_field"]["out"].get("held:funnel_F9") == 1,
          {k: v.get("out") for k, v in srcs.items()})
    check("  the capture-name scan flags the re-upload (role C once its sheet shows the pairs)",
          srcs["src_reup"]["capture_names"]["hits"] == 30 and srcs["src_field"]["capture_names"]["hits"] == 0)
    check("  one dHash copy of a cwd12 dev image quarantines its source whole (D28-v2)",
          srcs["src_devcopy"]["quarantine_whole"] and not srcs["src_field"]["quarantine_whole"])
    check("  SIU's units are EPPO codes, SORVU among them", "eppo:SORVU" in srcs[SIU]["units"] and
          srcs[SIU]["units"]["eppo:SORVU"]["suggested"] == "crop")
    check("  a screen revision is written once and covers its candidates", len(DD.screen_revisions(1)) == 1 and
          not DD.screen(1, inp)["candidates"])
    doc, pooled = DD.pools(1, inp)
    out_field = doc["sources"]["src_field"]["out"]
    check("pools: only ready sources (a reject and a quarantined source are not)", set(pooled) == {
        "src_field", SIU} and doc["not_ready"].get("src_reup") == "rejected" and "quarantined" in
          doc["not_ready"].get("src_devcopy", ""), (sorted(pooled), doc["not_ready"]))
    check("  src_crop's images hold only crop boxes: every one dropped (no weed box)", doc["sources"]["src_crop"][
        "admitted"] == 0)
    check("  the guard dropped the DINOv2-only ood23 copy and the H10d near copy; EXIF refused after the guard; "
          "the 2-bit duplicate deduped", out_field.get("guard") == 2 and out_field.get("exif_orientation") == 1 and
          out_field.get("duplicate") == 1, out_field)
    check("  a licence hold is admitted, flagged by its licence", any(r["key"] == "src_field__lic" for r in
                                                                      pooled["src_field"]))
    siu = pooled[SIU]
    check("  SIU: SORVU's frames dropped (crop), frames not admitted at species level kept, thin(10) marks the "
          "near-repeats", not any(r.get("eppo") == "SORVU" for r in siu) and any(r["key"].endswith("_0003") for r in
                                                                                    siu) and
          not all(r.get("thin") for r in siu), [(r["key"][-30:], r.get("thin")) for r in siu][:4])
    seeded_out = [r for r in siu if r["series"] == "AMAPA_week_1_IMG_0001" and r["key"].endswith(("_0001", "_0002"))]
    check("  frames 4 bits from S0's frame of their video are thinned out (S0 counts as kept)",
          seeded_out and not any(r.get("thin") for r in seeded_out), seeded_out)
    # MD-S0
    d1 = DD.md_s0(inp)
    check("MD-S0 drawn as a draft from base B's external non-SIU sources (base_v2 and SIU never)",
          d1["status"] == "drawn" and d1["images"] > 0 and not DD.frozen("md_s0"), d1.get("status"))
    md_rows = DD.read_jsonl(DD.exam_dir("md_s0") / "draft_r001" / "rows.jsonl")
    check("  no base_v2, SIU or NDSU row among its rows; its own root and LOCK", not any(r["source"] in (
        SIU, "cottonweeddet12/train") or r["key"].startswith("ndsu") for r in md_rows) and
          (DD.exam_dir("md_s0") / "draft_r001" / "root" / "splits" / "v1" / "LOCK.json").is_file())
    check("  the manifests wait for its freeze", "not frozen" in DD.manifests(1, inp)["status"])
    check("  md_s0 waits for its power record", "waiting" in DD.md_s0(inp)["status"])
    D.write_once(D.root() / "power" / "draft_r001.json", {"md_s0": {"enlarge": True, "mde": 0.03}})
    d2 = DD.md_s0(inp)
    check("an MDE over 0.02 enlarges MD-S0 in the same order (a second draft, the first one's groups first)",
          d2["status"] == "drawn" and d2["revision"] == 2 and d2["images"] >= d1["images"], d2.get("status"))
    md2 = {r["key"] for r in DD.read_jsonl(DD.exam_dir("md_s0") / "draft_r002" / "rows.jsonl")}
    check("  the enlargement keeps every earlier MD-S0 row", {r["key"] for r in md_rows} <= md2)
    D.write_once(D.root() / "power" / "draft_r002.json", {"md_s0": {"enlarge": False, "mde": 0.015}})
    d3 = DD.md_s0(inp)
    check("MD-S0 frozen (frozen.json naming the draft) and its rows on DS1's never-train list",
          d3["status"] == "frozen" and DD.frozen("md_s0")["draft"] == "draft_r002" and
          md2 <= {r["key"] for r in DD.never_train_rows() if r["exam"] == "md_s0"})
    check("  never drawn again after its freeze", DD.md_s0(inp)["status"] == "frozen")
    # draws
    dr = DD.draw(1, inp)
    mdn, mtw = DD.mdn_exam("src_field"), DD.mtw_exam("src_field")
    check("draw: src_field's MD-N frozen, its MT-W v2 a draft waiting for the check; SIU never drawn",
          DD.frozen(mdn) and not DD.frozen(mtw) and (DD.exam_dir(mtw) / "draft" / "rows.jsonl").is_file() and
          "SIU" in dr[SIU]["status"], dr.get("src_field"))
    nt = DD.never_train_rows()
    check("  every MD-N and MT-W v2 row on the never-train list at its draw", {r["exam"] for r in nt} >= {
        "md_s0", mdn, mtw})
    dr2 = DD.draw(1, inp, checks={"src_field": {"verdict": "pass", "sha256": "c" * 64, "corrected_labels": {}}})
    check("  after the person's check MT-W v2 is frozen in its own root", DD.frozen(mtw) is not None, dr2)
    # manifests
    built = DD.manifests(1, inp)
    rows = {a: DD.arm_rows(a)[0] for a in ("S0", "KF", "S1C", "S1U", "S1CR") if DD.arm_record_path(a).is_file()}
    check("manifests: S0, KF, S1c, S1u and S1c-R built (KD and KT: the K supply decides)", set(rows) == {
        "S0", "KF", "S1C", "S1U", "S1CR"}, sorted(built))
    s0 = {r["key"] for r in rows["S0"]}
    check("B'o: no MD-S0 row, no NDSU row, the ood23 near copy dropped by the guard, base_v2's dev hit kept",
          not (s0 & md2) and not any(k.startswith("ndsu") for k in s0) and "rf_field_a__img005" not in s0 and
          "train_core__20210901_x_0" in s0, sorted(s0)[:5])
    rec0 = DD.arm_rows("S0")[1]
    check("  the base_v2 row's hit listed in the record", "train_core__20210901_x_0" in rec0["b_prime_o"][
        "base_v2_kept_with_hits"])
    lines = {a: {json.dumps(r, sort_keys=True) for r in rs} for a, rs in rows.items()}
    check("S0 in S1c in S1u, row bytes unchanged", lines["S0"] <= lines["S1C"] <= lines["S1U"])
    s1c = rows["S1C"]
    reps = [r for r in s1c if "__r" in r["key"]]
    r_c = DD.arm_rows("S1C")[1]["repeats"]
    check("the cwd12 floor: train_core rows repeated r - 1 more times in S1c (r from the rows before repeats)",
          len(reps) == (r_c - 1) * 6 and all(r["key"].split("__r")[0] in inp.tc_keys for r in reps), r_c)
    exam_keys = {r["key"] for r in nt}
    check("no MD or MT-W row in any manifest; no row sharing their capture groups", not any(r["key"] in exam_keys
                                                                                           for rs in rows.values()
                                                                                           for r in rs))
    siu_c = sum(1 for r in s1c if r["source"] == SIU and "__r" not in r["key"])
    n_c = sum(1 for r in s1c if "__r" not in r["key"])
    siu_u = sum(1 for r in rows["S1U"] if r["source"] == SIU and "__r" not in r["key"])
    check("SIU capped at 35 % of S1c's rows before repeats (S0's frames never dropped), not in S1u",
          siu_c <= 0.35 * n_c + 1e-9 or siu_c == sum(1 for r in rows["S0"] if r["source"] == SIU), (siu_c, n_c))
    check("  S1u holds every thinned SIU frame", siu_u >= siu_c)
    check("S1c-R: S0's size, train_core kept whole, drawn from S1c's rows", len(rows["S1CR"]) == len(rows["S0"]) and
          inp.tc_keys <= {r["key"] for r in rows["S1CR"]} and {r["key"] for r in rows["S1CR"]} <= {r["key"] for r in
                                                                                                   s1c})
    check("each arm's record prints images, distinct photos and capture groups, and its recipe from its final count",
          all(set(DD.arm_rows(a)[1]["counts"]) >= {"images", "distinct_photos", "capture_groups"} for a in rows) and
          DD.arm_rows("S1C")[1]["recipe"]["epochs"] == RC.budget_epochs(len(s1c)))
    check("an arm's manifest is written once (another set of bytes refused)", refused(DD.write_arm, "S0", rows[
        "S0"][:3]))
    ps = DD.plan_state()
    check("plan_state for the platform: MD-S0 frozen, the arms built, src_field pooled with MD-N and MT-W frozen",
          ps["md_s0"]["frozen"] and set(ps["arms"]) >= {"S0", "S1C"} and ps["phases"]["1"]["sources"]["src_field"][
              "md_n"] and ps["phases"]["1"]["sources"]["src_field"]["mtw"], ps["phases"]["1"]["sources"].get(
              "src_field"))
    # the stream cutter's never-cut list
    from weed_optimizer_framework.tools.inc2 import step1_stream as S1
    L = S1.Layout(inc_dir=C.INC_DIR)
    md_row = next(r for r in nt if r["exam"] == mdn)
    queue_rows = [{"key": "q_by_sha", "sha256": md_row["sha256"], "dhash": h64(1)},
                  {"key": "q_by_hash", "sha256": "z", "dhash": flip(int(md_row["dhash"]), 2)},
                  {"key": "q_clean", "sha256": "y", "dhash": h64(2)}]
    n = S1.test_v1_rows(L, queue_rows)
    check("the cutter's never-cut list reads DS1's list: a row by bytes and one within 6 bits marked 'ds1:', the "
          "rest eligible", n == 2 and queue_rows[0]["test_v1"].startswith("ds1:") and queue_rows[1]["test_v1"]
          .startswith("ds1:") and not queue_rows[2].get("test_v1"), queue_rows)
    bad = DD.never_train_path()
    raw = bad.read_bytes()
    bad.write_bytes(raw + b"{not json\n")
    e = None
    try:
        S1.test_v1_rows(L, [{"key": "x", "sha256": "y", "dhash": 1}])
    except S1.StreamError as ex:
        e = ex
    bad.write_bytes(raw)
    check("  an unreadable list stops the cutter (fail closed)", e is not None)


def t_season():
    print("every S arm is season-free (DS.0 item 6)")
    base = {"image": "/x/a.jpg", "label": "/x/a.txt", "sha256": "a" * 64, "label_sha256": "b" * 64, "session": ""}
    rows = [dict(base, key="src_ok__a", source="src_ok"),
            dict(base, key="tsw22__b", source=DD.TSW[0]),
            dict(base, key="tsw23__c", source=DD.TSW[1]),
            dict(base, key="train_core__%s_d" % DD.SEASON_SESSION, source="cottonweeddet12/train",
                 session=DD.SEASON_SESSION),
            dict(base, key="train_core__x_%s_e" % DD.SEASON_SESSION, source="cottonweeddet12/train")]
    check("season_rows names the tsw22 and tsw23 rows and session 20220129's (by its field or its key), not the rest",
          DD.season_rows(rows) == sorted(r["key"] for r in rows[1:]), DD.season_rows(rows))
    for arm in ("S0", "S1C", "S1U", "S1CR", "S2"):
        for bad in rows[1:]:
            e = refused(DD.write_arm, arm, [rows[0], bad])
            if not (e and "season-free" in str(e)):
                check("%s with %s refused before anything is written" % (arm, bad["key"]), False, e)
                return
    check("an S arm holding any of them is refused before anything is written (each S arm, each kind)",
          not any(DD.arm_path(a).exists() or DD.arm_record_path(a).exists() for a in ("S0", "S1C", "S1U", "S1CR",
                                                                                         "S2")))


def main():
    for fn in (t_eligibility, t_roles_rules, t_thin_cap_floor, t_k_arms, t_season, t_groups_draws, t_guard, t_world):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2500:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
