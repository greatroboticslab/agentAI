#!/usr/bin/env python3
"""V3F data builders (inc2/v3fam_data.py; docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11): V3F, the v3
experiment family D0-D9", V3F.3.7, V3F.4, V3F.5 D1/D4/D5b).

The synthetic world of tests/test_inc2_train.py (LOCK v1 with dev, test, ImageWeeds, ood22 and ood23; LOCK v2
marked testing; the real inc2.guard) plus a small base B of its own: base_v2's parts under their real key forms
(train_core__<session>_<n> with sessions, tsw22__, tsw23__, NDSU), and external rows of every setting (rf_tuf's
three families among them). The build trusts stored hashes, so each never-train case is planted in the stores;
DINOv2 is a stand-in scanner.

What is pinned:
- the constants: the four registered key lists (counts and sha256s of V3F.4), base B's sha256, the setting map
  (20 sources, rf_tuf's rules in order), its counts, R15's seed text and composition, the probe's rule and its
  10 sessions / 476 images / keys sha256, the doses and T_d sizes, the copy families by rank mod 4, the known
  false match (sha256, key, line 12,882), the six E3-frozen sha256s of V3F.1;
- the setting map: the original name from a key (prefix, split prefix, Roboflow suffix), rf_tuf's three
  families and their order, a 21st source and an rf_tuf row of a fourth family refused; R15's ranking equal to
  an independent sha256 ranking; key_lists refusing a base B or a list that is not the registered one;
- the probe draw against an independent implementation of its text (whole sessions, <= 100 images each, <= 500,
  stopping at >= 400; a 101-image session first in order skipped); check_probe refusing another draw;
- the doses nest, families by rank mod 4, round_half_up sizes;
- the both-direction dHash search equal to a brute force (forward only, reverse only, both, 7 bits left out);
- judge: the known false match kept only for exactly its pair; base_v2 rows kept against dev, test and
  ImageWeeds; the season guard dropping base_v2 rows too;
- the stored-hash check: 100 seeded rows per source, a tampered stored hash refused;
- the label transforms by hand (crop90's 50 % and 2 px rules, letterbox, hflip) and on a synthetic image with
  coloured boxes for each copy family (each box's colour found where its label puts it, the box crop90 cuts
  below half dropped), PNGs at long side 640 that read back through load_image, written once by content;
- a probe image with EXIF orientation 6 refused;
- the build end to end: refused before the time stamp, and before D0-S's record; F, N, R15, B' and A' as their
  registered lists before the checks; a forward dHash copy of a dev image dropped (F, N, R15, B', BCLEAN) and R15
  redrawn to F's final count by the same ranking; the known false match (reverse direction only, its test image)
  kept in F, N, R15 and B' and recorded with P0's evidence; base_v2 rows with dev hits recorded and kept; the
  season guard dropping a forward copy of an ood22 image from B' only and a reverse copy of an ood23 image (a
  base_v2 row) from A'1, A'12 and B' (kept in D4); the DINOv2 season hit dropping an NDSU row from A' and B'; a
  copy flagged by DINOv2 dropped from the doses holding it and counted; D0-S's copy removed from every manifest
  (its list's sha256 recorded) and BCLEAN built; every row byte-identical to its parent's line; probe rows absent
  from every T_d except as copies; the doses' copies nested; cold_budget records from the final counts;
  inc2.train's checks recorded; the probe root (LOCK with scorer.py's sha256, the exam materialised, the probe
  exam only); the catch index; D8's train_core and D6's base_v2 recorded; no dev, test or ImageWeeds image or label
  opened; nothing written under splits/v1, v2, v3; a second build refused; arm_manifest; a build or a sweep run
  during which capacity/e3_* changes fails; lists read-only and, with the production values, refused on this
  world;
- with V3FAM_REAL_BASE_B and V3FAM_REAL_BASE_V2 set (the rehearsal of V3F.8 item 3): lists() on the real
  manifests reproduces every registered value.

Run:  python3 tests/test_inc2_v3fam_data.py
"""
import builtins
import collections
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO first)
import test_inc2_v3fam_sweep as SWT  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import splits as SP  # noqa: E402
from weed_optimizer_framework.tools.inc2 import base3 as B3  # noqa: E402
from weed_optimizer_framework.tools.inc2 import guard as G  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_data as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_sweep as SW  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = W.patched
TMP = W.TMP
FIX = TMP / "v3fam_fix"
EMB = SWT.EMB
WORLD = {}
HEX = "0123456789abcdef" * 2


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except F.V3famRefused as e:
        return e
    return None


def lsha(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


# ================================================================== UNITS
def u_constants():
    print("the pre-registered constants")
    check("the four registered key lists of V3F.4 (counts and sha256s) and base B's sha256",
          D.REGISTERED == {
              "F": (15432, "cf108c703d63e1a383322e1570bad349d09055bf19bab50859cb3d5a7383144d"),
              "N": (28953, "bd4861e0b42bd097bbe476c67ac00be97300ea41e3abf8374a840fe9930b15d7"),
              "R15": (15432, "14eb7da803a2df15b4ab8738dad67294cb74dbf255a01377805d77f62c48f987"),
              "BPRIME": (40762, "c2bb53a96bc5b6a4422b46e63464968cd61f674a6181e09de709182a3fe73119")}
          and D.BASE_B_SHA256 == "18d24eade70ba2a5c3c39061eaacd30d148fa00d5ac28fc5685479a7d104899a"
          and D.BASE_B_ROWS == 44485)
    check("the setting map: 20 sources, rf_tuf by family (cwd_weed_ Indoor, mh_ Field, DSC_ Field, in that order); "
          "Field 15,432, Turf 5,910, Indoor 23,143; R15's seed text and composition",
          len(D.SETTING_OF_SOURCE) == 20 and D.SETTING_OF_SOURCE[D.RF_TUF] is None
          and collections.Counter(v for v in D.SETTING_OF_SOURCE.values() if v) == {"Field": 10, "Turf": 6,
                                                                                    "Indoor": 3}
          and D.RF_TUF_RULES == (("starts", "cwd_weed_", "Indoor"), ("starts", "mh_", "Field"),
                                 ("contains", "DSC_", "Field"))
          and D.SETTING_COUNTS == {"Field": 15432, "Turf": 5910, "Indoor": 23143}
          and D.R15_SEED_TEXT == "inc2/v3fam/d4/r15|"
          and D.R15_COMPOSITION == {"Field": 5402, "Turf": 2077, "Indoor": 7953, "SIU": 5393} and D.SIU == "zenodo_15808623")
    check("D1: 3,697 tsw rows and the 26 rows of 20220129_CanonEOS4000D_EO leave A' (3,088)",
          (D.TSW_ROWS, D.SEASON_SESSION, D.SEASON_SESSION_ROWS, D.A_PRIME_ROWS) == (3697, "20220129_CanonEOS4000D_EO",
                                                                                  26, 3088))
    check("D5b: sessions <= 100, probe <= 500, stop at >= 400; 10 sessions, 476 images, keys 14604a97...8c90; doses "
          "0 / 0.25 / 1.0 with 6,335 / 6,454 / 6,811 images; families jpeg75, crop90, letterbox640, hflip by rank mod 4",
          (D.PROBE_SESSION_MAX, D.PROBE_MAX, D.PROBE_MIN, D.PROBE_SESSIONS, D.PROBE_IMAGES) == (100, 500, 400, 10, 476)
          and D.PROBE_KEYS_SHA256 == "14604a97f5e86b690fddb4e22e41f28ed56d41ad75dacfb3d502356da4958c90"
          and D.PROBE_SEED_TEXT == "inc2/v3fam/d5b/probe/%s" and D.COPY_SEED_TEXT == "inc2/v3fam/d5b/copy/%s"
          and D.DOSES == (("D5B0", 0.0), ("D5B25", 0.25), ("D5B100", 1.0))
          and D.DOSE_ROWS == {"D5B0": 6335, "D5B25": 6454, "D5B100": 6811}
          and D.COPY_FAMILIES == ("jpeg75", "crop90", "letterbox640", "hflip") and D.COPY_GREY == 124
          and D.COPY_MIN_AREA == 0.5 and D.COPY_MIN_PX == 2.0 and D.COPY_JPEG_QUALITY == 75
          and sum(D.PROBE_BOXES.values()) == 766 and D.PROBE_BOXES["Sicklepod"] == 0)
    k = D.KNOWN_FALSE_MATCH
    check("the known false match: P0's row (sha256 36b99d86...cacb6, line 12,882) against the test image f75aeb5c..."
          " only, at most 6 bits",
          k["sha256"] == "36b99d86cbc9f29682005467dc3ec2626aa4c91734618344253df6c3622cacb6"
          and k["exam_sha256"] == "f75aeb5c804e224b79c9ca07395124e7e39b6fa5f4b23433779f3e4976dac7ae"
          and k["key"].startswith("rf_school-5ult5__weed-6a90d__train_images_img_01665_png_jpg_rf_8a338581")
          and k["manifest_line"] == 12882 and k["exam"] == "test" and k["max_bits"] == 6 and "0.428" in k["evidence"])
    check("the six E3-frozen files hash as V3F.1 records",
          F.FROZEN_SHA256 == {
              "twostage": "8fe2edc0e0d2a00d07182c3746407281a112c9e50c76662bc9f48a558b328ddc",
              "scorer": "18c008371cd2c021104d034d06b837564016f4509e8a969c2a3f850b09dd2de4",
              "scorer_sidecar": "de7aacda903780b32944866be0b4daa5c6b1f76e2c86965f7a99d85f443d5e52",
              "scorer_native": "ba9eeb928a2b5a0756ae8ac0144b54df2fdfb3570fd011f66b4a8022039d6b7d",
              "verify": "602eba3c935f375179d6f922dce28bbf89006054e5bcec5894f0c9cc7b87aa10",
              "semisup_labeler": "03b28c50699b30a32720e383ff2b713a2638926650007acb92a367d1f450458c"}
          and F.check_frozen() == F.FROZEN_SHA256)
    check("the arms: cold_budget for D1A1, D1B1, D4F, D4N, D4R15, D0BCLEAN; cold for D1A12 and D5B*; the summary's "
          "format is not base3's (v3_claim never claims it)",
          set(D.BUDGET_ARMS) == {"D1A1", "D1B1", "D4F", "D4N", "D4R15", "D0BCLEAN"} and D.FORMAT == "inc2-v3fam-data/1"
          and D.FORMAT != B3.FORMAT and set(D.ARMS) | {"D8M", "D8S"} == set(F.BUILD_ARMS) | {"D0BCLEAN"}
          and {a for a, v in D.ARMS.items() if v["form"] == "12-class"} == {"D1A12", "D5B0", "D5B25", "D5B100"})


def u_setting_map():
    print("the setting map and its file-name families")
    tuf = lambda name: {"key": "rf_tuf__weed-3434e__%s" % name, "source": D.RF_TUF}  # noqa: E731
    check("the original name: source prefix, split prefix and Roboflow's _<ext>_rf_<32 hex> suffix stripped",
          D.orig_name(tuf("train_images_cwd_weed_henbit_w3_45_17_jpg_rf_%s" % HEX)) == "cwd_weed_henbit_w3_45_17"
          and D.orig_name({"key": "train_core__20210812_iPhoneSE_YL_3", "source": "cottonweeddet12/train"})
          == "20210812_iPhoneSE_YL_3"
          and D.orig_name(tuf("valid_images_Copy-of-DSC_0042_JPG_rf_%s" % HEX)) == "Copy-of-DSC_0042")
    got = {n: D.setting_of(tuf(n + "_jpg_rf_" + HEX)) for n in ("train_images_cwd_weed_henbit_w1_0_1",
                                                                "train_images_mh_soy_Class_3",
                                                                "test_images_DSC_0001", "test_images_Copy-of-DSC_9",
                                                                "train_images_mh_DSC_2", "train_images_cwd_weed_DSC_5")}
    check("rf_tuf: cwd_weed_ Indoor, mh_ Field, DSC_ (anywhere) Field; the first rule that matches wins (mh_DSC_ -> "
          "mh, cwd_weed_DSC_ -> cwd_weed)", [v[0] for v in got.values()] == ["Indoor", "Field", "Field", "Field",
                                                                            "Field", "Indoor"]
          and got["train_images_mh_DSC_2"][1] == "mh" and got["train_images_cwd_weed_DSC_5"][1] == "cwd_weed", got)
    check("an rf_tuf row of a fourth family refuses",
          refused(D.setting_of, tuf("train_images_IMG_0001_jpg_rf_" + HEX)) is not None)
    check("a 21st source refuses", refused(D.setting_of, {"key": "rf_new__x__a", "source": "rf_new__x"}) is not None)
    keys = ["k%03d" % i for i in range(40)]
    want = sorted(sorted(keys, key=lambda k: hashlib.sha256(("inc2/v3fam/d4/r15|" + k).encode()).hexdigest())[:13])
    check("R15's ranking is sha256('inc2/v3fam/d4/r15|' + key), the smallest n (no RNG library)",
          D.r15_draw(keys, 13) == want and D.r15_draw(list(reversed(keys)), 13) == want)


def independent_draw(rows, smax, pmax, pmin):
    """The probe rule as V3F.5 D5b's text states it, written apart from the module."""
    by = collections.defaultdict(list)
    for k, r in rows.items():
        if r["source"] == "cottonweeddet12/train":
            by[r["session"]].append(k)

    def si(t):
        return int(hashlib.sha256(t.encode()).hexdigest()[:12], 16) % (2 ** 31 - 1)
    total, out = 0, []
    for s in sorted(by, key=lambda s: (si("inc2/v3fam/d5b/probe/%s" % s), s)):
        if total >= pmin:
            break
        if len(by[s]) <= smax and total + len(by[s]) <= pmax:
            out.append(s)
            total += len(by[s])
    return out, sorted(k for s in out for k in by[s])


def u_probe():
    print("the probe draw, the copy plan and the doses")
    rows = {}
    sizes = [30, 101, 64, 12, 99, 100, 7, 55, 140, 3, 88, 41, 26, 75, 60]
    for i, n in enumerate(sizes):
        s = "2021%04d_iPhoneSE_YL" % (800 + i)
        for j in range(n):
            rows["train_core__%s_%d" % (s, j)] = {"source": "cottonweeddet12/train", "session": s}
    rows["tsw22__20220618_IphoneSE_YL_1"] = {"source": "3seasonweeddet10/data2022", "session": "x"}
    chosen, keys = D.probe_draw(rows)
    want_s, want_k = independent_draw(rows, 100, 500, 400)
    tot = sum(n for _s, n in chosen)
    check("the draw equals an independent implementation of its text (%d sessions, %d images)" % (len(chosen), tot),
          [s for s, _n in chosen] == want_s and keys == want_k, (chosen, want_s))
    check("whole sessions, each <= 100 images, the probe <= 500 (%d)" % tot,
          all(n <= 100 for _s, n in chosen) and tot <= 500
          and all(sum(1 for k in keys if k.startswith("train_core__%s_" % s)) == n for s, n in chosen))
    big = {k: r for k, r in rows.items()}
    first = min(set(r["session"] for r in rows.values() if r["source"] == "cottonweeddet12/train"),
                key=lambda s: (C.stable_int(D.PROBE_SEED_TEXT % s), s))
    for j in range(101):
        big["train_core__%s_x%d" % (first, j)] = {"source": "cottonweeddet12/train", "session": first}
    ch2, _k2 = D.probe_draw(big)
    check("a session over 100 images first in the order is skipped (the <= 100 rule)",
          first not in [s for s, _n in ch2] and [s for s, _n in ch2] == independent_draw(big, 100, 500, 400)[0])
    check("check_probe refuses a draw that is not the amendment's in production, records it in testing",
          refused(D.check_probe, chosen, keys, production=True) is not None
          and D.check_probe(chosen, keys, production=False)["keys_sha256"] == lsha(keys))
    rows2 = dict(rows, **{"train_core__nosession_1": {"source": "cottonweeddet12/train", "session": ""}})
    check("a train_core row without a session refuses the draw", refused(D.probe_draw, rows2) is not None)
    plan = D.copy_plan(keys)
    d25, d100 = D.dose_keys(plan, 0.25), D.dose_keys(plan, 1.0)
    check("the copy plan ranks the probe by stable_int(inc2/v3fam/d5b/copy/<key>), families by rank mod 4; the doses "
          "nest (round_half_up(0.25 x %d) = %d)" % (len(keys), len(d25)),
          [k for k, _i, _f in plan] == sorted(keys, key=lambda k: (C.stable_int(D.COPY_SEED_TEXT % k), k))
          and all(f == D.COPY_FAMILIES[i % 4] for _k, i, f in plan) and set(d25) <= set(d100)
          and len(d25) == B3.round_half_up(0.25 * len(keys)) and len(d100) == len(keys)
          and D.dose_keys(plan, 0.0) == [] and d25 == d100[:len(d25)])


def brute(HA, HE, bits):
    out = {}
    for a in range(len(HA)):
        for e in range(len(HE)):
            d1 = min(bin(int(HA[a, v]) ^ int(HE[e, 0])).count("1") for v in range(8))
            d2 = min(bin(int(HE[e, v]) ^ int(HA[a, 0])).count("1") for v in range(8))
            d = min(d1, d2)
            if d <= bits:
                out[(a, e)] = d
    return out


def u_near_hits():
    print("the both-direction dHash search")
    rng = np.random.RandomState(5)
    HA = rng.randint(0, 2 ** 63, (6, 8), dtype=np.int64).astype(np.uint64)
    HE = rng.randint(0, 2 ** 63, (5, 8), dtype=np.int64).astype(np.uint64)
    HA[0, 3] = HE[1, 0] ^ np.uint64(0b111)          # forward: A's rot90 vs E's dHash, 3 bits
    HA[1, 0] = HE[2, 5] ^ np.uint64(0b11)            # reverse: E's rot270 vs A's dHash, 2 bits
    HA[2, 0] = HE[3, 0] ^ np.uint64(0b1111111)       # 7 bits both ways: out
    HA[3, 0] = HE[4, 0] ^ np.uint64(0b111111)        # 6 bits: in
    ok = np.ones(len(HA), bool)
    got = D.near_hits(HA, ok, HE, np.ones(len(HE), bool))
    check("forward only, reverse only and 6 bits found with their bits, 7 bits left out; equal to a brute force",
          got.get((0, 1)) == 3 and got.get((1, 2)) == 2 and (2, 3) not in got and got.get((3, 4)) == 6
          and got == brute(HA, HE, 6), (got, brute(HA, HE, 6)))
    e = D.hash_matrix
    check("a stored entry without its 8 variants, or whose variants' id is not its dHash, refuses",
          refused(e, [{"key": "a", "dhash": 1, "variants": [1] * 7}]) is not None
          and refused(e, [{"key": "a", "dhash": 1, "variants": [2] + [1] * 7}]) is not None)


def u_judge():
    print("judge: what a hit drops and what it keeps")
    kfm = D.KNOWN_FALSE_MATCH
    rows = {kfm["key"]: {"sha256": kfm["sha256"]}, "ext_a": {"sha256": "aa"}, "train_core__s_1": {"sha256": "bb"}}
    hit_t = {"exam": "test", "exam_key": "t", "exam_sha256": kfm["exam_sha256"], "bits": 6, "how": "dhash"}
    hit_d = {"exam": "dev", "exam_key": "d", "exam_sha256": "dd", "bits": 2, "how": "dhash"}
    dr, kp = D.judge(rows, {kfm["key"]: [hit_t], "ext_a": [hit_d], "train_core__s_1": [hit_d]}, {"train_core__s_1"})
    check("P0's row against its test image is kept as known_false_match; an external hit is dropped; a base_v2 row is "
          "kept and recorded", kp[kfm["key"]]["why"] == "known_false_match" and "ext_a" in dr
          and kp["train_core__s_1"]["why"] == "base_v2_row" and kfm["key"] not in dr, (dr, kp))
    for name, hs in (("another exam image", [hit_t, hit_d]), ("7 bits", [dict(hit_t, bits=7)]),
                     ("its test image by the embedding detector", [dict(hit_t, how="embed")])):
        dr2, _k2 = D.judge(rows, {kfm["key"]: hs}, set())
        check("P0's row with %s is dropped (the exception is that one pair)" % name, kfm["key"] in dr2)
    dr3, _k3 = D.judge({"x": {"sha256": "cc"}}, {"x": [hit_t]}, set())
    check("another row against P0's test image is dropped", "x" in dr3)
    dr4, kp4 = D.judge(rows, {kfm["key"]: [hit_t], "train_core__s_1": [hit_d]}, {"train_core__s_1"}, season=True)
    check("the season guard drops every hit, base_v2 rows and P0's row included", set(dr4) == {kfm["key"],
                                                                                              "train_core__s_1"}
          and not kp4)


def u_hash_check():
    print("the stored-hash check (100 seeded rows of each source recomputed)")
    rows = []
    for i in range(7):
        p = FIX / "hc" / ("h%d.jpg" % i)
        W.make_image(p, 900 + i)
        rows.append({"key": "src%d__h%d" % (i % 2, i), "image": str(p), "source": "src%d" % (i % 2)})
    store = {}
    for r in rows:
        h, v = G.image_hashes(r["image"])
        store[r["key"]] = {"dhash": int(h), "variants": [int(v[k]) for k in G.VARIANTS]}
    rec = D.hash_check(rows, store, "t", n=100)
    check("every row of each source checked when a source has <= 100 (src0 4, src1 3), all equal",
          rec["per_source"] == {"src0": 4, "src1": 3} and rec["equal"], rec)
    rec2 = D.hash_check(rows, store, "t", n=2)
    check("with n = 2 the first two of each source by stable_int(inc2/v3fam/data/hashcheck/<store>/<key>)",
          rec2["per_source"] == {"src0": 2, "src1": 2})
    bad = json.loads(json.dumps(store))
    bad[rows[3]["key"]]["variants"][4] ^= 1
    check("a tampered stored hash (one variant, one bit) refuses", refused(D.hash_check, rows, bad, "t", n=100)
          is not None)


def u_transform_boxes():
    print("the copies' label transforms by hand")
    b = [(0, 0.5, 0.5, 0.2, 0.2)]
    got, dr = D.transform_boxes(b + [(1, 0.04, 0.5, 0.08, 0.2), (2, 0.06, 0.5, 0.12, 0.2), (3, 0.5, 0.5, 0.002, 0.2)],
                                "crop90", 100, 100, crop=(5, 5, 95, 95), out_wh=(640, 640))
    c = got[0]
    check("crop90: a box inside rescaled to the crop; one keeping 3/8 of its area dropped, one keeping 7/12 kept "
          "clipped, one under 2 px in the copy dropped",
          dr == 2 and len(got) == 2 and abs(c[1] - 0.5) < 1e-9 and abs(c[3] - 0.2 / 0.9) < 1e-9
          and abs(got[1][3] - 7 / 90.0) < 1e-9 and abs(got[1][1] - (5 + 12) / 2.0 / 90 + 5 / 90.0) < 1e-9, got)
    def same(a, b):
        return len(a) == len(b) and all(x[0] == y[0] and all(abs(u - v) < 1e-9 for u, v in zip(x[1:], y[1:]))
                                        for x, y in zip(a, b))
    got, _d = D.transform_boxes(b, "letterbox640", 800, 600, out_wh=(640, 480), pad=(0, 80))
    check("letterbox640: y and h follow the 80 px pad and the 480 / 640 content", same(got, [(0, 0.5, 0.5, 0.2, 0.15)]),
          got)
    got, _d = D.transform_boxes([(4, 0.3, 0.6, 0.2, 0.1)], "hflip", 10, 10)
    check("hflip: cx -> 1 - cx; jpeg75: unchanged", same(got, [(4, 0.7, 0.6, 0.2, 0.1)])
          and same(D.transform_boxes(b, "jpeg75", 10, 10)[0], b), got)
    check("an unknown family refuses", refused(D.transform_boxes, b, "blur", 10, 10) is not None)


def u_write_once():
    print("manifests are written once")
    p = FIX / "wo" / "m.jsonl"
    D.write_once_bytes(p, b"a\n")
    check("the same bytes again pass; other bytes refuse", D.write_once_bytes(p, b"a\n") == hashlib.sha256(b"a\n")
          .hexdigest() and refused(D.write_once_bytes, p, b"b\n") is not None)


UNITS = ("u_constants", "u_setting_map", "u_probe", "u_near_hits", "u_judge", "u_hash_check", "u_transform_boxes",
         "u_write_once")


# ================================================================== the fixture world
COLOURS = {0: (230, 30, 30), 1: (30, 200, 30), 2: (30, 30, 230)}


def coloured(path, w=800, h=600, boxes=((0, 100, 100, 300, 250), (1, 500, 300, 700, 500), (2, 0, 250, 60, 350)),
             exif=None):
    im = Image.new("RGB", (w, h), (100, 100, 100))
    d = ImageDraw.Draw(im)
    for c, x0, y0, x1, y1 in boxes:
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=COLOURS[c])
    path.parent.mkdir(parents=True, exist_ok=True)
    if exif is not None:
        ex = Image.Exif()
        ex[0x0112] = exif
        im.save(path, format="JPEG", quality=95, exif=ex)
    else:
        im.save(path, format="PNG")
    lab = [(c, (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, (x1 - x0) / float(w), (y1 - y0) / float(h))
           for c, x0, y0, x1, y1 in boxes]
    return lab


def mk(key, source, session, seed, coll):
    p = FIX / "img" / ("%s.jpg" % key)
    boxes = W.make_image(p, seed)
    l12 = FIX / "lab12" / ("%s.txt" % key)
    l1 = FIX / "lab1" / ("%s.txt" % key)
    C.write_yolo(l12, [(b[0] % 12,) + tuple(b[1:]) for b in boxes])
    C.write_yolo(l1, [(12,) + tuple(b[1:]) for b in boxes])
    base = {"image": str(p), "sha256": W.sha(p), "source": source, "session": session, "key": key}
    coll[key] = {"r12": dict(base, label=str(l12), label_sha256=W.sha(l12)),
                 "r1": dict(base, label=str(l1), label_sha256=W.sha(l1))}


def build_fixture():
    rows = {}
    sessions = [("20210805_iPhoneSE_YL", 3), ("20210809_iPhoneSE_YL", 2), ("20210812_NIKOND3300_YL", 5),
                ("20210813_iPhoneSE_YL", 1), ("20210818_iPhoneSE_YL", 2), ("20210903_YALAL00_EO", 4),
                (D.SEASON_SESSION, 2)]
    seed = 2000
    for s, n in sessions:
        for j in range(n):
            mk("train_core__%s_%d" % (s, j), "cottonweeddet12/train", s, seed, rows)
            seed += 1
    for j in range(3):
        mk("tsw22__20220618_IphoneSE_YL_%d" % j, "3seasonweeddet10/data2022", "20220618_IphoneSE_YL", seed, rows)
        seed += 1
    for j in range(2):
        mk("tsw23__20230614_HTRC_iPhoneSE_YL_%d" % j, "3seasonweeddet10/data2023", "20230614_HTRC_iPhoneSE_YL", seed, rows)
        seed += 1
    mk("project_agml__weed_crop_detection__default_train_000001", "project_agml__weed_crop_detection", "", seed, rows)
    mk("project_agml__imageweeds_aerial_weed_detection__default_train_000002",
       "project_agml__imageweeds_aerial_weed_detection", "", seed + 1, rows)
    seed += 2
    v2keys = sorted(rows)
    ext = [("rf_school-5ult5__weed-6a90d__train_images_img_0000%d_png_jpg_rf_%s" % (j, HEX), "rf_school-5ult5__weed-6a90d")
           for j in range(3)]
    ext += [("kg_yuzhenlu__cottonweeddet3__train_images_20190801_x_1", "kg_yuzhenlu__cottonweeddet3")]
    ext += [("rf_vizerion__weed-detection-5ell6__train_images_ridderzuring_%d_jpg_rf_%s" % (j, HEX),
             "rf_vizerion__weed-detection-5ell6") for j in range(2)]
    ext += [("zenodo_15808623__AMARE_week_1_IMG_%d_frame_1" % j, "zenodo_15808623") for j in range(3)]
    ext += [("rf_tuf__weed-3434e__train_images_cwd_weed_henbit_w1_0_%d_jpg_rf_%s" % (j, HEX), D.RF_TUF) for j in range(2)]
    ext += [("rf_tuf__weed-3434e__valid_images_mh_soy_Class_1_jpg_rf_%s" % HEX, D.RF_TUF),
            ("rf_tuf__weed-3434e__test_images_Copy-of-DSC_0042_JPG_rf_%s" % HEX, D.RF_TUF)]
    for k, s in ext:
        mk(k, s, "", seed, rows)
        seed += 1
    md = FIX / "manifests"
    paths = {"base_b": md / "base_v3_weed.jsonl", "base_a1": md / "base_v2_weed.jsonl", "base_v2": md / "base_v2.jsonl",
             "train_core": md / "train_core.jsonl"}
    C.write_manifest(paths["base_b"], [rows[k]["r1"] for k in rows])
    C.write_manifest(paths["base_a1"], [rows[k]["r1"] for k in v2keys])
    C.write_manifest(paths["base_v2"], [rows[k]["r12"] for k in v2keys])
    C.write_manifest(paths["train_core"], [rows[k]["r12"] for k in v2keys if k.startswith("train_core__")])
    return rows, v2keys, [k for k, _s in ext], paths


def store_of(parent):
    out = {}
    for k, r in parent["rows"].items():
        h, v = G.image_hashes(r["image"])
        out[k] = {"key": k, "image": r["image"], "sha256": r["sha256"], "dhash": int(h),
                  "variants": [int(v[x]) for x in G.VARIANTS], "source": r["source"]}
    return out


def exam_of(exam):
    rows = sorted(WORLD["world"][exam], key=lambda r: r["key"])
    H = []
    for r in rows:
        h, v = G.image_hashes(r["image"])
        H.append([int(v[x]) for x in G.VARIANTS])
    return {"keys": [r["key"] for r in rows], "sha256": [r["sha256"] for r in rows],
            "H": np.asarray(H, np.uint64), "ok": np.ones(len(rows), bool), "image": [r["image"] for r in rows]}


class FakeScanner(object):
    """The DINOv2 detector's interface (scan); flags the files whose image path or key a rule names."""

    def __init__(self, rule):
        self.rule = rule
        self.seen = []

    def scan(self, rows, desc_path=None, procs=1):
        self.seen.extend(r["key"] for r in rows)
        out = {}
        for r in rows:
            e = self.rule(r)
            if e:
                out[r["key"]] = [e]
        return out, set()


def expected_lists(rows, v2keys):
    """F, N, R15 and B' of the fixture by V3F.4's text, written apart from the module."""
    def field(k):
        s = rows[k]["r1"]["source"]
        if s == D.RF_TUF:
            o = k.split("__", 2)[2]
            o = o.split("_images_", 1)[1]
            return o.startswith("mh_") or ("DSC_" in o and not o.startswith("cwd_weed_"))
        return s in ("cottonweeddet12/train", "3seasonweeddet10/data2022", "3seasonweeddet10/data2023",
                     "project_agml__weed_crop_detection", "project_agml__imageweeds_aerial_weed_detection",
                     "rf_school-5ult5__weed-6a90d", "kg_yuzhenlu__cottonweeddet3")
    f = sorted(k for k in rows if field(k))
    n = sorted(k for k in rows if rows[k]["r1"]["source"] != "zenodo_15808623")
    r15 = sorted(sorted(rows, key=lambda k: hashlib.sha256(("inc2/v3fam/d4/r15|" + k).encode()).hexdigest())[:len(f)])
    drop = {k for k in rows if rows[k]["r1"]["source"].startswith("3seasonweeddet10/")
            or k.startswith("train_core__%s_" % D.SEASON_SESSION)}
    return {"F": f, "N": n, "R15": r15, "BPRIME": sorted(set(rows) - drop),
            "APRIME": sorted(set(v2keys) - drop)}


def tree(*dirs):
    out = {}
    for d in dirs:
        d = pathlib.Path(d)
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    out[str(p)] = W.sha(p)
    return out


# ================================================================== WORLD
def w_copies():
    print("the near copies on a synthetic image with coloured boxes")
    p = FIX / "copysrc" / "train_core__20210812_x_1.png"
    lab = coloured(p)
    lp = FIX / "copysrc" / "train_core__20210812_x_1.txt"
    C.write_yolo(lp, lab)
    row = {"key": "train_core__20210812_x_1", "image": str(p), "label": str(lp), "session": "20210812_x"}
    import cv2
    root = FIX / "copies"
    for fam in D.COPY_FAMILIES:
        mrow, rec = D.make_copy(row, fam, root)
        im = cv2.imread(mrow["image"])[:, :, ::-1]
        H, Wd = im.shape[:2]
        boxes = C.read_yolo(mrow["label"])
        okc = []
        for c, cx, cy, bw, bh in boxes:
            px = tuple(int(v) for v in im[int(cy * H), int(cx * Wd)])
            okc.append(all(abs(a - b) <= 30 for a, b in zip(px, COLOURS[c])))
        want_n = 2 if fam == "crop90" else 3
        check("%s: a PNG at long side 640 (%dx%d), %d boxes, each box's colour where its label puts it (%s)"
              % (fam, Wd, H, len(boxes), okc), max(H, Wd) == 640 and len(boxes) == want_n and all(okc)
              and rec["boxes_dropped"] == (1 if fam == "crop90" else 0) and mrow["key"] == D.COPY_KEY % row["key"]
              and mrow["source"] == D.COPY_SOURCE and mrow["session"] == "20210812_x"
              and mrow["sha256"] == W.sha(mrow["image"]) and mrow["label_sha256"] == W.sha(mrow["label"]), (boxes, rec))
        back, _hw = B3._load_640(mrow["image"])
        check("  %s reads back through load_image as written; a second copy is the same file (content-addressed)"
              % fam, back is not None and back.shape == (H, Wd, 3)
              and D.make_copy(row, fam, root)[0]["image"] == mrow["image"])
    lb = cv2.imread(D.make_copy(row, "letterbox640", root)[0]["image"])
    check("letterbox640 is 640 x 640 with grey 124 above the 640 x 480 content", lb.shape == (640, 640, 3)
          and tuple(int(v) for v in lb[5, 320]) == (124, 124, 124) and tuple(int(v) for v in lb[600, 320]) == (124,) * 3)
    p6 = FIX / "copysrc" / "train_core__tag6_1.jpg"
    coloured(p6, exif=6)
    base = {"rows": {"train_core__tag6_1": {"image": str(p6)}, "train_core__20210812_x_1": {"image": str(p)}}}
    check("a probe image with EXIF orientation 6 refuses (a pass on the probe reads none, 0 and 1 only)",
          refused(D.probe_tags, base, ["train_core__tag6_1"]) is not None
          and D.probe_tags(base, ["train_core__20210812_x_1"]) == {"None": 1})


def make_inputs(plants=True):
    rows, v2keys, ext, paths = WORLD["fixture"]
    parents = {k: D.read_parent(p) for k, p in paths.items()}
    hb, hv2 = store_of(parents["base_b"]), store_of(parents["base_v2"])
    ex = {e: exam_of(e) for e in ("dev", "test", "imageweeds", "ood22", "ood23")}
    lists = expected_lists(rows, v2keys)
    reg = {n: (len(lists[n]), lsha(lists[n])) for n in ("F", "N", "R15", "BPRIME")}
    with patched(D, "PROBE_SESSION_MAX", 4), patched(D, "PROBE_MAX", 8), patched(D, "PROBE_MIN", 5):
        _ch, probe = D.probe_draw(parents["base_v2"]["rows"])
    free_tc = [k for k in v2keys if k.startswith("train_core__") and k not in probe
               and not k.startswith("train_core__%s_" % D.SEASON_SESSION)]
    P = {"X1": ext[1], "KFM": ext[0], "X2": ext[3], "TURF": ext[4], "T1": free_tc[0], "T2": free_tc[1],
         "T3": free_tc[2], "N1": "project_agml__weed_crop_detection__default_train_000001",
         "N2": "project_agml__imageweeds_aerial_weed_detection__default_train_000002", "probe": probe,
         "lists": lists}
    if plants:
        def var(e, i):
            return int(e["H"][i][0]), [int(x) for x in e["H"][i]]
        dev0, _ = var(ex["dev"], 0)
        hb[P["X1"]]["variants"][3] = dev0                                   # forward: X1's rot90 = dev 0's dHash
        _t, tv = var(ex["test"], 2)
        h = tv[3] ^ 0b111                                                   # reverse only: test 2's rot90, 3 bits
        hb[P["KFM"]]["dhash"], hb[P["KFM"]]["variants"][0] = h, h
        _d1, dv = var(ex["dev"], 1)
        h = dv[5] ^ 0b11                                                    # reverse: dev 1's rot270, 2 bits
        hb[P["T1"]]["dhash"], hb[P["T1"]]["variants"][0] = h, h
        _i, iv = var(ex["imageweeds"], 0)
        hv2[P["T3"]]["variants"][6] = iv[0]                                 # forward on base_v2's file
        o22, _ = var(ex["ood22"], 0)
        hb[P["X2"]]["variants"][1] = o22                                    # season, forward, external
        _o, ov = var(ex["ood23"], 1)
        h = ov[2] ^ 0b1                                                     # season, reverse, a base_v2 row
        hv2[P["T2"]]["dhash"], hv2[P["T2"]]["variants"][0] = h, h
    kfm = {"key": P["KFM"], "sha256": rows[P["KFM"]]["r1"]["sha256"], "exam": "test",
           "exam_sha256": ex["test"]["sha256"][2], "exam_image": "fixture", "max_bits": 6, "manifest_line": 0,
           "evidence": "fixture"}
    plan = D.copy_plan(probe)
    flagged_copy = D.COPY_KEY % plan[0][0]                                  # in every dose with copies
    n1_img = rows[P["N1"]]["r1"]["image"]
    n2_img = rows[P["N2"]]["r1"]["image"]
    ev_sc = FakeScanner(lambda r: {"eval_split": "dev", "eval_key": "x", "cos": 0.97, "bits": 30}
                        if r["key"] == flagged_copy or r["image"] == n1_img else None)
    ood_sc = FakeScanner(lambda r: {"eval_split": "ood23", "eval_key": "y", "cos": 0.96, "bits": 25}
                         if r["image"] == n2_img else None)
    catch_sc = lambda rr: FakeScanner(lambda r: {"eval_split": "probe", "eval_key": r["key"].split("__", 1)[1],  # noqa
                                                 "cos": 0.99, "bits": 0} if "copy" in r["key"] else None)
    dev_keys = [k for k in ex["dev"]["keys"]]
    sweep = {"copies": [{"pair": "p1", "query": "test/" + ex["test"]["keys"][0], "ref": P["TURF"]}],
             "trigger": SW.trigger(["test/" + ex["test"]["keys"][0]], dev_keys)}
    inputs = dict(parents, hash_b=hb, hash_v2=hv2, exams=ex,
                  scanners={"eval": ev_sc, "ood": ood_sc, "catch_factory": catch_sc}, sweep=sweep,
                  prior=(set(), set()), production=False, registered=reg,
                  base_want=parents["base_b"]["sha256"], record={"sweep_record_sha256": "fixture"})
    P.update(flagged_copy=flagged_copy, kfm=kfm, plan=plan, ev_sc=ev_sc, ood_sc=ood_sc)
    return inputs, P


@contextlib.contextmanager
def small_probe(kfm):
    with patched(D, "PROBE_SESSION_MAX", 4), patched(D, "PROBE_MAX", 8), patched(D, "PROBE_MIN", 5), \
            patched(D, "KNOWN_FALSE_MATCH", kfm):
        yield


def keys_of(path):
    return [json.loads(l)["key"] for l in open(path) if l.strip()]


def w_build():
    print("the build end to end (testing; stored hashes with planted cases; stand-in scanners)")
    rows, v2keys, ext, paths = WORLD["fixture"]
    inputs, P = make_inputs()
    ts = F.root() / "timestamp.json"
    if ts.exists():
        ts.unlink()
    with small_probe(P["kfm"]):
        e = refused(D.build, inputs=inputs, testing=True, procs=1, embedder=EMB)
    check("the build refuses before the amendment's time stamp", e is not None and "time-stamped" in str(e), e)
    ts.parent.mkdir(parents=True, exist_ok=True)
    ts.write_text(json.dumps({"ots": "fixture", "revisions": [{"sha256": F.REVISION_SHA256}]}))
    with small_probe(P["kfm"]):
        e = refused(D.build, inputs=dict(inputs, sweep=None), testing=True, procs=1, embedder=EMB)
    check("the build refuses without D0-S's complete record (L23H writes no manifest before it)", e is not None
          and not D.out_dir().exists(), e)
    bad = dict(inputs, registered=dict(inputs["registered"], R15=(inputs["registered"]["R15"][0], "0" * 64)))
    with small_probe(P["kfm"]):
        e = refused(D.build, inputs=bad, testing=True, procs=1, embedder=EMB)
    check("a key list that is not the registered one refuses before anything is written", e is not None
          and "R15" in str(e) and not D.out_dir().exists(), e)
    with small_probe(P["kfm"]):
        e = refused(D.build, inputs=dict(inputs, base_want="1" * 64), testing=True, procs=1, embedder=EMB)
    check("a base B other than the amendment's refuses", e is not None and "base B" in str(e), e)
    world = WORLD["world"]
    exam_files = {str(pathlib.Path(r[k]).resolve()) for s in ("dev", "test", "imageweeds") for r in world[s]
                  for k in ("image", "label")}
    opened = []
    real = (builtins.open, Image.open, np.fromfile)

    def spy(fn):
        def f(path, *a, **k):
            if isinstance(path, (str, pathlib.Path)):
                opened.append(str(pathlib.Path(path).resolve()))
            return fn(path, *a, **k)
        return f
    before = tree(C.INC_DIR / "splits" / "v1", C.INC_DIR / "splits" / "v2", C.INC_DIR / "splits" / "v3")
    builtins.open, Image.open, np.fromfile = spy(real[0]), spy(real[1]), spy(real[2])
    try:
        with small_probe(P["kfm"]):
            s = D.build(inputs=inputs, testing=True, procs=1, embedder=EMB)
    finally:
        builtins.open, Image.open, np.fromfile = real
    WORLD["summary"], WORLD["P"] = s, P
    hit = sorted(set(opened) & exam_files)
    check("no dev, test or ImageWeeds image or label file was opened (%d)" % len(hit), not hit, hit[:3])
    check("nothing was written under splits/v1, v2 or v3", tree(C.INC_DIR / "splits" / "v1", C.INC_DIR / "splits" / "v2",
                                                                 C.INC_DIR / "splits" / "v3") == before)
    lists = P["lists"]
    kb = s["key_lists_before_checks"]
    check("the key lists before the checks are V3F.4's rule on this world (F %d, N %d, R15 %d, B' %d, A' %d)"
          % tuple(kb[n]["rows"] for n in ("F", "N", "R15", "BPRIME", "APRIME")),
          all(kb[n]["sha256"] == lsha(lists[n]) for n in ("F", "N", "R15", "BPRIME", "APRIME"))
          and s["setting_map"]["registered"]["F"]["sha256"] == lsha(lists["F"]))
    A = {a: set(keys_of(v["manifest"])) for a, v in s["arms"].items()}
    check("the arms built: D4F, D4N, D4R15, D1B1, D1A1, D1A12, D5B0, D5B25, D5B100 and (D0-S's test branch) D0BCLEAN",
          set(A) == set(D.ARMS), sorted(A))
    x1, kfm, x2, turf = P["X1"], P["KFM"], P["X2"], P["TURF"]
    check("a forward dHash copy of a dev image is dropped from F, N, R15, B' and BCLEAN",
          all(x1 not in A[a] for a in ("D4F", "D4N", "D4R15", "D1B1", "D0BCLEAN")) and x1 in s["nevertrain"]
          ["base_b_dropped"] and s["nevertrain"]["base_b_dropped"][x1][0]["exam"] == "dev", s["nevertrain"]["base_b_dropped"])
    check("R15 is redrawn by the same ranking over the eligible rows, to F's final count (%d)" % len(A["D4F"]),
          s["r15"]["redrawn"] and len(A["D4R15"]) == len(A["D4F"])
          and sorted(A["D4R15"]) == D.r15_draw(sorted(set(rows) - {x1, turf}), len(A["D4F"])))
    kn = s["nevertrain"]["known_false_match"]
    check("the known false match (reverse direction only, its test image) is kept in F, N, R15 and B' and recorded",
          all(kfm in A[a] for a in ("D4F", "D4N", "D1B1", "D0BCLEAN")) and (kfm in A["D4R15"]) ==
          (kfm in D.r15_draw(sorted(set(rows) - {x1, turf}), len(A["D4F"]))) and kn["kept"]
          and kn["hits"][0]["exam"] == "test" and s["nevertrain"]["base_b_kept"][kfm]["why"] == "known_false_match"
          and "evidence" in s["nevertrain"]["base_b_kept"][kfm] and s["arms"]["D4F"]["known_false_match"], kn)
    t1, t2, t3 = P["T1"], P["T2"], P["T3"]
    check("base_v2 rows with dev / ImageWeeds hits are recorded with the pair and kept (base B's and base_v2's file)",
          s["nevertrain"]["base_b_kept"][t1]["why"] == "base_v2_row" and t1 in A["D4F"] and t1 in A["D1A12"]
          and s["nevertrain"]["base_v2_kept"][t3]["hits"][0]["exam"] == "imageweeds" and t3 in A["D5B0"]
          and s["nevertrain"]["base_v2_kept"][P["N1"]]["hits"][0]["how"] == "embed")
    check("the season guard: a forward copy of an ood22 image leaves B' only; a reverse copy of an ood23 image (a "
          "base_v2 row) leaves A'1, A'12 and B' but stays in D4 and the doses",
          x2 not in A["D1B1"] and x2 in A["D4F"] and all(t2 not in A[a] for a in ("D1A1", "D1A12", "D1B1"))
          and t2 in A["D4F"] and t2 in A["D5B0"] and set(s["nevertrain"]["season_dropped"]) >= {x2, t2}, sorted(
              s["nevertrain"]["season_dropped"]))
    check("the DINOv2 season hit drops an NDSU row from A' and B' alike (base_v2 rows included)",
          all(P["N2"] not in A[a] for a in ("D1A1", "D1A12", "D1B1")) and P["N2"] in A["D4F"]
          and s["nevertrain"]["season_dropped"][P["N2"]][0]["how"] == "embed")
    check("the season scanner saw every A' and B' file once; the eval scanner only base_v2's train_core and NDSU files "
          "and the copies", s["nevertrain"]["embed_scanned"]["season"] == len(set(lists["BPRIME"]))
          and not [k for k in P["ev_sc"].seen if k.startswith(("tsw2", "rf_", "kg_", "zenodo"))])
    fc = P["flagged_copy"]
    check("a copy the DINOv2 detector flags is dropped from the doses holding it and counted",
          fc in s["nevertrain"]["copies_dropped"] and fc not in A["D5B100"] and fc not in A["D5B25"]
          and s["arms"]["D5B100"]["copies"]["dropped"] == [fc] and s["arms"]["D5B25"]["copies"]["kept"]
          == s["arms"]["D5B25"]["copies"]["listed"] - 1)
    check("D0-S's copy leaves every manifest; its list's sha256 is recorded; BCLEAN = base B minus it and the drops",
          all(turf not in A[a] for a in A) and s["d0s"]["removal_keys"] == [turf]
          and s["d0s"]["removal_sha256"] == lsha([turf]) and A["D0BCLEAN"] == set(rows) - {turf, x1}
          and s["arms"]["D4N"]["removed"]["d0s"] == 1)
    probe = set(P["probe"])
    copies = {D.COPY_KEY % k for k in probe}
    check("probe rows are absent from every T_d except as copies; D5B0 holds no copy; D5B25's copies within D5B100's",
          all(not (A[a] & probe) for a in ("D5B0", "D5B25", "D5B100")) and not (A["D5B0"] & copies)
          and (A["D5B25"] & copies) <= (A["D5B100"] & copies)
          and A["D5B0"] == set(v2keys) - probe and A["D5B100"] - copies == A["D5B0"])
    ok_bytes = True
    for a, v in s["arms"].items():
        parent = {k: l for k, _r, l in D.read_parent(v["parent"]["manifest"])["lines"]}
        for line in open(v["manifest"], "rb"):
            k = json.loads(line)["key"]
            if not k.startswith("v3fam_d5b_copy__") and parent.get(k) != line:
                ok_bytes = False
    check("every row is byte-identical to its parent's line (copies aside)", ok_bytes)
    check("cold_budget records from each budget arm's final count; cold 100 for the 12-class arms",
          all(s["arms"][a]["recipe"] == json.loads(json.dumps(RC.budget_record("m640", s["arms"][a]["images"])))
              for a in D.BUDGET_ARMS) and s["arms"]["D1A12"]["recipe"]["epochs"] == 100)
    check("inc2.train's check_manifest and guard_rows ran on every written manifest",
          all(v["train_checks"]["check_manifest"]["n_train_images"] == v["images"]
              and v["train_checks"]["guard"]["refused"] == 0 for v in s["arms"].values()))
    pr = D.probe_root()
    lock = json.loads((pr / "splits" / "v1" / "LOCK.json").read_text())
    prow = [json.loads(l) for l in open(pr / "splits" / "v1" / "probe.jsonl")]
    check("the probe root: probe.jsonl (base_v2's lines), LOCK.json with scorer.py's sha256, exams/v1/probe "
          "materialised (no problem), the probe exam only",
          lock["scorer_sha256"] == W.sha(pathlib.Path(S.__file__).resolve())
          and lock["manifests"]["probe"] == W.sha(pr / "splits" / "v1" / "probe.jsonl")
          and sorted(r["key"] for r in prow) == sorted(probe)
          and SP.exam_problems("probe", prow, pr / "exams" / "v1" / "probe") == []
          and sorted(os.listdir(pr / "exams" / "v1")) == ["probe"] and sorted(os.listdir(pr / "splits" / "v1"))
          == ["LOCK.json", "probe.jsonl"], lock)
    cr = json.loads((D.catch_dir() / "record.json").read_text())
    check("the catch index: per copy and family, GuardV2, the both-direction dHash, DINOv2 and D0-S's pipeline "
          "(hflip copies caught by both dHash checks)",
          len(cr["copies"]) == len(probe) and set(cr["per_family"]) == set(D.COPY_FAMILIES)
          and all("d0s" in c and "dinov2" in c for c in cr["copies"])
          and all(c["guardv2"] and c["dhash_both"] for c in cr["copies"] if c["family"] == "hflip"), cr["per_family"])
    check("D8's train_core and D6's base_v2 are checked the same way and recorded (kept)",
          s["recorded_only"]["D8_train_core"]["kept"] and t1 not in s["recorded_only"]["D8_train_core"]["hits"]
          and t3 in s["recorded_only"]["D6_base_v2"]["hits"])
    with small_probe(P["kfm"]):
        e = refused(D.build, inputs=inputs, testing=True, procs=1, embedder=EMB)
    check("a second build refuses (summary.json exists)", e is not None and "built once" in str(e), e)
    m, sha, n = D.arm_manifest("D4F")
    check("arm_manifest gives an arm's manifest, sha256 and images; another arm refuses",
          sha == s["arms"]["D4F"]["sha256"] and n == len(A["D4F"]) and refused(D.arm_manifest, "D8M") is not None)
    raw = pathlib.Path(m).read_bytes()
    pathlib.Path(m).write_bytes(raw + b"\n")
    try:
        check("an arm whose manifest no longer hashes as recorded refuses", refused(D.arm_manifest, "D4F") is not None)
    finally:
        pathlib.Path(m).write_bytes(raw)
    check("the family summary is not claimed by inc2.base3.v3_claim (another format)",
          B3.v3_claim(m, sha) is None)


def w_prior():
    print("the earlier test lists")
    s = WORLD["summary"]
    data = pathlib.Path(s["arms"]["D4N"]["manifest"]).read_bytes()
    k = json.loads(data.splitlines()[0])["key"]
    check("a manifest holding a key or sha256 of a test v1 list or its companions refuses",
          refused(D.check_prior, "D4N", data, {k}, set()) is not None
          and refused(D.check_prior, "D4N", data, set(), {json.loads(data.splitlines()[1])["sha256"]}) is not None
          and refused(D.check_prior, "D4N", data, set(), set()) is None)


def w_etrees():
    print("the E-series trees around a family job")
    cap = C.INC_DIR / "capacity"
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "e3_v1.json").write_text("{}")

    def tamper(**_k):
        (cap / "e3_v1.json").write_text(json.dumps({"changed": time.monotonic_ns()}))
        return {}
    with patched(D, "_build", tamper):
        e = refused(D.build, inputs={}, testing=True)
    check("a build during which capacity/e3_* changes fails (V3F.3.7 write isolation)", e is not None
          and "E-series" in str(e), e)
    with patched(SW, "_run", tamper):
        e = refused(SW.run, testing=True)
    check("so does a sweep run", e is not None and "E-series" in str(e), e)
    (cap / "e3_v1.json").unlink()


def w_lists():
    print("lists: read-only, the production values refused on this world")
    _rows, _v2, _ext, paths = WORLD["fixture"]
    t = tree(C.INC_DIR)
    with patched(D, "PROBE_SESSION_MAX", 4), patched(D, "PROBE_MAX", 8), patched(D, "PROBE_MIN", 5):
        got = D.lists(paths["base_b"], paths["base_v2"], production=False)
    check("lists() computes the key lists, the probe and the doses and writes nothing",
          got["lists"]["F"]["sha256"] == lsha(WORLD["P"]["lists"]["F"]) and tree(C.INC_DIR) == t
          and set(got["doses"]) == {"D5B0", "D5B25", "D5B100"})
    e = refused(D.lists, paths["base_b"], paths["base_v2"], production=True)
    check("with the amendment's values it refuses on another base B (the builder reproduces them or refuses)",
          e is not None and "base B" in str(e), e)


def w_real():
    bb, bv = os.environ.get("V3FAM_REAL_BASE_B"), os.environ.get("V3FAM_REAL_BASE_V2")
    if not (bb and bv):
        print("the real manifests: V3FAM_REAL_BASE_B / V3FAM_REAL_BASE_V2 not set (the rehearsal runs this)")
        return
    print("the real manifests (V3F.8 item 3)")
    got = D.lists(bb, bv, production=True)
    check("base B 18d24ead...899a; F, N, R15, B' as registered; A' 3,088; settings 15,432 / 5,910 / 23,143; the probe "
          "10 sessions, 476 images, keys 14604a97...; doses 6,335 / 6,454 / 6,811; P0's row in F, N, R15, B'",
          got["base_b"]["sha256"] == D.BASE_B_SHA256
          and all(got["lists"][n] == {"rows": D.REGISTERED[n][0], "sha256": D.REGISTERED[n][1]} for n in D.REGISTERED)
          and got["lists"]["APRIME"]["rows"] == 3088 and got["record"]["settings"] == D.SETTING_COUNTS
          and got["record"]["r15_composition"] == D.R15_COMPOSITION
          and got["probe"]["keys_sha256"] == D.PROBE_KEYS_SHA256
          and {a: v["images"] for a, v in got["doses"].items()} == D.DOSE_ROWS
          and all(got["known_false_match_in"][n] for n in ("F", "N", "R15", "BPRIME")), got)


def w_cli():
    print("the CLI")
    env = dict(os.environ)
    pkg = str(pathlib.Path(D.__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_data", "lists", "--base-b",
                        str(FIX / "missing.jsonl"), "--base-v2", str(FIX / "missing.jsonl")], env=env,
                       capture_output=True, text=True)
    p2 = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_data", "build"], env=env,
                        capture_output=True, text=True)
    check("lists on a missing file and a second build exit non-zero with [inc2.v3fam_data] (no traceback for a "
          "refusal)", p.returncode != 0 and p2.returncode == 2 and "[inc2.v3fam_data] ERROR" in p2.stderr,
          (p.returncode, p.stderr[-300:], p2.returncode, p2.stderr[-300:]))


WORLD_TESTS = ("w_copies", "w_build", "w_prior", "w_etrees", "w_lists", "w_real", "w_cli")


def main():
    t0 = time.time()
    try:
        WORLD["world"] = W.build_world()
        WORLD["fixture"] = build_fixture()
        for name in UNITS + WORLD_TESTS:
            globals()[name]()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
