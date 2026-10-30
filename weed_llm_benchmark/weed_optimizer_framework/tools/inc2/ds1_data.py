"""DS1's data builds (lever L30): the weed-level pools, the source screen and its sheets' bundles, MD v1 and MT-W v2,
the leakage guard on every manifest, and the manifests of every arm (docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-29): DS1, scaling verified training data at the weed level (pre-registered)", DS.0, DS.5, DS.9.1;
docs/prereg/DS1_2026-10-29.md).

    python -m weed_optimizer_framework.tools.inc2.ds1_data build --phase 1|2 --step screen|md_s0|pools|draw|manifests
                                                                  [--procs N]
    python -m weed_optimizer_framework.tools.inc2.ds1_data rehearse --phase 1|2 [--out FILE]   (read-only, DS.9.5)
    python -m weed_optimizer_framework.tools.inc2.ds1_data status                               (read-only)

Every output lives under INC_DIR/splits/ds1/ and INC_DIR/ds1/v1/ (nothing under splits/v1, v2, v3, v3fam, _zoo or
twostage is written; LOCK v2's never-train index is never changed: intake fails closed when it does not hash as
recorded). Each step is idempotent: a record written once is read back, never rewritten; summary.json is the index
the steps rewrite (atomically) and the snapshot ships.

The rows. A candidate is an image of a source new to DS1 (never in base B by key or original bytes) with its
original file and its intake label (INC class ids per box, the original boxes: a masked admission's original, never
its masked copy), from Step 1's stream (inc2.step1_stream): the queue's admitted rows (whole or masked; kind other
too: DS1 is at the weed level) and the rows a batch's admission did not admit at species level (admission.jsonl,
no refusal), their files from the intake batch's manifest or the v1 pool (b0000). Pools (DS.0 item 2, Q's names):
p1 SIU's video frames; p3 the class-name failures; p5 the species-gate rejects and the eligible rows the cutter
cannot take; intake the rows lane DATA's intake commits before the phase-1 freeze; phase 2 the named sources.

The rules, in order, each counted with its reason code (DS.5.1):
  1 eligibility: role C (cwd12, 3SeasonWeedDet10 and 2SeasonWeedDet8 copies: the D28 sources, cwp10, vanpe, the
    domain config's never-fetch and never-train sources) out; any copy of a cwd12 dev or test image at the
    calibrated copy threshold quarantines its source whole (D28-v2): its intake's and Step 1's refusals as copies
    and DS1's guard on its rows, a dHash hit counting where the DINOv2 detector confirms it (fail closed without
    it); a source a stream quarantines whole is out, except SIU under its D31 quarantine (DS.0 item 14, DS1
    deviation 2: its rows held out before are screened again); a source whose file names carry cwd12 or
    3SeasonWeedDet10 capture names is role C, whatever its sheet says, the sheet showing the pairs; evaluation
    groups by name; off-goal (role X); a queue row holding funnel_F9, join_conflict or h6_scan (the queue's holds,
    events folded; a row the queue does not hold, its intake record's: DS1 deviation 2), and a row refused at intake
    or admission (a copy, an exam, a hold-out, a licence, EXIF) out; a row only not admitted at species level stays;
    phase 2 reads only the named sources; the sources screened before DS1 deviation 2 with rows it reads otherwise
    are screened again once (RESCREEN);
  2 names to roles: each box's class unit (SIU: its video's EPPO code; else its source class, traced from the
    source's own label files where its INC class stands for several, Revision 1 item 2; else its INC class id) has
    the role its source's decided sheet gives it: weed (class 12), crop, plant part or non-plant (dropped as
    background), or unresolved (the image is held, never admitted); species identity is kept per box;
  3 base3's box and image rules (CLOOP:2912-2913): the source rule, boxes under 8 px masked, images dropped; each
    row's image size read from its header where its file record gave none (Step 1's v1 pool: DS1 deviation 2, whose
    superseded pool records are moved aside and pooled again), a row without one refused;
  4 the model audit (E1-A) per source: a source over train_core's missing-label rate by more than 0.15 enters only
    as 'accept for training only' with predicted unlabelled plants masked (inc2.mask); each labelled box read as
    (cx, cy, w, h), its class stripped (Revision 2, item 4); a screen revision audited before that is taken again by
    the pools step first (reaudit: screen/p<ph>_r<rev>_audit_v2.json and ds1/v1/sheets/audit_v2/, written once);
  5 dedupe (base3's copy rule; never at 4-6 bits): the same bytes; 3 bits plus an 80 % box match; an identical
    label layout; base B rows win over copies;
  6 thin(10) per capture series (an SIU video, an MFWD tray): frames in the series' seeded order
    (stable_int("inc2/ds1/thin/<series>")) kept when their dHash is more than 10 bits from every frame kept,
    S0's frames of the series counted as kept (KD is not thinned);
  7 the leakage guard, both directions, on every manifest including B'o: the 8-variant 6-bit dHash and the
    calibrated DINOv2 detector (cosine 0.946384) against cwd12 dev and test, ImageWeeds, ood22, ood23, test v1 and
    its companions, evalgroups_v1, MD v1, MT-W v2 and, once F9 wrote it, the H10d hold-out; and every manifest's
    rows against MD and MT-W at the capture-group level; a hit is dropped and counted, in B'o too;
  8 EXIF orientation 2-8 refused (exif_orientation), after the guard;
  9 licence: research-only allowed and flagged; the permissive-only subset counted per manifest;
  10 writes: a PNG only when a mask fill changes the pixels; every other row by its original path.

The exams (DS.5.4): capture groups join images within 6 bits under the 8 variants (both directions), copy edges,
the same export stem, the same intake capture group, or a calibrated DINOv2 link; a group is named by the sha256 of
its members' sorted sha256s; per source, groups in the order stable_int("inc2/ds1/md_v1/<source>/<digest>"): MD
until about 5 % of the source's admitted images (15 to 500), then, for new sources, MT-W v2 until about 10 % more
(30 to 1,000), both together at most 30 % of the source; never SIU, an evaluation group, a cwd12 copy, a test v1
group or companion, or a group of more than 500 images; never a group near a built manifest's row (6-bit dHash or
the DINOv2 detector: the guard's direction to a manifest built before the exam, Revision 1 item 1); MT-W v2 also
obeys MT.2 item 2 (no group near base B or a stream pool). MD-S0 (base B's external
sources, base_v2 and SIU excluded) is drawn, sized by the power pre-flight (enlarged in the same order while its
MDE is over 0.02, within the 30 % cap) and frozen before B'o is built; each new source's MD-N is frozen at its draw,
its MT-W v2 after the person's check. Every exam row and group enters splits/ds1/never_train.jsonl at its draw.

The manifests (DS.5.3): S0 = B'o (D1's B' minus MD-S0's groups, base_v2's 66 NDSU rows and the guard's hits); S1c
(S0 and the phase-1 pools, the SIU family capped at 35 % of the rows before repeats, whole thinned videos dropped in
the order stable_int("inc2/ds1/cap/siu/<video>")); S1c-R (S1c's composition at S0's size, train_core kept whole,
order stable_int("inc2/ds1/r/<key>")); S1u (no cap); S2 (S1c and phase 2, every family capped at 35 %); KD, KT and
KF (base A plus N_K = min(15,000, each K pool's supply) rows, draws stable_int("inc2/ds1/k/<arm>/<series or row>")).
The cwd12 exposure floor: each train_core row listed r = round_half_up((N - C) / (N_S0 - C)) times in every S arm
larger than S0 (repeat keys <key>__ds1rep<j>, the same files), N counted before repeats. Each larger S manifest holds
every row of each smaller one, bytes unchanged. Every manifest record prints images, distinct photos (6-bit
components) and capture groups, and the gap to the owner's scale rule.

A refusal raises DataRefused (an inc2.ds1.Ds1Refused); the CLI prints "[inc2.ds1_data] ERROR: ..." and exits 2.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from ..inc import common as C
from . import base3 as B3
from . import common as C2
from . import ds1 as D

# ------------------------------------------------------------------ constants (pre-registered)
FORMAT = "inc2-ds1-data/1"
FMT_ARM = "inc2-ds1-arm/1"
FMT_SCREEN = "inc2-ds1-screen/1"
FMT_REAUDIT = "inc2-ds1-reaudit/1"
FMT_POOL = "inc2-ds1-pool/1"
FMT_EXAM = "inc2-ds1-exam/1"
FMT_LOCK = "inc2-ds1-exam-lock/1"
STEPS = ("screen", "md_s0", "pools", "draw", "manifests")
PHASES = (1, 2)
SUMMARY = "summary.json"
NEVER_TRAIN = "never_train.jsonl"
CLASS_WEED = 12                                     # every weed box (E1's class 12, DS.0 item 1)
ROLES = ("weed", "crop", "part", "nonplant", "unresolved")
DROP_ROLES = ("crop", "part", "nonplant")
DECISIONS = ("accept", "accept_training_only", "reject")
SETTINGS = ("Field", "Turf", "Indoor", "Aerial")
# sources (DS.0 item 2, DS.5.1 item 1)
SIU = "zenodo_15808623"
SIU_FAMILY = "siu"
SIU_VIDEO_RE = re.compile(r"(?P<eppo>[A-Z]{5})_week_(?P<week>\d+)_IMG_(?P<img>\d+)_frame_")
SIU_TARGET_EPPO = ("AMAPA", "AMATA", "SIDSP")       # Palmer amaranth, waterhemp, prickly sida: KD/KT's T
CROP_EPPO_SUGGEST = ("SORVU", "ZEAMX", "GLXMA", "TRZAX", "HORVU", "BEAVA", "GOSHI", "ORYSA")
MFWD_SOURCES = ("mediatum_1717366",)
TRAIN_CORE = "cottonweeddet12/train"
TSW = ("3seasonweeddet10/data2022", "3seasonweeddet10/data2023")
SEASON_SESSION = "20220129_CanonEOS4000D_EO"
ROLE_C = ("rf_karthikeya-c8pvy__weed-detection-cwp10", "rf_zig-zag-lnodr__weed-detection-vanpe",
          "rf_school-wevfc__weeddec2-sqnot", "rf_priyankas-workspace-o43oc__weeds1-dsqzl",
          "rf_weedtown__weedsv2-46hkh")
NEVER_FETCHED = ("zenodo_10762138",)                # 2SeasonWeedDet8 (Amendment 2026-10-28)
# phase 2 reads only the sources DS.5.2 names (DS.0 item 2, DS.9.2: "their source ids given explicitly"); a named
# source's id is listed here once lane DATA's fetch names it (a recorded code change, deployed before its screen)
PHASE2_SOURCES = {"MFWD full": ("mediatum_1717366",), "CropAndWeed": (), "PhenoBench": (), "Sugar Beets 2016": ()}
EXCLUDED_HOLDS = ("funnel_F9", "join_conflict", "h6_scan")
# the refusals that name a copy of an evaluation image (intake's and Step 1's, masked or not): D28-v2's input
EVAL_COPY_REASONS = ("near_eval_v2", "near_eval_variant", "near_eval_embed")
DHASH_COPY_REASONS = ("near_eval_v2", "near_eval_variant")
CWD12_SPLITS = ("dev", "test")
NOT_ADMITTED_OK = ("not_admitted",)                 # admission outcomes that leave a weed-level candidate
# the capture names of cwd12 and 3SeasonWeedDet10 (date, camera and initials stems: iPhoneSE_YL, NIKOND3300_YL,
# iPhone11Pro_BT, CanonEOS4000D_EO, ...), DS.5.1 item 1(b)
CAPTURE_NAME_RE = re.compile(r"(?i)(?:^|[^a-z0-9])(?:\d{8}_)?(iphone ?(?:se|x[rs]?|\d{1,2}(?:pro(?:max)?)?)|"
                             r"nikon ?d\d{3,4}|canon ?eos ?\w{2,6}d?|htrc_iphone\d{1,2})[_ ]([a-z]{2})(?=[^a-z]|$)")
# the (camera, initials) stems cwd12 and 3SeasonWeedDet10 name their photographs with (read on 2026-10-29 from LOCK
# v2's dev, test, train_core, tsw22, tsw23 and base_v2 keys): a file name is a capture name only when its stem is
# one of these (other sets of the same laboratory, CottonWeedDet3's iPhone7_EB, are not cwd12's)
CAPTURE_STEMS = frozenset([("iphonese", "yl"), ("iphone11pro", "bt"), ("nikond3300", "yl"), ("nikond330", "yl"),
                           ("canoneos4000d", "eo"), ("canoneos4000d", "yl"), ("htrc_iphone12", "wy"),
                           ("htrc_iphone13", "bd"), ("iphone12", "wy"), ("iphone13", "bd")])
# rule numbers (base3_v2.json's, read at build; the copy rule never at 4-6 bits)
THIN_BITS = 10
COPY_BITS = 3
GROUP_BITS = 6
GUARD_BITS = 6
COPY_COS = 0.946384
PARTIAL_MISSING_EXCESS = 0.15
AUDIT_SAMPLE = 200
AUDIT_CONF = 0.5
AUDIT_IOU = 0.5
# Revision 2, item 4: the model audit reads each labelled box as (cx, cy, w, h), the class stripped, as train_core's
# reference always did; a screen's audit taken before it (no "version") read [class, cx, cy, w] as the box and is
# taken again (reaudit) before any pool applies it
AUDIT_VERSION = 2
# The audit's predictor, E1-A seed 0 (e1a_predictor: INC/e1_a_m640/runs/base__s0/weights/final.pt, the file the screen
# p1_r001 read; cited by sha256 in Revision 2's stamp file): the audit taken again refuses any other weights
AUDIT_PREDICTOR_SHA256 = "a8e1b61ad29806c11228b9a54a463e40ac466fb3a717a17fcb2d6fc620e4cfcb"
# DS1 deviation 2 (2026-10-30), item 2: DS.0 item 14 lets DS1's builds read SIU past its stream quarantine, which cites
# D31 (data blamed: redundant frames), once its sheet and species-role table are decided and audited (source_ready);
# never past a D28 (leak) quarantine, and never when the stream's cites cannot be read (fail closed). The screen's rows
# of SIU that a revision held out by that quarantine are screened again (rescreen_due), and the new revision's record
# of SIU replaces the earlier one.
SIU_STREAM_CITES = ("D31",)
QUARANTINED = "quarantined_d28"                     # the reason code of a source a stream quarantines whole
# DS1 deviation 2, item 2: the phase-1 sources screened before it with rows its code reads otherwise, screened again
# whole once (rescreen_due): SIU, held out whole by its D31 quarantine, and the sources whose queued rows the screen
# read with the intake record's h6_scan hold, which step1_stream's serve-holds had released (read on 2026-10-30:
# SIU 123,348 rows, mendeley_9yhd5b8kc3 6,604, kg_yuzhenlu__cottonweeddet3 755, zenodo_22033986 352, weedai_7e03cd97
# 8; candidate_row now takes a queued row's holds from the queue)
RESCREEN = (SIU, "mendeley_9yhd5b8kc3", "kg_yuzhenlu__cottonweeddet3", "zenodo_22033986",
            "weedai_7e03cd97-f462-439d-b4c7-b6289edcc21b")
SCREEN_DEVIATION = 2                                # a screen revision's "deviation": the last whose rules it applied
# DS1 deviation 2, item 1: the phase-1 pool records L30's pools step (job 50028368) wrote at 2026-10-30T05:38Z while
# every Step 1 v1-pool row (step1/pool.jsonl) had no image size (candidate_row read it from the file record only):
# base3's source rule skips a row without one, so each source showed "no weed box" and every row dropped as
# source_convention. Each is moved aside byte for byte (its record and its rows, which must hash as named) into
# splits/ds1/pools/p<phase>/superseded/dev2/ (supersede_pools), and the source is pooled again. A record at the path
# with other bytes is not the one named and stays.
# DS1 deviation 2's record (docs/CONTINUOUS_LOOP.md, "Deviation (2026-10-30): DS1 deviation 2"), kept byte for byte
# under docs/prereg/ and time-stamped as deviation 1 was; the ticker's gate pins it by sha256 (the domain's
# ds1.deviations). It is pinned here and not in inc2.ds1.DEVIATIONS: the reproduction records pin inc2/ds1.py's bytes
# (inc2.ds1.code_shas), so a change of that file would refuse every later score until they are taken again.
DEVIATION_2 = {"n": 2, "file": "docs/prereg/DS1_dev2_2026-10-30.md",
               "sha256": "b825aa66198553a7bae49efb6f0afdfa77ce46b71226e3791ad67b467ea3ed4d",
               "commit": "dbd6bea02e7d0b0d415b598d608d5bcecd6d056d",
               "stamp_file": "docs/prereg/DS1_dev2_2026-10-30.stamp.txt"}
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_DEV2_POOL_WHY = ("written 2026-10-30T05:38Z by job 50028368 while every row lacked its image size, so base3's source "
                  "rule saw no weed box and dropped all %d live rows as source_convention; pooled again under DS1 "
                  "deviation 2")
SUPERSEDED_POOLS = tuple({"deviation": 2, "phase": 1, "source": s, "record_sha256": h, "rows_sha256": EMPTY_SHA256,
                          "why": _DEV2_POOL_WHY % n} for s, h, n in (
    ("project_agml__mh_weed16_weed_detection", "c1b0e2c5f63d97960e6a477596ee8b923264b9b3b815a3c2f70195da12fa713c", 10),
    ("rf_tuf__weed-3434e", "43482abf523851690e9d36d8ad8089c96485138a20a48ec3e2e20e0f12661ecb", 1019),
    ("rf_vizerion__weed-detection-5ell6", "c90d64f128003c30cba2af70c7625e0e589914bf1d9aa9b178d752565a3844e6", 385),
    ("rf_weed-detection-o9sfb__weed-detection-in-a-field",
     "f76ee83fb74ebc9abda7ba97cd6838e12a1c00617a113bf89ffd4ec8fc9f9a90", 1229),
    ("rf_yolo-gyfyq__weed-ieda7", "1ac035e0ae4a98628488ec7f92c9be710786f3e1bd624b9d0a0ca1e421eb11b1", 1825),
    ("rf_zijian-peng__grass-weed-69hcm", "72bdb39ea7e0bc78c24f670edecfcf1de1b046fe5882a83e7f5ae1c4e06883b2", 1)))
FAMILY_CAP = 0.35
MD_SHARE, MD_MIN, MD_MAX = 0.05, 15, 500
MTW_SHARE, MTW_MIN, MTW_MAX = 0.10, 30, 1000
EXAM_SHARE_MAX = 0.30
GROUP_MAX = 500
N_K_MAX = 15000
EXIF_OK = (None, 0, 1)
SCALE_RULE = 10000                                  # the owner's scale rule: tens to hundreds of thousands
# seed texts
SEED_THIN = "inc2/ds1/thin/%s"
SEED_CAP = "inc2/ds1/cap/%s/%s"
SEED_R = "inc2/ds1/r/%s"
SEED_K = "inc2/ds1/k/%s/%s"
SEED_MD = "inc2/ds1/md_v1/%s/%s"
SEED_SAMPLE = "inc2/ds1/sheet/%s/%s"
# the arms built here (inc2.ds1.BUILD_ARMS) and what each holds
S_NEST = (("S0", "S1C"), ("S1C", "S1U"), ("S1C", "S2"))
FLOOR_ARMS = ("S1C", "S1CR", "S1U", "S2")


class DataRefused(D.Ds1Refused):
    pass


def deviations():
    """DS1's recorded deviations in order: inc2.ds1.DEVIATIONS (deviation 1), then deviation 2 (DEVIATION_2)."""
    return tuple(D.DEVIATIONS) + (DEVIATION_2,)


def phase2_ids():
    """The source ids phase 2 may read (PHASE2_SOURCES)."""
    return {s for ids in PHASE2_SOURCES.values() for s in ids}


def log(msg):
    print("[inc2.ds1_data] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths and small helpers
def out_dir():
    return D.splits_dir()


def summary_path():
    return out_dir() / SUMMARY


def never_train_path():
    return out_dir() / NEVER_TRAIN


def arm_path(arm):
    return out_dir() / "arms" / ("%s.jsonl" % arm.lower())


def arm_record_path(arm):
    return out_dir() / "arms" / ("%s.json" % arm.lower())


def exam_dir(exam):
    return out_dir() / ("mtw_v2" if exam.startswith("mtw_") else "md_v1") / exam


def slug(source):
    s = re.sub(r"[^a-z0-9]+", "_", str(source).lower()).strip("_")
    return (s[:40] + "_" + hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]) if len(s) > 48 else s


def mdn_exam(source):
    return "mdn_%s" % slug(source)


def mtw_exam(source):
    return "mtw_%s" % slug(source)


def stable(text):
    return C.stable_int(text)


def _read_json(path):
    return D._read_json(path)


def write_once(path, data):
    try:
        return D.write_once(path, data)
    except D.Ds1Refused as e:
        raise DataRefused(str(e))


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    with open(p) as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl_once(path, rows):
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    return write_once(path, data) and C.sha256_file(path)


def manifest_line(row):
    return (json.dumps({k: row[k] for k in C.MANIFEST_KEYS}, sort_keys=True) + "\n").encode("utf-8")


def list_sha256(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def label_text(boxes):
    return "".join("%d %.6f %.6f %.6f %.6f\n" % (CLASS_WEED, b[0], b[1], b[2], b[3]) for b in boxes)


def write_label(boxes, root=None):
    """(path, sha256) of a class-12 label, content-addressed under splits/ds1/labels/, written once."""
    text = label_text(boxes)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    p = Path(root or out_dir()) / "labels" / sha[:2] / ("%s.txt" % sha)
    if not p.is_file():
        C2.write_bytes_atomic(p, text.encode("utf-8"))
    elif C.sha256_file(p) != sha:
        raise DataRefused("%s exists with other bytes" % p)
    return str(p), sha


def read_inc_label(path):
    """[(cls, cx, cy, w, h)] of an INC label file (a polygon line becomes its bounding box, clipped to [0, 1])."""
    out = []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            c = int(float(t[0]))
            v = [float(x) for x in t[1:]]
            if len(v) == 4:
                cx, cy, w, h = v
            elif len(v) >= 6 and len(v) % 2 == 0:
                xs, ys = v[0::2], v[1::2]
                x0, x1, y0, y1 = max(0.0, min(xs)), min(1.0, max(xs)), max(0.0, min(ys)), min(1.0, max(ys))
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            else:
                raise DataRefused("%s: a label line of %d values" % (path, len(v)))
            x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
            y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
            out.append((c, (x0 + x1) / 2, (y0 + y1) / 2, max(0.0, x1 - x0), max(0.0, y1 - y0)))
    return out


def load_rules():
    """base3_v2.json's rule numbers (the source, box, image and dedupe rules DS.5.1 items 3 and 5 take)."""
    conf = json.loads(B3.CONFIG.read_text())
    return conf["rules"], conf


def three_counts(rows):
    """{images, distinct_photos (6-bit components), capture_groups} of a row set (DS.5.2): every status line prints
    them, with the gap to the owner's scale rule."""
    n = len(rows)
    if not n:
        return {"images": 0, "distinct_photos": 0, "capture_groups": 0, "scale_rule": SCALE_RULE}
    hashed = [r for r in rows if r.get("dhash") is not None]
    comps = len(set(B3.hash_components([_hashrow(r) for r in hashed], GROUP_BITS).tolist())) if hashed else 0
    comps += n - len(hashed)
    groups = len({r.get("group") or r["key"] for r in rows})
    return {"images": n, "distinct_photos": comps, "capture_groups": groups, "scale_rule": SCALE_RULE,
            "gap_to_scale_rule": max(0, SCALE_RULE - n)}


def _hashrow(r):
    return {"dhash": r.get("dhash"), "variants": r.get("variants")}


# ------------------------------------------------------------------ 1. eligibility (DS.5.1 item 1, DS.0 item 2)
def capture_name_hits(names):
    """[(name, matched stem)] of file names carrying a cwd12 or 3SeasonWeedDet10 capture name (camera and
    initials stems: iPhoneSE_YL, NIKOND3300_YL, iPhone11Pro_BT, ...): DS.5.1 item 1(b)."""
    out = []
    for n in names:
        for m in CAPTURE_NAME_RE.finditer(str(n)):
            if (re.sub(r"\s+", "", m.group(1)).lower(), m.group(2).lower()) in CAPTURE_STEMS:
                out.append((str(n), "%s_%s" % (m.group(1), m.group(2))))
                break
    return out


def siu_scoped(source, ctx):
    """True for SIU when every stream quarantine of it cites D31 (DS.0 item 14, DS1 deviation 2 item 2): DS1 reads it
    past that quarantine, its sheet and species-role table decided and audited before any of its rows enters a
    manifest (source_ready). False for every other source, for SIU quarantined under D28, and when the cites are not
    known (fail closed). ctx: quarantine_cites {source: [cites]}."""
    if source != SIU:
        return False
    cites = (ctx.get("quarantine_cites") or {}).get(source)
    return bool(cites) and set(cites) <= set(SIU_STREAM_CITES)


def eligibility(row, ctx):
    """The first reason a candidate row is out (DS.5.1 item 1, DS.0 item 2), or None. ctx: role_c (sources),
    quarantined (sources a stream quarantines whole) with quarantine_cites (their cites: SIU's D31 quarantine is
    read past, DS.0 item 14), eval_sources (evaluation groups by name), never (the domain config's never-fetch and
    never-train sources), base_keys and base_shas (base B's rows)."""
    s = row["source"]
    if s in ctx.get("role_c", ()) or s in ROLE_C or s in NEVER_FETCHED:
        return "role_c"
    if s in ctx.get("never", ()):
        return "never_train_source"
    if s in ctx.get("quarantined", ()) and not siu_scoped(s, ctx):
        return QUARANTINED
    if s in ctx.get("eval_sources", ()):
        return "evaluation_group"
    if row.get("refused"):
        return "refused:%s" % row["refused"]
    held = [h for h in row.get("holds") or () if h in EXCLUDED_HOLDS]
    if held:
        return "held:%s" % held[0]
    if row.get("test_v1"):
        return "test_v1"
    if row.get("admission") not in (None, "whole", "masked") + NOT_ADMITTED_OK:
        return "admission:%s" % row.get("admission")
    if row["key"] in ctx.get("base_keys", ()) or row.get("sha256") in ctx.get("base_shas", ()):
        return "in_base_b"
    if row["key"] in ctx.get("consumed", ()):
        return "stream_consumed"
    return None


# ------------------------------------------------------------------ 2. names to roles (DS.5.1 item 2)
def siu_video(text):
    """(video, EPPO code) of an SIU frame's original name or key, or (None, None)."""
    m = SIU_VIDEO_RE.search(str(text or ""))
    if not m:
        return None, None
    return "%s_week_%s_IMG_%s" % (m.group("eppo"), m.group("week"), m.group("img")), m.group("eppo")


def unit_of(row, cls, i=None):
    """The class unit a box's role is decided on (DS.5.1 item 2: each source class one role): SIU's video's EPPO code
    (per EPPO code, from its own metadata); else the box's source class ('src:<id>', from the source's own label files:
    row['box_src'][i], the class of box i) where it was traced; else the source's INC class id ('inc:<id>'), which is
    one source class unless several map to it (the sheet lists every name a unit stands for)."""
    if row["source"] == SIU:
        _v, eppo = siu_video(row.get("rel") or row["key"])
        return "eppo:%s" % (eppo or "?")
    bs = row.get("box_src")
    if bs is not None and i is not None and 0 <= int(i) < len(bs) and bs[int(i)] is not None:
        return "src:%s" % bs[int(i)]
    return "inc:%d" % int(cls)


def suggest_role(source, unit, inc_of=None):
    """The sheet's pre-filled suggestion for a unit (never a decision): the twelve target species are weeds; an
    SIU crop code is a crop; OtherPlant and the unmapped id are unresolved until the sheet decides. A source class
    ('src:<id>') is suggested as the INC class it maps to (inc_of {src id: INC id})."""
    if unit.startswith("eppo:"):
        return "crop" if unit[5:] in CROP_EPPO_SUGGEST else "weed"
    if unit.startswith("src:"):
        i = (inc_of or {}).get(unit[4:])
        return "weed" if i is not None and 0 <= int(i) < 12 else "unresolved"
    i = int(unit[4:])
    if 0 <= i < 12:
        return "weed"
    return "unresolved"


def weed_boxes(row, roles):
    """(boxes [(cx, cy, w, h)] of class 12, species [unit per kept box], dropped {role: n}, unresolved units) of a
    candidate under its source's decided roles."""
    keep, species, dropped, unres = [], [], collections.Counter(), set()
    for i, b in enumerate(row["boxes"]):
        u = unit_of(row, b[0], i)
        role = (roles or {}).get(u, "unresolved")
        if role == "weed":
            keep.append(tuple(float(x) for x in b[1:5]))
            species.append(u)
        elif role in DROP_ROLES:
            dropped[role] += 1
        else:
            unres.add(u)
    return keep, species, dict(dropped), sorted(unres)


# ------------------------------------------------------------------ 3. box and image rules (base3's)
def image_size(path):
    """(W, H) of an image as its header stores it (base3's reader for external rows), or None when unreadable."""
    return B3._header_size(path)


def fill_sizes(rows, read_size=None, procs=5):
    """Every live row's image size where its file record gave none, read from the image's header (DS1 deviation 2,
    item 1): Step 1's v1-pool rows (step1/pool.jsonl) carry no size, and base3's source rule skips a row without one,
    so a source whose rows all lack it showed no weed box and dropped whole. Sets W, H and size_from 'header'; a file
    whose header cannot be read is dropped 'unreadable' (base3's image rule's reason). Returns the rows read."""
    from concurrent.futures import ThreadPoolExecutor
    read_size = read_size or image_size
    todo = [r for r in rows if not r.get("drop") and (not r.get("W") or not r.get("H"))]

    def one(r):
        return r, read_size(r["image"])
    with ThreadPoolExecutor(max_workers=max(1, int(procs))) as ex:
        for r, wh in ex.map(one, todo):
            if not wh or not wh[0] or not wh[1]:
                r["drop"] = "unreadable"
                continue
            r["W"], r["H"], r["size_from"] = int(wh[0]), int(wh[1]), "header"
    return len(todo)


def apply_rules(rows, rules):
    """base3's source rule (over each source's weed boxes as delivered) and its per-box and per-image rules; sets
    row['drop'], row['mask'] and row['masked_area']. Returns {source: convention record}. Refuses a live row without
    its image size (DS1 deviation 2, item 1: base3's source rule would skip it unseen; fill_sizes reads it first)."""
    nosize = [r["key"] for r in rows if not r.get("drop") and (not r.get("W") or not r.get("H"))]
    if nosize:
        raise DataRefused("%d live row(s) have no image size (the first: %s): base3's source rule would skip them, and "
                          "a source whose rows all lack one would show no weed box and drop whole; read their sizes "
                          "first (fill_sizes)" % (len(nosize), nosize[0]))
    by = collections.defaultdict(list)
    for r in rows:
        if not r.get("drop"):
            by[r["source"]].append(r)
    conv = {}
    for s, rs in sorted(by.items()):
        conv[s] = B3.source_convention(rs, rules["source"])
        if not conv[s]["passes"]:
            for r in rs:
                r["drop"] = "source_convention"
    for r in rows:
        if r.get("drop"):
            continue
        r["kind"] = "ds1"
        r.setdefault("mask", [])
        r.setdefault("masked_area", 0.0)
        B3.image_rules(r, rules)
    return conv


# ------------------------------------------------------------------ 4. the model audit (DS.5.1 item 4)
def _iou_xywh(a, b):
    return B3.box_iou(a, b)


def audit_boxes(row):
    """A candidate's labelled boxes as the audit reads them: (cx, cy, w, h), its class stripped (Revision 2, item 4;
    train_core's reference reads b[1:5] of its labels)."""
    return [tuple(float(x) for x in b[1:5]) for b in row.get("boxes") or ()]


def audit_rates(rows, predict, conf=AUDIT_CONF, iou=AUDIT_IOU):
    """{images, predictions, missing_rate, false_box_rate, convention} of a detector's predictions on labelled
    rows: missing = confident predictions (conf >= 0.5) matching no labelled box at IoU 0.5 (unlabelled plants),
    false boxes = labelled boxes no prediction (conf >= 0.25) matches; convention = the median of log(predicted
    area / labelled area) over matched pairs. predict(row) -> [(cx, cy, w, h, conf)]; each row's boxes are
    (cx, cy, w, h) (audit_boxes), and a box of another length is refused (Revision 2, item 4)."""
    n_pred = n_miss = n_gt = n_false = 0
    logs = []
    for r in rows:
        gt = list(r.get("boxes") or ())
        if any(len(g) != 4 for g in gt):
            raise DataRefused("the model audit reads labelled boxes as (cx, cy, w, h); row %s holds a box of %d "
                              "numbers (strip the class: audit_boxes)" % (r.get("key"), len(next(
                                  g for g in gt if len(g) != 4))))
        pr = [p for p in predict(r)]
        conf_p = [p for p in pr if p[4] >= conf]
        n_pred += len(conf_p)
        for p in conf_p:
            best = max([_iou_xywh(p[:4], g) for g in gt] or [0.0])
            if best < iou:
                n_miss += 1
        n_gt += len(gt)
        for g in gt:
            ious = [(_iou_xywh(p[:4], g), p) for p in pr if p[4] >= 0.25]
            if not ious or max(ious)[0] < iou:
                n_false += 1
            else:
                p = max(ious)[1]
                if g[2] * g[3] > 0 and p[2] * p[3] > 0:
                    logs.append(math.log((p[2] * p[3]) / (g[2] * g[3])))
    logs.sort()
    return {"images": len(rows), "predictions": n_pred, "gt_boxes": n_gt,
            "missing_rate": n_miss / float(n_pred) if n_pred else 0.0,
            "false_box_rate": n_false / float(n_gt) if n_gt else 0.0,
            "convention_log_area": logs[len(logs) // 2] if logs else None}


def audit_sample(rows, source, n=AUDIT_SAMPLE):
    return sorted(rows, key=lambda r: (stable("inc2/ds1/audit/sample/%s/%s" % (source, r["key"])), r["key"]))[:n]


def audit_verdict(rates, ref):
    """Partially labelled (DS.5.1 item 4) when the source's missing-label rate exceeds train_core's by > 0.15."""
    excess = float(rates["missing_rate"]) - float(ref["missing_rate"])
    return {"partial": excess > PARTIAL_MISSING_EXCESS, "excess": excess, "threshold": PARTIAL_MISSING_EXCESS,
            "source": rates, "train_core": ref, "version": AUDIT_VERSION}


def source_audit(rows, source, predict, ref):
    """The model audit of one source's live screen rows (DS.5.1 item 4): its seeded sample, each row's boxes as
    (cx, cy, w, h) (audit_boxes), against train_core's reference rates under the same predictor."""
    smp = audit_sample(rows, source)
    return audit_verdict(audit_rates([dict(r, boxes=audit_boxes(r)) for r in smp], predict), ref)


def audit_valid(verdict):
    """True for a model audit taken as Revision 2 reads boxes (its version), False for one taken before."""
    return isinstance(verdict, dict) and int(verdict.get("version") or 1) >= AUDIT_VERSION


def unlabelled_masks(row, predict, conf=AUDIT_CONF, iou=AUDIT_IOU):
    """The predicted unlabelled plants of a partially labelled source's row (confident predictions matching no
    labelled box), masked as inc2.mask does (DS.5.1 item 4)."""
    out = []
    for p in predict(row):
        if p[4] >= conf and max([_iou_xywh(p[:4], g) for g in row["boxes"]] or [0.0]) < iou:
            out.append(tuple(float(x) for x in p[:4]))
    return out


# ------------------------------------------------------------------ 5. dedupe (base3's copy rule)
def dedupe(rows, base_rows, dd):
    """Copies among the candidates and against base B (base3's copy rule: the same bytes; 3 bits plus an 80 % box
    match under a variant; an identical label layout): base B rows win, then base3's rank (tier, pixels, source,
    key). Sets drop 'copy_of_base' or 'duplicate' and dup_of on the losers. Returns counts."""
    live = []
    for r in base_rows:
        live.append(dict(_dd_row(r), _base=True, kind=B3.BASE_KIND))
    cands = [r for r in rows if not r.get("drop")]
    for r in cands:
        live.append(dict(_dd_row(r), _base=False, kind="ds1"))
    uf, rec = B3.copy_components(live, dd)
    groups = uf.groups(range(len(live)))
    counts = collections.Counter()
    for _root, mem in groups.items():
        if len(mem) < 2:
            continue
        best = min(mem, key=lambda i: (0 if live[i]["_base"] else 1,) + B3._rank(live[i])[1:])
        for i in mem:
            if i == best or live[i]["_base"]:
                continue
            r = cands[i - len(base_rows)]
            r["drop"] = "copy_of_base" if live[best]["_base"] else "duplicate"
            r["dup_of"] = live[best]["key"]
            counts[r["drop"]] += 1
    return dict(counts, **rec)


def _dd_row(r):
    return {"key": r["key"], "source": r["source"], "sha256": r.get("sha256"), "dhash": r.get("dhash"),
            "variants": r.get("variants"), "boxes": [tuple(b[-4:]) for b in r.get("boxes") or ()],
            "W": r.get("W"), "H": r.get("H"), "bytes": r.get("bytes"), "tier": r.get("tier", 3)}


# ------------------------------------------------------------------ 6. thin(10) (DS.0 item 3)
def series_order(series, keys):
    """The seeded order of a series' frames: numpy's default_rng(stable_int("inc2/ds1/thin/<series>")) permutation
    of its keys sorted."""
    import numpy as np
    ks = sorted(keys)
    perm = np.random.default_rng(stable(SEED_THIN % series)).permutation(len(ks))
    return [ks[i] for i in perm]


def thin(frames, series, seeded_hashes=(), bits=THIN_BITS):
    """thin(10) on one capture series: frames [{key, dhash}] taken in the series' seeded order; a frame is kept
    when its dHash is more than `bits` bits from every frame already kept, the series' frames S0 already holds
    (seeded_hashes) counted as kept. Returns (kept keys in order, dropped {key: nearest kept distance})."""
    kept_h = [int(h) for h in seeded_hashes if h is not None]
    by = {f["key"]: f for f in frames}
    kept, dropped = [], {}
    for k in series_order(series, list(by)):
        h = by[k].get("dhash")
        if h is None:
            dropped[k] = None
            continue
        h = int(h)
        d = min([bin(h ^ x).count("1") for x in kept_h] or [65])
        if d > bits:
            kept.append(k)
            kept_h.append(h)
        else:
            dropped[k] = d
    return kept, dropped


def thin_rows(rows, s0_by_series, bits=THIN_BITS):
    """thin(10) on every series of the rows (rows without a series pass): sets row['thin'] True (kept) or False.
    s0_by_series {series: [dHash of S0's frames]}. Returns {series: (kept, dropped)} counts."""
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("drop"):
            continue
        if r.get("series"):
            by[r["series"]].append(r)
        else:
            r["thin"] = True
    out = {}
    for s, rs in sorted(by.items()):
        kept, dropped = thin([{"key": r["key"], "dhash": r.get("dhash")} for r in rs], s, s0_by_series.get(s, ()),
                             bits)
        ks = set(kept)
        for r in rs:
            r["thin"] = r["key"] in ks
        out[s] = {"frames": len(rs), "kept": len(kept), "dropped": len(dropped)}
    return out


# ------------------------------------------------------------------ hashes (the original files)
def hash_rows(rows, procs=5, hasher=None):
    """Every row's dHash and its 8 variants (inc2.guard.image_hashes on its original file) where they are not
    stored; a file that cannot be hashed is dropped ('unhashable': a row the guard cannot compare is one it cannot
    clear)."""
    from concurrent.futures import ThreadPoolExecutor
    if hasher is None:
        from . import guard as G
        hasher = G.image_hashes
    todo = [r for r in rows if not r.get("drop") and r.get("variants") is None]

    def one(r):
        return r, hasher(r["image"])
    with ThreadPoolExecutor(max_workers=max(1, int(procs))) as ex:
        for r, (h, v) in ex.map(one, todo):
            if h is None or v is None:
                r["drop"] = "unhashable"
                continue
            r["dhash"] = int(h)
            r["variants"] = [int(v[k]) for k in B3.VARIANTS] if isinstance(v, dict) else [int(x) for x in v]
    return len(todo)


# ------------------------------------------------------------------ 7. the leakage guard (DS.5.1 item 7)
def dhash_hits(rows, exams, bits=GUARD_BITS):
    """{key: [{exam, exam_key, bits, how}]}: each row within `bits` of an exam image under the 8 variants in both
    directions (inc2.base3's pair search, on stored hashes). exams {name: {"keys", "H", "ok"}}."""
    import numpy as np
    hits = collections.defaultdict(list)
    rs = [r for r in rows if r.get("variants") is not None or r.get("dhash") is not None]
    if not rs:
        return hits
    HA, okA, _has = B3.hash_matrix(rs)
    for name, ex in sorted(exams.items()):
        if not len(ex["keys"]):
            continue
        Q1, T1, D1_, _v = B3.near_pairs_between(HA, okA, ex["H"], int(bits))
        T2, Q2, D2_, _w = B3.near_pairs_between(ex["H"], ex["ok"], HA, int(bits))
        best = {}
        for a, e, d in zip(np.r_[Q1, Q2].tolist(), np.r_[T1, T2].tolist(), np.r_[D1_, D2_].tolist()):
            k = (int(a), int(e))
            if k not in best or d < best[k]:
                best[k] = int(d)
        for (a, e), d in sorted(best.items()):
            hits[rs[a]["key"]].append({"exam": name, "exam_key": ex["keys"][e], "bits": d, "how": "dhash"})
    return hits


_DESC_MEMO = {}                 # {embedder name: {(image, sha256): (descriptor, 8 hashes) or None}} within one job


def _memo_scan(items, scanner, desc_path=None, procs=1):
    """EmbedScanner.scan with the job's descriptors described once: every DS1 scanner uses LOCK v2's calibrated
    embedder, so an image described for one index is not described again for the next (the same descriptor and
    hashes the scanner would compute; funnel.leak.detect under the scanner's own calibration)."""
    import numpy as np
    from ..funnel import embed as E
    from ..funnel import leak as L
    emb = scanner.index.embedder
    memo = _DESC_MEMO.setdefault(str(getattr(emb, "name", id(emb))), {})
    need, seen = [], set()
    for it in items:
        k = (str(it["image"]), it.get("sha256"))
        if k not in memo and k not in seen:
            seen.add(k)
            need.append({"key": "m%d" % len(need), "image": str(it["image"]), "sha256": it.get("sha256"), "_k": k})
    if need:
        res = E.embed_images([{x: r[x] for x in ("key", "image", "sha256")} for r in need], desc_path, emb,
                             prepare=L.prepare_hashed, n_hashes=len(L.VARIANTS), procs=procs, batch=32)
        X = np.asarray(res["X"], dtype=np.float32)
        H = np.asarray(res["H"], dtype=np.uint64)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(X).all(axis=1) & np.asarray(res["hash_ok"], dtype=bool) & \
                (np.nan_to_num(np.linalg.norm(X, axis=1)) > 0)
        for i, r in enumerate(need):
            memo[r["_k"]] = (X[i], [int(h) for h in H[i]]) if ok[i] else None
    images, unscannable = [], set()
    for it in items:
        m = memo.get((str(it["image"]), it.get("sha256")))
        if m is None:
            unscannable.add(it["key"])
            continue
        images.append({"key": it["key"], "path": str(it["image"]), "desc": m[0], "hashes": m[1]})
    found = L.detect(images, scanner.index, scanner.calibration["calibration"]) if images else []
    copies = {}
    for e in found:
        copies.setdefault(e["key"], []).append(e)
    return copies, unscannable


def embed_hits(rows, scanner, desc_path=None, procs=1):
    """{key: [{exam, exam_key, cos, how}]} of the calibrated DINOv2 detector on rows (fail closed on a file it cannot
    describe). A calibrated EmbedScanner describes each image once per job (_memo_scan); any other scanner scans."""
    hits = collections.defaultdict(list)
    if not rows or scanner is None:
        return hits
    items = [{"key": r["key"], "image": r.get("train_image") or r["image"], "sha256": r.get("sha256")} for r in rows]
    if getattr(getattr(scanner, "index", None), "embedder", None) is not None and \
            isinstance(getattr(scanner, "calibration", None), dict):
        copies, unscannable = _memo_scan(items, scanner, desc_path=desc_path, procs=procs)
    else:
        copies, unscannable = scanner.scan(items, desc_path=desc_path, procs=procs)
    if unscannable:
        raise DataRefused("%d file(s) the DINOv2 detector cannot describe (e.g. %s): fail closed"
                          % (len(unscannable), sorted(unscannable)[:3]))
    for k, es in copies.items():
        for e in es:
            hits[k].append({"exam": e.get("eval_split") or e.get("split"), "exam_key": e.get("eval_key"),
                            "cos": e.get("cos"), "how": "embed"})
    return hits


def merge_hits(*ds):
    out = collections.defaultdict(list)
    for d in ds:
        for k, v in d.items():
            out[k].extend(v)
    return out


def group_hits(rows, exam_rows):
    """{key: [{exam, group, how 'capture'}]}: rows sharing a capture relation (export stem, intake capture group,
    series) or a capture group with an MD or MT-W row (DS.5.1 item 7, last line)."""
    caps = {}
    for e in exam_rows:
        for c in e.get("capture_keys") or ():
            caps.setdefault(tuple(c), (e.get("exam"), e.get("group")))
        if e.get("group"):
            caps.setdefault(("group", e["group"]), (e.get("exam"), e.get("group")))
        for s in (e.get("sha256"), e.get("original_sha256")):
            if s:
                caps.setdefault(("sha", s), (e.get("exam"), e.get("group")))
    out = collections.defaultdict(list)
    for r in rows:
        ks = [tuple(c) for c in capture_keys(r)] + ([("group", r["group"])] if r.get("group") else []) + \
            [("sha", s) for s in (r.get("sha256"), r.get("train_sha256")) if s]
        for k in ks:
            if k in caps:
                ex, g = caps[k]
                out[r["key"]].append({"exam": ex, "group": g, "how": "capture", "relation": list(k)})
                break
    return out


def capture_keys(r):
    ks = []
    if r.get("stem"):
        ks.append(("stem", r.get("family") or r["source"], r["stem"]))
    if r.get("capture"):
        ks.append(("capture", r["capture"]))
    if r.get("series"):
        ks.append(("series", r["series"]))
    return ks


# ------------------------------------------------------------------ 8. EXIF (DS.5.1 item 8)
def exif_refusal(row, read_orientation=None):
    """'exif_orientation' when the image's EXIF orientation is 2-8 (refused, never turned; after the guard)."""
    if read_orientation is None:
        from . import twostage as TS
        def read_orientation(p):  # noqa: E306
            try:
                return TS.header(p)[0]
            except Exception:  # noqa: BLE001 - an unreadable header is refused as unreadable
                return "unreadable"
    ori = read_orientation(row["image"])
    if ori == "unreadable":
        return "unreadable"
    return None if ori in EXIF_OK else "exif_orientation"


# ------------------------------------------------------------------ 10. writes (DS.5.1 item 10)
def materialise(row, root=None):
    """The file a row trains on: a lossless 640 px PNG (Ultralytics' load_image, then inc2.mask's fill) only when
    the mask fill changes its pixels, content-addressed under splits/ds1/images/; else its original path."""
    from . import guard as G
    from . import mask as MK
    if not row.get("mask"):
        row["train_image"], row["train_sha256"] = row["image"], row.get("sha256")
        return row
    import cv2
    im, _hw = B3._load_640(row["image"])
    if im is None:
        row["drop"] = "unreadable"
        return row
    out, _region, _kept, _mean = MK.masked_array(im.copy(), row["mask"], row["boxes"])
    if not (out != im).any():
        row["train_image"], row["train_sha256"] = row["image"], row.get("sha256")
        row["mask_no_change"] = True
        return row
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise DataRefused("PNG encoding failed for %s" % row["image"])
    data = buf.tobytes()
    psha = hashlib.sha256(data).hexdigest()
    dst = Path(root or out_dir()) / "images" / psha[:2] / ("%s.png" % psha)
    if not dst.is_file():
        C2.write_bytes_atomic(dst, data)
    back, _ = B3._load_640(dst)
    if back is None or back.shape != out.shape or not (back == out).all():
        raise DataRefused("%s does not read back as the pixels it was written from" % dst)
    row["train_image"], row["train_sha256"] = str(dst), psha
    h, v = G.image_hashes(str(dst))
    row["train_dhash"], row["train_variants"] = h, None if v is None else [int(v[k]) for k in B3.VARIANTS]
    return row


# ------------------------------------------------------------------ capture groups and the exam draws (DS.5.4)
def capture_groups(rows, links=(), bits=GROUP_BITS):
    """{key: group name} and {group name: digest}: 6-bit components under the 8 variants (both directions), the
    same capture relation (export stem, intake capture group, series), a DINOv2 link (links: [(key, key)]); a group
    named by the sha256 of its members' sorted sha256s (their keys without one)."""
    n = len(rows)
    uf = B3.UnionFind(n)
    idx = {r["key"]: i for i, r in enumerate(rows)}
    first = {}
    hashed = [i for i, r in enumerate(rows) if r.get("dhash") is not None]
    if hashed:
        comp = B3.hash_components([_hashrow(rows[i]) for i in hashed], bits).tolist()
        for i, c in zip(hashed, comp):
            k = ("hash", c)
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    for i, r in enumerate(rows):
        for k in capture_keys(r):
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    for a, b in links:
        if a in idx and b in idx:
            uf.union(idx[a], idx[b])
    names, digests = {}, {}
    for _root, mem in uf.groups(range(n)).items():
        ident = sorted(("sha:%s" % rows[i]["sha256"]) if rows[i].get("sha256") else ("key:%s" % rows[i]["key"])
                       for i in mem)
        dg = hashlib.sha256("\n".join(ident).encode("utf-8")).hexdigest()
        name = "g%s" % dg[:16]
        digests[name] = dg
        for i in mem:
            names[rows[i]["key"]] = name
    return names, digests


def draw_exams(rows_by_source, groups, digests, admitted, new_sources, md_frozen=None, md_target=None,
               excluded_groups=(), mtw_excluded_groups=()):
    """MD and (new sources only) MT-W v2 per source (DS.5.4): groups in the order stable_int("inc2/ds1/md_v1/
    <source>/<digest>"); MD until about 5 % of the source's admitted images (at least 15, at most 500; md_target
    {source: n} replaces the 5 % when MD-S0 is enlarged), then MT-W v2 until about 10 % more (30 to 1,000); both
    together at most 30 %; a group of more than 500 images, or one in excluded_groups, never; one in
    mtw_excluded_groups never in MT-W v2 (MT.2 item 2); md_frozen {source: [groups]} keeps an earlier draw's groups
    first (an enlargement keeps the order). A group not drawn stays a training candidate. Returns {source: {"md":
    [groups], "mtw": [groups], "md_images", "mtw_images", "admitted"}}."""
    out = {}
    for s in sorted(rows_by_source):
        rs = rows_by_source[s]
        size = collections.Counter(groups[r["key"]] for r in rs)
        n_adm = int(admitted.get(s) or len(rs))
        order = sorted(size, key=lambda g: (stable(SEED_MD % (s, digests[g])), g))
        cap = int(math.floor(EXAM_SHARE_MAX * n_adm))
        md_goal = int(md_target.get(s)) if md_target and s in md_target else \
            min(MD_MAX, max(MD_MIN, int(round(MD_SHARE * n_adm))))
        md, mtw, n_md, n_mtw = [], [], 0, 0
        for g in order:
            if size[g] > GROUP_MAX or g in excluded_groups:
                continue
            if n_md + n_mtw + size[g] > cap:
                continue
            if n_md < md_goal:
                md.append(g)
                n_md += size[g]
                continue
            if s in new_sources:
                mtw_goal = min(MTW_MAX, max(MTW_MIN, int(round(MTW_SHARE * n_adm))))
                if n_mtw < mtw_goal:
                    if g in mtw_excluded_groups:
                        continue
                    mtw.append(g)
                    n_mtw += size[g]
                    continue
            break
        out[s] = {"md": md, "mtw": mtw, "md_images": n_md, "mtw_images": n_mtw, "admitted": n_adm,
                  "cap_images": cap}
    return out


def enlarge_targets(draw, factor):
    """{source: MD images} for an enlargement of MD-S0 (DS.5.4): each source's MD scaled by the same factor, within
    its 30 % cap; the same order (draw_exams keeps the earlier groups first by construction)."""
    return {s: min(int(math.ceil(d["md_images"] * float(factor))), d["cap_images"]) for s, d in draw.items()}


# ------------------------------------------------------------------ the manifests (DS.5.3)
def floor_repeats(n_rows, n_s0, c):
    """r = round_half_up((N - C) / (N_S0 - C)), N the arm's rows before repeats (DS.0 item 15); 1 for S0's size."""
    if n_s0 <= c:
        raise DataRefused("S0 holds %d rows, no more than its %d train_core rows" % (n_s0, c))
    return max(1, int(B3.round_half_up((float(n_rows) - c) / (float(n_s0) - c))))


REPEAT_FMT = "%s__ds1rep%d"
REPEAT_RE = re.compile(r"__ds1rep\d+\Z")


def is_repeat(key):
    """True for a repeat row of the cwd12 exposure floor (<key>__ds1rep<j>; a source's own keys may hold '__r')."""
    return bool(REPEAT_RE.search(str(key)))


def repeat_base(key):
    return REPEAT_RE.sub("", str(key))


def repeat_rows(tc_rows, r):
    """The repeat rows of the cwd12 exposure floor: each train_core row listed r times (the row itself and r - 1
    rows <key>__ds1rep<j> naming the same files)."""
    out = []
    for row in tc_rows:
        for j in range(1, int(r)):
            out.append(dict(row, key=REPEAT_FMT % (row["key"], j)))
    return out


def family_cap(new_rows, base_count, other_count, family, cap=FAMILY_CAP):
    """(kept new rows, dropped videos): a family's new rows capped so that the family holds at most `cap` of the
    manifest's rows before repeats (family <= floor(cap / (1 - cap) x the other rows)), whole thinned videos dropped
    in the order stable_int("inc2/ds1/cap/<family>/<video>"), last first; base_count the family's rows the manifest
    holds already (never dropped: each larger manifest holds every row of the smaller ones)."""
    allowed = int(math.floor(cap / (1.0 - cap) * other_count)) - int(base_count)
    by = collections.defaultdict(list)
    for r in new_rows:
        by[r.get("series") or r["key"]].append(r)
    order = sorted(by, key=lambda v: (stable(SEED_CAP % (family, v)), v))
    kept, dropped, n = [], [], 0
    for v in order:
        if n + len(by[v]) <= max(0, allowed):
            kept += by[v]
            n += len(by[v])
        else:
            dropped.append(v)
    return kept, dropped, {"allowed_new": max(0, allowed), "kept_new": n, "family_base": int(base_count),
                           "other_rows": int(other_count), "dropped_videos": len(dropped)}


def s1cr_draw(s1c_rows, n_s0, tc_keys):
    """S1c-R (DS.5.3): exactly S0's size from S1c's rows before repeats, the train_core rows kept whole, the rest in
    the order stable_int("inc2/ds1/r/<key>")."""
    tc = [r for r in s1c_rows if r["key"] in tc_keys]
    rest = sorted((r for r in s1c_rows if r["key"] not in tc_keys), key=lambda r: (stable(SEED_R % r["key"]), r["key"]))
    need = int(n_s0) - len(tc)
    if need < 0 or need > len(rest):
        raise DataRefused("S1c-R needs %d rows beside train_core, S1c has %d" % (need, len(rest)))
    return tc + rest[:need]


def k_pools(siu_rows, kf_rows, crop_eppo=()):
    """The K arms' supplies (DS.5.3): SIU's target and non-target videos (no crop video), dense (3-bit deduped) and
    thinned, and KF's thinned Field and Turf rows. Returns (pools, T, N_K)."""
    tgt = [r for r in siu_rows if r.get("eppo") in SIU_TARGET_EPPO]
    non = [r for r in siu_rows if r.get("eppo") not in SIU_TARGET_EPPO and r.get("eppo") not in crop_eppo]
    t = sum(1 for r in tgt if r.get("thin"))
    supplies = {"KD": len(tgt) + len(non), "KT": sum(1 for r in tgt + non if r.get("thin")),
                "KF": sum(1 for r in kf_rows if r.get("thin"))}
    n_k = min([N_K_MAX] + list(supplies.values()))
    return {"target": tgt, "nontarget": non, "kf": kf_rows, "supplies": supplies}, t, n_k


def k_short(supplies, n_k):
    """{arm: supply} of KD and KT where the supply is under N_K (DS.5.3: N_K is the same in every K arm, so both are
    then not run, recorded in k_not_run.json)."""
    return {a: v for a, v in supplies.items() if a in ("KD", "KT") and v < n_k}


def _videos(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("series") or r["key"]].append(r)
    return by


def kd_draw(pools, t, n_k):
    """KD: whole target-species videos in seeded order until T frames, then whole non-target videos until N_K;
    every frame that passed the 3-bit dedupe (the last video cut in its own seeded order)."""
    out = []
    for part, goal in ((pools["target"], t), (pools["nontarget"], n_k)):
        vids = _videos(part)
        for v in sorted(vids, key=lambda v: (stable(SEED_K % ("KD", v)), v)):
            if len(out) >= goal:
                break
            for k in series_order(v, [r["key"] for r in vids[v]]):
                if len(out) >= goal:
                    break
                out.append(next(r for r in vids[v] if r["key"] == k))
    if len(out) != n_k:
        raise DataRefused("KD drew %d rows, N_K is %d" % (len(out), n_k))
    return out


def kt_draw(pools, t, n_k):
    """KT: T thinned target-species frames round-robin across all target videos, then N_K - T thinned non-target
    frames round-robin across all non-target videos."""
    out = []
    for part, goal in ((pools["target"], t), (pools["nontarget"], n_k - t)):
        vids = {v: [r for r in rs if r.get("thin")] for v, rs in _videos(part).items()}
        queues = {v: [next(r for r in rs if r["key"] == k) for k in series_order(v, [r["key"] for r in rs])]
                  for v, rs in vids.items() if rs}
        order = sorted(queues, key=lambda v: (stable(SEED_K % ("KT", v)), v))
        taken = 0
        while taken < goal and any(queues[v] for v in order):
            for v in order:
                if taken >= goal:
                    break
                if queues[v]:
                    out.append(queues[v].pop(0))
                    taken += 1
    if len(out) != n_k:
        raise DataRefused("KT drew %d rows, N_K is %d" % (len(out), n_k))
    return out


def kf_draw(pools, n_k):
    """KF: N_K thinned Field and Turf rows of base B in the order stable_int("inc2/ds1/k/KF/<row>")."""
    rows = sorted((r for r in pools["kf"] if r.get("thin")), key=lambda r: (stable(SEED_K % ("KF", r["key"])),
                                                                           r["key"]))
    if len(rows) < n_k:
        raise DataRefused("KF's supply %d is under N_K %d" % (len(rows), n_k))
    return rows[:n_k]


def nested_ok(small, large):
    """True when every line of the smaller manifest is a line of the larger one, byte for byte (DS.5.3)."""
    return set(small) <= set(large)


def manifest_bytes(rows):
    """(bytes, keys) of manifest rows in key order; refuses a repeated key."""
    rows = sorted(rows, key=lambda r: r["key"])
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise DataRefused("a manifest repeats a key")
    return b"".join(manifest_line(r) for r in rows), keys


def arm_record(arm, rows, path, sha, extra=None):
    from . import recipes as RC
    n = len(rows)
    base_rows = [r for r in rows if not is_repeat(r["key"])]
    rec = {"format": FMT_ARM, "arm": arm, "manifest": str(path), "sha256": sha, "images": n,
           "rows_before_repeats": len(base_rows), "counts": three_counts(base_rows),
           "recipe": RC.cold_budget("m640", n) if arm in D.BUILD_ARMS else None,
           "research_only": any(r.get("research_only") for r in rows),
           "permissive_only_rows": sum(1 for r in base_rows if not r.get("research_only")),
           "decided_by": D.DECIDED_BY, "written_utc": D._utc()}
    rec.update(extra or {})
    return rec


def loader_projection(rows):
    """base3's loader projection of a 1.2 M run on an arm's rows (CLOOP:2927): over 0.8 x 8 h the arm moves to the
    long path at the same 1.2 M (DS.6)."""
    rules, conf = load_rules()
    rate = B3.loader_rate(rows)
    w = rules if "walltime" in rules else conf
    return dict(B3.walltime(rate["ms_per_image"], w), loader=rate)


def season_rows(rows):
    """The keys of rows that break DS.0 item 6 (every S arm is season-free, as D1's B'): a tsw22 or tsw23 row (its
    source one of TSW) or a row of session 20220129_CanonEOS4000D_EO (its session field or its key). The copies of
    ood22 and ood23 images the guard removes before (DS.5.1 item 7)."""
    return sorted(str(r.get("key")) for r in rows if r.get("source") in TSW or (r.get("session") or "") ==
                  SEASON_SESSION or SEASON_SESSION in str(r.get("key") or ""))


def write_arm(arm, rows, extra=None):
    """An arm's manifest and record, each written once (an existing record must name the same bytes). An S arm with a
    row of the seasons (season_rows) is refused."""
    if arm.startswith("S"):
        bad = season_rows(rows)
        if bad:
            raise DataRefused("%s holds %d row(s) of tsw22, tsw23 or session %s (e.g. %s): every S arm is season-free "
                              "(DS.0 item 6)" % (arm, len(bad), SEASON_SESSION, bad[:3]))
    data, _keys = manifest_bytes([{k: r[k] for k in C.MANIFEST_KEYS} for r in rows])
    p = arm_path(arm)
    sha = hashlib.sha256(data).hexdigest()
    write_once(p, data)
    rp = arm_record_path(arm)
    old = _read_json(rp)
    if isinstance(old, dict):
        if old.get("sha256") != sha:
            raise DataRefused("%s exists naming other bytes: an arm's manifest is built once" % rp)
        return old
    rec = arm_record(arm, rows, p, sha, extra)
    if PROJECT_LOADER and arm in ("S1C", "S1U", "S2", "S1CR", "S0"):
        rec["loader_projection"] = loader_projection([r for r in rows if not is_repeat(r["key"])])
    write_once(rp, rec)
    return rec


PROJECT_LOADER = True            # tests switch it off (the projection reads Ultralytics' loader)


def arm_rows(arm):
    """The rows of a built arm (its manifest, checked against its record)."""
    rec = _read_json(arm_record_path(arm))
    if not isinstance(rec, dict) or C.sha256_file(arm_path(arm)) != rec.get("sha256"):
        raise DataRefused("arm %s is not built (or its manifest does not hash as recorded)" % arm)
    return read_jsonl(arm_path(arm)), rec


def arm_manifest(arm):
    """(path, sha256, images) of a built arm: what inc2.baseline build --ds1 reads."""
    _rows, rec = arm_rows(arm)
    return rec["manifest"], rec["sha256"], rec["images"]


# ------------------------------------------------------------------ the exam roots (Z1's pattern) and the never-train list
def write_exam_root(exam, rows, root, groups):
    """An exam in its own root: splits/v1/<exam>.jsonl (manifest rows, the class-12 labels), splits/v1/LOCK.json
    (inc/scorer.py's sha256), exams/v1/<exam>/ (symlinks and verified label copies). Every file written once."""
    from ..inc import scorer as S
    from ..inc import splits as SP
    root = Path(root)
    data, keys = manifest_bytes([{k: r[k] for k in C.MANIFEST_KEYS} for r in rows])
    sha = hashlib.sha256(data).hexdigest()
    write_once(root / "splits" / "v1" / ("%s.jsonl" % exam), data)
    lock = {"format": FMT_LOCK, "manifests": {exam: sha}, "splits_version": "v1",
            "scorer_sha256": C.sha256_file(Path(S.__file__).resolve()), "keys_sha256": list_sha256(keys),
            "decided_by": D.DECIDED_BY}
    lp = root / "splits" / "v1" / "LOCK.json"
    old = _read_json(lp)
    if isinstance(old, dict):
        if {k: old.get(k) for k in lock} != lock:
            raise DataRefused("%s exists with other contents: an exam's LOCK is written once" % lp)
    else:
        write_once(lp, dict(lock, created_utc=D._utc()))
    ed = root / "exams" / "v1" / exam
    man_rows = [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]
    if not ed.exists():
        C.materialise(man_rows, ed)
        for n in os.listdir(ed / "labels"):
            os.chmod(ed / "labels" / n, 0o444)
        os.chmod(ed / "data.yaml", 0o444)
    probs = SP.exam_problems(exam, man_rows, ed)
    if probs:
        raise DataRefused("the exam %s in %s has problems: %s" % (exam, root, probs[:3]))
    return {"root": str(root), "exam": exam, "manifest_sha256": sha, "lock_sha256": C.sha256_file(lp),
            "images": len(rows), "groups": len({groups[r["key"]] for r in rows})}


def never_train_rows():
    return read_jsonl(never_train_path())


def append_never_train(rows, exam, groups):
    """Every MD and MT-W row and group enters DS1's own never-train list at its draw (append-only; a row already
    listed for the exam is not listed again)."""
    have = {(r["exam"], r["key"]) for r in never_train_rows()}
    new = []
    for r in rows:
        if (exam, r["key"]) in have:
            continue
        new.append({"exam": exam, "key": r["key"], "source": r["source"], "sha256": r.get("sha256"),
                    "original_sha256": r.get("original_sha256") or r.get("sha256"),
                    # the file DS1's own DINOv2 index describes (own_exam_rows): the exam row's image as scored
                    "image": r.get("train_image") or r.get("image"),
                    "train_sha256": r.get("train_sha256"), "dhash": r.get("dhash"), "variants": r.get("variants"),
                    "group": groups.get(r["key"]), "capture_keys": [list(k) for k in capture_keys(r)],
                    "drawn_utc": D._utc()})
    if new:
        p = never_train_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            for r in new:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(new)


def freeze_record(exam, root_rec, extra=None):
    """splits/ds1/<md_v1|mtw_v2>/<exam>/frozen.json, written once: the root that is the exam, its LOCK's sha256 (whose
    OpenTimestamps proof the delegate takes), the draw it froze."""
    doc = {"format": FMT_EXAM, "exam": exam, "frozen": True, "root": root_rec["root"],
           "lock_sha256": root_rec["lock_sha256"], "manifest_sha256": root_rec["manifest_sha256"],
           "images": root_rec["images"], "groups": root_rec["groups"], "decided_by": D.DECIDED_BY,
           "frozen_utc": D._utc()}
    doc.update(extra or {})
    write_once(exam_dir(exam) / "frozen.json", doc)
    return doc


def frozen(exam):
    d = _read_json(exam_dir(exam) / "frozen.json")
    return d if isinstance(d, dict) and d.get("frozen") else None


# ------------------------------------------------------------------ the inputs (production: the cluster's files)
class Inputs(object):
    """Everything a build step reads, each recorded by sha256 (production loaders below; tests build one)."""

    def __init__(self, **kw):
        self.base_b = kw.get("base_b") or {}            # {key: provenance row} of base B (base3's provenance)
        self.bprime = kw.get("bprime") or {}            # {key: manifest row} of D1's B' (D1B1, after the season guard)
        self.base_a = kw.get("base_a") or {}            # {key: manifest row} of base A one-class (base_v2_weed)
        self.tc_keys = set(kw.get("tc_keys") or ())     # train_core keys (base_v2's)
        self.ndsu_keys = set(kw.get("ndsu_keys") or ())  # base_v2's 66 NDSU rows
        self.setting = kw.get("setting") or {}          # {key: Field | Turf | Indoor} of base B (V3F.4's map)
        self.candidates = kw.get("candidates") or []    # new rows (the queue, the admissions, the intake)
        self.decisions = kw.get("decisions") or {}      # {source: decided sheet} (inc2.ds1_sheets)
        self.exams = kw.get("exams") or {}              # {name: {"keys", "H", "ok"}} for the dHash guard
        self.scanners = kw.get("scanners") or {}        # {name: EmbedScanner}
        self.eval_sources = set(kw.get("eval_sources") or ())
        self.quarantined = set(kw.get("quarantined") or ())
        # {source: [cites]} of the stream's quarantines (DS1 deviation 2, item 2: SIU's D31 quarantine is read past)
        self.quarantine_cites = dict(kw.get("quarantine_cites") or {})
        self.read_size = kw.get("read_size")            # (path) -> (W, H) from the header (tests: a stub)
        self.never = set(kw.get("never") or ())
        self.families = kw.get("families") or {}        # {source: family}
        self.predict = kw.get("predict")                # E1-A's predictions (the model audit)
        self.audit_ref = kw.get("audit_ref")            # train_core's rates under the same predictor
        self.hasher = kw.get("hasher")                  # inc2.guard.image_hashes (tests: a stub)
        self.consumed = set(kw.get("consumed") or ())   # keys a stream consumed (an increment or a pool)
        self.base_b_man = kw.get("base_b_man") or {}    # {key: manifest row} of base B (KF's rows)
        self.links = kw.get("links")                    # DINOv2 links among rows (capture groups), given
        self.link_fn = kw.get("link_fn")                # (rows, tag) -> [(key, key)]: DINOv2 links computed
        self.ref_scan_fn = kw.get("ref_scan_fn")        # (rows, ref_rows, tag) -> {key: [hit]}: DINOv2 copies of rows
        self.box_src_fn = kw.get("box_src_fn")          # (source, rows) -> ({key: [src id per box]}, record)
        self.pool_ref_fn = kw.get("pool_ref_fn")        # () -> [{key, dhash, variants}] of the stream's pools
        self.intake_hits = kw.get("intake_hits")        # {source: [cwd12 copies intake refused]} (D28-v2's input)
        self.class_map = kw.get("class_map") or {}      # {source: {src id: {"name", "inc_id"}}} (intake's classes)
        self.read_orientation = kw.get("read_orientation")
        self.record = kw.get("record") or {}
        self.production = bool(kw.get("production", False))


def load_inputs(phase, procs=5, embedder=None):
    """The production inputs (the cluster): base B's provenance and D1's B' (splits/v3fam's D1B1), base A, LOCK v2's
    train_core and NDSU keys, V3F's setting map, the phase's candidates from Step 1's stream, the decided sheets,
    the exams' hashes and scanners, the evaluation groups by name, the stream's quarantines and the domain config's
    never lists."""
    from . import v3fam_data as VD
    summ = B3.read_summary()
    if summ is None:
        raise DataRefused("no complete splits/v3/summary.json: base B is not built")
    pf = summ["files"]["provenance"]
    prov = read_jsonl(B3.out_dir() / pf["file"])
    if C.sha256_file(B3.out_dir() / pf["file"]) != pf["sha256"]:
        raise DataRefused("base B's provenance does not hash as its summary records")
    base_b = {r["key"]: r for r in prov if "B" in (r.get("arms") or [])}
    for r in base_b.values():
        # base B's own boxes (its class-12 labels), for the copy rule's box match (DS.5.1 item 5)
        try:
            r["_boxes"] = [b[1:5] for b in read_inc_label(r["label"])]
        except (OSError, ValueError, DataRefused):
            r["_boxes"] = []
    bb = summ["arms"]["B"]
    base_b_man = {r["key"]: r for r in C.read_manifest(bb["manifest"])}
    vs = VD.read_summary()
    if vs is None or "D1B1" not in (vs.get("arms") or {}):
        raise DataRefused("no complete splits/v3fam/summary.json with D1's B' (D1B1): B'o starts from it")
    bp_path, bp_sha, _n = VD.arm_manifest("D1B1")
    bprime = {r["key"]: r for r in C.read_manifest(bp_path)}
    ba = summ["arms"]["A"]
    base_a = {r["key"]: r for r in C.read_manifest(ba["manifest"])}
    lock2 = C2.read_lock_v2()
    tc = {r["key"] for r in C.read_manifest(C2.v2_manifest_path("train_core"))}
    ndsu = {k for k, r in base_b.items() if r.get("source") in VD.NDSU_SOURCES}
    setting = {}
    for k, r in base_b.items():
        try:
            setting[k] = VD.setting_of(dict(r, key=k))[0]
        except VD.DataRefused:
            setting[k] = None
    rules, conf = load_rules()
    from . import ds1_sheets as SH
    decisions = SH.decided_sources()
    cands = load_candidates(phase, base_b)
    qcites = stream_quarantine_cites()
    exams, erec = exam_hashes()
    predict = e1a_predictor()
    tc_rows = [dict(r, boxes=[b[1:5] for b in read_inc_label(r["label"])])
               for r in C.read_manifest(C2.v2_manifest_path("train_core"))]
    audit_ref = audit_rates(audit_sample(tc_rows, TRAIN_CORE), predict)
    return Inputs(base_b=base_b, bprime=bprime, base_a=base_a, tc_keys=tc, ndsu_keys=ndsu, setting=setting,
                  base_b_man=base_b_man, consumed=stream_consumed(), predict=predict, audit_ref=audit_ref,
                  candidates=cands, decisions=decisions, exams=exams, scanners=scanners(embedder),
                  eval_sources={s for g in (conf.get("evaluation_groups") or {}).get("groups", {}).values() for s in g},
                  quarantined=set(qcites), quarantine_cites=qcites, never=domain_never(),
                  families=families(conf), production=True,
                  link_fn=lambda rows, tag: dino_links(rows, tag, embedder),
                  ref_scan_fn=lambda rows, ref, tag: dino_ref_hits(rows, ref, tag, embedder),
                  box_src_fn=lambda source, rows: recover_box_src(source, rows),
                  pool_ref_fn=stream_pool_hash_rows,
                  record={"step1_batches": step1_batches(),
                          "splits_v3_summary_sha256": C.sha256_file(B3.out_dir() / B3.SUMMARY),
                          "v3fam_summary_sha256": C.sha256_file(VD.out_dir() / VD.SUMMARY),
                          "bprime_sha256": bp_sha, "lock_v2_sha256": C.sha256_file(C2.LOCK_PATH),
                          "lock_v2_train_core": lock2["manifests"].get("train_core"), "exams": erec})


def eligibility_ctx(inputs):
    return {"base_keys": set(inputs.base_b), "base_shas": {r.get("original_sha256") for r in inputs.base_b.values()},
            "eval_sources": inputs.eval_sources, "quarantined": inputs.quarantined, "never": inputs.never,
            "consumed": inputs.consumed, "quarantine_cites": inputs.quarantine_cites}


def stream_consumed():
    """Keys any stream version consumed (a pool's rows and every increment's), which Q does not count as free."""
    try:
        from . import stream as ST
        out = set()
        sd = C.INC_DIR / "stream"
        for sid in sorted(p.name for p in sd.iterdir() if p.is_dir()) if sd.is_dir() else []:
            try:
                f = ST.Stream(sid, quiet=True).load(repair=False)
            except Exception:  # noqa: BLE001 - a stream that cannot be folded consumed nothing readable here
                continue
            for k, v in f.keys.items():
                if v.get("status") not in ST.ELIGIBLE_STATUSES:
                    out.add(k)
            out |= {r["key"] for r in f.pool_rows(f.pool)}
        return out
    except Exception:  # noqa: BLE001
        return set()


def stream_pool_hash_rows(hasher=None):
    """[{key, dhash, variants}] of every stream version's current pool, what MT-W v2 keeps away from (MAIN_TEST.md
    MT.2 item 2): each row's hashes from the pool's dHash map, else hashed from its file. Fail closed: a stream that
    cannot be folded, or a pool row that cannot be hashed, refuses (DataRefused)."""
    from . import stream as ST
    if hasher is None:
        from . import guard as G
        hasher = G.image_hashes
    out = []
    sd = C.INC_DIR / "stream"
    for sid in sorted(p.name for p in sd.iterdir() if p.is_dir()) if sd.is_dir() else []:
        try:
            f = ST.Stream(sid, quiet=True).load(repair=False)
            if not f.pool:
                continue
            rows = f.pool_rows(f.pool)
            try:
                hs = f.pool_hashes(f.pool)
            except Exception:  # noqa: BLE001 - no dHash map recorded: each row is hashed from its file below
                hs = {}
        except Exception as e:  # noqa: BLE001
            raise DataRefused("stream %s's pool cannot be read (%s: %s): MT-W v2 cannot be kept apart from it "
                              "(fail closed)" % (sid, type(e).__name__, str(e)[:200]))
        for r in rows:
            v = [int(x) for x in hs.get(r["key"]) or []]
            if not v:
                h, var = hasher(r["image"])
                if h is None or var is None:
                    raise DataRefused("stream %s's pool row %s cannot be hashed (fail closed)" % (sid, r["key"]))
                v = [int(var[k]) for k in B3.VARIANTS] if isinstance(var, dict) else [int(x) for x in var]
            out.append({"key": "pool|%s|%s" % (sid, r["key"]), "dhash": v[0],
                        "variants": v if len(v) == len(B3.VARIANTS) else None})
    return out


def e1a_predictor(conf=0.25):
    """E1-A seed 0's predictions (cx, cy, w, h, conf) on a row's original image (the model audit, DS.5.1 item 4):
    E1-A trained on base_v2 only, so it never saw these sources."""
    from . import v3fam_systems as SYS
    from . import twostage as TS
    src = SYS.run_source(D.POWER_EXPS["E1A"], 0, testing_ok=TS._testing_ok())
    state = {}

    def predict(row):
        if "m" not in state:
            from ultralytics import YOLO
            state["m"] = YOLO(src["weights"])
        res = state["m"].predict(row["image"], imgsz=640, conf=conf, verbose=False)[0]
        out = []
        for b, c in zip(res.boxes.xywhn.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy().tolist()):
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        return out
    predict.weights_sha256 = src["weights_sha256"]
    return predict


def step1_batches(inc_dir=None):
    """The committed Step 1 batches (ledger/batches.jsonl), sorted: what a screen saw, and whether one came since."""
    p = (Path(inc_dir) if inc_dir else C.INC_DIR) / "step1_stream" / "ledger" / "batches.jsonl"
    return sorted({str(r.get("batch")) for r in read_jsonl(p) if r.get("batch")})


def families(conf):
    out = {s: v.get("family") for s, v in (conf.get("sources") or {}).items() if v.get("family")}
    for s, v in (conf.get("intake") or {}).items():
        if v.get("family"):
            out[s] = v["family"]
    for s in MFWD_SOURCES:
        out[s] = "mfwd"
    out[SIU] = SIU_FAMILY
    return out


def stream_quarantine_cites():
    """{source: [the diagnoses its quarantines cite, D28 or D31]} over every stream (inc2.stream's ledger fold,
    q_sources {source: {cite, seq, utc}}, as base3 reads it); a quarantine without a cite reads '?'."""
    try:
        from .. import inc_autopilot  # noqa: F401  (the package is present wherever the stream runs)
        from . import stream as ST
        sids = sorted(p.name for p in (C.INC_DIR / "stream").iterdir() if p.is_dir()) \
            if (C.INC_DIR / "stream").is_dir() else []
        out = collections.defaultdict(set)
        for sid in sids:
            try:
                q = ST.Stream(sid, quiet=True).load(repair=False).q_sources
            except Exception:  # noqa: BLE001 - a stream that cannot be folded quarantines nothing here
                continue
            for s, v in q.items():
                out[s].add(str((v or {}).get("cite") or "?") if isinstance(v, dict) else "?")
        return {s: sorted(c) for s, c in out.items()}
    except Exception:  # noqa: BLE001
        return {}


def stream_quarantined():
    """The stream's quarantined sources (inc2.stream's ledger fold, as base3 reads it)."""
    return set(stream_quarantine_cites())


def domain_never():
    """The domain config's never-fetch and never-train sources (collect/domains/weed.json known items and the
    funnel's never-train list)."""
    out = set(NEVER_FETCHED) | set(ROLE_C)
    try:
        cfg = json.loads((Path(__file__).resolve().parents[1] / "collect" / "domains" / "weed.json").read_text())
        for it in cfg.get("known_items") or []:
            if it.get("never_fetch") or it.get("never_train"):
                out.add(str(it.get("id")))
    except (OSError, ValueError):
        pass
    return out


def load_candidates(phase, base_b, inc_dir=None):
    """The candidate rows of a phase from Step 1's stream (module docstring): the queue's rows (holds, refusals,
    test_v1 folded in by inc2.step1_stream.load_queue) and every committed batch's not-admitted rows, each with its
    original file and intake label (the intake batch's manifest, or the v1 pool for b0000)."""
    from . import step1_stream as S1
    inc = Path(inc_dir) if inc_dir else C.INC_DIR
    L = S1.Layout(inc_dir=inc)
    queue = {r["key"]: r for r in S1.load_queue(L)}
    files = {}
    for d in sorted((inc / "intake").glob("i0*")) if (inc / "intake").is_dir() else []:
        for r in read_jsonl(d / "manifest.jsonl"):
            files[r["key"]] = dict(r, batch_dir=str(d))
    pool = inc / "step1" / "pool.jsonl"
    for r in read_jsonl(pool):
        files.setdefault(r["key"], r)
    out = []
    seen = set()
    for e in read_jsonl(L.ledger):
        bdir = L.batch_dir(e["batch"])
        bj = _read_json(bdir / "batch.json") or {}
        if bj.get("kind") == "knowntruth":
            continue
        im = ingest_matches(bdir, bj)
        for a in read_jsonl(bdir / "admission.jsonl"):
            k = a["key"]
            if k in seen:
                continue
            seen.add(k)
            q = queue.get(k)
            f = files.get(k)
            if f is None:
                continue
            row = candidate_row(k, a, q, f, im.get(k))
            if row is not None:
                out.append(row)
    return out


def ingest_matches(bdir, bj=None):
    """{key: {"guard_match", "pair_cos"}} of a Step 1 batch's copy refusals (D28-v2's input: which evaluation split
    each copy matched, and the dHash hits' pair cosines): its ingest.jsonl rows refused as copies of an evaluation
    image, and b0000's eval_hit_pairs (b0000 writes no ingest.jsonl)."""
    out = {}
    p = Path(bdir) / "ingest.jsonl"
    if p.is_file():
        with open(p) as fh:
            for line in fh:
                if "near_eval" not in line:
                    continue
                r = json.loads(line)
                if str(r.get("decision") or "").replace("masked_", "") in EVAL_COPY_REASONS:
                    out[r["key"]] = {"guard_match": r.get("guard_match"), "pair_cos": r.get("pair_cos")}
    for r in (bj or {}).get("eval_hit_pairs") or []:
        if isinstance(r, dict) and r.get("key") is not None:
            out.setdefault(r["key"], {"guard_match": r.get("match"), "pair_cos": r.get("pair_cos")})
    # the batch's D28-v2 sidecar (step1_stream/eval_hits/<batch>.json: each dHash hit's match and pair cosine, weighed
    # again for the batches committed before D28-v2; its keys may carry the pool's prefix, 'v1|<key>')
    side = _read_json(Path(bdir).parent.parent / "eval_hits" / ("%s.json" % Path(bdir).name)) or {}
    for r in side.get("pairs") or []:
        if isinstance(r, dict) and r.get("key") is not None:
            k = str(r["key"])
            k = k.split("|", 1)[1] if "|" in k and "__" not in k.split("|", 1)[0] else k
            got = out.setdefault(k, {"guard_match": None, "pair_cos": None})
            got["guard_match"] = got.get("guard_match") or r.get("match")
            if got.get("pair_cos") is None:
                got["pair_cos"] = r.get("pair_cos")
    return out


def candidate_row(key, adm, q, f, im=None):
    """One candidate from its admission record, its queue row (None when not admitted), its original file and its
    Step 1 batch's copy-refusal match (im: ingest_matches). A queued row's holds are the queue's, events folded (a hold
    step1_stream's serve-holds released is gone, DS1 deviation 2 item 2: the intake file record keeps the holds it was
    admitted with); a row the queue does not hold keeps its intake record's."""
    try:
        boxes = read_inc_label(f["label"])
    except (OSError, ValueError, DataRefused):
        return None
    rel = f.get("rel") or os.path.basename(str(f.get("image") or ""))
    src = adm.get("source") or f.get("source")
    video, eppo = siu_video(rel or key) if src == SIU else (None, None)
    series = video if src == SIU else (f.get("capture_group") if src in MFWD_SOURCES else None)
    pool = "p1" if src == SIU else ("p5" if q is not None or any(c < 12 for c, *_ in boxes) else "p3")
    return {"key": key, "source": src, "pool": pool, "image": f.get("image"), "sha256": f.get("sha256"),
            "label": f.get("label"), "boxes": boxes, "W": f.get("width"), "H": f.get("height"),
            "rel": rel, "series": series, "eppo": eppo, "capture": ("%s|%s" % (src, f["capture_group"]))
            if f.get("capture_group") else None, "stem": _stem(rel), "session": f.get("session") or "",
            "dhash": (q or {}).get("dhash") if q else f.get("dhash"), "variants": None,
            "licence": (q or {}).get("licence") or f.get("licence"),
            "research_only": bool((q or {}).get("research_only") or f.get("research_only")),
            "holds": list((q.get("holds") if q is not None else f.get("holds")) or []),
            "refused": (q or {}).get("refused") or adm.get("refusal"), "admission": adm.get("admission"),
            "test_v1": (q or {}).get("test_v1"),
            "batch": adm.get("batch") or f.get("batch"), "kind_q": (q or {}).get("kind"),
            # Step 1's guard match and pair cosine of a refused copy (D28-v2's input: DS.5.1 item 1(a))
            "guard_match": adm.get("guard_match") or (im or {}).get("guard_match"),
            "pair_cos": adm.get("pair_cos", (im or {}).get("pair_cos", (q or {}).get("pair_cos"))),
            "box_src": None}


def _stem(rel):
    m = B3.RF_STEM_RE.match(os.path.splitext(os.path.basename(str(rel or "")))[0])
    return m.group("stem") if m else None


def exam_hashes():
    """{exam: {"keys", "H", "ok"}} of every exam the guard reads (DS.5.1 item 7): cwd12 dev and test and ImageWeeds
    (the zoo's checked cache), ood22 and ood23 (base_v2's copies' hashes), test v1 and its companions (every
    earlier test list), evalgroups_v1 (the evaluation groups' stored hashes), MD v1 and MT-W v2 (DS1's never-train
    list), H10d once F9 wrote it."""
    from . import v3fam_data as VD
    import numpy as np
    zoo, zrec = VD._zoo_exam_hashes()
    v2prov = VD._provenance(C2.BASE_PROVENANCE, C2.read_lock_v2().get("provenance_sha256"), "base_v2 provenance")
    ood, orec = VD._ood_exam_hashes(v2prov)
    out = dict(zoo, **ood)
    prior, prec = B3.prior_test_lists()
    pr = [p for p in prior if p.get("dhash") is not None]
    if pr:
        H, ok, _h = B3.hash_matrix(pr)
        out["test_v1"] = {"keys": [p.get("key") for p in pr], "H": H, "ok": ok}
    nt = [r for r in never_train_rows() if r.get("dhash") is not None]
    if nt:
        H, ok, _h = B3.hash_matrix(nt)
        out["ds1_exams"] = {"keys": ["%s|%s" % (r["exam"], r["key"]) for r in nt], "H": H, "ok": ok}
    # the evaluation groups (evalgroups_v1 and the OOD-dev groups, base3's evaluation_groups): every image's 8
    # variants (the Step 1 pool's stored hashes, else hashed here), as base3's 6-bit guard reads them
    egs, egrec = eval_group_hashes()
    for g, (HG, okG) in sorted(egs.items()):
        if len(HG):
            out["evalgroup:%s" % g] = {"keys": ["%s#%d" % (g, i) for i in range(len(HG))], "H": HG, "ok": okG}
    h10 = C.INC_DIR / "step1_r1" / "domain_dev.jsonl"
    rec = {"zoo": zrec, "ood": orec, "prior_test": prec, "ds1_never_train": len(nt), "h10d": h10.is_file(),
           "evaluation_groups": egrec}
    if h10.is_file():
        from ..inc import verify as V
        meta = {m["key"]: m for m in V._read_jsonl(V.POOL_META)} if Path(V.POOL_META).is_file() else {}
        keys = [r["key"] for r in C.read_manifest(h10)]
        rows = [{"dhash": (meta.get(k) or {}).get("dhash"), "variants": None} for k in keys]
        have = [(k, r) for k, r in zip(keys, rows) if r["dhash"] is not None]
        if len(have) != len(keys):
            raise DataRefused("%d H10d image(s) have no stored dHash: the guard cannot clear against it (fail "
                              "closed)" % (len(keys) - len(have)))
        H, ok, _h = B3.hash_matrix([r for _k, r in have])
        out["h10d"] = {"keys": [k for k, _r in have], "H": H, "ok": ok}
        rec["h10d_images"] = len(have)
    for k, v in out.items():
        v["ok"] = np.asarray(v["ok"], dtype=bool)
    return out, rec


def scanners(embedder=None):
    """The calibrated DINOv2 detector against cwd12 dev, test and ImageWeeds (LOCK v2's stored descriptors) and
    against ood22 and ood23 (D1's season guard's index), and DS1's own index over test v1 and its companions,
    evalgroups_v1, MD v1, MT-W v2 and H10d (built in ds1/v1/leak/)."""
    from . import v3fam_data as VD
    v2prov = VD._provenance(C2.BASE_PROVENANCE, C2.read_lock_v2().get("provenance_sha256"), "base_v2 provenance")
    ood, _r = VD._ood_exam_hashes(v2prov)
    out = {"eval": VD.eval_scanner(embedder), "ood": VD.ood_scanner(ood, embedder, cache=D.root() / "leak" /
                                                                       "ood_desc.npz")}
    own = own_exam_rows()
    exams = own.pop("ds1_exams", None)
    if own:
        # the fixed part (test v1 and its companions, the evaluation groups, H10d): its index is described once,
        # when a step first scans against it (a screen never does)
        out["own"] = LazyScanner(lambda: own_scanner(own, embedder))
    if exams:
        # DS1's own MD and MT-W rows, an index of its own (it grows with each draw; the fixed part is not redone)
        out["own_exams"] = LazyScanner(lambda: own_scanner({"ds1_exams": exams}, embedder, prefix="own_exams"))
    return out


class LazyScanner(object):
    """A scanner built the first time it is used (its index, its calibration or a scan)."""

    def __init__(self, make):
        self._make = make
        self._sc = None

    def _get(self):
        if self._sc is None:
            self._sc = self._make()
        return self._sc

    @property
    def index(self):
        return self._get().index

    @property
    def calibration(self):
        return self._get().calibration

    def scan(self, rows, desc_path=None, procs=1):
        return self._get().scan(rows, desc_path=desc_path, procs=procs)


def _registry():
    """The dataset registry base3 reads (its evaluation groups' image folders), or None when it cannot be read."""
    try:
        with open(B3.registry_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def eval_group_hashes():
    """({group: (H [n, 8], ok [n])}, record) of the evaluation groups (evalgroups_v1 and the OOD-dev groups, base3's
    evaluation_groups by name), base3's own reader: the Step 1 pool's stored hashes, else hashed here. Fail closed: a
    registry that cannot be read, or a group image that cannot be hashed, refuses."""
    _rules, conf = load_rules()
    reg = _registry()
    if reg is None:
        raise DataRefused("the dataset registry %s cannot be read: the guard cannot clear against the evaluation "
                          "groups (fail closed)" % B3.registry_path())
    # the images base3 hashed here are hashed once for DS1 and kept (ds1/v1/leak/eval_group_hashes.json)
    cache = D.root() / "leak" / "eval_group_hashes.json"
    stored = dict(B3.pool_hashes()[0])
    old = _read_json(cache) or {}
    stored.update({p: [int(x) for x in v] for p, v in (old.get("hashes") or {}).items()})
    imgs, _missing = B3.eval_group_images(conf, reg)
    todo = sorted({os.path.normpath(p) for slugs in imgs.values() for ps in slugs.values() for p in ps
                   if os.path.normpath(p) not in stored})
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        from . import guard as G

        def one(p):
            h, v = G.image_hashes(p)
            return p, (None if h is None or v is None else [int(h)] + [int(v[k]) for k in B3.VARIANTS[1:]])
        with ThreadPoolExecutor(max_workers=8) as ex:
            got = dict(ex.map(one, todo))
        new = {p: v for p, v in got.items() if v is not None}
        stored.update(new)
        C2.write_json_atomic(cache, {"format": "inc2-ds1-eval-group-hashes/1", "hashes": dict(
            sorted(dict(old.get("hashes") or {}, **new).items())), "written_utc": D._utc()})
    egs, rec = B3.eval_group_hashes(conf, reg, stored=stored)
    bad = {g: v for g, v in (rec.get("groups") or {}).items() if any(int((s or {}).get("unhashed") or 0)
                                                                     for s in (v.get("slugs") or {}).values())}
    if bad:
        raise DataRefused("evaluation group images that cannot be hashed (%s): the guard cannot clear against them "
                          "(fail closed)" % sorted(bad))
    return egs, rec


def eval_group_rows():
    """{group: [{key, image}]} of the evaluation groups' images (DS1's own DINOv2 index)."""
    _rules, conf = load_rules()
    reg = _registry()
    if reg is None:
        raise DataRefused("the dataset registry %s cannot be read (fail closed)" % B3.registry_path())
    imgs, _missing = B3.eval_group_images(conf, reg)
    out = {}
    for g, slugs in sorted(imgs.items()):
        out["evalgroup:%s" % g] = [{"key": "%s#%s#%d" % (g, slug, i), "image": p}
                                   for slug, paths in sorted(slugs.items()) for i, p in enumerate(paths)]
    return out


def h10d_rows():
    """[{key, image, sha256}] of the funnel's H10d hold-out once F9 wrote it (step1_r1/domain_dev.jsonl), else []."""
    h10 = C.INC_DIR / "step1_r1" / "domain_dev.jsonl"
    if not h10.is_file():
        return []
    return [{"key": "h10d|%s" % r["key"], "image": r["image"], "sha256": r.get("sha256")}
            for r in C.read_manifest(h10)]


def own_exam_rows():
    """{split: [{key, image, sha256}]} of DS1's own guard index: test v1 and its companions, MD v1 and MT-W v2 (their
    files, from DS1's never-train list), the evaluation groups (evalgroups_v1, the OOD-dev groups) and H10d once F9
    wrote it."""
    out = collections.defaultdict(list)
    prior, _rec = B3.prior_test_lists()
    for p in prior:
        img = p.get("image") or p.get("original_image")
        if img:
            out["test_v1"].append({"key": str(p.get("key")), "image": img, "sha256": p.get("sha256")})
    for r in never_train_rows():
        if r.get("train_image") or r.get("image"):
            out["ds1_exams"].append({"key": "%s|%s" % (r["exam"], r["key"]), "image": r.get("train_image") or
                                     r.get("image"), "sha256": r.get("train_sha256") or r.get("sha256")})
    for g, rows in eval_group_rows().items():
        out[g] += rows
    hr = h10d_rows()
    if hr:
        out["h10d"] = hr
    return dict(out)


def own_scanner(rows_by_split, embedder=None, prefix="own"):
    """The calibrated DINOv2 detector (LOCK v2's calibration and embedder) over an index of rows_by_split, its
    descriptors cached under ds1/v1/leak/<prefix>_<the rows' tag>.npz."""
    from ..funnel import embed as FE
    from . import embed_calibration as EC
    from . import guard as G
    cal, why = EC.locked(C2.LOCK_PATH, production=True)
    if cal is None:
        raise DataRefused("LOCK v2 records no calibration (%s): DS1's guard cannot run (fail closed)" % why)
    model, _, pooling = str(cal["embedder"]).rpartition(":")
    emb = embedder or FE.LazyEmbedder(model, pooling)
    tag = hashlib.sha256(json.dumps({s: sorted(r["key"] for r in rs) for s, rs in rows_by_split.items()},
                                    sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return G.EmbedScanner(G.eval_index(rows_by_split, emb, D.root() / "leak" / ("%s_%s.npz" % (prefix, tag))), cal)


def dino_links(rows, tag, embedder=None):
    """[(key, key)] of the calibrated DINOv2 detector's copies among rows (DS.5.4: capture groups join images it
    links): an index over the rows and the rows scanned against it, a row's match with itself left out. Fail closed:
    a row it cannot describe refuses."""
    if len(rows) < 2:
        return []
    items = [{"key": r["key"], "image": r.get("train_image") or r["image"], "sha256": r.get("sha256")} for r in rows]
    sc = own_scanner({"self": items}, embedder, prefix="links_%s" % slug(tag))
    hits = embed_hits(items, sc, D.root() / "leak" / ("links_%s_q.npz" % slug(tag)))
    return sorted({tuple(sorted((k, str(h["exam_key"])))) for k, hs in hits.items() for h in hs
                   if str(h.get("exam_key")) != str(k)})


def dino_ref_hits(rows, ref_rows, tag, embedder=None):
    """{key: [hit]} of rows the calibrated DINOv2 detector calls copies of ref_rows' images (an index over ref_rows,
    its descriptors cached by tag and the rows' keys, so an arm's index is described once). Fail closed on a file it
    cannot describe."""
    if not rows or not ref_rows:
        return {}
    ref = [{"key": r["key"], "image": r.get("train_image") or r["image"], "sha256": r.get("sha256")} for r in ref_rows]
    sc = own_scanner({"ref": ref}, embedder, prefix="ref_%s" % slug(tag))
    q = hashlib.sha256("\n".join(sorted(str(r["key"]) for r in rows)).encode("utf-8")).hexdigest()[:16]
    return embed_hits(rows, sc, D.root() / "leak" / ("query_%s.npz" % q))


# ------------------------------------------------------------------ the intake's records of a source
def _intake_batches(source, inc=None):
    """The committed intake batch directories of a source (intake/i<nnnn>_<source>), in order."""
    idir = (Path(inc) if inc else C.INC_DIR) / "intake"
    return [d for d in sorted(idir.glob("i0*")) if idir.is_dir() and _batch_source(d) == source]


_BATCH_SOURCE = {}


def _batch_source(d):
    """The source of an intake batch: its summary's, else its manifest's first row's (read once per process)."""
    key = str(d)
    if key in _BATCH_SOURCE:
        return _BATCH_SOURCE[key]
    got = None
    m = _read_json(d / "summary.json")
    if isinstance(m, dict) and m.get("source"):
        got = str(m["source"])
    elif (d / "manifest.jsonl").is_file():
        with open(d / "manifest.jsonl") as fh:
            for line in fh:
                if line.strip():
                    got = str(json.loads(line).get("source"))
                    break
    _BATCH_SOURCE[key] = got
    return got


def _decision_lines(d, needle):
    p = d / "decisions.jsonl"
    if not p.is_file():
        return
    with open(p) as fh:
        for line in fh:
            if needle in line:
                yield json.loads(line)


def intake_class_map(source, inc=None):
    """{src id: {"name", "inc_id", "status"}} of a source's classes as its intake batches mapped them (decisions.jsonl
    kind 'class')."""
    out = {}
    for d in _intake_batches(source, inc):
        for r in _decision_lines(d, '"class"'):
            if r.get("kind") == "class" and r.get("source") == source and r.get("src_id") is not None:
                out[str(r["src_id"])] = {"name": r.get("name"), "inc_id": r.get("inc_id"), "status": r.get("status")}
    return out


def intake_cwd12_hits(source, inc=None, copy_cos=COPY_COS):
    """[{key, reason, split, pair_cos}] of the source's images its intake refused as copies of a cwd12 dev or test
    image at the calibrated copy threshold (DS.5.1 item 1(a), D28-v2 unchanged): an embedding hit (its cosine is over
    the threshold by construction), or a dHash hit whose pair cosine reaches it, or one with no pair cosine (D28-v2:
    fail closed; the batch's eval_hits sidecar consulted first)."""
    out = []
    for d in _intake_batches(source, inc):
        side = _read_json(d / "eval_hits.json") or {}
        pc = {str(p.get("key")): p.get("pair_cos") for p in side.get("pairs") or [] if isinstance(p, dict)}
        for r in _decision_lines(d, '"near_eval'):
            if r.get("kind") != "image" or r.get("decision") != "rejected" or r.get("source") != source:
                continue
            hit = copy_hit(r.get("reason"), r.get("match"), r.get("pair_cos", pc.get(str(r.get("key")))), copy_cos)
            if hit:
                out.append(dict(hit, key=r.get("key"), where="intake"))
    return out


def copy_hit(reason, match, pair_cos, copy_cos=COPY_COS):
    """{reason, split, pair_cos} when a refusal names a copy of a cwd12 dev or test image at the calibrated copy
    threshold (D28-v2), else None: an embedding hit; a dHash hit at pair cosine >= the threshold, or without one
    (fail closed)."""
    r = str(reason or "")
    base = r[len("masked_"):] if r.startswith("masked_") else r
    if base not in EVAL_COPY_REASONS:
        return None
    m = match if isinstance(match, dict) else {}
    split = m.get("eval_split") or m.get("split")
    if split is None:
        split = "unknown"            # the refusal's match was not kept: fail closed (D28-v2), counted as a copy
    elif split not in CWD12_SPLITS:
        return None
    if base == "near_eval_embed":
        return {"reason": r, "split": split, "pair_cos": m.get("cos")}
    if pair_cos is None or float(pair_cos) >= float(copy_cos):
        return {"reason": r, "split": split, "pair_cos": pair_cos}
    return None


def recover_box_src(source, rows, inc=None, cfg_name="weed"):
    """({key: [src id per box]}, record): each candidate's boxes traced to their source classes by reading the
    source's own label files again (DS.5.1 item 2: each source class one role), through the collector's own reader
    (collect.intake's tree from the staging blobs, collect.normalize.read with the fetch's options) in a temporary
    directory; a row is traced only when its boxes, mapped by the source's class map, equal its intake label's
    classes box for box. Rows not traced keep their INC class as their unit (the record says why)."""
    import tempfile
    from ..collect import config as CC
    from ..collect import intake as CI
    from ..collect import normalize as NZ
    from .. import collect as CL
    rec = {"source": source, "rows": len(rows), "traced": 0}
    try:
        cfg = CC.load(cfg_name)
        sdir = CL.staging_dir(source, inc)
        fetch_doc, fetch_sha = CI.read_fetch(sdir)
        missing = [f["name"] for f in fetch_doc.get("files") or [] if not (sdir / "blobs" / f["sha256"]).is_file()]
        if missing:
            return {}, dict(rec, why="%d staging blob(s) gone (e.g. %s)" % (len(missing), missing[:2]))
        cmap = intake_class_map(source, inc)
        inc_of = {k: v.get("inc_id") for k, v in cmap.items()}
        with tempfile.TemporaryDirectory(prefix="ds1_srcclass_", dir=os.environ.get("LOCAL") or None) as tmp:
            root = Path(tmp) / "x"
            CI.materialise(sdir, fetch_doc, root)
            _tree, res = NZ.read(root, CI._read_options(cfg, fetch_doc), out_images=Path(tmp) / "parquet_images")
            parsed = {it["rel"]: [str(b[0]) for b in it["boxes"]] for it in res.items}
    except Exception as e:  # noqa: BLE001 - a source that cannot be read again keeps its INC units, recorded
        return {}, dict(rec, why="%s: %s" % (type(e).__name__, str(e)[:300]))
    out, bad = {}, collections.Counter()
    for r in rows:
        got = parsed.get(r.get("rel"))
        if got is None:
            bad["rel not in the source as read now"] += 1
            continue
        if len(got) != len(r["boxes"]):
            bad["another number of boxes"] += 1
            continue
        if any(inc_of.get(s) is not None and int(inc_of[s]) != int(b[0]) for s, b in zip(got, r["boxes"])):
            bad["a box's class maps to another INC class"] += 1
            continue
        out[r["key"]] = got
    return out, dict(rec, traced=len(out), not_traced=dict(bad), fetch_sha256=fetch_sha)


# ------------------------------------------------------------------ the steps
def ambiguous_ids(cmap):
    """The INC class ids that several source classes of a source map to (each source class one role, DS.5.1 item 2:
    their boxes must be traced to their source classes)."""
    by = collections.defaultdict(set)
    for sid, v in (cmap or {}).items():
        if v.get("inc_id") is not None:
            by[int(v["inc_id"])].add(str(sid))
    return {i for i, ss in by.items() if len(ss) > 1}


def set_box_src(rows, cmap, traced):
    """Each row's source class per box (row['box_src']): its traced classes (traced {key: [src id per box]}), else for
    each box the one source class its INC class stands for, None where several do (that box's unit stays its INC
    class: one unit for all the names it stands for, shown on the sheet). No class map: nothing set."""
    if not cmap:
        return
    one = {}
    for sid, v in cmap.items():
        if v.get("inc_id") is not None:
            one.setdefault(int(v["inc_id"]), set()).add(str(sid))
    uniq = {i: next(iter(ss)) for i, ss in one.items() if len(ss) == 1}
    for r in rows:
        if r["key"] in traced:
            r["box_src"] = [str(x) for x in traced[r["key"]]]
        else:
            r["box_src"] = [uniq.get(int(b[0])) for b in r["boxes"]]


def screen(phase, inputs, revision=None, write=True):
    """Step 'screen' (L30): every candidate of the phase not covered by an earlier screen revision (phase 2: only the
    sources DS.5.2 names, PHASE2_SOURCES), per source: eligibility counts; the class units (each box's source class,
    traced from the source's own label files where its INC class stands for several), their source names and
    suggested roles; the capture-name scan of every file name (a source with a cwd12 or 3SeasonWeedDet10 capture name
    is role C, DS.5.1 item 1(b), its sheet showing the pairs); the cwd12 copy screen (D28-v2 unchanged: any copy of a
    cwd12 dev or test image at the calibrated copy threshold quarantines the source whole: its intake's and Step 1's
    refusals as copies and DS1's guard on its live rows, a dHash hit counting only where the DINOv2 detector
    confirms it, fail closed without the detector); the model audit, the box convention; the candidates written for
    the next steps; the sheets' bundles (inc2.ds1_sheets.bundle). A revision is written whenever it screens new rows
    or a Step 1 batch came since the last one (its record names the batches it saw), so the platform's plan sees
    the screen done even when no row is new. A source screened before DS1 deviation 2 with rows its code reads
    otherwise (rescreen_due: SIU, held out by its D31 quarantine, and the sources read with released intake holds) is
    screened again whole, once, and this revision's record of it replaces the earlier ones (deviation 2, item 2).
    Each live row's image size is read from its header where its file record gave none (fill_sizes), so its
    convention sees every box (DS1 deviation 2, item 1)."""
    from . import ds1_sheets as SH
    prior = screened_keys(phase)
    named = phase2_ids()
    again = set(rescreen_due(phase))
    rows = [r for r in inputs.candidates if (r["key"] not in prior or r["source"] in again)
            and (int(phase) == 1 or r["source"] in named)]
    ctx = eligibility_ctx(inputs)
    per = collections.defaultdict(lambda: {"rows": 0, "out": collections.Counter(), "units": collections.Counter(),
                                           "pools": collections.Counter()})
    live = []
    outs = collections.defaultdict(list)
    refusal_hits = collections.defaultdict(list)
    names_by = collections.defaultdict(list)
    for r in rows:
        why = eligibility(r, ctx)
        p = per[r["source"]]
        p["rows"] += 1
        p["pools"][r.get("pool")] += 1
        names_by[r["source"]].append(r.get("rel") or r["key"])
        h = copy_hit(r.get("refused"), r.get("guard_match"), r.get("pair_cos"))
        if h:
            refusal_hits[r["source"]].append(dict(h, key=r["key"], where="step1"))
        if why:
            p["out"][why] += 1
            if why != "in_base_b":
                outs[r["source"]].append(dict(r, why=why))
            continue
        live.append(r)
    # each box's source class: traced from the source's own files where its INC class stands for several
    cmaps, traces = {}, {}
    for s in sorted({r["source"] for r in live}):
        if s == SIU:
            continue
        cm = inputs.class_map.get(s) if s in inputs.class_map else (intake_class_map(s) if inputs.production else {})
        cmaps[s] = cm or {}
        amb = ambiguous_ids(cm)
        srows = [r for r in live if r["source"] == s]
        traced = {}
        if amb:
            need = [r for r in srows if any(int(b[0]) in amb for b in r["boxes"])]
            if inputs.box_src_fn is not None and need:
                traced, trec = inputs.box_src_fn(s, need)
            else:
                trec = {"source": s, "rows": len(need), "traced": 0, "why": "no reader of the source's files"}
            traces[s] = dict(trec, ambiguous_inc_ids=sorted(amb))
        set_box_src(srows, cm, traced)
        set_box_src(outs.get(s, []), cm, {})
    for r in live:
        for i, b in enumerate(r["boxes"]):
            per[r["source"]]["units"][unit_of(r, b[0], i)] += 1
    fill_sizes(live, inputs.read_size)
    hash_rows(live, hasher=inputs.hasher)
    live = [r for r in live if not r.get("drop")]
    # DS1's guard against cwd12 dev and test on the live rows: an embedding hit is a copy at the calibrated threshold;
    # a dHash hit only where the detector confirms it (its pair cosine reaches the threshold, D28-v2), every dHash hit
    # when no detector ran (fail closed); the dHash hits still drop their images at the pools' guard
    cwd = {k: v for k, v in inputs.exams.items() if k in CWD12_SPLITS}
    dh = dhash_hits(live, cwd)
    scanner = inputs.scanners.get("eval")
    eh = embed_hits(live, scanner, D.root() / "leak" / ("screen_p%d.npz" % phase)) if scanner else {}
    eh = {k: [h for h in v if h["exam"] in CWD12_SPLITS] for k, v in eh.items()}
    guard_hits = {k for k, v in eh.items() if v} | (set(dh) if scanner is None else set())
    unconfirmed = {k for k in dh if k not in guard_hits}
    by_src = collections.defaultdict(list)
    for r in live:
        by_src[r["source"]].append(r)
    rec = {}
    for s in sorted(per):
        rs = by_src.get(s, [])
        hits = sorted({r["key"] for r in rs if r["key"] in guard_hits})
        ih = (intake_cwd12_hits(s) if inputs.production else list((inputs.intake_hits or {}).get(s) or []))
        copies = [{"key": k, "where": "ds1_guard"} for k in hits] + refusal_hits.get(s, []) + ih
        names = capture_name_hits(names_by.get(s, []))
        base = {"rows": per[s]["rows"], "live": len(rs), "out": dict(per[s]["out"]), "pools": dict(per[s]["pools"]),
                "cwd12_hits": len(copies), "cwd12_hit_keys": [str(c.get("key")) for c in copies][:20],
                "cwd12_copies": {"ds1_guard": len(hits), "step1_refusals": len(refusal_hits.get(s, [])),
                                 "intake_refusals": len(ih),
                                 "dhash_unconfirmed": sum(1 for r in rs if r["key"] in unconfirmed)},
                "quarantine_whole": bool(copies),
                "capture_names": {"hits": len(names), "examples": names[:12]}, "role_c": bool(names)}
        if not rs:
            rec[s] = base
            continue
        audit = None
        if inputs.predict is not None and inputs.audit_ref is not None:
            audit = source_audit(rs, s, inputs.predict, inputs.audit_ref)
        inc_of = {k: v.get("inc_id") for k, v in (cmaps.get(s) or {}).items()}
        rec[s] = dict(base, units={u: {"boxes": n, "suggested": suggest_role(s, u, inc_of)}
                                   for u, n in sorted(per[s]["units"].items())},
                      audit_verdict=audit, source_classes=traces.get(s),
                      convention=B3.source_convention([dict(r, boxes=[b[1:5] for b in r["boxes"]]) for r in rs],
                                                      load_rules()[0]["source"]),
                      family=inputs.families.get(s), series=bool(any(r.get("series") for r in rs)),
                      unit_names=unit_names(s, sorted(per[s]["units"]), cmaps.get(s)))
    revs = screen_revisions(phase)
    revision = revision or (len(revs) + 1)
    seen = ((inputs.record or {}).get("step1_batches"))
    last_seen = ((_read_json(revs[-1]) or {}).get("inputs") or {}).get("step1_batches") if revs else None
    doc = {"format": FMT_SCREEN, "phase": int(phase), "revision": int(revision), "candidates": len(rows),
           "live": len(live), "sources": rec, "inputs": inputs.record, "decided_by": D.DECIDED_BY,
           "written_utc": D._utc(), "deviation": SCREEN_DEVIATION}
    if again:
        # DS1 deviation 2, item 2: the sources screened again whole; screen_records reads this revision's record of
        # each in place of the earlier ones, screened_rows its candidates in place of theirs
        doc["rescreen"] = {s: {"of": [p.name for p in revs if s in ((_read_json(p) or {}).get("sources") or {})],
                               "rows": per[s]["rows"] if s in per else 0, "deviation": 2,
                               "why": ("held out whole before DS1 deviation 2 by its D31 stream quarantine, which DS.0 "
                                       "item 14 reads past, and its queued rows read with their intake hold"
                                       if s == SIU else "its queued rows read before DS1 deviation 2 with an intake "
                                       "hold (h6_scan) the queue had released")} for s in sorted(again)}
    if write and (rows or not revs or sorted(seen or []) != sorted(last_seen or [])):
        base_p = out_dir() / "screen" / ("p%d_r%03d" % (int(phase), int(revision)))
        write_jsonl_once(Path(str(base_p) + "_candidates.jsonl"), [_cand_out(r) for r in rows])
        write_once(Path(str(base_p) + ".json"), json.loads(json.dumps(doc, default=str)))
        if inputs.production:
            if by_src:
                SH.bundle_sources(phase, revision, by_src, rec, out_rows=outs)
            if int(phase) == 1 and int(revision) == 1:
                bundle_ln(inputs)
                bundle_mtw_v1()
    else:
        doc["status"] = "unchanged: no new row and no new Step 1 batch since revision %d" % len(revs)
    return doc


def unit_names(source, units, cmap=None):
    """{unit: [source class names]} for the sheet: a source class's own name ('src:<id>'); every name an INC class
    stands for ('inc:<id>': the intake's class map, several names when several source classes map to it); SIU's EPPO
    codes' binomials from the collector's pinned EPPO table."""
    out = collections.defaultdict(set)
    if cmap is None:
        cmap = intake_class_map(source) if source != SIU else {}
    for sid, v in (cmap or {}).items():
        out["src:%s" % sid].add(str(v.get("name")))
        if v.get("inc_id") is not None:
            out["inc:%d" % int(v["inc_id"])].add(str(v.get("name")))
    eppo = {}
    if any(u.startswith("eppo:") for u in units):
        try:
            t = json.loads((Path(__file__).resolve().parents[1] / "collect" / "domains" / "eppo_codes_v2.json")
                           .read_text())
            codes = t.get("codes") if isinstance(t, dict) else None
            eppo = {str(k): (v.get("name") if isinstance(v, dict) else v) for k, v in (codes or t).items()} \
                if isinstance(codes or t, dict) else {}
        except (OSError, ValueError):
            eppo = {}
    for u in units:
        if u.startswith("eppo:") and eppo.get(u[5:]):
            out[u].add(str(eppo[u[5:]]))
    return {u: sorted(out.get(u, ())) for u in units}


def _cand_out(r):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items() if not k.startswith("_")}


def bundle_ln(inputs):
    """The LN sheet's bundle (DS.8.2): the SIU increments inc0001, inc0003 and inc0004 pooled, inc0005, and cwd12
    train_core; each image with its boxes numbered, species, source and file name hidden."""
    from . import ds1_sheets as SH
    from . import stream as ST
    f = ST.Stream(SH.LN_STREAM, quiet=True).load(repair=False)
    groups = {}
    for g, incs in SH.LN_INCREMENTS.items():
        rows = []
        for inc in incs:
            for k, r in f.inc_rows(inc).items():
                img = r.get("image")
                if img and r.get("label"):
                    rows.append({"key": k, "image": img, "label": r["label"], "sha256": r.get("sha256") or k})
        groups[g] = rows
    groups["train_core"] = [{"key": r["key"], "image": r["image"], "label": r["label"], "sha256": r["sha256"]}
                            for r in C.read_manifest(C2.v2_manifest_path("train_core"))]
    items = []
    for r in SH.ln_draw(groups):
        boxes = [["box"] + [float(x) for x in b[1:5]] for b in read_inc_label(r["label"])]
        items.append({"id": "ln|%s" % r["sha256"], "ln_group": r["ln_group"], "key": r["key"], "image": r["image"],
                      "boxes": boxes, "caption": ""})
    return SH.write_bundle("ln_001", items, {"kind": "ln", "groups": {g: len(v) for g, v in groups.items()}})


def bundle_mtw_v1():
    """The MT-W check's bundles of test v1 without its SIU stratum (DS.10 item 4): per source min(60, all) seeded
    images plus every image whose stored verifier verdicts flag a box."""
    from . import ds1_sheets as SH
    flags = verifier_flags()
    made = []
    for f in sorted((B3.out_dir() / B3.HOLDOUT_DIR).glob("*.jsonl")):
        rows = read_jsonl(f)
        if not rows or rows[0].get("source") == SIU:
            continue
        s = rows[0]["source"]
        pick, whole = SH.mtw_sample(s, rows, [r["key"] for r in rows if flags.get(r["key"])])
        items = [{"id": "mtw|%s|%s" % (s, r["key"]), "source": s, "key": r["key"], "image": r["image"],
                  "boxes": [["weed"] + [float(x) for x in b[1:5]] for b in read_inc_label(r["label"])],
                  "flagged": bool(flags.get(r["key"])), "caption": r["key"], "total": len(rows)} for r in pick]
        made.append(SH.write_bundle("mtw_v1_%s" % slug(s), items, {"kind": "mtw", "part": "test_v1", "source": s,
                                                                   "total": len(rows), "whole": whole}))
    return made


def bundle_mtw_v2(source, rows):
    from . import ds1_sheets as SH
    flags = verifier_flags()
    pick, whole = SH.mtw_sample(source, rows, [r["key"] for r in rows if flags.get(r["key"])])
    items = [{"id": "mtw|%s|%s" % (source, r["key"]), "source": source, "key": r["key"], "image": r["image"],
              "boxes": [["weed"] + [float(x) for x in b] for b in r["boxes"]], "flagged": bool(flags.get(r["key"])),
              "caption": r["key"], "total": len(rows)} for r in pick]
    return SH.write_bundle("mtw_v2_%s" % slug(source), items, {"kind": "mtw", "part": "mtw_v2", "source": source,
                                                                "total": len(rows), "whole": whole})


def verifier_flags(inc_dir=None):
    """{key: True} of the rows whose stored verifier verdicts (Step 1's admission records) flag a box: records,
    never a model read of an exam image (DS.7.5)."""
    from . import ds1_sheets as SH
    inc = Path(inc_dir) if inc_dir else C.INC_DIR
    out = {}
    for d in sorted((inc / "step1_stream" / "batches").glob("b*")) if (inc / "step1_stream" / "batches").is_dir() \
            else []:
        for a in read_jsonl(d / "admission.jsonl"):
            if any(v in SH.FLAG_VERDICTS for v in a.get("verdicts") or ()):
                out[a["key"]] = True
    return out


def screen_revisions(phase):
    return sorted((out_dir() / "screen").glob("p%d_r[0-9][0-9][0-9].json" % int(phase))) \
        if (out_dir() / "screen").is_dir() else []


def reaudit_path(rev_path):
    """screen/p<ph>_r<rev>_audit_v2.json beside a screen revision: its model audit taken again (Revision 2, item 4)."""
    return Path(str(rev_path)[:-5] + "_audit_v2.json")


def reaudit_sheet_path(rev_path, inc=None):
    """The same record under ds1/v1/sheets/audit_v2/ (written with it), so the lab's pull of the sheets' state
    carries it to every later render of the sources it covers."""
    from . import ds1_sheets as SH
    return SH.sheets_dir(inc) / "audit_v2" / reaudit_path(rev_path).name


def audit_due(phase):
    """The screen revisions of a phase holding a model audit taken before Revision 2 (no version) and no record of
    it taken again: reaudit takes them before any pool reads an audit (Revision 2, item 4)."""
    out = []
    for p in screen_revisions(phase):
        d = _read_json(p) or {}
        old = [s for s, v in (d.get("sources") or {}).items()
               if v.get("audit_verdict") is not None and not audit_valid(v.get("audit_verdict"))]
        if old and not isinstance(_read_json(reaudit_path(p)), dict):
            out.append(p)
    return out


def screen_live_rows(rev_path, ctx):
    """{source: [rows]} of a screen revision's live rows as its screen held them: each candidate that passes
    eligibility now and that the screen did not drop (a file it could not hash)."""
    out = collections.defaultdict(list)
    for r in read_jsonl(Path(str(rev_path)[:-5] + "_candidates.jsonl")):
        if eligibility(r, ctx) or r.get("drop"):
            continue
        r["boxes"] = [tuple(b) for b in r.get("boxes") or ()]
        out[r["source"]].append(r)
    return out


def reaudit(phase, inputs, write=True):
    """The model audit taken again, as Revision 2 reads boxes, for every screen revision of a phase whose audit was
    taken before it (audit_due): each source the revision audited, its live rows' seeded sample (audit_sample: the
    same draw), E1-A's predictions against train_core's reference under the same predictor. Written once per
    revision (screen/p<ph>_r<rev>_audit_v2.json and its copy under ds1/v1/sheets/audit_v2/); screen_records reads
    it in place of the revision's own audit, which the record keeps beside it (superseded)."""
    if inputs.predict is None or inputs.audit_ref is None:
        raise DataRefused("the model audit is taken again only with E1-A's predictor and train_core's reference")
    wsha = getattr(inputs.predict, "weights_sha256", None)
    if (wsha is not None or inputs.production) and wsha != AUDIT_PREDICTOR_SHA256:
        raise DataRefused("the model audit is taken again with the screen's predictor, E1-A seed 0 (weights sha256 "
                          "%s...), not weights %s" % (AUDIT_PREDICTOR_SHA256[:12], wsha))
    ctx = eligibility_ctx(inputs)
    done = []
    for p in audit_due(phase):
        d = _read_json(p) or {}
        live = screen_live_rows(p, ctx)
        srcs = {}
        for s, v in sorted((d.get("sources") or {}).items()):
            old = v.get("audit_verdict")
            if old is None or audit_valid(old):
                continue
            rs = live.get(s, [])
            new = source_audit(rs, s, inputs.predict, inputs.audit_ref)
            # the rows the screen held: eligibility is read now, so a row a later quarantine, never-train entry or
            # stream consumed is out; each source's record says whether its live count is the screen's, and its
            # sample's keys, so any difference from the screen (and from Revision 2's check, R2) is on record
            srcs[s] = {"audit_verdict": new, "superseded": old, "live_then": v.get("live"), "live_now": len(rs),
                       "rows_as_screened": len(rs) == int(v.get("live") or 0),
                       "sample_keys": [str(r["key"]) for r in audit_sample(rs, s)],
                       "partial_changed": bool(old.get("partial")) != bool(new.get("partial"))}
        doc = {"format": FMT_REAUDIT, "phase": int(phase), "revision": d.get("revision"), "screen": p.name,
               "screen_sha256": C.sha256_file(p), "version": AUDIT_VERSION,
               "why": "Revision 2, item 4: the screen's audit read each labelled box as [class, cx, cy, w]; taken "
                      "again with the class stripped, as train_core's reference reads its boxes",
               "predictor_weights_sha256": wsha,
               "rows_differ": sorted(s for s, x in srcs.items() if not x["rows_as_screened"]),
               "train_core": inputs.audit_ref, "sources": srcs, "decided_by": D.DECIDED_BY, "written_utc": D._utc()}
        if write:
            write_once(reaudit_path(p), doc)
            sp = reaudit_sheet_path(p)
            if not sp.exists():
                write_once(sp, doc)
        done.append(doc)
    return done


def screened_keys(phase):
    """The keys every screen revision of this phase and of every earlier phase covered."""
    out = set()
    for p in [x for ph in PHASES if ph <= int(phase) for x in screen_revisions(ph)]:
        for r in read_jsonl(Path(str(p)[:-5] + "_candidates.jsonl")):
            out.add(r["key"])
    return out


def screened_rows(phase):
    """Every screen revision's candidates of a phase, each key once: a key a later revision screened again (DS1
    deviation 2, item 2) is that revision's row, in the place the key first held."""
    by = {}
    for p in screen_revisions(phase):
        for r in read_jsonl(Path(str(p)[:-5] + "_candidates.jsonl")):
            by[r["key"]] = r
    out = list(by.values())
    for r in out:
        r["boxes"] = [tuple(b) for b in r.get("boxes") or ()]
    return out


def rescreen_due(phase):
    """The sources the next screen of a phase takes again whole (DS1 deviation 2, item 2): each source of RESCREEN that
    a revision of phase 1 written before the deviation screened (its record names no deviation: SIU held out by its D31
    quarantine, which DS.0 item 14 reads past, and queued rows held by an intake hold the queue had released) and that
    no revision screened again (once only). Phase 2 reads only its named sources."""
    if int(phase) != 1:
        return []
    seen, again = set(), set()
    for p in screen_revisions(phase):
        d = _read_json(p) or {}
        again |= set(d.get("rescreen") or {})
        if int(d.get("deviation") or 0) < SCREEN_DEVIATION:
            # a revision written before deviation 2 (its code read SIU's quarantine and the queued rows' holds so)
            seen |= {s for s in (d.get("sources") or {}) if s in RESCREEN}
    return sorted(seen - again)


def screen_records(phase, reaudits=None):
    """{source: record} over every screen revision of a phase: the latest revision's record of the source, with the
    counts summed and the flags that settle a source taken from any revision (a cwd12 copy quarantines it whole
    whichever revision found it; a capture name sends it to role C). A revision that screened a source again whole
    (its 'rescreen', DS1 deviation 2 item 2) replaces the earlier revisions' record of it."""
    out = {}
    for p in screen_revisions(phase):
        d = _read_json(p) or {}
        ra = (reaudits or {}).get(p.name) or _read_json(reaudit_path(p))
        ra = (ra.get("sources") or {}) if isinstance(ra, dict) and ra.get("format") == FMT_REAUDIT else {}
        again = d.get("rescreen") or {}
        for s, v in (d.get("sources") or {}).items():
            if s in ra:
                # the audit taken again (Revision 2, item 4) in place of the revision's own, which stays beside it
                v = dict(v, audit_verdict=ra[s]["audit_verdict"], audit_superseded=ra[s].get("superseded"))
            if s in again:
                out[s] = dict(v, rescreen_of=list((again[s] or {}).get("of") or []))
                continue
            old = out.get(s)
            if old is None:
                out[s] = dict(v)
                continue
            new = dict(old)
            new.update({k: x for k, x in v.items() if k in ("audit_verdict", "convention", "family", "series",
                                                            "source_classes") and x is not None})
            for k in ("rows", "live", "cwd12_hits"):
                new[k] = int(old.get(k) or 0) + int(v.get(k) or 0)
            new["quarantine_whole"] = bool(old.get("quarantine_whole") or v.get("quarantine_whole"))
            new["role_c"] = bool(old.get("role_c") or v.get("role_c"))
            cn_o, cn_v = old.get("capture_names") or {}, v.get("capture_names") or {}
            new["capture_names"] = {"hits": int(cn_o.get("hits") or 0) + int(cn_v.get("hits") or 0),
                                    "examples": (list(cn_o.get("examples") or []) + list(cn_v.get("examples") or []))[:12]}
            units = dict(old.get("units") or {})
            for u, x in (v.get("units") or {}).items():
                units[u] = dict(x, boxes=int((units.get(u) or {}).get("boxes") or 0) + int(x.get("boxes") or 0))
            new["units"] = units
            names = dict(old.get("unit_names") or {})
            names.update(v.get("unit_names") or {})
            new["unit_names"] = names
            out[s] = new
    return out


def source_ready(source, decisions, screen_rec):
    """(ok, why) for a source's rows to enter a manifest (DS.0 item 4, DS.5.1 items 1 and 2): not role C by its
    capture names (DS.5.1 item 1(b)) and not quarantined whole by the cwd12 copy screen (item 1(a)), whatever its
    sheet says; its sheet decided accept or accept for training only (never reject), audited where the owner's audit
    applies."""
    srec = screen_rec or {}
    if srec.get("role_c") or int((srec.get("capture_names") or {}).get("hits") or 0):
        return False, "role C: its file names carry cwd12 or 3SeasonWeedDet10 capture names (DS.5.1 item 1(b))"
    if srec.get("quarantine_whole"):
        return False, "quarantined whole (a copy of a cwd12 dev or test image, D28-v2)"
    d = decisions.get(source)
    if not d:
        return False, "sheet pending"
    if d.get("decision") == "reject":
        return False, "rejected"
    if d.get("decision") not in ("accept", "accept_training_only"):
        return False, "decision %r" % d.get("decision")
    if d.get("audit_required") and not d.get("audited"):
        return False, "owner audit pending"
    a = srec.get("audit_verdict")
    if d.get("decision") == "accept" and audit_valid(a) and a.get("partial"):
        return False, ("partially labelled by the model audit (DS.5.1 item 4): admitted only as 'accept for training "
                       "only', so its sheet is decided again")
    return True, ""


def settled_out(source, decisions, screen_rec):
    """True for a source that enters no DS1 manifest whatever happens next: role C by its capture names, quarantined
    whole, or rejected (its last decision governs: the owner's after his audit)."""
    srec = screen_rec or {}
    if srec.get("role_c") or int((srec.get("capture_names") or {}).get("hits") or 0) or srec.get("quarantine_whole"):
        return True
    return (decisions.get(source) or {}).get("decision") == "reject"


def sources_out(decisions=None):
    """The screened sources that enter no DS1 manifest, MD or MT-W now (DS.0 item 4: a source the owner rejects
    leaves every DS1 manifest, MD and MT-W; role C and wholly quarantined sources likewise), whatever was frozen
    before: their MD-N and MT-W v2 are never scored and never in MT-W's headline (their rows stay never-train)."""
    from . import ds1_sheets as SH
    decs = decisions if decisions is not None else SH.decided_sources()
    out = set()
    for ph in PHASES:
        for s, v in screen_records(ph).items():
            if settled_out(s, decs, v):
                out.add(s)
    return out


def _pool_paths(phase, source):
    d = out_dir() / "pools" / ("p%d" % int(phase))
    return d / ("%s.json" % slug(source)), d / ("%s.jsonl" % slug(source))


def pools_superseded_due(phase):
    """The sources of a phase whose pool record a recorded deviation supersedes still stands at its path, hashing as
    named (SUPERSEDED_POOLS; DS1 deviation 2, item 1): the plan reads them as not pooled, and the pools step moves
    them aside first (supersede_pools)."""
    out = []
    for e in SUPERSEDED_POOLS:
        rec, _rows = _pool_paths(e["phase"], e["source"])
        if int(e["phase"]) == int(phase) and rec.is_file() and not rec.is_symlink() and \
                C.sha256_file(rec) == e["record_sha256"]:
            out.append(e["source"])
    return sorted(out)


def supersede_pools(phase):
    """The pool records of a phase that a recorded deviation supersedes (SUPERSEDED_POOLS; DS1 deviation 2, item 1),
    moved aside byte for byte into pools/p<phase>/superseded/dev<n>/: for each, its record (<slug>.json, hashing as
    named) and its rows (<slug>.jsonl, which must hash as named), each a hard link made there and then the original
    unlinked, the rows first and the record last, so a move cut off midway is still due and completes; never
    overwritten, never deleted. Each move is recorded in that folder's MANIFEST.json (the files' paths, sha256s and
    sizes, when, by which job and host, why, the deviation's file and sha256). A record at the path with other bytes
    is not the one named and stays (the source pooled again: moved_before; else other). A folder copy with other
    bytes, or rows at the path with other bytes than named, refuse. Returns [{source, status: moved | moved_before |
    other | absent}]."""
    out = []
    for e in SUPERSEDED_POOLS:
        if int(e["phase"]) != int(phase):
            continue
        rec, rows = _pool_paths(phase, e["source"])
        dd = rec.parent / "superseded" / ("dev%d" % int(e["deviation"]))
        named = ((rows, e["rows_sha256"]), (rec, e["record_sha256"]))
        for src, sha in named:
            dst = dd / src.name
            if dst.is_file() and C.sha256_file(dst) != sha:
                raise DataRefused("%s holds another file than the one DS1 deviation %d supersedes (%s...)" % (
                    dst, int(e["deviation"]), sha[:12]))
        got = C.sha256_file(rec) if rec.is_file() and not rec.is_symlink() else None
        if got != e["record_sha256"]:
            st_ = "moved_before" if (dd / rec.name).is_file() else ("other" if got is not None else "absent")
            out.append({"source": e["source"], "status": st_})
            continue
        dv = next((d for d in deviations() if int(d["n"]) == int(e["deviation"]) and d.get("sha256")), None)
        if dv is None:
            raise DataRefused("the superseded pool record of %s names DS1 deviation %s, which no pinned deviation "
                              "record holds" % (e["source"], e["deviation"]))
        if rows.is_file() and C.sha256_file(rows) != e["rows_sha256"]:
            raise DataRefused("%s does not hash as the rows DS1 deviation %d names (%s...): its record is not moved" % (
                rows, int(dv["n"]), e["rows_sha256"][:12]))
        if not rows.is_file() and not (dd / rows.name).is_file():
            raise DataRefused("%s's rows (%s) are neither at their path nor moved aside: its record is not moved" % (
                e["source"], rows.name))
        dd.mkdir(parents=True, exist_ok=True)
        files = []
        for src, sha in named:
            dst = dd / src.name
            if src.is_file():
                if not dst.is_file():
                    os.link(str(src), str(dst))
                size = src.stat().st_size
                os.unlink(str(src))
            else:
                size = dst.stat().st_size          # moved by a run cut off after its link and unlink
            files.append({"path": str(src.relative_to(C.INC_DIR)) if str(src).startswith(str(C.INC_DIR)) else str(src),
                          "sha256": sha, "size": size, "moved_to": str(dst.relative_to(C.INC_DIR))
                          if str(dst).startswith(str(C.INC_DIR)) else str(dst)})
        man = dd / "MANIFEST.json"
        doc = _read_json(man) if man.is_file() else None
        doc = doc if isinstance(doc, dict) and isinstance(doc.get("moves"), list) else {"format": D.FMT_SUPERSEDED,
                                                                                         "moves": []}
        doc["moves"].append({"source": e["source"], "phase": int(phase), "files": files, "moved_utc": D._utc(),
                             "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "why": e["why"],
                             "deviation": {"n": int(dv["n"]), "file": dv["file"], "sha256": dv["sha256"]}})
        D.write_json(man, doc)
        log("superseded the pool of %s (DS1 deviation %d): moved to %s" % (e["source"], int(dv["n"]), dd))
        out.append({"source": e["source"], "status": "moved"})
    return out


def pools(phase, inputs, s0_rows=None, write=True):
    """Step 'pools' (L30): for every screened source whose sheet is decided (and audited where required), the rows
    through DS.5.1 items 2-10: roles, base3's rules (each row's image size read from its header where its file
    record gave none: DS1 deviation 2, item 1), the model audit's masks, dedupe against base B and among
    themselves, thin(10) (kept flags; KD reads the unthinned), the guard against the fixed exams (both
    directions), EXIF, licence flags, the written files. Each source's pool is written once:
    splits/ds1/pools/p<phase>/<source slug>.jsonl; the pool records a recorded deviation supersedes are moved aside
    first (supersede_pools), so their sources are pooled again."""
    rules, _conf = load_rules()
    moved = supersede_pools(phase) if write else []
    taken = {}
    if audit_due(phase):
        # a screen's audit taken before Revision 2 is taken again before any pool reads it (masks, readiness)
        taken = {x["screen"]: x for x in reaudit(phase, inputs, write=write)}
        if audit_due(phase) and write:
            raise DataRefused("a screen revision's model audit is still the one taken before Revision 2")
    rows = screened_rows(phase)
    srec = screen_records(phase, reaudits=taken)
    ready = {}
    for s in sorted({r["source"] for r in rows}):
        ok, why = source_ready(s, inputs.decisions, srec.get(s))
        ready[s] = (ok, why)
    done = ({p.stem for p in (out_dir() / "pools" / ("p%d" % int(phase))).glob("*.jsonl")} -
            {slug(x) for x in pools_superseded_due(phase)}) if write else set()
    todo = [r for r in rows if ready.get(r["source"], (False,))[0] and slug(r["source"]) not in done]
    ctx = eligibility_ctx(inputs)
    counts = collections.defaultdict(collections.Counter)
    live = []
    for r in todo:
        why = eligibility(r, ctx)
        if why:
            counts[r["source"]][why] += 1
            continue
        dec = inputs.decisions[r["source"]]
        keep, species, dropped, unres = weed_boxes(r, dec.get("roles") or {})
        if unres:
            counts[r["source"]]["unresolved_class"] += 1
            continue
        r = dict(r, boxes=keep, species=species, dropped_boxes=dropped, drop=None, kind="ds1", mask=[],
                 masked_area=0.0, family=inputs.families.get(r["source"]) or dec.get("family"),
                 setting=dec.get("setting"), training_only=dec.get("decision") == "accept_training_only")
        live.append(r)
    hash_rows(live, hasher=inputs.hasher)
    fill_sizes(live, inputs.read_size)
    conv = apply_rules(live, rules)
    # the model audit's masks for a partially labelled source (accept for training only)
    for r in live:
        if r.get("drop"):
            continue
        a = (srec.get(r["source"]) or {}).get("audit_verdict") or {}
        if a.get("partial") and not audit_valid(a):
            raise DataRefused("%s's model audit was taken before Revision 2: its masks are not applied from it"
                              % r["source"])
        if a.get("partial") and inputs.predict is not None:
            extra = unlabelled_masks(r, inputs.predict)
            if extra:
                r["mask"] = list(r.get("mask") or []) + extra
                r["masked_area"] = B3.masked_area(r["mask"], r["boxes"], r["W"], r["H"])
                if r["masked_area"] > rules["image"]["max_masked_area"]:
                    r["drop"] = "masked_over_50"
    dd = dedupe(live, [_base_dd(k, v) for k, v in inputs.base_b.items()], rules["dedupe"])
    s0_series = collections.defaultdict(list)
    for k, v in inputs.base_b.items():
        if v.get("source") == SIU:
            vid, _e = siu_video(v.get("original_image") or k)
            if vid:
                s0_series[vid].append(v.get("dhash"))
    th = thin_rows(live, s0_series)
    kept = [r for r in live if not r.get("drop")]
    fixed = {k: v for k, v in inputs.exams.items() if k != "ds1_exams"}
    g = merge_hits(dhash_hits(kept, fixed), *[embed_hits(kept, sc, D.root() / "leak" / ("pools_p%d_%s.npz" % (
        phase, name))) for name, sc in sorted(inputs.scanners.items())])
    for r in kept:
        if g.get(r["key"]):
            r["drop"] = "guard"
            r["guard_hits"] = g[r["key"]][:5]
    for r in kept:
        if r.get("drop"):
            continue
        why = exif_refusal(r, inputs.read_orientation)
        if why:
            r["drop"] = why
    out = collections.defaultdict(list)
    for r in live:
        if r.get("drop"):
            counts[r["source"]][r["drop"]] += 1
            continue
        if write:
            materialise(r)
            if r.get("drop"):
                counts[r["source"]][r["drop"]] += 1
                continue
            r["label"], r["label_sha256"] = write_label(r["boxes"])
        r["original_sha256"] = r.get("sha256")
        out[r["source"]].append(r)
    recs = {}
    for s in sorted({r["source"] for r in todo}):
        rs = out.get(s, [])
        recs[s] = {"admitted": len(rs), "thinned": sum(1 for r in rs if r.get("thin")), "out": dict(counts[s]),
                   "convention": conv.get(s), "counts": three_counts(rs),
                   "sizes_from_header": sum(1 for r in live if r["source"] == s and r.get("size_from") == "header")}
        if write:
            pd = out_dir() / "pools" / ("p%d" % int(phase))
            sha = write_jsonl_once(pd / ("%s.jsonl" % slug(s)), [_pool_row(r) for r in sorted(rs, key=lambda r: r["key"])])
            write_once(pd / ("%s.json" % slug(s)), {"format": FMT_POOL, "source": s, "phase": int(phase),
                                                   "rows": len(rs), "sha256": sha, "record": recs[s],
                                                   "written_utc": D._utc()})
    doc = {"format": FMT_POOL, "phase": int(phase), "sources": recs, "dedupe": dd, "thin": {"series": len(th)},
           "not_ready": {s: why for s, (ok, why) in ready.items() if not ok}, "written_utc": D._utc()}
    if moved:
        doc["superseded"] = moved
    return doc, out


def _base_dd(k, v):
    return {"key": k, "source": v.get("source"), "sha256": v.get("original_sha256"), "dhash": v.get("dhash"),
            "variants": v.get("variants"), "boxes": v.get("_boxes") or [], "W": v.get("W"), "H": v.get("H"),
            "bytes": None, "tier": 0}


def _pool_row(r):
    keep = ("key", "source", "pool", "image", "sha256", "original_sha256", "label", "label_sha256", "train_image",
            "train_sha256", "session", "series", "eppo", "capture", "stem", "dhash", "variants", "thin", "species",
            "licence", "research_only", "family", "setting", "training_only", "W", "H", "mask", "rel")
    out = {k: r.get(k) for k in keep}
    out["boxes"] = [list(b) for b in r["boxes"]]
    out["mask"] = [list(b) for b in r.get("mask") or []]
    return out


def pool_rows(phase):
    """{source: [rows]} of every written pool of a phase, an empty pool included (its record names its source; manifest
    fields: image is the file trained on)."""
    out = {}
    d = out_dir() / "pools" / ("p%d" % int(phase))
    for p in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
        rec = _read_json(p.with_suffix(".json")) or {}
        rs = read_jsonl(p)
        for r in rs:
            r["boxes"] = [tuple(b) for b in r["boxes"]]
            r["image"] = r.get("train_image") or r["image"]
            r["sha256"] = r.get("train_sha256") or r["sha256"]
        src = rec.get("source") or (rs[0]["source"] if rs else None)
        if src:
            out[src] = rs
    return out


def md_s0(inputs, power_rec=None, write=True):
    """Step 'md_s0' (L30): MD-S0 drawn from base B's external sources (base_v2 and SIU excluded; test v1's
    companions, evaluation groups and cwd12 copies never) by capture group, written as a draft in its own root for
    the power pre-flight; then, once the power record of that draft exists, enlarged in the same order while its
    MDE is over 0.02 (within the 30 % cap) or frozen (frozen.json, the never-train list). Never after its freeze."""
    if frozen("md_s0"):
        return {"status": "frozen", "frozen": frozen("md_s0")}
    universe = {k: dict(v, key=k) for k, v in inputs.base_b.items()
                if v.get("kind") != B3.BASE_KIND and v.get("source") != SIU and v.get("source") not in
                inputs.eval_sources and not v.get("holdout_v1")}
    rows = list(universe.values())
    for r in rows:
        r.setdefault("sha256", r.get("original_sha256"))
        if r.get("capture_keys"):
            r["capture"] = next((c[1] for c in r["capture_keys"] if c[0] == "capture"), None)
            st = next((c for c in r["capture_keys"] if c[0] == "stem"), None)
            if st:
                r["stem"], r["family"] = st[2], st[1]
    groups, digests = capture_groups(rows, exam_links(inputs, rows, "md_s0_universe"))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    drafts = sorted(exam_dir("md_s0").glob("draft_r[0-9][0-9][0-9]")) if exam_dir("md_s0").is_dir() else []
    last = drafts[-1] if drafts else None
    target = None
    if last is not None:
        prec = power_rec if power_rec is not None else _read_json(D.root() / "power" / ("%s.json" % last.name))
        if not isinstance(prec, dict):
            return {"status": "waiting for the power pre-flight of %s" % last.name}
        md = prec.get("md_s0") or {}
        if not md.get("enlarge"):
            return _freeze_md_s0(last, prec, write)
        prev = _read_json(last / "draw.json") or {}
        factor = (float(md["mde"]) / D.MDE_MAX) ** 2
        target = enlarge_targets(prev.get("per_source") or {}, factor)
        if all(target[s] <= (prev["per_source"][s]["md_images"]) for s in target):
            return _freeze_md_s0(last, prec, write, capped=True)
    draw = draw_exams(by, groups, digests, {s: len(v) for s, v in by.items()}, new_sources=(),
                      md_target=target)
    keys = [r["key"] for s in sorted(by) for r in by[s] if groups[r["key"]] in set(draw[s]["md"])]
    rev = len(drafts) + 1
    d = exam_dir("md_s0") / ("draft_r%03d" % rev)
    md_rows = []
    for k in sorted(keys):
        v = universe[k]
        md_rows.append({"key": k, "image": v["image"], "label": v["label"], "sha256": v["sha256"],
                        "label_sha256": v["label_sha256"], "source": v["source"], "session": "", "group": groups[k],
                        "original_sha256": v.get("original_sha256"), "dhash": v.get("dhash"),
                        "variants": v.get("variants"), "capture": v.get("capture"), "stem": v.get("stem"),
                        "family": v.get("family"), "train_sha256": v.get("sha256")})
    doc = {"format": FMT_EXAM, "exam": "md_s0", "revision": rev, "images": len(md_rows),
           "groups": len({r["group"] for r in md_rows}), "per_source": {s: {k: v for k, v in x.items()
                                                                              if k not in ("md", "mtw")}
                                                                        for s, x in draw.items()},
           "groups_by_source": {s: x["md"] for s, x in draw.items()}, "enlarged_from": last.name if last else None,
           "target": target, "written_utc": D._utc()}
    if write:
        rr = write_exam_root("md_s0", md_rows, d / "root", groups)
        doc["root"] = rr
        write_once(d / "draw.json", json.loads(json.dumps(doc, default=str)))
        write_jsonl_once(d / "groups.jsonl", [{"exam": "md_s0", "key": r["key"], "group": r["group"]}
                                               for r in md_rows])
        write_jsonl_once(d / "rows.jsonl", md_rows)
    return dict(doc, status="drawn")


def _freeze_md_s0(draft, prec, write, capped=False):
    doc = _read_json(draft / "draw.json") or {}
    rows = read_jsonl(draft / "rows.jsonl")
    groups = {r["key"]: r["group"] for r in rows}
    if not write:
        return {"status": "would freeze", "draft": draft.name}
    rec = freeze_record("md_s0", doc["root"], {"draft": draft.name, "power_mde": (prec.get("md_s0") or {}).get("mde"),
                                                "enlargement_capped": capped})
    if capped and not D.capacity("ds1_power.json").exists():
        # the MDE stayed over 0.02 with every source at its 30 % cap: the last draft's power record is the final one
        D.write_once(D.capacity("ds1_power.json"), dict(prec, enlargement_capped=True))
    append_never_train(rows, "md_s0", groups)
    _append_groups("md_s0", rows)
    return {"status": "frozen", "frozen": rec}


def _append_groups(exam, rows):
    p = out_dir() / "md_v1" / "groups.jsonl"
    have = {(r["exam"], r["key"]) for r in read_jsonl(p)}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        for r in rows:
            if (exam, r["key"]) not in have:
                fh.write(json.dumps({"exam": exam, "key": r["key"], "group": r["group"]}, sort_keys=True) + "\n")


def exam_links(inputs, rows, tag):
    """The DINOv2 links among rows that join capture groups (DS.5.4): computed (inputs.link_fn: the calibrated
    detector over the rows) or given (inputs.links)."""
    if inputs.link_fn is not None:
        return inputs.link_fn(rows, tag)
    return inputs.links or ()


def built_arms(inputs):
    """{arm: (manifest sha256, [unique rows with their hashes])} of every built arm (a floor's repeats dropped); each
    row's dHash and variants from base B's provenance or DS1's pools."""
    pool_by = {r["key"]: r for ph in PHASES for rs in pool_rows(ph).values() for r in rs}
    out = {}
    for a in D.BUILD_ARMS:
        if not arm_record_path(a).is_file():
            continue
        rows, rec = arm_rows(a)
        uniq = {}
        for r in rows:
            k = repeat_base(r["key"])
            if k in uniq:
                continue
            src = inputs.base_b.get(k) or pool_by.get(k) or {}
            uniq[k] = dict(r, key=k, dhash=src.get("dhash"), variants=src.get("variants"))
        out[a] = (rec.get("sha256"), list(uniq.values()))
    return out


def _hash_exam(name, rows):
    rs = [r for r in rows if r.get("dhash") is not None or r.get("variants") is not None]
    if not rs:
        return {}
    H, ok, _h = B3.hash_matrix(rs)
    import numpy as np
    return {name: {"keys": [r["key"] for r in rs], "H": H, "ok": np.asarray(ok, dtype=bool)}}


def draw_exclusions(rows, groups, inputs, tag):
    """(groups never drawn into MD-N or MT-W v2, groups never drawn into MT-W v2, record) of a source's rows
    (Revision 1 of DS1, item 1): a manifest built before the draw (S0 = B'o, KF, the K arms, later S1's) never
    changes, so the guard's direction from these exams to it (DS.5.1 item 7) is kept at the draw: a group with an
    image within 6 dHash bits (8 variants, both directions) of a built manifest's row, or one the calibrated DINOv2
    detector calls a copy of it, is drawn into neither; MT-W v2 also obeys MAIN_TEST.md MT.2 item 2: no group with an
    image within 6 bits of a row of base B (base_v2 among them) or of a stream's pool, or holding a file of base B."""
    arms = built_arms(inputs)
    ref = {}
    for a, (_sha, rs) in sorted(arms.items()):
        for r in rs:
            ref.setdefault(r["key"], r)
    hit_any = set(dhash_hits(rows, _hash_exam("manifests", list(ref.values()))))
    dino = {}
    if inputs.ref_scan_fn is not None:
        for a, (sha, rs) in sorted(arms.items()):
            got = inputs.ref_scan_fn(rows, rs, "arm_%s_%s" % (a.lower(), str(sha)[:12]))
            for k in got:
                dino.setdefault(k, a)
    hit_any |= set(dino)
    base = [{"key": k, "dhash": v.get("dhash"), "variants": v.get("variants")} for k, v in inputs.base_b.items()]
    base += list(inputs.pool_ref_fn() if inputs.pool_ref_fn is not None else [])
    base_shas = {v.get("original_sha256") for v in inputs.base_b.values()} | {v.get("sha256") for v in
                                                                           inputs.base_b.values()}
    hit_mtw = set(dhash_hits(rows, _hash_exam("base_b", base))) | {r["key"] for r in rows
                                                                    if r.get("sha256") in base_shas or
                                                                    r.get("original_sha256") in base_shas}
    ex = {groups[k] for k in hit_any if k in groups}
    ex_mtw = {groups[k] for k in hit_mtw if k in groups} - ex
    rec = {"tag": tag, "built_arms": {a: sha for a, (sha, _r) in sorted(arms.items())},
           "rows_near_built_manifests": len(hit_any), "rows_dinov2_copies": len(dino),
           "rows_near_base_b": len(hit_mtw), "groups_excluded": len(ex), "groups_excluded_mtw": len(ex_mtw)}
    return ex, ex_mtw, rec


def corrected_label(chk, key):
    """The corrected class-12 boxes of an MT-W image the person's check corrected (the image's shown boxes without
    the ones the person removed, with the ones drawn), or None."""
    got = (chk.get("corrected_labels") or {})
    if key not in got:
        return None
    return [tuple(float(x) for x in b[-4:]) for b in got[key]]


def draw(phase, inputs, checks=None, write=True):
    """Step 'draw' (L30): MD-N and MT-W v2 for every new non-SIU source whose sheet is 'accept' (never 'accept for
    training only') and which is ready now (audited where required, not rejected since, not role C, not quarantined)
    and whose pool is written: capture groups over its thinned rows (6-bit dHash, capture relations and the
    calibrated DINOv2 detector's links), the draw in the source's order with the groups that touch a built manifest
    (both) or base B (MT-W v2, MT.2 item 2) left out (draw_exclusions); MD-N frozen at once (its own root and LOCK),
    MT-W v2 written as a draft and frozen only after the person's check (checks {source: the recorded check}), its
    corrected labels the person's whole corrected boxes; every drawn row enters the never-train list."""
    pr = pool_rows(phase)
    srec = screen_records(phase)
    out = {}
    for s, rows in sorted(pr.items()):
        dec = inputs.decisions.get(s) or {}
        ok, why = source_ready(s, inputs.decisions, srec.get(s))
        if s == SIU or dec.get("decision") != "accept" or not ok:
            out[s] = {"status": "not drawn (%s)" % ("SIU" if s == SIU else (why or dec.get("decision")))}
            continue
        mdn, mtw = mdn_exam(s), mtw_exam(s)
        if frozen(mdn) and (frozen(mtw) or not (exam_dir(mtw) / "draft").is_dir()):
            out[s] = {"status": "frozen"}
            continue
        thin_rows_ = [r for r in rows if r.get("thin")]
        if not thin_rows_:
            # an empty pool (every row dropped by a rule): nothing to draw; recorded so, an empty MD-N, no MT-W v2
            if write and not frozen(mdn):
                write_once(exam_dir(mdn) / "frozen.json", {"format": FMT_EXAM, "exam": mdn, "frozen": True,
                                                           "images": 0, "root": None, "source": s,
                                                           "frozen_utc": D._utc()})
            out[s] = {"status": "frozen (empty pool)"}
            continue
        rec = {}
        if not frozen(mdn) or not (exam_dir(mtw) / "draft").is_dir():
            groups, digests = capture_groups(thin_rows_, exam_links(inputs, thin_rows_, "draw_%s" % s))
            ex, ex_mtw, xrec = draw_exclusions(thin_rows_, groups, inputs, s)
            d = draw_exams({s: thin_rows_}, groups, digests, {s: len(rows)}, new_sources={s}, excluded_groups=ex,
                           mtw_excluded_groups=ex_mtw)[s]
            md_keys = {r["key"] for r in thin_rows_ if groups[r["key"]] in set(d["md"])}
            mt_keys = {r["key"] for r in thin_rows_ if groups[r["key"]] in set(d["mtw"])}
            rec = {"md": len(md_keys), "mtw": len(mt_keys), "groups": len(set(groups.values())), "draw": d,
                   "exclusions": xrec}
            if write and not frozen(mdn):
                mrows = [dict(r, group=groups[r["key"]]) for r in thin_rows_ if r["key"] in md_keys]
                if mrows:
                    rr = write_exam_root(mdn, mrows, exam_dir(mdn) / "root", groups)
                    freeze_record(mdn, rr, {"source": s, "exclusions": xrec})
                    append_never_train(mrows, mdn, groups)
                    _append_groups(mdn, mrows)
                else:
                    # a source with no group to draw (every group over 500 images, past the 30 % cap or left out):
                    # recorded so, an empty exam that no arm is scored on
                    write_once(exam_dir(mdn) / "frozen.json", {"format": FMT_EXAM, "exam": mdn, "frozen": True,
                                                               "images": 0, "root": None, "source": s,
                                                               "exclusions": xrec, "frozen_utc": D._utc()})
                rec["md_status"] = "frozen"
            if write and not (exam_dir(mtw) / "draft").is_dir():
                trows = [dict(r, group=groups[r["key"]]) for r in thin_rows_ if r["key"] in mt_keys]
                if trows:
                    write_jsonl_once(exam_dir(mtw) / "draft" / "rows.jsonl", trows)
                    append_never_train(trows, mtw, groups)
                    bundle_mtw_v2(s, trows)
                rec["mtw_status"] = "draft (waits for the person's check)"
        chk = (checks or {}).get(s)
        if write and chk and chk.get("verdict") == "fail" and not (exam_dir(mtw) / "left.json").is_file():
            # DS.10 item 4: a failing source leaves MT-W unless a person checks it image by image; its rows stay
            # never-train (they are on the list since the draw)
            write_once(exam_dir(mtw) / "left.json", {"format": FMT_EXAM, "exam": mtw, "left": True, "source": s,
                                                     "check": chk.get("sha256"), "written_utc": D._utc()})
            rec["mtw_status"] = "left MT-W (the check failed); its rows stay never-train"
        if write and chk and chk.get("verdict") == "pass" and not frozen(mtw) and (exam_dir(mtw) / "draft").is_dir():
            trows = read_jsonl(exam_dir(mtw) / "draft" / "rows.jsonl")
            for r in trows:
                cb = corrected_label(chk, r["key"])
                if cb is not None:
                    r["label"], r["label_sha256"] = write_label(cb)
                    r["corrected"] = True
            rr = write_exam_root("mtw_v2", trows, exam_dir(mtw) / "root", {r["key"]: r["group"] for r in trows})
            freeze_record(mtw, rr, {"source": s, "check": chk.get("sha256"),
                                    "corrected": sum(1 for r in trows if r.get("corrected"))})
            rec["mtw_status"] = "frozen"
        out[s] = rec
    if write and int(phase) == 1:
        out["mtw_v1_1"] = freeze_mtw_v1_1(checks or {})
    return out


def freeze_mtw_v1_1(checks):
    """MT-W v1.1 (DS.4, DS.10 item 4): once every test v1 source but SIU has a recorded check, test v1 without SIU
    as checked is frozen in its own root and LOCK whenever the check changed anything (test v1's own files never
    change): the labels the person corrected (each image's whole corrected boxes) and no row of a source that failed
    its check (it leaves MT-W; its rows stay never-train). MT-W's headline then reads v1.1, test v1 a stratum beside
    it. Written once."""
    d = out_dir() / "mtw_v1_1"
    if (d / "status.json").is_file():
        return _read_json(d / "status.json")
    files = sorted((B3.out_dir() / B3.HOLDOUT_DIR).glob("*.jsonl"))
    srcs = {}
    for f in files:
        rows = read_jsonl(f)
        if rows and rows[0].get("source") != SIU:
            srcs[rows[0]["source"]] = rows
    if not srcs or any(s not in checks for s in srcs):
        return {"status": "waiting for the MT-W check of %s" % sorted(s for s in srcs if s not in checks)}
    left = sorted(s for s in srcs if checks[s].get("verdict") != "pass")
    corrected = {}
    for s in srcs:
        for k in (checks[s].get("corrected_labels") or {}):
            corrected[k] = corrected_label(checks[s], k)
    rec = {"format": FMT_EXAM, "exam": "mtw_v1_1", "left_mtw": left, "corrections": len(corrected),
           "checks": {s: checks[s].get("sha256") for s in srcs}, "written_utc": D._utc()}
    if corrected or left:
        rows = []
        for s, rs in sorted(srcs.items()):
            if s in left:
                continue
            for r in rs:
                if r["key"] in corrected:
                    lab, lsha = write_label(corrected[r["key"]])
                    r = dict(r, label=lab, label_sha256=lsha)
                rows.append(dict({k: r.get(k) for k in C.MANIFEST_KEYS}, session=r.get("session") or ""))
        rec["root"] = write_exam_root("mtw_v2", rows, d / "root", {r["key"]: r["key"] for r in rows})
        rec["images"] = len(rows)
    write_once(d / "status.json", rec)
    return rec


def b_prime_o(inputs, md_rows, exam_rows, write=True):
    """B'o (DS.5.2, DS.0 item 16): D1's B' minus MD-S0's groups (every row sharing a capture relation with an MD-S0
    row), base_v2's 66 NDSU rows and the guard's hits in both directions (base_v2's rows that the detector flags
    against cwd12 dev or test are kept and listed, as V3F did); the rows of B' as they are, bytes unchanged."""
    bp = inputs.bprime
    prov = inputs.base_b
    rows = []
    for k, r in bp.items():
        p = prov.get(k) or {}
        rows.append(dict(r, original_sha256=p.get("original_sha256"), dhash=p.get("dhash"),
                         variants=p.get("variants"), group=p.get("group"), capture=next(
                             (c[1] for c in p.get("capture_keys") or () if c[0] == "capture"), None),
                         stem=next((c[2] for c in p.get("capture_keys") or () if c[0] == "stem"), None),
                         family=next((c[1] for c in p.get("capture_keys") or () if c[0] == "stem"), None),
                         kind_b=p.get("kind")))
    md_keys = {r["key"] for r in md_rows}
    gh = group_hits(rows, exam_rows)
    fixed = {k: v for k, v in inputs.exams.items()}
    dh = dhash_hits(rows, fixed)
    eh = merge_hits(*[embed_hits(rows, sc, D.root() / "leak" / ("bprime_%s.npz" % n))
                      for n, sc in sorted(inputs.scanners.items())])
    kept, out, listed = [], collections.Counter(), {}
    for r in rows:
        if r["key"] in md_keys:
            out["md_s0"] += 1
            continue
        if r["key"] in inputs.ndsu_keys:
            out["ndsu"] += 1
            continue
        if gh.get(r["key"]):
            out["capture_md_mtw"] += 1
            continue
        hits = dh.get(r["key"], []) + eh.get(r["key"], [])
        if hits:
            if r.get("kind_b") == B3.BASE_KIND and all(h["exam"] in ("dev", "test") for h in hits):
                listed[r["key"]] = hits[:5]
            else:
                out["guard"] += 1
                continue
        kept.append(r)
    return kept, {"removed": dict(out), "base_v2_kept_with_hits": listed, "rows": len(kept)}


def ready_pools(phase, inputs):
    """{source: rows} of a phase's written pools whose source is ready now (DS.0 item 4: a source the owner rejects,
    or one role C or quarantined whole, leaves every DS1 manifest whatever was pooled before)."""
    srec = screen_records(phase)
    return {s: rs for s, rs in pool_rows(phase).items() if source_ready(s, inputs.decisions, srec.get(s))[0]}


def base_a_hits(inputs, base_a):
    """{key: [hits]} of base A's rows the guard flags against the exams (the dHash guard on every exam the inputs
    hold, DS1's own MD and MT-W lists among them, and the DINOv2 detector of DS1's own index): kept in every K arm and
    listed (Revision 1 of DS1, item 4: K0 = E1-A trained on exactly base A, so every K arm holds them alike)."""
    rows = []
    for r in base_a:
        p = inputs.base_b.get(r["key"]) or {}
        rows.append(dict(r, dhash=p.get("dhash"), variants=p.get("variants")))
    hits = dhash_hits(rows, inputs.exams)
    sc = inputs.scanners.get("own")
    if sc is not None:
        hits = merge_hits(hits, embed_hits(rows, sc, D.root() / "leak" / "base_a_own.npz"))
    return {k: v[:5] for k, v in hits.items()}


def manifests(phase, inputs, write=True, on_date=None):
    """Step 'manifests' (L30): every arm whose conditions hold and whose manifest is not written: S0 and KF once
    MD-S0 is frozen; KD and KT once SIU's species-role table is decided and audited (not run, recorded, when SIU
    is rejected, role C or quarantined, or a K supply is under N_K); S1c, S1c-R and S1u once every phase-1 source
    with candidates is settled and every accepted one's MD-N frozen; S2 in phase 2. Only sources ready now enter
    (ready_pools); the K arms' added rows and every S arm's new rows pass the guard against DS1's own exams first
    (guard_new). Each written once (row bytes of every smaller manifest unchanged in the larger ones)."""
    if not frozen("md_s0"):
        return {"status": "MD-S0 is not frozen: no manifest is built before it (DS.5.4)"}
    md_rows = read_jsonl(Path(frozen("md_s0")["root"]).parent / "rows.jsonl")
    exam_rows = never_train_rows()
    built = {}
    if not arm_record_path("S0").is_file():
        s0, rec = b_prime_o(inputs, md_rows, exam_rows, write)
        if write:
            built["S0"] = write_arm("S0", s0, {"b_prime_o": rec})
    s0_rows, s0_rec = arm_rows("S0") if arm_record_path("S0").is_file() else ([], None)
    if not s0_rows:
        return {"status": "S0 not built"}
    s0_keys = {r["key"] for r in s0_rows}
    tc_rows = [r for r in s0_rows if r["key"] in inputs.tc_keys]
    c = len(tc_rows)
    n_s0 = len(s0_rows)
    # the K arms (base A plus N_K rows each; DS.5.3): KF from base B's thinned Field and Turf external rows (outside
    # MD-S0, test v1 and its companions), KD and KT from SIU's frames once its table is decided and audited; N_K =
    # min(15,000, each K pool's supply), the same in every K arm: KF records it, KD and KT take it or are not built
    pr = ready_pools(1, inputs)
    md_keys = {r["key"] for r in md_rows}
    companions = {str(r.get("key")) for r in B3.prior_test_lists()[0]} if inputs.production else set()
    kf_rows = []
    for k, v in inputs.base_b.items():
        if inputs.setting.get(k) in ("Field", "Turf") and v.get("kind") != B3.BASE_KIND and k not in md_keys \
                and k not in companions and k in inputs.base_b_man:
            kf_rows.append(dict(inputs.base_b_man[k], series=None, thin=True, dhash=v.get("dhash"),
                                variants=v.get("variants"), group=v.get("group"),
                                capture=next((c_[1] for c_ in v.get("capture_keys") or () if c_[0] == "capture"),
                                             None)))
    kf_rows = guard_new(_phase_rows({"base_b": kf_rows}, md_keys, exam_rows, set()), inputs)
    siu_dec = inputs.decisions.get(SIU) or {}
    siu_srec = screen_records(1).get(SIU)
    siu_ready = source_ready(SIU, inputs.decisions, siu_srec)[0]
    siu_out = siu_srec is not None and settled_out(SIU, inputs.decisions, siu_srec)
    crop = tuple(u[5:] for u, role in (siu_dec.get("roles") or {}).items() if u.startswith("eppo:") and role != "weed")
    siu = guard_new(_phase_rows({SIU: pr.get(SIU, [])}, md_keys, exam_rows, set()), inputs) if siu_ready else []
    kpools, t, n_k = k_pools(siu, kf_rows, crop)
    base_a = [dict(v, key=k) for k, v in inputs.base_a.items()]
    ahits = base_a_hits(inputs, base_a) if write and (not arm_record_path("KF").is_file() or siu_ready) else {}
    if write and not arm_record_path("KF").is_file() and kf_rows:
        nk_f = min(N_K_MAX, kpools["supplies"]["KF"]) if not siu_ready else n_k
        built["KF"] = write_arm("KF", base_a + [_man(r) for r in kf_draw(kpools, nk_f)],
                                {"n_k": nk_f, "pool": "base B Field and Turf, thinned", "supplies": kpools["supplies"],
                                 "base_a_kept_with_hits": ahits})
    k_nr = out_dir() / "arms" / "k_not_run.json"
    if write and siu_out and not k_nr.is_file():
        # SIU rejected, role C or quarantined whole: KD and KT have no supply; recorded not run, so verdict v1 is
        # never held by them (DS.9.3)
        why = "not run (the K source %s enters no DS1 manifest: %s)" % (SIU, source_ready(SIU, inputs.decisions,
                                                                                         siu_srec)[1])
        built["k_waits"] = why
        write_once(k_nr, {"format": FMT_ARM, "arms": ["KD", "KT"], "status": why, "written_utc": D._utc()})
    if write and siu_ready and arm_record_path("KF").is_file():
        nk = int((_read_json(arm_record_path("KF")) or {}).get("n_k") or 0)
        short = k_short(kpools["supplies"], nk)
        if short:
            why = "not run (supply under N_K %d: %s): N_K is the same in every K arm" % (nk, short)
            built["k_waits"] = why
            if not k_nr.is_file():
                write_once(k_nr, {"format": FMT_ARM, "arms": ["KD", "KT"], "status": why, "written_utc": D._utc()})
        else:
            tt = min(t, nk)
            if not arm_record_path("KD").is_file():
                built["KD"] = write_arm("KD", base_a + [_man(r) for r in kd_draw(kpools, tt, nk)],
                                        {"n_k": nk, "t": tt, "supplies": kpools["supplies"],
                                         "base_a_kept_with_hits": ahits})
            if not arm_record_path("KT").is_file():
                built["KT"] = write_arm("KT", base_a + [_man(r) for r in kt_draw(kpools, tt, nk)],
                                        {"n_k": nk, "t": tt, "supplies": kpools["supplies"],
                                         "base_a_kept_with_hits": ahits})
    # the S arms of phase 1: every screened source settled (role C, quarantined or rejected sources are settled out;
    # the others decided, audited where required), every accepted source's MD-N frozen
    srec1 = screen_records(1)
    pending = [s for s, v in srec1.items() if v.get("live") and not settled_out(s, inputs.decisions, v)
               and not source_ready(s, inputs.decisions, v)[0]]
    unfrozen = [s for s in pr if s != SIU and (inputs.decisions.get(s) or {}).get("decision") == "accept"
                and not frozen(mdn_exam(s))]
    # DS1 deviation 2: S1c, S1c-R and S1u wait for SIU's screen taken again and the superseded pools pooled again
    dev2 = {k: v for k, v in (("rescreen", rescreen_due(1)), ("repool", pools_superseded_due(1))) if v}
    if pending or unfrozen or dev2:
        built["phase1_waits"] = dict({"sheets": sorted(pending), "md_n": sorted(unfrozen)}, **dev2)
    elif write and not arm_record_path("S1U").is_file():
        new = guard_new(_phase_rows(pr, md_keys, exam_rows, s0_keys), inputs)
        siu_new = [r for r in new if r["source"] == SIU and r.get("thin")]
        other = [r for r in new if r["source"] != SIU and r.get("thin")]
        s1u = s0_rows + [_man(r) for r in siu_new + other]
        siu_base = sum(1 for r in s0_rows if r["source"] == SIU)
        kept, dropped, crec = family_cap(siu_new, siu_base, len(s0_rows) - siu_base + len(other), SIU_FAMILY)
        s1c = s0_rows + [_man(r) for r in kept + other]
        r_c, r_u = floor_repeats(len(s1c), n_s0, c), floor_repeats(len(s1u), n_s0, c)
        s1c_full = s1c + repeat_rows(tc_rows, r_c)
        s1u_full = s1u + repeat_rows(tc_rows, r_u)
        s1cr = s1cr_draw(s1c, n_s0, inputs.tc_keys)
        if not nested_ok({json.dumps(r, sort_keys=True) for r in s0_rows},
                         {json.dumps(_man(r), sort_keys=True) for r in s1c_full}):
            raise DataRefused("S1c does not hold every S0 row byte for byte")
        built["S1C"] = write_arm("S1C", s1c_full, {"repeats": r_c, "family_cap": crec, "dropped_videos": dropped,
                                                   "sources": sorted(pr)})
        built["S1U"] = write_arm("S1U", s1u_full, {"repeats": r_u, "sources": sorted(pr)})
        built["S1CR"] = write_arm("S1CR", s1cr, {"repeats": 1, "of": "S1C"})
    if int(phase) == 2 and write and not arm_record_path("S2").is_file() and arm_record_path("S1C").is_file():
        p2 = ready_pools(2, inputs)
        srec2 = screen_records(2)
        p2_pending = [s for s, v in srec2.items() if v.get("live") and not settled_out(s, inputs.decisions, v)
                      and not source_ready(s, inputs.decisions, v)[0]]
        p2_unfrozen = [s for s in p2 if (inputs.decisions.get(s) or {}).get("decision") == "accept"
                       and not frozen(mdn_exam(s))]
        if p2_pending or p2_unfrozen:
            built["phase2_waits"] = {"sheets": sorted(p2_pending), "md_n": sorted(p2_unfrozen)}
        else:
            s1c_rows, _r = arm_rows("S1C")
            s1c_base = [r for r in s1c_rows if not is_repeat(r["key"])]
            new = guard_new(_phase_rows(p2, md_keys, exam_rows, {r["key"] for r in s1c_base}), inputs)
            fams = collections.defaultdict(list)
            for r in new:
                if r.get("thin"):
                    fams[inputs.families.get(r["source"]) or r["source"]].append(r)
            rows2 = list(s1c_base)
            caps = {}
            for fam in sorted(fams):
                members = fams[fam]
                base = sum(1 for r in rows2 if (inputs.families.get(r["source"]) or r["source"]) == fam)
                if fam in (SIU_FAMILY, "mfwd", "dock"):
                    kept, dropped, rec = family_cap(members, base, len(rows2) - base + sum(
                        len(v) for f, v in fams.items() if f != fam), fam)
                    caps[fam] = rec
                else:
                    kept = members
                rows2 += [_man(r) for r in kept]
            r2 = floor_repeats(len(rows2), n_s0, c)
            built["S2"] = write_arm("S2", rows2 + repeat_rows(tc_rows, r2), {"repeats": r2, "family_caps": caps,
                                                                            "sources": sorted(p2)})
    return built


def _phase_rows(pr, md_keys, exam_rows, already):
    """A phase's admitted pool rows minus every MD and MT-W row, every row sharing a capture relation with one,
    and rows a smaller manifest holds already."""
    flat = [dict(r) for s, rs in sorted(pr.items()) for r in rs]
    gh = group_hits(flat, exam_rows)
    ex_keys = {r["key"] for r in exam_rows}
    return [r for r in flat if r["key"] not in ex_keys and r["key"] not in md_keys and not gh.get(r["key"])
            and r["key"] not in already]


def guard_new(rows, inputs):
    """The pools' rows against MD v1 and MT-W v2 (drawn after the pools' guard ran), both directions: the 8-variant
    6-bit dHash and the calibrated DINOv2 detector (DS1's own index); a hit is dropped (DS.5.1 item 7)."""
    ex = {k: v for k, v in inputs.exams.items() if k == "ds1_exams"}
    hits = dhash_hits(rows, ex) if ex else {}
    for name in ("own_exams", "own"):
        if inputs.scanners.get(name) and rows:
            hits = merge_hits(hits, embed_hits(rows, inputs.scanners[name]))
    return [r for r in rows if not hits.get(r["key"])]


def _man(r):
    return {k: r[k] for k in C.MANIFEST_KEYS} if all(k in r for k in C.MANIFEST_KEYS) else \
        {"image": r.get("train_image") or r["image"], "label": r["label"], "sha256": r.get("train_sha256") or
         r["sha256"], "label_sha256": r["label_sha256"], "source": r["source"], "session": r.get("session") or "",
         "key": r["key"]}


# ------------------------------------------------------------------ what the platform reads (no image opened)
def plan_state(inc_dir=None):
    """DS1's data state for the platform's gate and plan (stream_remote.ds1_status; read-only, cheap): per phase the
    screens and whether a Step 1 batch came since the last one, each screened source's sheet, audit, pool, MD-N
    and MT-W v2 states, MD-S0's drafts, power and freeze, the arms built (their images and loader projections),
    the sheets' bundles and renders, the MT-W checks and the LN record."""
    from . import ds1_sheets as SH
    inc = Path(inc_dir) if inc_dir else C.INC_DIR
    decs = SH.fold(inc)
    checks = SH.mtw_checks(inc)
    batches = set(step1_batches(inc))
    out = {"phases": {}, "md_s0": {}, "arms": {}, "sheets": {}, "checks": {s: v.get("verdict") for s, v in
                                                                            checks.items()}}
    for ph in PHASES:
        revs = screen_revisions(ph)
        seen = set(((_read_json(revs[-1]) or {}).get("inputs") or {}).get("step1_batches") or []) if revs else set()
        repool = set(pools_superseded_due(ph))         # DS1 deviation 2, item 1: pooled again, so not pooled now
        srcs = {}
        for s, v in screen_records(ph).items():
            if not v.get("live"):
                continue
            dec = decs.get(s) or {}
            ok, why = source_ready(s, decs, v)
            srcs[s] = {"decided": bool(dec.get("decision")), "decision": dec.get("decision"),
                       "audit_required": bool(dec.get("audit_required")), "audited": bool(dec.get("audited")),
                       "ready": ok, "why": why, "pooled": (out_dir() / "pools" / ("p%d" % ph) / ("%s.jsonl" % slug(s)))
                       .is_file() and s not in repool, "md_n": bool(frozen(mdn_exam(s))),
                       "mtw_draft": (exam_dir(mtw_exam(s)) / "draft")
                       .is_dir(), "mtw": bool(frozen(mtw_exam(s))) or (exam_dir(mtw_exam(s)) / "left.json").is_file(),
                       "checked": s in checks,
                       "quarantined_whole": bool(v.get("quarantine_whole")),
                       # role C by its capture names (DS.5.1 item 1(b)); settled out (role C, quarantined or
                       # rejected: in no manifest, no MD-N, no MT-W v2); SIU is never drawn into an exam
                       "role_c": bool(v.get("role_c") or int((v.get("capture_names") or {}).get("hits") or 0)),
                       "settled_out": settled_out(s, decs, v), "exam_draw": s != SIU}
        out["phases"][str(ph)] = {"revisions": len(revs), "new_batches": len(batches - seen) if revs else None,
                                  "sources": srcs, "audit_due": [p.name for p in audit_due(ph)],
                                  # DS1 deviation 2: the sources the next screen takes again, the pools moved aside
                                  "rescreen": rescreen_due(ph), "repool": sorted(repool)}
    drafts = sorted(exam_dir("md_s0").glob("draft_r[0-9][0-9][0-9]")) if exam_dir("md_s0").is_dir() else []
    last = drafts[-1].name if drafts else None
    pw = _read_json(D.root() / "power" / ("%s.json" % last)) if last else None
    out["md_s0"] = {"drafts": len(drafts), "draft": last, "power": isinstance(pw, dict),
                    "enlarge": bool(((pw or {}).get("md_s0") or {}).get("enlarge")) if isinstance(pw, dict) else None,
                    "frozen": bool(frozen("md_s0"))}
    for a in D.BUILD_ARMS:
        r = _read_json(arm_record_path(a))
        if isinstance(r, dict):
            out["arms"][a] = {"images": r.get("images"), "sha256": r.get("sha256"),
                              "over_line": bool((r.get("loader_projection") or {}).get("over_line"))}
    bdir = SH.sheets_dir(inc) / "bundles"
    names = sorted(p.name for p in bdir.iterdir() if p.is_dir() and (p / "bundle.json").is_file()) \
        if bdir.is_dir() else []
    rendered = sorted(p.stem for p in (SH.sheets_dir(inc) / "rendered").glob("*.json")) \
        if (SH.sheets_dir(inc) / "rendered").is_dir() else []
    src_b = [n for n in names if n.startswith("source_")]
    out["sheets"] = {"source_batches": len(SH.source_chunks(src_b)),
                     "ln_batches": len([n for n in names if n.startswith("ln_")]),
                     "mtw_batches": len([n for n in names if n.startswith("mtw_")]),
                     "rendered": rendered, "audit_pending": SH.audit_sources(decs),
                     "ln_recorded": bool(SH.ln_marks(inc)),
                     # the owner's audit batches rendered and not superseded, by their sources: a set rendered is
                     # never proposed again (Revision 2, item 2)
                     "audit_sets": SH.audit_sets(inc), "superseded": sorted(SH.superseded(inc))}
    out["mtw_v1_1"] = (_read_json(out_dir() / "mtw_v1_1" / "status.json") or {}).get("format") is not None
    out["k_not_run"] = (out_dir() / "arms" / "k_not_run.json").is_file()
    srec1 = screen_records(1)
    out["k_source_out"] = SIU in srec1 and settled_out(SIU, decs, srec1.get(SIU))
    out["ra"] = isinstance(_read_json(D.capacity("ds1_ra_gate.json")), dict)
    out["power_final"] = isinstance(_read_json(D.capacity("ds1_power.json")), dict)
    return out


# ------------------------------------------------------------------ the summary (the index the snapshot ships)
def write_summary(extra=None):
    """splits/ds1/summary.json (rewritten atomically by every step): the arms built (manifest sha256, images, the
    three counts, the recipe), the exams frozen, each phase's screens and pools, the steps' states. The platform's
    evidence reads it (DS.9.4)."""
    arms = {}
    for a in D.BUILD_ARMS:
        r = _read_json(arm_record_path(a))
        if isinstance(r, dict):
            arms[a] = {k: r.get(k) for k in ("manifest", "sha256", "images", "rows_before_repeats", "counts",
                                             "permissive_only_rows", "research_only")}
            arms[a]["loader_projection_h"] = r.get("loader_projection_h")
    exams = {}
    for p in sorted(out_dir().glob("*/*/frozen.json")) if out_dir().is_dir() else []:
        d = _read_json(p) or {}
        exams[d.get("exam")] = {k: d.get(k) for k in ("lock_sha256", "images", "groups", "frozen_utc")}
    drafts = {}
    for p in sorted(out_dir().glob("mtw_v2/*/draft/rows.jsonl")) if out_dir().is_dir() else []:
        drafts[p.parent.parent.name] = len(read_jsonl(p))
    doc = {"format": FORMAT, "status": "index", "arms": arms, "exams": exams, "mtw_v2_drafts": drafts,
           "screens": {"p%d" % ph: len(screen_revisions(ph)) for ph in PHASES},
           "pools": {"p%d" % ph: len(pool_rows(ph)) for ph in PHASES}, "md_s0_frozen": bool(frozen("md_s0")),
           "never_train_rows": len(never_train_rows()), "decided_by": D.DECIDED_BY, "written_utc": D._utc()}
    doc.update(extra or {})
    C2.write_json_atomic(summary_path(), json.loads(json.dumps(doc, default=str)))
    return doc


def read_summary():
    d = _read_json(summary_path())
    return d if isinstance(d, dict) and d.get("format") == FORMAT else None


# ------------------------------------------------------------------ the build verb and the rehearsal
def build(phase, step, procs=5, embedder=None, inputs=None):
    """L30: one step of one phase, the E-series and V3F trees unchanged around it."""
    if step not in STEPS or int(phase) not in PHASES:
        raise DataRefused("build --phase 1|2 --step %s, not %s %s" % ("|".join(STEPS), phase, step))
    D.check_frozen()
    D.require_timestamp()
    with D.trees_unchanged():
        inputs = inputs or load_inputs(phase, procs=procs, embedder=embedder)
        if step == "screen":
            got = screen(phase, inputs)
        elif step == "md_s0":
            got = md_s0(inputs)
        elif step == "pools":
            got = pools(phase, inputs)[0]
        elif step == "draw":
            from . import ds1_sheets as SH
            got = draw(phase, inputs, checks=SH.mtw_checks())
        else:
            got = manifests(phase, inputs)
        write_summary({"last_step": {"phase": int(phase), "step": step, "utc": D._utc()}})
    log("phase %s step %s: %s" % (phase, step, json.dumps(got, default=str)[:600]))
    return got


def rehearse(phase, out=None):
    """The read-only rehearsal (DS.9.5) on the login node: every pool's counts (images, distinct photos, capture
    groups) and the role C screen's hits per source, from the files as they are; writes nothing under INC_DIR."""
    t0 = time.time()
    summ = B3.read_summary()
    pf = summ["files"]["provenance"]
    base_b = {r["key"]: r for r in read_jsonl(B3.out_dir() / pf["file"]) if "B" in (r.get("arms") or [])}
    cands = load_candidates(phase, base_b)
    ctx = {"base_keys": set(base_b), "base_shas": {r.get("original_sha256") for r in base_b.values()},
           "eval_sources": {s for g in (load_rules()[1].get("evaluation_groups") or {}).get("groups", {}).values()
                            for s in g}, "quarantined": stream_quarantined(), "never": domain_never(),
           "quarantine_cites": stream_quarantine_cites()}
    per = collections.defaultdict(lambda: collections.Counter())
    live = collections.defaultdict(list)
    for r in cands:
        why = eligibility(r, ctx)
        per[r["source"]][why or "candidate"] += 1
        if not why:
            live[r["source"]].append(r)
    rec = {}
    for s, c in sorted(per.items()):
        rs = live.get(s, [])
        units = collections.Counter(unit_of(r, b[0]) for r in rs for b in r["boxes"])
        rec[s] = {"counts": dict(c), "live": three_counts(rs), "units": dict(units),
                  "capture_names": len(capture_name_hits([r.get("rel") or r["key"] for r in rs])),
                  "series": len({r.get("series") for r in rs if r.get("series")})}
    doc = {"format": "inc2-ds1-rehearsal/1", "phase": int(phase), "candidates": len(cands),
           "live": sum(len(v) for v in live.values()), "sources": rec, "seconds": round(time.time() - t0, 1),
           "base_b_rows": len(base_b), "written_utc": D._utc(), "writes_nothing_under_inc_dir": True}
    if out:
        Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="DS1's data builds (L30).")
    ap.add_argument("command", choices=("build", "rehearse", "status"))
    ap.add_argument("--phase", type=int, choices=PHASES, default=1)
    ap.add_argument("--step", choices=STEPS, default=None)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    try:
        if a.command == "build":
            if not a.step:
                raise DataRefused("build needs --step %s" % "|".join(STEPS))
            build(a.phase, a.step, procs=a.procs)
        elif a.command == "rehearse":
            doc = rehearse(a.phase, a.out)
            print(json.dumps({k: doc[k] for k in ("phase", "candidates", "live", "seconds")}))
        else:
            print(json.dumps(read_summary(), indent=1, sort_keys=True))
    except (D.Ds1Refused, C2.Inc2Error, B3.Base3Error, ValueError) as e:
        print("[inc2.ds1_data] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
