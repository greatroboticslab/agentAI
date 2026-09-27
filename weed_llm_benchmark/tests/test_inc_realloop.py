#!/usr/bin/env python3
"""INC builders for Step 1.6 and Steps 2-3: `pilot build-baseline` (a baseline
experiment on any training manifest, base B) and `realloop build` (the chain
experiment on real data).

The world is synthetic, written in inc/verify.py's own formats (pool.jsonl,
pool_meta.jsonl, verified.jsonl, conflicts.csv, crops.csv, emb_sNNN_of_NNN.npz,
pool_verdicts.npz, admit_summary.json), plus the locked train_core manifest,
the never-train index and LOCK.json. inc/select.py's real build then makes base
B and the increment pool from it. Every image is a real small PNG, because
both builders check manifests the way inc/train.py does (image bytes, dHash,
never-train guard).
  * train_core: 40 images in 8 sessions. The verified pool holds 360 images of
    harvA, harvB and harvU1, a fifth of them OtherPlant-only.
  * Images verify did not admit:
      - harvU1: 24 conflict and 16 unknown images, plus 7 unknown images whose
        pool_meta dHash is planted after select's build: 1-3 bits from an
        image of real_t's V1, V4, OTHER_HEAVY or base B's harvested part (6,
        excluded), and one 4 bits from a V1 image (eligible): 47, of which 41
        are eligible for real_t;
      - harvU2: 60 unknown images, 30 of them planted 1-2 dHash bits from a
        train_core photograph: the most non-admitted images, but only 30
        eligible;
      - harvU3: 8 conflict images, fewer than M.
    So the UNVERIFIED source must be harvU1, chosen on eligible images and not
    on raw counts. select's build reads only the verified images' dHashes, so
    rebuilding it after the planting gives the same base and pool.

Pinned:
  * build-baseline: the manifest is copied byte for byte, run cold with the
    protocol's recipe, one base run per seed, then final runs on every exam,
    test included; build_summary.json records the manifest's sha256, boxes per
    class, images and boxes per source, and for base B the select build's
    train_core and never-train index against LOCK.json (none for a manifest no
    select build names). It refuses before anything is written: a label class
    outside 0..12, a row without a manifest key, duplicate keys, an image
    within 6 bits of an evaluation image, an evaluation split's manifest, bad
    seeds, a production build without LOCK.json, with a never-train index
    LOCK.json did not record, or on a base_B selected under another train_core
    lock. The CLI builds and refuses the same way;
  * realloop build:
      - every increment holds exactly M images, M = 10 % of base B by default;
      - base, verified increments and UNVERIFIED are pairwise disjoint (key,
        path, sha256). No UNVERIFIED image is within NEAR_DUP_BITS of an image
        of the base or of a verified increment; each planted near-copy is
        excluded and listed with the image it is near, the 4-bit one is not.
        Every excluded image really is such a copy;
      - V1..VN and OTHER_HEAVY are select.increments' own draw (the same bytes
        as calling it directly);
      - UNVERIFIED is harvU1's, holds only non-admitted images with pool.jsonl's
        own rows (labels as joined), its keys are the rule's draw recomputed
        here (seed text exp + '/unverified'), and it records its verdict mix
        and conflict boxes, which the test recounts;
      - sequence V1, V2, UNVERIFIED, V3, OTHER_HEAVY, V4, V5, V6 (and N = 1..3);
        clean = every verified increment;
      - the recipes are the pilot's (subset), base and truth recipes the cold
        one; replay mode, the gate block (--gate-flips-mode: negative by
        default, net on request, in exp.json and build_summary.json) and
        --no-truth reach exp.json; the Step 1 provenance
        and the baselines already built on the same base are recorded; exp.json
        holds the definition, build_summary.json the counts;
      - driver init runs on FakeBackend: one array of base and truth runs. The
        truth arm's UNVERIFIED step trains T + UNVERIFIED, and the next step's T
        does not hold it. Driven to done with a synthetic executor, every chain
        accepts the verified increments and rejects UNVERIFIED, the truth arm
        agrees, T_final is base + every verified increment, and the report has
        UNVERIFIED's attribution per chain and no Bswap section;
      - the build is deterministic;
      - the default build (no --increment-sources) is the one built before
        that option existed: real_t's digest (build_digest: exp.json,
        build_summary.json and every manifest, with paths, sha256s and
        timestamps left out) equals the one recorded with realloop.py and
        select.py as they were then (DEFAULT_BUILD_DIGEST);
      - it refuses: a base that is not select's base_B, an admit summary that
        disagrees with conflicts.csv, a never-train hit among non-admitted
        images, no source with M eligible images, unknown recipes, replay
        modes or gate flips modes, a second build, and a production build
        without LOCK.json;
      - it refuses, with Step 1 edited so that only one check can object (the
        other summaries re-stamped): verified.jsonl or pool_meta.jsonl changed
        after select's build; a conflicts.csv row for an admitted image; a
        verified row whose label is not its pool.jsonl row's; a base_B with a
        class-13 box or an image near an evaluation image; increment label
        bytes that differ from the manifest (V1, and UNVERIFIED); a
        non-admitted image byte-identical to a train_core photograph or to a
        harvested base image; in production, a base selected under another
        train_core lock or never-train index (a testing build records it);
      - the relevance filter (inc/relevance.py, injected fake text tower at
        logit scale 100, calibration check holding): a production build
        refuses without relevance.json, and finds the one next to the base;
        with it, V1..V6 and OTHER_HEAVY hold no image of a source that does
        not pass (harvU1, too few crops left in the increment pool, is
        'insufficient'; the unfiltered real_t draw holds some) and are
        select's filtered draw byte for byte; the file's sha256 is in
        exp.json's step1 files and the exclusions in the summary; UNVERIFIED
        is not filtered (still harvU1, with its source's relevance status
        recorded), also when harvU1 FAILS relevance (its entry re-stamped
        consistently: no source of this world looks non-plant); a relevance
        file with a degenerate calibration (a weaker fake whose tau falls
        below 0.5) is refused before anything is written; a missing
        --relevance file refuses;
      - --increment-sources evidence (admit_summary.json per_slug box
        verdicts counted as verify admit counts them; --min-evidence one above
        harvU1's verified boxes): a production build builds without
        relevance.json; V1..V6 and OTHER_HEAVY hold only evidenced sources'
        images (real_t's default draw holds harvU1's) and are select's
        evidence draw byte for byte; the mode is in exp.json's and
        build_summary.json's step1 block (a default build has no such key);
        build_summary.json records the evidenced and excluded sources, the
        admit summary's sha256 and the capacity; UNVERIFIED is the rule's
        draw recomputed here, from harvU1 though it is not evidenced, with its
        verified boxes recorded; an explicit --increment-sources relevance
        gives the default build's manifests byte for byte, and the same
        increments summary, build summary and definition apart from
        timestamps; refusals before anything is written: --relevance with evidence, --min-evidence without it or 0,
        an unknown mode, the two files disagreeing (either edited), too few
        evidenced images for 6 + 1 increments of M (with the numbers), too
        few OtherPlant-heavy images (pool_capacity stubbed); the CLI.

No open_clip, no network, no GPU.

Run:  python3 tests/test_inc_realloop.py
"""
import collections
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import statistics
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_realloop_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"            # tests only: the experiments here are testing ones
os.environ.pop("INC_JOB_SCRIPT", None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc import realloop as RL  # noqa: E402
from weed_optimizer_framework.tools.inc import report as R  # noqa: E402
from weed_optimizer_framework.tools.inc import select as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.near_dup import HOLDOUT_NEAR_DUP_BITS, NEAR_DUP_BITS  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc=Exception, contains=None):
    try:
        fn()
    except exc as e:
        if contains and contains not in str(e):
            print("       raised %s without %r: %s" % (type(e).__name__, contains, e))
            return False
        return True
    return False


def keys_of(path):
    return [r["key"] for r in C.read_manifest(path)]


def bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


# ------------------------------------------------------------------ the world
EMB_D = 32
N_SCENES = 6
VERIFIED_SOURCES = (("harvA", 220), ("harvB", 130), ("harvU1", 10))
U1_CONFLICT, U1_UNKNOWN = 24, 16
U2_N, U2_NEAR = 60, 30
U3_N = 8
N_NEVER = 40
NEVER_MIN_BITS = 12          # evaluation dHashes this far from every pool hash (a planted one moves <= 4)
PLAN_EXP = "real_t"          # the experiment whose increments the planted near-copies sit next to
# harvU1 images verify did not admit, planted (pool_meta dHash) this many bits
# from an image of real_t's base (its harvested part) or of a verified
# increment: within NEAR_DUP_BITS they are excluded; the 4-bit one stays eligible
U1_PLANT = (("V1", 0, 1), ("V1", 1, 2), ("V4", 0, 2), ("OTHER_HEAVY", 0, 1), ("base_B", 0, 1),
            ("base_B", 1, 3), ("V1", 2, 4))
PLAN_FILES = {"V1": S.INC_NAME % 1, "V4": S.INC_NAME % 4, "OTHER_HEAVY": S.INC_OTHER}
U1_ELIGIBLE = U1_CONFLICT + U1_UNKNOWN + sum(1 for _o, _i, b in U1_PLANT if b > NEAR_DUP_BITS)
REPO = TMP / "repo"


def _png(path, rng, size=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(path)


def _box(rng, c):
    return (int(c), float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7)),
            float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.05, 0.3)))


def _row(key, image, label, source, session=""):
    return {"image": str(image), "label": str(label), "sha256": C.sha256_file(image),
            "label_sha256": C.sha256_file(label), "source": source, "session": session, "key": key}


def make_world():
    """Step 1 in verify's formats, select's build on it, the never-train index
    and LOCK.json. Returns ground truth for the checks."""
    rng = np.random.default_rng(11)
    basis = np.linalg.qr(rng.normal(size=(EMB_D, EMB_D)))[0]
    scene_dir, class_dir = basis[:N_SCENES], basis[N_SCENES:N_SCENES + C.NC]
    step1 = pathlib.Path(V.STEP1)
    assert str(step1).startswith(str(TMP)), step1           # never the machine's real step1
    step1.mkdir(parents=True)
    crops, emb, verdicts, meta, pool, verified, conflicts = [], [], [], [], [], [], []

    core, core_hash = [], []
    for s in range(8):
        sess = "2021%02d01_cam%d" % (s + 3, s % 2)
        for j in range(5):
            i = 5 * s + j
            stem = "%s_%d" % (sess, j + 1)
            img = REPO / "downloads" / "cwd12" / "images" / (stem + ".png")
            _png(img, rng)
            boxes = [_box(rng, (i + 4 * b) % 12) for b in range(2 + i % 2)]
            lab = REPO / "downloads" / "cwd12" / "labels" / (stem + ".txt")
            C.write_yolo(lab, boxes)
            key = "train_core__" + stem
            core.append(_row(key, img, lab, "cottonweeddet12/train", sess))
            core_hash.append(C.dhash(str(img)))
            for b, box in enumerate(boxes):
                z = rng.normal(size=EMB_D)
                z -= scene_dir.T @ (scene_dir @ z)          # noise off the scene directions
                crops.append(("core", key, str(img), "train_core", sess, b, box[0]))
                emb.append(class_dir[box[0]] + 0.75 * z)
    C.write_manifest(C.manifest_path("train_core"), core)

    def pool_image(key, source, classes, box_verdicts, dhash=None):
        img = REPO / "datasets" / source / "images" / (key + ".png")
        _png(img, rng)
        boxes = [_box(rng, c) for c in classes]
        lab = step1 / "labels" / (key + ".txt")
        C.write_yolo(lab, boxes)
        r = _row(key, img, lab, source)
        meta.append({"key": key, "source": source, "W": 16, "H": 16,
                     "dhash": C.dhash(str(img)) if dhash is None else dhash,
                     "boxes": [list(b) for b in boxes], "src": []})
        first = len(crops)
        scene = int(rng.integers(0, N_SCENES))
        for b, (c, v) in enumerate(zip(classes, box_verdicts)):
            crops.append(("pool", key, str(img), source, source, b, c))
            emb.append(4.0 * scene_dir[scene] + class_dir[c] + 0.25 * rng.normal(size=EMB_D))
            verdicts.append(v)
        pool.append(r)
        return r, first

    t = 0
    for source, n in VERIFIED_SOURCES:
        for j in range(n):
            nb = 1 + int(rng.integers(0, 3))
            cls = ([C.OTHER_PLANT] * nb if t % 5 == 0 else
                   [C.OTHER_PLANT if rng.random() < 0.25 else int(rng.integers(0, 12)) for _ in range(nb)])
            r, _ = pool_image("pool_%s_%03d" % (source, j), source, cls,
                              ["other_ok" if c == C.OTHER_PLANT else "verified" for c in cls])
            verified.append(r)
            t += 1

    truth = {"verdict": {}, "near_core": set()}

    def not_admitted(key, source, kind, dhash=None):
        cls = [int(rng.integers(0, 12))] + [int(rng.integers(0, 13)) for _ in range(int(rng.integers(0, 2)))]
        v = ["conflict" if kind == "conflict" else "unknown"] + [
            "other_ok" if c == C.OTHER_PLANT else "verified" for c in cls[1:]]
        r, first = pool_image(key, source, cls, v, dhash=dhash)
        truth["verdict"][key] = kind
        if kind == "conflict":
            pred = (cls[0] + 1) % 12
            conflicts.append({"source": source, "image": r["image"], "key": key, "box": 0, "label": cls[0],
                              "label_name": C.CLASS_NAMES[cls[0]], "pred": pred,
                              "pred_name": C.CLASS_NAMES[pred], "p": 0.9, "cosine": 0.5,
                              "crop_id": first, "sheet_index": ""})

    for j in range(U1_CONFLICT + U1_UNKNOWN):
        not_admitted("u1_%03d" % j, "harvU1", "conflict" if j < U1_CONFLICT else "unknown")
    for j in range(len(U1_PLANT)):
        not_admitted("u1_near_%d" % j, "harvU1", "unknown")     # dHash planted after select's build
    for j in range(U2_N):
        h = None
        if j < U2_NEAR:
            h = core_hash[j] ^ (1 << (j % 64))
            if j >= U2_NEAR // 2:
                h ^= 1 << ((j + 17) % 64)                    # 2 bits away
            truth["near_core"].add("u2_%03d" % j)
        not_admitted("u2_%03d" % j, "harvU2", "unknown", dhash=h)
    for j in range(U3_N):
        not_admitted("u3_%03d" % j, "harvU3", "conflict")

    hashes = [m["dhash"] for m in meta]
    never = []
    while len(never) < N_NEVER:
        h = int.from_bytes(rng.bytes(8), "big")
        if all(bits(h, x) > NEVER_MIN_BITS for x in hashes + core_hash):
            never.append(h)
    C.NEVER_TRAIN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(C.NEVER_TRAIN_INDEX, "w") as fh:
        json.dump({"entries": [[h, "dev" if i % 2 else "test", "/x/eval_%02d.jpg" % i]
                               for i, h in enumerate(never)], "min_expected": N_NEVER, "complete": True}, fh)
    C.write_manifest(V.POOL, pool)
    C.write_manifest(V.VERIFIED_MANIFEST, verified)

    def write_pool_meta():
        with open(V.POOL_META, "w") as fh:
            for m in sorted(meta, key=lambda m: m["key"]):
                fh.write(json.dumps(m, sort_keys=True) + "\n")

    write_pool_meta()
    with open(V.CONFLICTS, "w", newline="") as fh:
        fields = ["source", "image", "key", "box", "label", "label_name", "pred", "pred_name", "p",
                  "cosine", "crop_id", "sheet_index"]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for c in conflicts:
            wr.writerow(c)
    with open(V.CROPS, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        for cid, (st, key, img, src, group, b, c) in enumerate(crops):
            wr.writerow([cid, st, key, img, src, group, b, "0.500000", "0.500000", "0.100000", "0.100000",
                         16, 16, c, C.CLASS_NAMES[c]])
    crops_sha = C.sha256_file(V.CROPS)
    X = np.asarray(emb, dtype=np.float16)
    ids = np.arange(len(crops))
    pathlib.Path(V.EMB_DIR).mkdir(parents=True)
    for s, part in enumerate(np.array_split(rng.permutation(len(ids)), 2)):
        np.savez(pathlib.Path(V.EMB_DIR) / ("emb_s%03d_of_002.npz" % s),
                 meta=np.array(json.dumps({"crops_sha256": crops_sha, "embedder": "fake", "dim": EMB_D,
                                           "crops": int(len(part)), "stats": {}})),
                 crop_ids=ids[part], X=X[part])
    pool_ids = np.array([i for i, cr in enumerate(crops) if cr[0] == "pool"])
    np.savez(V.POOL_VERDICTS, meta=np.array(json.dumps({"crops_sha256": crops_sha,
                                                        "verdict_codes": list(V.VERDICT_CODES)})),
             crop_id=pool_ids, verdict=np.array([V.VERDICT_CODES.index(v) for v in verdicts], dtype=np.int8))
    vkeys = {r["key"] for r in verified}
    per = collections.defaultdict(collections.Counter)
    for r in pool:
        per[r["source"]][V.ADMITTED if r["key"] in vkeys else truth["verdict"][r["key"]]] += 1
    per_boxes = collections.defaultdict(collections.Counter)          # box verdicts per source, as admit counts them
    pool_crops = [cr for cr in crops if cr[0] == "pool"]
    assert len(pool_crops) == len(verdicts)
    for cr, v in zip(pool_crops, verdicts):
        per_boxes[cr[3]][v] += 1

    def write_admit():
        with open(V.ADMIT_SUMMARY, "w") as fh:
            json.dump({"verified_sha256": C.sha256_file(V.VERIFIED_MANIFEST), "crops_sha256": crops_sha,
                       "embeddings": {"nshards": 2, "embedder": "fake", "dim": EMB_D, "failed_crops": 0,
                                      "nan_rows": 0},
                       "inputs": {"pool_sha256": C.sha256_file(V.POOL),
                                  "pool_meta_sha256": C.sha256_file(V.POOL_META),
                                  "copies_sha256": "0" * 64,
                                  "train_core_sha256": C.sha256_file(C.manifest_path("train_core"))},
                       "images": V._counter(sum(per.values(), collections.Counter())),
                       "per_slug": {s: {"images": V._counter(c), "boxes": V._counter(per_boxes[s])}
                                    for s, c in sorted(per.items())}},
                      fh, indent=1, sort_keys=True)

    write_admit()
    write_lock()
    S.build(0.5, 0, 100.0, gate=0.0)

    # the planted near-copies: next to real_t's increments (select's draw for
    # that name) and base_B's harvested part. select's build reads only the
    # verified images' dHashes, so rebuilding it on the new pool_meta.jsonl
    # gives the same base and pool, and so the same increments.
    step1_out = {n: (step1 / n).read_bytes() for n in (S.BASE_B, S.BASE_SELECTED, S.POOL, S.CLUSTERS)}
    m = RL.default_size(len(C.read_manifest(step1 / S.BASE_B)))
    plan = TMP / "plan" / PLAN_EXP
    S.increments(PLAN_EXP, 6, m, other_heavy=True, base_dir=step1, exp_dir=plan)
    targets = {n: sorted(C.read_manifest(plan / "manifests" / f), key=lambda r: r["key"])
               for n, f in PLAN_FILES.items()}
    targets["base_B"] = sorted(C.read_manifest(step1 / S.BASE_SELECTED), key=lambda r: r["key"])
    by_key = {mm["key"]: mm for mm in meta}
    planted = []
    for j, (owner, i, nb) in enumerate(U1_PLANT):
        target = targets[owner][i]
        h = by_key[target["key"]]["dhash"]
        for pos in (5, 23, 41, 59)[:nb]:
            h ^= 1 << pos
        by_key["u1_near_%d" % j]["dhash"] = h
        planted.append({"key": "u1_near_%d" % j, "owner": owner, "target": target["key"], "bits": nb})
    write_pool_meta()
    write_admit()
    S.build(0.5, 0, 100.0, gate=0.0, force=True)
    assert all((step1 / n).read_bytes() == b for n, b in step1_out.items()), "select's rebuild changed"

    hashes = [mm["dhash"] for mm in meta]
    assert len(set(hashes)) == len(hashes), "pool dHashes are unique (verify drops exact repeats)"
    assert not set(hashes) & set(core_hash), "no pool image repeats a train_core dHash"
    assert all(bits(h, x) > HOLDOUT_NEAR_DUP_BITS for h in never for x in hashes), \
        "the evaluation dHashes stay clear of the pool"
    truth.update(core=core, core_hash=core_hash, pool=pool, verified=verified, conflicts=conflicts,
                 meta={mm["key"]: mm for mm in meta}, never=never, planted=planted, m=m)
    return truth


def write_lock(nevertrain=None):
    C.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.LOCK_PATH, "w") as fh:
        json.dump({"manifests": {"train_core": C.sha256_file(C.manifest_path("train_core"))},
                   "nevertrain_sha256": nevertrain or C.sha256_file(C.NEVER_TRAIN_INDEX)}, fh)


def relock(train_core=None):
    """LOCK.json re-stamped as a later lock would be: train_core as given (else
    the manifest's), the never-train index as it is now."""
    lock = json.loads(C.LOCK_PATH.read_text())
    lock["manifests"]["train_core"] = train_core or C.sha256_file(C.manifest_path("train_core"))
    lock["nevertrain_sha256"] = C.sha256_file(C.NEVER_TRAIN_INDEX)
    C.LOCK_PATH.write_text(json.dumps(lock))


def base_b():
    return pathlib.Path(V.STEP1) / S.BASE_B


def select_summary_path():
    return pathlib.Path(V.STEP1) / S.SUMMARY


class Edited:
    """Step 1's files (and any extra paths) as they are, restored on exit:
    one refusal case edits them in between."""

    def __init__(self, *extra):
        self.paths = [pathlib.Path(p) for p in (V.POOL, V.POOL_META, V.VERIFIED_MANIFEST, V.CONFLICTS,
                                                V.ADMIT_SUMMARY, select_summary_path(), base_b(),
                                                C.NEVER_TRAIN_INDEX, C.LOCK_PATH) + extra]

    def __enter__(self):
        self.saved = {p: p.read_bytes() for p in self.paths if p.exists()}
        return self

    def __exit__(self, *exc):
        for p in self.paths:
            if p in self.saved:
                p.write_bytes(self.saved[p])
            elif p.exists():
                p.unlink()
        return False


def restamp_admit():
    """admit_summary.json names the current verified.jsonl, pool.jsonl and pool_meta.jsonl."""
    a = json.loads(pathlib.Path(V.ADMIT_SUMMARY).read_text())
    a["verified_sha256"] = C.sha256_file(V.VERIFIED_MANIFEST)
    a["inputs"]["pool_sha256"] = C.sha256_file(V.POOL)
    a["inputs"]["pool_meta_sha256"] = C.sha256_file(V.POOL_META)
    pathlib.Path(V.ADMIT_SUMMARY).write_text(json.dumps(a, indent=1, sort_keys=True))


def restamp_select(*names):
    """select_summary.json names the current file for each of names: 'verified' and
    'pool_meta' (inputs), 'base_B' (outputs)."""
    p = select_summary_path()
    sel = json.loads(p.read_text())
    for n in names:
        if n == "base_B":
            sel["outputs"][S.BASE_B]["sha256"] = C.sha256_file(base_b())
        else:
            sel["inputs"][n]["sha256"] = C.sha256_file({"verified": V.VERIFIED_MANIFEST,
                                                        "pool_meta": V.POOL_META}[n])
    p.write_text(json.dumps(sel, indent=1, sort_keys=True))


def write_rows(path, rows):
    """Rows as JSON lines in the given order (not C.write_manifest's key order)."""
    pathlib.Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def set_pool_image(key, image, dhash):
    """Pool image `key` becomes `image` (pool.jsonl row: path and sha256;
    pool_meta.jsonl: dHash); admit and select summaries re-stamped, so only
    realloop's own checks can object."""
    rows = C.read_manifest(V.POOL)
    for r in rows:
        if r["key"] == key:
            r["image"], r["sha256"] = str(image), C.sha256_file(image)
    C.write_manifest(V.POOL, rows)
    meta = [json.loads(ln) for ln in pathlib.Path(V.POOL_META).read_text().splitlines() if ln.strip()]
    for m in meta:
        if m["key"] == key:
            m["dhash"] = dhash
    write_rows(V.POOL_META, meta)
    restamp_admit()
    restamp_select("pool_meta")


# ------------------------------------------------------ synthetic executor
NOISE = {0: 0.0, 1: 0.002, 2: -0.002}
FACTOR = {"dev": 1.0, "ood22": 0.7, "ood23": 0.6, "imageweeds": 0.5, "test": 0.95}
BAD_EFFECT = -0.05


class Executor:
    """What inc/train.py writes, with dev scores from a known model: a cold run
    scores 0.40 + 0.02 per verified increment it holds - 0.05 if it holds any
    UNVERIFIED image; a cand scores its parent + 0.02 (UNVERIFIED: - 0.05); a
    null its parent; a soup the mean of its cands + 0.001."""

    def __init__(self, verified_keys, unverified_keys):
        self.verified_keys = verified_keys        # {name: set of keys}
        self.unverified_keys = unverified_keys
        self.seen = []

    def __call__(self, spec):
        self.seen.append(spec)
        out = pathlib.Path(spec["out_dir"])
        w = out / "weights" / "final.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "final":
            if w.is_symlink() or w.exists():
                w.unlink()
            os.symlink(spec["init"], w)
        else:
            w.write_bytes(("weights of %s" % spec["run_id"]).encode())
        for exam in spec["exams"]:
            v, a = self.values(spec, exam)
            self._score(out / "scores" / ("%s.json" % exam), exam, v, a, w)
        (out / "run.json").write_text(json.dumps({"status": "done", "attempt": 1, "seconds": 60.0,
                                                  "error": None, "weights_sha256": C.sha256_file(w)}))
        return "COMPLETED"

    @staticmethod
    def parent(weights):
        s = json.loads((pathlib.Path(weights).parent.parent / "scores" / "dev.json").read_text())
        return s["map50_95"], s["agnostic_map50_95"]

    def values(self, spec, exam):
        kind, rid = spec["kind"], spec["run_id"]
        if kind in ("base", "union"):
            keys = set(keys_of(spec["train_manifest"]))
            n = sum(1 for ks in self.verified_keys.values() if ks <= keys)
            bad = bool(keys & self.unverified_keys)
            s = NOISE[spec["recipe"]["seed"]]
            return 0.40 + 0.02 * n + BAD_EFFECT * bad + s, 0.60 + 0.02 * n + s
        if kind in ("cand", "null"):
            pv, pa = self.parent(spec["init"])
            name = rid.split("__")[1].split("_", 1)[1]
            eff = (BAD_EFFECT if name == RL.UNVERIFIED else 0.02) if kind == "cand" else 0.0
            s = NOISE[spec["recipe"]["seed"]]
            return pv + eff + s, pa + max(eff, 0.0) + s
        if kind == "soup":
            vals = [self.parent(x) for x in spec["soup_of"]]
            return (statistics.fmean(v for v, _ in vals) + 0.001,
                    statistics.fmean(a for _, a in vals) + 0.001)
        pv, pa = self.parent(spec["init"])            # final
        return pv * FACTOR[exam], pa * FACTOR[exam]

    @staticmethod
    def _score(path, exam, v, a, weights):
        other = 0 if exam in ("dev", "test") else 25
        n_gt = {s: 40 for s in G.SPECIES}
        n_gt["OtherPlant"] = other
        per_class = {s: v for s in G.SPECIES}
        if other:
            per_class["OtherPlant"] = 0.5 * v
        s = {"exam": exam, "scorer_sha256": "TEST-" + "5" * 64, "production": False,
             "deviations": ["LOCK.json not checked"],
             "manifest_sha256": hashlib.sha256(("m/" + exam).encode()).hexdigest(),
             "key_order_sha256": hashlib.sha256(("k/" + exam).encode()).hexdigest(),
             "weights_sha256": C.sha256_file(weights), "n_images": 20,
             "map50_95": v, "map50": min(1.0, v + 0.2), "agnostic_map50_95": a,
             "agnostic_map50": min(1.0, a + 0.2), "per_class": per_class, "n_gt": n_gt,
             "image_correct": "1" * 20, "species_map50_95": v, "species_map50": v + 0.2}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s))


class Clock:
    def __init__(self):
        self.t = 1.8e9

    def __call__(self):
        return self.t


def state_of(exp):
    return json.loads(D.Paths(exp).state.read_text())


def ledger_of(exp):
    p = D.Paths(exp).ledger
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def all_specs(exp):
    runs = D.Paths(exp).runs
    return [json.loads((runs / d / "spec.json").read_text()) for d in sorted(os.listdir(runs))]


def drive(exp, fb, executor, max_iter=300):
    clock = Clock()
    for _ in range(max_iter):
        ran = fb.run_pending(executor)
        before = len(fb.submissions)
        D.Driver(exp, backend=fb, clock=clock, quiet=True).advance()
        st = state_of(exp)
        if st["done"] or (ran == 0 and len(fb.submissions) == before):
            return st
    return state_of(exp)


def not_built(exp):
    p = D.Paths(exp)
    return not p.exp_json.exists() and not p.state.exists()


# The default build's digest (build_digest of real_t, test_realloop's build), recorded with inc/realloop.py and
# inc/select.py as they were before --increment-sources existed, on this world (2026-09-27; the same value under
# Python 3.10 with numpy 2.2.6 and sklearn 1.7.2, and Python 3.12 with numpy 2.4.4 and sklearn 1.8.0).
DEFAULT_BUILD_DIGEST = "b14ae393f0b32694e2add05cf1be66e596362f35f88a5e989009b25a66f5b6fd"
VOLATILE = ("created", "created_utc", "built_utc", "initialised_utc", "seconds")      # timestamps and timings


def drop_volatile(o):
    """o without its VOLATILE keys, at any depth."""
    if isinstance(o, dict):
        return {k: drop_volatile(v) for k, v in o.items() if k not in VOLATILE}
    return [drop_volatile(v) for v in o] if isinstance(o, list) else o


def build_digest(exp):
    """sha256 of a chain build as the driver and a reader see it: exp.json (the
    definition driver init compares), build_summary.json and every file in
    manifests/. This run's temporary directory becomes '<TMP>', every 64-hex
    sha256 '<sha256>' (they hash bytes that carry those paths, or PNG bytes),
    and the VOLATILE keys are dropped at any depth. So it pins the definition,
    its step1 block included, the counts and every drawn row across machines."""
    paths = D.Paths(exp)

    def norm(text):
        for t in sorted({str(TMP.resolve()), str(TMP)}, key=len, reverse=True):
            text = text.replace(t, "<TMP>")
        return re.sub(r"\b[0-9a-f]{64}\b", "<sha256>", text)

    files = {"exp.json": paths.exp_json, "build_summary.json": paths.root / P.BUILD_SUMMARY}
    files.update(("manifests/" + f.name, f) for f in sorted(paths.manifests.iterdir()) if f.is_file())
    parts = {}
    for name, f in files.items():
        text = f.read_text()
        parts[name] = norm(json.dumps(drop_volatile(json.loads(text)), sort_keys=True) if f.suffix == ".json"
                           else text)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------ build-baseline
def bad_manifest(name, mutate, rows):
    """A copy of rows (and their labels) under TMP/bad/<name>, changed by mutate(rows, label_dir)."""
    d = TMP / "bad" / name
    ldir = d / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        lab = ldir / (r["key"] + ".txt")
        shutil.copyfile(r["label"], lab)
        out.append(dict(r, label=str(lab), label_sha256=C.sha256_file(lab)))
    out = mutate(out, ldir) or out
    path = d / "manifest.jsonl"
    with open(path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def test_baseline(W):
    print("build-baseline")
    base = base_b()
    base_rows = C.read_manifest(base)
    fb = D.FakeBackend()
    summary, defn, _ = P.build_baseline("base_b_t", base, testing=True, backend=fb, quiet=True)
    paths = D.Paths("base_b_t")
    copy = paths.manifests / "base_B.jsonl"
    check("baseline: the manifest is copied byte for byte and named after it",
          copy.read_bytes() == base.read_bytes() and defn["base"]["name"] == "base_B"
          and defn["base"]["manifest"] == str(copy) and defn["base"]["source_manifest"] == str(base))
    check("baseline: a 'baseline' experiment, cold protocol recipe, seeds 0-2, every exam at the end",
          defn["type"] == "baseline" and defn["base"]["recipe"] == P.cold_recipe()
          and defn["seeds"] == [0, 1, 2] and defn["final_exams"] == list(D.FINAL_EXAMS)
          and defn["builder"] == "inc.pilot build-baseline")
    sm = json.loads((paths.root / P.BUILD_SUMMARY).read_text())
    want_boxes = C.class_counts(base_rows)
    want_src = collections.Counter(r["source"] for r in base_rows)
    by_src = {s: C.class_counts([r for r in base_rows if r["source"] == s]) for s in want_src}
    check("baseline summary: manifest sha256, images, boxes per class, images and boxes per source",
          sm["manifest"]["manifest_sha256"] == C.sha256_file(base) and sm["manifest"]["images"] == len(base_rows)
          and sm["manifest"]["boxes"] == want_boxes and sm["manifest"]["sources"] == dict(want_src)
          and {s: v["boxes"] for s, v in sm["manifest"]["by_source"].items()} == by_src
          and {s: v["images"] for s, v in sm["manifest"]["by_source"].items()} == dict(want_src)
          and sm["manifest"]["sessions"] == 8, sm["manifest"].get("sources"))
    check("baseline summary: the guard checked every image, no hit",
          sm["manifest"]["guard"]["checked"] == len(base_rows) and sm["manifest"]["guard"]["hits"] == 0
          and sm["never_train"]["matches_lock"] is True)
    sel = json.loads(select_summary_path().read_text())
    sb = sm["select_build"] or {}
    check("baseline summary: base_B is a select build's; that build read LOCK.json's train_core and the "
          "current never-train index",
          sb.get("train_core", {}).get("matches_lock") is True
          and sb["train_core"]["sha256"] == sel["inputs"]["train_core"]["sha256"]
          and sb.get("never_train", {}).get("matches_current") is True
          and sb["summary"]["sha256"] == C.sha256_file(select_summary_path()), sb)
    specs = [json.loads(pathlib.Path(s).read_text()) for s in fb.submissions[0]["specs"]]
    check("baseline: driver init submits 3 cold base runs on the copy, scored on dev only",
          len(fb.submissions) == 1 and len(specs) == 3
          and all(s["kind"] == "base" and s["exams"] == ["dev"] and s["train_manifest"] == str(copy)
                  and s["recipe"] == dict(P.cold_recipe(), seed=i) for i, s in enumerate(specs)))
    st = drive("base_b_t", fb, Executor({}, set()))
    finals = [s for s in all_specs("base_b_t") if s["kind"] == "final"]
    check("baseline: after the base runs, final runs of every seed on every exam (test included); done",
          st["done"] and len(finals) == 3 and all(s["exams"] == list(D.FINAL_EXAMS) for s in finals))
    rep = R.build("base_b_t")
    check("baseline: the report's base row has test over 3 seeds",
          rep["final"][0]["exams"]["test"]["twelve"]["n"] == 3)

    fb2 = D.FakeBackend()
    P.build_baseline("base_b_two", base, seeds="0,1", testing=True, backend=fb2, quiet=True)
    check("baseline --seeds 0,1: two base runs", fb2.submissions[0]["n"] == 2
          and json.loads(D.Paths("base_b_two").exp_json.read_text())["seeds"] == [0, 1])
    check("baseline: a second build under the same name refuses",
          raises(lambda: P.build_baseline("base_b_t", base, testing=True, backend=D.FakeBackend(), quiet=True),
                 P.PilotError, "already built"))
    elsewhere = TMP / "baseline_elsewhere" / "base_B.jsonl"
    elsewhere.parent.mkdir(parents=True)
    shutil.copyfile(base, elsewhere)
    sm_other, _, _ = P.build_baseline("bl_other", elsewhere, testing=True, init=False, quiet=True)
    check("baseline on a manifest no select build names: no select record",
          sm_other["select_build"] is None)

    small = base_rows[:12]

    def cls13(rows, ldir):
        with open(rows[3]["label"], "a") as fh:
            fh.write("13 0.500000 0.500000 0.100000 0.100000\n")
        rows[3]["label_sha256"] = C.sha256_file(rows[3]["label"])

    def no_session(rows, ldir):
        del rows[2]["session"]

    def dup_key(rows, ldir):
        rows[1]["key"] = rows[0]["key"]

    def near_eval(rows, ldir):
        pass

    for name, mutate, contains in (("cls13", cls13, "class"), ("nosession", no_session, "lack"),
                                   ("dupkey", dup_key, "unique")):
        path = bad_manifest(name, mutate, small)
        exp = "bl_" + name
        check("baseline refuses %s before anything is written" % name,
              raises(lambda: P.build_baseline(exp, path, testing=True, backend=D.FakeBackend(), quiet=True),
                     P.PilotError, contains) and not_built(exp) and not D.Paths(exp).manifests.exists())

    # an image within 6 bits of an evaluation image
    near_path = bad_manifest("neareval", near_eval, small)
    saved = C.NEVER_TRAIN_INDEX.read_bytes()
    data = json.loads(saved)
    data["entries"].append([C.dhash(small[5]["image"]) ^ 0b10101, "dev", "/x/planted.jpg"])
    data["min_expected"] += 1
    C.NEVER_TRAIN_INDEX.write_text(json.dumps(data))
    try:
        check("baseline refuses an image within 6 dHash bits of an evaluation image (fail closed)",
              raises(lambda: P.build_baseline("bl_near", near_path, testing=True, backend=D.FakeBackend(),
                                              quiet=True), P.PilotError, "never-train")
              and not_built("bl_near"))
    finally:
        C.NEVER_TRAIN_INDEX.write_bytes(saved)
    dev = C.manifest_path("dev")
    shutil.copyfile(near_path, dev)
    try:
        check("baseline refuses an evaluation split's manifest",
              raises(lambda: P.build_baseline("bl_dev", dev, testing=True, backend=D.FakeBackend(), quiet=True),
                     P.PilotError, "never trained on") and not_built("bl_dev"))
    finally:
        dev.unlink()
    for seeds in ("0,0", "x", "-1", ""):
        check("baseline refuses seeds %r" % seeds,
              raises(lambda: P.build_baseline("bl_seeds", base, seeds=seeds, testing=True,
                                              backend=D.FakeBackend(), quiet=True), P.PilotError, "seeds")
              and not_built("bl_seeds"))
    lock = C.LOCK_PATH.read_bytes()
    C.LOCK_PATH.unlink()
    try:
        check("a production baseline refuses without LOCK.json",
              raises(lambda: P.build_baseline("bl_prod", base, backend=D.FakeBackend(), quiet=True),
                     P.PilotError, "LOCK.json") and not_built("bl_prod"))
    finally:
        C.LOCK_PATH.write_bytes(lock)
    write_lock(nevertrain="0" * 64)
    try:
        check("a production baseline refuses a never-train index LOCK.json did not record",
              raises(lambda: P.build_baseline("bl_prod", base, backend=D.FakeBackend(), quiet=True),
                     P.PilotError, "not the one LOCK.json recorded") and not_built("bl_prod"))
    finally:
        C.LOCK_PATH.write_bytes(lock)
    with Edited():
        relock(train_core="1" * 64)
        check("a production baseline refuses a base_B selected under another train_core lock",
              raises(lambda: P.build_baseline("bl_prod", base, backend=D.FakeBackend(), quiet=True),
                     P.PilotError, "under other splits") and not_built("bl_prod"))

    # the CLI
    fb3 = D.FakeBackend()
    saved_sb = D.SlurmBackend
    D.SlurmBackend = lambda *a, **k: fb3
    try:
        rc = P.main(["build-baseline", "--exp", "bl_cli", "--manifest", str(base), "--seeds", "0,1,2",
                     "--testing", "--quiet"])
        rc_nomanifest = P.main(["build-baseline", "--exp", "bl_cli2", "--testing", "--quiet"])
        rc_seeds_b0 = P.main(["build-b0", "--exp", "bl_cli3", "--seeds", "0", "--testing", "--quiet"])
        rc_replay = P.main(["build-baseline", "--exp", "bl_cli4", "--manifest", str(base), "--testing",
                            "--replay-mode", "full", "--quiet"])
    finally:
        D.SlurmBackend = saved_sb
    check("CLI build-baseline builds and inits (exit 0); without --manifest, with --seeds on build-b0 or "
          "--replay-mode on build-baseline it exits 1 and builds nothing",
          rc == 0 and fb3.submissions and fb3.submissions[0]["n"] == 3 and rc_nomanifest == 1
          and rc_seeds_b0 == 1 and rc_replay == 1
          and all(not_built(e) for e in ("bl_cli2", "bl_cli3", "bl_cli4")))


# --------------------------------------------------------------- realloop
def test_units():
    print("realloop units")
    check("sequence N=6: V1 V2 UNVERIFIED V3 OTHER_HEAVY V4 V5 V6",
          RL.sequence(6) == ["V1", "V2", "UNVERIFIED", "V3", "OTHER_HEAVY", "V4", "V5", "V6"])
    check("sequence N=1..3 keeps the two first, then UNVERIFIED, the third, OTHER_HEAVY",
          RL.sequence(1) == ["V1", "UNVERIFIED", "OTHER_HEAVY"]
          and RL.sequence(2) == ["V1", "V2", "UNVERIFIED", "OTHER_HEAVY"]
          and RL.sequence(3) == ["V1", "V2", "UNVERIFIED", "V3", "OTHER_HEAVY"])
    check("sequence refuses N < 1", raises(lambda: RL.sequence(0), RL.RealLoopError))
    check("choose_source: most eligible among those with >= M, ties to the lower name",
          RL.choose_source({"a": 30, "b": 40, "c": 50}, 45) == "c"
          and RL.choose_source({"b": 40, "a": 40, "c": 10}, 20) == "a"
          and RL.choose_source({"a": 30, "b": 40}, 20) == "b")
    check("choose_source refuses when no source has M",
          raises(lambda: RL.choose_source({"a": 3, "b": 9}, 10), RL.RealLoopError, "b with 9"))
    check("recipes: a subset of the pilot's; unknown or repeated names refused",
          RL.parse_recipes("full,lora") == ["full", "lora"]
          and raises(lambda: RL.parse_recipes("full,bogus"), RL.RealLoopError)
          and raises(lambda: RL.parse_recipes("full,full"), RL.RealLoopError)
          and raises(lambda: RL.parse_recipes(""), RL.RealLoopError))
    check("default size: 10% of the base, rounded", RL.default_size(220) == 22 and RL.default_size(3) == 1)


def test_realloop(W):
    print("realloop build")
    exp = "real_t"
    base = base_b()
    base_rows = C.read_manifest(base)
    fb = D.FakeBackend()
    summary, defn, _ = RL.build(exp, base=base, replay_mode="full", recipes="full,lora", testing=True,
                                backend=fb, quiet=True)
    got = build_digest(exp)
    check("the default build is the one built before --increment-sources existed: its digest (exp.json, "
          "build_summary.json and every manifest; paths, sha256s and timestamps left out) is the one recorded then",
          got == DEFAULT_BUILD_DIGEST, got)
    paths = D.Paths(exp)
    m = RL.default_size(len(base_rows))
    seq = ["V1", "V2", "UNVERIFIED", "V3", "OTHER_HEAVY", "V4", "V5", "V6"]
    steps = {s["name"]: s for s in defn["steps"]}
    rows = {n: C.read_manifest(steps[n]["manifest"]) for n in seq}
    check("the base is base_B.jsonl, copied byte for byte, cold recipe",
          (paths.manifests / "base_B.jsonl").read_bytes() == base.read_bytes()
          and defn["base"]["recipe"] == P.cold_recipe() and defn["base"]["n_images"] == len(base_rows))
    check("M defaults to 10%% of base B (%d of %d) and every increment holds exactly M" % (m, len(base_rows)),
          m == int(round(0.1 * len(base_rows))) and summary["size"]["images_per_increment"] == m
          and all(len(rows[n]) == m and steps[n]["n_images"] == m for n in seq))
    check("sequence and clean flags: every verified increment clean, UNVERIFIED not",
          [s["name"] for s in defn["steps"]] == seq
          and all(steps[n]["clean"] is (n != "UNVERIFIED") for n in seq) and summary["clean"] == [
              n for n in seq if n != "UNVERIFIED"])
    parts = [("base", base_rows)] + [(n, rows[n]) for n in seq]
    clash = []
    for i, (a, ra) in enumerate(parts):
        for b, rb in parts[:i]:
            for f in ("key", "image", "sha256"):
                if {r[f] for r in ra} & {r[f] for r in rb}:
                    clash.append((a, b, f))
    check("base and every increment pairwise disjoint by key, image path and sha256", not clash, clash[:3])

    # V1..V6 and OTHER_HEAVY are select's own draw
    ref = TMP / "ref" / exp
    S.increments(exp, 6, m, other_heavy=True, base_dir=pathlib.Path(V.STEP1), exp_dir=ref)
    same = all((ref / "manifests" / (S.INC_NAME % j)).read_bytes() == pathlib.Path(steps["V%d" % j]["manifest"])
               .read_bytes() for j in range(1, 7))
    check("V1..V6 and OTHER_HEAVY are select.increments' draw, byte for byte",
          same and (ref / "manifests" / S.INC_OTHER).read_bytes()
          == pathlib.Path(steps["OTHER_HEAVY"]["manifest"]).read_bytes())
    pool_keys = set(keys_of(pathlib.Path(V.STEP1) / S.POOL))
    ver_keys = {r["key"] for r in W["verified"]}
    check("verified increments come from the increment pool; OTHER_HEAVY is OtherPlant-heavy",
          all({r["key"] for r in rows[n]} <= pool_keys for n in seq if n != "UNVERIFIED")
          and summary["increments"]["OTHER_HEAVY"]["boxes"]["OtherPlant"]
          >= 0.5 * summary["increments"]["OTHER_HEAVY"]["boxes_total"])

    # UNVERIFIED
    un = rows["UNVERIFIED"]
    info = summary["unverified"]
    pool_by_key = {r["key"]: r for r in W["pool"]}
    n_plant = len(W["planted"])
    check("UNVERIFIED: harvU1, chosen on eligible images (harvU2 has more non-admitted, fewer eligible)",
          info["source"] == "harvU1" and steps["UNVERIFIED"]["source"] == "harvU1"
          and info["candidates"]["harvU2"]["non_admitted"] > info["candidates"]["harvU1"]["non_admitted"]
          and info["candidates"]["harvU1"] == {"non_admitted": U1_CONFLICT + U1_UNKNOWN + n_plant,
                                               "conflict": U1_CONFLICT, "unknown": U1_UNKNOWN + n_plant,
                                               "excluded_near_dup": n_plant - (U1_ELIGIBLE - U1_CONFLICT
                                                                               - U1_UNKNOWN),
                                               "eligible": U1_ELIGIBLE}
          and info["candidates"]["harvU2"]["eligible"] == U2_N - U2_NEAR
          and info["candidates"]["harvU3"]["non_admitted"] == U3_N, info["candidates"])
    check("UNVERIFIED: only non-admitted images of harvU1, as pool.jsonl's rows (labels as joined)",
          all(r["key"] not in ver_keys and r["source"] == "harvU1" and r == {k: pool_by_key[r["key"]][k]
                                                                            for k in C.MANIFEST_KEYS}
              for r in un))
    ex = {e["key"]: e for e in info["excluded_near_dup_examples"]}
    wrong_owner = [p for p in W["planted"] if p["bits"] <= NEAR_DUP_BITS and (
        p["key"] not in ex or ex[p["key"]]["near"] != "%s/%s" % (p["owner"], p["target"])
        or ex[p["key"]]["bits"] != p["bits"])]
    far = [p["key"] for p in W["planted"] if p["bits"] > NEAR_DUP_BITS]
    check("UNVERIFIED: the images planted 1-3 bits from an image of V1, V4, OTHER_HEAVY or base B's harvested "
          "part are excluded, each listed with the image it is near; the one 4 bits away stays eligible",
          not wrong_owner and far and not set(far) & set(ex)
          and info["candidates"]["harvU2"]["excluded_near_dup"] == U2_NEAR, wrong_owner[:2])
    taken = {r["key"] for n in seq if n != "UNVERIFIED" for r in rows[n]} | {r["key"] for r in base_rows}
    th = [W["meta"][k]["dhash"] for k in taken if k in W["meta"]] + list(W["core_hash"])
    near = [r["key"] for r in un if any(bits(W["meta"][r["key"]]["dhash"], h) <= NEAR_DUP_BITS for h in th)]
    check("UNVERIFIED: no image within %d bits of the base or a verified increment" % NEAR_DUP_BITS, not near,
          near[:3])
    wrong = [e for e in info["excluded_near_dup_examples"]
             if not any(bits(W["meta"][e["key"]]["dhash"], h) <= NEAR_DUP_BITS for h in th)]
    check("UNVERIFIED: every near-dup exclusion is a real near copy of a taken image", not wrong, wrong[:2])
    # the draw, recomputed from the rule: harvU1's non-admitted images minus the
    # planted near-copies, sorted by key; default_rng(stable_int(exp + "/unverified"))
    excluded = {p["key"] for p in W["planted"] if p["bits"] <= NEAR_DUP_BITS}
    elig = sorted(k for k, v in W["verdict"].items() if k.startswith("u1_") and k not in excluded)
    pick = np.random.default_rng(C.stable_int(exp + "/unverified")).choice(len(elig), size=m, replace=False)
    want = [elig[i] for i in sorted(int(x) for x in pick)]
    check("UNVERIFIED: the drawn keys are the rule's (eligible sorted by key, seed text exp + '/unverified')",
          len(elig) == U1_ELIGIBLE and [r["key"] for r in un] == want and info["seed_text"] == exp + "/unverified",
          ([r["key"] for r in un][:4], want[:4]))
    kinds = collections.Counter(W["verdict"][r["key"]] for r in un)
    cbox = [c for c in W["conflicts"] if c["key"] in {r["key"] for r in un}]
    check("UNVERIFIED: verdict mix and conflict boxes recorded as recounted here",
          info["verdicts"] == {k: v for k, v in sorted(kinds.items())} and info["conflict_boxes"] == len(cbox)
          and sum(info["conflict_pairs"].values()) == len(cbox)
          and info["source_admit_summary"]["images"] == {"admitted": 10, "conflict": U1_CONFLICT,
                                                         "unknown": U1_UNKNOWN + n_plant}, info["verdicts"])
    check("summary: per-source and per-species counts for every increment",
          all(summary["increments"][n]["boxes"] == C.class_counts(rows[n])
              and summary["increments"][n]["sources"] == dict(collections.Counter(r["source"] for r in rows[n]))
              for n in seq))

    # definition
    check("recipes are the pilot's own definitions; truth recipe = base recipe = cold",
          defn["recipes"] == {r: P.inc_recipes()[r] for r in ("full", "lora")}
          and defn["truth_recipe"] == P.cold_recipe() and defn["truth"] is True
          and defn["replay_mode"] == "full" and defn["type"] == "chain" and defn["builder"] == RL.BUILDER)
    check("the attribution scope says what the driver does not run",
          defn["attribution_scope"]["not_run"].keys() == {"4", "5"})
    sel = json.loads(select_summary_path().read_text())
    prov = summary["select_provenance"]
    check("Step 1 files: select's train_core and never-train index recorded (exp.json and summary); that "
          "build read LOCK.json's train_core and the current index",
          defn["step1"]["train_core"] == {k: sel["inputs"]["train_core"][k] for k in ("path", "sha256")}
          and defn["step1"]["never_train"] == {k: sel["inputs"]["never_train"][k] for k in ("path", "sha256")}
          and summary["step1"] == defn["step1"]
          and prov["train_core"]["matches_lock"] is True and prov["never_train"]["matches_current"] is True,
          prov)
    check("the baselines already built on the same base_B bytes are recorded",
          summary["same_base_baselines"] == ["base_b_t", "base_b_two", "bl_cli"],
          summary["same_base_baselines"])
    check("exp.json holds the definition, build_summary.json the counts, select's parameters and the "
          "unverified section",
          not {"unverified", "select_increments", "increments", "size"} & set(defn)
          and {"unverified", "select_increments", "increments", "size", "base"} <= set(summary))

    # driver init
    st = state_of(exp)
    specs = [json.loads(pathlib.Path(s).read_text()) for s in fb.submissions[0]["specs"]]
    base_specs = [s for s in specs if s["kind"] == "base"]
    truth_specs = {s["run_id"]: s for s in specs if s["kind"] == "union"}
    check("driver init: state.json, one array of 3 base + 8x3 truth runs, chains start",
          set(st["chains"]) == {"full", "lora"} and len(fb.submissions) == 1 and len(base_specs) == 3
          and len(truth_specs) == 24 and all(s["exams"] == ["dev"] for s in specs))

    def truth_keys(n):
        tag = "s%02d_%s" % (seq.index(n) + 1, n)
        return set(keys_of(truth_specs["truth__%s__union__s0" % tag]["train_manifest"]))

    bk = {r["key"] for r in base_rows}
    ks = {n: {r["key"] for r in rows[n]} for n in seq}
    check("truth arm: UNVERIFIED's 'with' = base + V1 + V2 + UNVERIFIED; V3's = base + V1 + V2 + V3 (T never "
          "holds UNVERIFIED)",
          truth_keys("UNVERIFIED") == bk | ks["V1"] | ks["V2"] | ks["UNVERIFIED"]
          and truth_keys("V3") == bk | ks["V1"] | ks["V2"] | ks["V3"]
          and truth_keys("V6") == bk | set().union(*(ks[n] for n in seq if n != "UNVERIFIED")))

    ex = Executor({n: ks[n] for n in seq if n != "UNVERIFIED"}, ks["UNVERIFIED"])
    st = drive(exp, fb, ex)
    gates = {(e["chain"], e["step"]): e["decision"]["verdict"] for e in ledger_of(exp) if e["type"] == "gate"}
    truths = {e["step"]: e["detail"]["verdict"] for e in ledger_of(exp) if e["type"] == "truth"}
    check("driven to done: every chain accepts the verified increments and rejects UNVERIFIED",
          st["done"] and all(gates.get((r, n)) == (G.REJECT if n == "UNVERIFIED" else G.ACCEPT)
                             for r in ("full", "lora") for n in seq), gates)
    check("the truth arm: verified increments help, UNVERIFIED hurts",
          all(truths.get(n) == (G.HURTS if n == "UNVERIFIED" else G.HELPS) for n in seq), truths)
    cand4 = set(keys_of(st["chains"]["full"]["steps"]["4"]["manifests"]["cand"]["path"]))
    check("full rehearsal: step V3's cand = base + V1 + V2 + V3 (the rejected UNVERIFIED is not in the pool)",
          cand4 == bk | ks["V1"] | ks["V2"] | ks["V3"])
    finals = {s["run_id"]: s for s in all_specs(exp) if s["kind"] == "final"}
    check("final runs: each chain's incumbent, the base seeds and T_final, on every exam",
          set(finals) == {"final__full__incumbent", "final__lora__incumbent", "final__base__s0",
                          "final__base__s1", "final__base__s2", "final__Tfinal__s0", "final__Tfinal__s1",
                          "final__Tfinal__s2"}
          and all(s["exams"] == list(D.FINAL_EXAMS) for s in finals.values()))
    rep = R.build(exp)
    md = (paths.root / "report.md").read_text()
    lines = [ln for ln in md.splitlines() if ln.startswith("- full:") or ln.startswith("- lora:")]
    check("the report: 8 steps; no Bswap section; UNVERIFIED's attribution per chain (REJECT, labels)",
          len(rep["steps"]) == 8 and "Attribution of Bswap" not in md and "## Attribution of UNVERIFIED" in md
          and all(rep["chains"][r]["bswap"] is None and rep["chains"][r]["unverified"] == {
              "verdict": G.REJECT, "class_vs_loc": G.LABELS, "attributed_to_labels": True}
              for r in ("full", "lora"))
          and rep["label_steps"] == {"unverified": "UNVERIFIED"}
          and all(("- %s: %s, class_vs_loc=%s, attributed to labels: yes" % (r, G.REJECT, G.LABELS)) in lines
                  for r in ("full", "lora"))
          and "UNVERIFIED: %d images of harvU1 that verify did not admit" % m in md, lines)
    return summary


def test_realloop_options(W):
    print("realloop options, determinism and refusals")
    base = base_b()
    fb = D.FakeBackend()
    _, d2, _ = RL.build("real_s", base=base, n_verified=2, size=15, replay_mode="sample", recipes="freeze",
                        truth=False, testing=True, backend=fb, quiet=True)
    st = state_of("real_s")
    check("--n-verified 2 --size 15 --replay-mode sample --recipes freeze --no-truth reach exp.json",
          [s["name"] for s in d2["steps"]] == ["V1", "V2", "UNVERIFIED", "OTHER_HEAVY"]
          and all(s["n_images"] == 15 for s in d2["steps"]) and d2["replay_mode"] == "sample"
          and d2["recipes"] == {"freeze": P.inc_recipes()["freeze"]} and d2["truth"] is False
          and st["truth"]["enabled"] is False
          and fb.submissions[0]["n"] == 3)
    check("a build without --gate-flips-mode writes gate {flips_mode: negative} (protocol v1) into exp.json and "
          "build_summary.json",
          d2["gate"] == {"flips_mode": "negative"}
          and json.loads(D.Paths("real_s").exp_json.read_text())["gate"] == {"flips_mode": "negative"}
          and json.loads((D.Paths("real_s").root / P.BUILD_SUMMARY).read_text())["gate"]
          == {"flips_mode": "negative"})

    s1, _, none = RL.build("real_det", base=base, replay_mode="full", recipes="full", testing=True,
                           init=False, quiet=True)
    mdir = D.Paths("real_det").manifests
    first = {p.name: p.read_bytes() for p in sorted(mdir.iterdir()) if p.name != S.INC_SUMMARY}
    s2, _, _ = RL.build("real_det", base=base, replay_mode="full", recipes="full", testing=True,
                        init=False, quiet=True)
    second = {p.name: p.read_bytes() for p in sorted(mdir.iterdir()) if p.name != S.INC_SUMMARY}
    strip = lambda s: {k: v for k, v in s.items() if k not in ("built_utc", "step1")}  # noqa: E731
    check("deterministic: a rebuild writes the same manifests and summary",
          none is None and first == second and strip(s1) == strip(s2) and not_built("real_det"), sorted(first))

    check("a second build under the same name refuses",
          raises(lambda: RL.build("real_t", base=base, replay_mode="full", recipes="full", testing=True,
                                  backend=D.FakeBackend(), quiet=True), P.PilotError, "already built"))
    for kw, contains in (({"replay_mode": "half", "recipes": "full"}, "replay mode"),
                         ({"replay_mode": "full", "recipes": "full,bogus"}, "--recipes"),
                         ({"replay_mode": "full", "recipes": "full", "n_verified": 0}, "n-verified"),
                         ({"replay_mode": "full", "recipes": "full", "size": 0}, "--size"),
                         ({"replay_mode": "full", "recipes": "full", "gate_flips_mode": "both"},
                          "gate flips mode")):
        check("refuses %s" % kw,
              raises(lambda: RL.build("real_bad", base=base, testing=True, backend=D.FakeBackend(), quiet=True,
                                      **kw), RL.RealLoopError, contains) and not_built("real_bad"))

    # a base that is not select's base_B
    other = pathlib.Path(V.STEP1) / "base_B_edited.jsonl"
    lines = base.read_text().splitlines()
    other.write_text("\n".join(lines[1:]) + "\n")
    elsewhere = TMP / "elsewhere" / "base_B.jsonl"
    elsewhere.parent.mkdir(parents=True)
    shutil.copyfile(base, elsewhere)
    check("refuses a base that is not the base_B.jsonl select's build wrote",
          raises(lambda: RL.build("real_b1", base=other, replay_mode="full", recipes="full", testing=True,
                                  backend=D.FakeBackend(), quiet=True), RL.RealLoopError, "is not the")
          and raises(lambda: RL.build("real_b2", base=elsewhere, replay_mode="full", recipes="full",
                                      testing=True, backend=D.FakeBackend(), quiet=True),
                     RL.RealLoopError, "select_summary.json")
          and not_built("real_b1") and not_built("real_b2"))
    other.unlink()

    # verify's files disagree with admit_summary.json
    saved = pathlib.Path(V.CONFLICTS).read_bytes()
    lines = saved.decode().splitlines()
    pathlib.Path(V.CONFLICTS).write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    try:
        check("refuses when conflicts.csv and admit_summary.json disagree on the image verdicts",
              raises(lambda: RL.build("real_c", base=base, replay_mode="full", recipes="full", testing=True,
                                      backend=D.FakeBackend(), quiet=True), RL.RealLoopError, "differ from")
              and not_built("real_c"))
    finally:
        pathlib.Path(V.CONFLICTS).write_bytes(saved)
    saved = pathlib.Path(V.POOL).read_bytes()
    pathlib.Path(V.POOL).write_bytes(saved + b"\n")
    try:
        check("refuses a pool.jsonl that admit_summary.json does not name",
              raises(lambda: RL.build("real_p", base=base, replay_mode="full", recipes="full", testing=True,
                                      backend=D.FakeBackend(), quiet=True), RL.RealLoopError, "does not name")
              and not_built("real_p"))
    finally:
        pathlib.Path(V.POOL).write_bytes(saved)

    # a never-train hit among the non-admitted images
    saved = C.NEVER_TRAIN_INDEX.read_bytes()
    data = json.loads(saved)
    data["entries"].append([W["meta"]["u3_002"]["dhash"] ^ 0b11, "test", "/x/planted.jpg"])
    data["min_expected"] += 1
    C.NEVER_TRAIN_INDEX.write_text(json.dumps(data))
    try:
        check("refuses when an image verify did not admit hits the never-train index (stale Step 1)",
              raises(lambda: RL.build("real_nt", base=base, replay_mode="full", recipes="full", testing=True,
                                      backend=D.FakeBackend(), quiet=True), RL.RealLoopError, "never-train")
              and not_built("real_nt"))
    finally:
        C.NEVER_TRAIN_INDEX.write_bytes(saved)

    # no source with M eligible images
    step1 = RL.load_step1(base)
    base_rows = C.read_manifest(base)
    taken = [("base_B", base_rows, {r["key"]: W["meta"][r["key"]]["dhash"] if r["key"] in W["meta"]
                                    else W["core_hash"][[c["key"] for c in W["core"]].index(r["key"])]
                                    for r in base_rows})]
    # against the base alone, only the copies planted next to base B stay out
    n_base_only = U1_CONFLICT + U1_UNKNOWN + sum(1 for p in W["planted"]
                                                 if p["owner"] != "base_B" or p["bits"] > NEAR_DUP_BITS)
    check("refuses when no harvested source has M eligible non-admitted images",
          raises(lambda: RL.draw_unverified("real_m", n_base_only + 1, step1, taken, C.NeverTrainGuard.load()),
                 RL.RealLoopError, "harvU1 with %d" % n_base_only))
    got, _ = RL.draw_unverified("real_m", n_base_only, step1, taken, C.NeverTrainGuard.load())
    check("M = the chosen source's eligible count draws all of them",
          len(got) == n_base_only and {r["source"] for r in got} == {"harvU1"})

    test_refusals(W, base)

    lock = C.LOCK_PATH.read_bytes()
    C.LOCK_PATH.unlink()
    try:
        check("a production realloop build refuses without LOCK.json",
              raises(lambda: RL.build("real_prod", base=base, replay_mode="full", recipes="full",
                                      backend=D.FakeBackend(), quiet=True), P.PilotError, "LOCK.json")
              and not_built("real_prod"))
    finally:
        C.LOCK_PATH.write_bytes(lock)

    fb4 = D.FakeBackend()
    saved_sb = D.SlurmBackend
    D.SlurmBackend = lambda *a, **k: fb4
    try:
        rc = RL.main(["build", "--exp", "real_cli", "--replay-mode", "full",
                      "--recipes", "full,freeze,lora", "--testing", "--quiet"])
        rc_bad = RL.main(["build", "--exp", "real_cli2", "--replay-mode", "full", "--recipes", "sgd",
                          "--testing", "--quiet"])
        rc_net = RL.main(["build", "--exp", "real_cli_net", "--replay-mode", "full", "--recipes", "full",
                          "--gate-flips-mode", "net", "--testing", "--quiet"])
    finally:
        D.SlurmBackend = saved_sb
    dn = json.loads(D.Paths("real_cli_net").exp_json.read_text()) if rc_net == 0 else {}
    bn = json.loads((D.Paths("real_cli_net").root / P.BUILD_SUMMARY).read_text()) if rc_net == 0 else {}
    lines = D.Driver("real_cli_net").status() if rc_net == 0 else []
    check("CLI: --gate-flips-mode net writes gate {flips_mode: net} into exp.json and build_summary.json, the "
          "driver takes it (status: gate flips net); without the flag the gate block is negative",
          rc_net == 0 and dn.get("gate") == {"flips_mode": "net"} and bn.get("gate") == {"flips_mode": "net"}
          and D.gate_config(dn) == G.GateConfig(flips_mode="net", require_production=False)
          and any(" chain full" in ln and "(gate flips net)" in ln for ln in lines)
          and json.loads(D.Paths("real_cli").exp_json.read_text()).get("gate") == {"flips_mode": "negative"},
          lines)
    d = json.loads(D.Paths("real_cli").exp_json.read_text()) if rc == 0 else {}
    check("CLI: realloop build with the default base and N, all three recipes (exit 0); a bad recipe exits 1",
          rc == 0 and d.get("recipes") == P.inc_recipes() and len(d.get("steps", [])) == 8
          and fb4.submissions and fb4.submissions[0]["n"] == 27 and rc_bad == 1 and not_built("real_cli2"))


def refuses(exp, contains, **kw):
    """A testing realloop build of exp refuses (a build error) with `contains`
    in the message, and defines nothing. Any other exception is a failed
    check, not a crash of the run."""
    args = dict(base=base_b(), replay_mode="full", recipes="full", testing=True, backend=D.FakeBackend(),
                quiet=True)
    args.update(kw)
    try:
        RL.build(exp, **args)
    except P.PilotError as e:
        if contains not in str(e):
            print("       refused without %r: %s" % (contains, e))
            return False
        return not_built(exp)
    except Exception as e:  # noqa: BLE001 - reported as the check's failure
        print("       raised %s, not a build refusal: %s" % (type(e).__name__, e))
        return False
    print("       built; expected a refusal with %r" % contains)
    return False


def test_refusals(W, base):
    """Step 1 edited so that exactly one of realloop's checks can object."""
    print("realloop refusals, one check each")
    # verify's files are not the ones select read (same rows, other bytes; admit re-stamped)
    with Edited():
        rows = C.read_manifest(V.VERIFIED_MANIFEST)
        write_rows(V.VERIFIED_MANIFEST, rows[::-1])
        restamp_admit()
        check("refuses a verified.jsonl the select build did not read (admit re-stamped)",
              refuses("real_r1", "verify's verified.jsonl"))
    with Edited():
        lines = pathlib.Path(V.POOL_META).read_text().splitlines()
        pathlib.Path(V.POOL_META).write_text("\n".join(lines[::-1]) + "\n")
        restamp_admit()
        check("refuses a pool_meta.jsonl the select build did not read (admit re-stamped)",
              refuses("real_r2", "verify's pool_meta.jsonl"))

    # conflicts.csv names an admitted image (the verdict counts stay the same)
    with Edited():
        admitted = W["verified"][0]
        with open(V.CONFLICTS, "a", newline="") as fh:
            csv.writer(fh).writerow([admitted["source"], admitted["image"], admitted["key"], 0, 0,
                                     C.CLASS_NAMES[0], 1, C.CLASS_NAMES[1], 0.9, 0.5, 0, ""])
        check("refuses a conflicts.csv row for an image verify admitted", refuses("real_r3", "were admitted"))

    # a verified row whose label is not the pool.jsonl row's (every summary re-stamped)
    with Edited():
        rows = C.read_manifest(V.VERIFIED_MANIFEST)
        lab = TMP / "edited" / "relabelled.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text(pathlib.Path(rows[0]["label"]).read_text() + "0 0.5 0.5 0.1 0.1\n")
        rows[0] = dict(rows[0], label=str(lab), label_sha256=C.sha256_file(lab))
        C.write_manifest(V.VERIFIED_MANIFEST, rows)
        restamp_admit()
        restamp_select("verified")
        check("refuses a verified.jsonl row whose label differs from its pool.jsonl row",
              refuses("real_r4", "are not pool rows"))

    # base_B itself: a class-13 box (select's summary re-stamped), an image near an evaluation image
    selected = sorted(C.read_manifest(pathlib.Path(V.STEP1) / S.BASE_SELECTED), key=lambda r: r["key"])
    with Edited():
        rows = C.read_manifest(base)
        lab = TMP / "edited" / "class13.txt"
        lab.write_text(pathlib.Path(selected[0]["label"]).read_text() + "13 0.5 0.5 0.1 0.1\n")
        rows = [dict(r, label=str(lab), label_sha256=C.sha256_file(lab)) if r["key"] == selected[0]["key"]
                else r for r in rows]
        C.write_manifest(base, rows)
        restamp_select("base_B")
        check("refuses a base_B with a label class outside 0..12 (select's summary re-stamped)",
              refuses("real_r5", "the base"))
    with Edited():
        data = json.loads(C.NEVER_TRAIN_INDEX.read_text())
        data["entries"].append([W["meta"][selected[3]["key"]]["dhash"] ^ 0b1001, "test", "/x/planted.jpg"])
        data["min_expected"] += 1
        C.NEVER_TRAIN_INDEX.write_text(json.dumps(data))
        check("refuses a base_B image within 6 dHash bits of an evaluation image (a testing build)",
              refuses("real_r6", "the base"))

    # an increment's label bytes are not the manifest's: only the per-increment check reads them
    pool_rows = C.read_manifest(pathlib.Path(V.STEP1) / S.POOL)
    labels = [pathlib.Path(r["label"]) for r in pool_rows]
    with Edited(*labels):
        for lp in labels:
            lp.write_text(lp.read_text() + "0 0.5 0.5 0.1 0.1\n")
        check("refuses when a verified increment's label bytes differ from its manifest (increment V1)",
              refuses("real_r7", "increment V1"))
    u1 = [pathlib.Path(r["label"]) for r in W["pool"] if r["key"].startswith("u1_")]
    with Edited(*u1):
        for lp in u1:
            lp.write_text(lp.read_text() + "0 0.5 0.5 0.1 0.1\n")
        check("refuses when an UNVERIFIED image's label bytes differ from its pool row",
              refuses("real_r8", "increment UNVERIFIED"))

    # a non-admitted image that is a train_core photograph, or a harvested base image, byte for byte
    core = W["core"][0]
    copy = TMP / "edited" / "u1_corecopy.png"
    shutil.copyfile(core["image"], copy)
    with Edited():
        set_pool_image("u1_039", copy, W["core_hash"][0])
        check("refuses a non-admitted image byte-identical to a train_core photograph (select's check)",
              refuses("real_r9", "byte-identical to a train_core image"))
    copy = TMP / "edited" / "u1_basecopy.png"
    shutil.copyfile(selected[2]["image"], copy)
    with Edited():
        set_pool_image("u1_039", copy, W["meta"][selected[2]["key"]]["dhash"])
        check("refuses a non-admitted image that is the same image as a harvested image of base B",
              refuses("real_r10", "share a key, image path or image sha256"))

    # a base selected under an earlier lock: refused in production, recorded in testing
    with Edited():
        relock(train_core="1" * 64)
        check("a production build refuses a base_B selected under another train_core lock",
              refuses("real_r11", "it read train_core", testing=False))
    with Edited():
        data = json.loads(C.NEVER_TRAIN_INDEX.read_text())
        data["entries"].append([int.from_bytes(b"\x5a" * 8, "big"), "test", "/x/later.jpg"])
        data["min_expected"] += 1
        C.NEVER_TRAIN_INDEX.write_text(json.dumps(data))
        relock()
        check("a production build refuses a base_B selected under an earlier never-train index",
              refuses("real_r12", "it read the never-train index", testing=False))
        sm, _, _ = RL.build("real_r13", base=base, replay_mode="full", recipes="full", testing=True,
                            init=False, quiet=True)
        check("a testing build records the stale never-train index and goes on",
              sm["select_provenance"]["never_train"]["matches_current"] is False
              and sm["select_provenance"]["train_core"]["matches_lock"] is True)


def test_relevance(W):
    """The relevance filter (inc/relevance.py) on the verified draws; UNVERIFIED is never filtered."""
    print("realloop and the relevance filter")
    from weed_optimizer_framework.tools.inc import relevance as REL
    base = base_b()
    step1 = pathlib.Path(V.STEP1)
    default = step1 / REL.OUT_NAME
    assert not default.exists()
    check("a production build refuses without relevance.json (nothing is drawn unfiltered)",
          refuses("real_rel_p0", "relevance", testing=False))
    basis = np.linalg.qr(np.random.default_rng(11).normal(size=(EMB_D, EMB_D)))[0]   # make_world's basis
    scene_dir, class_dir = basis[:N_SCENES], basis[N_SCENES:N_SCENES + C.NC]
    # plant prompts span the species directions (five groups of the 13 class
    # directions); the non-plant ones point away from the field scenes. At
    # BioCLIP-2's logit scale (about 100) at least 95 % of train_core crops
    # read as plant, so the calibration check holds.
    groups = [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]
    plant_vecs = dict(zip(REL.PROMPTS["plant"], (class_dir[g].sum(0) for g in groups)))

    class Enc:
        model, name, logit_scale = "fake", "fake text tower", 100.0

        def __call__(self, prompts):
            return np.array([plant_vecs.get(p, -scene_dir.sum(0)) for p in prompts])
    path = TMP / "relevance" / "relevance.json"
    rel = REL.build(base_dir=step1, out=path, text_encoder=Enc())
    status = {s: e["status"] for s, e in rel["increment_pool"]["sources"].items()}
    not_passing = rel["increment_pool"]["not_passing"]
    check("the world's relevance: every source is plant-like; a source with fewer than %d usable crops in the "
          "increment pool is insufficient" % REL.MIN_CROPS,
          not_passing and rel["calibration"]["check"]["ok"] is True
          and all(e["status"] == (REL.PASS if e["crops_usable"] >= REL.MIN_CROPS else REL.INSUFFICIENT)
                  for e in rel["increment_pool"]["sources"].values()), status)

    exp = "real_rel"
    summary, defn, none = RL.build(exp, base=base, replay_mode="full", recipes="full", testing=True, init=False,
                                   quiet=True, relevance=path)
    steps = {s["name"]: s for s in defn["steps"]}
    verified = [n for n in steps if n != "UNVERIFIED"]
    rows = {n: C.read_manifest(steps[n]["manifest"]) for n in steps}
    real_t = D.Paths("real_t").manifests
    unfiltered = [r for n in verified for r in C.read_manifest(real_t / steps[n]["select_manifest"])]
    check("V* and OTHER_HEAVY hold no image of a source the relevance file does not pass (the unfiltered real_t "
          "draw holds some)",
          none is None and not [r for n in verified for r in rows[n] if status[r["source"]] != REL.PASS]
          and any(status[r["source"]] != REL.PASS for r in unfiltered))
    ref = TMP / "ref" / exp
    S.increments(exp, 6, summary["size"]["images_per_increment"], other_heavy=True, base_dir=step1, exp_dir=ref,
                 relevance=path)
    check("they are select.increments' filtered draw, byte for byte",
          all((ref / "manifests" / steps[n]["select_manifest"]).read_bytes()
              == pathlib.Path(steps[n]["manifest"]).read_bytes() for n in verified))
    ri = summary["relevance"]
    sha = C.sha256_file(path)
    check("the relevance file's sha256 is in exp.json's step1 files; the summary records the excluded sources "
          "and images",
          defn["step1"]["relevance"] == {"path": str(path), "sha256": sha} and ri["sha256"] == sha
          and ri["excluded_sources"] == {s: {"images": n, "status": status[s]} for s, n in not_passing.items()}
          and ri["excluded_images"] == sum(not_passing.values()), ri)
    un = summary["unverified"]
    check("UNVERIFIED is not filtered: the same source rule over the non-admitted images, its source's relevance "
          "status recorded",
          un["source"] == "harvU1" and len(rows["UNVERIFIED"]) == summary["size"]["images_per_increment"]
          and un["source_relevance"] == status.get("harvU1", "no image in the increment pool"), un)

    # UNVERIFIED's source FAILS relevance: harvU1's entry re-stamped consistently
    # (20 usable crops, median 0), since no source of this world looks
    # non-plant. load() accepts the file (a status that follows from its
    # numbers); the verified draws exclude harvU1 as before, UNVERIFIED is
    # still harvU1's.
    fail_u1 = TMP / "relevance_u1fail" / "relevance.json"
    d = json.loads(path.read_text())
    u1 = d["increment_pool"]["sources"]["harvU1"]
    u1.update(crops_usable=max(u1["crops_usable"], REL.MIN_CROPS), p_plant_median=0.0, status=REL.FAIL,
              passes=False)
    fail_u1.parent.mkdir(parents=True)
    fail_u1.write_text(json.dumps(d))
    sm_f, _, _ = RL.build("real_rel_u1fail", base=base, replay_mode="full", recipes="full", testing=True,
                          init=False, quiet=True, relevance=fail_u1)
    un_f = sm_f["unverified"]
    un_rows = C.read_manifest(D.Paths("real_rel_u1fail").manifests / RL.UNVERIFIED_MANIFEST)
    check("UNVERIFIED is not filtered when its source FAILS relevance: still harvU1's M non-admitted images, "
          "with source_relevance 'fail'; the verified draws exclude harvU1 (status fail)",
          un_f["source"] == "harvU1" and un_f["source_relevance"] == REL.FAIL
          and len(un_rows) == sm_f["size"]["images_per_increment"] and {r["source"] for r in un_rows} == {"harvU1"}
          and sm_f["relevance"]["excluded_sources"]["harvU1"]["status"] == REL.FAIL, un_f)

    # a relevance file whose calibration is degenerate: the old fake (plant
    # prompts on the scene and species sums, non-plant on a noise direction, at
    # logit scale 20) cannot tell this world's noisy train_core crops from
    # non-plants, so tau falls below 0.5; the build fails closed and a realloop
    # build refuses the file it wrote
    u_mix = scene_dir.sum(0) / np.sqrt(N_SCENES) + class_dir.sum(0) / np.sqrt(C.NC)

    class WeakEnc:
        model, name, logit_scale = "fake", "fake text tower", 20.0

        def __call__(self, prompts):
            return np.array([u_mix if p in REL.PROMPTS["plant"] else basis[-1] for p in prompts])
    deg = TMP / "relevance_deg" / "relevance.json"
    try:
        REL.build(base_dir=step1, out=deg, text_encoder=WeakEnc())
        deg_err = None
    except REL.RelevanceError as e:
        deg_err = str(e)
    check("a degenerate calibration: the relevance build fails closed, and a realloop build refuses its file "
          "before anything is written",
          deg_err is not None and "degenerate calibration" in deg_err and deg.exists()
          and refuses("real_rel_deg", "degenerate calibration", relevance=deg)
          and not D.Paths("real_rel_deg").manifests.exists(), deg_err)

    shutil.copyfile(path, default)
    try:
        sm, dp, _ = RL.build("real_rel_p1", base=base, replay_mode="full", recipes="full", testing=False, init=False,
                             quiet=True)
        check("a production build finds relevance.json next to the base and records it",
              dp["step1"]["relevance"] == {"path": str(default), "sha256": sha}
              and sm["relevance"]["excluded_sources"] == ri["excluded_sources"])
        check("an explicit --relevance that does not exist refuses",
              refuses("real_rel_p2", "no such file", relevance=TMP / "none.json"))
    finally:
        default.unlink()


def test_evidence(W):
    """--increment-sources evidence (source-level species evidence): no relevance file; UNVERIFIED unchanged."""
    print("realloop and the source-evidence criterion (--increment-sources evidence)")
    base = base_b()
    step1 = pathlib.Path(V.STEP1)
    admit = json.loads(pathlib.Path(V.ADMIT_SUMMARY).read_text())
    ver = {s: d["boxes"].get("verified", 0) for s, d in admit["per_slug"].items()}
    pool = sorted(C.read_manifest(step1 / S.POOL), key=lambda r: r["key"])
    n_src = collections.Counter(r["source"] for r in pool)
    k = ver["harvU1"] + 1                 # harvU1 falls below it, harvA and harvB do not
    check("the world: harvU1 holds fewer verified species boxes than harvA and harvB, and images in the "
          "increment pool", 0 < ver["harvU1"] < k <= min(ver["harvA"], ver["harvB"]) and n_src["harvU1"] > 0,
          (ver, dict(n_src)))
    sel = json.loads(select_summary_path().read_text())
    check("the world's two files agree (species_crops >= verified boxes for every source)",
          all(ver.get(s, 0) <= e["species_crops"] for s, e in sel["retrieval"]["source_evidence"].items())
          and all(ver[s] == 0 or s in sel["retrieval"]["source_evidence"] for s in ver))
    assert not (step1 / "relevance.json").exists()

    exp = "real_ev"
    summary, defn, none = RL.build(exp, base=base, replay_mode="full", recipes="full", testing=False, init=False,
                                   quiet=True, increment_sources="evidence", min_evidence=k)
    m = summary["size"]["images_per_increment"]
    steps = {s["name"]: s for s in defn["steps"]}
    verified = [n for n in steps if n != "UNVERIFIED"]
    rows = {n: C.read_manifest(steps[n]["manifest"]) for n in steps}
    check("a production build with --increment-sources evidence builds without relevance.json; the mode is in "
          "exp.json's and build_summary.json's step1 block",
          none is None and defn["testing"] is False and defn["step1"]["relevance"] is None
          and defn["step1"]["increment_sources"] == {"mode": "evidence", "min_evidence": k, "rule": S.EVIDENCE_RULE}
          and summary["step1"] == defn["step1"] and summary["relevance"]["applied"] is False
          and len(defn["steps"]) == 8, defn["step1"].get("increment_sources"))
    real_t = D.Paths("real_t").manifests
    unfiltered = [r for n in verified for r in C.read_manifest(real_t / steps[n]["select_manifest"])]
    check("V1..V6 and OTHER_HEAVY hold only images of evidenced sources (the default real_t draw holds harvU1's)",
          not [r for n in verified for r in rows[n] if ver[r["source"]] < k]
          and all(len(rows[n]) == m for n in verified) and any(r["source"] == "harvU1" for r in unfiltered))
    ref = TMP / "ref" / exp
    S.increments(exp, 6, m, other_heavy=True, base_dir=step1, exp_dir=ref, sources="evidence", min_evidence=k)
    check("they are select.increments' evidence draw, byte for byte",
          all((ref / "manifests" / steps[n]["select_manifest"]).read_bytes()
              == pathlib.Path(steps[n]["manifest"]).read_bytes() for n in verified))
    ei = summary["evidence"]
    cap = S.pool_capacity([r for r in pool if ver[r["source"]] >= k], S.read_clusters(step1 / S.CLUSTERS))
    check("build_summary.json 'evidence': the excluded and evidenced sources, the admit summary's sha256 (as in "
          "step1), and the capacity checked before anything was written",
          ei["excluded_sources"] == {"harvU1": {"images": n_src["harvU1"], "verified_boxes": ver["harvU1"]}}
          and ei["evidenced_sources"] == {s: {"verified_boxes": ver[s], "pool_images": n_src.get(s, 0)}
                                          for s in sorted(ver) if ver[s] >= k}
          and ei["admit_summary"]["sha256"] == defn["step1"]["admit_summary"]["sha256"]
          == C.sha256_file(V.ADMIT_SUMMARY) and ei["cross_check"]["checked"] is True
          and ei["capacity"]["needed_images"] == 7 * m and ei["capacity"]["needed_other_heavy_images"] == m
          and ei["capacity"]["evidenced_pool_images"] == cap["images"] == len(pool) - n_src["harvU1"]
          and ei["capacity"]["evidenced_other_heavy_images"] == cap["other_heavy_images"]
          and summary["select_increments"]["params"]["sources"] == "evidence", ei)
    un = summary["unverified"]
    base_rows = C.read_manifest(base)
    core_ix = {c["key"]: i for i, c in enumerate(W["core"])}
    taken = [("base_B", base_rows, {r["key"]: W["meta"][r["key"]]["dhash"] if r["key"] in W["meta"]
                                    else W["core_hash"][core_ix[r["key"]]] for r in base_rows})]
    taken += [(n, rows[n], {r["key"]: W["meta"][r["key"]]["dhash"] for r in rows[n]}) for n in verified]
    want, _ = RL.draw_unverified(exp, m, RL.load_step1(base), taken, C.NeverTrainGuard.load())
    check("UNVERIFIED is unchanged: the rule's draw from harvU1's non-admitted images, although harvU1 is not "
          "evidenced; its source's verified boxes are recorded",
          un["source"] == "harvU1" and rows["UNVERIFIED"] == want and len(want) == m
          and un["source_evidence"] == {"verified_boxes": ver["harvU1"], "evidenced": False}, un.get("source_evidence"))
    check("a default build records no mode: no 'increment_sources' in exp.json's or build_summary.json's step1 "
          "block (a missing key means 'relevance'; test_realloop pins the whole default build by its digest)",
          "increment_sources" not in json.loads(D.Paths("real_t").exp_json.read_text())["step1"]
          and "increment_sources" not in json.loads((D.Paths("real_t").root / P.BUILD_SUMMARY).read_text())["step1"])

    # --increment-sources relevance given explicitly = the default: the same manifests byte for byte, and the same
    # increments summary, build summary and definition apart from timestamps (and from the increments summary's
    # sha256 in step1, a hash of a file that holds its own timestamp)
    def settled(x):
        x = drop_volatile(json.loads(json.dumps(x)))
        x["step1"]["increments_summary"].pop("sha256")
        return x
    _, d_def, _ = RL.build("real_src", base=base, replay_mode="full", recipes="full", testing=True, init=False,
                           quiet=True)
    mdir = D.Paths("real_src").manifests
    first = {p.name: p.read_bytes() for p in sorted(mdir.iterdir())}
    s_def = json.loads((D.Paths("real_src").root / P.BUILD_SUMMARY).read_text())
    _, d_exp, _ = RL.build("real_src", base=base, replay_mode="full", recipes="full", testing=True, init=False,
                           quiet=True, increment_sources="relevance")
    second = {p.name: p.read_bytes() for p in sorted(mdir.iterdir())}
    s_exp = json.loads((D.Paths("real_src").root / P.BUILD_SUMMARY).read_text())
    inc_s = [drop_volatile(json.loads(x[S.INC_SUMMARY])) for x in (first, second)]
    check("--increment-sources relevance given explicitly: the default build's manifests byte for byte, and the "
          "same increments summary, build summary and definition apart from timestamps",
          first.keys() == second.keys() and S.INC_SUMMARY in first
          and all(first[n] == second[n] for n in first if n != S.INC_SUMMARY) and inc_s[0] == inc_s[1]
          and settled(s_def) == settled(s_exp) and settled(d_def) == settled(d_exp)
          and "increment_sources" not in d_exp["step1"])

    # refusals, before anything is written
    for name, contains, kw in (
            ("real_ev_r1", "does not read a relevance file", {"relevance": TMP / "none.json"}),
            ("real_ev_r2", "applies to --increment-sources evidence", {"min_evidence": 1,
                                                                       "increment_sources": "relevance"}),
            ("real_ev_r3", "--min-evidence must be", {"min_evidence": 0}),
            ("real_ev_r4", "increment sources", {"increment_sources": "stars"})):
        kw = dict({"increment_sources": "evidence"}, **kw)
        check("refuses %s" % sorted(kw.items()), refuses(name, contains, **kw))
    with Edited():
        a = json.loads(pathlib.Path(V.ADMIT_SUMMARY).read_text())
        a["per_slug"]["harvA"]["boxes"]["verified"] = sel["retrieval"]["source_evidence"]["harvA"]["species_crops"] + 1
        pathlib.Path(V.ADMIT_SUMMARY).write_text(json.dumps(a, indent=1, sort_keys=True))
        check("refuses when admit_summary.json counts more verified boxes for a source than select_summary.json "
              "counts species crops (the two files disagree)",
              refuses("real_ev_d1", "disagree", increment_sources="evidence", min_evidence=k)
              and not D.Paths("real_ev_d1").manifests.exists())
    with Edited():
        s = json.loads(select_summary_path().read_text())
        del s["retrieval"]["source_evidence"]["harvB"]
        select_summary_path().write_text(json.dumps(s, indent=1, sort_keys=True))
        check("refuses when select_summary.json's source evidence lacks a source with verified boxes",
              refuses("real_ev_d2", "disagree", increment_sources="evidence", min_evidence=k))
    n6 = cap["images"] // 7 + 1
    check("refuses when the evidenced pool cannot supply 6 verified + OTHER_HEAVY increments of M images, with "
          "the numbers, before anything is written",
          refuses("real_ev_c1", "holds %d images (%d needed)" % (cap["images"], 7 * n6), increment_sources="evidence",
                  min_evidence=k, size=n6)
          and not D.Paths("real_ev_c1").manifests.exists())
    # this world's evidenced pool is mostly OtherPlant-heavy (no size runs short of heavy images before it runs
    # short of images), so the heavy branch is reached with pool_capacity reporting 5 heavy images
    saved_cap = S.pool_capacity
    S.pool_capacity = lambda keep, cl: dict(saved_cap(keep, cl), other_heavy_images=5)
    try:
        ok = refuses("real_ev_c2", "5 of them in OtherPlant-heavy near-dup groups (6 needed)",
                     increment_sources="evidence", min_evidence=k, size=6, n_verified=1)
        ok = ok and refuses("real_ev_c2", "no OTHER_HEAVY increment of this size fits", increment_sources="evidence",
                            min_evidence=k, size=6, n_verified=1)
    finally:
        S.pool_capacity = saved_cap
    check("refuses when the evidenced pool's OtherPlant-heavy groups hold fewer than M images (pool_capacity "
          "stubbed to 5), before anything is written",
          ok and not D.Paths("real_ev_c2").manifests.exists())

    fb = D.FakeBackend()
    saved_sb = D.SlurmBackend
    D.SlurmBackend = lambda *a, **kw_: fb
    try:
        rc = RL.main(["build", "--exp", "real_ev_cli", "--replay-mode", "full", "--recipes", "full",
                      "--increment-sources", "evidence", "--min-evidence", str(k), "--testing", "--quiet"])
        rc_bad = RL.main(["build", "--exp", "real_ev_cli2", "--replay-mode", "full", "--recipes", "full",
                          "--min-evidence", "1", "--testing", "--quiet"])
    finally:
        D.SlurmBackend = saved_sb
    d = json.loads(D.Paths("real_ev_cli").exp_json.read_text()) if rc == 0 else {}
    check("CLI: --increment-sources evidence --min-evidence builds (exit 0, driver init); --min-evidence alone "
          "exits 1",
          rc == 0 and d["step1"]["increment_sources"]["mode"] == "evidence"
          and d["step1"]["increment_sources"]["min_evidence"] == k and len(fb.submissions) == 1
          and rc_bad == 1 and not_built("real_ev_cli2"))


def main():
    W = make_world()
    test_units()
    test_baseline(W)
    test_realloop(W)
    test_realloop_options(W)
    test_relevance(W)
    test_evidence(W)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
