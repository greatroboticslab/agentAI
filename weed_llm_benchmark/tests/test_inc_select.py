#!/usr/bin/env python3
"""INC Step 1.4-1.5: base B's harvested part is a mix of retrieval (with a
moderate gate), coverage and per-species caps; OtherPlant-only images are
budgeted; near-copies never straddle base and pool; and the increments are
cluster-balanced, fixed-size, disjoint and keep near-copies together.

The world here is synthetic, written in inc/verify.py's own formats (crops.csv
columns, emb_sNNN_of_NNN.npz shards, pool_verdicts.npz, pool_meta.jsonl,
admit_summary.json) plus a never-train index in common.py's format:
  * 8 on-domain scenes of very different sizes and one off-domain scene of
    100 images (its own source), 12 species and OtherPlant, as orthonormal
    directions. A box embedding is its scene's direction (strong) plus its
    class's direction plus noise, so the level-1 clusters must recover the 9
    scenes. In the off-domain scene the species boxes point away from their
    species, so they sit at the bottom of train_core's own scale.
  * train_core crops are their class's direction plus noise (kept off the
    scene directions), wide enough that on-domain pool boxes sit mid-scale.
  * Waterhemp is abundant in the pool but has only 10 train_core boxes, so a
    cap of 3x binds on it and on nothing else.
  * A few admitted images hold only an OtherPlant box too small to crop, so
    they have no feature. About 15 % of the pool is OtherPlant-only.
  * 10 near-copy pairs (1-3 dHash bits, another slug) and one chain of three
    (3 + 3 bits, ends 6 bits apart) are planted.

Pinned:
  * the build is deterministic;
  * the selection and the increment pool partition the verified set exactly;
  * base_B holds every train_core row verbatim;
  * retrieval scores equal the train_core percentile, computed here
    independently; OtherPlant-only images take their source's median;
  * the gate: the off-domain scene is never selected, though its cluster is
    as large as an on-domain one;
  * clusters get equal shares with small clusters' leftovers redistributed,
    and each sub-cluster is taken by descending retrieval score;
  * the OtherPlant-only budget holds under caps and refill (and binds);
  * the caps hold, and drops are last-in-first-out and minimal;
  * near-copies land on one side, and in one increment;
  * the never-train guard refuses a build or a draw that holds an
    evaluation image;
  * a changed rerun refuses without --force;
  * increments are fixed-size (10 % of base B by default), disjoint, drawn
    from the pool, balanced, and the OtherPlant-heavy one is heavy;
  * a verified image that is a train_core key, image or byte copy, or comes
    from a never-train or cwd12-copy dataset, is refused;
  * increments --relevance (inc/relevance.py on this build, injected fake
    text tower at logit scale 100 whose calibration check holds): the
    off-domain source fails and the few harvC forks left in the pool are
    insufficient; no increment holds an image of either (the unfiltered draw
    does), the summary and the draw's parameters carry the file's sha256,
    the exclusions and the near-dup groups split by them; an unfiltered draw
    keeps its old parameters; a relevance file of another build is refused;
    apply_relevance refuses a file whose per-source image counts are not the
    pool's, or that misses a pool source;
  * increments --sources relevance (the default) is the draw as it was: its
    digest on a hand-written base (hand_base, no k-means) equals the one
    recorded with select.py before --sources existed, and an explicit
    --sources relevance gives the same bytes and parameters;
  * increments --sources evidence, on a variant world ('csgo') whose
    off-domain source has only OtherPlant boxes and so no verified species
    box (admit_summary.json per_slug counted as verify admit counts it): no
    increment, the OtherPlant-heavy one included, holds its images (the
    default draw does); the summary and the draw's parameters carry the
    evidenced sources with their verified boxes and pool images, the excluded
    source and images, the admit summary's sha256 and the cross-check;
    --min-evidence equal to harvC's count keeps it evidenced and drawn from
    (one increment the size of the evidenced pool holds all of its images);
    --min-evidence above harvC's count excludes it too and counts the
    near-dup groups the exclusion splits; a draw larger than the evidenced
    pool refuses; admit_summary.json and select_summary.json disagreeing
    (verified boxes above species crops either way, a source without a
    per_slug entry, another verified.jsonl or crops.csv named) is refused, as
    are an admit summary without per_slug counts, --min-evidence 0 and the
    flags misused; without select's source-evidence record the cross-check is
    recorded as not made; the CLI.

No open_clip, no network, no image file is opened.

Run:  python3 tests/test_inc_select.py
"""
import collections
import csv
import json
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_select_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import select as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn):
    try:
        fn()
    except S.SelectError as e:
        return str(e)
    return None


# ------------------------------------------------------------- the world
D = 32
SCENE_SIZES = [200, 150, 100, 80, 60, 40, 20, 10]    # on-domain scenes 0..7
OFF = 8                                               # the off-domain scene
OFF_SIZE = 100
N_SMALL = 6
CORE_BOXES = [10] + [200] * 11          # Waterhemp is rare in train_core
CORE_NOISE = 0.75
N_PAIRS = 10
N_NEVER = 40


def _hash(rng):
    return int.from_bytes(rng.bytes(8), "big")


def _bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


def make_world(core_manifest, defect=None):
    """Step 1 inputs in inc/verify.py's formats under verify's STEP1, the
    train_core manifest at core_manifest and the never-train index. Returns
    ground truth for the checks. The world is the same on every call; defect
    plants one inconsistency."""
    rng = np.random.default_rng(1234)
    basis = np.linalg.qr(rng.normal(size=(D, D)))[0]
    scene_dir, class_dir = basis[:9], basis[9:9 + C.NC]
    root = pathlib.Path(V.STEP1)
    assert str(root).startswith(str(TMP)), root    # never the machine's real step1
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    lab_dir = TMP / "repo" / "labels"
    crops, emb, verdict = [], [], []           # crops: (set, key, box, label, source)
    meta = []

    def label_row(key, source, boxes, img_dir):
        lp = lab_dir / (key + ".txt")
        C.write_yolo(lp, [(c, 0.5, 0.5, 0.1, 0.1) for c in boxes])
        return {"image": str(img_dir / (key + ".jpg")), "label": str(lp),
                "sha256": C.sha256_text(key), "label_sha256": C.sha256_file(lp),
                "source": source, "session": "", "key": key}

    # train_core: one species per image, five boxes per image
    core, n = [], 0
    for k, nb in enumerate(CORE_BOXES):
        for _ in range(nb // 5):
            key = "core_%04d" % n
            n += 1
            core.append(label_row(key, "cottonweeddet12/train", [k] * 5,
                                  TMP / "repo" / "downloads" / "cwd12" / "train" / "images"))
            for b in range(5):
                z = rng.normal(size=D)
                z -= scene_dir.T @ (scene_dir @ z)          # noise off the scene directions
                crops.append(("core", key, b, k, "train_core"))
                emb.append(class_dir[k] + CORE_NOISE * z)
    C.write_manifest(core_manifest, core)
    for r in range(4):                         # cwd12 copies: in crops.csv, never used here
        crops.append(("copy", "copy_%02d" % r, 0, r, "copyslug"))
        emb.append(rng.normal(size=D))

    # pool: scene-clustered images with 1-3 boxes
    scenes = np.r_[np.repeat(np.arange(8), SCENE_SIZES), np.full(OFF_SIZE, OFF)]
    rng.shuffle(scenes)                        # key order says nothing of scene
    imgs = []                                  # dicts: key, source, boxes, scene, embs, h
    for i, sc in enumerate(scenes):
        nb = 1 + int(rng.integers(0, 3))
        if i % 15 == 0:
            boxes = [C.OTHER_PLANT] * nb       # OtherPlant-only images
        elif sc == OFF:
            boxes = [C.OTHER_PLANT if x < 0.2 else int(rng.integers(1, 12)) for x in rng.random(nb)]
            if defect == "csgo":               # a non-plant source: every box OtherPlant (same rng draws)
                boxes = [C.OTHER_PLANT] * len(boxes)
        else:
            u = rng.random(nb)
            boxes = [0 if x < 0.35 else C.OTHER_PLANT if x < 0.55
                     else int(rng.integers(1, 12)) for x in u]
        src = "harvOff/train" if sc == OFF else ("harvA/train" if i % 3 else "harvB/valid")
        embedded = boxes[:1] if i == 5 else boxes          # box 2+ of image 5: too small
        embs = [4.0 * scene_dir[sc] + (-1.0 if sc == OFF and c < 12 else 1.0) * class_dir[c]
                + 0.25 * rng.normal(size=D) for c in embedded]
        imgs.append({"key": "pool_%04d" % i, "source": src, "boxes": boxes, "scene": int(sc),
                     "embs": embs, "h": _hash(rng)})
    # planted near-copies: a fork under another slug, a few bits away
    origs = [i for i in range(len(scenes)) if scenes[i] < OFF and i % 7 == 3 and i != 5]
    groups = []
    for t in range(N_PAIRS):
        o = imgs[origs[t]]
        flips = rng.choice(64, size=1 + t % 3, replace=False)
        h = o["h"]
        for f in flips:
            h ^= 1 << int(f)
        imgs.append({"key": "pool_c%02d" % t, "source": "harvC/train", "boxes": list(o["boxes"]),
                     "scene": o["scene"], "h": h,
                     "embs": [e + 0.01 * rng.normal(size=D) for e in o["embs"]]})
        groups.append({o["key"], "pool_c%02d" % t})
    o = imgs[origs[N_PAIRS]]                   # a chain: X -3- Y -3- Z, X and Z 6 bits apart
    flips = rng.choice(64, size=6, replace=False)
    hy, hz = o["h"], o["h"]
    for f in flips[:3]:
        hy ^= 1 << int(f)
    hz = hy
    for f in flips[3:]:
        hz ^= 1 << int(f)
    for t, h in ((N_PAIRS, hy), (N_PAIRS + 1, hz)):
        imgs.append({"key": "pool_c%02d" % t, "source": "harvC/train", "boxes": list(o["boxes"]),
                     "scene": o["scene"], "h": h,
                     "embs": [e + 0.01 * rng.normal(size=D) for e in o["embs"]]})
    groups.append({o["key"], "pool_c%02d" % N_PAIRS, "pool_c%02d" % (N_PAIRS + 1)})
    for e in range(N_SMALL):                   # admitted, every box an OtherPlant too small to crop
        imgs.append({"key": "pool_s%02d" % e, "source": "harvB/valid", "boxes": [C.OTHER_PLANT],
                     "scene": -1, "embs": [], "h": _hash(rng)})

    pool, truth = [], {"scene": {}, "boxes": {}, "groups": groups}
    for im in imgs:
        key = im["key"]
        idir = TMP / "repo" / "datasets" / im["source"].split("/")[0] / "images"
        pool.append(label_row(key, im["source"], im["boxes"], idir))
        truth["scene"][key] = im["scene"]
        truth["boxes"][key] = im["boxes"]
        for b, e in enumerate(im["embs"]):
            c = im["boxes"][b]
            crops.append(("pool", key, b, c, im["source"]))
            emb.append(e)
            verdict.append("other_ok" if c == C.OTHER_PLANT else "verified")
        meta.append({"key": key, "source": im["source"], "W": 640, "H": 480, "dhash": im["h"],
                     "boxes": [[c, 0.5, 0.5, 0.1, 0.1] for c in im["boxes"]], "src": []})
    for r in range(20):                        # pool crops of images that were not admitted
        c = int(r % 13)
        crops.append(("pool", "rejected_%02d" % r, 0, c, "harvR/train"))
        emb.append(rng.normal(size=D))
        verdict.append("other_ok" if c == C.OTHER_PLANT else "conflict")
        meta.append({"key": "rejected_%02d" % r, "source": "harvR/train", "W": 640, "H": 480,
                     "dhash": _hash(rng), "boxes": [[c, 0.5, 0.5, 0.1, 0.1]], "src": []})
    for r in range(10):                        # unknown species boxes of harvA: source evidence
        c = 1 + r % 11
        crops.append(("pool", "unknown_%02d" % r, 0, c, "harvA/train"))
        emb.append(4.0 * scene_dir[0] + 0.3 * class_dir[c] + 0.25 * rng.normal(size=D))
        verdict.append("unknown")
        meta.append({"key": "unknown_%02d" % r, "source": "harvA/train", "W": 640, "H": 480,
                     "dhash": _hash(rng), "boxes": [[c, 0.5, 0.5, 0.1, 0.1]], "src": []})
    never = [_hash(rng) for _ in range(N_NEVER)]
    hashes = [m["dhash"] for m in meta]
    assert len(set(hashes)) == len(hashes)
    assert all(_bits(h, x) > 6 for h in hashes for x in never)
    if defect == "admitted_nan":
        emb[[k for _s, k, _b, _c, _r in crops].index("pool_0003")] = np.full(D, np.nan)
    if defect == "never_train_hit":
        meta[7]["dhash"] = never[0] ^ 0b101
    if defect == "no_dhash":
        meta = meta[:3] + meta[4:]
    emb[0] = np.full(D, np.nan)                # one train_core crop failed to embed
    C.write_manifest(root / "verified.jsonl", pool)
    with open(V.POOL_META, "w") as fh:
        for m in sorted(meta, key=lambda m: m["key"]):
            fh.write(json.dumps(m, sort_keys=True) + "\n")
    C.NEVER_TRAIN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(C.NEVER_TRAIN_INDEX, "w") as fh:
        json.dump({"entries": [[h, "dev" if t % 2 else "test", "/x/eval_%02d.jpg" % t]
                               for t, h in enumerate(never)],
                   "min_expected": N_NEVER, "complete": True}, fh)
    with open(V.CROPS, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        for cid, (st, key, b, c, src) in enumerate(crops):
            wr.writerow([cid, st, key, "/x/%s.jpg" % key, src, src if st != "core" else "sess",
                         b, "0.500000", "0.500000", "0.100000", "0.100000", 640, 480, c,
                         C.CLASS_NAMES[c]])
    crops_sha = C.sha256_file(V.CROPS)
    X = np.asarray(emb, dtype=np.float16)
    ids = np.arange(len(crops))
    perm = rng.permutation(len(ids))           # shards hold crops in any order
    pathlib.Path(V.EMB_DIR).mkdir(parents=True)
    for s, part in enumerate(np.array_split(perm, 3)):
        meta_s = {"crops_sha256": "0" * 64 if (defect == "stale_shard" and s == 1) else crops_sha,
                  "embedder": "fake", "dim": D, "crops": int(len(part)), "stats": {}}
        np.savez(pathlib.Path(V.EMB_DIR) / ("emb_s%03d_of_003.npz" % s),
                 meta=np.array(json.dumps(meta_s)), crop_ids=ids[part], X=X[part])
    pool_ids = np.array([i for i, cr in enumerate(crops) if cr[0] == "pool"])
    np.savez(V.POOL_VERDICTS, meta=np.array(json.dumps({"crops_sha256": crops_sha,
                                                        "verdict_codes": list(V.VERDICT_CODES)})),
             crop_id=pool_ids, verdict=np.array([V.VERDICT_CODES.index(v) for v in verdict],
                                                dtype=np.int8))
    vsha = C.sha256_file(root / "verified.jsonl")
    if defect == "verified_changed":
        C.write_manifest(root / "verified.jsonl", pool[1:])
    embeddings = {"nshards": 3, "embedder": "fake", "dim": D, "failed_crops": 0,
                  "nan_rows": int((~np.isfinite(X.astype(np.float32)).all(1)).sum())}
    if defect == "admit_embedder":
        embeddings["embedder"] = "another-model"
    if defect == "admit_nshards":
        embeddings["nshards"] = 2
    # per_slug as verify admit counts it: box verdicts of every pool crop, the
    # boxes too small to crop as 'small'; admitted images are verified.jsonl's
    per_slug = {}

    def slug(s):
        return per_slug.setdefault(s, {"images": collections.Counter(), "boxes": collections.Counter()})
    pv = iter(verdict)
    img_v, img_src = {}, {}
    for st, key, _b, _c, src in crops:
        if st == "pool":
            v = next(pv)
            slug(src)["boxes"][v] += 1
            img_src[key] = src
            img_v[key] = "conflict" if v == "conflict" or img_v.get(key) == "conflict" else "unknown"
    for im in imgs:
        slug(im["source"])["boxes"]["small"] += len(im["boxes"]) - len(im["embs"])
    admitted = {r["key"] for r in pool}
    for key, v in img_v.items():
        if key not in admitted:                # the images verify did not admit
            slug(img_src[key])["images"][v] += 1
    for r in pool:
        slug(r["source"])["images"]["admitted"] += 1
    with open(V.ADMIT_SUMMARY, "w") as fh:
        json.dump({"verified_sha256": vsha, "crops_sha256": crops_sha,
                   "embeddings": embeddings,
                   "per_slug": {s: {"images": V._counter(d["images"]), "boxes": V._counter(+d["boxes"])}
                                for s, d in sorted(per_slug.items())}}, fh)
    truth["core_rows"] = core
    truth["pool_rows"] = pool
    truth["X16"] = X
    truth["crops"] = crops
    truth["verdict"] = verdict
    truth["never"] = never
    truth["hash"] = {m["key"]: m["dhash"] for m in meta}
    return truth


def keys(path):
    return [r["key"] for r in C.read_manifest(path)]


def summary_boxes(sm, part):
    return [sm["boxes"][part][n] for n in C.CLASS_NAMES]


def expected_scores(W, verified):
    """The retrieval scores, computed here from the world directly: kind-1
    images the mean train_core percentile of their species boxes, kind-2
    images their source's median; plus train_core's own image scores."""
    X = W["X16"].astype(np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    fin = np.isfinite(X).all(1)
    st = np.array([c[0] for c in W["crops"]])
    ck = [c[1] for c in W["crops"]]
    lab = np.array([c[3] for c in W["crops"]])
    src = [c[4] for c in W["crops"]]
    core = (st == "core") & fin
    P, ref, core_u = {}, {}, {}
    for k in range(12):
        idx = np.flatnonzero(core & (lab == k))
        Sk = X[idx].sum(0)
        P[k] = Sk / np.linalg.norm(Sk)
        loo = [X[i] @ ((Sk - X[i]) / np.linalg.norm(Sk - X[i])) for i in idx]
        ref[k] = np.sort(loo)
        for i, v in zip(idx, loo):
            core_u[i] = np.searchsorted(ref[k], v, side="right") / len(ref[k])

    def pct(i):
        c = float(X[i] @ P[lab[i]])
        return c, np.searchsorted(ref[lab[i]], c, side="right") / len(ref[lab[i]])

    per_img, per_src = {}, {}
    pv = iter(W["verdict"])
    for i in range(len(ck)):
        if st[i] != "pool":
            continue
        v = next(pv)
        if lab[i] < 12 and fin[i] and ck[i] in verified:
            per_img.setdefault(ck[i], []).append(pct(i))
        if lab[i] < 12 and fin[i] and v not in ("conflict", "failed"):
            per_src.setdefault(src[i], []).append(pct(i)[1])
    med = {s: float(np.median(v)) for s, v in per_src.items()}
    core_img = {}
    for i, u in core_u.items():
        core_img.setdefault(ck[i], []).append(u)
    return ({k: (np.mean([c for c, _u in v]), np.mean([u for _c, u in v]))
             for k, v in per_img.items()}, med, [np.mean(v) for v in core_img.values()])


def group_units(cl, keyset):
    """{dup_group: [keys]} over keyset, from a select_clusters.csv."""
    out = {}
    for k in sorted(keyset):
        out.setdefault(cl[k]["dup_group"], []).append(k)
    return out


def hand_base(W, out):
    """The increment side of a select build, written by hand (no k-means):
    every verified image in the increment pool, level-1 / level-2 clusters from
    a hash of the key (no feature for the pool_s images), one near-dup group per
    image except the planted ones. `increments` reads only these files,
    pool_meta.jsonl, the never-train index and the label files, so a draw here
    depends on nothing but numpy's seeded Generator: its digest can be pinned
    across machines (sklearn versions do not enter)."""
    out = pathlib.Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rows = sorted(W["pool_rows"], key=lambda r: r["key"])
    rep = {k: min(g) for g in W["groups"] for k in g}
    gids = {}
    crow = []
    for r in rows:
        k = r["key"]
        g = gids.setdefault(rep.get(k, k), len(gids))
        b = W["boxes"][k]
        feat = not k.startswith("pool_s")
        crow.append([k, g, C.stable_int("hand/l1/" + k) % 6 if feat else -1,
                     C.stable_int("hand/l2/" + k) % 3 if feat else -1, "", "", "", "",
                     sum(1 for c in b if c < C.OTHER_PLANT), sum(1 for c in b if c == C.OTHER_PLANT), -1, "pool"])
    psha = C.write_manifest(out / S.POOL, rows)
    with open(out / S.CLUSTERS, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(S.CLUSTER_COLS)
        wr.writerows(crow)
    sm = {"outputs": {S.POOL: {"sha256": psha, "rows": len(rows), "path": str(out / S.POOL)},
                      S.CLUSTERS: {"sha256": C.sha256_file(out / S.CLUSTERS), "rows": len(rows),
                                   "path": str(out / S.CLUSTERS)}},
          "sizes": {"base_B": 600},
          "inputs": {"pool_meta": {"path": str(V.POOL_META), "sha256": C.sha256_file(V.POOL_META)}}}
    (out / S.SUMMARY).write_text(json.dumps(sm, indent=1, sort_keys=True))
    return out


def draw_digest(man_dir, summary):
    """sha256 of a draw with every path and file hash left out (the rows hold
    this run's temporary paths): the drawn keys of each manifest, the
    parameters without their sha256 fields, the parameter names, and the
    summary's keys, counts and per-increment table."""
    import hashlib
    names = sorted(summary["outputs"])
    d = {"keys": {n: [r["key"] for r in C.read_manifest(pathlib.Path(man_dir) / n)] for n in names},
         "params": {k: v for k, v in summary["params"].items() if not k.endswith("sha256")},
         "param_names": sorted(summary["params"]), "summary_keys": sorted(summary),
         "counts": [summary[k] for k in ("pool_images", "drawn", "left_in_pool", "pool_level1_clusters",
                                         "groups_left_out_for_size")],
         "relevance": summary["relevance"], "increments": summary["increments"]}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


# The default draw's digest (draw_digest) on hand_base, recorded with
# inc/select.py as it was before --sources existed (2026-09-27; the same value
# under Python 3.10 with numpy 1.26.4 and 2.2.6, and Python 3.12 with numpy 2.4.4).
DEFAULT_DRAW_DIGEST = "8aa86cdd81400cbc5ac00651e87e3191547c65738d861d155f07fce45956f324"


def test_default_draw(W):
    """--sources relevance (the default) is the draw as it was, byte for byte."""
    print("the default draw is unchanged (--sources relevance)")
    hb = hand_base(W, TMP / "hand")
    sd = S.increments("hand_e", 3, 40, other_heavy=True, base_dir=hb)
    md = TMP / "inc" / "hand_e" / "manifests"
    got = draw_digest(md, sd)
    check("default draw: its digest (drawn keys, parameters, summary) is the one recorded before --sources existed",
          got == DEFAULT_DRAW_DIGEST, got)
    first = {n: (md / n).read_bytes() for n in sd["outputs"]}
    sx = S.increments("hand_e", 3, 40, other_heavy=True, base_dir=hb, sources="relevance", force=True)
    check("--sources relevance given explicitly: the same manifest bytes, parameters and digest",
          {n: (md / n).read_bytes() for n in sx["outputs"]} == first and sx["params"] == sd["params"]
          and draw_digest(md, sx) == DEFAULT_DRAW_DIGEST
          and not {"increment_sources", "evidence"} & set(sx) and "sources" not in sx["params"])


def test_evidence(core_path):
    """--sources evidence: source-level species evidence from admit_summary.json."""
    print("increments with the source-evidence criterion (--sources evidence)")
    make_world(core_path, "csgo")         # harvOff: a non-plant source, every box OtherPlant
    bev = TMP / "bev"
    sb = S.build(0.6, 0, 100, out_dir=bev, other_only_frac=1.0)
    admit = json.loads(pathlib.Path(V.ADMIT_SUMMARY).read_text())
    ver = {s: d["boxes"].get("verified", 0) for s, d in admit["per_slug"].items()}
    pool = sorted(C.read_manifest(bev / S.POOL), key=lambda r: r["key"])
    n_src = collections.Counter(r["source"] for r in pool)
    check("the world: harvOff holds no verified species box and all 100 of its images are in the increment pool "
          "(no species evidence, never selected); harvA, harvB and harvC hold verified boxes",
          ver["harvOff/train"] == 0 and n_src["harvOff/train"] == 100
          and "harvOff/train" not in sb["retrieval"]["source_evidence"]
          and all(ver[s] > 0 for s in ("harvA/train", "harvB/valid", "harvC/train")), (ver, dict(n_src)))

    # the default draw (no relevance file) takes harvOff's images; the evidence draw takes none
    sd = S.increments("ev_def", 2, 40, other_heavy=True, base_dir=bev)
    names = ("inc_01.jsonl", "inc_02.jsonl", S.INC_OTHER)
    d_def = [r for n in names for r in C.read_manifest(TMP / "inc" / "ev_def" / "manifests" / n)]
    check("the default draw holds images of the non-plant source (the filter below is not vacuous)",
          any(r["source"] == "harvOff/train" for r in d_def) and sd["relevance"] is None)
    se = S.increments("ev_e", 2, 40, other_heavy=True, base_dir=bev, sources="evidence")
    me = TMP / "inc" / "ev_e" / "manifests"
    d_ev = {n: C.read_manifest(me / n) for n in names}
    check("--sources evidence: every increment, the OtherPlant-heavy one too, holds only images of sources with "
          "a verified species box",
          [len(d_ev[n]) for n in names] == [40, 40, 40]
          and all(ver[r["source"]] >= 1 for rows in d_ev.values() for r in rows))
    ei = se["evidence"]
    asha = C.sha256_file(V.ADMIT_SUMMARY)
    check("the summary records the evidenced sources (verified boxes, pool images), the excluded source and "
          "images, the admit summary's sha256 (also a draw parameter) and the cross-check",
          ei["evidenced_sources"] == {s: {"verified_boxes": n, "pool_images": n_src.get(s, 0)}
                                      for s, n in sorted(ver.items()) if n >= 1}
          and ei["excluded_sources"] == {"harvOff/train": {"images": 100, "verified_boxes": 0}}
          and ei["excluded_images"] == 100 and ei["pool_images_before"] == len(pool)
          and se["pool_images"] == len(pool) - 100 and ei["min_evidence"] == 1
          and ei["admit_summary"]["sha256"] == asha == se["params"]["admit_summary_sha256"]
          and se["params"]["sources"] == "evidence" and se["params"]["min_evidence"] == 1
          and se["increment_sources"] == "evidence" and se["relevance"] is None
          and ei["cross_check"]["checked"] is True and ei["rule"] == S.EVIDENCE_RULE, ei)
    check("an evidence draw over the default draw of the same experiment refuses without --force",
          raises(lambda: S.increments("ev_def", 2, 40, other_heavy=True, base_dir=bev,
                                      sources="evidence")) is not None)

    # --min-evidence above harvC's count: its forks leave, near-dup groups with their originals are split
    k = ver["harvC/train"] + 1
    groups = S.read_clusters(bev / S.CLUSTERS)
    by_g = collections.defaultdict(set)
    for r in pool:
        by_g[groups[r["key"]]["dup_group"]].add(ver[r["source"]] >= k)
    n_split = sum(1 for v in by_g.values() if v == {True, False})
    sk = S.increments("ev_k", 2, 40, base_dir=bev, sources="evidence", min_evidence=k)
    d_k = [r for n in ("inc_01.jsonl", "inc_02.jsonl") for r in C.read_manifest(TMP / "inc" / "ev_k" / "manifests" / n)]
    check("--min-evidence %d: harvC (%d verified boxes) is excluded as well; the near-dup groups the exclusion "
          "splits are counted" % (k, ver["harvC/train"]),
          n_src["harvC/train"] > 0 and not [r for r in d_k if r["source"] in ("harvC/train", "harvOff/train")]
          and set(sk["evidence"]["excluded_sources"]) == {"harvC/train", "harvOff/train"}
          and sk["params"]["min_evidence"] == k and sk["evidence"]["near_dup_groups_split"] == n_split > 0,
          (sk["evidence"]["excluded_sources"], sk["evidence"]["near_dup_groups_split"], n_split))
    # --min-evidence equal to harvC's count: "at least", so harvC stays evidenced and is drawn from. One increment
    # the size of the whole evidenced pool must then take every one of its images, harvC's included.
    k0 = ver["harvC/train"]
    cap0 = S.pool_capacity([r for r in pool if ver[r["source"]] >= k0], groups)
    try:
        s0 = S.increments("ev_k0", 1, cap0["images"], base_dir=bev, sources="evidence", min_evidence=k0)
        d_0 = C.read_manifest(TMP / "inc" / "ev_k0" / "manifests" / "inc_01.jsonl")
    except S.SelectError as e:
        s0, d_0 = {"evidence": {"evidenced_sources": {}, "excluded_sources": {}}, "error": str(e)}, []
    check("--min-evidence %d, exactly harvC's verified boxes: harvC stays evidenced and its %d increment-pool images "
          "are drawn (only harvOff is excluded)" % (k0, n_src["harvC/train"]),
          s0["evidence"]["evidenced_sources"].get("harvC/train") == {"verified_boxes": k0,
                                                                     "pool_images": n_src["harvC/train"]}
          and set(s0["evidence"]["excluded_sources"]) == {"harvOff/train"}
          and s0.get("pool_images") == len(pool) - 100 == len(d_0)
          and sum(r["source"] == "harvC/train" for r in d_0) == n_src["harvC/train"] > 0,
          (s0.get("error"), s0["evidence"]["evidenced_sources"], s0["evidence"]["excluded_sources"], len(d_0)))
    cap = S.pool_capacity([r for r in pool if ver[r["source"]] >= 1], groups)
    err = raises(lambda: S.increments("ev_cap", 1, cap["images"] + 1, base_dir=bev, sources="evidence"))
    check("a draw larger than the evidenced pool refuses, naming the filter, and writes nothing",
          err is not None and "after the source-evidence filter" in err
          and not (TMP / "inc" / "ev_cap").exists(), err)
    check("pool_capacity: every image, and the OtherPlant-heavy groups' images (by the draw's rule)",
          cap["images"] == len(pool) - 100 and 40 <= cap["other_heavy_images"] < cap["images"], cap)

    # refusals: the two files disagree, or the evidence is not usable
    saved = pathlib.Path(V.ADMIT_SUMMARY).read_bytes()
    sel_saved = (bev / S.SUMMARY).read_bytes()

    def with_admit(edit, **kw):
        a = json.loads(saved)
        edit(a)
        pathlib.Path(V.ADMIT_SUMMARY).write_text(json.dumps(a))
        try:
            return raises(lambda: S.increments("ev_bad", 1, 5, base_dir=bev, sources="evidence", **kw))
        finally:
            pathlib.Path(V.ADMIT_SUMMARY).write_bytes(saved)
    sp_a = sb["retrieval"]["source_evidence"]["harvA/train"]["species_crops"]
    e1 = with_admit(lambda a: a["per_slug"]["harvA/train"]["boxes"].update(verified=sp_a + 1))
    e2 = with_admit(lambda a: a["per_slug"].pop("harvB/valid"))
    e3 = with_admit(lambda a: a.update(verified_sha256="0" * 64))
    e4 = with_admit(lambda a: a.update(crops_sha256="0" * 64))
    check("refused when admit_summary.json and select_summary.json disagree: more verified boxes than species "
          "crops; a source select counts without a per_slug entry; another verified.jsonl or crops.csv named",
          all(e is not None and "disagree" in e for e in (e1, e2, e3, e4))
          and "harvA/train" in e1 and "harvB/valid" in e2 and "verified.jsonl" in e3 and "crops.csv" in e4,
          (e1, e2, e3, e4))
    sm = json.loads(sel_saved)
    sm["retrieval"]["source_evidence"]["harvB/valid"]["species_crops"] = ver["harvB/valid"] - 1
    (bev / S.SUMMARY).write_text(json.dumps(sm))
    e5 = raises(lambda: S.increments("ev_bad", 1, 5, base_dir=bev, sources="evidence"))
    (bev / S.SUMMARY).write_bytes(sel_saved)
    check("refused when select_summary.json's source evidence counts fewer species crops than verified boxes",
          e5 is not None and "disagree" in e5 and "harvB/valid" in e5, e5)
    e6 = with_admit(lambda a: a.pop("per_slug"))
    e7 = with_admit(lambda a: a["per_slug"]["harvA/train"].pop("boxes"))
    e8 = with_admit(lambda a: None, min_evidence=0)
    e9 = raises(lambda: S.increments("ev_bad", 1, 5, base_dir=bev, sources="evidence",
                                     relevance=TMP / "none.json"))
    e10 = raises(lambda: S.increments("ev_bad", 1, 5, base_dir=bev, min_evidence=1))
    e11 = raises(lambda: S.increments("ev_bad", 1, 5, base_dir=bev, sources="stars"))
    check("refused: no per_slug counts, a per_slug entry without box counts, --min-evidence 0, --relevance with "
          "--sources evidence, --min-evidence without it, an unknown --sources; nothing written",
          e6 and "per_slug" in e6 and e7 and "box verdict counts" in e7 and e8 and "min-evidence" in e8
          and e9 and "does not take --relevance" in e9 and e10 and "apply to --sources evidence" in e10
          and e11 and "--sources" in e11 and not (TMP / "inc" / "ev_bad").exists(),
          (e6, e7, e8, e9, e10, e11))
    pool_rows = [dict(r, source="harvNew/train") if i == 0 else r for i, r in enumerate(pool)]
    ld = S.load_evidence(json.loads(sel_saved))
    e12 = raises(lambda: S.apply_evidence(pool_rows, ld, groups))
    check("apply_evidence refuses a pool source without a per_slug entry",
          e12 is not None and "no per_slug entry" in e12 and "harvNew/train" in e12, e12)
    ln = S.load_evidence(dict(json.loads(sel_saved), retrieval={}))
    check("where select_summary.json records no source evidence, the cross-check is recorded as not made",
          ln["cross_check"]["checked"] is False and ln["evidenced"] == ld["evidenced"])

    rc = S.main(["increments", "--exp", "cli_ev", "--n", "2", "--size", "5", "--base-dir", str(bev),
                 "--sources", "evidence", "--min-evidence", "1", "--admit-summary", str(V.ADMIT_SUMMARY)])
    rc_bad = S.main(["increments", "--exp", "cli_ev2", "--n", "2", "--size", "5", "--base-dir", str(bev),
                     "--min-evidence", "1"])
    check("CLI increments --sources evidence (exit 0); --min-evidence without it exits 2",
          rc == 0 and json.loads((TMP / "inc" / "cli_ev" / "manifests" / S.INC_SUMMARY).read_text())
          ["evidence"]["excluded_sources"] == {"harvOff/train": {"images": 100, "verified_boxes": 0}}
          and rc_bad == 2)


def main():
    core_path = C.manifest_path("train_core")
    W = make_world(core_path)
    C.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.LOCK_PATH, "w") as fh:
        json.dump({"manifests": {"train_core": C.sha256_file(core_path)}}, fh)
    verified = {r["key"] for r in W["pool_rows"]}
    n_ver = len(verified)
    counts = {k: np.bincount(b, minlength=C.NC) for k, b in W["boxes"].items()}
    other_only = {k for k in verified if counts[k][:12].sum() == 0}
    q = round(0.6 * n_ver)

    # --- 1. uncapped build without the OtherPlant-only budget ------------------
    print("uncapped build (cap 100x, no OtherPlant-only budget)")
    b1 = TMP / "b1"
    s1 = S.build(0.6, 0, 100, out_dir=b1, other_only_frac=1.0)
    sel = set(keys(b1 / S.BASE_SELECTED))
    rest = set(keys(b1 / S.POOL))
    check("selected and increment pool are disjoint", not sel & rest)
    check("selected + increment pool = the verified set", sel | rest == verified,
          "%d + %d vs %d" % (len(sel), len(rest), n_ver))
    check("quota = round(F * |verified|) images are selected", len(sel) == q == s1["sizes"]["quota"],
          "%d %d %s" % (len(sel), q, s1["sizes"]))
    baseB = C.read_manifest(b1 / S.BASE_B)
    by_key = {r["key"]: r for r in baseB}
    check("base_B = train_core + selected",
          sorted(by_key) == sorted([r["key"] for r in W["core_rows"]] + list(sel)))
    check("every train_core row is in base_B verbatim",
          all(by_key.get(r["key"]) == {k: r[k] for k in C.MANIFEST_KEYS} for r in W["core_rows"]))
    empty = {k for k in verified if k.startswith("pool_s")}
    check("images without a feature are never selected", empty <= rest)

    cl = S.read_clusters(b1 / S.CLUSTERS)
    check("clusters file covers every verified image", set(cl) == verified)
    check("no_feature status = exactly the images with no croppable box",
          {k for k in verified if cl[k]["status"] == "no_feature"} == empty)
    feat = [k for k in verified if cl[k]["status"] != "no_feature"]
    by_l1 = {}
    for k in feat:
        by_l1.setdefault(int(cl[k]["l1"]), set()).add(W["scene"][k])
    check("k-means level 1 recovers the 9 scenes (k1 = round(sqrt(772 / 10)) = 9)",
          len(by_l1) == 9 and all(len(v) == 1 for v in by_l1.values())
          and sorted(next(iter(v)) for v in by_l1.values()) == list(range(9)), by_l1)

    # retrieval: the train_core percentile scale, computed independently
    exp1, med, core_scores = expected_scores(W, verified)
    err1 = max(abs(float(cl[k]["score"]) - u) for k, (_c, u) in exp1.items())
    errc = max(abs(float(cl[k]["cosine"]) - c) for k, (c, _u) in exp1.items())
    check("species-scored images: score = mean train_core percentile of their boxes "
          "(leave-one-out reference), cosine = mean raw cosine (CSV: 6 decimals)",
          err1 <= 5e-7 and errc <= 5e-7 and all(cl[k]["score_kind"] == "cwd12" for k in exp1),
          (err1, errc))
    src_of = {r["key"]: r["source"] for r in W["pool_rows"]}
    oo_feat = [k for k in feat if k not in exp1]
    err2 = max(abs(float(cl[k]["score"]) - med[src_of[k]]) for k in oo_feat)
    check("OtherPlant-only images take their source's median (unknown boxes count, "
          "conflicts do not)",
          oo_feat and err2 <= 5e-7 and all(cl[k]["score_kind"] == "other" for k in oo_feat),
          (len(oo_feat), err2))
    check("the source evidence is recorded per source",
          {s: round(v, 6) for s, v in med.items()}
          == {s: v["median"] for s, v in s1["retrieval"]["source_evidence"].items()},
          s1["retrieval"]["source_evidence"])
    check("OtherPlant-only images have a typicality (cross-fitted prototypes)",
          all(cl[k]["typicality"] != "" for k in oo_feat)
          and s1["retrieval"]["other_typicality"]["prototypes"][0] > 0
          and s1["retrieval"]["other_typicality"]["prototypes"][1] > 0,
          s1["retrieval"]["other_typicality"])
    tc = np.array(core_scores)
    check("the summary records the share of train_core's own images that pass the gate",
          abs(s1["retrieval"]["train_core_images_passing_gate"] - float((tc >= 0.1).mean())) < 1e-6
          and s1["retrieval"]["gate"] == 0.1, s1["retrieval"]["train_core_images_passing_gate"])
    on = [float(cl[k]["score"]) for k in feat if 0 <= W["scene"][k] < OFF]
    off = [float(cl[k]["score"]) for k in feat if W["scene"][k] == OFF]
    check("on-domain images sit mid-scale, the off-domain scene at the bottom",
          min(on) > 0.15 and max(off) < 0.1, (min(on), max(off)))
    sp_rate = np.mean([k in sel for k in feat if 0 <= W["scene"][k] < OFF and k not in other_only])
    oo_rate = np.mean([k in sel for k in feat if 0 <= W["scene"][k] < OFF and k in other_only])
    check("OtherPlant-only images are not preferred over species images (one scale)",
          oo_rate <= sp_rate + 0.1, (oo_rate, sp_rate))

    # the gate
    offk = [k for k in feat if W["scene"][k] == OFF]
    check("the gate: every off-domain image is below it and none is selected",
          all(cl[k]["status"] == "below_gate" for k in offk) and not set(offk) & sel,
          sum(k in sel for k in offk))
    check("the gate: nothing on-domain is below it",
          not [k for k in feat if W["scene"][k] != OFF and cl[k]["status"] == "below_gate"])
    same = [k for k in feat if W["scene"][k] == 2]          # an on-domain scene of 100 images
    check("the off-domain scene is under-sampled against an on-domain scene of the same size",
          len(offk) == 100 <= len(same) and sum(k in sel for k in same) >= 60,
          (len(offk), len(same), sum(k in sel for k in same)))
    check("the summary counts the gated images per source",
          s1["sizes"]["below_gate"] == 100 and s1["sources"]["below_gate"] == {"harvOff/train": 100},
          (s1["sizes"]["below_gate"], s1["sources"]["below_gate"]))

    # coverage: water-filling over units (near-dup groups) of the on-domain scenes
    tot_u, sel_u = {}, {}
    for s in range(8):
        ks = [k for k in feat if W["scene"][k] == s]
        tot_u[s] = len(group_units(cl, ks))
        sel_u[s] = len(group_units(cl, [k for k in ks if k in sel]))
    full = [s for s in range(8) if sel_u[s] == tot_u[s]]
    part = [sel_u[s] for s in range(8) if s not in full]
    check("small scenes are taken whole, the large ones share the rest equally",
          full and part and max(part) - min(part) <= 1
          and all(tot_u[s] <= min(part) + 1 for s in full)
          and sorted(tot_u[s] for s in full)[:4] == [10, 20, 40, 60], (tot_u, sel_u))
    sub = {}
    for k in feat:
        if cl[k]["status"] != "below_gate":
            sub.setdefault((cl[k]["l1"], cl[k]["l2"]), []).append((float(cl[k]["score"]), k in sel))
    check("level 2 splits clusters (up to 8 sub-clusters each)",
          len(sub) > 8 and max(sum(1 for (a, _b) in sub if a == c) for c in {a for a, _ in sub}) <= 8,
          len(sub))
    # a group scores as its lowest member; the planted groups are within one sub-cluster
    grp_score = {}
    for k in feat:
        g = cl[k]["dup_group"]
        grp_score[g] = min(grp_score.get(g, 9.0), float(cl[k]["score"]))
    sub_g = {}
    for k in feat:
        if cl[k]["status"] != "below_gate":
            sub_g.setdefault((cl[k]["l1"], cl[k]["l2"]), {})[cl[k]["dup_group"]] = \
                (grp_score[cl[k]["dup_group"]], k in sel)
    prefix_ok = all(min([s for s, t in v.values() if t], default=9)
                    >= max([s for s, t in v.values() if not t], default=-9) for v in sub_g.values())
    check("inside a sub-cluster, groups are taken by descending retrieval score", prefix_ok)
    check("image 5 keeps a feature from its one embedded box", cl["pool_0005"]["status"] != "no_feature")

    # near-dup groups
    planted = W["groups"]
    gid = {k: cl[k]["dup_group"] for k in verified}
    in_planted = set().union(*planted)
    check("every planted near-copy group is one dup_group, and no other image shares one",
          all(len({gid[k] for k in g}) == 1 for g in planted)
          and len({gid[k] for k in verified}) == n_ver - sum(len(g) - 1 for g in planted)
          and all(gid[k] not in {gid[x] for x in in_planted} for k in verified - in_planted))
    check("the chain X-Y-Z (ends 6 bits apart) is one group", len(planted[-1]) == 3
          and _bits(*[W["hash"][k] for k in sorted(planted[-1])][::2]) == 6)
    check("near-copies land on one side (base or pool)",
          all(len({k in sel for k in g}) == 1 for g in planted), [[k in sel for k in g] for g in planted])
    check("summary: near-dup groups",
          s1["near_dup"]["multi_image_groups"] == len(planted)
          and s1["near_dup"]["images_in_multi_groups"] == sum(len(g) for g in planted)
          and s1["near_dup"]["largest_group"] == 3, s1["near_dup"])
    check("summary: every verified image passed the never-train guard",
          s1["never_train"]["checked"] == n_ver and s1["never_train"]["evaluation_images"] == N_NEVER)

    tot = sum(counts[k] for k in verified)
    check("summary boxes: selected + pool = all verified boxes",
          [a + b for a, b in zip(summary_boxes(s1, "selected"), summary_boxes(s1, "increment_pool"))]
          == tot.tolist())
    check("summary boxes: train_core per species",
          summary_boxes(s1, "train_core") == CORE_BOXES + [0], summary_boxes(s1, "train_core"))
    check("summary sources add up",
          sum(s1["sources"]["selected"].values()) == len(sel)
          and sum(s1["sources"]["increment_pool"].values()) == len(rest)
          and set(s1["sources"]["selected"]) <= {"harvA/train", "harvB/valid", "harvC/train"})
    check("summary records 9 level-1 clusters and the parameters",
          s1["clusters"]["level1"] == 9 and s1["params"]["base_frac"] == 0.6
          and s1["params"]["seed"] == 0 and s1["params"]["cap_mult"] == 100
          and s1["params"]["kmeans_threads"] == 1 and "sklearn" in s1["params"]["versions"])
    check("crops of images that were not admitted, and cwd12 copies, are not image crops",
          s1["crops"]["admitted_image_crops"] == sum(1 for st, k, _b, _c, _s in W["crops"]
                                                     if st == "pool" and k in verified)
          and s1["crops"]["pool_crops"] - s1["crops"]["admitted_image_crops"] == 30, s1["crops"])
    check("a train_core crop that failed to embed (NaN) is left out of the scale",
          s1["crops"]["train_core_crops_failed"] == 1, s1["crops"])
    check("typicality folds use every other_ok pool crop, admitted or not",
          s1["crops"]["other_ok_crops_with_features"] == sum(
              1 for st, k, _b, c, _s in W["crops"] if st == "pool" and c == C.OTHER_PLANT),
          s1["crops"])

    # --- 2. determinism ---------------------------------------------------------
    print("determinism")
    b2 = TMP / "b2"
    S.build(0.6, 0, 100, out_dir=b2, other_only_frac=1.0)
    same = all((b1 / n).read_bytes() == (b2 / n).read_bytes()
               for n in (S.BASE_SELECTED, S.BASE_B, S.POOL, S.CLUSTERS))
    check("same seed -> byte-identical manifests and clusters", same)
    b3 = TMP / "b3"
    s3 = S.build(0.6, 1, 100, out_dir=b3, other_only_frac=1.0)
    sel3, rest3 = set(keys(b3 / S.BASE_SELECTED)), set(keys(b3 / S.POOL))
    check("another seed is still a partition of the same size",
          not sel3 & rest3 and sel3 | rest3 == verified and len(sel3) == q, s3["sizes"])
    rc = S.main(["build", "--base-frac", "0.6", "--seed", "0", "--cap-mult", "100",
                 "--other-only-frac", "1", "--out-dir", str(TMP / "b_cli")])
    check("CLI build (LOCK-checked default inputs) = the same bytes",
          rc == 0 and all((b1 / n).read_bytes() == (TMP / "b_cli" / n).read_bytes()
                          for n in (S.BASE_SELECTED, S.BASE_B, S.POOL)))

    # --- 3. the OtherPlant-only budget and the caps -------------------------------
    print("OtherPlant-only budget and caps (3x train_core)")

    def share(sel_keys):
        return sum(k in other_only for k in sel_keys) / max(1, len(sel_keys))

    b4 = TMP / "b4"
    s4 = S.build(0.6, 0, 3, out_dir=b4)
    sel4 = set(keys(b4 / S.BASE_SELECTED))
    cl4 = S.read_clusters(b4 / S.CLUSTERS)
    got = sum((counts[k] for k in sel4), np.zeros(C.NC, dtype=np.int64))
    check("every species within 3 x its train_core boxes",
          all(got[s] <= 3 * CORE_BOXES[s] for s in range(12)), got.tolist())
    check("only Waterhemp was over its cap", s4["boxes"]["capped_species"] == ["Waterhemp"],
          s4["boxes"]["capped_species"])
    dropped = [k for k in verified if cl4[k]["status"] == "dropped_cap"]
    check("drops happened and are counted", dropped and len(dropped) == s4["sizes"]["dropped_by_cap"])
    picked4 = [k for k in verified if cl4[k]["status"] in ("selected", "dropped_cap",
                                                            "dropped_other_budget")]
    check("the first pick is the quota (selected + dropped)", len(picked4) == q, (len(picked4), q))
    check("every cap-dropped image carries Waterhemp (its group does)",
          all(sum(counts[x][0] for x in verified if cl4[x]["dup_group"] == cl4[k]["dup_group"]) > 0
              for k in dropped))
    kept_wh = [int(cl4[k]["rank"]) for k in sel4 if counts[k][0] > 0]
    check("drops are last-in-first-out over the pick order",
          min(int(cl4[k]["rank"]) for k in dropped) > max(kept_wh, default=-1))
    first_drop = min(dropped, key=lambda k: int(cl4[k]["rank"]))
    fd_wh = sum(counts[x][0] for x in verified if cl4[x]["dup_group"] == cl4[first_drop]["dup_group"])
    check("drops are minimal (keeping the earliest dropped group would break the cap)",
          got[0] + fd_wh > 30, (got[0], fd_wh))
    check("partition holds under caps",
          sel4 | set(keys(b4 / S.POOL)) == verified and not sel4 & set(keys(b4 / S.POOL)))
    check("the OtherPlant-only budget holds under caps (<= 10 % of the selected images)",
          0 < share(sel4) <= 0.1 and s4["otherplant"]["only_images_share_of_selected"] == round(share(sel4), 6),
          (share(sel4), s4["otherplant"]))
    check("the budget trimmed after the cap drops, and passed some over in the pick",
          s4["sizes"]["dropped_other_budget"] > 0 and s4["sizes"]["passed_over_other_budget"] > 0,
          s4["sizes"])
    b5 = TMP / "b5"
    s5 = S.build(0.6, 0, 3, refill=True, out_dir=b5)
    sel5 = set(keys(b5 / S.BASE_SELECTED))
    cl5 = S.read_clusters(b5 / S.CLUSTERS)
    got5 = sum((counts[k] for k in sel5), np.zeros(C.NC, dtype=np.int64))
    refilled = [k for k in sel5 if cl5[k]["status"] == "refilled"]
    check("refill: caps and budget still hold, quota not exceeded, more selected than without",
          all(got5[s] <= 3 * CORE_BOXES[s] for s in range(12)) and len(sel5) <= q
          and refilled and len(sel5) > len(sel4) and share(sel5) <= 0.1,
          (len(sel5), len(sel4), share(sel5), s5["sizes"]))
    check("refill: no refilled image was in the first pick",
          all(k not in picked4 for k in refilled))
    b6 = TMP / "b6"
    S.build(0.6, 0, 3, refill=True, out_dir=b6, other_only_frac=1.0)
    sel6 = set(keys(b6 / S.BASE_SELECTED))
    pool_share = len(other_only & set(feat)) / len(feat)
    check("without the budget, caps + refill drift toward OtherPlant-only images (it binds)",
          share(sel6) > pool_share > 0.1 >= share(sel5), (share(sel6), pool_share, share(sel5)))
    for b in (b4, b5, b6):
        clb = S.read_clusters(b / S.CLUSTERS)
        selb = set(keys(b / S.BASE_SELECTED))
        check("near-copies on one side in %s" % b.name,
              all(len({k in selb for k in g}) == 1 for g in planted)
              and all(len({clb[k]["status"] for k in g}) == 1 for g in planted))

    # --- 4. reruns ----------------------------------------------------------------
    print("reruns")
    created = json.load(open(b1 / S.SUMMARY))["created"]
    S.build(0.6, 0, 100, out_dir=b1, other_only_frac=1.0)
    check("same inputs and parameters -> no-op", json.load(open(b1 / S.SUMMARY))["created"] == created)
    check("different parameters over an existing build -> refused",
          raises(lambda: S.build(0.5, 0, 100, out_dir=b1, other_only_frac=1.0)) is not None)
    check("a different gate over an existing build -> refused",
          raises(lambda: S.build(0.6, 0, 100, out_dir=b1, other_only_frac=1.0, gate=0.2)) is not None)
    check("--force overwrites", S.build(0.5, 0, 100, out_dir=TMP / "b2", force=True)
          ["sizes"]["quota"] == round(0.5 * n_ver))

    # --- 5. increments --------------------------------------------------------------
    print("increments")
    pool_keys = set(keys(b1 / S.POOL))
    e1 = TMP / "inc" / "e1" / "manifests"
    si = S.increments("e1", 3, 60, base_dir=b1)
    incs = [keys(e1 / ("inc_%02d.jsonl" % j)) for j in (1, 2, 3)]
    check("three increments of exactly 60 images", [len(x) for x in incs] == [60, 60, 60])
    allk = [k for x in incs for k in x]
    check("increments are disjoint", len(set(allk)) == 180)
    check("increments come from the increment pool only", set(allk) <= pool_keys)
    where = {k: j for j, x in enumerate(incs) for k in x}
    pool_groups = [g for g in planted if g <= pool_keys]
    check("near-copies in the pool are drawn whole, into one increment",
          pool_groups and all(len({where.get(k, -1) for k in g}) == 1 for g in pool_groups),
          [[where.get(k, -1) for k in g] for g in pool_groups])
    first = [(e1 / ("inc_%02d.jsonl" % j)).read_bytes() for j in (1, 2, 3)]
    S.increments("e1", 3, 60, base_dir=b1, force=True)
    check("same experiment name -> identical increments",
          first == [(e1 / ("inc_%02d.jsonl" % j)).read_bytes() for j in (1, 2, 3)])
    S.increments("e2", 3, 60, base_dir=b1)
    e2 = [set(keys(TMP / "inc" / "e2" / "manifests" / ("inc_%02d.jsonl" % j))) for j in (1, 2, 3)]
    check("another experiment name draws differently", set().union(*e2) != set(allk))
    units = group_units(cl, pool_keys)
    ucl = {g: next((int(cl[k]["l1"]) for k in ks if int(cl[k]["l1"]) >= 0), -1)
           for g, ks in units.items()}
    avail, drawn = {}, {}
    for g in units:
        avail[ucl[g]] = avail.get(ucl[g], 0) + 1
    drawn_g = {cl[k]["dup_group"] for k in allk}
    for g in drawn_g:
        drawn[ucl[g]] = drawn.get(ucl[g], 0) + 1
    notfull = [drawn.get(c, 0) for c in avail if drawn.get(c, 0) < avail[c]]
    check("the draw is balanced across clusters (water-filling, in groups)",
          all(drawn.get(c, 0) >= min(avail[c], max(drawn.values()) - 1) for c in avail)
          and (not notfull or max(notfull) - min(notfull) <= 1), (avail, drawn))
    big = max(len(units[g]) for g in drawn_g)
    spread = all(max(c) - min(c) <= big for c in
                 ([len({cl[k]["dup_group"] for k in x if ucl[cl[k]["dup_group"]] == c0}) for x in incs]
                  for c0 in avail))
    check("each cluster's share splits evenly across the increments", spread)
    check("increments summary lists per-species boxes and sizes",
          [x["images"] for x in si["increments"]] == [60, 60, 60]
          and all(sum(x["boxes"].values()) == sum(len(W["boxes"][k]) for k in inc)
                  for x, inc in zip(si["increments"], incs))
          and si["never_train"]["checked"] == 180)
    check("a draw larger than the pool is refused",
          raises(lambda: S.increments("e3", 10, 100, base_dir=b1)) is not None)
    check("a different draw over existing increments is refused",
          raises(lambda: S.increments("e1", 2, 60, base_dir=b1)) is not None)
    S.increments("e1", 2, 60, base_dir=b1, force=True)
    check("--force with fewer increments removes the stale manifest",
          not (e1 / "inc_03.jsonl").exists() and (e1 / "inc_02.jsonl").exists())
    sd = S.increments("e_def", 1, base_dir=b1)
    want = round(0.1 * s1["sizes"]["base_B"])
    check("default increment size = 10 % of base B's images",
          sd["params"]["size"] == want and len(keys(TMP / "inc" / "e_def" / "manifests" /
                                                    "inc_01.jsonl")) == want, (sd["params"], want))
    so = S.increments("e_oh", 2, 40, other_heavy=True, base_dir=b1)
    eo = TMP / "inc" / "e_oh" / "manifests"
    oh = keys(eo / S.INC_OTHER)
    reg = keys(eo / "inc_01.jsonl") + keys(eo / "inc_02.jsonl")
    heavy_ok = all(counts[k][12] >= 0.5 * counts[k].sum() for k in oh)
    check("--other-heavy: one more increment of the same size, every image OtherPlant-heavy, "
          "disjoint from the regular ones",
          len(oh) == 40 and heavy_ok and not set(oh) & set(reg) and len(reg) == 80
          and so["increments"][-1]["kind"] == "otherplant_heavy"
          and so["increments"][-1]["otherplant_box_share"] >= 0.5, so["increments"][-1])
    rc = S.main(["increments", "--exp", "cli_e", "--n", "2", "--size", "5", "--base-dir", str(b1)])
    check("CLI increments", rc == 0 and len(keys(TMP / "inc" / "cli_e" / "manifests" / "inc_02.jsonl")) == 5)
    saved = C.NEVER_TRAIN_INDEX.read_bytes()
    idx = json.loads(saved)
    idx["entries"] += [[W["hash"][k] ^ 1, "dev", "/x/near_%s.jpg" % k] for k in sorted(pool_keys)]
    C.NEVER_TRAIN_INDEX.write_text(json.dumps(idx))
    err = raises(lambda: S.increments("e_nt", 1, 10, base_dir=b1))
    C.NEVER_TRAIN_INDEX.write_bytes(saved)
    check("an increment holding an evaluation image is refused (never-train guard)",
          err is not None and "never-train" in err
          and not (TMP / "inc" / "e_nt" / "manifests" / "inc_01.jsonl").exists(), err)

    # --- 5b. the relevance filter (inc/relevance.py) ------------------------------------
    print("increments with the relevance filter")
    from weed_optimizer_framework.tools.inc import relevance as REL
    rng_w = np.random.default_rng(1234)                   # the world's basis (make_world's first draw)
    basis = np.linalg.qr(rng_w.normal(size=(D, D)))[0]
    scene_dir, class_dir = basis[:9], basis[9:9 + C.NC]
    # plant prompts span the species directions (five groups of the 13 class
    # directions), the non-plant ones sit on the off-domain scene. train_core
    # crops have no scene component, so at BioCLIP-2's logit scale (about 100)
    # at least 95 % of them read as plant: the calibration check holds.
    groups = [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]
    plant_vecs = dict(zip(REL.PROMPTS["plant"], (class_dir[g].sum(0) for g in groups)))

    class Enc:
        model, name, logit_scale = "fake", "fake text tower", 100.0

        def __call__(self, prompts):
            return np.array([plant_vecs.get(p, scene_dir[OFF]) for p in prompts])
    rel = REL.build(base_dir=b1, text_encoder=Enc())
    rel_path = b1 / REL.OUT_NAME
    st_rel = {s: e["status"] for s, e in rel["increment_pool"]["sources"].items()}
    usable = {}
    for i, (st_, k, _b, _c, s_) in enumerate(W["crops"]):
        if st_ == "pool" and k in pool_keys and np.isfinite(W["X16"][i].astype(np.float32)).all():
            usable[s_] = usable.get(s_, 0) + 1
    want_st = {s: ("fail" if s == "harvOff/train" else "insufficient" if usable.get(s, 0) < REL.MIN_CROPS
                   else "pass") for s in s1["sources"]["increment_pool"]}
    n_pool = s1["sources"]["increment_pool"]
    check("relevance on the select build: the off-domain source fails, the on-domain ones pass, the forks "
          "left in the pool (harvC) are too few to judge",
          st_rel == want_st and want_st["harvC/train"] == "insufficient" and want_st["harvA/train"] == "pass"
          and rel["increment_pool"]["not_passing"] == {s: n_pool[s] for s in n_pool if want_st[s] != "pass"}
          and n_pool["harvOff/train"] == 100 and rel["calibration"]["check"]["ok"] is True, (st_rel, usable))
    excl = {s: {"images": n_pool[s], "status": want_st[s]} for s in n_pool if want_st[s] != "pass"}
    n_excl = sum(x["images"] for x in excl.values())
    n_split = sum(1 for g in planted if g <= pool_keys and {src_of[k] == "harvC/train" for k in g} == {True, False})
    sr = S.increments("e_rel", 2, 40, other_heavy=True, base_dir=b1, relevance=rel_path)
    er = TMP / "inc" / "e_rel" / "manifests"
    drawn_r = [r for n in ("inc_01.jsonl", "inc_02.jsonl", S.INC_OTHER) for r in C.read_manifest(er / n)]
    check("no increment (regular or OtherPlant-heavy) holds an image of a source that does not pass",
          len(drawn_r) == 120 and not [r for r in drawn_r if want_st[r["source"]] != "pass"])
    S.increments("e_norel", 2, 40, other_heavy=True, base_dir=b1)
    en = TMP / "inc" / "e_norel" / "manifests"
    check("... while the same draw without the filter does hold some (the filter is not vacuous)",
          any(r["source"] == "harvOff/train" for n in ("inc_01.jsonl", "inc_02.jsonl")
              for r in C.read_manifest(en / n)))
    ri = sr["relevance"]
    check("the summary records the relevance file's sha256 (also a draw parameter) and the excluded sources "
          "with their images",
          ri["sha256"] == C.sha256_file(rel_path) == sr["params"]["relevance_sha256"]
          and ri["excluded_sources"] == excl and ri["excluded_images"] == n_excl
          and ri["pool_images_before"] == len(pool_keys) and sr["pool_images"] == len(pool_keys) - n_excl
          and ri["tau"] == rel["calibration"]["tau"], ri)
    check("a near-dup group split by the exclusion is counted (a harvC fork out, its original drawable)",
          ri["near_dup_groups_split"] == n_split > 0, (ri["near_dup_groups_split"], n_split))
    check("an unfiltered draw keeps its parameters as before (no relevance key) and records no filter",
          "relevance_sha256" not in json.loads((en / S.INC_SUMMARY).read_text())["params"]
          and json.loads((en / S.INC_SUMMARY).read_text())["relevance"] is None)
    check("a filtered draw over an unfiltered one of the same experiment refuses without --force",
          raises(lambda: S.increments("e_norel", 2, 40, other_heavy=True, base_dir=b1,
                                      relevance=rel_path)) is not None)
    err = raises(lambda: S.increments("e_rel3", 1, 10, base_dir=b3, relevance=rel_path))
    check("a relevance file made for another select build's increment pool is refused",
          err is not None and "another" in err and not (TMP / "inc" / "e_rel3").exists(), err)
    # apply_relevance on its own: the file must hold every pool source with the pool's image counts
    pool_rows = sorted(C.read_manifest(b1 / S.POOL), key=lambda r: r["key"])
    cl = S.read_clusters(b1 / S.CLUSTERS)
    ld = REL.load(rel_path, json.loads((b1 / S.SUMMARY).read_text()))
    miscount = dict(ld, images=dict(ld["images"], **{"harvA/train": ld["images"]["harvA/train"] + 1}))
    err_c = raises(lambda: S.apply_relevance(pool_rows, miscount, cl))
    unknown = dict(ld, status={s: v for s, v in ld["status"].items() if s != "harvB/valid"})
    err_u = raises(lambda: S.apply_relevance(pool_rows, unknown, cl))
    kept, info = S.apply_relevance(pool_rows, ld, cl)
    check("apply_relevance refuses a file whose per-source image counts are not the pool's, or that misses a "
          "pool source; with the file as made, it keeps exactly the passing sources' rows",
          err_c is not None and "counts other" in err_c and "harvA/train" in err_c
          and err_u is not None and "no entry" in err_u
          and kept == [r for r in pool_rows if want_st[r["source"]] == "pass"]
          and info["excluded_images"] == n_excl, (err_c, err_u))
    rc = S.main(["increments", "--exp", "cli_rel", "--n", "2", "--size", "5", "--base-dir", str(b1),
                 "--relevance", str(rel_path)])
    check("CLI increments --relevance",
          rc == 0 and json.loads((TMP / "inc" / "cli_rel" / "manifests" / S.INC_SUMMARY).read_text())
          ["relevance"]["excluded_images"] == n_excl)

    # --- 6. refusals --------------------------------------------------------------------
    print("refusals")
    core = W["core_rows"]
    row = dict(W["pool_rows"][1])
    check("a verified key equal to a train_core key is refused",
          raises(lambda: S.check_pool_against_core([dict(row, key=core[0]["key"])], core)) is not None)
    check("a verified image that is a train_core image is refused",
          raises(lambda: S.check_pool_against_core([dict(row, image=core[0]["image"])], core)) is not None)
    check("a verified image byte-identical to a train_core image is refused",
          raises(lambda: S.check_pool_against_core([dict(row, sha256=core[0]["sha256"])], core)) is not None)
    check("a verified image from a never-train dataset is refused",
          raises(lambda: S.check_pool_against_core(
              [dict(row, source="project_agml__imageweeds_weed_detection/train")], core)) is not None
          and raises(lambda: S.check_pool_against_core(
              [dict(row, image="/r/datasets/weedsense/images/a.jpg")], core)) is not None)
    check("a verified image from a slug holding cwd12 copies is refused",
          all(raises(lambda s=s: S.check_pool_against_core(
              [dict(row, source=s, image="/r/results/leave4out/data/test/images/b.jpg")], core))
              is not None for s in ("cottonweed_holdout", "cottonweed_sp8")))
    check("an ordinary harvested row passes", raises(lambda: S.check_pool_against_core([row], core)) is None)
    for defect, what, needle in (
            ("stale_shard", "an embedding shard made from another crops.csv", "verify inputs"),
            ("verified_changed", "verified.jsonl changed after verify admit", "admit"),
            ("admitted_nan", "an admitted image's crop without features", "no features"),
            ("admit_embedder", "shards that are not the ones admit judged with", "not the ones"),
            ("admit_nshards", "admit judged with another shard count", "nshards=2"),
            ("never_train_hit", "a verified image within 6 bits of an evaluation image", "never-train"),
            ("no_dhash", "a verified image without a dHash in pool_meta.jsonl", "no dHash")):
        make_world(core_path, defect)
        err = raises(lambda: S.build(0.6, 0, 3, out_dir=TMP / ("bad_" + defect)))
        check("refused: %s" % what, err is not None and needle in err, err)
        check("refused before writing: %s" % defect,
              not (TMP / ("bad_" + defect) / S.BASE_B).exists())
    make_world(core_path)
    C.write_manifest(TMP / "core_minus_one.jsonl", core[1:])
    err = raises(lambda: S.build(0.6, 0, 3, out_dir=TMP / "b7",
                                 core_manifest=TMP / "core_minus_one.jsonl"))
    check("a train_core manifest that is not the locked one is refused", err and "LOCK" in err, err)
    S.build(0.6, 0, 100, out_dir=TMP / "b8", other_only_frac=1.0)
    check("the rebuilt world selects the same bytes as before",
          (TMP / "b8" / S.BASE_B).read_bytes() == (b1 / S.BASE_B).read_bytes())

    test_default_draw(W)
    test_evidence(core_path)
    make_world(core_path)

    # --- 7. pieces ------------------------------------------------------------------------
    print("pieces")
    rr = S.round_robin([np.array([1, 2, 3, 4]), np.array([10]), np.array([20, 21])]).tolist()
    check("round_robin interleaves and skips exhausted groups", rr == [1, 10, 20, 2, 21, 3, 4], rr)
    check("canonical_labels numbers by first appearance",
          S.canonical_labels(np.array([5, 5, 2, 9, 2])).tolist() == [0, 0, 1, 2, 1])
    rng = np.random.default_rng(7)
    cnt = rng.integers(0, 3, size=(200, C.NC))
    caps = [40, 60, 1000] + [80] * 9
    order = rng.permutation(200)
    res = S.choose(order, 150, cnt, caps)
    sel, dropped = res["sel"], res["cap_drops"]
    tot = cnt[order[:150]].sum(0)
    live = set(int(i) for i in order[:150])
    replay_ok, n_over = True, int((tot[:12] > np.array(caps)).sum())
    for i in dropped:                  # each drop: the worst species' last-picked carrier
        ratio = np.where(tot[:12] > caps, tot[:12] / np.maximum(caps, 0.5), -1.0)
        k = int(np.argmax(ratio))
        last = next(int(j) for j in order[:150][::-1] if int(j) in live and cnt[j, k] > 0)
        replay_ok &= i == last
        live.discard(i)
        tot = tot - cnt[i]
    check("caps: several species over at once (%d) all end within cap" % n_over,
          n_over >= 2 and all(cnt[sel].sum(0)[k] <= caps[k] for k in range(12)) and len(dropped) > 0)
    check("caps: each drop is the most-over species' last-picked carrier", replay_ok)
    check("caps: stops once nothing is over (undoing the last drop breaks a cap)",
          any(cnt[sel].sum(0)[k] + cnt[dropped[-1]][k] > caps[k] for k in range(12)))
    # the budget alone: 100 units, every third OtherPlant-only
    cnt_b = np.zeros((100, C.NC), dtype=np.int64)
    oo = np.arange(100) % 3 == 0
    cnt_b[~oo, 1] = 1
    cnt_b[oo, 12] = 1
    rb = S.choose(np.arange(100), 50, cnt_b, [1000] * 12, other_only=oo, other_frac=0.1)
    check("budget: the pick passes over OtherPlant-only units beyond floor(0.1 * Q)",
          rb["sel"].sum() == 50 and (rb["sel"] & oo).sum() == 5
          and min(u for u, s in rb["status"].items() if s == "other_budget") > 12, rb["sel"].sum())
    cnt_c = cnt_b.copy()
    rc_ = S.choose(np.arange(100), 50, cnt_c, [20] + [10] + [1000] * 10, other_only=oo,
                   other_frac=0.1)
    n_sel = int(rc_["sel"].sum())
    check("budget: after cap drops the last-picked OtherPlant-only units are dropped to fit",
          (rc_["sel"] & oo).sum() <= 0.1 * n_sel and "dropped_other_budget" in rc_["status"].values(),
          (n_sel, int((rc_["sel"] & oo).sum())))
    rs = S.choose(np.arange(3), 4, np.zeros((3, C.NC), dtype=np.int64), [0] * 12,
                  sizes=[3, 3, 1])
    check("sizes: a group that would overshoot the quota is passed over",
          rs["sel"].tolist() == [True, False, True], rs["sel"].tolist())
    rs = S.choose(np.arange(5), 4, np.zeros((5, C.NC), dtype=np.int64), [0] * 12,
                  sizes=[1, 1, 3, 1, 1])
    check("sizes: ... and the pick goes on with the next groups",
          rs["sel"].tolist() == [True, True, False, True, True])
    check("k1 = round(sqrt(N/10)) clamped to [8, 256]",
          (S.k1_for(100), S.k1_for(100_000), S.k1_for(10_000_000), S.k1_for(5)) == (8, 100, 256, 5))

    # k-means runs on one thread (sklearn's Lloyd reduction order depends on the thread count)
    import sklearn.cluster
    import threadpoolctl
    seen, orig_km = [], sklearn.cluster.KMeans

    class Spy(orig_km):
        def fit(self, X, y=None, sample_weight=None):
            seen.append({i["user_api"]: i["num_threads"] for i in threadpoolctl.threadpool_info()})
            return super().fit(X, y, sample_weight)
    sklearn.cluster.KMeans = Spy
    try:
        S._kmeans(np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32), 5, 0)
        S._kmeans_centres(np.random.default_rng(1).normal(size=(200, 8)).astype(np.float32), 5, 0)
    finally:
        sklearn.cluster.KMeans = orig_km
    check("k-means fits run with every thread pool limited to 1",
          len(seen) == 2 and all(v == 1 for d in seen for v in d.values()), seen)

    # dup_groups against brute force, with a crowded bucket (> 256 hashes share a block)
    rng = np.random.default_rng(11)
    hs = [int.from_bytes(rng.bytes(8), "big") for _ in range(300)]
    hs += [int.from_bytes(rng.bytes(8), "big") & ~0xFFFF for _ in range(400)]   # zero low block
    for t in range(60):
        h = hs[int(rng.integers(0, len(hs)))]
        for f in rng.choice(64, size=int(rng.integers(0, 4)), replace=False):
            h ^= 1 << int(f)
        hs.append(h)
    g = S.dup_groups(hs)
    parent = list(range(len(hs)))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i in range(len(hs)):
        for j in range(i):
            if _bits(hs[i], hs[j]) <= 3:
                parent[find(i)] = find(j)
    brute = S.canonical_labels(np.array([find(i) for i in range(len(hs))]))
    check("dup_groups = brute-force transitive near-dup components (crowded bucket too)",
          g.tolist() == brute.tolist() and len(set(g.tolist())) < len(hs),
          (len(set(g.tolist())), len(set(brute.tolist()))))

    # typicality: an outlier crop is atypical only if it did not fit its own prototypes
    rng = np.random.default_rng(3)
    A, B = np.eye(8)[0], np.eye(8)[1]
    E = np.array([B] + [A + 0.05 * rng.normal(size=8) for _ in range(399)], dtype=np.float16)
    oj = np.arange(400)
    img_fold = np.r_[0, np.arange(1, 400) % 2]
    typ, _info = S.typicality(E, oj, img_fold[oj], oj, np.arange(400), img_fold, 400)
    leak_fold = np.zeros(400, dtype=np.int64)              # its own crop in the fitted fold
    img_fold_l = np.r_[1, np.zeros(399, dtype=np.int64)]
    typ_l, _ = S.typicality(E, oj, leak_fold, oj[:1], np.zeros(1, dtype=np.int64), img_fold_l, 1)
    check("typicality is cross-fitted: an outlier scored on the other fold is atypical",
          typ[0] < 0.3 and np.nanmin(typ[1:]) > 0.9, (typ[0], np.nanmin(typ[1:])))
    check("typicality: the same outlier scored on centres fitted to its own crop looks typical",
          typ_l[0] > 0.9, typ_l[0])

    # draw_parts: a group that fits no part is replaced by the next ones
    parts, left = S.draw_parts([np.arange(8)], np.array([3, 3, 3, 1, 1, 1, 1, 1]), 2, 5)
    sz = np.array([3, 3, 3, 1, 1, 1, 1, 1])
    check("draw_parts: exact sizes, whole groups, a misfit left in the pool",
          [int(sz[p].sum()) for p in parts] == [5, 5] and left == [2]
          and not set(parts[0]) & set(parts[1]), (parts, left))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
