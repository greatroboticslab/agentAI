#!/usr/bin/env python3
"""inc2/embed_calibration.py: the v2 calibration of the embedding copy detector
(decision L-9(c), docs/CONTINUOUS_LOOP.md §2.6, §4.2 step 5b), on a synthetic
world of descriptors.

The world (64-dimensional unit descriptors, 8 dHash variants per image, every
file in funnel.embed's own format, so the calibration reads them exactly as
it reads the funnel's on the cluster):
  * evaluation: dev 20, test 60, imageweeds 20;
  * train_core 200 in capture sessions of other dates, among them 20 frames
    of test scenes (cosine 0.86-0.90 with a test image: the incident's scene
    similarity), 5 burst frames that share a capture session and date with a
    test image (cosine 0.97: excluded from their maximum, never negatives
    against it) and 1 dHash copy of a test image (left out of the
    negatives, counted);
  * Step 1's pool: 150 images of the funnel config's provenance-disjoint
    group, 60 + 60 of the two non-plant sources (easy), 30 of another source
    (in no tier);
  * the base: a funnel leak_v1.json whose descriptor files and seeds are
    recorded, its positives drawn exactly as funnel.leak draws them, its
    threshold the one they give (about 0.80, the incident's 0.8256 in
    spirit); the crop family has 30 % of its positives at cosine 0.84-0.87.

Pinned:
  * the threshold rises above the planted same-scene negatives (per-image
    false-positive rate <= 1 %, with the rate at the base threshold reported
    and far above it), never below the base, ignores the same-session burst
    frames, and an augmented copy of a test image is still caught by
    funnel.leak.detect while a same-scene image is not (it is at the base
    threshold);
  * every tier: n, false hits, rate, the one-sided 97.5 % bound, at both
    thresholds; the dHash copy left out and counted; strict threshold above
    every negative;
  * recall per family at the new threshold; crop below 0.95 is recorded as a
    known limit and the record still loads (not a refusal);
  * a rerun is a no-op without the embedder; compute=False gives None for a
    changed identity; a changed base descriptor file is not trusted, and
    positives that do not give back the base's threshold refuse;
  * load / record_problems / locked / state refuse: an unlisted family below
    the gate, a threshold below the floor, a small hard tier outside a
    testing record, a constraining tier over the gate, a file LOCK v2 does
    not record or that does not hash; inc2.guard.load_calibration and
    calibration_problems read the v2 record;
  * the source rule: a hit count within what the per-image rate predicts is
    not flagged; an improbable count, a dHash hit or a strict hit is;
  * step1_stream.load_scanner judges with the v2 threshold (taking its base's
    role) and falls back, recorded, when LOCK v2 does not record the file.

No network, no GPU. Run:  python3 tests/test_inc2_embed_calibration.py
"""
import copy
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc2_embed_cal_"))
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_DIR"] = str(TMP / "inc")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, text="", exc=Exception):
    try:
        fn()
    except exc as e:
        return text in str(e), str(e)
    return False, "no error"


try:
    import numpy as np
except ImportError as e:
    print("SKIP: every check (numpy not importable: %s)" % e)
    sys.exit(0)

from weed_optimizer_framework.tools.funnel import embed as E  # noqa: E402
from weed_optimizer_framework.tools.funnel import leak as L  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import embed_calibration as EC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import guard as G  # noqa: E402

DIM = 64
NAME = "facebook/dinov2-base:cls"
DOMAIN = json.loads((ROOT / "weed_optimizer_framework" / "tools" / "funnel" / "domains" / "weed.json").read_text())
MH = "project_agml__mh_weed16_weed_detection"
RNG = np.random.default_rng(20260929)
FDIR = TMP / "inc" / "funnel"
LDIR = TMP / "inc" / "splits" / "v2" / "leak"
OUT = TMP / "inc" / "splits" / "v2" / EC.NAME


class NoEmbedder:
    """Names the calibrated embedder; describing an image is a failure of the test."""
    name = NAME
    dim = DIM

    def __init__(self):
        self.calls = 0

    def __call__(self, pils):
        self.calls += 1
        raise AssertionError("the embedder was asked to describe images")


class BlankEmbedder(NoEmbedder):
    """Describes every image as the same vector (a recomputation that cannot
    give back the base's positives)."""

    def __call__(self, pils):
        self.calls += 1
        return np.ones((len(pils), DIM), dtype=np.float32)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def at_cos(e, c):
    """A unit vector at cosine c with the unit vector e."""
    u = RNG.normal(size=DIM)
    u -= (u @ e) * e
    u = unit(u)
    return unit(c * e + np.sqrt(max(0.0, 1.0 - c * c)) * u)


def rand_hashes(n):
    return RNG.integers(0, 2 ** 63, size=(n, 8), dtype=np.int64).astype(np.uint64)


def write_npz(path, rows, X, H, prepare):
    """An embed_images file of rows (funnel.embed's format and identity)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(E.image_ident(rows, NoEmbedder(), prepare, 8), format=E.IMAGE_FORMAT, n=len(rows), dim=DIM,
                stats={}, seconds=0.0, built_utc="2026-09-29T00:00:00Z")
    E._save_npz(path, meta, **E._pack([r["key"] for r in rows], np.asarray(X, dtype=np.float32),
                                      np.asarray(H, dtype=np.uint64), np.ones(len(rows), dtype=bool),
                                      [None] * len(rows)))
    return {"path": str(path), "sha256": C2.sha256_file(path)}


def norm16(X):
    """The descriptors as the files store them (float16), normalised as the code does."""
    X = np.asarray(X, dtype=np.float16).astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------- world
def make_world():
    W = {}
    ev_rows, ev_X, ev_H = {}, {}, {}
    for split, n, sess in (("dev", 20, "20210601_CamD_S%d"), ("test", 60, "20210810_CamT_S%d"),
                           ("imageweeds", 20, None)):
        rows = [{"key": "%s__e%03d" % (split, i), "image": "/eval/%s/%03d.jpg" % (split, i),
                 "sha256": "%064x" % (hash((split, i)) & (2 ** 256 - 1)), "session": (sess % (i % 6)) if sess else ""}
                for i in range(n)]
        ev_rows[split] = rows
        ev_X[split] = np.stack([unit(RNG.normal(size=DIM)) for _ in rows])
        ev_H[split] = rand_hashes(n)
    W["eval_rows"] = ev_rows
    items, X, H, splits, ekeys = [], [], [], [], []
    for split in sorted(ev_rows):
        order = sorted(range(len(ev_rows[split])), key=lambda i: ev_rows[split][i]["key"])
        for i in order:
            r = ev_rows[split][i]
            items.append({"key": "%s|%s" % (split, r["key"]), "image": r["image"], "sha256": r["sha256"]})
            X.append(ev_X[split][i])
            H.append(ev_H[split][i])
            splits.append(split)
            ekeys.append(r["key"])
    cache = write_npz(LDIR / "eval_desc_v2.npz", items, X, H, L.prepare_hashed)
    res = E.load_images(cache["path"])
    res["Xn"] = norm16(res["X"])
    W["index"] = L.EvalIndex(res, splits, ekeys, embedder=NoEmbedder())
    W["eval_cache"] = cache
    test_X = {r["key"]: ev_X["test"][i] for i, r in enumerate(ev_rows["test"])}
    test_H = {r["key"]: ev_H["test"][i] for i, r in enumerate(ev_rows["test"])}
    W["test_X"], W["test_H"] = test_X, test_H
    tkeys = sorted(test_X)

    # train_core: 174 unrelated, 20 frames of test scenes, 5 burst frames of a test session, 1 dHash copy
    core, cX, cH = [], [], []
    scene_cos = []
    for i in range(200):
        date = "202107%02d" % (1 + i % 28)
        row = {"key": "train_core__%s_CamA_S%d_%d" % (date, i % 9, i), "image": "/core/%03d.jpg" % i,
               "sha256": "%064x" % (10 ** 6 + i), "session": "%s_CamA_S%d" % (date, i % 9), "source": "cwd12/train"}
        h = rand_hashes(1)[0]
        if i < 20:
            c = float(RNG.uniform(0.86, 0.90))
            x = at_cos(test_X[tkeys[i]], c)
            scene_cos.append(c)
        elif i < 25:
            t = ev_rows["test"][i]                               # same session and date as this test image
            row["session"] = t["session"]
            row["key"] = "train_core__%s_%d" % (t["session"], i)
            x = at_cos(test_X[t["key"]], 0.97)
        elif i == 25:
            x = unit(RNG.normal(size=DIM))
            h = test_H[tkeys[30]].copy()                        # its dHash is a test image's
        else:
            x = unit(RNG.normal(size=DIM))
        core.append(row)
        cX.append(x)
        cH.append(h)
    order = sorted(range(len(core)), key=lambda i: core[i]["key"])
    core = [core[i] for i in order]
    cX = np.stack([cX[i] for i in order])
    cH = np.stack([cH[i] for i in order])
    W["core"], W["scene_cos"] = core, scene_cos
    files = {"reference": write_npz(FDIR / "emb_dinov2_images_reference.npz", core, cX, cH, L.prepare_hashed)}
    ref_Xn = norm16(cX)
    tn = {k: W["index"].Xn[j] for j, k in enumerate(W["index"].eval_key) if W["index"].split[j] == "test"}
    W["scene_cos_stored"] = [float(ref_Xn[j] @ tn[tkeys[int(r["key"].rsplit("_", 1)[1])]])
                             for j, r in enumerate(core) if int(r["key"].rsplit("_", 1)[1]) < 20]

    # the base's positives, drawn as funnel.leak draws them
    fams = L.family_params(DOMAIN)
    seeds = {"positives/%s" % f: "funnel/v1/leak/pos/%s" % f for f in L.FAMILIES}
    pos_cos, pos_dh = {}, {}
    for fam in L.FAMILIES:
        rows, pick = EC._positive_rows(core, fam, seeds["positives/%s" % fam], fams[fam])
        X, H = [], []
        for j, i in enumerate(pick):
            e = ref_Xn[i].astype(np.float64)
            if fam == "crop" and j % 10 < 3:
                c = float(RNG.uniform(0.84, 0.87))              # 30 %: below the v2 threshold, above the base's
            elif fam == "jpeg" and j % 20 < 3:
                c = float(RNG.uniform(0.78, 0.80))              # 15 %: the base's threshold comes from these
            else:
                c = float(RNG.uniform(0.95, 0.995))
            X.append(at_cos(e, c))
            h = rand_hashes(1)[0]
            if fam in ("flip", "rot90"):
                h[3] = cH[i][0]                                  # a flip or rotation: a variant is the original's dHash
            elif fam == "jpeg":
                h[0] = cH[i][0]                                  # re-encoding keeps the dHash
            H.append(h)
        pos_dh[fam] = fam in ("flip", "rot90", "jpeg")
        files["pos_%s" % fam] = write_npz(FDIR / ("emb_dinov2_images_pos_%s.npz" % fam), rows, X, H,
                                          L.prepare_augmented)
        Xn = norm16(X)
        pos_cos[fam] = np.einsum("ij,ij->i", Xn, ref_Xn[pick]).astype(np.float64)
    theta = round(L.threshold(pos_cos, L.RECALL_MIN), 6)
    W["theta"] = theta

    # Step 1's pool
    pool, pX = [], []
    for src, n in ((MH, 150), (EC.EASY_SOURCES[0], 60), (EC.EASY_SOURCES[1], 60), ("rf_other__x", 30)):
        for i in range(n):
            pool.append({"key": "%s__p%03d" % (src, i), "image": "/pool/%s/%03d.jpg" % (src, i),
                         "sha256": "%064x" % (hash((src, i)) & (2 ** 256 - 1)), "source": src, "session": ""})
            pX.append(unit(RNG.normal(size=DIM)))
    order = sorted(range(len(pool)), key=lambda i: pool[i]["key"])
    pool = [pool[i] for i in order]
    pX = np.stack([pX[i] for i in order])
    files["pool"] = write_npz(FDIR / "emb_dinov2_images_pool.npz", pool, pX, rand_hashes(len(pool)), L.prepare_hashed)
    W["pool"] = pool
    base_cal = {"ok": True, "why": [], "cos_threshold": theta, "dhash_bits_max": 6, "recall_min": L.RECALL_MIN,
                "fpr_max": L.FPR_MAX, "seeds": seeds,
                "positives": {f: {"n": len(pos_cos[f]), "hits": len(pos_cos[f]) if pos_dh[f]
                                  else int((pos_cos[f] >= theta).sum())} for f in L.FAMILIES},
                "negatives": {"pairs_7_10": {"n": 200, "false_hits": 0}, "hard": {"n": 2000, "false_hits": 12}}}
    base = {"format": "funnel-leak/1", "status": "complete", "calibration": base_cal,
            "detector": {"descriptor": {"embedder": NAME}}, "params": {"families": fams},
            "descriptor_files": files}
    W["base_path"] = FDIR / "leak_v1.json"
    C2.write_json_atomic(W["base_path"], base)
    W["files"] = files
    return W


def run(W, embedder=None, compute=True, force=False, **kw):
    base = EC.base_info(W["base_path"], "funnel_leak_v1", NAME)
    emb = embedder or NoEmbedder()
    args = dict(min_negatives=EC.MIN_NEGATIVES, dev_sessions=sorted({r["session"] for r in W["eval_rows"]["dev"]}),
                procs=1, testing=False, force=force, compute=compute, domain_sha="d" * 64)
    args.update(kw)
    return EC.calibrate(OUT, base, NAME, lambda: emb, lambda: W["index"], W["eval_cache"]["sha256"], W["eval_rows"],
                        W["core"], W["pool"], DOMAIN, LDIR, L.RECALL_MIN, L.FPR_MAX, **args), emb


# ------------------------------------------------------------------ checks
def test_calibration(W):
    print("the v2 calibration on hard same-domain negatives")
    rec, emb = run(W)
    doc = json.loads(OUT.read_text())
    cal = doc["calibration"]
    t, floor = rec["cos_threshold"], W["theta"]
    hard = cal["negatives"]["hard"]
    sc = sorted(W["scene_cos"])
    check("the base's positives, rebuilt from its seeds and files, give back its threshold %.6f" % floor,
          cal["base"]["recomputed_threshold"] == floor and cal["base"]["positives"] == "reproduced from its seeds"
          and cal["floor"] == floor and emb.calls == 0, cal["base"])
    sc = sorted(W["scene_cos_stored"])
    above = sum(1 for c in W["scene_cos_stored"] if np.float32(c) >= np.float32(t))
    check("the threshold rises above the planted same-scene negatives (%.3f-%.3f as stored): %.6f, with %d of the "
          "20 at or above it (floor(1 %% of %d) = 1 allowed); never below the base %.6f"
          % (sc[0], sc[-1], t, above, hard["n"], floor),
          above <= 1 and t > sc[-2] and t > floor and t < 0.95 and cal["ok"], (t, sc[-3:]))
    check("per-image false-positive rate <= 1 %% on the hard tier (%d images): %d false hits, one-sided 97.5 %% "
          "bound %.4f; at the base threshold %d (%.1f %%), bound %.4f"
          % (hard["n"], hard["false_hits"], hard["ub"], hard["at_base"]["false_hits"], 100 * hard["at_base"]["fpr"],
             hard["at_base"]["ub"]),
          hard["false_hits"] <= 0.01 * hard["n"] and hard["at_base"]["false_hits"] >= 20
          and hard["at_base"]["fpr"] > 0.09 and 0 < hard["ub"] < hard["at_base"]["ub"])
    check("the burst frames of a test capture session and date are scored without that session's images (they "
          "would be false hits at 0.97); the dHash copy of a test image is left out and counted",
          hard["n"] == 199 and hard["excluded_dhash_copies"] == 1 and hard["cos"]["max"] < 0.95, hard)
    for tier, n in (("provenance_disjoint", 150), ("easy", 120)):
        v = cal["negatives"][tier]
        check("tier %s: %d images, constraining, rates and bounds at both thresholds" % (tier, n),
              v["n"] == n and v["constraining"] and v["false_hits"] == 0 and v["ub"] is not None
              and "at_base" in v and v["sources"], v)
    neg_cos = [float(ln.split(",")[3]) for ln in (OUT.parent / EC.NEGATIVES_NAME).read_text().splitlines()[1:]]
    check("strict threshold above every negative image (%.6f; the highest negative %.6f)"
          % (cal["strict_threshold"], max(neg_cos)), cal["strict_threshold"] >= t
          and cal["strict_threshold"] >= max(neg_cos) and len(neg_cos) == doc["negatives_csv"]["rows"])
    crop = cal["positives"]["crop"]
    check("recall per family at the new threshold: crop %.3f (%.3f at the base) is a known limit, recorded; every "
          "other family at the gate" % (crop["recall"], crop["at_base"]["recall"]),
          [x["family"] for x in cal["known_limits"]] == ["crop"] and crop["recall"] < 0.95
          and crop["at_base"]["recall"] >= 0.95
          and all(v["recall"] >= 0.95 for f, v in cal["positives"].items() if f != "crop")
          and cal["positives"]["flip"]["dhash_hits"] == cal["positives"]["flip"]["n"])
    check("... and a record with a known limit still loads (L-9(c): a limit, not a refusal)",
          rec["known_limits"][0]["family"] == "crop" and G.load_calibration(OUT)["cos_threshold"] == t
          and G.calibration_problems(cal) == [])
    # an augmented copy of a test image is still caught, a same-scene image is not
    tk = sorted(W["test_X"])
    e0, e1 = W["test_X"][tk[40]], W["test_X"][tk[41]]
    imgs = [{"key": "copy", "path": "/q/copy.jpg", "desc": at_cos(e0, 0.975), "hashes": list(rand_hashes(1)[0])},
            {"key": "scene", "path": "/q/scene.jpg", "desc": at_cos(e1, 0.875), "hashes": list(rand_hashes(1)[0])}]
    new = {e["key"] for e in L.detect(imgs, W["index"], rec["calibration"])}
    old = {e["key"] for e in L.detect(imgs, W["index"], G.load_calibration(W["base_path"])["calibration"])}
    check("funnel.leak.detect at the v2 threshold: an augmented copy of a test image (cos 0.975) is caught, a "
          "same-scene image (0.875) is not; at the base threshold both are", new == {"copy"}
          and old == {"copy", "scene"}, (new, old))
    csv_text = (OUT.parent / EC.NEGATIVES_NAME).read_text()
    check("the negatives file lists the negatives by key with the evaluation key they are nearest to, never an "
          "evaluation image path", doc["negatives_csv"]["rows"] > 400 and "test__e" in csv_text
          and "/eval/" not in csv_text and doc["negatives_csv"]["sha256"] == C2.sha256_file(OUT.parent /
                                                                                          EC.NEGATIVES_NAME))
    check("the record names its decision, protocol, embedder and the evaluation descriptors",
          doc["decided_by"].startswith("L-9") and cal["protocol"] == EC.PROTOCOL
          and doc["detector"]["descriptor"]["embedder"] == NAME
          and doc["identity"]["eval_descriptors_sha256"] == W["eval_cache"]["sha256"])

    print("reruns and inputs")
    sha = C2.sha256_file(OUT)
    rec2, emb2 = run(W)
    check("a rerun with the same identity is a no-op (the embedder is never made to work)",
          rec2["cos_threshold"] == t and emb2.calls == 0 and C2.sha256_file(OUT) == sha)
    none, _e = run(W, compute=False, min_negatives=150)
    check("compute=False: a calibration whose identity changed is not made (None)", none is None
          and C2.sha256_file(OUT) == sha)
    ok, msg = raises(lambda: run(W, easy_sources=("not_a_listed_source",)), "not_recoverable", EC.CalibrationError)
    check("an easy negative source the funnel config does not name non-plant refuses", ok, msg)
    ok, msg = raises(lambda: run(W, min_negatives=250), "fewer than 250", EC.CalibrationError)
    check("a hard tier smaller than min_negatives fails the calibration (written with ok false, refused)",
          ok and json.loads(OUT.read_text())["calibration"]["ok"] is False, msg)
    ok, msg = raises(lambda: run(W, min_negatives=250), "failed", EC.CalibrationError)
    check("... and a rerun of the failed identity refuses again", ok, msg)
    run(W, force=True)
    return rec


def test_base_trust(W):
    print("the base's files are not taken on trust")
    p = pathlib.Path(W["files"]["pos_crop"]["path"])
    keep = p.read_bytes()
    rows, _pick = EC._positive_rows(W["core"], "crop", "funnel/v1/leak/pos/crop", L.family_params(DOMAIN)["crop"])
    res = E.load_images(p)
    write_npz(p, rows, res["X"].astype(np.float32)[::-1], res["H"], L.prepare_augmented)     # other descriptors
    emb = BlankEmbedder()
    try:
        ok, msg = raises(lambda: run(W, embedder=emb, force=True), "not the positives", EC.CalibrationError)
    finally:
        p.write_bytes(keep)
    check("a positives file that changed since the base recorded it is not used; recomputed, the positives do not "
          "give back the base's threshold and the calibration refuses", ok, msg[:300])
    check("... the recomputation went to the stream's own directory, never the funnel's",
          (LDIR / "v2cal_pos_crop.npz").exists() and C2.sha256_file(p) == W["files"]["pos_crop"]["sha256"])
    (LDIR / "v2cal_pos_crop.npz").unlink()
    run(W, force=True)


def test_record_problems(W):
    print("what a v2 record must show")
    doc = json.loads(OUT.read_text())
    good = doc["calibration"]
    check("the good record has no problem", EC.record_problems(good) == [])
    muts = {
        "an unlisted family below the gate": lambda c: c.update(known_limits=[]),
        "a known limit that is not below the gate": lambda c: c["known_limits"].append({"family": "flip"}),
        "a threshold below the floor": lambda c: c.update(cos_threshold=c["base"]["cos_threshold"] - 0.01),
        "a strict threshold below the threshold": lambda c: c.update(strict_threshold=c["cos_threshold"] - 0.01),
        "a small hard tier outside a testing record": lambda c: c.update(min_negatives=5),
        "a constraining tier over the gate": lambda c: c["negatives"]["hard"].update(false_hits=10),
        "the hard tier not constraining": lambda c: c.update(constraining=["easy"]),
        "looser gates": lambda c: c.update(fpr_max=0.05),
        "another dHash radius": lambda c: c.update(dhash_bits_max=8),
        "no base sha256": lambda c: c["base"]["file"].pop("sha256"),
        "ok false": lambda c: c.update(ok=False),
        "another protocol": lambda c: c.update(protocol="per_pair/v1"),
    }
    for why, fn in muts.items():
        c = copy.deepcopy(good)
        fn(c)
        probs = EC.record_problems(c)
        check("refused: %s" % why, bool(probs) and G.calibration_problems(c) == probs, probs)
    c = copy.deepcopy(good)
    c.update(min_negatives=5, testing=True)
    check("a testing record may hold a small hard tier", EC.record_problems(c) == [])
    bad = copy.deepcopy(doc)
    bad["calibration"]["known_limits"] = []
    p = TMP / "bad_v2.json"
    C2.write_json_atomic(p, bad)
    check("load and inc2.guard.load_calibration refuse such a file",
          raises(lambda: EC.load(p), "known limits", EC.CalibrationError)[0]
          and raises(lambda: G.load_calibration(p), "known limits", G.GuardError)[0])
    check("load refuses another embedder", raises(lambda: EC.load(OUT, embedder_name="x:cls"), "calibrated with")[0])

    print("LOCK v2 binds the file")
    lockp = TMP / "inc" / "splits" / "v2" / "LOCK.json"
    base_lock = {"splits_version": "v2", "eval_splits": ["dev", "test", "imageweeds"], "testing": False}
    C2.write_json_atomic(lockp, dict(base_lock, **{EC.LOCK_KEY: C2.sha256_file(OUT)}))
    rec, why = EC.locked(lockp)
    check("locked(): the file LOCK v2 records", rec is not None and rec["file"]["sha256"] == C2.sha256_file(OUT))
    C2.write_json_atomic(lockp, dict(base_lock, **{EC.LOCK_KEY: "0" * 64}))
    check("locked(): a file that does not hash as LOCK v2 records refuses",
          raises(lambda: EC.locked(lockp), "hashes to", EC.CalibrationError)[0])
    st, why = EC.state(OUT, lockp)
    check("state() (step1_stream's reader) refuses it too, with the reason", st is None and "hashes to" in why, why)
    C2.write_json_atomic(lockp, base_lock)
    check("locked(): a production LOCK that records none refuses (fail closed)",
          raises(lambda: EC.locked(lockp), "records no", EC.CalibrationError)[0])
    C2.write_json_atomic(lockp, dict(base_lock, testing=True))
    check("locked(): a testing LOCK that records none gives None, recorded", EC.locked(lockp)[0] is None)
    lockp.unlink()


def test_source_rule():
    print("the source rule accounts for the expected false hits")
    v = EC.source_verdict(100, 1, p_false=0.01)
    check("1 hit in 100 images at a 1 %% per-image rate: expected %.2f, not flagged" % v["expected_false_hits"],
          not v["flagged"] and v["p_value"] > EC.SOURCE_ALPHA, v)
    v = EC.source_verdict(6341, 60, p_false=0.012)
    check("60 hits in 6,341 images at 1.2 %%: within chance (expected %.1f), not flagged" % v["expected_false_hits"],
          not v["flagged"], v)
    v = EC.source_verdict(200, 12, p_false=0.01)
    check("12 hits in 200 images at 1 %%: improbable (P = %.2g), flagged" % v["p_value"], v["flagged"], v)
    check("any hit within 6 dHash bits flags, and any hit at or above the strict threshold",
          EC.source_verdict(500, 1, dhash_hits=1, p_false=0.01)["flagged"]
          and EC.source_verdict(500, 1, strict_hits=1, p_false=0.01)["flagged"])
    check("hits with no rate to compare with flag (fail closed)", EC.source_verdict(10, 1, p_false=None)["flagged"])
    f32 = lambda xs: np.array(xs, dtype=np.float32)     # noqa: E731
    t3 = EC.threshold_for(f32([0.9, 0.8, 0.7]), 0.01)          # n 3: no false hit allowed
    t100 = EC.threshold_for(f32([0.5] * 97 + [0.9, 0.95, 0.99]), 0.01)    # n 100: one allowed
    t202 = EC.threshold_for(f32([0.5] * 200 + [0.9, 0.95]), 0.01)          # n 202: two allowed
    check("the threshold is the smallest 6-decimal cosine with at most floor(1 %% of n) images at or above it, "
          "compared in float32 as the scan compares (%s, %s, %s)" % (t3, t100, t202),
          0.9 < t3 <= 0.900002 and (f32([0.9]) >= np.float32(t3)).sum() == 0
          and 0.95 < t100 <= 0.950002 and 0.5 < t202 <= 0.500002 and EC.threshold_for(f32([]), 0.01) is None
          and EC.above_all(f32([0.93]), 0.5) > 0.93 and EC.above_all(f32([0.3]), 0.5) == 0.5)
    check("capture sessions: a date_camera stem is one, a field name is not",
          EC.capture_session("20221007_iPhoneSE_YL") and EC.capture_date("20221007_iPhoneSE_YL") == "20221007"
          and EC.capture_session("field22_a") is None and EC.capture_session("") is None)


def test_step1_scanner(W, rec):
    print("step1_stream's copy scan judges with the v2 threshold")
    from weed_optimizer_framework.tools.inc2 import step1_stream as SS
    lay = types.SimpleNamespace(inc_dir=TMP / "inc", leak_dir=TMP / "inc" / "stream_leak")
    rows = {s: sorted(v, key=lambda r: r["key"]) for s, v in W["eval_rows"].items()}
    items = [{"key": "%s|%s" % (s, r["key"]), "image": r["image"], "sha256": r["sha256"]}
             for s in sorted(rows) for r in rows[s]]
    res = E.load_images(W["eval_cache"]["path"])
    cache = lay.leak_dir / ("eval_desc_%s.npz" % SS._sha_bytes(NAME.encode())[:12])
    write_npz(cache, items, res["X"].astype(np.float32), res["H"], L.prepare_hashed)
    sc = SS.load_scanner(lay, embedder=NoEmbedder(), eval_rows=rows, procs=1)
    check("the funnel's leak_v1.json passes, but the v2 calibration takes its role: threshold %.6f, not %.6f"
          % (rec["cos_threshold"], W["theta"]), sc.cal["funnel"] is not None and sc.cal["funnel"]["path"] == str(OUT)
          and sc.cal["funnel"]["threshold"] == rec["cos_threshold"] and sc.which(True, False) == "funnel"
          and sc.record()["funnel"]["cos_threshold"] == rec["cos_threshold"], sc.record())
    own = TMP / "inc" / "splits" / "v2" / "leak" / "leak_calibration.json"
    shutil.copyfile(W["base_path"], own)
    sc = SS.load_scanner(lay, embedder=NoEmbedder(), eval_rows=rows, procs=1)
    check("... and the role of the stream's own calibration when one is present",
          sc.cal["own"]["path"] == str(OUT) and sc.cal["funnel"]["path"] == str(OUT))
    own.unlink()
    lockp = TMP / "inc" / "splits" / "v2" / "LOCK.json"
    C2.write_json_atomic(lockp, {"splits_version": "v2", "eval_splits": ["dev", "test", "imageweeds"],
                                 "testing": False, EC.LOCK_KEY: "0" * 64})
    sc = SS.load_scanner(lay, embedder=NoEmbedder(), eval_rows=rows, procs=1)
    check("a v2 file LOCK v2 does not record is rejected (recorded) and the funnel's threshold stays in force",
          sc.cal["funnel"]["path"] == str(W["base_path"]) and str(OUT) in sc.record().get("rejected", {}),
          sc.record())
    lockp.unlink()


def main():
    W = make_world()
    check("the world's base threshold (the funnel's positives) is about 0.80: %.6f" % W["theta"],
          0.77 < W["theta"] < 0.83)
    rec = test_calibration(W)
    test_base_trust(W)
    test_record_problems(W)
    test_source_rule()
    test_step1_scanner(W, rec)
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), FAILURES))
        return 1
    print("all inc2.embed_calibration checks passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
