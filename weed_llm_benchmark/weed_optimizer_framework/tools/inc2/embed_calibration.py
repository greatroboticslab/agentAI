"""The v2 calibration of the embedding copy detector: per-image false
positives on hard same-domain negatives (decision L-9(c), docs/CONTINUOUS_LOOP.md
§2.6, §4.2 step 5b).

Why a second calibration. The funnel's leak_v1 calibration (and the stream's
own, which uses the same code) sets the DINOv2 CLS cosine threshold from
augmented positives and scores its negatives per PAIR: a negative image
against one partner. The scan flags an IMAGE when its best cosine over every
evaluation image (about 5,800) reaches the threshold, so the per-image
false-positive rate is far above the per-pair one; and its negatives were
provenance-disjoint sources, never same-domain scenes. On the real candidates
the threshold 0.8256 flagged 805 of tsw22's 1,915 images, every high pair a
2022 capture against a 2021 capture of another session: scene similarity,
not copying. Nothing in the funnel (funnel/**, leak_v1.json, its H6) changes;
this calibration is inc2's, and it is what inc2 judges non-provenance images
by (base B's kept part in splits v2, every intake row in step1_stream).

Protocol (PROTOCOL):
  base        the calibration the embedding scan used (the funnel's passed
              leak_v1.json, else the stream's own under splits/v2/leak/). Its
              threshold is the floor: the v2 threshold is never below it.
              When the base records its seeds, its positives are rebuilt from
              the descriptor files it lists (or recomputed into this
              directory) and must reproduce its threshold, else refused.
  negatives   per IMAGE, scored the way the scan scores an image: the
              maximum cosine over every v2 evaluation image (dev, test,
              imageweeds), leaving out the evaluation images of the negative's
              own capture session and of its capture date where those are
              known. A negative within 6 dHash bits of an evaluation image
              (its dHash or one of the 8 flips and rotations) is a dHash copy,
              not a negative: left out and counted (the L-8 case).
      hard                   train_core images with a capture session (same
                             lab, same fields, other sessions and days)
      hard_session_disjoint  the hard images whose session has no dev or test
                             frame (cwd12's own split is random, so most
                             train_core sessions also have test frames; those
                             frames are excluded from the image's maximum)
      provenance_disjoint    the funnel config's provenance-disjoint negative
                             groups (leak.negative_source_pairs, not the
                             reference), from Step 1's pool
      easy                   clearly unrelated harvested sources (EASY_SOURCES,
                             each named a non-plant or leaf-disease source by
                             the funnel config's sources.not_recoverable)
  threshold   the smallest cosine (6 decimals) at which the per-image
              false-positive rate is <= fpr_max on every constraining tier:
              hard always (it must hold >= min_negatives images), each other
              tier when it holds >= min_negatives images; never below the
              base threshold. Per tier: false hits, rate and the one-sided
              97.5 % upper bound (Clopper-Pearson), at the new threshold and
              at the base one.
  strict      the smallest cosine above every negative image of every tier.
  recall      the base's positives per augmentation family (funnel.leak
              FAMILIES) re-scored at the new threshold (cos >= t or dHash
              within 6 bits under a variant). A family below recall_min is a
              known limit, recorded, not a refusal: the dHash variants still
              catch flips, rotations and re-encoding.
  source rule a source is flagged only on a hit within 6 dHash bits, a hit at
              or above the strict threshold, or a hit count improbable under
              the per-image false-positive rate: P(Binom(n, p) >= hits) <
              SOURCE_ALPHA with p the hard tier's upper bound at the new
              threshold (source_verdict).

Outputs (splits/v2/, beside LOCK v2, which records the sha256): NAME and
NEGATIVES_NAME (one row per negative image: tier, key, source, cosine, the
evaluation split and key it is nearest to; evaluation keys only, never an
evaluation image path). Descriptors that must be computed go to the caller's
leak directory; the funnel's files are only read.

Standard library at module level; numpy, funnel.embed, funnel.leak and
funnel.estimate are imported where used.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from . import common as C2

FORMAT = "inc2-embed-calibration/2"
PROTOCOL = "per_image_max_cosine/v2"
NAME = "embed_calibration_v2.json"
NEGATIVES_NAME = "embed_calibration_v2_negatives.csv"
NEGATIVES_HEADER = ("tier", "key", "source", "cos", "eval_split", "eval_key", "bits")
LOCK_KEY = "embed_calibration_v2_sha256"
DECISION = "L-9(c), docs/CONTINUOUS_LOOP.md 2.6 (human-delegated, 2026-09-29)"
TIERS = ("hard", "hard_session_disjoint", "provenance_disjoint", "easy")
MIN_NEGATIVES = 100                 # a tier smaller than this cannot resolve a 1 % rate
MIN_NEGATIVES_TESTING = 5           # a synthetic world (recorded; lock refuses a testing build)
SOURCE_ALPHA = 0.001
UB_CONF = 0.95                      # a two-sided 95 % interval: its upper end is the one-sided 97.5 % bound
DHASH_BITS = C2.HOLDOUT_NEAR_DUP_BITS
EASY_SOURCES = ("fvossel__csgo_player_detection",
                "kg_farukalam__tomato-leaf-diseases-detection-computer-vision")
DATE_RE = re.compile(r"^(\d{8})_")
CAPTURE_RE = re.compile(r"^\d{8}_[A-Za-z0-9]")
CHUNK = 256
RULE = ("per image: copy iff max over the v2 evaluation images of the cosine >= cos_threshold, or min over the 8 "
        "variants of dHash bits <= dhash_bits_max")
SOURCE_RULE = ("a source is flagged only on a hit within %d dHash bits, a hit at or above strict_threshold, or "
               "P(Binom(images, p_false) >= hits) < %s (p_false: the hard tier's one-sided 97.5 %% upper bound at "
               "cos_threshold)" % (DHASH_BITS, SOURCE_ALPHA))


class CalibrationError(C2.Inc2Error):
    """The v2 calibration cannot be made, or a record cannot be used."""


def log(msg):
    print("[inc2.embed_calibration] %s" % msg, flush=True)


# ------------------------------------------------------------------ sessions
def capture_session(session):
    """The session when it is a capture session (the date_camera prefix of a
    file name: 8 digits, "_", a camera token), else None."""
    s = str(session or "")
    return s if CAPTURE_RE.match(s) else None


def capture_date(session):
    """The capture date (YYYYMMDD) of a date_camera session, else None."""
    m = DATE_RE.match(str(session or ""))
    return m.group(1) if m else None


# ------------------------------------------------------------------ statistics
def rate_record(scores, t):
    """{n, false_hits, fpr, ub} of per-image scores (float32 maxima) at
    threshold t, compared in float32 as the scan compares; ub is the
    one-sided 97.5 % Clopper-Pearson upper bound (None without images)."""
    import numpy as np
    from ..funnel import estimate as ES
    c = np.asarray(scores, dtype=np.float32)
    n = int(len(c))
    k = int((c >= np.float32(t)).sum()) if n else 0
    ub = round(float(ES.clopper_pearson(k, n, UB_CONF)[1]), 6) if n else None
    return {"n": n, "false_hits": k, "fpr": round(k / float(n), 6) if n else None, "ub": ub}


def threshold_for(scores, fpr_max):
    """The smallest 6-decimal cosine t with #(scores >= float32(t)) <=
    floor(fpr_max * n); None for an empty tier."""
    import numpy as np
    c = np.sort(np.asarray(scores, dtype=np.float32))[::-1]
    n = len(c)
    if n == 0:
        return None
    k = int(math.floor(fpr_max * n + 1e-9))
    if k >= n:
        return -1.0
    t = math.floor(float(c[k]) * 1e6 + 1.0) / 1e6
    while int((c >= np.float32(t)).sum()) > k:
        t = round(t + 1e-6, 6)
    return round(t, 6)


def above_all(scores, t_min):
    """The smallest 6-decimal cosine >= t_min above every score (float32)."""
    import numpy as np
    c = np.asarray(scores, dtype=np.float32)
    if not len(c):
        return round(float(t_min), 6)
    top = float(c.max())
    t = max(float(t_min), math.floor(top * 1e6 + 1.0) / 1e6)
    while bool((c >= np.float32(t)).any()):
        t = round(t + 1e-6, 6)
    return round(t, 6)


def source_verdict(images, hits, strict_hits=0, dhash_hits=0, p_false=None, alpha=SOURCE_ALPHA):
    """Whether a source's embedding hits say it holds copies, given the
    expected false hits (module docstring, source rule). images: the source's
    images scanned; hits: those with a hit at the v2 threshold."""
    from ..funnel import estimate as ES
    n, h = int(images), int(hits)
    p = None if p_false is None else float(p_false)
    pv = 1.0 if (h <= 0 or p is None) else float(ES.binom_upper_tail(h, n, p))
    why = []
    if int(dhash_hits) > 0:
        why.append("%d hit(s) within %d dHash bits" % (int(dhash_hits), DHASH_BITS))
    if int(strict_hits) > 0:
        why.append("%d hit(s) at or above the strict threshold" % int(strict_hits))
    if p is None and h > 0:
        why.append("no per-image false-positive rate to compare with (fail closed)")
    elif h > 0 and pv < alpha:
        why.append("%d hits of %d images, %.2f expected by chance (P = %.2g < %s)" % (h, n, n * p, pv, alpha))
    return {"images": n, "hits": h, "strict_hits": int(strict_hits), "dhash_hits": int(dhash_hits), "p_false": p,
            "expected_false_hits": None if p is None else round(n * p, 3), "p_value": round(pv, 6),
            "alpha": alpha, "flagged": bool(why), "why": why}


# ------------------------------------------------------------------ validation
def record_problems(cal):
    """Why a v2 calibration record (the "calibration" block) cannot be used:
    the protocol, a threshold that is not a cosine or lies below its base's,
    a strict threshold below it, the dHash radius, gates looser than H6's, a
    family without positives, a family below its recall gate that is not
    listed as a known limit (or one listed that is not below it), a hard tier
    smaller than min_negatives or than MIN_NEGATIVES outside a testing
    record, and a constraining tier above the false-positive gate."""
    from ..funnel import leak as L
    probs = []
    if not isinstance(cal, dict):
        return ["no calibration record"]
    if cal.get("protocol") != PROTOCOL:
        return ["protocol %r is not %s" % (cal.get("protocol"), PROTOCOL)]
    if cal.get("ok") is not True:
        probs.append("ok is not true (%s)" % "; ".join(cal.get("why") or []))
    try:
        t = float(cal.get("cos_threshold"))
        ts = float(cal.get("strict_threshold"))
        floor = float((cal.get("base") or {}).get("cos_threshold"))
    except (TypeError, ValueError):
        return probs + ["cos_threshold, strict_threshold or the base threshold is not a number"]
    if not (math.isfinite(t) and -1.0 <= t <= 1.0):
        probs.append("cos_threshold %r is not a cosine in [-1, 1]" % t)
    if t + 1e-9 < floor:
        probs.append("cos_threshold %s is below its base's %s (never below the floor)" % (t, floor))
    if not (math.isfinite(ts) and ts + 1e-9 >= t):
        probs.append("strict_threshold %r is below cos_threshold %s" % (ts, t))
    if int(cal.get("dhash_bits_max", -1)) != DHASH_BITS:
        probs.append("dhash_bits_max %r is not the never-train radius %d" % (cal.get("dhash_bits_max"), DHASH_BITS))
    if not ((cal.get("base") or {}).get("file") or {}).get("sha256"):
        probs.append("the base calibration is not recorded by sha256")
    try:
        recall_min, fpr_max = float(cal.get("recall_min")), float(cal.get("fpr_max"))
    except (TypeError, ValueError):
        return probs + ["the record names no recall_min / fpr_max gates"]
    if recall_min < L.RECALL_MIN or fpr_max > L.FPR_MAX:
        probs.append("gates recall >= %s, false positives <= %s are looser than H6's (%s, %s)"
                     % (recall_min, fpr_max, L.RECALL_MIN, L.FPR_MAX))
    pos = cal.get("positives") or {}
    below = set()
    for fam in L.FAMILIES:
        p = pos.get(fam) if isinstance(pos, dict) else None
        try:
            n, k = int((p or {}).get("n")), int((p or {}).get("hits"))
        except (TypeError, ValueError):
            probs.append("family %s has no positives record" % fam)
            continue
        if n <= 0:
            probs.append("family %s has no positives" % fam)
        elif k < recall_min * n:
            below.add(fam)
    listed = {str(x.get("family")) for x in (cal.get("known_limits") or []) if isinstance(x, dict)}
    if below - listed:
        probs.append("family(ies) %s are below the recall gate and not recorded as known limits" % sorted(below - listed))
    if listed - below:
        probs.append("known limits %s are not below the recall gate" % sorted(listed - below))
    try:
        min_neg = int(cal.get("min_negatives"))
    except (TypeError, ValueError):
        return probs + ["min_negatives is not recorded"]
    if min_neg < MIN_NEGATIVES and not cal.get("testing"):
        probs.append("min_negatives %d is below %d outside a testing record" % (min_neg, MIN_NEGATIVES))
    neg = cal.get("negatives") or {}
    hard = neg.get("hard") if isinstance(neg, dict) else None
    try:
        n_hard = int((hard or {}).get("n"))
    except (TypeError, ValueError):
        return probs + ["the hard negative tier has no record"]
    if n_hard < max(1, min_neg):
        probs.append("the hard negative tier holds %d images, fewer than %d" % (n_hard, max(1, min_neg)))
    constraining = cal.get("constraining") or []
    if "hard" not in constraining:
        probs.append("the hard tier does not constrain the threshold")
    for tier in constraining:
        v = neg.get(tier) if isinstance(neg, dict) else None
        try:
            n, fh = int((v or {}).get("n")), int((v or {}).get("false_hits"))
        except (TypeError, ValueError):
            probs.append("constraining tier %s has no record" % tier)
            continue
        if n <= 0 or fh > fpr_max * n + 1e-9:
            probs.append("tier %s: %d false hits of %d images (> %s)" % (tier, fh, n, fpr_max))
    return probs


def is_v2(doc_or_cal):
    """True for a v2 document or its calibration block."""
    if not isinstance(doc_or_cal, dict):
        return False
    return doc_or_cal.get("format") == FORMAT or doc_or_cal.get("protocol") == PROTOCOL


def load(path, lock=None, lock_path=None, embedder_name=None):
    """A usable v2 calibration: {"calibration" (the record without its long
    lists), "cos_threshold", "strict_threshold", "dhash_bits_max",
    "embedder", "file", "format", "role", "p_false", "known_limits", "base",
    "protocol"}. With a LOCK (or its path), the file must hash to the sha256
    it records. A record whose own numbers do not show it passed
    (record_problems), without an embedder, or made with another embedder
    than embedder_name refuses (CalibrationError)."""
    path = Path(path)
    if lock_path is not None and lock is None:
        lock = C2.read_lock_v2(lock_path)
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise CalibrationError("v2 calibration %s unreadable (%s)" % (path, e))
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise CalibrationError("%s is not a %s record" % (path, FORMAT))
    sha = C2.sha256_file(path)
    if lock is not None:
        want = lock.get(LOCK_KEY)
        if not want:
            raise CalibrationError("LOCK v2 records no %s" % LOCK_KEY)
        if sha != want:
            raise CalibrationError("%s hashes to %s, LOCK v2 records %s" % (path, sha[:12], str(want)[:12]))
    cal = doc.get("calibration")
    probs = record_problems(cal)
    if probs:
        raise CalibrationError("%s does not show a usable v2 calibration (%s); nothing may be judged by it"
                               % (path, "; ".join(probs[:5])))
    emb = ((doc.get("detector") or {}).get("descriptor") or {}).get("embedder")
    if not emb:
        raise CalibrationError("%s does not name the embedder its threshold belongs to" % path)
    if embedder_name is not None and emb != embedder_name:
        raise CalibrationError("%s was calibrated with %s, the scan would describe images with %s"
                               % (path, emb, embedder_name))
    slim = {k: v for k, v in cal.items() if k not in ("pairs",)}
    hard = (cal.get("negatives") or {}).get("hard") or {}
    return {"calibration": slim, "cos_threshold": float(cal["cos_threshold"]),
            "strict_threshold": float(cal["strict_threshold"]), "dhash_bits_max": int(cal["dhash_bits_max"]),
            "embedder": emb, "file": {"path": str(path), "sha256": sha, "bytes": path.stat().st_size},
            "format": FORMAT, "protocol": PROTOCOL, "role": (cal.get("base") or {}).get("role") or "own",
            "p_false": hard.get("ub"), "known_limits": list(cal.get("known_limits") or []),
            "base": dict(cal.get("base") or {}), "testing": bool(cal.get("testing"))}


def lock_path_of(splits_dir):
    return Path(splits_dir) / "LOCK.json"


def locked(lock_path=None, production=True):
    """(record or None, why): the v2 calibration LOCK v2 records, beside it.
    A LOCK that records none: refused in production (CalibrationError),
    (None, why) for a testing LOCK."""
    lock_path = Path(lock_path or C2.LOCK_PATH)
    lock = C2.read_lock_v2(lock_path)
    if not lock.get(LOCK_KEY):
        if production and not lock.get("testing"):
            raise CalibrationError("LOCK v2 %s records no %s: the v2 embedding calibration cannot be bound "
                                   "(fail closed)" % (lock_path, LOCK_KEY))
        return None, "LOCK v2 records none (testing)"
    return load(lock_path.parent / NAME, lock=lock), None


def state(path, lock_path=None):
    """(the record step1_stream's CopyScanner holds, or None, why): {"path",
    "sha256", "record" (the calibration block funnel.leak.detect reads),
    "threshold", "embedder", "role", "strict_threshold", "p_false"}. With a
    LOCK that exists, the file must hash to what it records."""
    p = Path(path)
    if not p.is_file():
        return None, "absent"
    try:
        lock = None
        if lock_path is not None and Path(lock_path).is_file():
            lock = C2.read_lock_v2(lock_path)
            if not lock.get(LOCK_KEY):
                lock = None if lock.get("testing") else lock
        rec = load(p, lock=lock)
    except (C2.Inc2Error, OSError, ValueError) as e:
        return None, str(e)
    return {"path": str(p), "sha256": rec["file"]["sha256"], "record": rec["calibration"],
            "threshold": rec["cos_threshold"], "embedder": rec["embedder"], "role": rec["role"],
            "strict_threshold": rec["strict_threshold"], "p_false": rec["p_false"], "format": FORMAT}, None


# ------------------------------------------------------------------ descriptors
def _rows_digest(rows):
    body = [[str(r["key"]), str(r["image"]), r.get("sha256") or ""] for r in rows]
    return C2.sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))


def _prepare_name(prepare):
    from ..funnel import embed as E
    return E._prepare_name(prepare)


def _usable_file(path, want_sha, universe, embedder_name, prepare):
    """(content, None) of an embed_images file made by this embedder and
    preparation from rows equal to `universe`'s rows for its keys (the rows
    digest), hashing to want_sha when one is given; else (None, why)."""
    from ..funnel import embed as E
    p = Path(path)
    if not p.is_file():
        return None, "missing"
    if want_sha and C2.sha256_file(p) != want_sha:
        return None, "changed since its calibration recorded it"
    try:
        res = E.load_images(p)
    except Exception as e:  # noqa: BLE001 - an unreadable file is not usable, never a crash
        return None, "unreadable (%s)" % e
    meta = res["meta"]
    if meta.get("embedder") != embedder_name:
        return None, "made by %s" % meta.get("embedder")
    if meta.get("prepare") != _prepare_name(prepare) or int(meta.get("n_hashes") or 0) != 8:
        return None, "another preparation (%s, %s hashes)" % (meta.get("prepare"), meta.get("n_hashes"))
    rows = []
    for k in res["keys"]:
        r = universe.get(k)
        if r is None:
            return None, "holds key %s that its rows do not" % k
        rows.append(r)
    if E.rows_digest(rows) != meta.get("rows_sha256"):
        return None, "made from other rows"
    return res, None


class _Desc:
    """Normalised descriptors, the 8 variant dHashes and a usable mask, by key
    (normalised and masked exactly as funnel.leak's store does, so the base's
    positives give back its threshold)."""

    def __init__(self, keys, X, H, hash_ok, record):
        import numpy as np
        X = np.asarray(X, dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self.H = np.asarray(H, dtype=np.uint64)
        self.ok = np.isfinite(self.Xn).all(axis=1) & np.asarray(hash_ok, dtype=bool)
        self.keys = list(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.record = record


def _descriptors(name, rows, prepare, candidates, own_dir, embedder, procs):
    """_Desc of `rows` (in their order): from the first candidate file
    ((path, sha256 or None, {key: row}, label)) that is usable and holds every
    row, else computed into own_dir/v2cal_<name>.npz (the funnel's files are
    only read). Returns (desc, record)."""
    from ..funnel import embed as E
    want = [str(r["key"]) for r in rows]
    tried = []
    for path, sha, universe, label in candidates:
        if not path:
            continue
        res, why = _usable_file(path, sha, universe, embedder.name, prepare)
        if res is None:
            tried.append({"path": str(path), "why": why})
            continue
        idx = {k: i for i, k in enumerate(res["keys"])}
        if not all(k in idx for k in want):
            tried.append({"path": str(path), "why": "does not hold every row"})
            continue
        sel = [idx[k] for k in want]
        rec = {"path": str(path), "sha256": res["sha256"], "from": label, "rows": len(want)}
        return _Desc(want, res["X"][sel], res["H"][sel], res["hash_ok"][sel], rec), rec
    out = Path(own_dir) / ("v2cal_%s.npz" % name)
    log("describing %d %s image(s) into %s (%s)" % (len(rows), name, out, "; ".join(
        "%s: %s" % (Path(t["path"]).name, t["why"]) for t in tried) or "no recorded file"))
    res = E.embed_images(rows, out, embedder, prepare=prepare, n_hashes=8, procs=procs)
    rec = {"path": str(out), "sha256": res["sha256"], "from": "computed", "rows": len(want), "tried": tried}
    return _Desc(res["keys"], res["X"], res["H"], res["hash_ok"], rec), rec


# ------------------------------------------------------------------ the base
def base_info(path, source, embedder_name):
    """The base calibration: its loaded record (inc2.guard.load_calibration),
    its seeds, families, descriptor files and role ("funnel" for the funnel's
    leak_v1.json, "own" for the stream's)."""
    from . import guard as G
    path = Path(path)
    rec = G.load_calibration(path, embedder_name)
    if rec.get("format") == FORMAT:
        raise CalibrationError("%s is a v2 calibration; the base must be a funnel.leak calibration" % path)
    with open(path) as fh:
        doc = json.load(fh)
    cal = doc.get("calibration") or {}
    role = "funnel" if (source == "funnel_leak_v1" or str(doc.get("format") or "").startswith("funnel-leak")) \
        else "own"
    families = ((doc.get("params") or {}).get("families") or (doc.get("identity") or {}).get("families"))
    return {"path": str(path), "sha256": rec["file"]["sha256"], "format": doc.get("format"), "source": source,
            "role": role, "cos_threshold": float(rec["cos_threshold"]), "recall_min": float(cal.get("recall_min")),
            "fpr_max": float(cal.get("fpr_max")), "seeds": dict(cal.get("seeds") or doc.get("seeds") or {}),
            "families": families, "files": dict(doc.get("descriptor_files") or {}), "embedder": rec["embedder"]}


def _positive_rows(ref_rows, fam, seed_text, params):
    import numpy as np
    from ..funnel import leak as L
    n_pos = min(L.POS_PER_FAMILY, len(ref_rows))
    pick = np.sort(np.random.default_rng(C2.stable_int(seed_text)).choice(len(ref_rows), n_pos, replace=False))
    return [{"key": "%s|%s" % (fam, ref_rows[i]["key"]), "image": ref_rows[i]["image"],
             "sha256": ref_rows[i].get("sha256"),
             "spec": {"family": fam, "seed_text": "%s/%s" % (seed_text, ref_rows[i]["key"]), "params": params}}
            for i in pick], pick


# ------------------------------------------------------------------ calibrate
def _score(q, eval_desc, q_sess, q_date, e_sess, e_date):
    """Per query image: (max cosine over the evaluation images outside its
    session and date, the index of that image, the min over the 8 variants of
    dHash bits against any evaluation image). Sessions and dates are integer
    codes; a negative code is unknown and never matches."""
    import numpy as np
    from ..funnel import leak as L
    n = len(q.keys)
    best = np.full(n, -np.inf, dtype=np.float32)
    arg = np.full(n, -1, dtype=np.int64)
    bits = np.full(n, 65, dtype=np.int64)
    He0 = eval_desc.H[:, 0]
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        S = (q.Xn[s:e] @ eval_desc.Xn.T).astype(np.float32)
        qs, qd = q_sess[s:e, None], q_date[s:e, None]
        mask = ((qs >= 0) & (qs == e_sess[None, :])) | ((qd >= 0) & (qd == e_date[None, :]))
        S = np.where(mask, np.float32(-np.inf), S)
        best[s:e] = S.max(axis=1)
        arg[s:e] = S.argmax(axis=1)
        D = np.full(S.shape, 65, dtype=np.int64)
        for v in range(q.H.shape[1]):
            D = np.minimum(D, L.popcount64(q.H[s:e, v][:, None] ^ He0[None, :]))
        bits[s:e] = D.min(axis=1)
    return best, arg, bits


def _codes(values, table):
    import numpy as np
    out = np.full(len(values), -1, dtype=np.int64)
    for i, v in enumerate(values):
        if v:
            out[i] = table.setdefault(v, len(table))
    return out


def identity_of(base, embedder_name, eval_cache_sha, eval_rows, core_rows, pool_rows, domain_sha, params):
    """What makes two v2 calibrations the same one."""
    from ..funnel import embed as E
    from ..funnel import leak as L
    from . import guard as G
    code = {"inc2/embed_calibration.py": C2.sha256_file(__file__), "inc2/guard.py": C2.sha256_file(G.__file__),
            "funnel/leak.py": C2.sha256_file(L.__file__), "funnel/embed.py": C2.sha256_file(E.__file__)}
    ev = [dict(r, key="%s|%s" % (s, r["key"])) for s in sorted(eval_rows) for r in sorted(eval_rows[s],
                                                                                         key=lambda r: r["key"])]
    return {"format": FORMAT, "protocol": PROTOCOL, "base": {"path": base["path"], "sha256": base["sha256"],
                                                             "source": base["source"]},
            "embedder": embedder_name, "eval_descriptors_sha256": eval_cache_sha,
            "eval_rows": _rows_digest(ev), "core_rows": _rows_digest(sorted(core_rows, key=lambda r: r["key"])),
            "pool_rows": _rows_digest(sorted(pool_rows, key=lambda r: r["key"])), "domain_config_sha256": domain_sha,
            "params": dict(params), "code": code}


def current(path, identity):
    """The existing record at path when it was made with this identity."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("identity") == identity else None


def calibrate(out_path, base, embedder_name, get_embedder, get_eval_index, eval_cache_sha, eval_rows, core_rows,
              pool_rows, domain_raw, own_dir, recall_min, fpr_max, min_negatives=MIN_NEGATIVES,
              easy_sources=EASY_SOURCES, dev_sessions=(), procs=1, testing=False, force=False, inputs=None,
              domain_sha=None, compute=True):
    """Write the v2 calibration (module docstring) to out_path and its
    negatives CSV beside it; return load()'s record. A rerun with the same
    identity is a no-op (a failed one refuses again); with compute False a
    calibration that is not current gives None. A calibration whose record
    fails its own gates is written with ok false and raises.

    base: base_info(); get_embedder() / get_eval_index(): called only when
    something is computed (the index is the funnel.leak EvalIndex the scan
    used, dev + test + imageweeds, keys "split|key", cached in the file whose
    sha256 is eval_cache_sha); eval_rows: {split: v1 rows}; core_rows: v1
    train_core rows; pool_rows: Step 1's pool rows; domain_raw: the funnel
    config (negative groups, not_recoverable, families); dev_sessions: v1's
    dev sessions (test's come from its rows)."""
    import numpy as np
    from ..funnel import leak as L
    out_path = Path(out_path)
    own_dir = Path(own_dir)
    fpr_max, recall_min = float(fpr_max), float(recall_min)
    if fpr_max > L.FPR_MAX or recall_min < L.RECALL_MIN:
        raise CalibrationError("gates recall >= %s, false positives <= %s are looser than H6's" % (recall_min, fpr_max))
    raw = domain_raw or {}
    ref_name = (raw.get("sources") or {}).get("reference")
    neg_groups = sorted({s for p in ((raw.get("leak") or {}).get("negative_source_pairs") or []) for s in p}
                        - {ref_name})
    not_recoverable = (raw.get("sources") or {}).get("not_recoverable") or {}
    easy = [s for s in easy_sources if s in not_recoverable]
    if len(easy) != len(easy_sources):
        raise CalibrationError("easy negative source(s) %s are not named non-plant or leaf-disease sources by the "
                               "funnel config's sources.not_recoverable" % sorted(set(easy_sources) - set(easy)))
    params = {"fpr_max": fpr_max, "recall_min": recall_min, "min_negatives": int(min_negatives),
              "easy_sources": list(easy), "provenance_disjoint_sources": neg_groups, "testing": bool(testing),
              "floor": base["cos_threshold"], "chunk": CHUNK}
    identity = identity_of(base, embedder_name, eval_cache_sha, eval_rows, core_rows, pool_rows, domain_sha, params)
    old = None if force else current(out_path, identity)
    if old is not None:
        if not (old.get("calibration") or {}).get("ok"):
            raise CalibrationError("%s: the v2 calibration failed (%s)" % (out_path, "; ".join(
                (old.get("calibration") or {}).get("why") or [])))
        return load(out_path, embedder_name=embedder_name)
    if not compute:
        return None
    own_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder()
    if embedder.name != embedder_name:
        raise CalibrationError("the embedder is %s, the base calibration's is %s" % (embedder.name, embedder_name))
    eval_index = get_eval_index()
    if (eval_index.record or {}).get("sha256") != eval_cache_sha:
        raise CalibrationError("the evaluation descriptors changed since the scan (%s != %s)"
                               % (str((eval_index.record or {}).get("sha256"))[:12], str(eval_cache_sha)[:12]))
    why, files = [], {}
    # ---- evaluation side
    ev_sess_of = {}
    for s, rows in eval_rows.items():
        for r in rows:
            ev_sess_of["%s|%s" % (s, r["key"])] = str(r.get("session") or "")
    missing = [k for k in eval_index.keys if k not in ev_sess_of]
    if missing:
        raise CalibrationError("the evaluation index holds %d image(s) the evaluation rows do not (e.g. %s)"
                               % (len(missing), missing[:3]))

    class _EvalDesc:
        pass
    E_ = _EvalDesc()
    E_.Xn = np.asarray(eval_index.Xn, dtype=np.float32)
    E_.H = np.asarray(eval_index.H, dtype=np.uint64)
    sess_table, date_table = {}, {}
    e_sess = _codes([ev_sess_of[k] for k in eval_index.keys], sess_table)
    e_date = _codes([capture_date(ev_sess_of[k]) for k in eval_index.keys], date_table)
    eval_sessions = {v for k, v in ev_sess_of.items() if k.split("|", 1)[0] in ("dev", "test") and v}
    eval_sessions |= set(dev_sessions or ())

    # ---- the reference split and the base's positives
    ref_rows = sorted(core_rows, key=lambda r: r["key"])
    ref_universe = {r["key"]: r for r in ref_rows}
    base_ref = base["files"].get("reference") or {}
    ref, rec = _descriptors("reference", ref_rows, L.prepare_hashed,
                            [(base_ref.get("path"), base_ref.get("sha256"), ref_universe, "base"),
                             (own_dir / "emb_dinov2_images_reference.npz", None, ref_universe, "stream")],
                            own_dir, embedder, procs)
    files["reference"] = rec
    fams = base.get("families")
    fam_basis = "the base's recorded families"
    if not isinstance(fams, dict) or sorted(fams) != sorted(L.FAMILIES):
        fams, fam_basis = L.family_params(raw), "the funnel config (the base records none)"
    seeds = base.get("seeds") or {}
    reproduce = all(("positives/%s" % f) in seeds for f in L.FAMILIES)
    from . import guard as G
    seed_prefix = None if reproduce else G.SEED_PREFIX
    pos_cos, pos_bits, pos_failed = {}, {}, {}
    for fam in L.FAMILIES:
        sp = seeds["positives/%s" % fam] if reproduce else "%s/pos/%s" % (seed_prefix, fam)
        rows, pick = _positive_rows(ref_rows, fam, sp, fams[fam])
        universe = {r["key"]: r for r in rows}
        bf = base["files"].get("pos_%s" % fam) or {}
        aug, rec = _descriptors("pos_%s" % fam, rows, L.prepare_augmented,
                                [(bf.get("path"), bf.get("sha256"), universe, "base"),
                                 (own_dir / ("emb_dinov2_images_pos_%s.npz" % fam), None, universe, "stream")],
                                own_dir, embedder, procs)
        files["pos_%s" % fam] = rec
        ri = np.array([ref.index[ref_rows[i]["key"]] for i in pick], dtype=np.int64)
        cos = np.einsum("ij,ij->i", aug.Xn, ref.Xn[ri]).astype(np.float32)
        ok = aug.ok & ref.ok[ri]
        b, _v = L._min_variant_bits(aug.H, ref.H[ri, 0])
        pos_cos[fam] = np.where(ok, cos, np.float32(-np.inf))
        pos_bits[fam] = np.where(ok, b, 65)
        pos_failed[fam] = int((~ok).sum())
    recomputed = L.threshold({f: pos_cos[f].astype(np.float64) for f in L.FAMILIES}, base["recall_min"])
    base_rec = {"source": base["source"], "role": base["role"], "format": base["format"],
                "file": {"path": base["path"], "sha256": base["sha256"]}, "cos_threshold": base["cos_threshold"],
                "recall_min": base["recall_min"], "fpr_max": base["fpr_max"],
                "positives": "reproduced from its seeds" if reproduce else
                             "drawn with this stream's seed prefix %s (the base records no seeds)" % seed_prefix,
                "families_basis": fam_basis,
                "recomputed_threshold": round(float(recomputed), 6) if math.isfinite(recomputed) else None}
    if reproduce and not (math.isfinite(recomputed) and abs(round(float(recomputed), 6) - base["cos_threshold"])
                          <= 1.5e-6):
        raise CalibrationError("the base's positives, rebuilt from its seeds and descriptor files, give threshold "
                               "%s, but %s records %s: they are not the positives it was calibrated with"
                               % (base_rec["recomputed_threshold"], base["path"], base["cos_threshold"]))

    # ---- negative tiers
    tiers, tier_info, csv_rows = {}, {}, []
    hard_rows = [r for r in ref_rows if capture_session(r.get("session"))]
    no_session = len(ref_rows) - len(hard_rows)
    pool_universe = {r["key"]: r for r in pool_rows}
    base_pool = base["files"].get("pool") or {}

    def pool_desc(name, rows):
        if not rows:
            return None, None
        return _descriptors(name, sorted(rows, key=lambda r: r["key"]), L.prepare_hashed,
                            [(base_pool.get("path"), base_pool.get("sha256"), pool_universe, "base"),
                             (own_dir / "emb_dinov2_images_pool.npz", None, pool_universe, "stream")],
                            own_dir, embedder, procs)

    def add_tier(name, desc, rows, sessions, basis, sources):
        if desc is None or not rows:
            tier_info[name] = {"n": 0, "basis": basis, "sources": sources, "excluded_dhash_copies": 0,
                               "unscored": 0}
            tiers[name] = (np.zeros(0, dtype=np.float32), [])
            return
        # a query session or date the evaluation side never holds gets -1, which matches nothing
        q_sess = np.array([sess_table.get(s, -1) if s else -1 for s in sessions], dtype=np.int64)
        q_date = np.array([date_table.get(capture_date(s), -1) if capture_date(s) else -1 for s in sessions],
                          dtype=np.int64)
        best, arg, bits = _score(desc, E_, q_sess, q_date, e_sess, e_date)
        keep = desc.ok & np.isfinite(best) & (bits > DHASH_BITS)
        info = {"basis": basis, "sources": sources, "candidates": len(rows),
                "excluded_dhash_copies": int((desc.ok & (bits <= DHASH_BITS)).sum()),
                "unscored": int((~desc.ok).sum() + (desc.ok & ~np.isfinite(best) & (bits > DHASH_BITS)).sum())}
        kept = []
        for i in np.flatnonzero(keep):
            j = int(arg[i])
            kept.append((rows[i]["key"], str(rows[i].get("source") or ""), float(best[i]), eval_index.split[j],
                         eval_index.eval_key[j], int(bits[i])))
        tiers[name] = (best[keep].astype(np.float32), kept)
        tier_info[name] = info

    add_tier("hard", _descriptors_subset(ref, [r["key"] for r in hard_rows]), hard_rows,
             [capture_session(r.get("session")) for r in hard_rows],
             "train_core images with a capture session, against the evaluation images of other sessions and "
             "other dates", ["train_core"])
    tier_info["hard"]["excluded_no_capture_session"] = no_session
    disj = [r for r in hard_rows if r.get("session") not in eval_sessions]
    add_tier("hard_session_disjoint", _descriptors_subset(ref, [r["key"] for r in disj]), disj,
             [capture_session(r.get("session")) for r in disj],
             "the hard images whose session has no dev or test frame", ["train_core"])
    prov_rows = [r for r in pool_rows if r.get("source") in neg_groups]
    pdesc, rec = pool_desc("provenance_disjoint", prov_rows)
    if rec:
        files["provenance_disjoint"] = rec
    prov_sorted = sorted(prov_rows, key=lambda r: r["key"])
    add_tier("provenance_disjoint", pdesc, prov_sorted, [""] * len(prov_sorted),
             "the funnel config's provenance-disjoint negative groups (leak.negative_source_pairs)", neg_groups)
    easy_rows = [r for r in pool_rows if r.get("source") in easy]
    edesc, rec = pool_desc("easy", easy_rows)
    if rec:
        files["easy"] = rec
    easy_sorted = sorted(easy_rows, key=lambda r: r["key"])
    add_tier("easy", edesc, easy_sorted, [""] * len(easy_sorted),
             "clearly unrelated harvested sources (funnel config sources.not_recoverable: non-plant or "
             "leaf-disease)", list(easy))
    for s in easy:
        if not any(r.get("source") == s for r in pool_rows):
            tier_info["easy"].setdefault("absent_from_pool", []).append(s)

    # ---- threshold
    floor = float(base["cos_threshold"])
    constraining = [t for t in TIERS if t == "hard" or len(tiers[t][0]) >= int(min_negatives)]
    t_raw = {}
    for t in constraining:
        v = threshold_for(tiers[t][0], fpr_max)
        t_raw[t] = v
    finite = [v for v in t_raw.values() if v is not None]
    theta = round(max([floor] + finite), 6)
    all_scores = np.concatenate([tiers[t][0] for t in TIERS if t != "hard_session_disjoint"]) \
        if any(len(tiers[t][0]) for t in TIERS) else np.zeros(0, dtype=np.float32)
    strict = above_all(all_scores, theta)
    negatives = {}
    for t in TIERS:
        sc = tiers[t][0]
        rec_new, rec_old = rate_record(sc, theta), rate_record(sc, floor)
        negatives[t] = dict(tier_info[t], **rec_new)
        negatives[t]["at_base"] = {k: rec_old[k] for k in ("false_hits", "fpr", "ub")}
        negatives[t]["constraining"] = t in constraining
        negatives[t]["tier_threshold"] = t_raw.get(t)
        if len(sc):
            negatives[t]["cos"] = {"max": round(float(sc.max()), 6), "q99": round(float(np.quantile(sc, 0.99)), 6),
                                   "median": round(float(np.median(sc)), 6)}
        for key, src, c, es, ek, b in tiers[t][1]:
            csv_rows.append([t, key, src, "%.6f" % c, es, ek, b])
    n_hard = negatives["hard"]["n"]
    if n_hard < max(1, int(min_negatives)):
        why.append("the hard negative tier holds %d images, fewer than %d" % (n_hard, max(1, int(min_negatives))))
    for t in constraining:
        if negatives[t]["n"] and negatives[t]["false_hits"] > fpr_max * negatives[t]["n"] + 1e-9:
            why.append("tier %s: %d false hits of %d at %s" % (t, negatives[t]["false_hits"], negatives[t]["n"],
                                                              theta))

    # ---- recall per family at the new threshold
    from ..funnel import estimate as ES
    positives, limits = {}, []
    for fam in L.FAMILIES:
        c, b = pos_cos[fam], pos_bits[fam]
        fin = np.isfinite(c)
        hits = (((c >= np.float32(theta)) | (b <= DHASH_BITS)) & fin)
        hits0 = (((c >= np.float32(floor)) | (b <= DHASH_BITS)) & fin)
        n, k = len(c), int(hits.sum())
        lo = ES.clopper_pearson(k, n, UB_CONF)[0] if n else 0.0
        positives[fam] = {"n": n, "hits": k, "recall": round(k / float(n), 6) if n else None, "lb": round(lo, 6),
                          "failed": pos_failed[fam], "dhash_hits": int(((b <= DHASH_BITS) & fin).sum()),
                          "cos_q05": round(float(np.quantile(c[fin], 0.05)), 6) if fin.any() else None,
                          "at_base": {"hits": int(hits0.sum()), "recall": round(int(hits0.sum()) / float(n), 6)
                                      if n else None}}
        if n and k < recall_min * n:
            limits.append({"family": fam, "recall": positives[fam]["recall"], "lb": positives[fam]["lb"], "n": n,
                           "why": "below the %s recall gate at the v2 threshold: recorded as a known limit, not a "
                                  "refusal (L-9(c)); the dHash variants still catch flips, rotations and "
                                  "re-encoding" % recall_min})
    ok = not why
    cal = {"ok": ok, "why": why, "protocol": PROTOCOL, "decided_by": DECISION, "testing": bool(testing),
           "cos_threshold": theta, "strict_threshold": strict, "dhash_bits_max": DHASH_BITS,
           "recall_min": recall_min, "fpr_max": fpr_max, "min_negatives": int(min_negatives),
           "floor": floor, "tier_thresholds": t_raw, "constraining": constraining,
           "threshold_rule": "the smallest 6-decimal cosine with per-image false-positive rate <= fpr_max on every "
                             "constraining tier (hard always; another tier when it holds >= min_negatives images), "
                             "never below the base threshold",
           "base": base_rec, "positives": positives, "known_limits": limits, "negatives": negatives,
           "source_rule": {"rule": SOURCE_RULE, "p_false": negatives["hard"]["ub"], "alpha": SOURCE_ALPHA,
                           "strict_threshold": strict, "dhash_bits_max": DHASH_BITS}}
    neg_path = out_path.parent / NEGATIVES_NAME
    csv_sha = C2.write_csv_atomic(neg_path, NEGATIVES_HEADER, sorted(csv_rows))
    doc = {"format": FORMAT, "built_utc": C2.utc(), "testing": bool(testing), "identity": identity,
           "inputs": dict(inputs or {}), "decided_by": DECISION,
           "detector": {"descriptor": {"embedder": embedder.name}, "dhash_variants": list(L.VARIANTS),
                        "dhash_bits_max": DHASH_BITS, "cos_threshold": theta, "strict_threshold": strict,
                        "rule": RULE, "eval_descriptors": dict(eval_index.record), "eval_images": eval_index.n},
           "calibration": cal, "descriptor_files": files,
           "negatives_csv": {"path": str(neg_path), "sha256": csv_sha, "rows": len(csv_rows)}}
    C2.write_json_atomic(out_path, doc)
    log("v2 threshold %.6f (base %.6f; hard tier %d images: %s false hits at the new threshold, %s at the base); "
        "strict %.6f; known limits %s%s" % (theta, floor, n_hard, negatives["hard"]["false_hits"],
                                             negatives["hard"]["at_base"]["false_hits"], strict,
                                             [x["family"] for x in limits], "" if ok else "; FAILED: %s" % why))
    if not ok:
        raise CalibrationError("the v2 calibration failed (%s); %s written, nothing may be judged by it"
                               % ("; ".join(why), out_path))
    return load(out_path, embedder_name=embedder.name)


def _descriptors_subset(desc, keys):
    """A _Desc of the given keys of desc (in that order), or None for none."""
    if not keys:
        return None
    idx = [desc.index[k] for k in keys]
    out = _Desc.__new__(_Desc)
    out.Xn = desc.Xn[idx]
    out.H = desc.H[idx]
    out.ok = desc.ok[idx]
    out.keys = list(keys)
    out.index = {k: i for i, k in enumerate(out.keys)}
    out.record = desc.record
    return out
