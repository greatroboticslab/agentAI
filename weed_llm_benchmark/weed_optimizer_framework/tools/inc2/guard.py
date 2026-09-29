"""GuardV2: the one copy guard every v2 entry point calls (docs/CONTINUOUS_LOOP.md
§3.2 "Guard", §3.3 stage 4, §4.2, §8 "Never-train v2").

The checks, in order; the first hit wins and is counted (self.counts):

  1 unhashable         no dHash, or not all eight variant dHashes, or the
                       variants' "id" differs from the dHash (the two were
                       computed from different pixels): an image the guard
                       cannot compare is an image it cannot clear
  2 near_eval_v2       the dHash within 6 bits of an evaluation image
  3 near_eval_variant  one of the eight flips and rotations within 6 bits of one
  4 base_copy          the dHash or one of its variants within 6 bits of a
                       base v2 image
  5 exact_dup          the dHash equals one in the seen index (add_seen)
  6 near_dup_intake    within 3 bits of an earlier intake batch image (add_intake)
  7 near_eval_embed    the calibrated embedding copy detector (EmbedScanner,
                       check_embed): run where descriptors are computed anyway
                       (the Step 1 stream job, the pre-lock scan of splits v2)

The dHash is inc.common.dhash (mega_trainer._dhash, the stored pixels, no EXIF
orientation); the eight variants are funnel.leak.dhash_variants of the same
stored pixels, whose "id" equals that dHash. The evaluation and base indexes
come from the files LOCK v2 records, each checked against its sha256 there.

The embedding detector is funnel/leak.py used as a library (§3.2 [review]):
its calibration (positives: augmented reference images per family;
negatives: 7-10-bit and hardest pairs between provenance-disjoint groups; the
pre-registered recall and false-positive gates) runs under a directory the
caller owns, with its own seed prefix, and never writes the funnel's
directory. A passed funnel leak_v1.json may be reused instead (its threshold,
recorded by path and sha256).

Standard library and near_dup at module level; PIL, numpy and funnel.leak are
imported where used.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from . import common as C2
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NEAR_DUP_BITS, NearHashIndex

REASONS = ("unhashable", "near_eval_v2", "near_eval_variant", "base_copy", "exact_dup", "near_dup_intake",
           "near_eval_embed")
# funnel.leak.VARIANTS, restated so this module loads without numpy (a test checks they agree)
VARIANTS = ("id", "hflip", "vflip", "rot90", "rot180", "rot270", "transpose", "transverse")
EVAL_BITS = HOLDOUT_NEAR_DUP_BITS              # 6: the never-train radius
BASE_BITS = HOLDOUT_NEAR_DUP_BITS              # 6: base copies (§4.2 step 7)
INTAKE_BITS = NEAR_DUP_BITS                    # 3: near-duplicates of earlier intake batches
SEED_PREFIX = "stream/v1/leak"                 # §3.2 [review]: the stream's own calibration seeds
CAL_FORMAT = "inc2-leak-calibration/1"
CAL_NAME = "leak_calibration.json"
CAL_PAIRS = "leak_calibration_pairs.csv"
NEVERTRAIN_NAME = "nevertrain_dhash.json"
BASE_COPIES_NAME = "base_copies_dhash.json"


class GuardError(C2.Inc2Error):
    """The guard cannot be built or used as the LOCK describes (fail closed)."""


# ------------------------------------------------------------------ hashing
def dhash_variants(path):
    """{variant: dHash} of the eight flips and rotations of the image at path,
    as stored (no EXIF orientation), from funnel.leak.dhash_variants; None when
    the image cannot be read."""
    try:
        from PIL import Image
        from ..funnel import leak as L
        with Image.open(path) as im:
            im.load()
            return {k: int(v) for k, v in L.dhash_variants(im).items()}
    except Exception:  # noqa: BLE001 - an unreadable image is "unhashable", never a crash
        return None


def image_hashes(path):
    """(dHash, variants) of the image at path: inc.common.dhash and
    dhash_variants. Either may be None (unreadable)."""
    h = C2.dhash(path)
    return (None if h is None else int(h)), dhash_variants(path)


def variant_list(variants):
    """[(name, int)] in VARIANTS order from a {name: hash} dict or an
    8-sequence in VARIANTS order; None when any of the eight is missing."""
    if variants is None:
        return None
    if isinstance(variants, dict):
        if any(variants.get(v) is None for v in VARIANTS):
            return None
        vals = [variants[v] for v in VARIANTS]
    else:
        vals = list(variants)
        if len(vals) != len(VARIANTS) or any(v is None for v in vals):
            return None
    try:
        out = [(n, int(v)) for n, v in zip(VARIANTS, vals)]
    except (TypeError, ValueError):
        return None
    if any(not 0 <= v < 2 ** 64 for _n, v in out):
        return None
    return out


# -------------------------------------------------------------------- guard
class GuardV2:
    """The never-train v2 guard plus the base-copy, seen and intake indexes."""

    def __init__(self, eval_entries, base_entries=(), record=None):
        self._eval = NearHashIndex()
        self._base = NearHashIndex()
        self._seen = {}
        self._intake = NearHashIndex()
        self.n_eval = self.n_base = self.n_seen = self.n_intake = 0
        for h, split, key in eval_entries:
            self._eval.add(int(h), (str(split), str(key)), max_bits=EVAL_BITS)
            self.n_eval += 1
        for h, part, key in base_entries:
            self._base.add(int(h), (str(part), str(key)), max_bits=BASE_BITS)
            self.n_base += 1
        self.counts = collections.Counter()
        self.record = dict(record or {})

    # --- loading
    @classmethod
    def load(cls, lock_path=None):
        """The guard LOCK v2 describes: its never-train index and base-copy
        index (in the LOCK's directory), each hashing to the LOCK's record,
        each marked complete. The LOCK must list exactly the v2 evaluation
        splits, and the never-train index must hold exactly the images of
        their manifests beside the LOCK (each manifest hashing to the LOCK's
        record): an index that misses an evaluation split, or some of its
        images, would clear copies of them. Anything else refuses."""
        lock_path = Path(lock_path or C2.LOCK_PATH)
        lock = C2.read_lock_v2(lock_path)
        d = lock_path.parent
        nt = _read_index(d / NEVERTRAIN_NAME, lock.get("nevertrain_sha256"), "never-train")
        bc = _read_index(d / BASE_COPIES_NAME, lock.get("base_copies_sha256"), "base-copy")
        evs = set(lock.get("eval_splits") or ())
        if not evs:
            raise GuardError("%s lists no eval_splits" % lock_path)
        bad = sorted({e[1] for e in nt["entries"]} - evs)
        if bad:
            raise GuardError("the never-train index holds split(s) %s that LOCK v2 does not list as evaluation "
                             "splits %s" % (bad, sorted(evs)))
        if evs != set(C2.EVAL_SPLITS):
            raise GuardError("%s lists evaluation splits %s; the v2 never-train guard covers exactly %s"
                             % (lock_path, sorted(evs), sorted(C2.EVAL_SPLITS)))
        held = {(str(e[1]), str(e[2])) for e in nt["entries"]}
        want = {(s, str(r["key"])) for s, rows in v2_eval_rows(lock_path).items() for r in rows}
        if held != want:
            raise GuardError("the never-train index does not hold exactly the evaluation manifests' images "
                             "(%d missing, e.g. %s; %d extra)" % (len(want - held), sorted(want - held)[:3],
                                                                 len(held - want)))
        for name, data, want in (("never-train", nt, lock.get("nevertrain_entries")),
                                 ("base-copy", bc, lock.get("base_copies_entries"))):
            if want is not None and len(data["entries"]) != int(want):
                raise GuardError("the %s index holds %d entries, LOCK v2 says %s" % (name, len(data["entries"]), want))
        rec = {"lock": C2.file_record(lock_path), "nevertrain_sha256": lock["nevertrain_sha256"],
               "base_copies_sha256": lock["base_copies_sha256"], "eval_splits": sorted(evs),
               "testing": bool(lock.get("testing"))}
        return cls(nt["entries"], bc["entries"], record=rec)

    # --- the state other groups add
    def add_seen(self, dhash, owner):
        """Step 1's seen index: an image whose dHash equals this one is exact_dup."""
        h = int(dhash)
        if h not in self._seen:
            self._seen[h] = owner
            self.n_seen += 1

    def add_intake(self, dhash, owner):
        """An image of an earlier intake batch (near_dup_intake at 3 bits)."""
        self._intake.add(int(dhash), owner, max_bits=INTAKE_BITS)
        self.n_intake += 1

    def index_record(self):
        """What guard.json records: the index shas and sizes."""
        return dict(self.record, eval_entries=self.n_eval, base_entries=self.n_base, seen=self.n_seen,
                    intake=self.n_intake, eval_bits=EVAL_BITS, base_bits=BASE_BITS, intake_bits=INTAKE_BITS)

    # --- the check
    def _decide(self, dhash, variants):
        if dhash is None:
            return "unhashable", {"why": "no dHash"}
        try:
            h = int(dhash)
        except (TypeError, ValueError):
            return "unhashable", {"why": "dHash %r is not an integer" % (dhash,)}
        vs = variant_list(variants)
        if vs is None:
            return "unhashable", {"why": "the eight variant dHashes are required"}
        if vs[0][1] != h:
            return "unhashable", {"why": "the variants' id hash differs from the dHash (different pixels)"}
        m = self._eval.find(h)
        if m is not None:
            (split, key), bits = m
            return "near_eval_v2", {"split": split, "key": key, "bits": bits, "variant": "id"}
        best = _best(self._eval, vs[1:])
        if best is not None:
            (split, key), bits, var = best
            return "near_eval_variant", {"split": split, "key": key, "bits": bits, "variant": var}
        best = _best(self._base, vs)
        if best is not None:
            (part, key), bits, var = best
            return "base_copy", {"part": part, "key": key, "bits": bits, "variant": var}
        if h in self._seen:
            return "exact_dup", {"owner": self._seen[h], "bits": 0}
        m = self._intake.find(h)
        if m is not None:
            return "near_dup_intake", {"owner": m[0], "bits": m[1]}
        return None, None

    def check(self, dhash, variants=None):
        """(reason, match) of the first check that refuses, or (None, None)."""
        reason, match = self._decide(dhash, variants)
        self.counts[reason or "pass"] += 1
        return reason, match

    def check_path(self, path):
        """(reason, match, (dhash, variants)) of the image at path."""
        h, v = image_hashes(path)
        reason, match = self.check(h, v)
        return reason, match, (h, v)

    def check_embed(self, rows, scanner, desc_path=None, procs=1):
        """The seventh check over rows ([{"key", "image", "sha256"?}]):
        {key: (reason, match)} for every row the embedding detector refuses:
        near_eval_embed (a copy, the best by cosine, with the number found), or
        unhashable when the row cannot be described (fail closed). A refusal
        is counted under its reason, a cleared row as embed_pass (the same row's
        check() already counted it once)."""
        copies, unscannable = scanner.scan(rows, desc_path=desc_path, procs=procs)
        out = {}
        for r in rows:
            k = r["key"]
            if k in unscannable:
                out[k] = ("unhashable", {"stage": "embed", "why": "the image cannot be described"})
            elif copies.get(k):
                best = max(copies[k], key=lambda e: (e["cos"], -e["bits"]))
                out[k] = ("near_eval_embed", dict(best, n_copies=len(copies[k])))
            self.counts[out[k][0] if k in out else "embed_pass"] += 1
        return out


def _best(index, vs):
    """((owner), bits, variant name) of the nearest hit over the variants, the
    first variant (in VARIANTS order) winning a tie; None without a hit."""
    best = None
    for name, h in vs:
        m = index.find(h)
        if m is not None and (best is None or m[1] < best[1]):
            best = (m[0], m[1], name)
    return best


def _read_index(path, want_sha, what):
    if not want_sha:
        raise GuardError("LOCK v2 records no sha256 for the %s index" % what)
    if not Path(path).is_file():
        raise GuardError("the %s index %s is missing" % (what, path))
    got = C2.sha256_file(path)
    if got != want_sha:
        raise GuardError("the %s index %s changed since LOCK v2 (%s != %s)" % (what, path, got[:12], want_sha[:12]))
    with open(path) as fh:
        data = json.load(fh)
    if data.get("complete") is not True:
        raise GuardError("the %s index %s is not complete (lock marks it complete)" % (what, path))
    if data.get("bits") != HOLDOUT_NEAR_DUP_BITS:
        raise GuardError("the %s index %s has bits %s, not %d" % (what, path, data.get("bits"), HOLDOUT_NEAR_DUP_BITS))
    if len(data.get("entries") or ()) < int(data.get("min_expected", 0)):
        raise GuardError("the %s index %s holds %d entries, expected >= %s"
                         % (what, path, len(data.get("entries") or ()), data.get("min_expected")))
    return data


def v2_eval_rows(lock_path=None):
    """{split: manifest rows} of the evaluation splits LOCK v2 lists, read from
    the manifests beside it, each checked against the LOCK's sha256."""
    lock_path = Path(lock_path or C2.LOCK_PATH)
    lock = C2.read_lock_v2(lock_path)
    out = {}
    for split in lock.get("eval_splits") or ():
        p = lock_path.parent / ("%s.jsonl" % split)
        want = (lock.get("manifests") or {}).get(split)
        if not want or not p.is_file() or C2.sha256_file(p) != want:
            raise GuardError("evaluation manifest %s does not match LOCK v2" % p)
        out[split] = C2.read_manifest(p)
    return out


# ------------------------------------------------ the embedding copy detector
def _rows_digest(rows):
    body = [[str(r["key"]), str(r["image"]), r.get("sha256") or ""] for r in rows]
    return C2.sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))


def _code():
    from ..funnel import embed as E
    from ..funnel import leak as L
    out = {"funnel/%s" % Path(m.__file__).name: C2.sha256_file(m.__file__) for m in (L, E)}
    out["inc2/guard.py"] = C2.sha256_file(__file__)
    return out


def calibration_problems(cal):
    """Why a calibration record ("ok": true is not taken on trust) cannot
    clear anything: a threshold that is not a cosine in [-1, 1] (above 1 the
    embedding half never fires), a dHash radius other than the never-train
    radius, gates looser than the pre-registered H6 ones (recall 0.95 per
    family, false positives 0.01), a family of funnel.leak.FAMILIES without
    positives or below its recall gate, and an empty or failing negative set."""
    import math
    from ..funnel import leak as L
    probs = []
    try:
        theta = float(cal.get("cos_threshold"))
    except (TypeError, ValueError):
        return ["cos_threshold %r is not a number" % (cal.get("cos_threshold"),)]
    if not (math.isfinite(theta) and -1.0 <= theta <= 1.0):
        probs.append("cos_threshold %r is not a cosine in [-1, 1]" % theta)
    if int(cal.get("dhash_bits_max", EVAL_BITS)) != EVAL_BITS:
        probs.append("dhash_bits_max %r is not the never-train radius %d" % (cal.get("dhash_bits_max"), EVAL_BITS))
    try:
        recall_min, fpr_max = float(cal.get("recall_min")), float(cal.get("fpr_max"))
    except (TypeError, ValueError):
        return probs + ["the record names no recall_min / fpr_max gates"]
    if recall_min < L.RECALL_MIN or fpr_max > L.FPR_MAX:
        probs.append("gates recall >= %s, false positives <= %s are looser than H6's (%s, %s)"
                     % (recall_min, fpr_max, L.RECALL_MIN, L.FPR_MAX))
    pos = cal.get("positives") or {}
    for fam in L.FAMILIES:
        p = pos.get(fam) if isinstance(pos, dict) else None
        try:
            n, k = int((p or {}).get("n")), int((p or {}).get("hits"))
        except (TypeError, ValueError):
            probs.append("family %s has no positives record" % fam)
            continue
        if n <= 0 or k < recall_min * n:
            probs.append("family %s: %d of %d positives found (< %s)" % (fam, k, n, recall_min))
    neg = cal.get("negatives") or {}
    for kind in ("pairs_7_10", "hard"):
        v = neg.get(kind) if isinstance(neg, dict) else None
        try:
            n, fh = int((v or {}).get("n")), int((v or {}).get("false_hits"))
        except (TypeError, ValueError):
            probs.append("negative set %s has no record" % kind)
            continue
        if n <= 0 or fh > fpr_max * n:
            probs.append("negative set %s: %d false hits of %d (> %s)" % (kind, fh, n, fpr_max))
    return probs


def load_calibration(path, embedder_name=None):
    """A passed calibration: this module's file (CAL_FORMAT) or the funnel's
    leak_v1.json. Returns {"calibration", "cos_threshold", "dhash_bits_max",
    "embedder", "file", "format"}. A failed calibration, one whose record does
    not show it passed (calibration_problems), one without a threshold or
    embedder, or one made with another embedder refuses."""
    path = Path(path)
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise GuardError("calibration %s unreadable (%s)" % (path, e))
    cal = doc.get("calibration") if isinstance(doc, dict) else None
    if not isinstance(cal, dict) or cal.get("ok") is not True or cal.get("cos_threshold") is None:
        raise GuardError("%s holds no passed calibration; nothing may be judged by it (%s)"
                         % (path, "; ".join((cal or {}).get("why") or []) or "no calibration record"))
    probs = calibration_problems(cal)
    if probs:
        raise GuardError("%s says ok, but its record does not show a passed calibration (%s); nothing may be "
                         "judged by it" % (path, "; ".join(probs[:5])))
    emb = ((doc.get("detector") or {}).get("descriptor") or {}).get("embedder")
    if not emb:
        raise GuardError("%s does not name the embedder its threshold belongs to" % path)
    if embedder_name is not None and emb != embedder_name:
        raise GuardError("%s was calibrated with %s, the scan would describe images with %s"
                         % (path, emb, embedder_name))
    cal = {k: v for k, v in cal.items() if k != "pairs"}
    return {"calibration": cal, "cos_threshold": float(cal["cos_threshold"]),
            "dhash_bits_max": int(cal.get("dhash_bits_max", EVAL_BITS)), "embedder": emb,
            "file": C2.file_record(path), "format": doc.get("format")}


def calibrate(out_dir, domain_raw, embedder, ref_rows, pool_rows, recall_min, fpr_max, seed_prefix=SEED_PREFIX,
              procs=1, inputs=None, testing=False, force=False):
    """funnel.leak.calibrate under out_dir (its descriptor files, CAL_NAME and
    CAL_PAIRS). domain_raw supplies leak.families, leak.negative_source_pairs
    and sources.reference; ref_rows are the reference split's rows and
    pool_rows the rows of the negative groups' other sources. A rerun with the
    same identity is a no-op (a failed one refuses again). Returns
    load_calibration's record; a failed calibration raises GuardError after
    writing its record with ok false."""
    from ..funnel import embed as E
    from ..funnel import leak as L
    out_dir = Path(out_dir)
    path = out_dir / CAL_NAME
    ref_rows = sorted(ref_rows, key=lambda r: r["key"])
    pool_rows = sorted(pool_rows, key=lambda r: r["key"])
    identity = {"embedder": embedder.name, "seed_prefix": seed_prefix, "recall_min": float(recall_min),
                "fpr_max": float(fpr_max), "families": L.family_params(domain_raw),
                "negative_source_pairs": [list(p) for p in L._negative_pairs_cfg(domain_raw)],
                "reference": {"n": len(ref_rows), "sha256": _rows_digest(ref_rows)},
                "pool": {"n": len(pool_rows), "sha256": _rows_digest(pool_rows)},
                "view": E.VIEW, "code": _code()}
    if path.is_file() and not force:
        with open(path) as fh:
            old = json.load(fh)
        if old.get("identity") == identity:
            if not (old.get("calibration") or {}).get("ok"):
                raise GuardError("%s: the calibration failed (%s)" % (path, "; ".join(old["calibration"].get("why", []))))
            return load_calibration(path, embedder.name)
    store = L._Store(embedder, out_dir, procs=procs, force=force)
    cal = L.calibrate(None, domain_raw, embedder, seed_prefix, store=store, recall_min=recall_min,
                      fpr_max=fpr_max, ref_rows=ref_rows, pool_rows=pool_rows, procs=procs)
    pairs = cal.pop("pairs")
    rows = [[p["set"], p["key"], p["eval_split"], p["eval_key"], "%.6f" % p["cos"], p["bits"], p["variant"],
             p["kind"]] for p in pairs]
    pairs_sha = C2.write_csv_atomic(out_dir / CAL_PAIRS, L.PAIRS_HEADER, rows)
    doc = {"format": CAL_FORMAT, "built_utc": C2.utc(), "testing": bool(testing), "identity": identity,
           "inputs": dict(inputs or {}), "seeds": dict(cal.get("seeds") or {}),
           "detector": {"descriptor": {"embedder": embedder.name, "view": E.VIEW},
                        "dhash_variants": list(L.VARIANTS), "dhash_bits_max": L.DHASH_BITS_MAX,
                        "cos_threshold": cal["cos_threshold"], "rule": L.RULE},
           "calibration": cal, "descriptor_files": dict(sorted(store.files.items())),
           "pairs_csv": {"path": str(out_dir / CAL_PAIRS), "sha256": pairs_sha, "rows": len(rows)}}
    C2.write_json_atomic(path, doc)
    if not cal["ok"]:
        raise GuardError("the copy detector failed its calibration (%s); %s written, nothing may be judged"
                         % ("; ".join(cal["why"]), path))
    return load_calibration(path, embedder.name)


class _EvalRows:
    """The one adapter method funnel.leak.eval_index reads."""

    def __init__(self, rows_by_split):
        self._rows = rows_by_split

    def eval_rows(self):
        return {s: [dict(r) for r in rows] for s, rows in self._rows.items()}


def eval_index(eval_rows, embedder, cache_path, procs=1, force=False):
    """funnel.leak's EvalIndex over {split: rows}, its descriptors cached in
    cache_path (an existing cache made from other rows or another embedder
    refuses unless force)."""
    from ..funnel import leak as L
    return L.eval_index(_EvalRows(eval_rows), embedder, cache_path, procs=procs, force=force)


class EmbedScanner:
    """near_eval_embed: a calibrated threshold plus an evaluation index made by
    the same embedder."""

    def __init__(self, index, calibration):
        emb = getattr(index, "embedder", None)
        if emb is None:
            raise GuardError("the evaluation index carries no embedder; new images cannot be described")
        if emb.name != calibration["embedder"]:
            raise GuardError("the evaluation index describes images with %s, the calibration's embedder is %s"
                             % (emb.name, calibration["embedder"]))
        self.index = index
        self.calibration = calibration
        self.threshold = calibration["cos_threshold"]

    def record(self):
        return {"embedder": self.calibration["embedder"], "cos_threshold": self.threshold,
                "dhash_bits_max": self.calibration["dhash_bits_max"], "calibration": self.calibration["file"],
                "calibration_format": self.calibration["format"], "eval_descriptors": dict(self.index.record),
                "eval_images": self.index.n}

    def scan(self, rows, desc_path=None, procs=1, batch=32):
        """({key: [copy entries]}, set of keys that cannot be described) for
        rows [{"key", "image", "sha256"?}] (keys unique). Descriptors are
        cached in desc_path when given. Entries carry evaluation keys only."""
        import numpy as np
        from ..funnel import embed as E
        from ..funnel import leak as L
        rows = [{"key": str(r["key"]), "image": str(r["image"]), "sha256": r.get("sha256")} for r in rows]
        if not rows:
            return {}, set()
        res = E.embed_images(rows, desc_path, self.index.embedder, prepare=L.prepare_hashed,
                             n_hashes=len(L.VARIANTS), procs=procs, batch=batch)
        X = np.asarray(res["X"], dtype=np.float32)
        H = np.asarray(res["H"], dtype=np.uint64)
        ok = np.isfinite(X).all(axis=1) & np.asarray(res["hash_ok"], dtype=bool) & (np.linalg.norm(X, axis=1) > 0)
        images = [{"key": r["key"], "path": r["image"], "desc": X[i], "hashes": [int(h) for h in H[i]]}
                  for i, r in enumerate(rows) if ok[i]]
        unscannable = {r["key"] for i, r in enumerate(rows) if not ok[i]}
        found = L.detect(images, self.index, self.calibration["calibration"]) if images else []
        copies = {}
        for e in found:
            copies.setdefault(e["key"], []).append(e)
        return copies, unscannable
