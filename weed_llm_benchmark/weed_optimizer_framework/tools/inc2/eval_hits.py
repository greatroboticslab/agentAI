"""D28-v2: what a dHash hit on an evaluation image says about its SOURCE
(docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-03): D28-v2, cosine-confirmed
source leaks (pre-registered)", amendment V3-1).

Why. Every image within 6 dHash bits of an evaluation image, under any of the
8 flips and rotations, is dropped by GuardV2, and that does not change. Until
this amendment D28 also quarantined the image's whole source on that one hit.
Measured on 2026-10-01 (read-only): the 12 sources quarantined that way were
all dHash false positives. 49 of the 102,920 Step 1 pool images hit the
5,802-image never-train index at 5-6 bits, only under flip or rotation
variants, and their DINOv2 pair cosine with the matched evaluation image is
-0.05..0.66 (median 0.106), 0.744 at most, while true flip and rotation
copies score >= 0.896 (5th percentile; the weakest augmentation family
0.826). The per-image chance rate of the 6-bit 8-variant rule is 4.76e-4
(one-sided 97.5 % upper bound 6.29e-4), so a source of 200,000 images holds
about 95 chance hits.

The rule (fixed in advance):
  1. every dHash hit still drops its own image (GuardV2 unchanged);
  2. a hit is CONFIRMED when its pair cosine (the hit image's and the matched
     evaluation image's descriptors, made the way the copy scan makes them)
     reaches confirm_cos (0.80, stream_thresholds.json D28);
  3. a source leaks when a hit reaches the copy threshold of the v2 embedding
     calibration (its cos_threshold, 0.946384 on the cluster), or when its
     confirmed hits are improbable by chance: P(Binom(images, p_confirmed) >=
     confirmed) < source_alpha (p_confirmed in stream_thresholds.json D28);
  4. fail closed: a source with a hit that has no pair cosine falls back to
     the old rule (one hit is a leak).
Embedding hits keep their own binomial rule (embed_calibration.source_verdict)
and the base-copy share keeps its rule; both live in D28.

The producers record each batch's pair cosines where D28 can read them
(record): collect intake in its summary.json (the hit images exist only in
the intake job, which drops them), step1_stream in batch.json, folded per
source into status.json (its copy scanner holds the evaluation descriptors).

The pair. A hit is weighed against the evaluation image GuardV2 matched it
to AND every other evaluation image within the never-train radius of it
under any of the 8 variants (eval_matches), and its pair cosine is the
highest of those (a choice where the rule is silent: GuardV2 returns the
first match it finds, and an unrelated image there must not hide a copy
within the radius as well). A hit one of whose evaluation images cannot be
described is not weighed (fail closed).

Batches committed before this amendment carry no record. A sidecar weighs
their hits again, deterministically, from the files the batch was judged on
(sidecar, usable_sidecar): step1_stream eval-hits writes
step1_stream/eval_hits/<batch>.json for a Step 1 batch (b0000: the v1 pool
images and dHash cache through GuardV2; a later batch: its ingest.jsonl
rows) and, through collect.intake.rescore_eval_hits, intake/<batch>/
eval_hits.json for an intake batch (each hit image re-derived from the
staging blob by its decision's rel and checked against its sha256). A
sidecar holds the record, the hit keys and the pair cosines, no pixels and
no paths; a Step 1 sidecar, which stays on the cluster, also names each
hit's match (evaluation split and key), an intake sidecar, which the
snapshot ships, does not (its decisions.jsonl does). A batch file is never
rewritten (b0000's batch.json is hash-locked by the stream ledger).

Standard library, inc2.common and inc2.guard (no numpy) at module level;
numpy, funnel.embed, funnel.leak and funnel.estimate are imported where
used. This module is not part of the v2 calibration's identity
(embed_calibration.identity_of hashes embed_calibration.py and guard.py), so
a change here never makes the locked calibration stale.
"""
from __future__ import annotations

from . import common as C2
from . import guard as G

DHASH_HIT_REASONS = ("near_eval_v2", "near_eval_variant")    # GuardV2's dHash checks of an evaluation image
FORMAT = "inc2-eval-hit-cosines/1"
SIDECAR_FORMAT = "inc2-eval-hit-sidecar/1"
DECISION = "D28-v2, amendment V3-1, docs/CONTINUOUS_LOOP.md (pre-registered 2026-10-03)"
DECIMALS = 6
DHASH_BITS = C2.HOLDOUT_NEAR_DUP_BITS                        # the never-train radius (inc2.guard.EVAL_BITS)
RULE = ("pair_cos: the highest cosine of the hit image's descriptor with those of the evaluation image GuardV2 "
        "matched and of every other evaluation image within the never-train radius under any of the 8 variants (the "
        "copy scan's embedder and preparation); a hit is confirmed at pair_cos >= confirm_cos; a source leaks on a "
        "hit at or above the v2 calibration's cos_threshold, or when P(Binom(images, p_confirmed) >= confirmed) < "
        "source_alpha; a hit without a pair cosine is a leak (fail closed)")


def eval_matches(guard, variants):
    """Every evaluation image within the never-train radius of an image under
    any of its 8 variants: [{"split", "eval_key", "bits"}], nearest first
    (ties by split and key), or None when they cannot be listed (no variants,
    or a guard without a never-train index). guard: a GuardV2, or
    step1_stream's Guards around one. GuardV2 keeps that index as _eval (a
    near_dup.NearHashIndex, whose matches() lists every stored hash within
    range); inc2/guard.py is part of the v2 calibration's identity, so it is
    read here rather than given an accessor there."""
    idx = getattr(getattr(guard, "guard", guard), "_eval", None)
    vs = G.variant_list(variants)
    if idx is None or not hasattr(idx, "matches") or vs is None:
        return None
    best = {}
    for _name, h in vs:
        for owner, bits in idx.matches(int(h)):
            if isinstance(owner, (tuple, list)) and len(owner) == 2:
                ek = (str(owner[0]), str(owner[1]))
                best[ek] = min(best.get(ek, bits), bits)
    return [{"split": s, "eval_key": k, "bits": int(b)} for (s, k), b in sorted(best.items(),
                                                                              key=lambda t: (t[1], t[0]))]


def pair_cosines(hits, embedder, eval_desc=None, procs=1, batch=32):
    """The pair cosine of each dHash hit: {key: {"pair_cos": float or None,
    "why": None or why not, "weighed": evaluation images weighed, "best":
    [split, key] of the highest or None}}.

    hits: [{"key", "image", "split", "eval_key", "eval_image"?, "also"?}], the
    hit image and the evaluation image GuardV2 matched it to (split, key);
    "also" ([{"split", "eval_key", "eval_image"?}], eval_matches) the other
    evaluation images within the never-train radius, the guard's match among
    them or not. The pair cosine is the highest over all of them.
    eval_desc(split, eval_key) gives an evaluation image's descriptor as the
    copy scan already holds it (an EvalIndex row), or None; an evaluation
    image it does not hold is described here from its "eval_image" when
    given. Images are described the way the copy scan describes them
    (funnel.leak.prepare_hashed through funnel.embed.embed_images, with the
    calibration's embedder), so a pair cosine is on the scale of the
    calibration's threshold. A hit that cannot be scored (the hit image or
    one of its evaluation images that cannot be described, an evaluation
    image neither held nor given, any failure of the embedder) gets None and
    the reason: its source then falls back to the one-hit rule (fail closed),
    so this never raises."""
    hits = [dict(h) for h in hits or ()]
    out = {str(h["key"]): {"pair_cos": None, "why": None, "weighed": 0, "best": None} for h in hits}
    if not hits:
        return out
    try:
        import numpy as np
        from ..funnel import embed as E
        from ..funnel import leak as L
    except Exception as e:  # noqa: BLE001 - no descriptor stack: every hit stays unscored (fail closed)
        for v in out.values():
            v["why"] = "the descriptor stack does not import (%s: %s)" % (type(e).__name__, e)
        return out

    def unit(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(x)) if x.size and np.isfinite(x).all() else 0.0
        return x / n if n > 0 else None

    def refs(h):
        """[(split, key, image or None)] of a hit: the guard's match first, then the others, once each."""
        got, seen = [], set()
        for s, k, img in [(h.get("split"), h.get("eval_key"), h.get("eval_image"))] + [
                (a.get("split"), a.get("eval_key"), a.get("eval_image")) for a in h.get("also") or ()
                if isinstance(a, dict)]:
            ek = (str(s or ""), str(k or ""))
            if ek not in seen:
                seen.add(ek)
                got.append((ek[0], ek[1], img))
        return got

    held, need = {}, {}
    for h in hits:
        for s, k, img in refs(h):
            ek = (s, k)
            if ek in held or ek in need:
                continue
            vec = None
            if eval_desc is not None:
                try:
                    vec = eval_desc(s, k)
                except Exception:  # noqa: BLE001 - a lookup that fails is a lookup that misses
                    vec = None
            if vec is not None:
                held[ek] = unit(vec)
            elif img:
                need[ek] = str(img)
    ekeys = sorted(need)
    rows = [{"key": "hit|%d" % i, "image": str(h["image"])} for i, h in enumerate(hits)]
    rows += [{"key": "eval|%d" % j, "image": need[ek]} for j, ek in enumerate(ekeys)]
    try:
        res = E.embed_images(rows, None, embedder, prepare=L.prepare_hashed, n_hashes=len(L.VARIANTS), procs=procs,
                             batch=batch)
        X = np.asarray(res["X"], dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - the embedder failed: nothing is scored (fail closed)
        for v in out.values():
            v["why"] = "the images could not be described (%s: %s)" % (type(e).__name__, str(e)[:200])
        return out
    for j, ek in enumerate(ekeys):
        held[ek] = unit(X[len(hits) + j])
    for i, h in enumerate(hits):
        k, rs = str(h["key"]), refs(h)
        q = unit(X[i])
        if q is None:
            out[k]["why"] = "the hit image cannot be described"
            continue
        best, why = None, None
        for n, (s, e, _img) in enumerate(rs):
            ek = (s, e)
            what = "the matched evaluation image" if n == 0 else "an evaluation image within %d dHash bits" % DHASH_BITS
            if ek not in held:
                why = "%s is neither in the copy scan's descriptors nor given" % what
                break
            if held[ek] is None:
                why = "%s cannot be described" % what
                break
            c = round(float(np.dot(q, held[ek])), DECIMALS)
            if best is None or c > best[0]:
                best = (c, [s, e])
        if why is not None:
            out[k]["why"] = why
        else:
            out[k].update(pair_cos=best[0], best=best[1], weighed=len(rs))
    return out


def record(hits, scored, embedder_name=None, copy_threshold=None, calibration=None, why=None):
    """The per-batch record D28 reads (collect intake's summary.json
    "eval_hits", step1_stream's batch.json "eval_hits"): {"format",
    "decided_by", "rule", "embedder", "copy_threshold" (the v2 calibration's
    cos_threshold as the producer read it through inc2.embed_calibration),
    "calibration", "hits", "scored", "why", "per_source": {source: {"hits",
    "scored", "pair_cos" (descending), "max_pair_cos"}}}. hits: [{"key",
    "source"}]; scored: pair_cosines' result, or {} when nothing could be
    scored (why then says why). Evaluation keys are not recorded here (the
    batch's own files name them)."""
    per, whys = {}, {}
    for h in hits or ():
        v = per.setdefault(str(h.get("source") or ""), {"hits": 0, "scored": 0, "pair_cos": []})
        v["hits"] += 1
        got = (scored or {}).get(str(h["key"])) or {}
        if got.get("pair_cos") is not None:
            v["scored"] += 1
            v["pair_cos"].append(float(got["pair_cos"]))
        elif got.get("why"):
            whys[got["why"]] = whys.get(got["why"], 0) + 1
    for v in per.values():
        v["pair_cos"] = sorted(v["pair_cos"], reverse=True)
        v["max_pair_cos"] = v["pair_cos"][0] if v["pair_cos"] else None
    reasons = ([str(why)] if why else []) + ["%s (%d)" % (w, n) for w, n in sorted(whys.items())]
    return {"format": FORMAT, "decided_by": DECISION, "rule": RULE, "embedder": embedder_name,
            "copy_threshold": None if copy_threshold is None else float(copy_threshold),
            "calibration": calibration, "hits": sum(v["hits"] for v in per.values()),
            "scored": sum(v["scored"] for v in per.values()), "why": "; ".join(reasons) or None,
            "per_source": dict(sorted(per.items()))}


def verdict(images, hits, pair_cos, confirm_cos, copy_cos, p_confirmed, alpha):
    """Whether a source's dHash hits say it holds copies (the module's rule,
    fixed in advance). images: the images checked; hits: its dHash hits;
    pair_cos: the pair cosines of its scored hits, None without a record. A
    source with more hits than pair cosines falls back to the one-hit rule
    (fail closed). copy_cos None (no calibration threshold known): confirm_cos
    stands in, which is never less strict."""
    from ..funnel import estimate as ES
    n, h = int(images), int(hits)
    confirm_cos, p_c = float(confirm_cos), float(p_confirmed)
    copy_cos = confirm_cos if copy_cos is None else float(copy_cos)
    cos = None if pair_cos is None else sorted((float(c) for c in pair_cos if c is not None), reverse=True)
    out = {"images": n, "hits": h, "scored": None if cos is None else len(cos), "confirmed": None,
           "copy_hits": None, "max_pair_cos": cos[0] if cos else None, "confirm_cos": confirm_cos,
           "copy_cos": copy_cos, "p_confirmed": p_c, "expected_confirmed": round(n * p_c, 3), "p_value": 1.0,
           "alpha": alpha, "fail_closed": False, "flagged": False, "verdict": "no dHash hit", "why": []}
    if h <= 0:
        return out
    k = sum(1 for c in cos or () if c >= confirm_cos)
    kc = sum(1 for c in cos or () if c >= copy_cos)
    if cos is None or len(cos) < h:
        # the weighed hits' confirmed and copy counts are stated, but decide nothing
        missing = h if cos is None else h - len(cos)
        why = ["%d of %d hit(s) within %d dHash bits without a pair cosine: one hit is a leak (fail closed)"
               % (missing, h, DHASH_BITS)]
        if kc > 0:
            why.append("%d weighed hit(s) at or above the copy threshold %s (pair cos max %.3f)" % (kc, copy_cos,
                                                                                                cos[0]))
        out.update(fail_closed=True, flagged=True, verdict="leak (fail closed)", confirmed=k, copy_hits=kc, why=why)
        return out
    pv = 1.0 if k <= 0 else float(ES.binom_upper_tail(k, n, p_c))
    why = []
    if kc > 0:
        why.append("%d hit(s) at or above the copy threshold %s (pair cos max %.3f)" % (kc, copy_cos, cos[0]))
    if k > 0 and pv < alpha:
        why.append("%d confirmed hit(s) (pair cos >= %s) in %d images, %.3g expected by chance (P = %.2g < %s)"
                   % (k, confirm_cos, n, n * p_c, pv, alpha))
    out.update(confirmed=k, copy_hits=kc, p_value=round(pv, 6), flagged=bool(why),
               verdict="leak" if why else "chance", why=why)
    return out


def describe(v):
    """A verdict in one line, as D28 states it for each source: hits,
    confirmed hits (on a fail-closed row, among the weighed ones), max pair
    cosine, P and the verdict."""
    if v.get("fail_closed"):
        return "%d dHash hit(s) in %d images, %s with a pair cosine, %d confirmed among them (pair cos >= %g), " \
               "max pair cos %s, P n/a -> %s" % (
                   v["hits"], v["images"], "none" if not v.get("scored") else "only %d" % v["scored"],
                   v.get("confirmed") or 0, v["confirm_cos"],
                   "n/a" if v["max_pair_cos"] is None else "%.3f" % v["max_pair_cos"], v["verdict"])
    return "%d dHash hit(s) in %d images, %d confirmed (pair cos >= %g), max pair cos %s, P = %.3g -> %s" % (
        v["hits"], v["images"], v["confirmed"] or 0, v["confirm_cos"],
        "n/a" if v["max_pair_cos"] is None else "%.3f" % v["max_pair_cos"], v["p_value"], v["verdict"])


# ------------------------------------------------------------------ sidecars
def sidecar_pairs(items, scored, eval_keys=True):
    """The per-hit rows a sidecar keeps, in hit order: {"key", "source",
    "reason", "pair_cos", "pair_cos_why", "weighed" (evaluation images
    weighed)} and, with eval_keys, "match" (the guard's: split, key, bits,
    variant) and "best" ([split, key] of the highest). items: [{"key",
    "source", "reason", "match"}]; scored: pair_cosines' result. No pixels and
    no paths; evaluation keys only where asked: a sidecar the snapshot ships
    (an intake batch's, whose decisions.jsonl names each match) leaves them
    out, as the records the autopilot reads do."""
    out = []
    for it in items or ():
        got = (scored or {}).get(str(it["key"])) or {}
        m = it.get("match") if isinstance(it.get("match"), dict) else {}
        row = {"key": str(it["key"]), "source": it.get("source"), "reason": it.get("reason"),
               "pair_cos": got.get("pair_cos"), "pair_cos_why": None if got.get("pair_cos") is not None
               else (got.get("why") or None), "weighed": int(got.get("weighed") or 0)}
        if eval_keys:
            row.update(match={k: m.get(k) for k in ("split", "key", "bits", "variant") if k in m},
                       best=got.get("best"))
        out.append(row)
    return out


def sidecar(batch, rec, pairs, kind, **extra):
    """A sidecar document: {"format", "decided_by", "kind" (step1 or intake),
    "batch", "eval_hits" (record), "pairs" (sidecar_pairs), ...extra}; extra
    names what it was derived from (sha256 of the batch files) and when."""
    doc = {"format": SIDECAR_FORMAT, "decided_by": DECISION, "kind": kind, "batch": str(batch), "eval_hits": rec,
           "pairs": list(pairs or ())}
    doc.update(extra)
    return doc


def usable_sidecar(doc, batch, counts):
    """(the sidecar's eval_hits record, None) when doc is a sidecar of this
    batch that weighed exactly the dHash hits the batch counted per source
    (counts: {source: hits}), else (None, why). A sidecar that names another
    batch, or holds another number of hits for any source, is not used: its
    hits are then read by the one-hit rule (fail closed)."""
    if not isinstance(doc, dict) or doc.get("format") != SIDECAR_FORMAT:
        return None, "not an %s document" % SIDECAR_FORMAT
    if str(doc.get("batch")) != str(batch):
        return None, "it is the sidecar of batch %r, not %r" % (doc.get("batch"), batch)
    rec = doc.get("eval_hits") if isinstance(doc.get("eval_hits"), dict) else {}
    per = rec.get("per_source") if isinstance(rec.get("per_source"), dict) else None
    if per is None:
        return None, "it holds no per-source record"
    want = {str(s): int(n) for s, n in (counts or {}).items() if int(n or 0) > 0}
    got = {str(s): int((v or {}).get("hits") or 0) for s, v in per.items() if int((v or {}).get("hits") or 0) > 0}
    if want != got:
        return None, "it weighed %s dHash hit(s) per source, the batch counted %s" % (got, want)
    return rec, None
