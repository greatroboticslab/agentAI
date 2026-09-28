"""The seeded stratified draw and the sample lock (contract §5.2, F6; runner
§4.7-4.9, §5.1.5).

draw() builds the frames (strata.frames), refuses when a group's frame is
below its pre-registered minimum, draws every group by its design, writes
frames_v1.json and frames_v1/<group>.csv, sample_v1.csv and
sample_v1_key.jsonl, and appends the "sample lock" amendment to the prereg
before any reference label exists. After the lock, draw, strata and census
refuse to rewrite their outputs (SampleLocked); a rerun with the same inputs
is a no-op.

Designs (runner §4.8):
  * srs: within each stratum, the first n of a permutation of the stratum's
    units (sorted by unit id) drawn with seed funnel/v1/<stratum id>;
    pi = n_h / N_h.
  * joint (G2v with G2a): two independent srs draws; a unit in both frames is
    listed once, pi = 1 - (1 - pi_G2v)(1 - pi_G2a), and counts for both.
  * two_stage (G3): units by srs (seed funnel/v1/G3/units) unless all fit,
    then crops by srs inside each unit; pi = (m/M)(n_u/N_u).
  * dual (G4): uniform srs per name frame plus Poisson sampling over the whole
    group with p_i = min(1, n_prio s_i / sum s); pi = 1 - (1 - u_h)(1 - p_i).
  * planted (G0): the planted share s = 0.2 + 0.6 u, positives half from the
    in-domain set and half from the independent set, negatives from the
    independent set's attractors; share, parts and truth go to the key only.
  * sentinels: 3 per 12 pool items (a sheet holds 15 items, 3 of them
    sentinels), split over known-truth sets by the configured weights and
    within a set over truth kinds by availability; pair sentinels likewise
    per 12 G5 items.
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

from . import (DrawError, SampleLocked, StaleInput, check_records, file_record, header,
               read_csv, read_jsonl, rng, write_csv_atomic, write_jsonl_atomic)
from ..inc import common as C
from . import domain as D
from . import strata as S

SAMPLE_FIELDS = ("item_id", "unit_id", "unit", "group", "stratum", "source", "image_key", "crop_id",
                 "pi", "pi_parts", "seed_text", "draw_rank", "sheet_class", "lab", "near_dup3",
                 "provenance", "kt")
KEY_FIELDS = ("item_id", "unit_id", "truth", "truth_taxon", "truth_kind", "pair_truth")
EVAL_GROUPS = ("G5", "pair_sentinel")
TRUTH_GROUPS = ("G0", "sentinel", "identity", "pair_sentinel")
PAIR_NEGATIVE_PREFIX = "neg"


def item_id(unit_id, group):
    """The opaque id a labeller sees (runner §1.3)."""
    return C.sha256_text("funnel/v1/item/%s/%s" % (unit_id, group))[:16]


def _fmt(x):
    return repr(float(x))


def _json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def srs_order(unit_ids, text):
    """The unit ids (sorted) in the order of a permutation seeded by text."""
    ids = sorted(unit_ids)
    perm = rng(text).permutation(len(ids))
    return [ids[i] for i in perm]


def poisson_draw(units_scores, n_target, text):
    """Poisson sampling: unit i is drawn when u_i < p_i, p_i = min(1, n s_i / S),
    with u_i from one generator seeded by text over the units sorted by id.
    Returns ({unit: p_i}, [drawn units in id order])."""
    ids = sorted(units_scores)
    S_ = float(sum(units_scores[i] for i in ids))
    p = {}
    for i in ids:
        s = float(units_scores[i])
        if s < 0:
            raise DrawError("a Poisson score is negative (%s)" % i)
        p[i] = min(1.0, n_target * s / S_) if S_ > 0 else 0.0
    u = rng(text).random(len(ids))
    drawn = [i for i, ui in zip(ids, u) if ui < p[i]]
    return p, drawn


def inclusion(design, **kw):
    """Inclusion probability of one unit under a design (runner §4.8)."""
    if design == "srs":
        return kw["n"] / float(kw["N"]) if kw["N"] else 0.0
    if design == "two_stage":
        return (kw["m"] / float(kw["M"])) * (kw["n_u"] / float(kw["N_u"]))
    if design == "dual":
        return 1.0 - (1.0 - kw["u"]) * (1.0 - kw["p"])
    if design == "joint":
        q = 1.0
        for p in kw["parts"]:
            q *= (1.0 - p)
        return 1.0 - q
    if design == "poisson":
        return min(1.0, kw["n"] * kw["s"] / float(kw["S"])) if kw["S"] else 0.0
    raise DrawError("unknown design %r" % design)


def _pick(unit, group, stratum, pi, parts, seed_texts, rank, sheet_class="pool"):
    return {"unit": unit, "group": group, "stratum": stratum, "pi": pi, "pi_parts": parts,
            "seed_text": ";".join(seed_texts), "draw_rank": rank, "sheet_class": sheet_class}


def _by_stratum(units):
    out = {}
    for u in units:
        out.setdefault(u["stratum"], []).append(u)
    return out


def draw_srs_group(group, grp, prefix=S.SEED_PREFIX, sheet_class="pool"):
    picks = []
    byu = _by_stratum(grp["units"])
    for sid in sorted(grp["strata"]):
        st = grp["strata"][sid]
        units = {u["unit_id"]: u for u in byu.get(sid, [])}
        n, N = int(st["n_planned"]), len(units)
        if n > N:
            raise DrawError("%s: planned %d > frame %d" % (sid, n, N))
        text = S.seed_text(sid, prefix)
        for rank, uid in enumerate(srs_order(units, text)[:n]):
            picks.append(_pick(units[uid], group, sid, inclusion("srs", n=n, N=N), {group: n / float(N)},
                               [text], rank, sheet_class))
    return picks


def draw_joint(groups, names=("G2v", "G2a"), prefix=S.SEED_PREFIX):
    """Two independent srs draws; a unit in both frames is listed once with the
    union inclusion probability (under its first drawing group)."""
    pi_of, drawn = {}, {}
    for g in names:
        grp = groups[g]
        byu = _by_stratum(grp["units"])
        for sid, st in grp["strata"].items():
            N = len(byu.get(sid, []))
            n = int(st["n_planned"])
            for u in byu.get(sid, []):
                pi_of.setdefault(u["unit_id"], {})[g] = n / float(N) if N else 0.0
        for p in draw_srs_group(g, grp, prefix):
            drawn.setdefault(p["unit"]["unit_id"], []).append(p)
    picks = []
    for uid in sorted(drawn):
        first = drawn[uid][0]
        parts = {g: pi_of[uid].get(g, 0.0) for g in names}
        first["pi"] = inclusion("joint", parts=list(parts.values()))
        first["pi_parts"] = parts
        first["drawn_by"] = [p["group"] for p in drawn[uid]]
        picks.append(first)
    return picks


def draw_g3(grp, prefix=S.SEED_PREFIX):
    info = grp["info"]
    units = sorted(grp["strata"])
    M = len(units)
    if M == 0:
        return []
    if info.get("take_all_units", True):
        chosen, m = units, M
    else:
        m = int(info["m"])
        chosen = srs_order(units, "%s/G3/units" % prefix)[:m]
    byu = _by_stratum(grp["units"])
    picks = []
    for sid in sorted(chosen):
        st = grp["strata"][sid]
        members = {u["unit_id"]: u for u in byu.get(sid, [])}
        N_u, n_u = len(members), int(st["n_planned"])
        text = S.seed_text(sid, prefix)
        pi = inclusion("two_stage", m=m, M=M, n_u=n_u, N_u=N_u)
        for rank, uid in enumerate(srs_order(members, text)[:n_u]):
            picks.append(_pick(members[uid], "G3", sid, pi, {"stage1": m / float(M), "stage2": n_u / float(N_u)},
                               [text], rank))
    return picks


def draw_g4(grp, prefix=S.SEED_PREFIX):
    info = grp["info"]
    by_frame = {}
    for u in grp["units"]:
        by_frame.setdefault(S.unit_extra(u)["frame"], {})[u["unit_id"]] = u
    uni_pi, uni_rank, uni_seed = {}, {}, {}
    for f, spec in sorted(info.get("uniform", {}).items()):
        units = by_frame.get(f, {})
        n, N = int(spec["n"]), len(units)
        text = "%s/G4/uniform/frame=%s" % (prefix, f)
        for rank, uid in enumerate(srs_order(units, text)[:n]):
            uni_rank[uid], uni_seed[uid] = rank, text
        for uid in units:
            uni_pi[uid] = n / float(N) if N else 0.0
    prio = info.get("prio", {})
    text_p = "%s/G4/prio" % prefix
    scores = {u["unit_id"]: float(u["score"]) for u in grp["units"]}
    p_i, drawn_p = poisson_draw(scores, int(prio.get("n", 0)), text_p) if scores else ({}, [])
    prio_rank = {uid: i for i, uid in enumerate(drawn_p)}
    all_units = {u["unit_id"]: u for u in grp["units"]}
    picks = []
    for uid in sorted(set(uni_rank) | set(prio_rank)):
        u = all_units[uid]
        uu, pp = uni_pi.get(uid, 0.0), p_i.get(uid, 0.0)
        seeds = ([uni_seed[uid]] if uid in uni_rank else []) + ([text_p] if uid in prio_rank else [])
        rank = uni_rank[uid] if uid in uni_rank else prio_rank[uid]
        picks.append(_pick(u, "G4", u["stratum"], inclusion("dual", u=uu, p=pp),
                           {"uniform": uu, "prio": pp}, seeds, rank))
    return picks


def draw_g0(grp, planned, prefix=S.SEED_PREFIX):
    """The planted control. Returns (picks, planted record)."""
    parts = {}
    for u in grp["units"]:
        parts.setdefault(S.unit_extra(u)["part"], {})[u["unit_id"]] = u
    n_g0 = min(int(planned), sum(len(v) for v in parts.values()))
    share_seed = "%s/G0/share" % prefix
    u = C.stable_int(share_seed) / float(2 ** 31 - 1)
    s = 0.2 + 0.6 * u
    n_pos = int(round(s * n_g0))
    avail = {k: len(v) for k, v in parts.items()}
    want_ind = n_pos // 2
    want_dom = n_pos - want_ind
    n_dom = min(want_dom, avail.get("pos_in_domain", 0))
    n_ind = min(want_ind + (want_dom - n_dom), avail.get("pos_independent", 0))
    n_dom = min(n_pos - n_ind, avail.get("pos_in_domain", 0))
    n_neg = min(n_g0 - (n_dom + n_ind), avail.get("neg_independent", 0))
    seeds = {"pos_in_domain": "%s/G0/pos/%s" % (prefix, S.G0_IN_DOMAIN_SET.lower()),
             "pos_independent": "%s/G0/pos/%s" % (prefix, str(grp["info"].get("independent_set", "")).lower()),
             "neg_independent": "%s/G0/neg" % prefix}
    counts = {"pos_in_domain": n_dom, "pos_independent": n_ind, "neg_independent": n_neg}
    chosen = []
    for part in sorted(counts):
        units = parts.get(part, {})
        n = counts[part]
        for rank, uid in enumerate(srs_order(units, seeds[part])[:n]):
            chosen.append((uid, part, rank, n / float(len(units)) if units else 0.0))
    order = srs_order([c[0] for c in chosen], "%s/G0/order" % prefix)
    pos = {uid: i for i, uid in enumerate(order)}
    picks = []
    for uid, part, rank, pi in chosen:
        unit = parts[part][uid]
        p = _pick(unit, "G0", "G0/all", pi, {}, [seeds[part]], pos[uid])
        p["g0"] = {"part": part, "seed_text": seeds[part], "part_rank": rank, "pi": pi}
        picks.append(p)
    n_pos_real = n_dom + n_ind
    tot = n_pos_real + n_neg
    planted = {"share": (n_pos_real / float(tot)) if tot else None, "share_drawn": s,
               "n_pos": n_pos_real, "n_neg": n_neg, "seed_text": share_seed, "parts": counts,
               "available": avail, "part_seeds": seeds}
    return picks, planted


def _sentinel_frames(fr, drawn_ids, domain):
    """Sentinel units per known-truth set: every item of a weighted set except
    exemplar items, items of exemplar sessions and units already drawn; for
    the independent set, its sentinel-role photos only."""
    weights = (domain.section("sampling", required=False) or {}).get("sentinel_weights") or {}
    known = fr["known_truth"]
    ex_sess = {int(k): set(v) for k, v in (fr.get("exemplar_sessions_by_class") or {}).items()}
    indep = set(domain.independent_sets())
    grp = S._group("sentinel", 0, None)
    for kt in sorted(weights):
        for it in sorted(known.get(kt, []), key=lambda x: x["id"]):
            if it["id"] in drawn_ids or it.get("role") == "exemplar":
                continue
            if kt in indep and it.get("role") != "sentinel":
                continue
            if S.in_exemplar_session(it, ex_sess):
                continue
            tk = it.get("truth_kind") or "unknown"
            grp["units"].append(S._kt_unit(it, "sentinel", S.stratum_id("sentinel", kt=kt, truth_kind=tk)))
    S._finish(grp, lambda sid: "sentinel %s" % sid)
    return grp, weights


def _plan_sentinels(grp, weights, total):
    avail = {}
    for sid, st in grp["strata"].items():
        kt = S.parse_stratum(sid)[1]["kt"]
        avail[kt] = avail.get(kt, 0) + st["N"]
    per_kt = S.allocate_weighted({k: float(v) for k, v in weights.items()}, avail, total)
    for kt, n in per_kt.items():
        sids = {sid: st["N"] for sid, st in grp["strata"].items() if S.parse_stratum(sid)[1]["kt"] == kt}
        for sid, nn in S.allocate_weighted(sids, sids, n).items():
            grp["strata"][sid]["n_planned"] = nn
    grp["info"]["per_kt"] = per_kt
    grp["info"]["total"] = total


def _pair_sentinel_frame(fr, domain, leak_pairs_path):
    grp = S._group("pair_sentinel", 0, None)
    ref = domain.reference_source
    seen = set()
    for it in sorted(fr["known_truth"].get(S.PAIR_POSITIVE_SET, []), key=lambda x: x["id"]):
        prov = it.get("provenance") or ""
        if not prov.startswith("prov:"):
            continue
        key = it["id"].split(":", 1)[1].split("#", 1)[0]
        uid = "p:%s|%s|%s|%s" % (it.get("source"), key, ref, prov[len("prov:"):])
        if uid in seen:
            continue
        seen.add(uid)
        u = S._unit(uid, "pair", S.stratum_id("pair_sentinel", kind="positive"), source=it.get("source") or "",
                    image_key=key, lab=it.get("lab") or "", near_dup3=it.get("near_dup3") or "",
                    provenance=prov, extra={"pair_truth": "same", "kt": S.PAIR_POSITIVE_SET})
        grp["units"].append(u)
    if leak_pairs_path is not None and Path(leak_pairs_path).exists():
        _, rows = read_csv(leak_pairs_path)
        for r in rows:
            if not str(r.get("kind", "")).startswith(PAIR_NEGATIVE_PREFIX):
                continue
            uid = "p:%s|%s|%s|%s" % (r["set"], r["key"], r["eval_split"], r["eval_key"])
            if uid in seen:
                continue
            seen.add(uid)
            grp["units"].append(S._unit(uid, "pair", S.stratum_id("pair_sentinel", kind="negative"),
                                        source=r["set"], image_key=r["key"], lab=domain.lab_of(r["set"]),
                                        extra={"pair_truth": "different", "bits": r.get("bits"),
                                               "other": [r["eval_split"], r["eval_key"]]}))
    S._finish(grp, lambda sid: "pair sentinel %s" % sid)
    return grp


def _plan_pair_sentinels(grp, domain, total):
    cfg = (domain.section("sampling", required=False) or {}).get("pair_sentinels") or {}
    sizes = {}
    for sid, st in grp["strata"].items():
        sizes[S.parse_stratum(sid)[1]["kind"]] = (sid, st["N"])
    weights = {k: float(cfg.get(k, 0)) for k in ("positive", "negative")}
    caps = {k: sizes[k][1] if k in sizes else 0 for k in weights}
    per = S.allocate_weighted(weights, caps, total)
    for k, n in per.items():
        if k in sizes:
            grp["strata"][sizes[k][0]]["n_planned"] = n
    grp["info"]["per_kind"] = per
    grp["info"]["total"] = total


def _sheet_numbers(domain):
    sheet = (domain.section("reference_labeller", required=False) or {}).get("sheet") or {}
    if "items" not in sheet or "sentinels" not in sheet:
        raise DrawError("reference_labeller.sheet (items, sentinels) is not set in the domain config")
    return int(sheet["items"]), int(sheet["sentinels"])


def sentinel_count(n_items, domain):
    items, sent = _sheet_numbers(domain)
    per = items - sent
    return sent * int(math.ceil(n_items / float(per))) if n_items > 0 else 0


def draw_all(fr, prereg, domain, prefix=S.SEED_PREFIX, leak_pairs_path=None):
    """Every group's draw from in-memory frames. Returns (picks, planted, sentinel frames)."""
    groups = fr["groups"]
    picks = []
    g0_picks, planted = draw_g0(groups["G0"], groups["G0"]["planned"], prefix)
    picks += g0_picks
    picks += draw_srs_group("G1", groups["G1"], prefix)
    picks += draw_srs_group("G2", groups["G2"], prefix)
    picks += draw_joint(groups, ("G2v", "G2a"), prefix)
    picks += draw_g3(groups["G3"], prefix)
    picks += draw_srs_group("identity", groups["identity"], prefix)
    picks += draw_g4(groups["G4"], prefix)
    g5 = draw_srs_group("G5", groups["G5"], prefix, sheet_class="eval")
    picks += g5
    n_pool = len(picks) - len(g5)
    drawn_ids = {p["unit"]["unit_id"] for p in picks}
    sgrp, weights = _sentinel_frames(fr, drawn_ids, domain)
    _plan_sentinels(sgrp, weights, sentinel_count(n_pool, domain))
    groups["sentinel"] = sgrp
    picks += draw_srs_group("sentinel", sgrp, prefix)
    pgrp = _pair_sentinel_frame(fr, domain, leak_pairs_path)
    _plan_pair_sentinels(pgrp, domain, sentinel_count(len(g5), domain))
    if len(g5) and sum(st["n_planned"] for st in pgrp["strata"].values()) == 0:
        raise DrawError("G5 has %d items but no pair sentinel can be drawn (copy twins or "
                        "leak_pairs_v1.csv negatives missing)" % len(g5))
    groups["pair_sentinel"] = pgrp
    picks += draw_srs_group("pair_sentinel", pgrp, prefix, sheet_class="eval")
    return picks, planted


def _kt_of(unit):
    ex = S.unit_extra(unit)
    kt = ex.get("kt")
    if isinstance(kt, list):
        return ";".join(kt)
    return kt or ""


def build_rows(picks, fr, domain):
    """sample_v1.csv rows and sample_v1_key.jsonl rows from the picks."""
    kt_items = {}
    for kt, items in fr["known_truth"].items():
        for it in items:
            kt_items.setdefault(it["id"], it)
    rows, keys = [], []
    for p in picks:
        u, g = p["unit"], p["group"]
        iid = item_id(u["unit_id"], g)
        # truth goes to the key for the planted, sentinel and identity items
        # only: an estimation item's claimed label is what the audit tests
        it = kt_items.get(u["unit_id"]) if g in TRUTH_GROUPS else None
        ex = S.unit_extra(u)
        truth = it.get("truth") if it else None
        key = {"item_id": iid, "unit_id": u["unit_id"], "truth": truth,
               "truth_taxon": it.get("truth_taxon") if it else None,
               "truth_kind": it.get("truth_kind") if it else None,
               "pair_truth": ex.get("pair_truth")}
        if g == "G0":
            key["g0"] = p["g0"]
            rows.append({"item_id": iid, "unit_id": "G0:%s" % iid, "unit": "", "group": "G0",
                         "stratum": "G0/all", "source": "", "image_key": "", "crop_id": "", "pi": "",
                         "pi_parts": "{}", "seed_text": "%s/G0" % S.SEED_PREFIX, "draw_rank": p["draw_rank"],
                         "sheet_class": "pool", "lab": "", "near_dup3": "", "provenance": "", "kt": ""})
        else:
            rows.append({"item_id": iid, "unit_id": u["unit_id"], "unit": u["unit"], "group": g,
                         "stratum": p["stratum"], "source": u["source"], "image_key": u["image_key"],
                         "crop_id": u["crop_id"], "pi": _fmt(p["pi"]),
                         "pi_parts": _json({k: float(v) for k, v in p["pi_parts"].items()}),
                         "seed_text": p["seed_text"], "draw_rank": p["draw_rank"],
                         "sheet_class": p["sheet_class"], "lab": u["lab"], "near_dup3": u["near_dup3"],
                         "provenance": u["provenance"], "kt": _kt_of(u)})
        keys.append(key)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["group"], rows[i]["stratum"],
                                                    int(rows[i]["draw_rank"]), rows[i]["item_id"]))
    rows = [rows[i] for i in order]
    keys = sorted(keys, key=lambda k: k["item_id"])
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise DrawError("two sample rows share an item id")
    return rows, keys


def dinov2_loader(adapter, funnel_dir):
    """crop ids -> DINOv2 crop features (float64), read on first use from
    funnel_dir/emb_dinov2 through embed.load against the adapter's crop table;
    record() names the shards read (an input of the frames)."""
    cache = {}

    def features(crop_ids):
        import numpy as np
        if "X" not in cache:
            from . import embed as EM
            X, info = EM.load(adapter.crop_table(), Path(funnel_dir) / "emb_dinov2")
            cache["X"], cache["info"] = X, info
        return np.asarray(cache["X"][np.asarray(list(crop_ids), dtype=np.int64)], dtype=np.float64)

    def record():
        info = cache.get("info")
        if not info:
            return None
        return {"path": "", "sha256": info.get("sha256"), "files": info.get("files")}
    features.record = record
    return features


def planned_sample(group, grp, prefix=S.SEED_PREFIX):
    """The number of items a group's design draws from its frame, checked
    against the pre-registered minimum before anything is drawn: the sum of
    the strata's planned n (srs, allocation, identity); for the planted
    control min(planned, N); for the two-stage design the planned n of the
    units its seeded first stage picks; for the dual design the expected size
    of the union of the uniform and Poisson parts (sum of the union
    inclusion probabilities). A frame at or above the minimum can still yield
    fewer items (a two-stage group of few units draws at most 30 from each)."""
    design = grp["design"]
    if design == "planted":
        return min(int(grp["planned"]), int(grp["N"]))
    if design == "two_stage":
        units = sorted(grp["strata"])
        if not units:
            return 0
        info = grp["info"]
        chosen = units if info.get("take_all_units", True) else \
            srs_order(units, "%s/G3/units" % prefix)[:int(info["m"])]
        return sum(int(grp["strata"][s]["n_planned"]) for s in chosen)
    if design == "dual":
        info = grp["info"]
        by_frame = {}
        for u in grp["units"]:
            by_frame.setdefault(S.unit_extra(u)["frame"], []).append(u)
        n_prio = int((info.get("prio") or {}).get("n", 0))
        tot_s = float(sum(float(u["score"]) for u in grp["units"]))
        expected = 0.0
        for f, us in sorted(by_frame.items()):
            spec = (info.get("uniform") or {}).get(f)
            uh = int(spec["n"]) / float(len(us)) if spec and us else 0.0
            for u in us:
                p = min(1.0, n_prio * float(u["score"]) / tot_s) if tot_s > 0 else 0.0
                expected += 1.0 - (1.0 - uh) * (1.0 - p)
        return expected
    return sum(int(st["n_planned"]) for st in grp["strata"].values())


def _today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _paths(out_dir):
    out_dir = Path(out_dir)
    return {"frames": out_dir / "frames_v1.json", "sample": out_dir / "sample_v1.csv",
            "key": out_dir / "sample_v1_key.jsonl"}


def _existing_matches_lock(paths, lock):
    for name, want in (("sample", lock.get("sample_sha256")), ("key", lock.get("key_sha256")),
                       ("frames", lock.get("frames_sha256"))):
        p = paths[name]
        if not p.exists() or file_record(p)["sha256"] != want:
            return False
    return True


def draw(prereg_path, out_dir, adapter, force=False, domain=None, judge_qual=None, dinov2=None,
         allowed_fn=None, testing=False):
    """F6: frames, the draw, the sample files and the sample lock. Returns a
    summary record. A rerun after the lock is a no-op when every recorded input
    is unchanged, SampleLocked otherwise (force is refused after the lock)."""
    prereg_path = Path(prereg_path)
    out_dir = Path(out_dir)
    pre = D.load_prereg(prereg_path)
    dom = D.load(domain) if domain is not None else D.load(pre.domain_name)
    D.check_prereg_domain(pre, dom)
    paths = _paths(out_dir)
    lock = pre.sample_lock
    if lock is not None:
        if lock.get("prereg_core_sha256") != pre.core_sha256:
            raise SampleLocked("the sample is locked (%s) under prereg core %s, but the prereg's core is now %s: "
                               "it was edited outside its amendments" % (lock.get("id"),
                                                                         str(lock.get("prereg_core_sha256"))[:12],
                                                                         pre.core_sha256[:12]))
        if not _existing_matches_lock(paths, lock):
            raise SampleLocked("the sample is locked (%s) and its files differ from the lock; draw refuses "
                               "to rewrite them%s" % (lock.get("id"), " (force is refused after the lock)"
                                                       if force else ""))
        from . import read_json
        doc = read_json(paths["frames"])
        try:
            check_records(doc.get("inputs") or {})
        except StaleInput as e:
            raise SampleLocked("the sample is locked (%s) and an input changed: %s" % (lock.get("id"), e))
        if adapter is not None:
            digest = S.known_truth_digest(adapter.known_truth(dom, out_dir))
            if digest != doc.get("known_truth_digest"):
                raise SampleLocked("the sample is locked (%s) and the known truth changed" % lock.get("id"))
        return {"status": "no-op", "lock": lock, "sample": file_record(paths["sample"])}
    if judge_qual is None and (out_dir / "judge_qualification.json").exists():
        from . import read_json
        judge_qual = read_json(out_dir / "judge_qualification.json")
    if dinov2 is None:
        dinov2 = dinov2_loader(adapter, out_dir)
    fr = S.frames(pre, dom, out_dir, adapter, judge_qual=judge_qual, dinov2=dinov2, allowed_fn=allowed_fn)
    if paths["frames"].exists() and not force:
        from . import read_json
        old = read_json(paths["frames"])
        old_in = {k: v.get("sha256") for k, v in (old.get("inputs") or {}).items()}
        new_in = {k: v.get("sha256") for k, v in fr["inputs"].items()}
        if old_in != new_in or old.get("known_truth_digest") != fr["known_truth_digest"] \
                or (old.get("prereg") or {}).get("core_sha256") != pre.core_sha256:
            raise DrawError("frames_v1.json exists from other inputs; rerun with --force to replace it "
                            "(no sample lock exists yet)")
    short = []
    for g, (planned, minimum) in sorted(pre.groups.items()):
        grp = fr["groups"].get(g)
        if grp is None or minimum is None:
            continue
        if grp["N"] < minimum:
            short.append("%s: frame %d < minimum %d" % (g, grp["N"], minimum))
            continue
        n_design = planned_sample(g, grp)
        if n_design < minimum:
            short.append("%s: the design draws %g < minimum %d from a frame of %d" % (g, n_design, minimum,
                                                                                    grp["N"]))
    if short:
        raise DrawError("frames below their pre-registered minimum (a decision for a person, F7): %s"
                        % "; ".join(short))
    picks, planted = draw_all(fr, pre, dom, leak_pairs_path=out_dir / "leak_pairs_v1.csv")
    rows, keys = build_rows(picks, fr, dom)
    from . import code_record  # noqa: F401  (header records the modules)
    seeds = {"prefix": S.SEED_PREFIX, "within_stratum": "%s/<stratum id>" % S.SEED_PREFIX,
             "G0_share": planted["seed_text"], "G3_units": "%s/G3/units" % S.SEED_PREFIX,
             "G4_prio": "%s/G4/prio" % S.SEED_PREFIX}

    def header_fn(inputs):
        return header("frames", dom, pre, inputs, seeds=seeds, modules=(S, sys.modules[__name__]),
                      testing=testing)
    fres = S.write_frames(fr, out_dir, header_fn=header_fn)
    sample_sha = write_csv_atomic(paths["sample"], SAMPLE_FIELDS, rows)
    key_rows = list(keys) + [{"planted": planted}]
    key_sha = write_jsonl_atomic(paths["key"], key_rows)
    frame_sizes = {g: grp["N"] for g, grp in sorted(fr["groups"].items())}
    g1 = fr["groups"]["G1"]["info"].get("frame_N", {})
    g4 = fr["groups"]["G4"]["info"].get("frame_N", {})
    amendment = {"id": D.next_amendment_id(pre), "kind": "sample_lock", "date": _today(),
                 "prereg_core_sha256": pre.core_sha256, "sample_sha256": sample_sha, "key_sha256": key_sha,
                 "frames_sha256": fres["sha256"], "name_status_v2_sha256": fr["name_status_v2"]["sha256"],
                 "frame_sizes": frame_sizes,
                 "confirmatory_frames": {"H2a": g1.get("noinfo", 0), "H2b": g1.get("named", 0),
                                         "H4_named": g4.get("named", 0), "H4_noinfo": g4.get("noinfo", 0)}}
    D.append_amendment(prereg_path, amendment)
    counts = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    return {"status": "drawn", "lock": amendment, "counts": counts, "planted_share": planted["share"],
            "sample": {"path": str(paths["sample"]), "sha256": sample_sha},
            "key": {"path": str(paths["key"]), "sha256": key_sha}, "frames": {"path": fres["path"],
                                                                                "sha256": fres["sha256"]}}


# ------------------------------------------------------------ re-derivation
def recompute(sample_row, frames, key_rows=None):
    """Re-derive the draw of one sample row's stratum from its seed text and
    the frames (strata.load_frames output). Returns {"ok", "selected",
    "why"}: ok when the row's unit sits at its draw_rank (srs, G3, G4
    uniform), is drawn by the Poisson part (G4 prio), or, for G0, when the
    key's part draw reproduces it."""
    g = sample_row["group"]
    sid = sample_row["stratum"]
    doc, fgroups = frames["doc"], frames["groups"]
    rank = int(sample_row["draw_rank"])
    uid = sample_row["unit_id"]
    if g == "G0":
        if key_rows is None:
            return {"ok": False, "why": "G0 rows are re-derived from the key", "selected": []}
        kr = {k["item_id"]: k for k in key_rows if "item_id" in k}.get(sample_row["item_id"])
        if kr is None or "g0" not in kr:
            return {"ok": False, "why": "no key row", "selected": []}
        part = kr["g0"]["part"]
        units = [u["unit_id"] for u in fgroups["G0"] if S.unit_extra(u).get("part") == part]
        planted = [k for k in key_rows if "planted" in k][0]["planted"]
        n = planted["parts"][part]
        sel = srs_order(units, kr["g0"]["seed_text"])[:n]
        ok = kr["g0"]["part_rank"] < len(sel) and sel[kr["g0"]["part_rank"]] == kr["unit_id"]
        return {"ok": ok, "selected": sel, "why": "" if ok else "G0 part draw does not reproduce the unit"}
    grp_doc = doc["groups"][g]
    if g == "G4":
        texts = sample_row["seed_text"].split(";")
        units = fgroups["G4"]
        ok = True
        sel = []
        for t in texts:
            if "/G4/uniform/frame=" in t:
                f = t.rsplit("=", 1)[1]
                ids = [u["unit_id"] for u in units if S.unit_extra(u)["frame"] == f]
                n = int(grp_doc["info"]["uniform"][f]["n"])
                sel = srs_order(ids, t)[:n]
                ok = ok and rank < len(sel) and sel[rank] == uid
            elif t.endswith("/G4/prio"):
                scores = {u["unit_id"]: float(u["score"]) for u in units}
                _, drawn = poisson_draw(scores, int(grp_doc["info"]["prio"]["n"]), t)
                if uid not in drawn:
                    ok = False
                elif len(texts) == 1:
                    ok = ok and drawn.index(uid) == rank
                if len(texts) == 1:
                    sel = drawn
            else:
                return {"ok": False, "selected": [], "why": "unknown G4 seed text %r" % t}
        return {"ok": ok, "selected": sel, "why": "" if ok else "G4 draw does not reproduce the unit"}
    ids = [u["unit_id"] for u in fgroups[g] if u["stratum"] == sid]
    n = int(grp_doc["strata"][sid]["n_planned"])
    text = sample_row["seed_text"]
    if not text.endswith("/" + sid):
        return {"ok": False, "selected": [], "why": "seed text %r is not the stratum's" % text}
    sel = srs_order(ids, text)[:n]
    ok = rank < len(sel) and sel[rank] == uid
    if ok and g == "G3" and not grp_doc["info"].get("take_all_units", True):
        chosen = srs_order(sorted(grp_doc["strata"]), "%s/G3/units" % S.SEED_PREFIX)[:int(grp_doc["info"]["m"])]
        ok = sid in chosen
    return {"ok": ok, "selected": sel, "why": "" if ok else "the stratum draw does not reproduce the unit"}


def _lock_of(pre):
    """The prereg's sample lock, refused when the prereg's core is no longer
    the one the lock was made under (the prereg was edited outside its
    amendments after the draw: every threshold read from it is suspect)."""
    lock = pre.sample_lock
    if lock is None:
        raise DrawError("the prereg holds no sample lock; the sample is not fixed")
    if lock.get("prereg_core_sha256") != pre.core_sha256:
        raise StaleInput("the sample lock %s was made under prereg core %s, the prereg's core is now %s: it was "
                         "edited outside its amendments" % (lock.get("id"), str(lock.get("prereg_core_sha256"))[:12],
                                                            pre.core_sha256[:12]))
    return lock


def load_sample(path, prereg):
    """sample_v1.csv rows, after checking the file against the prereg's sample
    lock (StaleInput when it changed or the prereg's core changed since the
    lock, DrawError when there is no lock)."""
    path = Path(path)
    pre = prereg if isinstance(prereg, D.Prereg) else D.load_prereg(prereg)
    lock = _lock_of(pre)
    got = file_record(path)["sha256"]
    if got != lock.get("sample_sha256"):
        raise StaleInput("sample_v1.csv hashes to %s, the sample lock records %s"
                         % (got[:12], str(lock.get("sample_sha256"))[:12]))
    header_, rows = read_csv(path)
    if tuple(header_) != SAMPLE_FIELDS:
        raise DrawError("sample_v1.csv columns %s" % header_)
    return rows


def load_key(path, prereg):
    """sample_v1_key.jsonl rows, checked against the lock's key_sha256 (and
    the lock against the prereg's core)."""
    path = Path(path)
    pre = prereg if isinstance(prereg, D.Prereg) else D.load_prereg(prereg)
    lock = _lock_of(pre)
    got = file_record(path)["sha256"]
    if got != lock.get("key_sha256"):
        raise StaleInput("sample_v1_key.jsonl hashes to %s, the lock records %s"
                         % (got[:12], str(lock.get("key_sha256"))[:12]))
    return read_jsonl(path)
