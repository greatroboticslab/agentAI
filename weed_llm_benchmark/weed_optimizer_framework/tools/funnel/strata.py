"""Sampling frames, strata and allocation (contract §5.1-5.2, runner §4.7-4.8).

frames() turns the census outputs (census_v1.json, ledger.jsonl,
name_status_v2.json, funnel_ledger.json, guard_pairs_v1.csv) and the
adapter's known truth into one frame per group, each split into strata with
a planned sample size, before any label exists. draw.py draws from them.

Groups (runner §4.8): G0 planted control, G1 other-labelled boxes the probe
called a target, G2 target-labelled boxes not verified, G2v verified boxes in
vetoed images, G2a verified boxes outside the reference lab group, G3 class
and visual-cluster units of uninformative label spaces, G4 other-labelled
boxes the probe let through, G5 guard pairs, sentinels, pair sentinels and
the identity check.

The engine reads the stage of each role (size, target_check, other_check,
image_rule) from funnel_ledger.json and the name-status frames from the
domain config, so nothing here names a domain's classes, sources or stages.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import (STEP1_DIR, SampleLocked, StaleInput, StrataError, check_prereg_core, check_records,
               file_record, read_csv, read_json, rng, seed, write_csv_atomic, write_json_atomic)
from . import ledger as L

FRAME_FIELDS = ("unit_id", "unit", "stratum", "source", "image_key", "crop_id", "lab", "near_dup3",
                "provenance", "label", "pred", "score", "allowed_judges", "extra")
GROUPS = ("G0", "G1", "G2", "G2v", "G2a", "G3", "G4", "G5", "identity", "sentinel", "pair_sentinel")
STRATUM_KEYS = {"G0": (), "G1": ("frame", "status", "pred"), "G2": ("source", "label", "fail"),
                "G2v": ("source",), "G2a": ("source",), "G3": ("unit",),
                "G4": ("frame", "argmax_target", "band"), "G5": ("source", "split", "bits"),
                "sentinel": ("kt", "truth_kind"), "identity": ("class",),
                "pair_sentinel": ("kind",)}
DESIGNS = {"G0": "planted", "G1": "srs", "G2": "srs", "G2v": "srs", "G2a": "srs", "G3": "two_stage",
           "G4": "dual", "G5": "allocation", "sentinel": "sentinel", "identity": "identity",
           "pair_sentinel": "sentinel"}
POOL_GROUPS = ("G0", "G1", "G2", "G2v", "G2a", "G3", "identity", "G4")   # sheet order (runner §4.10)
SEED_PREFIX = "funnel/v1"
FLOOR = 5
G1_EXCLUDED_MAX = 10
G3_MIN_BOXES = 100
G3_CLUSTERS = 8
G3_UNIT_N_MIN, G3_UNIT_N_MAX = 20, 30
G4_UNIFORM_SPLIT = (("named", 0.75), ("noinfo", 0.25))    # the H4 power rule (runner §9 item 9)
BIT_BANDS = (("0-2", 0, 2), ("3-4", 3, 4), ("5-6", 5, 6))
G0_IN_DOMAIN_SET = "KT2"          # contract §5.2: planted positives in the reference domain
PAIR_POSITIVE_SET = "KT2"         # copy <-> reference twin pairs (runner §4.8)
G3_STATUS_EXTRA = ("target_related", "target_synonym")
NO_NAME_STATUS = "no_name"
EXEMPLAR_SESSIONS_PER_CLASS = 2
# the §4.2 outcome vocabulary of the roles the frames read
SIZE_EMBEDDED = "embedded"
VERIFIED, CONFLICT, OTHER_OK, ADMITTED = "verified", "conflict", "other_ok", "admitted"


# ------------------------------------------------------------------ helpers
def stratum_id(group, **keys):
    """<group>/<key>=<value>/... in the fixed key order of runner §4.8
    (G0's one stratum is "G0/all")."""
    if group not in STRATUM_KEYS:
        raise StrataError("unknown group %r" % group)
    want = STRATUM_KEYS[group]
    if set(keys) != set(want):
        raise StrataError("stratum of %s needs keys %s, got %s" % (group, want, sorted(keys)))
    if not want:
        return "%s/all" % group
    parts = [group]
    for k in want:
        v = str(keys[k])
        if v == "" or "/" in v or "=" in v:
            raise StrataError("stratum key %s=%r cannot be part of an id" % (k, v))
        parts.append("%s=%s" % (k, v))
    return "/".join(parts)


def parse_stratum(sid):
    """(group, {key: value}) of a stratum id."""
    parts = sid.split("/")
    group = parts[0]
    if group not in STRATUM_KEYS:
        raise StrataError("unknown group in stratum %r" % sid)
    if parts[1:] == ["all"] and not STRATUM_KEYS[group]:
        return group, {}
    keys = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        keys[k] = v
    if tuple(keys) != STRATUM_KEYS[group]:
        raise StrataError("stratum %r does not follow the key order %s" % (sid, STRATUM_KEYS[group]))
    return group, keys


def seed_text(sid, prefix=SEED_PREFIX):
    """funnel/v1/<group>/<stratum> for a within-stratum draw (contract §5.2)."""
    return "%s/%s" % (prefix, sid)


def allocate(sizes, n, floor=FLOOR):
    """Allocate n over strata of the given sizes: each non-empty stratum gets
    min(N_h, floor), the rest goes in proportion to N_h by largest remainder
    (ties by stratum id), never above N_h. If the floors alone exceed n the
    floor is lowered until they fit; if n >= the frame, every unit is taken."""
    n = int(n)
    if n < 0:
        raise StrataError("cannot allocate a negative sample size")
    ids = sorted(k for k, v in sizes.items() if int(v) > 0)
    out = {k: 0 for k in sizes}
    total = sum(int(sizes[k]) for k in ids)
    if n >= total:
        for k in ids:
            out[k] = int(sizes[k])
        return out
    f = int(floor)
    while f > 0 and sum(min(int(sizes[k]), f) for k in ids) > n:
        f -= 1
    for k in ids:
        out[k] = min(int(sizes[k]), f)
    add = allocate_weighted({k: int(sizes[k]) for k in ids},
                            {k: int(sizes[k]) - out[k] for k in ids}, n - sum(out.values()))
    for k, v in add.items():
        out[k] += v
    return out


def allocate_weighted(weights, caps, n):
    """Split n in proportion to weights by largest remainder (ties by key),
    never above caps; what a capped key cannot take goes to the others in
    proportion to their weights. Returns {key: n_k}; the sum is min(n, sum of
    caps)."""
    keys = sorted(k for k in weights if float(weights[k]) > 0 and int(caps.get(k, 0)) > 0)
    out = {k: 0 for k in weights}
    rest = min(int(n), sum(int(caps[k]) for k in keys))
    while rest > 0:
        elig = [k for k in keys if out[k] < int(caps[k])]
        if not elig:
            break
        tot = float(sum(float(weights[k]) for k in elig))
        quota = {k: rest * float(weights[k]) / tot for k in elig}
        added = 0
        for k in elig:
            b = min(int(math.floor(quota[k])), int(caps[k]) - out[k])
            out[k] += b
            added += b
        rest -= added
        if rest <= 0:
            break
        order = sorted((k for k in elig if out[k] < int(caps[k])),
                       key=lambda k: (-(quota[k] - math.floor(quota[k])), k))
        for k in order:
            if rest == 0:
                break
            out[k] += 1
            rest -= 1
    return out


def g4_bands(scores):
    """Tertile edges (numpy quantile, linear) and band 1..3 of each score; a
    score equal to an edge falls in the lower band."""
    import numpy as np
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return [], np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(s)):
        raise StrataError("a G4 score is not finite")
    e1, e2 = np.quantile(s, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    bands = 1 + (s > e1).astype(np.int64) + (s > e2).astype(np.int64)
    return [float(e1), float(e2)], bands


def visual_clusters(X, seed_text_, k=G3_CLUSTERS):
    """KMeans (sklearn, n_init=10) cluster of each row of X after L2
    normalisation, seeded by seed_text_. Rows with non-finite features are -1."""
    import numpy as np
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise StrataError("visual clusters need scikit-learn (%s)" % e)
    X = np.asarray(X, dtype=np.float64)
    ok = np.all(np.isfinite(X), axis=1)
    out = np.full(len(X), -1, dtype=np.int64)
    Xo = X[ok]
    if len(Xo) < k:
        raise StrataError("%d finite rows cannot make %d clusters" % (len(Xo), k))
    nrm = np.linalg.norm(Xo, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    km = KMeans(n_clusters=k, n_init=10, random_state=seed(seed_text_))
    out[ok] = km.fit_predict(Xo / nrm)
    return out


def bits_band(bits):
    b = int(bits)
    for name, lo, hi in BIT_BANDS:
        if lo <= b <= hi:
            return name
    raise StrataError("guard pair at %d bits is outside the bands %s" % (b, [x[0] for x in BIT_BANDS]))


def frame_of(status, domain):
    """noinfo | named | excluded | target: the name-status frame of a v2 status."""
    if status == "target":
        return "target"
    fr = (domain.section("names") or {}).get("frames") or {}
    for f in ("noinfo", "named", "excluded"):
        if status in fr.get(f, []):
            return f
    raise StrataError("name status %r is in no frame of the domain config" % status)


def role_stage_ids(funnel_ledger):
    """{role: stage id} for the roles the frames read; each must be unique."""
    out = {}
    for role in ("size", "join", "target_check", "other_check", "image_rule", "evidence"):
        st = L.stage_by_role(funnel_ledger, role)
        if len(st) != 1:
            raise StrataError("the funnel ledger has %d stage(s) of role %s, need exactly 1"
                              % (len(st), role))
        out[role] = st[0]["id"]
    return out


def exemplar_sessions(domain, known_truth):
    """{board id: {option n: [session, ...]}}: the capture sessions a board's
    exemplars of one option come from, for options whose exemplar set is not
    independent and carries sessions. Per option: the option's items sorted by
    id, their sessions sorted, the first two of a permutation seeded
    funnel/v1/board/<board>/<option n> (the rule sheets.py applies when it
    renders the board). Sentinel and identity draws exclude these sessions."""
    rl = domain.section("reference_labeller", required=False) or {}
    indep = set(domain.independent_sets())
    options = domain.options() if rl.get("options_tail") else []
    out = {}
    for b in rl.get("boards", []) or []:
        chosen = {}
        for o in options:
            if o["kind"] == "tail":
                continue
            kt_id = b["targets_from"] if o["kind"] == "target" else b["attractors_from"]
            if kt_id in indep:
                continue
            items = known_truth.get(kt_id, [])
            if o["kind"] == "target":
                pool = [it for it in items if it.get("truth_kind") == "target" and it.get("truth") is not None
                        and int(it["truth"]) == int(o["class"])]
            else:
                pool = [it for it in items if it.get("truth_kind") == "attractor"
                        and it.get("truth_taxon") == o["taxon"]]
            all_s = sorted({it.get("session") for it in pool if it.get("session")})
            if not all_s:
                continue
            pick = rng("%s/board/%s/%d" % (SEED_PREFIX, b["id"], o["n"])).permutation(len(all_s))
            chosen[o["n"]] = sorted(all_s[i] for i in pick[:EXEMPLAR_SESSIONS_PER_CLASS])
        if chosen:
            out[b["id"]] = chosen
    return out


def _unit(unit_id, unit, stratum, source="", image_key="", crop_id="", lab="", near_dup3="",
          provenance="", label="", pred="", score="", extra=None):
    return {"unit_id": unit_id, "unit": unit, "stratum": stratum, "source": source,
            "image_key": image_key, "crop_id": "" if crop_id is None else crop_id, "lab": lab,
            "near_dup3": near_dup3, "provenance": provenance, "label": label, "pred": pred,
            "score": score, "allowed_judges": "",
            "extra": json.dumps(extra or {}, sort_keys=True, separators=(",", ":"))}


def unit_extra(u):
    return json.loads(u["extra"]) if u.get("extra") else {}


def _kt_unit(it, group, stratum, extra=None):
    iid = it["id"]
    unit = "photo" if it.get("crop_set") == "kt7" else ("box" if "#" in iid else "image")
    key = iid.split(":", 1)[1].split("#", 1)[0] if ":" in iid else ""
    ex = {"kt": it.get("kt"), "role": it.get("role")}
    ex.update(extra or {})
    return _unit(iid, unit, stratum, source=it.get("source") or "", image_key=key,
                 crop_id=it.get("crop_id"), lab=it.get("lab") or "",
                 near_dup3=it.get("near_dup3") or "", provenance=it.get("provenance") or "",
                 label="" if it.get("truth") is None else str(it.get("truth")), extra=ex)


def _group(group, planned, minimum):
    return {"design": DESIGNS[group], "N": 0, "planned": planned, "minimum": minimum,
            "strata": {}, "units": [], "info": {}}


def _add_stratum(g, sid, definition):
    if sid not in g["strata"]:
        g["strata"][sid] = {"N": 0, "n_planned": 0, "definition": definition, "allowed_judges": []}
    return g["strata"][sid]


def _keys_of(units):
    return {"source": sorted({u["source"] for u in units if u["source"]}),
            "near_dup3": sorted({u["near_dup3"] for u in units if u["near_dup3"]}),
            "provenance": sorted({u["provenance"] for u in units if u["provenance"]}),
            "lab": sorted({u["lab"] for u in units if u["lab"]})}


def stratum_keys(units):
    """The disjointness keys a stratum's units carry (runner §1.3)."""
    return _keys_of(units)


# ------------------------------------------------------------------- frames
def _status_frames(domain):
    fr = (domain.section("names") or {}).get("frames") or {}
    out = {"target": "target"}
    for f in ("noinfo", "named", "excluded"):
        for s in fr.get(f, []):
            out[s] = f
    return out


def _box_frames(domain, rows, roles, groups):
    """G1, G2, G2v, G2a, G4 and the raw material of G3 from ledger box rows."""
    other = domain.other["id"]
    status_frame = _status_frames(domain)
    ref_lab = domain.reference_lab()
    size_s, tc_s, oc_s, ir_s = roles["size"], roles["target_check"], roles["other_check"], roles["image_rule"]
    g1, g2, g2v, g2a, g4 = (groups[g] for g in ("G1", "G2", "G2v", "G2a", "G4"))
    g3_raw = {}
    small = {"boxes": 0}
    g1_pred_counts = {}
    for r in rows:
        path = r["path"]
        if path.get(size_s) != SIZE_EMBEDDED:
            small["boxes"] += 1
            continue
        label = int(r["label"])
        pred = r.get("pred")
        status = r["name_status_v2"]
        base = dict(source=r["source"], image_key=r["key"], crop_id=r["crop_id"], lab=r["lab"],
                    near_dup3=r["near_dup3"], provenance=r["provenance"],
                    label=domain.class_name(label),
                    pred="" if pred is None else domain.class_name(int(pred)))
        kt = list(r.get("kt") or [])

        def bx(stratum, extra, score=""):
            ex = dict(extra)
            ex["kt"] = kt
            return dict(_unit(r["id"], "box", stratum, score=score, extra=ex), **base)
        if label == other:
            fr = status_frame.get(status)
            if fr is None:
                raise StrataError("name status %r of box %s is in no frame of the domain config"
                                  % (status, r["id"]))
            oc = path.get(oc_s)
            if oc == CONFLICT:
                g1["units"].append(bx("", {"status": status, "frame": fr, "src_id": r["src_id"]}))
                g1_pred_counts.setdefault(fr, {})
                g1_pred_counts[fr][base["pred"]] = g1_pred_counts[fr].get(base["pred"], 0) + 1
            elif oc == OTHER_OK:
                s = r.get("p_target_max")
                if s is None:
                    raise StrataError("box %s has no p_target_max; the G4 design needs every score" % r["id"])
                g4["units"].append(bx("", {"status": status, "frame": fr, "src_id": r["src_id"],
                                           "argmax_target": "yes" if domain.is_target(pred) else "no"},
                                      score=repr(float(s))))
            if fr == "noinfo" or status in G3_STATUS_EXTRA:
                cu = g3_raw.setdefault((r["source"], str(r["src_id"])),
                                       {"status": status, "name": r["src_name"], "units": []})
                cu["units"].append(bx("", {"status": status}))
        elif domain.is_target(label):
            tc = path.get(tc_s)
            if tc != VERIFIED:
                fail = r.get("fail") or "none"
                g2["units"].append(bx(stratum_id("G2", source=r["source"], label=domain.class_name(label),
                                                 fail=fail), {"fail": fail}))
            else:
                if path.get(ir_s) != ADMITTED:
                    g2v["units"].append(bx(stratum_id("G2v", source=r["source"]),
                                           {"image_rule": path.get(ir_s)}))
                if r["lab"] != ref_lab:
                    g2a["units"].append(bx(stratum_id("G2a", source=r["source"]),
                                           {"admitted": path.get(ir_s) == ADMITTED}))
        else:
            raise StrataError("box %s has label %r, neither a target nor the other class" % (r["id"], label))
    # G1 strata: frame, status, pred (the frame's top 3 predictions, else "rest")
    top = {}
    for fr, cnt in g1_pred_counts.items():
        ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        top[fr] = {p for p, _ in ranked[:3]}
    for u in g1["units"]:
        ex = unit_extra(u)
        p = u["pred"] if u["pred"] in top[ex["frame"]] else "rest"
        u["stratum"] = stratum_id("G1", frame=ex["frame"], status=ex["status"], pred=p)
    g1["info"]["top_pred"] = {fr: sorted(v) for fr, v in sorted(top.items())}
    # G4 post-strata: frame, argmax_target, band (tertile of the score within the frame)
    by_frame = {}
    for u in g4["units"]:
        by_frame.setdefault(unit_extra(u)["frame"], []).append(u)
    edges_all = {}
    for fr in sorted(by_frame):
        us = by_frame[fr]
        edges, bands = g4_bands([float(u["score"]) for u in us])
        edges_all[fr] = edges
        for u, b in zip(us, bands):
            ex = unit_extra(u)
            u["stratum"] = stratum_id("G4", frame=fr, argmax_target=ex["argmax_target"], band=int(b))
    g4["info"]["band_edges"] = edges_all
    return g3_raw, small


def _g3(domain, g3_raw, group, dinov2):
    """Class units, a no-name class replaced by its visual clusters; units under
    G3_MIN_BOXES boxes are listed, not sampled."""
    listed_small = []
    for (src, sid), cu in sorted(g3_raw.items()):
        cid = "c:%s|%s" % (src, sid)
        if cu["status"] == NO_NAME_STATUS:
            if len(cu["units"]) < G3_MIN_BOXES:
                listed_small.append({"unit": cid, "boxes": len(cu["units"]), "status": cu["status"]})
                continue
            if dinov2 is None:
                raise StrataError("no-name class %s needs visual clusters: pass DINOv2 features" % cid)
            crop_ids = [int(u["crop_id"]) for u in cu["units"]]
            X = dinov2(crop_ids)
            lab = visual_clusters(X, "%s/G3/kmeans/%s|%s" % (SEED_PREFIX, src, sid), G3_CLUSTERS)
            per = {}
            for u, j in zip(cu["units"], lab):
                per.setdefault(int(j), []).append(u)
            for j in sorted(per):
                if j < 0:
                    listed_small.append({"unit": cid, "cluster": None, "boxes": len(per[j]),
                                         "status": cu["status"], "why": "non-finite features"})
                    continue
                kid = "k:%s|%s|%d" % (src, sid, j)
                if len(per[j]) < G3_MIN_BOXES:
                    listed_small.append({"unit": kid, "boxes": len(per[j]), "status": cu["status"]})
                    continue
                _g3_unit(group, kid, per[j], cu)
        else:
            if len(cu["units"]) < G3_MIN_BOXES:
                listed_small.append({"unit": cid, "boxes": len(cu["units"]), "status": cu["status"]})
                continue
            _g3_unit(group, cid, cu["units"], cu)
    group["info"]["listed_not_sampled"] = listed_small


def _g3_unit(group, uid, units, cu):
    sid = stratum_id("G3", unit=uid)
    st = _add_stratum(group, sid, "class or cluster unit %s (status %s, name %r)" % (uid, cu["status"], cu["name"]))
    for u in sorted(units, key=lambda x: x["unit_id"]):
        u = dict(u)
        u["stratum"] = sid
        ex = unit_extra(u)
        ex["class_unit"] = uid
        u["extra"] = json.dumps(ex, sort_keys=True, separators=(",", ":"))
        group["units"].append(u)
        st["N"] += 1


def _finish(group, definitions):
    for u in group["units"]:
        st = _add_stratum(group, u["stratum"], definitions(u["stratum"]))
        if group["design"] != "two_stage":
            st["N"] += 1
    group["N"] = len(group["units"])
    group["units"].sort(key=lambda u: (u["stratum"], u["unit_id"]))


def _alloc_group(group, n):
    sizes = {sid: st["N"] for sid, st in group["strata"].items()}
    al = allocate(sizes, n)
    for sid, st in group["strata"].items():
        st["n_planned"] = al.get(sid, 0)


def _plan_g1(g, planned):
    frames = {}
    for sid, st in g["strata"].items():
        frames.setdefault(parse_stratum(sid)[1]["frame"], {})[sid] = st["N"]
    tot = {f: sum(v.values()) for f, v in frames.items()}
    n_ex = min(tot.get("excluded", 0), G1_EXCLUDED_MAX)
    rest = max(0, planned - n_ex)
    want = {"noinfo": rest // 2, "named": rest - rest // 2}
    got = {f: min(want[f], tot.get(f, 0)) for f in want}
    spare = sum(want.values()) - sum(got.values())
    for f in ("noinfo", "named"):
        extra = min(spare, tot.get(f, 0) - got[f])
        got[f] += extra
        spare -= extra
    got["excluded"] = n_ex
    for f, sizes in frames.items():
        al = allocate(sizes, got.get(f, 0))
        for sid, n in al.items():
            g["strata"][sid]["n_planned"] = n
    g["info"]["frame_n"] = {f: got.get(f, 0) for f in sorted(frames)}
    g["info"]["frame_N"] = {f: tot[f] for f in sorted(tot)}


def _plan_g3(g, planned):
    units = sorted(g["strata"])
    M = len(units)
    info = {"M": M, "planned": planned}
    if M == 0:
        info.update({"m": 0, "per_unit_n": 0, "take_all_units": True})
    elif G3_UNIT_N_MIN * M <= planned:
        per = min(G3_UNIT_N_MAX, planned // M)
        info.update({"m": M, "per_unit_n": per, "take_all_units": True})
    else:
        info.update({"m": planned // G3_UNIT_N_MIN, "per_unit_n": G3_UNIT_N_MIN, "take_all_units": False,
                     "units_seed": "%s/G3/units" % SEED_PREFIX})
    for sid in units:
        st = g["strata"][sid]
        st["n_planned"] = min(info["per_unit_n"], st["N"])
    g["info"].update(info)


def _plan_g4(g, planned, uniform_total):
    frames = {}
    for u in g["units"]:
        frames.setdefault(unit_extra(u)["frame"], []).append(u)
    uni = {}
    for f, share in G4_UNIFORM_SPLIT:
        N = len(frames.get(f, []))
        uni[f] = {"N": N, "n": min(N, int(round(uniform_total * share)))}
    S = sum(float(u["score"]) for u in g["units"])
    g["info"]["uniform"] = uni
    g["info"]["uniform_seed"] = {f: "%s/G4/uniform/frame=%s" % (SEED_PREFIX, f) for f in sorted(uni)}
    g["info"]["prio"] = {"n": max(0, planned - uniform_total), "S": S, "seed": "%s/G4/prio" % SEED_PREFIX}
    g["info"]["frame_N"] = {f: len(v) for f, v in sorted(frames.items())}
    g["info"]["reported_not_uniform"] = sorted(f for f in frames if f not in dict(G4_UNIFORM_SPLIT))


def _g5(domain, pairs_path, g, planned):
    alloc_cfg = (domain.section("sampling", required=False) or {}).get("G5_allocation") or {}
    dup_rule = (domain.section("sampling", required=False) or {}).get("G5_dup_twins")
    ref_lab = domain.reference_lab()
    if pairs_path is None or not Path(pairs_path).exists():
        raise StrataError("guard_pairs_v1.csv is missing; G5 cannot be framed")
    header_, rows = read_csv(pairs_path)
    id_col = "unit_id" if "unit_id" in header_ else "pair_id"
    need = (id_col, "kind", "source", "split", "bits", "sha_differs")
    missing = [k for k in need if k not in header_]
    if missing:
        raise StrataError("guard_pairs_v1.csv lacks columns %s" % missing)
    for slug in alloc_cfg:
        if domain.lab_of(slug) == ref_lab:
            raise StrataError("G5_allocation names %s, which is in the reference lab group" % slug)
    pair_units, dup_units = {}, {}
    for r in rows:
        lab = r.get("lab") or domain.lab_of(r["source"])
        if lab == ref_lab:
            continue
        band = bits_band(r["bits"])
        u = _unit(r[id_col], "pair", stratum_id("G5", source=r["source"], split=r["split"], bits=band),
                  source=r["source"], image_key=r.get("rel") or "", lab=lab,
                  near_dup3=r.get("near_dup3") or "", provenance=r.get("provenance") or "",
                  extra={"kind": r["kind"], "bits": int(r["bits"])})
        if r["kind"] == "exact_dup":
            if dup_rule == "all_sha_differs" and str(r["sha_differs"]).lower() in ("1", "true", "yes"):
                dup_units.setdefault(r["source"], []).append(u)
        elif r["source"] in alloc_cfg:
            pair_units.setdefault(r["source"], []).append(u)
    per_source = {}
    for src in sorted(pair_units):
        want = alloc_cfg[src]
        per_source[src] = len(pair_units[src]) if want == "all" else min(int(want), len(pair_units[src]))
    used = sum(per_source.values())
    dup_alloc = allocate({s: len(v) for s, v in dup_units.items()}, max(0, planned - used), floor=0)
    kind_of = {}
    for units in list(pair_units.values()) + list(dup_units.values()):
        for u in units:
            g["units"].append(u)
            k = "dup" if unit_extra(u)["kind"] == "exact_dup" else "pair"
            if kind_of.setdefault(u["stratum"], k) != k:
                raise StrataError("G5 stratum %s mixes exact_dup twins and guard pairs" % u["stratum"])
    _finish(g, lambda sid: "guard pairs %s" % sid)
    for src in sorted(set(u["source"] for u in g["units"])):
        for kind, n_src in (("pair", per_source.get(src, 0)), ("dup", dup_alloc.get(src, 0))):
            sizes = {sid: st["N"] for sid, st in g["strata"].items()
                     if parse_stratum(sid)[1]["source"] == src and kind_of.get(sid) == kind}
            for sid, n in allocate(sizes, n_src).items():
                g["strata"][sid]["n_planned"] = n
    g["info"].update({"per_source": per_source, "dup_twins": dup_alloc,
                      "allocation": {k: alloc_cfg[k] for k in sorted(alloc_cfg)}, "dup_rule": dup_rule})


def hidden_targets(adapter, domain):
    """(hidden, record): hidden(item) is True when the adapter's crop table
    gives the item's crop the other class, i.e. a known-truth target the join
    hid. Contract §5.2 (runner §4.8): G0's in-domain positives are the
    reference copies' hidden targets, not every copy box. Refuses when the
    adapter cannot tell (no crop table, or one without labels)."""
    fn = getattr(adapter, "crop_table", None)
    if not callable(fn):
        raise StrataError("the adapter offers no crop table: G0's hidden in-domain targets cannot be told apart")
    crops = fn()
    labels = getattr(crops, "label", None)
    if labels is None:
        raise StrataError("the adapter's crop table has no labels")
    other = int(domain.other["id"])
    n = len(labels)

    def hidden(it):
        c = it.get("crop_id")
        return c not in (None, "") and 0 <= int(c) < n and int(labels[int(c)]) == other
    return hidden, {"path": str(getattr(crops, "path", "") or ""), "sha256": getattr(crops, "sha", None)}


def _kt_groups(domain, known_truth, groups, planned_g0, hidden=None):
    """G0 frame (in-domain planted positives, independent-set g0 photos) and
    the identity frames; sentinels are framed at draw time, after the other
    groups, from what remains. hidden(item) selects the in-domain set's
    hidden targets (hidden_targets)."""
    indep = domain.independent_sets()
    if len(indep) != 1:
        raise StrataError("the domain needs exactly one independent known-truth set, has %s" % indep)
    kti = indep[0]
    if G0_IN_DOMAIN_SET not in known_truth:
        raise StrataError("known truth lacks %s (G0 in-domain positives)" % G0_IN_DOMAIN_SET)
    if hidden is None:
        raise StrataError("G0 needs the hidden-target test of the in-domain set (hidden_targets)")
    g0 = groups["G0"]
    parts = {"pos_in_domain": [it for it in known_truth[G0_IN_DOMAIN_SET]
                               if it.get("truth_kind") == "target" and hidden(it)],
             "pos_independent": [it for it in known_truth.get(kti, [])
                                 if it.get("role") == "g0" and it.get("truth_kind") == "target"],
             "neg_independent": [it for it in known_truth.get(kti, [])
                                 if it.get("role") == "g0" and it.get("truth_kind") == "attractor"]}
    for part, items in sorted(parts.items()):
        for it in sorted(items, key=lambda x: x["id"]):
            g0["units"].append(_kt_unit(it, "G0", "G0/all", extra={"part": part}))
    _finish(g0, lambda sid: "planted control")
    g0["strata"]["G0/all"]["n_planned"] = min(planned_g0, g0["N"])
    g0["info"]["parts_N"] = {p: len(v) for p, v in sorted(parts.items())}
    g0["info"]["independent_set"] = kti
    ident = groups["identity"]
    sessions = exemplar_sessions(domain, known_truth)
    ex_sess = exemplar_sessions_by_class(domain, sessions)
    for ic in domain.section("identity_checks", required=False) or []:
        cid = domain.class_id(ic["class"])
        sid = stratum_id("identity", **{"class": ic["class"]})
        items = [it for it in known_truth.get(ic["from"], [])
                 if it.get("truth") is not None and int(it["truth"]) == cid
                 and it.get("role") != "exemplar" and not in_exemplar_session(it, ex_sess)]
        for it in sorted(items, key=lambda x: x["id"]):
            ident["units"].append(_kt_unit(it, "identity", sid, extra={"check": ic["class"]}))
        _add_stratum(ident, sid, "identity check of %s from %s" % (ic["class"], ic["from"]))
        ident["strata"][sid]["n_planned_config"] = int(ic["n"])
    _finish(ident, lambda sid: "identity check")
    for sid, st in ident["strata"].items():
        st["n_planned"] = min(st.pop("n_planned_config"), st["N"])
    ident["info"]["exemplar_sessions"] = {b: {str(k): v for k, v in per.items()} for b, per in sessions.items()}
    return sessions, ex_sess


def exemplar_sessions_by_class(domain, sessions):
    """{class id: set of sessions} the boards draw that class's exemplars from."""
    by_n = {o["n"]: o for o in domain.options()} if sessions else {}
    out = {}
    for per in sessions.values():
        for n, ss in per.items():
            o = by_n.get(int(n))
            if o is not None and o["kind"] == "target":
                out.setdefault(int(o["class"]), set()).update(ss)
    return out


def in_exemplar_session(item, ex_sess):
    """An item of a target class taken in a session its class's board
    exemplars come from (sentinels and identity items exclude these)."""
    if not item.get("session") or item.get("truth") in (None, "") or item.get("truth_kind") != "target":
        return False
    return item["session"] in ex_sess.get(int(item["truth"]), ())


def _definition(group):
    return {"G1": "other-labelled embedded box the probe confidently called a target",
            "G2": "target-labelled embedded box not verified",
            "G2v": "verified target box in an image the image rule did not admit",
            "G2a": "verified target box of a source outside the reference lab group",
            "G4": "other-labelled embedded box the probe let through"}.get(group, group)


def frames(prereg, domain, census_dir, adapter, judge_qual=None, dinov2=None, allowed_fn=None):
    """Every group's frame, strata and planned n, in memory (runner §4.7-4.8).

    prereg: domain.Prereg; domain: domain.Domain; census_dir: the funnel
    directory holding the census outputs; adapter: the domain's adapter (its
    known_truth); judge_qual: judge_qualification.json content or None;
    dinov2: crop_ids -> features (for the no-name visual clusters);
    allowed_fn: (stratum keys, judge material) -> allowed judges (defaults to
    qualify.allowed_judges when judge_qual is given)."""
    fd = Path(census_dir)
    census_p = fd / "census_v1.json"
    census = read_json(census_p)
    if (census.get("reconciliation") or {}).get("ok") is False:
        raise StrataError("census_v1.json does not reconcile with the Step 1 summaries; nothing is framed on it")
    check_prereg_core(census, prereg, "census_v1.json")
    ns_rec = census.get("name_status_v2") or {}
    ns_p = fd / "name_status_v2.json"
    if not ns_p.exists():
        raise StrataError("name_status_v2.json is missing")
    ns_file = file_record(ns_p)
    if ns_rec.get("sha256") != ns_file["sha256"]:
        raise StaleInput("name_status_v2.json hashes to %s, census_v1.json records %s"
                         % (ns_file["sha256"][:12], str(ns_rec.get("sha256"))[:12]))
    fl_p = fd / "funnel_ledger.json"
    fl = L.load(fl_p)
    if fl.get("derivation") != "census":
        raise StrataError("frames need the census-derived funnel ledger, not %r" % fl.get("derivation"))
    check_prereg_core(fl, prereg, "funnel_ledger.json")
    roles = role_stage_ids(fl)
    ledger_p = fd / "ledger.jsonl"
    if not ledger_p.exists():
        raise StrataError("ledger.jsonl is missing")
    groups_cfg = prereg.groups
    groups = {g: _group(g, *(groups_cfg.get(g, (0, None)))) for g in GROUPS}
    g3_raw, small = _box_frames(domain, L.iter_units(ledger_p, unit="box"), roles, groups)
    for g in ("G1", "G2", "G2v", "G2a", "G4"):
        _finish(groups[g], lambda sid, g=g: "%s: %s" % (_definition(g), sid))
    _g3(domain, g3_raw, groups["G3"], dinov2)
    groups["G3"]["N"] = len(groups["G3"]["units"])
    groups["G3"]["units"].sort(key=lambda u: (u["stratum"], u["unit_id"]))
    _plan_g1(groups["G1"], groups["G1"]["planned"])
    for g in ("G2", "G2v", "G2a"):
        _alloc_group(groups[g], groups[g]["planned"])
    _plan_g3(groups["G3"], groups["G3"]["planned"])
    uniform_total = int((prereg.raw.get("sampling") or {}).get("G4_uniform", 0))
    _plan_g4(groups["G4"], groups["G4"]["planned"], uniform_total)
    pairs_p = fd / "guard_pairs_v1.csv"
    _g5(domain, pairs_p, groups["G5"], groups["G5"]["planned"])
    known = adapter.known_truth(domain, fd)
    hidden, crops_rec = hidden_targets(adapter, domain)
    sessions, ex_sess = _kt_groups(domain, known, groups, groups["G0"]["planned"], hidden)
    inputs = {"census_v1": file_record(census_p), "name_status_v2": ns_file,
              "funnel_ledger": file_record(fl_p), "ledger_jsonl": file_record(ledger_p),
              "guard_pairs_v1": file_record(pairs_p)}
    if crops_rec.get("sha256"):
        inputs["crop_table"] = crops_rec
    for opt in ("relation_geometry_v1.json", "judge_qualification.json", "leak_pairs_v1.csv", "leak_pairs_v2.csv"):
        if (fd / opt).exists():
            inputs[opt.rsplit(".", 1)[0]] = file_record(fd / opt)
    rec = getattr(dinov2, "record", None)
    if callable(rec) and rec():
        inputs["emb_dinov2"] = rec()
    out = {"groups": groups, "inputs": inputs, "name_status_v2": ns_file,
           "roles": roles, "known_truth": known, "known_truth_digest": known_truth_digest(known),
           "exemplar_sessions": sessions,
           "exemplar_sessions_by_class": {str(k): sorted(v) for k, v in sorted(ex_sess.items())},
           "reported": {"small_boxes": small["boxes"],
                        "G3_listed_not_sampled": groups["G3"]["info"].get("listed_not_sampled", [])}}
    set_allowed_judges(out, judge_qual, allowed_fn, funnel_dir=fd)
    return out


def known_truth_digest(known):
    """sha256 of the canonical JSON of the adapter's known truth (in-memory
    input of the draw; recorded so a rerun can tell whether it changed)."""
    from . import canonical_json, sha256_bytes
    return sha256_bytes(canonical_json(known).encode("utf-8"))


PPI_GROUPS = ("G1", "G2", "G2v", "G2a", "G3", "G4")


def set_allowed_judges(fr, judge_qual, allowed_fn=None, funnel_dir=None):
    """Fill allowed_judges per stratum of the estimation groups: the judges
    whose calibration material (in the lab scope qualify.material_for picks
    for the stratum) shares nothing with it. allowed_fn(stratum keys,
    judge_qual) -> [judge] replaces the default (qualify.material_for and
    qualify.allowed_judges on judge_material_v1.json)."""
    if judge_qual is None:
        return
    if allowed_fn is None:
        from . import qualify as Q
        materials = Q.judge_materials(funnel_dir)

        def allowed_fn(keys, jq):
            mats = {}
            for j in sorted(jq.get("judges") or {}):
                _scope, mat, _bt = Q.material_for(j, keys, jq, materials)
                mats[j] = mat
            return Q.allowed_judges(keys, mats)
    for g in PPI_GROUPS:
        grp = fr["groups"].get(g)
        if grp is None:
            continue
        by = {}
        for u in grp["units"]:
            by.setdefault(u["stratum"], []).append(u)
        for sid, units in sorted(by.items()):
            aj = sorted(allowed_fn(_keys_of(units), judge_qual))
            grp["strata"][sid]["allowed_judges"] = aj
            joined = ";".join(aj)
            for u in units:
                u["allowed_judges"] = joined


# -------------------------------------------------------------- files
def write_frames(fr, out_dir, header_fn=None):
    """frames_v1.json and frames_v1/<group>.csv. header_fn(inputs, seeds) ->
    header dict (draw passes funnel.header). Refuses to rewrite frames fixed by
    a sample lock (the caller checks the lock and passes locked=True)."""
    out_dir = Path(out_dir)
    csv_dir = out_dir / "frames_v1"
    groups_out = {}
    for g in GROUPS:
        grp = fr["groups"].get(g)
        if grp is None:
            continue
        path = csv_dir / ("%s.csv" % g)
        sha = write_csv_atomic(path, FRAME_FIELDS, [[u[k] for k in FRAME_FIELDS] for u in grp["units"]])
        groups_out[g] = {"design": grp["design"], "N": grp["N"], "planned": grp["planned"],
                         "minimum": grp["minimum"], "file": {"path": str(path), "sha256": sha},
                         "strata": {sid: dict(st) for sid, st in sorted(grp["strata"].items())},
                         "info": grp.get("info", {})}
    body = {"name_status_v2": {"path": fr["name_status_v2"]["path"], "sha256": fr["name_status_v2"]["sha256"]},
            "groups": groups_out, "roles": fr.get("roles"), "reported": fr.get("reported", {}),
            "known_truth_digest": fr.get("known_truth_digest"),
            "exemplar_sessions": {b: {str(k): v for k, v in per.items()}
                                  for b, per in (fr.get("exemplar_sessions") or {}).items()}}
    if header_fn is not None:
        doc = header_fn(fr["inputs"])
        doc.update(body)
    else:
        doc = dict(body, format="funnel-frames/1", inputs=fr["inputs"])
    sha = write_json_atomic(out_dir / "frames_v1.json", doc)
    return {"path": str(out_dir / "frames_v1.json"), "sha256": sha, "doc": doc}


def load_frames(path):
    """frames_v1.json with every group csv re-hashed (StaleInput on a change);
    returns {"doc", "groups": {group: [unit dicts]}}."""
    path = Path(path)
    doc = read_json(path)
    if doc.get("format") != "funnel-frames/1":
        raise StrataError("%s is not a funnel-frames/1" % path)
    groups = {}
    for g, grp in doc.get("groups", {}).items():
        rec = grp.get("file") or {}
        p = Path(rec.get("path", ""))
        if not p.is_absolute() or not p.exists():
            p = path.parent / "frames_v1" / ("%s.csv" % g)
        if not p.exists():
            raise StaleInput("frame file of %s is missing (%s)" % (g, p))
        got = file_record(p)["sha256"]
        if got != rec.get("sha256"):
            raise StaleInput("frame file of %s changed: %s, recorded %s" % (g, got[:12], str(rec.get("sha256"))[:12]))
        header, rows = read_csv(p)
        if tuple(header) != FRAME_FIELDS:
            raise StrataError("%s: columns %s" % (p, header))
        groups[g] = rows
    return {"doc": doc, "groups": groups}


def check_frame_inputs(doc):
    """The frames' recorded inputs still hash as recorded."""
    check_records(doc.get("inputs") or {})


def step1_dir():
    return STEP1_DIR
