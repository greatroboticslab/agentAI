#!/usr/bin/env python3
"""Sampling frames, strata and allocation (contract §5.1-5.2; runner §4.7-4.8,
§5.1.4).

Pinned:
  * allocate: sums to n, gives every non-empty stratum min(N_h, floor), never
    exceeds N_h, breaks remainder ties by stratum id, lowers the floor when
    the floors alone exceed n, and takes every unit when n >= the frame;
    allocate_weighted splits by weights under caps;
  * stratum ids follow the key order of runner §4.8 and parse back;
  * every group's definition on the synthetic world, one unit per rule edge:
    a small box is in no frame; a verified box in a vetoed image is in G2v
    (and G2a when its source is outside the reference lab); a named-other
    conflict is in G1's named frame; a target-labelled rejected box is in G2
    with its fail code; an other_ok box is in G4 with its frame, argmax flag
    and score band; a numeric class of >= 100 boxes is a G3 unit, one under
    100 is listed and not sampled; a no-name class is split into 8 visual
    clusters and only clusters of >= 100 boxes are units; reference-lab
    guard pairs never enter G5;
  * G1 allocation: at most 10 in the excluded frame, the rest split equally
    between the no-information and named frames; G3 takes every unit when
    20 M <= planned; G4 plans 225/75 uniform per 300 and the rest Poisson;
  * the G4 bands are tertiles (numpy quantile, linear);
  * frames refuse a name_status_v2.json whose sha256 differs from census's,
    and a ledger that is not census-derived;
  * the frames are deterministic, and write_frames/load_frames round-trip
    with every csv re-hashed (a changed csv is StaleInput);
  * the exemplar sessions equal those sheets.py renders boards from, and
    identity items exclude their class's exemplar sessions.

Run:  python3 tests/test_funnel_strata.py
"""
import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_strata_")
check, raises = W.check, W.raises

import numpy as np  # noqa: E402
from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import strata as S  # noqa: E402

print("allocate")
sizes = {"a": 100, "b": 50, "c": 3, "d": 0, "e": 47}
al = S.allocate(sizes, 60)
check("sums to n", sum(al.values()) == 60, al)
check("floor min(N, 5) for non-empty strata", al["c"] == 3 and all(al[k] >= 5 for k in ("a", "b", "e")))
check("empty stratum gets 0", al["d"] == 0)
check("never above N", all(al[k] <= sizes[k] for k in sizes))
check("proportional rest", al["a"] > al["b"] > al["c"])
al = S.allocate({"x": 10, "y": 10}, 11, floor=0)
check("remainder ties broken by stratum id", al == {"x": 6, "y": 5}, al)
al = S.allocate({"s%02d" % i: 10 for i in range(20)}, 30)
check("floors above n lower the floor", sum(al.values()) == 30 and max(al.values()) - min(al.values()) <= 1, al)
check("n >= frame takes everything", S.allocate({"a": 3, "b": 4}, 100) == {"a": 3, "b": 4})
check("caps respected when a stratum is tiny", S.allocate({"a": 2, "b": 1000}, 500)["a"] == 2)
check("negative n refused", raises(lambda: S.allocate({"a": 1}, -1), F.StrataError))
aw = S.allocate_weighted({"KT1": 100, "KT7": 150}, {"KT1": 1000, "KT7": 1000}, 50)
check("weighted split", aw == {"KT1": 20, "KT7": 30}, aw)
aw = S.allocate_weighted({"KT1": 100, "KT7": 150}, {"KT1": 1000, "KT7": 10}, 50)
check("a capped key's share goes to the others", aw == {"KT1": 40, "KT7": 10}, aw)

print("stratum ids")
sid = S.stratum_id("G1", frame="noinfo", status="numeric", pred="rest")
check("G1 key order", sid == "G1/frame=noinfo/status=numeric/pred=rest")
check("parse back", S.parse_stratum(sid) == ("G1", {"frame": "noinfo", "status": "numeric", "pred": "rest"}))
check("G0 is G0/all", S.stratum_id("G0") == "G0/all" and S.parse_stratum("G0/all") == ("G0", {}))
check("missing key refused", raises(lambda: S.stratum_id("G2", source="x", label="y"), F.StrataError))
check("a '/' in a value refused", raises(lambda: S.stratum_id("G2v", source="a/b"), F.StrataError))
check("seed text", S.seed_text("G2v/source=x") == "funnel/v1/G2v/source=x")

print("G4 bands")
sc = np.linspace(0, 1, 31)
edges, bands = S.g4_bands(sc)
check("edges are the 1/3 and 2/3 quantiles", np.allclose(edges, np.quantile(sc, [1 / 3, 2 / 3])))
check("three bands of about a third each", sorted(np.bincount(bands)[1:].tolist()) == [10, 10, 11])
check("an edge value falls in the lower band", bands[np.argmin(abs(sc - edges[0]))] == 1)

print("frames on the synthetic world")
fd = TMP / "inc" / "funnel"
PREREG = fd / "prereg_v1.json"
raw = json.loads(PREREG.read_text())
raw["sampling"]["groups"] = {"G0": [20, 10], "G1": [30, 10], "G2": [30, 10], "G2v": [8, 4], "G2a": [8, 4],
                             "G3": [60, 20], "G4": [40, 20], "G5": [20, 10], "sentinels": [400, None],
                             "KT7_sentinels": [150, None]}
raw["sampling"]["G4_uniform"] = 20
PREREG.write_text(json.dumps(raw, indent=1))
dom = D.load("weed")
pre = D.load_prereg(PREREG)
adapter = W.build_world(fd, dom, PREREG)
fr = S.frames(pre, dom, fd, adapter, dinov2=W.fake_dinov2)
G = fr["groups"]
rows = {r["id"]: r for r in W.world_rows(dom)}
where = {}
for g, grp in G.items():
    for u in grp["units"]:
        where.setdefault(u["unit_id"], set()).add(g)
small = [i for i, r in rows.items() if r["path"]["S6"] == "small"]
check("small boxes are in no frame", small and not any(i in where for i in small))
check("small boxes reported", fr["reported"]["small_boxes"] == len(small))
vetoed = [i for i, r in rows.items() if r["path"]["S8"] == "verified" and r["path"]["S10"] != "admitted"]
check("verified boxes in vetoed images are in G2v", vetoed and all("G2v" in where[i] for i in vetoed))
check("... and in G2a when outside the reference lab", all("G2a" in where[i] for i in vetoed
                                                           if rows[i]["lab"] != dom.reference_lab()))
ref_ok = [i for i, r in rows.items() if r["path"]["S8"] == "verified" and r["lab"] == dom.reference_lab()]
check("verified admitted reference-lab boxes are in no frame", ref_ok and not any(i in where for i in ref_ok))
named = [u for u in G["G1"]["units"] if S.parse_stratum(u["stratum"])[1]["frame"] == "named"]
check("a named-other conflict is in G1's named frame", named and all(rows[u["unit_id"]]["name_status_v2"] ==
                                                                       "taxon_resolved" for u in named))
g2 = G["G2"]["units"]
check("G2 = target-labelled, embedded, not verified", g2 and all(rows[u["unit_id"]]["path"]["S8"] != "verified"
                                                                 and dom.is_target(rows[u["unit_id"]]["label"])
                                                                 for u in g2))
check("G2 strata carry source, label and fail", all(S.parse_stratum(u["stratum"])[1]["fail"] ==
                                                    (rows[u["unit_id"]]["fail"] or "none") for u in g2))
g4 = G["G4"]["units"]
check("G4 = other_ok", g4 and all(rows[u["unit_id"]]["path"]["S9"] == "other_ok" for u in g4))
check("G4 score is p_target_max", all(float(u["score"]) == rows[u["unit_id"]]["p_target_max"] for u in g4))
for frame in ("named", "noinfo"):
    us = [u for u in g4 if S.parse_stratum(u["stratum"])[1]["frame"] == frame]
    edges, bands = S.g4_bands([float(u["score"]) for u in us])
    check("G4 %s band = tertile of the score within the frame" % frame,
          all(int(S.parse_stratum(u["stratum"])[1]["band"]) == int(b) for u, b in zip(us, bands)))
roles_ = {"size": "S6", "join": "S7", "target_check": "S8", "other_check": "S9", "image_rule": "S10",
          "evidence": "S12"}
grp_x = {g: S._group(g, 10, 1) for g in S.GROUPS}
rows_x = [W._box(dom, "x_img%d" % i, 0, W.AN, dom.other["id"], "*", "", "no_name", oc="other_ok",
                 pred=dom.class_id("Ragweed") if i < 3 else dom.other["id"], p=0.4, cos=0.5, ptm=0.1 * (i + 1),
                 crop_id=i) for i in range(6)]
S._box_frames(dom, rows_x, roles_, grp_x)
flags = {u["unit_id"]: S.parse_stratum(u["stratum"])[1]["argmax_target"] for u in grp_x["G4"]["units"]}
check("G4's argmax_target key is yes exactly when J1's argmax is a target",
      flags == {r["id"]: ("yes" if dom.is_target(r["pred"]) else "no") for r in rows_x} and "yes" in flags.values(),
      flags)
g3 = G["G3"]
units = sorted(g3["strata"])
check("numeric classes of >= 100 boxes are G3 units", units == ["G3/unit=c:%s|12" % W.MH, "G3/unit=c:%s|5" % W.MH],
      units)
listed = {x["unit"] for x in fr["reported"]["G3_listed_not_sampled"]}
check("a numeric class under 100 boxes is listed, not sampled", "c:%s|3" % W.MH in listed)
clusters = sorted(x["unit"] for x in fr["reported"]["G3_listed_not_sampled"] if x["unit"].startswith("k:"))
check("the no-name class is split into 8 visual clusters (all under 100 here: listed)",
      clusters == ["k:%s|*|%d" % (W.AN, j) for j in range(8)], clusters)
check("G3 takes every unit when 20 M <= planned", g3["info"]["take_all_units"] and g3["info"]["per_unit_n"] == 30)
g5 = G["G5"]
check("reference-lab guard pairs never enter G5", not any(u["source"] == W.LU for u in g5["units"]))
check("G5 allocation: 'all' for rf_tuf, 20 for peradeniya", g5["info"]["per_source"] == {W.AN: 13, W.PER: 20})
check("dup twins only when the sha differs", all(json.loads(u["extra"])["kind"] != "exact_dup" or u["source"] == W.ND2
                                                 for u in g5["units"])
      and sum(1 for u in g5["units"] if json.loads(u["extra"])["kind"] == "exact_dup") == 4)
check("G5 strata by source, split and bits band", all(S.parse_stratum(u["stratum"])[1]["bits"] in
                                                      ("0-2", "3-4", "5-6") for u in g5["units"]))
g1 = G["G1"]
check("G1 frame allocation: excluded <= 10, rest split", g1["info"]["frame_n"].get("excluded", 0) <= 10 and
      sum(g1["info"]["frame_n"].values()) == min(30, g1["N"]))
check("G1 pred key: top-3 of the frame or rest", all(S.parse_stratum(s)[1]["pred"] in
                                                     set(g1["info"]["top_pred"][S.parse_stratum(s)[1]["frame"]]) | {"rest"}
                                                     for s in g1["strata"]))
g1x = S._group("G1", 60, 30)
for fr_, st_, n_ in (("excluded", "state", 50), ("noinfo", "no_name", 100), ("named", "taxon_resolved", 100)):
    S._add_stratum(g1x, S.stratum_id("G1", frame=fr_, status=st_, pred="rest"), "x")["N"] = n_
S._plan_g1(g1x, 60)
check("G1: the excluded frame takes at most 10, the rest splits equally (10/25/25 of 60)",
      g1x["info"]["frame_n"] == {"excluded": 10, "noinfo": 25, "named": 25}, g1x["info"]["frame_n"])
g1y = S._group("G1", 60, 30)
for fr_, st_, n_ in (("noinfo", "no_name", 12), ("named", "taxon_resolved", 100)):
    S._add_stratum(g1y, S.stratum_id("G1", frame=fr_, status=st_, pred="rest"), "x")["N"] = n_
S._plan_g1(g1y, 60)
check("G1: a frame smaller than its half passes the rest to the other (12 + 48)",
      g1y["info"]["frame_n"] == {"noinfo": 12, "named": 48}, g1y["info"]["frame_n"])
check("G4 uniform plan 75/25 of G4_uniform", G["G4"]["info"]["uniform"]["named"]["n"] == min(15, 26)
      and G["G4"]["info"]["uniform"]["noinfo"]["n"] == 5 and G["G4"]["info"]["prio"]["n"] == 20)
check("planned n never above N", all(st["n_planned"] <= st["N"] for grp in G.values() for st in grp["strata"].values()))
check("G0 frame parts", set(G["G0"]["info"]["parts_N"]) == {"pos_in_domain", "pos_independent", "neg_independent"})
known_w = W.known_truth(dom)
pos_dom = [u["unit_id"] for u in G["G0"]["units"] if json.loads(u["extra"]).get("part") == "pos_in_domain"]
want_dom = sorted(it["id"] for it in known_w["KT2"] if W.hidden_copy(it))
check("G0's in-domain positives are the reference copies' hidden targets only (%d of %d)"
      % (len(want_dom), len(known_w["KT2"])), sorted(pos_dom) == want_dom and len(want_dom) < len(known_w["KT2"]))


class NoCrops(object):
    def known_truth(self, domain, funnel_dir):
        return known_w


check("an adapter that cannot tell hidden targets refuses the frames",
      raises(lambda: S.frames(pre, dom, fd, NoCrops(), dinov2=W.fake_dinov2), F.StrataError))
check("the crop table is an input of the frames", "crop_table" in fr["inputs"])

print("visual clusters with units of >= 100")
blobs = []
for j, n in enumerate((300, 200, 150, 100, 60, 40, 30, 20)):
    c = np.zeros(8)
    c[j] = 50.0
    blobs.append(np.random.default_rng(j).normal(size=(n, 8)) + c)
X = np.vstack(blobs)
lab = S.visual_clusters(X, "funnel/test/km")
sizes_ = sorted(np.bincount(lab).tolist(), reverse=True)
check("KMeans recovers the 8 blobs", sizes_ == [300, 200, 150, 100, 60, 40, 30, 20], sizes_)
check("clusters are deterministic", (S.visual_clusters(X, "funnel/test/km") == lab).all())
Xn = X.copy()
Xn[0, 0] = np.nan
check("a non-finite row is -1", S.visual_clusters(Xn, "funnel/test/km")[0] == -1)
grp = S._group("G3", 450, 250)
cu = {"status": "no_name", "name": "", "units": [
    dict(S._unit("b:blob%04d#0" % i, "box", "", source=W.AN, crop_id=i, extra={"status": "no_name"}))
    for i in range(len(X))]}
S._g3(dom, {(W.AN, "*"): cu}, grp, lambda ids: X[ids])
unit_sizes = sorted((st["N"] for st in grp["strata"].values()), reverse=True)
check("only clusters of >= 100 boxes become units", unit_sizes == [300, 200, 150, 100], unit_sizes)
check("the smaller clusters are listed", len(grp["info"]["listed_not_sampled"]) == 4)
S._plan_g3(grp, 60)
check("G3 draws floor(planned/20) units when 20 M > planned", grp["info"]["m"] == 3 and
      not grp["info"]["take_all_units"] and grp["info"]["per_unit_n"] == 20)

print("refusals and determinism")
fr2 = S.frames(pre, dom, fd, adapter, dinov2=W.fake_dinov2)
strip = lambda f: {g: [(u["unit_id"], u["stratum"]) for u in grp["units"]] for g, grp in f["groups"].items()}
check("frames are deterministic", strip(fr) == strip(fr2) and
      {g: {s: st["n_planned"] for s, st in grp["strata"].items()} for g, grp in fr["groups"].items()} ==
      {g: {s: st["n_planned"] for s, st in grp["strata"].items()} for g, grp in fr2["groups"].items()})
ns = fd / "name_status_v2.json"
keep = ns.read_bytes()
ns.write_bytes(keep + b"\n")
check("a name_status_v2.json other than census's refuses", raises(lambda: S.frames(pre, dom, fd, adapter,
                                                                                   dinov2=W.fake_dinov2),
                                                                  F.StaleInput))
ns.write_bytes(keep)
fl = fd / "funnel_ledger.json"
keep_fl = fl.read_bytes()
led = json.loads(keep_fl)
led["derivation"] = "summaries"
fl.write_text(json.dumps(led))
check("a summaries-derived ledger refuses", raises(lambda: S.frames(pre, dom, fd, adapter, dinov2=W.fake_dinov2),
                                                   F.StrataError))
fl.write_bytes(keep_fl)
check("a no-name class without features refuses", raises(lambda: S.frames(pre, dom, fd, adapter), F.StrataError))
cp = fd / "census_v1.json"
keep_c = cp.read_bytes()
cj = json.loads(keep_c)
cj["reconciliation"] = {"ok": False, "checks": []}
cp.write_text(json.dumps(cj))
check("a census that does not reconcile refuses", raises(lambda: S.frames(pre, dom, fd, adapter,
                                                                          dinov2=W.fake_dinov2), F.StrataError))
cj = json.loads(keep_c)
cj["prereg"]["core_sha256"] = "0" * 64
cp.write_text(json.dumps(cj))
check("a census made under another prereg core refuses (StaleInput, runner §3.3)",
      raises(lambda: S.frames(pre, dom, fd, adapter, dinov2=W.fake_dinov2), F.StaleInput))
cj.pop("prereg")
cp.write_text(json.dumps(cj))
check("a census that records no prereg refuses", raises(lambda: S.frames(pre, dom, fd, adapter,
                                                                         dinov2=W.fake_dinov2), F.StaleInput))
cp.write_bytes(keep_c)
led = json.loads(keep_fl)
led["prereg"]["core_sha256"] = "0" * 64
fl.write_text(json.dumps(led))
check("a funnel ledger made under another prereg core refuses",
      raises(lambda: S.frames(pre, dom, fd, adapter, dinov2=W.fake_dinov2), F.StaleInput))
fl.write_bytes(keep_fl)

print("write and load")
out = S.write_frames(fr, TMP / "frames_out")
lf = S.load_frames(out["path"])
check("frames_v1.json names every group", set(lf["doc"]["groups"]) == set(S.GROUPS) - {"sentinel", "pair_sentinel"}
      or set(lf["doc"]["groups"]) >= {"G0", "G1", "G2", "G2v", "G2a", "G3", "G4", "G5", "identity"})
check("csv rows round trip", [r["unit_id"] for r in lf["groups"]["G2"]] == [u["unit_id"] for u in G["G2"]["units"]])
check("csv columns", list(lf["groups"]["G1"][0]) == list(S.FRAME_FIELDS))
csvp = TMP / "frames_out" / "frames_v1" / "G2.csv"
csvp.write_text(csvp.read_text() + "x\n")
check("a changed frame csv is StaleInput", raises(lambda: S.load_frames(out["path"]), F.StaleInput))

print("allowed judges")
mat_doc = {"format": "funnel-judge-material/1",
           "judges": {"J-knn2": {"all": {"source": [W.ND], "near_dup3": [], "provenance": [],
                                         "lab": [dom.lab_of(W.ND)]}}}}
msha = F.write_json_atomic(fd / "judge_material_v1.json", mat_doc)
jq = {"format": "funnel-judge-qualification/1", "judges": {"J-knn2": {"by_type": {}, "by_lab_scope": {},
                                                                      "calibration_material": {}}},
      "material": {"path": str(fd / "judge_material_v1.json"), "sha256": msha}}
F.write_json_atomic(fd / "judge_qualification.json", jq)
fj = S.frames(pre, dom, fd, adapter, judge_qual=jq, dinov2=W.fake_dinov2)
(fd / "judge_qualification.json").unlink()
check("the judge files are inputs of the frames", "judge_qualification" in fj["inputs"])
nd_strata = [s for g in S.PPI_GROUPS for s, st in fj["groups"][g]["strata"].items()
             if any(u["source"] == W.ND for u in fj["groups"][g]["units"] if u["stratum"] == s)]
other = [s for g in S.PPI_GROUPS for s, st in fj["groups"][g]["strata"].items() if s not in nd_strata]
check("a judge calibrated on a stratum's lab is not allowed there",
      nd_strata and all(fj["groups"][s.split("/")[0]]["strata"][s]["allowed_judges"] == [] for s in nd_strata))
check("it is allowed on strata that share nothing with its material",
      other and all(fj["groups"][s.split("/")[0]]["strata"][s]["allowed_judges"] == ["J-knn2"] for s in other))
check("the csv column carries the allowed judges", all(u["allowed_judges"] == ";".join(
    fj["groups"]["G4"]["strata"][u["stratum"]]["allowed_judges"]) for u in fj["groups"]["G4"]["units"]))
calls = []
S.set_allowed_judges(fj, jq, allowed_fn=lambda keys, q: calls.append(keys) or ["X"])
check("an injected allowed_fn is called once per estimation stratum",
      len(calls) == sum(len(fj["groups"][g]["strata"]) for g in S.PPI_GROUPS))

print("exemplar sessions")
known = W.known_truth(dom)
sess = S.exemplar_sessions(dom, known)
try:
    from weed_optimizer_framework.tools.funnel import sheets as SH
    plans = SH.board_plans(dom, known)
    theirs = {b: {int(k): v for k, v in p["sessions"].items()} for b, p in plans.items() if p["sessions"]}
    check("exemplar sessions = the sessions sheets.py draws board exemplars from", sess == theirs, (sess, theirs))
except ImportError as e:
    W.skip("exemplar sessions against sheets.py", "sheets.py not importable (%s)" % e)
by_cls = S.exemplar_sessions_by_class(dom, sess)
ident = G["identity"]["units"]
check("identity items exclude their class's exemplar sessions",
      ident and not any(S.in_exemplar_session({"truth": dom.class_id("Ragweed"), "truth_kind": "target",
                                               "session": [it for it in known["KT1"] if it["id"] == u["unit_id"]][0]
                                               ["session"]}, by_cls) for u in ident))

W.finish()
