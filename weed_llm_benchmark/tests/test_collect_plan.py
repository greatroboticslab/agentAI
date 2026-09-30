#!/usr/bin/env python3
"""Lever L15, discovery (docs/CONTINUOUS_LOOP.md §3.1, §6.8 S1b, §7.1-7.3), on
recorded provider answers.

tests/fixtures/collect/provider_recording.json holds the answers of one
narrow discovery run recorded live on 2026-09-28 (Zenodo, Mendeley Data,
Hugging Face, the annotation index, the record server), trimmed, plus
synthetic Kaggle and Roboflow answers (no credentials on the recording
machine; the fixture says so). A replaying network answers exactly those
requests (method, URL, parameters, body); anything else is a 404.

Pinned:
  * S1b: the known items are audited: PAGS8 is found by the annotation
    index's taxon search, CottonWeedDet3 by the (synthetic) Kaggle search;
    the others are listed as misses ("a discovery defect"), the recall is
    found/known items, and the owner's decision stamp is carried;
  * every known item is a candidate whether a search found it or not
    (fetchable after a recorded miss), MH-Weed16 held as the funnel's;
  * the never-train reference dataset's upload on the annotation index and
    the base's own release are rejected (the exam's title rule is pinned in
    test_collect_prefilter.py), and a release of the evaluation lab
    recognised by its author is lab-grouped and held for the copy scan;
  * a Roboflow project declaring a species with no public source outside
    the evaluation lab is a presumed derivative: lab group, h6_scan hold;
  * the kept candidates are ranked by expected target boxes per GB, with
    declared box counts where the provider gives them;
  * candidates.json carries its format, the queries run per provider (from
    the config's templates, deficit classes interleaved), provider errors
    (a missing credential, a failed query: never fatal) and a pre-check per
    candidate; plan_latest.json points at it; each new source gets one
    candidate event, and a second plan adds none;
  * the queries come only from the config (the deficit list names only
    targets; a non-target refuses);
  * --resolve-names resolves missing names through the authority (the
    recorded GBIF answers here) into the names layer;
  * a source the collector holds for licence_unresolved that a person's
    licence override names is released by plan (a released event with the
    override's decided_by, decided_utc and reason; a candidate again), once;
    a source no override names, or held for another reason, stays held; an
    unresolved candidate under an override records licence_ok true, the
    override's id and the override, its licence record still unresolved and
    no licence hold in its pre-check; an override never lifts a refused
    licence (licence_ok false, no override recorded, still closed).

Run:  python3 tests/test_collect_plan.py
"""
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_plan_")
check, raises = W.check, W.raises
FIX = W.TESTS / "fixtures" / "collect" / "provider_recording.json"


def replay_net(rec):
    from weed_optimizer_framework.tools.collect.transport import Net

    table = {}
    for c in rec["calls"] + rec["synthetic"]:
        key = (c["method"], c["url"], json.dumps(c.get("params") or {}, sort_keys=True), c.get("data"))
        table[key] = c

    class Replay(Net):
        def __init__(self):
            super().__init__({"base_timeout_s": 5, "min_rate_bytes_per_s": 1e9, "read_timeout_s": 5, "attempts": 1})
            self.misses = []

        def _open(self, url, params=None, headers=None, data=None, method=None, timeout=None):
            m = method or ("POST" if data is not None else "GET")
            p = {k: v for k, v in (params or {}).items() if k != "api_key"}
            d = data.decode("utf-8") if isinstance(data, bytes) else data
            c = table.get((m, url, json.dumps(p, sort_keys=True), d))
            if c is None:
                self.misses.append((m, url, p))
                return 404, {}, io.BytesIO(b"{}")
            body = json.dumps(c["json"]).encode("utf-8") if "json" in c else (c.get("text") or "").encode("utf-8")
            return c["status"], dict(c.get("headers") or {}), io.BytesIO(body)

        def cookie(self, name):
            return "recorded-token" if name == "csrftoken" else None

    return Replay()


def configured(rec):
    cfg = W.config()
    cfg.raw["query_grammar"].update(rec["settings"]["query_grammar"])
    return cfg


def test_plan():
    import os
    from weed_optimizer_framework.tools.collect import plan as PL
    from weed_optimizer_framework.tools.collect import state as S
    from weed_optimizer_framework.tools.collect import intake_dir
    print("plan on the recorded answers (S1b)")
    rec = json.loads(FIX.read_text())
    check("the fixture says which answers are synthetic and why", rec["synthetic"] and rec["synthetic_why"]
          and "SYNTHETIC" in rec["why"])
    os.environ["KAGGLE_API_TOKEN"] = "test-token-not-real"
    os.environ["ROBOFLOW_API_KEY"] = "test-key-not-real"
    cfg = configured(rec)
    net = replay_net(rec)
    out = TMP / "lab" / "candidates.json"
    res = PL.plan(cfg, out, classes=rec["settings"]["classes"], providers=rec["settings"]["providers"], net=net,
                  testing=True)
    doc = json.loads(out.read_text())
    by = {c["source_id"]: c for c in doc["candidates"]}
    check("the candidates file has its format and header", doc["format"] == "collect-candidates/1"
          and doc["config"]["sha256"] == cfg.sha256 and doc["testing"] is True)
    rc = doc["recall"]
    found = {f["name"] for f in rc["found"]}
    check("S1b: PAGS8 is found by the annotation index's taxon search", "pags8" in found, rc)
    check("S1b: CottonWeedDet3 is found by the Kaggle search (synthetic answer)", "cottonweeddet3" in found, rc)
    check("the misses are listed as discovery defects, with the owner's decision stamp",
          all("discovery defect" in m["why"] for m in rc["missed"]) and rc["decided_by"] == "owner 2026-09-28 D-C"
          and rc["recall"] == round(len(found) / 6, 4), rc)
    ki = {c["known_item_name"] for c in doc["candidates"] if c.get("known_item_role") == "primary"}
    check("every fetchable known item is a candidate, found or not",
          {"pags8", "mfwd_porol", "cottonweeddet3", "mh_weed16", "cottonweedid15"} <= ki, sorted(ki))
    # the autopilot's recall (inc_autopilot.stream._check_recall) counts as found every row whose known_item
    # is false: that reading must give the collector's own recall
    ids = [it["id"] for it in cfg.known_items()]
    found_rows = {c["id"] for c in doc["candidates"] if not c.get("known_item")}
    f_missed = sorted(i for i in ids if i not in found_rows)
    check("known_item marks exactly the known items no search found, so the autopilot's recall is the collector's",
          f_missed == sorted(m["id"] for m in rc["missed"]), (f_missed, [m["id"] for m in rc["missed"]]))
    tri = {(c["licence"]["class"], c["licence_ok"]) for c in doc["candidates"]}
    check("licence_ok is True for a usable licence, None for an unresolved one (a person decides), never False",
          ("unresolved", None) in tri and ("permissive", True) in tri
          and all(ok is (True if cls in ("permissive", "research_only") else (False if cls == "refused" else None))
                  for cls, ok in tri) and all(c["licence_class"] == c["licence"]["class"] for c in doc["candidates"]),
          sorted(tri, key=str))
    check("names_unresolved is set exactly when class names await L26 (the autopilot routes those to L26)",
          any(c["names_unresolved"] for c in doc["candidates"])
          and all(c["names_unresolved"] == bool(c.get("names_pending")) for c in doc["candidates"]),
          sum(1 for c in doc["candidates"] if c["names_unresolved"]))
    mh = [c for c in doc["candidates"] if c.get("known_item_name") == "mh_weed16"][0]
    check("MH-Weed16 is held as the funnel's (owned_by_funnel), never fetched here",
          not mh["precheck"]["ok"] and "owned_by_funnel" in [f["code"] for f in mh["precheck"]["failures"]])
    pags = by["weedai_5c78d067-8750-4803-9cbe-57df8fae55e4"]
    check("PAGS8 is kept on its declared classes (the pinned card) with its box counts",
          pags["decision"]["status"] == "kept" and pags["target_classes"] == ["PalmerAmaranth"]
          and pags["estimate"]["basis"] == "declared_box_counts" and pags["estimate"]["target_boxes"] == 1940.0,
          (pags["decision"], pags["estimate"]))
    check("PAGS8 is provenance-cleared (TAMU), licence cc-by-4.0, and passes every pre-check",
          pags["hold_until"] is None and pags["lab_group"] == "TAMU" and pags["licence"]["id"] == "cc-by-4.0"
          and pags["precheck"]["ok"], pags["precheck"])
    rej = {sid: c["decision"]["reasons"][0]["code"] for sid, c in by.items() if c["decision"]["status"] == "rejected"}
    cwd = [sid for sid, c in by.items() if "cottonweeddet12" in str(c.get("title")).lower().replace(" ", "")]
    check("the reference dataset's upload on the annotation index is rejected as never-train",
          cwd and all(rej.get(s) == "never_train" for s in cwd), cwd)
    check("the base's own release (3SeasonWeedDet10) is never fetched", rej.get("zenodo_14861516") == "never_train",
          by.get("zenodo_14861516", {}).get("decision"))
    two = by.get("zenodo_10762138")
    check("a release by the evaluation lab's author is lab-grouped (declared) and held for the copy scan",
          two is not None and two["lab_group"] == "LuLab" and two["lab_group_basis"] == "declared"
          and "copy_scan_pending" in [f["code"] for f in two["precheck"]["failures"]],
          two and (two["lab_group"], two["precheck"]))
    pd = by.get("rf_other-ws__mixed-boxes")
    check("a Roboflow project declaring an evaluation-lab-only species is a presumed derivative (h6_scan)",
          pd is not None and pd["lab_group"] == "LuLab" and pd["lab_group_basis"] == "presumed_derivative"
          and pd["hold_until"] == "h6_scan" and pd["decision"]["status"] == "kept",
          pd and (pd["lab_group"], pd["lab_group_basis"], pd["decision"]))
    ok_rf = by.get("rf_some-ws__field-survey")
    check("an ordinary Roboflow project is kept, not lab-grouped, and held for the scan (not provenance-cleared)",
          ok_rf is not None and ok_rf["lab_group"] is None and ok_rf["hold_until"] == "h6_scan"
          and ok_rf["decision"]["status"] == "kept", ok_rf and ok_rf["decision"])
    kept = [c for c in doc["candidates"] if c.get("rank")]
    check("kept candidates are ranked by score (expected target boxes per GB, deficit bonus)",
          [c["rank"] for c in kept] == list(range(1, len(kept) + 1))
          and all(kept[i]["estimate"]["score"] >= kept[i + 1]["estimate"]["score"] for i in range(len(kept) - 1)))
    q = doc["queries"]
    check("queries run per provider from the config's templates, deficit classes interleaved",
          [x["target"] for x in q["weedai"][:3]] == ["PalmerAmaranth", "Sicklepod", "Purslane"]
          and q["weedai"][0]["query"] == "Amaranthus palmeri" and q["zenodo"][0]["query"] == '"Amaranthus palmeri"'
          and len(q["zenodo"]) <= 6, q.get("weedai"))
    check("a failed query or describe is reported per provider, never fatal",
          "kaggle" in doc["provider_errors"] and res["status"] == "planned", doc["provider_errors"].keys())
    check("every candidate carries a pre-check (ok, failures, risk)",
          all(set(c["precheck"]) >= {"ok", "failures", "risk"} for c in doc["candidates"]))
    ptr = json.loads((intake_dir() / "plan_latest.json").read_text())
    import hashlib
    check("plan_latest.json points at the candidates file with its sha256",
          ptr["path"] == str(out) and ptr["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest(), ptr)
    rows = S.read()
    n1 = len([r for r in rows if r["event"] == "candidate"])
    check("one candidate event per new source", n1 == len(doc["candidates"]), (n1, len(doc["candidates"])))
    PL.plan(cfg, out, classes=rec["settings"]["classes"], providers=rec["settings"]["providers"], net=replay_net(rec),
            testing=True)
    n2 = len([r for r in S.read() if r["event"] == "candidate"])
    check("a second plan adds no candidate event for sources it saw", n2 == n1, (n1, n2))
    from weed_optimizer_framework.tools.collect import verify_chain, sources_ledger
    check("sources.jsonl's hash chain verifies", verify_chain(sources_ledger()) == [])
    import funnel_world as FWD
    calls = []
    PL.plan(cfg, TMP / "lab" / "c_names.json", classes=rec["settings"]["classes"],
            providers=rec["settings"]["providers"], net=replay_net(rec), resolve_names=True, testing=True,
            transport=FWD.replay_transport(calls=calls))
    dn = json.loads((TMP / "lab" / "c_names.json").read_text())
    layer = W.TMP / "inc" / "intake" / "names" / "names_cache.json"
    check("--resolve-names asks the authority for the missing names and writes the names layer",
          calls and layer.is_file() and dn["names_layer"]["path"] == str(layer), (len(calls), dn.get("names_layer")))


def test_release():
    from weed_optimizer_framework.tools.collect import plan as PL
    from weed_optimizer_framework.tools.collect import sources_ledger, state as S, verify_chain
    print("a person's licence override releases the collector's licence hold")
    rec = json.loads(FIX.read_text())
    cfg = configured(rec)
    S.append(None, "mediatum_1717366", "held", reason="licence_unresolved", codes=["licence_unresolved"], risk="R3",
             stage="fetch")
    S.append(None, "zenodo_no_override", "held", reason="licence_unresolved", codes=["licence_unresolved"],
             risk="R3", stage="fetch")
    S.append(None, "kg_yuzhenlu__cottonweeddet3", "held", reason="copy_scan_pending", codes=["copy_scan_pending"],
             risk="R3", stage="fetch")
    out = TMP / "lab" / "c_release.json"
    res = PL.plan(cfg, out, classes=rec["settings"]["classes"], providers=rec["settings"]["providers"],
                  net=replay_net(rec), testing=True)
    fold = S.fold(S.read())
    rel = [r for r in S.read() if r["event"] == "released"]
    ov = cfg.raw["licence_overrides"]["mediatum_1717366"]
    check("plan releases the source held for licence_unresolved that an override names: a candidate again",
          res["released"] == ["mediatum_1717366"] and fold["mediatum_1717366"]["status"] == "candidate"
          and fold["mediatum_1717366"]["holds"] == [], (res.get("released"), fold["mediatum_1717366"]))
    check("... the released event carries the override's decided_by, decided_utc and reason",
          len(rel) == 1 and rel[0]["source"] == "mediatum_1717366" and rel[0]["decided_by"] == ov["decided_by"]
          and rel[0]["decided_utc"] == ov["decided_utc"] and rel[0]["reason"] == ov["reason"]
          and rel[0]["codes"] == ["licence_unresolved"] and rel[0]["research_only"] is True, rel)
    check("... a source no override names, or held for another reason, stays held",
          fold["zenodo_no_override"]["status"] == "held" and fold["kg_yuzhenlu__cottonweeddet3"]["status"] == "held",
          (fold["zenodo_no_override"]["status"], fold["kg_yuzhenlu__cottonweeddet3"]["status"]))
    doc = json.loads(out.read_text())
    check("... and the candidates file records it", doc["released"] == ["mediatum_1717366"], doc.get("released"))
    res2 = PL.plan(cfg, out, classes=rec["settings"]["classes"], providers=rec["settings"]["providers"],
                   net=replay_net(rec), testing=True)
    check("idempotent: a second plan releases nothing more",
          res2["released"] == [] and len([r for r in S.read() if r["event"] == "released"]) == 1, res2.get("released"))
    check("sources.jsonl's hash chain still verifies", verify_chain(sources_ledger()) == [])
    unres = sorted(c["source_id"] for c in doc["candidates"] if c["licence"]["class"] == "unresolved"
                   and "licence_unresolved" in [f["code"] for f in c["precheck"]["failures"]])
    check("fixture: the recorded answers hold candidates whose licence is unresolved", unres, unres)
    from weed_optimizer_framework.tools.collect.config import CollectConfig
    raw = json.loads(json.dumps(cfg.raw))
    raw["licence_overrides"] = dict(raw["licence_overrides"], **{unres[0]: dict(ov)})
    cfg2 = CollectConfig(raw, cfg.path, cfg.sha256, cfg.funnel, cfg.eppo, cfg.eppo_record)
    PL.plan(cfg2, out, classes=rec["settings"]["classes"], providers=rec["settings"]["providers"],
            net=replay_net(rec), testing=True)
    by = {c["source_id"]: c for c in json.loads(out.read_text())["candidates"]}
    c, c2 = by[unres[0]], by[unres[1]] if len(unres) > 1 else None
    check("under an override plan records licence_ok true, the override's id and the override; the licence record "
          "and its class stay unresolved; no licence hold in its pre-check",
          c["licence_ok"] is True and c["licence_id"] == "research-only" and c["licence_override"] == ov
          and c["licence_class"] == "unresolved" and c["licence"]["class"] == "unresolved"
          and "licence_unresolved" not in [f["code"] for f in c["precheck"]["failures"]],
          {k: c.get(k) for k in ("licence_ok", "licence_id", "licence_class", "licence_override", "precheck")})
    check("... another unresolved candidate is unchanged (licence_ok None, held)", c2 is None
          or (c2["licence_ok"] is None and c2["licence_override"] is None
              and "licence_unresolved" in [f["code"] for f in c2["precheck"]["failures"]]), c2 and c2["precheck"])
    # a refused licence: the policy refuses cc-by-nc here, and an override names such a candidate
    raw3 = json.loads(json.dumps(raw))
    pol = raw3["licence_policy"]
    pol["research_only"] = [x for x in pol["research_only"] if x != "cc-by-nc"]
    pol["refused"] = pol["refused"] + ["cc-by-nc"]
    refd = sorted(c["source_id"] for c in doc["candidates"] if c["licence"]["id"] == "cc-by-nc-4.0")
    check("fixture: the recorded answers hold a cc-by-nc-4.0 candidate", refd, refd)
    raw3["licence_overrides"] = dict(raw3["licence_overrides"], **{refd[0]: dict(ov)})
    cfg3 = CollectConfig(raw3, cfg.path, cfg.sha256, cfg.funnel, cfg.eppo, cfg.eppo_record)
    PL.plan(cfg3, out, classes=rec["settings"]["classes"], providers=rec["settings"]["providers"],
            net=replay_net(rec), testing=True)
    c3 = {c["source_id"]: c for c in json.loads(out.read_text())["candidates"]}[refd[0]]
    check("an override never lifts a refused licence: licence_ok false, no override recorded, the licence_refused "
          "close still in its pre-check",
          c3["licence"]["class"] == "refused" and c3["licence_ok"] is False and c3["licence_override"] is None
          and c3["licence_id"] == "cc-by-nc-4.0"
          and [f["action"] for f in c3["precheck"]["failures"] if f["code"] == "licence_refused"] == ["close"],
          {k: c3.get(k) for k in ("licence", "licence_ok", "licence_id", "licence_override", "precheck")})


def test_refusals():
    from weed_optimizer_framework.tools.collect import ConfigError, CollectError
    from weed_optimizer_framework.tools.collect import plan as PL
    print("plan refusals")
    rec = json.loads(FIX.read_text())
    cfg = configured(rec)
    e = raises(lambda: PL.plan(cfg, TMP / "x.json", classes=["NotATarget"], net=replay_net(rec)), ConfigError)
    check("a deficit class that is not a target refuses", e is not None and "non-targets" in str(e), e)
    e = raises(lambda: PL.plan(cfg, TMP / "x.json", providers=["nosuch"], net=replay_net(rec)), CollectError)
    check("an unknown provider refuses", e is not None, e)
    import os
    os.environ.pop("KAGGLE_API_TOKEN", None)
    doc_res = PL.plan(cfg, TMP / "y.json", classes=["PalmerAmaranth"], providers=["kaggle", "weedai"],
                      net=replay_net(rec), testing=True)
    doc = json.loads((TMP / "y.json").read_text())
    check("a provider without its credential is skipped and reported (card X16), not fatal",
          doc["provider_errors"]["kaggle"][0]["code"] == "credentials_missing" and doc_res["status"] == "planned",
          doc["provider_errors"].get("kaggle"))


def main():
    try:
        W.build_cache()
        test_plan()
        test_release()
        test_refusals()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
