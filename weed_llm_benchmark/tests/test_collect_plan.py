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
    recorded GBIF answers here) into the names layer.

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
        test_refusals()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
