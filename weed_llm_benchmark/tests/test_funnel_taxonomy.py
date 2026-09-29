#!/usr/bin/env python3
"""The taxonomy authority and its cache (docs/FUNNEL_AUDIT.md §4.3 J-taxon, §8.5
L12; runner §4.5, §5.2.3).

What is pinned, and why:
  * the resolver is offline by default and refuses a miss with the lever-L12
    message (the cluster has no network; census must never guess a name);
  * the queries go to the config's URLs with the GBIF parameter names checked
    on 2026-09-28 (match: name, strict; search: q, qField=VERNACULAR, limit,
    datasetKey), and every answer is cached with the sha256 of its body;
  * overrides (project policy, R4) win over the authority and are recorded;
  * a vernacular-only match is informative but never mappable;
  * is_target_synonym accepts a synonym (Amaranthus rudis -> Waterhemp), a
    variety of the target and a species of a genus-rank target, and refuses a
    sibling species, which is_relative accepts (same genus, or a "not" taxon);
  * load_cache refuses a cache built against another authority URL or with
    other overrides; the cache holds every taxon the config names;
  * query variants: separators, a leading numeric token, "sp."/"spp.".

Every authority answer comes from tests/fixtures/funnel/gbif_recording.json
(real GBIF answers recorded on 2026-09-28) or from a fake transport; the
socket is blocked.

Run:  python3 tests/test_funnel_taxonomy.py
"""
import copy
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import funnel_prereg as FPR  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_taxonomy_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
(TMP / "repo" / "docs").mkdir(parents=True)
(TMP / "inc" / "funnel").mkdir(parents=True)
shutil.copyfile(ROOT.parent / "docs" / "FUNNEL_AUDIT.md", TMP / "repo" / "docs" / "FUNNEL_AUDIT.md")
FPR.write_pre_draw(TMP / "inc" / "funnel" / "prereg_v1.json",
                   ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json")


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise AssertionError("the test touched the network")


socket.socket = _NoNet

import funnel_world as FW  # noqa: E402
from weed_optimizer_framework.tools.funnel import TaxonomyError  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import names as N  # noqa: E402
from weed_optimizer_framework.tools.funnel import taxonomy as T  # noqa: E402

FAILURES, SKIPS = [], []
FD = TMP / "inc" / "funnel"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return None


DOM = D.load("weed")
PRE = D.load_prereg(FD / "prereg_v1.json")
NAMES = ["Amaranthus rudis", "Redroot Pigweed", "BroWeed", "Kochia", "Taraxacum_officinale", "Eclipta alba",
         "0 ridderzuring", "Giant ragweed", "Cassia obtusifolia"]


def build(names=NAMES, domain=DOM, out=None, transport=None):
    calls = []
    tr = transport or FW.replay_transport(calls=calls)
    out = out or FD / "taxonomy_cache.json"
    cache = T.build_cache(names, domain, tr, out, prereg=PRE, testing=True)
    return cache, calls, out


def test_build_and_queries():
    print("build_cache through the recorded answers")
    cache, calls, out = build()
    urls = {u for u, _p in calls}
    auth = DOM.section("taxonomy")["authority"]
    check("scientific lookups go to the config's match_url with name and strict=true",
          any(u == auth["match_url"] and p.get("strict") == "true" and "name" in p for u, p in calls), calls[:3])
    check("vernacular lookups go to search_url with q, qField=VERNACULAR, limit and the backbone datasetKey",
          any(u == auth["search_url"] and p.get("qField") == "VERNACULAR" and "q" in p and p.get("limit") == "20"
              and p.get("datasetKey") == auth["search_params"]["datasetKey"] for u, p in calls))
    check("the backbone version is read from version_url", auth["version_url"] in urls
          and cache["authority"]["backbone_version"])
    check("the cache has the funnel header and format", cache["format"] == "funnel-taxonomy-cache/1"
          and cache["prereg"]["core_sha256"] == PRE.core_sha256 and cache["domain_config"]["sha256"] == DOM.sha256)
    check("every response records the sha256 of its body", all(
        len(r["sha256"]) == 64 and isinstance(r["body"], dict) for r in cache["responses"].values()))
    want = {N.key(t) for t in T.domain_taxa(DOM)}
    check("every target, 'not', attractor, KT7 and card taxon is resolved", want <= set(cache["resolved"]),
          sorted(want - set(cache["resolved"]))[:5])
    check("every requested name is resolved", {N.key(n) for n in NAMES} <= set(cache["resolved"]))
    check("the targets' kingdom is recorded", cache["kingdom"] == "Plantae")
    again, _c, _o = build(out=FD / "again.json")
    check("the build is deterministic apart from times", again["resolved"] == cache["resolved"])
    loaded = T.load_cache(out, DOM)
    check("load_cache accepts the file it wrote", loaded["resolved"] == cache["resolved"])


def test_offline():
    print("offline by default")
    cache = T.load_cache(FD / "taxonomy_cache.json", DOM)
    r = T.Resolver(cache, DOM)
    check("a cached name resolves offline", r.resolve("Amaranthus rudis")["via"] == "scientific")
    msg = raises(lambda: r.resolve("Spermacoce hispida"), TaxonomyError)
    check("a miss refuses, naming lever L12", msg is not None and "lever L12" in msg and "fetch --what taxonomy" in msg,
          msg)
    msg = raises(lambda: r.check_complete(["Amaranthus rudis", "Alternanthera pungens", "Leptadenia reticulata"]),
                 TaxonomyError)
    check("check_complete counts the missing names", msg is not None and "2 name(s)" in msg, msg)
    r2 = T.Resolver({"responses": {}, "resolved": {}}, DOM, transport=None, allow_network=True)
    check("allow_network without a transport stays offline",
          raises(lambda: r2.resolve("Kochia"), TaxonomyError) is not None)


def test_resolution_rules():
    print("resolution rules")
    cache = T.load_cache(FD / "taxonomy_cache.json", DOM)
    r = T.Resolver(cache, DOM)
    wh = DOM.target("Waterhemp")
    pa = DOM.target("PalmerAmaranth")
    mg = DOM.target("MorningGlory")
    ec = DOM.target("Eclipta")
    rud = r.resolve("Amaranthus rudis")
    check("Amaranthus rudis is a synonym of Waterhemp's taxon", r.is_target_synonym(rud, wh)
          and rud["accepted"] == "Amaranthus tuberculatus", rud)
    check("... and not of Palmer amaranth; it is Palmer amaranth's relative (same genus)",
          not r.is_target_synonym(rud, pa) and r.is_relative(rud, pa))
    alba = r.resolve("Eclipta alba")
    check("the override wins: Eclipta alba resolves to Eclipta prostrata via override",
          alba["via"] == "override" and alba["canonical"] == "Eclipta prostrata" and alba["match_type"] == "OVERRIDE",
          alba)
    check("an override is mappable: Eclipta alba is Eclipta's synonym", r.is_target_synonym(alba, ec))
    rr = r.resolve("Redroot Pigweed")
    check("a vernacular-only match is informative (via vernacular) ...", rr["via"] == "vernacular"
          and rr["accepted"] == "Amaranthus retroflexus", rr)
    check("... never mappable", not any(r.is_target_synonym(rr, t) for t in DOM.targets))
    check("... and a relative of both Amaranthus targets (their 'not' list and genus)",
          r.is_relative(rr, wh) and r.is_relative(rr, pa))
    gr = r.resolve("Giant ragweed")
    check("Giant ragweed is Ragweed's relative, never its synonym",
          r.is_relative(gr, DOM.target("Ragweed")) and not r.is_target_synonym(gr, DOM.target("Ragweed")), gr)
    check("an unresolvable name is nobody's synonym or relative",
          r.resolve("BroWeed")["via"] == "none" and not any(r.is_relative(r.resolve("BroWeed"), t)
                                                           for t in DOM.targets))
    ipo = r.resolve("Ipomoea hederacea") if N.key("Ipomoea hederacea") in cache["resolved"] else None
    if ipo is None:
        c2, _calls, _o = build(names=["Ipomoea hederacea"], out=FD / "ipo.json")
        ipo = T.Resolver(c2, DOM).resolve("Ipomoea hederacea")
    check("a species of a genus-rank target is its synonym (Ipomoea hederacea -> MorningGlory)",
          r.is_target_synonym(ipo, mg), ipo)
    print("variety and sibling, from answers in GBIF's match format")
    variety = {"usageKey": 5548360, "acceptedUsageKey": 8577467,
               "scientificName": "Amaranthus tuberculatus var. rudis (J.D.Sauer) Costea & Tardif",
               "canonicalName": "Amaranthus tuberculatus rudis", "rank": "VARIETY", "status": "SYNONYM",
               "confidence": 99, "matchType": "EXACT", "kingdom": "Plantae", "phylum": "Tracheophyta",
               "order": "Caryophyllales", "family": "Amaranthaceae", "genus": "Amaranthus",
               "species": "Amaranthus tuberculatus", "speciesKey": 8577467, "class": "Magnoliopsida"}
    e = T.entry_from_match("Amaranthus tuberculatus var. rudis", variety, "scientific")
    check("a variety of the target species is its synonym", r.is_target_synonym(e, wh), e)
    sib = r.resolve("Amaranthus palmeri")
    check("a sibling species is refused as a synonym (Palmer amaranth for Waterhemp)",
          not r.is_target_synonym(sib, wh) and r.is_relative(sib, wh))
    cp = r.resolve("Chamaecrista pumila")
    check("a taxon of a target's 'not' list outside its genus is its relative (Chamaecrista pumila for "
          "Sicklepod, Senna)", cp["via"] == "scientific" and r.is_relative(cp, DOM.target("Sicklepod")), cp)
    fake_v = dict(T.empty_entry("carelessweed"), via="vernacular", canonical="Amaranthus palmeri",
                  accepted="Amaranthus palmeri", accepted_key=sib["accepted_key"], rank="species",
                  lineage=dict(sib["lineage"]), vernacular_candidates=["Amaranthus palmeri"])
    check("a vernacular match whose first candidate IS a target species is still no synonym (a common "
          "name alone never maps)", not r.is_target_synonym(fake_v, pa) and r.is_relative(fake_v, wh))

    print("only an EXACT match at genus rank or below is a scientific resolution")

    def one_answer(body):
        def tr(url, params):
            return 200, json.dumps(body).encode("utf-8"), {}
        return T.Resolver({"responses": {}, "resolved": {}}, DOM, transport=tr, allow_network=True)
    for body, what in (({"matchType": "HIGHERRANK", "rank": "GENUS", "canonicalName": "Foo"}, "a higher-rank match"),
                       ({"matchType": "FUZZY", "rank": "SPECIES", "canonicalName": "Foo bar"}, "a fuzzy match"),
                       ({"matchType": "EXACT", "rank": "FAMILY", "canonicalName": "Fooaceae"}, "a family")):
        check("%s is not a scientific resolution" % what, one_answer(body)._scientific("Foo bar") is None)
    ok = one_answer({"matchType": "EXACT", "rank": "SPECIES", "canonicalName": "Foo bar", "usageKey": 1,
                     "kingdom": "Plantae", "genus": "Foo", "species": "Foo bar"})._scientific("Foo bar")
    check("an EXACT species match is", ok is not None and ok["accepted"] == "Foo bar", ok)
    print("config taxa that are homonyms across kingdoms")
    for g in ("Digitaria", "Phaseolus"):
        e = r.resolve(g)
        check("%s (the plain strict match answers 'Multiple equal matches') resolves in the targets' kingdom, "
              "at genus rank" % g, e["via"] == "scientific" and e["rank"] == "genus"
              and e["lineage"]["kingdom"] == "Plantae" and e["accepted"] == g, e)
    check("the zero-shot prompt's lineage of a homonym genus attractor exists",
          r.lineage_string("Digitaria").endswith("Poaceae Digitaria"), r.lineage_string("Digitaria"))
    check("every taxon the config names resolves in the cache",
          all(cache["resolved"][N.key(t)]["via"] != "none" for t in T.domain_taxa(DOM)),
          [t for t in T.domain_taxa(DOM) if cache["resolved"][N.key(t)]["via"] == "none"])
    check("the config's project override wins for 'Kochia' (Bassia scoparia, R4, contract 4.3)",
          r.resolve("Kochia")["via"] == "override" and r.resolve("Kochia")["accepted"] == "Bassia scoparia",
          r.resolve("Kochia"))
    # Without that override the same name shows the rule the override exists for: a source class
    # name that is a homonym is not read in the targets' kingdom, so 'Kochia' stays unresolved
    # scientifically and resolves only by its common name (to the wrong genus, Neokochia).
    raw = json.loads(D.resolve_path("weed").read_text())
    del raw["taxonomy"]["overrides"]["kochia"]
    p2 = TMP / "weed_no_kochia_override.json"
    p2.write_text(json.dumps(raw))
    dom2 = D.load(p2)
    c2, _calls2, _o2 = build(domain=dom2, out=TMP / "cache_no_kochia_override.json")
    k2 = T.Resolver(c2, dom2).resolve("Kochia")
    check("without the override, 'Kochia' is not read in the targets' kingdom (vernacular only)",
          k2["via"] == "vernacular" and k2["accepted"] != "Bassia scoparia", k2)
    check("lineage string of a species target", r.lineage_string("Amaranthus tuberculatus") ==
          "Plantae Tracheophyta Magnoliopsida Caryophyllales Amaranthaceae Amaranthus tuberculatus",
          r.lineage_string("Amaranthus tuberculatus"))
    check("lineage string of a genus target ends with the genus",
          r.lineage_string("Ipomoea").endswith("Convolvulaceae Ipomoea"), r.lineage_string("Ipomoea"))


def test_query_variants():
    print("query variants")
    check("a leading numeric token is dropped after the name as written",
          T.query_variants("0 ridderzuring") == ["0 ridderzuring", "ridderzuring"], T.query_variants("0 ridderzuring"))
    check("underscores become spaces", T.query_variants("Taraxacum_officinale") == ["Taraxacum officinale"])
    check("'Genus SP' also queries the genus", T.query_variants("Digitaria SP") == ["Digitaria SP", "Digitaria"])
    check("a lone number is left as it is", T.query_variants("12") == ["12"])
    check("an empty name has no query", T.query_variants("") == [])
    cache = T.load_cache(FD / "taxonomy_cache.json", DOM)
    got = T.Resolver(cache, DOM).resolve("0 ridderzuring")
    check("'0 ridderzuring' resolves through its stripped variant", got["via"] == "vernacular"
          and got["accepted"] == "Rumex obtusifolius", got)


def test_load_refusals():
    print("load_cache refusals")
    raw = copy.deepcopy(DOM.raw)
    raw["taxonomy"]["overrides"]["ecliptaprostratavar"] = {"taxon": "Eclipta prostrata", "why": "test"}
    p = TMP / "weed_edited.json"
    p.write_text(json.dumps(raw))
    edited = D.load(str(p))
    msg = raises(lambda: T.load_cache(FD / "taxonomy_cache.json", edited), TaxonomyError)
    check("a cache built with other overrides is refused", msg is not None and "overrides" in msg, msg)
    raw2 = copy.deepcopy(DOM.raw)
    raw2["taxonomy"]["authority"]["match_url"] = "https://example.org/match"
    p2 = TMP / "weed_edited2.json"
    p2.write_text(json.dumps(raw2))
    msg = raises(lambda: T.load_cache(FD / "taxonomy_cache.json", D.load(str(p2))), TaxonomyError)
    check("a cache built against another authority URL is refused", msg is not None and "match_url" in msg, msg)
    msg = raises(lambda: T.load_cache(FD / "nothing.json", DOM), TaxonomyError)
    check("a missing cache refuses, naming lever L12", msg is not None and "L12" in msg, msg)
    bad = json.load(open(FD / "taxonomy_cache.json"))
    bad["format"] = "something-else/1"
    (FD / "bad.json").write_text(json.dumps(bad))
    check("a file of another format is refused",
          raises(lambda: T.load_cache(FD / "bad.json", DOM), TaxonomyError) is not None)
    msg = raises(lambda: T.build_cache(["x"], DOM, FW.replay_transport(), FD / "x.json"), TaxonomyError)
    check("build_cache needs the prereg for its header", msg is not None and "prereg" in msg, msg)

    raw3 = copy.deepcopy(DOM.raw)
    raw3["attractors"].append(dict(raw3["attractors"][0], id="A99", taxon="BroWeed", common="Test attractor",
                                   option="Test attractor (BroWeed)"))
    p3 = TMP / "weed_edited3.json"
    p3.write_text(json.dumps(raw3))
    msg = raises(lambda: T.build_cache(["Kochia"], D.load(str(p3)), FW.replay_transport(), FD / "z.json",
                                       prereg=PRE), TaxonomyError)
    check("a taxon the config names that the authority does not resolve refuses the build (no literal "
          "stand-in)", msg is not None and "BroWeed" in msg and "config" in msg, msg)

    def failing(url, params):
        return 503, b"busy", {}
    msg = raises(lambda: T.build_cache(["Kochia"], DOM, failing, FD / "y.json", prereg=PRE), TaxonomyError)
    check("an authority error refuses the build (nothing is guessed)", msg is not None and "503" in msg, msg)


if __name__ == "__main__":
    try:
        test_build_and_queries()
        test_offline()
        test_resolution_rules()
        test_query_variants()
        test_load_refusals()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)
