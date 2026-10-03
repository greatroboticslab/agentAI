#!/usr/bin/env python3
"""The class map (docs/CONTINUOUS_LOOP.md §3.2 "Class map"; group D acceptance
"Class map") and the names layer of lever L26.

Pinned, through the funnel's offline taxonomy cache (built from the recorded
GBIF answers) and funnel.names.status_v2:
  * the MH-Weed16 card (the funnel domain's card resolver, read only): id 5
    (Ipomoea obscura) -> MorningGlory under the genus-rank target (DEC-3),
    id 12 (Senna obtusifolia) -> Sicklepod, every other id -> the reject class;
  * numeric names without a card -> the unmapped id 13, as are generic words
    and objects; an unmapped id is never a target or the reject class;
  * an EPPO code: POROL -> Purslane, CHEAL -> the reject class (the pinned
    EPPO table, then the resolver);
  * a WeedCOCO "<role>: <taxon> (<stage>)" name -> its taxon's target, and the
    PAGS8 card table (pinned in the collector config) maps its eight growth
    stages to Palmer amaranth;
  * the alias table's join (a plural legacy label) -> the target;
  * a relative named at genus rank (a target's genus) is unmapped, a relative
    at species rank is the reject class, and a vernacular relative is
    unmapped (a common name never maps);
  * a name the caches lack is pending (never guessed): the class map lists it,
    and lever L26 (collect names) resolves it through the authority on the
    lab into the names layer, after which the same class maps; the funnel's
    cache file is never written;
  * provenance: the card's sha256 and origin, the taxonomy cache's and names
    layer's sha256, the EPPO table, the status map and the funnel domain, and
    the status of each name.

Run:  python3 tests/test_collect_classmap.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_classmap_")
check, raises = W.check, W.raises
MH = "project_agml__mh_weed16_weed_detection"
PAGS = "weedai_5c78d067-8750-4803-9cbe-57df8fae55e4"


def cls(names, hints=None):
    return [{"id": i, "name": n, "hints": (hints or {}).get(n, [])} for i, n in enumerate(names)]


def test_map():
    from weed_optimizer_framework.tools.collect import classmap as CM
    from weed_optimizer_framework.tools.collect import names as NM
    from weed_optimizer_framework.tools.collect.targets import Targets
    print("class map")
    cfg = W.config()
    W.build_cache()
    nm = NM.load(cfg)
    tg = Targets(cfg, nm)
    ids = {t["name"]: t["id"] for t in cfg.targets}
    mh = CM.build(MH, cls([str(i) for i in range(16)]), cfg, nm, tg)
    by = {c["src_id"]: c for c in mh["classes"]}
    check("MH-Weed16 card id 5 -> MorningGlory (Ipomoea obscura under the genus target, DEC-3)",
          by[5]["inc_id"] == ids["MorningGlory"] and by[5]["basis"] == "card" and by[5]["status"] == "target_synonym",
          by[5])
    check("MH-Weed16 card id 12 -> Sicklepod", by[12]["inc_id"] == ids["Sicklepod"] and by[12]["basis"] == "card",
          by[12])
    check("the other card ids -> the reject class", all(by[i]["inc_id"] == cfg.other_id for i in by if i not in (5, 12)),
          {i: by[i]["inc_id"] for i in by})
    check("the card's origin and sha256 are recorded (the funnel domain's resolver, read only)",
          mh["provenance"]["card"]["origin"] == "funnel_config" and len(mh["provenance"]["card"]["sha256"]) == 64)
    m = CM.build("other_src", cls(["0", "12", "weed", "car", "POROL", "CHEAL", "Carpetweeds", "Crabgrass",
                                   "weed: amaranthus palmeri (BBCH10-12)", "Redroot Pigweed"]), cfg, nm, tg)
    b = {c["name"]: c for c in m["classes"]}
    check("numeric names without a card -> 13 (unmapped)", b["0"]["inc_id"] == 13 and b["12"]["inc_id"] == 13
          and b["0"]["status"] == "numeric", (b["0"], b["12"]))
    check("generic and object names -> 13", b["weed"]["inc_id"] == 13 and b["car"]["inc_id"] == 13)
    check("the unmapped id is neither a target nor the reject class",
          cfg.unmapped_id == 13 and 13 not in ids.values() and cfg.other_id == 12)
    check("EPPO POROL -> Purslane", b["POROL"]["inc_id"] == ids["Purslane"] and b["POROL"]["basis"] == "eppo", b["POROL"])
    check("EPPO CHEAL -> the reject class", b["CHEAL"]["inc_id"] == cfg.other_id and b["CHEAL"]["taxon"]
          == "Chenopodium album", b["CHEAL"])
    check("the alias table's join maps a plural legacy label", b["Carpetweeds"]["inc_id"] == ids["Carpetweed"]
          and b["Carpetweeds"]["via"] == "join")
    check("a named non-target (an override) -> the reject class", b["Crabgrass"]["inc_id"] == cfg.other_id)
    check("a '<role>: <taxon> (<stage>)' name maps by its taxon",
          b["weed: amaranthus palmeri (BBCH10-12)"]["inc_id"] == ids["PalmerAmaranth"]
          and b["weed: amaranthus palmeri (BBCH10-12)"]["basis"] == "hint")
    check("a vernacular relative is unmapped (a common name never maps)", b["Redroot Pigweed"]["inc_id"] == 13
          and b["Redroot Pigweed"]["via"] == "vernacular", b["Redroot Pigweed"])
    # zenodo_15808623 (SIU Weed Growth Stage) names its 174 classes "<EPPO>_week_<n>"
    siu = CM.build("zenodo_15808623", cls(["AMAPA_week_5", "CHEAL_week_11", "AMAPA week 2"]), cfg, nm, tg)
    sb = {c["name"]: c for c in siu["classes"]}
    check("an EPPO code leading a name maps by its binomial: AMAPA_week_5 -> PalmerAmaranth (basis eppo_prefix)",
          sb["AMAPA_week_5"]["inc_id"] == ids["PalmerAmaranth"] and sb["AMAPA_week_5"]["basis"] == "eppo_prefix"
          and sb["AMAPA week 2"]["inc_id"] == ids["PalmerAmaranth"], sb["AMAPA_week_5"])
    check("  CHEAL_week_11 -> the reject class (Chenopodium album)", sb["CHEAL_week_11"]["inc_id"] == cfg.other_id
          and sb["CHEAL_week_11"]["taxon"] == "Chenopodium album" and not siu["pending"], sb["CHEAL_week_11"])
    check("  a leading token the EPPO table lacks, a lower-case code or a code without a separator is no EPPO prefix",
          CM.eppo_prefix_binomial(cfg, "WEEDS_1") is None and CM.eppo_prefix_binomial(cfg, "amapa_week_5") is None
          and CM.eppo_prefix_binomial(cfg, "AMAPAX_1") is None and CM.eppo_prefix_binomial(cfg, None) is None
          and CM.eppo_prefix_binomial(cfg, "SETFA_week_3") == "Setaria faberi")
    hb = CM.build("zenodo_15808623", cls(["AMAPA_x"], hints={"AMAPA_x": ["CHEAL"]}), cfg, nm, tg)["classes"][0]
    check("  a hint the format carries outranks a code read off the name's first token",
          hb["basis"] == "hint" and hb["inc_id"] == cfg.other_id, hb)
    check("  the EPPO table v2 holds the five SIU codes v1 lacked",
          [cfg.eppo_binomial(c) for c in ("ABUTH", "PANDI", "SETFA", "SETPU", "SORHA")]
          == ["Abutilon theophrasti", "Panicum dichotomiflorum", "Setaria faberi", "Setaria pumila",
              "Sorghum halepense"] and cfg.eppo_record["version"] == "v2")
    pags = CM.build(PAGS, cls(["weed: amaranthus palmeri (BBCH60-69 - bushy)", "weed: amaranthus palmeri (BBCH10-12)"]),
                    cfg, nm, tg)
    check("the PAGS8 card table maps its growth stages to Palmer amaranth",
          all(c["inc_id"] == ids["PalmerAmaranth"] and c["basis"] == "card" for c in pags["classes"])
          and pags["provenance"]["card"]["origin"] == "collect_config", pags["classes"])
    prov = m["provenance"]
    check("provenance: taxonomy cache, EPPO table, status map, funnel domain and every name's status",
          prov["taxonomy_cache"]["sha256"] and prov["eppo"]["sha256"] == cfg.raw["eppo"]["sha256"]
          and len(prov["status_map_sha256"]) == 64 and prov["funnel_domain"]["sha256"] == cfg.funnel.sha256
          and prov["statuses"]["0"] == "numeric", prov)
    return cfg


def test_rank_rule():
    from weed_optimizer_framework.tools.collect import classmap as CM
    print("relatives by rank")
    cfg = W.config()
    by_name = {t["name"]: t for t in cfg.targets}
    st_genus = {"status": "target_related", "via": "scientific", "rank": "genus", "target": None}
    st_sp = {"status": "target_related", "via": "scientific", "rank": "species", "target": None}
    check("a relative named at genus rank (a target's genus) is unmapped", CM.inc_of(st_genus, cfg, by_name)[0] == 13)
    check("a relative at species rank is the reject class", CM.inc_of(st_sp, cfg, by_name)[0] == cfg.other_id)
    amb = {"status": "target_synonym", "via": "scientific", "rank": "species", "target": None, "ambiguous": True}
    check("an ambiguous target synonym is unmapped", CM.inc_of(amb, cfg, by_name)[0] == 13)


def test_pending_and_l26():
    import hashlib
    from weed_optimizer_framework.tools.collect import classmap as CM
    from weed_optimizer_framework.tools.collect import names as NM
    from weed_optimizer_framework.tools.collect.targets import Targets
    import funnel_world as FWD
    print("pending names and lever L26")
    cfg = W.config()
    nm = NM.load(cfg)
    m = CM.build("src_p", cls(["Plantago_major", "Palmer Amaranth"]), cfg, nm, Targets(cfg, nm))
    check("a name the caches lack is pending, never guessed", m["pending"] == ["Plantago_major"]
          and m["classes"][0]["pending"], m["pending"])
    fcache = W.TMP / "inc" / "funnel" / "taxonomy_cache.json"
    before = hashlib.sha256(fcache.read_bytes()).hexdigest()
    wd = W.TMP / "inc" / "intake" / "work" / "src_p"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "pending_names.json").write_text(json.dumps({"names": m["pending"]}))
    calls = []
    res = NM.run_names(cfg, "src_p", transport=FWD.replay_transport(calls=calls), testing=True)
    check("collect names resolves the pending names through the authority", res["names"] >= 1 and res["errors"] == 0
          and calls, res)
    check("the funnel's cache file is not written", hashlib.sha256(fcache.read_bytes()).hexdigest() == before)
    layer = W.TMP / "inc" / "intake" / "names" / "names_cache.json"
    doc = json.loads(layer.read_text())
    check("the names layer holds only what the funnel cache lacked, in its own format",
          doc["format"] == "collect-names-cache/1" and "plantagomajor" in doc["resolved"]
          and "portulacaoleracea" not in doc["resolved"], sorted(doc["resolved"])[:5])
    rep = json.loads((W.TMP / "inc" / "intake" / "names" / "names_src_p.json").read_text())
    check("the report lists each name's status", any(r["name"] == "Plantago_major" and r["status"] ==
                                                     "taxon_resolved" for r in rep["names"]), rep["names"])
    nm2 = NM.load(cfg)
    m2 = CM.build("src_p", cls(["Plantago_major", "Palmer Amaranth"]), cfg, nm2, Targets(cfg, nm2))
    check("after L26 the class maps (the reject class) and nothing is pending",
          not m2["pending"] and m2["classes"][0]["inc_id"] == cfg.other_id
          and m2["provenance"]["names_layer"]["sha256"], m2["classes"][0])
    import os
    os.environ["SLURM_JOB_ID"] = "1"
    try:
        from weed_optimizer_framework.tools.collect import Refusal
        e = raises(lambda: NM.run_names(cfg, "src_p", transport=FWD.replay_transport()), Refusal)
        check("collect names refuses inside a Slurm job (no network there)", e is not None and e.code == "needs_network")
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
    bad = dict(doc, overrides={"sha256": "0" * 64})
    layer.write_text(json.dumps(bad))
    from weed_optimizer_framework.tools.collect import CollectError
    check("a names layer built with other overrides refuses", raises(lambda: NM.load(cfg), CollectError) is not None)


def main():
    try:
        test_map()
        test_rank_rule()
        test_pending_and_l26()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
