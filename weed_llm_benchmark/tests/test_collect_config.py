#!/usr/bin/env python3
"""The collector's domain config, collect/domains/weed.json (format
collect-domain/1; docs/CONTINUOUS_LOOP.md §3.1 "Inputs", §6.5, §7.1-7.3, §10
item 5).

Pinned:
  * it loads, pointing to the funnel domain config by path and sha256 (never a
    copy), and pins the EPPO table by path and sha256; a stale pointer or a
    stale pin refuses (a governance change, by a person or a deploy);
  * validation refuses an unknown key, a missing section, an unknown provider
    kind, a provider whose host must come from the config without it, a known
    item without match rules or with a duplicate id, a yield floor given as a
    bare number (a placeholder, §7.4), an unmapped id that collides with a
    class id, and search terms or presumed-derivative classes naming
    non-targets;
  * the known items are the owner's D-C list with its decision stamp, as a
    recall audit (every item has match rules), and the D-C sources are there
    (MH-Weed16 held as the funnel's, PAGS8, the MFWD trays, CottonWeedDet3,
    the NDSU classification sets, CottonWeedID15);
  * the lab groups: the evaluation labs are the lab of dev and test and the
    lab of the ImageWeeds exam; the funnel's groups are read in; the PAGS8
    and MFWD labs are not evaluation labs;
  * every target has an EPPO code in the pinned table; the PAGS8 card table is
    pinned with its source and who pinned it; the alias table resolves to
    targets only; the yield floors are unset (a person sets them);
  * no collector module writes under collect/domains/.

Run:  python3 tests/test_collect_config.py
"""
import copy
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_config_")
check, raises = W.check, W.raises


def load_raw(raw, name="c.json"):
    from weed_optimizer_framework.tools.collect import config as CF
    p = TMP / name
    p.write_text(json.dumps(raw))
    return CF.load(str(p))


def main():
    try:
        from weed_optimizer_framework.tools.collect import ConfigError
        from weed_optimizer_framework.tools.collect import config as CF
        print("load and pins")
        cfg = W.config()
        fpath = W.ROOT / "weed_optimizer_framework" / "tools" / "funnel" / "domains" / "weed.json"
        import hashlib
        check("the config points to the funnel domain config by path and sha256", cfg.funnel.path.resolve()
              == fpath.resolve() and cfg.raw["targets"]["sha256"] == hashlib.sha256(fpath.read_bytes()).hexdigest())
        check("the class space: targets 0..11, the reject class 12, unmapped 13", cfg.target_names[0] == "Waterhemp"
              and len(cfg.targets) == 12 and cfg.other_id == 12 and cfg.unmapped_id == 13)
        check("the EPPO table is pinned by sha256", cfg.eppo_record["sha256"] == cfg.raw["eppo"]["sha256"]
              and cfg.eppo_binomial("POROL") == "Portulaca oleracea")
        raw = json.loads(cfg.path.read_text())
        bad = copy.deepcopy(raw)
        bad["targets"]["sha256"] = "0" * 64
        e = raises(lambda: load_raw(bad), ConfigError)
        check("a stale pointer to the funnel config refuses", e is not None and "targets changed" in str(e), e)
        bad = copy.deepcopy(raw)
        bad["eppo"]["sha256"] = "1" * 64
        e = raises(lambda: load_raw(bad), ConfigError)
        check("a stale EPPO pin refuses", e is not None and "EPPO" in str(e), e)
        print("validation")
        cases = []
        for mut, want in (
                (lambda r: r.update(extra=1), "unknown top-level key"),
                (lambda r: r.pop("prefilter"), "missing top-level key 'prefilter'"),
                (lambda r: r["providers"]["zenodo"].update(kind="nosuch"), "kind 'nosuch'"),
                (lambda r: r["providers"]["weedai"].pop("base_url"), "base_url is required"),
                (lambda r: r["known_items"][1].pop("match"), "a non-empty match list"),
                (lambda r: r["known_items"].append(dict(r["known_items"][1])), "duplicate id"),
                (lambda r: r["known_items"][1].pop("decided_by"), "decided_by is required"),
                (lambda r: r["placement"].update(lab_only=["nosuch"]), "placement.lab_only"),
                (lambda r: r["budgets"].update(floor_gb=5.0), "placeholder"),
                (lambda r: r["class_map"]["status_map"].update(default="nowhere"), "status_map"),
        ):
            r = copy.deepcopy(raw)
            mut(r)
            probs = CF.validate(r)
            cases.append((want, any(want in p for p in probs), probs[:3]))
        check("validation refuses each malformed section", all(ok for _w, ok, _p in cases),
              [c for c in cases if not c[1]])
        r = copy.deepcopy(raw)
        r["budgets"]["floor_gb"] = {"value": 120.0, "set_by": "person 2026-10-15 from the first wave"}
        check("a floor written by a person with its value and who set it is valid", CF.validate(r) == [])
        for mut, want in ((lambda r: r["class_space"].update(unmapped_id=12), "collides"),
                          (lambda r: r["search_terms"].update(NotATarget=["x"]), "not a target"),
                          (lambda r: r["prefilter"]["presumed_derivative"].update(declares_any=["NotATarget"]),
                           "not a target")):
            r = copy.deepcopy(raw)
            mut(r)
            e = raises(lambda: load_raw(r), ConfigError)
            check("load refuses: %s" % want, e is not None and want in str(e), e)
        print("the owner's lists and tables")
        check("the known items carry the owner's decision stamp", cfg.known_decided_by() == "owner 2026-09-28 D-C")
        ids = [it["name"] for it in cfg.known_items()]
        check("the D-C sources are listed", ids == ["mh_weed16", "pags8", "mfwd_porol", "cottonweeddet3",
                                                    "ndsu_classification", "cottonweedid15"], ids)
        check("each known item's id is its source id (one identifier in candidates, the ledger and fetch)",
              [it["id"] for it in cfg.known_items()][:4] == ["mendeley_d3n3mgjjbv_v2",
                                                            "weedai_5c78d067-8750-4803-9cbe-57df8fae55e4",
                                                            "mediatum_1717366", "kg_yuzhenlu__cottonweeddet3"]
              and all(it["decided_by"] == "owner 2026-09-28 D-C" for it in cfg.known_items()))
        check("every known item has match rules (a recall audit, not a seed list)",
              all(it.get("match") for it in cfg.known_items()))
        check("MH-Weed16 is the funnel's", cfg.known_item("mh_weed16")["owned_by"] == "funnel")
        check("the evaluation labs are the lab of dev and test and the exam's lab",
              cfg.evaluation_labs() == ["LuLab", "NDSU"], cfg.evaluation_labs())
        lg = cfg.lab_groups()
        check("the funnel's lab groups are read in; PAGS8's and the MFWD trays' labs are not evaluation labs",
              "train_core" in lg["LuLab"]["members"] and lg["TAMU"]["evaluation"] is False
              and lg["TUM"]["evaluation"] is False)
        missing = [t["name"] for t in cfg.targets if not cfg.eppo_codes_of(t["taxon"])]
        check("every target has an EPPO code in the pinned table", not missing, missing)
        pags = cfg.raw["card_class_tables"]["weedai_5c78d067-8750-4803-9cbe-57df8fae55e4"]
        check("the PAGS8 card table is pinned with its source and who pinned it", len(pags["by_name"]) == 8
              and all(v["taxon"] == "Amaranthus palmeri" for v in pags["by_name"].values())
              and pags["table_source"] and pags["pinned_by"])
        al = CF.alias_table(cfg)
        check("the alias table resolves to targets only", al and set(al.values()) <= set(cfg.target_names))
        check("the yield floors are unset until a person sets them", cfg.yield_floors() == {"floor_gb": None,
                                                                                            "floor_su": None})
        col = W.ROOT / "weed_optimizer_framework" / "tools" / "collect"
        writers = []
        for p in sorted(col.rglob("*.py")):
            for i, ln in enumerate(p.read_text().splitlines(), 1):
                if re.search(r"(write\w*|open|replace|rename|unlink)\([^)]*DOMAINS_DIR", ln):
                    writers.append((p.name, i))
        check("no collector module writes under collect/domains/ (a governance file)", not writers, writers)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
