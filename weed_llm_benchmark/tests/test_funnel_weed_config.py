#!/usr/bin/env python3
"""The weed domain config, funnel/domains/weed.json (runner §3.2, §5.2.6).

What is pinned, and why (every expected value is read from its source, never
typed): the config validates; the targets are the INC class space's twelve
species in id order, with cwd12_species' binomials and common names except
MorningGlory (genus Ipomoea, class policy DEC-3); the class policy's "not"
lists and genus-unsure list are the prereg's; the lab groups are the prereg's
(train_core added to the reference lab); the name word lists together hold
verify.py's GENERIC_NAME_KEYS, NON_PLANT_WORDS and CWD12_RELATED_TOKENS, each
word in exactly one list; the non-decision exams are C.EVAL_SPLITS without
dev; guard stages are never recoverable and the stage table is the Step 1
adapter's; the card class tables agree with the contract's statements and
with the pool's own class names; the runner's attractors are present.

Run:  python3 tests/test_funnel_weed_config.py
"""
import json
import os
import pathlib
import re
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_weed_config_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import names as N  # noqa: E402
from weed_optimizer_framework.tools.funnel.adapters import inc_step1 as A  # noqa: E402

FAILURES, SKIPS = [], []
PREREG = json.load(open(ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json"))
CONTRACT = (ROOT.parent / "docs" / "FUNNEL_AUDIT.md").read_text(encoding="utf-8")
CENSUS_V0 = json.load(open(ROOT / "results" / "framework" / "inc" / "funnel" / "census_v0.json"))


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def main():
    dom = D.load("weed")
    raw = dom.raw
    print("validation")
    check("weed.json validates", D.validate(raw) == [], D.validate(raw))
    check("its domain is the prereg's", raw["domain"] == PREREG["domain"])
    print("classes")
    tg = raw["classes"]["targets"]
    check("targets are C.CLASS_NAMES[:12] with ids 0..11", [t["name"] for t in tg] == C.CLASS_NAMES[:12]
          and [t["id"] for t in tg] == list(range(12)))
    bad = [t["name"] for t in tg if t["name"] != "MorningGlory" and t["taxon"] != S.CWD12_BINOMIAL[t["name"]]]
    check("taxa are cwd12_species.CWD12_BINOMIAL", not bad, bad)
    mg = dom.target("MorningGlory")
    pol = PREREG["class_policy"]["MorningGlory"]
    check("MorningGlory is the genus the class policy names", mg["rank"] == pol["rank"] == "genus"
          and mg["taxon"] == pol["taxon"])
    check("common names are CWD12_COMMON", all(t["common"] == S.CWD12_COMMON[t["name"]] for t in tg))
    for t in tg:
        want = PREREG["class_policy"].get(t["name"], {}).get("not", [])
        if t["not"] != want:
            check("%s 'not' list is the prereg's" % t["name"], False, (t["not"], want))
    check("every 'not' list is the prereg's class policy", all(
        t["not"] == PREREG["class_policy"].get(t["name"], {}).get("not", []) for t in tg))
    check("genus_answer_unsure_for is the prereg's", raw["classes"]["genus_answer_unsure_for"]
          == PREREG["class_policy"]["genus_answers_unsure_for"])
    check("OtherPlant is class 12", raw["classes"]["other"] == {"id": C.OTHER_PLANT, "name": "OtherPlant"})
    check("PalmerAmaranth and Waterhemp are each other's siblings",
          dom.target("PalmerAmaranth")["siblings"] == ["Waterhemp"] and dom.target("Waterhemp")["siblings"]
          == ["PalmerAmaranth"])
    print("attractors")
    at = {a["taxon"] for a in dom.attractors}
    want = {"Amaranthus retroflexus", "Amaranthus hybridus", "Bassia scoparia", "Erigeron canadensis",
            "Euphorbia hirta", "Euphorbia hypericifolia", "Chamaecrista", "Digitaria", "Cyperus", "Ambrosia trifida"}
    check("the runner's attractors are present", want <= at, want - at)
    check("genus attractors are genus rank", all(a["rank"] == "genus" for a in dom.attractors
                                                   if a["taxon"] in ("Chamaecrista", "Digitaria", "Cyperus")))
    check("each attractor names what it is confused with and an option text",
          all(a["confused_with"] and a["option"] for a in dom.attractors))
    print("names")
    nm = raw["names"]
    lists = {k: set(N.key(w) for w in nm[k]) for k in ("generic_keys", "non_object_words", "non_object_keys",
                                                       "state_words", "state_keys", "related_tokens")}
    words = set(V.GENERIC_NAME_KEYS) | set(V.NON_PLANT_WORDS) | set(V.CWD12_RELATED_TOKENS)
    where = {w: [k for k, ws in lists.items() if w in ws] for w in words}
    check("verify's word lists are covered, each word in exactly one list",
          all(len(v) == 1 for v in where.values()), {w: v for w, v in where.items() if len(v) != 1})
    check("related_tokens is verify.CWD12_RELATED_TOKENS", nm["related_tokens"] == list(V.CWD12_RELATED_TOKENS))
    check("related_allowed_keys is verify.OTHER_ALLOWED_KEYS", set(nm["related_allowed_keys"]) == set(V.OTHER_ALLOWED_KEYS))
    check("generic_regex is verify._GENERIC_RE", nm["generic_regex"] == V._GENERIC_RE.pattern)
    check("role names are crop and crops", nm["role_names"] == ["crop", "crops"])
    check("the contract's object and state names are listed",
          {"greenhouse", "solar", "musor"} <= lists["non_object_words"] and {"hut", "shed", "pave"} <= lists["non_object_keys"]
          and {"cercospora", "xanthomonas", "mosaic", "leafcurl", "healthy", "dryleaf", "drygrass"} <= lists["state_words"])
    check("the frames are the runner's", nm["frames"] == {
        "noinfo": ["no_name", "numeric", "generic", "unresolvable"],
        "named": ["taxon_resolved", "target_related", "target_synonym", "role"], "excluded": ["non_object", "state"]})
    print("stages and exams")
    check("the stage table is the Step 1 adapter's", [(s["id"], s["role"]) for s in dom.stages]
          == [(s, A.STAGE_ROLES[s]) for s in A.ADAPTER_STAGES])
    check("guard stages (S4, S5) are never recoverable", all(s["recoverable"] is False for s in dom.stages if s["guard"])
          and {s["id"] for s in dom.stages if s["guard"]} == {"S4", "S5"})
    check("non_decision exams are C.EVAL_SPLITS without dev", dom.exam_splits()["non_decision"]
          == tuple(s for s in C.EVAL_SPLITS if s != "dev") and dom.exam_splits()["decision"] == "dev")
    check("the H10d domain dev is an extra non-decision exam", dom.exam_splits()["extra_non_decision"] == ("domain_dev",))
    print("sources and known truth")
    lg = {g: list(v) for g, v in PREREG["known_truth"]["lab_groups"].items()}
    lg["LuLab"] = ["train_core"] + lg["LuLab"]
    check("lab groups are the prereg's, train_core added to LuLab", dom.lab_groups() == lg, dom.lab_groups())
    check("the reference source is in the reference lab", dom.reference_lab() == "LuLab")
    check("the authoritative sources are the two NDSU sources whose papers name the species",
          dom.authoritative_sources() == ["project_agml__greenhouse_crop_weed_detection",
                                          "project_agml__weed_crop_detection"])
    kt = raw["known_truth"]
    check("qualify_rl_on is KT1-KT3 and KT7", kt["qualify_rl_on"] == PREREG["known_truth"]["qualify_RL_on"])
    check("claimed sets: KT4 H1, KT5 H2b, KT6 H3a", (kt["KT4"]["claimed_by"], kt["KT5"]["claimed_by"],
                                                   kt["KT6"]["claimed_by"]) == ("H1", "H2b", "H3a"))
    check("only KT7 is independent", dom.independent_sets() == ["KT7"])
    check("KT7 never qualifies J1 or J-zs", kt["KT7"]["never_qualifies"] == PREREG["known_truth"]["KT7_never_qualifies"])
    check("forbidden splits are the prereg's", kt["forbidden"] == PREREG["known_truth"]["forbidden"])
    check("KT7: at most 30 per taxon (DEC-8), CC licences only, the targets and attractors",
          kt["kt7"]["per_taxon_max"] == 30 and all(l.startswith("cc") for l in kt["kt7"]["licences"])
          and set(kt["kt7"]["taxa"]) == {t["taxon"] for t in tg} | at)
    check("the KT7 query is research grade with CC photo licences",
          kt["kt7"]["provider"]["params"]["quality_grade"] == "research"
          and kt["kt7"]["provider"]["params"]["photo_license"].split(",") == kt["kt7"]["licences"])
    print("card resolvers")
    mh = raw["sources"]["card_resolvers"]["project_agml__mh_weed16_weed_detection"]["class_table"]
    check("MH-Weed16's table has 16 classes (ids 0-15)", sorted(int(k) for k in mh) == list(range(16)))
    for cid, taxon in re.findall(r"id (\d+) = \*([A-Z][a-z]+ [a-z]+)\*", CONTRACT):
        check("the contract's 'id %s = %s' matches the card table" % (cid, taxon), mh[cid]["taxon"] == taxon,
              mh[cid])
    wc = raw["sources"]["card_resolvers"]["project_agml__weed_crop_detection"]["class_table"]
    pool_names = {N.key(r["src_name"]) for r in CENSUS_V0 if r["source"] == "project_agml__weed_crop_detection"}
    check("weed_crop's card names are the pool's own class names (census_v0)",
          {N.key(v["name"]) for v in wc.values()} == pool_names, ({N.key(v["name"]) for v in wc.values()}, pool_names))
    gh = raw["sources"]["card_resolvers"]["project_agml__greenhouse_crop_weed_detection"]["names"]
    gh_pool = {N.key(r["src_name"]) for r in CENSUS_V0 if r["source"] == "project_agml__greenhouse_crop_weed_detection"}
    check("greenhouse's card names cover the pool's class names (census_v0)", gh_pool <= set(gh), gh_pool - set(gh))
    check("MH-Weed16's card image count is the contract's 6,656",
          raw["sources"]["card_image_counts"]["project_agml__mh_weed16_weed_detection"]
          == int(re.search(r"MH-Weed16's card gives ([0-9,]+)", CONTRACT).group(1).replace(",", "")))
    check("the not-recoverable sources are the contract's non-plant and leaf-disease sources",
          set(raw["sources"]["not_recoverable"]) == {
              "fvossel__csgo_player_detection", "rf_uav-qnoms__uav-wqshy",
              "kg_farukalam__tomato-leaf-diseases-detection-computer-vision", "rf_bishwarup-halder__crop-health-advisor"})
    print("terms")
    t = dom.terms()
    check("domain_terms add the substrings the engine must not hold", {"cwd12", "bioclip", "ndsu"} <= set(t["substring"]))
    check("class names and taxon words are whole tokens", {"waterhemp", "amaranthus", "otherplant", "weed"}
          <= set(t["token"]))


if __name__ == "__main__":
    main()
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)
