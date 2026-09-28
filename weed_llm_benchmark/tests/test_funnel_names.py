#!/usr/bin/env python3
"""Name status v2 (docs/FUNNEL_AUDIT.md §3.1 S7b; runner §4.4, §5.2.3).

What is pinned, and why:
  * the rule order: one name per status, and names that hit two rules land on
    the earlier one (a state word beats a role name, a pattern beats the
    authority, a synonym beats a relative);
  * the authority is consulted through the resolver only when no pattern
    applies, and a vernacular match never makes a target synonym;
  * on the real census_v0.json (read from the local artifact), with the
    taxonomy cache built from the recorded GBIF answers, "BroWeed" and
    "NarWeed" are unresolvable and "crop"/"Crop" are role names, with the box
    counts read from census_v0 by name; the non-object and state totals are
    recorded next to the contract's numbers (read from the contract), not
    asserted;
  * frames: every status but "target" is in exactly one frame of the config;
  * build_name_status refuses a repeated class and a row without its fields;
    a name the offline cache lacks refuses with the lever-L12 message.

No network: the socket is blocked; the cache is built from the recording.

Run:  python3 tests/test_funnel_names.py
"""
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_names_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
(TMP / "repo" / "docs").mkdir(parents=True)
(TMP / "inc" / "funnel").mkdir(parents=True)
shutil.copyfile(ROOT.parent / "docs" / "FUNNEL_AUDIT.md", TMP / "repo" / "docs" / "FUNNEL_AUDIT.md")
shutil.copyfile(ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json",
                TMP / "inc" / "funnel" / "prereg_v1.json")


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise AssertionError("the test touched the network")


socket.socket = _NoNet

import funnel_world as FW  # noqa: E402
from weed_optimizer_framework.tools.funnel import TaxonomyError, NamesError  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import names as N  # noqa: E402
from weed_optimizer_framework.tools.funnel import taxonomy as T  # noqa: E402
from weed_optimizer_framework.tools.funnel.adapters import inc_step1 as A  # noqa: E402

FAILURES, SKIPS = [], []
LOCAL_INC = ROOT / "results" / "framework" / "inc"
CONTRACT = ROOT.parent / "docs" / "FUNNEL_AUDIT.md"


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
CENSUS_V0 = json.load(open(LOCAL_INC / "funnel" / "census_v0.json"))
RULE_NAMES = ["Amaranthus rudis", "Redroot Pigweed", "Taraxacum_officinale", "BroWeed", "Giant ragweed",
              "Crab Grass", "Cassia obtusifolia", "Kochia"]
CACHE_PATH = FW.build_taxonomy_cache(TMP / "inc" / "funnel",
                                     sorted({r["src_name"] for r in CENSUS_V0 if r["src_name"]} | set(RULE_NAMES)))
RES = T.Resolver(T.load_cache(CACHE_PATH, DOM), DOM)


def st(name, joined=False, resolver=RES):
    return N.status_v2(name, joined, DOM, resolver)["status"]


def test_rule_order():
    print("rule order: one name per status")
    cases = [("Waterhemp", True, "target"), ("", False, "no_name"), ("12", False, "numeric"),
             ("leaf curl", False, "state"), ("rot", False, "state"), ("greenhouse", False, "non_object"),
             ("hut", False, "non_object"), ("Crop", False, "role"), ("weed", False, "generic"),
             ("class3", False, "generic"), ("Amaranthus rudis", False, "target_synonym"),
             ("Cassia obtusifolia", False, "target_synonym"), ("Redroot Pigweed", False, "target_related"),
             ("Giant ragweed", False, "target_related"), ("Taraxacum_officinale", False, "taxon_resolved"),
             ("Kochia", False, "taxon_resolved"), ("BroWeed", False, "unresolvable")]
    for name, joined, want in cases:
        got = st(name, joined)
        check("%r -> %s" % (name, want), got == want, got)
    seen = {w for _n, _j, w in cases}
    check("every status of RULE_ORDER is exercised", set(N.RULE_ORDER) == seen, set(N.RULE_ORDER) - seen)

    print("rule order: the earlier rule wins")

    class Boom(object):
        mappable_via = ("scientific", "override")
        informative_via = ("scientific", "vernacular")

        def resolve(self, name):
            raise AssertionError("the resolver was consulted for %r" % name)
    boom = Boom()
    for name, want in (("healthy crop", "state"), ("greenhouse crop", "non_object"), ("crop", "role"),
                       ("pest damage", "state"), ("weed12", "generic"), ("Waterhemp", "target")):
        try:
            got = N.status_v2(name, name == "Waterhemp", DOM, boom)["status"]
        except AssertionError as e:
            got = str(e)
        check("%r is %s without asking the authority" % (name, want), got == want, got)
    check("'Amaranthus rudis' (a scientific synonym) is a synonym, not a relative, though it holds a "
          "related token", st("Amaranthus rudis") == "target_synonym")
    check("the rule order is the runner's", N.RULE_ORDER == (
        "target", "no_name", "numeric", "state", "non_object", "role", "generic", "target_synonym",
        "target_related", "taxon_resolved", "unresolvable"))

    print("a vernacular match never makes a synonym")
    fake = {"responses": {}, "resolved": dict(RES.cache["resolved"])}
    fake["resolved"][N.key("carelessweed")] = dict(
        T.empty_entry("carelessweed"), via="vernacular", canonical="Amaranthus palmeri",
        accepted="Amaranthus palmeri", rank="species",
        lineage={"kingdom": "Plantae", "phylum": "Tracheophyta", "class": "Magnoliopsida",
                 "order": "Caryophyllales", "family": "Amaranthaceae", "genus": "Amaranthus",
                 "species": "Amaranthus palmeri"},
        vernacular_candidates=["Amaranthus palmeri"])
    r2 = T.Resolver(fake, DOM)
    got = N.status_v2("carelessweed", False, DOM, r2)
    check("a common name whose first match is a target species is target_related, never target_synonym",
          got["status"] == "target_related" and got["via"] == "vernacular", got)
    fake["resolved"][N.key("palmer seedlings")] = T.empty_entry("palmer seedlings")
    fake["resolved"][N.key("giantragweed")] = T.empty_entry("giantragweed")
    got = N.status_v2("palmer seedlings", False, DOM, r2)
    check("a name the authority cannot resolve but that holds a related token is target_related (by pattern)",
          got["status"] == "target_related" and got["via"] == "pattern", got)
    got = N.status_v2("giantragweed", False, DOM, r2)
    check("... unless it is an allowed key (giant ragweed is its own species)", got["status"] == "unresolvable", got)
    bean = RES.resolve("Phaseolus vulgaris")
    fake["resolved"][N.key("Blackbean")] = dict(bean, name="Blackbean", via="override", match_type="OVERRIDE")
    got = N.status_v2("Blackbean", False, DOM, r2)
    check("a project override to a non-target taxon resolves the name: taxon_resolved (named frame), not "
          "unresolvable", got["status"] == "taxon_resolved" and got["via"] == "override"
          and got["taxon"] == "Phaseolus vulgaris" and N.frame_of(got["status"], DOM) == "named", got)


def test_frames():
    print("frames")
    fr = {s: N.frame_of(s, DOM) for s in N.RULE_ORDER}
    check("target is its own frame", fr["target"] == "target")
    check("no-information frame: no_name, numeric, generic, unresolvable",
          sorted(s for s, f in fr.items() if f == "noinfo") == ["generic", "no_name", "numeric", "unresolvable"], fr)
    check("named frame: taxon_resolved, target_related, target_synonym, role",
          sorted(s for s, f in fr.items() if f == "named") == ["role", "target_related", "target_synonym",
                                                               "taxon_resolved"], fr)
    check("excluded frame: non_object, state",
          sorted(s for s, f in fr.items() if f == "excluded") == ["non_object", "state"], fr)
    check("an unknown status refuses", raises(lambda: N.frame_of("mystery", DOM), NamesError) is not None)


def test_census_v0():
    print("name status v2 on the real census_v0.json")
    rows = {}
    for r in CENSUS_V0:
        k = (r["source"], r["src_name"])
        a = rows.setdefault(k, {"source": r["source"], "src_id": r["src_name"], "name": r["src_name"],
                                "joined_target": r["label"] != "OtherPlant",
                                "status_v1": A.name_status_v1(r["src_name"]), "boxes": 0, "conflicts": 0})
        a["boxes"] += r["n"]
        a["conflicts"] += r["verdict"].get("conflict", 0)
    cn = A.contract_numbers(CONTRACT)
    ns = N.build_name_status(list(rows.values()), DOM, RES, contract_check={
        "broweed_narweed_unresolvable_boxes": {"status": "unresolvable", "keys": ["broweed", "narweed"],
                                               "contract": cn["s7b_broweed_narweed"]},
        "crop_role_boxes": {"status": "role", "keys": ["crop"], "contract": cn["s7b_crop"]},
        "non_object_boxes": {"status": "non_object", "contract": cn["s7b_non_object"]},
        "state_boxes": {"status": "state", "contract": cn["s7b_state"]}})
    want_bro = sum(r["n"] for r in CENSUS_V0 if r["src_name"] in ("BroWeed", "NarWeed"))
    want_crop = sum(r["n"] for r in CENSUS_V0 if r["src_name"] in ("crop", "Crop"))
    got = {n["name"]: n["status_v2"] for n in ns["names"]}
    check("BroWeed and NarWeed are unresolvable", got.get("BroWeed") == "unresolvable"
          and got.get("NarWeed") == "unresolvable", (got.get("BroWeed"), got.get("NarWeed")))
    check("their boxes (%d, read from census_v0) are counted as unresolvable" % want_bro,
          ns["contract_check"]["broweed_narweed_unresolvable_boxes"] == want_bro, ns["contract_check"])
    check("crop and Crop are role names", got.get("crop") == "role" and got.get("Crop") == "role")
    check("their boxes (%d, read from census_v0) are counted as role" % want_crop,
          ns["contract_check"]["crop_role_boxes"] == want_crop, ns["contract_check"])
    check("the contract's own S7b numbers are recorded beside them",
          ns["contract_check"]["contract"]["broweed_narweed_unresolvable_boxes"] == cn["s7b_broweed_narweed"]
          and ns["contract_check"]["contract"]["crop_role_boxes"] == cn["s7b_crop"])
    print("       non_object boxes %d (contract %d), state boxes %d (contract %d); recorded, not asserted"
          % (ns["contract_check"]["non_object_boxes"], cn["s7b_non_object"], ns["contract_check"]["state_boxes"],
             cn["s7b_state"]))
    check("the non-object and state totals are written to contract_check",
          isinstance(ns["contract_check"].get("non_object_boxes"), int)
          and isinstance(ns["contract_check"].get("state_boxes"), int))
    total = sum(r["n"] for r in CENSUS_V0)
    check("every census box is in one status", sum(b["boxes"] for b in ns["by_status"].values()) == total)
    check("target-joined names are 'target' with the target frame",
          all(n["status_v2"] == "target" for n in ns["names"] if rows[(n["source"], n["name"])]["joined_target"]))
    check("the output is frozen and carries the rule order", ns["frozen"] is True
          and ns["rule_order"] == list(N.RULE_ORDER))
    v2_noinfo = sum(ns["by_status"][s]["boxes"] for s in ("no_name", "numeric", "generic", "unresolvable"))
    print("       v2 uninformative boxes %d of %d (the contract's post hoc estimate is 417,717)" % (v2_noinfo, total))


def test_refusals():
    print("refusals")
    row = {"source": "s", "src_id": "0", "name": "weed", "joined_target": False, "status_v1": "generic",
           "boxes": 1, "conflicts": 0}
    check("a repeated class refuses",
          raises(lambda: N.build_name_status([row, dict(row)], DOM, RES), NamesError) is not None)
    bad = dict(row)
    bad.pop("conflicts")
    check("a row without conflicts refuses", raises(lambda: N.build_name_status([bad], DOM, RES), NamesError))
    empty = T.Resolver({"responses": {}, "resolved": {}}, DOM)
    msg = raises(lambda: N.status_v2("Spermacoce hispida", False, DOM, empty), TaxonomyError)
    check("a name the offline cache lacks refuses, naming lever L12", msg is not None and "lever L12" in msg, msg)
    check("the key keeps letters and digits only", N.key("Field_Pea 2") == "fieldpea2")


if __name__ == "__main__":
    try:
        test_rule_order()
        test_frames()
        test_census_v0()
        test_refusals()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)
