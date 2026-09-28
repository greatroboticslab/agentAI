#!/usr/bin/env python3
"""The domain config loader and the pre-registration (runner §3, §5.1.2).

Pinned:
  * domains/weed.json validates, and a synthetic "vehicles" config (the R13
    shape: car, truck, bus + OtherObject, a numeric source, no out-of-domain
    calibration) loads through the same code;
  * every schema refusal of runner §5.1.2 refuses a copy of weed.json edited
    to break exactly it: a guard stage marked recoverable (mutation point M5),
    target ids out of order, an other id colliding with a target id, a
    claimed set declared independent, a claimed set in qualify_rl_on, an exam
    both deciding and not, a domain term equal to an authority/provider kind,
    an unknown judge kind, a board naming an unknown known-truth set, an
    unknown top-level key, and a prereg for another domain;
  * exam_splits() is common.EVAL_SPLITS split on dev, plus domain_dev, and
    non_dev_exams() is the non-decision list;
  * options() are numbered 1..n: targets in config order, then attractors,
    then the four tail options, with their answers;
  * load_prereg refuses a contract whose sha256 differs from the prereg's;
  * core_sha256 excludes the amendments: append_amendment keeps it, writes
    only the amendments list (the file stays json.dumps(indent=1) bytes), and
    refuses an amendment naming another core (the file was edited outside
    its amendments), a malformed amendment, a repeated id and a second
    sample lock;
  * terms() gives the grep terms of runner §7.5 from the config alone and
    leaves out the known-truth pseudo-source of the independent set.

Run:  python3 tests/test_funnel_domain.py
"""
import copy
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_domain_")
check, raises = W.check, W.raises

from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402

PREREG = TMP / "inc" / "funnel" / "prereg_v1.json"
WEED = D.DOMAINS_DIR / "weed.json"

print("weed.json")
dom = D.load("weed")
check("weed.json validates", D.validate(json.loads(WEED.read_text())) == [])
check("load by path equals load by name", D.load(WEED).sha256 == dom.sha256)
check("targets are ids 0..n-1", dom.target_ids == list(range(len(dom.targets))))
check("targets are C.CLASS_NAMES[:12]", dom.target_names == C.CLASS_NAMES[:12], dom.target_names)
check("other class", dom.other == {"id": C.OTHER_PLANT, "name": C.CLASS_NAMES[C.OTHER_PLANT]})
check("class_name / class_id round trip", all(dom.class_id(dom.class_name(i)) == i for i in range(13)))
check("unknown target refused", raises(lambda: dom.target("NotAClass"), F.DomainError))
check("exam splits = EVAL_SPLITS split on dev (+ domain_dev)",
      dom.exam_splits() == {"decision": "dev", "non_decision": tuple(s for s in C.EVAL_SPLITS if s != "dev"),
                            "extra_non_decision": ("domain_dev",)}, dom.exam_splits())
check("non_dev_exams", dom.non_dev_exams() == tuple(s for s in C.EVAL_SPLITS if s != "dev") + ("domain_dev",))
check("reference lab from lab_groups", dom.reference_lab() == dom.lab_of(dom.reference_source) != "src:" + dom.reference_source)
check("a source in no lab group is its own lab", dom.lab_of("some_unlisted_slug") == "src:some_unlisted_slug")
check("recoverable stages are recoverable-true only",
      all(dom.stage(s)["recoverable"] is True for s in dom.recoverable_stages()))
check("no guard stage is recoverable", all(not s["guard"] or s["recoverable"] is False for s in dom.stages))
check("the independent set", dom.independent_sets() == ["KT7"], dom.independent_sets())
check("claimed sets by hypothesis", dom.claimed_sets("H1") == ["KT4"] and dom.claimed_sets("H3a") == ["KT6"])

print("options")
opts = dom.options()
nt, na = len(dom.targets), len(dom.attractors)
check("numbered 1..n", [o["n"] for o in opts] == list(range(1, len(opts) + 1)))
check("targets first, in config order", [o["answer"] for o in opts[:nt]] == dom.target_names)
check("then attractors", [o["attractor"] for o in opts[nt:nt + na]] == [a["id"] for a in dom.attractors])
check("then the four tail options", [o["answer"] for o in opts[nt + na:]] == list(D.TAIL_ANSWERS))
check("attractor and tail 'other' answer the other class",
      all(o["class"] == dom.other["id"] for o in opts[nt:nt + na]) and opts[nt + na]["class"] == dom.other["id"])
check("genus-rank target option names the genus", any(o["rank"] == "genus" and "spp." in o["text"] for o in opts[:nt]))
check("pair options", [o["answer"] for o in dom.pair_options()] == list(D.PAIR_ANSWERS))

print("schema refusals")
base = json.loads(WEED.read_text())


def broken(edit):
    raw = copy.deepcopy(base)
    edit(raw)
    return D.validate(raw)


def guard_recoverable(raw):
    for s in raw["stages"]:
        if s["guard"]:
            s["recoverable"] = True
            return


def ids_out_of_order(raw):
    t = raw["classes"]["targets"]
    t[0]["id"], t[1]["id"] = t[1]["id"], t[0]["id"]


def other_collides(raw):
    raw["classes"]["other"]["id"] = 3


def claimed_independent(raw):
    raw["known_truth"]["KT4"]["independent"] = True


def claimed_qualifies(raw):
    raw["known_truth"]["qualify_rl_on"] = raw["known_truth"]["qualify_rl_on"] + ["KT4"]


def exam_both(raw):
    raw["exams"]["non_decision"] = raw["exams"]["non_decision"] + ["dev"]


def term_is_kind(raw):
    raw["domain_terms"] = list(raw.get("domain_terms", [])) + [raw["taxonomy"]["authority"]["kind"]]


def judge_kind(raw):
    raw["judges"]["panel"][0]["kind"] = "crystal_ball"


def board_kt(raw):
    raw["reference_labeller"]["boards"][0]["targets_from"] = "KT99"


def unknown_key(raw):
    raw["surprise"] = 1


def lab_twice(raw):
    lg = raw["sources"]["lab_groups"]
    a, b_ = sorted(lg)[:2]
    lg[b_] = lg[b_] + [lg[a][0]]


for name, fn, word in (("a guard stage marked recoverable (M5)", guard_recoverable, "guard"),
                       ("target ids not 0..n-1 in order", ids_out_of_order, "0..n-1"),
                       ("other id colliding with a target id", other_collides, "collides"),
                       ("a claimed set declared independent", claimed_independent, "independent"),
                       ("a claimed set in qualify_rl_on", claimed_qualifies, "claimed"),
                       ("an exam both deciding and not", exam_both, "decision"),
                       ("a domain term equal to an authority kind", term_is_kind, "kind"),
                       ("an unknown judge kind", judge_kind, "unknown kind"),
                       ("a board naming an unknown set", board_kt, "KT99"),
                       ("an unknown top-level key", unknown_key, "surprise"),
                       ("a source in two lab groups", lab_twice, "both")):
    probs = broken(fn)
    check("refuses %s" % name, probs and any(word in p for p in probs), probs[:3])
    bad = TMP / ("bad_%s.json" % fn.__name__)
    raw = copy.deepcopy(base)
    fn(raw)
    bad.write_text(json.dumps(raw))
    check("load() raises DomainError for %s" % name, raises(lambda: D.load(bad), F.DomainError))
check("the M5 marker sits on the guard check", sum(1 for ln in (D.__file__ and open(D.__file__).read().splitlines())
                                                   if "funnel-mutation: M5" in ln) == 1)

print("vehicles config (R13 shape)")
veh = {
    "format": "funnel-domain/1", "domain": "vehicles", "adapter": "generic",
    "domain_terms": ["vehicles"],
    "classes": {"targets": [{"id": 0, "name": "car", "common": "car", "taxon": "car", "rank": "species"},
                            {"id": 1, "name": "truck", "common": "truck", "taxon": "truck", "rank": "species"},
                            {"id": 2, "name": "bus", "common": "bus", "taxon": "bus", "rank": "species"}],
                "other": {"id": 3, "name": "OtherObject"}},
    "names": {"numeric_regex": "[0-9]+", "generic_keys": ["object"], "generic_regex": "object[0-9]*",
              "non_object_words": [], "state_words": [], "role_names": [], "related_tokens": [],
              "related_allowed_keys": [],
              "frames": {"noinfo": ["no_name", "numeric", "generic", "unresolvable"],
                         "named": ["taxon_resolved", "target_related", "target_synonym", "role"],
                         "excluded": ["non_object", "state"]}},
    "stages": [{"id": "V1", "name": "read", "role": "read", "unit": "image", "guard": False, "recoverable": False,
                "depends_on": []},
               {"id": "V2", "name": "eval guard", "role": "guard", "unit": "image", "guard": True,
                "recoverable": False, "depends_on": ["V1"]},
               {"id": "V3", "name": "join", "role": "join", "unit": "box", "guard": False, "recoverable": True,
                "depends_on": ["V2"]},
               {"id": "V4", "name": "check", "role": "target_check", "unit": "box", "guard": False,
                "recoverable": True, "depends_on": ["V3"]}],
    "exams": {"decision": "dev", "non_decision": ["highway_test"], "extra_non_decision": []},
    "sources": {"reference": "city_core", "lab_groups": {"CityLab": ["city_core", "numeric_cams"]}},
    "known_truth": {"KT1": {"what": "reference crops", "allowed_uses": ["calibration"], "independent": False,
                            "claimed_by": None, "never_qualifies": [], "sources": ["city_core"], "split": None},
                    "qualify_rl_on": ["KT1"], "forbidden": ["dev", "highway_test"]},
}
vp = TMP / "vehicles.json"
vp.write_text(json.dumps(veh))
vd = D.load(vp)
check("vehicles config loads", vd.name == "vehicles" and vd.target_names == ["car", "truck", "bus"])
check("vehicles exams", vd.non_dev_exams() == ("highway_test",))
t = vd.terms()
check("vehicles terms: domain, classes, exam, lab", {"vehicles", "car", "truck", "bus", "otherobject", "highwaytest",
                                                     "citylab"} <= set(t["token"]), t)
check("vehicles substring terms carry the slugs, not the reference source",
      "numeric_cams" in t["substring"] and "city_core" not in t["substring"], t)
os.environ["FUNNEL_DOMAINS_DIR"] = str(TMP)
check("FUNNEL_DOMAINS_DIR resolves a name", D.load("vehicles").sha256 == vd.sha256)
del os.environ["FUNNEL_DOMAINS_DIR"]

print("weed terms")
wt = dom.terms()
check("domain name and class tokens", {"weed", "waterhemp", "otherplant", "morningglory"} <= set(wt["token"]))
check("taxon words of >= 5 letters", {"amaranthus", "tuberculatus", "indica"} <= set(wt["token"]))
check("exam tokens (not dev, test)", {"ood22", "ood23", "imageweeds"} <= set(wt["token"])
      and not {"dev", "test"} & set(wt["token"]))
check("lab group tokens", {"ndsu", "lulab"} <= set(wt["token"]))
check("source slugs are substrings", "project_agml__weed_crop_detection" in wt["substring"])
check("the independent set's pseudo-source is not a term", "kt7" not in wt["substring"])

print("prereg")
pre = D.load_prereg(PREREG)
check("prereg loads", pre.domain_name == "weed" and pre.amendments == [] and pre.sample_lock is None)
check("groups (planned, minimum)", pre.groups["G1"] == (300, 160) and pre.groups["sentinels"] == (400, None))
core = {k: v for k, v in json.loads(PREREG.read_text()).items() if k != "amendments"}
check("core sha = canonical JSON without amendments",
      pre.core_sha256 == F.sha256_bytes(F.canonical_json(core).encode("utf-8")))
check("load_prereg with the matching domain", D.load_prereg(PREREG, domain=dom).core_sha256 == pre.core_sha256)
check("prereg for another domain refused", raises(lambda: D.load_prereg(PREREG, domain=vd), F.DomainError))
contract = pathlib.Path(os.environ["REPO"]) / "docs" / "FUNNEL_AUDIT.md"
text = contract.read_bytes()
contract.write_bytes(text + b"\nedited\n")
check("an edited contract refuses the prereg", raises(lambda: D.load_prereg(PREREG), F.PreregError))
contract.write_bytes(text)
os.environ["FUNNEL_CONTRACT"] = str(TMP / "nowhere.md")
check("FUNNEL_CONTRACT overrides the contract path", raises(lambda: D.load_prereg(PREREG), F.PreregError))
del os.environ["FUNNEL_CONTRACT"]

print("amendments")
raw_before = PREREG.read_bytes()
am = {"id": "A1", "kind": "sample_lock", "date": "2026-09-28", "prereg_core_sha256": pre.core_sha256,
      "sample_sha256": "0" * 64}
new = D.append_amendment(PREREG, am)
pre2 = D.load_prereg(PREREG)
check("amendment appended", pre2.amendments == [am] and pre2.sample_lock["id"] == "A1")
check("core sha unchanged", pre2.core_sha256 == pre.core_sha256)
check("raw sha changed", pre2.sha256 != pre.sha256)
check("file is json.dumps(indent=1) of the new object", PREREG.read_text() == json.dumps(new, indent=1,
                                                                                          ensure_ascii=False))
check("everything but amendments byte-equal in JSON",
      {k: v for k, v in json.loads(PREREG.read_text()).items() if k != "amendments"} == core)
check("next amendment id", D.next_amendment_id(pre2) == "A2")
check("repeated id refused", raises(lambda: D.append_amendment(PREREG, dict(am, kind="note")), F.PreregError))
check("second sample lock refused", raises(lambda: D.append_amendment(PREREG, dict(am, id="A2")), F.PreregError))
check("missing date refused", raises(lambda: D.append_amendment(PREREG, {"id": "A2", "kind": "note"}),
                                     F.PreregError))
check("bad date refused", raises(lambda: D.append_amendment(PREREG, {"id": "A2", "kind": "note", "date": "28/09"}),
                                 F.PreregError))
edited = json.loads(PREREG.read_text())
edited["hypotheses"]["H1"]["supported"] = "LB>=0.50 overall and for Ragweed"
PREREG.write_text(json.dumps(edited, indent=1))
check("an edit outside amendments is refused (the amendment names the old core)",
      raises(lambda: D.append_amendment(PREREG, {"id": "A2", "kind": "note", "date": "2026-09-29",
                                                 "prereg_core_sha256": pre.core_sha256}), F.PreregError))
PREREG.write_bytes(raw_before)
check("prereg restored", D.load_prereg(PREREG).sha256 == pre.sha256)

W.finish()
