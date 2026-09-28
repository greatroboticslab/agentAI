#!/usr/bin/env python3
"""The funnel audit's replay fixtures (docs/FUNNEL_AUDIT.md 8.9; runner 5.5.10).

tests/fixtures/inc_replay/funnel/ holds three kinds of file, each pinned by
sha256 in its MANIFEST.json:

  copies     byte copies of results/framework/inc files (census_v0.json, the
             four Step 1 summaries, realloop_v1's exp, report, ledger and
             build summary);
  derived    funnel_ledger_summaries.json: adapters.inc_step1.
             ledger_from_summaries on those copies, run from the fixture
             directory so the ledger's input records carry relative paths;
  synthetic  claims registers, synthetic audits and class maps for R10-R12,
             the R13 vehicles domain (its config, ledger and claim), the R9b
             ledger reconstructed from the artifacts of 2026-08-25, the
             high-yield negative control, and the R14 devil's-advocate
             replies. Each carries its why in the manifest.

No file sits directly at funnel/<name> under a name evidence.load_dir reads
(funnel_ledger.json, audit_v1.json, class_maps.json, recovery.json,
prospective_da.json, files.json): the replay tree tests/fixtures/inc_replay is
loaded whole by other cases, and the funnel files enter a test's evidence only
when the test copies them into its own tree under those names.

    python3 tests/funnel_ap_fixtures.py [--check]

builds every file into the fixture directory and writes MANIFEST.json
(--check builds into a temporary directory and reports any file whose bytes
differ from the pinned ones). Every number of a synthetic file that describes
real data is read from the copies here, never typed.
"""
import copy
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile

TESTS = pathlib.Path(__file__).resolve().parent
PKG_ROOT = TESTS.parent
sys.path.insert(0, str(PKG_ROOT))

LOCAL_INC = PKG_ROOT / "results" / "framework" / "inc"
FIX = TESTS / "fixtures" / "inc_replay" / "funnel"
FROZEN_UTC = "2026-09-28T12:00:00Z"
FORMAT = "inc-autopilot/funnel-replay-fixtures/1"

COPIES = (("census_v0.json", "funnel/census_v0.json"),
          ("step1/admit_summary.json", "step1/admit_summary.json"),
          ("step1/select_summary.json", "step1/select_summary.json"),
          ("step1/pool_summary.json", "step1/pool_summary.json"),
          ("step1/calibration.json", "step1/calibration.json"),
          ("realloop_v1/exp.json", "realloop_v1/exp.json"),
          ("realloop_v1/report.json", "realloop_v1/report.json"),
          ("realloop_v1/ledger.jsonl", "realloop_v1/ledger.jsonl"),
          ("realloop_v1/build_summary.json", "realloop_v1/build_summary.json"))
LEDGER = "funnel_ledger_summaries.json"
RESEARCH_LOG_LINE = 154
C1_TEXT = ("The harvested increments do not improve the twelve target species. They are mostly other plants "
           "plus about 50 target-species boxes each.")
C2_TEXT = "web harvest at this scale supplies volume, not usable supervision"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else dump(obj), encoding="utf-8")


def _cite(artifact, value, pointer=None, line=None):
    return {"artifact": artifact, "line": line, "pointer": pointer, "value": value}


def _claim(cid, text, polarity, scope, made_by, cites, history_extra=()):
    hist = [{"utc": FROZEN_UTC, "from": None, "to": "open", "by": made_by, "reason": "filed",
             "cites": copy.deepcopy(cites)}]
    status = "open"
    for to, by, reason in history_extra:
        hist.append({"utc": FROZEN_UTC, "from": status, "to": to, "by": by, "reason": reason, "cites": []})
        status = to
    return {"id": cid, "text": text, "polarity": polarity, "scope": scope, "made_by": made_by,
            "cites": cites, "status": status, "history": hist}


def seed_claims(root, challenged=False):
    """C1 (the RESEARCH_LOG sentence of 2026-09-27) and C2 (the S1-gate reading
    of 2026-08-25), filed as the runner's 5.5.10 says. `challenged` gives C1
    the status a surviving DA pass leaves (R10's starting point)."""
    repo = PKG_ROOT.parent
    # C1 is cited as filed: RESEARCH_LOG.md line RESEARCH_LOG_LINE at commit 8b50e52. Entries
    # are added above it (newest first), so today the sentence is found by its text, and the
    # cite keeps the line and text it had when it was filed.
    lines = (repo / "RESEARCH_LOG.md").read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if C1_TEXT in ln]
    if len(hits) != 1:
        raise SystemExit("RESEARCH_LOG.md holds C1's sentence %d times, not once" % len(hits))
    line = hits[0]
    line_no = RESEARCH_LOG_LINE
    rep = json.loads((root / "realloop_v1" / "report.json").read_text())
    fig = json.loads((repo / "docs" / "poster" / "figures_data.json").read_text())
    reading = fig["s1_gate_verdict_2026_08_25"]["reading"]
    if C2_TEXT not in reading:
        raise SystemExit("figures_data.json no longer holds C2's reading")
    c1 = _claim("C1", C1_TEXT, "negative", "realloop_v1 on Step 1 of 2026-09-27: the harvested increments",
                "human-transcribed",
                [_cite("RESEARCH_LOG.md", line, line=line_no),
                 _cite("realloop_v1/report.json", rep["agreement"]["full"], pointer="/agreement/full")],
                history_extra=(("challenged", "tier2:adversary/glm-4.7-flash",
                                "the devil's-advocate pass left grounded counter-arguments standing"),)
                if challenged else ())
    c2 = _claim("C2", C2_TEXT, "scarcity", "the S1 gate of 2026-08-25: six audited harvested sources",
                "human-transcribed",
                [_cite("docs/poster/figures_data.json", reading, pointer="/s1_gate_verdict_2026_08_25/reading")])
    return {"format": "funnel-claims/1", "claims": [c1, c2]}


def vehicles_config():
    """The R13 domain: car, truck, bus and OtherObject; a hypernym table; no
    known-truth set outside the reference domain."""
    stages = [
        {"id": "V0", "name": "read", "role": "read", "unit": "image", "guard": False, "recoverable": False,
         "depends_on": [], "filter": "a label file with no valid box"},
        {"id": "V1", "name": "eval guard", "role": "guard", "unit": "image", "guard": True, "recoverable": False,
         "depends_on": ["V0"], "filter": "near an evaluation image"},
        {"id": "V2", "name": "join", "role": "join", "unit": "box", "guard": False, "recoverable": True,
         "depends_on": ["V1"], "filter": "source class name joined to a target"},
        {"id": "V3", "name": "name status", "role": "name_status", "unit": "class", "guard": False,
         "recoverable": True, "depends_on": ["V2"], "filter": "informative names"},
        {"id": "V4", "name": "reject sample", "role": "reject_class_sample", "unit": "box", "guard": False,
         "recoverable": "not a discard", "depends_on": ["V3"], "filter": "the reject class's training sample"},
        {"id": "V5", "name": "target check", "role": "target_check", "unit": "box", "guard": False,
         "recoverable": True, "depends_on": ["V2", "V4"], "filter": "probe verifies target-labelled boxes"},
        {"id": "V6", "name": "other check", "role": "other_check", "unit": "box", "guard": False,
         "recoverable": True, "depends_on": ["V2", "V4"], "filter": "probe checks other-labelled boxes"},
        {"id": "V7", "name": "image rule", "role": "image_rule", "unit": "image", "guard": False,
         "recoverable": True, "depends_on": ["V5", "V6"], "filter": "every box verified"},
        {"id": "V8", "name": "evidence", "role": "evidence", "unit": "source", "guard": False,
         "recoverable": "derived", "depends_on": ["V5"], "filter": "a source with verified boxes"}]
    return {
        "format": "funnel-domain/1", "domain": "vehicles", "adapter": "vehicle_logs",
        "domain_terms": ["roadcam", "fleetcam"],
        "classes": {"targets": [
            {"id": 0, "name": "car", "common": "car", "taxon": "car", "rank": "species", "genus": "motorcar",
             "not": ["van"], "siblings": []},
            {"id": 1, "name": "truck", "common": "truck", "taxon": "truck", "rank": "species", "genus": "lorry",
             "not": [], "siblings": ["bus"]},
            {"id": 2, "name": "bus", "common": "bus", "taxon": "bus", "rank": "species", "genus": "coach",
             "not": [], "siblings": ["truck"]}],
            "other": {"id": 3, "name": "OtherObject"}, "genus_answer_unsure_for": [],
            "small_class_train_boxes_below": 100},
        "attractors": [{"id": "van", "taxon": "van", "common": "van", "rank": "species",
                        "confused_with": ["car"], "option": "A van"}],
        "names": {"numeric_regex": "[0-9]+", "generic_keys": ["object", "vehicle"], "generic_regex": "(class|obj)[0-9]*",
                  "non_object_words": ["sign", "sky"], "state_words": ["blurred"], "role_names": ["traffic"],
                  "related_tokens": ["lorry"], "related_allowed_keys": [],
                  "frames": {"noinfo": ["no_name", "numeric", "generic", "unresolvable"],
                             "named": ["taxon_resolved", "target_related", "target_synonym", "role"],
                             "excluded": ["non_object", "state"]}},
        "taxonomy": {"authority": {"kind": "hypernym_table",
                                   "table": {"sedan": "car", "hatchback": "car", "pickup": "truck",
                                             "lorry": "truck", "coach": "bus", "minibus": "bus", "van": "van"}},
                     "overrides": {}, "informative_via": ["scientific"], "mappable_via": ["scientific", "override"]},
        "stages": stages,
        "exams": {"decision": "dev", "non_decision": ["test", "night_exam", "rain_exam"], "extra_non_decision": []},
        "sources": {"reference": "roadref", "lab_groups": {"RoadLab": ["roadref", "roadcam_a"]}},
        "known_truth": {"KV1": {"what": "copies of reference images", "allowed_uses": ["calibration"],
                                "independent": False, "claimed_by": None, "never_qualifies": [],
                                "sources": ["roadref"], "split": None},
                        "qualify_rl_on": ["KV1"], "forbidden": ["dev", "test", "night_exam", "rain_exam"]},
    }


def _sha_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def vehicles_ledger(config_path):
    """A funnel-ledger/1 of the vehicles domain (synthetic): S1-S5 hold with the
    weed domain's thresholds, and no known-truth set lies outside the
    reference domain."""
    from weed_optimizer_framework.tools.funnel import domain as FD
    from weed_optimizer_framework.tools.funnel import ledger as FL
    dom = FD.load(config_path)
    ident = {"events": _sha_text("vehicles/events"), "summary": _sha_text("vehicles/summary")}
    led = FL.new(dom, dict(ident), "summaries", "v1", ident)
    kts = [{"id": "copies:roadref", "sha256": _sha_text("vehicles/kv1"), "sources": ["roadref"],
            "domain_score": [0.1, 1.0], "domain_basis": "copies of the reference images"}]
    cal = {"known_truth_sets": kts, "precision": 1.0, "recall": 0.95, "domains_covered": ["roadref"]}
    for st in [
        {"id": "V0", "filter": "read", "version": "1", "unit": "image", "role": "read", "depends_on": [],
         "recoverable": False, "guard": False, "in": 5000, "kept": 4800, "discarded": {"no_boxes": 200}},
        {"id": "V1", "filter": "eval guard", "version": "1", "unit": "image", "role": "guard", "depends_on": ["V0"],
         "recoverable": False, "guard": True, "in": 4800, "kept": 4700, "discarded": {"near_eval": 100}},
        {"id": "V2", "filter": "join", "version": "1", "unit": "box", "role": "join", "depends_on": ["V1"],
         "recoverable": True, "guard": False, "in": 20000, "kept": 20000, "discarded": {},
         "kept_by_label": {"car": 900, "truck": 300, "bus": 200, "OtherObject": 18600}},
        {"id": "V3", "filter": "name status", "version": "1", "unit": "class", "role": "name_status",
         "depends_on": ["V2"], "recoverable": True, "guard": False, "in": 9, "kept": 9, "discarded": {}},
        {"id": "V4", "filter": "reject sample", "version": "1", "unit": "box", "role": "reject_class_sample",
         "depends_on": ["V3"], "recoverable": "not a discard", "guard": False},
        {"id": "V5", "filter": "target check", "version": "1", "unit": "box", "role": "target_check",
         "depends_on": ["V2", "V4"], "recoverable": True, "guard": False, "in": 1400, "kept": 300,
         "discarded": {"conflict": 400, "unknown": 700}, "kept_by_label": {"car": 250, "truck": 45, "bus": 5},
         "by_source": {"fleet_b": {"in": 600, "kept": 20, "discarded": {"conflict": 280, "unknown": 300}}},
         "calibration": copy.deepcopy(cal)},
        {"id": "V6", "filter": "other check", "version": "1", "unit": "box", "role": "other_check",
         "depends_on": ["V2", "V4"], "recoverable": True, "guard": False, "in": 18600, "kept": 18000,
         "discarded": {"conflict": 600}, "calibration": copy.deepcopy(cal)},
        {"id": "V7", "filter": "image rule", "version": "1", "unit": "image", "role": "image_rule",
         "depends_on": ["V5", "V6"], "recoverable": True, "guard": False, "in": 4700, "kept": 4000,
         "discarded": {"conflict": 500, "unknown": 200}},
        {"id": "V8", "filter": "evidence", "version": "1", "unit": "source", "role": "evidence",
         "depends_on": ["V5"], "recoverable": "derived", "guard": False, "in": 3, "kept": 1,
         "discarded": {"not_evidenced": 2}}]:
        FL.add_stage(led, st)
    led["label_spaces"] = {
        "roadcam_a": {"classes": 4, "kinds": {"named": 8000, "numeric": 0, "none": 0, "generic": 0}, "boxes": 8000},
        "fleet_b": {"classes": 3, "kinds": {"named": 0, "numeric": 7000, "none": 0, "generic": 0}, "boxes": 7000},
        "dashcam_c": {"classes": 1, "kinds": {"named": 0, "numeric": 0, "none": 5000, "generic": 0}, "boxes": 5000}}
    led["domain_scores"] = {"roadcam_a": 0.45, "fleet_b": 0.03, "dashcam_c": 0.02,
                            "reference": {"q05": 0.1, "q50": 0.5}}
    probs = FL.validate(led)
    if probs:
        raise SystemExit("the vehicles ledger does not validate: %s" % probs)
    return led


def r9b_ledger(q05_q50, s1_gate):
    """The funnel ledger that could have been written from the artifacts of
    2026-08-25 alone (synthetic): the S1 gate's audited sources as one
    target-check stage (13,527 labelled images, 3,208 in a source that cleared
    the 0.90 bar) and its one known truth, human-labelled cwd12, which is the
    reference domain itself. The reference scale (train_core's q05 and q50)
    is a property of train_core, which is unchanged since then."""
    from weed_optimizer_framework.tools.funnel import domain as FD
    from weed_optimizer_framework.tools.funnel import ledger as FL
    dom = FD.load("weed")
    ident = {"s1_gate_verdict_2026_08_25": _sha_text(json.dumps(s1_gate, sort_keys=True))}
    led = FL.new(dom, dict(ident), "summaries", "v1", ident)
    total, passing = s1_gate["audited_harvested_labelled_images"], s1_gate["passing_bar_labelled_images"]
    FL.add_stage(led, {"id": "S8", "filter": "the S1 gate's probe audit of harvested labels (0.90 bar)",
                       "version": "2026-08-25", "unit": "image", "role": "target_check", "depends_on": [],
                       "recoverable": True, "guard": False, "in": total, "kept": passing,
                       "discarded": {"below_bar": total - passing},
                       "calibration": {"known_truth_sets": [
                           {"id": "cwd12_human_labelled", "sha256": _sha_text("cwd12 human labels"),
                            "sources": ["train_core"], "domain_score": [q05_q50[0], 1.0],
                            "domain_basis": "the reference domain itself (the probe reads 1.000 on it)"}],
                           "precision": 1.0, "recall": None, "domains_covered": ["train_core"]}})
    led["domain_scores"] = {"reference": {"q05": q05_q50[0], "q50": q05_q50[1]}}
    probs = FL.validate(led)
    if probs:
        raise SystemExit("the R9b ledger does not validate: %s" % probs)
    return led


def highyield_ledger(weed_ledger):
    """The D19 negative control (synthetic): a Step 1 whose filters kept most
    of the target evidence (S1 < 1), with a known-truth set outside the
    reference domain (S4 false) and no reject-class record (S6 not evaluated)."""
    led = copy.deepcopy(weed_ledger)
    ref = led["domain_scores"]["reference"]
    for st in led["stages"]:
        if st["role"] == "target_check":
            st["in"], st["kept"], st["discarded"] = 7281, 6000, {"conflict": 500, "unknown": 781}
            st["kept_by_label"] = None
            st["discarded_by_label"] = None
        if st["role"] == "other_check":
            st["kept"] = st["in"] - 300
            st["discarded"] = {"conflict": 300}
        if st["role"] in ("target_check", "other_check") and st.get("calibration"):
            st["calibration"]["known_truth_sets"].append(
                {"id": "shifted:independent", "sha256": _sha_text("shifted known truth"), "sources": ["kt7"],
                 "domain_score": [0.0, ref["q05"] / 2.0], "domain_basis": "an independent shifted set"})
    return led


def audits(fingerprint, ledger):
    """The R10-R12 audits (synthetic funnel-audit/1): R10 every stratum's FN
    upper bound < 0.05; R11 weed_crop's rejected Ragweed stratum (FN lb 0.6)
    and greenhouse's OtherPlant -> Palmer stratum (lb 0.4, but its source
    taxon is A. retroflexus, a relative); R12 invalid (a judge's calibration
    shares a lab with a stratum)."""
    stages = {s["id"]: s for s in ledger["stages"]}
    verified = stages["S8"]["kept"]

    def row(stage, stratum, kind, lb, rlb, taxa=(), rel=False):
        return {"stage": stage, "stratum": stratum, "kind": kind, "fn_lb": lb, "recoverable_lb": rlb,
                "source_taxa": list(taxa), "relative_of_prediction": rel}
    base = {"format": "funnel-audit/1", "synthetic": True, "domain": "weed", "ledger_fingerprint": fingerprint,
            "valid": True, "calibration_overlap": []}
    r10 = dict(base, why="R10: every stratum's FN upper bound is below 0.05",
               strata=[{"group": "G2", "stratum": "G2/source=project_agml__weed_crop_detection/label=Ragweed/fail=p_below_tau",
                        "N": 1072, "n": 60, "estimate": 0.0, "interval": [0.0, 0.041], "method": "jeffreys"}],
               stages={"S8": {"fn_rate": {"estimate": 0.01, "interval": [0.0, 0.04], "n": 300, "method": "kg"}},
                       "S9": {"fn_rate": {"estimate": 0.01, "interval": [0.0, 0.03], "n": 300, "method": "kg"}}},
               d18_inputs=[row("S8", "G2/source=project_agml__weed_crop_detection/label=Ragweed/fail=p_below_tau",
                               "target_rejected", 0.0, 0.0),
                           row("S9", "G1/frame=noinfo/status=no_name/pred=PalmerAmaranth",
                               "other_predicted_target", 0.01, 3.0)],
               stage_ranking=[{"stage": "S9", "recoverable_lb": 3.0}, {"stage": "S8", "recoverable_lb": 0.0}])
    r11 = dict(base, why="R11: weed_crop's rejected Ragweed stratum is a false-negative source; greenhouse's "
                         "OtherPlant -> Palmer stratum names a relative of the prediction",
               stages={"S8": {"fn_rate": {"estimate": 0.7, "interval": [0.6, 0.8], "n": 60, "method": "wilson"}},
                       "S9": {"fn_rate": {"estimate": 0.5, "interval": [0.4, 0.6], "n": 60, "method": "wilson"}}},
               d18_inputs=[row("S8", "G2/source=project_agml__weed_crop_detection/label=Ragweed/fail=p_below_tau",
                               "target_rejected", 0.6, round(0.5 * verified + 1.0, 1)),
                           row("S9", "G1/frame=named/status=taxon_resolved/pred=PalmerAmaranth",
                               "other_predicted_target", 0.4, round(0.25 * verified + 1.0, 1),
                               taxa=["Amaranthus retroflexus"], rel=True)],
               stage_ranking=[{"stage": "S8", "recoverable_lb": round(0.5 * verified + 1.0, 1)},
                              {"stage": "S9", "recoverable_lb": round(0.25 * verified + 1.0, 1)}])
    r12 = dict(base, valid=False, why="R12: J-knn2's bank shares a lab with an audited stratum",
               calibration_overlap=[{"judge": "J-knn2", "stratum": "G2/source=project_agml__weed_crop_detection",
                                     "shared": {"lab": ["NDSU"]}}],
               d18_inputs=copy.deepcopy(r11["d18_inputs"]), stage_ranking=copy.deepcopy(r11["stage_ranking"]))
    return r10, r11, r12


def class_maps_r11():
    return {"format": "funnel-class-maps/1", "synthetic": True,
            "proposals": [
                {"source": "project_agml__mh_weed16_weed_detection", "src_id": "12", "src_name": "12",
                 "map_to": "Sicklepod", "via": "card+geometry", "card": {"path": "cards/mh_weed16/table2.json",
                                                                          "sha256": _sha_text("table2"),
                                                                          "table_source": "PMC12179629 Table 2"},
                 "geometry": {"alignment": "identity", "agreement": 0.995}, "status": "proposed",
                 "reason": "card and geometry agree"},
                {"source": "rf_srec__crop-weed-poxtn", "src_id": "2", "src_name": "2", "map_to": None,
                 "via": "card", "card": None, "geometry": None, "status": "to_L14",
                 "reason": "the card lists 3 classes, the source has 4: the class count disagrees with the card"}]}


def _ca(argument, mechanism, cites, stage, metric, direction, threshold, test, falsifier, stratum=None):
    return {"argument": argument, "mechanism": mechanism, "evidence_cites": cites, "lit_cites": [],
            "prediction": {"stage": stage, "stratum": stratum, "metric": metric, "direction": direction,
                           "threshold": threshold},
            "cheapest_test": test, "falsifier": falsifier}


def da_positive(root, ledger):
    """The R14 positive reply: the six counter-arguments and two concessions of
    contract 8.6, each grounded in values read from the copies here."""
    j = lambda rel: json.loads((root / rel).read_text())      # noqa: E731
    sel, pool, admit, cal = (j("step1/select_summary.json"), j("step1/pool_summary.json"),
                             j("step1/admit_summary.json"), j("step1/calibration.json"))
    rep = j("realloop_v1/report.json")
    led = "funnel/funnel_ledger.json"
    idx = {s["id"]: i for i, s in enumerate(ledger["stages"])}
    gh, wc = "project_agml__greenhouse_crop_weed_detection", "project_agml__weed_crop_detection"
    ragweed_id = next(k for k, v in pool["per_slug"][wc]["join"].items() if v[1] == "Ragweed")
    pig_id = next(k for k, v in pool["per_slug"][gh]["join"].items() if v[0] == "Redroot Pigweed")
    mh = "project_agml__mh_weed16_weed_detection"
    unv = next(i for i, s in enumerate(rep["steps"]) if s["step"] == "UNVERIFIED")
    rec = sorted(s["id"] for s in ledger["stages"] if s.get("recoverable") is True)
    weights = {"S8": 0.4, "S9": 0.2, "S7": 0.15, "S10": 0.1, "S7b": 0.1, "S3": 0.03, "S0": 0.02}
    forecast = {s: weights[s] for s in rec}
    cas = [
        _ca("The verifier's recall outside its own domain was never measured.",
            "Every known-truth set the thresholds were set on is a copy of the reference photographs, while most "
            "species-bearing sources score below the reference domain's 5th percentile.",
            [_cite("step1/select_summary.json", sel["retrieval"]["train_core_image_score_q05_q50"],
                   "/retrieval/train_core_image_score_q05_q50"),
             _cite("step1/select_summary.json", sel["retrieval"]["source_evidence"][wc]["median"],
                   "/retrieval/source_evidence/%s/median" % wc),
             _cite("step1/calibration.json", cal["cwd12_copies"]["calibration_only_slugs"],
                   "/cwd12_copies/calibration_only_slugs")],
            "S8", "fn_rate", "above", 0.1, {"lever": "L10", "params": {"verb": "census"}},
            "The audit's S8 false-negative rate has an upper bound below 0.10."),
        _ca("Paper-confirmed Ragweed labels were rejected.",
            "The source's class list names Ragweed and the join maps it to the target, yet almost none of those "
            "boxes were verified.",
            [_cite("step1/pool_summary.json", pool["per_slug"][wc]["join"][ragweed_id],
                   "/per_slug/%s/join/%s" % (wc, ragweed_id)),
             _cite("step1/admit_summary.json", admit["per_species"]["Ragweed"]["boxes"]["verified"],
                   "/per_species/Ragweed/boxes/verified")],
            "S8", "fn_rate", "above", 0.5, {"card": "X11"},
            "Reference labels show the rejected Ragweed boxes are mostly not Ragweed.",
            stratum="rejected Ragweed of an authoritative source"),
        _ca("Uninformative label spaces were never resolved.",
            "Sources with no class names or numeric ids can only be joined to the other class, whatever they hold.",
            [_cite(led, ledger["label_spaces"]["rf_tuf__weed-3434e"]["kinds"]["none"],
                   "/label_spaces/rf_tuf__weed-3434e/kinds/none"),
             _cite("step1/pool_summary.json", pool["per_slug"][mh]["join"]["12"], "/per_slug/%s/join/12" % mh)],
            "S7", "fn_rate", "above", 0.1, {"lever": "L12", "params": {}},
            "Every uninformative class resolves to non-target taxa."),
        _ca("The evidence criterion is circular.",
            "It admits a source only when the verifier verified its boxes, so it inherits the verifier's recall.",
            [_cite(led, ledger["stages"][idx["S12"]]["kept"], "/stages/%d/kept" % idx["S12"]),
             _cite("step1/admit_summary.json", admit["per_slug"][gh]["boxes"]["verified"],
                   "/per_slug/%s/boxes/verified" % gh)],
            "S8", "fn_rate", "above", 0.1, {"lever": "L10", "params": {"verb": "census"}},
            "The audited S8 false negatives do not change which sources are evidenced."),
        _ca("The truth arm contradicts the conclusion.",
            "The only increment drawn from rejected data is the only one the truth arm scored helps.",
            [_cite("realloop_v1/report.json", rep["steps"][unv]["truth"]["verdict"],
                   "/steps/%d/truth/verdict" % unv),
             _cite("step1/admit_summary.json", admit["per_slug"]["rf_tuf__weed-3434e"]["boxes"]["conflict"],
                   "/per_slug/rf_tuf__weed-3434e/boxes/conflict")],
            "S10", "truth_verdict", "helps", None, {"lever": "L10", "params": {"verb": "census"}},
            "Recovered data is neutral or hurts in the dose arm."),
        _ca("The reject class may be confounded with one source.",
            "The other-class training sample is drawn from named other-plant boxes, a pool one source's generic "
            "abbreviation dominates.",
            [_cite("step1/admit_summary.json", admit["otherplant_training_boxes"], "/otherplant_training_boxes"),
             _cite(led, ledger["reject_class"], "/reject_class")],
            "S9", "fn_rate", "above", 0.1, {"card": "X11"},
            "The reject-class sample's largest (source, name) cell holds less than half of it."),
    ]
    concessions = [
        {"claim_id": "C1", "why": "Much of the greenhouse other-plant to Palmer conflict may be redroot pigweed, a "
                                  "relative the source names, not a hidden Palmer amaranth.",
         "checked": [_cite("step1/pool_summary.json", pool["per_slug"][gh]["join"][pig_id],
                           "/per_slug/%s/join/%s" % (gh, pig_id)),
                     _cite("step1/admit_summary.json", admit["per_slug"][gh]["boxes"]["conflict"],
                           "/per_slug/%s/boxes/conflict" % gh)]},
        {"claim_id": "C1", "why": "Data that exists may still not help: the truth arm scores most verified steps "
                                  "neutral.",
         "checked": [_cite("realloop_v1/report.json", rep["steps"][0]["truth"]["verdict"], "/steps/0/truth/verdict"),
                     _cite("realloop_v1/report.json", rep["agreement"]["full"]["agree"], "/agreement/full/agree")]},
    ]
    return {"claim_id": "C1", "counter_arguments": cas, "concessions": concessions, "stage_forecast": forecast}


def da_sycophantic(ledger):
    """The R14 sycophantic reply: agreement without a single checked value (its
    stage forecast is valid, uniform over the recoverable stages, so what is
    tested is the concessions rule, not the forecast's)."""
    rec = sorted(s["id"] for s in ledger["stages"] if s.get("recoverable") is True)
    return {"claim_id": "C1", "counter_arguments": [],
            "concessions": [{"claim_id": "C1", "checked": [], "why": "The conclusion is right; the data are poor."},
                            {"claim_id": "C1", "why": "Nothing suggests the filters erred."}],
            "stage_forecast": {s: 1.0 / len(rec) for s in rec}}


def build(root):
    """Every fixture file under root; returns the manifest dict."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for rel, src in COPIES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL_INC / src, dst)
        files[rel] = {"sha256": sha(dst), "bytes": dst.stat().st_size, "from": "results/framework/inc/" + src}
    from weed_optimizer_framework.tools.funnel.adapters import inc_step1 as AD
    cwd = os.getcwd()
    try:
        os.chdir(str(root))
        if (root / LEDGER).exists():
            (root / LEDGER).unlink()
        AD.ledger_from_summaries("weed", "step1", "census_v0.json", LEDGER)
    finally:
        os.chdir(cwd)
    ledger = json.loads((root / LEDGER).read_text())
    derived = {LEDGER: {"sha256": sha(root / LEDGER), "bytes": (root / LEDGER).stat().st_size,
                        "by": "adapters.inc_step1.ledger_from_summaries('weed', 'step1', 'census_v0.json', ...) "
                              "run from this directory",
                        "inputs": ["census_v0.json", "step1/admit_summary.json", "step1/select_summary.json",
                                   "step1/pool_summary.json", "step1/calibration.json"]}}
    syn = {}

    def put(rel, obj, why):
        _write(root, rel, obj)
        syn[rel] = {"sha256": sha(root / rel), "bytes": (root / rel).stat().st_size, "why": why}
    put("claims/claims_seed.json", seed_claims(root),
        "C1 and C2 as runner 5.5.10 files them (C1 from RESEARCH_LOG.md line 154, C2 from docs/poster/"
        "figures_data.json /s1_gate_verdict_2026_08_25/reading), both open, timestamps fixed at the freeze")
    put("claims/claims_challenged.json", seed_claims(root, challenged=True),
        "the same register after a surviving devil's-advocate pass moved C1 to challenged (R10's starting point)")
    vc = vehicles_config()
    put("vehicles/vehicles.json", vc, "R13: a second domain (car, truck, bus and OtherObject; a hypernym table; "
                                      "no out-of-domain known truth)")
    put("vehicles/vehicles_ledger.json", vehicles_ledger(root / "vehicles" / "vehicles.json"),
        "R13: a funnel ledger of the vehicles domain with a numeric-named and a no-name source")
    put("vehicles/vehicles_claims.json",
        {"format": "funnel-claims/1", "claims": [_claim(
            "C1", "The harvested dashcam logs hold few usable bus boxes.", "scarcity", "vehicles harvest",
            "human-transcribed", [_cite("funnel/funnel_ledger.json", 5, "/stages/5/kept_by_label/bus")])]},
        "R13: the vehicles domain's negative claim")
    fig = json.loads((PKG_ROOT.parent / "docs" / "poster" / "figures_data.json").read_text())
    sel = json.loads((root / "step1" / "select_summary.json").read_text())
    put("r9b/r9b_ledger.json", r9b_ledger(sel["retrieval"]["train_core_image_score_q05_q50"],
                                          fig["s1_gate_verdict_2026_08_25"]),
        "R9b: the ledger reconstructed from what existed on 2026-08-25 (the S1 gate verdict in docs/poster/"
        "figures_data.json; train_core's reference scale from select_summary.json)")
    put("r9b/r9b_claims.json", {"format": "funnel-claims/1", "claims": [seed_claims(root)["claims"][1]]},
        "R9b: claim C2 alone")
    put("controls/highyield_ledger.json", highyield_ledger(ledger),
        "negative control: a high-yield Step 1 (S1 < 1) with an out-of-domain known-truth set (S4 false)")
    r10, r11, r12 = audits(ledger["fingerprint"], ledger)
    put("audits/r10_audit.json", r10, "R10: an audited negative (every stratum's FN upper bound < 0.05)")
    put("audits/r11_audit.json", r11, "R11: an audited positive with a sibling stratum")
    put("audits/r12_audit.json", r12, "R12: an invalid audit (a judge's calibration shares a lab with a stratum)")
    put("audits/r11_class_maps.json", class_maps_r11(),
        "R11: one card+geometry class map proposal and one whose class count disagrees with its card")
    put("da/da_positive.json", da_positive(root, ledger),
        "R14: the positive devil's-advocate reply (contract 8.6: six counter-arguments, two concessions); "
        "written by the contract's author, so it tests the validator's mechanics only")
    put("da/da_sycophantic.json", da_sycophantic(ledger), "R14: a sycophantic reply (concessions without checks)")
    return {"format": FORMAT, "frozen_utc": FROZEN_UTC,
            "what": "The funnel audit's replay fixtures (docs/FUNNEL_AUDIT.md 8.9; runner 5.5.10), read by "
                    "tests/test_funnel_ap_replay.py and pinned here: byte copies of the local results files, the "
                    "ledger derived from them by the Step 1 adapter, and synthetic files, each with its why. None "
                    "sits under a name evidence.load_dir reads from INC_DIR/funnel/.",
            "files": files, "derived": derived, "synthetic": syn,
            "pending": {"step1/verifier_fit_info.json": "the cluster projection of step1/verifier/fit_info.json "
                                                        "(adapters.inc_step1.verifier_fit_info_projection): not "
                                                        "local; S6 stays not evaluated until it is pulled"}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="funnel_fix_"))
        try:
            man = build(tmp)
            pinned = json.loads((FIX / "MANIFEST.json").read_text())
            bad = []
            for sec in ("files", "derived", "synthetic"):
                for rel, rec in man[sec].items():
                    if (pinned.get(sec) or {}).get(rel, {}).get("sha256") != rec["sha256"]:
                        bad.append(rel)
            print("rebuilt %d file(s); %d differ from the pins: %s"
                  % (sum(len(man[s]) for s in ("files", "derived", "synthetic")), len(bad), bad or "none"))
            return 1 if bad else 0
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)
    man = build(FIX)
    (FIX / "MANIFEST.json").write_text(dump(man), encoding="utf-8")
    print("wrote %s (%d copies, %d derived, %d synthetic)" % (FIX / "MANIFEST.json", len(man["files"]),
                                                              len(man["derived"]), len(man["synthetic"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
