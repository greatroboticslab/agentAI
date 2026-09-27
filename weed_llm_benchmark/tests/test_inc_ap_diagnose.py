#!/usr/bin/env python3
"""Each INC autopilot diagnosis fires, or stays silent, exactly by its
written rule (docs/INC_AUTOPILOT.md, (b); thresholds.json).

Pinned:
  * thresholds.json: every leaf has a value and a reason; every 'mirrors'
    value equals the pinned INC constant it restates (GateConfig
    p_recipe_flag, p_reject / p_accept, LABELS), and so does diagnose.py's
    restated vocabulary (verdicts, truth verdicts, default replay mode,
    builder names, the Bswap step, the attribution-not-run keys, the job
    name prefix); a missing threshold makes its rule 'unknown' and names the
    key; a rule that raises is 'unknown' with the error; a diagnosis fired
    without a cite is downgraded (model.diagnosis);
  * D1 on pilot_v1 (20/21, 21/21, 9/9 on I2, I3, I5; L1 + X1) and on
    synthetic ledgers: silent just below each share, when every truth-helps
    step is ACCEPTed, below min_gate_entries; X1 only in full mode; L2 (full
    replay) for a real loop; the same counts from report.json alone;
  * D2 on a synthetic Step 1 in remote.py's aggregate format: names the
    sources without domain evidence (L3), silent with a relevance.json made
    for this select build, fires again when it was made for another (stale)
    or when relevance.load would refuse it as malformed (another format, no
    calibration, made under another rule, a check that does not follow from
    its tau in either direction), then escalating
    to a person (OP_ESCALATE, crit) instead of an L3 the relevance builder
    would refuse; fires on a builder refusal naming relevance with no Step 1
    at all, not evaluated without the aggregate. A file made for this select
    build whose own calibration check failed (tau 0.0557 < 0.5): with Step 1
    evidence that holds the default real loop, D2 does not escalate: warn,
    card X9, then 'L2 with --increment-sources evidence', the criterion and
    its capacity in detail.increment_criterion, the flagged source among the
    excluded ones; with a pool that holds only a smaller loop, the R4 sizing
    rule (N 4, M = min(the default M, floor(images / 5))): warn, X9, then
    'L2 with --increment-sources evidence --size M --n-verified 4', the
    default and sized N and M and the rule in detail.increment_criterion.
    sizing; with a pool whose sized M falls below 5 % of B, summaries that
    disagree or no admit summary, it escalates (OP_ESCALATE and X9) with the
    reason; it fires on such a file whatever the per-source evidence says
    (no aggregate, no source flagged);
  * D2b: a passing source whose crops are mostly leaf disease; not a failing
    one, not a low share, not a file relevance.load would refuse;
  * D3 on pilot_v1 (Bswap, 3 chains, L4) and with an audit (at
    <exp>/audit/label_audit.json or INC_DIR/audit/<exp>_audit.json): X3
    when it separates the planted step, no lever when it does not; silent when the
    planted step is attributed to labels or there is none (b0_v1);
    D3b on pilot_v1 (s05_Breal, X4);
  * D4: pilot_v1 no_recipe_tracks_truth (3/7); synthetic pilots for each
    tie-break in order (rate, non-ACCEPT on truth-helps steps, final dev gap
    to T_final, GPU-hours), the exact 5/7 line (5/7 and 10/14 ready, 4/7 and
    9/14 not), an incomplete pilot, a full-replay pilot where D1 fires
    (silent), a sample-replay pilot that is ready while D1 fires on it
    (D1 takes precedence: decision_slot_ready with blocked_by D1 and D1's
    levers L1 + X1, never L2), the latest complete pilot, a real loop
    ignored, the levers when not ready (L1 + X1 in sample mode, X1 in full);
    D1 blocks D4 only when D4's own choice is touched (R4 review 2026-09-27):
    a chain D4 selects that ACCEPTed every truth-helps step leaves D4 ready
    (L2) in either replay mode while D1 fires with its own levers; a tie on
    rate goes to the chain with fewer misses; a selected chain with a miss,
    or a threshold not met, blocks; a real loop has no D4 choice (None);
  * the rules version (sha256 of diagnose.py + thresholds.json +
    levers.json, 12 hex), versioned prospective record names, and
    current_prospective answering only with a READY record of the current
    rules version;
  * D15 on pilot_v2 (pooled over (chain, step) pairs: the five flips-only
    REJECTs of truth-helps steps, freeze I2, I3, full I2, I3 and lora I3,
    each ledger line cited; pinned v1 from the decisions' config; L9; three
    truth-helps misses that failed another guard too, freeze I5 on
    regression and species, lora I2 and I5 on species, so D1 is NOT blocked:
    it fires with X1 and names them; D4 silent) and on synthetic ledgers:
    min_pairs, pairs in two chains adding up, the exact p_accept line and a
    decision's own config.p_accept, another failed guard, a non-helps or
    non-clean step, an unfinished experiment, a v2 pin (card X6, state.json
    gate_pin over exp.json), a real loop (card X8, D1 blocked with L2 + X1
    withheld), a mode that is neither (no lever, so D1 not blocked), a miss
    that is not flips-only (no block), and a ready sample pilot whose D1 D15
    blocks (D4 blocked_by D15, no lever); pilot_v1, b0_v1 and base_b_v1
    silent;
  * D16 on synthetic v2 pilots: refuted (crit, X7, D4 blocked_by D16, no
    L2; a non-clean acceptance named), supported (info, D4 ready with the
    gate recorded), inconclusive (warn, D4 ready only on its own threshold,
    v2 unproven), provisional, missing and unfinished checks silent; D4 on a
    ready v2 pilot whose check is missing or provisional: blocked_by D16
    (pending), no L2; prospective_d4 records the gate and the check, and
    writes nothing while the check is not final;
  * D5-D14 and DREF on synthetic state, context and ledgers, each both ways;
  * every cite of every fired diagnosis resolves to its value.

Run:  python3 tests/test_inc_ap_diagnose.py
"""
import copy
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "inc_replay"
# The fixture tree as it stood at intervention 1, before pilot_v2 existed: a whole-tree
# load also holds pilot_v2 (L1 on pilot_v1 applied, D4 campaign-wide on pilot_v2).
V1_ERA = ["pilot_v1", "b0_v1"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
FAILURES = []
EQ = {"ACCEPT": "helps", "HOLD": "neutral", "REJECT": "hurts"}


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def one(ev, did, **kw):
    return DG.by_id(DG.detect(ev, **kw))[did]


def cites_ok(ev, d):
    return all(ev.check_cite(c) for c in d["cites"])


def ev_of(objs, exp, context=None):
    texts = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in objs.items()}
    return E.from_texts(texts, exp, context=context)


def ledger(entries):
    return "".join(json.dumps(e) + "\n" for e in entries)


def gate(chain, k, step, verdict, clean=True, p_recipe=0.5, p_data=1.0, null_mean=0.73, inc=0.72,
         cand_mean=0.75, cand_sd=0.002, null_sd=0.002, cvl="none", not_run=("label_audit",)):
    return {"id": "gate/%s/%d" % (chain, k), "type": "gate", "chain": chain, "k": k, "step": step,
            "clean": clean, "attribution_not_run": {n: "not run" for n in not_run},
            "decision": {"verdict": verdict, "p_recipe": p_recipe, "p_data": p_data, "null_mean": null_mean,
                         "inc": inc, "cand_mean": cand_mean, "cand_sd": cand_sd, "null_sd": null_sd,
                         "attribution": {"class_vs_loc": cvl}}}


def truth(k, verdict, clean=True):
    return {"id": "truth/%d" % k, "type": "truth", "k": k, "clean": clean, "detail": {"verdict": verdict}}


def defn(exp, steps, recipes=("full", "freeze", "lora"), builder="inc.pilot build", replay_mode=None,
         utc="2026-09-27T04:00:00Z", **extra):
    d = {"exp": exp, "type": "chain", "builder": builder, "seeds": [0, 1, 2], "initialised_utc": utc,
         "base": {"name": "P0", "manifest": "/ocean/x/inc/%s/manifests/P0.jsonl" % exp, "n_images": 100},
         "steps": [dict({"name": n, "clean": c, "manifest": "/ocean/x/inc/%s/manifests/%s.jsonl" % (exp, n),
                         "n_images": 10}, **({"planted": p} if p else {})) for n, c, p in steps],
         "recipes": {r: {"epochs": 30} for r in recipes}}
    if replay_mode is not None:
        d["replay_mode"] = replay_mode
    d.update(extra)
    return d


def d4_report(exp, chains, truths, finals=None, gpu=None, replay_mode="sample", compared=None, p_recipe=0.5,
              null_mean=0.73, inc=0.72):
    steps = []
    for i, tv in enumerate(truths):
        steps.append({"k": i + 1, "step": "S%d" % (i + 1), "tag": "s%02d_S%d" % (i + 1, i + 1), "clean": True,
                      "truth": {"verdict": tv, "p": 1.0},
                      "chains": {r: {"verdict": v[i], "agree": EQ[v[i]] == tv, "p_recipe": p_recipe,
                                     "p_data": 1.0, "null_mean": null_mean, "inc": inc, "cand_mean": 0.75,
                                     "cand_sd": 0.002, "null_sd": 0.002, "attribution": {"class_vs_loc": "none"},
                                     "attribution_not_run": ["label_audit"]}
                                 for r, v in chains.items()}})
    agreement = {}
    for r, v in chains.items():
        a = sum(EQ[x] == t for x, t in zip(v, truths))
        n = len(truths) if compared is None else compared
        agreement[r] = {"agree": a, "compared": n, "rate": a / float(n)}
    rep = {"exp": exp, "type": "chain", "replay_mode": replay_mode, "steps": steps, "agreement": agreement,
           "label_steps": {"bswap": "Bswap"}, "gpu_hours": {}, "gpu_hours_total": 10.0, "final": []}
    for r in chains:
        rep["final"].append({"model": "chain %s: final incumbent" % r,
                             "exams": {"dev": {"twelve": {"mean": (finals or {}).get(r, 0.7), "n": 1}},
                                       "test": {"twelve": {"mean": 0.9}}}})
        rep["gpu_hours"]["chain:%s" % r] = {"hours": (gpu or {}).get(r, 2.0), "runs": 42}
    rep["final"].append({"model": "T_final (union of clean data)",
                         "exams": {"dev": {"twelve": {"mean": (finals or {}).get("T", 0.7), "n": 3}}}})
    rep["gpu_hours"]["truth"] = {"hours": 4.0, "runs": 21}
    return rep


def pilot_ev(chains, truths, exp="pilot_v1", replay_mode="sample", **kw):
    rep = d4_report(exp, chains, truths, replay_mode=replay_mode, **kw)
    d = defn(exp, [("S%d" % (i + 1), True, None) for i in range(len(truths))], recipes=tuple(chains),
             replay_mode=replay_mode)
    return {"%s/report.json" % exp: rep, "%s/exp.json" % exp: d}


# ------------------------------------------------------------------ tests
def test_thresholds():
    print("thresholds and vocabulary")
    from weed_optimizer_framework.tools.inc import driver as D
    from weed_optimizer_framework.tools.inc import gate as G
    from weed_optimizer_framework.tools.inc import report as R
    th = DG.load_thresholds()
    leaves = [(d, k, v) for d, blk in th.items() if not d.startswith("_") for k, v in blk.items()]
    check("every threshold has a value and a reason",
          all(isinstance(v, dict) and "value" in v and isinstance(v.get("why"), str) and len(v["why"]) > 20
              for _, _, v in leaves), [(d, k) for d, k, v in leaves if "value" not in v or not v.get("why")])
    cfg = G.GateConfig()
    check("D1.p_recipe_flag mirrors GateConfig.p_recipe_flag", th["D1"]["p_recipe_flag"]["value"] == cfg.p_recipe_flag)
    check("D8.p_band mirrors GateConfig p_reject, p_accept",
          th["D8"]["p_band"]["value"] == [cfg.p_reject, cfg.p_accept])
    check("D3.labels_attribution mirrors gate.LABELS", th["D3"]["labels_attribution"]["value"] == G.LABELS)
    check("D15.p_accept mirrors GateConfig.p_accept", th["D15"]["p_accept"]["value"] == cfg.p_accept)
    gsrc = (ROOT / "weed_optimizer_framework/tools/inc/gate.py").read_text()
    check("D15.flips_guard mirrors the flips guard's key in gate.decide",
          '"%s": _flips_guard(' % th["D15"]["flips_guard"]["value"] in gsrc)
    check("D15.v1_mode mirrors gate FLIPS_NEGATIVE, the default",
          th["D15"]["v1_mode"]["value"] == G.FLIPS_NEGATIVE == G.DEFAULT_FLIPS_MODE)
    check("D15.v2_mode and D16.v2_mode mirror gate FLIPS_NET",
          th["D15"]["v2_mode"]["value"] == th["D16"]["v2_mode"]["value"] == G.FLIPS_NET)
    check("D16.severity covers exactly report.py's v2_check outcomes",
          set(th["D16"]["severity"]["value"]) == set(R.V2_OUTCOME_TEXT), sorted(R.V2_OUTCOME_TEXT))
    mirrored = [(d, k) for d, k, v in leaves if "mirrors" in v]
    check("the checked mirrors are all the mirrors",
          sorted(mirrored) == [("D1", "p_recipe_flag"), ("D15", "flips_guard"), ("D15", "p_accept"),
                               ("D15", "v1_mode"), ("D15", "v2_mode"), ("D16", "severity"), ("D16", "v2_mode"),
                               ("D3", "labels_attribution"), ("D8", "p_band")], mirrored)
    check("the gate_pin key is the driver's", DG.GATE_PIN == D.GATE_PIN)
    check("the verdict vocabulary is the gate's", (DG.ACCEPT, DG.HOLD, DG.REJECT) == (G.ACCEPT, G.HOLD, G.REJECT)
          and (DG.HELPS, DG.HURTS, DG.NEUTRAL) == (G.HELPS, G.HURTS, G.NEUTRAL))
    check("the default replay mode is the driver's", DG.DEFAULT_REPLAY_MODE == D.DEFAULT_REPLAY_MODE)
    check("the attribution-not-run keys are the driver's",
          {DG.LABEL_AUDIT, DG.LOSO} == set(D.ATTRIBUTION_NOT_RUN))
    check("the Bswap step name is the report's", DG.BSWAP_STEP == R.BSWAP_STEP)
    src = (ROOT / "weed_optimizer_framework/tools/inc/pilot.py").read_text()
    check("the pilot builder name is pilot.py's", '"builder": "%s"' % DG.PILOT_BUILDER in src)
    rsrc = (ROOT / "weed_optimizer_framework/tools/inc/realloop.py").read_text()
    check("the real-loop builder name is realloop.py's", 'BUILDER = "%s"' % DG.REALLOOP_BUILDER in rsrc)
    dsrc = (ROOT / "weed_optimizer_framework/tools/inc/driver.py").read_text()
    check("D6's job prefix matches the driver's job names",
          '"inc_%s_%04d" % (self.exp, n)' in dsrc and th["D6"]["job_prefix"]["value"] == "inc_%s_")
    check("the D7 patterns match the driver's two code refusals",
          re.search(th["D7"]["error_patterns"]["value"][0], "the running package a differs from the git-tracked "
                    "copy b in [x]; sync") and re.search(th["D7"]["error_patterns"]["value"][1],
                                                         "the driver's code changed since experiment pilot_v2 was "
                                                         "pinned (t, init): x"))
    check("both D7 message fragments are in driver.py",
          "differs from the git-tracked copy" in dsrc and "code changed since experiment %s was pinned" in dsrc)
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    base = copy.deepcopy(th)
    del base["D1"]["share_null_below_inc_min"]
    d = one(ev, "D1", base=base)
    check("a missing threshold makes its rule unknown and names the key",
          not d["fired"] and "D1.share_null_below_inc_min" in d["summary"] and d["summary"].startswith("unknown"),
          d["summary"])
    base = copy.deepcopy(th)
    base["D4"]["ready_rate"]["value"] = 0.71
    d = one(ev, "D4", base=base)
    check("a malformed fraction is unknown too", not d["fired"] and "D4.ready_rate" in d["summary"], d["summary"])
    bad = ev_of({"x_v1/report.json": {"type": "chain", "steps": [{"step": "Bswap"}],
                                      "agreement": {"full": {"agree": "three", "compared": 1}}}}, "x_v1")
    d = one(bad, "D4")
    check("a rule that raises is unknown with the error", not d["fired"] and d["summary"].startswith("unknown")
          and "raised" in d["summary"], d["summary"])
    m = M.diagnosis("D9", "x", True, "warn", "s", [])
    check("a diagnosis fired without cites is downgraded", not m["fired"] and m["summary"].startswith("unknown"))
    only = DG.detect(ev, only=DG.HEALTH)
    check("detect(only=HEALTH) runs the health rules", [d["id"] for d in only] == list(DG.HEALTH))
    allids = [d["id"] for d in DG.detect(ev)]
    check("detect lists every rule, fired or not, in order", allids == [r[0] for r in DG.RULES], allids)


def test_d1():
    print("D1 recipe_forgets")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    d = one(ev, "D1")
    c = d["detail"]["counts"]
    check("pilot_v1: fires", d["fired"] and d["severity"] == "warn")
    check("pilot_v1: 20/21 P_recipe flagged, 21/21 null < inc, 9/9 on truth-helps steps not ACCEPT",
          (c["recipe_flagged"], c["gate_entries"], c["null_below_inc"], c["helps_not_accepted"], c["helps_entries"])
          == (20, 21, 21, 9, 9), c)
    check("pilot_v1: the truth-helps steps are I2, I3, I5", d["detail"]["helps_steps"] == ["I2", "I3", "I5"])
    check("pilot_v1: L1 then X1 (sample replay)", d["levers"] == ["L1", "X1"], d["levers"])
    check("pilot_v1: every cite resolves", cites_ok(ev, d))
    pr = [x for x in d["cites"] if x["pointer"] == "/decision/p_recipe"]
    check("pilot_v1: the 20 flagged P_recipe values are cited, each 0.0",
          len(pr) == 20 and all(x["value"] == 0.0 for x in pr), len(pr))
    check("pilot_v1: the clean flags come from exp.json",
          {x["pointer"] for x in d["cites"] if x["artifact"] == "pilot_v1/exp.json"}
          >= {"/steps/1/clean", "/steps/3/clean", "/steps/6/clean"})
    rep_only = ev_of({"pilot_v1/report.json": json.loads((FIX / "pilot_v1" / "report.json").read_text())},
                     "pilot_v1")
    d2 = one(rep_only, "D1")
    check("the same counts from report.json alone", d2["fired"] and d2["detail"]["counts"] == c
          and d2["detail"]["source"] == "report" and cites_ok(rep_only, d2), d2["detail"])

    steps = [("A", True, None), ("B", True, None), ("C", True, None), ("D", True, None)]

    def synth(pr_flags, null_below, verdict_on_helps="REJECT", n=4, mode=None, builder="inc.pilot build"):
        entries = [truth(k, "helps") for k in range(1, n + 1)]
        for k in range(1, n + 1):
            entries.append(gate("full", k, "ABCD"[k - 1], verdict_on_helps,
                                p_recipe=0.0 if k <= pr_flags else 0.5,
                                null_mean=0.70 if k <= null_below else 0.74))
        return ev_of({"x_v1/exp.json": defn("x_v1", steps[:n], recipes=("full",), replay_mode=mode, builder=builder),
                      "x_v1/ledger.jsonl": ledger(entries)}, "x_v1")
    check("2/4 flagged and 2/4 below: fires (>= 0.5)", one(synth(2, 2), "D1")["fired"])
    check("1/4 flagged: silent", not one(synth(1, 4), "D1")["fired"])
    check("1/4 null below inc: silent", not one(synth(4, 1), "D1")["fired"])
    check("every truth-helps step ACCEPTed: silent", not one(synth(4, 4, "ACCEPT"), "D1")["fired"])
    check("fewer than min_gate_entries: silent", not one(synth(2, 2, n=2), "D1")["fired"])
    d = one(synth(4, 4, mode="full"), "D1")
    check("full replay: X1 only", d["fired"] and d["levers"] == ["X1"], d["levers"])
    d = one(synth(4, 4, builder="inc.realloop build"), "D1")
    check("a real loop in sample mode: L2 with full replay, then X1",
          d["levers"] == ["L2", "X1"] and d["detail"]["child_replay_mode"] == "full", d["levers"])
    d = one(synth(4, 4), "D1", thresholds={"D1": {"share_recipe_flagged_min": 1.01}})
    check("a threshold override is honoured", not d["fired"])


def step1_world(rel=None, agg_sources=None):
    sel = {"sizes": {"base_B": 3927},
           "sources": {"increment_pool": {"csgo": 400, "weeds": 900, "tomato": 60}},
           "retrieval": {"source_evidence": {"weeds": {"species_crops": 800, "median": 0.8}}},
           "outputs": {"increment_pool.jsonl": {"sha256": "a" * 64}, "base_selected.jsonl": {"sha256": "d" * 64}}}
    agg = {"artifact": "step1/select_clusters.csv", "sources": agg_sources or {
        "csgo": {"increment_pool": {"images": 400, "status": {"no_evidence": 380, "below_gate": 20}}},
        "weeds": {"selected": {"images": 300, "status": {"selected": 300}},
                  "increment_pool": {"images": 900, "status": {"pool": 850, "below_gate": 50}}},
        "tomato": {"increment_pool": {"images": 60, "status": {"no_evidence": 50, "pool": 10}}}}}
    admit = {"per_slug": {"csgo": {"images": {"admitted": 400}, "boxes": {"other_ok": 3000, "verified": 0}},
                          "weeds": {"images": {"admitted": 1200}, "boxes": {"verified": 5000, "other_ok": 100}},
                          "tomato": {"images": {"admitted": 60}, "boxes": {"other_ok": 100}}}}
    objs = {"step1/select_summary.json": sel, "step1/select_clusters_by_source.json": agg,
            "step1/admit_summary.json": admit}
    if rel is not None:
        objs["step1/relevance.json"] = rel
    return objs


def relevance(sha="a" * 64, tomato=("pass", 0.8), base_sha="d" * 64, fmt="inc.relevance/1", check_ok=True):
    # made under relevance.py's rule (MIN_CROPS 20, CAL_PERCENTILE 5, TAU_MIN 0.5); tau 0.62 passes
    return {"format": fmt, "params": {"min_crops": 20, "calibration_percentile": 5.0, "tau_min": 0.5},
            "calibration": {"tau": 0.62, "check": {"ok": check_ok, "tau_min": 0.5}},
            "inputs": {"increment_pool": {"sha256": sha}, "base_selected": {"sha256": base_sha}},
            "increment_pool": {"sources": {
                "csgo": {"status": "fail", "images": 400, "top_set_share": {"plant": 0.0, "non_plant": 1.0,
                                                                            "leaf_disease": 0.0}},
                "weeds": {"status": "pass", "images": 900, "top_set_share": {"plant": 0.95, "non_plant": 0.0,
                                                                             "leaf_disease": 0.05}},
                "tomato": {"status": tomato[0], "images": 60, "top_set_share": {"plant": 0.1, "non_plant": 0.1,
                                                                                "leaf_disease": tomato[1]}}}}}


def test_d2():
    print("D2 unmeasured_source_property, D2b relevance_blind_spot")
    ev = ev_of(step1_world(), "x_v1")
    d = one(ev, "D2")
    names = [s["source"] for s in d["detail"]["sources"]]
    check("no relevance.json: fires naming the sources without domain evidence", d["fired"] and names == ["csgo"],
          (d["summary"], names))
    check("tomato (83% unevidenced) and weeds (species crops, verified boxes) are not named", "weeds" not in names)
    check("levers: L3 now, L2 with --relevance after", d["levers"] == ["L3"]
          and d["detail"]["then"] == ["L2 with --relevance"])
    check("its cites resolve (select summary, aggregate, admit summary)", cites_ok(ev, d)
          and {c["artifact"] for c in d["cites"]} == {"step1/select_summary.json",
                                                       "step1/select_clusters_by_source.json",
                                                       "step1/admit_summary.json"})
    ev = ev_of(step1_world(rel=relevance()), "x_v1")
    check("a relevance.json made for this select build: silent", not one(ev, "D2")["fired"])
    ev = ev_of(step1_world(rel=relevance(sha="b" * 64)), "x_v1")
    d = one(ev, "D2")
    check("a relevance.json made for another select build: fires (stale)", d["fired"]
          and d["detail"]["relevance"] == "stale" and cites_ok(ev, d), d["summary"])
    check("... and escalates to a person, not L3 (relevance build refuses over it without --force)",
          d["levers"] == ["OP_ESCALATE"] and d["severity"] == "crit" and d["detail"]["then"] == []
          and "--force" in d["detail"]["needs"], (d["levers"], d["detail"].get("needs")))
    ev = ev_of(step1_world(rel=relevance(base_sha="e" * 64)), "x_v1")
    d = one(ev, "D2")
    check("... also when only its base_selected differs (relevance.load's rule)", d["fired"]
          and d["detail"]["relevance"] == "stale" and cites_ok(ev, d), d["summary"])
    for label, rel in (("a check that does not follow from its tau (ok false at tau 0.62 >= tau_min 0.5)",
                        relevance(check_ok=False)),
                       ("a check that does not follow from its tau (ok true at tau 0.0557 < tau_min 0.5)",
                        dict(relevance(), calibration={"tau": 0.0557, "check": {"ok": True, "tau_min": 0.5}})),
                       ("a failed check made under another rule (params tau_min 0.3)",
                        dict(relevance(), params={"min_crops": 20, "calibration_percentile": 5.0, "tau_min": 0.3},
                             calibration={"tau": 0.0557, "check": {"ok": False, "tau_min": 0.5}})),
                       ("another format", relevance(fmt="inc.relevance/0")),
                       ("no calibration at all", dict(relevance(), calibration={}))):
        ev = ev_of(step1_world(rel=rel), "x_v1")
        d = one(ev, "D2")
        check("a relevance.json with %s is refused (relevance.load's cheap checks): D2 escalates" % label,
              d["fired"] and d["detail"]["relevance"] == "refused" and d["levers"] == ["OP_ESCALATE"]
              and cites_ok(ev, d), d["summary"])
        d = one(ev, "D2b")
        check("... and D2b does not read its statuses (%s)" % label,
              not d["fired"] and "not evaluated" in d["summary"], d["summary"])
    cal_failed = dict(relevance(), calibration={"tau": 0.0557, "check": {"ok": False, "tau_min": 0.5}})
    ev = ev_of(step1_world(rel=cal_failed), "x_v1")
    d = one(ev, "D2")
    check("a relevance.json whose own check failed, with Step 1 summaries that disagree (5000 verified boxes, 800 "
          "species crops, no file hashes): D2 escalates with the reason and card X9",
          d["fired"] and d["detail"]["relevance"] == "calibration_failed" and d["severity"] == "crit"
          and d["levers"] == ["OP_ESCALATE", "X9"] and "cannot be evaluated" in d["detail"]["needs"]
          and cites_ok(ev, d), (d["levers"], d["detail"].get("needs")))

    def evidenced_objs(base_b):
        objs = step1_world(rel=cal_failed)
        objs["step1/select_summary.json"]["sizes"]["base_B"] = base_b
        objs["step1/select_summary.json"]["inputs"] = {"verified": {"sha256": "1" * 64}, "crops": {"sha256": "2" * 64}}
        objs["step1/select_summary.json"]["retrieval"]["source_evidence"]["weeds"]["species_crops"] = 5000
        objs["step1/admit_summary.json"].update(verified_sha256="1" * 64, crops_sha256="2" * 64)
        return objs

    def evidenced(base_b):
        return ev_of(evidenced_objs(base_b), "x_v1")
    ev = evidenced(1000)
    d = one(ev, "D2")
    crit = d["detail"].get("increment_criterion") or {}
    check("the same file with evidence that holds the default loop (weeds: 900 images >= 7 x 100): D2 does not "
          "escalate: warn, X9, then L2 with --increment-sources evidence",
          d["fired"] and d["severity"] == "warn" and d["levers"] == ["X9"]
          and d["detail"]["then"] == ["L2 with --increment-sources evidence"] and "needs" not in d["detail"]
          and cites_ok(ev, d), (d["levers"], d["summary"]))
    check("... detail.increment_criterion: evidence, why (the failed check), the capacity, csgo excluded",
          crit.get("criterion") == "evidence" and "calibration check" in crit.get("why", "")
          and crit["capacity"]["evidenced_pool_images"] == 900 and crit["capacity"]["needed_images"] == 700
          and crit.get("flagged_excluded") == ["csgo"], crit)
    check("... it cites the failed check", {("step1/relevance.json", "/calibration/check/ok"),
                                            ("step1/relevance.json", "/calibration/tau")}
          <= {(c["artifact"], c["pointer"]) for c in d["cites"]})
    check("... and D2b does not read the refused file's statuses", not one(ev, "D2b")["fired"])
    ev = evidenced(3927)
    d = one(ev, "D2")
    sz = d["detail"]["increment_criterion"].get("sizing") or {}
    check("too small an evidenced pool (900 images < 7 x 393; the R4 sizing rule's M = min(393, floor(900 / 5)) = "
          "180 < 0.05 x 3,927 = 196.35): D2 escalates with the numbers (a review decision)",
          d["fired"] and d["severity"] == "crit" and d["levers"] == ["OP_ESCALATE", "X9"]
          and "review decision" in d["detail"]["needs"] and "at most --n-verified 1" in d["detail"]["needs"]
          and "= 180 images" in d["detail"]["needs"] and "196.35" in d["detail"]["needs"]
          and sz.get("applied") is False and (sz.get("sized") or {}).get("increment_images") == 180
          and d["detail"]["increment_criterion"]["criterion"] is None and cites_ok(ev, d), d["detail"].get("needs"))
    ev = evidenced(2000)
    d = one(ev, "D2")
    crit = d["detail"].get("increment_criterion") or {}
    sz = crit.get("sizing") or {}
    check("an evidenced pool that holds only a smaller loop (900 images < 7 x 200, base B 2,000): D2 does not "
          "escalate: the R4 sizing rule, N 4 x M 180 (min(200, floor(900 / 5)), above 0.05 x 2,000 = 100); warn, X9, "
          "then L2 with --increment-sources evidence --size 180 --n-verified 4; the detail records default, sized, rule",
          d["fired"] and d["severity"] == "warn" and d["levers"] == ["X9"] and "needs" not in d["detail"]
          and d["detail"]["then"] == ["L2 with --increment-sources evidence --size 180 --n-verified 4"]
          and crit.get("criterion") == "evidence" and sz.get("applied") is True
          and (sz["default"]["n_verified"], sz["default"]["increment_images"], sz["default"]["needed_images"])
          == (6, 200, 1400)
          and (sz["sized"]["n_verified"], sz["sized"]["increment_images"], sz["sized"]["needed_images"]) == (4, 180, 900)
          and crit["capacity"]["needed_images"] == 900 and "R4 review" in sz.get("rule", "") and cites_ok(ev, d),
          (d["levers"], d["detail"]["then"], sz.get("sized")))
    # A calibration-failed file fires D2 whatever the per-source domain evidence says.
    objs = evidenced_objs(1000)
    del objs["step1/select_clusters_by_source.json"]
    ev = ev_of(objs, "x_v1")
    d = one(ev, "D2")
    check("calibration failed, no per-source aggregate (nothing flagged): D2 still fires, warn, X9, then L2 with "
          "--increment-sources evidence (the capacity reads only the select and admit summaries)",
          d["fired"] and d["severity"] == "warn" and d["levers"] == ["X9"] and d["detail"]["sources"] == []
          and d["detail"]["then"] == ["L2 with --increment-sources evidence"]
          and "not evaluated" in d["summary"] and d["detail"]["increment_criterion"]["criterion"] == "evidence"
          and cites_ok(ev, d), (d["fired"], d["levers"], d["summary"][:160]))
    objs = step1_world(rel=cal_failed)
    del objs["step1/admit_summary.json"]
    ev = ev_of(objs, "x_v1")
    d = one(ev, "D2")
    check("calibration failed, no admit_summary.json: D2 escalates (crit, OP_ESCALATE, X9): the evidence criterion "
          "cannot be read",
          d["fired"] and d["severity"] == "crit" and d["levers"] == ["OP_ESCALATE", "X9"]
          and "admit_summary.json is not in the evidence" in d["detail"]["needs"]
          and d["detail"]["increment_criterion"]["criterion"] is None and cites_ok(ev, d),
          (d["fired"], d["levers"], d["detail"].get("needs")))
    objs = evidenced_objs(1000)
    objs["step1/select_clusters_by_source.json"]["sources"]["csgo"]["increment_pool"]["status"] = {"pool": 400}
    objs["step1/select_clusters_by_source.json"]["sources"]["tomato"]["increment_pool"]["status"] = {"pool": 60}
    ev = ev_of(objs, "x_v1")
    d = one(ev, "D2")
    check("calibration failed, no source flagged: D2 still fires, warn, X9, then L2 with --increment-sources evidence",
          d["fired"] and d["severity"] == "warn" and d["levers"] == ["X9"] and d["detail"]["sources"] == []
          and d["detail"]["then"] == ["L2 with --increment-sources evidence"] and cites_ok(ev, d),
          (d["fired"], d["levers"], d["summary"][:160]))
    ctx = {"refusals": [{"builder": "inc.realloop", "message": "[inc.realloop] ERROR: no /ocean/x/inc/step1/"
                                                               "relevance.json: the verified and OTHER_HEAVY "
                                                               "increments are drawn only from sources that pass "
                                                               "the relevance filter (run inc.relevance build, or "
                                                               "pass --relevance)"}]}
    ev = ev_of({}, "x_v1", context=ctx)
    d = one(ev, "D2")
    check("a realloop refusal naming relevance fires D2 with no Step 1 at all",
          d["fired"] and d["levers"] == ["L3"] and cites_ok(ev, d), d["summary"])
    check("... and DREF leaves it to D2", not one(ev, "DREF")["fired"])
    objs = step1_world()
    del objs["step1/select_clusters_by_source.json"]
    d = one(ev_of(objs, "x_v1"), "D2")
    check("without the per-source aggregate: not evaluated", not d["fired"] and "not evaluated" in d["summary"])
    check("without Step 1: not evaluated", "not evaluated" in one(ev_of({}, "x_v1"), "D2")["summary"])
    ev = ev_of(step1_world(rel=relevance()), "x_v1")
    d = one(ev, "D2b")
    check("D2b: a passing source whose crops are mostly leaf disease fires, X2",
          d["fired"] and [s["source"] for s in d["detail"]["sources"]] == ["tomato"] and d["levers"] == ["X2"]
          and cites_ok(ev, d), d["summary"])
    check("D2b: a failing one is silent",
          not one(ev_of(step1_world(rel=relevance(tomato=("fail", 0.8))), "x_v1"), "D2b")["fired"])
    check("D2b: a low leaf-disease share is silent",
          not one(ev_of(step1_world(rel=relevance(tomato=("pass", 0.3))), "x_v1"), "D2b")["fired"])
    check("D2b: no relevance.json is silent", not one(ev_of(step1_world(), "x_v1"), "D2b")["fired"])


def test_d3():
    print("D3 attribution_control_failed, D3b source_attribution_missing")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    d = one(ev, "D3")
    check("pilot_v1: fires on Bswap in all three chains, L4",
          d["fired"] and len(d["detail"]["failed"]) == 3 and d["levers"] == ["L4"], d["summary"])
    ptrs = {(c["artifact"], c["pointer"]) for c in d["cites"]}
    check("pilot_v1: cites exp.steps[2].planted", ("pilot_v1/exp.json", "/steps/2/planted") in ptrs)
    check("pilot_v1: cites chains.*.bswap.attributed_to_labels = false",
          all(("pilot_v1/report.json", "/chains/%s/bswap/attributed_to_labels" % r) in ptrs
              for r in ("freeze", "full", "lora"))
          and all(c["value"] is False for c in d["cites"] if c["pointer"].endswith("attributed_to_labels")))
    check("pilot_v1: cites the label audit as not run (ledger and report)",
          any(c["pointer"] == "/attribution_not_run/label_audit" for c in d["cites"])
          and ("pilot_v1/report.json", "/attribution_scope/not_run/4") in ptrs)
    check("pilot_v1: every cite resolves", cites_ok(ev, d))
    steps = [("I1", True, None), ("Bswap", False, "40% of boxes relabelled to another species")]
    base = {"x_v1/exp.json": defn("x_v1", steps, recipes=("full",)),
            "x_v1/ledger.jsonl": ledger([gate("full", 1, "I1", "ACCEPT"),
                                         gate("full", 2, "Bswap", "REJECT", clean=False, cvl="domain/localisation")])}
    audit = {"audits": {"I1": {"species": {"above_baseline": False}}, "Bswap": {"species": {"above_baseline": True}}}}
    ev = ev_of(dict(base, **{"x_v1/audit/label_audit.json": audit}), "x_v1")
    d = one(ev, "D3")
    check("an audit that separates the planted step: X3", d["fired"] and d["levers"] == ["X3"]
          and d["detail"]["audit_separates"] and cites_ok(ev, d), d["summary"])
    ev = ev_of(dict(base, **{"audit/x_v1_audit.json": audit}), "x_v1")
    d = one(ev, "D3")
    check("the same audit written under INC_DIR/audit/ (as pilot_v1's was) is read: X3, not a second L4",
          d["fired"] and d["levers"] == ["X3"] and cites_ok(ev, d)
          and any(c["artifact"] == "audit/x_v1_audit.json" for c in d["cites"]), d["summary"])
    audit["audits"]["I1"]["species"]["above_baseline"] = True
    d = one(ev_of(dict(base, **{"x_v1/audit/label_audit.json": audit}), "x_v1"), "D3")
    check("an audit that flags a clean step too: no lever (a person decides)",
          d["fired"] and d["levers"] == [] and not d["detail"]["audit_separates"], d["summary"])
    ok = dict(base, **{"x_v1/ledger.jsonl": ledger([gate("full", 2, "Bswap", "REJECT", clean=False, cvl="labels")])})
    check("the planted step attributed to labels: silent", not one(ev_of(ok, "x_v1"), "D3")["fired"])
    ev = E.load_dir(FIX, "b0_v1", exps=["b0_v1"])
    check("b0_v1 (a baseline): silent", not one(ev, "D3")["fired"])
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    d = one(ev, "D3b")
    check("D3b pilot_v1: fires on s05_Breal in 3 chains, X4", d["fired"] and d["detail"]["steps"] == ["s05_Breal"]
          and d["levers"] == ["X4"] and cites_ok(ev, d), d["summary"])
    check("D3b: cites rejects_without_source_attribution",
          any(c["pointer"] == "/chains/full/rejects_without_source_attribution" and c["value"] == ["s05_Breal"]
              for c in d["cites"]))
    one_src = ev_of({"x_v1/ledger.jsonl": ledger([gate("full", 1, "I1", "REJECT")])}, "x_v1")
    check("D3b: a single-source REJECT is silent", not one(one_src, "D3b")["fired"])


def test_d4():
    print("D4 decision_slot_ready / no_recipe_tracks_truth")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    d = one(ev, "D4")
    check("pilot_v1: no_recipe_tracks_truth", d["fired"] and d["name"] == "no_recipe_tracks_truth"
          and d["detail"]["rates"] == {"full": "3/7", "freeze": "3/7", "lora": "3/7"}, d["summary"])
    check("pilot_v1: best rate 0.4286 < 5/7; levers L1 + X1", abs(d["detail"]["ranking"][0]["rate"] - 3 / 7.0) < 1e-12
          and d["levers"] == ["L1", "X1"])
    check("pilot_v1: every cite resolves", cites_ok(ev, d))
    H, U, N = "helps", "hurts", "neutral"
    A, R_, Ho = "ACCEPT", "REJECT", "HOLD"

    def run(chains, truths, **kw):
        ev = ev_of(pilot_ev(chains, truths, **kw), "pilot_v1")
        return ev, one(ev, "D4")
    ev, d = run({"full": [A] * 6 + [R_], "freeze": [A] * 5 + [R_, R_]}, [H] * 7)
    check("rate decides: full 6/7 over freeze 5/7, ready, L2",
          d["name"] == "decision_slot_ready" and d["detail"]["recipes"] == ["full"]
          and d["detail"]["decided_by"] == "rate" and d["levers"] == ["L2"] and cites_ok(ev, d), d["summary"])
    truths = [H, H, H, H, U, U, N]
    x = [A, A, A, A, A, A, Ho]          # 5/7, every helps step ACCEPTed
    y = [A, A, A, R_, R_, R_, A]        # 5/7, one helps step not ACCEPTed
    ev, d = run({"full": y, "freeze": x}, truths, gpu={"full": 1.0, "freeze": 3.0})
    check("tie on rate: fewer non-ACCEPT on truth-helps steps wins (before GPU-hours)",
          d["detail"]["recipes"] == ["freeze"] and d["detail"]["decided_by"] == "helps_not_accepted", d["detail"])
    ev, d = run({"full": x, "freeze": x}, truths, finals={"full": 0.75, "freeze": 0.80, "T": 0.79},
                gpu={"full": 1.0, "freeze": 3.0})
    check("tie on both: the smaller final dev gap to T_final wins (before GPU-hours)",
          d["detail"]["recipes"] == ["freeze"] and d["detail"]["decided_by"] == "final_dev_gap"
          and cites_ok(ev, d), d["detail"])
    ev, d = run({"full": x, "freeze": x, "lora": x}, truths, gpu={"full": 2.2, "freeze": 1.9, "lora": 2.3})
    check("tie on all three: lower GPU-hours wins",
          d["detail"]["recipes"] == ["freeze"] and d["detail"]["decided_by"] == "gpu_hours", d["detail"])
    ev, d = run({"full": [A] * 5 + [R_, R_]}, [H] * 7)
    check("exactly 5/7: ready", d["name"] == "decision_slot_ready", d["summary"])
    ev, d = run({"full": [A] * 4 + [R_] * 3}, [H] * 7)
    check("4/7: not ready, L1 + X1 (sample)", d["name"] == "no_recipe_tracks_truth" and d["levers"] == ["L1", "X1"])
    ev, d = run({"full": [A] * 10 + [R_] * 4}, [H] * 14)
    check("10/14 (= 5/7 exactly): ready", d["name"] == "decision_slot_ready", d["summary"])
    ev, d = run({"full": [A] * 9 + [R_] * 5}, [H] * 14)
    check("9/14: not ready", d["name"] == "no_recipe_tracks_truth")
    ev, d = run({"full": [A] * 4 + [R_] * 3}, [H] * 7, replay_mode="full")
    check("not ready in full replay (D1 silent): X1 only", d["name"] == "no_recipe_tracks_truth"
          and d["levers"] == ["X1"], d["levers"])
    ev, d = run({"full": [R_] * 7}, [H] * 7, replay_mode="full", p_recipe=0.0, null_mean=0.70)
    check("full replay and D1 fires on the pilot: D4 silent", not d["fired"] and one(ev, "D1")["fired"], d["summary"])
    ready_forgets = [A] * 5 + [R_, R_]           # 5/7, two truth-helps steps REJECTed
    ev, d = run({"full": ready_forgets}, [H] * 7, p_recipe=0.0, null_mean=0.70)
    check("sample replay, ready (5/7) and D1 fires on the pilot: D1 takes precedence -- blocked, L1 + X1, no L2",
          one(ev, "D1")["fired"] and d["fired"] and d["name"] == "decision_slot_ready"
          and d["detail"]["blocked_by"]["id"] == "D1" and d["levers"] == ["L1", "X1"]
          and "L2" not in d["levers"] and d["detail"]["then"][0].startswith("L2 is withheld")
          and cites_ok(ev, d), (d["levers"], d["summary"]))
    ev, d = run({"full": ready_forgets}, [H] * 7)
    check("the same pilot without D1 (P_recipe 0.5): ready with L2", d["levers"] == ["L2"]
          and "blocked_by" not in d["detail"], d["levers"])
    # D1 blocks D4 only when D4's own choice is touched (R4 review 2026-09-27, after pilot_v3): D4's
    # threshold not met, or the chain D4 selects did not ACCEPT a clean truth-helps step.
    forgets = dict(p_recipe=0.0, null_mean=0.70)
    t7 = [H, H, H, U, U, N, N]
    tracks = [A, A, A, R_, R_, A, A]             # 5/7: every truth-helps step ACCEPTed
    misses = [R_, R_, A, R_, R_, A, A]           # 3/7: S1, S2 (truth helps) REJECTed
    for mode, d1_levers in (("full", ["X1"]), ("sample", ["L1", "X1"])):
        ev, d = run({"full": tracks, "freeze": misses}, t7, replay_mode=mode, **forgets)
        d1 = one(ev, "D1")
        check("%s replay: D1 fires (freeze missed S1, S2) with its own levers %s, but the chain D4 selects, full "
              "(5/7), ACCEPTed every truth-helps step: blocks_d4 false" % (mode, " + ".join(d1_levers)),
              d1["fired"] and d1["levers"] == d1_levers and d1["detail"]["blocks_d4"] is False
              and d1["detail"]["chains"] == {"freeze": {"helps": ["S1", "S2", "S3"], "misses": ["S1", "S2"]},
                                             "full": {"helps": ["S1", "S2", "S3"], "misses": []}}
              and d1["detail"]["d4_selection"]["selected"] == "full" and cites_ok(ev, d1), d1["summary"])
        check("  so D4 applies its rule unchanged: ready, L2, recipes ['full'], replay %s, D1 recorded" % mode,
              d["fired"] and d["name"] == "decision_slot_ready" and d["levers"] == ["L2"]
              and d["detail"]["recipes"] == ["full"] and d["detail"]["replay_mode"] == mode
              and "blocked_by" not in d["detail"] and d["detail"]["d1"]["blocks_d4"] is False
              and cites_ok(ev, d), (d["levers"], d["summary"]))
    ev, d = run({"full": tracks, "freeze": [A, A, R_, R_, R_, Ho, A]}, t7, replay_mode="full", **forgets)
    check("a tie on rate (5/7 each) goes to the chain with fewer truth-helps misses (full): D1 does not block",
          d["fired"] and d["levers"] == ["L2"] and d["detail"]["recipes"] == ["full"]
          and d["detail"]["decided_by"] == "helps_not_accepted" and one(ev, "D1")["detail"]["blocks_d4"] is False,
          d["summary"])
    selected_misses = [A, A, R_, R_, R_, Ho, Ho]   # 6/7, but S3 (truth helps) REJECTed
    for mode in ("full", "sample"):
        ev, d = run({"full": selected_misses, "freeze": tracks}, t7, replay_mode=mode, **forgets)
        d1 = one(ev, "D1")
        blocked = (not d["fired"]) if mode == "full" else (d["detail"].get("blocked_by") or {}).get("id") == "D1"
        check("%s replay: the chain D4 selects by rate, full (6/7), missed S3, so D1 blocks D4 (%s), though freeze "
              "(5/7) missed nothing" % (mode, "silent" if mode == "full" else "blocked_by D1, L1 + X1"),
              d1["fired"] and d1["detail"]["blocks_d4"] is True and "S3" in d1["detail"]["blocks_d4_why"]
              and blocked and "L2" not in d["levers"], (d["summary"], d1["detail"]["blocks_d4_why"]))
    ev, d = run({"full": [A] * 4 + [R_] * 3}, [H] * 7, replay_mode="full", **forgets)
    check("D4's threshold not met (4/7) while D1 fires: D1 blocks D4 (silent in full replay)",
          not d["fired"] and one(ev, "D1")["detail"]["blocks_d4"] is True
          and "threshold is not met" in one(ev, "D1")["detail"]["blocks_d4_why"], d["summary"])
    loop = pilot_ev({"full": [R_] * 7}, [H] * 7, exp="realloop_v1", **forgets)
    loop["realloop_v1/exp.json"]["builder"] = "inc.realloop build"
    d1 = one(ev_of(loop, "realloop_v1"), "D1")
    check("D1 on a real loop: D4 does not evaluate it, blocks_d4 None (no choice of D4's to block)",
          d1["fired"] and d1["detail"]["blocks_d4"] is None and d1["levers"] == ["L2", "X1"], d1["detail"].get("blocks_d4"))
    # Only CLEAN truth-helps steps are D1's misses, and so the only ones that block (the report-side count too).
    t_nc = [H, H, H, H, U, N, N]                 # S4 helps, but S4 is not clean
    full_nc = [A, A, A, R_, R_, Ho, Ho]          # 6/7: REJECTs only S4
    for with_defn in (True, False):
        objs = pilot_ev({"full": full_nc, "freeze": misses}, t_nc, replay_mode="full", **forgets)
        objs["pilot_v1/report.json"]["steps"][3]["clean"] = False
        objs["pilot_v1/exp.json"]["steps"][3]["clean"] = False
        if not with_defn:
            del objs["pilot_v1/exp.json"]  # the report's own clean flags
        ev = ev_of(objs, "pilot_v1")
        d1, d = one(ev, "D1"), one(ev, "D4")
        sel = d1["detail"]["d4_selection"]
        check("a truth-helps step that is not clean (S4, %s) is not a miss of the chain D4 selects: full (6/7) "
              "REJECTed it, D1 fires on freeze's misses but does not block D4" % ("exp.json" if with_defn else
                                                                                 "report.json's flag"),
              d1["fired"] and d1["detail"]["blocks_d4"] is False and sel["selected"] == "full"
              and sel["misses"] == [] and sel["report_misses"] == [] and sel["report_helps_not_accepted"] == 1
              and d["fired"] and d["levers"] == ["L2"] and d["detail"]["recipes"] == ["full"] and cites_ok(ev, d1),
              (d1["detail"].get("blocks_d4_why"), sel))
    # D4 ranks from report.json; D1 reads the ledger. A clean miss only the report shows still blocks, named.
    v3 = {"pilot_v3/%s" % n: (FIX / "pilot_v3" / n).read_text() for n in ("exp.json", "ledger.jsonl", "report.json")}
    rep = json.loads(v3["pilot_v3/report.json"])
    for st in rep["steps"]:
        if st["step"] in ("I1", "I3"):
            st["chains"]["full"]["verdict"] = R_       # I3 (helps) missed, I1 (hurts) now agrees: still 5/7
    ev = ev_of(dict(v3, **{"pilot_v3/report.json": rep}), "pilot_v3")
    d1 = one(ev, "D1")
    sel = d1["detail"]["d4_selection"]
    check("pilot_v3 with the report (not the ledger) showing full REJECTing I3: D1 blocks D4 and names I3 "
          "(report.json)", d1["fired"] and d1["detail"]["blocks_d4"] is True and sel["misses"] == []
          and sel["report_misses"] == ["I3"] and "I3 (report.json)" in d1["detail"]["blocks_d4_why"]
          and not one(ev, "D4")["fired"] and cites_ok(ev, d1), d1["detail"].get("blocks_d4_why"))
    # D1's own verdict never depends on D4's inputs: a malformed report of another pilot.
    base = {"pilot_v1/%s" % n: (FIX / "pilot_v1" / n).read_text() for n in ("exp.json", "ledger.jsonl",
                                                                            "report.json")}
    clean_d1 = one(ev_of(base, "pilot_v1"), "D1")
    bad = dict(base, **{"pilot_x/exp.json": {"exp": "pilot_x", "type": "chain", "builder": DG.PILOT_BUILDER},
                        "pilot_x/report.json": json.dumps("str")})
    d1 = one(ev_of(bad, "pilot_v1"), "D1")
    check("a malformed report.json of another pilot: D1 on pilot_v1 still fires with the same levers; its D4 view "
          "is unreadable and blocks D4",
          clean_d1["fired"] and d1["fired"] and d1["levers"] == clean_d1["levers"]
          and d1["detail"]["counts"] == clean_d1["detail"]["counts"] and d1["detail"]["blocks_d4"] is True
          and "cannot be read" in d1["detail"]["blocks_d4_why"], (d1["summary"], d1["detail"].get("blocks_d4_why")))
    ev, d = run({"full": [A] * 7}, [H] * 7, compared=6)
    check("a pilot that did not compare every step: silent", not d["fired"], d["summary"])
    objs = pilot_ev({"full": [A] * 4 + [R_] * 3}, [H] * 7, exp="pilot_v1")
    newer = pilot_ev({"lora": [A] * 6 + [R_]}, [H] * 7, exp="pilot_v2")
    newer["pilot_v2/exp.json"]["initialised_utc"] = "2026-09-28T04:00:00Z"
    loop = pilot_ev({"full": [R_] * 7}, [H] * 7, exp="realloop_v1")
    loop["realloop_v1/exp.json"].update(builder="inc.realloop build", initialised_utc="2026-09-29T00:00:00Z")
    ev = ev_of(dict(objs, **newer, **loop), "realloop_v1")
    d = one(ev, "D4")
    check("the latest complete pilot is chosen and a real loop is not a pilot",
          d["exp"] == "pilot_v2" and d["detail"]["recipes"] == ["lora"], (d["exp"], d["summary"]))


def state(**kw):
    st = {"exp": "x_v1", "generation": 5, "done": False, "blocked": {}, "unblocks": [], "runs": {}}
    st.update(kw)
    return st


def gate15(chain, k, step, verdict, p_data=1.0, failed=(), clean=True, p_recipe=0.5, config=None,
           guards=("regression", "species", "flips")):
    """A gate entry with the guard dicts gate.decide records (and, when given, its config)."""
    e = gate(chain, k, step, verdict, clean=clean, p_recipe=p_recipe, p_data=p_data)
    e["decision"]["guards"] = {g: {"passed": g not in failed} for g in guards}
    if config is not None:
        e["decision"]["config"] = config
    return e


def d15_world(rows, truths, exp="x_v1", done=True, mode=None, builder="inc.pilot build", replay_mode="full",
              state_pin=None, clean=None):
    """A finished chain experiment: rows (chain, k, verdict, p_data, failed guards[, config]) over steps
    S1..Sn with the given truth verdicts; mode sets exp.json's gate block."""
    n = len(truths)
    steps = [("S%d" % (i + 1), True if clean is None else clean[i], None) for i in range(n)]
    extra = {"gate": {"flips_mode": mode}} if mode else {}
    d = defn(exp, steps, recipes=tuple(sorted({r[0] for r in rows})), builder=builder, replay_mode=replay_mode,
             **extra)
    ents = [truth(i + 1, t) for i, t in enumerate(truths)]
    for r in rows:
        chain, k, verdict, p, failed = r[:5]
        cfg = r[5] if len(r) > 5 else None
        ents.append(gate15(chain, k, "S%d" % k, verdict, p_data=p, failed=failed,
                           clean=d["steps"][k - 1]["clean"], config=cfg))
    st = {"done": done, "generation": 9}
    if state_pin:
        st["gate_pin"] = {"block": {"flips_mode": state_pin}, "config": {"flips_mode": state_pin, "p_accept": 0.75}}
    return ev_of({"%s/exp.json" % exp: d, "%s/ledger.jsonl" % exp: ledger(ents), "%s/state.json" % exp: st}, exp)


def v2_report(outcome, final=True, non_clean=(), chains=None, **kw):
    """A net-mode pilot report (d4_report) with report.py's gate record and v2_check; by default
    full agrees with the truth arm on 7/7 and freeze on 5/7."""
    A, R_, H, U = "ACCEPT", "REJECT", "helps", "hurts"
    chains = chains or {"full": [A, A, R_, A, R_, A, A], "freeze": [A, R_, R_, A, R_, R_, A]}
    objs = pilot_ev(chains, [H, H, U, H, U, H, H], exp=kw.pop("exp", "pilot_v3"), replay_mode="full", **kw)
    exp = [k for k in objs if k.endswith("/report.json")][0].split("/")[0]
    rep = objs["%s/report.json" % exp]
    rep.update(done=True, gate={"flips_mode": "net", "pinned": True, "block": {"flips_mode": "net"}})
    rep["v2_check"] = {"final": final, "truth_arm": True, "outcome": outcome, "compared": 14,
                       "agree_v2": {"refuted": 9, "supported": 13, "inconclusive": 11}[outcome],
                       "agree_v1_counterfactual": {"refuted": 10, "supported": 9, "inconclusive": 11}[outcome],
                       "discordant": [], "non_clean_accepted": list(non_clean)}
    objs["%s/exp.json" % exp]["gate"] = {"flips_mode": "net"}
    return objs, exp


def test_d15_d16():
    print("D15 guard_blocks_truth_helps, D16 gate_v2_check, and their precedence")
    # The tree as it stood when pilot_v2 was diagnosed (pilot_v3, the v2 pilot L9 built from it, is
    # pinned too: a whole-tree load would look at it for the campaign-wide D4).
    ev = E.load_dir(FIX, "pilot_v2", exps=["pilot_v1", "pilot_v2", "b0_v1", "base_b_v1"])
    by = DG.by_id(DG.detect(ev))
    d = by["D15"]
    pairs = {(p["chain"], p["step"], p["line"]) for p in d["detail"]["pairs"]}
    check("pilot_v2: D15 fires, L9, warn", d["fired"] and d["levers"] == ["L9"] and d["severity"] == "warn",
          d["summary"])
    check("pilot_v2: the five flips-only REJECTs of truth-helps steps, pooled over chains (freeze I2, I3; "
          "full I2, I3; lora I3), each its ledger line",
          pairs == {("freeze", "I2", 15), ("freeze", "I3", 22), ("full", "I2", 13), ("full", "I3", 19),
                    ("lora", "I3", 23)}, sorted(pairs))
    lines = {c["line"] for c in d["cites"] if c["artifact"] == "pilot_v2/ledger.jsonl"}
    check("pilot_v2: each of the five lines is cited with its verdict, P_data and every guard",
          all({("/decision/verdict", "REJECT"), ("/decision/p_data", 1.0), ("/decision/guards/flips/passed", False),
               ("/decision/guards/regression/passed", True), ("/decision/guards/species/passed", True)}
              <= {(c["pointer"], c["value"]) for c in d["cites"] if c.get("line") == ln}
              for ln in (13, 15, 19, 22, 23))
          and {13, 15, 19, 22, 23} <= lines, sorted(lines))
    check("pilot_v2: per chain, lora's I3 counts like the others (no per-chain floor)",
          d["detail"]["chains"]["lora"]["flips_only"] == ["I3"]
          and d["detail"]["chains"]["freeze"]["flips_only"] == ["I2", "I3"]
          and d["detail"]["chains"]["full"]["flips_only"] == ["I2", "I3"], d["detail"]["chains"])
    check("pilot_v2: pinned to v1 (a v1 decision's config, no gate block), cited",
          d["detail"]["flips_mode"] == "negative"
          and any(c["pointer"] == "/decision/config" and "flips_mode" not in c["value"] for c in d["cites"]),
          d["detail"]["flips_mode_source"])
    unx = sorted((u["chain"], u["step"], u["line"], tuple(u["failed_guards"])) for u in d["detail"]["unexplained"])
    check("pilot_v2: three truth-helps misses failed another guard too (freeze I5 regression + species, lora I2 "
          "and I5 species), so D15 does not explain every miss and does not block D1",
          unx == [("freeze", "I5", 31, ("flips", "regression", "species")), ("lora", "I2", 18, ("flips", "species")),
                  ("lora", "I5", 32, ("flips", "species"))] and d["detail"]["blocks_d1"] is False, unx)
    check("pilot_v2: every D15 cite resolves", cites_ok(ev, d))
    d1 = by["D1"]
    check("pilot_v2: D1 fires unblocked with X1 (full replay), naming the misses D15 does not explain, and "
          "citing their guards",
          d1["fired"] and d1["levers"] == ["X1"] and "blocked_by" not in d1["detail"]
          and d1["detail"]["precedence"]["D15"] == "fired, does not block D1"
          and [(u["chain"], u["step"]) for u in d1["detail"]["precedence"]["unexplained"]]
          == [("freeze", "I5"), ("lora", "I2"), ("lora", "I5")]
          and {(31, "/decision/guards/regression/passed"), (18, "/decision/guards/species/passed")}
          <= {(c.get("line"), c["pointer"]) for c in d1["cites"]}
          and len(d1["cites"]) > 40 and cites_ok(ev, d1), d1["summary"])
    check("pilot_v2: D4 silent (full replay, D1 fired: the recipe forgets)",
          not by["D4"]["fired"] and "X1" in by["D4"]["summary"] and "D15" not in by["D4"]["summary"],
          by["D4"]["summary"])
    check("pilot_v2: D16 silent (not v2)", not by["D16"]["fired"], by["D16"]["summary"])
    ev1 = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    by1 = DG.by_id(DG.detect(ev1))
    check("pilot_v1: D15 silent (no truth-helps step failed on the flips guard alone), D1 not blocked",
          not by1["D15"]["fired"] and by1["D1"]["levers"] == ["L1", "X1"]
          and "blocked_by" not in by1["D1"]["detail"], by1["D15"]["summary"])
    for e in ("b0_v1", "base_b_v1"):
        eb = E.load_dir(FIX, e, exps=[e])
        check("%s (a baseline): D15 and D16 silent" % e,
              not one(eb, "D15")["fired"] and not one(eb, "D16")["fired"])

    H, U, N = "helps", "hurts", "neutral"
    A, R_ = "ACCEPT", "REJECT"
    F = ("flips",)
    base = [("full", 1, R_, 1.0, F), ("full", 2, R_, 1.0, F), ("full", 3, A, 1.0, ())]
    d = one(d15_world(base, [H, H, H]), "D15")
    check("synthetic: two flips-only truth-helps REJECTs in one chain fire (min_pairs 2), L9, blocks D1",
          d["fired"] and d["levers"] == ["L9"] and d["detail"]["blocks_d1"], d["summary"])
    check("synthetic: one is not enough", not one(d15_world(base[:1] + base[2:], [H, H, H]), "D15")["fired"])
    check("synthetic: a threshold override is honoured",
          not one(d15_world(base, [H, H, H]), "D15", thresholds={"D15": {"min_pairs": 3}})["fired"])
    d = one(d15_world([("full", 1, R_, 1.0, F), ("freeze", 2, R_, 1.0, F)], [H, H]), "D15")
    check("synthetic: pairs in two chains add up (pooled over (chain, step) pairs, as the v2 check pools them)",
          d["fired"] and len(d["detail"]["pairs"]) == 2 and d["detail"]["blocks_d1"], d["summary"])
    check("synthetic: P_data below p_accept does not count",
          not one(d15_world([("full", 1, R_, 0.7, F), ("full", 2, R_, 1.0, F)], [H, H]), "D15")["fired"])
    check("synthetic: exactly p_accept counts",
          one(d15_world([("full", 1, R_, 0.75, F), ("full", 2, R_, 1.0, F)], [H, H]), "D15")["fired"])
    cfg = {"p_accept": 0.9, "p_reject": 0.25}
    d = one(d15_world([("full", 1, R_, 0.8, F, cfg), ("full", 2, R_, 1.0, F, cfg)], [H, H]), "D15")
    check("synthetic: the decision's own config.p_accept (0.9) is used, not the default", not d["fired"], d["summary"])
    check("synthetic: another failed guard does not count",
          not one(d15_world([("full", 1, R_, 1.0, ("flips", "species")), ("full", 2, R_, 1.0, F)], [H, H]),
                  "D15")["fired"])
    check("synthetic: a truth-neutral or -hurts step does not count",
          not one(d15_world([("full", 1, R_, 1.0, F), ("full", 2, R_, 1.0, F)], [H, N]), "D15")["fired"])
    check("synthetic: a step not marked clean does not count (a non-clean acceptance refutes v2)",
          not one(d15_world([("full", 1, R_, 1.0, F), ("full", 2, R_, 1.0, F)], [H, H], clean=[True, False]),
                  "D15")["fired"])
    check("synthetic: an experiment that is not finished is not evaluated",
          not one(d15_world(base, [H, H, H], done=False), "D15")["fired"])
    d = one(d15_world(base, [H, H, H], mode="net"), "D15")
    check("synthetic: pinned to v2 (exp.json gate block): no L9, card X6", d["fired"] and d["levers"] == ["X6"]
          and d["detail"]["flips_mode"] == "net", d["levers"])
    ev = d15_world(base, [H, H, H], mode="negative", state_pin="net")
    d = one(ev, "D15")
    check("synthetic: state.json's gate_pin wins over exp.json, and is cited",
          d["detail"]["flips_mode"] == "net" and d["levers"] == ["X6"]
          and any(c["artifact"] == "x_v1/state.json" and c["pointer"] == "/gate_pin/config/flips_mode"
                  for c in d["cites"]) and cites_ok(ev, d), d["detail"]["flips_mode_source"])
    d = one(d15_world(base, [H, H, H], builder="inc.realloop build"), "D15")
    check("synthetic: a real loop on v1: fires, card X8 for a person (L9 rebuilds a pilot)",
          d["fired"] and d["levers"] == ["X8"] and d["detail"]["blocks_d1"], d["levers"])

    rows = base + [("full", 4, R_, 0.3, ("regression", "flips"))]
    ev = d15_world(rows, [H, H, H, H])
    d = one(ev, "D15")
    check("synthetic: D15 fires but a miss that is not flips-only means it does not block D1",
          d["fired"] and not d["detail"]["blocks_d1"], d["detail"]["chains"])

    ready = [A] * 5 + [R_, R_]
    ents = [truth(k, H) for k in range(1, 8)]
    for k, v in enumerate(ready, 1):
        ents.append(gate15("full", k, "S%d" % k, v, p_data=1.0, failed=F if v == R_ else (), p_recipe=0.0))
        ents[-1]["decision"].update(null_mean=0.70, inc=0.72)
    objs = pilot_ev({"full": ready}, [H] * 7, p_recipe=0.0, null_mean=0.70)
    objs["pilot_v1/ledger.jsonl"] = ledger(ents)
    objs["pilot_v1/report.json"]["done"] = True
    ev = ev_of(objs, "pilot_v1")
    by = DG.by_id(DG.detect(ev))
    check("sample replay, ready 5/7, D1 fires and D15 blocks it: D4 blocked_by D15, no lever (not L1 + X1, not L2)",
          by["D1"]["fired"] and by["D1"]["levers"] == [] and by["D15"]["levers"] == ["L9"]
          and by["D1"]["detail"]["blocked_by"]["pairs"] == ["full S6 (line 13)", "full S7 (line 14)"]
          and by["D4"]["name"] == "decision_slot_ready" and by["D4"]["detail"]["blocked_by"]["id"] == "D15"
          and by["D4"]["levers"] == [] and cites_ok(ev, by["D4"]), (by["D4"]["levers"], by["D4"]["summary"],
                                                                     by["D1"]["detail"].get("blocked_by")))
    from weed_optimizer_framework.tools.inc_autopilot import levers as LV
    loop = copy.deepcopy(objs)
    loop["pilot_v1/exp.json"]["builder"] = "inc.realloop build"
    ev = ev_of(loop, "pilot_v1")
    diags = DG.detect(ev)
    by = DG.by_id(diags)
    res = LV.propose(diags, ev)
    check("the same ledger as a v1 real loop: D15 raises card X8, D1 is blocked (L2 + X1 withheld); the card "
          "reaches a person, nothing is built",
          by["D15"]["levers"] == ["X8"] and by["D1"]["fired"] and by["D1"]["levers"] == []
          and by["D1"]["detail"]["withheld"] == ["L2", "X1"] and "X8" in [c["lever"] for c in res["cards"]]
          and not res["proposals"], (by["D15"]["levers"], by["D1"]["levers"], [c["lever"] for c in res["cards"]],
                                     [p["lever"] for p in res["proposals"]]))
    odd = copy.deepcopy(objs)
    odd["pilot_v1/exp.json"]["gate"] = {"flips_mode": "odd"}
    ev = ev_of(odd, "pilot_v1")
    by = DG.by_id(DG.detect(ev))
    check("pinned to a mode that is neither v1 nor v2: D15 fires with no lever or card, so it does not block D1 "
          "(L1 + X1 stand; D4 blocked_by D1)",
          by["D15"]["fired"] and by["D15"]["levers"] == [] and not by["D15"]["detail"]["blocks_d1"]
          and by["D1"]["levers"] == ["L1", "X1"] and "blocked_by" not in by["D1"]["detail"]
          and "no lever or card" in by["D1"]["detail"]["precedence"]["why"]
          and by["D4"]["detail"]["blocked_by"]["id"] == "D1", (by["D15"]["levers"], by["D1"]["levers"]))

    for outcome, sev, levers in (("refuted", "crit", ["X7"]), ("supported", "info", []),
                                 ("inconclusive", "warn", [])):
        objs, exp = v2_report(outcome)
        ev = ev_of(objs, exp)
        by = DG.by_id(DG.detect(ev))
        d16 = by["D16"]
        check("D16 %s: fires %s with levers %s, cites resolve" % (outcome, sev, levers),
              d16["fired"] and d16["severity"] == sev and d16["levers"] == levers and cites_ok(ev, d16)
              and any(c["pointer"] == "/v2_check/outcome" and c["value"] == outcome for c in d16["cites"]),
              d16["summary"])
        d4 = by["D4"]
        if outcome == "refuted":
            check("D16 refuted: D4 (full 6/7) is blocked by D16, no L2", d4["fired"] and d4["levers"] == []
                  and d4["detail"]["blocked_by"]["id"] == "D16" and cites_ok(ev, d4), d4["summary"])
        else:
            check("D16 %s: D4 ready on its own threshold, L2, gate net and the check recorded" % outcome,
                  d4["name"] == "decision_slot_ready" and d4["levers"] == ["L2"]
                  and d4["detail"]["gate"] == {"flips_mode": "net", "v2_check": outcome}
                  and bool(d4["detail"].get("v2_unproven")) == (outcome == "inconclusive"), d4["detail"].get("gate"))
    objs, exp = v2_report("inconclusive", chains={"full": [A, R_, R_, R_, R_, A, R_]})     # 4/7
    d4 = one(ev_of(objs, exp), "D4")
    check("D16 inconclusive and D4 below its threshold: not ready (v2 does not lift it)",
          d4["name"] == "no_recipe_tracks_truth" and "L2" not in d4["levers"], d4["summary"])
    objs, exp = v2_report("refuted", final=False)
    check("D16: a provisional v2_check is not read", not one(ev_of(objs, exp), "D16")["fired"])
    objs, exp = v2_report("supported")
    del objs["%s/report.json" % exp]["v2_check"]
    check("D16: a v2 experiment whose report has no v2_check: not evaluated",
          not one(ev_of(objs, exp), "D16")["fired"] and "no v2_check" in one(ev_of(objs, exp), "D16")["summary"])
    objs, exp = v2_report("supported")
    objs["%s/report.json" % exp]["done"] = False
    check("D16: not finished: silent", not one(ev_of(objs, exp), "D16")["fired"])
    from weed_optimizer_framework.tools.inc_autopilot import levers as LV
    for state, how in (("missing", "no v2_check in report.json"), ("not final", "a provisional v2_check")):
        objs, exp = v2_report("supported")
        if state == "missing":
            del objs["%s/report.json" % exp]["v2_check"]
        else:
            objs["%s/report.json" % exp]["v2_check"]["final"] = False
        ev = ev_of(objs, exp)
        diags = DG.detect(ev)
        d4 = DG.by_id(diags)["D4"]
        res = LV.propose(diags, ev)
        check("D4 on a ready v2 pilot with %s: blocked_by D16 (pending %s), no L2 proposed" % (how, state),
              d4["name"] == "decision_slot_ready" and d4["levers"] == []
              and d4["detail"]["blocked_by"] == {"id": "D16", "exp": exp, "pending": state,
                                                 "summary": "the protocol v2 check on %s is %s" % (exp, state)}
              and d4["detail"]["gate"] == {"flips_mode": "net", "v2_check": state}
              and d4["detail"]["v2_pending"] == state and cites_ok(ev, d4)
              and not [p for p in res["proposals"] if p["lever"] == "L2"]
              and any(x["lever"] == "L2" and "waits for" in x["reason"] for x in res["deferred"]),
              (d4["summary"], d4["detail"].get("blocked_by"), res["deferred"]))
    objs, exp = v2_report("refuted", non_clean=[{"chain": "full", "step": "Breal", "v1_counterfactual": "REJECT"}])
    d16 = one(ev_of(objs, exp), "D16")
    check("D16 refuted by a non-clean acceptance: named and cited",
          "Breal" in d16["summary"] and any(c["pointer"] == "/v2_check/non_clean_accepted" for c in d16["cites"]))

    import shutil
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_diag_"))
    try:
        _prospective_v2(tmp)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def _prospective_v2(tmp):
    objs, exp = v2_report("supported")
    rec = DG.prospective_d4(objs["%s/report.json" % exp], out_path=tmp / "p.json",
                            defn=objs["%s/exp.json" % exp], now="2026-09-28T00:00:00Z")
    check("prospective D4 on a supported v2 pilot: ready, full replay, recipe full, gate net, v2 supported",
          rec["ready"] and rec["replay_mode"] == "full" and rec["recipes"] == ["full"]
          and rec["gate_flips_mode"] == "net" and rec["v2_check"] == "supported", rec)
    cmp = DG.compare_prospective(rec, "full", "full", gate_flips_mode="negative")
    check("compare_prospective: a loop built on another gate does not match", not cmp["match"]
          and cmp["gate_match"] is False, cmp)
    objs, exp = v2_report("refuted")
    rec = DG.prospective_d4(objs["%s/report.json" % exp], out_path=tmp / "r.json",
                            defn=objs["%s/exp.json" % exp], now="2026-09-28T00:00:00Z")
    check("prospective D4 on a refuted v2 pilot: not ready, blocked by D16",
          not rec["ready"] and rec["blocked_by"] == "D16" and rec["recipes"] is None, rec)
    for state in ("missing", "not final"):
        objs, exp = v2_report("supported")
        if state == "missing":
            del objs["%s/report.json" % exp]["v2_check"]
        else:
            objs["%s/report.json" % exp]["v2_check"]["final"] = False
        out = tmp / ("pending_%s.json" % state.replace(" ", "_"))
        rec = DG.prospective_d4(objs["%s/report.json" % exp], out_path=out, defn=objs["%s/exp.json" % exp],
                                now="2026-09-28T00:00:00Z")
        check("prospective D4 on a v2 pilot whose check is %s: nothing written (pending), not ready" % state,
              rec["pending"] == state and rec["written"] is False and not rec["ready"]
              and rec["blocked_by"] == "D16" and not out.exists(), rec.get("pending"))
    objs, exp = v2_report("supported")
    objs["%s/report.json" % exp]["v2_check"]["final"] = False
    out = tmp / "later.json"
    DG.prospective_d4(objs["%s/report.json" % exp], out_path=out, defn=objs["%s/exp.json" % exp])
    objs["%s/report.json" % exp]["v2_check"]["final"] = True
    rec = DG.prospective_d4(objs["%s/report.json" % exp], out_path=out, defn=objs["%s/exp.json" % exp],
                            now="2026-09-28T00:00:00Z")
    check("... and once the check is final the decision is written (ready, gate net, supported): the provisional "
          "report froze nothing", out.is_file() and rec["ready"] and rec["v2_check"] == "supported"
          and "pending" not in rec, rec.get("outcome"))
    rec = DG.prospective_d4(FIX / "pilot_v2" / "report.json", out_path=tmp / "v2.json",
                            defn=json.loads((FIX / "pilot_v2" / "exp.json").read_text()), now="2026-09-28T00:00:00Z")
    check("prospective D4 on pilot_v2 (report only): silent, D1 fired in full replay (the recipe forgets, X1)",
          rec["outcome"] == "silent" and not rec["ready"] and "X1" in rec["diagnosis"]["summary"]
          and "pending" not in rec, rec["diagnosis"]["summary"])
    _rules_version(tmp)


def _rules_version(tmp):
    import hashlib
    from weed_optimizer_framework.tools.inc_autopilot import levers as LV
    here = ROOT / "weed_optimizer_framework/tools/inc_autopilot"
    raw = b"".join((here / n).read_bytes() for n in ("diagnose.py", "thresholds.json", "levers.json", "levers.py"))
    ver = DG.rules_version()
    check("the rules version is the first 12 hex of sha256(diagnose.py + thresholds.json + levers.json + levers.py)",
          ver == hashlib.sha256(raw).hexdigest()[:12] and re.fullmatch(r"[0-9a-f]{12}", ver), ver)
    real_lv = LV._SELF_BYTES
    LV._SELF_BYTES = real_lv + b"\n# the R4 sizing formula changed\n"
    try:
        other_lv = DG.rules_version()
    finally:
        LV._SELF_BYTES = real_lv
    check("a change to levers.py (where D2's sizing rule and evidence capacity live) is a new rules version",
          other_lv != ver and DG.rules_version() == ver, other_lv)
    real = DG.THRESHOLDS_FILE
    moved = tmp / "thresholds.json"
    moved.write_bytes(real.read_bytes() + b"\n")
    DG.THRESHOLDS_FILE = moved
    try:
        other = DG.rules_version()
    finally:
        DG.THRESHOLDS_FILE = real
    check("a change to thresholds.json is a new rules version", other != ver and DG.rules_version() == ver, other)
    check("a record's name carries its pilot and rules version",
          DG.prospective_name("pilot_v3", ver) == "prospective_d4_pilot_v3__%s.json" % ver)
    objs, exp = v2_report("supported")
    replay = tmp / "replay"
    out = replay / DG.prospective_name(exp, ver)
    rec = DG.prospective_d4(objs["%s/report.json" % exp], out_path=out, defn=objs["%s/exp.json" % exp],
                            now="2026-09-28T00:00:00Z")
    check("prospective_d4 records the rules version and each rules file's sha256",
          rec["rules_version"] == ver
          and sorted(rec["rules_files"]) == ["diagnose.py", "levers.json", "levers.py", "thresholds.json"]
          and rec["rules_files"]["levers.json"] == hashlib.sha256((here / "levers.json").read_bytes()).hexdigest()
          and rec["rules_files"]["levers.py"] == hashlib.sha256((here / "levers.py").read_bytes()).hexdigest(),
          rec.get("rules_files"))
    got, path, why = DG.current_prospective(replay, exp)
    check("current_prospective: the READY record of the current rules version", why == "" and got["ready"]
          and path == out, why)
    older = replay / ("prospective_d4_%s.json" % exp)
    older.write_text(json.dumps(dict(rec, rules_version=None, ready=False)))
    older_v = replay / DG.prospective_name(exp, "0" * 12)
    older_v.write_text(json.dumps(dict(rec, rules_version="0" * 12)))
    check("prospective_records lists every record of the pilot, versioned or not",
          [(p.name, v) for p, v, _r in DG.prospective_records(replay, exp)]
          == sorted([(older.name, None), (older_v.name, "0" * 12), (out.name, ver)]),
          [(p.name, v) for p, v, _r in DG.prospective_records(replay, exp)])
    out.unlink()
    got, path, why = DG.current_prospective(replay, exp)
    check("current_prospective never answers with a record of other rules, and names them",
          got is None and older.name in why and older_v.name in why, why)
    try:
        DG.prospective_d4(objs["%s/report.json" % exp], out_path=older_v, defn=objs["%s/exp.json" % exp])
        refused = False
    except ValueError as e:
        refused = "rules version" in str(e)
    check("prospective_d4 refuses a path that holds another rules version's record", refused)
    try:
        DG.prospective_d4(objs["%s/report.json" % exp], out_path=tmp / "x.json", version="f" * 12)
        refused = False
    except ValueError:
        refused = True
    check("prospective_d4 refuses a rules version other than the one loaded", refused)


def test_health():
    print("health: D5-D8, D10, D14")
    blocked = {"chain:full": {"error": "reading ...: OSError", "cause": {"kind": "transient"}},
               "truth": {"error": "run x failed for good", "cause": {"kind": "failed_run", "run_id": "x"}}}
    ev = ev_of({"x_v1/state.json": state(blocked={"chain:full": blocked["chain:full"]})}, "x_v1")
    d = one(ev, "D5")
    check("D5: a transient block never auto-unblocked: L7, warn",
          d["fired"] and d["levers"] == ["L7"] and d["severity"] == "warn" and cites_ok(ev, d), d["summary"])
    ev = ev_of({"x_v1/state.json": state(blocked={"chain:full": blocked["chain:full"]},
                                         unblocks=[{"unit": "chain:full", "auto": False,
                                                    "reason": "auto: transient"}])}, "x_v1")
    d = one(ev, "D5")
    check("D5: the same unit after one auto unblock: pause, crit", d["levers"] == ["OP_PAUSE"]
          and d["severity"] == "crit", d["summary"])
    ev = ev_of({"x_v1/state.json": state(blocked=blocked)}, "x_v1")
    d = one(ev, "D5")
    check("D5: a failed-run block beside a transient one: L7 and pause",
          d["levers"] == ["L7", "OP_PAUSE"] and d["severity"] == "crit", d["levers"])
    ev = ev_of({"x_v1/report.json": {"blocked": {"truth": blocked["truth"]}, "interventions": []}}, "x_v1")
    check("D5: from report.json when there is no state.json", one(ev, "D5")["fired"])
    check("D5: nothing blocked: silent", not one(ev_of({"x_v1/state.json": state()}, "x_v1"), "D5")["fired"])

    hist = [{"utc": "2026-09-27T01:00:00Z", "generation": 4}, {"utc": "2026-09-27T02:00:00Z", "generation": 5},
            {"utc": "2026-09-27T03:00:00Z", "generation": 5}]
    ctx = {"history": hist, "squeue": ["inc_other_0001"], "now_utc": "2026-09-27T04:30:00Z"}
    ev = ev_of({"x_v1/state.json": state()}, "x_v1", context=ctx)
    d = one(ev, "D6")
    check("D6: generation unchanged 2.5 h, nothing queued: advance, then pause",
          d["fired"] and d["levers"] == ["OP_ADVANCE"] and cites_ok(ev, d), d["summary"])
    ev = ev_of({"x_v1/state.json": state()}, "x_v1", context=dict(ctx, squeue=["inc_x_v1_0007"]))
    check("D6: a job of the experiment queued: silent", not one(ev, "D6")["fired"])
    ev = ev_of({"x_v1/state.json": state()}, "x_v1", context=dict(ctx, now_utc="2026-09-27T03:30:00Z"))
    check("D6: 1.5 h: silent", not one(ev, "D6")["fired"])
    ev = ev_of({"x_v1/state.json": state(done=True)}, "x_v1", context=ctx)
    check("D6: done: silent", not one(ev, "D6")["fired"])
    ev = ev_of({"x_v1/state.json": state(generation=6)}, "x_v1", context=ctx)
    check("D6: a new generation: silent", not one(ev, "D6")["fired"])

    err = ("the running package /ocean/a differs from the git-tracked copy /ocean/b in ['tools/inc/gate.py']; "
           "sync the outer copy")
    ev = ev_of({}, "x_v1", context={"advance": {"error": err}})
    d = one(ev, "D7")
    check("D7: drift: pause and X5, crit", d["fired"] and d["levers"] == ["OP_PAUSE", "X5"]
          and d["severity"] == "crit" and cites_ok(ev, d))
    check("D7: another advance error: silent",
          not one(ev_of({}, "x_v1", context={"advance": {"error": "sbatch failed"}}), "D7")["fired"])

    rows = [gate("full", k, "V%d" % k, "HOLD", p_data=0.5, cand_mean=0.721, null_mean=0.720) for k in (1, 2, 3)]
    rows.append(gate("full", 4, "V4", "ACCEPT", p_data=1.0))
    d_real = defn("x_v1", [("V%d" % k, True, None) for k in range(1, 5)], recipes=("full",),
                  builder="inc.realloop build", increment_images=393)
    ev = ev_of({"x_v1/ledger.jsonl": ledger(rows), "x_v1/exp.json": d_real}, "x_v1")
    d = one(ev, "D8")
    check("D8: 3/4 underpowered in a real loop: L5 with --size doubled",
          d["fired"] and d["levers"] == ["L5"] and d["detail"]["size"] == 786 and cites_ok(ev, d), d["summary"])
    ev = ev_of({"x_v1/ledger.jsonl": ledger(rows[:1] + [rows[3]] * 1 + [gate("full", 5, "V5", "ACCEPT")])}, "x_v1")
    check("D8: 1/3 underpowered: silent", not one(ev, "D8")["fired"])
    ev = ev_of({"x_v1/ledger.jsonl": ledger(rows)}, "x_v1")
    d = one(ev, "D8")
    check("D8: in a pilot: informational (no --size)", d["fired"] and d["levers"] == [])

    ctx = {"budget": {"envelope_su": 300.0, "spent_su": 280.0, "projected_su": 30.0}}
    ev = ev_of({}, "x_v1", context=ctx)
    d = one(ev, "D10")
    check("D10: spent + projected over the envelope: pause, crit", d["fired"] and d["levers"] == ["OP_PAUSE"]
          and cites_ok(ev, d))
    ctx["budget"]["projected_su"] = 10.0
    check("D10: inside the envelope: silent", not one(ev_of({}, "x_v1", context=ctx), "D10")["fired"])
    st = state(runs={"a": {"owner": "chain:full", "status": "submitted"}, "b": {"owner": "truth", "status": "queued"},
                     "c": {"owner": "truth", "status": "complete"}})
    rep = {"gpu_hours": {"chain:full": {"hours": 2.0, "runs": 40}, "truth": {"hours": 9.0, "runs": 3}}}
    ev = ev_of({"x_v1/state.json": st, "x_v1/report.json": rep}, "x_v1",
               context={"budget": {"envelope_su": 300.0, "spent_su": 297.0}})
    d = one(ev, "D10")
    check("D10: projected from open runs x mean hours per run (0.05 + 3.0)",
          d["fired"] and abs(d["detail"]["projected_su"] - 3.05) < 1e-9, d["detail"])
    ev = ev_of({"x_v1/derived/state_runs.json": {"by_owner": {"truth": {"queued": 1, "complete": 1}}},
                "x_v1/state.json": {"generation": 1}, "x_v1/report.json": rep}, "x_v1",
               context={"budget": {"envelope_su": 300.0, "spent_su": 298.0}})
    check("D10: from remote.py's derived state_runs when state.json has no runs",
          one(ev, "D10")["fired"] and abs(one(ev, "D10")["detail"]["projected_su"] - 3.0) < 1e-9)
    check("D10: no budget: not evaluated", "not evaluated" in one(ev_of({}, "x_v1"), "D10")["summary"])

    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    check("D14: the fixture is clean", not one(ev, "D14")["fired"])
    ev.touched.append("pilot_v1/runs/final__base__s0/scores/test.json")
    d = one(ev, "D14")
    check("D14: a non-allowed file read: hard stop", d["fired"] and d["levers"] == ["OP_HALT"]
          and d["severity"] == "crit", d["summary"])
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    ev.artifacts["pilot_v1/report.json"]["final"][0]["exams"]["test"] = {"twelve": {"mean": 0.77}}
    d = one(ev, "D14")
    check("D14: a non-dev exam value in the evidence: hard stop, the value is not cited",
          d["fired"] and all(c["value"] != {"twelve": {"mean": 0.77}} for c in d["cites"]), d["summary"])


def test_other():
    print("D9, D11, D12, D13, DREF")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    d = one(ev, "D9")
    check("D9 pilot_v1: 6.67 / 30 = 0.22 > 0.15, informational", d["fired"] and abs(d["detail"]["ratio"] - 6.67 / 30)
          < 1e-12 and d["levers"] == [] and cites_ok(ev, d))
    check("D9: silent at a 0.3 limit", not one(ev, "D9", thresholds={"D9": {"warmup_share_max": 0.3}})["fired"])
    d = one(ev, "D11")
    check("D11 pilot_v1: truth arm 52% but no pilot >= 6/7: silent", not d["fired"] and "52%" in d["summary"],
          d["summary"])
    A, R_, H = "ACCEPT", "REJECT", "helps"
    p1 = pilot_ev({"full": [A] * 6 + [R_]}, [H] * 7, exp="pilot_v1")
    p2 = pilot_ev({"full": [A] * 7}, [H] * 7, exp="pilot_v2")
    p2["pilot_v2/exp.json"]["initialised_utc"] = "2026-09-28T00:00:00Z"
    ev = ev_of(dict(p1, **p2), "pilot_v2")
    d = one(ev, "D11")
    check("D11: the truth arm at exactly 40% is not more than 40%: silent", not d["fired"], d["summary"])
    p2["pilot_v2/report.json"]["gpu_hours"]["truth"]["hours"] = 4.5
    ev = ev_of(dict(p1, **p2), "pilot_v2")
    d = one(ev, "D11")
    check("D11: two pilots >= 6/7 and the truth arm 45% of 10 GPU-h: L6 allowed",
          d["fired"] and d["levers"] == ["L6"] and cites_ok(ev, d), d["summary"])
    check("D11: one such pilot: silent", not one(ev_of(p2, "pilot_v2"), "D11")["fired"])
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    d = one(ev, "D12")
    check("D12 pilot_v1: identical verdicts in all chains; cheapest freeze",
          d["fired"] and d["detail"]["cheapest"] == "freeze" and cites_ok(ev, d), d["summary"])
    ev = ev_of(pilot_ev({"full": [A] * 7, "freeze": [A] * 6 + [R_]}, [H] * 7), "pilot_v1")
    check("D12: one differing step: silent", not one(ev, "D12")["fired"])
    outs = [{"lever": "L1", "child_exp": "pilot_v2", "predicted": {"direction": "up"}, "verdict": "worse"},
            {"lever": "L5", "child_exp": "r2", "predicted": {"direction": "up"}, "verdict": "within_noise"}]
    ev = ev_of({}, "x_v1", context={"outcomes": outs})
    d = one(ev, "D13")
    check("D13: up predicted, worse measured: escalate", d["fired"] and d["levers"] == ["OP_ESCALATE"]
          and len(d["detail"]["contradicted"]) == 1 and cites_ok(ev, d))
    check("D13: within_noise is not a contradiction",
          not one(ev_of({}, "x_v1", context={"outcomes": outs[1:]}), "D13")["fired"])
    ctx = {"refusals": [{"builder": "inc.pilot", "message": "[inc.pilot] ERROR: /ocean/x/splits/v1/LOCK.json does "
                                                            "not exist: a production experiment is built only on "
                                                            "locked splits"},
                        {"builder": "inc.realloop", "message": "something nobody mapped"}]}
    ev = ev_of({}, "x_v1", context=ctx)
    d = one(ev, "DREF")
    check("DREF: a LOCK refusal and an unmapped one: no menu lever, crit, escalated to a person (OP_ESCALATE)",
          d["fired"] and d["levers"] == ["OP_ESCALATE"] and d["severity"] == "crit"
          and len(d["detail"]["refusals"]) == 2 and all(x["retry"] for x in d["detail"]["refusals"])
          and cites_ok(ev, d), (d["levers"], d["summary"]))
    # realloop's evidenced-pool refusal and select's draw shortfall (the sized evidence loop's two build-time
    # refusals): mapped to a person, escalated, and marked as refusals the identical build meets again.
    from weed_optimizer_framework.tools.inc import select as S0
    try:
        S0.draw_parts([[0]], [5], 2, 5)
        draw_msg = None
    except S0.SelectError as e:
        draw_msg = "[inc.realloop] ERROR: %s" % e
    cap_msg = ("[inc.realloop] ERROR: increment sources 'evidence': the evidenced increment pool cannot supply 4 "
               "verified increments + OTHER_HEAVY of 287 images. It holds 1439 images (1435 needed), 250 of them in "
               "OtherPlant-heavy near-dup groups (287 needed)")
    ctx = {"refusals": [{"builder": "inc_build_realloop", "message": cap_msg},
                        {"builder": "inc_build_realloop", "message": draw_msg}]}
    ev = ev_of({}, "x_v1", context=ctx)
    d = one(ev, "DREF")
    check("DREF: realloop's evidenced-pool refusal and select's draw shortfall ('%s') are mapped to a person, "
          "crit, escalated, retry false" % draw_msg,
          draw_msg is not None and d["fired"] and d["levers"] == ["OP_ESCALATE"] and d["severity"] == "crit"
          and [x["prerequisite"] for x in d["detail"]["refusals"]] == [None, None]
          and all(x["needs"].startswith("a person") and x["retry"] is False for x in d["detail"]["refusals"])
          and cites_ok(ev, d), d["detail"]["refusals"])
    ctx = {"refusals": [{"builder": "inc.realloop", "message": "[inc.realloop] ERROR: no /ocean/x/inc/step1/"
                                                               "relevance.json: the verified and OTHER_HEAVY "
                                                               "increments are drawn only from sources that pass "
                                                               "the relevance filter"},
                        {"builder": "inc.realloop", "message": "run inc.verify admit first"}]}
    d = one(ev_of({}, "x_v1", context=ctx), "DREF")
    check("DREF: a relevance refusal (D2's, prerequisite L3) beside a verify-admit one: DREF holds only the latter, "
          "crit, escalated",
          d["fired"] and d["severity"] == "crit" and d["levers"] == ["OP_ESCALATE"]
          and len(d["detail"]["refusals"]) == 1 and d["detail"]["refusals"][0]["prerequisite"] is None,
          (d["levers"], d["summary"]))


def main():
    test_thresholds()
    test_d1()
    test_d2()
    test_d3()
    test_d4()
    test_d15_d16()
    test_health()
    test_other()
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
