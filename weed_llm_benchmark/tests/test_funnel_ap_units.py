#!/usr/bin/env python3
"""Unit checks of the autopilot side of the funnel audit (docs/FUNNEL_AUDIT.md
8.2-8.9) that the replay cases (tests/test_funnel_ap_replay.py) do not reach.

  * thresholds.json: every value marked checked_against equals the
    pre-registration (funnel/prereg_v1.json) or the funnel module it names;
  * levers.json and remote.py mirror the funnel CLI's tables (VERB_CLASS,
    SBATCH_RESOURCES, the job verbs) and realloop's RECOVERED_SEQUENCE, which
    equals the pre-registered realloop_v2 sequence;
  * model_router: the adversary role, model_family and same_family;
  * panel.build_panel on synthetic dev scores: the 3 v 3 rule, the 5 v 5
    exact permutation test (smallest p 1/252) on scores the gate's own score
    rule accepts, dev-only refusal, seed checks, and spec_v2 against the
    pre-registered arms; outcome.h10 (H10c with and without same-image
    controls), h11 (uniform over the ledger's recoverable stages, the audit's
    own H11, no invalid audit), score_da;
  * remote.py funnel verbs on a temporary INC_DIR: dev-scores (dev only),
    summary (the ship list, never the never-list, the listing, dev_only),
    ledger-summaries (derived in memory, written once), submit funnel
    (dry run, the verb's sbatch resources, refusals);
  * executor.py: render and params_from_argv of the funnel actions,
    --step1-overlay and --seeds, the DA plan segment (role adversary) and the
    cluster-side submit script run against a fake sbatch; the lab hooks
    (sync list and refusals, sync with a fake runner, the arrival check run
    on a local copy, the pull cluster -> lab with the real rsync on local
    directories, verify queue); levers.funnel_sync_needed both ways;
  * campaign.py: DEC-1..10 logged once, the DA staged blind one claim per
    pass, merged into the claims register (open -> challenged), recorded once
    with its runner 1.2 header, a reply for an unstaged claim refused, the
    COMPLETE card held while a pass is in flight, the record scored once a
    valid audit lands, and the COMPLETE note;
  * campaign.py, a funnel job that failed (the live incident of 2026-09-28:
    embed-judges refused without the KT7 photos and was never run again):
    WAIT_JOB records each job's final state from sacct (job_finished,
    funnel_jobs, the log path), the lineage record turns 'failed', the step
    passes the stop-loss counted per step and runs again under a new id, at
    most funnel_job_retries times, then stays with a person; a job that
    ended before states were recorded is asked about and found; UNKNOWN
    after SACCT_TRIES snapshots; an experiment-mode job's wait unchanged;
  * remote.py campaign-snapshot --sacct (job_states) and
    executor.campaign_snapshot's sacct (no flag when none is asked);
  * the funnel fixtures' MANIFEST.json pins; run_inc_plan.sh's PLAN_ROLE.

No network, no GPU, no cluster.

Run:  python3 tests/test_funnel_ap_units.py
"""
import base64
import contextlib
import copy
import gzip
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_ap_units_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
for k in ("INCAP_LEVERS_JSON", "INCAP_SCRIPT_DIR", "INCAP_SBATCH", "INCAP_MAX_FILE_BYTES"):
    os.environ.pop(k, None)
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))

from weed_optimizer_framework.tools import model_router as MR  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as X  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as LV  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import outcome as O  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import panel as PN  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as RM  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import validate as V  # noqa: E402

INC = pathlib.Path(C.INC_DIR)
FF = TESTS / "fixtures" / "inc_replay" / "funnel"
PREREG = ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json"
CONTRACT = ROOT.parent / "docs" / "FUNNEL_AUDIT.md"
THRESHOLDS = ROOT / "weed_optimizer_framework" / "tools" / "inc_autopilot" / "thresholds.json"
OCEAN_INC = M.CLUSTER_INC_DIR.rstrip("/") + "/"
ADV, PLANNER = "vllm:glm-4.7-flash", "ollama:qwen3.8:27b"
FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def skip(name, why):
    print("  skip %s: %s" % (name, why))
    SKIPS.append(name)


def jload(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def raises(fn, exc, contains=None):
    try:
        fn()
    except exc as e:
        return contains is None or contains in str(e)
    return False


# ------------------------------------------------------------------ mirrors
def test_thresholds_checked_against():
    print("thresholds.json: every checked_against value agrees with what it names")
    from weed_optimizer_framework.tools.funnel import claims as FC
    th, pre = jload(THRESHOLDS), jload(PREREG)
    seen = 0
    for blk, rows in sorted(th.items()):
        if blk.startswith("_") or not isinstance(rows, dict):
            continue
        for key, leaf in sorted(rows.items()):
            if not isinstance(leaf, dict) or "checked_against" not in leaf:
                continue
            seen += 1
            src, val = leaf["checked_against"], leaf.get("value")
            if src.startswith("funnel/prereg_v1.json "):
                got = pre
                for part in src.split(" ", 1)[1].split("."):
                    got = got.get(part) if isinstance(got, dict) else None
                check("%s.%s = %r equals the pre-registration's %s" % (blk, key, val, src.split(" ", 1)[1]),
                      got is not None and got == val, got)
            elif src == "funnel/claims.py POLARITIES without positive":
                check("%s.%s = funnel.claims' polarities without positive (its NEGATIVE_POLARITIES)" % (blk, key),
                      val == [p for p in FC.POLARITIES if p != "positive"] == list(FC.NEGATIVE_POLARITIES), val)
            elif src == "funnel/claims.py open_negative":
                reg = {"claims": [{"id": "C%d" % i, "status": s, "polarity": "scarcity"}
                                  for i, s in enumerate(FC.STATUSES)]}
                got = sorted(c["status"] for c in FC.open_negative(reg))
                check("%s.%s = the statuses funnel.claims.open_negative keeps" % (blk, key),
                      sorted(val) == got, (val, got))
            else:
                check("%s.%s: checked_against %r is a form this test knows" % (blk, key, src), False)
    check("six values are checked against the pre-registration or funnel.claims", seen == 6, seen)


def test_mirrors():
    print("levers.json and remote.py mirror the funnel CLI and realloop")
    from weed_optimizer_framework.tools.funnel import __main__ as FM
    from weed_optimizer_framework.tools.inc import realloop as RL
    from weed_optimizer_framework.tools.brain import policy as POL
    menu = LV.load_menu()
    proto = menu["protocol"]
    check("protocol funnel_verb_class = funnel/__main__.py VERB_CLASS",
          proto["funnel_verb_class"]["value"] == FM.VERB_CLASS, proto["funnel_verb_class"]["value"])
    check("protocol funnel_sbatch_resources = funnel/__main__.py SBATCH_RESOURCES",
          proto["funnel_sbatch_resources"]["value"] == FM.SBATCH_RESOURCES)
    check("protocol funnel_script = funnel/__main__.py JOB_SCRIPT", proto["funnel_script"]["value"] == FM.JOB_SCRIPT)
    job_verbs = [v for v in FM.VERBS if FM.VERB_CLASS.get(v) not in (None, "lab") and v not in ("map", "recover")]
    l10 = menu["levers"]["L10"]["param_bounds"]["verb"]["values"]
    prow = POL.describe("inc_funnel_audit")
    check("L10's verbs = remote.FUNNEL_AUDIT_VERBS = the CLI's job verbs but map and recover = the policy row's",
          l10 == list(RM.FUNNEL_AUDIT_VERBS) == job_verbs == prow["param_bounds"]["verb"]["values"],
          (l10, RM.FUNNEL_AUDIT_VERBS, job_verbs))
    check("remote.funnel_resources reads the CLI's table (rl-b holds an H100, census nothing extra)",
          RM.funnel_resources("rl-b") == FM.SBATCH_RESOURCES["gpu_large"] and RM.funnel_resources("census") == [])
    check("remote.funnel_resources refuses a verb the table does not know",
          raises(lambda: RM.funnel_resources("nope"), RM.Refused))
    pre = jload(PREREG)
    check("realloop.RECOVERED_SEQUENCE = the pre-registered realloop_v2 sequence = panel.RECOVERY_STEPS; levers "
          "recovered_steps = its length",
          list(RL.RECOVERED_SEQUENCE) == pre["realloop_v2"]["sequence"] == list(PN.RECOVERY_STEPS)
          and proto["recovered_steps"]["value"] == len(RL.RECOVERED_SEQUENCE))
    check("remote's realloop --increment-sources enum = realloop.INCREMENT_SOURCE_MODES",
          list(RM.REALLOOP_INCREMENT_SOURCES) == list(RL.INCREMENT_SOURCE_MODES))
    check("policy_actions.json validates with the funnel rows", POL.errors() == [], POL.errors()[:3])
    for act, risk in (("inc_funnel_audit", "R2"), ("inc_funnel_map", "R2"), ("inc_funnel_recover", "R3"),
                      ("inc_funnel_fetch", "R0"), ("inc_verify_queue", "R2"), ("inc_funnel_sync", "R0"),
                      ("inc_funnel_summary", "R0"), ("inc_funnel_dev_scores", "R0")):
        row = POL.describe(act)
        check("policy row %s is known at %s" % (act, risk), row.get("known") and row.get("risk") == risk,
              (row.get("known"), row.get("risk"), row.get("reason")))


# ------------------------------------------------------------------ model_router
def test_model_router():
    print("model_router: the adversary role, model_family, same_family")
    r = MR.resolve("adversary")
    check("adversary resolves to vllm:glm-4.7-flash on the cluster, a judgement, authoritative, async",
          r["ok"] and r["model"] == ADV and r["place"] == "cluster" and r["judgement"] and r["authoritative"]
          and r["is_async"], r)
    check("  its fallback is ollama:gemma4", MR.ROLES["adversary"]["fallbacks"] == ["ollama:gemma4"])
    check("  it is in the role table", any(x.get("role") == "adversary" for x in MR.role_table()))
    cases = {"vllm:glm-4.7-flash": "glm", "ollama:qwen3.8:27b": "qwen", "ollama:gemma4": "gemma",
             "qwen3.8:27b": "qwen", "glm-4.7-flash": "glm", "": "", "vllm:4bit-model": ""}
    got = {k: MR.model_family(k) for k in cases}
    check("model_family reads the name's leading letters after the provider", got == cases, got)
    check("same_family: glm v qwen False, gemma v gemma True, unknown None",
          MR.same_family(ADV, PLANNER) is False and MR.same_family("ollama:gemma4", "gemma4") is True
          and MR.same_family("", PLANNER) is None)
    check("the defaults differ in family, but the planner's first fallback is the adversary's default "
          "(so validate_da compares the models actually resolved)",
          MR.same_family(MR.ROLES["planner"]["model"], MR.ROLES["adversary"]["model"]) is False
          and MR.ROLES["planner"]["fallbacks"][0] == MR.ROLES["adversary"]["model"])
    check("an unknown role resolves to no model", MR.resolve("devils_advocate")["model"] is None)


# ------------------------------------------------------------------ panel and outcome
N_GT = {s: 40 for s in G.SPECIES}
_W = iter(range(1, 10 ** 6))


def score(m, exam="dev"):
    n_gt = dict(N_GT, OtherPlant=0)
    return {"exam": exam, "scorer_sha256": "5" * 64, "manifest_sha256": "6" * 64, "key_order_sha256": "a" * 64,
            "weights_sha256": "%064x" % next(_W), "n_images": 40, "map50_95": m, "map50": min(1.0, m + 0.2),
            "per_class": {s: m for s in G.SPECIES}, "n_gt": n_gt, "agnostic_map50_95": m + 0.1,
            "agnostic_map50": min(1.0, m + 0.3), "image_correct": "1" * 40, "production": True}


def base_runs(values, seeds):
    return {"base__s%d" % s: score(v) for s, v in zip(seeds, values)}


def truth_runs(step, k, values):
    return {"truth__s%d_%s__union__s%d" % (k, step, s): score(v) for s, v in enumerate(values)}


def dev_scores(u=(0.62, 0.63, 0.64, 0.65, 0.66)):
    rv2 = {}
    rv2.update(truth_runs("REC-CLASS-1", 3, [0.60, 0.61, 0.62]))
    rv2.update(truth_runs("REC-JUDGE-1", 4, [0.55, 0.56, 0.57]))
    rv2["truth__s3_REC-CLASS-1__without__s0"] = score(0.1)          # not a "with" run: ignored
    return {"base_b_v1": base_runs([0.50, 0.51, 0.52], [0, 1, 2]),
            "rv2_B_extra": base_runs([0.53, 0.54], [3, 4]),
            "rv2_U": base_runs(list(u), [0, 1, 2, 3, 4]),
            "rv2_Uctl": base_runs([0.40, 0.41, 0.42], [0, 1, 2]),
            "rv2_CLASSctl": base_runs([0.45, 0.46, 0.47], [0, 1, 2]),
            "rv2_JUDGEctl": base_runs([0.58, 0.59, 0.60], [0, 1, 2]),
            "realloop_v2": rv2}


def test_panel_and_outcome():
    print("panel.build_panel on synthetic dev scores; outcome.h10")
    spec = PN.spec_v2()
    pre = jload(PREREG)["realloop_v2"]["arms"]
    a = spec["arms"]
    check("spec_v2's arms carry the pre-registered seeds (U 5, B 3 + 2 extra, the controls 3)",
          len(a["U"]["seeds"]) == pre["U"]["seeds"] and len(a["B"]["seeds"]) == 3 + pre["B_extra_seeds"]
          and len(a["U_ctl"]["seeds"]) == pre["U-ctl"] and len(a["CLASS_ctl"]["seeds"]) == pre["CLASS-ctl"]
          and len(a["JUDGE_ctl"]["seeds"]) == pre["JUDGE-ctl"])
    ds = dev_scores()
    p1 = PN.build_panel(ds, spec, built_utc="2026-09-28T00:00:00Z")
    p2 = PN.build_panel(copy.deepcopy(ds), spec, built_utc="2026-09-28T00:00:00Z")
    comps = {c["id"]: c for c in p1["comparisons"]}
    check("the panel is funnel-panel/1, reads dev only, and is deterministic",
          p1["format"] == PN.FORMAT and p1["exams_read"] == ["dev"] and p1 == p2)
    h10a = comps["H10a"]
    check("H10a: U beats B on every seed: the rule says helps; the 5 v 5 exact test's p is its smallest, 1/252",
          h10a["truth_detail_3v3"]["verdict"] == "helps" and h10a["perm_5v5"]["n_splits"] == 252
          and abs(h10a["perm_5v5"]["p"] - 1 / 252.0) < 1e-12, h10a)
    check("  B's seeds 3-4 come from rv2_B_extra",
          p1["arms"]["B"]["runs"]["3"] == "rv2_B_extra/base__s3" and p1["arms"]["B"]["runs"]["0"] == "base_b_v1/base__s0")
    check("H10c-CLASS reads the truth arm's 'with' runs of REC-CLASS-1 only",
          p1["arms"]["REC-CLASS-1"]["runs"] == {str(s): "realloop_v2/truth__s3_REC-CLASS-1__union__s%d" % s
                                               for s in range(3)})
    check("every input score is pinned by its sha256",
          set(p1["inputs"]["rv2_U"]["dev_scores"]) == set(ds["rv2_U"]) and len(p1["spec_sha256"]) == 64)
    bad = copy.deepcopy(ds)
    bad["rv2_U"]["base__s2"]["exam"] = "test"
    check("a score of any other exam refuses the whole panel",
          raises(lambda: PN.build_panel(bad, spec), PN.PanelError, "dev"))
    miss = copy.deepcopy(ds)
    del miss["rv2_U"]["base__s4"]
    check("a missing seed refuses the panel", raises(lambda: PN.build_panel(miss, spec), PN.PanelError, "seed"))
    dup = copy.deepcopy(ds)
    dup["rv2_B_extra"]["base__s0"] = score(0.5)
    check("a seed two experiments claim refuses the panel",
          raises(lambda: PN.build_panel(dup, spec), PN.PanelError, "claimed"))
    tm = copy.deepcopy(ds)
    tm["rv2_B_extra"]["base__s3"].update(production=False, scorer_sha256="TEST-" + "0" * 59)
    check("the 5 v 5 test's seeds 3-4 pass the gate's score rule too: a test-mode B seed 3 refuses the panel",
          raises(lambda: PN.build_panel(tm, spec), PN.PanelError, "production"))
    osc = copy.deepcopy(ds)
    osc["rv2_U"]["base__s4"]["scorer_sha256"] = "7" * 64
    check("  a U seed 4 from another scorer refuses it", raises(lambda: PN.build_panel(osc, spec), PN.PanelError,
                                                                "scorer_sha256"))
    rw = copy.deepcopy(ds)
    rw["rv2_U"]["base__s4"]["weights_sha256"] = rw["rv2_U"]["base__s0"]["weights_sha256"]
    check("  a U seed 4 scored from seed 0's weights refuses it", raises(lambda: PN.build_panel(rw, spec),
                                                                         PN.PanelError, "same weights"))
    t3 = copy.deepcopy(ds)
    t3["rv2_Uctl"]["base__s1"]["production"] = False
    check("  a refused 3 v 3 score is a PanelError too", raises(lambda: PN.build_panel(t3, spec), PN.PanelError,
                                                                "H10b"))
    report = {"steps": [{"step": "REC-CLASS-1", "truth": {"verdict": "helps"}},
                        {"step": "REC-JUDGE-1", "truth": {"verdict": "neutral"}}]}
    h = O.h10(p1, report)
    check("outcome.h10: H10a supported, H10b supported, H10c supported through REC-CLASS-1",
          h["H10a"]["verdict"] == "supported" and h["H10b"]["verdict"] == "supported"
          and h["H10c"]["verdict"] == "supported"
          and [x["passes"] for x in h["H10c"]["parts"]] == [True, False], h)
    flat = PN.build_panel(dev_scores(u=(0.50, 0.51, 0.52, 0.53, 0.54)), spec)
    hf = O.h10(flat, {"steps": [{"step": s, "truth": {"verdict": "neutral"}} for s in ("REC-VETO", "REC-CLASS-1")]})
    check("U equal to B: the rule does not say helps, so H10a is falsified; no step helps, H10c not supported",
          hf["H10a"]["verdict"] == "falsified" and hf["H10c"]["verdict"] == "not_supported", (hf["H10a"], hf["H10c"]))
    auth = {"steps": [{"step": "REC-AUTH-1", "truth": {"verdict": "helps"}},
                      {"step": "REC-CLASS-1", "truth": {"verdict": "neutral"}}]}
    ha = O.h10(p1, auth)["H10c"]
    check("H10c: a recovery step with no same-image control (REC-AUTH-1) that helps supports it (contract 6 H10c)",
          ha["verdict"] == "supported" and [x["step"] for x in ha["parts"] if x["passes"]] == ["REC-AUTH-1"], ha)
    check("  without realloop_v2's report H10c is not evaluated (not 'not supported')",
          O.h10(p1, None)["H10c"]["verdict"] == "not_evaluated"
          and O.h10(p1, {"steps": []})["H10c"]["verdict"] == "not_evaluated")
    lose = copy.deepcopy(ds)
    lose["rv2_CLASSctl"] = base_runs([0.70, 0.71, 0.72], [0, 1, 2])
    hl = O.h10(PN.build_panel(lose, spec), {"steps": [{"step": "REC-CLASS-1", "truth": {"verdict": "helps"}}]})["H10c"]
    check("  a step that helps but loses to its same-image control does not support it",
          hl["verdict"] == "not_supported" and hl["parts"][0]["p_vs_control"] == 0.0, hl)
    nc = {"comparisons": [c for c in p1["comparisons"] if c["id"] != "H10c-CLASS"], "arms": p1["arms"]}
    hn = O.h10(nc, {"steps": [{"step": "REC-CLASS-1", "truth": {"verdict": "helps"}}]})["H10c"]
    check("  a step that helps while its pre-registered control is missing from the panel is not passed",
          hn["verdict"] == "not_evaluated" and hn["parts"][0]["passes"] is None, hn)
    rule_only = {"comparisons": [{"id": "H10a", "truth_detail_3v3": {"verdict": "helps", "p": 0.8},
                                  "perm_5v5": {"p": 0.05}}]}
    check("helps by the rule but 5 v 5 p 0.05 > 0.025: inconclusive, never supported",
          O.h10(rule_only)["H10a"]["verdict"] == "inconclusive")
    check("no panel comparison: not_evaluated", O.h10({"comparisons": []})["H10a"]["verdict"] == "not_evaluated")
    check("outcome's 'helps' line is the pinned gate's (H10_P_MIN = GateConfig.p_accept)",
          O.H10_P_MIN == G.GateConfig().p_accept, (O.H10_P_MIN, G.GateConfig().p_accept))


def test_h11_and_score_da():
    print("outcome.h11 and score_da, record_da")
    fc = {"S8": 0.5, "S9": 0.3, "S12": 0.2}
    rec = ["S8", "S9", "S10", "S12"]
    audit = {"valid": True, "stage_ranking": [{"stage": "S8", "recoverable_lb": 900.0},
                                              {"stage": "S9", "recoverable_lb": 40.0}],
             "stages": {"S8": {"fn_rate": {"interval": [0.2, 0.4]}}, "S9": {"fn_rate": {"interval": [0.01, 0.05]}}},
             "strata": [{"stratum": "S8|Ragweed", "interval": [0.7, 0.9]}]}
    h = O.h11({"stage_forecast": fc}, audit, rec)
    check("H11 supported: top-1 S8 is the audit's first stage and its log-loss beats uniform",
          h["verdict"] == "supported" and h["top1"] == "S8" and h["log_loss"] < h["uniform_log_loss"], h)
    h2 = O.h11({"stage_forecast": {"S8": 0.1, "S9": 0.8, "S12": 0.1}}, audit, rec)
    check("H11 falsified when the top-1 stage is not the audit's first", h2["verdict"] == "falsified")
    check("H11 not_evaluated without the audit", O.h11({"stage_forecast": fc}, None, rec)["verdict"] == "not_evaluated")
    import math
    tie = O.h11({"stage_forecast": {"S8": 0.5, "S9": 0.5}}, audit, rec)
    check("  the uniform forecast is over the ledger's %d recoverable stages, not the %d the DA named "
          "(log %d, as funnel/estimate.py h11)" % (len(rec), 2, len(rec)),
          tie["verdict"] == "supported" and abs(tie["uniform_log_loss"] - math.log(len(rec))) < 1e-12
          and tie["stages"] == len(rec), tie)
    check("  an invalid audit (calibration overlap) is never scored",
          O.h11({"stage_forecast": fc}, dict(audit, valid=False), rec)["verdict"] == "not_evaluated")
    check("  without the recoverable stages and without the audit's own H11: not evaluated",
          O.h11({"stage_forecast": fc}, audit)["verdict"] == "not_evaluated")
    own = dict(audit, hypotheses={"H11": {"verdict": "falsified", "why": "estimate's",
                                          "parts": {"top_stage": "S9", "forecast_top1": "S8"}}})
    ho = O.h11({"stage_forecast": fc}, own, rec)
    check("  the audit's own decided H11 (funnel/estimate.py) is taken as it is",
          ho["verdict"] == "falsified" and ho["truth"] == "S9" and "estimate.py" in ho["source"], ho)
    cas = [{"index": 0, "kept": True, "counted": True,
            "prediction": {"metric": "fn_rate", "stage": "S8", "direction": "above", "threshold": 0.1}},
           {"index": 1, "kept": True, "counted": True,
            "prediction": {"metric": "fn_rate", "stage": "S9", "direction": "above", "threshold": 0.1}},
           {"index": 2, "kept": True, "counted": True,
            "prediction": {"metric": "purity", "stratum": "S8|Ragweed", "direction": "above", "threshold": 0.5}},
           {"index": 3, "kept": True, "counted": True,
            "prediction": {"metric": "truth_verdict", "direction": "helps"}},
           {"index": 4, "kept": True, "counted": False,
            "prediction": {"metric": "fn_rate", "stage": "S8", "direction": "above", "threshold": 0.1}}]
    pda = {"model": {"resolved": ADV}, "digest_sha256": "d" * 64, "stage_forecast": fc,
           "validation": {"counter_arguments": cas}}
    s = O.score_da(pda, audit, recoverable_stages=rec)
    check("score_da: fn_rate above on S8 correct, on S9 wrong, purity correct, truth_verdict insufficient "
          "without a panel; an uncounted echo is not scored",
          [p["verdict"] for p in s["predictions"]] == ["correct", "wrong", "correct", "insufficient"]
          and s["correct"] == 2 and s["scored"] == 3, s["predictions"])
    from weed_optimizer_framework.tools.brain import policy as POL
    check("the adversary's actor string is one policy._ACTOR_RE accepts (tier2:adversary/<model>)",
          s["proposed_by"] == "tier2:adversary/vllm_glm-4.7-flash" and POL._ACTOR_RE.match(s["proposed_by"]),
          s["proposed_by"])
    ev_path, sum_path = TMP / "track.jsonl", TMP / "track.json"
    O.record_da(pda, audit, events_path=ev_path, summary_path=sum_path, recoverable_stages=rec)
    sinv = O.score_da(pda, dict(audit, valid=False), recoverable_stages=rec)
    check("score_da on an invalid audit: its fn_rate and purity predictions stay insufficient, H11 not evaluated",
          [p["verdict"] for p in sinv["predictions"]] == ["insufficient"] * 4 and sinv["scored"] == 0
          and sinv["h11"]["verdict"] == "not_evaluated", sinv["predictions"])
    summ = jload(sum_path)["proposers"][s["proposed_by"]]
    check("record_da folds into the adversary's track record (predictions, H11)",
          summ["correct"] == 2 and summ["h11_supported"] == 1 and summ["h11_scored"] == 1, summ)


# ------------------------------------------------------------------ remote.py on a temporary INC_DIR
class LocalPolicy(object):
    """policy.describe with the cluster's INC_DIR in each path pattern
    replaced by this test's INC_DIR."""

    def __enter__(self):
        pol = RM._policy()
        self.pol, self.real = pol, pol.describe
        local = re.escape(str(INC) + "/")
        ocean = re.escape(OCEAN_INC)

        def fix(o):
            if isinstance(o, dict):
                return {k: (v.replace(ocean, local) if k == "pattern" and isinstance(v, str) else fix(v))
                        for k, v in o.items()}
            return o
        pol.describe = lambda action: fix(self.real(action))
        return self

    def __exit__(self, *exc):
        self.pol.describe = self.real
        return False


def write(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, (str, bytes)) else obj)
    return path


def test_remote_dev_scores():
    print("remote funnel dev-scores: dev.json of the base or truth 'with' runs, nothing else")
    for s in range(3):
        write(INC / "rv2_U" / "runs" / ("base__s%d" % s) / "scores" / "dev.json", score(0.6 + s / 100.0))
        write(INC / "rv2_U" / "runs" / ("base__s%d" % s) / "scores" / "test.json", score(0.9, exam="test"))
    write(INC / "rv2_U" / "runs" / "inc__s0" / "scores" / "dev.json", score(0.1))
    write(INC / "realloop_v2" / "runs" / "truth__s3_REC-CLASS-1__union__s0" / "scores" / "dev.json", score(0.61))
    write(INC / "realloop_v2" / "runs" / "truth__s3_REC-CLASS-1__without__s0" / "scores" / "dev.json", score(0.5))
    rec = RM.funnel_dev_scores(["rv2_U"])
    check("base runs' dev scores only (no test.json, no other run)",
          rec["ok"] and sorted(rec["scores"]["rv2_U"]) == ["base__s0", "base__s1", "base__s2"]
          and all(k.endswith("/scores/dev.json") for k in rec["files"]) and rec["exams_read"] == ["dev"], rec)
    t = RM.funnel_dev_scores(["realloop_v2"], truth_step="REC-CLASS-1")
    check("--truth-step: the step's 'with' runs only",
          t["ok"] and list(t["scores"]["realloop_v2"]) == ["truth__s3_REC-CLASS-1__union__s0"], t.get("scores"))
    write(INC / "rv2_bad" / "runs" / "base__s0" / "scores" / "dev.json", score(0.6, exam="test"))
    b = RM.funnel_dev_scores(["rv2_bad"])
    check("a dev.json stamped with another exam is refused", not b["ok"] and b.get("error_kind") == "refused", b)
    check("an experiment name that is not a name is refused",
          not RM.funnel_dev_scores(["../x"])["ok"] and not RM.funnel_dev_scores(["x"], truth_step="a/b")["ok"])
    spec = {"arms": {"U": {"kind": "base", "exps": ["rv2_U"], "seeds": [0, 1, 2]}}, "comparisons": []}
    p = PN.build_panel(rec["scores"], spec)
    check("its record builds a panel as it is", p["arms"]["U"]["runs"]["2"] == "rv2_U/base__s2")


def test_remote_summary():
    print("remote funnel summary and ledger-summaries")
    for f in ("admit_summary.json", "select_summary.json", "pool_summary.json", "calibration.json"):
        shutil.copyfile(FF / "step1" / f, str(INC / "step1" / f) if (INC / "step1").is_dir()
                        else str((INC / "step1").mkdir(parents=True) or INC / "step1" / f))
    (INC / "funnel").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FF / "census_v0.json", INC / "funnel" / "census_v0.json")
    write(INC / "funnel" / "audit_v1.json", {"format": "funnel-audit/1", "valid": True,
                                             "stages": {"S8": {"fn_rate": {"interval": [0.0, 0.1]}}},
                                             "by_exam": {"test": {"map50_95": 0.9}, "dev": {"map50_95": 0.5}}})
    write(INC / "funnel" / "ledger.jsonl", '{"row": 1}\n')
    write(INC / "funnel" / "sample_v1_key.jsonl", '{"key": 1}\n')
    write(INC / "funnel" / "rl_answers" / "RL-B" / "sheet_001.json", {"answers": []})
    write(INC / "step1" / "conflicts.csv", "a,b\n")
    rec = RM.funnel_summary(derive_ledger=True)
    arts = rec["decision"]["artifacts"]
    check("the summary ships the audit, the Step 1 pool summary and calibration, and the funnel listing",
          rec["ok"] and "funnel/audit_v1.json" in arts and "step1/pool_summary.json" in arts
          and "step1/calibration.json" in arts and arts["funnel/files.json"]["format"] == "funnel-files/1",
          sorted(arts))
    listing = arts["funnel/files.json"]["files"]
    check("the listing names files with sha256 and size, to depth 3, never their content",
          "funnel/rl_answers/RL-B/sheet_001.json" in listing and "funnel/ledger.jsonl" in listing
          and set(listing["funnel/ledger.jsonl"]) == {"sha256", "bytes"}, sorted(listing))
    text = json.dumps(rec)
    check("never shipped: the row-level ledger, the key file, conflicts.csv (their contents are absent)",
          '"row": 1' not in text and '"key": 1' not in text and "a,b" not in text
          and "funnel/ledger.jsonl" in rec["never_shipped"])
    check("dev_only redacts the audit's non-dev exam", "test" not in arts["funnel/audit_v1.json"]["by_exam"]
          and rec["redacted"], rec["redacted"])
    led = rec["decision"]["derived"].get("funnel_ledger")
    fix = jload(FF / "funnel_ledger_summaries.json")
    check("--derive-ledger: with no ledger file, the summaries-derived ledger is built in memory, the fixture's "
          "(same fingerprint), and nothing is written",
          isinstance(led, dict) and led.get("fingerprint") == fix["fingerprint"]
          and not (INC / "funnel" / "funnel_ledger.json").exists(), (led or {}).get("fingerprint"))
    ev = E.from_snapshot(rec, "realloop_v1")
    check("evidence.from_snapshot reads the record: the derived ledger as funnel/funnel_ledger.json, marked derived",
          ev.json(E.FUNNEL_LEDGER) == led and ev.provenance[E.FUNNEL_LEDGER]["source"] == "derived"
          and ev.json(E.FUNNEL_AUDIT) is not None, ev.provenance.get(E.FUNNEL_LEDGER))
    w = RM.funnel_ledger_summaries(write=True)
    check("ledger-summaries --write writes the derived ledger once",
          w["ok"] and pathlib.Path(w["written"]).is_file() and w["fingerprint"] == fix["fingerprint"], w.get("error"))
    w2 = RM.funnel_ledger_summaries(write=True)
    check("  a second --write of the same derivation writes nothing and is not refused",
          w2["ok"] and "written" not in w2, w2.get("error"))
    cur = jload(INC / "funnel" / "funnel_ledger.json")
    cur["derivation"] = "census"
    write(INC / "funnel" / "funnel_ledger.json", cur)
    w3 = RM.funnel_ledger_summaries(write=True)
    check("  a census-derived ledger is never replaced", not w3["ok"] and w3.get("error_kind") == "refused", w3)
    s2 = RM.funnel_summary(derive_ledger=True)
    check("with a ledger file on disk, the summary ships it and derives nothing",
          "funnel/funnel_ledger.json" in s2["decision"]["artifacts"] and not s2["decision"]["derived"])


def test_remote_submit():
    print("remote submit funnel: dry run, resources, refusals")
    prereg = write(INC / "funnel" / "prereg_v1.json", {"format": "x"})
    for f in ("audit_v1.json", "class_maps.json"):
        if not (INC / "funnel" / f).exists():
            write(INC / "funnel" / f, {})
    (INC / "step1_r1").mkdir(exist_ok=True)
    out = str(INC / "funnel") + "/"
    with LocalPolicy():
        r = RM.submit("funnel", ["rl-b", "--prereg", str(prereg), "--out", out], dry_run=True)
        check("rl-b dry run: admitted by L10, job inc_funnel_rl-b, the H100 flags before the script",
              r["ok"] and [x["lever"] for x in r["menu"]["levers"]] == ["L10"] and r["job_name"] == "inc_funnel_rl-b"
              and r["sbatch_argv"][3:6] == ["--gres=gpu:h100-80:1", "--cpus-per-task=12", "--mem=80G"]
              and r["sbatch_argv"][6].endswith("run_inc_funnel.sh")
              and r["sbatch_argv"][7:] == ["rl-b", "--prereg", str(prereg), "--out", out], r.get("error") or r)
        r = RM.submit("funnel", ["census", "--prereg", str(prereg), "--out", out], dry_run=True)
        check("census dry run: no extra flags", r["ok"] and r["sbatch_argv"][3].endswith("run_inc_funnel.sh"),
              r.get("error"))
        r = RM.submit("funnel", ["qualify", "--prereg", str(prereg), "--out", out, "--rl"], dry_run=True)
        check("qualify --rl is admitted", r["ok"] and r["params"].get("rl") is True, r.get("error"))
        r = RM.submit("funnel", ["map", "--prereg", str(prereg), "--out", out, "--part", "geometry"], dry_run=True)
        check("map --part geometry: L11, job inc_funnel_map_geometry",
              r["ok"] and r["job_name"] == "inc_funnel_map_geometry"
              and [x["lever"] for x in r["menu"]["levers"]] == ["L11"], r.get("error"))
        r = RM.submit("funnel", ["recover", "--prereg", str(prereg), "--audit", str(INC / "funnel" / "audit_v1.json"),
                                 "--maps", str(INC / "funnel" / "class_maps.json"), "--policy", "R-A",
                                 "--out", str(INC / "step1_r1") + "/"], dry_run=True)
        check("recover --policy R-A: L13", r["ok"] and [x["lever"] for x in r["menu"]["levers"]] == ["L13"],
              r.get("error"))
        for args, why in ((["census", "--prereg", str(prereg), "--out", out, "--rl"], "--rl belongs to qualify"),
                          (["map", "--prereg", str(prereg), "--out", out], "needs --part"),
                          (["fetch", "--prereg", str(prereg), "--out", out], "does not run"),
                          (["census", "--prereg", "/etc/hosts", "--out", out], "outside INC_DIR"),
                          (["census", "--prereg", str(prereg), "--out", out, "--part", "geometry"], "not accepted"),
                          (["--prereg", str(prereg)], "start with the verb"),
                          (["recover", "--prereg", str(prereg), "--audit", str(INC / "funnel" / "audit_v1.json"),
                            "--maps", str(INC / "funnel" / "class_maps.json"), "--policy", "R-Z",
                            "--out", str(INC / "step1_r1") + "/"], "recovery policies")):
            r = RM.submit("funnel", args, dry_run=True)
            check("refused: %s" % why, not r["ok"] and why in (r.get("error") or ""), r.get("error"))
        ov = INC / "step1_r1"
        base = write(INC / "step1" / "base_B.jsonl", "{}\n")
        req = RM.parse_builder_args("build", ["realloop", "build", "--exp", "realloop_v2", "--replay-mode", "full",
                                              "--recipes", "full", "--base", str(base), "--increment-sources",
                                              "recovered", "--step1-overlay", str(ov), "--size", "287"])
        check("realloop build --increment-sources recovered --step1-overlay parses",
              req["params"]["step1_overlay"] == str(ov) and req["params"]["increment_sources"] == "recovered")
        check("  recovered without --step1-overlay is refused",
              raises(lambda: RM.parse_builder_args("build", ["realloop", "build", "--exp", "x", "--replay-mode", "full",
                                                             "--recipes", "full", "--increment-sources", "recovered",
                                                             "--size", "5"]), RM.Refused, "go together"))


# ------------------------------------------------------------------ executor
def test_executor_render():
    print("executor: the funnel actions' rendering and parsing")
    P, O_ = OCEAN_INC + "funnel/prereg_v1.json", OCEAN_INC + "funnel/"
    argv = ["sbatch", "--gres=gpu:h100-80:1", "--cpus-per-task=12", "--mem=80G", "run_inc_funnel.sh", "rl-b",
            "--prereg", P, "--out", O_]
    p = X.params_from_argv("inc_funnel_audit", argv)
    r = X.render("inc_funnel_audit", p)
    check("L10's argv (sbatch flags before the script) parses to verb, prereg, out",
          p == {"verb": "rl-b", "prereg": P, "out": O_}, p)
    check("  it renders the script line and 'submit funnel -- VERB ...', and argv_check accepts the lever's argv",
          r["builder"] == ["run_inc_funnel.sh", "rl-b", "--prereg", P, "--out", O_]
          and r["remote"] == ["submit", "funnel", "--", "rl-b", "--prereg", P, "--out", O_]
          and X.argv_check(r, argv)[0], r)
    q = X.params_from_argv("inc_funnel_audit", ["sbatch", "run_inc_funnel.sh", "qualify", "--prereg", P,
                                                "--out", O_, "--rl"])
    check("qualify --rl: rl = 1, rendered back as --rl",
          q.get("rl") == 1 and X.render("inc_funnel_audit", q)["remote"][-1] == "--rl", q)
    m = X.render("inc_funnel_map", {"prereg": P, "out": O_, "part": "relation"})
    check("map renders with its --part", m["remote"] == ["submit", "funnel", "--", "map", "--prereg", P, "--out", O_,
                                                         "--part", "relation"], m)
    rc = X.render("inc_funnel_recover", {"prereg": P, "audit": OCEAN_INC + "funnel/audit_v1.json",
                                         "maps": OCEAN_INC + "funnel/class_maps.json", "policy": "R-A,R-C",
                                         "out": OCEAN_INC + "step1_r1/"})
    check("recover renders with its policy", rc["remote"][3] == "recover" and "R-A,R-C" in rc["remote"], rc)
    f = X.render("inc_funnel_fetch", {"prereg": "p", "what": "taxonomy", "names_from": "n", "out": "o"})
    check("fetch runs on the lab (local), never as a remote verb",
          f["local"] and f["remote"] is None and f["builder"][:2] == ["funnel", "fetch"])
    check("verify queue and funnel sync are local", X.render("inc_verify_queue", {})["local"]
          and X.render("inc_funnel_sync", {})["local"])
    check("a missing verb refuses", raises(lambda: X.render("inc_funnel_audit", {"prereg": P, "out": O_}), X.ExecError))
    rl = X.render("inc_build_realloop", {"exp": "realloop_v2", "replay_mode": "full", "recipes": "full",
                                         "increment_sources": "recovered", "step1_overlay": OCEAN_INC + "step1_r1/",
                                         "size": 287})
    check("realloop renders --step1-overlay", "--step1-overlay" in rl["builder"]
          and X.params_from_argv("inc_build_realloop", ["python", "-m"] + ["weed_optimizer_framework.tools."
                                                                          + rl["builder"][0]] + rl["builder"][1:])
          ["step1_overlay"] == OCEAN_INC + "step1_r1/", rl["builder"])
    bl = X.render("inc_build_baseline", {"exp": "rv2_U", "manifest": OCEAN_INC + "x.jsonl", "seeds": "0,1,2,3,4"})
    check("build-baseline renders --seeds", bl["builder"][-2:] == ["--seeds", "0,1,2,3,4"], bl["builder"])


def test_lab_hooks():
    print("executor: the lab hooks")
    lab = TMP / "labinc"
    write(lab / "funnel" / "taxonomy_cache.json", {})
    write(lab / "funnel" / "cards" / "a" / "card.json", {})
    write(lab / "funnel" / "prospective_da.json", {})
    write(lab / "funnel" / "audit_v1.json", {})               # the lab never pushes an audit
    files = X.funnel_sync_list(lab)
    check("the sync list holds only the fixed entries that exist",
          files == ["funnel/taxonomy_cache.json", "funnel/cards/a/card.json", "funnel/prospective_da.json"], files)
    check("a path outside the list, or with '..', is refused",
          X.funnel_sync_refusals(["funnel/audit_v1.json", "funnel/cards/../audit_v1.json", "funnel/cards/x"])
          == ["funnel/audit_v1.json", "funnel/cards/../audit_v1.json"])
    calls = []

    def runner(argv, **kw):
        calls.append((argv, kw.get("input")))
        if argv[0] == "ssh" and "funnel-pull-list" in argv[2]:
            return types.SimpleNamespace(returncode=0, stderr="",
                                         stdout='INCAP {"files": {}, "prereg": null, "verb": "funnel-pull-list"}\n')
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    res = X.funnel_sync_hook(lab, "cluster-host", cluster_inc="/c/inc", runner=runner, repo="/c/repo")({})
    want = {f: hashlib.sha256((lab / f).read_bytes()).hexdigest() for f in files}
    check("the sync rsyncs the list (--files-from on stdin), then checks the arrival over ssh: the pushed "
          "files' sha256 on stdin, no fetch manifest among them (flag 0)",
          res["ok"] and calls[0][0][:3] == ["rsync", "-a", "--files-from=-"] and calls[0][1].split() == files
          and calls[0][0][-1] == "cluster-host:/c/inc/" and calls[1][0][:2] == ["ssh", "cluster-host"]
          and shlex.quote(X._SYNC_CHECK_PY) in calls[1][0][2] and calls[1][0][2].endswith(" /c/inc 0")
          and json.loads(calls[1][1]) == want, calls)
    check("  then the pull's listing (one ssh; nothing listed, nothing pulled)",
          len(calls) == 3 and calls[2][0][:2] == ["ssh", "cluster-host"] and "funnel-pull-list" in calls[2][0][2]
          and json.loads(calls[2][1]) == list(X.FUNNEL_PULL_FILES) and res.get("pulled") == [], calls[2:])
    check("no ssh target: refused, nothing run",
          not X.funnel_sync_hook(lab, "", runner=runner)({})["ok"] and len(calls) == 3)
    del calls[:]
    res = X.funnel_sync_hook(lab, "cluster-host", cluster_inc="/c/inc", runner=runner, repo="/c/repo",
                             data_target="dtn-host")({})
    check("with a data-transfer node, rsync goes to it and the ssh checks stay on the login target (the "
          "login node has no rsync)",
          res["ok"] and calls[0][0][0] == "rsync" and calls[0][0][-1] == "dtn-host:/c/inc/"
          and all(c[0][:2] == ["ssh", "cluster-host"] for c in calls[1:]), [c[0][:2] + c[0][-1:] for c in calls])
    check("the campaign's hook uses the data-transfer node (model.CLUSTER_DATA_SSH)",
          "data_target=M.CLUSTER_DATA_SSH" in (ROOT / "weed_optimizer_framework" / "tools" / "inc_autopilot"
                                               / "campaign.py").read_text())
    print("executor: the sync's arrival check, run here on a copy standing for the cluster")
    arrive = TMP / "arrive_inc"
    shutil.copytree(str(lab), str(arrive))

    def arrival(stdin_map, manifest="0"):
        p = subprocess.run([sys.executable, "-c", X._SYNC_CHECK_PY, str(arrive), manifest],
                           input=json.dumps(stdin_map), capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, PYTHONPATH=str(ROOT)))
        rec = json.loads(p.stdout.split("INCAP ", 1)[1]) if "INCAP " in p.stdout else {}
        return p.returncode, rec
    rc, rec = arrival(want)
    check("every pushed file hashes as on the lab: exit 0", rc == 0 and rec.get("checked") == len(want)
          and rec.get("mismatched") == [], rec)
    (arrive / "funnel" / "prospective_da.json").write_text('{"tampered": true}')
    rc, rec = arrival(want)
    check("  a file changed in transit (the DA record, in no fetch manifest) fails the check",
          rc == 1 and [m[0] for m in rec.get("mismatched") or []] == ["funnel/prospective_da.json"], rec)
    shutil.copyfile(lab / "funnel" / "prospective_da.json", arrive / "funnel" / "prospective_da.json")
    man = {"files": {"cards/a/card.json": hashlib.sha256((lab / "funnel" / "cards" / "a" / "card.json")
                                                         .read_bytes()).hexdigest(),
                     "kt7/never_pushed.jpg": "0" * 64}, "domain": "weed"}
    write(arrive / "funnel" / "fetch_manifest.json", man)
    want_m = dict(want, **{"funnel/fetch_manifest.json":
                           hashlib.sha256((arrive / "funnel" / "fetch_manifest.json").read_bytes()).hexdigest()})
    rc, rec = arrival(want_m, "1")
    check("  with a fetch manifest pushed, fetch.check_manifest runs too and refuses a file it lists that is missing",
          rc == 1 and "StaleInput" in str(rec.get("manifest")), rec)
    test_funnel_pull()


def test_funnel_pull():
    """Cluster -> lab (runner 6.3), with a directory standing for the cluster:
    the listing script and rsync run for real, the ssh and the host are
    stripped."""
    print("executor: the pull (cluster -> lab), run on a directory standing for the cluster")
    cl, lab = TMP / "pull_cluster", TMP / "pull_lab"
    pre = jload(PREREG)
    write(cl / "funnel" / "census_v1.json", {"rows": []})
    write(cl / "funnel" / "sheets_v1" / "index.json", {"sheets": 1})
    write(cl / "funnel" / "sheets_v1" / "s001" / "sheet.json", {"items": []})
    write(cl / "funnel" / "sheets_v1_key" / "s001.json", {"key": "never moved"})
    write(cl / "funnel" / "sample_v1_key.jsonl", '{"k": 1}\n')
    write(cl / "step1_r1" / "recovery.json", {"status": "complete"})
    write(cl / "step1" / "pool_verdicts.json", {"never": True})
    write(cl / "funnel" / "prereg_v1.json", dict(pre, amendments=list(pre.get("amendments") or [])
                                                 + [{"kind": "sample_lock", "sha256": "a" * 64}]))
    write(lab / "funnel" / "prereg_v1.json", pre)

    def local(corrupt=None, listing=None):
        def runner(argv, **kw):
            if argv[0] == "ssh":
                if listing is not None:
                    return types.SimpleNamespace(returncode=0, stderr="", stdout="INCAP " + json.dumps(listing) + "\n")
                toks = shlex.split(argv[2])
                i = toks.index("-c")
                args = [str(cl) if a == "/c/inc" else a for a in toks[i + 2:]]
                return subprocess.run([sys.executable, "-c", toks[i + 1]] + args, input=kw.get("input"),
                                      capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT)))
            a = [x.split(":", 1)[1].replace("/c/inc", str(cl)) if x.startswith("cluster-host:") else x for x in argv]
            r = subprocess.run(a, input=kw.get("input"), capture_output=True, text=True)
            if corrupt:
                corrupt()
            return r
        return runner
    res = X.funnel_pull(lab, "cluster-host", cluster_inc="/c/inc", runner=local(), repo="/c/repo")
    moved = sorted(str(p.relative_to(lab)) for p in lab.rglob("*") if p.is_file())
    check("the listed files come back verified: the census, the sheets, the recovery record, and the "
          "pre-registration whose amendments grew; no key file, nothing under step1/",
          res["ok"] and res["pulled"] == ["funnel/census_v1.json", "funnel/prereg_v1.json",
                                          "funnel/sheets_v1/index.json", "funnel/sheets_v1/s001/sheet.json",
                                          "step1_r1/recovery.json"]
          and moved == res["pulled"] and "grew" in res["prereg"]
          and len(jload(lab / "funnel" / "prereg_v1.json")["amendments"]) == len(pre.get("amendments") or []) + 1
          and not (lab / ".funnel_pull").exists(), (res, moved))
    again = X.funnel_pull(lab, "cluster-host", cluster_inc="/c/inc", runner=local(), repo="/c/repo")
    check("  a second pull moves nothing", again["ok"] and again["pulled"] == [] and again["prereg"] == "unchanged",
          again)
    write(cl / "funnel" / "census_v1.json", {"rows": [1]})
    before = (lab / "funnel" / "census_v1.json").read_bytes()

    def tamper():
        for f in (lab / ".funnel_pull").rglob("census_v1.json"):
            f.write_text('{"rows": ["tampered"]}')
    bad = X.funnel_pull(lab, "cluster-host", cluster_inc="/c/inc", runner=local(tamper), repo="/c/repo")
    check("  a file changed in transit is refused and the lab's copy is left as it was",
          not bad["ok"] and "changed in transit" in bad["error"]
          and (lab / "funnel" / "census_v1.json").read_bytes() == before, bad)
    out = X.funnel_pull(lab, "cluster-host", cluster_inc="/c/inc", repo="/c/repo",
                        runner=local(listing={"files": {"funnel/sample_v1_key.jsonl": "0" * 64}, "prereg": None}))
    check("  a listing naming a path outside the pull list (a key file) is refused",
          not out["ok"] and "outside the pull list" in out["error"], out)
    write(cl / "funnel" / "prereg_v1.json", dict(pre, version="another"))
    other = X.funnel_pull(lab, "cluster-host", cluster_inc="/c/inc", runner=local(), repo="/c/repo")
    check("  a cluster pre-registration with another core is refused, nothing pulled",
          not other["ok"] and "another core" in other["error"]
          and (lab / "funnel" / "census_v1.json").read_bytes() == before, other)
    print("levers.funnel_sync_needed: both directions")
    listing = {"format": "funnel-files/1", "files": {
        "funnel/sheets_v1/index.json": {"sha256": "1" * 64, "bytes": 10},
        "funnel/ledger.jsonl": {"sha256": "2" * 64, "bytes": 10},
        "funnel/taxonomy_cache.json": {"sha256": "3" * 64, "bytes": 10}}}

    def need(ctx):
        ev = E.from_texts({"funnel/files.json": json.dumps(listing)}, "realloop_v1", context=ctx)
        return LV.funnel_sync_needed(ev)
    check("a pull-list file the lab lacks is needed; a never-moved one (ledger.jsonl) is not",
          need({}) == ["funnel/sheets_v1/index.json"], need({}))
    check("  the lab's copy of the same version is not needed; another version is",
          need({"funnel_lab_pull": {"funnel/sheets_v1/index.json": "1" * 64}}) == []
          and need({"funnel_lab_pull": {"funnel/sheets_v1/index.json": "9" * 64}}) == ["funnel/sheets_v1/index.json"])
    check("  a lab push-list file the cluster holds in another version is needed (push)",
          need({"funnel_lab": {"funnel/taxonomy_cache.json": "4" * 64},
                "funnel_lab_pull": {"funnel/sheets_v1/index.json": "1" * 64}}) == ["funnel/taxonomy_cache.json"])
    q = TMP / "verify_queue.jsonl"
    rows = [{"source": "src_a", "source_class": "3", "proposed": "Ragweed", "why": "card contradicts"}]
    a1, a2 = X.write_verify_queue(rows, q), X.write_verify_queue(rows, q)
    lines = q.read_text().splitlines()
    check("the verify queue gets each task once", a1["added"] == 1 and a2["added"] == 0 and len(lines) == 1
          and json.loads(lines[0])["kind"] == "class_map_verify")
    argv = X.funnel_fetch_argv({"prereg": "p.json", "what": "taxonomy", "names_from": "n.json", "out": "o/"}, "py")
    check("the fetch hook runs the funnel CLI's fetch",
          argv == ["py", "-m", X.FUNNEL_MODULE, "fetch", "--prereg", "p.json", "--what", "taxonomy",
                   "--names-from", "n.json", "--out", "o/"], argv)


# ------------------------------------------------------------------ campaign: the DA flow
def world(name):
    root = TMP / name
    (root / "step1").mkdir(parents=True)
    (root / "funnel").mkdir(parents=True)
    (root / "realloop_v1").mkdir()
    for f in ("admit_summary.json", "select_summary.json", "pool_summary.json", "calibration.json"):
        shutil.copyfile(FF / "step1" / f, root / "step1" / f)
    for f in ("exp.json", "report.json", "ledger.jsonl", "build_summary.json"):
        shutil.copyfile(FF / "realloop_v1" / f, root / "realloop_v1" / f)
    shutil.copyfile(FF / "funnel_ledger_summaries.json", root / "funnel" / "funnel_ledger.json")
    return root


def test_campaign_da():
    print("campaign: decisions, the DA pass staged blind, merged, recorded once")
    from weed_optimizer_framework.tools.inc_autopilot import campaign as CP
    lab = TMP / "lab"
    paths = CP.Paths(lab)
    paths.claims.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FF / "claims" / "claims_seed.json", paths.claims)
    lab_tree(paths)
    r = CP._Run("funnelcamp", {"enabled": True}, paths, CP._SshBudget(None), lambda: 1790000000.0,
                CP._Log(), None, {}, None, None, None)
    r.st = CP._blank_state("funnelcamp", r.cfg)
    r.st["exp"] = "realloop_v1"
    root = world("campaign_world")
    ev = E.load_dir(root, "realloop_v1", exps=["realloop_v1"], claims=jload(paths.claims))
    r.ev = ev
    diags = DG.detect(ev)
    prop = LV.propose(diags, ev)
    r._funnel_step(ev, diags, prop)
    r._funnel_step(ev, diags, prop)
    led = [json.loads(x) for x in paths.ledger.read_text().splitlines()]
    dec = [e for e in led if e["event"] == "decisions_recorded"]
    check("DEC-1..DEC-10 are logged once, decided_by human-delegated",
          len(dec) == 1 and dec[0]["decided_by"] == "human-delegated" and len(dec[0]["decisions"]) == 10, dec)
    da = r.st["da"]
    staged = [e for e in led if e["event"] == "da_staged"]
    check("D17's OP_DA stages one DA pass, for C1 only: a reply answers one claim (the second tick does not "
          "stage again)", da["status"] == "staged" and da["claim_ids"] == ["C1"] and len(staged) == 1, da)
    digest = jload(paths.plan_input("funnelcamp", da["n"]))
    check("  the staged digest is an inc-da-digest/1 that check_staged accepts, with its sha256 recorded, "
          "holding C1 alone",
          digest["schema"] == BP.DA_DIGEST_SCHEMA and BP.check_staged(digest) == digest["prompt"]
          and da["digest_sha256"] == digest["sha256"] and digest["claim_ids"] == ["C1"]
          and [c["id"] for c in digest["sections"]["claims"]] == ["C1"], digest.get("claim_ids"))
    real = CP.Paths().r14_fixtures
    marks = BP.blind_markers(None, None, real, [])
    txt = json.dumps(digest)
    check("  the lab's R14 fixture paths are the two DA fixtures; their sha256 are markers, absent from the digest",
          real == [FF / "da" / "da_positive.json", FF / "da" / "da_sycophantic.json"]
          and marks == [hashlib.sha256(x.read_bytes()).hexdigest() for x in real]
          and not [m for m in marks if m in txt], (real, marks))
    ctx = types.SimpleNamespace(campaign_dir=paths.campaign_dir)
    seg = X._plan_segment(ctx, "inc_plan_submit", {"campaign": "funnelcamp", "n": da["n"],
                                                   "digest_sha256": da["digest_sha256"]})
    args = shlex.split(seg)
    check("the plan segment ships it as the adversary's job (inc_da_<campaign>_<n>, role adversary)",
          args[-1] == "adversary" and args[8] == "inc_da_funnelcamp_%d" % da["n"], args[4:9])
    test_plan_submit_script(args, paths.plan_input("funnelcamp", da["n"]).read_bytes())
    pos = jload(FF / "da" / "da_positive.json")
    job = {"schema": V.DA_REPLY_SCHEMA, "ok": True, "reply": pos, "model": "glm-4.7-flash",
           "model_used": "glm-4.7-flash", "digest_sha256": da["digest_sha256"]}
    r.st["brain"] = {"model_resolved": PLANNER}
    da["status"] = "submitted"
    r._merge_da(job)
    reg = jload(paths.claims)
    c1 = [c for c in reg["claims"] if c["id"] == "C1"][0]
    pda = paths.lab_inc / "funnel" / "prospective_da.json"
    check("the reply is validated (6 kept) and C1 moves open -> challenged by tier2:adversary/glm-4.7-flash",
          r.st["da"]["status"] == "merged" and r.st["da"]["surviving"] == 6 and c1["status"] == "challenged"
          and c1["history"][-1]["by"] == "tier2:adversary/glm-4.7-flash", (r.st["da"], c1["status"]))
    rec = jload(pda) if pda.is_file() else {}
    check("  funnel/prospective_da.json holds the reply, its validation and the stage forecast",
          rec.get("format") == "funnel-prospective-da/1" and rec.get("stage_forecast") == pos["stage_forecast"]
          and rec["validation"]["counts"]["kept"] == 6 and rec["digest_sha256"] == da["digest_sha256"],
          [e.get("reasons") for e in (json.loads(x) for x in paths.ledger.read_text().splitlines())
           if e["event"] == "da_not_recorded"])
    pre_raw = PREREG.read_bytes()
    from weed_optimizer_framework.tools.funnel import domain as FD
    check("  its header records the pre-registration (sha256 and core sha256), the contract's sha256, the "
          "domain config and the digest it answers (runner 1.2)",
          rec.get("prereg", {}).get("sha256") == hashlib.sha256(pre_raw).hexdigest()
          and rec["prereg"]["core_sha256"] == FD.prereg_core_sha256(json.loads(pre_raw))
          and rec.get("contract", {}).get("sha256") == jload(PREREG)["contract"]["sha256"]
          == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
          and rec.get("domain") == "weed" and rec.get("domain_config", {}).get("sha256")
          and rec.get("inputs", {}).get("digest", {}).get("sha256") == da["digest_sha256"],
          {k: rec.get(k) for k in ("prereg", "contract", "inputs")})
    check("  its blind check lists the digest's hashed markers and finds none of them in the staged digest",
          rec.get("blind_check", {}).get("markers") == digest["blind_markers"]
          and len(digest["blind_markers"]) >= 3 and rec["blind_check"]["found"] == [], rec.get("blind_check"))
    led = [json.loads(x) for x in paths.ledger.read_text().splitlines()]
    check("  its tests are logged (L10, L12) and card X11 is recorded",
          sorted({e.get("lever") for e in led if e["event"] == "da_test_proposed"}) == ["L10", "L12"]
          and any(e["event"] == "da_merged" and e["moves_claims"] for e in led))
    before = pda.read_bytes()
    r.st["da"]["status"] = "submitted"
    r._merge_da(job)
    led = [json.loads(x) for x in paths.ledger.read_text().splitlines()]
    check("the prospective record is written once", pda.read_bytes() == before
          and any(e["event"] == "da_not_recorded" for e in led))
    why, ids = r._da_wanted(ev, prop)
    check("C1 is not asked again; C2, open and unanswered, gets its own pass", ids == ["C2"] and why, (why, ids))
    r._stage_da(ev, why, ids, prop)
    da2 = r.st["da"]
    dg2 = jload(paths.plan_input("funnelcamp", da2["n"]))
    check("  the second pass's digest holds C2 alone", da2["status"] == "staged" and da2["claim_ids"] == ["C2"]
          and dg2["claim_ids"] == ["C2"] and da2["n"] == da["n"] + 1, da2)
    da2["status"] = "submitted"
    r._merge_da(dict(job, digest_sha256=da2["digest_sha256"]))
    reg2 = jload(paths.claims)
    check("  a reply answering C1 to the pass staged for C2 is invalid and moves nothing",
          r.st["da"]["status"] == "invalid" and "C1" in (r.st["da"].get("invalid_reason") or "")
          and [c["status"] for c in reg2["claims"]] == ["challenged", "open"], (r.st["da"], reg2["claims"]))
    why3, ids3 = r._da_wanted(ev, prop)
    check("  no third pass: every negative claim had its pass (an invalid one is recorded, not retried)",
          why3 == "" and ids3 == [], (why3, ids3))
    note = r._complete_claims_note()
    check("the COMPLETE note names the claims not concluded and card X12",
          "C1 challenged" in note and "C2 open" in note and "X12" in note, note)
    test_da_wait(CP, r, ev, prop)
    test_da_scored(CP, paths, pos)
    r2 = CP._Run("funnelcamp2", {"enabled": True}, CP.Paths(TMP / "lab2"), CP._SshBudget(None),
                 lambda: 1790000000.0, CP._Log(), None, {}, None, None, None)
    r2.st = CP._blank_state("funnelcamp2", r2.cfg)
    r2.st["brain"] = {"model_resolved": "ollama:qwen3.8:27b"}
    r2.st["da"] = {"n": 1, "status": "submitted", "claim_ids": ["C1"], "digest_sha256": "d" * 64}
    r2.paths.claims.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FF / "claims" / "claims_seed.json", r2.paths.claims)
    r2.ev = ev
    same = dict(job, model="qwen3.8:27b", model_used="qwen3.8:27b")
    r2._merge_da(same)
    c1b = [c for c in jload(r2.paths.claims)["claims"] if c["id"] == "C1"][0]
    check("a same-family adversary's reply is recorded but moves no claim",
          r2.st["da"]["status"] == "merged" and r2.st["da"]["moves_claims"] is False and c1b["status"] == "open")
    hooks = r._lab_hooks()
    check("the campaign's lab hooks: fetch, sync, verify queue",
          sorted(hooks) == ["inc_funnel_fetch", "inc_funnel_sync", "inc_verify_queue"])


# ------------------------------------------------------------------ campaign: a funnel job that failed
SEG_RE = re.compile(r'echo "INCAP_SEG (\d+)"; (.*?); echo "INCAP_SEG_END', re.S)
LIVE_FILES = ("funnel/taxonomy_cache.json", "funnel/census_v1.json", "funnel/leak_v1.json", "funnel/cards/index.json",
              "funnel/relation_geometry_v1.json", "funnel/emb_dinov2/emb_s000_of_001.npz",
              "funnel/kt7/kt7_items.jsonl", "funnel/kt7/crops_kt7.csv")


class FakeCluster(object):
    """slurm_sh for the executor: every submit gets the next job id."""

    def __init__(self, first=47300001):
        self.next_job, self.scripts, self.submits = first, [], []

    def __call__(self, script, timeout=60):
        self.scripts.append(script)
        out = ["Welcome"]
        for m in SEG_RE.finditer(script):
            i, argv = int(m.group(1)), shlex.split(m.group(2))
            args = argv[4:]
            if args and args[0] == "submit":
                self.submits.append(args)
                rec = {"verb": "submit", "ok": True, "job_id": str(self.next_job), "argv": args}
                self.next_job += 1
            else:
                rec = {"verb": args[0] if args else None, "ok": False, "error": "not on this fake cluster"}
            out += ["INCAP_SEG %d" % i, "INCAP " + json.dumps(rec), "INCAP_SEG_END %d %d" % (i, 0 if rec["ok"] else 1)]
        return {"ok": True, "stdout": "\n".join(out), "stderr": "", "returncode": 0}


def test_campaign_job_failure():
    print("campaign: a funnel job that ended FAILED (sacct) makes its step 'failed' in the lineage; the step runs "
          "again under a new id, at most funnel_job_retries times; experiment-mode waits are unchanged")
    from weed_optimizer_framework.tools.inc_autopilot import campaign as CP
    cluster = FakeCluster()
    lab = TMP / "lab_jobs"
    paths = CP.Paths(lab)
    t = [1790000000.0]
    r = CP._Run("funnelcamp", {"enabled": True}, paths, CP._SshBudget(cluster), lambda: t[0], CP._Log(), None,
                {"mongo_ok": True, "cluster_reachable": True}, None, None, None)
    r.st = CP._blank_state("funnelcamp", r.cfg)
    r.st["exp"] = "realloop_v1"
    root = world("jobs_world")
    (root / "funnel" / "files.json").write_text(json.dumps(
        {"format": "funnel-files/1", "files": {p: {"sha256": "a" * 64, "bytes": 1} for p in LIVE_FILES}}))
    inc = M.CLUSTER_INC_DIR
    # census and leak ran as L10 jobs before (2 of L10's submissions), the geometry match as L11
    for n, (lever, action, params) in enumerate((
            ("L10", "inc_funnel_audit", {"verb": "census"}), ("L10", "inc_funnel_audit", {"verb": "leak"}),
            ("L11", "inc_funnel_map", {"part": "geometry"}))):
        X._log(r.xctx, {"ts": "2026-09-28T10:0%d:00Z" % n, "campaign": "funnelcamp", "action": action, "lever": lever,
                        "params": dict(params, prereg=inc + "/funnel/prereg_v1.json", out=inc + "/funnel/"),
                        "status": "executed", "charged": True, "job_ids": [str(47200001 + n)],
                        "proposal_id": "old%d" % n, "est_su": 8.0})

    def evidence():
        return E.load_dir(root, "realloop_v1", exps=["realloop_v1"], context={"lineage": r.lineage()},
                          claims=jload(FF / "claims" / "claims_seed.json"))

    def propose_embed():
        ev = evidence()
        res = LV.propose(DG.detect(ev), ev)
        return ev, res, [p for p in res["proposals"] if p["lever"] == "L10"]

    def snapshot(squeue_ids, sacct):
        """One observation: what _observe does with a campaign-snapshot record
        before the phase machine (the jobs asked, then their states noted).
        sacct knows the earlier jobs completed."""
        r._sacct_asked = r._funnel_sacct_ids()
        sacct = dict({j: {"state": "COMPLETED"} for j in ("47200001", "47200002", "47200003")}, **sacct)
        status = {"squeue": {"ok": True, "jobs": [{"id": j, "name": "inc_funnel_x", "state": "RUNNING"}
                                                  for j in squeue_ids]},
                  "experiments": [{"exp": "realloop_v1", "done": True, "report": "current"}]}
        payload = {"verb": "campaign-snapshot", "experiments": {}, "status": status,
                   "sacct": {"ok": True, "jobs": sacct}}
        r._note_job_states(payload, status)
        return payload, status

    diagnosed = []
    r._diagnose = lambda ev: diagnosed.append(ev)

    def run_step(state):
        """The step proposed, filtered, driven through the executor, waited for,
        and its job ended in `state`. Returns (proposal id, job id)."""
        ev, res, l10 = propose_embed()
        cands, stop = r._filter(l10, [])
        if stop or not cands:
            return None, (stop, [x["reason"] for x in res["deferred"] if x["lever"] == "L10"])
        r._new_item(cands[0], [])
        pid = r.st["item"]["proposal"]["id"]
        r.ssh = r.xctx.slurm_sh = CP._SshBudget(cluster)        # a new tick: its one ssh
        r._drive_item()
        jid = (r.st.get("wait_jobs") or [None])[0]
        payload, status = snapshot([jid], {jid: {"state": "RUNNING"}})
        r._phase_machine(ev, payload, status)
        still = r.st["phase"] == "WAIT_JOB"
        t[0] += 600
        payload, status = snapshot([], {jid: {"state": state}})
        r._phase_machine(evidence(), payload, status)
        return pid, (jid, still)

    ev, res, l10 = propose_embed()
    check("the next funnel step is embed-judges, KT7 on the cluster: proposed with no wait",
          [p["argv"][2] for p in l10] == ["embed-judges"] and not l10[0].get("waits_for"), [p["argv"] for p in l10])
    pid1, (jid1, still) = run_step("FAILED")
    check("it ran as a job (R2, direct) and the campaign waited for it while squeue listed it",
          jid1 == "47300001" and still and cluster.submits, (jid1, still))
    led = [json.loads(x) for x in paths.ledger.read_text().splitlines()]
    fin = [e for e in led if e["event"] == "job_finished"]
    log1 = "%s/funnel/logs/inc_funnel_embed-judges_%s.out" % (inc, jid1)
    check("its job left squeue ended FAILED: job_finished records each job's final state, the step and the log",
          fin and fin[-1].get("job_states") == {jid1: "FAILED"} and fin[-1].get("failed") is True
          and fin[-1].get("params") == {"verb": "embed-judges"} and fin[-1].get("log") == log1
          and "at most 2 times" in fin[-1]["reasons"][0], fin[-1:])
    check("  the state records it (funnel_jobs), with the log path run_inc_funnel.sh writes (remote._job_name)",
          (r.st["funnel_jobs"].get(jid1) or {}).get("state") == "FAILED"
          and r.st["funnel_jobs"][jid1]["log"] == log1
          and RM._job_name({"builder": "funnel", "command": "embed-judges", "params": {}}) == "inc_funnel_embed-judges",
          r.st.get("funnel_jobs"))
    check("  the earlier funnel jobs' final states were read on the way (census, leak, the geometry match: "
          "COMPLETED), and their lineage records stay executed",
          all(r.st["funnel_jobs"][j]["state"] == "COMPLETED" for j in ("47200001", "47200002", "47200003"))
          and [x["status"] for x in r.lineage()][:3] == ["executed"] * 3, r.st.get("funnel_jobs"))
    check("  WAIT_JOB is over: DIAGNOSE (the next observation diagnoses)", r.st["phase"] == "DIAGNOSE"
          and "wait_funnel" not in r.st, r.st["phase"])
    lin = [x for x in r.lineage() if (x.get("params") or {}).get("verb") == "embed-judges"]
    check("the lineage record of that run is 'failed', with the job's ids, states and log (levers.py ignores it)",
          len(lin) == 1 and lin[0]["status"] == "failed"
          and lin[0]["job"] == {"ids": [jid1], "states": {jid1: "FAILED"}, "log": log1}, lin)
    check("L10 ran 3 times in this campaign (census, leak, embed-judges): the per-lever count would stop it",
          X._lever_count(r.xctx, "funnelcamp", "L10") == 3 >= X.MAX_LEVER_SUBMISSIONS)
    pid2, (jid2, _s) = run_step("FAILED")
    check("the step is proposed again, passes the stop-loss (counted per step: embed-judges ran once) and runs "
          "under a new proposal id", pid2 and pid2 != pid1 and jid2 == "47300002", (pid1, pid2, jid2))
    pid3, (jid3, _s) = run_step("OUT_OF_MEMORY")
    check("  a third run after the second failure (2 retries), again a new id", pid3 not in (None, pid1, pid2)
          and jid3 == "47300003", (pid3, jid3))
    pid4, why = run_step("FAILED")
    why_txt = " ".join(why[1] or []) if isinstance(why, tuple) and len(why) > 1 else str(why)
    check("the third failure stays with a person: not proposed, deferred with the state and the last job's log",
          pid4 is None and "failed 3 times" in why_txt and "OUT_OF_MEMORY" in why_txt
          and "inc_funnel_embed-judges_47300003.out" in why_txt, why)
    check("  every job's final state is recorded, so the snapshot asks sacct about none",
          r._funnel_sacct_ids() == []
          and [r.st["funnel_jobs"][j]["state"] for j in (jid1, jid2, jid3)] == ["FAILED", "FAILED", "OUT_OF_MEMORY"],
          r._funnel_sacct_ids())

    # a state from before this record: the job ran, its WAIT_JOB ended unread
    r2 = CP._Run("funnelcamp", {"enabled": True}, CP.Paths(TMP / "lab_jobs2"), CP._SshBudget(None), lambda: t[0],
                 CP._Log(), None, {}, None, None, None)
    r2.st = CP._blank_state("funnelcamp", r2.cfg)
    X._log(r2.xctx, {"ts": "2026-09-28T12:00:00Z", "campaign": "funnelcamp", "action": "inc_funnel_audit",
                     "lever": "L10", "status": "executed", "charged": True, "job_ids": ["46999999"],
                     "params": {"verb": "embed-judges", "prereg": inc + "/funnel/prereg_v1.json", "out": inc + "/funnel/"},
                     "proposal_id": "live1", "est_su": 8.0})
    check("a live campaign whose job ended before states were recorded: the next snapshot asks sacct about it",
          r2._funnel_sacct_ids() == ["46999999"] and r2.lineage()[0]["status"] == "executed", r2._funnel_sacct_ids())
    r2._sacct_asked = r2._funnel_sacct_ids()
    r2._note_job_states({"sacct": {"ok": True, "jobs": {"46999999": {"state": "FAILED"}}}},
                        {"squeue": {"ok": True, "jobs": []}})
    check("  sacct says FAILED: its lineage record turns 'failed' (the step may run again)",
          r2.lineage()[0]["status"] == "failed" and r2._funnel_sacct_ids() == [], r2.lineage())
    X._log(r2.xctx, {"ts": "2026-09-28T13:00:00Z", "campaign": "funnelcamp", "action": "inc_funnel_audit",
                     "lever": "L10", "status": "executed", "charged": True, "job_ids": ["47000001"],
                     "params": {"verb": "qualify", "prereg": inc + "/funnel/prereg_v1.json", "out": inc + "/funnel/"},
                     "proposal_id": "live2", "est_su": 8.0})
    for n in range(CP.SACCT_TRIES):
        r2._sacct_asked = r2._funnel_sacct_ids()
        r2._note_job_states({"sacct": {"ok": True, "jobs": {}}}, {"squeue": {"ok": True, "jobs": []}})
        if n == 0:
            check("  a job sacct does not show ended is asked again", r2._funnel_sacct_ids() == ["47000001"])
    check("  ... and after %d snapshots it is recorded UNKNOWN: its step stays with a person, as before"
          % CP.SACCT_TRIES, r2.st["funnel_jobs"]["47000001"]["state"] == CP.JOB_UNKNOWN
          and [x["status"] for x in r2.lineage()] == ["failed", "executed"], r2.st["funnel_jobs"])
    check("job_state reads sacct's states: 'CANCELLED by <uid>', array tasks, and nothing until the job ended",
          CP.job_state({"1": {"state": "CANCELLED by 512"}}, "1") == "CANCELLED"
          and CP.job_state({"2_0": {"state": "COMPLETED"}, "2_1": {"state": "TIMEOUT"}}, "2") == "TIMEOUT"
          and CP.job_state({"3_0": {"state": "COMPLETED"}, "3_1": {"state": "COMPLETED"}}, "3") == "COMPLETED"
          and CP.job_state({"4": {"state": "RUNNING"}}, "4") is None and CP.job_state({}, "5") is None)

    # experiment mode: a non-funnel job's wait is what it was
    r3 = CP._Run("expcamp", {"enabled": True}, CP.Paths(TMP / "lab_jobs3"), CP._SshBudget(None), lambda: t[0],
                 CP._Log(), None, {}, None, None, None)
    r3.st = CP._blank_state("expcamp", r3.cfg)
    r3.st["exp"] = "pilot_v1"
    r3._diagnose = lambda ev: None
    item = {"proposal": {"lever": "L4", "policy_action": "inc_label_audit", "id": "a4", "params": {}},
            "key": "k", "attempt": 0}
    r3._on_executed(item, {"status": "executed", "job_ids": ["7001"], "params": {}})
    check("an L4 audit job: WAIT_JOB as before, no funnel wait, and its snapshot asks sacct nothing",
          r3.st["phase"] == "WAIT_JOB" and "wait_funnel" not in r3.st and r3._funnel_sacct_ids() == [], r3.st)
    r3._sacct_asked = []
    r3._phase_machine(ev, {"experiments": {}}, {"squeue": {"ok": True, "jobs": []}, "experiments": []})
    fin3 = [json.loads(x) for x in CP.Paths(TMP / "lab_jobs3").ledger.read_text().splitlines()
            if json.loads(x)["event"] == "job_finished"]
    check("  and its job_finished entry is the one it always was (its job ids; no job state, step or log)",
          fin3 and fin3[-1]["job_ids"] == ["7001"]
          and not {"job_states", "failed", "log", "lever", "params", "reasons"} & set(fin3[-1])
          and "funnel_jobs" not in r3.st and r3.st["phase"] == "RUN", fin3)


def test_snapshot_sacct():
    print("remote campaign-snapshot --sacct and executor.campaign_snapshot's sacct")
    script = TMP / "fake_sacct.sh"
    script.write_text("#!/bin/sh\necho 'JobIDRaw|JobName|State|Elapsed|AllocTRES|NodeList'\n"
                      "echo '47300001|inc_funnel_embed-judges|FAILED|01:02:03|billing=5,gres/gpu:v100-32=1|v001'\n"
                      "echo '47300002|inc_funnel_qualify|CANCELLED by 512|00:00:03||None'\n")
    script.chmod(0o755)
    old = os.environ.get("INCAP_SACCT")
    os.environ["INCAP_SACCT"] = str(script)
    try:
        js = RM.job_states(["47300001", "47300002"])
        check("remote.job_states reads sacct (stream_remote.sacct): each job's state, no log for a funnel job",
              js.get("ok") and js["jobs"]["47300001"]["state"] == "FAILED"
              and js["jobs"]["47300002"]["state"].startswith("CANCELLED")
              and not any("refusal" in v for v in js["jobs"].values()), js)
        (INC / "snapx").mkdir(parents=True, exist_ok=True)
        rec = RM.dispatch(["campaign-snapshot", "--exp", "snapx", "--no-step1", "--sacct", "47300001"])
        check("campaign-snapshot --sacct ships the record under 'sacct' (the snapshot's ok unchanged by it)",
              rec.get("verb") == "campaign-snapshot" and (rec.get("sacct") or {}).get("jobs", {}).get("47300001", {})
              .get("state") == "FAILED", {k: rec.get(k) for k in ("sacct", "ok")})
        rec0 = RM.dispatch(["campaign-snapshot", "--exp", "snapx", "--no-step1"])
        check("  without --sacct there is no such key", "sacct" not in rec0, sorted(rec0))
        raised = raises(lambda: RM.dispatch(["campaign-snapshot", "--exp", "snapx", "--sacct", "1;rm"]), RM._ArgError)
        check("  a job id that is not one is refused", raised)
    finally:
        if old is None:
            os.environ.pop("INCAP_SACCT", None)
        else:
            os.environ["INCAP_SACCT"] = old
    seen = []

    def sh(script_text, timeout=60):
        seen.append(script_text)
        return {"ok": True, "stdout": "", "stderr": "", "returncode": 0}
    ctx = X.Context(slurm_sh=sh, lab_repo=str(TMP / "lab_snap"), resources={"mongo_ok": True, "cluster_reachable": True})
    X.campaign_snapshot(["realloop_v1"], ctx=ctx, funnel=True)
    X.campaign_snapshot(["realloop_v1"], ctx=ctx, funnel=True, sacct=["47300001", "47300002_1"])
    check("executor.campaign_snapshot: no --sacct when none is asked (the experiment-mode command as it was), "
          "one --sacct per job otherwise", len(seen) == 2 and "--sacct" not in seen[0]
          and "--funnel --sacct 47300001 --sacct 47300002_1" in seen[1], [s[-200:] for s in seen])
    bad = X.campaign_snapshot(["realloop_v1"], ctx=ctx, sacct=["1;rm -rf /"])
    check("  a job id that is not one is refused before anything runs", bad["status"] == "refused" and len(seen) == 2,
          bad.get("reasons"))


def lab_tree(paths):
    """The lab tree a campaign reads beside its INC copy: the pre-registration
    in its funnel/ and the contract in docs/ (runner 1.5)."""
    (paths.lab_inc / "funnel").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PREREG, paths.lab_inc / "funnel" / "prereg_v1.json")
    paths.contract.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONTRACT, paths.contract)


def test_da_wait(CP, r, ev, prop):
    """Contract 8.6: the DA pass runs before a COMPLETE card that carries a
    negative claim. COMPLETE neither submits nor pulls, so DIAGNOSE with
    nothing to propose and a pass in flight waits in BRAIN_WAIT."""
    print("campaign: a DA pass in flight holds the COMPLETE card")
    st = r.st
    keep = copy.deepcopy(st)
    st["phase"], st["item"] = "DIAGNOSE", None
    st["da"] = dict(st["da"], status="staged", staged_utc="2026-09-28T00:00:00Z")
    real_filter = r._filter
    r._filter = lambda props, diags: ([], None)
    r.status = {}
    try:
        r._diagnose(ev)
    finally:
        r._filter = real_filter
    check("nothing to propose and the pass staged: BRAIN_WAIT (da_wait), not COMPLETE",
          st["phase"] == "BRAIN_WAIT" and st.get("da_wait") is True, (st["phase"], (st.get("card") or {}).get("title")))
    st["da"]["status"] = "submitted"
    r._brain_wait_step()
    check("  it keeps waiting while the pass is submitted", st["phase"] == "BRAIN_WAIT")
    st["da"]["status"] = "merged"
    r._brain_wait_step()
    check("  once the pass has ended it diagnoses again (the next claim's pass, or the COMPLETE card)",
          st["phase"] == "DIAGNOSE" and "da_wait" not in st, st["phase"])
    st["da"] = dict(st["da"], status="staged", staged_utc="2000-01-01T00:00:00Z")
    st["phase"], st["da_wait"] = "BRAIN_WAIT", True
    r._brain_wait_step()
    check("  a pass staged and never submitted within the plan timeout is given up, then DIAGNOSE",
          st["da"]["status"] == "timeout" and st["phase"] == "DIAGNOSE", (st["da"]["status"], st["phase"]))
    r.st = keep


def test_da_scored(CP, paths, pos):
    """The prospective DA record is scored (outcome.record_da) once a valid
    audit of this Step 1 is in the evidence, and once only."""
    print("campaign: the DA record scored when the audit lands")
    root = world("campaign_scored")
    shutil.copyfile(FF / "audits" / "r10_audit.json", root / "funnel" / "audit_v1.json")
    r = CP._Run("funnelcamp", {"enabled": True}, paths, CP._SshBudget(None), lambda: 1790000000.0,
                CP._Log(), None, {}, None, None, None)
    r.st = CP._blank_state("funnelcamp", r.cfg)
    r.st["exp"] = "realloop_v1"
    ev = E.load_dir(root, "realloop_v1", exps=["realloop_v1"], claims=jload(paths.claims))
    diags = DG.detect(ev)
    prop = LV.propose(diags, ev)
    r._funnel_step(ev, diags, prop)            # DIAGNOSE's funnel part scores the record
    r._funnel_step(ev, diags, prop)
    events = [e for e in O.read_events(paths.track_events) if e.get("kind") == "da_outcome"]
    aud = jload(FF / "audits" / "r10_audit.json")
    top = sorted(pos["stage_forecast"], key=lambda k: (-pos["stage_forecast"][k], k))[0]
    check("one da_outcome enters the adversary's track record, H11 decided on the valid audit",
          len(events) == 1 and events[0]["proposed_by"] == "tier2:adversary/glm-4.7-flash"
          and events[0]["h11"]["verdict"] == ("supported" if top == aud["stage_ranking"][0]["stage"] else "falsified")
          and events[0]["h11"]["stages"] == 7 and events[0]["scored"] >= 1, events)
    inv = world("campaign_scored_invalid")
    shutil.copyfile(FF / "audits" / "r12_audit.json", inv / "funnel" / "audit_v1.json")
    r.st["da_scored"] = None
    r._score_da(E.load_dir(inv, "realloop_v1", exps=["realloop_v1"], claims=jload(paths.claims)))
    check("  an invalid audit scores nothing", len([e for e in O.read_events(paths.track_events)
                                                 if e.get("kind") == "da_outcome"]) == 1)


def test_plan_submit_script(args, raw):
    """The cluster side of inc_plan_submit, run here against a fake sbatch:
    PLAN_ROLE=adversary reaches the job's --export."""
    repo = TMP / "clusterrepo"
    plans = repo / "results" / "framework" / "inc" / "_campaign" / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    fake = TMP / "fakebin"
    fake.mkdir(exist_ok=True)
    log = TMP / "sbatch_args.txt"
    (fake / "sbatch").write_text("#!/bin/sh\nprintf '%%s\\n' \"$@\" > %s\necho 4242\n" % shlex.quote(str(log)))
    (fake / "sbatch").chmod(0o755)
    script = args[3]
    inp, out = plans / "c" / "1.input.json", plans / "c" / "1.json"
    argv = [sys.executable, "-c", script, str(inp), str(out)] + args[6:9] + [str(repo)] + args[10:]
    env = dict(os.environ, PATH="%s:%s" % (fake, os.environ.get("PATH", "")))
    p = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
    rec = json.loads(p.stdout.strip().split("INCAP ", 1)[1]) if "INCAP " in p.stdout else {}
    got = log.read_text().splitlines() if log.is_file() else []
    exp = [x for x in got if x.startswith("--export=")]
    check("the submit script writes the staged bytes and sbatches run_inc_plan.sh with PLAN_ROLE=adversary",
          p.returncode == 0 and rec.get("ok") and rec.get("job_id") == "4242" and inp.read_bytes() == raw
          and exp and exp[0].endswith(",PLAN_ROLE=adversary") and got[-1] == "weed_llm_benchmark/run_inc_plan.sh"
          and "--job-name=%s" % args[8] in got, (p.returncode, p.stdout[-400:], p.stderr[-400:], got))
    bad = argv[:-1] + ["planner2"]
    p2 = subprocess.run(bad, capture_output=True, text=True, env=env, timeout=60)
    check("  a role other than adversary is refused", p2.returncode != 0 and "is not adversary" in p2.stdout)


# ------------------------------------------------------------------ fixtures and the job script
def test_fixture_manifest():
    print("the funnel fixtures' MANIFEST.json pins")
    man = jload(FF / "MANIFEST.json")
    bad = []
    for sect in ("files", "derived", "synthetic"):
        for rel, row in sorted(man[sect].items()):
            p = FF / rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"] \
                    or p.stat().st_size != row["bytes"]:
                bad.append(rel)
    check("every pinned fixture has its sha256 and size (%d files)"
          % sum(len(man[s]) for s in ("files", "derived", "synthetic")), not bad, bad)
    stale = [rel for rel, row in man["files"].items()
             if (ROOT / row["from"]).is_file() and hashlib.sha256((ROOT / row["from"]).read_bytes()).hexdigest()
             != row["sha256"]]
    check("every byte copy still equals its source under results/ (where present)", not stale, stale)
    pinned = {str(p.relative_to(FF)) for p in FF.rglob("*") if p.is_file()} - {"MANIFEST.json"}
    listed = set(man["files"]) | set(man["derived"]) | set(man["synthetic"])
    check("no fixture file is unpinned", pinned == listed, sorted(pinned ^ listed))
    check("no fixture is an image sheet", not [p for p in pinned if p.endswith(".jpg")])
    p = subprocess.run([sys.executable, str(TESTS / "funnel_ap_fixtures.py"), "--check"], capture_output=True,
                       text=True, cwd=str(ROOT), timeout=600)
    check("funnel_ap_fixtures.py --check rebuilds every fixture byte for byte", p.returncode == 0,
          (p.stdout[-600:], p.stderr[-600:]))


def _heredoc(text, opener):
    i = text.index(opener)
    body = text[text.index("\n", i) + 1:]
    return body[:body.index("\nPY\n")]


def test_plan_script():
    print("run_inc_plan.sh: PLAN_ROLE")
    path = ROOT / "run_inc_plan.sh"
    text = path.read_text()
    check("bash -n accepts it", subprocess.run(["bash", "-n", str(path)]).returncode == 0)
    check("PLAN_ROLE is planner or adversary, the reply schema follows it, brain_plan run gets --role",
          'ROLE="${PLAN_ROLE:-planner}"' in text and "planner|adversary) ;;" in text
          and '[ "$ROLE" = "adversary" ] && REPLY_SCHEMA="inc-da-reply/1"' in text
          and 'brain_plan run --role "$ROLE"' in text)
    resolve = _heredoc(text, "ROLE=\"$ROLE\" python3 - <<'PY'")
    env = dict(os.environ, PYTHONPATH=str(ROOT))

    def pick(role, tags):
        t = TMP / ("tags_%s.json" % role)
        t.write_text(json.dumps({"models": [{"name": n} for n in tags]}))
        p = subprocess.run([sys.executable, "-c", resolve], capture_output=True, text=True,
                           env=dict(env, TAGS=str(t), ROLE=role), timeout=60)
        return p.stdout.strip()
    check("the adversary takes glm-4.7-flash when the store has it",
          pick("adversary", ["glm-4.7-flash:latest", "qwen3.8:27b", "gemma4:latest"]) == "glm-4.7-flash")
    check("  and its fallback gemma4 otherwise", pick("adversary", ["qwen3.8:27b", "gemma4:latest"]) == "gemma4")
    check("the planner still takes qwen3.8:27b", pick("planner", ["glm-4.7-flash:latest", "qwen3.8:27b"])
          == "qwen3.8:27b")
    fail_py = _heredoc(text, "REPLY_SCHEMA=\"$REPLY_SCHEMA\" python3 - <<'PY'")
    inp, out = TMP / "plan.input.json", TMP / "plan.json"
    inp.write_text(json.dumps({"campaign": "c", "n": 1, "sha256": "e" * 64}))
    p = subprocess.run([sys.executable, "-c", fail_py], capture_output=True, text=True, timeout=60,
                       env=dict(env, PLAN_FAIL_REASON="x", INPUT=str(inp), OUTPUT=str(out), MODEL="",
                                ROLE="adversary", REPLY_SCHEMA="inc-da-reply/1"))
    rep = jload(out) if out.is_file() else {}
    check("a failed adversary job leaves an inc-da-reply/1 with reply null",
          p.returncode == 0 and rep.get("schema") == "inc-da-reply/1" and "reply" in rep and rep["reply"] is None
          and rep.get("digest_sha256") == "e" * 64 and BP.collect(rep, "2026-09-28T00:00:00Z", role="adversary",
                                                                  digest_sha256="e" * 64)["status"] != "pending",
          (p.stderr[-300:], rep))


def main():
    try:
        for t in (test_thresholds_checked_against, test_mirrors, test_model_router, test_panel_and_outcome,
                  test_h11_and_score_da, test_remote_dev_scores, test_remote_summary, test_remote_submit,
                  test_executor_render, test_lab_hooks, test_campaign_da, test_campaign_job_failure,
                  test_snapshot_sacct, test_fixture_manifest, test_plan_script):
            try:
                t()
            except Exception as e:
                import traceback
                traceback.print_exc()
                check("%s ran to its end" % t.__name__, False, "%s: %s" % (type(e).__name__, e))
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
