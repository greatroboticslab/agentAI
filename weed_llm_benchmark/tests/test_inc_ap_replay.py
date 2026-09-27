#!/usr/bin/env python3
"""Replay of the four human INC interventions, negative controls, test
blindness and the prospective case (docs/INC_AUTOPILOT.md, (f)).

Cases:
  R1   pilot_v1: D1 fires with cites (20/21 P_recipe 0.0, 21/21 null < inc,
       9/9 non-ACCEPT on I2, I3, I5); the one proposal of lever L1 is the
       manual command 'inc.pilot build --exp pilot_v2 --replay-mode full';
       X1 appears only as an R4 card; no realloop is proposed.
  R2   Step 1 without relevance.json: D2 names fvossel__csgo_player_detection
       and rf_bishwarup-halder__crop-health-advisor; L3, then L2 with
       --relevance. With relevance.json: D2 silent, D2b on the tomato-leaf
       source if it passes, X2 as a card. SKIPPED until the Step 1 files are
       pulled into tests/fixtures/inc_replay/step1/ (see its MANIFEST.json).
  R3   pilot_v1: D3 fires citing exp.steps[2].planted,
       chains.*.bswap.attributed_to_labels = false and attribution_not_run;
       the L4 command is run_inc_audit.sh's documented one; D3b on s05_Breal
       with X4 as a card.
  R4a  pilot_v1: no_recipe_tracks_truth (0.4286 < 5/7); no L2 proposal.
  R4b  prospective_d4(report) writes D4's decision once (a second, different
       decision refuses); a ready report on which D1 also fires records
       ready false, blocked by D1; on pilot_v2's report (pinned since the
       fixtures took it) D4 is silent: D1 fires in full replay (the recipe
       forgets, X1), so no recipe is chosen. The committed record
       (model.REPLAY_DIR/prospective_4.json) must reproduce, and must predate
       the person's real-loop build (its exp.json initialised_utc), against
       which it is compared. SKIPPED while nothing is committed.
  R5   pilot_v2 (full rehearsal, gate v1): D15 fires, citing the five
       (chain, step) pairs, pooled over the chains, of clean truth-helps
       steps the flips guard alone REJECTed at P_data 1.00 (freeze I2, I3;
       full I2, I3; lora I3: ledger lines 15, 22, 13, 19, 23); three
       truth-helps misses failed another guard too (freeze I5 on regression
       and species, line 31; lora I2 and I5 on species, lines 18 and 32), so
       D15 does not explain every miss and D1 is NOT blocked: it fires with X1
       (full replay), a card; the one build is L9, 'inc.pilot build --exp
       pilot_v3 --replay-mode full --gate-flips-mode net', ranked first; D4
       proposes no real loop. Written after the evidence was in view
       (docs/INCREMENTAL_PROTOCOL.md, Protocol v2), so it is a reproduction
       test, like R1 and R3.
  R6   pilot_v3 (full rehearsal, gate v2 net; pinned 2026-09-27 after its
       result): D16 supported (v2 11/21, its v1 counterfactual 9/21); D1
       fires (18/21 P_recipe <= 0.25, 19/21 null < inc, 3/9 truth-helps
       steps not ACCEPT: freeze I3, I5, lora I2) with X1 as a card, but does
       NOT block D4 (blocks_d4 false): the chain D4 selects, full (5/7, by
       rate), ACCEPTed every clean truth-helps step (I2, I3, I5); D4 is
       ready with recipes ['full'], replay mode full and gate net, and the
       prospective record says so under the current rules version. With no
       relevance.json (the pinned fixtures have no Step 1) the L2 waits for
       L3 first; with one made for the select build in view (the pinned
       step1 files when pulled, else a synthetic select_summary.json and a
       matching relevance.json) the L2 argv is exactly 'inc.realloop build
       --exp realloop_v1 --base <INC_DIR>/step1/base_B.jsonl --replay-mode
       full --recipes full --relevance <INC_DIR>/step1/relevance.json
       --gate-flips-mode net', which realloop's own argparse reads back.
       The D1 refinement was decided at an R4 review after pilot_v3's result
       (docs/INC_AUTOPILOT.md, D1), so R6 is a reproduction test; the real
       loop's truth arm is its prospective test. R1 and R5 are unchanged
       under it (every chain misses on pilot_v1; D4's selected chain full
       misses I2, I3 on pilot_v2).
  R7   pilot_v3 with a Step 1 whose relevance.json is made for the select
       build but failed its own calibration check (the pinned synthetic
       tests/fixtures/inc_replay/synthetic/step1_calibration_failed/, with
       the cluster build's calibration numbers: tau 0.0557 < 0.5, 34.5% of
       train_core below 0.5; relevance.load refuses it as a calibration
       failure, not as malformed): D2 fires on the unevidenced sources but
       does not escalate (warn, card X9, no OP_ESCALATE, no L3); its detail
       records the criterion (source-level species evidence), why, and the
       capacity it rests on (5 evidenced sources, 3,213 images >= 2,751 for
       N 6 x M 393); D4 is R6's; the one L2 argv is exactly 'inc.realloop
       build --exp realloop_v1 --base <INC_DIR>/step1/base_B.jsonl
       --replay-mode full --recipes full --increment-sources evidence
       --gate-flips-mode net', which realloop's own argparse reads back,
       select.load_evidence finds the same evidenced sources, the executor
       renders it within the policy row's bounds, and it follows D4's READY
       record (prospective_guard); the same under the test-blindness
       perturbation. D2's then is not also deferred (D4's L2 carries it).
       D2 fires on the file whatever the per-source evidence says: without
       the per-source aggregate it still names the evidence L2 (warn, X9);
       without admit_summary.json it escalates (OP_ESCALATE and X9). With a
       passing relevance.json the L2 is R6's --relevance command. A stale or
       malformed relevance.json (ok false at tau >= 0.5, ok true below it, no
       check tau_min, params.tau_min 0.3, another format) still escalates,
       and relevance.load refuses each malformed one as malformed. R7's
       Step 1 carrying the real Step 1's numbers (1,439 images < 2,751) gives
       R8's sized L2 (below), not an escalation. Written with the
       criterion's design in view, so it is a reproduction test.
  R8   pilot_v3 with the real Step 1 (byte copies of the local
       results/framework/inc/step1/select_summary.json and admit_summary.json,
       pinned in tests/fixtures/inc_replay/step1_copies/) and a relevance.json
       made for that select build whose calibration check failed (the pinned
       synthetic tests/fixtures/inc_replay/synthetic/step1_real_calibration_failed/,
       the cluster build's calibration numbers): the evidenced pool (5
       sources, 1,439 images) cannot hold the default loop (N 6 x M 393 needs
       2,751), and D2 does NOT escalate: the R4 sizing rule of 2026-09-27
       (levers.evidence_sizing) gives N = 4 (six decided increments with
       UNVERIFIED and OTHER_HEAVY) and M = min(393, floor(1439 / 5)) = 287
       (7.3% of B, above the 5% floor); warn, card X9, no OP_ESCALATE, no L3;
       D2's detail records the default and the sized N and M and the rule.
       The one L2 argv is exactly 'inc.realloop build --exp realloop_v1 --base
       <INC_DIR>/step1/base_B.jsonl --replay-mode full --recipes full
       --increment-sources evidence --size 287 --n-verified 4
       --gate-flips-mode net' (the L2 template's order), which realloop's own
       argparse reads back; the executor re-renders it inside the policy
       row's and L2's bounds; validate.materialise gives a brain L2 without
       --size and --n-verified the same request; it follows D4's READY
       record; the same under the test-blindness perturbation. An evidenced
       pool of 980 images (M 196 < 196.35 = 5% of B) escalates with the
       numbers and no L2; 985 images (M 197) is sized; 2,000 images renders
       only --n-verified 4; no admit_summary.json escalates. The rule was
       decided with the real Step 1's numbers in view, so R8 is a
       reproduction test; the sized loop's truth arm is its prospective test.
  Lineage (the lever already ran): pilot_v1 with a full-replay pilot_v2
       built after it, looking at either: no L1 again (no pilot_v3); a
       ready pilot whose real loop (its replay mode and recipe) is built,
       looking at the loop: no second real loop; b0_v1 in the fixture tree
       as it stood at intervention 1 (pilot_v1, b0_v1): D4 is campaign-wide,
       so it proposes pilot_v1's L1 once, and nothing once pilot_v2 exists
       (a stub, and the whole pinned tree with the real pilot_v2).
  Negative controls: b0_v1 and base_b_v1 alone, and a synthetic healthy
       pilot driven to done on FakeBackend by the real driver and report
       (clean increments help and every chain accepts them, the planted
       relabelling is rejected and attributed to labels): D1, D2, D3, D3b,
       D15 and D16 stay silent; D4 reaches its positive outcome,
       decision_slot_ready (not no_recipe_tracks_truth), and its L2 waits for
       Step 1 (deferred). pilot_v1: D15 silent (no truth-helps step failed on
       the flips guard alone).
  Test blindness: each fixture (pilot_v1, pilot_v2 in the whole tree,
       pilot_v3 in the tree with it, b0_v1, the synthetic pilot) with
       every value under a non-dev exam perturbed (the known ones, anywhere,
       and every exam but dev under an "exams" dict) and an exam the lists
       here do not know (ood24) added to every final row gives
       byte-identical diagnoses, proposals (ids and timestamps aside), argv
       and canonical evidence bytes; a perturbed dev value changes them.
  Live path: remote.py's own snapshot of pilot_v1 (report.json without its
       final table, derived.report_final_dev, the ledger as numbered
       entries), read by evidence.from_snapshot, gives byte-identical
       diagnoses and proposals to the files (skipped if remote.py cannot be
       imported).
  Earliest fire: ledger prefixes of pilot_v1; reports the first gate entry
       at which D1 fires and the GPU-hours the rest of the pilot then cost
       (an estimate from the report's per-arm hours); asserts only that it
       fires by the end.
Not here: the governance case (authorize / approvals / executor), which
belongs to the executor's tests.

Honest caveat (contract (f)): R1 and R3 were written with pilot_v1 in view,
R5 with pilot_v2, R6 with pilot_v3, R7 with the failed relevance build and R8
with the real Step 1's numbers in view, so they are reproduction tests; only
R4b (and the real loop's truth arm, for R6's refinement) is evidence of
generalisation.

Run:  python3 tests/test_inc_ap_replay.py
"""
import collections
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_replay_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"            # the synthetic pilot is a testing experiment
os.environ.pop("INC_JOB_SCRIPT", None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as LV  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "inc_replay"
ROOT = pathlib.Path(__file__).resolve().parents[1]
FAILURES, SKIPS = [], []
FAILURE_IDS = ("D1", "D2", "D3", "D3b", "D15", "D16")
# The fixture tree as it stood at intervention 1 (pilot_v2 not built yet); R1-R4a replay
# the decision taken then.
V1_ERA = ["pilot_v1", "b0_v1"]
WHOLE = ["pilot_v1", "pilot_v2", "b0_v1", "base_b_v1"]
# The tree once pilot_v3 (the v2 pilot L9 built) was done: R6 reads it. R5 keeps WHOLE,
# the tree as it stood when pilot_v2 was diagnosed.
WHOLE_V3 = WHOLE + ["pilot_v3"]
L9_ARGV = ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build", "--exp", "pilot_v3",
           "--replay-mode", "full", "--gate-flips-mode", "net"]


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def skip(name, why):
    print("  skip %s: %s" % (name, why))
    SKIPS.append(name)


def documented(script, marker):
    """The first sbatch command in the comment block after `marker` in a job
    script's header, with the block's $VAR assignments expanded (the same
    parser as tests/test_inc_ap_levers.py)."""
    import re
    import shlex
    lines = (ROOT / script).read_text().splitlines()
    i = next(j for j, ln in enumerate(lines) if marker in ln)
    stmts, cur = [], ""
    for ln in lines[i + 1:]:
        if not ln.startswith("#"):
            break
        body = ln[1:].strip()
        if not body and not cur:
            break
        if body.endswith("\\"):
            cur += body[:-1] + " "
            continue
        stmts.append(cur + body)
        cur = ""
    env, cmd = {}, None

    def expand(x):
        return re.sub(r"\$([A-Z]+)", lambda m: env[m.group(1)], x)
    for st in stmts:
        for part in (q.strip() for q in st.split(";")):
            m = re.fullmatch(r"([A-Z]+)=(\S+)", part)
            if m:
                env[m.group(1)] = expand(m.group(2))
            elif part.startswith("sbatch ") and cmd is None:
                cmd = part
    return [expand(t) for t in shlex.split(cmd, comments=True)]


def run(ev):
    diags = DG.detect(ev)
    return diags, DG.by_id(diags), LV.propose(diags, ev)


def realloop_argvs(res):
    return [p for p in res["proposals"] if any("inc.realloop" in a for a in p["argv"])]


UNKNOWN_EXAM = "ood24"        # an exam no list in the autopilot names (added upstream later)


def perturb_non_dev(obj):
    """Every number under a non-dev exam changed: under a known non-dev exam
    key anywhere, and under every key but dev of an "exams" dict; each
    "exams" dict also gains UNKNOWN_EXAM."""
    def go(x, under, parent=None):
        if isinstance(x, dict):
            out = {k: go(v, under or k in E.NON_DEV_EXAMS or (parent == "exams" and k != "dev"), k)
                   for k, v in x.items()}
            if parent == "exams":
                out[UNKNOWN_EXAM] = {"twelve": {"mean": 0.123456789, "n": 3}}
            return out
        if isinstance(x, list):
            return [go(v, under) for v in x]
        if under and isinstance(x, (int, float)) and not isinstance(x, bool):
            return -x + 0.5 if isinstance(x, float) else x + 11
        return x
    return go(obj, False)


def snapshot_copy(src_root, exps, dst):
    """The allow-listed files of `exps` (and step1) under src_root, copied."""
    for e in exps:
        for f in E.EXP_FILES:
            p = pathlib.Path(src_root) / e / f
            if p.is_file():
                (dst / e / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst / e / f)
    for f in E.STEP1_FILES:
        p = pathlib.Path(src_root) / "step1" / f
        if p.is_file():
            (dst / "step1").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst / "step1" / f)
    return dst


# --------------------------------------------------------------- R1 .. R4a
def test_r1_r3_r4a():
    print("R1 pilot_v1: recipe_forgets -> L1 (the manual pilot_v2 command)")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    diags, by, res = run(ev)
    d1 = by["D1"]
    c = d1["detail"]["counts"]
    check("D1 fires", d1["fired"])
    check("20/21 P_recipe <= 0.25, each cited at 0.0",
          c["recipe_flagged"] == 20 and c["gate_entries"] == 21
          and sum(1 for x in d1["cites"] if x["pointer"] == "/decision/p_recipe" and x["value"] == 0.0) == 20, c)
    check("21/21 null_mean < inc, each cited",
          c["null_below_inc"] == 21 and sum(1 for x in d1["cites"] if x["pointer"] == "/decision/null_mean") == 21)
    check("9/9 non-ACCEPT on the truth-helps steps I2, I3, I5",
          (c["helps_not_accepted"], c["helps_entries"], d1["detail"]["helps_steps"]) == (9, 9, ["I2", "I3", "I5"]))
    l1 = [p for p in res["proposals"] if p["lever"] == "L1"]
    check("one L1 proposal whose argv is the manual command",
          len(l1) == 1 and l1[0]["argv"] == ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build",
                                             "--exp", "pilot_v2", "--replay-mode", "full"],
          [p["argv"] for p in l1])
    check("X1 appears only as an R4 card", not [p for p in res["proposals"] if p["lever"] == "X1"]
          and [x["risk"] for x in res["cards"] if x["lever"] == "X1"] == ["R4"])
    check("no realloop is proposed", not realloop_argvs(res)
          and not [p for p in res["proposals"] if p["lever"] in ("L2", "L5", "L6")])
    check("every cite of every fired diagnosis and proposal resolves",
          all(ev.check_cite(x) for d in diags if d["fired"] for x in d["cites"])
          and all(ev.check_cite(x) for p in res["proposals"] for x in p["cites"]))

    print("R3 pilot_v1: attribution_control_failed -> L4 (the documented audit)")
    d3 = by["D3"]
    ptrs = {(x["artifact"], x["pointer"]): x["value"] for x in d3["cites"]}
    check("D3 fires", d3["fired"])
    check("cites exp.steps[2].planted", ptrs.get(("pilot_v1/exp.json", "/steps/2/planted"))
          == "40% of boxes relabelled to another species")
    check("cites chains.*.bswap.attributed_to_labels = false",
          all(ptrs.get(("pilot_v1/report.json", "/chains/%s/bswap/attributed_to_labels" % r)) is False
              for r in ("freeze", "full", "lora")))
    check("cites attribution_not_run", any(x["pointer"] == "/attribution_not_run/label_audit" for x in d3["cites"])
          and ("pilot_v1/report.json", "/attribution_scope/not_run/4") in ptrs)
    l4 = [p for p in res["proposals"] if p["lever"] == "L4"]
    doc = documented("run_inc_audit.sh", "The pilot's post-hoc item 4")
    check("L4's argv is run_inc_audit.sh's documented command", len(l4) == 1 and l4[0]["argv"] == doc,
          (l4 and l4[0]["argv"], doc))
    d3b = by["D3b"]
    check("D3b fires on s05_Breal", d3b["fired"] and d3b["detail"]["steps"] == ["s05_Breal"])
    check("X4 appears as a card", [x["risk"] for x in res["cards"] if x["lever"] == "X4"] == ["R4"])

    print("R4a pilot_v1: no_recipe_tracks_truth, no L2")
    d4 = by["D4"]
    check("D4 is no_recipe_tracks_truth", d4["fired"] and d4["name"] == "no_recipe_tracks_truth")
    check("best rate 3/7 = 0.4286 < 5/7", d4["detail"]["ranking"][0]["agree"] == 3
          and round(d4["detail"]["ranking"][0]["rate"], 4) == 0.4286 and d4["detail"]["ready_rate"] == "5/7")
    check("no L2 proposal", not [p for p in res["proposals"] if p["lever"] == "L2"])


# ---------------------------------------------------------------- lineage
def child_pilot(name="pilot_v2", replay_mode="full", utc="2026-09-28T00:00:00Z"):
    d = json.loads((FIX / "pilot_v1" / "exp.json").read_text())
    d.update(exp=name, replay_mode=replay_mode, initialised_utc=utc)
    d["base"]["manifest"] = d["base"]["manifest"].replace("pilot_v1", name)
    for st in d["steps"]:
        st["manifest"] = st["manifest"].replace("pilot_v1", name)
    return d


def test_lineage():
    print("lineage: a lever that already ran is not proposed again")
    root = snapshot_copy(FIX, ["pilot_v1", "b0_v1"], TMP / "lineage_child")
    (root / "pilot_v2").mkdir()
    (root / "pilot_v2" / "exp.json").write_text(json.dumps(child_pilot()))
    for cur in ("pilot_v1", "pilot_v2"):
        ev = E.load_dir(root, cur)
        diags, by, res = run(ev)
        check("child already built (pilot_v2, full replay), looking at %s: no L1 and no build at all" % cur,
              not [p for p in res["proposals"] if p["lever"] in ("L1", "L2", "L5", "L6", "L8")]
              and any(x["lever"] == "L1" and "pilot_v2" in x["reason"] for x in res["deferred"]),
              ([(p["lever"], p["child_exp"]) for p in res["proposals"]], res["deferred"]))
    ev = E.load_dir(FIX, "b0_v1", exps=V1_ERA)
    diags, by, res = run(ev)
    check("b0_v1 in the intervention-1 tree: D4 is campaign-wide and proposes pilot_v1's L1 once (pilot_v2)",
          by["D4"]["fired"] and by["D4"]["exp"] == "pilot_v1"
          and [(p["lever"], p["child_exp"]) for p in res["proposals"]] == [("L1", "pilot_v2")],
          [(p["lever"], p["child_exp"]) for p in res["proposals"]])
    ev = E.load_dir(root, "b0_v1")
    diags, by, res = run(ev)
    check("... and nothing once pilot_v2 is built", not res["proposals"], res["proposals"])
    ev = E.load_dir(FIX, "b0_v1", exps=WHOLE)
    diags, by, res = run(ev)
    check("... nor in the whole pinned tree with the real pilot_v2 (D4 silent on it; L9 belongs to pilot_v2's "
          "own diagnosis)", not res["proposals"] and not by["D4"]["fired"], res["proposals"])

    rep = json.loads((FIX / "pilot_v1" / "report.json").read_text())
    rep["agreement"]["lora"] = {"agree": 6, "compared": 7, "rate": 6 / 7.0}
    for st in rep["steps"]:
        for c in (st.get("chains") or {}).values():
            if isinstance(c, dict) and "p_recipe" in c:
                c["p_recipe"] = 0.5
    sel = {"sizes": {"base_B": 3927}, "outputs": {"increment_pool.jsonl": {"sha256": "a" * 64},
                                                  "base_B.jsonl": {"sha256": "b" * 64}}}
    rel = dict(R6_RELEVANCE)
    inc = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc"
    loop = {"exp": "realloop_v1", "type": "chain", "builder": DG.REALLOOP_BUILDER, "replay_mode": "sample",
            "initialised_utc": "2026-10-01T00:00:00Z", "recipes": {"lora": {"epochs": 30}}, "truth": True,
            "increment_images": 393, "base": {"name": "base_B", "manifest": inc + "/realloop_v1/manifests/base_B.jsonl",
                                              "manifest_sha256": "b" * 64, "source_manifest": inc + "/step1/base_B.jsonl"},
            "steps": [], "step1": {"relevance": {"path": inc + "/step1/relevance.json"}}}
    texts = {"pilot_v1/exp.json": (FIX / "pilot_v1" / "exp.json").read_text(), "pilot_v1/report.json": json.dumps(rep),
             "step1/select_summary.json": json.dumps(sel), "step1/relevance.json": json.dumps(rel)}
    ev = E.from_texts(texts, "pilot_v1")
    diags, by, res = run(ev)
    check("a ready pilot (lora 6/7, D1 silent) without its loop: one L2, realloop_v1",
          [(p["lever"], p["child_exp"]) for p in res["proposals"] if p["lever"] == "L2"] == [("L2", "realloop_v1")],
          ([(p["lever"], p["child_exp"]) for p in res["proposals"]], res["deferred"]))
    texts["realloop_v1/exp.json"] = json.dumps(loop)
    for cur in ("pilot_v1", "realloop_v1"):
        ev = E.from_texts(texts, cur)
        diags, by, res = run(ev)
        check("loop already built (realloop_v1: sample, lora), looking at %s: no second real loop" % cur,
              not realloop_argvs(res)
              and any(x["lever"] == "L2" and "realloop_v1" in x["reason"] for x in res["deferred"]),
              ([p["argv"] for p in realloop_argvs(res)], res["deferred"]))


# --------------------------------------------------------------------- R5
def test_r5():
    print("R5 pilot_v2: the flips guard alone blocks truth-helps steps -> L9 (gate v2)")
    ev = E.load_dir(FIX, "pilot_v2", exps=WHOLE)
    diags, by, res = run(ev)
    d15 = by["D15"]
    pairs = sorted((p["chain"], p["step"], p["line"]) for p in d15["detail"]["pairs"])
    check("D15 fires", d15["fired"] and d15["name"] == "guard_blocks_truth_helps", d15["summary"])
    check("its cites are the five flips-only rejections, pooled over chains: freeze I2 (line 15), I3 (22); "
          "full I2 (13), I3 (19); lora I3 (23)",
          pairs == [("freeze", "I2", 15), ("freeze", "I3", 22), ("full", "I2", 13), ("full", "I3", 19),
                    ("lora", "I3", 23)]
          and {c["line"] for c in d15["cites"] if c["artifact"] == "pilot_v2/ledger.jsonl"
               and c["pointer"].startswith("/decision/")} == {8, 13, 15, 19, 22, 23}, pairs)
    by_line = {}
    for c in d15["cites"]:
        if c["artifact"] == "pilot_v2/ledger.jsonl" and c.get("line") in (13, 15, 19, 22, 23):
            by_line.setdefault(c["line"], {})[c["pointer"]] = c["value"]
    check("each of the five: REJECT at P_data 1.0, flips failed, regression and species passed",
          len(by_line) == 5 and all(v.get("/decision/verdict") == "REJECT" and v.get("/decision/p_data") == 1.0
                                    and v.get("/decision/guards/flips/passed") is False
                                    and v.get("/decision/guards/regression/passed") is True
                                    and v.get("/decision/guards/species/passed") is True
                                    for v in by_line.values()), by_line)
    check("each is on a step the truth arm says helps (truth/2, truth/4 cited as helps)",
          {(c["line"], c["value"]) for c in d15["cites"] if c["pointer"] == "/detail/verdict"}
          == {(3, "helps"), (4, "helps")})
    check("the gate is v1 (a v1 decision's config), so the lever is L9", d15["levers"] == ["L9"]
          and d15["detail"]["flips_mode"] == "negative")
    d1 = by["D1"]
    check("D1 still fires with its cites (20/21 P_recipe, 20/21 null < inc, 8/9 truth-helps not ACCEPT)",
          d1["fired"] and d1["detail"]["counts"]["helps_not_accepted"] == 8 and len(d1["cites"]) > 40,
          d1["detail"].get("counts"))
    unx = [(u["chain"], u["step"], u["line"], u["failed_guards"]) for u in d15["detail"]["unexplained"]]
    check("three truth-helps misses failed another guard too (freeze I5 line 31: regression, species; lora I2 "
          "line 18 and I5 line 32: species): D15 does not explain every miss",
          unx == [("freeze", "I5", 31, ["flips", "regression", "species"]), ("lora", "I2", 18, ["flips", "species"]),
                  ("lora", "I5", 32, ["flips", "species"])] and d15["detail"]["blocks_d1"] is False, unx)
    check("so D1 is not blocked: it fires with X1 (full replay) and names the misses D15 leaves", d1["levers"] == ["X1"]
          and "blocked_by" not in d1["detail"] and d1["detail"]["precedence"]["D15"] == "fired, does not block D1",
          d1["detail"].get("precedence"))
    builds = [p for p in res["proposals"] if p["risk"] == "R3"]
    check("the proposal is L9, with argv exactly '%s'" % " ".join(L9_ARGV),
          [(p["lever"], p["argv"]) for p in builds] == [("L9", L9_ARGV)] and res["proposals"][0]["lever"] == "L9",
          [(p["lever"], p["argv"]) for p in res["proposals"]])
    menu = LV.load_menu()
    known = set(menu["levers"]) | set(menu["cards"]) | set(menu["operations"])
    check("every deferred entry names a menu lever, card or operation",
          all(x["lever"] in known for x in res["deferred"]), res["deferred"])
    check("X1 is an R4 card, no realloop, D4 silent", [x["risk"] for x in res["cards"] if x["lever"] == "X1"] == ["R4"]
          and not realloop_argvs(res) and not by["D4"]["fired"], [c["lever"] for c in res["cards"]])
    check("every cite of every fired diagnosis and proposal resolves",
          all(ev.check_cite(x) for d in diags if d["fired"] for x in d["cites"])
          and all(ev.check_cite(x) for p in res["proposals"] for x in p["cites"]))


# --------------------------------------------------------------------- R6
R6_SELECT = {"sizes": {"base_B": 3927},
             "outputs": {"increment_pool.jsonl": {"sha256": "a" * 64}, "base_B.jsonl": {"sha256": "b" * 64}}}
# A passing file under relevance.py's rule (params MIN_CROPS 20, CAL_PERCENTILE 5, TAU_MIN 0.5;
# a consistent check: ok true, tau_min 0.5, tau 0.62 >= 0.5), as relevance.load reads it.
R6_RELEVANCE = {"format": "inc.relevance/1",
                "params": {"min_crops": 20, "calibration_percentile": 5.0, "tau_min": 0.5},
                "calibration": {"tau": 0.62, "check": {"ok": True, "tau_min": 0.5}},
                "inputs": {"increment_pool": {"sha256": "a" * 64}}, "increment_pool": {"sources": {}}}


def r6_step1(root):
    """Step 1 beside the fixtures for R6's L2: the pinned step1 files when
    select_summary.json and relevance.json are both pulled, else a synthetic
    pair (the real ones are cluster-only): base B's size and the tables'
    sha256 in the select summary, and a relevance.json made for exactly that
    build (format inc.relevance/1, a calibrated tau, a passed check)."""
    step1 = root / "step1"
    if (step1 / "select_summary.json").is_file() and (step1 / "relevance.json").is_file():
        return "pinned"
    step1.mkdir(parents=True, exist_ok=True)
    (step1 / "select_summary.json").write_text(json.dumps(R6_SELECT))
    (step1 / "relevance.json").write_text(json.dumps(R6_RELEVANCE))
    return "synthetic"


def test_r6():
    print("R6 pilot_v3: D1 fires but does not block D4 (its selected chain full ACCEPTed every truth-helps step) "
          "-> D4 ready -> L2 (full replay, recipe full, gate net)")
    ev = E.load_dir(FIX, "pilot_v3", exps=WHOLE_V3)
    diags, by, res = run(ev)
    d16 = by["D16"]
    check("D16: protocol v2 supported on pilot_v3 (v2 11/21, its v1 counterfactual 9/21)",
          d16["fired"] and d16["detail"]["outcome"] == "supported" and d16["detail"]["agree_v2"] == 11
          and d16["detail"]["agree_v1_counterfactual"] == 9 and d16["detail"]["compared"] == 21, d16["summary"])
    d1 = by["D1"]
    c = d1["detail"].get("counts") or {}
    check("D1 fires with its cites: 18/21 P_recipe <= 0.25, 19/21 null_mean < inc, 3/9 truth-helps not ACCEPT",
          d1["fired"] and (c.get("recipe_flagged"), c.get("null_below_inc"), c.get("gate_entries"),
                           c.get("helps_not_accepted"), c.get("helps_entries")) == (18, 19, 21, 3, 9)
          and sum(1 for x in d1["cites"] if x["pointer"] == "/decision/p_recipe") == 18, c)
    check("D1's misses per chain: freeze I3, I5; full none; lora I2",
          {k: v["misses"] for k, v in (d1["detail"].get("chains") or {}).items()}
          == {"freeze": ["I3", "I5"], "full": [], "lora": ["I2"]}, d1["detail"].get("chains"))
    sel = d1["detail"].get("d4_selection") or {}
    check("D1 records D4's selection: full, 5/7, ready, no miss",
          (sel.get("selected"), sel.get("rate"), sel.get("ready"), sel.get("misses")) == ("full", "5/7", True, []),
          sel)
    check("D1 does not block D4 (blocks_d4 false) and is not itself blocked",
          d1["detail"].get("blocks_d4") is False and "blocked_by" not in d1["detail"], d1["detail"].get("blocks_d4_why"))
    check("D1's lever is X1, an R4 card (full replay)", d1["levers"] == ["X1"]
          and [x["risk"] for x in res["cards"] if x["lever"] == "X1"] == ["R4"]
          and not [p for p in res["proposals"] if p["lever"] == "X1"], d1["levers"])
    check("D15 silent (no flips-only miss under the net guard)", not by["D15"]["fired"], by["D15"]["summary"])
    d4 = by["D4"]
    det = d4["detail"]
    check("D4 ready: decision_slot_ready, L2, recipes ['full'], replay mode full, gate net, v2 supported",
          d4["fired"] and d4["name"] == "decision_slot_ready" and d4["levers"] == ["L2"]
          and det.get("recipes") == ["full"] and det.get("replay_mode") == "full"
          and det.get("gate") == {"flips_mode": "net", "v2_check": "supported"} and "blocked_by" not in det,
          d4["summary"])
    check("D4's ranking: full 5/7 over freeze 3/7 and lora 3/7, decided by rate",
          det.get("rates") == {"full": "5/7", "freeze": "3/7", "lora": "3/7"} and det.get("decided_by") == "rate",
          det.get("rates"))
    check("D4 records D1 under detail.d1 (fired, not blocking, selected full)",
          (det.get("d1") or {}).get("blocks_d4") is False and (det.get("d1") or {}).get("selected") == "full",
          det.get("d1"))
    check("without relevance.json (no Step 1 in the fixtures): no real loop yet, L2 waits for L3 first",
          not realloop_argvs(res) and any(x["lever"] == "L2" and "L3 first" in x["reason"] for x in res["deferred"]),
          ([p["argv"] for p in res["proposals"]], res["deferred"]))
    check("every cite of every fired diagnosis and proposal resolves",
          all(ev.check_cite(x) for d in diags if d["fired"] for x in d["cites"])
          and all(ev.check_cite(x) for p in res["proposals"] for x in p["cites"]))

    rec = DG.prospective_d4(FIX / "pilot_v3" / "report.json", out_path=TMP / "prospective" / "r6.json",
                            defn=json.loads((FIX / "pilot_v3" / "exp.json").read_text()),
                            now="2026-09-27T16:00:00Z")
    check("the prospective record on pilot_v3: ready, full replay, recipes ['full'], gate net, v2 supported, "
          "under the current rules version",
          (rec["ready"], rec["replay_mode"], rec["recipes"], rec["gate_flips_mode"], rec["v2_check"])
          == (True, "full", ["full"], "net", "supported") and rec["rules_version"] == DG.rules_version()
          and len(rec["rules_version"]) == 12, {k: rec.get(k) for k in ("ready", "outcome", "blocked_by")})

    root = snapshot_copy(FIX, WHOLE_V3, TMP / "r6")
    how = r6_step1(root)
    ev = E.load_dir(root, "pilot_v3", exps=WHOLE_V3)
    diags, by, res = run(ev)
    inc = LV.inc_dir(ev, "pilot_v3")
    want = ["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", "realloop_v1",
            "--base", inc + "/step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
            "--relevance", inc + "/step1/relevance.json", "--gate-flips-mode", "net"]
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    check("with a relevance.json made for the select build (%s step1): one L2, argv exactly '%s'"
          % (how, " ".join(want)), len(l2) == 1 and l2[0]["argv"] == want, [p["argv"] for p in l2])
    check("  its parent is pilot_v3 and its child realloop_v1, triggered by D4",
          l2 and (l2[0]["parent_exp"], l2[0]["child_exp"], l2[0]["trigger"][0]) == ("pilot_v3", "realloop_v1", "D4"),
          l2 and (l2[0]["parent_exp"], l2[0]["child_exp"], l2[0]["trigger"]))
    from weed_optimizer_framework.tools.inc import realloop as RL
    got = {}
    real = RL.build

    def fake(*a, **k):
        got["args"], got["kwargs"] = a, k
    RL.build = fake
    try:
        rc = RL.main(want[3:])
    finally:
        RL.build = real
    k = got.get("kwargs") or {}
    check("  realloop's own argparse reads it back (base, full, full, relevance, net, truth on)",
          rc in (0, None) and got.get("args") == ("realloop_v1",)
          and (k.get("base"), k.get("replay_mode"), k.get("recipes"), k.get("relevance"), k.get("gate_flips_mode"),
               k.get("truth")) == (want[7], "full", "full", want[13], "net", True), got)
    check("  every cite of the L2 proposal resolves", l2 and all(ev.check_cite(x) for x in l2[0]["cites"]))


# --------------------------------------------------------------------- R7
R7_STEP1 = FIX / "synthetic" / "step1_calibration_failed"
R7_FILES = ("select_summary.json", "admit_summary.json", "select_clusters_by_source.json", "relevance.json")
# The real Step 1 of 2026-09-27 (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3, "On the Step 1 of
# 2026-09-27"): the five sources with a verified cwd12-species box, as (verified boxes,
# increment-pool images). Their images total 1,439 of the pool; N = 6 at M = 393 needs 2,751.
REAL_EVIDENCED = {"project_agml__greenhouse_crop_weed_detection": (461, 13),
                  "project_agml__imageweeds_aerial_weed_detection": (253, 113),
                  "project_agml__weed_crop_detection": (394, 206),
                  "rf_karthikeya-c8pvy__weed-detection-cwp10": (647, 755),
                  "rf_zig-zag-lnodr__weed-detection-vanpe": (294, 352)}


def r7_world(name, edit=None):
    """The fixture tree with pilot_v3 and R7's synthetic Step 1 under TMP/<name>;
    edit(step1_dir) may change the Step 1 files first."""
    root = snapshot_copy(FIX, WHOLE_V3, TMP / name)
    (root / "step1").mkdir(parents=True, exist_ok=True)
    for f in R7_FILES:
        shutil.copyfile(R7_STEP1 / f, root / "step1" / f)
    if edit is not None:
        edit(root / "step1")
    return root, E.load_dir(root, "pilot_v3", exps=WHOLE_V3)


def _edit_json(path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


def real_numbers(step1):
    """R7's Step 1 turned into the real one's evidence: the two synthetic evidenced
    sources leave, and the five real evidenced sources carry their real verified boxes
    and increment-pool images (base B 3,927 images, as R7's)."""
    def sel(o):
        pool = o["sources"]["increment_pool"]
        for s in [s for s in pool if s.startswith("synthetic__")]:
            del pool[s]
            o["retrieval"]["source_evidence"].pop(s, None)
        for s, (ver, img) in REAL_EVIDENCED.items():
            pool[s] = img
            o["retrieval"]["source_evidence"][s] = {"median": 0.1, "species_crops": ver + 10}
        o["sizes"]["increment_pool"] = sum(pool.values())

    def adm(o):
        for s in [s for s in o["per_slug"] if s.startswith("synthetic__")]:
            del o["per_slug"][s]
        for s, (ver, img) in REAL_EVIDENCED.items():
            o["per_slug"][s] = {"boxes": {"verified": ver, "other_ok": 50}, "images": {"admitted": img}}
    _edit_json(step1 / "select_summary.json", sel)
    _edit_json(step1 / "admit_summary.json", adm)


def test_r7():
    print("R7 pilot_v3 + a Step 1 whose relevance.json failed its calibration check -> D2 does not escalate; "
          "L2 with --increment-sources evidence")
    man = json.loads((FIX / "MANIFEST.json").read_text())
    syn = man.get("synthetic") or {}
    check("R7's Step 1 is the pinned synthetic fixture (MANIFEST.json 'synthetic')",
          all(("synthetic/step1_calibration_failed/%s" % f) in syn for f in R7_FILES)
          and all(hashlib.sha256((R7_STEP1 / f).read_bytes()).hexdigest()
                  == syn["synthetic/step1_calibration_failed/%s" % f]["sha256"] for f in R7_FILES), sorted(syn))
    rel = json.loads((R7_STEP1 / "relevance.json").read_text())
    from weed_optimizer_framework.tools.inc import relevance as REL
    try:
        REL.load(R7_STEP1 / "relevance.json", json.loads((R7_STEP1 / "select_summary.json").read_text()))
        why = ""
    except REL.RelevanceError as e:
        why = str(e)
    check("its relevance.json has the real result's shape: tau 0.0557 < tau_min 0.5, 34.5% of train_core below "
          "0.5, and relevance.load refuses it for its calibration (not as malformed)",
          (rel["calibration"]["tau"], rel["calibration"]["check"]["tau_min"], rel["calibration"]["check"]["ok"],
           rel["calibration"]["check"]["share_below_0.5"]) == (0.0557, 0.5, False, 0.345)
          and why.startswith("degenerate calibration in") and "relevance.load refuses it" in why, why[:160])

    root, ev = r7_world("r7")
    diags, by, res = run(ev)
    st = LV.relevance_status(ev)
    check("relevance_status: calibration_failed (made for this select build; its own check failed)",
          st["state"] == "calibration_failed" and (st["tau"], st["tau_min"]) == (0.0557, 0.5), st)
    d2 = by["D2"]
    crit = d2["detail"].get("increment_criterion") or {}
    cap = crit.get("capacity") or {}
    check("D2 fires on the unevidenced sources (csgo, tomato-leaf) but does not escalate: warn, card X9, no "
          "OP_ESCALATE, no L3",
          d2["fired"] and d2["severity"] == "warn" and d2["levers"] == ["X9"]
          and {s["source"] for s in d2["detail"]["sources"]}
          == {"fvossel__csgo_player_detection", "kg_farukalam__tomato-leaf-diseases-detection-computer-vision"}
          and not res["operations"] and not [p for p in res["proposals"] if p["lever"] == "L3"],
          (d2["levers"], d2["severity"], [o["op"] for o in res["operations"]]))
    check("D2 names the L2 variant: then 'L2 with --increment-sources evidence'; D4 proposes that L2 in the same "
          "tick, so it is not also listed as deferred",
          d2["detail"]["then"] == ["L2 with --increment-sources evidence"]
          and not [x for x in res["deferred"] if x["lever"] == "L2"],
          (d2["detail"]["then"], [x for x in res["deferred"] if x["lever"] == "L2"]))
    check("D2's detail records the criterion and why: evidence, the failed calibration, the verifier-grounded rule",
          crit.get("criterion") == "evidence" and crit.get("flag") == "--increment-sources evidence"
          and crit["relevance"]["state"] == "calibration_failed" and "calibration check" in crit.get("why", "")
          and "source-level species evidence" in crit.get("why", ""), crit.get("why"))
    check("  and the capacity it rests on: 5 evidenced sources hold 3,213 images; N 6 x M 393 needs 2,751; the "
          "OtherPlant-heavy share is left to realloop",
          (cap.get("evidenced_pool_images"), cap.get("needed_images"), cap.get("n_verified"),
           cap.get("increment_images"), cap.get("fits"), len(cap.get("evidenced_sources") or {}))
          == (3213, 2751, 6, 393, True, 5) and "realloop build checks it" in cap.get("other_heavy", ""), cap)
    check("  the sources D2 flagged are ones the evidence criterion excludes (0 verified boxes)",
          sorted(crit.get("flagged_excluded") or []) == ["fvossel__csgo_player_detection",
                                                         "kg_farukalam__tomato-leaf-diseases-detection-computer-vision"]
          and crit.get("flagged_evidenced") == [], crit.get("flagged_excluded"))
    ptrs = {(c["artifact"], c["pointer"]) for c in d2["cites"]}
    check("D2 cites the failed check (relevance.json /calibration/tau, /calibration/check/ok) and the evidence "
          "(admit_summary per_slug verified boxes); every cite resolves",
          {("step1/relevance.json", "/calibration/tau"), ("step1/relevance.json", "/calibration/check/ok"),
           ("step1/admit_summary.json", "/per_slug/synthetic__weed-source-a/boxes/verified")} <= ptrs
          and all(ev.check_cite(c) for c in d2["cites"]), sorted(ptrs)[:6])
    check("X9 (revisit the zero-shot criterion) is an R4 card", [c["risk"] for c in res["cards"] if c["lever"] == "X9"]
          == ["R4"], [c["lever"] for c in res["cards"]])
    check("D2b does not read the refused file's statuses (the tomato-leaf source passes there)",
          not by["D2b"]["fired"] and "not evaluated" in by["D2b"]["summary"], by["D2b"]["summary"])
    d4 = by["D4"]
    check("D4 is R6's: ready, recipes ['full'], replay mode full, gate net",
          d4["fired"] and d4["name"] == "decision_slot_ready" and d4["detail"].get("recipes") == ["full"]
          and d4["detail"].get("replay_mode") == "full" and d4["detail"]["gate"]["flips_mode"] == "net", d4["summary"])
    inc = LV.inc_dir(ev, "pilot_v3")
    want = ["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", "realloop_v1",
            "--base", inc + "/step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
            "--increment-sources", "evidence", "--gate-flips-mode", "net"]
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    check("one L2, argv exactly '%s'" % " ".join(want), len(l2) == 1 and l2[0]["argv"] == want,
          [p["argv"] for p in l2])
    check("  its params: increment_sources evidence, no relevance, parent pilot_v3, child realloop_v1, from D4",
          l2 and l2[0]["params"].get("increment_sources") == "evidence" and "relevance" not in l2[0]["params"]
          and (l2[0]["parent_exp"], l2[0]["child_exp"], l2[0]["trigger"][0]) == ("pilot_v3", "realloop_v1", "D4"),
          l2 and l2[0]["params"])
    lptrs = {(c["artifact"], c["pointer"]) for c in (l2[0]["cites"] if l2 else [])}
    check("  its cites carry the criterion's evidence (the failed check, the verified boxes) and resolve",
          l2 and ("step1/relevance.json", "/calibration/check/ok") in lptrs
          and ("step1/admit_summary.json", "/per_slug/rf_karthikeya-c8pvy__weed-detection-cwp10/boxes/verified")
          in lptrs and all(ev.check_cite(x) for x in l2[0]["cites"]), sorted(lptrs)[:4])
    from weed_optimizer_framework.tools.inc import realloop as RL
    got = {}
    real = RL.build

    def fake(*a, **k):
        got["args"], got["kwargs"] = a, k
    RL.build = fake
    try:
        rc = RL.main(want[3:])
    finally:
        RL.build = real
    k = got.get("kwargs") or {}
    check("  realloop's own argparse reads it back (evidence, no relevance file, default min_evidence, full, "
          "full, net, truth on)",
          rc in (0, None) and got.get("args") == ("realloop_v1",)
          and (k.get("base"), k.get("replay_mode"), k.get("recipes"), k.get("increment_sources"), k.get("relevance"),
               k.get("min_evidence"), k.get("gate_flips_mode"), k.get("truth"))
          == (want[7], "full", "full", "evidence", None, None, "net", True), got)
    from weed_optimizer_framework.tools.inc import select as S
    sel = json.loads((root / "step1" / "select_summary.json").read_text())
    ld = S.load_evidence(sel, root / "step1" / "admit_summary.json", S.MIN_EVIDENCE)
    check("  the builder's own evidence loader (select.load_evidence) on the same files: the same evidenced "
          "sources and verified boxes",
          ld["evidenced"] == {s: e["verified_boxes"] for s, e in cap["evidenced_sources"].items()}
          and ld["cross_check"]["checked"], (ld["evidenced"], cap.get("evidenced_sources")))
    from weed_optimizer_framework.tools.inc_autopilot import executor as X
    from weed_optimizer_framework.tools.brain import policy as POL
    if l2:
        row = POL.describe("inc_build_realloop")
        pol, _meta = X.resolve_params("inc_build_realloop", row, l2[0]["params"], l2[0]["argv"],
                                      l2[0]["est_gpu_hours"])
        ok_argv, why_argv = X.argv_check(X.render("inc_build_realloop", pol), l2[0]["argv"])
        ok_pol, bad = POL._check_params(pol, row["param_bounds"])
        check("  the executor renders the same command from its params, inside the policy row's bounds",
              ok_argv and ok_pol and pol.get("increment_sources") == "evidence", (why_argv, bad))
        rec = DG.prospective_d4(FIX / "pilot_v3" / "report.json",
                                out_path=TMP / "r7_replay" / DG.prospective_name("pilot_v3", DG.rules_version()),
                                defn=json.loads((FIX / "pilot_v3" / "exp.json").read_text()),
                                now="2026-09-27T16:00:00Z")
        why_p, differs = DG.prospective_guard(TMP / "r7_replay", "pilot_v3", l2[0]["params"])
        check("  it follows D4's READY record under the current rules (prospective_guard passes)",
              rec["ready"] and why_p == "" and not differs, why_p)
    check("L3 is never proposed over the calibration-failed file (levers refuses it for any proposer)",
          not [p for p in res["proposals"] if p["lever"] == "L3"]
          and raises_defer(lambda: LV._l3_ok(ev), "L3 is not proposed"))
    check("every cite of every fired diagnosis and proposal resolves",
          all(ev.check_cite(x) for d in diags if d["fired"] for x in d["cites"])
          and all(ev.check_cite(x) for p in res["proposals"] for x in p["cites"]))
    _a, evb, db = blind_case("pilot_v3_r7", root, "pilot_v3", WHOLE_V3)
    check("the same under the test-blindness perturbation (L2 with --increment-sources evidence)",
          [p["argv"] for p in LV.propose(db, evb)["proposals"] if p["lever"] == "L2"] == [want])

    # the old path: a passing relevance.json made for the same select build
    def passing(step1):
        def fn(o):
            o["calibration"]["tau"] = 0.62
            o["calibration"]["check"]["ok"] = True
        _edit_json(step1 / "relevance.json", fn)
    _r, ev = r7_world("r7_pass", passing)
    diags, by, res = run(ev)
    want_rel = want[:12] + ["--relevance", inc + "/step1/relevance.json", "--gate-flips-mode", "net"]
    check("a relevance.json that passed its check: the --relevance L2 is unchanged (R6's argv), D2 silent",
          [p["argv"] for p in res["proposals"] if p["lever"] == "L2"] == [want_rel] and not by["D2"]["fired"]
          and LV.relevance_status(ev)["state"] == "matching", [p["argv"] for p in res["proposals"]])

    # negatives: a stale or malformed relevance.json still escalates
    def stale(step1):
        _edit_json(step1 / "relevance.json", lambda o: o["inputs"]["increment_pool"].update(sha256="c" * 64))

    def malformed(step1):          # ok false while tau is at or above tau_min: the check does not follow from tau
        _edit_json(step1 / "relevance.json", lambda o: o["calibration"].update(tau=0.62))

    def no_tau_min(step1):         # a failed check without the rule it was checked under
        _edit_json(step1 / "relevance.json", lambda o: o["calibration"]["check"].pop("tau_min"))

    def ok_true_below(step1):      # ok true while tau is below tau_min: the check does not follow from tau
        _edit_json(step1 / "relevance.json", lambda o: o["calibration"]["check"].update(ok=True))

    def other_rule(step1):         # made under another rule: params.tau_min 0.3 (the check block still 0.5)
        _edit_json(step1 / "relevance.json", lambda o: o["params"].update(tau_min=0.3))

    def other_format(step1):
        _edit_json(step1 / "relevance.json", lambda o: o.update(format="inc.relevance/0"))
    from weed_optimizer_framework.tools.inc import relevance as REL
    for i, (label, edit, state) in enumerate((("stale (made for another select build)", stale, "stale"),
                                              ("malformed (ok false at tau >= tau_min)", malformed, "refused"),
                                              ("malformed (ok true at tau < tau_min)", ok_true_below, "refused"),
                                              ("malformed (a failed check without its tau_min)", no_tau_min,
                                               "refused"),
                                              ("made under another rule (params.tau_min 0.3)", other_rule,
                                               "refused"),
                                              ("of another format", other_format, "refused"))):
        _r, ev = r7_world("r7_neg%d" % i, edit)
        diags, by, res = run(ev)
        d2 = by["D2"]
        check("a relevance.json %s: D2 escalates (crit, OP_ESCALATE), no L2 and no L3" % label,
              LV.relevance_status(ev)["state"] == state and d2["fired"] and d2["severity"] == "crit"
              and "OP_ESCALATE" in d2["levers"] and not realloop_argvs(res)
              and not [p for p in res["proposals"] if p["lever"] == "L3"]
              and any(x["lever"] == "L2" and "a person" in x["reason"] for x in res["deferred"]),
              (LV.relevance_status(ev), d2["levers"], [p["argv"] for p in res["proposals"]]))
        if state == "refused":
            try:
                REL.load(_r / "step1" / "relevance.json", json.loads((_r / "step1" / "select_summary.json").read_text()))
                why = ""
            except REL.RelevanceError as e:
                why = str(e)
            check("  ... and relevance.load refuses it as malformed, not for its calibration",
                  why and not why.startswith("degenerate calibration in"), why[:160])

    # a calibration-failed file fires D2 whatever the per-source evidence says
    def no_admit(step1):
        (step1 / "admit_summary.json").unlink()
    _r, ev = r7_world("r7_no_admit", no_admit)
    diags, by, res = run(ev)
    d2 = by["D2"]
    check("calibration failed, no admit_summary.json: D2 escalates (crit, OP_ESCALATE, X9): the evidence criterion "
          "cannot be read; no L2, no L3; the L2 is deferred naming the missing summary",
          d2["fired"] and d2["severity"] == "crit" and d2["levers"] == ["OP_ESCALATE", "X9"]
          and "admit_summary.json is not in the evidence" in d2["detail"].get("needs", "")
          and not realloop_argvs(res) and not [p for p in res["proposals"] if p["lever"] == "L3"]
          and any(x["lever"] == "L2" and "admit_summary.json is not in the evidence" in x["reason"]
                  for x in res["deferred"])
          and [c["lever"] for c in res["cards"]].count("X9") == 1,
          (d2["fired"], d2["levers"], [o["op"] for o in res["operations"]]))

    def no_clusters(step1):
        (step1 / "select_clusters_by_source.json").unlink()
    _r, ev = r7_world("r7_no_clusters", no_clusters)
    diags, by, res = run(ev)
    d2 = by["D2"]
    check("calibration failed, no per-source aggregate: D2 still fires (warn, X9, nothing flagged) and the L2 is R7's "
          "--increment-sources evidence command",
          d2["fired"] and d2["severity"] == "warn" and d2["levers"] == ["X9"] and d2["detail"]["sources"] == []
          and [p["argv"] for p in res["proposals"] if p["lever"] == "L2"] == [want]
          and [c["lever"] for c in res["cards"]].count("X9") == 1 and not res["operations"],
          (d2["fired"], d2["levers"], [p["argv"] for p in res["proposals"]]))

    # the real Step 1's evidenced pool cannot hold the default build: R8 (the pinned real
    # summaries) is that case; here R7's synthetic Step 1 carrying the real numbers must
    # agree with it (the sizing rule, not an escalation)
    _r, ev = r7_world("r7_real", real_numbers)
    diags, by, res = run(ev)
    d2 = by["D2"]
    sz = (d2["detail"].get("increment_criterion") or {}).get("sizing") or {}
    check("the real Step 1's numbers in R7's synthetic Step 1 (1,439 images < 2,751 for N 6 x M 393): D2 does not "
          "escalate; the R4 sizing rule gives R8's L2 (N 4 x M 287)",
          d2["fired"] and d2["severity"] == "warn" and d2["levers"] == ["X9"] and not res["operations"]
          and (sz.get("default") or {}).get("needed_images") == 2751 and sz.get("applied") is True
          and [p["argv"] for p in res["proposals"] if p["lever"] == "L2"] == [r8_argv(inc)],
          (d2["levers"], sz.get("sized"), [p["argv"] for p in res["proposals"] if p["lever"] == "L2"]))


def raises_defer(fn, contains):
    try:
        fn()
    except LV.Defer as e:
        return contains in str(e)
    return False


# --------------------------------------------------------------------- R8
R8_COPIES = FIX / "step1_copies"
R8_COPY_FILES = ("select_summary.json", "admit_summary.json")
R8_REL = FIX / "synthetic" / "step1_real_calibration_failed" / "relevance.json"


def r8_argv(inc, size="287", n_verified="4"):
    """R8's L2: D4's decision on pilot_v3 (full replay, recipe full, gate net) on the
    evidence criterion, sized by the R4 rule; the flags in the L2 template's order
    (levers.json L2 argv, executor.ARGV_FORMS): --increment-sources, --size,
    --n-verified, --gate-flips-mode; a flag at the builders' default is not rendered."""
    out = ["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", "realloop_v1",
           "--base", inc + "/step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
           "--increment-sources", "evidence"]
    if size is not None:
        out += ["--size", size]
    if n_verified is not None:
        out += ["--n-verified", n_verified]
    return out + ["--gate-flips-mode", "net"]


def r8_world(name, edit=None):
    """The fixture tree with pilot_v3, the pinned byte copies of the real Step 1's
    select and admit summaries, and R8's calibration-failed relevance.json, under
    TMP/<name>; edit(step1_dir) may change the Step 1 files first."""
    root = snapshot_copy(FIX, WHOLE_V3, TMP / name)
    (root / "step1").mkdir(parents=True, exist_ok=True)
    for f in R8_COPY_FILES:
        shutil.copyfile(R8_COPIES / f, root / "step1" / f)
    shutil.copyfile(R8_REL, root / "step1" / "relevance.json")
    if edit is not None:
        edit(root / "step1")
    return root, E.load_dir(root, "pilot_v3", exps=WHOLE_V3)


def evidenced_pool(images):
    """An edit of R8's select_summary.json: the five evidenced sources' increment-pool
    images scaled down to about `images` in total (every other number as pinned)."""
    def edit(step1):
        def fn(o):
            pool = o["sources"]["increment_pool"]
            tot = sum(pool[s] for s in REAL_EVIDENCED)
            new = {s: max(1, pool[s] * images // tot) for s in REAL_EVIDENCED}
            new["rf_karthikeya-c8pvy__weed-detection-cwp10"] += images - sum(new.values())
            pool.update(new)
            o["sizes"]["increment_pool"] = sum(pool.values())
        _edit_json(step1 / "select_summary.json", fn)
    return edit


def test_r8():
    print("R8 pilot_v3 + the real Step 1 summaries (1,439 evidenced images < 2,751) + a calibration-failed "
          "relevance.json -> D2 does not escalate: the R4 sizing rule, L2 at N 4 x M 287")
    man = json.loads((FIX / "MANIFEST.json").read_text())
    cps, syn = man.get("step1_copies") or {}, man.get("synthetic") or {}
    check("R8's Step 1 is pinned: byte copies of results/framework/inc/step1/{select,admit}_summary.json "
          "(MANIFEST.json 'step1_copies') and a synthetic relevance.json (MANIFEST.json 'synthetic')",
          all(hashlib.sha256((R8_COPIES / f).read_bytes()).hexdigest() == cps["step1_copies/%s" % f]["sha256"]
              and cps["step1_copies/%s" % f]["from"] == "results/framework/inc/step1/%s" % f for f in R8_COPY_FILES)
          and hashlib.sha256(R8_REL.read_bytes()).hexdigest()
          == syn["synthetic/step1_real_calibration_failed/relevance.json"]["sha256"], sorted(cps))
    sel = json.loads((R8_COPIES / "select_summary.json").read_text())
    check("the copies carry the protocol's numbers: base B 3,927 images, 96,088 increment-pool images, the five "
          "evidenced sources' pool images",
          sel["sizes"]["base_B"] == 3927 and sum(sel["sources"]["increment_pool"].values()) == 96088
          and {s: sel["sources"]["increment_pool"][s] for s in REAL_EVIDENCED}
          == {s: img for s, (_v, img) in REAL_EVIDENCED.items()}, sel["sizes"])
    from weed_optimizer_framework.tools.inc import relevance as REL
    try:
        REL.load(R8_REL, sel)
        why = ""
    except REL.RelevanceError as e:
        why = str(e)
    check("its relevance.json is made for the real select build and relevance.load refuses it for its calibration "
          "(tau 0.0557 < 0.5), not as malformed", why.startswith("degenerate calibration in"), why[:160])

    root, ev = r8_world("r8")
    diags, by, res = run(ev)
    st = LV.relevance_status(ev)
    check("relevance_status: calibration_failed", st["state"] == "calibration_failed", st)
    d2 = by["D2"]
    crit = d2["detail"].get("increment_criterion") or {}
    cap, sz = crit.get("capacity") or {}, crit.get("sizing") or {}
    check("D2 fires but does not escalate: warn, card X9, no OP_ESCALATE, no L3",
          d2["fired"] and d2["severity"] == "warn" and d2["levers"] == ["X9"] and not res["operations"]
          and not [p for p in res["proposals"] if p["lever"] == "L3"] and "needs" not in d2["detail"],
          (d2["levers"], d2["severity"], [o["op"] for o in res["operations"]], d2["detail"].get("needs")))
    check("D2's detail records the default N and M: N 6 x M 393 needs 2,751 > 1,439 (at most --n-verified 2), "
          "8 decided increments",
          sz.get("default") == {"n_verified": 6, "increment_images": 393, "needed_images": 2751, "fits": False,
                                "max_n_verified": 2, "decided_increments": 8}, sz.get("default"))
    check("  the sized N and M: N 4 (6 decided increments), M = min(393, floor(1439 / 5)) = 287 (7.3% of B), "
          "needing 1,435; above the floor 0.05 x 3,927 = 196.35",
          sz.get("sized") == {"n_verified": 4, "increment_images": 287, "needed_images": 1435,
                              "decided_increments": 6, "increment_frac": 0.0731, "fits": True}
          and sz.get("applied") is True and (sz.get("min_decided_increments"), sz.get("min_increment_frac"),
                                             sz.get("min_increment_images")) == (6, 0.05, 196.35)
          and sz.get("params") == {"size": 287, "n_verified": 4}
          and sz.get("flags") == ["--size", "287", "--n-verified", "4"], sz)
    check("  and the rule (the R4 review of 2026-09-27)",
          "R4 review of 2026-09-27" in sz.get("rule", "") and "min_decided_increments" in sz.get("rule", "")
          and "min_increment_frac" in sz.get("rule", ""), sz.get("rule"))
    check("  the capacity at the sized N and M: 5 evidenced sources (their real verified boxes and pool images) hold "
          "1,439 images >= 1,435",
          (cap.get("evidenced_pool_images"), cap.get("needed_images"), cap.get("n_verified"),
           cap.get("increment_images"), cap.get("fits"), cap.get("base_images")) == (1439, 1435, 4, 287, True, 3927)
          and {s: (e["verified_boxes"], e["pool_images"]) for s, e in (cap.get("evidenced_sources") or {}).items()}
          == REAL_EVIDENCED and "realloop build checks it" in cap.get("other_heavy", ""), cap)
    check("  criterion evidence, its flags with the sized N and M; then 'L2 with --increment-sources evidence --size "
          "287 --n-verified 4', not also deferred (D4's L2 carries it)",
          crit.get("criterion") == "evidence"
          and crit.get("flags") == ["--increment-sources", "evidence", "--size", "287", "--n-verified", "4"]
          and d2["detail"]["then"] == ["L2 with --increment-sources evidence --size 287 --n-verified 4"]
          and not [x for x in res["deferred"] if x["lever"] == "L2"],
          (crit.get("flags"), d2["detail"]["then"], [x for x in res["deferred"] if x["lever"] == "L2"]))
    check("D2's summary names the default and the sized loop", "the default build's 2751 = (6 + 1) x 393" in d2["summary"]
          and "M = min(393, floor(1439 / 5)) = 287" in d2["summary"], d2["summary"][-400:])
    check("every cite of D2 resolves", all(ev.check_cite(c) for c in d2["cites"]))
    d4 = by["D4"]
    check("D4 is R6's: ready, recipes ['full'], replay mode full, gate net",
          d4["fired"] and d4["name"] == "decision_slot_ready" and d4["detail"].get("recipes") == ["full"]
          and d4["detail"].get("replay_mode") == "full" and d4["detail"]["gate"]["flips_mode"] == "net", d4["summary"])
    inc = LV.inc_dir(ev, "pilot_v3")
    want = r8_argv(inc)
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    check("one L2, argv exactly '%s'" % " ".join(want), len(l2) == 1 and l2[0]["argv"] == want,
          [p["argv"] for p in l2])
    check("  its params: evidence, size 287, n_verified 4, no relevance; parent pilot_v3, child realloop_v1, from D4",
          l2 and (l2[0]["params"].get("increment_sources"), l2[0]["params"].get("size"),
                  l2[0]["params"].get("n_verified")) == ("evidence", 287, 4) and "relevance" not in l2[0]["params"]
          and (l2[0]["parent_exp"], l2[0]["child_exp"], l2[0]["trigger"][0]) == ("pilot_v3", "realloop_v1", "D4"),
          l2 and l2[0]["params"])
    check("  priced at the sized loop (estimate_realloop at N 4 x M 287, plus the build job)",
          l2 and l2[0]["est_gpu_hours"] == LV.price("L2", l2[0]["params"], ev, "pilot_v3")[0]
          and (l2[0]["estimate"].get("size"), l2[0]["estimate"].get("n_verified")) == (287, 4),
          l2 and l2[0].get("estimate"))
    check("  every cite of the L2 resolves", l2 and all(ev.check_cite(x) for x in l2[0]["cites"]))
    from weed_optimizer_framework.tools.inc import realloop as RL
    got = {}
    real = RL.build

    def fake(*a, **k):
        got["args"], got["kwargs"] = a, k
    RL.build = fake
    try:
        rc = RL.main(want[3:])
    finally:
        RL.build = real
    k = got.get("kwargs") or {}
    check("  realloop's own argparse reads every flag back (n_verified 4, size 287, evidence, no relevance file, "
          "default min_evidence, full, full, net, truth on)",
          rc in (0, None) and got.get("args") == ("realloop_v1",)
          and (k.get("base"), k.get("n_verified"), k.get("size"), k.get("replay_mode"), k.get("recipes"),
               k.get("increment_sources"), k.get("relevance"), k.get("min_evidence"), k.get("gate_flips_mode"),
               k.get("truth")) == (want[7], 4, 287, "full", "full", "evidence", None, None, "net", True), got)
    check("  realloop's sequence at --n-verified 4 decides 6 increments (the acceptance), with UNVERIFIED and "
          "OTHER_HEAVY", len(RL.sequence(4)) == 6 and len(RL.sequence(3)) == 5
          and [n for n, _c in LV.realloop_sequence(4)] == RL.sequence(4), RL.sequence(4))
    from weed_optimizer_framework.tools.inc import select as S
    ld = S.load_evidence(sel, root / "step1" / "admit_summary.json", S.MIN_EVIDENCE)
    check("  the builder's own evidence loader (select.load_evidence) on the same files: the same evidenced "
          "sources and verified boxes",
          ld["evidenced"] == {s: e["verified_boxes"] for s, e in cap["evidenced_sources"].items()}
          and ld["cross_check"]["checked"], ld["evidenced"])
    from weed_optimizer_framework.tools.inc_autopilot import executor as X
    from weed_optimizer_framework.tools.inc_autopilot import validate as V
    from weed_optimizer_framework.tools.brain import policy as POL
    if l2:
        row = POL.describe("inc_build_realloop")
        pol, _meta = X.resolve_params("inc_build_realloop", row, l2[0]["params"], l2[0]["argv"],
                                      l2[0]["est_gpu_hours"])
        ok_argv, why_argv = X.argv_check(X.render("inc_build_realloop", pol), l2[0]["argv"])
        ok_pol, bad = POL._check_params(pol, row["param_bounds"])
        check("  the executor renders the same command from its params (ARGV_FORMS), inside the policy row's and "
              "L2's own bounds",
              ok_argv and ok_pol and (pol.get("size"), pol.get("n_verified")) == (287, 4)
              and not X._lever_params_check("L2", dict(l2[0]["params"])), (why_argv, bad))
        rec = DG.prospective_d4(FIX / "pilot_v3" / "report.json",
                                out_path=TMP / "r8_replay" / DG.prospective_name("pilot_v3", DG.rules_version()),
                                defn=json.loads((FIX / "pilot_v3" / "exp.json").read_text()),
                                now="2026-09-27T16:00:00Z")
        why_p, differs = DG.prospective_guard(TMP / "r8_replay", "pilot_v3", l2[0]["params"])
        check("  it follows D4's READY record under the current rules (prospective_guard passes)",
              rec["ready"] and why_p == "" and not differs, why_p)
        ctx = {"ev": ev, "lmenu": LV.load_menu(), "diags": by, "exp": "pilot_v3"}
        mat = V.materialise("L2", LV.row("L2"), {}, ctx)
        check("  validate.materialise: a brain L2 that names no --size or --n-verified is the same sized request",
              {k: v for k, v in mat["params"].items() if k != "est_gpu_hours"}
              == {k: v for k, v in l2[0]["params"].items() if k != "est_gpu_hours"}
              and LV.argv("L2", dict(mat["params"], est_gpu_hours=mat["est"])) == want,
              mat["params"])
        mat = V.materialise("L2", LV.row("L2"), {"size": 287, "n_verified": 4}, ctx)
        check("  ... and one that names N 4 x M 287 itself is checked at them and admitted",
              (mat["params"].get("size"), mat["params"].get("n_verified"), mat["params"].get("increment_sources"))
              == (287, 4, "evidence"), mat["params"])
        check("  ... one that names only --size 287 is checked at the default N (7 x 287 > 1,439) and deferred",
              raises_defer(lambda: V.materialise("L2", LV.row("L2"), {"size": 287}, ctx), "cannot supply"))
        for bp, want_why in (({"size": 196, "n_verified": 4}, "196.35 images"),
                             ({"size": 287, "n_verified": 3}, "fewer than min_decided_increments 6"),
                             ({"size": 50, "n_verified": 1}, "--n-verified 1 --size 50 is below the R4 rule's floor")):
            try:
                V.materialise("L2", LV.row("L2"), bp, ctx)
                why_f = None
            except LV.LeverError as e:
                why_f = str(e)
            check("  ... one that names N %d x M %d, below the R4 rule's floor (M >= 196.35, 6 decided), is refused "
                  "with the numbers, never filed" % (bp["n_verified"], bp["size"]),
                  why_f is not None and want_why in why_f, why_f)
        # The sized loop meets realloop's OtherPlant-heavy check or select's whole-group draw on the cluster:
        # DREF escalates (a person), and the refusal is one the identical build meets again (retry false).
        for msg in ("[inc.realloop] ERROR: increment sources 'evidence': the evidenced increment pool cannot supply 4 "
                    "verified increments + OTHER_HEAVY of 287 images. It holds 1439 images (1435 needed), 212 of "
                    "them in OtherPlant-heavy near-dup groups (287 needed)",
                    "[inc.realloop] ERROR: 4 increments of 287 images need 1148, the draw found 1140"):
            evr = E.load_dir(root, "pilot_v3", exps=WHOLE_V3,
                             context={"refusals": [{"builder": "inc_build_realloop", "message": msg}]})
            dr = DG.by_id(DG.detect(evr))["DREF"]
            check("  a cluster refusal of the sized loop (%s...): DREF crit, OP_ESCALATE, a person, retry false"
                  % msg[22:70], dr["fired"] and dr["severity"] == "crit" and dr["levers"] == ["OP_ESCALATE"]
                  and dr["detail"]["refusals"][0]["retry"] is False
                  and dr["detail"]["refusals"][0]["needs"].startswith("a person"),
                  (dr["levers"], dr["detail"].get("refusals")))
    _a, evb, db = blind_case("pilot_v3_r8", root, "pilot_v3", WHOLE_V3)
    check("the same under the test-blindness perturbation (the sized L2)",
          [p["argv"] for p in LV.propose(db, evb)["proposals"] if p["lever"] == "L2"] == [want])

    # the capacity below the floor: the rule does not size, D2 escalates with the numbers
    _r, ev = r8_world("r8_floor", evidenced_pool(980))
    diags, by, res = run(ev)
    d2 = by["D2"]
    crit = d2["detail"].get("increment_criterion") or {}
    sz = crit.get("sizing") or {}
    check("an evidenced pool of 980 images: M = min(393, floor(980 / 5)) = 196 < 0.05 x 3,927 = 196.35, so the rule "
          "does not size: D2 escalates (crit, OP_ESCALATE, X9) with the numbers",
          d2["fired"] and d2["severity"] == "crit" and d2["levers"] == ["OP_ESCALATE", "X9"]
          and crit.get("criterion") is None and sz.get("applied") is False
          and (sz.get("sized") or {}).get("increment_images") == 196 and (sz.get("sized") or {}).get("fits") is False
          and "196.35" in d2["detail"].get("needs", "") and "review decision" in d2["detail"].get("needs", ""),
          (d2["levers"], sz.get("sized"), d2["detail"].get("needs", "")[-300:]))
    check("  no L2 and no L3: the L2 is deferred naming the sized M and the floor",
          not realloop_argvs(res) and not [p for p in res["proposals"] if p["lever"] == "L3"]
          and any(x["lever"] == "L2" and "= 196 images" in x["reason"] and "196.35" in x["reason"]
                  for x in res["deferred"]), [x["reason"][-200:] for x in res["deferred"] if x["lever"] == "L2"])
    _r, ev = r8_world("r8_floor_ok", evidenced_pool(985))
    diags, by, res = run(ev)
    check("  five images more (985 images: M = 197 >= 196.35): sized, L2 at --size 197 --n-verified 4",
          by["D2"]["severity"] == "warn" and [p["argv"] for p in res["proposals"] if p["lever"] == "L2"]
          == [r8_argv(inc, size="197")], [p["argv"] for p in res["proposals"] if p["lever"] == "L2"])
    _r, ev = r8_world("r8_n_only", evidenced_pool(2000))
    diags, by, res = run(ev)
    check("an evidenced pool of 2,000 images (< 2,751, >= 5 x 393): N 4 at the default M, so only --n-verified 4 "
          "is rendered (a flag at the builders' default is not)",
          by["D2"]["severity"] == "warn" and [p["argv"] for p in res["proposals"] if p["lever"] == "L2"]
          == [r8_argv(inc, size=None)], [p["argv"] for p in res["proposals"] if p["lever"] == "L2"])
    _r, ev = r8_world("r8_no_admit", lambda step1: (step1 / "admit_summary.json").unlink())
    diags, by, res = run(ev)
    check("the capacity cannot be computed (no admit_summary.json): D2 escalates, no L2",
          by["D2"]["severity"] == "crit" and "OP_ESCALATE" in by["D2"]["levers"] and not realloop_argvs(res),
          by["D2"]["levers"])


# --------------------------------------------------------------------- R2
def test_r2():
    print("R2 Step 1: unmeasured source property -> L3, then L2 --relevance")
    need = ["step1/select_summary.json", "step1/admit_summary.json", "step1/select_clusters_by_source.json"]
    missing = [n for n in need if not (FIX / n).is_file()]
    if missing:
        skip("R2", "the Step 1 fixture is not pulled yet (%s); see tests/fixtures/inc_replay/MANIFEST.json"
             % ", ".join(missing))
        return
    no_rel = snapshot_copy(FIX, [], TMP / "r2_norel")
    if (no_rel / "step1" / "relevance.json").exists():
        (no_rel / "step1" / "relevance.json").unlink()
    ev = E.load_dir(no_rel, "pilot_v1", exps=[])
    diags, by, res = run(ev)
    names = [s["source"] for s in by["D2"]["detail"]["sources"]]
    check("D2 names fvossel__csgo_player_detection and rf_bishwarup-halder__crop-health-advisor",
          by["D2"]["fired"] and {"fvossel__csgo_player_detection", "rf_bishwarup-halder__crop-health-advisor"}
          <= set(names), names)
    check("lever L3 is proposed", [p["lever"] for p in res["proposals"]] == ["L3"], res["proposals"])
    check("then L2 with --relevance (deferred)", any(x["lever"] == "L2" and "--relevance" in x["reason"]
                                                     for x in res["deferred"]), res["deferred"])
    if not (FIX / "step1" / "relevance.json").is_file():
        skip("R2 with relevance.json", "step1/relevance.json is not pulled yet")
        return
    ev = E.load_dir(FIX, "pilot_v1", exps=[])
    diags, by, res = run(ev)
    check("with relevance.json: D2 silent", not by["D2"]["fired"], by["D2"]["summary"])
    rel = ev.json("step1/relevance.json")
    tomato = [s for s in rel["increment_pool"]["sources"] if s.startswith("kg_farukalam__tomato-leaf")]
    passes = [s for s in tomato if rel["increment_pool"]["sources"][s]["status"] == "pass"]
    if passes:
        check("D2b fires on the tomato-leaf source, which passes", by["D2b"]["fired"]
              and set(passes) <= {s["source"] for s in by["D2b"]["detail"]["sources"]}, by["D2b"]["summary"])
        check("X2 appears as a card", [x["risk"] for x in res["cards"] if x["lever"] == "X2"] == ["R4"])
    else:
        print("  note the tomato-leaf source does not pass relevance (%s): D2b has nothing to flag there" % tomato)


# --------------------------------------------------------------------- R4b
def test_r4b():
    print("R4b prospective D4")
    rep_path = FIX / "pilot_v1" / "report.json"
    out = TMP / "prospective" / "prospective_4.json"
    rec = DG.prospective_d4(rep_path, out_path=out, now="2026-09-27T10:00:00Z")
    check("writes D4's decision for pilot_v1: no recipe is ready",
          out.is_file() and rec["outcome"] == "no_recipe_tracks_truth" and rec["ready"] is False
          and rec["recipes"] is None and rec["exp"] == "pilot_v1", rec["outcome"])
    check("it records the report's sha256 and the thresholds' sha256",
          rec["report_sha256"] == hashlib.sha256(rep_path.read_bytes()).hexdigest()
          and len(rec["thresholds_sha256"]) == 64)
    again = DG.prospective_d4(rep_path, out_path=out, now="2026-09-28T10:00:00Z")
    check("the same decision again is a no-op (the first record stands)",
          again["decided_utc"] == "2026-09-27T10:00:00Z")
    rep = json.loads(rep_path.read_text())
    rep["agreement"]["full"] = {"agree": 6, "compared": 7, "rate": 6 / 7.0}
    try:
        DG.prospective_d4(rep, out_path=out)
        refused = False
    except ValueError:
        refused = True
    check("a different decision over it refuses (written once)", refused)
    blocked = DG.prospective_d4(rep, out_path=TMP / "prospective" / "blocked.json", now="2026-09-27T11:00:00Z")
    check("a ready report on which D1 fires (pilot_v1's P_recipe 0.0) is not ready: blocked by D1",
          blocked["outcome"] == "decision_slot_ready" and blocked["ready"] is False
          and blocked["blocked_by"] == "D1" and blocked["recipes"] is None, blocked["diagnosis"]["summary"])
    for st in rep["steps"]:
        for c in (st.get("chains") or {}).values():
            if isinstance(c, dict) and "p_recipe" in c:
                c["p_recipe"] = 0.5
    other = DG.prospective_d4(rep, out_path=TMP / "prospective" / "other.json", now="2026-09-27T11:00:00Z")
    check("on a ready report (D1 silent) it names the replay mode and recipe",
          other["ready"] and other["replay_mode"] == "sample" and other["recipes"] == ["full"]
          and other["blocked_by"] is None, other["outcome"])
    cmp = DG.compare_prospective(other, "sample", "full")
    check("compare_prospective: match", cmp["match"] and cmp["recipes_match"] and cmp["replay_mode_match"])
    check("compare_prospective: a different choice does not match",
          not DG.compare_prospective(other, "full", "full,lora")["match"])
    blind = DG.prospective_d4(perturb_non_dev(json.loads(rep_path.read_text())),
                              out_path=TMP / "prospective" / "blind.json", now="2026-09-27T10:00:00Z")
    base = DG.prospective_d4(json.loads(rep_path.read_text()), out_path=TMP / "prospective" / "base.json",
                             now="2026-09-27T10:00:00Z")
    check("the prospective decision is test-blind",
          json.dumps(blind["diagnosis"], sort_keys=True) == json.dumps(base["diagnosis"], sort_keys=True))
    v2 = FIX / "pilot_v2" / "report.json"
    if not v2.is_file():
        skip("R4b on pilot_v2", "pilot_v2/report.json is not pulled yet; when it is, run prospective_d4 on it and "
                                "commit %s before any real-loop build" % DG.PROSPECTIVE_FILE)
        return
    defn = json.loads((FIX / "pilot_v2" / "exp.json").read_text()) if (FIX / "pilot_v2" / "exp.json").is_file() \
        else None
    fresh = DG.prospective_d4(v2, out_path=TMP / "prospective" / "pilot_v2.json", defn=defn)
    print("  info pilot_v2: %s" % fresh["diagnosis"]["summary"])
    check("pilot_v2: no recipe is chosen (D1 fires in full replay: the recipe forgets, X1)",
          fresh["outcome"] == "silent" and not fresh["ready"] and "X1" in fresh["diagnosis"]["summary"],
          fresh["outcome"])
    committed = DG.PROSPECTIVE_FILE
    if not committed.is_file():
        skip("R4b committed record", "%s is not committed yet" % committed)
        return
    rec = json.loads(committed.read_text())
    check("the committed record is pilot_v2's and reproduces", rec["report_sha256"] == fresh["report_sha256"]
          and (rec["replay_mode"], rec["recipes"]) == (fresh["replay_mode"], fresh["recipes"]), rec)
    loops = [p for p in FIX.iterdir() if p.is_dir() and (p / "exp.json").is_file()
             and json.loads((p / "exp.json").read_text()).get("builder") == DG.REALLOOP_BUILDER]
    if not loops:
        skip("R4b comparison", "no real-loop exp.json pulled yet")
        return
    manual = json.loads((loops[0] / "exp.json").read_text())
    check("the record predates the person's real-loop build",
          rec["decided_utc"] < manual.get("initialised_utc", ""), (rec["decided_utc"], manual.get("initialised_utc")))
    res = DG.compare_prospective(rec, manual.get("replay_mode", "sample"), list(manual.get("recipes") or {}))
    print("  RESULT R4b (the generalisation case): rule %s vs person %s -> match %s"
          % (res["rule"], res["manual"], res["match"]))


# ------------------------------------------------- synthetic healthy pilot
SEEDS_NOISE = {0: 0.0, 1: 0.002, 2: -0.002}
FACTOR = {"dev": 1.0, "ood22": 0.7, "ood23": 0.6, "imageweeds": 0.5, "test": 0.95}
SECONDS = {"base": 7200.0, "union": 7200.0, "cand": 1800.0, "null": 1800.0, "soup": 60.0, "final": 300.0}
STEPS = [("I1", True), ("I2", True), ("Bswap", False), ("I3", True)]
EFFECT = {"I1": 0.02, "I2": 0.02, "I3": 0.02, "Bswap": -0.06}


class Executor:
    """What inc/train.py writes, with dev scores from a known model: a cand
    run scores its incumbent + the increment's effect (clean +0.02, Bswap
    -0.06 on the 12-class score only, so the agnostic score holds and the
    gate attributes it to labels) + a seed offset; a null run scores the
    incumbent + the seed offset (so P_recipe = 0.5); a cold run 0.40 + 0.02
    per clean increment it holds - 0.06 with Bswap."""

    def __init__(self, keys):
        self.keys = keys

    @staticmethod
    def parent(weights):
        s = json.loads((pathlib.Path(weights).parent.parent / "scores" / "dev.json").read_text())
        return s["map50_95"], s["agnostic_map50_95"]

    def values(self, spec):
        from weed_optimizer_framework.tools.inc import common as C
        kind, rid, s = spec["kind"], spec["run_id"], SEEDS_NOISE.get((spec.get("recipe") or {}).get("seed"), 0.0)
        if kind in ("base", "union"):
            keys = {r["key"] for r in C.read_manifest(spec["train_manifest"])}
            n_clean = sum(1 for n, c in STEPS if c and self.keys[n] <= keys)
            bswap = bool(self.keys["Bswap"] & keys)
            return 0.40 + 0.02 * n_clean - 0.06 * bswap + s, 0.60 + 0.02 * n_clean + s
        if kind in ("cand", "null"):
            pv, pa = self.parent(spec["init"])
            eff = EFFECT[rid.split("__")[1].split("_", 1)[1]] if kind == "cand" else 0.0
            return pv + eff + s, pa + max(eff, 0.0) + s
        if kind == "soup":
            vals = [self.parent(w) for w in spec["soup_of"]]
            return statistics.fmean(v for v, _ in vals) + 0.001, statistics.fmean(a for _, a in vals) + 0.001
        return self.parent(spec["init"])

    def __call__(self, spec):
        from weed_optimizer_framework.tools.inc import common as C
        from weed_optimizer_framework.tools.inc import gate as G
        out = pathlib.Path(spec["out_dir"])
        w = out / "weights" / "final.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "final":
            if w.is_symlink() or w.exists():
                w.unlink()
            os.symlink(spec["init"], w)
        else:
            w.write_bytes(("weights of %s" % spec["run_id"]).encode())
        v0, a0 = self.values(spec)
        for exam in spec["exams"]:
            v, a = v0 * FACTOR[exam], a0 * FACTOR[exam]
            n_gt = {sp: 40 for sp in G.SPECIES}
            n_gt["OtherPlant"] = 0
            sc = {"exam": exam, "scorer_sha256": "TEST-" + "5" * 64, "production": False,
                  "manifest_sha256": hashlib.sha256(("m/" + exam).encode()).hexdigest(),
                  "key_order_sha256": hashlib.sha256(("k/" + exam).encode()).hexdigest(),
                  "deviations": ["LOCK.json not checked"], "weights_sha256": C.sha256_file(w), "n_images": 20,
                  "map50_95": v, "map50": min(1.0, v + 0.2), "agnostic_map50_95": a,
                  "agnostic_map50": min(1.0, a + 0.2), "per_class": {sp: v for sp in G.SPECIES}, "n_gt": n_gt,
                  "image_correct": "1" * 20, "species_map50_95": v, "species_map50": v + 0.2}
            (out / "scores").mkdir(parents=True, exist_ok=True)
            (out / "scores" / ("%s.json" % exam)).write_text(json.dumps(sc))
        (out / "run.json").write_text(json.dumps({"status": "done", "attempt": 1, "seconds": SECONDS[spec["kind"]],
                                                  "error": None, "weights_sha256": C.sha256_file(w)}))
        return "COMPLETED"


def healthy_pilot():
    """A pilot-shaped chain experiment (P0, I1, I2, Bswap planted, I3; full
    and freeze chains; truth arm) driven to done by the real driver on
    FakeBackend, then inc.report. Returns (INC_DIR, exp)."""
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.inc import driver as D
    from weed_optimizer_framework.tools.inc import pilot as P
    from weed_optimizer_framework.tools.inc import report as R
    exp = "healthy_v1"
    mdir = pathlib.Path(C.INC_DIR) / exp / "manifests"

    def rows(name, n):
        return [{"image": "/nowhere/%s/%d.png" % (name, i), "label": "/nowhere/%s/%d.txt" % (name, i),
                 "sha256": hashlib.sha256(("%s/%d" % (name, i)).encode()).hexdigest(),
                 "label_sha256": hashlib.sha256(("L%s/%d" % (name, i)).encode()).hexdigest(),
                 "source": "cottonweeddet12/train", "session": "%s_s" % name, "key": "%s__%02d" % (name, i)}
                for i in range(n)]
    keys, entries = {}, []
    base_rows = rows("P0", 40)
    for name, clean in STEPS:
        rs = rows(name, 10)
        keys[name] = {r["key"] for r in rs}
        p = mdir / ("%s.jsonl" % name)
        e = {"name": name, "manifest": str(p), "manifest_sha256": C.write_manifest(p, rs), "n_images": 10,
             "clean": clean}
        if name == "Bswap":
            e["planted"] = "40% of boxes relabelled to another species"
        entries.append(e)
    bp = mdir / "P0.jsonl"
    defn = {"exp": exp, "type": "chain", "builder": DG.PILOT_BUILDER, "testing": True, "seeds": [0, 1, 2],
            "init_weights": "yolo11n.pt", "decision_exam": "dev", "final_exams": list(D.FINAL_EXAMS),
            "base": {"name": "P0", "manifest": str(bp), "manifest_sha256": C.write_manifest(bp, base_rows),
                     "n_images": len(base_rows), "recipe": P.cold_recipe()},
            "steps": entries, "recipes": {r: P.inc_recipes()[r] for r in ("full", "freeze")},
            "truth": True, "truth_recipe": P.cold_recipe()}
    fb = D.FakeBackend(Executor(keys))
    clock_t = [1.8e9]
    drv = lambda: D.Driver(exp, backend=fb, clock=lambda: clock_t[0], quiet=True)  # noqa: E731
    drv().init(defn)
    for _ in range(200):
        ran = fb.run_pending()
        before = len(fb.submissions)
        drv().advance()
        st = json.loads(D.Paths(exp).state.read_text())
        if st["done"] or (ran == 0 and len(fb.submissions) == before):
            break
    R.build(exp)
    return pathlib.Path(C.INC_DIR), exp, st


def test_negative_controls():
    print("negative controls")
    ev = E.load_dir(FIX, "b0_v1", exps=["b0_v1"])
    diags, by, res = run(ev)
    check("b0_v1: none of D1-D4 (or D3b) fires", not any(by[x]["fired"] for x in FAILURE_IDS + ("D4",)),
          [(x, by[x]["summary"]) for x in FAILURE_IDS + ("D4",) if by[x]["fired"]])
    check("b0_v1: nothing is proposed", not res["proposals"] and not res["cards"], res)
    ev = E.load_dir(FIX, "base_b_v1", exps=["base_b_v1"])
    diags, by, res = run(ev)
    check("base_b_v1: none of D1-D4, D15, D16 fires, nothing proposed",
          not any(by[x]["fired"] for x in FAILURE_IDS + ("D4",)) and not res["proposals"] and not res["cards"],
          [(x, by[x]["summary"]) for x in FAILURE_IDS + ("D4",) if by[x]["fired"]])
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    diags, by, res = run(ev)
    check("pilot_v1: D15 and D16 silent (no flips-only rejection of a truth-helps step; gate v1)",
          not by["D15"]["fired"] and not by["D16"]["fired"], by["D15"]["summary"])
    inc, exp, st = healthy_pilot()
    check("the synthetic pilot reached done", st["done"], {r: c["phase"] for r, c in st["chains"].items()})
    ev = E.load_dir(inc, exp)
    diags, by, res = run(ev)
    rep = ev.json("%s/report.json" % exp)
    check("it tracks truth: every chain agrees on every step", all(a["agree"] == a["compared"] == 4
                                                                    for a in rep["agreement"].values()),
          rep["agreement"])
    check("its chains accept the clean steps and reject Bswap, attributed to labels",
          all(c["accepted"] == ["I1", "I2", "I3"] and c["bswap"]["attributed_to_labels"]
              for c in rep["chains"].values()),
          {r: (c["accepted"], c["bswap"]) for r, c in rep["chains"].items()})
    check("D1, D2, D3, D3b, D15 and D16 stay silent", not any(by[x]["fired"] for x in FAILURE_IDS),
          [(x, by[x]["summary"]) for x in FAILURE_IDS if by[x]["fired"]])
    check("D4 reaches its positive outcome, decision_slot_ready (not no_recipe_tracks_truth)",
          by["D4"]["fired"] and by["D4"]["name"] == "decision_slot_ready", by["D4"]["summary"])
    check("the loader never opened the run directories", all(E.allowed(n) for n in ev.touched)
          and not any("/runs/" in n for n in ev.touched), ev.touched)
    check("no build is proposed without Step 1: L2 is deferred", not res["proposals"]
          and any(x["lever"] == "L2" for x in res["deferred"]), (res["proposals"], res["deferred"]))
    return inc, exp


# ---------------------------------------------------------- test blindness
def blind_case(name, src_root, exp, exps):
    a = snapshot_copy(src_root, exps, TMP / ("blind_%s_a" % name))
    b = snapshot_copy(src_root, exps, TMP / ("blind_%s_b" % name))
    n_changed = 0
    for e in exps:
        for f in E.EXP_FILES:
            p = b / e / f
            if p.is_file() and p.suffix == ".json":
                obj = json.loads(p.read_text())
                new = perturb_non_dev(obj)
                n_changed += json.dumps(new) != json.dumps(obj)
                p.write_text(json.dumps(new, indent=1))
    ea, eb = E.load_dir(a, exp, exps=exps), E.load_dir(b, exp, exps=exps)
    da, db = DG.detect(ea), DG.detect(eb)
    pa, pb = LV.stable(LV.propose(da, ea)), LV.stable(LV.propose(db, eb))
    has_final = any("final" in (ea.json("%s/report.json" % e) or {}) for e in exps)
    if has_final:
        check("%s: the perturbation touched a file" % name, n_changed >= 1, n_changed)
    check("%s: diagnoses byte-identical" % name, json.dumps(da, sort_keys=True) == json.dumps(db, sort_keys=True))
    check("%s: proposals, cards and argv byte-identical" % name,
          json.dumps(pa, sort_keys=True) == json.dumps(pb, sort_keys=True))
    check("%s: canonical evidence bytes identical" % name, ea.canonical() == eb.canonical())
    return a, ea, da


def test_blindness(inc, exp):
    print("test blindness (metamorphic)")
    blind_case("pilot_v1", FIX, "pilot_v1", ["pilot_v1", "b0_v1"])
    _a, ev2, d2 = blind_case("pilot_v2", FIX, "pilot_v2", WHOLE)
    check("pilot_v2 (whole tree): the perturbed run still gives R5 (D15 -> L9)",
          DG.by_id(d2)["D15"]["fired"] and [p["argv"] for p in LV.propose(d2, ev2)["proposals"]
                                            if p["lever"] == "L9"] == [L9_ARGV])
    _a, ev3, d3 = blind_case("pilot_v3", FIX, "pilot_v3", WHOLE_V3)
    b3 = DG.by_id(d3)
    check("pilot_v3 (whole tree): the perturbed run still gives R6 (D1 not blocking, D4 ready on full)",
          b3["D1"]["fired"] and b3["D1"]["detail"].get("blocks_d4") is False and b3["D4"]["levers"] == ["L2"]
          and b3["D4"]["detail"].get("recipes") == ["full"])
    blind_case("b0_v1", FIX, "b0_v1", ["b0_v1"])
    root, ev, diags = blind_case("synthetic", inc, exp, [exp])
    p = root / exp / "report.json"
    rep = json.loads(p.read_text())
    j = next(i for i, r in enumerate(rep["final"]) if r["model"].startswith("chain full"))
    rep["final"][j]["exams"]["dev"]["twelve"]["mean"] += 0.05
    p.write_text(json.dumps(rep))
    ev2 = E.load_dir(root, exp, exps=[exp])
    d2 = DG.detect(ev2)
    check("a perturbed dev value does change the diagnoses (the comparison has teeth)",
          json.dumps(d2, sort_keys=True) != json.dumps(diags, sort_keys=True))


# ------------------------------------------------ the live path (remote.py)
def test_remote_snapshot():
    print("the live path: remote.py's snapshot of the same files gives the same decisions")
    try:
        from weed_optimizer_framework.tools.inc_autopilot import remote as RM
        snap = RM.snapshot
    except (ImportError, AttributeError) as e:
        skip("remote snapshot", "remote.py is not importable here (%s)" % e)
        return
    from weed_optimizer_framework.tools.inc import common as C
    for e in ("pilot_v1", "b0_v1"):
        shutil.copytree(FIX / e, pathlib.Path(C.INC_DIR) / e)
    for p in pathlib.Path(C.INC_DIR).rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    rec = snap("pilot_v1")
    check("remote.snapshot is ok and ships report.json without its final table",
          rec.get("ok") and "final" not in rec["decision"]["artifacts"].get("pilot_v1/report.json", {"final": 1}),
          rec.get("error"))
    ev_s = E.from_snapshot(rec, "pilot_v1")
    ev_f = E.load_dir(FIX, "pilot_v1", exps=["pilot_v1"])
    ds, df = DG.detect(ev_s), DG.detect(ev_f)
    check("the same diagnoses from the INCAP record as from the files",
          json.dumps(ds, sort_keys=True) == json.dumps(df, sort_keys=True),
          [(a["id"], a["summary"][:80]) for a, b in zip(ds, df) if a != b])
    check("the same proposals and argv", json.dumps(LV.stable(LV.propose(ds, ev_s)), sort_keys=True)
          == json.dumps(LV.stable(LV.propose(df, ev_f)), sort_keys=True))
    shutil.copytree(FIX / "pilot_v2", pathlib.Path(C.INC_DIR) / "pilot_v2")
    for p in (pathlib.Path(C.INC_DIR) / "pilot_v2").rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    rec = snap("pilot_v2")
    ev_s = E.from_snapshot(rec, "pilot_v2")
    ev_f = E.load_dir(FIX, "pilot_v2", exps=["pilot_v2"])
    ds, df = DG.detect(ev_s), DG.detect(ev_f)
    check("pilot_v2 through remote.snapshot: the same diagnoses (D15 read off the shipped ledger)",
          rec.get("ok") and json.dumps(ds, sort_keys=True) == json.dumps(df, sort_keys=True)
          and DG.by_id(ds)["D15"]["fired"], [(a["id"], a["summary"][:80]) for a, b in zip(ds, df) if a != b])
    check("  and the same L9", [p["argv"] for p in LV.propose(ds, ev_s)["proposals"] if p["lever"] == "L9"]
          == [L9_ARGV])


# ------------------------------------------------------------ earliest fire
def test_earliest_fire():
    print("earliest fire (pilot_v1 ledger prefixes)")
    lines = (FIX / "pilot_v1" / "ledger.jsonl").read_text().splitlines(True)
    exp_text = (FIX / "pilot_v1" / "exp.json").read_text()
    rep = json.loads((FIX / "pilot_v1" / "report.json").read_text())
    first = None
    for n in range(1, len(lines) + 1):
        ev = E.from_texts({"pilot_v1/exp.json": exp_text, "pilot_v1/ledger.jsonl": "".join(lines[:n])}, "pilot_v1")
        if DG.by_id(DG.detect(ev, only=("D1",)))["D1"]["fired"]:
            first = n
            break
    check("D1 fires by the end of the ledger", first is not None)
    if first is None:
        return
    seen = [json.loads(x) for x in lines[:first]]
    gates = [e for e in seen if e.get("type") == "gate"]
    truths = [e for e in seen if e.get("type") == "truth"]
    per_chain = collections.Counter(e["chain"] for e in gates)
    gh = rep["gpu_hours"]
    steps = len(rep["steps"])
    saved = sum(gh["chain:%s" % r]["hours"] * (steps - per_chain.get(r, 0)) / float(steps) for r in rep["agreement"])
    saved += gh["truth"]["hours"] * (steps - len(truths)) / float(steps) + gh["final"]["hours"]
    print("  info D1 first fires at ledger line %d (%s), after %d of 21 gate entries and %d of 7 truth steps;"
          % (first, seen[-1]["id"], len(gates), len(truths)))
    print("  info the rest of pilot_v1 then cost about %.1f of its %.1f GPU-hours (estimate from the report's "
          "per-arm hours)" % (saved, rep["gpu_hours_total"]))


def main():
    try:
        test_r1_r3_r4a()
        test_lineage()
        test_r2()
        test_r4b()
        test_r5()
        test_r6()
        test_r7()
        test_r8()
        inc, exp = test_negative_controls()
        test_blindness(inc, exp)
        test_remote_snapshot()
        test_earliest_fire()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
