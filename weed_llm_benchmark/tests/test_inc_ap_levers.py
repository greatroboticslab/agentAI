#!/usr/bin/env python3
"""The INC autopilot's levers render exactly the commands a person runs by
hand, inside their bounds (docs/INC_AUTOPILOT.md, (b) lever catalogue).

Pinned:
  * the menu: every in-menu lever has a command, a policy action, a risk in
    R0-R3, param_bounds in policy_actions.json's form, a control, a success
    criterion and a falsifier; every card (X1-X9) is R4 with no command; no
    lever takes a source list, a gate threshold or a verdict; every refusal
    pattern compiles and names a menu lever or none; the protocol constants
    equal select.INC_FRAC, realloop.N_VERIFIED and the driver's seeds;
  * byte equality with the documented commands:
      L1  python -m weed_optimizer_framework.tools.inc.pilot build --exp <child> --replay-mode full
      L2  python -m weed_optimizer_framework.tools.inc.realloop build --exp <n> --base <base_B>
          --replay-mode <m> --recipes <r,...> [--relevance P] [--size S] [--n-verified N]
      L3  the command in run_inc_relevance.sh's header
      L4  the pilot command in run_inc_audit.sh's header, parsed from the file (its $REPO, $INC and
          $M expanded), built from pilot_v1's own exp.json
      L5/L6  L2 with --size / --no-truth; L7 driver unblock ... --reason "auto: <cause>";
      L8  pilot build-baseline --exp base_b_v1 --manifest <INC>/step1/base_B.jsonl;
  * each builder's real argparse accepts the rendered argv and hands its
    dispatch function the intended values (the dispatch is replaced, so
    nothing is built);
  * bounds: bad experiment names, replay modes, recipe subsets, sizes,
    paths, units, causes, costs and undeclared params are refused; a fixed
    flag cannot be changed; L5 without a size is refused;
  * child names follow the manual one (pilot_v1 -> pilot_v2) and never
    reuse an experiment;
  * cost: rates measured in pilot_v1; a sample-replay rebuild of pilot_v1
    prices at exactly its measured 17.28 GPU-hours; the full-replay rebuild
    (L1) at 30-50 (the contract's estimate); a real loop from Step 1's base
    size, with and without the truth arm; the job scripts' walltimes;
  * propose(): pilot_v1 gives L1 (D1, D4, D9) and L4 (D3) and the cards X1
    and X4, nothing else; a ready pilot gives L2 with the chosen recipe and
    --relevance when relevance.json matches the select build, and defers it
    otherwise (missing: L3 first; stale or refused by relevance.load's
    cheap checks: a person); a ready pilot on which D1 also fires gives L1,
    not L2; a transient block gives L7; a stale advance an advance
    operation; D9's support joins only a proposal on its own experiment;
  * lineage: a lever is applied to a parent once. A full-replay pilot
    initialised after pilot_v1 defers L1 (a sample one does not, and the
    name steps past it); a real loop of D4's decision defers L2 (and L6),
    unless it was built on another Step 1 base; a context lineage record
    defers its lever while the build is in flight, a failed one does not;
    base_b_v1 defers L8;
  * real loops (a realloop-shaped exp.json and build_summary.json): L5 from
    D8 takes --base from base.source_manifest (Step 1's base_B.jsonl, not
    the experiment's own copy), --n-verified from build_summary.json,
    --relevance and --no-truth from the parent, and realloop's own argparse
    reads it back; L2 from D1 on a sample loop keeps its size, number of
    increments and relevance with --replay-mode full; a later loop of the
    same kind defers each; a stale relevance defers them;
  * the relevance criterion (docs/INCREMENTAL_PROTOCOL.md, Steps 2-3):
    relevance_status tells a file made for this select build whose own
    calibration check failed ('calibration_failed') from a stale or a
    malformed one; with that file, L2 takes --increment-sources evidence
    (no --relevance) when the evidenced pool (evidence_capacity, the same
    evidenced sources select.load_evidence finds) holds (N + 1) x M images;
    when it cannot hold the default N and M, the R4 sizing rule
    (evidence_sizing: N 4 for six decided increments, M = min(393,
    floor(images / 5))) renders --size and --n-verified in the template's
    order (1,439 images: N 4 x M 287), and a sized M below 5 % of B (980
    images: 196 < 196.35) or an evidence that cannot be read defers with
    the numbers; a given --size or --n-verified is checked as given; the
    rule's constants are the protocol's acceptance and below inc_frac; L3 is
    never proposed over it; an evidence loop's rebuild (L5) keeps the
    criterion, and realloop's own argparse reads every such command back;
    increment_sources is an enum of select.SOURCE_MODES on L2, L5 and L6,
    rendered in the slot --relevance takes;
  * price(): the proposals' prices are price()'s, and it refuses to guess;
  * the gate: L9 on pilot_v2 is exactly 'inc.pilot build --exp pilot_v3
    --replay-mode full --gate-flips-mode net', ranked first, from D15, with
    the five flips-only ledger lines among its cites; no L1 or realloop, X1
    a card (D1 unblocked); a later net pilot defers it (applied cites its
    gate), a later v1 pilot does not, an in-flight lineage record does, a
    failed one does not; a parent already on net gets card X6; L2 from D4,
    L1 and L5 (loop_params) carry a net parent's --gate-flips-mode last, a
    v1 parent none; lineage is gate-aware (a v1 loop never stands for a net
    L2, nor a pilot or loop on the other gate for L1 or L5); a parent whose
    final v2_check is refuted is rebuilt by no lever (gate_params defers;
    a provisional or supported check does not); the gate is the one gate
    setting on the menu, an enum of driver.FLIPS_MODES.

Run:  python3 tests/test_inc_ap_levers.py
"""
import json
import pathlib
import re
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as LV  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "inc_replay"
# The fixture tree as it stood at intervention 1, before pilot_v2 existed: a whole-tree
# load also holds pilot_v2 (L1 on pilot_v1 applied, D4 campaign-wide on pilot_v2).
V1_ERA = ["pilot_v1", "b0_v1"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
INC = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc"
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc=Exception, contains=None):
    try:
        fn()
    except exc as e:
        return contains is None or contains in str(e)
    return False


def documented(script, marker):
    """The first sbatch command in the comment block after `marker` in the
    job script's header, its $VARs expanded from the block's assignments."""
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

    def expand(s):
        return re.sub(r"\$([A-Z]+)", lambda m: env[m.group(1)], s)
    for st in stmts:
        for part in (p.strip() for p in st.split(";")):
            m = re.fullmatch(r"([A-Z]+)=(\S+)", part)
            if m:
                env[m.group(1)] = expand(m.group(2))
            elif part.startswith("sbatch ") and cmd is None:
                cmd = part
    return [expand(t) for t in shlex.split(cmd, comments=True)], env


def capture(mod, attr, call, argv):
    got = {}
    real = getattr(mod, attr)

    def fake(*a, **k):
        got["args"], got["kwargs"] = a, k
    setattr(mod, attr, fake)
    try:
        rc = call(argv)
    finally:
        setattr(mod, attr, real)
    return rc, got


# The rule a relevance file is made under (relevance.load: MIN_CROPS, CAL_PERCENTILE, TAU_MIN).
RULE = {"min_crops": 20, "calibration_percentile": 5.0, "tau_min": 0.5}
# A consistent passing check: ok true, tau_min 0.5, tau >= 0.5.
RELEVANCE = {"format": "inc.relevance/1", "params": dict(RULE),
             "calibration": {"tau": 0.62, "check": {"ok": True, "tau_min": 0.5}},
             # the summary records no base_selected.jsonl, so only increment_pool is compared
             "inputs": {"increment_pool": {"sha256": "a" * 64}}, "increment_pool": {"sources": {}}}
SELECT = {"sizes": {"base_B": 3927},
          "outputs": {"increment_pool.jsonl": {"sha256": "a" * 64}, "base_B.jsonl": {"sha256": "b" * 64}}}


def ready_world():
    """pilot_v1 whose report has lora at 6/7 and whose gate entries leave
    P_recipe at 0.5 (so D1 is silent), with Step 1 and a matching relevance."""
    rep = json.loads((FIX / "pilot_v1" / "report.json").read_text())
    rep["agreement"]["lora"] = {"agree": 6, "compared": 7, "rate": 6 / 7.0}
    lines = []
    for ln in (FIX / "pilot_v1" / "ledger.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if e.get("type") == "gate":
            e["decision"]["p_recipe"] = 0.5
        lines.append(json.dumps(e))
    return {"pilot_v1/exp.json": (FIX / "pilot_v1/exp.json").read_text(), "pilot_v1/report.json": json.dumps(rep),
            "pilot_v1/ledger.jsonl": "\n".join(lines) + "\n",
            "step1/select_summary.json": json.dumps(SELECT), "step1/relevance.json": json.dumps(RELEVANCE)}


def child_pilot(name="pilot_v2", replay_mode="full", utc="2026-09-28T00:00:00Z"):
    """pilot_v1's exp.json as a later pilot (built, running: no report)."""
    d = json.loads((FIX / "pilot_v1" / "exp.json").read_text())
    d.update(exp=name, replay_mode=replay_mode, initialised_utc=utc)
    d["base"]["manifest"] = d["base"]["manifest"].replace("pilot_v1", name)
    for st in d["steps"]:
        st["manifest"] = st["manifest"].replace("pilot_v1", name)
    return json.dumps(d)


def loop_defn(exp="realloop_v1", replay_mode="full", recipes=("full",), size=393, n_verified=4, truth=True,
              utc="2026-10-01T00:00:00Z", base_sha="b" * 64):
    """A realloop-shaped exp.json (realloop.build's defn keys)."""
    seq = ["V1", "V2", "UNVERIFIED", "V3", "OTHER_HEAVY"] + ["V%d" % i for i in range(4, n_verified + 1)]
    kind = {"UNVERIFIED": "unverified", "OTHER_HEAVY": "otherplant_heavy"}
    return {"exp": exp, "type": "chain", "builder": "inc.realloop build", "testing": False,
            "replay_mode": replay_mode, "seeds": [0, 1, 2], "decision_exam": "dev", "initialised_utc": utc,
            "base": {"name": "base_B", "manifest": "%s/%s/manifests/base_B.jsonl" % (INC, exp),
                     "manifest_sha256": base_sha, "n_images": 3927, "recipe": {"epochs": 100},
                     "source_manifest": INC + "/step1/base_B.jsonl"},
            "steps": [{"name": n, "clean": n != "UNVERIFIED", "kind": kind.get(n, "verified"), "n_images": size,
                       "manifest": "%s/%s/manifests/%s.jsonl" % (INC, exp, n)} for n in seq],
            "recipes": {r: {"epochs": 30} for r in recipes}, "truth": truth, "truth_recipe": {"epochs": 100},
            "increment_images": size,
            "step1": {"relevance": {"path": INC + "/step1/relevance.json", "sha256": "e" * 64}}}


def loop_world(defn, gates, truths=(), build_summary=None):
    """pilot_v1 (the measured rates), Step 1, a matching relevance and the loop."""
    exp = defn["exp"]
    objs = {"pilot_v1/exp.json": (FIX / "pilot_v1/exp.json").read_text(),
            "pilot_v1/report.json": (FIX / "pilot_v1/report.json").read_text(),
            "step1/select_summary.json": json.dumps(SELECT), "step1/relevance.json": json.dumps(RELEVANCE),
            "%s/exp.json" % exp: json.dumps(defn),
            "%s/ledger.jsonl" % exp: "".join(json.dumps(e) + "\n" for e in list(truths) + list(gates))}
    if build_summary is not None:
        objs["%s/build_summary.json" % exp] = json.dumps(build_summary)
    return objs


def loop_gate(k, step, verdict="HOLD", p_recipe=0.5, p_data=0.5, cand=0.700, null=0.699, inc=0.69):
    return {"id": "gate/full/%d" % k, "type": "gate", "chain": "full", "k": k, "step": step, "clean": True,
            "attribution_not_run": {"label_audit": "not run"},
            "decision": {"verdict": verdict, "p_data": p_data, "p_recipe": p_recipe, "cand_mean": cand,
                         "null_mean": null, "inc": inc, "cand_sd": 0.005, "null_sd": 0.005,
                         "attribution": {"class_vs_loc": "none"}}}


def test_menu():
    print("the menu")
    menu = LV.load_menu()
    from weed_optimizer_framework.tools.brain import policy
    lv = menu["levers"]
    # L10-L14 (and L11a, the card fetch L11 reads) are the funnel audit's levers
    # (docs/FUNNEL_AUDIT.md 8.5); L14 is a lab hook with no command.
    check("L1-L14 and L11a are on the menu",
          sorted(lv) == sorted(["L%d" % i for i in range(1, 15)] + ["L11a"]), sorted(lv))
    for lid, r in sorted(lv.items()):
        command = (isinstance(r.get("argv"), list) and r["argv"]) or \
            (r.get("kind") == "lab_hook" and r.get("argv") is None and isinstance(r.get("hook"), str))
        good = (command and r.get("policy_action", "").startswith("inc_")
                and r.get("risk") in ("R0", "R1", "R2", "R3") and isinstance(r.get("param_bounds"), dict)
                and all(isinstance(r.get(k), str) and r[k] for k in ("control", "success", "falsifier", "title"))
                and r.get("estimator") in ("pilot_rebuild", "realloop", "walltime", "zero", "baseline",
                                           "funnel_walltime"))
        check("%s: command, action, risk, bounds, control, success, falsifier, estimator" % lid, good, r.keys())
        ok, why = policy._check_params({k: 0 for k in []}, r["param_bounds"])
        kinds = {b.get("type") for b in r["param_bounds"].values()}
        check("%s: bounds use policy's types" % lid, ok and kinds <= {"str", "int", "float", "enum"}, kinds)
        check("%s: est_gpu_hours is bounded in [0, 200]" % lid,
              r["param_bounds"].get("est_gpu_hours") == {"type": "float", "min": 0.0, "max": 200.0})
    check("risks: builds R3, relevance/audit/unblock R2; the funnel: audit and maps R2, lab fetches R0, recovery R3, "
          "the verify queue R2",
          {k: v["risk"] for k, v in lv.items()} == {"L1": "R3", "L2": "R3", "L3": "R2", "L4": "R2", "L5": "R3",
                                                     "L6": "R3", "L7": "R2", "L8": "R3", "L9": "R3",
                                                     "L10": "R2", "L11": "R2", "L11a": "R0", "L12": "R0",
                                                     "L13": "R3", "L14": "R2"})
    cards = menu["cards"]
    check("X1-X12 are R4 cards with no command",
          sorted(cards) == sorted("X%d" % i for i in range(1, 13))
          and all(c["risk"] == "R4" and "argv" not in c and c.get("required_change") for c in cards.values()))
    names = {p for r in lv.values() for p in r["param_bounds"]}
    # The gate's flips mode is the one gate setting on the menu: the choice between the two
    # pre-registered protocol versions (levers.json _meta.gate_flips_mode), never a threshold.
    # increment_sources is the choice between the two pre-registered relevance criteria
    # (levers.json _meta.increment_sources), an enum that names no source.
    forbidden = [n for n in names if re.search(r"exclu|source|threshold|p_accept|p_reject|verdict|gate", n)
                 and n not in ("gate_flips_mode", "increment_sources")]
    check("no lever takes a source list, a gate threshold or a verdict", not forbidden, forbidden)
    from weed_optimizer_framework.tools.inc import select as S0
    crit = {k: r["param_bounds"]["increment_sources"] for k, r in lv.items() if "increment_sources" in r["param_bounds"]}
    from weed_optimizer_framework.tools.inc import realloop as RL0
    check("increment_sources is an enum of select.SOURCE_MODES on the real-loop levers only (L2, L5, L6), L2 adds "
          "the funnel's recovered overlay (realloop.INCREMENT_SOURCE_MODES, realloop_v2), and the protocol's modes "
          "are realloop's, its default and min_evidence select's",
          sorted(crit) == ["L2", "L5", "L6"]
          and all(crit[k] == {"type": "enum", "value_type": "str", "values": list(S0.SOURCE_MODES)} for k in ("L5", "L6"))
          and crit["L2"] == {"type": "enum", "value_type": "str", "values": list(RL0.INCREMENT_SOURCE_MODES)}
          and LV.protocol("increment_sources_modes") == list(RL0.INCREMENT_SOURCE_MODES)
          and LV.protocol("recovered_steps") == len(RL0.RECOVERED_SEQUENCE)
          and LV.protocol("increment_sources_default") == S0.SOURCES_RELEVANCE
          and (LV.SOURCES_RELEVANCE, LV.SOURCES_EVIDENCE) == (S0.SOURCES_RELEVANCE, S0.SOURCES_EVIDENCE)
          and LV.protocol("min_evidence_default") == S0.MIN_EVIDENCE, crit)
    from weed_optimizer_framework.tools.inc import driver as D0
    modes = [r["param_bounds"]["gate_flips_mode"] for r in lv.values() if "gate_flips_mode" in r["param_bounds"]]
    check("gate_flips_mode is an enum of the pre-registered flips modes only (driver.FLIPS_MODES)",
          modes and all(m.get("type") == "enum" and m.get("value_type") == "str"
                        and set(m["values"]) <= set(D0.FLIPS_MODES) for m in modes)
          and sorted(k for k, r in lv.items() if "gate_flips_mode" in r["param_bounds"])
          == ["L1", "L2", "L5", "L6", "L9"], modes)
    check("the protocol's flips modes and default are the driver's",
          LV.protocol("gate_flips_modes") == list(D0.FLIPS_MODES)
          and LV.protocol("gate_flips_mode_default") == D0.DEFAULT_FLIPS_MODE)
    l9 = lv["L9"]
    check("L9: inc_build_pilot, R3, fixes --gate-flips-mode net, states --replay-mode, only after D15",
          l9["policy_action"] == "inc_build_pilot" and l9["fixed"] == {"gate_flips_mode": "net"}
          and "{replay_mode}" in l9["argv"] and l9["only_after"] == ["D15"]
          and "supported" in l9["success"] and "5 of 7" in l9["success"] and "refuted" in l9["falsifier"],
          l9)
    for r in menu["refusals"]:
        re.compile(r["pattern"])
    check("every refusal names a menu lever or none (with what it needs)",
          all((r["prerequisite"] in lv) or (r["prerequisite"] is None and r.get("needs")) for r in menu["refusals"]))
    no_retry = [r for r in menu["refusals"] if r.get("retry") is False]
    check("the refusals the identical build meets again (retry false) are realloop's evidenced-pool capacity and "
          "select's draw shortfall, each a person's with its retry_why; every other entry leaves retry unset",
          len(no_retry) == 2
          and no_retry[0]["pattern"].startswith("increment sources 'evidence': the evidenced increment pool")
          and no_retry[1]["pattern"].endswith("the draw found \\d+")
          and all(r["prerequisite"] is None and r.get("retry_why") for r in no_retry)
          and all("retry" not in r for r in menu["refusals"] if r not in no_retry),
          [r["pattern"] for r in no_retry])
    from weed_optimizer_framework.tools.inc import driver as D
    from weed_optimizer_framework.tools.inc import realloop as RL
    from weed_optimizer_framework.tools.inc import select as S
    from weed_optimizer_framework.tools.inc import relevance as REL
    check("protocol constants equal their modules",
          (LV.protocol("inc_frac"), LV.protocol("n_verified_default"), LV.protocol("seeds"),
           LV.protocol("relevance_format"), LV.KIND_VERIFIED, LV.protocol("relevance_tau_min"),
           LV.protocol("relevance_min_crops"), LV.protocol("relevance_calibration_percentile"))
          == (S.INC_FRAC, RL.N_VERIFIED, len(D.SEEDS), REL.FORMAT, RL.KIND_VERIFIED, REL.TAU_MIN,
              REL.MIN_CROPS, REL.CAL_PERCENTILE))
    proto_doc = (ROOT.parent / "docs" / "INCREMENTAL_PROTOCOL.md").read_text()
    m_acc = re.search(r"^- at least (\d+) increments decided;$", proto_doc, re.M)
    check("the sizing rule's constants: min_decided_increments is Steps 2-3's acceptance ('at least 6 increments "
          "decided'), which the default loop meets (realloop.sequence(N_VERIFIED)); min_increment_frac 0.05 is below "
          "inc_frac; each carries its why",
          m_acc is not None and LV.protocol("min_decided_increments") == int(m_acc.group(1))
          and len(RL.sequence(RL.N_VERIFIED)) >= LV.protocol("min_decided_increments")
          and 0 < LV.protocol("min_increment_frac") == 0.05 < S.INC_FRAC
          and all(menu["protocol"][k].get("why") for k in ("min_decided_increments", "min_increment_frac")),
          (m_acc and m_acc.group(0), LV.protocol("min_decided_increments"), LV.protocol("min_increment_frac")))
    check("the test's relevance rule is relevance.py's (so RELEVANCE is a file relevance.load reads as passing)",
          RULE == {"min_crops": REL.MIN_CROPS, "calibration_percentile": REL.CAL_PERCENTILE, "tau_min": REL.TAU_MIN}
          and REL.check_of(RELEVANCE["calibration"]["tau"]) is True)
    rows = []
    for lid, r in sorted(lv.items()):
        d = policy.describe(r["policy_action"])
        if d.get("known"):
            rows.append(lid)
            check("%s: the policy row %s has the same risk" % (lid, r["policy_action"]), d["risk"] == r["risk"],
                  (d["risk"], r["risk"]))
    if not rows:
        print("  note no inc_* rows in brain/policy_actions.json yet; risk agreement not checked")


def test_argv():
    print("commands, byte for byte")
    a = LV.argv("L1", {"exp": "pilot_v2", "replay_mode": "full"})
    check("L1 is the manual command",
          a == "python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v2 --replay-mode full".split(), a)
    base, rel = INC + "/step1/base_B.jsonl", INC + "/step1/relevance.json"
    a = LV.argv("L2", {"exp": "realloop_v1", "base": base, "replay_mode": "full", "recipes": "full,lora",
                       "relevance": rel, "size": 400, "n_verified": 6})
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base %s "
            "--replay-mode full --recipes full,lora --relevance %s --size 400 --n-verified 6" % (base, rel)).split()
    check("L2 with every option is the documented command", a == want, a)
    a = LV.argv("L2", {"exp": "realloop_v1", "base": base, "replay_mode": "sample", "recipes": "freeze"})
    check("L2 without options", a == ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 "
                                      "--base %s --replay-mode sample --recipes freeze" % base).split(), a)
    doc, _ = documented("run_inc_relevance.sh", "Submit (Slurm opens the log file")
    check("L3 is run_inc_relevance.sh's documented command", LV.argv("L3", {}) == doc, (LV.argv("L3", {}), doc))
    doc, env = documented("run_inc_audit.sh", "The pilot's post-hoc item 4")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    derived, cites = LV.audit_args(ev, "pilot_v1")
    a = LV.argv("L4", {"exp": "pilot_v1"}, derived)
    check("L4 from pilot_v1's exp.json is run_inc_audit.sh's documented command, byte for byte", a == doc,
          "\n%s\n%s" % (a, doc))
    check("the header's INC is model.CLUSTER_INC_DIR and the one derived from exp.json",
          env["INC"] == M.CLUSTER_INC_DIR == LV.inc_dir(ev, "pilot_v1"), env)
    check("L4's derived paths cite exp.json", all(ev.check_cite(c) for c in cites) and len(cites) == 8)
    a = LV.argv("L5", {"exp": "realloop_v2", "base": base, "replay_mode": "full", "recipes": "full", "size": 786})
    check("L5 is L2 with --size", a == ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v2 "
                                        "--base %s --replay-mode full --recipes full --size 786" % base).split(), a)
    a = LV.argv("L6", {"exp": "realloop_v2", "base": base, "replay_mode": "full", "recipes": "full",
                       "relevance": rel, "no_truth": 1})
    check("L6 is L2 with --no-truth", a[-1] == "--no-truth" and a[:-1] == LV.argv(
        "L2", {"exp": "realloop_v2", "base": base, "replay_mode": "full", "recipes": "full", "relevance": rel}))
    a = LV.argv("L7", {"exp": "pilot_v2", "unit": "chain:full", "cause": "transient"})
    check("L7 is driver unblock with an 'auto:' reason",
          a == ["python", "-m", "weed_optimizer_framework.tools.inc.driver", "unblock", "--exp", "pilot_v2",
                "--unit", "chain:full", "--reason", "auto: transient"], a)
    a = LV.argv("L8", {"exp": "base_b_v1", "manifest": base})
    check("L8 is pilot build-baseline on base_B", a == ("python -m weed_optimizer_framework.tools.inc.pilot "
                                                         "build-baseline --exp base_b_v1 --manifest %s" % base).split())
    a = LV.argv("OP_ADVANCE", {"exp": "pilot_v2"})
    check("the advance operation", a == "python -m weed_optimizer_framework.tools.inc.driver advance --exp pilot_v2"
          .split(), a)
    a = LV.argv("L9", {"exp": "pilot_v3", "replay_mode": "full", "gate_flips_mode": "net"})
    check("L9 is pilot build with the parent's replay mode and --gate-flips-mode net",
          a == ("python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v3 --replay-mode full "
                "--gate-flips-mode net").split(), a)
    a = LV.argv("L1", {"exp": "pilot_v4", "replay_mode": "full", "gate_flips_mode": "net"})
    check("L1 of a v2 parent carries --gate-flips-mode net (last)",
          a[-2:] == ["--gate-flips-mode", "net"] and a[:-2] == LV.argv("L1", {"exp": "pilot_v4"}), a)
    for lid, extra in (("L2", {}), ("L5", {"size": 786}), ("L6", {"no_truth": 1})):
        p = dict({"exp": "realloop_v2", "base": base, "replay_mode": "full", "recipes": "full", "relevance": rel,
                  "n_verified": 6}, **extra)
        a = LV.argv(lid, dict(p, gate_flips_mode="net"))
        check("%s carries --gate-flips-mode last (the executor's rendering order)" % lid,
              a[-2:] == ["--gate-flips-mode", "net"] and a[:-2] == LV.argv(lid, p), a)
        q = dict({k: v for k, v in p.items() if k != "relevance"}, increment_sources="evidence")
        a = LV.argv(lid, dict(q, gate_flips_mode="net"))
        i = a.index("--recipes") + 2
        check("%s on source-level species evidence: --increment-sources evidence in the slot --relevance takes, "
              "--gate-flips-mode still last" % lid,
              a[i:i + 2] == ["--increment-sources", "evidence"] and "--relevance" not in a
              and a[:i] + a[i + 2:] == LV.argv(lid, dict({k: v for k, v in q.items() if k != "increment_sources"},
                                                         gate_flips_mode="net")), a)
    a = LV.argv("L2", {"exp": "realloop_v1", "base": base, "replay_mode": "full", "recipes": "full",
                       "increment_sources": "evidence", "gate_flips_mode": "net"})
    check("L2 on evidence from a v2 pilot is R7's command",
          a == ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base %s "
                "--replay-mode full --recipes full --increment-sources evidence --gate-flips-mode net" % base).split(),
          a)
    check("increment_sources outside the enum is refused",
          raises(lambda: LV.argv("L2", {"exp": "r", "base": base, "replay_mode": "full", "recipes": "full",
                                        "increment_sources": "hand_list"}), LV.LeverError))
    from weed_optimizer_framework.tools.inc_autopilot import model as M0
    pair = {"exp": "realloop_v1", "base": base, "replay_mode": "full", "recipes": "full",
            "increment_sources": "evidence", "relevance": rel}
    for lid, extra in (("L2", {}), ("L5", {"size": 786}), ("L6", {"no_truth": 1})):
        ok, why = LV.check_params(lid, dict(pair, **extra))
        check("%s: --increment-sources evidence with --relevance is refused by check_params (realloop refuses the "
              "pair; remote.py's message) and argv never renders it" % lid,
              not ok and M0.EVIDENCE_WITH_RELEVANCE in why
              and raises(lambda: LV.argv(lid, dict(pair, **extra)), LV.LeverError, "does not go with it"), why)
    check("... --increment-sources relevance with --relevance is the builders' default pair, accepted",
          LV.check_params("L2", dict(pair, increment_sources="relevance"))[0])


def test_real_argparse():
    print("each builder's own argparse accepts the commands")
    from weed_optimizer_framework.tools.inc import audit as AU
    from weed_optimizer_framework.tools.inc import driver as D
    from weed_optimizer_framework.tools.inc import pilot as P
    from weed_optimizer_framework.tools.inc import realloop as RL
    from weed_optimizer_framework.tools.inc import relevance as REL
    a = LV.argv("L1", {"exp": "pilot_v2", "replay_mode": "full"})
    rc, got = capture(P, "build_pilot", P.main, a[3:])
    check("L1: pilot.main calls build_pilot('pilot_v2', replay_mode='full', testing=False)",
          rc == 0 and got["args"] == ("pilot_v2",) and got["kwargs"].get("replay_mode") == "full"
          and got["kwargs"].get("testing") is False, got)
    base, rel = INC + "/step1/base_B.jsonl", INC + "/step1/relevance.json"
    a = LV.argv("L2", {"exp": "realloop_v1", "base": base, "replay_mode": "full", "recipes": "full,lora",
                       "relevance": rel, "size": 400, "n_verified": 5})
    rc, got = capture(RL, "build", RL.main, a[3:])
    k = got.get("kwargs", {})
    check("L2: realloop.main calls build with every value", rc == 0 and got["args"] == ("realloop_v1",)
          and (k["base"], k["replay_mode"], k["recipes"], k["relevance"], k["size"], k["n_verified"], k["truth"])
          == (base, "full", "full,lora", rel, 400, 5, True), got)
    a = LV.argv("L2", {"exp": "realloop_v1", "base": base, "replay_mode": "full", "recipes": "full",
                       "increment_sources": "evidence", "size": 287, "n_verified": 4, "gate_flips_mode": "net"})
    rc, got = capture(RL, "build", RL.main, a[3:])
    k = got.get("kwargs", {})
    check("L2 on evidence: realloop.main calls build with increment_sources 'evidence', no relevance file, the "
          "default --min-evidence",
          rc == 0 and got["args"] == ("realloop_v1",)
          and (k["increment_sources"], k["relevance"], k["min_evidence"], k["size"], k["n_verified"],
               k["gate_flips_mode"]) == ("evidence", None, None, 287, 4, "net"), got)
    a = LV.argv("L6", {"exp": "realloop_v1", "base": base, "replay_mode": "sample", "recipes": "lora",
                       "no_truth": 1})
    rc, got = capture(RL, "build", RL.main, a[3:])
    check("L6: realloop.main gets truth=False", rc == 0 and got["kwargs"]["truth"] is False
          and got["kwargs"]["relevance"] is None and got["kwargs"]["size"] is None, got)
    a = LV.argv("L9", {"exp": "pilot_v3", "replay_mode": "full", "gate_flips_mode": "net"})
    rc, got = capture(P, "build_pilot", P.main, a[3:])
    check("L9: pilot.main calls build_pilot('pilot_v3', replay_mode='full', gate_flips_mode='net')",
          rc == 0 and got["args"] == ("pilot_v3",) and got["kwargs"].get("replay_mode") == "full"
          and got["kwargs"].get("gate_flips_mode") == "net", got)
    a = LV.argv("L2", {"exp": "realloop_v1", "base": base, "replay_mode": "full", "recipes": "full",
                       "relevance": rel, "gate_flips_mode": "net"})
    rc, got = capture(RL, "build", RL.main, a[3:])
    check("L2 with a v2 pilot's gate: realloop.main gets gate_flips_mode='net'",
          rc == 0 and got["kwargs"].get("gate_flips_mode") == "net", got)
    check("L2's recipe subsets all parse", all(RL.parse_recipes(v) for v in
                                               LV.row("L2")["param_bounds"]["recipes"]["values"]))
    a = LV.argv("L8", {"exp": "base_b_v1", "manifest": base})
    rc, got = capture(P, "build_baseline", P.main, a[3:])
    check("L8: pilot.main calls build_baseline('base_b_v1', base_B)", rc == 0 and got["args"] == ("base_b_v1", base),
          got)
    rec = {}

    class FakeDriver:
        def __init__(self, exp, quiet=False, **kw):
            rec["exp"] = exp

        def unblock(self, units, reason, all_units=False):
            rec.update(units=units, reason=reason, all_units=all_units)
    real = D.Driver
    D.Driver = FakeDriver
    try:
        rc = D.main(LV.argv("L7", {"exp": "pilot_v2", "unit": "chain:full", "cause": "transient"})[3:])
    finally:
        D.Driver = real
    check("L7: driver.main unblocks exactly that unit with that reason",
          rc == 0 and rec == {"exp": "pilot_v2", "units": ["chain:full"], "reason": "auto: transient",
                              "all_units": False}, rec)
    rc, got = capture(D, "advance", D.main, LV.argv("OP_ADVANCE", {"exp": "pilot_v2"})[3:])
    check("OP_ADVANCE: driver.main advances that experiment", rc == 0 and got["args"] == ("pilot_v2",), got)
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    derived, _ = LV.audit_args(ev, "pilot_v1")
    a = LV.argv("L4", {"exp": "pilot_v1"}, derived)
    ns = AU.build_parser().parse_args(a[2:])
    audits = AU.parse_audits(ns.audit)
    check("L4: audit's parser reads trusted, seven NAME=manifest pairs and out",
          ns.trusted.endswith("/pilot_v1/manifests/P0.jsonl") and len(ns.audit) == 7
          and ns.out.endswith("/pilot_v1/audit/label_audit.json") and len(audits) == 7, ns)
    ns = REL.build_parser().parse_args(LV.argv("L3", {})[2:])
    check("L3: relevance's parser reads 'build' with its defaults", ns.cmd == "build" and ns.out is None, ns)


def test_bounds():
    print("bounds")
    ok = {"exp": "realloop_v1", "base": INC + "/step1/base_B.jsonl", "replay_mode": "full", "recipes": "full"}
    for bad_exp in ("../x", "a b", "", "x" * 65, "-x", "a/b"):
        check("exp %r refused" % bad_exp, raises(lambda: LV.argv("L1", {"exp": bad_exp, "replay_mode": "full"}),
                                                 LV.LeverError))
    check("L1 fixes --replay-mode full", raises(lambda: LV.argv("L1", {"exp": "p", "replay_mode": "sample"}),
                                                LV.LeverError))
    for k, v in (("replay_mode", "fast"), ("recipes", "full,full"), ("recipes", "all"), ("recipes", ["full"]),
                 ("size", 0), ("size", 20001), ("size", "400"), ("size", True), ("n_verified", 13),
                 ("base", "/ocean/../etc/base_B.jsonl"), ("base", "relative/base_B.jsonl"),
                 ("base", INC + "/step1/base_B.json"), ("relevance", INC + "/x.jsonl"), ("est_gpu_hours", 201.0),
                 ("est_gpu_hours", float("nan")), ("foo", 1), ("no_truth", 1)):
        check("L2 %s=%r refused" % (k, v), raises(lambda: LV.argv("L2", dict(ok, **{k: v})), LV.LeverError))
    check("L5 without --size refused", raises(lambda: LV.argv("L5", ok), LV.LeverError, "requires"))
    for u, c in (("chain:../x", "transient"), ("chain:full", "failed_run"), ("run:x", "transient")):
        check("L7 unit %r cause %r refused" % (u, c),
              raises(lambda: LV.argv("L7", {"exp": "p", "unit": u, "cause": c}), LV.LeverError))
    check("a card has no command", raises(lambda: LV.argv("X1", {}), LV.LeverError))
    check("an unknown lever is refused", raises(lambda: LV.argv("L10", {}), LV.LeverError))
    ok9 = {"exp": "pilot_v3", "replay_mode": "full", "gate_flips_mode": "net"}
    check("L9 fixes --gate-flips-mode net", raises(lambda: LV.argv("L9", dict(ok9, gate_flips_mode="negative")),
                                                   LV.LeverError))
    check("L9 needs --replay-mode", raises(lambda: LV.argv("L9", {"exp": "pilot_v3", "gate_flips_mode": "net"}),
                                           LV.LeverError, "replay_mode"))
    check("L9 refuses another replay mode", raises(lambda: LV.argv("L9", dict(ok9, replay_mode="fast")),
                                                   LV.LeverError))
    check("an unknown flips mode is refused", raises(lambda: LV.argv("L1", {"exp": "p", "gate_flips_mode": "both"}),
                                                     LV.LeverError))


def test_names_and_cost():
    print("names and cost")
    check("pilot_v1 -> pilot_v2 (the manual name)", LV.child_name("pilot_v1", ["pilot_v1", "b0_v1"]) == "pilot_v2")
    check("an existing name is skipped", LV.child_name("pilot_v1", ["pilot_v2", "pilot_v3"]) == "pilot_v4")
    check("an unversioned name gets _rf", LV.child_name("pilot", []) == "pilot_rf"
          and LV.child_name("pilot", ["pilot_rf"]) == "pilot_rf2")
    check("realloop names",
          LV.realloop_name([]) == "realloop_v1" and LV.realloop_name(["realloop_v1"]) == "realloop_v2")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    rt, cites = LV.rates(ev, "pilot_v1")
    check("pilot_v1 rates: every recipe and the cold arms", sorted(rt["inc_s_per_image_epoch"]) == ["freeze", "full",
                                                                                                      "lora"]
          and rt["cold_s_per_image_epoch"] > 0 and all(ev.check_cite(c) for c in cites), rt)
    est, info, _ = LV.estimate_pilot_rebuild(ev, "pilot_v1", "sample")
    total = ev.get("pilot_v1/report.json", "/gpu_hours_total")
    check("a sample-replay rebuild of pilot_v1 prices at its measured %.3f GPU-h" % total, abs(est - total) < 1e-3,
          (est, total))
    est, info, _ = LV.estimate_pilot_rebuild(ev, "pilot_v1", "full")
    check("the full-replay rebuild (L1) prices at 30-50 GPU-h, an upper bound", 30 <= est <= 50 and info["upper_bound"],
          (est, info))
    check("with pool sizes 1540 .. 2503 and cand up to 2777", info["cand_null_images"][0] == (1778, 1540)
          and info["cand_null_images"][-1] == (2777, 2503), info["cand_null_images"])
    objs = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl")}
    objs["step1/select_summary.json"] = json.dumps({"sizes": {"base_B": 3927}})
    ev2 = E.from_texts(objs, "pilot_v1")
    p = {"exp": "realloop_v1", "base": "/x.jsonl", "replay_mode": "full", "recipes": "full"}
    e1, i1, c1 = LV.estimate_realloop(ev2, p)
    e2, i2, _ = LV.estimate_realloop(ev2, dict(p, no_truth=1))
    e3, i3, _ = LV.estimate_realloop(ev2, dict(p, replay_mode="sample"))
    check("a real loop: M = 10 percent of 3927 = 393, 6 verified", i1["size"] == 393 and i1["n_verified"] == 6, i1)
    check("without the truth arm it costs its truth hours less", abs((e1 - e2) - i1["truth_hours"]
                                                                     - 3 * LV.rates(ev2, "pilot_v1")[0]
                                                                     ["final_h_per_run"]) < 1e-2, (e1, e2, i1))
    check("sample replay costs less than full", e3 < e1, (e3, e1))
    check("the real-loop estimate cites base B's size", any(c["pointer"] == "/sizes/base_B" for c in c1))
    check("no Step 1: the real loop is not priced (deferred)",
          raises(lambda: LV.estimate_realloop(ev, p), LV.Defer))
    check("walltimes: relevance 1 h, audit 2 h", LV.estimate_walltime("run_inc_relevance.sh")[0] == 1.0
          and LV.estimate_walltime("run_inc_audit.sh")[0] == 2.0)


def test_propose():
    print("propose")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    res = LV.propose(DG.detect(ev), ev)
    props = {p["lever"]: p for p in res["proposals"]}
    check("pilot_v1: proposals are L1 and L4 only", sorted(props) == ["L1", "L4"], sorted(props))
    check("L1: triggered by D1, D4 and D9, R3 inc_build_pilot",
          props["L1"]["trigger"] == ["D1", "D4", "D9"] and props["L1"]["risk"] == "R3"
          and props["L1"]["policy_action"] == "inc_build_pilot", props["L1"]["trigger"])
    check("L1: params are the policy-checked flags and the cost",
          set(props["L1"]["params"]) == {"exp", "replay_mode", "est_gpu_hours"}
          and props["L1"]["parent_exp"] == "pilot_v1" and props["L1"]["child_exp"] == "pilot_v2")
    check("L1 and L4 cites resolve", all(ev.check_cite(c) for p in props.values() for c in p["cites"]))
    check("cards X1 (D1, D4) and X4 (D3b)", sorted((c["lever"], tuple(c["trigger"])) for c in res["cards"])
          == [("X1", ("D1", "D4")), ("X4", ("D3b",))], [(c["lever"], c["trigger"]) for c in res["cards"]])
    check("no operation, nothing refused", res["operations"] == [] and res["refused"] == [])
    st = LV.stable(res)
    check("stable() drops only per-call ids and timestamps",
          all("id" not in x and "created_utc" not in x for x in st["proposals"] + st["cards"])
          and st["proposals"][0]["argv"] == res["proposals"][0]["argv"]
          and all("paper_id" in lit for x in st["proposals"] for lit in x["lit"]))

    objs = ready_world()
    ev = E.from_texts(objs, "pilot_v1")
    d4 = DG.by_id(DG.detect(ev))["D4"]
    res = LV.propose(DG.detect(ev), ev)
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 "
            "--base %s/step1/base_B.jsonl --replay-mode sample --recipes lora "
            "--relevance %s/step1/relevance.json" % (INC, INC)).split()
    check("a ready pilot (lora 6/7, D1 silent): L2 with the pilot's replay mode, the chosen recipe and --relevance",
          d4["name"] == "decision_slot_ready" and len(l2) == 1 and l2[0]["argv"] == want
          and not DG.by_id(DG.detect(ev))["D1"]["fired"], (d4["summary"], [p["argv"] for p in l2]))
    check("L2 is priced from pilot_v1's rates and Step 1's base size", l2 and 0 < l2[0]["est_gpu_hours"] <= 200
          and l2[0]["estimate"]["base_images"] == 3927)
    check("... by price()", l2 and l2[0]["est_gpu_hours"] == LV.price("L2", l2[0]["params"], ev, "pilot_v1")[0])
    check("no L1 from a ready pilot", not [p for p in res["proposals"] if p["lever"] == "L1"])
    for state, rel, why in (("missing", None, "L3 first"),
                            ("stale", dict(RELEVANCE, inputs={"increment_pool": {"sha256": "c" * 64}}), "a person"),
                            ("refused", dict(RELEVANCE, calibration={"tau": 0.1, "check": {"ok": False}}), "a person"),
                            ("refused", dict(RELEVANCE, format="inc.relevance/0"), "a person")):
        o = dict(objs)
        if rel is None:
            del o["step1/relevance.json"]
        else:
            o["step1/relevance.json"] = json.dumps(rel)
        ev = E.from_texts(o, "pilot_v1")
        res = LV.propose(DG.detect(ev), ev)
        check("relevance.json %s: relevance_state says so, L2 deferred (%s), not proposed" % (state, why),
              LV.relevance_state(ev) == state and not [p for p in res["proposals"] if p["lever"] == "L2"]
              and any(x["lever"] == "L2" and why in x["reason"] for x in res["deferred"]), res["deferred"])

    # the same ready report with the real ledger: D1 fires on the pilot and takes precedence
    o = dict(objs, **{"pilot_v1/ledger.jsonl": (FIX / "pilot_v1/ledger.jsonl").read_text()})
    ev = E.from_texts(o, "pilot_v1")
    diags = DG.detect(ev)
    res = LV.propose(diags, ev)
    levers = sorted(p["lever"] for p in res["proposals"])
    check("ready (lora 6/7) while D1 fires on the pilot: L1 and L4, no L2 (D1 takes precedence)",
          DG.by_id(diags)["D1"]["fired"] and DG.by_id(diags)["D4"]["detail"]["blocked_by"]["id"] == "D1"
          and levers == ["L1", "L4"], levers)
    check("... the L2 is reported as withheld because of D1",
          any(x["lever"] == "L2" and "D1 fired on pilot_v1" in x["reason"] for x in res["deferred"]),
          res["deferred"])

    st = {"exp": "x_v1", "generation": 5, "done": False, "unblocks": [], "runs": {},
          "blocked": {"chain:lora": {"error": "read error", "cause": {"kind": "transient"}}}}
    ctx = {"history": [{"utc": "2026-09-27T01:00:00Z", "generation": 5}], "squeue": [], "now_utc":
           "2026-09-27T04:00:00Z"}
    ev = E.from_texts({"x_v1/state.json": json.dumps(st)}, "x_v1", context=ctx)
    res = LV.propose(DG.detect(ev), ev)
    l7 = [p for p in res["proposals"] if p["lever"] == "L7"]
    check("a transient block: L7 for that unit, R2, no cost",
          len(l7) == 1 and l7[0]["argv"][-4:] == ["--unit", "chain:lora", "--reason", "auto: transient"]
          and l7[0]["risk"] == "R2" and l7[0]["est_gpu_hours"] == 0.0, [p["argv"] for p in l7])
    ops = {o["op"]: o for o in res["operations"]}
    check("a stale advance: the advance operation with its command",
          "OP_ADVANCE" in ops and ops["OP_ADVANCE"]["argv"][-2:] == ["--exp", "x_v1"] and ops["OP_ADVANCE"]["risk"]
          == "R1", ops)


def test_supports_same_exp():
    print("support joins a proposal on its own experiment only")
    objs = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl")}
    objs["pilot_v2/exp.json"] = child_pilot(replay_mode="sample")     # a sample pilot_v2: L1 on pilot_v1 stands
    ev = E.from_texts(objs, "pilot_v2")
    diags = DG.detect(ev)
    by = DG.by_id(diags)
    res = LV.propose(diags, ev)
    l1 = [p for p in res["proposals"] if p["lever"] == "L1"]
    check("D9 fires on pilot_v2 (the experiment looked at)", by["D9"]["fired"] and by["D9"]["exp"] == "pilot_v2")
    check("the L1 on pilot_v1 (from D4) steps past pilot_v2 and does not take D9 from pilot_v2",
          len(l1) == 1 and l1[0]["parent_exp"] == "pilot_v1" and l1[0]["child_exp"] == "pilot_v3"
          and l1[0]["trigger"] == ["D4"]
          and not any(c["artifact"].startswith("pilot_v2/") for c in l1[0]["cites"]),
          [(p["trigger"], p["child_exp"]) for p in l1])


def test_lineage():
    print("lineage: a lever is applied to a parent once")
    objs = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl")}
    objs["pilot_v2/exp.json"] = child_pilot()
    for cur in ("pilot_v1", "pilot_v2"):
        ev = E.from_texts(objs, cur)
        res = LV.propose(DG.detect(ev), ev)
        check("pilot_v2 (full replay) built, looking at %s: no L1 again (no pilot_v3)" % cur,
              not [p for p in res["proposals"] if p["lever"] == "L1"]
              and any(x["lever"] == "L1" and "already applied to pilot_v1: pilot_v2" in x["reason"]
                      for x in res["deferred"]), (res["proposals"] and [p["argv"] for p in res["proposals"]],
                                                   res["deferred"]))
    child, c = LV.applied(E.from_texts(objs, "pilot_v1"), "L1", "pilot_v1")
    check("applied() names pilot_v2 and cites its replay mode",
          child == "pilot_v2" and c == {"artifact": "pilot_v2/exp.json", "line": None, "pointer": "/replay_mode",
                                        "value": "full"}, (child, c))
    ev = E.from_texts(objs, "pilot_v1")
    ctx_objs = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json",
                                                   "pilot_v1/ledger.jsonl")}
    for status, deferred in ((None, True), ("submitted", True), ("failed", False), ("refused", False)):
        rec = {"lever": "L1", "parent_exp": "pilot_v1", "child_exp": "pilot_v2"}
        if status:
            rec["status"] = status
        ev = E.from_texts(ctx_objs, "pilot_v1", context={"lineage": [rec]})
        res = LV.propose(DG.detect(ev), ev)
        got = not [p for p in res["proposals"] if p["lever"] == "L1"]
        check("a context lineage record of L1 on pilot_v1 (status %s): L1 %s" % (status, "deferred" if deferred
                                                                                    else "proposed"),
              got == deferred, res["deferred"])

    objs = ready_world()
    loop = loop_defn(replay_mode="sample", recipes=("lora",))
    objs["realloop_v1/exp.json"] = json.dumps(loop)
    ev = E.from_texts(objs, "realloop_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("realloop_v1 of D4's decision (sample, lora) built: no realloop_v2",
          not [p for p in res["proposals"] if p["lever"] in ("L2", "L6")]
          and any(x["lever"] == "L2" and "realloop_v1" in x["reason"] for x in res["deferred"]), res["deferred"])
    for label, change in (("other recipes", {"recipes": {"full": {"epochs": 30}}}),
                          ("another replay mode", {"replay_mode": "full"}),
                          ("another Step 1 base", {"base": dict(loop["base"], manifest_sha256="f" * 64)})):
        o = dict(objs, **{"realloop_v1/exp.json": json.dumps(dict(loop, **change))})
        ev = E.from_texts(o, "realloop_v1")
        res = LV.propose(DG.detect(ev), ev)
        l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
        check("a loop with %s is not D4's decision: L2 proposed as realloop_v2" % label,
              len(l2) == 1 and l2[0]["child_exp"] == "realloop_v2", res["deferred"])
    o = dict(ready_world())
    ev = E.from_texts(o, "pilot_v1", context={"lineage": [{"lever": "L2", "parent_exp": "pilot_v1",
                                                           "child_exp": "realloop_v1", "status": "submitted"}]})
    res = LV.propose(DG.detect(ev), ev)
    check("an in-flight L2 from pilot_v1 (context lineage) defers L2",
          not [p for p in res["proposals"] if p["lever"] == "L2"], res["proposals"])
    ev = E.from_texts({"base_b_v1/exp.json": json.dumps({"exp": "base_b_v1", "type": "baseline"}),
                       "step1/select_summary.json": json.dumps(SELECT)}, "base_b_v1")
    check("base_b_v1 exists: L8 is applied", LV.applied(ev, "L8", None)[0] == "base_b_v1")


def test_loop_levers():
    print("levers on a real loop")
    gates = [loop_gate(k, "V%d" % k) for k in (1, 2, 3)]
    defn = loop_defn(truth=False)
    objs = loop_world(defn, gates, build_summary={"n_verified": 4, "exp": "realloop_v1"})
    ev = E.from_texts(objs, "realloop_v1")
    diags = DG.detect(ev)
    by = DG.by_id(diags)
    res = LV.propose(diags, ev)
    l5 = [p for p in res["proposals"] if p["lever"] == "L5"]
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v2 --base %s/step1/base_B.jsonl "
            "--replay-mode full --recipes full --relevance %s/step1/relevance.json --size 786 --n-verified 4 "
            "--no-truth" % (INC, INC)).split()
    check("D8 fires on the loop with size 786", by["D8"]["fired"] and by["D8"]["detail"]["size"] == 786,
          by["D8"]["summary"])
    check("L5: the parent's source base (not its own copy), n_verified from build_summary, --no-truth",
          len(l5) == 1 and l5[0]["argv"] == want, [p["argv"] for p in l5])
    check("L5: every cite resolves, incl. base.source_manifest and build_summary n_verified",
          l5 and all(ev.check_cite(c) for c in l5[0]["cites"])
          and {"/base/source_manifest", "/n_verified", "/truth"} <= {c["pointer"] for c in l5[0]["cites"]})
    check("L5: priced without the truth arm", l5 and l5[0]["estimate"]["truth"] is False
          and l5[0]["est_gpu_hours"] == LV.price("L5", l5[0]["params"], ev, "realloop_v1")[0])
    from weed_optimizer_framework.tools.inc import realloop as RL
    rc, got = capture(RL, "build", RL.main, l5[0]["argv"][3:] if l5 else ["build"])
    k = got.get("kwargs", {})
    check("L5: realloop's argparse reads it back (truth False, size, n_verified, base)",
          rc == 0 and (k.get("base"), k.get("size"), k.get("n_verified"), k.get("truth"))
          == (INC + "/step1/base_B.jsonl", 786, 4, False), got)
    bs_less = loop_world(loop_defn(), gates)
    ev2 = E.from_texts(bs_less, "realloop_v1")
    params, cites = LV.loop_params(ev2, "realloop_v1")
    check("without build_summary.json, n_verified counts exp.json's verified steps (4) and truth stays on",
          params.get("n_verified") == 4 and "no_truth" not in params
          and sum(1 for c in cites if c["pointer"].endswith("/kind")) == 4, (params, len(cites)))
    later = dict(objs, **{"realloop_v2/exp.json": json.dumps(loop_defn("realloop_v2", size=786, truth=False,
                                                                        utc="2026-10-02T00:00:00Z"))})
    ev = E.from_texts(later, "realloop_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("a later loop at a larger size exists: L5 deferred naming it",
          not [p for p in res["proposals"] if p["lever"] == "L5"]
          and any(x["lever"] == "L5" and "realloop_v2" in x["reason"] for x in res["deferred"]), res["deferred"])
    stale = dict(objs, **{"step1/relevance.json": json.dumps(dict(RELEVANCE, inputs={"increment_pool":
                                                                                     {"sha256": "c" * 64}}))})
    ev = E.from_texts(stale, "realloop_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("the parent's relevance is Step 1's and stale now: L5 deferred (a person)",
          not [p for p in res["proposals"] if p["lever"] == "L5"]
          and any(x["lever"] == "L5" and "stale" in x["reason"] for x in res["deferred"]), res["deferred"])

    # D1 on a sample-mode loop: the same loop with full replay
    truths = [{"id": "truth/%d" % k, "type": "truth", "k": k, "clean": True, "detail": {"verdict": "helps"}}
              for k in (1, 2, 3)]
    forgets = [loop_gate(k, "V%d" % k, verdict="REJECT", p_recipe=0.0, p_data=0.0, cand=0.60, null=0.68, inc=0.70)
               for k in (1, 2, 3)]
    objs = loop_world(loop_defn(replay_mode="sample", size=500, n_verified=5), forgets, truths,
                      build_summary={"n_verified": 5})
    ev = E.from_texts(objs, "realloop_v1")
    diags = DG.detect(ev)
    res = LV.propose(diags, ev)
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v2 --base %s/step1/base_B.jsonl "
            "--replay-mode full --recipes full --relevance %s/step1/relevance.json --size 500 --n-verified 5"
            % (INC, INC)).split()
    check("D1 on a sample loop: L2 is the same loop (size 500, 5 verified, its relevance) with full replay",
          DG.by_id(diags)["D1"]["fired"] and len(l2) == 1 and l2[0]["argv"] == want
          and l2[0]["parent_exp"] == "realloop_v1", [p["argv"] for p in l2])
    check("... every cite resolves", l2 and all(ev.check_cite(c) for c in l2[0]["cites"]))
    rebuilt = dict(objs, **{"realloop_v2/exp.json": json.dumps(loop_defn("realloop_v2", size=500, n_verified=5,
                                                                          utc="2026-10-02T00:00:00Z"))})
    ev = E.from_texts(rebuilt, "realloop_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("that rebuild exists (realloop_v2, full): no realloop_v3",
          not [p for p in res["proposals"] if p["lever"] == "L2"]
          and any(x["lever"] == "L2" and "realloop_v2" in x["reason"] for x in res["deferred"]), res["deferred"])


CAL_FAILED = dict(RELEVANCE, calibration={"tau": 0.0557, "check": {"ok": False, "tau_min": 0.5}})


def evidence_world(objs, pool=None, verified=None, crops=None):
    """objs with Step 1's evidence summaries: select_summary.json's per-source
    increment-pool images, the files it read and retrieval.source_evidence, and
    admit_summary.json's per-source verified boxes naming the same files."""
    pool = pool or {"csgo": 4000, "weeds": 1800, "crops": 1200, "grass": 2500}
    verified = verified if verified is not None else {"weeds": 900, "crops": 400}
    crops = crops if crops is not None else {s: n + 20 for s, n in verified.items()}
    sel = json.loads(objs["step1/select_summary.json"])
    sel.update(inputs={"verified": {"sha256": "1" * 64}, "crops": {"sha256": "2" * 64}},
               sources={"increment_pool": pool},
               retrieval={"source_evidence": {s: {"species_crops": n} for s, n in crops.items()}})
    admit = {"verified_sha256": "1" * 64, "crops_sha256": "2" * 64,
             "per_slug": {s: {"boxes": dict({"other_ok": 100}, **({"verified": verified[s]} if s in verified else {}))}
                          for s in pool}}
    return dict(objs, **{"step1/select_summary.json": json.dumps(sel), "step1/admit_summary.json": json.dumps(admit)})


def test_increment_criterion():
    print("the relevance criterion of a real loop (--relevance or --increment-sources evidence)")
    objs = ready_world()
    for label, rel, state in (("passed, made for this build", RELEVANCE, "matching"),
                              ("failed its own check (tau 0.0557 < 0.5)", CAL_FAILED, "calibration_failed"),
                              ("failed, made for another build", dict(CAL_FAILED, inputs={"increment_pool":
                                                                                           {"sha256": "c" * 64}}),
                               "stale"),
                              ("ok false at tau >= tau_min (does not follow from its tau)",
                               dict(RELEVANCE, calibration={"tau": 0.62, "check": {"ok": False, "tau_min": 0.5}}),
                               "refused"),
                              ("failed under another tau_min", dict(RELEVANCE, calibration={
                                  "tau": 0.1, "check": {"ok": False, "tau_min": 0.2}}), "refused"),
                              ("ok true at tau < tau_min (does not follow from its tau)",
                               dict(RELEVANCE, calibration={"tau": 0.0557, "check": {"ok": True, "tau_min": 0.5}}),
                               "refused"),
                              ("passed under another tau_min (tau 0.31, tau_min 0.2)",
                               dict(RELEVANCE, calibration={"tau": 0.31, "check": {"ok": True, "tau_min": 0.2}}),
                               "refused"),
                              ("ok not a boolean", dict(RELEVANCE, calibration={
                                  "tau": 0.0557, "check": {"ok": "false", "tau_min": 0.5}}), "refused"),
                              ("failed, made under another rule (params tau_min 0.3)",
                               dict(CAL_FAILED, params=dict(RULE, tau_min=0.3)), "refused"),
                              ("failed, made at another percentile (params calibration_percentile 10)",
                               dict(CAL_FAILED, params=dict(RULE, calibration_percentile=10.0)), "refused"),
                              ("passed, made with another min_crops (params min_crops 10)",
                               dict(RELEVANCE, params=dict(RULE, min_crops=10)), "refused"),
                              ("failed, with no params block", {k: v for k, v in CAL_FAILED.items() if k != "params"},
                               "refused"),
                              ("passed, with the rule's values written as 20.0 and 5 (relevance.load compares with !=)",
                               dict(RELEVANCE, params={"min_crops": 20.0, "calibration_percentile": 5,
                                                       "tau_min": 0.5}), "matching"),
                              ("another format", dict(CAL_FAILED, format="inc.relevance/0"), "refused")):
        ev = E.from_texts(dict(objs, **{"step1/relevance.json": json.dumps(rel)}), "pilot_v1")
        check("relevance_status: a file that %s is %s" % (label, state), LV.relevance_status(ev)["state"] == state,
              LV.relevance_status(ev))
    base = INC + "/step1/base_B.jsonl"
    ev = E.from_texts(dict(objs, **{"step1/relevance.json": json.dumps(CAL_FAILED)}), "pilot_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("calibration failed, no admit_summary.json in the evidence: L2 deferred (the evidence cannot be read)",
          not [p for p in res["proposals"] if p["lever"] == "L2"]
          and any(x["lever"] == "L2" and "admit_summary.json is not in the evidence" in x["reason"]
                  for x in res["deferred"]), res["deferred"])
    world = evidence_world(dict(objs, **{"step1/relevance.json": json.dumps(CAL_FAILED)}))
    ev = E.from_texts(world, "pilot_v1")
    rec, cites = LV.evidence_capacity(ev)
    check("evidence_capacity: weeds and crops are evidenced (3,000 images >= 2,751 = 7 x 393); csgo and grass "
          "(no verified box) are excluded",
          sorted(rec["evidenced_sources"]) == ["crops", "weeds"] and rec["evidenced_pool_images"] == 3000
          and rec["needed_images"] == 2751 and rec["fits"] and rec["excluded_images"] == 6500
          and rec["max_n_verified"] == 6 and all(ev.check_cite(c) for c in cites), rec)
    res = LV.propose(DG.detect(ev), ev)
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base %s "
            "--replay-mode sample --recipes lora --increment-sources evidence" % base).split()
    check("calibration failed and the evidenced pool holds the build: L2 on --increment-sources evidence",
          len(l2) == 1 and l2[0]["argv"] == want and l2[0]["params"]["increment_sources"] == "evidence"
          and all(ev.check_cite(c) for c in l2[0]["cites"]), [p["argv"] for p in l2])
    diags = DG.detect(ev)
    check("... D2's then 'L2 with --increment-sources evidence' is not also deferred: D4's L2 carries it",
          DG.by_id(diags)["D2"]["detail"]["then"] == ["L2 with --increment-sources evidence"]
          and not [x for x in res["deferred"] if x["lever"] == "L2" and x["reason"].startswith("then: ")],
          [x for x in res["deferred"] if x["lever"] == "L2"])
    no_d4 = LV.propose([d for d in diags if d["id"] != "D4"], ev)
    check("... without D4's L2 in the same tick the then is deferred (and no L2 is proposed)",
          not [p for p in no_d4["proposals"] if p["lever"] == "L2"]
          and [x["reason"] for x in no_d4["deferred"] if x["lever"] == "L2"]
          == ["then: L2 with --increment-sources evidence"], no_d4["deferred"])
    check("_then_proposed: only '<lever> with <flags>' whose flags an argv of that lever carries side by side",
          LV._then_proposed("L2 with --increment-sources evidence", {("L2", tuple(want)): {}})
          and not LV._then_proposed("L2 with --relevance", {("L2", tuple(want)): {}})
          and not LV._then_proposed("L2 with --increment-sources evidence", {("L5", tuple(want)): {}})
          and not LV._then_proposed("L2 waits for the v2 pilot's report", {("L2", tuple(want)): {}}))
    check("... priced like any real loop (the criterion changes the draw, not the runs)",
          l2 and l2[0]["est_gpu_hours"] == LV.price("L2", l2[0]["params"], ev, "pilot_v1")[0])
    check("... L3 is not proposed over the calibration-failed file, for any proposer",
          not [p for p in res["proposals"] if p["lever"] == "L3"]
          and raises(lambda: LV._l3_ok(ev), LV.Defer, "L3 is not proposed"))
    small = evidence_world(dict(objs, **{"step1/relevance.json": json.dumps(CAL_FAILED)}),
                           pool={"csgo": 4000, "weeds": 900, "crops": 539})
    ev = E.from_texts(small, "pilot_v1")
    res = LV.propose(DG.detect(ev), ev)
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v1 --base %s "
            "--replay-mode sample --recipes lora --increment-sources evidence --size 287 --n-verified 4" % base).split()
    check("an evidenced pool that cannot hold 7 x 393 (1,439 images): the R4 sizing rule, L2 at N 4 x M 287 (the "
          "non-default --size and --n-verified in the template's order)",
          len(l2) == 1 and l2[0]["argv"] == want and (l2[0]["params"]["size"], l2[0]["params"]["n_verified"])
          == (287, 4) and all(ev.check_cite(c) for c in l2[0]["cites"])
          and not [x for x in res["deferred"] if x["lever"] == "L2"], ([p["argv"] for p in l2], res["deferred"]))
    from weed_optimizer_framework.tools.inc import realloop as RL0
    rec, _c = LV.evidence_sizing(ev)
    check("evidence_sizing records the default (N 6 x M 393 = 2,751, does not fit) and the sized loop (N 4, M = "
          "min(393, floor(1439 / 5)) = 287, 1,435 needed) with the rule",
          rec["sizing"]["default"]["needed_images"] == 2751 and not rec["sizing"]["default"]["fits"]
          and rec["sizing"]["applied"] and (rec["n_verified"], rec["increment_images"], rec["needed_images"])
          == (4, 287, 1435) and rec["sizing"]["params"] == {"size": 287, "n_verified": 4}
          and "R4 review" in rec["sizing"]["rule"], rec["sizing"])
    check("the rule's N decides protocol min_decided_increments with UNVERIFIED and OTHER_HEAVY, and no smaller N "
          "does (realloop.sequence)",
          LV.sized_n_verified() == 4 and len(RL0.sequence(4)) >= LV.protocol("min_decided_increments")
          > len(RL0.sequence(3)), (LV.sized_n_verified(), RL0.sequence(4)))
    fits = LV.increment_criterion(ev, INC, n_verified=4, size=287)
    check("... the same pool fits N 4 x M 287 (the protocol's six decided increments): increment_criterion at a "
          "person's N and M (checked as given, never sized)",
          fits[0] == {"increment_sources": "evidence"} and fits[1]["needed_images"] == 1435, fits[1])
    check("... at a given --size alone the default N is kept and checked (7 x 287 > 1,439): deferred with the numbers",
          raises(lambda: LV.increment_criterion(ev, INC, size=287), LV.Defer, "at most --n-verified 4"))
    check("... a given N or M below the R4 rule's floor is refused with the numbers (LeverError), never sized: "
          "M 196 < 0.05 x 3,927 = 196.35; N 3 decides 5 increments; N 1 x M 50; N 1 x M 1",
          raises(lambda: LV.increment_criterion(ev, INC, n_verified=4, size=196), LV.LeverError, "196.35 images")
          and raises(lambda: LV.increment_criterion(ev, INC, n_verified=3, size=287), LV.LeverError,
                     "N = 3 decides 5 increments with UNVERIFIED and OTHER_HEAVY, fewer than min_decided_increments 6")
          and raises(lambda: LV.increment_criterion(ev, INC, n_verified=1, size=50), LV.LeverError,
                     "--n-verified 1 --size 50 is below the R4 rule's floor")
          and raises(lambda: LV.increment_criterion(ev, INC, n_verified=1, size=1), LV.LeverError, "a person's"))
    check("... check_criterion (an L5 or D1 rebuild at its parent's N and M) holds the same floor, then the capacity",
          raises(lambda: LV.check_criterion(ev, {"increment_sources": "evidence", "n_verified": 3, "size": 287}, INC),
                 LV.LeverError, "fewer than min_decided_increments")
          and raises(lambda: LV.check_criterion(ev, {"increment_sources": "evidence", "n_verified": 4, "size": 196},
                                                INC), LV.LeverError, "196.35")
          and bool(LV.check_criterion(ev, {"increment_sources": "evidence", "n_verified": 4, "size": 287}, INC)))
    check("r4_floor: the sized loop (N 4 x M 287) and the default one (N 6 x M 393) are at or above the floor",
          LV.r4_floor({"base_images": 3927, "n_verified": 4, "increment_images": 287}) == ""
          and LV.r4_floor({"base_images": 3927, "n_verified": 6, "increment_images": 393}) == ""
          and LV.r4_floor({"base_images": 3927, "n_verified": 4, "increment_images": 197}) == "")
    tiny = evidence_world(dict(objs, **{"step1/relevance.json": json.dumps(CAL_FAILED)}),
                          pool={"csgo": 4000, "weeds": 600, "crops": 380})
    ev = E.from_texts(tiny, "pilot_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("an evidenced pool of 980 images: the sized M = min(393, floor(980 / 5)) = 196 is below 0.05 x 3,927 = "
          "196.35: not sized, L2 deferred with the numbers and 'a person'",
          not [p for p in res["proposals"] if p["lever"] == "L2"]
          and any(x["lever"] == "L2" and "hold 980" in x["reason"] and "need 2751" in x["reason"]
                  and "= 196 images" in x["reason"] and "196.35" in x["reason"] and "a person" in x["reason"]
                  for x in res["deferred"]), [x["reason"][-240:] for x in res["deferred"] if x["lever"] == "L2"])
    ev = E.from_texts(small, "pilot_v1")
    bad = evidence_world(dict(objs, **{"step1/relevance.json": json.dumps(CAL_FAILED)}), crops={"weeds": 10})
    ev = E.from_texts(bad, "pilot_v1")
    check("summaries that disagree (verified boxes above select's species crops): deferred, as realloop refuses",
          raises(lambda: LV.evidence_capacity(ev), LV.Defer, "disagree"))
    from weed_optimizer_framework.tools.inc import select as S0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ap = pathlib.Path(td) / "admit_summary.json"
        ap.write_text(world["step1/admit_summary.json"])
        ld = S0.load_evidence(json.loads(world["step1/select_summary.json"]), ap, S0.MIN_EVIDENCE)
    check("select.load_evidence on the same summaries finds the same evidenced sources and boxes",
          ld["evidenced"] == {"weeds": 900, "crops": 400}, ld["evidenced"])

    # a real loop built on evidence keeps its criterion when it is rebuilt (L5)
    gates = [loop_gate(k, "V%d" % k) for k in (1, 2, 3)]
    defn = loop_defn(truth=False)
    defn["step1"] = {"relevance": None, "increment_sources": {"mode": "evidence", "min_evidence": 1,
                                                              "rule": S0.EVIDENCE_RULE}}
    lw = loop_world(defn, gates, build_summary={"n_verified": 4, "exp": "realloop_v1"})
    lw["step1/relevance.json"] = json.dumps(CAL_FAILED)
    ev = E.from_texts(lw, "realloop_v1")
    res = LV.propose(DG.detect(ev), ev)
    check("an evidence loop without Step 1's evidence in view: L5 deferred (the capacity cannot be read)",
          not [p for p in res["proposals"] if p["lever"] == "L5"]
          and any(x["lever"] == "L5" and "not in the evidence" in x["reason"] for x in res["deferred"]),
          res["deferred"])
    big = evidence_world(lw, pool={"csgo": 4000, "weeds": 3000, "crops": 1500})
    ev = E.from_texts(big, "realloop_v1")
    params, cites = LV.loop_params(ev, "realloop_v1")
    check("loop_params of an evidence loop: increment_sources evidence, no relevance, cited",
          params.get("increment_sources") == "evidence" and "relevance" not in params
          and "/step1/increment_sources/mode" in {c["pointer"] for c in cites}, params)
    res = LV.propose(DG.detect(ev), ev)
    l5 = [p for p in res["proposals"] if p["lever"] == "L5"]
    want = ("python -m weed_optimizer_framework.tools.inc.realloop build --exp realloop_v2 --base %s "
            "--replay-mode full --recipes full --increment-sources evidence --size 786 --n-verified 4 --no-truth"
            % base).split()
    check("L5 of an evidence loop carries --increment-sources evidence (4,500 images hold 5 x 786)",
          len(l5) == 1 and l5[0]["argv"] == want and all(ev.check_cite(c) for c in l5[0]["cites"]),
          [p["argv"] for p in l5])
    from weed_optimizer_framework.tools.inc import realloop as RL
    rc, got = capture(RL, "build", RL.main, want[3:])
    k = got.get("kwargs", {})
    check("... realloop's argparse reads it back", rc == 0 and (k.get("increment_sources"), k.get("relevance"),
                                                                k.get("size"), k.get("truth"))
          == ("evidence", None, 786, False), got)
    defn["step1"]["increment_sources"]["min_evidence"] = 2
    ev = E.from_texts(dict(big, **{"realloop_v1/exp.json": json.dumps(defn)}), "realloop_v1")
    check("an evidence loop built with --min-evidence 2 is not rebuilt by the menu (the flag is not on it)",
          raises(lambda: LV.loop_params(ev, "realloop_v1"), LV.Defer, "--min-evidence"))


def _with_gate(text, mode):
    d = json.loads(text)
    d["gate"] = {"flips_mode": mode}
    return json.dumps(d)


def _net_ready():
    """ready_world decided on protocol v2 with a final 'supported' v2_check (D4
    waits for a final check before a real loop is built on that gate)."""
    objs = ready_world()
    objs["pilot_v1/exp.json"] = _with_gate(objs["pilot_v1/exp.json"], "net")
    rep = json.loads(objs["pilot_v1/report.json"])
    rep["gate"] = {"flips_mode": "net", "pinned": True, "block": {"flips_mode": "net"}}
    rep["v2_check"] = {"final": True, "truth_arm": True, "outcome": "supported", "compared": 21, "agree_v2": 12,
                       "agree_v1_counterfactual": 9, "discordant": [], "non_clean_accepted": []}
    objs["pilot_v1/report.json"] = json.dumps(rep)
    return objs


def test_gate_and_l9():
    print("the gate is carried through, and L9 (D15)")
    whole = [e for e in ("pilot_v1", "pilot_v2", "b0_v1", "base_b_v1")]
    ev = E.load_dir(FIX, "pilot_v2", exps=whole)
    diags = DG.detect(ev)
    res = LV.propose(diags, ev)
    l9 = [p for p in res["proposals"] if p["lever"] == "L9"]
    want = ("python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v3 --replay-mode full "
            "--gate-flips-mode net").split()
    check("pilot_v2: one L9, ranked first, the exact command", len(l9) == 1 and res["proposals"][0]["lever"] == "L9"
          and l9[0]["argv"] == want, [p["argv"] for p in res["proposals"]])
    check("L9: triggered by D15, R3 inc_build_pilot, parent pilot_v2, child pilot_v3",
          l9 and l9[0]["trigger"] == ["D15"] and l9[0]["risk"] == "R3" and l9[0]["policy_action"] == "inc_build_pilot"
          and (l9[0]["parent_exp"], l9[0]["child_exp"]) == ("pilot_v2", "pilot_v3"))
    check("L9: params are the policy-checked flags and the cost",
          l9 and set(l9[0]["params"]) == {"exp", "replay_mode", "gate_flips_mode", "est_gpu_hours"}
          and l9[0]["params"]["gate_flips_mode"] == "net")
    check("L9: priced by price() from pilot_v2's own rates (a full-replay rebuild plus the build job)",
          l9 and l9[0]["est_gpu_hours"] == LV.price("L9", {"exp": "pilot_v3", "replay_mode": "full"}, ev,
                                                    "pilot_v2")[0] and 20 <= l9[0]["est_gpu_hours"] <= 60,
          l9 and l9[0]["est_gpu_hours"])
    check("L9: every cite resolves and the five flips-only ledger lines are among them",
          l9 and all(ev.check_cite(c) for c in l9[0]["cites"])
          and {13, 15, 19, 22, 23} <= {c["line"] for c in l9[0]["cites"]
                                       if c["artifact"] == "pilot_v2/ledger.jsonl"})
    check("pilot_v2: no L1 and no realloop; X1 is a card (D1 fires unblocked: freeze I5 failed the regression "
          "guard too)", not [p for p in res["proposals"] if p["lever"] in ("L1", "L2", "L5", "L6")]
          and "X1" in [c["lever"] for c in res["cards"]], [c["lever"] for c in res["cards"]])
    check("pilot_v2's L9 is admitted by the policy row and levers.json L9's bounds",
          l9 and LV.check_params("L9", l9[0]["params"])[0])

    texts = {n: (FIX / n).read_text() for n in ("pilot_v2/exp.json", "pilot_v2/report.json", "pilot_v2/ledger.jsonl")}
    v3 = json.loads(texts["pilot_v2/exp.json"].replace("pilot_v2", "pilot_v3"))
    v3.update(initialised_utc="2026-09-28T00:00:00Z", gate={"flips_mode": "net"})
    for cur in ("pilot_v2", "pilot_v3"):
        e2 = E.from_texts(dict(texts, **{"pilot_v3/exp.json": json.dumps(v3)}), cur)
        res = LV.propose(DG.detect(e2), e2)
        check("pilot_v3 (net) built, looking at %s: no L9 again, deferred naming pilot_v3" % cur,
              not [p for p in res["proposals"] if p["lever"] == "L9"]
              and any(x["lever"] == "L9" and "already applied to pilot_v2: pilot_v3" in x["reason"]
                      for x in res["deferred"]) if cur == "pilot_v2" else
              not [p for p in res["proposals"] if p["lever"] == "L9"], res["deferred"])
    e2 = E.from_texts(dict(texts, **{"pilot_v3/exp.json": json.dumps(v3)}), "pilot_v2")
    child, c = LV.applied(e2, "L9", "pilot_v2")
    check("applied(L9) names pilot_v3 and cites its gate", child == "pilot_v3"
          and c == {"artifact": "pilot_v3/exp.json", "line": None, "pointer": "/gate/flips_mode", "value": "net"},
          (child, c))
    v3n = dict(v3, gate={"flips_mode": "negative"})
    e2 = E.from_texts(dict(texts, **{"pilot_v3/exp.json": json.dumps(v3n)}), "pilot_v2")
    res = LV.propose(DG.detect(e2), e2)
    check("a later pilot on v1 is not L9's child: L9 proposed, its name stepping past it (pilot_v4)",
          [(p["lever"], p["child_exp"]) for p in res["proposals"] if p["lever"] == "L9"] == [("L9", "pilot_v4")])
    for status, deferred in (("in_flight", True), ("failed", False)):
        e2 = E.from_texts(texts, "pilot_v2", context={"lineage": [{"lever": "L9", "parent_exp": "pilot_v2",
                                                                    "child_exp": "pilot_v3", "status": status}]})
        res = LV.propose(DG.detect(e2), e2)
        check("a context lineage record of L9 on pilot_v2 (%s): L9 %s" % (status, "deferred" if deferred
                                                                           else "proposed"),
              (not [p for p in res["proposals"] if p["lever"] == "L9"]) == deferred, res["deferred"])
    e2 = E.from_texts(dict(texts, **{"pilot_v2/exp.json": _with_gate(texts["pilot_v2/exp.json"], "net")}),
                      "pilot_v2")
    by = DG.by_id(DG.detect(e2))
    res = LV.propose(list(by.values()), e2)
    check("a parent already pinned to net: D15 gives card X6, no L9",
          by["D15"]["levers"] == ["X6"] and not [p for p in res["proposals"] if p["lever"] == "L9"]
          and "X6" in [c["lever"] for c in res["cards"]], by["D15"]["levers"])
    check("... and L9 refuses it directly (Defer)",
          raises(lambda: LV._build_l9(dict(by["D15"], exp="pilot_v2"), e2, LV.load_menu()), LV.Defer, "already"))

    objs = _net_ready()
    ev = E.from_texts(objs, "pilot_v1")
    res = LV.propose(DG.detect(ev), ev)
    l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
    check("L2 from D4 carries the gate of the pilot D4 evaluated (net), last, cited",
          len(l2) == 1 and l2[0]["argv"][-2:] == ["--gate-flips-mode", "net"]
          and l2[0]["params"]["gate_flips_mode"] == "net"
          and any(c["artifact"] in ("pilot_v1/report.json", "pilot_v1/exp.json") and c["pointer"] == "/gate/flips_mode"
                  and c["value"] == "net" for c in l2[0]["cites"])
          and all(ev.check_cite(c) for c in l2[0]["cites"]), [p["argv"] for p in l2])
    objs = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl")}
    objs["pilot_v1/exp.json"] = _with_gate(objs["pilot_v1/exp.json"], "net")
    ev = E.from_texts(objs, "pilot_v1")
    res = LV.propose(DG.detect(ev), ev)
    l1 = [p for p in res["proposals"] if p["lever"] == "L1"]
    check("L1 of a net pilot carries --gate-flips-mode net",
          len(l1) == 1 and l1[0]["argv"] == "python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v2 "
          "--replay-mode full --gate-flips-mode net".split(), [p["argv"] for p in l1])
    gates = [loop_gate(k, "V%d" % k) for k in (1, 2, 3)]
    ld = dict(loop_defn(truth=False), gate={"flips_mode": "net"})
    ev = E.from_texts(loop_world(ld, gates, build_summary={"n_verified": 4}), "realloop_v1")
    params, cites = LV.loop_params(ev, "realloop_v1")
    res = LV.propose(DG.detect(ev), ev)
    l5 = [p for p in res["proposals"] if p["lever"] == "L5"]
    check("loop_params and L5 carry the parent loop's net gate",
          params.get("gate_flips_mode") == "net" and any(c["pointer"] == "/gate/flips_mode" for c in cites)
          and len(l5) == 1 and l5[0]["argv"][-2:] == ["--gate-flips-mode", "net"], [p["argv"] for p in l5])
    ev = E.from_texts(loop_world(loop_defn(truth=False), gates, build_summary={"n_verified": 4}), "realloop_v1")
    check("a v1 loop (no gate block): no --gate-flips-mode", "gate_flips_mode" not in LV.loop_params(ev,
                                                                                                    "realloop_v1")[0])
    _gate_lineage()
    _refuted_gate()


def _gate_lineage():
    """applied() counts an earlier experiment only on the gate the lever forwards."""
    objs = _net_ready()
    loop = loop_defn(replay_mode="sample", recipes=("lora",))
    for mode, applied_ in ((None, False), ("net", True)):
        o = dict(objs, **{"realloop_v1/exp.json": json.dumps(dict(loop, gate={"flips_mode": mode}) if mode
                                                             else loop)})
        ev = E.from_texts(o, "pilot_v1")
        det = {"replay_mode": "sample", "recipes": ["lora"]}
        child = LV.applied(ev, "L2", "pilot_v1", detail=det)[0]
        res = LV.propose(DG.detect(ev), ev)
        l2 = [p for p in res["proposals"] if p["lever"] == "L2"]
        if applied_:
            check("a net loop of the net pilot's D4 decision: L2 applied, deferred naming it",
                  child == "realloop_v1" and not l2
                  and any(x["lever"] == "L2" and "realloop_v1" in x["reason"] for x in res["deferred"]),
                  (child, res["deferred"]))
        else:
            check("a v1 loop with the same replay mode and recipe does not stand for the net pilot's L2: "
                  "L2 proposed as realloop_v2 with --gate-flips-mode net",
                  child is None and len(l2) == 1 and l2[0]["child_exp"] == "realloop_v2"
                  and l2[0]["argv"][-2:] == ["--gate-flips-mode", "net"], (child, [p["argv"] for p in l2]))
    base = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl")}
    for parent_mode, child_mode, applied_ in (("net", None, False), ("net", "net", True), (None, "net", False),
                                              (None, None, True)):
        o = dict(base)
        if parent_mode:
            o["pilot_v1/exp.json"] = _with_gate(o["pilot_v1/exp.json"], parent_mode)
        o["pilot_v2/exp.json"] = _with_gate(child_pilot(), child_mode) if child_mode else child_pilot()
        ev = E.from_texts(o, "pilot_v1")
        child = LV.applied(ev, "L1", "pilot_v1")[0]
        check("L1 on a %s pilot, a later full-replay pilot on %s: %s" % (parent_mode or "v1", child_mode or "v1",
                                                                        "applied" if applied_ else "not applied"),
              (child == "pilot_v2") == applied_, child)
    gates = [loop_gate(k, "V%d" % k) for k in (1, 2, 3)]
    parent = dict(loop_defn(truth=False), gate={"flips_mode": "net"})
    for later_mode, applied_ in ((None, False), ("net", True)):
        later = loop_defn(exp="realloop_v2", size=800, truth=False, utc="2026-10-02T00:00:00Z")
        if later_mode:
            later["gate"] = {"flips_mode": later_mode}
        o = loop_world(parent, gates, build_summary={"n_verified": 4})
        o["realloop_v2/exp.json"] = json.dumps(later)
        ev = E.from_texts(o, "realloop_v1")
        child = LV.applied(ev, "L5", "realloop_v1", params={"replay_mode": "full"})[0]
        check("L5 on a net loop, a later larger loop on %s: %s" % (later_mode or "v1",
                                                                  "applied" if applied_ else "not applied"),
              (child == "realloop_v2") == applied_, child)


def _refuted_gate():
    """No lever rebuilds on a gate whose v2 check came out refuted on the parent."""
    objs = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl")}
    objs["pilot_v1/exp.json"] = _with_gate(objs["pilot_v1/exp.json"], "net")
    rep = json.loads(objs["pilot_v1/report.json"])
    rep["gate"] = {"flips_mode": "net", "pinned": True, "block": {"flips_mode": "net"}}
    for final, outcome, deferred in ((True, "refuted", True), (False, "refuted", False), (True, "supported", False)):
        rep["v2_check"] = {"final": final, "truth_arm": True, "outcome": outcome, "compared": 21, "agree_v2": 8,
                           "agree_v1_counterfactual": 9, "discordant": [], "non_clean_accepted": []}
        ev = E.from_texts(dict(objs, **{"pilot_v1/report.json": json.dumps(rep)}), "pilot_v1")
        diags = DG.detect(ev)
        res = LV.propose(diags, ev)
        l1 = [p for p in res["proposals"] if p["lever"] == "L1"]
        if deferred:
            check("D1 on a net pilot whose final v2_check is refuted: no L1 on the retired gate (deferred to X7), "
                  "X7 and X1 are cards", DG.by_id(diags)["D1"]["levers"] == ["L1", "X1"] and not l1
                  and any(x["lever"] == "L1" and "refuted" in x["reason"] for x in res["deferred"])
                  and {"X1", "X7"} <= {c["lever"] for c in res["cards"]},
                  ([p["argv"] for p in l1], res["deferred"], [c["lever"] for c in res["cards"]]))
            check("... and gate_params refuses the refuted gate for any lever",
                  raises(lambda: LV.gate_params(ev, "pilot_v1"), LV.Defer, "refuted"))
        else:
            check("a %s %s check does not retire the gate: L1 carries --gate-flips-mode net"
                  % ("final" if final else "provisional", outcome),
                  len(l1) == 1 and l1[0]["argv"][-2:] == ["--gate-flips-mode", "net"], [p["argv"] for p in l1])
    ld = dict(loop_defn(truth=False), gate={"flips_mode": "net"})
    o = loop_world(ld, [loop_gate(k, "V%d" % k) for k in (1, 2, 3)], build_summary={"n_verified": 4})
    o["realloop_v1/report.json"] = json.dumps({"exp": "realloop_v1", "type": "chain",
                                               "v2_check": {"final": True, "outcome": "refuted"}})
    ev = E.from_texts(o, "realloop_v1")
    check("loop_params of a real loop whose own v2 check is refuted defers (no L2 or L5 on that gate)",
          raises(lambda: LV.loop_params(ev, "realloop_v1"), LV.Defer, "refuted"))


def test_price():
    print("price()")
    ev = E.load_dir(FIX, "pilot_v1", exps=V1_ERA)
    check("L1 from pilot_v1: the rebuild estimate plus the build job's own walltime (open issue 3)",
          LV.price("L1", {"exp": "pilot_v2"}, ev, "pilot_v1")[0]
          == round(LV.estimate_pilot_rebuild(ev, "pilot_v1", "full")[0] + LV.build_job_hours()[0], 3)
          and LV.build_job_hours()[0] == 4.0)
    check("L1 without a parent: deferred, never guessed", raises(lambda: LV.price("L1", {}, ev), LV.Defer))
    check("L2 without Step 1: deferred", raises(lambda: LV.price("L2", {"replay_mode": "full", "recipes": "full"},
                                                                 ev), LV.Defer))
    check("L2 without recipes: deferred", raises(lambda: LV.price("L2", {"replay_mode": "full"}, ev), LV.Defer))
    check("L3, L4: the job scripts' walltimes; L7: zero",
          LV.price("L3", {}, ev)[0] == 1.0 and LV.price("L4", {"exp": "pilot_v1"}, ev)[0] == 2.0
          and LV.price("L7", {}, ev)[0] == 0.0)
    ev2 = E.from_texts({n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json")}
                       | {"step1/select_summary.json": json.dumps(SELECT)}, "pilot_v1")
    est, info, cites = LV.price("L8", {}, ev2)
    check("L8: the baseline estimate, citing base B's size", est > 0 and info["images"] == 3927
          and any(c["pointer"] == "/sizes/base_B" for c in cites))


def main():
    test_menu()
    test_argv()
    test_real_argparse()
    test_bounds()
    test_names_and_cost()
    test_propose()
    test_supports_same_exp()
    test_lineage()
    test_loop_levers()
    test_increment_criterion()
    test_gate_and_l9()
    test_price()
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
