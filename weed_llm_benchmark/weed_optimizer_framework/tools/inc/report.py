"""The INC experiment report (docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Report").

    python -m weed_optimizer_framework.tools.inc.report --exp EXP

Writes INC_DIR/<exp>/report.md and report.json from exp.json, state.json, the
ledger and the runs' score and run files; it computes no metric of its own
beyond means and sample sds (ddof 1) of scorer outputs. It can be run at any
time: steps and runs that are not finished show as missing.

  * Per step: each chain's verdict, P_data, P_recipe, guards and attribution
    (from the ledger's gate entries), the soup choice after an ACCEPT, the
    truth arm's decision (ledger truth entries), and whether each chain agrees
    with the truth arm (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts).
  * The gate config the experiment is decided with, as pinned at init
    (driver.pinned_gate_config: state.json's gate_pin for a definition with
    a gate block, else GateConfig()'s defaults, which is protocol v1), in the
    header: every GateConfig value, the block it was pinned from (or none),
    and whether the flips guard counts negative flips (v1) or net flips (v2,
    docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)"). The
    header cross-checks the pin against the config every gate, soup, truth
    and gate_pin ledger entry recorded (GATE CONFIG MISMATCH per entry that
    differs) and against what exp.json resolves to now (GATE CONFIG CHANGED;
    the driver refuses to advance then). exp.json is never the source.
  * For an experiment decided on net flips: each step's v1 verdict on the
    same runs (gate.v1_counterfactual) and its agreement with the truth arm,
    and the pre-registered protocol v2 check (v2_check: supported, refuted
    or inconclusive; provisional until the experiment is done).
  * The replay mode (exp.json "replay_mode"; absent = 'sample') and, per
    chain and step, the replay mode that built the step's manifests, the
    cand and null train sizes (the manifests' n_images in the gate entry),
    the accepted pool and |D_k|, and the warmup those sizes run
    (inc.pilot.effective_warmup).
  * Final table, from the 'final' runs only (the only runs that read test):
    dev, ood22, ood23, imageweeds and test, each as the 12-class score
    (species_map50_95: the cwd12 ids present in the exam) and the
    class-agnostic score, mean +- sd over seeds where there are seeds: each
    chain's final incumbent, the base seeds and the T_final truth runs.
  * GPU-hours per chain and per arm (base, truth, final), from the 'seconds'
    of every run.json: the current one, plus each failed attempt's, which the
    driver keeps in state.json (the executor removes a failed run.json when
    the retry starts).
  * Interventions: every unblock (automatic or by hand) and code re-pin in
    the ledger, the pinned code, and the attribution steps the driver did not
    run (protocol steps 4-5), with the REJECTs that lack them.
  * The attribution of each step whose labels are known to be bad or were
    not verified, per chain (verdict, class_vs_loc, attributed to labels):
    a step named Bswap (the pilot's planted label swaps) and a step whose
    exp.json kind is 'unverified' (inc/realloop.py's UNVERIFIED). A section
    is written only for a step the experiment has.
  * The effective warmup the builder recorded (exp.json "effective_warmup"),
    and the range the decided steps' actual train sizes give.

A ledger whose last line is partial (an advance killed mid-append; the next
advance repairs it) is read up to its last complete entry, with a note.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import statistics
import sys

from . import common as C
from . import driver as D
from . import gate as G
from . import pilot as P

REPORT_EXAMS = ("dev", "ood22", "ood23", "imageweeds", "test")
BSWAP_STEP = "Bswap"                 # inc/pilot.py: the planted label-swap step
UNVERIFIED_KIND = "unverified"       # inc/realloop.py KIND_UNVERIFIED: exp.json step "kind"


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def read_ledger(paths, notes=None):
    """{entry id: entry}; a later entry with the same id replaces an earlier one.
    A partial last line (no trailing newline) is skipped and noted; any other
    line that is not JSON raises ValueError."""
    out = {}
    if not paths.ledger.is_file():
        return out
    with open(paths.ledger) as fh:
        text = fh.read()
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except ValueError as err:
            if i == len(lines) - 1:         # after the last newline: an interrupted append
                if notes is not None:
                    notes.append("the ledger's last line is partial (%d bytes, an advance killed "
                                 "mid-append) and was skipped; the next advance repairs it" % len(ln))
                continue
            raise ValueError("%s line %d is not JSON: %s" % (paths.ledger, i + 1, err))
        out[e.get("id")] = e
    return out


def mean_sd(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(xs), "mean": statistics.fmean(xs),
            "sd": statistics.stdev(xs) if len(xs) >= 2 else None}


def score_values(paths, rid, exam):
    """(12-class, agnostic, production) from runs/<rid>/scores/<exam>.json, or None."""
    p = paths.score(rid, exam)
    if not p.is_file():
        return None
    s = _read(p)
    twelve = s.get("species_map50_95")
    if twelve is None:
        twelve = s.get("map50_95")
    return twelve, s.get("agnostic_map50_95"), s.get("production")


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0


def run_seconds(paths, rid, run):
    """This attempt's run.json seconds plus the seconds of every failed attempt
    in the run's state history (a run.json already in the history is counted
    once)."""
    counted = {h.get("run_json_sha256") for h in run.get("history", []) if h.get("run_json_sha256")}
    total = sum(_num(h.get("seconds")) for h in run.get("history", []))
    f = paths.run_json(rid)
    if f.is_file():
        try:
            if C.sha256_file(f) not in counted:
                total += _num(_read(f).get("seconds"))
        except (OSError, ValueError):
            pass
    return total


def _train_sizes(defn, state, r, n, g):
    """The replay mode that built chain r's step n, its cand / null train sizes
    (the manifests the gate entry recorded), pool and |D_k| (state), and the
    warmup those sizes run with the chain's recipe."""
    st = (((state.get("chains") or {}).get(r) or {}).get("steps") or {}).get(str(n)) or {}
    mode = g.get("replay_mode") or st.get("replay_mode") or D.DEFAULT_REPLAY_MODE
    sizes = {arm: (g.get("manifests") or {}).get(arm, {}).get("n_images") for arm in ("cand", "null")}
    rec = defn["recipes"][r]
    warm = {arm: (P.effective_warmup(rec, k)["warmup_epochs_effective"] if isinstance(k, int) and k > 0 else None)
            for arm, k in sizes.items()}
    return {"replay_mode": mode, "train_images": sizes,
            "pool_images": g.get("pool_images", st.get("pool_images")),
            "d_images": g.get("d_images", st.get("d_images")),
            "warmup_epochs_effective": warm}


def _step_rows(defn, ledger, state=None):
    rows = []
    state = state or {}
    chains = list(defn["recipes"])
    for n, step in enumerate(defn["steps"], 1):
        tag = "s%02d_%s" % (n, step["name"])
        te = ledger.get("truth/%d" % n)
        truth = None
        if te:
            d = te["detail"]
            truth = {"verdict": d["verdict"], "p": d["p"], "with_mean": d["with_mean"],
                     "with_sd": d["with_sd"], "without_mean": d["without_mean"],
                     "without_sd": d["without_sd"], "species_failed": d["species"]["failed"]}
        row = {"k": n, "step": step["name"], "tag": tag, "clean": step["clean"], "truth": truth,
               "chains": {}}
        for r in chains:
            g = ledger.get("gate/%s/%d" % (r, n))
            if not g:
                row["chains"][r] = None
                continue
            d = g["decision"]
            a = d["attribution"]
            s = ledger.get("soup/%s/%d" % (r, n))
            eq = D.VERDICT_TO_TRUTH[d["verdict"]]
            row["chains"][r] = {
                "sources": g.get("sources"), "attribution_not_run": sorted(g.get("attribution_not_run") or {}),
                "verdict": d["verdict"], "truth_equivalent": eq,
                "agree": (truth["verdict"] == eq) if truth else None,
                "p_data": d["p_data"], "p_recipe": d["p_recipe"], "inc": d["inc"],
                "cand_mean": d["cand_mean"], "cand_sd": d["cand_sd"],
                "null_mean": d["null_mean"], "null_sd": d["null_sd"],
                "guards": {k: bool(v["passed"]) for k, v in d["guards"].items()},
                "attribution": {"blame": a.get("blame"), "recipe_flag": a.get("recipe_flag"),
                                "class_vs_loc": a.get("class_vs_loc"),
                                "map_delta": a.get("map_delta"),
                                "agnostic_delta": a.get("agnostic_delta"),
                                "species_failed": a.get("species_failed")},
                "reason": d["reason"], "warnings": d.get("warnings", []),
                "soup_choice": s["choice"] if s else None,
                "incumbent_after": (s["incumbent_after"]["run_id"] if s
                                    else g["incumbent_before"]["run_id"]
                                    if d["verdict"] != G.ACCEPT else None)}
            row["chains"][r].update(_train_sizes(defn, state, r, n, g))
            if d["guards"]["flips"].get("mode") == G.FLIPS_NET:
                # protocol v2: v1's verdict on the very same runs, from the recorded negative counts
                v1 = G.v1_counterfactual(d)["verdict"]
                row["chains"][r].update(v1_counterfactual=v1, v1_agree=(
                    truth["verdict"] == D.VERDICT_TO_TRUTH[v1]) if truth else None)
        rows.append(row)
    return rows


def _label_steps(defn):
    """{"bswap": name, "unverified": name} of the steps whose labels are known
    to be bad or were not verified, for the steps the experiment has: the
    pilot's Bswap, and the step whose kind is 'unverified' (inc/realloop.py)."""
    out = {}
    for s in defn["steps"]:
        if s["name"] == BSWAP_STEP and "bswap" not in out:
            out["bswap"] = s["name"]
        if s.get("kind") == UNVERIFIED_KIND and "unverified" not in out:
            out["unverified"] = s["name"]
    return out


def _label_attribution(steps, name, r):
    """Chain r's decision on step `name`, or None when there is no such step or
    the chain has not decided it."""
    c = next((s["chains"][r] for s in steps if s["step"] == name), None) if name else None
    if c is None:
        return None
    return {"verdict": c["verdict"], "class_vs_loc": c["attribution"]["class_vs_loc"],
            "attributed_to_labels": c["attribution"]["class_vs_loc"] == G.LABELS}


def _final_groups(defn):
    seeds = defn["seeds"]
    groups = []
    if defn["type"] == "chain":
        for r in defn["recipes"]:
            groups.append(("chain %s: final incumbent" % r, ["final__%s__incumbent" % r]))
    groups.append(("base %s" % defn["base"]["name"], ["final__base__s%d" % s for s in seeds]))
    if defn["type"] == "chain" and defn.get("truth"):
        groups.append(("T_final (union of clean data)", ["final__Tfinal__s%d" % s for s in seeds]))
    return groups


def build(exp):
    """Write report.md and report.json for exp; returns the report dict."""
    paths = D.Paths(exp)
    defn = _read(paths.exp_json)
    state = _read(paths.state)
    notes = []
    ledger = read_ledger(paths, notes)
    rep = {"exp": exp, "type": defn["type"], "testing": bool(defn.get("testing")),
           "generated_utc": D._utc(), "done": bool(state.get("done")),
           "done_utc": state.get("done_utc"), "blocked": state.get("blocked", {}),
           "seeds": defn["seeds"], "exams": list(REPORT_EXAMS), "notes": notes,
           "code": state.get("code"),
           "interventions": [e for e in ledger.values() if e.get("type") in ("unblock", "code_repin")],
           "attribution_scope": defn.get("attribution_scope"),
           "effective_warmup": defn.get("effective_warmup"),
           "replay_mode": (defn.get("replay_mode", D.DEFAULT_REPLAY_MODE) if defn["type"] == "chain"
                           else None),
           "gate": gate_record(defn, state, ledger)}

    if defn["type"] == "chain":
        steps = _step_rows(defn, ledger, state)
        rep["steps"] = steps
        rep["agreement"] = {}
        rep["chains"] = {}
        rep["label_steps"] = _label_steps(defn)
        for r in defn["recipes"]:
            cmp = [s["chains"][r]["agree"] for s in steps
                   if s["chains"][r] is not None and s["chains"][r]["agree"] is not None]
            rep["agreement"][r] = {"agree": sum(cmp), "compared": len(cmp),
                                   "rate": (sum(cmp) / float(len(cmp))) if cmp else None}
            v1 = [s["chains"][r]["v1_agree"] for s in steps
                  if s["chains"][r] is not None and s["chains"][r].get("v1_agree") is not None]
            if v1:
                rep["agreement"][r]["v1_counterfactual"] = {"agree": sum(v1), "compared": len(v1)}
            ch = state["chains"][r]
            rep["chains"][r] = {
                "rejects_without_source_attribution": [
                    s["tag"] for s in steps if s["chains"][r] is not None and s["chains"][r]["verdict"] == G.REJECT
                    and "leave_one_source_out" in s["chains"][r]["attribution_not_run"]],
                "phase": ch["phase"], "accepted": ch["accepted"], "neutral": ch["neutral"],
                "quarantined": ch["quarantined"],
                "final_incumbent": (ch.get("incumbent") or {}).get("run_id"),
                "recipe": defn["recipes"][r],
                "bswap": _label_attribution(steps, rep["label_steps"].get("bswap"), r),
                "unverified": _label_attribution(steps, rep["label_steps"].get("unverified"), r)}
        check = v2_check(defn, state, steps, rep["gate"])
        if check is not None:
            rep["v2_check"] = check

    final = []
    for label, ids in _final_groups(defn):
        row = {"model": label, "runs": ids, "exams": {}}
        prod = set()
        for exam in REPORT_EXAMS:
            vals = [score_values(paths, rid, exam) for rid in ids]
            got = [v for v in vals if v is not None]
            prod |= {v[2] for v in got}
            row["exams"][exam] = {"twelve": mean_sd([v[0] for v in got]),
                                  "agnostic": mean_sd([v[1] for v in got])}
        row["production"] = sorted(prod, key=str)
        final.append(row)
    rep["final"] = final

    by_owner, n_runs = collections.defaultdict(float), collections.Counter()
    for rid, r in state["runs"].items():
        by_owner[r["owner"]] += run_seconds(paths, rid, r)
        n_runs[r["owner"]] += 1
    rep["gpu_hours"] = {o: {"runs": n_runs[o], "hours": by_owner[o] / 3600.0} for o in sorted(by_owner)}
    rep["gpu_hours_total"] = sum(by_owner.values()) / 3600.0

    D._write_json(paths.root / "report.json", rep)
    text = render_md(rep, defn)
    tmp = paths.root / (".report.md.%d.tmp" % os.getpid())
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, paths.root / "report.md")
    return rep


def gate_record(defn, state, ledger):
    """The gate config a chain experiment is decided with, as pinned at init
    (driver.pinned_gate_config: state.json's gate_pin, else the protocol's
    defaults), cross-checked against the config every gate, soup, truth and
    gate_pin entry of the ledger recorded, and against what exp.json resolves
    to now. None for a baseline, which makes no gate decision."""
    if defn["type"] != "chain":
        return None
    pinned = D.pinned_gate_config(state)
    cfg = dataclasses.asdict(pinned)
    pin = state.get(D.GATE_PIN)
    want = G.config_record(pinned)
    checked, mismatches = collections.Counter(), []
    for e in ledger.values():
        t = e.get("type")
        if t == "gate":
            got, ok = e["decision"].get("config"), e["decision"].get("config") == want
        elif t in ("soup", "truth"):
            got = {"metric": e.get("metric") if t == "soup" else (e.get("detail") or {}).get("metric")}
            ok = got["metric"] == pinned.metric
        elif t == "gate_pin":
            got, ok = e.get("config"), e.get("config") == cfg
        else:
            continue
        checked[t] += 1
        if not ok:
            mismatches.append({"id": e["id"], "recorded": got})
    try:
        changed = D.gate_changes(pinned, D.gate_config(defn))
    except D.DriverError as e:
        changed = {"exp.json": str(e)}
    return {"flips_mode": cfg["flips_mode"], "pinned": pin is not None,
            "block": pin["block"] if pin is not None else None, "config": cfg,
            "ledger_checked": dict(checked), "ledger_mismatches": mismatches, "exp_json_changes": changed}


def v2_check(defn, state, steps, gate):
    """The pre-registered prospective check of protocol v2 (docs/
    INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)", Pre-registration),
    for an experiment decided on net flips; None otherwise.

    Over every (chain, step) with a gate decision and a truth decision:
    agree_v2 counts the net verdicts that match the truth arm, agree_v1 the v1
    counterfactual verdicts on the same runs (gate.v1_counterfactual).
    Outcome: 'refuted' when agree_v2 < agree_v1 or when v2 ACCEPTed any step
    not marked clean (Bswap, Breal, UNVERIFIED), whatever the truth arm says;
    else 'supported' when agree_v2 > agree_v1; else 'inconclusive' (a tie,
    including no step on which the two rules differ). 'final' is true once the
    experiment is done; before that the outcome is provisional."""
    if gate is None or gate["flips_mode"] != G.FLIPS_NET:
        return None
    out = {"final": bool(state.get("done")), "truth_arm": bool(defn.get("truth"))}
    compared, discordant, bad = [], [], []
    for s in steps:
        for r, c in s["chains"].items():
            if c is None:
                continue
            if not s["clean"] and c["verdict"] == G.ACCEPT:
                bad.append({"chain": r, "step": s["step"], "v1_counterfactual": c.get("v1_counterfactual")})
            if c.get("v1_counterfactual") is not None and c["verdict"] != c["v1_counterfactual"]:
                discordant.append({"chain": r, "step": s["step"], "v2": c["verdict"],
                                   "v1_counterfactual": c["v1_counterfactual"],
                                   "truth": s["truth"]["verdict"] if s["truth"] else None})
            if s["truth"] is not None and c.get("v1_agree") is not None:
                compared.append((c["agree"], c["v1_agree"]))
    a2, a1 = sum(1 for x, _ in compared if x), sum(1 for _, y in compared if y)
    if bad or a2 < a1:
        outcome = "refuted"
    elif a2 > a1:
        outcome = "supported"
    else:
        outcome = "inconclusive"
    out.update(outcome=outcome, compared=len(compared), agree_v2=a2, agree_v1_counterfactual=a1,
               discordant=discordant, non_clean_accepted=bad)
    return out


# ------------------------------------------------------------------ markdown
def _f(x, nd=4):
    return "-" if x is None else ("%.*f" % (nd, x))


def _ms(m):
    if m["n"] == 0:
        return "-"
    if m["sd"] is None:
        return "%.4f" % m["mean"]
    return "%.4f +- %.4f" % (m["mean"], m["sd"])


def _yn(b):
    return "-" if b is None else ("yes" if b else "no")


REPLAY_TEXT = {"sample": "sample (cand on D_k + R1, null on R1 + R2; R1, R2 disjoint samples of |D_k| images "
                         "from the chain's accepted pool)",
               "full": "full rehearsal (cand on the chain's whole accepted pool + D_k, null on the whole "
                       "accepted pool)"}


FLIPS_TEXT = {"negative": "negative flips (protocol v1: images correct under the incumbent and incorrect "
                           "under the run)",
              "net": "net flips (protocol v2: negative - positive flips; a positive flip is an image "
                     "incorrect under the incumbent and correct under the run)"}


def _gate_lines(g):
    """The header's gate lines: the pinned config, then the cross-checks."""
    block = ("pinned at init (state.json gate_pin) from exp.json gate block %s"
             % json.dumps(g["block"], sort_keys=True) if g["pinned"]
             else "no gate block in exp.json at init (the protocol's defaults; nothing pinned)")
    out = ["- Gate: the flips guard counts %s; %s; GateConfig %s"
           % (FLIPS_TEXT.get(g["flips_mode"], g["flips_mode"]), block,
              ", ".join("%s=%s" % kv for kv in g["config"].items()))]
    n = g["ledger_checked"]
    if g["ledger_mismatches"]:
        for m in g["ledger_mismatches"]:
            out.append("- GATE CONFIG MISMATCH: ledger entry %s recorded %s, not the pinned config"
                       % (m["id"], json.dumps(m["recorded"], sort_keys=True)))
    else:
        out.append("- Gate config check: the ledger's %d gate, %d soup and %d truth entries%s record the pinned "
                   "config" % (n.get("gate", 0), n.get("soup", 0), n.get("truth", 0),
                               " and its gate_pin entry" if n.get("gate_pin") else ""))
    if g["exp_json_changes"]:
        out.append("- GATE CONFIG CHANGED: exp.json no longer resolves to the pinned config (%s as [pinned, now]); "
                   "the driver refuses to advance until it is restored" % g["exp_json_changes"])
    return out


V2_OUTCOME_TEXT = {
    "supported": "supported: v2 agrees with the truth arm on more steps than v1 on the same runs, and accepted no "
                 "step not marked clean",
    "refuted": "refuted: v2 agrees with the truth arm on fewer steps than v1 on the same runs, or accepted a step "
               "not marked clean",
    "inconclusive": "inconclusive: v2 and v1 on the same runs agree with the truth arm on as many steps, and v2 "
                    "accepted no step not marked clean"}


def _v2_check_lines(c):
    out = ["## Protocol v2 check (pre-registered)", "",
           "docs/INCREMENTAL_PROTOCOL.md, Gate, \"Protocol v2 (net flips)\", Pre-registration. The v1 verdict "
           "of each step is gate.v1_counterfactual: the same runs, P_data and regression and species guards, with "
           "v1's flips guard on the negative counts the net guard recorded.", ""]
    if not c["truth_arm"]:
        out.append("- No truth arm: agreement cannot be measured; only the non-clean acceptances below apply.")
    out.append("- Outcome%s: %s" % ("" if c["final"] else " (provisional, the experiment is not done)",
                                    V2_OUTCOME_TEXT[c["outcome"]]))
    out.append("- Agreement with the truth arm over %d (chain, step) pairs: v2 %d, v1 on the same runs %d"
               % (c["compared"], c["agree_v2"], c["agree_v1_counterfactual"]))
    out.append("- Steps where the two rules differ: %s" % ("; ".join(
        "%s %s: v2 %s, v1 %s, truth %s" % (x["chain"], x["step"], x["v2"], x["v1_counterfactual"], x["truth"] or "-")
        for x in c["discordant"]) or "none"))
    out.append("- Steps not marked clean that v2 accepted: %s" % ("; ".join(
        "%s %s (v1 on the same runs: %s)" % (x["chain"], x["step"], x["v1_counterfactual"] or "-")
        for x in c["non_clean_accepted"]) or "none"))
    out.append("")
    return out


def _sizes(c):
    t = c.get("train_images") or {}
    return "%s/%s" % (t.get("cand") if t.get("cand") is not None else "-",
                      t.get("null") if t.get("null") is not None else "-")


def _builder_warmup(wu):
    """Every incremental effective-warmup record of the builder's table, for
    either replay mode's layout."""
    out = []
    for rv in wu["incremental"].values():
        for v in rv.values():
            out += [v] if "warmup_epochs_effective" in v else [x for arm in v.values() for x in arm.values()]
    return out


def render_md(rep, defn):
    out = []
    t = " [TESTING]" if rep["testing"] else ""
    out.append("# INC report: %s%s" % (rep["exp"], t))
    out.append("")
    if rep["testing"]:
        out.append("**TESTING experiment: every score here is a test-mode score (production=false, "
                   "TEST- stamp). None of it is a protocol result.**")
        out.append("")
    out.append("- Type: %s; seeds %s; generated %s" % (rep["type"], rep["seeds"], rep["generated_utc"]))
    out.append("- Done: %s" % ("yes (%s)" % rep["done_utc"] if rep["done"] else "no"))
    for unit, b in sorted(rep["blocked"].items()):
        out.append("- BLOCKED %s: %s" % (unit, b["error"]))
    for n in rep.get("notes") or []:
        out.append("- Note: %s" % n)
    if rep.get("replay_mode"):
        out.append("- Replay mode: %s" % REPLAY_TEXT.get(rep["replay_mode"], rep["replay_mode"]))
    if rep.get("gate"):
        out.extend(_gate_lines(rep["gate"]))
    code = rep.get("code") or {}
    if code.get("modules"):
        out.append("- Decision code pinned %s (%s): %s" % (
            code.get("pinned_utc"), code.get("reason"),
            ", ".join("%s %s" % (m.split("/")[-1], (h or "missing")[:12])
                      for m, h in sorted(code["modules"].items()))))
    out.append("")

    if rep["type"] == "chain":
        chains = list(defn["recipes"])
        out.append("## Decisions per step")
        out.append("")
        out.append("Chain verdicts against the truth arm (ACCEPT ~ helps, HOLD ~ neutral, "
                   "REJECT ~ hurts).")
        out.append("")
        net = "v2_check" in rep
        head = ["Step", "Clean", "Truth (P)"] + ["%s (P_data)" % r for r in chains] + \
               ["agrees: %s" % r for r in chains] + (["v1 same runs: %s" % r for r in chains] if net else [])
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for s in rep["steps"]:
            tr = s["truth"]
            cells = [s["tag"], _yn(s["clean"]),
                     "%s (%.2f)" % (tr["verdict"], tr["p"]) if tr else "-"]
            for r in chains:
                c = s["chains"][r]
                cells.append("%s (%.2f)" % (c["verdict"], c["p_data"]) if c else "-")
            for r in chains:
                c = s["chains"][r]
                cells.append(_yn(c["agree"]) if c else "-")
            for r in (chains if net else ()):
                c = s["chains"][r]
                cells.append("%s (agrees: %s)" % (c["v1_counterfactual"], _yn(c["v1_agree"]))
                             if c and c.get("v1_counterfactual") else "-")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        for r in chains:
            a = rep["agreement"][r]
            out.append("- %s agrees with the truth arm on %d of %d compared steps%s"
                       % (r, a["agree"], a["compared"],
                          "; v1 on the same runs would on %d of %d" % (a["v1_counterfactual"]["agree"],
                                                                     a["v1_counterfactual"]["compared"])
                          if a.get("v1_counterfactual") else ""))
        out.append("")
        if net:
            out.extend(_v2_check_lines(rep["v2_check"]))

        out.append("## Gate details")
        for r in chains:
            ch = rep["chains"][r]
            out.append("")
            out.append("### %s (%s)" % (r, ch["phase"]))
            out.append("")
            out.append("Accepted %s; neutral %s; quarantined %s; final incumbent %s."
                       % (ch["accepted"] or "none", ch["neutral"] or "none",
                          ch["quarantined"] or "none", ch["final_incumbent"] or "-"))
            out.append("")
            head = ["Step", "Verdict", "P_data", "P_recipe", "inc", "cand", "null", "images cand/null",
                    "regression", "species", "flips", "blame", "class vs loc", "species failed", "soup"]
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "---|" * len(head))
            for s in rep["steps"]:
                c = s["chains"][r]
                if not c:
                    out.append("| %s |" % s["tag"] + " - |" * (len(head) - 1))
                    continue
                g, a = c["guards"], c["attribution"]
                out.append("| " + " | ".join([
                    s["tag"], c["verdict"], "%.3f" % c["p_data"], "%.3f" % c["p_recipe"],
                    _f(c["inc"]), "%s +- %s" % (_f(c["cand_mean"]), _f(c["cand_sd"])),
                    "%s +- %s" % (_f(c["null_mean"]), _f(c["null_sd"])), _sizes(c),
                    "pass" if g.get("regression") else "FAIL",
                    "pass" if g.get("species") else "FAIL",
                    "pass" if g.get("flips") else "FAIL",
                    a["blame"] or "-", a["class_vs_loc"] or "-",
                    ", ".join(a["species_failed"] or []) or "-",
                    c["soup_choice"] or "-"]) + " |")
        out.append("")
        steps_by_name = {s["name"]: s for s in defn["steps"]}
        for key in ("bswap", "unverified"):
            name = (rep.get("label_steps") or {}).get(key)
            if not name:
                continue
            out.append("## Attribution of %s" % name)
            out.append("")
            st = steps_by_name.get(name) or {}
            if key == "unverified":
                out.append("%s: %s images of %s that verify did not admit (image verdicts %s); its labels "
                           "are as the species join gave them." % (
                               name, st.get("n_images", "-"), st.get("source", "-"),
                               ", ".join("%s %s" % kv for kv in sorted((st.get("verdicts") or {}).items()))
                               or "-"))
                out.append("")
            for r in chains:
                b = rep["chains"][r][key]
                out.append("- %s: %s" % (r, "not decided yet" if b is None else
                                         "%s, class_vs_loc=%s, attributed to labels: %s"
                                         % (b["verdict"], b["class_vs_loc"], _yn(b["attributed_to_labels"]))))
            out.append("")

    if rep["type"] == "chain":
        out.append("## Attribution not run")
        out.append("")
        scope = rep.get("attribution_scope") or {}
        for k, v in sorted((scope.get("not_run") or {}).items()):
            out.append("- Step %s: %s" % (k, v))
        if not scope:
            out.append("- The driver runs protocol attribution steps 1-3 only (gate.decide); steps 4 "
                       "(label audit) and 5 (leave-one-source-out) are not run.")
        for r in chains:
            miss = rep["chains"][r]["rejects_without_source_attribution"]
            if miss:
                out.append("- %s: REJECT of a multi-source increment without leave-one-source-out "
                           "attribution: %s" % (r, ", ".join(miss)))
        out.append("")
        wu = rep.get("effective_warmup")
        if wu and wu.get("incremental_effective_epochs"):
            inc = _builder_warmup(wu)
            out.append("## Warmup as run")
            out.append("")
            out.append("%s. Incremental runs: warmup lasts %.2f-%.2f of their %d epochs (the recipe asks for %s); "
                       "the cold base run's lasts %.2f of %d."
                       % (wu["note"], wu["incremental_effective_epochs"]["min"],
                          wu["incremental_effective_epochs"]["max"], inc[0]["epochs"],
                          inc[0]["warmup_epochs_nominal"], wu["cold"]["base"]["warmup_epochs_effective"],
                          wu["cold"]["base"]["epochs"]))
            ran = {arm: [c["warmup_epochs_effective"][arm] for s in rep["steps"] for c in s["chains"].values()
                         if c and c.get("warmup_epochs_effective", {}).get(arm) is not None]
                   for arm in ("cand", "null")}
            if ran["cand"] or ran["null"]:
                out.append("")
                out.append("From the decided steps' train sizes: cand warmup %s, null warmup %s epochs."
                           % tuple("%.2f-%.2f" % (min(v), max(v)) if v else "-" for v in (ran["cand"], ran["null"])))
            out.append("")

    if rep.get("interventions"):
        out.append("## Interventions")
        out.append("")
        for e in sorted(rep["interventions"], key=lambda e: e.get("utc") or ""):
            if e["type"] == "unblock":
                out.append("- %s unblock %s (%s): %s; runs reset: %s"
                           % (e.get("utc"), e.get("unit"), "auto" if e.get("auto") else "by hand",
                              e.get("reason"), ", ".join(e.get("reset_runs") or []) or "none"))
            else:
                out.append("- %s code re-pinned (%s): %s" % (e.get("utc"), ", ".join(e.get("changed") or []),
                                                            e.get("reason")))
        out.append("")

    out.append("## Final quality")
    out.append("")
    out.append("From the final runs only (the only runs that read test). 12-class = the scorer's "
               "species_map50_95 (cwd12 ids present in the exam); agn = class-agnostic "
               "mAP50-95; mean +- sd over seeds where there are seeds.")
    out.append("")
    head = ["Model"] + ["%s %s" % (e, k) for e in REPORT_EXAMS for k in ("12-class", "agn")]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for row in rep["final"]:
        cells = [row["model"]]
        for e in REPORT_EXAMS:
            cells += [_ms(row["exams"][e]["twelve"]), _ms(row["exams"][e]["agnostic"])]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("## GPU-hours")
    out.append("")
    out.append("| Unit | Runs | GPU-hours |")
    out.append("|---|---|---|")
    for o, v in rep["gpu_hours"].items():
        out.append("| %s | %d | %.2f |" % (o.replace("chain:", "chain "), v["runs"], v["hours"]))
    out.append("| total | %d | %.2f |" % (sum(v["runs"] for v in rep["gpu_hours"].values()),
                                          rep["gpu_hours_total"]))
    out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write report.md and report.json for an INC experiment.")
    ap.add_argument("--exp", required=True)
    a = ap.parse_args(argv)
    try:
        rep = build(a.exp)
    except (OSError, ValueError, KeyError, D.DriverError) as e:
        print("[inc.report] ERROR: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1
    print("[inc.report] %s: wrote %s and %s%s"
          % (a.exp, D.Paths(a.exp).root / "report.md", D.Paths(a.exp).root / "report.json",
             " (TESTING)" if rep["testing"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
