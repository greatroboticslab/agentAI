"""Score a finished child experiment against what its proposal predicted (docs/INC_AUTOPILOT.md (c)).

A lever or a brain earns trust only by being right about experiments that
then ran. This module turns a child experiment's report into
`experiments.record_result` + `experiments.verdict` and folds the result into
the track record the page and the digest show.

Noise floor. The floor is the dev spread of the b0_v1 seeds: the sample sd of
the dev 12-class mAP50-95 over B0's three cold seeds (b0_v1/report.json,
final row "base ...", exams.dev.twelve.sd), times NOISE_K. NOISE_K = 2 is the
same multiple the pinned gate uses for its regression guard (mean(cand) >=
inc - 2 sd(null)) and D8 uses for "underpowered": a dev difference smaller
than two seed sds is not distinguishable from reseeding. It is not the M1
recipe-keyed floor at db.py:549. No b0 report -> no floor -> `insufficient`,
never a guessed floor.

Metrics (the brain's predicted.metric):
  * agreement - whether each chain agrees with the truth arm, step by step
    (report.steps[k].chains[r].agree). Scored PER CHAIN, pairing the parent's
    and the child's steps by step tag: `gained` steps the child agrees on and
    the parent did not, `lost` the reverse. The noise model is the exact
    two-sided sign test (McNemar) on those discordant steps at
    AGREEMENT_ALPHA = 0.05: p < alpha -> better / worse; otherwise
    within_noise, but only when the design could have shown a change at all
    (every step flipping in the parent's more frequent direction would give
    p < alpha); when it could not, the chain is `insufficient`. A 7-step pilot
    whose parent agrees on 3 steps has at most 4 steps to gain (p >= 0.125),
    so its agreement verdicts are `insufficient` until more steps or
    replicated pilots exist. The verdict is the chain the prediction names
    (predicted.chain), else the common verdict of every chain compared in
    both (chains that disagree -> insufficient). Per-chain deltas, gained,
    lost and p are recorded. experiments.verdict's seed floor does not apply
    (steps are not seeds), so the step count is never passed to it as n; the
    record's n is 0 and its noise_floor None, and extra.verdict_rule names
    this rule. The rule lives here; D13 (thresholds.json) reads only the
    verdict it produces.
  * dev_twelve - dev 12-class mAP50-95 of one final-table row: the base row
    for a baseline experiment, otherwise the chain final incumbent with the
    highest dev mean (or the row the prediction names). The delta is child
    minus parent; n is the smaller of the two rows' seed counts, so a chain's
    single-run final incumbent gives `insufficient` (experiments.verdict needs
    n >= 3) and says so.

Only dev is read: every report passes through brain_plan.dev_only() first.

Files (lab, model.CAMPAIGN_DIR unless given):
  track_record.jsonl  append-only events: {"kind": "outcome" | "validation", ...}
  track_record.json   the fold of those events (per lever, per proposer)
  _brain/<domain>/experiments.jsonl  one experiments.record_result row per
                      outcome, which /api/brain/{domain}/experiments already reads.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from . import brain_plan as BP
from . import model as M

NOISE_K = 2.0
# Two-sided level of the per-chain sign test on paired steps (agreement metric).
AGREEMENT_ALPHA = 0.05
AGREEMENT_RULE = ("agreement: exact two-sided sign test (McNemar) per chain on steps paired by tag, "
                  "alpha %g; within_noise only when p < alpha was attainable; otherwise "
                  "insufficient" % AGREEMENT_ALPHA)
TRACK_EVENTS = M.CAMPAIGN_DIR / "track_record.jsonl"
TRACK_SUMMARY = M.CAMPAIGN_DIR / "track_record.json"
EXPERIMENTS_FILE = M.BRAIN_DIR / M.DOMAIN / "experiments.jsonl"
EXPECTED = {"up": "better", "down": "worse", "none": "within_noise"}


THRESHOLDS_FILE = Path(__file__).resolve().parent / "thresholds.json"


def contradiction_table(path=None):
    """{direction: [verdicts that contradict it]} from thresholds.json D13, or None."""
    try:
        with open(path or THRESHOLDS_FILE, "r", encoding="utf-8") as fh:
            table = json.load(fh)["D13"]["contradicts"]["value"]
        return table if isinstance(table, dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_report(path):
    with open(path, "r", encoding="utf-8") as fh:
        return BP.dev_only(json.load(fh))


def _dev_twelve(row):
    d = (((row or {}).get("exams") or {}).get(M.DECISION_EXAM) or {}).get("twelve") or {}
    return d.get("mean"), d.get("sd"), d.get("n") or 0


def noise_floor(b0_report, k=NOISE_K, artifact="b0_v1/report.json"):
    """{"floor", "sd", "n", "k", "cite", "reason"} from B0's base row on dev."""
    rep = BP.dev_only(b0_report) if isinstance(b0_report, dict) else None
    if not rep:
        return {"floor": None, "sd": None, "n": 0, "k": k, "cite": None,
                "reason": "no b0 report: no noise floor"}
    for i, row in enumerate(rep.get("final") or []):
        if not str((row or {}).get("model", "")).startswith("base"):
            continue
        mean, sd, n = _dev_twelve(row)
        if isinstance(sd, (int, float)) and not isinstance(sd, bool) and n >= 2:
            return {"floor": k * float(sd), "sd": float(sd), "n": int(n), "k": k,
                    "cite": M.cite(artifact, sd, pointer="/final/%d/exams/dev/twelve/sd" % i),
                    "reason": ""}
        return {"floor": None, "sd": sd, "n": n, "k": k, "cite": None,
                "reason": "b0 base row has no dev sd over >= 2 seeds"}
    return {"floor": None, "sd": None, "n": 0, "k": k, "cite": None,
            "reason": "b0 report has no base row"}


def agreement(report):
    """{"best": chain, "rate", "compared", "per_chain": {...}} or None."""
    ag = (report or {}).get("agreement") or {}
    rows = {r: v for r, v in ag.items() if isinstance(v, dict)
            and isinstance(v.get("rate"), (int, float)) and v.get("compared")}
    if not rows:
        return None
    best = sorted(rows, key=lambda r: (-rows[r]["rate"], r))[0]
    return {"best": best, "rate": rows[best]["rate"], "compared": int(rows[best]["compared"]),
            "per_chain": {r: rows[r]["rate"] for r in sorted(rows)}}


def agreement_flags(report):
    """{chain: {step tag: agree}} of every step whose chain was compared with the truth arm."""
    out = {}
    for i, st in enumerate((report or {}).get("steps") or []):
        if not isinstance(st, dict):
            continue
        key = str(st.get("tag") or st.get("step") or st.get("k") or i)
        for r, c in (st.get("chains") or {}).items():
            a = (c or {}).get("agree") if isinstance(c, dict) else None
            if isinstance(a, bool):
                out.setdefault(r, {})[key] = a
    return out


def sign_test_p(gained, lost):
    """Exact two-sided sign-test p for `gained` vs `lost` discordant pairs."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def chain_agreement(parent_flags, child_flags, alpha=AGREEMENT_ALPHA):
    """One chain's paired comparison: steps, gained, lost, delta, p, verdict, reason."""
    steps = sorted(set(parent_flags) & set(child_flags))
    n = len(steps)
    gained = sum(1 for s in steps if child_flags[s] and not parent_flags[s])
    lost = sum(1 for s in steps if parent_flags[s] and not child_flags[s])
    agree_p = sum(1 for s in steps if parent_flags[s])
    out = {"paired_steps": n, "parent_agree": agree_p,
           "child_agree": sum(1 for s in steps if child_flags[s]),
           "gained": gained, "lost": lost, "delta": None, "p": None,
           "best_attainable_p": None, "verdict": "insufficient", "reason": ""}
    if not n:
        out["reason"] = "no step compared in both experiments"
        return out
    out["delta"] = (out["child_agree"] - agree_p) / float(n)
    p = sign_test_p(gained, lost)
    best = sign_test_p(max(agree_p, n - agree_p), 0)
    out["p"], out["best_attainable_p"] = p, best
    if p < alpha:
        out["verdict"] = "better" if gained > lost else "worse"
        out["reason"] = "sign test p %.4g < %g" % (p, alpha)
    elif best >= alpha:
        out["reason"] = ("%d paired steps from a parent agreeing on %d: even every step flipping "
                         "one way gives p %.4g >= %g" % (n, agree_p, best, alpha))
    else:
        out["verdict"] = "within_noise"
        out["reason"] = "sign test p %.4g >= %g (%d gained, %d lost)" % (p, alpha, gained, lost)
    return out


def agreement_outcome(child, parent, chain=None, alpha=AGREEMENT_ALPHA):
    """{"verdict", "reason", "delta", "chain", "per_chain"} of the agreement metric."""
    fc, fp = agreement_flags(child), agreement_flags(parent)
    per = {r: chain_agreement(fp[r], fc[r], alpha) for r in sorted(set(fc) & set(fp))}
    out = {"chain": chain, "per_chain": per, "alpha": alpha, "rule": AGREEMENT_RULE}
    if chain:
        one = per.get(chain)
        if one is None:
            out.update(verdict="insufficient", delta=None,
                       reason="chain %r is not compared in both experiments" % chain)
        else:
            out.update(verdict=one["verdict"], delta=one["delta"], reason=one["reason"])
        return out
    if not per:
        out.update(verdict="insufficient", delta=None, reason="no chain is compared in both")
        return out
    deltas = [v["delta"] for v in per.values() if v["delta"] is not None]
    out["delta"] = sum(deltas) / len(deltas) if deltas else None
    verdicts = {v["verdict"] for v in per.values()}
    if len(verdicts) == 1:
        out.update(verdict=verdicts.pop(), reason="every chain: %s"
                   % "; ".join("%s %s" % (r, v["reason"]) for r, v in per.items()))
    else:
        out.update(verdict="insufficient",
                   reason="the chains disagree (%s); name a chain to score one"
                          % ", ".join("%s %s" % (r, v["verdict"]) for r, v in per.items()))
    return out


def helps_accepted(report):
    """{chain: number of truth-'helps' steps the chain ACCEPTed} (L1's success criterion)."""
    out = {}
    for st in (report or {}).get("steps") or []:
        if ((st.get("truth") or {}).get("verdict")) != "helps":
            continue
        for r, c in (st.get("chains") or {}).items():
            out.setdefault(r, 0)
            if (c or {}).get("verdict") == "ACCEPT":
                out[r] += 1
    return out


def dev_row(report, row_label=None):
    """(index, label, mean, sd, n) of the final-table row the dev_twelve metric reads."""
    rows = [(i, r) for i, r in enumerate((report or {}).get("final") or [])
            if isinstance(r, dict) and _dev_twelve(r)[0] is not None]
    if row_label:
        rows = [(i, r) for i, r in rows if str(r.get("model", "")).startswith(row_label)]
    elif (report or {}).get("type") == "baseline":
        rows = [(i, r) for i, r in rows if str(r.get("model", "")).startswith("base")]
    else:
        rows = [(i, r) for i, r in rows if str(r.get("model", "")).startswith("chain")]
    if not rows:
        return None
    i, r = sorted(rows, key=lambda x: (-_dev_twelve(x[1])[0], str(x[1].get("model"))))[0]
    mean, sd, n = _dev_twelve(r)
    return i, r.get("model"), mean, sd, int(n)


def score(proposal, child_report, parent_report, b0_report=None, child_exp="child",
          parent_exp="parent", k=NOISE_K):
    """The outcome record of one child experiment against its proposal.

    Returns {"result" (experiments.record_result row), "verdict", "expected",
    "correct", "contradicted", "measure", "floor", ...}. `correct` is None when
    the verdict is insufficient or nothing machine-readable was predicted;
    `contradicted` follows D13's table (None when thresholds.json lacks it).
    The record's lever / child_exp / predicted / verdict keys are what D13
    reads from the campaign context's "outcomes" list.
    A proposal without its own `predicted` block (a deterministic one) may be
    scored against a lever row's by passing
    dict(proposal, predicted=row["predicted"]).
    """
    from ..brain import experiments
    child = BP.dev_only(child_report or {})
    parent = BP.dev_only(parent_report or {})
    pred = (proposal or {}).get("predicted") or {}
    metric = pred.get("metric") or "agreement"
    measure, mean, std, n, floor, floor_info = {}, None, None, 0, None, None
    fixed_verdict = None
    if metric == "agreement":
        ag = agreement_outcome(child, parent, pred.get("chain"))
        measure = {"child": agreement(child), "parent": agreement(parent),
                   "chain": ag["chain"], "per_chain": ag["per_chain"], "reason": ag["reason"],
                   "helps_accepted": {"child": helps_accepted(child),
                                      "parent": helps_accepted(parent)}}
        mean, fixed_verdict = ag["delta"], ag["verdict"]
        floor_info = {"floor": None, "rule": ag["rule"], "alpha": ag["alpha"]}
    elif metric == "dev_twelve":
        floor_info = noise_floor(b0_report, k=k)
        floor = floor_info["floor"]
        r_c = dev_row(child, pred.get("row"))
        r_p = dev_row(parent, pred.get("row"))
        measure = {"child": r_c and {"index": r_c[0], "model": r_c[1], "mean": r_c[2],
                                     "sd": r_c[3], "n": r_c[4]},
                   "parent": r_p and {"index": r_p[0], "model": r_p[1], "mean": r_p[2],
                                      "sd": r_p[3], "n": r_p[4]}}
        if r_c and r_p:
            mean = r_c[2] - r_p[2]
            n = min(r_c[4], r_p[4])
            if r_c[3] is not None and r_p[3] is not None and r_c[4] and r_p[4]:
                # standard error of the difference of the two means
                std = math.sqrt(r_c[3] ** 2 / r_c[4] + r_p[3] ** 2 / r_p[4])
    else:
        measure = {"error": "unknown metric %r" % (metric,)}
    lever = (proposal or {}).get("lever")
    row = experiments.record_result(
        {"domain": M.DOMAIN, "lever_id": lever, "control": (proposal or {}).get("control"),
         "variants": (proposal or {}).get("params"), "seeds": None},
        recipe=lever, mean=mean, std=std, n=n, noise_floor=floor,
        extra={"metric": metric, "child_exp": child_exp, "parent_exp": parent_exp,
               "proposal_id": (proposal or {}).get("id"),
               "proposed_by": (proposal or {}).get("proposed_by"),
               "floor_source": floor_info, "source": "inc_autopilot.outcome",
               "verdict_rule": (AGREEMENT_RULE if metric == "agreement" else
                                "experiments.verdict: |delta| against %g x B0's dev seed sd, "
                                "n >= 3 seeds" % k)})
    verdict = fixed_verdict if metric == "agreement" else experiments.verdict(row, floor)
    row["verdict"] = verdict
    expected = EXPECTED.get(pred.get("direction")) if pred else None
    correct = None if (verdict == "insufficient" or expected is None) else (verdict == expected)
    # Contradicted is D13's own table (thresholds.json D13.contradicts), so the
    # flag on this record and the diagnosis that reads it cannot disagree. A
    # predicted move that stayed within noise is unconfirmed (correct False),
    # not contradicted. No table -> None (unknown), never a guess.
    table = contradiction_table()
    contradicted = None if table is None else \
        verdict in (table.get(pred.get("direction")) or [])
    return {"kind": "outcome", "utc": M.utc_now(), "lever": lever,
            "proposal_id": (proposal or {}).get("id"),
            "proposed_by": (proposal or {}).get("proposed_by") or "unknown",
            "child_exp": child_exp, "parent_exp": parent_exp, "metric": metric,
            "predicted": pred or None, "expected": expected, "verdict": verdict,
            "correct": correct, "contradicted": contradicted, "delta": mean,
            "n": None if metric == "agreement" else n,
            "floor": floor, "measure": measure, "result": row}


# --- the track record -----------------------------------------------------------------

def _append(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def read_events(path=None):
    path = Path(path or TRACK_EVENTS)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue        # a torn last line from a killed writer
    return out


def _blank_proposer():
    return {"plans": 0, "items": 0, "valid": 0, "cards": 0, "dropped": 0,
            "cites_checked": 0, "cites_failed": 0, "lit_checked": 0, "lit_failed": 0,
            "lit_project_notes": 0, "leaks": 0, "param_refusals": 0,
            "precondition_refusals": 0, "unpriced": 0, "render_refusals": 0, "duplicates": 0,
            "no_diagnosis": 0, "scored": 0, "correct": 0, "contradicted": 0,
            "insufficient": 0}


def fold(events):
    """{"levers": {id: {...}}, "proposers": {actor: {...}}} from track-record events."""
    levers, props = {}, {}
    for ev in events or []:
        if ev.get("kind") == "validation":
            p = props.setdefault(ev.get("proposed_by") or "unknown", _blank_proposer())
            p["plans"] += 1
            for k, v in (ev.get("counts") or {}).items():
                if k in p and isinstance(v, int):
                    p[k] += v
        elif ev.get("kind") == "da_outcome":
            # The adversary role's track record: its predictions scored when their
            # tests landed, and H11 (docs/FUNNEL_AUDIT.md 6 H11, 8.6).
            p = props.setdefault(ev.get("proposed_by") or "unknown", _blank_proposer())
            p["scored"] += len(ev.get("predictions") or [])
            p["correct"] += int(ev.get("correct") or 0)
            p["insufficient"] += sum(1 for x in ev.get("predictions") or [] if x.get("verdict") == "insufficient")
            h = ev.get("h11") or {}
            if h.get("verdict") in ("supported", "falsified"):
                p["h11_scored"] = p.get("h11_scored", 0) + 1
                p["h11_supported"] = p.get("h11_supported", 0) + (1 if h["verdict"] == "supported" else 0)
        elif ev.get("kind") == "outcome":
            lv = levers.setdefault(ev.get("lever") or "unknown",
                                   {"scored": 0, "correct": 0, "contradicted": 0,
                                    "insufficient": 0, "last": None})
            p = props.setdefault(ev.get("proposed_by") or "unknown", _blank_proposer())
            for rec in (lv, p):
                rec["scored"] += 1
                if ev.get("verdict") == "insufficient" or ev.get("correct") is None:
                    rec["insufficient"] += 1
                elif ev.get("correct"):
                    rec["correct"] += 1
                if ev.get("contradicted") is True:
                    rec["contradicted"] += 1
            lv["last"] = {"child_exp": ev.get("child_exp"), "verdict": ev.get("verdict"),
                          "correct": ev.get("correct"), "utc": ev.get("utc")}
    for p in props.values():
        judged = p["scored"] - p["insufficient"]
        p["prediction_accuracy"] = (p["correct"] / float(judged)) if judged else None
        p["citation_failure_rate"] = (
            (p["cites_failed"] + p["lit_failed"]) / float(p["cites_checked"] + p["lit_checked"])
            if (p["cites_checked"] + p["lit_checked"]) else None)
    for lv in levers.values():
        judged = lv["scored"] - lv["insufficient"]
        lv["accuracy"] = (lv["correct"] / float(judged)) if judged else None
    return {"levers": levers, "proposers": props}


def write_summary(events_path=None, summary_path=None):
    summary = fold(read_events(events_path))
    summary["utc"] = M.utc_now()
    path = Path(summary_path or TRACK_SUMMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))
    return summary


def record_validation(validation, events_path=None, summary_path=None):
    """Append one validate.validate() record's counts to the track record."""
    ev = {"kind": "validation", "utc": M.utc_now(),
          "proposed_by": validation.get("proposed_by") or "unknown",
          "model": validation.get("model"), "digest_sha256": validation.get("digest_sha256"),
          "counts": dict(validation.get("counts") or {})}
    _append(events_path or TRACK_EVENTS, ev)
    write_summary(events_path, summary_path)
    return ev


def record(proposal, child_report, parent_report, b0_report=None, child_exp="child",
           parent_exp="parent", events_path=None, summary_path=None, experiments_path=None):
    """score() + append to the track record + the experiments.jsonl row. Returns the outcome."""
    out = score(proposal, child_report, parent_report, b0_report, child_exp, parent_exp)
    _append(events_path or TRACK_EVENTS, out)
    _append(experiments_path or EXPERIMENTS_FILE, out["result"])
    write_summary(events_path, summary_path)
    return out


# --- the funnel audit: H10 from the panel, the devil's advocate's predictions and H11 -----
# docs/FUNNEL_AUDIT.md 6 (H10, H11) and 8.6; runner 5.5.5 and 5.5.7.

H10A_ALPHA = 0.025          # contract 6 H10a: one-sided exact permutation p at 5 v 5 seeds
H10_P_MIN = 0.75            # gate.GateConfig.p_accept: the pinned "helps" line (H10b, H10c)
DA_ROLE = "adversary"


def h10(panel, realloop_report=None):
    """{"H10a", "H10b", "H10c"}: each {"verdict", "why", ...} from panel.json
    (panel.build_panel) and realloop_v2's report.json (the truth arm's verdicts
    of the recovery steps, dev only).

    H10a supported only when the pinned rule says helps AND the 5 v 5 exact
    permutation p <= 0.025; the rule alone is 'helps (decision rule, alpha about
    0.20)', inconclusive; neutral or hurts falsifies it. H10b: U beats U-ctl at
    P >= 0.75. H10c: at least one recovery step is truth 'helps' and, where a
    same-image control exists, its "with" runs beat that control at P >= 0.75
    (_h10c)."""
    comps = {c.get("id"): c for c in (panel or {}).get("comparisons") or [] if isinstance(c, dict)}
    out = {}
    a = comps.get("H10a")
    if a is None:
        out["H10a"] = {"verdict": "not_evaluated", "why": "no H10a comparison in the panel"}
    else:
        det, perm = a.get("truth_detail_3v3") or {}, a.get("perm_5v5") or {}
        v, p = det.get("verdict"), perm.get("p")
        if v == "helps" and isinstance(p, (int, float)) and p <= H10A_ALPHA:
            verdict, why = "supported", "the pinned rule says helps and the 5 v 5 permutation p %.4g <= %g" % (
                p, H10A_ALPHA)
        elif v == "helps":
            verdict = "inconclusive"
            why = ("helps (decision rule, alpha about 0.20) but the 5 v 5 permutation p %s is not <= %g: not "
                   "confirmed" % ("%.4g" % p if isinstance(p, (int, float)) else "is missing", H10A_ALPHA))
        elif v in ("neutral", "hurts"):
            verdict, why = "falsified", "the pinned rule says %s (P %.3f)" % (v, det.get("p") or 0.0)
        else:
            verdict, why = "not_evaluated", "the H10a comparison has no truth verdict"
        out["H10a"] = {"verdict": verdict, "why": why, "rule": v, "p_perm": p, "p_rule": det.get("p")}
    b = comps.get("H10b")
    if b is None:
        out["H10b"] = {"verdict": "not_evaluated", "why": "no H10b comparison in the panel"}
    else:
        p = (b.get("truth_detail_3v3") or {}).get("p")
        out["H10b"] = {"verdict": "supported" if isinstance(p, (int, float)) and p >= H10_P_MIN else "not_supported",
                       "why": "P(U > U-ctl) = %s against %g" % (p, H10_P_MIN), "p": p}
    out["H10c"] = _h10c(panel, realloop_report)
    return out


def _h10c(panel, realloop_report):
    """H10c (contract 6): at least one M-size recovery increment is truth
    'helps' and, where a same-image control exists, beats it at P >= 0.75.
    Every recovery step of realloop_v2 counts (panel.RECOVERY_STEPS, which is
    realloop.RECOVERED_SEQUENCE). The steps with a pre-registered same-image
    control are those of panel.spec_v2()'s H10c comparisons, and such a step
    counts only through its control's comparison (a control missing from the
    panel leaves that step unjudged, never passed). Without realloop_v2's
    report the truth verdicts are unknown: not_evaluated, not 'not supported'."""
    from . import panel as PN
    if not isinstance(realloop_report, dict):
        return {"verdict": "not_evaluated", "why": "needs realloop_v2's report.json (the truth arm's verdicts)"}
    verdicts = {}
    for s in (BP.dev_only(realloop_report) or {}).get("steps") or []:
        v = (s.get("truth") or {}).get("verdict") if isinstance(s, dict) else None
        if isinstance(s, dict) and s.get("step") in PN.RECOVERY_STEPS and v in ("helps", "neutral", "hurts"):
            verdicts[s["step"]] = v
    if not verdicts:
        return {"verdict": "not_evaluated", "why": "the report has no truth verdict for a recovery step (%s)"
                % ", ".join(PN.RECOVERY_STEPS)}
    spec = PN.spec_v2()
    controlled = {}                                   # step -> comparison id
    for c in spec["comparisons"]:
        arm = spec["arms"].get(c["with"]) or {}
        if str(c["id"]).startswith("H10c") and arm.get("kind") == "truth_with":
            controlled[arm.get("step")] = c["id"]
    comps = {c.get("id"): c for c in (panel or {}).get("comparisons") or [] if isinstance(c, dict)}
    parts = []
    for step in PN.RECOVERY_STEPS:
        if step not in verdicts:
            continue
        cid = controlled.get(step)
        rec = {"step": step, "truth_verdict": verdicts[step], "comparison": cid, "p_vs_control": None}
        if cid is None:
            rec["passes"] = verdicts[step] == "helps"
        elif cid not in comps:
            rec["passes"] = False if verdicts[step] != "helps" else None
            rec["why"] = "its same-image control (%s) is not in the panel" % cid
        else:
            p = (comps[cid].get("truth_detail_3v3") or {}).get("p")
            rec["p_vs_control"] = p
            rec["passes"] = verdicts[step] == "helps" and isinstance(p, (int, float)) and p >= H10_P_MIN
        parts.append(rec)
    if any(x["passes"] is True for x in parts):
        return {"verdict": "supported", "parts": parts,
                "why": "a recovery step helps in the truth arm and, where a same-image control exists, beats it at "
                       "P >= %g" % H10_P_MIN}
    if any(x["passes"] is None for x in parts):
        return {"verdict": "not_evaluated", "parts": parts,
                "why": "a recovery step that helps waits for its same-image control in the panel"}
    return {"verdict": "not_supported", "parts": parts,
            "why": "no recovery step helps in the truth arm while beating its same-image control where one exists"}


def _da_actor(model):
    """The adversary's actor string, in the form policy._ACTOR_RE accepts (the
    campaign's ledger uses the same form)."""
    return BP.actor_for("%s/%s" % (DA_ROLE, model or "unknown"))


def _audit_usable(audit):
    """(ok, why): an audit a DA outcome may be scored on. An invalid audit
    (calibration overlap, contract 4.1 and 8.4) may not be cited by anything."""
    if not isinstance(audit, dict) or not audit:
        return False, "no audit_v1.json yet"
    if audit.get("valid") is not True:
        return False, "the audit is not valid (valid %r: calibration overlap); nothing may cite it" % (
            audit.get("valid"),)
    return True, ""


def h11(prospective_da, audit, recoverable_stages=None):
    """H11 (contract 6, review): the DA's top-1 forecast stage is the audit's
    stage with the largest recoverable lower bound, and its log-loss on that
    stage beats the uniform forecast over the ledger's recoverable stages.
    Scored either way; not_evaluated without both files or on an invalid audit.

    The audit's own H11 (audit_v1.json hypotheses.H11, written by
    funnel/estimate.py, the only producer of audit numbers) is taken as it is
    when it is decided. Otherwise it is computed here with the same rule, which
    needs `recoverable_stages` (the ledger's): the uniform forecast is over
    those, never over the stages the DA happened to name."""
    ok, why = _audit_usable(audit)
    if not ok:
        return {"verdict": "not_evaluated", "why": why}
    hyp = (audit.get("hypotheses") or {}).get("H11") if isinstance(audit.get("hypotheses"), dict) else None
    if isinstance(hyp, dict) and hyp.get("verdict") in ("supported", "falsified"):
        pt = hyp.get("parts") or {}
        return {"verdict": hyp["verdict"], "source": "audit_v1.json /hypotheses/H11 (funnel/estimate.py)",
                "top1": pt.get("forecast_top1"), "truth": pt.get("top_stage"), "p_truth": pt.get("p_top"),
                "log_loss": pt.get("log_loss"), "uniform_log_loss": pt.get("uniform_log_loss"),
                "stages": pt.get("stages"), "why": hyp.get("why")}
    fc = ((prospective_da or {}).get("stage_forecast") or
          (((prospective_da or {}).get("reply") or {}).get("stage_forecast")))
    if recoverable_stages is None:
        return {"verdict": "not_evaluated", "why": "the audit has no decided H11, and the ledger's recoverable "
                                                   "stages (the uniform forecast's support) were not given"}
    rec = [s for s in recoverable_stages]
    rank = [r for r in audit.get("stage_ranking") or [] if isinstance(r, dict) and r.get("stage") in rec]
    if not isinstance(fc, dict) or not fc or not rank or not rec:
        return {"verdict": "not_evaluated", "why": "needs the DA's stage_forecast, the audit's stage_ranking and "
                                                   "the recoverable stages"}
    if all(isinstance(r.get("recoverable_lb"), (int, float)) for r in rank):
        rank = sorted(rank, key=lambda r: (-float(r["recoverable_lb"]), str(r["stage"])))
    truth = rank[0]["stage"]
    top = sorted(fc, key=lambda s: (-float(fc[s]), s))[0]
    p = float(fc.get(truth, 0.0))
    ll = -math.log(p) if p > 0 else float("inf")
    uni = math.log(len(rec))
    ok = top == truth and ll < uni
    return {"verdict": "supported" if ok else "falsified", "source": "outcome.h11", "top1": top, "truth": truth,
            "p_truth": p, "log_loss": ll, "uniform_log_loss": uni, "stages": len(rec),
            "why": "top-1 %s %s the largest recoverable lower bound (%s); log-loss %.4g %s uniform %.4g over %d "
                   "recoverable stage(s)" % (top, "is" if top == truth else "is not", truth, ll,
                                             "<" if ll < uni else ">=", uni, len(rec))}


def _prediction_outcome(pred, audit, panel_h10):
    """(verdict, why) of one DA prediction against the audit (fn_rate, purity)
    or the panel (truth_verdict): correct / wrong / insufficient."""
    metric, direction, thr = pred.get("metric"), pred.get("direction"), pred.get("threshold")
    if metric == "fn_rate":
        st = ((audit or {}).get("stages") or {}).get(pred.get("stage")) or {}
        iv = (st.get("fn_rate") or {}).get("interval")
        if not (isinstance(iv, list) and len(iv) == 2) or not isinstance(thr, (int, float)):
            return "insufficient", "the audit has no fn_rate interval for stage %s (or no threshold)" % pred.get("stage")
        lo, hi = iv
        if direction == "above":
            return ("correct", "fn_rate lb %.3g >= %g" % (lo, thr)) if lo >= thr else \
                (("wrong", "fn_rate ub %.3g < %g" % (hi, thr)) if hi < thr else ("insufficient", "the interval "
                                                                                  "straddles %g" % thr))
        if direction == "below":
            return ("correct", "fn_rate ub %.3g < %g" % (hi, thr)) if hi < thr else \
                (("wrong", "fn_rate lb %.3g >= %g" % (lo, thr)) if lo >= thr else ("insufficient", "the interval "
                                                                                    "straddles %g" % thr))
        return "insufficient", "direction %r does not apply to fn_rate" % direction
    if metric == "purity":
        # a stratum's purity is its target share: the audit also holds, per stratum, the estimates the
        # recovery gates read ("label", "pred", "purity:<class>", "label:<class>"), never this one
        rows = [r for r in (audit or {}).get("strata") or [] if isinstance(r, dict)
                and r.get("stratum") == pred.get("stratum") and r.get("event", "target") == "target"]
        if not rows or not isinstance(thr, (int, float)):
            return "insufficient", "the audit has no stratum %r (or no threshold)" % pred.get("stratum")
        lo, hi = rows[0].get("interval") or [None, None]
        if lo is None:
            return "insufficient", "stratum %r has no interval" % pred.get("stratum")
        if direction == "above":
            return ("correct", "purity lb %.3g >= %g" % (lo, thr)) if lo >= thr else \
                (("wrong", "purity ub %.3g < %g" % (hi, thr)) if hi < thr else ("insufficient", "straddles"))
        if direction == "below":
            return ("correct", "purity ub %.3g < %g" % (hi, thr)) if hi < thr else \
                (("wrong", "purity lb %.3g >= %g" % (lo, thr)) if lo >= thr else ("insufficient", "straddles"))
        return "insufficient", "direction %r does not apply to purity" % direction
    if metric == "truth_verdict":
        v = ((panel_h10 or {}).get("H10a") or {}).get("rule")
        if v not in ("helps", "neutral", "hurts"):
            return "insufficient", "no realloop_v2 panel verdict yet"
        return ("correct" if v == direction else "wrong"), "the dose arm's truth verdict is %s" % v
    return "insufficient", "unknown metric %r" % (metric,)


def score_da(prospective_da, audit=None, panel=None, realloop_report=None, recoverable_stages=None):
    """The outcome record of one devil's-advocate pass: H11 and each kept
    prediction's verdict once its test landed (the audit for fn_rate and
    purity, the panel for truth_verdict). An invalid audit scores nothing:
    its predictions stay insufficient and H11 is not evaluated."""
    val = (prospective_da or {}).get("validation") or {}
    model = ((prospective_da or {}).get("model") or {}).get("resolved")
    ph10 = h10(panel, realloop_report) if panel else None
    usable, why_not = _audit_usable(audit)
    preds = []
    for ca in val.get("counter_arguments") or []:
        if not (ca.get("kept") and ca.get("counted")):
            continue
        pred = ca.get("prediction") or {}
        if not usable and pred.get("metric") in ("fn_rate", "purity"):
            verdict, why = "insufficient", why_not
        else:
            verdict, why = _prediction_outcome(pred, audit if usable else None, ph10)
        preds.append({"counter_argument": ca.get("index"), "prediction": ca.get("prediction"),
                      "verdict": verdict, "why": why})
    return {"kind": "da_outcome", "utc": M.utc_now(), "proposed_by": _da_actor(model), "role": DA_ROLE,
            "model": model, "digest_sha256": (prospective_da or {}).get("digest_sha256"),
            "h11": h11(prospective_da, audit, recoverable_stages), "predictions": preds,
            "scored": sum(1 for x in preds if x["verdict"] != "insufficient"),
            "correct": sum(1 for x in preds if x["verdict"] == "correct")}


def record_da(prospective_da, audit=None, panel=None, realloop_report=None, events_path=None, summary_path=None,
              recoverable_stages=None):
    """score_da + append to the track record (the adversary role's row)."""
    out = score_da(prospective_da, audit, panel, realloop_report, recoverable_stages)
    _append(events_path or TRACK_EVENTS, out)
    write_summary(events_path, summary_path)
    return out
