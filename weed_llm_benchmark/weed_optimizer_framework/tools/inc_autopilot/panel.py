"""The realloop_v2 panel: cold baseline arms compared on dev only (docs/FUNNEL_AUDIT.md 9.2, H10;
runner 5.5.7 and 4.19).

    build_panel(dev_scores, spec) -> panel.json content (format funnel-panel/1)
    spec_v2() -> the pre-registered arms and comparisons of realloop_v2

What it compares. Each arm is a set of cold runs of existing experiments: the
base runs (run id base__s<seed>) of a build-baseline experiment, or the truth
arm's "with" runs of one step of a real loop (truth__<step tag>__union__s<seed>,
the fixed-base design of driver._truth_transitions). Every comparison is the
pinned truth rule, gate.truth_detail, on seeds 0-2 of each arm (the 3 v 3 rule
of the truth arm). H10a adds the pre-registered 5 v 5 test: the exact
one-sided permutation test of the difference of means over seeds 0-4
(funnel.estimate.perm_test_one_sided, the only producer of audit numbers);
H10a is supported only when both pass (contract 6 H10a). The verdicts are
outcome.h10's to read.

Test blindness. The panel reads dev scores only: every score must be stamped
exam "dev" (the scorer's own stamp), and a score of any other exam refuses the
whole panel. The dev scores come from remote.py funnel dev-scores, which ships
runs/<run>/scores/dev.json and nothing else; the lab never opens a run
directory. The record says exams_read ["dev"].

Pure: no file is read here; the caller passes the scores and writes the
record (atomically, with the sha256 of each score it read).
"""
from __future__ import annotations

import hashlib
import json
import re

from . import model as M

FORMAT = "funnel-panel/1"
BASE_RUN_RE = re.compile(r"^base__s(?P<seed>\d+)$")
TRUTH_RUN_RE = r"^truth__s\d+_%s__union__s(?P<seed>\d+)$"      # % re.escape(step name)
SEEDS_3 = (0, 1, 2)
SEEDS_5 = (0, 1, 2, 3, 4)
# realloop_v2's recovery steps, in order: inc/realloop.py RECOVERED_SEQUENCE
# (tests/test_funnel_ap_units.py checks the equality; realloop itself needs
# numpy, which the lab's scoring of H10 does not).
RECOVERY_STEPS = ("REC-VETO", "REC-AUTH-1", "REC-CLASS-1", "REC-JUDGE-1", "REC-AUTH-2", "REC-CLASS-2")


class PanelError(ValueError):
    """Scores that cannot make the panel: a missing seed, a non-dev score, a run
    named twice, an arm or comparison the spec does not define."""


def spec_v2():
    """The arms and comparisons of realloop_v2 (contract 9.2, runner 4.19 and
    6.1 F10c): B = base_b_v1's seeds 0-2 and rv2_B_extra's 3-4 (the same manifest
    bytes), U 5 seeds, the controls 3 seeds; REC-CLASS-1 and REC-JUDGE-1 are the
    truth arm's "with" runs of those steps of realloop_v2."""
    return {
        "arms": {
            "B": {"kind": "base", "exps": ["base_b_v1", "rv2_B_extra"], "seeds": list(SEEDS_5)},
            "U": {"kind": "base", "exps": ["rv2_U"], "seeds": list(SEEDS_5)},
            "U_ctl": {"kind": "base", "exps": ["rv2_Uctl"], "seeds": list(SEEDS_3)},
            "CLASS_ctl": {"kind": "base", "exps": ["rv2_CLASSctl"], "seeds": list(SEEDS_3)},
            "JUDGE_ctl": {"kind": "base", "exps": ["rv2_JUDGEctl"], "seeds": list(SEEDS_3)},
            "REC-CLASS-1": {"kind": "truth_with", "exps": ["realloop_v2"], "step": "REC-CLASS-1",
                            "seeds": list(SEEDS_3)},
            "REC-JUDGE-1": {"kind": "truth_with", "exps": ["realloop_v2"], "step": "REC-JUDGE-1",
                            "seeds": list(SEEDS_3)},
        },
        "comparisons": [
            {"id": "H10a", "with": "U", "without": "B", "perm_5v5": True},
            {"id": "H10b", "with": "U", "without": "U_ctl", "perm_5v5": False},
            {"id": "H10c-CLASS", "with": "REC-CLASS-1", "without": "CLASS_ctl", "perm_5v5": False},
            {"id": "H10c-JUDGE", "with": "REC-JUDGE-1", "without": "JUDGE_ctl", "perm_5v5": False},
        ],
    }


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def arm_scores(dev_scores, arm):
    """{seed: (exp, run id, score)} of one arm, from {exp: {run id: score}}.
    Refuses a score not stamped dev, and a seed two runs claim."""
    if arm.get("kind") == "base":
        rx = BASE_RUN_RE
    elif arm.get("kind") == "truth_with":
        rx = re.compile(TRUTH_RUN_RE % re.escape(str(arm.get("step"))))
    else:
        raise PanelError("arm kind %r is neither base nor truth_with" % (arm.get("kind"),))
    out = {}
    for exp in arm.get("exps") or []:
        for rid, score in sorted((dev_scores.get(exp) or {}).items()):
            m = rx.match(rid)
            if not m:
                continue
            if not isinstance(score, dict) or score.get("exam") != M.DECISION_EXAM:
                raise PanelError("%s/%s is not a dev score (exam %r): the panel reads dev only"
                                 % (exp, rid, (score or {}).get("exam") if isinstance(score, dict) else None))
            seed = int(m.group("seed"))
            if seed in out:
                raise PanelError("seed %d of the arm is claimed by %s/%s and %s/%s"
                                 % (seed, out[seed][0], out[seed][1], exp, rid))
            out[seed] = (exp, rid, score)
    return out


def _seeds(scores, seeds, name):
    missing = [s for s in seeds if s not in scores]
    if missing:
        raise PanelError("arm %s has no dev score for seed(s) %s" % (name, missing))
    return [scores[s][2] for s in seeds]


def _validated(G, cfg, arms):
    """The scores of [(arm name, seeds, {seed: (exp, run, score)})], checked by
    the pinned gate's own score rule (gate._validate: the schema, a production
    score, one exam, scorer, image order and exam manifest shared by every
    score, and distinct weights per run) before any of them enters a test.
    truth_detail checks the 3 v 3 part itself; the 5 v 5 permutation test reads
    seeds 3-4 as well, which nothing else checks. Raises PanelError."""
    named = [("%s seed %d (%s/%s)" % (name, s, sc[s][0], sc[s][1]), sc[s][2])
             for name, seeds, sc in arms for s in seeds]
    try:
        G._validate(named, cfg)
    except (TypeError, KeyError, ValueError) as e:
        raise PanelError("the 5 v 5 test's scores fail the gate's score rule: %s" % e)
    return [s for _n, s in named]


def build_panel(dev_scores, spec, built_utc=None):
    """panel.json content (runner 4.19) from {exp: {run id: dev score}}.

    Every comparison's 3 v 3 part is gate.truth_detail on seeds 0-2 of its two
    arms; a comparison with perm_5v5 adds funnel.estimate.perm_test_one_sided
    on the metric of seeds 0-4, whose ten scores pass the gate's score rule
    first (_validated). Raises PanelError when a score the spec needs is
    missing, not a dev score, or refused by the gate's score rule."""
    from ..inc import gate as G
    from ..funnel import estimate as FE
    cfg = G.GateConfig()
    arms, used = {}, {}
    for name, arm in sorted((spec.get("arms") or {}).items()):
        sc = arm_scores(dev_scores, arm)
        arms[name] = {"exps": list(arm.get("exps") or []), "seeds": list(arm.get("seeds") or []),
                      "kind": arm.get("kind"), "step": arm.get("step"),
                      "runs": {str(s): "%s/%s" % (sc[s][0], sc[s][1]) for s in sorted(sc)}}
        used[name] = sc
    comps = []
    for c in spec.get("comparisons") or []:
        w, wo = c.get("with"), c.get("without")
        if w not in used or wo not in used:
            raise PanelError("comparison %s names an arm the spec does not define (%s, %s)" % (c.get("id"), w, wo))
        try:
            det = G.truth_detail(_seeds(used[w], SEEDS_3, w), _seeds(used[wo], SEEDS_3, wo), cfg)
        except (TypeError, KeyError, ValueError) as e:
            raise PanelError("comparison %s: the gate refuses its 3 v 3 scores: %s" % (c.get("id"), e))
        rec = {"id": c.get("id"), "with": w, "without": wo,
               "truth_detail_3v3": {"verdict": det["verdict"], "p": det["p"], "metric": det["metric"],
                                    "with_values": det["with_values"], "without_values": det["without_values"],
                                    "with_mean": det["with_mean"], "with_sd": det["with_sd"],
                                    "without_mean": det["without_mean"], "without_sd": det["without_sd"],
                                    "species_passed": bool((det.get("species") or {}).get("passed")),
                                    "warnings": det.get("warnings") or []},
               "perm_5v5": None}
        if c.get("perm_5v5"):
            _seeds(used[w], SEEDS_5, w)
            _seeds(used[wo], SEEDS_5, wo)
            _validated(G, cfg, [(w, SEEDS_5, used[w]), (wo, SEEDS_5, used[wo])])
            a = [float(s[cfg.metric]) for s in _seeds(used[w], SEEDS_5, w)]
            b = [float(s[cfg.metric]) for s in _seeds(used[wo], SEEDS_5, wo)]
            pt = FE.perm_test_one_sided(a, b)
            rec["perm_5v5"] = {"p": pt["p"], "observed_diff": pt["observed_diff"], "n_splits": pt["n_splits"],
                               "with_values": a, "without_values": b}
        comps.append(rec)
    inputs = {}
    for exp in sorted({e for a in (spec.get("arms") or {}).values() for e in a.get("exps") or []}):
        inputs[exp] = {"dev_scores": {rid: _sha(s) for rid, s in sorted((dev_scores.get(exp) or {}).items())}}
    return {"format": FORMAT, "built_utc": built_utc or M.utc_now(), "inputs": inputs, "arms": arms,
            "comparisons": comps, "exams_read": [M.DECISION_EXAM], "spec_sha256": _sha(spec)}
