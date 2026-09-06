"""Tests for brain/escalation.py -- the tier-boundary rules of the supervision layer.

The load-bearing claims: every trigger family fires on its own real evidence and
does not fire without it; a missing threshold disables its trigger and names the
missing key rather than silently never firing; the highest destination wins while
every fired trigger stays in the record; an E3 trip holds tier-1 autonomy but never
the deterministic signals; a hostile context value never fires a trigger by accident
and never crashes `decide()`; and `decide()` reads no clock, so a context replayed
twice produces byte-identical decisions.
"""
import inspect
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weed_optimizer_framework.tools.brain import escalation as E  # noqa: E402


def ctx(**over):
    """A context with every trigger's input present and clean; each test
    narrows or breaks exactly the field(s) it means to exercise."""
    base = {
        "domain": "weed", "round": 5, "step": 3, "ts": 1234567890.0,
        "step_end": False,
        "signals": [],
        "review": {"confidence": 0.9, "findings": []},
        "corrections_history": [],
        "new_corrections": [],
        "plateau": {"rounds": 1, "recipe_changed": False},
        "actions": [],
        "stop_loss": False,
        "citation_validity": 100,
        "domain_config": {"brain": {"periodic_audit_every": 4}},
    }
    base.update(over)
    return base


def _rules_without(*keys):
    """A temp copy of the real rules file with `keys` removed. Caller restores
    WEED_ESCALATION_RULES and deletes the temp file when done."""
    with open(E.rules_path(), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for k in keys:
        data.pop(k, None)
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


class _RulesOverride(object):
    """Point WEED_ESCALATION_RULES at a rules file missing `keys`, for the
    duration of a `with` block, then restore the environment exactly."""

    def __init__(self, *keys):
        self.keys = keys
        self.path = None
        self.old = None

    def __enter__(self):
        self.path = _rules_without(*self.keys)
        self.old = os.environ.get("WEED_ESCALATION_RULES")
        os.environ["WEED_ESCALATION_RULES"] = self.path
        return self

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("WEED_ESCALATION_RULES", None)
        else:
            os.environ["WEED_ESCALATION_RULES"] = self.old
        try:
            os.remove(self.path)
        except OSError:
            pass


def _rules_of(result, rule):
    return [t for t in result["triggers"] if t["rule"] == rule]


class TestE1SignalWarn(unittest.TestCase):

    def test_fires_on_warn_signal(self):
        r = E.decide(ctx(signals=[{"signal": "walltime_bound", "severity": "warn",
                                   "value": 1, "reason": "hit the wall"}]))
        hits = _rules_of(r, "E1.signal_warn")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["status"], "fired")
        self.assertEqual(hits[0]["destination"], "tier1")
        self.assertEqual(r["escalate_to"], "tier1")

    def test_fires_on_crit_signal(self):
        r = E.decide(ctx(signals=[{"signal": "ownership_violation", "severity": "crit",
                                   "value": 1, "reason": "mirror diverged"}]))
        # ownership_violation also fires its own E3 rule, so escalate_to is
        # human here -- what matters for this test is that E1 still fired too.
        self.assertTrue(_rules_of(r, "E1.signal_warn"))

    def test_does_not_fire_below_warn(self):
        r = E.decide(ctx(signals=[{"signal": "source_degraded", "severity": "info",
                                   "value": 0, "reason": "SOCKS proxy skip, designed"}]))
        self.assertFalse(_rules_of(r, "E1.signal_warn"))
        self.assertEqual(r["escalate_to"], "none")

    def test_missing_signals_is_unknown_not_clean(self):
        c = ctx()
        del c["signals"]
        r = E.decide(c)
        hits = _rules_of(r, "E1.signal_warn")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "signals")
        self.assertEqual(r["escalate_to"], "none")


class TestE1StepEnd(unittest.TestCase):

    def test_fires_on_step_end(self):
        r = E.decide(ctx(step_end=True))
        hits = _rules_of(r, "E1.step_end")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier1")
        self.assertEqual(r["escalate_to"], "tier1")

    def test_does_not_fire_mid_step(self):
        r = E.decide(ctx(step_end=False))
        self.assertFalse(_rules_of(r, "E1.step_end"))
        self.assertEqual(r["escalate_to"], "none")

    def test_missing_step_end_is_unknown(self):
        c = ctx()
        del c["step_end"]
        r = E.decide(c)
        hits = _rules_of(r, "E1.step_end")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "step_end")


class TestE1PeriodicAudit(unittest.TestCase):

    def test_fires_on_the_nth_step(self):
        r = E.decide(ctx(step=4))
        hits = _rules_of(r, "E1.periodic_audit")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier1")
        self.assertEqual(r["escalate_to"], "tier1")

    def test_does_not_fire_off_cadence(self):
        r = E.decide(ctx(step=5))
        self.assertFalse(_rules_of(r, "E1.periodic_audit"))
        self.assertEqual(r["escalate_to"], "none")

    def test_domain_config_overrides_the_file_default(self):
        # File default is 4; a domain declaring 5 must audit on 5, not 4.
        r_off_file_cadence = E.decide(ctx(step=4, domain_config={"brain": {"periodic_audit_every": 5}}))
        self.assertFalse(_rules_of(r_off_file_cadence, "E1.periodic_audit"))
        r_on_domain_cadence = E.decide(ctx(step=5, domain_config={"brain": {"periodic_audit_every": 5}}))
        self.assertTrue(_rules_of(r_on_domain_cadence, "E1.periodic_audit"))

    def test_missing_threshold_disables_trigger_and_names_it(self):
        with _RulesOverride("periodic_audit_every_steps"):
            r = E.decide(ctx(step=4, domain_config={}))
        hits = _rules_of(r, "E1.periodic_audit")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "periodic_audit_every_steps")
        self.assertNotEqual(r["escalate_to"], "tier1")


class TestE2RecurringSignal(unittest.TestCase):

    def _hist(self, n, signal="plateau"):
        return [{"seq": i + 1, "author": "tier1:x", "kind": "config",
                 "target": {"key": "epochs"}, "signal": signal, "reason": "r", "ts": 1.0}
                for i in range(n)]

    def test_one_prior_correction_does_not_reach_tier2(self):
        r = E.decide(ctx(signals=[{"signal": "plateau", "severity": "warn",
                                   "value": 0.01, "reason": "still flat"}],
                         corrections_history=self._hist(1)))
        self.assertFalse(_rules_of(r, "E2.recurring_signal"))
        self.assertNotEqual(r["escalate_to"], "tier2")

    def test_two_prior_corrections_reaches_tier2(self):
        r = E.decide(ctx(signals=[{"signal": "plateau", "severity": "warn",
                                   "value": 0.01, "reason": "still flat"}],
                         corrections_history=self._hist(2)))
        hits = _rules_of(r, "E2.recurring_signal")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier2")
        self.assertEqual(r["escalate_to"], "tier2")

    def test_missing_corrections_history_is_unknown(self):
        c = ctx(signals=[{"signal": "plateau", "severity": "warn", "value": 0.01, "reason": "r"}])
        del c["corrections_history"]
        r = E.decide(c)
        hits = _rules_of(r, "E2.recurring_signal")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "corrections_history")

    def test_missing_threshold_disables_trigger_and_names_it(self):
        with _RulesOverride("recurring_after_corrections"):
            r = E.decide(ctx(signals=[{"signal": "plateau", "severity": "warn",
                                       "value": 0.01, "reason": "r"}],
                             corrections_history=self._hist(5)))
        hits = _rules_of(r, "E2.recurring_signal")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "recurring_after_corrections")


class TestE2Plateau(unittest.TestCase):

    def test_fires_with_enough_rounds_and_a_recipe_change(self):
        r = E.decide(ctx(plateau={"rounds": 3, "recipe_changed": True}))
        hits = _rules_of(r, "E2.plateau")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier2")
        self.assertEqual(r["escalate_to"], "tier2")

    def test_does_not_fire_without_a_recipe_change(self):
        r = E.decide(ctx(plateau={"rounds": 6, "recipe_changed": False}))
        self.assertFalse(_rules_of(r, "E2.plateau"))

    def test_does_not_fire_below_the_round_floor(self):
        r = E.decide(ctx(plateau={"rounds": 2, "recipe_changed": True}))
        self.assertFalse(_rules_of(r, "E2.plateau"))

    def test_missing_threshold_disables_trigger_and_names_it(self):
        with _RulesOverride("plateau_min_rounds"):
            r = E.decide(ctx(plateau={"rounds": 10, "recipe_changed": True}))
        hits = _rules_of(r, "E2.plateau")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "plateau_min_rounds")


class TestE2CorrectionClass(unittest.TestCase):

    def test_fires_on_code_class(self):
        r = E.decide(ctx(new_corrections=[{"seq": 9, "kind": "code",
                                           "target": {"key": "trainer.py"}}]))
        hits = _rules_of(r, "E2.correction_class")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier2")
        # Both rules fire on one correction and both are reported: tier-2 is
        # where a code or design change is ASKED for, a person is who MAKES it.
        # The higher destination wins, which is also what the benchmark's frozen
        # scoring rubric records for every code, design and plan correction in
        # the campaign record. A module that answered tier2 here would disagree
        # with the rubric about the same case.
        self.assertTrue(_rules_of(r, "E3.correction_needs_a_person"))
        self.assertEqual(r["escalate_to"], "human")

    def test_fires_on_design_class(self):
        r = E.decide(ctx(new_corrections=[{"seq": 9, "kind": "design",
                                           "target": {"key": "recipe"}}]))
        self.assertTrue(_rules_of(r, "E2.correction_class"))

    def test_does_not_fire_on_config_class(self):
        r = E.decide(ctx(new_corrections=[{"seq": 9, "kind": "config",
                                           "target": {"key": "epochs"}}]))
        self.assertFalse(_rules_of(r, "E2.correction_class"))

    def test_missing_new_corrections_is_unknown(self):
        c = ctx()
        del c["new_corrections"]
        r = E.decide(c)
        hits = _rules_of(r, "E2.correction_class")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "new_corrections")


class TestE2LowConfidence(unittest.TestCase):

    def test_fires_below_floor_on_a_warn_finding(self):
        r = E.decide(ctx(review={"confidence": 0.2,
                                 "findings": [{"signal": "gate_noop", "severity": "warn",
                                              "diagnosis": "kept == raw"}]}))
        hits = _rules_of(r, "E2.low_confidence")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier2")
        self.assertEqual(r["escalate_to"], "tier2")

    def test_does_not_fire_when_confidence_is_high(self):
        r = E.decide(ctx(review={"confidence": 0.9,
                                 "findings": [{"signal": "gate_noop", "severity": "warn"}]}))
        self.assertFalse(_rules_of(r, "E2.low_confidence"))

    def test_does_not_fire_without_a_warn_finding(self):
        r = E.decide(ctx(review={"confidence": 0.1,
                                 "findings": [{"signal": "gate_noop", "severity": "info"}]}))
        self.assertFalse(_rules_of(r, "E2.low_confidence"))

    def test_missing_threshold_disables_trigger_and_names_it(self):
        with _RulesOverride("confidence_floor"):
            r = E.decide(ctx(review={"confidence": 0.0,
                                     "findings": [{"signal": "x", "severity": "crit"}]}))
        hits = _rules_of(r, "E2.low_confidence")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "confidence_floor")


class TestE2BudgetBreach(unittest.TestCase):

    def test_fires_on_crit_budget_signal(self):
        r = E.decide(ctx(signals=[{"signal": "budget", "severity": "crit",
                                   "value": 1.4, "reason": "campaign over envelope"}]))
        hits = _rules_of(r, "E2.budget_breach")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "tier2")
        self.assertEqual(r["escalate_to"], "tier2")

    def test_does_not_fire_on_info_budget_signal(self):
        r = E.decide(ctx(signals=[{"signal": "budget", "severity": "info",
                                   "value": 0.1, "reason": "fine"}]))
        self.assertFalse(_rules_of(r, "E2.budget_breach"))

    def test_absent_budget_signal_is_clean_not_unknown(self):
        # signals.detect() omits an 'ok' finding entirely; this rule must read
        # that absence the same way, not as "could not check".
        r = E.decide(ctx(signals=[]))
        self.assertFalse(_rules_of(r, "E2.budget_breach"))


class TestE3R3R4Action(unittest.TestCase):

    def test_fires_on_r4_action_by_direct_risk(self):
        r = E.decide(ctx(actions=[{"risk": "R4"}]))
        hits = _rules_of(r, "E3.r3_r4_action")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["destination"], "human")
        self.assertEqual(r["escalate_to"], "human")

    def test_fires_on_r3_action_resolved_through_policy(self):
        r = E.decide(ctx(actions=[{"action": "restart_dashboard"}]))
        hits = _rules_of(r, "E3.r3_r4_action")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["evidence"]["risk"], "R3")

    def test_does_not_fire_on_r2_action(self):
        r = E.decide(ctx(actions=[{"action": "round_train"}]))
        self.assertFalse(_rules_of(r, "E3.r3_r4_action"))

    def test_missing_actions_is_unknown(self):
        c = ctx()
        del c["actions"]
        r = E.decide(c)
        hits = _rules_of(r, "E3.r3_r4_action")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "actions")


class TestE3StopLoss(unittest.TestCase):

    def test_fires_on_bare_true(self):
        r = E.decide(ctx(stop_loss=True))
        hits = _rules_of(r, "E3.stop_loss")
        self.assertEqual(len(hits), 1)
        self.assertEqual(r["escalate_to"], "human")

    def test_fires_on_active_dict(self):
        r = E.decide(ctx(stop_loss={"active": True, "reason": "walltime projection exceeded"}))
        self.assertTrue(_rules_of(r, "E3.stop_loss"))

    def test_does_not_fire_when_false(self):
        r = E.decide(ctx(stop_loss=False))
        self.assertFalse(_rules_of(r, "E3.stop_loss"))

    def test_missing_stop_loss_is_unknown(self):
        c = ctx()
        del c["stop_loss"]
        r = E.decide(c)
        hits = _rules_of(r, "E3.stop_loss")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "stop_loss")


class TestE3OwnershipViolation(unittest.TestCase):

    def test_fires_on_crit_signal(self):
        r = E.decide(ctx(signals=[{"signal": "ownership_violation", "severity": "crit",
                                   "value": 1, "reason": "mirror hash diverged from ledger"}]))
        hits = _rules_of(r, "E3.ownership_violation")
        self.assertEqual(len(hits), 1)
        self.assertEqual(r["escalate_to"], "human")

    def test_absent_signal_is_clean_not_unknown(self):
        r = E.decide(ctx(signals=[]))
        self.assertFalse(_rules_of(r, "E3.ownership_violation"))


class TestE3CitationValidity(unittest.TestCase):

    def test_fires_below_floor_as_a_bare_percent(self):
        r = E.decide(ctx(citation_validity=50))
        hits = _rules_of(r, "E3.citation_validity")
        self.assertEqual(len(hits), 1)
        self.assertEqual(r["escalate_to"], "human")

    def test_fires_below_floor_as_a_valid_over_total_ratio(self):
        r = E.decide(ctx(citation_validity={"valid": 60, "total": 100}))
        self.assertTrue(_rules_of(r, "E3.citation_validity"))

    def test_does_not_fire_above_the_floor(self):
        r = E.decide(ctx(citation_validity=85))
        self.assertFalse(_rules_of(r, "E3.citation_validity"))

    def test_missing_threshold_disables_trigger_and_names_it(self):
        with _RulesOverride("citation_validity_floor_pct"):
            r = E.decide(ctx(citation_validity=10))
        hits = _rules_of(r, "E3.citation_validity")
        self.assertEqual(hits[0]["status"], "unknown")
        self.assertEqual(hits[0]["missing"], "citation_validity_floor_pct")


class TestHighestDestinationWins(unittest.TestCase):

    def test_all_three_families_reported_but_human_wins(self):
        r = E.decide(ctx(
            signals=[{"signal": "walltime_bound", "severity": "warn",
                     "value": 1, "reason": "projected over"}],
            plateau={"rounds": 4, "recipe_changed": True},
            stop_loss=True,
        ))
        self.assertEqual(r["escalate_to"], "human")
        rules_fired = {t["rule"] for t in r["triggers"] if t["status"] == "fired"}
        self.assertIn("E1.signal_warn", rules_fired)
        self.assertIn("E2.plateau", rules_fired)
        self.assertIn("E3.stop_loss", rules_fired)


class TestE3Hold(unittest.TestCase):

    def test_e3_pauses_tier1_autonomy_but_not_signals(self):
        r = E.decide(ctx(stop_loss=True))
        self.assertEqual(r["held"]["tier1_autonomy"], True)
        self.assertEqual(r["held"]["signals"], False)
        self.assertTrue(r["held"]["reason"])

    def test_no_hold_without_an_e3_trigger(self):
        r = E.decide(ctx(step_end=True))
        self.assertEqual(r["held"], {"tier1_autonomy": False, "signals": False, "reason": None})


class TestHostileInputs(unittest.TestCase):
    """Garbage in must never fire a trigger by accident and must never crash
    decide(); it must also never be reported as if the check ran clean."""

    def test_numeric_looking_string_step_is_unknown_not_fired(self):
        r = E.decide(ctx(step="4"))
        hits = _rules_of(r, "E1.periodic_audit")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_numeric_looking_string_citation_validity_is_unknown(self):
        r = E.decide(ctx(citation_validity="10"))
        hits = _rules_of(r, "E3.citation_validity")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_nan_confidence_is_unknown_not_fired(self):
        r = E.decide(ctx(review={"confidence": float("nan"),
                                 "findings": [{"signal": "x", "severity": "crit"}]}))
        hits = _rules_of(r, "E2.low_confidence")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_infinite_plateau_rounds_is_unknown_not_fired(self):
        r = E.decide(ctx(plateau={"rounds": float("inf"), "recipe_changed": True}))
        hits = _rules_of(r, "E2.plateau")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_none_stop_loss_value_is_unknown_not_fired(self):
        r = E.decide(ctx(stop_loss=None))
        hits = _rules_of(r, "E3.stop_loss")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_list_where_dict_belongs_is_unknown_not_crash(self):
        r = E.decide(ctx(review=[1, 2, 3]))
        hits = _rules_of(r, "E2.low_confidence")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_dict_where_list_belongs_is_unknown_not_crash(self):
        r = E.decide(ctx(signals={"signal": "budget"}))
        hits = _rules_of(r, "E1.signal_warn")
        self.assertEqual(hits[0]["status"], "unknown")

    def test_malformed_top_level_context_never_raises(self):
        for garbage in (None, "not a dict", [1, 2, 3], 42):
            r = E.decide(garbage)
            self.assertEqual(r["escalate_to"], "human")
            self.assertIn("triggers", r)
            self.assertIn("held", r)
            self.assertIn("ledger_row", r)

    def test_bool_is_not_accepted_as_a_number(self):
        # True/False are ints in Python; treating them as a valid `step` would
        # let a schema bug produce a periodic-audit firing with no real number.
        r = E.decide(ctx(step=True))
        hits = _rules_of(r, "E1.periodic_audit")
        self.assertEqual(hits[0]["status"], "unknown")


class TestDeterminism(unittest.TestCase):

    def test_same_context_yields_identical_decisions(self):
        c = ctx(signals=[{"signal": "plateau", "severity": "warn", "value": 0.01, "reason": "r"}],
               corrections_history=[{"seq": 1, "signal": "plateau", "kind": "config"},
                                    {"seq": 2, "signal": "plateau", "kind": "config"}],
               plateau={"rounds": 3, "recipe_changed": True})
        first = E.decide(c)
        second = E.decide(ctx(**c))
        self.assertEqual(first, second)


class TestNoClockReads(unittest.TestCase):

    def test_decide_source_never_calls_the_clock(self):
        src = inspect.getsource(E.decide) + inspect.getsource(E._decide)
        for check in E._CHECKS:
            src += inspect.getsource(check)
        banned = ("time.time(", "time.monotonic(", "time.perf_counter(",
                  "datetime.now(", "datetime.utcnow(", "datetime.today(",
                  "time.localtime(", "time.gmtime(")
        for token in banned:
            self.assertNotIn(token, src, "clock read found: %s" % token)

    def test_module_does_not_even_import_a_clock(self):
        self.assertFalse(hasattr(E, "time"), "escalation.py must not import the time module")
        self.assertFalse(hasattr(E, "datetime"), "escalation.py must not import datetime")


class TestExplainAndRules(unittest.TestCase):

    def test_explain_known_rule(self):
        out = E.explain("E3.stop_loss")
        self.assertTrue(out["known"])
        self.assertEqual(out["destination"], "human")

    def test_explain_unknown_rule_names_the_valid_ones(self):
        out = E.explain("not_a_rule")
        self.assertFalse(out["known"])
        self.assertIn("E1.signal_warn", out["reason"])

    def test_rules_reports_every_declared_threshold_with_a_reason(self):
        out = E.rules()
        for key in ("confidence_floor", "citation_validity_floor_pct",
                   "recurring_after_corrections", "plateau_min_rounds",
                   "periodic_audit_every_steps"):
            self.assertIn(key, out["values"])
            self.assertTrue(out["reasons"][key])
        self.assertEqual(out["errors"], [])


class TestRubricAgreement(unittest.TestCase):
    """The live escalator and the frozen scoring rubric must not disagree.

    The rubric is pre-registration: it is how every case's expected escalation
    destination was labelled before any model ran. If this module routes a
    correction class somewhere else, one of the two is wrong about the same
    fact, and the benchmark would score the layer against a destination the
    layer does not produce.
    """

    def test_correction_classes_route_where_the_rubric_says(self):
        from weed_optimizer_framework.tools.brain import inventory_adapter as IA
        for kind in ("code", "design"):
            expected = IA.ESCALATION_RULES[kind][0]
            r = E.decide(ctx(new_corrections=[{"seq": 1, "kind": kind,
                                               "target": {"key": "x"}}]))
            self.assertEqual(r["escalate_to"], expected, kind)

    def test_a_config_correction_stays_below_the_deep_tiers(self):
        from weed_optimizer_framework.tools.brain import inventory_adapter as IA
        self.assertEqual(IA.ESCALATION_RULES["config"][0], "tier1")
        r = E.decide(ctx(new_corrections=[{"seq": 1, "kind": "config",
                                           "target": {"key": "epochs"}}]))
        self.assertIn(r["escalate_to"], ("none", "tier1"))


if __name__ == "__main__":
    unittest.main()
