"""Tests for brain/parallelism.py -- the cap-and-refuse logic of the supervision layer.

These assert behaviour that is load-bearing for the research claim, not just for
correctness: a claim is never silently downsized, an unknown fact never widens a
cap, and no caller can obtain more than one collector.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weed_optimizer_framework.tools.brain import parallelism as P  # noqa: E402


def resources(**over):
    """Abundant, fully-known resources; individual tests remove or narrow one."""
    base = {"su_remaining": 1000.0, "queue_pending": 100, "free_gb": 400.0,
            "job_scoped_metric": True, "sealed_holdout": True,
            "per_shard_outputs": True}
    base.update(over)
    return base


def proposal(**over):
    base = {"role": "train", "purpose": "routine", "n_requested": 1,
            "est_su_per_unit": 24.0, "requested_by": "tier1:deepseek-v4-flash",
            "citation": {"signal": "plateau", "quote": "mAP50-95 0.5951 vs 0.6019"}}
    base.update(over)
    return base


class TestRoleCeilings(unittest.TestCase):

    def test_collector_is_always_one(self):
        """The collector writes the shared registry; two of them is a data-integrity bug."""
        d = P.plan("weed", proposal(role="collect", n_requested=8), resources())
        self.assertTrue(d["feasible"])
        self.assertEqual(d["n_parallel"], 1)
        self.assertEqual(d["binding"], "role_rule")
        self.assertIn("always", d["rationale"])

    def test_collector_cannot_be_raised_by_abundant_resources(self):
        d = P.plan("weed", proposal(role="collect", n_requested=8),
                   resources(su_remaining=10 ** 6, queue_pending=0, free_gb=10 ** 5))
        self.assertEqual(d["n_parallel"], 1)

    def test_unknown_role_is_refused(self):
        d = P.plan("weed", proposal(role="delete_everything", n_requested=1), resources())
        self.assertFalse(d["feasible"])
        self.assertEqual(d["n_parallel"], 0)
        self.assertTrue(any("no declared rule" in r for r in d["refusals"]))


class TestClaims(unittest.TestCase):

    def test_claim_runs_at_the_seed_count(self):
        d = P.plan("weed", proposal(purpose="claim", n_requested=1), resources())
        self.assertTrue(d["feasible"])
        self.assertEqual(d["n_parallel"], 3)
        self.assertEqual(len(d["slots"]), 3)

    def test_claim_is_refused_not_downsized_when_budget_is_short(self):
        """The important one: one seed of a three-seed claim is not a cheaper
        experiment, it is an unpublishable one, so the answer is a refusal."""
        d = P.plan("weed", proposal(purpose="claim", est_su_per_unit=50.0),
                   resources(su_remaining=100.0))
        self.assertFalse(d["feasible"])
        self.assertEqual(d["n_parallel"], 0)
        self.assertEqual(d["binding"], "su_envelope")
        self.assertTrue(any("needs 3 seeds" in r for r in d["refusals"]))

    def test_claim_slots_carry_the_seeds_given(self):
        d = P.plan("weed", proposal(purpose="claim", seeds=[101, 102, 103]), resources())
        self.assertEqual([s["seed"] for s in d["slots"]], [101, 102, 103])


class TestPreconditions(unittest.TestCase):

    def test_train_falls_to_one_without_job_scoped_metrics(self):
        """Before job-scoped artefacts existed, a second concurrent trainer's
        metric file was matched by the same glob and read as a second seed."""
        d = P.plan("weed", proposal(n_requested=3), resources(job_scoped_metric=False))
        self.assertEqual(d["n_parallel"], 1)
        self.assertEqual(d["binding"], "role_rule")
        self.assertTrue(any("job_scoped_metric" in c["reason"]
                            for c in d["caps"] if c["source"] == "role_rule"))

    def test_experiment_needs_a_sealed_holdout(self):
        d = P.plan("weed", proposal(role="experiment", purpose="claim"),
                   resources(sealed_holdout=False))
        self.assertFalse(d["feasible"])

    def test_filter_needs_per_shard_outputs(self):
        d = P.plan("weed", proposal(role="filter", n_requested=2),
                   resources(per_shard_outputs=False))
        self.assertEqual(d["n_parallel"], 1)

    def test_a_precondition_present_but_not_true_is_not_satisfied(self):
        """`"yes"` and `1` are not `True`; a truthy string must not satisfy a gate."""
        d = P.plan("weed", proposal(n_requested=3), resources(job_scoped_metric="yes"))
        self.assertEqual(d["n_parallel"], 1)


class TestUnknownsFailClosed(unittest.TestCase):

    def test_missing_queue_depth_collapses_to_the_floor_and_is_named(self):
        r = resources()
        del r["queue_pending"]
        d = P.plan("weed", proposal(n_requested=3), r)
        self.assertEqual(d["n_parallel"], 1)
        self.assertEqual(d["binding"], "queue_depth")
        self.assertTrue(any("queue_pending is unknown" in u for u in d["unknown"]))

    def test_missing_su_does_not_mean_unlimited(self):
        r = resources()
        del r["su_remaining"]
        d = P.plan("weed", proposal(n_requested=3), r)
        self.assertEqual(d["n_parallel"], 1)
        self.assertTrue(any("su_remaining is unknown" in u for u in d["unknown"]))

    def test_missing_disk_does_not_mean_unlimited(self):
        r = resources()
        del r["free_gb"]
        d = P.plan("weed", proposal(n_requested=3), r)
        self.assertEqual(d["n_parallel"], 1)

    def test_empty_resources_still_yields_one_unit_not_a_crash(self):
        d = P.plan("weed", proposal(n_requested=3), {})
        self.assertTrue(d["feasible"])
        self.assertEqual(d["n_parallel"], 1)
        self.assertGreaterEqual(len(d["unknown"]), 3)


class TestHostileInputs(unittest.TestCase):

    def test_numeric_strings_are_not_coerced(self):
        d = P.plan("weed", proposal(n_requested=3, est_su_per_unit="24"), resources())
        self.assertTrue(any("est_su_per_unit is unknown" in u for u in d["unknown"]))
        self.assertEqual(d["n_parallel"], 1)

    def test_nan_and_inf_requests_are_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            d = P.plan("weed", proposal(n_requested=bad), resources())
            self.assertTrue(any("not a number" in r for r in d["refusals"]),
                            "%r was accepted" % (bad,))

    def test_negative_request_is_refused(self):
        d = P.plan("weed", proposal(n_requested=-5), resources())
        self.assertTrue(any("not a number" in r for r in d["refusals"]))
        self.assertEqual(d["n_parallel"], 1)

    def test_absurd_request_is_capped_not_honoured(self):
        d = P.plan("weed", proposal(n_requested=10 ** 6), resources())
        self.assertLessEqual(d["n_parallel"], 3)

    def test_boolean_is_not_a_number(self):
        d = P.plan("weed", proposal(n_requested=True), resources())
        self.assertTrue(any("not a number" in r for r in d["refusals"]))

    def test_planner_never_raises(self):
        for bad in (None, [], "train", {"role": None}, {"role": "train", "purpose": []}):
            d = P.plan("weed", bad, resources())
            self.assertIn("feasible", d)


class TestCitation(unittest.TestCase):

    def test_more_than_one_unit_needs_a_citation(self):
        d = P.plan("weed", proposal(role="audit", n_requested=2, citation=None), resources())
        self.assertEqual(d["n_parallel"], 1)
        self.assertTrue(any("verbatim quote" in r for r in d["refusals"]))

    def test_a_citation_without_a_quote_is_not_a_citation(self):
        d = P.plan("weed", proposal(role="audit", n_requested=2,
                                    citation={"signal": "plateau"}), resources())
        self.assertEqual(d["n_parallel"], 1)

    def test_a_cited_request_is_granted_within_caps(self):
        d = P.plan("weed", proposal(role="audit", n_requested=2), resources())
        self.assertEqual(d["n_parallel"], 2)
        self.assertIsNotNone(d["citation"])

    def test_a_claim_still_needs_no_citation_to_reach_the_seed_count(self):
        """Seeds are required by the evidence rule, not requested by a model."""
        d = P.plan("weed", proposal(purpose="claim", n_requested=1, citation=None), resources())
        self.assertEqual(d["n_parallel"], 3)


class TestDeterminismAndLedger(unittest.TestCase):

    def test_same_inputs_same_decision(self):
        a = P.plan("weed", proposal(purpose="claim"), resources())
        b = P.plan("weed", proposal(purpose="claim"), resources())
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_ledger_row_records_the_refusal_too(self):
        d = P.plan("weed", proposal(purpose="claim", est_su_per_unit=50.0),
                   resources(su_remaining=100.0))
        row = P.ledger_row(d, actor="tier1:deepseek-v4-flash", ts=1788690000.0,
                           review_id="rv-1")
        self.assertEqual(row["granted"], 0)
        self.assertFalse(row["feasible"])
        self.assertEqual(row["binding"], "su_envelope")
        self.assertTrue(row["refusals"])
        self.assertEqual(row["ts"], 1788690000.0)

    def test_plan_reads_no_clock(self):
        """A decision must be replayable from its inputs alone."""
        src = open(os.path.join(os.path.dirname(P.__file__), "parallelism.py")).read()
        body = src[src.index("def _plan("):src.index("def _refuse(")]
        for banned in ("time.time", "datetime.now", "utcnow"):
            self.assertNotIn(banned, body)


class TestRuleTable(unittest.TestCase):

    def test_every_value_carries_a_reason(self):
        rules = P.load_rules()
        self.assertTrue(rules, "rule table did not load")
        for key, entry in rules.items():
            if key.startswith("_") or key == "roles":
                continue
            self.assertIn("reason", entry, key)
            self.assertGreater(len(entry["reason"]), 40, key)
        for role, entry in rules["roles"].items():
            self.assertGreater(len(entry.get("reason") or ""), 40, role)

    def test_a_missing_rule_table_disables_scheduling_rather_than_defaulting(self):
        old = os.environ.get("WEED_PARALLELISM_RULES")
        os.environ["WEED_PARALLELISM_RULES"] = "/nonexistent/parallelism_rules.json"
        try:
            P._RULES_CACHE.update({"path": None, "mtime": None, "data": None})
            d = P.plan("weed", proposal(), resources())
            self.assertFalse(d["feasible"])
        finally:
            if old is None:
                os.environ.pop("WEED_PARALLELISM_RULES", None)
            else:
                os.environ["WEED_PARALLELISM_RULES"] = old
            P._RULES_CACHE.update({"path": None, "mtime": None, "data": None})


if __name__ == "__main__":
    unittest.main(verbosity=2)
