#!/usr/bin/env python3
"""Unit tests for agent-proposed experiments (tools/brain/experiments.py).
No cluster, no Mongo: `propose()` is driven from `db.DEFAULT_DOMAIN_CONFIG`
directly, the same table `db.py` ships and `policy.py`'s own action table
prices, so nothing here can drift from what production actually declares.

Covered:
  * every proposal's lever_id is a real row of DEFAULT_DOMAIN_CONFIG's
    lever_menu -- an invented lever is not something this module can produce
  * the three weed proposals TIERED_SUPERVISION_PLAN names, with their exact
    controls and variants
  * verdict(): within_noise under the floor, better/worse above it,
    insufficient at n<3
  * to_approval()'s est_su is policy.estimate_su's own number, not a second
    computation
  * record_result() always bundles mean/std/n/noise_floor on one row
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db                              # noqa: E402
from weed_optimizer_framework.tools.brain import experiments, policy       # noqa: E402


def _config():
    cfg = {}
    cfg.update(db.DEFAULT_DOMAIN_CONFIG)
    cfg["lever_menu"] = db.DEFAULT_DOMAIN_CONFIG["lever_menu"]
    return cfg


LEVER_IDS = {str(l["id"]) for l in db.DEFAULT_DOMAIN_CONFIG["lever_menu"]}


class TestProposeOnlyFromTheMenu(unittest.TestCase):

    def test_every_proposal_traces_to_a_real_menu_row(self):
        proposals = experiments.propose("weed", _config())
        self.assertTrue(proposals)
        for p in proposals:
            self.assertIn(p["lever_id"], LEVER_IDS,
                          "propose() produced a lever id not on DEFAULT_DOMAIN_CONFIG's "
                          "lever_menu: %r" % (p["lever_id"],))

    def test_proposal_count_matches_menu_size(self):
        proposals = experiments.propose("weed", _config())
        self.assertEqual(len(proposals), len(db.DEFAULT_DOMAIN_CONFIG["lever_menu"]))

    def test_empty_menu_yields_no_proposals(self):
        cfg = _config()
        cfg["lever_menu"] = []
        self.assertEqual(experiments.propose("weed", cfg), [])

    def test_a_lever_absent_from_the_menu_is_never_proposed(self):
        cfg = _config()
        cfg["lever_menu"] = [l for l in db.DEFAULT_DOMAIN_CONFIG["lever_menu"]
                             if l["id"] != "model_family"]
        proposals = experiments.propose("weed", cfg)
        self.assertNotIn("model_family", [p["lever_id"] for p in proposals])


class TestTheThreeNamedWeedProposals(unittest.TestCase):
    """The exact worked example from TIERED_SUPERVISION_PLAN / the WP6 spec:
    (1) cwd12_core+audited vs merged_curated, control = cwd12_core alone;
    (2) per-box verification gate on vs off, control = gate off;
    (3) fresh_start vs continue, control = continue, same recipe."""

    def setUp(self):
        self.proposals = experiments.propose("weed", _config())
        self.by_id = {p["lever_id"]: p for p in self.proposals}

    def test_first_three_are_the_named_three_in_order(self):
        first_three = [p["lever_id"] for p in self.proposals[:3]]
        self.assertEqual(first_three,
                         ["core_recipe", "per_box_verification_gate", "fresh_start"])

    def test_core_recipe_control_and_variants(self):
        p = self.by_id["core_recipe"]
        self.assertEqual(p["control"], "cwd12_core alone")
        self.assertEqual(p["variants"], ["cwd12_core+audited", "merged_curated"])

    def test_per_box_verification_gate_control_and_variant(self):
        p = self.by_id["per_box_verification_gate"]
        self.assertEqual(p["control"], "gate off, same corpus")
        self.assertEqual(p["variants"], [True])

    def test_fresh_start_control_and_variant(self):
        p = self.by_id["fresh_start"]
        self.assertEqual(p["control"], "continue, same recipe")
        self.assertEqual(p["variants"], [True])

    def test_controls_match_the_menu_verbatim(self):
        menu_by_id = {str(l["id"]): l for l in db.DEFAULT_DOMAIN_CONFIG["lever_menu"]}
        for lever_id in ("core_recipe", "per_box_verification_gate", "fresh_start"):
            self.assertEqual(self.by_id[lever_id]["control"], menu_by_id[lever_id]["control"])


class TestEpochBudgetPairing(unittest.TestCase):
    """epoch_budget's options are two parallel lists keyed by parameter name;
    variants must pair them positionally, not cross the two axes."""

    def test_variants_are_paired_dicts(self):
        proposals = experiments.propose("weed", _config())
        p = next(x for x in proposals if x["lever_id"] == "epoch_budget")
        self.assertEqual(p["variants"],
                         [{"epochs": 40, "train_time_cap_h": 8.0},
                          {"epochs": 60, "train_time_cap_h": 10.8},
                          {"epochs": 90, "train_time_cap_h": 16.0}])


class TestToApproval(unittest.TestCase):

    def test_est_su_matches_policy_estimate_su_directly(self):
        proposal = experiments.propose("weed", _config())[0]
        approval = experiments.to_approval(proposal)
        direct = policy.estimate_su("round_train", {})
        self.assertEqual(approval["est_su"], direct["su"])
        self.assertEqual(approval["est_su_confident"], direct["confident"])
        self.assertEqual(approval["est_su_reason"], direct["reason"])

    def test_approval_is_filed_at_r3_regardless_of_lever_risk(self):
        proposal = experiments.propose("weed", _config())[0]
        self.assertEqual(proposal["risk"], "R2")           # the menu's own tag
        approval = experiments.to_approval(proposal)
        self.assertEqual(approval["risk"], "R3")            # the filed item's risk
        self.assertEqual(approval["lever_risk"], "R2")       # kept for reference
        self.assertTrue(approval["needs_approval"])

    def test_array_total_scales_the_per_run_estimate_by_seed_count(self):
        proposal = experiments.propose("weed", _config())[0]
        approval = experiments.to_approval(proposal)
        n_seeds = len(proposal["seeds"])
        self.assertEqual(approval["est_su_array_total"],
                         round(approval["est_su"] * n_seeds, 3))

    def test_malformed_proposal_never_raises(self):
        approval = experiments.to_approval(None)
        self.assertEqual(approval["kind"], "experiment_proposal")


class TestRecordResultBundlesEverything(unittest.TestCase):

    def test_result_carries_mean_std_n_and_noise_floor_together(self):
        proposal = experiments.propose("weed", _config())[0]
        result = experiments.record_result(proposal, "merged_curated",
                                           mean=0.01, std=0.002, n=3,
                                           noise_floor=0.005)
        for key in ("mean", "std", "n", "noise_floor"):
            self.assertIn(key, result)
        self.assertEqual(result["mean"], 0.01)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["noise_floor"], 0.005)

    def test_non_numeric_mean_is_recorded_as_none_not_guessed(self):
        proposal = experiments.propose("weed", _config())[0]
        result = experiments.record_result(proposal, "merged_curated",
                                           mean="not a number", std=None, n=3,
                                           noise_floor=0.005)
        self.assertIsNone(result["mean"])


class TestVerdict(unittest.TestCase):

    def _result(self, mean, n=3, std=0.001):
        return {"mean": mean, "std": std, "n": n}

    def test_within_noise_below_the_floor(self):
        self.assertEqual(experiments.verdict(self._result(0.003), 0.005), "within_noise")

    def test_within_noise_is_symmetric_for_a_negative_delta(self):
        self.assertEqual(experiments.verdict(self._result(-0.003), 0.005), "within_noise")

    def test_better_above_the_floor_positive(self):
        self.assertEqual(experiments.verdict(self._result(0.02), 0.005), "better")

    def test_worse_above_the_floor_negative(self):
        self.assertEqual(experiments.verdict(self._result(-0.02), 0.005), "worse")

    def test_exactly_at_the_floor_is_not_within_noise(self):
        # "less than the floor" is strict (digest.py's own _compare rule);
        # a delta exactly at the floor is a measured effect, not noise.
        self.assertEqual(experiments.verdict(self._result(0.005), 0.005), "better")

    def test_n_less_than_3_is_insufficient_never_a_winner(self):
        self.assertEqual(experiments.verdict(self._result(0.05, n=1), 0.005), "insufficient")
        self.assertEqual(experiments.verdict(self._result(0.05, n=2), 0.005), "insufficient")

    def test_n_less_than_3_is_insufficient_even_for_a_huge_delta(self):
        """A single-seed run cannot be a winner no matter how large its delta
        looks -- the sealed floor has no meaning at n=1."""
        self.assertEqual(experiments.verdict(self._result(10.0, n=1), 0.005), "insufficient")

    def test_missing_mean_is_insufficient(self):
        self.assertEqual(experiments.verdict({"mean": None, "n": 3}, 0.005), "insufficient")

    def test_missing_noise_floor_is_insufficient(self):
        self.assertEqual(experiments.verdict(self._result(0.02), None), "insufficient")

    def test_malformed_result_never_raises(self):
        self.assertEqual(experiments.verdict("not a dict", 0.005), "insufficient")
        self.assertEqual(experiments.verdict(None, 0.005), "insufficient")


if __name__ == "__main__":
    unittest.main()
