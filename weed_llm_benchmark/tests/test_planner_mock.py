#!/usr/bin/env python3
"""Unit tests for the campaign planner (tools/brain/planner.py). No cluster,
no Mongo, no real model: every input is an in-memory fixture, and the
`cluster:<model>` backend is exercised through an injected fake client.

Covered:
  * a mock plan is deterministic (same digest -> byte-identical plan) and
    always marked simulated
  * the lever blindfold: a digest built with omit_levers=True yields a plan
    that names no lever id, even when a caller mistakenly hands the sealed
    menu in anyway; the same digest with levers visible does use it
  * make_experiment refuses an entry with no control, and one with no
    success_criterion, each with a reason naming the defect
  * the file backend never overwrites an existing version
  * a cluster backend given an unreachable/failing client comes back as a
    refused plan, not an empty-looking success
  * validate() catches each malformed shape it is asked to check
"""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import db                        # noqa: E402
from weed_optimizer_framework.tools.brain import digest, planner     # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _config():
    return {
        "target_metric": "map50_95",
        "round_params": {"epochs": 60, "train_time_h": 12, "tier": "curated"},
        "thresholds": {"dino_threshold": 0.45},
        "budget": {"su_envelope": 1500.0, "daily_cap": 120, "per_round_cap": 60},
        "noise_floor": {"merged_curated": 0.005},
        "taxonomy": ["broadleaf", "grass"],
        "lever_menu": db.DEFAULT_DOMAIN_CONFIG["lever_menu"],
    }


def _round(n, metric, epochs=60):
    return {
        "round_num": n, "status": "done", "actor": "round-scheduler",
        "steps": {"train": {"status": "done", "actor": "round-scheduler",
                            "params": {"epochs": epochs, "train_time_h": 12,
                                      "tier": "curated"}}},
        "metrics": {"map50_95": metric},
    }


def _campaign():
    rounds = [_round(1, 0.80, epochs=60), _round(2, 0.81, epochs=24)]
    signals_history = [
        {"round": 2, "step": "train", "job_id": "44727703", "signal": "pool_growth",
         "severity": "warn", "value": 0.37, "reason": "iterations/epoch grew 37%"},
    ]
    ledger = [{"job": "44727703", "round": 2, "su": 24.6, "gpu_type": "h100"}]
    return rounds, signals_history, [], ledger, _config()


def _digest(omit_levers, num_ctx=8000):
    rounds, sig, corr, ledger, cfg = _campaign()
    return digest.build("weed", rounds, sig, corr, ledger, cfg, num_ctx,
                        omit_levers=omit_levers)


LEVER_MENU = db.DEFAULT_DOMAIN_CONFIG["lever_menu"]
LEVER_IDS = [str(l["id"]) for l in LEVER_MENU]


def _fake_client(error=None, text=None):
    def _call(prompt, model_id, num_ctx):
        out = {"tokens_in": 10, "tokens_out": 0, "latency_s": 0.01, "su": 0.0,
              "error": error or ""}
        out["text"] = "" if text is None else text
        return out
    return _call


# --------------------------------------------------------------------------
# mock: determinism + simulated
# --------------------------------------------------------------------------
class TestMockDeterminism(unittest.TestCase):

    def test_same_digest_yields_byte_identical_plan(self):
        d1 = _digest(omit_levers=False)
        d2 = _digest(omit_levers=False)          # rebuilt fixtures, not shared objects
        self.assertEqual(d1["sha256"], d2["sha256"])
        p1 = planner.plan(d1, "mock", lever_menu=LEVER_MENU)
        p2 = planner.plan(d2, "mock", lever_menu=LEVER_MENU)
        self.assertEqual(json.dumps(p1, sort_keys=True), json.dumps(p2, sort_keys=True))

    def test_mock_plan_is_marked_simulated(self):
        d = _digest(omit_levers=False)
        p = planner.plan(d, "mock", lever_menu=LEVER_MENU)
        self.assertTrue(p["simulated"])
        self.assertFalse(p["refused"])
        self.assertEqual(p["backend"], "mock")

    def test_mock_plan_validates(self):
        d = _digest(omit_levers=False)
        p = planner.plan(d, "mock", lever_menu=LEVER_MENU)
        ok, problems = planner.validate(p)
        self.assertTrue(ok, problems)


# --------------------------------------------------------------------------
# the lever blindfold
# --------------------------------------------------------------------------
class TestLeverBlindfold(unittest.TestCase):

    def test_levers_omitted_digest_leaks_no_lever_id(self):
        d = _digest(omit_levers=True)
        # Deliberately hand the sealed menu in anyway: the planner itself,
        # not the caller's discipline, is what must withhold it.
        p = planner.plan(d, "mock", lever_menu=LEVER_MENU)
        blob = json.dumps(p)
        leaked = [lid for lid in LEVER_IDS if lid in blob]
        self.assertEqual(leaked, [], "lever id(s) leaked into a plan built "
                        "from a levers-omitted digest: %s" % leaked)

    def test_levers_visible_digest_does_use_the_menu(self):
        """Positive control for the test above: the same lever_menu opt, on a
        digest that DOES show its levers section, produces at least one
        lever-named experiment -- proving the blindfold test isn't vacuous."""
        d = _digest(omit_levers=False)
        p = planner.plan(d, "mock", lever_menu=LEVER_MENU)
        blob = json.dumps(p)
        found = [lid for lid in LEVER_IDS if lid in blob]
        self.assertTrue(found, "no lever id appeared even with levers visible")

    def test_levers_omitted_digest_still_proposes_a_generic_experiment(self):
        d = _digest(omit_levers=True)
        p = planner.plan(d, "mock", lever_menu=LEVER_MENU)
        self.assertGreaterEqual(len(p["ordered_experiments"]), 1)
        self.assertEqual(p["ordered_experiments"][0]["recipe"], "repeat_current_recipe")


# --------------------------------------------------------------------------
# make_experiment: control / success_criterion are mandatory
# --------------------------------------------------------------------------
class TestExperimentConstruction(unittest.TestCase):

    def test_no_control_is_refused(self):
        res = planner.make_experiment(
            recipe="core_recipe_swap", params={}, seeds=[101, 102, 103],
            control="", est_su=72, risk="R2",
            success_criterion="3-seed mean beats the floor", stop_rule="")
        self.assertFalse(res["ok"])
        self.assertIn("control", res["reason"])

    def test_no_success_criterion_is_refused(self):
        res = planner.make_experiment(
            recipe="core_recipe_swap", params={}, seeds=[101, 102, 103],
            control="cwd12_core alone", est_su=72, risk="R2",
            success_criterion="", stop_rule="")
        self.assertFalse(res["ok"])
        self.assertIn("success_criterion", res["reason"])

    def test_well_formed_experiment_is_accepted(self):
        res = planner.make_experiment(
            recipe="core_recipe_swap", params={"lever": "core_recipe"},
            seeds=[101, 102, 103], control="cwd12_core alone", est_su=72,
            risk="R2", success_criterion="3-seed mean beats the floor",
            stop_rule="stop on any seed failure")
        self.assertTrue(res["ok"], res.get("reason"))
        self.assertEqual(res["experiment"]["control"], "cwd12_core alone")

    def test_bad_risk_is_refused(self):
        res = planner.make_experiment(
            recipe="x", control="c", success_criterion="s", risk="R9")
        self.assertFalse(res["ok"])


# --------------------------------------------------------------------------
# file backend: never overwrite a version
# --------------------------------------------------------------------------
class TestFileBackend(unittest.TestCase):

    def test_never_overwrites_a_version(self):
        d = _digest(omit_levers=False)
        with tempfile.TemporaryDirectory() as td:
            p1 = planner.plan(d, "file", domain="weed", root=td, lever_menu=LEVER_MENU)
            p2 = planner.plan(d, "file", domain="weed", root=td, lever_menu=LEVER_MENU)
            self.assertEqual(p1["version"], 1)
            self.assertEqual(p2["version"], 2)
            self.assertFalse(p1["refused"])
            self.assertFalse(p2["refused"])

            plans_dir = pathlib.Path(td) / "results" / "framework" / "_brain" / "weed" / "plans"
            v1_on_disk = json.loads((plans_dir / "v1.json").read_text(encoding="utf-8"))
            self.assertEqual(v1_on_disk["version"], 1,
                            "v1.json was overwritten by the second call")
            self.assertTrue((plans_dir / "v2.json").exists())

    def test_third_call_advances_to_version_three(self):
        d = _digest(omit_levers=False)
        with tempfile.TemporaryDirectory() as td:
            for expected in (1, 2, 3):
                p = planner.plan(d, "file", domain="weed", root=td, lever_menu=LEVER_MENU)
                self.assertEqual(p["version"], expected)


# --------------------------------------------------------------------------
# cluster backend
# --------------------------------------------------------------------------
class TestClusterBackend(unittest.TestCase):

    def test_unreachable_endpoint_yields_a_failed_plan_not_an_empty_one(self):
        d = _digest(omit_levers=False)
        client = _fake_client(error="connection refused")
        p = planner.plan(d, "cluster:some-model", client=client)
        self.assertTrue(p["refused"])
        self.assertTrue(p["reason"])
        self.assertIn("connection refused", p["reason"])
        self.assertEqual(p["ordered_experiments"], [])
        self.assertFalse(p["simulated"])
        self.assertEqual(p["backend"], "cluster:some-model")

    def test_reply_with_no_json_is_a_failed_plan(self):
        d = _digest(omit_levers=False)
        client = _fake_client(text="I refuse to answer in JSON today.")
        p = planner.plan(d, "cluster:some-model", client=client)
        self.assertTrue(p["refused"])
        self.assertTrue(p["reason"])

    def test_client_that_raises_yields_a_failed_plan(self):
        d = _digest(omit_levers=False)

        def _boom(prompt, model_id, num_ctx):
            raise ConnectionError("no route to host")
        p = planner.plan(d, "cluster:some-model", client=_boom)
        self.assertTrue(p["refused"])
        self.assertIn("no route to host", p["reason"])

    def test_well_formed_reply_yields_an_unrefused_non_simulated_plan(self):
        d = _digest(omit_levers=False)
        good_json = json.dumps({
            "hypotheses": ["the corpus may be the plateau's driver"],
            "ordered_experiments": [{
                "recipe": "cwd12_core_only", "params": {}, "seeds": [101, 102, 103],
                "control": "cwd12_core alone", "est_su": 72, "risk": "R2",
                "success_criterion": "3-seed mean beats the recipe's noise floor",
                "stop_rule": "stop on any seed failure",
            }],
            "stop_rules": ["stop if su_remaining runs out"],
        })
        client = _fake_client(text=good_json)
        p = planner.plan(d, "cluster:some-model", client=client)
        self.assertFalse(p["refused"])
        self.assertFalse(p["simulated"])
        self.assertEqual(len(p["ordered_experiments"]), 1)
        ok, problems = planner.validate(p)
        self.assertTrue(ok, problems)

    def test_reply_experiment_missing_control_is_dropped_not_smuggled(self):
        d = _digest(omit_levers=False)
        bad_json = json.dumps({
            "hypotheses": [], "stop_rules": [],
            "ordered_experiments": [{
                "recipe": "no_control_here", "params": {}, "seeds": [101],
                "control": "", "est_su": 1, "risk": "R2",
                "success_criterion": "something", "stop_rule": "",
            }],
        })
        client = _fake_client(text=bad_json)
        p = planner.plan(d, "cluster:some-model", client=client)
        self.assertFalse(p["refused"])           # the call itself succeeded
        self.assertEqual(p["ordered_experiments"], [])  # but the bad entry never lands
        self.assertTrue(p.get("problems"))


# --------------------------------------------------------------------------
# validate(): structural checks only
# --------------------------------------------------------------------------
class TestValidate(unittest.TestCase):

    def _good_plan(self):
        d = _digest(omit_levers=False)
        return planner.plan(d, "mock", lever_menu=LEVER_MENU)

    def test_not_a_dict(self):
        ok, problems = planner.validate(["not", "a", "plan"])
        self.assertFalse(ok)
        self.assertTrue(problems)

    def test_hypotheses_not_a_list(self):
        p = self._good_plan()
        p["hypotheses"] = "not a list"
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("hypotheses" in x for x in problems))

    def test_experiment_missing_control_caught(self):
        p = self._good_plan()
        p["ordered_experiments"][0]["control"] = ""
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("no control" in x for x in problems))

    def test_experiment_missing_success_criterion_caught(self):
        p = self._good_plan()
        p["ordered_experiments"][0]["success_criterion"] = ""
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("success_criterion" in x for x in problems))

    def test_experiment_bad_risk_caught(self):
        p = self._good_plan()
        p["ordered_experiments"][0]["risk"] = "R9"
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("risk" in x for x in problems))

    def test_experiment_params_not_a_dict_caught(self):
        p = self._good_plan()
        p["ordered_experiments"][0]["params"] = "not a dict"
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("params" in x for x in problems))

    def test_backend_empty_caught(self):
        p = self._good_plan()
        p["backend"] = ""
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("backend" in x for x in problems))

    def test_simulated_not_bool_caught(self):
        p = self._good_plan()
        p["simulated"] = "yes"
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("simulated" in x for x in problems))

    def test_stop_rules_not_a_list_caught(self):
        p = self._good_plan()
        p["stop_rules"] = {}
        ok, problems = planner.validate(p)
        self.assertFalse(ok)
        self.assertTrue(any("stop_rules" in x for x in problems))


if __name__ == "__main__":
    unittest.main()
