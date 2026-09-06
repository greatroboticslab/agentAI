#!/usr/bin/env python3
"""WP5 policy-gate tests (no cluster, no Mongo, no model).

`policy.authorize()` is the one gate a cluster action or a scheduler step is
supposed to pass through before it reaches SLURM or a corpus mutation. These
tests exist to keep two things honest as the codebase grows:

  * **Coverage cannot silently rot.** `test_every_cluster_action_has_a_policy_row`
    and `test_every_scheduler_step_has_a_policy_row` parse the *live* source of
    `dashboard_server.py` and `round_scheduler.py` with `ast` rather than
    importing them (dashboard_server.py touches `/ocean` at import time, which
    does not exist off the cluster) or hardcoding the action list. Add a new
    `_CLUSTER_ACTIONS` entry or a new `round_scheduler._WEED_STEPS` step with no
    corresponding row in `policy_actions.json`, and these tests fail — that is
    the point, not a bug in the test.
  * **The ceiling table is exactly the spec's, not almost.** `Ceilings: tier-0
    <= R1 propose, tier-1 <= R2 + propose R3, tier-2 proposes only, human all`
    is asserted cell by cell against `policy._decide`, so a future edit to the
    risk model has to change this test to pass, not just change behaviour that
    nothing was watching.

Run:  python -m pytest tests/test_policy.py -q   (or)  python3 tests/test_policy.py
"""
import ast
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import policy  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
_DASHBOARD_SRC = REPO / "weed_optimizer_framework" / "tools" / "dashboard_server.py"
_SCHEDULER_SRC = REPO / "weed_optimizer_framework" / "tools" / "round_scheduler.py"


# --- source-parsing helpers (no import: dashboard_server touches /ocean at
# import time, which is only present on the cluster) --------------------------
def _dict_literal_keys(src_text, var_name):
    """String keys of `var_name = {...}` at module scope, parsed with `ast`.

    Deliberately does not evaluate the dict's values: several real entries
    hold names (`_ROBOFLOW_KEY_FILE`) rather than literals, which would make
    `ast.literal_eval` on the whole dict raise. The keys themselves are always
    plain string literals, and the key set is all this module's coverage test
    needs.
    """
    tree = ast.parse(src_text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = []
        for k in node.value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
        return keys
    raise AssertionError("could not find %r as a dict literal" % var_name)


def real_cluster_action_ids():
    return _dict_literal_keys(_DASHBOARD_SRC.read_text(encoding="utf-8"), "_CLUSTER_ACTIONS")


def real_scheduler_step_names():
    return _dict_literal_keys(_SCHEDULER_SRC.read_text(encoding="utf-8"), "_WEED_STEPS")


# The one place this test file names the mapping from a round_scheduler step to
# its policy action id — the step names themselves are never hardcoded (they
# come from real_scheduler_step_names()), only their correspondence to a
# governance row is. A new step round_scheduler learns to submit (label/eval do
# not — see round_scheduler.py's own docstring: label is recorded
# "autolabel-in-collect" and eval is parsed off the train artifact, neither ever
# calls `_submit`) must be added here AND to policy_actions.json, or
# test_every_scheduler_step_has_a_policy_row fails.
_STEP_TO_ACTION = {"collect": "round_collect", "filter": "round_filter", "train": "round_train"}


# ============================================================================
# coverage: every real action / step has a table row
# ============================================================================
def test_every_cluster_action_has_a_policy_row():
    ids = real_cluster_action_ids()
    assert len(ids) >= 25, "sanity: expected dashboard_server.py to declare many actions"
    missing = [a for a in ids if not policy.describe(a)["known"]]
    assert not missing, (
        "these _CLUSTER_ACTIONS entries have no row in policy_actions.json: %s"
        % missing)


def test_every_cluster_action_row_is_well_formed():
    for action_id in real_cluster_action_ids():
        row = policy.describe(action_id)
        assert row["known"], action_id
        assert row["risk"] in policy.RISKS, "%s: bad risk %r" % (action_id, row["risk"])
        assert isinstance(row["reversible"], bool), action_id
        assert isinstance(row["allowed_tiers"], list) and row["allowed_tiers"], action_id
        assert row["description"].strip(), "%s: empty description" % action_id
        assert row["template"].strip(), "%s: empty template" % action_id


def test_every_scheduler_step_has_a_policy_row():
    steps = real_scheduler_step_names()
    assert set(steps) == {"collect", "filter", "train"}, (
        "round_scheduler._WEED_STEPS changed shape (%r); update _STEP_TO_ACTION "
        "in this test and policy_actions.json together" % steps)
    for step in steps:
        assert step in _STEP_TO_ACTION, "scheduler step %r has no policy mapping" % step
        action_id = _STEP_TO_ACTION[step]
        row = policy.describe(action_id)
        assert row["known"], "step %r maps to unknown action %r" % (step, action_id)
        assert "round-scheduler" in row["allowed_tiers"], (
            "policy action %r for scheduler step %r must allow the "
            "round-scheduler actor tier" % (action_id, step))


def test_policy_table_loads_with_no_errors():
    assert policy.errors() == [], policy.errors()


def test_real_cluster_action_count_matches_table_expectation():
    # Not load-bearing on its own (the coverage test above already fails if a
    # row goes missing) but pins the count actually observed on 2026-09-06 so a
    # silent *removal* of table rows this test wouldn't otherwise catch is
    # visible in a diff.
    ids = set(real_cluster_action_ids())
    assert len(ids) == 30, sorted(ids)


# ============================================================================
# bounds enforcement
# ============================================================================
def test_enum_bound_rejects_a_value_outside_the_set():
    r = policy.authorize("human:harry", "brain_harvest", {"time_h": 3})
    assert r["allowed"] is False
    assert any("time_h" in why for why in r["reasons"])


def test_enum_bound_accepts_a_value_inside_the_set():
    r = policy.authorize("human:harry", "brain_harvest", {"time_h": 4})
    assert r["allowed"] is True


def test_int_bound_rejects_below_minimum():
    r = policy.authorize("human:harry", "brain_harvest", {"max_new": 0})
    assert r["allowed"] is False
    assert any("max_new" in why for why in r["reasons"])


def test_int_bound_rejects_above_maximum():
    r = policy.authorize("human:harry", "brain_harvest", {"max_imgs": 50001})
    assert r["allowed"] is False
    assert any("max_imgs" in why for why in r["reasons"])


def test_float_bound_accepts_the_closed_interval_edges():
    lo = policy.authorize("round-scheduler", "round_train", {"min_dino_score": 0.0})
    hi = policy.authorize("round-scheduler", "round_train", {"min_dino_score": 1.0})
    assert lo["allowed"] is True and hi["allowed"] is True


def test_str_pattern_bound_rejects_a_bad_domain_name():
    r = policy.authorize("human:harry", "brain_harvest", {"domain": "not a domain!"})
    assert r["allowed"] is False
    assert any("domain" in why for why in r["reasons"])


def test_str_pattern_bound_accepts_a_good_domain_name():
    r = policy.authorize("human:harry", "brain_harvest", {"domain": "corn_v1"})
    assert r["allowed"] is True


def test_undeclared_param_key_is_refused():
    r = policy.authorize("human:harry", "refresh_registry", {"anything": 1})
    assert r["allowed"] is False
    assert any("anything" in why for why in r["reasons"])


def test_zero_params_is_always_a_valid_shape():
    r = policy.authorize("human:harry", "roboflow_state_audit")
    assert r["allowed"] is True


# ============================================================================
# tier ceilings — the spec's own sentence, cell by cell
# ============================================================================
def test_tier_ceiling_table_matches_the_spec():
    # "tier-0 <= R1 propose"
    assert policy._decide("tier0", "R0") == "direct"
    assert policy._decide("tier0", "R1") == "propose"
    assert policy._decide("tier0", "R2") == "refuse"
    assert policy._decide("tier0", "R3") == "refuse"
    assert policy._decide("tier0", "R4") == "refuse"
    # "tier-1 <= R2 + propose R3"
    assert policy._decide("tier1", "R0") == "direct"
    assert policy._decide("tier1", "R1") == "direct"
    assert policy._decide("tier1", "R2") == "direct"
    assert policy._decide("tier1", "R3") == "propose"
    assert policy._decide("tier1", "R4") == "refuse"
    # "tier-2 proposes only" (except the universally-open R0)
    assert policy._decide("tier2", "R0") == "direct"
    assert policy._decide("tier2", "R1") == "propose"
    assert policy._decide("tier2", "R2") == "propose"
    assert policy._decide("tier2", "R3") == "propose"
    assert policy._decide("tier2", "R4") == "refuse"
    # round-scheduler: same direct authority as tier-1 up to R2 (what the
    # scripted loop already exercises unattended), nothing past it
    assert policy._decide("round-scheduler", "R0") == "direct"
    assert policy._decide("round-scheduler", "R1") == "direct"
    assert policy._decide("round-scheduler", "R2") == "direct"
    assert policy._decide("round-scheduler", "R3") == "refuse"
    assert policy._decide("round-scheduler", "R4") == "refuse"
    # "human all"
    for risk in policy.RISKS:
        assert policy._decide("human", risk) == "direct", risk


def test_tier0_proposes_r1_never_applies_directly():
    r = policy.authorize("tier0:gemma4", "refresh_registry", {})
    assert r["risk"] == "R1"
    assert r["allowed"] is True
    assert r["needs_approval"] is True


def test_tier0_is_refused_at_r2():
    r = policy.authorize("tier0:gemma4", "brain_harvest", {})
    assert r["risk"] == "R2"
    assert r["allowed"] is False
    assert r["needs_approval"] is False


def test_tier1_applies_r2_directly():
    r = policy.authorize("tier1:deepseek-v4-flash", "brain_harvest", {})
    assert r["risk"] == "R2"
    assert r["allowed"] is True
    assert r["needs_approval"] is False


def test_tier1_proposes_r3():
    r = policy.authorize("tier1:deepseek-v4-flash", "sync_all_to_roboflow", {})
    assert r["risk"] == "R3"
    assert r["allowed"] is True
    assert r["needs_approval"] is True


def test_tier2_never_applies_directly_above_r0():
    for action_id in ("refresh_registry", "brain_harvest", "sync_all_to_roboflow"):
        r = policy.authorize("tier2:some-vlm", action_id, {})
        assert r["allowed"] is True, action_id
        assert r["needs_approval"] is True, action_id


def test_round_scheduler_applies_its_own_steps_directly():
    r = policy.authorize("round-scheduler", "round_collect",
                         {"collect_time_h": 10, "max_new": 3})
    assert r["allowed"] is True and r["needs_approval"] is False


def test_round_scheduler_is_refused_outside_its_three_steps():
    r = policy.authorize("round-scheduler", "brain_harvest", {})
    assert r["allowed"] is False


def test_human_applies_every_risk_directly():
    cases = [("roboflow_state_audit", {}), ("refresh_registry", {}),
            ("brain_harvest", {}), ("sync_all_to_roboflow", {}),
            ("audit_registry_garbage_APPLY", {})]
    for action_id, params in cases:
        r = policy.authorize("human:harry", action_id, params)
        assert r["allowed"] is True, action_id
        assert r["needs_approval"] is False, action_id


# ============================================================================
# SU estimation arithmetic
# ============================================================================
def test_su_gpu_hour_formula_h100():
    est = policy.estimate_su("round_train", {"train_time_h": 12})
    assert est["confident"] is True
    assert math.isclose(est["su"], 24.0)      # 1 GPU * 12 h * 2.0 SU/GPU-h


def test_su_gpu_hour_formula_v100_default_hours():
    est = policy.estimate_su("round_collect", {})   # no collect_time_h supplied
    assert est["confident"] is True
    assert math.isclose(est["su"], 10.0)      # hours_default 10 * 1 GPU * 1.0 SU/GPU-h


def test_su_gpu_hour_formula_fixed_hours_no_params_needed():
    est = policy.estimate_su("round_filter", {})
    assert est["confident"] is True
    assert math.isclose(est["su"], 8.0)


def test_su_fixed_su_zero_for_non_gpu_action():
    est = policy.estimate_su("restart_dashboard", {})
    assert est["confident"] is True
    assert est["su"] == 0.0


def test_su_unknown_for_needs_cluster_subprocess():
    est = policy.estimate_su("export_owl_exemplars", {})
    assert est["confident"] is False
    assert est["su"] is None


def test_su_unknown_for_unrecognised_action():
    est = policy.estimate_su("not_a_real_action", {})
    assert est["su"] is None
    assert est["confident"] is False


def test_su_scales_with_supplied_hours_param():
    cheap = policy.estimate_su("brain_harvest", {"time_h": 1})
    costly = policy.estimate_su("brain_harvest", {"time_h": 8})
    assert math.isclose(cheap["su"], 1.0)
    assert math.isclose(costly["su"], 8.0)


def test_budget_state_escalates_an_over_budget_direct_apply_to_approval():
    direct = policy.authorize("tier1:deepseek-v4-flash", "brain_harvest",
                              {"time_h": 8}, budget_state={"su_remaining": 100})
    assert direct["allowed"] is True and direct["needs_approval"] is False
    tight = policy.authorize("tier1:deepseek-v4-flash", "brain_harvest",
                             {"time_h": 8}, budget_state={"su_remaining": 1})
    assert tight["allowed"] is True
    assert tight["needs_approval"] is True


def test_mongo_down_resource_blocks_r1_and_above_but_not_r0():
    blocked = policy.authorize("human:harry", "refresh_registry", {},
                               resources={"mongo_down": True})
    assert blocked["allowed"] is False
    read = policy.authorize("human:harry", "roboflow_state_audit", {},
                            resources={"mongo_down": True})
    assert read["allowed"] is True


# ============================================================================
# determinism
# ============================================================================
def test_authorize_is_deterministic():
    args = ("tier1:deepseek-v4-flash", "brain_harvest", {"time_h": 4, "max_new": 3})
    a = policy.authorize(*args)
    b = policy.authorize(*args)
    assert a == b


def test_authorize_result_is_json_serialisable():
    r = policy.authorize("human:harry", "round_train", {"train_time_h": 12})
    assert json.loads(json.dumps(r)) == r


def test_estimate_su_is_deterministic():
    a = policy.estimate_su("round_train", {"train_time_h": 12})
    b = policy.estimate_su("round_train", {"train_time_h": 12})
    assert a == b


# ============================================================================
# small read helpers
# ============================================================================
def test_risk_of_known_and_unknown():
    assert policy.risk_of("round_train") == "R2"
    assert policy.risk_of("nonexistent_action") is None


def test_describe_unknown_action_says_so():
    d = policy.describe("nonexistent_action")
    assert d["known"] is False
    assert "reason" in d


def test_actions_for_tier_excludes_refused_risk_levels():
    tier0_menu = policy.actions_for_tier("tier0")
    assert "roboflow_state_audit" in tier0_menu    # R0
    assert "refresh_registry" in tier0_menu        # R1, propose
    assert "brain_harvest" not in tier0_menu       # R2, refused for tier0
    assert "audit_registry_garbage_APPLY" not in tier0_menu   # R4


def test_actions_for_tier_human_includes_r4():
    human_menu = policy.actions_for_tier("human")
    assert "audit_registry_garbage_APPLY" in human_menu
    assert "roboflow_delete_junk_apply" in human_menu


# ============================================================================
# CLI
# ============================================================================
def test_cli_authorize_exit_code_reflects_the_decision():
    assert policy.main(["authorize", "human:harry", "roboflow_state_audit"]) == 0
    assert policy.main(["authorize", "tier0:gemma4", "brain_harvest"]) == 1


def test_cli_table_and_explain_do_not_raise():
    assert policy.main(["table"]) == 0
    assert policy.main(["explain", "round_train"]) == 0


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
