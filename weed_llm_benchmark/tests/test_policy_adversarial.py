#!/usr/bin/env python3
"""WP5 policy-gate adversarial tests — the actual gate, not the happy path.

Every case here is a way `authorize()` could fail open, taken from the WP5
spec's own gate description almost verbatim: a model proposing something above
its authority must be queued or refused, never silently applied; a value that
merely *looks* like a valid parameter (a numeric string, NaN, an absurd
magnitude, an extra key) must be refused rather than coerced; a malformed
action id or actor string must be refused on shape before it ever reaches a
table lookup. `authorize()` never raises — every case below is expressed as
`allowed is False` or `needs_approval is True`, not as an exception, because a
policy gate that can be crashed into "allow" by an odd input is not a gate.

Run:  python -m pytest tests/test_policy_adversarial.py -q
 (or) python3 tests/test_policy_adversarial.py
"""
import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import policy  # noqa: E402


# ============================================================================
# the gate's own headline scenario: a fake tier-1 verdict at R3 and at R4
# ============================================================================
def test_fake_tier1_quarantine_delete_lands_as_needs_approval_never_outright():
    r = policy.authorize("tier1:some-fine-tuned-model", "quarantine_delete_slug",
                         {"slug": "kaggle_weed_dupe_2026", "reason": "duplicate of an already-curated source"})
    assert policy.risk_of("quarantine_delete_slug") == "R3"
    assert r["allowed"] is True          # tier-1 IS permitted to propose an R3
    assert r["needs_approval"] is True   # but never to apply it outright


def test_fake_tier1_dataset_delete_is_refused_not_queued():
    for action_id in ("audit_registry_garbage_APPLY", "roboflow_delete_junk_apply"):
        assert policy.risk_of(action_id) == "R4"
        r = policy.authorize("tier1:some-fine-tuned-model", action_id, {})
        assert r["allowed"] is False, action_id
        # the spec is explicit that this is a refusal, not a pending approval —
        # a needs_approval left dangling True on a disallowed request would be
        # exactly the ambiguity the _result() builder exists to make impossible
        assert r["needs_approval"] is False, action_id


def test_fake_tier2_dataset_delete_is_also_refused():
    r = policy.authorize("tier2:some-vlm", "audit_registry_garbage_APPLY", {})
    assert r["allowed"] is False
    assert r["needs_approval"] is False


def test_r4_from_any_non_human_actor_is_refused():
    for actor in ("tier0:gemma4", "tier1:x", "tier2:x", "round-scheduler"):
        r = policy.authorize(actor, "roboflow_delete_junk_apply", {})
        assert r["allowed"] is False, actor
        assert r["needs_approval"] is False, actor


def test_r4_from_human_is_allowed_directly():
    r = policy.authorize("human:harry", "roboflow_delete_junk_apply", {})
    assert r["allowed"] is True
    assert r["needs_approval"] is False


# ============================================================================
# params that merely look valid
# ============================================================================
@pytest.mark.parametrize("bad_max_new", [
    "5",            # numeric string, not an int — never coerced
    "5.0",
    " 5",
    True,           # bool is a Python int subclass; must not stand in for 1
    False,
    -1,             # negative, below the declared min of 1
    0,
    51,             # one past the declared max of 50
    10**9,          # absurdly large
    3.5,            # a float where an int is declared
])
def test_brain_harvest_max_new_rejects_everything_that_is_not_a_clean_int(bad_max_new):
    r = policy.authorize("human:harry", "brain_harvest", {"max_new": bad_max_new})
    assert r["allowed"] is False, bad_max_new


@pytest.mark.parametrize("bad_score", [
    float("nan"),
    float("inf"),
    float("-inf"),
    "0.5",          # numeric string
    -0.01,          # just under the declared min
    1.01,           # just over the declared max
    1e308,          # finite but absurd
])
def test_round_train_min_dino_score_rejects_nan_inf_and_out_of_range(bad_score):
    r = policy.authorize("round-scheduler", "round_train", {"min_dino_score": bad_score})
    assert r["allowed"] is False, bad_score


def test_extra_unknown_key_is_refused_even_alongside_valid_ones():
    r = policy.authorize("human:harry", "brain_harvest",
                         {"time_h": 4, "max_new": 3, "sudo": True})
    assert r["allowed"] is False
    assert any("sudo" in why for why in r["reasons"])


def test_extra_unknown_key_on_a_zero_param_action_is_refused():
    r = policy.authorize("human:harry", "round_filter", {"threshold": 0.9})
    assert r["allowed"] is False


def test_params_that_is_not_an_object_is_refused():
    for bad_params in ("time_h=4", ["time_h", 4], 4, None):
        r = policy.authorize("human:harry", "brain_harvest", bad_params)
        # None is the documented "no params supplied" shape and must succeed;
        # everything else that is not a dict must be refused.
        if bad_params is None:
            assert r["allowed"] is True
        else:
            assert r["allowed"] is False, bad_params


def test_a_non_string_param_key_is_refused():
    r = policy.authorize("human:harry", "brain_harvest", {3: "x"})
    assert r["allowed"] is False


# ============================================================================
# malformed action ids
# ============================================================================
@pytest.mark.parametrize("bad_action", [
    "brain_harvest; rm -rf /",
    "brain_harvest && cat /etc/passwd",
    "brain_harvest|cat",
    "$(whoami)",
    "`whoami`",
    "brain_harvest\nsudo reboot",
    "../../../etc/passwd",
    "brain_harvest ",     # trailing whitespace is still not the real id
    "",
    " ",
])
def test_action_id_with_shell_metacharacters_or_bad_shape_is_refused(bad_action):
    r = policy.authorize("human:harry", bad_action, {})
    assert r["allowed"] is False
    assert r["needs_approval"] is False
    assert r["risk"] is None    # the action was never resolved at all


def test_action_id_wrong_type_is_refused():
    for bad_action in (None, 123, ["brain_harvest"], {"a": 1}):
        r = policy.authorize("human:harry", bad_action, {})
        assert r["allowed"] is False


def test_unknown_but_well_formed_action_id_is_refused():
    r = policy.authorize("human:harry", "definitely_not_a_real_action", {})
    assert r["allowed"] is False
    assert r["risk"] is None


# ============================================================================
# actor strings that try to smuggle a higher tier
# ============================================================================
@pytest.mark.parametrize("sneaky_actor", [
    "tier1:x human:harry",
    "human:harry tier1:x",
    "tier1:x;human:harry",
    "tier1:x\nhuman:harry",
    "tier1:x\thuman:harry",
    " human:harry",
    "human:harry ",
    "humanharry",              # missing the separating colon
    "HUMAN:harry",             # case-sensitive on purpose
    "Tier1:model",
    "tier3:model",             # not a real tier
    "round-scheduler:extra:colon",
    "human:",                  # empty identity
    "",
])
def test_malformed_actor_string_does_not_escalate_and_is_refused(sneaky_actor):
    r = policy.authorize(sneaky_actor, "restart_dashboard", {})
    assert r["allowed"] is False
    assert r["needs_approval"] is False


def test_the_same_action_with_the_real_human_actor_does_succeed():
    # proves the refusals above are about the malformed string, not the action
    r = policy.authorize("human:harry", "restart_dashboard", {})
    assert r["allowed"] is True


def test_actor_wrong_type_is_refused():
    for bad_actor in (None, 123, ["human:harry"], {"tier": "human"}):
        r = policy.authorize(bad_actor, "roboflow_state_audit", {})
        assert r["allowed"] is False


def test_round_scheduler_identity_suffix_does_not_grant_a_different_tier():
    # round-scheduler carries no model identity in the spec's examples, but an
    # optional identity suffix must not be read as anything but round-scheduler
    r = policy.authorize("round-scheduler:weed", "round_train", {"train_time_h": 12})
    assert r["allowed"] is True
    assert r["needs_approval"] is False
    r2 = policy.authorize("round-scheduler:weed", "audit_registry_garbage_APPLY", {})
    assert r2["allowed"] is False


# ============================================================================
# internal exceptions never propagate as an allow
# ============================================================================
def test_authorize_never_raises_on_hostile_input():
    hostile = [
        (object(), "brain_harvest", {}),
        ("human:harry", object(), {}),
        ("human:harry", "brain_harvest", object()),
        ("human:harry", "brain_harvest", {"time_h": object()}),
        ("human:harry", "brain_harvest", {"time_h": [1, 2, 3]}),
        (None, None, None),
        ("human:harry", "brain_harvest", {"domain": "x" * 100000}),   # oversize string
    ]
    for actor, action, params in hostile:
        r = policy.authorize(actor, action, params)
        assert r["allowed"] is False
        assert r["needs_approval"] is False
        assert isinstance(r["reasons"], list) and r["reasons"]


def test_authorize_survives_a_budget_state_that_is_not_an_object():
    r = policy.authorize("human:harry", "brain_harvest", {}, budget_state="not a dict")
    assert r["allowed"] is True     # a malformed budget_state must not block an otherwise-fine request


def test_authorize_survives_a_resources_state_that_is_not_an_object():
    r = policy.authorize("human:harry", "brain_harvest", {}, resources=12345)
    assert r["allowed"] is True


# ============================================================================
# table integrity under adversarial monkeypatching of the loaded cache
# ============================================================================
def test_a_malformed_row_injected_into_the_live_cache_is_refused_not_crashed():
    """authorize() must fail closed even if the in-memory table were corrupted
    (e.g. a future bug in _table()'s caching). This does not touch the file on
    disk and restores the cache afterward."""
    table = policy._table()
    saved_actions = dict(table["actions"])
    try:
        table["actions"]["evil_action"] = {"template": "sbatch rm.sh"}  # missing every other key
        r = policy.authorize("human:harry", "evil_action", {})
        assert r["allowed"] is False
    finally:
        table["actions"].clear()
        table["actions"].update(saved_actions)


def test_gpu_count_param_absent_from_params_is_unknown_not_guessed():
    # No real row uses gpu_count_param today; exercise the formula path
    # directly so a future table entry that does gets a proven-safe behaviour.
    table = policy._table()
    saved_actions = dict(table["actions"])
    saved_rates = table["rates"]
    try:
        # Rates are resolved through su_ledger's shared table, so a fixture rate
        # is injected by swapping the whole resolver rather than by writing into
        # it -- the resolver is deliberately not writable, since a rate edited in
        # one place and not the other is the drift it exists to prevent.
        table["rates"] = {"fake-gpu": 3.0}
        table["actions"]["fake_multi_gpu_action"] = {
            "template": "sbatch fake.sh", "param_bounds": {},
            "risk": "R2", "reversible": True,
            "est_su": {"gpu_type": "fake-gpu", "gpu_count_param": "gpus",
                      "hours": 1.0, "why": "test fixture"},
            "dry_run_variant": None, "allowed_tiers": ["human"],
            "description": "test fixture",
        }
        est = policy.estimate_su("fake_multi_gpu_action", {})
        assert est["su"] is None and est["confident"] is False
        est2 = policy.estimate_su("fake_multi_gpu_action", {"gpus": 4})
        assert math.isclose(est2["su"], 12.0)
    finally:
        table["actions"].clear()
        table["actions"].update(saved_actions)
        table["rates"] = saved_rates


def test_su_rates_come_from_the_shared_ledger_table():
    """There must be exactly one statement of the H100 and V100 rates.

    policy_actions.json carried its own copy until v3.29.0. Two files stating the
    same pre-registered number drift, and the pair that drifts here is the price
    that gates an approval against the price that is reconciled with sacct.
    """
    from weed_optimizer_framework.tools.brain import su_ledger
    raw = json.loads(pathlib.Path(policy._table_path()).read_text())
    assert "gpu_su_per_hour" not in raw, \
        "the action catalogue must not restate the SU rates"
    rates = policy._table()["rates"]
    for gres, family in (("h100-80", "h100"), ("v100-32", "v100")):
        shared = su_ledger.su_for(gres, 1, 3600.0)
        assert not shared["unknown_rate"]
        assert rates.get(gres) == shared["value"], gres
        assert family in rates.keys()
    # An unresolvable family is absent, not priced at the ledger's fallback:
    # approving against a guessed price is what this gate exists to prevent.
    assert "definitely-not-a-gpu" not in rates
    assert rates.get("definitely-not-a-gpu") is None


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
