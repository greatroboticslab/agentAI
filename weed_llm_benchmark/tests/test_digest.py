#!/usr/bin/env python3
"""Unit tests for the campaign digest (tools/brain/digest.py). No cluster, no
Mongo, no model: every input is an in-memory fixture built to exercise one
requirement of the module's contract.

Covered:
  * determinism — two builds of the same inputs are byte-identical, same sha256
  * trimming order — a too-small num_ctx drops the lowest-priority section
    first, and `omitted` names it
  * hard refusal — mandatory sections alone overflowing num_ctx comes back as
    a structured refusal, not an exception, and the refusal itself fits num_ctx
  * omit_levers removes the lever menu and nothing else
  * a within-noise round pair is reported as "within noise", never a winner
  * a single-seed round is labelled as such, never merged into a mean±std
  * a torn/malformed entry in any of the four historical inputs is skipped,
    counted, and never crashes the build
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import digest  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _config(noise_floor=0.005):
    return {
        "target_metric": "map50_95",
        "round_params": {"epochs": 60, "train_time_h": 12, "tier": "curated"},
        "budget": {"su_envelope": 1500.0, "daily_cap": 120, "per_round_cap": 60},
        "noise_floor": {"merged_curated": noise_floor, "merged_raw": 0.009},
        "taxonomy": ["broadleaf", "grass", "sedge"],
    }


def _round(n, metric, epochs=60, tier="curated", status="done", actor="round-scheduler"):
    return {
        "round_num": n, "status": status, "actor": actor,
        "created_at": "2026-08-%02dT00:00:00Z" % n,
        "updated_at": "2026-08-%02dT12:00:00Z" % n,
        "steps": {
            "train": {"status": "done", "actor": actor,
                      "params": {"epochs": epochs, "train_time_h": 12, "tier": tier}},
        },
        "metrics": {} if metric is None else {"map50_95": metric},
    }


def _campaign(n_rounds=6):
    """A small, otherwise-ordinary campaign: steadily improving single-seed
    rounds, one signal firing, one correction that addresses it, and a small
    ledger. Everything fits comfortably under a generous num_ctx."""
    rounds = [_round(i, 0.80 + i * 0.01,
                     epochs=60 if i < 3 else 24) for i in range(1, n_rounds + 1)]
    signals_history = [
        {"round": 3, "step": "train", "job_id": "44727703", "signal": "pool_growth",
         "severity": "warn", "value": 0.37, "reason": "iterations/epoch grew 37%"},
    ]
    corrections_history = [
        {"seq": 1, "author": "tier1:glm-4.7-flash", "kind": "set_round_param",
         "target": {"key": "epochs", "old": 60, "new": 24},
         "scope": {"from_round": 3}, "reason": "bound the walltime overrun",
         "signal": "pool_growth"},
    ]
    ledger = [{"job": "44727703", "round": 3, "su": 24.6, "gpu_type": "h100"}]
    return rounds, signals_history, corrections_history, ledger, _config()


def _build(num_ctx=8000, **overrides):
    rounds, sig, corr, ledger, cfg = _campaign()
    kwargs = dict(domain="weed", rounds=rounds, signals_history=sig,
                 corrections_history=corr, ledger=ledger, config=cfg,
                 num_ctx=num_ctx)
    kwargs.update(overrides)
    return digest.build(**kwargs)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def test_determinism_byte_identical_text_and_hash():
    rounds, sig, corr, ledger, cfg = _campaign()
    r1 = digest.build("weed", rounds, sig, corr, ledger, cfg, 8000)
    # Fresh, independently-built fixtures (not the same objects) so this
    # actually exercises determinism rather than object identity.
    rounds2, sig2, corr2, ledger2, cfg2 = _campaign()
    r2 = digest.build("weed", rounds2, sig2, corr2, ledger2, cfg2, 8000)
    assert r1["refused"] is False
    assert r1["text"] == r2["text"]
    assert r1["sha256"] == r2["sha256"]
    assert r1["sha256"] == digest.hashlib.sha256(r1["text"].encode("utf-8")).hexdigest()


def test_determinism_survives_input_reordering_via_sort():
    """Signals/corrections are sorted internally, so handing them over in a
    different order must not change the rendered text."""
    rounds, sig, corr, ledger, cfg = _campaign()
    r1 = digest.build("weed", rounds, sig, corr, ledger, cfg, 8000)
    r2 = digest.build("weed", list(reversed(rounds)), list(reversed(sig)),
                      list(reversed(corr)), list(reversed(ledger)), cfg, 8000)
    assert r1["text"] == r2["text"]


# --------------------------------------------------------------------------
# trimming order
# --------------------------------------------------------------------------
def test_trim_drops_lowest_priority_section_first():
    baseline = _build(num_ctx=100000)
    assert baseline["refused"] is False
    assert baseline["omitted"] == []
    budget = digest.load_budget()
    reserve = budget["meta"]["prompt_reserve_tokens"]
    # Just enough of a squeeze to force exactly one whole-section drop.
    forced_ctx = baseline["tokens_est"] + reserve - 1
    result = _build(num_ctx=forced_ctx)
    assert result["refused"] is False

    priorities = {n: cfg["priority"] for n, cfg in budget["sections"].items()}
    dropped = [o for o in result["omitted"] if "dropped whole section" in o["action"]]
    assert dropped, "expected at least one whole-section drop under a forced squeeze"
    dropped_names = {d["section"] for d in dropped}
    included_names = {s["name"] for s in result["sections"] if s["included"]}
    # every dropped section's priority must be <= every surviving non-mandatory
    # section's priority (lowest priority goes first)
    surviving_non_mandatory = [s for s in result["sections"]
                               if s["included"] and not s["mandatory"]]
    for d in dropped_names:
        for s in surviving_non_mandatory:
            assert priorities[d] <= priorities[s["name"]], (
                "%s (priority %d) was dropped while %s (priority %d) survived"
                % (d, priorities[d], s["name"], priorities[s["name"]]))
    # mandatory sections are never in the dropped set
    assert "identity" not in dropped_names
    assert "state" not in dropped_names
    # the lowest-priority section overall (levers, priority 0) is the one that
    # actually left, since only one drop was forced
    assert "levers" in dropped_names
    assert "levers" not in included_names
    assert result["tokens_est"] + reserve <= forced_ctx


def test_trim_names_what_it_removed_and_never_touches_the_middle_silently():
    """A section trimmed internally (windowed, not dropped whole) must show up
    in `omitted` with a removed-row count, and the surviving text still shows
    the earliest and the most recent rows — never a blind mid-string cut."""
    # Many rounds and many signal firings so the incidents section overflows
    # its own per-section budget (2,600 tokens in digest_budget.json) and must
    # window internally well before the whole-digest num_ctx is ever a factor.
    n = 200
    rounds = [_round(i, 0.80 + i * 0.0001) for i in range(1, n + 1)]
    signals_history = [
        {"round": i, "step": "train", "job_id": "job%d" % i, "signal": "pool_growth",
         "severity": "warn", "value": 0.3,
         "reason": "iterations/epoch grew on round %d of the campaign" % i}
        for i in range(1, n + 1)
    ]
    cfg = _config()
    result = digest.build("weed", rounds, signals_history, [], [], cfg, 400000)
    assert result["refused"] is False
    windowed = [o for o in result["omitted"]
               if o["section"] == "incidents" and "windowed" in o["action"]]
    assert windowed, "expected the incidents section to window under its own budget"
    assert windowed[0]["removed"] > 0
    assert windowed[0]["tokens_saved"] > 0
    text = result["text"]
    assert "round 1 " in text or "round 1:" in text  # earliest kept
    assert ("round %d" % n) in text                  # most recent kept
    assert ("row(s) omitted here" in text)            # the cut is named in-line


# --------------------------------------------------------------------------
# hard refusal
# --------------------------------------------------------------------------
def test_hard_refusal_when_mandatory_sections_alone_overflow():
    result = _build(num_ctx=90)
    assert result["refused"] is True
    assert result["reason"]
    assert "identity" in result["offending_sections"] or "state" in result["offending_sections"]
    assert result["sections"] == []
    assert result["omitted"] == []
    assert result["tokens_est"] < 90
    assert "REFUSED" in result["text"]
    assert result["sha256"] == digest.hashlib.sha256(
        result["text"].encode("utf-8")).hexdigest()


def test_refusal_is_structured_not_an_exception():
    # A pathologically tiny num_ctx must come back as data, never a traceback,
    # and the refusal must still obey the module's own "fits num_ctx" contract
    # even when no ordinary sentence can (the empty string is the true floor).
    for tiny in (1, 2, 5, 20):
        result = _build(num_ctx=tiny)
        assert result["refused"] is True
        assert isinstance(result["text"], str)
        assert result["tokens_est"] < tiny


def test_refusal_message_is_readable_at_a_realistic_tiny_num_ctx():
    result = _build(num_ctx=90)
    assert result["refused"] is True
    assert "REFUSED" in result["text"]
    assert "mandatory section" in result["reason"]


def test_invalid_num_ctx_is_a_programming_error_not_a_refusal():
    import pytest
    rounds, sig, corr, ledger, cfg = _campaign()
    with pytest.raises(ValueError):
        digest.build("weed", rounds, sig, corr, ledger, cfg, 0)


# --------------------------------------------------------------------------
# omit_levers
# --------------------------------------------------------------------------
def test_omit_levers_removes_only_the_lever_menu():
    with_levers = _build(num_ctx=100000)
    without_levers = _build(num_ctx=100000, omit_levers=True)
    assert with_levers["refused"] is False and without_levers["refused"] is False

    assert "PRE-REGISTERED LEVERS" in with_levers["text"]
    assert "PRE-REGISTERED LEVERS" not in without_levers["text"]
    assert "round_params.epochs" in with_levers["text"]
    assert "round_params.epochs" not in without_levers["text"]

    # Every other section's own rendered text is untouched.
    names_with = {s["name"]: s for s in with_levers["sections"]}
    names_without = {s["name"]: s for s in without_levers["sections"]}
    for name in digest.SECTION_NAMES:
        if name == "levers":
            assert names_with[name]["included"] is True
            assert names_without[name]["included"] is False
            continue
        assert names_with[name]["tokens_est"] == names_without[name]["tokens_est"], name
        assert names_with[name]["rows_kept"] == names_without[name]["rows_kept"], name

    assert names_without["levers"]["excluded_reason"] == "omit_levers=True"
    # Neither call trimmed anything for lack of room (num_ctx is generous),
    # so omit_levers is the ONLY difference between the two digests.
    assert with_levers["omitted"] == []
    assert without_levers["omitted"] == []


# --------------------------------------------------------------------------
# honest metric aggregation
# --------------------------------------------------------------------------
def test_within_noise_pair_is_not_reported_as_a_winner():
    cfg = _config(noise_floor=0.01)
    rounds = [
        _round(1, 0.8000),
        _round(2, 0.8030),   # delta 0.0030 < 0.01 floor -> within noise
    ]
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    assert result["refused"] is False
    assert "within noise" in result["text"]
    assert "improved" not in result["text"]
    assert "regressed" not in result["text"]


def test_a_real_improvement_above_the_floor_is_reported_as_one():
    cfg = _config(noise_floor=0.005)
    rounds = [_round(1, 0.8000), _round(2, 0.8300)]  # delta 0.03 >> floor
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    assert "improved" in result["text"]
    assert "within noise" not in result["text"]


def test_recipe_change_between_rounds_skips_noise_comparison():
    cfg = _config(noise_floor=0.005)
    rounds = [_round(1, 0.80, tier="raw"), _round(2, 0.803, tier="curated")]
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    assert "recipe changed from merged_raw to merged_curated" in result["text"]
    assert "within noise" not in result["text"]


# --------------------------------------------------------------------------
# single-seed vs mean+/-std
# --------------------------------------------------------------------------
def test_single_seed_round_is_labelled_and_not_a_mean():
    cfg = _config()
    rounds = [_round(1, 0.8123)]
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    text = result["text"]
    assert "single seed, n=1" in text
    assert "not a mean" in text
    assert "std=" not in text.split("== RECIPE")[0].split("== METRIC")[1]


def test_multi_seed_round_reports_mean_and_std_with_n():
    cfg = _config()
    rounds = [_round(1, {"mean": 0.8500, "std": 0.0041, "n": 3})]
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    text = result["text"]
    assert "mean=0.8500 std=0.0041 (n=3 seeds)" in text
    assert "single seed" not in text.split("== RECIPE")[0].split("== METRIC")[1]


def test_a_reported_n_of_one_dict_is_honestly_single_not_a_mean():
    """A `{"mean":...,"n":1}` block is exactly one seed; it must never be
    rendered next to a std as though it were comparable to a real mean+/-std."""
    cfg = _config()
    rounds = [_round(1, {"mean": 0.81, "std": 0.0, "n": 1})]
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    section = result["text"].split("== RECIPE")[0].split("== METRIC")[1]
    assert "single seed" in section
    assert "mean=" not in section


# --------------------------------------------------------------------------
# torn / malformed inputs never crash the build
# --------------------------------------------------------------------------
def test_torn_ledger_line_is_skipped_and_counted():
    cfg = _config()
    rounds = [_round(1, 0.80)]
    ledger = [{"job": "1", "round": 1, "su": 12.0}, "{not valid json,,,", "{}", 5, None]
    result = digest.build("weed", rounds, [], [], ledger, cfg, 8000)
    assert result["refused"] is False
    assert "unparseable ledger line(s) were skipped" in result["text"]


def test_torn_round_entry_is_skipped_and_counted():
    cfg = _config()
    rounds = [_round(1, 0.80), "{not json", None, 3.14, _round(2, 0.81)]
    result = digest.build("weed", rounds, [], [], [], cfg, 8000)
    assert result["refused"] is False
    assert "malformed round entry(ies) were skipped" in result["text"]
    assert "round 2" in result["text"]


def test_torn_signal_and_correction_entries_are_skipped_and_counted():
    cfg = _config()
    rounds = [_round(1, 0.80)]
    signals_history = [
        {"round": 1, "step": "train", "signal": "gate_noop", "severity": "warn",
         "reason": "kept everything scored"},
        "{torn signal line",
        {"round": 1, "step": "train"},   # no "signal" key: also skipped
    ]
    corrections_history = [
        {"seq": 1, "author": "human:harry", "kind": "set_round_param",
         "target": {"key": "epochs", "old": 60, "new": 30},
         "scope": {"from_round": 1}, "reason": "test"},
        "{torn correction",
    ]
    result = digest.build("weed", rounds, signals_history, corrections_history, [], cfg, 8000)
    assert result["refused"] is False
    text = result["text"]
    assert "malformed signal-history entry(ies) were skipped" in text
    assert "malformed correction entry(ies) were skipped" in text
    assert "gate_noop" in text
    assert "human:harry" in text


def test_a_bundle_shaped_signals_history_group_is_also_accepted():
    """The grouped `{"round","step","signals":[...]}` shape must normalize the
    same way the flat one-firing-per-row shape does."""
    cfg = _config()
    rounds = [_round(1, 0.80)]
    grouped = [{"round": 1, "step": "train", "job_id": "j1",
               "signals": [{"signal": "walltime_bound", "severity": "crit",
                           "reason": "sacct reports TIMEOUT"},
                          "{torn nested signal",
                          {"severity": "warn"}]}]     # no "signal": skipped
    result = digest.build("weed", rounds, grouped, [], [], cfg, 8000)
    assert result["refused"] is False
    assert "walltime_bound crit" in result["text"]
    assert "2 malformed signal-history entry(ies) were skipped" in result["text"]


# --------------------------------------------------------------------------
# recurrence bookkeeping (corrections <-> signals)
# --------------------------------------------------------------------------
def test_correction_recurrence_is_reported_when_the_signal_fires_again():
    cfg = _config()
    rounds = [_round(1, 0.80), _round(2, 0.81), _round(3, 0.82)]
    signals_history = [
        {"round": 1, "step": "train", "signal": "pool_growth", "severity": "warn",
         "reason": "first firing"},
        {"round": 3, "step": "train", "signal": "pool_growth", "severity": "warn",
         "reason": "fired again after the fix"},
    ]
    corrections_history = [
        {"seq": 1, "author": "tier1:m", "kind": "set_round_param",
         "target": {"key": "epochs", "old": 60, "new": 30},
         "scope": {"from_round": 1}, "reason": "bound it", "signal": "pool_growth"},
    ]
    result = digest.build("weed", rounds, signals_history, corrections_history, [], cfg, 8000)
    assert "recurred at round 3" in result["text"]


def test_correction_without_a_signal_field_says_recurrence_cannot_be_checked():
    cfg = _config()
    rounds = [_round(1, 0.80)]
    corrections_history = [
        {"seq": 1, "author": "human:harry", "kind": "set_round_param",
         "target": {"key": "epochs", "old": 60, "new": 30},
         "scope": {"from_round": 1}, "reason": "manual tweak"},
    ]
    result = digest.build("weed", rounds, [], corrections_history, [], cfg, 8000)
    assert "not stated on this correction record" in result["text"]
    assert "recurred" not in result["text"]


# --------------------------------------------------------------------------
# module-level sanity
# --------------------------------------------------------------------------
def test_load_budget_declares_every_section_with_a_reason():
    budget = digest.load_budget()
    assert budget["error"] is None
    for name in digest.SECTION_NAMES:
        assert name in budget["sections"], name
        spec = budget["sections"][name]
        assert isinstance(spec["priority"], (int, float))
        assert isinstance(spec["mandatory"], bool)
        assert isinstance(spec["max_tokens"], (int, float))
        assert spec["why"].strip()


def test_mandatory_sections_are_identity_and_state_only():
    budget = digest.load_budget()
    mandatory = {n for n, cfg in budget["sections"].items() if cfg["mandatory"]}
    assert mandatory == {"identity", "state"}


def test_cli_sections_and_budget_do_not_crash():
    rc = digest.main(["sections"])
    assert rc == 0
    rc = digest.main(["budget"])
    assert rc == 0
