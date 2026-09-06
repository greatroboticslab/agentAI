"""Agent-proposed experiments: pick from the sealed lever menu, never invent
one (v3.29.0, WP6).

Why this exists
----------------
`db.py`'s `DEFAULT_DOMAIN_CONFIG["lever_menu"]` is pre-registered for a
specific reason its own comment states: a lever chosen after seeing the
result is not an experiment, and a planner shown this table during its own
evaluation is doing answer-key lookup rather than planning
(`digest.build`'s `omit_levers` exists for that arm). This module is the
other side of that contract for the PRODUCTION path -- the one that is
allowed to see the menu and turn it into concrete, filed proposals:
`propose()` reads `config["lever_menu"]` and nothing else, so a lever id that
is not already a row in that table can never appear in its output.

What a "proposal" is, and what an "approval" is
-------------------------------------------------
`propose()` returns proposals: one entry per lever the config declares,
carrying the lever's own `control` string, its candidate variant value(s),
and a short hypothesis built from the menu's own `reason` field. Nothing
here invents a control or a variant that is not already data on the row.

`to_approval()` turns one proposal into the R3 approval-queue item a human
reviews before `run_experiment_array.sh` submits it. It is filed at R3
regardless of the underlying lever's own risk tag (every current menu entry
is tagged R2, describing ONE config change) because submitting a 3-seed
array is itself the SU-heavy, design-changing act `policy.py`'s own R3
definition names -- the lever's tag is kept alongside as `lever_risk` for
reference, not overwritten. `est_su` is `policy.estimate_su`'s own number for
the shape every named proposal submits through (a `round_train` job); this
module never re-derives a GPU-hour formula of its own, because a second copy
of that formula is exactly the kind of drift `policy.py`'s own `_RateTable`
docstring warns about.

Results, and why a mean never travels alone
---------------------------------------------
`record_result()` builds one row for the `brain_experiments` collection that
carries `mean`, `std`, `n` and the recipe's `noise_floor` together, always.
`mean`/`std` describe the DELTA this experiment measured against its own
control (the caller has already compared the tested variant's seeds against
the control's) -- `noise_floor` is sealed as a round-to-round delta
threshold, the same way `digest.py`'s own trajectory section uses it, and a
single arm's raw score cannot be judged against a delta threshold at all.

`verdict()` reads a result plus a noise floor and returns exactly one of
`better` / `worse` / `within_noise` / `insufficient`. `insufficient` at
n < 3 is not a lesser win: the sealed noise floor was derived from a
three-seed spread (seeds 101/102/103), and a run with fewer seeds has no
spread of its own to compare against it -- reporting anything else would let
a one-seed run be quoted as if it had already beaten the campaign's noise.

Pure stdlib, no Mongo import: like every other module in this package, this
takes data the caller already fetched (`config`, a `proposal`, a `result`)
and never queries a database itself.
"""
from . import policy

TOOL_VERSION = "wp6-experiments/1"

# The pre-registered seed IDs a 3-seed array indexes into
# (`run_experiment_array.sh --array=1-3`). This is the actual identifier set
# the sbatch array uses, not a re-derivation of the SEED COUNT rule
# `parallelism_rules.json`'s `seeds_for_claim` already owns; the two are
# different things (an id set vs. a count) that happen to share cardinality.
_ARRAY_SEEDS = (101, 102, 103)

# The three levers TIERED_SUPERVISION_PLAN's worked example names as the
# campaign's first proposals, and the exact order they are named in. Every
# other lever in a given config's menu follows in that menu's own order.
_PRIORITY_LEVER_IDS = ("core_recipe", "per_box_verification_gate", "fresh_start")

# For these three pinned levers only, the specific variant value(s) under
# test -- hand-matched to the spec's literal wording: "(1) cwd12_core +
# audited sources vs merged_curated (control = cwd12_core alone); (2)
# per-box verification gate on vs off ...; (3) fresh_start vs continue ...".
# Every OTHER lever is proposed with ALL of its own sealed options as
# candidate variants; a variant equal to the control there is a legitimate
# no-op arm, not a defect, so no exclusion logic is needed for those.
_NAMED_VARIANTS = {
    "core_recipe": ["cwd12_core+audited", "merged_curated"],
    "per_box_verification_gate": [True],
    "fresh_start": [True],
}

# Every named proposal submits through the same per-round training job
# shape (`run_experiment_array.sh` wraps the same recipe `run_m1_merged_seeds.sh`
# already runs, three times). `round_train` is the policy table's own,
# already-authorised estimate for that shape; pointing every proposal at it
# is what keeps `to_approval`'s `est_su` sourced from `policy.estimate_su`
# rather than a second, per-lever GPU-hour formula this module would have to
# maintain and could let drift from the rate table `policy.py` owns.
_ESTIMATE_ACTION = "round_train"


def _variants_for(lever):
    lid = str(lever.get("id") or "")
    if lid in _NAMED_VARIANTS:
        return list(_NAMED_VARIANTS[lid])
    options = lever.get("options")
    if isinstance(options, list):
        return list(options)
    if isinstance(options, dict):
        # epoch_budget-shaped: parallel lists keyed by parameter name, paired
        # POSITIONALLY -- the menu's own control string ("60 epochs at a
        # 10.8 h cap") names a pair, not two independently-varying axes, so
        # index i of one list goes with index i of every other.
        keys = sorted(k for k in options if isinstance(options.get(k), list))
        lengths = {len(options[k]) for k in keys}
        if keys and len(lengths) == 1:
            n = next(iter(lengths))
            return [{k: options[k][i] for k in keys} for i in range(n)]
        return [options]
    return [options] if options is not None else []


def _one_proposal(domain, lever, digest_sha):
    return {
        "domain": str(domain),
        "lever_id": str(lever.get("id")),
        "question": str(lever.get("question") or ""),
        "control": str(lever.get("control") or ""),
        "variants": _variants_for(lever),
        "applies_to": lever.get("applies_to"),
        "risk": lever.get("risk"),
        "menu_est_su": lever.get("est_su"),
        "hypothesis": "Testing %s: %s" % (
            lever.get("question") or lever.get("id"),
            lever.get("reason") or "(no reason recorded on the menu)"),
        "seeds": list(_ARRAY_SEEDS),
        "digest_sha256": digest_sha,
    }


def propose(domain, config, digest=None):
    """[proposal, ...], one per lever in `config["lever_menu"]`, in an order
    that puts the three pinned levers (core_recipe,
    per_box_verification_gate, fresh_start) first, in that order, followed
    by every other lever the menu declares, in the menu's own order.

    `digest` is optional grounding only: when given, its `sha256` travels on
    every proposal so a later approval or result can be traced back to the
    campaign state that suggested it. Nothing here reads the digest's text --
    proposing FROM the sealed menu, not from a symptom read, is what makes
    this the counterpart to `planner.py`'s digest-only mock backend rather
    than a duplicate of it.
    """
    menu = config.get("lever_menu") if isinstance(config, dict) else None
    menu = menu if isinstance(menu, list) else []

    by_id, order = {}, []
    for lever in menu:
        if not isinstance(lever, dict) or not lever.get("id"):
            continue
        lid = str(lever["id"])
        if lid in by_id:
            continue    # a duplicate id in config would otherwise shadow the first
        by_id[lid] = lever
        order.append(lid)

    ordered_ids = [lid for lid in _PRIORITY_LEVER_IDS if lid in by_id]
    ordered_ids += [lid for lid in order if lid not in _PRIORITY_LEVER_IDS]

    digest_sha = digest.get("sha256") if isinstance(digest, dict) else None
    digest_sha = str(digest_sha) if digest_sha is not None else None

    return [_one_proposal(domain, by_id[lid], digest_sha) for lid in ordered_ids]


def to_approval(proposal):
    """One R3 approval-queue item for `proposal`.

    `est_su` is `policy.estimate_su(_ESTIMATE_ACTION, {})["su"]` verbatim --
    an empty `params` lets the policy table's own declared `hours_default`
    price the estimate, so this module supplies no hours/GPU figure of its
    own. `est_su_array_total` scales that one, already-authorised per-run
    number by the proposal's own seed count for a human-facing total; that is
    the same "per-unit times n" scaling `parallelism.py`'s own
    `est_su_total` performs on an externally-sourced per-unit figure, not a
    second derivation of the rate itself.
    """
    proposal = proposal if isinstance(proposal, dict) else {}
    est = policy.estimate_su(_ESTIMATE_ACTION, {})
    seeds = proposal.get("seeds") or list(_ARRAY_SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, list) else len(_ARRAY_SEEDS)
    su = est.get("su")
    total = round(su * n_seeds, 3) if isinstance(su, (int, float)) and not isinstance(su, bool) else None
    return {
        "kind": "experiment_proposal",
        "domain": proposal.get("domain"),
        "lever_id": proposal.get("lever_id"),
        "question": proposal.get("question"),
        "control": proposal.get("control"),
        "variants": proposal.get("variants"),
        "hypothesis": proposal.get("hypothesis"),
        "seeds": seeds,
        "risk": "R3",
        "lever_risk": proposal.get("risk"),
        "needs_approval": True,
        "est_su_action": _ESTIMATE_ACTION,
        "est_su": su,
        "est_su_confident": est.get("confident"),
        "est_su_reason": est.get("reason"),
        "est_su_array_total": total,
        "digest_sha256": proposal.get("digest_sha256"),
    }


def _as_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def record_result(proposal, recipe, mean, std, n, noise_floor, extra=None):
    """One `brain_experiments` row for a completed run.

    `mean`/`std` are the DELTA this experiment measured (tested variant
    against its own control across `n` seeds), not a single arm's raw score
    -- see the module docstring for why a bare arm score cannot be judged
    against a noise floor sealed as a delta threshold. `noise_floor` is
    stored on this same row on purpose: a consumer reading one row from
    `brain_experiments` always sees what the mean was measured against, and
    can never quote the mean by itself.
    """
    proposal = proposal if isinstance(proposal, dict) else {}
    return {
        "domain": proposal.get("domain"),
        "lever_id": proposal.get("lever_id"),
        "control": proposal.get("control"),
        "variants": proposal.get("variants"),
        "recipe": recipe,
        "seeds": proposal.get("seeds"),
        "mean": _as_float(mean),
        "std": _as_float(std),
        "n": int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) else 0,
        "noise_floor": _as_float(noise_floor),
        "extra": dict(extra) if isinstance(extra, dict) else {},
    }


def verdict(result, noise_floor):
    """`"better"` | `"worse"` | `"within_noise"` | `"insufficient"`.

    `insufficient` covers both n < 3 (the sealed floor was derived from a
    three-seed spread; fewer seeds have no comparable spread of their own)
    and a missing mean or floor (nothing to compare at all) -- in every case
    the honest answer is that no verdict can be reached, never a winner
    reported on incomplete evidence.
    """
    result = result if isinstance(result, dict) else {}
    n = result.get("n")
    if not isinstance(n, (int, float)) or isinstance(n, bool) or n < 3:
        return "insufficient"
    mean = _as_float(result.get("mean"))
    floor = _as_float(noise_floor)
    if mean is None or floor is None:
        return "insufficient"
    if abs(mean) < floor:
        return "within_noise"
    return "better" if mean > 0 else "worse"


# --- CLI ---------------------------------------------------------------------
def _main(argv):
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="experiments", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    pp = sub.add_parser("propose", help="proposals from a domain config's lever_menu")
    pp.add_argument("config_file")
    pp.add_argument("--domain", default="weed")

    ap_ = sub.add_parser("approval", help="the R3 approval item for one proposal")
    ap_.add_argument("proposal_file")

    vp = sub.add_parser("verdict", help="better/worse/within_noise/insufficient")
    vp.add_argument("result_file")
    vp.add_argument("noise_floor", type=float)

    args = ap.parse_args(argv)
    if args.cmd == "propose":
        with open(args.config_file, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        print(json.dumps(propose(args.domain, config), indent=2, sort_keys=True))
        return 0
    if args.cmd == "approval":
        with open(args.proposal_file, "r", encoding="utf-8") as fh:
            proposal = json.load(fh)
        print(json.dumps(to_approval(proposal), indent=2, sort_keys=True))
        return 0
    if args.cmd == "verdict":
        with open(args.result_file, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        print(verdict(result, args.noise_floor))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
