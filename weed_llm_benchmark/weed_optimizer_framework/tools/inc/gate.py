"""The INC gate: accept, hold or reject one increment, and record why.

docs/INCREMENTAL_PROTOCOL.md, section "Gate", is the contract and this module is
its only implementation, so every ledger decision is made by a rule rather than
by a person or a model. Inputs are score dicts from inc/scorer.py (score.json).
Nothing here trains, scores or imports Ultralytics, so a decision can be
re-derived from the ledger's inputs on any machine.

For one step, three kinds of dev score go in:
    inc     the incumbent W before the step (one score);
    cands   W fine-tuned on D_k plus replay, one score per seed;
    nulls   the same run with D_k replaced by more replay, one score per seed.
cand vs null isolates the data (same recipe, same compute); null vs inc isolates
the recipe. With three seeds per arm, a difference of means lets one lucky seed
carry the step, so the decision statistic is P(a > b) over every seed pair, a
tie counting one half (Bouthillier et al. 2021).

The guards are the written ones, with sample sds (ddof=1) and no floor:
    regression  mean(cand) >= inc - 2 sd(null)
    species     every one of the 12 species: mean(cand) - mean(null)
                >= -max(0.03, 3 sd_species(null))
    flips       mean(cand flips) - mean(null flips) <= 2 sd(null flips) + 3
A three-seed sd of exactly 0 therefore means "inc itself" for the regression
guard and "3 images" for the flips guard; the protocol's 0.03 and +3 are the
only slack. Seeds that tie exactly are recorded in Decision.warnings.

Flips, per run and against inc's image_correct: a negative flip is an image
correct under inc and incorrect under the run, a positive flip the reverse.
GateConfig.flips_mode picks what the flips guard counts:
    'negative'  protocol v1, the default: negative flips only, exactly as the
                guard was written first; the decision (config and guard dict
                included) is byte for byte what it was before the option;
    'net'       protocol v2 (docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2
                (net flips)"): negative - positive flips per run, with the same
                threshold form, mean(cand net) - mean(null net)
                <= 2 sd(null net) + 3. The guard records the negative, positive
                and net counts of every run, and the decision's config says
                flips_mode 'net'.
What 'net' tests. Per run, net = #(inc 1, run 0) - #(inc 0, run 1)
= inc_correct - run_correct exactly, so the incumbent's bits cancel:
mean(cand net) - mean(null net) = mean(null correct) - mean(cand correct) and
sd(null net) = sd(null correct). The v2 guard is therefore a guard on the
number of exactly-correct dev images, cand against null: cand may get at most
2 sd(null correct) + 3 fewer images right than the null. It does not measure
churn against W; the per-run negative and positive counts it records are
informational (and give v1's verdict on the same runs, v1_counterfactual()).
Decision.config is asdict(GateConfig) with flips_mode written only when it is
not 'negative' (absent = 'negative', as a v1 ledger has it). The driver pins
the whole config of an experiment defined with a gate block in state.json
(flips_mode included), so a new ledger never depends on the absence alone.

The score.json schema. The gate is its only consumer, so it checks all of it
and refuses rather than deciding on a score it cannot vouch for:
    exam              split name; every decision is taken on "dev"
    scorer_sha256     sha256 of the scorer.py that produced the score
    manifest_sha256   sha256 of the exam manifest it scored
    key_order_sha256  sha256 of the image order behind image_correct
    weights_sha256    sha256 of the weights scored (one per run)
    n_images          rows in the exam manifest, >= 1
    map50_95, map50, agnostic_map50_95, agnostic_map50   finite floats
    n_gt              {class name: box count} for all 13 CLASS_NAMES
    per_class         {class name: AP50-95}, every class with boxes present
    image_correct     n_images characters of 0/1, in key order
Every score in one call must share exam, scorer, manifest, key order, n_images
and n_gt: a score taken months earlier by an older scorer.py, or on another
exam, is not comparable and raises. No two scores in one call may come from the
same weights file: that is one run passed twice, not two seeds.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, fields

from .common import CLASS_NAMES, OTHER_PLANT
from .splits import DEV_MIN_BOXES

ACCEPT, HOLD, REJECT = "ACCEPT", "HOLD", "REJECT"
HELPS, HURTS, NEUTRAL = "helps", "hurts", "neutral"
LABELS, DOMAIN, NONE = "labels", "domain/localisation", "none"

# The species guard covers the twelve cwd12 species. OtherPlant is a catch-all
# class, not a species, and dev (cwd12 train sessions) has none.
SPECIES = tuple(CLASS_NAMES[:OTHER_PLANT])
DECISION_EXAM = "dev"       # test is read at anchor points only, never in a decision
STAMP_KEYS = ("exam", "scorer_sha256", "manifest_sha256", "key_order_sha256", "weights_sha256")
SHARED_KEYS = ("exam", "scorer_sha256", "manifest_sha256", "key_order_sha256", "n_images")
METRIC_KEYS = ("map50_95", "map50", "agnostic_map50_95", "agnostic_map50")
SCORE_KEYS = STAMP_KEYS + ("n_images",) + METRIC_KEYS + ("per_class", "n_gt", "image_correct")

# What the flips guard counts (GateConfig.flips_mode); 'negative' is protocol v1.
FLIPS_NEGATIVE, FLIPS_NET = "negative", "net"
FLIPS_MODES = (FLIPS_NEGATIVE, FLIPS_NET)
DEFAULT_FLIPS_MODE = FLIPS_NEGATIVE

# P values are k/2 over n*m pairs and sd products are floats; a threshold that is
# met exactly on paper must not fail on the last bit.
_EPS = 1e-12


@dataclass(frozen=True)
class GateConfig:
    """Every threshold of the gate; the defaults are the protocol's."""
    metric: str = "map50_95"            # the 12-class score; its agnostic twin is "agnostic_" + metric
    p_accept: float = 0.75              # ACCEPT / "helps" at P >= this
    p_reject: float = 0.25              # REJECT / "hurts" at P <= this
    p_recipe_flag: float = 0.25         # P(null > inc) <= this: the recipe itself degrades W
    regression_sd_mult: float = 2.0     # mean(cand) >= inc - 2 sd(null)
    species_min_drop: float = 0.03      # a species may drop by max(0.03, 3 sd_species(null))
    species_sd_mult: float = 3.0
    # dev is built with >= DEV_MIN_BOXES boxes of every species; fewer means a
    # broken dev or score, which raises instead of leaving a species unguarded.
    # A dev built with a relaxation (splits --dev-min-boxes / --dev-cap-rare)
    # departs from the protocol and must pass its own minimum here explicitly.
    min_species_gt: int = DEV_MIN_BOXES
    flips_sd_mult: float = 2.0          # mean(cand flips) - mean(null flips) <= 2 sd(null flips) + 3
    flips_slack_images: float = 3.0
    attr_drop_sd_mult: float = 2.0      # attribution: a score "drops" when delta < -2 sd(null)
    min_seeds: int = 3                  # fewer finished seeds is an error, not a decision
    # The scorer marks a score taken under INC_SCORER_TESTING (lock bypass, CPU,
    # other imgsz or Ultralytics version) production=false with a "TEST-" stamp.
    # Such a score may never decide a real step, even when every input is one.
    require_production: bool = True
    # What the flips guard counts: 'negative' (protocol v1, the default) or
    # 'net' (protocol v2: negative - positive flips); see the module docstring.
    # Last, so no positional use of the fields above changes meaning.
    flips_mode: str = DEFAULT_FLIPS_MODE

    def __post_init__(self):
        if self.flips_mode not in FLIPS_MODES:
            raise ValueError("flips_mode %r not in %s" % (self.flips_mode, FLIPS_MODES))


def config_record(cfg):
    """The config a decision records: asdict(cfg), with flips_mode only when it
    is not v1's 'negative', so a v1 decision is byte for byte what it was
    before the option existed (absent = 'negative')."""
    out = asdict(cfg)
    if out["flips_mode"] == DEFAULT_FLIPS_MODE:
        del out["flips_mode"]
    return out


@dataclass
class Decision:
    """One gate decision. to_dict() is plain JSON (the ledger stores it) and
    from_dict() restores it. Everything order-bearing is a list, so no JSON
    serialiser (sort_keys included) can reorder it."""
    verdict: str                 # ACCEPT | HOLD | REJECT
    reason: str                  # one line, for people reading the ledger
    metric: str
    p_data: float                # P(cand > null)
    p_recipe: float              # P(null > inc)
    inc: float
    cand_values: list
    null_values: list
    cand_mean: float
    cand_sd: float
    null_mean: float
    null_sd: float
    guards: dict                 # regression / species / flips, each with its numbers and "passed"
    attribution: dict
    stamps: dict                 # the exam, scorer, manifest and image order every input shares
    weights: dict                # weights_sha256 of inc, of each cand and of each null
    warnings: list               # suspicious but decidable inputs, e.g. seeds that tie exactly
    config: dict

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls)})


# ------------------------------------------------------------------ statistics
def p_greater(a_list, b_list):
    """P(a > b) over all len(a) * len(b) pairs, a tie counting 0.5."""
    a = [_finite(x, "p_greater a") for x in a_list]
    b = [_finite(x, "p_greater b") for x in b_list]
    if not a or not b:
        raise ValueError("p_greater needs two non-empty lists (got %d and %d)" % (len(a), len(b)))
    wins = 0.0
    for x in a:
        for y in b:
            if x > y:
                wins += 1.0
            elif x == y:
                wins += 0.5
    return wins / (len(a) * len(b))


def negative_flips(ref_bits, run_bits):
    """Images correct under the reference and incorrect under the run."""
    if len(ref_bits) != len(run_bits):
        raise ValueError("image_correct lengths differ (%d vs %d)" % (len(ref_bits), len(run_bits)))
    return sum(1 for r, x in zip(ref_bits, run_bits) if r == "1" and x == "0")


def positive_flips(ref_bits, run_bits):
    """Images incorrect under the reference and correct under the run."""
    if len(ref_bits) != len(run_bits):
        raise ValueError("image_correct lengths differ (%d vs %d)" % (len(ref_bits), len(run_bits)))
    return sum(1 for r, x in zip(ref_bits, run_bits) if r == "0" and x == "1")


def _finite(x, where):
    if isinstance(x, bool):
        raise ValueError("%s: %r is not a number" % (where, x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError("%s: %r is not a number" % (where, x)) from None
    if not math.isfinite(v):
        raise ValueError("%s: %r is not finite" % (where, x))
    return v


def _is_count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _sd(xs):
    """Sample sd (ddof=1); 0 for a single value (only reachable with min_seeds=1)."""
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _ge(a, b):
    return a >= b - _EPS


def _le(a, b):
    return a <= b + _EPS


# ------------------------------------------------------------------ validation
def _check_classes(name, s):
    n_gt, pc = s["n_gt"], s["per_class"]
    if not isinstance(n_gt, dict) or set(n_gt) != set(CLASS_NAMES):
        keys = set(n_gt) if isinstance(n_gt, dict) else set()
        raise ValueError("%s: n_gt must count boxes for exactly the %d class names (unknown %s, "
                         "missing %s); a scorer that keys by id or misspells a name would leave "
                         "species unguarded" % (name, len(CLASS_NAMES),
                                                sorted(map(str, keys - set(CLASS_NAMES))),
                                                [c for c in CLASS_NAMES if c not in keys]))
    bad = {c: v for c, v in n_gt.items() if not _is_count(v)}
    if bad:
        raise ValueError("%s: n_gt counts must be non-negative integers, got %s" % (name, bad))
    if not isinstance(pc, dict):
        raise ValueError("%s: per_class must be a dict, got %s" % (name, type(pc).__name__))
    unknown = sorted(str(k) for k in pc if k not in CLASS_NAMES)
    if unknown:
        raise ValueError("%s: per_class has keys that are not class names: %s" % (name, unknown))
    lacking = [c for c in CLASS_NAMES if n_gt[c] > 0 and c not in pc]
    if lacking:
        raise KeyError("%s: per_class has no AP for %s although the exam has boxes of them"
                       % (name, lacking))
    for c, v in pc.items():
        _finite(v, "%s per_class[%s]" % (name, c))


def _validate(named_scores, cfg):
    """Raise unless every score follows the schema (module docstring), shares
    every stamp with the others, is on dev, and comes from its own weights.
    Returns (shared stamps, {input name: weights_sha256})."""
    ref_name, ref = named_scores[0]
    need = SCORE_KEYS + tuple(k for k in (cfg.metric, "agnostic_" + cfg.metric)
                              if k not in SCORE_KEYS)
    weights = {}
    for name, s in named_scores:
        if not isinstance(s, dict):
            raise TypeError("%s: score must be a dict, got %s" % (name, type(s).__name__))
        missing = [k for k in need if k not in s]
        if missing:
            raise KeyError("%s: score is missing %s" % (name, missing))
        for k in STAMP_KEYS:
            if not isinstance(s[k], str) or not s[k]:
                raise ValueError("%s: %s must be a non-empty string, got %r" % (name, k, s[k]))
        for k in METRIC_KEYS + (cfg.metric, "agnostic_" + cfg.metric):
            _finite(s[k], "%s %s" % (name, k))
        n = s["n_images"]
        if not _is_count(n) or n < 1:
            raise ValueError("%s: n_images must be a positive integer, got %r" % (name, n))
        bits = s["image_correct"]
        if not isinstance(bits, str) or set(bits) - {"0", "1"}:
            raise ValueError("%s: image_correct must be a string of 0/1" % name)
        if len(bits) != n:
            raise ValueError("%s: image_correct has %d bits for %d images" % (name, len(bits), n))
        _check_classes(name, s)
        if cfg.require_production and (s.get("production") is not True
                                       or s["scorer_sha256"].startswith("TEST-")):
            raise ValueError("%s: not a production score (production=%r, scorer %s); test-mode "
                             "scores cannot decide" % (name, s.get("production"),
                                                       s["scorer_sha256"][:16]))
        for k in SHARED_KEYS:
            if s[k] != ref[k]:
                raise ValueError("%s has %s %s but %s has %s; scores from different exams, "
                                 "scorers or image orders cannot be compared"
                                 % (name, k, str(s[k])[:16], ref_name, str(ref[k])[:16]))
        if s["n_gt"] != ref["n_gt"]:
            raise ValueError("%s: n_gt differs from %s's; not the same exam" % (name, ref_name))
        w = s["weights_sha256"]
        dup = [other for other, ow in weights.items() if ow == w]
        if dup:
            raise ValueError("%s and %s were scored from the same weights (%s); one run was "
                             "passed twice, or a seed was not applied" % (dup[0], name, w[:12]))
        weights[name] = w
    if ref["exam"] != DECISION_EXAM:
        raise ValueError("scores are on exam %r; the gate decides on %r only"
                         % (ref["exam"], DECISION_EXAM))
    return {k: ref[k] for k in SHARED_KEYS}, weights


def _check_seeds(runs, arm, cfg):
    if len(runs) < cfg.min_seeds:
        raise ValueError("%s has %d scored seed(s), the gate needs >= %d"
                         % (arm, len(runs), cfg.min_seeds))


def _tied_arms(arms, metric):
    """A warning for every arm whose seeds all scored exactly the same. Distinct
    weights cannot be a repeated run, so this is either a collapsed model (all
    0, which must still be decided: REJECT, blame recipe) or a seed that did not
    reach training; raising would block the chain in both cases."""
    out = []
    for arm, values in arms:
        if len(values) >= 2 and len(set(values)) == 1:
            out.append("%s: all %d seeds scored exactly %r on %s"
                       % (arm, len(values), values[0], metric))
    return out


def _ap(score, name, run):
    return _finite(score["per_class"][name], "%s per_class[%s]" % (run, name))


# ---------------------------------------------------------------------- guards
def _regression_guard(inc_v, cand_mean, null_sd, cfg):
    threshold = inc_v - cfg.regression_sd_mult * null_sd
    return {"passed": _ge(cand_mean, threshold), "cand_mean": cand_mean, "inc": inc_v,
            "null_sd": null_sd, "threshold": threshold}


def _species_guard(n_gt, a_runs, b_runs, cfg, a="cand", b="null"):
    """Every one of the 12 species: mean(a) - mean(b) >= -max(0.03, 3 sd(b))."""
    few = {s: n_gt[s] for s in SPECIES if n_gt[s] < cfg.min_species_gt}
    if few:
        raise ValueError("the exam has fewer than %d boxes of %s; dev guarantees that many of "
                         "every species, so the split or its score is broken and the species "
                         "guard cannot run" % (cfg.min_species_gt, few))
    per_species, failed = {}, []
    for s in SPECIES:
        av = [_ap(x, s, "%s[%d]" % (a, i)) for i, x in enumerate(a_runs)]
        bv = [_ap(x, s, "%s[%d]" % (b, i)) for i, x in enumerate(b_runs)]
        delta = _mean(av) - _mean(bv)
        sd = _sd(bv)
        threshold = max(cfg.species_min_drop, cfg.species_sd_mult * sd)
        ok = _ge(delta, -threshold)
        per_species[s] = {"delta": delta, "sd": sd, "threshold": threshold,
                          "n_gt": n_gt[s], "passed": ok}
        if not ok:
            failed.append(s)
    return {"passed": not failed, "arms": [a, b], "min_species_gt": cfg.min_species_gt,
            "checked": list(SPECIES), "failed": failed, "per_species": per_species}


def _flips_guard(inc, cands, nulls, cfg):
    ref = inc["image_correct"]
    cf = [negative_flips(ref, x["image_correct"]) for x in cands]
    nf = [negative_flips(ref, x["image_correct"]) for x in nulls]
    if cfg.flips_mode == FLIPS_NET:
        return _net_flips_guard(ref, cands, nulls, cf, nf, cfg)
    return _negative_flips_guard(cf, nf, len(ref), ref.count("1"), cfg)


def _negative_flips_guard(cf, nf, n_images, inc_correct, cfg):
    """Protocol v1 on per-run negative flip counts. The dict is the ledger's v1
    record, key for key; v1_counterfactual() rebuilds it from a net record."""
    sd = _sd(nf)
    excess = _mean(cf) - _mean(nf)
    threshold = cfg.flips_sd_mult * sd + cfg.flips_slack_images
    return {"passed": _le(excess, threshold), "cand_flips": cf, "null_flips": nf,
            "cand_mean": _mean(cf), "null_mean": _mean(nf), "null_sd": sd,
            "excess": excess, "threshold": threshold,
            "n_images": n_images, "inc_correct": inc_correct}


def _net_flips_guard(ref, cands, nulls, cf, nf, cfg):
    """Protocol v2: net flips (negative - positive) per run, the v1 threshold
    form on them. Every count is recorded; the tested statistic is the net one
    (cand_mean, null_mean, null_sd, excess, threshold). There is no
    cand_flips / null_flips key, which in a v1 record means negative flips.
    A run's net count is inc_correct - run_correct (module docstring), so
    excess = mean(null correct) - mean(cand correct) and null_sd =
    sd(null correct)."""
    cp = [positive_flips(ref, x["image_correct"]) for x in cands]
    npos = [positive_flips(ref, x["image_correct"]) for x in nulls]
    cn = [a - b for a, b in zip(cf, cp)]
    nn = [a - b for a, b in zip(nf, npos)]
    sd = _sd(nn)
    excess = _mean(cn) - _mean(nn)
    threshold = cfg.flips_sd_mult * sd + cfg.flips_slack_images
    return {"passed": _le(excess, threshold), "mode": FLIPS_NET,
            "cand_negative_flips": cf, "cand_positive_flips": cp, "cand_net_flips": cn,
            "null_negative_flips": nf, "null_positive_flips": npos, "null_net_flips": nn,
            "cand_mean": _mean(cn), "null_mean": _mean(nn), "null_sd": sd,
            "excess": excess, "threshold": threshold,
            "n_images": len(ref), "inc_correct": ref.count("1")}


# ------------------------------------------------------------------- decision
def decide(inc, cands, nulls, cfg=GateConfig()):
    """Apply the protocol's gate to one step; see the module docstring."""
    cands, nulls = list(cands), list(nulls)
    _check_seeds(cands, "cand", cfg)
    _check_seeds(nulls, "null", cfg)
    stamps, weights = _validate([("inc", inc)]
                                + [("cand[%d]" % i, s) for i, s in enumerate(cands)]
                                + [("null[%d]" % i, s) for i, s in enumerate(nulls)], cfg)
    m = cfg.metric
    inc_v = float(inc[m])
    cv = [float(s[m]) for s in cands]
    nv = [float(s[m]) for s in nulls]
    cand_mean, null_mean = _mean(cv), _mean(nv)
    cand_sd, null_sd = _sd(cv), _sd(nv)

    p_data = p_greater(cv, nv)
    p_recipe = p_greater(nv, [inc_v])

    guards = {
        "regression": _regression_guard(inc_v, cand_mean, null_sd, cfg),
        "species": _species_guard(inc["n_gt"], cands, nulls, cfg),
        "flips": _flips_guard(inc, cands, nulls, cfg),
    }
    verdict = _verdict(all(g["passed"] for g in guards.values()), p_data, cfg)

    warnings = _tied_arms((("cand", cv), ("null", nv)), m)
    attribution = _attribute(inc, cands, nulls, cand_mean, null_mean, null_sd, p_recipe,
                             verdict, guards, cfg)
    reason = _reason(verdict, p_data, p_recipe, inc_v, cand_mean, cand_sd, null_mean,
                     null_sd, guards, attribution, warnings, cfg)
    weight_record = {"inc": weights["inc"],
                     "cand": [weights["cand[%d]" % i] for i in range(len(cands))],
                     "null": [weights["null[%d]" % i] for i in range(len(nulls))]}
    return Decision(verdict=verdict, reason=reason, metric=m, p_data=p_data, p_recipe=p_recipe,
                    inc=inc_v, cand_values=cv, null_values=nv, cand_mean=cand_mean,
                    cand_sd=cand_sd, null_mean=null_mean, null_sd=null_sd, guards=guards,
                    attribution=attribution, stamps=stamps, weights=weight_record,
                    warnings=warnings, config=config_record(cfg))


def _verdict(guards_pass, p_data, cfg):
    """REJECT on a failed guard, else ACCEPT / REJECT / HOLD by P_data."""
    if not guards_pass:
        return REJECT
    if _ge(p_data, cfg.p_accept):
        return ACCEPT
    if _le(p_data, cfg.p_reject):
        return REJECT
    return HOLD


def v1_counterfactual(decision):
    """The verdict protocol v1 gives the very runs a decision was taken on, from
    its ledger record alone (a Decision or its to_dict()).

    A net-mode decision records every run's negative flips, so v1's flips guard
    is rebuilt from them (_negative_flips_guard, the thresholds of the recorded
    config); P_data and the regression and species guards do not depend on the
    flips mode and are taken as recorded. A v1 decision returns its own verdict
    and flips guard. Returns {"verdict", "flips"}.

    This is a per-step counterfactual on the same incumbent, cands and nulls,
    not a counterfactual chain: after a step the two rules decide differently,
    the chain's incumbent is the net rule's."""
    d = decision.to_dict() if isinstance(decision, Decision) else decision
    g = d["guards"]["flips"]
    if g.get("mode") != FLIPS_NET:
        return {"verdict": d["verdict"], "flips": g}
    cfg = GateConfig(**d["config"])
    flips =_negative_flips_guard(list(g["cand_negative_flips"]), list(g["null_negative_flips"]),
                                  g["n_images"], g["inc_correct"], cfg)
    guards_pass = (d["guards"]["regression"]["passed"] and d["guards"]["species"]["passed"]
                   and flips["passed"])
    return {"verdict": _verdict(guards_pass, d["p_data"], cfg), "flips": flips}


def _attribute(inc, cands, nulls, cand_mean, null_mean, null_sd, p_recipe, verdict,
               guards, cfg):
    """Recorded for every step; the driver acts on it at REJECT.

    A score "drops" when cand - null < -attr_drop_sd_mult * sd(null), the same
    noise yardstick as the regression guard. 12-class down with the agnostic
    score holding means the boxes are found but named wrongly (labels); both
    down means they are not found (domain or localisation)."""
    agn = "agnostic_" + cfg.metric
    ac = [float(s[agn]) for s in cands]
    an = [float(s[agn]) for s in nulls]
    map_delta = cand_mean - null_mean
    agnostic_delta = _mean(ac) - _mean(an)
    map_drop_at = cfg.attr_drop_sd_mult * null_sd
    agn_null_sd = _sd(an)
    agn_drop_at = cfg.attr_drop_sd_mult * agn_null_sd
    map_drops = map_delta < -map_drop_at
    agn_drops = agnostic_delta < -agn_drop_at
    if not map_drops:
        class_vs_loc = NONE
    else:
        class_vs_loc = DOMAIN if agn_drops else LABELS

    # Every class the exam has boxes of, most negative first ("the species that
    # moved"), as [name, delta] pairs.
    deltas = []
    for name in CLASS_NAMES:
        if inc["n_gt"][name] > 0:
            c = [_ap(x, name, "cand[%d]" % i) for i, x in enumerate(cands)]
            n = [_ap(x, name, "null[%d]" % i) for i, x in enumerate(nulls)]
            deltas.append((_mean(c) - _mean(n), name))
    per_species_delta = [[name, d] for d, name in sorted(deltas)]

    recipe_flag = _le(p_recipe, cfg.p_recipe_flag)
    return {
        "recipe_flag": recipe_flag,
        # At REJECT: a recipe that degrades W regardless of data is the recipe's
        # fault, not D_k's (the 2026-09-10 warm-start finding).
        "blame": ("recipe" if recipe_flag else "data") if verdict == REJECT else None,
        "class_vs_loc": class_vs_loc,
        "map_delta": map_delta,
        "agnostic_delta": agnostic_delta,
        "class_only_delta": map_delta - agnostic_delta,
        "map_drop_threshold": map_drop_at,
        "agnostic_drop_threshold": agn_drop_at,
        "agnostic_null_sd": agn_null_sd,
        "per_species_delta": per_species_delta,
        "species_failed": list(guards["species"]["failed"]),
    }


def _reason(verdict, p_data, p_recipe, inc_v, cm, cs, nm, ns, guards, attr, warnings, cfg):
    head = ("%s: P_data=%.3f (cand %.4f+-%.4f vs null %.4f+-%.4f, inc %.4f)"
            % (verdict, p_data, cm, cs, nm, ns, inc_v))
    bad = []
    g = guards["regression"]
    if not g["passed"]:
        bad.append("regression (cand %.4f < %.4f)" % (g["cand_mean"], g["threshold"]))
    g = guards["species"]
    if not g["passed"]:
        bad.append("species (%s)" % ", ".join(
            "%s %+.3f < -%.3f" % (s, g["per_species"][s]["delta"], g["per_species"][s]["threshold"])
            for s in g["failed"]))
    g = guards["flips"]
    if not g["passed"]:
        bad.append("flips (%s%+.1f > %.1f images)" % ("net " if g.get("mode") == FLIPS_NET else "",
                                                      g["excess"], g["threshold"]))
    parts = [head]
    if bad:
        parts.append("guard failed: " + "; ".join(bad))
    elif verdict == REJECT:
        parts.append("P_data <= %.2f, guards pass" % cfg.p_reject)
    elif verdict == HOLD:
        parts.append("effect not detectable, guards pass")
    else:
        parts.append("guards pass")
    parts.append("P_recipe=%.3f%s" % (p_recipe, " (recipe flag)" if attr["recipe_flag"] else ""))
    parts.append("class_vs_loc=%s" % attr["class_vs_loc"])
    if warnings:
        parts.append("WARNING " + "; ".join(warnings))
    return "; ".join(parts)


# ----------------------------------------------------------------- soup, truth
def choose_soup(soup_score, cand_scores, cfg=GateConfig()):
    """After ACCEPT: the new incumbent is the uniform soup of the cand weights if
    the soup scores >= mean(cand) on dev, else cand[0]."""
    cand_scores = list(cand_scores)
    if not cand_scores:
        raise ValueError("choose_soup needs the cand scores")
    _validate([("soup", soup_score)]
              + [("cand[%d]" % i, s) for i, s in enumerate(cand_scores)], cfg)
    soup_v = float(soup_score[cfg.metric])
    return "soup" if _ge(soup_v, _mean([float(s[cfg.metric]) for s in cand_scores])) else "cand0"


def truth_detail(with_scores, without_scores, cfg=GateConfig()):
    """The pilot's truth arm: the gate rule on cold union runs trained with and
    without D_k, "with" in the role of cand and "without" in the role of null.
    That is P = P(with > without) and the species guard between the two arms.
    The regression and flips guards compare against an incumbent W, which cold
    runs do not have, so they have no truth-arm counterpart.

    'hurts' when the species guard fails or P <= p_reject (the gate's REJECT),
    'helps' at P >= p_accept (ACCEPT), 'neutral' otherwise (HOLD)."""
    w, wo = list(with_scores), list(without_scores)
    _check_seeds(w, "with", cfg)
    _check_seeds(wo, "without", cfg)
    stamps, _ = _validate([("with[%d]" % i, s) for i, s in enumerate(w)]
                          + [("without[%d]" % i, s) for i, s in enumerate(wo)], cfg)
    wv = [float(s[cfg.metric]) for s in w]
    wov = [float(s[cfg.metric]) for s in wo]
    p = p_greater(wv, wov)
    species = _species_guard(w[0]["n_gt"], w, wo, cfg, a="with", b="without")
    if not species["passed"] or _le(p, cfg.p_reject):
        verdict = HURTS
    elif _ge(p, cfg.p_accept):
        verdict = HELPS
    else:
        verdict = NEUTRAL
    return {"verdict": verdict, "p": p, "metric": cfg.metric,
            "with_values": wv, "without_values": wov,
            "with_mean": _mean(wv), "with_sd": _sd(wv),
            "without_mean": _mean(wov), "without_sd": _sd(wov),
            "species": species, "stamps": stamps,
            "warnings": _tied_arms((("with", wv), ("without", wov)), cfg.metric)}


def truth_decision(with_scores, without_scores, cfg=GateConfig()):
    """'helps', 'hurts' or 'neutral'; truth_detail() has the numbers."""
    return truth_detail(with_scores, without_scores, cfg)["verdict"]
