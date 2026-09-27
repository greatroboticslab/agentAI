#!/usr/bin/env python3
"""The INC gate decides each increment by the protocol's rule and says why.

docs/INCREMENTAL_PROTOCOL.md ("Gate") fixes the rule: P_data = P(cand > null)
over every seed pair with ties at one half, ACCEPT at >= 0.75, HOLD strictly
between 0.25 and 0.75, REJECT otherwise, and three guards (regression vs the
incumbent, per-species drop, negative flips) that can turn any verdict into
REJECT. The ledger trusts these decisions without a person re-checking them,
so each rule is pinned here on hand-built score dicts: every verdict path, each
guard failing on its own, the recipe flag, the labels vs domain/localisation
attribution, ties, and a lossless JSON round trip of the decision the ledger
stores (per-species order included, under sort_keys).

The guards are pinned to the written thresholds with no sd floor: cases where
a floor would have flipped the verdict are checked against the written rule.

The gate must also refuse to decide on inputs it cannot vouch for, because a
silent pass there looks exactly like a real ACCEPT in the ledger: n_gt keyed by
id, empty or misspelled (the species guard would check nothing); a species
below dev's guaranteed 30 boxes; scores from another scorer.py, manifest, exam
or image order; test instead of dev; an empty image_correct (the flips guard
would pass on zero images); one weights file passed as several seeds. The truth
arm applies the same species guard as the gate, so a step the gate rejects on a
species is not counted as a truth "helps".

GateConfig.flips_mode (docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net
flips)"): 'negative' (v1, the default) must decide byte for byte as the gate
did before the option existed: a battery of 11 decisions covering every
verdict and guard path, positive flips in cand and null included, hashes to
the digest the pre-change gate.py gave (pinned below), for GateConfig() and
for an explicit flips_mode='negative'. 'net' counts negative - positive flips
per run with the same threshold form: a clearly better model with many
positive flips passes where v1 rejects it; a model that breaks more than it
fixes fails at P_data = 1; the null's own positive flips count (so net is not
simply more lenient); the sd-0 edge is +3 images; the guard records every
count and the config says 'net'; P_data, the other guards and attribution are
the v1 decision's; choose_soup and the truth arm do not depend on the mode;
an unknown mode is refused. v1_counterfactual rebuilds, from a net decision's
record alone, exactly the verdict and flips guard v1 gives the same runs; and
each run's net count is inc_correct - run_correct (the identity the protocol
doc states).

Run:  python3 tests/test_inc_gate.py
"""
import dataclasses
import hashlib
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402

FAILURES = []

# A dev-like exam: every species has >= 30 boxes (dev's guarantee), OtherPlant none.
N_GT = {"Waterhemp": 120, "MorningGlory": 80, "Purslane": 60, "SpottedSpurge": 40,
        "Carpetweed": 30, "Ragweed": 31, "Eclipta": 33, "PricklySida": 32,
        "PalmerAmaranth": 90, "Sicklepod": 45, "Goosegrass": 35, "CutleafGroundcherry": 50,
        "OtherPlant": 0}
KEY = "a" * 64
N_IMG = 40
INC_BITS = "1" * 30 + "0" * 10
_WEIGHTS = itertools.count()


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc, contains=None):
    try:
        fn()
    except exc as e:
        if contains and contains not in str(e):
            print("       raised %s without %r: %s" % (type(e).__name__, contains, e))
            return False
        return True
    except Exception as e:  # the wrong error is a failure too
        print("       raised %s: %s" % (type(e).__name__, e))
        return False
    return False


def bits(neg=0, pos=0):
    """inc's correctness string with `neg` of its correct images now wrong and
    `pos` of its wrong images now right."""
    b = list(INC_BITS)
    for i in range(neg):
        b[i] = "0"
    for i in range(pos):
        b[30 + i] = "1"
    return "".join(b)


def score(m, agn=None, species=None, neg=0, pos=0, key=KEY, n_gt=None, **stamps):
    """A score.json-shaped dict: every class AP equals m unless overridden; each
    call gets its own weights_sha256, as a separate run would."""
    n_gt = dict(N_GT if n_gt is None else n_gt)
    per_class = {s: m for s, n in n_gt.items() if n > 0}
    per_class.update(species or {})
    s = {"exam": "dev", "scorer_sha256": "5" * 64, "manifest_sha256": "6" * 64,
         "key_order_sha256": key, "weights_sha256": "%064x" % next(_WEIGHTS),
         "n_images": N_IMG,
         "map50_95": m, "map50": min(1.0, m + 0.2), "per_class": per_class, "n_gt": n_gt,
         "agnostic_map50_95": m + 0.1 if agn is None else agn,
         "agnostic_map50": min(1.0, m + 0.3),
         "image_correct": bits(neg, pos), "production": True}
    s.update(stamps)
    return s


def runs(values, **kw):
    return [score(v, **kw) for v in values]


def delta_names(d):
    return [name for name, _ in d.attribution["per_species_delta"]]


def names_first(d):
    return d.attribution["per_species_delta"][0][0]


def test_production_only():
    """Scores the scorer took in test mode never decide a real step."""
    inc = score(0.60)
    ok = lambda: G.decide(inc, runs([0.62, 0.63, 0.61]), runs([0.60, 0.59, 0.60]))
    check("production scores decide", ok().verdict in ("ACCEPT", "HOLD", "REJECT"))
    for label, kw in (("production=false", {"production": False}),
                      ("no production field", {"production": None}),
                      ("a TEST- scorer stamp", {"scorer_sha256": "TEST-" + "5" * 59})):
        bad = score(0.60, **kw)
        if kw.get("production", True) is None:
            bad.pop("production")
        try:
            G.decide(bad, runs([0.62, 0.63, 0.61]), runs([0.60, 0.59, 0.60]))
            check("a score with %s is refused" % label, False)
        except ValueError as e:
            check("a score with %s is refused" % label, "production" in str(e) or "scorer" in str(e), str(e)[:120])
    tests = [score(0.60, production=False, scorer_sha256="TEST-" + "5" * 59) for _ in range(7)]
    d = G.decide(tests[0], tests[1:4], tests[4:7], G.GateConfig(require_production=False))
    check("an explicit require_production=False (tests only) lets test scores through", d.verdict in ("ACCEPT", "HOLD", "REJECT"))


# sha256[:16] of the v1_battery() decisions' to_dict() (sort_keys JSON), computed
# with inc/gate.py as it was before GateConfig.flips_mode existed (2026-09-27).
V1_BATTERY_DIGEST = "057503ae0bf53be2"
V1_FLIPS_KEYS = {"passed", "cand_flips", "null_flips", "cand_mean", "null_mean", "null_sd", "excess",
                 "threshold", "n_images", "inc_correct"}


def v1_battery():
    """(label, inc, cands, nulls) cases covering every v1 verdict and guard path,
    positive flips included, with fixed weights stamps so the decisions do not
    depend on how many scores were made before."""
    n = iter(range(1000, 2000))

    def s(m, **kw):
        return score(m, weights_sha256="%064x" % next(n), **kw)

    def arm(values, **kw):
        return [s(v, **kw) for v in values]
    return [
        ("accept", s(0.60), arm([0.62, 0.625, 0.63]), arm([0.60, 0.605, 0.61])),
        ("flips alone", s(0.60), arm([0.62, 0.625, 0.63], neg=10),
         [s(0.60, neg=1), s(0.605, neg=2), s(0.61, neg=1)]),
        ("better, many positive flips", s(0.60), arm([0.62, 0.625, 0.63], neg=10, pos=10),
         [s(0.60, neg=1), s(0.605, neg=2), s(0.61, neg=1)]),
        ("positive flips do not offset", s(0.60), arm([0.62, 0.625, 0.63], neg=3, pos=10),
         arm([0.60, 0.605, 0.61])),
        ("null with positive flips", s(0.60), arm([0.62, 0.625, 0.63], neg=4, pos=2),
         [s(0.60, neg=1, pos=5), s(0.605, neg=2, pos=4), s(0.61, neg=1, pos=6)]),
        ("noisy null", s(0.60), arm([0.62, 0.625, 0.63], neg=20),
         [s(0.60, neg=0), s(0.605, neg=6), s(0.61, neg=12)]),
        ("sd 0 edge", s(0.60), arm([0.62, 0.625, 0.63], neg=5), arm([0.60, 0.605, 0.61], neg=2)),
        ("regression", s(0.70), arm([0.62, 0.625, 0.63]), arm([0.60, 0.605, 0.61])),
        ("species", s(0.60), [s(v, species={"Carpetweed": v - 0.075}) for v in (0.62, 0.625, 0.63)],
         arm([0.60, 0.605, 0.61])),
        ("hold", s(0.60), arm([0.60, 0.61, 0.62]), arm([0.605, 0.615, 0.60])),
        ("reject by P", s(0.60), arm([0.57, 0.575, 0.58], agn=0.70), arm([0.60, 0.605, 0.61], agn=0.70)),
    ]


def battery_digest(cfg=None):
    """sha256[:16] of every battery decision's to_dict() (sort_keys JSON), in order."""
    ds = [(G.decide(i, c, nl) if cfg is None else G.decide(i, c, nl, cfg)).to_dict()
          for _, i, c, nl in v1_battery()]
    return hashlib.sha256(json.dumps(ds, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def test_flips_modes():
    """GateConfig.flips_mode: v1 unchanged, v2 (net flips) as written."""
    check("the default flips mode is v1's 'negative'", G.GateConfig().flips_mode == "negative"
          and G.DEFAULT_FLIPS_MODE == "negative" and G.FLIPS_MODES == ("negative", "net"))
    for bad in ("both", "Net", "", None):
        check("flips_mode %r is refused" % (bad,),
              raises(lambda: G.GateConfig(flips_mode=bad), ValueError, "flips_mode"))
    got = battery_digest()
    check("negative mode (GateConfig()): the 11 battery decisions are byte for byte the pre-change gate's "
          "(pinned digest)", got == V1_BATTERY_DIGEST, got)
    got = battery_digest(G.GateConfig(flips_mode="negative"))
    check("... and so with an explicit flips_mode='negative'", got == V1_BATTERY_DIGEST, got)
    v1 = [G.decide(i, c, nl) for _, i, c, nl in v1_battery()]
    check("a v1 decision records no flips_mode (absent = 'negative') and the v1 flips-guard keys only",
          all("flips_mode" not in d.config and set(d.guards["flips"]) == V1_FLIPS_KEYS for d in v1)
          and set(v1[0].config) == {f.name for f in dataclasses.fields(G.GateConfig)} - {"flips_mode"})
    check("positive_flips counts only 0 -> 1", G.positive_flips("1100", "1010") == 1
          and G.positive_flips("0000", "1111") == 4 and G.positive_flips("1111", "0000") == 0)
    check("positive_flips refuses strings of different lengths",
          raises(lambda: G.positive_flips("10", "101"), ValueError, "lengths differ"))

    net = G.GateConfig(flips_mode="net")
    null = [score(0.60, neg=1), score(0.605, neg=2), score(0.61, neg=1)]
    cand = runs([0.62, 0.625, 0.63], neg=10, pos=10)
    d1 = G.decide(score(0.60), cand, null)
    d2 = G.decide(score(0.60), cand, null, net)
    g = d2.guards["flips"]
    check("v1 rejects a clearly better model (P_data 1) on its 10 negative flips alone",
          d1.verdict == "REJECT" and d1.p_data == 1.0 and not d1.guards["flips"]["passed"]
          and d1.guards["regression"]["passed"] and d1.guards["species"]["passed"], d1.reason)
    check("net: the same model fixes as many images as it breaks (net 0 vs null 1, 2, 1) and is ACCEPTed",
          d2.verdict == "ACCEPT" and g["passed"] and g["cand_net_flips"] == [0, 0, 0]
          and g["null_net_flips"] == [1, 2, 1] and abs(g["excess"] + 4 / 3) < 1e-12, d2.reason)
    check("net: the guard records every count per run and the tested net statistic",
          g["mode"] == "net" and g["cand_negative_flips"] == [10, 10, 10] and g["cand_positive_flips"] == [10, 10, 10]
          and g["null_negative_flips"] == [1, 2, 1] and g["null_positive_flips"] == [0, 0, 0]
          and abs(g["null_sd"] - 0.5773502691896258) < 1e-12
          and abs(g["threshold"] - (2 * 0.5773502691896258 + 3)) < 1e-12
          and abs(g["cand_mean"]) < 1e-12 and abs(g["null_mean"] - 4 / 3) < 1e-12
          and g["n_images"] == N_IMG and g["inc_correct"] == 30
          and "cand_flips" not in g and "null_flips" not in g, g)
    check("net: the decision's config says flips_mode 'net' and holds every GateConfig field",
          d2.config["flips_mode"] == "net"
          and set(d2.config) == {f.name for f in dataclasses.fields(G.GateConfig)}, d2.config)
    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=6, pos=10), null, net)
    check("net: more fixed than broken (net -4 per run) passes", d.verdict == "ACCEPT"
          and d.guards["flips"]["cand_net_flips"] == [-4, -4, -4], d.reason)

    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=10, pos=2), null, net)
    check("net: a model that breaks more than it fixes (net 8 vs null 1, 2, 1) is REJECTed although "
          "P_data = 1, and the reason names the net flips",
          d.verdict == "REJECT" and d.p_data == 1.0 and not d.guards["flips"]["passed"]
          and d.guards["flips"]["cand_net_flips"] == [8, 8, 8] and "flips (net +6.7 > 4.2 images)" in d.reason,
          d.reason)

    null_fix = runs([0.60, 0.605, 0.61], neg=1, pos=5)            # null net -4 each, sd 0
    cand = runs([0.62, 0.625, 0.63], neg=3, pos=3)                # cand net 0
    d1 = G.decide(score(0.60), cand, null_fix)
    d2 = G.decide(score(0.60), cand, null_fix, net)
    check("net counts the null's own positive flips: cand net 0 vs null net -4 fails (excess 4 > 3), where "
          "v1 (3 vs 1 negative flips) passes",
          d1.verdict == "ACCEPT" and d2.verdict == "REJECT" and d2.guards["flips"]["excess"] == 4.0
          and d2.guards["flips"]["threshold"] == 3.0, (d1.reason, d2.reason))
    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=4, pos=5), null_fix, net)
    check("net, sd(null net) 0: an excess of exactly 3 images passes", d.guards["flips"]["excess"] == 3.0
          and d.guards["flips"]["passed"], d.guards["flips"])

    same = True
    for (label, i, c, nl), a in zip(v1_battery(), v1):
        b = G.decide(i, c, nl, net)
        attr = lambda d: {k: v for k, v in d.attribution.items() if k != "blame"}  # noqa: E731
        same = same and (a.p_data, a.p_recipe, a.guards["regression"], a.guards["species"], attr(a),
                         a.stamps, a.cand_values, a.null_values) == (
                             b.p_data, b.p_recipe, b.guards["regression"], b.guards["species"], attr(b),
                             b.stamps, b.cand_values, b.null_values)
    check("net changes only the flips guard: P_data, P_recipe, regression, species and attribution (but "
          "blame, which exists only at REJECT) are the v1 decision's on every battery case", same)
    verdicts = [G.decide(i, c, nl, net).verdict for _, i, c, nl in v1_battery()]
    check("net on the battery: the clearly better model with many positive flips is the one verdict that "
          "changes (REJECT -> ACCEPT)",
          [a.verdict for a in v1] != verdicts
          and [k for k, (a, b) in enumerate(zip(v1, verdicts)) if a.verdict != b] == [2]
          and verdicts[2] == "ACCEPT", verdicts)

    ok_cf, ok_id = True, True
    for (label, i, c, nl), a in zip(v1_battery(), v1):
        b = G.decide(i, c, nl, net)
        for rec in (b, json.loads(json.dumps(b.to_dict(), sort_keys=True))):
            ok_cf = ok_cf and G.v1_counterfactual(rec) == {"verdict": a.verdict, "flips": a.guards["flips"]}
        g = b.guards["flips"]
        correct = lambda runs_: [x["image_correct"].count("1") for x in runs_]  # noqa: E731
        ok_id = ok_id and g["cand_net_flips"] == [g["inc_correct"] - k for k in correct(c)] \
            and g["null_net_flips"] == [g["inc_correct"] - k for k in correct(nl)] \
            and abs(g["excess"] - (sum(correct(nl)) / len(nl) - sum(correct(c)) / len(c))) < 1e-9
    check("v1_counterfactual: on every battery case, v1's verdict and flips guard rebuilt from the net decision "
          "(object or its JSON) are exactly what decide() gives in v1 mode on the same runs", ok_cf)
    check("v1_counterfactual: a v1 decision returns its own verdict and flips guard",
          all(G.v1_counterfactual(a) == {"verdict": a.verdict, "flips": a.guards["flips"]} for a in v1))
    check("net identity: each run's net flips = inc_correct - run_correct, so the tested excess is "
          "mean(null correct) - mean(cand correct) (docs, Protocol v2, Principle)", ok_id)

    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=10, pos=2), null, net)
    text = json.dumps(d.to_dict(), allow_nan=False, sort_keys=True)
    back = G.Decision.from_dict(json.loads(text))
    check("a net decision survives a strict sort_keys JSON round trip unchanged",
          back == d and back.to_dict() == d.to_dict())

    cands = runs([0.62, 0.625, 0.63], neg=10, pos=10)
    check("choose_soup does not depend on the flips mode",
          G.choose_soup(score(0.626, neg=10), cands, net) == G.choose_soup(score(0.626, neg=10), cands) == "soup"
          and G.choose_soup(score(0.624), cands, net) == G.choose_soup(score(0.624), cands) == "cand0")
    w, wo = runs([0.62, 0.63, 0.603], neg=10), runs([0.60, 0.605, 0.61])
    check("the truth arm does not depend on the flips mode (it has no flips guard)",
          G.truth_detail(w, wo, net) == G.truth_detail(w, wo))


def main():
    # --- p_greater ------------------------------------------------------------
    check("all a above all b gives 1", G.p_greater([3, 4, 5], [0, 1, 2]) == 1.0)
    check("all a below all b gives 0", G.p_greater([0, 1, 2], [3, 4, 5]) == 0.0)
    check("identical lists give 0.5 (every pair a tie)",
          G.p_greater([0.6, 0.6, 0.6], [0.6, 0.6, 0.6]) == 0.5)
    check("a tie counts one half: [1, 2] vs [1] = (0.5 + 1) / 2",
          G.p_greater([1, 2], [1]) == 0.75)
    check("3x3 with one tie: (0.5 + 2 + 3) / 9",
          abs(G.p_greater([0.5, 0.6, 0.7], [0.5, 0.55, 0.65]) - 5.5 / 9) < 1e-12,
          G.p_greater([0.5, 0.6, 0.7], [0.5, 0.55, 0.65]))
    check("an empty list raises", raises(lambda: G.p_greater([], [1]), ValueError))
    check("a NaN raises", raises(lambda: G.p_greater([float("nan")], [1]), ValueError))

    # --- verdicts ---------------------------------------------------------------
    d = G.decide(score(0.60), runs([0.62, 0.63, 0.625]), runs([0.60, 0.605, 0.61]))
    check("ACCEPT: every cand above every null, guards pass",
          d.verdict == "ACCEPT" and d.p_data == 1.0, d.reason)
    check("ACCEPT records every guard as passed",
          all(g["passed"] for g in d.guards.values()) and set(d.guards) ==
          {"regression", "species", "flips"})
    check("ACCEPT reason is one line naming the verdict",
          d.reason.startswith("ACCEPT:") and "\n" not in d.reason, d.reason)
    check("means and sds are recorded (sd is ddof=1: [0.60, 0.605, 0.61] -> 0.005)",
          abs(d.null_mean - 0.605) < 1e-12 and abs(d.null_sd - 0.005) < 1e-12
          and abs(d.cand_mean - 0.625) < 1e-12, (d.null_mean, d.null_sd))
    check("no blame outside REJECT", d.attribution["blame"] is None)
    check("the decision records the shared stamps it was taken on",
          d.stamps == {"exam": "dev", "scorer_sha256": "5" * 64, "manifest_sha256": "6" * 64,
                       "key_order_sha256": KEY, "n_images": N_IMG}, d.stamps)
    check("...and every input's weights", len(d.weights["cand"]) == 3
          and len(d.weights["null"]) == 3 and len(set(d.weights["cand"] + d.weights["null"]
                                                       + [d.weights["inc"]])) == 7, d.weights)
    check("no warning when the seeds differ", d.warnings == [], d.warnings)

    # P = 5.5 / 9 = 0.61
    d = G.decide(score(0.60), runs([0.60, 0.61, 0.62]), runs([0.605, 0.615, 0.60]))
    check("HOLD: 0.25 < P_data < 0.75 with guards passing",
          d.verdict == "HOLD" and abs(d.p_data - 5.5 / 9) < 1e-12, d.reason)

    d = G.decide(score(0.60), runs([0.60, 0.60, 0.60]), runs([0.60, 0.60, 0.60]))
    check("identical cand and null: P_data 0.5 by ties, HOLD", d.verdict == "HOLD"
          and d.p_data == 0.5, d.reason)

    d = G.decide(score(0.60), runs([0.595, 0.598, 0.60]), runs([0.605, 0.61, 0.615]))
    check("REJECT by data: P_data <= 0.25 although every guard passes",
          d.verdict == "REJECT" and d.p_data == 0.0
          and all(g["passed"] for g in d.guards.values()), d.reason)
    check("REJECT by data blames the data when the recipe is fine",
          d.attribution["blame"] == "data" and not d.attribution["recipe_flag"])

    # the 3x3 grid's edges: 7/9 accepts, 6.5/9 (a tie) holds, 3/9 holds, 2/9 rejects
    d = G.decide(score(0.50), runs([0.62, 0.63, 0.615]), runs([0.61, 0.605, 0.625]))
    check("P_data = 7/9 accepts", abs(d.p_data - 7 / 9) < 1e-12 and d.verdict == "ACCEPT",
          (d.p_data, d.verdict))
    d = G.decide(score(0.50), runs([0.62, 0.63, 0.61]), runs([0.61, 0.605, 0.625]))
    check("P_data = 6.5/9 (one tie) holds",
          abs(d.p_data - 6.5 / 9) < 1e-12 and d.verdict == "HOLD", (d.p_data, d.verdict))
    d = G.decide(score(0.50), runs([0.62, 0.63, 0.607]), runs([0.61, 0.605, 0.625]))
    check("P_data = 6/9 holds", abs(d.p_data - 6 / 9) < 1e-12 and d.verdict == "HOLD",
          (d.p_data, d.verdict))
    d = G.decide(score(0.50), runs([0.600, 0.601, 0.630]), runs([0.61, 0.605, 0.625]))
    check("P_data = 3/9 holds",
          abs(d.p_data - 3 / 9) < 1e-12 and d.verdict == "HOLD", (d.p_data, d.verdict))
    d = G.decide(score(0.50), runs([0.600, 0.601, 0.611]), runs([0.61, 0.605, 0.625]))
    check("P_data = 2/9 rejects", abs(d.p_data - 2 / 9) < 1e-12 and d.verdict == "REJECT",
          (d.p_data, d.verdict))

    # --- regression guard -------------------------------------------------------
    # cand beats null in every pair, but both sit far below the incumbent.
    d = G.decide(score(0.70), runs([0.63, 0.64, 0.65]), runs([0.60, 0.61, 0.62]))
    g = d.guards["regression"]
    check("REJECT by the regression guard although P_data = 1",
          d.verdict == "REJECT" and d.p_data == 1.0 and not g["passed"], d.reason)
    check("regression threshold is inc - 2 sd(null) = 0.70 - 0.02",
          abs(g["threshold"] - 0.68) < 1e-12 and abs(g["null_sd"] - 0.01) < 1e-12, g)
    check("the reason names the failed guard", "regression" in d.reason, d.reason)
    check("null << inc raises the recipe flag and puts the blame on the recipe",
          d.attribution["recipe_flag"] and d.p_recipe == 0.0
          and d.attribution["blame"] == "recipe", d.attribution)

    d = G.decide(score(0.70), runs([0.69, 0.70, 0.71]), runs([0.705, 0.71, 0.715]))
    check("null above inc: no recipe flag", not d.attribution["recipe_flag"]
          and d.p_recipe == 1.0, (d.p_recipe, d.attribution["recipe_flag"]))

    # null sd 0.001: the written threshold is 0.60 - 0.002 = 0.598, and cand mean
    # 0.5975 is below it (a 0.002 sd floor would have moved it to 0.596 and passed).
    d = G.decide(score(0.60), runs([0.5975, 0.597, 0.598]), runs([0.590, 0.591, 0.592]))
    g = d.guards["regression"]
    check("regression uses the measured sd with no floor: 0.5975 < 0.598 rejects",
          not g["passed"] and d.verdict == "REJECT" and abs(g["threshold"] - 0.598) < 1e-12
          and "null_sd_used" not in g, g)

    # --- species guard ----------------------------------------------------------
    base_null = runs([0.60, 0.605, 0.61])
    # null Carpetweed = null mAP (mean 0.605); cand Carpetweed mean 0.625 - 0.075 = 0.55
    cand = [score(v, species={"Carpetweed": v - 0.075}) for v in (0.62, 0.625, 0.63)]
    d = G.decide(score(0.60), cand, base_null)
    g = d.guards["species"]
    check("REJECT by the species guard: Carpetweed -0.055 < -0.03 while P_data = 1",
          d.verdict == "REJECT" and d.p_data == 1.0 and g["failed"] == ["Carpetweed"]
          and abs(g["per_species"]["Carpetweed"]["delta"] + 0.055) < 1e-12, d.reason)
    check("the species threshold is max(0.03, 3 sd): 0.03 here",
          abs(g["per_species"]["Carpetweed"]["threshold"] - 0.03) < 1e-12, g["per_species"])
    check("the failed species is in the reason and the attribution",
          "Carpetweed" in d.reason and d.attribution["species_failed"] == ["Carpetweed"])
    check("every one of the 12 species is checked, OtherPlant is not",
          g["checked"] == list(G.SPECIES) and len(g["checked"]) == 12
          and "OtherPlant" not in g["checked"], g["checked"])

    # distinct per-species deltas: most negative first, as [name, delta] pairs
    cand = [score(v, species={"Ragweed": v - 0.02, "PricklySida": v - 0.01,
                              "Goosegrass": v + 0.01}) for v in (0.62, 0.625, 0.63)]
    d = G.decide(score(0.60), cand, base_null)
    names = delta_names(d)
    deltas = [x for _, x in d.attribution["per_species_delta"]]
    check("per_species_delta runs most negative first",
          names[:2] == ["Ragweed", "PricklySida"] and names[-1] == "Goosegrass"
          and deltas == sorted(deltas), d.attribution["per_species_delta"])
    check("per_species_delta is a list of [name, delta] pairs",
          isinstance(d.attribution["per_species_delta"], list)
          and all(isinstance(p, list) and len(p) == 2 for p in d.attribution["per_species_delta"]))
    check("OtherPlant with no boxes in the exam is not reported", "OtherPlant" not in names)

    # OtherPlant with boxes: its delta is reported, but it is not a species and not guarded
    gt = dict(N_GT, OtherPlant=40)
    cand = [score(v, n_gt=gt, species={"OtherPlant": 0.1}) for v in (0.62, 0.625, 0.63)]
    d = G.decide(score(0.60, n_gt=gt), cand, runs([0.60, 0.605, 0.61], n_gt=gt))
    check("OtherPlant boxes are reported in the deltas but never guarded",
          d.verdict == "ACCEPT" and names_first(d) == "OtherPlant"
          and "OtherPlant" not in d.guards["species"]["per_species"], d.reason)

    # a noisy species: null sd 0.1 widens its allowance to 0.3, so a 0.1 drop passes
    noisy_null = [score(v, species={"Eclipta": e})
                  for v, e in ((0.60, 0.5), (0.605, 0.6), (0.61, 0.7))]
    cand = [score(v, species={"Eclipta": 0.5}) for v in (0.62, 0.625, 0.63)]
    d = G.decide(score(0.60), cand, noisy_null)
    e = d.guards["species"]["per_species"]["Eclipta"]
    check("a species' allowance widens to 3 sd(null) when that is above 0.03",
          d.verdict == "ACCEPT" and abs(e["threshold"] - 0.3) < 1e-9
          and abs(e["delta"] + 0.1) < 1e-9, e)

    missing = runs([0.62, 0.625, 0.63])
    del missing[1]["per_class"]["Sicklepod"]
    check("a species with boxes missing from per_class raises",
          raises(lambda: G.decide(score(0.60), missing, base_null), KeyError))

    # The guard must never pass by checking nothing.
    def id_keyed(v, **kw):
        s = score(v, **kw)
        s["n_gt"] = {str(i): n for i, n in enumerate(N_GT.values())}
        return s
    cand = [id_keyed(v, species={"Carpetweed": 0.0}) for v in (0.62, 0.625, 0.63)]
    check("n_gt keyed by class id raises (the guard would have checked nothing)",
          raises(lambda: G.decide(id_keyed(0.60), cand,
                                  [id_keyed(v) for v in (0.60, 0.605, 0.61)]),
                 ValueError, "n_gt"))
    check("an empty n_gt raises",
          raises(lambda: G.decide(score(0.60, n_gt={}), runs([0.62, 0.625, 0.63], n_gt={}),
                                  runs([0.60, 0.605, 0.61], n_gt={})), ValueError, "n_gt"))
    misspelled = dict(N_GT)
    misspelled["Morningglory"] = misspelled.pop("MorningGlory")
    check("a misspelled species in n_gt raises",
          raises(lambda: G.decide(score(0.60, n_gt=misspelled),
                                  runs([0.62, 0.625, 0.63], n_gt=misspelled),
                                  runs([0.60, 0.605, 0.61], n_gt=misspelled)),
                 ValueError, "Morningglory"))
    stray = runs([0.62, 0.625, 0.63])
    stray[0]["per_class"]["Morningglory"] = 0.0
    check("a per_class key that is not a class name raises",
          raises(lambda: G.decide(score(0.60), stray, base_null), ValueError, "per_class"))
    fractional = dict(N_GT, Waterhemp=120.5)
    check("a non-integer box count raises",
          raises(lambda: G.decide(score(0.60, n_gt=fractional),
                                  runs([0.62, 0.625, 0.63], n_gt=fractional),
                                  runs([0.60, 0.605, 0.61], n_gt=fractional)), ValueError))
    few = dict(N_GT, Ragweed=8)
    check("a species below dev's 30 boxes raises instead of going unguarded",
          raises(lambda: G.decide(score(0.60, n_gt=few), runs([0.62, 0.625, 0.63], n_gt=few),
                                  runs([0.60, 0.605, 0.61], n_gt=few)), ValueError, "Ragweed"))
    check("the default minimum is dev's guarantee (splits.DEV_MIN_BOXES = 30)",
          G.GateConfig().min_species_gt == 30)

    # --- flips guard ------------------------------------------------------------
    cand = runs([0.62, 0.625, 0.63], neg=10)
    null = [score(0.60, neg=1), score(0.605, neg=2), score(0.61, neg=1)]
    d = G.decide(score(0.60), cand, null)
    g = d.guards["flips"]
    check("REJECT by the flips guard although P_data = 1",
          d.verdict == "REJECT" and d.p_data == 1.0 and not g["passed"], d.reason)
    check("flip counts are per run: cand [10,10,10], null [1,2,1]",
          g["cand_flips"] == [10, 10, 10] and g["null_flips"] == [1, 2, 1], g)
    check("flip threshold is 2 sd(null flips) + 3 with the measured sd (0.577)",
          abs(g["null_sd"] - 0.5773502691896258) < 1e-12
          and abs(g["threshold"] - (2 * 0.5773502691896258 + 3)) < 1e-12
          and "null_sd_used" not in g, g)
    check("the reason names the flips guard", "flips" in d.reason, d.reason)

    # excess 4.667 > 4.155 under the written rule (a 1-image sd floor gave 5 and passed)
    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=6), null)
    check("cand flips [6,6,6] vs null [1,2,1] rejects under the written rule",
          not d.guards["flips"]["passed"] and d.verdict == "REJECT", d.guards["flips"])

    cand = runs([0.62, 0.625, 0.63], neg=3, pos=10)
    d = G.decide(score(0.60), cand, runs([0.60, 0.605, 0.61], neg=0))
    check("images that became correct do not offset negative flips",
          d.guards["flips"]["cand_flips"] == [3, 3, 3] and d.verdict == "ACCEPT", d.reason)
    check("negative_flips counts only 1 -> 0", G.negative_flips("1100", "1010") == 1)

    null = [score(0.60, neg=0), score(0.605, neg=6), score(0.61, neg=12)]
    cand = runs([0.62, 0.625, 0.63], neg=20)
    d = G.decide(score(0.60), cand, null)
    g = d.guards["flips"]
    check("a noisy null widens the flip allowance to 2 sd + 3 (sd 6 -> 15)",
          g["threshold"] == 15.0 and g["excess"] == 14.0 and g["passed"], g)

    null = runs([0.60, 0.605, 0.61], neg=2)
    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=5), null)
    check("identical null flip counts: sd 0, threshold is the +3 images; excess 3 passes",
          d.guards["flips"]["null_sd"] == 0.0 and d.guards["flips"]["threshold"] == 3.0
          and d.guards["flips"]["passed"] and d.guards["flips"]["excess"] == 3.0,
          d.guards["flips"])
    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63], neg=6), null)
    check("...and excess 4 fails", not d.guards["flips"]["passed"], d.guards["flips"])

    # --- identical seeds ----------------------------------------------------------
    flat_null = runs([0.60, 0.60, 0.60])
    d = G.decide(score(0.60), runs([0.6005, 0.6015, 0.601]), flat_null)
    g = d.guards["regression"]
    check("identical null seeds: sd 0, so the regression threshold is inc itself",
          g["null_sd"] == 0.0 and g["threshold"] == 0.60 and g["passed"], g)
    check("...and the tie is recorded as a warning in the decision and its reason",
          len(d.warnings) == 1 and d.warnings[0].startswith("null:")
          and "WARNING" in d.reason, (d.warnings, d.reason))
    d = G.decide(score(0.60), runs([0.5995, 0.5998, 0.5999]), flat_null)
    check("with a zero null sd any cand mean below inc fails the regression guard",
          not d.guards["regression"]["passed"] and d.verdict == "REJECT", d.guards["regression"])
    c = score(0.62)
    check("one score passed as all three cand seeds raises (same weights)",
          raises(lambda: G.decide(score(0.60), [c, c, c], runs([0.60, 0.605, 0.61])),
                 ValueError, "same weights"))
    dup = runs([0.62, 0.625, 0.63])
    shared = runs([0.60, 0.605, 0.61])
    shared[0]["weights_sha256"] = dup[2]["weights_sha256"]
    check("a cand and a null from the same weights raise",
          raises(lambda: G.decide(score(0.60), dup, shared), ValueError, "same weights"))
    check("the config records no sd floor",
          not any("floor" in k for k in G.GateConfig().__dataclass_fields__))

    # --- classification vs localisation ---------------------------------------
    null = [score(v, agn=a) for v, a in ((0.60, 0.70), (0.605, 0.705), (0.61, 0.71))]
    labels = [score(v, agn=a) for v, a in ((0.57, 0.70), (0.575, 0.705), (0.58, 0.71))]
    d = G.decide(score(0.60), labels, null)
    check("12-class down with agnostic flat is attributed to labels",
          d.attribution["class_vs_loc"] == "labels" and abs(d.attribution["agnostic_delta"]) < 1e-12
          and abs(d.attribution["map_delta"] + 0.03) < 1e-12, d.attribution)
    check("labels is in the reason", "class_vs_loc=labels" in d.reason, d.reason)
    domain = [score(v, agn=a) for v, a in ((0.57, 0.66), (0.575, 0.665), (0.58, 0.67))]
    d = G.decide(score(0.60), domain, null)
    check("both down is attributed to domain/localisation",
          d.attribution["class_vs_loc"] == "domain/localisation", d.attribution)
    d = G.decide(score(0.60), runs([0.62, 0.625, 0.63]), null)
    check("no 12-class drop means no class_vs_loc attribution",
          d.attribution["class_vs_loc"] == "none", d.attribution)
    tiny = [score(v, agn=a) for v, a in ((0.598, 0.66), (0.603, 0.665), (0.608, 0.67))]
    d = G.decide(score(0.60), tiny, null)
    check("a 12-class dip inside 2 sd(null) is not a drop",
          d.attribution["class_vs_loc"] == "none", d.attribution)

    # --- consistency checks -----------------------------------------------------
    cands = runs([0.62, 0.625, 0.63])
    other = runs([0.60, 0.605, 0.61])
    other[2]["key_order_sha256"] = "b" * 64
    check("scores taken in a different image order raise",
          raises(lambda: G.decide(score(0.60), cands, other), ValueError, "key_order_sha256"))
    check("an incumbent scored by an older scorer.py raises",
          raises(lambda: G.decide(score(0.60, scorer_sha256="0" * 64), cands,
                                  runs([0.60, 0.605, 0.61])), ValueError, "scorer_sha256"))
    check("a score of another manifest raises",
          raises(lambda: G.decide(score(0.60), cands,
                                  runs([0.60, 0.605, 0.61], manifest_sha256="7" * 64)),
                 ValueError, "manifest_sha256"))
    check("an incumbent scored on test while the runs are on dev raises",
          raises(lambda: G.decide(score(0.60, exam="test"), cands, runs([0.60, 0.605, 0.61])),
                 ValueError, "exam"))
    check("every score on test raises: decisions are taken on dev only",
          raises(lambda: G.decide(score(0.60, exam="test"), runs([0.62, 0.625, 0.63], exam="test"),
                                  runs([0.60, 0.605, 0.61], exam="test")), ValueError, "'dev'"))
    unstamped = runs([0.60, 0.605, 0.61])
    del unstamped[0]["scorer_sha256"]
    check("a score without its scorer stamp raises",
          raises(lambda: G.decide(score(0.60), cands, unstamped), KeyError, "scorer_sha256"))
    check("an empty stamp raises",
          raises(lambda: G.decide(score(0.60, manifest_sha256=""), cands,
                                  runs([0.60, 0.605, 0.61])), ValueError, "manifest_sha256"))
    short = runs([0.60, 0.605, 0.61])
    short[0]["image_correct"] = short[0]["image_correct"][:-1]
    check("an image_correct shorter than n_images raises",
          raises(lambda: G.decide(score(0.60), cands, short), ValueError, "image_correct"))

    def emptied(s):
        s["image_correct"], s["n_images"] = "", 0
        return s
    check("an empty image_correct raises (the flips guard would pass on 0 images)",
          raises(lambda: G.decide(emptied(score(0.60)),
                                  [emptied(x) for x in runs([0.62, 0.625, 0.63], neg=30)],
                                  [emptied(x) for x in runs([0.60, 0.605, 0.61])]),
                 ValueError, "n_images"))
    no_bits = runs([0.60, 0.605, 0.61])
    no_bits[1]["image_correct"] = ""
    check("an empty image_correct against a positive n_images raises",
          raises(lambda: G.decide(score(0.60), cands, no_bits), ValueError, "image_correct"))
    diff_gt = runs([0.60, 0.605, 0.61], n_gt=dict(N_GT, Waterhemp=121))
    check("different n_gt (a different exam) raises",
          raises(lambda: G.decide(score(0.60), cands, diff_gt), ValueError))
    check("two seeds instead of three raise",
          raises(lambda: G.decide(score(0.60), runs([0.62, 0.63]), runs([0.6, 0.61, 0.6])),
                 ValueError))
    d = G.decide(score(0.60), runs([0.62, 0.63]), runs([0.6, 0.61]), G.GateConfig(min_seeds=2))
    check("...unless the config allows fewer", d.verdict == "ACCEPT", d.reason)
    nan = runs([0.60, 0.605, 0.61])
    nan[0]["map50_95"] = float("nan")
    check("a NaN score raises",
          raises(lambda: G.decide(score(0.60), cands, nan), ValueError))

    # --- JSON round trip --------------------------------------------------------
    spread = [score(v, species={"Waterhemp": v - 0.02, "Carpetweed": v + 0.01,
                                "Eclipta": v - 0.005})
              for v in (0.62, 0.625, 0.63)]
    rejected = [score(v, agn=a, species={"Sicklepod": v - 0.01})
                for v, a in ((0.57, 0.70), (0.575, 0.705), (0.58, 0.71))]
    base_agn = [score(v, agn=a) for v, a in ((0.60, 0.70), (0.605, 0.705), (0.61, 0.71))]
    for label, (c, n) in (("ACCEPT", (spread, runs([0.60, 0.605, 0.61]))),
                          ("REJECT", (rejected, base_agn))):
        d = G.decide(score(0.60), c, n)
        check("%s fixture has distinct per-species deltas" % label,
              len({round(x, 12) for _, x in d.attribution["per_species_delta"]}) > 1,
              d.attribution["per_species_delta"])
        check("%s fixture is a %s" % (label, label), d.verdict == label, d.reason)
        text = json.dumps(d.to_dict(), allow_nan=False, sort_keys=True)
        back = G.Decision.from_dict(json.loads(text))
        check("%s decision survives a strict sort_keys JSON round trip unchanged" % label,
              back == d and back.to_dict() == d.to_dict())
        check("%s round trip keeps the per-species order (most negative first)" % label,
              delta_names(back) == delta_names(d)
              and delta_names(d) != sorted(delta_names(d)), delta_names(back))
    check("the decision records the config it was made with",
          d.config["p_accept"] == 0.75 and d.config["min_species_gt"] == 30
          and d.config["flips_slack_images"] == 3.0, d.config)

    # --- soup -------------------------------------------------------------------
    cands = runs([0.62, 0.625, 0.63])
    check("a soup at or above mean(cand) becomes the incumbent",
          G.choose_soup(score(0.626), cands) == "soup")
    check("a soup equal to mean(cand) (identical cands) is kept",
          G.choose_soup(score(0.6), runs([0.6, 0.6, 0.6])) == "soup")
    check("a soup below mean(cand) loses to cand0",
          G.choose_soup(score(0.624), cands) == "cand0")
    check("a soup scored in another image order raises",
          raises(lambda: G.choose_soup(score(0.63, key="c" * 64), cands), ValueError))
    check("a soup scored by another scorer.py raises",
          raises(lambda: G.choose_soup(score(0.63, scorer_sha256="0" * 64), cands), ValueError,
                 "scorer_sha256"))
    check("a soup scored on test raises",
          raises(lambda: G.choose_soup(score(0.63, exam="test"), runs([0.62, 0.625, 0.63],
                                                                      exam="test")),
                 ValueError, "'dev'"))
    soup_is_cand = score(0.63)
    soup_is_cand["weights_sha256"] = cands[0]["weights_sha256"]
    check("a 'soup' that is one of the cands' weights raises",
          raises(lambda: G.choose_soup(soup_is_cand, cands), ValueError, "same weights"))

    # --- truth arm --------------------------------------------------------------
    check("truth: with D_k above without in 7/9 pairs helps",
          G.truth_decision(runs([0.62, 0.63, 0.603]), runs([0.60, 0.605, 0.61])) == "helps")
    check("truth: 6.5/9 (one tie) is still neutral",
          G.truth_decision(runs([0.62, 0.63, 0.60]), runs([0.60, 0.605, 0.61])) == "neutral")
    check("truth: with D_k below without in every pair hurts",
          G.truth_decision(runs([0.55, 0.56, 0.57]), runs([0.60, 0.605, 0.61])) == "hurts")
    check("truth: indistinguishable is neutral",
          G.truth_decision(runs([0.60, 0.61, 0.62]), runs([0.605, 0.615, 0.60])) == "neutral")
    t = G.truth_detail(runs([0.62, 0.63, 0.603]), runs([0.60, 0.605, 0.61]))
    check("truth_detail carries P and the means", abs(t["p"] - 7 / 9) < 1e-12
          and abs(t["without_mean"] - 0.605) < 1e-12, t)
    check("truth_detail records the species guard and the stamps",
          t["species"]["passed"] and t["species"]["arms"] == ["with", "without"]
          and t["stamps"]["exam"] == "dev", t["species"]["arms"])

    # The gate rejects this step on Carpetweed; the truth arm must agree, not say "helps".
    with_ = [score(v, species={"Carpetweed": v - 0.075}) for v in (0.62, 0.625, 0.63)]
    without = runs([0.60, 0.605, 0.61])
    t = G.truth_detail(with_, without)
    gate_d = G.decide(score(0.605), [score(v, species={"Carpetweed": v - 0.075})
                                     for v in (0.62, 0.625, 0.63)], runs([0.60, 0.605, 0.61]))
    check("truth: a species drop the gate would reject is 'hurts' even at P = 1",
          t["p"] == 1.0 and t["verdict"] == "hurts" and t["species"]["failed"] == ["Carpetweed"]
          and gate_d.verdict == "REJECT", (t["verdict"], t["species"]["failed"], gate_d.verdict))
    check("truth runs from different exams raise",
          raises(lambda: G.truth_decision(runs([0.62, 0.63, 0.60]),
                                          runs([0.6, 0.6, 0.6], key="d" * 64)), ValueError))
    check("truth runs from different scorers raise",
          raises(lambda: G.truth_decision(runs([0.62, 0.63, 0.60]),
                                          runs([0.6, 0.61, 0.6], scorer_sha256="0" * 64)),
                 ValueError, "scorer_sha256"))
    few_gt = dict(N_GT, Carpetweed=5)
    check("truth runs on an exam with a species below 30 boxes raise",
          raises(lambda: G.truth_decision(runs([0.62, 0.63, 0.60], n_gt=few_gt),
                                          runs([0.6, 0.61, 0.6], n_gt=few_gt)),
                 ValueError, "Carpetweed"))
    r = runs([0.62, 0.63, 0.60])
    check("truth: one run passed as with and without raises",
          raises(lambda: G.truth_decision(r, [r[0], score(0.6), score(0.61)]), ValueError,
                 "same weights"))

    print("production scores only")
    test_production_only()

    print("flips mode: v1 negative (default, unchanged) and v2 net")
    test_flips_modes()

    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0



if __name__ == "__main__":
    sys.exit(main())
