#!/usr/bin/env python3
"""The audit estimators and hypothesis evaluators (contract §5.3-5.4, §6;
runner §4.16, §5.1.6).

Numerics, against references computed independently here (and against
scipy when it is installed):
  * betainc against Simpson integration of the beta density; beta_ppf
    against the closed forms of Beta(1, 1), Beta(a, 1), Beta(1, b) and the
    arcsine Beta(1/2, 1/2), and against scipy.stats.beta.ppf to 1e-9;
  * binom_interval against the Wilson half-widths of contract §5.3 (parsed
    from docs/FUNNEL_AUDIT.md) to 3 decimals; the table's first column, the
    exact one-sided 95 % bound for 0 of n, against clopper_pearson; Wilson
    and Jeffreys against their formulas; Clopper-Pearson at x = 0 and x = n
    against (alpha/2)^(1/n);
  * stratified: coverage of the Korn-Graubard interval >= 0.93 over 400
    seeded draws from a stratified population;
  * Horvitz-Thompson unbiasedness by exhaustive enumeration: srs (all 20
    samples of 3 from 6), Poisson sampling with unequal probabilities (all 64
    subsets) and the G4 dual design (srs x Poisson, union inclusion);
  * PPI++: coverage >= 0.93 by simulation with a fixed seed, narrower than
    the labelled-only interval with an informative predictor, lambda 0 for a
    useless one; a stratum takes PPI only when it narrows the interval;
  * Rogan-Gladen on a worked example (obs 0.30, Se 0.90, Sp 0.95 -> 0.25/0.85),
    the melded interval reducing to Clopper-Pearson at perfect Se and Sp,
    theta_obs at Se = Sp = 1, and the flag below Se + Sp - 1 = 0.7;
  * the exact permutation test: minimum p 1/20 at 3 v 3 and 1/252 at 5 v 5;
    gate.truth_detail's "helps" rule (P >= 0.75) has null probability 4/20
    over the 20 equally likely rank splits (contract §1, §6 H10a);
  * holm on a textbook example; binom_upper_tail against exact sums;
    cluster_bootstrap's design effect; webber_recall; ltt_threshold.

Evaluators, on synthetic gold where the truth is set so that each verdict
(supported, falsified, inconclusive, bounded) occurs: H0, H1, H2a, H2b,
H3a, H3b, H4, H5a/H5b, H9', H11, H12, with the thresholds read from
prereg_v1.json. Unsure answers are taken both ways; H0 falsified sets the
stop; a labeller scope or a PPI predictor sharing material with a stratum
makes the audit invalid (R12).

End to end: draw on the synthetic world, gold and labeller qualification
written in their runner formats, evaluate() -> audit_v1.json/.md; the
refusals (missing labeller file or DA file, a changed sample or gold);
H0(a) covers the world's planted share; the audit is deterministic.

Run:  python3 tests/test_funnel_estimate.py
"""
import copy
import csv
import itertools
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_estimate_")
check, raises = W.check, W.raises

import numpy as np  # noqa: E402
from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import estimate as E  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402

try:
    import scipy.stats as SS
except ImportError:
    SS = None

PREREG = TMP / "inc" / "funnel" / "prereg_v1.json"
dom = D.load("weed")
pre = D.load_prereg(PREREG)
RAW = json.loads(PREREG.read_text())


# ======================================================================== numerics
print("betainc and beta_ppf")


def simpson_cdf(a, b, x, n=20000):
    """I_x(a, b) by Simpson's rule on the density (a, b >= 1 here)."""
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    h = x / n
    s = 0.0
    for i in range(n + 1):
        t = i * h
        f = 0.0 if (t <= 0 and a > 1) else math.exp(lb + (a - 1) * math.log(max(t, 1e-300)) +
                                                     (b - 1) * math.log1p(-t)) if t < 1 else 0.0
        s += f * (1 if i in (0, n) else (4 if i % 2 else 2))
    return s * h / 3


worst = 0.0
for a, b, x in ((2, 3, 0.4), (5, 1.5, 0.7), (1, 1, 0.3), (3.5, 27.5, 0.1), (10, 10, 0.55), (1.5, 8, 0.05)):
    worst = max(worst, abs(E.betainc(a, b, x) - simpson_cdf(a, b, x)))
check("betainc = Simpson integral (max err %.2e)" % worst, worst < 1e-7)
check("betainc edges", E.betainc(2, 3, 0) == 0.0 and E.betainc(2, 3, 1) == 1.0)
check("betainc point masses", E.betainc(0, 3, 0.2) == 1.0 and E.betainc(3, 0, 0.2) == 0.0)
worst = 0.0
for q in (0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999):
    worst = max(worst, abs(E.beta_ppf(q, 1, 1) - q), abs(E.beta_ppf(q, 3.5, 1) - q ** (1 / 3.5)),
                abs(E.beta_ppf(q, 1, 7) - (1 - (1 - q) ** (1 / 7.0))),
                abs(E.beta_ppf(q, 0.5, 0.5) - math.sin(math.pi * q / 2) ** 2))
check("beta_ppf closed forms (max err %.2e)" % worst, worst < 1e-10)
check("beta_ppf edges", E.beta_ppf(0, 2, 3) == 0.0 and E.beta_ppf(1, 2, 3) == 1.0
      and E.beta_ppf(0.3, 0, 2) == 0.0 and E.beta_ppf(0.3, 2, 0) == 1.0)
check("beta_ppf refuses q outside [0, 1]", raises(lambda: E.beta_ppf(1.5, 2, 2), F.EstimateError))
if SS is not None:
    worst = 0.0
    for a, b in ((0.5, 30.5), (3.5, 27.5), (12, 3), (150.5, 50.5), (1, 400), (2.2, 0.7)):
        for q in (0.0005, 0.025, 0.5, 0.975, 0.9995):
            worst = max(worst, abs(E.beta_ppf(q, a, b) - float(SS.beta.ppf(q, a, b))))
    check("beta_ppf = scipy.stats.beta.ppf to 1e-9 (max err %.2e)" % worst, worst < 1e-9)
else:
    W.skip("beta_ppf against scipy", "scipy is not installed")

print("binomial intervals against contract §5.3")
contract = (W.GIT_ROOT / "docs" / "FUNNEL_AUDIT.md").read_text(encoding="utf-8")
sec = contract.split("### 5.3 Sample-size reference", 1)[1].split("###", 1)[0]
rows = [r for r in re.findall(r"^\|\s*(\d+)\s*\|\s*([0-9.]+) %\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
                              sec, re.M)]
check("the §5.3 table has five rows", len(rows) == 5, rows)
for n, up0, h5, h2, h1 in rows:
    n = int(n)
    cp_up = E.clopper_pearson(0, n, 0.90)[1]
    check("n=%d: exact one-sided 95%% bound for 0 of n = %s %%" % (n, up0), round(100 * cp_up, 1) == float(up0),
          100 * cp_up)
    for p, want in ((0.5, h5), (0.2, h2), (0.1, h1)):
        lo, hi = E.binom_interval(round(p * n), n)
        check("n=%d p=%.1f: Wilson half-width %s" % (n, p, want), round((hi - lo) / 2, 3) == float(want),
              (hi - lo) / 2)
z = E.Z975
for k, n in ((0, 30), (7, 40), (33, 50), (60, 60), (123, 400)):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = E.binom_interval(k, n)
    check("Wilson formula k=%d n=%d" % (k, n), abs(lo - max(0, c - h)) < 1e-12 and abs(hi - min(1, c + h)) < 1e-12)
for k, n in ((0, 10), (3, 12), (7, 20), (29, 29)):
    lo, hi = E.binom_interval(k, n)
    wl = 0.0 if k == 0 else E.beta_ppf(0.025, k + 0.5, n - k + 0.5)
    wh = 1.0 if k == n else E.beta_ppf(0.975, k + 0.5, n - k + 0.5)
    check("Jeffreys below n=30 (k=%d n=%d)" % (k, n), (lo, hi) == (wl, wh))
    if SS is not None and 0 < k < n:
        check("Jeffreys = scipy beta quantiles (k=%d n=%d)" % (k, n),
              abs(lo - SS.beta.ppf(0.025, k + .5, n - k + .5)) < 1e-9 and
              abs(hi - SS.beta.ppf(0.975, k + .5, n - k + .5)) < 1e-9)
for n in (5, 30, 100):
    check("Clopper-Pearson x=0 upper = 1-(0.025)^(1/n), n=%d" % n,
          abs(E.clopper_pearson(0, n)[1] - (1 - 0.025 ** (1.0 / n))) < 1e-10 and E.clopper_pearson(0, n)[0] == 0)
    check("Clopper-Pearson x=n lower = 0.025^(1/n), n=%d" % n,
          abs(E.clopper_pearson(n, n)[0] - 0.025 ** (1.0 / n)) < 1e-10 and E.clopper_pearson(n, n)[1] == 1)
if SS is not None:
    lo, hi = E.clopper_pearson(7, 25)
    check("Clopper-Pearson = scipy (7 of 25)", abs(lo - SS.beta.ppf(0.025, 7, 19)) < 1e-9
          and abs(hi - SS.beta.ppf(0.975, 8, 18)) < 1e-9)
lo, hi = E.clopper_pearson(7.5, 25.2)
check("Clopper-Pearson accepts non-integer x and n", 0 < lo < 7.5 / 25.2 < hi < 1)
check("binom_upper_tail = exact sums", all(abs(E.binom_upper_tail(x, 20, 0.1) -
                                               sum(math.comb(20, i) * 0.1 ** i * 0.9 ** (20 - i)
                                                   for i in range(x, 21))) < 1e-12 for x in (1, 2, 5, 11)))
check("binom_upper_tail edges", E.binom_upper_tail(0, 10, 0.1) == 1.0 and E.binom_upper_tail(11, 10, 0.1) == 0.0)

print("stratified coverage")
r = np.random.default_rng(20260928)
Ns = [800, 300, 120, 60]
ps = [0.05, 0.3, 0.6, 0.9]
pop = [np.array([1] * int(round(N * p)) + [0] * (N - int(round(N * p)))) for N, p in zip(Ns, ps)]
theta = sum(x.sum() for x in pop) / float(sum(Ns))
cover = 0
for rep in range(400):
    st = []
    for x, n in zip(pop, (40, 25, 15, 10)):
        s = r.choice(len(x), size=n, replace=False)
        st.append({"N": len(x), "n": n, "k": int(x[s].sum())})
    e = E.stratified(st)
    cover += e["interval"][0] <= theta <= e["interval"][1]
check("Korn-Graubard coverage %.3f >= 0.93 (nominal 0.95)" % (cover / 400.0), cover / 400.0 >= 0.93)
e = E.stratified([{"N": 100, "n": 10, "k": 0}, {"N": 50, "n": 5, "k": 0}])
check("all-zero strata: n_eff = sum n, CP(0, 15)", e["n_eff"] == 15 and abs(e["interval"][1] -
                                                                              E.clopper_pearson(0, 15)[1]) < 1e-12)
e = E.stratified([{"N": 100, "n": 1, "k": 1}, {"N": 100, "n": 20, "k": 10}])
check("n_h = 1 gives W^2/4", abs(e["var"] - (0.5 ** 2 * 0.25 + 0.5 ** 2 * (1 - 0.2) * 0.25 / 19)) < 1e-12)
e = E.stratified([{"N": 100, "n": 20, "k": 10}, {"N": 50, "n": 0, "k": 0}])
check("an unlabelled stratum widens the interval over its weight", e["uncovered_weight"] == 50 / 150.0
      and e["interval"][1] >= 1 / 3.0 and e["interval"][0] <= (2 / 3.0) * 0.5)

print("Horvitz-Thompson unbiasedness by enumeration")
y = [1, 0, 1, 1, 0, 1]
N = len(y)
vals = []
for s in itertools.combinations(range(N), 3):
    vals.append(E.ht([y[i] for i in s], [3 / 6.0] * 3, N)["estimate"])
check("srs: mean over all 20 samples = theta", abs(sum(vals) / len(vals) - sum(y) / 6.0) < 1e-12 and len(vals) == 20)
p = [0.9, 0.2, 0.5, 0.35, 0.7, 0.15]
exp_ = 0.0
for mask in range(1 << N):
    s = [i for i in range(N) if mask >> i & 1]
    prob = 1.0
    for i in range(N):
        prob *= p[i] if mask >> i & 1 else 1 - p[i]
    t = E.ht([y[i] for i in s], [p[i] for i in s], N)["estimate"] if s else 0.0
    exp_ += prob * t
check("Poisson (unequal p): E[theta_hat] = theta over all 64 subsets", abs(exp_ - sum(y) / 6.0) < 1e-12)
u = 2 / 6.0
exp_ = 0.0
combos = list(itertools.combinations(range(N), 2))
for s1 in combos:
    for mask in range(1 << N):
        prob = 1.0 / len(combos)
        for i in range(N):
            prob *= p[i] if mask >> i & 1 else 1 - p[i]
        s = sorted(set(s1) | {i for i in range(N) if mask >> i & 1})
        pis = [1 - (1 - u) * (1 - p[i]) for i in s]
        exp_ += prob * (E.ht([y[i] for i in s], pis, N)["estimate"] if s else 0.0)
check("dual design (srs x Poisson, union pi): E[theta_hat] = theta", abs(exp_ - sum(y) / 6.0) < 1e-12)
check("ht refuses pi outside (0, 1]", raises(lambda: E.ht([1], [0.0], 3), F.EstimateError))

print("PPI++")
r = np.random.default_rng(7)
Npop = 4000
x = r.random(Npop)
ytrue = (r.random(Npop) < x).astype(float)
f = np.clip(x + r.normal(0, 0.05, Npop), 0, 1)
theta = ytrue.mean()
cover = narrower = 0
for rep in range(400):
    s = r.choice(Npop, 80, replace=False)
    pp = E.ppi_pp(ytrue[s], f[s], f)
    cover += pp["interval"][0] <= theta <= pp["interval"][1]
    lo, hi = E.binom_interval(ytrue[s].sum(), 80)
    narrower += (pp["interval"][1] - pp["interval"][0]) < (hi - lo)
check("PPI++ coverage %.3f >= 0.93" % (cover / 400.0), cover / 400.0 >= 0.93)
check("PPI++ narrower than the labelled-only interval in most draws (%d/400)" % narrower, narrower >= 300)
pp = E.ppi_pp([1, 0, 1, 0, 1, 1], [0.5] * 6, [0.5] * 50)
check("a constant predictor gets lambda 0", pp["lambda"] == 0.0 and abs(pp["estimate"] - 4 / 6.0) < 1e-12)
pp = E.ppi_pp([1, 0, 1, 1], [0.9, 0.1, 0.8, 0.7], [0.9, 0.1, 0.8, 0.7] + [0.5] * 20, pi_lab=[0.2] * 4)
check("PPI with HT weights runs and clips lambda to [0, 1]", 0 <= pp["lambda"] <= 1)

print("Rogan-Gladen")
rg = E.rogan_gladen(30, 100, 90, 100, 95, 100, "funnel/test/rg")
check("worked example: (0.30 + 0.95 - 1) / (0.90 + 0.95 - 1) = %.6f" % (0.25 / 0.85),
      abs(rg["estimate"] - 0.25 / 0.85) < 1e-12 and rg["applied"])
cp = E.clopper_pearson(30, 100)
check("the melded interval contains the point and is wider than CP",
      rg["interval"][0] <= rg["estimate"] <= rg["interval"][1] and rg["interval"][1] - rg["interval"][0] >
      cp[1] - cp[0])
rg1 = E.rogan_gladen(30, 100, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, "funnel/test/rg1")
check("perfect Se and Sp: theta_obs and the Clopper-Pearson interval (MC tolerance 0.01)",
      abs(rg1["estimate"] - 0.3) < 1e-12 and abs(rg1["interval"][0] - cp[0]) < 0.01
      and abs(rg1["interval"][1] - cp[1]) < 0.01,
      (rg1["interval"], cp))
rg2 = E.rogan_gladen(30, 100, 50, 50, 50, 50, "funnel/test/rg2")
check("Se = Sp = 1 returns theta_obs", abs(rg2["estimate"] - 0.3) < 1e-12)
rg3 = E.rogan_gladen(30, 100, 60, 100, 80, 100, "funnel/test/rg3")
check("Se + Sp - 1 = 0.4 < 0.7: uncorrected and flagged", rg3["flag"] == "se_sp_low" and rg3["estimate"] == 0.3
      and not rg3["applied"])
check("Rogan-Gladen is deterministic under its seed",
      E.rogan_gladen(30, 100, 90, 100, 95, 100, "funnel/test/rg") == rg)

print("permutation test and the gate's null")
pt = E.perm_test_one_sided([3, 4, 5], [0, 1, 2])
check("3 v 3 minimum p = 1/20", pt["p"] == 1 / 20.0 and pt["n_splits"] == 20)
pt = E.perm_test_one_sided([5.1, 6, 7, 8, 9], [0, 1, 2, 3, 4])
check("5 v 5 minimum p = 1/252", pt["p"] == 1 / 252.0 and pt["n_splits"] == 252)
check("p = 1 when a is the smallest", E.perm_test_one_sided([0, 1, 2], [3, 4, 5])["p"] == 1.0)
helps = 0
for w in itertools.combinations(range(6), 3):
    wo = [i for i in range(6) if i not in w]
    helps += G.p_greater([float(i) for i in w], [float(i) for i in wo]) >= 0.75
check("gate.truth_detail 'helps' (P >= 0.75) under no effect: %d/20 = 0.20" % helps, helps == 4)

print("holm, bootstrap, recall, LTT")
hm = E.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, alpha=0.05)
check("holm textbook adjusted p", [round(hm[k]["p_adjusted"], 10) for k in "dacb"] == [0.02, 0.03, 0.06, 0.06])
check("holm rejects d and a", [k for k in "abcd" if hm[k]["reject"]] == ["a", "d"])
cb = E.cluster_bootstrap([1, 1, 1, 0, 0, 0] * 5, ["c%d" % (i // 3) for i in range(30)], "funnel/test/boot")
cb0 = E.cluster_bootstrap([1, 0] * 15, ["c%d" % i for i in range(30)], "funnel/test/boot")
check("design effect > 2 for perfectly clustered data (%.2f)" % cb["deff"], cb["deff"] > 2)
check("design effect near 1 for singleton clusters (%.2f)" % cb0["deff"], 0.7 < cb0["deff"] < 1.3)
wr = E.webber_recall({"N": 100, "k": 50, "n": 50}, [{"N": 100, "n": 20, "k": 10}], "funnel/test/recall", 5000)
check("Webber plug-in = 100 / (100 + 10 + 80 x 0.5)", abs(wr["plug_in"] - 100 / 150.0) < 1e-12)
check("Webber estimate is the Monte Carlo median, inside its interval (%.3f)" % wr["estimate"],
      wr["interval"][0] <= wr["estimate"] <= wr["interval"][1] and abs(wr["estimate"] - 100 / 150.0) < 0.05)
check("no discard: recall 1", E.webber_recall({"N": 10, "k": 5, "n": 5}, [], "s", 100)["estimate"] == 1.0)
check("Webber is deterministic", E.webber_recall({"N": 100, "k": 50, "n": 50}, [{"N": 100, "n": 20, "k": 10}],
                                                 "funnel/test/recall", 5000) == wr)
sc = list(np.linspace(0.01, 1, 400))
corr = [s >= 0.5 or (i % 2 == 0) for i, s in enumerate(sc)]
lt = E.ltt_threshold(sc, corr)
sel = [c for s, c in zip(sc, corr) if s >= lt["threshold"]]
check("LTT certifies a threshold whose precision is >= 0.95 (%.3f)" % lt["threshold"],
      math.isfinite(lt["threshold"]) and sum(sel) / len(sel) >= 0.95 and lt["threshold"] >= 0.45)
check("LTT gives inf when nothing is precise", math.isinf(E.ltt_threshold(sc, [False] * 400)["threshold"]))
# the top 100 of 400 are all correct: p = 0.95^100 = 0.006 passes delta = 0.025 alone, but not delta / 20
check("LTT applies Bonferroni over its candidate grid (0.95^100 > 0.025 / 20: nothing certified)",
      math.isinf(E.ltt_threshold(sc, [i >= 300 for i in range(400)])["threshold"]))

# ======================================================================== rules
print("rules from prereg_v1.json")
R = E.rules(pre, dom)
H = RAW["hypotheses"]
check("conf from decisions_one_sided", abs(R["conf"] - (2 * RAW["estimators"]["decisions_one_sided"] - 1)) < 1e-12)
for hid, key, field in (("H1", "lb_min", "supported"), ("H1", "ub_max", "falsified"), ("H2a", "lb_min", "supported"),
                        ("H2a", "ub_max", "falsified"), ("H2b", "ub_max", "supported"), ("H2b", "lb_min", "falsified"),
                        ("H3b", "lb_min", "supported"), ("H3b", "ub_max", "falsified"), ("H5b", "lb_min", "supported"),
                        ("H5b", "ub_max", "falsified"), ("H12", "lb_min", "supported"), ("H12", "ub_max", "falsified"),
                        ("H9_prime", "lb_min", "supported"), ("H9_prime", "ub_max", "falsified")):
    txt = H[hid][field]
    check("%s %s %g is in the prereg text %r" % (hid, key, R[hid][key], txt),
          any(abs(float(v) - R[hid][key]) < 1e-12 for v in re.findall(r"[0-9]+(?:\.[0-9]+)?", txt)))
check("H1 class is a target named in the prereg", R["H1"]["class"] in dom.target_names and
      R["H1"]["class"] in H["H1"]["supported"])
check("H3a thresholds are the prereg's", R["H3a"]["geometry_match_min"] == H["H3a"]["geometry_match_min"] and
      R["H3a"]["id_agreement_min"] == H["H3a"]["id_agreement_min"] and
      R["H3a"]["purity_lb_min"] == H["H3a"]["purity"]["lb_min"] and R["H3a"]["classes"] == H["H3a"]["purity"]["classes"])
check("H4 thresholds", [R["H4"][k] for k in ("named_ub_max", "noinfo_ub_max", "named_lb_min", "noinfo_lb_min")] ==
      [float(v) for v in re.findall(r"0\.[0-9]+", H["H4"]["supported"])] +
      [float(v) for v in re.findall(r"0\.[0-9]+", H["H4"]["falsified"])])
check("H5a threshold", str(R["H5a"]["boxes_max"]) in H["H5a"]["exact"])
check("screening", R["screening"]["R_min"] == RAW["estimators"]["screening"]["R_min_boxes"]
      and R["screening"]["fn_rate_null"] == RAW["estimators"]["screening"]["fn_rate_null"])
bad = copy.deepcopy(RAW)
bad["hypotheses"]["H2a"]["supported"] = "at least half"
check("an unreadable rule text refuses", raises(lambda: E.rules(bad, dom), F.EstimateError))


# ======================================================================== evaluators
def est(k, n):
    lo, hi = E.binom_interval(k, n)
    return {"k": k, "n": n, "estimate": k / float(n), "lb": lo, "ub": hi}


GENERA = sorted({t["taxon"].split()[0] for t in dom.targets})


def by_genus(unqualified=()):
    """rl_qualification.json by_genus for one backend and scope (qualify.rl's
    record): every target genus qualified at species level except those named."""
    ok = {"se": est(300, 300), "sp": est(300, 300), "qualified": True}
    bad = {"se": est(20, 40), "sp": est(20, 40), "qualified": False}
    return {g: {"species": bad if g in unqualified else ok, "genus": ok} for g in GENERA}


class Builder(object):
    """A Context from strata of synthetic units, sampled items and answers."""

    def __init__(self, level="species", scope="KT7", se=(1000, 1000), sp=(1000, 1000), unqualified_genera=()):
        self.units = {}
        self.sample = []
        self.gold = []
        self.key = []
        self.level, self.scope = level, scope
        self.se, self.sp = se, sp
        self.unqualified = set(unqualified_genera)
        self.n_units = 0
        self.extra = {}

    def stratum(self, sid, N, answers, label="", pred="", source=W.ND, status="", frame="", pi=None,
                unit_prefix=None, src_id="1", allowed=""):
        """N units; the first len(answers) are sampled; an answer is a gold
        dict (or None: sampled, not answered)."""
        group = sid.split("/", 1)[0]
        tag = unit_prefix or F.sha256_bytes(sid.encode())[:8]
        lab = dom.lab_of(source)
        units = []
        for i in range(N):
            uid = "b:%s_%d#0" % (tag, i)
            u = {"unit_id": uid, "unit": "box", "stratum": sid, "source": source, "image_key": "%s_%d" % (tag, i),
                 "crop_id": str(self.n_units), "lab": lab, "near_dup3": "n:%s_%d" % (tag, i),
                 "provenance": "prov:%s_%d" % (tag, i), "label": label, "pred": pred, "score": "0.1",
                 "allowed_judges": allowed,
                 "extra": json.dumps({"status": status, "frame": frame, "src_id": src_id, "kt": []})}
            self.n_units += 1
            units.append(u)
        self.units.setdefault(group, []).extend(units)
        n = len(answers)
        for i, a in enumerate(answers):
            u = units[i]
            iid = F.sha256_bytes(("%s/%s" % (u["unit_id"], group)).encode())[:16]
            self.sample.append({"item_id": iid, "unit_id": u["unit_id"], "unit": "box", "group": group,
                                "stratum": sid, "source": source, "image_key": u["image_key"],
                                "crop_id": u["crop_id"], "pi": repr(pi if pi is not None else n / float(N)),
                                "pi_parts": "{}", "seed_text": "funnel/v1/" + sid, "draw_rank": str(i),
                                "sheet_class": "eval" if group == "G5" else "pool", "lab": lab,
                                "near_dup3": u["near_dup3"], "provenance": u["provenance"], "kt": ""})
            if a is not None:
                self.gold.append(dict({"item_id": iid, "unit_id": u["unit_id"], "backend": "RL-B",
                                       "answer_level": "species", "answer_taxon": "", "box_ok": "yes"}, **a))
        return units

    def g0(self, truths, answers, share):
        for i, (t, a) in enumerate(zip(truths, answers)):
            iid = "g0item%04d" % i
            self.sample.append({"item_id": iid, "unit_id": "G0:" + iid, "unit": "", "group": "G0", "stratum": "G0/all",
                                "source": "", "image_key": "", "crop_id": "", "pi": "", "pi_parts": "{}",
                                "seed_text": "funnel/v1/G0", "draw_rank": str(i), "sheet_class": "pool", "lab": "",
                                "near_dup3": "", "provenance": "", "kt": ""})
            self.key.append({"item_id": iid, "unit_id": "t7:o%d/p0" % i, "truth": t, "truth_kind": "target"
                             if t is not None else "attractor", "truth_taxon": None, "pair_truth": None})
            if a is not None:
                self.gold.append(dict({"item_id": iid, "unit_id": "G0:" + iid, "backend": "RL-B",
                                       "answer_level": "species", "answer_taxon": "", "box_ok": "na"}, **a))
        self.key.append({"planted": {"share": share}})

    def sentinel(self, kt, source, n=3):
        for i in range(n):
            self.sample.append({"item_id": "sent%s%d" % (kt, i), "unit_id": "t1:s%s%d#0" % (kt, i), "unit": "box",
                                "group": "sentinel", "stratum": "sentinel/kt=%s/truth_kind=target" % kt,
                                "source": source, "image_key": "", "crop_id": "", "pi": "1.0", "pi_parts": "{}",
                                "seed_text": "x", "draw_rank": str(i), "sheet_class": "pool",
                                "lab": dom.lab_of(source), "near_dup3": "n:s%d" % i, "provenance": "prov:s%d" % i,
                                "kt": kt})

    def rlq(self, strata_scopes=None, pairs_ok=True):
        blk = {"se": est(*self.se), "sp": est(*self.sp), "qualified": True}
        prim = {h: {"backend": "RL-B", "level": self.level, "scope": self.scope}
                for h in ("H0", "H1", "H2a", "H2b", "H3a", "H3b", "H4")}
        scopes = {}
        for g, us in self.units.items():
            for u in us:
                scopes[u["stratum"]] = self.scope
        scopes.update(strata_scopes or {})
        return {"gold_sha256": "0" * 64, "primary": prim,
                "backends": {"RL-B": {sc: {self.level: blk} for sc in set(scopes.values()) | {self.scope}}},
                "levels": {"RL-B": {self.scope: self.level}}, "strata_scopes": scopes,
                "pairs": {"RL-B": {"qualified": pairs_ok}},
                "by_genus": {"RL-B": {sc: by_genus(self.unqualified) for sc in set(scopes.values()) | {self.scope}}}}

    def ctx(self, **kw):
        groups = {g: list(us) for g, us in self.units.items()}
        strata = {}
        for g, us in groups.items():
            for u in us:
                strata.setdefault(g, {}).setdefault(u["stratum"], {"N": 0, "n_planned": 0})
                strata[g][u["stratum"]]["N"] += 1
        frames = {"doc": {"groups": {g: {"strata": st, "info": {}} for g, st in strata.items()}}, "groups": groups}
        rlq = kw.pop("rlq", None) or self.rlq(kw.pop("strata_scopes", None), kw.pop("pairs_ok", True))
        return E.Context(dom, pre, frames, self.sample, self.key, self.gold, rlq, **kw)


RAG = R["H1"]["class"]
WH = [n for n in dom.target_names if n != RAG][0]
ANS_T = {"answer": RAG}
OTHER = {"answer": "other", "box_ok": "yes"}
UNSURE = {"answer": "unsure", "box_ok": ""}


def ans(k, n, yes=None, no=None):
    return [dict(yes or ANS_T)] * k + [dict(no or OTHER)] * (n - k)


def verdict_of(fn, b, **kw):
    return fn(b.ctx(**kw))["verdict"]


print("H1")


def h1_world(k_rag, k_wh, level="species", h1pre=None, unq=()):
    b = Builder(level=level, unqualified_genera=unq)
    b.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), 80, ans(k_rag, 60, {"answer": RAG}), label=RAG)
    b.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, WH), 80, ans(k_wh, 60, {"answer": WH}), label=WH)
    b.stratum("G2/source=%s/label=%s/fail=none" % (W.LU, RAG), 30, ans(0, 20), label=RAG, source=W.LU)
    geo = {"h1_pre": {W.ND: {"pass": True if h1pre is None else h1pre, "why": ""}}}
    return b, geo


b, geo = h1_world(60, 60)
check("H1 supported (every rejected label right)", E.h1(b.ctx(rel_geo=geo))["verdict"] == "supported")
check("H1 waits for H1-pre (not evaluated without the geometry record)", E.h1(b.ctx())["verdict"] == "not_evaluated")
b, geo = h1_world(0, 0)
check("H1 falsified (none right)", E.h1(b.ctx(rel_geo=geo))["verdict"] == "falsified")
b, geo = h1_world(51, 60)
check("H1 inconclusive (the named class at 0.85, lower bound below)",
      E.h1(b.ctx(rel_geo=geo))["verdict"] == "inconclusive")
b, geo = h1_world(60, 60, level="genus")
check("H1 bounded (labeller qualified at genus only)", E.h1(b.ctx(rel_geo=geo))["verdict"] == "bounded")
b, geo = h1_world(60, 60, h1pre=False)
check("H1 not evaluated when H1-pre fails", E.h1(b.ctx(rel_geo=geo))["verdict"] == "not_evaluated")
h = E.h1(h1_world(60, 60)[0].ctx(rel_geo=h1_world(60, 60)[1]))
check("H1 reports per label, label right and box valid", set(h["parts"]["per_label"]) == {RAG, WH}
      and "label_right" in h["parts"] and "box_valid" in h["parts"])
check("H1 strata exclude sources that are not authoritative",
      all(W.LU not in s for s in E.h1_strata(h1_world(60, 60)[0].ctx())))

print("H2a / H2b")


def g1(frame, k, n=60, level="species", unsure=0):
    b = Builder(level=level)
    answers = ans(k, n - unsure) + [dict(UNSURE)] * unsure
    b.stratum("G1/frame=%s/status=%s/pred=%s" % (frame, "no_name" if frame == "noinfo" else "taxon_resolved", RAG),
              200, answers, label=dom.other["name"], pred=RAG, source=W.AN if frame == "noinfo" else W.ND,
              frame=frame)
    return b


check("H2a supported", verdict_of(lambda c: E.h2(c, "H2a"), g1("noinfo", 60)) == "supported")
check("H2a falsified", verdict_of(lambda c: E.h2(c, "H2a"), g1("noinfo", 0)) == "falsified")
check("H2a inconclusive", verdict_of(lambda c: E.h2(c, "H2a"), g1("noinfo", 30)) == "inconclusive")
check("H2a bounded", verdict_of(lambda c: E.h2(c, "H2a"), g1("noinfo", 60, level="genus")) == "bounded")
check("both-assignments rule: 30 targets + 30 unsure is inconclusive",
      verdict_of(lambda c: E.h2(c, "H2a"), g1("noinfo", 30, unsure=30)) == "inconclusive")
check("... while 60 targets + 0 unsure is supported and 30 + 30 as yes would be",
      verdict_of(lambda c: E.h2(c, "H2a"), g1("noinfo", 60)) == "supported")
check("H2b supported (no target among named conflicts)", verdict_of(lambda c: E.h2(c, "H2b"), g1("named", 0))
      == "supported")
check("H2b falsified", verdict_of(lambda c: E.h2(c, "H2b"), g1("named", 60)) == "falsified")
check("H2b inconclusive", verdict_of(lambda c: E.h2(c, "H2b"), g1("named", 9)) == "inconclusive")
check("H2b bounded", verdict_of(lambda c: E.h2(c, "H2b"), g1("named", 0, level="genus")) == "bounded")
h = E.h2(g1("noinfo", 30).ctx(), "H2a")
check("H2 reports the relabel precision per stratum", len(h["parts"]["relabel_precision"]) == 1)
gen = g1("noinfo", 0)
gen.gold = [dict(g, answer="other", answer_level="genus", answer_taxon="Amaranthus") for g in gen.gold]
h = E.h2(gen.ctx(), "H2a")
check("a genus answer in an unsure genus is unsure (taken both ways)",
      h["under_both_unsure"]["as_yes"]["interval"][0] > 0.9 and h["under_both_unsure"]["as_no"]["interval"][1] < 0.1)

print("H3a")
MH_TABLE = dom.raw["sources"]["card_resolvers"][W.MH]["class_table"]
up12, up5 = sorted(R["H3a"]["classes"], key=int)[::-1][0], sorted(R["H3a"]["classes"], key=int)[0]
cls12 = [t["name"] for t in dom.targets if t["taxon"] == R["H3a"]["classes"][up12]][0]
cls5 = [t["name"] for t in dom.targets if t["taxon"] == R["H3a"]["classes"][up5]][0]


def h3a_world(k12, k5, matched=0.99, agree=1.0, level="species", n=100, unq=()):
    b = Builder(level=level, unqualified_genera=unq)
    b.stratum("G3/unit=c:%s|%s" % (W.MH, up12), 110, ans(k12, n, {"answer": cls12}), label=dom.other["name"],
              source=W.MH, status="numeric", src_id=up12)
    b.stratum("G3/unit=c:%s|%s" % (W.MH, up5), 110, ans(k5, n, {"answer": cls5, "answer_level": "genus",
                                                                "answer_taxon": "Ipomoea"}),
              label=dom.other["name"], source=W.MH, status="numeric", src_id=up5)
    geo = {"matches": {W.MH: {"matched_share": matched, "chosen": "identity", "confirmed": True,
                              "alignments": [{"name": "identity", "agreement_a": agree, "agreement_b": agree},
                                             {"name": "plus1", "agreement_a": 0.01, "agreement_b": 0.02}]}},
           "h3a_exact": {"pass": matched >= 0.95 and agree >= 0.99}}
    return b.ctx(rel_geo=geo)


check("H3a class table has the upstream ids", up12 in MH_TABLE and up5 in MH_TABLE)
check("H3a supported (geometry exact, both purities high)", E.h3a(h3a_world(100, 100))["verdict"] == "supported")
check("H3a falsified (agreement < 0.99 under every alignment)",
      E.h3a(h3a_world(100, 100, agree=0.9))["verdict"] == "falsified")
check("H3a falsified (a purity lower bound < 0.6)", E.h3a(h3a_world(100, 40))["verdict"] == "falsified")
check("H3a inconclusive (geometry match 0.90 < 0.95, nothing falsified)",
      E.h3a(h3a_world(100, 100, matched=0.90))["verdict"] == "inconclusive")
check("H3a inconclusive (purity 0.75: lb >= 0.6, point < 0.8)", E.h3a(h3a_world(75, 100))["verdict"] == "inconclusive")
check("H3a bounded (genus-only labeller; Sicklepod needs species)",
      E.h3a(h3a_world(100, 100, level="genus"))["verdict"] == "bounded")

print("H3b")


def h3b_world(ks, level="species"):
    b = Builder(level=level)
    for j, k in enumerate(ks):
        b.stratum("G3/unit=k:%s|*|%d" % (W.AN, j), 120, ans(k, 60), label=dom.other["name"], source=W.AN,
                  status="no_name", src_id="*")
    b.stratum("G3/unit=c:%s|%s" % (W.MH, up12), 110, ans(60, 60), label=dom.other["name"], source=W.MH,
              status="numeric", src_id=up12)
    return b


check("H3b supported (a unit's lower bound >= 0.5)", verdict_of(E.h3b, h3b_world([0, 60])) == "supported")
check("H3b falsified (every upper bound < 0.3)", verdict_of(E.h3b, h3b_world([0, 0])) == "falsified")
check("H3b inconclusive", verdict_of(E.h3b, h3b_world([0, 20])) == "inconclusive")
check("H3b bounded", verdict_of(E.h3b, h3b_world([0, 60], level="genus")) == "bounded")
check("H3b leaves out the H3a source", len(E.h3b(h3b_world([0, 0]).ctx())["parts"]["units"]) == 2)

print("H4")


def h4_world(k_named, level="species", k_noinfo=0):
    b = Builder(level=level)
    b.stratum("G4/frame=named/argmax_target=no/band=1", 1000, ans(k_named, 200), label=dom.other["name"],
              source=W.ND, frame="named", pi=0.2)
    b.stratum("G4/frame=noinfo/argmax_target=no/band=1", 500, ans(k_noinfo, 60), label=dom.other["name"],
              source=W.AN, frame="noinfo", pi=0.12)
    return b


check("H4 supported (named ub < 0.02, no-information ub < 0.10)", verdict_of(E.h4, h4_world(0)) == "supported")
check("H4 falsified (named lb >= 0.02)", verdict_of(E.h4, h4_world(20)) == "falsified")
check("H4 inconclusive", verdict_of(E.h4, h4_world(2)) == "inconclusive")
check("H4 falsified by the no-information frame alone (lb >= 0.10, named clean)",
      verdict_of(E.h4, h4_world(0, k_noinfo=30)) == "falsified")
check("H4 bounded", verdict_of(E.h4, h4_world(0, level="genus")) == "bounded")

print("H5a / H5b / H12")
b = Builder()
check("H5a supported (7 < 100 boxes)", E.h5a(b.ctx(census={"h5a": {"dropped_twins_with_target_box_kept_lacks":
                                                                     {"boxes": 7}}}))["verdict"] == "supported")
check("H5a falsified (150 boxes)", E.h5a(b.ctx(census={"h5a": {"dropped_twins_with_target_box_kept_lacks":
                                                                 {"boxes": 150}}}))["verdict"] == "falsified")


def g5(k, ok=True):
    b = Builder()
    b.stratum("G5/source=%s/split=ood23/bits=0-2" % W.AN, 80, ans(k, 60, {"answer": "same"}, {"answer": "different"}),
              source=W.AN)
    return b.ctx(pairs_ok=ok)


check("H5b supported", E.h5b(g5(60))["verdict"] == "supported")
check("H5b falsified", E.h5b(g5(10))["verdict"] == "falsified")
check("H5b inconclusive", E.h5b(g5(50))["verdict"] == "inconclusive")
check("H5b not evaluated without a pair-qualified RL-B", E.h5b(g5(60, ok=False))["verdict"] == "not_evaluated")
for k, want in ((95, "supported"), (50, "falsified"), (75, "inconclusive")):
    check("H12 %s (%d of 100 known items present)" % (want, k),
          E.h12(Builder().ctx(census={"h12": {"present": k, "total": 100, "items": []}}))["verdict"] == want)

print("H0")


def h0_world(k_pos, n_pos, n_neg, share, level="species", unsure=0, h0b=True):
    b = Builder(level=level)
    truths = [0] * n_pos + [None] * n_neg
    answers = ([{"answer": RAG}] * k_pos + [{"answer": "other"}] * (n_pos - k_pos - unsure) +
               [{"answer": "unsure"}] * unsure + [{"answer": "other"}] * n_neg)
    b.g0(truths, answers, share)
    return b.ctx(rel_audit={"h0b": {"pass": h0b}, "h0c": {"pass": True, "class_accuracy": 0.9}})


check("H0 supported (the interval covers the planted share)", E.h0(h0_world(40, 40, 60, 0.4))["verdict"] == "supported")
check("H0 falsified (a labeller that misses every target)", E.h0(h0_world(0, 40, 60, 0.4))["verdict"] == "falsified")
check("H0 falsified when (b) fails", E.h0(h0_world(40, 40, 60, 0.4, h0b=False))["verdict"] == "falsified")
check("H0 inconclusive (unsure answers decide coverage)",
      E.h0(h0_world(25, 40, 60, 0.4, unsure=15))["verdict"] == "inconclusive")
check("H0 bounded (genus-only labeller)", E.h0(h0_world(40, 40, 60, 0.4, level="genus"))["verdict"] == "bounded")
a = E.audit_from_context(h0_world(0, 40, 60, 0.4), testing=True)
check("H0 falsified sets the stop", a["stop"] == {"rule": "H0 falsified", "at": "F8"})
a = E.audit_from_context(h0_world(40, 40, 60, 0.4), testing=True)
check("H0 supported leaves no stop", a["stop"] is None and a["valid"] is True)
b = Builder()
b.g0([0] * 40 + [None] * 60, [{"answer": RAG}] * 40 + [{"answer": "other"}] * 60, 0.4)
a = E.audit_from_context(b.ctx(), testing=True)
check("H0 not passed (no relation audit: not evaluated) also stops at F8 (F8 needs H0 to pass)",
      a["hypotheses"]["H0"]["verdict"] == "not_evaluated" and a["stop"] == {"rule": "H0 not passed (not_evaluated)",
                                                                           "at": "F8"}, a["stop"])
h7 = E.h7(Builder().ctx(jq={"h7": {"J-zs": {"predicted": "fail", "qualified": False},
                                   "J-knn2": {"predicted": "qualify", "qualified": False}}}))
check("H7 scores each judge's own prediction", h7["verdict"] == "reported" and
      h7["parts"]["J-zs"]["verdict"] == "supported" and h7["parts"]["J-knn2"]["verdict"] == "falsified", h7["parts"])
check("every hypothesis is present with a known verdict", set(a["hypotheses"]) >= {
    "H0", "H1", "H2a", "H2b", "H3a", "H3b", "H4", "H5a", "H5b", "H6", "H7", "H8", "H9", "H9'", "H10", "H11", "H12"}
      and all(h["verdict"] in E.VERDICTS for h in a["hypotheses"].values()))
check("rule texts are copied from the prereg", a["hypotheses"]["H1"]["rule"] == json.dumps(H["H1"], sort_keys=True))

print("H9' and H11")


def cell(lb, ub, status="evaluated"):
    return {"join": "strict", "admission": "image", "thresholds": "in_domain", "evidence": "on", "boxes": 10,
            "expected_true": {"estimate": lb, "lb": lb, "ub": ub}, "status": status, "uncovered_boxes": 0}


c0 = Builder().ctx()
check("H9' supported (best lb >= 4098)", E.h9_prime(c0, [cell(4100, 5000), cell(10, 20)])["verdict"] == "supported")
check("H9' falsified (every ub < 3073.5)", E.h9_prime(c0, [cell(1000, 3000), cell(10, 20)])["verdict"] == "falsified")
check("H9' inconclusive", E.h9_prime(c0, [cell(2000, 5000)])["verdict"] == "inconclusive")
check("H9' not evaluated without an evaluated cell",
      E.h9_prime(c0, [cell(0, 0, status="not_evaluated")])["verdict"] == "not_evaluated")
check("H9' falsification reads the best cell's upper bound, as the prereg says ('best cell UB<3073.5')",
      E.h9_prime(c0, [cell(1000, 3000), cell(10, 5000)])["verdict"] == "falsified"
      and E.h9_prime(c0, [cell(1000, 3100), cell(10, 2000)])["verdict"] == "inconclusive")
c1 = Builder().ctx(da={"stage_forecast": {"S8": 0.5, "S9": 0.3, "S10": 0.2}})
rank = [{"stage": "S8", "recoverable_lb": 900.0}, {"stage": "S9", "recoverable_lb": 100.0}]
check("H11 supported (top-1 right, log-loss beats uniform)", E.h11(c1, rank, ["S8", "S9", "S10"])["verdict"]
      == "supported")
rank_s9 = [{"stage": "S9", "recoverable_lb": 900.0}, {"stage": "S8", "recoverable_lb": 100.0}]
check("H11 falsified (top-1 wrong)", E.h11(c1, rank_s9, ["S8", "S9", "S10"])["verdict"] == "falsified")
c2 = Builder().ctx(da={"stage_forecast": {"S8": 0.25, "S9": 0.25, "S10": 0.25, "S3": 0.25}})
check("H11 falsified (a uniform forecast: log-loss no better than uniform over 4)",
      E.h11(c2, rank, ["S8", "S9", "S10", "S3"])["verdict"] == "falsified")

print("PPI in a stratum and R12")
jq = {"judges": {"J-knn2": {"by_type": {"shifted_target": {"qualified": True, "precision_at_half": {"lb": 0.9},
                                                            "rescue": {"lb": 0.6}}}, "by_lab_scope": {}}}}
labels = dom.target_names + ["other"]


def ppi_world(mat_lab, scores_useful=True):
    b = Builder()
    units = b.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), 400, ans(30, 60), label=RAG,
                      allowed="J-knn2")
    P = {}
    for i, u in enumerate(units):
        row = np.zeros(len(labels))
        if scores_useful:
            row[labels.index(RAG)] = 0.95 if i < 30 else 0.02
            row[-1] = 1 - row[labels.index(RAG)]
        else:
            row[labels.index(RAG)] = 0.5
            row[-1] = 0.5
        P[int(u["crop_id"])] = row
    mats = {"J-knn2": {"all": {"source": [], "near_dup3": [], "provenance": [], "lab": [mat_lab]}}}
    return b.ctx(jq=jq, materials=mats, judge_scores=lambda j: {"labels": labels, "P": P})


c = ppi_world("src:elsewhere")
s = c.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), "H1", "target")
check("an informative allowed predictor is used (PPI narrower)", s["predictor"] == "J-knn2" and s["method"] == "ppi")
check("the PPI interval is not wider than the design interval",
      (s["raw"]["as_no"]["interval"][1] - s["raw"]["as_no"]["interval"][0]) <=
      (E.binom_interval(30, 60)[1] - E.binom_interval(30, 60)[0]))
b = Builder()
units_small = b.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), 400, ans(10, 20), label=RAG,
                        allowed="J-knn2")
P_small = {}
for i, u in enumerate(units_small):
    row = np.zeros(len(labels))
    row[labels.index(RAG)] = 0.95 if i < 10 else 0.02
    row[-1] = 1 - row[labels.index(RAG)]
    P_small[int(u["crop_id"])] = row
c = b.ctx(jq=jq, materials={"J-knn2": {"all": {"source": [], "near_dup3": [], "provenance": [],
                                               "lab": ["src:elsewhere"]}}},
          judge_scores=lambda j: {"labels": labels, "P": P_small})
s = c.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), "H1", "target")
check("PPI needs at least 30 labelled units in the stratum (20 here: no predictor)",
      s["predictor"] is None and s["method"] != "ppi", s["method"])
c = ppi_world("src:elsewhere", scores_useful=False)
s = c.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), "H1", "target")
check("a useless predictor is not used", s["predictor"] is None and s["method"] != "ppi")
c = ppi_world(dom.lab_of(W.ND))
c.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), "H1", "target")
check("a predictor sharing the stratum's lab is recorded as a calibration overlap",
      any(o.get("judge") == "J-knn2" for o in c.overlap))
a = E.audit_from_context(c, testing=True)
check("... and the audit is invalid (R12)", a["valid"] is False and a["calibration_overlap"])
b = Builder(scope="KT1+KT7")
b.stratum("G2/source=%s/label=%s/fail=none" % (W.LU, RAG), 50, ans(10, 20), label=RAG, source=W.LU)
b.sentinel("KT1", W.REF)
a = E.audit_from_context(b.ctx(), testing=True)
check("a labeller scope whose sentinels share a lab with a stratum invalidates the audit (R12)",
      a["valid"] is False and any(o.get("rl") == "KT1" for o in a["calibration_overlap"]))
b = Builder(scope="KT7")
b.stratum("G2/source=%s/label=%s/fail=none" % (W.LU, RAG), 50, ans(10, 20), label=RAG, source=W.LU)
b.sentinel("KT7", "kt7")
check("a disjoint scope keeps the audit valid", E.audit_from_context(b.ctx(), testing=True)["valid"] is True)

# ======================================================================== review fixes and guards
print("interval coverage where the design effect binds")
r = np.random.default_rng(99)
Npop = 3000
score = r.beta(0.6, 3.0, Npop)
# targets sit where the prioritised part rarely looks (low scores): large weights, n_eff well below n
ytrue = (r.random(Npop) < 0.25 * (score < np.quantile(score, 0.3))).astype(float)
theta = ytrue.mean()
p_prio = np.minimum(1.0, 150 * score / score.sum())
cover, deff_bound = 0, 0
for rep in range(400):
    uni = np.zeros(Npop, bool)
    uni[r.choice(Npop, 150, replace=False)] = True
    s = uni | (r.random(Npop) < p_prio)
    e = E.ht(ytrue[s], 1 - (1 - 150 / Npop) * (1 - p_prio[s]), Npop)
    cover += e["interval"][0] <= theta <= e["interval"][1]
    deff_bound += e["n_eff"] < e["n"]
check("HT interval coverage %.3f >= 0.93 on the dual design with a binding design effect (%d/400)"
      % (cover / 400.0, deff_bound), cover / 400.0 >= 0.93 and deff_bound > 300)
r = np.random.default_rng(11)
worst = []
for th, se_, sp_, n, sen, spn in ((0.3, 0.9, 0.95, 400, 30, 30), (0.5, 0.9, 0.9, 800, 20, 20),
                                  (0.1, 0.9, 0.95, 500, 25, 25)):
    lo_miss = hi_miss = applied = 0
    for i in range(300):
        k = r.binomial(n, th * se_ + (1 - th) * (1 - sp_))
        rg = E.rogan_gladen(k, n, r.binomial(sen, se_), sen, r.binomial(spn, sp_), spn, "funnel/test/rgcov/%d" % i,
                            draws=4000)
        if not rg["applied"]:
            continue      # flagged (Se + Sp - 1 < 0.7): the uncorrected estimate is reported by design
        applied += 1
        lo_miss += rg["interval"][0] > th
        hi_miss += rg["interval"][1] < th
    worst.append((lo_miss / float(applied), hi_miss / float(applied), applied))
check("melded Rogan-Gladen: each one-sided miss rate <= 0.03 (nominal 0.025) with uncertain Se and Sp %s" % worst,
      all(a <= 0.03 and b_ <= 0.03 and m >= 250 for a, b_, m in worst))
wr = E.webber_recall({"N": 100, "k": 50, "n": 50}, [{"N": 100, "n": 50, "k": 40}], "funnel/test/recall2", 20000)
check("Webber counts the sampled targets of a discard stratum: median %.3f near the plug-in 100 / 180"
      % wr["estimate"], abs(wr["plug_in"] - 100 / 180.0) < 1e-12 and abs(wr["estimate"] - 100 / 180.0) < 0.03)

print("genus stop rule (contract §10)")
RAG_GENUS = dom.target(RAG)["taxon"].split()[0]
WH_GENUS = dom.target(WH)["taxon"].split()[0]
b, geo = h1_world(60, 60, unq={RAG_GENUS})
check("H1 bounded when the H1 class's genus is not qualified at species level",
      E.h1(b.ctx(rel_geo=geo))["verdict"] == "bounded")
b, geo = h1_world(60, 60, unq={WH_GENUS})
check("H1 bounded when another H1 label's genus is not qualified", E.h1(b.ctx(rel_geo=geo))["verdict"] == "bounded")
b, geo = h1_world(60, 60, unq={g for g in GENERA if g not in (RAG_GENUS, WH_GENUS)})
check("a genus H1 is not about does not bound it", E.h1(b.ctx(rel_geo=geo))["verdict"] == "supported")
b, geo = h1_world(60, 60)
rq = b.rlq()
del rq["by_genus"]["RL-B"][b.scope][RAG_GENUS]
check("a genus with no per-genus record is not qualified (fail closed)",
      E.h1(b.ctx(rel_geo=geo, rlq=rq))["verdict"] == "bounded")
G12 = R["H3a"]["classes"][up12].split()[0]
G5_ = R["H3a"]["classes"][up5].split()[0]
check("H3a bounded when the species-level card class's genus is not qualified",
      E.h3a(h3a_world(100, 100, unq={G12}))["verdict"] == "bounded")
check("H3a is not bounded by the genus-rank card class's genus (it needs genus level only)",
      E.h3a(h3a_world(100, 100, unq={G5_}))["verdict"] == "supported")

print("H5b reads near-evaluation pairs only")
b = Builder()
ood_sid = "G5/source=%s/split=ood23/bits=0-2" % W.AN
b.stratum(ood_sid, 80, ans(40, 60, {"answer": "same"}, {"answer": "different"}), source=W.AN)
b.stratum("G5/source=%s/split=dup/bits=0-2" % W.ND2, 400, ans(60, 60, {"answer": "same"}, {"answer": "different"}),
          source=W.ND2)
h = E.h5b(b.ctx())
check("exact-dup twins are not pooled into H5b (40 of 60 near-eval pairs the same: falsified)",
      h["verdict"] == "falsified", h["why"])
check("... they are reported apart", h["parts"]["strata"] == [ood_sid] and "dup" in h["parts"]["other_guard_pairs"])

print("H11: tied stages and the forecast's validity")
rank_tie = [{"stage": "S7", "recoverable_lb": 800.0}, {"stage": "S7b", "recoverable_lb": 800.0},
            {"stage": "S8", "recoverable_lb": 100.0}]
rec4 = ["S7", "S7b", "S8", "S9"]
h = E.h11(Builder().ctx(da={"stage_forecast": {"S7b": 0.5, "S7": 0.1, "S8": 0.2, "S9": 0.2}}), rank_tie, rec4)
check("stages tied on the largest lower bound are one outcome: a top-1 naming either is right",
      h["verdict"] == "supported" and h["parts"]["top_stages"] == ["S7", "S7b"], h["why"])
check("... scored on the forecast's mass on the tied set against |set| / K",
      abs(h["parts"]["p_top"] - 0.6) < 1e-12 and abs(h["parts"]["uniform_log_loss"] - math.log(2)) < 1e-12)
h = E.h11(Builder().ctx(da={"stage_forecast": {s: 1.0 for s in rec4}}), rank_tie, rec4)
check("a forecast that does not sum to 1 is falsified, not scored", h["verdict"] == "falsified" and "sums to" in h["why"])
check("a DA record without a forecast (an invalid reply) is falsified",
      E.h11(Builder().ctx(da={"stage_forecast": None}), rank_tie, rec4)["verdict"] == "falsified")
check("a forecast naming a stage that is not recoverable is falsified",
      E.h11(Builder().ctx(da={"stage_forecast": {"S7": 0.5, "S2": 0.5}}), rank_tie, rec4)["verdict"] == "falsified")

print("joint G2v/G2a items, label frequency and funnel recall")
b = Builder()
v_units = b.stratum("G2v/source=%s" % W.ND, 10, [{"answer": RAG}] * 10, label=RAG, pi=1.0)
a_sid = "G2a/source=%s" % W.ND
b.units["G2a"] = [dict(u, stratum=a_sid) for u in v_units]
b.stratum("G1/frame=noinfo/status=no_name/pred=%s" % RAG, 10, [{"answer": RAG}] * 10, label=dom.other["name"],
          pred=RAG, frame="noinfo", pi=1.0)
ctx = b.ctx()
check("a G2a stratum reads the items drawn under G2v for units in both frames",
      ctx.stratum(a_sid, "H1", "target")["n_labelled"] == 10)
lf = E.label_frequency(ctx)
check("label frequency counts each item once (10 join-target of 20 confirmed: c = 0.5)",
      abs(lf[W.ND]["estimate"] - 0.5) < 1e-12 and lf[W.ND]["n"] == 20, lf)
FL_ROLES = {"stages": [{"id": "S6", "role": "size"}, {"id": "S8", "role": "target_check"},
                       {"id": "S10", "role": "image_rule"}, {"id": "S12", "role": "evidence"}]}


def ref_admitted(n):
    return [{"unit": "box", "id": "b:lu%d#0" % i, "source": W.LU, "lab": dom.lab_of(W.LU),
             "label": dom.class_id(RAG), "path": {"S8": "verified", "S10": "admitted"}} for i in range(n)]


def calib(prec):
    return {"copies": {"current_join": {"overall": {"verdicts": {"verified": 1000}, "verified_precision": prec}}}}


b = Builder()
b.stratum("G1/frame=noinfo/status=no_name/pred=%s" % RAG, 10, [{"answer": RAG}] * 10, label=dom.other["name"],
          pred=RAG, frame="noinfo", pi=1.0, source=W.LU)
led30 = ref_admitted(30)
lf = E.label_frequency(b.ctx(ledger_rows=lambda: iter(led30), funnel_ledger=FL_ROLES, calib=calib(0.9)))
check("the reference lab's admitted boxes (in no frame) enter at the in-domain precision: c = 27 / 37",
      abs(lf[W.LU]["estimate"] - 27 / 37.0) < 1e-12 and lf[W.LU]["admitted_unsampled"] == 30, lf)
lf = E.label_frequency(b.ctx(ledger_rows=lambda: iter(led30), funnel_ledger=FL_ROLES))
check("... and without that precision the source gets no estimate, with the reason",
      lf[W.LU]["estimate"] is None and "calibration" in lf[W.LU]["why"], lf)
b = Builder()
b.stratum("G4/frame=noinfo/argmax_target=no/band=1", 1000, [{"answer": RAG}] * 10 + [dict(OTHER)] * 90,
          label=dom.other["name"], source=W.AN, frame="noinfo", pi=0.1)
led100 = ref_admitted(100)
fr_ = E.funnel_recall(b.ctx(ledger_rows=lambda: iter(led100), funnel_ledger=FL_ROLES, calib=calib(1.0)))
check("funnel recall counts the targets hidden among other_ok boxes: plug-in 100 / (100 + 1000 x 0.1)",
      abs(fr_["plug_in"] - 0.5) < 1e-9, fr_)

print("Horvitz-Thompson strata, labeller blocks, levels")
b = Builder()
sid4 = "G4/frame=named/argmax_target=yes/band=3"
b.stratum(sid4, 1000, [{"answer": RAG}] * 10 + [dict(OTHER)] * 30, label=dom.other["name"], source=W.ND,
          frame="named")
for i, row in enumerate([x for x in b.sample if x["stratum"] == sid4]):
    row["pi"] = repr(0.05 if i < 10 else 0.5)
s4 = b.ctx().stratum(sid4, "H4", "target")
check("a G4 stratum is Horvitz-Thompson: prioritised items weigh 1/pi (0.2, not the sample share 0.25)",
      s4["method"] == "ht" and abs(s4["raw"]["as_no"]["theta"] - 0.2) < 1e-12, s4["raw"]["as_no"])
b = Builder(se=(90, 100), sp=(95, 100))
sidq = "G2/source=%s/label=%s/fail=none" % (W.ND, RAG)
b.stratum(sidq, 200, ans(30, 60), label=RAG)
rq = b.rlq()
for sc_ in rq["backends"]["RL-B"]:
    rq["backends"]["RL-B"][sc_]["species"] = dict(rq["backends"]["RL-B"][sc_]["species"], qualified=False)
sq = b.ctx(rlq=rq).stratum(sidq, "H1", "target")
check("an Se/Sp block not qualified in the stratum's scope is not applied",
      sq["rogan_gladen"]["applied"] is False and sq["rogan_gladen"]["flag"] == "not_qualified_in_scope"
      and sq["estimate"] == 0.5, sq["rogan_gladen"])
cg = Builder().ctx()
MG = [t["name"] for t in dom.targets if t.get("rank") == "genus"][0]
check("at genus level a genus-rank target answer is a target",
      cg.event("target", {"answer": MG, "box_ok": "yes"}, None, "genus") == 1)
check("at genus level a species-rank target answer is unsure",
      cg.event("target", {"answer": RAG, "box_ok": "yes"}, None, "genus") is None)
check("at plant level no answer establishes a target; not-a-plant is a no",
      cg.event("target", {"answer": RAG, "box_ok": "yes"}, None, "plant") is None
      and cg.event("target", {"answer": "non_object", "box_ok": "no"}, None, "plant") == 0)
b = Builder(scope="KT4+KT7")
b.stratum("G2/source=%s/label=%s/fail=none" % (W.LU, RAG), 50, ans(10, 20), label=RAG, source=W.LU)
ov = E.rl_overlap(b.ctx(), {})
check("a labeller scope holding a claimed set is an overlap (R12, circularity)",
      any(o.get("rl") == "KT4" and "claimed_by" in o.get("shared", {}) for o in ov), ov)

print("stage screening")
FL_S8 = {"stages": [{"id": "S8", "role": "target_check", "recoverable": True}]}


def stage_world(N, k, n=60):
    b_ = Builder()
    b_.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), N, ans(k, n), label=RAG)
    return b_.ctx(funnel_ledger=FL_S8, census={"train_core_boxes_per_class": {x: 400 for x in dom.target_names}})


st = E.stages_block(stage_world(100000, 55))["S8"]
check("a stage with an FN rate far above 0.10 and a large recoverable count is suspect",
      st["suspect"] is True and st["p_holm"] < 0.025, st)
st = E.stages_block(stage_world(100000, 6))["S8"]
check("an FN rate at the null 0.10 is not suspect, however large the stage",
      st["suspect"] is False and st["recoverable"]["interval"][0] >= 500, st)
st = E.stages_block(stage_world(300, 55))["S8"]
check("a high FN rate on a stage too small to reach R_min is not suspect",
      st["suspect"] is False and st["recoverable"]["interval"][0] < 500, st)
st = E.stages_block(Builder().ctx(funnel_ledger=FL_S8))
check("no stage without strata", st == {})
b = Builder()
b.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), 1000, ans(55, 60), label=RAG)
st = E.stages_block(b.ctx(funnel_ledger=FL_S8, census={}))["S8"]
check("a class missing from the census's reference box counts gets no R_min, not a default",
      st["by_class"][RAG]["r_min"] is None and st["by_class"][RAG]["screen"] is None, st["by_class"])

b = Builder(se=(80, 100), sp=(95, 100))      # Rogan-Gladen would move 1/3 to 0.378
nsid = "G1/frame=named/status=taxon_resolved/pred=%s" % RAG
isid = "G1/frame=noinfo/status=no_name/pred=%s" % RAG
b.stratum(nsid, 100, ans(0, 30), label=dom.other["name"], pred=RAG, frame="named")
b.stratum(isid, 100, ans(20, 30), label=dom.other["name"], pred=RAG, frame="noinfo", source=W.AN)
rq = b.rlq()
rq["primary"]["H2b"] = dict(rq["primary"]["H2b"], backend="RL-A")
rq["backends"]["RL-A"] = rq["backends"]["RL-B"]
rq["by_genus"]["RL-A"] = rq["by_genus"]["RL-B"]
named_items = {x["item_id"] for x in b.sample if x["stratum"] == nsid}
b.gold += [dict(g, backend="RL-A") for g in b.gold if g["item_id"] in named_items]
ctx = b.ctx(rlq=rq, funnel_ledger={"stages": [{"id": "S9", "role": "other_check", "recoverable": True}]})
check("G1's named strata are answered by H2b's primary labeller, its no-information strata by H2a's",
      ctx.stratum(nsid, ctx.hyp_of(nsid), "target")["labeller"] == "RL-A"
      and ctx.stratum(isid, ctx.hyp_of(isid), "target")["labeller"] == "RL-B")
st9 = E.stages_block(ctx)["S9"]
check("a stage aggregate over strata of different labellers takes no single labeller's Se/Sp",
      st9["mixed_labellers"] is True and abs(st9["fn_rate"]["estimate"] - (100 * 0 + 100 * 20 / 30.0) / 200.0) < 1e-12
      and st9["unsure_as_yes"]["rg"]["flag"] == "mixed_labellers", st9)
d18 = {d["stratum"]: d for d in E.d18_inputs(ctx, {"S9": st9})}
check("D18 inputs read each G1 stratum through its own hypothesis's labeller",
      d18[nsid]["fn_lb"] == ctx.stratum(nsid, "H2b", "target")["interval"][0], d18[nsid])

print("H9: a mapped box takes its class unit's precision")
b = Builder()
g3sid = "G3/unit=c:%s|%s" % (W.MH, up12)
mh_units = b.stratum(g3sid, 40, [{"answer": cls12}] * 40, label=dom.other["name"], source=W.MH, status="numeric",
                     src_id=up12, pi=1.0)
g4sid = "G4/frame=noinfo/argmax_target=yes/band=3"
b.stratum(g4sid, 40, [dict(OTHER)] * 20, label=dom.other["name"], source=W.AN, frame="noinfo", pi=0.5)
b.units["G4"] = b.units["G4"] + [dict(u, stratum=g4sid) for u in mh_units]
nd_units = b.stratum("G2a/source=%s" % W.ND, 30, [{"answer": RAG}] * 10 + [dict(OTHER)] * 10, label=RAG,
                     pi=20 / 30.0)
# a card class whose unit fails the class gate (purity 10 of 40): its map is never applied
b.stratum("G3/unit=c:%s|%s" % (W.MH, up5), 40, [{"answer": cls5}] * 10 + [dict(OTHER)] * 30,
          label=dom.other["name"], source=W.MH, status="numeric", src_id=up5, pi=1.0)
rows_h9 = [{"unit": "box", "id": u["unit_id"], "key": u["image_key"], "source": W.MH, "lab": u["lab"],
            "label": dom.other["id"], "src_id": up12, "pred": dom.class_id(cls12), "p": 0.9, "cos": 0.9,
            "path": {"S6": "embedded", "S8": "n/a", "S12": "evidenced"}} for u in mh_units]
rows_h9 += [{"unit": "box", "id": u["unit_id"], "key": u["image_key"], "source": W.ND, "lab": u["lab"],
             "label": dom.class_id(RAG), "src_id": "8", "pred": dom.class_id(RAG), "p": 0.9, "cos": 0.9,
             "path": {"S6": "embedded", "S8": "verified", "S12": "not_evidenced" if i < 5 else "evidenced"}}
            for i, u in enumerate(nd_units)]
# an unverified target box in the image of the last ND box blocks that image under image admission
rows_h9.append({"unit": "box", "id": "b:%s#9" % nd_units[-1]["image_key"], "key": nd_units[-1]["image_key"],
                "source": W.ND, "lab": nd_units[-1]["lab"], "label": dom.class_id(RAG), "src_id": "8",
                "pred": dom.other["id"], "p": 0.9, "cos": 0.9,
                "path": {"S6": "embedded", "S8": "conflict", "S12": "evidenced"}})
ctx9 = b.ctx(ledger_rows=lambda: iter(rows_h9), funnel_ledger=FL_ROLES,
             verifier_thresholds={"tau_p": {x: 0.5 for x in dom.class_names},
                                  "sigma": {x: 0.8 for x in dom.class_names}},
             class_maps={"proposals": [{"source": W.MH, "src_id": up12, "map_to": cls12, "via": "card+geometry",
                                        "status": "proposed"},
                                       {"source": W.MH, "src_id": up5, "map_to": cls5, "via": "card+geometry",
                                        "status": "proposed"}]})
cells9, info9 = E.h9_grid(ctx9, False, True)


def cell_of(join):
    return [c for c in cells9 if c["join"] == join and c["thresholds"] == "in_domain" and c["admission"] == "box"
            and c["evidence"] == "off"][0]


check("the card join admits the mapped boxes as targets", cell_of("card")["boxes"] == 70 and
      cell_of("strict")["boxes"] == 30, (cell_of("card")["boxes"], cell_of("strict")["boxes"]))
check("mapped boxes take their class unit's precision, not the pooled other_ok stratum's (lb >= 0.85 x 40)",
      cell_of("card")["expected_true"]["lb"] >= 0.85 * 40, cell_of("card"))
check("a stratum whose precision lower bound is below 0.85 adds nothing to the lower bound, but to the upper",
      cell_of("strict")["expected_true"]["lb"] == 0 and cell_of("strict")["expected_true"]["ub"] > 0,
      cell_of("strict"))
check("only the card map whose class unit passes the class gate (purity lb >= 0.8) is applied",
      info9["maps"]["card"] == 1, info9["maps"])
strict_on = [c for c in cells9 if c["join"] == "strict" and c["thresholds"] == "in_domain" and c["admission"] == "box"
             and c["evidence"] == "on"][0]
strict_img = [c for c in cells9 if c["join"] == "strict" and c["thresholds"] == "in_domain"
              and c["admission"] == "image" and c["evidence"] == "off"][0]
check("the evidence axis drops boxes whose source has no evidence (25 of 30)", strict_on["boxes"] == 25, strict_on)
check("image admission drops a verified box in an image with an unverified target box (29 of 30)",
      strict_img["boxes"] == 29, strict_img)
check("shift-threshold cells are not evaluated unless H1 and H3a are supported",
      all(c["status"] == "not_evaluated" for c in cells9 if c["thresholds"] == "shift"))

print("Learn-then-Test thresholds come from other labs' truth")
b = Builder()
ndx = b.stratum("G2/source=%s/label=%s/fail=none" % (W.ND, RAG), 150, [{"answer": RAG}] * 150, label=RAG)
lux = b.stratum("G2/source=%s/label=%s/fail=none" % (W.LU, RAG), 150, [dict(OTHER)] * 150, label=RAG, source=W.LU)
boxes_ltt = [(u["unit_id"], u["image_key"], u["source"], u["lab"], dom.class_id(RAG), "8", dom.class_id(RAG), 0.9,
              0.9, True, True, False) for u in ndx + lux]
ctxl = b.ctx()
tau0 = {cid: 0.5 for cid in range(len(dom.class_names))}
shift = E._shift_thresholds(ctxl, boxes_ltt, tau0)
check("a lab whose own rejected labels are wrong gets its threshold from the other lab's correct ones (finite)",
      math.isfinite(shift[dom.lab_of(W.LU)][0][dom.class_id(RAG)]), shift[dom.lab_of(W.LU)][0][dom.class_id(RAG)])
check("... and a lab is never calibrated on its own gold (the other lab is all wrong: inf)",
      math.isinf(shift[dom.lab_of(W.ND)][0][dom.class_id(RAG)]))

# ======================================================================== end to end
print("end to end on the synthetic world")
from weed_optimizer_framework.tools.funnel import draw as DR  # noqa: E402
fd = TMP / "inc" / "funnel"
raw = json.loads(PREREG.read_text())
raw["sampling"]["groups"] = {"G0": [20, 10], "G1": [30, 10], "G2": [30, 10], "G2v": [8, 4], "G2a": [8, 4],
                             "G3": [60, 20], "G4": [40, 20], "G5": [20, 10], "sentinels": [400, None],
                             "KT7_sentinels": [150, None]}
raw["sampling"]["G4_uniform"] = 20
PREREG.write_text(json.dumps(raw, indent=1))
adapter = W.build_world(fd, dom, PREREG)
W.write_step1(F.STEP1_DIR, dom)
res = DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2, testing=True)
pre2 = D.load_prereg(PREREG)
sample = DR.load_sample(fd / "sample_v1.csv", pre2)
keyrows = DR.load_key(fd / "sample_v1_key.jsonl", pre2)
truth = {k["item_id"]: k for k in keyrows if "item_id" in k}
rows = {r["id"]: r for r in W.world_rows(dom)}
share = [k for k in keyrows if "planted" in k][0]["planted"]["share"]
GOLD_FIELDS = ("item_id", "unit_id", "unit", "group", "stratum", "labeller", "backend", "option", "answer",
               "answer_level", "answer_taxon", "box_ok", "is_sentinel", "kt", "truth", "truth_kind", "correct_species",
               "correct_genus", "correct_plant", "sheet_id", "position", "answers_sha256")


def truthful(r):
    """A labeller that tells the truth: KT items by their truth, pool boxes by
    their label (a target label is right; other-labelled boxes are other)."""
    k = truth.get(r["item_id"]) or {}
    if r["group"] in ("G5", "pair_sentinel"):
        return {"answer": "same" if k.get("pair_truth") == "same" else "different", "box_ok": ""}
    if k.get("truth") is not None and k.get("truth_kind") == "target":
        return {"answer": dom.class_name(int(k["truth"])), "box_ok": "yes"}
    if k.get("truth_kind") in ("attractor", "other"):
        return {"answer": "other", "box_ok": "yes"}
    return {"answer": W.hidden_truth(dom, rows.get(r["unit_id"])), "box_ok": "yes"}


gold = []
for r in sample:
    a = truthful(r)
    gold.append(dict({f: "" for f in GOLD_FIELDS}, item_id=r["item_id"], unit_id=r["unit_id"], unit=r["unit"],
                     group=r["group"], stratum=r["stratum"], labeller="RL-B:test", backend="RL-B",
                     answer_level="species", kt=r["kt"], is_sentinel=int(r["group"] == "sentinel"), **a))
F.write_csv_atomic(fd / "gold_v1.csv", GOLD_FIELDS, gold)
gsha = F.file_record(fd / "gold_v1.csv")["sha256"]
frames_doc = json.loads((fd / "frames_v1.json").read_text())
scopes = {}
for g, grp in frames_doc["groups"].items():
    for sid in grp["strata"]:
        scopes[sid] = "KT7"
blk = {"se": est(300, 300), "sp": est(300, 300), "qualified": True}
rlq = {"format": "funnel-rl-qualification/1", "gold_sha256": gsha, "prereg": pre2.record(),
       "backends": {"RL-B": {"KT7": {"species": blk, "genus": blk, "plant": blk}}},
       "levels": {"RL-B": {"KT7": "species"}}, "strata_scopes": scopes, "pairs": {"RL-B": blk},
       "by_genus": {"RL-B": {"KT7": by_genus()}},
       "primary": {h: {"backend": "RL-B", "level": "species", "scope": "KT7", "qualified": True}
                   for h in ("H0", "H1", "H2a", "H2b", "H3a", "H3b", "H4")}}
check("estimate refuses without rl_qualification.json",
      raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True), F.EstimateError))
F.write_json_atomic(fd / "rl_qualification.json", rlq)
check("estimate refuses without prospective_da.json",
      raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True), F.EstimateError))
F.write_json_atomic(fd / "prospective_da.json", {"format": "funnel-prospective-da/1",
                                                 "stage_forecast": {"S8": 0.4, "S9": 0.3, "S10": 0.1, "S7": 0.1,
                                                                    "S7b": 0.05, "S3": 0.05}})
F.write_json_atomic(fd / "relation_audit_v1.json", {"format": "funnel-relation-audit/1", "h0b": {"pass": True},
                                                    "h0c": {"pass": True, "class_accuracy": 1.0}})
F.write_json_atomic(fd / "relation_geometry_v1.json", {
    "format": "funnel-relation-geometry/1", "h3a_exact": {"pass": True, "why": ""},
    "h1_pre": {W.ND: {"pass": True, "why": ""}},
    "matches": {W.MH: {"matched_share": 0.99, "chosen": "identity", "confirmed": True,
                       "alignments": [{"name": "identity", "agreement_a": 1.0, "agreement_b": 1.0}]}}})
audit = E.evaluate(PREREG, fd, adapter, testing=True)
check("audit_v1.json and audit_v1.md written", (fd / "audit_v1.json").exists() and (fd / "audit_v1.md").exists())
check("audit header and format", audit["format"] == "funnel-audit/1" and audit["prereg"]["core_sha256"] ==
      pre2.core_sha256 and "sample_v1" in audit["inputs"])
check("audit valid, ledger fingerprint recorded", audit["valid"] is True and
      audit["ledger_fingerprint"] == json.loads((fd / "funnel_ledger.json").read_text())["fingerprint"])
h0a = audit["hypotheses"]["H0"]["parts"]["a"]
check("H0(a) covers the planted share %.3f with a truthful labeller" % share,
      h0a["verdict"] == "supported" and audit["hypotheses"]["H0"]["verdict"] == "supported", h0a)
check("stages are the recoverable stages with frames", {"S8", "S9", "S10", "S7", "S7b"} <= set(audit["stages"]))
check("the dedup stage is exact from census h5a", audit["stages"]["S3"]["exact"] and
      audit["stages"]["S3"]["recoverable"]["estimate"] == 7.0)
check("stage ranking sorted by recoverable lower bound",
      [x["recoverable_lb"] for x in audit["stage_ranking"]] == sorted(
          [x["recoverable_lb"] for x in audit["stage_ranking"]], reverse=True))
check("d18 inputs carry kinds", {d["kind"] for d in audit["d18_inputs"]} <= set(E.D18_KIND.values())
      and audit["d18_inputs"])
check("24 grid cells", len(audit["h9_grid"]) == 24)
check("H1 and H3a supported on the world (card classes pure, rejected labels right)",
      audit["hypotheses"]["H1"]["verdict"] == "supported" and audit["hypotheses"]["H3a"]["verdict"] == "supported",
      (audit["hypotheses"]["H1"]["why"], audit["hypotheses"]["H3a"]["why"]))
check("so every grid cell is evaluated, shift thresholds included",
      all(c["status"] == "evaluated" for c in audit["h9_grid"]))
strict = [c for c in audit["h9_grid"] if c["join"] == "strict" and c["thresholds"] == "in_domain"
          and c["evidence"] == "off"]
check("box admission yields at least what image admission yields",
      [c["boxes"] for c in strict if c["admission"] == "box"][0] >=
      [c["boxes"] for c in strict if c["admission"] == "image"][0] > 0, strict)
card = [c for c in audit["h9_grid"] if c["join"] == "card" and c["thresholds"] == "in_domain"]
check("funnel recall estimated from the calibration counts", audit["funnel_recall"]["estimate"] is not None and
      0 < audit["funnel_recall"]["estimate"] <= 1, audit["funnel_recall"])
check("the cap stage counts the images the cap left out",
      audit["stages"]["S0"]["recoverable_images"] ==
      {W.MH: dom.raw["sources"]["card_image_counts"][W.MH] - 250})
check("shift cells not evaluated unless H1 and H3a are supported",
      all(c["status"] == "not_evaluated" for c in audit["h9_grid"] if c["thresholds"] == "shift")
      == (not (audit["hypotheses"]["H1"]["verdict"] == "supported" and
               audit["hypotheses"]["H3a"]["verdict"] == "supported")))
check("strata records carry both unsure assignments", all("as_no" in s["unsure"] for s in audit["strata"]))
check("the markdown lists every hypothesis", all(("| %s |" % h) in (fd / "audit_v1.md").read_text()
                                                 for h in audit["hypotheses"]))
again = E.evaluate(PREREG, fd, adapter, testing=True)
check("the audit is deterministic", F.strip_volatile(again) == F.strip_volatile(audit))
text = (fd / "sample_v1.csv").read_text()
(fd / "sample_v1.csv").write_text(text.replace("\n", "\n", 1) + "\n")
check("a changed sample refuses (StaleInput)", raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True),
                                                     F.StaleInput))
(fd / "sample_v1.csv").write_text(text)
g = (fd / "gold_v1.csv").read_text()
(fd / "gold_v1.csv").write_text(g + "\n")
check("gold not matching rl_qualification refuses (StaleInput)",
      raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True), F.StaleInput))
(fd / "gold_v1.csv").write_text(g)
lu_g2 = [s for s in scopes if s.startswith("G2/source=%s/" % W.LU)]
if lu_g2:
    rlq2 = copy.deepcopy(rlq)
    for s in lu_g2:
        rlq2["strata_scopes"][s] = "KT1+KT7"
    rlq2["backends"]["RL-B"]["KT1+KT7"] = rlq["backends"]["RL-B"]["KT7"]
    F.write_json_atomic(fd / "rl_qualification.json", rlq2)
    check("a stratum qualified on material of its own lab: audit written invalid, then refused (R12)",
          raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True), F.EstimateError)
          and json.loads((fd / "audit_v1.json").read_text())["valid"] is False)
    F.write_json_atomic(fd / "rl_qualification.json", rlq)
else:
    W.skip("R12 on the world", "no reference-lab G2 stratum was framed")
locked = PREREG.read_bytes()
ed = json.loads(locked)
ed["hypotheses"]["H1"]["supported"] = "LB>=0.50 overall and for %s" % RAG
PREREG.write_text(json.dumps(ed, indent=1))
check("a prereg edited outside its amendments after the lock refuses the estimate (StaleInput)",
      raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True), F.StaleInput))
PREREG.write_bytes(locked)
rq_other = copy.deepcopy(rlq)
rq_other["prereg"] = dict(rq_other["prereg"], core_sha256="0" * 64)
F.write_json_atomic(fd / "rl_qualification.json", rq_other)
check("a labeller qualification made under another prereg core refuses (StaleInput)",
      raises(lambda: E.evaluate(PREREG, fd, adapter, testing=True), F.StaleInput))
F.write_json_atomic(fd / "rl_qualification.json", rlq)
check("restored inputs evaluate again", E.evaluate(PREREG, fd, adapter, testing=True)["valid"] is True)

W.finish()
