"""Every audit number (contract §5.4, §6, §8.7; runner §4.16, §5.1.6).

The only producer of audit numbers: per-stratum proportions, stratified and
Horvitz-Thompson aggregates, prediction-powered estimates, reference-labeller
error correction, design effects, funnel recall, stage screening and the
hypothesis verdicts, written to audit_v1.json and audit_v1.md.

Estimators, as pinned by the runner:
  * binom_interval: Wilson for n >= 30, Jeffreys below; the lower end is the
    one-sided 97.5 % bound (two-sided 95 %).
  * clopper_pearson: exact, non-integer x and n allowed (Korn-Graubard).
  * stratified / combine: theta = sum W_h p_h, v = sum W_h^2 (1 - n_h/N_h)
    p_h (1 - p_h) / (n_h - 1) (n_h = 1: W_h^2 / 4), n_eff = theta (1 - theta)
    / v capped at sum n_h, interval clopper_pearson(n_eff theta, n_eff).
  * ht: theta = sum y/pi / N, v = sum (1 - pi) y^2 / pi^2 / N^2.
  * ppi_pp: PPI++ with power tuning, lambda = Cov(y, f) / ((1 + n/N) Var(f)).
  * rogan_gladen: the melded Monte Carlo interval of theta = (obs + Sp - 1) /
    (Se + Sp - 1), flagged and uncorrected when Se + Sp - 1 < 0.7.
  * cluster_bootstrap, webber_recall (the Monte Carlo median and its 2.5 /
    97.5 % quantiles; the plug-in ratio is reported beside it), holm,
    perm_test_one_sided.
The beta and binomial quantiles are implemented here (no scipy).

Verdicts use one-sided 97.5 % bounds (prereg estimators.decisions_one_sided).
Unsure and unparsed answers are taken both ways; a verdict holds only when
it holds under both. A hypothesis whose needed level the reference labeller
is not qualified at is "bounded". Nothing here names a domain: classes,
sources, lab groups and thresholds come from the domain config and the
prereg.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist

from . import (STEP1_DIR, EstimateError, FunnelError, StaleInput, check_prereg_core, check_records, file_record,
               header, json_text, read_csv, read_json, read_jsonl, seed, write_json_atomic, _atomic_write_bytes)
from . import domain as D

Z975 = NormalDist().inv_cdf(0.975)
LEVELS = ("species", "genus", "plant")
VERDICTS = ("supported", "falsified", "inconclusive", "bounded", "not_evaluated", "descriptive", "reported")
SEED_PREFIX = "funnel/v1"
RG_DRAWS = 20000
BOOT_B = 2000
RECALL_DRAWS = 20000
RG_MIN_INFORMATIVE = 0.7
PPI_MIN_N = 30
H9_JOINS = ("strict", "card", "card_taxonomy")
H9_ADMISSION = ("image", "box")
H9_THRESHOLDS = ("in_domain", "shift")
H9_EVIDENCE = ("on", "off")
LTT_LEVEL, LTT_DELTA, LTT_GRID = 0.95, 0.025, 20
STAGE_GROUPS = {"target_check": ("G2",), "other_check": ("G1",), "image_rule": ("G2v",),
                "join": ("G3",), "name_status": ("G3",)}
D18_KIND = {"G1": "other_predicted_target", "G2": "target_rejected", "G2v": "target_rejected",
            "G3": "uninformative_label_space"}
# the hypothesis whose primary labeller answers each group's strata
GROUP_HYP = {"G0": "H0", "G1": "H2a", "G2": "H1", "G2v": "H1", "G2a": "H1", "G3": "H3b", "G4": "H4",
             "G5": "H5b"}
ESTIMATION_GROUPS = ("G1", "G2", "G2v", "G2a", "G3", "G4", "G5")
JOINT_GROUPS = ("G2v", "G2a")
HT_GROUPS = ("G2v", "G2a", "G4")
PRECISION_ORDER = ("G2", "G2v", "G2a", "G1", "G4", "G3")
STRATUM_TYPE = {"G2": "shifted_target", "G2v": "shifted_target", "G2a": "shifted_target",
                "G1:named": "other_named", "G4:named": "other_named",
                "G1:noinfo": "other_noinfo", "G4:noinfo": "other_noinfo", "G3": "other_noinfo"}
# The per-stratum estimates the recovery gates read (recover.gates, BOX_EVENT / CLASS_EVENT /
# CLASS_BOX_EVENT), written beside each stratum's "target" share: "label" (the source label right, box
# valid) for the target-labelled groups, "pred" (the step-1 probe's predicted class right, box valid) for
# the other-labelled ones, and for a class unit "purity:<class>" and "label:<class>" of the one class its
# gates are for (recover.unit_classes). A gate is never read from the "target" share.
GATE_EVENTS = {"G2": ("label",), "G2v": ("label",), "G1": ("pred",), "G4": ("pred",)}
UNIT_GATE_EVENTS = ("purity:%s", "label:%s")
UNSURE_ANSWERS = ("unsure", "unparsed", "")
POSITIVE_PAIRS = ("same", "consecutive")


# ====================================================================== numerics
def _betacf(a, b, x):
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            return h
    raise EstimateError("the incomplete beta continued fraction did not converge (a=%g, b=%g, x=%g)" % (a, b, x))


def betainc(a, b, x):
    """The regularised incomplete beta function I_x(a, b). a = 0 (b = 0) is the
    point mass at 0 (at 1)."""
    a, b, x = float(a), float(b), float(x)
    if a < 0 or b < 0 or (a == 0 and b == 0):
        raise EstimateError("betainc needs a, b >= 0, not both 0 (a=%g, b=%g)" % (a, b))
    if x <= 0.0:
        return 1.0 if a == 0 else 0.0
    if x >= 1.0:
        return 1.0
    if a == 0:
        return 1.0
    if b == 0:
        return 0.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q, a, b):
    """The q quantile of Beta(a, b), by bisection on betainc to 1e-13."""
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise EstimateError("beta_ppf needs 0 <= q <= 1, got %r" % q)
    if a == 0:
        return 0.0
    if b == 0:
        return 1.0
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def binom_upper_tail(x, n, p0):
    """P(Binom(n, p0) >= x), continuous in x and n: I_p0(x, n - x + 1)."""
    x, n = float(x), float(n)
    if x <= 0:
        return 1.0
    if x > n:
        return 0.0
    return betainc(x, n - x + 1.0, p0)


def binom_interval(k, n, conf=0.95):
    """Wilson if n >= 30, else Jeffreys; (lo, hi) of a two-sided conf interval,
    so lo is the one-sided (1 + conf)/2 lower bound."""
    k, n = float(k), float(n)
    if n <= 0:
        return (0.0, 1.0)
    if k < -1e-12 or k > n + 1e-12:
        raise EstimateError("binom_interval needs 0 <= k <= n (k=%g, n=%g)" % (k, n))
    k = min(max(k, 0.0), n)
    alpha = 1.0 - conf
    if n >= 30:
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        p = k / n
        den = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / den
        half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
        lo = 0.0 if k <= 0 else max(0.0, center - half)
        hi = 1.0 if k >= n else min(1.0, center + half)
        return (lo, hi)
    lo = 0.0 if k <= 0 else beta_ppf(alpha / 2.0, k + 0.5, n - k + 0.5)
    hi = 1.0 if k >= n else beta_ppf(1.0 - alpha / 2.0, k + 0.5, n - k + 0.5)
    return (lo, hi)


def clopper_pearson(x, n, conf=0.95):
    """Exact interval; x and n may be non-integer (Korn-Graubard effective n)."""
    x, n = float(x), float(n)
    if n <= 0:
        return (0.0, 1.0)
    x = min(max(x, 0.0), n)
    alpha = 1.0 - conf
    lo = 0.0 if x <= 0 else beta_ppf(alpha / 2.0, x, n - x + 1.0)
    hi = 1.0 if x >= n else beta_ppf(1.0 - alpha / 2.0, x + 1.0, n - x)
    return (lo, hi)


def srs_var(k, n, N, fpc=True):
    """Design variance of a stratum proportion under srs (runner §5.1.6)."""
    n, N = float(n), float(N)
    if n <= 0:
        raise EstimateError("a stratum with no labelled unit has no variance")
    if fpc and N > 0 and n >= N:
        return 0.0
    if n == 1:
        return 0.25
    p = float(k) / n
    f = (1.0 - n / N) if (fpc and N > 0) else 1.0
    return f * p * (1.0 - p) / (n - 1.0)


def kg_n_eff(theta, v, n):
    """Korn-Graubard effective sample size theta (1 - theta) / v, capped at
    n. When theta is 0 or 1 (every labelled unit agrees; an HT total can
    also reach past 1 before clipping) the ratio says nothing about the
    design, and n is used, as Korn and Graubard (1998) do; without this a
    stratum whose every labelled unit is positive got the vacuous [0, 1]."""
    n = float(n)
    if v <= 0 or theta * (1.0 - theta) <= 0:
        return n
    return min(n, theta * (1.0 - theta) / v)


def combine(parts, conf=0.95):
    """Korn-Graubard aggregate of stratum estimates. parts: [{"N", "theta",
    "var", "n"}]; a part with n == 0 is uncovered: the interval spans it from 0
    to 1 (its weight is reported as uncovered_weight)."""
    parts = [p for p in parts if float(p["N"]) > 0]
    N = float(sum(float(p["N"]) for p in parts))
    if N <= 0:
        return {"estimate": None, "interval": [0.0, 1.0], "var": None, "n_eff": 0.0, "n": 0, "N": 0,
                "uncovered_weight": 1.0, "method": "korn_graubard_no_df_adjustment"}
    cov = [p for p in parts if float(p["n"]) > 0]
    w_e = sum(float(p["N"]) for p in parts if float(p["n"]) <= 0) / N
    Nc = sum(float(p["N"]) for p in cov)
    if Nc <= 0:
        return {"estimate": None, "interval": [0.0, 1.0], "var": None, "n_eff": 0.0, "n": 0, "N": N,
                "uncovered_weight": 1.0, "method": "korn_graubard_no_df_adjustment"}
    theta = sum(float(p["N"]) / Nc * float(p["theta"]) for p in cov)
    v = sum((float(p["N"]) / Nc) ** 2 * float(p["var"]) for p in cov)
    n = float(sum(float(p["n"]) for p in cov))
    theta = min(max(theta, 0.0), 1.0)
    n_eff = kg_n_eff(theta, v, n)
    lo, hi = clopper_pearson(n_eff * theta, n_eff, conf) if n_eff > 0 else (0.0, 1.0)
    if w_e > 0:
        lo, hi = lo * (1.0 - w_e), hi * (1.0 - w_e) + w_e
    return {"estimate": theta, "interval": [lo, hi], "var": v, "n_eff": n_eff, "n": int(round(n)), "N": N,
            "uncovered_weight": w_e, "method": "korn_graubard_no_df_adjustment"}


def stratified(strata, fpc=True, conf=0.95):
    """Stratified proportion over [{"N", "n", "k"}] (runner §5.1.6)."""
    parts = []
    for s in strata:
        N, n, k = float(s["N"]), float(s["n"]), float(s["k"])
        if N <= 0:
            continue
        if n > N:
            raise EstimateError("a stratum has n %g > N %g" % (n, N))
        if n <= 0:
            parts.append({"N": N, "theta": 0.0, "var": 0.0, "n": 0})
            continue
        parts.append({"N": N, "theta": k / n, "var": srs_var(k, n, N, fpc), "n": n})
    return combine(parts, conf)


def ht(y, pi, N, conf=0.95):
    """Horvitz-Thompson proportion over a frame of N units from a sample with
    inclusion probabilities pi."""
    y = [float(v) for v in y]
    pi = [float(v) for v in pi]
    if len(y) != len(pi):
        raise EstimateError("ht needs one inclusion probability per value")
    if any(not 0 < p <= 1 for p in pi):
        raise EstimateError("an inclusion probability is outside (0, 1]")
    N = float(N)
    if N <= 0:
        raise EstimateError("ht needs a frame size")
    t = sum(v / p for v, p in zip(y, pi))
    theta = t / N
    v = sum((1.0 - p) * v * v / (p * p) for v, p in zip(y, pi)) / (N * N)
    n = len(y)
    th = min(max(theta, 0.0), 1.0)
    n_eff = kg_n_eff(th, v, n)
    lo, hi = clopper_pearson(n_eff * th, n_eff, conf) if n_eff > 0 else (0.0, 1.0)
    return {"estimate": theta, "interval": [lo, hi], "var": v, "n_eff": n_eff, "n": n, "N": N, "method": "ht"}


def _mean(xs):
    return sum(xs) / float(len(xs))


def _var(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1.0)


def _cov(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1.0)


def ppi_pp(y_lab, f_lab, f_all, pi_lab=None, conf=0.95):
    """Prediction-powered mean with power tuning (PPI++): theta = lambda
    mean(f over all units) + mean over labelled units of (y - lambda f), with
    lambda = Cov(y, f) / ((1 + n/N) Var(f)) clipped to [0, 1]; normal interval.
    With pi_lab the labelled term is Horvitz-Thompson weighted."""
    y = [float(v) for v in y_lab]
    f = [float(v) for v in f_lab]
    fa = [float(v) for v in f_all]
    n, N = len(y), len(fa)
    if n == 0 or N == 0 or len(f) != n:
        raise EstimateError("ppi_pp needs labelled pairs and the predictor on every unit")
    var_f = _var(fa)
    lam = 0.0 if var_f <= 0 else _cov(y, f) / ((1.0 + n / float(N)) * var_f)
    lam = min(max(lam, 0.0), 1.0)
    r = [a - lam * b for a, b in zip(y, f)]
    if pi_lab is None:
        theta = lam * _mean(fa) + _mean(r)
        v = lam * lam * var_f / N + _var(r) / n
    else:
        pi = [float(p) for p in pi_lab]
        theta = lam * _mean(fa) + sum(ri / p for ri, p in zip(r, pi)) / N
        v = lam * lam * var_f / N + sum((1.0 - p) * ri * ri / (p * p) for ri, p in zip(r, pi)) / (N * N)
    z = NormalDist().inv_cdf(0.5 + conf / 2.0)
    half = z * math.sqrt(max(v, 0.0))
    return {"estimate": theta, "interval": [max(0.0, theta - half), min(1.0, theta + half)],
            "lambda": lam, "var": v, "n": n, "N": N, "method": "ppi"}


def _beta_draws(r, a, b, size):
    import numpy as np
    if a <= 0:
        return np.zeros(size)
    if b <= 0:
        return np.ones(size)
    return r.beta(a, b, size)


def rogan_gladen(k, n, se_k, se_n, sp_k, sp_n, seed_text, draws=RG_DRAWS, conf=0.95):
    """Prevalence corrected for the labeller's sensitivity and specificity,
    with the melded Monte Carlo interval (runner §5.1.6). Below
    Se + Sp - 1 = 0.7 the uncorrected estimate is returned, flagged."""
    import numpy as np
    k, n = float(k), float(n)
    se_k, se_n, sp_k, sp_n = float(se_k), float(se_n), float(sp_k), float(sp_n)
    if n <= 0:
        return {"estimate": None, "interval": [0.0, 1.0], "flag": "no_data", "applied": False}
    if se_n <= 0 or sp_n <= 0:
        lo, hi = binom_interval(k, n, conf)
        return {"estimate": k / n, "interval": [lo, hi], "flag": "no_se_sp", "applied": False}
    obs, se, sp = k / n, se_k / se_n, sp_k / sp_n
    if se + sp - 1.0 < RG_MIN_INFORMATIVE:
        lo, hi = binom_interval(k, n, conf)
        return {"estimate": obs, "interval": [lo, hi], "flag": "se_sp_low", "applied": False,
                "se": se, "sp": sp}
    point = min(max((obs + sp - 1.0) / (se + sp - 1.0), 0.0), 1.0)
    r = np.random.default_rng(seed(seed_text))
    a = (1.0 - conf) / 2.0
    o = _beta_draws(r, k, n - k + 1.0, draws)
    e = _beta_draws(r, se_k + 1.0, se_n - se_k, draws)
    s = _beta_draws(r, sp_k, sp_n - sp_k + 1.0, draws)
    den = e + s - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.where(den > 0, (o + s - 1.0) / den, 0.0)
    lo = float(np.quantile(np.clip(th, 0.0, 1.0), a))
    o = _beta_draws(r, k + 1.0, n - k, draws)
    e = _beta_draws(r, se_k, se_n - se_k + 1.0, draws)
    s = _beta_draws(r, sp_k + 1.0, sp_n - sp_k, draws)
    den = e + s - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.where(den > 0, (o + s - 1.0) / den, 1.0)
    hi = float(np.quantile(np.clip(th, 0.0, 1.0), 1.0 - a))
    return {"estimate": point, "interval": [min(lo, point), max(hi, point)], "flag": None, "applied": True,
            "se": se, "sp": sp, "draws": int(draws), "seed_text": seed_text}


def cluster_bootstrap(y, clusters, seed_text, B=BOOT_B):
    """Design effect and percentile interval of a mean from a bootstrap that
    resamples whole clusters (images, or sources across sources)."""
    import numpy as np
    y = np.asarray([float(v) for v in y])
    if len(y) < 2:
        return {"deff": None, "interval": None, "B": B, "clusters": len(set(clusters))}
    keys = sorted(set(clusters))
    idx = {c: i for i, c in enumerate(keys)}
    sums = np.zeros(len(keys))
    cnt = np.zeros(len(keys))
    for v, c in zip(y, clusters):
        sums[idx[c]] += v
        cnt[idx[c]] += 1
    r = np.random.default_rng(seed(seed_text))
    pick = r.integers(0, len(keys), size=(B, len(keys)))
    means = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    p = float(y.mean())
    v_srs = p * (1.0 - p) / len(y)
    v_b = float(means.var(ddof=1))
    deff = (v_b / v_srs) if v_srs > 0 else None
    return {"deff": deff, "interval": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "B": B, "clusters": len(keys), "seed_text": seed_text}


def webber_recall(admitted, strata, seed_text, draws=RECALL_DRAWS):
    """Funnel recall with Webber's beta-binomial Monte Carlo (runner §5.1.6).

    admitted: {"N", "k", "n"} or a list of them (admitted boxes and the
    labelled precision of their stratum); strata: discard strata [{"N", "n",
    "k", "design": "srs" | "ht"}] (an ht stratum gives k = theta n_eff and
    n = n_eff, and its recoverable count is N p)."""
    import numpy as np
    parts = admitted if isinstance(admitted, list) else [admitted]
    r = np.random.default_rng(seed(seed_text))
    A = np.zeros(draws)
    A0 = 0.0
    for a in parts:
        N, k, n = float(a["N"]), float(a["k"]), float(a["n"])
        A += N * _beta_draws(r, k + 1.0, n - k + 1.0, draws)
        A0 += N * ((k / n) if n > 0 else 0.5)
    R = np.zeros(draws)
    R0 = 0.0
    for s in strata:
        N, n, k = float(s["N"]), float(s["n"]), float(s["k"])
        p = _beta_draws(r, k + 1.0, n - k + 1.0, draws)
        if s.get("design", "srs") == "srs":
            rest = max(int(round(N - n)), 0)
            R += k + (r.binomial(rest, p) if rest > 0 else 0.0)
            R0 += k + (N - n) * ((k / n) if n > 0 else 0.5)
        else:
            R += N * p
            R0 += N * ((k / n) if n > 0 else 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(A + R > 0, A / (A + R), 1.0)
    plug = A0 / (A0 + R0) if (A0 + R0) > 0 else None
    return {"estimate": float(np.median(rec)), "plug_in": plug,
            "interval": [float(np.quantile(rec, 0.025)), float(np.quantile(rec, 0.975))],
            "draws": int(draws), "seed_text": seed_text, "method": "webber beta-binomial Monte Carlo (median)"}


def holm(pvalues, alpha=0.025):
    """Holm's step-down adjustment: {name: {"p", "p_adjusted", "reject"}}."""
    items = sorted(pvalues.items(), key=lambda kv: (float(kv[1]), kv[0]))
    m = len(items)
    out = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * float(p))
        running = max(running, adj)
        out[name] = {"p": float(p), "p_adjusted": running, "reject": running <= alpha}
    return out


def perm_test_one_sided(a, b):
    """Exact one-sided permutation test of mean(a) > mean(b) over every split
    of the pooled values; p = share of splits with a difference >= the
    observed one."""
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if not a or not b:
        raise EstimateError("perm_test_one_sided needs two non-empty samples")
    pooled = a + b
    na, n = len(a), len(a) + len(b)
    total = sum(pooled)
    obs = _mean(a) - _mean(b)
    tol = 1e-12 * max(1.0, abs(obs))
    count = splits = 0
    for comb in itertools.combinations(range(n), na):
        sa = sum(pooled[i] for i in comb)
        diff = sa / na - (total - sa) / (n - na)
        splits += 1
        if diff >= obs - tol:
            count += 1
    return {"p": count / float(splits), "observed_diff": obs, "n_splits": splits}


def ltt_threshold(scores, correct, level=LTT_LEVEL, delta=LTT_DELTA, grid=LTT_GRID):
    """Learn-then-Test: the smallest score threshold whose precision >= level
    is certified at delta, with Bonferroni over a grid of candidate
    thresholds (score quantiles) and exact binomial p-values. inf when no
    candidate is certified."""
    pairs = sorted(zip((float(s) for s in scores), (bool(c) for c in correct)))
    if not pairs:
        return {"threshold": float("inf"), "certified": [], "grid": []}
    ss = [p[0] for p in pairs]
    cands = sorted({ss[min(len(ss) - 1, int(math.floor(q * len(ss) / float(grid))))] for q in range(grid)})
    certified = []
    for lam in cands:
        sel = [c for s, c in pairs if s >= lam]
        n, k = len(sel), sum(sel)
        pval = binom_upper_tail(k, n, level) if n else 1.0
        if pval <= delta / float(len(cands)):
            certified.append(lam)
    return {"threshold": min(certified) if certified else float("inf"), "certified": certified,
            "grid": cands}


# ====================================================================== prereg rules
def _num(pattern, text, what):
    m = re.search(pattern, str(text))
    if not m:
        raise EstimateError("prereg %s: %r does not match %r" % (what, text, pattern))
    return m


def rules(prereg, domain=None):
    """The decision thresholds, read from prereg_v1.json (runner §5.1.6: the
    rule texts are copied into the audit and their numbers drive the
    verdicts). EstimateError when a rule text cannot be read."""
    raw = prereg.raw if isinstance(prereg, D.Prereg) else prereg
    H = raw.get("hypotheses") or {}
    est = raw.get("estimators") or {}
    one = float(est.get("decisions_one_sided", 0.975))
    r = {"conf": 2.0 * one - 1.0, "one_sided": one}
    m = _num(r"LB>=([0-9.]+) overall and for ([A-Za-z0-9_]+)", H["H1"]["supported"], "H1.supported")
    r["H1"] = {"lb_min": float(m.group(1)), "class": m.group(2),
               "ub_max": float(_num(r"UB<([0-9.]+) overall", H["H1"]["falsified"], "H1.falsified").group(1))}
    if domain is not None and r["H1"]["class"] not in domain.target_names:
        raise EstimateError("prereg H1 names %r, which is not a target class" % r["H1"]["class"])
    r["H2a"] = {"lb_min": float(_num(r"LB>=([0-9.]+)", H["H2a"]["supported"], "H2a").group(1)),
                "ub_max": float(_num(r"UB<([0-9.]+)", H["H2a"]["falsified"], "H2a").group(1))}
    r["H2b"] = {"ub_max": float(_num(r"UB<=([0-9.]+)", H["H2b"]["supported"], "H2b").group(1)),
                "lb_min": float(_num(r"LB>([0-9.]+)", H["H2b"]["falsified"], "H2b").group(1))}
    h3a = H["H3a"]
    r["H3a"] = {"geometry_match_min": float(h3a["geometry_match_min"]),
                "id_agreement_min": float(h3a["id_agreement_min"]),
                "purity_point_min": float(h3a["purity"]["point_min"]),
                "purity_lb_min": float(h3a["purity"]["lb_min"]),
                "classes": dict(h3a["purity"]["classes"])}
    r["H3b"] = {"lb_min": float(_num(r"LB>=([0-9.]+)", H["H3b"]["supported"], "H3b").group(1)),
                "ub_max": float(_num(r"UB<([0-9.]+)", H["H3b"]["falsified"], "H3b").group(1))}
    m = _num(r"UB<([0-9.]+) named-other and UB<([0-9.]+) no-information", H["H4"]["supported"], "H4.supported")
    m2 = _num(r"LB>=([0-9.]+) named-other or LB>=([0-9.]+) no-information", H["H4"]["falsified"], "H4.falsified")
    r["H4"] = {"named_ub_max": float(m.group(1)), "noinfo_ub_max": float(m.group(2)),
               "named_lb_min": float(m2.group(1)), "noinfo_lb_min": float(m2.group(2))}
    r["H5a"] = {"boxes_max": int(_num(r"<\s*([0-9]+)\s*boxes", H["H5a"]["exact"], "H5a").group(1))}
    r["H5b"] = {"lb_min": float(_num(r">=([0-9.]+)", H["H5b"]["supported"], "H5b").group(1)),
                "ub_max": float(_num(r"UB<([0-9.]+)", H["H5b"]["falsified"], "H5b").group(1))}
    r["H9_prime"] = {"lb_min": float(_num(r"LB>=([0-9.]+)", H["H9_prime"]["supported"], "H9'").group(1)),
                     "ub_max": float(_num(r"UB<([0-9.]+)", H["H9_prime"]["falsified"], "H9'").group(1))}
    r["H12"] = {"lb_min": float(_num(r">=([0-9.]+)", H["H12"]["supported"], "H12").group(1)),
                "ub_max": float(_num(r"UB<([0-9.]+)", H["H12"]["falsified"], "H12").group(1))}
    sc = est.get("screening") or {}
    r["screening"] = {"holm": bool(sc.get("holm", True)), "fn_rate_null": float(sc["fn_rate_null"]),
                      "R_min": int(sc["R_min_boxes"]), "R_min_small": int(sc["R_min_small_class"]),
                      "small_below": int(sc["small_class_train_core_boxes_below"])}
    r["alpha"] = 1.0 - one
    return r


def rule_text(prereg, hid):
    raw = prereg.raw if isinstance(prereg, D.Prereg) else prereg
    key = {"H9'": "H9_prime", "H1-pre": "H1_pre"}.get(hid, hid)
    v = (raw.get("hypotheses") or {}).get(key)
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, ensure_ascii=False)


# ====================================================================== context
def _genus(taxon):
    t = str(taxon or "").strip()
    return t.split()[0] if t else None


def _f(v):
    if v in (None, ""):
        return None
    return float(v)


def _i(v):
    if v in (None, ""):
        return None
    return int(float(v))


def _stratum_part(sid, key):
    for seg in str(sid).split("/")[1:]:
        k, _, v = seg.partition("=")
        if k == key:
            return v
    return None


def verdict_both(fn, pair):
    """The verdict under both unsure assignments: it holds only when the two
    agree, else inconclusive."""
    a, b = fn(pair["as_no"]), fn(pair["as_yes"])
    return a if a == b else "inconclusive"


class Context(object):
    """Everything the evaluators read, indexed once. Built by evaluate() from
    the funnel directory, or directly (tests)."""

    def __init__(self, domain, prereg, frames, sample, key_rows, gold, rlq, jq=None, materials=None,
                 census=None, rel_geo=None, rel_audit=None, leak=None, da=None, calib=None,
                 class_maps=None, name_status=None, verifier_thresholds=None, pool_summary=None,
                 ledger_rows=None, funnel_ledger=None, judge_scores=None, known_truth=None, adapter=None,
                 funnel_dir=None):
        self.domain = domain
        self.prereg = prereg
        self.R = rules(prereg, domain)
        self.conf = self.R["conf"]
        self.frames_doc = frames["doc"]
        self.frame_groups = frames["groups"]
        self.stratum_units = {}
        self.unit_frames = {}
        self.unit_row = {}
        for g, rows in self.frame_groups.items():
            for u in rows:
                self.stratum_units.setdefault(u["stratum"], []).append(u)
                self.unit_frames.setdefault(u["unit_id"], []).append((g, u["stratum"]))
                self.unit_row.setdefault(u["unit_id"], u)
        self.sample = list(sample)
        self.items = {r["item_id"]: r for r in self.sample}
        self.items_by_stratum = {}
        self.items_by_group = {}
        for r in self.sample:
            self.items_by_stratum.setdefault(r["stratum"], []).append(r)
            self.items_by_group.setdefault(r["group"], []).append(r)
        self.key = {}
        self.planted = None
        for k in key_rows:
            if "planted" in k:
                self.planted = k["planted"]
            else:
                self.key[k["item_id"]] = k
        self.gold = list(gold)
        self.answers = {}
        for g in self.gold:
            self.answers.setdefault(g.get("backend"), {})[g["item_id"]] = g
        self.rlq = rlq or {}
        self.jq = jq
        self.materials = materials
        self.census = census or {}
        self.rel_geo = rel_geo
        self.rel_audit = rel_audit
        self.leak = leak
        self.da = da
        self.calib = calib
        self.class_maps = class_maps
        self.name_status = name_status
        self.verifier_thresholds = verifier_thresholds
        self.pool_summary = pool_summary
        self.ledger_rows = ledger_rows
        self.funnel_ledger = funnel_ledger
        self.judge_scores = judge_scores
        self.known_truth = known_truth
        self.adapter = adapter
        self.funnel_dir = funnel_dir
        self.overlap = []
        self._cache = {}
        self._targets = set(domain.target_names)
        self._target_genera = {_genus(t.get("taxon")) for t in domain.targets if _genus(t.get("taxon"))}
        self._unsure_genera = set(domain.genus_unsure())
        self._ref_lab = domain.reference_lab()
        from . import qualify as Q
        self._card_sources = set(Q._card_resolved_sources(domain))

    # -------------------------------------------------------------- labeller
    def primary(self, hid):
        p = (self.rlq.get("primary") or {})
        for k in (hid, re.sub(r"[a-z]$", "", hid)):
            if k in p:
                return p[k]
        return None

    def pair_backend(self):
        """The backend that may see evaluation pixels (DEC-2), from the config."""
        bes = (self.domain.section("reference_labeller", required=False) or {}).get("backends") or {}
        cands = sorted(b for b, v in bes.items() if "eval" in (v.get("may_see") or []))
        if len(cands) != 1:
            raise EstimateError("exactly one labeller backend may see evaluation pixels; the config has %s" % cands)
        return cands[0]

    def se_sp(self, backend, scope, level):
        blk = (((self.rlq.get("backends") or {}).get(backend) or {}).get(scope) or {}).get(level)
        if not blk:
            return None
        return blk

    @staticmethod
    def level_ok(level, needed):
        if level is None:
            return False
        return LEVELS.index(level) <= LEVELS.index(needed)

    def genus_species_ok(self, prim, genus):
        """Is the primary labeller qualified at species level for a target
        genus in its scope? Fail closed: a genus with no record is not."""
        bg = ((self.rlq.get("by_genus") or {}).get(prim.get("backend")) or {}).get(prim.get("scope"))
        blk = bg.get(genus) if isinstance(bg, dict) else None
        return bool(isinstance(blk, dict) and (blk.get("species") or {}).get("qualified") is True)

    def hyp_of(self, sid):
        """The hypothesis whose primary labeller answers a stratum: its group's
        (GROUP_HYP), except the named frame of G1, which is H2b's, and a class
        unit of a card-resolved source, which is H3a's. This is the split
        qualify.hypothesis_strata makes when it picks each hypothesis's
        primary, so a stratum is answered by the labeller qualified for it."""
        g = sid.split("/", 1)[0]
        if g == "G1" and _stratum_part(sid, "frame") == "named":
            return "H2b"
        if g == "G3" and _source_of_unit_stratum(sid) in self._card_sources:
            return "H3a"
        return GROUP_HYP[g]

    def class_genus(self, name):
        try:
            return _genus(self.domain.target(name).get("taxon"))
        except FunnelError:
            return None

    # -------------------------------------------------------------- events
    def target_answer(self, g, level):
        """The target class an answer names at a level, "no" for a confident
        non-target, None for unsure (contract §5.1)."""
        ans = (g.get("answer") or "").strip()
        if ans in UNSURE_ANSWERS:
            return None
        genus_answer = g.get("answer_level") == "genus"
        ag = _genus(g.get("answer_taxon"))
        if ans in self._targets:
            t = self.domain.target(ans)
            if genus_answer and (ag or _genus(t.get("taxon"))) in self._unsure_genera:
                return None
            if level == "species":
                return ans
            if level == "genus":
                return ans if t.get("rank") == "genus" else None
            return None
        if level == "species" and genus_answer and ag in self._unsure_genera:
            return None
        if level == "genus" and ag in self._target_genera and ans not in ("non_object", "invalid"):
            return None
        if level == "plant" and ans == "other":
            return None
        return "no"

    def event(self, kind, g, unit, level):
        """1, 0 or None (unsure) for one answered item."""
        if kind == "pair":
            a = (g.get("answer") or "").strip()
            if a in POSITIVE_PAIRS:
                return 1
            if a == "different":
                return 0
            return None
        box = (g.get("box_ok") or "").strip().lower()
        t = self.target_answer(g, level)
        if kind == "target_id":
            return None if t is None else (0 if t == "no" else 1)
        if kind.startswith("purity:"):
            return None if t is None else int(t == kind.split(":", 1)[1])
        if kind == "box_valid":
            if (g.get("answer") or "") in UNSURE_ANSWERS:
                return None
            return 1 if box == "yes" else (0 if box in ("no",) or g.get("answer") == "invalid" else None)
        if t is None:
            return None
        if kind == "target":
            want_ok = t != "no"
        elif kind in ("label", "label_right"):
            want_ok = t == (unit or {}).get("label")
        elif kind == "pred":
            want_ok = t == (unit or {}).get("pred")
        elif kind.startswith("label:"):
            want_ok = t == kind.split(":", 1)[1]
        else:
            raise EstimateError("unknown event %r" % kind)
        if not want_ok:
            return 0
        if kind == "label_right":
            return 1
        if box == "yes":
            return 1
        if box == "no":
            return 0
        return None

    # -------------------------------------------------------------- strata
    def stratum_items(self, sid):
        group = sid.split("/", 1)[0]
        if group in JOINT_GROUPS:
            units = {u["unit_id"] for u in self.stratum_units.get(sid, [])}
            rows = [r for g in JOINT_GROUPS for r in self.items_by_group.get(g, []) if r["unit_id"] in units]
            return sorted(rows, key=lambda r: r["item_id"])
        return sorted(self.items_by_stratum.get(sid, []), key=lambda r: r["item_id"])

    def stratum_keys(self, sid):
        units = self.stratum_units.get(sid, [])
        return {"source": sorted({u["source"] for u in units if u["source"]}),
                "near_dup3": sorted({u["near_dup3"] for u in units if u["near_dup3"]}),
                "provenance": sorted({u["provenance"] for u in units if u["provenance"]}),
                "lab": sorted({u["lab"] for u in units if u["lab"]})}

    def stratum_type(self, sid):
        g = sid.split("/", 1)[0]
        fr = _stratum_part(sid, "frame")
        return STRATUM_TYPE.get("%s:%s" % (g, fr)) if fr else STRATUM_TYPE.get(g)

    def _labelled(self, sid, backend, kind, level):
        out = []
        for r in self.stratum_items(sid):
            g = (self.answers.get(backend) or {}).get(r["item_id"]) if backend else None
            if g is None:
                continue
            unit = self.unit_row.get(r["unit_id"])
            out.append((r, unit, self.event(kind, g, unit, level)))
        return out

    def _design(self, sid, lab, assign):
        """(theta, var, n, n_eff, interval, method) of one assignment."""
        group = sid.split("/", 1)[0]
        N = len(self.stratum_units.get(sid, []))
        ys = [(assign if y is None else y) for _r, _u, y in lab]
        n = len(ys)
        if n == 0:
            return None
        if group in HT_GROUPS:
            pis = [float(r["pi"]) for r, _u, _y in lab]
            e = ht(ys, pis, N, self.conf)
            return {"theta": min(max(e["estimate"], 0.0), 1.0), "var": e["var"], "n": n, "n_eff": e["n_eff"],
                    "interval": e["interval"], "method": "ht", "N": N}
        k = float(sum(ys))
        lo, hi = binom_interval(k, n, self.conf)
        return {"theta": k / n, "var": srs_var(k, n, N), "n": n, "n_eff": float(n), "interval": [lo, hi],
                "method": "wilson" if n >= 30 else "jeffreys", "N": N}

    def _predictor(self, sid, kind):
        """The PPI predictor of a stratum: among the judges the frames allow
        there and that qualify for its type, the largest precision_at_half.lb
        + rescue.lb (ties by id); re-checked for disjointness (R12)."""
        if self.jq is None or self.materials is None or self.judge_scores is None:
            return None
        typ = self.stratum_type(sid)
        if typ is None:
            return None
        units = self.stratum_units.get(sid, [])
        allowed = set()
        for u in units[:1]:
            allowed = {j for j in (u.get("allowed_judges") or "").split(";") if j}
        from . import qualify as Q
        keys = self.stratum_keys(sid)
        best = None
        for j in sorted(allowed):
            if j not in (self.jq.get("judges") or {}):
                continue
            try:
                scope, mat, by_type = Q.material_for(j, keys, self.jq, self.materials)
            except (FunnelError, KeyError):
                continue
            blk = by_type.get(typ) or {}
            if not blk.get("qualified"):
                continue
            sh = Q.shares(keys, mat)
            if any(sh.values()):
                self.overlap.append({"judge": j, "stratum": sid, "scope": scope,
                                     "shared": {k: v[:5] for k, v in sh.items() if v}})
            score = (blk.get("precision_at_half") or {}).get("lb", 0.0) + (blk.get("rescue") or {}).get("lb", 0.0)
            if best is None or score > best[0] + 1e-15:
                best = (score, j)
        return best[1] if best else None

    def _ppi(self, sid, judge, lab, kind):
        scores = self.judge_scores(judge) if judge else None
        if not scores:
            return None
        labels, P_of = scores["labels"], scores["P"]
        tset = [i for i, name in enumerate(labels) if name in self._targets]

        def f_of(unit):
            cid = _i(unit.get("crop_id"))
            row = P_of.get(cid) if cid is not None else None
            if row is None:
                return None
            if kind in ("label", "label_right") and unit.get("label") in labels:
                return float(row[labels.index(unit["label"])])
            return float(sum(row[i] for i in tset))
        f_all = [f_of(u) for u in self.stratum_units.get(sid, [])]
        if any(v is None for v in f_all):
            return None
        return f_all, f_of

    def stratum(self, sid, hid, kind):
        """The estimate record of one stratum (runner §4.16 "strata")."""
        ck = (sid, hid, kind)
        if ck in self._cache:
            return self._cache[ck]
        group = sid.split("/", 1)[0]
        N = len(self.stratum_units.get(sid, []))
        prim = self.primary(hid) if hid != "H5b" else None
        if hid == "H5b":
            backend = self.pair_backend()
            level = "species"
        else:
            backend = (prim or {}).get("backend")
            level = (prim or {}).get("level")
        lab = self._labelled(sid, backend, kind, level) if backend else []
        drawn = len(self.stratum_items(sid))
        rec = {"group": group, "stratum": sid, "N": N, "n": drawn, "n_labelled": len(lab), "event": kind,
               "labeller": backend, "level": level, "estimate": None, "interval": [0.0, 1.0], "method": None,
               "rogan_gladen": {"applied": False, "se": None, "sp": None, "flag": "no_labels"},
               "unsure": {"as_no": None, "as_yes": None}, "predictor": None, "deff": None,
               "n_unsure": sum(1 for _r, _u, y in lab if y is None), "raw": {}}
        if not lab:
            self._cache[ck] = rec
            return rec
        judge = None
        if group not in HT_GROUPS and len(lab) >= PPI_MIN_N and kind != "pair":
            judge = self._predictor(sid, kind)
        scope = (self.rlq.get("strata_scopes") or {}).get(sid)
        blk = self.se_sp(backend, scope, level) if (scope and level and kind != "pair") else None
        for name, assign in (("as_no", 0), ("as_yes", 1)):
            d = self._design(sid, lab, assign)
            if judge:
                pp = self._ppi(sid, judge, lab, kind)
                if pp is not None:
                    f_all, f_of = pp
                    ys = [(assign if y is None else y) for _r, _u, y in lab]
                    fl = [f_of(u) for _r, u, _y in lab]
                    p = ppi_pp(ys, fl, f_all, conf=self.conf)
                    if (p["interval"][1] - p["interval"][0]) < (d["interval"][1] - d["interval"][0]):
                        d = {"theta": min(max(p["estimate"], 0.0), 1.0), "var": p["var"], "n": d["n"],
                             "n_eff": d["n_eff"], "interval": p["interval"], "method": "ppi", "N": N,
                             "lambda": p["lambda"]}
                        rec["predictor"] = judge
            rec["raw"][name] = d
            corrected = self._rg(d, blk, "%s/rg/%s" % (SEED_PREFIX, sid) if name == "as_no"
                                 else "%s/rg/%s/as_yes" % (SEED_PREFIX, sid))
            rec["unsure"][name] = corrected
        rec["rogan_gladen"] = {"applied": rec["unsure"]["as_no"]["rg"]["applied"],
                               "se": (blk or {}).get("se"), "sp": (blk or {}).get("sp"),
                               "flag": rec["unsure"]["as_no"]["rg"]["flag"], "scope": scope}
        rec["estimate"] = rec["unsure"]["as_no"]["estimate"]
        rec["interval"] = [min(rec["unsure"]["as_no"]["interval"][0], rec["unsure"]["as_yes"]["interval"][0]),
                           max(rec["unsure"]["as_no"]["interval"][1], rec["unsure"]["as_yes"]["interval"][1])]
        rec["method"] = rec["raw"]["as_no"]["method"]
        if len(lab) >= 2:
            keys = [r.get("image_key") or r["unit_id"] for r, _u, _y in lab]
            cb = cluster_bootstrap([(0 if y is None else y) for _r, _u, y in lab], keys,
                                   "%s/boot/%s" % (SEED_PREFIX, sid))
            rec["deff"] = cb["deff"]
        self._cache[ck] = rec
        return rec

    def _rg(self, d, blk, seed_text):
        """Rogan-Gladen on a design estimate (theta, n_eff) with a labeller
        block {"se", "sp", "qualified"}; uncorrected and flagged when the
        block is missing or unqualified."""
        if d is None:
            return None
        if not blk or not blk.get("qualified"):
            return {"estimate": d["theta"], "interval": list(d["interval"]), "n": d["n"],
                    "rg": {"applied": False, "flag": "not_qualified_in_scope" if blk else "no_scope"}}
        se, sp = blk["se"], blk["sp"]
        rg = rogan_gladen(d["theta"] * d["n_eff"], d["n_eff"], se["k"], se["n"], sp["k"], sp["n"], seed_text,
                          conf=self.conf)
        if not rg["applied"]:
            # flagged (Se + Sp - 1 < 0.7): the uncorrected design estimate is reported
            return {"estimate": d["theta"], "interval": list(d["interval"]), "n": d["n"],
                    "rg": {"applied": False, "flag": rg["flag"]}}
        return {"estimate": rg["estimate"], "interval": list(rg["interval"]), "n": d["n"],
                "rg": {"applied": True, "flag": None}}

    def aggregate(self, sids, hid, kind, scope=None, seed_key=None, hid_of=None):
        """Korn-Graubard aggregate over strata, then Rogan-Gladen with the
        hypothesis's (or the given) scope; both unsure assignments. hid_of
        (stratum -> hypothesis) lets each stratum be answered by its own
        hypothesis's primary labeller; when the strata's labellers or levels
        differ, no single Se/Sp applies and the aggregate stays uncorrected
        (flag mixed_labellers)."""
        sids = sorted(sids)
        recs = [self.stratum(s, hid_of(s) if hid_of else hid, kind) for s in sids]
        labs = {(r["labeller"], r["level"]) for r in recs if r["n_labelled"]}
        mixed = len(labs) > 1
        if hid == "H5b":
            backend, level = self.pair_backend(), "species"
            prim = None
        else:
            prim = self.primary(hid)
            backend, level = (prim or {}).get("backend"), (prim or {}).get("level")
            if hid_of is not None and len(labs) == 1:
                backend, level = next(iter(labs))
        if scope is None:
            scope = (prim or {}).get("scope")
        # scope "" means no set is disjoint from every stratum: no correction is possible
        blk = self.se_sp(backend, scope, level) if (backend and scope and level and kind != "pair"
                                                    and not mixed) else None
        out = {"strata": sids, "N": sum(r["N"] for r in recs), "n": sum(r["n"] for r in recs),
               "n_labelled": sum(r["n_labelled"] for r in recs), "labeller": backend, "level": level,
               "scope": scope, "event": kind, "mixed_labellers": mixed}
        for name in ("as_no", "as_yes"):
            parts = []
            for r in recs:
                d = r["raw"].get(name)
                if d is None:
                    parts.append({"N": r["N"], "theta": 0.0, "var": 0.0, "n": 0})
                else:
                    parts.append({"N": r["N"], "theta": d["theta"], "var": d["var"], "n": d["n"]})
            c = combine(parts, self.conf)
            if c["estimate"] is None:
                out[name] = {"estimate": None, "interval": [0.0, 1.0], "n_eff": 0.0, "method": c["method"],
                             "rg": {"applied": False, "flag": "no_labels"}, "uncovered_weight": 1.0}
                continue
            w_e = c["uncovered_weight"]
            cov_n_eff = c["n_eff"]
            base = {"theta": c["estimate"], "n_eff": cov_n_eff, "n": c["n"], "interval": [
                (c["interval"][0] / (1.0 - w_e)) if w_e < 1 else 0.0,
                ((c["interval"][1] - w_e) / (1.0 - w_e)) if w_e < 1 else 1.0]}
            sk = "%s/rg/%s/%s%s" % (SEED_PREFIX, seed_key or hid, kind, "" if name == "as_no" else "/as_yes")
            corr = self._rg(base, blk, sk)
            if mixed:
                corr["rg"] = {"applied": False, "flag": "mixed_labellers"}
            lo, hi = corr["interval"]
            if w_e > 0:
                lo, hi = lo * (1.0 - w_e), hi * (1.0 - w_e) + w_e
            out[name] = {"estimate": corr["estimate"], "interval": [lo, hi], "n_eff": cov_n_eff,
                         "method": c["method"], "rg": corr["rg"], "uncovered_weight": w_e}
        return out


# ====================================================================== hypotheses
def _hyp(prereg, hid, verdict, why, estimator=None, n=None, estimate=None, interval=None, parts=None,
         both=None):
    if verdict not in VERDICTS:
        raise EstimateError("verdict %r" % verdict)
    return {"verdict": verdict, "rule": rule_text(prereg, hid), "estimator": estimator, "n": n,
            "estimate": estimate, "interval": interval, "parts": parts or {}, "under_both_unsure": both,
            "why": why}


def _agg_summary(a):
    return {"as_no": {"estimate": a["as_no"]["estimate"], "interval": a["as_no"]["interval"]},
            "as_yes": {"estimate": a["as_yes"]["estimate"], "interval": a["as_yes"]["interval"]}}


def _bounded_check(ctx, hid, needed="species", genera=()):
    """Why hid is bounded, or None: no primary labeller, a primary qualified
    only at a coarser level than needed, or (contract §10 stop rules) a genus
    the hypothesis is about for which the primary is not qualified at species
    level. The per-genus record is rl_qualification.json by_genus (or the
    primary's genera_not_qualified_at_species); a genus absent from it has no
    sentinels and so is not qualified."""
    prim = ctx.primary(hid)
    if prim is None or prim.get("backend") is None:
        return "no reference labeller qualifies for %s in its scope (%s)" % (
            hid, (prim or {}).get("why", "no primary"))
    if not Context.level_ok(prim.get("level"), needed):
        return "the primary labeller %s is qualified at %s only; %s needs %s" % (
            prim.get("backend"), prim.get("level"), hid, needed)
    if needed == "species":
        bad = sorted(g for g in set(genera) if g and not ctx.genus_species_ok(prim, g))
        if bad:
            return "the primary labeller %s is not qualified at species level for %s (stop rule: hypotheses " \
                   "about that genus are bounded)" % (prim.get("backend"), ", ".join(bad))
    return None


def _strata_of(ctx, group, pred=None):
    out = []
    for sid in sorted(ctx.stratum_units):
        if sid.split("/", 1)[0] != group:
            continue
        if pred is None or pred(sid):
            out.append(sid)
    return out


def _source_of_unit_stratum(sid):
    u = _stratum_part(sid, "unit") or ""
    if u[:2] in ("c:", "k:"):
        return u[2:].split("|")[0]
    return _stratum_part(sid, "source")


def h0(ctx):
    """H0: (a) the planted share is covered; (b) the relation audit passes;
    (c) reported."""
    pre, R = ctx.prereg, ctx.R
    parts = {}
    why = []
    # (a)
    g0_items = [r for r in ctx.items_by_group.get("G0", [])]
    b = _bounded_check(ctx, "H0")
    prim = ctx.primary("H0") or {}
    planted = (ctx.planted or {}).get("share")
    a_verdict = "not_evaluated"
    a_part = {"planted_share": planted, "n": len(g0_items)}
    if planted is None or not g0_items:
        why.append("(a) no planted control in the sample")
    elif b:
        a_verdict = "bounded"
        why.append("(a) " + b)
    else:
        backend, level = prim["backend"], prim["level"]
        ys = []
        for r in g0_items:
            g = (ctx.answers.get(backend) or {}).get(r["item_id"])
            if g is not None:
                ys.append(ctx.event("target_id", g, None, level))
        blk = ctx.se_sp(backend, prim.get("scope"), level)
        res = {}
        for name, assign in (("as_no", 0), ("as_yes", 1)):
            vals = [assign if y is None else y for y in ys]
            n = len(vals)
            if n == 0:
                res[name] = {"estimate": None, "interval": [0.0, 1.0]}
                continue
            k = float(sum(vals))
            lo, hi = binom_interval(k, n, 0.95)
            d = {"theta": k / n, "n_eff": float(n), "n": n, "interval": [lo, hi]}
            res[name] = ctx._rg(d, blk, "%s/rg/G0%s" % (SEED_PREFIX, "" if name == "as_no" else "/as_yes"))
        a_part.update({"answered": len(ys), "labeller": backend, "level": level, "scope": prim.get("scope"),
                       "as_no": res["as_no"], "as_yes": res["as_yes"]})

        def covers(e):
            lo, hi = e["interval"]
            return "supported" if lo <= planted <= hi else "falsified"
        if not ys:
            a_verdict = "inconclusive"
            why.append("(a) no G0 item was answered")
        else:
            a_verdict = verdict_both(covers, res)
            why.append("(a) planted share %.4f vs interval %s (unsure as no) and %s (as yes)"
                       % (planted, _fmt_iv(res["as_no"]["interval"]), _fmt_iv(res["as_yes"]["interval"])))
    a_part["verdict"] = a_verdict
    parts["a"] = a_part
    # (b)
    rb = (ctx.rel_audit or {}).get("h0b") if ctx.rel_audit else None
    if rb is None:
        b_verdict = "not_evaluated"
        why.append("(b) relation_audit_v1.json is missing")
    else:
        b_verdict = "supported" if rb.get("pass") is True else "falsified"
        why.append("(b) relation audit %s" % ("passes" if rb.get("pass") else "fails"))
    parts["b"] = {"verdict": b_verdict, "pass": None if rb is None else rb.get("pass")}
    rc = (ctx.rel_audit or {}).get("h0c") if ctx.rel_audit else None
    parts["c"] = {"verdict": "reported", "pass": None if rc is None else rc.get("pass"),
                  "class_accuracy": None if rc is None else rc.get("class_accuracy"),
                  "gates": "L11(b) visual class maps"}
    if "falsified" in (a_verdict, b_verdict):
        v = "falsified"
    elif a_verdict == "supported" and b_verdict == "supported":
        v = "supported"
    elif "bounded" in (a_verdict, b_verdict):
        v = "bounded"
    elif "not_evaluated" in (a_verdict, b_verdict):
        v = "not_evaluated"
    else:
        v = "inconclusive"
    ae = a_part.get("as_no") or {}
    return _hyp(pre, "H0", v, "; ".join(why), estimator="binomial (Wilson/Jeffreys) + Rogan-Gladen; exact",
                n=len(g0_items), estimate=ae.get("estimate"), interval=ae.get("interval"), parts=parts,
                both={"as_no": a_part.get("as_no"), "as_yes": a_part.get("as_yes")})


def _fmt_iv(iv):
    if iv is None:
        return "-"
    return "[%.4f, %.4f]" % (iv[0], iv[1])


def _authoritative(ctx):
    return set(ctx.domain.authoritative_sources())


def h1_strata(ctx):
    auth = _authoritative(ctx)
    return _strata_of(ctx, "G2", lambda s: _stratum_part(s, "source") in auth)


def h1(ctx):
    pre, R = ctx.prereg, ctx.R["H1"]
    sids = h1_strata(ctx)
    pre_rec = (ctx.rel_geo or {}).get("h1_pre") if ctx.rel_geo else None
    parts = {"h1_pre": pre_rec}
    if not sids:
        return _hyp(pre, "H1", "not_evaluated", "no G2 stratum of an authoritative source", parts=parts)
    srcs = sorted({_stratum_part(s, "source") for s in sids})
    missing = [x for x in srcs if x not in (pre_rec or {})]
    if missing:
        return _hyp(pre, "H1", "not_evaluated", "H1-pre (the class-order check against the upstream annotations) "
                    "is not recorded for %s; H1 waits for it" % missing, parts=parts)
    failed = [x for x in srcs if not (pre_rec[x] or {}).get("pass")]
    if failed:
        return _hyp(pre, "H1", "not_evaluated", "H1-pre failed for %s: a join error, fixed through L11 before H1 "
                    "is evaluated" % failed, parts=parts)
    agg = ctx.aggregate(sids, "H1", "label", seed_key="H1/overall")
    cls = R["class"]
    per_label = {}
    for lab in sorted({_stratum_part(s, "label") for s in sids}):
        ss = [s for s in sids if _stratum_part(s, "label") == lab]
        per_label[lab] = _agg_summary(ctx.aggregate(ss, "H1", "label", seed_key="H1/label=%s" % lab))
    parts.update({"per_label": per_label,
                  "label_right": _agg_summary(ctx.aggregate(sids, "H1", "label_right", seed_key="H1/label_right")),
                  "box_valid": _agg_summary(ctx.aggregate(sids, "H1", "box_valid", seed_key="H1/box_valid"))})
    # H1 is about every label of its strata: a genus of one of them that the
    # labeller is not qualified for at species level bounds H1 (§10)
    b = _bounded_check(ctx, "H1", genera={ctx.class_genus(_stratum_part(s, "label")) for s in sids})
    cls_sids = [s for s in sids if _stratum_part(s, "label") == cls]
    cls_agg = ctx.aggregate(cls_sids, "H1", "label", seed_key="H1/label=%s" % cls) if cls_sids else None

    def rule(which):
        o = agg[which]
        c = cls_agg[which] if cls_agg else None
        if o["interval"][1] < R["ub_max"]:
            return "falsified"
        if o["interval"][0] >= R["lb_min"] and c is not None and c["interval"][0] >= R["lb_min"]:
            return "supported"
        return "inconclusive"
    v = "bounded" if b else (rule("as_no") if rule("as_no") == rule("as_yes") else "inconclusive")
    return _hyp(pre, "H1", v, b or "overall %s, %s %s (unsure as no)" % (
        _fmt_iv(agg["as_no"]["interval"]), cls, _fmt_iv(cls_agg["as_no"]["interval"]) if cls_agg else "-"),
        estimator="stratified KG + Rogan-Gladen", n=agg["n_labelled"], estimate=agg["as_no"]["estimate"],
        interval=agg["as_no"]["interval"], parts=parts, both=_agg_summary(agg))


def _g1_frame(ctx, frame):
    return _strata_of(ctx, "G1", lambda s: _stratum_part(s, "frame") == frame)


def _pred_precision(ctx, sids, hid):
    return {s: {"estimate": ctx.stratum(s, hid, "pred")["estimate"],
                "interval": ctx.stratum(s, hid, "pred")["interval"]} for s in sids}


def h2(ctx, hid):
    pre = ctx.prereg
    R = ctx.R[hid]
    frame = "noinfo" if hid == "H2a" else "named"
    sids = _g1_frame(ctx, frame)
    if not sids:
        return _hyp(pre, hid, "not_evaluated", "the G1 %s frame is empty" % frame)
    agg = ctx.aggregate(sids, hid, "target", seed_key=hid)
    parts = {"relabel_precision": _pred_precision(ctx, sids, hid)}
    role = [s for s in sids if _stratum_part(s, "status") == "role"]
    if role:
        parts["role_substratum"] = _agg_summary(ctx.aggregate(role, hid, "target", seed_key="%s/role" % hid))
    b = _bounded_check(ctx, hid)
    if hid == "H2a":
        def rule(e):
            lo, hi = e["interval"]
            if hi < R["ub_max"]:
                return "falsified"
            if lo >= R["lb_min"]:
                return "supported"
            return "inconclusive"
    else:
        def rule(e):
            lo, hi = e["interval"]
            if lo > R["lb_min"]:
                return "falsified"
            if hi <= R["ub_max"]:
                return "supported"
            return "inconclusive"
    v = "bounded" if b else verdict_both(rule, agg)
    return _hyp(pre, hid, v, b or "target share %s (unsure as no), %s (as yes)" % (
        _fmt_iv(agg["as_no"]["interval"]), _fmt_iv(agg["as_yes"]["interval"])),
        estimator="stratified KG + Rogan-Gladen", n=agg["n_labelled"], estimate=agg["as_no"]["estimate"],
        interval=agg["as_no"]["interval"], parts=parts, both=_agg_summary(agg))


def h3a_source(ctx):
    srcs = []
    for kt in ctx.domain.claimed_sets("H3a"):
        srcs.extend(ctx.domain.kt(kt).get("sources") or [])
    return sorted(set(srcs))


def _alignment_map(name, n_classes):
    """{upstream id: pool id | None} of an alignment name, from relation.alignments."""
    from . import relation
    maps = relation.alignments(n_classes)
    if name not in maps:
        raise EstimateError("alignment %r is not one of relation.alignments(%d)" % (name, n_classes))
    return {int(k): (None if v is None else int(v)) for k, v in maps[name].items()}


def h3a(ctx):
    pre, R = ctx.prereg, ctx.R["H3a"]
    srcs = h3a_source(ctx)
    if len(srcs) != 1:
        return _hyp(pre, "H3a", "not_evaluated", "the H3a claimed set names %d sources" % len(srcs))
    src = srcs[0]
    geo = ((ctx.rel_geo or {}).get("matches") or {}).get(src)
    if geo is None:
        return _hyp(pre, "H3a", "not_evaluated", "relation_geometry_v1.json has no match record for the source")
    aligns = geo.get("alignments") or []
    chosen = geo.get("chosen")
    ch = [a for a in aligns if a.get("name") == chosen]
    best_any = max([max(float(a.get("agreement_a") or 0), float(a.get("agreement_b") or 0)) for a in aligns] or [0.0])
    exact = {"matched_share": geo.get("matched_share"), "chosen": chosen, "confirmed": geo.get("confirmed"),
             "agreement_chosen": (ch[0].get("agreement_b") if ch else None), "best_agreement_any": best_any,
             "h3a_exact": (ctx.rel_geo or {}).get("h3a_exact")}
    exact_falsified = best_any < R["id_agreement_min"]
    exact_ok = (float(geo.get("matched_share") or 0) >= R["geometry_match_min"] and bool(ch)
                and float(ch[0].get("agreement_b") or 0) >= R["id_agreement_min"] and bool(geo.get("confirmed")))
    parts = {"exact": exact, "purity": {}}
    card = ((ctx.domain.raw.get("sources") or {}).get("card_resolvers") or {}).get(src) or {}
    table = card.get("class_table") or {}
    bounded = []
    purity_fals = False
    purity_ok = True
    if not exact_ok or chosen is None:
        purity_ok = False
    for up_id, taxon in sorted(R["classes"].items(), key=lambda kv: int(kv[0])):
        cls = [t for t in ctx.domain.targets if t.get("taxon") == taxon]
        if not cls:
            raise EstimateError("prereg H3a purity taxon %r is no target's taxon" % taxon)
        cls = cls[0]
        needed = "genus" if cls.get("rank") == "genus" else "species"
        rec = {"class": cls["name"], "taxon": taxon, "needed_level": needed}
        if chosen is None or not table:
            rec["verdict"] = "not_evaluated"
            parts["purity"][up_id] = rec
            purity_ok = False
            continue
        amap = _alignment_map(chosen, len(table))
        pool_id = amap.get(int(up_id))
        rec["pool_id"] = pool_id
        unit = "c:%s|%s" % (src, pool_id)
        sid = "G3/unit=%s" % unit
        b = _bounded_check(ctx, "H3a", needed, genera=(_genus(taxon),))
        if pool_id is None or sid not in ctx.stratum_units:
            rec["verdict"] = "not_evaluated"
            rec["why"] = "unit %s is not a G3 stratum" % unit
            purity_ok = False
        else:
            s = ctx.stratum(sid, "H3a", "purity:%s" % cls["name"])
            rec.update({"stratum": sid, "n_labelled": s["n_labelled"], "as_no": s["raw"].get("as_no"),
                        "as_yes": s["raw"].get("as_yes")})
            if b:
                rec["verdict"] = "bounded"
                bounded.append(b)
                purity_ok = False
            elif not s["n_labelled"]:
                rec["verdict"] = "inconclusive"
                purity_ok = False
            else:
                def pur(which):
                    d = s["raw"][which]
                    if d["interval"][0] < R["purity_lb_min"]:
                        return "falsified"
                    if d["theta"] >= R["purity_point_min"] and d["interval"][0] >= R["purity_lb_min"]:
                        return "supported"
                    return "inconclusive"
                pv = pur("as_no") if pur("as_no") == pur("as_yes") else "inconclusive"
                rec["verdict"] = pv
                purity_fals = purity_fals or pv == "falsified"
                purity_ok = purity_ok and pv == "supported"
        parts["purity"][up_id] = rec
    if exact_falsified or purity_fals:
        v = "falsified"
    elif bounded:
        v = "bounded"
    elif exact_ok and purity_ok:
        v = "supported"
    else:
        v = "inconclusive"
    return _hyp(pre, "H3a", v, "; ".join(bounded) or "exact %s (agreement %s, matched %s); purity %s" % (
        "ok" if exact_ok else "not met", exact["agreement_chosen"], exact["matched_share"],
        {k: p.get("verdict") for k, p in parts["purity"].items()}),
        estimator="exact geometry + binomial purity", parts=parts)


def h3b(ctx):
    pre, R = ctx.prereg, ctx.R["H3b"]
    excl = set(h3a_source(ctx))
    frames = (ctx.domain.section("names") or {}).get("frames") or {}
    noinfo = set(frames.get("noinfo", []))
    sids = []
    for sid in _strata_of(ctx, "G3"):
        if _source_of_unit_stratum(sid) in excl:
            continue
        units = ctx.stratum_units[sid]
        st = json.loads(units[0]["extra"]).get("status") if units and units[0].get("extra") else None
        if st in noinfo and len(units) >= 100:
            sids.append(sid)
    if not sids:
        return _hyp(pre, "H3b", "not_evaluated", "no anonymous G3 unit with >= 100 boxes")
    b = _bounded_check(ctx, "H3b")
    per = {}
    any_sup, all_fals = False, True
    for sid in sids:
        s = ctx.stratum(sid, "H3b", "target")
        per[sid] = {"n_labelled": s["n_labelled"], "as_no": s["unsure"]["as_no"], "as_yes": s["unsure"]["as_yes"]}
        lo = min((s["unsure"][k] or {"interval": [0, 1]})["interval"][0] for k in ("as_no", "as_yes"))
        hi = max((s["unsure"][k] or {"interval": [0, 1]})["interval"][1] for k in ("as_no", "as_yes"))
        if s["n_labelled"] and lo >= R["lb_min"]:
            any_sup = True
        if not (s["n_labelled"] and hi < R["ub_max"]):
            all_fals = False
    if b:
        v = "bounded"
    elif any_sup:
        v = "supported"
    elif all_fals:
        v = "falsified"
    else:
        v = "inconclusive"
    return _hyp(pre, "H3b", v, b or "%d units; supported by some unit: %s; every upper bound below %.2f: %s"
                % (len(sids), any_sup, R["ub_max"], all_fals), estimator="binomial per unit + Rogan-Gladen",
                n=sum(p["n_labelled"] for p in per.values()), parts={"units": per})


def h4(ctx):
    pre, R = ctx.prereg, ctx.R["H4"]
    res = {}
    for frame in ("named", "noinfo"):
        sids = _strata_of(ctx, "G4", lambda s, f=frame: _stratum_part(s, "frame") == f)
        res[frame] = ctx.aggregate(sids, "H4", "target", seed_key="H4/%s" % frame) if sids else None
    b = _bounded_check(ctx, "H4")
    if res["named"] is None or res["noinfo"] is None:
        return _hyp(pre, "H4", "not_evaluated", "a G4 frame is empty")

    def rule(which):
        nm, ni = res["named"][which]["interval"], res["noinfo"][which]["interval"]
        if nm[0] >= R["named_lb_min"] or ni[0] >= R["noinfo_lb_min"]:
            return "falsified"
        if nm[1] < R["named_ub_max"] and ni[1] < R["noinfo_ub_max"]:
            return "supported"
        return "inconclusive"
    v = "bounded" if b else (rule("as_no") if rule("as_no") == rule("as_yes") else "inconclusive")
    return _hyp(pre, "H4", v, b or "named %s, no-information %s (unsure as no)" % (
        _fmt_iv(res["named"]["as_no"]["interval"]), _fmt_iv(res["noinfo"]["as_no"]["interval"])),
        estimator="Horvitz-Thompson per frame + Rogan-Gladen",
        n=res["named"]["n_labelled"] + res["noinfo"]["n_labelled"],
        parts={f: _agg_summary(a) for f, a in res.items()})


def h5a(ctx):
    pre, R = ctx.prereg, ctx.R["H5a"]
    h = (ctx.census or {}).get("h5a")
    if not h:
        return _hyp(pre, "H5a", "not_evaluated", "census_v1.json has no h5a record")
    boxes = int((h.get("dropped_twins_with_target_box_kept_lacks") or {}).get("boxes"))
    v = "supported" if boxes < R["boxes_max"] else "falsified"
    return _hyp(pre, "H5a", v, "%d boxes (threshold %d)" % (boxes, R["boxes_max"]), estimator="exact",
                n=boxes, estimate=float(boxes),
                parts={"more_informative_names": h.get("dropped_twins_more_informative_names")})


def h5b(ctx):
    pre, R = ctx.prereg, ctx.R["H5b"]
    be = ctx.pair_backend()
    q = ((ctx.rlq.get("pairs") or {}).get(be) or {})
    if not q.get("qualified"):
        return _hyp(pre, "H5b", "not_evaluated", "%s is not qualified on pair sentinels (DEC-2)" % be)
    # H5b is about near-evaluation hits only: the G5 strata whose split is an
    # exam split. The exact-dup twins and the reference-copy pairs G5 also
    # holds answer other questions and are reported apart, never pooled in.
    ex = ctx.domain.exam_splits()
    exam = {ex["decision"]} | set(ex["non_decision"])
    all_g5 = _strata_of(ctx, "G5")
    sids = [s for s in all_g5 if _stratum_part(s, "split") in exam]
    others = [s for s in all_g5 if s not in sids]
    if not sids:
        return _hyp(pre, "H5b", "not_evaluated", "the G5 frame holds no near-evaluation pair")
    agg = ctx.aggregate(sids, "H5b", "pair", seed_key="H5b")
    other_parts = {}
    for split in sorted({_stratum_part(s, "split") for s in others}):
        ss = [s for s in others if _stratum_part(s, "split") == split]
        other_parts[split] = _agg_summary(ctx.aggregate(ss, "H5b", "pair", seed_key="H5b/other/%s" % split))

    def rule(e):
        lo, hi = e["interval"]
        if hi < R["ub_max"]:
            return "falsified"
        if lo >= R["lb_min"]:
            return "supported"
        return "inconclusive"
    v = verdict_both(rule, agg)
    return _hyp(pre, "H5b", v, "same-or-consecutive share %s" % _fmt_iv(agg["as_no"]["interval"]),
                estimator="stratified KG", n=agg["n_labelled"], estimate=agg["as_no"]["estimate"],
                interval=agg["as_no"]["interval"], both=_agg_summary(agg),
                parts={"backend": be, "strata": sids, "other_guard_pairs": other_parts})


def h6(ctx):
    if ctx.leak is None:
        return _hyp(ctx.prereg, "H6", "not_evaluated", "leak_v1.json is missing")
    lk = ctx.leak
    return _hyp(ctx.prereg, "H6", "reported", "routing from leak_v1.json", estimator="copy detector",
                parts={"calibration_ok": (lk.get("calibration") or {}).get("ok"), "h6a": lk.get("h6a"),
                       "h6b": lk.get("h6b"), "h6c": lk.get("h6c")})


def h7(ctx):
    if ctx.jq is None or not ctx.jq.get("h7"):
        return _hyp(ctx.prereg, "H7", "not_evaluated", "judge_qualification.json has no h7 record")
    parts = {}
    for j, rec in sorted(ctx.jq["h7"].items()):
        pred = rec.get("predicted")
        q = bool(rec.get("qualified"))
        ok = (pred == "fail" and not q) or (pred == "qualify" and q)
        parts[j] = {"predicted": pred, "qualified": q, "verdict": "supported" if ok else "falsified"}
    return _hyp(ctx.prereg, "H7", "reported", "each judge's prediction is scored on its own",
                estimator="qualification bounds", parts=parts)


def h8(ctx):
    return _hyp(ctx.prereg, "H8", "not_evaluated", "the verifier refit is F11 (card X11), outside this audit")


def h10(ctx):
    return _hyp(ctx.prereg, "H10a", "not_evaluated", "evaluated from panel.json by outcome.py (realloop_v2)")


def h12(ctx):
    pre, R = ctx.prereg, ctx.R["H12"]
    h = (ctx.census or {}).get("h12")
    if not h:
        return _hyp(pre, "H12", "not_evaluated", "census_v1.json has no h12 record (known-item list not fetched)")
    k, n = int(h["present"]), int(h["total"])
    lo, hi = binom_interval(k, n, ctx.conf)
    if hi < R["ub_max"]:
        v = "falsified"
    elif lo >= R["lb_min"]:
        v = "supported"
    else:
        v = "inconclusive"
    return _hyp(pre, "H12", v, "%d of %d known items present" % (k, n), estimator="binomial (Wilson/Jeffreys)",
                n=n, estimate=(k / n if n else None), interval=[lo, hi],
                parts={"missing": [i.get("name") for i in (h.get("items") or []) if not i.get("present")]})


FORECAST_TOL = 1e-6      # the DA validator's tolerance on the forecast's sum


def _forecast_problem(fc, recoverable_stages):
    """Why a stage forecast is not a probability over the recoverable stages
    summing to 1 (contract §6 H11), or None."""
    if not isinstance(fc, dict) or not fc:
        return "the DA record holds no stage forecast (an invalid reply, or none)"
    bad = sorted(k for k in fc if k not in recoverable_stages)
    if bad:
        return "the forecast names stages %s that are not recoverable stages of the ledger" % bad
    vals = list(fc.values())
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1
           for v in vals):
        return "the forecast holds a value that is not a probability"
    if abs(sum(vals) - 1.0) > FORECAST_TOL:
        return "the forecast sums to %r, not 1" % sum(vals)
    return None


def h11(ctx, ranking, recoverable_stages):
    """H11: the DA's top-1 stage is the stage with the largest recoverable
    lower bound and its log-loss beats uniform over the recoverable stages.
    Stages whose lower bounds tie (they are estimated from the same frames,
    as a join stage and its name-status stage are) are one outcome: the
    top-1 is right when it names any of them, and the log-loss is scored on
    the forecast's mass on that set against the uniform mass |set| / K. A
    forecast that is not a probability over the recoverable stages cannot be
    supported: it is falsified, with the reason."""
    pre = ctx.prereg
    if ctx.da is None:
        return _hyp(pre, "H11", "not_evaluated", "prospective_da.json is missing")
    fc = ctx.da.get("stage_forecast")
    ranked = [r for r in ranking if r["stage"] in recoverable_stages]
    if not ranked:
        return _hyp(pre, "H11", "not_evaluated", "no recoverable stage has an estimate")
    K = len(recoverable_stages)
    best = max(float(r["recoverable_lb"]) for r in ranked)
    tol = 1e-9 * max(1.0, abs(best))
    top_set = sorted(r["stage"] for r in ranked if abs(float(r["recoverable_lb"]) - best) <= tol)
    prob = _forecast_problem(fc, set(recoverable_stages))
    if prob:
        return _hyp(pre, "H11", "falsified", prob, estimator="top-1 and log-loss",
                    parts={"top_stages": top_set, "forecast_top1": None, "p_top": None, "log_loss": None,
                           "uniform_log_loss": math.log(K / float(len(top_set))) if K else None, "stages": K,
                           "forecast_problem": prob})
    top_fc = sorted(fc.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    q = float(sum(float(fc.get(s, 0.0)) for s in top_set))
    logloss = -math.log(q) if q > 0 else float("inf")
    uniform = math.log(K / float(len(top_set)))
    ok = top_fc in top_set and logloss < uniform
    return _hyp(pre, "H11", "supported" if ok else "falsified",
                "largest recoverable lower bound at %s; the forecast's top-1 is %s (p %.3f on that outcome); "
                "log-loss %.3f vs uniform %.3f" % ("/".join(top_set), top_fc, q, logloss, uniform),
                estimator="top-1 and log-loss",
                parts={"top_stage": top_set[0], "top_stages": top_set, "forecast_top1": top_fc, "p_top": q,
                       "log_loss": logloss, "uniform_log_loss": uniform, "stages": K})


# ====================================================================== stages
def _role_ids(ctx):
    out = {}
    for st in (ctx.funnel_ledger or {}).get("stages", []):
        out.setdefault(st.get("role"), []).append(st["id"])
    return out


def stages_block(ctx):
    """Per recoverable stage: FN rate, recoverable count, Holm screening
    (runner §5.1.6 stages table)."""
    R = ctx.R["screening"]
    roles = _role_ids(ctx)
    ledger_stages = {s["id"]: s for s in (ctx.funnel_ledger or {}).get("stages", [])}
    out, pvals = {}, {}
    small = int(R["small_below"])
    tc_boxes = (ctx.census or {}).get("train_core_boxes_per_class") or {}
    for role, groups in STAGE_GROUPS.items():
        for sid_stage in roles.get(role, []):
            if ledger_stages.get(sid_stage, {}).get("recoverable") is not True:
                continue
            sids = [s for g in groups for s in _strata_of(ctx, g)]
            if not sids:
                continue
            hid = GROUP_HYP[groups[0]]
            agg = ctx.aggregate(sids, hid, "target", scope=_common_scope(ctx, sids) or "",
                                seed_key="stage/%s" % sid_stage, hid_of=ctx.hyp_of)
            e = agg["as_no"]
            N = agg["N"]
            est_ = e["estimate"]
            rec = {"frames": list(groups), "fn_rate": {"estimate": est_, "interval": e["interval"],
                                                       "n": agg["n_labelled"], "method": e["method"]},
                   "recoverable": {"estimate": None if est_ is None else N * est_,
                                   "interval": [N * e["interval"][0], N * e["interval"][1]]},
                   "N": N, "r_min": R["R_min"], "scope": agg["scope"], "unsure_as_yes": agg["as_yes"],
                   "mixed_labellers": agg["mixed_labellers"]}
            n_eff = e.get("n_eff") or 0.0
            if est_ is not None and n_eff > 0:
                pvals[sid_stage] = binom_upper_tail(n_eff * est_, n_eff, R["fn_rate_null"])
            ys, srcs = [], []
            for s_ in sids:
                rec_ = ctx.stratum(s_, ctx.hyp_of(s_), "target")
                for r_, _u, y_ in ctx._labelled(s_, rec_["labeller"], "target", rec_["level"]):
                    ys.append(0 if y_ is None else y_)
                    srcs.append(r_.get("source") or r_["unit_id"])
            cb = cluster_bootstrap(ys, srcs, "%s/boot/stage/%s" % (SEED_PREFIX, sid_stage)) if len(ys) >= 2 else None
            rec["deff_by_source"] = None if cb is None else cb["deff"]
            by_class = {}
            key = "label" if groups[0] in ("G2", "G2v") else ("pred" if groups[0] == "G1" else None)
            if key:
                for cls in sorted({_stratum_part(s, key) for s in sids} - {None, "rest"}):
                    cs = [s for s in sids if _stratum_part(s, key) == cls]
                    ca = ctx.aggregate(cs, hid, "target", scope=_common_scope(ctx, cs) or "",
                                       seed_key="stage/%s/%s" % (sid_stage, cls), hid_of=ctx.hyp_of)
                    Nc = ca["N"]
                    lbc = Nc * ca["as_no"]["interval"][0]
                    rec_c = {"fn_rate": {"estimate": ca["as_no"]["estimate"], "interval": ca["as_no"]["interval"]},
                             "recoverable_lb": lbc}
                    if cls not in tc_boxes:
                        # R_min depends on the class's reference box count; without it no
                        # threshold is guessed
                        rec_c.update({"r_min": None, "screen": None,
                                      "why": "census_v1.json train_core_boxes_per_class has no %s" % cls})
                    else:
                        rmin = R["R_min_small"] if int(tc_boxes[cls]) < small else R["R_min"]
                        rec_c.update({"r_min": rmin, "screen": lbc >= rmin})
                    by_class[cls] = rec_c
            rec["by_class"] = by_class
            out[sid_stage] = rec
    adj = holm(pvals, ctx.R["alpha"]) if pvals else {}
    for s, rec in out.items():
        a = adj.get(s)
        rec["p"] = None if a is None else a["p"]
        rec["p_holm"] = None if a is None else a["p_adjusted"]
        rec["suspect"] = bool(a and a["reject"] and rec["recoverable"]["interval"][0] >= rec["r_min"])
    # exact stages: dedup (census h5a) and cap (card image counts)
    for sid_stage in roles.get("dedup", []):
        h = (ctx.census or {}).get("h5a") or {}
        boxes = (h.get("dropped_twins_with_target_box_kept_lacks") or {}).get("boxes")
        if boxes is not None:
            out[sid_stage] = {"frames": [], "exact": True, "fn_rate": None,
                              "recoverable": {"estimate": float(boxes), "interval": [float(boxes), float(boxes)]},
                              "p_holm": None, "suspect": False, "r_min": R["R_min"], "source": "census_v1.h5a"}
    for sid_stage in roles.get("cap", []):
        cic = ((ctx.domain.raw.get("sources") or {}).get("card_image_counts") or {})
        per = ((ctx.pool_summary or {}).get("per_slug") or {})
        imgs = {}
        for slug, n in sorted(cic.items()):
            listed = (per.get(slug) or {}).get("images_listed")
            if listed is not None:
                imgs[slug] = max(0, int(n) - int(listed))
        out[sid_stage] = {"frames": [], "exact": True, "fn_rate": None, "unit": "image",
                          "recoverable_images": imgs, "recoverable": None, "p_holm": None, "suspect": False}
    return out


def _common_scope(ctx, sids):
    """The qualify_rl_on sets every stratum's scope keeps (the scope disjoint
    from all of them); None when a stratum has no scope or none is left. The
    labeller file may have no Se/Sp for it: the aggregate then stays
    uncorrected, flagged."""
    scopes = (ctx.rlq.get("strata_scopes") or {})
    sets = None
    for s in sids:
        sc = scopes.get(s)
        if sc is None:
            return None
        ss = set(x for x in sc.split("+") if x)
        sets = ss if sets is None else (sets & ss)
    if not sets:
        return None
    order = ctx.domain.qualify_rl_on()
    return "+".join(k for k in order if k in sets)


def stage_ranking(stages):
    rows = []
    for s, rec in stages.items():
        r = rec.get("recoverable")
        if not r:
            continue
        rows.append({"stage": s, "recoverable_lb": float(r["interval"][0])})
    return sorted(rows, key=lambda x: (-x["recoverable_lb"], x["stage"]))


def d18_inputs(ctx, stages):
    roles = _role_ids(ctx)
    stage_of_group = {}
    for role, groups in STAGE_GROUPS.items():
        for sid_stage in roles.get(role, []):
            if sid_stage in stages:
                for g in groups:
                    stage_of_group.setdefault(g, sid_stage)
    taxa = _name_taxa(ctx)
    out = []
    for g, stage in sorted(stage_of_group.items()):
        for sid in _strata_of(ctx, g):
            s = ctx.stratum(sid, ctx.hyp_of(sid), "target")
            lb = s["interval"][0] if s["n_labelled"] else 0.0
            if s["n_labelled"]:
                lb = min((s["unsure"][k] or {"interval": [0, 1]})["interval"][0] for k in ("as_no", "as_yes"))
            src_taxa, relative = _stratum_taxa(ctx, sid, g, taxa)
            srcs = {u["source"] for u in ctx.stratum_units.get(sid, []) if u["source"]}
            out.append({"stage": stage, "stratum": sid, "kind": D18_KIND[g], "fn_lb": lb,
                        "recoverable_lb": s["N"] * lb, "source": srcs.pop() if len(srcs) == 1 else None,
                        "source_taxa": src_taxa, "relative_of_prediction": relative})
    return out


def _name_taxa(ctx):
    out = {}
    for n in (ctx.name_status or {}).get("names", []):
        out[(n.get("source"), str(n.get("src_id")))] = (n.get("taxon"), n.get("status_v2"))
    return out


def _stratum_taxa(ctx, sid, group, taxa):
    units = ctx.stratum_units.get(sid, [])
    found = set()
    statuses = set()
    for u in units:
        ex = json.loads(u["extra"]) if u.get("extra") else {}
        t = taxa.get((u["source"], str(ex.get("src_id"))))
        if t:
            if t[0]:
                found.add(t[0])
            statuses.add(t[1])
    if group in ("G2", "G2v"):
        lab = _stratum_part(sid, "label") or _stratum_part(sid, "source")
        try:
            found = {ctx.domain.target(lab).get("taxon")}
        except FunnelError:
            found = set()
        return sorted(x for x in found if x), False
    if group == "G3":
        u = _stratum_part(sid, "unit") or ""
        parts = u[2:].split("|") if u[:2] in ("c:", "k:") else []
        if len(parts) >= 2:
            t = taxa.get((parts[0], parts[1]))
            found = {t[0]} if t and t[0] else set()
        return sorted(found), False
    pred = _stratum_part(sid, "pred")
    relative = "target_related" in statuses or _stratum_part(sid, "status") == "target_related"
    if pred and pred != "rest" and pred in ctx._targets:
        t = ctx.domain.target(pred)
        pg = _genus(t.get("taxon"))
        nots = set(t.get("not") or [])
        for tx in found:
            if _genus(tx) == pg or tx in nots:
                relative = True
    return sorted(found), relative


# ====================================================================== recall, label frequency
def _calib_counts(calib):
    """(verified, correct) of the reference-lab in-domain calibration: the
    first block with a "current_join" record."""
    if not isinstance(calib, dict):
        return None
    for v in calib.values():
        if isinstance(v, dict) and isinstance(v.get("current_join"), dict):
            ov = v["current_join"].get("overall") or {}
            n = int((ov.get("verdicts") or {}).get("verified", 0))
            prec = ov.get("verified_precision")
            if n and prec is not None:
                return n, int(round(float(prec) * n))
    return None


def _ledger_admitted(ctx, by_source=False):
    """Admitted target boxes by lab kind (reference / other), from
    ledger.jsonl; with by_source, also the reference-lab count per source."""
    if ctx.ledger_rows is None:
        return None
    ck = ("_ledger_admitted",)
    if ck not in ctx._cache:
        roles = _role_ids(ctx)
        tc = (roles.get("target_check") or [None])[0]
        ir = (roles.get("image_rule") or [None])[0]
        ref, other, ref_src = 0, {}, {}
        for r in ctx.ledger_rows():
            if r.get("unit") != "box" or not ctx.domain.is_target(r.get("label")):
                continue
            p = r.get("path") or {}
            if p.get(tc) == "verified" and p.get(ir) == "admitted":
                if r.get("lab") == ctx._ref_lab:
                    ref += 1
                    ref_src[r["source"]] = ref_src.get(r["source"], 0) + 1
                else:
                    other[r["source"]] = other.get(r["source"], 0) + 1
        ctx._cache[ck] = (ref, other, ref_src)
    ref, other, ref_src = ctx._cache[ck]
    return (ref, dict(other), dict(ref_src)) if by_source else (ref, dict(other))


def funnel_recall(ctx):
    cal = _calib_counts(ctx.calib)
    adm = _ledger_admitted(ctx)
    small = ((ctx.census or {}).get("totals") or {}).get("small_boxes")
    if cal is None or adm is None:
        return {"estimate": None, "interval": None, "why": "calibration counts or ledger.jsonl unavailable",
                "small_boxes_outside": small}
    ref_n, other = adm
    parts = [{"N": ref_n, "n": cal[0], "k": cal[1]}]
    g2a_sids = _strata_of(ctx, "G2a")
    for src, n_adm in sorted(other.items()):
        ss = [s for s in g2a_sids if _stratum_part(s, "source") == src]
        k = n = 0.0
        for s in ss:
            rec = ctx.stratum(s, "H1", "target")
            d = rec["raw"].get("as_no")
            if d:
                k += d["theta"] * d["n_eff"]
                n += d["n_eff"]
        parts.append({"N": n_adm, "n": n, "k": k})
    strata = []
    for g in ("G1", "G2", "G2v", "G4"):
        for sid in _strata_of(ctx, g):
            rec = ctx.stratum(sid, ctx.hyp_of(sid), "target")
            d = rec["raw"].get("as_no")
            if d is None:
                strata.append({"N": rec["N"], "n": 0, "k": 0, "design": "srs" if g not in HT_GROUPS else "ht"})
            elif g in HT_GROUPS:
                strata.append({"N": rec["N"], "n": d["n_eff"], "k": d["theta"] * d["n_eff"], "design": "ht"})
            else:
                strata.append({"N": rec["N"], "n": d["n"], "k": d["theta"] * d["n"], "design": "srs"})
    w = webber_recall(parts, strata, "%s/recall" % SEED_PREFIX, RECALL_DRAWS)
    w["small_boxes_outside"] = small
    w["admitted"] = {"reference_lab": ref_n, "other": other}
    return w


def label_frequency(ctx):
    """Elkan-Noto c per source: among labeller-confirmed targets, the share
    the join labelled a target (Horvitz-Thompson weighted).

    Each sampled item counts once: the joint G2v/G2a design lists a unit in
    both frames once, and a stratum of either frame reads it, so items are
    de-duplicated by item id. The reference lab's verified admitted boxes are
    in no frame (nothing samples them); they enter the target-labelled mass
    at the in-domain verified precision (calibration.json), as in funnel
    recall. Without that precision a source holding such boxes gets no
    estimate (its c would be biased low), with the reason."""
    out = {}
    tot = {}
    seen = set()
    for g in ("G1", "G2", "G2v", "G2a", "G4"):
        for sid in _strata_of(ctx, g):
            rec = ctx.stratum(sid, ctx.hyp_of(sid), "target")
            if not rec["n_labelled"]:
                continue
            backend, level = rec["labeller"], rec["level"]
            for r, unit, y in ctx._labelled(sid, backend, "target", level):
                if y != 1 or r["item_id"] in seen:
                    continue
                pi = float(r["pi"]) if r.get("pi") not in (None, "") else None
                if not pi:
                    continue
                seen.add(r["item_id"])
                src = r["source"]
                t = tot.setdefault(src, {"target": 0.0, "other": 0.0, "n": 0})
                t["target" if g in ("G2", "G2v", "G2a") else "other"] += 1.0 / pi
                t["n"] += 1
    adm = _ledger_admitted(ctx, by_source=True)
    ref_src = adm[2] if adm else {}
    cal = _calib_counts(ctx.calib)
    for src, t in sorted(tot.items()):
        n_ref = int(ref_src.get(src, 0))
        rec = {"n": t["n"], "method": "ht share, CP at n confirmed", "admitted_unsampled": n_ref}
        if n_ref and cal is None:
            rec.update({"estimate": None, "interval": [0.0, 1.0],
                        "why": "%d verified admitted boxes are in no frame and the in-domain verified precision "
                               "(calibration.json) is unavailable" % n_ref})
            out[src] = rec
            continue
        if n_ref:
            t["target"] += n_ref * cal[1] / float(cal[0])
        c = t["target"] / (t["target"] + t["other"]) if (t["target"] + t["other"]) > 0 else None
        lo, hi = clopper_pearson(c * t["n"], t["n"]) if c is not None else (0.0, 1.0)
        rec.update({"estimate": c, "interval": [lo, hi]})
        out[src] = rec
    return out


# ====================================================================== H9 grid
def _thresholds_vectors(ctx):
    """{class id: tau}, {class id: sigma} of the step-1 verifier (a class with
    no threshold is never confident: inf)."""
    vt = ctx.verifier_thresholds or {}
    tau, sig = vt.get("tau_p") or {}, vt.get("sigma") or {}
    out_t, out_s = {}, {}
    for n in ctx.domain.class_names:
        cid = ctx.domain.class_id(n)
        t, s = tau.get(n), sig.get(n)
        out_t[cid] = float("inf") if t is None else float(t)
        out_s[cid] = float("inf") if s is None else float(s)
    return out_t, out_s


def _class_maps(ctx, h3a_ok):
    """{"card": {(source, src_id): class id}, "card_taxonomy": {...}}: maps whose
    G3 unit passes the class gate (purity lb >= 0.8) under the primary labeller."""
    card, tax = {}, {}
    gate = float((ctx.prereg.raw.get("recovery_gates") or {}).get("class_purity_lb_min", 0.8))

    def gate_ok(src, sid_, cls):
        st = "G3/unit=c:%s|%s" % (src, sid_)
        if st not in ctx.stratum_units:
            return False
        s = ctx.stratum(st, "H3b", "purity:%s" % cls)
        d = s["raw"].get("as_no")
        return bool(d and d["interval"][0] >= gate)
    for p in (ctx.class_maps or {}).get("proposals", []) if h3a_ok else []:
        if p.get("via") != "card+geometry" or p.get("status") != "proposed" or p.get("map_to") not in ctx._targets:
            continue
        if gate_ok(p["source"], p["src_id"], p["map_to"]):
            card[(p["source"], str(p["src_id"]))] = ctx.domain.class_id(p["map_to"])
    tax.update(card)
    for n in (ctx.name_status or {}).get("names", []):
        if n.get("status_v2") != "target_synonym" or n.get("via") not in ("scientific", "override"):
            continue
        tx = n.get("taxon") or ""
        for t in ctx.domain.targets:
            tt = t.get("taxon") or ""
            if tx == tt or tx.startswith(tt + " "):
                if gate_ok(n.get("source"), n.get("src_id"), t["name"]):
                    tax[(n.get("source"), str(n.get("src_id")))] = t["id"]
                break
    return {"strict": {}, "card": card, "card_taxonomy": tax}


def h9_grid(ctx, h1_ok, h3a_ok):
    """The 24-cell multiverse of target boxes the harvest yields (contract §6
    H9), with the expected true count per cell."""
    if ctx.ledger_rows is None:
        return [], {"why": "ledger.jsonl unavailable"}
    if not ctx.verifier_thresholds:
        return [], {"why": "the step-1 verifier thresholds are unavailable"}
    roles = _role_ids(ctx)
    size_s = (roles.get("size") or [None])[0]
    tc_s = (roles.get("target_check") or [None])[0]
    ev_s = (roles.get("evidence") or [None])[0]
    other = ctx.domain.other["id"]
    tau, sig = _thresholds_vectors(ctx)
    maps = _class_maps(ctx, h3a_ok)
    boxes = []
    for r in ctx.ledger_rows():
        if r.get("unit") != "box":
            continue
        p = r.get("path") or {}
        boxes.append((r["id"], r["key"], r["source"], r["lab"], int(r["label"]), str(r.get("src_id")),
                      r.get("pred"), r.get("p"), r.get("cos"), p.get(size_s) == "embedded",
                      p.get(ev_s) != "not_evidenced", p.get(tc_s) == "verified"))
    shift = None
    if h1_ok and h3a_ok:
        shift = _shift_thresholds(ctx, boxes, tau)
    cover = {}
    for g in PRECISION_ORDER:
        for u in ctx.frame_groups.get(g, []):
            cover.setdefault(u["unit_id"], u["stratum"])
    # a box whose label a card or taxonomy map changed is judged in the class
    # unit (G3) the map was gated on: its other frames (G1, G4) pool it with
    # boxes the map does not touch, whose mapped-label precision is zero
    g3_of = {u["unit_id"]: u["stratum"] for u in ctx.frame_groups.get("G3", [])}
    cal = _calib_counts(ctx.calib)
    ref_prec = None
    if cal:
        lo, hi = binom_interval(cal[1], cal[0], ctx.conf)
        ref_prec = {"estimate": cal[1] / float(cal[0]), "lb": lo, "ub": hi}
    prec_cache = {}
    cells = []
    for join in H9_JOINS:
        jmap = maps[join]
        lab_j = [jmap.get((b[2], b[5]), b[4]) if b[4] == other else b[4] for b in boxes]
        for thr in H9_THRESHOLDS:
            if thr == "shift" and shift is None:
                for adm in H9_ADMISSION:
                    for ev in H9_EVIDENCE:
                        cells.append({"join": join, "admission": adm, "thresholds": thr, "evidence": ev,
                                      "boxes": None, "expected_true": None, "status": "not_evaluated",
                                      "uncovered_boxes": None,
                                      "why": "shift thresholds need H1 and H3a supported"})
                continue
            ver, conf_ = [], []
            for b, L in zip(boxes, lab_j):
                pred, p, c = b[6], b[7], b[8]
                if not b[9] or pred is None or p is None:
                    ver.append(False)
                    conf_.append(False)
                    continue
                t_, s_ = (tau, sig) if thr == "in_domain" else shift[b[3]]
                confident = p >= t_.get(pred, float("inf")) and (
                    pred == other or (c is not None and c >= s_.get(pred, float("inf"))))
                ver.append(ctx.domain.is_target(L) and pred == L and confident)
                conf_.append(L == other and ctx.domain.is_target(pred) and confident)
            img_ok = {}
            for b, L, v, cf in zip(boxes, lab_j, ver, conf_):
                ok = img_ok.get(b[1], True)
                if (ctx.domain.is_target(L) and not v) or cf:
                    ok = False
                img_ok[b[1]] = ok
            for adm in H9_ADMISSION:
                for ev in H9_EVIDENCE:
                    tally, uncovered, n_ref = {}, 0, 0
                    total = 0
                    for b, L, v in zip(boxes, lab_j, ver):
                        if not v:
                            continue
                        if adm == "image" and not img_ok[b[1]]:
                            continue
                        if ev == "on" and not b[10]:
                            continue
                        total += 1
                        sidp = g3_of.get(b[0]) if (b[4] == other and L != other) else cover.get(b[0])
                        if sidp is None:
                            if b[11] and b[3] == ctx._ref_lab and b[4] == L:
                                n_ref += 1
                            else:
                                uncovered += 1
                            continue
                        tally[sidp] = tally.get(sidp, 0) + 1
                    est_, lb, ub = 0.0, 0.0, float(uncovered)
                    for sidp, cnt in sorted(tally.items()):
                        pr = prec_cache.get((sidp, join))
                        if pr is None:
                            pr = _stratum_precision(ctx, sidp, jmap)
                            prec_cache[(sidp, join)] = pr
                        if pr is None:
                            uncovered += cnt
                            ub += cnt
                            continue
                        ub += cnt * pr["ub"]
                        if pr["lb"] >= 0.85:
                            est_ += cnt * pr["estimate"]
                            lb += cnt * pr["lb"]
                    if n_ref:
                        if ref_prec is None:
                            uncovered += n_ref
                            ub += n_ref
                        else:
                            ub += n_ref * ref_prec["ub"]
                            if ref_prec["lb"] >= 0.85:
                                est_ += n_ref * ref_prec["estimate"]
                                lb += n_ref * ref_prec["lb"]
                    cells.append({"join": join, "admission": adm, "thresholds": thr, "evidence": ev,
                                  "boxes": total, "expected_true": {"estimate": est_, "lb": lb, "ub": ub},
                                  "status": "evaluated", "uncovered_boxes": uncovered,
                                  "reference_boxes": n_ref})
    info = {"maps": {k: len(v) for k, v in maps.items()}, "shift": None if shift is None else
            {lab: {"tau": {str(k): (None if math.isinf(x) else x) for k, x in sorted(v[0].items())}}
             for lab, v in sorted(shift.items())}}
    return cells, info


def _stratum_precision(ctx, sid, jmap):
    """Precision of the (mapped) labels in one audit stratum: label right and
    box valid; the lower bound takes unsure answers as wrong, the upper as
    right."""
    group = sid.split("/", 1)[0]
    hid = ctx.hyp_of(sid)
    prim = ctx.primary(hid) or {}
    backend, level = prim.get("backend"), prim.get("level")
    if not backend or level is None:
        return None
    other = ctx.domain.other["id"]
    ys = []
    for r in ctx.stratum_items(sid):
        g = (ctx.answers.get(backend) or {}).get(r["item_id"])
        if g is None:
            continue
        u = ctx.unit_row.get(r["unit_id"]) or {}
        lab = u.get("label")
        if lab == ctx.domain.other["name"]:
            ex = json.loads(u["extra"]) if u.get("extra") else {}
            cid = jmap.get((u.get("source"), str(ex.get("src_id"))))
            lab = ctx.domain.class_name(cid) if cid is not None and cid != other else lab
        ys.append((r, ctx.event("label", g, dict(u, label=lab), level)))
    if not ys:
        return None
    N = len(ctx.stratum_units.get(sid, []))
    out = {}
    for name, assign in (("lo", 0), ("hi", 1)):
        vals = [assign if y is None else y for _r, y in ys]
        if group in HT_GROUPS:
            e = ht(vals, [float(r["pi"]) for r, _y in ys], N, ctx.conf)
            out[name] = (e["estimate"], e["interval"])
        else:
            k = float(sum(vals))
            out[name] = (k / len(vals), binom_interval(k, len(vals), ctx.conf))
    return {"estimate": min(max(out["lo"][0], 0.0), 1.0), "lb": out["lo"][1][0], "ub": out["hi"][1][1]}


def _shift_thresholds(ctx, boxes, tau):
    """Learn-then-Test thresholds per lab: for each lab, per class, from the
    gold-labelled rejected target boxes of other labs (label right counts as
    correct, unsure as wrong) and the independent set's J1 scores when the
    file exists. The prototype gate is dropped under shift (-inf)."""
    prim = ctx.primary("H1") or {}
    backend, level = prim.get("backend"), prim.get("level")
    by_unit = {b[0]: b for b in boxes}
    calib = []
    for r in ctx.items_by_group.get("G2", []):
        g = (ctx.answers.get(backend) or {}).get(r["item_id"])
        b = by_unit.get(r["unit_id"])
        if g is None or b is None or b[6] is None or b[6] != b[4]:
            continue
        u = ctx.unit_row.get(r["unit_id"]) or {}
        y = ctx.event("label_right", g, u, level)
        calib.append((b[3], b[4], float(b[7]), y == 1))
    kt = _j1_independent(ctx)
    labs = sorted({b[3] for b in boxes})
    out = {}
    for lab in labs:
        t = dict(tau)
        for cid in ctx.domain.target_ids:
            sc = [(p, c) for (l, k, p, c) in calib if l != lab and k == cid] + \
                 [(p, c) for (k, p, c) in kt if k == cid]
            res = ltt_threshold([p for p, _c in sc], [c for _p, c in sc])
            t[cid] = res["threshold"]
        out[lab] = (t, {cid: float("-inf") for cid in tau})
    return out


def _j1_independent(ctx):
    """[(argmax class, p, correct)] of the step-1 probe on the independent set
    (judges/J1__<set>.npz), when present."""
    if ctx.funnel_dir is None or ctx.known_truth is None:
        return []
    out = []
    for kt in ctx.domain.independent_sets():
        p = Path(ctx.funnel_dir) / "judges" / ("J1__%s.npz" % kt.lower())
        if not p.exists():
            continue
        import numpy as np
        z = np.load(p, allow_pickle=False)
        truth = {}
        for it in ctx.known_truth.get(kt, []):
            if it.get("crop_id") is not None and it.get("truth_kind") in ("target", "attractor"):
                truth[int(it["crop_id"])] = int(it["truth"]) if it.get("truth_kind") == "target" else None
        P = np.asarray(z["P"], dtype=np.float64)
        for idx, cid in enumerate(z["unit_index"].tolist()):
            if int(cid) not in truth:
                continue
            j = int(np.argmax(P[idx]))
            if not ctx.domain.is_target(j):
                continue
            out.append((j, float(P[idx, j]), truth[int(cid)] == j))
    return out


def h9(ctx, cells, info):
    if not cells:
        return _hyp(ctx.prereg, "H9", "not_evaluated", info.get("why") or "no grid")
    ev = [c for c in cells if c["status"] == "evaluated"]
    counts = [c["boxes"] for c in ev if c["boxes"]]
    ratio = (max(counts) / float(min(counts))) if counts and min(counts) > 0 else None
    return _hyp(ctx.prereg, "H9", "descriptive", "multiverse of %d cells (%d evaluated)" % (len(cells), len(ev)),
                estimator="recomputed admission x stratum precision",
                parts={"max_over_min_boxes": ratio, "maps": info.get("maps")})


def h9_prime(ctx, cells):
    R = ctx.R["H9_prime"]
    ev = [c for c in cells if c["status"] == "evaluated"]
    if not ev:
        return _hyp(ctx.prereg, "H9'", "not_evaluated", "no evaluated cell")
    # the best cell is the one with the largest lower bound; the prereg's rule
    # reads that cell's bounds for both verdicts ("best cell LB>=4098",
    # "best cell UB<3073.5"). The largest upper bound of any cell is reported.
    best = sorted(ev, key=lambda c: (-c["expected_true"]["lb"], c["join"], c["admission"], c["thresholds"],
                                     c["evidence"]))[0]
    best_ub = best["expected_true"]["ub"]
    max_ub = max(c["expected_true"]["ub"] for c in ev)
    if best["expected_true"]["lb"] >= R["lb_min"]:
        v = "supported"
    elif best_ub < R["ub_max"]:
        v = "falsified"
    else:
        v = "inconclusive"
    return _hyp(ctx.prereg, "H9'", v, "best cell (%s/%s/%s/%s): lower bound %.1f, upper bound %.1f; largest upper "
                "bound of any cell %.1f" % (best["join"], best["admission"], best["thresholds"], best["evidence"],
                                             best["expected_true"]["lb"], best_ub, max_ub),
                estimator="sum of N_h x precision bounds", estimate=best["expected_true"]["estimate"],
                interval=[best["expected_true"]["lb"], best_ub],
                parts={"best_cell": {k: best[k] for k in ("join", "admission", "thresholds", "evidence")},
                       "max_ub_any_cell": max_ub, "not_evaluated_cells": len(cells) - len(ev)})


def hypothesis(hid, ctx):
    """One hypothesis evaluator by id (H0 ... H12, H9')."""
    fn = {"H0": h0, "H1": h1, "H2a": lambda c: h2(c, "H2a"), "H2b": lambda c: h2(c, "H2b"), "H3a": h3a,
          "H3b": h3b, "H4": h4, "H5a": h5a, "H5b": h5b, "H6": h6, "H7": h7, "H8": h8, "H10": h10,
          "H12": h12}.get(hid)
    if fn is None:
        raise EstimateError("no evaluator for %r (H9, H9' and H11 need the grid and the stage ranking)" % hid)
    return fn(ctx)


# ====================================================================== calibration overlap (R12)
def rl_overlap(ctx, hyp_strata):
    """Every stratum whose labeller scope shares material with it (R12). The
    counted material of a set is its sentinel sample rows' keys."""
    from . import qualify as Q
    indep = set(ctx.domain.independent_sets())
    kt_keys = {}
    for r in ctx.items_by_group.get("sentinel", []):
        for kt in [k for k in str(r.get("kt") or "").replace("+", ";").split(";") if k]:
            kt_keys.setdefault(kt, []).append(Q.item_keys(
                {"id": r["unit_id"], "source": r["source"], "lab": r["lab"], "near_dup3": r["near_dup3"],
                 "provenance": r["provenance"]}, independent=kt in indep))
    out = []
    scopes = ctx.rlq.get("strata_scopes") or {}
    checks = []
    for sid in sorted(ctx.stratum_units):
        if sid.split("/", 1)[0] in ESTIMATION_GROUPS and sid.split("/", 1)[0] != "G5":
            checks.append((sid, scopes.get(sid), "stratum"))
    for hid, sids in sorted(hyp_strata.items()):
        sc = (ctx.primary(hid) or {}).get("scope")
        for sid in sids:
            checks.append((sid, sc, hid))
    seen = set()
    for sid, scope, why in checks:
        if not scope:
            continue
        keys = ctx.stratum_keys(sid)
        for kt in [s for s in scope.split("+") if s]:
            if (sid, kt) in seen:
                continue
            seen.add((sid, kt))
            sh = Q.shares(keys, kt_keys.get(kt, []))
            if any(sh.values()):
                out.append({"rl": kt, "stratum": sid, "scope": scope, "via": why,
                            "shared": {k: v[:5] for k, v in sh.items() if v}})
            if ctx.domain.kt(kt).get("claimed_by"):
                out.append({"rl": kt, "stratum": sid, "scope": scope, "via": why,
                            "shared": {"claimed_by": [ctx.domain.kt(kt)["claimed_by"]]}})
    return out


# ====================================================================== evaluate
def _load_optional(path):
    p = Path(path)
    return read_json(p) if p.exists() else None


def _judge_scores_loader(funnel_dir):
    cache = {}

    def load(judge):
        if judge in cache:
            return cache[judge]
        p = Path(funnel_dir) / "judges" / ("%s__crops.npz" % judge)
        if not p.exists():
            cache[judge] = None
            return None
        import numpy as np
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        P = np.asarray(z["P"], dtype=np.float64)
        idx = z["unit_index"].tolist()
        cache[judge] = {"labels": list(meta.get("labels") or []), "P": {int(c): P[i] for i, c in enumerate(idx)}}
        return cache[judge]
    return load


def _check_lock(pre, fd):
    from . import draw as DR
    from . import DrawError
    lock = pre.sample_lock
    if lock is None:
        raise EstimateError("the prereg holds no sample lock: draw (F6) must run before estimate")
    try:
        sample = DR.load_sample(fd / "sample_v1.csv", pre)
        key = DR.load_key(fd / "sample_v1_key.jsonl", pre)
    except DrawError as e:
        raise EstimateError(str(e))
    fr = file_record(fd / "frames_v1.json")
    if fr["sha256"] != lock.get("frames_sha256"):
        raise StaleInput("frames_v1.json hashes to %s, the lock records %s" % (fr["sha256"][:12],
                                                                           str(lock.get("frames_sha256"))[:12]))
    return sample, key


def build_context(prereg_path, funnel_dir, adapter, domain=None):
    """A Context from the funnel directory, with every refusal of runner
    §5.1.6 (missing labeller qualification or DA file, a sample lock that does
    not match, a stale input)."""
    from . import strata as S
    fd = Path(funnel_dir)
    pre = D.load_prereg(prereg_path)
    dom = D.load(domain) if domain is not None else D.load(pre.domain_name)
    D.check_prereg_domain(pre, dom)
    for need in ("rl_qualification.json", "prospective_da.json"):
        if not (fd / need).exists():
            raise EstimateError("%s is missing: estimate refuses to run without it (runner §5.1.6)" % need)
    sample, key = _check_lock(pre, fd)
    frames = S.load_frames(fd / "frames_v1.json")
    # every threshold comes from the prereg: each upstream artifact must have
    # been made under this prereg's core (runner §3.3)
    check_prereg_core(frames["doc"], pre, "frames_v1.json")
    # the freshness chain (runner §7.3): the census outputs the frames were
    # drawn from, the gold the labeller file was fitted on, the judge files
    check_records(frames["doc"].get("inputs") or {})
    gold_p = fd / "gold_v1.csv"
    if not gold_p.exists():
        raise EstimateError("gold_v1.csv is missing (F7 ingest)")
    rlq = read_json(fd / "rl_qualification.json")
    check_prereg_core(rlq, pre, "rl_qualification.json")
    gold_sha = file_record(gold_p)["sha256"]
    if rlq.get("gold_sha256") != gold_sha:
        raise StaleInput("gold_v1.csv hashes to %s, rl_qualification.json records %s"
                         % (gold_sha[:12], str(rlq.get("gold_sha256"))[:12]))
    check_records(rlq.get("inputs") or {})
    _h, gold = read_csv(gold_p)
    jq = _load_optional(fd / "judge_qualification.json")
    if jq is not None:
        check_prereg_core(jq, pre, "judge_qualification.json")
        check_records(jq.get("inputs") or {})
    census = _load_optional(fd / "census_v1.json")
    if census is not None:
        check_prereg_core(census, pre, "census_v1.json")
    materials = None
    if jq is not None and jq.get("material"):
        from . import qualify as Q
        try:
            materials = Q.judge_materials(fd)
        except FunnelError:
            materials = None
    led_p = fd / "ledger.jsonl"
    from . import ledger as L

    def rows():
        return L.iter_units(led_p)
    known = None
    if adapter is not None:
        try:
            known = adapter.known_truth(dom, fd)
        except Exception as e:
            raise EstimateError("the adapter's known truth could not be read (%s)" % e)
    ctx = Context(dom, pre, frames, sample, key, gold, rlq, jq=jq, materials=materials,
                  census=census, rel_geo=_load_optional(fd / "relation_geometry_v1.json"),
                  rel_audit=_load_optional(fd / "relation_audit_v1.json"), leak=_load_optional(fd / "leak_v1.json"),
                  da=read_json(fd / "prospective_da.json"), calib=_load_optional(STEP1_DIR / "calibration.json"),
                  class_maps=_load_optional(fd / "class_maps.json"),
                  name_status=_load_optional(fd / "name_status_v2.json"),
                  verifier_thresholds=_load_optional(STEP1_DIR / "verifier" / "thresholds.json"),
                  pool_summary=_load_optional(STEP1_DIR / "pool_summary.json"),
                  ledger_rows=rows if led_p.exists() else None,
                  funnel_ledger=_load_optional(fd / "funnel_ledger.json"),
                  judge_scores=_judge_scores_loader(fd), known_truth=known, adapter=adapter, funnel_dir=fd)
    inputs = {"prereg": pre.record(), "sample_v1": file_record(fd / "sample_v1.csv"),
              "sample_v1_key": file_record(fd / "sample_v1_key.jsonl"), "frames_v1": file_record(fd / "frames_v1.json"),
              "gold_v1": file_record(gold_p), "rl_qualification": file_record(fd / "rl_qualification.json"),
              "prospective_da": file_record(fd / "prospective_da.json")}
    for opt in ("judge_qualification.json", "census_v1.json", "relation_geometry_v1.json", "relation_audit_v1.json",
                "leak_v1.json", "class_maps.json", "name_status_v2.json", "funnel_ledger.json", "ledger.jsonl"):
        if (fd / opt).exists():
            inputs[opt.rsplit(".", 1)[0]] = file_record(fd / opt)
    for opt, p in (("calibration", STEP1_DIR / "calibration.json"),
                   ("verifier_thresholds", STEP1_DIR / "verifier" / "thresholds.json"),
                   ("pool_summary", STEP1_DIR / "pool_summary.json")):
        if p.exists():
            inputs[opt] = file_record(p)
    ctx.inputs = inputs
    return ctx


def gate_unit_classes(ctx):
    """{class unit: class} the recovery gates of a class unit are read for:
    recover.unit_classes on the audit's own class maps and name status (the
    one definition, so the audit writes the estimates the gates look up)."""
    if not (ctx.class_maps or ctx.name_status):
        return {}
    from . import recover as RC
    return RC.unit_classes(ctx.class_maps, ctx.name_status, ctx.domain)


def audit_from_context(ctx, testing=False):
    """The audit_v1.json content from a Context."""
    hyps = {}
    for hid in ("H0", "H1", "H2a", "H2b", "H3a", "H3b", "H4", "H5a", "H5b", "H6", "H7", "H8", "H10", "H12"):
        hyps[hid] = hypothesis(hid, ctx)
    stages = stages_block(ctx)
    ranking = stage_ranking(stages)
    rec_stages = [s["id"] for s in (ctx.funnel_ledger or {}).get("stages", []) if s.get("recoverable") is True]
    hyps["H11"] = h11(ctx, ranking, rec_stages)
    h1_ok = hyps["H1"]["verdict"] == "supported"
    h3a_ok = hyps["H3a"]["verdict"] == "supported"
    cells, info = h9_grid(ctx, h1_ok, h3a_ok)
    hyps["H9"] = h9(ctx, cells, info)
    hyps["H9'"] = h9_prime(ctx, cells)
    hyp_strata = {"H1": h1_strata(ctx), "H2a": _g1_frame(ctx, "noinfo"), "H2b": _g1_frame(ctx, "named"),
                  "H4": _strata_of(ctx, "G4")}
    strata_recs = []
    unit_class = gate_unit_classes(ctx)
    for sid in sorted(ctx.stratum_units):
        g = sid.split("/", 1)[0]
        if g not in ESTIMATION_GROUPS:
            continue
        hid = ctx.hyp_of(sid)
        kinds = ["pair" if g == "G5" else "target"] + list(GATE_EVENTS.get(g, ()))
        if g == "G3" and unit_class.get(_stratum_part(sid, "unit")):
            kinds += [ev % unit_class[_stratum_part(sid, "unit")] for ev in UNIT_GATE_EVENTS]
        for kind in kinds:
            rec = dict(ctx.stratum(sid, hid, kind))
            rec.pop("raw", None)
            strata_recs.append(rec)
    overlap, seen = [], set()
    for o in rl_overlap(ctx, hyp_strata) + list(ctx.overlap):
        k = json.dumps(o, sort_keys=True)
        if k not in seen:
            seen.add(k)
            overlap.append(o)
    stop = None
    if hyps["H0"]["verdict"] == "falsified":
        stop = {"rule": "H0 falsified", "at": "F8"}
    elif hyps["H0"]["verdict"] != "supported":
        stop = {"rule": "H0 not passed (%s)" % hyps["H0"]["verdict"], "at": "F8"}
    body = {
        "ledger_fingerprint": (ctx.funnel_ledger or {}).get("fingerprint"),
        "valid": not overlap,
        "calibration_overlap": overlap,
        "rl": {"primary": ctx.rlq.get("primary"), "levels_used": ctx.rlq.get("levels")},
        "strata": strata_recs,
        "stages": stages,
        "funnel_recall": funnel_recall(ctx),
        "label_frequency": label_frequency(ctx),
        "h9_grid": cells,
        "hypotheses": hyps,
        "d18_inputs": d18_inputs(ctx, stages),
        "stage_ranking": ranking,
        "h11": hyps["H11"]["parts"] or None,
        "stop": stop,
        "rules": ctx.R,
    }
    seeds = {"rogan_gladen": "%s/rg/<stratum id>" % SEED_PREFIX, "bootstrap": "%s/boot/<stratum id>" % SEED_PREFIX,
             "recall": "%s/recall" % SEED_PREFIX}
    doc = header("audit", ctx.domain, ctx.prereg, getattr(ctx, "inputs", {}), seeds=seeds,
                 modules=(sys.modules[__name__],), testing=testing)
    doc.update(body)
    doc["funnel_recall"]["seed_text"] = "%s/recall" % SEED_PREFIX
    return json.loads(json_text(finite(doc)))


def finite(obj):
    """A copy with every non-finite float replaced by None (JSON has no
    infinities; an infinite log-loss or threshold means 'none')."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def evaluate(prereg_path, funnel_dir, adapter, domain=None, testing=False, write=True):
    """F8: audit_v1.json and audit_v1.md in funnel_dir. With a calibration
    overlap the file is written with valid false and EstimateError is raised
    (R12): nothing may cite it."""
    ctx = build_context(prereg_path, funnel_dir, adapter, domain=domain)
    audit = audit_from_context(ctx, testing=testing)
    if write:
        fd = Path(funnel_dir)
        write_json_atomic(fd / "audit_v1.json", audit)
        _atomic_write_bytes(fd / "audit_v1.md", render_md(audit).encode("utf-8"))
    if not audit["valid"]:
        raise EstimateError("calibration overlap in %d place(s): audit_v1.json is written with valid=false and "
                            "may not be cited" % len(audit["calibration_overlap"]))
    return audit


# ====================================================================== markdown
def _n(x, fmt="%.4f"):
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return "{:,}".format(x)
    try:
        return fmt % x
    except TypeError:
        return str(x)


def render_md(audit):
    """audit_v1.md: every hypothesis with its verdict, estimator, n and
    interval, in the same form whichever way it came out."""
    L = ["# Funnel audit v1", ""]
    L.append("Built %s. Valid: %s. Ledger fingerprint `%s`." % (audit.get("built_utc"), audit.get("valid"),
                                                               str(audit.get("ledger_fingerprint"))[:16]))
    if audit.get("stop"):
        L.append("")
        L.append("**Stop:** %s (at %s)." % (audit["stop"]["rule"], audit["stop"]["at"]))
    if audit.get("calibration_overlap"):
        L.append("")
        L.append("**Calibration overlap:** %d; this audit may not be cited." % len(audit["calibration_overlap"]))
    L += ["", "## Hypotheses", "", "| Id | Verdict | Estimator | n | Estimate | Interval | Why |",
          "|---|---|---|---|---|---|---|"]
    for hid, h in sorted(audit.get("hypotheses", {}).items(), key=lambda kv: _hkey(kv[0])):
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            hid, h["verdict"], h.get("estimator") or "-", _n(h.get("n")), _n(h.get("estimate")),
            _fmt_iv(h.get("interval")), (h.get("why") or "").replace("|", "/")))
    L += ["", "## Stages", "", "| Stage | Frames | FN rate | Interval | Recoverable (interval) | Holm p | Suspect |",
          "|---|---|---|---|---|---|---|"]
    for s, rec in sorted(audit.get("stages", {}).items()):
        fr = rec.get("fn_rate") or {}
        rc = rec.get("recoverable") or {}
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            s, ",".join(rec.get("frames") or []) or "exact", _n(fr.get("estimate")), _fmt_iv(fr.get("interval")),
            _fmt_iv(rc.get("interval")) if rc else "images %s" % rec.get("recoverable_images"),
            _n(rec.get("p_holm")), rec.get("suspect")))
    fr = audit.get("funnel_recall") or {}
    L += ["", "## Funnel recall", "", "Estimate %s, interval %s (%s draws); small boxes outside the denominator: %s."
          % (_n(fr.get("estimate")), _fmt_iv(fr.get("interval")), fr.get("draws", "-"), fr.get("small_boxes_outside"))]
    L += ["", "## Multiverse (H9)", "", "| Join | Admission | Thresholds | Evidence | Boxes | Expected true (lb, ub) |",
          "|---|---|---|---|---|---|"]
    for c in audit.get("h9_grid", []):
        et = c.get("expected_true") or {}
        L.append("| %s | %s | %s | %s | %s | %s |" % (c["join"], c["admission"], c["thresholds"], c["evidence"],
                                                       _n(c.get("boxes")), "%s (%s, %s)" % (
                                                           _n(et.get("estimate"), "%.1f"), _n(et.get("lb"), "%.1f"),
                                                           _n(et.get("ub"), "%.1f")) if et else c["status"]))
    L.append("")
    return "\n".join(L)


def _hkey(h):
    m = re.match(r"H(\d+)(.*)", h)
    return (int(m.group(1)), m.group(2)) if m else (999, h)
