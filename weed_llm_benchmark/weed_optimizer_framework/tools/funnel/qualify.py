"""Qualification of the machine judges and of the reference labeller, with
the disjointness and circularity rules (contract docs/FUNNEL_AUDIT.md §4.1,
§4.3, DEC-1; runner docs/FUNNEL_AUDIT_RUNNER.md §4.15, §5.4.3, §7.2).

    python -m <package>.tools.funnel qualify --prereg PATH          (F5, machine judges)
    python -m <package>.tools.funnel qualify --rl --prereg PATH     (F7, labeller backends)

Disjointness. Every unit and known-truth item carries four keys (runner
§1.3): source, near_dup3, provenance, lab. `shares` is the only
implementation of "shares": two key sets share when any value of any kind
coincides. A judge or labeller calibrated on material may not estimate a
stratum that shares with it. Items of a known-truth set the config declares
`independent` (a collection of independent observations) are compared per
observation: their source and lab keys are suffixed with the observation (the
unit id up to its last "/"), so two photos of one observation share and two
observations of the collection do not (`item_keys`).

Circularity. Truth used to qualify for a hypothesis never comes from the
population that hypothesis tests: a set whose `claimed_by` names the
hypothesis is refused (`check_circularity`). The reference labeller counts
only sentinels of `qualify_rl_on` sets that share nothing with the stratum
(its scope); claimed sets are reported as agreement only.

Machine judges (F5, `judges`) are scored on known-truth items directly.
  * Positives (Se): target-truth items of the `qualify_rl_on` sets the judge
    may use (never a set whose `never_qualifies` names it).
  * Negatives (Sp): attractor-truth items with their `confused_with` targets
    as probes, and target-truth items with siblings (probe = the sibling).
    For the stratum type `other_named` the attractors come from the
    independent sets; for the other types from the claimed sets (runner
    §4.15).
  * precision_at_half = Se / (Se + 1 - Sp), its lower bound from Se.lb and
    Sp.lb; rescue = accuracy on the items the step-1 probe got wrong in the
    independent sets and the sets whose allowed uses name "rescue" (contract
    §4.3: the copy set and the independent set); P(wrong | probe wrong) next
    to P(wrong); agreement with the claimed labels is reported apart.
  * The step-1 probe (kind step1_probe) never qualifies and its material is
    flagged judges_nothing: it is never a judge of its own discards, and
    allowed_judges never lists it.
  * Two judges whose errors on the common items correlate at phi >= the
    prereg's cut count as one (correlated_groups).
  * Qualification is computed on all usable items ("all") and, for every lab
    value of the material, with that lab's items left out ("not:<lab>"), so a
    judge can be used on a stratum of that lab without having been calibrated
    on it (`material_for`).

Reference labeller (F7, `rl`): Se and Sp per backend, per scope and per
level (species, genus, plant) from the gold rows of sentinels and identity
items only; unsure, invalid and unparsed answers count as wrong. A backend is
qualified at a level when Se.lb and Sp.lb both reach the prereg minima; the
level used is the finest qualified one. The primary backend per hypothesis is
the qualified one with the larger Se.lb + Sp.lb at the needed level, in the
scope of all the hypothesis's strata (a tie goes to the platform-native
backend, DEC-1). Se and Sp are also given per target genus (by_genus), and
each primary lists the genera its backend does not qualify for at species
level (contract §10 stop rules: those genera are bounded).

Bounds: Wilson for n >= 30 and exact Clopper-Pearson below (both from
estimate.py); the lower bound is the one-sided 97.5 % bound.

Nothing here names a domain.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path

from . import (FUNNEL_DIR, CircularityError, FunnelError, QualifyError, SampleLocked, StaleInput,
               canonical_json, check_records, file_record, header, json_text, read_csv, read_json,
               strip_volatile, write_json_atomic)
from . import domain as D

KEYS = ("source", "near_dup3", "provenance", "lab")
TYPES = {"shifted_target": ("G2", "G2v", "G2a"), "other_named": ("G1:named", "G4:named"),
         "other_noinfo": ("G1:noinfo", "G4:noinfo", "G3")}
# The hypotheses each stratum type serves (runner §5.1.6), for the circularity check.
# G3 (other_noinfo) holds the card-resolved units H3a tests as well as H3b's.
TYPE_HYPOTHESES = {"shifted_target": ("H1",), "other_named": ("H2b", "H4"),
                   "other_noinfo": ("H2a", "H3a", "H3b", "H4")}
# Where the negatives of each type come from (runner §4.15).
TYPE_NEGATIVES = {"shifted_target": "claimed", "other_named": "independent", "other_noinfo": "claimed"}
# The strata each hypothesis is estimated on (runner §5.1.6): group and a filter.
HYPOTHESIS_STRATA = {"H0": ("G0", None), "H1": ("G2", "authoritative"), "H2a": ("G1", "frame=noinfo"),
                     "H2b": ("G1", "frame=named"), "H3a": ("G3", "card_resolved"),
                     "H3b": ("G3", "not_card_resolved"), "H4": ("G4", None)}
NEEDED_LEVEL = "species"
LEVELS = D.LEVELS
BAD_ANSWERS = ("unsure", "invalid", "unparsed")
SENTINEL_GROUP = "sentinel"
IDENTITY_GROUP = "identity"
PAIR_SENTINEL_GROUP = "pair_sentinel"
QUALIFY_GROUPS = (SENTINEL_GROUP, IDENTITY_GROUP, PAIR_SENTINEL_GROUP)
POSITIVE_PAIRS = ("same", "consecutive")
NEGATIVE_PAIRS = ("different",)
PLATFORM_BACKEND = "RL-B"            # DEC-1: a tie goes to the platform-native backend
JUDGE_FILE = "judge_qualification.json"
MATERIAL_FILE = "judge_material_v1.json"
RL_FILE = "rl_qualification.json"
GOLD_CSV = "gold_v1.csv"
GOLD_JSON = "gold_v1.json"


def log(msg):
    print("[funnel.qualify] %s" % msg, flush=True)


# ------------------------------------------------------------ disjointness
def _as_sets(keys):
    """{kind: set of str} from one key dict (string values), a dict of
    collections (a material), or a list of key dicts. Empty and missing values
    never share. A material kind given as a single string where the other
    kinds are collections is a hash, which cannot be checked, and is refused."""
    out = {k: set() for k in KEYS}
    if keys is None:
        return out
    colls = (list, tuple, set, frozenset)
    if isinstance(keys, dict) and any(isinstance(keys.get(k), colls) for k in KEYS):
        for k in KEYS:
            v = keys.get(k)
            if isinstance(v, str) and v:
                raise QualifyError("key kind %s is a single string in a material of collections (a hashed "
                                   "material cannot be checked; load the lists with "
                                   "qualify.judge_materials)" % k)
            for x in (v or ()):
                if x not in (None, ""):
                    out[k].add(str(x))
        return out
    rows = [keys] if isinstance(keys, dict) else list(keys)
    for r in rows:
        if not isinstance(r, dict):
            raise QualifyError("keys must be dicts, got %r" % (r,))
        for k in KEYS:
            v = r.get(k)
            if isinstance(v, colls):
                out[k].update(str(x) for x in v if x not in (None, ""))
            elif v not in (None, ""):
                out[k].add(str(v))
    return out


def shares(a_keys, b_keys):
    """{kind: sorted shared values} for the four disjointness kinds; every list
    empty means the two are disjoint (contract §4.1)."""
    a, b = _as_sets(a_keys), _as_sets(b_keys)
    return {k: sorted(a[k] & b[k]) for k in KEYS}


def disjoint(a_keys, b_keys):
    return not any(shares(a_keys, b_keys).values())


def observation_of(unit_id):
    """The observation of an item of an independent set: its unit id up to
    the last "/" (the whole id when there is none)."""
    s = str(unit_id)
    return s.rsplit("/", 1)[0] if "/" in s else s


def item_keys(item, domain=None, independent=None):
    """The four keys of a known-truth item or unit, with the per-observation
    reading for items of an independent set. `independent` overrides the
    membership test (True/False); otherwise it is read from item["kt"]."""
    keys = {k: item.get(k) for k in KEYS}
    ind = independent
    if ind is None and domain is not None:
        kts = item.get("kt")
        kts = [kts] if isinstance(kts, str) else list(kts or [])
        ind = any(k in domain.independent_sets() for k in kts if k)
    if ind:
        obs = observation_of(item.get("id") or item.get("unit_id"))
        for k in ("source", "lab"):
            if keys.get(k) not in (None, ""):
                keys[k] = "%s|%s" % (keys[k], obs)
    return keys


NEVER_JUDGES = "judges_nothing"      # a material flag: the step-1 probe never judges its own discards


def allowed_judges(stratum_keys, judge_material):
    """The judges whose calibration material shares nothing with the stratum.

    judge_material maps a judge id to its material: a key dict of
    collections (source/sources, near_dup3, provenance, lab/labs), or a
    judge entry of judge_qualification.json. For an entry, the lab scope that
    leaves the stratum's lab out is used when the stratum lies in one lab and
    that scope exists; its hashed near_dup3 and provenance are read from the
    material file the entry names (judge_material_v1.json), checked against
    the hashes. A material flagged judges_nothing (the step-1 probe, whose
    reject class was fitted on pool boxes) is never allowed anywhere."""
    out = []
    labs = sorted(_as_sets(stratum_keys)["lab"])
    for judge in sorted(judge_material):
        mat = judge_material[judge]
        if isinstance(mat, dict) and "calibration_material" in mat:
            if mat.get("kind") == "step1_probe":
                continue
            scoped = (mat.get("by_lab_scope") or {}).get("not:%s" % labs[0]) if len(labs) == 1 else None
            mat = (scoped or {}).get("calibration_material") or mat["calibration_material"]
        if isinstance(mat, dict) and mat.get(NEVER_JUDGES):
            continue
        if disjoint(stratum_keys, _material_keys(mat)):
            out.append(judge)
    return out


_LISTS_CACHE = {}


def _material_lists(rec):
    """The lists a calibration-material record's "lists" entry names."""
    path, sha = rec.get("path"), rec.get("sha256")
    key = (path, sha)
    if key not in _LISTS_CACHE:
        p = Path(path or "")
        if not p.is_file():
            raise QualifyError("judge material file %s is missing; disjointness cannot be checked" % path)
        if file_record(p)["sha256"] != sha:
            raise StaleInput("judge material file %s does not hash to its record" % p)
        _LISTS_CACHE[key] = read_json(p)["judges"]
    return _LISTS_CACHE[key][rec["judge"]][rec["scope"]]


def _hash_list(values):
    return hashlib.sha256(canonical_json(sorted(values)).encode("utf-8")).hexdigest()


def _material_keys(mat):
    if not isinstance(mat, dict):
        raise QualifyError("a judge material must be a dict, got %r" % (type(mat).__name__,))
    lists = None
    out = {}
    for k, alias in (("source", "sources"), ("lab", "labs"), ("near_dup3", "near_dup3"),
                     ("provenance", "provenance")):
        v = mat.get(k, mat.get(alias))
        if isinstance(v, str) or (isinstance(v, dict) and "sha256" in v):
            if not isinstance(mat.get("lists"), dict):
                raise QualifyError("judge material %s is recorded as a hash and names no material file; "
                                   "disjointness cannot be checked from it" % k)
            lists = lists or _material_lists(mat["lists"])
            full = lists.get(k) or []
            want = v if isinstance(v, str) else v["sha256"]
            if _hash_list(full) != want:
                raise StaleInput("judge material %s lists do not hash to the recorded %s" % (k, want[:12]))
            v = full
        out[k] = list(v or [])
    return out


def material_of(items, domain=None):
    """{"source", "near_dup3", "provenance", "lab"} sorted lists over items."""
    sets = _as_sets([item_keys(i, domain) for i in items])
    return {k: sorted(sets[k]) for k in KEYS}


def scope_for(stratum_keys, domain, kt_keys):
    """The "+"-joined qualify_rl_on sets that share nothing with the stratum
    (runner §4.15); "" when none is left. kt_keys maps a KT id to the keys of
    its counted material (its sentinel items)."""
    out = []
    for kt in domain.qualify_rl_on():
        if kt not in kt_keys:
            continue
        if disjoint(stratum_keys, kt_keys[kt]):
            out.append(kt)
    return "+".join(out)


def scope_sets(scope):
    return [s for s in str(scope or "").split("+") if s]


def check_circularity(hypothesis, sets, domain):
    """CircularityError when a set in `sets` holds the claimed labels the
    hypothesis tests (known_truth.<set>.claimed_by)."""
    for s in sets:
        cb = domain.kt(s).get("claimed_by")
        if cb and cb == hypothesis:
            raise CircularityError("known-truth set %s holds the claimed labels %s tests; it never "
                                   "qualifies for %s (contract §4.1)" % (s, hypothesis, hypothesis))


def judge_may_use(judge, kind, kt_id, domain):
    """CircularityError when a judge may not be qualified on a set: the
    step-1 probe never judges its own discards, and a set's never_qualifies
    list names the judges it never qualifies (contract §4.1, §4.3)."""
    never = domain.kt(kt_id).get("never_qualifies") or []
    if kind == "step1_probe" or judge in never:  # funnel-mutation: M2
        raise CircularityError("judge %s (%s) may not be qualified on %s: %s"
                               % (judge, kind, kt_id, "the step-1 probe never judges its own discards"
                                  if kind == "step1_probe" else "%s never qualifies it" % kt_id))


# ------------------------------------------------------------------ bounds
def est(k, n):
    """{"k", "n", "estimate", "lb", "ub", "method"}: Wilson for n >= 30,
    exact Clopper-Pearson below; lb is the one-sided 97.5 % bound."""
    from . import estimate as E
    k, n = int(k), int(n)
    if n < 0 or k < 0 or k > n:
        raise QualifyError("a proportion needs 0 <= k <= n (k=%d, n=%d)" % (k, n))
    if n == 0:
        return {"k": 0, "n": 0, "estimate": None, "lb": 0.0, "ub": 1.0, "method": "none"}
    if n >= 30:
        lo, hi = E.binom_interval(k, n)
        method = "wilson"
    else:
        lo, hi = E.clopper_pearson(k, n)
        method = "clopper_pearson"
    return {"k": k, "n": n, "estimate": k / n, "lb": float(lo), "ub": float(hi), "method": method}


def se_sp(pairs):
    """(se est, sp est) from trials (truth, probe, answer).

    A trial is positive when probe == truth: correct when answer == probe.
    Otherwise it is negative (probe may be a label or a collection of labels,
    the item's confusions): correct when the answer is none of the probes.
    An answer of None (unsure, invalid, unparsed) is wrong either way."""
    kp = npos = kn = nneg = 0
    for truth, probe, answer in pairs:
        probes = set(probe) if isinstance(probe, (list, tuple, set, frozenset)) else {probe}
        if len(probes) == 1 and truth in probes:
            npos += 1
            kp += int(answer is not None and answer == truth)
        else:
            if truth in probes:
                raise QualifyError("a negative trial names its own truth %r as a probe" % (truth,))
            nneg += 1
            kn += int(answer is not None and answer not in probes)
    return est(kp, npos), est(kn, nneg)


def phi(errors_a, errors_b):
    """phi coefficient of two paired error indicators; NaN when a margin is
    empty (no correlation can be measured)."""
    a = [bool(x) for x in errors_a]
    b = [bool(x) for x in errors_b]
    if len(a) != len(b):
        raise QualifyError("phi needs paired indicators (%d vs %d)" % (len(a), len(b)))
    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n00 = len(a) - n11 - n10 - n01
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if den == 0:
        return float("nan")
    return (n11 * n00 - n10 * n01) / math.sqrt(den)


def precision_at_half(se, sp):
    """Se / (Se + 1 - Sp) at prevalence 1/2; lb from the two lower bounds, ub
    from the two upper bounds."""
    def f(s, p):
        if s is None or p is None:
            return None
        d = s + 1.0 - p
        return s / d if d > 0 else 0.0
    point = f(se["estimate"], sp["estimate"])
    return {"estimate": point, "lb": f(se["lb"], sp["lb"]) or 0.0, "ub": f(se["ub"], sp["ub"]),
            "k": None, "n": min(se["n"], sp["n"])}


# ------------------------------------------------------------ level labels
def _genus(taxon):
    t = str(taxon or "").strip()
    return t.split()[0] if t else None


def target_genera(domain):
    return sorted({_genus(t.get("taxon")) for t in domain.targets if _genus(t.get("taxon"))})


def truth_label(truth, truth_kind, truth_taxon, level, domain):
    """The truth of an item at a level: species -> class name / "other" /
    "non_object"; genus -> a target genus / "other" / "non_object"; plant ->
    "plant" / "non_object"."""
    if truth_kind == "non_object":
        return "non_object"
    if level == "plant":
        return "plant" if truth_kind in ("target", "attractor", "other") else None
    if level == "species":
        if truth_kind == "target":
            return domain.class_name(int(truth))
        return "other" if truth_kind in ("attractor", "other") else None
    if level == "genus":
        genera = target_genera(domain)
        if truth_kind == "target":
            taxon = domain.target(int(truth)).get("taxon") or truth_taxon
            return _genus(taxon)
        g = _genus(truth_taxon)
        return g if g in genera else ("other" if truth_kind in ("attractor", "other") else None)
    raise QualifyError("unknown level %r" % (level,))


def answer_label(answer, answer_taxon, answer_level, level, domain):
    """An answer (a gold row's answer, answer_taxon, answer_level) at a level;
    None for an answer that is wrong at every level (unsure, invalid,
    unparsed) and for a genus answer in a genus the class policy leaves
    unsure (contract §5.1)."""
    if answer in BAD_ANSWERS or answer in (None, ""):
        return None
    if answer == "non_object":
        return "non_object"
    if level == "plant":
        return "plant"
    genus_answer = answer_level == "genus"
    if level == "species":
        if genus_answer and _genus(answer_taxon) in domain.genus_unsure():
            return None
        if answer in domain.target_names:
            return answer
        return "other" if answer == "other" else None
    if level == "genus":
        genera = target_genera(domain)
        if answer in domain.target_names:
            return _genus(domain.target(answer).get("taxon"))
        g = _genus(answer_taxon)
        if g and g in genera:
            return g
        return "other" if answer == "other" else None
    raise QualifyError("unknown level %r" % (level,))


def _probes(truth, truth_kind, truth_taxon, level, domain, attractor_confusions):
    """The negative probes of an item at a level (its confusions), [] if none."""
    probes = set()
    if level == "plant":
        return ["plant"] if truth_kind == "non_object" else []
    confused = []
    if truth_kind == "target":
        confused = list(domain.target(int(truth)).get("siblings") or [])
    elif truth_kind == "attractor":
        confused = list(attractor_confusions.get(truth_taxon) or [])
    own = truth_label(truth, truth_kind, truth_taxon, level, domain)
    for name in confused:
        lab = name if level == "species" else _genus(domain.target(name).get("taxon"))
        if lab and lab != own:
            probes.add(lab)
    return sorted(probes)


def attractor_confusions(domain):
    """{attractor taxon: [target names]} from the config."""
    return {a["taxon"]: list(a.get("confused_with") or []) for a in domain.attractors}


def trials(item, answer, level, domain, confusions=None):
    """The Se and Sp trials [(truth, probe, answer)] of one item at a level.
    item: {"truth", "truth_kind", "truth_taxon"}; answer: the level label (or
    None)."""
    confusions = attractor_confusions(domain) if confusions is None else confusions
    tk = item.get("truth_kind")
    truth = item.get("truth")
    if tk == "target" and truth in (None, ""):
        raise QualifyError("target item %s has no truth class" % (item.get("id") or item.get("item_id")))
    truth = int(truth) if tk == "target" else truth
    tl = truth_label(truth, tk, item.get("truth_taxon"), level, domain)
    out = []
    positive = (tl is not None and ((level == "species" and tk == "target")
                                    or (level == "genus" and tk in ("target", "attractor")
                                        and tl in target_genera(domain))
                                    or (level == "plant" and tl == "plant")))
    if positive:
        out.append((tl, tl, answer))
    probes = _probes(truth, tk, item.get("truth_taxon"), level, domain, confusions)
    if probes and tl is not None:
        out.append((tl, tuple(probes), answer))
    return out


# ------------------------------------------------------------ machine judges
def negative_sets(typ, domain):
    """The known-truth sets whose attractor items are a type's negatives: the
    independent sets for other_named, else the claimed sets, never one whose
    claimed labels a hypothesis of the type tests (runner §4.15)."""
    if TYPE_NEGATIVES[typ] == "independent":
        return domain.independent_sets()
    return [s for s in domain.claimed_sets()
            if domain.kt(s).get("claimed_by") not in TYPE_HYPOTHESES[typ]]


def _usable_sets(judge, kind, sets, domain, refusals):
    ok = []
    for s in sets:
        try:
            judge_may_use(judge, kind, s, domain)
        except CircularityError as e:
            refusals.append({"set": s, "why": str(e)})
            continue
        ok.append(s)
    return ok


def _set_role_items(kt_items, sets, truth_kind):
    out = []
    for s in sets:
        for it in kt_items.get(s, []):
            if it.get("truth_kind") == truth_kind:
                out.append(dict(it, kt=s))
    return out


RESCUE_USE = "rescue"


def rescue_set_ids(domain):
    """The known-truth sets the rescue rate is measured on (contract §4.3:
    J1's errors on the copy set and the independent set, whose truth does not
    come from the audited sources): the qualify_rl_on sets that are
    independent or list "rescue" among their allowed uses."""
    ind = set(domain.independent_sets())
    return [s for s in domain.qualify_rl_on()
            if s in ind or RESCUE_USE in (domain.kt(s).get("allowed_uses") or [])]


def _call_label(call, level, domain):
    """A judge's top-1 label name (a target name, "other" or "non_object") at a level."""
    if call is None:
        return None
    if call == "non_object":
        return "non_object"
    if level == "plant":
        return "plant"
    if call in domain.target_names:
        return call if level == "species" else _genus(domain.target(call).get("taxon"))
    return "other" if call == "other" else None


def _qualify_block(judge, kind, typ, items_by_role, calls, j1_wrong, domain, thresholds, confusions):
    pos, neg, rescue_items = items_by_role
    pairs = []
    used = []
    n_attr = sum(1 for it in neg if it["id"] in calls)
    for it in pos + neg:
        c = calls.get(it["id"])
        if c is None and it["id"] not in calls:
            continue
        a = _call_label(c, "species", domain)
        pairs.extend(trials(it, a, "species", domain, confusions))
        used.append(it)
    se, sp = se_sp(pairs)
    pah = precision_at_half(se, sp)
    k_r = n_r = 0
    for it in rescue_items:
        if it["id"] not in calls or not j1_wrong.get(it["id"]):
            continue
        n_r += 1
        k_r += int(_correct(it, calls[it["id"]], domain))
    rescue = est(k_r, n_r)
    wrong = [not _correct(it, calls.get(it["id"]), domain) for it in used]
    j1w = [bool(j1_wrong.get(it["id"])) for it in used]
    n_j1w = sum(j1w)
    p_wrong = (sum(wrong) / len(wrong)) if wrong else None
    p_wrong_j1 = (sum(1 for w, j in zip(wrong, j1w) if w and j) / n_j1w) if n_j1w else None
    qualified = bool(se["n"] and sp["n"] and n_attr and pah["lb"] >= thresholds["precision_lb_min"]
                     and rescue["n"] and rescue["lb"] >= thresholds["rescue_lb_min"])
    why = []
    if not se["n"] or not sp["n"]:
        why.append("no positives or no negatives it may be scored on")
    elif not n_attr:
        why.append("no attractor negatives of the set this type is qualified against")
    elif pah["lb"] < thresholds["precision_lb_min"]:
        why.append("precision_at_half lb %.3f < %.2f" % (pah["lb"], thresholds["precision_lb_min"]))
    if not rescue["n"]:
        why.append("no step-1 probe errors to measure rescue on")
    elif rescue["lb"] < thresholds["rescue_lb_min"]:
        why.append("rescue lb %.3f < %.2f" % (rescue["lb"], thresholds["rescue_lb_min"]))
    return {"se": se, "sp": sp, "precision_at_half": pah, "rescue": rescue,
            "p_wrong_given_j1_wrong": p_wrong_j1, "p_wrong": p_wrong, "qualified": qualified,
            "why": "; ".join(why) or "qualified", "n_items": len(used), "n_attractor_negatives": n_attr}, used


def _correct(item, call, domain):
    """Is a species-level call right for a known-truth item?"""
    tl = truth_label(item.get("truth"), item.get("truth_kind"), item.get("truth_taxon"), "species", domain)
    return tl is not None and _call_label(call, "species", domain) == tl


def _thresholds(prereg):
    q = ((prereg.raw.get("judges") or {}).get("qualification") or {})
    for k in ("precision_lb_min_vs_attractors", "rescue_rate_lb_min", "correlated_phi"):
        if k not in q:
            raise QualifyError("prereg judges.qualification.%s is missing" % k)
    return {"precision_lb_min": float(q["precision_lb_min_vs_attractors"]),
            "rescue_lb_min": float(q["rescue_rate_lb_min"]), "phi": float(q["correlated_phi"])}


def _h7_prediction(entry):
    """The prereg's H7 prediction by judge kind: a zero-shot judge and a kNN
    judge on the reference domain alone (session hold-out) are expected to
    fail under shift; a kNN judge with the disjoint (leave-lab-out) bank is
    expected to qualify."""
    if entry.get("kind") == "zero_shot":
        return "fail"
    if entry.get("kind") == "knn":
        return "qualify" if entry.get("holdout") == "disjoint" else "fail"
    return None


def _load_scores(funnel_dir, judge):
    """{("crops"|"kt7", crop_id): label name} of a judge's score files."""
    import numpy as np
    out = {}
    records = {}
    for set_name in ("crops", "kt7"):
        p = Path(funnel_dir) / "judges" / ("%s__%s.npz" % (judge, set_name))
        if not p.is_file():
            continue
        records["%s__%s" % (judge, set_name)] = file_record(p)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            labels = list(meta.get("labels") or [])
            idx = z["unit_index"].astype(np.int64)
            top = z["top"].astype(np.int64)
        if len(idx) != len(top):
            raise QualifyError("%s: unit_index and top differ in length" % p)
        for i, t in zip(idx.tolist(), top.tolist()):
            out[(set_name, int(i))] = labels[t] if 0 <= t < len(labels) else None
    return out, records


def _item_score_key(it):
    cs = it.get("crop_set")
    if it.get("crop_id") in (None, ""):
        return None
    return ("kt7" if cs == "kt7" else "crops", int(it["crop_id"]))


def _j1_calls(adapter, funnel_dir, items, domain):
    """{item id: species-level call} of the step-1 probe on known-truth items:
    the ledger's out-of-fold values for pool boxes, the probe on the step-1
    features for reference and copy crops, and judges/J1__kt7.npz for the
    independent photos."""
    import numpy as np
    out = {}
    want_pool = {}
    want_feat = {}
    for it in items:
        cs = it.get("crop_set")
        if cs == "pool":
            want_pool[it["id"]] = it
        elif cs in ("core", "copy") and it.get("crop_id") not in (None, ""):
            want_feat[it["id"]] = int(it["crop_id"])
    if want_pool:
        led = Path(funnel_dir) / "ledger.jsonl"
        if not led.is_file():
            raise QualifyError("ledger.jsonl is missing: census (F3) must run before qualify")
        with open(led, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("id") in want_pool:
                    p = r.get("pred")
                    out[r["id"]] = None if p is None else _class_call(int(p), domain)
    if want_feat:
        X, _info = adapter.step1_features()
        ids = sorted(want_feat)
        rows = np.array([want_feat[i] for i in ids], dtype=np.int64)
        Xr = np.asarray(X[rows], dtype=np.float32)
        ok = np.isfinite(Xr).all(axis=1)
        if ok.any():
            Xn = Xr[ok] / np.maximum(np.linalg.norm(Xr[ok], axis=1, keepdims=True), 1e-8)
            P, _cos = adapter.j1_scores(Xn)
            top = np.asarray(P).argmax(axis=1)
            for i, t in zip([ids[j] for j in np.flatnonzero(ok)], top.tolist()):
                out[i] = _class_call(int(t), domain)
        for j in np.flatnonzero(~ok):
            out[ids[j]] = None
    kt7_scores, _rec = _load_scores(funnel_dir, "J1")
    for it in items:
        if it.get("crop_set") == "kt7":
            key = _item_score_key(it)
            if key in kt7_scores:
                out[it["id"]] = kt7_scores[key]
    return out


def _class_call(cid, domain):
    if domain.is_target(cid):
        return domain.class_name(cid)
    if cid == domain.other["id"]:
        return "other"
    return None


def declared_keys(kt_id, domain):
    """The key rows of a known-truth set's config-declared sources: a set's
    labelling convention is its sources' (a copy set's labels are its
    reference twins'), so its declared sources and their labs are material
    wherever the set is used."""
    return [{"source": s, "lab": domain.lab_of(s)} for s in (domain.kt(kt_id).get("sources") or [])]


_DECLARED_LABS = {}


def _declared_labs(kt_id, domain):
    key = (domain.sha256, kt_id)
    if key not in _DECLARED_LABS:
        _DECLARED_LABS[key] = frozenset(k["lab"] for k in declared_keys(kt_id, domain))
    return _DECLARED_LABS[key]


def _lab_scopes(items, domain):
    labs = {str(it.get("lab")) for it in items if it.get("lab") not in (None, "")}
    for kt in {it.get("kt") for it in items if it.get("kt")}:
        labs |= _declared_labs(kt, domain)
    return ["all"] + ["not:%s" % lab for lab in sorted(labs)]


def _in_scope(it, scope, domain):
    """Scope "not:<lab>" leaves out every item of that lab, of a source in it,
    and of a set whose declared sources lie in it."""
    if scope == "all":
        return True
    lab = scope[len("not:"):]
    members = set(domain.lab_groups().get(lab, []))
    if it.get("kt") and lab in _declared_labs(it["kt"], domain):
        return False
    return it.get("lab") != lab and it.get("source") not in members


def judges(prereg, domain, funnel_dir=None, adapter=None, force=False, known_truth=None, testing=False):
    """judge_qualification.json (F5, locked) and judge_material_v1.json (the
    material key lists, cluster-only)."""
    from . import adapters as A
    prereg, domain = _load(prereg, domain)
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    if adapter is None:
        adapter = A.load(domain.adapter)
    thresholds = _thresholds(prereg)
    kt_items = known_truth if known_truth is not None else adapter.known_truth(domain, funnel_dir)
    kt_items = {k: [dict(it, kt=k) for it in v] for k, v in kt_items.items()}
    confusions = attractor_confusions(domain)
    panel = ((domain.raw.get("judges") or {}).get("panel") or [])
    rescue_sets_all = rescue_set_ids(domain)
    all_items = [it for v in kt_items.values() for it in v]
    j1 = _j1_calls(adapter, funnel_dir, all_items, domain)
    j1_wrong = {it["id"]: (not _correct(it, j1.get(it["id"]), domain)) for it in all_items
                if it["id"] in j1}
    inputs = {}
    out_judges = {}
    materials = {}
    calls_by_judge = {}
    for entry in panel:
        jid, kind = entry["id"], entry.get("kind")
        if kind == "rl":
            continue                                   # qualified at F7 from sentinels (rl())
        refusals = []
        if kind == "step1_probe":
            calls = j1
        else:
            scores, rec = _load_scores(funnel_dir, jid)
            inputs.update(rec)
            if not rec:
                raise QualifyError("judge %s has no score file under %s/judges (run embed-judges)"
                                   % (jid, funnel_dir))
            calls = {}
            for it in all_items:
                key = _item_score_key(it)
                if key is not None and key in scores:
                    calls[it["id"]] = scores[key]
        calls_by_judge[jid] = calls
        pos_sets = _usable_sets(jid, kind, domain.qualify_rl_on(), domain, refusals)
        by_scope = {}
        mat_scope = {}
        for scope in _lab_scopes(all_items, domain):
            by_type = {}
            used_all = []
            for typ in TYPES:
                neg_sets = _usable_sets(jid, kind, negative_sets(typ, domain), domain, refusals)
                for h in TYPE_HYPOTHESES[typ]:
                    check_circularity(h, pos_sets + neg_sets, domain)
                pos = [it for it in _set_role_items(kt_items, pos_sets, "target") if _in_scope(it, scope, domain)]
                neg = [it for it in _set_role_items(kt_items, neg_sets, "attractor") if _in_scope(it, scope, domain)]
                rescue_sets = _usable_sets(jid, kind, rescue_sets_all, domain, [])
                rescue_items = [it for s in rescue_sets for it in kt_items.get(s, [])
                                if _in_scope(it, scope, domain)]
                block, used = _qualify_block(jid, kind, typ, (pos, neg, rescue_items), calls, j1_wrong,
                                             domain, thresholds, confusions)
                if kind == "step1_probe":
                    block["qualified"] = False
                    block["why"] = "the step-1 probe never judges its own discards"
                block["sets"] = {"positives": pos_sets, "negatives": neg_sets, "rescue": rescue_sets}
                by_type[typ] = block
                used_all.extend(used)
                used_all.extend(it for it in rescue_items if it["id"] in calls)
            bank_mat = _bank_material(funnel_dir, jid, entry, kt_items, domain)
            mat = material_of(used_all + [dict(k, id="", kt=None) for kt in sorted({it["kt"] for it in used_all})
                                          for k in declared_keys(kt, domain)], domain)
            for k in KEYS:
                mat[k] = sorted(set(mat[k]) | set(bank_mat.get(k, [])))
            kts = sorted({it["kt"] for it in used_all} | set(bank_mat.get("kt", [])))
            if kind == "step1_probe":
                # its reject class was fitted on pool boxes: it is never allowed to judge any stratum
                mat[NEVER_JUDGES] = True
            mat_scope[scope] = mat
            by_scope[scope] = {"by_type": by_type, "calibration_material": _material_record(mat, kts),
                               "_scope": scope}
        materials[jid] = mat_scope
        by_scope["all"].pop("_scope", None)
        # agreement with the claimed labels (KT4, KT5, KT6): reported, never a qualification (contract §4.3)
        agreement = {}
        for s in domain.claimed_sets():
            its = [it for it in kt_items.get(s, []) if it["id"] in calls]
            agreement[s] = est(sum(1 for it in its if _correct(it, calls[it["id"]], domain)), len(its))
        out_judges[jid] = {"kind": kind, "calibration_material": by_scope["all"]["calibration_material"],
                           "by_type": by_scope["all"]["by_type"],
                           "by_lab_scope": {s: v for s, v in by_scope.items() if s != "all"},
                           "never_qualifies_on": sorted({r["set"] for r in refusals}),
                           "agreement_only": agreement,
                           "refusals": refusals}
    # phi between judges' error indicators on the common items
    phis = {}
    ids = sorted(calls_by_judge)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            common = [it for it in all_items if it["id"] in calls_by_judge[a] and it["id"] in calls_by_judge[b]
                      and it.get("truth_kind") in ("target", "attractor", "other")]
            ea = [not _correct(it, calls_by_judge[a][it["id"]], domain) for it in common]
            eb = [not _correct(it, calls_by_judge[b][it["id"]], domain) for it in common]
            v = phi(ea, eb)
            phis["%s|%s" % (a, b)] = None if math.isnan(v) else round(v, 6)
    groups = _groups(ids, phis, thresholds["phi"])
    h7 = {}
    for entry in panel:
        pred = _h7_prediction(entry)
        if pred is None:
            continue
        rec = out_judges.get(entry["id"])
        q_types = sorted(t for t, b in (rec or {}).get("by_type", {}).items() if b.get("qualified"))
        h7[entry["id"]] = {"predicted": pred, "qualified": bool(q_types), "types_qualified": q_types}
    material_path = funnel_dir / MATERIAL_FILE
    mat_doc = {"format": "funnel-judge-material/1", "judges": materials}
    mat_sha = hashlib.sha256(json_text(mat_doc).encode("utf-8")).hexdigest()
    for jid, rec in out_judges.items():
        for scope, blk in [("all", rec)] + sorted(rec["by_lab_scope"].items()):
            blk["calibration_material"]["lists"] = {"path": str(material_path), "sha256": mat_sha,
                                                    "judge": jid, "scope": scope}
            blk.pop("_scope", None)
    kt_record = {"sha256": hashlib.sha256(canonical_json(
        {k: sorted(it["id"] for it in v) for k, v in kt_items.items()}).encode("utf-8")).hexdigest(),
        "counts": {k: len(v) for k, v in sorted(kt_items.items())}}
    kt_of = {it["id"]: it["kt"] for it in all_items}
    j1_errors = sorted(i for i, w in j1_wrong.items() if w and kt_of.get(i) in rescue_sets_all)
    doc_body = {"locked": True, "judges": out_judges, "phi": phis, "correlated_groups": groups, "h7": h7,
                "thresholds": thresholds, "known_truth": kt_record,
                "j1_errors": {"sets": rescue_sets_all, "n": len(j1_errors), "items": j1_errors},
                "material": None}
    out_path = funnel_dir / JUDGE_FILE
    existing = _existing(out_path)
    mat_text_sha = mat_sha
    ledger_path = funnel_dir / "ledger.jsonl"
    if ledger_path.is_file():
        inputs["ledger"] = file_record(ledger_path)
    doc = dict(header("judge_qualification", domain, prereg, inputs,
                      modules=(sys.modules[__name__],), testing=testing), **doc_body)
    doc["material"] = {"path": str(material_path), "sha256": mat_text_sha}
    if existing is not None:
        if _same(existing, doc):
            return existing
        if prereg.sample_lock is not None:
            raise SampleLocked("%s is locked by the sample lock; it cannot be rewritten" % out_path)
        if not force:
            raise QualifyError("%s exists and was made from other inputs; rerun with --force" % out_path)
    sha = write_json_atomic(material_path, mat_doc)
    if sha != mat_text_sha:
        raise QualifyError("the judge material file hashed to %s, expected %s" % (sha[:12], mat_text_sha[:12]))
    write_json_atomic(out_path, doc)
    log("judges: %s; correlated groups %s" % (
        {j: sorted(t for t, b in v["by_type"].items() if b["qualified"]) for j, v in out_judges.items()},
        groups))
    return doc


def _bank_material(funnel_dir, judge, entry, kt_items, domain):
    """The part of a kNN bank that counts as calibration material: the whole
    bank unless the judge excludes shared entries per query (holdout
    "disjoint"), which keeps any stratum out of its own bank."""
    if entry.get("kind") != "knn" or entry.get("holdout") == "disjoint":
        return {}
    sets = []
    for spec in entry.get("bank") or []:
        kt, _, role = str(spec).partition(":")
        sets.append((kt, role or None))
    items = []
    for kt, role in sets:
        for it in kt_items.get(kt, []):
            if role and it.get("role") != role:
                continue
            items.append(it)
    mat = material_of(items + [dict(k, id="", kt=None) for kt, _r in sets for k in declared_keys(kt, domain)], domain)
    mat["kt"] = sorted({kt for kt, _ in sets})
    return mat


def _material_record(mat, kts):
    def h(values):
        return hashlib.sha256(canonical_json(sorted(values)).encode("utf-8")).hexdigest()
    rec = {"kt": kts, "sources": mat["source"], "labs": mat["lab"],
           "near_dup3": h(mat["near_dup3"]), "provenance": h(mat["provenance"]),
           "n_near_dup3": len(mat["near_dup3"]), "n_provenance": len(mat["provenance"])}
    if mat.get(NEVER_JUDGES):
        rec[NEVER_JUDGES] = True
    return rec


def _groups(ids, phis, cut):
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for pair, v in phis.items():
        if v is not None and v >= cut:
            a, b = pair.split("|")
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comp = collections.defaultdict(list)
    for i in ids:
        comp[find(i)].append(i)
    return sorted(sorted(v) for v in comp.values())


def judge_materials(funnel_dir=None):
    """{judge: {scope: material lists}} from judge_material_v1.json, checked
    against the sha256 judge_qualification.json records."""
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    jq = read_json(funnel_dir / JUDGE_FILE)
    rec = jq.get("material") or {}
    p = Path(rec.get("path") or funnel_dir / MATERIAL_FILE)
    if not p.is_file():
        p = funnel_dir / MATERIAL_FILE
    got = file_record(p)["sha256"]
    if got != rec.get("sha256"):
        raise StaleInput("%s does not hash to the sha256 %s records" % (p, JUDGE_FILE))
    return read_json(p)["judges"]


def material_for(judge, stratum_keys, jq, materials):
    """(scope, material lists, by_type) a judge uses on a stratum: the lab
    scope that leaves the stratum's lab out when the stratum lies in one lab
    and that scope exists, else "all"."""
    labs = sorted(_as_sets(stratum_keys)["lab"])
    entry = jq["judges"][judge]
    scope = "all"
    if len(labs) == 1 and "not:%s" % labs[0] in entry.get("by_lab_scope", {}):
        scope = "not:%s" % labs[0]
    by_type = entry["by_type"] if scope == "all" else entry["by_lab_scope"][scope]["by_type"]
    return scope, materials[judge][scope], by_type


def qualified_judges(stratum_type, stratum_keys, jq, materials):
    """[(judge, scope)] qualified for the stratum type in the scope whose
    material shares nothing with the stratum (R-J, the PPI predictor)."""
    out = []
    for judge in sorted(jq.get("judges", {})):
        scope, mat, by_type = material_for(judge, stratum_keys, jq, materials)
        blk = by_type.get(stratum_type) or {}
        if blk.get("qualified") and not mat.get(NEVER_JUDGES) and disjoint(stratum_keys, mat):
            out.append((judge, scope))
    return out


# --------------------------------------------------------- the labeller (F7)
def _row_kts(r):
    v = r.get("kt")
    if v in (None, ""):
        return []
    return [x for x in str(v).replace("+", ";").split(";") if x]


def counted_rows(rows, count_sets):
    """The rows whose known-truth sets all lie in count_sets: the sentinels a
    scope may count (a claimed set is never among them)."""
    out = []
    count_sets = set(count_sets)
    for r in rows:
        kts = _row_kts(r)
        if not kts or not set(kts) <= count_sets:  # funnel-mutation: M6
            continue
        out.append(r)
    return out


def _check_qualify_rows(rows):
    """Qualification reads sentinels, identity items and pair sentinels only:
    a threshold is never fitted on the audit sample (mutation point M3)."""
    for r in rows:
        if r.get("group") not in QUALIFY_GROUPS:  # funnel-mutation: M3
            raise QualifyError("gold row %s is from group %r: qualification reads sentinel, identity "
                               "and pair-sentinel rows only, never the audit sample"
                               % (r.get("item_id"), r.get("group")))


def _item_of(r):
    t = r.get("truth")
    return {"id": r.get("unit_id"), "item_id": r.get("item_id"),
            "truth": None if t in (None, "") else int(t), "truth_kind": r.get("truth_kind"),
            "truth_taxon": r.get("truth_taxon") or None}


def rl_block(rows, level, domain, thresholds, confusions=None):
    """{"se", "sp", "qualified"} of one backend's rows at one level."""
    pairs = []
    for r in rows:
        a = answer_label(r.get("answer"), r.get("answer_taxon"), r.get("answer_level"), level, domain)
        pairs.extend(trials(_item_of(r), a, level, domain, confusions))
    se, sp = se_sp(pairs)
    q = bool(se["n"] and sp["n"] and se["lb"] >= thresholds["se_lb_min"] and sp["lb"] >= thresholds["sp_lb_min"])
    return {"se": se, "sp": sp, "qualified": q}


def genus_rows(rows, domain, confusions=None):
    """{target genus: [gold rows about it]}: the sentinels whose truth is a
    target of the genus, or an attractor confused with such a target."""
    confusions = attractor_confusions(domain) if confusions is None else confusions
    genus_of = {t["name"]: _genus(t.get("taxon")) for t in domain.targets}
    out = collections.defaultdict(list)
    for r in rows:
        tk = r.get("truth_kind")
        gs = set()
        if tk == "target" and _int(r.get("truth")) is not None and domain.is_target(_int(r.get("truth"))):
            gs.add(genus_of[domain.class_name(_int(r.get("truth")))])
        elif tk == "attractor":
            gs.update(genus_of[n] for n in confusions.get(r.get("truth_taxon") or None, []) if n in genus_of)
        for g in gs:
            if g:
                out[g].append(r)
    return dict(out)


def pair_block(rows, thresholds):
    kp = npos = kn = nneg = 0
    for r in rows:
        truth = r.get("pair_truth")
        a = r.get("answer")
        if truth in POSITIVE_PAIRS:
            npos += 1
            kp += int(a in POSITIVE_PAIRS)
        elif truth in NEGATIVE_PAIRS:
            nneg += 1
            kn += int(a in NEGATIVE_PAIRS)
    se, sp = est(kp, npos), est(kn, nneg)
    return {"se": se, "sp": sp,
            "qualified": bool(npos and nneg and se["lb"] >= thresholds["se_lb_min"]
                              and sp["lb"] >= thresholds["sp_lb_min"])}


def identity_check(rows, check):
    """{"n", "as_class", "as_excluded", "share", "pass"} of one identity check
    (contract §7: it passes when no crop is answered as an excluded taxon and
    the share answered as the class reaches pass_share_min over at least n
    answered crops)."""
    n = len(rows)
    as_class = sum(1 for r in rows if r.get("answer") == check["class"])
    excluded = set(check.get("excluded_taxa") or [])
    as_excl = sum(1 for r in rows if r.get("answer_taxon") in excluded)
    share = (as_class / n) if n else 0.0
    ok = bool(n >= int(check["n"]) and as_excl == 0 and share >= float(check["pass_share_min"]))
    return {"n": n, "as_class": as_class, "as_excluded": as_excl, "share": share, "pass": ok}


def _rl_thresholds(prereg):
    q = ((prereg.raw.get("reference_labeller") or {}).get("qualification") or {})
    for k in ("se_lb_min", "sp_lb_min"):
        if k not in q:
            raise QualifyError("prereg reference_labeller.qualification.%s is missing" % k)
    return {"se_lb_min": float(q["se_lb_min"]), "sp_lb_min": float(q["sp_lb_min"])}


def _frames_strata(funnel_dir):
    """{stratum: {"group", "keys": [row key dicts], "sources": set}} from
    frames_v1.json and its csvs (each csv's sha256 checked)."""
    fj = Path(funnel_dir) / "frames_v1.json"
    frames = read_json(fj)
    out = {}
    for g, info in sorted((frames.get("groups") or {}).items()):
        rec = info.get("file") or {}
        p = Path(rec.get("path") or (Path(funnel_dir) / "frames_v1" / ("%s.csv" % g)))
        if not p.is_file():
            p = Path(funnel_dir) / "frames_v1" / ("%s.csv" % g)
        if rec.get("sha256") and file_record(p)["sha256"] != rec["sha256"]:
            raise StaleInput("frame file %s does not hash to what frames_v1.json records" % p)
        _h, rows = read_csv(p)
        for r in rows:
            s = out.setdefault(r["stratum"], {"group": g, "keys": [], "units": []})
            s["keys"].append({k: r.get(k) for k in KEYS})
            s["units"].append(r.get("unit_id"))
    return out


def _stratum_part(stratum, key):
    for seg in str(stratum).split("/")[1:]:
        k, _, v = seg.partition("=")
        if k == key:
            return v
    return None


def _card_resolved_sources(domain):
    res = ((domain.raw.get("sources") or {}).get("card_resolvers") or {})
    return sorted(s for s, v in res.items() if isinstance(v, dict) and v.get("class_table"))


def _unit_source(unit):
    u = str(unit or "")
    if u[:2] in ("c:", "k:", "s:"):
        return u[2:].split("|")[0]
    return None


def hypothesis_strata(strata, domain):
    """{hypothesis: [stratum ids]} by the runner's estimator table."""
    auth = set(domain.authoritative_sources())
    card = set(_card_resolved_sources(domain))
    out = {}
    for h, (group, filt) in HYPOTHESIS_STRATA.items():
        sel = []
        for s, info in sorted(strata.items()):
            if info["group"] != group:
                continue
            srcs = {k.get("source") for k in info["keys"]} | {_unit_source(u) for u in info["units"]}
            srcs.discard(None)
            srcs.discard("")
            if filt == "authoritative" and not (srcs & auth):
                continue
            if filt == "card_resolved" and not (srcs & card):
                continue
            if filt == "not_card_resolved" and (srcs & card):
                continue
            if filt and filt.startswith("frame=") and _stratum_part(s, "frame") != filt[len("frame="):]:
                continue
            sel.append(s)
        out[h] = sel
    return out


def rl(prereg, domain, funnel_dir=None, rows=None, testing=False, force=False):
    """rl_qualification.json (F7, after ingest). `rows` injects gold rows
    (tests); they must all be sentinel, identity or pair-sentinel rows."""
    prereg, domain = _load(prereg, domain)
    funnel_dir = Path(funnel_dir or FUNNEL_DIR)
    thresholds = _rl_thresholds(prereg)
    inputs = {}
    if rows is None:
        gold_json = read_json(funnel_dir / GOLD_JSON)
        gp = funnel_dir / GOLD_CSV
        if file_record(gp)["sha256"] != gold_json.get("gold_sha256"):
            raise StaleInput("%s does not hash to the sha256 %s records" % (gp, GOLD_JSON))
        check_records(gold_json.get("inputs"))
        inputs["gold"] = file_record(gp)
        inputs["gold_record"] = file_record(funnel_dir / GOLD_JSON)
        _h, all_rows = read_csv(gp)
        rows = [r for r in all_rows if r.get("group") in QUALIFY_GROUPS
                or str(r.get("is_sentinel")) in ("1", "true", "True")]
        rows = [dict(r, group=(PAIR_SENTINEL_GROUP if r.get("group") == SENTINEL_GROUP and _pair_row(r)
                               else r.get("group"))) for r in rows]
        gold_sha = inputs["gold"]["sha256"]
    else:
        rows = [dict(r) for r in rows]
        gold_sha = hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()
    _check_qualify_rows(rows)
    confusions = attractor_confusions(domain)
    fj = funnel_dir / "frames_v1.json"
    lock = prereg.sample_lock or {}
    if lock.get("frames_sha256") and (not fj.is_file() or file_record(fj)["sha256"] != lock["frames_sha256"]):
        raise StaleInput("%s is missing or is not the frames the sample lock records" % fj)
    strata = _frames_strata(funnel_dir) if fj.is_file() else {}
    if strata:
        inputs["frames"] = file_record(fj)
    box_rows = [r for r in rows if r["group"] in (SENTINEL_GROUP, IDENTITY_GROUP) and not _pair_row(r)]
    pair_rows = [r for r in rows if r["group"] == PAIR_SENTINEL_GROUP or _pair_row(r)]
    sentinel_rows = [r for r in box_rows if r["group"] == SENTINEL_GROUP]
    # the counted material of each qualify_rl_on set: its sentinel items' keys
    kt_keys = collections.defaultdict(list)
    for r in sentinel_rows:
        for k in _row_kts(r):
            kt_keys[k].append(item_keys({"id": r.get("unit_id"), **{x: r.get(x) for x in KEYS}},
                                        independent=k in domain.independent_sets()))
    for k in list(kt_keys):
        kt_keys[k].extend(declared_keys(k, domain))
    # frame rows of an independent set's source are read per observation, as its sentinels are
    ind = set(domain.independent_sets())
    ind_sources = {r.get("source") for r in sentinel_rows if set(_row_kts(r)) & ind and r.get("source")}
    for info in strata.values():
        info["keys"] = [item_keys(dict(k, id=u), independent=True) if k.get("source") in ind_sources else k
                        for k, u in zip(info["keys"], info["units"])]
    strata_scopes = {s: scope_for(info["keys"], domain, kt_keys) for s, info in sorted(strata.items())}
    hyp = hypothesis_strata(strata, domain)
    # a hypothesis spans several strata: its scope is the sets that share nothing with any of them, which
    # need not be any single stratum's scope, so it is qualified in its own right
    hyp_scopes = {h: (scope_for([k for s in sel for k in strata[s]["keys"]], domain, kt_keys) if sel else "")
                  for h, sel in sorted(hyp.items())}
    scopes = sorted(set(strata_scopes.values()) | set(v for v in hyp_scopes.values() if v)
                    | {"+".join(s for s in domain.qualify_rl_on() if s in kt_keys)})
    backends = sorted({r.get("backend") for r in box_rows + pair_rows if r.get("backend")})
    out_b, levels, by_genus = {}, {}, {}
    for b in backends:
        brows = [r for r in sentinel_rows if r.get("backend") == b]
        out_b[b], levels[b] = {}, {}
        for scope in scopes:
            if not scope:
                out_b[b][scope] = {lv: {"se": est(0, 0), "sp": est(0, 0), "qualified": False} for lv in LEVELS}
                levels[b][scope] = None
                continue
            for s in scope_sets(scope):
                if domain.kt(s).get("claimed_by"):
                    raise CircularityError("scope %s holds the claimed set %s" % (scope, s))
            counted = counted_rows(brows, scope_sets(scope))
            out_b[b][scope] = {lv: rl_block(counted, lv, domain, thresholds, confusions) for lv in LEVELS}
            q = [lv for lv in LEVELS if out_b[b][scope][lv]["qualified"]]
            levels[b][scope] = q[0] if q else None
            # per target genus (contract §4.3, §10 stop rules: a genus the labeller does not qualify for at
            # species level is bounded, whatever the pooled result)
            by_genus.setdefault(b, {})[scope] = {
                g: {lv: rl_block(rows_g, lv, domain, thresholds, confusions) for lv in ("species", "genus")}
                for g, rows_g in sorted(genus_rows(counted, domain, confusions).items())}
    agreement = {}
    for b in backends:
        brows = [r for r in sentinel_rows if r.get("backend") == b]
        agreement[b] = {}
        for s in domain.claimed_sets():
            srows = [r for r in brows if s in _row_kts(r)]
            k = sum(1 for r in srows if answer_label(r.get("answer"), r.get("answer_taxon"),
                                                     r.get("answer_level"), "species", domain)
                    == truth_label(_int(r.get("truth")), r.get("truth_kind"), r.get("truth_taxon"),
                                   "species", domain))
            agreement[b][s] = est(k, len(srows))
    pairs = {}
    for b in sorted({r.get("backend") for r in pair_rows}):
        pairs[b] = pair_block([r for r in pair_rows if r.get("backend") == b], thresholds)
    identity = {}
    for check in domain.raw.get("identity_checks") or []:
        irows = [r for r in box_rows if r["group"] == IDENTITY_GROUP
                 and _truth_class(r, domain) == check["class"]]
        by_b = {b: identity_check([r for r in irows if r.get("backend") == b], check)
                for b in sorted({r.get("backend") for r in irows})}
        pooled = identity_check(irows, check)
        pooled["pass"] = bool(by_b) and all(v["pass"] for v in by_b.values()) and pooled["pass"]
        pooled["by_backend"] = by_b
        pooled["before"] = list(check.get("before") or [])
        identity[check["class"]] = pooled
    primary = {}
    for h, sel in sorted(hyp.items()):
        primary[h] = _primary(out_b, hyp_scopes[h], NEEDED_LEVEL, sel)
        if primary[h].get("backend"):
            gs = by_genus.get(primary[h]["backend"], {}).get(hyp_scopes[h], {})
            primary[h]["genera_not_qualified_at_species"] = sorted(
                g for g, blk in gs.items() if not blk["species"]["qualified"])
    j_vlm = _j_vlm(domain, sentinel_rows, funnel_dir, prereg, confusions)
    doc_body = {"gold_sha256": gold_sha, "backends": out_b, "levels": levels, "by_genus": by_genus,
                "agreement_only": agreement,
                "pairs": pairs, "identity": identity, "primary": primary, "strata_scopes": strata_scopes,
                "hypothesis_strata": hyp, "j_vlm": j_vlm, "unsure_policy": "counted as wrong",
                "thresholds": thresholds,
                "counts": {"sentinel_rows": len(sentinel_rows), "identity_rows":
                           len([r for r in box_rows if r["group"] == IDENTITY_GROUP]),
                           "pair_rows": len(pair_rows)}}
    doc = dict(header("rl_qualification", domain, prereg, inputs, modules=(sys.modules[__name__],),
                      testing=testing), **doc_body)
    out_path = funnel_dir / RL_FILE
    existing = _existing(out_path)
    if existing is not None and _same(existing, doc):
        return existing
    if existing is not None and not force:
        raise QualifyError("%s exists and was made from other inputs; rerun with --force" % out_path)
    write_json_atomic(out_path, doc)
    log("rl: backends %s; levels %s; primary %s" % (
        backends, levels, {h: (v.get("backend"), v.get("level")) for h, v in primary.items()}))
    return doc


def _int(v):
    return None if v in (None, "") else int(v)


def _pair_row(r):
    return bool(r.get("pair_truth") not in (None, "")) or str(r.get("unit_id", "")).startswith("p:")


def _truth_class(r, domain):
    t = _int(r.get("truth"))
    return domain.class_name(t) if t is not None and domain.is_target(t) else None


def _primary(out_b, scope, level, strata):
    """DEC-1: the qualified backend with the larger Se.lb + Sp.lb at the
    needed level in the scope; a tie goes to the platform-native backend.
    Without a qualified backend at that level, the finest level some backend
    qualifies at (demoted), else none."""
    if not strata:
        return {"backend": None, "level": None, "scope": scope, "qualified": False, "demoted": False,
                "needed_level": level, "why": "no strata in the frames"}
    if not scope:
        return {"backend": None, "level": None, "scope": "", "qualified": False, "demoted": False,
                "needed_level": level, "why": "every qualification set shares material with the strata"}
    for lv in LEVELS[LEVELS.index(level):]:
        cands = []
        for b, by_scope in out_b.items():
            blk = (by_scope.get(scope) or {}).get(lv)
            if blk and blk["qualified"]:
                cands.append((-(blk["se"]["lb"] + blk["sp"]["lb"]), 0 if b == PLATFORM_BACKEND else 1, b, blk))
        if cands:
            cands.sort(key=lambda c: (round(c[0], 12), c[1], c[2]))
            _s, _t, b, blk = cands[0]
            return {"backend": b, "level": lv, "scope": scope, "qualified": lv == level,
                    "demoted": lv != level, "needed_level": level,
                    "se_lb": blk["se"]["lb"], "sp_lb": blk["sp"]["lb"]}
    return {"backend": None, "level": None, "scope": scope, "qualified": False, "demoted": False,
            "needed_level": level, "why": "no backend qualifies at any level in this scope"}


def _j_vlm(domain, sentinel_rows, funnel_dir, prereg, confusions):
    """The rl-kind judges (J-vlm) as machine judges, from their backend's
    sentinel answers, by stratum type."""
    out = {}
    panel = ((domain.raw.get("judges") or {}).get("panel") or [])
    try:
        thresholds = _thresholds(prereg)
    except QualifyError:
        return {}
    jq_path = Path(funnel_dir) / JUDGE_FILE
    j1_err = set()
    if jq_path.is_file():
        j1_err = set((read_json(jq_path).get("j1_errors") or {}).get("items") or [])
    for entry in panel:
        if entry.get("kind") != "rl":
            continue
        backend = entry.get("backend")
        rows = [r for r in sentinel_rows if r.get("backend") == backend]
        by_type = {}
        for typ in TYPES:
            pos_sets = [s for s in domain.qualify_rl_on() if entry["id"] not in
                        (domain.kt(s).get("never_qualifies") or [])]
            neg_sets = [s for s in negative_sets(typ, domain)
                        if entry["id"] not in (domain.kt(s).get("never_qualifies") or [])]
            pairs = []
            n_attr = 0
            for r in rows:
                kts = set(_row_kts(r))
                it = _item_of(r)
                a = answer_label(r.get("answer"), r.get("answer_taxon"), r.get("answer_level"),
                                 "species", domain)
                if kts & set(pos_sets) and it["truth_kind"] == "target":
                    pairs.extend(trials(it, a, "species", domain, confusions))
                elif kts & set(neg_sets) and it["truth_kind"] == "attractor":
                    n_attr += 1
                    pairs.extend(trials(it, a, "species", domain, confusions))
            se, sp = se_sp(pairs)
            pah = precision_at_half(se, sp)
            resc = [r for r in rows if r.get("unit_id") in j1_err]
            k = sum(1 for r in resc if answer_label(r.get("answer"), r.get("answer_taxon"), r.get("answer_level"),
                                                    "species", domain)
                    == truth_label(_int(r.get("truth")), r.get("truth_kind"), r.get("truth_taxon"), "species",
                                   domain))
            rescue = est(k, len(resc))
            # as for the machine judges: no qualification without attractor negatives of the type's set
            by_type[typ] = {"se": se, "sp": sp, "precision_at_half": pah, "rescue": rescue,
                            "n_attractor_negatives": n_attr,
                            "qualified": bool(se["n"] and sp["n"] and rescue["n"] and n_attr
                                              and pah["lb"] >= thresholds["precision_lb_min"]
                                              and rescue["lb"] >= thresholds["rescue_lb_min"]),
                            "sets": {"positives": pos_sets, "negatives": neg_sets}}
        out[entry["id"]] = {"backend": backend, "by_type": by_type,
                            "calibration_material": material_of(
                                [{"id": r.get("unit_id"), "kt": _row_kts(r), **{x: r.get(x) for x in KEYS}}
                                 for r in rows], domain)}
    return out


# ------------------------------------------------------------------ helpers
def _load(prereg, domain):
    if not isinstance(prereg, D.Prereg):
        prereg = D.load_prereg(prereg)
    if not isinstance(domain, D.Domain):
        domain = D.load(domain if domain is not None else prereg.domain_name)
    D.check_prereg_domain(prereg, domain)
    return prereg, domain


def _existing(path):
    try:
        return read_json(path) if Path(path).is_file() else None
    except FunnelError:
        return None


def _same(a, b):
    return canonical_json(strip_volatile(a)) == canonical_json(strip_volatile(b))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="funnel qualify")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", default=str(FUNNEL_DIR))
    ap.add_argument("--rl", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.rl:
            doc = rl(a.prereg, None, a.out, force=a.force)
        else:
            doc = judges(a.prereg, None, a.out, force=a.force)
    except FunnelError as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    print("[funnel] qualify: %s written" % doc["format"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
