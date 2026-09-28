"""INC autopilot diagnoses: pure rules over dev-only evidence
(docs/INC_AUTOPILOT.md, (a) component 2 and (b)).

    detect(evidence, thresholds=None) -> [diagnosis, ...]     (model.diagnosis dicts)

Every rule reads only fields INC already writes, takes every number and
pattern from thresholds.json (no compiled fallback: a missing key makes its
diagnosis 'unknown' and names the key), and cites each value it read
(model.cite addresses resolvable by Evidence.resolve). A diagnosis that fires
with no cite is downgraded to unknown by model.diagnosis. A rule that raises
is reported as unknown with the error, never as silence.

The result lists every diagnosis, fired or not, in a fixed order, so a caller
can show "checked and clean" apart from "not checked". Each record carries
the model.diagnosis keys plus 'detail' (machine-readable specifics levers.py
builds argv from; an additive key, not in model.py's docstring).

Gate rows. D1, D3, D3b, D8 and D12 read one normalised row per chain step:
from the ledger's latest gate entries when the ledger is in the evidence
(mid-run), else from report.json's steps (a report alone, as the prospective
case has). The truth verdicts come the same way. Each row cites its own
source, so a cite always points at what was read.

Experiment-level: D1 recipe_forgets, D15 guard_blocks_truth_helps, D2
unmeasured_source_property, D2b relevance_blind_spot, D3
attribution_control_failed, D3b source_attribution_missing, D4
decision_slot_ready / no_recipe_tracks_truth, D16 gate_v2_check, D9
warmup_dominates, D11 truth_cost_share, D12 chains_indistinguishable, D13
prediction_contradicted, DREF builder_refusal (the contract's "builder
refusal -> prerequisite" mechanism for refusals D2 does not own).
Health (every RUN tick): D5 chain_blocked, D6 stale_advance, D7 code_drift,
D8 gate_underpowered, D10 budget_projection, D14 test_touch.
RULES order is the order levers.propose ranks proposals in: D15 sits next to
D1, which it takes precedence over, and D16 next to D4, which it gates.

The funnel audit (docs/FUNNEL_AUDIT.md 8.4; runner 5.5.2): D17
scarcity_conclusion_unaudited, D18 filter_false_negatives and D19
filter_recall_unmeasured sit after D2, so D19's L10 ranks before D4's
real-loop L2 (DEC-4). Their suspicion signals S1-S7 (funnel_signals) are
functions of the funnel ledger (funnel/funnel_ledger.json), the claims
register (campaign/claims.json) and the loop reports only; the ledger is read
through its stages' "role" keys and its own class lists, never a stage id or
a class name, so the same code and thresholds run on any domain. D17 fires
when a negative conclusion exists (C: the latest finished real loop accepted
nothing and no clean step helps, or the sizing rule shrank its M, or an open
or challenged claim is negative), a signal holds (S), and no valid audit of
this Step 1 exists (A: an audit_v1.json whose ledger_fingerprint is the
ledger's fingerprint and which is not invalid). It proposes L10 first, L12
and L11 for the uninformative sources, the devil's-advocate pass (OP_DA) and
card X11 on S4 or S6; never L13. D19 needs no conclusion: S1, S4 or S6 for a
recoverable stage with no audit proposes L10. D18 reads the audit's
d18_inputs, proposes L13 with the recovery policies of the strata that pass
its two bounds, guards a stratum whose prediction is a relative of its
source taxa, sends class maps the card contradicts to L14, escalates an
invalid audit, and when silent on a valid audit proposes the claims'
transition to tested_survives (detail.claim_transitions).

prospective_d4(report) freezes D4's decision on a pilot report into a JSON
file before a person chooses the real-loop recipe (contract (f), R4b). The
record carries the rules version (rules_version(): the first 12 hex of the
sha256 of diagnose.py + thresholds.json + levers.json + levers.py,
concatenated in that order); the ticker writes one record per (pilot, rules version), named
prospective_name(exp, version), and never rewrites an earlier one.
current_prospective() is the one reader of those records: it answers only
with a READY record of the current rules version. prospective_guard() is the
one check every L2/L6 path applies (the ticker before it proposes, adopts or
runs one, the INC page's build and execute-approved routes): that READY
record, its sha256 when the ticker's is known, and a build whose replay
mode, recipes and gate flips mode are the record's.

D1 blocks D4 on the same pilot, in either replay mode, only when D4's own
choice is touched by the forgetting (R4 review 2026-09-27, made after
pilot_v3's result): when D4's threshold is not met on the pilot, or when the
chain D4 would select (rank_recipes: its argmax-agreement rule with its
tie-breaks) did not ACCEPT at least one clean truth-helps step (by D1's rows,
and by report.json, which D4 ranks from; a truth-helps step that is not
clean is not one of D1's misses and never blocks). D1 then
wins: in full mode D4 is silent (the contract's rule); in sample mode D4
reports decision_slot_ready with detail.blocked_by = D1 and D1's levers
(L1 + X1), so no real loop is built from a recipe that forgets
(levers._l2_params refuses a blocked D4 for any proposer, and prospective_d4
records ready false). D1 itself still fires, with its cites and levers (X1
stays a card), whenever its pooled conditions hold; its detail records the
misses per chain (chains), D4's selection (d4_selection) and blocks_d4.
When blocks_d4 is false, D4 applies its pre-registered rule unchanged and
records D1 under detail.d1. D1's fire decision never depends on D4's
inputs: an unreadable D4 view (a malformed report of any pilot) blocks D4
and leaves D1's own verdict and levers as they are.

D15 counts (chain, step) pairs pooled over every chain, as the v2 check
itself pools them (docs/INC_AUTOPILOT.md (b), D15). It takes precedence over
D1 only when it fires, carries a lever or card, and explains every miss:
every clean truth-helps pair in every chain that is not ACCEPTed was
REJECTed by the flips guard alone at P_data >= p_accept. D1 then still fires
with its cites but reports detail.blocked_by = D15 and proposes nothing; D4
carries D15's block instead of D1's levers. The gate protocol (L9, v1 -> v2)
is tested before the recipe (L1, X1) only when the flips guard is the whole
story; a miss another guard also failed (a regression or species guard: D1's
own forgetting signal) leaves D1 unblocked, and both proposals stand.

D16 (report.json v2_check of an experiment pinned to protocol v2) gates D4:
'refuted' blocks it (blocked_by D16, no L2 on that gate); 'supported' and
'inconclusive' are recorded in D4's detail (gate.v2_check), and D4 proceeds
only on its own ready threshold. A v2 pilot whose check is not final yet
(provisional, or missing from its report) is not ready either: D4 reports
blocked_by D16 with detail.v2_pending, and prospective_d4 writes nothing
until the check is final.

The pinned gate (pinned_gate): state.json gate_pin, else the ledger's
gate_pin entry, else report.json gate (the pinned config report.py shows),
else exp.json's gate block, else the config the gate decisions recorded (a
v1 decision omits flips_mode); an experiment built before gate blocks
existed (pilot_v1, pilot_v2) resolves to v1 from its decisions' config.

D4 is campaign-wide: it reads the latest complete pilot whatever experiment
the evidence is about, so it keeps firing after its lever ran; levers.applied
keeps a lever from being proposed twice on one parent.

D2 proposes L3 only when relevance.json is missing. A stale file (another
select build) or a malformed one (another format, made under another rule,
no calibrated tau, a check that does not follow from its tau) escalates to
a person: relevance build refuses to overwrite another build's file without
--force, which is not a lever param. A file made for this select build
under the protocol's rule whose own calibration check failed
(levers.relevance_status 'calibration_failed') fires D2 whatever the
per-source domain evidence says (no source flagged, or the aggregate or
the admit summary not in the evidence), since it decides which criterion a
real loop can use at all. It is not escalated when Step 1's evidence holds
the build: D2 names the L2 variant with --increment-sources evidence
(source-level species evidence, docs/INCREMENTAL_PROTOCOL.md Steps 2-3;
levers.increment_criterion renders it from D4's decision), records the
criterion and why in detail.increment_criterion, and puts card X9 up so a
person can revisit the zero-shot criterion. When the evidenced pool cannot
hold the default build, the loop is sized by the R4 rule of 2026-09-27
(levers.evidence_sizing): N = the fewest verified increments that decide
protocol min_decided_increments (6) with UNVERIFIED and OTHER_HEAVY (4),
M = min(the default M, floor(evidenced images / (N + 1))); D2 names that
L2 with its --size and --n-verified and records the default and sized N
and M and the rule (detail.increment_criterion.sizing). When the evidenced
pool cannot be read (no admit_summary.json, summaries the builder refuses)
or the sized M is below protocol min_increment_frac (5 %) of base B, it
escalates with the numbers and X9: choosing N and M is then a review
decision. L3 is never proposed over a calibration-failed file.

Nothing here imports a pinned INC module; the gate's vocabulary is restated
below and tests/test_inc_ap_diagnose.py checks it (and every 'mirrors'
threshold) against inc/gate.py.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path

from . import evidence as E
from . import model as M
from .evidence import walk

THRESHOLDS_FILE = Path(__file__).resolve().parent / "thresholds.json"
LEVERS_FILE = Path(__file__).resolve().parent / "levers.json"
PROSPECTIVE_FILE = M.REPLAY_DIR / "prospective_4.json"
PROSPECTIVE_FORMAT = "inc-autopilot/prospective-d4/1"
PROSPECTIVE_PREFIX = "prospective_d4_"
RULES_VERSION_HEX = 12
_PROSPECTIVE_RE = re.compile(r"\Aprospective_d4_(?P<exp>.+?)(?:__(?P<ver>[0-9a-f]{12}))?\.json\Z")
# The bytes of this module as imported: the rules that run are the ones
# loaded, even when the file on disk changed after the import (a deploy
# without a restart), so the version names the code that decided.
_SELF_BYTES = Path(__file__).resolve().read_bytes()

# inc/gate.py:53-55 and driver.py:208, restated (checked by the tests).
ACCEPT, HOLD, REJECT = "ACCEPT", "HOLD", "REJECT"
HELPS, HURTS, NEUTRAL = "helps", "hurts", "neutral"
DEFAULT_REPLAY_MODE = "sample"
PILOT_BUILDER = "inc.pilot build"            # pilot.py:681
REALLOOP_BUILDER = "inc.realloop build"      # realloop.py:146
BSWAP_STEP = "Bswap"                         # report.py BSWAP_STEP
RECIPE_ORDER = ("full", "freeze", "lora")    # pilot.inc_recipes(): D4's last, deterministic tie-break
T_FINAL_PREFIX = "T_final"                   # report.py _final_groups
LABEL_AUDIT, LOSO = "label_audit", "leave_one_source_out"   # driver.py ATTRIBUTION_NOT_RUN keys

NAMES = {"D1": "recipe_forgets", "D2": "unmeasured_source_property", "D2b": "relevance_blind_spot",
         "D3": "attribution_control_failed", "D3b": "source_attribution_missing",
         "D4": "decision_slot_ready", "D5": "chain_blocked", "D6": "stale_advance", "D7": "code_drift",
         "D8": "gate_underpowered", "D9": "warmup_dominates", "D10": "budget_projection",
         "D11": "truth_cost_share", "D12": "chains_indistinguishable", "D13": "prediction_contradicted",
         "D14": "test_touch", "D15": "guard_blocks_truth_helps", "D16": "gate_v2_check",
         "D17": "scarcity_conclusion_unaudited", "D18": "filter_false_negatives",
         "D19": "filter_recall_unmeasured", "DREF": "builder_refusal"}
D4_NOT_READY = "no_recipe_tracks_truth"
HEALTH = ("D5", "D6", "D7", "D8", "D10", "D14")
GATE_PIN = "gate_pin"                        # driver.py GATE_PIN (state.json key and ledger entry type)
P_EPS = 1e-12                                # gate.py _EPS: a P met exactly on paper is met


class _Missing(KeyError):
    """A threshold thresholds.json does not declare."""


# ------------------------------------------------------------- thresholds
_CACHE = {}


def load_thresholds(path=None):
    """thresholds.json as a dict (cached on path and mtime)."""
    p = Path(path or THRESHOLDS_FILE)
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _CACHE:
        with open(p) as fh:
            _CACHE.clear()
            _CACHE[key] = json.load(fh)
    return json.loads(json.dumps(_CACHE[key]))


def rules_files():
    """[(name, bytes)] of the files whose bytes make the rules version, in
    order: diagnose.py (as imported), thresholds.json, levers.json (as they
    are on disk now, the way load_thresholds and levers.load_menu read them),
    levers.py (as imported: D2's decision and the L2 argv rest on its
    evidence capacity and the R4 sizing rule, levers.evidence_sizing)."""
    from . import levers as LV
    return [("diagnose.py", _SELF_BYTES), ("thresholds.json", THRESHOLDS_FILE.read_bytes()),
            ("levers.json", LEVERS_FILE.read_bytes()), ("levers.py", LV._SELF_BYTES)]


def rules_version():
    """The rules version: the first 12 hex of the sha256 of diagnose.py +
    thresholds.json + levers.json + levers.py, concatenated in that order (by
    hand: cat diagnose.py thresholds.json levers.json levers.py | shasum -a 256
    | cut -c1-12)."""
    h = hashlib.sha256()
    for _name, raw in rules_files():
        h.update(raw)
    return h.hexdigest()[:RULES_VERSION_HEX]


def rules_digest():
    """{version, files: {name: sha256}}: what a prospective record cites."""
    return {"version": rules_version(),
            "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in rules_files()}}


def _merge(base, override):
    th = json.loads(json.dumps(base))
    for did, blk in (override or {}).items():
        for k, v in blk.items():
            th.setdefault(did, {})[k] = v if isinstance(v, dict) and "value" in v else {"value": v,
                                                                                         "why": "override"}
    return th


def _t(th, did, key):
    blk = th.get(did)
    if not isinstance(blk, dict) or not isinstance(blk.get(key), dict) or "value" not in blk[key]:
        raise _Missing("%s.%s" % (did, key))
    return blk[key]["value"]


def _frac(v, name):
    if not isinstance(v, dict) or not isinstance(v.get("num"), int) or not isinstance(v.get("den"), int) \
            or v["den"] <= 0:
        raise _Missing("%s (not a {num, den} fraction)" % name)
    return Fraction(v["num"], v["den"])


# ------------------------------------------------------------- helpers
def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _diag(did, fired, severity, summary, cites, levers=(), exp=None, detail=None, name=None):
    d = M.diagnosis(did, name or NAMES[did], fired, severity, summary, cites, levers=levers, exp=exp)
    d["detail"] = detail if detail is not None else {}
    return d


def _silent(did, summary, exp=None, cites=(), detail=None, name=None):
    return _diag(did, False, "info", summary, list(cites), exp=exp, detail=detail, name=name)


def _unknown(did, summary, exp=None):
    return _diag(did, False, "info", "unknown: " + summary, [], exp=exp)


def _defn(ev, exp):
    return ev.json("%s/exp.json" % exp)


def _report(ev, exp):
    return ev.json("%s/report.json" % exp)


def _ctx(ev):
    return ev.json(E.CONTEXT) or {}


def _utc(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def _replay_mode(ev, exp):
    """(mode, cite or None): exp.json replay_mode (absent = sample), else the report's."""
    d = _defn(ev, exp)
    if d is not None and "replay_mode" in d:
        return d["replay_mode"], ev.cite("%s/exp.json" % exp, "/replay_mode")
    r = _report(ev, exp)
    if r is not None and r.get("replay_mode"):
        return r["replay_mode"], ev.cite("%s/report.json" % exp, "/replay_mode")
    return DEFAULT_REPLAY_MODE, None


def _builder(ev, exp):
    d = _defn(ev, exp) or {}
    return d.get("builder")


class Row:
    """One chain step's gate decision, from the ledger or the report."""

    _LEDGER = {"p_recipe": "/decision/p_recipe", "p_data": "/decision/p_data",
               "null_mean": "/decision/null_mean", "inc": "/decision/inc",
               "cand_mean": "/decision/cand_mean", "cand_sd": "/decision/cand_sd",
               "null_sd": "/decision/null_sd", "verdict": "/decision/verdict",
               "class_vs_loc": "/decision/attribution/class_vs_loc", "clean": "/clean", "step": "/step"}
    _REPORT = {"p_recipe": "/p_recipe", "p_data": "/p_data", "null_mean": "/null_mean", "inc": "/inc",
               "cand_mean": "/cand_mean", "cand_sd": "/cand_sd", "null_sd": "/null_sd",
               "verdict": "/verdict", "class_vs_loc": "/attribution/class_vs_loc"}

    def __init__(self, ev, exp, chain, k, src, line=None, index=None, entry=None):
        self.ev, self.exp, self.chain, self.k, self.src = ev, exp, chain, k, src
        self.line, self.index = line, index
        if src == "ledger":
            self.entry = entry
            dec = self.entry.get("decision") or {}
            att = dec.get("attribution") or {}
            self.vals = {"p_recipe": dec.get("p_recipe"), "p_data": dec.get("p_data"),
                         "null_mean": dec.get("null_mean"), "inc": dec.get("inc"),
                         "cand_mean": dec.get("cand_mean"), "cand_sd": dec.get("cand_sd"),
                         "null_sd": dec.get("null_sd"), "verdict": dec.get("verdict"),
                         "class_vs_loc": att.get("class_vs_loc"), "clean": self.entry.get("clean"),
                         "step": self.entry.get("step")}
            self.not_run = sorted((self.entry.get("attribution_not_run") or {}).keys())
        else:
            rep = _report(ev, exp)
            st = rep["steps"][index]
            c = st["chains"][chain]
            self.vals = {k2: E.walk(c, p) if self._has(c, p) else None for k2, p in self._REPORT.items()}
            self.vals.update(clean=st.get("clean"), step=st.get("step"))
            self.not_run = sorted(c.get("attribution_not_run") or [])

    @staticmethod
    def _has(obj, ptr):
        try:
            E.walk(obj, ptr)
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        return self.vals.get(key)

    def cite(self, field):
        if self.src == "ledger":
            if field == "not_run":
                raise KeyError(field)
            return self.ev.lcite(self.line, self._LEDGER[field], exp=self.exp)
        base = "/steps/%d" % self.index
        if field in ("clean", "step"):
            return self.ev.cite("%s/report.json" % self.exp, "%s/%s" % (base, field))
        return self.ev.cite("%s/report.json" % self.exp,
                            "%s/chains/%s%s" % (base, E._esc(self.chain), self._REPORT[field]))

    def guards(self):
        """{guard name: passed (True / False), or None when the row does not say}:
        the ledger decision's guards.<g>.passed, else the report's guards map."""
        if self.src == "ledger":
            g = (self.entry.get("decision") or {}).get("guards")
            g = g if isinstance(g, dict) else {}
            return {k: (v.get("passed") if isinstance(v, dict) and isinstance(v.get("passed"), bool) else None)
                    for k, v in g.items()}
        rep = _report(self.ev, self.exp)
        g = ((rep["steps"][self.index].get("chains") or {}).get(self.chain) or {}).get("guards")
        g = g if isinstance(g, dict) else {}
        return {k: (v if isinstance(v, bool) else None) for k, v in g.items()}

    def cite_guard(self, name):
        if self.src == "ledger":
            return self.ev.lcite(self.line, "/decision/guards/%s/passed" % E._esc(name), exp=self.exp)
        return self.ev.cite("%s/report.json" % self.exp, "/steps/%d/chains/%s/guards/%s"
                            % (self.index, E._esc(self.chain), E._esc(name)))

    def p_accept(self, default):
        """(p_accept, cite or None): the decision's own recorded config.p_accept
        (a gate block may change it per experiment), else `default`."""
        if self.src == "ledger":
            v = _num(((self.entry.get("decision") or {}).get("config") or {}).get("p_accept"))
            if v is not None:
                return v, self.ev.lcite(self.line, "/decision/config/p_accept", exp=self.exp)
        return default, None

    def cite_not_run(self, what):
        if self.src == "ledger":
            return self.ev.lcite(self.line, "/attribution_not_run/%s" % what, exp=self.exp)
        lst = self.ev.get("%s/report.json" % self.exp, "/steps/%d/chains/%s/attribution_not_run"
                          % (self.index, E._esc(self.chain)))
        return self.ev.cite("%s/report.json" % self.exp, "/steps/%d/chains/%s/attribution_not_run/%d"
                            % (self.index, E._esc(self.chain), lst.index(what)))


def gate_rows(ev, exp):
    """[Row] of every decided chain step: the ledger's latest gate entries
    when the ledger holds any, else report.json's steps."""
    gates = sorted(ev.latest(exp, "gate").values(), key=lambda le: le[0])
    if gates:
        return [Row(ev, exp, e.get("chain"), e.get("k"), "ledger", line=ln, entry=e) for ln, e in gates]
    rep = _report(ev, exp)
    rows = []
    for i, st in enumerate((rep or {}).get("steps") or []):
        for r, c in sorted((st.get("chains") or {}).items()):
            if isinstance(c, dict) and c.get("verdict"):
                rows.append(Row(ev, exp, r, st.get("k", i + 1), "report", index=i))
    return rows


def truth_rows(ev, exp):
    """{k: (verdict, cite)} of the truth arm's decided steps (ledger, else report)."""
    out = {}
    for _, (ln, e) in sorted(ev.latest(exp, "truth").items(), key=lambda kv: kv[1][0]):
        v = (e.get("detail") or {}).get("verdict")
        if v is not None:
            out[e.get("k")] = (v, ev.lcite(ln, "/detail/verdict", exp=exp))
    if out:
        return out
    rep = _report(ev, exp)
    for i, st in enumerate((rep or {}).get("steps") or []):
        if isinstance(st.get("truth"), dict) and st["truth"].get("verdict"):
            out[st.get("k", i + 1)] = (st["truth"]["verdict"],
                                       ev.cite("%s/report.json" % exp, "/steps/%d/truth/verdict" % i))
    return out


def _step_clean(ev, exp, k, row=None):
    """(clean, cite) of step k: exp.json when present, else the row's source."""
    d = _defn(ev, exp)
    if d is not None and 1 <= k <= len(d.get("steps") or []):
        return bool(d["steps"][k - 1].get("clean")), ev.cite("%s/exp.json" % exp, "/steps/%d/clean" % (k - 1))
    if row is not None:
        return bool(row["clean"]), row.cite("clean")
    return None, None


def _finished(ev, exp):
    """(done, cite): state.json done, else report.json done."""
    st = ev.json("%s/state.json" % exp)
    if isinstance(st, dict) and st.get("done") is True:
        return True, ev.cite("%s/state.json" % exp, "/done")
    rep = _report(ev, exp)
    if isinstance(rep, dict) and rep.get("done") is True:
        return True, ev.cite("%s/report.json" % exp, "/done")
    return False, None


def pinned_gate(ev, exp):
    """(flips_mode or None, cite or None, source): the gate config exp is decided
    with, as pinned at init. In order: state.json gate_pin, the ledger's
    gate_pin entry, report.json gate (report.py shows the pinned config there),
    exp.json's gate block (init pins it as written; a block without flips_mode
    is GateConfig's default), then the config the gate decisions recorded (a v1
    decision omits flips_mode). None: no record names a mode, i.e. GateConfig's
    default (protocol v1) with nothing to cite, or no definition at all; the
    caller maps it to its v1 threshold. Never reads exp.json over a pin."""
    art = "%s/state.json" % exp
    st = ev.json(art)
    pin = st.get(GATE_PIN) if isinstance(st, dict) else None
    if isinstance(pin, dict) and isinstance((pin.get("config") or {}).get("flips_mode"), str):
        return pin["config"]["flips_mode"], ev.cite(art, "/%s/config/flips_mode" % GATE_PIN), "state.json gate_pin"
    for ln, e in sorted(ev.latest(exp, GATE_PIN).values(), key=lambda x: x[0]):
        if isinstance((e.get("config") or {}).get("flips_mode"), str):
            return e["config"]["flips_mode"], ev.lcite(ln, "/config/flips_mode", exp=exp), "ledger gate_pin"
    rep = _report(ev, exp)
    g = (rep or {}).get("gate") if isinstance(rep, dict) else None
    if isinstance(g, dict) and isinstance(g.get("flips_mode"), str):
        return g["flips_mode"], ev.cite("%s/report.json" % exp, "/gate/flips_mode"), "report.json gate (pinned)"
    d = _defn(ev, exp)
    if isinstance(d, dict) and isinstance(d.get("gate"), dict):
        if isinstance(d["gate"].get("flips_mode"), str):
            return d["gate"]["flips_mode"], ev.cite("%s/exp.json" % exp, "/gate/flips_mode"), "exp.json gate block"
        return None, ev.cite("%s/exp.json" % exp, "/gate"), "exp.json gate block without flips_mode (the default)"
    for ln, e in sorted(ev.latest(exp, "gate").values(), key=lambda x: x[0]):
        cfg = (e.get("decision") or {}).get("config")
        if isinstance(cfg, dict):
            if isinstance(cfg.get("flips_mode"), str):
                return cfg["flips_mode"], ev.lcite(ln, "/decision/config/flips_mode", exp=exp), "decision config"
            return None, ev.lcite(ln, "/decision/config", exp=exp), \
                "decision config without flips_mode (a v1 decision; no gate block)"
    if d is not None or rep is not None:
        return None, None, "no gate block, gate_pin or recorded config (GateConfig's default)"
    return None, None, "no definition or report"


def v2_check_of(ev, exp):
    """(report.json v2_check or None, cite of its outcome or None)."""
    c = (_report(ev, exp) or {}).get("v2_check")
    if isinstance(c, dict) and isinstance(c.get("outcome"), str):
        return c, ev.cite("%s/report.json" % exp, "/v2_check/outcome")
    return None, None


def _d1_levers(ev, exp):
    """The D1 lever list and detail for exp's builder and replay mode."""
    mode, _ = _replay_mode(ev, exp)
    builder = _builder(ev, exp)
    if mode == "full":
        return ["X1"], {"replay_mode": mode, "builder": builder}
    if builder == REALLOOP_BUILDER:
        return ["L2", "X1"], {"replay_mode": mode, "builder": builder, "child_replay_mode": "full"}
    return ["L1", "X1"], {"replay_mode": mode, "builder": builder, "child_replay_mode": "full"}


# ------------------------------------------------------------------ D1
def d1(ev, th, exp=None):
    exp = exp or ev.exp
    p_flag = _t(th, "D1", "p_recipe_flag")
    s_flag = _t(th, "D1", "share_recipe_flagged_min")
    s_null = _t(th, "D1", "share_null_below_inc_min")
    n_min = _t(th, "D1", "min_gate_entries")
    h_min = _t(th, "D1", "min_helps_not_accepted")
    rows = gate_rows(ev, exp)
    if not rows:
        return _silent("D1", "%s has no decided chain step in the evidence" % exp, exp)
    n = len(rows)
    flagged = [r for r in rows if _num(r["p_recipe"]) is not None and r["p_recipe"] <= p_flag]
    below = [r for r in rows if _num(r["null_mean"]) is not None and _num(r["inc"]) is not None
             and r["null_mean"] < r["inc"]]
    truths = truth_rows(ev, exp)
    helps, cites = {}, []
    for k, (v, c) in sorted(truths.items()):
        clean, cc = _step_clean(ev, exp, k, next((r for r in rows if r.k == k), None))
        if v == HELPS and clean:
            helps[k] = (c, cc)
    on_helps = [r for r in rows if r.k in helps]
    miss = [r for r in on_helps if r["verdict"] != ACCEPT]
    for r in flagged:
        cites.append(r.cite("p_recipe"))
    for r in below:
        cites += [r.cite("null_mean"), r.cite("inc")]
    for k, (c, cc) in sorted(helps.items()):
        cites += [c] + ([cc] if cc else [])
    for r in miss:
        cites.append(r.cite("verdict"))
    levers, detail = _d1_levers(ev, exp)
    mode, mc = _replay_mode(ev, exp)
    if mc:
        cites.append(mc)
    names = []
    for r in sorted(on_helps, key=lambda x: x.k):
        if str(r["step"]) not in names:
            names.append(str(r["step"]))
    chains = _d1_chains(rows, on_helps)
    detail.update(counts={"gate_entries": n, "recipe_flagged": len(flagged), "null_below_inc": len(below),
                          "helps_entries": len(on_helps), "helps_not_accepted": len(miss)},
                  helps_steps=names, source=rows[0].src, chains=chains)
    summary = ("%d/%d gate entries have P_recipe <= %g; %d/%d have null_mean < inc; %d/%d chain steps on "
               "clean steps the truth arm says helps (%s) are not ACCEPT; replay_mode %s"
               % (len(flagged), n, p_flag, len(below), n, len(miss), len(on_helps),
                  ", ".join(names) or "none", mode))
    fire = n >= n_min and len(flagged) >= s_flag * n and len(below) >= s_null * n and len(miss) >= h_min
    # D1's block on D4 (R4 review 2026-09-27, after pilot_v3): only when D4's
    # own choice is touched: its threshold is not met, or the chain it would
    # select did not ACCEPT a clean truth-helps step.
    view, vcites = _d4_view(ev, th, exp, chains, on_helps)
    blocks, why_block = _d1_blocks_d4(view, exp) if fire else (False, "D1 is silent on %s" % exp)
    detail.update(d4_selection=view, blocks_d4=blocks, blocks_d4_why=why_block)
    have = {json.dumps(c, sort_keys=True) for c in cites}
    cites += [c for c in vcites if json.dumps(c, sort_keys=True) not in have]
    if n < n_min:
        return _silent("D1", "only %d gate entries (fewer than %d): %s" % (n, n_min, summary), exp,
                       detail=detail)
    if not fire:
        return _silent("D1", summary, exp, cites=cites, detail=detail)
    # D15 takes precedence (module doc): D1 still fires, with its cites, and
    # proposes nothing while the flips guard alone explains every miss and D15
    # itself carries a lever or card (so a person or a build acts on it).
    try:
        e15 = d15_eval(ev, th, exp)
        l15 = _d15_levers(ev, th, exp, e15)[0] if e15["fired"] else []
    except _Missing as e:
        e15, l15 = None, []
        detail["precedence"] = "D15 not evaluated: threshold %s is not declared" % e.args[0]
    if e15 is not None and e15["fired"] and e15["blocks_d1"] and l15:
        pairs = ["%s %s (line %s)" % (p["chain"], p["step"], p["line"]) for p in e15["pairs"]]
        detail["blocked_by"] = {"id": "D15", "exp": exp, "pairs": pairs, "levers": list(l15),
                                "summary": e15["summary"]}
        detail["withheld"] = list(levers)
        return _diag("D1", True, "warn",
                     summary + " -> blocked by D15: every truth-helps pair that is not ACCEPT (%s) was REJECTed "
                     "by the flips guard alone at P_data >= p_accept, so the gate protocol is tested first "
                     "(D15: %s), not %s" % ("; ".join(pairs), " + ".join(l15), " + ".join(levers)),
                     cites, [], exp, detail)
    if e15 is not None and e15["fired"]:
        # D15 fired but does not block D1: name why, and cite the guards read.
        why = ("%d truth-helps pair(s) not ACCEPTed failed more than the flips guard or were not decided at "
               "P_data >= p_accept" % len(e15["unexplained"])) if e15["unexplained"] else \
            "D15 carries no lever or card on %s" % exp
        detail["precedence"] = {"D15": "fired, does not block D1", "why": why,
                                "unexplained": [{k: u[k] for k in ("chain", "step", "line", "verdict", "failed_guards",
                                                                   "p_data")} for u in e15["unexplained"]]}
        have = {json.dumps(c, sort_keys=True) for c in cites}
        cites += [c for c in e15["unexplained_cites"] if json.dumps(c, sort_keys=True) not in have]
        summary += "; D15 also fired but does not block D1 (%s: %s)" % (
            why, ", ".join("%s %s %s" % (u["chain"], u["step"], "+".join(u["failed_guards"]) or u["verdict"])
                           for u in e15["unexplained"]) or "none")
    tail = ("; " + why_block) if blocks is not None else ""
    return _diag("D1", True, "warn", summary + " -> " + " + ".join(levers) + tail, cites, levers, exp, detail)


def _d1_chains(rows, on_helps):
    """{chain: {helps: [step], misses: [step]}}: per chain, the clean
    truth-helps steps it decided and those it did not ACCEPT (D1's misses)."""
    out = {}
    for r in rows:
        out.setdefault(str(r.chain), {"helps": [], "misses": []})
    for r in sorted(on_helps, key=lambda x: (str(x.chain), x.k if isinstance(x.k, int) else -1)):
        c = out[str(r.chain)]
        c["helps"].append(str(r["step"]))
        if r["verdict"] != ACCEPT:
            c["misses"].append(str(r["step"]))
    return {k: out[k] for k in sorted(out)}


def _d4_view(ev, th, exp, chains, on_helps):
    """(view, cites): the recipe D4 would choose on exp, as D1's block on D4
    reads it -- rank_recipes' first chain (the argmax-agreement rule with its
    tie-breaks), D4's ready threshold, and that chain's misses: D1's (its
    rows, the ledger's when the ledger is in the evidence) and the report's
    (report_misses: the clean truth-helps steps report.json, which D4 ranks
    from, shows the chain did not ACCEPT). view["selected"] is None when D4
    does not evaluate exp (not a pilot that compared every step for every
    chain) or its inputs cannot be read; the reason is in view["why"]. D4's
    inputs are never allowed to decide whether D1 fires: anything that
    raises here (a malformed report of this or any other pilot) gives an
    unreadable view, which blocks D4."""
    try:
        return _d4_view_of(ev, th, exp, chains, on_helps)
    except _Missing as e:
        return {"selected": None, "evaluated": True, "why": "D4's threshold %s is not declared" % e.args[0]}, []
    except Exception as e:
        return {"selected": None, "evaluated": True,
                "why": "D4's choice on %s cannot be read (%s: %s)" % (exp, type(e).__name__, str(e)[:160])}, []


def _d4_view_of(ev, th, exp, chains, on_helps):
    if exp not in complete_pilots(ev):
        return {"selected": None, "evaluated": False,
                "why": "D4 does not evaluate %s: it is not a pilot whose report compared every step for every "
                       "chain" % exp}, []
    ready = _frac(_t(th, "D4", "ready_rate"), "D4.ready_rate")
    rows, decided = rank_recipes(ev, exp)
    if not rows:
        return {"selected": None, "evaluated": True, "why": "D4 ranks no chain on %s" % exp}, []
    best = rows[0]
    sel = best["recipe"]
    is_ready = best["agree"] * ready.denominator >= ready.numerator * best["compared"]
    c = chains.get(str(sel)) or {"helps": [], "misses": []}
    art = "%s/report.json" % exp
    cites = [ev.cite(art, "/agreement/%s/agree" % E._esc(sel)), ev.cite(art, "/agreement/%s/compared" % E._esc(sel))]
    cites += [r.cite("verdict") for r in sorted(on_helps, key=lambda x: x.k if isinstance(x.k, int) else -1)
              if str(r.chain) == str(sel)]
    report_misses = []
    for i, st in enumerate(_report(ev, exp).get("steps") or []):
        v = (st.get("chains") or {}).get(sel)
        if not (isinstance(st.get("truth"), dict) and st["truth"].get("verdict") == HELPS and isinstance(v, dict)):
            continue
        k = st.get("k", i + 1)
        clean, cc = _step_clean(ev, exp, k if isinstance(k, int) and not isinstance(k, bool) else i + 1)
        if clean is None:
            clean, cc = bool(st.get("clean")), ev.cite(art, "/steps/%d/clean" % i) if "clean" in st else None
        if clean and v.get("verdict") != ACCEPT:
            report_misses.append(str(st.get("step", k)))
            cites += [ev.cite(art, "/steps/%d/truth/verdict" % i),
                      ev.cite(art, "/steps/%d/chains/%s/verdict" % (i, E._esc(sel)))] + ([cc] if cc else [])
    view = {"selected": sel, "evaluated": True, "agree": best["agree"], "compared": best["compared"],
            "rate": "%d/%d" % (best["agree"], best["compared"]),
            "ready_rate": "%d/%d" % (ready.numerator, ready.denominator), "ready": is_ready, "decided_by": decided,
            "helps": list(c["helps"]), "misses": list(c["misses"]), "report_misses": report_misses,
            "report_helps_not_accepted": best["helps_not_accepted"]}
    return view, cites


def _d1_blocks_d4(view, exp):
    """(blocks, why) of a fired D1 on D4's choice on exp: it blocks unless
    D4 selects a chain, at its ready threshold, that ACCEPTed every clean
    truth-helps step, by D1's rows (misses) and by report.json
    (report_misses). A truth-helps step that is not clean is not one of
    D1's misses and does not block (rank_recipes still counts it in D4's
    own tie-break, report_helps_not_accepted). None when D4 does not
    evaluate exp (a real loop, a pilot still running): there is no choice
    of D4's to block."""
    sel = view.get("selected")
    if not view.get("evaluated"):
        return None, "no D4 choice to block (%s)" % view.get("why")
    if sel is None:
        return True, "blocks D4 (%s)" % view.get("why")
    if not view["ready"]:
        return True, ("blocks D4: D4's threshold is not met (its best chain %s agrees on %s < %s)"
                      % (sel, view["rate"], view["ready_rate"]))
    names = list(view["misses"]) + ["%s (report.json)" % s for s in view.get("report_misses") or []
                                    if s not in view["misses"]]
    if names:
        return True, ("blocks D4: the chain D4 selects on %s, %s (%s), did not ACCEPT clean truth-helps step(s) %s"
                      % (exp, sel, view["rate"], ", ".join(names)))
    return False, ("does not block D4: the chain D4 selects on %s, %s (%s >= %s, decided by %s), ACCEPTed every "
                   "clean truth-helps step (%s); D1's own levers stand"
                   % (exp, sel, view["rate"], view["ready_rate"], view["decided_by"] or "sole chain",
                      ", ".join(view["helps"]) or "none"))


# ------------------------------------------------------------------ D15
def d15_eval(ev, th, exp):
    """D15's reading of a finished chain experiment (thresholds D15), pooled
    over (chain, step) pairs as the v2 check pools them: the clean steps the
    truth arm says help, and in every chain the pairs REJECTed with P_data >=
    p_accept (the decision's own config.p_accept, else the threshold) whose
    only failed guard is the flips guard (every other guard the decision
    records passed). It fires with at least min_pairs such pairs; it explains
    the misses (blocks_d1) when every truth-helps pair not ACCEPTed, in every
    chain, is such a pair. Shared by d15 and D1's precedence. Raises _Missing
    for an undeclared threshold."""
    n_min = _t(th, "D15", "min_pairs")
    p_def = _t(th, "D15", "p_accept")
    fg = _t(th, "D15", "flips_guard")
    v1 = _t(th, "D15", "v1_mode")
    out = {"exp": exp, "fired": False, "blocks_d1": False, "chains": {}, "pairs": [], "unexplained": [],
           "cites": [], "unexplained_cites": [], "summary": "", "mode": None, "mode_source": None,
           "why_silent": None, "min_pairs": n_min}
    d, rep = _defn(ev, exp), _report(ev, exp)
    if (d or rep or {}).get("type") != "chain":
        out["why_silent"] = "%s is not a chain experiment in the evidence" % exp
        return out
    done, dcite = _finished(ev, exp)
    if not done:
        out["why_silent"] = "%s is not finished (state.json / report.json done)" % exp
        return out
    rows, truths = gate_rows(ev, exp), truth_rows(ev, exp)
    if not rows or not truths:
        out["why_silent"] = "%s has no decided chain step with a truth-arm decision" % exp
        return out
    mode, mcite, msrc = pinned_gate(ev, exp)
    out["mode"], out["mode_source"] = (mode if mode is not None else v1), msrc
    chains, qual, unexplained = {}, [], []
    for r in sorted(rows, key=lambda x: (str(x.chain), x.k if isinstance(x.k, int) else -1)):
        tv = truths.get(r.k)
        if tv is None or tv[0] != HELPS:
            continue
        clean, ccite = _step_clean(ev, exp, r.k, r)
        if not clean:
            continue                      # a non-clean acceptance refutes v2: never evidence for it
        ch = chains.setdefault(r.chain, {"helps": [], "misses": [], "flips_only": []})
        ch["helps"].append(str(r["step"]))
        if r["verdict"] == ACCEPT:
            continue
        ch["misses"].append(str(r["step"]))
        p = _num(r["p_data"])
        pa, pcite = r.p_accept(p_def)
        g = r.guards()
        others = [k for k in g if k != fg]
        failed = sorted(k for k, v in g.items() if v is False)
        if (r["verdict"] == REJECT and p is not None and p >= pa - P_EPS and g.get(fg) is False and others
                and all(g[k] is True for k in others)):
            ch["flips_only"].append(str(r["step"]))
            qual.append((r, tv[1], ccite, pcite, sorted(g)))
            continue
        unrecorded = sorted(k for k in others if g[k] is None) + ([] if others else ["every guard but %s" % fg])
        if r["verdict"] != REJECT:
            why = "verdict %s" % r["verdict"]
        elif p is None or p < pa - P_EPS:
            why = "P_data %s below p_accept %s" % (p, pa)
        elif g.get(fg) is not False or len(failed) > 1:
            why = "failed guards %s" % (", ".join(failed) or "none recorded")
        else:
            why = "guard(s) not recorded: %s" % ", ".join(unrecorded)
        unexplained.append((r, {"chain": r.chain, "k": r.k, "step": str(r["step"]), "line": r.line,
                                "source": r.src, "verdict": r["verdict"], "p_data": p, "failed_guards": failed,
                                "why": why}, sorted(g)))
    fired = len(qual) >= n_min
    cites, pairs = [], []
    for r, tcite, ccite, pcite, gnames in qual:
        pairs.append({"chain": r.chain, "k": r.k, "step": str(r["step"]), "line": r.line, "source": r.src,
                      "p_data": _num(r["p_data"]), "failed_guards": [fg]})
        if fired:
            cites += [r.cite("verdict"), r.cite("p_data")] + [r.cite_guard(k) for k in gnames] + [tcite]
            cites += [x for x in (ccite, pcite) if x is not None]
    if fired:
        cites += [x for x in (mcite, dcite) if x is not None]
    ucites = []
    for r, _u, gnames in unexplained:
        ucites += [r.cite("verdict"), r.cite("p_data")] + [r.cite_guard(k) for k in gnames]
    out["chains"] = {c: dict(v) for c, v in sorted(chains.items())}
    per = "; ".join("%s %s" % (c, ", ".join(v["flips_only"]) or "none") for c, v in out["chains"].items())
    unx = ", ".join("%s %s (%s)" % (u["chain"], u["step"], u["why"]) for _r, u, _g in unexplained)
    out.update(fired=fired, blocks_d1=fired and not unexplained, pairs=pairs,
               unexplained=[u for _r, u, _g in unexplained], cites=cites, unexplained_cites=ucites,
               summary=("%d (chain, step) pair(s) of clean truth-helps steps REJECTed at P_data >= p_accept by the "
                        "flips guard alone (%s; fires at >= %d); %s; gate pinned to flips_mode %s (%s)"
                        % (len(pairs), per or "none", n_min,
                           ("every truth-helps pair not ACCEPTed is one of them" if not unexplained
                            else "not explained by it: " + unx), out["mode"], msrc)))
    return out


def _d15_levers(ev, th, exp, e):
    """(levers, detail extras, lever text) of a fired D15 reading e on exp:
    L9 on a v1 pilot, card X8 on a v1 experiment that is not a pilot, card X6
    on a v2 one, nothing on a mode that is neither."""
    v1 = _t(th, "D15", "v1_mode")
    v2 = _t(th, "D15", "v2_mode")
    if e["mode"] == v1:
        if _builder(ev, exp) == PILOT_BUILDER:
            return ["L9"], {"then": ["L2 waits for the v2 pilot's report (D16 on its v2_check, D4 on its "
                                     "agreement)"]}, \
                "L9 (rebuild the pilot with --gate-flips-mode net, the pre-registered v2 test)"
        needs = ("a person: %s is not a pilot, and L9 rebuilds a pilot; a real loop is built on v2 only after a v2 "
                 "pilot's check (docs/INCREMENTAL_PROTOCOL.md, Protocol v2)" % exp)
        return ["X8"], {"needs": needs}, "X8 (human card: " + needs + ")"
    if e["mode"] == v2:
        needs = ("the pre-registered v2 guard (net flips) still blocks truth-helps steps on its own: the guard "
                 "family needs rethinking (R4), not another rebuild")
        return ["X6"], {"needs": needs}, "X6 (human card: " + needs + ")"
    needs = "a person: flips_mode %r is neither v1 nor v2" % e["mode"]
    return [], {"needs": needs}, "no lever (" + needs + ")"


def d15(ev, th):
    exp = ev.exp
    e = d15_eval(ev, th, exp)
    if e["why_silent"]:
        return _silent("D15", e["why_silent"], exp)
    detail = {"flips_mode": e["mode"], "flips_mode_source": e["mode_source"], "min_pairs": e["min_pairs"],
              "chains": e["chains"], "pairs": e["pairs"], "unexplained": e["unexplained"],
              "blocks_d1": e["blocks_d1"]}
    if not e["fired"]:
        return _silent("D15", e["summary"], exp, detail=detail)
    mode, mc = _replay_mode(ev, exp)
    detail["replay_mode"] = mode
    cites = list(e["cites"]) + ([mc] if mc and mc not in e["cites"] else [])
    levers, extra, text = _d15_levers(ev, th, exp, e)
    detail.update(extra)
    detail["blocks_d1"] = e["blocks_d1"] and bool(levers)
    tail = ("; D1 is blocked on it (every truth-helps miss is the flips guard alone)" if detail["blocks_d1"]
            else "; D1 is not blocked by it")
    return _diag("D15", True, "warn", e["summary"] + tail + " -> " + text, cites, levers, exp, detail)


# ------------------------------------------------------------------ D16
def _v2_state(ev, th, exp):
    """(mode or None, v2_check or None, outcome cite or None, final or None)
    of exp: v2_check only when exp is pinned to thresholds D16 v2_mode."""
    v2 = _t(th, "D16", "v2_mode")
    mode, _mc, _src = pinned_gate(ev, exp)
    if mode != v2:
        return mode, None, None, None
    c, cc = v2_check_of(ev, exp)
    return mode, c, cc, (c or {}).get("final")


def d16(ev, th):
    exp = ev.exp
    v2 = _t(th, "D16", "v2_mode")
    sev = _t(th, "D16", "severity")
    d, rep = _defn(ev, exp), _report(ev, exp)
    if (d or rep or {}).get("type") != "chain":
        return _silent("D16", "%s is not a chain experiment in the evidence" % exp, exp)
    mode, mcite, msrc = pinned_gate(ev, exp)
    if mode != v2:
        return _silent("D16", "%s is not pinned to flips_mode %s (%s: %s): no v2 check to read"
                       % (exp, v2, msrc, mode or "the default, v1"), exp)
    done, dcite = _finished(ev, exp)
    if not done:
        return _silent("D16", "%s is not finished: its v2_check is provisional" % exp, exp)
    c, ccite = v2_check_of(ev, exp)
    if c is None:
        return _silent("D16", "%s is pinned to v2 and done, but report.json has no v2_check (a report older "
                              "than the check): not evaluated" % exp, exp)
    if c.get("final") is not True:
        return _silent("D16", "%s report.json v2_check is provisional (final %r)" % (exp, c.get("final")), exp)
    outcome = c["outcome"]
    if outcome not in sev:
        return _unknown("D16", "v2_check outcome %r is not one of %s" % (outcome, sorted(sev)), exp)
    art = "%s/report.json" % exp
    cites = [ccite, ev.cite(art, "/v2_check/final")]
    for k in ("agree_v2", "agree_v1_counterfactual", "compared"):
        if k in c:
            cites.append(ev.cite(art, "/v2_check/%s" % k))
    if c.get("non_clean_accepted"):
        cites.append(ev.cite(art, "/v2_check/non_clean_accepted"))
    cites += [x for x in (mcite, dcite) if x is not None]
    detail = {"outcome": outcome, "flips_mode": mode, "flips_mode_source": msrc,
              "agree_v2": c.get("agree_v2"), "agree_v1_counterfactual": c.get("agree_v1_counterfactual"),
              "compared": c.get("compared"), "non_clean_accepted": c.get("non_clean_accepted") or []}
    head = ("protocol v2 check on %s: %s (v2 agrees with the truth arm on %s of %s (chain, step) pairs, its v1 "
            "counterfactual on %s%s)" % (exp, outcome, c.get("agree_v2"), c.get("compared"),
                                         c.get("agree_v1_counterfactual"),
                                         "; v2 ACCEPTed a step not marked clean: %s"
                                         % ", ".join("%s %s" % (x.get("chain"), x.get("step"))
                                                     for x in c["non_clean_accepted"])
                                         if c.get("non_clean_accepted") else ""))
    # 'then' lists levers (levers.propose reads each entry's first word as one);
    # 'consequence' is what the outcome means for the gate.
    if outcome == "refuted":
        detail["then"] = ["L2 is withheld on this gate: D4 is blocked_by D16"]
        detail["consequence"] = ("v2 is retired for this gate; v1 stays the default (docs/INCREMENTAL_PROTOCOL.md, "
                                 "Protocol v2, What follows from it)")
        return _diag("D16", True, sev[outcome], head + " -> X7 (human card: revert to v1 or rethink the guard; "
                     "no further L2 on this gate)", cites, ["X7"], exp, detail)
    if outcome == "supported":
        detail["consequence"] = "recorded; D4 may proceed on its own threshold"
        return _diag("D16", True, sev[outcome], head + " -> recorded; D4 may proceed", cites, [], exp, detail)
    detail["consequence"] = "D4 may proceed only on its own threshold; v2 is unproven on %s" % exp
    return _diag("D16", True, sev[outcome], head + " -> v2 unproven; D4 proceeds only on its own threshold",
                 cites, [], exp, detail)


# ------------------------------------------------------------------ D2
def _prereq(message):
    from . import levers as LV
    return LV.match_refusal(message)


def d2(ev, th):
    statuses = _t(th, "D2", "unevidenced_statuses")
    s_min = _t(th, "D2", "unevidenced_share_min")
    c_max = _t(th, "D2", "species_crops_max")
    v_max = _t(th, "D2", "verified_box_share_max")
    ctx = _ctx(ev)
    cites, notes = [], []
    refused = []
    for i, r in enumerate(ctx.get("refusals") or []):
        m = _prereq(str((r or {}).get("message") or ""))
        if m is not None and m.get("prerequisite") == "L3":
            refused.append((i, r))
            cites.append(ev.cite(E.CONTEXT, "/refusals/%d/message" % i))
    sel = ev.json("step1/select_summary.json")
    flagged, rel_state = [], None
    if sel is None:
        notes.append("no Step 1 select build in the evidence")
    else:
        agg = ev.json("step1/" + E.CLUSTERS_BY_SOURCE)
        admit = ev.json("step1/admit_summary.json")
        from . import levers as LV
        rel_state = LV.relevance_state(ev)
        if rel_state == "stale":
            for table, fname in LV.RELEVANCE_TABLES:
                for art, ptr in (("step1/relevance.json", E.pointer("inputs", table, "sha256")),
                                 ("step1/select_summary.json", E.pointer("outputs", fname, "sha256"))):
                    try:
                        cites.append(ev.cite(art, ptr))
                    except KeyError:
                        pass
        if agg is None or admit is None:
            notes.append("the per-source %s is not in the evidence"
                         % ("status aggregate of select_clusters.csv" if agg is None else "admit_summary.json"))
        else:
            src_ev = (sel.get("retrieval") or {}).get("source_evidence") or {}
            for src in sorted((sel.get("sources") or {}).get("increment_pool") or {}):
                tags = (agg.get("sources") or {}).get(src) or {}
                counts = {}
                for tag, blk in sorted(tags.items()):
                    for s, v in ((blk or {}).get("status") or {}).items():
                        if isinstance(v, int) and not isinstance(v, bool):
                            counts[s] = counts.get(s, 0) + v
                tot = sum(counts.values())
                if not tot:
                    continue
                unev = sum(counts.get(s, 0) for s in statuses)
                crops = (src_ev.get(src) or {}).get("species_crops", 0)
                boxes = ((admit.get("per_slug") or {}).get(src) or {}).get("boxes") or {}
                btot = sum(v for v in boxes.values() if isinstance(v, int) and not isinstance(v, bool))
                ver = boxes.get("verified", 0)
                if unev >= s_min * tot and crops <= c_max and (ver / btot if btot else 0.0) <= v_max:
                    flagged.append({"source": src, "images": tot, "unevidenced": unev,
                                    "species_crops": crops, "verified_boxes": ver, "boxes": btot})
                    cites.append(ev.cite("step1/select_summary.json", E.pointer("sources", "increment_pool", src)))
                    for tag, blk in sorted(tags.items()):
                        for s in statuses:
                            if s in ((blk or {}).get("status") or {}):
                                cites.append(ev.cite("step1/" + E.CLUSTERS_BY_SOURCE,
                                                     E.pointer("sources", src, tag, "status", s)))
                    if src in src_ev:
                        cites.append(ev.cite("step1/select_summary.json",
                                             E.pointer("retrieval", "source_evidence", src, "species_crops")))
                    bp = ("per_slug", src, "boxes", "verified") if "verified" in boxes else ("per_slug", src, "boxes")
                    if src in (admit.get("per_slug") or {}):
                        cites.append(ev.cite("step1/admit_summary.json", E.pointer(*bp)))
    fire = bool(flagged and rel_state != "matching") or bool(refused)
    detail = {"sources": flagged, "relevance": rel_state, "refusals": [r for _, r in refused],
              "then": ["L2 with --relevance"], "notes": notes}
    if rel_state is not None:
        detail["increment_criterion"] = _d2_criterion(ev, rel_state, flagged)
    names = ", ".join("%s (%d images, %d/%d without domain evidence)"
                      % (f["source"], f["images"], f["unevidenced"], f["images"]) for f in flagged)
    summary = ("%d increment-pool source(s) have no domain evidence: %s; relevance.json %s%s"
               % (len(flagged), names or "none", rel_state or "not evaluated",
                  ("; %d builder refusal(s) name the relevance prerequisite" % len(refused)) if refused else ""))
    from . import levers as LV
    if rel_state == LV.CALIBRATION_FAILED:
        # Fires whatever the per-source domain evidence says (no source flagged,
        # or the aggregate or admit summary not in the evidence): the file
        # decides which criterion a real loop can use at all, so D2 always
        # records it, puts X9 up, and escalates when the evidence criterion
        # cannot be read or cannot hold the build.
        if notes and not flagged:
            summary = ("the per-source domain evidence is not evaluated (%s); relevance.json %s%s"
                       % ("; ".join(notes), rel_state,
                          ("; %d builder refusal(s) name the relevance prerequisite" % len(refused))
                          if refused else ""))
        return _d2_calibration_failed(ev, summary, cites, detail)
    if notes and not flagged and not refused:
        return _silent("D2", "not evaluated: " + "; ".join(notes), cites=cites, detail=detail)
    if not fire:
        return _silent("D2", summary, cites=cites, detail=detail)
    if rel_state in ("stale", "refused"):
        # L3 cannot fix these: relevance build refuses over another build's file
        # without --force (deliberately not a lever param), and a malformed file
        # (a check that does not follow from its tau, another format) is not the
        # rule's verdict on anything. A person decides. A file made for this
        # select build whose own check failed is the calibration_failed branch
        # above: the evidence criterion, not an escalation, when it fits.
        for ptr in ("/format", "/params/tau_min", "/calibration/check/ok"):
            try:
                cites.append(ev.cite("step1/relevance.json", ptr))
            except KeyError:
                pass
        detail["then"] = []
        detail["needs"] = ("a person: step1/relevance.json is %s (%s); rebuilding it over another build's file "
                           "(relevance build --force) or reading a malformed file is not on the menu"
                           % (rel_state, "made for another select build" if rel_state == "stale"
                              else "not an inc.relevance/1 file made under the protocol's rule with a calibration "
                                   "check that follows from its tau: %s" % LV.relevance_status(ev)["why"]))
        return _diag("D2", True, "crit", summary + " -> escalate (" + detail["needs"] + ")", cites,
                     ["OP_ESCALATE"], None, detail)
    return _diag("D2", True, "warn", summary + " -> L3, then L2 with --relevance", cites, ["L3"], None, detail)


def _d2_criterion(ev, rel_state, flagged):
    """D2's record of the relevance criterion a real-loop build from this
    Step 1 would use (levers.increment_criterion), and why: 'relevance'
    (a relevance.json made for this select build that passed its
    calibration check), 'evidence' (that file failed its own calibration
    check and the evidenced pool holds the build's increments, at the
    default N and M or at N and M sized by the R4 rule), or None with the
    reason no build can be drawn now. On 'evidence' it also names the
    sources D2 flagged that the criterion excludes. With a calibration-
    failed file, 'sizing' records the default and the sized N and M and the
    rule (levers.evidence_sizing), and 'capacity' is the capacity at the N
    and M the build uses (the default one when the rule refused to size)."""
    from . import levers as LV
    st = LV.relevance_status(ev)
    rec = {"relevance": {k: st.get(k) for k in ("state", "why", "tau", "tau_min")}}
    if rel_state == "matching":
        return dict(rec, criterion=LV.SOURCES_RELEVANCE, flag="--relevance",
                    why="step1/relevance.json is made for this select build and passed its calibration check")
    if rel_state != LV.CALIBRATION_FAILED:
        return dict(rec, criterion=None, why=LV.RELEVANCE_WHY.get(rel_state, rel_state))
    try:
        cap, _c = LV.evidence_sizing(ev)
    except LV.Defer as e:
        return dict(rec, criterion=None, why="%s; and source-level species evidence cannot be evaluated: %s"
                    % (LV.EVIDENCE_WHY, e))
    rec["capacity"] = {k: v for k, v in cap.items() if k != "sizing"}
    rec["sizing"] = cap["sizing"]
    if not cap["fits"]:
        return dict(rec, criterion=None, why="%s; %s" % (LV.EVIDENCE_WHY, LV.capacity_why(cap)))
    return dict(rec, criterion=LV.SOURCES_EVIDENCE, flag="--increment-sources evidence", why=LV.EVIDENCE_WHY,
                flags=["--increment-sources", LV.SOURCES_EVIDENCE] + list(cap["sizing"]["flags"]),
                flagged_excluded=[f["source"] for f in flagged if f["source"] not in cap["evidenced_sources"]],
                flagged_evidenced=[f["source"] for f in flagged if f["source"] in cap["evidenced_sources"]])


def _d2_calibration_failed(ev, summary, cites, detail):
    """D2 on a relevance.json made for this select build whose own
    calibration check failed. relevance.load refuses the file, so the
    zero-shot criterion cannot build a production loop; the protocol's
    second criterion, source-level species evidence, can. When the evidenced
    pool holds the default build's increments, D2 names the L2 variant with
    --increment-sources evidence (levers.increment_criterion renders it from
    D4's decision), and card X9 lets a person revisit the zero-shot
    criterion later. When it holds only a smaller build, the loop is sized
    by the R4 rule of 2026-09-27 (levers.evidence_sizing: N verified
    increments for protocol min_decided_increments decided ones, M =
    min(the default M, floor(evidenced images / (N + 1)))), and D2 names
    the L2 variant with the sized --size and --n-verified too; its detail
    records the default and the sized N and M and the rule. When even the
    sized M falls below protocol min_increment_frac of base B, or the
    evidence cannot be read, D2 escalates with the numbers: choosing N and
    M is then a review decision."""
    from . import levers as LV
    crit = detail.get("increment_criterion") or {}
    extra = LV.relevance_cites(ev)
    cap = crit.get("capacity")
    if cap is not None:
        try:
            extra += LV.evidence_sizing(ev)[1]
        except LV.Defer:
            pass
    cites = list(cites)
    have = {json.dumps(c, sort_keys=True) for c in cites}
    for c in extra:
        k = json.dumps(c, sort_keys=True)
        if k not in have:
            have.add(k)
            cites.append(c)
    rel = crit.get("relevance") or {}
    failed = "step1/relevance.json failed its calibration check (tau %s < %s)" % (rel.get("tau"), rel.get("tau_min"))
    sz = crit.get("sizing") or {}
    if crit.get("criterion") == LV.SOURCES_EVIDENCE and sz.get("applied"):
        dflt, sized = sz["default"], sz["sized"]
        detail["then"] = ["L2 with " + " ".join(crit["flags"])]
        return _diag("D2", True, "warn",
                     "%s; %s -> %s (source-level species evidence: %d source(s) with verified cwd12-species boxes "
                     "hold %d increment-pool images, less than the default build's %d = (%d + 1) x %d, so the loop is "
                     "sized by the R4 rule: N = %d for %d decided increments, M = min(%d, floor(%d / %d)) = %d, %.1f%% "
                     "of base B (the floor is %g x |B| = %s), needing %d = (%d + 1) x %d; the OtherPlant-heavy share "
                     "is checked by realloop build), X9 (a person may revisit the zero-shot criterion)"
                     % (summary, failed, detail["then"][0],
                        len([s for s, e in cap["evidenced_sources"].items() if e["pool_images"]]),
                        cap["evidenced_pool_images"], dflt["needed_images"], dflt["n_verified"],
                        dflt["increment_images"], sized["n_verified"], sized["decided_increments"],
                        dflt["increment_images"], cap["evidenced_pool_images"], sized["n_verified"] + 1,
                        sized["increment_images"], 100.0 * sized["increment_frac"], sz["min_increment_frac"],
                        sz["min_increment_images"], sized["needed_images"], sized["n_verified"],
                        sized["increment_images"]),
                     cites, ["X9"], None, detail)
    if crit.get("criterion") == LV.SOURCES_EVIDENCE:
        detail["then"] = ["L2 with --increment-sources evidence"]
        return _diag("D2", True, "warn",
                     "%s; %s -> L2 with --increment-sources evidence (source-level species evidence: %d source(s) "
                     "with verified cwd12-species boxes hold %d increment-pool images, the default build needs %d = "
                     "(%d + 1) x %d; the OtherPlant-heavy share is checked by realloop build), X9 (a person may "
                     "revisit the zero-shot criterion)"
                     % (summary, failed, len([s for s, e in cap["evidenced_sources"].items() if e["pool_images"]]),
                        cap["evidenced_pool_images"], cap["needed_images"], cap["n_verified"],
                        cap["increment_images"]),
                     cites, ["X9"], None, detail)
    detail["then"] = []
    detail["needs"] = "a person: %s; %s" % (failed, crit.get("why"))
    return _diag("D2", True, "crit", summary + " -> escalate (" + detail["needs"] + ")", cites,
                 ["OP_ESCALATE", "X9"], None, detail)


def d2b(ev, th):
    theta = _t(th, "D2b", "leaf_disease_share_min")
    rel = ev.json("step1/relevance.json")
    if rel is None:
        return _silent("D2b", "no relevance.json in the evidence")
    from . import levers as LV
    if LV.relevance_state(ev) in ("refused", LV.CALIBRATION_FAILED):
        return _silent("D2b", "not evaluated: relevance.json is not an inc.relevance/1 file with a passed "
                              "calibration check, so its statuses are not trusted")
    flagged, cites = [], []
    for src, e in sorted(((rel.get("increment_pool") or {}).get("sources") or {}).items()):
        share = _num(((e or {}).get("top_set_share") or {}).get("leaf_disease"))
        if (e or {}).get("status") == "pass" and share is not None and share >= theta:
            flagged.append({"source": src, "leaf_disease_share": share, "images": e.get("images")})
            cites.append(ev.cite("step1/relevance.json", E.pointer("increment_pool", "sources", src, "status")))
            cites.append(ev.cite("step1/relevance.json",
                                 E.pointer("increment_pool", "sources", src, "top_set_share", "leaf_disease")))
    summary = ("%d passing source(s) whose sampled crops' top prompt is leaf disease for >= %g of them: %s"
               % (len(flagged), theta, ", ".join("%s (%.2f)" % (f["source"], f["leaf_disease_share"])
                                                 for f in flagged) or "none"))
    if not flagged:
        return _silent("D2b", summary)
    return _diag("D2b", True, "warn", summary + " -> X2 (human card)", cites, ["X2"], None, {"sources": flagged})


# ------------------------------------------------------------------ D3
def d3(ev, th):
    exp = ev.exp
    pat = _t(th, "D3", "planted_relabel_pattern")
    labels = _t(th, "D3", "labels_attribution")
    d = _defn(ev, exp)
    if d is None or d.get("type") != "chain":
        return _silent("D3", "%s is not a chain experiment with exp.json in the evidence" % exp, exp)
    planted = [(i, s) for i, s in enumerate(d.get("steps") or [])
               if not s.get("clean") and re.search(pat, str(s.get("planted") or ""), re.I)]
    if not planted:
        return _silent("D3", "%s has no planted relabelling step" % exp, exp)
    rows = gate_rows(ev, exp)
    rep = _report(ev, exp)
    cites, fails, not_run = [], [], False
    for i, s in planted:
        cites.append(ev.cite("%s/exp.json" % exp, "/steps/%d/planted" % i))
        for r in [r for r in rows if r.k == i + 1]:
            if r["class_vs_loc"] != labels:
                fails.append({"step": s.get("name"), "chain": r.chain, "class_vs_loc": r["class_vs_loc"]})
                cites.append(r.cite("class_vs_loc"))
                if LABEL_AUDIT in r.not_run:
                    not_run = True
                    cites.append(r.cite_not_run(LABEL_AUDIT))
                bs = ((rep or {}).get("chains") or {}).get(r.chain, {}) if rep else {}
                if s.get("name") == _bswap_step(rep) and isinstance(bs.get("bswap"), dict):
                    cites.append(ev.cite("%s/report.json" % exp,
                                         "/chains/%s/bswap/attributed_to_labels" % E._esc(r.chain)))
    if rep is not None and ((rep.get("attribution_scope") or {}).get("not_run") or {}).get("4") is not None:
        cites.append(ev.cite("%s/report.json" % exp, "/attribution_scope/not_run/4"))
    names = sorted({f["step"] for f in fails})
    detail = {"failed": fails, "planted_steps": [s.get("name") for _, s in planted]}
    if not fails:
        return _silent("D3", "every chain attributed the planted relabelling step(s) %s to labels"
                       % [s.get("name") for _, s in planted], exp, detail=detail)
    audit_art = "%s/audit/label_audit.json" % exp
    audit = ev.json(audit_art)
    if audit is None:                   # the same audit written under INC_DIR/audit/ (pilot_v1's)
        audit_art = E.ROOT_AUDIT % exp
        audit = ev.json(audit_art)
    if audit is None:
        levers = ["L4"] if not_run else []
        detail["then"] = ["X3 if the audit separates the planted step where class_vs_loc did not"]
        tail = "label audit not run -> L4" if not_run else "no label audit listed as not run"
    else:
        aud = audit.get("audits") or {}
        clean = [s.get("name") for s in d.get("steps") or [] if s.get("clean")]
        above = {n: bool(((aud.get(n) or {}).get("species") or {}).get("above_baseline")) for n in aud}
        for n in sorted(above):
            cites.append(ev.cite(audit_art, E.pointer("audits", n, "species", "above_baseline")))
        sep = all(above.get(n) for n in names) and not any(above.get(c) for c in clean)
        detail["audit_separates"] = sep
        levers = ["X3"] if sep else []
        tail = ("the audit separates the planted step -> X3 (human card)" if sep
                else "the audit does not separate it either (a person decides)")
    summary = ("planted relabelling step %s: %d chain(s) attributed it to %s, not %s; %s"
               % (", ".join(names), len(fails), sorted({str(f["class_vs_loc"]) for f in fails}), labels, tail))
    return _diag("D3", True, "warn", summary, cites, levers, exp, detail)


def d3b(ev, th):
    exp = ev.exp
    rows = [r for r in gate_rows(ev, exp) if r["verdict"] == REJECT and LOSO in r.not_run]
    if not rows:
        return _silent("D3b", "no REJECT of a multi-source increment lacks leave-one-source-out", exp)
    cites = []
    for r in rows:
        cites += [r.cite("verdict"), r.cite_not_run(LOSO)]
    rep = _report(ev, exp)
    for c in sorted(((rep or {}).get("chains") or {})):
        if (rep["chains"][c] or {}).get("rejects_without_source_attribution"):
            cites.append(ev.cite("%s/report.json" % exp,
                                 "/chains/%s/rejects_without_source_attribution" % E._esc(c)))
    if rep is not None and ((rep.get("attribution_scope") or {}).get("not_run") or {}).get("5") is not None:
        cites.append(ev.cite("%s/report.json" % exp, "/attribution_scope/not_run/5"))
    steps = sorted({"s%02d_%s" % (r.k, r["step"]) for r in rows})
    summary = ("%d REJECT(s) of multi-source increment(s) %s without leave-one-source-out attribution "
               "(chains %s) -> X4 (human card)" % (len(rows), ", ".join(steps), sorted({r.chain for r in rows})))
    return _diag("D3b", True, "info", summary, cites, ["X4"], exp, {"steps": steps})


def _bswap_step(rep):
    """The report's Bswap step: label_steps.bswap, or (a report written
    before label_steps existed, as pilot_v1's) a step named Bswap."""
    if not rep:
        return None
    ls = (rep.get("label_steps") or {}).get("bswap")
    if ls:
        return ls
    return BSWAP_STEP if any(st.get("step") == BSWAP_STEP for st in rep.get("steps") or []) else None


# ------------------------------------------------------------------ D4
def pilots(ev):
    """Pilot experiments in the evidence, oldest first: chain experiments
    built by inc.pilot build (exp.json), or, without exp.json, a chain report
    that has a Bswap step."""
    out = []
    for e in ev.exps():
        d, r = _defn(ev, e), _report(ev, e)
        if d is not None:
            if d.get("type") == "chain" and d.get("builder") == PILOT_BUILDER:
                out.append(e)
        elif r is not None and r.get("type") == "chain" and _bswap_step(r) == BSWAP_STEP:
            out.append(e)
    return out


def complete_pilots(ev):
    """Pilots whose report compared every step for every chain."""
    out = []
    for e in pilots(ev):
        r = _report(ev, e)
        steps = (r or {}).get("steps") or []
        ag = (r or {}).get("agreement") or {}
        if steps and ag and all(isinstance(a, dict) and a.get("compared") == len(steps) for a in ag.values()):
            out.append(e)
    return out


def _final_dev(rep, label):
    for j, row in enumerate(rep.get("final") or []):
        if row.get("model") == label or (label == T_FINAL_PREFIX and str(row.get("model", "")).startswith(label)):
            v = _num((((row.get("exams") or {}).get("dev") or {}).get("twelve") or {}).get("mean"))
            if v is not None:
                return v, j
    return None, None


def rank_recipes(ev, exp):
    """D4's ranking of the pilot's chains, best first, with the inputs and
    cites of every criterion: rate; fewer non-ACCEPT on truth-helps steps;
    smaller |final dev twelve(chain) - dev twelve(T_final)|; lower GPU-hours;
    then the recipe table's order (deterministic, not in the contract)."""
    rep = _report(ev, exp)
    art = "%s/report.json" % exp
    t_val, t_j = _final_dev(rep, T_FINAL_PREFIX)
    rows = []
    for r in sorted(rep["agreement"], key=lambda x: (RECIPE_ORDER.index(x) if x in RECIPE_ORDER else 99, x)):
        a = rep["agreement"][r]
        cites = [ev.cite(art, "/agreement/%s/agree" % E._esc(r)), ev.cite(art, "/agreement/%s/compared" % E._esc(r))]
        rate = Fraction(int(a["agree"]), int(a["compared"]))
        nonacc = 0
        for i, st in enumerate(rep.get("steps") or []):
            c = (st.get("chains") or {}).get(r)
            if isinstance(st.get("truth"), dict) and st["truth"].get("verdict") == HELPS and isinstance(c, dict):
                cites += [ev.cite(art, "/steps/%d/truth/verdict" % i),
                          ev.cite(art, "/steps/%d/chains/%s/verdict" % (i, E._esc(r)))]
                nonacc += c.get("verdict") != ACCEPT
        c_val, c_j = _final_dev(rep, "chain %s: final incumbent" % r)
        diff = abs(c_val - t_val) if c_val is not None and t_val is not None else None
        if diff is not None:
            cites += [ev.cite(art, "/final/%d/exams/dev/twelve/mean" % c_j),
                      ev.cite(art, "/final/%d/exams/dev/twelve/mean" % t_j)]
        gh = _num(((rep.get("gpu_hours") or {}).get("chain:%s" % r) or {}).get("hours"))
        if gh is not None:
            cites.append(ev.cite(art, "/gpu_hours/%s/hours" % E._esc("chain:%s" % r)))
        key = (-rate, nonacc, diff if diff is not None else math.inf, gh if gh is not None else math.inf,
               RECIPE_ORDER.index(r) if r in RECIPE_ORDER else 99, r)
        rows.append({"recipe": r, "agree": int(a["agree"]), "compared": int(a["compared"]),
                     "rate": float(rate), "helps_not_accepted": nonacc, "final_dev_gap": diff,
                     "gpu_hours": gh, "_key": key, "_cites": cites})
    rows.sort(key=lambda x: x["_key"])
    crit = ("rate", "helps_not_accepted", "final_dev_gap", "gpu_hours", "recipe_order")
    decided = None
    if len(rows) > 1:
        for i, name in enumerate(crit):
            if rows[0]["_key"][i] != rows[1]["_key"][i]:
                decided = name
                break
    return rows, decided


def d4(ev, th, exp=None):
    ready = _frac(_t(th, "D4", "ready_rate"), "D4.ready_rate")
    done = complete_pilots(ev)
    if exp is not None:
        done = [e for e in done if e == exp]
    if not done:
        return _silent("D4", "no pilot in the evidence has compared every step for every chain")
    pilot = done[-1]
    mode, mc = _replay_mode(ev, pilot)
    one = d1(ev, th, exp=pilot) if gate_rows(ev, pilot) else None
    d1_fired = one is not None and one["fired"]
    d1_det = (one or {}).get("detail") or {}
    d1_by15 = d1_fired and (d1_det.get("blocked_by") or {}).get("id") == "D15"
    # D1 blocks D4 only when D4's own choice is touched (module doc; R4 review
    # 2026-09-27): anything but an explicit blocks_d4 false blocks.
    d1_blocks = d1_fired and not d1_by15 and d1_det.get("blocks_d4") is not False
    if d1_fired and mode == "full":
        if d1_by15:
            return _silent("D4", "D1 fired on the full-replay pilot %s, and D15 blocks it: the flips guard alone "
                                 "stopped every truth-helps miss (%s); no recipe choice is made before the gate "
                                 "protocol is tested (D15: %s)"
                           % (pilot, "; ".join(one["detail"]["blocked_by"].get("pairs") or []),
                              " + ".join(one["detail"]["blocked_by"].get("levers") or [])),
                           pilot, cites=one["cites"])
        if d1_blocks:
            return _silent("D4", "D1 fired on the full-replay pilot %s: the recipe itself forgets (X1), and it %s; "
                                 "no recipe choice is made" % (pilot, d1_det.get("blocks_d4_why") or "blocks D4"),
                           pilot, cites=one["cites"])
    rows, decided = rank_recipes(ev, pilot)
    best = rows[0]
    cites = [c for r in rows for c in r["_cites"]] + ([mc] if mc else [])
    is_ready = best["agree"] * ready.denominator >= ready.numerator * best["compared"]
    detail = {"pilot": pilot, "replay_mode": mode, "best": best["recipe"],
              "rates": {r["recipe"]: "%d/%d" % (r["agree"], r["compared"]) for r in rows},
              "decided_by": decided,
              "ranking": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
              "ready_rate": "%d/%d" % (ready.numerator, ready.denominator)}
    rates = ", ".join("%s %d/%d" % (r["recipe"], r["agree"], r["compared"]) for r in rows)
    # The gate the pilot was decided with (L2 carries it) and, on protocol v2,
    # its pre-registered check (D16): refuted blocks the real loop on it.
    gmode, gcite, _gsrc = pinned_gate(ev, pilot)
    v2m, v2c, v2cite, v2final = _v2_state(ev, th, pilot)
    outcome = (v2c or {}).get("outcome") if v2final is True else None
    detail["gate"] = {"flips_mode": gmode or _t(th, "D15", "v1_mode"), "v2_check": outcome}
    pending = None
    if v2m == _t(th, "D16", "v2_mode"):
        if outcome is None:
            pending = detail["gate"]["v2_check"] = "not final" if v2c else "missing"
            detail["v2_pending"] = pending
        if v2cite is not None:
            cites.append(v2cite)
        if gcite is not None:
            cites.append(gcite)
    if outcome == "refuted":
        if is_ready:
            detail["recipes"] = [best["recipe"]]
        detail["blocked_by"] = {"id": "D16", "exp": pilot, "summary": "protocol v2 refuted on %s" % pilot}
        detail["then"] = ["L2 is withheld: protocol v2 (net flips) was refuted on %s; a person reverts to v1 or "
                          "rethinks the guard (X7)" % pilot]
        summary = ("pilot %s: best recipe %s agrees with the truth arm on %d/%d (%s), but protocol v2, the gate it "
                   "was decided with, is refuted there (D16), so no real loop is built on that gate -> X7"
                   % (pilot, best["recipe"], best["agree"], best["compared"], rates))
        return _diag("D4", True, "warn", summary, cites, [], pilot, detail,
                     name=None if is_ready else D4_NOT_READY)
    v2_note = ""
    if outcome == "supported":
        v2_note = "; protocol v2 supported on %s (D16)" % pilot
    elif outcome == "inconclusive":
        v2_note = "; protocol v2 inconclusive on %s (D16): v2 is unproven, D4 proceeds on its own threshold" % pilot
        detail["v2_unproven"] = True
    if d1_by15:
        # D15 takes precedence over D1, and so over D4 as D1's proxy: the pilot's
        # misses are the flips guard's, not (yet) the recipe's.
        if is_ready:
            detail["recipes"] = [best["recipe"]]
        detail["blocked_by"] = {"id": "D15", "exp": pilot, "summary": one["detail"]["blocked_by"].get("summary")}
        detail["then"] = ["L2 is withheld: no recipe choice and no D1 lever on %s before the gate protocol is "
                          "tested (D15's lever)" % pilot]
        have = {json.dumps(c, sort_keys=True) for c in cites}
        cites += [c for c in one["cites"] if json.dumps(c, sort_keys=True) not in have]
        summary = ("pilot %s: best recipe %s agrees with the truth arm on %d/%d (%s), and D1 fired on it but is "
                   "blocked by D15 (the flips guard alone stopped every truth-helps miss: %s): no recipe choice "
                   "before the gate protocol is tested%s"
                   % (pilot, best["recipe"], best["agree"], best["compared"], rates,
                      "; ".join(one["detail"]["blocked_by"].get("pairs") or []), v2_note))
        return _diag("D4", True, "warn", summary, cites, [], pilot, detail,
                     name=None if is_ready else D4_NOT_READY)
    if d1_fired and not d1_blocks and not d1_by15:
        # D1 fired, but D4's choice is untouched: the chain D4 selects ACCEPTed
        # every clean truth-helps step at D4's threshold. D4 applies its rule
        # unchanged; D1's own levers (X1, a card; L1 in sample mode) stand.
        detail["d1"] = {"fired": True, "blocks_d4": False, "selected": (d1_det.get("d4_selection") or {})
                        .get("selected"), "why": d1_det.get("blocks_d4_why"), "levers": list(one["levers"])}
        have = {json.dumps(c, sort_keys=True) for c in cites}
        cites += [c for c in one["cites"] if json.dumps(c, sort_keys=True) not in have]
        v2_note += "; D1 fired on %s (%s) but %s" % (pilot, " + ".join(one["levers"]) or "no lever",
                                                     d1_det.get("blocks_d4_why"))
    if is_ready and d1_blocks:
        # D1 takes precedence over a ready D4 on the same pilot, in either replay
        # mode (full: silent above, the contract's rule; sample: here), when the
        # chain D4 selects did not ACCEPT a truth-helps step. A recipe that
        # forgets under the pilot's replay is not built into the real loop:
        # D1's levers apply, and L2 waits for a pilot on which D1 does not block.
        detail["recipes"] = [best["recipe"]]
        levers, d1d = _d1_levers(ev, pilot)
        detail.update(d1d)
        detail["blocked_by"] = {"id": "D1", "exp": pilot, "summary": one["summary"]}
        detail["then"] = ["L2 is withheld: D1 fired on %s (the recipe forgets under %s replay); %s first"
                          % (pilot, mode, " + ".join(levers))]
        have = {json.dumps(c, sort_keys=True) for c in cites}
        cites += [c for c in one["cites"] if json.dumps(c, sort_keys=True) not in have]
        summary = ("pilot %s: best recipe %s agrees with the truth arm on %d/%d >= %s (%s), but D1 fired on it "
                   "(%s replay: the recipe forgets) and %s, so no real loop is built from it -> %s"
                   % (pilot, best["recipe"], best["agree"], best["compared"], detail["ready_rate"], rates, mode,
                      d1_det.get("blocks_d4_why") or "blocks D4", " + ".join(levers)))
        return _diag("D4", True, "warn", summary, cites, levers, pilot, detail)
    if is_ready and pending:
        # Decided on protocol v2, whose pre-registered check (D16) is not final:
        # L2 would carry a gate that may yet be refuted, and L9's success
        # criterion needs 'supported'. Not ready until the check is final.
        detail["recipes"] = [best["recipe"]]
        detail["blocked_by"] = {"id": "D16", "exp": pilot, "pending": pending,
                                "summary": "the protocol v2 check on %s is %s" % (pilot, pending)}
        detail["then"] = ["L2 waits for %s's final v2_check (D16): the pilot was decided on protocol v2 and its "
                          "pre-registered check is %s (%s)"
                          % (pilot, pending, "a provisional check: the pilot is not done" if pending == "not final"
                             else "report.json has no v2_check: rerun the report with the current inc.report")]
        summary = ("pilot %s: best recipe %s agrees with the truth arm on %d/%d >= %s (%s), but it was decided on "
                   "protocol v2 and its v2_check is %s: no real loop is built on that gate before the check is "
                   "final (D16)" % (pilot, best["recipe"], best["agree"], best["compared"], detail["ready_rate"],
                                    rates, pending))
        return _diag("D4", True, "warn", summary, cites, [], pilot, detail)
    if is_ready:
        detail["recipes"] = [best["recipe"]]
        summary = ("pilot %s: best recipe %s agrees with the truth arm on %d/%d >= %s (%s; decided by %s) -> "
                   "L2 --replay-mode %s --recipes %s (gate flips_mode %s)%s"
                   % (pilot, best["recipe"], best["agree"], best["compared"], detail["ready_rate"], rates,
                      decided or "sole chain", mode, best["recipe"], detail["gate"]["flips_mode"], v2_note))
        return _diag("D4", True, "info", summary, cites, ["L2"], pilot, detail)
    levers, d1d = _d1_levers(ev, pilot)
    detail.update(d1d)
    summary = ("pilot %s: no recipe tracks the truth arm (best %s %d/%d = %.4f < %s; %s) -> %s%s"
               % (pilot, best["recipe"], best["agree"], best["compared"], best["rate"], detail["ready_rate"],
                  rates, " + ".join(levers), v2_note))
    return _diag("D4", True, "warn", summary, cites, levers, pilot, detail, name=D4_NOT_READY)


# ------------------------------------------------------------------ health
def d5(ev, th):
    exp = ev.exp
    kinds = _t(th, "D5", "transient_cause_kinds")
    max_auto = _t(th, "D5", "max_auto_unblocks_per_unit")
    prefix = _t(th, "D5", "auto_reason_prefix")
    state, rep = ev.json("%s/state.json" % exp), _report(ev, exp)
    art = "%s/state.json" % exp if state is not None else "%s/report.json" % exp
    blocked = (state if state is not None else (rep or {})).get("blocked") or {}
    if not blocked:
        return _silent("D5", "no blocked unit" if (state or rep) else "no state.json or report.json", exp)
    if state is not None:
        unblocks = state.get("unblocks") or []
    else:
        unblocks = [i for i in (rep or {}).get("interventions") or [] if i.get("type") == "unblock"]
    units, cites, levers = [], [], []
    for u, b in sorted(blocked.items()):
        kind = ((b or {}).get("cause") or {}).get("kind")
        autos = [x for x in unblocks if x.get("unit") == u
                 and (x.get("auto") or str(x.get("reason", "")).startswith(prefix))]
        transient = kind in kinds and len(autos) < max_auto
        units.append({"unit": u, "cause": kind, "transient": transient, "auto_unblocks": len(autos)})
        cites.append(ev.cite(art, E.pointer("blocked", u, "error")))
        cites.append(ev.cite(art, E.pointer("blocked", u, "cause")))
    if any(x["transient"] for x in units):
        levers.append("L7")
    if any(not x["transient"] for x in units):
        levers.append("OP_PAUSE")
    sev = "crit" if "OP_PAUSE" in levers else "warn"
    summary = "blocked: %s -> %s" % (", ".join("%s (%s%s)" % (x["unit"], x["cause"], ", transient" if x["transient"]
                                                               else "") for x in units), " + ".join(levers))
    return _diag("D5", True, sev, summary, cites, levers, exp, {"units": units})


def d6(ev, th):
    exp = ev.exp
    hours = _t(th, "D6", "stale_hours")
    prefix = _t(th, "D6", "job_prefix") % exp
    state, ctx = ev.json("%s/state.json" % exp), _ctx(ev)
    if state is None:
        return _silent("D6", "no state.json in the evidence", exp)
    if state.get("done"):
        return _silent("D6", "%s is done" % exp, exp)
    hist, squeue, now = ctx.get("history"), ctx.get("squeue"), ctx.get("now_utc")
    if not isinstance(hist, list) or not isinstance(squeue, list) or not now:
        return _silent("D6", "not evaluated: the context lacks history, squeue or now_utc", exp)
    gen = state.get("generation")
    since = None
    for i, h in sorted(enumerate(hist), key=lambda ih: str(ih[1].get("utc"))):
        if h.get("generation") != gen:
            since = None
        elif since is None:
            since = (i, h)
    queued = [j for j in squeue if str(j).startswith(prefix)]
    if since is None:
        return _silent("D6", "generation %s is new since the last snapshot" % gen, exp)
    age = (_utc(now) - _utc(since[1]["utc"])).total_seconds() / 3600.0
    cites = [ev.cite("%s/state.json" % exp, "/generation"), ev.cite(E.CONTEXT, "/history/%d/utc" % since[0]),
             ev.cite(E.CONTEXT, "/squeue"), ev.cite(E.CONTEXT, "/now_utc")]
    summary = ("generation %s unchanged for %.1f h (> %g h), %d job(s) %s* queued, not done"
               % (gen, age, hours, len(queued), prefix))
    if age <= hours or queued:
        return _silent("D6", summary, exp, cites=cites)
    return _diag("D6", True, "warn", summary + " -> OP_ADVANCE, then OP_PAUSE if still stale", cites,
                 ["OP_ADVANCE"], exp, {"then": ["OP_PAUSE if still stale"], "hours": age})


def d7(ev, th):
    pats = _t(th, "D7", "error_patterns")
    err = (_ctx(ev).get("advance") or {}).get("error")
    if not err:
        return _silent("D7", "no advance error in the context", ev.exp)
    hit = [p for p in pats if re.search(p, str(err))]
    if not hit:
        return _silent("D7", "the advance error is not a code refusal", ev.exp)
    return _diag("D7", True, "crit", "advance refused on code (%s) -> pause; X5 (human card)" % hit[0],
                 [ev.cite(E.CONTEXT, "/advance/error")], ["OP_PAUSE", "X5"], ev.exp, {"pattern": hit[0]})


def d8(ev, th):
    exp = ev.exp
    lo, hi = _t(th, "D8", "p_band")
    s_min = _t(th, "D8", "share_min")
    mult = _t(th, "D8", "sd_mult")
    size_mult = _t(th, "D8", "size_multiplier")
    rows = gate_rows(ev, exp)
    if not rows:
        return _silent("D8", "no decided chain step", exp)
    hits, cites = [], []
    for r in rows:
        vals = [_num(r[k]) for k in ("p_data", "cand_mean", "null_mean", "cand_sd", "null_sd")]
        if None in vals:
            continue
        p, cm, nm, cs, ns = vals
        if lo < p < hi and abs(cm - nm) < mult * max(cs, ns):
            hits.append(r)
            cites += [r.cite(k) for k in ("p_data", "cand_mean", "null_mean", "cand_sd", "null_sd")]
    summary = ("%d/%d gate entries have P_data in (%g, %g) and |cand - null| < %g * max(sd)"
               % (len(hits), len(rows), lo, hi, mult))
    if len(hits) < s_min * len(rows) or not hits:
        return _silent("D8", summary, exp)
    d = _defn(ev, exp) or {}
    detail = {"entries": len(hits)}
    levers = []
    if d.get("builder") == REALLOOP_BUILDER and isinstance(d.get("increment_images"), int):
        detail["size"] = int(d["increment_images"]) * int(size_mult)
        cites.append(ev.cite("%s/exp.json" % exp, "/increment_images"))
        levers = ["L5"]
        tail = " -> L5 (--size %d)" % detail["size"]
    else:
        tail = " (informational: a pilot has no --size)"
    return _diag("D8", True, "warn", summary + tail, cites, levers, exp, detail)


def d9(ev, th):
    exp = ev.exp
    share = _t(th, "D9", "warmup_share_max")
    d = _defn(ev, exp)
    iee = ((d or {}).get("effective_warmup") or {}).get("incremental_effective_epochs")
    recipes = (d or {}).get("recipes") or {}
    if not isinstance(iee, dict) or _num(iee.get("max")) is None or not recipes:
        return _silent("D9", "no incremental effective warmup in exp.json", exp)
    r_max = max(recipes, key=lambda r: (_num(recipes[r].get("epochs")) or 0, r))
    epochs = _num(recipes[r_max].get("epochs"))
    if not epochs:
        return _silent("D9", "no recipe epochs in exp.json", exp)
    ratio = iee["max"] / epochs
    cites = [ev.cite("%s/exp.json" % exp, "/effective_warmup/incremental_effective_epochs/max"),
             ev.cite("%s/exp.json" % exp, E.pointer("recipes", r_max, "epochs"))]
    summary = ("the warmup floor covers up to %.2f of %g incremental epochs (%.3f > %g)"
               % (iee["max"], epochs, ratio, share)) if ratio > share else (
        "the warmup floor covers up to %.2f of %g incremental epochs (%.3f <= %g)" % (iee["max"], epochs, ratio, share))
    if ratio <= share:
        return _silent("D9", summary, exp, cites=cites)
    return _diag("D9", True, "info", summary + "; supports L1", cites, [], exp, {"ratio": ratio})


def _owner_hours(rep):
    out = {}
    for o, g in ((rep or {}).get("gpu_hours") or {}).items():
        if isinstance(g, dict) and _num(g.get("hours")) is not None and g.get("runs"):
            out[o] = g["hours"] / g["runs"]
    return out


def d10(ev, th):
    exp = ev.exp
    margin = _t(th, "D10", "margin")
    budget = _ctx(ev).get("budget")
    if not isinstance(budget, dict) or _num(budget.get("envelope_su")) is None or _num(budget.get("spent_su")) is None:
        return _silent("D10", "not evaluated: no campaign budget in the context", exp)
    cites = [ev.cite(E.CONTEXT, "/budget/envelope_su"), ev.cite(E.CONTEXT, "/budget/spent_su")]
    if _num(budget.get("projected_su")) is not None:
        proj, how = float(budget["projected_su"]), "the ticker's projection"
        cites.append(ev.cite(E.CONTEXT, "/budget/projected_su"))
    else:
        state, rep = ev.json("%s/state.json" % exp), _report(ev, exp)
        derived = ev.json("%s/%s" % (exp, E.DERIVED_STATE_RUNS))
        per = _owner_hours(rep)
        open_by = {}
        if state is not None and isinstance(state.get("runs"), dict):
            for rid, r in sorted(state["runs"].items()):
                if r.get("status") != "complete":
                    open_by[r.get("owner")] = open_by.get(r.get("owner"), 0) + 1
            src = ("%s/state.json" % exp, "/runs")
        elif isinstance(derived, dict) and isinstance(derived.get("by_owner"), dict):
            for o, c in sorted(derived["by_owner"].items()):
                open_by[o] = sum(v for s, v in c.items() if s != "complete" and isinstance(v, int))
            src = ("%s/%s" % (exp, E.DERIVED_STATE_RUNS), "/by_owner")
        else:
            src = None
        if src is None or not per:
            return _silent("D10", "not evaluated: no projection (needs the runs of state.json and GPU-hours "
                                  "per run)", exp, cites=cites)
        proj = sum(per[o] * n for o, n in open_by.items() if o in per)
        how = "%d open run(s) at the report's mean GPU-hours per run of their owner" % sum(
            n for o, n in open_by.items() if o in per)
        cites.append(ev.cite(*src))
    env, spent = float(budget["envelope_su"]), float(budget["spent_su"])
    summary = "spent %.1f + projected %.1f SU (%s) vs envelope %.1f SU" % (spent, proj, how, env)
    if spent + proj <= env * margin:
        return _silent("D10", summary, exp, cites=cites)
    return _diag("D10", True, "crit", summary + " -> pause", cites, ["OP_PAUSE"], exp,
                 {"spent_su": spent, "projected_su": proj, "envelope_su": env})


def d11(ev, th):
    share_min = _t(th, "D11", "truth_share_min")
    agree_min = _frac(_t(th, "D11", "agreement_min"), "D11.agreement_min")
    n_min = _t(th, "D11", "min_pilots")
    done = complete_pilots(ev)
    if not done:
        return _silent("D11", "no complete pilot")
    cites, good = [], []
    for e in done:
        rep = _report(ev, e)
        best = max(Fraction(int(a["agree"]), int(a["compared"])) for a in rep["agreement"].values())
        for r in rep["agreement"]:
            cites.append(ev.cite("%s/report.json" % e, "/agreement/%s/agree" % E._esc(r)))
        if best >= agree_min:
            good.append(e)
    last = _report(ev, done[-1])
    truth = _num(((last.get("gpu_hours") or {}).get("truth") or {}).get("hours"))
    total = _num(last.get("gpu_hours_total"))
    if truth is None or not total:
        return _silent("D11", "no truth-arm GPU-hours in %s's report" % done[-1])
    cites += [ev.cite("%s/report.json" % done[-1], "/gpu_hours/truth/hours"),
              ev.cite("%s/report.json" % done[-1], "/gpu_hours_total")]
    share = truth / total
    summary = ("truth arm %.2f of %.2f GPU-h (%.0f%%) in %s; %d pilot(s) with best agreement >= %s (need %d)"
               % (truth, total, 100 * share, done[-1], len(good), agree_min, n_min))
    if share > share_min and len(good) >= n_min:
        return _diag("D11", True, "info", summary + " -> L6 allowed", cites, ["L6"], done[-1],
                     {"share": share, "pilots": good})
    return _silent("D11", summary, done[-1], cites=cites)


def d12(ev, th):
    exp = ev.exp
    rows = gate_rows(ev, exp)
    d, rep = _defn(ev, exp), _report(ev, exp)
    chains = sorted((d or {}).get("recipes") or (rep or {}).get("agreement") or {r.chain for r in rows})
    n_steps = len((d or {}).get("steps") or (rep or {}).get("steps") or [])
    by = {}
    for r in rows:
        by.setdefault(r.k, {})[r.chain] = r
    if len(chains) < 2 or not n_steps or len(by) < n_steps or any(set(v) != set(chains) for v in by.values()):
        return _silent("D12", "not every step is decided by %d or more chains" % 2, exp)
    same = all(len({r["verdict"] for r in v.values()}) == 1 for v in by.values())
    if not same:
        return _silent("D12", "the chains decide differently on at least one step", exp)
    cites = [r.cite("verdict") for k in sorted(by) for r in by[k].values()]
    per = {}
    for c in chains:
        h = _num((((rep or {}).get("gpu_hours") or {}).get("chain:%s" % c) or {}).get("hours"))
        if h is not None:
            per[c] = h
            cites.append(ev.cite("%s/report.json" % exp, "/gpu_hours/%s/hours" % E._esc("chain:%s" % c)))
    cheapest = min(per, key=lambda c: (per[c], c)) if per else None
    summary = ("all %d chains give identical verdicts on all %d steps; cheapest %s: only one recipe goes forward "
               "(D4 picks it by its own tie-break)" % (len(chains), n_steps, cheapest or "unknown"))
    return _diag("D12", True, "info", summary, cites, [], exp, {"cheapest": cheapest, "gpu_hours": per})


def d13(ev, th):
    table = _t(th, "D13", "contradicts")
    outs = _ctx(ev).get("outcomes") or []
    bad, cites = [], []
    for i, o in enumerate(outs):
        direction = ((o or {}).get("predicted") or {}).get("direction")
        if o and o.get("verdict") in (table.get(direction) or []):
            bad.append({"lever": o.get("lever"), "child_exp": o.get("child_exp"), "direction": direction,
                        "verdict": o.get("verdict")})
            cites += [ev.cite(E.CONTEXT, "/outcomes/%d/predicted/direction" % i),
                      ev.cite(E.CONTEXT, "/outcomes/%d/verdict" % i)]
    if not bad:
        return _silent("D13", "%d outcome(s), none contradicts its prediction" % len(outs))
    summary = "; ".join("%s on %s predicted %s, came out %s" % (b["lever"], b["child_exp"], b["direction"],
                                                                 b["verdict"]) for b in bad)
    return _diag("D13", True, "crit", summary + " -> escalate", cites, ["OP_ESCALATE"], None, {"contradicted": bad})


def d14(ev, th):
    rec = ev.loader_record()
    bad = [(i, n) for i, n in enumerate(rec["touched"]) if not E.allowed(n) or E._NON_DEV_SCORE.search(n)]
    leak = ev.leaks()
    if not bad and not leak:
        return _silent("D14", "the loader read %d allow-listed artifact(s); no non-dev exam value in the evidence"
                       % len(rec["touched"]))
    cites = [M.cite(E.LOADER, n, pointer="/touched/%d" % i) for i, n in bad]
    cites += [M.cite(E.LOADER, "non-dev exam key", pointer=p) for p in leak]
    return _diag("D14", True, "crit", "a decision path touched a non-dev exam: %s -> hard stop"
                 % ([n for _, n in bad] + leak)[:5], cites, ["OP_HALT"], ev.exp, {"touched": bad, "leaks": leak})


def dref(ev, th):
    refs = _ctx(ev).get("refusals") or []
    items, cites = [], []
    for i, r in enumerate(refs):
        m = _prereq(str((r or {}).get("message") or ""))
        if m is not None and m.get("prerequisite") == "L3":
            continue                                   # D2 owns the relevance prerequisite
        items.append({"builder": (r or {}).get("builder"), "prerequisite": (m or {}).get("prerequisite"),
                      "needs": (m or {}).get("needs") if m else "unmapped: a person reads the refusal",
                      "retry": (m or {}).get("retry") is not False})
        cites.append(ev.cite(E.CONTEXT, "/refusals/%d/message" % i))
    if not items:
        return _silent("DREF", "no builder refusal outside D2's")
    levers = sorted({x["prerequisite"] for x in items if x["prerequisite"]})
    sev = "crit" if any(not x["prerequisite"] for x in items) else "warn"
    if sev == "crit":
        # A refusal no menu lever answers (mapped to none, or unmapped) is a
        # person's: the ticker puts the escalation card up (campaign._diagnose).
        levers.append("OP_ESCALATE")
    summary = "; ".join("%s refused: %s" % (x["builder"], x["prerequisite"] or x["needs"]) for x in items)
    return _diag("DREF", True, sev, summary, cites, levers, ev.exp, {"refusals": items})


# ------------------------------------------------------ the funnel audit (D17-D19)
# docs/FUNNEL_AUDIT.md 8.4. The suspicion signals S1-S7 are functions of the
# funnel ledger (funnel/funnel_ledger.json, format funnel-ledger/1), the claims
# register and the loop reports only. They read stages by their "role" key and
# the ledger's own class lists, never a stage id or a class name, so the same
# code and thresholds run on any domain (replay case R13).
FUNNEL_LEDGER_FORMAT = "funnel-ledger/1"
FUNNEL_AUDIT_FORMAT = "funnel-audit/1"
SIGNAL_IDS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
RECOVER_POLICY_ORDER = ("R-A", "R-C", "R-T", "R-V", "R-J", "R-F")   # funnel/recover.py POLICIES


def _funnel_ledger(ev):
    """(ledger, why): the funnel ledger in the evidence, or None with why not."""
    led = ev.json(E.FUNNEL_LEDGER)
    if led is None:
        return None, "no %s in the evidence" % E.FUNNEL_LEDGER
    if not isinstance(led, dict) or led.get("format") != FUNNEL_LEDGER_FORMAT \
            or not isinstance(led.get("stages"), list):
        return None, "%s is not a %s file with stages" % (E.FUNNEL_LEDGER, FUNNEL_LEDGER_FORMAT)
    return led, ""


def _stage_index(led, roles):
    """[(index, stage)] of the ledger's stages whose role is in roles, in stage order."""
    return [(i, s) for i, s in enumerate(led.get("stages") or [])
            if isinstance(s, dict) and s.get("role") in roles]


def _lcite(ev, *parts):
    return ev.cite(E.FUNNEL_LEDGER, E.pointer(*parts))


def _int(x):
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def _signal(holds, summary, cites=(), detail=None):
    return {"holds": holds, "summary": summary, "cites": list(cites), "detail": detail or {}}


def _s1(ev, led, th):
    roles = _t(th, "SIGNALS", "S1_roles")
    tmin = _t(th, "SIGNALS", "S1_reject_accept_min")
    oreason = _t(th, "SIGNALS", "S1_other_reason")
    tc = _stage_index(led, [roles[0]])
    oc = _stage_index(led, roles[1:])
    if len(tc) != 1:
        return _signal(None, "not evaluated: the ledger has %d stage(s) of role %s, not one" % (len(tc), roles[0]))
    i, st = tc[0]
    kept = _int(st.get("kept"))
    disc = st.get("discarded") if isinstance(st.get("discarded"), dict) else None
    if kept is None or disc is None or any(_int(v) is None for v in disc.values()):
        return _signal(None, "not evaluated: the %s stage records no whole kept and discarded counts" % roles[0])
    cites = [_lcite(ev, "stages", i, "kept")] + [_lcite(ev, "stages", i, "discarded", r) for r in sorted(disc)]
    rejected = sum(disc.values())
    parts = {"%s:%s" % (st.get("id"), r): disc[r] for r in sorted(disc)}
    for j, o in oc:
        od = o.get("discarded") if isinstance(o.get("discarded"), dict) else {}
        if _int(od.get(oreason)) is None:
            return _signal(None, "not evaluated: the %s stage records no whole %r discard" % (o.get("role"), oreason))
        rejected += od[oreason]
        parts["%s:%s" % (o.get("id"), oreason)] = od[oreason]
        cites.append(_lcite(ev, "stages", j, "discarded", oreason))
    if not kept:
        return _signal(None, "not evaluated: the %s stage kept nothing (the ratio is undefined)" % roles[0], cites)
    ratio = rejected / float(kept)
    summary = ("S1 reject/accept = %d / %d = %.2f (%s) %s %g"
               % (rejected, kept, ratio, " + ".join("%s %d" % (k, v) for k, v in parts.items()),
                  ">=" if ratio >= tmin else "<", tmin))
    return _signal(ratio >= tmin, summary, cites, {"rejected": rejected, "kept": kept, "ratio": ratio,
                                                    "parts": parts, "stage": st.get("id")})


def _s2(ev, led, th):
    kinds = _t(th, "SIGNALS", "S2_uninformative_kinds")
    tmin = _t(th, "SIGNALS", "S2_uninformative_share_min")
    smin = _t(th, "SIGNALS", "S2_source_share_min")
    ls = led.get("label_spaces")
    if not isinstance(ls, dict) or not ls:
        return _signal(None, "not evaluated: the ledger has no label_spaces")
    unin, boxes, cites, sources = 0, 0, [], []
    for src in sorted(ls):
        e = ls[src] if isinstance(ls[src], dict) else {}
        k = e.get("kinds") if isinstance(e.get("kinds"), dict) else {}
        b = _int(e.get("boxes"))
        if b is None or any(_int(k.get(x, 0)) is None for x in kinds):
            return _signal(None, "not evaluated: label_spaces/%s has no whole box counts" % src)
        u = sum(k.get(x, 0) for x in kinds)
        unin += u
        boxes += b
        cites.append(_lcite(ev, "label_spaces", src, "boxes"))
        cites += [_lcite(ev, "label_spaces", src, "kinds", x) for x in kinds if x in k]
        if b and u / float(b) > smin:
            sources.append({"source": src, "uninformative": u, "boxes": b})
    try:
        cites.insert(0, _lcite(ev, "name_status_version"))
    except KeyError:
        return _signal(None, "not evaluated: the ledger does not record its name_status_version", cites)
    if not boxes:
        return _signal(None, "not evaluated: the label spaces hold no boxes", cites)
    share = unin / float(boxes)
    sources.sort(key=lambda x: (-x["uninformative"], x["source"]))
    summary = ("S2 uninformative names (%s, name status %s) = %d / %d = %.3f %s %g; %d source(s) more than %g "
               "uninformative" % ("+".join(kinds), led.get("name_status_version"), unin, boxes, share,
                                  ">=" if share >= tmin else "<", tmin, len(sources), smin))
    return _signal(share >= tmin, summary, cites, {"uninformative": unin, "boxes": boxes, "share": share,
                                                    "name_status_version": led.get("name_status_version"),
                                                    "sources": sources})


def _s3(ev, led, th):
    roles = _t(th, "SIGNALS", "S3_roles")
    nmin = _t(th, "SIGNALS", "S3_min_joined_boxes")
    frac = _t(th, "SIGNALS", "S3_yield_frac_of_median")
    js, ts = _stage_index(led, [roles[0]]), _stage_index(led, [roles[1]])
    if len(js) != 1 or len(ts) != 1:
        return _signal(None, "not evaluated: the ledger needs one %s and one %s stage" % tuple(roles))
    (ji, j), (ti, t) = js[0], ts[0]
    jk, tk = j.get("kept_by_label"), t.get("kept_by_label")
    targets = led.get("target_classes") or []
    if not isinstance(jk, dict) or not isinstance(tk, dict) or not targets:
        return _signal(None, "not evaluated: kept_by_label or target_classes is missing from the ledger")
    rows = []
    for k in targets:
        n = _int(jk.get(k))
        if n is None or n < nmin:
            continue
        v = _int(tk.get(k, 0))
        if v is None:
            return _signal(None, "not evaluated: %s kept_by_label[%s] is not a whole count" % (roles[1], k))
        rows.append({"class": k, "verified": v, "joined": n, "share": v / float(n)})
    if len(rows) < 2:
        return _signal(None, "not evaluated: %d class(es) with >= %d joined boxes" % (len(rows), nmin))
    shares = sorted(r["share"] for r in rows)
    m = len(shares)
    median = shares[m // 2] if m % 2 else (shares[m // 2 - 1] + shares[m // 2]) / 2.0
    cut = frac * median
    out = [r for r in rows if r["share"] < cut]
    cites = []
    for r in rows:
        cites += [_lcite(ev, "stages", ji, "kept_by_label", r["class"])]
        if r["class"] in tk:
            cites += [_lcite(ev, "stages", ti, "kept_by_label", r["class"])]
    summary = ("S3 class yield: median verified share %.3f over %d class(es) with >= %d joined boxes; below %g x "
               "median: %s" % (median, len(rows), nmin, frac, ", ".join("%s %d/%d = %.3f" % (
                   r["class"], r["verified"], r["joined"], r["share"]) for r in out) or "none"))
    return _signal(bool(out), summary, cites, {"median": median, "cut": cut, "classes": rows, "outliers": out})


def _s4(ev, led, th):
    roles = _t(th, "SIGNALS", "S4_roles")
    nmin = _t(th, "SIGNALS", "S4_min_sources_below_q05")
    ds = led.get("domain_scores")
    ref = (ds or {}).get("reference") if isinstance(ds, dict) else None
    q05 = _num((ref or {}).get("q05"))
    if q05 is None:
        return _signal(None, "not evaluated: the ledger has no domain_scores.reference.q05")
    cites = [_lcite(ev, "domain_scores", "reference", "q05")]
    sets, unknown = [], []
    for i, st in _stage_index(led, roles):
        cal = st.get("calibration")
        if not isinstance(cal, dict):
            continue
        for k, s in enumerate(cal.get("known_truth_sets") or []):
            sc = (s or {}).get("domain_score")
            if not (isinstance(sc, list) and len(sc) == 2 and all(_num(x) is not None for x in sc)):
                unknown.append((s or {}).get("id"))
                continue
            sets.append({"stage": st.get("id"), "id": s.get("id"), "domain_score": sc,
                         "in_domain": sc[0] >= q05 and sc[1] <= 1.0})
            cites.append(_lcite(ev, "stages", i, "calibration", "known_truth_sets", k, "domain_score"))
    below = []
    for src in sorted(k for k in ds if k != "reference"):
        v = _num(ds[src])
        if v is not None and v < q05:
            below.append({"source": src, "median": v})
            cites.append(_lcite(ev, "domain_scores", src))
    if unknown:
        return _signal(None, "not evaluated: known-truth set(s) %s record no domain score" % unknown, cites)
    out_of_domain = [s for s in sets if not s["in_domain"]]
    holds = not out_of_domain and len(below) >= nmin
    summary = ("S4 calibration domain gap: %d known-truth set(s), %d in the reference domain [q05 %.4g, 1]; %d "
               "source median(s) below q05 (%s)%s"
               % (len(sets), len(sets) - len(out_of_domain), q05, len(below),
                  ", ".join("%s %.4g" % (b["source"], b["median"]) for b in below) or "none",
                  "" if holds else ("; set(s) %s cover other domains" % [s["id"] for s in out_of_domain]
                                    if out_of_domain else "")))
    return _signal(holds, summary, cites, {"q05": q05, "sets": sets, "below_q05": below})


def _s5(ev, led, th):
    from ..funnel import ledger as FL
    pairs = FL.unaudited_dependencies(led)
    idx = {s.get("id"): i for i, s in enumerate(led.get("stages") or []) if isinstance(s, dict)}
    cites, rows = [], []
    for stage, dep in pairs:
        i, j = idx.get(stage), idx.get(dep)
        if i is None or j is None:
            continue
        st = led["stages"][i]
        row = {"stage": stage, "depends_on": dep, "in": _int(st.get("in")), "kept": _int(st.get("kept")),
               "unit": st.get("unit"), "role": st.get("role")}
        rows.append(row)
        cites += [_lcite(ev, "stages", i, "depends_on"), _lcite(ev, "stages", j, "audit")]
        if row["in"] is not None and row["kept"] is not None:
            cites += [_lcite(ev, "stages", i, "in"), _lcite(ev, "stages", i, "kept")]
    summary = ("S5 derived dependency: %d keep or discard criterion(s) depend on an unaudited stage: %s"
               % (len(rows), ", ".join("%s -> %s%s" % (r["stage"], r["depends_on"],
                                                        " (%d of %d %ss kept)" % (r["kept"], r["in"], r["unit"])
                                                        if r["in"] is not None and r["kept"] is not None else "")
                                       for r in rows) or "none"))
    return _signal(bool(rows), summary, cites, {"pairs": rows})


def _cell_share(cell):
    """The share a reject-class cell holds of its sample: its 'share', or 'n'
    over 'of' (or 'total'); None when the cell gives neither."""
    if not isinstance(cell, dict):
        return None
    s = _num(cell.get("share"))
    if s is not None:
        return s
    n, of = _num(cell.get("n")), _num(cell.get("of", cell.get("total")))
    return n / of if n is not None and of else None


def _s6(ev, led, th):
    smin = _t(th, "SIGNALS", "S6_top_cell_share_min")
    rc = led.get("reject_class")
    if not isinstance(rc, dict):
        return _signal(None, "not evaluated: the ledger records no reject-class sample (reject_class is null)")
    tc = _stage_index(led, ["target_check"])
    disc_src = set()
    if len(tc) == 1:
        for src, e in sorted((tc[0][1].get("by_source") or {}).items()):
            d = (e or {}).get("discarded") if isinstance(e, dict) else None
            if isinstance(d, dict) and any(_int(v) for v in d.values()):
                disc_src.add(src)
    cites = []
    samp = rc.get("sample_sources") if isinstance(rc.get("sample_sources"), dict) else None
    overlap = sorted(disc_src & set(samp or {})) if samp is not None else []
    if samp is not None:
        cites.append(_lcite(ev, "reject_class", "sample_sources"))
    key = "drawn_top_cell" if isinstance(rc.get("drawn_top_cell"), dict) else "eligible_top_cell"
    share = _cell_share(rc.get(key))
    if share is not None:
        cites.append(_lcite(ev, "reject_class", key))
    if samp is None and share is None:
        return _signal(None, "not evaluated: the reject-class record has neither its sample's sources nor a top "
                             "cell with a share")
    holds = bool(overlap) or (share is not None and share >= smin)
    summary = ("S6 reject-class confound: %d sample source(s) among the sources whose target boxes were rejected%s; "
               "the %s holds %s of the sample (cut %g)"
               % (len(overlap), " (%s)" % ", ".join(overlap) if overlap else "",
                  key.replace("_", " "), "%.3f" % share if share is not None else "an unknown share", smin))
    return _signal(holds, summary, cites, {"overlap": overlap, "top_cell": rc.get(key), "top_cell_key": key,
                                           "share": share})


def _s7(ev, th):
    kinds = _t(th, "SIGNALS", "S7_rejected_kinds")
    rows, cites, seen = [], [], []
    for e in ev.exps():
        d, rep = _defn(ev, e), _report(ev, e)
        done, _dc = _finished(ev, e)
        if not (isinstance(d, dict) and isinstance(rep, dict) and done):
            continue
        dsteps = {s.get("name"): (k, s) for k, s in enumerate(d.get("steps") or []) if isinstance(s, dict)}
        helps_rej, helps_clean = [], []
        for k, s in enumerate(rep.get("steps") or []):
            if not isinstance(s, dict):
                continue
            name = s.get("step")
            if name not in dsteps:
                continue
            dk, ds = dsteps[name]
            if ((s.get("truth") or {}).get("verdict")) != HELPS:
                continue
            if ds.get("kind") in kinds:
                helps_rej.append((k, dk, name))
            if ds.get("clean") is True:
                helps_clean.append((k, dk, name))
        if not helps_rej and not helps_clean:
            continue
        seen.append(e)
        if helps_rej and not helps_clean:
            for k, dk, name in helps_rej:
                rows.append({"exp": e, "step": name, "report_index": k, "exp_index": dk})
                cites += [ev.cite("%s/report.json" % e, "/steps/%d/truth/verdict" % k),
                          ev.cite("%s/exp.json" % e, "/steps/%d/kind" % dk)]
    summary = ("S7 truth-arm contradiction: %s" % ("; ".join(
        "%s step %s (kind in %s) is truth %s while no clean step is" % (r["exp"], r["step"], kinds, HELPS)
        for r in rows) if rows else "no finished loop has a step drawn from rejected data that helps while no "
                                    "clean step does"))
    return _signal(bool(rows), summary, cites, {"steps": rows})


def funnel_signals(ev, th):
    """{S id: {"holds": True | False | None, "summary", "cites", "detail"}}.
    None means not evaluated (the summary says why); a rule that raises is
    reported as not evaluated with the error, never as a silent False."""
    led, why = _funnel_ledger(ev)
    out = {}
    for sid, fn in (("S1", _s1), ("S2", _s2), ("S3", _s3), ("S4", _s4), ("S5", _s5), ("S6", _s6)):
        if led is None:
            out[sid] = _signal(None, "not evaluated: %s" % why)
            continue
        try:
            out[sid] = fn(ev, led, th)
        except _Missing:
            raise
        except Exception as e:                      # the ledger is untrusted input
            out[sid] = _signal(None, "not evaluated: %s raised %s (%s)" % (sid, type(e).__name__, str(e)[:200]))
    try:
        out["S7"] = _s7(ev, th)
    except _Missing:
        raise
    except Exception as e:
        out["S7"] = _signal(None, "not evaluated: S7 raised %s (%s)" % (type(e).__name__, str(e)[:200]))
    return out


def _latest_loop(ev):
    """(exp, report) of the latest finished real loop (exp.json builder
    inc.realloop build, report done), or (None, None)."""
    last = (None, None)
    for e in ev.exps():
        d, rep = _defn(ev, e), _report(ev, e)
        if isinstance(d, dict) and d.get("builder") == REALLOOP_BUILDER and isinstance(rep, dict) \
                and rep.get("done") is True:
            last = (e, rep)
    return last


def _open_negative_claims(ev, th):
    """[(index, claim)] of the claims register's open or challenged claims of a
    negative polarity (D17 (C), claims.open_negative)."""
    pol = _t(th, "D17", "conclusion_polarities")
    sts = _t(th, "D17", "claim_statuses")
    reg = ev.json(E.CLAIMS)
    out = []
    for i, c in enumerate((reg or {}).get("claims") or [] if isinstance(reg, dict) else []):
        if isinstance(c, dict) and c.get("polarity") in pol and c.get("status") in sts:
            out.append((i, c))
    return out


def _conclusion(ev, th):
    """(holds, cites, detail): D17's condition (C), a negative conclusion exists."""
    cites, forms = [], []
    exp, rep = _latest_loop(ev)
    if exp is not None:
        accepted, helps = [], []
        vcites = []
        d = _defn(ev, exp) or {}
        clean = {s.get("name"): s.get("clean") for s in d.get("steps") or [] if isinstance(s, dict)}
        for k, s in enumerate(rep.get("steps") or []):
            if not isinstance(s, dict):
                continue
            for r, c in sorted((s.get("chains") or {}).items()):
                if isinstance(c, dict) and "verdict" in c:
                    vcites.append(ev.cite("%s/report.json" % exp, E.pointer("steps", k, "chains", r, "verdict")))
                    if c.get("verdict") == ACCEPT:
                        accepted.append("%s/%s" % (s.get("step"), r))
            if clean.get(s.get("step")) is True and "verdict" in (s.get("truth") or {}):
                vcites.append(ev.cite("%s/report.json" % exp, E.pointer("steps", k, "truth", "verdict")))
                if s["truth"]["verdict"] == HELPS:
                    helps.append(s.get("step"))
        if not accepted and not helps:
            forms.append({"form": "loop_negative", "exp": exp,
                          "why": "%s accepted nothing and no clean step is truth %s" % (exp, HELPS)})
            cites += vcites
        sized = _t(th, "D17", "sized_flag")
        bs = ev.json("%s/build_summary.json" % exp)
        try:
            flag = walk(bs, E.pointer(*sized)) if isinstance(bs, dict) else None
        except KeyError:
            flag = None
        if flag is False:
            forms.append({"form": "sized_down", "exp": exp,
                          "why": "%s's increment size is not the builders' default (the evidence sizing rule "
                                 "shrank M)" % exp})
            cites.append(ev.cite("%s/build_summary.json" % exp, E.pointer(*sized)))
            try:
                cites.append(ev.cite("%s/build_summary.json" % exp, "/size/images_per_increment"))
            except KeyError:
                pass
    claims = _open_negative_claims(ev, th)
    if claims:
        forms.append({"form": "claims", "claims": [c.get("id") for _i, c in claims],
                      "why": "open or challenged claim(s) of a negative polarity: %s"
                             % ", ".join("%s (%s, %s)" % (c.get("id"), c.get("polarity"), c.get("status"))
                                         for _i, c in claims)})
        for i, _c in claims:
            cites += [ev.cite(E.CLAIMS, "/claims/%d/polarity" % i), ev.cite(E.CLAIMS, "/claims/%d/status" % i)]
    return bool(forms), cites, {"forms": forms, "loop": exp}


def _valid_audit(ev, led):
    """(audit or None, cites, why): the evidence's funnel/audit_v1.json when it
    is a valid audit of this ledger (its ledger_fingerprint is the ledger's
    fingerprint and it is not invalid for calibration overlap)."""
    a = ev.json(E.FUNNEL_AUDIT)
    if a is None:
        return None, [], "no %s in the evidence" % E.FUNNEL_AUDIT
    if not isinstance(a, dict) or a.get("format") != FUNNEL_AUDIT_FORMAT:
        return None, [], "%s is not a %s file" % (E.FUNNEL_AUDIT, FUNNEL_AUDIT_FORMAT)
    cites = []
    try:
        cites.append(ev.cite(E.FUNNEL_AUDIT, "/ledger_fingerprint"))
        cites.append(ev.cite(E.FUNNEL_AUDIT, "/valid"))
    except KeyError:
        return None, cites, "%s records no ledger_fingerprint or valid flag" % E.FUNNEL_AUDIT
    fp = (led or {}).get("fingerprint")
    if not fp or a.get("ledger_fingerprint") != fp:
        return None, cites, ("%s audits another Step 1 (ledger fingerprint %s..., this ledger's %s...)"
                             % (E.FUNNEL_AUDIT, str(a.get("ledger_fingerprint"))[:12], str(fp)[:12]))
    if a.get("valid") is not True:
        return None, cites, ("%s is invalid (calibration overlap, contract 4.1): nothing may cite it"
                             % E.FUNNEL_AUDIT)
    return a, cites, ""


def d17(ev, th):
    """D17 scarcity_conclusion_unaudited (contract 8.4): (C) a negative
    conclusion exists, (S) at least one of S1-S7 holds, (A) no valid audit of
    this Step 1 exists. Proposes L10 (census) first, L12 and L11 (with L11a,
    the card fetch it reads) for the S2 sources, the devil's-advocate pass
    (OP_DA), and card X11 when S4 or S6 holds. Never L13 (only after D18)."""
    use = _t(th, "D17", "signals")
    led, why = _funnel_ledger(ev)
    if led is None:
        return _silent("D17", "not evaluated: %s" % why)
    conclusion, ccites, cdet = _conclusion(ev, th)
    if not conclusion:  # funnel-mutation: M1
        return _silent("D17", "no negative conclusion: no finished real loop that accepted nothing, no sized-down "
                              "loop and no open negative claim", detail={"conclusion": cdet})
    sig = funnel_signals(ev, th)
    held = [s for s in use if sig[s]["holds"] is True]
    audit, acites, awhy = _valid_audit(ev, led)
    detail = {"conclusion": cdet, "signals": sig, "held": held, "audit": awhy or "valid",
              "ledger": {"derivation": led.get("derivation"), "fingerprint": led.get("fingerprint"),
                         "name_status_version": led.get("name_status_version")}}
    if audit is not None:
        return _silent("D17", "audited: %s is a valid audit of this Step 1 (fingerprint %s...)"
                       % (E.FUNNEL_AUDIT, str(led.get("fingerprint"))[:12]), cites=acites, detail=detail)
    if not held:
        return _silent("D17", "no suspicion signal holds (%s)" % "; ".join(sig[s]["summary"] for s in use),
                       cites=ccites, detail=detail)
    cites = list(ccites)
    for s in held:
        cites += sig[s]["cites"]
    try:
        cites.append(_lcite(ev, "fingerprint"))
    except KeyError:
        pass
    levers = ["L10"]
    s2_sources = [x["source"] for x in (sig["S2"]["detail"].get("sources") or [])] if "S2" in held else []
    if s2_sources:
        levers += ["L12", "L11a", "L11"]
    levers += _sync_op(ev)
    levers.append("OP_DA")
    if "S4" in held or "S6" in held:
        levers.append("X11")
    detail.update(first_verb=_t(th, "D17", "first_verb"), s2_sources=s2_sources,
                  then=[], never=["L13 (only after D18)"])
    gap, gap_why = _kt_gap(ev, led, th)
    if gap:
        detail["needs"] = gap_why
    summary = ("a negative conclusion (%s) rests on unaudited filters (%s; %s) -> L10 %s%s, the devil's-advocate "
               "pass%s" % ("; ".join(f["why"] for f in cdet["forms"]), ", ".join(held), awhy,
                          detail["first_verb"],
                          ", L12 and L11 for %d uninformative source(s)" % len(s2_sources) if s2_sources else "",
                          ", X11" if "X11" in levers else ""))
    return _diag("D17", True, "warn", summary, _dedupe(cites), levers, None, detail)


def _kt_gap(ev, led, th):
    """(gap, why): True when no known-truth set of the class-deciding stages
    (SIGNALS.S4_roles) lies outside the reference domain [q05, 1] (contract 8.8:
    a domain with no shifted truth gets "filter recall cannot be measured",
    not a silent audit). None when the ledger records no reference q05."""
    q05 = _num(((led.get("domain_scores") or {}).get("reference") or {}).get("q05"))
    if q05 is None:
        return None, ""
    sets = []
    for _i, st in _stage_index(led, _t(th, "SIGNALS", "S4_roles")):
        for s in ((st.get("calibration") or {}).get("known_truth_sets") or []):
            sc = (s or {}).get("domain_score")
            if isinstance(sc, list) and len(sc) == 2 and all(_num(x) is not None for x in sc):
                sets.append((s.get("id"), sc[0] >= q05 and sc[1] <= 1.0))
    if any(not inside for _i, inside in sets):
        return False, ""
    return True, ("filter recall cannot be measured outside the reference domain: %s; it needs an out-of-domain "
                  "known-truth set whose labels are independent of every audited source"
                  % ("every one of the %d known-truth set(s) lies in the reference domain [q05 %.4g, 1]"
                     % (len(sets), q05) if sets else "the ledger records no known-truth set"))


def _sync_op(ev):
    """["OP_FUNNEL_SYNC"] when the lab holds funnel files the cluster does not
    (levers.funnel_sync_needed: the ticker's funnel_lab context against the
    cluster's funnel/files.json), else []."""
    from . import levers as LV
    return ["OP_FUNNEL_SYNC"] if LV.funnel_sync_needed(ev) else []


def _dedupe(cites):
    out, have = [], set()
    for c in cites:
        k = json.dumps(c, sort_keys=True)
        if k not in have:
            have.add(k)
            out.append(c)
    return out


def d19(ev, th):
    """D19 filter_recall_unmeasured (contract 8.4): S1, S4 or S6 holds for a
    recoverable stage with no audit; needs no conclusion. Proposes L10, which
    levers.propose ranks before L2 (RULES order; DEC-4)."""
    use = _t(th, "D19", "signals")
    roles = _t(th, "D19", "stage_roles")
    led, why = _funnel_ledger(ev)
    if led is None:
        return _silent("D19", "not evaluated: %s" % why)
    from ..funnel import ledger as FL
    rec = set(FL.recoverable_stages(led))
    audit, _ac, _aw = _valid_audit(ev, led)
    audited = {k for k, v in ((audit or {}).get("stages") or {}).items()
               if isinstance(v, dict) and isinstance(v.get("fn_rate"), dict)}
    stages = [(i, s) for i, s in _stage_index(led, roles)
              if s.get("id") in rec and s.get("audit") is None and s.get("id") not in audited]
    if not stages:
        return _silent("D19", "every recoverable stage of role %s carries an audit (in the ledger, or in the valid "
                              "audit of this Step 1), or none is recoverable" % roles)
    sig = funnel_signals(ev, th)
    held = [s for s in use if sig[s]["holds"] is True]
    detail = {"signals": {s: sig[s] for s in use}, "held": held,
              "stages": [s.get("id") for _i, s in stages]}
    if not held:
        return _silent("D19", "none of %s holds (%s)" % (use, "; ".join(sig[s]["summary"] for s in use)),
                       detail=detail)
    cites = []
    for s in held:
        cites += sig[s]["cites"]
    for i, _s in stages:
        cites += [_lcite(ev, "stages", i, "recoverable"), _lcite(ev, "stages", i, "audit")]
    gap, gap_why = _kt_gap(ev, led, th)
    if gap:
        detail["needs"] = gap_why
    summary = ("filter recall is unmeasured: %s hold for the recoverable, unaudited stage(s) %s -> L10 (ranked "
               "before any real-loop build, DEC-4)%s" % (", ".join(held), ", ".join(detail["stages"]),
                                                         "; %s" % gap_why if gap else ""))
    return _diag("D19", True, "warn", summary, _dedupe(cites), ["L10"] + _sync_op(ev), None, detail)


def _stratum_source(row):
    """The source slug a d18_inputs row is about: its 'source' key, else the
    slug inside its stratum id (a 'c:<slug>|<id>' or 'k:<slug>|<id>|<j>' unit,
    or a 'source=<slug>' key), else None."""
    if isinstance(row.get("source"), str) and row["source"]:
        return row["source"]
    sid = str(row.get("stratum") or "")
    m = re.search(r"(?:^|[/=])[ck]:([^|/]+)\|", sid)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|/)source=([^/]+)", sid)
    return m.group(1) if m else None


def d18(ev, th):
    """D18 filter_false_negatives (contract 8.4): per stratum of the audit's
    d18_inputs, FN lower bound >= fn_lb_min and recoverable lower bound >=
    recover_min_frac_of_verified x the target_check kept count. Levers by
    kind (thresholds D18); a stratum whose prediction is a relative of its
    source taxa is a known confusion, never recovered. A row naming a stage
    the ledger does not mark recoverable, or a guard stage, is refused
    (detail refused_stages; contract 7: guards are never touched), and when
    only such rows pass the bounds D18 escalates and no claim moves. Silent on
    a valid audit of this Step 1: its detail proposes the claim transition to
    tested_survives. An invalid audit (calibration overlap) escalates."""
    led, why = _funnel_ledger(ev)
    a = ev.json(E.FUNNEL_AUDIT)
    if a is None:
        return _silent("D18", "no %s in the evidence" % E.FUNNEL_AUDIT)
    if led is None:
        return _silent("D18", "not evaluated: %s" % why)
    if isinstance(a, dict) and a.get("format") == FUNNEL_AUDIT_FORMAT and a.get("valid") is False:
        cites = [ev.cite(E.FUNNEL_AUDIT, "/valid")]
        try:
            cites.append(ev.cite(E.FUNNEL_AUDIT, "/calibration_overlap"))
        except KeyError:
            pass
        overlap = a.get("calibration_overlap") or []
        return _diag("D18", True, "crit",
                     "calibration_overlap: %s is invalid (%d stratum/judge overlap(s) with calibration material, "
                     "contract 4.1); nothing may cite it -> escalate" % (E.FUNNEL_AUDIT, len(overlap)),
                     cites, ["OP_ESCALATE"], None, {"calibration_overlap": overlap, "valid": False})
    audit, acites, awhy = _valid_audit(ev, led)
    if audit is None:
        return _silent("D18", "not evaluated: %s" % awhy, cites=acites)
    fn_min = _t(th, "D18", "fn_lb_min")
    frac = _t(th, "D18", "recover_min_frac_of_verified")
    by_role = _t(th, "D18", "rejected_policy_by_role")
    unin = _t(th, "D18", "uninformative_policies")
    other_policy = _t(th, "D18", "other_policy")
    tc = _stage_index(led, ["target_check"])
    kept = _int(tc[0][1].get("kept")) if len(tc) == 1 else None
    if kept is None:
        return _silent("D18", "not evaluated: the ledger has no target_check stage with a kept count")
    rmin = frac * kept
    roles = {s.get("id"): s.get("role") for s in led.get("stages") or [] if isinstance(s, dict)}
    maps = ev.json(E.FUNNEL_CLASS_MAPS)
    card_sources = set()
    to_l14 = []
    for k, p in enumerate((maps or {}).get("proposals") or [] if isinstance(maps, dict) else []):
        if not isinstance(p, dict):
            continue
        if p.get("status") == "proposed" and p.get("via") == "card+geometry":
            card_sources.add(p.get("source"))
        if p.get("status") == "to_L14":
            to_l14.append((k, p))
    fire, confusions, refused, cites = [], [], [], list(acites)
    try:
        cites.append(_lcite(ev, "stages", tc[0][0], "kept"))
    except KeyError:
        pass
    from ..funnel import ledger as FL
    recoverable = set(FL.recoverable_stages(led))
    guards = {s.get("id") for s in led.get("stages") or [] if isinstance(s, dict) and s.get("guard") is True}
    for i, row in enumerate(a.get("d18_inputs") or []):
        if not isinstance(row, dict):
            continue
        fl, rl = _num(row.get("fn_lb")), _num(row.get("recoverable_lb"))
        if fl is None or rl is None or fl < fn_min or rl < rmin:
            continue
        if row.get("stage") not in recoverable or row.get("stage") in guards:
            # Contract 7 and 8.4: D18 fires for a recoverable stage only, and
            # guard stages are never touched, whatever the audit says.
            refused.append({"stage": row.get("stage"), "stratum": row.get("stratum"), "kind": row.get("kind"),
                            "why": "stage %s is %s in the ledger: never recovered"
                                   % (row.get("stage"), "a guard" if row.get("stage") in guards
                                      else "not recoverable (recoverable is not true)")})
            for key in ("stage", "fn_lb", "recoverable_lb"):
                cites.append(ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/%d/%s" % (i, key)))
            continue
        rc = [ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/%d/fn_lb" % i),
              ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/%d/recoverable_lb" % i),
              ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/%d/kind" % i)]
        kind = row.get("kind")
        rec = {"stage": row.get("stage"), "stratum": row.get("stratum"), "kind": kind, "fn_lb": fl,
               "recoverable_lb": rl, "source": _stratum_source(row)}
        if kind == "other_predicted_target" and row.get("relative_of_prediction") is not False:
            rec["why"] = ("the source taxa %s include a relative of the prediction (or it is unknown): a known "
                          "confusion, never relabelled (the sibling guard)" % (row.get("source_taxa") or []))
            try:
                rc.append(ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/%d/relative_of_prediction" % i))
            except KeyError:
                pass
            confusions.append(rec)
            cites += rc
            continue
        if kind == "target_rejected":
            rec["policy"] = by_role.get(roles.get(row.get("stage")))
            rec["card"] = "X11"
        elif kind == "uninformative_label_space":
            # Contract 8.4: a card map is applied only "when the card and the purity check
            # agree" (H3a supported); otherwise the stratum goes to the judges (R-J), whose
            # own gates (H2a, H7) recover.py applies. Without this, D18 would propose R-C
            # for a map recover.py must refuse.
            h3a = ((a.get("hypotheses") or {}).get("H3a") or {}).get("verdict")
            if rec["source"] in card_sources and h3a == "supported":
                rec["policy"] = unin["card_map"]
            else:
                rec["policy"] = unin["judges"]
                if rec["source"] in card_sources:
                    rec["why_not_card_map"] = "H3a is %s" % h3a
                    try:
                        rc.append(ev.cite(E.FUNNEL_AUDIT, "/hypotheses/H3a/verdict"))
                    except KeyError:
                        pass
        elif kind == "other_predicted_target":
            rec["policy"] = other_policy
            rc.append(ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/%d/relative_of_prediction" % i))
        if not rec.get("policy"):
            rec["why"] = "kind %r of stage %s has no recovery policy" % (kind, row.get("stage"))
            confusions.append(rec)
            continue
        fire.append(rec)
        cites += rc
    for k, p in to_l14:
        cites.append(ev.cite(E.FUNNEL_CLASS_MAPS, "/proposals/%d/status" % k))
    policies = [p for p in RECOVER_POLICY_ORDER if any(r["policy"] == p for r in fire)]
    detail = {"strata": fire, "known_confusions": confusions, "refused_stages": refused,
              "policy": ",".join(policies) or None,
              "to_l14": [{"source": p.get("source"), "src_id": p.get("src_id"), "src_name": p.get("src_name"),
                          "map_to": p.get("map_to"), "reason": p.get("reason")} for _k, p in to_l14],
              "recover_min": rmin, "fn_lb_min": fn_min, "target_check_kept": kept}
    if not fire and refused:
        # An audit that asks to recover a guard or non-recoverable stage is not
        # evidence the filters were right: no claim moves, a person looks.
        summary = ("the valid audit puts %d stratum(s) of guard or non-recoverable stage(s) above D18's bounds "
                   "(%s): never recovered, and no claim moves on this audit -> escalate"
                   % (len(refused), ", ".join("%s %s" % (r["stage"], r["stratum"]) for r in refused)))
        detail["claim_transitions"] = []
        return _diag("D18", True, "warn", summary, _dedupe(cites), ["OP_ESCALATE"] + (["L14"] if to_l14 else []),
                     None, detail)
    if not fire:
        trans = []
        for i, c in _open_negative_claims(ev, th):
            if c.get("status") == "challenged":
                trans.append({"claim_id": c.get("id"), "to": "tested_survives", "actor_kind": "autopilot",
                              "audit_sha256": (ev.provenance.get(E.FUNNEL_AUDIT) or {}).get("sha256"),
                              "ledger_fingerprint": led.get("fingerprint")})
        detail["claim_transitions"] = trans
        levers = ["L14"] if to_l14 else []
        summary = ("no stratum of the valid audit has FN lb >= %g and recoverable lb >= %.1f (%g x %d verified)%s%s"
                   % (fn_min, rmin, frac, kept,
                      "; the claim(s) %s may move to tested_survives" % ", ".join(t["claim_id"] for t in trans)
                      if trans else "",
                      "; %d class map(s) to the verify queue (L14)" % len(to_l14) if to_l14 else ""))
        if to_l14:
            return _diag("D18", True, "info", summary, _dedupe(cites), levers, None, detail)
        return _silent("D18", summary, cites=_dedupe(cites), detail=detail)
    levers = ["L13"] + (["L14"] if to_l14 else []) + _sync_op(ev) + (["X11"] if any(r.get("card") for r in fire)
                                                                    else [])
    rec = ev.json(E.FUNNEL_RECOVERY)
    if isinstance(rec, dict) and rec.get("status") == "complete":
        levers.insert(1, "L2")
        cites.append(ev.cite(E.FUNNEL_RECOVERY, "/status"))
    else:
        detail["then"] = ["L2 with --increment-sources recovered"]
    summary = ("the audit finds filter false negatives in %d stratum(s) (%s) -> L13 --policy %s%s%s"
               % (len(fire), "; ".join("%s %s fn_lb %.2f, recoverable lb %.0f -> %s"
                                        % (r["stage"], r["stratum"], r["fn_lb"], r["recoverable_lb"], r["policy"])
                                        for r in fire),
                  detail["policy"],
                  "; %d known confusion(s) guarded" % len(confusions) if confusions else "",
                  "; %d class map(s) to L14" % len(to_l14) if to_l14 else ""))
    if refused:
        summary += ("; %d stratum(s) of guard or non-recoverable stage(s) refused (%s)"
                    % (len(refused), ", ".join(r["stage"] for r in refused)))
    return _diag("D18", True, "warn", summary, _dedupe(cites), levers, None, detail)


# The order levers.propose ranks proposals in (module doc): D15 next to D1,
# which it takes precedence over; D16 next to D4, which it gates; the funnel
# diagnoses D17, D18 and D19 after D2 (runner 5.5.2), so D19's L10 ranks ahead
# of D4's real-loop L2 (DEC-4).
RULES = (("D1", d1), ("D15", d15), ("D2", d2), ("D17", d17), ("D18", d18), ("D19", d19), ("D2b", d2b),
         ("D3", d3), ("D3b", d3b), ("D4", d4),
         ("D16", d16), ("D5", d5), ("D6", d6), ("D7", d7), ("D8", d8), ("D9", d9), ("D10", d10), ("D11", d11),
         ("D12", d12), ("D13", d13), ("D14", d14), ("DREF", dref))


def detect(ev, thresholds=None, base=None, only=None):
    """Every diagnosis (fired or not) on the evidence, in RULES order.
    thresholds: a partial override {D: {key: value}} (tests); base: the full
    table (default thresholds.json); only: an iterable of ids (e.g. HEALTH)."""
    th = _merge(base if base is not None else load_thresholds(), thresholds)
    out = []
    for did, fn in RULES:
        if only is not None and did not in only:
            continue
        try:
            out.append(fn(ev, th))
        except _Missing as e:
            out.append(_unknown(did, "threshold %s is not declared in thresholds.json, so this rule cannot run"
                                % e.args[0], ev.exp))
        except Exception as e:                        # evidence is untrusted input
            out.append(_unknown(did, "the rule raised %s (%s)" % (type(e).__name__, str(e)[:200]), ev.exp))
    return out


def fired(diags):
    return [d for d in diags if d["fired"]]


def by_id(diags):
    return {d["id"]: d for d in diags}


# ------------------------------------------------------ prospective D4 (R4b)
def _atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def prospective_name(exp, version):
    """The file name of exp's prospective D4 record under rules `version`:
    prospective_d4_<exp>__<version>.json (one per pilot and rules version)."""
    if not re.fullmatch(r"[0-9a-f]{%d}" % RULES_VERSION_HEX, str(version or "")):
        raise ValueError("a rules version is %d hex characters, not %r" % (RULES_VERSION_HEX, version))
    return "%s%s__%s.json" % (PROSPECTIVE_PREFIX, exp, version)


def prospective_records(replay_dir, exp):
    """[(path, rules version or None, record or None)] of every prospective D4
    record of exp in replay_dir, by name: the name without a version
    (prospective_d4_<exp>.json, written before records were versioned) and
    each prospective_d4_<exp>__<version>.json. A record whose own 'exp' is
    another experiment is left out; an unreadable one is listed as None."""
    out = []
    d = Path(replay_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("%s*.json" % PROSPECTIVE_PREFIX)):
        m = _PROSPECTIVE_RE.match(p.name)
        if not m or m.group("exp") != exp:
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        if isinstance(rec, dict) and rec.get("exp") not in (None, exp):
            continue
        out.append((p, m.group("ver"), rec if isinstance(rec, dict) else None))
    return out


def current_prospective(replay_dir, exp, version=None):
    """(record or None, path, why) of exp's prospective D4 record under the
    current rules version (or `version`). why is "" only for a READY record
    of exp written under that version; otherwise it says what is missing. A
    record written under another rules version never counts, whatever it
    says: every L2/L6 guard reads this function."""
    version = version or rules_version()
    path = Path(replay_dir) / prospective_name(exp, version)
    if not path.is_file():
        older = [p.name for p, v, _r in prospective_records(replay_dir, exp) if v != version]
        return None, path, ("D4's decision on %s is not frozen in a prospective record under the current rules "
                            "version %s (%s)%s" % (exp, version, path.name,
                                                   "; the record(s) %s were written under other rules and do not "
                                                   "count" % ", ".join(older) if older else ""))
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, path, "the prospective record %s is unreadable (%s)" % (path.name, type(e).__name__)
    if not isinstance(rec, dict) or rec.get("exp") != exp or rec.get("rules_version") != version:
        return (rec if isinstance(rec, dict) else None), path, (
            "%s does not hold D4's decision on %s under rules version %s" % (path.name, exp, version))
    if rec.get("ready") is not True:
        return rec, path, ("D4's decision on %s is frozen under rules version %s as not ready (%s%s)"
                           % (exp, version, rec.get("outcome"),
                              ", blocked by %s" % rec["blocked_by"] if rec.get("blocked_by") else ""))
    return rec, path, ""


def prospective_d4(report, out_path=None, defn=None, now=None, version=None):
    """D4 on a pilot's report.json (a path or a dict), written to out_path
    (default model.REPLAY_DIR/prospective_4.json) before any real-loop build:
    the ticker commits the file, and a person's --replay-mode / --recipes
    choice is compared with it afterwards (compare_prospective). The file is
    written once: the same decision again is a no-op, a different one
    refuses (ValueError). Returns the record.

    The record carries the rules version it was decided under (`version`,
    default rules_version()) and each rules file's sha256; the ticker names
    the file prospective_name(exp, version), so a change of rules writes a
    new record beside the old one and never rewrites it. A record already at
    out_path written under another rules version refuses (ValueError).

    A pilot decided on protocol v2 whose report.json v2_check is not final
    (provisional, or missing) has no decision yet: nothing is written, and
    the record comes back with 'pending' (the check's state) and 'written'
    false, so the caller tries again on a later report."""
    if isinstance(report, (str, Path)):
        raw = E._read_file(report)
        rep = json.loads(raw.decode("utf-8"))
    else:
        rep = report
        raw = json.dumps(report, sort_keys=True).encode("utf-8")
    exp = rep.get("exp")
    texts = {"%s/report.json" % exp: raw}
    if defn is not None:
        texts["%s/exp.json" % exp] = json.dumps(defn).encode("utf-8")
    ev = E.from_texts(texts, exp)
    th = _merge(load_thresholds(), None)
    rules = rules_digest()
    if version is not None and version != rules["version"]:
        raise ValueError("prospective_d4 was asked for rules version %s, but the rules loaded here are %s"
                         % (version, rules["version"]))
    d = d4(ev, th, exp=exp)
    det = d.get("detail") or {}
    v2m, v2c, _v2cite, v2final = _v2_state(ev, th, exp)
    pending = ("not final" if v2c else "missing") if v2m == _t(th, "D16", "v2_mode") and v2final is not True \
        else None
    ready = d["fired"] and d["name"] == NAMES["D4"] and not det.get("blocked_by")
    rec = {"format": PROSPECTIVE_FORMAT, "case": "R4b", "exp": exp,
           "report_sha256": hashlib.sha256(raw).hexdigest(),
           "thresholds_sha256": hashlib.sha256(E._read_file(THRESHOLDS_FILE)).hexdigest(),
           "rules_version": rules["version"], "rules_files": rules["files"],
           "rule": "docs/INC_AUTOPILOT.md (b) D4: best agreement rate, ties by fewer non-ACCEPT on truth-helps "
                   "steps, then |final dev(chain) - dev(T_final)|, then chain GPU-hours; ready at >= 5/7, and "
                   "not ready while D1 blocks it on the pilot (D1 fires and D4's threshold is not met or the "
                   "chain D4 selects did not ACCEPT a clean truth-helps step: the R4 review of 2026-09-27, made "
                   "after pilot_v3's result), while D15 blocks D1 there, or while protocol v2 is refuted on the "
                   "gate the pilot was decided with (D16); a pilot decided on protocol v2 is recorded only once "
                   "its v2_check is final",
           "outcome": d["name"] if d["fired"] else "silent", "ready": ready,
           "blocked_by": (det.get("blocked_by") or {}).get("id"),
           "replay_mode": det.get("replay_mode") if ready else None,
           "recipes": det.get("recipes") if ready else None,
           # The gate the real loop is built with (L2 carries the pilot's flips
           # mode) and the pilot's protocol v2 check, when it was decided on v2.
           "gate_flips_mode": (det.get("gate") or {}).get("flips_mode") if ready else None,
           "v2_check": (det.get("gate") or {}).get("v2_check"),
           "d1": det.get("d1"),
           "levers": d["levers"], "diagnosis": d,
           "decided_utc": now or M.utc_now()}
    if pending:
        rec.update(pending=pending, written=False)
        return rec
    path = Path(out_path or PROSPECTIVE_FILE)
    if path.exists():
        old = json.loads(E._read_file(path).decode("utf-8"))
        if old.get("rules_version") not in (None, rec["rules_version"]):
            raise ValueError("%s holds a prospective decision for %s written under rules version %s, not %s; "
                             "a record is written once per rules version, under its own name (prospective_name)"
                             % (path, old.get("exp"), old.get("rules_version"), rec["rules_version"]))
        same = all(old.get(k) == rec[k] for k in ("exp", "report_sha256", "outcome", "ready", "replay_mode",
                                                   "recipes", "blocked_by", "gate_flips_mode"))
        if same:
            return old
        raise ValueError("%s already holds a prospective decision for %s (%s, %s); it is written once"
                         % (path, old.get("exp"), old.get("replay_mode"), old.get("recipes")))
    _atomic_write(path, rec)
    return rec


def compare_prospective(record, replay_mode, recipes, gate_flips_mode=None):
    """The prospective record against the person's realloop choice. With
    `gate_flips_mode` (the loop's exp.json gate.flips_mode) the gate is
    compared too; a record written before the gate was carried (no
    gate_flips_mode) is not compared on it."""
    if isinstance(record, (str, Path)):
        record = json.loads(E._read_file(record).decode("utf-8"))
    manual = [r for r in (recipes.split(",") if isinstance(recipes, str) else list(recipes)) if r]
    out = {"exp": record.get("exp"), "rule": {"replay_mode": record.get("replay_mode"),
                                              "recipes": record.get("recipes")},
           "manual": {"replay_mode": replay_mode, "recipes": manual},
           "replay_mode_match": record.get("replay_mode") == replay_mode,
           "recipes_match": sorted(record.get("recipes") or []) == sorted(manual)}
    out["match"] = out["replay_mode_match"] and out["recipes_match"]
    if gate_flips_mode is not None and record.get("gate_flips_mode") is not None:
        out["rule"]["gate_flips_mode"] = record.get("gate_flips_mode")
        out["manual"]["gate_flips_mode"] = gate_flips_mode
        out["gate_match"] = record.get("gate_flips_mode") == gate_flips_mode
        out["match"] = out["match"] and out["gate_match"]
    return out


def _builders_flips_default():
    """The builders' default --gate-flips-mode (levers.json protocol), or None
    when levers.json cannot be read (the gate is then not compared)."""
    try:
        from . import levers as LV
        return LV.protocol("gate_flips_mode_default")
    except Exception:
        return None


def prospective_guard(replay_dir, pilot, params=None, sha256=None, version=None):
    """(why, differs): the one check every path that builds a real loop from
    a pilot's D4 decision applies (the ticker's own L2/L6 before it is
    proposed and again before it runs, an approved item it adopts, the INC
    page's build and execute-approved routes). why is "" when the build may
    run: D4's decision on `pilot` is frozen in a READY record under the
    current rules version (or `version`; current_prospective), the record's
    bytes are the ones the ticker wrote (when `sha256`, the ticker's
    sha256 of it, is given), and the build follows it (when `params`, the
    inc_build_realloop params, are given): its replay_mode, recipes and gate
    flips mode (params' gate_flips_mode, else the builders' default) are the
    record's (compare_prospective). differs is True only for the last case:
    a READY record of the current rules that the build does not follow."""
    rec, path, why = current_prospective(replay_dir, pilot, version)
    if why:
        return why, False
    if sha256 is not None:
        try:
            got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as e:
            return "the prospective record %s cannot be read (%s)" % (Path(path).name, type(e).__name__), False
        if got != sha256:
            return ("the prospective record %s is not the one the ticker wrote (sha256 %s..., the ticker's %s...)"
                    % (Path(path).name, got[:12], str(sha256)[:12])), False
    if params is None:
        return "", False
    params = params if isinstance(params, dict) else {}
    recipes = params.get("recipes")
    cmp = compare_prospective(rec, params.get("replay_mode"),
                              recipes if isinstance(recipes, (str, list, tuple)) else [],
                              params.get("gate_flips_mode") or _builders_flips_default())
    if cmp["match"]:
        return "", False
    diffs = []
    for key, ok in (("replay_mode", cmp["replay_mode_match"]), ("recipes", cmp["recipes_match"]),
                    ("gate_flips_mode", cmp.get("gate_match", True))):
        if not ok:
            diffs.append("%s %s, the record's %s" % (key, json.dumps(cmp["manual"].get(key)),
                                                    json.dumps(cmp["rule"].get(key))))
    return ("the build does not follow D4's decision on %s frozen in %s (%s)"
            % (pilot, Path(path).name, "; ".join(diffs))), True
