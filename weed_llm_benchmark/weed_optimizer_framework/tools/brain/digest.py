#!/usr/bin/env python3
"""Campaign-scale digest for the tier-2 planner (v3.27.0, WP6).

Why this exists
---------------
`evidence.py` compresses ONE step so a cheap or mid-tier reviewer can see it in
one call. This module compresses the WHOLE campaign — every round, every fired
signal, every correction and whether it held, every SU spent — so the rarely
consulted top-tier planner gets the full compressed history ONCE instead of a
truncated tail. The difference matters for the same reason `signals.py` exists
at all: on 2026-08-29 the fact that would have stopped two burnt 12 h jobs
(pool growth of 37% since the previous train) was sitting in an artifact the
whole time, and nothing read it. A planner shown only the last few rounds is in
exactly that position at campaign scale — it cannot see that `pool_growth` has
now fired in three consecutive rounds and that the correction after the first
firing only bought one round of headroom, because that pattern lives in the
rounds a truncated tail would have cut. Compressing is not summarising away
that pattern; it is the only way to fit the whole campaign under one model
call at all, so this module trims on a stated, inspectable priority order and
never on a middle cut nobody can see happened.

What `build()` is given, and a naming note
-------------------------------------------
`build(domain, rounds, signals_history, corrections_history, ledger, config,
num_ctx, omit_levers=False)`. Two of these names collide with vocabulary used
elsewhere in this package and are deliberately NOT the same thing here:

* `rounds` — the campaign's round-doc history (`db.py`'s `domain_rounds`
  collection, `get_rounds()`'s shape: `round_num`, `status`, `actor`, `steps`
  with per-step `status`/`actor`/`params`/`attempts[]`, `metrics`). This is the
  FULL history a top-tier call needs, not the last-5-rounds slice
  `evidence.py` stages into its own `sections["ledger"]` for a single live
  step — a campaign digest exists precisely because that slice is too short.
* `ledger` — the resource ledger: rows shaped like WP5's `su_ledger`
  collection (`job`, `domain`, `round`, `step`, `actor`, `gpu_count`,
  `gpu_type`, `elapsed_s`, `su`, `ts`). It is kept as a separate parameter from
  `rounds` because SU accounting and round provenance are reconciled against
  different sources of truth (`sacct` vs the round doc) and conflating them
  would hide a reconciliation gap instead of reporting one.

`signals_history` is a list of signal firings, each shaped like one row from
`signals.detect()` (`signal`, `severity`, `value`, `reason`) stamped with
`round`/`step`/`job_id` by whoever logged the review, OR grouped as
`{"round", "step", "signals": [...]}`. `corrections_history` follows WP5's
`brain_corrections` shape (`seq`, `author`, `kind`, `target{key,old,new}`,
`scope{from_round,until_round}`, `reason`, `ts`), plus an OPTIONAL `signal`
field this module reads (but never invents) to check whether the signal a
correction targeted recurred afterwards; a correction record that does not
carry it is reported as not-stated, never guessed at from its `target.key`.

Every one of the four historical inputs is untrusted the same way a bundle is
untrusted in `signals.py`: a row that is not a dict, or a raw JSON string that
does not parse (the shape a half-written last line of a JSONL mirror leaves
behind, same as `trace.py` faces on append), is skipped and counted rather
than raised on. Counts of skipped rows are reported in the digest text itself
(the "input hygiene" line), never silently absorbed.

Sections, priorities and the trim discipline
---------------------------------------------
Seven sections carry the required content (identity, trajectory, recipe
changes, incidents, corrections, resources, state); an eighth (`levers`) is
the pre-registered menu of what a plan may propose changing, and is left out
entirely — not trimmed, not stubbed — when `omit_levers=True`. That flag
exists for one reason: `TIERED_SUPERVISION_PLAN.md` states it as "the digest
given to any planner under evaluation omits the sealed lever table", because
an evaluation that hands a planner the menu it is being scored on measures
answer-key lookup, not planning. The production digest always keeps it.

Priorities and per-section token budgets are DATA, in `digest_budget.json`
next to this file, each with a stated reason — the same discipline
`signals.py` applies to `thresholds.json`: no compiled fallback, so a section
the file does not declare makes `build()` refuse rather than guess a budget
for it, and changing a priority is a diffable data edit, not a silent
recompile. Two sections (`identity`, `state`) are marked mandatory and are
never dropped whole, only windowed; the rest are dropped ENTIRELY, lowest
priority first, only after every section has already been windowed down to
its own configured budget. Windowing a section keeps its earliest and its
most recent rows and states, in the text and in `omitted`, exactly how many
rows were cut from the middle and how many tokens that saved — "the campaign
had a quiet middle" is a fact a planner needs, and a digest that cuts the
middle without saying so is worse than one that says it did.

If the two mandatory sections alone do not fit `num_ctx` even after the six
optional sections are dropped, `build()` refuses: it returns the same five
keys with a short, structured `text` naming the offending section and the
`refused`/`reason` fields set, exactly as `evidence.py`'s `over_num_ctx`
refusal is a returned, inspectable state rather than a raised exception. The
returned `tokens_est` is asserted below `num_ctx` before every return,
refusal included — a digest module that could itself overflow the window it
exists to protect would be the exact defect class this whole layer is for.

Determinism
-----------
Same inputs -> byte-identical `text` and `sha256`. No wall-clock read happens
inside `build()`; every timestamp printed comes from the data it was given.
This is required so a planner call can be replayed against the same digest
months later, the same reason `corpus.py`'s exports are deterministic.

Pure stdlib, no network, no Mongo import: like `evidence.py`, this module
takes data the caller already fetched and never queries a database itself, so
it imports inside a SLURM job body as freely as on the always-on server.

CLI:
    python3 -m weed_optimizer_framework.tools.brain.digest build \\
        --domain weed [--num-ctx 32768] [--omit-levers] \\
        [--rounds-file F] [--signals-file F] [--corrections-file F] \\
        [--ledger-file F] [--config-file F] [--text | --out FILE]
    python3 -m weed_optimizer_framework.tools.brain.digest sections
    python3 -m weed_optimizer_framework.tools.brain.digest budget
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import sys

try:                                     # package import (the normal path)
    from . import corpus
except ImportError:                      # direct execution from this directory
    import corpus                        # type: ignore

TOOL_VERSION = "wp6-digest/1"

# Narrative order the text is rendered in. NOT the trim order — trim priority
# is data in digest_budget.json, on purpose, so a priority change is a diff
# there rather than a reshuffle of this tuple.
DISPLAY_ORDER = ("identity", "trajectory", "recipe_changes", "incidents",
                 "corrections", "resources", "state", "levers")
SECTION_NAMES = DISPLAY_ORDER

TITLES = {
    "identity": "CAMPAIGN IDENTITY AND GOAL",
    "trajectory": "METRIC TRAJECTORY BY ROUND",
    "recipe_changes": "RECIPE AND CONFIG CHANGES BY ROUND",
    "incidents": "INCIDENTS (SIGNALS THAT FIRED)",
    "corrections": "CORRECTIONS APPLIED",
    "resources": "RESOURCE SPEND AND REMAINING ENVELOPE",
    "state": "CURRENT STATE AND OPEN QUESTION",
    "levers": "PRE-REGISTERED LEVERS (WHAT CAN BE CHANGED)",
}

_REQUIRED_SECTION_KEYS = ("priority", "mandatory", "max_tokens", "why")

_BUDGET_FILE = pathlib.Path(__file__).resolve().parent / "digest_budget.json"
_BUDGET_CACHE = {"key": None, "data": None}


# --- budget file (data, not code; see the module docstring) ---------------
def budget_path():
    return pathlib.Path(os.environ.get("BRAIN_DIGEST_BUDGET") or _BUDGET_FILE)


def load_budget():
    """Effective section priorities/budgets, cached on (path, mtime, size).

    Returns `{"sections": {name: {priority, mandatory, max_tokens, why}},
    "meta": {...}, "error": str or None}`. Mirrors `signals.py`'s
    `_file_block()`: a missing, malformed or incomplete file is a reported
    error, never a compiled fallback, because two copies of a pre-registered
    budget drift and a drifted one silently rescores what a planner is shown.
    """
    path = budget_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError as exc:
        data = {"sections": {}, "meta": {},
                "error": "digest_budget.json unreadable at %s (%s)"
                        % (path, type(exc).__name__)}
        _BUDGET_CACHE.update(key=None, data=data)
        return data
    if _BUDGET_CACHE["key"] == key:
        return _BUDGET_CACHE["data"]
    data = {"sections": {}, "meta": {}, "error": None}
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        secs = obj.get("sections") if isinstance(obj, dict) else None
        meta = obj.get("meta") if isinstance(obj, dict) else None
        if not isinstance(secs, dict):
            data["error"] = "digest_budget.json carries no 'sections' object"
        else:
            problems, clean = [], {}
            for name in sorted(secs):
                spec = secs[name]
                if not isinstance(spec, dict):
                    problems.append("section %s is not an object" % name)
                    continue
                missing = [k for k in _REQUIRED_SECTION_KEYS if k not in spec]
                if missing:
                    problems.append("section %s is missing %s" % (name, missing))
                    continue
                clean[name] = {"priority": spec["priority"],
                               "mandatory": bool(spec["mandatory"]),
                               "max_tokens": spec["max_tokens"],
                               "why": str(spec["why"])}
            data["sections"] = clean
            if problems:
                data["error"] = "; ".join(problems)
        data["meta"] = meta if isinstance(meta, dict) else {}
    except Exception as exc:
        data["error"] = "digest_budget.json unreadable: %s" % type(exc).__name__
    _BUDGET_CACHE.update(key=key, data=data)
    return data


# --- small utilities (this module keeps its own copies; see signals.py and
# evidence.py, which each do the same rather than sharing private helpers) --
def _num(value, default=None):
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=None):
    v = _num(value)
    return default if v is None else int(v)


def _text(value, default=""):
    return value if isinstance(value, str) else default


def _get(holder, *names):
    if not isinstance(holder, dict):
        return None
    for name in names:
        if name in holder and holder[name] is not None:
            return holder[name]
    return None


def _nested(holder, *keys):
    cur = holder
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _tokens(text):
    """ceil(chars / 4) — the same 4-chars-per-token rule `corpus.token_estimate`
    uses, applied directly to the rendered text rather than to a JSON section
    object, since a digest's payload to the model IS this text."""
    return (len(text) + 3) // 4


def _coerce_dict(row):
    """A dict, or a dict parsed from a raw (possibly torn) JSON string, or None.

    Every historical input to `build()` is, in real use, read back from a
    JSONL/JSON mirror staged on /ocean or from Mongo; either can hand this
    module a half-written last line from a writer killed mid-append —
    `trace.py`'s own `append()`/`read()` face exactly this and skip it rather
    than raise. A caller that already parsed its input hands over dicts, which
    pass through unchanged.
    """
    if isinstance(row, dict):
        return row
    if isinstance(row, str):
        try:
            parsed = json.loads(row)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


# --- normalising the four historical inputs --------------------------------
def _sorted_rounds(rounds):
    """(round docs oldest-first, skipped_count). Never assumes an ordering."""
    out, skipped = [], 0
    for raw in (rounds or []):
        d = _coerce_dict(raw)
        if d is None:
            skipped += 1
            continue
        out.append(d)
    out.sort(key=lambda d: _num(_get(d, "round_num", "round"), 0.0))
    return out, skipped


def _sig_sort_key(s):
    rnd = s.get("round")
    return (0 if rnd is not None else 1, rnd if rnd is not None else 0,
            s.get("step") or "", s.get("signal") or "")


def _normalize_signals(signals_history):
    """[{"round","step","job_id","signal","severity","value","reason"}], skipped.

    Accepts either a flat one-firing-per-row shape (matching `signals.py`'s own
    signal dict, stamped with round/step/job_id) or a grouped
    `{"round","step","signals":[...]}` shape; anything else is a reason to
    skip and count, never a shape to guess at.
    """
    out, skipped = [], 0
    for raw in (signals_history or []):
        row = _coerce_dict(raw)
        if row is None:
            skipped += 1
            continue
        if isinstance(row.get("signals"), list):
            rnd = _int(_get(row, "round", "round_num"))
            step = _text(row.get("step"))
            job = row.get("job_id")
            for raw_sig in row["signals"]:
                sig = _coerce_dict(raw_sig)
                if sig is None or not sig.get("signal"):
                    skipped += 1
                    continue
                out.append({"round": _int(sig.get("round"), rnd),
                           "step": _text(sig.get("step")) or step,
                           "job_id": sig.get("job_id") or job,
                           "signal": sig.get("signal"),
                           "severity": sig.get("severity"),
                           "value": sig.get("value"),
                           "reason": sig.get("reason")})
            continue
        if not row.get("signal"):
            skipped += 1
            continue
        out.append({"round": _int(_get(row, "round", "round_num")),
                   "step": _text(row.get("step")), "job_id": row.get("job_id"),
                   "signal": row.get("signal"), "severity": row.get("severity"),
                   "value": row.get("value"), "reason": row.get("reason")})
    out.sort(key=_sig_sort_key)
    return out, skipped


def _normalize_corrections(corrections_history):
    """[correction dict, ...], skipped. Sorted by (from_round, seq)."""
    out, skipped = [], 0
    for raw in (corrections_history or []):
        row = _coerce_dict(raw)
        if row is None:
            skipped += 1
            continue
        out.append(row)

    def key(c):
        frm = _int(_nested(c, "scope", "from_round"))
        return (0 if frm is not None else 1, frm or 0, _int(c.get("seq")) or 0)

    out.sort(key=key)
    return out, skipped


def _normalize_ledger(ledger):
    """[ledger row dict, ...], skipped. See `_coerce_dict` for why a raw
    string entry is attempted as JSON rather than rejected outright."""
    out, skipped = [], 0
    for raw in (ledger or []):
        row = _coerce_dict(raw)
        if row is None:
            skipped += 1
            continue
        out.append(row)
    return out, skipped


# --- round-doc readers -------------------------------------------------------
def _step_entry(round_doc, step):
    steps = round_doc.get("steps") if isinstance(round_doc, dict) else None
    entry = steps.get(step) if isinstance(steps, dict) else None
    return entry if isinstance(entry, dict) else {}


def _recipe_of(round_doc):
    """The sealed-noise-floor recipe id for a round's train step, or "".

    Mirrors `signals.py`'s `_recipe()`: an explicit `recipe`/`recipe_id` wins,
    else `"merged_%s" % tier` — the same fallback, because the noise_floor
    block is keyed the same way there.
    """
    params = _step_entry(round_doc, "train").get("params")
    params = params if isinstance(params, dict) else {}
    recipe = _text(params.get("recipe") or params.get("recipe_id"))
    if recipe:
        return recipe
    tier = _text(params.get("tier"))
    return ("merged_%s" % tier) if tier else ""


def _target_metric_key(config, rounds):
    """(key, reason) — which metrics-dict key is the campaign's target metric.

    A declared `target_metric` wins. Otherwise this looks for the one metric
    key that is actually on the ledger; with more than one candidate and no
    declaration, no key is guessed — `signals.py`'s `_round_metrics` makes the
    same refusal for the same reason: a wrong guess turns someone else's
    metric into "the" trajectory.
    """
    declared = _text((config or {}).get("target_metric"))
    if declared:
        return declared, "declared in the domain config"
    keys = set()
    for d in rounds:
        m = d.get("metrics")
        if isinstance(m, dict):
            keys.update(k for k in m if isinstance(k, str))
    if not keys:
        return "", "no target_metric in the domain config and no round carries a metrics block"
    preferred = sorted(k for k in keys
                       if re.sub(r"[^a-z0-9]", "", k.lower()) == "map5095")
    if preferred:
        return preferred[0], ("no target_metric in the domain config; the only "
                              "map50-95-shaped metric key on the ledger was used")
    if len(keys) == 1:
        return next(iter(keys)), ("no target_metric in the domain config; it is "
                                  "the only metric key on the ledger")
    return "", ("no target_metric in the domain config and %d candidate metric "
               "keys are on the ledger (%s); none is reported without knowing "
               "which one the campaign is optimizing"
               % (len(keys), ", ".join(sorted(keys))))


def _metric_of(round_doc, key):
    """(value, kind, n, std, reason) for one round's target metric.

    kind is "mean_std" (n>=2, a real mean and std), "single" (n==1 — labelled
    as such and never rendered as a mean), or "missing". A `metrics[key]` that
    is a plain number is a single-seed reading by construction: the automated
    per-round loop submits one train job per round unless a sealed multi-seed
    evaluation staged a `{"mean","std","n"}` block instead.
    """
    if not key:
        return None, "missing", None, None, "no target metric key was resolved"
    metrics = round_doc.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    raw = metrics.get(key)
    if raw is None:
        return None, "missing", None, None, "no %s on this round's metrics" % key
    if isinstance(raw, dict):
        mean = _num(raw.get("mean"))
        n = _int(raw.get("n") if raw.get("n") is not None else raw.get("n_seeds"))
        std = _num(raw.get("std"))
        if mean is None:
            return None, "missing", None, None, ("the %s block carries no numeric "
                                                  "mean" % key)
        if n and n >= 2:
            return mean, "mean_std", n, std, ""
        return mean, "single", n or 1, None, ("the %s block reports %s seed(s); a "
                                              "single-seed number is not a mean"
                                              % (key, n or 1))
    val = _num(raw)
    if val is None:
        return None, "missing", None, None, "%s=%r is not numeric" % (key, raw)
    return val, "single", 1, None, ""


def _noise_floor(config, recipe):
    floors = (config or {}).get("noise_floor")
    if not isinstance(floors, dict) or not recipe:
        return None
    return _num(floors.get(recipe))


def _compare(prev_val, cur_val, floor):
    """(verdict, delta) where verdict is "within_noise"/"improved"/"regressed",
    or None when either value or the floor is unavailable. "less than the
    floor" is strict, matching the letter of the campaign's own rule."""
    if prev_val is None or cur_val is None or floor is None:
        return None
    delta = cur_val - prev_val
    if abs(delta) < floor:
        return "within_noise", delta
    return ("improved" if delta > 0 else "regressed"), delta


def _recurrence(sig_rows, signal, from_round):
    """The earliest round after `from_round` at which `signal` fired again at
    warn/crit, or None if the given signal history shows no such firing."""
    cands = []
    for s in sig_rows:
        if s.get("signal") != signal or s.get("severity") not in ("warn", "crit"):
            continue
        rnd = s.get("round")
        if rnd is None or rnd <= from_round:
            continue
        cands.append(rnd)
    return min(cands) if cands else None


def _correction_for_change(corr_rows, round_num, key):
    """The correction, if any, whose target key matches and whose scope took
    effect exactly on `round_num` — an exact match only, never the nearest
    one, so a change is never attributed to a correction that did not name it."""
    for c in corr_rows:
        if (_text(_nested(c, "target", "key")) == key
                and _int(_nested(c, "scope", "from_round")) == round_num):
            return c
    return None


def _correction_for_signal(corr_rows, signal, round_num):
    """The earliest correction on record that names `signal` and took effect
    at or after `round_num`, or None."""
    cands = [c for c in corr_rows if c.get("signal") == signal]
    if round_num is not None:
        at_or_after = [c for c in cands
                       if (_int(_nested(c, "scope", "from_round")) or 0) >= round_num]
        if at_or_after:
            return at_or_after[0]
    return cands[0] if cands else None


def _step_outcome(round_doc, step):
    entry = _step_entry(round_doc, step) if round_doc else {}
    if not entry:
        return "no ledger entry for step %s on this round" % step
    status = entry.get("status") or "unknown"
    detail = entry.get("detail")
    return "step %s%s" % (status, (" (%s)" % detail) if detail else "")


def _trend_note(rounds, config):
    """A one-sentence honest read of the most recent measured round-to-round
    change, reusing the same same-recipe / noise-floor rule the trajectory
    section renders per row, so the two never disagree with each other."""
    target_key, _reason = _target_metric_key(config, rounds)
    if not target_key:
        return "no resolved target metric to describe a trend"
    prev_val, prev_recipe = None, None
    verdicts = []
    for d in rounds:
        val, _kind, _n, _std, _why = _metric_of(d, target_key)
        recipe = _recipe_of(d)
        if val is not None and prev_val is not None and recipe == prev_recipe:
            v = _compare(prev_val, val, _noise_floor(config, recipe))
            if v:
                verdicts.append(v[0])
        if val is not None:
            prev_val, prev_recipe = val, recipe
    if not verdicts:
        return "not enough same-recipe round pairs on record to describe a trend"
    last = verdicts[-1]
    if last == "within_noise":
        return "the most recent round-to-round change is within noise"
    return "the most recent round-to-round change %s" % last


# --- windowing: keep the earliest and the most recent, name the middle ------
def _window_rows(rows, max_tokens, min_head, min_tail):
    """(kept_rows, removed_count, tokens_saved).

    Grows the kept tail before the kept head on each shrink step — a live
    decision reads the newest rows first — but at least `min_head` of the
    earliest rows survive every window this returns, so a trend cannot be
    manufactured by a window that only ever shows the end of the campaign.
    Deterministic and pure: same rows and budget in, same window out.
    """
    if not rows:
        return list(rows), 0, 0
    n = len(rows)
    full_cost = _tokens("\n".join(rows))
    if full_cost <= max_tokens:
        return list(rows), 0, 0
    head = max(0, min(min_head, n))
    tail = max(0, min(min_tail, n - head))
    while True:
        removed = n - head - tail
        if removed <= 0:
            kept = rows
        else:
            marker = "... [%d row(s) omitted here] ..." % removed
            kept = rows[:head] + [marker] + (rows[n - tail:] if tail else [])
        cost = _tokens("\n".join(kept))
        if cost <= max_tokens or removed <= 0:
            return kept, max(0, removed), max(0, full_cost - cost)
        if tail < n - head:
            tail += 1
        else:
            head += 1


def _assemble_section(name, budget_cfg, context_lines, rows, footer_lines,
                      min_head, min_tail, has_rows=True):
    """One section's rendered text plus its own bookkeeping.

    `context_lines` and `footer_lines` are kept in full (they are a handful of
    fixed facts, not a list that grows with the campaign); `rows` is the part
    that can grow without bound and is the only thing this function windows.
    `has_rows` is False for the two sections (identity, state) that never
    carry a row list at all, so an empty `rows` there is not rendered as
    "(none recorded)" — a marker that means something for, say, a campaign
    with no incidents yet, but nothing for a section that never has rows.
    """
    title_line = "== %s ==" % TITLES[name]
    max_tokens = _int(budget_cfg.get("max_tokens"))
    kept_rows, removed, saved = list(rows), 0, 0
    if rows and max_tokens is not None:
        kept_rows, removed, saved = _window_rows(rows, max_tokens, min_head, min_tail)
    body = [title_line]
    body.extend(context_lines or [])
    if kept_rows:
        body.extend(kept_rows)
    elif has_rows:
        body.append("(none recorded)")
    body.extend(footer_lines or [])
    text = "\n".join(l for l in body if l is not None)
    omitted = []
    if removed:
        omitted.append({"section": name,
                        "action": "windowed rows (kept the earliest and the "
                                 "most recent, dropped the middle)",
                        "removed": removed, "tokens_saved": saved})
    return {"name": name, "title": TITLES[name],
            "priority": _int(budget_cfg.get("priority"), 0),
            "mandatory": bool(budget_cfg.get("mandatory")),
            "budget_tokens": max_tokens, "text": text, "tokens_est": _tokens(text),
            "rows_total": len(rows), "rows_kept": len(kept_rows),
            "omitted": omitted}


# --- section builders --------------------------------------------------------
def _build_identity(domain, config, rounds, budget_cfg):
    target_key, target_reason = _target_metric_key(config, rounds)
    taxonomy = config.get("taxonomy") if isinstance(config.get("taxonomy"), list) else []
    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    first, last = (rounds[0], rounds[-1]) if rounds else (None, None)
    rnd_first = _int(_get(first, "round_num", "round")) if first else None
    rnd_last = _int(_get(last, "round_num", "round")) if last else None
    ctx = [
        "domain: %s" % domain,
        "target metric: %s (%s)" % (target_key or "unresolved", target_reason),
        "taxonomy size: %d class(es)" % len(taxonomy),
        "rounds on record: %d (round %s opened %s -> round %s last updated %s)"
        % (len(rounds), rnd_first if rnd_first is not None else "?",
           (_text(first.get("created_at")) or "unknown") if first else "unknown",
           rnd_last if rnd_last is not None else "?",
           (_text(last.get("updated_at")) or "unknown") if last else "unknown"),
        "SU envelope: %s (per_round_cap=%s, daily_cap=%s)"
        % (budget.get("su_envelope", "unrecorded"),
           budget.get("per_round_cap", "unrecorded"),
           budget.get("daily_cap", "unrecorded")),
        "goal: maximize %s under this envelope and the sealed holdout guard "
        "(NEVER_TRAIN/TRUSTED_SLUGS material is never trained on); no numeric "
        "target threshold is recorded in the domain config, so none is stated "
        "here" % (target_key or "the domain's target metric"),
    ]
    return _assemble_section("identity", budget_cfg, ctx, [], [], 0, 0, has_rows=False)


def _build_trajectory(rounds, config, target_key, target_reason, budget_cfg,
                      min_head, min_tail):
    ctx = ["target metric: %s (%s)" % (target_key or "unresolved", target_reason)]
    rows = []
    prev_val, prev_recipe = None, None
    for d in rounds:
        rnd = _int(_get(d, "round_num", "round"))
        recipe = _recipe_of(d) or "unrecorded recipe"
        val, kind, n, std, reason = _metric_of(d, target_key)
        floor = _noise_floor(config, recipe)
        if kind == "missing":
            row = "round %s (%s): no %s recorded (%s)" % (
                rnd, recipe, target_key or "metric", reason)
        elif kind == "mean_std":
            row = ("round %s (%s): %s mean=%.4f std=%.4f (n=%d seeds)"
                  % (rnd, recipe, target_key, val, std if std is not None else 0.0, n))
        else:
            row = ("round %s (%s): %s=%.4f (single seed, n=1 -- not a mean)"
                  % (rnd, recipe, target_key, val))
        if val is not None and prev_val is not None:
            if recipe != prev_recipe:
                row += (("; vs previous round: not compared (recipe changed "
                        "from %s to %s)") % (prev_recipe, recipe))
            else:
                cmp = _compare(prev_val, val, floor)
                if cmp is None:
                    row += ("; vs previous round: no comparison (%s)"
                           % ("no noise floor on record for %s" % recipe
                              if floor is None else "a value is missing"))
                else:
                    verdict, delta = cmp
                    if verdict == "within_noise":
                        row += ("; vs previous round: %+.4f, within noise "
                               "(below the %.4f floor for %s)"
                               % (delta, floor, recipe))
                    else:
                        row += ("; vs previous round: %+.4f, %s (at/above the "
                               "%.4f floor for %s)" % (delta, verdict, floor, recipe))
        rows.append(row)
        if val is not None:
            prev_val, prev_recipe = val, recipe
    return _assemble_section("trajectory", budget_cfg, ctx, rows, [], min_head, min_tail)


def _build_recipe_changes(rounds, corr_rows, config, budget_cfg, min_head, min_tail):
    watch_keys = sorted((config.get("round_params") or {}).keys())
    ctx = ["parameters tracked (from the domain config's round_params): %s"
          % (", ".join(watch_keys) if watch_keys else "none declared")]
    rows = []
    prev_params = {}
    for d in rounds:
        rnd = _int(_get(d, "round_num", "round"))
        steps = d.get("steps") if isinstance(d.get("steps"), dict) else {}
        for step in sorted(steps):
            entry = _step_entry(d, step)
            params = entry.get("params") if isinstance(entry.get("params"), dict) else None
            if not params:
                continue
            prev = prev_params.get(step)
            if prev is not None:
                changed = sorted(k for k in watch_keys
                                 if k in params and params.get(k) != prev.get(k))
                for k in changed:
                    corr = _correction_for_change(corr_rows, rnd, k)
                    if corr:
                        note = ("correction on record (seq %s, author %s): %s"
                               % (corr.get("seq", "?"), corr.get("author") or "unrecorded",
                                  corr.get("reason") or "(no reason recorded)"))
                    else:
                        note = "no correction record links this change to a reviewer"
                    rows.append("round %s %s: %s changed %r -> %r (actor %s); %s"
                               % (rnd, step, k, prev.get(k), params.get(k),
                                  entry.get("actor") or "unrecorded", note))
            prev_params[step] = dict(params)
    return _assemble_section("recipe_changes", budget_cfg, ctx, rows, [], min_head, min_tail)


def _build_incidents(sig_rows, rounds, corr_rows, budget_cfg, min_head, min_tail):
    by_round = {}
    for d in rounds:
        rnd = _int(_get(d, "round_num", "round"))
        if rnd is not None:
            by_round[rnd] = d
    ctx = ["%d signal firing(s) on record" % len(sig_rows)]
    rows = []
    for s in sig_rows:
        rnd = s.get("round")
        step = s.get("step") or "unrecorded step"
        name = s.get("signal") or "unnamed signal"
        sev = s.get("severity") or "unknown"
        job = s.get("job_id")
        reason = s.get("reason") or "(no reason recorded)"
        d = by_round.get(rnd)
        if d is not None:
            outcome = _step_outcome(d, step)
        else:
            outcome = "outcome unknown (round %s is not on the given round history)" % (
                rnd if rnd is not None else "?")
        corr = _correction_for_signal(corr_rows, name, rnd)
        addressed = (("addressed by a correction on record (seq %s, round %s, "
                     "author %s)" % (corr.get("seq", "?"),
                                    _nested(corr, "scope", "from_round"),
                                    corr.get("author") or "unrecorded"))
                    if corr else "no correction on record for this signal")
        rows.append("round %s%s: %s %s%s -- %s. outcome: %s. %s"
                   % (rnd if rnd is not None else "?",
                      (" step %s" % step) if step else "", name, sev,
                      (" job %s" % job) if job else "", reason, outcome, addressed))
    return _assemble_section("incidents", budget_cfg, ctx, rows, [], min_head, min_tail)


def _build_corrections(corr_rows, sig_rows, rounds, budget_cfg, min_head, min_tail):
    valid_rounds = [r for r in (_int(_get(d, "round_num", "round")) for d in rounds)
                    if r is not None]
    max_round = max(valid_rounds) if valid_rounds else None
    ctx = ["%d correction(s) on record" % len(corr_rows)]
    rows = []
    for c in corr_rows:
        seq = c.get("seq", "?")
        author = c.get("author") or "unrecorded"
        kind = c.get("kind") or "unrecorded kind"
        key = _text(_nested(c, "target", "key")) or "(no target key)"
        old = _nested(c, "target", "old")
        new = _nested(c, "target", "new")
        frm = _int(_nested(c, "scope", "from_round"))
        until = _int(_nested(c, "scope", "until_round"))
        reason = c.get("reason") or "(no reason recorded)"
        signal = c.get("signal")
        row = ("correction seq %s (round %s%s, author %s): %s %s %r -> %r -- %s"
              % (seq, frm if frm is not None else "?",
                 (" until round %s" % until) if until is not None else " onward",
                 author, kind, key, old, new, reason))
        if not signal:
            row += (". target signal: not stated on this correction record, so "
                    "recurrence cannot be checked.")
        elif frm is None:
            row += (". target signal %s: recurrence not checked (this correction "
                    "records no scope.from_round)." % signal)
        else:
            recurred_at = _recurrence(sig_rows, signal, frm)
            if recurred_at is not None:
                row += ". target signal %s recurred at round %s." % (signal, recurred_at)
            elif max_round is not None and max_round > frm:
                row += (". target signal %s did not recur through round %s."
                       % (signal, max_round))
            else:
                row += (". target signal %s: not yet observed (no round after %s "
                       "is on the given round history)." % (signal, frm))
        rows.append(row)
    return _assemble_section("corrections", budget_cfg, ctx, rows, [], min_head, min_tail)


def _build_resources(ledger_rows, ledger_skipped, config, budget_cfg, min_head, min_tail):
    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    envelope = _num(budget.get("su_envelope"))
    per_round_cap = budget.get("per_round_cap", "unrecorded")
    daily_cap = budget.get("daily_cap", "unrecorded")
    total_su, have_su = 0.0, 0
    by_round = {}
    for r in ledger_rows:
        su = _num(r.get("su"))
        if su is None:
            continue
        have_su += 1
        total_su += su
        rnd = _int(r.get("round"))
        by_round[rnd] = by_round.get(rnd, 0.0) + su
    skip_note = (", %d unparseable ledger line(s) skipped" % ledger_skipped
                if ledger_skipped else "")
    ctx = []
    if envelope is not None:
        remaining = envelope - total_su
        pct = (remaining / envelope * 100.0) if envelope else None
        ctx.append("su_envelope=%.1f spent_to_date=%.1f (from %d of %d ledger "
                  "row(s) carrying an su figure%s) remaining=%.1f%s"
                  % (envelope, total_su, have_su, len(ledger_rows), skip_note,
                     remaining, (" (%.1f%% of the envelope)" % pct)
                     if pct is not None else ""))
    else:
        ctx.append("no su_envelope is recorded in the domain config's budget "
                  "block; spend is reported without a remaining figure")
        ctx.append("spent_to_date=%.1f across %d of %d ledger row(s) carrying "
                  "an su figure%s" % (total_su, have_su, len(ledger_rows), skip_note))
    ctx.append("per_round_cap=%s daily_cap=%s" % (per_round_cap, daily_cap))
    rows = []
    for rnd in sorted(r for r in by_round if r is not None):
        rows.append("round %s: %.1f SU" % (rnd, by_round[rnd]))
    if by_round.get(None):
        rows.append("(unattributed to a round): %.1f SU" % by_round[None])
    return _assemble_section("resources", budget_cfg, ctx, rows, [], min_head, min_tail)


def _build_state(rounds, sig_rows, corr_rows, config, budget_cfg):
    if not rounds:
        ctx = ["no rounds are on record for this domain.",
               "OPEN QUESTION: should this domain's first round be started, "
               "and with which recipe?"]
        return _assemble_section("state", budget_cfg, ctx, [], [], 0, 0, has_rows=False)
    last = rounds[-1]
    rnd = _int(_get(last, "round_num", "round"))
    status = last.get("status") or "unknown"
    steps = last.get("steps") if isinstance(last.get("steps"), dict) else {}
    ctx = ["current round: %s (status %s)" % (rnd, status)]
    for step in sorted(steps):
        entry = steps[step] if isinstance(steps.get(step), dict) else {}
        ctx.append("  step %s: %s" % (step, entry.get("status", "unknown")))
    unresolved = [s for s in sig_rows
                 if s.get("round") == rnd and s.get("severity") in ("warn", "crit")
                 and not _correction_for_signal(corr_rows, s.get("signal"), rnd)]
    ctx.append("unresolved signal(s) on the current round: %d" % len(unresolved))
    for s in unresolved:
        ctx.append("  - %s (%s): %s" % (s.get("signal"), s.get("severity"),
                                        s.get("reason") or "(no reason recorded)"))
    trend = _trend_note(rounds, config)
    question = ("OPEN QUESTION: given that %s and %d unresolved signal(s) stand "
               "on the current round, what should the next round's recipe be, "
               "and should any unresolved signal gate the next submission?"
               % (trend, len(unresolved)))
    return _assemble_section("state", budget_cfg, ctx, [], [question], 0, 0, has_rows=False)


def _build_levers(config, budget_cfg, min_head, min_tail):
    round_params = config.get("round_params") if isinstance(config.get("round_params"), dict) else {}
    thresholds = config.get("thresholds") if isinstance(config.get("thresholds"), dict) else {}
    ctx = ["pre-registered parameters a plan may propose changing (policy.py's "
          "own risk bounds are the authority on what a reviewer may apply, and "
          "are not duplicated here)"]
    rows = []
    for k in sorted(round_params):
        rows.append("round_params.%s = %r" % (k, round_params[k]))
    for k in sorted(thresholds):
        rows.append("thresholds.%s = %r" % (k, thresholds[k]))
    return _assemble_section("levers", budget_cfg, ctx, rows, [], min_head, min_tail)


# --- refusal ------------------------------------------------------------
def _fit_refusal(text, num_ctx):
    """`text`, or a shorter and shorter stand-in until one fits `num_ctx`.

    The empty string is the true floor (0 tokens, so it fits any num_ctx >= 1):
    a num_ctx too small even for a one-word refusal is still a state this
    module must return rather than raise on, per its own contract that
    tokens_est is asserted below num_ctx on every return, refusal included.
    """
    if _tokens(text) < num_ctx:
        return text
    short = "REFUSED (num_ctx %d too small)" % num_ctx
    if _tokens(short) < num_ctx:
        return short
    if _tokens("REFUSED") < num_ctx:
        return "REFUSED"
    return ""


def _refuse(domain, num_ctx, reason, offending=None):
    text = _fit_refusal("REFUSED\ndomain: %s\n%s" % (domain, reason), num_ctx)
    tokens = _tokens(text)
    payload = {"text": text, "sections": [], "tokens_est": tokens, "omitted": [],
              "refused": True, "reason": reason,
              "offending_sections": list(offending or [])}
    payload["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert payload["tokens_est"] < num_ctx, (
        "digest.build()'s own refusal must fit num_ctx; got %d against %d"
        % (payload["tokens_est"], num_ctx))
    return payload


def _trim_note(internal_omitted, dropped_count, input_notes):
    lines = ["== TRIM NOTE =="]
    total = internal_omitted + dropped_count
    if total:
        lines.append(
            "%d item(s) (rows windowed or whole sections dropped) were reduced "
            "to fit this digest under its token budget; nothing was dropped "
            "silently -- see 'omitted' in this digest's returned metadata for "
            "exactly what and how much." % total)
    else:
        lines.append("nothing was trimmed; this digest carries every section "
                     "built from the given inputs in full.")
    if input_notes:
        lines.append("input hygiene: " + "; ".join(input_notes) + ".")
    return "\n".join(lines)


# --- the public entry point --------------------------------------------------
def build(domain, rounds, signals_history, corrections_history, ledger, config,
          num_ctx, omit_levers=False):
    """The one-shot campaign digest for a tier-2 planner call. Never raises on
    malformed history rows; may refuse (a returned, structured state, not an
    exception) when even the mandatory sections cannot fit `num_ctx`. See the
    module docstring for the full contract, the `rounds`/`ledger` naming note
    and the `omit_levers` research rationale."""
    domain_s = _text(domain) or "unknown-domain"
    n_ctx = _int(num_ctx)
    if n_ctx is None or n_ctx <= 0:
        raise ValueError("num_ctx must be a positive integer; got %r" % (num_ctx,))

    budget = load_budget()
    if budget.get("error"):
        return _refuse(domain_s, n_ctx, "digest_budget.json is invalid: %s"
                      % budget["error"])
    missing_budget = sorted(n for n in SECTION_NAMES if n not in budget["sections"])
    if missing_budget:
        return _refuse(domain_s, n_ctx, "digest_budget.json declares no budget "
                      "for section(s): %s" % ", ".join(missing_budget))

    meta = budget.get("meta") or {}
    reserve = max(1, _int(meta.get("prompt_reserve_tokens"), 220))
    min_head = max(0, _int(meta.get("min_head_rows"), 1))
    min_tail = max(0, _int(meta.get("min_tail_rows"), 2))

    cfg = config if isinstance(config, dict) else {}
    rounds_n, rounds_skipped = _sorted_rounds(rounds)
    sig_rows, sig_skipped = _normalize_signals(signals_history)
    corr_rows, corr_skipped = _normalize_corrections(corrections_history)
    ledger_rows, ledger_skipped = _normalize_ledger(ledger)
    target_key, target_reason = _target_metric_key(cfg, rounds_n)

    sb = budget["sections"]
    built = {
        "identity": _build_identity(domain_s, cfg, rounds_n, sb["identity"]),
        "trajectory": _build_trajectory(rounds_n, cfg, target_key, target_reason,
                                        sb["trajectory"], min_head, min_tail),
        "recipe_changes": _build_recipe_changes(rounds_n, corr_rows, cfg,
                                                sb["recipe_changes"], min_head, min_tail),
        "incidents": _build_incidents(sig_rows, rounds_n, corr_rows,
                                      sb["incidents"], min_head, min_tail),
        "corrections": _build_corrections(corr_rows, sig_rows, rounds_n,
                                          sb["corrections"], min_head, min_tail),
        "resources": _build_resources(ledger_rows, ledger_skipped, cfg,
                                      sb["resources"], min_head, min_tail),
        "state": _build_state(rounds_n, sig_rows, corr_rows, cfg, sb["state"]),
    }
    if not omit_levers:
        built["levers"] = _build_levers(cfg, sb["levers"], min_head, min_tail)

    input_notes = []
    if rounds_skipped:
        input_notes.append("%d malformed round entry(ies) were skipped" % rounds_skipped)
    if sig_skipped:
        input_notes.append("%d malformed signal-history entry(ies) were skipped" % sig_skipped)
    if corr_skipped:
        input_notes.append("%d malformed correction entry(ies) were skipped" % corr_skipped)
    if ledger_skipped:
        input_notes.append("%d unparseable ledger line(s) were skipped" % ledger_skipped)

    active = [n for n in DISPLAY_ORDER if n in built]
    dropped = []

    def render(active_names):
        parts = [built[n]["text"] for n in active_names]
        internal_omitted = sum(len(built[n]["omitted"]) for n in active_names)
        note = _trim_note(internal_omitted, len(dropped), input_notes)
        text = "\n\n".join(parts + [note])
        return text, _tokens(text)

    text, tokens = render(active)
    while tokens + reserve > n_ctx:
        droppable = [n for n in active if not built[n]["mandatory"]]
        if not droppable:
            break
        droppable.sort(key=lambda n: built[n]["priority"])
        victim = droppable[0]
        active.remove(victim)
        dropped.append({"section": victim,
                        "action": "dropped whole section (lowest priority "
                                 "remaining; the budget could not otherwise be met)",
                        "removed": built[victim]["rows_total"],
                        "tokens_saved": built[victim]["tokens_est"]})
        text, tokens = render(active)

    if tokens + reserve > n_ctx:
        offenders = [n for n in active if built[n]["mandatory"]]
        worst = max(offenders, key=lambda n: built[n]["tokens_est"]) if offenders else None
        worst_tokens = built[worst]["tokens_est"] if worst else tokens
        reason = ("REFUSED: mandatory section %r alone needs %d token(s) (of %d "
                 "total against a %d-token reserve) against num_ctx %d; nothing "
                 "further can be trimmed" % (worst, worst_tokens, tokens, reserve, n_ctx))
        return _refuse(domain_s, n_ctx, reason, offending=[worst] if worst else [])

    all_omitted = []
    for n in DISPLAY_ORDER:
        if n in built:
            all_omitted.extend(built[n]["omitted"])
    all_omitted.extend(dropped)

    sections_meta = []
    for n in SECTION_NAMES:
        cfg_n = sb.get(n, {})
        if n in built:
            b = built[n]
            sections_meta.append({
                "name": n, "title": b["title"], "priority": b["priority"],
                "mandatory": b["mandatory"], "included": n in active,
                "budget_tokens": b["budget_tokens"],
                "tokens_est": b["tokens_est"] if n in active else 0,
                "rows_total": b["rows_total"],
                "rows_kept": b["rows_kept"] if n in active else 0,
                "why": cfg_n.get("why"),
                "excluded_reason": None if n in active else
                    "dropped to fit num_ctx"})
        else:
            sections_meta.append({
                "name": n, "title": TITLES.get(n, n),
                "priority": _int(cfg_n.get("priority"), 0),
                "mandatory": bool(cfg_n.get("mandatory")), "included": False,
                "budget_tokens": _int(cfg_n.get("max_tokens")), "tokens_est": 0,
                "rows_total": 0, "rows_kept": 0, "why": cfg_n.get("why"),
                "excluded_reason": ("omit_levers=True" if n == "levers" and omit_levers
                                    else "not built")})

    payload = {"text": text, "sections": sections_meta, "tokens_est": tokens,
              "omitted": all_omitted, "refused": False, "reason": None}
    payload["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert payload["tokens_est"] < n_ctx, (
        "digest.build() must never hand back text at or over num_ctx; got %d "
        "against %d" % (payload["tokens_est"], n_ctx))
    return payload


# --- CLI ---------------------------------------------------------------------
def _load_json_file(path, default):
    if not path:
        return default
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _default_paths(root, domain):
    base = pathlib.Path(str(root)) / "results" / "framework" / "_brain" / str(domain)
    return {"rounds": base / "digest_rounds.json",
            "signals": base / "digest_signals.json",
            "corrections": base / "digest_corrections.json",
            "ledger": base / "digest_ledger.json",
            "config": base / "digest_config.json"}


def main(argv=None):
    """`build` (from JSON files staged under results/framework/_brain/<domain>/,
    the same convention `evidence.py` stages ledger.json/su.json/corrections.json
    under) | `sections` (the priorities and budgets, with their reasons) |
    `budget` (the raw effective digest_budget.json)."""
    ap = argparse.ArgumentParser(prog="digest", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    bp = sub.add_parser("build", help="build a campaign digest from staged files")
    bp.add_argument("--domain", required=True)
    bp.add_argument("--num-ctx", type=int, default=32768)
    bp.add_argument("--omit-levers", action="store_true")
    bp.add_argument("--root", default=None)
    bp.add_argument("--rounds-file", default=None)
    bp.add_argument("--signals-file", default=None)
    bp.add_argument("--corrections-file", default=None)
    bp.add_argument("--ledger-file", default=None)
    bp.add_argument("--config-file", default=None)
    bp.add_argument("--text", action="store_true", help="print only the digest text")
    bp.add_argument("--out", default=None)

    sub.add_parser("sections", help="the section priorities and budgets, with reasons")
    sub.add_parser("budget", help="the raw effective digest_budget.json")

    args = ap.parse_args(argv)
    if args.cmd == "budget":
        print(json.dumps(load_budget(), indent=2, sort_keys=True))
        return 0
    if args.cmd == "sections":
        b = load_budget()
        rows = [dict(name=n, title=TITLES.get(n, n), **b["sections"].get(n, {}))
               for n in SECTION_NAMES]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if args.cmd != "build":
        ap.print_help()
        return 0

    root = args.root or str(corpus.REPO)
    paths = _default_paths(root, args.domain)
    rounds = _load_json_file(args.rounds_file or paths["rounds"], [])
    signals_history = _load_json_file(args.signals_file or paths["signals"], [])
    corrections_history = _load_json_file(args.corrections_file or paths["corrections"], [])
    ledger = _load_json_file(args.ledger_file or paths["ledger"], [])
    config = _load_json_file(args.config_file or paths["config"], {})

    result = build(args.domain, rounds, signals_history, corrections_history,
                   ledger, config, args.num_ctx, omit_levers=args.omit_levers)
    if args.out:
        corpus.write_json(args.out, result)
    elif args.text:
        sys.stdout.write(result["text"] + "\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result.get("refused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
