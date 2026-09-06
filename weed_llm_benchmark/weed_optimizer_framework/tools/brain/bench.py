"""Supervision benchmark harness — every arm scored over the same frozen cases (v3.25.4).

Why this exists
---------------
The scripted loop has one recorded detection behaviour and it is written in
code, not in prose: a step is recorded `failed` and the domain pauses. On
2026-08-29 that behaviour cost two 12 h H100 jobs before it fired (the first
TIMEOUT was recorded failed and the identical command was resubmitted on the
next tick), and once it fired nobody was told for six days. Any claim that a
model supervises better than that has to be measured against *that* definition
over *those* artifacts, not against a description of it.

This module is the scorer. It reads the exported corpus
(`results/framework/supervision_bench/cases/<case_id>/`), runs each arm over
every case, and reports each arm as a **margin over the deterministic signals
arm (A0+)** per incident class, with a Wilson interval on every proportion and
a minimum detectable difference printed beside every comparison. At the corpus
sizes this campaign can produce (order 12-30 incidents) the MDD is large — n=12
at p0=0.5 cannot resolve a difference below 0.48 — and printing it next to the
delta is the difference between a result and a decoration.

Arms
----
  A0    the scripted scheduler. Detection is the code's own definition: the
        bundle's own step is recorded `failed` on the ledger (head entry or an
        attempt, not superseded) **and** the domain records a pause within one
        scheduler tick of that record. Implemented in `arm_a0` against the
        bundle's ledger section — a first failure with no pause is a miss,
        which is the 08-29 behaviour, and a step recorded `done` is a miss,
        which is the whole silent-wrongness class.
  A0+   deterministic signals only, no model, no notification channel. Signals
        come from the bundle's frozen `signals` section when the export carries
        one, otherwise from `tools/brain/signals.py` (WP3, imported lazily).
        With neither, the arm is marked degraded and every comparison against
        it is refused rather than silently scored against an empty baseline.
  A0++  A0+ plus the alarm. Same findings; it can page a person, so it is the
        only deterministic arm with a notification latency at all. A0 and A0+
        have none — not zero: `notify_latency_s` is null and `notifies` false,
        because a channel that does not exist has no latency.
  L1    a model given status fields and metrics only (what a worker reports
        about itself). No raw lines, so its quotes cannot resolve.
  L2    a model given raw excerpts with the signals section removed, and with
        the case's own covering signal names scrubbed from the remaining text
        (signal-blind).
  L3    a model given signals plus excerpts.
  L4    L3 plus exactly one bounded retrieval round from a per-case whitelist.

Model access
------------
None of A0/A0+/A0++ and none of `--reproduce` touch a model. Model arms call a
single pluggable callable supplied by the integrator:

    model_fn(prompt: str, model_id: str, num_ctx: int) -> {
        "text": str, "tokens_in": int, "tokens_out": int,
        "latency_s": float, "su": float}

so this file carries no provider dependency and imports on the always-on
server, inside a SLURM job and on a laptop (stdlib only). `FakeModel` below is
a deterministic stand-in used by the tests.

Contracts this file depends on (named here so the other halves can target them)
-----------------------------------------------------------------------------
* Case layout `cases/<case_id>/{bundle.json, truth.json, artifacts/}` and
  `split.json` as WP2 defines them. `truth.json` may carry one optional field
  beyond that schema, `incident_ts` (ISO-8601 or epoch): without it A0++ has no
  onset to measure a notification latency from and reports null.
* The bundle's `ledger` section is the only thing A0 reads, and A0 is the
  baseline every other arm is measured against, so its shape matters more than
  the rest: `{"rounds":[round_doc,...], "pause":{"reason","at"}, "tick_s":120}`
  with round docs as `db.py` writes them (`round_num`, `steps.<step>` with
  `status`, `at`, and `attempts[]`). Four looser shapes are accepted (see
  `ledger_view`); an export that carries none of them makes every incident
  undecidable for A0, which the report prints rather than scoring as a miss.
* `out_tail` (and any section shaped like it) is the bundle's only addressable
  text: `{artifact_id, path, sha256, lines: [[absolute_line_number, text], ...]}`.
  A quote can only be validated against a line that has an address.
* WP3, when it lands: `signals.detect(bundle) -> list[Signal]` and
  `citations.resolve(bundle, quote) -> {"artifact_id","line"} | None`. Both are
  imported lazily and both have a local fallback, so this file develops and
  runs without them; when they are importable they win, and a disagreement
  between WP3's validator and the fallback shows up as a changed number.
* `rubric.md` must quote the value of `rubric_sha256()`; it is the hash of
  every constant a verdict is scored against and it is written into every
  results file.

Statistics
----------
* n is the number of **cases**, never the number of case x repeat rows. Three
  repeats of one case are not three cases; repeats are collapsed by majority
  and their disagreement is reported separately as `repeat_disagreement`.
* Wilson score intervals on every proportion (normal-approximation intervals
  put 0/12 at exactly zero, which is how a small corpus is made to look
  certain).
* Differences carry a Newcombe hybrid-score interval, which treats the arms as
  independent. They are paired here (same cases), so the interval is
  conservative; that is stated in the output rather than silently corrected,
  because a paired McNemar analysis over n=12 has its own credibility problem.
* `mdd_proportion(n, p0)` is the smallest difference detectable at 80 % power,
  two-sided alpha 0.05, under the normal approximation; it is attached to every
  comparison and the renderer marks any |delta| below it as not a result.
* `su_per_review` is whatever the model callable reports. VERIFY: reconcile it
  against the cluster's own accounting before any cost-frontier number is
  published -- `sacct -j <jobid> --format=JobID,AllocTRES,Elapsed,State -P`
  (SU = GPU count x elapsed hours x 2 on H100, x 1 on V100); a callable that
  reports 0.0 is reporting "not measured", not "free".

Pre-registration
----------------
Everything a verdict is scored against is a module constant below
(`DETECTION_SEVERITY`, `ESCALATION_POLICY`, `L1_SECTIONS`,
`DETERMINISTIC_CORRECTIONS`, `CORRECTABLE_RISKS`, `REPEATS`, `BASELINE_ARM`)
and its sha256 is written into every results file as `rubric_sha256`, so a
result carries the rules it was scored under and a later edit to those rules is
visible in the diff of that hash.

Nothing here raises out of `main()`: a corpus that cannot be read is reported
as the cases it could not read, and the arms still run over the rest.

CLI
---
  python -m weed_optimizer_framework.tools.brain.bench list --root DIR
  python -m weed_optimizer_framework.tools.brain.bench run  --root DIR \\
      --split test --arms A0,A0p,A0pp --repeats 3 \\
      [--model-entry pkg.module:callable --model glm-4.7-flash --num-ctx 32768]
  python -m weed_optimizer_framework.tools.brain.bench reproduce --root DIR \\
      --verdicts DIR/verdicts          # or: bench.py --reproduce ...
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import sys
import time

# ---------------------------------------------------------------- pre-registered --

# A finding counts as a detection (and, on a healthy control, as a false alarm)
# at this severity or above. Fixed before any model run: a threshold chosen
# after seeing the verdicts is a free parameter worth several points of recall.
DETECTION_SEVERITY = "warn"
SEVERITY_ORDER = {"info": 0, "warn": 1, "crit": 2}

# Every arm is reported as a margin over this one. A0+ (signals, no model) is
# the baseline because the question is what the model adds to the deterministic
# checks, not what it adds to nothing.
BASELINE_ARM = "A0p"

# The scheduler's tick. A0's definition is "pauses within one tick" of the
# failure record; a bundle that carries its own `tick_s` overrides this.
A0_TICK_S = 120.0

# Incident classes. A case's class comes from truth.json; healthy controls keep
# whatever class they were exported with so a false-alarm rate can be read per
# class, and "all" is always reported.
CLASSES = ("operational", "config", "code", "design")

# Correction correctness is scored only where a correction was in the agent's
# reach: reversible config (R1) and compute/corpus state (R2). Proposals on
# R3/R4 cases are governance events, not corrections, and are counted by WP5.
CORRECTABLE_RISKS = ("R1", "R2", "r1", "r2")

# What each deterministic arm does with its findings. A0 pauses the domain and
# tells nobody (the 6-day silent pause is the record), so its escalation is
# "none" however severe the incident. A0+ has findings and no channel. A0++ has
# the site alarm, whose only recipient is a person.
ESCALATION_POLICY = {
    "A0": {"notifies": False, "at_or_above": None, "to": "none"},
    "A0p": {"notifies": False, "at_or_above": None, "to": "none"},
    "A0pp": {"notifies": True, "at_or_above": DETECTION_SEVERITY, "to": "human"},
}

# The deterministic correction each signal carries, mirroring what WP1 shipped
# in the scheduler (halve the epochs of a walltime-shaped failure, and nothing
# else). A0+/A0++ propose exactly these and nothing more: an arm that proposes
# a correction it has no rule for is a model arm.
DETERMINISTIC_CORRECTIONS = {
    "walltime_bound": {"action": "set_round_param",
                       "params": {"epochs": "half"},
                       "risk": "R1"},
    "pool_growth": {"action": "set_round_param",
                    "params": {"epochs": "half"},
                    "risk": "R1"},
}

# L1 is "status fields and metrics" — the information a worker reports about
# itself. sacct belongs here (a job's own final state is a status field); raw
# `.out` lines, traces, harvest records and registry diffs do not. The renderer
# additionally drops every line-addressed array from these sections, so L1
# cannot quote an artifact even by accident.
L1_SECTIONS = ("ledger", "sacct", "results_csv", "strategy", "su", "resources")

# L4's one retrieval round draws from the case's own artifact files and from
# nothing else. The bundle is a trimmed view of those files, so un-trimming one
# is the only new information a retrieval can carry; a request for anything
# outside that whitelist is refused and counted.
RETRIEVAL_MAX_LINES = 200
RETRIEVAL_ROUNDS = 1

REPEATS = 3
# The callable owns sampling; this value is recorded in the run record so the
# number can be checked against what the integrator actually configured.
TEMPERATURE = 0.3

# A quote shorter than this is not evidence (WP3 citation rule, restated here
# because bench must score committed verdicts on a machine where WP3's module
# is not installed).
MIN_QUOTE_CHARS = 20

ARM_IDS = ("A0", "A0p", "A0pp", "L1", "L2", "L3", "L4")
MODEL_ARMS = ("L1", "L2", "L3", "L4")

# Bundle sections, in the order the prompt renders them.
SECTION_ORDER = ("ledger", "sacct", "out_tail", "results_csv", "strategy",
                 "trace", "slug_scores", "registry_diff", "harvest",
                 "resources", "su", "corrections", "plan", "signals")

PROMPT_TASK = (
    "You are reviewing one step of an autonomous data-collection and training "
    "loop on an HPC cluster. Decide whether this step went wrong.\n"
    "Answer with one JSON object and nothing else:\n"
    '{"verdict":"ok|issue",'
    '"findings":[{"signal":"","quote":"","diagnosis":"","severity":"info|warn|crit"}],'
    '"corrections":[{"action":"","params":{},"risk":"R1|R2|R3|R4","reason":"","quote":""}],'
    '"escalate":{"to":"none|tier1|tier2|human","reason":""},'
    '"confidence":0.0}\n'
    "Every finding must quote a line of the evidence below verbatim; a quote "
    "that does not resolve to a line is discarded and is never actionable.\n"
)
PROMPT_RETRIEVAL_NOTE = (
    'You may request more evidence once by adding "retrieve":["<artifact or '
    'section name>"] to the object; you will then be asked again with it.\n'
)


# ---------------------------------------------------------------- small utils --

def _canonical(obj):
    """Canonical JSON: sorted keys, no padding. Used for every hash here."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _num(v, default=None):
    try:
        if v is None or isinstance(v, bool):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _ts(v):
    """Epoch seconds from a float, a numeric string or an ISO-8601 string.

    Ledger entries carry ISO-8601 (`db._now_iso`), scheduler state carries
    epoch floats, and an exported case may carry either. Returns None when the
    value is missing or unparseable — A0 then reports the case undecidable
    rather than guessing an ordering.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:
        from datetime import datetime
        txt = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _norm_text(s):
    """Whitespace-normalised, case-folded text for quote matching."""
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _artifact_key(name):
    """Artifact identity for matching truth lines to citations: basename only.

    truth.json names an artifact as it sits in `artifacts/`; a bundle may carry
    an absolute /ocean path in the same field. Comparing basenames keeps the
    two agreeing without rewriting either.
    """
    return os.path.basename(str(name or "").strip()).lower()


def _sev(s):
    return SEVERITY_ORDER.get(str(s or "").strip().lower(), 0)


def _at_or_above(sev_name, threshold=DETECTION_SEVERITY):
    return _sev(sev_name) >= _sev(threshold)


def repo_root():
    """Repo root: REPO_ROOT if set, else three levels up from this package."""
    env = os.environ.get("REPO_ROOT")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[3]


def default_root():
    return repo_root() / "results" / "framework" / "supervision_bench"


def rubric_sha256():
    """Hash of everything a verdict is scored against.

    Written into every results file. Two results files with different values
    here were not scored under the same rules, whatever their prose says.
    """
    return _sha256({
        "detection_severity": DETECTION_SEVERITY,
        "severity_order": SEVERITY_ORDER,
        "baseline": BASELINE_ARM,
        "a0_tick_s": A0_TICK_S,
        "classes": list(CLASSES),
        "correctable_risks": list(CORRECTABLE_RISKS),
        "escalation_policy": ESCALATION_POLICY,
        "deterministic_corrections": DETERMINISTIC_CORRECTIONS,
        "l1_sections": list(L1_SECTIONS),
        "retrieval": {"source": "case artifact files",
                      "max_lines": RETRIEVAL_MAX_LINES,
                      "rounds": RETRIEVAL_ROUNDS},
        "min_quote_chars": MIN_QUOTE_CHARS,
        "repeats": REPEATS,
        "temperature": TEMPERATURE,
    })


# ---------------------------------------------------------------- corpus --

def _read_json(path):
    """(object, error). Never raises: a corpus is read by a reporting tool."""
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing: %s" % os.path.basename(str(path))
    except Exception as e:
        return None, "unreadable %s: %s" % (os.path.basename(str(path)), e)


def load_case(case_dir):
    """One case from disk: {case_id, dir, bundle, truth, artifacts, errors}.

    A case with an unreadable bundle or truth is returned with `ok` False and
    its reasons; the caller reports it and carries on. Losing the whole run
    because one export is half written is the failure mode this campaign is
    about.
    """
    p = pathlib.Path(str(case_dir))
    case = {"case_id": p.name, "dir": str(p), "bundle": None, "truth": None,
            "artifacts": {}, "errors": [], "ok": False}
    bundle, err = _read_json(p / "bundle.json")
    if err:
        case["errors"].append(err)
    if bundle is not None and not isinstance(bundle, dict):
        case["errors"].append("bundle.json is not an object")
        bundle = None
    truth, err = _read_json(p / "truth.json")
    if err:
        case["errors"].append(err)
    if truth is not None and not isinstance(truth, dict):
        case["errors"].append("truth.json is not an object")
        truth = None
    case["bundle"] = bundle or {}
    case["truth"] = truth or {}
    if truth and truth.get("case_id") and str(truth["case_id"]) != p.name:
        case["errors"].append("truth.case_id %r != directory %r"
                              % (truth.get("case_id"), p.name))
    try:
        adir = p / "artifacts"
        if adir.is_dir():
            for f in sorted(adir.rglob("*")):
                if f.is_file():
                    case["artifacts"][_artifact_key(f.name)] = str(f)
    except Exception as e:
        case["errors"].append("artifacts unreadable: %s" % e)
    case["bundle_sha256"] = _sha256(case["bundle"]) if bundle else ""
    case["truth_sha256"] = _sha256(case["truth"]) if truth else ""
    case["ok"] = bool(bundle) and bool(truth)
    return case


def load_corpus(root=None):
    """Every case under `<root>/cases/`. Returns {root, cases, errors, hash}."""
    root = pathlib.Path(str(root or default_root()))
    out = {"root": str(root), "cases": {}, "errors": [], "hash": ""}
    cdir = root / "cases"
    if not cdir.is_dir():
        out["errors"].append("no cases directory: %s" % cdir)
        return out
    try:
        entries = sorted([d for d in cdir.iterdir() if d.is_dir()])
    except Exception as e:
        out["errors"].append("cases directory unreadable: %s" % e)
        return out
    for d in entries:
        case = load_case(d)
        for e in case["errors"]:
            out["errors"].append("%s: %s" % (case["case_id"], e))
        out["cases"][case["case_id"]] = case
    out["hash"] = corpus_hash(out["cases"])
    return out


def corpus_hash(cases):
    """One hash over the cases actually scored, bundles and truth together.

    Printed in every results file: a number produced from a different corpus
    than the one committed is then visible without re-reading the cases.
    """
    rows = [[cid, c.get("bundle_sha256", ""), c.get("truth_sha256", "")]
            for cid, c in sorted((cases or {}).items())]
    return _sha256(rows)


def load_split(root=None):
    """split.json plus a check of its own committed sha256.

    The sha is taken over the split object without its `sha256` field, so the
    file certifies itself: an edited split no longer matches the hash printed
    in the results files produced before the edit.
    """
    root = pathlib.Path(str(root or default_root()))
    obj, err = _read_json(root / "split.json")
    out = {"dev": [], "test": [], "rule": "", "sha256": "", "sha_ok": None,
           "errors": []}
    if err or not isinstance(obj, dict):
        out["errors"].append(err or "split.json is not an object")
        return out
    out["dev"] = [str(x) for x in (obj.get("dev") or [])]
    out["test"] = [str(x) for x in (obj.get("test") or [])]
    out["rule"] = str(obj.get("rule") or "")
    out["sha256"] = str(obj.get("sha256") or "")
    body = {k: v for k, v in obj.items() if k != "sha256"}
    computed = _sha256(body)
    out["computed_sha256"] = computed
    if out["sha256"]:
        out["sha_ok"] = (out["sha256"] == computed)
        if not out["sha_ok"]:
            out["errors"].append(
                "split.json sha256 does not match its content "
                "(committed %s, computed %s)" % (out["sha256"][:12], computed[:12]))
    else:
        out["errors"].append("split.json carries no sha256")
    overlap = sorted(set(out["dev"]) & set(out["test"]))
    if overlap:
        out["errors"].append("dev and test overlap: %s" % ",".join(overlap))
    return out


def select_cases(corpus, split, which="test"):
    """Case ids for a split name. `all` takes everything that loaded.

    The dev case never appears in reported numbers, so `--split test` is the
    default everywhere and `--split dev` has to be asked for by name.
    """
    have = corpus.get("cases", {}) or {}
    w = str(which or "test").lower()
    if w == "all":
        return sorted(have.keys()), []
    want = [str(x) for x in (split.get(w) or [])]
    known = [c for c in want if c in have]
    missing = [c for c in want if c not in have]
    return known, missing


# ---------------------------------------------------------------- bundle views --

def _sections(bundle):
    s = (bundle or {}).get("sections")
    return s if isinstance(s, dict) else {}


def _as_line_artifacts(section):
    """Line-addressed artifacts in a section, or [].

    Canonical shape is `out_tail = {artifact_id, path, sha256, lines:[[n,text]]}`;
    a list of those is accepted so a bundle can carry more than one tail, and
    `lines` entries may be pairs or `{line,text}` objects. Anything else is not
    line-addressed and therefore not quotable: a finding can only cite what has
    an (artifact, line) address.
    """
    out = []
    items = section if isinstance(section, list) else [section]
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("lines"), list):
            continue
        aid = _artifact_key(item.get("artifact_id") or item.get("path") or "artifact")
        rows = []
        for row in item["lines"]:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                n, text = row[0], row[1]
            elif isinstance(row, dict):
                n, text = row.get("line"), row.get("text")
            else:
                continue
            n = _num(n)
            if n is None:
                continue
            rows.append((int(n), str(text)))
        out.append({"artifact_id": aid, "sha256": str(item.get("sha256") or ""),
                    "path": str(item.get("path") or ""), "rows": rows})
    return out


def bundle_lines(bundle):
    """Every line-addressed line in the bundle as {artifact_id, line, text}.

    This is the address space a citation must resolve into. Sections with no
    line addresses (sacct rows, the strategy dict, signal values) are evidence
    a model may reason from but cannot cite — by construction, because the
    validator has nowhere to point.
    """
    out = []
    for name, sec in _sections(bundle).items():
        for art in _as_line_artifacts(sec):
            for n, text in art["rows"]:
                out.append({"artifact_id": art["artifact_id"], "line": n,
                            "text": text, "section": name})
    return out


def ledger_view(bundle):
    """The ledger section normalised for A0: rounds, the pause, the tick.

    Accepted shapes (WP3's builder should emit the first; the rest exist
    because an exported case may have been assembled by hand from an archived
    round doc):
      {"rounds":[round_doc,...], "pause":{"reason","at"}, "tick_s":120}
      {"rounds":[...], "paused_reason":"...", "paused_at":...}
      {"rounds":[...], "domain":{"paused_reason":..., "paused_at":...}}
      {"rounds":[...], "events":[{"kind":"pause","reason":...,"ts":...}]}
      [round_doc, ...]
    `present` False means the export carries no ledger at all, which A0 reports
    as undecidable rather than as a miss.
    """
    sec = _sections(bundle).get("ledger")
    view = {"present": False, "rounds": [], "pause": None,
            "tick_s": A0_TICK_S, "notes": []}
    if sec is None:
        return view
    view["present"] = True
    if isinstance(sec, list):
        view["rounds"] = [r for r in sec if isinstance(r, dict)]
        return view
    if not isinstance(sec, dict):
        view["present"] = False
        view["notes"].append("ledger section is neither a list nor an object")
        return view
    rounds = sec.get("rounds")
    if rounds is None:
        rounds = sec.get("docs")
    if rounds is None and isinstance(sec.get("round"), dict):
        rounds = [sec["round"]]
    if rounds is None and isinstance(sec.get("steps"), dict):
        rounds = [sec]                       # the section IS one round doc
    view["rounds"] = [r for r in (rounds or []) if isinstance(r, dict)]
    tick = _num(sec.get("tick_s"))
    if tick and tick > 0:
        view["tick_s"] = tick

    pause = None
    if isinstance(sec.get("pause"), dict):
        p = sec["pause"]
        pause = {"reason": str(p.get("reason") or p.get("paused_reason") or ""),
                 "at": _ts(p.get("at") if p.get("at") is not None else p.get("paused_at"))}
    if pause is None:
        for holder in (sec, sec.get("domain") if isinstance(sec.get("domain"), dict) else {}):
            reason = str((holder or {}).get("paused_reason") or "")
            if reason:
                pause = {"reason": reason, "at": _ts((holder or {}).get("paused_at"))}
                break
    if pause is None and isinstance(sec.get("events"), list):
        evs = [e for e in sec["events"]
               if isinstance(e, dict) and str(e.get("kind") or "").startswith("paus")]
        if evs:
            last = evs[-1]
            when = last.get("ts") if last.get("ts") is not None else last.get("at")
            pause = {"reason": str(last.get("reason") or "pause event"),
                     "at": _ts(when)}
    if pause is not None and not pause.get("reason"):
        pause["reason"] = "paused"
    view["pause"] = pause
    return view


def _round_matches(doc, want):
    """Does this round doc belong to the round the bundle is about?"""
    if want in (None, ""):
        return True
    for key in ("round_num", "round", "n"):
        v = doc.get(key)
        if v is None:
            continue
        if _num(v) is not None and _num(want) is not None:
            if _num(v) == _num(want):
                return True
        if str(v) == str(want):
            return True
    rid = str(doc.get("round_id") or doc.get("_id") or "")
    if rid and rid.endswith("#%s" % want):
        return True
    return False


def step_entries(bundle, round_no=None, step=None):
    """Head entry plus attempts for one step of one round, oldest first.

    v3.25.0 keeps a step's history in `attempts[]` and only the last write in
    the head, so a step that failed and was then retried carries its failure in
    `attempts`. A0 has to look at both or it would miss every incident that was
    followed by another write.
    """
    lv = ledger_view(bundle)
    if round_no is None:
        round_no = bundle.get("round")
    if step is None:
        step = bundle.get("step")
    step = str(step or "")
    rounds = [r for r in lv["rounds"] if _round_matches(r, round_no)]
    if not rounds and len(lv["rounds"]) == 1:
        rounds = lv["rounds"]                # single-round export, no numbering
    out = []
    for doc in rounds:
        steps = doc.get("steps")
        if not isinstance(steps, dict):
            continue
        entry = steps.get(step)
        if not isinstance(entry, dict):
            continue
        for a in (entry.get("attempts") or []):
            if isinstance(a, dict):
                out.append(a)
        out.append({k: v for k, v in entry.items() if k != "attempts"})
    return out


def _is_superseded(entry):
    """A supervisor-cancelled step, recorded `skipped: superseded: ...`.

    It is not a failure and must not be read as one — that semantics is what
    keeps a supervisor's own cancel from tripping the stop-loss.
    """
    return (isinstance(entry, dict) and entry.get("status") == "skipped"
            and str(entry.get("detail") or "").startswith("superseded: "))


def _current_param(bundle, name):
    """The parameter the step actually ran with, wherever the export put it."""
    secs = _sections(bundle)
    holders = []
    strat = secs.get("strategy")
    if isinstance(strat, dict):
        holders += [strat, strat.get("params"), strat.get("round_params"),
                    strat.get("hyperparameters")]
    plan = secs.get("plan")
    if isinstance(plan, dict):
        holders += [plan, plan.get("round_params")]
    for entry in step_entries(bundle):
        if isinstance(entry.get("params"), dict):
            holders.append(entry["params"])
    for h in holders:
        if isinstance(h, dict) and h.get(name) is not None:
            v = _num(h.get(name))
            if v is not None:
                return v
    return None


# ---------------------------------------------------------------- WP3 bridges --

_WP3 = {}


def _wp3(name):
    """Import a sibling WP3 module lazily; None until it lands.

    bench is developed and tested before signals.py/citations.py exist, and it
    also has to run on a laptop from a checkout that carries only the corpus.
    Import failure is a state to report, never an exception to raise.
    """
    if name in _WP3:
        return _WP3[name]
    mod = None
    for target in ([__package__ + "." + name] if __package__ else []) + [
            "weed_optimizer_framework.tools.brain." + name]:
        try:
            mod = importlib.import_module(target)
            break
        except Exception:
            mod = None
    _WP3[name] = mod
    return mod


def resolve_quote(bundle, quote, lines=None):
    """Address a quote inside the bundle: {artifact_id, line} or None.

    Prefers WP3's `citations.resolve` when it is importable, so there is one
    validator in the campaign. The fallback implements the same pre-registered
    rule (>= MIN_QUOTE_CHARS characters, whitespace-normalised substring match
    against the bundle's line texts) because scoring a committed verdict must
    not require the cluster-side package.
    """
    cit = _wp3("citations")
    if cit is not None and hasattr(cit, "resolve"):
        try:
            hit = cit.resolve(bundle, quote)
            if isinstance(hit, dict) and hit.get("artifact_id") is not None:
                return {"artifact_id": _artifact_key(hit.get("artifact_id")),
                        "line": int(_num(hit.get("line"), 0) or 0)}
            if hit is None:
                return None
        except Exception:
            pass                              # fall through to the local rule
    q = _norm_text(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return None
    for row in (lines if lines is not None else bundle_lines(bundle)):
        if q in _norm_text(row["text"]):
            return {"artifact_id": row["artifact_id"], "line": int(row["line"])}
    return None


def signals_for(bundle):
    """The deterministic signals for a case: {signals, source, degraded, reason}.

    Order: the signals the export froze into the bundle (they are the facts the
    case was labelled against), then WP3's `signals.detect(bundle)`, then a
    stub. The stub is marked degraded and every comparison against a degraded
    A0+ is refused — "no signals module" and "the signals found nothing" score
    identically otherwise, and one of those is a result while the other is a
    missing dependency.
    """
    sec = _sections(bundle).get("signals")
    if isinstance(sec, list):
        return {"signals": [s for s in sec if isinstance(s, dict)],
                "source": "bundle", "degraded": False, "reason": ""}
    mod = _wp3("signals")
    if mod is not None:
        for fname in ("detect", "run", "evaluate", "all_signals"):
            fn = getattr(mod, fname, None)
            if callable(fn):
                try:
                    got = fn(bundle)
                    return {"signals": [s for s in (got or []) if isinstance(s, dict)],
                            "source": "module:%s" % fname, "degraded": False,
                            "reason": ""}
                except Exception as e:
                    return {"signals": [], "source": "module", "degraded": True,
                            "reason": "signals.%s raised: %s" % (fname, e)}
        return {"signals": [], "source": "module", "degraded": True,
                "reason": "signals module exports no detect()"}
    return {"signals": [], "source": "stub", "degraded": True,
            "reason": "signals module not available"}


# ---------------------------------------------------------------- prompt views --

# The deployed alarm's own delay before a person can see it: scheduler_health
# memoises its verdict for 15 s on top of the scheduler's 120 s tick. A0++'s
# replayed notification latency carries this constant, which is read from code,
# not measured. VERIFY: measure it live in the WP8 integration round —
# `curl -s localhost:8000/api/health/scheduler` timestamps against the journal
# line of the pause that caused it.
ALARM_SURFACE_S = 135.0


def _redact_strings(obj, needles, protected_lines, counter):
    """Replace every occurrence of a covering signal name, in place, in a copy.

    Line-addressed rows listed in truth.load_bearing_lines are left alone: a
    signal-blind arm must lose the *signal*, not the evidence it is supposed to
    find. When a load-bearing line happens to name the signal, the arm keeps
    seeing that name and `redaction_skipped_load_bearing` counts it, so the
    ablation reports how leaky it was instead of quietly being weaker.
    """
    if isinstance(obj, dict):
        arts = _as_line_artifacts(obj)
        if arts:
            aid = arts[0]["artifact_id"]
            rows = []
            for row in obj.get("lines") or []:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    n, text = row[0], str(row[1])
                elif isinstance(row, dict):
                    n, text = row.get("line"), str(row.get("text") or "")
                else:
                    rows.append(row)
                    continue
                key = (aid, int(_num(n, -1) or -1))
                low = text.lower()
                if any(x in low for x in needles):
                    if key in protected_lines:
                        counter["skipped_load_bearing"] += 1
                    else:
                        text = "[redacted: signal name]"
                        counter["redacted_lines"] += 1
                rows.append([n, text] if not isinstance(row, dict)
                            else {"line": n, "text": text})
            out = {k: v for k, v in obj.items() if k != "lines"}
            out["lines"] = rows
            return out
        return {k: _redact_strings(v, needles, protected_lines, counter)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_strings(v, needles, protected_lines, counter) for v in obj]
    if isinstance(obj, str):
        low = obj.lower()
        if any(x in low for x in needles):
            counter["redacted_values"] += 1
            return "[redacted: signal name]"
    return obj


def view_for_arm(case, arm, retrieved=None):
    """The sections one arm is allowed to see, plus what was removed.

    L1  status fields and metrics only (L1_SECTIONS, scalars only at render).
    L2  everything except `signals`, with the case's covering signal names
        scrubbed from the remaining text.
    L3  everything.
    L4  everything, plus whatever a bounded retrieval round pulled in.
    """
    bundle = case.get("bundle") or {}
    truth = case.get("truth") or {}
    secs = _sections(bundle)
    note = {"arm": arm, "sections": [], "removed": [], "redacted_lines": 0,
            "redacted_values": 0, "skipped_load_bearing": 0}
    if arm == "L1":
        keep = {k: v for k, v in secs.items() if k in L1_SECTIONS}
        note["removed"] = sorted(set(secs) - set(keep))
        view = {"case_id": case.get("case_id"), "sections": keep,
                "scalars_only": True, "note": note}
    elif arm == "L2":
        keep = {k: v for k, v in secs.items() if k != "signals"}
        note["removed"] = ["signals"] if "signals" in secs else []
        needles = sorted({str(s).strip().lower()
                          for s in (truth.get("signals_expected") or [])
                          if str(s).strip()})
        protected = {(_artifact_key(l.get("artifact")), int(_num(l.get("line"), -1) or -1))
                     for l in (truth.get("load_bearing_lines") or [])
                     if isinstance(l, dict)}
        counter = {"redacted_lines": 0, "redacted_values": 0,
                   "skipped_load_bearing": 0}
        if needles:
            keep = _redact_strings(keep, needles, protected, counter)
        note.update(counter)
        note["covering_signals"] = needles
        view = {"case_id": case.get("case_id"), "sections": keep,
                "scalars_only": False, "note": note}
    else:
        view = {"case_id": case.get("case_id"), "sections": dict(secs),
                "scalars_only": False, "note": note}
    if retrieved:
        view["sections"] = dict(view["sections"])
        view["sections"]["retrieved"] = retrieved
        note["retrieved"] = [r.get("artifact_id") for r in retrieved]
    note["sections"] = sorted(view["sections"].keys())
    return view


# A section rendered as key: value must still show the ledger's step statuses,
# which live two dicts and one list deep. Depth and width are capped so a long
# trace cannot push the prompt past num_ctx by itself.
_RENDER_MAX_ITEMS = 50
_RENDER_MAX_DEPTH = 6


def _render_scalar(key, value, out, prefix="", depth=0):
    """Flatten one value into `prefix.key: value` lines, recursing into lists.

    A line-addressed dict (an artifact with `lines`) is summarised rather than
    flattened: its lines are rendered by the caller with their absolute
    numbers, and flattening them here would emit the same text at an address no
    citation can use.
    """
    path = (prefix + str(key)) if key != "" else prefix.rstrip(".")
    if depth > _RENDER_MAX_DEPTH:
        out.append("%s: (nested deeper than %d)" % (path, _RENDER_MAX_DEPTH))
        return
    if isinstance(value, dict):
        if isinstance(value.get("lines"), list) and value.get("artifact_id"):
            out.append("%s: artifact %s, %d line(s) rendered above"
                       % (path, value.get("artifact_id"), len(value["lines"])))
            return
        for k in sorted(value):
            _render_scalar(k, value[k], out, path + ".", depth + 1)
        return
    if isinstance(value, (list, tuple)):
        items = list(value)
        flat = [v for v in items if not isinstance(v, (dict, list, tuple))]
        if len(flat) == len(items):
            if flat:
                out.append("%s: %s" % (path, ", ".join(str(v) for v in flat)))
            return
        for i, v in enumerate(items[:_RENDER_MAX_ITEMS]):
            _render_scalar(i, v, out, path + ".", depth + 1)
        if len(items) > _RENDER_MAX_ITEMS:
            out.append("%s: ... %d more item(s) not rendered"
                       % (path, len(items) - _RENDER_MAX_ITEMS))
        return
    out.append("%s: %s" % (path, value))


def render_prompt(view, task=PROMPT_TASK, retrieval=False):
    """One prompt string from a view. Deterministic: same view, same bytes.

    Line-addressed sections render as `%6d\\t%s` with the artifact's absolute
    line numbers, which is the address a quote is later resolved to. Under
    `scalars_only` (L1) those arrays are dropped entirely rather than
    summarised — L1's claim is that status fields are enough, and leaving it a
    sample of the raw file would answer that question by accident.
    """
    parts = [task]
    if retrieval:
        parts.append(PROMPT_RETRIEVAL_NOTE)
    parts.append("case: %s" % view.get("case_id"))
    scalars_only = bool(view.get("scalars_only"))
    secs = view.get("sections") or {}
    order = [s for s in SECTION_ORDER if s in secs]
    order += [s for s in sorted(secs) if s not in order]
    for name in order:
        sec = secs[name]
        parts.append("")
        parts.append("== %s ==" % name)
        arts = _as_line_artifacts(sec)
        if arts:
            if scalars_only:
                parts.append("(withheld: raw lines are not a status field)")
                continue
            for art in arts:
                parts.append("artifact: %s  sha256:%s" % (art["artifact_id"],
                                                          (art["sha256"] or "")[:12]))
                for n, text in art["rows"]:
                    parts.append("%6d\t%s" % (n, text))
            continue
        if isinstance(sec, dict):
            body = []
            for k in sorted(sec):
                _render_scalar(k, sec[k], body)
            parts.extend(body)
        elif isinstance(sec, list):
            for i, row in enumerate(sec[:200]):
                if isinstance(row, dict):
                    # sacct rows are status fields and L1 is entitled to them;
                    # rendering them as key: value keeps that arm's view free of
                    # anything that looks like a quotable artifact line.
                    if scalars_only:
                        body = []
                        _render_scalar(i, row, body)
                        parts.extend(body)
                    else:
                        parts.append(_canonical(row))
                elif isinstance(row, (list, tuple)):
                    if not scalars_only:
                        parts.append(_canonical(row))
                else:
                    parts.append(str(row))
        else:
            parts.append(str(sec))
    return "\n".join(parts) + "\n"


def _json_from_text(text):
    """First balanced JSON object in a model's reply, or None.

    Models wrap JSON in prose and fences even under a JSON mode; a scorer that
    treats that as a refusal measures the harness, not the model.
    """
    s = str(text or "")
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


def empty_verdict(reason=""):
    v = {"verdict": "ok", "findings": [], "corrections": [],
         "escalate": {"to": "none", "reason": reason}, "confidence": 0.0}
    return v


def normalise_verdict(obj):
    """Coerce a model reply into the WP4 verdict shape without inventing content.

    Missing lists become empty, an unknown verdict word becomes `ok` (an arm
    that cannot say "issue" in the agreed vocabulary has not raised one), and
    the original is kept under `_raw_keys` so a malformed reply is still
    visible in the committed verdict file.
    """
    if not isinstance(obj, dict):
        return empty_verdict("no JSON object in reply")
    v = empty_verdict()
    word = str(obj.get("verdict") or "").strip().lower()
    v["verdict"] = "issue" if word == "issue" else "ok"
    findings = []
    for f in (obj.get("findings") or []):
        if not isinstance(f, dict):
            continue
        findings.append({"signal": str(f.get("signal") or ""),
                         "quote": str(f.get("quote") or ""),
                         "diagnosis": str(f.get("diagnosis") or ""),
                         "severity": str(f.get("severity") or "info").lower()})
    v["findings"] = findings
    corrections = []
    for c in (obj.get("corrections") or []):
        if not isinstance(c, dict):
            continue
        params = c.get("params")
        corrections.append({"action": str(c.get("action") or ""),
                            "params": params if isinstance(params, dict) else {},
                            "risk": str(c.get("risk") or ""),
                            "reason": str(c.get("reason") or ""),
                            "quote": str(c.get("quote") or "")})
    v["corrections"] = corrections
    esc = obj.get("escalate")
    if isinstance(esc, dict):
        to = str(esc.get("to") or "none").strip().lower()
        v["escalate"] = {"to": to if to in ("none", "tier1", "tier2", "human") else "none",
                         "reason": str(esc.get("reason") or "")}
    conf = _num(obj.get("confidence"), 0.0)
    v["confidence"] = max(0.0, min(1.0, conf if conf is not None else 0.0))
    if isinstance(obj.get("retrieve"), list):
        v["retrieve"] = [str(x) for x in obj["retrieve"]][:8]
    extra = sorted(k for k in obj
                   if k not in ("verdict", "findings", "corrections", "escalate",
                                "confidence", "retrieve"))
    if extra:
        v["_raw_keys"] = extra
    return v


# ---------------------------------------------------------------- arms --

# The ledger writes `at` as an ISO-8601 second, the pause writes an epoch float
# from the same tick; a pause can therefore read a few seconds "before" the
# failure it followed. Ordering is enforced with that tolerance, not exactly.
A0_CLOCK_TOL_S = 5.0


def _record(case, arm, repeat, verdict, meta):
    """One row of the case x arm x repeat matrix, as committed to disk."""
    return {"case_id": case.get("case_id"), "arm": arm, "repeat": int(repeat),
            "model": str((meta or {}).get("model") or ""),
            "verdict": verdict, "meta": meta or {},
            "bundle_sha256": case.get("bundle_sha256", ""),
            "rubric_sha256": rubric_sha256(), "ts": time.time()}


def a0_decision(bundle):
    """The scripted scheduler's detection, exactly as the code defines it.

    Fires iff the bundle's own step is recorded `failed` on the ledger — head
    entry or an attempt, not a superseded cancel — **and** the domain records a
    pause within one tick of that record. Both halves are load bearing:

      * A step recorded `done` never fires, which is every silent-wrongness
        case in the corpus (a COMPLETED harvest booked failed, a gate that did
        nothing, a stale artifact, a permuted class order).
      * A first failure with no pause never fires, which is the 2026-08-29
        behaviour: TIMEOUT recorded, identical command resubmitted on the next
        tick, and only the second failure tripped the stop-loss. A0 costs a
        second 12 h H100 job before it detects anything.

    `undecidable` marks a case the export cannot answer (no ledger, no
    timestamps). Those are counted and printed, never quietly scored as misses:
    an under-reported baseline flatters every arm measured against it.
    """
    lv = ledger_view(bundle)
    d = {"fired": False, "undecidable": False, "reason": "",
         "round": bundle.get("round"), "step": bundle.get("step"),
         "failed_at": None, "paused_at": None, "dt_s": None,
         "tick_s": lv["tick_s"], "pause_reason": ""}
    if not lv["present"]:
        d.update(undecidable=True, reason="bundle carries no ledger section")
        return d
    if not d["step"]:
        d.update(undecidable=True, reason="bundle names no step")
        return d
    entries = step_entries(bundle)
    if not entries:
        d["reason"] = ("step %s of round %s has no ledger entry"
                       % (d["step"], d["round"]))
        return d
    failed = [e for e in entries
              if str(e.get("status")) == "failed" and not _is_superseded(e)]
    if not failed:
        seen = ",".join(sorted({str(e.get("status")) for e in entries}))
        d["reason"] = "step %s recorded %s, not failed" % (d["step"], seen)
        return d
    stamps = [t for t in (_ts(e.get("at")) for e in failed) if t is not None]
    if not stamps:
        d.update(undecidable=True,
                 reason="failed entry carries no parseable timestamp")
        return d
    d["failed_at"] = max(stamps)
    pause = lv["pause"]
    if not pause:
        d["reason"] = "step failed but the domain is not recorded paused"
        return d
    d["pause_reason"] = pause.get("reason", "")
    if pause.get("at") is None:
        d.update(undecidable=True,
                 reason="pause carries no parseable timestamp")
        return d
    d["paused_at"] = pause["at"]
    d["dt_s"] = d["paused_at"] - d["failed_at"]
    if d["dt_s"] < -A0_CLOCK_TOL_S:
        d["reason"] = ("pause precedes the failure record by %.0fs — it belongs "
                       "to an earlier incident" % abs(d["dt_s"]))
        return d
    if d["dt_s"] > d["tick_s"]:
        d["reason"] = ("pause %.0fs after the failure record, more than one "
                       "tick (%.0fs)" % (d["dt_s"], d["tick_s"]))
        return d
    d["fired"] = True
    d["reason"] = ("step %s recorded failed and the domain paused %.0fs later"
                   % (d["step"], d["dt_s"]))
    return d


def arm_a0(case, ctx, repeat=0):
    """A0 — the scripted scheduler. No signals, no model, no notification."""
    bundle = case.get("bundle") or {}
    d = a0_decision(bundle)
    if d["fired"]:
        v = {"verdict": "issue",
             "findings": [{"signal": "stop_loss", "quote": "",
                           "diagnosis": d["reason"], "severity": "crit"}],
             "corrections": [],
             "escalate": {"to": ESCALATION_POLICY["A0"]["to"],
                          "reason": "the scripted loop pauses; it notifies nobody"},
             "confidence": 1.0}
    else:
        v = empty_verdict(d["reason"])
    meta = {"deterministic": True, "notifies": False, "notify_latency_s": None,
            "notify_latency_note": "no notification channel",
            "su": 0.0, "su_measured": False, "tokens_in": 0, "tokens_out": 0,
            "latency_s": 0.0, "a0": d, "undecidable": d["undecidable"],
            "model": ""}
    return _record(case, "A0", repeat, v, meta)


def _resolve_correction_params(bundle, params):
    """Turn a symbolic deterministic correction into the numbers it would write.

    WP1's fallback halves the epochs of a walltime-shaped failure. Scored
    against a truth range that is numeric, the symbol `half` matches nothing,
    so it is resolved against the value the step actually ran with. When the
    export does not carry that value the symbol is left in place and the
    correction is marked unresolved — it then fails the range check, which is
    the honest outcome for a correction whose target is unknown.
    """
    out, unresolved = {}, []
    for k, v in (params or {}).items():
        if v == "half":
            cur = _current_param(bundle, k)
            if cur is None:
                out[k] = v
                unresolved.append(k)
            else:
                out[k] = max(1, int(cur // 2))
        else:
            out[k] = v
    return out, unresolved


def _signal_evidence_ts(sig):
    for key in ("ts", "at", "fired_ts", "detected_at"):
        t = _ts(sig.get(key))
        if t is not None:
            return t
    for ev in (sig.get("evidence") or []):
        if isinstance(ev, dict):
            t = _ts(ev.get("ts"))
            if t is not None:
                return t
    return None


def _step_start_ts(bundle):
    """When the bundle's own step started, from the ledger's `running` entry."""
    stamps = [t for t in (_ts(e.get("at")) for e in step_entries(bundle)
                          if str(e.get("status")) == "running") if t is not None]
    return min(stamps) if stamps else None


def _notify_latency(case, fired, notify):
    """Replayed seconds from the incident's onset to the alarm reaching a person.

    The zero point is stated with the number, because there are two and they
    answer different questions: `truth.incident_ts` when the corpus states one,
    otherwise the step's own start on the ledger (time-to-detect within the
    step, which is what the 20 % vs 100 % of walltime comparison measures).
    The step's *failure* record is deliberately not used as an onset: a signal
    that fires 13 minutes before the walltime kill would then read as a latency
    of zero, turning early detection into no detection.

    A negative result means the onset is wrong for this case, and is reported as
    such rather than clamped to zero. The value includes ALARM_SURFACE_S and is
    a replay estimate, not a measurement — the measurement is the live round.
    """
    if not notify:
        return None, "no notification channel"
    if not fired:
        return None, "nothing fired"
    onset = _ts((case.get("truth") or {}).get("incident_ts"))
    source = "truth.incident_ts"
    if onset is None:
        onset = _step_start_ts(case.get("bundle") or {})
        source = "the step's start on the ledger"
    if onset is None:
        return None, ("no onset: truth carries no incident_ts and the ledger "
                      "tail carries no start for this step")
    stamps = [t for t in (_signal_evidence_ts(s) for s in fired) if t is not None]
    detect = min(stamps) if stamps else _ts((case.get("bundle") or {}).get("built_ts"))
    if detect is None:
        return None, "no detection timestamp in the bundle"
    latency = detect + ALARM_SURFACE_S - onset
    if latency < 0:
        return None, ("detection precedes the onset taken from %s, so that "
                      "onset is wrong for this case" % source)
    return latency, "replay from %s" % source


def arm_signals(case, ctx, repeat=0, notify=False):
    """A0+ (notify False) and A0++ (notify True) — deterministic signals.

    Same findings and same corrections; the only difference is that A0++ has a
    channel, which is why it is the arm that has a notification latency at all.
    """
    arm = "A0pp" if notify else "A0p"
    bundle = case.get("bundle") or {}
    info = signals_for(bundle)
    fired = [s for s in info["signals"] if _at_or_above(s.get("severity"))]
    findings, corrections = [], []
    for s in fired:
        ev = [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
        findings.append({"signal": str(s.get("signal") or ""),
                         "quote": str((ev[0].get("quote") if ev else "") or ""),
                         "diagnosis": "deterministic signal (value %s)"
                                      % (s.get("value"),),
                         "severity": str(s.get("severity") or "warn")})
        rule = DETERMINISTIC_CORRECTIONS.get(str(s.get("signal") or ""))
        if rule:
            params, unresolved = _resolve_correction_params(bundle, rule["params"])
            c = {"action": rule["action"], "params": params,
                 "risk": rule.get("risk", "R1"),
                 "reason": "pre-registered deterministic correction for %s"
                           % s.get("signal"),
                 "quote": findings[-1]["quote"]}
            if unresolved:
                c["unresolved_params"] = unresolved
            if not any(x["action"] == c["action"] and x["params"] == c["params"]
                       for x in corrections):
                corrections.append(c)
    pol = ESCALATION_POLICY[arm]
    to = pol["to"] if (fired and pol["notifies"]) else "none"
    v = {"verdict": "issue" if fired else "ok", "findings": findings,
         "corrections": corrections,
         "escalate": {"to": to,
                      "reason": "signals fired" if fired else "no signal fired"},
         "confidence": 1.0 if fired else 0.0}
    latency, latency_note = _notify_latency(case, fired, pol["notifies"])
    meta = {"deterministic": True, "notifies": pol["notifies"],
            "notify_latency_s": latency, "notify_latency_note": latency_note,
            "su": 0.0, "su_measured": False, "tokens_in": 0, "tokens_out": 0,
            "latency_s": 0.0, "model": "",
            "signals_source": info["source"], "signals_n": len(info["signals"]),
            "degraded": info["degraded"], "degraded_reason": info["reason"]}
    return _record(case, arm, repeat, v, meta)


def _estimate_tokens(text):
    """Characters/4. Crude on purpose: it is a guard, not an accounting.

    The number that matters is the provider's own `tokens_in`, which the
    callable returns; this estimate exists so a prompt that will be silently
    truncated is flagged before the call rather than inferred from a bad
    answer afterwards.
    """
    return max(1, len(str(text or "")) // 4)


_LINE_RE = re.compile(r"^\s*(\d+)\t(.*)$")


def retrieve_artifacts(case, names):
    """One bounded retrieval round over the case's artifact files.

    Returns (artifacts, refused). Each artifact is the same line-addressed
    shape the bundle uses, so a quote taken from retrieved text resolves like
    any other. Files are read from their tail (the last RETRIEVAL_MAX_LINES
    lines) and say so, because the trimmer that produced the bundle trimmed the
    same end.
    """
    arts, refused = [], []
    have = case.get("artifacts") or {}
    for name in (names or [])[:4]:
        key = _artifact_key(name)
        path = have.get(key)
        if not path:
            refused.append(str(name))
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read().splitlines()
        except Exception as e:
            refused.append("%s (unreadable: %s)" % (name, e))
            continue
        total = len(raw)
        tail = raw[-RETRIEVAL_MAX_LINES:]
        rows = []
        for i, line in enumerate(tail):
            m = _LINE_RE.match(line)
            if m:
                rows.append([int(m.group(1)), m.group(2)])
            else:
                rows.append([total - len(tail) + i + 1, line])
        arts.append({"artifact_id": key, "path": path, "sha256": "",
                     "note": "last %d of %d lines" % (len(rows), total),
                     "lines": rows})
    return arts, refused


def arm_model(case, ctx, arm, repeat=0):
    """L1-L4 — one model call (L4: at most two) through the pluggable callable.

    The callable is the integrator's: (prompt, model_id, num_ctx) ->
    {text, tokens_in, tokens_out, latency_s, su}. Anything it raises is
    recorded as a failed review, which scores as no detection — a provider
    outage is a real supervision outcome and must not vanish from the matrix.
    """
    model_fn = ctx.get("model_fn")
    model_id = str(ctx.get("model") or "")
    num_ctx = int(_num(ctx.get("num_ctx"), 32768) or 32768)
    base_meta = {"deterministic": False, "notifies": True, "model": model_id,
                 "num_ctx": num_ctx, "arm": arm, "su": 0.0, "tokens_in": 0,
                 "tokens_out": 0, "latency_s": 0.0, "calls": 0,
                 "retrieval_rounds": 0, "retrieval_refused": [],
                 "notify_latency_s": None,
                 "notify_latency_note": "not measured for model arms"}
    if not callable(model_fn):
        base_meta.update(skipped="no model callable supplied")
        return _record(case, arm, repeat, empty_verdict("no model callable"),
                       base_meta)

    def _call(prompt):
        base_meta["calls"] += 1
        est = _estimate_tokens(prompt)
        base_meta["token_estimate"] = base_meta.get("token_estimate", 0) + est
        if est > num_ctx:
            # Silent truncation is the defect class under study; a review whose
            # prompt did not fit is reported, never averaged in as a miss.
            base_meta["context_overflow"] = True
        try:
            res = model_fn(prompt, model_id, num_ctx)
        except Exception as e:
            base_meta["model_error"] = "%s: %s" % (type(e).__name__, e)
            return None
        if not isinstance(res, dict):
            base_meta["model_error"] = "callable returned %s" % type(res).__name__
            return None
        base_meta["tokens_in"] += int(_num(res.get("tokens_in"), 0) or 0)
        base_meta["tokens_out"] += int(_num(res.get("tokens_out"), 0) or 0)
        base_meta["latency_s"] += float(_num(res.get("latency_s"), 0.0) or 0.0)
        base_meta["su"] += float(_num(res.get("su"), 0.0) or 0.0)
        return str(res.get("text") or "")

    view = view_for_arm(case, arm)
    base_meta["view"] = view["note"]
    prompt = render_prompt(view, retrieval=(arm == "L4"))
    base_meta["prompt_sha256"] = _sha256(prompt)
    base_meta["prompt_chars"] = len(prompt)
    text = _call(prompt)
    if text is None:
        return _record(case, arm, repeat,
                       empty_verdict(base_meta.get("model_error", "model call failed")),
                       base_meta)
    obj = _json_from_text(text)
    if obj is None:
        base_meta["parse_error"] = "no JSON object in the reply"
    verdict = normalise_verdict(obj)

    if arm == "L4" and verdict.get("retrieve"):
        arts, refused = retrieve_artifacts(case, verdict["retrieve"])
        base_meta["retrieval_refused"] = refused
        if arts:
            base_meta["retrieval_rounds"] = RETRIEVAL_ROUNDS
            view2 = view_for_arm(case, arm, retrieved=arts)
            base_meta["view"] = view2["note"]
            prompt2 = render_prompt(view2, retrieval=False)
            base_meta["prompt_sha256_round2"] = _sha256(prompt2)
            text2 = _call(prompt2)
            if text2 is not None:
                obj2 = _json_from_text(text2)
                if obj2 is None:
                    base_meta["parse_error"] = "no JSON object in the retrieval reply"
                else:
                    base_meta.pop("parse_error", None)
                    verdict = normalise_verdict(obj2)
                    # The second prompt carries the retrieved lines, so quotes
                    # from them must resolve: score against that bundle.
                    base_meta["retrieved_artifacts"] = [a["artifact_id"] for a in arts]
    return _record(case, arm, repeat, verdict, base_meta)


def run_arm(arm, case, ctx, repeat=0):
    """Dispatch one arm over one case. Unknown arm ids are reported, not raised."""
    if arm == "A0":
        return arm_a0(case, ctx, repeat)
    if arm == "A0p":
        return arm_signals(case, ctx, repeat, notify=False)
    if arm == "A0pp":
        return arm_signals(case, ctx, repeat, notify=True)
    if arm in MODEL_ARMS:
        return arm_model(case, ctx, arm, repeat)
    return _record(case, arm, repeat, empty_verdict("unknown arm"),
                   {"skipped": "unknown arm %r" % arm, "deterministic": True,
                    "su": 0.0, "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0})


def is_deterministic(arm):
    """Arms whose output cannot vary between repeats (so repeats are copied)."""
    return arm in ("A0", "A0p", "A0pp")


# ---------------------------------------------------------------- scoring --

def _resolvable_lines(case, meta):
    """Lines a quote may resolve into: the bundle, plus whatever L4 retrieved.

    A retrieval round puts real artifact lines in front of the model, so a
    quote from them is a valid citation. Re-reading the file at scoring time
    keeps `--reproduce` exact: the corpus is frozen, so the same bytes come
    back.
    """
    lines = bundle_lines(case.get("bundle") or {})
    names = (meta or {}).get("retrieved_artifacts") or []
    if names:
        arts, _ = retrieve_artifacts(case, names)
        for art in arts:
            for row in art["lines"]:
                lines.append({"artifact_id": art["artifact_id"], "line": int(row[0]),
                              "text": str(row[1]), "section": "retrieved"})
    return lines


def _in_range(value, spec):
    """Is a proposed parameter inside the truth's allowed range?

    Accepted spec forms, in the order they are tried:
      {"range":[lo,hi]} / {"min":lo,"max":hi}   numeric, inclusive
      {"in":[...]}                              membership
      [lo,hi] of two numbers                    numeric, inclusive
      any other list                            membership
      scalar                                    equality
    A two-number list is a range, so an enum of exactly two numbers must be
    written {"in":[a,b]}. That ambiguity is resolved here once rather than per
    case.
    """
    v = value
    if isinstance(spec, dict):
        if "in" in spec:
            return any(_same(v, x) for x in (spec["in"] or []))
        rng = spec.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            return _between(v, rng[0], rng[1])
        if "min" in spec or "max" in spec:
            return _between(v, spec.get("min"), spec.get("max"))
        return False
    if isinstance(spec, (list, tuple)):
        nums = [x for x in spec if _num(x) is not None]
        if len(spec) == 2 and len(nums) == 2:
            return _between(v, spec[0], spec[1])
        return any(_same(v, x) for x in spec)
    return _same(v, spec)


def _same(a, b):
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-9
    return str(a).strip().lower() == str(b).strip().lower()


def _between(v, lo, hi):
    n = _num(v)
    if n is None:
        return False
    nlo, nhi = _num(lo), _num(hi)
    if nlo is not None and n < nlo:
        return False
    if nhi is not None and n > nhi:
        return False
    return True


def correction_match(proposals, acceptable):
    """(matched, note) for the R1/R2-correctable subset.

    A proposal matches when its action is one of the accepted actions and every
    parameter the truth constrains is present and inside its range. Parameters
    the truth does not constrain are allowed and named in the note, so a
    correction that quietly also changed something is visible.
    """
    if not acceptable:
        return False, "no acceptable correction in the R1/R2 subset"
    notes = []
    for p in (proposals or []):
        act = str(p.get("action") or "").strip().lower()
        params = p.get("params") if isinstance(p.get("params"), dict) else {}
        for a in acceptable:
            if act != str(a.get("action") or "").strip().lower():
                continue
            ranges = a.get("params_range") if isinstance(a.get("params_range"), dict) else {}
            missing = [k for k in ranges if k not in params]
            if missing:
                notes.append("%s: missing %s" % (act, ",".join(sorted(missing))))
                continue
            bad = [k for k, spec in ranges.items() if not _in_range(params[k], spec)]
            if bad:
                notes.append("%s: out of range %s" % (act, ",".join(sorted(bad))))
                continue
            extra = sorted(set(params) - set(ranges))
            return True, ("matched %s%s" % (act, (" (unconstrained params: %s)"
                                                  % ",".join(extra)) if extra else ""))
        if not any(act == str(a.get("action") or "").strip().lower() for a in acceptable):
            notes.append("%s: action not in the acceptable set" % (act or "(blank)"))
    return False, "; ".join(notes) if notes else "no correction proposed"


def score_record(case, record, ctx=None):
    """Score one case x arm x repeat row against the case's truth.

    Every field here is a pre-registered definition, restated where it bites:
      fired      the arm said `issue` AND carried a finding at
                 DETECTION_SEVERITY or above. `issue` with no finding is
                 counted as `fired_unsupported` and is not a detection: there
                 is nothing to act on and nothing to check.
      evidence   case-level, "did the arm quote at least one load-bearing
                 line". Lines inside one case are not independent, so the
                 line-level pool is reported beside it and is never the number
                 an interval is put on.
      citations  pooled over findings that carry a quote (and averaged over
                 repeats first, so the denominator stays in units of findings
                 per case). Findings with no
                 quote are counted separately: A0 cites nothing by design and
                 would otherwise read as an arm with perfect citation validity
                 or with none, depending on which way the ratio was defined.
    """
    truth = case.get("truth") or {}
    verdict = record.get("verdict") or {}
    meta = record.get("meta") or {}
    findings = [f for f in (verdict.get("findings") or []) if isinstance(f, dict)]
    incident = bool(truth.get("incident"))
    klass = str(truth.get("class") or "unclassified")
    sev_ok = [f for f in findings if _at_or_above(f.get("severity"))]
    said_issue = str(verdict.get("verdict") or "").lower() == "issue"
    fired = bool(said_issue and sev_ok)

    lines = _resolvable_lines(case, meta)
    lb = {(_artifact_key(l.get("artifact")), int(_num(l.get("line"), -1) or -1))
          for l in (truth.get("load_bearing_lines") or []) if isinstance(l, dict)}
    cited, valid, no_quote = set(), 0, 0
    for f in findings:
        q = str(f.get("quote") or "")
        if not q.strip():
            no_quote += 1
            continue
        hit = resolve_quote(case.get("bundle") or {}, q, lines=lines)
        if hit:
            valid += 1
            cited.add((hit["artifact_id"], int(hit["line"])))
    quoted = len(findings) - no_quote
    hits = cited & lb

    accept = [c for c in (truth.get("acceptable_corrections") or [])
              if isinstance(c, dict)
              and str(c.get("risk") or "").strip() in CORRECTABLE_RISKS]
    correctable = bool(accept)
    corr_ok, corr_note = (False, "case is not in the R1/R2-correctable subset")
    if correctable:
        corr_ok, corr_note = correction_match(verdict.get("corrections") or [], accept)

    expected_esc = str(truth.get("escalation_expected") or "none").strip().lower()
    emitted_esc = str(((verdict.get("escalate") or {}).get("to")) or "none").strip().lower()

    want_sigs = {str(s).strip().lower()
                 for s in (truth.get("signals_expected") or []) if str(s).strip()}
    named = {str(f.get("signal") or "").strip().lower() for f in findings}

    return {
        "case_id": case.get("case_id"), "arm": record.get("arm"),
        "repeat": int(record.get("repeat", 0)), "model": record.get("model", ""),
        "class": klass, "incident": incident,
        "provenance": str(truth.get("provenance") or ""),
        "fired": fired,
        "fired_unsupported": bool(said_issue and not sev_ok),
        "detected": bool(fired and incident),
        "false_alarm": bool(fired and not incident),
        "detected_grounded": bool(fired and incident and valid > 0),
        "citations_total": quoted, "citations_valid": valid,
        "findings_without_quote": no_quote, "findings_n": len(findings),
        "evidence_total": len(lb), "evidence_hits": len(hits),
        "evidence_hit": bool(lb and hits),
        "correctable": correctable, "correction_correct": bool(corr_ok),
        "correction_proposed": bool(verdict.get("corrections")),
        "correction_note": corr_note,
        "signals_expected_n": len(want_sigs),
        "signals_named": len(want_sigs & named),
        "escalation_expected": expected_esc, "escalation_emitted": emitted_esc,
        "escalation_ok": bool(expected_esc == emitted_esc),
        "su": float(_num(meta.get("su"), 0.0) or 0.0),
        "tokens_in": float(_num(meta.get("tokens_in"), 0.0) or 0.0),
        "tokens_out": float(_num(meta.get("tokens_out"), 0.0) or 0.0),
        "latency_s": float(_num(meta.get("latency_s"), 0.0) or 0.0),
        "notify_latency_s": _num(meta.get("notify_latency_s")),
        "undecidable": bool(meta.get("undecidable")),
        "degraded": bool(meta.get("degraded")),
        "skipped": str(meta.get("skipped") or ""),
        "parse_error": str(meta.get("parse_error") or ""),
        "model_error": str(meta.get("model_error") or ""),
        "context_overflow": bool(meta.get("context_overflow")),
    }


_BOOL_FIELDS = ("fired", "fired_unsupported", "detected", "false_alarm",
                "detected_grounded", "evidence_hit", "correction_correct",
                "correction_proposed", "escalation_ok", "undecidable",
                "degraded", "context_overflow")
# Counting fields are averaged over repeats, not summed: a pooled denominator
# must stay in units of findings-per-case, or three repeats would triple every
# citation-validity n and narrow its interval by sqrt(3) for nothing.
_COUNT_FIELDS = ("citations_total", "citations_valid", "findings_without_quote",
                 "findings_n", "evidence_hits", "signals_named")
_MEAN_FIELDS = ("su", "tokens_in", "tokens_out", "latency_s", "notify_latency_s")


def collapse_repeats(rows):
    """One row per case from its repeats: majority for decisions, pooled counts.

    n is the number of cases everywhere downstream. Three repeats of one case
    are three looks at the same evidence, not three cases, and counting them as
    independent would shrink every interval by sqrt(3) for free. The repeats
    are still used: they are pooled for the counting metrics and their
    disagreement is reported as `repeat_disagreement`.
    """
    if not rows:
        return {}
    out = dict(rows[0])
    n = len(rows)
    disagree = []
    for f in _BOOL_FIELDS:
        vals = [bool(r.get(f)) for r in rows]
        out[f] = (sum(1 for v in vals if v) * 2 > n)
        if len(set(vals)) > 1:
            disagree.append(f)
    for f in _COUNT_FIELDS:
        out[f] = sum(float(_num(r.get(f), 0.0) or 0.0) for r in rows) / float(n)
    for f in _MEAN_FIELDS:
        vals = [_num(r.get(f)) for r in rows]
        vals = [v for v in vals if v is not None]
        out[f] = (sum(vals) / len(vals)) if vals else None
    # evidence_total and the truth-derived fields are properties of the case.
    out["repeats"] = n
    out["repeat_disagreement"] = sorted(disagree)
    out["repeat"] = -1
    notes = sorted({str(r.get("correction_note") or "") for r in rows})
    out["correction_note"] = "; ".join([x for x in notes if x])[:400]
    return out


# ---------------------------------------------------------------- statistics --

Z_95 = 1.959963984540054


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _zq(p):
    """Inverse standard normal by bisection. Stdlib only, accurate to ~1e-12."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilson(k, n, z=Z_95):
    """Wilson score interval for k successes in n trials.

    Reference values (95 %): 0/10 -> (0.0000, 0.2775); 1/10 -> (0.0179,
    0.4042); 5/10 -> (0.2366, 0.7634); 10/10 -> (0.7225, 1.0000). The normal
    approximation puts 0/10 at exactly (0, 0), which is how a corpus of a dozen
    cases is made to look certain.
    """
    n = int(_num(n, 0) or 0)
    k = int(_num(k, 0) or 0)
    if n <= 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    k = max(0, min(k, n))
    p = k / float(n)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return {"k": k, "n": n, "p": p, "lo": max(0.0, centre - half),
            "hi": min(1.0, centre + half)}


def newcombe_diff(k1, n1, k0, n0, z=Z_95):
    """Newcombe hybrid-score interval for p1 - p0 (independent samples).

    The arms here are paired (every arm sees every case), so this interval is
    conservative. It is used anyway, and said to be conservative, because a
    paired interval at n = 12 rests on the discordant-pair count, which at that
    size is often 1 or 2 — an interval computed from two observations is not
    more honest for being narrower.
    """
    a, b = wilson(k1, n1, z), wilson(k0, n0, z)
    if a["p"] is None or b["p"] is None:
        return {"delta": None, "lo": None, "hi": None}
    d = a["p"] - b["p"]
    lo = d - math.sqrt((a["p"] - a["lo"]) ** 2 + (b["hi"] - b["p"]) ** 2)
    hi = d + math.sqrt((a["hi"] - a["p"]) ** 2 + (b["p"] - b["lo"]) ** 2)
    return {"delta": d, "lo": max(-1.0, lo), "hi": min(1.0, hi)}


def power_two_proportions(n, p0, p1, alpha=0.05):
    """Power of a two-sided two-proportion test at n per arm (normal approx)."""
    n = _num(n, 0) or 0
    if n <= 0:
        return 0.0
    za = _zq(1.0 - alpha / 2.0)
    pbar = (p0 + p1) / 2.0
    se0 = math.sqrt(max(0.0, 2.0 * pbar * (1.0 - pbar) / n))
    se1 = math.sqrt(max(0.0, (p0 * (1.0 - p0) + p1 * (1.0 - p1)) / n))
    if se1 <= 0.0:
        return 1.0 if abs(p1 - p0) > 0 else 0.0
    return _phi((abs(p1 - p0) - za * se0) / se1)


def mdd_proportion(n, p0=0.5, power=0.8, alpha=0.05):
    """Smallest difference in proportions detectable at n per arm.

    Anchor: n = 58 per arm at p0 = 0.5 returns 0.249, which is the textbook
    sample size for 0.50 vs 0.75 at 80 % power and two-sided alpha 0.05. At the
    sizes this corpus can reach it is brutal and that is the point — n = 12,
    p0 = 0.5 gives 0.478, so any margin under 48 points is not a result at that
    n and the renderer says so next to it.
    """
    n = _num(n, 0) or 0
    if n <= 0:
        return None
    p0 = min(max(float(p0), 0.0), 1.0)
    lo, hi = 0.0, 1.0 - p0
    if power_two_proportions(n, p0, p0 + hi, alpha) < power:
        return None                    # not detectable at any difference
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if power_two_proportions(n, p0, p0 + mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mdd_mean(n, sd, power=0.8, alpha=0.05):
    """Smallest difference in means detectable at n per arm (two-sided)."""
    n = _num(n, 0) or 0
    sd = _num(sd, 0.0) or 0.0
    if n <= 0 or sd <= 0:
        return None
    return (_zq(1.0 - alpha / 2.0) + _zq(power)) * sd * math.sqrt(2.0 / n)


def mean_sd(values):
    vals = [v for v in (_num(x) for x in (values or [])) if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None}
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return {"n": len(vals), "mean": m, "sd": 0.0}
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return {"n": len(vals), "mean": m, "sd": math.sqrt(var)}


# ---------------------------------------------------------------- aggregation --

# Every metric, its subset and the direction that is "better". `prop` metrics
# are one Bernoulli per case (n = cases). `ratio` metrics pool lines or
# findings, whose units are not independent within a case — their intervals are
# optimistic and the renderer labels them.
METRICS = (
    {"key": "detection_recall", "kind": "prop", "subset": "incident",
     "field": "detected", "better": "higher"},
    {"key": "detection_grounded", "kind": "prop", "subset": "incident",
     "field": "detected_grounded", "better": "higher"},
    {"key": "false_alarm_rate", "kind": "prop", "subset": "healthy",
     "field": "fired", "better": "lower"},
    {"key": "correction_correct", "kind": "prop", "subset": "correctable",
     "field": "correction_correct", "better": "higher"},
    {"key": "evidence_hit_rate", "kind": "prop", "subset": "incident_lines",
     "field": "evidence_hit", "better": "higher"},
    {"key": "evidence_line_rate", "kind": "ratio", "subset": "incident_lines",
     "num": "evidence_hits", "den": "evidence_total", "better": "higher"},
    {"key": "citation_validity", "kind": "ratio", "subset": "quoting",
     "num": "citations_valid", "den": "citations_total", "better": "higher"},
    {"key": "escalation_ok", "kind": "prop", "subset": "all",
     "field": "escalation_ok", "better": "higher"},
    {"key": "su_per_review", "kind": "mean", "subset": "all", "field": "su",
     "better": "lower"},
    {"key": "tokens_in", "kind": "mean", "subset": "all", "field": "tokens_in",
     "better": "lower"},
)


def _in_subset(row, subset):
    if subset == "all":
        return True
    if subset == "incident":
        return bool(row.get("incident"))
    if subset == "healthy":
        return not bool(row.get("incident"))
    if subset == "correctable":
        return bool(row.get("incident")) and bool(row.get("correctable"))
    if subset == "incident_lines":
        return bool(row.get("incident")) and (_num(row.get("evidence_total"), 0) or 0) > 0
    if subset == "quoting":
        return (_num(row.get("citations_total"), 0) or 0) > 0
    return False


def aggregate(case_rows, arm, model=""):
    """Per-class and overall statistics for one arm, from collapsed case rows."""
    groups = {}
    keys = ["all"] + sorted({str(r.get("class") or "unclassified") for r in case_rows})
    for gk in keys:
        rows = [r for r in case_rows
                if gk == "all" or str(r.get("class") or "unclassified") == gk]
        stats = {}
        for m in METRICS:
            sub = [r for r in rows if _in_subset(r, m["subset"])]
            if m["kind"] == "prop":
                k = sum(1 for r in sub if bool(r.get(m["field"])))
                st = wilson(k, len(sub))
                st["unit"] = "case"
            elif m["kind"] == "ratio":
                num = sum(_num(r.get(m["num"]), 0.0) or 0.0 for r in sub)
                den = sum(_num(r.get(m["den"]), 0.0) or 0.0 for r in sub)
                st = wilson(int(round(num)), int(round(den)))
                st["unit"] = "line-or-finding (not independent within a case)"
                st["cases"] = len(sub)
            else:
                st = mean_sd([r.get(m["field"]) for r in sub])
                st["unit"] = "case"
            st["subset"] = m["subset"]
            st["better"] = m["better"]
            st["kind"] = m["kind"]
            stats[m["key"]] = st
        groups[gk] = stats
    counts = {
        "cases": len(case_rows),
        "incidents": sum(1 for r in case_rows if r.get("incident")),
        "healthy": sum(1 for r in case_rows if not r.get("incident")),
        "undecidable": sum(1 for r in case_rows if r.get("undecidable")),
        "degraded": sum(1 for r in case_rows if r.get("degraded")),
        "skipped": sum(1 for r in case_rows if r.get("skipped")),
        "parse_errors": sum(1 for r in case_rows if r.get("parse_error")),
        "model_errors": sum(1 for r in case_rows if r.get("model_error")),
        "context_overflow": sum(1 for r in case_rows if r.get("context_overflow")),
        "fired_unsupported": sum(1 for r in case_rows if r.get("fired_unsupported")),
        "findings_without_quote": sum(int(_num(r.get("findings_without_quote"), 0) or 0)
                                      for r in case_rows),
        "repeat_disagreement": sum(1 for r in case_rows
                                   if r.get("repeat_disagreement")),
    }
    notify = [r.get("notify_latency_s") for r in case_rows
              if r.get("notify_latency_s") is not None]
    return {"arm": arm, "model": model, "groups": groups, "counts": counts,
            "notify_latency_s": mean_sd(notify) if notify else
                                {"n": 0, "mean": None, "sd": None}}


def compare(aggs, baseline=BASELINE_ARM):
    """Every arm as a margin over the baseline, with an MDD on every row.

    A row whose |delta| is below its MDD is marked `below_mdd`; the renderer
    prints that instead of the delta's sign, because at these corpus sizes the
    sign of a small margin is not information. An MDD of None — no difference
    at all is detectable at that n, which is the usual case below about 20
    cases per arm — also marks `below_mdd`: it is the strongest form of "not a
    result", and the row that says so must not read as the weakest.
    """
    rows = []
    base = aggs.get(baseline)
    if base is None:
        return rows
    base_degraded = bool((base.get("counts") or {}).get("degraded"))
    for arm, agg in sorted(aggs.items()):
        if arm == baseline:
            continue
        for gk, stats in sorted(agg.get("groups", {}).items()):
            bstats = (base.get("groups") or {}).get(gk) or {}
            for m in METRICS:
                a = stats.get(m["key"]) or {}
                b = bstats.get(m["key"]) or {}
                row = {"arm": arm, "baseline": baseline, "group": gk,
                       "metric": m["key"], "kind": m["kind"],
                       "better": m["better"], "unit": a.get("unit"),
                       "baseline_degraded": base_degraded}
                if m["kind"] in ("prop", "ratio"):
                    n1, n0 = a.get("n") or 0, b.get("n") or 0
                    row.update({"arm_n": n1, "arm_k": a.get("k"), "arm_p": a.get("p"),
                                "arm_ci": [a.get("lo"), a.get("hi")],
                                "baseline_n": n0, "baseline_k": b.get("k"),
                                "baseline_p": b.get("p"),
                                "baseline_ci": [b.get("lo"), b.get("hi")]})
                    if not n1 or not n0:
                        row.update({"delta": None, "delta_ci": [None, None],
                                    "mdd": None, "below_mdd": None,
                                    "note": "empty subset"})
                    else:
                        d = newcombe_diff(a.get("k"), n1, b.get("k"), n0)
                        mdd = mdd_proportion(min(n1, n0), b.get("p") or 0.0)
                        row.update({"delta": d["delta"],
                                    "delta_ci": [d["lo"], d["hi"]], "mdd": mdd,
                                    "below_mdd": (True if mdd is None
                                                  else abs(d["delta"]) < mdd)})
                else:
                    n1, n0 = a.get("n") or 0, b.get("n") or 0
                    row.update({"arm_n": n1, "arm_mean": a.get("mean"),
                                "arm_sd": a.get("sd"), "baseline_n": n0,
                                "baseline_mean": b.get("mean"),
                                "baseline_sd": b.get("sd")})
                    if not n1 or not n0 or a.get("mean") is None or b.get("mean") is None:
                        row.update({"delta": None, "delta_ci": [None, None],
                                    "mdd": None, "below_mdd": None,
                                    "note": "empty subset"})
                    else:
                        d = a["mean"] - b["mean"]
                        sd1, sd0 = a.get("sd") or 0.0, b.get("sd") or 0.0
                        se = math.sqrt(sd1 * sd1 / n1 + sd0 * sd0 / n0)
                        pooled = math.sqrt((sd1 * sd1 + sd0 * sd0) / 2.0)
                        mdd = mdd_mean(min(n1, n0), pooled)
                        if mdd is None and pooled == 0.0 and n1 > 1 and n0 > 1:
                            # Both arms constant across cases (SU per review of a
                            # deterministic arm, tokens for a fixed prompt): with
                            # no variance any non-zero difference is real, so the
                            # MDD is 0 rather than undefined. Saying "not
                            # detectable" about a difference measured without
                            # noise would bury the cost frontier.
                            mdd = 0.0
                        row.update({"delta": d,
                                    "delta_ci": [d - Z_95 * se, d + Z_95 * se],
                                    "mdd": mdd,
                                    "below_mdd": (True if mdd is None
                                                  else abs(d) < mdd)})
                rows.append(row)
    return rows


# ---------------------------------------------------------------- report --

def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    try:
        return ("%." + str(nd) + "f") % float(v)
    except (TypeError, ValueError):
        return str(v)


def render_report(result, metrics=None, group="all"):
    """Plain-text report. Every comparison line carries its MDD."""
    out = []
    add = out.append
    split = result.get("split") or {}
    add("supervision benchmark — %s" % result.get("run_id", ""))
    add("  corpus %s  rubric %s" % ((result.get("corpus_hash") or "")[:12],
                                    (result.get("rubric_sha256") or "")[:12]))
    add("  split %s (%s cases, sha %s, self-hash %s)"
        % (split.get("name"), split.get("n_cases"),
           (split.get("sha256") or "")[:12],
           {True: "ok", False: "MISMATCH", None: "absent"}.get(split.get("sha_ok"))))
    add("  repeats %s  temperature %s  model %s"
        % (result.get("repeats"), result.get("temperature"),
           result.get("model") or "(none)"))
    for w in result.get("warnings") or []:
        add("  WARNING: %s" % w)
    for e in result.get("errors") or []:
        add("  ERROR: %s" % e)
    add("")
    add("per-arm counts")
    for arm in result.get("arms") or []:
        agg = (result.get("per_arm") or {}).get(arm) or {}
        c = agg.get("counts") or {}
        add("  %-5s cases %-3s incidents %-3s healthy %-3s undecidable %-3s "
            "degraded %-3s no-quote findings %-3s parse-err %-3s"
            % (arm, c.get("cases"), c.get("incidents"), c.get("healthy"),
               c.get("undecidable"), c.get("degraded"),
               c.get("findings_without_quote"), c.get("parse_errors")))
    add("")
    wanted = set(metrics or [m["key"] for m in METRICS])
    add("margins over %s (group: %s)" % (result.get("baseline", BASELINE_ARM), group))
    add("  %-6s %-20s %8s %8s %9s %9s %6s" %
        ("arm", "metric", "arm", "base", "delta", "mdd", "n"))
    for row in result.get("comparisons") or []:
        if row.get("group") != group or row.get("metric") not in wanted:
            continue
        is_mean = row.get("kind") == "mean"
        arm_v = row.get("arm_mean") if is_mean else row.get("arm_p")
        base_v = row.get("baseline_mean") if is_mean else row.get("baseline_p")
        n = min(row.get("arm_n") or 0, row.get("baseline_n") or 0)
        tail = ""
        if row.get("delta") is None:
            tail = "  (%s)" % (row.get("note") or "not computable")
        elif row.get("mdd") is None:
            tail = "  no difference is detectable at this n"
        elif row.get("below_mdd"):
            tail = "  below MDD - not a result"
        if row.get("baseline_degraded"):
            tail += "  [baseline degraded: no signals]"
        add("  %-6s %-20s %8s %8s %9s %9s %6d%s"
            % (row.get("arm"), row.get("metric"), _fmt(arm_v), _fmt(base_v),
               _fmt(row.get("delta")), _fmt(row.get("mdd")), n, tail))
        if row.get("delta") is not None:
            add("         95%% CI on the margin [%s, %s]%s"
                % (_fmt((row.get("delta_ci") or [None, None])[0]),
                   _fmt((row.get("delta_ci") or [None, None])[1]),
                   "  (interval treats the arms as independent; they are paired)"
                   if row.get("kind") != "mean" else ""))
    return "\n".join(out)


# ---------------------------------------------------------------- run / reproduce --

def score_all(cases, records):
    """Score a list of records against the corpus. Returns (rows, warnings).

    A record whose `bundle_sha256` no longer matches the case it names is
    dropped with a warning: re-scoring a verdict against a bundle it never saw
    produces a number from two different corpora, which is worse than a gap.
    """
    rows, warnings = [], []
    for rec in records:
        cid = str(rec.get("case_id") or "")
        case = (cases or {}).get(cid)
        if case is None:
            warnings.append("verdict for unknown case %r dropped" % cid)
            continue
        want = str(rec.get("bundle_sha256") or "")
        have = str(case.get("bundle_sha256") or "")
        if want and have and want != have:
            warnings.append("%s/%s: verdict was produced against bundle %s, corpus "
                            "now holds %s — dropped"
                            % (cid, rec.get("arm"), want[:12], have[:12]))
            continue
        if rec.get("rubric_sha256") and rec["rubric_sha256"] != rubric_sha256():
            warnings.append("%s/%s: verdict was produced under rubric %s, re-scored "
                            "under %s" % (cid, rec.get("arm"),
                                          str(rec["rubric_sha256"])[:12],
                                          rubric_sha256()[:12]))
        rows.append(score_record(case, rec))
    return rows, warnings


def _arm_keys(rows):
    """Arm labels for aggregation, split by model where one arm carries several.

    A verdicts directory can hold two models' answers for the same arm (two
    runs, one `--verdicts` path). Collapsing those into one "L3" would take a
    majority vote across two different models and report the result as one
    arm's recall. They are aggregated separately instead, as `L3@<model>`, and
    the caller is told why.
    """
    models = {}
    for r in rows:
        models.setdefault(str(r.get("arm")), set()).add(str(r.get("model") or ""))
    split_arms = {a for a, m in models.items() if len({x for x in m if x}) > 1}

    def key(row):
        arm = str(row.get("arm"))
        if arm in split_arms:
            return "%s@%s" % (arm, str(row.get("model") or "none"))
        return arm

    return key, sorted(split_arms), models


def build_result(rows, corpus, split_info, meta):
    """Matrix -> collapsed cases -> per-arm aggregates -> margins over A0+."""
    key_of, split_arms, models = _arm_keys(rows)
    arms = sorted({key_of(r) for r in rows}, key=lambda a: (
        ARM_IDS.index(a.split("@")[0]) if a.split("@")[0] in ARM_IDS else 99, a))
    per_arm, case_rows_all, warnings = {}, [], []
    for arm in split_arms:
        warnings.append("%s carries verdicts from %d models (%s) — aggregated "
                        "separately as %s@<model>; point --verdicts at one "
                        "model's directory to avoid this"
                        % (arm, len({x for x in models[arm] if x}),
                           ",".join(sorted(x for x in models[arm] if x)), arm))
    for arm in arms:
        by_case = {}
        for r in rows:
            if key_of(r) == arm:
                by_case.setdefault(r.get("case_id"), []).append(r)
        collapsed = []
        for c in sorted(by_case):
            row = collapse_repeats(by_case[c])
            row["arm"] = arm                 # the aggregation label, not the arm id
            collapsed.append(row)
        case_rows_all.extend(collapsed)
        split_model = collapsed and "@" in arm
        model = collapsed[0].get("model") if split_model else meta.get("model", "")
        per_arm[arm] = aggregate(collapsed, arm, model)
        c = per_arm[arm]["counts"]
        if c.get("undecidable"):
            warnings.append("%s: %d case(s) undecidable from the export — the "
                            "comparison is unsafe until those cases carry the "
                            "fields the arm needs" % (arm, c["undecidable"]))
        if c.get("degraded"):
            warnings.append("%s: %d case(s) scored with the signals module "
                            "unavailable" % (arm, c["degraded"]))
        if c.get("model_errors"):
            warnings.append("%s: %d model call(s) failed and scored as no "
                            "detection" % (arm, c["model_errors"]))
        if c.get("context_overflow"):
            warnings.append("%s: %d prompt(s) exceeded num_ctx" % (arm, c["context_overflow"]))
    comparisons = compare(per_arm, BASELINE_ARM)
    if BASELINE_ARM not in per_arm and len(arms) > 1:
        warnings.append("baseline %s was not run — no margins computed" % BASELINE_ARM)
    result = {
        "run_id": meta.get("run_id", ""), "ts": time.time(),
        "root": str(meta.get("root", "")), "corpus_hash": corpus.get("hash", ""),
        "rubric_sha256": rubric_sha256(), "baseline": BASELINE_ARM,
        "repeats": meta.get("repeats", REPEATS),
        "temperature": meta.get("temperature", TEMPERATURE),
        "model": meta.get("model", ""), "num_ctx": meta.get("num_ctx"),
        "split": split_info, "arms": arms, "matrix": rows,
        "cases": case_rows_all, "per_arm": per_arm, "comparisons": comparisons,
        "warnings": warnings + list(meta.get("warnings") or []),
        "errors": list(meta.get("errors") or []),
    }
    return result


def _verdict_path(out_dir, arm, model, case_id, repeat):
    tag = "%s_%s" % (arm, re.sub(r"[^A-Za-z0-9._-]+", "-", model)) if model else arm
    return pathlib.Path(out_dir) / "verdicts" / tag / ("%s_r%d.json" % (case_id, repeat))


def _write_json(path, obj):
    try:
        p = pathlib.Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        os.replace(tmp, str(p))
        return str(p), None
    except Exception as e:
        return None, "cannot write %s: %s" % (path, e)


def run(root=None, arms=None, split_name="test", repeats=REPEATS, model_fn=None,
        model="", num_ctx=32768, out_dir=None, corpus=None, split=None,
        case_ids=None, write=True, temperature=TEMPERATURE):
    """Run the arms over a split and score them. Never raises.

    Deterministic arms are executed once and their record copied across the
    repeats: re-running a pure function three times measures nothing, and
    pretending it does would put three identical rows into a matrix whose
    disagreement column is supposed to mean something.
    """
    root = pathlib.Path(str(root or default_root()))
    out_dir = pathlib.Path(str(out_dir or root))
    corpus = corpus if corpus is not None else load_corpus(root)
    split = split if split is not None else load_split(root)
    arms = [a for a in (arms or ["A0", "A0p", "A0pp"])]
    errors = list(corpus.get("errors") or []) + list(split.get("errors") or [])
    if case_ids is None:
        case_ids, missing = select_cases(corpus, split, split_name)
    else:
        missing = [c for c in case_ids if c not in corpus.get("cases", {})]
        case_ids = [c for c in case_ids if c in corpus.get("cases", {})]
    warnings = []
    if missing:
        warnings.append("split %s names %d case(s) that are not in the corpus: %s"
                        % (split_name, len(missing), ",".join(missing[:8])))
    if split_name == "dev":
        warnings.append("dev split: these cases are excluded from reported numbers")

    ctx = {"model_fn": model_fn, "model": model, "num_ctx": num_ctx,
           "temperature": temperature}
    records = []
    reps = max(1, int(repeats))
    for arm in arms:
        for cid in case_ids:
            case = corpus["cases"][cid]
            if not case.get("ok"):
                warnings.append("%s: case did not load (%s) — skipped"
                                % (cid, "; ".join(case.get("errors") or [])))
                continue
            first = None
            for rep in range(reps):
                if is_deterministic(arm) and first is not None:
                    rec = dict(first)
                    rec["repeat"] = rep
                    rec["copied_from_repeat"] = 0
                else:
                    rec = run_arm(arm, case, ctx, repeat=rep)
                    if first is None:
                        first = rec
                records.append(rec)
                if write:
                    _, err = _write_json(_verdict_path(out_dir, arm, model, cid, rep), rec)
                    if err:
                        errors.append(err)
    rows, score_warn = score_all(corpus.get("cases", {}), records)
    warnings.extend(score_warn)
    split_info = {"name": split_name, "rule": split.get("rule", ""),
                  "sha256": split.get("sha256", ""), "sha_ok": split.get("sha_ok"),
                  "n_cases": len(case_ids), "cases": list(case_ids),
                  "missing": missing}
    meta = {"run_id": time.strftime("%Y%m%dT%H%M%S"), "root": str(root),
            "repeats": reps, "model": model, "num_ctx": num_ctx,
            "temperature": temperature, "warnings": warnings, "errors": errors}
    result = build_result(rows, corpus, split_info, meta)
    if write:
        for err in write_results(result, out_dir):
            result["errors"].append(err)
    return result


def load_verdicts(verdicts_dir):
    """Every committed verdict JSON under a directory. Returns (records, errors)."""
    recs, errors = [], []
    p = pathlib.Path(str(verdicts_dir))
    if not p.is_dir():
        return recs, ["no verdicts directory: %s" % p]
    for f in sorted(p.rglob("*.json")):
        obj, err = _read_json(f)
        if err:
            errors.append(err)
            continue
        items = obj if isinstance(obj, list) else [obj]
        for rec in items:
            if isinstance(rec, dict) and rec.get("case_id") and rec.get("arm"):
                recs.append(rec)
            else:
                errors.append("%s: not a verdict record" % f.name)
    recs.sort(key=lambda r: (str(r.get("arm")), str(r.get("case_id")),
                             int(_num(r.get("repeat"), 0) or 0)))
    return recs, errors


def reproduce(root=None, verdicts_dir=None, out_dir=None, corpus=None, split=None,
              split_name="test", write=False):
    """Re-score committed verdicts with no model call.

    The verdicts carry everything scoring reads (the verdict object, the token
    and SU accounting, which artifacts a retrieval round pulled in), and the
    corpus is frozen, so this reproduces the numbers of the run that wrote
    them — or says why it cannot.
    """
    root = pathlib.Path(str(root or default_root()))
    out_dir = pathlib.Path(str(out_dir or root))
    verdicts_dir = pathlib.Path(str(verdicts_dir or (out_dir / "verdicts")))
    corpus = corpus if corpus is not None else load_corpus(root)
    split = split if split is not None else load_split(root)
    records, errors = load_verdicts(verdicts_dir)
    errors = list(corpus.get("errors") or []) + list(split.get("errors") or []) + errors
    rows, warnings = score_all(corpus.get("cases", {}), records)
    seen_cases = sorted({str(r.get("case_id")) for r in rows})
    reps = max([1] + [int(_num(r.get("repeat"), 0) or 0) + 1 for r in records])
    models = sorted({str(r.get("model") or "") for r in records if r.get("model")})
    split_info = {"name": split_name, "rule": split.get("rule", ""),
                  "sha256": split.get("sha256", ""), "sha_ok": split.get("sha_ok"),
                  "n_cases": len(seen_cases), "cases": seen_cases, "missing": []}
    meta = {"run_id": "reproduce-" + time.strftime("%Y%m%dT%H%M%S"),
            "root": str(root), "repeats": reps,
            "model": ",".join(models), "num_ctx": None,
            "temperature": TEMPERATURE, "warnings": warnings, "errors": errors}
    result = build_result(rows, corpus, split_info, meta)
    result["reproduced_from"] = str(verdicts_dir)
    result["verdicts_read"] = len(records)
    if write:
        for err in write_results(result, out_dir):
            result["errors"].append(err)
    return result


def write_results(result, out_dir):
    """`results/<arm>_<model>.json` per arm plus the whole run. Returns errors."""
    errors = []
    rdir = pathlib.Path(str(out_dir)) / "results"
    _, err = _write_json(rdir / ("run_%s.json" % result.get("run_id", "x")), result)
    if err:
        errors.append(err)
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", str(result.get("model") or "")) or "none"
    for arm in result.get("arms") or []:
        agg = (result.get("per_arm") or {}).get(arm) or {}
        obj = {"arm": arm, "model": result.get("model", ""),
               "run_id": result.get("run_id"), "ts": result.get("ts"),
               "corpus_hash": result.get("corpus_hash"),
               "rubric_sha256": result.get("rubric_sha256"),
               "baseline": result.get("baseline"), "split": result.get("split"),
               "repeats": result.get("repeats"),
               "temperature": result.get("temperature"),
               "groups": agg.get("groups"), "counts": agg.get("counts"),
               "notify_latency_s": agg.get("notify_latency_s"),
               "comparisons": [c for c in (result.get("comparisons") or [])
                               if c.get("arm") == arm],
               "cases": [c for c in (result.get("cases") or [])
                         if c.get("arm") == arm],
               "warnings": result.get("warnings"), "errors": result.get("errors")}
        _, err = _write_json(rdir / ("%s_%s.json" % (arm, model)), obj)
        if err:
            errors.append(err)
    return errors


# ---------------------------------------------------------------- fake model --

class FakeModel:
    """Deterministic supervisor stand-in for the tests. No network, no provider.

    It scans the prompt for pre-registered trigger tokens and quotes the first
    line that carries one, stripping the rendered line-number prefix so the
    quote resolves through the citation rule. That makes it sensitive to the
    information ablation on purpose: the same instance detects on L2/L3 (which
    carry raw lines) and stays silent on L1 (which does not), which is the
    behaviour the ablation is supposed to measure, produced here without a
    model so the harness itself can be tested.
    """

    def __init__(self, triggers=None, corrections=None, escalate="tier1",
                 severity="crit", fabricate_quote=False, retrieve=None,
                 su_per_call=0.0, latency_s=0.0, text_wrapper=None):
        self.triggers = dict(triggers or {"TIMEOUT": "walltime_bound"})
        self.corrections = list(corrections or [])
        self.escalate = escalate
        self.severity = severity
        self.fabricate_quote = bool(fabricate_quote)
        self.retrieve = list(retrieve or [])
        self.su_per_call = float(su_per_call)
        self.latency_s = float(latency_s)
        self.text_wrapper = text_wrapper
        self.calls = 0
        self.prompts = []

    @staticmethod
    def _quotable(line):
        m = _LINE_RE.match(line)
        return (m.group(2) if m else line).strip()

    def __call__(self, prompt, model_id="fake", num_ctx=32768):
        self.calls += 1
        self.prompts.append(prompt)
        findings = []
        for token, signal in sorted(self.triggers.items()):
            for line in str(prompt).splitlines():
                if token.lower() in line.lower():
                    quote = ("no such line exists anywhere in this bundle"
                             if self.fabricate_quote else self._quotable(line))
                    findings.append({"signal": signal, "quote": quote[:200],
                                     "diagnosis": "%s seen in the evidence" % token,
                                     "severity": self.severity})
                    break
        obj = {"verdict": "issue" if findings else "ok", "findings": findings,
               "corrections": self.corrections if findings else [],
               "escalate": {"to": self.escalate if findings else "none",
                            "reason": "trigger matched" if findings else "nothing matched"},
               "confidence": 0.9 if findings else 0.1}
        if self.retrieve and PROMPT_RETRIEVAL_NOTE.strip() in str(prompt):
            obj["retrieve"] = list(self.retrieve)
        text = json.dumps(obj)
        if self.text_wrapper:
            text = self.text_wrapper(text)
        return {"text": text, "tokens_in": _estimate_tokens(prompt),
                "tokens_out": _estimate_tokens(text), "latency_s": self.latency_s,
                "su": self.su_per_call}


# ---------------------------------------------------------------- CLI --

def load_model_entry(spec):
    """Resolve `package.module:attribute` to the integrator's callable.

    Returns (callable, reason). A class is instantiated with no arguments so a
    stateful client can be shipped as one. Nothing about a provider is assumed
    here: this file must import on the lab, in a SLURM job and on a laptop.
    """
    text = str(spec or "").strip()
    if not text:
        return None, "no model entry given"
    if ":" not in text:
        return None, "model entry must be package.module:attribute"
    mod_name, attr = text.rsplit(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return None, "cannot import %s: %s" % (mod_name, e)
    obj = getattr(mod, attr, None)
    if obj is None:
        return None, "%s has no attribute %s" % (mod_name, attr)
    if isinstance(obj, type):
        try:
            obj = obj()
        except Exception as e:
            return None, "cannot instantiate %s: %s" % (attr, e)
    if not callable(obj):
        return None, "%s.%s is not callable" % (mod_name, attr)
    return obj, ""


def cmd_list(args):
    corpus = load_corpus(args.root)
    split = load_split(args.root)
    dev, test = set(split.get("dev") or []), set(split.get("test") or [])
    print("corpus %s  (%d cases, hash %s)"
          % (corpus.get("root"), len(corpus.get("cases") or {}),
             (corpus.get("hash") or "")[:12]))
    print("split rule: %s" % (split.get("rule") or "(none)"))
    print("  %-28s %-9s %-12s %-11s %-6s %s"
          % ("case", "split", "class", "provenance", "arts", "label"))
    for cid, case in sorted((corpus.get("cases") or {}).items()):
        t = case.get("truth") or {}
        where = "dev" if cid in dev else ("test" if cid in test else "-")
        print("  %-28s %-9s %-12s %-11s %-6d %s"
              % (cid, where, t.get("class") or "?", t.get("provenance") or "?",
                 len(case.get("artifacts") or {}),
                 "incident" if t.get("incident") else "healthy"))
    for e in (corpus.get("errors") or []) + (split.get("errors") or []):
        print("  ERROR: %s" % e)
    return 1 if (corpus.get("errors") or split.get("errors")) else 0


def cmd_run(args):
    model_fn, reason = (None, "")
    if args.model_entry:
        model_fn, reason = load_model_entry(args.model_entry)
    arms = [a.strip() for a in str(args.arms or "").split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_IDS]
    if unknown:
        print("ERROR: unknown arm(s) %s; known: %s"
              % (",".join(unknown), ",".join(ARM_IDS)))
        return 2
    if any(a in MODEL_ARMS for a in arms) and model_fn is None:
        # Recorded, not fatal: the model rows land as "skipped" and the
        # deterministic arms in the same invocation still produce their numbers.
        print("WARNING: model arms requested with no usable callable (%s)"
              % (reason or "none given"))
    case_ids = [c.strip() for c in str(args.cases or "").split(",") if c.strip()] or None
    result = run(root=args.root, arms=arms, split_name=args.split,
                 repeats=args.repeats, model_fn=model_fn, model=args.model,
                 num_ctx=args.num_ctx, out_dir=args.out, case_ids=case_ids,
                 write=not args.no_write, temperature=args.temperature)
    print(render_report(result, group=args.group))
    return 1 if result.get("errors") else 0


def cmd_reproduce(args):
    result = reproduce(root=args.root, verdicts_dir=args.verdicts, out_dir=args.out,
                       split_name=args.split, write=bool(args.write))
    print(render_report(result, group=args.group))
    print("  verdicts re-scored: %d (no model call)" % result.get("verdicts_read", 0))
    return 1 if result.get("errors") else 0


COMMANDS = ("list", "run", "reproduce")


def build_parser():
    """Parser with the shared options on every subcommand.

    They live on the subparsers rather than on the top level because argparse
    lets a subparser default overwrite a value the top level already parsed,
    which silently sends `--root` somewhere else. `_normalise_argv` puts the
    subcommand first so either order still works on the command line.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(default_root()),
                        help="supervision_bench directory (cases/, split.json, results/)")
    common.add_argument("--out", default=None,
                        help="where to write verdicts/ and results/")
    common.add_argument("--group", default="all", help="report this class group")

    p = argparse.ArgumentParser(
        prog="bench", description="score supervision arms over the exported cases")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", parents=[common],
                   help="list the corpus and its split membership")

    r = sub.add_parser("run", parents=[common],
                       help="run arms over a split and score them")
    r.add_argument("--arms", default="A0,A0p,A0pp")
    r.add_argument("--split", default="test", choices=("test", "dev", "all"))
    r.add_argument("--repeats", type=int, default=REPEATS)
    r.add_argument("--model-entry", default="", dest="model_entry",
                   help="package.module:callable -> (prompt, model_id, num_ctx) "
                        "-> {text, tokens_in, tokens_out, latency_s, su}")
    r.add_argument("--model", default="", help="model id recorded in every row")
    r.add_argument("--num-ctx", type=int, default=32768, dest="num_ctx")
    r.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help="recorded in the run file; the callable owns sampling")
    r.add_argument("--cases", default="", help="comma-separated case ids (overrides --split)")
    r.add_argument("--no-write", action="store_true")

    q = sub.add_parser("reproduce", parents=[common],
                       help="re-score committed verdicts, no model call")
    q.add_argument("--verdicts", default=None)
    q.add_argument("--split", default="test")
    q.add_argument("--write", action="store_true")
    return p


def _normalise_argv(argv):
    """Subcommand first, `--reproduce` accepted as a flag anywhere.

    `bench.py --reproduce --root X` is how the protocol names the re-scoring
    path, and `bench.py reproduce --root X` is how argparse wants it; both have
    to work or the documented command is a trap.
    """
    forced = "--reproduce" in argv
    argv = [a for a in argv if a != "--reproduce"]
    cmd = None
    for i, tok in enumerate(argv):
        if tok in COMMANDS:
            cmd = argv.pop(i)
            break
    if forced:
        cmd = "reproduce"
    if cmd is None:
        if any(a in ("-h", "--help") for a in argv):
            return argv          # top-level help, which lists the subcommands
        cmd = "list" if not argv else "run"
    return [cmd] + argv


def main(argv=None):
    """Entry point. Returns an exit code; raises nothing."""
    try:
        argv = _normalise_argv(list(sys.argv[1:] if argv is None else argv))
        args = build_parser().parse_args(argv)
        if args.cmd == "list":
            return cmd_list(args)
        if args.cmd == "reproduce":
            return cmd_reproduce(args)
        return cmd_run(args)
    except SystemExit as e:                 # argparse's own exit
        return int(e.code or 0)
    except Exception as e:
        print("ERROR: bench failed: %s: %s" % (type(e).__name__, e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
