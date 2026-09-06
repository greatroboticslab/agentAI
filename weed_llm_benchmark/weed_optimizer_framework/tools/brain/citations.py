#!/usr/bin/env python3
"""Citation validator for the tiered supervision layer (v3.27.0).

Why this exists
---------------
The defect this campaign keeps producing is not a model that refuses to answer,
it is a model that answers plausibly with nothing behind it: a small model
rubber-stamped 100 % of crops as good, and a causal claim survived several human
passes before the artifacts contradicted it. A supervisor that writes "the job
was walltime-bound" without pointing at the line that says so is the same defect
with more steps, and on this loop it would arrive with authority to cancel jobs.

So a finding counts as a finding only when it quotes an artifact line verbatim,
and this module is what checks that. `resolve` turns a quote into an absolute
address `(artifact_id, line)` inside one evidence bundle; a finding whose quote
does not resolve is recorded `rejected_unverifiable` and is never actionable.
The address is the line number the file itself has, never an index into the
trimmed tail the model was shown, so a citation stays checkable months later
with `sed -n '8123p' <file>` against the original artifact.

The worked example is the 2026-08-29 pair 44727703 / 44767709: both jobs died at
a 12 h walltime because the merged pool had grown to 8,583 iterations per epoch.
A supervisor's verdict on that bundle has to quote the `sacct` TIMEOUT row or
the `slurmstepd` cancellation line, and its quote has to land on the line the
truth marks load-bearing. The same bundle also carries the chronic
`SKIPPING github` warning, which is a designed control whose correct verdict is
"no incident" — it resolves like any other line, because resolving a quote says
the line exists, not that it means anything.

What this module does not do
----------------------------
It does not judge whether a finding is true. A resolved quote proves the line
exists and that the model read it; it does not prove the line supports the
claim. That is why `validate_verdict` counts three failures separately — no
quote at all, a quote that does not resolve, and a quote that resolves to a line
the truth does not consider load-bearing. Collapsing them would hide the
interesting one: a model that cites a real line which does not support its claim
looks exactly like a grounded supervisor until those counts are separated.

Address space
-------------
Only line-addressed evidence is citable. The canonical shape is the bundle's
`out_tail` section as `corpus.py` freezes it — `{"artifact_id", "path",
"sha256", "lines": [[absolute_line_number, text], ...]}` — and any section
carrying that shape (one object, a list of them, or a mapping of name to one) is
indexed the same way. `sacct` rows, the strategy dict and the signal values are
evidence a supervisor may reason from but cannot cite, by construction: there is
no address to point at, so the validator would have nothing to check.

Matching rule (pre-registered, restated in `bench.py` so a committed verdict can
be re-scored without this package):

* the quote is whitespace-normalised and case-folded, so a reflowed or re-cased
  quote still resolves — models rewrap long log lines and this must not be
  scored as a fabrication;
* it must be at least `min_quote_chars` characters after normalisation, because
  a short fragment matches by accident;
* it must appear inside a SINGLE line. A quote spanning two lines is not a line
  citation and is refused with that reason stated, so the caller gives two
  quotes instead of one unaddressable one.

Nothing here raises out of a public entry point, and nothing here reads the
network, a database or the filesystem beyond the optional pre-registered
threshold file next to this module: it is imported inside SLURM job bodies, in
the always-on server process and on a laptop.

Commands
--------
    resolve  --bundle <bundle.json> --quote <text>
    validate --bundle <bundle.json> --verdict <verdict.json> [--truth <truth.json>]
    index    --bundle <bundle.json>
    thresholds
"""
import argparse
import json
import os
import pathlib
import re
import sys

TOOL_VERSION = "wp3-citations/1"

# The 14 bundle sections in the order `corpus.py` freezes them. Duplicated here
# rather than imported so this module keeps working where only part of the
# package is deployed (it is imported from a job body and from the server);
# `tests/test_brain_citations.py` asserts it still equals `corpus.SECTIONS`, so
# the duplication cannot drift silently.
SECTION_ORDER = ("ledger", "sacct", "out_tail", "results_csv", "strategy", "trace",
                 "slug_scores", "registry_diff", "harvest", "resources", "su",
                 "corrections", "plan", "signals")

# Every rejection is recorded under this one status because the consequence is
# the same for all three: the finding is not actionable. The `reason` field says
# which failure it was, and the counts are kept apart in `stats`.
REJECTED = "rejected_unverifiable"
NO_QUOTE = "no_quote"
TOO_SHORT = "quote_too_short"
UNRESOLVED = "unresolved"
MALFORMED = "malformed_finding"
REJECTION_REASONS = (NO_QUOTE, TOO_SHORT, UNRESOLVED, MALFORMED)
# reason -> the counter it increments. Written out rather than derived from
# the reason string, so a new reason has to be given a counter deliberately
# instead of landing in whichever bucket a name collision picks.
_REASON_STAT = {NO_QUOTE: "no_quote", TOO_SHORT: "quote_too_short",
                UNRESOLVED: "unresolved", MALFORMED: "malformed"}

# --- thresholds ------------------------------------------------------------
# Every number this module decides with lives in this table with the reason it
# has the value it has. Resolution order, lowest to highest: this default, the
# pre-registered `thresholds.json` beside this module (the committed record),
# then the environment (a deliberate one-off, e.g. a test). `thresholds()`
# reports the effective value AND its source so a result file can record which
# rule it was scored under.
THRESHOLDS = {
    "min_quote_chars": {
        "default": 20,
        "env": "BRAIN_CITATION_MIN_QUOTE_CHARS",
        "why": ("A quote shorter than this resolves by accident. An out_tail "
                "section holds a few hundred log lines and fragments such as "
                "'Epoch', '0.0' or 'train' occur on most of them, so a short "
                "quote would 'ground' any claim at all, which is exactly the "
                "failure the citation rule exists to catch. 20 characters is "
                "about three log tokens - long enough to be specific, short "
                "enough that a genuine one-clause citation still passes. "
                "Restated as MIN_QUOTE_CHARS in bench.py; the two must agree."),
    },
    "span_window_lines": {
        "default": 3,
        "env": "BRAIN_CITATION_SPAN_WINDOW_LINES",
        "why": ("How many consecutive lines are joined when explaining WHY an "
                "unresolved quote did not resolve. A quote that spans lines is "
                "still refused - it has no single address - but saying so is "
                "worth more than 'not found', because the fix is to send two "
                "quotes. Three lines covers a wrapped log line and its "
                "continuations without turning the diagnostic into a search "
                "that would accept a quote assembled from unrelated lines."),
    },
}

_THRESHOLD_FILE = pathlib.Path(__file__).resolve().parent / "thresholds.json"
# The file is read once per process: it is pre-registered and does not change
# under a running job. The environment is re-read on every call so a test or a
# one-off run can override without reloading the module.
_FILE_CACHE = {"loaded": False, "values": {}, "error": None}

_WS_RE = re.compile(r"\s+")
# `%06d\t<line>` is how corpus.py stores an artifact on disk; a caller that
# indexes a stored artifact file goes through `index_numbered_text`, which needs
# to split that prefix off. The digit bound is a shape, not a threshold.
_NUMBERED_RE = re.compile(r"^\s*(\d{1,12})\t(.*)$")


def _file_thresholds():
    """The pre-registered threshold file's `citations` block, or {}.

    Missing, unreadable or malformed is a state to report, never an exception:
    this module is imported from an Ultralytics callback and from the server.
    """
    if not _FILE_CACHE["loaded"]:
        _FILE_CACHE["loaded"] = True
        try:
            if _THRESHOLD_FILE.exists():
                with open(str(_THRESHOLD_FILE), "r", encoding="utf-8") as f:
                    obj = json.load(f)
                block = (obj or {}).get("citations")
                if isinstance(block, dict):
                    _FILE_CACHE["values"] = block
                elif block is not None:
                    _FILE_CACHE["error"] = "thresholds.json: 'citations' is not an object"
        except Exception as exc:              # unreadable/invalid JSON: report it
            _FILE_CACHE["error"] = "thresholds.json unreadable: %s" % type(exc).__name__
    return _FILE_CACHE["values"]


def _int_or_none(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def line_number(value):
    """An absolute line number from a bundle or truth row, or None.

    An int, or a string of digits (a hand-assembled case may carry either, and
    JSON round-trips through both). A fractional value is refused rather than
    truncated: an address off by a line sends a reviewer to the wrong line of
    the file, which is worse than having no address at all.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def thresholds():
    """Effective thresholds plus, for each, where the value came from."""
    out = {"tool_version": TOOL_VERSION, "values": {}, "sources": {}, "errors": []}
    fromfile = _file_thresholds()
    if _FILE_CACHE["error"]:
        out["errors"].append(_FILE_CACHE["error"])
    for name, spec in THRESHOLDS.items():
        value, source = spec["default"], "default"
        got = _int_or_none(fromfile.get(name)) if isinstance(fromfile, dict) else None
        if got is not None and got > 0:
            value, source = got, "thresholds.json"
        env = _int_or_none(os.environ.get(spec["env"]))
        if env is not None and env > 0:
            value, source = env, "env:" + spec["env"]
        elif os.environ.get(spec["env"]):
            out["errors"].append("%s is set but is not a positive integer; ignored"
                                 % spec["env"])
        out["values"][name] = value
        out["sources"][name] = source
    return out


def threshold(name):
    """One effective threshold value."""
    return thresholds()["values"].get(name, THRESHOLDS.get(name, {}).get("default"))


# --- text and identity -----------------------------------------------------
def normalize(text):
    """Whitespace-normalised, case-folded text — the form every match uses.

    Runs of whitespace collapse to one space so a quote a model rewrapped still
    resolves; case is folded because a re-cased quote points at the same line
    and refusing it would score a formatting habit as a fabrication. `bench.py`
    normalises identically, so the two implementations cannot disagree about
    which line an address points at.
    """
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text)).strip().casefold()


def artifact_key(name):
    """Comparison identity for an artifact: basename, case-folded.

    `truth.json` names an artifact as it sits in `artifacts/`; a bundle may
    carry the absolute /ocean path it was read from. Comparing basenames keeps
    the two agreeing without rewriting either.
    """
    return os.path.basename(str(name or "").strip()).casefold()


# --- the address space -----------------------------------------------------
def _line_objects(section_name, value):
    """[(artifact_id, obj), ...] — the line-addressed objects in one section.

    Three shapes are accepted, each because a bundle really carries it:
      * one object                         `out_tail` as corpus.py writes it
      * a list of objects                  a builder carrying several tails
      * a mapping of name -> object        the same, keyed by artifact name
    Anything else is not line-addressed and is skipped: a finding can only cite
    what has an (artifact, line) address.
    """
    out = []
    if isinstance(value, dict) and isinstance(value.get("lines"), list):
        out.append((value.get("artifact_id") or value.get("path") or section_name, value))
        return out
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("lines"), list):
                out.append((item.get("artifact_id") or item.get("path") or section_name,
                            item))
        return out
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            item = value[key]
            if isinstance(item, dict) and isinstance(item.get("lines"), list):
                out.append((item.get("artifact_id") or item.get("path") or key, item))
    return out


def _rows_of(artifact_id, section_name, obj):
    """(rows, skipped) for one line-addressed object.

    A row is `[absolute_line_number, text]` (canonical) or `{"line","text"}`
    (hand-assembled cases). A row whose line number is not a positive integer is
    dropped and counted: an address that is not absolute is worse than no
    address, because it would point a reviewer at the wrong line of the file.
    """
    rows, skipped = [], 0
    aid = str(artifact_id)
    for raw in obj.get("lines") or []:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            num, text = raw[0], raw[1]
        elif isinstance(raw, dict):
            num, text = raw.get("line"), raw.get("text")
        else:
            skipped += 1
            continue
        n = line_number(num)
        if n is None or n < 1 or not isinstance(text, str):
            skipped += 1
            continue
        rows.append({"artifact_id": aid, "line": n, "text": text,
                     "section": section_name, "norm": normalize(text)})
    return rows, skipped


def index_bundle(bundle):
    """Index every citable line in a bundle.

    Returns `{"lines": [{artifact_id, line, text, section, norm}, ...],
              "artifacts": [{artifact_id, section, lines, sha256, path}, ...],
              "skipped": [{section, artifact_id, rows, reason}, ...],
              "source": str, "errors": [str, ...]}`.

    Sections are walked in the frozen `SECTION_ORDER` first and any remaining
    keys in sorted order, so "the first match" is a stable answer that does not
    depend on the key order of the JSON the bundle was loaded from.
    """
    out = {"lines": [], "artifacts": [], "skipped": [], "source": "", "errors": []}
    if not isinstance(bundle, dict):
        out["errors"].append("bundle is not an object")
        return out
    sections = bundle.get("sections")
    if isinstance(sections, dict):
        out["source"] = "bundle.sections"
    elif "sections" in bundle:
        out["errors"].append("bundle.sections is not an object")
        return out
    else:
        # A caller that hands over the sections object itself is a common slip
        # and costs nothing to accept; the source string records which it was,
        # so a report never has to guess what was indexed.
        sections = bundle
        out["source"] = "bundle-as-sections"

    names = [n for n in SECTION_ORDER if n in sections]
    names += sorted(k for k in sections.keys() if k not in SECTION_ORDER)
    for name in names:
        for aid, obj in _line_objects(name, sections.get(name)):
            rows, skipped = _rows_of(aid, name, obj)
            out["lines"].extend(rows)
            out["artifacts"].append({
                "artifact_id": str(aid), "section": name, "lines": len(rows),
                "sha256": str(obj.get("sha256") or ""), "path": str(obj.get("path") or "")})
            if skipped:
                out["skipped"].append({
                    "section": name, "artifact_id": str(aid), "rows": skipped,
                    "reason": "row is not [absolute_line_number, text] with a "
                              "positive integer line number and text"})
    return out


def bundle_lines(bundle):
    """Every citable line as `{artifact_id, line, text, section, norm}`."""
    return index_bundle(bundle)["lines"]


def index_numbered_text(artifact_id, text, section="retrieved"):
    """Index a stored artifact file (`%06d\\t<line>`) into the same row shape.

    `corpus.py` writes artifacts with their absolute line numbers in front, and
    a retrieval round hands the supervisor lines straight out of such a file. A
    quote taken from those lines has to resolve like any other, so both paths
    use one indexer rather than two that can drift. Export markers (the lines
    corpus writes for omitted ranges, which start with "[") carry no line number
    and are skipped.
    """
    rows = []
    for raw in str(text or "").split("\n"):
        m = _NUMBERED_RE.match(raw)
        if not m:
            continue
        rows.append({"artifact_id": str(artifact_id), "line": int(m.group(1)),
                     "text": m.group(2), "section": section,
                     "norm": normalize(m.group(2))})
    return rows


# --- resolution ------------------------------------------------------------
def _norm_of(row):
    norm = row.get("norm")
    return norm if isinstance(norm, str) else normalize(row.get("text"))


def _address(row):
    return {"artifact_id": str(row.get("artifact_id") or ""),
            "line": int(row.get("line") or 0)}


def _spans_lines(rows, needle, window):
    """Would the quote resolve if consecutive lines were joined? (address|None)

    Only ever used to explain a refusal. A quote that needs two lines has no
    single address, so it stays refused; saying which line it starts on turns an
    unusable "not found" into an instruction to send two quotes.

    Only CONSECUTIVE lines of one artifact are joined. The out_tail section is
    trimmed, so two kept lines can sit either side of an omitted range; reporting
    a span across that gap would name a stretch of file the model was never
    shown and could not have quoted.

    Widths are tried shortest first, so the range reported is the smallest one
    that contains the quote — the two lines a caller has to re-cite, not a
    window that happens to enclose them.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row.get("artifact_id"), []).append(row)
    ordered = [sorted(g, key=lambda r: int(r.get("line") or 0))
               for _, g in sorted(groups.items(), key=lambda kv: str(kv[0]))]
    for width in range(2, window + 1):
        for group in ordered:
            for i in range(0, len(group) - width + 1):
                block = group[i:i + width]
                if any(int(block[k]["line"]) != int(block[k - 1]["line"]) + 1
                       for k in range(1, width)):
                    continue
                if needle in " ".join(_norm_of(r) for r in block).strip():
                    return {"first": _address(block[0]), "last": _address(block[-1])}
    return None


def resolve_detail(bundle, quote, lines=None):
    """`resolve` with the reason attached — what the validator records.

    Returns `{"ok", "hit", "reason", "detail", "matches", "addresses",
              "quote_chars", "min_quote_chars", "index"}`. `ok` False always
    carries a reason; a quote is never silently dropped, because "the model
    cited nothing" and "the validator could not read the bundle" are different
    results and only one of them is the model's fault.
    """
    limits = thresholds()
    minimum = limits["values"]["min_quote_chars"]
    out = {"ok": False, "hit": None, "reason": "", "detail": "", "matches": 0,
           "addresses": [], "quote_chars": 0, "min_quote_chars": minimum,
           "index": {"lines": 0, "artifacts": 0, "skipped": 0, "source": "",
                     "errors": list(limits["errors"])}}
    try:
        if lines is None:
            idx = index_bundle(bundle)
            rows = idx["lines"]
            out["index"].update({"lines": len(rows), "artifacts": len(idx["artifacts"]),
                                 "skipped": sum(s["rows"] for s in idx["skipped"]),
                                 "source": idx["source"]})
            out["index"]["errors"].extend(idx["errors"])
        else:
            rows = [r for r in lines if isinstance(r, dict)]
            out["index"].update({"lines": len(rows), "source": "caller"})

        if not isinstance(quote, str):
            out["reason"] = NO_QUOTE
            out["detail"] = "quote is not a string"
            return out
        needle = normalize(quote)
        out["quote_chars"] = len(needle)
        if not needle:
            out["reason"] = NO_QUOTE
            out["detail"] = "quote is empty after whitespace normalisation"
            return out
        if len(needle) < minimum:
            out["reason"] = TOO_SHORT
            out["detail"] = ("quote is %d characters after normalisation, below the "
                             "pre-registered minimum of %d; a shorter quote matches "
                             "by accident" % (len(needle), minimum))
            return out
        if not rows:
            out["reason"] = UNRESOLVED
            out["detail"] = ("the bundle carries no line-addressed evidence, so no "
                             "quote can resolve against it")
            return out

        hits = [r for r in rows if needle in _norm_of(r)]
        if not hits:
            out["reason"] = UNRESOLVED
            span = _spans_lines(rows, needle, limits["values"]["span_window_lines"])
            if span:
                out["detail"] = ("quote spans more than one line (%s line %d through "
                                 "line %d); a citation addresses a single line, so "
                                 "cite each line separately"
                                 % (span["first"]["artifact_id"], span["first"]["line"],
                                    span["last"]["line"]))
                out["addresses"] = [span["first"], span["last"]]
            else:
                out["detail"] = ("quote does not appear in any line of the bundle's "
                                 "%d indexed lines" % len(rows))
            return out

        first = hits[0]
        out["ok"] = True
        out["matches"] = len(hits)
        out["addresses"] = [_address(r) for r in hits]
        out["hit"] = {"artifact_id": str(first.get("artifact_id") or ""),
                      "line": int(first.get("line") or 0),
                      "matched": str(first.get("text") or ""),
                      "section": str(first.get("section") or ""),
                      "matches": len(hits),
                      "ambiguous": len(hits) > 1}
        if len(hits) > 1:
            # The first match in bundle order is returned so the answer is
            # deterministic; the count travels with it so a reviewer can see the
            # citation was not unique rather than discovering it later.
            out["detail"] = ("quote resolves in %d places; the first in bundle order "
                             "is returned" % len(hits))
        return out
    except Exception as exc:                  # a validator never kills its caller
        out["ok"] = False
        out["hit"] = None
        out["reason"] = UNRESOLVED
        out["detail"] = "validator error: %s: %s" % (type(exc).__name__, exc)
        return out


def resolve(bundle, quote, lines=None):
    """Address a quote inside a bundle, or None.

    Returns `{"artifact_id", "line", "matched", "section", "matches",
    "ambiguous"}` — `line` is the absolute line number of the artifact, the
    address a reviewer can check against the original file, never an index into
    the trimmed excerpt the model was shown. `matched` is the full text of the
    line the quote was found in, so a caller can show what the claim rests on.

    None means the quote is not usable as evidence: shorter than the
    pre-registered minimum, not present in any single line of the bundle, or
    spanning two lines. `resolve_detail` returns the same answer with the reason
    attached; `validate_verdict` records that reason.
    """
    return resolve_detail(bundle, quote, lines=lines)["hit"]


# --- verdict validation ----------------------------------------------------
def _load_bearing_set(load_bearing_lines):
    """{(artifact_key, line)} from truth.load_bearing_lines, plus what it skipped."""
    want, bad = set(), []
    for item in (load_bearing_lines or []):
        if not isinstance(item, dict):
            bad.append("load-bearing entry is not an object")
            continue
        name = item.get("artifact") if item.get("artifact") is not None \
            else item.get("artifact_id")
        line = line_number(item.get("line"))
        if not str(name or "").strip() or line is None or line < 1:
            bad.append("load-bearing entry needs an artifact name and a positive line")
            continue
        want.add((artifact_key(name), line))
    return want, bad


def validate_verdict(bundle, verdict, load_bearing_lines=None):
    """Check every finding in one supervisor verdict against one bundle.

    Returns `{"ok", "findings", "rejected", "stats"}`.

    * `findings` — the findings that carry a resolvable quote, each with the
      address it resolved to (`artifact_id`, `line`, `matched`), whether the
      quote was ambiguous, and, when the truth's load-bearing lines were
      supplied, whether the cited line is one of them. The original object is
      kept under `finding`, so nothing a model wrote is lost or overwritten.
    * `rejected` — the rest, each recorded `rejected_unverifiable` with the
      reason. These are never actionable: an unresolvable citation is exactly
      the shape of the failure the citation rule exists to catch.
    * `stats` — the three failures counted SEPARATELY (`no_quote`,
      `unresolved`, `not_load_bearing`), because they are different defects. A
      model that quotes nothing is not grounded at all; one whose quote does not
      resolve invented a line; one that cites a real line the truth does not
      consider load-bearing read the evidence and drew on the wrong part of it.
      Collapsing them into a single "bad citations" number hides the third,
      which is the one that looks like success.

    `ok` is True when the verdict is well-shaped, every finding resolved, and
    the validator itself had nothing to report - an unreadable bundle or a
    misconfigured threshold makes it False, because a check that could not run
    is not a pass. It says the verdict is checkable, never that it is correct:
    whether a finding is right is decided against truth by the bench harness.

    `load_bearing_lines` is optional because a live round has no truth labels;
    when it is absent the load-bearing counts are None and `stats` says so,
    rather than reporting a zero that would read as "cited nothing useful".
    """
    limits = thresholds()
    stats = {
        "findings": 0, "accepted": 0, "rejected": 0,
        "no_quote": 0, "quote_too_short": 0, "unresolved": 0, "malformed": 0,
        "ambiguous": 0,
        "load_bearing_source": "supplied" if load_bearing_lines is not None
                               else "not supplied",
        "load_bearing_total": None, "load_bearing_cited": None,
        "load_bearing_hit": None, "not_load_bearing": None,
        "evidence_hit_rate": None,
        "min_quote_chars": limits["values"]["min_quote_chars"],
        "index": {"lines": 0, "artifacts": 0, "skipped": 0, "source": ""},
        "tool_version": TOOL_VERSION,
        "errors": list(limits["errors"]),
    }
    out = {"ok": False, "findings": [], "rejected": [], "stats": stats}
    try:
        idx = index_bundle(bundle)
        rows = idx["lines"]
        stats["index"] = {"lines": len(rows), "artifacts": len(idx["artifacts"]),
                          "skipped": sum(s["rows"] for s in idx["skipped"]),
                          "source": idx["source"]}
        stats["errors"].extend(idx["errors"])

        want, bad_truth = _load_bearing_set(load_bearing_lines)
        stats["errors"].extend(bad_truth)
        if load_bearing_lines is not None:
            stats["load_bearing_total"] = len(want)
            stats["load_bearing_cited"] = 0
            stats["load_bearing_hit"] = 0
            stats["not_load_bearing"] = 0
            stats["evidence_hit_rate"] = 0.0

        if not isinstance(verdict, dict):
            stats["errors"].append("verdict is not an object")
            return out
        raw = verdict.get("findings")
        if raw is None:
            stats["errors"].append("verdict carries no findings array")
            raw = []
        elif not isinstance(raw, list):
            stats["errors"].append("verdict.findings is not an array")
            return out

        cited = set()
        for i, finding in enumerate(raw):
            stats["findings"] += 1
            if not isinstance(finding, dict):
                stats["malformed"] += 1
                out["rejected"].append({
                    "index": i, "status": REJECTED, "reason": MALFORMED,
                    "detail": "finding is not an object", "quote": "",
                    "signal": "", "severity": "", "finding": finding})
                continue
            signal = str(finding.get("signal") or "")
            severity = str(finding.get("severity") or "")
            quote = finding.get("quote")
            det = resolve_detail(bundle, quote, lines=rows)
            if not det["ok"]:
                reason = det["reason"] or UNRESOLVED
                stats[_REASON_STAT.get(reason, "unresolved")] += 1
                out["rejected"].append({
                    "index": i, "status": REJECTED, "reason": reason,
                    "detail": det["detail"],
                    "quote": quote if isinstance(quote, str) else "",
                    "signal": signal, "severity": severity, "finding": finding})
                continue

            hit = det["hit"]
            addr = (artifact_key(hit["artifact_id"]), int(hit["line"]))
            row = {"index": i, "signal": signal, "severity": severity,
                   "quote": quote, "artifact_id": hit["artifact_id"],
                   "line": hit["line"], "matched": hit["matched"],
                   "section": hit["section"], "matches": hit["matches"],
                   "ambiguous": hit["ambiguous"], "load_bearing": None,
                   "finding": finding}
            if hit["ambiguous"]:
                stats["ambiguous"] += 1
            if load_bearing_lines is not None:
                row["load_bearing"] = addr in want
                if row["load_bearing"]:
                    stats["load_bearing_hit"] += 1
                    cited.add(addr)
                else:
                    stats["not_load_bearing"] += 1
            out["findings"].append(row)
            stats["accepted"] += 1

        stats["rejected"] = len(out["rejected"])
        if load_bearing_lines is not None:
            stats["load_bearing_cited"] = len(cited)
            stats["evidence_hit_rate"] = (float(len(cited)) / len(want)) if want else 0.0
        out["ok"] = bool(not out["rejected"] and not stats["errors"])
        return out
    except Exception as exc:                  # a validator never kills its caller
        stats["errors"].append("validator error: %s: %s" % (type(exc).__name__, exc))
        out["ok"] = False
        return out


def _findings_of(verdict):
    """The findings of a raw verdict, of a validated result, or of a bare list."""
    if isinstance(verdict, list):
        return [f for f in verdict if isinstance(f, dict)]
    if isinstance(verdict, dict):
        raw = verdict.get("findings")
        if isinstance(raw, list):
            return [f for f in raw if isinstance(f, dict)]
    return []


def cited_addresses(verdict):
    """{(artifact_key, line)} a verdict actually points at.

    Reads an address wherever a verdict can carry one: a row from
    `validate_verdict` (`artifact_id` + `line`), a finding that carries its own
    resolved address, or the `evidence` list of the signal shape
    (`[{"artifact_id","line","quote"}]`). A finding carrying only a quote
    contributes nothing — an unresolved quote is not evidence — so run
    `validate_verdict` first when the addresses have not been resolved yet.
    """
    out = set()
    for finding in _findings_of(verdict):
        holders = [finding]
        ev = finding.get("evidence")
        if isinstance(ev, list):
            holders.extend([e for e in ev if isinstance(e, dict)])
        for holder in holders:
            name = holder.get("artifact_id") if holder.get("artifact_id") is not None \
                else holder.get("artifact")
            line = line_number(holder.get("line"))
            if str(name or "").strip() and line is not None and line >= 1:
                out.add((artifact_key(name), line))
    return out


def evidence_hit_rate(verdict, load_bearing_lines):
    """Fraction of the truth's load-bearing lines this verdict cited.

    `|{load-bearing addresses the verdict cites}| / |{load-bearing addresses}|`,
    over distinct addresses: quoting the same line in three findings is one
    piece of evidence, not three, and counting it three times would let a model
    reach a perfect score by repeating itself.

    An empty denominator returns 0.0 rather than 1.0. A case with no
    load-bearing lines has nothing to hit, and a rate that read 1.0 there would
    silently inflate any average taken over cases; callers report n beside the
    rate (bench keeps `evidence_total` for exactly this reason).

    Findings whose quotes were never resolved contribute nothing, because an
    unresolved citation is not evidence. Pass the output of `validate_verdict`
    (whose findings carry their resolved addresses) to score a model's raw
    quotes.
    """
    try:
        want, _ = _load_bearing_set(load_bearing_lines)
        if not want:
            return 0.0
        return float(len(cited_addresses(verdict) & want)) / float(len(want))
    except Exception:                         # a metric never kills its caller
        return 0.0


# --- CLI -------------------------------------------------------------------
def _read_json(path):
    """(object, error) — never raises."""
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing: %s" % path
    except Exception as exc:
        return None, "unreadable %s: %s: %s" % (path, type(exc).__name__, exc)


def _dump(obj):
    print(json.dumps(obj, sort_keys=True, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="citations",
        description="Resolve supervisor citations against an evidence bundle")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("resolve", help="address one quote inside a bundle")
    p.add_argument("--bundle", required=True)
    p.add_argument("--quote", required=True)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("validate", help="check every finding in a verdict")
    p.add_argument("--bundle", required=True)
    p.add_argument("--verdict", required=True)
    p.add_argument("--truth", default=None,
                   help="truth.json; its load_bearing_lines enable the third count")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("index", help="list the citable lines of a bundle")
    p.add_argument("--bundle", required=True)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("thresholds", help="print the effective thresholds and sources")
    p.add_argument("--json", action="store_true")

    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:                 # argparse's own exit, including --help
        return int(exc.code or 0)
    if not a.cmd:
        ap.print_help()
        return 2

    try:
        if a.cmd == "thresholds":
            _dump(thresholds())
            return 0

        bundle, err = _read_json(a.bundle)
        if err:
            print("ERROR %s" % err)
            return 1

        if a.cmd == "index":
            idx = index_bundle(bundle)
            if a.json:
                _dump({k: v for k, v in idx.items() if k != "lines"})
            else:
                print("source=%s lines=%d artifacts=%d"
                      % (idx["source"], len(idx["lines"]), len(idx["artifacts"])))
                for art in idx["artifacts"]:
                    print("  %-40s section=%-12s lines=%d"
                          % (art["artifact_id"], art["section"], art["lines"]))
                for s in idx["skipped"]:
                    print("  SKIPPED %s: %d row(s): %s"
                          % (s["artifact_id"], s["rows"], s["reason"]))
                for e in idx["errors"]:
                    print("  ERROR %s" % e)
            return 0 if idx["lines"] else 1

        if a.cmd == "resolve":
            det = resolve_detail(bundle, a.quote)
            if a.json:
                _dump(det)
            elif det["ok"]:
                print("%s:%d  %s" % (det["hit"]["artifact_id"], det["hit"]["line"],
                                     det["hit"]["matched"]))
                if det["detail"]:
                    print("  note: %s" % det["detail"])
            else:
                print("%s (%s): %s" % (REJECTED, det["reason"], det["detail"]))
            return 0 if det["ok"] else 1

        verdict, err = _read_json(a.verdict)
        if err:
            print("ERROR %s" % err)
            return 1
        lb = None
        if a.truth:
            truth, err = _read_json(a.truth)
            if err:
                print("ERROR %s" % err)
                return 1
            lb = (truth or {}).get("load_bearing_lines") or []
        rep = validate_verdict(bundle, verdict, load_bearing_lines=lb)
        if a.json:
            _dump(rep)
        else:
            st = rep["stats"]
            print("ok=%s findings=%d accepted=%d rejected=%d"
                  % (rep["ok"], st["findings"], st["accepted"], st["rejected"]))
            print("  no_quote=%d too_short=%d unresolved=%d malformed=%d ambiguous=%d"
                  % (st["no_quote"], st["quote_too_short"], st["unresolved"],
                     st["malformed"], st["ambiguous"]))
            print("  load_bearing=%s hit=%s not_load_bearing=%s evidence_hit_rate=%s"
                  % (st["load_bearing_source"], st["load_bearing_hit"],
                     st["not_load_bearing"], st["evidence_hit_rate"]))
            for f in rep["findings"]:
                print("  finding %d %s -> %s:%d%s" % (
                    f["index"], f["signal"] or "(unnamed)", f["artifact_id"], f["line"],
                    "" if f["load_bearing"] is None
                    else ("  load-bearing" if f["load_bearing"] else "  NOT load-bearing")))
            for r in rep["rejected"]:
                print("  finding %d %s: %s: %s" % (r["index"], r["status"], r["reason"],
                                                   r["detail"]))
            for e in st["errors"]:
                print("  ERROR %s" % e)
        return 0 if rep["ok"] else 1
    except Exception as exc:                  # no command line ever tracebacks
        print("ERROR %s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
