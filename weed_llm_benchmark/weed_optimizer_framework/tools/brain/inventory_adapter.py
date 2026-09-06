#!/usr/bin/env python3
"""Translate the WP2 incident inventory into the corpus exporter's case shape.

Why this exists
---------------
The inventory and the exporter were written against different field names, and
the disagreement is total: a dry run of `corpus.py export` over
`results/framework/supervision_bench/inventory.json` refused **162 of 162**
cases on the first validation line it reached. The inventory records
`correction_class` (empty for a control) where the exporter wants `class`; it
records `artifact_paths_expected`, a flat list of strings — some of them `sacct`
invocations, some of them globs — where the exporter wants
`artifacts: [{name, path, section}]`; and it does not record
`escalation_expected` at all.

This module is that translation, kept out of both sides on purpose. The
inventory is a hand-built record of what the campaign actually did and must not
be edited to please a validator; the exporter's validation is the thing that
keeps a malformed case out of a pre-registered corpus and must not be loosened.
A separate, pure, testable adapter is the only place where the two shapes can be
reconciled without weakening either.

What it does not do
-------------------
It does not rewrite the inventory's prose. Three inventory `notes` fields and
one `sacct` command name the cluster account, and this module carries them
through verbatim: the exporter's scrubber will substitute what it can and refuse
the case when a target survives, naming the rule and the line. That refusal is
the designed signal that the source text needs fixing, and swallowing it here
would hide it.

It does not invent evidence. A predicted path that is not on this machine still
becomes an artifact entry, so the exporter records it as absent with a reason
and books the case `record-only`, rather than the case quietly shrinking to one
with nothing missing.

Determinism
-----------
Same inputs, byte-identical output: keys are sorted, artifacts keep inventory
order, and a glob's matches are ordered newest-mtime-first with the path as the
tie-break. Nothing here reads the wall clock.

Command line
------------
    python3 -m weed_optimizer_framework.tools.brain.inventory_adapter \\
        --spec results/framework/supervision_bench/inventory.json \\
        --root . --out adapted.json [--collect-commands]

`--collect-commands` runs each accounting command with a 60 s timeout and stores
its output as an artifact beside the adapted spec; without it the command text
is recorded in the case notes and nothing is executed. The cluster is the only
place `sacct` answers, so the flag is off by default.

Pure stdlib; nothing raises out of `main()`.
"""
import argparse
import fnmatch
import glob as globmod
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

try:                                  # package import (python3 -m ...)
    from . import corpus
except ImportError:                   # plain-script import from this directory
    import corpus                     # noqa: F401

TOOL_VERSION = "wp2-inventory-adapter/1"

# --- entry classification --------------------------------------------------
# `artifact_paths_expected` mixes two kinds of string. The rule that separates
# them is explicit here rather than buried in a branch, because a command
# mistaken for a path is a silently missing artifact and a path mistaken for a
# command is an execution the export never asked for.
#
# A COMMAND is a string starting with one of these program names followed by a
# space. All three are SLURM accounting queries: they answer only on the
# cluster, they take no input from the repo, and their output is evidence about
# a job rather than a file the campaign produced.
COMMAND_PREFIXES = ("sacct ", "squeue ", "scontrol ")

# A GLOB is any remaining string containing one of these characters; it is
# expanded against `--root` at adaptation time and each match becomes its own
# artifact. Anything else is a path, relative to `--root` unless absolute.
GLOB_CHARS = "*?["

# Expansion cap. A pattern like `results/framework/*_44275211.out` can match a
# whole recipe's history; the bundle only has room for the recent ones, and a
# case whose artifact list is 200 rows long is not reviewable. Truncation is
# stated in the case notes, never silent.
GLOB_CAP = 8

COMMAND_TIMEOUT_S = 60

# --- bundle-section mapping ------------------------------------------------
# One row per section in corpus.SECTIONS (checked below), matched against the
# artifact's file name, first row wins. An artifact matching no row maps to
# `None`, which the exporter allows: a missing section makes a smaller bundle,
# a wrong section makes a bundle that misdescribes what the model was shown.
SECTION_RULES = (
    # SLURM stdout: every job log in the record ends `.out` (the `#SBATCH
    # --output` convention), and out_tail is the section that keeps line numbers.
    (("*.out",), "out_tail"),
    # One JSON object per line: every `.jsonl` in the record is an append-only
    # event stream (the harvest/train trace, cluster actions, labeling events).
    (("*.jsonl",), "trace"),
    # Ultralytics per-epoch table; the exporter reads its epoch and time columns.
    (("results.csv", "results_*.csv", "*_results.csv"), "results_csv"),
    # Accounting table, including the `sacct_<jobids>.txt` this module writes
    # when `--collect-commands` runs the query.
    (("sacct_*", "sacct.*", "*.sacct"), "sacct"),
    # Per-slug DINO scores, the file the `gate_noop` signal reads for staleness.
    (("slug_scores*.json",), "slug_scores"),
    # The job-scoped strategy dict the trainer writes beside its run.
    (("strategy*.json", "*_strategy.json"), "strategy"),
    # The round ledger, staged from the lab into the cluster-side bundle.
    (("ledger*.json", "*_ledger.json"), "ledger"),
    # `registry_lock.diff` output. A whole `dataset_registry.json` snapshot is
    # deliberately excluded: this section is a between-round diff, and a full
    # registry placed here would misstate what changed.
    (("*.diff", "registry_diff*.json", "registry_lock*.json"), "registry_diff"),
    # Per-source harvest report (`{"per_source": {...}}`).
    (("harvest*.json", "*_harvest.json"), "harvest"),
    # The df / quota block; `df_*.txt` is the shape a collected `df` would take.
    (("resources*.json", "df_*.txt"), "resources"),
    # SU accounting. Anchored on purpose — a bare `su*` would swallow
    # `summary.json`, which is not this section.
    (("su.json", "su_*.json"), "su"),
    # Active corrections staged from the lab.
    (("corrections*.json",), "corrections"),
    # The active plan. Anchored so `plain_*.json` cannot match it.
    (("plan.json", "plan_*.json"), "plan"),
    # Signal output, when a case carries a recorded signal run rather than a
    # bundle built live.
    (("signals*.json", "*_signals.json"), "signals"),
)

# The table must describe the exporter's sections and no others; a section name
# that drifts out of corpus.SECTIONS would be validated away one case at a time.
_RULE_SECTIONS = tuple(sec for _, sec in SECTION_RULES)
assert set(_RULE_SECTIONS) == set(corpus.SECTIONS), (
    "SECTION_RULES must cover exactly corpus.SECTIONS")

# --- escalation derivation -------------------------------------------------
# The inventory does not record an expected escalation, so it is derived, and
# the derivation is written into the case notes so a reviewer scores against a
# stated rule rather than against this file.
ESCALATION_RULES = {
    "control": ("none",
                "healthy or designed control: nothing to escalate"),
    "code": ("human",
             "correction_class=code: no agent tier may edit code"),
    "design": ("human",
               "correction_class=design: no agent tier may redesign an experiment"),
    "plan": ("human",
             "correction_class=plan: no agent tier may stop or redirect a phase"),
    "config": ("tier1",
               "correction_class=config: a bounded parameter change is tier1's reach"),
    "other": ("human",
              "correction_class outside the four edit kinds: escalate conservatively"),
}

# Fields the exporter reads out of a case. Everything else the inventory carries
# is folded into `notes` so the translation loses nothing.
_EXPORTER_FIELDS = frozenset((
    "case_id", "date", "incident", "class", "domain", "round", "step", "job_id",
    "signals_expected", "load_bearing_lines", "acceptable_corrections",
    "escalation_expected", "labels", "notes", "artifacts", "sections"))

# Inventory fields, in the order they are written into `notes`. The second entry
# is the label used in the notes block; `notes` is renamed so the inventory's own
# prose stays distinguishable from the derived lines above it.
_NOTES_ORDER = (
    ("title", "title"),
    ("one_line_symptom", "symptom"),
    ("root_cause", "root_cause"),
    ("correction_made", "correction_made"),
    ("correction_class", "correction_class"),
    ("actor_who_corrected", "actor_who_corrected"),
    ("version_tag", "version_tag"),
    ("category", "category"),
    ("scope", "scope"),
    ("healthy_control", "healthy_control"),
    ("job_ids", "job_ids"),
    ("detectable_from", "detectable_from"),
    ("artifact_paths_expected", "artifact_paths_expected"),
    ("provenance_guess", "provenance_guess"),
    ("provenance_reason", "provenance_reason"),
    ("notes", "inventory_notes"),
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
# A job-id argument: digits, optionally comma-joined (`44727703,44767709`) or
# array-suffixed (`44234060_2`). Used to name a collected command after the job
# it asks about, which is what a reader looks for.
_JOBID_TOKEN_RE = re.compile(r"^[0-9][0-9,_]*$")


# --- pure helpers ----------------------------------------------------------
def classify_entry(raw):
    """('command'|'glob'|'path', text). The rule is COMMAND_PREFIXES first."""
    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        return "path", ""
    for prefix in COMMAND_PREFIXES:
        if text.startswith(prefix):
            return "command", text
    if any(ch in text for ch in GLOB_CHARS):
        return "glob", text
    return "path", text


def section_for(name):
    """Bundle section for an artifact file name, or None when nothing fits."""
    low = (name or "").lower()
    for patterns, section in SECTION_RULES:
        for pat in patterns:
            if fnmatch.fnmatchcase(low, pat):
                return section
    return None


def safe_basename(raw_path):
    """A safe file name derived from a path's last segment.

    Safe means what the exporter checks for: no separator, not starting with a
    dot. Everything outside `[A-Za-z0-9._-]` collapses to `_`, so a glob pattern
    or a placeholder like `<jobid>` still yields a readable name. The original
    string is not lost — the exporter records the full path beside the name.
    """
    text = (raw_path or "").replace("\\", "/")
    segments = [s for s in text.split("/") if s.strip()]
    base = segments[-1] if segments else ""
    base = _SAFE_NAME_RE.sub("_", base).lstrip(".")
    return base or "artifact"


def unique_name(name, taken):
    """`name`, or `name` with a numeric suffix before the extension if taken."""
    if name not in taken:
        return name
    stem, ext = os.path.splitext(name)
    i = 2
    while "%s_%d%s" % (stem, i, ext) in taken:
        i += 1
    return "%s_%d%s" % (stem, i, ext)


def command_artifact_name(command):
    """File name for a collected command: `sacct_44727703.txt`.

    Named after the job ids the query asks about, which is the part a reader
    matches against a case. A query with no job-id argument (a date-ranged
    recovery query, say) is named by a short digest of the command instead, so
    the name stays deterministic and carries no account name or free text.
    """
    parts = shlex.split(command) if command else []
    prog = _SAFE_NAME_RE.sub("_", parts[0]) if parts else "command"
    ids = [t for t in parts[1:] if _JOBID_TOKEN_RE.match(t)]
    if ids:
        tail = "_".join(t.replace(",", "_") for t in ids)
    else:
        tail = "q" + hashlib.sha256(command.encode("utf-8")).hexdigest()[:8]
    return "%s_%s.txt" % (prog, tail)


def derive_class(entry):
    """`class` for the exporter: correction_class, or 'none' for a control."""
    cc = entry.get("correction_class")
    cc = cc.strip() if isinstance(cc, str) else ""
    return cc or "none"


def derive_escalation(entry):
    """(escalation_expected, reason). The reason is written into the notes."""
    if entry.get("healthy_control") or not entry.get("incident"):
        return ESCALATION_RULES["control"]
    cc = derive_class(entry)
    return ESCALATION_RULES.get(cc, ESCALATION_RULES["other"])


def expand_glob(pattern, root):
    """([absolute paths], truncated_count) for a glob, newest file first.

    Only files are returned: a directory cannot carry a citable line. Ordering is
    by mtime descending with the path as the tie-break, so two runs over an
    unchanged tree produce the same list.
    """
    base = pathlib.Path(str(root))
    joined = pattern if os.path.isabs(pattern) else str(base / pattern)
    hits = []
    for p in globmod.glob(joined):
        try:
            st = os.stat(p)
        except OSError:
            continue
        if not os.path.isfile(p):
            continue
        hits.append((-st.st_mtime, p))
    hits.sort()
    kept = [p for _, p in hits[:GLOB_CAP]]
    return kept, max(0, len(hits) - GLOB_CAP)


def _fmt(value):
    """One inventory value as a compact notes line."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(_fmt(v) for v in value)
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_notes(entry, derived_lines):
    """The notes block: derived lines first, then every unconsumed field.

    Ordering is fixed so the block is diffable, and any inventory key this
    module has never seen is appended rather than dropped — the inventory is the
    record, and a field it grows later must survive the translation.
    """
    lines = list(derived_lines)
    for key, label in _NOTES_ORDER:
        if key not in entry:
            continue
        text = _fmt(entry.get(key))
        if text:
            lines.append("%s: %s" % (label, text))
    known = {k for k, _ in _NOTES_ORDER} | _EXPORTER_FIELDS
    for key in sorted(entry):
        if key in known:
            continue
        text = _fmt(entry.get(key))
        if text:
            lines.append("%s: %s" % (key, text))
    return "\n".join(lines)


# --- command collection ----------------------------------------------------
def run_command(command, timeout=COMMAND_TIMEOUT_S):
    """(body, ok) for one command. Never raises.

    A failure is recorded as the artifact body with a stated first line, because
    an accounting query that cannot answer is itself evidence about the case —
    an empty file would look like a query that returned nothing.
    """
    header = "$ %s" % command
    try:
        proc = subprocess.run(shlex.split(command), timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return ("[command failed] no output within %d s\n%s\n" % (timeout, header)), False
    except (OSError, ValueError) as exc:
        return ("[command failed] %s: %s\n%s\n"
                % (type(exc).__name__, exc, header)), False
    out = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    if proc.returncode != 0:
        return ("[command failed] exit=%d\n%s\n%s%s"
                % (proc.returncode, header, out, err)), False
    return "%s\n%s" % (header, out), True


def _write_collected(collect_dir, case_id, name, body):
    """Write one collected command body; returns its path or None on failure."""
    d = pathlib.Path(str(collect_dir)) / case_id
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        with open(str(p), "w", encoding="utf-8") as f:
            f.write(body)
        return p
    except OSError:
        return None


# --- one case --------------------------------------------------------------
def adapt_case(entry, root, collect_dir=None, runner=run_command):
    """(case, stats) for one inventory entry. Writes only collected commands."""
    stats = {"artifacts": 0, "present": 0, "missing": 0, "not_a_file": 0,
             "commands_collected": 0, "commands_skipped": 0,
             "globs_expanded": 0, "globs_unmatched": 0, "globs_capped": 0,
             "sections": {}}
    case_id = entry.get("case_id") if isinstance(entry.get("case_id"), str) else ""
    root = pathlib.Path(str(root))
    artifacts = []
    taken = set()
    derived = []

    def add(name, path_text, note=None):
        nm = unique_name(name, taken)
        taken.add(nm)
        sec = section_for(nm)
        art = {"name": nm, "path": path_text}
        if sec:
            art["section"] = sec
        artifacts.append(art)
        stats["artifacts"] += 1
        key = sec or "unmapped"
        stats["sections"][key] = stats["sections"].get(key, 0) + 1
        if note:
            derived.append(note)
        return nm

    for raw in (entry.get("artifact_paths_expected") or []):
        kind, text = classify_entry(raw)
        if not text:
            continue
        if kind == "command":
            name = command_artifact_name(text)
            if collect_dir is None:
                stats["commands_skipped"] += 1
                derived.append("command_not_collected: %s" % text)
                continue
            body, ok = runner(text)
            written = _write_collected(collect_dir, case_id or "unnamed", name, body)
            if written is None:
                stats["commands_skipped"] += 1
                derived.append("command_not_stored: %s" % text)
                continue
            stats["commands_collected"] += 1
            stats["present"] += 1
            add(name, str(written),
                "command_collected%s: %s" % ("" if ok else " (failed)", text))
            continue
        if kind == "glob":
            matches, truncated = expand_glob(text, root)
            if not matches:
                stats["globs_unmatched"] += 1
                stats["missing"] += 1
                add(safe_basename(text), text,
                    "glob_unmatched: %s" % text)
                continue
            stats["globs_expanded"] += 1
            if truncated:
                stats["globs_capped"] += 1
                derived.append("glob_truncated: %s kept %d newest of %d matches"
                               % (text, len(matches), len(matches) + truncated))
            for m in matches:
                stats["present"] += 1
                add(safe_basename(m), m)
            continue
        # A plain path. Existence is checked here so the run can report what the
        # record predicted and this machine does not have; the entry is emitted
        # either way, and the exporter books an unreadable one as record-only.
        resolved = pathlib.Path(text) if os.path.isabs(text) else root / text
        if resolved.is_file():
            stats["present"] += 1
        elif resolved.exists():
            stats["not_a_file"] += 1
        else:
            stats["missing"] += 1
        add(safe_basename(text), text)

    escalation, why = derive_escalation(entry)
    derived.insert(0, "escalation_expected: %s (rule: %s)" % (escalation, why))
    cls = derive_class(entry)
    derived.insert(0, "class: %s (from correction_class%s)"
                   % (cls, "" if entry.get("correction_class") else ", empty = none"))

    job_ids = [str(j) for j in (entry.get("job_ids") or []) if str(j).strip()]
    case = {
        "case_id": case_id,
        "date": entry.get("date") if isinstance(entry.get("date"), str) else "",
        "incident": bool(entry.get("incident")),
        "class": cls,
        "escalation_expected": escalation,
        "signals_expected": list(entry.get("signals_expected") or []),
        "artifacts": artifacts,
        "labels": {"pre_registered": {
            "detect": bool(entry.get("incident")),
            "correction_class": cls,
        }},
        "notes": build_notes(entry, derived),
    }
    if job_ids:
        case["job_id"] = job_ids[0]
    return case, stats


# --- whole inventory -------------------------------------------------------
def adapt(spec, root=None, collect_dir=None, runner=run_command):
    """Adapt a parsed inventory. Returns {'spec': ..., 'report': ...}."""
    if isinstance(spec, list):
        spec = {"cases": spec}
    if not isinstance(spec, dict):
        spec = {}
    entries = spec.get("cases")
    entries = entries if isinstance(entries, list) else []
    root = pathlib.Path(str(root or corpus.REPO))
    report = {"cases": 0, "skipped": 0, "artifacts": 0, "artifacts_present": 0,
              "paths_missing": 0, "paths_not_a_file": 0,
              "commands_collected": 0, "commands_skipped": 0,
              "globs_expanded": 0, "globs_unmatched": 0, "globs_capped": 0,
              "sections": {}, "root": str(root), "errors": []}
    cases = []
    for entry in entries:
        if not isinstance(entry, dict):
            report["skipped"] += 1
            report["errors"].append("a case entry is not an object; skipped")
            continue
        case, stats = adapt_case(entry, root, collect_dir=collect_dir, runner=runner)
        cases.append(case)
        report["cases"] += 1
        report["artifacts"] += stats["artifacts"]
        report["artifacts_present"] += stats["present"]
        report["paths_missing"] += stats["missing"]
        report["paths_not_a_file"] += stats["not_a_file"]
        report["commands_collected"] += stats["commands_collected"]
        report["commands_skipped"] += stats["commands_skipped"]
        report["globs_expanded"] += stats["globs_expanded"]
        report["globs_unmatched"] += stats["globs_unmatched"]
        report["globs_capped"] += stats["globs_capped"]
        for k, v in stats["sections"].items():
            report["sections"][k] = report["sections"].get(k, 0) + v

    out_spec = {"_adapter": {"tool_version": TOOL_VERSION,
                             "source_fields": "correction_class -> class; derived "
                                              "escalation_expected; "
                                              "artifact_paths_expected -> artifacts",
                             "glob_cap": GLOB_CAP,
                             "command_prefixes": list(COMMAND_PREFIXES)},
                "cases": cases}
    # Carried through so the exporter's scrub configuration still comes from the
    # inventory rather than from this module.
    for key in ("root", "domain", "scrub_users", "scrub_literals", "scrub_repo_paths"):
        if key in spec:
            out_spec[key] = spec[key]
    return {"spec": out_spec, "report": report}


def adapt_file(spec_path, out_path, root=None, collect_commands=False,
               runner=run_command):
    """Read an inventory, write the adapted spec, return the report."""
    spec = corpus.read_json(spec_path)
    if spec is None:
        return {"ok": False, "report": {"errors": [
            "inventory not readable as JSON: %s" % spec_path]}}
    out_path = pathlib.Path(str(out_path))
    collect_dir = None
    if collect_commands:
        collect_dir = out_path.parent / (out_path.stem + "_commands")
    result = adapt(spec, root=root, collect_dir=collect_dir, runner=runner)
    corpus.write_json(out_path, result["spec"])
    result["ok"] = not result["report"]["errors"]
    result["out"] = str(out_path)
    result["spec_path"] = str(spec_path)
    if collect_dir is not None:
        result["report"]["collect_dir"] = str(collect_dir)
    return result


# --- CLI -------------------------------------------------------------------
def _print_report(result):
    rep = result.get("report") or {}
    print("adapt  spec=%s  root=%s  out=%s"
          % (result.get("spec_path"), rep.get("root"), result.get("out")))
    print("  %d case(s) adapted, %d skipped" % (rep.get("cases", 0),
                                                rep.get("skipped", 0)))
    print("  %d artifact(s): %d resolved to a real file, %d path(s) not on this "
          "machine, %d path(s) present but not a file"
          % (rep.get("artifacts", 0), rep.get("artifacts_present", 0),
             rep.get("paths_missing", 0), rep.get("paths_not_a_file", 0)))
    print("  commands: %d collected, %d recorded in notes only"
          % (rep.get("commands_collected", 0), rep.get("commands_skipped", 0)))
    print("  globs: %d expanded, %d matched nothing, %d truncated at the cap of %d"
          % (rep.get("globs_expanded", 0), rep.get("globs_unmatched", 0),
             rep.get("globs_capped", 0), GLOB_CAP))
    if rep.get("collect_dir"):
        print("  collected command output under %s" % rep["collect_dir"])
    hist = rep.get("sections") or {}
    print("  sections:")
    for key in sorted(hist):
        print("    %-14s %d" % (key, hist[key]))
    for e in (rep.get("errors") or []):
        print("  ERROR %s" % e)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="inventory_adapter",
        description="Adapt the WP2 incident inventory to the corpus case shape")
    ap.add_argument("--spec", required=True, help="inventory JSON to read")
    ap.add_argument("--out", required=True, help="adapted spec JSON to write")
    ap.add_argument("--root", default=None,
                    help="base for relative artifact paths (default: %s)" % corpus.REPO)
    ap.add_argument("--collect-commands", action="store_true",
                    help="run each accounting command (60 s timeout) and store its "
                         "output as an artifact; off by default because sacct only "
                         "answers on the cluster")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:                 # argparse's own exit, including --help
        return int(exc.code or 0)
    try:
        result = adapt_file(a.spec, a.out, root=a.root,
                            collect_commands=a.collect_commands)
        if a.json:
            print(json.dumps(result.get("report") or {}, sort_keys=True, indent=1))
        else:
            _print_report(result)
        return 0 if result.get("ok") else 1
    except Exception as exc:
        # A top-level entry point reports what it could not do; it does not die
        # with a traceback in the middle of an export run.
        print("inventory_adapter: %s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
