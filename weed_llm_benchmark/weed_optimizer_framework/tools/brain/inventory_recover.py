#!/usr/bin/env python3
"""Recover un-listed evidence artifacts for the WP2 incident inventory.

Why this exists
----------------
Twelve deterministic signal checks read named sections of a case's evidence
bundle (`corpus.SECTIONS`). Measured across the real 162-case inventory,
`results_csv` fills in 17 cases and `strategy` in 10 -- yet nearly every
training case that ran under `run_m1_merged_seeds.sh` (and its predecessors)
left a `results.csv`, a job-scoped recipe JSON and a job log on shared storage.
The inventory's own `artifact_paths_expected` list simply never named them: a
case was written down as "job 44767709 timed out" with the `.out` and a
`sacct` command, and nobody went back to add the run directory the same job
also produced. That is a listing gap, not an archival one, and it is closable
without touching a single truth field.

This module closes it by deriving each case's job ids from data the inventory
already carries (`job_ids`, occasionally suffixed as a SLURM array task id) and
searching the storage tree for files that carry independently checkable
evidence of belonging to that job:

* the job id appears in the artifact's own file name (the `%j` SBATCH output
  convention, `command_artifact_name`'s `sacct_<jobid>.txt`);
* the job id appears in an ancestor directory name (`run_m1_merged_seeds.sh`
  sets `RUN_TAG="job$SLURM_JOB_ID"` specifically so a rerun's `results.csv`
  cannot be silently attributed to the wrong job -- the exact property this
  tool leans on to attribute it *at all*);
* a strategy JSON already matched by one of the two rules above is opened and
  found to record the exact `save_dir` (or `trace_path`) ultralytics wrote to
  -- the run's own recipe naming its own run directory, which is the strongest
  evidence there is because nothing was inferred from a name at all;
* a job's accounting window (`Start`..`End`, read from a `sacct` artifact
  already matched by one of the rules above via `corpus.parse_sacct_text`, the
  one sacct parser) contains a candidate file's mtime -- real but weak on a
  cluster running many jobs at once, so it is never enough by itself.

What it does not do
--------------------
It does not guess. A file merely sitting in a plausibly-named directory earns
no proposal at all: every proposal carries at least one of the evidence kinds
above, spelled out, and the weakest of them (mtime) never crosses the default
apply threshold. It does not invent an artifact: a case whose job ids match
nothing under `--root` says so in its `notes`, and is left exactly as thin as
its own record. It does not touch a case's truth. `apply` writes into
`artifact_paths_expected` and nothing else -- `signals_expected`, `class`,
`date`, `incident`, `job_ids`, every other field a label depends on, are read
here and never assigned to; that is a pre-registration boundary, not a style
choice, and `apply_proposal` is written so there is exactly one place a
mutation can happen. Section names are never invented either: what a
recovered path maps to is decided by `inventory_adapter.section_for`, the same
function the exporter itself will apply later, so there is one section table
in the whole pipeline, not two that can drift apart.

Determinism
-----------
The storage walk visits directories and files in sorted order and the
resulting index is sorted by relative path before anything reads it, so two
runs over an unchanged tree produce byte-identical proposals. Nothing here
reads the wall clock; the only times involved are artifact mtimes and sacct
`Start`/`End` fields, which are themselves evidence.

Bounded search
---------------
A directory this tool must never descend into is a large or image-filled one
-- the campaign has already paid once for treating a Lustre image directory
like an ordinary one. `SKIP_DIR_NAMES` names the obvious ones (including this
corpus's own frozen export, which is not source evidence); `MAX_DIR_ENTRIES`
catches anything not on that list by size instead of by name, and every skip
is recorded rather than silently absorbed. `MAX_PROPOSALS_PER_CASE` caps what
a reviewer is asked to look at per case, mirroring `inventory_adapter.GLOB_CAP`;
a cap that bites is stated in that case's `notes`, never truncated quietly.

Command line
------------
    scan   --spec <inventory.json> --root <repo root> [--out proposal.json]
    apply  --proposal <file> --spec <inventory.json> [--min-confidence X] [--out <file>]
    report --proposal <file>

`scan` never writes to the inventory. `apply` writes only `artifact_paths_expected`,
appended in a stable order (confidence, then path), never removing an entry
already there. Running `scan` then `apply` twice changes nothing the second
time: the second `apply` finds every accepted path already present and adds
none of it again.

Pure stdlib; nothing raises out of `main()`.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

try:                                  # package import (python3 -m ...)
    from . import corpus
    from . import inventory_adapter as ia
except ImportError:                   # plain-script import from this directory
    import corpus                     # noqa: F401
    import inventory_adapter as ia    # noqa: F401

TOOL_VERSION = "wp2-inventory-recover/1"

# --- bounded search ----------------------------------------------------------
# Directories never descended into. Each one either holds many thousands of
# small files a case's evidence could never plausibly live beside (an image
# corpus, a labeling cache, a venv) or is this tool's own blind spot by
# construction: `supervision_bench` is the corpus exporter's frozen, scrubbed
# output, and re-discovering it here as "found evidence" would be circular.
SKIP_DIR_NAMES = frozenset((
    "images", "img", "imgs", "thumbs", "thumbnails", "labels", "downloads",
    "class_pool", "class_exemplars", "dataset_analysis", "cache",
    "llm_labeled", "cottonweeddet12", "node_modules", "site-packages",
    ".git", ".venv", "venv", "__pycache__", "supervision_bench",
))

# A directory this large is a bulk data dump whatever its name; recording the
# skip beats silently truncating it or, worse, walking it anyway.
MAX_DIR_ENTRIES = 2000

# Safety valve against a pathological tree. Never expected to bite on this
# repository; present because a search that cannot terminate is not bounded.
MAX_DIRS_VISITED = 200000

# A case whose evidence review has to include dozens of proposed files is not
# reviewable one artifact at a time -- the same reasoning as
# `inventory_adapter.GLOB_CAP`, at case rather than glob granularity.
MAX_PROPOSALS_PER_CASE = 12

# At most this many weak mtime-window proposals per case: the window is real
# but says nothing about which of several jobs running at once produced a
# given file, so it is capped hard rather than allowed to flood a case.
MAX_MTIME_PROPOSALS_PER_CASE = 3

# A recipe or sacct artifact worth opening is a few KB; this is generous
# headroom over the largest one seen and stops a misclassified multi-MB file
# from being parsed as if it were one.
MAX_SIDE_READ_BYTES = 1 << 20

# --- confidence ---------------------------------------------------------------
# The file's own name (or its immediate run directory's name) carrying the job
# id is the mechanism `run_m1_merged_seeds.sh` is written to guarantee
# (`RUN_TAG="job$SLURM_JOB_ID"`, `%j` in `--output`); it is evidence, not an
# inference. A recipe JSON already matched by job id, opened to read the exact
# path it recorded, is stronger still: the artifact's own author named it. A
# shared mtime window is real but weak on a cluster running many jobs
# concurrently, so it sits well under the default apply threshold.
CONF_FILENAME = 0.95
CONF_RECIPE_PATH = 0.93
CONF_DIRNAME = 0.90
CONF_MTIME_WINDOW = 0.35
DEFAULT_MIN_CONFIDENCE_APPLY = 0.80

# Band labels for reporting only; `apply` compares the numeric confidence to
# `--min-confidence` directly. "medium" is reserved for a future evidence kind
# weaker than a direct id match and stronger than mtime; nothing here produces
# it today.
_BAND_HIGH, _BAND_MEDIUM = 0.8, 0.5


def _band(confidence):
    if confidence >= _BAND_HIGH:
        return "high"
    if confidence >= _BAND_MEDIUM:
        return "medium"
    return "low"


# Fixed, explicit keys read out of an already-matched recipe/strategy JSON.
# Deliberately a small allow-list rather than a scan for anything path-shaped:
# a blind scan would treat any string containing a slash as a path to chase,
# which is a guess wearing a confidence number.
RECIPE_PATH_KEYS = ("save_dir", "trace_path", "best_pt")
RECIPE_SUMMARY_KEY = "summary"

# Sections a compute job could plausibly have produced, for the mtime-window
# hop only. `out_tail`/`sacct` are excluded there on purpose: those need a
# direct id match to be trusted at all, and letting mtime alone attribute a job
# log would misattribute exactly the artifact a citation depends on.
_MTIME_SECTIONS = frozenset(("results_csv", "strategy", "trace", "harvest",
                             "slug_scores"))


# --- pure helpers --------------------------------------------------------
def _case_list(spec):
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        cases = spec.get("cases")
        return cases if isinstance(cases, list) else []
    return []


def _job_ids_of(entry):
    return [str(j).strip() for j in (entry.get("job_ids") or []) if str(j).strip()]


def _search_tokens(raw_job_id):
    """[(token, is_derived), ...] to search for, from one recorded job id.

    A SLURM array task is recorded as `<array_job_id>_<task_id>` (sacct's own
    external form for it), but `run_m1_merged_seeds.sh` names its job-scoped
    recipe file after `JOB_KEY` (`SLURM_ARRAY_JOB_ID`, the numeric prefix
    only). Both are searched. The derived one is a real fact about the
    submission read straight off the recorded id, not a guess: an array task
    id always has its own array job id as a prefix, by construction.
    """
    raw = str(raw_job_id).strip()
    out = [(raw, False)]
    if "_" in raw:
        base = raw.split("_", 1)[0]
        if base and base != raw:
            out.append((base, True))
    return out


def _relstr(path, root):
    rel = os.path.relpath(str(path), str(root))
    return rel.replace(os.sep, "/")


def _within_root(path, root):
    try:
        a = os.path.abspath(str(path))
        r = os.path.abspath(str(root))
        return os.path.commonpath([a, r]) == r
    except ValueError:              # different drives on Windows; never on the cluster
        return False


def existing_abs_paths(entry, root):
    """Absolute paths this case's `artifact_paths_expected` already covers.

    Reuses `inventory_adapter`'s own classification and glob expansion rather
    than re-deriving what counts as a command/glob/path: a second reading of
    that list would be the exact drift risk the adapter's docstring warns
    about for the sacct parser, just relocated to this file.
    """
    out = set()
    for raw in (entry.get("artifact_paths_expected") or []):
        kind, text = ia.classify_entry(raw)
        if not text or kind == "command":
            continue
        if kind == "glob":
            matches, _truncated = ia.expand_glob(text, root)
            out.update(os.path.abspath(m) for m in matches)
            continue
        resolved = pathlib.Path(text) if os.path.isabs(text) else pathlib.Path(str(root)) / text
        out.add(os.path.abspath(str(resolved)))
    return out


def _job_regex(tokens):
    tokens = sorted({t for t in tokens if t}, key=len, reverse=True)
    if not tokens:
        return None
    # `(?<!\d)...(?!\d)`: a shorter id can be a true prefix of a longer one
    # (an array base id of a longer array task id), and the boundary makes the
    # engine backtrack into the longer alternative rather than stopping short
    # -- so a decoy job id that merely shares a prefix never matches.
    return re.compile(r"(?<!\d)(" + "|".join(re.escape(t) for t in tokens) + r")(?!\d)")


# --- storage walk ----------------------------------------------------------
def walk_storage(root):
    """(index, skipped, stats) -- one bounded pass over `root`.

    `index` holds only files `inventory_adapter.section_for` recognises as a
    bundle-section artifact shape; nothing else is worth indexing, and
    restricting the walk to that shape keeps it small even before any
    directory skip rule fires. Every skip -- by name, by size, by the global
    directory cap -- is recorded in `skipped`, never silently absorbed.
    """
    root = pathlib.Path(str(root))
    index = []
    skipped = []
    stats = {"dirs_visited": 0, "files_seen": 0, "files_indexed": 0,
             "dirs_skipped_by_name": 0, "dirs_skipped_by_size": 0,
             "dirs_visit_cap_hit": False}
    stack = [root]
    while stack:
        d = stack.pop()
        if stats["dirs_visited"] >= MAX_DIRS_VISITED:
            stats["dirs_visit_cap_hit"] = True
            skipped.append({"path": _relstr(d, root),
                            "reason": "global directory-visit cap (%d) reached; "
                                      "the rest of the tree was not scanned"
                                      % MAX_DIRS_VISITED})
            break
        stats["dirs_visited"] += 1
        try:
            entries = sorted(os.scandir(str(d)), key=lambda e: e.name)
        except OSError as exc:
            skipped.append({"path": _relstr(d, root),
                            "reason": "not readable: %s" % exc})
            continue
        if len(entries) > MAX_DIR_ENTRIES:
            stats["dirs_skipped_by_size"] += 1
            skipped.append({"path": _relstr(d, root),
                            "reason": "%d entries exceeds the %d-entry cap; "
                                      "treated as a bulk data directory and not "
                                      "descended" % (len(entries), MAX_DIR_ENTRIES)})
            continue
        subdirs = []
        for e in entries:
            try:
                is_dir = e.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if e.name.startswith(".") or e.name.lower() in SKIP_DIR_NAMES:
                    stats["dirs_skipped_by_name"] += 1
                    continue
                subdirs.append(pathlib.Path(e.path))
                continue
            stats["files_seen"] += 1
            sec = ia.section_for(e.name)
            if sec is None:
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            index.append({"abs": os.path.abspath(e.path), "rel": _relstr(e.path, root),
                          "name": e.name, "section": sec, "mtime": st.st_mtime})
            stats["files_indexed"] += 1
        # Reversed so the stack (LIFO) still visits subdirectories in sorted
        # order overall -- determinism, not correctness.
        stack.extend(reversed(subdirs))
    index.sort(key=lambda r: r["rel"])
    skipped.sort(key=lambda r: r["path"])
    return index, skipped, stats


# --- job-id matching (hop 1) ------------------------------------------------
def build_job_index(index, all_job_ids):
    """(job_index, token_map) over the walked index.

    `job_index` maps a search token to every `(item, kind, component)` it
    matched, `kind` being `"filename"` when the token is in the file's own
    name and `"dirname"` when it is only in an ancestor directory name.
    `token_map` maps a token back to the recorded job id(s) it was derived
    from, so a case looks itself up by its own tokens without re-deriving them.
    """
    token_map = {}
    for jid in all_job_ids:
        for tok, derived in _search_tokens(jid):
            token_map.setdefault(tok, set()).add((jid, derived))
    job_index = {}
    regex = _job_regex(token_map.keys())
    if regex is None:
        return job_index, token_map
    for item in index:
        parts = item["rel"].split("/")
        last = len(parts) - 1
        for i, part in enumerate(parts):
            kind = "filename" if i == last else "dirname"
            for m in sorted(set(regex.findall(part))):
                job_index.setdefault(m, []).append((item, kind, part))
    return job_index, token_map


# --- recipe-named path (hop 2) ----------------------------------------------
def _open_json_bounded(path):
    try:
        if os.path.getsize(path) > MAX_SIDE_READ_BYTES:
            return None
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def recipe_named_paths(obj):
    """[(key, raw_path), ...] for the fixed allow-list of path-shaped keys."""
    out = []
    if not isinstance(obj, dict):
        return out
    for key in RECIPE_PATH_KEYS:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            out.append((key, v.strip()))
    summary = obj.get(RECIPE_SUMMARY_KEY)
    if isinstance(summary, dict):
        for key in RECIPE_PATH_KEYS:
            v = summary.get(key)
            if isinstance(v, str) and v.strip():
                out.append(("summary.%s" % key, v.strip()))
    return out


def _resolve_recipe_path(raw, root):
    p = pathlib.Path(raw)
    resolved = p if p.is_absolute() else pathlib.Path(str(root)) / p
    return resolved if _within_root(resolved, root) else None


def _artifact_at(path):
    """The file a recipe-named path resolves to: itself, or its `results.csv`."""
    try:
        if path.is_dir():
            cand = path / "results.csv"
            return cand if cand.is_file() else None
        if path.is_file():
            return path
    except OSError:
        pass
    return None


def extend_with_recipe_hop(job_index, root):
    """New `(item, "recipe_path", detail)` entries, keyed by the token that led
    to them, from opening every already-matched strategy JSON.

    A strategy artifact only reaches this function because its own name or
    directory already carried a case's job id (`job_index` is built from hop 1
    alone before this runs); what it records inside is therefore read as this
    job's own account of where it wrote its results, not as a guess about an
    unrelated file.
    """
    extra = {}
    opened = {}
    stats = {"opened": 0, "paths_found": 0, "paths_resolved": 0}
    for token, matches in sorted(job_index.items()):
        for item, _kind, _part in matches:
            if item["section"] != "strategy":
                continue
            key = item["abs"]
            if key not in opened:
                obj = _open_json_bounded(key)
                opened[key] = recipe_named_paths(obj) if obj is not None else []
                stats["opened"] += 1
            for rkey, raw_path in opened[key]:
                stats["paths_found"] += 1
                resolved = _resolve_recipe_path(raw_path, root)
                if resolved is None:
                    continue
                found = _artifact_at(resolved)
                if found is None:
                    continue
                stats["paths_resolved"] += 1
                rel = _relstr(found, root)
                sec = ia.section_for(os.path.basename(str(found)))
                try:
                    mtime = os.path.getmtime(str(found))
                except OSError:
                    mtime = None
                synth = {"abs": os.path.abspath(str(found)), "rel": rel,
                        "name": os.path.basename(str(found)), "section": sec,
                        "mtime": mtime}
                detail = ("recipe %s (already matched to this case by job id) "
                          "records %s=%s, resolving to %s"
                          % (item["rel"], rkey, raw_path, rel))
                extra.setdefault(token, []).append((synth, "recipe_path", detail))
    return extra, stats


# --- mtime window (hop 3, weak) ---------------------------------------------
def _sacct_epoch(text):
    text = (text or "").strip()
    if not text or text.lower() in ("unknown", "none"):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _parse_sacct_window(accept_ids, sacct_path):
    """(start, end) epoch for the first row in `sacct_path` matching an id in
    `accept_ids`, or None. Uses `corpus.parse_sacct_text` -- the one parser.
    """
    try:
        if os.path.getsize(sacct_path) > MAX_SIDE_READ_BYTES:
            return None
        with open(str(sacct_path), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    rows, _reason = corpus.parse_sacct_text(text)
    for row in rows:
        rid = str(row.get("JobID") or row.get("JobIDRaw") or "").strip()
        rid_base = rid.split(".", 1)[0]
        if rid not in accept_ids and rid_base not in accept_ids:
            continue
        start = _sacct_epoch(row.get("Start"))
        end = _sacct_epoch(row.get("End"))
        if start is not None and end is not None and end >= start:
            return start, end
    return None


def _find_case_window(jids, candidates):
    accept = set(jids)
    for jid in jids:
        for tok, _derived in _search_tokens(jid):
            accept.add(tok)
    for abs_path, cand in sorted(candidates.items()):
        if cand["section"] != "sacct" or cand["confidence"] < CONF_DIRNAME:
            continue
        window = _parse_sacct_window(accept, abs_path)
        if window:
            return window
    return None


def _fmt_epoch(e):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(e))


def _add_mtime_candidates(candidates, index, jids, window, existing, cap):
    start, end = window
    added = 0
    who = ", ".join(sorted(jids))
    for item in index:                       # already sorted by rel path
        if added >= cap:
            break
        if item["section"] not in _MTIME_SECTIONS:
            continue
        if item["abs"] in existing or item["abs"] in candidates:
            continue
        mt = item.get("mtime")
        if mt is None or not (start <= mt <= end):
            continue
        candidates[item["abs"]] = {
            "path": item["rel"], "section": item["section"],
            "confidence": CONF_MTIME_WINDOW,
            "evidence": [{"kind": "mtime_in_job_window",
                          "detail": ("mtime falls inside job %s's sacct window "
                                     "(%s .. %s); mtime alone does not identify "
                                     "which of possibly several concurrent jobs "
                                     "produced this file, so it is not applied "
                                     "by default"
                                     % (who, _fmt_epoch(start), _fmt_epoch(end)))}],
        }
        added += 1


# --- per-case assembly -------------------------------------------------------
def _reason_for(kind, job_id, token, derived, component, item):
    if kind == "recipe_path":
        return {"kind": "recipe_named_path", "detail": component}
    who = (("array base job id %s (from recorded task id %s)" % (token, job_id))
           if derived else ("job id %s" % job_id))
    if kind == "filename":
        return {"kind": "job_id_in_filename",
                "detail": "%s appears in the file name %s" % (who, item["name"])}
    return {"kind": "job_id_in_dirname",
            "detail": "%s appears in run directory %s" % (who, component)}


def _merge_candidate(candidates, item, kind, job_id, token, derived, component, existing):
    abs_path = item["abs"]
    if abs_path in existing:
        return
    if kind == "recipe_path":
        conf = CONF_RECIPE_PATH
    elif kind == "filename":
        conf = CONF_FILENAME
    else:
        conf = CONF_DIRNAME
    reason = _reason_for(kind, job_id, token, derived, component, item)
    slot = candidates.get(abs_path)
    if slot is None:
        candidates[abs_path] = {"path": item["rel"], "section": item["section"],
                                "confidence": conf, "evidence": [reason]}
        return
    slot["confidence"] = max(slot["confidence"], conf)
    if reason not in slot["evidence"]:
        slot["evidence"].append(reason)


def _case_candidates(entry, job_index, index, root):
    """(candidates: {abs_path: proposal}, notes: [str]) for one case."""
    jids = _job_ids_of(entry)
    existing = existing_abs_paths(entry, root)
    notes = []
    candidates = {}
    if not jids:
        notes.append("no job id on record for this case (job_ids is empty); "
                     "nothing to search for")
        return candidates, notes
    for jid in jids:
        for tok, derived in _search_tokens(jid):
            for item, kind, part in job_index.get(tok, []):
                _merge_candidate(candidates, item, kind, jid, tok, derived, part, existing)
    window = _find_case_window(jids, candidates)
    if window is not None:
        _add_mtime_candidates(candidates, index, jids, window, existing,
                              cap=MAX_MTIME_PROPOSALS_PER_CASE)
    if not candidates:
        notes.append("no artifact under the search root carries job id(s) %s; "
                     "this job's evidence did not survive, or was never "
                     "produced under this root" % ", ".join(jids))
    return candidates, notes


def build_proposal(spec, root):
    """The full proposal structure for `spec` over the tree at `root`."""
    root = pathlib.Path(str(root))
    cases = _case_list(spec)
    all_job_ids = sorted({j for c in cases if isinstance(c, dict)
                          for j in _job_ids_of(c)})
    index, skipped_dirs, walk_stats = walk_storage(root)
    job_index, _token_map = build_job_index(index, all_job_ids)
    extra, recipe_stats = extend_with_recipe_hop(job_index, root)
    for tok, lst in extra.items():
        job_index.setdefault(tok, []).extend(lst)

    out_cases = []
    report = {"cases": 0, "cases_with_job_ids": 0, "cases_without_job_ids": 0,
              "cases_with_no_match": 0, "cases_capped": 0,
              "proposals": 0,
              "proposals_by_confidence_band": {"high": 0, "medium": 0, "low": 0},
              "proposals_by_section": {}, "walk": walk_stats,
              "recipe_hop": recipe_stats, "skipped_dirs": len(skipped_dirs),
              "skipped_dir_samples": skipped_dirs[:50]}

    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case_id = entry.get("case_id") if isinstance(entry.get("case_id"), str) else ""
        jids = _job_ids_of(entry)
        candidates, notes = _case_candidates(entry, job_index, index, root)
        prop_list = sorted(candidates.values(),
                           key=lambda p: (-p["confidence"], p["path"]))
        capped = False
        if len(prop_list) > MAX_PROPOSALS_PER_CASE:
            capped = True
            dropped = len(prop_list) - MAX_PROPOSALS_PER_CASE
            prop_list = prop_list[:MAX_PROPOSALS_PER_CASE]
            notes.append("proposal cap (%d) reached; %d lower-ranked match(es) "
                         "were not included" % (MAX_PROPOSALS_PER_CASE, dropped))
            report["cases_capped"] += 1

        report["cases"] += 1
        if jids:
            report["cases_with_job_ids"] += 1
        else:
            report["cases_without_job_ids"] += 1
        if jids and not prop_list:
            report["cases_with_no_match"] += 1
        for p in prop_list:
            report["proposals"] += 1
            report["proposals_by_confidence_band"][_band(p["confidence"])] += 1
            sec_key = p["section"] or "unmapped"
            report["proposals_by_section"][sec_key] = \
                report["proposals_by_section"].get(sec_key, 0) + 1

        out_cases.append({
            "case_id": case_id,
            "job_ids": jids,
            # Read-only echo for `report`'s per-signal breakdown; `apply` never
            # reads this field. Copying it here does not widen the case's
            # truth -- it is not written back into the inventory by anything.
            "signals_expected": list(entry.get("signals_expected") or []),
            "existing_artifact_count": len(entry.get("artifact_paths_expected") or []),
            "capped": capped,
            "notes": notes,
            "proposals": prop_list,
        })

    return {"_tool": {"tool_version": TOOL_VERSION, "root": str(root)},
            "cases": out_cases, "report": report}


# --- apply -------------------------------------------------------------------
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def apply_proposal(spec, proposal, min_confidence=DEFAULT_MIN_CONFIDENCE_APPLY):
    """Append accepted proposals into `spec`'s cases, in place.

    The only field ever assigned to is `artifact_paths_expected`, and only by
    appending: an entry already present (by exact string) is never duplicated
    and nothing already there is ever removed. Every other key on a case --
    `signals_expected`, `class`/`correction_class`, `date`, `incident`,
    `job_ids`, and everything else a pre-registered label rests on -- is never
    read for a decision here and never assigned to; that is the boundary this
    tool exists inside of.

    Returns {"spec": spec, "report": {...}}. Idempotent: calling this again
    with the same `proposal` against the returned `spec` changes nothing,
    because every accepted path is already present.
    """
    by_case = {c.get("case_id"): c for c in (proposal.get("cases") or [])
              if isinstance(c, dict) and c.get("case_id")}
    report = {"cases_seen": 0, "cases_changed": 0, "artifacts_added": 0,
              "artifacts_skipped_low_confidence": 0, "artifacts_already_present": 0,
              "by_section": {}}
    for entry in _case_list(spec):
        if not isinstance(entry, dict):
            continue
        prop = by_case.get(entry.get("case_id"))
        if not prop:
            continue
        report["cases_seen"] += 1
        existing_list = entry.get("artifact_paths_expected")
        existing_list = existing_list if isinstance(existing_list, list) else []
        existing_set = set(existing_list)
        ordered = sorted((p for p in (prop.get("proposals") or [])
                         if isinstance(p, dict)),
                        key=lambda p: (-(_num(p.get("confidence")) or 0.0),
                                       str(p.get("path"))))
        to_add = []
        for p in ordered:
            path = p.get("path")
            conf = _num(p.get("confidence"))
            if not isinstance(path, str) or not path:
                continue
            if conf is None or conf < min_confidence:
                report["artifacts_skipped_low_confidence"] += 1
                continue
            if path in existing_set:
                report["artifacts_already_present"] += 1
                continue
            to_add.append(path)
            existing_set.add(path)
            sec_key = p.get("section") or "unmapped"
            report["by_section"][sec_key] = report["by_section"].get(sec_key, 0) + 1
        if to_add:
            entry["artifact_paths_expected"] = existing_list + to_add
            report["cases_changed"] += 1
            report["artifacts_added"] += len(to_add)
    return {"spec": spec, "report": report}


def write_inventory(path, obj):
    """Write the inventory back with its own key order intact.

    `corpus.write_json` sorts keys, which is right for a corpus artifact this
    tool owns outright but would silently reorder every field of a hand-built
    inventory it does not own. A diff of an inventory this tool touched must
    show only the appended list entries.
    """
    p = pathlib.Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, indent=1, ensure_ascii=True) + "\n"
    with open(str(p), "w", encoding="utf-8") as f:
        f.write(body)
    return len(body.encode("utf-8"))


# --- report --------------------------------------------------------------
def impact_stats(proposal):
    """(by_section, by_signal, cases_with_new_high) for `report`."""
    by_section = {}
    by_signal = {}
    cases_with_new_high = 0
    for c in proposal.get("cases") or []:
        props = c.get("proposals") or []
        bands = [_band(_num(p.get("confidence")) or 0.0) for p in props]
        for p, band in zip(props, bands):
            sec = p.get("section") or "unmapped"
            row = by_section.setdefault(sec, {"high": 0, "medium": 0, "low": 0})
            row[band] += 1
        case_has_high = "high" in bands
        if case_has_high:
            cases_with_new_high += 1
        for sig in c.get("signals_expected") or []:
            if sig == "none":
                continue
            srow = by_signal.setdefault(sig, {"cases": 0, "cases_gaining_evidence": 0,
                                              "sections": {}})
            srow["cases"] += 1
            if case_has_high:
                srow["cases_gaining_evidence"] += 1
                for p, band in zip(props, bands):
                    if band != "high":
                        continue
                    sec = p.get("section") or "unmapped"
                    srow["sections"][sec] = srow["sections"].get(sec, 0) + 1
    return by_section, by_signal, cases_with_new_high


# --- CLI ---------------------------------------------------------------------
def _default_proposal_path(spec_path):
    p = pathlib.Path(str(spec_path))
    return str(p.parent / (p.stem + "_recover_proposal.json"))


def cmd_scan(args):
    spec = corpus.read_json(args.spec)
    if spec is None:
        print("inventory_recover: inventory not readable as JSON: %s" % args.spec)
        return 1
    proposal = build_proposal(spec, args.root)
    out_path = args.out or _default_proposal_path(args.spec)
    corpus.write_json(out_path, proposal)
    rep = proposal["report"]
    print("scan  spec=%s  root=%s  out=%s" % (args.spec, args.root, out_path))
    print("  %d case(s): %d with job ids, %d without, %d with a job id but no "
          "match found" % (rep["cases"], rep["cases_with_job_ids"],
                           rep["cases_without_job_ids"], rep["cases_with_no_match"]))
    print("  %d proposal(s): %d high, %d medium, %d low confidence"
          % (rep["proposals"], rep["proposals_by_confidence_band"]["high"],
             rep["proposals_by_confidence_band"]["medium"],
             rep["proposals_by_confidence_band"]["low"]))
    print("  by section:")
    for k in sorted(rep["proposals_by_section"]):
        print("    %-14s %d" % (k, rep["proposals_by_section"][k]))
    if rep["cases_capped"]:
        print("  %d case(s) hit the per-case proposal cap of %d"
              % (rep["cases_capped"], MAX_PROPOSALS_PER_CASE))
    if rep["skipped_dirs"]:
        print("  %d director(y/ies) skipped during the walk (bulk size or "
              "name rule)" % rep["skipped_dirs"])
    return 0


def cmd_apply(args):
    proposal = corpus.read_json(args.proposal)
    if proposal is None:
        print("inventory_recover: proposal not readable as JSON: %s" % args.proposal)
        return 1
    spec = corpus.read_json(args.spec)
    if spec is None:
        print("inventory_recover: inventory not readable as JSON: %s" % args.spec)
        return 1
    min_conf = (args.min_confidence if args.min_confidence is not None
               else DEFAULT_MIN_CONFIDENCE_APPLY)
    result = apply_proposal(spec, proposal, min_confidence=min_conf)
    out_path = args.out or args.spec
    write_inventory(out_path, result["spec"])
    rep = result["report"]
    print("apply  proposal=%s  spec=%s  out=%s  min_confidence=%.2f"
          % (args.proposal, args.spec, out_path, min_conf))
    print("  %d case(s) touched, %d changed" % (rep["cases_seen"], rep["cases_changed"]))
    print("  %d artifact(s) added, %d already present, %d below the confidence "
          "threshold" % (rep["artifacts_added"], rep["artifacts_already_present"],
                         rep["artifacts_skipped_low_confidence"]))
    for k in sorted(rep["by_section"]):
        print("    %-14s +%d" % (k, rep["by_section"][k]))
    return 0


def cmd_report(args):
    proposal = corpus.read_json(args.proposal)
    if proposal is None:
        print("inventory_recover: proposal not readable as JSON: %s" % args.proposal)
        return 1
    by_section, by_signal, cases_with_new_high = impact_stats(proposal)
    print("report  proposal=%s" % args.proposal)
    print("  %d case(s) would gain at least one high-confidence artifact"
          % cases_with_new_high)
    print("  by section (high/medium/low):")
    for sec in sorted(by_section):
        row = by_section[sec]
        print("    %-14s %d/%d/%d" % (sec, row["high"], row["medium"], row["low"]))
    print("  by expected signal (cases carrying it / gaining new evidence):")
    for sig in sorted(by_signal):
        row = by_signal[sig]
        secs = ", ".join("%s:%d" % (k, v) for k, v in sorted(row["sections"].items()))
        print("    %-20s %d/%d  %s" % (sig, row["cases_gaining_evidence"],
                                       row["cases"], secs))
    return 0


def build_arg_parser():
    ap = argparse.ArgumentParser(
        prog="inventory_recover",
        description="Propose and apply un-listed evidence artifacts for the "
                    "WP2 incident inventory")
    sub = ap.add_subparsers(dest="cmd")

    p_scan = sub.add_parser("scan", help="search storage and propose additions")
    p_scan.add_argument("--spec", required=True, help="inventory JSON to read")
    p_scan.add_argument("--root", required=True, help="storage tree to search")
    p_scan.add_argument("--out", default=None, help="proposal JSON to write")

    p_apply = sub.add_parser("apply", help="write accepted proposals into the inventory")
    p_apply.add_argument("--proposal", required=True, help="proposal JSON from scan")
    p_apply.add_argument("--spec", required=True, help="inventory JSON to update")
    p_apply.add_argument("--min-confidence", type=float, default=None,
                         help="default: %.2f" % DEFAULT_MIN_CONFIDENCE_APPLY)
    p_apply.add_argument("--out", default=None,
                         help="where to write the updated inventory "
                              "(default: overwrite --spec)")

    p_report = sub.add_parser("report", help="summarise a proposal's impact")
    p_report.add_argument("--proposal", required=True, help="proposal JSON from scan")
    return ap


def main(argv=None):
    ap = build_arg_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:                 # argparse's own exit, including --help
        return int(exc.code or 0)
    if not args.cmd:
        ap.print_help()
        return 1
    try:
        if args.cmd == "scan":
            return cmd_scan(args)
        if args.cmd == "apply":
            return cmd_apply(args)
        if args.cmd == "report":
            return cmd_report(args)
    except Exception as exc:
        # A top-level entry point reports what it could not do; it does not
        # die with a traceback in the middle of an inventory update.
        print("inventory_recover: %s: %s" % (type(exc).__name__, exc))
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
