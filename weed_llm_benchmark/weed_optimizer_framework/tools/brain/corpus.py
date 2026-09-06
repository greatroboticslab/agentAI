#!/usr/bin/env python3
"""Incident-corpus export for the supervision benchmark (v3.25.4).

Why this exists
---------------
The supervision layer's defensible claim is not its architecture but a
*replayable* benchmark of real incidents from this campaign: the 2026-08-29
double walltime TIMEOUT (44727703 / 44767709), the COMPLETED harvest booked as
failed, the gate that kept every image it was asked to filter, the pool report
that counted 15,789 of 120,515. Those incidents are only evidence while their
raw artifacts exist, and job output on /ocean is overwritten by the next run of
the same recipe. This tool freezes them: it copies the artifacts a case rests
on, records the sha256 of each original, and writes the pre-registered truth
label beside them, so a reviewer can re-run any arm months later against bytes
rather than against a paragraph someone wrote afterwards.

Three properties are the whole point, and each one exists because the opposite
would let a hostile reviewer throw the corpus out:

* **Citations resolve to real lines of real files.** Logs are stored with
  absolute line numbers preserved (`%06d\\t<line>`, the same numbering `awk`
  gives with NR), and the recorded sha256 is that of the ORIGINAL file, never of
  the numbered copy. A finding that quotes line 8,123 of a job log can be
  checked against the job log.
* **Nothing is fabricated.** An incident whose artifacts did not survive is
  still exported, with `provenance: "record-only"`, an empty `artifacts/`
  directory and a stated reason. It is never reconstructed from the CHANGELOG
  prose that described it.
* **Nothing is silently lost.** An artifact over `MAX_ARTIFACT_BYTES` is stored
  as a head+tail excerpt whose omitted line ranges are stated in the file
  itself, together with the full sha256 and byte count of the original; a
  bundle section with no artifact behind it is recorded as `null` with a reason
  rather than dropped; a scrub target that survives scrubbing refuses the case
  instead of shipping it.

Determinism: exporting the same inventory twice produces byte-identical output.
There is no wall-clock timestamp anywhere inside a case — `built_ts` is the
case's own event date, and the only other times recorded are the mtimes of the
original artifacts, which are evidence (the `stale_artifact` signal reads them).
A corpus that changed on every export could not be hashed, and an unhashed
corpus cannot be pre-registered.

Scrubbing
---------
Every byte written passes the scrubber: the cluster username, the /ocean and
/jet/home path prefixes (to `<REPO>` and `<HOME>`), token shapes (`hf_`,
`KGAT`, `ghp_`, `sk-`, AWS, Slack, bearer) and password-shaped assignments.
Substitution is deliberately conservative (the username is replaced only on word
boundaries, so unrelated text is not mangled) while detection is deliberately
aggressive (a plain substring search). When they disagree — `byler2` in a
filename, say — the case is refused rather than written, because the alternative
is a committed corpus that fails the release scrub grep. The residue report
names the rule and the line, never the matched text: a refusal message must not
be the thing that leaks the secret.

The cluster password is not a known constant, so it cannot be a compiled rule.
Pass it through the environment for the export run:

    CORPUS_SCRUB_LITERALS='<password>,<any other exact string>' \\
    CORPUS_SCRUB_USERS='byler' python3 -m ...tools.brain.corpus export --spec ...

Commands
--------
    export --spec <inventory.json> [--only <case_id>...] [--dry-run]
    verify [--case <id>] [--root <dir>]
    list
    freeze [--cutoff YYYY-MM-DD] [--dev-extra <case_id>...]

Inventory (`--spec`) shape; paths are relative to `--root` (default: the repo
root) unless absolute, and `${REPO}` is expanded:

    {"root": "<optional default root>",
     "domain": "weed",
     "cases": [
       {"case_id": "2026-08-29_train_timeout_44727703",
        "date": "2026-08-29", "incident": true, "class": "operational",
        "domain": "weed", "round": 4, "step": "train", "job_id": "44727703",
        "signals_expected": ["walltime_bound", "pool_growth"],
        "load_bearing_lines": [{"artifact": "job_44727703.out", "line": 8123}],
        "acceptable_corrections": [{"action": "set_round_param",
                                    "params_range": {"epochs": [20, 30]},
                                    "risk": "R1"}],
        "escalation_expected": "tier1",
        "labels": {"pre_registered": {"detect": true, "correction_class": "config"}},
        "notes": "60 epochs on a pool that had grown to 8,583 it/epoch",
        "artifacts": [
          {"name": "job_44727703.out", "path": "results/framework/rndtrain_44727703.out",
           "section": "out_tail"},
          {"name": "sacct_44727703.tsv", "path": "results/framework/sacct_44727703.tsv",
           "section": "sacct"},
          {"name": "results_44727703.csv", "section": "results_csv",
           "path": "results/framework/mega_iter.../results.csv"}],
        "sections": {"su": {"round": 24.0}}}]}

`artifacts[].section` maps an artifact onto a bundle section; `sections` carries
values the inventory states inline (for sections whose artifact is a lab-side
object that never existed as a file on the cluster). An artifact needs no
`section` — it is still stored, hashed and citable.

Case layout written under `results/framework/supervision_bench/`:

    cases/<case_id>/bundle.json     evidence bundle (WP3 shape + an `export` block)
    cases/<case_id>/artifacts/      scrubbed, line-numbered copies
    cases/<case_id>/artifacts/raw/  optional unscrubbed originals (--keep-raw; gitignored)
    cases/<case_id>/truth.json      the pre-registered label
    split.json                      written by `freeze`, never overwritten

Nothing in this module raises out of a command: a broken export reports what it
could not read and exits non-zero. It is pure stdlib so it imports on the
always-on server, inside a SLURM job body and on a laptop.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

TOOL_VERSION = "wp2-corpus/1"

# The cluster checkout every campaign artifact names in its own log lines. It is
# always a scrub target, even when the export runs somewhere else against a
# synced copy — otherwise a laptop export leaves the account name in the paths.
# VERIFY: `ls -d /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark`
# on the cluster — a path that does not exist here scrubs nothing.
CLUSTER_REPO = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"

# Repo root, resolved exactly as db.py does it: the cluster path when it exists,
# otherwise the checkout this file lives in. The export runs on the cluster,
# where the artifacts are.
REPO = pathlib.Path(os.environ.get("REPO_ROOT", CLUSTER_REPO))
if not REPO.exists():
    REPO = pathlib.Path(__file__).resolve().parents[3]

DEFAULT_OUT_DIR = pathlib.Path(os.environ.get(
    "SUPERVISION_BENCH_DIR",
    str(REPO / "results" / "framework" / "supervision_bench"),
))

# --- case format constants -------------------------------------------------
# The 14 bundle sections, in the order WP3's live builder produces them. Every
# one of these keys is present in every bundle; a section with nothing behind it
# is null and carries a reason in export.missing.
SECTIONS = ("ledger", "sacct", "out_tail", "results_csv", "strategy", "trace",
            "slug_scores", "registry_diff", "harvest", "resources", "su",
            "corrections", "plan", "signals")
CASE_CLASSES = ("operational", "config", "code", "design")
ESCALATIONS = ("none", "tier1", "tier2", "human")
# Vocabulary for truth.provenance. Computed by the exporter from what it could
# actually read, never taken from the inventory: a case cannot declare itself raw.
PROVENANCE = ("raw", "record-only")

# Size discipline. Above MAX_ARTIFACT_BYTES an artifact is stored as an excerpt
# (never a silent truncation); above MAX_READ_BYTES only its head and tail are
# read at all, because a job .out on a shared filesystem can be arbitrarily
# large and this tool also runs inside the scheduler's tick.
MAX_ARTIFACT_BYTES = 1 << 20
MAX_READ_BYTES = 64 << 20
EXCERPT_HEAD_LINES = 2000
EXCERPT_TAIL_LINES = 2000
EXCERPT_HEAD_BYTES = 4 << 20
EXCERPT_TAIL_BYTES = 4 << 20
# A load-bearing line must survive excerpting, or the citation it anchors cannot
# be checked; it is kept with this much context on each side.
CITE_CONTEXT_LINES = 10

# Per-section caps applied at export, recorded in bundle.caps so a replay can
# see what the model was and was not shown.
CAPS = {
    "out_tail_lines": 400,      # total lines kept in the out_tail section
    "out_tail_tail_lines": 120,  # the last N lines are always kept
    "sacct_rows": 200,
    "trace_records": 50,
}

# Lines the out_tail trimmer keeps wherever they appear. Deliberately blunt: the
# cost of keeping an uninteresting line is tokens, the cost of dropping the one
# that says TIMEOUT is the whole case.
_OUT_TAIL_KEEP_RE = re.compile(
    r"(WARN|ERROR|FAIL|TIMEOUT|CANCELLED|Traceback|slurmstepd|"
    r"out of memory|CUDA|Killed|exceeded)", re.I)

_LINE_PREFIX_RE = re.compile(r"^\d{6}\t")
# A citation has to resolve to a line; bytes with no lines cannot carry one.
_BINARY = "binary file (NUL byte); artifacts must be text to be citable"


# --- scrubbing -------------------------------------------------------------
# Token shapes. `hf_` and `KGAT` are the two that have actually been committed
# to this repository before; the rest are the shapes any grep-based release
# check looks for.
_TOKEN_PATTERNS = (
    ("token_hf", re.compile(r"hf_[A-Za-z0-9]{10,}")),
    ("token_kaggle", re.compile(r"KGAT[A-Za-z0-9_\-]{8,}")),
    ("token_github", re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}|"
                                r"github_pat_[A-Za-z0-9_]{20,}")),
    ("token_sk", re.compile(r"sk-[A-Za-z0-9_\-]{16,}")),
    ("token_aws", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("token_slack", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("token_bearer", re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]{20,}")),
)

# /ocean/projects/<allocation>/<username>/... and /jet/home/<username>/... — the
# two absolute prefixes on Bridges-2 that carry an account name.
_OCEAN_HOME_RE = re.compile(r"/ocean/projects/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
_JET_HOME_RE = re.compile(r"/jet/home/[A-Za-z0-9_.\-]+")

# A secret-shaped assignment: `password=...`, `"api_key": "..."`, `TOKEN = ...`.
# The value is required to be 4+ characters and not to start with `<`, so the
# rule is idempotent and `<REDACTED>` is never redacted a second time.
_ASSIGN_RE = re.compile(
    r"(?i)\b(pass(?:word|wd)?|pwd|secret|api[_-]?key|access[_-]?key|token|auth)"
    r"(\"?\s*[:=]\s*\"?)(?!<)([^\s\"',;)]{4,})")
_ASSIGN_KEEP = ("null", "none", "true", "false", "undefined")

REDACTED = "<REDACTED>"
REDACTED_TOKEN = "<REDACTED_TOKEN>"


def _env_list(name):
    """Comma-separated env var as a tuple of non-empty stripped strings."""
    raw = os.environ.get(name, "") or ""
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def scrub_config(spec=None, root=None):
    """Build the scrub configuration for one export run.

    Users and literals come from the inventory and from the environment
    (`CORPUS_SCRUB_USERS`, `CORPUS_SCRUB_LITERALS`) so that the cluster password
    can be scrubbed without ever being written into a committed spec file.
    """
    spec = spec if isinstance(spec, dict) else {}
    users = list(spec.get("scrub_users") or ()) + list(_env_list("CORPUS_SCRUB_USERS"))
    if not users:
        # The account every campaign artifact was produced under.
        # VERIFY: `whoami` on the cluster login node matches this string; if it
        # does not, every artifact ships with an unscrubbed username.
        users = ["byler"]
    literals = (list(spec.get("scrub_literals") or ())
                + list(_env_list("CORPUS_SCRUB_LITERALS")))
    repos = [str(root or REPO), CLUSTER_REPO]
    for extra in (spec.get("scrub_repo_paths") or ()):
        repos.append(str(extra))
    # Longest first: the repo path is itself an /ocean home prefix, so it has to
    # win before the home rule turns its first four segments into <HOME>.
    repos = sorted({r.rstrip("/") for r in repos if r}, key=len, reverse=True)
    return {
        "users": sorted({u for u in users if u}, key=len, reverse=True),
        "literals": sorted({s for s in literals if s}, key=len, reverse=True),
        "repos": repos,
    }


def _redact_assign(m):
    if m.group(3).strip().lower() in _ASSIGN_KEEP:
        return m.group(0)
    return m.group(1) + m.group(2) + REDACTED


def scrub_text(text, cfg):
    """Scrub `text`; return (scrubbed_text, {rule_name: substitutions}).

    Order is fixed and matters: exact literals, then the repo prefix, then the
    remaining home prefixes, then tokens and assignments, then bare usernames.
    """
    counts = {}

    def _sub(name, pattern, repl, s):
        if isinstance(pattern, str):
            n = s.count(pattern)
            if n:
                counts[name] = counts.get(name, 0) + n
                s = s.replace(pattern, repl)
            return s
        s, n = pattern.subn(repl, s)
        if n:
            counts[name] = counts.get(name, 0) + n
        return s

    if not isinstance(text, str):
        return "", counts
    out = text
    for lit in cfg.get("literals", ()):
        out = _sub("literal", lit, REDACTED, out)
    for repo in cfg.get("repos", ()):
        out = _sub("repo_path", repo, "<REPO>", out)
    out = _sub("ocean_home", _OCEAN_HOME_RE, "<HOME>", out)
    out = _sub("jet_home", _JET_HOME_RE, "<HOME>", out)
    for name, pat in _TOKEN_PATTERNS:
        out = _sub(name, pat, REDACTED_TOKEN, out)
    out = _sub("secret_assignment", _ASSIGN_RE, _redact_assign, out)
    for user in cfg.get("users", ()):
        # Word-bounded on purpose: replacing every substring occurrence would
        # rewrite unrelated words. What word boundaries miss is caught by the
        # residue check, which refuses the case rather than guessing.
        out = _sub("username", re.compile(r"\b%s\b" % re.escape(user)), "<USER>", out)
    return out, counts


def scrub_residue(text, cfg):
    """Scrub targets still present after scrubbing, as [{rule, line, count}].

    Detection is a substring search for usernames and literals (broader than the
    substitution rules) plus the token and path patterns. The matched text is
    never reported — a refusal message that quotes the secret defeats itself.
    """
    hits = []
    if not isinstance(text, str) or not text:
        return hits
    users = [u.lower() for u in cfg.get("users", ())]
    literals = list(cfg.get("literals", ()))
    for i, line in enumerate(text.split("\n"), start=1):
        low = line.lower()
        for u in users:
            n = low.count(u)
            if n:
                hits.append({"rule": "username", "line": i, "count": n})
        for lit in literals:
            n = line.count(lit)
            if n:
                hits.append({"rule": "literal", "line": i, "count": n})
        for name, pat in _TOKEN_PATTERNS:
            n = len(pat.findall(line))
            if n:
                hits.append({"rule": name, "line": i, "count": n})
        for name, pat in (("ocean_home", _OCEAN_HOME_RE), ("jet_home", _JET_HOME_RE)):
            n = len(pat.findall(line))
            if n:
                hits.append({"rule": name, "line": i, "count": n})
    return hits


# --- deterministic serialisation ------------------------------------------
def canonical_json(obj):
    """Compact, key-sorted, ASCII JSON — the form everything is hashed over.

    Float determinism comes from Python's shortest-round-trip repr, so numbers
    parsed out of an artifact are hashed exactly as they were written. Numbers
    this module *computes* (quantiles, per-epoch seconds) are rounded at the
    point of computation instead, so a recomputed section hashes the same.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path, obj):
    """Write JSON deterministically (sorted keys, indent 1, trailing newline)."""
    p = pathlib.Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True,
                      default=str) + "\n"
    with open(str(p), "w", encoding="utf-8") as f:
        f.write(body)
    return len(body.encode("utf-8"))


def read_json(path):
    """Parsed JSON, or None. Never raises."""
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# --- artifact reading ------------------------------------------------------
def _hash_and_count(path):
    """Stream `path` once: (sha256, bytes, lines, mtime, error).

    Line count follows awk's NR: lines are separated by "\\n", and a final
    fragment with no terminator still counts as a line.
    """
    h = hashlib.sha256()
    nbytes = 0
    nlines = 0
    last = b""
    try:
        mtime = os.path.getmtime(str(path))
        with open(str(path), "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                nbytes += len(chunk)
                nlines += chunk.count(b"\n")
                last = chunk[-1:]
    except OSError as exc:
        return None, 0, 0, None, "%s: %s" % (type(exc).__name__, exc)
    if nbytes and last != b"\n":
        nlines += 1
    return h.hexdigest(), nbytes, nlines, mtime, None


def _decode(blob):
    return blob.decode("utf-8", "replace")


def _split_lines(text):
    """awk NR semantics: split on "\\n" only, drop the empty final element."""
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def load_artifact(path):
    """Read one artifact for export.

    Returns {sha256, bytes, lines, mtime, read, chunks, error} where `chunks` is
    a list of (first_absolute_line_number, [line_text, ...]). A file at or under
    MAX_READ_BYTES yields one chunk covering the whole file; a larger one yields
    a head chunk and a tail chunk, and `read` is "partial" so every consumer
    knows the middle was never examined (including the scrubber, whose residue
    check can only speak for what it saw).
    """
    sha, nbytes, nlines, mtime, err = _hash_and_count(path)
    if err:
        return {"sha256": None, "bytes": 0, "lines": 0, "mtime": None,
                "read": "none", "chunks": [], "error": err}
    out = {"sha256": sha, "bytes": nbytes, "lines": nlines, "mtime": mtime,
           "read": "full", "chunks": [], "error": None}
    try:
        if nbytes <= MAX_READ_BYTES:
            with open(str(path), "rb") as f:
                blob = f.read()
            if b"\x00" in blob:
                out["read"] = "none"
                out["error"] = _BINARY
                return out
            out["chunks"] = [(1, _split_lines(_decode(blob)))]
            return out
        with open(str(path), "rb") as f:
            head = f.read(EXCERPT_HEAD_BYTES)
            f.seek(max(0, nbytes - EXCERPT_TAIL_BYTES))
            tail = f.read()
        if b"\x00" in head or b"\x00" in tail:
            out["read"] = "none"
            out["error"] = _BINARY
            return out
        head_lines = _split_lines(_decode(head))
        if head_lines:
            head_lines.pop()          # the window ends mid-line; that is not a record
        tail_lines = _decode(tail).split("\n")
        if tail_lines:
            tail_lines.pop(0)         # ditto at the front of the tail window
        if tail_lines and tail_lines[-1] == "":
            tail_lines.pop()
        out["read"] = "partial"
        out["chunks"] = [(1, head_lines[:EXCERPT_HEAD_LINES])]
        if tail_lines:
            tail_keep = tail_lines[-EXCERPT_TAIL_LINES:]
            out["chunks"].append((nlines - len(tail_keep) + 1, tail_keep))
        return out
    except OSError as exc:
        out["read"] = "none"
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out


def _available_ranges(chunks):
    return [(start, start + len(texts) - 1) for start, texts in chunks if texts]


def _text_at(chunks, lineno):
    for start, texts in chunks:
        if start <= lineno < start + len(texts):
            return texts[lineno - start]
    return None


def _merge(ranges):
    out = []
    for a, b in sorted(r for r in ranges if r[0] <= r[1]):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _keep_ranges(info, cite_lines):
    """Absolute line ranges to store for this artifact.

    Whole file when it fits; otherwise head + tail + a window around every
    load-bearing line, clipped to the ranges actually read. Citations come first
    because a case whose quoted line was excerpted away cannot be scored.
    """
    avail = _available_ranges(info["chunks"])
    if not avail:
        return []
    n = info["lines"]
    if info["read"] == "full" and info["bytes"] <= MAX_ARTIFACT_BYTES:
        return _merge(avail)
    want = [(1, min(EXCERPT_HEAD_LINES, n)), (max(1, n - EXCERPT_TAIL_LINES + 1), n)]
    for ln in cite_lines:
        want.append((max(1, ln - CITE_CONTEXT_LINES), min(n, ln + CITE_CONTEXT_LINES)))
    keep = []
    for a, b in _merge(want):
        for x, y in avail:
            lo, hi = max(a, x), min(b, y)
            if lo <= hi:
                keep.append((lo, hi))
    return _merge(keep)


def render_numbered(info, keep, header=None):
    """The stored artifact text: "<6-digit line>\\t<line>" per kept line.

    Every line that is not content is an export marker starting with "[", so a
    reader can split on the first tab and `int()` the prefix. Omitted ranges are
    stated where they were omitted; nothing is dropped without a line saying so.
    """
    parts = []
    if header:
        parts.append(header)
    n = info["lines"]
    prev_end = 0
    for a, b in keep:
        if a > prev_end + 1:
            parts.append("[omitted lines %d-%d (%d lines)]" % (prev_end + 1, a - 1,
                                                               a - 1 - prev_end))
        for ln in range(a, b + 1):
            text = _text_at(info["chunks"], ln)
            if text is None:
                continue
            parts.append("%06d\t%s" % (ln, text))
        prev_end = b
    if n > prev_end:
        parts.append("[omitted lines %d-%d (%d lines)]" % (prev_end + 1, n, n - prev_end))
    return "\n".join(parts) + ("\n" if parts else "")


# --- section builders ------------------------------------------------------
def _pairs(info, keep):
    """[(absolute_line_number, text), ...] over the kept lines."""
    out = []
    for a, b in keep:
        for ln in range(a, b + 1):
            text = _text_at(info["chunks"], ln)
            if text is not None:
                out.append([ln, text])
    return out


def _trim_out_tail(pairs):
    """WARN/ERROR/TIMEOUT lines plus the tail, capped — WP3's trimmer."""
    if not pairs:
        return []
    tail_from = pairs[max(0, len(pairs) - CAPS["out_tail_tail_lines"])][0]
    sel = [p for p in pairs if p[0] >= tail_from or _OUT_TAIL_KEEP_RE.search(p[1])]
    if len(sel) > CAPS["out_tail_lines"]:
        # Drop the earliest matches, never the tail: a job's failure is at its end.
        sel = sel[-CAPS["out_tail_lines"]:]
    return sel


def _delimiter(line):
    for d in ("|", "\t", ","):
        if d in line:
            return d
    return None


def _table_rows(pairs, cap):
    """Parse a delimited table into row dicts; fall back to raw line records."""
    texts = [t for _, t in pairs if t.strip()]
    if not texts:
        return []
    d = _delimiter(texts[0])
    if not d:
        return [{"line": t} for t in texts[:cap]]
    header = [h.strip() for h in texts[0].split(d)]
    rows = []
    for t in texts[1:]:
        cells = [c.strip() for c in t.split(d)]
        if len(cells) < 2:
            continue
        rows.append({header[i] if i < len(header) else "col%d" % i: cells[i]
                     for i in range(len(cells))})
        if len(rows) >= cap:
            break
    return rows or [{"line": t} for t in texts[:cap]]


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _results_csv_summary(pairs, mtime):
    """{rows, best, last, epoch_time_s, mtime} from an Ultralytics results.csv.

    `best`/`last` read the mAP50-95 column and `epoch_time_s` the per-epoch
    delta of the cumulative `time` column — the two numbers the walltime
    projection rests on. Unknown layouts return counts only rather than guesses.
    """
    texts = [t for _, t in pairs if t.strip()]
    out = {"rows": 0, "best": None, "last": None, "epoch_time_s": None, "mtime": mtime}
    if not texts:
        return out
    header = [h.strip() for h in texts[0].split(",")]
    body = [t.split(",") for t in texts[1:] if t.strip()]
    out["rows"] = len(body)
    if not body:
        return out
    def col(pred):
        for i, h in enumerate(header):
            if pred(h.lower()):
                return i
        return None
    imap = col(lambda h: "map50-95" in h or "map50_95" in h)
    itime = col(lambda h: h == "time" or h.endswith("/time"))
    if imap is not None:
        vals = [_float(r[imap]) for r in body if len(r) > imap]
        vals = [v for v in vals if v is not None]
        if vals:
            out["best"] = round(max(vals), 6)
            out["last"] = round(vals[-1], 6)
    if itime is not None and len(body) >= 2:
        times = [_float(r[itime]) for r in body if len(r) > itime]
        times = [v for v in times if v is not None]
        if len(times) >= 2:
            out["epoch_time_s"] = round((times[-1] - times[0]) / (len(times) - 1), 3)
    return out


def _jsonl_records(pairs, cap):
    recs = []
    for _, t in pairs:
        t = t.strip()
        if not t:
            continue
        try:
            r = json.loads(t)
        except Exception:
            continue
        if isinstance(r, dict):
            recs.append(r)
    return recs[-cap:]


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = int(round(q * (len(sorted_vals) - 1)))
    return round(sorted_vals[max(0, min(idx, len(sorted_vals) - 1))], 6)


def _slug_scores_summary(obj, mtime):
    """{n, unscored, p25, median, p75, mtime} when the snapshot is slug->score."""
    if not isinstance(obj, dict):
        return obj
    body = obj.get("scores") if isinstance(obj.get("scores"), dict) else obj
    vals, unscored = [], 0
    for v in body.values():
        f = _float(v if not isinstance(v, dict) else v.get("score"))
        if f is None:
            unscored += 1
        else:
            vals.append(f)
    if not vals and not unscored:
        return obj
    vals.sort()
    return {"n": len(vals) + unscored, "unscored": unscored,
            "p25": _quantile(vals, 0.25), "median": _quantile(vals, 0.5),
            "p75": _quantile(vals, 0.75), "mtime": mtime}


def _parse_json_artifact(pairs):
    """Parse a stored JSON artifact from its (scrubbed) lines."""
    try:
        return json.loads("\n".join(t for _, t in pairs)), None
    except Exception as exc:
        return None, ("artifact is not parseable JSON after scrubbing: %s"
                      % type(exc).__name__)


def token_estimate(sections):
    """Deterministic token estimate for the sections object that becomes a
    prompt: ceil(len(canonical JSON of `sections`) / 4).

    Four characters per token is the English-plus-JSON rule of thumb; WP3
    asserts a real tokenizer count against it within 15 % before any call, and
    the number is only ever used to keep a bundle under `num_ctx`.
    """
    return (len(canonical_json(sections)) + 3) // 4


# --- case export -----------------------------------------------------------
def _clean_str(x, default=""):
    return x if isinstance(x, str) else default


def _validate_case(case):
    """Structural problems that make a case unexportable. Returns [reason]."""
    bad = []
    cid = _clean_str(case.get("case_id"))
    if not cid or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,120}$", cid):
        bad.append("case_id missing or not a safe path segment")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", _clean_str(case.get("date"))):
        bad.append("date must be YYYY-MM-DD (it is the case's only timestamp)")
    if not isinstance(case.get("incident"), bool):
        bad.append("incident must be a boolean")
    if case.get("class") not in CASE_CLASSES:
        bad.append("class must be one of %s" % (CASE_CLASSES,))
    esc = case.get("escalation_expected", "none")
    if esc not in ESCALATIONS:
        bad.append("escalation_expected must be one of %s" % (ESCALATIONS,))
    arts = case.get("artifacts") or []
    if not isinstance(arts, list):
        bad.append("artifacts must be a list")
    else:
        names = set()
        for a in arts:
            if not isinstance(a, dict):
                bad.append("artifact entries must be objects")
                continue
            nm = _clean_str(a.get("name"))
            if not nm or "/" in nm or nm.startswith("."):
                bad.append("artifact name %r is not a safe file name" % nm)
            if nm in names:
                bad.append("duplicate artifact name %r" % nm)
            names.add(nm)
            sec = a.get("section")
            if sec is not None and sec not in SECTIONS:
                bad.append("artifact %r maps to unknown section %r" % (nm, sec))
    inline = case.get("sections") or {}
    if not isinstance(inline, dict):
        bad.append("sections must be an object")
    else:
        for k in inline:
            if k not in SECTIONS:
                bad.append("inventory section %r is not one of the 14 "
                           "bundle sections" % k)
    return bad


def _resolve_path(raw, root):
    p = _clean_str(raw).replace("${REPO}", str(root))
    if not p:
        return None
    q = pathlib.Path(p)
    return q if q.is_absolute() else pathlib.Path(str(root)) / q


def _stored_name(name):
    """artifacts/<name>.txt — the original name is kept visible in the stored one."""
    return name if name.endswith(".txt") else name + ".txt"


def _cite_lines_for(case, art_name):
    out = []
    for c in (case.get("load_bearing_lines") or []):
        if isinstance(c, dict) and c.get("artifact") == art_name:
            try:
                out.append(int(c.get("line")))
            except (TypeError, ValueError):
                continue
    return sorted(set(out))


def _build_case(case, root, cfg):
    """Read, scrub and assemble one case in memory. Writes nothing.

    Returns (payload, report). `payload` is None when the case is refused; the
    report always says why.
    """
    cid = _clean_str(case.get("case_id"), "<unnamed>")
    report = {"case_id": cid, "status": "refused", "reason": None, "provenance": None,
              "artifacts": 0, "artifacts_missing": [], "bytes_stored": 0,
              "bytes_original": 0, "scrub": {}, "scrub_total": 0,
              "residue": [], "unresolved_load_bearing": [], "warnings": []}

    bad = _validate_case(case)
    if bad:
        report["reason"] = "invalid inventory entry: " + "; ".join(bad)
        return None, report

    entries = []            # manifest rows, in inventory order
    stored = {}             # stored relative path -> text
    raw_copies = {}         # artifact name -> source path (for --keep-raw)
    by_section = {}         # section -> (artifact_id, pairs, info)
    missing = {}
    scrub_counts = {}

    for a in (case.get("artifacts") or []):
        name = _clean_str(a.get("name"))
        src = _resolve_path(a.get("path"), root)
        sec = a.get("section")
        info = load_artifact(src) if src else {
            "sha256": None, "bytes": 0, "lines": 0, "mtime": None,
            "read": "none", "chunks": [], "error": "no path given"}
        disp = scrub_text(str(src), cfg)[0] if src else ""
        if info["error"] or not info["chunks"]:
            # An OSError's text carries the absolute path it failed on, and that
            # text is written into bundle.json — scrub it like any other byte.
            why = scrub_text(info["error"] or "file is empty", cfg)[0]
            report["artifacts_missing"].append(name)
            if sec:
                missing[sec] = "artifact %s not readable at %s (%s)" % (name, disp, why)
            entries.append({"artifact_id": name, "stored": None, "path": disp,
                            "sha256": info["sha256"], "bytes": info["bytes"],
                            "lines": info["lines"], "mtime": info["mtime"],
                            "read": info["read"], "present": False, "reason": why,
                            "section": sec, "excerpt": None, "stored_bytes": 0,
                            "stored_sha256": None})
            continue

        # Scrub every chunk before anything else looks at the text.
        chunks = []
        for start, texts in info["chunks"]:
            joined, counts = scrub_text("\n".join(texts), cfg)
            for k, v in counts.items():
                scrub_counts[k] = scrub_counts.get(k, 0) + v
            chunks.append((start, joined.split("\n")))
        info = dict(info, chunks=chunks, display_path=disp)

        cites = _cite_lines_for(case, name)
        keep = _keep_ranges(info, cites)
        header = None
        excerpt = None
        if info["read"] != "full" or info["bytes"] > MAX_ARTIFACT_BYTES:
            kept_lines = sum(b - a + 1 for a, b in keep)
            header = ("[excerpt] original sha256=%s bytes=%d lines=%d read=%s "
                      "kept=%d lines in %d block(s); omitted ranges are stated below]"
                      % (info["sha256"], info["bytes"], info["lines"], info["read"],
                         kept_lines, len(keep)))
            excerpt = {"kept_lines": kept_lines, "kept_ranges": [list(r) for r in keep],
                       "omitted_lines": info["lines"] - kept_lines,
                       "head_lines": EXCERPT_HEAD_LINES, "tail_lines": EXCERPT_TAIL_LINES,
                       "reason": ("file exceeds %d bytes" % MAX_ARTIFACT_BYTES
                                  if info["read"] == "full" else
                                  "file exceeds the %d byte read ceiling; the middle was "
                                  "never read and could not be scrubbed or scanned"
                                  % MAX_READ_BYTES)}
        text = render_numbered(info, keep, header)

        res = scrub_residue(text, cfg)
        if res:
            report["residue"].extend([dict(r, artifact=name) for r in res])

        rel = "artifacts/" + _stored_name(name)
        stored[rel] = text
        if info["read"] == "full":
            raw_copies[name] = src
        pairs = _pairs(info, keep)
        entries.append({"artifact_id": name, "stored": rel, "path": disp,
                        "sha256": info["sha256"], "bytes": info["bytes"],
                        "lines": info["lines"], "mtime": info["mtime"],
                        "read": info["read"], "present": True, "reason": None,
                        "section": sec, "excerpt": excerpt,
                        "stored_bytes": len(text.encode("utf-8")),
                        "stored_sha256": sha256_str(text)})
        if info["lines"] == 0:
            report["warnings"].append("artifact %s is empty (0 lines)" % name)
        report["bytes_original"] += info["bytes"]
        report["bytes_stored"] += len(text.encode("utf-8"))
        if sec and sec not in by_section:
            by_section[sec] = (name, pairs, info)
        elif sec:
            report["warnings"].append(
                "section %s already filled by %s; %s is stored but not sectioned"
                % (sec, by_section[sec][0], name))

        for ln in cites:
            if not any(a <= ln <= b for a, b in keep):
                report["unresolved_load_bearing"].append(
                    {"artifact": name, "line": ln,
                     "why": "line is outside the stored region of the artifact"})

    # A scrub target that survived refuses the case. Writing it would put the
    # secret in git and fail the release grep; the reason names the rule and the
    # line, never the text.
    if report["residue"]:
        rules = sorted({r["rule"] for r in report["residue"]})
        report["reason"] = ("scrub residue after scrubbing (%d hit(s), rules: %s) — "
                            "case not written"
                            % (len(report["residue"]), ", ".join(rules)))
        report["scrub"] = scrub_counts
        report["scrub_total"] = sum(scrub_counts.values())
        return None, report

    # Load-bearing citations pointing at an artifact that is not here at all.
    have = {e["artifact_id"] for e in entries if e.get("present")}
    for c in (case.get("load_bearing_lines") or []):
        if isinstance(c, dict) and c.get("artifact") not in have:
            report["unresolved_load_bearing"].append(
                {"artifact": c.get("artifact"), "line": c.get("line"),
                 "why": "artifact was not exported"})

    n_present = len(have)
    provenance = "raw" if n_present else "record-only"
    prov_reason = None
    if provenance == "record-only":
        if case.get("artifacts"):
            prov_reason = ("no declared artifact could be read: " + "; ".join(
                "%s (%s)" % (e["artifact_id"], e["reason"])
                for e in entries if not e.get("present")))
        else:
            prov_reason = ("the inventory declares no artifact for this case; it is a "
                           "record from the engineering log and is never reconstructed "
                           "from that prose")

    sections, sec_missing, sec_source = _build_sections(case, by_section, missing)
    bundle = {
        "bundle_id": _clean_str(case.get("case_id")),
        "sha256": "",
        "domain": _clean_str(case.get("domain"), "weed"),
        "round": case.get("round"),
        "step": case.get("step"),
        # The event date, not the export time: a case is a frozen artifact and
        # must hash the same on every export.
        "built_ts": _clean_str(case.get("date")),
        "sections": sections,
        "token_estimate": 0,
        "caps": dict(CAPS),
        "export": {
            "tool_version": TOOL_VERSION,
            "case_id": _clean_str(case.get("case_id")),
            "provenance": provenance,
            "provenance_reason": prov_reason,
            "job_id": case.get("job_id"),
            "artifacts": entries,
            "missing": sec_missing,
            "section_source": sec_source,
            "scrub": scrub_counts,
            "scrub_total": sum(scrub_counts.values()),
            "unresolved_load_bearing": report["unresolved_load_bearing"],
            "warnings": report["warnings"],
        },
    }
    bundle["token_estimate"] = token_estimate(bundle["sections"])
    bundle["sha256"] = sha256_str(canonical_json(bundle))

    truth = {
        "case_id": _clean_str(case.get("case_id")),
        "date": _clean_str(case.get("date")),
        "incident": bool(case.get("incident")),
        "class": case.get("class"),
        "signals_expected": list(case.get("signals_expected") or []),
        "load_bearing_lines": [
            {"artifact": c.get("artifact"), "line": c.get("line")}
            for c in (case.get("load_bearing_lines") or []) if isinstance(c, dict)],
        "acceptable_corrections": [
            {"action": c.get("action"), "params_range": c.get("params_range"),
             "risk": c.get("risk")}
            for c in (case.get("acceptable_corrections") or []) if isinstance(c, dict)],
        "escalation_expected": case.get("escalation_expected", "none"),
        "provenance": provenance,
        "labels": {
            "pre_registered": (case.get("labels") or {}).get("pre_registered") or {},
            # Filled by the second rater after the split is frozen; empty here so
            # the pre-registered label cannot be quietly edited into agreement.
            "adjudicated": (case.get("labels") or {}).get("adjudicated") or {},
        },
        "notes": _clean_str(case.get("notes")),
    }

    final = scrub_residue(canonical_json(bundle) + "\n" + canonical_json(truth), cfg)
    if final:
        rules = sorted({r["rule"] for r in final})
        report["residue"] = final
        report["reason"] = ("scrub residue in the assembled case (%d hit(s), "
                            "rules: %s) — case not written; the inventory's own "
                            "text is not rewritten, so fix it there"
                            % (len(final), ", ".join(rules)))
        report["scrub"] = scrub_counts
        report["scrub_total"] = sum(scrub_counts.values())
        return None, report

    report.update({"status": "ok", "provenance": provenance,
                   "artifacts": n_present, "scrub": scrub_counts,
                   "scrub_total": sum(scrub_counts.values())})
    payload = {"bundle": bundle, "truth": truth, "stored": stored,
               "raw_copies": raw_copies}
    return payload, report


def _build_sections(case, by_section, missing):
    """Assemble the 14 bundle sections. Missing ones are null with a reason."""
    sections = {}
    sec_missing = dict(missing)
    sec_source = {}
    inline = case.get("sections") or {}

    for name in SECTIONS:
        if name in by_section:
            art_id, pairs, info = by_section[name]
            value, why = _section_value(name, art_id, pairs, info)
            if why:
                sections[name] = None
                sec_missing[name] = why
            else:
                sections[name] = value
                sec_source[name] = art_id
            continue
        if name in inline:
            sections[name] = inline[name]
            sec_source[name] = "inventory:sections"
            sec_missing.pop(name, None)
            continue
        sections[name] = None
        sec_missing.setdefault(
            name, "no artifact mapped to this section and no inventory value given")
    return sections, sec_missing, sec_source


def _section_value(name, art_id, pairs, info):
    """Section value from one artifact's kept lines. Returns (value, reason)."""
    if name == "out_tail":
        return {"artifact_id": art_id, "path": info.get("display_path"),
                "sha256": info["sha256"], "lines": _trim_out_tail(pairs)}, None
    if name == "sacct":
        return _table_rows(pairs, CAPS["sacct_rows"]), None
    if name == "results_csv":
        return _results_csv_summary(pairs, info.get("mtime")), None
    if name == "trace":
        return _jsonl_records(pairs, CAPS["trace_records"]), None
    obj, why = _parse_json_artifact(pairs)
    if why:
        return None, why
    if name == "slug_scores":
        return _slug_scores_summary(obj, info.get("mtime")), None
    return obj, None


def _case_dir_replaceable(d):
    """True when `d` is absent or looks like a case directory this tool wrote."""
    p = pathlib.Path(str(d))
    if not p.exists():
        return True
    if not p.is_dir():
        return False
    names = {x.name for x in p.iterdir()}
    return not names or bool(names & {"bundle.json", "truth.json", "artifacts"})


def _write_case(out_dir, payload, keep_raw=False):
    """Write one case directory. Returns (bytes_written | None, [warning]).

    `None` means nothing was written: the target exists and holds something this
    tool did not put there, so it is left alone rather than deleted.
    """
    warns = []
    cid = payload["bundle"]["export"]["case_id"]
    d = pathlib.Path(str(out_dir)) / "cases" / cid
    if not _case_dir_replaceable(d):
        return None, ["%s exists and is not a case directory; not touched" % d]
    if d.exists():
        shutil.rmtree(str(d))
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    total = 0
    for rel, text in sorted(payload["stored"].items()):
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), "w", encoding="utf-8") as f:
            f.write(text)
        total += len(text.encode("utf-8"))
    if keep_raw:
        for name, src in sorted(payload["raw_copies"].items()):
            dst = d / "artifacts" / "raw" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(str(src), str(dst))
            except OSError as exc:
                warns.append("raw copy of %s failed: %s" % (name, exc))
        warns.append("raw copies written under %s/artifacts/raw — unscrubbed; "
                     "gitignored only under the cluster repo root" % d)
    total += write_json(d / "bundle.json", payload["bundle"])
    total += write_json(d / "truth.json", payload["truth"])
    return total, warns


def export(spec_path, out_dir=None, only=None, dry_run=False, root=None,
           keep_raw=False):
    """Build cases from an inventory file. Returns a report dict; never raises."""
    out_dir = pathlib.Path(str(out_dir or DEFAULT_OUT_DIR))
    rep = {"ok": False, "out_dir": str(out_dir), "dry_run": bool(dry_run),
           "spec": str(spec_path), "cases": [], "errors": [],
           "counts": {"written": 0, "refused": 0, "skipped": 0,
                      "raw": 0, "record_only": 0}}
    spec = read_json(spec_path)
    if spec is None:
        rep["errors"].append("inventory not readable as JSON: %s" % spec_path)
        return rep
    if isinstance(spec, list):
        spec = {"cases": spec}
    if not isinstance(spec, dict) or not isinstance(spec.get("cases"), list):
        rep["errors"].append("inventory must be an object with a 'cases' list")
        return rep

    root = pathlib.Path(str(root or spec.get("root") or REPO))
    cfg = scrub_config(spec, root)
    rep["root"] = str(root)
    wanted = set(only or ())
    seen = set()

    for case in spec["cases"]:
        if not isinstance(case, dict):
            rep["errors"].append("a case entry is not an object; skipped")
            continue
        cid = _clean_str(case.get("case_id"), "<unnamed>")
        seen.add(cid)
        if wanted and cid not in wanted:
            rep["counts"]["skipped"] += 1
            continue
        if not case.get("domain") and spec.get("domain"):
            case = dict(case, domain=spec["domain"])
        try:
            payload, creport = _build_case(case, root, cfg)
        except Exception as exc:          # a broken case is not a broken export
            rep["cases"].append({"case_id": cid, "status": "refused",
                                 "reason": "export raised %s: %s"
                                           % (type(exc).__name__, exc)})
            rep["counts"]["refused"] += 1
            continue
        if payload is None:
            creport["status"] = "refused"
            rep["cases"].append(creport)
            rep["counts"]["refused"] += 1
            continue
        if dry_run:
            creport["status"] = "would-write"
            rep["cases"].append(creport)
        else:
            try:
                nbytes, warns = _write_case(out_dir, payload, keep_raw=keep_raw)
            except Exception as exc:
                creport["status"] = "refused"
                creport["reason"] = "write failed: %s: %s" % (type(exc).__name__, exc)
                rep["cases"].append(creport)
                rep["counts"]["refused"] += 1
                continue
            if nbytes is None:
                creport["status"] = "refused"
                creport["reason"] = "; ".join(warns)
                rep["cases"].append(creport)
                rep["counts"]["refused"] += 1
                continue
            creport["status"] = "written"
            creport["bytes_written"] = nbytes
            creport["warnings"] = list(creport.get("warnings") or []) + list(warns)
            rep["cases"].append(creport)
            rep["counts"]["written"] += 1
        rep["counts"]["raw" if creport["provenance"] == "raw" else "record_only"] += 1

    for cid in sorted(wanted - seen):
        rep["errors"].append("--only %s: no such case in the inventory" % cid)
    rep["ok"] = not rep["errors"] and rep["counts"]["refused"] == 0
    return rep


# --- verify / list / freeze ------------------------------------------------
def _case_ids(out_dir):
    d = pathlib.Path(str(out_dir)) / "cases"
    if not d.is_dir():
        return []
    return sorted(x.name for x in d.iterdir() if x.is_dir())


def verify(out_dir=None, case=None, root=None):
    """Re-hash every stored artifact (and, with `root`, every original).

    Reports drift rather than repairing it: an artifact whose bytes moved after
    the split was frozen invalidates the case, and a case that quietly repaired
    itself would be worse than one that failed loudly.
    """
    out_dir = pathlib.Path(str(out_dir or DEFAULT_OUT_DIR))
    rep = {"ok": True, "out_dir": str(out_dir), "cases": [], "errors": [],
           "drift": 0, "checked": 0}
    ids = [case] if case else _case_ids(out_dir)
    if not ids:
        rep["errors"].append("no cases under %s" % (out_dir / "cases"))
        rep["ok"] = False
        return rep
    for cid in ids:
        d = out_dir / "cases" / cid
        entry = {"case_id": cid, "ok": True, "artifacts": 0, "problems": []}
        bundle = read_json(d / "bundle.json")
        truth = read_json(d / "truth.json")
        if bundle is None:
            entry["problems"].append("bundle.json missing or unparseable")
        if truth is None:
            entry["problems"].append("truth.json missing or unparseable")
        if bundle is not None:
            recomputed = sha256_str(canonical_json(dict(bundle, sha256="")))
            if recomputed != bundle.get("sha256"):
                entry["problems"].append(
                    "bundle sha256 does not recompute (recorded %s, recomputed %s)"
                    % (str(bundle.get("sha256"))[:12], recomputed[:12]))
            for a in (bundle.get("export", {}).get("artifacts") or []):
                if not a.get("present"):
                    continue
                entry["artifacts"] += 1
                rep["checked"] += 1
                p = d / a["stored"]
                try:
                    with open(str(p), "r", encoding="utf-8") as f:
                        text = f.read()
                except OSError as exc:
                    entry["problems"].append("%s: stored copy unreadable (%s)"
                                             % (a["artifact_id"], exc))
                    continue
                if sha256_str(text) != a.get("stored_sha256"):
                    entry["problems"].append("%s: stored copy has drifted"
                                             % a["artifact_id"])
                if root:
                    src = _clean_str(a.get("path")).replace("<REPO>", str(root))
                    sha, nbytes, _, _, err = _hash_and_count(src)
                    if err:
                        entry["problems"].append("%s: original not readable at %s"
                                                 % (a["artifact_id"], src))
                    elif sha != a.get("sha256"):
                        entry["problems"].append(
                            "%s: ORIGINAL has changed on disk (recorded %s bytes, now %d)"
                            % (a["artifact_id"], a.get("bytes"), nbytes))
        if entry["problems"]:
            entry["ok"] = False
            rep["ok"] = False
            rep["drift"] += len(entry["problems"])
        rep["cases"].append(entry)
    split = read_json(out_dir / "split.json")
    if isinstance(split, dict):
        digest = corpus_digest(out_dir, sorted(set(list(split.get("dev") or []) +
                                                    list(split.get("test") or []))))
        rep["split_sha256_recorded"] = split.get("sha256")
        rep["split_sha256_recomputed"] = digest
        if digest != split.get("sha256"):
            rep["ok"] = False
            rep["errors"].append("split.json sha256 does not recompute over its case set")
    return rep


def list_cases(out_dir=None):
    """One row per case: id, provenance, class, artifact count, bytes."""
    out_dir = pathlib.Path(str(out_dir or DEFAULT_OUT_DIR))
    rows = []
    for cid in _case_ids(out_dir):
        d = out_dir / "cases" / cid
        bundle = read_json(d / "bundle.json") or {}
        truth = read_json(d / "truth.json") or {}
        arts = [a for a in (bundle.get("export", {}).get("artifacts") or [])
                if a.get("present")]
        rows.append({
            "case_id": cid,
            "provenance": (truth.get("provenance")
                           or bundle.get("export", {}).get("provenance")),
            "class": truth.get("class"),
            "incident": truth.get("incident"),
            "date": truth.get("date"),
            "artifacts": len(arts),
            "bytes_stored": sum(int(a.get("stored_bytes") or 0) for a in arts),
            "bytes_original": sum(int(a.get("bytes") or 0) for a in arts),
            "token_estimate": bundle.get("token_estimate"),
        })
    return rows


# Dev/test assignment, §4.2 of the plan. Stated in split.json verbatim so a
# reader never has to trust this docstring.
DEV_CUTOFF = "2026-08-25"
DEV_EXAMPLE_DATE = "2026-08-29"
_SPLIT_RULE = (
    "dev = every case dated on or before {cutoff}, plus every case dated {example} "
    "(the walltime pair that is the worked example and is therefore never reported), "
    "plus the explicitly listed ids {extra}; test = every other case, incidents and "
    "healthy controls alike, so each side carries its own false-alarm denominator. "
    "Assignment is a pure function of (date, case_id) and is recomputed by "
    "`corpus.py freeze --dry-run`. sha256 is taken over the canonical JSON of "
    "[[case_id, bundle.sha256, sha256 of truth.json bytes], ...] sorted by case_id."
)


def corpus_digest(out_dir, ids):
    """sha256 over the frozen case set: the number `freeze` writes into
    split.json as `sha256`.

    Rows are [case_id, the bundle's own recorded sha256, sha256 of the
    truth.json bytes], sorted by case id, hashed as canonical JSON. Public
    because the scoring harness has to certify that it scored the corpus that
    was committed, and two hashes of the same case set that disagree are worse
    than one.
    """
    rows = []
    for cid in sorted(ids):
        d = pathlib.Path(str(out_dir)) / "cases" / cid
        bundle = read_json(d / "bundle.json") or {}
        try:
            with open(str(d / "truth.json"), "rb") as f:
                tsha = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            tsha = ""
        rows.append([cid, bundle.get("sha256") or "", tsha])
    return sha256_str(canonical_json(rows))


# The name this shipped under before it was needed outside the module.
_corpus_digest = corpus_digest


def freeze(out_dir=None, cutoff=DEV_CUTOFF, dev_extra=(), dry_run=False):
    """Write split.json over the exported case set. Never overwrites one.

    Pre-registration is the whole value of this file: a split that can be
    rewritten after a model has been run is not a split. There is deliberately
    no --force; removing the file is a decision a person makes by hand.
    """
    out_dir = pathlib.Path(str(out_dir or DEFAULT_OUT_DIR))
    path = out_dir / "split.json"
    rep = {"ok": False, "path": str(path), "dry_run": bool(dry_run),
           "split": None, "warnings": [], "errors": []}
    ids = _case_ids(out_dir)
    if not ids:
        rep["errors"].append("no cases under %s — nothing to freeze"
                             % (out_dir / "cases"))
        return rep
    if path.exists() and not dry_run:
        rep["errors"].append(
            "%s already exists; the split is frozen and this tool will not overwrite it "
            "(remove the file by hand if the corpus is being re-registered)" % path)
        return rep
    extra = set(dev_extra or ())
    dev, test = [], []
    for cid in ids:
        truth = read_json(out_dir / "cases" / cid / "truth.json") or {}
        date = _clean_str(truth.get("date"))
        if not date:
            rep["warnings"].append("%s has no date; assigned to test" % cid)
        if cid in extra or (date and (date <= cutoff or date == DEV_EXAMPLE_DATE)):
            dev.append(cid)
        else:
            test.append(cid)
    split = {"dev": sorted(dev), "test": sorted(test),
             "rule": _SPLIT_RULE.format(cutoff=cutoff, example=DEV_EXAMPLE_DATE,
                                        extra=sorted(extra) or "[]"),
             "sha256": corpus_digest(out_dir, ids)}
    rep["split"] = split
    if path.exists():
        rep["warnings"].append("%s already exists; this is a dry run of what a "
                               "fresh freeze would produce" % path)
    if not (out_dir / "rubric.md").exists():
        rep["warnings"].append(
            "rubric.md is not present; the gate requires it committed with the split")
    if not dry_run:
        write_json(path, split)
    rep["ok"] = True
    return rep


# --- CLI -------------------------------------------------------------------
def _print_export(rep):
    print("export  spec=%s  out=%s%s" % (rep.get("spec"), rep.get("out_dir"),
                                         "  [dry-run]" if rep.get("dry_run") else ""))
    for c in rep["cases"]:
        print("  %-8s %-46s %-11s art=%-3s scrub=%-4s stored=%s"
              % (c.get("status"), c.get("case_id"), c.get("provenance") or "-",
                 c.get("artifacts"), c.get("scrub_total"), c.get("bytes_stored")))
        if c.get("reason"):
            print("           reason: %s" % c["reason"])
        for m in (c.get("artifacts_missing") or []):
            print("           missing artifact: %s" % m)
        for u in (c.get("unresolved_load_bearing") or []):
            print("           unresolved citation: %s line %s (%s)"
                  % (u.get("artifact"), u.get("line"), u.get("why")))
        for w in (c.get("warnings") or []):
            print("           note: %s" % w)
        if c.get("scrub"):
            print("           scrub: %s" % json.dumps(c["scrub"], sort_keys=True))
    k = rep["counts"]
    print("  %d written, %d refused, %d skipped (%d raw, %d record-only)"
          % (k["written"], k["refused"], k["skipped"], k["raw"], k["record_only"]))
    for e in rep["errors"]:
        print("  ERROR %s" % e)


def _print_verify(rep):
    print("verify  out=%s  %d artifact(s) checked" % (rep["out_dir"], rep["checked"]))
    for c in rep["cases"]:
        print("  %-4s %-46s art=%s" % ("ok" if c["ok"] else "DRIFT", c["case_id"],
                                       c["artifacts"]))
        for p in c["problems"]:
            print("        %s" % p)
    if "split_sha256_recorded" in rep:
        print("  split sha256 recorded=%s recomputed=%s"
              % (str(rep["split_sha256_recorded"])[:16],
                 str(rep["split_sha256_recomputed"])[:16]))
    for e in rep["errors"]:
        print("  ERROR %s" % e)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="corpus", description="Supervision-benchmark incident corpus export")
    ap.add_argument("--out", default=None, help="benchmark directory (default: %s)"
                                                % DEFAULT_OUT_DIR)
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    sub = ap.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    p = sub.add_parser("export", parents=[common],
                       help="build cases from an inventory file")
    p.add_argument("--spec", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--root", default=None, help="base for relative artifact paths")
    p.add_argument("--keep-raw", action="store_true",
                   help="also copy unscrubbed originals to artifacts/raw (never commit)")

    p = sub.add_parser("verify", parents=[common],
                       help="re-hash every artifact and report drift")
    p.add_argument("--case", default=None)
    p.add_argument("--root", default=None,
                   help="repo root to re-hash the ORIGINAL artifacts against")

    sub.add_parser("list", parents=[common], help="one line per exported case")

    p = sub.add_parser("freeze", parents=[common],
                       help="write split.json (refuses to overwrite)")
    p.add_argument("--cutoff", default=DEV_CUTOFF)
    p.add_argument("--dev-extra", nargs="*", default=())
    p.add_argument("--dry-run", action="store_true")

    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:                 # argparse's own exit, including --help
        return int(exc.code or 0)
    if not a.cmd:
        ap.print_help()
        return 2

    try:
        if a.cmd == "export":
            rep = export(a.spec, out_dir=a.out, only=a.only, dry_run=a.dry_run,
                         root=a.root, keep_raw=a.keep_raw)
            if a.json:
                print(json.dumps(rep, sort_keys=True, indent=1))
            else:
                _print_export(rep)
            return 0 if rep["ok"] else 1
        if a.cmd == "verify":
            rep = verify(out_dir=a.out, case=a.case, root=a.root)
            if a.json:
                print(json.dumps(rep, sort_keys=True, indent=1))
            else:
                _print_verify(rep)
            return 0 if rep["ok"] else 1
        if a.cmd == "list":
            rows = list_cases(out_dir=a.out)
            if a.json:
                print(json.dumps(rows, sort_keys=True, indent=1))
            else:
                for r in rows:
                    print("%-46s %-11s %-12s %-9s art=%-3d bytes=%d"
                          % (r["case_id"], r["provenance"] or "-", r["class"] or "-",
                             "incident" if r["incident"] else "healthy",
                             r["artifacts"], r["bytes_stored"]))
                print("%d case(s)" % len(rows))
            return 0 if rows else 1
        if a.cmd == "freeze":
            rep = freeze(out_dir=a.out, cutoff=a.cutoff, dev_extra=a.dev_extra,
                         dry_run=a.dry_run)
            if a.json:
                print(json.dumps(rep, sort_keys=True, indent=1))
            else:
                for w in rep["warnings"]:
                    print("  note: %s" % w)
                for e in rep["errors"]:
                    print("  ERROR %s" % e)
                if rep["split"]:
                    print("dev=%d test=%d sha256=%s%s"
                          % (len(rep["split"]["dev"]), len(rep["split"]["test"]),
                             rep["split"]["sha256"],
                             "  [dry-run]" if rep["dry_run"] else ""))
            return 0 if rep["ok"] else 1
    except Exception as exc:
        # A top-level entry point reports what it could not do; it does not die
        # with a traceback in the middle of a scheduler tick or a batch job.
        print("corpus: %s: %s" % (type(exc).__name__, exc))
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
