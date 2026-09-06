"""Deterministic signals over an evidence bundle (v3.27.0).

Why this exists
---------------
On 2026-08-29 two train jobs (44727703, 44767709) requested 60 epochs on a
merged pool that had grown to 8,583 iterations per epoch, so the recipe needed
about 29 h against a 12 h walltime. Both were killed at the wall (`sacct` State
TIMEOUT, Elapsed 12:00:18 and 12:00:20 against Timelimit 12:00:00), the second
because the loop resubmitted the identical command, and the stop-loss then held
the domain for six days. Every fact needed to call that failure early was in an
artifact the whole time: the growth was in the previous round's iteration count,
the projection was in the epoch times, the verdict was in one `sacct` field.
Nothing read them.

This module is that reading, as pure functions over one bundle dict. It is the
`A0+` arm of the benchmark and the escalation authority of the supervision
layer: a model's confidence never decides whether something is wrong, an
invariant over an artifact does (`TIERED_SUPERVISION_PLAN` §2.2).

What a signal is
----------------
    {"signal": str,
     "severity": "unknown" | "info" | "warn" | "crit",
     "value": <the number that decided it, or None>,
     "reason": str,
     "evidence": [{"artifact_id": str, "line": int, "quote": str}, ...]}

Three properties are load-bearing:

* **Every fired signal carries evidence.** A finding without an address is not
  actionable and is not a signal; `detect()` downgrades a fired check that
  produced no evidence to `unknown` rather than emitting it, so the invariant
  holds structurally instead of by inspection. Evidence is ordered most-citable
  first: a row taken from a line-addressed artifact (`out_tail`) comes before a
  row taken from a parsed section, because the benchmark's deterministic arm
  quotes `evidence[0]` and that quote is checked by the citation validator.
  A `section:<name>` artifact id is deliberately not a file address — the
  section it names (parsed `sacct` rows, the strategy JSON) has no line numbers
  in the bundle, and saying so is better than inventing a line.
* **A check that cannot run says so.** Nothing here raises out of `detect()`,
  and a section that is missing, empty or shaped differently produces
  `severity: "unknown"` with a reason naming what was missing. A detector that
  switches itself off silently is the failure class this campaign exists to
  close; an unknown ranks below `info`, so it can never become a false alarm.
* **Every threshold is in `thresholds.json`, next to the reason it has the
  value it has.** There are no compiled fallbacks: a threshold the file does
  not carry makes its check `unknown` and names the missing key. That is
  deliberate. Two copies of a pre-registered number drift, and a drifted
  threshold silently rescores the whole benchmark. The file is read at call
  time (cached on mtime), so changing a number needs no code edit and no
  restart; environment overrides win over the file, and can only override a key
  the file already declares.

The twelve
----------
Eleven signals are the ones the incident corpus actually weights (162 cases:
walltime_bound 20, pool_growth 18, gate_noop 13, stale_artifact 9, plateau 7,
source_degraded 7, job_unknown 6, ownership_violation 4, budget 2, mongo_down 1,
plus disk_low, which has no case in the record but guards the shared project
quota every job depends on). `epochs_truncated` is the twelfth and is newer than
the corpus: the v3.25.0 Ultralytics `time=` cap created it. A capped run ends
early, writes a valid `best.pt` and reports COMPLETED, so a 60-epoch recipe that
ran 24 is indistinguishable from one that ran in full unless something compares
the two counts. The signal that used to be `walltime_bound` for that run is now
this one, and the two are mutually exclusive by construction: a run WITH an
active cap that ends early is a truncated recipe, not a walltime kill.

`source_degraded` reports NEW degradation only. The line
`[net] WARN: SOCKS proxy via <login node> failed (compute->login SSH disabled) -
SKIPPING github, using Kaggle/HF only` is in every harvest log since June: GitHub
is unreachable from Bridges-2 compute nodes and the collector is designed to
skip it. It is an environment fact and is reported at `info`; an arm that
escalates it produces a false alarm on the dev case's own bundle.

Bundle contract
---------------
`bundle["sections"]` holds the 14 sections frozen by `corpus.py` (`SECTION_ORDER`
below is the same tuple, restated so this module has no import). This module
reads them and never writes; it ignores `sections["signals"]`, which is where its
own output is stored. Shapes it relies on are named in `explain()` per signal and
in the extractor docstrings. Anything unrecognised is a reason, not a guess.

Pure stdlib, no network, no I/O beyond `thresholds.json`: this imports inside a
SLURM job body, on the always-on server and on a laptop.

    from weed_optimizer_framework.tools.brain import signals
    for sig in signals.detect(bundle):
        print(sig["signal"], sig["severity"], sig["reason"])

CLI (a bundle file in, the signals out):
    python -m weed_optimizer_framework.tools.brain.signals detect <bundle.json>
    python -m weed_optimizer_framework.tools.brain.signals thresholds
    python -m weed_optimizer_framework.tools.brain.signals explain [<signal>]
"""
import json
import os
import pathlib
import re

TOOL_VERSION = "wp3-signals/1"

# Ordered by the weight the incident corpus gives them, so a reader of a signal
# list sees the classes that actually happened first. `epochs_truncated` sits
# second because it is the other half of the walltime story (v3.25.0's cap).
SIGNALS = (
    "walltime_bound",
    "epochs_truncated",
    "pool_growth",
    "gate_noop",
    "stale_artifact",
    "plateau",
    "source_degraded",
    "job_unknown",
    "budget",
    "disk_low",
    "ownership_violation",
    "mongo_down",
)

# "ok" is internal: a check that ran and found nothing is not reported by
# detect(), only by detect_all(). "unknown" ranks below "info" so a check that
# could not run can never be counted as a detection or as a false alarm.
SEVERITIES = ("ok", "unknown", "info", "warn", "crit")
_SEV_RANK = {"ok": -2, "unknown": -1, "info": 0, "warn": 1, "crit": 2}

# The 14 bundle sections, in the order corpus.py froze them. Restated rather
# than imported: this module runs inside a SLURM job that has no corpus.
SECTION_ORDER = ("ledger", "sacct", "out_tail", "results_csv", "strategy",
                 "trace", "slug_scores", "registry_diff", "harvest",
                 "resources", "su", "corrections", "plan", "signals")

# Terminal SLURM states. Used only to decide whether a run "reported success",
# which is what separates epochs_truncated from a failure.
_FAILED_STATES = ("TIMEOUT", "FAILED", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY",
                  "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT")

# --- thresholds -------------------------------------------------------------
# One place, and only one. Resolution order, lowest to highest: thresholds.json
# beside this module, a dict the caller passes to detect(), then the
# environment. An env override is named BRAIN_SIGNAL_<KEY WITH DOTS AS
# UNDERSCORES>, upper-cased, and can only override a key the file declares —
# the file is the pre-registered record of which thresholds exist at all.
_THRESHOLD_FILE = pathlib.Path(__file__).resolve().parent / "thresholds.json"
_FILE_CACHE = {"key": None, "block": {}, "error": None, "path": ""}


class _MissingThreshold(Exception):
    """A check asked for a threshold the file does not declare."""


def _threshold_path():
    return pathlib.Path(os.environ.get("BRAIN_SIGNAL_THRESHOLDS") or _THRESHOLD_FILE)


def _file_block():
    """The `signals` block of thresholds.json as {dotted key: (value, why)}.

    Cached on (path, mtime, size) so the always-on server does not re-parse the
    file every tick while an operator editing it still takes effect on the next
    call. Missing or malformed is a reported state, never an exception.
    """
    path = _threshold_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError as exc:
        _FILE_CACHE.update({"key": None, "block": {}, "path": str(path),
                            "error": "thresholds file unreadable at %s (%s)"
                                     % (path.name, type(exc).__name__)})
        return {}
    if _FILE_CACHE["key"] == key:
        return _FILE_CACHE["block"]
    block, error = {}, None
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        raw = (obj or {}).get("signals")
        if not isinstance(raw, dict):
            error = "thresholds.json carries no 'signals' object"
        else:
            block = _flatten_thresholds(raw)
    except Exception as exc:
        error = "thresholds.json unreadable: %s" % type(exc).__name__
    _FILE_CACHE.update({"key": key, "block": block, "error": error,
                        "path": str(path)})
    return block


def _flatten_thresholds(raw, prefix=""):
    """{"a": {"b": {"value": 1, "why": "..."}}} -> {"a.b": (1, "...")}.

    A bare scalar is accepted with an empty reason so a caller's override dict
    can be flat; the committed file always carries the reason, and
    `thresholds()` reports which keys arrived without one.
    """
    out = {}
    for name in sorted(raw.keys()):
        if str(name).startswith("_"):
            continue                      # _meta / _why notes, not thresholds
        val = raw[name]
        key = ("%s.%s" % (prefix, name)) if prefix else str(name)
        if isinstance(val, dict) and "value" in val:
            out[key] = (val.get("value"), str(val.get("why") or ""))
        elif isinstance(val, dict):
            out.update(_flatten_thresholds(val, key))
        else:
            out[key] = (val, "")
    return out


def _env_name(key):
    return "BRAIN_SIGNAL_" + re.sub(r"[^A-Za-z0-9]", "_", str(key)).upper()


def _parse_env(text):
    """A threshold from the environment: JSON first, then a bare number."""
    raw = str(text).strip()
    try:
        return json.loads(raw), None
    except Exception:
        pass
    try:
        return float(raw), None
    except ValueError:
        return None, "not JSON and not a number"


def thresholds(overrides=None):
    """Effective thresholds and, per key, where the value came from.

    Returned so a results file can record the rules it was scored under:
    `{"values": {...}, "sources": {...}, "why": {...}, "errors": [...]}`.
    """
    out = {"tool_version": TOOL_VERSION, "values": {}, "sources": {},
           "why": {}, "errors": [], "file": str(_threshold_path())}
    block = _file_block()
    if _FILE_CACHE.get("error"):
        out["errors"].append(_FILE_CACHE["error"])
    for key, (value, why) in block.items():
        out["values"][key] = value
        out["sources"][key] = "thresholds.json"
        out["why"][key] = why
        if not why:
            out["errors"].append("%s has no 'why' in thresholds.json" % key)
    if isinstance(overrides, dict):
        for key, (value, why) in _flatten_thresholds(overrides).items():
            if key not in out["values"]:
                out["errors"].append(
                    "override %s is not declared in thresholds.json; ignored" % key)
                continue
            out["values"][key] = value
            out["sources"][key] = "caller"
            if why:
                out["why"][key] = why
    for key in sorted(out["values"]):
        name = _env_name(key)
        raw = os.environ.get(name)
        if raw in (None, ""):
            continue
        value, err = _parse_env(raw)
        if err:
            out["errors"].append("%s is set but is %s; ignored" % (name, err))
            continue
        out["values"][key] = value
        out["sources"][key] = "env:" + name
    return out


# --- small utilities --------------------------------------------------------
def _num(value, default=None):
    """A float, or `default`. Booleans are not numbers here."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=None):
    v = _num(value)
    return default if v is None else int(v)


def _text(value):
    return value if isinstance(value, str) else ""


def _round(value, places=6):
    """Round for reporting AND for comparison at a boundary.

    Round-metric deltas arrive as 4-decimal numbers whose float difference
    carries noise in the 17th place; the campaign's three scripted rounds sit
    exactly on the plateau boundary (spread 0.0100 against 2 x 0.005), so
    without this the verdict on real data would depend on binary rounding.
    """
    v = _num(value)
    return None if v is None else round(v, places)


def _hms_s(text):
    """SLURM duration to seconds: `[DD-]HH:MM:SS[.us]`, `MM:SS`, or a number."""
    raw = str(text or "").strip()
    if not raw:
        return None
    v = _num(raw)
    if v is not None and ":" not in raw and "-" not in raw:
        return v
    days = 0.0
    if "-" in raw:
        head, _, raw = raw.partition("-")
        days = _num(head, 0.0) or 0.0
    parts = raw.split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.insert(0, 0.0)
    return days * 86400.0 + nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]


def _ts(value):
    """Epoch seconds from a float, a numeric string or ISO-8601.

    The ledger writes ISO-8601 (`db._now_iso`), the scheduler state and the
    trace write epoch floats, and `sacct` writes `2026-08-29T05:52:33`. None
    when it cannot be dated — a comparison against an unknown time is not made.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw or raw.lower() in ("unknown", "none", "n/a"):
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone
        txt = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _hours(seconds):
    v = _num(seconds)
    return "unknown" if v is None else "%.1f h" % (v / 3600.0)


def _get(holder, *names):
    """First present, non-None value among `names` in a dict. Case-insensitive."""
    if not isinstance(holder, dict):
        return None
    for name in names:
        if name in holder and holder[name] is not None:
            return holder[name]
    lowered = {str(k).lower(): v for k, v in holder.items()}
    for name in names:
        v = lowered.get(str(name).lower())
        if v is not None:
            return v
    return None


def _dicts(*values):
    """The dicts among `values`, in order — the holder lists checks search."""
    return [v for v in values if isinstance(v, dict)]


# --- evidence ---------------------------------------------------------------
def _ev_line(row):
    """Evidence from a line-addressed artifact: a real (artifact, line) address."""
    return {"artifact_id": str(row.get("artifact_id") or ""),
            "line": _int(row.get("line"), 0) or 0,
            "quote": _text(row.get("text"))}


def _ev_section(section, index, quote):
    """Evidence from a parsed section, which has no line numbers in the bundle.

    `artifact_id` is `section:<name>` and `line` is the 1-based index of the row
    or record inside that section as the bundle carries it. It is provenance,
    not a citation: `citations.resolve` will not resolve it, and that is the
    honest answer for a fact that was read out of parsed JSON.
    """
    return {"artifact_id": "section:%s" % section,
            "line": max(1, _int(index, 1) or 1),
            "quote": str(quote)}


def _kv(holder, *names):
    """`k=v k=v` rendering of the fields a check actually read, for a quote."""
    parts = []
    for name in names:
        if isinstance(holder, dict) and holder.get(name) is not None:
            parts.append("%s=%s" % (name, holder[name]))
    return " ".join(parts)


def _order_evidence(items):
    """Line-addressed evidence first: `evidence[0]` is what an arm will quote."""
    seen, out = set(), []
    for ev in items:
        if not isinstance(ev, dict) or not ev.get("artifact_id"):
            continue
        key = (ev["artifact_id"], ev.get("line"), ev.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    out.sort(key=lambda e: 1 if str(e["artifact_id"]).startswith("section:") else 0)
    return out


# --- the bundle view --------------------------------------------------------
class _Bundle(object):
    """Cached, defensive reader for one bundle. Nothing here raises."""

    def __init__(self, bundle, overrides=None):
        self.bundle = bundle if isinstance(bundle, dict) else {}
        secs = self.bundle.get("sections")
        if isinstance(secs, dict):
            self.sections = secs
        elif isinstance(self.bundle.get("bundle_id"), str) or "export" in self.bundle:
            self.sections = {}
        else:
            # A caller that hands over the sections object itself is a common
            # slip and costs nothing to accept.
            self.sections = {k: v for k, v in self.bundle.items()
                             if k in SECTION_ORDER}
        self._th = thresholds(overrides)
        self._cache = {}

    # -- thresholds
    def th(self, key):
        if key not in self._th["values"]:
            raise _MissingThreshold(key)
        return self._th["values"][key]

    def th_num(self, key):
        value = _num(self.th(key))
        if value is None:
            raise _MissingThreshold(key)
        return value

    @property
    def threshold_report(self):
        return self._th

    # -- sections
    def sec(self, name):
        return self.sections.get(name)

    def sec_dict(self, name):
        v = self.sections.get(name)
        return v if isinstance(v, dict) else {}

    def sec_list(self, name):
        v = self.sections.get(name)
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    def missing_reason(self, name):
        """Why a section is null, from the export block, or ''."""
        exp = self.bundle.get("export")
        if isinstance(exp, dict) and isinstance(exp.get("missing"), dict):
            return _text(exp["missing"].get(name))
        return ""

    def declared_absent(self, name):
        """True when the case declared an artifact for `name` and it was absent.

        The distinction matters: `slug_scores.json is missing` is an incident,
        while `this bundle carries no slug_scores section` is a gap in the
        bundle, and a check must never report the second as the first.
        """
        exp = self.bundle.get("export")
        if not isinstance(exp, dict):
            return False
        for entry in (exp.get("artifacts") or []):
            if (isinstance(entry, dict) and entry.get("section") == name
                    and entry.get("present") is False):
                return True
        return False

    # -- line-addressed rows (the same address space citations.py indexes)
    def lines(self):
        if "lines" in self._cache:
            return self._cache["lines"]
        rows = []
        names = [n for n in SECTION_ORDER if n in self.sections]
        names += sorted(k for k in self.sections if k not in SECTION_ORDER)
        for name in names:
            if name == "signals":
                continue
            for aid, obj in _line_objects(name, self.sections.get(name)):
                for raw in (obj.get("lines") or []):
                    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                        num, text = raw[0], raw[1]
                    elif isinstance(raw, dict):
                        num, text = raw.get("line"), raw.get("text")
                    else:
                        continue
                    n = _int(num)
                    if n is None or n < 1 or not isinstance(text, str):
                        continue
                    rows.append({"artifact_id": str(aid), "line": n,
                                 "text": text, "section": name})
        self._cache["lines"] = rows
        return rows

    def grep(self, pattern, last=True, limit=1):
        """Rows whose text matches `pattern` (case-insensitive), newest last."""
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            return []
        hits = [r for r in self.lines() if rx.search(r["text"])]
        if not hits:
            return []
        return hits[-limit:] if last else hits[:limit]

    # -- identity
    def job_id(self):
        if "job" in self._cache:
            return self._cache["job"]
        cand = None
        exp = self.bundle.get("export")
        if isinstance(exp, dict):
            cand = exp.get("job_id")
        cand = cand or self.bundle.get("job_id")
        if not cand:
            cand = _get(self.sec_dict("strategy"), "job_id", "jobid", "slurm_job_id")
        if not cand:
            for entry in self.step_entries():
                if entry.get("job"):
                    cand = entry["job"]
                    break
        digits = re.sub(r"[^0-9]", "", str(cand or ""))
        self._cache["job"] = digits
        return digits

    def step(self):
        return _text(self.bundle.get("step")) or _text(
            _get(self.sec_dict("strategy"), "step"))

    def round_no(self):
        return _int(self.bundle.get("round"))

    # -- ledger
    def rounds(self):
        """Round docs oldest first. Accepts the shapes an export really carries."""
        if "rounds" in self._cache:
            return self._cache["rounds"]
        sec = self.sec("ledger")
        docs = []
        if isinstance(sec, list):
            docs = [d for d in sec if isinstance(d, dict)]
        elif isinstance(sec, dict):
            for key in ("rounds", "docs"):
                if isinstance(sec.get(key), list):
                    docs = [d for d in sec[key] if isinstance(d, dict)]
                    break
            if not docs and isinstance(sec.get("round"), dict):
                docs = [sec["round"]]
            if not docs and isinstance(sec.get("steps"), dict):
                docs = [sec]                        # the section IS one round doc
        docs = sorted(docs, key=lambda d: _num(_get(d, "round_num", "round"), 0.0))
        self._cache["rounds"] = docs
        return docs

    def current_round_doc(self):
        want = self.round_no()
        docs = self.rounds()
        if want is not None:
            for doc in docs:
                if _int(_get(doc, "round_num", "round")) == want:
                    return doc
        return docs[-1] if docs else {}

    def previous_round_docs(self):
        """Rounds before the bundle's own, newest first."""
        want = self.round_no()
        docs = self.rounds()
        if want is None:
            return list(reversed(docs[:-1])) if len(docs) > 1 else []
        out = [d for d in docs if (_int(_get(d, "round_num", "round")) or 0) < want]
        return list(reversed(out))

    def step_entries(self, doc=None, step=None):
        """Step entries for one step, attempts first then the head entry."""
        doc = self.current_round_doc() if doc is None else doc
        step = step or self.step()
        steps = doc.get("steps") if isinstance(doc, dict) else None
        entry = steps.get(step) if isinstance(steps, dict) else None
        if not isinstance(entry, dict):
            return []
        out = [a for a in (entry.get("attempts") or []) if isinstance(a, dict)]
        out.append({k: v for k, v in entry.items() if k != "attempts"})
        return out

    # -- sacct
    def sacct_rows(self):
        """Normalised `sacct` rows: jobid, state, elapsed_s, timelimit_s, start."""
        if "sacct" in self._cache:
            return self._cache["sacct"]
        rows = []
        for idx, raw in enumerate(self.sec_list("sacct")):
            jobid = _text(_get(raw, "JobID", "JobIDRaw", "jobid"))
            state = _text(_get(raw, "State", "state")).strip().upper()
            row = {
                "index": idx + 1,
                "jobid": jobid,
                "base_jobid": jobid.split(".")[0].split("_")[0],
                "state": state,
                "elapsed_s": _hms_s(_get(raw, "Elapsed", "elapsed")),
                "timelimit_s": _hms_s(_get(raw, "Timelimit", "TimeLimit", "timelimit")),
                "start_ts": _ts(_get(raw, "Start", "start")),
                "end_ts": _ts(_get(raw, "End", "end")),
                "raw": raw,
            }
            row["quote"] = _kv(raw, *[k for k in ("JobID", "JobName", "State",
                                                  "Elapsed", "Timelimit", "Start",
                                                  "End", "ExitCode")
                                      if k in raw]) or json.dumps(raw, sort_keys=True)
            rows.append(row)
        self._cache["sacct"] = rows
        return rows

    def sacct_row(self, job=None):
        """The bundle's own job row: exact id first, then the base id, then the
        single row a one-job export carries."""
        want = re.sub(r"[^0-9]", "", str(job or self.job_id() or ""))
        rows = self.sacct_rows()
        if not rows:
            return None
        if want:
            for row in rows:
                if row["jobid"] == want:
                    return row
            for row in rows:
                if row["base_jobid"] == want:
                    return row
        primary = [r for r in rows if "." not in r["jobid"]]
        if len(primary) == 1:
            return primary[0]
        return None

    # -- trace
    def trace_records(self, kind=None):
        recs = self.sec_list("trace")
        if kind is None:
            return recs
        return [r for r in recs if _text(r.get("kind")) == kind]

    # -- strategy holders
    def strategy_holders(self):
        """[(section label, dict)] that may carry a run's parameters.

        Labelled because the label becomes the evidence address: a number this
        module decided with has to say which part of the bundle it came from.
        Ordered by authority — the job-scoped strategy artifact first, then the
        run's own trace, then the ledger entry, then the domain config.
        """
        if "holders" in self._cache:
            return self._cache["holders"]
        strat = self.sec_dict("strategy")
        holders = [("strategy", h) for h in
                   _dicts(strat, strat.get("strategy"), strat.get("params"),
                          strat.get("round_params"), strat.get("summary"),
                          strat.get("effective"))]
        for rec in (self.trace_records("end")[-1:]
                    + self.trace_records("start")[-1:]):
            holders += [("trace", h) for h in _dicts(rec, rec.get("strategy"))]
        for entry in self.step_entries():
            holders += [("ledger", h) for h in
                        _dicts(entry.get("params"), entry.get("metrics"))]
        holders += [("ledger", h) for h in
                    _dicts(self.config_block("round_params"))]
        self._cache["holders"] = holders
        return holders

    def config_holders(self):
        """Dicts that may carry the domain config, in decreasing authority.

        The config is not a bundle section: it reaches a bundle inside the
        ledger staging (`ledger.config`) or the plan. A check that needs a value
        from it and does not find one reports unknown rather than assuming a
        default — `noise_floor` in particular is a per-recipe measurement, and a
        guessed noise floor turns noise into a result.
        """
        holders = []
        for base in (self.sec_dict("ledger"), self.sec_dict("plan"), self.bundle):
            holders += _dicts(base.get("config"), base.get("domain_config"), base)
        return holders

    def config_block(self, name):
        """A dict-valued domain-config block (`noise_floor`, `budget`), or {}."""
        for holder in self.config_holders():
            block = holder.get(name)
            if isinstance(block, dict) and block:
                return block
        return {}

    def config_str(self, name):
        """A string-valued domain-config field (`target_metric`), or ""."""
        for holder in self.config_holders():
            value = holder.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


def _line_objects(section_name, value):
    """[(artifact_id, obj)] for the line-addressed objects in one section.

    Matches `citations._line_objects`: one object, a list of them, or a mapping
    of name -> object. Anything else has no (artifact, line) address.
    """
    out = []
    if isinstance(value, dict) and isinstance(value.get("lines"), list):
        out.append((value.get("artifact_id") or value.get("path") or section_name,
                    value))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("lines"), list):
                out.append((item.get("artifact_id") or item.get("path")
                            or section_name, item))
    elif isinstance(value, dict):
        for key in sorted(value.keys()):
            item = value[key]
            if isinstance(item, dict) and isinstance(item.get("lines"), list):
                out.append((item.get("artifact_id") or item.get("path") or key,
                            item))
    return out


# --- signal constructors ----------------------------------------------------
def _sig(name, severity, value, reason, evidence=None):
    return {"signal": name, "severity": severity, "value": value,
            "reason": str(reason), "evidence": _order_evidence(evidence or [])}


def _ok(name, reason):
    return _sig(name, "ok", None, reason)


def _unknown(name, reason):
    return _sig(name, "unknown", None, reason)


# --- shared extractors ------------------------------------------------------
# Keys a builder may use for the same fact. Listed rather than guessed at: an
# unrecognised spelling makes a check say "unknown", never assume a number.
_ITER_KEYS = ("iterations_per_epoch", "iters_per_epoch", "steps_per_epoch",
              "iterations", "iters")
_PREV_ITER_KEYS = tuple("previous_" + k for k in _ITER_KEYS) + \
                  tuple("prev_" + k for k in _ITER_KEYS)
_IMAGE_KEYS = ("merged_images", "train_images", "pool_images", "images")
_BATCH_KEYS = ("batch", "batch_size")
_EPOCH_REQ_KEYS = ("epochs_requested", "epochs_total", "epochs")
_EPOCH_DONE_KEYS = ("epochs_completed", "epochs_done", "last_epoch")
_CAP_KEYS = ("train_time_cap_h", "time_h", "time_cap_h")
_SUCCESS_WORDS = ("done", "ok", "success", "succeeded", "complete", "completed",
                  "finished")
_FAILURE_WORDS = ("fail", "failed", "error", "timeout", "cancelled", "canceled",
                  "killed", "oom")
# The ultralytics progress bar closes each epoch with `<n>/<n> [`: the only
# place a job log states the iteration count of the pool it really trained on.
_PROGRESS_RE = re.compile(r"\b(\d{2,})/(\d{2,})\b\s*\[")


def _scan(holders, keys):
    """(value, label, key) for the first of `keys` any labelled holder carries."""
    for label, holder in holders:
        for key in keys:
            value = _num(_get(holder, key))
            if value is not None:
                return value, label, key
    return None, "", ""


def _time_cap_h(b):
    """(hours, label, key) of an active Ultralytics `time=` cap, or (None,...).

    v3.25.0 passes `TRAIN_TIME_H` (0.9 x walltime) into `time=`, which overrides
    the epoch budget and ends the run with a valid best.pt. A run under a cap
    cannot be walltime-bound in the 2026-08-29 sense; it is truncated instead.
    """
    for label, holder in b.strategy_holders():
        for key in _CAP_KEYS:
            value = _num(_get(holder, key))
            if value is not None and value > 0:
                return value, label, key
    return None, "", ""


def _run_outcome(b):
    """("success" | "failed" | "unknown", [evidence]) for the bundle's own run."""
    row = b.sacct_row()
    if row is not None and row["state"]:
        ev = [_ev_section("sacct", row["index"], row["quote"])]
        if row["state"].startswith("COMPLETED"):
            return "success", ev
        if row["state"].startswith(_FAILED_STATES):
            return "failed", ev
    strat = b.sec_dict("strategy")
    status = _text(_get(strat, "status", "state", "result")).strip().lower()
    if status:
        ev = [_ev_section("strategy", 1, "status=%s" % status)]
        if status in _SUCCESS_WORDS:
            return "success", ev
        if any(word in status for word in _FAILURE_WORDS):
            return "failed", ev
    for rec in b.trace_records("end")[-1:]:
        if isinstance(rec.get("ok"), bool):
            ev = [_ev_section("trace", 1, "kind=end ok=%s" % rec["ok"])]
            return ("success" if rec["ok"] else "failed"), ev
    return "unknown", []


def _epoch_counts(b):
    """(requested, completed, [evidence], reason-when-unavailable).

    v3.25.0 records the pair side by side (`epochs_requested` /
    `epochs_completed`) in the job-scoped artifact and in the trace `end`
    record, precisely so a capped run is distinguishable from a full one. The
    fallbacks are the last epoch record and the row count of results.csv, which
    is what a pre-v3.25.0 artifact leaves behind.
    """
    holders = b.strategy_holders()
    req, req_label, req_key = _scan(holders, _EPOCH_REQ_KEYS)
    done, done_label, done_key = _scan(holders, _EPOCH_DONE_KEYS)
    ev = []
    if req is not None:
        ev.append(_ev_section(req_label, 1, "%s=%s" % (req_key, int(req))))
    if done is not None:
        ev.append(_ev_section(done_label, 1, "%s=%s" % (done_key, int(done))))
    if done is None:
        epochs = b.trace_records("epoch")
        if epochs:
            done = _num(epochs[-1].get("epoch"))
            if done is not None:
                ev.append(_ev_section("trace", len(epochs),
                                      "kind=epoch epoch=%s elapsed_s=%s"
                                      % (epochs[-1].get("epoch"),
                                         epochs[-1].get("elapsed_s"))))
    if done is None:
        rows = _num(_get(b.sec_dict("results_csv"), "rows"))
        if rows is not None and rows > 0:
            done = rows
            ev.append(_ev_section("results_csv", 1, "rows=%d" % int(rows)))
    if req is None or done is None:
        missing = []
        if req is None:
            missing.append("epochs_requested")
        if done is None:
            missing.append("epochs_completed")
        return None, None, ev, ("the bundle carries no %s (looked in the "
                                "strategy artifact, the trace and results.csv)"
                                % " and no ".join(missing))
    return req, done, ev, ""


def _step_start(b):
    """(epoch seconds, source label, [evidence]) for when this step started.

    `sacct` Start is preferred over the ledger's `running` entry because the
    artifact mtimes it will be compared against come from the cluster
    filesystem — the same clock. The ledger's timestamp is the lab's, and the
    skew between the two is the whole reason `stale_artifact` has a grace.
    """
    row = b.sacct_row()
    if row is not None and row["start_ts"]:
        return row["start_ts"], "sacct Start", [
            _ev_section("sacct", row["index"], row["quote"])]
    stamps = []
    for idx, entry in enumerate(b.step_entries()):
        if _text(entry.get("status")).lower() == "running":
            ts = _ts(entry.get("at"))
            if ts is not None:
                stamps.append((ts, idx + 1, entry))
    if stamps:
        ts, idx, entry = min(stamps, key=lambda x: x[0])
        return ts, "the ledger's running entry", [
            _ev_section("ledger", idx, _kv(entry, "status", "at", "job", "actor"))]
    return None, "", []


def _pool_pair(b):
    """(now, previous, unit, [evidence], reason) — this pool against the last.

    Iterations per epoch is the operational form of the pool and is tried
    first, on both sides, from the strategy artifact, the log's own progress
    bar and the previous round's train entry. Merged image counts are the
    fallback and are only ever compared with each other: the two are different
    units, and a growth computed across them would be arithmetic on nothing.
    """
    now, ev_now, why_now = _iters_now(b)
    prev, ev_prev, why_prev = _iters_prev(b)
    if now is not None and prev is not None:
        return now, prev, "iterations/epoch", ev_now + ev_prev, ""
    img_now, bat_now, ev_now2, why_now2 = _images_now(b)
    img_prev, bat_prev, ev_prev2, why_prev2 = _images_prev(b)
    if img_now is not None and img_prev is not None:
        if bat_now and bat_prev and bat_now != bat_prev:
            return None, None, "", [], (
                "only image counts are available and the batch size changed "
                "from %d to %d, so the two rounds' iteration counts are not in "
                "the same ratio as their pools" % (int(bat_prev), int(bat_now)))
        note = ("" if (bat_now and bat_prev) else
                " (measured on the merged image count, which grows in the same "
                "ratio as iterations per epoch only if the batch size did not "
                "change; no batch size is recorded on both sides)")
        return img_now, img_prev, "images", ev_now2 + ev_prev2, note
    missing = [w for w in (why_now or why_now2, why_prev or why_prev2) if w]
    return None, None, "", [], "; ".join(missing) or (
        "neither an iteration count nor a merged image count is available for "
        "both this round and the one before it")


def _iters_now(b):
    """(iterations per epoch, [evidence], reason) for this run."""
    value, label, key = _scan(b.strategy_holders(), _ITER_KEYS)
    if value is not None and value > 0:
        ev = [_ev_section(label, 1, "%s=%d" % (key, int(value)))]
        ev += [_ev_line(r) for r in _progress_rows(b, int(value))]
        return value, ev, ""
    rows = _progress_rows(b)
    if rows:
        row = rows[-1]
        return _int(_PROGRESS_RE.search(row["text"]).group(1)), [_ev_line(row)], ""
    return None, [], ("no iteration count for this run: the strategy carries "
                      "none of %s and no completed progress line in the log "
                      "states one" % ", ".join(_ITER_KEYS))


def _iters_prev(b):
    """(iterations per epoch, [evidence], reason) for the previous train.

    Never from this run's own numbers: growth measured against itself is zero.
    """
    value, label, key = _scan(b.strategy_holders(), _PREV_ITER_KEYS)
    if value is not None and value > 0:
        return value, [_ev_section(label, 1, "%s=%d" % (key, int(value)))], ""
    for doc in b.previous_round_docs():
        rnd = _int(_get(doc, "round_num", "round"))
        for idx, entry in enumerate(b.step_entries(doc, "train")):
            holders = [("ledger", h) for h in
                       _dicts(entry, entry.get("params"), entry.get("metrics"))]
            got, _lab, gkey = _scan(holders, _ITER_KEYS)
            if got is not None and got > 0:
                return got, [_ev_section("ledger", idx + 1,
                                         "round=%s train %s=%d"
                                         % (rnd, gkey, int(got)))], ""
    return None, [], ("no previous train to compare against: neither the "
                      "strategy nor the staged ledger records an earlier "
                      "round's iterations per epoch")


def _images_now(b):
    """(merged images, batch, [evidence], reason) for this run."""
    value, label, key = _scan(b.strategy_holders(), _IMAGE_KEYS)
    batch, _bl, _bk = _scan(b.strategy_holders(), _BATCH_KEYS)
    if value is not None and value > 0:
        return value, batch, [_ev_section(label, 1, "%s=%d" % (key, int(value)))], ""
    diff = b.sec_dict("registry_diff")
    after = _num(_get(diff, "images_after", "images_now", "images"))
    if after is not None and after > 0:
        return after, batch, [_ev_section("registry_diff", 1,
                                          _kv(diff, "images_before",
                                              "images_after"))], ""
    return None, batch, [], "no merged image count is recorded for this run"


def _images_prev(b):
    """(merged images, batch, [evidence], reason) for the previous train."""
    diff = b.sec_dict("registry_diff")
    before = _num(_get(diff, "images_before", "images_previous"))
    if before is not None and before > 0:
        return before, None, [_ev_section("registry_diff", 1,
                                          _kv(diff, "images_before",
                                              "images_after"))], ""
    for doc in b.previous_round_docs():
        rnd = _int(_get(doc, "round_num", "round"))
        for idx, entry in enumerate(b.step_entries(doc, "train")):
            holders = [("ledger", h) for h in
                       _dicts(entry, entry.get("params"), entry.get("metrics"))]
            got, _lab, gkey = _scan(holders, _IMAGE_KEYS)
            batch, _bl, _bk = _scan(holders, _BATCH_KEYS)
            if got is not None and got > 0:
                return got, batch, [_ev_section("ledger", idx + 1,
                                               "round=%s train %s=%d"
                                               % (rnd, gkey, int(got)))], ""
    return None, None, [], ("no previous train's merged image count is on the "
                            "staged ledger")


def _progress_rows(b, want=None):
    """Log rows whose progress bar closed an epoch (`8583/8583 [`)."""
    out = []
    for row in b.lines():
        m = _PROGRESS_RE.search(row["text"])
        if not m or m.group(1) != m.group(2):
            continue
        if want is not None and _int(m.group(1)) != want:
            continue
        out.append(row)
    return out[-1:]


# --- the twelve -------------------------------------------------------------
def _check_walltime_bound(b):
    name = "walltime_bound"
    frac = b.th_num("walltime_bound.eta_fraction")
    need = int(b.th_num("walltime_bound.min_epoch_records"))
    why = []

    row = b.sacct_row()
    if row is None:
        why.append("no sacct row for job %s" % (b.job_id() or "(no job id)"))
    elif row["state"].startswith("TIMEOUT"):
        ratio = None
        if row["elapsed_s"] and row["timelimit_s"]:
            ratio = _round(row["elapsed_s"] / row["timelimit_s"], 4)
        ev = [_ev_section("sacct", row["index"], row["quote"])]
        ev += [_ev_line(r) for r in b.grep(
            r"TIME LIMIT|TIMEOUT|CANCELLED AT .*DUE TO TIME", last=True, limit=2)]
        return _sig(name, "crit", ratio if ratio is not None else row["elapsed_s"],
                    "sacct reports State %s for job %s (Elapsed %s against "
                    "Timelimit %s): the run was killed at the wall, so its "
                    "epochs, its checkpoint and its metric are whatever it had "
                    "reached when SLURM stopped it"
                    % (row["state"], row["jobid"] or b.job_id(),
                       _get(row["raw"], "Elapsed", "elapsed"),
                       _get(row["raw"], "Timelimit", "TimeLimit", "timelimit")),
                    ev)

    # The projection: available from epoch 1, which is the point. The trace is
    # preferred; results.csv is what a pre-v3.25.0 run leaves behind.
    eta = wall = None
    ev, source = [], ""
    epochs = b.trace_records("epoch")
    if len(epochs) >= need:
        last = epochs[-1]
        eta = _num(last.get("eta_total_s"))
        wall = _num(last.get("walltime_s"))
        if wall is None:
            for rec in b.trace_records("start")[-1:]:
                wall = _num(rec.get("walltime_s"))
        source = "the epoch trace (%d records)" % len(epochs)
        ev = [_ev_section("trace", len(epochs),
                          _kv(last, "kind", "epoch", "elapsed_s", "eta_total_s",
                              "walltime_s"))]
    elif epochs:
        why.append("the trace carries %d epoch record(s), fewer than the %d the "
                   "projection needs" % (len(epochs), need))
    if eta is None or wall is None:
        csv = b.sec_dict("results_csv")
        rows = _num(csv.get("rows"))
        per_epoch = _num(csv.get("epoch_time_s"))
        req, _done, _ev, _why = _epoch_counts(b)
        limit = row["timelimit_s"] if row is not None else None
        if (rows is not None and rows >= need and per_epoch and req and limit):
            eta, wall = per_epoch * req, limit
            source = "the results.csv time column (%d rows)" % int(rows)
            ev = [_ev_section("results_csv", 1,
                              _kv(csv, "rows", "epoch_time_s", "last", "best"))]
            if row is not None:
                ev.append(_ev_section("sacct", row["index"], row["quote"]))
        elif rows is not None and rows < need:
            why.append("results.csv carries %d row(s), fewer than the %d the "
                       "projection needs" % (int(rows), need))

    if eta is None or wall is None or wall <= 0:
        why.append("no projection is available (an epoch time and a walltime are "
                   "both needed)")
        return _unknown(name, "; ".join(why) or "nothing in this bundle dates the "
                                                "run against its walltime")
    ratio = _round(eta / wall, 4)
    if ratio <= frac:
        return _ok(name, "projected %s against a %s walltime from %s (%.2f of it)"
                         % (_hours(eta), _hours(wall), source, ratio))
    cap_h, cap_label, cap_key = _time_cap_h(b)
    detail = ("%s projects %s total against a %s walltime (%.2f of it, over the "
              "%.2f trigger)" % (source, _hours(eta), _hours(wall), ratio, frac))
    if cap_h:
        # A capped run is not walltime-bound: it stops on time and leaves a
        # valid checkpoint. What it leaves behind is a recipe that did not run
        # in full, which is epochs_truncated's finding, and reporting both would
        # double-count one fact. The projection is kept in the reason so
        # detect_all() still shows that this check ran and what it saw.
        return _ok(name, detail + ("; a %s h time cap is active (%s.%s), so the "
                                   "run stops on time with a valid checkpoint "
                                   "rather than being killed. The shortfall it "
                                   "leaves is reported by epochs_truncated, not "
                                   "as a walltime kill"
                                   % (cap_h, cap_label, cap_key)))
    return _sig(name, "warn", ratio,
                detail + "; no time cap is active, so this run ends at the wall "
                         "with no checkpoint and no metric", ev)


def _check_epochs_truncated(b):
    name = "epochs_truncated"
    floor = b.th_num("epochs_truncated.completed_fraction_floor")
    req, done, ev, why = _epoch_counts(b)
    if req is None:
        return _unknown(name, why)
    if req <= 0:
        return _unknown(name, "the recipe records %s epochs requested, which "
                              "cannot be compared against" % req)
    frac = _round(done / req, 4)
    if frac >= floor:
        return _ok(name, "%d of %d epochs ran (%.2f of the recipe, at or above "
                         "the %.2f floor)" % (done, req, frac, floor))
    cap_h, cap_label, cap_key = _time_cap_h(b)
    cap_note = ""
    if cap_h:
        ev.append(_ev_section(cap_label, 1, "%s=%s" % (cap_key, cap_h)))
        cap_note = (" A %s h time cap was active, which is what ended it early."
                    % cap_h)
    outcome, oev = _run_outcome(b)
    ev = ev + oev
    detail = ("the recipe asked for %d epochs and %d ran (%.2f of it, under the "
              "%.2f floor), so this run is not the recipe it names.%s"
              % (req, done, frac, floor, cap_note))
    if outcome == "failed":
        return _ok(name, detail + " The run did not report success, so the "
                                  "shortfall belongs to that failure and is "
                                  "reported by the signal that covers it.")
    if outcome == "success":
        return _sig(name, "warn", frac,
                    detail + " The run reported success, so nothing else "
                             "distinguishes it from a full one.", ev)
    return _sig(name, "info", frac,
                detail + " The bundle does not say whether the run succeeded, so "
                         "this is reported without claiming it completed.", ev)


def _check_pool_growth(b):
    name = "pool_growth"
    trigger = b.th_num("pool_growth.growth_fraction")
    floor = b.th_num("pool_growth.min_previous_iterations")
    now, prev, unit, ev, note = _pool_pair(b)
    if now is None or prev is None:
        return _unknown(name, note)
    if prev < floor:
        return _unknown(name, "the previous train ran %d %s, below the %d floor: "
                              "a probe run is not a round, and dividing by it "
                              "would report every real round as growth"
                              % (int(prev), unit, int(floor)))
    growth = _round((now - prev) / prev, 4)
    if growth <= trigger:
        return _ok(name, "%d %s against %d last round (%+.1f%%, at or under the "
                         "%.0f%% trigger)%s"
                         % (int(now), unit, int(prev), growth * 100.0,
                            trigger * 100.0, note))
    return _sig(name, "warn", growth,
                "%d %s against %d last round: %+.1f%%, over the %.0f%% trigger. "
                "The epoch budget was chosen for the smaller pool, so the same "
                "recipe now needs %.2fx the time it did%s"
                % (int(now), unit, int(prev), growth * 100.0, trigger * 100.0,
                   now / prev, note), ev)


def _check_gate_noop(b):
    name = "gate_noop"
    grace = b.th_num("gate_noop.stale_scores_grace_s")
    kept_max = b.th_num("gate_noop.kept_fraction_max")
    unscored_max = b.th_num("gate_noop.max_unscored_in_merge")
    tier, tier_label, _key = "", "", ""
    for label, holder in b.strategy_holders():
        got = _text(_get(holder, "tier", "TIER"))
        if got:
            tier, tier_label = got.strip().lower(), label
            break
    if not tier:
        return _unknown(name, "no tier is recorded for this run, and the gate "
                              "only has a meaning relative to the tier it claims")
    tier_ev = _ev_section(tier_label, 1, "tier=%s" % tier)
    if tier != "curated":
        return _ok(name, "tier is %s; the DINO gate is only claimed by the "
                         "curated tier" % tier)

    scores = b.sec_dict("slug_scores")
    hits, ev, value = [], [tier_ev], None
    if not scores:
        if b.declared_absent("slug_scores"):
            return _sig(name, "warn", 0.0,
                        "the run claims the curated tier and the slug-score "
                        "artifact it filters with is absent (%s), so nothing "
                        "filtered anything" % (b.missing_reason("slug_scores")
                                               or "not readable"),
                        [tier_ev])
        return _unknown(name, "the bundle carries no slug_scores section and no "
                              "export record saying the artifact was absent, so "
                              "'missing' cannot be told from 'not exported'")
    ev.append(_ev_section("slug_scores", 1,
                          _kv(scores, "n", "unscored", "median", "mtime")))

    mtime = _ts(scores.get("mtime"))
    collect_end, collect_ev = _collect_end(b)
    if mtime is not None and collect_end is not None:
        age = _round(collect_end - mtime - grace, 3)
        if age > 0:
            hits.append("the slug scores are %.0f s older than the round's own "
                        "collect step finished, so they cannot describe what it "
                        "collected" % (collect_end - mtime))
            value = age if value is None else value
            ev += collect_ev
    elif mtime is None:
        hits.append("(the slug-score artifact carries no mtime, so its age "
                    "against the collect step was not checked)")
    elif collect_end is None:
        hits.append("(the ledger records no end for this round's collect step, "
                    "so the scores' age was not checked)")

    n = _num(scores.get("n"))
    unscored = _num(scores.get("unscored"))
    scored = None if n is None else n - (unscored or 0.0)
    kept, kept_label, kept_key = _scan(b.strategy_holders(),
                                       ("slugs_kept", "kept", "datasets_kept"))
    if kept is None:
        used = None
        for label, holder in b.strategy_holders():
            got = _get(holder, "datasets_used", "used_datasets", "slugs_used")
            if isinstance(got, (list, tuple)):
                used, kept_label, kept_key = float(len(got)), label, "datasets_used"
                break
        kept = used
    if kept is not None and scored:
        ev.append(_ev_section(kept_label, 1, "%s=%d" % (kept_key, int(kept))))
        ratio = _round(kept / scored, 4)
        if ratio >= kept_max:
            hits.append("the merge kept %d of the %d slugs the gate scored "
                        "(%.2f), so the gate removed nothing"
                        % (int(kept), int(scored), ratio))
            value = ratio if value is None else value
        extra = _round(kept - scored, 3)
        if extra > unscored_max:
            hits.append("%d slug(s) entered the merge that the gate never "
                        "scored" % int(extra))
            value = extra if value is None else value
    firing = [h for h in hits if not h.startswith("(")]
    if not firing:
        note = " ".join(h for h in hits if h.startswith("("))
        if kept is None or not scored:
            return _unknown(name, ("the curated tier's gate cannot be checked: no "
                                   "kept/scored slug counts are recorded. " + note
                                   ).strip())
        return _ok(name, ("the gate scored %d slugs and the merge kept %d. %s"
                          % (int(scored), int(kept), note)).strip())
    return _sig(name, "warn", value, "the run claims the curated tier but " +
                "; ".join(firing), ev)


def _collect_end(b):
    """(epoch seconds, [evidence]) for when this round's collect step finished."""
    best, ev = None, []
    for idx, entry in enumerate(b.step_entries(step="collect")):
        if _text(entry.get("status")).lower() not in _SUCCESS_WORDS:
            continue
        ts = _ts(entry.get("at"))
        if ts is not None and (best is None or ts > best):
            best = ts
            ev = [_ev_section("ledger", idx + 1,
                              _kv(entry, "status", "at", "job", "detail"))]
    return best, ev


def _check_stale_artifact(b):
    name = "stale_artifact"
    grace = b.th_num("stale_artifact.grace_s")
    watch = b.th("stale_artifact.sections")
    if not isinstance(watch, (list, tuple)) or not watch:
        return _unknown(name, "stale_artifact.sections is not a list of section "
                              "names, so there is nothing to check")
    start, source, start_ev = _step_start(b)
    if start is None:
        return _unknown(name, "nothing in this bundle dates the start of the "
                              "step (no sacct Start and no running entry on the "
                              "ledger), so no artifact can be called older "
                              "than it")
    checked, stale, ev = [], [], list(start_ev)
    for section in watch:
        holder = b.sec_dict(section)
        mtime = _ts(holder.get("mtime")) if holder else None
        if mtime is None:
            mtime = _artifact_mtime(b, section)
        if mtime is None:
            continue
        checked.append(section)
        age = _round(start - mtime, 3)
        if age > grace:
            stale.append((age, section, mtime))
    if not checked:
        return _unknown(name, "none of the watched sections (%s) carries an "
                              "mtime" % ", ".join(str(s) for s in watch))
    if not stale:
        return _ok(name, "every dated artifact (%s) was written after the step "
                         "started" % ", ".join(checked))
    stale.sort(reverse=True)
    for age, section, _mtime in stale:
        ev.append(_ev_section(section, 1, "mtime is %.0f s before the step start"
                                          % age))
    worst = stale[0]
    return _sig(name, "warn", worst[0],
                "%s predates the start of the step it is read for by %.0f s "
                "(step start from %s). A metric read out of a file the step "
                "never wrote belongs to some earlier run.%s"
                % (worst[1], worst[0], source,
                   "" if len(stale) == 1 else " Also stale: %s."
                   % ", ".join(s[1] for s in stale[1:])), ev)


def _artifact_mtime(b, section):
    """mtime recorded for a section's artifact in the export manifest."""
    exp = b.bundle.get("export")
    if not isinstance(exp, dict):
        return None
    for entry in (exp.get("artifacts") or []):
        if isinstance(entry, dict) and entry.get("section") == section:
            ts = _ts(entry.get("mtime"))
            if ts is not None:
                return ts
    return None


def _check_plateau(b):
    name = "plateau"
    multiple = b.th_num("plateau.noise_multiple")
    window = int(b.th_num("plateau.window"))
    floors = b.config_block("noise_floor")
    if not floors:
        return _unknown(name, "the bundle carries no domain config with a "
                              "noise_floor block; the sealed seed std is a "
                              "per-recipe measurement and is never assumed")
    recipe, recipe_ev = _recipe(b)
    if not recipe:
        return _unknown(name, "no recipe is recorded for this run, so which "
                              "sealed seed std applies is unknown (have: %s)"
                              % ", ".join(sorted(floors)))
    floor = _num(floors.get(recipe))
    if floor is None:
        return _unknown(name, "the noise_floor block carries no entry for recipe "
                              "%s (have: %s)" % (recipe, ", ".join(sorted(floors))))
    series, key, ev = _round_metrics(b, window)
    if len(series) < window:
        return _unknown(name, "only %d of the %d round metrics the window needs "
                              "are on the staged ledger" % (len(series), window))
    values = [v for _r, v in series[-window:]]
    span = _round(max(values) - min(values), 9)
    limit = _round(multiple * floor, 9)
    ev = ev + recipe_ev
    shown = ", ".join("round %s %.4f" % (r, v) for r, v in series[-window:])
    if span >= limit:
        return _ok(name, "the last %d %s values span %.4f, at or above %gx the "
                         "%s seed std (%.4f): %s"
                         % (window, key, span, multiple, recipe, limit, shown))
    return _sig(name, "warn", span,
                "the last %d %s values span %.4f, below %gx the sealed seed std "
                "of the %s recipe (%.4f): %s. Nothing measurable is moving, so "
                "another round of the same recipe buys nothing"
                % (window, key, span, multiple, recipe, limit, shown), ev)


def _recipe(b):
    """(recipe id, [evidence]) — which sealed seed std applies to this run."""
    for label, holder in b.strategy_holders():
        got = _text(_get(holder, "recipe", "recipe_id"))
        if got:
            return got.strip(), [_ev_section(label, 1, "recipe=%s" % got.strip())]
    for label, holder in b.strategy_holders():
        tier = _text(_get(holder, "tier", "TIER")).strip().lower()
        if tier:
            return ("merged_%s" % tier,
                    [_ev_section(label, 1, "tier=%s" % tier)])
    return "", []


def _round_metrics(b, window):
    """([(round, value)], metric key, [evidence]) from the staged ledger."""
    key = b.config_str("target_metric")
    docs = [d for d in b.rounds() if isinstance(d.get("metrics"), dict)]
    if not key:
        names = []
        for doc in docs:
            names += [k for k in doc["metrics"] if _num(doc["metrics"][k]) is not None]
        preferred = [n for n in names
                     if re.sub(r"[^a-z0-9]", "", n.lower()) == "map5095"]
        if preferred:
            key = preferred[0]
        elif len(set(names)) == 1:
            key = names[0]
    if not key:
        return [], "", []
    series, ev = [], []
    for idx, doc in enumerate(docs):
        value = _num(doc["metrics"].get(key))
        if value is None:
            continue
        rnd = _int(_get(doc, "round_num", "round"), idx + 1)
        series.append((rnd, value))
        ev.append(_ev_section("ledger", idx + 1,
                              "round=%s %s=%s" % (rnd, key, value)))
    return series, key, ev[-window:]


def _check_source_degraded(b):
    name = "source_degraded"
    floor = b.th_num("source_degraded.min_previous_candidates")
    now, now_ev = _per_source(b.sec_dict("harvest"), "harvest")
    if now is None:
        return _unknown(name, "the bundle carries no harvest section with a "
                              "per_source block, so no source can be compared "
                              "against itself")
    history = _source_history(b)
    zero = sorted(s for s, n in now.items() if n is not None and n <= 0)
    if not zero:
        return _ok(name, "every source in the per-source report returned "
                         "candidates (%d source(s))" % len(now))
    if not history:
        ev = [_ev_section("harvest", 1, "per_source %s=0" % zero[0])] + \
             _source_lines(b, zero)
        return _sig(name, "info", float(len(zero)),
                    "%s returned nothing this round, and the bundle carries no "
                    "previous round to compare against — a zero with no history "
                    "is not evidence of a new degradation"
                    % ", ".join(zero), ev)
    fresh, chronic = [], []
    for src in zero:
        past = [(label, counts.get(src)) for label, counts, _ev in history
                if counts.get(src) is not None]
        if any(n >= floor for _label, n in past):
            best = [(label, n) for label, n in past if n >= floor][0]
            fresh.append((src, best[0], best[1]))
        else:
            chronic.append((src, len(past)))
    ev = _source_lines(b, [s for s, _l, _n in fresh] or [s for s, _n in chronic])
    ev.append(_ev_section("harvest", 1, "per_source " + " ".join(
        "%s=%s" % (s, now.get(s)) for s in zero)))
    for _label, _counts, hev in history[:1]:
        ev += hev
    if fresh:
        return _sig(name, "warn", float(len(fresh)),
                    "; ".join("%s returned %d candidate(s) in %s and none now"
                              % (src, int(n), label) for src, label, n in fresh)
                    + ". A source that yielded and stopped is a change, and the "
                      "harvest that follows it is smaller for a reason nothing "
                      "else records", ev)
    return _sig(name, "info", float(len(chronic)),
                "; ".join("%s returned nothing this round and nothing in the %d "
                          "previous round(s) in this bundle" % (src, n)
                          for src, n in chronic)
                + ". A condition present in every round is an environment fact, "
                  "not a new degradation: reported so it is on the record, and "
                  "kept below warn so it cannot be escalated as an incident", ev)


def _per_source(holder, label):
    """({source: count}, [evidence]) from a harvest-shaped block, or (None, [])."""
    if not isinstance(holder, dict):
        return None, []
    block = None
    for key in ("per_source", "sources", "by_source"):
        if isinstance(holder.get(key), dict):
            block = holder[key]
            break
    if block is None:
        return None, []
    out = {}
    for src, value in block.items():
        count = _num(value)
        if count is None and isinstance(value, dict):
            count = _num(_get(value, "candidates", "count", "n", "kept",
                              "downloaded", "results"))
        out[str(src)] = count
    return out, [_ev_section(label, 1, "per_source " + " ".join(
        "%s=%s" % (k, out[k]) for k in sorted(out)))]


def _source_history(b):
    """[(label, {source: count}, [evidence])] for earlier rounds, newest first."""
    out = []
    harvest = b.sec_dict("harvest")
    for key in ("previous_per_source", "per_source_previous", "prev_per_source"):
        if isinstance(harvest.get(key), dict):
            counts, _ev = _per_source({"per_source": harvest[key]}, "harvest")
            out.append(("the previous round", counts,
                        [_ev_section("harvest", 1, "%s %s" % (key, " ".join(
                            "%s=%s" % (k, counts[k]) for k in sorted(counts))))]))
            break
    for doc in b.previous_round_docs():
        rnd = _int(_get(doc, "round_num", "round"))
        for idx, entry in enumerate(b.step_entries(doc, "collect")):
            for holder in _dicts(entry, entry.get("metrics"), entry.get("params")):
                counts, _ev = _per_source(holder, "ledger")
                if counts:
                    out.append(("round %s" % rnd, counts,
                                [_ev_section("ledger", idx + 1,
                                             "round=%s collect per_source %s"
                                             % (rnd, " ".join(
                                                 "%s=%s" % (k, counts[k])
                                                 for k in sorted(counts))))]))
                    break
    return out


def _source_lines(b, sources):
    """Log lines that name a source and say it was skipped or refused."""
    ev = []
    for src in sources[:3]:
        rows = b.grep(r"(SKIP\w*|403|401|404|no candidates|WARN).{0,80}%s|"
                      r"%s.{0,80}(SKIP\w*|403|401|404|no candidates)"
                      % (re.escape(src), re.escape(src)), last=False, limit=1)
        ev += [_ev_line(r) for r in rows]
    return ev


def _check_job_unknown(b):
    name = "job_unknown"
    limit = b.th_num("job_unknown.max_consecutive_polls")
    polls, ev = None, []
    for label, holder in b.strategy_holders() + [
            ("ledger", b.sec_dict("ledger"))]:
        got = _num(_get(holder, "unknown_polls", "state_unknown_polls",
                        "job_unknown_polls"))
        if got is not None:
            polls, ev = got, [_ev_section(label, 1, "unknown_polls=%d" % int(got))]
            break
    if polls is None:
        run = 0
        entries = b.step_entries()
        for idx, entry in enumerate(entries):
            text = " ".join([_text(entry.get("status")), _text(entry.get("detail")),
                             _text(entry.get("state"))]).upper()
            if "UNKNOWN" in text:
                run += 1
                ev.append(_ev_section("ledger", idx + 1,
                                      _kv(entry, "status", "at", "job", "detail")))
            else:
                run, ev = 0, []
        if entries:
            polls = float(run)
    if polls is None:
        return _unknown(name, "nothing in this bundle counts polls of the job's "
                              "state: no unknown-poll counter and no ledger "
                              "attempts for this step")
    if polls <= limit:
        return _ok(name, "the job's state has been UNKNOWN for %d consecutive "
                         "poll(s), at or under the limit of %d"
                         % (int(polls), int(limit)))
    return _sig(name, "warn", polls,
                "the job's state has been UNKNOWN for %d consecutive polls, over "
                "the limit of %d. An UNKNOWN job is neither running nor finished "
                "as far as the loop is concerned, and a step that stays there "
                "blocks its domain behind a page that still looks healthy"
                % (int(polls), int(limit)), ev)


def _check_budget(b):
    name = "budget"
    warn_frac = b.th_num("budget.campaign_fraction_warn")
    crit_frac = b.th_num("budget.campaign_fraction_crit")
    su = b.sec_dict("su")
    if not su:
        return _unknown(name, "the bundle carries no su section, so neither the "
                              "round's cost nor the campaign's is known")
    budget_cfg = b.config_block("budget")
    round_su = _num(_get(su, "round", "round_su"))
    campaign = _num(_get(su, "campaign", "campaign_su", "spent"))
    envelope = _num(_get(su, "envelope", "su_envelope")) or _num(
        _get(budget_cfg, "su_envelope", "envelope"))
    cap = _num(_get(su, "per_round_cap", "round_cap")) or _num(
        _get(budget_cfg, "per_round_cap"))
    ev = [_ev_section("su", 1, _kv(su, "round", "campaign", "envelope",
                                   "per_round_cap"))]
    if budget_cfg:
        ev.append(_ev_section("ledger", 1, "budget " + _kv(
            budget_cfg, "su_envelope", "daily_cap", "per_round_cap")))
    hits, value, severity = [], None, "ok"
    if round_su is not None and cap:
        ratio = _round(round_su / cap, 4)
        if round_su > cap:
            hits.append("this round has spent %.1f SU against a per-round cap of "
                        "%.1f (%.2fx)" % (round_su, cap, ratio))
            value, severity = ratio, "warn"
    elif round_su is None:
        hits.append("(the su section records no round cost)")
    elif not cap:
        hits.append("(no per-round cap is configured)")
    if campaign is not None and envelope:
        ratio = _round(campaign / envelope, 4)
        if ratio >= crit_frac:
            hits.append("the campaign has spent %.0f of its %.0f SU envelope "
                        "(%.0f%%)" % (campaign, envelope, ratio * 100.0))
            value, severity = ratio, "crit"
        elif ratio >= warn_frac:
            hits.append("the campaign has spent %.0f of its %.0f SU envelope "
                        "(%.0f%%, past the %.0f%% mark)"
                        % (campaign, envelope, ratio * 100.0, warn_frac * 100.0))
            if severity != "crit":
                value, severity = ratio, "warn"
    elif campaign is None or not envelope:
        hits.append("(the campaign total or its envelope is not recorded)")
    firing = [h for h in hits if not h.startswith("(")]
    if not firing:
        notes = " ".join(h for h in hits if h.startswith("("))
        if len(notes.split("(")) > 2:
            return _unknown(name, ("neither budget comparison could be made: "
                                   + notes).strip())
        return _ok(name, ("round and campaign spend are inside their limits. "
                          + notes).strip())
    return _sig(name, severity, value, "; ".join(firing) +
                ". SU spent past a cap is spent whatever the next round decides",
                ev)


def _check_disk_low(b):
    name = "disk_low"
    warn_gb = b.th_num("disk_low.quota_headroom_warn_gb")
    crit_gb = b.th_num("disk_low.quota_headroom_crit_gb")
    fs_warn_tb = b.th_num("disk_low.fs_free_warn_tb")
    res = b.sec_dict("resources")
    if not res:
        return _unknown(name, "the bundle carries no resources section")
    quota = _num(_get(res, "quota_headroom_gb", "quota_headroom", "headroom_gb"))
    fs_free = _num(_get(res, "fs_free_tb", "filesystem_free_tb", "fs_free"))
    ev = [_ev_section("resources", 1, _kv(res, "df_project", "quota_headroom_gb",
                                          "fs_free_tb", "squeue_depth"))]
    fs_note = ("" if fs_free is None else
               " The whole filesystem reports %.2f TB free; that is a different "
               "number and is not the binding one — the project quota is what "
               "every job in this allocation shares." % fs_free)
    if quota is None:
        if fs_free is not None and fs_free < fs_warn_tb:
            return _sig(name, "info", fs_free,
                        "no project-quota headroom is recorded, so the binding "
                        "number is unknown." + fs_note, ev)
        return _unknown(name, "the resources section records no project-quota "
                              "headroom, and the filesystem figure is a "
                              "different number that cannot stand in for it")
    if quota < crit_gb:
        return _sig(name, "crit", quota,
                    "project-quota headroom is %.0f GB, under the %.0f GB "
                    "floor. Exhausting the quota breaks every job in the "
                    "allocation, not only this one.%s" % (quota, crit_gb, fs_note),
                    ev)
    if quota < warn_gb:
        return _sig(name, "warn", quota,
                    "project-quota headroom is %.0f GB, under the %.0f GB "
                    "trigger.%s" % (quota, warn_gb, fs_note), ev)
    if fs_free is not None and fs_free < fs_warn_tb:
        return _sig(name, "info", fs_free,
                    "project-quota headroom is %.0f GB, which is fine.%s"
                    % (quota, fs_note), ev)
    return _ok(name, "project-quota headroom is %.0f GB, above the %.0f GB "
                     "trigger.%s" % (quota, warn_gb, fs_note))


def _check_ownership_violation(b):
    name = "ownership_violation"
    limit = b.th_num("ownership_violation.max_divergences")
    corr = b.sec("corrections")
    holders = _dicts(corr if isinstance(corr, dict) else None,
                     b.sec_dict("registry_diff"))
    if isinstance(corr, list):
        holders += _dicts(*[c for c in corr if isinstance(c, dict)])
    if not holders:
        return _unknown(name, "the bundle carries no corrections section and no "
                              "registry diff, so the staged mirror has nothing "
                              "to be compared against")
    hits, ev, checked = [], [], False
    for holder in holders:
        mirror = _text(_get(holder, "mirror_sha256", "mirror_sha", "mirror_hash"))
        ledger = _text(_get(holder, "ledger_sha256", "ledger_sha", "ledger_hash"))
        if not mirror and isinstance(holder.get("mirror"), dict):
            mirror = _text(_get(holder["mirror"], "sha256", "sha"))
        if not ledger and isinstance(holder.get("ledger"), dict):
            ledger = _text(_get(holder["ledger"], "sha256", "sha"))
        if mirror and ledger:
            checked = True
            if mirror != ledger:
                hits.append("the staged corrections mirror hashes to %s and the "
                            "ledger copy to %s" % (mirror[:12], ledger[:12]))
                ev.append(_ev_section("corrections", 1,
                                      "mirror_sha256=%s ledger_sha256=%s"
                                      % (mirror, ledger)))
        chain = holder.get("chain_ok")
        if isinstance(chain, bool):
            checked = True
            if not chain:
                hits.append("the corrections chain does not verify")
                ev.append(_ev_section("corrections", 1, "chain_ok=false"))
        reverts = _get(holder, "supervisory_reverts", "reverted_fields")
        if isinstance(reverts, (list, tuple)):
            checked = True
            for idx, item in enumerate(reverts):
                label = item if isinstance(item, str) else json.dumps(
                    item, sort_keys=True, default=str)
                hits.append("a supervisory field was reverted (%s)" % label[:120])
                ev.append(_ev_section("registry_diff", idx + 1, label[:200]))
    if not checked:
        return _unknown(name, "no mirror hash, chain flag or supervisory-revert "
                              "list is recorded, so ownership cannot be checked")
    if len(hits) <= limit:
        return _ok(name, "the staged mirror matches the ledger copy")
    return _sig(name, "crit", float(len(hits)),
                "; ".join(hits) + ". All jobs run under one account, so this is "
                "detection, not prevention: the mirror is read-only by "
                "convention and a divergence means the convention was broken", ev)


def _check_mongo_down(b):
    name = "mongo_down"
    led = b.sec_dict("ledger")
    holders = _dicts(led, led.get("heartbeat"), led.get("health"),
                     led.get("scheduler"), b.bundle.get("heartbeat"))
    for holder in holders:
        for key in ("mongo_ok", "ledger_ok", "ledger_writes_ok"):
            value = holder.get(key)
            if isinstance(value, bool):
                if value:
                    return _ok(name, "the heartbeat reports ledger writes are "
                                     "landing (%s=true)" % key)
                return _sig(name, "crit", 1.0,
                            "the heartbeat reports the round-ledger writes are "
                            "failing (%s=false)%s. The loop keeps ticking and "
                            "keeps spending, and nothing it does is recorded — "
                            "the same blind spot as a stopped loop, reached "
                            "while jobs are still burning SU"
                            % (key, _mongo_since(holder)),
                            [_ev_section("ledger", 1,
                                         _kv(holder, key, "mongo_last_error_ts",
                                             "ts"))])
        for key in ("mongo_down", "ledger_write_failing"):
            value = holder.get(key)
            if isinstance(value, bool):
                if not value:
                    return _ok(name, "the heartbeat reports ledger writes are "
                                     "landing (%s=false)" % key)
                return _sig(name, "crit", 1.0,
                            "the heartbeat reports the round-ledger writes are "
                            "failing (%s=true)%s" % (key, _mongo_since(holder)),
                            [_ev_section("ledger", 1, _kv(holder, key,
                                                          "mongo_last_error_ts"))])
    return _unknown(name, "no heartbeat in this bundle says whether the ledger "
                          "writes are landing; absent is unknown, not "
                          "healthy-and-not-checked")


def _mongo_since(holder):
    since = _ts(_get(holder, "mongo_last_error_ts", "last_error_ts"))
    now = _ts(_get(holder, "ts", "checked_ts"))
    if since is None or now is None or now < since:
        return ""
    return ", failing for %.0f s" % (now - since)


_CHECKS = {
    "walltime_bound": _check_walltime_bound,
    "epochs_truncated": _check_epochs_truncated,
    "pool_growth": _check_pool_growth,
    "gate_noop": _check_gate_noop,
    "stale_artifact": _check_stale_artifact,
    "plateau": _check_plateau,
    "source_degraded": _check_source_degraded,
    "job_unknown": _check_job_unknown,
    "budget": _check_budget,
    "disk_low": _check_disk_low,
    "ownership_violation": _check_ownership_violation,
    "mongo_down": _check_mongo_down,
}


# --- rationale --------------------------------------------------------------
# One paragraph per signal: what it means, what it reads, and the failure it
# comes from. Rendered into the evidence bundle and into the engineering record,
# so a reviewer never has to read this file to know what a name means.
_EXPLAIN = {
    "walltime_bound": (
        "The run cannot finish inside the walltime it was given. It fires on "
        "sacct State TIMEOUT for the step's own job, and — from the third epoch "
        "onward, which is the point — on a projection: the epoch trace (or the "
        "results.csv time column) times the epochs the recipe asked for, "
        "against the job's Timelimit. On 2026-08-29 jobs 44727703 and 44767709 "
        "were both 12 h jobs whose 60-epoch recipe needed about 29 h; both were "
        "killed at the wall with no checkpoint and no metric, the second "
        "because the identical command was resubmitted. A run under an "
        "Ultralytics time= cap is excluded from the projection branch: it ends "
        "early on purpose and leaves a valid best.pt, which is epochs_truncated, "
        "not this."),
    "epochs_truncated": (
        "A run that reported success ran fewer epochs than its recipe asked "
        "for. This signal is newer than the incident corpus and exists because "
        "the v3.25.0 time cap created the failure mode it covers: a capped run "
        "stops on time, writes a valid checkpoint and reports COMPLETED, so a "
        "60-epoch recipe that ran 24 is indistinguishable from one that ran in "
        "full unless something compares epochs_requested against "
        "epochs_completed. Both are recorded side by side in the job-scoped "
        "artifact and in the trace end record for exactly this comparison. A "
        "run that did NOT report success is left to the signal that covers its "
        "failure, so the same shortfall is never counted twice."),
    "pool_growth": (
        "The training pool grew enough since the previous train that the "
        "recipe's epoch budget no longer fits its walltime. It compares this "
        "run's iterations per epoch against the previous round's — from the "
        "strategy artifact, the log's own progress line, or the previous "
        "round's train entry on the staged ledger — and fires past the "
        "pre-registered growth fraction. This is the leading indicator of "
        "walltime_bound: the merged pool went from 6,246 to 8,583 iterations "
        "per epoch between rounds, which is what made a 12 h job impossible "
        "before it was ever submitted. A previous value below the floor is "
        "refused rather than divided by: a one-epoch probe is not a round."),
    "gate_noop": (
        "The run claims the curated tier while the gate that defines that tier "
        "did nothing. Four ways, all read from artifacts: the slug-score "
        "artifact is absent; it is older than the round's own collect step "
        "finished, so it cannot describe what was collected; the merge kept "
        "every slug the gate scored; or more slugs entered the merge than were "
        "ever scored. The measured value of the DINO gate is that it removes "
        "garbage, not that it raises accuracy (curated 0.5894 against raw "
        "0.6032), so a gate that silently passes everything changes no metric "
        "and mislabels the tier of every number the round produces."),
    "stale_artifact": (
        "A metric was read from a file older than the step that supposedly "
        "produced it. The step start is taken from sacct Start where possible "
        "(the same clock as the artifact mtimes) and from the ledger's running "
        "entry otherwise, with a grace that absorbs the skew between the lab "
        "and cluster clocks. This is the shape of several of the campaign's "
        "worst hours: a results.csv from an earlier run attached to a new "
        "round, a summary quoting a number no artifact on disk supported, six "
        "hours of successful training reported as a missing file."),
    "plateau": (
        "The last few round metrics are inside the noise of the recipe that "
        "produced them, so another round of the same recipe buys nothing "
        "measurable. The spread of the last N round metrics is compared with a "
        "multiple of that recipe's own sealed seed standard deviation, read "
        "from the domain config's noise_floor block (merged_curated 0.005, "
        "merged_raw 0.009, cwd12_core 0.006) and never assumed: a noise floor "
        "is a measurement of a specific recipe, and guessing one is how noise "
        "gets reported as an effect. The scheduler's own three scripted rounds "
        "(0.6019 / 0.5919 / 0.5951) sit on that boundary."),
    "source_degraded": (
        "A harvest source that returned candidates in a previous round returns "
        "none now. NEW degradation only. A source that has been at zero for "
        "every round in the bundle is an environment fact and is reported at "
        "info: the line '[net] WARN: SOCKS proxy ... SKIPPING github, using "
        "Kaggle/HF only' has been in every harvest log since June because "
        "GitHub is unreachable from the compute nodes and the collector is "
        "designed to skip it. It sits inside the 2026-08-29 bundle, so an arm "
        "that escalates it produces a false alarm on the very case it is meant "
        "to solve. A zero with no history at all is reported at info too — a "
        "zero is only evidence of a change when there is something to change "
        "from."),
    "job_unknown": (
        "The job's state has been UNKNOWN for more consecutive polls than the "
        "limit. An UNKNOWN job is neither running nor finished as far as the "
        "loop is concerned: it holds its step open, blocks the domain, and does "
        "it behind a page that still looks healthy. The record carries both "
        "directions of this — a COMPLETED harvest booked as failed across an "
        "UNKNOWN gap, and submitted jobs with no result row anywhere after "
        "their walltime had elapsed."),
    "budget": (
        "The round's service units are past the per-round cap, or the campaign "
        "is past its envelope. Read from the su section, with the caps from the "
        "domain config's budget block. SU spent is spent whatever the next "
        "round decides, and the two 12 h H100 jobs of 2026-08-29 bought nothing "
        "at all — this is the signal that makes that visible while a round is "
        "still deciding what to submit."),
    "disk_low": (
        "The project-quota headroom is under its floor. Quota headroom and "
        "whole-filesystem free space are different numbers and are reported "
        "separately: the quota is the binding one, it is shared with every job "
        "in the allocation, and exhausting it breaks all of them rather than "
        "only the job that filled it. The filesystem figure is reported at info "
        "and can never raise this signal on its own."),
    "ownership_violation": (
        "The staged corrections mirror does not hash to the ledger copy, its "
        "chain does not verify, or a supervisory field was reverted by a later "
        "writer. Every job runs under one account, so this is detection and "
        "reversion, never prevention — the campaign's two registry incidents "
        "were exactly this: a collector's stale snapshot erasing a quarantine, "
        "and a harvester rolling back corrected audit verdicts."),
    "mongo_down": (
        "The scheduler's heartbeat says its round-ledger writes are failing. "
        "The loop keeps ticking and keeps submitting, and nothing it does is "
        "recorded: the same blind spot as a stopped loop, reached while jobs "
        "are still burning SU. Absent is unknown, not healthy — a pre-v3.25.0 "
        "heartbeat carries no such field, and reading its silence as health is "
        "how a mirror failure stayed invisible once already."),
}


def explain(name):
    """The one-paragraph rationale for one signal. Never raises."""
    key = str(name or "").strip()
    if key in _EXPLAIN:
        return _EXPLAIN[key]
    return ("%r is not one of the signals this module computes (%s)"
            % (key, ", ".join(SIGNALS)))


def explain_all():
    """{signal: rationale} for every signal, in SIGNALS order."""
    return [{"signal": n, "explain": _EXPLAIN[n]} for n in SIGNALS]


# --- entry points -----------------------------------------------------------
def _run_check(name, b):
    """One check, with every failure mode turned into a reported state."""
    try:
        sig = _CHECKS[name](b)
    except _MissingThreshold as exc:
        return _unknown(name, "threshold %s is not declared in %s, so this check "
                              "cannot run; thresholds live in that file and "
                              "nowhere else, so the fix is to add it there"
                              % (exc, os.path.basename(str(_threshold_path()))))
    except Exception as exc:                  # a bundle is untrusted input
        return _unknown(name, "the check raised %s (%s), which is a defect in "
                              "the check or a bundle shape it does not know"
                              % (type(exc).__name__, str(exc)[:160]))
    if not isinstance(sig, dict) or sig.get("signal") != name:
        return _unknown(name, "the check returned something that is not its own "
                              "signal object")
    severity = sig.get("severity")
    if severity not in SEVERITIES:
        return _unknown(name, "the check returned severity %r, which is not one "
                              "of %s" % (severity, ", ".join(SEVERITIES)))
    if _SEV_RANK[severity] >= _SEV_RANK["info"] and not sig.get("evidence"):
        # Structural, not stylistic: an unaddressed finding cannot be checked by
        # anyone, and a signal nobody can check is worth less than silence.
        return _unknown(name, "the check fired at %s but produced no evidence; a "
                              "signal with no artifact address is not actionable "
                              "and is reported as unknown instead. Reason it "
                              "gave: %s" % (severity, sig.get("reason")))
    return sig


def detect_all(bundle, thresholds=None):
    """Every signal, including the ones that ran and found nothing (`ok`).

    Use this to show that a check ran: `detect()` is deliberately quiet about a
    clean result, and "checked and clean" is a different statement from "not
    checked", which is the distinction the whole module is built around.
    """
    b = _Bundle(bundle, thresholds)
    return [_run_check(name, b) for name in SIGNALS]


def detect(bundle, thresholds=None):
    """The signals for one evidence bundle: fired ones, plus the unknowns.

    A check that ran and found nothing is omitted; a check that could not run is
    reported with `severity: "unknown"`, a reason and no evidence, because a
    detector that switches itself off silently is the failure this campaign
    exists to close. Nothing raises: a bundle is untrusted input, and this runs
    inside a scheduler tick.
    """
    return [s for s in detect_all(bundle, thresholds) if s["severity"] != "ok"]


def summary(bundle, thresholds=None):
    """Signals plus the rules they were computed under, for a results file."""
    b = _Bundle(bundle, thresholds)
    rows = [_run_check(name, b) for name in SIGNALS]
    counts = {}
    for row in rows:
        counts[row["severity"]] = counts.get(row["severity"], 0) + 1
    return {"tool_version": TOOL_VERSION,
            "bundle_id": b.bundle.get("bundle_id"),
            "signals": [r for r in rows if r["severity"] != "ok"],
            "checked": [r["signal"] for r in rows],
            "counts": counts,
            "thresholds": b.threshold_report}


# --- CLI --------------------------------------------------------------------
def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv=None):
    """`detect <bundle.json>` | `thresholds` | `explain [<signal>]`.

    Exit 1 when a signal fired at or above `warn`, so a shell step can gate on
    it; exit 0 for a clean bundle, for unknowns and for the other commands.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="signals", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    det = sub.add_parser("detect", help="signals for a bundle JSON file")
    det.add_argument("bundle")
    det.add_argument("--all", action="store_true",
                     help="include checks that ran and found nothing")
    sub.add_parser("thresholds", help="effective thresholds and their sources")
    exp = sub.add_parser("explain", help="the rationale for one signal, or all")
    exp.add_argument("signal", nargs="?")
    args = ap.parse_args(argv)

    if args.cmd == "thresholds":
        _dump(thresholds())
        return 0
    if args.cmd == "explain":
        _dump(explain(args.signal) if args.signal else explain_all())
        return 0
    if args.cmd != "detect":
        ap.print_help()
        return 0
    try:
        with open(args.bundle, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
    except Exception as exc:
        print("cannot read %s: %s" % (args.bundle, exc))
        return 2
    rows = detect_all(bundle) if args.all else detect(bundle)
    _dump(rows)
    return 1 if any(_SEV_RANK[r["severity"]] >= _SEV_RANK["warn"]
                    for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
