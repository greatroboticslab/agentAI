#!/usr/bin/env python3
"""Service-unit ledger and sacct reconciliation for the supervision layer (WP5).

Why this exists
----------------
The WP5 gate line is concrete: "SU ledger for the last 20 jobs matches sacct
within 1%." That requires three things this module is the whole of: an
append-only record of what the scheduler believes each job cost, a rate table
that turns a GPU allocation into a service-unit number without inventing one,
and a reconciliation that compares the two on a job-by-job basis rather than
trusting a total that can be right by accident. A ledger whose total matches
sacct because a job that ran too hot cancels one that ran too cheap is not
a ledger anyone can trust to gate a submission, so `reconcile()` reports a
per-job breakdown and fails the whole check on either job even when the
aggregate looks clean.

Two integrity rules are load-bearing and are enforced structurally, not by
convention at each call site:

* **Unknown is never 0.** A job whose `Elapsed` is unresolvable -- the literal
  `Unknown` sacct prints for a job that has not finished accounting, a blank
  field, or anything else that does not parse as a duration -- contributes
  `unknown` service units. Reporting 0 would say the job was free; reporting
  `unknown` says the number has not been measured yet, which is the truth.
  `total()` excludes an unknown job from its sum and lists it separately so
  the gap is visible in the same place the total is.
* **An unresolved GPU family is never free compute.** `su_for()` looks a GPU
  type up in `su_rates.json`; a type that matches neither `h100` nor `v100`
  (a new node class, a typo in `AllocTRES`) is charged at the higher of the
  two known rates rather than 0, and the result carries `unknown_rate: true`
  so the number is visibly a stand-in and not a measurement. `reconcile()`
  surfaces every such job by id under `unknown_rate_jobs` regardless of
  whether its number happens to land inside tolerance.

Every threshold this module decides with -- the two per-GPU-hour rates, the
fallback rate for an unresolved family, and the reconciliation tolerance --
lives in `su_rates.json` next to the reason it has the value it has, in the
same style as `thresholds.json` next to `signals.py`. There is no compiled
fallback for a rate the file does not declare: a missing key makes `su_for()`
report a gap by name, exactly as a missing threshold makes a `signals.py`
check report `unknown`.

sacct parsing follows the same shape `corpus.py`/`evidence.py` use -- a header
row identified by its first column being `JobID`/`JobIDRaw`, `-P` (pipe, tab
or comma) delimited or fixed-width under a `-`-ruler line -- reimplemented
here rather than imported so this module has no hard dependency on either of
those modules' internals while they are under concurrent revision. `.batch`,
`.extern` and numbered step rows (`<jobid>.0`, `<jobid>.1`, ...) are the same
allocation as their parent and are folded into it rather than counted again;
an array task's own sub-steps (`<jobid>_<task>.batch`) fold into that task's
row, not into a different task's. A job's SU is always read from its parent
allocation row (the one whose id carries no step suffix), which is where
sacct puts the job's own `AllocTRES` and `Elapsed`; the sub-step rows exist
in `sacct` only to account for the batch script and anything run outside an
`srun` step, and their own `Elapsed`/`AllocTRES` are not summed in.

Storage is one append-only JSONL per domain
(`results/framework/_brain/<domain>/su_ledger.jsonl`, matching the convention
`evidence.py`'s `stage_commands()` and the WAL use for other brain state).
"Append-only" and "idempotent" are not in tension: recording the same
`(job, step)` twice never deletes the first line -- the audit trail of what
was believed and when stays on disk -- but every read (`total`, `by_actor`,
`by_step`, `reconcile`) folds a key down to its last-written entry, so a job
recorded once with an estimated elapsed time and again after it finishes is
billed once, at the newer number, and `record()`'s own return value says
whether that call updated an existing key so the update is visible to the
caller at the moment it happens.

Pure stdlib, no network, no imports of sibling brain modules: this runs
inside a SLURM job body, in the round-scheduler thread and standalone from a
shell, and it must not take on a dependency that only some of those callers
can satisfy.

CLI (each prints JSON; `reconcile` exits 1 when `ok` is false):
    python3 -m weed_optimizer_framework.tools.brain.su_ledger record '<entry-json>'
    python3 -m weed_optimizer_framework.tools.brain.su_ledger total <domain> [--since ISO|EPOCH]
    python3 -m weed_optimizer_framework.tools.brain.su_ledger reconcile <domain> --sacct-file <path> [--tolerance PCT]
    python3 -m weed_optimizer_framework.tools.brain.su_ledger rates
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import sys

TOOL_VERSION = "wp5-su-ledger/1"

# --- rates.json ---------------------------------------------------------
_RATES_FILE = pathlib.Path(__file__).resolve().parent / "su_rates.json"
_RATES_CACHE = {"key": None, "block": {}, "error": None, "path": ""}


def _rates_path():
    return pathlib.Path(os.environ.get("BRAIN_SU_RATES") or _RATES_FILE)


def _flatten(raw, prefix=""):
    """{"a": {"b": {"value": 1, "why": "..."}}} -> {"a.b": (1, "...")}.

    Mirrors signals.py's threshold flattening so the two pre-registered-value
    files read the same way to anyone auditing both. A bare scalar is accepted
    with an empty reason so a caller's override dict can be flat.
    """
    out = {}
    for name in sorted(raw.keys()):
        if str(name).startswith("_"):
            continue                      # _meta notes, not a rate
        val = raw[name]
        key = ("%s.%s" % (prefix, name)) if prefix else str(name)
        if isinstance(val, dict) and "value" in val:
            out[key] = (val.get("value"), str(val.get("why") or ""))
        elif isinstance(val, dict):
            out.update(_flatten(val, key))
        else:
            out[key] = (val, "")
    return out


def _file_block():
    """The flattened `rates` + `reconcile` objects of su_rates.json.

    Cached on (path, mtime, size): an operator editing the file takes effect
    on the next call with no restart, and a missing or malformed file is a
    reported state, never an exception.
    """
    path = _rates_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError as exc:
        _RATES_CACHE.update({"key": None, "block": {}, "path": str(path),
                             "error": "su_rates.json unreadable at %s (%s)"
                                      % (path.name, type(exc).__name__)})
        return {}
    if _RATES_CACHE["key"] == key:
        return _RATES_CACHE["block"]
    block, error = {}, None
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        for top in ("rates", "reconcile"):
            raw = (obj or {}).get(top)
            if isinstance(raw, dict):
                block.update(_flatten({top: raw}))
        if not block:
            error = "su_rates.json carries neither a 'rates' nor a 'reconcile' object"
    except Exception as exc:
        error = "su_rates.json unreadable: %s" % type(exc).__name__
    _RATES_CACHE.update({"key": key, "block": block, "error": error,
                         "path": str(path)})
    return block


def rates(overrides=None):
    """Effective SU rates and tolerances, and where each one came from.

    Same shape as signals.thresholds(): `{"values": {...}, "sources": {...},
    "why": {...}, "errors": [...], "file": str}`, so a results file that
    records both can treat them the same way.
    """
    out = {"tool_version": TOOL_VERSION, "values": {}, "sources": {},
           "why": {}, "errors": [], "file": str(_rates_path())}
    block = _file_block()
    if _RATES_CACHE.get("error"):
        out["errors"].append(_RATES_CACHE["error"])
    for key, (value, why) in block.items():
        out["values"][key] = value
        out["sources"][key] = "su_rates.json"
        out["why"][key] = why
        if not why:
            out["errors"].append("%s has no 'why' in su_rates.json" % key)
    if isinstance(overrides, dict):
        for key, (value, why) in _flatten(overrides).items():
            if key not in out["values"]:
                out["errors"].append(
                    "override %s is not declared in su_rates.json; ignored" % key)
                continue
            out["values"][key] = value
            out["sources"][key] = "caller"
            if why:
                out["why"][key] = why
    for key in sorted(out["values"]):
        name = "BRAIN_SU_RATE_" + re.sub(r"[^A-Za-z0-9]", "_", str(key)).upper()
        raw = os.environ.get(name)
        if raw in (None, ""):
            continue
        try:
            value = json.loads(raw)
        except Exception:
            try:
                value = float(raw)
            except ValueError:
                out["errors"].append(
                    "%s is set but is not JSON and not a number; ignored" % name)
                continue
        out["values"][key] = value
        out["sources"][key] = "env:" + name
    return out


# --- small utilities ------------------------------------------------------
def _num(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value, default=None):
    v = _num(value)
    return default if v is None else int(v)


def _domain(value):
    return re.sub(r"[^a-z0-9_]", "", str(value or "").strip().lower())[:40]


def _get_ci(holder, *names):
    """First present, non-None value among `names`, case-insensitive."""
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


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(value):
    """Epoch seconds from a float, a numeric string or ISO-8601, or None."""
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
        txt = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        dt = datetime.datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _hms_to_seconds(text):
    """SLURM duration `[DD-]HH:MM:SS[.us]` to seconds, or None."""
    raw = str(text or "").strip()
    if not raw:
        return None
    days = 0.0
    if "-" in raw:
        head, _, raw = raw.partition("-")
        days = _num(head) or 0.0
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


def _elapsed_value(raw):
    """Seconds as a float, or None when unknown/"Unknown"/empty/unparseable.

    None is the signal, not an exception: a job still accounting, or a field
    sacct printed as literal `Unknown`, is a real and common state, not a
    parse failure to raise on.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text or text.lower() in ("unknown", "none", "n/a"):
        return None
    if ":" in text or "-" in text:
        return _hms_to_seconds(text)
    try:
        return float(text)
    except ValueError:
        return None


# --- GPU family / rate resolution -----------------------------------------
# Same charset and shape as evidence.py's _TRES_GPU_RE: gres/gpu[:<type>]=<n>.
_TRES_GPU_RE = re.compile(r"gres/gpu(?::([a-z0-9\-]+))?=(\d+)", re.I)


def _gpu_from_tres(tres):
    """(gpu_count, gpu_type) from an AllocTRES/ReqTRES string, or (0, None)."""
    m = _TRES_GPU_RE.search(str(tres or ""))
    if not m:
        return 0, None
    return int(m.group(2)), (m.group(1) or None)


def _family_of(gpu_type):
    """"h100" | "v100" | "unknown" by substring, matching evidence.py's own
    `"h100" in fam` / `"v100" in fam` check so the two modules agree on the
    same job."""
    t = str(gpu_type or "").strip().lower()
    if "h100" in t:
        return "h100"
    if "v100" in t:
        return "v100"
    return "unknown"


def su_for(gpu_type, gpu_count, elapsed_s, estimated=False, overrides=None):
    """SU for one allocation: GPU-count x elapsed hours x the family's rate.

    Returns a dict, never a bare number, because a bare number cannot carry
    the two facts that matter more than the value itself: whether the elapsed
    time behind it was measured or estimated, and whether the rate it was
    charged at is a real, known one.

        {"value": float | None, "gpu_count": int, "gpu_type": str | None,
         "family": "h100" | "v100" | "unknown" | None,
         "rate_su_per_gpu_hour": float | None,
         "unknown_rate": bool, "unknown_elapsed": bool,
         "estimated": bool, "measured": bool, "reason": str}

    `value` is None only when `elapsed_s` cannot be read (never 0 for that
    case). `gpu_count` <= 0 legitimately costs 0 SU: Bridges-2 bills GPU-hours,
    and a CPU-only step has none to bill. Anything with GPUs and an unresolved
    family is charged `su_rates.json`'s fallback ceiling and flagged
    `unknown_rate` rather than costing 0.
    """
    r = rates(overrides)
    values = r["values"]
    count = _int(gpu_count, 0) or 0
    elapsed = _elapsed_value(elapsed_s)
    out = {"value": None, "gpu_count": count, "gpu_type": gpu_type,
           "family": None, "rate_su_per_gpu_hour": None,
           "unknown_rate": False, "unknown_elapsed": False,
           "estimated": bool(estimated), "measured": not bool(estimated),
           "reason": ""}
    if elapsed is None:
        out["unknown_elapsed"] = True
        out["reason"] = ("elapsed time is unknown for this allocation; it "
                         "contributes 'unknown' SU, never 0")
        return out
    if count <= 0:
        out["value"] = 0.0
        out["reason"] = "no GPU allocation (gpu_count=0): 0 SU, not a gap"
        return out
    family = _family_of(gpu_type)
    out["family"] = family
    if family in ("h100", "v100"):
        rate_key = "rates.%s.su_per_gpu_hour" % family
        rate = _num(values.get(rate_key))
        if rate is not None:
            out["value"] = round(count * (elapsed / 3600.0) * rate, 6)
            out["rate_su_per_gpu_hour"] = rate
            out["reason"] = ("%d x %s GPU-hour(s) at %.2f SU/GPU-hour"
                             % (count, family, rate))
            return out
        out["reason"] = ("su_rates.json declares no rates.%s.su_per_gpu_hour; "
                         "SU for this allocation is unknown, not 0" % family)
        out["unknown_elapsed"] = False
        out["unknown_rate"] = True
        return out
    fallback = _num(values.get("rates.unknown_gpu_fallback.su_per_gpu_hour"))
    out["unknown_rate"] = True
    if fallback is None:
        out["reason"] = ("gpu_type %r matches no known family and su_rates.json "
                         "carries no fallback rate; SU is unknown, not 0"
                         % (gpu_type,))
        return out
    out["value"] = round(count * (elapsed / 3600.0) * fallback, 6)
    out["rate_su_per_gpu_hour"] = fallback
    out["reason"] = ("gpu_type %r matches no known family (h100/v100); charged "
                     "at the %.2f SU/GPU-hour fallback ceiling so this does not "
                     "read as free compute" % (gpu_type, fallback))
    return out


# --- sacct row semantics (the layouts are parsed by corpus.parse_sacct_text) --
# The parent allocation's own id carries no step suffix; .batch/.extern and a
# bare numbered step (".0", ".1", ...) are the same allocation, not a second
# one. An array task's own suffix ("_<task>") is NOT stripped: task 1 and
# task 2 of the same array are different allocations with different elapsed
# times and must not fold together.
_SUBSTEP_RE = re.compile(r"\.(batch|extern|\d+)$", re.I)


def _rows_from_text(text):
    """Raw `sacct` text (either layout) -> [{col: val}], via the one shared parser.

    Delegates to `corpus.parse_sacct_text` rather than reading the layouts here.
    The v3.27.1 repair was caused by exactly one duplicate of this logic drifting
    from the other -- a stored sacct section that one reader parsed and another
    left as raw lines, which made every job-state check answer `unknown` on a
    bundle that held the TIMEOUT row all along. A second copy in this module
    would be free to drift the same way the next time the collected format
    changes, and the ledger would silently stop seeing jobs.

    A failure to import the shared parser returns no rows and is reported by the
    caller, because a ledger that quietly reads nothing looks exactly like a
    campaign that spent nothing.
    """
    try:
        from . import corpus
    except Exception:
        return []
    rows, _reason = corpus.parse_sacct_text(text)
    return [r for r in rows if isinstance(r, dict)]


def _rows_of(rows):
    """Raw sacct dict rows from `rows`, whichever shape it arrived in.

    Accepted: raw multi-line sacct text (either layout), a list of such text
    lines, or a list of already-parsed {column: value} dicts (a caller that
    ran its own sacct query and has JSON rows to hand). Anything else is
    empty rather than guessed at.
    """
    if isinstance(rows, str):
        return _rows_from_text(rows)
    if isinstance(rows, (list, tuple)):
        if not rows:
            return []
        if all(isinstance(r, dict) for r in rows):
            return list(rows)
        if all(isinstance(r, str) for r in rows):
            return _rows_from_text("\n".join(rows))
    return []


def _canon_row(d):
    jobid = str(_get_ci(d, "JobID") or _get_ci(d, "JobIDRaw") or "").strip()
    state = str(_get_ci(d, "State") or "").strip().upper()
    elapsed_raw = _get_ci(d, "Elapsed")
    tres = _get_ci(d, "AllocTRES") or _get_ci(d, "ReqTRES") or ""
    gpu_count, gpu_type = _gpu_from_tres(tres)
    return {"jobid": jobid, "state": state, "elapsed_raw": elapsed_raw,
            "elapsed_s": _elapsed_value(elapsed_raw), "alloc_tres": str(tres),
            "gpu_count": gpu_count, "gpu_type": gpu_type, "raw": d}


def _parent_id(jobid):
    return _SUBSTEP_RE.sub("", jobid)


def parse_sacct(rows):
    """Raw sacct rows -> one normalised record per job, sub-steps folded in.

    `rows` may be raw sacct text (`-P` delimited or fixed-width with a ruler),
    a list of such lines, or a list of already-parsed row dicts. Each element
    of the result is:

        {"jobid": str, "state": str, "elapsed_s": float | None,
         "elapsed_raw": str, "gpu_count": int, "gpu_type": str | None,
         "alloc_tres": str, "substep_ids": [str, ...], "source": str,
         "raw": dict}

    `jobid` is the parent allocation's id (an array task's own id, e.g.
    "44700000_2", if it is one). Its `elapsed_s`/`gpu_count`/`gpu_type` come
    from the row with that exact id -- sacct's parent allocation row, which is
    where the job's own AllocTRES and Elapsed live -- never summed with a
    `.batch`/`.extern`/numbered-step row's own fields, which is what keeps a
    job's SU counted once. When no bare parent row is present (a filtered
    query, or a step someone recorded without `-X` and the parent line was
    dropped), the first sub-step carrying a GPU allocation stands in, and
    `source` says so, rather than the job disappearing.
    """
    raw = _rows_of(rows)
    canon = [_canon_row(d) for d in raw if isinstance(d, dict)]
    canon = [c for c in canon if c["jobid"]]
    groups = {}
    for row in canon:
        parent = _parent_id(row["jobid"])
        g = groups.setdefault(parent, {"parent": None, "substeps": []})
        if row["jobid"] == parent:
            g["parent"] = row
        else:
            g["substeps"].append(row)
    jobs = []
    for parent_id, g in groups.items():
        if g["parent"] is not None:
            main, source = g["parent"], "parent allocation row"
        elif g["substeps"]:
            with_gpu = [s for s in g["substeps"] if s["gpu_count"]]
            main = with_gpu[0] if with_gpu else g["substeps"][0]
            source = ("no bare parent row was present for %s; used substep "
                      "%s as a fallback" % (parent_id, main["jobid"]))
        else:
            continue
        jobs.append({
            "jobid": parent_id, "state": main["state"],
            "elapsed_s": main["elapsed_s"], "elapsed_raw": main["elapsed_raw"],
            "gpu_count": main["gpu_count"], "gpu_type": main["gpu_type"],
            "alloc_tres": main["alloc_tres"],
            "substep_ids": [s["jobid"] for s in g["substeps"]],
            "source": source, "raw": main["raw"],
        })
    jobs.sort(key=lambda j: j["jobid"])
    return jobs


# --- ledger storage ---------------------------------------------------------
_DEFAULT_BASE_DIR = "results/framework/_brain"


def _ledger_dir(domain, base_dir=None):
    base = base_dir or os.environ.get("BRAIN_SU_LEDGER_DIR") or _DEFAULT_BASE_DIR
    return pathlib.Path(base) / _domain(domain)


def _ledger_path(domain, base_dir=None):
    return _ledger_dir(domain, base_dir) / "su_ledger.jsonl"


def _normalize_su(su_in, gpu_type, gpu_count, elapsed_s, estimated):
    """A caller-supplied `su` value into the standard flagged shape.

    `None` computes it here via su_for(); a dict is trusted but back-filled so
    every stored entry carries the same keys; a bare number is accepted at
    face value but recorded as such, because a raw float alone cannot say
    whether it was measured or estimated -- that is exactly what this
    function exists not to lose.
    """
    if su_in is None:
        return su_for(gpu_type, gpu_count, elapsed_s, estimated=estimated)
    if isinstance(su_in, dict):
        out = dict(su_in)
        out.setdefault("value", None)
        if out["value"] is not None:
            out["value"] = float(out["value"])
        out.setdefault("unknown_rate", False)
        out.setdefault("unknown_elapsed", out["value"] is None)
        out.setdefault("estimated", bool(estimated))
        out.setdefault("measured", not out.get("estimated", estimated))
        out.setdefault("family", None)
        out.setdefault("rate_su_per_gpu_hour", None)
        out.setdefault("reason", "caller-provided SU value")
        return out
    try:
        value = float(su_in)
    except (TypeError, ValueError):
        value = None
    return {"value": value, "unknown_rate": False,
            "unknown_elapsed": value is None,
            "estimated": bool(estimated), "measured": not bool(estimated),
            "family": None, "rate_su_per_gpu_hour": None,
            "reason": "caller-provided SU value, not computed by su_for()"}


def _normalize_entry(entry):
    """The caller's entry into the stored shape, or raise on a broken key.

    `domain`/`job`/`step` are the entry's identity: without them the entry
    cannot be filed, deduplicated or reconciled, and accepting it anyway would
    corrupt every aggregate silently rather than failing loudly at the one
    point the mistake is still local to the caller.
    """
    if not isinstance(entry, dict):
        raise ValueError("su_ledger entry must be a dict")
    domain = _domain(entry.get("domain"))
    if not domain:
        raise ValueError("su_ledger entry carries no domain")
    job = str(entry.get("job") or "").strip()
    if not job:
        raise ValueError("su_ledger entry carries no job id")
    step = str(entry.get("step") or "").strip()
    if not step:
        raise ValueError("su_ledger entry carries no step")
    gpu_count = _int(entry.get("gpu_count"), 0) or 0
    gpu_type_raw = entry.get("gpu_type")
    gpu_type = str(gpu_type_raw).strip() if gpu_type_raw not in (None, "") else None
    elapsed_s = _elapsed_value(entry.get("elapsed_s"))
    estimated = bool(entry.get("estimated", False))
    su = _normalize_su(entry.get("su"), gpu_type, gpu_count, elapsed_s, estimated)
    return {
        "job": job, "domain": domain, "round": entry.get("round"), "step": step,
        "actor": entry.get("actor"), "gpu_count": gpu_count, "gpu_type": gpu_type,
        "elapsed_s": elapsed_s if elapsed_s is not None else "unknown",
        "su": su, "sacct_row": entry.get("sacct_row"),
        "ts": entry.get("ts") or _now_iso(),
    }


def _find_latest(path, key):
    """The last stored entry matching `(job, step)`, or None."""
    if not path.exists():
        return None
    found = None
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if (str(obj.get("job")), str(obj.get("step"))) == key:
                    found = obj
    except OSError:
        return None
    return found


def record(entry, base_dir=None):
    """Append one entry to its domain's ledger.

    Append-only: nothing is ever rewritten or deleted on disk. Idempotent by
    `(job, step)`: recording the same key again does not create a second bill
    when the ledger is totalled -- every reader folds a key down to its last
    line -- and this call's own return says `updated: true` plus the entry it
    superseded, so the update is visible at the moment it happens rather than
    only inferable later from file order.
    """
    norm = _normalize_entry(entry)
    path = _ledger_path(norm["domain"], base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (norm["job"], norm["step"])
    previous = _find_latest(path, key)
    with open(str(path), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(norm, sort_keys=True) + "\n")
    return {"stored": True, "key": list(key), "updated": previous is not None,
            "previous": previous, "entry": norm}


def _read_deduped(domain, base_dir=None):
    """Every `(job, step)` in the domain's ledger, each at its last-written
    value, in first-seen order."""
    path = _ledger_path(domain, base_dir)
    if not path.exists():
        return []
    latest, order = {}, []
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = (str(obj.get("job")), str(obj.get("step")))
                if key not in latest:
                    order.append(key)
                latest[key] = obj
    except OSError:
        return []
    return [latest[k] for k in order]


def _su_value_of(entry):
    """(value, unknown_rate, unknown_elapsed) from a stored entry's `su`."""
    su = entry.get("su")
    if isinstance(su, dict):
        value = su.get("value")
        return (float(value) if value is not None else None,
                bool(su.get("unknown_rate")), value is None)
    try:
        v = float(su)
        return v, False, False
    except (TypeError, ValueError):
        return None, False, True


def _aggregate(entries):
    total_su, n_unknown = 0.0, 0
    unknown_su_jobs, unknown_rate_jobs = [], []
    for e in entries:
        value, unk_rate, unk_elapsed = _su_value_of(e)
        if unk_rate:
            unknown_rate_jobs.append(e.get("job"))
        if value is None:
            n_unknown += 1
            unknown_su_jobs.append(e.get("job"))
            continue
        total_su += value
    return {"su": round(total_su, 6), "n_entries": len(entries),
            "n_unknown": n_unknown, "unknown_su_jobs": unknown_su_jobs,
            "unknown_rate_jobs": unknown_rate_jobs}


def total(domain, since=None, base_dir=None):
    """SU spent in `domain`, optionally restricted to entries at/after `since`.

    `{"su": float, "n_entries": int, "n_unknown": int, "unknown_su_jobs": [...],
      "unknown_rate_jobs": [...]}`. `su` sums only entries with a known value;
    an unknown-elapsed job is named in `unknown_su_jobs`, never folded into
    `su` as a 0.
    """
    entries = _read_deduped(domain, base_dir)
    if since is not None:
        cutoff = _ts(since)
        if cutoff is not None:
            entries = [e for e in entries if (_ts(e.get("ts")) or 0.0) >= cutoff]
    return _aggregate(entries)


def by_actor(domain, base_dir=None):
    """{actor: totals} over the domain's deduplicated entries."""
    entries = _read_deduped(domain, base_dir)
    groups = {}
    for e in entries:
        groups.setdefault(str(e.get("actor")), []).append(e)
    return {k: _aggregate(v) for k, v in groups.items()}


def by_step(domain, base_dir=None):
    """{step: totals} over the domain's deduplicated entries."""
    entries = _read_deduped(domain, base_dir)
    groups = {}
    for e in entries:
        groups.setdefault(str(e.get("step")), []).append(e)
    return {k: _aggregate(v) for k, v in groups.items()}


def remaining(domain, budget, base_dir=None):
    """Campaign SU spent against `budget`'s envelope.

    `budget` is the domain config's own `budget` block (`su_envelope` /
    `envelope`, `per_round_cap` / `round_cap`) -- the same block
    `signals._check_budget` reads via `config_block("budget")`. Shaped so its
    output can be merged straight into a bundle's `su` section: `campaign_su`,
    `envelope` and `per_round_cap` are exactly the keys that check looks for.
    """
    campaign = total(domain, base_dir=base_dir)
    b = budget if isinstance(budget, dict) else {}
    envelope = _num(_get_ci(b, "su_envelope", "envelope"))
    per_round_cap = _num(_get_ci(b, "per_round_cap", "round_cap"))
    spent = campaign["su"]
    remaining_su = None if envelope is None else round(envelope - spent, 6)
    pct_used = round(spent / envelope, 4) if envelope else None
    reason = None
    if envelope is None:
        reason = ("no su_envelope in the given budget config; remaining SU is "
                  "unknown, not the full envelope")
    return {"domain": domain, "envelope": envelope,
            "per_round_cap": per_round_cap, "campaign_su": spent,
            "remaining_su": remaining_su, "pct_used": pct_used,
            "unknown_su_jobs": campaign["unknown_su_jobs"],
            "unknown_rate_jobs": campaign["unknown_rate_jobs"], "reason": reason}


# --- reconciliation ---------------------------------------------------------
def reconcile(domain, sacct_rows, tolerance=None, base_dir=None):
    """The domain's ledger against a fresh `sacct` pull, job by job.

    `sacct_rows` is anything `parse_sacct()` accepts. `tolerance` is a percent
    (1.0 = 1%); when omitted it is read from su_rates.json's
    `reconcile.tolerance_pct`.

    Returns `{ok, domain, tolerance_pct, ledger_su, sacct_su, delta,
    delta_pct, missing_jobs, extra_jobs, unknown_rate_jobs,
    unknown_elapsed_jobs, jobs}`. `jobs` is the per-job breakdown named in the
    WP5 spec's own reasoning for why a total is not enough: two jobs whose
    errors cancel at the aggregate must still both appear here with
    `ok: false`, which is why `ok` is computed from every per-job verdict and
    not from `delta_pct` alone. `missing_jobs` (in sacct, never recorded) and
    `extra_jobs` (recorded, absent from this sacct window) are each also a
    reason `ok` is false, independent of tolerance.
    """
    tol = _num(tolerance)
    tol_source = "caller"
    if tol is None:
        r = rates()
        tol = _num(r["values"].get("reconcile.tolerance_pct"))
        tol_source = "su_rates.json" if tol is not None else "hardcoded fallback"
    if tol is None:
        tol = 1.0

    sacct_jobs = parse_sacct(sacct_rows)
    sacct_by_id = {}
    for j in sacct_jobs:
        sacct_by_id[j["jobid"]] = su_for(j["gpu_type"], j["gpu_count"],
                                         j["elapsed_s"])

    ledger_by_id = {}
    for e in _read_deduped(domain, base_dir):
        job = str(e.get("job"))
        value, unk_rate, unk_elapsed = _su_value_of(e)
        cur = ledger_by_id.get(job)
        if cur is None:
            ledger_by_id[job] = {"value": value, "unknown_rate": unk_rate,
                                 "unknown_elapsed": unk_elapsed}
        else:
            # More than one ledger step shares this sacct job id (unusual but
            # not invalid, e.g. a step re-recorded under a new step name):
            # summed rather than one silently overwriting the other.
            if value is not None and cur["value"] is not None:
                cur["value"] = round(cur["value"] + value, 6)
            elif value is not None:
                cur["value"] = value
            cur["unknown_rate"] = cur["unknown_rate"] or unk_rate
            cur["unknown_elapsed"] = cur["unknown_elapsed"] or unk_elapsed

    all_ids = sorted(set(sacct_by_id) | set(ledger_by_id))
    jobs, missing_jobs, extra_jobs = [], [], []
    unknown_rate_jobs, unknown_elapsed_jobs = [], []
    ledger_total, sacct_total = 0.0, 0.0

    for jid in all_ids:
        sac, led = sacct_by_id.get(jid), ledger_by_id.get(jid)
        sac_val = sac["value"] if sac else None
        led_val = led["value"] if led else None
        if sac and sac.get("unknown_rate"):
            unknown_rate_jobs.append({"job": jid, "side": "sacct"})
        if led and led.get("unknown_rate"):
            unknown_rate_jobs.append({"job": jid, "side": "ledger"})
        row = {"job": jid, "ledger_su": led_val, "sacct_su": sac_val,
               "delta": None, "delta_pct": None, "ok": False, "note": ""}

        if sac is None:
            row["note"] = "recorded in the ledger but absent from this sacct window"
            extra_jobs.append({"job": jid, "ledger_su": led_val})
        elif led is None:
            row["note"] = "present in sacct but never recorded in the ledger"
            missing_jobs.append({"job": jid, "sacct_su": sac_val})
        elif sac.get("unknown_elapsed") or led.get("unknown_elapsed") \
                or sac_val is None or led_val is None:
            row["note"] = ("elapsed time is unknown on at least one side; not "
                           "numerically comparable")
            unknown_elapsed_jobs.append({"job": jid})
        else:
            delta = round(led_val - sac_val, 6)
            row["delta"] = delta
            if sac_val:
                delta_pct = round(delta / sac_val * 100.0, 4)
                row["delta_pct"] = delta_pct
                row["ok"] = abs(delta_pct) <= tol
                row["note"] = ("within tolerance" if row["ok"] else
                               "outside the %.2f%% tolerance" % tol)
            else:
                row["delta_pct"] = 0.0 if delta == 0 else None
                row["ok"] = delta == 0
                row["note"] = ("within tolerance" if row["ok"] else
                               "sacct measured 0 SU for this job but the "
                               "ledger recorded %.6f" % led_val)
        jobs.append(row)
        if led_val is not None:
            ledger_total += led_val
        if sac_val is not None:
            sacct_total += sac_val

    ledger_total = round(ledger_total, 6)
    sacct_total = round(sacct_total, 6)
    delta = round(ledger_total - sacct_total, 6)
    delta_pct = round(delta / sacct_total * 100.0, 4) if sacct_total else \
        (0.0 if delta == 0 else None)

    ok = (not missing_jobs and not extra_jobs and not unknown_elapsed_jobs
          and all(j["ok"] for j in jobs)
          and delta_pct is not None and abs(delta_pct) <= tol)

    return {"ok": ok, "domain": domain, "tolerance_pct": tol,
            "tolerance_source": tol_source, "ledger_su": ledger_total,
            "sacct_su": sacct_total, "delta": delta, "delta_pct": delta_pct,
            "missing_jobs": missing_jobs, "extra_jobs": extra_jobs,
            "unknown_rate_jobs": unknown_rate_jobs,
            "unknown_elapsed_jobs": unknown_elapsed_jobs, "jobs": jobs}


# --- CLI --------------------------------------------------------------------
def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="su_ledger",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--base-dir", default=None,
                    help="ledger root (default: results/framework/_brain, "
                         "or $BRAIN_SU_LEDGER_DIR)")
    sub = ap.add_subparsers(dest="cmd")

    rec = sub.add_parser("record", help="append one ledger entry (JSON)")
    rec.add_argument("entry", nargs="?",
                     help="entry as a JSON string; reads stdin if omitted")

    tot = sub.add_parser("total", help="SU spent in one domain")
    tot.add_argument("domain")
    tot.add_argument("--since", default=None)

    rec2 = sub.add_parser("reconcile", help="ledger vs a fresh sacct pull")
    rec2.add_argument("domain")
    rec2.add_argument("--sacct-file", default=None,
                      help="raw sacct text or a JSON row list; reads stdin "
                           "if omitted")
    rec2.add_argument("--tolerance", type=float, default=None,
                      help="percent; default is su_rates.json's "
                           "reconcile.tolerance_pct")

    sub.add_parser("rates", help="effective SU rates and their sources")

    args = ap.parse_args(argv)

    if args.cmd == "record":
        raw = args.entry if args.entry is not None else sys.stdin.read()
        try:
            entry = json.loads(raw)
        except Exception as exc:
            print("entry is not valid JSON: %s" % exc)
            return 2
        try:
            _dump(record(entry, base_dir=args.base_dir))
        except ValueError as exc:
            print("cannot record entry: %s" % exc)
            return 2
        return 0
    if args.cmd == "total":
        _dump(total(args.domain, since=args.since, base_dir=args.base_dir))
        return 0
    if args.cmd == "reconcile":
        raw = (open(args.sacct_file, "r", encoding="utf-8").read()
              if args.sacct_file else sys.stdin.read())
        try:
            sacct_rows = json.loads(raw)
        except Exception:
            sacct_rows = raw
        result = reconcile(args.domain, sacct_rows, tolerance=args.tolerance,
                           base_dir=args.base_dir)
        _dump(result)
        return 0 if result["ok"] else 1
    if args.cmd == "rates":
        _dump(rates())
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
