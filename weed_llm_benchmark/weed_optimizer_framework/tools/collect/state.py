"""Per-source state: INC_DIR/intake/sources.jsonl (docs/CONTINUOUS_LOOP.md §3.1
"State", §7.4, §7.5).

An append-only, hash-chained ledger of events (format collect-source-event/1):
  candidate      plan saw the source for the first time          -> candidate
  held           a pre-check or intake refusal that waits          -> held
  closed         a pre-check that closes the source for good      -> closed
  fetch_started  a fetch attempt began (no status change)
  fetched        a fetch attempt ended with files in staging      -> fetched
  fetch_failed   a fetch attempt failed (counted toward the 3)
  intaken        an intake batch was committed                    -> intaken
  released       a person or a condition released a hold          -> candidate
Each event records what it needs: reasons, bytes, seconds, the Slurm job id
(so the autopilot can settle its SU from sacct), the batch and its yield.
fold() gives the current state of every source; the lab and the cluster
each append to their own copy, and fold() reads any number of them in time
order. A "quarantined" status is read from the registry (lever L24 is the
stream's), never written here.
"""
from __future__ import annotations

from . import (FORMATS, append_chained, parse_utc, read_jsonl, slurm_job_id, sources_ledger, utc)

EVENTS = ("candidate", "held", "closed", "fetch_started", "fetched", "fetch_failed", "intaken", "released")
STATUS_OF = {"candidate": "candidate", "held": "held", "closed": "closed", "fetched": "fetched",
             "intaken": "intaken", "released": "candidate"}


def append(inc, source, event, **fields):
    """Append one event. An event with seconds inside a Slurm job carries its
    SU estimate (one V100: 1 SU per GPU-hour; the autopilot settles it from
    sacct by the job id)."""
    if event not in EVENTS:
        raise ValueError("unknown source event %r" % event)
    row = {"format": FORMATS["event"], "ts": fields.pop("ts", None) or utc(), "source": source, "event": event,
           "status": STATUS_OF.get(event), "slurm_job_id": slurm_job_id()}
    row.update(fields)
    if row.get("seconds") is not None and "su" not in row:
        row["su"] = round(float(row["seconds"]) / 3600.0, 6) if row["slurm_job_id"] else 0.0
    return append_chained(sources_ledger(inc), row)


def read(inc=None, extra=()):
    rows = list(read_jsonl(sources_ledger(inc), missing_ok=True))
    for p in extra or ():
        rows.extend(read_jsonl(p, missing_ok=True))
    rows.sort(key=lambda r: (r.get("ts") or "", r.get("source") or ""))
    return rows


def fold(rows):
    """{source: {"status", "reason", "provider", "ref", "attempts",
    "failed_attempts", "bytes", "seconds", "su", "batches", "yield", "events",
    "first_ts", "last_ts", "last_event", "holds", "jobs"}}."""
    out = {}
    for r in rows:
        s = out.setdefault(r["source"], {"status": None, "reason": None, "provider": None, "ref": None,
                                         "attempts": 0, "failed_attempts": 0, "bytes": 0, "seconds": 0.0, "su": 0.0,
                                         "batches": [], "yield": None, "events": 0, "first_ts": r.get("ts"),
                                         "last_ts": None, "last_event": None, "holds": [], "jobs": []})
        ev = r.get("event")
        s["events"] += 1
        s["last_ts"], s["last_event"] = r.get("ts"), ev
        for k in ("provider", "ref"):
            if r.get(k):
                s[k] = r[k]
        s["su"] = round(s["su"] + float(r.get("su") or 0), 6)
        if r.get("slurm_job_id") and r["slurm_job_id"] not in s["jobs"]:
            s["jobs"].append(r["slurm_job_id"])
        if ev == "fetch_started":
            s["attempts"] += 1
        elif ev == "fetch_failed":
            s["failed_attempts"] += 1
            s["seconds"] += float(r.get("seconds") or 0)
            s["reason"] = r.get("reason")
        elif ev == "fetched":
            s["bytes"] += int(r.get("bytes") or 0)
            s["seconds"] += float(r.get("seconds") or 0)
        elif ev == "intaken":
            if r.get("batch") and r["batch"] not in s["batches"]:
                s["batches"].append(r["batch"])
            s["yield"] = r.get("yield")
            s["seconds"] += float(r.get("seconds") or 0)
        if ev in ("held", "closed"):
            s["reason"] = r.get("reason")
            s["holds"] = list(r.get("codes") or [])
        elif ev in ("released", "fetched", "intaken"):
            s["holds"] = []
        st = STATUS_OF.get(ev)
        if st is not None:
            if ev == "candidate" and s["status"] not in (None, "candidate"):
                continue                       # plan never demotes a source it saw before
            s["status"] = st
    return out


def bytes_since(rows, since):
    """Bytes of fetched events at or after `since` (a datetime)."""
    n = 0
    for r in rows:
        if r.get("event") == "fetched" and r.get("ts"):
            try:
                if parse_utc(r["ts"]) >= since:
                    n += int(r.get("bytes") or 0)
            except ValueError:
                continue
    return n


def bytes_total(rows):
    return sum(int(r.get("bytes") or 0) for r in rows if r.get("event") == "fetched")
