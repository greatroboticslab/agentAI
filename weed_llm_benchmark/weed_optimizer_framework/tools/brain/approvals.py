"""The approval queue: what an agent tier asked for, and what a person decided.

Why this exists
---------------
The policy gate answers "may this actor take this action". For R3 -- external
side effects, SU-heavy runs, design changes -- the answer is "not on its own",
and something has to hold the request until a person rules on it. Without that
queue, R3 collapses into R4 in practice: an action nobody can ever request is
indistinguishable from one nobody may take.

Three rules make this a governance record rather than a to-do list.

* **Only a person decides.** `decide()` refuses any `decided_by` that is not a
  `human:*` actor. A tier that could approve its own request has no ceiling, and
  the whole risk table would be decoration.
* **R4 is never queued.** An irreversible action is refused at proposal time,
  not parked where one impatient click would run it. `policy.authorize` already
  refuses R4 for a non-human; queuing it anyway would reintroduce exactly the
  path that refusal exists to close.
* **A decision is final and the log is append-only.** A second decision on the
  same item is refused and recorded as an attempt, so "this was approved twice
  by two different people" is a readable event rather than a silent overwrite.

State is the fold of an append-only JSONL log, the same shape the correction
channel and the SU ledger use. Nothing here rewrites history.
"""

import json
import os
import sys
import time

DEFAULT_BASE_DIR = "results/framework/_brain"
STATUSES = ("pending", "approved", "denied")
TERMINAL = ("approved", "denied")


def _dir(domain, base_dir=None, root=None):
    base = base_dir or os.environ.get("BRAIN_APPROVALS_DIR") or DEFAULT_BASE_DIR
    dom = "".join(c for c in str(domain or "").strip().lower()
                  if c.isalnum() or c in "_-")[:40]
    return os.path.join(str(root or "."), base, dom)


def path(domain, base_dir=None, root=None):
    return os.path.join(_dir(domain, base_dir, root), "approvals.jsonl")


def _append(p, record):
    """Append one record. Never raises; returns the record or {} on failure."""
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        line = json.dumps(record, sort_keys=True)
        with open(p, "ab") as fh:
            fh.write(line.encode("utf-8") + b"\n")
        return record
    except Exception:
        return {}


def read(domain, base_dir=None, root=None):
    """Every record in the log, oldest first. A torn final line is skipped.

    A walltime-killed writer leaves one; losing the whole log over it would lose
    every decision ever made.
    """
    out = []
    try:
        with open(path(domain, base_dir, root), "rb") as fh:
            for raw in fh.read().decode("utf-8", "replace").splitlines():
                raw = raw.strip()
                if not raw.startswith("{"):
                    continue
                try:
                    out.append(json.loads(raw))
                except Exception:
                    continue
    except OSError:
        return []
    return out


def state(domain, base_dir=None, root=None):
    """Current state of every item, as the fold of the log."""
    items = {}
    for rec in read(domain, base_dir, root):
        rid = rec.get("id")
        if not rid:
            continue
        kind = rec.get("kind")
        if kind == "request":
            items.setdefault(rid, dict(rec, status="pending", attempts=[]))
        elif kind == "decision":
            item = items.get(rid)
            if item is None:
                continue
            if item.get("status") in TERMINAL:
                # Recorded, never applied: a second ruling on a settled item is
                # an event worth reading, not an overwrite.
                item.setdefault("attempts", []).append(rec)
                continue
            item.update(status=rec.get("decision"), decided_by=rec.get("decided_by"),
                        decided_at=rec.get("ts"), decision_reason=rec.get("reason"))
    return items


def propose(domain, action, params, risk, requested_by, reason, ts,
            review_id=None, est_su=None, base_dir=None, root=None):
    """Queue a request. Returns the item, or a refusal naming the reason.

    `ts` is supplied by the caller so a queue can be rebuilt deterministically
    from its inputs in a test or a replay.
    """
    risk = str(risk or "").upper()
    if risk == "R4":
        return {"ok": False, "reason": "R4 is irreversible and is never queued; "
                                       "a person takes it directly or not at all"}
    if risk not in ("R0", "R1", "R2", "R3"):
        return {"ok": False, "reason": "unknown risk tier %r" % (risk,)}
    if not str(reason or "").strip():
        return {"ok": False, "reason": "a request with no reason cannot be ruled on"}
    if not str(requested_by or "").strip():
        return {"ok": False, "reason": "a request with no requester cannot be audited"}
    rid = "ap-%s-%s" % (int(ts), abs(hash((domain, action, json.dumps(params or {},
                                                                     sort_keys=True),
                                           requested_by, ts))) % 10 ** 8)
    rec = {"kind": "request", "id": rid, "ts": ts, "domain": domain,
           "action": action, "params": params or {}, "risk": risk,
           "requested_by": requested_by, "reason": reason,
           "review_id": review_id, "est_su": est_su}
    if not _append(path(domain, base_dir, root), rec):
        return {"ok": False, "reason": "the approval log could not be written"}
    return {"ok": True, "item": dict(rec, status="pending")}


def decide(domain, item_id, decision, decided_by, reason, ts,
           base_dir=None, root=None):
    """Approve or deny one queued request. Only a `human:*` actor may."""
    decision = str(decision or "").strip().lower()
    if decision not in ("approve", "deny"):
        return {"ok": False, "reason": "decision must be approve or deny"}
    actor = str(decided_by or "")
    if not actor.startswith("human:"):
        return {"ok": False,
                "reason": "only a person decides an approval; %r is not a human "
                          "actor. A tier that could approve its own request has "
                          "no ceiling." % (actor,)}
    if not str(reason or "").strip():
        return {"ok": False, "reason": "a decision with no reason is not a record"}
    items = state(domain, base_dir, root)
    item = items.get(item_id)
    if item is None:
        return {"ok": False, "reason": "no such approval item"}
    if item.get("status") in TERMINAL:
        rec = {"kind": "decision", "id": item_id, "ts": ts,
               "decision": "approved" if decision == "approve" else "denied",
               "decided_by": actor, "reason": reason, "superseded": True}
        _append(path(domain, base_dir, root), rec)
        return {"ok": False, "reason": "already %s by %s; the attempt is recorded"
                                       % (item["status"], item.get("decided_by"))}
    rec = {"kind": "decision", "id": item_id, "ts": ts,
           "decision": "approved" if decision == "approve" else "denied",
           "decided_by": actor, "reason": reason}
    if not _append(path(domain, base_dir, root), rec):
        return {"ok": False, "reason": "the approval log could not be written"}
    return {"ok": True, "item": state(domain, base_dir, root).get(item_id)}


def pending(domain, base_dir=None, root=None):
    return [i for i in state(domain, base_dir, root).values()
            if i.get("status") == "pending"]


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="approvals")
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list", help="every item and its status")
    l.add_argument("domain")
    q = sub.add_parser("pending", help="items still waiting on a person")
    q.add_argument("domain")
    try:
        a = ap.parse_args(argv[1:])
    except SystemExit as exc:
        return int(exc.code or 0)
    if a.cmd == "list":
        print(json.dumps(state(a.domain, root=a.root), indent=1, sort_keys=True))
        return 0
    print(json.dumps(pending(a.domain, root=a.root), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
