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

Two record kinds were added for the INC autopilot (docs/INC_AUTOPILOT.md,
section d); neither changes what `propose` or `decide` do.

* **grant** -- the campaign-envelope rule. The owner decided on 2026-09-27
  that the INC autopilot may run its R3 menu builds without a per-item
  approval, inside a campaign envelope a person enabled, once the replay tests
  pass. That is recorded here as a grant by the one actor the code names in
  `ENVELOPE_GRANTEES`, never through `decide()`, which still refuses every
  non-human actor. A grant applies only to a pending R3 item of an action in
  `ENVELOPE_ACTIONS` that the grantee itself requested (a brain's proposal,
  or any other tier's, waits for a person), and must carry the item's own
  lever, trigger diagnoses, cites and estimate, the envelope balance, and the
  person whose envelope it draws on. The grantee and action lists are code,
  not data, for the same reason `policy._CEILING` is; the fold applies the
  same scope, so a grant line written by hand does not approve anything else.
* **executed** -- what the executor did with an approved item. `started` is
  appended before anything runs and only if the item has no execution record
  yet, under an exclusive lock, so one approval id runs at most once even when
  two callers race. `done` or `failed` closes it. A crash between the two
  leaves `started` with no outcome, which reads as "may have run" and is never
  retried automatically. `released` closes a started execution whose call
  certainly never reached the cluster (the executor's `never_ran`: ssh never
  connected): the item is approved and executable again, and the released
  attempt is kept under `released` on the item. It is never written for a
  call that ran or may have run.

`decide()` takes the same lock for its read-check-append, so a person's
ruling and a grant on the same item cannot both be told they won.
"""

import json
import os
import sys
import time

try:
    import fcntl
except ImportError:                     # not on the lab or the cluster; no lock there
    fcntl = None

DEFAULT_BASE_DIR = "results/framework/_brain"
STATUSES = ("pending", "approved", "denied")
TERMINAL = ("approved", "denied")

# Actors that may record an approve-within-envelope grant. Code, not data: a
# config that could add a name here could grant itself R3 authority.
ENVELOPE_GRANTEES = ("round-scheduler:inc-autopilot",)
# The only actions a grant may approve: the INC menu builds (levers L1, L2,
# L5 and L8; docs/INC_AUTOPILOT.md section d), and a stream campaign's builds
# (docs/CONTINUOUS_LOOP.md 6.5: L18, L22 and L28 through inc_build_segment,
# L20 and L27 through inc_build_consolidation, L21's rollback to the pool the
# last 'hurts' milestone recommended, L23's baselines, L23N's native-resolution
# rescore of a done measurement arm, E1's base v3 build (L23V) and agnostic
# rescore (L23E), E2's rescore and verdict (L23C), E2-C's attribution rescore
# (L23D), E3's scores of one arm (L23F) and its verdict (L23G), L25's Stage A
# pilot and LI, the stream's
# creation). L23's splits build
# and lock, a source that failed its pre-check and a funnel_F9 release are not
# here: a person decides them. Anything else in the queue is decided by a
# person, whoever filed it.
ENVELOPE_ACTIONS = ("inc_build_pilot", "inc_build_realloop", "inc_build_baseline",
                    "inc_build_segment", "inc_build_consolidation", "inc_stream_rollback",
                    "inc_build_baseline_v2", "inc_build_pilot4", "inc_stream_init", "inc_rescore_native",
                    "inc_build_base3", "inc_rescore_agnostic", "inc_rescore_e2", "inc_rescore_e2_attr",
                    "inc_score_e3", "inc_verdict_e3")
EXEC_PHASES = ("started", "done", "failed", "released")


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


def envelope_scope(item):
    """"" when a grant may approve this item, else why not.

    Pending, R3, an action in ENVELOPE_ACTIONS, requested by a grantee, and
    not a proposal someone else made that a grantee re-filed
    (`context.proposed_by`). The fold and `approve_within_envelope` both use it.
    """
    if not isinstance(item, dict):
        return "no such approval item"
    if item.get("status") != "pending":
        return "already %s by %s" % (item.get("status"), item.get("decided_by"))
    if str(item.get("risk") or "").upper() != "R3":
        return "the envelope covers R3 builds only; this item is %s" % item.get("risk")
    if item.get("action") not in ENVELOPE_ACTIONS:
        return ("the envelope covers %s only; %r is decided by a person"
                % (", ".join(ENVELOPE_ACTIONS), item.get("action")))
    if item.get("requested_by") not in ENVELOPE_GRANTEES:
        return ("the item was requested by %r; a grant covers only the grantee's own requests, "
                "so it waits for a person" % (item.get("requested_by"),))
    cx = item.get("context") if isinstance(item.get("context"), dict) else {}
    by = cx.get("proposed_by")
    if by not in (None, "") and by not in ENVELOPE_GRANTEES:
        return ("the item carries a proposal by %r; its autonomy is never self-granted, so it "
                "waits for a person" % (by,))
    return ""


def _fold_grant(item, rec):
    """Apply one grant record, or keep it as an attempt when it is not valid."""
    valid = (rec.get("decided_by") in ENVELOPE_GRANTEES and not envelope_scope(item))
    if not valid:
        item.setdefault("attempts", []).append(rec)
        return
    item.update(status="approved", decided_by=rec.get("decided_by"),
                decided_at=rec.get("ts"), decision_reason=rec.get("reason"),
                decision_basis="envelope", grant=rec.get("grant"))


def _fold_executed(item, rec):
    """Apply one executed record, or keep it as an attempt when out of order."""
    phase = rec.get("phase")
    ex = item.get("execution")
    if phase == "started":
        ok = item.get("status") == "approved" and ex is None
    elif phase in ("done", "failed", "released"):
        ok = isinstance(ex, dict) and ex.get("phase") == "started"
    else:
        ok = False
    if not ok:
        item.setdefault("execution_attempts", []).append(rec)
        return
    if phase == "started":
        item["execution"] = {"phase": "started", "executed_by": rec.get("executed_by"),
                             "started_at": rec.get("ts"), "finished_at": None,
                             "outcome": None}
    elif phase == "released":
        # The call never reached the cluster: the claim is given back.
        item.setdefault("released", []).append(
            dict(ex, phase="released", finished_at=rec.get("ts"), outcome=rec.get("outcome")))
        item["execution"] = None
    else:
        ex.update(phase=phase, finished_at=rec.get("ts"), outcome=rec.get("outcome"))


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
        elif kind == "grant":
            item = items.get(rid)
            if item is not None:
                _fold_grant(item, rec)
        elif kind == "executed":
            item = items.get(rid)
            if item is not None:
                _fold_executed(item, rec)
    return items


def propose(domain, action, params, risk, requested_by, reason, ts,
            review_id=None, est_su=None, base_dir=None, root=None, context=None):
    """Queue a request. Returns the item, or a refusal naming the reason.

    `ts` is supplied by the caller so a queue can be rebuilt deterministically
    from its inputs in a test or a replay.

    `context` is optional caller metadata stored with the request (the INC
    autopilot puts its lever, trigger diagnoses, cites, argv and campaign
    there). A request filed without it is byte-for-byte the record it always
    was.
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
    if context is not None and not isinstance(context, dict):
        return {"ok": False, "reason": "context must be an object"}
    rec = {"kind": "request", "id": rid, "ts": ts, "domain": domain,
           "action": action, "params": params or {}, "risk": risk,
           "requested_by": requested_by, "reason": reason,
           "review_id": review_id, "est_su": est_su}
    if context is not None:
        rec["context"] = context
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
    p = path(domain, base_dir, root)
    if not os.path.exists(p):
        return {"ok": False, "reason": "no such approval item"}
    # The read-check-append runs under the same lock as the envelope grant and
    # the execution record, so a person's ruling and the autopilot's grant on
    # one item cannot both be told they won.
    try:
        with _LogLock(p):
            item = state(domain, base_dir, root).get(item_id)
            if item is None:
                return {"ok": False, "reason": "no such approval item"}
            if item.get("status") in TERMINAL:
                rec = {"kind": "decision", "id": item_id, "ts": ts,
                       "decision": "approved" if decision == "approve" else "denied",
                       "decided_by": actor, "reason": reason, "superseded": True}
                _append(p, rec)
                return {"ok": False, "reason": "already %s by %s; the attempt is recorded"
                                               % (item["status"], item.get("decided_by"))}
            rec = {"kind": "decision", "id": item_id, "ts": ts,
                   "decision": "approved" if decision == "approve" else "denied",
                   "decided_by": actor, "reason": reason}
            if not _append(p, rec):
                return {"ok": False, "reason": "the approval log could not be written"}
    except OSError as e:
        return {"ok": False, "reason": "the approval log could not be locked: %s" % e}
    return {"ok": True, "item": state(domain, base_dir, root).get(item_id)}


def pending(domain, base_dir=None, root=None):
    return [i for i in state(domain, base_dir, root).values()
            if i.get("status") == "pending"]


class _LogLock(object):
    """Exclusive lock on a sidecar of the log, held across a read-check-append.

    fcntl.flock is per open file description, so two threads of the dashboard
    process exclude each other as well as two processes do. Without fcntl the
    lock is a no-op; neither the lab nor the cluster is such a host.
    """

    def __init__(self, log_path):
        self.lock_path = log_path + ".lock"
        self.fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self.fh = open(self.lock_path, "a")
        if fcntl is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            if fcntl is not None:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
        return False


def _canon(v):
    return json.dumps(v, sort_keys=True, default=str)


def _grant_differs(item, g):
    """What a grant states differently from the filed item it approves."""
    cx = item.get("context") if isinstance(item.get("context"), dict) else {}
    out = []
    for k in ("lever", "trigger", "cites"):
        if _canon(g.get(k)) != _canon(cx.get(k)):
            out.append("%s %s is not the item's %s" % (k, _canon(g.get(k))[:120],
                                                        _canon(cx.get(k))[:120]))
    est = (g.get("envelope") or {}).get("est_su")
    filed = item.get("est_su")
    if not isinstance(filed, (int, float)) or isinstance(filed, bool) \
            or abs(float(est) - float(filed)) > 1e-9:
        out.append("est_su %s is not the item's %r" % (est, filed))
    return out


def approve_within_envelope(domain, item_id, granted_by, grant, reason, ts,
                            base_dir=None, root=None):
    """Record the campaign-envelope approval of one pending R3 item.

    Not a decision: `decide()` stays human-only. This is the standing rule the
    owner granted on 2026-09-27 (docs/INC_AUTOPILOT.md, section d), recorded by
    the one actor it names, with what it rests on. The caller (the INC executor)
    has already checked that the envelope is on, the replay tests pass on the
    current code, and the estimate fits the balance; this function refuses a
    grant that does not even say so, an item outside `envelope_scope`, and a
    grant whose lever, trigger, cites or estimate are not the item's own (the
    audit record must describe what was filed, not what a later request said).
    """
    actor = str(granted_by or "")
    if actor.startswith("human:"):
        return {"ok": False, "reason": "a person approves through decide(); a grant is "
                                       "the envelope rule's record, not a person's"}
    if actor not in ENVELOPE_GRANTEES:
        return {"ok": False, "reason": "%r holds no envelope grant (grantees: %s)"
                                       % (actor, ", ".join(ENVELOPE_GRANTEES))}
    if not str(reason or "").strip():
        return {"ok": False, "reason": "a grant with no reason is not a record"}
    g = grant if isinstance(grant, dict) else {}
    missing = []
    if not str(g.get("lever") or "").strip():
        missing.append("lever")
    if not g.get("trigger"):
        missing.append("trigger")
    if not g.get("cites"):
        missing.append("cites")
    env = g.get("envelope") if isinstance(g.get("envelope"), dict) else {}
    for k in ("remaining_su", "est_su"):
        v = env.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            missing.append("envelope." + k)
    if not str(g.get("authority") or "").startswith("human:"):
        missing.append("authority (the person whose envelope this draws on)")
    if missing:
        return {"ok": False, "reason": "the grant does not state %s" % ", ".join(missing)}
    if env["est_su"] > env["remaining_su"]:
        return {"ok": False, "reason": "estimated %.3g SU exceeds the %.3g SU left in the "
                                       "envelope" % (env["est_su"], env["remaining_su"])}
    p = path(domain, base_dir, root)
    with _LogLock(p):
        item = state(domain, base_dir, root).get(item_id)
        why = envelope_scope(item)
        if why:
            return {"ok": False, "reason": why}
        differs = _grant_differs(item, g)
        if differs:
            return {"ok": False, "reason": "the grant does not describe the filed item: %s"
                                           % "; ".join(differs)}
        rec = {"kind": "grant", "id": item_id, "ts": ts, "decision": "approved",
               "decided_by": actor, "basis": "envelope", "grant": g, "reason": reason}
        if not _append(p, rec):
            return {"ok": False, "reason": "the approval log could not be written"}
    return {"ok": True, "item": state(domain, base_dir, root).get(item_id)}


def record_executed(domain, item_id, phase, executed_by, ts, outcome=None,
                    base_dir=None, root=None):
    """Record what the executor did with an approved item; one run per id.

    `started` succeeds only for an approved item with no execution record, so
    a second call for the same approval id is refused, not run again. `done`
    and `failed` close a started execution once; `released` closes it as a
    call that never reached the cluster and makes the item executable again
    (the executor writes it only then).
    """
    if phase not in EXEC_PHASES:
        return {"ok": False, "reason": "phase must be one of %s" % (EXEC_PHASES,)}
    if not str(executed_by or "").strip():
        return {"ok": False, "reason": "an execution with no executor cannot be audited"}
    p = path(domain, base_dir, root)
    with _LogLock(p):
        item = state(domain, base_dir, root).get(item_id)
        if item is None:
            return {"ok": False, "reason": "no such approval item"}
        ex = item.get("execution")
        if phase == "started":
            if item.get("status") != "approved":
                return {"ok": False, "reason": "item is %s, not approved" % item.get("status")}
            if ex is not None:
                return {"ok": False,
                        "reason": "approval %s was already executed (%s by %s at %s); "
                                  "one approval runs once" % (item_id, ex.get("phase"),
                                                              ex.get("executed_by"),
                                                              ex.get("started_at"))}
        else:
            if not isinstance(ex, dict) or ex.get("phase") != "started":
                return {"ok": False, "reason": "no open execution to close for %s" % item_id}
        rec = {"kind": "executed", "id": item_id, "ts": ts, "phase": phase,
               "executed_by": executed_by}
        if outcome is not None:
            rec["outcome"] = outcome
        if not _append(p, rec):
            return {"ok": False, "reason": "the approval log could not be written"}
    return {"ok": True, "item": state(domain, base_dir, root).get(item_id)}


def awaiting_execution(domain, base_dir=None, root=None):
    """Approved items nothing has executed yet, oldest first."""
    rows = [i for i in state(domain, base_dir, root).values()
            if i.get("status") == "approved" and i.get("execution") is None]
    return sorted(rows, key=lambda r: (r.get("ts") or 0))


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
