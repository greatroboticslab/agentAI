"""The claims register (contract §8.3, runner §4.17 and §5.1.7).

A conclusion the platform or a person draws ("the harvest holds little target
data") is stored as a claim with its polarity, scope, author and cites, so it
can be challenged and tested instead of living only in prose. A claim's status
moves only along TRANSITIONS, and only by the actor kinds allowed there:

  * adversary: a validated devil's-advocate reply with at least one surviving
    counter-argument, from a model whose family differs from the planner's
    (a same-family reply may be recorded elsewhere but moves no claim);
  * autopilot: D18 silent on a valid audit whose ledger fingerprint matches;
    one cite must carry the audit's sha256;
  * person: card X12 (by = "human:<actor>").

History is append-only: save() refuses a register in which any claim's
recorded history is not a prefix of the new one, or whose filed fields
changed.

Standard library only.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

from . import (ClaimsError, FunnelError, _atomic_write_bytes, json_text, read_json, utc)

FORMAT = "funnel-claims/1"
POLARITIES = ("scarcity", "negative", "positive")
STATUSES = ("open", "challenged", "tested_survives", "refuted", "accepted_open")
NEGATIVE_POLARITIES = ("scarcity", "negative")
ACTOR_KINDS = ("adversary", "autopilot", "person")
TRANSITIONS = {("open", "challenged"): ("adversary", "person"),
               ("challenged", "tested_survives"): ("autopilot", "person"),
               ("challenged", "refuted"): ("person",),
               ("challenged", "accepted_open"): ("person",),
               ("open", "accepted_open"): ("person",),
               ("tested_survives", "challenged"): ("adversary", "person")}
FILED_KEYS = ("id", "text", "polarity", "scope", "made_by")
CLAIM_KEYS = FILED_KEYS + ("cites", "status", "history")
HISTORY_KEYS = ("utc", "from", "to", "by", "reason", "cites")
_MADE_BY_RE = re.compile(r"^(human-transcribed|card:[A-Za-z0-9_.-]+|tier2:\S+|human:\S+)$")
_ID_RE = re.compile(r"^C([1-9][0-9]*)$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def model_family(model_id):
    """The leading letters of a model's name after its provider prefix,
    lowercased: "vllm:glm-4.7-flash" -> "glm", "ollama:qwen3.8:27b" -> "qwen"."""
    s = str(model_id or "").strip().lower()
    parts = s.split(":")
    if len(parts) >= 2 and re.match(r"^[a-z_-]+$", parts[0]):
        s = ":".join(parts[1:])
    m = re.match(r"^[a-z]+", s)
    if not m:
        raise ClaimsError("cannot tell the model family of %r" % (model_id,))
    return m.group(0)


def cite_problems(c):
    """A cite is the inc_autopilot.model.cite form: {artifact, line, pointer,
    value} with a ledger line or a JSON pointer."""
    if not isinstance(c, dict):
        return ["a cite is not an object"]
    probs = []
    for k in ("artifact", "line", "pointer", "value"):
        if k not in c:
            probs.append("cite lacks %r" % k)
    if probs:
        return probs
    if not isinstance(c["artifact"], str) or not c["artifact"]:
        probs.append("cite artifact must be a path")
    if c["line"] is None and c["pointer"] is None:
        probs.append("a cite needs a line or a pointer")
    if c["line"] is not None and (not isinstance(c["line"], int) or isinstance(c["line"], bool)
                                  or c["line"] < 1):
        probs.append("cite line must be a 1-based integer")
    if c["pointer"] is not None and (not isinstance(c["pointer"], str) or
                                     not (c["pointer"] == "" or c["pointer"].startswith("/"))):
        probs.append("cite pointer must be a JSON pointer")
    return probs


def new_register():
    return {"format": FORMAT, "claims": []}


def _history_problems(cl):
    probs = []
    hist = cl.get("history")
    if not isinstance(hist, list) or not hist:
        return ["claim %s: history must be a non-empty list" % cl.get("id")]
    prev = None
    for i, h in enumerate(hist):
        where = "claim %s history[%d]" % (cl.get("id"), i)
        if not isinstance(h, dict) or set(h) != set(HISTORY_KEYS):
            probs.append("%s: keys must be %s" % (where, HISTORY_KEYS))
            continue
        if h["from"] != prev:
            probs.append("%s: from %r, previous status %r" % (where, h["from"], prev))
        if i == 0:
            if h["from"] is not None or h["to"] != "open":
                probs.append("%s: a claim is filed open" % where)
        elif (h["from"], h["to"]) not in TRANSITIONS:
            probs.append("%s: %r -> %r is not an allowed transition" % (where, h["from"], h["to"]))
        if not isinstance(h["by"], str) or not h["by"]:
            probs.append("%s: by must name the actor" % where)
        if not isinstance(h["cites"], list):
            probs.append("%s: cites must be a list" % where)
        else:
            for c in h["cites"]:
                probs.extend("%s: %s" % (where, p) for p in cite_problems(c))
        prev = h.get("to")
    if prev != cl.get("status"):
        probs.append("claim %s: status %r, history ends at %r" % (cl.get("id"), cl.get("status"), prev))
    return probs


def validate(claims):
    """Every problem of a funnel-claims/1 register (empty = valid)."""
    if not isinstance(claims, dict):
        return ["the register is not a JSON object"]
    probs = []
    if claims.get("format") != FORMAT:
        probs.append("format %r is not %s" % (claims.get("format"), FORMAT))
    cl = claims.get("claims")
    if not isinstance(cl, list):
        return probs + ["claims must be a list"]
    ids = []
    for c in cl:
        if not isinstance(c, dict):
            probs.append("a claim is not an object")
            continue
        for k in CLAIM_KEYS:
            if k not in c:
                probs.append("claim %s: missing %r" % (c.get("id"), k))
        if not isinstance(c.get("id"), str) or not _ID_RE.match(c.get("id") or ""):
            probs.append("claim id %r is not C<n>" % c.get("id"))
        elif c["id"] in ids:
            probs.append("claim id %s is repeated" % c["id"])
        ids.append(c.get("id"))
        if not isinstance(c.get("text"), str) or not c.get("text", "").strip():
            probs.append("claim %s: text required" % c.get("id"))
        if c.get("polarity") not in POLARITIES:
            probs.append("claim %s: polarity %r not in %s" % (c.get("id"), c.get("polarity"), POLARITIES))
        if not isinstance(c.get("scope"), str):
            probs.append("claim %s: scope must be a string" % c.get("id"))
        if not isinstance(c.get("made_by"), str) or not _MADE_BY_RE.match(c.get("made_by") or ""):
            probs.append("claim %s: made_by %r" % (c.get("id"), c.get("made_by")))
        if c.get("status") not in STATUSES:
            probs.append("claim %s: status %r not in %s" % (c.get("id"), c.get("status"), STATUSES))
        if not isinstance(c.get("cites"), list):
            probs.append("claim %s: cites must be a list" % c.get("id"))
        else:
            for ct in c["cites"]:
                probs.extend("claim %s: %s" % (c.get("id"), p) for p in cite_problems(ct))
        probs.extend(_history_problems(c))
    return probs


def load(path):
    """The register at path; a missing file is an empty register."""
    path = Path(path)
    if not path.exists():
        return new_register()
    try:
        claims = read_json(path)
    except FunnelError as e:
        raise ClaimsError(str(e))
    probs = validate(claims)
    if probs:
        raise ClaimsError("%s: %s" % (path, "; ".join(probs)))
    return claims


def _append_only(old, new):
    old_by = {c["id"]: c for c in old.get("claims", [])}
    new_by = {c["id"]: c for c in new.get("claims", [])}
    for cid, oc in old_by.items():
        nc = new_by.get(cid)
        if nc is None:
            raise ClaimsError("claim %s would be removed; the register is append-only" % cid)
        for k in FILED_KEYS:
            if oc.get(k) != nc.get(k):
                raise ClaimsError("claim %s: filed field %r would change" % (cid, k))
        oh, nh = oc.get("history", []), nc.get("history", [])
        if nh[:len(oh)] != oh:
            raise ClaimsError("claim %s: its history would be rewritten; history is append-only" % cid)


def save(path, claims):
    """Validate, check the on-disk register is a prefix of this one, write
    atomically. Returns the sha256 of the file."""
    probs = validate(claims)
    if probs:
        raise ClaimsError("refusing to save an invalid register: %s" % "; ".join(probs))
    path = Path(path)
    if path.exists():
        _append_only(load(path), claims)
    return _atomic_write_bytes(path, json_text(claims).encode("utf-8"))


def get(claims, claim_id):
    for c in claims.get("claims", []):
        if c.get("id") == claim_id:
            return c
    raise ClaimsError("no claim %r" % (claim_id,))


def file_claim(claims, text, polarity, scope, made_by, cites):
    """Add an open claim; returns its id (C<n>, one above the largest)."""
    if polarity not in POLARITIES:
        raise ClaimsError("polarity %r not in %s" % (polarity, POLARITIES))
    if not isinstance(text, str) or not text.strip():
        raise ClaimsError("a claim needs its text")
    if not isinstance(scope, str):
        raise ClaimsError("scope must be a string")
    if not isinstance(made_by, str) or not _MADE_BY_RE.match(made_by):
        raise ClaimsError("made_by %r is not human-transcribed, card:<id>, tier2:<model> or "
                          "human:<actor>" % (made_by,))
    cites = copy.deepcopy(list(cites or []))
    for c in cites:
        p = cite_problems(c)
        if p:
            raise ClaimsError("; ".join(p))
    if claims.get("format") != FORMAT or not isinstance(claims.get("claims"), list):
        raise ClaimsError("not a funnel-claims/1 register")
    n = 0
    for c in claims["claims"]:
        m = _ID_RE.match(c.get("id") or "")
        if m:
            n = max(n, int(m.group(1)))
    cid = "C%d" % (n + 1)
    claims["claims"].append({
        "id": cid, "text": text, "polarity": polarity, "scope": scope, "made_by": made_by,
        "cites": cites, "status": "open",
        "history": [{"utc": utc(), "from": None, "to": "open", "by": made_by, "reason": "filed",
                     "cites": copy.deepcopy(cites)}]})
    return cid


def _check_adversary(by, cites, proof):
    if not isinstance(proof, dict):
        raise ClaimsError("an adversary transition needs the validation record of the DA reply")
    if proof.get("valid") is not True:
        raise ClaimsError("the DA reply did not validate; it moves no claim")
    surv = proof.get("surviving")
    if not isinstance(surv, int) or isinstance(surv, bool) or surv < 1:
        raise ClaimsError("the DA reply has no surviving counter-argument; it moves no claim")
    model, planner = proof.get("model"), proof.get("planner_model")
    if not model or not planner:
        raise ClaimsError("the DA record must name the resolved adversary and planner models")
    if proof.get("same_family") is True or model_family(model) == model_family(planner):
        raise ClaimsError("the adversary %r is of the planner's family (%r); a same-family reply moves "
                          "no claim" % (model, planner))
    if not cites:
        raise ClaimsError("an adversary transition cites the reply's evidence")


def _check_autopilot(by, cites, proof):
    if not isinstance(proof, dict):
        raise ClaimsError("an autopilot transition needs the audit record")
    sha = proof.get("audit_sha256")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        raise ClaimsError("the autopilot record must carry the audit's sha256")
    if proof.get("valid") is not True:
        raise ClaimsError("the audit is not valid (calibration overlap); nothing may cite it")
    if proof.get("fingerprint_match") is not True:
        raise ClaimsError("the audit was made on another ledger (fingerprint mismatch)")
    if proof.get("d18_fired") is not False:
        raise ClaimsError("D18 fired on the audit; the claim does not survive by the autopilot")
    if not any(isinstance(c, dict) and c.get("value") == sha for c in cites):
        raise ClaimsError("no cite carries the audit's sha256 %s" % sha[:12])


def transition(claims, claim_id, to, by, reason, cites, actor_kind, proof=None):
    """Move a claim along TRANSITIONS; returns the claim. ClaimsError when the
    move, the actor kind or its proof is not allowed."""
    cl = get(claims, claim_id)
    frm = cl.get("status")
    if to not in STATUSES:
        raise ClaimsError("status %r not in %s" % (to, STATUSES))
    if actor_kind not in ACTOR_KINDS:
        raise ClaimsError("actor kind %r not in %s" % (actor_kind, ACTOR_KINDS))
    allowed = TRANSITIONS.get((frm, to))
    if allowed is None:
        raise ClaimsError("claim %s: %s -> %s is not an allowed transition" % (claim_id, frm, to))
    if actor_kind not in allowed:
        raise ClaimsError("claim %s: %s -> %s may be made by %s, not %s"
                          % (claim_id, frm, to, " or ".join(allowed), actor_kind))
    if not isinstance(by, str) or not by:
        raise ClaimsError("by must name the actor")
    if not isinstance(reason, str) or not reason.strip():
        raise ClaimsError("a transition needs its reason")
    cites = copy.deepcopy(list(cites or []))
    for c in cites:
        p = cite_problems(c)
        if p:
            raise ClaimsError("; ".join(p))
    if actor_kind == "adversary":
        _check_adversary(by, cites, proof)
    elif actor_kind == "autopilot":
        _check_autopilot(by, cites, proof)
    else:
        if not by.startswith("human:"):
            raise ClaimsError("a person's transition is signed human:<actor> (card X12), not %r" % by)
    cl["history"].append({"utc": utc(), "from": frm, "to": to, "by": by, "reason": reason,
                          "cites": cites})
    cl["status"] = to
    return cl


def open_negative(claims):
    """Claims of polarity scarcity or negative that are open or challenged."""
    return [c for c in claims.get("claims", [])
            if c.get("status") in ("open", "challenged") and c.get("polarity") in NEGATIVE_POLARITIES]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="funnel.claims", description="The claims register.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--path", required=True)
    p = sub.add_parser("file")
    p.add_argument("--path", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--polarity", required=True, choices=POLARITIES)
    p.add_argument("--scope", required=True)
    p.add_argument("--made-by", required=True)
    p.add_argument("--cite", action="append", default=[], help="a cite as JSON")
    p = sub.add_parser("transition")
    p.add_argument("--path", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--to", required=True, choices=STATUSES)
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--actor-kind", required=True, choices=ACTOR_KINDS)
    p.add_argument("--cite", action="append", default=[], help="a cite as JSON")
    p.add_argument("--proof", default=None, help="the actor's proof record as JSON")
    args = ap.parse_args(argv)
    try:
        claims = load(args.path)
        if args.cmd == "list":
            for c in claims["claims"]:
                print("%s\t%s\t%s\t%s" % (c["id"], c["status"], c["polarity"], c["text"]))
            return 0
        cites = [json.loads(c) for c in args.cite]
        if args.cmd == "file":
            cid = file_claim(claims, args.text, args.polarity, args.scope, args.made_by, cites)
            save(args.path, claims)
            print("[funnel] claims: filed %s" % cid)
            return 0
        proof = json.loads(args.proof) if args.proof else None
        cl = transition(claims, args.id, args.to, args.by, args.reason, cites, args.actor_kind, proof)
        save(args.path, claims)
        print("[funnel] claims: %s -> %s" % (cl["id"], cl["status"]))
        return 0
    except (FunnelError, ValueError) as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
