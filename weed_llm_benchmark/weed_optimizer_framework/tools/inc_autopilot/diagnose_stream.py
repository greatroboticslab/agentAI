"""Stream-mode diagnoses D20-D33 (docs/CONTINUOUS_LOOP.md 6.4), the lanes' own
work items, and the R0 records as the other groups' verbs write them.

Pure functions of the evidence: `detect(ev, dom, th)` reads a dev-only
evidence.Evidence (the stream snapshot's artifacts, the experiments' files,
and the ticker's context, all scrubbed by evidence.scrub) and returns
model.diagnosis records with cites and a `detail`. A diagnosis names the lever
it calls for; `detail.propose` carries the parameters the ticker (stream.py)
renders it with. Nothing here submits, writes or reads a file outside the
evidence, except the pinned prior ledger the stream-domain config names by
path and sha256 (D30's prior before Stage A is READY).

One implementation of each rule: the stream's dispositions, the Stage B
choice, the milestone test, the capacity switch, Stage A's survival and Stage
C's reading are inc2's (groups B and E); the diagnoses read what those record
(the commit lines of the stream ledger, queue_summary.json, the verdict
files) and apply the thresholds 6.4 pre-registers to the recorded numbers. The
guard-based disposition of 3.5 (`disposition`) is applied here only to the
pinned prior's gate decisions.

Thresholds: stream_thresholds.json (levers_stream.load_thresholds), each with
its reason. No compiled fallback: a missing key makes the rule 'unknown'.

Domain-free (contract S13): no class, source, lab, exam or domain name is
written here; species names, source ids and exam names arrive as data.

Mutation harness (tests/test_stream_ap_mutations.py): each comparison marked
'# stream-mutation: SMn' is an 'if <condition>:' line; switching it off must
make some S-case of tests/test_stream_ap_replay.py fail.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from . import evidence as E
from . import levers_stream as LS
from . import model as M

NAMES = {"D20": "data_needed", "D21": "source_low_yield", "D22": "cut_ready", "D23": "segment_finished",
         "D24": "milestone_due", "D25": "stream_regressed", "D26": "walltime_bound", "D27": "resource_reserve",
         "D28": "source_leak", "D29": "collection_exhausted", "D30": "recipe_blocks_stream",
         "D31": "data_blamed", "D32": "stream_not_accepting", "D33": "rare_species_guard",
         "D8S": "gate_underpowered", "D10S": "budget_exhausted",
         "DSA": "stage_a", "DSC": "stage_c", "DCAP": "capacity_decision", "DCAN": "canary_failed",
         "DR0": "r0_due", "DCMP": "compare_due", "DPIPE": "data_pipeline", "DHOLD": "hold_deadline",
         "DBIS": "bisect_due", "DKT": "known_truth_precision", "DNAT": "native_verdict"}
RULES_FILES = ("diagnose_stream.py", "stream_thresholds.json", "stream_levers.json", "levers_stream.py")
RULES_VERSION_HEX = 12
EPS = 1e-12
DISPOSITIONS = ("accepted", "data", "species", "flips", "recipe", "hold")
_SELF_BYTES = Path(__file__).read_bytes()


class _Missing(Exception):
    pass


# ------------------------------------------------------------------ plumbing
def rules_files():
    here = Path(__file__).resolve().parent
    return [("diagnose_stream.py", _SELF_BYTES), ("stream_thresholds.json", (here / "stream_thresholds.json").read_bytes()),
            ("stream_levers.json", (here / "stream_levers.json").read_bytes()),
            ("levers_stream.py", (here / "levers_stream.py").read_bytes())]


def rules_version():
    """The stream rules version: the first 12 hex of the sha256 of
    diagnose_stream.py + stream_thresholds.json + stream_levers.json +
    levers_stream.py, in that order."""
    h = hashlib.sha256()
    for _n, raw in rules_files():
        h.update(raw)
    return h.hexdigest()[:RULES_VERSION_HEX]


def _t(th, block, key):
    try:
        return LS.t(th, block, key)
    except LS.Missing as e:
        raise _Missing(str(e))


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _utc(s):
    try:
        return datetime.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        return None


def _days(a, b):
    ta, tb = _utc(a), _utc(b)
    if ta is None or tb is None:
        return None
    return (ta - tb).total_seconds() / 86400.0


def _diag(did, fired, sev, summary, cites, levers=(), exp=None, detail=None):
    cites = [c for c in cites or [] if c]
    d = M.diagnosis(did, NAMES[did], fired, sev, summary, cites, levers, exp)
    d["detail"] = detail or {}
    return d


def _silent(did, why, exp=None, cites=()):
    return _diag(did, False, "info", why, list(cites), [], exp)


def _unknown(did, why, exp=None):
    d = _silent(did, "unknown: " + why, exp)
    d["unknown"] = True
    return d


class View(object):
    """Addressed reads of the stream evidence. Every value a diagnosis uses
    comes through `cite`, so it is recorded with its address."""

    def __init__(self, ev, dom, th):
        self.ev, self.dom, self.th = ev, dom, th
        self.ctx = ev.json(E.CONTEXT) or {}
        self.sid = str(self.ctx.get("sid") or dom.get("sid"))

    def cite(self, name, ptr):
        return self.ev.cite(name, ptr)

    def ccite(self, ptr):
        """A cite into the ticker's context, or None when it lacks the value."""
        try:
            return self.ev.cite(E.CONTEXT, ptr)
        except KeyError:
            return None

    def get(self, name, ptr, default=None):
        return self.ev.get(name, ptr, default)

    def c(self, ptr, default=None):
        return self.ev.get(E.CONTEXT, ptr, default)

    @property
    def qname(self):
        return "stream/%s/queue_summary.json" % self.sid

    @property
    def lname(self):
        return "stream/%s/ledger.jsonl" % self.sid

    def queue(self):
        return self.ev.json(self.qname)

    def stream_ledger(self):
        led = self.ev.json(self.lname)
        return [e for e in led if isinstance(e, dict)] if isinstance(led, list) else []

    def M(self):
        return int(self.c("/M") or (self.dom.get("increment") or {}).get("M") or 0)

    def Q(self):
        """(Q, cite): eligible target images, from the queue summary when there
        is one, else the context's count (0 before any queue exists)."""
        q = self.queue()
        if isinstance(q, dict) and _num((q.get("eligible") or {}).get("images")) is not None:
            return int(q["eligible"]["images"]), self.cite(self.qname, "/eligible/images")
        return int(self.c("/queue/Q") or 0), self.ccite("/queue/Q")

    def lane(self, name):
        return self.c("/lanes/%s" % name) or {}

    def lane_idle(self, name):
        ln = self.lane(name)
        return not ln.get("busy") and not ln.get("hold")


# ------------------------------------------------------------- dispositions
def _failed_guards(decision):
    return sorted(g for g, v in ((decision or {}).get("guards") or {}).items()
                  if isinstance(v, dict) and v.get("passed") is False)


def disposition(entry, truth_verdict=None, p_reject=0.25):
    """The guard-based disposition of one gate entry (contract 3.5 review).

    ACCEPT -> 'accepted'. Otherwise the first that applies:
      1. 'data'    P_data <= p_reject (the decision's own config.p_reject when
                   recorded), unless the step's truth arm says 'helps';
      2. 'species' the species guard failed;
      3. 'flips'   the flips guard failed;
      4. 'recipe'  only the regression guard failed;
    and 'hold' for a HOLD (no guard failed, P_data in the gate's band).
    attribution.blame is never read: the gate sets 'recipe' on every REJECT
    whose P_recipe <= 0.25, which is every full-recipe step recorded so far."""
    d = (entry or {}).get("decision") or {}
    verdict = d.get("verdict")
    if verdict == "ACCEPT":
        return "accepted"
    pr = _num(((d.get("config") or {}).get("p_reject")))
    pr = p_reject if pr is None else pr
    p_data = _num(d.get("p_data"))
    failed = _failed_guards(d)
    if p_data is not None and p_data <= pr + EPS and truth_verdict != "helps":  # stream-mutation: SM16
        return "data"
    if "species" in failed:  # stream-mutation: SM17
        return "species"
    if "flips" in failed:
        return "flips"
    if failed == ["regression"]:
        return "recipe"
    if verdict == "HOLD" or not failed:
        return "hold"
    return "recipe"


def steps(ev, exp, chain=None, p_reject=0.25):
    """[step record] of an experiment's gate decisions (the latest entry per
    id), in step order, for `chain` (None: every chain). A record carries the
    ledger line, verdict, P_data, P_recipe, failed guards, species failed,
    sources, the step's truth verdict and the disposition."""
    truth = {}
    for _id, (ln, e) in ev.latest(exp, "truth").items():
        det = e.get("detail") if isinstance(e.get("detail"), dict) else {}
        truth[e.get("step")] = (ln, det.get("verdict"))
    out = []
    for _id, (ln, e) in sorted(ev.latest(exp, "gate").items(), key=lambda kv: (kv[1][1].get("k") or 0,
                                                                               str(kv[1][1].get("chain")))):
        if chain is not None and e.get("chain") != chain:
            continue
        d = e.get("decision") or {}
        tv = truth.get(e.get("step"))
        rec = {"exp": exp, "line": ln, "step": e.get("step"), "k": e.get("k"), "chain": e.get("chain"),
               "clean": e.get("clean"), "verdict": d.get("verdict"), "p_data": _num(d.get("p_data")),
               "p_recipe": _num(d.get("p_recipe")), "failed": _failed_guards(d),
               "species_failed": list(((d.get("attribution") or {}).get("species_failed")) or
                                      (((d.get("guards") or {}).get("species") or {}).get("failed")) or []),
               "sources": list(e.get("sources") or []), "truth": tv[1] if tv else None,
               "truth_line": tv[0] if tv else None, "blame": (d.get("attribution") or {}).get("blame"),
               "cand_mean": _num(d.get("cand_mean")), "null_mean": _num(d.get("null_mean")),
               "cand_sd": _num(d.get("cand_sd")), "null_sd": _num(d.get("null_sd"))}
        rec["disposition"] = disposition(e, rec["truth"], p_reject)
        out.append(rec)
    return out


def step_cites(ev, r):
    """The cites of one step record: the gate line's verdict, P_data, guards,
    and the truth line's verdict."""
    exp, ln = r["exp"], r["line"]
    out = [ev.lcite(ln, "/decision/verdict", exp), ev.lcite(ln, "/decision/p_data", exp)]
    try:
        out.append(ev.lcite(ln, "/decision/guards", exp))
    except KeyError:
        pass
    if r.get("truth_line"):
        out.append(ev.lcite(r["truth_line"], "/detail/verdict", exp))
    return out


def _prior_ledger(dom):
    """(rows, why) of the pinned prior ledger the domain config names, checked
    against its recorded sha256; ([], why) when it cannot be read as pinned."""
    pr = dom.get("prior") or {}
    rel = pr.get("ledger")
    if not rel:
        return [], "no prior ledger in the stream-domain config"
    p = Path(rel)
    if not p.is_absolute():
        p = LS.CODE_ROOT / rel
    try:
        raw = p.read_bytes()
    except OSError as e:
        return [], "the prior ledger %s cannot be read (%s)" % (rel, type(e).__name__)
    if hashlib.sha256(raw).hexdigest() != pr.get("sha256"):
        return [], "the prior ledger %s does not hash to its pinned sha256" % rel
    return raw.decode("utf-8"), ""


def prior_evidence(dom):
    """Evidence of the pinned prior experiment (its ledger only), or None."""
    text, why = _prior_ledger(dom)
    if not text:
        return None
    exp = str((dom.get("prior") or {}).get("exp") or "prior")
    return E.from_texts({"%s/ledger.jsonl" % exp: text}, exp, domain=LS.funnel_ref(dom))


# ------------------------------------------------------------ the segments
def _cut_sources(v):
    """{increment: [source]} from the stream ledger's cut lines."""
    out = {}
    for e in v.stream_ledger():
        if e.get("event") == "cut" and e.get("increment"):
            src = e.get("sources")
            out[str(e["increment"])] = sorted(src) if isinstance(src, (dict, list)) else []
    return out


def stream_commits(v):
    """[(ledger index, commit line)] of the stream's committed segments whose
    base was still the current pool (a stale commit returns its images
    uncounted and decides nothing), oldest first."""
    return [(i, e) for i, e in enumerate(v.stream_ledger())
            if e.get("event") == "commit" and e.get("exp") and not e.get("stale_base")]


def commit_rows(v, i, e):
    """The step records of one commit line (inc2.stream commit, contract 3.5):
    the chosen chain's steps under Protocol v3 (L-3), with the disposition the
    commit recorded by the guard rule, each value cited at its address in the
    stream ledger. attribution.blame is carried, never read."""
    chain = e.get("chosen") or e.get("recipe")
    steps_ = ((e.get("steps") or {}).get(chain)) or []
    srcs = _cut_sources(v)
    out = []
    for j, st in enumerate(steps_):
        if not isinstance(st, dict):
            continue
        v3 = st.get("v3") or {}
        base = "/%d/steps/%s/%d" % (i, chain, j)
        g = v3.get("guards") or {}
        rec = {"exp": e.get("exp"), "line": i, "step": st.get("increment"), "k": st.get("k"), "chain": chain,
               "clean": False, "verdict": v3.get("verdict"), "p_data": _num(v3.get("p_data")),
               "p_recipe": _num(v3.get("p_recipe")), "failed": sorted(k for k, ok in g.items() if ok is False),
               "species_failed": list(v3.get("species_failed") or []),
               "sources": srcs.get(str(st.get("increment")), []), "truth": st.get("truth"), "truth_line": None,
               "blame": v3.get("blame"), "cand_mean": _num(v3.get("cand_mean")),
               "null_mean": _num(v3.get("null_mean")), "cand_sd": None, "null_sd": _num(v3.get("null_sd")),
               "disposition": st.get("disposition"), "v3_applied": v3.get("v3_applied")}
        rec["cites"] = [v.cite(v.lname, base + "/v3/verdict"), v.cite(v.lname, base + "/disposition")]
        for ptr in ("/v3/p_data", "/v3/guards"):
            try:
                rec["cites"].append(v.cite(v.lname, base + ptr))
            except KeyError:
                pass
        out.append(rec)
    return out


def _row_cites(ev, r):
    return list(r["cites"]) if r.get("cites") else step_cites(ev, r)


# ------------------------------------------------------------ D20 and D29
def _deficit(v):
    """([species], cites): a species has fewer queued target boxes than the
    last D20.deficit_segments segments consumed (queue summary)."""
    q = v.queue()
    if not isinstance(q, dict):
        return [], []
    queued = (q.get("eligible") or {}).get("target_boxes") or {}
    used = q.get("consumed_last") or {}
    out, cites = [], []
    for sp in sorted(used):
        u, h = _num(used.get(sp)), _num(queued.get(sp)) or 0.0
        if u is not None and u > 0 and h < u:
            out.append(sp)
            cites.append(v.cite(v.qname, E.pointer("consumed_last", sp)))
    return out, cites


def _priority_classes(v, deficit):
    pri = ((v.dom.get("species_priority") or {}).get("value")) or {}
    first = sorted(s for s, p in pri.items() if p == 1)
    bonus = list(v.c("/d33_species") or [])
    return sorted(set(deficit) | set(first) | set(bonus))


def mid_shards(s):
    """A source part-way through the continuation shards of its intake
    (collect.intake step 1b, amendment 2026-10-03): it has an admitted batch,
    and its latest batch left images deferred or a committed batch is not
    admitted yet. Read from these facts, never from the status alone, so a
    detour of the status (a failed shard, the collector's hold and release, a
    person's reopening) ends with the source waiting for its next shard
    (shard_pending), never a candidate that D20 would fetch anew."""
    if not isinstance(s, dict) or not s.get("admitted_batches"):
        return False
    if s.get("shards_done") and s.get("shards_done") == s.get("batch"):
        return False
    ab = set(s["admitted_batches"])
    return int(_num(s.get("intake_deferred")) or 0) > 0 or any(b not in ab for b in s.get("batches") or [])


def precheck(v, cand):
    """(refuse, review): refuse [] or the reasons the source is never fetched
    (never-train, quarantined, closed); review [] or the reasons a person
    decides (licence unknown, no target class, image-level labels only, over
    the byte cap, missing credentials, an evaluation lab without the copy
    scan). S1: the automatic L16 takes only a candidate with neither."""
    refuse, review = [], []
    if cand.get("never_train"):
        refuse.append("in the never-train list")
    # a source mid-pipeline (being fetched, fetched, intaken, or waiting for
    # its next intake shard) takes its next pipeline step first; an admitted
    # one is collected again only when its last fetch stopped at a byte cap
    # (partial: the next shard)
    if cand.get("status") in ("quarantined", "closed", "held", "fetching", "fetched", "names_pending", "intaken",
                              "shard_pending") or (cand.get("status") == "admitted" and not cand.get("partial")):
        refuse.append("source status %s" % cand.get("status"))
    elif cand.get("in_shards"):
        refuse.append("part-way through its intake shards (the next shard, never a new fetch)")
    lic = str(cand.get("licence") or "").strip().lower()
    if lic in ("", "unknown", "unresolved", "none"):
        review.append("licence unknown")
    elif cand.get("licence_ok") is False:
        refuse.append("licence not research-usable")
    # a source whose class names are pending resolves them first (L26, the
    # stream's own resolver); once they are resolved, the collector's fetch
    # decides again on the fresh names layer (and refuses a source that still
    # declares no target), so an empty class list read before L26 is not a
    # reason to ask a person
    if not cand.get("target_classes") and not cand.get("known_item") and not cand.get("names_unresolved") \
            and not cand.get("names_resolved"):
        review.append("no target class in its card or the taxonomy")
    if cand.get("image_level"):
        review.append("image-level labels only")
    cap = (v.c("/limits/gb_per_source") or 0) * 1e9
    if cap and (_num(cand.get("bytes")) or 0) > cap:
        review.append("over the %d GB per-source cap" % int(cap / 1e9))
    if cand.get("credentials") is False:
        review.append("missing credentials (card X16)")
    # a partial source whose attempts are used up (6.6: <= 3 per source) still
    # has shards: a person approves more (L16R) -- the automatic L16 never
    # proposes the 4th attempt, which would pause the whole campaign (S15)
    lim_att = v.c("/limits/attempts_per_source")
    if cand.get("partial") and cand.get("status") == "admitted" and lim_att \
            and int(cand.get("attempts") or 0) >= int(lim_att):
        review.append("%d collection attempts used with shards remaining: a person approves more"
                      % int(cand.get("attempts") or 0))
    evl = ((v.dom.get("eval_lab_groups") or {}).get("value")) or []
    if (cand.get("lab_group") in evl or cand.get("evaluation_lab")) and not cand.get("copy_scan_done"):
        review.append("a source of an evaluation lab without the copy scan")
    # the collector's own pre-check (collect.prefilter.precheck), defence in depth
    cp = cand.get("collector_precheck") or {}
    refuse += ["collector: %s" % c for c in cp.get("refuse") or []]
    refuse += ["waits: %s" % c for c in cp.get("wait") or []]
    review += ["collector: %s" % c for c in cp.get("review") or [] if "collector: %s" % c not in review]
    return refuse, review


def rank(v, cands, deficit):
    """[(score, candidate)] best first: expected verified target boxes per GB,
    times (1 + D20.deficit_bonus) for a candidate declaring a deficit species."""
    rate = float(_t(v.th, "D20", "verify_rate_default"))
    bonus = float(_t(v.th, "D20", "deficit_bonus"))
    out = []
    for c in cands:
        exp_boxes = _num(c.get("expected_target_boxes"))
        if exp_boxes is None:
            tb = _num(c.get("target_boxes"))
            exp_boxes = tb * rate if tb is not None else (_num(c.get("images")) or 0.0) * rate
        gb = max((_num(c.get("bytes")) or 0.0) / 1e9, 0.01)
        s = exp_boxes / gb
        if set(c.get("target_classes") or []) & set(deficit):
            s *= (1.0 + bonus)
        out.append((round(s, 9), c))
    out.sort(key=lambda sc: (-sc[0], str(sc[1].get("id"))))
    return out


def d20(v):
    th = v.th
    Mv = v.M()
    Q, qc = v.Q()
    low = int(_t(th, "D20", "low_water_M_mult")) * Mv
    cites = [qc, v.ccite("/M"), v.ccite("/lanes/DATA")]
    if not v.lane_idle("DATA"):
        return _silent("D20", "the DATA lane is busy or held", cites=cites)
    if not v.c("/stage/lock"):
        return _silent("D20", "splits v2 are not locked: intake needs the never-train v2 index (R0 first)",
                       cites=cites)
    s1 = v.c("/stage/step1_stream") or {}
    r1 = [k for k in ("bootstrap", "knowntruth", "backfill") if not s1.get(k)]
    if r1:
        # contract 10: collection (R3) starts once R1 passes. D20 comes before DR0
        # in the diagnosis order, so without this the DATA lane (one item) would
        # take discovery or a fetch before Step 1's one-time jobs (DR0 -> L17)
        return _silent("D20", "Step 1's one-time jobs (R1) have not run (%s): collection waits for them"
                       % ", ".join(r1), cites=cites + [v.ccite("/stage/step1_stream")])
    # DR0's Step 1 item on the DATA lane (L17 eval-hits) comes before a new source, as R1's jobs do: D20 comes
    # before DR0 in the diagnosis order, so without this a fetch would take the one DATA item whenever Q is low,
    # and eval-hits, with E1's base v3 (proposed only when DR0 has no DATA item due) and every intake shard behind
    # it, would wait for as long as there are candidates (2026-10-03: eval-hits due from 06:56Z, D20 took the
    # DATA lane three times)
    try:
        d0 = r0(v)
    except Exception:  # noqa: BLE001 - an R0 record that cannot be read holds collection (fail closed), as shards
        return _silent("D20", "DR0 cannot be read: collection waits", cites=cites + [v.ccite("/stage")])
    due0 = ((d0.get("detail") or {}).get("due") or {}).get("DATA") if d0.get("fired") else None
    if due0 and due0.get("lever") == "L17":
        return _silent("D20", "DR0's L17 %s is due on the DATA lane: collection waits for it" % due0.get("verb"),
                       cites=cites + [v.ccite("/stage")])
    deficit, dc = _deficit(v)
    refusal = [r for r in (v.c("/refusals") or []) if "eligible images against" in str((r or {}).get("message"))]
    low_water = False
    if Q < low:  # stream-mutation: SM1
        low_water = True
    if not low_water and not deficit and not refusal:
        return _silent("D20", "Q %d >= %d (2M) and no species in deficit" % (Q, low), cites=cites)
    cites += dc
    cands = [c for c in (v.c("/candidates") or []) if isinstance(c, dict)]
    ok, review = [], []
    for i, c in enumerate(cands):
        ref, rev = precheck(v, c)
        if ref:
            continue
        if rev:
            review.append({"source": c.get("id"), "reasons": rev, "index": i})
            continue
        ok.append(c)
    classes = _priority_classes(v, deficit)
    ranked = rank(v, ok, classes)
    why = ("Q %d < %d (2M)" % (Q, low)) if low_water else ("species in deficit: %s" % ", ".join(deficit)) \
        if deficit else "a builder refusal asks for data"
    detail = {"Q": Q, "low_water": low, "deficit": deficit, "classes": classes,
              "ranked": [{"source": c.get("id"), "score": s} for s, c in ranked[:10]], "review": review}
    if ranked:
        top = ranked[0][1]
        idx = cands.index(top)
        cites.append(v.ccite("/candidates/%d/id" % idx))
        if top.get("names_unresolved"):
            # a new source's class names first (L26, the stream's own resolver;
            # it never writes the funnel's directory)
            detail["propose"] = {"lever": "L26", "source": top.get("id")}
            return _diag("D20", True, "info", "%s -> L26: resolve %s's class names before collecting it" % (
                why, top.get("id")), cites, ["L26"], None, detail)
        detail["propose"] = {"lever": "L16", "source": top.get("id"), "placement": top.get("placement") or "cluster",
                             "bytes": top.get("bytes"), "known_item": bool(top.get("known_item"))}
        return _diag("D20", True, "info", "%s -> L16 on %s (rank %.4g)" % (why, top.get("id"), ranked[0][0]),
                     cites, ["L16"], None, detail)
    last = v.c("/discover/last_utc")
    age = _days(v.c("/now_utc"), last) if last else None
    after = float(_t(th, "D20", "discover_after_days"))
    cites.append(v.ccite("/discover"))
    if last is None or (age is not None and age > after):
        detail["propose"] = {"lever": "L15", "classes": classes}
        return _diag("D20", True, "info", "%s, no open candidate; the last discovery is %s -> L15 for %s"
                     % (why, "never run" if last is None else "%.1f days old" % age, ", ".join(classes)),
                     cites, ["L15"], None, detail)
    return _diag("D20", True, "info", "%s, no open candidate, discovery ran %.1f days ago (D29 decides)" % (why, age),
                 cites, [], None, detail)


def d29(v):
    th = v.th
    recent = float(_t(th, "D29", "recent_days"))
    backoff = list(_t(th, "D29", "backoff_days"))
    d = v.c("/discover") or {}
    last, found = d.get("last_utc"), d.get("found_new")
    cites = [v.ccite("/discover")]
    if not last:
        return _silent("D29", "discovery has not run", cites=cites)
    pend = sorted(s for s, r in (v.c("/sources") or {}).items() if (r or {}).get("status") == "shard_pending"
                  or ((r or {}).get("status") not in ("closed", "quarantined") and mid_shards(r)))
    if pend:
        # a source whose intake left images deferred still has data to give (its next shards): not exhausted
        return _silent("D29", "%s wait%s for the next intake shard" % (", ".join(pend), "s" if len(pend) == 1
                                                                        else ""), cites=cites)
    age = _days(v.c("/now_utc"), last)
    open_c = [c for c in (v.c("/candidates") or []) if isinstance(c, dict) and not precheck(v, c)[0]
              and not precheck(v, c)[1]]
    if open_c:
        return _silent("D29", "%d open candidate(s)" % len(open_c), cites=cites)
    if age is not None and age <= recent and found == 0:  # stream-mutation: SM11
        n = int(d.get("empty_runs") or 1)
        wait = backoff[min(n, len(backoff)) - 1]
        return _diag("D29", True, "warn", "no open candidate and the discovery %.1f days ago found nothing new -> "
                     "WAIT_DATA, discovery again in %d days (never COMPLETE)" % (age, wait),
                     cites, ["OP_WAIT_DATA"], None, {"wait_days": wait, "empty_runs": n})
    return _silent("D29", "no open candidate, but the last discovery is not an empty one within %g days" % recent,
                   cites=cites)


# ------------------------------------------------------------------ D21, D28
def d21(v):
    th = v.th
    fgb, fsu = _t(th, "D21", "floor_gb"), _t(th, "D21", "floor_su")
    min_gb = float(_t(th, "D21", "min_gb"))
    blames = int(_t(th, "D31", "blames_to_quarantine"))
    srcs = v.c("/sources") or {}
    close, cites = [], []
    for s in sorted(srcs):
        r = srcs[s] or {}
        if r.get("status") in ("closed", "quarantined"):
            continue
        if int(r.get("blames") or 0) >= blames:
            close.append({"source": s, "why": "blamed for data %d times (D31)" % r["blames"]})
            cites.append(v.ccite(E.pointer("sources", s, "blames")))
            continue
        if fgb is None or fsu is None:
            continue
        # the yield is verified target boxes ADMITTED (7.4): judged only once the
        # admission of the source's fetched data has been observed; a source
        # still being fetched, intaken or admitted has admitted nothing yet, and
        # judging it then would close every source at its first fetch. A source
        # whose intake left images deferred (continuation shards, amendment
        # 2026-10-03) is judged once its last shard is admitted: its bytes are
        # all fetched, its boxes only partly admitted
        if r.get("status") != "admitted" or not r.get("yield_recorded") or deferred_left(v, r) != 0:
            continue
        got = (_num(r.get("bytes")) or 0.0) / 1e9
        total = (_num(r.get("total_bytes")) or 0.0) / 1e9 or got
        if got < min(min_gb, total) - EPS or got <= 0:
            continue
        boxes = _num(r.get("admitted_target_boxes")) or 0.0
        su = _num(r.get("su"))
        y_gb = boxes / got
        y_su = boxes / su if su else None
        if y_gb < float(fgb) or (y_su is not None and y_su < float(fsu)):  # stream-mutation: SM2
            close.append({"source": s, "yield_gb": y_gb, "yield_su": y_su,
                          "why": "yield %.4g boxes/GB (floor %g), %s boxes/SU (floor %g)"
                                 % (y_gb, fgb, "n/a" if y_su is None else "%.4g" % y_su, fsu)})
            cites += [v.ccite(E.pointer("sources", s, "admitted_target_boxes")),
                      v.ccite(E.pointer("sources", s, "bytes"))]
    if not close:
        why = "floors not set (placeholders): not evaluated" if (fgb is None or fsu is None) else "no source below the floors"
        return _silent("D21", why)
    return _diag("D21", True, "warn", "close %s" % ", ".join("%s (%s)" % (c["source"], c["why"]) for c in close),
                 cites, ["OP_CLOSE_SOURCE"], None, {"close": close})


# D28's guard reasons (inc2.guard.REASONS): a dHash copy of an evaluation image
# (within 6 bits, the image's own dHash or one of its 8 flips and rotations)
# and an embedding hit (the calibrated copy detector).
D28_DHASH_REASONS = ("near_eval_v2", "near_eval_variant")
D28_EMBED_REASON = "near_eval_embed"
def _d28_pair_cos(values):
    """The pair cosines a producer recorded (a list), or None without a list.
    A value that is not a number is dropped, so its hit counts as unscored
    (fail closed)."""
    if not isinstance(values, list):
        return None
    return [c for c in (_num(x) for x in values) if c is not None]


def _d28_counts(s):
    """(images checked, dHash hits, embedding hits, counted) of an intake
    batch summary (collect.intake: the flat "images" the guard checked and its
    refusals per reason, "guard"). counted is False when the summary holds no
    guard counts."""
    g = s.get("guard")
    counted = isinstance(g, dict)
    g = g if counted else {}
    dh = sum(int(_num(g.get(k)) or 0) for k in D28_DHASH_REASONS)
    return int(_num(s.get("images")) or 0), dh, int(_num(g.get(D28_EMBED_REASON)) or 0), counted


def _d28_batch_cos(v, name, s, src, dh, counted):
    """(pair cosines or None, the producer's copy threshold or None, the
    artifact they come from) of one intake batch's dHash hits (D28-v2): the
    summary's own record (eval_hits), or, when that weighs fewer than the
    batch's hits (a batch committed before the amendment), its sidecar
    intake/<batch>/eval_hits.json (collect.intake.rescore_eval_hits) when the
    sidecar weighed exactly the hits the summary counts
    (inc2.eval_hits.usable_sidecar). None: nothing weighs them, and the
    one-hit rule reads them (fail closed)."""
    from ..inc2 import eval_hits as EH
    eh = s.get("eval_hits") if isinstance(s.get("eval_hits"), dict) else {}
    per = eh.get("per_source") if isinstance(eh.get("per_source"), dict) else {}
    row = per.get(src)
    cos = _d28_pair_cos(row.get("pair_cos")) if counted and isinstance(row, dict) else None
    rec_t, used = _num(eh.get("copy_threshold")), name
    if counted and dh and (cos is None or len(cos) < dh):
        side = name[:-len("summary.json")] + "eval_hits.json"
        rec, _why = EH.usable_sidecar(v.ev.json(side), name.split("/")[1], {src: dh})
        srow = (rec.get("per_source") or {}).get(src) if rec else None
        scos = _d28_pair_cos(srow.get("pair_cos")) if isinstance(srow, dict) else None
        if scos is not None and len(scos) > len(cos or ()):
            cos, rec_t, used = scos, _num(rec.get("copy_threshold")), side
    return cos, rec_t, used


def _d28_intake_sources(v, lock_p):
    """{source: what its intake batches say together} (D28-v2: a source is
    judged over all its batches, so splitting it into shards never escapes the
    binomial rule): the images, dHash and embedding hits summed; the pair
    cosines of every batch (each batch's at most as many as its hits, so a
    batch weighing fewer than its hits leaves the source short of pair
    cosines: fail closed); the lowest copy threshold a producer recorded; the
    per-image false-positive rate of the embedding rule (the lowest of the
    batches with embedding hits, None when one of them has none: fail
    closed); the highest base-copy share of any batch (a share is a batch's,
    and the highest is never less strict than their mean)."""
    out = {}
    for name in sorted(v.ev.artifacts):
        if not (name.startswith("intake/") and name.endswith("/summary.json")):
            continue
        s = v.ev.json(name) or {}
        leak = s.get("source_leak") or {}
        ev_share, base_share = _num(leak.get("eval_share")), _num(leak.get("base_share"))
        n, dh, emb, counted = _d28_counts(s)
        if ev_share is None and base_share is None and not counted:
            continue
        cs = s.get("copy_scan") if isinstance(s.get("copy_scan"), dict) else {}
        p = _num(cs.get("p_false")) if cs.get("checked") else None
        p = p if p is not None else lock_p
        if not counted and (ev_share or 0.0) > 0:
            # never-train refusals without their reasons: a dHash copy cannot be ruled out (fail closed)
            dh = max(dh, 1)
        src = str(s.get("source"))
        cos, rec_t, used = _d28_batch_cos(v, name, s, src, dh, counted)
        a = out.setdefault(src, {"source": s.get("source"), "names": [], "sidecars": [], "images": 0, "dhash": 0,
                                 "embed": 0, "cos": None, "thresholds": [], "p_embed": [], "p_any": [],
                                 "p_missing": False, "eval_refused": None, "base_share": None})
        a["names"].append(name)
        a["images"] += n
        a["dhash"] += dh
        a["embed"] += emb
        if dh and cos is not None:
            a["cos"] = (a["cos"] or []) + sorted(cos, reverse=True)[:dh]
        if used != name:
            a["sidecars"].append(used)
        if rec_t is not None:
            a["thresholds"].append(rec_t)
        if emb and p is None:
            a["p_missing"] = True
        elif emb:
            a["p_embed"].append(p)
        if p is not None:
            a["p_any"].append(p)
        if ev_share is not None:
            a["eval_refused"] = (a["eval_refused"] or 0.0) + ev_share * n
        if base_share is not None:
            a["base_share"] = max(a["base_share"] or 0.0, base_share)
    for a in out.values():
        ps = a["p_embed"] or a["p_any"]
        a["p_false"] = None if a["p_missing"] else (min(ps) if ps else lock_p)
        a["copy_threshold"] = min(a["thresholds"]) if a["thresholds"] else None
        a["never_train_share"] = (None if a["eval_refused"] is None else
                                  round(a["eval_refused"] / float(a["images"]), 4) if a["images"] else 0.0)
        a["batches"] = [nm.split("/")[1] for nm in a["names"]]
    return out


def d28(v):
    """D28 source_leak (contract 6.4, amended 2026-09-29 by decision L-9 and the
    funnel's amendment A2, and 2026-10-03 by D28-v2, amendment V3-1). A source
    leaks when
      * its dHash hits say so (inc2.eval_hits.verdict; a hit is an image within
        6 bits of an evaluation image under any of the 8 variants, which
        GuardV2 drops on its own): one hit's pair cosine with its matched
        evaluation image reaches the v2 calibration's cos_threshold, or its
        confirmed hits (pair cosine >= confirm_cos) are improbable,
        P(Binom(images, p_confirmed) >= confirmed) < source_alpha. A hit
        without a pair cosine falls back to the old rule, one hit is a leak
        (fail closed). The pair cosines are the producers' records: the
        intake summary's "eval_hits" (or, for a batch committed before the
        amendment, its sidecar intake/<batch>/eval_hits.json) and Step 1's
        per-source "eval_hit_pair_cos" (status.json folds Step 1's sidecars
        too). A source is judged over all its intake batches together
        (_d28_intake_sources): a source split into shards is one source;
      * or it has more embedding hits than the copy detector's per-image
        false-positive rate predicts (inc2.embed_calibration.source_verdict:
        P(Binom(images, p_false) >= hits) < source_alpha);
      * or its base-copy share reaches base_copy_share.
    A share of never-train refusals alone is no rule: at a per-image rate p, a
    source of n images holds about n p chance hits. p_false is the one the
    batch was judged under (its copy_scan record), else the splits LOCK's v2
    calibration (splits/v2/lock_status.json); with neither, any embedding hit
    flags (fail closed). The copy threshold is the lower of the producer's
    record and the LOCK's v2 cos_threshold; with neither, confirm_cos stands
    in (never less strict). The summary gives, for every source with dHash
    hits, its hits, confirmed hits, max pair cosine, P and verdict. A source
    has one verdict: one that leaks in its intake batches or in Step 1 is
    never also judged chance (detail.cleared, detail.cleared_quarantined),
    and one judged chance in both is stated once, with each path's
    numbers. detail.lift_pending names the sources judged chance that the
    stream itself still quarantines on D28's word (its queue summary's
    quarantined_sources, the ledger fold inc2.base3 reads, with cite D28
    or no cite recorded; a quarantine D31 or a person cited is not D28's to
    reconsider): DR0 waits a bounded time for a person to lift them before
    base v3 is built. It is None (unknown, never an empty list) when the
    queue summary cannot be read or does not state quarantined_sources."""
    from ..inc2 import embed_calibration as EC
    from ..inc2 import eval_hits as EH
    th = v.th
    alpha, bc = float(_t(th, "D28", "source_alpha")), float(_t(th, "D28", "base_copy_share"))
    confirm, p_conf = float(_t(th, "D28", "confirm_cos")), float(_t(th, "D28", "p_confirmed"))
    lock = v.ev.json("splits/v2/lock_status.json") or {}
    lec = (lock.get("embed_calibration_v2") or {}) if isinstance(lock, dict) else {}
    lec = lec if isinstance(lec, dict) else {}
    lock_p, lock_t = _num(lec.get("p_false")), _num(lec.get("cos_threshold"))

    def judge(n, dh, emb, p, cos, rec_t):
        ts = [t for t in (rec_t, lock_t) if t is not None]
        dv = EH.verdict(n, dh, cos, confirm, min(ts) if ts else None, p_conf, alpha)
        ev = EC.source_verdict(n, emb, 0, 0, p, alpha=alpha)
        return dict(ev, dhash_hits=int(dh), dhash=dv, flagged=bool(dv["flagged"] or ev["flagged"]),
                    why=list(dv["why"]) + list(ev["why"]))

    hits, cites, chance = [], [], []
    for src, a in sorted(_d28_intake_sources(v, lock_p).items()):
        verdict = judge(a["images"], a["dhash"], a["embed"], a["p_false"], a["cos"], a["copy_threshold"])
        if verdict["flagged"] or (a["base_share"] or 0.0) >= bc - EPS:  # stream-mutation: SM10
            hits.append({"source": a["source"], "batch": a["batches"][0], "batches": a["batches"],
                         "never_train_share": a["never_train_share"], "base_copy_share": a["base_share"],
                         "verdict": verdict})
            for name in a["names"]:
                cites.append(v.cite(name, "/source"))
                for ptr in ("/source_leak/eval_share", "/source_leak/base_share", "/images", "/copy_scan/p_false",
                            E.pointer("eval_hits", "per_source", src, "pair_cos"),
                            "/eval_hits/copy_threshold") + tuple(
                        "/guard/%s" % k for k in D28_DHASH_REASONS + (D28_EMBED_REASON,)):
                    try:
                        cites.append(v.cite(name, ptr))
                    except KeyError:
                        pass
            for side in a["sidecars"]:
                for ptr in (E.pointer("eval_hits", "per_source", src, "pair_cos"), "/eval_hits/copy_threshold"):
                    try:
                        cites.append(v.cite(side, ptr))
                    except KeyError:
                        pass
        elif a["dhash"]:
            chance.append({"source": a["source"], "batch": a["batches"][0], "batches": a["batches"],
                           "verdict": verdict, "path": "intake"})
    st = v.ev.json("step1_stream/status.json") or {}
    for src, row in sorted(((st.get("per_source") or {}).items())):
        row = row or {}
        seen = int(_num(row.get("images_seen")) or 0)
        emb = int(_num(row.get("near_eval_embed")) or 0)
        dh = sum(int(_num(row.get("decision:%s" % k)) or 0) for k in D28_DHASH_REASONS)
        if not seen:
            continue
        verdict = judge(seen, dh, emb, lock_p, _d28_pair_cos(row.get("eval_hit_pair_cos")),
                        _num(row.get("eval_hit_copy_threshold")))
        known = next((h for h in hits if str(h["source"]) == str(src)), None)
        if known is not None:
            # the source already leaks by its intake batches: its Step 1 dHash numbers are stated with that leak
            if dh:
                known.setdefault("dhash_elsewhere", []).append({"path": "Step 1", "dhash": verdict["dhash"]})
            continue
        if verdict["flagged"]:
            hits.append({"source": src, "batch": None, "never_train_share": (emb + dh) / float(seen),
                         "base_copy_share": None, "near_eval_embed": emb, "verdict": verdict})
            cites += [v.cite("step1_stream/status.json", E.pointer("per_source", src, "near_eval_embed")),
                      v.cite("step1_stream/status.json", E.pointer("per_source", src, "images_seen"))]
            for k in ["decision:%s" % r for r in D28_DHASH_REASONS] + ["eval_hit_pair_cos", "eval_hit_copy_threshold"]:
                try:
                    cites.append(v.cite("step1_stream/status.json", E.pointer("per_source", src, k)))
                except KeyError:
                    pass
        elif dh:
            chance.append({"source": src, "batch": None, "verdict": verdict, "path": "Step 1"})
    # a source has one verdict: a leak in either path wins, so a source that leaks is never also judged chance
    # (cleared, or named below for lifting its quarantine), and its dHash numbers of the other path are stated
    # with its leak; a source judged chance in both paths is stated once, with each path's numbers
    leaking = {str(h["source"]): h for h in hits}
    one = {}
    for c in chance:
        if str(c["source"]) in leaking:
            leaking[str(c["source"])].setdefault("dhash_elsewhere", []).append(
                {"path": c.get("path") or "intake", "dhash": c["verdict"]["dhash"]})
            continue
        got = one.setdefault(str(c["source"]), dict(c, verdicts=[]))
        got["verdicts"].append((c.get("path") or "intake", c["verdict"]["dhash"]))
    chance = list(one.values())

    def stated(c):
        vs = c["verdicts"]
        if len(vs) == 1:
            return "%s: %s" % (c["source"], EH.describe(vs[0][1]))
        return "%s: %s" % (c["source"], " and ".join("%s (%s)" % (EH.describe(dv), p) for p, dv in vs))
    # every source with dHash hits is stated with its numbers (requirement 5): the leaks below, the rest here
    judged = "; ".join(stated(c) for c in chance)
    judged = (". dHash hits judged chance: %s" % judged) if chance else ""
    cleared = [dict({"source": c["source"], "batch": c["batch"], "batches": c.get("batches") or [],
                     "dhash": c["verdicts"][0][1]},
                    **({"dhash_by_path": {p: dv for p, dv in c["verdicts"]}} if len(c["verdicts"]) > 1 else {}))
               for c in chance]
    # a source quarantined before D28-v2 (by the one-hit rule) whose hits are now judged chance stays
    # quarantined: only a person lifts a quarantine (inc2.stream unquarantine), and this names them
    qdoc = v.queue()
    qrec = qdoc.get("quarantined_sources") if isinstance(qdoc, dict) else None
    qs = set(qrec or {})
    srcs = v.c("/sources") or {}
    requeued = sorted({str(c["source"]) for c in chance if str(c["source"]) in qs
                       or (srcs.get(str(c["source"])) or {}).get("status") == "quarantined"})
    if requeued:
        judged += (". Quarantined, now judged chance (a person lifts a quarantine: inc2.stream unquarantine "
                   "--source S --stream %s --decided-by human:<who>): %s" % (v.sid, ", ".join(requeued)))
    # of those, the ones the stream itself still quarantines on D28's word (a person's unquarantine leaves the
    # platform's own source record as it was): what inc2.base3 build reads as quarantined; unknown (None) when
    # the queue summary does not state the stream's quarantine
    def d28_cited(src):
        rec = qrec.get(src) if isinstance(qrec, dict) else None
        cite = rec.get("cite") if isinstance(rec, dict) else None
        return cite is None or str(cite).startswith("D28")
    lift = (sorted({str(c["source"]) for c in chance if str(c["source"]) in qs and d28_cited(str(c["source"]))})
            if isinstance(qrec, (dict, list)) else None)
    if not hits:
        return _diag("D28", False, "info", "no source leaks: no dHash hit at or above the copy threshold in pair "
                     "cosine, no confirmed dHash hits (pair cos >= %g) improbable by chance (P(Binom(images, %g) >= "
                     "confirmed) < %g), no more embedding hits than the false-positive rate predicts (P < %g), no %g "
                     "base-copy share%s" % (confirm, p_conf, alpha, alpha, bc, judged), cites, [], None,
                     {"cleared": cleared, "cleared_quarantined": requeued, "lift_pending": lift})
    # Until each leaking source is quarantined (L24) or a person has kept it
    # (denied the quarantine), the rows of it already admitted -- including
    # augmented copies the per-row checks missed -- are still cuttable, and a
    # copy of dev raises dev, so the gate would prefer exactly those increments
    # (contract 3.2). TRAIN holds meanwhile: no segment is cut from a source
    # under a pending leak quarantine (shadow mode files L24 for a person).
    pending = sorted({str(h["source"]) for h in hits if str(h["source"]) not in qs
                      and (srcs.get(str(h["source"])) or {}).get("status") != "quarantined"
                      and not (srcs.get(str(h["source"])) or {}).get("leak_kept_by")})
    # L24 is inc2.stream's quarantine: it needs the stream to exist. A leak that
    # Step 1's one-time jobs (R1) find before the stream's init (R0) is carded and
    # holds TRAIN now; L24 runs once the stream ledger shows init (an earlier L24
    # exits 1 and, twice, holds the STOP lane on a stop-loss for good).
    inited = any(e.get("event") == "init" for e in v.stream_ledger())

    def why(h):
        vd = h.get("verdict") or {}
        dv = vd.get("dhash") or {}
        parts = ([EH.describe(dv)] if dv.get("hits") else []) + list(vd.get("why") or [])
        parts += ["its %s dHash hits alone: %s" % (o["path"], EH.describe(o["dhash"]))
                  for o in h.get("dhash_elsewhere") or ()]
        if (h.get("base_copy_share") or 0.0) >= bc - EPS:
            parts.append("%.1f %% base copies" % (100 * h["base_copy_share"]))
        return "; ".join(parts)
    return _diag("D28", True, "crit", "source leak: %s -> %s and a card%s%s" % (
        ", ".join("%s (%s)" % (h["source"], why(h)) for h in hits), "L24" if inited else "L24 once the stream exists",
        "; TRAIN holds until %s is quarantined or kept by a person" % ", ".join(pending) if pending else "", judged),
        cites, ["L24", "OP_CARD"], None, {"leaks": hits, "pending": pending, "hold": "TRAIN" if pending else None,
                                          "stream_exists": inited, "cleared": cleared,
                                          "cleared_quarantined": requeued, "lift_pending": lift,
                                          "propose": [{"lever": "L24", "source": h["source"], "cite": "D28"}
                                                      for h in hits] if inited else []})


# ------------------------------------------------------------ D22, D23, D24
REFUSAL_MARKS = ("cannot fill", "no exact fill", "after the guard excluded")


def d22(v):
    th = v.th
    Mv = v.M()
    Q, qc = v.Q()
    kmax = int(_t(th, "D22", "K_max"))
    cites = [qc, v.ccite("/M"), v.ccite("/lanes/TRAIN")]
    q = v.queue() or {}
    probe = ((q.get("cut") or {}).get("probe")) or {}
    refusal = [(i, r) for i, r in enumerate(v.c("/refusals") or [])
               if any(m in str((r or {}).get("message")) for m in REFUSAL_MARKS)]
    if probe.get("exact_fill") is True:
        # the cutter's own probe on the current queue fills exactly M: a build
        # refusal recorded before it is superseded (the queue has changed), and
        # holding TRAIN on it would wait for a person that nothing needs
        refusal = []
    if Q >= Mv > 0 and (probe.get("exact_fill") is False or refusal):
        # the cutter holds Q >= M and cannot fill exactly M: never a silent wait (S24)
        if probe.get("exact_fill") is False:
            why = str(probe.get("why") or probe.get("reason") or "the cut probe found no exact fill")
            cites.append(v.cite(v.qname, "/cut/probe/exact_fill"))
        else:
            i, r = refusal[-1]
            why = str(r.get("message"))
            cites.append(v.ccite("/refusals/%d/message" % i))
        return _diag("D22", True, "warn", "the cutter holds Q %d >= M %d and cannot fill an increment: %s -> a person "
                     "reads the refusal" % (Q, Mv, why[:300]), cites, ["OP_ESCALATE"], None,
                     {"refused": why, "Q": Q, "M": Mv})
    if not v.lane_idle("TRAIN"):
        return _silent("D22", "the TRAIN lane is busy or held", cites=cites)
    oldest = (q.get("eligible") or {}).get("oldest_utc")
    age = _days(v.c("/now_utc"), oldest) if oldest else None
    full = Mv > 0 and Q >= int(_t(th, "D22", "cut_M_mult")) * Mv
    stale = Mv > 0 and Q >= Mv and age is not None and age >= float(_t(th, "D22", "stale_days"))
    if full or stale:  # stream-mutation: SM3
        K = max(1, min(kmax, Q // Mv))
        if oldest:
            cites.append(v.cite(v.qname, "/eligible/oldest_utc"))
        nxt = q.get("next_segment")
        if nxt:
            cites.append(v.cite(v.qname, "/next_segment"))
        return _diag("D22", True, "info", "Q %d, M %d (%s) -> L18 with K %d" % (
            Q, Mv, "Q >= 4M" if full else "oldest eligible row %.1f days old" % age, K), cites, ["L18"], None,
            {"K": K, "Q": Q, "M": Mv, "propose": {"lever": "L18", "k": K, "exp": nxt}})
    return _silent("D22", "Q %d, M %d: no cut yet" % (Q, Mv), cites=cites)


def _commits(v):
    return {e.get("exp"): e for e in v.stream_ledger() if e.get("event") == "commit" and e.get("exp")}


def d23(v):
    commits = _commits(v)
    out = []
    for i, s in enumerate(v.c("/segments") or []):
        if not isinstance(s, dict):
            continue
        waiting = bool(s.get("done")) and s.get("exp") not in commits and not s.get("committed")
        if waiting:  # stream-mutation: SM18
            out.append((s["exp"], [v.ccite("/segments/%d/done" % i), v.ccite("/segments/%d/exp" % i)]))
    q = v.queue() or {}
    for j, exp in enumerate(q.get("uncommitted_done") or []):
        if exp not in commits and exp not in [x for x, _c in out]:
            out.append((exp, [v.cite(v.qname, "/uncommitted_done/%d" % j)]))
    if not out:
        return _silent("D23", "no finished segment waits for its commit")
    exp, cites = out[0]
    return _diag("D23", True, "info", "segment %s is finished and not committed -> L19" % exp, cites, ["L19"], exp,
                 {"propose": {"lever": "L19", "exp": exp}})


def d24(v):
    """A milestone is due (contract 5.5): 4 accepted increments, 3 segments, or
    30 days since the first ACCEPT, counted since the last good milestone on
    the current pool's lineage (the stream summary's counts, its day count
    carried forward to the tick's clock), or a boundary alarm there; never
    while a milestone is in flight or a rollback is pending."""
    th = v.th
    q = v.queue() or {}
    ms = q.get("milestones") or {}
    cites = [v.ccite("/lanes/MAINT")]
    if not isinstance(ms, dict) or "accepted_since" not in ms:
        return _silent("D24", "no stream summary with milestone counts", cites=cites)
    cites += [v.cite(v.qname, "/milestones/accepted_since"), v.cite(v.qname, "/milestones/segments_since"),
              v.cite(v.qname, "/milestones/in_flight")]
    if not v.lane_idle("MAINT"):
        return _silent("D24", "the MAINT lane is busy or held", cites=cites)
    if [x for x in (q.get("rollback_pending") or []) if isinstance(x, dict) and x.get("to_pool")]:
        # a 'hurts' milestone's rollback (D25 -> L21) decides the pool first: the
        # counts above still include the increments it is about to suspend, and a
        # milestone built on them (or, once L21 ran, on the rolled-back pool, which
        # inc2.stream refuses) would be a failed MAINT step
        return _silent("D24", "a rollback is pending (D25 -> L21): the pool is decided before a milestone",
                       cites=cites + [v.cite(v.qname, "/rollback_pending")])
    acc, segs = int(_num(ms.get("accepted_since")) or 0), int(_num(ms.get("segments_since")) or 0)
    days = _num(ms.get("days_since_first_accepted"))
    # the summary counts days at the moment it was written; a quiet stream is not
    # rewritten, so the days since then are added (the 30-day trigger never waits
    # on a stream write)
    gen, now = _utc(q.get("generated_utc")), _utc(v.c("/now_utc"))
    if days is not None and gen is not None and now is not None and now > gen:
        days += (now - gen).total_seconds() / 86400.0
    a_n, s_n, d_n = int(_t(th, "D24", "accepted")), int(_t(th, "D24", "segments")), float(_t(th, "D24", "days"))
    boundary = "boundary_check" in (ms.get("due_reasons") or [])
    busy = ms.get("in_flight")
    if (acc >= a_n or segs >= s_n or (days is not None and days >= d_n) or boundary) and not busy:  # stream-mutation: SM4
        if ms.get("next"):
            cites.append(v.cite(v.qname, "/milestones/next"))
        return _diag("D24", True, "info", "%d accepted increment(s) over %d segment(s) since the last good milestone%s%s "
                     "-> L20" % (acc, segs, "" if days is None else ", the first ACCEPT %.1f days ago" % days,
                                 "; a boundary alarm" if boundary else ""), cites, ["L20"], None,
                     {"accepted": acc, "segments": segs, "propose": {"lever": "L20", "exp": ms.get("next")}})
    return _silent("D24", "%d accepted, %d segment(s) since the last good milestone%s" % (
        acc, segs, " (%s in flight)" % busy if busy else ""), cites=cites)


# ------------------------------------------------------------------ D25
def _milestone_compares(v):
    return [(i, e) for i, e in enumerate(v.stream_ledger())
            if e.get("event") == "milestone" and e.get("phase") == "compare"]


def d25(v):
    """The stream regressed (contract 5.5, 3.6): (b) a milestone the stream's
    5 v 5 dev comparison found 'hurts' (one-sided permutation p <= 0.025 with a
    lower mean; inc2.stream compare) and whose recommended rollback has not
    run -> L21 to that pool, TRAIN holds, card X4; a species-guard-only 'hurts'
    -> card X17 (no rollback); (a) the boundary check (segment s+1's base seeds
    below segment s's by more than 2 sd, 3 v 3: an alarm) -> L20 at once, TRAIN
    holds."""
    th = v.th
    sdm, pmax = float(_t(th, "D25", "boundary_sd_mult")), float(_t(th, "D25", "perm_p"))
    q = v.queue() or {}
    pend = [x for x in (q.get("rollback_pending") or []) if isinstance(x, dict) and x.get("to_pool")]
    cmp_ = _milestone_compares(v)
    cites = []
    if pend:  # stream-mutation: SM5
        j = len(pend) - 1
        x = pend[j]
        cites += [v.cite(v.qname, "/rollback_pending/%d/to_pool" % j), v.cite(v.qname, "/rollback_pending/%d/milestone" % j)]
        last = next(((i, e) for i, e in reversed(cmp_) if e.get("exp") == x.get("milestone")), None)
        p_ = last[1].get("perm_p") if last else None
        if last:
            cites += [v.cite(v.lname, "/%d/verdict" % last[0]), v.cite(v.lname, "/%d/perm_p" % last[0]),
                      v.cite(v.lname, "/%d/new_mean_dev" % last[0]), v.cite(v.lname, "/%d/old_mean_dev" % last[0])]
        agrees = last is not None and _num(p_) is not None and float(p_) <= pmax + EPS \
            and (_num(last[1].get("new_mean_dev")) or 0.0) < (_num(last[1].get("old_mean_dev")) or 0.0)
        if not agrees:
            # the stream recommends a rollback its recorded comparison does not
            # support under the pre-registered rule: a person reads it
            return _diag("D25", True, "crit", "milestone %s: a rollback to %s is recommended, but its recorded "
                         "comparison (p %s) does not meet the pre-registered p <= %g with a lower mean -> a person "
                         "reads it; TRAIN holds" % (x.get("milestone"), x["to_pool"], p_, pmax), cites,
                         ["OP_ESCALATE"], x.get("milestone"), {"kind": "disagreement", "hold": "TRAIN"})
        return _diag("D25", True, "crit", "milestone %s hurts on dev (5 v 5 permutation p %s) -> L21 to %s, TRAIN holds, "
                     "card X4" % (x.get("milestone"), "%.4g" % p_ if _num(p_) is not None else "n/a", x["to_pool"]),
                     cites, ["L21", "X4"], x.get("milestone"),
                     {"kind": "milestone", "p": p_, "new": x.get("milestone"), "to": x["to_pool"], "hold": "TRAIN",
                      "propose": {"lever": "L21", "to": x["to_pool"]}})
    if cmp_ and cmp_[-1][1].get("verdict") == "species_only":
        i, e = cmp_[-1]
        return _diag("D25", True, "warn", "milestone %s: a species-guard-only 'hurts' (%s) -> card X17, no rollback"
                     % (e.get("exp"), ", ".join(e.get("species_failed") or [])),
                     [v.cite(v.lname, "/%d/verdict" % i), v.cite(v.lname, "/%d/species_failed" % i)], ["X17"],
                     e.get("exp"), {"kind": "species_only", "species": e.get("species_failed")})
    bc = q.get("boundary_check")
    nm, om, osd = ((_num(bc.get(k)) for k in ("new_mean", "old_mean", "old_sd")) if isinstance(bc, dict)
                   else (None, None, None))
    if None not in (nm, om, osd):
        cites += [v.cite(v.qname, "/boundary_check/new_mean"), v.cite(v.qname, "/boundary_check/old_mean"),
                  v.cite(v.qname, "/boundary_check/old_sd")]
        if nm < om - sdm * osd - EPS:  # stream-mutation: SM6
            return _diag("D25", True, "crit", "boundary check: %s's base seeds %.4f < %.4f - %g sd (%.4f) of %s's -> "
                         "L20 at once, TRAIN holds" % (bc.get("segment"), nm, om, sdm, osd, bc.get("previous")),
                         cites, ["L20"], bc.get("segment"),
                         {"kind": "boundary", "new": bc.get("segment"), "old": bc.get("previous"), "hold": "TRAIN",
                          "propose": {"lever": "L20", "exp": ((q.get("milestones") or {}).get("next"))}})
    if not cites:
        return _silent("D25", "no milestone comparison or boundary check yet")
    return _silent("D25", "no regression on dev", cites=cites)


# ------------------------------------------------------------------ D26, D27
def d26(v):
    th = v.th
    share = float(_t(th, "D26", "walltime_share"))
    wt = v.dom.get("walltime") or {}
    n = int(v.c("/pool/images") or (v.dom.get("increment") or {}).get("base_images") or 0)
    m = v.M()
    arm = v.c("/arm") or {}
    factor = float(arm.get("cost_factor") or 1.0)
    cold = LS._hours(n + m, LS.cost(v.dom, "cold_epochs"), LS.cost(v.dom, "cold_ms_per_image_epoch")) * factor
    # the incremental runs of the recipes the next segment runs (the chosen
    # recipe, else the stream's Stage B arms), not of every recipe the domain
    # prices: an unused longer recipe would hold the lane on a run never made
    epochs = LS.cost(v.dom, "chain_epochs") or {}
    used = [epochs[r] for r in (v.c("/recipes") or []) if r in epochs]
    ep = max(used or list(epochs.values()) or [30])
    incr = LS._hours(n + m, ep, LS.cost(v.dom, "incr_ms_per_image_epoch")) * factor
    build = _num(v.c("/build_hours_max"))
    rows = [("cold", cold, float(wt.get("cold_h") or 8.0)), ("incremental", incr, float(wt.get("incremental_h") or 3.0))]
    if build is not None:
        rows.append(("build", build, float(wt.get("build_h") or 4.0)))
    cites = [v.ccite("/pool/images"), v.ccite("/M"), v.ccite("/recipes")]
    hit = [(k, h, lim) for k, h, lim in rows if h >= share * lim - EPS]  # projected >= 0.8 x the limit
    if hit:  # stream-mutation: SM7
        return _diag("D26", True, "crit", "walltime: %s -> builds hold; card X15" % ", ".join(
            "%s run %.2f h >= %g x %g h" % (k, h, share, lim) for k, h, lim in hit), cites, ["X15"], None,
            {"hold": ["TRAIN", "MAINT"], "projected": {k: h for k, h, _l in rows}})
    return _silent("D26", "projected runs %s under %g of their limits" % (
        ", ".join("%s %.2f h" % (k, h) for k, h, _l in rows), share), cites=cites)


def d27(v):
    th = v.th
    frac = float(_t(th, "D27", "quota_free_frac"))
    renew = float(_t(th, "D27", "renewal_card_days"))
    cites, why, levers, sev, detail = [], [], [], "info", {}
    al = v.c("/allocation") or {}
    end = al.get("end_date")
    now = v.c("/now_utc")
    if end:
        left = _days("%sT00:00:00Z" % end if "T" not in str(end) else end, now)
        cites.append(v.ccite("/allocation/end_date"))
        if left is not None and left <= 0:
            return _diag("D27", True, "crit", "the allocation ended on %s -> PAUSE allocation_ended" % end, cites,
                         ["OP_PAUSE"], None, {"reason": "allocation_ended"})
        if left is not None and left <= renew:
            why.append("the allocation ends on %s (%.0f days): renewal card" % (end, left))
            levers.append("X14")
            sev = "warn"
            detail["renewal"] = end
    bal, com, res = _num(al.get("balance_su")), _num(al.get("committed_all_su")), _num(al.get("reserve_su"))
    if bal is not None and res is not None:
        cites += [v.ccite("/allocation/balance_su"), v.ccite("/allocation/reserve_su")]
        if bal - (com or 0.0) < res:  # stream-mutation: SM9
            return _diag("D27", True, "crit", "allocation balance %.1f - committed %.1f < reserve %.1f -> OP_PAUSE"
                         % (bal, com or 0.0, res), cites, ["OP_PAUSE", "X14"], None, {"reason": "allocation_reserve"})
    qu = v.c("/quota") or {}
    quota, free = _num(qu.get("quota_gb")), _num(qu.get("free_gb"))
    proj = (_num(qu.get("staging_gb")) or 0.0) + (_num(qu.get("projected_gb")) or 0.0)
    if quota is not None and free is not None:
        cites += [v.ccite("/quota/free_gb"), v.ccite("/quota/quota_gb")]
        if free - proj < frac * quota:  # stream-mutation: SM8
            return _diag("D27", True, "crit", "/ocean: %.0f GB free - %.0f GB staged or projected < %g of %.0f GB -> "
                         "OP_PAUSE; the largest directories are named for a person"
                         % (free, proj, frac, quota), cites, ["OP_PAUSE"], None,
                         {"reason": "quota", "largest": qu.get("largest") or []})
        env_gb = _num(v.c("/collect_gb_envelope"))
        if env_gb is not None and free < frac * quota + env_gb:
            why.append("free %.0f GB < %g of the quota + the %.0f GB collection envelope: collection does not start"
                       % (free, frac, env_gb))
            detail["hold"] = "DATA"
            detail["largest"] = qu.get("largest") or []
            sev = "warn"
    if not why:
        return _silent("D27", "allocation and quota within their reserves", cites=cites)
    return _diag("D27", True, sev, "; ".join(why), cites, levers, None, detail)


# ------------------------------------------------------------ D30-D33, D8S
def last_decided(v, prior=None):
    """(rows, label, ev): the step records D30/D33 read -- the last committed
    stream segment (its commit line); before Stage A is READY with no segment
    committed, the pinned prior (realloop_v1, read through the guard rule);
    with Stage A READY and none committed, nothing."""
    commits = stream_commits(v)
    if commits:
        i, e = commits[-1]
        return commit_rows(v, i, e), e["exp"], v.ev
    if not v.c("/stage/stage_a_ready") and prior is not None:
        return steps(prior, prior.exp, None, float(_t(v.th, "disposition", "p_reject"))), prior.exp, prior
    return [], None, v.ev


def d30(v, prior=None):
    rows, label, ev = last_decided(v, prior)
    rej = [r for r in rows if r["verdict"] == "REJECT"]
    if not rej:
        return _silent("D30", "no decided REJECT to read" if label else "no segment decided")
    rec = [r for r in rej if r["disposition"] == "recipe"]
    share = len(rec) / float(len(rej))
    cites = [c for r in rej for c in _row_cites(ev, r)]
    if share >= float(_t(v.th, "D30", "share_min")) - EPS:  # stream-mutation: SM12
        return _diag("D30", True, "crit", "%d of %d REJECTs on %s are recipe-caused (only the regression guard failed) -> "
                     "TRAIN holds; cards X1 and X13" % (len(rec), len(rej), label), cites, ["X1", "X13"], label,
                     {"hold": "TRAIN", "steps": [r["step"] for r in rec], "of": len(rej)})
    return _silent("D30", "%d of %d REJECTs on %s are recipe-caused" % (len(rec), len(rej), label), cites=cites)


def d31(v, prior=None, include_prior=False):
    """Data blamed: a 'data' disposition or a truth 'hurts'. Reads every
    committed stream segment (the prior only when include_prior: a replay of it)."""
    runs = [(v.ev, e["exp"], commit_rows(v, i, e)) for i, e in stream_commits(v)]
    if include_prior and prior is not None:
        runs.append((prior, prior.exp, steps(prior, prior.exp, None, float(_t(v.th, "disposition", "p_reject")))))
    blamed, cites, per_source = [], [], {}
    for ev, exp, rows in runs:
        for r in rows:
            if r["disposition"] == "data" or r["truth"] == "hurts":  # stream-mutation: SM13
                blamed.append({"exp": exp, "step": r["step"], "chain": r["chain"], "disposition": r["disposition"],
                               "truth": r["truth"], "sources": r["sources"]})
                cites += _row_cites(ev, r)
                if len(r["sources"]) == 1:
                    per_source[r["sources"][0]] = per_source.get(r["sources"][0], 0) + 1
    if not blamed:
        return _silent("D31", "no step disposed 'data' and no truth 'hurts'")
    n = int(_t(v.th, "D31", "blames_to_quarantine"))
    q = sorted(s for s, k in per_source.items() if k >= n)
    audits = {}
    for b in blamed:
        audits.setdefault(b["exp"], []).append(b["step"])
    props = [{"lever": "L4", "exp": e, "steps": sorted(set(s))} for e, s in sorted(audits.items())]
    props += [{"lever": "L24", "source": s, "cite": "D31"} for s in q]
    return _diag("D31", True, "warn", "data blamed on %s -> L4 label audit first%s" % (
        ", ".join("%s/%s (%s%s)" % (b["exp"], b["step"], b["disposition"],
                                     ", truth hurts" if b["truth"] == "hurts" else "") for b in blamed),
        "; L24 on %s" % ", ".join(q) if q else ""), cites, ["L4"] + (["L24"] if q else []), blamed[-1]["exp"],
        {"blamed": blamed, "per_source": per_source, "quarantine": q, "propose": props})


def d32(v, d30_fired=False):
    window = int(_t(v.th, "D32", "window"))
    rows = []
    for i, e in stream_commits(v):
        rows += commit_rows(v, i, e)
    last = rows[-window:]
    if len(last) < window:
        return _silent("D32", "%d decided increment(s), fewer than %d" % (len(rows), window))
    acc = [r for r in last if r["verdict"] == "ACCEPT"]
    cites = [r["cites"][0] for r in last]
    if not acc and not d30_fired:  # stream-mutation: SM14
        return _diag("D32", True, "crit", "0 ACCEPT in the last %d decided increments while D30 is silent -> L18 holds, "
                     "card X17, the DATA lane at half cadence" % window, cites, ["X17"], None,
                     {"hold": "TRAIN", "half_cadence": True})
    return _silent("D32", "%d ACCEPT in the last %d decided increments" % (len(acc), window), cites=cites)


def _d33_names(rej, share):
    """([species], {species: REJECTs it failed}) of D33's rule over one
    segment's REJECT records: the species failing the species guard in at
    least `share` of them."""
    count = {}
    for r in rej:
        if "species" not in r["failed"]:
            continue
        for sp in r["species_failed"]:
            count[sp] = count.get(sp, 0) + 1
    worst = sorted(((n, sp) for sp, n in count.items()), reverse=True)
    return [sp for n, sp in worst if n / float(len(rej)) >= share - EPS], count


def d33(v, prior=None, stage_c=None):
    share = float(_t(v.th, "D33", "share"))
    hold_n = int(_t(v.th, "D33", "hold_after_segments"))
    if stage_c and stage_c.get("species_only_reject"):
        sp = stage_c.get("species") or []
        return _diag("D33", True, "crit", "Stage C: the known-good increment was REJECTed on the species guard alone "
                     "(%s) -> card X17 to the owner before any collected data is spent" % ", ".join(sp),
                     stage_c.get("cites") or [], ["X17"], stage_c.get("exp"),
                     {"prospective": True, "species": sp, "stage_c": True})
    rows, label, ev = last_decided(v, prior)
    rej = [r for r in rows if r["verdict"] == "REJECT"]
    if not rej:
        return _silent("D33", "no decided REJECT to read")
    names, count = _d33_names(rej, share)
    cites = [c for r in rej for c in _row_cites(ev, r)]
    if names:  # stream-mutation: SM15
        # consecutive = the trailing run of the stream's own committed segments
        # (stale commits decide nothing) on which the rule fires, counted from
        # the stream ledger itself: the pinned prior is not a stream segment,
        # and a segment on which D33 was silent breaks the run
        consecutive = 0
        if ev is v.ev:
            for i, e in reversed(stream_commits(v)):
                rj = [r for r in commit_rows(v, i, e) if r["verdict"] == "REJECT"]
                if rj and _d33_names(rj, share)[0]:
                    consecutive += 1
                    if e.get("exp") != label:
                        cites += [c for r in rj for c in r["cites"][:1]]
                else:
                    break
        hold = consecutive >= hold_n
        return _diag("D33", True, "warn", "%s fails the species guard in %s of %d REJECTs on %s -> card X17; a D20 "
                     "ranking bonus for sources declaring it%s" % (", ".join(names), ", ".join(
                         str(count[n]) for n in names), len(rej), label,
                         "; fired on %d consecutive segments: L18 holds" % consecutive if hold else ""),
                     cites, ["X17"], label, {"species": names, "counts": count, "of": len(rej),
                                             "consecutive": consecutive, "hold": "TRAIN" if hold else None})
    return _silent("D33", "no species fails the guard in half of the REJECTs on %s" % label, cites=cites)


def d8s(v):
    lo, hi = _t(v.th, "D8S", "p_band")
    s_min, mult = float(_t(v.th, "D8S", "share_min")), float(_t(v.th, "D8S", "sd_mult"))
    maxd = int(_t(v.th, "D8S", "max_doublings"))
    commits = stream_commits(v)
    if not commits:
        return _silent("D8S", "no segment decided")
    i, e = commits[-1]
    exp = e["exp"]
    rows = [r for r in commit_rows(v, i, e) if None not in (r["p_data"], r["cand_mean"], r["null_mean"], r["null_sd"])]
    hits = [r for r in rows if lo < r["p_data"] < hi and abs(r["cand_mean"] - r["null_mean"])
            < mult * max(r["cand_sd"] or 0.0, r["null_sd"])]
    if not rows or len(hits) < s_min * len(rows) or not hits:
        return _silent("D8S", "%d of %d steps underpowered" % (len(hits), len(rows)))
    done = int(v.c("/doublings") or 0)
    cites = [c for r in hits for c in r["cites"][:2]]
    if done >= maxd:
        return _diag("D8S", True, "warn", "%d of %d steps underpowered after %d doublings -> card X17" % (
            len(hits), len(rows), done), cites, ["X17"], exp, {"doublings": done})
    return _diag("D8S", True, "warn", "%d of %d steps have P_data in (%g, %g) and |cand - null| < %g sd -> L22 (M %d -> %d)"
                 % (len(hits), len(rows), lo, hi, mult, v.M(), 2 * v.M()), cites, ["L22"], exp,
                 {"propose": {"lever": "L22", "m": 2 * v.M()}, "doublings": done})


# ------------------------------------------------------------------ D10S
def d10s(v):
    b = v.c("/budget") or {}
    rem = _num(b.get("remaining_su"))
    # 6.6 review: the domain's own ledger, every step counted (the pre-INC
    # round history included, as signals._check_budget counts it), against the
    # live domain's su_envelope: exhausted, it is resolved by a person (X14)
    # before the stream spends anything, or the scheduler's budget alarm fires
    # beside it
    dl = v.c("/domain_ledger") or {}
    drem = _num(dl.get("remaining_su"))
    if drem is not None and drem <= 0:
        return _diag("D10S", True, "crit", "the domain's ledger (%.1f SU over every step, the round history included) "
                     "exhausts its su_envelope %s -> PAUSE; card X14 (a person raises the envelope or accepts the "
                     "history)" % (_num(dl.get("campaign_su")) or 0.0, dl.get("envelope")),
                     [v.ccite("/domain_ledger/remaining_su"), v.ccite("/domain_ledger/envelope")],
                     ["OP_PAUSE", "X14"], None, {"reason": "domain_envelope_exhausted"})
    if rem is None:
        return _silent("D10S", "no envelope in the context")
    cites = [v.ccite("/budget/remaining_su")]
    if rem <= 0:
        return _diag("D10S", True, "crit", "the campaign envelope is exhausted (%.2f SU left) -> PAUSE" % rem, cites,
                     ["OP_PAUSE", "X14"], None, {"reason": "budget_exhausted"})
    return _silent("D10S", "%.2f SU left in the envelope" % rem, cites=cites)


# --------------------------------------------------- R0 records (DSA, DSC, DCAP, DCAN)
def _record(v, exp, fname):
    """(the verdict record <exp>/<fname>, its evidence name) or (None, name)."""
    name = "%s/%s" % (exp, fname)
    rec = v.ev.json(name) if exp else None
    return (rec if isinstance(rec, dict) else None), name


def stage_a(v):
    """DSA: Stage A's recorded verdict (contract 5.1; inc2.pilot4 verdict wrote
    <exp>/stage_a.json by the pre-registered survival rule, dev only): READY
    gives segment 1's recipes (R0 and the best survivor). PENDING although the
    pilot is done and its verdict ran is a card for a person."""
    sa = v.dom.get("stage_a") or {}
    rec, name = _record(v, sa.get("exp"), sa.get("record") or "stage_a.json")
    if rec is None:
        return _silent("DSA", "no Stage A verdict recorded (%s)" % name)
    cites = [v.cite(name, "/status"), v.cite(name, "/segment1_recipes")]
    if rec.get("status") == "READY":
        recipes = [str(x) for x in rec.get("segment1_recipes") or [] if x]
        return _diag("DSA", True, "info", "Stage A READY: survivors %s, segment 1 runs %s" % (
            rec.get("survivors") or [], ",".join(recipes)), cites, [], sa.get("exp"),
            {"ready": True, "survivors": rec.get("survivors") or [], "chosen": rec.get("best_survivor"),
             "recipes": ",".join(recipes) or "r0"})
    ran = (v.c("/stage/verdicts") or {}).get("stage_a")
    if v.c("/stage/stage_a/status") == "done" and ran:
        return _diag("DSA", True, "warn", "Stage A is PENDING although %s is done and its verdict ran: %s -> a person "
                     "reads it" % (sa.get("exp"), json.dumps(rec.get("pending") or {}, sort_keys=True)[:300]),
                     cites + [v.cite(name, "/pending")], ["OP_ESCALATE"], sa.get("exp"), {"ready": False})
    return _silent("DSA", "Stage A is %s" % rec.get("status"), cites=cites)


def _feasibility(v):
    """(ledger index, the Stage C read line) of the stream, or (None, None)."""
    for i, e in reversed(list(enumerate(v.stream_ledger()))):
        if e.get("event") == "feasibility" and e.get("phase") == "read":
            return i, e
    return None, None


def stage_c(v):
    """DSC: Stage C's verdict (contract 5.1), as inc2.stream compare read it
    through gate3: the known-good increment ACCEPTed on some chain -> M
    feasible; a species-guard-only REJECT -> D33 prospectively and card X17."""
    i, e = _feasibility(v)
    if e is None:
        return _silent("DSC", "Stage C has not been read")
    res = e.get("result") or {}
    per = res.get("per_recipe") or {}
    cites = [v.cite(v.lname, "/%d/result/m_feasible" % i), v.cite(v.lname, "/%d/result/species_only_reject" % i)]
    acc = sorted(r for r, x in per.items() if (x or {}).get("verdict") == "ACCEPT")
    rec = {"exp": e.get("exp"), "accepted_chains": acc, "feasible": bool(res.get("m_feasible")),
           "species_only_reject": bool(res.get("species_only_reject")) and not res.get("m_feasible"),
           "species": list(res.get("d33_prospective") or []), "cites": cites}
    return _diag("DSC", True, "info" if rec["feasible"] else "warn",
                 "Stage C: %s" % ("ACCEPTed on %s: M is feasible" % acc if rec["feasible"] else
                                  "REJECTed%s" % (" on the species guard alone (%s)" % rec["species"]
                                                  if rec["species_only_reject"] else "")),
                 cites, [] if rec["feasible"] else ["X17"], e.get("exp"), rec)


def capacity(v):
    """DCAP: L-4's capacity decision as inc2.baseline capacity-verdict recorded
    it (capacity/capacity_v1.json, dev only): the chosen arm, its experiment
    (milestone 0) and truth_every."""
    name = "capacity/capacity_v1.json"
    rec = v.ev.json(name)
    if not isinstance(rec, dict) or not rec.get("chosen_arm"):
        return _silent("DCAP", "no capacity decision recorded")
    cites = [v.cite(name, "/chosen_arm"), v.cite(name, "/chosen_exp"), v.cite(name, "/qualifying")]
    arms = {k: {"arm": (x or {}).get("arm"), "mean": (x or {}).get("mean"), "sd": (x or {}).get("sd")}
            for k, x in (rec.get("arms") or {}).items()}
    chosen = str(rec["chosen_arm"])
    cap = (v.dom.get("capacity") or {}).get("arms") or {}
    return _diag("DCAP", True, "info", "capacity: %s (%s); qualifying %s" % (chosen, rec.get("chosen_exp"),
                                                                         rec.get("qualifying") or []),
                 cites, [], rec.get("chosen_exp"),
                 {"chosen": chosen, "chosen_exp": rec.get("chosen_exp"), "arms": arms,
                  "eligible": rec.get("qualifying") or [], "truth_every": rec.get("truth_every"),
                  "arm": dict(cap.get(chosen) or {}, name=chosen)})


def native(v):
    """DNAT: the measurement arms' native-resolution verdict as inc2.baseline
    native-verdict recorded it (capacity/native_v1.json, dev only; the
    stream-domain config's capacity.native.record). An arm that qualifies
    for a stream fork proposal is card X18 for a person; nothing switches
    and no lane holds."""
    name = ((v.dom.get("capacity") or {}).get("native") or {}).get("record")
    rec = v.ev.json(name) if name else None
    if not isinstance(rec, dict) or not isinstance(rec.get("arms"), dict):
        return _silent("DNAT", "no native-resolution verdict recorded")
    cites = [v.cite(name, "/qualifying")]
    arms = rec["arms"]
    qual = [e for e in rec.get("qualifying") or [] if (arms.get(e) or {}).get("qualifies") is True]
    if not qual:
        return _silent("DNAT", "native-resolution verdict: no measurement arm qualifies (%s)"
                       % ", ".join("%s %s" % (e, (a or {}).get("status")) for e, a in sorted(arms.items())),
                       cites=cites)
    cites += [v.cite(name, E.pointer("arms", e, "qualifies")) for e in qual]
    summ = "; ".join("%s at %s px: dev D %+.4f > 2 pooled sd %.4f and SE %.4f; improved %s"
                     % (e, arms[e].get("imgsz"), _num(arms[e].get("diff")) or 0.0,
                        _num(arms[e].get("two_pooled_sd")) or 0.0, _num(arms[e].get("se_diff")) or 0.0,
                        ", ".join(arms[e].get("improved_targets") or [])) for e in qual)
    return _diag("DNAT", True, "info", "native-resolution verdict: %s qualif%s for a stream fork proposal (%s) -> "
                 "card X18; nothing switches" % (", ".join(qual), "ies" if len(qual) == 1 else "y", summ),
                 cites, ["X18"], None, {"qualifying": qual})


def canary(v):
    """DCAN: the canary's recorded verdict (inc2.baseline canary-verdict,
    <exp>/canary.json): a failed canary means the v2 executor does not
    reproduce b0_v1; the TRAIN lane waits and a person reads it."""
    b = next((x for x in (v.dom.get("baselines") or {}).get("items") or [] if x.get("verdict") == "canary-verdict"),
             None)
    if not b:
        return _silent("DCAN", "the stream-domain config names no canary")
    rec, name = _record(v, b["exp"], "canary.json")
    if rec is None:
        return _silent("DCAN", "no canary verdict recorded")
    cites = [v.cite(name, "/passed")]
    if rec.get("passed") is True:
        return _silent("DCAN", "the canary passed", cites=cites)
    return _diag("DCAN", True, "crit", "the canary %s failed its rule (within one sd %s, sidecar %s, production %s) -> "
                 "TRAIN holds; a person reads %s" % (b["exp"], rec.get("within_one_sd"), rec.get("sidecar_ok"),
                                                     rec.get("production"), name),
                 cites, ["OP_ESCALATE"], b["exp"], {"hold": "TRAIN", "passed": False})


# ------------------------------------------------ the lanes' own work items
def _stream_state(v):
    """What the stream ledger says about its own set-up: initialised, the arm
    adopted (choose-arm), Stage C built and read.

    Stage C is the campaign's feasibility check at R0: read once (DSC) and
    recorded on the campaign (/stage/stage_c_decided). A later stream version
    (L22's fork, which adopts the pool and the quarantine) starts a ledger of
    its own without a feasibility event and never builds Stage C again, so the
    campaign's recorded decision counts as Stage C built and read for it. Seen
    live on 2026-10-04: after the 05:57Z fork, R0 read as incomplete and every
    measurement arm, E2's six builds included, waited on a Stage C the fork
    would never read."""
    led = v.stream_ledger()
    ev = [e.get("event") for e in led]
    fz = [e for e in led if e.get("event") == "feasibility"]
    decided = bool(v.c("/stage/stage_c_decided"))
    return {"init": "init" in ev, "arm": "arm" in ev,
            "stage_c_built": any(e.get("phase") == "build" for e in fz) or decided,
            "stage_c_read": any(e.get("phase") == "read" for e in fz) or decided}


def _eval_hits_due(v):
    """([batch], cites) whose dHash hits on evaluation images no record weighs
    and for which no sidecar has been made (D28-v2): Step 1's
    (status.json eval_hits.due, written by step1_stream's write_status) and
    the intake batches' (a summary whose eval_hits weighs fewer hits than its
    guard counted, without intake/<batch>/eval_hits.json in the snapshot).
    step1_stream eval-hits (L17) makes them, one attempt per batch: a batch
    it cannot give a sidecar fails the job."""
    due, cites = [], []
    st = v.ev.json("step1_stream/status.json") or {}
    if isinstance(st.get("eval_hits"), dict):
        for b in st["eval_hits"].get("due") or []:
            due.append("Step 1 batch %s" % b)
        if due:
            cites.append(v.cite("step1_stream/status.json", "/eval_hits/due"))
    else:
        # a status.json written before the sidecars existed lists no batch: a source row with more dHash hits
        # than pair cosines says a batch needs one (eval-hits finds which, and rewrites status.json)
        for src, row in sorted((st.get("per_source") or {}).items()):
            row = row or {}
            dh = sum(int(_num(row.get("decision:%s" % k)) or 0) for k in D28_DHASH_REASONS)
            if dh > len(_d28_pair_cos(row.get("eval_hit_pair_cos")) or []):
                due.append("Step 1 (source %s; status.json predates the sidecars)" % src)
                cites.append(v.cite("step1_stream/status.json", E.pointer("per_source", src, "decision:%s" % next(
                    k for k in D28_DHASH_REASONS if int(_num(row.get("decision:%s" % k)) or 0)))))
                break
    for name in sorted(v.ev.artifacts):
        if not (name.startswith("intake/") and name.endswith("/summary.json")):
            continue
        s = v.ev.json(name) or {}
        _n, dh, _emb, counted = _d28_counts(s)
        if not (counted and dh):
            continue
        cos, _t, _used = _d28_batch_cos(v, name, s, str(s.get("source")), dh, counted)
        if (cos is None or len(cos) < dh) and v.ev.json(name[:-len("summary.json")] + "eval_hits.json") is None:
            due.append("intake batch %s" % name.split("/")[1])
            for k in D28_DHASH_REASONS:
                try:
                    cites.append(v.cite(name, "/guard/%s" % k))
                except KeyError:
                    pass
    return due, cites


def lift_wait(v, d28=None):
    """The wait DR0 keeps before it proposes the base v3 build (L23V;
    docs/CONTINUOUS_LOOP.md, E1, amendments 2026-10-03), or None (no wait,
    and none recorded). inc2.base3 build reads the stream's quarantine when
    its job starts, and a quarantined source never enters arm B; a
    quarantine is lifted only by a person (inc2.stream unquarantine). So
    while D28 lists sources it now judges chance that the stream still
    quarantines on its word (detail.lift_pending), L23V waits for a person,
    at most D28.lift_wait_hours from the first tick DR0 deferred it (the
    ticker records that tick in context /lift_wait and keeps it across
    ticks; diagnoses are recomputed every tick). Returns {"state": ...}:
      waiting  L23V waits (sources named, or None while D28 cannot judge);
      expired  the bound passed: L23V is proposed as it stands;
      lifted   D28, judging a readable queue summary, lists nothing: L23V.
    A D28 that could not be judged (unknown, or lift_pending None: the queue
    summary unread) never ends a wait: the recorded one goes on with its
    sources, and without one a wait starts on the unknown (fail closed,
    bounded alike)."""
    det = (d28 or {}).get("detail") or {}
    known = bool(d28) and not d28.get("unknown") and det.get("lift_pending") is not None
    srcs = sorted({str(x) for x in det.get("lift_pending") or []}) if known else None
    rec = v.c("/lift_wait")
    rec = rec if isinstance(rec, dict) and rec.get("first_seen_utc") else None
    if known and not srcs:
        return {"state": "lifted", "first_seen_utc": rec["first_seen_utc"]} if rec else None
    if srcs is None and rec:
        srcs = rec.get("sources")              # D28 cannot judge this tick: the recorded wait goes on
    hours = float(_t(v.th, "D28", "lift_wait_hours"))
    now_s = v.c("/now_utc")
    first = rec["first_seen_utc"] if rec else now_s
    now, t0 = _utc(now_s), _utc(first)
    if now is None or t0 is None:
        return None
    until = t0 + datetime.timedelta(hours=hours)
    out = {"sources": srcs, "first_seen_utc": first, "until_utc": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "hours": hours, "stream": v.sid, "basis": "D28" if known else ("recorded" if rec and srcs else "unknown")}
    if now >= until:                           # the bound: L23V as it stands
        return dict(out, state="expired")
    out["commands"] = ["python -m weed_optimizer_framework.tools.inc2.stream unquarantine --source %s --stream %s "
                       "--decided-by human:<id>" % (x, v.sid) for x in srcs or []]
    return dict(out, state="waiting")


def _e1_qualified(v, exps):
    """(True, cites) when E2 may be built (amendment 2026-10-04): E1's
    verdict record (stream-domain e1.record) is decided, qualifies E1-B and
    names E1-B's experiment, and that experiment is done; else (False,
    why). Whether its three base weights still exist and hash as their
    run.json records is the build's check (inc2.baseline e2_record): a
    refusal there fails one E2 build, and E2's other builds wait behind it
    (r0), so a missing weight costs one build job and one card. Cites only after presence is checked
    (a cite of an absent value raises)."""
    e1 = v.dom.get("e1") or {}
    by_id = {b["id"]: b for b in (v.dom.get("baselines") or {}).get("items") or []}
    eb = (by_id.get((e1.get("arms") or {}).get("B")) or {}).get("exp")
    rec_name = e1.get("record")
    if not eb or not rec_name:
        return False, "the domain names no E1-B or no E1 record"
    rec = v.ev.json(rec_name)
    if not isinstance(rec, dict) or rec.get("status") != "decided":
        return False, "E1's verdict %s is not decided" % rec_name
    if rec.get("qualifies") is not True or rec.get("exp") != eb:
        return False, "E1's verdict does not qualify E1-B (%s)" % eb
    if exps.get(eb) != "done":
        return False, "E1-B (%s) is not done" % eb
    return True, [v.cite(rec_name, "/qualifies"), v.cite(rec_name, "/exp"),
                  v.ccite(E.pointer("stage", "exp_status", eb))]


def _e1a_done(v, exps):
    """(True, cites) when E2-C may be built besides E2's own gate
    (_e1_qualified; amendment 2026-10-04, later): E1's verdict names E1-A's
    experiment as its arm A (its `reference`) and that experiment is done;
    else (False, why). Whether E1-A's three base weights still exist and
    hash as their run.json records is the build's check, as for E1-B.
    Cites only after presence is checked."""
    e1 = v.dom.get("e1") or {}
    by_id = {b["id"]: b for b in (v.dom.get("baselines") or {}).get("items") or []}
    ea = (by_id.get((e1.get("arms") or {}).get("A")) or {}).get("exp")
    rec_name = e1.get("record")
    if not ea or not rec_name:
        return False, "the domain names no E1-A or no E1 record"
    rec = v.ev.json(rec_name)
    if not isinstance(rec, dict) or rec.get("reference") != ea:
        return False, "E1's verdict does not name E1-A (%s) as its arm A" % ea
    if exps.get(ea) != "done":
        return False, "E1-A (%s) is not done" % ea
    return True, [v.cite(rec_name, "/reference"), v.ccite(E.pointer("stage", "exp_status", ea))]


def r0(v, d28=None):
    """DR0: the rollout's prerequisites (contract 10 R0, R0b, R1, R2), each
    proposed once in its lane when due, in order: MAINT -- the splits build
    then lock (L23, a person approves the exact D-A command); the baselines
    (L23B: B_v2, the canary, the capacity arms, B0 u tsw); the canary's and
    the capacity grid's verdicts (LV); Stage A (L25, once a person accepted
    Protocol v3) and its verdict (LV); the stream's creation with Stage A's
    recipes (LI); the capacity decision adopted (LA); Stage C (L28); then,
    R0 complete, the measurement arms (L23B, baselines marked measure), and
    once one is done, its native-resolution rescore (L23N, once; not for an
    arm marked native false); E1 (2026-10-03): an arm that requires base3
    waits for splits v3, which is proposed once when that arm is next (L23V),
    after a bounded wait for a person to lift the quarantines D28 now judges
    chance (lift_wait; `d28` is D28's diagnosis of the same evidence), and
    once both E1 arms are done, their agnostic rescore and E1's verdict
    (L23E, once); E2 (2026-10-04): an arm that requires e1 is proposed only
    while E1's verdict is decided, qualifies E1-B and E1-B is done
    (_e1_qualified), one build at a time in the domain's order (a build that
    failed stays failed: /stage/baselines says so, and E2's other builds
    wait while it is), and once its six
    experiments and the reference are done, E2's rescore and verdict (L23C,
    once); E2-C (2026-10-04, later): its three builds after E2's six, under
    E2's gate and E1-A done (_e1a_done), in E2's group of builds, and once
    E2's verdict is recorded and E2-W's and E2-C's experiments are done,
    the attribution's rescore and record (L23D, once). DATA --
    the network probe (LP), then Step 1's one-time jobs after the lock (L17
    bootstrap, knowntruth, backfill), then D28-v2's sidecars for batches
    committed before the amendment (L17 eval-hits, _eval_hits_due)."""
    st = v.c("/stage") or {}
    out = {"MAINT": None, "DATA": None}
    cites = [v.ccite("/stage")]
    ss = _stream_state(v)
    wait, ended = None, None
    exps = v.c("/stage/exp_status") or {}
    ran = st.get("verdicts") or {}
    if not st.get("lock"):
        verb = "lock" if st.get("splits_built") else "build"
        out["MAINT"] = {"lever": "L23", "verb": verb, "why": "splits v2 %s" % ("built, not locked" if verb == "lock"
                                                                                else "not built")}
    else:
        items = [x for x in (v.dom.get("baselines") or {}).get("items") or [] if not x.get("measure")]
        for b in sorted(items, key=lambda x: not x.get("required")):
            if (st.get("baselines") or {}).get(b["id"]) not in (None, "missing"):
                continue
            out["MAINT"] = {"lever": "L23B", "baseline": b["id"], "why": "baseline %s (%s) not built" % (b["id"], b["exp"])}
            break
    if out["MAINT"] is None and st.get("lock"):
        for b in (v.dom.get("baselines") or {}).get("items") or []:
            if b.get("verdict") and exps.get(b["exp"]) == "done" and _record(v, b["exp"], "canary.json")[0] is None \
                    and not ran.get(b["exp"]):
                out["MAINT"] = {"lever": "LV", "module": "baseline", "verb": b["verdict"], "exp": b["exp"],
                                "key": b["exp"], "why": "%s is done without its verdict" % b["exp"]}
                break
    cap = v.dom.get("capacity") or {}
    cap_exps = [a.get("exp") for _k, a in sorted((cap.get("arms") or {}).items()) if a.get("exp")]
    if out["MAINT"] is None and cap_exps and all(exps.get(e) == "done" for e in cap_exps) \
            and v.ev.json("capacity/capacity_v1.json") is None and not ran.get("capacity"):
        out["MAINT"] = {"lever": "LV", "module": "baseline", "verb": cap.get("verdict") or "capacity-verdict",
                        "key": "capacity", "why": "the capacity arms %s are done without a decision" % cap_exps}
    sa = v.dom.get("stage_a") or {}
    sa_state = (st.get("stage_a") or {}).get("status")
    if out["MAINT"] is None and st.get("lock") and st.get("protocol_v3_accepted") and sa_state in (None, "missing"):
        out["MAINT"] = {"lever": "L25", "why": "Stage A (%s) not built; Protocol v3 accepted" % sa.get("exp")}
    sa_rec = _record(v, sa.get("exp"), sa.get("record") or "stage_a.json")[0]
    if out["MAINT"] is None and sa_state == "done" and (sa_rec or {}).get("status") != "READY" \
            and not ran.get("stage_a"):
        out["MAINT"] = {"lever": "LV", "module": "pilot4", "verb": "verdict", "exp": sa.get("exp"), "key": "stage_a",
                        "why": "%s is done without a READY verdict" % sa.get("exp")}
    if out["MAINT"] is None and (sa_rec or {}).get("status") == "READY" and not ss["init"] and st.get("lock"):
        recipes = ",".join(str(x) for x in sa_rec.get("segment1_recipes") or ["r0"])
        out["MAINT"] = {"lever": "LI", "stage_b": recipes, "why": "Stage A READY; the stream does not exist"}
        cites.append(v.cite("%s/%s" % (sa.get("exp"), sa.get("record") or "stage_a.json"), "/segment1_recipes"))
    if out["MAINT"] is None and ss["init"] and not ss["arm"] and v.ev.json("capacity/capacity_v1.json") is not None:
        out["MAINT"] = {"lever": "LA", "why": "the capacity decision is recorded; the stream has not adopted it"}
    if out["MAINT"] is None and ss["arm"] and not ss["stage_c_built"] and not st.get("stage_c_submitted"):
        out["MAINT"] = {"lever": "L28", "why": "Stage C not built; the stream's arm is fixed"}
    s1 = st.get("step1_stream") or {}
    if not st.get("placement") and not st.get("probe_ran"):
        out["DATA"] = {"lever": "LP", "why": "no placement.json: the network probe has not run"}
    elif st.get("lock"):
        for verb in ("bootstrap", "knowntruth", "backfill"):
            if not s1.get(verb):
                out["DATA"] = {"lever": "L17", "verb": verb, "why": "step1_stream %s has not run" % verb}
                break
        else:
            # D28-v2 (amendment 2026-10-03): batches committed before the amendment are weighed again into
            # sidecars, once each; until then D28 reads their dHash hits by the one-hit rule (fail closed)
            due, dcites = _eval_hits_due(v)
            if due:
                out["DATA"] = {"lever": "L17", "verb": "eval-hits",
                               "why": "D28-v2: %s count%s dHash hits on evaluation images that no record weighs and "
                                      "no sidecar has weighed yet" % (", ".join(due), "s" if len(due) == 1 else "")}
                cites += dcites
    # the measurement arms (baselines marked measure, 2026-09-30): proposed only
    # once R0 is complete (the stream's arm adopted, Stage C read) and nothing
    # else of it is due, so R0 READY never waits for them (they share the
    # stream's envelope, and a daily or monthly cap when the campaign declares
    # one; none has a default since the 2026-10-04 amendment, so a grant of
    # theirs no longer defers a segment's to the next UTC day);
    # capacity-verdict never reads them as candidates. They are built while the TRAIN lane runs, and /stage changes
    # with every experiment it builds: an envelope grant needs the diagnosis's
    # cites unchanged at submission, so the item cites only what it rests on
    # (the lock and the arm's own state)
    if out["MAINT"] is None and out["DATA"] is None and st.get("lock") and ss["arm"] and ss["stage_c_read"]:
        for b in (v.dom.get("baselines") or {}).get("items") or []:
            if not b.get("measure") or (st.get("baselines") or {}).get(b["id"]) not in (None, "missing"):
                continue
            if b.get("requires") == "e1":
                # E2 (2026-10-04): built only from an E1-B that qualified and is done (the build checks its weights)
                ok, ec = _e1_qualified(v, exps)
                if not ok:
                    continue
                if b.get("e2") == "C":
                    # E2-C (2026-10-04, later): E2's gate, and E1-A (its init) done
                    ok, ac = _e1a_done(v, exps)
                    if not ok:
                        continue
                    ec = ec + ac
                # E2's builds are one group: while one of them is failed (its build ran and ended without the
                # experiment; its card names the refusal and the build command), the others wait. A refusal one
                # E2 build meets (E1-B's weights or records changed, the reference's definition) the next would
                # meet too, and L23C needs all six; the wait ends once the failed one's experiment exists
                if any(x.get("requires") == "e1" and (st.get("baselines") or {}).get(x["id"]) == "failed"
                       for x in (v.dom.get("baselines") or {}).get("items") or []):
                    continue
                out["MAINT"] = {"lever": "L23B", "baseline": b["id"],
                                "why": "E2 arm %s (%s, E2-%s seed %s) not built: E1-B qualified and is done%s; "
                                       "record only" % (b["id"], b["exp"], b.get("e2"), b.get("seeds"),
                                                        ", E1-A (E2-C's init) is done" if b.get("e2") == "C" else "")}
                cites = [v.ccite("/stage/lock"), v.ccite("/stage/baselines/%s" % b["id"])] + ec
                break
            if b.get("requires") == "base3" and st.get("base3") != "done":
                # E1 (2026-10-03): its manifest is splits v3's; proposed once, when this arm is next, never while
                # it runs or after it failed (a card); the arm is built only once summary.json says complete
                if st.get("base3") in (None, "missing"):
                    # the build reads the quarantine at its job's start: a source D28 now judges chance stays out
                    # of arm B for good unless a person lifts its quarantine first, so L23V waits for that, bounded
                    wait = lift_wait(v, d28)
                    if wait and wait["state"] == "waiting":
                        names = ", ".join(wait["sources"]) if wait["sources"] else None
                        wait["why"] = ("E1 arm %s (%s) trains splits v3, which is not built: base v3 build waits "
                                       "until %s %s" % (b["id"], b["exp"], wait["until_utc"], (
                                           "for a person to lift the stream's quarantine of %s, which D28 now judges "
                                           "chance (inc2.stream unquarantine; a card)" % names) if names else (
                                           "for D28 to judge the sources again (it could not: %s; a card)"
                                           % str((d28 or {}).get("summary") or "no diagnosis")[:200])))
                        cites = [v.ccite("/stage/lock"), v.ccite("/stage/base3"), v.ccite("/lift_wait")]
                        for x in wait["sources"] or []:
                            try:
                                cites.append(v.cite(v.qname, E.pointer("quarantined_sources", x)))
                            except KeyError:
                                pass
                        break
                    if wait:
                        ended = wait                   # expired or lifted: L23V as it stands (the ticker ends it)
                    out["MAINT"] = {"lever": "L23V", "baseline": b["id"],
                                    "why": "E1 arm %s (%s) trains splits v3, which is not built: base v3 build "
                                           "(record only)" % (b["id"], b["exp"])}
                    cites = [v.ccite("/stage/lock"), v.ccite("/stage/base3")]
                    break
                continue
            out["MAINT"] = {"lever": "L23B", "baseline": b["id"],
                            "why": "measurement arm %s (%s) not built; recorded, never a candidate of the capacity "
                                   "decision" % (b["id"], b["exp"])}
            cites = [v.ccite("/stage/lock"), v.ccite("/stage/baselines/%s" % b["id"])]
            if b.get("requires") == "base3":
                cites.append(v.ccite("/stage/base3"))
            break
    # a done measurement arm's native-resolution rescore (2026-10-01, pre-registered): proposed once, on the
    # arms' own conditions, when its experiment is done and its native scores are missing (/stage/native: the
    # rescore's record in the evidence, else what the platform ran); a failure of it is a card and it stays
    # failed, so it is never proposed again. Cites only what it rests on, as the arms' builds do.
    nat = (v.dom.get("capacity") or {}).get("native") or {}
    if out["MAINT"] is None and out["DATA"] is None and st.get("lock") and ss["arm"] and ss["stage_c_read"] \
            and nat.get("reference_exp"):
        for b in (v.dom.get("baselines") or {}).get("items") or []:
            if not b.get("measure") or b.get("native") is False or exps.get(b["exp"]) != "done" \
                    or (st.get("native") or {}).get(b["id"]) not in (None, "missing"):
                continue
            out["MAINT"] = {"lever": "L23N", "baseline": b["id"],
                            "why": "measurement arm %s (%s) is done without its native-resolution scores; read at "
                                   "its own imgsz against %s at 640 (record only)" % (b["id"], b["exp"],
                                                                                   nat["reference_exp"])}
            cites = [v.ccite("/stage/lock"), v.ccite(E.pointer("stage", "exp_status", b["exp"])),
                     v.ccite("/stage/native/%s" % b["id"])]
            break
    # E1 (2026-10-03): once both arms are done, their agnostic rescore and E1's verdict, once (record only)
    e1 = v.dom.get("e1") or {}
    if out["MAINT"] is None and out["DATA"] is None and st.get("lock") and ss["arm"] and ss["stage_c_read"] \
            and e1.get("arms"):
        by_id = {b["id"]: b for b in (v.dom.get("baselines") or {}).get("items") or []}
        ea, eb = by_id.get(e1["arms"].get("A")), by_id.get(e1["arms"].get("B"))
        if ea and eb and exps.get(ea["exp"]) == "done" and exps.get(eb["exp"]) == "done" \
                and st.get("agnostic") in (None, "missing"):
            out["MAINT"] = {"lever": "L23E",
                            "why": "E1's arms %s and %s are done without their agnostic rescore: E1's verdict "
                                   "(record only, dev)" % (ea["exp"], eb["exp"])}
            cites = [v.ccite("/stage/lock"), v.ccite(E.pointer("stage", "exp_status", ea["exp"])),
                     v.ccite(E.pointer("stage", "exp_status", eb["exp"])), v.ccite("/stage/agnostic")]
    # E2 (2026-10-04): once its six experiments and the reference are done, E2's rescore and verdict, once (record
    # only); /stage/e2 is done once capacity/e2_rescore.json says complete, else what the platform ran
    e2 = v.dom.get("e2") or {}
    if out["MAINT"] is None and out["DATA"] is None and st.get("lock") and ss["arm"] and ss["stage_c_read"] \
            and e2.get("arms"):
        by_id = {b["id"]: b for b in (v.dom.get("baselines") or {}).get("items") or []}
        e2_items = [by_id.get(i) for k in sorted(e2["arms"]) for i in e2["arms"][k]]
        ref = e2.get("reference_exp")
        if e2_items and all(e2_items) and ref and all(exps.get(b["exp"]) == "done" for b in e2_items) \
                and exps.get(ref) == "done" and st.get("e2") in (None, "missing"):
            out["MAINT"] = {"lever": "L23C",
                            "why": "E2's runs %s are done without their 12-class rescore at 640: E2's verdict (record "
                                   "only, dev)" % ", ".join(b["exp"] for b in e2_items)}
            cites = [v.ccite("/stage/lock")] + [v.ccite(E.pointer("stage", "exp_status", b["exp"]))
                                                for b in e2_items] + \
                [v.ccite(E.pointer("stage", "exp_status", ref)), v.ccite("/stage/e2")]
    # E2-C (2026-10-04, later): once E2's verdict is recorded (/stage/e2 done: whether E2-S - E2-C is computed rests
    # on its choice, and L23C scored E2-W's files) and E2-W's and E2-C's experiments are done, the attribution's
    # rescore and record, once (record only); /stage/e2_attr is done once capacity/e2_attr_rescore.json says complete
    ea2 = v.dom.get("e2_attr") or {}
    if out["MAINT"] is None and out["DATA"] is None and st.get("lock") and ss["arm"] and ss["stage_c_read"] \
            and ea2.get("arms"):
        by_id = {b["id"]: b for b in (v.dom.get("baselines") or {}).get("items") or []}
        at_items = [by_id.get(i) for k in sorted(ea2["arms"]) for i in ea2["arms"][k]]
        if at_items and all(at_items) and all(exps.get(b["exp"]) == "done" for b in at_items) \
                and st.get("e2") == "done" and st.get("e2_attr") in (None, "missing"):
            out["MAINT"] = {"lever": "L23D",
                            "why": "E2's verdict is recorded and E2-W's and E2-C's runs %s are done without E2-C's "
                                   "attribution (record only, dev)" % ", ".join(b["exp"] for b in at_items)}
            cites = [v.ccite("/stage/lock")] + [v.ccite(E.pointer("stage", "exp_status", b["exp"]))
                                                for b in at_items] + [v.ccite("/stage/e2"), v.ccite("/stage/e2_attr")]
    items = {k: x for k, x in out.items() if x}
    wait = wait if wait and wait["state"] == "waiting" else None
    if not items and not wait:
        return _silent("DR0", "no rollout prerequisite is due", cites=cites)
    levers = sorted({x["lever"] for x in items.values()})
    summary = ["%s: %s -> %s" % (k, x["why"], x["lever"]) for k, x in sorted(items.items())]
    if wait:
        summary.append("L23V waits: %s" % wait["why"])
    extra = {"lift_wait": wait} if wait else ({"lift_wait_end": ended} if ended else {})
    return _diag("DR0", True, "info", "; ".join(summary), cites, levers, None, dict({"due": items}, **extra))


def compare_due(v):
    """DCMP: a finished milestone, Stage C chain or bisect arm the stream has
    not decided -> LC (inc2.stream compare --exp), one at a time."""
    exps = v.c("/stage/exp_status") or {}
    led = v.stream_ledger()
    todo = []
    compared = {e.get("exp") for e in led if e.get("event") == "milestone" and e.get("phase") == "compare"}
    for i, e in enumerate(led):
        if e.get("event") == "milestone" and e.get("phase") == "build" and e.get("exp") not in compared:
            todo.append((e["exp"], "milestone", i))
    read = {e.get("exp") for e in led if e.get("event") == "feasibility" and e.get("phase") == "read"}
    for i, e in enumerate(led):
        if e.get("event") == "feasibility" and e.get("phase") == "build" and e.get("exp") not in read:
            todo.append((e["exp"], "stage C", i))
    decided = set()
    for e in led:
        if e.get("event") == "bisect" and e.get("phase") == "decide":
            decided |= set((e.get("decisions") or {}).keys())
    for i, e in enumerate(led):
        if e.get("event") == "bisect" and e.get("phase") == "build":
            for inc, exp in sorted((e.get("arms") or {}).items()):
                if inc not in decided:
                    todo.append((exp, "bisect arm for %s" % inc, i))
    ready = [(exp, what, i) for exp, what, i in todo if exps.get(exp) == "done"]
    if not ready:
        return _silent("DCMP", "nothing finished waits for its comparison" if not todo else
                       "%d comparison(s) wait for their experiment to finish" % len(todo))
    exp, what, i = ready[0]
    return _diag("DCMP", True, "info", "%s %s is finished and not decided -> LC" % (what, exp),
                 [v.cite(v.lname, "/%d/event" % i), v.ccite(E.pointer("stage", "exp_status", exp))], ["LC"], exp,
                 {"propose": {"lever": "LC", "exp": exp}})


def deferred_left(v, r):
    """The images a source's intake has left deferred (collect.intake step
    1b): 0 once its shards are done or when it has no intake batch; what the
    stream folded (`intake_deferred`); else its latest batch's summary.json
    in the evidence (shard.deferred_remaining, else yield.images_deferred: a
    batch committed before shards existed is the first of its fetch); None
    when neither says (unknown: D21 does not judge the source then)."""
    if not r.get("batch") or (r.get("shards_done") and r.get("shards_done") == r.get("batch")):
        return 0
    n = _num(r.get("intake_deferred"))
    if n is None:
        s = v.ev.json("intake/%s/summary.json" % r["batch"])
        if not isinstance(s, dict):
            return None
        sh = s.get("shard") if isinstance(s.get("shard"), dict) else {}
        n = _num(sh.get("deferred_remaining"))
        if n is None:
            n = _num((s.get("yield") or {}).get("images_deferred"))
    return int(n or 0)


def _unadmitted(v, r):
    """The source's committed intake batches not yet admitted, oldest first,
    whose summary.json the evidence holds (a shard committed by a job the
    stream saw fail is admitted too, never skipped). A source the stream
    recorded before admitted_batches existed has only its latest batch to
    admit (the stream admitted each batch while it was the latest)."""
    if isinstance(r.get("admitted_batches"), list):
        done, todo = set(r["admitted_batches"]), list(r.get("batches") or []) or (
            [r["batch"]] if r.get("batch") else [])
    else:
        done, todo = set(), [r["batch"]] if r.get("batch") else []
    out = []
    for b in todo:
        if b not in done and b not in out and v.ev.json("intake/%s/summary.json" % b) is not None:
            out.append(b)
    return out


def shard_waits(v):
    """Why a source's next intake shard (L16I on a shard_pending source)
    waits, or []. It comes after the work that delays E1's first result:
    DR0's DATA item (L17 eval-hits, Step 1's one-time jobs, the probe), which
    must run before DR0 can propose L23V (only when DATA has nothing due);
    and E1's base v3 build (L23V) while an arm that requires it is not built
    and the build is due (base v3 missing, R0 complete, so DR0 proposes it)
    or running. Every other state of base v3 (built, failed, over walltime,
    ended without a complete summary) lets the shards run, so they never
    wait without an end: inc2.base3 reads only the first shard of each
    fetch record, so a shard committed before or after a build never enters
    base v3."""
    out = []
    try:
        d = r0(v)
    except Exception:  # noqa: BLE001 - an R0 record that cannot be read holds the shard (fail closed)
        return ["DR0 cannot be read"]
    due = ((d.get("detail") or {}).get("due") or {}) if d.get("fired") else {}
    if due.get("DATA"):
        out.append("DR0's %s %s is due on the DATA lane" % (due["DATA"].get("lever"), due["DATA"].get("verb") or ""))
    st = v.c("/stage") or {}
    e1 = [b for b in (v.dom.get("baselines") or {}).get("items") or [] if b.get("requires") == "base3"
          and (st.get("baselines") or {}).get(b["id"]) in (None, "missing")]
    ss = _stream_state(v)
    r0_done = bool(st.get("lock")) and ss["arm"] and ss["stage_c_read"]
    if e1 and (st.get("base3") == "running" or (st.get("base3") in (None, "missing") and r0_done)):
        out.append("E1's base v3 build (L23V) is %s and arm %s waits for it: the shard runs after it, so E1's first "
                   "result is not delayed" % ("running" if st.get("base3") == "running" else "due", e1[0]["id"]))
    return out


def pipeline(v):
    """DPIPE: a source part-way through the DATA pipeline takes its next step
    (fetched on the lab -> sync; fetched -> intake; intaken -> admit its batch;
    an intake refused for class names -> L26 on the lab, then L16S of the
    names layer, after which the stream makes it fetched again). Then, last,
    a source whose intake left images deferred (shard_pending) -> L16I, its
    next shard, unless shard_waits holds it; a shard committed but not yet
    admitted is admitted first."""
    srcs = v.c("/sources") or {}
    boot = v.c("/stage/step1_stream/bootstrap")
    for s in sorted(srcs):
        r = srcs[s] or {}
        nxt = None
        if r.get("status") == "names_pending":
            if not r.get("pending_names"):
                continue                     # the round waits for the fold's names (a refusal line came first)
            if not r.get("names_resolved"):
                nxt = {"lever": "L26", "source": s}
            elif not r.get("names_synced"):
                nxt = {"lever": "L16S", "source": s, "names": 1}
        elif r.get("status") == "fetched" and r.get("placement") == "lab" and not r.get("synced"):
            nxt = {"lever": "L16S", "source": s}
        elif r.get("status") == "fetched":
            nxt = {"lever": "L16I", "source": s}
        elif r.get("status") in ("intaken", "shard_pending") and boot and _unadmitted(v, r):
            # admitted only once its intake summary (the guard's counts, D28) is observed; every committed
            # batch of the source, oldest first, so no shard is skipped
            nxt = {"lever": "L17", "verb": "admit", "intake": _unadmitted(v, r)[0], "source": s}
        if nxt:
            return _diag("DPIPE", True, "info", "source %s is %s -> %s" % (s, r.get("status"), nxt["lever"]),
                         [v.ccite(E.pointer("sources", s, "status"))], [nxt["lever"]], None, {"propose": nxt})
    # continuation shards (amendment 2026-10-03), after every other pipeline step and after DR0's DATA item and
    # E1's base v3 (shard_waits)
    pending = [s for s in sorted(srcs) if (srcs[s] or {}).get("status") == "shard_pending"]
    if pending:
        waits = shard_waits(v)
        if waits:
            return _silent("DPIPE", "%s wait%s for the next intake shard: %s" % (
                ", ".join(pending), "s" if len(pending) == 1 else "", "; ".join(waits)),
                cites=[v.ccite(E.pointer("sources", s, "status")) for s in pending] + [v.ccite("/stage")])
        s = pending[0]
        n = deferred_left(v, srcs[s] or {})
        return _diag("DPIPE", True, "info", "source %s has %d image(s) its intake deferred -> L16I (the next shard)"
                     % (s, n), [v.ccite(E.pointer("sources", s, "status"))], ["L16I"], None,
                     {"propose": {"lever": "L16I", "source": s, "deferred": n}})
    return _silent("DPIPE", "no source waits for its next pipeline step")


def holds(v):
    """DHOLD (contract 6.7 review, S26): a hold on the funnel past its deadline
    never waits silently -- h6_scan is served by the stream's own copy
    detector (L17 scan-holds); funnel_F9 becomes an R3 item for a person (LH).
    The counts are the stream summary's (the queue as the cutter folds it,
    releases included), else step1_stream's status."""
    q = v.queue() or {}
    qq = q.get("queue") if isinstance(q.get("queue"), dict) else {}
    past = qq.get("held_past_deadline")
    name, base = v.qname, "/queue/held_past_deadline"
    if not isinstance(past, dict):
        st = v.ev.json("step1_stream/status.json") or {}
        past, name, base = st.get("holds_past_deadline"), "step1_stream/status.json", "/holds_past_deadline"
    past = past if isinstance(past, dict) else {}
    out = []
    for h, lever in (("h6_scan", "L17"), ("funnel_F9", "LH")):
        n = _num(past.get(h))
        if n:
            out.append({"hold": h, "lever": lever, "rows": int(n)})
    if not out:
        return _silent("DHOLD", "no hold past its deadline")
    cites = [v.cite(name, base + E.pointer(x["hold"])) for x in out]
    return _diag("DHOLD", True, "warn", "; ".join("%d %s row(s) past the deadline -> %s" % (x["rows"], x["hold"],
                                                                                           x["lever"]) for x in out),
                 cites, sorted({x["lever"] for x in out}), None, {"holds": out})


def _good_milestone_on(v, pool):
    recs = ((v.queue() or {}).get("milestones") or {}).get("records") or {}
    for n, m in sorted(recs.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        if (m or {}).get("pool") == pool and (m.get("state") in ("external", "compared")) and m.get("verdict") != "hurts":
            return m.get("exp")
    return None


def bisect(v):
    """DBIS: after a rollback, the suspect increments are bisected by the
    platform (L27, at most one per rollback) once a good milestone on P_c can
    decide the arms; X4 stays for what it cannot separate."""
    led = v.stream_ledger()
    rb = [(i, e) for i, e in enumerate(led) if e.get("event") == "rollback"]
    if not rb:
        return _silent("DBIS", "no rollback")
    i, last = rb[-1]
    done = [e for e in led if e.get("event") == "bisect" and e.get("rollback_utc") == last.get("utc")]
    if done or v.c("/bisected") == last.get("utc"):
        return _silent("DBIS", "the last rollback was bisected")
    if not last.get("suspect"):
        return _silent("DBIS", "the rollback to %s suspended no increment" % last.get("to"))
    ms = _good_milestone_on(v, last.get("to"))
    if not ms:
        return _silent("DBIS", "no good milestone on %s to decide bisect arms with: card X4 stays with a person"
                       % last.get("to"))
    return _diag("DBIS", True, "info", "rollback to %s at %s: %d suspect increment(s) -> L27 (against %s)" % (
        last.get("to"), last.get("utc"), len(last.get("suspect") or []), ms),
        [v.cite(v.lname, "/%d/event" % i), v.cite(v.lname, "/%d/to" % i), v.cite(v.lname, "/%d/suspect" % i)],
        ["L27"], None, {"propose": {"lever": "L27", "from_pool": last.get("to"), "utc": last.get("utc"),
                                    "n_suspect": len(last.get("suspect") or [])}})


def known_truth(v):
    """DKT: the verifier refit triggers of 3.3 (proposed there, pre-registered
    here): a batch whose known-truth verified precision is shown under
    known_truth.min_precision, P(Binom(n, 1 - min_precision) >= errors) <
    known_truth.alpha, on at least known_truth.min_matched matched verified
    boxes (recomputed here from step1_stream's counts), or a species
    step1_stream lists with at least min_new_boxes new target boxes of which at
    least max_unknown_share are 'unknown' -> card X11 (a verifier refit is a
    versioned event, R4). A batch with no error never fires: the Wilson lower
    bound this rule replaced stays under 0.99 for a perfect batch below 381
    boxes."""
    from ..funnel import estimate as ES
    name = "step1_stream/status.json"
    st = v.ev.json(name) or {}
    p_min = float(_t(v.th, "known_truth", "min_precision"))
    alpha = float(_t(v.th, "known_truth", "alpha"))
    n_min = int(_t(v.th, "known_truth", "min_matched"))
    hits, cites = [], []
    for batch, b in sorted((st.get("knowntruth") or {}).items()):
        n, k = _num((b or {}).get("matched_verified")), _num((b or {}).get("verified_correct"))
        if n is None or k is None or n < n_min or k > n:
            continue
        errors = n - k
        pv = float(ES.binom_upper_tail(errors, n, 1.0 - p_min)) if errors > 0 else 1.0
        if pv < alpha:
            hits.append("batch %s: %d of %d verified boxes correct (%.4f), P(Binom(%d, %g) >= %d) = %.2g < %g, "
                        "precision under %g" % (batch, k, n, k / float(n), n, 1.0 - p_min, errors, pv, alpha,
                                                p_min))
            cites += [v.cite(name, E.pointer("knowntruth", batch, "matched_verified")),
                      v.cite(name, E.pointer("knowntruth", batch, "verified_correct"))]
    trig = (st.get("refit_triggers") or {}).get("species_unknown_share") or []
    for j, sp in enumerate(trig):
        hits.append("%s: >= %d new target boxes, >= %.2f unknown" % (
            sp, int(_t(v.th, "known_truth", "min_new_boxes")), float(_t(v.th, "known_truth", "max_unknown_share"))))
        cites.append(v.cite(name, "/refit_triggers/species_unknown_share/%d" % j))
    if not hits:
        return _silent("DKT", "no verifier refit trigger")
    return _diag("DKT", True, "warn", "; ".join(hits) + " -> card X11 (a verifier refit, R4)", cites, ["X11"], None,
                 {"triggers": hits})


# --------------------------------------------------------------- detect
HEALTH_STREAM = ("D10S", "D26", "D27")
ORDER = ("D10S", "D27", "D26", "D28", "DCAN", "D25", "D23", "D30", "D31", "D33", "D32", "D8S", "D22", "D24",
         "DCMP", "DBIS", "D21", "DHOLD", "DPIPE", "D20", "D29", "DR0", "DSA", "DSC", "DCAP", "DKT", "DNAT")


def detect(ev, dom, th=None, prior=None, only=None, include_prior_d31=False):
    """Every stream diagnosis (fired or not), in ORDER (the order the ticker
    takes proposals in: stops, then TRAIN, then DATA, then MAINT's R0 records).
    `prior`: the prior experiment's evidence (prior_evidence(dom)) for D30/D33
    before Stage A is READY."""
    th = th if th is not None else LS.load_thresholds()
    v = View(ev, dom, th)
    out = {}
    sc = None

    def run(did, fn, *a, **kw):
        if only is not None and did not in only:
            return None
        try:
            out[did] = fn(v, *a, **kw)
        except _Missing as e:
            out[did] = _unknown(did, "threshold %s is not declared in stream_thresholds.json" % e.args[0])
        except Exception as e:                       # evidence is untrusted input
            out[did] = _unknown(did, "the rule raised %s (%s)" % (type(e).__name__, str(e)[:200]))
        return out.get(did)

    run("D10S", d10s)
    run("D27", d27)
    run("D26", d26)
    r28 = run("D28", d28)
    run("DCAN", canary)
    run("D25", d25)
    run("D23", d23)
    r30 = run("D30", d30, prior)
    run("D31", d31, prior, include_prior_d31)
    sc = run("DSC", stage_c)
    run("D33", d33, prior, (sc or {}).get("detail") if (sc or {}).get("fired") else None)
    run("D32", d32, bool((r30 or {}).get("fired")))
    run("D8S", d8s)
    run("D22", d22)
    run("D24", d24)
    run("DCMP", compare_due)
    run("DBIS", bisect)
    run("D21", d21)
    run("DHOLD", holds)
    run("DPIPE", pipeline)
    run("D20", d20)
    run("D29", d29)
    run("DR0", r0, r28)
    run("DSA", stage_a)
    run("DCAP", capacity)
    run("DKT", known_truth)
    run("DNAT", native)
    return [out[k] for k in ORDER if k in out]


def fired(diags):
    return [d for d in diags if d.get("fired")]


def by_id(diags):
    return {d["id"]: d for d in diags}


# ------------------------------------------------------- prospective record
def prospective_record(dom, th, cfg, stage, now_utc):
    """The prospective stream record (contract 6.8, as R4b): before the first
    L18, the sha256 of (M, K, the truth policy, the thresholds, the recipe rule
    and the gate block) goes to the ledger. `stage`: the Stage A record and
    the capacity decision."""
    body = {"format": "inc-autopilot/prospective-stream/1", "sid": dom.get("sid"),
            "M": (stage or {}).get("M") or (dom.get("increment") or {}).get("M"), "K_max": _t(th, "D22", "K_max"),
            "truth": {"policy": "on for every step (P3), every ceil(cost/25)-th step above 25 GPU-h (L-4)",
                      "step_hours_max": _t(th, "capacity", "truth_step_hours_max")},
            "thresholds_sha256": hashlib.sha256(json.dumps(th, sort_keys=True).encode("utf-8")).hexdigest(),
            "recipe_rule": {"stage_a": (stage or {}).get("stage_a"), "stage_b": "smallest median delta_min; then fewer "
                            "recipe flags; then the cheaper recipe; a tie to r0 (contract 5.1)"},
            "gate": {"flips_mode": "net", "species_tolerance": "per species, L-3 (Protocol v3)"},
            "capacity": (stage or {}).get("capacity"), "rules_version": rules_version(),
            "domain_config_sha256": dom.get("_sha256"), "decided_utc": now_utc,
            "protocol_package": cfg.get("protocol_package") or dom.get("protocol_package")}
    body["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != "decided_utc"},
                                               sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return body
