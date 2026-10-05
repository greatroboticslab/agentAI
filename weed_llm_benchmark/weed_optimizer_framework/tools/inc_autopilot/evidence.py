"""INC autopilot evidence: a dev-only snapshot of an INC campaign in which
every value has an address (docs/INC_AUTOPILOT.md, (a) component 1 and (d)).

What it loads, and nothing else (the allow-list, ALLOWED):
    <exp>/exp.json, <exp>/state.json, <exp>/report.json, <exp>/build_summary.json,
    <exp>/ledger.jsonl, <exp>/audit/label_audit.json,
    audit/<exp>_audit.json                 (the same label audit written under
                                            INC_DIR/audit/, as pilot_v1's was:
                                            its "outputs" record that path)
    <exp>/manifests/increments_summary.json,
    step1/select_summary.json, step1/admit_summary.json,
    step1/increments_summary.json, step1/relevance.json,
    step1/select_clusters_by_source.json   (remote.py clusters_by_source: the
                                            per-source status counts; the CSV
                                            itself is never shipped)
    <exp>/derived/state_runs.json          (remote.py summarize_runs, when the
                                            snapshot ships state.json without runs)
    step1/pool_summary.json, step1/calibration.json,
    step1/verifier_fit_info.json           (the FUNNEL audit's Step 1 reads,
                                            docs/FUNNEL_AUDIT.md 8.2: label spaces
                                            and drops, what the known truth
                                            covers, and the verifier's fit record
                                            projected to its OtherPlant sample,
                                            which remote.py derives on the cluster)
    funnel/funnel_ledger.json, funnel/audit_v1.json, funnel/class_maps.json,
    funnel/recovery.json, funnel/prospective_da.json
                                           (the funnel audit's aggregates;
                                            recovery.json lives in step1_r1/ and is
                                            shipped under this name)
    funnel/files.json                      (remote.py funnel summary: the name,
                                            sha256 and size of every file under
                                            INC_DIR/funnel/, never their content;
                                            lever preconditions read it)
plus the campaign context the ticker passes in as a dict (artifact
campaign/context.json; every key optional; diagnose.py and levers.py read
exactly these):
    refusals  [{"builder": str, "message": str}]       builder refusals (D2, DREF)
    advance   {"error": str}                           the last advance's error (D7)
    squeue    [job name, ...]                          the user's queued jobs (D6)
    now_utc   "YYYY-MM-DDTHH:MM:SSZ"                   the tick's clock (D6)
    history   [{"utc": str, "generation": int}]        earlier snapshots of this experiment (D6)
    budget    {"envelope_su", "spent_su", "projected_su"?}                              (D10)
    outcomes  [{"lever", "child_exp", "predicted": {"direction": "up"|"down"},
                "verdict": "better"|"worse"|"within_noise"|"insufficient"}]              (D13)
    lineage   [{"lever", "parent_exp", "child_exp", "status"?}]  the campaign's
              lever executions that ran or are in flight (the campaign ledger /
              executions.jsonl); levers.py never proposes a lever again on a
              parent it was already applied to. A record whose status is
              "failed", "refused" or "cancelled" does not count.
and the campaign's claims register, passed in the same way (artifact
campaign/claims.json, funnel-claims/1, docs/FUNNEL_AUDIT.md 8.3).
A name outside the list is refused and recorded, so a run directory, a score
file or a manifest can never enter the evidence.

Test-blind by construction (docs/INC_AUTOPILOT.md (d), "Test split is never
read in decisions", item 1). Every artifact is scrubbed before anything can
read it. The rules are an allow-list of the one decision exam (dev), so an
exam added upstream later is dropped without anyone updating a list here:
  * under a dict key "exams" (report.json's final rows, report.py:56 puts
    test in every one), a dict keeps only its "dev" entry;
  * anywhere else, a dict key naming a known non-dev split (BLOCKED_SPLITS:
    the domain config's non-decision exams, NON_DEV_EXAMS, which equal
    driver.FINAL_EXAMS and report.REPORT_EXAMS without dev, checked by
    tests/test_inc_ap_evidence.py, plus its extra non-decision splits, the
    weed domain's H10d domain dev; model.non_dev_exams) is dropped with
    everything under it. A loader given another domain (domain=) uses that
    domain's list;
  * a dict whose own "exam" field is a string other than "dev" (a score
    stamp) is dropped;
  * a string that is the path of a score file other than scores/dev.json
    is dropped.
A dropped value under a dict key goes with its key; a dropped list element
becomes None, so list positions are kept and a JSON pointer into the scrubbed
artifact addresses the same value in the file. Every
dropped address is recorded in the loader record (the virtual artifact
LOADER), which diagnose.D14 checks. The files are opened by _read_file only,
and only for allow-listed names under the snapshot root: no score file, run
directory or test manifest is ever opened.

Addresses (model.cite): a JSON artifact is cited by JSON pointer (RFC 6901)
into the scrubbed artifact; a ledger value by its 1-based line and a pointer
into that line's entry. Evidence.resolve(cite) re-reads a cite, and
Evidence.check_cite(cite) says whether its value still matches.

Loaders: load_dir(root, exp) reads a snapshot tree laid out like INC_DIR
(the replay fixtures, a pulled copy); from_texts({name: bytes}) takes offered
files; from_snapshot(record) takes remote.py's INCAP snapshot record and
gives the same evidence as the files it was made from (tests/test_inc_ap_replay.py,
"live path").

Evidence.canonical() is the byte form of everything a decision can read (the
scrubbed artifacts, ledgers and context, and what the loader read or
refused). It holds neither the raw files' hashes nor the list of dropped
addresses: both change with non-dev content, and the canonical bytes must
not (tests/test_inc_ap_evidence.py, the metamorphic case). The raw sha256 of
every file is kept apart, in Evidence.provenance, for the campaign ledger:
{"sha256", "bytes", "source": "file"} for a file read here or offered as
bytes; from a remote snapshot, the cluster file's own sha256 and size as the
record gives them ("source": "cluster file", plus "shipped_as" when the
record shipped the file compacted); for an artifact the snapshot derived
(derived/state_runs.json, step1/select_clusters_by_source.json, a report.json
whose final table was put back from its dev column) the sha256 of the JSON
this module built, marked "source": "derived", never presented as a file hash.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import model as M

FORMAT = "inc-autopilot/evidence/1"
# The domain's non-decision exams (model.exam_splits: the domain config's
# exams.non_decision), which are driver.FINAL_EXAMS / report.REPORT_EXAMS
# without dev (tests/test_inc_ap_evidence.py checks the equality), and every
# split no decision may read (model.non_dev_exams: those plus the config's
# extra non-decision splits). Only the key deny-list outside an "exams" dict
# uses them; the "exams", stamp and score-path rules are an allow-list of
# DECISION_EXAM.
NON_DEV_EXAMS = M.exam_splits()["non_decision"]
BLOCKED_SPLITS = M.non_dev_exams()
DECISION_EXAM = M.DECISION_EXAM
EXAMS_KEY = "exams"                        # report.json final[].exams: {exam: {...}}
assert DECISION_EXAM == "dev" and DECISION_EXAM not in BLOCKED_SPLITS
assert set(NON_DEV_EXAMS) <= set(BLOCKED_SPLITS)

CONTEXT = "campaign/context.json"          # passed in by the ticker, never read from disk
CLAIMS = "campaign/claims.json"            # the claims register, passed in by the ticker
LOADER = "evidence/loader.json"            # virtual: what the loader opened, refused and dropped
CLUSTERS_BY_SOURCE = "select_clusters_by_source.json"
VERIFIER_FIT_INFO = "verifier_fit_info.json"   # remote.py's projection of step1/verifier/fit_info.json

EXP_RE = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
RESERVED_DIRS = ("step1", "splits", "exams", "logs", "audit", "funnel",   # INC_DIR entries that are not experiments
                 "stream", "step1_stream", "intake")
ROOT_AUDIT = "audit/%s_audit.json"          # a label audit run with --out $INC/audit/<exp>_audit.json
EXP_FILES = ("exp.json", "state.json", "report.json", "build_summary.json", "ledger.jsonl",
             "audit/label_audit.json", "manifests/increments_summary.json")
STEP1_FILES = ("select_summary.json", "admit_summary.json", "increments_summary.json",
               "relevance.json", CLUSTERS_BY_SOURCE, "pool_summary.json", "calibration.json",
               VERIFIER_FIT_INFO)
# The funnel audit's aggregates (docs/FUNNEL_AUDIT.md 8.2). recovery.json lives
# in INC_DIR/step1_r1/ and is shipped as funnel/recovery.json; files.json is the
# funnel directory's listing (names, sha256, sizes), never a file of the tree.
FUNNEL_FILES = ("funnel_ledger.json", "audit_v1.json", "class_maps.json", "recovery.json",
                "prospective_da.json", "files.json")
FUNNEL_LEDGER = "funnel/funnel_ledger.json"
FUNNEL_AUDIT = "funnel/audit_v1.json"
FUNNEL_CLASS_MAPS = "funnel/class_maps.json"
FUNNEL_RECOVERY = "funnel/recovery.json"
FUNNEL_DA = "funnel/prospective_da.json"
FUNNEL_LISTING = "funnel/files.json"
# The shipped name of a file that lives elsewhere on the cluster, and the
# derived artifacts built there from a file that is never shipped.
SHIPPED_AS = {FUNNEL_RECOVERY: "step1_r1/recovery.json"}
DERIVED_FROM = {"step1/" + VERIFIER_FIT_INFO: "step1/verifier/fit_info.json",
                FUNNEL_LISTING: "funnel/ (a listing of names, sha256 and sizes)"}
DERIVED_STATE_RUNS = "derived/state_runs.json"
# The stream's artifacts (docs/CONTINUOUS_LOOP.md 3.8): the cutter's queue
# summary, the stream ledger and the dev scores of the stream's base runs
# (remote stream-snapshot, dev-stamped only), Step 1's stream status, the intake
# batch summaries, the fold of intake/sources.jsonl and the network probe's
# placement, the splits lock's status (derived: presence, sha256, version;
# never the LOCK's manifest table), and the R0 verdicts, which read dev only
# (inc2.baseline's capacity/capacity_v1.json and <exp>/canary.json,
# inc2.pilot4's <exp>/stage_a.json; capacity_v1_report.* holds test and is not
# on the list), and the measurement arms' native-resolution records, which
# read dev only (inc2.baseline's capacity/native_v1.json and
# <exp>/native_rescore.json; native_v1_report.* holds a non-decision exam and
# is not on the list), and D28-v2's intake sidecars, intake/<batch>/eval_hits.json,
# which weigh again the dHash hits of an intake batch committed before the
# amendment and which the snapshot ships for D28 (pair cosines and hit keys,
# no evaluation key, no pixels), and E1's records (2026-10-03): splits v3's
# summary.json (inc2.base3: the two arms' manifests by sha256 and per-source
# counts; its held-out set is named holdout_v1, never after a split),
# <exp>/agnostic_rescore.json (inc2.baseline rescore-agnostic: the dev files'
# names and sha256s) and capacity/e1_v1.json (E1's verdict, dev only;
# e1_v1_report.* is for people and not on the list), and E2's records
# (2026-10-04): capacity/e2_v1.json (E2's verdict, dev only) and
# capacity/e2_rescore.json (inc2.baseline rescore-e2: the dev files' names and
# sha256s and the verdict's sha256); e2_v1_report.* (ImageWeeds, for people),
# capacity/e2_test_*.* and <exp>/e2_test_read.json (the person's read of the
# sealed test) are never on the list; E2-C's (2026-10-04, later):
# capacity/e2_attr_v1.json (the attribution record, dev only) and
# capacity/e2_attr_rescore.json (inc2.baseline rescore-e2-attr: the dev files'
# names and sha256s and the record's sha256), never e2_attr_v1_report.md; and
# the model-zoo audit's record (2026-10-04, L23Z): capacity/zoo_v1.json
# (status, job ids, counts and sha256s: no exam name as a key, no score path,
# no metric), never its reports under _zoo/. A Step 1 sidecar,
# step1_stream/eval_hits/<batch>.json, is not on the list: status.json carries
# its fold, and the sidecar names each hit's matched evaluation key. A stream
# ledger has no experiment: it is kept as a JSON list artifact, not under
# Evidence.ledgers.
BATCH_RE = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
STREAM_FILES = ("queue_summary.json", "ledger.jsonl", "dev_scores.json")
ALLOWED = tuple(re.compile(p) for p in (
    r"(?P<exp>%s)/(?P<file>%s)\Z" % (EXP_RE, "|".join(re.escape(f) for f in EXP_FILES + (DERIVED_STATE_RUNS,))),
    r"step1/(?P<file>%s)\Z" % "|".join(re.escape(f) for f in STEP1_FILES),
    r"audit/(?P<exp>%s)_audit\.json\Z" % EXP_RE,
    r"funnel/(?P<file>%s)\Z" % "|".join(re.escape(f) for f in FUNNEL_FILES),
    r"stream/(?P<sid>%s)/(?P<file>%s)\Z" % (EXP_RE, "|".join(re.escape(f) for f in STREAM_FILES)),
    r"step1_stream/status\.json\Z",
    r"intake/(?P<batch>%s)/(?:summary|eval_hits)\.json\Z" % BATCH_RE,
    r"intake/(?P<file>sources|placement)\.json\Z",
    r"splits/(?P<ver>v[0-9]{1,3})/lock_status\.json\Z",
    # the model-zoo audit's record (L23Z, 2026-10-04): status, job ids, counts and sha256s only
    r"capacity/(?:(?:capacity|native|e1|e2|e2_attr|zoo)_v1|e2_rescore|e2_attr_rescore)\.json\Z",
    r"splits/v3/summary\.json\Z",
    r"(?!(?:%s)/)(?P<rexp>%s)/(?P<record>canary|stage_a|native_rescore|agnostic_rescore)\.json\Z"
    % ("|".join(RESERVED_DIRS), EXP_RE),
))
# the path of any score file but the decision exam's (scores/<exam>.json, driver.Paths.score)
_NON_DEV_SCORE = re.compile(r"(^|/)scores/(?!%s\.json\Z)[^/]+\.json\Z" % re.escape(DECISION_EXAM))
_DROP = object()


class EvidenceError(ValueError):
    """A snapshot that cannot be turned into evidence."""


# ------------------------------------------------------------------ names
def allowed(name):
    """True when `name` (a snapshot-relative artifact name) is on the allow-list."""
    if not isinstance(name, str):
        return False
    m = ALLOWED[0].match(name)
    if m:
        return m.group("exp") not in RESERVED_DIRS
    return any(rx.match(name) for rx in ALLOWED[1:])


def exp_of(name):
    """The experiment directory an allow-listed artifact sits in, or None
    (step1, and audit/<exp>_audit.json, which sits in INC_DIR/audit/)."""
    m = ALLOWED[0].match(name) if isinstance(name, str) else None
    return m.group("exp") if m and m.group("exp") not in RESERVED_DIRS else None


def _esc(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def pointer(*parts):
    """A JSON pointer from its parts: pointer("chains", "full", "verdict")."""
    return "".join("/" + _esc(p) for p in parts)


def _parts(ptr):
    if ptr in ("", None):
        return []
    if not ptr.startswith("/"):
        raise KeyError("JSON pointer %r does not start with '/'" % ptr)
    return [p.replace("~1", "/").replace("~0", "~") for p in ptr[1:].split("/")]


def walk(obj, ptr):
    """The value at JSON pointer `ptr` in obj; KeyError when it is absent."""
    cur = obj
    for p in _parts(ptr):
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(ptr)
            cur = cur[p]
        elif isinstance(cur, list):
            if not p.isdigit() or int(p) >= len(cur):
                raise KeyError(ptr)
            cur = cur[int(p)]
        else:
            raise KeyError(ptr)
    return cur


# ------------------------------------------------------------------ scrub
def _non_dev_stamp(d):
    """A dict stamped with an exam other than the decision exam (a score record)."""
    return isinstance(d.get("exam"), str) and d["exam"] != DECISION_EXAM


def blocked_for(domain=None):
    """The split names the key deny-list drops for a domain (a name or a config
    path; None: this module's own domain, BLOCKED_SPLITS)."""
    return BLOCKED_SPLITS if domain is None else M.non_dev_exams(domain)


def _non_dev_key(k, parent, blocked=None):
    """A dict key that names a non-dev exam: any key but dev directly under an
    "exams" dict (allow-list), or a known non-dev split name anywhere."""
    return k in (BLOCKED_SPLITS if blocked is None else blocked) or (parent == EXAMS_KEY and k != DECISION_EXAM)


def scrub(obj, blocked=None):
    """(dev-only copy of obj, [dropped JSON pointers]). See the module doc.
    `blocked`: the split names to drop (default BLOCKED_SPLITS; blocked_for)."""
    dropped = []

    def go(x, ptr, parent):
        if isinstance(x, dict):
            if _non_dev_stamp(x):
                dropped.append(ptr or "/")
                return _DROP
            out = {}
            for k, v in x.items():
                p = ptr + "/" + _esc(k)
                if _non_dev_key(k, parent, blocked):
                    dropped.append(p)
                    continue
                w = go(v, p, k)
                if w is not _DROP:
                    out[k] = w
                # a dropped value under a kept key: the key goes with it
            return out
        if isinstance(x, list):
            out = []
            for i, v in enumerate(x):
                w = go(v, "%s/%d" % (ptr, i), None)
                out.append(None if w is _DROP else w)     # positions are kept
            return out
        if isinstance(x, str) and _NON_DEV_SCORE.search(x):
            dropped.append(ptr or "/")
            return _DROP
        return x

    res = go(obj, "", None)
    return (None if res is _DROP else res), dropped


def leaks(obj, ptr="", parent=None, blocked=None):
    """JSON pointers in obj that still name a non-dev exam (a dict key, a
    stamp or a score path, by scrub's rules); [] for scrubbed evidence."""
    out = []
    if isinstance(obj, dict):
        if _non_dev_stamp(obj):
            out.append(ptr or "/")
        for k, v in obj.items():
            p = ptr + "/" + _esc(k)
            if _non_dev_key(k, parent, blocked):
                out.append(p)
            out.extend(leaks(v, p, k, blocked))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(leaks(v, "%s/%d" % (ptr, i), None, blocked))
    elif isinstance(obj, str) and _NON_DEV_SCORE.search(obj):
        out.append(ptr or "/")
    return out


# ------------------------------------------------------------------ files
def _read_file(path):
    """The only function here that opens a file (the test-blindness tests
    wrap it to record every path)."""
    with open(path, "rb") as fh:
        return fh.read()


def _sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def _text(data):
    return data.decode("utf-8") if isinstance(data, bytes) else str(data)


def parse_ledger(text):
    """[(1-based line, entry)] of every complete JSON line. A partial last
    line (no trailing newline: an advance killed mid-append, which the next
    advance repairs; report.read_ledger skips it the same way) is skipped and
    returned as a note. Any other line that is not JSON raises EvidenceError."""
    lines = text.split("\n")
    out, notes = [], []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except ValueError as err:
            if i == len(lines) - 1:
                notes.append("line %d is partial (%d bytes) and was skipped" % (i + 1, len(ln)))
                continue
            raise EvidenceError("ledger line %d is not JSON: %s" % (i + 1, err))
        out.append((i + 1, e))
    return out, notes


# ---------------------------------------------------------------- evidence
class Evidence:
    """One dev-only snapshot. `exp` is the experiment the ticker is looking
    at; the snapshot may hold other experiments of the campaign (earlier
    pilots, the baseline) for the cross-experiment diagnoses."""

    def __init__(self, exp, blocked=None):
        self.exp = exp
        self.blocked = BLOCKED_SPLITS if blocked is None else tuple(blocked)
        self.artifacts = {}          # name -> scrubbed JSON
        self.ledgers = {}            # exp -> [(line, scrubbed entry or None)]
        self.provenance = {}         # name -> {"sha256": raw bytes, "bytes": n}; not in canonical()
        self.touched = []            # names the loader read (in order)
        self.refused = []            # names that were offered and refused
        self.dropped = {}            # name -> [pointers dropped by the scrub]
        self.notes = []

    # ---- reading
    def has(self, name):
        return name in self.artifacts or (name.endswith("/ledger.jsonl") and exp_of(name) in self.ledgers)

    def json(self, name):
        """The scrubbed artifact, or None when the snapshot does not hold it."""
        if name == LOADER:
            return self.loader_record()
        return self.artifacts.get(name)

    def get(self, name, ptr, default=KeyError):
        obj = self.json(name)
        try:
            if obj is None:
                raise KeyError(name)
            return walk(obj, ptr)
        except KeyError:
            if default is KeyError:
                raise
            return default

    def ledger(self, exp=None):
        """[(line, entry)] of exp's ledger, entries the scrub removed left out."""
        return [(ln, e) for ln, e in self.ledgers.get(exp or self.exp, []) if isinstance(e, dict)]

    def latest(self, exp=None, type_=None):
        """{entry id: (line, entry)}, a later line replacing an earlier one with
        the same id (the report's rule, report.read_ledger)."""
        out = {}
        for ln, e in self.ledger(exp):
            if type_ is None or e.get("type") == type_:
                out[e.get("id")] = (ln, e)
        return out

    def exps(self):
        """Experiments in the snapshot, oldest first (exp.json initialised_utc,
        then name); an experiment without exp.json sorts last by name."""
        names = set(self.ledgers)
        names |= {exp_of(n) for n in self.artifacts if exp_of(n)}
        names.discard(None)

        def key(e):
            d = self.json("%s/exp.json" % e) or {}
            return (0 if d.get("initialised_utc") else 1, str(d.get("initialised_utc") or ""), e)
        return sorted(names, key=key)

    # ---- addresses
    def cite(self, name, ptr):
        """model.cite of the value at ptr in artifact `name` (KeyError if absent)."""
        return M.cite(name, self.get(name, ptr), pointer=ptr)

    def lcite(self, line, ptr, exp=None):
        """model.cite of the value at ptr in the ledger entry on `line`."""
        exp = exp or self.exp
        for ln, e in self.ledgers.get(exp, []):
            if ln == line:
                if not isinstance(e, dict):
                    raise KeyError("%s/ledger.jsonl line %d was scrubbed" % (exp, line))
                return M.cite("%s/ledger.jsonl" % exp, walk(e, ptr), line=line, pointer=ptr)
        raise KeyError("%s/ledger.jsonl has no line %d" % (exp, line))

    def resolve(self, c):
        """The value a cite addresses, from this snapshot (KeyError if absent)."""
        art = c.get("artifact")
        if c.get("line") is not None:
            exp = exp_of(art) if art and art.endswith("/ledger.jsonl") else None
            if exp is None:
                raise KeyError("a line cite must name <exp>/ledger.jsonl, not %r" % art)
            return self.lcite(int(c["line"]), c.get("pointer") or "", exp=exp)["value"]
        return self.get(art, c.get("pointer") or "")

    def check_cite(self, c):
        """True when the cite resolves here to exactly its recorded value."""
        try:
            v = self.resolve(c)
        except (KeyError, TypeError, ValueError):
            return False
        return json.dumps(v, sort_keys=True) == json.dumps(c.get("value"), sort_keys=True)

    # ---- records
    def loader_record(self):
        return {"format": FORMAT, "exp": self.exp, "touched": list(self.touched),
                "refused": list(self.refused),
                "dropped": {k: list(v) for k, v in sorted(self.dropped.items())},
                "notes": list(self.notes)}

    def canonical(self):
        """Deterministic bytes of everything a decision can read: the scrubbed
        artifacts and ledgers and what the loader read or refused. Neither the
        raw file hashes nor the list of dropped addresses is in it: both depend
        on non-dev content (its bytes, its presence), and these bytes must not."""
        rec = self.loader_record()
        rec.pop("dropped")
        body = {"format": FORMAT, "exp": self.exp,
                "artifacts": self.artifacts,
                "ledgers": {e: [[ln, x] for ln, x in v] for e, v in self.ledgers.items()},
                "loader": rec}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def leaks(self):
        """Pointers that name a non-dev exam anywhere in the evidence ([] by
        construction); D14 fires when it is not empty."""
        out = []
        for n, a in sorted(self.artifacts.items()):
            out.extend("%s#%s" % (n, p) for p in leaks(a, blocked=self.blocked))
        for e, v in sorted(self.ledgers.items()):
            for ln, x in v:
                out.extend("%s/ledger.jsonl:%d#%s" % (e, ln, p) for p in leaks(x, blocked=self.blocked))
        return out


# ------------------------------------------------------------------ loaders
def from_texts(texts, exp, context=None, touched=None, claims=None, domain=None):
    """Evidence from {artifact name: bytes or str} (the remote snapshot verb's
    small files, or load_dir's reads). `context` is the ticker's dict
    (refusals, advance, squeue, history, budget, outcomes); `claims` the
    campaign's claims register (funnel-claims/1), kept as CLAIMS; `domain`
    the domain whose non-decision splits the scrub drops (default this
    package's, model.DOMAIN)."""
    if not isinstance(exp, str) or not re.fullmatch(EXP_RE, exp):
        raise EvidenceError("experiment %r is not a valid name" % (exp,))
    blocked = blocked_for(domain)
    ev = Evidence(exp, blocked=blocked)
    # touched = what was read: load_dir passes the files it opened; for
    # offered texts it is the allow-listed ones (a refused name is recorded
    # in refused and never parsed).
    ev.touched = list(touched if touched is not None else [n for n in sorted(texts) if allowed(n)])
    for name in sorted(texts):
        if not allowed(name):
            ev.refused.append(name)
            continue
        data = texts[name]
        raw = data if isinstance(data, bytes) else str(data).encode("utf-8")
        ev.provenance[name] = {"sha256": _sha(raw), "bytes": len(raw), "source": "file"}
        if name in SHIPPED_AS:
            ev.provenance[name]["shipped_as"] = SHIPPED_AS[name]
        if name in DERIVED_FROM:
            ev.provenance[name] = {"source": "derived", "json_sha256": _sha(raw),
                                   "derived_from": DERIVED_FROM[name]}
        text = _text(data)
        if name.endswith("/ledger.jsonl") and exp_of(name) is None:
            # a stream ledger (stream/<sid>/ledger.jsonl): a list artifact
            try:
                entries, notes = parse_ledger(text)
            except EvidenceError as e:
                ev.notes.append("%s: %s; not loaded" % (name, e))
                continue
            ev.notes.extend("%s: %s" % (name, n) for n in notes)
            clean, drop = scrub([x for _ln, x in entries], blocked)
            ev.artifacts[name] = clean
            if drop:
                ev.dropped[name] = drop
            continue
        if name.endswith("/ledger.jsonl"):
            try:
                entries, notes = parse_ledger(text)
            except EvidenceError as e:
                ev.notes.append("%s: %s; not loaded" % (name, e))
                continue
            ev.notes.extend("%s: %s" % (name, n) for n in notes)
            rows, drop = [], []
            for ln, entry in entries:
                clean, d = scrub(entry, blocked)
                rows.append((ln, clean))
                drop.extend("%d#%s" % (ln, p) for p in d)
            ev.ledgers[exp_of(name)] = rows
            if drop:
                ev.dropped[name] = drop
            continue
        try:
            obj = json.loads(text)
        except ValueError as e:
            ev.notes.append("%s is not JSON (%s); not loaded" % (name, e))
            continue
        clean, drop = scrub(obj, blocked)
        ev.artifacts[name] = clean
        if drop:
            ev.dropped[name] = drop
    if isinstance(context, dict) and "claims" in context:
        # The ticker passes the claims register in its context (runner 5.5.4);
        # it is kept as its own artifact, CLAIMS, not inside the context.
        context = dict(context)
        in_ctx = context.pop("claims")
        claims = claims if claims is not None else in_ctx
    for vname, vobj in ((CONTEXT, context), (CLAIMS, claims)):
        if vobj is None:
            continue
        cpy = json.loads(json.dumps(vobj))             # a JSON copy: no live objects
        clean, drop = scrub(cpy, blocked)
        ev.artifacts[vname] = clean
        if drop:
            ev.dropped[vname] = drop
    return ev


def exp_dirs(root):
    """Experiment directories directly under root (those holding exp.json).
    Only root's own entries are listed; nothing is walked."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and re.fullmatch(EXP_RE, p.name) and p.name not in RESERVED_DIRS
                  and (p / "exp.json").is_file())


def load_dir(root, exp, exps=None, context=None, claims=None, domain=None):
    """Evidence from a snapshot directory laid out like INC_DIR (the replay
    fixtures, or a pulled copy): the allow-listed files of `exp`, of every
    other experiment in `exps` (default: every experiment dir under root), of
    step1/ and of funnel/. Nothing else under root is opened."""
    root = Path(root)
    names = []
    others = exp_dirs(root) if exps is None else list(exps)
    for e in [exp] + [x for x in others if x != exp]:
        names.extend("%s/%s" % (e, f) for f in EXP_FILES)
        names.append(ROOT_AUDIT % e)
    names.extend("step1/%s" % f for f in STEP1_FILES)
    names.extend("funnel/%s" % f for f in FUNNEL_FILES)
    texts, touched = {}, []
    for n in names:
        if not allowed(n):
            continue
        p = root / n
        if p.is_file():
            texts[n] = _read_file(p)
            touched.append(n)
    return from_texts(texts, exp, context=context, touched=touched, claims=claims, domain=domain)


# ------------------------------------------------------- remote snapshot
def _snapshot_records(record):
    """[snapshot, step1 or funnel-summary record] inside a remote.py snapshot,
    step1, funnel-summary or campaign-snapshot record."""
    verb = (record or {}).get("verb")
    if verb in ("snapshot", "step1", "funnel-summary"):
        return [record]
    if verb == "campaign-snapshot":
        out = [(s or {}).get("snapshot") or {} for _, s in sorted((record.get("experiments") or {}).items())]
        if record.get("step1"):
            out.append(record["step1"])
        if isinstance(record.get("funnel"), dict) and record["funnel"].get("verb") == "funnel-summary":
            out.append(record["funnel"])
        if isinstance(record.get("stream"), dict) and record["stream"].get("verb") == "stream-summary":
            out.append(record["stream"])
        return out
    raise EvidenceError("not a remote snapshot record (verb %r)" % (verb,))


def _json_sha(text):
    return _sha(text.encode("utf-8") if isinstance(text, str) else text)


def _cluster_file(info, **extra):
    """Provenance of a cluster file as the remote record states it."""
    out = {"sha256": info.get("sha256"), "bytes": info.get("bytes"), "source": "cluster file"}
    out.update(extra)
    return out


def from_snapshot(record, exp, context=None, ledger_prefix=None, claims=None, domain=None):
    """Evidence from remote.py's INCAP record (snapshot, step1 or
    campaign-snapshot), equal to what load_dir gives on the same files:
      * decision.artifacts are the files' (dev-only) contents;
      * report.json comes without its final table; derived.report_final_dev
        (the same rows, dev column only, in the same order) is put back at
        report.json /final, so a pointer into it addresses the file's value;
      * state.json comes without runs; derived.state_runs becomes
        <exp>/derived/state_runs.json;
      * derived.select_clusters_by_source becomes
        step1/select_clusters_by_source.json;
      * a funnel-summary record (remote.py funnel summary, alone or under a
        campaign-snapshot's "funnel") gives its decision.artifacts (the
        funnel aggregates and the Step 1 funnel reads) as they are, and its
        derived.funnel_ledger (the summaries-derived ledger, when no ledger
        file exists on the cluster) as funnel/funnel_ledger.json, marked
        derived;
      * the ledger's entries keep their 1-based line numbers; ledger_prefix
        ({exp: [(line, entry)]}, the lab's earlier copy) supplies lines the
        record did not ship (remote snapshot --ledger-from).
    The record's display_only part (every exam of the final table) is never
    read.

    Provenance: a file the record hashed keeps the cluster file's own sha256
    and size (the record's "files"), so a campaign-ledger entry can be matched
    to /ocean; a file shipped compacted (state.json without runs, report.json
    without its final table) says so ("shipped_as"); an artifact built here
    from derived values is marked "source": "derived" with the sha256 of the
    JSON built ("json_sha256"), never a file hash."""
    texts, ledgers, prov = {}, {}, {}
    for sub in _snapshot_records(record):
        dec = (sub or {}).get("decision") or {}
        files = (sub or {}).get("files") or {}
        omitted = dec.get("omitted") or {}
        derived = dec.get("derived") or {}
        arts = dict(dec.get("artifacts") or {})
        for name, obj in sorted(arts.items()):
            put_back = False
            if name.endswith("/report.json") and isinstance(obj, dict) and "final" not in obj \
                    and isinstance(derived.get("report_final_dev"), list):
                obj = dict(obj, final=derived["report_final_dev"])
                put_back = True
            if name.endswith("/ledger.jsonl") and exp_of(name) is None and isinstance(obj, list):
                # a stream ledger shipped as its list of entries: back to JSON lines
                texts[name] = "".join(json.dumps(x, sort_keys=True) + "\n" for x in obj)
            else:
                texts[name] = json.dumps(obj, sort_keys=True)
            info = files.get(name)
            if name in DERIVED_FROM:
                prov[name] = {"source": "derived", "json_sha256": _json_sha(texts[name]),
                              "derived_from": DERIVED_FROM[name],
                              "from_file": _cluster_file(info) if isinstance(info, dict) and info.get("sha256")
                              else None}
            elif isinstance(info, dict) and info.get("sha256"):
                extra = {}
                if omitted.get(name):
                    extra["shipped_as"] = "compacted: without %s" % ", ".join(omitted[name])
                if name in SHIPPED_AS:
                    extra["shipped_as"] = SHIPPED_AS[name]
                if put_back:
                    extra["final_from"] = "derived.report_final_dev (the dev column)"
                prov[name] = _cluster_file(info, **extra)
            else:
                prov[name] = {"source": "derived", "json_sha256": _json_sha(texts[name]),
                              "note": "the record gives no cluster file hash for it"}
            e = exp_of(name)
            if name.endswith("/state.json") and e and isinstance(derived.get("state_runs"), dict):
                dn = "%s/%s" % (e, DERIVED_STATE_RUNS)
                texts[dn] = json.dumps(derived["state_runs"], sort_keys=True)
                prov[dn] = {"source": "derived", "json_sha256": _json_sha(texts[dn]),
                            "derived_from": name, "from_file": prov.get(name)}
        if isinstance(derived.get("funnel_ledger"), dict) and FUNNEL_LEDGER not in arts:
            texts[FUNNEL_LEDGER] = json.dumps(derived["funnel_ledger"], sort_keys=True)
            prov[FUNNEL_LEDGER] = {"source": "derived", "json_sha256": _json_sha(texts[FUNNEL_LEDGER]),
                                   "derived_from": "adapters.inc_step1.ledger_from_summaries on the cluster's "
                                                   "Step 1 summaries and census_v0.json"}
        if isinstance(derived.get("select_clusters_by_source"), dict):
            dn = "step1/" + CLUSTERS_BY_SOURCE
            texts[dn] = json.dumps(derived["select_clusters_by_source"], sort_keys=True)
            csv = files.get("step1/select_clusters.csv")
            prov[dn] = {"source": "derived", "json_sha256": _json_sha(texts[dn]),
                        "derived_from": "step1/select_clusters.csv",
                        "from_file": _cluster_file(csv) if isinstance(csv, dict) and csv.get("sha256") else None}
        led = dec.get("ledger")
        if isinstance(led, dict) and exp_of(str(led.get("artifact"))):
            e = exp_of(led["artifact"])
            rows = dict((int(ln), x) for ln, x in (ledger_prefix or {}).get(e, []))
            for it in led.get("entries") or []:
                rows[int(it["line"])] = it.get("entry")
            ledgers[e] = sorted(rows.items())
            info = files.get(led["artifact"])
            lp = {"from_line": led.get("from_line"), "through_line": led.get("next_line"),
                  "through_sha256": led.get("through_sha256")}
            prov[led["artifact"]] = (_cluster_file(info, **lp) if isinstance(info, dict) and info.get("sha256")
                                     else dict(lp, source="derived", note="the record gives no file hash"))
    ev = from_texts(texts, exp, context=context, claims=claims, domain=domain)
    for name, pv in prov.items():
        if name in ev.provenance or name.endswith("/ledger.jsonl"):
            ev.provenance[name] = pv
    for e, rows in ledgers.items():
        clean_rows, drop = [], []
        for ln, entry in rows:
            clean, d = scrub(entry, ev.blocked) if entry is not None else (None, [])
            clean_rows.append((ln, clean))
            drop.extend("%d#%s" % (ln, p) for p in d)
        ev.ledgers[e] = clean_rows
        ev.touched.append("%s/ledger.jsonl" % e)
        if drop:
            ev.dropped["%s/ledger.jsonl" % e] = drop
    ev.notes.append("from a remote %s record (%s)" % (record.get("verb"), record.get("utc") or "no utc"))
    return ev
