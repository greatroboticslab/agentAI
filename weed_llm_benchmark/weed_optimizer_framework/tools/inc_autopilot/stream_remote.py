"""The cluster side of stream mode (docs/CONTINUOUS_LOOP.md 6.2): the fixed
verbs remote.py dispatches to for a stream campaign. Run on the Bridges-2
login node from the nested (git-tracked) copy, like every remote.py verb; each
prints exactly one "INCAP <json>" line (remote.emit).

    stream-snapshot --sid SID [--exp E ...] [--advance E ...] [--report auto|always|never]
                    [--ledger-from E=N[:SHA256] ...] [--dev-scores E ...] [--sacct JOBID ...]
                    [--largest]
        ONE call per tick covering every lane: per experiment advance (only
        those named by --advance; with INC_JOB_SCRIPT exported as
        run_inc2_job.sh, never the v1 script), report and snapshot (remote.py's
        own verbs, so the decision/display_only split and the dev-only scrub
        are theirs); then the stream summary (the queue summary, the stream
        ledger with its hash chain checked, the dev scores of the named
        experiments' base runs, step1_stream/status.json, every intake batch
        summary and its D28-v2 sidecar eval_hits.json when there is one, the
        fold of intake/sources.jsonl, the network probe's
        placement, the splits lock status), squeue and the builds' provenance
        (remote status), sacct of the named jobs, the allocation balance and
        end date (`projects`), the /ocean quota, and the sha256 of every stream
        module of this checkout (S23). Aggregates only; test is never read.
    stream-submit KIND [--parent-exp P] [--child-exp E] [--trigger D1,..] [--approval-id ID]
                  [--decided-by A] [--dry-run] -- ARGS ...
        sbatch -p GPU-shared of a stream job script, ARGS checked against the
        script's grammar: collect (run_inc_collect.sh fetch|intake|probe),
        admit (run_inc2_stream.sh admit|bootstrap|knowntruth|backfill|scan-holds|eval-hits),
        build (run_inc2_build.sh <pkg>.stream init|build|milestone|fork|feasibility|bisect,
        <pkg>.splits build|lock, <pkg>.baseline build|rescore-native|rescore-agnostic,
        <pkg>.base3 build, <pkg>.pilot4 build).
        Always GPU-shared: the allocation refuses RM-shared ("Invalid qos"),
        and a qos refusal comes back as error_kind 'qos', a platform defect,
        never retried elsewhere (S21). Refused while a job of the same name is
        queued.
    stream-run <pkg>.<module> VERB [--approval-id ID] [--decided-by A] -- ARGS ...
        The login-node verbs (each deterministic, short, and reading dev
        only): <pkg>.stream commit, compare, choose-arm, rollback,
        quarantine, release; <pkg>.baseline canary-verdict, capacity-verdict;
        <pkg>.pilot4 verdict. A person's approval (--decided-by human:...) is
        passed as INCAP_DECIDED_BY, which inc2.stream takes for --decided-by;
        exit 3 (another writer holds stream.lease) is error_kind 'busy', a
        transient. The call and who authorised it are appended to
        _campaign/provenance/_actions.jsonl.

Test blindness: every shipped artifact goes through remote.dev_only; a score
not stamped dev is refused (remote.funnel_dev_scores).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import model as M
from . import remote as R

FORMAT = "inc_autopilot.stream_remote/1"
V2_JOB_SCRIPT = "run_inc2_job.sh"
PARTITION = "GPU-shared"
SCRIPTS = {"collect": "run_inc_collect.sh", "admit": "run_inc2_stream.sh", "build": "run_inc2_build.sh"}
VERBS = {"collect": ("fetch", "intake", "probe"),
         "admit": ("admit", "bootstrap", "knowntruth", "backfill", "scan-holds", "eval-hits")}
BUILD_VERBS = {"stream": ("init", "build", "milestone", "fork", "feasibility", "bisect"), "splits": ("build", "lock"),
               "baseline": ("build", "rescore-native", "rescore-agnostic"), "pilot4": ("build",), "base3": ("build",)}
RUN_VERBS = {"stream": ("commit", "compare", "choose-arm", "rollback", "quarantine", "release"),
             "baseline": ("canary-verdict", "capacity-verdict"), "pilot4": ("verdict",)}
EXIT_BUSY = 3                                   # inc2.stream: another writer holds stream.lease
# inc2.baseline rescore-native's job (L23N): its own name, never the arm's
# build job's (inc_build_<exp>); the platform follows it by this name when
# its submission's outcome is unknown
NATIVE_JOB_NAME = "inc_build_native_%s"
# E1 (2026-10-03): inc2.baseline rescore-agnostic's job (L23E) and inc2.base3
# build's (L23V), each under a name of its own, followed by it the same way
AGNOSTIC_JOB_NAME = "inc_build_agnostic_%s"
BASE3_JOB_NAME = "inc_build_base3_v3"
PKG_RE = re.compile(r"(?:weed_optimizer_framework\.tools\.)?(?P<pkg>[a-z][a-z0-9_]{0,31})\.(?P<mod>[a-z0-9_]+)\Z")
SOURCE_RE = re.compile(r"(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
BATCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
POOL_RE = re.compile(r"P_[0-9]{1,4}(\.[0-9a-f]{16})?\Z")
HOLDS = ("h6_scan", "funnel_F9", "licence")
QOS_RE = re.compile(r"invalid qos|qos.*(not|invalid)|Invalid qos specification", re.I)
PROJECTS_CMD = "projects"
QUOTA_CMD = "my_quotas"
CHAIN_KEY = "prev_sha256"
MAX_LEDGER_ROWS = 20000
STREAM_MODULE_GLOBS = (("tools/inc2", "*.py"), ("tools/inc2", "*.json"), ("tools/collect", "*.py"),
                       ("tools/collect/providers", "*.py"),
                       ("tools/collect/domains", "*.json"), ("tools/inc_autopilot", "stream*.py"),
                       ("tools/inc_autopilot", "stream*.json"), ("tools/inc_autopilot/stream_domains", "*.json"),
                       ("tools/inc_autopilot", "diagnose_stream.py"), ("tools/inc_autopilot", "levers_stream.py"))


# ------------------------------------------------------------------ helpers
def _pkg_root():
    """The weed_optimizer_framework directory of this checkout."""
    return Path(__file__).resolve().parents[2]


def module_hashes(root=None):
    """{relative path: sha256} of every stream module of a checkout (S23).
    Each listed directory is read one level deep (no rglob)."""
    root = Path(root) if root else _pkg_root()
    out = {}
    for d, pat in STREAM_MODULE_GLOBS:
        p = root / d
        if not p.is_dir():
            continue
        for f in sorted(p.glob(pat)):
            if f.is_file():
                out["%s/%s" % (d, f.name)] = R._sha256_file(f)
    return out


def _dev(obj):
    out, _red = R.dev_only(obj)
    return out


def _read_json(path):
    try:
        obj, info = R.read_small(path)
    except Exception as e:                      # an unreadable file is reported, never fatal
        return None, {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    return obj, info


def _jsonl(path, cap=MAX_LEDGER_ROWS, info=None):
    """(rows, unparsed lines) of the first `cap` lines of a JSON-lines file,
    (None, 0) when it cannot be read. `info`, a dict, gets "truncated": True
    when lines past the cap were left unread."""
    rows, bad = [], 0
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= cap:
                    if info is not None:
                        info["truncated"] = True
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    bad += 1
    except OSError:
        return None, 0
    return rows, bad


def chain_check(path):
    """(ok, n, why) of a hash-chained ledger: every line carries the sha256 of
    the file's bytes before it (contract 3.5, 'Stream ledger')."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        return None, 0, "unreadable (%s)" % type(e).__name__
    pos, n = 0, 0
    for line in raw.splitlines(True):
        if not line.strip():
            pos += len(line)
            continue
        try:
            e = json.loads(line.decode("utf-8"))
        except ValueError:
            return False, n, "line %d is not JSON" % (n + 1)
        want = hashlib.sha256(raw[:pos]).hexdigest()
        got = e.get(CHAIN_KEY) if isinstance(e, dict) else None
        if got is not None and got != want:
            return False, n, "line %d does not chain to the bytes before it" % (n + 1)
        pos += len(line)
        n += 1
    return True, n, ""


def _run(argv, timeout=120, stdin=None, env=None):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=stdin, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", "%s: %s" % (type(e).__name__, str(e)[:200])
    return p.returncode, p.stdout or "", p.stderr or ""


_SIZE_RE = re.compile(r"(?<![\w.])([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]i?B?|B)\b")
_UNITS = {"": 1.0 / 1e9, "B": 1.0 / 1e9, "K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3, "P": 1e6}


def _gb(num, unit):
    u = (unit or "").upper().rstrip("B").rstrip("I") or ""
    return float(num) * _UNITS.get(u, 1.0 / 1e9)


def parse_projects(text):
    """The `projects` output, block by block: [{"resource", "allocation_su",
    "balance_su", "end_date"}]. Tolerant of layout (to verify on the cluster:
    the parser reads 'Resource:', 'Allocation:', 'Balance:' and 'End Date:'
    lines, numbers with thousands separators)."""
    out, cur = [], None
    for line in (text or "").splitlines():
        s = line.strip()
        m = re.match(r"(?i)resource\s*:\s*(.+)$", s)
        if m:
            cur = {"resource": m.group(1).strip(), "allocation_su": None, "balance_su": None, "end_date": None}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"(?i)(allocation|balance)\s*:\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", s)
        if m:
            cur["%s_su" % m.group(1).lower()] = float(m.group(2).replace(",", ""))
            continue
        m = re.match(r"(?i)end\s*date\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", s)
        if m:
            cur["end_date"] = m.group(1)
    return out


def parse_quota(text, needle="/ocean"):
    """(used_gb, quota_gb) from the first line naming `needle` with at least
    two sizes (the quota tool's layout is to verify on the cluster)."""
    for line in (text or "").splitlines():
        if needle not in line:
            continue
        sizes = _SIZE_RE.findall(line.split(needle, 1)[1])
        if len(sizes) >= 2:
            return _gb(*sizes[0]), _gb(*sizes[1])
    return None, None


def projects():
    rc, out, err = _run(shlex.split(os.environ.get("INCAP_PROJECTS", PROJECTS_CMD)), timeout=120)
    if rc != 0:
        return {"ok": False, "error": (err or out)[-300:] or "rc %s" % rc}
    return {"ok": True, "resources": parse_projects(out)}


def quota(largest=False):
    rc, out, err = _run(shlex.split(os.environ.get("INCAP_QUOTA", QUOTA_CMD)), timeout=120)
    rec = {"ok": rc == 0}
    if rc != 0:
        rec["error"] = (err or out)[-300:] or "rc %s" % rc
        return rec
    used, q = parse_quota(out)
    rec.update(used_gb=used, quota_gb=q, free_gb=(None if used is None or q is None else q - used))
    if largest:
        root = R.inc_dir()
        rows = []
        try:
            names = sorted(os.listdir(root))
        except OSError:
            names = []
        for n in names:
            rc2, o2, _e2 = _run(["du", "-sk", str(root / n)], timeout=300)
            if rc2 == 0 and o2.split():
                rows.append({"dir": n, "gb": int(o2.split()[0]) / 1e6})
        rec["largest"] = sorted(rows, key=lambda r: -r["gb"])[:10]
    return rec


REFUSAL_LINE_RE = re.compile(r"(ERROR|FATAL|[Rr]efus|StreamError|CollectError)")
LOG_TAIL_BYTES = 65536


def _job_refusal(jid, name):
    """The last refusal line of a stream data job's log (run_inc_collect.sh,
    run_inc2_stream.sh: INC_DIR/logs or INC_DIR/intake/logs, <name>_<id>.out),
    at most 300 characters, or None. Only the data jobs' logs are read: a
    build's log is never shipped."""
    if not str(name or "").startswith("inc_stream_"):
        return None
    inc = R.inc_dir()
    for d in (inc / "logs", inc / "intake" / "logs"):
        try:
            logs = sorted(d.glob("*_%s.out" % jid))
        except OSError:
            continue
        for p in logs:
            try:
                with open(str(p), "rb") as fh:
                    fh.seek(max(0, p.stat().st_size - LOG_TAIL_BYTES))
                    lines = fh.read().decode("utf-8", "replace").splitlines()
            except OSError:
                continue
            hits = [x.strip() for x in lines if REFUSAL_LINE_RE.search(x)]
            if hits:
                return R._short(hits[-1], 300)
    return None


def sacct(job_ids):
    ids = [str(j) for j in job_ids if re.match(r"^[0-9]+(_[0-9]+)?$", str(j))]
    if not ids:
        return {"ok": True, "jobs": {}}
    cmd = os.environ.get("INCAP_SACCT", "sacct")
    rc, out, err = _run([cmd, "-X", "-P", "-j", ",".join(ids),
                         "--format=JobIDRaw,JobName,State,Elapsed,AllocTRES,NodeList"], timeout=120)
    if rc != 0:
        return {"ok": False, "error": (err or out)[-300:]}
    from ..brain import su_ledger
    jobs = {}
    for j in su_ledger.parse_sacct(out):
        jobs[j["jobid"]] = {k: j.get(k) for k in ("state", "elapsed_s", "gpu_count", "gpu_type")}
        if j.get("state") and not str(j["state"]).startswith(("COMPLETED", "RUNNING", "PENDING")):
            name = ((j.get("raw") or {}).get("JobName") or (j.get("raw") or {}).get("jobname"))
            ref = _job_refusal(j["jobid"], name)
            if ref:
                jobs[j["jobid"]]["refusal"] = ref
    return {"ok": True, "jobs": jobs}


def _advance_v2(exp):
    """remote.advance with INC_JOB_SCRIPT = run_inc2_job.sh (S18): an in-verb
    advance submits the v2 executor; without that script nothing is
    advanced (the v1 script fails closed on the never-train guard, and is
    never submitted for a v2 experiment)."""
    script = R.script_dir() / V2_JOB_SCRIPT
    rec = R.base_record("advance")
    rec["exp"] = exp
    if not script.is_file():
        return R.fail(rec, "%s not found in %s: a stream experiment is never advanced with the v1 executor"
                      % (V2_JOB_SCRIPT, R.script_dir()), error_kind="refused")
    old = os.environ.get("INC_JOB_SCRIPT")
    os.environ["INC_JOB_SCRIPT"] = str(script)
    try:
        out = R.advance(exp)
    finally:
        if old is None:
            os.environ.pop("INC_JOB_SCRIPT", None)
        else:
            os.environ["INC_JOB_SCRIPT"] = old
    if isinstance(out, dict):
        out["job_script"] = str(script)
    return out


# The source events that close a fetch attempt the collector started
# (collect.fetch: an attempt that returns, or raises a refusal, after its
# fetch_started records one of them).
FETCH_CLOSE_EVENTS = ("fetch_failed", "fetched", "held", "closed")


def fetch_facts(rows):
    """{source: {"fetched_events": [[ts, bytes]], "open_fetches": [ts]}} of
    source-ledger rows in time order, or None when a fetched event's bytes
    are not a number. `open_fetches` holds each fetch_started that no closing
    event (FETCH_CLOSE_EVENTS) follows before the source's next fetch_started:
    an attempt still running, or one that ended without recording what it
    fetched (killed by a timeout or the walltime, or an exception the
    collector does not catch, with finished files left in staging). The
    autopilot's byte limits count the fetched events by the time each was
    fetched, and an ended attempt with an open start at its requested
    max_bytes (executor.fetched_bytes)."""
    out, cur = {}, {}
    for r in rows:
        src, ev = str(r["source"]), r.get("event")
        row = out.setdefault(src, {"fetched_events": [], "open_fetches": []})
        if ev == "fetch_started":
            if src in cur:
                row["open_fetches"].append(cur[src])
            cur[src] = r.get("ts")
        elif ev in FETCH_CLOSE_EVENTS:
            cur.pop(src, None)
        if ev == "fetched":
            try:
                row["fetched_events"].append([r.get("ts"), int(r.get("bytes") or 0)])
            except (TypeError, ValueError):
                return None
    for src, ts in cur.items():
        out[src]["open_fetches"].append(ts)
    return out


def _fold_sources(rows, facts=True):
    """{source: folded state} of intake/sources.jsonl: the collector's own fold
    (collect.state.fold: status, attempts, failed_attempts, bytes, su,
    batches, yield, jobs), else the last row per source. Each source also
    carries `fetch_complete` and `fetch_remaining` of its last `fetched` event
    (collect.fetch: a fetch that stopped at a byte cap records complete false
    and the files left, and the next fetch continues them), which the fold
    does not keep; None when no fetched event says. With the collector's
    fold, and `facts` (the whole ledger was read), each source also carries
    `fetched_events` and `open_fetches` (fetch_facts), which the autopilot's
    byte limits count; with any of these missing, no source carries them.
    Each source also carries `pending_names` and `pending_ts`: the class names
    and time of its last `held` event for names_pending at intake (collect.intake: the
    names the offline resolver lacked, lever L26), None once a later event
    of the source supersedes that hold."""
    rows = [r for r in rows if isinstance(r, dict) and r.get("source")]
    rows = sorted(rows, key=lambda r: (str(r.get("ts") or ""), str(r.get("source") or "")))
    last_fetch, pending = {}, {}
    for r in rows:
        if r.get("event") == "fetched" and isinstance(r.get("complete"), bool):
            last_fetch[str(r["source"])] = (r["complete"], r.get("remaining"))
        if r.get("event") == "held" and r.get("reason") == "names_pending" and r.get("stage") == "intake":
            pending[str(r["source"])] = ([str(n) for n in (r.get("names") or [])][:200], r.get("ts"))
        elif r.get("event") in ("held", "closed", "released", "intaken"):
            pending.pop(str(r["source"]), None)
    try:
        from ..collect import state as CS
        out = {str(k): v for k, v in CS.fold(rows).items()}
        ff = fetch_facts(rows) if facts else None
        if ff is not None:
            for src, row in out.items():
                row.update(ff.get(src) or {"fetched_events": [], "open_fetches": []})
    except Exception:  # noqa: BLE001 - no collector here: the last row per source
        out = {}
        for r in rows:
            out[str(r["source"])] = dict(r)
    for src, row in out.items():
        if isinstance(row, dict):
            done, left = last_fetch.get(src, (None, None))
            row["fetch_complete"], row["fetch_remaining"] = done, left
            row["pending_names"], row["pending_ts"] = pending.get(src, (None, None))
    return out


# inc2.baseline / inc2.pilot4 verdicts, inc2.baseline rescore-native's and rescore-agnostic's records (dev only)
RECORD_FILES = ("canary.json", "stage_a.json", "native_rescore.json", "agnostic_rescore.json")


SUMMARY_REFRESH_AGE_S = 24 * 3600
SUMMARY_REFRESH_TIMEOUT_S = 240                 # inside the snapshot's own 600 s (executor timeouts)
STREAM_PKG = "inc2"


def refresh_summary(sid, pkg=STREAM_PKG):
    """Rewrite stream/<sid>/queue_summary.json through inc2.stream's own
    `summary` verb (its writer, under stream.lease) when it is stale, before
    the snapshot reads it. inc2.stream rewrites the summary only in its
    writing verbs, so without this a Step 1 batch (admit, backfill,
    scan-holds) that adds supply while the TRAIN lane waits for Q >= M is
    never seen by D22, and the time-based counts (days since the first
    ACCEPT, holds past their deadline) stop at the last stream write.
    Stale = step1_stream's queue or events file is newer than the summary,
    or the summary is older than SUMMARY_REFRESH_AGE_S. A busy lease (exit 3)
    or a refusal keeps the old file; the record says so. None when nothing
    was due (or the stream does not exist yet)."""
    inc = R.inc_dir()
    sd = inc / "stream" / sid
    summ = sd / "queue_summary.json"
    if not (sd / "ledger.jsonl").is_file():
        return None
    try:
        s_mtime = summ.stat().st_mtime if summ.is_file() else None
    except OSError:
        s_mtime = None
    why = []
    for n in ("queue.jsonl", "events.jsonl"):
        p = inc / "step1_stream" / "queue" / n
        try:
            if p.is_file() and (s_mtime is None or p.stat().st_mtime > s_mtime):
                why.append("step1_stream/queue/%s is newer than the summary" % n)
        except OSError:
            continue
    if s_mtime is not None and time.time() - s_mtime > SUMMARY_REFRESH_AGE_S:
        why.append("the summary is older than %d h" % (SUMMARY_REFRESH_AGE_S // 3600))
    if not why:
        return None
    argv = [sys.executable, "-m", "weed_optimizer_framework.tools.%s.stream" % pkg, "summary", "--stream", str(sid),
            "--quiet"]
    runner = os.environ.get("INCAP_STREAM_RUN")
    rc, _out, err = _run(shlex.split(runner) + argv[1:] if runner else argv, timeout=SUMMARY_REFRESH_TIMEOUT_S)
    return {"why": why, "rc": rc, "refreshed": rc == 0, "busy": rc == EXIT_BUSY,
            "stderr_tail": R._short(err, 300) if rc else None}


def stream_summary(sid, dev_exps=()):
    """{"verb": "stream-summary", "decision": {"artifacts": {...}}, "files": {...}}."""
    rec = {"verb": "stream-summary", "ok": True, "sid": sid}
    inc = R.inc_dir()
    arts, files, notes = {}, {}, []
    try:
        rec["refresh"] = refresh_summary(sid)
    except Exception as e:                      # a refresh never fails the snapshot; the old summary is read
        rec["refresh"] = {"refreshed": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}

    def put(name, path, obj=None, info=None):
        if obj is None:
            if not Path(path).is_file():
                return
            obj, info = _read_json(path)
        if obj is None:
            if isinstance(info, dict) and info.get("error"):
                notes.append("%s: %s" % (name, info["error"]))
            return
        arts[name] = _dev(obj)
        if isinstance(info, dict) and info.get("sha256"):
            files[name] = {"sha256": info.get("sha256"), "bytes": info.get("bytes")}

    sd = inc / "stream" / sid
    put("stream/%s/queue_summary.json" % sid, sd / "queue_summary.json")
    lp = sd / "ledger.jsonl"
    rows, bad = _jsonl(lp)
    if rows is not None:
        ok, n, why = chain_check(lp)
        # The chain is checked here; each line's prev_sha256 hashes bytes that
        # may hold a milestone's test values, so it is not shipped as decision data.
        arts["stream/%s/ledger.jsonl" % sid] = _dev([{k: v for k, v in r.items() if k != CHAIN_KEY}
                                                     if isinstance(r, dict) else r for r in rows])
        files["stream/%s/ledger.jsonl" % sid] = {"sha256": R._sha256_file(lp), "bytes": lp.stat().st_size}
        rec["ledger_chain"] = {"ok": ok, "lines": n, "why": why, "unparsed": bad}
    if dev_exps:
        ds = R.funnel_dev_scores(list(dev_exps))
        if ds.get("ok"):
            arts["stream/%s/dev_scores.json" % sid] = _dev(ds.get("scores") or {})
        else:
            notes.append("dev scores: %s" % ds.get("error"))
    put("step1_stream/status.json", inc / "step1_stream" / "status.json")
    idir = inc / "intake"
    try:
        batches = sorted(p for p in os.listdir(idir) if (idir / p).is_dir())
    except OSError:
        batches = []
    for b in batches:
        if R.NAME_RE.match(b) or BATCH_RE.match(b):
            if (idir / b / "summary.json").is_file():
                put("intake/%s/summary.json" % b, idir / b / "summary.json")
                # D28-v2: the sidecar that weighs again the dHash hits of a batch committed before the
                # amendment (collect.intake.rescore_eval_hits; pair cosines and keys, no pixels)
                put("intake/%s/eval_hits.json" % b, idir / b / "eval_hits.json")
    sinfo = {}
    srows, sbad = _jsonl(idir / "sources.jsonl", info=sinfo)
    if srows is not None:
        whole = not sbad and not sinfo.get("truncated")
        if not whole:
            # the fold misses events: it carries no fetched facts, so the byte limits count the cluster's
            # ended fetches at their requested max_bytes (executor.fetched_bytes)
            notes.append("intake/sources.jsonl: %d unparsed line(s)%s; its fold carries no fetched bytes"
                         % (sbad, ", read stopped at %d lines" % MAX_LEDGER_ROWS if sinfo.get("truncated") else ""))
        fold = _fold_sources(srows, facts=whole)
        if fold or whole:
            arts["intake/sources.json"] = _dev(fold)
    put("intake/placement.json", idir / "placement.json")
    # the R0 verdicts: L-4's capacity decision, the canary, Stage A (dev only;
    # capacity_v1_report.* holds test and is never read here)
    put("capacity/capacity_v1.json", inc / "capacity" / "capacity_v1.json")
    # the measurement arms' native-resolution verdict (inc2.baseline native-verdict, dev only;
    # native_v1_report.* holds a non-decision exam and is never read here)
    put("capacity/native_v1.json", inc / "capacity" / "native_v1.json")
    # E1 (2026-10-03): splits v3's summary (inc2.base3: the arms' manifests by sha256, per-source counts; its
    # held-out set is holdout_v1) and E1's verdict (dev only; e1_v1_report.* is for people, never read here)
    put("splits/v3/summary.json", inc / "splits" / "v3" / "summary.json")
    put("capacity/e1_v1.json", inc / "capacity" / "e1_v1.json")
    try:
        tops = sorted(p.name for p in inc.iterdir() if p.is_dir())   # INC_DIR's top level only, as remote.status
    except OSError:
        tops = []
    for exp in tops:
        if R.NAME_RE.match(exp):
            for f in RECORD_FILES:
                if (inc / exp / f).is_file():
                    put("%s/%s" % (exp, f), inc / exp / f)
    # the splits lock: presence, sha256, and the training manifests it names
    blocked = set(M.non_dev_exams()) | {M.DECISION_EXAM}
    for ver in ("v2",):
        lock = inc / "splits" / ver / "LOCK.json"
        st = {"splits_version": ver, "locked": lock.is_file()}
        if lock.is_file():
            obj, info = _read_json(lock)
            mans = sorted((obj or {}).get("manifests") or {}) if isinstance(obj, dict) else []
            st["train_manifests"] = [m for m in mans if m not in blocked]
            st["h6"] = (obj or {}).get("h6_status") if isinstance(obj, dict) else None
            # the v2 embedding calibration the LOCK binds (decision L-9(c)): D28 compares a source's
            # embedding hits with the count its per-image false-positive rate predicts
            ec = (obj or {}).get("embed_calibration_v2") if isinstance(obj, dict) else None
            if isinstance(ec, dict):
                st["embed_calibration_v2"] = {k: ec.get(k) for k in ("sha256", "cos_threshold", "strict_threshold",
                                                                     "p_false", "testing")}
            # the LOCK's sha256 hashes the evaluation manifests' entries: provenance
            # (files), never decision data
            files["splits/%s/lock_status.json" % ver] = {"sha256": (info or {}).get("sha256"), "of": "LOCK.json"}
        arts["splits/%s/lock_status.json" % ver] = _dev(st)
    rec["decision"] = {"artifacts": arts}
    rec["files"] = files
    rec["notes"] = notes
    return rec


def stream_snapshot(sid, exps=(), advance=(), report_mode="auto", ledger_from=None, dev_exps=(), sacct_ids=(),
                    largest=False):
    rec = R.base_record("stream-snapshot")
    rec["format_stream"] = FORMAT
    ledger_from = ledger_from or {}
    out = {}
    for exp in list(dict.fromkeys(exps)):
        sub = {}
        if exp in advance:
            if not (R.inc_dir() / str(exp) / "exp.json").is_file():
                sub["advance"] = {"verb": "advance", "ok": True, "skipped": "not built"}
            elif R.NAME_RE.match(str(exp)) and R.abandoned(exp) is not None:
                sub["advance"] = {"verb": "advance", "ok": True, "skipped": "abandoned"}
            else:
                sub["advance"] = R._guard("advance", _advance_v2, exp)
        try:
            due = R._report_due(exp, report_mode)
        except (OSError, ValueError):
            due = False
        if due:
            sub["report"] = R._guard("report", R.report, exp)
        lf, lsha = R._ledger_pos(ledger_from.get(exp, 0))
        sub["snapshot"] = R._guard("snapshot", R.snapshot, exp, ledger_from=lf, step1=False, ledger_sha256=lsha)
        out[exp] = sub
    rec["experiments"] = out
    rec["stream"] = R._guard("stream-summary", stream_summary, sid, dev_exps)
    rec["status"] = R._guard("status", R.status)
    rec["sacct"] = sacct(sacct_ids)
    rec["projects"] = projects()
    rec["quota"] = quota(largest)
    rec["code"] = {"modules": module_hashes()}
    subs = [r for s in out.values() for r in s.values() if isinstance(r, dict)] + [rec["stream"], rec["status"]]
    rec["ok"] = all(r.get("ok", False) for r in subs)
    return rec


# ------------------------------------------------------------------ submit
def _flags(args, spec):
    """{param: value} of ARGS against {flag: (param, regex)}; raises R.Refused."""
    out, i = {}, 0
    while i < len(args):
        tok = args[i]
        if tok not in spec:
            raise R.Refused("%r is not a flag here (%s)" % (tok, sorted(spec)))
        param, rx = spec[tok]
        if param in out:
            raise R.Refused("%s is given twice" % tok)
        if i + 1 >= len(args):
            raise R.Refused("%s needs a value" % tok)
        val = args[i + 1]
        if R.BAD_TOKEN_RE.search(val) or not rx.match(val):
            raise R.Refused("%s %r is not admitted" % (tok, val))
        out[param] = val
        i += 2
    return out


_INT = re.compile(r"[1-9][0-9]{0,11}\Z")
_NAME = R.NAME_RE
_RECIPES = re.compile(r"[a-z0-9]{1,8}(,[a-z0-9]{1,8}){0,2}\Z")
_INC_MANIFEST = r"/[A-Za-z0-9_./-]{1,400}\.jsonl"
_SEG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,58}_s[0-9]{3}\Z")
_DECIDED = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@/+-]{0,127}\Z")
_ENUM = lambda *vals: re.compile("(%s)\\Z" % "|".join(re.escape(v) for v in vals))   # noqa: E731
BUILD_FLAGS = {
    ("stream", "init"): {"--stream": ("stream", _NAME), "--stage-b": ("stage_b", _RECIPES)},
    ("stream", "build"): {"--stream": ("stream", _NAME), "--k": ("k", re.compile(r"[1-9]\Z")),
                          "--exp": ("exp", _SEG), "--recipes": ("recipes", _RECIPES)},
    ("stream", "milestone"): {"--stream": ("stream", _NAME)},
    ("stream", "fork"): {"--stream": ("stream", _NAME), "--m": ("m", _INT)},
    ("stream", "feasibility"): {"--stream": ("stream", _NAME), "--holdout": ("holdout", _NAME), "--m": ("m", _INT)},
    ("stream", "bisect"): {"--stream": ("stream", _NAME), "--from": ("from_pool", POOL_RE)},
    ("splits", "build"): {}, ("splits", "lock"): {},
    ("baseline", "build"): {"--exp": ("exp", _NAME), "--manifest": ("manifest", re.compile(_INC_MANIFEST + r"\Z")),
                            "--union": ("union", re.compile(_INC_MANIFEST + r"(," + _INC_MANIFEST + r"){1,5}\Z")),
                            "--seeds": ("seeds", re.compile(r"[0-9]{1,2}(,[0-9]{1,2}){0,9}\Z")),
                            "--arm": ("arm", _ENUM("n640", "s640", "m640", "m832", "s1024", "l640", "y26m640",
                                                   "y26l640")),
                            "--role": ("role", _ENUM("b_v2", "capacity", "canary", "union", "baseline"))},
    ("baseline", "rescore-native"): {"--exp": ("exp", _NAME), "--reference": ("reference", _NAME)},
    ("baseline", "rescore-agnostic"): {"--exp": ("exp", _NAME), "--reference": ("reference", _NAME)},
    ("base3", "build"): {"--stream": ("stream", _NAME)},
    ("pilot4", "build"): {"--exp": ("exp", _NAME), "--from": ("from_exp", _NAME), "--recipes": ("recipes", _RECIPES)},
}
_CAND = re.compile(r"(?!.*\.\.)/[A-Za-z0-9_./-]{1,400}/intake/candidates_sync/[A-Za-z0-9_.-]{1,160}\.json\Z")
COLLECT_FLAGS = {"fetch": {"--source": ("source", SOURCE_RE), "--max-bytes": ("max_bytes", _INT),
                           "--candidates": ("candidates", _CAND)},
                 "intake": {"--source": ("source", SOURCE_RE)}, "probe": {}}
ADMIT_FLAGS = {"admit": {"--intake": ("intake", BATCH_RE)}, "bootstrap": {}, "knowntruth": {}, "backfill": {},
               "scan-holds": {"--hold": ("hold", _ENUM(*HOLDS))}, "eval-hits": {}}
REQUIRED = {("collect", "fetch"): ("source", "max_bytes"), ("collect", "intake"): ("source",),
            ("admit", "admit"): ("intake",), ("admit", "scan-holds"): ("hold",),
            ("stream", "init"): ("stream", "stage_b"),
            ("stream", "build"): ("stream", "k"), ("stream", "milestone"): ("stream",),
            ("stream", "fork"): ("stream", "m"), ("stream", "feasibility"): ("stream", "holdout", "m"),
            ("stream", "bisect"): ("stream", "from_pool"), ("baseline", "build"): ("exp", "seeds", "arm", "role"),
            ("baseline", "rescore-native"): ("exp", "reference"),
            ("baseline", "rescore-agnostic"): ("exp", "reference"), ("base3", "build"): ("stream",),
            ("pilot4", "build"): ("exp", "from_exp", "recipes")}


def parse_submit(kind, args):
    """{"kind", "script", "verb", "module", "pkg", "params", "script_args"} or R.Refused."""
    args = [str(a) for a in args]
    if kind in ("collect", "admit"):
        if not args or args[0] not in VERBS[kind]:
            raise R.Refused("%s takes a verb among %s" % (kind, VERBS[kind]))
        verb = args[0]
        spec = (COLLECT_FLAGS if kind == "collect" else ADMIT_FLAGS)[verb]
        params = _flags(args[1:], spec)
        missing = [p for p in REQUIRED.get((kind, verb), ()) if p not in params]
        if missing:
            raise R.Refused("%s %s needs %s" % (kind, verb, missing))
        return {"kind": kind, "script": SCRIPTS[kind], "verb": verb, "module": None, "pkg": None,
                "params": params, "script_args": args}
    if kind == "build":
        if len(args) < 2:
            raise R.Refused("build takes <pkg>.<module> <verb> FLAGS")
        m = PKG_RE.match(args[0])
        if not m:
            raise R.Refused("%r is not <pkg>.<module>" % args[0])
        mod, verb = m.group("mod"), args[1]
        if verb not in BUILD_VERBS.get(mod, ()):
            raise R.Refused("%s %s is not a stream build (%s)" % (mod, verb, BUILD_VERBS))
        params = _flags(args[2:], BUILD_FLAGS[(mod, verb)])
        missing = [p for p in REQUIRED.get((mod, verb), ()) if p not in params]
        if missing:
            raise R.Refused("%s %s needs %s" % (mod, verb, missing))
        if (mod, verb) == ("baseline", "build") and ("manifest" in params) == ("union" in params):
            raise R.Refused("baseline build takes exactly one of --manifest and --union")
        short = "%s.%s" % (m.group("pkg"), mod)
        return {"kind": kind, "script": SCRIPTS[kind], "verb": verb, "module": mod, "pkg": m.group("pkg"),
                "params": params, "script_args": [short] + args[1:]}
    raise R.Refused("stream-submit kind %r is not one of %s" % (kind, sorted(SCRIPTS)))


def job_name(req, meta):
    p = req["params"]
    if req["kind"] == "build":
        if (req["module"], req["verb"]) == ("baseline", "rescore-native"):
            # it builds nothing: its own name, never the arm's build job's (inc_build_<exp>)
            return NATIVE_JOB_NAME % p["exp"]
        if (req["module"], req["verb"]) == ("baseline", "rescore-agnostic"):
            return AGNOSTIC_JOB_NAME % p["exp"]
        if (req["module"], req["verb"]) == ("base3", "build"):
            return BASE3_JOB_NAME
        tag = meta.get("child_exp") or (p.get("exp") if req["module"] in ("baseline", "pilot4") else None) \
            or "%s_%s_%s" % (req["module"], req["verb"], p.get("stream") or req["pkg"])
        return "inc_build_%s" % tag
    tag = p.get("source") or p.get("intake") or p.get("hold") or req["verb"]
    return "inc_stream_%s_%s_%s" % (req["kind"], req["verb"].replace("-", ""), re.sub(r"[^A-Za-z0-9_.-]", "_", tag))


def stream_submit(kind, args, meta=None, dry_run=False):
    rec = R.base_record("stream-submit")
    rec.update(kind=kind, dry_run=bool(dry_run))
    meta = dict(meta or {})
    try:
        child = meta.pop("child_exp", None)
        if child is not None and not R.NAME_RE.match(child):
            raise R.Refused("--child-exp %r is not an experiment name" % child)
        prov = R._meta(meta)
        if child:
            prov["child_exp"] = child
        req = parse_submit(kind, args)
        rec.update(script=req["script"], verb=req["verb"], params=req["params"], provenance=prov)
        script = R.script_dir() / req["script"]
        if not script.is_file():
            raise R.Refused("job script %s not found" % script)
        name = job_name(req, {"child_exp": child})
        sbatch = os.environ.get("INCAP_SBATCH", "sbatch")
        argv = [sbatch, "--parsable", "--job-name=%s" % name, "-p", PARTITION, str(script)] + req["script_args"]
        rec.update(job_name=name, sbatch_argv=argv, partition=PARTITION)
        if req["kind"] == "build" and child and (R.inc_dir() / child / "exp.json").exists():
            raise R.Refused("experiment %s is already built; an experiment is built once" % child)
        if dry_run:
            return rec
        q = R.squeue_jobs()
        if not q["ok"]:
            raise R.Refused("squeue unavailable (%s): cannot rule out a duplicate job, nothing submitted" % q["error"])
        dup = [j for j in q["jobs"] if j["name"] == name]
        if dup:
            raise R.Refused("%s is already queued or running as job %s" % (name, dup[0]["id"]))
    except R.Refused as e:
        return R.fail(rec, e, error_kind="refused")
    (R.inc_dir() / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for k, v in prov.items():
        env["INCAP_" + k.upper()] = ",".join(v) if isinstance(v, list) else str(v)
    env["INCAP_REQUESTED_UTC"] = M.utc_now()
    try:
        pr = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return R.fail(rec, "sbatch could not be run: %s: %s" % (type(e).__name__, str(e)[:200]),
                      type(e).__name__, "submit")
    rec.update(sbatch_rc=pr.returncode, sbatch_stderr=R._short(pr.stderr, 400))
    if pr.returncode != 0:
        msg = pr.stderr or pr.stdout
        kind_ = "qos" if QOS_RE.search(msg or "") else "submit"
        return R.fail(rec, "sbatch exited %d: %s" % (pr.returncode, R._short(msg, 300)), "SubmitFailed", kind_)
    jid = ((pr.stdout or "").strip().splitlines() or [""])[-1].split(";")[0].strip()
    if not re.match(r"^[0-9]+$", jid):
        return R.fail(rec, "sbatch printed no job id: %r" % (pr.stdout or "")[-200:], "SubmitFailed", "submit")
    rec["job_id"] = jid
    return rec


RUN_FLAGS = {("stream", "commit"): {"--exp": ("exp", _SEG)},
             ("stream", "compare"): {"--exp": ("exp", re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,58}_[mcb][0-9]{3}\Z"))},
             ("stream", "choose-arm"): {"--stream": ("stream", _NAME)},
             ("stream", "rollback"): {"--stream": ("stream", _NAME), "--to": ("to", POOL_RE)},
             ("stream", "quarantine"): {"--source": ("source", SOURCE_RE), "--cite": ("cite", _ENUM("D28", "D31"))},
             ("stream", "release"): {"--stream": ("stream", _NAME), "--hold": ("hold", _ENUM(*HOLDS))},
             ("baseline", "canary-verdict"): {"--exp": ("exp", _NAME)},
             ("baseline", "capacity-verdict"): {},
             ("pilot4", "verdict"): {"--exp": ("exp", _NAME)}}
RUN_REQUIRED = {("stream", "commit"): ("exp",), ("stream", "compare"): ("exp",), ("stream", "choose-arm"): ("stream",),
                ("stream", "rollback"): ("stream", "to"), ("stream", "quarantine"): ("source", "cite"),
                ("stream", "release"): ("stream", "hold"), ("pilot4", "verdict"): ("exp",)}


def stream_run(pkg_module, verb, args, meta=None, timeout=600):
    """<pkg>.<module> VERB FLAGS on the login node (the verbs of RUN_VERBS)."""
    rec = R.base_record("stream-run")
    rec.update(module=pkg_module, run_verb=verb)
    meta = dict(meta or {})
    try:
        m = PKG_RE.match(str(pkg_module))
        if not m or m.group("mod") not in RUN_VERBS:
            raise R.Refused("%r is not <pkg>.<module> for a module of %s" % (pkg_module, sorted(RUN_VERBS)))
        mod = m.group("mod")
        if verb not in RUN_VERBS[mod]:
            raise R.Refused("stream-run verb %r is not one of %s for %s" % (verb, RUN_VERBS[mod], mod))
        params = _flags(list(args), RUN_FLAGS[(mod, verb)])
        need = [p for p in RUN_REQUIRED.get((mod, verb), ()) if p not in params]
        if need:
            raise R.Refused("%s needs %s" % (verb, need))
        who = meta.get("decided_by")
        if who is not None and not _DECIDED.match(str(who)):
            raise R.Refused("--decided-by %r cannot be recorded" % (who,))
        prov = R._meta(meta)
    except R.Refused as e:
        return R.fail(rec, e, error_kind="refused")
    argv = [sys.executable, "-m", "weed_optimizer_framework.tools.%s.%s" % (m.group("pkg"), mod), verb] + list(args)
    rec.update(argv=argv, params=params, provenance=prov)
    runner = os.environ.get("INCAP_STREAM_RUN")
    env = None
    if who:
        # a person's approval reaches inc2.stream as INCAP_DECIDED_BY (it takes
        # only a human:... value); an envelope approval is the platform's own
        env = dict(os.environ, INCAP_DECIDED_BY=str(who))
    rc, out, err = _run(shlex.split(runner) + argv[1:] if runner else argv, timeout=timeout, env=env)
    rec.update(rc=rc, stdout_tail=R._short(out, 1500), stderr_tail=R._short(err, 800))
    if rc == EXIT_BUSY and mod == "stream":
        R.fail(rec, "%s %s: another writer holds stream.lease (exit 3): %s" % (pkg_module, verb, R._short(err or out, 300)),
               "LeaseBusy", "busy")
    elif rc != 0:
        R.fail(rec, "%s %s exited %s: %s" % (pkg_module, verb, rc, R._short(err or out, 300)), "RunFailed", "run")
    else:
        for line in reversed(out.splitlines()):
            s_ = line.strip()
            if s_.startswith("{"):
                try:
                    rec["result"] = json.loads(s_)
                    break
                except ValueError:
                    continue
    R.log_action(rec, prov, {"verb": verb, "params": params})
    return rec


# ------------------------------------------------------------------ dispatch
def _split(argv, keys):
    """(meta, dry_run, args after '--')."""
    if "--" not in argv:
        raise R._ArgError("options -- ARGS")
    k = argv.index("--")
    head, args = argv[:k], argv[k + 1:]
    meta, dry, i = {}, False, 0
    while i < len(head):
        tok = head[i]
        if tok == "--dry-run":
            dry = True
            i += 1
            continue
        if tok not in keys or i + 1 >= len(head):
            raise R._ArgError("option %r is not one of %s" % (tok, sorted(keys)))
        meta[keys[tok]] = head[i + 1]
        i += 2
    return meta, dry, args


def dispatch(verb, rest):
    keys = {"--parent-exp": "parent_exp", "--trigger": "trigger", "--approval-id": "approval_id",
            "--decided-by": "decided_by", "--child-exp": "child_exp"}
    if verb == "stream-submit":
        if not rest:
            raise R._ArgError("stream-submit KIND [options] -- ARGS")
        meta, dry, args = _split(rest[1:], keys)
        return stream_submit(rest[0], args, meta, dry)
    if verb == "stream-run":
        if len(rest) < 2:
            raise R._ArgError("stream-run <pkg>.stream VERB [options] -- FLAGS")
        meta, _dry, args = _split(rest[2:], {k: v for k, v in keys.items() if k in ("--approval-id", "--decided-by")})
        return stream_run(rest[0], rest[1], args, meta)
    if verb == "stream-snapshot":
        ap = R._Parser(prog="inc_autopilot.remote stream-snapshot")
        ap.add_argument("--sid", required=True)
        ap.add_argument("--exp", action="append", default=[])
        ap.add_argument("--advance", action="append", default=[])
        ap.add_argument("--report", choices=("auto", "always", "never"), default="auto")
        ap.add_argument("--ledger-from", action="append", default=[])
        ap.add_argument("--dev-scores", action="append", default=[])
        ap.add_argument("--sacct", action="append", default=[])
        ap.add_argument("--largest", action="store_true")
        a = ap.parse_args(rest)
        for e in a.exp + a.advance + a.dev_scores + [a.sid]:
            if not R.NAME_RE.match(str(e)):
                raise R._ArgError("%r is not an experiment or stream name" % e)
        bad = [x for x in a.advance if x not in a.exp]
        if bad:
            raise R._ArgError("--advance %s: every advanced experiment is also snapshotted (--exp)" % bad)
        return stream_snapshot(a.sid, a.exp, a.advance, a.report, R._ledger_map(a.ledger_from), a.dev_scores,
                               a.sacct, a.largest)
    raise R._ArgError("unknown stream verb %r" % verb)
