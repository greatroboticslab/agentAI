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
        script's grammar (with --export naming the INCAP_* provenance variables
        it passes: Bridges-2's SBATCH_EXPORT=NONE dropped them; deviation 3,
        2026-10-12): collect (run_inc_collect.sh fetch|intake|probe),
        admit (run_inc2_stream.sh admit|bootstrap|knowntruth|backfill|scan-holds|eval-hits),
        build (run_inc2_build.sh <pkg>.stream init|build|milestone|fork|feasibility|bisect,
        <pkg>.splits build|lock, <pkg>.baseline build|rescore-native|rescore-agnostic|rescore-e2|
        rescore-e2-attr, <pkg>.twostage score-arm|verdict,
        <pkg>.base3 build, <pkg>.pilot4 build; V3F, 2026-10-11: <pkg>.v3fam_data build,
        <pkg>.v3fam score --arm X, <pkg>.v3fam_verdict verdict (deviation 5, 2026-10-24: the
        verdict's CLI outside inc2/v3fam.py), <pkg>.baseline build --exp E --v3fam ARM
        --seeds 0,1,2 --role baseline; a person's score-test, reanalyse-test and the
        D0-S sweep are refused; verdict takes --cap-refused ITEM,... the ticker refused at
        the family cap and --ticker-failed ITEM,... it recorded failed twice); detr
        (run_inc2_detr.sh <pkg>.v3fam_detr smoke | train --seeds S: D6's RF-DETR jobs, L23K; train submits one job per seed named,
        inc_detr_v3fam_s<k> (0,1,2 first; a retry names only the seeds not yet
        trained), and returns their ids; refused while any is queued);
        zoo (run_inc2_zoo.sh submit: the model-zoo
        audit, L23Z, 2026-10-04: the script itself runs on the login node and submits its
        chain of five jobs; the last INCZOO line carries their ids; refused while any of the
        five names is queued; a timeout is an unknown outcome, error_kind 'uncertain').
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

import collections
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
SCRIPTS = {"collect": "run_inc_collect.sh", "admit": "run_inc2_stream.sh", "build": "run_inc2_build.sh",
           "zoo": "run_inc2_zoo.sh", "detr": "run_inc2_detr.sh", "long": "run_inc2_long.sh"}
VERBS = {"collect": ("fetch", "intake", "probe"),
         "admit": ("admit", "bootstrap", "knowntruth", "backfill", "scan-holds", "eval-hits"), "zoo": ("submit",)}
BUILD_VERBS = {"stream": ("init", "build", "milestone", "fork", "feasibility", "bisect"), "splits": ("build", "lock"),
               "baseline": ("build", "rescore-native", "rescore-agnostic", "rescore-e2", "rescore-e2-attr"),
               "pilot4": ("build",),
               "base3": ("build",),
               # E3 (2026-10-05): one arm's scores (L23F) and the verdict (L23G); its score-test is a person's job
               "twostage": ("score-arm", "verdict"),
               # V3F (2026-10-11): the family's data (L23H), one arm's scores (L23J) and its verdict (L23L); its
               # builds are inc2.baseline build --v3fam (L23I); score-test, reanalyse-test and the sweep are a
               # person's or the delegate's jobs, never the platform's. The verdict runs from inc2.v3fam_verdict
               # (deviation 5, 2026-10-24): inc2/v3fam.py, whose sha256 every reproduction record pins, keeps only
               # the verbs it had when they were taken
               "v3fam_data": ("build",), "v3fam": ("score",), "v3fam_verdict": ("verdict",),
               # DS1 (2026-10-29): its data steps (L30), one arm's scores (L33) and a phase verdict (L34); its 1.2 M
               # builds are inc2.baseline build --ds1 (L32); score-test, test-read and record are a person's or the
               # delegate's
               "ds1_data": ("build",), "ds1": ("score", "verdict")}
# V3F (2026-10-11): D6's RF-DETR jobs (L23K, TRAIN lane) through run_inc2_detr.sh
DETR_VERBS = {"v3fam_detr": ("smoke", "train")}
# DS1 (2026-10-29): the long path (L32X, TRAIN lane) through run_inc2_long.sh
LONG_VERBS = {"ds1_long": ("smoke", "train")}
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
# E2 (2026-10-04): inc2.baseline rescore-e2's job (L23C: E2's six final runs on dev at 640, then E2's verdict),
# one name for the one verdict, followed by it the same way
E2_JOB_NAME = "inc_build_e2_v1"
# E2-C (2026-10-04, later): inc2.baseline rescore-e2-attr's job (L23D: E2-C's final runs on dev at 640, then the
# attribution record), one name for the one record, followed by it the same way
E2_ATTR_JOB_NAME = "inc_build_e2_attr_v1"
# E3 (2026-10-05): inc2.twostage score-arm's job per arm (L23F; the arm lower-cased, as run_inc2_build.sh names its
# provenance e3_score_<arm>) and inc2.twostage verdict's (L23G), followed by them the same way
E3_SCORE_JOB_NAME = "inc_build_e3_score_%s"
E3_JOB_NAME = "inc_build_e3_v1"
# V3F (2026-10-11): inc2.v3fam_data build's job (L23H), inc2.v3fam score's per arm (L23J; the arm lower-cased, as
# run_inc2_build.sh names its provenance v3fam_score_<arm>) and inc2.v3fam_verdict verdict's (L23L, provenance
# v3fam_v1; deviation 5); D6's RF-DETR jobs
# (L23K) as run_inc2_detr.sh is submitted: the smoke's and each seed's; a family build (L23I) is inc_build_<exp>
V3FAM_DATA_JOB_NAME = "inc_build_v3fam_data"
V3FAM_SCORE_JOB_NAME = "inc_build_v3fam_score_%s"
V3FAM_JOB_NAME = "inc_build_v3fam_v1"
# L23K's train names the seeds it submits: 0,1,2 first, and on a retry only those not yet trained (a seed already
# trained refuses a second training, so resubmitting it would fail the retry whatever the others do)
V3FAM_DETR_SEEDS_RE = re.compile(r"(0,1,2|0,1|0,2|1,2|0|1|2)\Z")
# L23L's --cap-refused: the family items (stream.py _v3fam_key) the ticker recorded "not run (family cap)"
V3FAM_CAP_ITEMS_RE = re.compile(r"(data|score:[A-Z0-9]{2,8}|build:[A-Z0-9]{2,8}|detr:(smoke|train))"
                                r"(,(data|score:[A-Z0-9]{2,8}|build:[A-Z0-9]{2,8}|detr:(smoke|train))){0,39}\Z")
# L23L's --ticker-failed (deviation 4, 2026-10-20): the score, build and D6 items the ticker recorded failed after
# their own second failure (never the data build or the verdict)
V3FAM_FAILED_ITEMS_RE = re.compile(r"(score:[A-Z0-9]{2,8}|build:[A-Z0-9]{2,8}|detr:(smoke|train))"
                                   r"(,(score:[A-Z0-9]{2,8}|build:[A-Z0-9]{2,8}|detr:(smoke|train))){0,39}\Z")
V3FAM_DETR_JOB_NAMES = {"smoke": ("inc_detr_v3fam_smoke",),
                        "train": ("inc_detr_v3fam_s0", "inc_detr_v3fam_s1", "inc_detr_v3fam_s2")}
# DS1 (2026-10-29): its jobs' names (run_inc2_build.sh names their provenance ds1_data_p<P>_<step>, ds1_score_<arm>,
# ds1_v<P>; run_inc2_long.sh ds1_long_<arm>_smoke and ds1_long_<arm>_s<k>), a build (L32) inc_build_<exp>
DS1_DATA_JOB_NAME = "inc_build_ds1_data_p%d_%s"
DS1_SCORE_JOB_NAME = "inc_build_ds1_score_%s"
DS1_VERDICT_JOB_NAME = "inc_build_ds1_v%d"
DS1_LONG_JOB_NAME = "inc_long_ds1_%s_%s"
DS1_STEPS = ("screen", "md_s0", "pools", "draw", "manifests")
DS1_BUILD_ARMS = ("KD", "KT", "KF", "S0", "S1CR", "S1C", "S1U", "S2")
DS1_LONG_ARMS = ("S0X", "S2X", "S1UX", "S1C", "S1U", "S2", "S1CR", "S0", "KD", "KT", "KF")
DS1_SCORE_ARMS = ("POWER", "K0", "KD", "KT", "KF", "S0", "S1CR", "S1C", "S1U", "S2", "S0X", "S2X", "S1UX")
DS1_ITEM = r"(data:[12]:(screen|md_s0|pools|draw|manifests)|build:[A-Z0-9]{2,5}|long:[A-Z0-9]{2,5}:(smoke|train)" \
           r"|score:[A-Z0-9]{2,5})"
DS1_ITEMS_RE = re.compile(r"%s(,%s){0,39}\Z" % (DS1_ITEM, DS1_ITEM))
DS1_SEEDS_RE = V3FAM_DETR_SEEDS_RE
V3FAM_SCORE_ARMS = ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640",
                    "D2FT", "D3", "D9", "D1", "D4F", "D4N", "D4R15", "D5B", "D6", "D8", "BCLEAN")
V3FAM_BUILD_ARMS = ("D1A1", "D1B1", "D1A12", "D4F", "D4N", "D4R15", "D5B0", "D5B25", "D5B100", "D8S", "D8M",
                    "D0BCLEAN")
# The model-zoo audit (L23Z, 2026-10-04): its chain's five job names (inc2.zoo.ZOO_JOB_NAMES restates them; a
# test checks they agree); the platform follows the chain by its five job ids, or by these names after an unknown
# outcome
ZOO_JOB_NAMES = ("inc_zoo_%s_inventory", "inc_zoo_%s_score_a", "inc_zoo_%s_select", "inc_zoo_%s_score_c",
                 "inc_zoo_%s_report")
ZOO_SUBMIT_TIMEOUT_S = 240
PKG_RE = re.compile(r"(?:weed_optimizer_framework\.tools\.)?(?P<pkg>[a-z][a-z0-9_]{0,31})\.(?P<mod>[a-z0-9_]+)\Z")
SOURCE_RE = re.compile(r"(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
BATCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
POOL_RE = re.compile(r"P_[0-9]{1,4}(\.[0-9a-f]{16})?\Z")
HOLDS = ("h6_scan", "funnel_F9", "licence")
QOS_RE = re.compile(r"invalid qos|qos.*(not|invalid)|Invalid qos specification", re.I)
PROJECTS_CMD = "projects"
QUOTA_CMD = "my_quotas"
# The submission's provenance as stream_submit passes it to a job (INCAP_<field> of --parent-exp, --child-exp,
# --trigger, --approval-id, --decided-by, and INCAP_REQUESTED_UTC); run_inc2_build.sh and run_inc2_detr.sh record
# them in the job's provenance. Deviation 3 (2026-10-12): Bridges-2 exports SBATCH_EXPORT=NONE to every login shell
# (/etc/profile.d/slurm.sh), so sbatch gave a job none of the submitter's variables and every stream job's
# provenance recorded them empty (decided_by, trigger and requested_utc of the 40 latest, D0-S's sweep among them).
# sbatch_export names exactly these variables on the command line, which overrides SBATCH_EXPORT; the values stay in
# the environment sbatch runs with, never in the argv. Measured on Bridges-2 (Slurm 22.05.11, two probe jobs,
# 2026-10-12): a job submitted so holds the variables a NONE job holds (its SLURM_* and the login environment Slurm
# loads on the node; 115, PATH identical) plus exactly the ones named, and no other variable of the submitter's.
PROVENANCE_ENV = ("INCAP_PARENT_EXP", "INCAP_CHILD_EXP", "INCAP_TRIGGER", "INCAP_APPROVAL_ID", "INCAP_DECIDED_BY",
                  "INCAP_REQUESTED_UTC")
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


_SIZE_RE = re.compile(r"(?<![\w.])([0-9]+(?:\.[0-9]+)?)\s*([KMGTP][iI]?B?|B)\b")
_POWER = {"": 0, "B": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}


def _gb(num, unit):
    """A size in GB (10^9 bytes): binary units (KiB, MiB, GiB, TiB, PiB) are powers of 1024, decimal ones (K, M,
    G, T, P, with or without a B) powers of 1000; B or no unit is bytes. Deviation 1 (2026-10-12): every binary
    unit used to count as decimal, 6.9 % low for a GiB and 9.1 % low for a TiB (parse_quota is the only caller)."""
    u = (unit or "").strip()
    base = 1024.0 if len(u) >= 2 and u[1] in "iI" else 1000.0
    return float(num) * base ** _POWER.get(u[:1].upper(), 0) / 1e9


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


_QUOTA_HEAD_RE = re.compile(r"\s*the quota for\b", re.I)
_QUOTA_SIZE_RE = re.compile(r"\s*storage\s+(quota|used)\s*:\s*(.*)\Z", re.I)


def parse_quota(text, needle="/ocean"):
    """(used_gb, quota_gb) of the filesystem `needle` names, in GB (10^9 bytes; _gb), or (None, None) when the
    quota tool's output does not give both. Two layouts:

    * Bridges-2's `my_quotas` (read on the login node 2026-10-12; deviation 1): one block per directory, a header
      "The quota for Project directory /ocean/projects/cis240145p" followed by "Storage quota: 6.84TiB" and
      "Storage used: 6.56TiB" (and inode lines), up to the next header. The first block whose header names
      `needle` is read; a block without both sizes gives (None, None).
    * A table: the first line naming `needle` that holds two sizes after it, the used size first
      ("/ocean  cis240145p  6.44T  7.00T")."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if needle not in line or not _QUOTA_HEAD_RE.match(line):
            continue
        got = {}
        for ln in lines[i + 1:]:
            if _QUOTA_HEAD_RE.match(ln):
                break
            m = _QUOTA_SIZE_RE.match(ln)
            sizes = _SIZE_RE.findall(m.group(2)) if m else []
            if sizes and m.group(1).lower() not in got:
                got[m.group(1).lower()] = _gb(*sizes[0])
        if "used" in got and "quota" in got:
            return got["used"], got["quota"]
        return None, None
    for line in lines:
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


ARRAY_LIVE = ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED")


def fold_arrays(jobs, ids):
    """Each requested id whose sacct rows are array tasks (<id>_<n> or a
    pending <id>_[...] row) folded into one entry: RUNNING while a task is
    live (PENDING when none runs), else COMPLETED when every task completed,
    else the first other state by task index; elapsed_s summed (each task
    holds one GPU), the GPU of the first task, array_tasks {n, states, failed}.
    A requested id with its own row and no task rows is left as it is."""
    out = dict(jobs)
    for b in ids:
        rx = re.compile(r"^%s_(\d+|\[.*\])$" % re.escape(str(b)))
        tasks = sorted(((k, v) for k, v in jobs.items() if rx.match(k)),
                       key=lambda kv: (0, int(kv[0].split("_", 1)[1])) if kv[0].split("_", 1)[1].isdigit()
                       else (1, kv[0]))
        if not tasks:
            continue
        states = [str(v.get("state") or "") for _k, v in tasks]
        live = [s for s in states if s.startswith(ARRAY_LIVE)]
        if live:
            st = "RUNNING" if any(s.startswith(("RUNNING", "COMPLETING", "CONFIGURING")) for s in states) \
                else "PENDING"
        elif all(s.startswith("COMPLETED") for s in states):
            st = "COMPLETED"
        else:
            st = next(s for s in states if not s.startswith("COMPLETED"))
        failed = [int(k.split("_", 1)[1]) for (k, v), s in zip(tasks, states)
                  if k.split("_", 1)[1].isdigit() and not s.startswith(("COMPLETED",) + ARRAY_LIVE)]
        first = tasks[0][1]
        el = [v.get("elapsed_s") for _k, v in tasks if isinstance(v.get("elapsed_s"), (int, float))]
        e = {"state": st, "elapsed_s": sum(el) if el else None, "gpu_count": first.get("gpu_count"),
             "gpu_type": first.get("gpu_type"),
             "array_tasks": {"n": len(tasks), "states": dict(collections.Counter(states)), "failed": failed}}
        ref = next((v.get("refusal") for (k, v) in tasks if v.get("refusal")), None)
        if ref:
            e["refusal"] = ref
        out[str(b)] = e
        for k, _v in tasks:
            out.pop(k, None)
    return out


def sacct(job_ids):
    ids = [str(j) for j in job_ids if re.match(r"^[0-9]+(_[0-9]+)?$", str(j))]
    if not ids:
        return {"ok": True, "jobs": {}}
    cmd = os.environ.get("INCAP_SACCT", "sacct")
    rc, out, err = _run([cmd, "-X", "-P", "-j", ",".join(ids),
                         "--format=JobID,JobIDRaw,JobName,State,Elapsed,AllocTRES,NodeList"], timeout=120)
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
    # an array job (the zoo's score arrays) is one entry under its own id: every task folded into it
    return {"ok": True, "jobs": fold_arrays(jobs, ids)}


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


V3FAM_STATUS = "v3fam/v1/status.json"         # derived on the cluster each snapshot, never a file of the tree
V3FAM_PERSON_JOBS = ("v3fam_sweep", "v3famtest_", "v3fam_retest")
# Deviation 5 (2026-10-24): the code every reproduction record of the family pins (inc2.v3fam.code_shas: the six
# E3-frozen files of twostage.CODE_FILES and inc2/v3fam.py), relative to weed_optimizer_framework/; restated here so
# that the snapshot hashes the files without importing the family's modules (torch); a test checks it against both
V3FAM_CODE_FILES = {"twostage": "tools/inc2/twostage.py", "scorer": "tools/inc/scorer.py",
                    "scorer_sidecar": "tools/inc2/scorer_sidecar.py", "scorer_native": "tools/inc2/scorer_native.py",
                    "verify": "tools/inc/verify.py", "semisup_labeler": "tools/semisup_labeler.py",
                    "v3fam": "tools/inc2/v3fam.py"}
V3FAM_EQUIV_DIR = "v3fam/v1/equivalence"


def _v3fam_hours(att):
    """A finished attempt's wall time in hours (one V100 per family job), else None."""
    import datetime
    try:
        a = datetime.datetime.strptime(att.get("started_utc"), "%Y-%m-%dT%H:%M:%SZ")
        b = datetime.datetime.strptime(att.get("finished_utc"), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None
    return max(0.0, (b - a).total_seconds() / 3600.0)


def v3fam_code():
    """({name: sha256} of the code the family's jobs run, {copy, nested_equal}): V3FAM_CODE_FILES in the outer copy
    the INC jobs import (remote.package_copies), else in the running package; nested_equal says whether the
    nested copy holds the same bytes (None when there is no other copy)."""
    nested, outer = R.package_copies()
    here = Path(__file__).resolve().parents[2]
    base = outer if outer.is_dir() else here

    def shas(root):
        return {k: (R._sha256_file(root / rel) if (root / rel).is_file() else None)
                for k, rel in sorted(V3FAM_CODE_FILES.items())}
    code = shas(base)
    other = nested if base != nested else None
    eq = (shas(other) == code) if other is not None and other.is_dir() and other.resolve() != base.resolve() else None
    return code, {"copy": "outer" if base == outer else "running", "nested_equal": eq}


def v3fam_equivalence(inc):
    """The family's reproduction records against the code its jobs run (deviation 5, 2026-10-24): each record of
    v3fam/v1/equivalence (its name, status, sha256, whether its code map equals v3fam_code(), the keys that differ),
    the names taken by other code (stale; inc2.v3fam.require_equivalence refuses every family score resting on
    one), the records a recorded deviation supersedes by sha256 (inc2.v3fam_supersede: not stale, the next family
    job moves them aside) and that module's status. Hashes and names only, no score."""
    from ..inc2 import v3fam_supersede as SUP
    code, where = v3fam_code()
    d = inc / V3FAM_EQUIV_DIR
    recs, stale, sup = [], [], []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        rec, info = _read_json(p)
        sha = (info or {}).get("sha256") if isinstance(info, dict) else None
        sha = sha or (R._sha256_file(p) if p.is_file() else None)
        rc = (rec or {}).get("code") if isinstance(rec, dict) else None
        rc = rc if isinstance(rc, dict) else {}
        diff = sorted(k for k in set(code) | set(rc) if code.get(k) != rc.get(k))
        rel = "%s/%s" % (V3FAM_EQUIV_DIR, p.name)
        superseded = SUP.is_superseded(rel, sha)
        recs.append({"name": p.name, "status": (rec or {}).get("status") if isinstance(rec, dict) else None,
                     "sha256": sha, "code_match": not diff, "differs": diff, "superseded": superseded})
        if superseded:
            sup.append(p.name)
        elif diff:
            stale.append(p.name)
    st = SUP.status(inc)
    return {"code": code, "fingerprint": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
            "copy": where["copy"], "nested_equal": where["nested_equal"], "records": recs, "stale": stale,
            "superseded": sup, "supersede": {k: st[k] for k in ("pending", "moved", "other", "absent")}}


def v3fam_code_at(package):
    """{name: sha256 or None} of V3FAM_CODE_FILES under a weed_optimizer_framework directory (the deploy's tree)."""
    package = Path(package)
    return {k: (R._sha256_file(package / rel) if (package / rel).is_file() else None)
            for k, rel in sorted(V3FAM_CODE_FILES.items())}


V3FAM_CODE_RE = re.compile(r"weed_optimizer_framework/(%s)$" % "|".join(
    re.escape(v) for _k, v in sorted(V3FAM_CODE_FILES.items())))


def v3fam_reopen_blocks(dom):
    """The domain's reopen blocks in order: v3fam.reopen (deviation 4's; an object, as a ticker of an earlier deploy
    reads it while a deploy copies the new domain file before the restart; a list is read too) and then
    v3fam.reopens (deviation 5's and later, a list)."""
    fam = (dom or {}).get("v3fam") or {}
    out = []
    for key in ("reopen", "reopens"):
        ro = fam.get(key) or []
        out += [ro] if isinstance(ro, dict) else [b for b in ro if isinstance(b, dict)]
    return out


def deploy_descends(git_root, deployed_text, head="HEAD"):
    """(ok, why): the tree deployed now (the git_head of the lab's stream_deployed.json, deployed_text) is an ancestor
    of the head a deploy ships (deploy/deploy_funnel.sh, before anything is copied). A deploy from a branch that does
    not hold the deployed commit would put older files back over newer ones (rsync without --delete keeps only what
    it does not overwrite), so it is refused: so is a deployed head this checkout does not know (fetch it first). A
    deploy record that is absent or unreadable passes with that said (a first deploy)."""
    try:
        d = json.loads(deployed_text or "")
    except ValueError:
        d = None
    dep = str((d or {}).get("git_head") or "").strip() if isinstance(d, dict) else ""
    if not re.fullmatch(r"[0-9a-f]{40}", dep):
        return True, "no deployed git_head on the lab (%s): nothing to compare" % ("unreadable record" if deployed_text
                                                                                 else "no record")
    try:
        known = subprocess.run(["git", "-C", str(git_root), "cat-file", "-e", dep + "^{commit}"],
                               capture_output=True, timeout=60).returncode == 0
        if not known:
            return False, ("the deployed head %s is not in this checkout: fetch it, and deploy only a head that "
                           "descends from it" % dep[:12])
        anc = subprocess.run(["git", "-C", str(git_root), "merge-base", "--is-ancestor", dep, head],
                             capture_output=True, timeout=60).returncode
    except (OSError, subprocess.SubprocessError) as e:
        return False, "git could not compare the deployed head %s with %s: %s" % (dep[:12], head, e)
    if anc != 0:
        return False, ("%s does not descend from the deployed head %s: the deploy would put older files back over "
                       "newer ones; merge or rebase onto it first" % (head, dep[:12]))
    return True, "%s descends from the deployed head %s" % (head, dep[:12])


def v3fam_allow_ok(dom, git_root, allow, roots):
    """(ok, why) for --allow-v3fam-code-change=<file>: the file is the newest recorded deviation, i.e. the file of the
    domain's last reopen block (a deviation that changes the code the records pin comes with its own reopen block,
    which names what it moves aside and retakes), shipped among the git-root paths (roots) and hashing in git_root
    as that block pins it. Any other file (the research log, an earlier deviation, an unpinned or edited one) is
    refused."""
    if not allow:
        return False, "no --allow-v3fam-code-change"
    if allow not in set(roots or ()):
        return False, "%s is not a git-root path this deploy ships" % allow
    blocks = v3fam_reopen_blocks(dom)
    blk = blocks[-1] if blocks and blocks[-1].get("file") == allow else None
    if blk is None:
        return False, "%s is not the newest recorded deviation (the domain's last reopen block names %s)" % (
            allow, blocks[-1].get("file") if blocks else "none")
    p = Path(git_root) / allow
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    if got != blk.get("sha256"):
        return False, "%s hashes %s, its reopen block pins %s" % (allow, str(got)[:12], str(blk.get("sha256"))[:12])
    return True, "%s (sha256 %s), the deviation of the reopen block %r" % (allow, got[:12], blk.get("marker"))


def v3fam_deploy_check(dom, git_root, root_paths, allow=None, package=None):
    """(ok, [lines]): deviation 5's deploy-time checks on the tree a deploy ships (deploy/deploy_funnel.sh runs them
    before anything is copied, and in --dry-run; tests/test_stream_pipeline.py runs them):
      (a) the code every reproduction record of the family pins (V3FAM_CODE_FILES, inc2.v3fam.code_shas) hashes as
          the domain's v3fam.equivalence_code, the code map the cluster's records were taken with; a changed file
          is refused by name (on 2026-10-21 deviation 4's CLI-only change of inc2/v3fam.py left every record taken
          by other code, and twelve score jobs of six arms were refused); allow, a recorded deviation's file among
          the deploy's git-root paths, lets it pass with a notice (the deviation then says which records it moves
          aside and retakes);
      (b) every file the domain's v3fam block names by sha256 (the amendment, its Revisions, each reopen block's
          deviation file) is among the git-root paths and hashes so in this tree; else the ticker would never
          reopen what a deviation names, and nothing would say so."""
    fam = (dom or {}).get("v3fam") or {}
    git_root = Path(git_root)
    package = Path(package) if package else Path(__file__).resolve().parents[2]
    roots = set(root_paths or ())
    lines, ok = [], True
    pin = fam.get("equivalence_code")
    now = v3fam_code_at(package)
    if not isinstance(pin, dict) or set(pin) != set(V3FAM_CODE_FILES):
        ok = False
        lines.append("refused: the domain's v3fam.equivalence_code does not name the code the family's reproduction "
                     "records pin (%s)" % ", ".join(sorted(V3FAM_CODE_FILES)))
    else:
        changed = sorted(k for k in V3FAM_CODE_FILES if now.get(k) != pin.get(k))
        if not changed:
            lines.append("pass: the code the reproduction records pin (inc2.v3fam.code_shas) is the domain's "
                         "v3fam.equivalence_code (v3fam %s)" % str(now.get("v3fam"))[:12])
        elif allow and v3fam_allow_ok(dom, git_root, allow, roots)[0]:
            lines.append("NOTICE: %s change(s) the code the reproduction records pin (%s); allowed by the recorded "
                         "deviation %s: every record taken by other code is refused by inc2.v3fam.require_equivalence "
                         "until that deviation moves it aside and retakes it" % (
                             "this tree", ", ".join("%s %s -> %s" % (k, str(pin.get(k))[:12], str(now.get(k))[:12])
                                                    for k in changed), v3fam_allow_ok(dom, git_root, allow, roots)[1]))
        else:
            ok = False
            lines.append("refused: this tree changes the code every reproduction record of the family pins (%s; "
                         "inc2.v3fam.code_shas): each family score would be refused ('the code changed since ... "
                         "ran'). Keep those files' bytes (put CLI or plumbing elsewhere, as deviation 5 did with "
                         "inc2/v3fam_verdict.py), or record a deviation that moves the records aside and retakes "
                         "them, ship it among the git-root paths and pass --allow-v3fam-code-change=<its file>%s" % (
                             ", ".join("%s %s -> %s" % (k, str(pin.get(k))[:12], str(now.get(k))[:12])
                                       for k in changed),
                             "" if not allow else " (not allowed: %s)" % v3fam_allow_ok(dom, git_root, allow,
                                                                                      roots)[1]))
    named = [(fam.get("amendment_file"), fam.get("amendment_sha256")),
             (fam.get("revision_file"), fam.get("revision_sha256"))]
    named += [(b.get("file"), b.get("sha256")) for b in v3fam_reopen_blocks(dom)]
    bad = []
    for f, sha in named:
        if not f:
            continue
        p = git_root / f
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if f not in roots:
            bad.append("%s is not a git-root path the deploy ships" % f)
        elif got != sha:
            bad.append("%s hashes %s, the domain pins %s" % (f, str(got)[:12], str(sha)[:12]))
    if bad:
        ok = False
        lines.append("refused: a file the domain's v3fam block pins by sha256 would not be deployed as pinned (the "
                     "ticker would never act on it, and nothing would say so): %s" % "; ".join(bad))
    else:
        lines.append("pass: every file the domain's v3fam block pins by sha256 (%d) is shipped with that sha256"
                     % len([f for f, _s in named if f]))
    return ok, lines


def _records_list(records):
    """[{name, sha256, code}] from step 2's fetch ({"records": [...]}, its JSON text, or a list), or None when the
    fetch failed or says so; an int n is n records of unknown code (an older caller's count)."""
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except ValueError:
            try:
                records = int(records)
            except ValueError:
                return None
    if isinstance(records, bool):
        return None
    if isinstance(records, int):
        return [{"name": "record %d" % (k + 1), "sha256": None, "code": None} for k in range(records)]
    if isinstance(records, dict):
        if records.get("error"):
            return None
        records = records.get("records")
    if not isinstance(records, list):
        return None
    return [r for r in records if isinstance(r, dict)]


def v3fam_records_guard(changed, records, allow=None, roots=(), dom=None, git_root=None, package=None):
    """(ok, message) for deploy_funnel.sh step 2, beside PINNED_RE: `changed`, the paths rsync would change on the
    cluster; `records`, the cluster's reproduction records under v3fam/v1/equivalence as step 2 fetched them (each
    with its name, sha256 and code map; a failed fetch reads as records present, taken by unknown code). A change to
    a file the records pin (V3FAM_CODE_RE) passes when, after it, every record but those a recorded deviation
    supersedes by sha256 (inc2.v3fam_supersede) pins exactly the code this tree ships (v3fam_code_at(package)): a
    deploy that restores the bytes the records pin (deviation 5) or changes nothing they pin. Otherwise it is
    refused unless `allow` is a recorded deviation (v3fam_allow_ok: the file of one of the domain's reopen blocks,
    shipped and hashing as pinned)."""
    hit = sorted({c for c in changed or () if V3FAM_CODE_RE.search(str(c))})
    if not hit:
        return True, "pass: no file the family's reproduction records pin changes on the cluster"
    recs = _records_list(records)
    if recs is not None and not recs:
        return True, "NOTICE: %s change on the cluster; it holds no reproduction record" % ", ".join(hit)
    shipped = v3fam_code_at(Path(package) if package else Path(__file__).resolve().parents[2])
    if recs is not None:
        try:
            from ..inc2 import v3fam_supersede as SUP
            sup = {r.get("name") for r in recs
                   if SUP.is_superseded("%s/%s" % (V3FAM_EQUIV_DIR, r.get("name")), r.get("sha256"))}
        except ImportError:
            sup = set()
        other = sorted(str(r.get("name")) for r in recs if r.get("name") not in sup and r.get("code") != shipped)
        if not other:
            return True, ("pass: %s change on the cluster to the code every one of its %d reproduction record(s) pins "
                          "(v3fam %s)%s" % (", ".join(hit), len(recs) - len(sup), str(shipped.get("v3fam"))[:12],
                                            "; superseded by a recorded deviation and moved aside by the next family "
                                            "job: %s" % ", ".join(sorted(sup)) if sup else ""))
        held = "%d reproduction record(s) taken by other code than this tree ships (%s)" % (len(other),
                                                                                           ", ".join(other[:8]))
    else:
        held = "reproduction records (the fetch failed: read as present, taken by unknown code)"
    ok_allow, why_allow = v3fam_allow_ok(dom, git_root, allow, roots) if allow else (False, "")
    if ok_allow:
        return True, "NOTICE: %s change on the cluster, which holds %s; allowed by the recorded deviation %s" % (
            ", ".join(hit), held, why_allow)
    return False, ("refused: the deploy would change %s on the cluster, which holds %s "
                   "(inc2.v3fam.require_equivalence would refuse every family score resting on them); pass "
                   "--allow-v3fam-code-change=<a recorded deviation's file> only with a deviation that moves them "
                   "aside and retakes them%s" % (", ".join(hit), held,
                                                 " (not allowed: %s)" % why_allow if allow else ""))


def v3fam_status(inc):
    """V3F's state the ticker's gate reads (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-11, V3F.6.4), derived from
    the family's own files: the time stamp's presence and sha256, D0-S's record (complete, the trigger's
    branches, counts), the EXIF frame pre-flight (passed), D6's smoke and seeds (status, the walltime rule), the
    items recorded 'not run (family cap)', and the person steps' GPU-h from their provenance. No score, no exam
    key, no path of an evaluation image."""
    v = inc / "v3fam" / "v1"
    out = {"format": "inc_autopilot.v3fam_status/1"}
    # the family's data summary, projected (it may pass the shipped-file cap): complete, its sha256, each arm's
    # images and manifest sha256 (what L23I's price and the builds' order read)
    sp = inc / "splits" / "v3fam" / "summary.json"
    summ = None
    if sp.is_file():
        try:
            with open(str(sp)) as fh:
                summ = json.load(fh)
        except (OSError, ValueError):
            summ = None
    ok = isinstance(summ, dict) and summ.get("format") == "inc2-v3fam-data/1" and summ.get("status") == "complete"
    out["data"] = {"complete": ok, "sha256": R._sha256_file(sp) if sp.is_file() else None,
                   "arms": {a: {"images": (x or {}).get("images"), "sha256": (x or {}).get("sha256")}
                            for a, x in sorted(((summ or {}).get("arms") or {}).items())} if ok else {}}
    ts = v / "timestamp.json"
    out["timestamp"] = {"exists": ts.is_file(), "sha256": R._sha256_file(ts) if ts.is_file() else None}
    rec, _i = _read_json(v / "sweep" / "record.json")
    ok = isinstance(rec, dict) and rec.get("format") == "inc2-v3fam-sweep-record/1" and rec.get("status") == "complete"
    trig = (rec or {}).get("trigger") or {} if ok else {}
    out["sweep"] = {"complete": ok, "bclean": bool(trig.get("bclean")), "test_branch": bool(trig.get("test_branch")),
                    "dev_branch": bool(trig.get("dev_branch")), "h_images": len((rec or {}).get("H") or []) if ok
                    else None}
    pre, _i = _read_json(v / "exif" / "preflight.json")
    ver, _i = _read_json(v / "exif" / "verdict.json")
    pre_ok = isinstance(pre, dict) and pre.get("numeric_passed") is True
    out["exif"] = {"preflight": isinstance(pre, dict), "verdict": (ver or {}).get("frame") if isinstance(ver, dict)
                   else None, "passed": bool(pre_ok and isinstance(ver, dict) and ver.get("frame") == "yes"
                                         and ver.get("preflight_sha256") == R._sha256_file(v / "exif" /
                                                                                          "preflight.json"))}
    sm, _i = _read_json(v / "d6" / "smoke" / "smoke.json")
    # deviation 5 (2026-10-24): the smoke record of 2026-10-21, which took the whole one-epoch fit as the epoch time,
    # is superseded by sha256 (inc2.v3fam_supersede): reported as no smoke, so the ticker proposes the smoke again;
    # the smoke's own job moves it aside
    sm_sha = R._sha256_file(v / "d6" / "smoke" / "smoke.json") if (v / "d6" / "smoke" / "smoke.json").is_file() \
        else None
    try:
        from ..inc2 import v3fam_supersede as SUP
        sm_sup = SUP.is_superseded(SUP.D6_SMOKE, sm_sha)
    except Exception:                           # noqa: BLE001 - an older tree: nothing is superseded
        sm_sup = False
    if sm_sup:
        sm = None
    out["detr"] = {"smoke": (sm or {}).get("status") if isinstance(sm, dict) else None,
                   "walltime_ok": (sm or {}).get("walltime_ok") if isinstance(sm, dict) else None,
                   "projection_h": (sm or {}).get("projection_h") if isinstance(sm, dict) else None,
                   "epoch_seconds": (sm or {}).get("epoch_seconds") if isinstance(sm, dict) else None,
                   "smoke_sha256": sm_sha, "smoke_superseded": bool(sm_sup),
                   "trained": sorted(int(p.parent.name[1:]) for p in (v / "d6").glob("s[0-9]/train.json"))
                   if (v / "d6").is_dir() else []}
    out["cap_refused"] = sorted(p.stem for p in (v / "cap").glob("*.json")) if (v / "cap").is_dir() else []
    items, hours = [], 0.0
    pdir = inc / "_campaign" / "provenance"
    for p in sorted(pdir.glob("v3fam*.json")) if pdir.is_dir() else []:
        if not any(p.stem.startswith(n) for n in V3FAM_PERSON_JOBS):
            continue
        prov, _i = _read_json(p)
        for att in (prov or {}).get("attempts") or [] if isinstance(prov, dict) else []:
            h = _v3fam_hours(att)
            items.append({"job": p.stem, "state": "ended" if att.get("finished_utc") else "running", "hours": h})
            hours += h or 0.0
    out["person_jobs"] = items
    out["person_settled_gpuh"] = round(hours, 4)
    try:
        out["equivalence"] = v3fam_equivalence(inc)
    except Exception as e:                      # reported, never fatal: the ticker then reads it as unknown
        out["equivalence_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return out


DS1_STATUS = "ds1/v1/status.json"             # derived on the cluster each snapshot, never a file of the tree
DS1_PERSON_JOBS = ("ds1test_",)


def ds1_status(inc):
    """DS1's state the ticker's gate and plan read (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29, DS.9.3), derived
    from DS1's own files: the time stamp, RA, the power pre-flight, MD-S0's drafts and freeze, each phase's screens and
    sources (sheet, audit, pool, MD-N, MT-W v2), the sheets' bundles and renders, the MT-W checks and LN, the arms
    built, the long path (the pair's compute, smokes, seeds), the score records and verdicts (their statuses), the
    reads prepared and taken, and the person jobs' GPU-h. No score, no exam key, no path of an evaluation image.
    DS1's modules read INC_DIR (inc.common): it is bound to `inc` for the call (on the cluster they are the same
    directory)."""
    from ..inc import common as CM
    prev = CM.INC_DIR
    CM.INC_DIR = Path(inc)
    try:
        return _ds1_status(Path(inc))
    finally:
        CM.INC_DIR = prev


def _ds1_status(inc):
    from ..inc2 import ds1 as D
    from ..inc2 import ds1_data as DD
    from ..inc2 import ds1_long as LG
    out = {"format": "inc_autopilot.ds1_status/1", "timestamp": D.timestamp_ok(inc / "ds1" / "v1" / "timestamp.json"),
           # what the time stamp still lacks: the card names only those steps (DS1 Revision 2, item 6)
           "timestamp_missing": D.timestamp_missing(inc / "ds1" / "v1" / "timestamp.json")}
    out.update(DD.plan_state(inc))
    out["long"] = LG.status()
    sc = {}
    for a in D.SCORE_ARMS:
        r, _i = _read_json(inc / "capacity" / ("ds1_score_%s.json" % a.lower()))
        sc[a] = (r or {}).get("status") if isinstance(r, dict) else None
    out["scores"] = sc
    out["verdicts"] = {}
    for ph in (1, 2):
        r, _i = _read_json(inc / "capacity" / ("ds1_v%d.json" % ph))
        out["verdicts"][str(ph)] = {"status": (r or {}).get("status") if isinstance(r, dict) else None,
                                    "reads": ((r or {}).get("reads") or {}).get("order") if isinstance(r, dict) else None,
                                    "substitution": ((r or {}).get("substitution") or {}).get("applied")
                                    if isinstance(r, dict) else None}
    out["reads_done"] = sorted("%s:%s" % k for k in D.reads_done())
    items, hours = [], 0.0
    pdir = inc / "_campaign" / "provenance"
    for p in sorted(pdir.glob("ds1*.json")) if pdir.is_dir() else []:
        if not any(p.stem.startswith(n) for n in DS1_PERSON_JOBS):
            continue
        prov, _i = _read_json(p)
        for att in (prov or {}).get("attempts") or [] if isinstance(prov, dict) else []:
            h = _v3fam_hours(att)
            items.append({"job": p.stem, "state": "ended" if att.get("finished_utc") else "running", "hours": h})
            hours += h or 0.0
    out["person_jobs"] = items
    out["person_settled_gpuh"] = round(hours, 4)
    return out


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
    # E2 (2026-10-04): its verdict and its rescore's record (dev only; e2_v1_report.* and the test read's
    # e2_test_*.* are for people, never read here)
    put("capacity/e2_v1.json", inc / "capacity" / "e2_v1.json")
    put("capacity/e2_rescore.json", inc / "capacity" / "e2_rescore.json")
    # E2-C (2026-10-04, later): the attribution record and its rescore's record (dev only; e2_attr_v1_report.md
    # and E2-C's test read are for people, never read here)
    put("capacity/e2_attr_v1.json", inc / "capacity" / "e2_attr_v1.json")
    put("capacity/e2_attr_rescore.json", inc / "capacity" / "e2_attr_rescore.json")
    # the model-zoo audit (L23Z, 2026-10-04): its record holds status, job ids, counts and sha256s only (no exam
    # name as a key, no score path, no metric); the zoo's reports are for people and never read here
    put("capacity/zoo_v1.json", inc / "capacity" / "zoo_v1.json")
    # E3 (2026-10-05): each arm's score record, the verdict and its rescore's record (dev only; e3_v1_report.*,
    # the test read's e3_test_*.* and INC_DIR/twostage are for people, never read here)
    for arm in ("M", "A", "B"):
        put("capacity/e3_score_%s.json" % arm, inc / "capacity" / ("e3_score_%s.json" % arm))
    put("capacity/e3_v1.json", inc / "capacity" / "e3_v1.json")
    put("capacity/e3_rescore.json", inc / "capacity" / "e3_rescore.json")
    # V3F (2026-10-11): the family's data summary, each arm's score record and the verdict (dev only; the external
    # records, the reports, the test reads and INC_DIR/v3fam are for people, never read here), and the derived
    # status of the family's person steps and gates (v3fam_status: booleans, counts and sha256s, no score)
    put("splits/v3fam/summary.json", inc / "splits" / "v3fam" / "summary.json")
    for arm in V3FAM_SCORE_ARMS:
        put("capacity/v3fam_score_%s.json" % arm.lower(), inc / "capacity" / ("v3fam_score_%s.json" % arm.lower()))
    put("capacity/v3fam_v1.json", inc / "capacity" / "v3fam_v1.json")
    # deviation 4 (2026-10-20): the continuation verdict over every arm (dev only), the record the domain's
    # v3fam.verdict_record named once the verdict of 2026-10-13 was in place; deviation 5 (2026-10-24): the second
    # continuation (dev only), the record it names once the continuation of 2026-10-22 is in place
    put("capacity/v3fam_v1_cont.json", inc / "capacity" / "v3fam_v1_cont.json")
    put("capacity/v3fam_v1_cont2.json", inc / "capacity" / "v3fam_v1_cont2.json")
    try:
        arts[V3FAM_STATUS] = _dev(v3fam_status(inc))
    except Exception as e:                      # a status that cannot be read is reported, never fatal
        notes.append("%s: %s: %s" % (V3FAM_STATUS, type(e).__name__, str(e)[:200]))
    # DS1 (2026-10-29): its data index, each arm's score record, the power pre-flight, RA and the phase verdicts (dev
    # and MD only; the external records, the reports, the sealed reads and INC_DIR/ds1 are for people, never read
    # here), and the derived status of its gates, person steps and items (ds1_status: booleans, counts and sha256s)
    put("splits/ds1/summary.json", inc / "splits" / "ds1" / "summary.json")
    for arm in DS1_SCORE_ARMS:
        put("capacity/ds1_score_%s.json" % arm.lower(), inc / "capacity" / ("ds1_score_%s.json" % arm.lower()))
    for n in ("ds1_v1", "ds1_v2", "ds1_power", "ds1_ra_gate"):
        put("capacity/%s.json" % n, inc / "capacity" / ("%s.json" % n))
    if (inc / "ds1").is_dir() or (inc / "splits" / "ds1").is_dir():
        try:
            arts[DS1_STATUS] = _dev(ds1_status(inc))
        except Exception as e:                  # reported, never fatal: the ticker then reads DS1's state as unknown
            notes.append("%s: %s: %s" % (DS1_STATUS, type(e).__name__, str(e)[:200]))
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
                            "--role": ("role", _ENUM("b_v2", "capacity", "canary", "union", "baseline")),
                            "--e2": ("e2", _ENUM("W", "S", "C")),
                            # V3F (2026-10-11): the family's build arms; the manifest, arm and finals are the
                            # family's (inc2.baseline build --v3fam)
                            "--v3fam": ("v3fam", _ENUM(*V3FAM_BUILD_ARMS)),
                            # DS1 (2026-10-29): its 1.2 M arms (inc2.baseline build --ds1)
                            "--ds1": ("ds1", _ENUM(*DS1_BUILD_ARMS))},
    ("baseline", "rescore-native"): {"--exp": ("exp", _NAME), "--reference": ("reference", _NAME)},
    ("baseline", "rescore-e2"): {},
    ("baseline", "rescore-e2-attr"): {},
    ("twostage", "score-arm"): {"--arm": ("arm", _ENUM("M", "A", "B"))},
    ("twostage", "verdict"): {},
    ("v3fam_data", "build"): {},
    ("v3fam", "score"): {"--arm": ("arm", _ENUM(*V3FAM_SCORE_ARMS))},
    ("v3fam_verdict", "verdict"): {"--cap-refused": ("cap_refused", V3FAM_CAP_ITEMS_RE),
                                   "--ticker-failed": ("ticker_failed", V3FAM_FAILED_ITEMS_RE)},
    ("ds1_data", "build"): {"--phase": ("phase", _ENUM("1", "2")), "--step": ("step", _ENUM(*DS1_STEPS))},
    ("ds1", "score"): {"--arm": ("arm", _ENUM(*DS1_SCORE_ARMS))},
    ("ds1", "verdict"): {"--phase": ("phase", _ENUM("1", "2")), "--cap-refused": ("cap_refused", DS1_ITEMS_RE),
                         "--ticker-failed": ("ticker_failed", DS1_ITEMS_RE)},
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
_ZOO_SHARDS = re.compile(r"(?:[1-9]|[1-9][0-9]|1[0-9][0-9]|200)\Z")
ZOO_FLAGS = {"--version": ("version", _ENUM("v1")), "--shards-a": ("shards_a", _ZOO_SHARDS),
             "--shards-c": ("shards_c", _ZOO_SHARDS), "--concurrency": ("concurrency", re.compile(r"[1-8]\Z")),
             "--max-gpu-hours": ("max_gpu_hours", re.compile(r"(?:[1-9]|[1-3][0-9]|4[0-5])\Z"))}
REQUIRED = {("collect", "fetch"): ("source", "max_bytes"), ("collect", "intake"): ("source",),
            ("zoo", "submit"): ("version", "shards_a", "shards_c", "concurrency", "max_gpu_hours"),
            ("admit", "admit"): ("intake",), ("admit", "scan-holds"): ("hold",),
            ("stream", "init"): ("stream", "stage_b"),
            ("stream", "build"): ("stream", "k"), ("stream", "milestone"): ("stream",),
            ("stream", "fork"): ("stream", "m"), ("stream", "feasibility"): ("stream", "holdout", "m"),
            ("stream", "bisect"): ("stream", "from_pool"), ("baseline", "build"): ("exp", "seeds", "arm", "role"),
            ("baseline", "rescore-native"): ("exp", "reference"),
            ("baseline", "rescore-agnostic"): ("exp", "reference"), ("base3", "build"): ("stream",),
            ("pilot4", "build"): ("exp", "from_exp", "recipes"), ("twostage", "score-arm"): ("arm",),
            ("v3fam", "score"): ("arm",), ("v3fam_detr", "train"): ("seeds",),
            ("ds1_data", "build"): ("phase", "step"), ("ds1", "score"): ("arm",), ("ds1", "verdict"): ("phase",),
            ("ds1_long", "smoke"): ("arm",), ("ds1_long", "train"): ("arm", "seeds")}
DETR_FLAGS = {("v3fam_detr", "smoke"): {}, ("v3fam_detr", "train"): {"--seeds": ("seeds", V3FAM_DETR_SEEDS_RE)}}
LONG_FLAGS = {("ds1_long", "smoke"): {"--arm": ("arm", _ENUM(*DS1_LONG_ARMS))},
              ("ds1_long", "train"): {"--arm": ("arm", _ENUM(*DS1_LONG_ARMS)), "--seeds": ("seeds", DS1_SEEDS_RE),
                                      "--walltime-h": ("walltime_h", _ENUM("24", "36"))}}


def parse_submit(kind, args):
    """{"kind", "script", "verb", "module", "pkg", "params", "script_args"} or R.Refused."""
    args = [str(a) for a in args]
    if kind in ("collect", "admit", "zoo"):
        if not args or args[0] not in VERBS[kind]:
            raise R.Refused("%s takes a verb among %s" % (kind, VERBS[kind]))
        verb = args[0]
        spec = ZOO_FLAGS if kind == "zoo" else (COLLECT_FLAGS if kind == "collect" else ADMIT_FLAGS)[verb]
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
        need = REQUIRED.get((mod, verb), ())
        if (mod, verb) == ("baseline", "build") and "ds1" in params:
            # DS1 (2026-10-29): a DS1 build takes its manifest, arm and finals from DS1's definition
            extra = sorted(k for k in ("manifest", "union", "arm", "e2", "v3fam") if k in params)
            if extra:
                raise R.Refused("baseline build --ds1 takes no %s" % ", ".join(extra))
            need = ("exp", "seeds", "role")
            if params.get("seeds") != "0,1,2" or params.get("role") != "baseline":
                raise R.Refused("a DS1 build trains seeds 0,1,2 as role baseline")
        elif (mod, verb) == ("baseline", "build") and "v3fam" in params:
            # V3F (2026-10-11): a family build takes its manifest, arm and finals from the family's definition
            extra = sorted(k for k in ("manifest", "union", "arm", "e2") if k in params)
            if extra:
                raise R.Refused("baseline build --v3fam takes no %s" % ", ".join(extra))
            need = ("exp", "seeds", "role")
            if params.get("seeds") != "0,1,2" or params.get("role") != "baseline":
                raise R.Refused("a V3F build trains seeds 0,1,2 as role baseline")
        missing = [p for p in need if p not in params]
        if missing:
            raise R.Refused("%s %s needs %s" % (mod, verb, missing))
        if (mod, verb) == ("baseline", "build") and "v3fam" not in params and "ds1" not in params \
                and ("manifest" in params) == ("union" in params):
            raise R.Refused("baseline build takes exactly one of --manifest and --union")
        short = "%s.%s" % (m.group("pkg"), mod)
        return {"kind": kind, "script": SCRIPTS[kind], "verb": verb, "module": mod, "pkg": m.group("pkg"),
                "params": params, "script_args": [short] + args[1:]}
    if kind == "detr":
        # V3F (2026-10-11): D6's RF-DETR jobs (L23K): run_inc2_detr.sh <pkg>.v3fam_detr smoke | train --seeds 0,1,2
        if len(args) < 2:
            raise R.Refused("detr takes <pkg>.v3fam_detr <verb> FLAGS")
        m = PKG_RE.match(args[0])
        if not m or args[1] not in DETR_VERBS.get(m.group("mod"), ()):
            raise R.Refused("%s is not a V3F RF-DETR job (%s)" % (" ".join(args[:2]), DETR_VERBS))
        mod, verb = m.group("mod"), args[1]
        params = _flags(args[2:], DETR_FLAGS[(mod, verb)])
        missing = [p for p in REQUIRED.get((mod, verb), ()) if p not in params]
        if missing:
            raise R.Refused("%s %s needs %s" % (mod, verb, missing))
        short = "%s.%s" % (m.group("pkg"), mod)
        return {"kind": kind, "script": SCRIPTS[kind], "verb": verb, "module": mod, "pkg": m.group("pkg"),
                "params": params, "script_args": [short, verb]}
    if kind == "long":
        # DS1 (2026-10-29): the long path (L32X): run_inc2_long.sh <pkg>.ds1_long smoke --arm A | train --arm A
        # --seeds S [--walltime-h 36]
        if len(args) < 2:
            raise R.Refused("long takes <pkg>.ds1_long <verb> FLAGS")
        m = PKG_RE.match(args[0])
        if not m or args[1] not in LONG_VERBS.get(m.group("mod"), ()):
            raise R.Refused("%s is not a DS1 long-path job (%s)" % (" ".join(args[:2]), LONG_VERBS))
        mod, verb = m.group("mod"), args[1]
        params = _flags(args[2:], LONG_FLAGS[(mod, verb)])
        missing = [p for p in REQUIRED.get((mod, verb), ()) if p not in params]
        if missing:
            raise R.Refused("%s %s needs %s" % (mod, verb, missing))
        if params.get("walltime_h") == "36" and params.get("arm") not in ("S2X", "S1UX"):
            raise R.Refused("only S2x's (S1ux's) one re-submission takes 36 h (DS.6)")
        short = "%s.%s" % (m.group("pkg"), mod)
        return {"kind": kind, "script": SCRIPTS[kind], "verb": verb, "module": mod, "pkg": m.group("pkg"),
                "params": params, "script_args": [short, verb, "--arm", params["arm"]]}
    raise R.Refused("stream-submit kind %r is not one of %s" % (kind, sorted(SCRIPTS)))


def long_job_names(req):
    """The job names of an L32X submission: the smoke's, or one per seed named by --seeds (ascending)."""
    arm = req["params"]["arm"].lower()
    if req["verb"] == "smoke":
        return [DS1_LONG_JOB_NAME % (arm, "smoke")]
    return [DS1_LONG_JOB_NAME % (arm, "s%s" % k) for k in sorted(str(req["params"].get("seeds") or "").split(","))
            if k != ""]


def detr_job_names(req):
    """The job names of an L23K submission: the smoke's, or one per seed named by --seeds (ascending)."""
    names = list(V3FAM_DETR_JOB_NAMES[req["verb"]])
    if req["verb"] != "train":
        return names
    want = str(req["params"].get("seeds") or "").split(",")
    return [n for n in names if n.rsplit("_s", 1)[1] in want]


def job_name(req, meta):
    p = req["params"]
    if req["kind"] == "zoo":
        return ZOO_JOB_NAMES[0] % p["version"]
    if req["kind"] == "detr":
        return detr_job_names(req)[0]
    if req["kind"] == "long":
        return long_job_names(req)[0]
    if req["kind"] == "build":
        if (req["module"], req["verb"]) == ("baseline", "rescore-native"):
            # it builds nothing: its own name, never the arm's build job's (inc_build_<exp>)
            return NATIVE_JOB_NAME % p["exp"]
        if (req["module"], req["verb"]) == ("baseline", "rescore-agnostic"):
            return AGNOSTIC_JOB_NAME % p["exp"]
        if (req["module"], req["verb"]) == ("baseline", "rescore-e2"):
            return E2_JOB_NAME
        if (req["module"], req["verb"]) == ("baseline", "rescore-e2-attr"):
            return E2_ATTR_JOB_NAME
        if (req["module"], req["verb"]) == ("twostage", "score-arm"):
            return E3_SCORE_JOB_NAME % p["arm"].lower()
        if (req["module"], req["verb"]) == ("twostage", "verdict"):
            return E3_JOB_NAME
        if (req["module"], req["verb"]) == ("v3fam_data", "build"):
            return V3FAM_DATA_JOB_NAME
        if (req["module"], req["verb"]) == ("v3fam", "score"):
            return V3FAM_SCORE_JOB_NAME % p["arm"].lower()
        if (req["module"], req["verb"]) == ("v3fam_verdict", "verdict"):
            return V3FAM_JOB_NAME
        if (req["module"], req["verb"]) == ("ds1_data", "build"):
            return DS1_DATA_JOB_NAME % (int(p["phase"]), p["step"])
        if (req["module"], req["verb"]) == ("ds1", "score"):
            return DS1_SCORE_JOB_NAME % p["arm"].lower()
        if (req["module"], req["verb"]) == ("ds1", "verdict"):
            return DS1_VERDICT_JOB_NAME % int(p["phase"])
        if (req["module"], req["verb"]) == ("base3", "build"):
            return BASE3_JOB_NAME
        tag = meta.get("child_exp") or (p.get("exp") if req["module"] in ("baseline", "pilot4") else None) \
            or "%s_%s_%s" % (req["module"], req["verb"], p.get("stream") or req["pkg"])
        return "inc_build_%s" % tag
    tag = p.get("source") or p.get("intake") or p.get("hold") or req["verb"]
    return "inc_stream_%s_%s_%s" % (req["kind"], req["verb"].replace("-", ""), re.sub(r"[^A-Za-z0-9_.-]", "_", tag))


def provenance_env(prov):
    """{INCAP_<FIELD>: value} of a submission's provenance and INCAP_REQUESTED_UTC: what a stream job is given."""
    out = {"INCAP_" + k.upper(): (",".join(v) if isinstance(v, list) else str(v)) for k, v in prov.items()}
    out["INCAP_REQUESTED_UTC"] = M.utc_now()
    return out


def sbatch_export(prov):
    """sbatch's --export naming the provenance variables of `prov` (deviation 3; PROVENANCE_ENV): their names only,
    in PROVENANCE_ENV's order (a field outside it after them), INCAP_REQUESTED_UTC always among them."""
    have = {"INCAP_" + k.upper() for k in prov} | {"INCAP_REQUESTED_UTC"}
    return "--export=%s" % ",".join([k for k in PROVENANCE_ENV if k in have] + sorted(have - set(PROVENANCE_ENV)))


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
        if kind == "zoo":
            # the script runs here, on the login node, and submits its own chain (inc2.zoo submit)
            argv = [os.environ.get("INCAP_BASH", "bash"), str(script)] + req["script_args"]
            names = [n % req["params"]["version"] for n in ZOO_JOB_NAMES]
            rec.update(job_name=name, job_names=names, argv=argv, partition=PARTITION)
        elif kind == "long":
            # DS1's long path (L32X): the smoke, or one sbatch of run_inc2_long.sh per seed (36 h only for S2x's one
            # re-submission: --time=36:00:00 with --walltime-h 36)
            sbatch = os.environ.get("INCAP_SBATCH", "sbatch")
            names = long_job_names(req)
            wall = req["params"].get("walltime_h")
            argvs = []
            for n in names:
                tail = req["script_args"] + (["--seed", n.rsplit("_s", 1)[1]] if req["verb"] == "train" else []) + \
                    (["--walltime-h", wall] if wall else [])
                argvs.append([sbatch, "--parsable", "--job-name=%s" % n, "-p", PARTITION] +
                             (["--time=36:00:00"] if wall == "36" else []) + [sbatch_export(prov), str(script)] + tail)
            argv = argvs[0]
            rec.update(job_name=name, job_names=names, sbatch_argvs=argvs, partition=PARTITION)
        elif kind == "detr":
            # V3F's D6 (L23K): the smoke, or the three seed jobs, each its own sbatch of run_inc2_detr.sh
            sbatch = os.environ.get("INCAP_SBATCH", "sbatch")
            names = detr_job_names(req)
            argvs = []
            for n in names:
                tail = req["script_args"] + (["--seed", n.rsplit("_s", 1)[1]] if req["verb"] == "train" else [])
                argvs.append([sbatch, "--parsable", "--job-name=%s" % n, "-p", PARTITION, sbatch_export(prov),
                              str(script)] + tail)
            argv = argvs[0]
            rec.update(job_name=name, job_names=names, sbatch_argvs=argvs, partition=PARTITION)
        else:
            sbatch = os.environ.get("INCAP_SBATCH", "sbatch")
            argv = [sbatch, "--parsable", "--job-name=%s" % name, "-p", PARTITION, sbatch_export(prov),
                    str(script)] + req["script_args"]
            rec.update(job_name=name, sbatch_argv=argv, partition=PARTITION)
        if req["kind"] == "build" and child and (R.inc_dir() / child / "exp.json").exists():
            raise R.Refused("experiment %s is already built; an experiment is built once" % child)
        if dry_run:
            return rec
        q = R.squeue_jobs()
        if not q["ok"]:
            raise R.Refused("squeue unavailable (%s): cannot rule out a duplicate job, nothing submitted" % q["error"])
        dup = [j for j in q["jobs"] if j["name"] == name or (kind in ("zoo", "detr", "long")
                                                             and j["name"] in rec["job_names"])]
        if dup:
            raise R.Refused("%s is already queued or running as job %s" % (dup[0]["name"], dup[0]["id"]))
    except R.Refused as e:
        return R.fail(rec, e, error_kind="refused")
    if kind == "zoo":
        return _zoo_submit(rec, argv, prov)
    if kind in ("detr", "long"):
        return _detr_submit(rec, prov)
    (R.inc_dir() / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(provenance_env(prov))
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


def _detr_submit(rec, prov):
    """V3F's D6 (L23K): each sbatch of rec's sbatch_argvs in turn; the ids of all of them, or, when one fails, a
    failure after the ones already queued are cancelled (a qos message 'qos', else 'submit'); nothing is left
    half-submitted silently (an scancel that fails is reported in the failure)."""
    (R.inc_dir() / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(provenance_env(prov))
    ids = []

    def undo(why, kind_):
        left = []
        for j in ids:
            try:
                pr_ = subprocess.run([os.environ.get("INCAP_SCANCEL", "scancel"), j], capture_output=True, text=True,
                                     timeout=60)
                if pr_.returncode != 0:
                    left.append(j)
            except (OSError, subprocess.TimeoutExpired):
                left.append(j)
        msg = "%s%s" % (why, ("; cancelled %s" % ",".join(ids)) if ids and not left else
                        ("; could not cancel %s: a person cancels them" % ",".join(left) if left else ""))
        return R.fail(rec, msg, "SubmitFailed", kind_)
    for argv in rec["sbatch_argvs"]:
        try:
            pr = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            return undo("sbatch could not be run: %s: %s" % (type(e).__name__, str(e)[:200]), "submit")
        if pr.returncode != 0:
            msg = pr.stderr or pr.stdout
            return undo("sbatch exited %d: %s" % (pr.returncode, R._short(msg, 300)),
                        "qos" if QOS_RE.search(msg or "") else "submit")
        jid = ((pr.stdout or "").strip().splitlines() or [""])[-1].split(";")[0].strip()
        if not re.match(r"^[0-9]+$", jid):
            return undo("sbatch printed no job id: %r" % (pr.stdout or "")[-200:], "submit")
        ids.append(jid)
    rec.update(job_ids=ids, job_id=ids[0])
    return rec


def _zoo_submit(rec, argv, prov):
    """run_inc2_zoo.sh submit on the login node (it submits the chain and
    writes the record): exit 2 a refusal, a qos message 'qos', any other
    failure 'submit' (the script cancels what it queued); a timeout an
    unknown outcome ('uncertain': part of the chain may be queued); the last
    INCZOO line gives the five job ids."""
    env = dict(os.environ)
    env.update(provenance_env(prov))
    try:
        pr = subprocess.run(argv, capture_output=True, text=True, timeout=ZOO_SUBMIT_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired as e:
        return R.fail(rec, "run_inc2_zoo.sh submit timed out after %d s: part of the chain may be queued; it is "
                           "followed by its job names (%s)" % (ZOO_SUBMIT_TIMEOUT_S, str(e)[:200]),
                      "SubmitTimeout", "uncertain")
    except OSError as e:
        return R.fail(rec, "run_inc2_zoo.sh could not be run: %s" % e, type(e).__name__, "submit")
    msg = (pr.stderr or "") + (pr.stdout or "")
    rec.update(submit_rc=pr.returncode, submit_stderr=R._short(pr.stderr, 400))
    if pr.returncode == 2:
        return R.fail(rec, "run_inc2_zoo.sh submit refused: %s" % R._short(msg[-600:], 300), "SubmitRefused",
                      "refused")
    if QOS_RE.search(msg):
        return R.fail(rec, "sbatch refused on qos: %s" % R._short(msg[-600:], 300), "SubmitFailed", "qos")
    if pr.returncode != 0:
        return R.fail(rec, "run_inc2_zoo.sh submit exited %d: %s" % (pr.returncode, R._short(msg[-600:], 300)),
                      "SubmitFailed", "submit")
    line = next((ln for ln in reversed((pr.stdout or "").splitlines()) if ln.startswith("INCZOO ")), None)
    try:
        got = json.loads(line[len("INCZOO "):]) if line else None
        ids = [str(j) for j in got["job_ids"]]
    except (ValueError, TypeError, KeyError):
        ids = []
    if len(ids) != len(ZOO_JOB_NAMES) or not all(re.match(r"^[0-9]+$", j) for j in ids):
        return R.fail(rec, "run_inc2_zoo.sh submit printed no job ids", "SubmitFailed", "submit")
    rec.update(job_ids=ids, job_id=ids[0], job_names=list(got.get("names") or rec.get("job_names") or []))
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
