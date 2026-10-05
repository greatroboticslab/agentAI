"""The model-zoo audit: every detector the project trained, scored under one
protocol (docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-04): Z1, the model-zoo
audit (pre-registered usage)").

    python -m weed_optimizer_framework.tools.inc2.zoo VERB --version v1 [...]

Verbs (login node: preflight, submit, status, eval-names; batch jobs through
run_inc2_zoo.sh: inventory, exam-root, score, select, report; anywhere:
claims, codever; the inventory's steps one at a time: list, meta,
provenance, convert, exams, contamination, pilot, plan).

Why. The project trained about 1,000 detectors from 2026-03 to 2026-10 and
their published numbers are not comparable: four evaluators, the cwd12 test
used for checkpoint selection until 09-26, test copies in training sets, and
species names wrong before 09-21. Z1 writes one table: every detector, its
date, method, data, recipe and code version, scored by the locked scorer
(inc/scorer.py, pinned, used as a library: S.score() and
scorer_agnostic.capture() run every one of their checks) and flagged for
what its training could have seen. The table is descriptive (USAGE_RULE).

Layout. Everything lives under INC_DIR/_zoo/<version>/ (the leading
underscore keeps it out of every experiment-name glob). Its exam root
(root/) is a separate INC tree with its own LOCK.json, whose scorer_sha256 is
inc/scorer.py's and whose dev, test and imageweeds manifests are byte copies
of LOCK v1's; the scorer runs with INC_DIR pointing there and reads only that
tree. Score records live under scores/ (format inc2-zoo-score/1, production
false, stamped ZOO-<scorer sha256>), never in a run directory. The platform
reads INC_DIR/capacity/zoo_<version>.json (status, job ids, counts and
sha256s; no exam name as a key, no score path, no metric).

What the zoo never does: score an INC row on anything but dev (its other
columns are the records its own amendments wrote, reused only when every
stamp matches); write under INC_DIR outside _zoo/<version>/ and
capacity/zoo_<version>.json; edit a pinned module.

Imports at load: the standard library and inc.common (stdlib, cwd12_species,
near_dup). numpy, torch, ultralytics, PIL, yaml, inc.scorer, inc.splits,
inc2.base3, inc2.guard and inc2.scorer_agnostic are imported where used;
preflight, submit, status and eval-names never import torch.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gc
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..cwd12_species import (CWD12_LEGACY_LABELS, CWD12_SPECIES, TRAINER_SLOT_LEGACY, TRAINER_SLOT_SPECIES,
                             name_key, species_of)
from ..inc import common as C

CONFIG_FORMAT = "inc2-zoo-config/1"
FILE_FORMAT = "inc2-zoo-file/1"
INVENTORY_FORMAT = "inc2-zoo-inventory/1"
META_FORMAT = "inc2-zoo-meta/1"
MODEL_FORMAT = "inc2-zoo-model/1"
CONVERSION_FORMAT = "inc2-zoo-conversion/1"
PROVENANCE_FORMAT = "inc2-zoo-provenance/1"
CONTAMINATION_FORMAT = "inc2-zoo-contamination/1"
LOCK_FORMAT = "inc2-zoo-lock/1"
EXAMS_FORMAT = "inc2-zoo-exams/1"
PILOT_FORMAT = "inc2-zoo-pilot/1"
PLAN_FORMAT = "inc2-zoo-plan/1"
SHARD_FORMAT = "inc2-zoo-shard/1"
SCORE_FORMAT = "inc2-zoo-score/1"
REFUSAL_FORMAT = "inc2-zoo-refusal/1"
ERROR_FORMAT = "inc2-zoo-error/1"
TASK_FORMAT = "inc2-zoo-task/1"
LEDGER_FORMAT = "inc2-zoo-ledger/1"
SHORTLIST_FORMAT = "inc2-zoo-shortlist/1"
REPORT_FORMAT = "inc2-zoo-report/1"
RECORD_FORMAT = "inc2-zoo-record/1"
STEP_FORMAT = "inc2-zoo-step/1"
CODEVER_FORMAT = "inc2-zoo-codever/1"
ROOT_MARKER = "ZOO_ROOT.json"

VERSION_RE = re.compile(r"v[0-9]{1,3}\Z")
CONFIG_DIR = Path(__file__).resolve().parent
# The five jobs of one chain (stream_remote.ZOO_JOB_NAMES restates them: a test checks they agree)
ZOO_JOB_NAMES = ("inc_zoo_%s_inventory", "inc_zoo_%s_score_a", "inc_zoo_%s_select", "inc_zoo_%s_score_c",
                 "inc_zoo_%s_report")
PINNED_ULTRALYTICS = "8.4.37"           # inc/scorer.py's pin (a test checks they agree)

# ------------------------------------------------------------------- exams
COPIED_EXAMS = ("dev", "test", "imageweeds")      # byte copies of LOCK v1's manifests
EXAM_TV1 = "test_v1"                                # base v3's held-out capture groups (splits/v3/test_v1)
EXAM_EVG = "evalgroups_v1"                          # a sample of the five test groups (base3_v2.json)
EXAM_OOD = "ooddev_v1"                              # a sample of the two OOD-dev groups
ALL_EXAMS = ("dev", "imageweeds", "test", EXAM_TV1, EXAM_EVG, EXAM_OOD)
AGNOSTIC_ONLY = (EXAM_TV1, EXAM_EVG, EXAM_OOD)      # weed boxes only (class 12): agnostic columns only
SEALED_EXAMS = (EXAM_TV1, EXAM_EVG)                 # per-row values only in report_external.*
EXAM_ROLE = {"dev": "decision", "imageweeds": "external", EXAM_OOD: "external", "test": "descriptive",
             EXAM_TV1: "sealed", EXAM_EVG: "sealed"}
REUSE_EXAMS = ("dev", "imageweeds", "test")         # an INC run's own records the zoo may reuse
INC_ZOO_EXAMS = ("dev",)                            # the only exam the zoo itself scores for an INC row
HASH_EXAMS = ("dev", "test", "imageweeds", EXAM_TV1, EXAM_EVG, EXAM_OOD)

# ------------------------------------------------------------- decisions
# list-level reasons in the order they are tried (the first that applies is recorded), then the
# reasons a checkpoint itself gives (meta)
LIST_SKIPS = ("not_checkpoint", "gone", "changed_since_list", "epoch_snapshot", "third_party", "test_fixture",
              "inc_not_done", "inc_testing", "inc_final_link", "inc_train_weights", "mlflow_inc_best",
              "inc_superseded_attempt", "lora_unmerged", "sha256_duplicate")
META_SKIPS = ("classifier", "segmentation", "other_task", "stock_weights", "stock_coco")
SKIP_REASONS = LIST_SKIPS + META_SKIPS
UNSCORABLE_REASONS = ("unscorable_load", "unscorable_foreign", "unscorable_head", "unscorable_fidelity")
# test_selected values that mean chosen on the cwd12 test (S); selected_on values whose val list decides
TEST_CHOSEN = ("best", "best_partial", "early_stop", "early_stop_partial")
VAL_LISTED = ("own_split", "unknown")
RULES = ("R0", "R1", "R2", "R3", "R4", "R5")
RATINGS = ("exact", "listed_exact", "listed_after_rebuild", "derived_superset", "inherits", "none")
STATE_RANK = {"N": 0, "U": 1, "P": 2, "Y": 3}
SIZE_CLASSES = ("S", "M", "L")
SPECIES_NAMES_FROM = "2026-09-21"       # v3.60.0: model.names name species from this date on
PRE_INC_DATE = "2026-09-27"             # the INC protocol's first run; earlier rows used random cwd12 splits
RECIPE_KEYS = ("model", "data", "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
               "warmup_epochs", "cos_lr", "close_mosaic", "freeze", "seed", "patience", "time", "single_cls",
               "fraction", "pretrained", "mosaic", "mixup", "copy_paste")
FALLBACK_IMG_FORMATS = ("avif", "bmp", "dng", "heic", "heif", "jp2", "jpeg", "jpeg2000", "jpg", "mpo", "png", "tif",
                        "tiff", "webp")
COCO_FIRST, COCO_LAST = "person", "toothbrush"

USAGE_RULE = (
    "1. The zoo table is descriptive. No checkpoint is adopted, called \"best\", ranked as a result, or quoted as "
    "a model's accuracy from its cwd12 test, test v1 or evaluation-group number. RESEARCH_LOG, CHANGELOG, README "
    "and posters may cite the zoo only as \"the historical record under one protocol\", with its flags. "
    "2. The only ranking is by dev, among dev-clean rows (dev state N, not selected on dev, not dev-gated), "
    "within 12-species, partial-species and agnostic-only rows separately. External exams are shown beside it "
    "and never re-order it. Rows that are not dev-clean are listed by family and date, unranked. "
    "3. A checkpoint the project wants to use (an init, an incumbent, a deployable or robot model) re-qualifies "
    "under a dev rule pre-registered in its own amendment, as E1 and E2 did. A row that is not dev-clean cannot "
    "re-qualify on dev. "
    "4. The zoo's test v1 and evaluation-group reads are sealed descriptive reads of historical checkpoints "
    "(report_external.*): no recipe, data or model choice may cite them, they are no model's pre-registered test "
    "read, and no zoo checkpoint becomes a candidate through them; test v1 and the five test groups stay the "
    "frozen test of every model trained after Z1. "
    "5. A retracted claim is not reinstated by its zoo row. The row shows the claim's citation.")
FOOTER = ("No row is adopted, ranked as a result, or quoted from its cwd12 test, test v1 or evaluation-group "
          "number (Amendment 2026-10-04, Z1).")


class ZooRefused(RuntimeError):
    """What was asked is not what Z1 pre-registered, or an input is not what it must be (exit 2)."""


def log(msg):
    # stderr: a verb's stdout carries only what a caller parses (exam-root's path, submit's INCZOO line)
    print("[inc2.zoo] %s" % msg, file=sys.stderr, flush=True)


def warn(msg):
    print("[inc2.zoo] WARNING: %s" % msg, file=sys.stderr, flush=True)


def testing():
    return os.environ.get("INC_SCORER_TESTING") == "1"


def _utc(t=None):
    t = time.time() if t is None else t
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _secs(stamp):
    try:
        return datetime.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ------------------------------------------------------------------- paths
def source_inc():
    """The INC tree the zoo reads and writes under (INC_ZOO_SOURCE when the
    scorer runs with INC_DIR pointing at the exam root, else INC_DIR)."""
    return Path(os.environ.get("INC_ZOO_SOURCE") or str(C.INC_DIR))


def zoo_dir(version):
    return source_inc() / "_zoo" / version


def root_dir(version):
    return zoo_dir(version) / "root"


def record_path(version):
    return source_inc() / "capacity" / ("zoo_%s.json" % version)


def config_path(version):
    over = os.environ.get("INC_ZOO_CONFIG")
    if over:
        if not testing():
            raise ZooRefused("INC_ZOO_CONFIG replaces the committed config only in tests (INC_SCORER_TESTING=1)")
        return Path(over)
    return CONFIG_DIR / ("zoo_%s.json" % version)


# --------------------------------------------------------------------- io
def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def _dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n"


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_jsonl(path):
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except OSError:
        return None
    return rows


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return _sha_bytes(data)


def _write_json(path, obj):
    return _atomic_bytes(path, _dumps(obj).encode("utf-8"))


def _jsonl_bytes(rows):
    return "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows).encode("utf-8")


def _write_jsonl(path, rows):
    return _atomic_bytes(path, _jsonl_bytes(rows))


def _once_bytes(path, data):
    """Write `data` to path once (tmp + os.link): 'written', or 'kept' when the
    file holds these bytes already; another file there refuses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.once" % (path.name, os.getpid()))
    with open(tmp, "wb") as fh:
        fh.write(data)
    try:
        os.link(tmp, path)
        return "written"
    except FileExistsError:
        if _sha_file(path) == _sha_bytes(data):
            return "kept"
        raise ZooRefused("%s exists with other bytes: written once" % path)
    finally:
        tmp.unlink()


def _once_json(path, obj):
    return _once_bytes(path, _dumps(obj).encode("utf-8"))


def _link_once(path, obj):
    """A record written once whatever its bytes: 'written', or 'exists' when a
    record is already there (score records: their timings differ per pass)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.once" % (path.name, os.getpid()))
    with open(tmp, "w") as fh:
        fh.write(_dumps(obj))
    try:
        os.link(tmp, path)
        return "written"
    except FileExistsError:
        return "exists"
    finally:
        tmp.unlink()


# ------------------------------------------------------------------- config
def load_config(version):
    """(config, sha256) of the committed config of `version`; refuses a
    malformed one, or one of another version."""
    if not VERSION_RE.match(str(version)):
        raise ZooRefused("--version %r is not a zoo version (v<N>)" % (version,))
    path = config_path(version)
    try:
        raw = path.read_bytes()
        conf = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ZooRefused("cannot read the zoo config %s: %s" % (path, e))
    if conf.get("format") != CONFIG_FORMAT:
        raise ZooRefused("%s is not a %s config" % (path, CONFIG_FORMAT))
    if conf.get("version") != version:
        raise ZooRefused("--version %s, but %s is the config of %s" % (version, path, conf.get("version")))
    for k in ("inventory", "families", "class_map", "conversion", "exams", "contamination", "stages", "shortlist",
              "report", "stock_re"):
        if k not in conf:
            raise ZooRefused("%s has no %r block" % (path, k))
    for f in conf["families"]:
        re.compile(f["re"])
    if conf["families"][-1]["re"] != ".*":
        raise ZooRefused("%s: the last family must take every remaining path (.*)" % path)
    return conf, _sha_bytes(raw)


def family_of(rel, conf):
    for f in conf["families"]:
        if re.search(f["re"], rel):
            return f["name"]
    return "other"


def family_method(conf, name):
    return next((f.get("method") for f in conf["families"] if f["name"] == name), None)


def family_order(conf):
    return {f["name"]: i for i, f in enumerate(conf["families"])}


def is_inc_family(name):
    return str(name or "").startswith("inc_")


def step_marker(version, step):
    return zoo_dir(version) / "steps" / ("%s.json" % step)


def step_done(version, step, conf_sha):
    m = _read_json(step_marker(version, step))
    return isinstance(m, dict) and m.get("format") == STEP_FORMAT and m.get("config_sha256") == conf_sha \
        and m.get("status") == "done"


def mark_step(version, step, conf_sha, **extra):
    rec = {"format": STEP_FORMAT, "step": step, "status": "done", "config_sha256": conf_sha, "utc": _utc()}
    rec.update(extra)
    return _write_json(step_marker(version, step), rec)


def _code_sha():
    from ..inc import scorer as S
    return {"zoo": _sha_file(Path(__file__).resolve()), "scorer": _sha_file(Path(S.__file__).resolve())}


# ------------------------------------------------------------- the record
_T0 = time.time()
RECORD_KEYS = ("status", "version", "jobs", "job_names", "submitted_utc", "updated_utc", "decided_by", "approval_id",
               "trigger", "counts", "shards", "gpu_hours", "sha256", "resumed_from", "refusal")
LIVE_STATUSES = ("submitting", "submitted", "inventory_running", "plan_ready", "selected")
FINAL_STATUSES = ("complete", "partial", "refused", "failed")


def read_record(version):
    r = _read_json(record_path(version))
    return r if isinstance(r, dict) and r.get("format") == RECORD_FORMAT else None


def update_record(version, **fields):
    """Merge `fields` into capacity/zoo_<version>.json (counts and hashes only:
    never an exam name as a key, never a score path)."""
    bad = sorted(set(fields) - set(RECORD_KEYS))
    if bad:
        raise ValueError("not record fields: %s" % bad)
    rec = read_record(version) or {"format": RECORD_FORMAT, "version": version}
    rec.update({k: v for k, v in fields.items()})
    rec["updated_utc"] = _utc()
    _write_json(record_path(version), rec)
    return rec


# ------------------------------------------------------------- the ledger
def _job_ident():
    """(job, task) of this process: the array's job id and task index, the
    job id, or local<pid> outside Slurm."""
    aj, at = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if aj and at is not None:
        return str(aj), str(at)
    return str(os.environ.get("SLURM_JOB_ID") or "local%d" % os.getpid()), "-"


def _job_start():
    """When this job's GPU was allocated: SLURM_JOB_START_TIME (seconds since
    the epoch) when Slurm sets it, else this process's start."""
    v = os.environ.get("SLURM_JOB_START_TIME")
    try:
        return float(v) if v else _T0
    except ValueError:
        return _T0


def ledger_touch(version, kind, ended=False, **extra):
    """Write this process's ledger entry (ZOO/ledger/<kind>__<job>_<task>.json):
    started (the job's start), updated (now), ended. Every attempt of every
    stage has its own entry, so a resubmission adds to what was spent."""
    job, task = _job_ident()
    path = zoo_dir(version) / "ledger" / ("%s__%s_%s.json" % (kind, job, task))
    prev = _read_json(path) or {}
    rec = {"format": LEDGER_FORMAT, "kind": kind, "job": job, "task": task,
           "started_s": prev.get("started_s") or _job_start(), "updated_s": time.time(),
           "ended_s": time.time() if ended else prev.get("ended_s")}
    if prev.get("extra"):
        rec["extra"] = prev["extra"]
    if extra:
        rec["extra"] = dict(rec.get("extra") or {}, **extra)
    _write_json(path, rec)
    return rec


def ledger_hours(version, now=None, own=None):
    """GPU-hours spent so far by every attempt of every job of the chain (the
    ledger's entries: (ended or updated) - started; this process's own entry
    counts up to `now`)."""
    now = time.time() if now is None else now
    d = zoo_dir(version) / "ledger"
    out = collections.Counter()
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        files = []
    me = "%s_%s" % _job_ident()
    for p in files:
        r = _read_json(p)
        if not isinstance(r, dict) or r.get("format") != LEDGER_FORMAT:
            continue
        first = next((r.get(k) for k in ("ended_s", "updated_s", "started_s") if _num(r.get(k))), None)
        start = r.get("started_s") if _num(r.get("started_s")) else first
        end = now if (own is not None and r.get("kind") == own and "%s_%s" % (r.get("job"), r.get("task")) == me
                      and not _num(r.get("ended_s"))) else first
        h = max(0.0, float(end or 0) - float(start if start is not None else end or 0)) / 3600.0
        out[str(r.get("kind"))] += h
        out["total"] += h
    return {k: round(v, 4) for k, v in out.items()}


# ------------------------------------------------------------------- list
_EPOCH_RE = re.compile(r"epoch[0-9]+\.pt\Z")


def parse_list_text(text):
    """[{line, size, mtime, rel}] of a `size mtime ./rel` list; a blank line is
    no file (counted), a malformed one is a row with `malformed`."""
    rows, blank = [], 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            blank += 1
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or not parts[0].isdigit() or not re.fullmatch(r"[0-9]+(\.[0-9]+)?", parts[1]):
            rows.append({"line": i, "rel": line, "size": None, "mtime": None, "malformed": True})
            continue
        rel = parts[2]
        while rel.startswith("./"):
            rel = rel[2:]
        rows.append({"line": i, "rel": rel, "size": int(parts[0]), "mtime": int(float(parts[1]))})
    return rows, blank


def ckpt_role(name):
    if name == "final.pt":
        return "final"
    if name == "last_merged.pt":
        return "last_merged"
    if name in ("best.pt", "last.pt"):
        return name[:-3]
    return "other"


def run_dir_rel(rel):
    d = os.path.dirname(rel)
    return os.path.dirname(d) if os.path.basename(d) == "weights" else d


def _inc_parts(rel, inc_rel):
    """(exp, run, rest) when rel lies in an INC run directory
    (<inc_rel>/<exp>/runs/<run>/<rest>), else None."""
    m = re.match(r"%s/([^/]+)/runs/([^/]+)/(.+)\Z" % re.escape(inc_rel.rstrip("/")), rel)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def _small_text(path, cap=4096):
    try:
        with open(path, "rb") as fh:
            return fh.read(cap).decode("utf-8", "replace").strip()
    except OSError:
        return None


def mlflow_meta(path):
    """{project, name, start_time_ms, run_dir} of an MLflow artifact copy
    (<run>/artifacts/weights/<file>: params/project, params/name, meta.yaml)."""
    p = Path(path)
    run = p.parent.parent.parent if p.parent.name == "weights" and p.parent.parent.name == "artifacts" else None
    if run is None:
        return {"project": None, "name": None, "start_time_ms": None, "run_dir": None}
    st = None
    meta = _small_text(run / "meta.yaml", 65536) or ""
    m = re.search(r"^start_time:\s*([0-9]+)", meta, re.M)
    if m:
        st = int(m.group(1))
    return {"project": _small_text(run / "params" / "project"), "name": _small_text(run / "params" / "name"),
            "start_time_ms": st, "run_dir": str(run)}


def _under(path, prefixes):
    p = os.path.normpath(str(path))
    return any(p == q or p.startswith(q.rstrip("/") + "/") for q in prefixes if q)


def _inc_prefixes(conf):
    src = source_inc()
    out = {os.path.normpath(str(src))}
    try:
        out.add(os.path.realpath(str(src)))
    except OSError:
        pass
    root = conf["inventory"]["list_root"]
    out.add(os.path.normpath(os.path.join(root, conf["inventory"]["inc_rel"])))
    return sorted(out)


def _hash_many(paths, procs=5):
    def one(p):
        try:
            return p, C.sha256_file(p)
        except OSError:
            return p, None
    with ThreadPoolExecutor(max_workers=max(1, procs)) as ex:
        return dict(ex.map(one, list(paths)))


def _canon_class(rec):
    """The duplicate precedence: an INC final.pt, then a run directory's file, then an MLflow copy."""
    if rec["ckpt_role"] == "final" and rec.get("inc") and not rec.get("mlflow"):
        return 0
    return 2 if rec.get("mlflow") else 1


def list_step(version, conf, conf_sha, procs=5):
    """files.jsonl and summary.json (module docstring, the amendment's selection
    rule): one decision per listed line and per INC glob hit; refuses unless the
    counts reconcile."""
    inv = conf["inventory"]
    lpath = Path(inv["list"])
    try:
        raw = lpath.read_bytes()
    except OSError as e:
        raise ZooRefused("cannot read the inventory list %s: %s" % (lpath, e))
    lsha = _sha_bytes(raw)
    if lsha != inv["list_sha256"]:
        raise ZooRefused("the inventory list %s hashes to %s, not the pinned %s" % (lpath, lsha[:12],
                                                                                   inv["list_sha256"][:12]))
    zd = zoo_dir(version)
    _once_bytes(zd / "inputs" / ("list_%s.txt" % lsha[:16]), raw)
    rows, blank = parse_list_text(raw.decode("utf-8", "surrogateescape"))
    root = inv["list_root"]
    inc_rel = inv["inc_rel"].rstrip("/")
    listed = {r["rel"] for r in rows}
    src = source_inc()
    glob_hits = []
    for p in sorted(src.glob(inv["inc_glob"])):
        rel = "%s/%s" % (inc_rel, p.relative_to(src).as_posix())
        if rel not in listed:
            glob_hits.append({"line": None, "rel": rel, "size": None, "mtime": None, "origin": "inc_glob"})
    files = []
    for r in rows + glob_hits:
        rel = r["rel"]
        path = os.path.join(root, rel)
        rec = {"format": FILE_FORMAT, "rel": rel, "path": path, "origin": r.get("origin") or "list",
               "line": r.get("line"), "size": r.get("size"), "mtime": r.get("mtime"),
               "family": family_of(rel, conf), "run_dir": run_dir_rel(rel), "ckpt_role": ckpt_role(os.path.basename(rel)),
               "decision": None, "reason": None, "detail": None, "sha256": None, "dup_of": None, "mlflow": None,
               "inc": None, "also_role": []}
        files.append(rec)
        if r.get("malformed") or not rel.endswith(".pt"):
            rec.update(decision="skip", reason="not_checkpoint", detail="malformed line" if r.get("malformed") else None)
            continue
        if not os.path.exists(path):
            rec.update(decision="skip", reason="gone")
            continue
        if rec["origin"] == "list":
            ok = False
            for fn in (os.lstat, os.stat):
                try:
                    st = fn(path)
                except OSError:
                    continue
                if st.st_size == r["size"] and int(st.st_mtime) == r["mtime"]:
                    ok = True
            if not ok:
                rec.update(decision="skip", reason="changed_since_list",
                           detail="size or mtime differs from the pinned list")
                continue
        else:
            st = os.stat(path)
            rec["size"], rec["mtime"] = st.st_size, int(st.st_mtime)
        base = os.path.basename(rel)
        if _EPOCH_RE.match(base):
            rec.update(decision="skip", reason="epoch_snapshot")
            continue
        if rec["family"] == "third_party":
            rec.update(decision="skip", reason="third_party")
            continue
        if rec["family"] == "mlflow":
            mm = mlflow_meta(path)
            rec["mlflow"] = mm
            proj = mm.get("project") or ""
            if proj and _under(proj, inv.get("fixture_prefixes") or ["/tmp", "/var"]):
                rec.update(decision="skip", reason="test_fixture", detail="MLflow params/project %s" % proj)
                continue
            if proj and _under(proj, _inc_prefixes(conf)):
                rec["inc_copy"] = True
                if base == "best.pt":
                    rec.update(decision="skip", reason="mlflow_inc_best",
                               detail="best.pt of an INC run's training: chosen on a 4-image subset of training")
                    continue
                rj = _read_json(os.path.join(proj, "run.json")) or {}
                rec["inc"] = {"run_dir": proj, "weights_sha256": rj.get("weights_sha256"),
                              "status": rj.get("status")}
            elif proj:
                srcf = os.path.join(proj, mm.get("name") or "", "weights", base)
                rec["mlflow"]["source"] = srcf
                rec["mlflow"]["source_exists"] = os.path.exists(srcf)
                prel = os.path.relpath(os.path.normpath(proj), root)
                if not prel.startswith(".."):
                    rec["family"] = family_of(prel + "/x", conf)
                    rec["mlflow"]["project_family"] = rec["family"]
        parts = _inc_parts(rel, inc_rel)
        if parts is not None and rec["family"] != "mlflow":
            exp, run, rest = parts
            rdir = os.path.join(root, inc_rel, exp, "runs", run)
            rj = _read_json(os.path.join(rdir, "run.json"))
            rec["inc"] = {"exp": exp, "run": run, "run_dir": rdir,
                          "status": (rj or {}).get("status"), "testing": (rj or {}).get("testing"),
                          "kind": ((rj or {}).get("spec") or {}).get("kind") or (rj or {}).get("kind"),
                          "weights_sha256": (rj or {}).get("weights_sha256")}
            if rest != "weights/final.pt":
                rec.update(decision="skip", reason="inc_train_weights",
                           detail="%s is not the run's weights/final.pt" % rest)
                continue
            if not isinstance(rj, dict) or rj.get("status") != "done":
                rec.update(decision="skip", reason="inc_not_done",
                           detail="run.json %s" % ("missing" if rj is None else "status %r" % rj.get("status")))
                continue
            if rj.get("testing"):
                rec.update(decision="skip", reason="inc_testing")
                continue
            if os.path.islink(path):
                rec.update(decision="skip", reason="inc_final_link",
                           detail="weights/final.pt links to %s" % os.readlink(path))
                continue
        if base == "last.pt" and os.path.exists(os.path.join(os.path.dirname(path), "last_merged.pt")):
            rec.update(decision="skip", reason="lora_unmerged", detail="a merged last_merged.pt lies beside it")
            continue
        rec["decision"] = "candidate"
    cands = [f for f in files if f["decision"] == "candidate"]
    shas = _hash_many([f["path"] for f in cands], procs)
    for f in cands:
        f["sha256"] = shas.get(f["path"])
        if f["sha256"] is None:
            f.update(decision="skip", reason="gone", detail="unreadable when hashed")
            continue
        inc = f.get("inc") or {}
        if f.get("inc_copy"):
            if inc.get("weights_sha256") != f["sha256"]:
                f.update(decision="skip", reason="inc_superseded_attempt",
                         detail="its sha256 is not the INC run's weights_sha256 (an earlier attempt's weights)")
            continue
        if inc and f["ckpt_role"] == "final" and inc.get("weights_sha256") != f["sha256"]:
            f.update(decision="unscorable", reason="unscorable_load", detail="weights differ from run.json")
    groups = collections.defaultdict(list)
    for f in cands:
        if f["decision"] == "candidate":
            groups[f["sha256"]].append(f)
    for sha, fs in groups.items():
        fs.sort(key=lambda f: (_canon_class(f), len(f["rel"]), f["rel"]))
        keep = fs[0]
        keep["decision"] = "keep"
        for f in fs[1:]:
            f.update(decision="skip", reason="sha256_duplicate", dup_of=keep["rel"])
            if f["run_dir"] == keep["run_dir"] and f["ckpt_role"] not in keep["also_role"] + [keep["ckpt_role"]]:
                keep["also_role"].append(f["ckpt_role"])
    for f in files:
        f.pop("inc_copy", None)
    counts = reconcile(files, len(rows), len(glob_hits))
    _write_jsonl(zd / "inventory" / "files.jsonl", files)
    summary = {"format": INVENTORY_FORMAT, "version": version, "list": str(lpath), "list_sha256": lsha,
               "list_root": root, "config_sha256": conf_sha, "files_listed": len(rows), "blank_lines": blank,
               "inc_glob_added": len(glob_hits), "counts": counts, "code": {"zoo": _sha_file(Path(__file__).resolve())},
               "created_utc": _utc()}
    _write_json(zd / "inventory" / "summary.json", summary)
    log("list: %d listed (+%d INC runs the glob found), %d kept, %d unscorable, %d skipped"
        % (len(rows), len(glob_hits), counts["by_decision"].get("keep", 0), counts["by_decision"].get("unscorable", 0),
           counts["by_decision"].get("skip", 0)))
    return summary


def reconcile(files, n_listed, n_glob, meta=None):
    """Counts by decision and reason over files.jsonl (joined with meta.jsonl's
    decisions when given); refuses unless every file has exactly one decision
    and the counts add up to what was listed."""
    meta = meta or {}
    by_dec, by_reason = collections.Counter(), collections.Counter()
    for f in files:
        d = meta.get(f["rel"], {}).get("decision") or f["decision"]
        rsn = meta.get(f["rel"], {}).get("reason") if f["rel"] in meta else f["reason"]
        if d not in ("keep", "skip", "unscorable"):
            raise ZooRefused("%s has no decision (%r): every listed file gets exactly one" % (f["rel"], d))
        if d != "keep" and rsn not in SKIP_REASONS + UNSCORABLE_REASONS:
            raise ZooRefused("%s: %s with reason %r, which is not a pre-registered reason" % (f["rel"], d, rsn))
        by_dec[d] += 1
        if d != "keep":
            by_reason[rsn] += 1
    total = sum(by_dec.values())
    if total != n_listed + n_glob or sum(by_reason.values()) != by_dec["skip"] + by_dec["unscorable"]:
        raise ZooRefused("the inventory does not reconcile: %d listed + %d INC glob hits != %d decisions (%s)"
                         % (n_listed, n_glob, total, dict(by_dec)))
    return {"by_decision": dict(by_dec), "skipped_by_reason": {r: by_reason.get(r, 0) for r in SKIP_REASONS},
            "unscorable_by_reason": {r: by_reason.get(r, 0) for r in UNSCORABLE_REASONS},
            "reconciled": {"listed": n_listed, "inc_glob_added": n_glob, "decisions": total}}


def read_files(version):
    rows = _read_jsonl(zoo_dir(version) / "inventory" / "files.jsonl")
    if rows is None:
        raise ZooRefused("no inventory/files.jsonl: run the list step first")
    return rows


# ------------------------------------------------------------- class maps
def _names_list(names):
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        except (TypeError, ValueError):
            return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in (names or [])]


def is_coco(names):
    names = _names_list(names)
    return len(names) == 80 and names[0] == COCO_FIRST and names[-1] == COCO_LAST


def is_legacy_date(date):
    """True when a checkpoint's date is before v3.60.0 (2026-09-21), or unknown:
    its model.names may be the legacy vocabulary, where 'Ragweed' is Sicklepod."""
    d = str(date or "")[:10]
    return not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", d) or d < SPECIES_NAMES_FROM


def name_role(name, cm):
    """'species:<S>', 'weed', 'drop' or None for one class name, by the pinned
    tables: species_of, then the weed keys, suffixes and raw prefixes and the
    weed names, then the drop keys. A name none of them knows is unresolved."""
    sp = species_of(name)
    if sp:
        return "species:%s" % sp
    k = name_key(name)
    raw = str(name).strip().lower()
    if k in cm["weed_keys"] or k in cm.get("weed_names", []) or any(k.endswith(s) for s in cm["weed_suffixes"]) \
            or any(raw.startswith(p) for p in cm.get("weed_prefixes_raw", [])):
        return "weed"
    if k in cm["drop_keys"]:
        return "drop"
    return None


def class_map(names, conf, legacy=True):
    """{rule, groups (13 lists of source channel ids), dropped, n_species_channels,
    reason} of a model's whole names list (the amendment's R0-R5). `legacy`: the
    checkpoint predates v3.60.0 (or its date is unknown), so a name of the legacy
    vocabulary resolves only through a whole legacy list (R1-R3)."""
    cm = conf["class_map"]
    names = _names_list(names)
    nc = len(names)
    groups = [[] for _ in range(C.NC)]
    dropped = []

    def done(rule, reason=None):
        return {"rule": rule, "groups": groups, "dropped": sorted(dropped),
                "n_species_channels": sum(1 for i in range(C.OTHER_PLANT) if groups[i]), "reason": reason}

    if names == C.CLASS_NAMES:
        for i in range(C.NC):
            groups[i] = [i]
        return done("R0")
    if names in (CWD12_LEGACY_LABELS, CWD12_SPECIES):
        for i in range(12):
            groups[i] = [i]
        return done("R1")
    aux = re.compile(cm["aux_re"])
    if nc >= 12 and names[:12] in (TRAINER_SLOT_LEGACY, TRAINER_SLOT_SPECIES) and all(aux.match(n) for n in names[12:]):
        for j in range(12):
            groups[CWD12_SPECIES.index(TRAINER_SLOT_SPECIES[j])].append(j)
        groups[C.OTHER_PLANT] = list(range(12, nc))
        return done("R2")
    if nc >= 8 and names[:8] in (TRAINER_SLOT_LEGACY[:8], TRAINER_SLOT_SPECIES[:8]) \
            and all(n not in CWD12_LEGACY_LABELS and species_of(n) is None for n in names[8:]):
        for j in range(8):
            groups[CWD12_SPECIES.index(TRAINER_SLOT_SPECIES[j])].append(j)
        groups[C.OTHER_PLANT] = list(range(8, nc))
        return done("R3")

    def r5(reason):
        for i in range(C.NC):
            groups[i] = []
        del dropped[:]
        groups[C.OTHER_PLANT] = list(range(nc))
        return done("R5", reason)

    if nc == 0:
        return done("R5", "no class names")
    if any(n in cm["legacy_only"] for n in names):
        return r5("partial legacy vocabulary (%s): a legacy-only label is never read as a species"
                  % ", ".join(sorted({n for n in names if n in cm["legacy_only"]})))
    if legacy and any(n in CWD12_LEGACY_LABELS for n in names):
        return r5("legacy vocabulary before %s (%s): read only as a whole legacy list"
                  % (SPECIES_NAMES_FROM, ", ".join(sorted({n for n in names if n in CWD12_LEGACY_LABELS}))))
    unresolved = []
    for i, n in enumerate(names):
        role = name_role(n, cm)
        if role is None:
            unresolved.append(n)
        elif role.startswith("species:"):
            groups[CWD12_SPECIES.index(role.split(":", 1)[1])].append(i)
        elif role == "weed":
            groups[C.OTHER_PLANT].append(i)
        else:
            dropped.append(i)
    if unresolved:
        return r5("unresolved names %s" % unresolved[:6])
    if not any(groups):
        return r5("every name is dropped")
    return done("R4")


def size_class(params, conf):
    s, m = conf["stages"]["size_classes_m"]
    p = float(params or 0) / 1e6
    return "S" if p <= s else ("M" if p <= m else "L")


# ------------------------------------------------------------------- meta
def _ul_offline():
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        import ultralytics.utils.checks as checks
        checks.AUTOINSTALL = False
    except Exception:  # noqa: BLE001 - no ultralytics here: the caller's import fails next, with its own message
        pass


def _foreign(model):
    return sorted({"%s.%s" % (type(m).__module__, type(m).__name__) for m in model.modules()
                   if not type(m).__module__.startswith(("torch.", "ultralytics."))})


def weights_digest(model, names):
    """sha256 over the class names and every state_dict tensor (key, dtype,
    shape, bytes) in key order: two files with the same digest hold the same
    detector, whatever else their pickles carry."""
    import torch
    h = hashlib.sha256(json.dumps(_names_list(names)).encode("utf-8"))
    for k, v in model.state_dict().items():
        t = v.detach().cpu().contiguous()
        h.update(("%s|%s|%s|" % (k, t.dtype, tuple(t.shape))).encode("utf-8"))
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_ckpt(path):
    """(ckpt, model) of an Ultralytics checkpoint (EMA when present), loaded
    without installing anything."""
    _ul_offline()
    from ultralytics.nn.tasks import torch_safe_load
    ckpt, _f = torch_safe_load(str(path))
    if not isinstance(ckpt, dict):
        raise ValueError("the checkpoint is a %s, not a dict" % type(ckpt).__name__)
    m = ckpt.get("ema") or ckpt.get("model")
    if m is None or not hasattr(m, "modules"):
        raise ValueError("no model in the checkpoint (keys %s)" % sorted(ckpt)[:8])
    return ckpt, m


def _train_args_subset(ta):
    ta = ta if isinstance(ta, dict) else {}
    return {k: (ta.get(k) if isinstance(ta.get(k), (str, int, float, bool, type(None), list)) else str(ta.get(k)))
            for k in RECIPE_KEYS if k in ta}


def meta_one(f, conf):
    """(meta row, model row or None) of one kept file."""
    row = {"rel": f["rel"], "model_id": f["sha256"], "decision": "keep", "reason": None, "detail": None}
    import torch
    try:
        ckpt, m = load_ckpt(f["path"])
    except Exception as e:  # noqa: BLE001 - recorded: the file cannot be loaded under this Ultralytics
        row.update(decision="unscorable", reason="unscorable_load", detail="%s: %s" % (type(e).__name__, str(e)[:300]))
        return row, None
    try:
        from ultralytics.nn.tasks import guess_model_task
        task = getattr(m, "task", None) or guess_model_task(m)
        names = _names_list(getattr(m, "names", None) or ckpt.get("names"))
        head = m.model[-1] if hasattr(m, "model") and len(m.model) else None
        if task == "classify":
            row.update(decision="skip", reason="classifier")
        elif task == "segment":
            row.update(decision="skip", reason="segmentation")
        elif task in ("pose", "obb"):
            row.update(decision="skip", reason="other_task", detail=task)
        elif is_coco(names):
            stock = re.match(conf["stock_re"], os.path.basename(f["rel"]))
            row.update(decision="skip", reason="stock_weights" if stock else "stock_coco",
                       detail="the 80 COCO classes")
        if row["decision"] != "keep":
            return row, None
        foreign = _foreign(m)
        htype = type(head).__name__ if head is not None else None
        if foreign:
            row.update(decision="unscorable", reason="unscorable_foreign", detail=", ".join(foreign)[:300])
        elif htype not in ("Detect", "v10Detect"):
            row.update(decision="unscorable", reason="unscorable_head", detail="head %s, task %s" % (htype, task))
        date = ckpt.get("date")
        legacy = is_legacy_date(date)
        cmap = class_map(names, conf, legacy=legacy)
        params = int(sum(p.numel() for p in m.parameters()))
        dtype = str(next(m.parameters()).dtype).replace("torch.", "")
        model = {"format": MODEL_FORMAT, "model_id": f["sha256"], "short_id": f["sha256"][:12], "rel": f["rel"],
                 "path": f["path"], "family": f["family"], "method": family_method(conf, f["family"]),
                 "run_dir": f["run_dir"], "ckpt_role": f["ckpt_role"], "also_role": f.get("also_role") or [],
                 "params": params, "size_class": size_class(params, conf), "task": task, "dtype": dtype,
                 "head": {"type": htype, "nc": getattr(head, "nc", None), "end2end": bool(getattr(head, "end2end", False)),
                          "reg_max": getattr(head, "reg_max", None)},
                 "names": names, "names_sha256": _sha_text(json.dumps(names)), "class_map": cmap,
                 "legacy_names": legacy and cmap["rule"] in ("R1", "R2", "R3"),
                 "scorable": row["decision"] == "keep", "unscorable_reason": row["reason"],
                 "ckpt": {"date": date, "version": ckpt.get("version"), "epoch": ckpt.get("epoch"),
                          "best_fitness": ckpt.get("best_fitness") if _num(ckpt.get("best_fitness")) else None,
                          "train_args": _train_args_subset(ckpt.get("train_args"))},
                 "weights_digest": weights_digest(m, names), "inc": f.get("inc"), "mlflow": f.get("mlflow"),
                 "origin": f.get("origin")}
        return row, model
    finally:
        del m
        ckpt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def meta_step(version, conf, conf_sha):
    """meta.jsonl (each kept file's decision after loading it), models.jsonl
    (every kept or unscorable row) and meta_summary.json (the reconciliation
    over files.jsonl joined with meta.jsonl)."""
    zd = zoo_dir(version)
    files = read_files(version)
    summ = _read_json(zd / "inventory" / "summary.json") or {}
    meta_rows, models = [], []
    keep = sorted((f for f in files if f["decision"] == "keep"), key=lambda f: f["rel"])
    t0 = time.time()
    for i, f in enumerate(keep):
        row, model = meta_one(f, conf)
        meta_rows.append(row)
        if model is not None:
            models.append(model)
        elif row["decision"] == "unscorable":
            models.append(_bare_model(f, conf, row["reason"], row["detail"]))
        if (i + 1) % 100 == 0:
            log("meta: %d / %d checkpoints read (%.0fs)" % (i + 1, len(keep), time.time() - t0))
    for f in files:
        if f["decision"] == "unscorable":
            models.append(_bare_model(f, conf, f["reason"], f["detail"]))
    by_rel = {m["rel"]: m for m in models}
    for f in files:
        if f["decision"] == "skip" and f["reason"] == "sha256_duplicate" and f.get("mlflow") and f.get("dup_of") in by_rel:
            by_rel[f["dup_of"]].setdefault("mlflow_copies", []).append(f["rel"])
    _same_weights(models, files)
    meta = {r["rel"]: r for r in meta_rows}
    counts = reconcile(files, summ.get("files_listed", 0), summ.get("inc_glob_added", 0), meta=meta)
    _write_jsonl(zd / "inventory" / "meta.jsonl", meta_rows)
    _write_jsonl(zd / "models.jsonl", sorted(models, key=lambda m: m["model_id"]))
    out = {"format": META_FORMAT, "version": version, "config_sha256": conf_sha, "counts": counts,
           "models": len(models), "scorable": sum(1 for m in models if m.get("scorable")),
           "rules": dict(collections.Counter((m.get("class_map") or {}).get("rule") for m in models
                                             if m.get("scorable"))), "created_utc": _utc()}
    _write_json(zd / "inventory" / "meta_summary.json", out)
    log("meta: %d models (%d scorable), rules %s" % (len(models), out["scorable"], out["rules"]))
    return out


def _bare_model(f, conf, reason, detail):
    return {"format": MODEL_FORMAT, "model_id": f["sha256"] or _sha_text(f["rel"]), "short_id": (f["sha256"] or
            _sha_text(f["rel"]))[:12], "rel": f["rel"], "path": f["path"], "family": f["family"],
            "method": family_method(conf, f["family"]), "run_dir": f["run_dir"], "ckpt_role": f["ckpt_role"],
            "also_role": f.get("also_role") or [], "params": None, "size_class": None, "task": None, "head": None,
            "names": None, "class_map": None, "scorable": False, "unscorable_reason": reason,
            "unscorable_detail": detail, "ckpt": {}, "weights_digest": None, "inc": f.get("inc"),
            "mlflow": f.get("mlflow"), "origin": f.get("origin")}


def _same_weights(models, files):
    """same_weights_as: the canonical model_id among scorable rows that hold the
    same weights (weights_digest), by the duplicate precedence."""
    cls = {f["rel"]: _canon_class(f) for f in files}
    by = collections.defaultdict(list)
    for m in models:
        if m.get("scorable") and m.get("weights_digest"):
            by[m["weights_digest"]].append(m)
    for ms in by.values():
        ms.sort(key=lambda m: (cls.get(m["rel"], 1), len(m["rel"]), m["rel"]))
        for m in ms[1:]:
            m["same_weights_as"] = ms[0]["model_id"]


def read_models(version):
    rows = _read_jsonl(zoo_dir(version) / "models.jsonl")
    if rows is None:
        raise ZooRefused("no models.jsonl: run the meta step first")
    return rows


# ------------------------------------------------------------- provenance
def img_formats():
    try:
        from ultralytics.data.utils import IMG_FORMATS
        return frozenset(IMG_FORMATS)
    except Exception:  # noqa: BLE001 - no ultralytics: the release's list as of 8.4.37
        return frozenset(FALLBACK_IMG_FORMATS)


class Lister:
    """A dataset's training images as Ultralytics lists them (get_img_files):
    a directory by a recursive glob that follows directory symlinks and skips
    hidden names (here with a visited set, so a symlink loop ends), a .txt
    file by its lines ('./' replaced by the file's directory; another relative
    line is resolved against the training process's working directory, which
    is unknown: counted unresolved). A .txt list or a directory that cannot
    be read is counted unreadable: what it held is unknown, never empty. Each
    entry's realpath (readlink plus the cached realpath of its directory: one
    lstat per link instead of one per path component) and its own lstat mtime
    (a merge rebuilt after training shows as entries newer than the run)."""

    def __init__(self, fmts=None):
        self.fmts = fmts or img_formats()
        self._dirs = {}

    def _rdir(self, d):
        r = self._dirs.get(d)
        if r is None:
            r = os.path.realpath(d)
            self._dirs[d] = r
        return r

    def realpath(self, p):
        p = os.path.normpath(os.path.abspath(p))
        d, b = os.path.split(p)
        q = os.path.join(self._rdir(d), b)
        for _hop in range(16):
            try:
                st = os.lstat(q)
            except OSError:
                return q
            if not stat.S_ISLNK(st.st_mode):
                return q
            t = os.readlink(q)
            q = os.path.normpath(t if os.path.isabs(t) else os.path.join(os.path.dirname(q), t))
            d, b = os.path.split(q)
            q = os.path.join(self._rdir(d), b)
        return os.path.realpath(q)

    def _is_img(self, name):
        return "." in name and not name.startswith(".") and name.rpartition(".")[-1].lower() in self.fmts

    def entries(self, p):
        """({'entries': [(path, lstat mtime)], 'unresolved': n, 'unreadable': n,
        'missing': bool}): unresolved counts the relative lines of a .txt list,
        unreadable the .txt lists and directories that could not be read."""
        out, unresolved, unreadable = [], 0, 0
        if os.path.isdir(p):
            seen = set()
            stack = [p]
            while stack:
                d = stack.pop()
                try:
                    st = os.stat(d)
                except OSError:
                    unreadable += 1
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    with os.scandir(d) as it:
                        ents = sorted(it, key=lambda e: e.name)
                except OSError:
                    unreadable += 1
                    continue
                subdirs = []
                for e in ents:
                    if e.name.startswith("."):
                        continue
                    try:
                        isdir = e.is_dir()           # follows a symlink, as glob's ** does
                    except OSError:
                        isdir = False
                    if isdir:
                        subdirs.append(e.path)
                    elif self._is_img(e.name):
                        try:
                            mt = e.stat(follow_symlinks=False).st_mtime
                        except OSError:
                            mt = None
                        out.append((e.path, mt))
                stack.extend(reversed(subdirs))
            return {"entries": out, "unresolved": 0, "unreadable": unreadable, "missing": False}
        if os.path.isfile(p):
            parent = os.path.dirname(p) + os.sep
            try:
                with open(p, encoding="utf-8") as fh:
                    lines = fh.read().strip().splitlines()
            except (OSError, UnicodeDecodeError):
                return {"entries": [], "unresolved": 0, "unreadable": 1, "missing": False}
            for x in lines:
                x = x.strip()
                if not x:
                    continue
                if x.startswith("./"):
                    x = x.replace("./", parent, 1)
                if x.rpartition(".")[-1].lower() not in self.fmts:
                    continue
                if not os.path.isabs(x):
                    unresolved += 1
                    continue
                try:
                    mt = os.lstat(x).st_mtime
                except OSError:
                    mt = None
                out.append((x, mt))
            return {"entries": out, "unresolved": unresolved, "unreadable": 0, "missing": False}
        return {"entries": [], "unresolved": 0, "unreadable": 0, "missing": True}


def dataset_dirs(yaml_path, key):
    """(the train or val entries of a data.yaml, resolved as Ultralytics'
    check_det_dataset resolves them, or None, why). A relative `path:` is
    resolved against the training process's working directory: unknown here."""
    import yaml
    try:
        with open(yaml_path) as fh:
            d = yaml.safe_load(fh) or {}
    except Exception as e:  # noqa: BLE001 - an unreadable yaml lists nothing, recorded
        return None, "the data yaml cannot be read (%s)" % str(e)[:200]
    if not isinstance(d, dict):
        return None, "the data yaml is not a mapping"
    base = d.get("path")
    if base in (None, ""):
        base = os.path.dirname(os.path.abspath(yaml_path))
    elif not os.path.isabs(str(base)):
        return None, "relative dataset path %r: resolved against the training process's working directory" % base
    vals = d.get(key)
    if key == "val" and key not in d:
        vals = d.get("validation")          # check_det_dataset renames a 'validation' key to 'val'
    if vals in (None, ""):
        return [], None
    vals = [vals] if isinstance(vals, str) else [str(v) for v in vals]
    out = []
    for v in vals:
        x = os.path.normpath(os.path.join(str(base), v))
        if not os.path.exists(x) and v.startswith("../"):
            x = os.path.normpath(os.path.join(str(base), v[3:]))
        out.append(os.path.realpath(x) if os.path.exists(x) else x)
    return out, None


def _cluster_utc(date, tz):
    """A checkpoint's local date (Ultralytics' datetime.now().isoformat() on the
    cluster) in seconds since the epoch, or None."""
    try:
        dt = datetime.datetime.fromisoformat(str(date))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        try:
            from zoneinfo import ZoneInfo
            dt = dt.replace(tzinfo=ZoneInfo(tz))
        except Exception:  # noqa: BLE001 - no tz database: the date stays unanchored
            return None
    return dt.timestamp()


def _mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return None


def _results_csv(run_dir):
    """{rows, best_index, fitness} of a run's results.csv (fitness 0.1 mAP50 +
    0.9 mAP50-95, Ultralytics' detection fitness), or None when missing."""
    p = os.path.join(run_dir, "results.csv")
    try:
        with open(p) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines:
        return {"rows": 0, "best_index": None}
    head = [h.strip() for h in lines[0].split(",")]
    try:
        i50 = head.index("metrics/mAP50(B)")
        i95 = head.index("metrics/mAP50-95(B)")
    except ValueError:
        return {"rows": len(lines) - 1, "best_index": None}
    fit = []
    for ln in lines[1:]:
        t = ln.split(",")
        try:
            fit.append(0.1 * float(t[i50]) + 0.9 * float(t[i95]))
        except (IndexError, ValueError):
            fit.append(float("-inf"))
    best = max(range(len(fit)), key=lambda k: (fit[k], -k)) if fit else None
    return {"rows": len(fit), "best_index": best}


def _selected_on(val_dirs, train_dirs, base, cont):
    hay = " ".join(val_dirs or [])
    if any(s in hay for s in cont["cwd12_test_val"]):
        return "cwd12_test"
    if any(s in hay for s in cont["cwd12_test_part_val"]):
        return "cwd12_test_part"
    if any(s in hay for s in cont["dev_val"]):
        return "dev"
    # val entries are real paths (dataset_dirs): the dataset's root is compared as written and as its real path
    if val_dirs and (set(val_dirs) <= set(train_dirs or []) or
                     (base and all(_under(v, [base, os.path.realpath(base)]) for v in val_dirs))):
        return "own_split"
    return "unknown"


def write_list(version, paths):
    """The sorted unique realpaths of a training (or val) list, written once
    under lists/ by the sha256 of its text; returns that sha256."""
    text = "".join("%s\n" % p for p in sorted(set(paths)))
    sha = _sha_text(text)
    p = zoo_dir(version) / "lists" / sha[:2] / ("%s.txt" % sha)
    if not p.is_file():
        _once_bytes(p, text.encode("utf-8"))
    return sha


def read_list(version, sha):
    p = zoo_dir(version) / "lists" / sha[:2] / ("%s.txt" % sha)
    try:
        return [ln for ln in p.read_text().splitlines() if ln]
    except OSError:
        return None


def _stock_name(path, conf):
    return bool(path) and bool(re.match(conf["stock_re"], os.path.basename(str(path))))


class Index:
    """The inventory's files by path and realpath, and INC run directories by
    the weights they record, for init and soup resolution and score reuse."""

    def __init__(self, files, models):
        self.by_path, self.by_real = {}, {}
        rel_sha = {f["rel"]: f.get("sha256") for f in files}
        self.model_ids = {m["model_id"] for m in models}
        self.family = {m["model_id"]: m.get("family") for m in models}
        for f in files:
            sha = f.get("sha256") if f["decision"] in ("keep", "unscorable") else \
                (rel_sha.get(f.get("dup_of")) if f.get("reason") == "sha256_duplicate" else None)
            for k in (os.path.normpath(f["path"]),):
                self.by_path[k] = sha
            try:
                real = os.path.realpath(f["path"])
            except OSError:
                continue
            # a skipped link (a kind-final link) never hides the row its target is
            if sha or real not in self.by_real:
                self.by_real[real] = sha
        self.inc_runs = collections.defaultdict(list)
        for f in files:
            inc = f.get("inc") or {}
            if inc.get("run_dir") and inc.get("weights_sha256") and not f.get("mlflow"):
                if inc["run_dir"] not in self.inc_runs[inc["weights_sha256"]]:
                    self.inc_runs[inc["weights_sha256"]].append(inc["run_dir"])

    def model_of(self, path=None, sha=None):
        if sha and sha in self.model_ids:
            return sha
        if not path:
            return None
        for key, tab in ((os.path.normpath(str(path)), self.by_path), (os.path.realpath(str(path)), self.by_real)):
            got = tab.get(key)
            if got and got in self.model_ids:
                return got
        return None


def _resolve_init(path, sha, idx, conf):
    if path in (None, "", True, False):
        return {"kind": "unknown", "ref": path, "model_id": None}
    sp = str(path)
    mid = idx.model_of(sp, sha)
    if mid:
        return {"kind": "row", "ref": sp, "model_id": mid}
    if sp.endswith((".yaml", ".yml")):
        return {"kind": "scratch", "ref": sp, "model_id": None}
    if _stock_name(sp, conf):
        return {"kind": "stock", "ref": sp, "model_id": None}
    return {"kind": "unknown", "ref": sp, "model_id": None}


def _inc_parent(refs, idx):
    """Whether an init or a soup member resolves to an INC row (its weights
    were chosen by an INC dev gate or verdict)."""
    return any((x or {}).get("kind") == "row" and is_inc_family(idx.family.get((x or {}).get("model_id")))
               for x in refs or ())


def _dev_gated(m, rj, init, conf, soup=(), idx=None):
    """The amendment's dev_gated: a row whose data or init a dev gate chose.
    Any row (of any family) initialised from an INC row, or a soup of one; a
    stream pool after an accepted increment (a manifest that is not P_0); a
    pilot, real-loop or other INC candidate or soup, or one initialised from
    another row. Inheritance through the init chain is the contamination
    step's."""
    if idx is not None and _inc_parent([init] + list(soup or ()), idx):
        return True
    fam = m["family"]
    kind = ((rj.get("spec") or {}).get("kind") or rj.get("kind")) if isinstance(rj, dict) else None
    if fam == "inc_stream":
        man = os.path.basename(str((rj or {}).get("train_manifest") or ""))
        return not man.startswith("P_0.")
    if fam in ("inc_realloop", "inc_pilot", "inc_other"):
        return kind in ("cand", "soup") or (init or {}).get("kind") == "row"
    return False


def provenance_one(m, idx, conf, lister, cache):
    cont = conf["contamination"]
    tz = conf["class_map"].get("cluster_tz", "America/New_York")
    out = {"format": PROVENANCE_FORMAT, "model_id": m["model_id"], "rel": m["rel"], "family": m["family"],
           "kind": "inc" if is_inc_family(m["family"]) and not m.get("mlflow") else "ultralytics",
           "dates": {}, "recipe": {}, "init": {"kind": "unknown", "ref": None, "model_id": None}, "soup_of": [],
           "data": {"source": "none", "rating": "none", "reason": None, "list_sha256": None, "n_entries": 0,
                    "n_unique": 0, "n_unresolved": 0, "n_unreadable": 0},
           "val": None, "selected_on": "unknown", "test_selected": "unknown", "chosen_on_val": None,
           "dev_selected": False, "dev_selected_unknown": False,
           "dev_gated": False, "species_trained": None, "code": {"kind": "unknown"}, "imgsz_trained": None,
           "notes": []}
    ck = m.get("ckpt") or {}
    ta = ck.get("train_args") or {}
    out["dates"]["end"] = ck.get("date")
    out["dates"]["end_tz"] = "cluster local" if ck.get("date") else None
    out["dates"]["end_source"] = "ckpt" if ck.get("date") else None
    if out["kind"] == "inc":
        inc = m.get("inc") or {}
        rdir = inc.get("run_dir") or os.path.dirname(os.path.dirname(m["path"]))
        rj = _read_json(os.path.join(rdir, "run.json")) or {}
        spec = rj.get("spec") if isinstance(rj.get("spec"), dict) else (_read_json(os.path.join(rdir, "spec.json")) or {})
        out["dates"].update(start_utc=rj.get("started_utc"), end=rj.get("finished_utc") or ck.get("date"),
                            end_tz="UTC" if rj.get("finished_utc") else out["dates"]["end_tz"],
                            end_source="run.json" if rj.get("finished_utc") else out["dates"]["end_source"],
                            start_source="run.json")
        out["recipe"] = {"recipe": spec.get("recipe"), "protocol_recipe": rj.get("protocol_recipe"),
                         "recipe_deviations": rj.get("recipe_deviations"), "recipe_name": rj.get("recipe_name"),
                         "arm": rj.get("arm"), "kind": spec.get("kind") or rj.get("kind")}
        out["imgsz_trained"] = ((spec.get("recipe") or {}).get("imgsz") if isinstance(spec.get("recipe"), dict)
                                else None) or ta.get("imgsz")
        if spec.get("kind") == "soup" or rj.get("soup_of"):
            mem = rj.get("soup_of") or spec.get("soup_of") or []
            shas = rj.get("soup_sha256") or [None] * len(mem)
            out["soup_of"] = [_resolve_init(p, s, idx, conf) for p, s in zip(mem, shas)]
            out["data"].update(source="none", rating="inherits", reason="a soup: the members' data")
        else:
            out["init"] = _resolve_init(rj.get("init") or spec.get("init"), rj.get("init_sha256"), idx, conf)
            man, want = rj.get("train_manifest") or spec.get("train_manifest"), rj.get("train_manifest_sha256")
            got = _sha_file(man) if man else None
            if not man:
                out["data"].update(rating="inherits", reason="no training manifest (kind %s)" % spec.get("kind"))
            elif got is None or got != want:
                out["data"].update(source="manifest", ref=man, rating="none",
                                   reason="the manifest is gone" if got is None else
                                   "the manifest no longer hashes as run.json records")
            else:
                key = ("manifest", want)
                if key not in cache:
                    rows = C.read_manifest(man)
                    paths = [lister.realpath(r["image"]) for r in rows]
                    cache[key] = {"sha": write_list(cache["version"], paths), "n_entries": len(rows),
                                  "n_unique": len(set(paths))}
                got = cache[key]
                out["data"].update(source="manifest", ref=man, rating="exact", list_sha256=got["sha"],
                                   n_entries=got["n_entries"], n_unique=got["n_unique"], manifest_sha256=want)
        cc = rj.get("train_class_counts")
        if isinstance(cc, dict):
            out["species_trained"] = any(int(cc.get(s) or 0) > 0 for s in CWD12_SPECIES)
        mods = ((rj.get("code") or {}).get("modules") or {}) if isinstance(rj.get("code"), dict) else {}
        out["code"] = {"kind": "exact" if mods else "unknown", "modules": mods,
                       "digest": _sha_text("".join("%s %s\n" % (k, mods[k]) for k in sorted(mods))) if mods else None,
                       "ultralytics": rj.get("ultralytics_version"), "torch": rj.get("torch_version")}
        out["selected_on"], out["test_selected"] = "train_subset", "none"
        out["dev_gated"] = _dev_gated(m, rj, out["init"], conf, out["soup_of"], idx)
        out["inc_kind"] = spec.get("kind") or rj.get("kind")
        out["inc_exp"], out["inc_run"] = inc.get("exp"), inc.get("run")
        out["research_only"] = rj.get("research_only")
        return out
    # an Ultralytics run directory (or an MLflow copy whose run directory is gone)
    root = conf["inventory"]["list_root"]
    rdir = os.path.join(root, m["run_dir"])
    if m.get("mlflow") and (m["mlflow"] or {}).get("project"):
        mm = m["mlflow"]
        cand = os.path.join(mm["project"], mm.get("name") or "")
        if os.path.isdir(cand):
            rdir = cand
    args_y = os.path.join(rdir, "args.yaml")
    a_mt = _mtime(args_y)
    if not ta:
        try:
            import yaml
            with open(args_y) as fh:
                ta = _train_args_subset(yaml.safe_load(fh) or {})
        except Exception:  # noqa: BLE001 - no args.yaml: the recipe is what the checkpoint holds (none)
            ta = {}
    out["recipe"] = dict(ta)
    out["imgsz_trained"] = ta.get("imgsz")
    if a_mt is not None:
        out["dates"].update(start_utc=_utc(a_mt), start_source="args.yaml mtime")
        start = a_mt
    elif (m.get("mlflow") or {}).get("start_time_ms"):
        start = m["mlflow"]["start_time_ms"] / 1000.0
        out["dates"].update(start_utc=_utc(start), start_source="MLflow start_time")
    else:
        start = None
    end_s = _cluster_utc(ck.get("date"), tz)
    init = _resolve_init(ta.get("model"), None, idx, conf)
    pre = ta.get("pretrained")
    if isinstance(pre, str) and pre not in ("True", "False", "true", "false"):
        pre_init = _resolve_init(pre, None, idx, conf)
        init = pre_init if pre_init["kind"] == "row" else {"kind": "unknown", "ref": pre, "model_id": None,
                                                           "why": "pretrained is a path the zoo cannot resolve"}
    out["init"] = init
    data = ta.get("data")
    fam = m["family"]
    derived = (cont.get("derived_lists") or {}).get(fam)
    train_dirs, val_dirs, why, base = None, None, None, None
    yaml_path = str(data) if data else None
    if yaml_path and not os.path.isabs(yaml_path):
        yaml_path = None
        why = "data %r is not a path (a dataset name or a relative yaml)" % data
    if yaml_path and os.path.isfile(yaml_path):
        train_dirs, why = dataset_dirs(yaml_path, "train")
        val_dirs, _w = dataset_dirs(yaml_path, "val")
        try:
            import yaml
            with open(yaml_path) as fh:
                bd = (yaml.safe_load(fh) or {}).get("path")
            base = str(bd) if bd and os.path.isabs(str(bd)) else os.path.dirname(os.path.abspath(yaml_path))
        except Exception:  # noqa: BLE001
            base = None
        out["data"].update(source="yaml", ref=yaml_path)
    elif derived:
        train_dirs = [os.path.join(root, x) for x in derived["superset_of"]]
        out["data"].update(source="derived", ref=derived.get("cite"))
        why = None
    else:
        out["data"].update(source="none", ref=yaml_path or data,
                           reason=why or ("the data yaml %s is gone" % yaml_path if yaml_path else "no data yaml"))
    if train_dirs is not None:
        if any(not os.path.exists(x) for x in train_dirs):
            out["data"].update(rating="none", reason="the training list is gone (%s)"
                               % next(x for x in train_dirs if not os.path.exists(x)))
        else:
            key = ("train", tuple(train_dirs))
            if key not in cache:
                ents, unres, unread = [], 0, 0
                for x in train_dirs:
                    got = lister.entries(x)
                    ents += got["entries"]
                    unres += got["unresolved"]
                    unread += got["unreadable"]
                reals = [lister.realpath(p) for p, _mt in ents]
                newest = max((mt for _p, mt in ents if mt is not None), default=None)
                # entries listed under the cottonweed_holdout slug's merge prefix (their realpath loses the name)
                pre = tuple(cont.get("cwd12_holdout_prefixes") or ())
                slug = {r for (p, _mt), r in zip(ents, reals) if pre and os.path.basename(p).startswith(pre)}
                cache[key] = {"sha": write_list(cache["version"], reals), "n_entries": len(ents),
                              "n_unique": len(set(reals)), "unresolved": unres, "unreadable": unread,
                              "newest": newest,
                              "holdout_slug": len(slug)}
            got = cache[key]
            if derived and out["data"]["source"] == "derived":
                rating = "derived_superset"
            else:
                y_mt = _mtime(yaml_path) if yaml_path else None
                rebuilt = []
                if start is None:
                    rebuilt.append("no start date")
                else:
                    if y_mt is not None and y_mt > start:
                        rebuilt.append("the data yaml is newer than the run")
                    if got["newest"] is not None and got["newest"] > start:
                        rebuilt.append("listed entries are newer than the run")
                if a_mt is not None and end_s is not None and a_mt > end_s + 120:
                    rebuilt.append("args.yaml is newer than the checkpoint (its start date is not the run's)")
                rating = "listed_after_rebuild" if rebuilt else "listed_exact"
                if rebuilt:
                    out["data"]["reason"] = "; ".join(rebuilt)
            out["data"].update(rating=rating, list_sha256=got["sha"], n_entries=got["n_entries"],
                               n_unique=got["n_unique"], n_unresolved=got["unresolved"],
                               n_unreadable=got["unreadable"],
                               holdout_slug=got.get("holdout_slug", 0),
                               newest_entry_utc=_utc(got["newest"]) if got["newest"] else None)
    if val_dirs:
        vkey = ("val", tuple(val_dirs))
        if vkey not in cache and all(os.path.exists(x) for x in val_dirs):
            ents, unres, unread = [], 0, 0
            for x in val_dirs:
                got = lister.entries(x)
                ents += got["entries"]
                unres += got["unresolved"]
                unread += got["unreadable"]
            reals = [lister.realpath(p) for p, _mt in ents]
            cache[vkey] = {"sha": write_list(cache["version"], reals), "n_unique": len(set(reals)),
                           "unresolved": unres, "unreadable": unread}
        if vkey in cache:
            out["val"] = {"list_sha256": cache[vkey]["sha"], "n_unique": cache[vkey]["n_unique"],
                          "n_unresolved": cache[vkey]["unresolved"], "n_unreadable": cache[vkey]["unreadable"],
                          "ref": val_dirs}
    sel = derived.get("val") if (derived and out["data"]["source"] == "derived") else \
        _selected_on(val_dirs, train_dirs, base, cont)
    out["selected_on"] = sel
    role = m["ckpt_role"]
    res = _results_csv(rdir)
    out["results"] = res
    test_sets = ("cwd12_test", "cwd12_test_part")
    # chosen on its val set: best.pt always; last.pt when its run stopped early (unknown without results.csv)
    chosen = "no"
    if role == "best":
        chosen = "yes"
    elif role in ("last", "last_merged"):
        ep, pat, tl = ta.get("epochs"), ta.get("patience"), ta.get("time")
        if res is None:
            chosen = "unknown"
        elif tl not in (None, 0, 0.0, "null") and res["rows"] < (ep or 0):
            out["notes"].append("time-limited run (time %s h): its last epoch is not an early stop" % tl)
        elif _num(ep) and _num(pat) and res["rows"] < ep and pat < ep \
                and res.get("best_index") is not None and res["rows"] - 1 - res["best_index"] >= pat:
            chosen = "yes"
    out["chosen_on_val"] = chosen
    out["dev_selected"] = sel == "dev" and chosen == "yes"
    out["dev_selected_unknown"] = sel == "dev" and chosen == "unknown"
    if chosen == "no":
        out["test_selected"] = "none"
    elif sel in test_sets:
        out["test_selected"] = ("best" if role == "best" else "early_stop") if chosen == "yes" else "unknown"
    elif sel in VAL_LISTED:
        # a val under the dataset's root, or one the zoo cannot place: its list decides (contamination_step)
        out["test_selected"] = "pending_val_check"
    else:
        out["test_selected"] = "none"
    out["code"] = {"kind": "approx", "date_start": out["dates"].get("start_utc"), "ultralytics": ck.get("version")}
    out["dev_gated"] = _dev_gated(m, None, init, conf, (), idx)
    return out


def provenance_step(version, conf, conf_sha):
    zd = zoo_dir(version)
    files, models = read_files(version), read_models(version)
    idx = Index(files, models)
    lister = Lister()
    cache = {"version": version}
    rows = []
    t0 = time.time()
    for i, m in enumerate(sorted(models, key=lambda m: m["model_id"])):
        try:
            rows.append(provenance_one(m, idx, conf, lister, cache))
        except Exception as e:  # noqa: BLE001 - one row's provenance failing is recorded, never the step's end
            rows.append({"format": PROVENANCE_FORMAT, "model_id": m["model_id"], "rel": m["rel"],
                         "family": m["family"], "kind": "error", "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                         "data": {"source": "none", "rating": "none", "reason": "provenance failed"},
                         "init": {"kind": "unknown"}, "soup_of": [], "selected_on": "unknown",
                         "test_selected": "unknown", "chosen_on_val": "unknown", "dev_selected": False,
                         "dev_selected_unknown": True, "dev_gated": False, "notes": []})
        if (i + 1) % 100 == 0:
            log("provenance: %d / %d rows (%.0fs)" % (i + 1, len(models), time.time() - t0))
    _write_jsonl(zd / "provenance.jsonl", rows)
    ratings = dict(collections.Counter(r["data"]["rating"] for r in rows))
    log("provenance: %d rows, ratings %s, %d distinct lists" % (len(rows), ratings,
                                                                len({r["data"].get("list_sha256") for r in rows} - {None})))
    return {"rows": len(rows), "ratings": ratings}


def read_provenance(version):
    rows = _read_jsonl(zoo_dir(version) / "provenance.jsonl")
    if rows is None:
        raise ZooRefused("no provenance.jsonl: run the provenance step first")
    return {r["model_id"]: r for r in rows}


# ---------------------------------------------------------------- convert
def _test_setting(name, default, cast=str):
    """A scoring or fidelity setting a test may change (INC_ZOO_<NAME>, only
    with INC_SCORER_TESTING=1); production refuses the variable."""
    v = os.environ.get("INC_ZOO_%s" % name.upper())
    if v in (None, ""):
        return default
    if not testing():
        raise ZooRefused("INC_ZOO_%s changes a protocol setting: tests only (INC_SCORER_TESTING=1)" % name.upper())
    return cast(v)


def _device():
    import torch
    dev = _test_setting("device", None)
    if dev:
        return dev
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return "cuda:0"
    return "cpu"


def convert_module(model, groups, fill_bias):
    """Rewrite a Detect / v10Detect head into the 13 INC channels in place:
    the last Conv2d of every cv3[i] (and one2one_cv3[i]) becomes Conv2d(c3, 13K)
    whose channel t*K+s copies source channel groups[t][s] (empty slots weight
    0, bias fill_bias), then Unflatten(1, (13, K)) -> MaxPool3d((K, 1, 1)) ->
    Flatten(2, 3): channel t is the maximum of its group's logits. Box
    branches are untouched."""
    import torch
    import torch.nn as nn
    head = model.model[-1]
    if type(head).__name__ not in ("Detect", "v10Detect"):
        raise ValueError("head %s is not Detect or v10Detect" % type(head).__name__)
    if len(groups) != C.NC:
        raise ValueError("%d groups, not %d" % (len(groups), C.NC))
    nc = int(head.nc)
    used = [s for g in groups for s in g]
    if any(not 0 <= s < nc for s in used) or len(used) != len(set(used)):
        raise ValueError("the class map's source channels are not distinct ids below nc=%d" % nc)
    K = max(1, max(len(g) for g in groups))
    branches = [head.cv3] + ([head.one2one_cv3] if getattr(head, "one2one_cv3", None) is not None else [])
    for br in branches:
        for i in range(len(br)):
            old = br[i][-1]
            if not isinstance(old, nn.Conv2d) or old.out_channels != nc or old.kernel_size != (1, 1):
                raise ValueError("cls branch %d ends in %s, not Conv2d(c3, %d, 1)" % (i, type(old).__name__, nc))
            w = old.weight
            new = nn.Conv2d(old.in_channels, C.NC * K, 1, bias=True).to(device=w.device, dtype=w.dtype)
            with torch.no_grad():
                new.weight.zero_()
                new.bias.fill_(float(fill_bias))
                for t, g in enumerate(groups):
                    for s, src in enumerate(g):
                        new.weight[t * K + s].copy_(w[src])
                        new.bias[t * K + s] = old.bias[src] if old.bias is not None else 0.0
            new.requires_grad_(False)
            br[i][-1] = new if K == 1 else nn.Sequential(new, nn.Unflatten(1, (C.NC, K)), nn.MaxPool3d((K, 1, 1)),
                                                          nn.Flatten(2, 3))
    head.nc = C.NC
    head.no = C.NC + 4 * int(head.reg_max)
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml["nc"] = C.NC
    if hasattr(model, "nc"):
        model.nc = C.NC
    model.names = {i: n for i, n in enumerate(C.CLASS_NAMES)}
    return K


def fidelity_batch(n, imgsz):
    """The first n dev images (key order) letterboxed to imgsz, as a float batch in [0, 1]."""
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    man = source_inc() / "splits" / "v1" / "dev.jsonl"
    rows = sorted(C.read_manifest(man), key=lambda r: r["key"])[:n]
    if len(rows) < n:
        raise ZooRefused("the dev manifest %s holds %d images, the fidelity check needs %d" % (man, len(rows), n))
    lb = LetterBox((imgsz, imgsz), auto=False)
    ims = []
    for r in rows:
        im = cv2.imread(r["image"])
        if im is None:
            raise ZooRefused("cannot read the dev image %s for the fidelity check" % r["image"])
        im = lb(image=im)
        ims.append(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)))
    return torch.from_numpy(np.stack(ims)).float() / 255.0


def _head_output(path, x, device):
    """The head's decoded output (boxes and class scores before any top-k) of
    the checkpoint as the scorer loads it: S.load_model, fused like the
    validator's AutoBackend, fp32, eval."""
    import torch
    from ..inc import scorer as S
    m = S.load_model(path).model
    m = m.float().to(device).eval()
    m.fuse(verbose=False)
    head = m.model[-1]
    store = []
    orig = head._inference

    def wrapped(z):
        y = orig(z)
        store.append(y.detach().float().cpu())
        return y
    head._inference = wrapped
    with torch.no_grad():
        m(x.to(device))
    if not store:
        raise RuntimeError("the head's _inference was not called")
    del m
    return store[-1], bool(getattr(head, "end2end", False))


def fidelity_check(src_path, conv_path, groups, x, conf, device):
    """{images, imgsz, boxes_equal, max_score_diff, empty_max_score, ok, end2end}:
    both files loaded as the scorer loads them, their head outputs on the
    same images; boxes must be equal and every INC channel's score the maximum
    of its source group's (within conversion.logit_tol), every empty channel
    near 0 (conversion.empty_tol)."""
    cv = conf["conversion"]
    ys, e2e = _head_output(src_path, x, device)
    yc, _e = _head_output(conv_path, x, device)
    rec = {"images": int(x.shape[0]), "imgsz": int(x.shape[-1]), "device": str(device), "end2end": e2e,
           "boxes_equal": False, "max_score_diff": None, "empty_max_score": 0.0, "ok": False}
    if ys.shape[0] != yc.shape[0] or ys.shape[-1] != yc.shape[-1] or yc.shape[1] != 4 + C.NC:
        rec["error"] = "output shapes %s and %s" % (tuple(ys.shape), tuple(yc.shape))
        return rec
    rec["boxes_equal"] = bool((ys[:, :4] == yc[:, :4]).all())
    diff, empty = 0.0, 0.0
    for t, g in enumerate(groups):
        if g:
            want = ys[:, [4 + s for s in g]].max(1).values
            diff = max(diff, float((yc[:, 4 + t] - want).abs().max()))
        else:
            empty = max(empty, float(yc[:, 4 + t].max()))
    rec.update(max_score_diff=diff, empty_max_score=empty)
    rec["ok"] = rec["boxes_equal"] and diff <= float(cv["logit_tol"]) and empty <= float(cv.get("empty_tol", 1e-4))
    return rec


def convert_file(src_path, groups, conf, out_pt, x, device, meta=None):
    """Write the converted checkpoint of src_path to out_pt (once), then check
    it: fidelity on the reloaded files and the scorer's own model check.
    Returns the conversion record (without its id fields)."""
    import copy
    import torch
    import ultralytics
    from ..inc import scorer as S
    cv = conf["conversion"]
    out_pt = Path(out_pt)
    single = None
    if not out_pt.is_file():
        ckpt, mod = load_ckpt(src_path)
        conv = copy.deepcopy(mod)
        convert_module(conv, groups, cv["fill_bias"])
        ta = dict(ckpt.get("train_args")) if isinstance(ckpt.get("train_args"), dict) else {}
        single = ta.get("single_cls")
        ta["single_cls"] = False          # Ultralytics keeps single_cls from a checkpoint: never a collapsed val
        obj = {"model": conv, "ema": None, "optimizer": None, "updates": None, "train_args": ta,
               "train_metrics": ckpt.get("train_metrics"), "date": ckpt.get("date"), "version": ckpt.get("version"),
               "zoo_conversion": dict(meta or {}, groups=groups, single_cls_original=single)}
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_pt.with_name(".%s.%d.tmp" % (out_pt.name, os.getpid()))
        try:
            torch.save(obj, str(tmp))
            try:
                os.link(tmp, out_pt)
            except FileExistsError:
                pass
        finally:
            if tmp.exists():
                tmp.unlink()
        del ckpt, mod, conv, obj
        gc.collect()
    fid = fidelity_check(src_path, out_pt, groups, x, conf, device)
    try:
        S._check_model(S.load_model(out_pt), out_pt)
        fid["load_check"] = "ok"
    except S.ScorerRefused as e:
        fid["load_check"] = str(e)[:300]
        fid["ok"] = False
    return {"converted_sha256": _sha_file(out_pt), "single_cls_original": single, "fidelity": fid,
            "ok": bool(fid["ok"]), "ultralytics": ultralytics.__version__, "torch": torch.__version__}


def self_test(conf, x, device):
    """The converter on three synthetic heads under this Ultralytics (a
    yolo11n R1 head, a yolo26n end2end R2 head of 100 classes, a yolo11n R3
    head): refuses unless each converts, reloads, passes the scorer's model
    check and the fidelity check."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    cases = (("yolo11n.yaml", list(CWD12_LEGACY_LABELS), "R1"),
             ("yolo26n.yaml", list(TRAINER_SLOT_LEGACY) + ["aux_%d" % i for i in range(12, 100)], "R2"),
             ("yolo11n.yaml", list(TRAINER_SLOT_LEGACY[:8]) + ["novel_weed"], "R3"))
    out = []
    with tempfile.TemporaryDirectory(prefix="inc2_zoo_selftest_") as tmp:
        for i, (cfg, names, want) in enumerate(cases):
            try:
                torch.manual_seed(i)
                net = DetectionModel(cfg, nc=len(names), verbose=False)
                net.names = {k: n for k, n in enumerate(names)}
                src = Path(tmp) / ("src%d.pt" % i)
                torch.save({"model": net.half(), "train_args": {}, "date": "2026-08-01T00:00:00"}, str(src))
                cmap = class_map(names, conf, legacy=True)
                if cmap["rule"] != want:
                    raise ValueError("class map %s, not %s" % (cmap["rule"], want))
                rec = convert_file(src, cmap["groups"], conf, Path(tmp) / ("conv%d.pt" % i), x, device)
            except Exception as e:  # noqa: BLE001 - any failure of the converter here refuses the inventory
                raise ZooRefused("the converter's self-test failed on %s (%s) under this Ultralytics: %s: %s"
                                 % (cfg, want, type(e).__name__, str(e)[:300]))
            if not rec["ok"]:
                raise ZooRefused("the converter's self-test failed on %s (%s): fidelity %s"
                                 % (cfg, want, {k: rec["fidelity"].get(k) for k in ("boxes_equal", "max_score_diff",
                                                                                    "empty_max_score", "load_check")}))
            out.append({"cfg": cfg, "rule": want, "fidelity": rec["fidelity"]})
    log("convert: self-test passed (%s)" % ", ".join("%s %s" % (c["cfg"], c["rule"]) for c in out))
    return out


def _norm_error(e):
    msg = re.sub(r"/[^\s'\"]+", "<path>", str(e))
    msg = re.sub(r"[0-9a-f]{12,}", "<hex>", msg)
    return "%s: %s" % (type(e).__name__, re.sub(r"[0-9]+", "<n>", msg)[:200])


def conversion_path(version, model_id, suffix):
    return zoo_dir(version) / "converted" / model_id[:2] / ("%s%s" % (model_id, suffix))


def convert_step(version, conf, conf_sha, device=None, systemic=3):
    """Every scorable row outside the INC class space (R1-R5, one per weights
    digest) converted once and checked; the self-test first. A converter
    exception with one normalised message on `systemic` models refuses (the
    converter cannot read this Ultralytics); a tolerance miss is that row's
    unscorable_fidelity."""
    zd = zoo_dir(version)
    models = read_models(version)
    device = device or _device()
    imgsz = _test_setting("imgsz", 640, int)
    x = fidelity_batch(int(conf["conversion"]["fidelity_images"]), imgsz)
    st = self_test(conf, x, device)
    todo = [m for m in models if m.get("scorable") and not m.get("same_weights_as")
            and (m.get("class_map") or {}).get("rule") not in (None, "R0")]
    errors = collections.defaultdict(set)
    pending, n_ok, n_bad = [], 0, 0
    for m in sorted(todo, key=lambda m: m["model_id"]):
        mid = m["model_id"]
        jp = conversion_path(version, mid, ".json")
        prev = _read_json(jp)
        if isinstance(prev, dict) and prev.get("format") == CONVERSION_FORMAT:
            n_ok += bool(prev.get("ok"))
            n_bad += not prev.get("ok")
            continue
        meta = {"source_sha256": mid, "rule": m["class_map"]["rule"], "converter_sha256": _sha_file(Path(__file__).resolve())}
        base = {"format": CONVERSION_FORMAT, "source_sha256": mid, "source": m["rel"], "rule": m["class_map"]["rule"],
                "groups": m["class_map"]["groups"], "dropped": m["class_map"].get("dropped"),
                "converter_sha256": meta["converter_sha256"],
                "converted": str(conversion_path(version, mid, ".pt").relative_to(zd)), "created_utc": None}
        try:
            if _sha_file(m["path"]) != mid:
                raise ValueError("the source no longer hashes as listed")
            rec = dict(base, **convert_file(m["path"], m["class_map"]["groups"], conf,
                                            conversion_path(version, mid, ".pt"), x, device, meta))
        except ZooRefused:
            raise
        except Exception as e:  # noqa: BLE001 - recorded per row; one message on `systemic` rows refuses
            key = _norm_error(e)
            errors[key].add(mid)
            if len(errors[key]) >= systemic:
                raise ZooRefused("the converter raised %r on %d models: systemic, the inventory stops (rows "
                                 "written so far are kept)" % (key, len(errors[key])))
            rec = dict(base, ok=False, fidelity={"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                                                 "traceback": traceback.format_exc()[-1500:]})
            pending.append((jp, rec))
            n_bad += 1
            continue
        rec["created_utc"] = _utc()
        _once_json(jp, rec)
        n_ok += bool(rec["ok"])
        n_bad += not rec["ok"]
    for jp, rec in pending:
        rec["created_utc"] = _utc()
        _once_json(jp, rec)
    log("convert: %d converted and checked, %d unscorable_fidelity" % (n_ok, n_bad))
    return {"self_test": st, "ok": n_ok, "failed": n_bad}


def conversion_of(version, model_id):
    r = _read_json(conversion_path(version, model_id, ".json"))
    return r if isinstance(r, dict) and r.get("format") == CONVERSION_FORMAT else None


# ------------------------------------------------------------- hash cache
class HashCache:
    """hashcache/images.jsonl, append-only: per image path (with its size and
    mtime_ns) its sha256, dHash and 8 variant dHashes (VARIANTS order), and
    where they came from. An image is decoded once across steps and attempts."""

    def __init__(self, version):
        self.path = zoo_dir(version) / "hashcache" / "images.jsonl"
        self.rows = {}
        for r in _read_jsonl(self.path) or []:
            self.rows[r["path"]] = r

    def get(self, path):
        r = self.rows.get(path)
        if r is None:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        return r if r.get("size") == st.st_size and r.get("mtime_ns") == st.st_mtime_ns else None

    def _append(self, rows):
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                self.rows[r["path"]] = r

    def compute(self, paths, procs=5, deadline=None, want_sha=True):
        """{path: entry} for every path (cached, or hashed here). With a
        deadline (seconds since the epoch), images not reached by then stay
        unhashed (entry None) and the call returns."""
        from . import guard as G
        out, todo = {}, []
        for p in dict.fromkeys(paths):
            e = self.get(p)
            if e is not None and e.get("dhash") is not None and (e.get("sha256") or not want_sha):
                out[p] = e
            else:
                todo.append(p)

        def one(p):
            if deadline is not None and time.time() > deadline:
                return p, None
            try:
                st = os.stat(p)
            except OSError:
                return p, {"path": p, "size": None, "mtime_ns": None, "sha256": None, "dhash": None,
                           "variants": None, "source": "unreadable"}
            prev = self.get(p) or {}
            sha = prev.get("sha256") or (_sha_file(p) if want_sha else None)
            if prev.get("dhash") is not None:
                h, vals = prev["dhash"], prev.get("variants")
            else:
                h, v = G.image_hashes(p)
                vl = G.variant_list(v)
                vals = [x for _n, x in vl] if vl else None
                if vals is not None and (h is None or vals[0] != int(h)):
                    vals = None                # the variants' id is not the dHash: different pixels, not trusted
            return p, {"path": p, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha,
                       "dhash": None if h is None else int(h), "variants": vals, "source": "computed"}
        done = []
        for i in range(0, len(todo), 2000):
            with ThreadPoolExecutor(max_workers=max(1, procs)) as ex:
                got = list(ex.map(one, todo[i:i + 2000]))
            new = [e for _p, e in got if e is not None and e.get("source") == "computed"]
            self._append(new)
            for p, e in got:
                out[p] = e
            done += new
            if deadline is not None and time.time() > deadline:
                for p in todo[i + 2000:]:
                    out[p] = None
                break
        return out

    def seed(self, entries):
        """Record stored hashes (a pool, a provenance file) for paths that
        exist, so they are never decoded again."""
        new = []
        for p, (sha, dh, var, src) in entries.items():
            if p in self.rows:
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            new.append({"path": p, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha,
                        "dhash": None if dh is None else int(dh),
                        "variants": None if var is None else [int(x) for x in var], "source": src})
        self._append(new)
        return len(new)


def _hash_rows(entries):
    """[{dhash, variants}] for base3.hash_matrix from cache entries (None: no hash)."""
    return [{"dhash": (e or {}).get("dhash"), "variants": (e or {}).get("variants")} for e in entries]


# ------------------------------------------------------------------ exams
def exam_rows(version, exam):
    p = root_dir(version) / "splits" / "v1" / ("%s.jsonl" % exam)
    return C.read_manifest(p) if p.is_file() else None


def _materialise(rows, out_dir):
    from ..inc import splits as SP
    yaml_path = C.materialise(rows, out_dir)
    for n in os.listdir(Path(out_dir) / "labels"):
        os.chmod(Path(out_dir) / "labels" / n, 0o444)
    os.chmod(yaml_path, 0o444)
    probs = SP.exam_problems(Path(out_dir).name, rows, out_dir)
    if probs:
        raise ZooRefused("the materialised exam %s has problems: %s" % (out_dir, probs[:3]))


def _short_key(prefix, key):
    k = prefix + key
    return k if len(k) <= 180 else "%s_%s" % (k[:160], hashlib.sha1(k.encode("utf-8")).hexdigest()[:12])


def _write_label(version, exam, key, boxes):
    p = root_dir(version) / "labels" / exam / ("%s.txt" % key)
    text = "".join("%d %.6f %.6f %.6f %.6f\n" % (C.OTHER_PLANT, b[0], b[1], b[2], b[3]) for b in boxes)
    _once_bytes(p, text.encode("utf-8"))
    return str(p), _sha_text(text)


def _copy_exams(version, src, root):
    """dev, test, imageweeds: LOCK v1's manifests copied byte for byte (each
    hashing to LOCK v1's entry) and their exam directories linked, after
    inc.splits.exam_problems finds nothing."""
    from ..inc import splits as SP
    lock1 = _read_json(src / "splits" / "v1" / "LOCK.json") or {}
    out = {}
    for e in COPIED_EXAMS:
        mp = src / "splits" / "v1" / ("%s.jsonl" % e)
        raw = mp.read_bytes()
        sha = _sha_bytes(raw)
        if sha != (lock1.get("manifests") or {}).get(e):
            raise ZooRefused("the %s manifest %s does not hash as LOCK v1 records" % (e, mp))
        _once_bytes(root / "splits" / "v1" / ("%s.jsonl" % e), raw)
        rows = C.read_manifest(mp)
        exam_src = src / "exams" / "v1" / e
        probs = SP.exam_problems(e, rows, exam_src)
        if probs:
            raise ZooRefused("LOCK v1's exam directory %s has problems: %s" % (exam_src, probs[:3]))
        link = root / "exams" / "v1" / e
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if os.readlink(link) != str(exam_src):
                raise ZooRefused("%s links to %s, not %s" % (link, os.readlink(link), exam_src))
        else:
            os.symlink(str(exam_src), str(link))
        out[e] = {"rows": rows, "sha256": sha}
    return out, lock1


def _copy_hashes(version, copies, src, cache, procs):
    """The copies' images hashed (8 variants) and each dHash checked against
    the v1 never-train index entry of its split and key."""
    idx = _read_json(src / "splits" / "v1" / "nevertrain_dhash.json") or {}
    want = {(s, k): int(h) for h, s, k in idx.get("entries") or []}
    out = {}
    for e in COPIED_EXAMS:
        rows = copies[e]["rows"]
        got = cache.compute([r["image"] for r in rows], procs)
        bad = [r["key"] for r in rows if (got.get(r["image"]) or {}).get("dhash") is None
               or want.get((e, r["key"])) != got[r["image"]]["dhash"]]
        if bad:
            raise ZooRefused("the exam images differ from the never-train index (%s: %d, e.g. %s)" % (e, len(bad), bad[:3]))
        out[e] = [dict(got[r["image"]], key=r["key"], image=r["image"], manifest_sha256=r["sha256"],
                       session=r.get("session")) for r in rows]
    return out


def _test_v1(version, conf, src, root):
    """test v1 (splits/v3/test_v1/*.jsonl): every row's written image and
    class-12 label as stored, checked byte for byte; key tv1__<sanitised key>."""
    from ..inc.splits import sanitise
    from . import base3 as B3
    tconf = conf["exams"]["test_v1"]
    summ = B3.read_summary(src / "splits" / "v3")
    if summ is None:
        raise ZooRefused("splits/v3/summary.json is missing or not complete: test v1 cannot be read")
    files = sorted(src.glob(tconf["glob"]))
    if not files:
        raise ZooRefused("no test v1 lists match %s" % tconf["glob"])
    rows, keymap, used = [], [], set()
    for f in files:
        for r in C.read_manifest(f):
            key = tconf["key_prefix"] + sanitise(r["key"])
            if key in used:
                key = "%s__%s" % (key, _sha_text(r["key"])[:8])
            used.add(key)
            if _sha_file(r["image"]) != r["sha256"] or _sha_file(r["label"]) != r["label_sha256"]:
                raise ZooRefused("test v1 row %s: its image or label no longer hashes as %s records" % (r["key"], f))
            bad = [b for b in C.read_yolo(r["label"]) if b[0] != C.OTHER_PLANT]
            if bad:
                raise ZooRefused("test v1 row %s: a label box of class %d, not %d" % (r["key"], bad[0][0], C.OTHER_PLANT))
            rows.append({"image": r["image"], "label": r["label"], "sha256": r["sha256"],
                         "label_sha256": r["label_sha256"], "source": r["source"], "session": str(r.get("group") or ""),
                         "key": _short_key("", key), "_hash": {"dhash": r.get("dhash"), "variants": r.get("variants"),
                                                               "original_sha256": r.get("original_sha256"),
                                                               "original_image": r.get("original_image")}})
            keymap.append({"key": rows[-1]["key"], "base3_key": r["key"], "file": f.name})
    comps = []
    for f in sorted(src.glob(tconf["companions"])):
        comps += C.read_manifest(f)
    return rows, keymap, comps, {"summary_sha256": _sha_file(src / "splits" / "v3" / "summary.json"),
                                 "lists": {f.name: _sha_file(f) for f in files}}


def eval_roles(conf, slug, names):
    """{name: weed|drop|unresolved} of a slug from the committed role table."""
    roles = ((conf["exams"].get("eval") or {}).get("roles") or {}).get(slug) or {}
    return {n: roles.get(n) for n in names}


def _eval_exams(version, conf, src, root, cwd12_hashes, cache, procs):
    """evalgroups_v1 (the five test groups) and ooddev_v1 (the two OOD-dev
    groups): per group, whole capture groups sampled from its slugs' images
    (the amendment's rules); also every group's whole-image hashes for the
    contamination flags. Returns ({exam: rows}, records, {group: hashes})."""
    from . import base3 as B3
    ev = conf["exams"]["eval"]
    bits = int(ev["bits"])
    b3conf, _sha = B3.load_config()
    registry = _read_json(B3.registry_path()) or {}
    groups = (b3conf.get("evaluation_groups") or {}).get("groups") or {}
    excl = set(ev.get("exclude") or [])
    stored, sha_by = B3.pool_hashes(src)[:2]
    seeds = {p: (sha_by.get(p), v[0], v, "pool") for p, v in stored.items()}
    rec = {"groups": {}, "not_read": {}}
    cand = []
    ents = registry.get("datasets", registry) or {}
    for exam, gl in ((EXAM_EVG, ev["test_groups"]), (EXAM_OOD, ev["ood_dev_groups"])):
        for g in gl:
            slugs = [s for s in groups.get(g) or [] if s not in excl]
            rec["groups"][g] = {"exam": exam, "slugs": {}}
            for slug in slugs:
                e = ents.get(slug)
                if not isinstance(e, dict):
                    rec["not_read"][slug] = {"group": g, "exam": exam, "why": "not in the registry"}
                    continue
                names, _basis = B3._names_of(e, e.get("local_path") or "")
                roles = eval_roles(conf, slug, names or [])
                if names is None:
                    rec["not_read"][slug] = {"group": g, "exam": exam, "why": "no class names"}
                    continue
                unknown = sorted(n for n, r in roles.items() if r is None)
                if unknown:
                    rec["not_read"][slug] = {"group": g, "exam": exam,
                                             "why": "class names %s are not in the role table" % unknown}
                    continue
                if "weed" not in roles.values():
                    rec["not_read"][slug] = {"group": g, "exam": exam, "why": "no class resolves to weed (%s)" % ", ".join(
                        "%s: %s" % (n, roles[n]) for n in names)}
                    continue
                rows, rr = B3.registry_rows({"sources": {slug: {"classes": roles, "family": g, "tier": 3}}},
                                            registry, sources=[slug])
                if (rr.get(slug) or {}).get("error"):
                    rec["not_read"][slug] = {"group": g, "exam": exam, "why": rr[slug]["error"]}
                    continue
                for r in rows:
                    r.update(group=g, exam=exam, names=names, roles=roles)
                cand += rows
                rec["groups"][g]["slugs"][slug] = {"images": len(rows)}
    cache.seed({p: s for p, s in seeds.items()})
    got = cache.compute([r["image"] for r in cand], procs)
    for r in cand:
        h = got.get(r["image"]) or {}
        r.update(sha256=h.get("sha256"), dhash=h.get("dhash"), variants=h.get("variants"))
    # every group's whole-image hashes (the contamination flags read each group as a whole)
    whole = {}
    for g in rec["groups"]:
        rs = [r for r in cand if r["group"] == g]
        H, ok, has = B3.hash_matrix(rs)
        whole[g] = {"H": H, "ok": ok, "has": has, "paths": [r["image"] for r in rs],
                    "sha256": [r["sha256"] for r in rs]}
        rec["groups"][g]["images"] = len(rs)
        rec["groups"][g]["unhashed"] = int((~has).sum()) if len(rs) else 0
    # labels: weed -> 12, drop removed, an unresolved or unnamed class refuses the image
    live = []
    for r in cand:
        why = None
        if r["dhash"] is None:
            why = "unhashed"
        elif not r.get("label") or not os.path.isfile(r["label"]):
            why = "no_label"
        else:
            boxes, _polys, probs = B3.parse_label(r["label"])
            weed = []
            for b in boxes:
                name = r["names"][b[0]] if b[0] < len(r["names"]) else None
                role = r["roles"].get(name) if name is not None else None
                if role == "weed":
                    weed.append(b[1:])
                elif role != "drop":
                    why = "unresolved_class" if name is not None else "unnamed_class"
                    break
            if probs:
                why = "label_error"
            elif why is None and not weed:
                why = "no_weed"
            r["weed_boxes"] = weed
        r["drop"] = why
        if why is None:
            live.append(r)
    # never a copy of a cwd12 dev or test image (6 bits, 8 variants, both directions)
    H, ok, has = B3.hash_matrix(live)
    HC, okc, hasc = cwd12_hashes
    if len(live) and len(HC):
        d, _a = B3.cross_nearest(H, ok, HC[hasc], okc[hasc], bits)
        for r, dist in zip(live, d.tolist()):
            if dist <= bits:
                r["drop"] = "near_cwd12_eval"
    live = [r for r in live if r["drop"] is None]
    # within and across groups: one row per 6-bit component, the first group by name (case-insensitive) keeps
    if live:
        lab = B3.hash_components(live, bits)
        comp = collections.defaultdict(list)
        for i, l in enumerate(lab.tolist()):
            comp[l].append(i)
        for members in comp.values():
            members.sort(key=lambda i: (live[i]["group"].lower(), live[i]["group"], live[i]["key"]))
            for i in members[1:]:
                live[i]["drop"] = "near_dup_across" if live[i]["group"] != live[members[0]]["group"] else "near_dup_within"
    live = [r for r in live if r["drop"] is None]
    out = {EXAM_EVG: [], EXAM_OOD: []}
    per = int(ev["per_group"])
    for g, gr in sorted(rec["groups"].items()):
        rs = [r for r in live if r["group"] == g]
        if not rs:
            gr.update(sampled=0, capture_groups=0)
            continue
        names, digest = B3.capture_groups(rs, [], bits)
        members = collections.defaultdict(list)
        for i, n in names.items():
            members[n].append(i)
        order = sorted(members, key=lambda n: (C.stable_int("%s/%s/%s" % (ev["seed_text"], g, digest[n])), n))
        took, n_groups, skipped_big = [], 0, 0
        for n in order:
            k = len(members[n])
            if k > per:
                skipped_big += 1
                continue
            if len(took) + k > per:
                continue
            took += members[n]
            n_groups += 1
            if len(took) == per:
                break
        gr.update(sampled=len(took), capture_groups=n_groups, larger_than_cap=skipped_big, after_dedupe=len(rs))
        prefix = ev["key_prefix"][gr["exam"]]
        for i in sorted(took, key=lambda i: rs[i]["key"]):
            r = rs[i]
            key = _short_key(prefix, r["key"])
            lab_path, lab_sha = _write_label(version, gr["exam"], key, r["weed_boxes"])
            out[gr["exam"]].append({"image": r["image"], "label": lab_path, "sha256": r["sha256"],
                                    "label_sha256": lab_sha, "source": r["source"], "session": g, "key": key,
                                    "_hash": {"dhash": r["dhash"], "variants": r["variants"]}})
    drops = collections.Counter(r["drop"] for r in cand if r["drop"])
    for g, gr in rec["groups"].items():
        gr["dropped"] = dict(collections.Counter(r["drop"] for r in cand if r["group"] == g and r["drop"]))
    rec["dropped"] = dict(drops)
    return out, rec, whole


def exams_step(version, conf, conf_sha, procs=5):
    """The zoo's exam root (root/): the copies, test v1, evalgroups_v1 and
    ooddev_v1, the zoo LOCK and ZOO_ROOT.json; exams.json and the exam hashes."""
    import numpy as np
    from ..inc import scorer as S
    from . import base3 as B3
    zd, src, root = zoo_dir(version), source_inc(), root_dir(version)
    lock_p = root / "splits" / "v1" / "LOCK.json"
    scorer_sha = _sha_file(Path(S.__file__).resolve())
    copies, lock1 = _copy_exams(version, src, root)
    if scorer_sha != lock1.get("scorer_sha256"):
        raise ZooRefused("inc/scorer.py hashes to %s, not LOCK v1's scorer_sha256 %s"
                         % (scorer_sha[:12], str(lock1.get("scorer_sha256"))[:12]))
    cache = HashCache(version)
    chash = _copy_hashes(version, copies, src, cache, procs)
    tv_rows, keymap, comps, tv_rec = _test_v1(version, conf, src, root)
    cwd = chash["dev"] + chash["test"]
    HC, okc, hasc = B3.hash_matrix(cwd)
    ev_rows, ev_rec, whole = _eval_exams(version, conf, src, root, (HC, okc, hasc), cache, procs)
    shas = {e: copies[e]["sha256"] for e in COPIED_EXAMS}
    for exam, rows in ((EXAM_TV1, tv_rows), (EXAM_EVG, ev_rows[EXAM_EVG]), (EXAM_OOD, ev_rows[EXAM_OOD])):
        man = [{k: r[k] for k in C.MANIFEST_KEYS} for r in rows]
        mp = root / "splits" / "v1" / ("%s.jsonl" % exam)
        with tempfile.TemporaryDirectory(prefix="inc2_zoo_") as tmp:
            sha = C.write_manifest(Path(tmp) / "m.jsonl", man)
            data = (Path(tmp) / "m.jsonl").read_bytes()
        _once_bytes(mp, data)
        shas[exam] = sha
        ed = root / "exams" / "v1" / exam
        if not ed.exists():
            _materialise(man, ed)
        else:
            from ..inc import splits as SP
            probs = SP.exam_problems(exam, man, ed)
            if probs:
                raise ZooRefused("the exam directory %s differs from its manifest: %s" % (ed, probs[:3]))
    _write_jsonl(zd / "exams_test_v1_keys.jsonl", keymap)
    lock = {"format": LOCK_FORMAT, "scorer_sha256": scorer_sha, "manifests": shas,
            "source_lock_v1_sha256": _sha_file(src / "splits" / "v1" / "LOCK.json"),
            "test_v1_summary_sha256": tv_rec["summary_sha256"], "config_sha256": conf_sha}
    old = _read_json(lock_p)
    if isinstance(old, dict):
        if {k: old.get(k) for k in lock} != lock:
            raise ZooRefused("the zoo LOCK %s exists with other contents: written once" % lock_p)
    else:
        lock["created_utc"] = _utc()
        _once_json(lock_p, lock)
    lock_sha = _sha_file(lock_p)
    marker = {"format": "inc2-zoo-root/1", "version": version, "lock_sha256": lock_sha, "canonical": str(root)}
    _once_json(root / ROOT_MARKER, marker)
    # hashes: every exam (key order) and every evaluation group as a whole
    arrays, meta = {}, {"exams": {}, "groups": {}}
    def put(name, rows, extra=None):
        H, ok, has = B3.hash_matrix(_hash_rows(rows))
        arrays["%s__H" % name], arrays["%s__ok" % name], arrays["%s__has" % name] = H, ok, has
        meta_d = {"n": len(rows), "unhashed": int((~has).sum()) if len(rows) else 0}
        meta_d.update(extra or {})
        return meta_d
    for e in COPIED_EXAMS:
        meta["exams"][e] = put("exam_" + e, chash[e], {"paths": [r["image"] for r in chash[e]],
                                                       "sha256": [r["manifest_sha256"] for r in chash[e]],
                                                       "keys": [r["key"] for r in chash[e]],
                                                       "sessions": [r.get("session") or "" for r in chash[e]]})
    for exam, rows in ((EXAM_TV1, tv_rows), (EXAM_EVG, ev_rows[EXAM_EVG]), (EXAM_OOD, ev_rows[EXAM_OOD])):
        hr = []
        for r in rows:
            h = r["_hash"]
            v = h.get("variants")
            if isinstance(v, dict):
                from . import guard as G
                vl = G.variant_list(v)
                v = [x for _n, x in vl] if vl else None
            hr.append({"dhash": h.get("dhash"), "variants": v})
        meta["exams"][exam] = put("exam_" + exam, hr, {"paths": [r["image"] for r in rows],
                                                       "sha256": [r["sha256"] for r in rows],
                                                       "original_sha256": [r["_hash"].get("original_sha256") for r in rows],
                                                       "keys": [r["key"] for r in rows],
                                                       "groups": [r["session"] for r in rows]})
    cr = []
    for r in comps:
        v = r.get("variants")
        if isinstance(v, dict):
            from . import guard as G
            vl = G.variant_list(v)
            v = [x for _n, x in vl] if vl else None
        cr.append({"dhash": r.get("dhash"), "variants": v})
    meta["exams"]["test_v1_companions"] = put("exam_test_v1_companions", cr, {
        "paths": [r.get("original_image") for r in comps],
        "sha256": [r.get("sha256") for r in comps], "original_sha256": [r.get("original_sha256") for r in comps]})
    for g, w in whole.items():
        arrays["group_%s__H" % g], arrays["group_%s__ok" % g], arrays["group_%s__has" % g] = w["H"], w["ok"], w["has"]
        meta["groups"][g] = {"n": len(w["paths"]), "unhashed": int((~w["has"]).sum()) if len(w["paths"]) else 0,
                             "paths": w["paths"], "sha256": w["sha256"]}
    hp = zd / "hashcache" / "exams.npz"
    hp.parent.mkdir(parents=True, exist_ok=True)
    tmpz = hp.with_name(".exams.%d.npz" % os.getpid())
    np.savez(str(tmpz), **arrays)
    os.replace(tmpz, hp)
    meta["npz_sha256"] = _sha_file(hp)
    _write_json(zd / "hashcache" / "exams.json", meta)
    exams = {}
    for e in ALL_EXAMS:
        rows = C.read_manifest(root / "splits" / "v1" / ("%s.jsonl" % e))
        exams[e] = {"n_images": len(rows), "n_boxes": sum(len(C.read_yolo(r["label"])) for r in rows),
                    "manifest_sha256": shas[e], "role": EXAM_ROLE[e], "agnostic_only": e in AGNOSTIC_ONLY,
                    "labels": {"dev": "cwd12, 12 species (INC ids 0-11)", "test": "cwd12, 12 species (descriptive)",
                               "imageweeds": "ImageWeeds: Ragweed (INC 5) and OtherPlant (12)",
                               EXAM_TV1: "base v3's weed boxes, class 12; crops are background",
                               EXAM_EVG: "the five test groups' weed boxes, class 12 (role table); crops removed",
                               EXAM_OOD: "the two OOD-dev groups' weed boxes, class 12 (role table); crops removed"}[e]}
        if e in (EXAM_EVG, EXAM_OOD):
            exams[e]["groups"] = {g: {k: v for k, v in gr.items() if k != "exam"}
                                  for g, gr in ev_rec["groups"].items() if gr["exam"] == e}
            exams[e]["not_read"] = {s: "%s (group %s)" % (v["why"], v["group"])
                                    for s, v in sorted(ev_rec["not_read"].items()) if v["exam"] == e}
    rec = {"format": EXAMS_FORMAT, "version": version, "config_sha256": conf_sha, "lock_sha256": lock_sha,
           "exams": exams, "test_v1": tv_rec, "test_v1_keys_sha256": _sha_file(zd / "exams_test_v1_keys.jsonl"),
           "eval_dropped": ev_rec.get("dropped"), "created_utc": _utc()}
    _write_json(zd / "exams.json", rec)
    log("exams: %s" % ", ".join("%s %d" % (e, exams[e]["n_images"]) for e in ALL_EXAMS))
    return rec


def read_exams(version):
    r = _read_json(zoo_dir(version) / "exams.json")
    if not isinstance(r, dict) or r.get("format") != EXAMS_FORMAT:
        raise ZooRefused("no exams.json: run the exams step first")
    return r


def load_exam_hashes(version):
    import numpy as np
    zd = zoo_dir(version)
    meta = _read_json(zd / "hashcache" / "exams.json")
    if not isinstance(meta, dict):
        raise ZooRefused("no hashcache/exams.json: run the exams step first")
    with np.load(str(zd / "hashcache" / "exams.npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


# ---------------------------------------------------------- contamination
def _seeds(version, conf, union, prov):
    """Stored hashes for training images: the v1 Step 1 pool (8 variants), splits
    v3's and v2's provenance (8 variants; a written 640-px file takes its
    original's), the INC manifests (sha256), the registry's dHash cache (dHash
    only). {path: (sha256, dhash, variants, source)} for paths in `union`."""
    from . import base3 as B3
    src = source_inc()
    want = set(union)
    out = {}

    def put(p, sha, dh, var, s):
        if p is None:
            return
        p = os.path.normpath(str(p))
        if p not in want:
            return
        old = out.get(p)
        if old is None or (old[1] is None and dh is not None) or (old[2] is None and var is not None):
            out[p] = (sha or (old or (None,))[0], dh if dh is not None else (old[1] if old else None),
                      var if var is not None else (old[2] if old else None), s)
    by_path, sha_by = B3.pool_hashes(src)[:2]
    for p, v in by_path.items():
        put(p, sha_by.get(p), v[0], v, "pool")
    for r in _read_jsonl(src / "splits" / "v3" / "provenance_v1.jsonl") or []:
        put(r.get("original_image"), r.get("original_sha256"), r.get("dhash"), r.get("variants"), "v3prov")
        put(r.get("image"), r.get("sha256"), r.get("dhash"), r.get("variants"), "v3prov")
    for r in _read_jsonl(src / "splits" / "v2" / "base_v2_provenance.jsonl") or []:
        put(r.get("image"), r.get("sha256"), r.get("dhash"), r.get("variants"), "v2prov")
    seen = set()
    for pr in prov.values():
        d = pr.get("data") or {}
        if d.get("source") == "manifest" and d.get("rating") == "exact" and d.get("ref") not in seen:
            seen.add(d.get("ref"))
            for r in C.read_manifest(d["ref"]):
                put(os.path.realpath(r["image"]), r.get("sha256"), None, None, "manifest")
    reg = _read_json(B3.registry_path()) or {}
    for slug, e in (reg.get("datasets", reg) or {}).items():
        if not isinstance(e, dict) or not e.get("local_path"):
            continue
        cache = e.get("dhash_cache") or {}
        if isinstance(cache, str):
            try:
                import ast
                cache = ast.literal_eval(cache)
            except (ValueError, SyntaxError):
                cache = {}
        for rel, h in (cache.items() if isinstance(cache, dict) else []):
            if h is not None:
                put(os.path.join(e["local_path"], rel), None, int(h), None, "registry")
    return out


@__import__("functools").lru_cache(maxsize=64)
def _roots_cached(rels, root):
    return _roots_impl(rels, root)


def _roots(rels, root):
    return _roots_cached(tuple(rels or ()), root)


def _roots_impl(rels, root):
    """The absolute cwd12 roots, each as configured and as its realpath (list
    entries are realpaths)."""
    out = []
    for r in rels or []:
        a = os.path.join(root, r)
        out += [a, os.path.realpath(a)]
    return sorted(set(out))


def _scan_images(d):
    """The image files of one flat directory (no recursion), sorted."""
    fmts = img_formats()
    try:
        with os.scandir(d) as it:
            return sorted(e.path for e in it if not e.name.startswith(".") and "." in e.name
                          and e.name.rpartition(".")[-1].lower() in fmts and e.is_file())
    except OSError:
        return []


def cwd12_reference(version, conf, cache, procs=5):
    """{sha256: (kind, path)} of cwd12's own images, kind 'train' (cwd12 train,
    dev included) or 'holdout' (cwd12 valid and test: the sealed 1,977): the
    stored sha256s of LOCK v1's dev, train_core and test manifests, and every
    image under the configured cwd12 roots (hashed once, cached). A training
    image is a cwd12 image when its bytes are one of these, wherever its copy
    lies (the leave4out datasets keep the original names; merges symlink to
    them)."""
    cont = conf["contamination"]
    root = conf["inventory"]["list_root"]
    ref = {}
    src = source_inc() / "splits" / "v1"
    for name, kind in (("dev", "train"), ("train_core", "train"), ("test", "holdout")):
        p = src / ("%s.jsonl" % name)
        for r in (C.read_manifest(p) if p.is_file() else []):
            if r.get("sha256"):
                ref.setdefault(r["sha256"], (kind, r["image"]))
    paths = {}
    for key, kind in (("cwd12_roots", "train"), ("cwd12_holdout_roots", "holdout")):
        for rel in cont.get(key) or []:
            got = _scan_images(os.path.join(root, rel))
            if not got:
                warn("the cwd12 root %s holds no image: its images are known only through LOCK v1's manifests" % rel)
            for q in got:
                paths[q] = kind
    if paths:
        hashed = cache.compute(sorted(paths), procs, want_sha=True)
        for q, kind in sorted(paths.items()):
            sha = (hashed.get(q) or {}).get("sha256")
            if sha:
                ref.setdefault(sha, (kind, q))
    return ref


def _cwd12_kind(p, sha, ref, cont, root):
    """('train' | 'holdout' | None, the cwd12 original's path): a training
    image's cwd12 membership by its bytes (sha256 in `ref`), else by its
    realpath under a cwd12 root or a configured copy root (the split folder
    gives the kind), else by a copy prefix of its name (kind train)."""
    if sha and sha in ref:
        return ref[sha]
    if _under(p, _roots(cont["cwd12_roots"], root)):
        return "train", p
    if _under(p, _roots(cont.get("cwd12_holdout_roots"), root)):
        return "holdout", p
    for rel, kind in sorted((cont.get("cwd12_copy_roots") or {}).items()):
        if _under(p, _roots([rel], root)):
            return kind, p
    b = os.path.basename(p)
    if any(b.startswith(x) for x in cont["cwd12_copy_prefixes"]):
        return "train", p
    return None, None


def _cwd12_stem(p, cont):
    """The cwd12 stem of an image (its original's path, or a copy's name
    without its merge prefix and Roboflow suffix)."""
    from . import base3 as B3
    stem = os.path.splitext(os.path.basename(p))[0]
    for x in cont["cwd12_copy_prefixes"]:
        if stem.startswith(x):
            stem = stem[len(x):]
            break
    m = B3.RF_STEM_RE.match(stem)
    if m:
        stem = m.group("stem")
    return stem


def own_state(rating, counts, n_unique, tol, incomplete=0):
    """Y / P / U / N of one list on one exam (the amendment's state rule).
    `incomplete` counts the training entries the list could not read (a
    relative line of a .txt list, an unreadable .txt list or directory):
    with any, the list is not the whole training set, so the state is never
    N (clean): U, unless the list read shows the exam's images (Y) or the
    state is P anyway."""
    if rating == "inherits":
        return "N"
    if rating == "none" or counts is None:
        return "U"
    hits = int(counts.get("exact") or 0) + int(counts.get("near") or 0) + int(counts.get("companion") or 0)
    if rating in ("listed_after_rebuild", "derived_superset"):
        state = "P"
    elif hits:
        state = "Y"
    elif not n_unique:
        state = "U"
    elif int(counts.get("unhashed") or 0) > tol * max(1, int(counts.get("n") or 0)):
        state = "P"
    else:
        state = "N"
    return _worse(state, "U") if incomplete else state


def _worse(a, b):
    return a if STATE_RANK.get(a, 1) >= STATE_RANK.get(b, 1) else b


def contamination_step(version, conf, conf_sha, procs=5, hash_budget_s=None):
    """contamination.jsonl: per model and exam the hits of its training list
    (exact by realpath or sha256, near within 6 dHash bits under the 8
    variants both ways, dev sessions, test v1 companions), its own state, the
    inherited states and the final one; also each evaluation group as a whole,
    the val list's test and dev hits, and legacy_join."""
    import numpy as np
    from . import base3 as B3
    from ..inc.splits import session_of
    zd = zoo_dir(version)
    cont = conf["contamination"]
    bits, tol = int(cont["bits"]), float(cont["unhashed_tolerance"])
    root = conf["inventory"]["list_root"]
    prov, models = read_provenance(version), {m["model_id"]: m for m in read_models(version)}
    meta, arrays = load_exam_hashes(version)
    lists = {}
    for pr in prov.values():
        for d in (pr.get("data") or {}, pr.get("val") or {}):
            sha = d.get("list_sha256")
            if sha and sha not in lists:
                got = read_list(version, sha)
                if got is None:
                    raise ZooRefused("the list %s a provenance row names is missing" % sha[:12])
                lists[sha] = got
    union = sorted({p for ps in lists.values() for p in ps})
    pos = {p: i for i, p in enumerate(union)}
    cache = HashCache(version)
    t0 = time.time()
    seeded = cache.seed(_seeds(version, conf, union, prov))
    ref = cwd12_reference(version, conf, cache, procs)
    todo = [p for p in union if (cache.get(p) or {}).get("dhash") is None]
    sample = todo[:200]
    ts = time.time()
    # training images are hashed for their dHashes only (a byte copy of an exam image is 0 bits away: near, the
    # same state); a sha256 is compared when a stored one is known
    cache.compute(sample, procs, want_sha=False)
    rate = (time.time() - ts) / max(1, len(sample))
    budget = float(hash_budget_s if hash_budget_s is not None else cont.get("hash_budget_s", 14400))
    predicted = rate * max(0, len(todo) - len(sample))
    log("contamination: %d distinct training images in %d lists; %d seeded from stored hashes, %d to hash "
        "(about %.0f s at %.3f s/image; budget %.0f s)" % (len(union), len(lists), seeded, len(todo), predicted, rate,
                                                           budget))
    got = cache.compute(union, procs, deadline=time.time() + budget, want_sha=False)
    ents = [got.get(p) for p in union]
    H, ok, has = B3.hash_matrix(_hash_rows(ents))
    sha = [((e or {}).get("sha256")) for e in ents]
    # a byte copy has its original's size: a training image whose size is an exam image's, or a cwd12 image's,
    # gets its sha256 read (cwd12 membership is decided by the bytes)
    sizes = set()
    for q in [q for em in list(meta["exams"].values()) + list(meta["groups"].values()) for q in em.get("paths") or []] \
            + [v[1] for v in ref.values()]:
        try:
            sizes.add(os.path.getsize(q))
        except (OSError, TypeError):
            pass
    need = [p for p, e in zip(union, ents) if e and not e.get("sha256") and e.get("size") in sizes]
    if need:
        shas = _hash_many(need, procs)
        upd = []
        for i, p in enumerate(union):
            if p in shas and shas[p]:
                sha[i] = shas[p]
                upd.append(dict(ents[i], sha256=shas[p]))
        cache._append(upd)
    idx = np.flatnonzero(has)
    if len(idx):
        keys = np.concatenate([H[idx], ok[idx, None].astype(np.uint64)], 1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        UH, Uok = np.ascontiguousarray(uniq[:, :8]), uniq[:, 8].astype(bool)
    else:
        UH, Uok, inv = np.zeros((0, 8), np.uint64), np.zeros(0, bool), np.zeros(0, np.int64)

    def near_to(HE, okE, hasE):
        out = np.zeros(len(union), dtype=bool)
        if not len(idx) or not len(HE) or not hasE.any():
            return out
        d, _a = B3.cross_nearest(UH, Uok, HE[hasE], okE[hasE], bits)
        out[idx] = (d <= bits)[inv]
        return out

    def exact_to(paths, shas):
        rp = {os.path.realpath(p) for p in paths if p}
        ss = {s for s in shas if s}
        return np.asarray([p in rp or (sha[i] is not None and sha[i] in ss) for i, p in enumerate(union)], dtype=bool)

    hits = {}
    for e in HASH_EXAMS:
        em = meta["exams"][e]
        shas_e = list(em.get("sha256") or []) + [x for x in em.get("original_sha256") or [] if x]
        ex = exact_to(em.get("paths") or [], shas_e)
        nr = near_to(arrays["exam_%s__H" % e], arrays["exam_%s__ok" % e], arrays["exam_%s__has" % e]) & ~ex
        hits[e] = {"exact": ex, "near": nr}
    cm = meta["exams"]["test_v1_companions"]
    comp = exact_to(cm.get("paths") or [], list(cm.get("sha256") or []) + [x for x in cm.get("original_sha256") or [] if x]) \
        | near_to(arrays["exam_test_v1_companions__H"], arrays["exam_test_v1_companions__ok"],
                  arrays["exam_test_v1_companions__has"])
    dev_sessions = {session_of(os.path.splitext(os.path.basename(p))[0]) for p in meta["exams"]["dev"]["paths"]}
    kind_of = [_cwd12_kind(p, sha[i], ref, cont, root) for i, p in enumerate(union)]
    kinds = [k for k, _o in kind_of]
    session = np.asarray([k == "train" and session_of(_cwd12_stem(o, cont)) in dev_sessions
                          for k, o in kind_of], dtype=bool)
    groups = {}
    for g, gm in meta["groups"].items():
        groups[g] = exact_to(gm.get("paths") or [], gm.get("sha256") or []) | near_to(
            arrays["group_%s__H" % g], arrays["group_%s__ok" % g], arrays["group_%s__has" % g])
    unhashed = ~has

    def counts_of(sha_l):
        ii = np.asarray([pos[p] for p in lists.get(sha_l) or []], dtype=np.int64)
        out = {}
        for e in HASH_EXAMS:
            ex = int(hits[e]["exact"][ii].sum()) if len(ii) else 0
            nr = int(hits[e]["near"][ii].sum()) if len(ii) else 0
            c = {"exact": ex, "near": nr, "unhashed": int(unhashed[ii].sum()) if len(ii) else 0, "n": int(len(ii))}
            if e == "dev":
                c["session"] = int((session[ii] & ~hits[e]["exact"][ii] & ~hits[e]["near"][ii]).sum()) if len(ii) else 0
            if e == EXAM_TV1:
                c["companion"] = int(comp[ii].sum()) if len(ii) else 0
            out[e] = c
        kk = [kinds[i] for i in ii.tolist()]
        extra = {"groups": {g: int(a[ii].sum()) if len(ii) else 0 for g, a in groups.items()},
                 "cwd12_train": kk.count("train"), "cwd12_holdout": kk.count("holdout"),
                 "non_cwd12": sum(1 for k in kk if k is None)}
        return out, extra

    by_list = {}
    for sha_l in lists:
        by_list[sha_l] = counts_of(sha_l)
    rows = {}
    for mid, pr in prov.items():
        d = pr.get("data") or {}
        cnt, extra = by_list.get(d.get("list_sha256"), (None, None)) if d.get("list_sha256") else (None, None)
        # training entries the list could not read: their images are unknown, so no exam is clean on this list
        n_unres, n_unread = int(d.get("n_unresolved") or 0), int(d.get("n_unreadable") or 0)
        own = {e: own_state(d.get("rating") or "none", (cnt or {}).get(e), d.get("n_unique") or 0, tol,
                            incomplete=n_unres + n_unread) for e in HASH_EXAMS}
        rows[mid] = {"format": CONTAMINATION_FORMAT, "model_id": mid, "list_sha256": d.get("list_sha256"),
                     "rating": d.get("rating"), "n_unresolved": n_unres, "n_unreadable": n_unread,
                     "counts": cnt, "own": own, "eval_groups": (extra or {}).get("groups"),
                     "cwd12": {k: (extra or {}).get(k) for k in ("cwd12_train", "cwd12_holdout", "non_cwd12")},
                     "selected_on": pr.get("selected_on"), "test_selected": pr.get("test_selected"),
                     "dev_selected": bool(pr.get("dev_selected")),
                     "dev_selected_unknown": bool(pr.get("dev_selected_unknown")),
                     "dev_gated": bool(pr.get("dev_gated")), "notes": list(pr.get("notes") or [])}
        v = pr.get("val") or {}
        # a checkpoint chosen on its val set: best.pt, or last.pt of an early stop ("yes"; "unknown" without
        # results.csv). Dev images in its val list make it dev_selected whatever its selected_on; on a val set the
        # zoo places by its list (own_split or unknown: pending), the list's test hits and unread entries decide too
        pending = pr.get("test_selected") == "pending_val_check"
        role = (models.get(mid) or {}).get("ckpt_role")
        chosen = pr.get("chosen_on_val") or "no"
        what = "best.pt" if role == "best" else ("last.pt of an early stop" if chosen == "yes" else
                                                 "last.pt without results.csv (an early stop unknown)")
        if v.get("list_sha256"):
            vc, _vx = by_list[v["list_sha256"]]
            vt = vc["test"]["exact"] + vc["test"]["near"]
            vd = vc["dev"]["exact"] + vc["dev"]["near"]
            # val entries the list could not read: a val set without a hit may still hold test or dev images
            v_inc = int(v.get("n_unresolved") or 0) + int(v.get("n_unreadable") or 0)
            rows[mid]["val"] = {"test_hits": vt, "dev_hits": vd, "n": vc["test"]["n"], "n_unresolved":
                                int(v.get("n_unresolved") or 0), "n_unreadable": int(v.get("n_unreadable") or 0)}
            if pending:
                if vt:
                    rows[mid]["test_selected"] = ("best_partial" if role == "best" else "early_stop_partial") \
                        if chosen == "yes" else "unknown"
                else:
                    rows[mid]["test_selected"] = "unknown" if v_inc else "none"
            if vd and chosen == "yes" and not rows[mid]["dev_selected"]:
                rows[mid]["dev_selected"] = True
                rows[mid]["notes"].append("%s chosen on a val set holding %d dev images" % (what, vd))
            elif vd and chosen == "unknown" and not rows[mid]["dev_selected_unknown"]:
                rows[mid]["dev_selected_unknown"] = True
                rows[mid]["notes"].append("%s on a val set holding %d dev images" % (what, vd))
            elif pending and v_inc and not vd:
                rows[mid]["dev_selected_unknown"] = True
                rows[mid]["notes"].append("%s on a val set of which %d entries could not be read" % (what, v_inc))
            if vt:
                rows[mid]["notes"].append("selected on a val set holding %d cwd12 test images" % vt)
        elif pending:
            rows[mid]["test_selected"] = "unknown"
            rows[mid]["dev_selected_unknown"] = True
            rows[mid]["notes"].append("%s on a val set that could not be listed" % what)
        if n_unres + n_unread:
            rows[mid]["notes"].append("%d training entries could not be read (%d relative, %d unreadable): no exam "
                                      "is clean on this list" % (n_unres + n_unread, n_unres, n_unread))
        m = models.get(mid) or {}
        cmap = m.get("class_map") or {}
        date = (m.get("ckpt") or {}).get("date") or (pr.get("dates") or {}).get("start_utc")
        # the merge joined other datasets' names into legacy slots: an image that is no cwd12 image, or an entry of
        # the cottonweed_holdout slug (listed under its merge prefix: its ids 4-11 deleted, 0-3 moved)
        rows[mid]["holdout_slug"] = int(d.get("holdout_slug") or 0)
        rows[mid]["legacy_join"] = bool(cmap.get("rule") in ("R1", "R2", "R3") and is_legacy_date(date) and (
            d.get("rating") in ("none", None) or not extra or extra["non_cwd12"] or d.get("holdout_slug")))
        if extra and extra["cwd12_train"] and str(date or "")[:10] < PRE_INC_DATE:
            rows[mid]["notes"].append("trained on cwd12 train; dev is 8 sessions of it")
    # inheritance: the init's and every soup member's states (Y > P > U > N); an unknown init is U
    memo = {}

    def parents(mid):
        pr = prov.get(mid) or {}
        out = []
        for x in [pr.get("init") or {}] + list(pr.get("soup_of") or []):
            k = x.get("kind")
            if k == "row" and x.get("model_id"):
                out.append(("row", x["model_id"]))
            elif k == "unknown":
                out.append(("unknown", x.get("ref")))
        return out

    def final_of(mid, stack=()):
        if mid in memo:
            return memo[mid]
        if mid in stack or mid not in rows:
            return {e: "U" for e in HASH_EXAMS}, {"test": False}
        st = dict(rows[mid]["own"])
        sel = {"test": rows[mid]["test_selected"] in TEST_CHOSEN,
               "dev": bool(rows[mid]["dev_gated"] or rows[mid]["dev_selected"]
                           or rows[mid].get("dev_selected_unknown"))}
        inh = {e: "N" for e in HASH_EXAMS}
        for kind, ref in parents(mid):
            if kind == "unknown":
                ps, psel = {e: "U" for e in HASH_EXAMS}, {"test": False}
            else:
                ps, psel = final_of(ref, stack + (mid,))
                sel["test_inherited"] = sel.get("test_inherited") or psel["test"] or psel.get("test_inherited", False)
                # an init (or soup member) chosen by dev, directly or up its own chain: this row is dev-gated
                sel["dev_inherited"] = bool(sel.get("dev_inherited") or psel.get("dev") or psel.get("dev_inherited"))
            for e in HASH_EXAMS:
                inh[e] = _worse(inh[e], ps[e])
        rows[mid]["inherited"] = {"parents": [{"kind": k, "ref": r} for k, r in parents(mid)], "states": inh}
        for e in HASH_EXAMS:
            st[e] = _worse(st[e], inh[e])
        memo[mid] = (st, sel)
        return memo[mid]

    for mid in rows:
        st, sel = final_of(mid)
        rows[mid]["final"] = st
        rows[mid]["test_selected_inherited"] = bool(sel.get("test_inherited"))
        rows[mid]["dev_gated_inherited"] = bool(sel.get("dev_inherited")) and not rows[mid]["dev_gated"]
    for mid in rows:
        rows[mid]["dev_gated"] = bool(rows[mid]["dev_gated"] or rows[mid]["dev_gated_inherited"])
    out = [rows[k] for k in sorted(rows)]
    _write_jsonl(zd / "contamination.jsonl", out)
    summ = {"images": len(union), "lists": len(lists), "seeded": seeded, "hashed_unhashed": int(unhashed.sum()),
            "seconds": round(time.time() - t0, 1), "predicted_hash_s": round(predicted, 1), "hash_budget_s": budget,
            "dev": dict(collections.Counter(r["final"]["dev"] for r in out))}
    log("contamination: %d rows; %d of %d images unhashed; dev states %s" % (len(out), summ["hashed_unhashed"],
                                                                           len(union), summ["dev"]))
    return summ


def read_contamination(version):
    rows = _read_jsonl(zoo_dir(version) / "contamination.jsonl")
    if rows is None:
        raise ZooRefused("no contamination.jsonl: run the contamination step first")
    return {r["model_id"]: r for r in rows}


# ----------------------------------------------------------- score records
class ZooSystemic(RuntimeError):
    """A failure no item can get past here (the root, the GPU, the release, a
    filesystem): the task exits 1 and a rerun retries what is not final."""


def score_path(version, model_id, exam, kind="score"):
    suffix = {"score": ".json", "refusal": ".refused.json", "error": ".error.json"}[kind]
    return zoo_dir(version) / "scores" / model_id[:2] / model_id / ("%s%s" % (exam, suffix))


def record_of(version, model_id, exam):
    """('score'|'reused'|'pilot'|'refused'|'error_final', record) when a final
    record exists, else (None, the retried error record or None)."""
    r = _read_json(score_path(version, model_id, exam))
    if isinstance(r, dict) and r.get("format") == SCORE_FORMAT:
        return r.get("origin") or "score", r
    f = _read_json(score_path(version, model_id, exam, "refusal"))
    if isinstance(f, dict) and f.get("format") == REFUSAL_FORMAT:
        return "refused", f
    e = _read_json(score_path(version, model_id, exam, "error"))
    if isinstance(e, dict) and e.get("format") == ERROR_FORMAT and e.get("final"):
        return "error_final", e
    return None, e if isinstance(e, dict) else None


def scorable(m, version):
    """(scored file path, sha256, conversion record or None) of a row the zoo
    scores, or None: kept, not a weights duplicate, and in the INC class space
    or converted with a passed fidelity check."""
    if not m.get("scorable") or m.get("same_weights_as"):
        return None
    rule = (m.get("class_map") or {}).get("rule")
    if rule == "R0":
        return m["path"], m["model_id"], None
    conv = conversion_of(version, m["model_id"])
    if not conv or not conv.get("ok"):
        return None
    return str(zoo_dir(version) / conv["converted"]), conv["converted_sha256"], conv


def fidelity_failed(m, version):
    """True for a kept, scorable row outside the INC class space whose
    conversion (its own, or that of the row holding the same weights) is
    missing or failed its check: the convert step's decision unscorable_fidelity."""
    if not m.get("scorable") or (m.get("class_map") or {}).get("rule") in (None, "R0"):
        return False
    conv = conversion_of(version, m.get("same_weights_as") or m["model_id"])
    return not conv or not conv.get("ok")


def final_counts(version, rows):
    """The inventory's counts after the convert step: meta's reconciliation
    (files.jsonl joined with meta.jsonl) with every row the convert step made
    unscorable_fidelity moved from keep to unscorable, reconciled again (one
    decision per listed file)."""
    zd = zoo_dir(version)
    inv = _read_json(zd / "inventory" / "summary.json") or {}
    meta_rows = _read_jsonl(zd / "inventory" / "meta.jsonl")
    if meta_rows is None:
        raise ZooRefused("no inventory/meta.jsonl: run the meta step first")
    meta = {r["rel"]: r for r in meta_rows}
    for r in rows:
        if r.get("unscorable_reason") == "unscorable_fidelity":
            meta[r["rel"]] = dict(meta.get(r["rel"]) or {}, decision="unscorable", reason="unscorable_fidelity")
    return reconcile(read_files(version), inv.get("files_listed", 0), inv.get("inc_glob_added", 0), meta=meta)


def _strip_result(res, exam):
    out = {k: v for k, v in res.items() if k != "out"}
    out["locked_scorer_sha256"] = out.pop("scorer_sha256", None)
    out["scorer_production"] = out.pop("production", None)
    if exam in AGNOSTIC_ONLY:
        for k in ("map50_95", "map50", "per_class", "per_class_ap50", "species_map50_95", "species_map50",
                  "image_correct", "n_images_correct"):
            out.pop(k, None)
    return out


def _settings():
    from ..inc import scorer as S
    return {"imgsz": _test_setting("imgsz", S.IMGSZ, int), "batch": _test_setting("batch", S.BATCH, int),
            "device": _test_setting("device", None)}


def _per_group(images, rows, field):
    """{name: {n_images, n_gt, agnostic_map50_95, agnostic_map50}} over the
    manifest rows of each value of `field` (source, or the evaluation group)."""
    from . import scorer_agnostic as SA
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get(field) or ""].append(r["key"])
    out = {}
    for name, keys in sorted(by.items()):
        arr = SA.flatten({k: images[k] for k in keys}, sorted(keys))
        ap, ap50 = SA.collapsed(arr)
        out[name] = {"n_images": len(keys), "n_gt": int(arr["n_gt"].sum()), "agnostic_map50_95": ap,
                     "agnostic_map50": ap50}
    return out


def score_item(version, conf, conf_sha, it, task, origin="scored"):
    """Score one item under the prepared root (C.INC_DIR): writes and returns
    the score record. Raises S.ScorerRefused (a final refusal) or anything
    else (an error)."""
    from ..inc import scorer as S
    from . import scorer_agnostic as SA
    path, exam = it["file"], it["exam"]
    if _sha_file(path) != it["file_sha256"]:
        raise ValueError("%s does not hash as planned (%s): not the planned file" % (path, it["file_sha256"][:12]))
    st = _settings()
    kw = {"lock_check": True, "imgsz": st["imgsz"], "batch": st["batch"], "device": st["device"]}
    per, checks = None, {}
    if exam in AGNOSTIC_ONLY:
        res, images, order = SA.capture(path, exam, **kw)
        full = SA.collapsed(SA.flatten(images, order))
        rd = abs(full[0] - float(res["agnostic_map50_95"]))
        checks["recompute"] = rd
        if rd > SA.MAX_RECOMPUTE_DIFF:
            raise RuntimeError("the captured arrays give %.6g, the pass %.6g: not the pass's inputs"
                               % (full[0], res["agnostic_map50_95"]))
        rows = C.read_manifest(C.manifest_path(exam))
        per = _per_group(images, rows, "source" if exam == EXAM_TV1 else "session")
    else:
        with tempfile.TemporaryDirectory(prefix="inc2_zoo_score_") as tmp:
            res = S.score(path, exam, Path(tmp) / "score.json", **kw)
    marker = _read_json(C.INC_DIR / ROOT_MARKER) or {}
    stamp = ("TEST-" if testing() or not res.get("production") else "") + "ZOO-" + str(res.get("scorer_sha256"))\
        .replace("TEST-", "")
    rec = {"format": SCORE_FORMAT, "model_id": it["model_id"], "exam": exam, "origin": origin,
           "scored_file": os.path.relpath(path, str(source_inc())) if str(path).startswith(str(source_inc())) else path,
           "scored_sha256": it["file_sha256"], "conversion": it.get("conversion"),
           "result": _strip_result(res, exam), "per_source" if exam == EXAM_TV1 else "per_group": per,
           "checks": checks, "exam_root": "lustre" if str(marker.get("canonical")) == str(C.INC_DIR) else "local",
           "zoo_lock_sha256": marker.get("lock_sha256"), "scorer_stamp": stamp, "production": False,
           "testing": testing(), "task": task, "config_sha256": conf_sha,
           "code": {"zoo": _sha_file(Path(__file__).resolve()), "scorer": _sha_file(Path(S.__file__).resolve()),
                    "scorer_agnostic": _sha_file(Path(SA.__file__).resolve())},
           "created_utc": _utc()}
    if per is None:
        rec.pop("per_group", None)
    got = _link_once(score_path(version, it["model_id"], exam), rec)
    if got == "exists":
        return record_of(version, it["model_id"], exam)[1]
    ep = score_path(version, it["model_id"], exam, "error")
    if ep.exists():
        ep.unlink()
    return rec


def check_root(version):
    """The process's INC_DIR must be a prepared zoo root of this version (its
    marker naming the canonical LOCK by sha256), Ultralytics the pinned
    release and a GPU present (tests excepted). ZooSystemic otherwise."""
    from ..inc import scorer as S
    marker = _read_json(C.INC_DIR / ROOT_MARKER)
    canon = root_dir(version) / "splits" / "v1" / "LOCK.json"
    want = _sha_file(canon)
    if not isinstance(marker, dict) or marker.get("version") != version:
        raise ZooSystemic("INC_DIR %s is not a zoo root of %s (no %s)" % (C.INC_DIR, version, ROOT_MARKER))
    if want is None or marker.get("lock_sha256") != want or _sha_file(C.LOCK_PATH) != want:
        raise ZooSystemic("the root's LOCK does not hash as the canonical zoo LOCK %s" % canon)
    import ultralytics
    if ultralytics.__version__ != S.PINNED_ULTRALYTICS and not testing():
        raise ZooSystemic("ultralytics %s, not the pinned %s" % (ultralytics.__version__, S.PINNED_ULTRALYTICS))
    import torch
    if not torch.cuda.is_available() and not testing():
        raise ZooSystemic("no CUDA device: the locked scorer's protocol is fp16 on a GPU")


_SYSTEMIC_RE = re.compile(r"CUDA|cuda|cuDNN|out of memory|NCCL|device-side|Input/output error|Stale file handle|"
                          r"No space left", re.I)


def _is_systemic(e):
    import torch
    return isinstance(e, (MemoryError, OSError)) or (hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
        e, torch.cuda.OutOfMemoryError)) or bool(_SYSTEMIC_RE.search(str(e)))


def _write_error(version, it, e, task, final_after):
    ep = score_path(version, it["model_id"], it["exam"], "error")
    prev = _read_json(ep) or {}
    n = int(prev.get("attempts") or 0) + 1
    rec = {"format": ERROR_FORMAT, "model_id": it["model_id"], "exam": it["exam"], "attempts": n,
           "final": n >= int(final_after), "error": "%s: %s" % (type(e).__name__, str(e)[:500]),
           "normalised": _norm_error(e), "traceback": traceback.format_exc()[-2000:], "task": task,
           "created_utc": _utc()}
    _write_json(ep, rec)
    return rec


def run_items(version, conf, conf_sha, items, task, stage, shard_predicted_s=None, cap_h=None):
    """Score items in order (the score verb's loop): skip final records, stop
    at the task deadline (not_scored_time), the task's allowance or the
    chain's GPU-hour cap (not_scored_budget), write the task record after
    each item. Returns the task record. A scoring task's deadline runs from
    its job's start; the pilot's from its own start (it runs inside the
    inventory job, hours after that job started)."""
    from ..inc import scorer as S
    stg = conf["stages"]
    t_start = time.time() if stage == "pilot" else _job_start()
    deadline = float(_test_setting("task_deadline_s", stg["task_deadline_s"], float))
    allowance = max(float(stg.get("task_allowance_min_s", 900)),
                    float(stg.get("task_allowance_factor", 2.0)) * float(shard_predicted_s or 0)) \
        if shard_predicted_s else None
    job, tidx = _job_ident()
    rec = {"format": TASK_FORMAT, "stage": stage, "task": task, "job": job, "array_task": tidx,
           "host": os.uname()[1], "started_utc": _utc(t_start), "ended_utc": None, "status": "running",
           "exam_root": "lustre" if str((_read_json(C.INC_DIR / ROOT_MARKER) or {}).get("canonical")) == str(C.INC_DIR)
           else "local", "items": [], "counts": {}, "seconds": 0.0, "allowance_s": allowance,
           "versions": {}}
    tp = zoo_dir(version) / "tasks" / ("%s.json" % task)
    # the scoring arrays' GPU time goes to the ledger (the pilot's is the inventory job's own, already counted)
    led = stage in ("a", "c")
    if led:
        ledger_touch(version, stage)
    try:
        import torch
        import ultralytics
        rec["versions"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
    except Exception:  # noqa: BLE001
        pass
    same = collections.defaultdict(set)
    stop = None
    for i, it in enumerate(items):
        now = time.time()
        row = {"model_id": it["model_id"], "exam": it["exam"], "status": None}
        rec["items"].append(row)
        if stop is None:
            if now - t_start > deadline:
                stop = "not_scored_time"
            elif allowance is not None and now - _T0 > allowance:
                stop = "not_scored_budget"
                rec["stopped"] = "the task's allowance (%.0f s) is spent" % allowance
            elif cap_h is not None:
                spent = ledger_hours(version, now=now, own=stage).get("total", 0.0)
                if spent + float(it.get("predicted_s") or 0) / 3600.0 > cap_h:
                    stop = "not_scored_budget"
                    rec["stopped"] = "the chain's GPU-hour cap: %.2f GPU-h spent of %.2f" % (spent, cap_h)
        if stop is not None:
            row["status"] = stop
            continue
        kind, _r = record_of(version, it["model_id"], it["exam"])
        if kind is not None:
            row["status"] = "kept_%s" % kind
            continue
        t0 = time.time()
        try:
            score_item(version, conf, conf_sha, it, task, origin="pilot" if stage == "pilot" else "scored")
            row["status"] = "scored"
        except S.ScorerRefused as e:
            _once_json(score_path(version, it["model_id"], it["exam"], "refusal"),
                       {"format": REFUSAL_FORMAT, "model_id": it["model_id"], "exam": it["exam"],
                        "refusal": str(e)[:1000], "task": task, "created_utc": _utc()})
            row["status"] = "refused"
        except Exception as e:  # noqa: BLE001 - one item's error is recorded and retried; a systemic one stops the task
            er = _write_error(version, it, e, task, stg.get("error_attempts", 2))
            row["status"] = "error_final" if er["final"] else "error"
            same[er["normalised"]].add(it["model_id"])
            if _is_systemic(e) or len(same[er["normalised"]]) >= 3:
                row["systemic"] = True
                rec.update(status="systemic", ended_utc=_utc(), error=er["error"])
                _finish_task(version, rec, tp, stage, led)
                raise ZooSystemic("item %s:%s: %s" % (it["model_id"][:12], it["exam"], er["error"]))
        row["seconds"] = round(time.time() - t0, 2)
        rec["seconds"] = round(time.time() - _T0, 1)
        rec["counts"] = dict(collections.Counter(r["status"] for r in rec["items"]))
        _write_json(tp, rec)
        if led:
            ledger_touch(version, stage)
    rec.update(status="done", ended_utc=_utc())
    _finish_task(version, rec, tp, stage, led)
    return rec


def _finish_task(version, rec, tp, stage, led=True):
    rec["counts"] = dict(collections.Counter(r["status"] for r in rec["items"]))
    rec["seconds"] = round(time.time() - _T0, 1)
    _write_json(tp, rec)
    if led:
        ledger_touch(version, stage, ended=True)


def shard_path(version, stage, index=None):
    if stage == "pilot":
        return zoo_dir(version) / "shards" / "pilot.json"
    return zoo_dir(version) / "shards" / ("%s_%03d.json" % (stage, int(index)))


def read_shard(version, stage, index=None):
    sh = _read_json(shard_path(version, stage, index))
    if not isinstance(sh, dict) or sh.get("format") != SHARD_FORMAT:
        raise ZooRefused("no shard %s %s: the plan (or the selection) has not written it" % (stage, index))
    return sh


def inc_off_dev(version, items):
    """The items that would read an INC row on an exam other than dev. An INC
    row is read on dev alone (its other columns are its own records): the
    plan, the pilot and the shortlist never put one in a shard, and a scoring
    task that meets one refuses before it scores anything."""
    fam = {m["model_id"]: m.get("family") for m in read_models(version)}
    return [it for it in items if it.get("exam") != "dev" and is_inc_family(fam.get(it.get("model_id")) or "")]


def score_cmd(version, stage=None, shard=None, pilot=False, item=None):
    conf, conf_sha = load_config(version)
    check_root(version)
    if pilot:
        sh, stg, task = read_shard(version, "pilot"), "pilot", "pilot"
    elif item:
        mid, _c, exam = item.partition(":")
        m = next((x for x in read_models(version) if x["model_id"] == mid), None)
        got = scorable(m, version) if m else None
        if got is None:
            raise ZooRefused("%s is not a scorable row" % mid)
        it = {"model_id": mid, "exam": exam, "file": got[0], "file_sha256": got[1],
              "conversion": _conv_brief(got[2])}
        sh, stg, task = {"items": [it], "predicted_s": None}, "item", "item_%s_%s" % (mid[:12], exam)
    else:
        idx = int(shard if shard is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        sh, stg, task = read_shard(version, stage, idx), stage, "%s_%03d" % (stage, idx)
    off = inc_off_dev(version, sh["items"])
    if off:
        raise ZooRefused("an INC row is read on dev alone: %s; nothing scored"
                         % ", ".join("%s:%s" % (it["model_id"][:12], it["exam"]) for it in off[:5]))
    cap = None
    if stg in ("a", "c"):
        st = conf["stages"]
        plan = _read_json(zoo_dir(version) / "plan.json") or {}
        cap = float(plan.get("max_gpu_hours") or 40) - float(st["reserve_report"]) - (
            float(st["reserve_select"]) if stg == "a" else 0.0)
    return run_items(version, conf, conf_sha, sh["items"], task, stg, sh.get("predicted_s"), cap)


def _conv_brief(conv):
    if not conv:
        return None
    return {"rule": conv.get("rule"), "converted_sha256": conv.get("converted_sha256"),
            "record_sha256": _sha_text(json.dumps(conv, sort_keys=True))}


# ---------------------------------------------------------------- exam-root
def exam_root_cmd(version, stage=None, shard=None, pilot=False):
    """The root a scoring task reads: a node-local copy ($LOCAL) of the
    shard's exams when there is room (1.5 x their images' bytes), else the
    Lustre root. Prints only the path (logs go to stderr)."""
    import shutil as _sh
    root = root_dir(version)
    if not (root / ROOT_MARKER).is_file():
        raise ZooRefused("the zoo root %s is not prepared (no %s)" % (root, ROOT_MARKER))
    sh = read_shard(version, "pilot") if pilot else read_shard(version, stage, shard)
    exams = sorted({it["exam"] for it in sh["items"]})
    local = os.environ.get("LOCAL")
    need = 0
    rows = {}
    for e in exams:
        rows[e] = C.read_manifest(root / "splits" / "v1" / ("%s.jsonl" % e))
        for r in rows[e]:
            try:
                need += os.path.getsize(r["image"])
            except OSError:
                pass
    if not local or not os.path.isdir(local) or _sh.disk_usage(local).free < 1.5 * need:
        log("exam-root: the Lustre root (%s)" % ("no $LOCAL" if not local else "too little room on $LOCAL"))
        return str(root)
    job, task = _job_ident()
    lr = Path(local) / ("inc_zoo_%s_%s_%s" % (version, job, "pilot" if pilot else "%s%s" % (stage, shard)))
    if lr.exists():
        _sh.rmtree(lr)
    (lr / "splits" / "v1").mkdir(parents=True)
    for f in (root / "splits" / "v1").iterdir():
        if f.is_file():
            _sh.copyfile(f, lr / "splits" / "v1" / f.name)
    for e in exams:
        ed, src = lr / "exams" / "v1" / e, root / "exams" / "v1" / e
        (ed / "images").mkdir(parents=True)
        (ed / "labels").mkdir()
        for r in rows[e]:
            ext = os.path.splitext(r["image"])[1].lower() or ".jpg"
            _sh.copyfile(src / "images" / (r["key"] + ext), ed / "images" / (r["key"] + ext))
            _sh.copyfile(src / "labels" / (r["key"] + ".txt"), ed / "labels" / (r["key"] + ".txt"))
        with open(ed / "data.yaml", "w") as fh:
            fh.write("path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n" % (ed, C.NC))
            for n in C.CLASS_NAMES:
                fh.write("  - %s\n" % n)
    marker = dict(_read_json(root / ROOT_MARKER))
    _write_json(lr / ROOT_MARKER, marker)
    log("exam-root: a node-local copy of %s (%.1f GB) at %s" % (", ".join(exams), need / 1e9, lr))
    return str(lr)


# ------------------------------------------------------------------- pilot
def _pilot_env(version):
    env = dict(os.environ)
    env["INC_ZOO_SOURCE"] = str(source_inc())
    env["INC_DIR"] = str(root_dir(version))
    pkg = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def pilot_step(version, conf, conf_sha, run=True):
    """One scorable non-INC checkpoint per size class, scored on every exam (real
    records, origin pilot) in a subprocess with INC_DIR at the zoo root; the
    rates (seconds per image, per exam and size class) the plan prices from."""
    zd = zoo_dir(version)
    models = read_models(version)
    picks = {}
    for m in sorted(models, key=lambda m: m["model_id"]):
        if is_inc_family(m["family"]) or m.get("size_class") not in SIZE_CLASSES:
            continue
        got = scorable(m, version)
        if got is not None and m["size_class"] not in picks:
            picks[m["size_class"]] = (m, got)
    n_img = {e: int(x.get("n_images") or 0) for e, x in read_exams(version)["exams"].items()}
    items = []
    for sc, (m, got) in sorted(picks.items()):
        for e in ALL_EXAMS:
            if not n_img.get(e):
                continue                                  # an exam without images needs no rate
            items.append({"model_id": m["model_id"], "exam": e, "file": got[0], "file_sha256": got[1],
                          "conversion": _conv_brief(got[2]), "size_class": sc})
    _write_json(shard_path(version, "pilot"), {"format": SHARD_FORMAT, "stage": "pilot", "index": None,
                                               "exams": sorted({i["exam"] for i in items}), "items": items,
                                               "predicted_s": None})
    t0 = time.time()
    if run and items:
        p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.zoo", "score", "--pilot",
                            "--version", version], env=_pilot_env(version))
        if p.returncode not in (0,):
            raise ZooRefused("the pilot's scoring pass exited %d" % p.returncode)
    rates = collections.defaultdict(dict)
    measured = []
    for it in items:
        kind, r = record_of(version, it["model_id"], it["exam"])
        if kind in ("pilot", "score", "scored") and r:
            n = int((r.get("result") or {}).get("n_images") or 0)
            sec = float((r.get("result") or {}).get("seconds") or 0)
            if n:
                rates[it["exam"]][it["size_class"]] = sec / n
                measured.append({"model_id": it["model_id"], "exam": it["exam"], "size_class": it["size_class"],
                                 "seconds": sec, "n_images": n})
    for e in ALL_EXAMS:
        have = rates.get(e) or {}
        top = max(have.values()) if have else None
        for sc in SIZE_CLASSES:
            if sc not in have and top is not None:
                rates[e][sc] = top
    missing = unpriced_exams(version, rates)
    task = (_read_json(zd / "tasks" / "pilot.json") or {}) if run and items else {}
    rec = {"format": PILOT_FORMAT, "version": version, "config_sha256": conf_sha, "models": {
        sc: m["model_id"] for sc, (m, _g) in picks.items()}, "rates": {e: dict(v) for e, v in rates.items()},
        "measured": measured, "status": "incomplete" if missing else "complete", "missing_exams": missing,
        "task_counts": task.get("counts"), "seconds": round(time.time() - t0, 1), "created_utc": _utc()}
    _write_json(zd / "pilot.json", rec)
    log("pilot: %d models, %d items, %.0f s%s" % (len(picks), len(items), rec["seconds"],
                                                  "; no rate measured for %s" % ", ".join(missing) if missing else ""))
    if run and missing:
        # not marked done: a resubmission runs the pilot again (its final records are kept, errors retried)
        raise ZooRefused("the pilot measured no rate for %s (pilot task counts %s): the plan prices only from "
                         "measured rates" % (", ".join(missing), json.dumps(task.get("counts"), sort_keys=True)))
    return rec


def unpriced_exams(version, rates):
    """The exams (with images) for which `rates` holds no measured rate."""
    exams = read_exams(version)["exams"]
    return [e for e in ALL_EXAMS if int((exams.get(e) or {}).get("n_images") or 0) > 0
            and not any(_num(v) and v > 0 for v in ((rates or {}).get(e) or {}).values())]


# -------------------------------------------------------------------- plan
def reuse_records(version, conf, conf_sha, models, files):
    """An INC run's recorded score copied as a zoo record (origin reused) when
    production is true (tests: a test score), scorer and manifest are the
    zoo LOCK's, the weights are the file, the settings the protocol's and the
    release the pinned one. Returns {exam: n}."""
    from ..inc import scorer as S
    lock = _read_json(root_dir(version) / "splits" / "v1" / "LOCK.json") or {}
    idx = Index(files, models)
    n = collections.Counter()
    for m in models:
        if not is_inc_family(m["family"]) or m.get("mlflow") or not m.get("scorable") or m.get("same_weights_as"):
            continue
        for exam in REUSE_EXAMS:
            if record_of(version, m["model_id"], exam)[0] is not None:
                n[exam] += 1
                continue
            for rdir in idx.inc_runs.get(m["model_id"]) or []:
                p = Path(rdir) / "scores" / ("%s.json" % exam)
                r = _read_json(p)
                if not isinstance(r, dict) or r.get("exam") != exam:
                    continue
                sst = r.get("settings") or {}
                prod_ok = r.get("production") is True and r.get("scorer_sha256") == lock.get("scorer_sha256") \
                    and sst.get("imgsz") == S.IMGSZ and sst.get("batch") == S.BATCH and sst.get("half") is True \
                    and sst.get("conf") == S.CONF and sst.get("iou") == S.IOU \
                    and r.get("ultralytics_version") == S.PINNED_ULTRALYTICS
                test_ok = testing() and str(r.get("scorer_sha256")).replace("TEST-", "") == lock.get("scorer_sha256")
                if not ((prod_ok or test_ok) and r.get("manifest_sha256") == (lock.get("manifests") or {}).get(exam)
                        and r.get("weights_sha256") == m["model_id"]):
                    continue
                rec = {"format": SCORE_FORMAT, "model_id": m["model_id"], "exam": exam, "origin": "reused",
                       "scored_file": m["rel"], "scored_sha256": m["model_id"], "conversion": None,
                       "result": _strip_result(dict(r), exam), "checks": {}, "exam_root": None,
                       "zoo_lock_sha256": _sha_file(root_dir(version) / "splits" / "v1" / "LOCK.json"),
                       "scorer_stamp": ("TEST-" if not prod_ok else "") + "ZOO-" + str(lock.get("scorer_sha256")),
                       "production": False, "testing": not prod_ok, "task": "plan", "config_sha256": conf_sha,
                       "from": {"name": os.path.relpath(str(p), str(source_inc())), "sha256": _sha_file(p)},
                       "created_utc": _utc()}
                _link_once(score_path(version, m["model_id"], exam), rec)
                n[exam] += 1
                break
    return dict(n)


def _lpt(items, k):
    bins = [{"items": [], "s": 0.0} for _ in range(int(k))]
    for it in sorted(items, key=lambda i: (-float(i["predicted_s"]), i["model_id"], i["exam"])):
        b = min(bins, key=lambda b: (b["s"], bins.index(b)))
        b["items"].append(it)
        b["s"] += float(it["predicted_s"])
    return bins


def _write_shards(version, stage, bins, n):
    for i in range(int(n)):
        b = bins[i]
        p = shard_path(version, stage, i)
        rec = {"format": SHARD_FORMAT, "stage": stage, "index": i, "exams": sorted({x["exam"] for x in b["items"]}),
               "items": b["items"], "predicted_s": round(b["s"], 1)}
        if p.is_file():
            old = _read_json(p) or {}
            if old.get("items") != rec["items"]:
                raise ZooRefused("%s exists with other items: written once" % p)
            continue
        _once_json(p, rec)


def _order_group(m, conf):
    if not is_inc_family(m["family"]):
        return "non_inc"
    if m["family"] == "inc_pilot":
        mm = re.search(r"/(pilot_v[1-4])/", m["rel"])
        return "inc_%s" % mm.group(1) if mm else "inc_pilot"
    return m["family"]


def plan_step(version, conf, conf_sha, shards_a, max_gpu_hours, inventory_h=None):
    """Reused records, then stage A (dev for every scorable row without one)
    and stage B (test v1 for every scorable non-INC row, in order_b) priced
    from the pilot's rates x margin, cut to the cap; shards by
    longest-processing-time. Refuses (exit 2, record refused) when stage A
    alone does not fit."""
    zd = zoo_dir(version)
    stg = conf["stages"]
    plan_p = zd / "plan.json"
    old = _read_json(plan_p)
    if isinstance(old, dict) and old.get("format") == PLAN_FORMAT and old.get("status") == "plan_ready":
        if int(old.get("shards_a") or -1) != int(shards_a):
            raise ZooRefused("plan.json was written for --shards-a %s, not %s: resubmit with the same shards"
                             % (old.get("shards_a"), shards_a))
        return old
    models, files = read_models(version), read_files(version)
    exams = read_exams(version)["exams"]
    pilot = _read_json(zd / "pilot.json") or {}
    rates = pilot.get("rates") or {}
    missing = unpriced_exams(version, rates)
    if missing:
        # every item is priced from the pilot's measured rates (x margin): never from an assumed rate
        msg = "the pilot measured no rate for %s (pilot.json status %r): nothing is priced from an assumed rate" % (
            ", ".join(missing), pilot.get("status"))
        update_record(version, status="refused", refusal=msg)
        raise ZooRefused(msg)
    reused = reuse_records(version, conf, conf_sha, models, files)
    margin = float(stg["margin"])
    inv_h = float(inventory_h if inventory_h is not None else ledger_hours(version, own="inventory").get("inventory", 0.0))

    def price(m, exam):
        have = {k: float(v) for k, v in (rates.get(exam) or {}).items() if _num(v) and v > 0}
        r = have.get(m.get("size_class") or "L") or max(have.values(), default=0.0)
        return float(r) * int(exams[exam]["n_images"]) * margin

    items_a, items_b = [], []
    order = {g: i for i, g in enumerate(stg["order_b"])}
    for m in sorted(models, key=lambda m: m["model_id"]):
        got = scorable(m, version)
        if got is None:
            continue
        base = {"model_id": m["model_id"], "file": got[0], "file_sha256": got[1], "conversion": _conv_brief(got[2]),
                "size_class": m.get("size_class")}
        if record_of(version, m["model_id"], "dev")[0] is None:
            items_a.append(dict(base, exam="dev", predicted_s=round(price(m, "dev"), 2)))
        if not is_inc_family(m["family"]) and record_of(version, m["model_id"], EXAM_TV1)[0] is None:
            items_b.append(dict(base, exam=EXAM_TV1, predicted_s=round(price(m, EXAM_TV1), 2),
                                group=_order_group(m, conf)))
    items_b.sort(key=lambda i: (order.get(i["group"], len(order)), i["model_id"]))
    available = float(max_gpu_hours) - inv_h - float(stg["reserve_select"]) - float(stg["reserve_report"]) \
        - float(stg["reserve_c"])
    sum_a = sum(i["predicted_s"] for i in items_a) / 3600.0
    if sum_a > available:
        msg = "stage A alone costs %.2f GPU-h > available %.2f (cap %s, inventory %.2f, reserves)" % (
            sum_a, available, max_gpu_hours, inv_h)
        update_record(version, status="refused", refusal=msg)
        raise ZooRefused(msg)
    limit = min(available, sum_a + float(stg["budget_b_gpu_hours"]))
    take, dropped, tot = [], [], sum_a
    for it in items_b:
        h = it["predicted_s"] / 3600.0
        if tot + h <= limit:
            take.append(it)
            tot += h
        else:
            dropped.append({"model_id": it["model_id"], "exam": it["exam"], "why": "not_scored_budget"})
    bins = _lpt(items_a + take, shards_a)
    _write_shards(version, "a", bins, shards_a)
    rec = {"format": PLAN_FORMAT, "version": version, "status": "plan_ready", "config_sha256": conf_sha,
           "shards_a": int(shards_a), "max_gpu_hours": float(max_gpu_hours), "rates": rates, "margin": margin,
           "inventory_elapsed_h": round(inv_h, 4), "available_h": round(available, 4),
           "items_a": len(items_a), "items_b": len(take), "items_b_dropped": dropped,
           "predicted_h": round(tot, 4), "reused": reused,
           "shards": [{"index": i, "items": len(b["items"]), "predicted_s": round(b["s"], 1)}
                      for i, b in enumerate(bins)], "created_utc": _utc()}
    _write_json(plan_p, rec)
    log("plan: stage A %d items (%.2f GPU-h), stage B %d items (%d not scored: budget), %d shards"
        % (len(items_a), sum_a, len(take), len(dropped), shards_a))
    return rec


# ------------------------------------------------------------------- rows
def _planned(version):
    """{(model_id, exam): (stage, shard index or None)} of every item the
    pilot, the plan and the selection put in a shard."""
    out = {}
    d = zoo_dir(version) / "shards"
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        files = []
    for p in files:
        sh = _read_json(p) or {}
        if sh.get("format") != SHARD_FORMAT:
            continue
        for it in sh.get("items") or []:
            out[(it["model_id"], it["exam"])] = (sh.get("stage"), sh.get("index"))
    return out


def _task_marks(version):
    """{(model_id, exam): status} from the task records (the latest attempt of
    each shard): not_scored_time, not_scored_budget, error."""
    out = {}
    d = zoo_dir(version) / "tasks"
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        files = []
    for p in files:
        t = _read_json(p) or {}
        for it in t.get("items") or []:
            if it.get("status") in ("not_scored_time", "not_scored_budget", "error"):
                out[(it["model_id"], it["exam"])] = it["status"]
    return out


def _cell(version, mid, exam, marks, dropped_b):
    kind, r = record_of(version, mid, exam)
    c = {"status": None, "origin": None, "species_map50_95": None, "map50_95": None, "agnostic_map50_95": None,
         "agnostic_map50": None, "per_class": None, "n_images": None, "record": None, "record_sha256": None}
    if kind in ("score", "scored", "reused", "pilot"):
        res = r.get("result") or {}
        c.update(status=kind if kind != "score" else "scored", origin=r.get("origin"),
                 species_map50_95=res.get("species_map50_95"), map50_95=res.get("map50_95"),
                 agnostic_map50_95=res.get("agnostic_map50_95"), agnostic_map50=res.get("agnostic_map50"),
                 per_class=res.get("per_class"), n_images=res.get("n_images"),
                 record=os.path.relpath(str(score_path(version, mid, exam)), str(zoo_dir(version))),
                 record_sha256=_sha_file(score_path(version, mid, exam)))
        c["per"] = r.get("per_source") or r.get("per_group")
    elif kind == "refused":
        c.update(status="ref", refusal=(r or {}).get("refusal"))
    elif kind == "error_final":
        c.update(status="err", error=(r or {}).get("error"))
    else:
        m = marks.get((mid, exam))
        c["status"] = {"not_scored_time": "nt", "not_scored_budget": "nb", "error": "err"}.get(m) or (
            "nb" if (mid, exam) in dropped_b else None)
    return c


def _sortdate(pr, m):
    d = (pr or {}).get("dates") or {}
    return str(d.get("end") or d.get("start_utc") or (m.get("ckpt") or {}).get("date") or "")


def read_codever(version):
    """{model_id: {kind, commit, ...}} from `zoo codever`'s codever_<version>.json ({} before it ran)."""
    r = _read_json(zoo_dir(version) / ("codever_%s.json" % version))
    return dict(r.get("rows") or {}) if isinstance(r, dict) and r.get("format") == CODEVER_FORMAT else {}


def _row_code(pr, cv):
    """A row's code version: provenance's (exact module sha256s for INC rows, a
    start date otherwise) joined with its commit from codever, when it ran."""
    code = dict(pr.get("code") or {"kind": "unknown"}, modules=None)
    code.update(commit=(cv or {}).get("commit"), commit_kind=(cv or {}).get("kind"),
                modules_unmatched=(cv or {}).get("modules_unmatched"), commit_before=(cv or {}).get("before"))
    return code


RECIPE_BRIEF = ("model", "epochs", "imgsz", "batch", "lr0", "optimizer")


def _init_brief(init, soup_of=()):
    if soup_of:
        return "soup of %d" % len(soup_of)
    i = init or {}
    k = i.get("kind") or "unknown"
    if k == "row":
        return "row:%s" % str(i.get("model_id") or "")[:12]
    ref = i.get("ref")
    return "%s:%s" % (k, os.path.basename(str(ref))) if ref not in (None, "", True, False) else k


def _recipe_brief(r):
    """model, epochs, imgsz, batch, lr0, optimizer of a row's recipe (INC: the
    run's recipe and its name; Ultralytics: the train args), compact."""
    rec = r.get("recipe") or {}
    src = rec.get("recipe") if isinstance(rec.get("recipe"), dict) else rec
    out = []
    if isinstance(rec.get("recipe_name"), str) and rec.get("recipe_name"):
        out.append("name=%s" % rec["recipe_name"])
    if r["flags"].get("arm"):
        out.append("arm=%s" % r["flags"]["arm"])
    for k in RECIPE_BRIEF:
        v = src.get(k) if isinstance(src, dict) else None
        if v in (None, "") and k == "model":
            v = (r.get("arch") or {}).get("model")
        if v in (None, ""):
            continue
        out.append("%s=%s" % (k, os.path.basename(str(v)) if k == "model" else v))
    return " ".join(out)


def _code_cell(r):
    c = r.get("code") or {}
    kind = c.get("kind") or "unknown"
    return "%s %s" % (kind, str(c["commit"])[:10]) if c.get("commit") else kind


def build_rows(version, conf):
    models = read_models(version)
    prov = read_provenance(version)
    cont = read_contamination(version)
    plan = _read_json(zoo_dir(version) / "plan.json") or {}
    dropped_b = {(d["model_id"], d["exam"]) for d in plan.get("items_b_dropped") or []}
    dropped_b |= {(d["model_id"], d["exam"]) for d in (_read_json(zoo_dir(version) / "shortlist.json") or {}).get(
        "items_c_dropped") or []}
    marks = _task_marks(version)
    claims = conf["shortlist"].get("claims") or []
    retracted = conf["shortlist"].get("retracted") or []
    short = _read_json(zoo_dir(version) / "shortlist.json") or {}
    cv = read_codever(version)
    srules = collections.defaultdict(list)
    for rule, ids in (short.get("rules") or {}).items():
        for i in ids:
            srules[i].append(rule)
    rows = []
    for m in sorted(models, key=lambda m: m["model_id"]):
        mid = m["model_id"]
        sid = m.get("same_weights_as") or mid
        pr, ct = prov.get(mid) or {}, cont.get(mid) or {}
        cmap = m.get("class_map") or {}
        conv = conversion_of(version, sid) if cmap.get("rule") not in (None, "R0") else None
        scores = {e: _cell(version, sid, e, marks, dropped_b) for e in ALL_EXAMS}
        nsp = int(cmap.get("n_species_channels") or 0)
        legacy_join = bool(ct.get("legacy_join"))
        species_level = cmap.get("rule") in ("R0", "R1", "R2", "R3", "R4") and nsp >= 1 \
            and pr.get("species_trained") is not False and not legacy_join
        groups = cmap.get("groups") or []
        have = [CWD12_SPECIES[i] for i in range(12) if i < len(groups) and groups[i]]

        def named(cell):
            pc = cell.get("per_class") or {}
            vals = [pc[s] for s in have if s in pc]
            return sum(vals) / len(vals) if vals else None
        dev, iw, t12 = scores["dev"], scores["imageweeds"], scores["test"]
        cols = {"dev_sp": dev["species_map50_95"] if species_level else None, "dev_ag": dev["agnostic_map50_95"],
                "dev_named": named(dev) if species_level else None,
                "iw_rag": (iw.get("per_class") or {}).get("Ragweed") if species_level and "Ragweed" in have else None,
                "iw_ag": iw["agnostic_map50_95"], "od1_ag": scores[EXAM_OOD]["agnostic_map50_95"],
                "t12_sp_desc": t12["species_map50_95"] if species_level else None,
                "t12_ag_desc": t12["agnostic_map50_95"]}
        ext = {"tv1_ag": scores[EXAM_TV1]["agnostic_map50_95"], "ev1_ag": scores[EXAM_EVG]["agnostic_map50_95"]}
        fin = ct.get("final") or {}
        dev_sel = bool(ct.get("dev_selected"))
        dev_sel_unknown = bool(ct.get("dev_selected_unknown"))
        dev_gated = bool(ct.get("dev_gated") or pr.get("dev_gated"))
        rel = m["rel"]
        cl = [c.get("cite") for c in claims if c.get("match") and re.search(c["match"], rel)]
        rt = [c.get("cite") for c in retracted if c.get("match") and re.search(c["match"], rel)]
        imgsz = pr.get("imgsz_trained")
        rid = (pr.get("inc_run") or "") + " " + json.dumps(pr.get("recipe") or {})
        flags = {"dev": fin.get("dev"), "dev_selected": dev_sel, "dev_selected_unknown": dev_sel_unknown,
                 "dev_gated": dev_gated,
                 "dev_gated_inherited": bool(ct.get("dev_gated_inherited")), "test": fin.get("test"),
                 "test_selected": ct.get("test_selected") or pr.get("test_selected"),
                 "test_selected_inherited": bool(ct.get("test_selected_inherited")),
                 "imageweeds": fin.get("imageweeds"), "test_v1": fin.get(EXAM_TV1), "eval_v1": fin.get(EXAM_EVG),
                 "ooddev_v1": fin.get(EXAM_OOD), "eval_groups": ct.get("eval_groups"),
                 "provenance": (pr.get("data") or {}).get("rating"),
                 "class_map_suspect": bool(species_level and _num(cols["dev_named"]) and _num(cols["dev_ag"])
                                           and cols["dev_named"] < 0.5 * cols["dev_ag"]),
                 "species_not_trained": pr.get("species_trained") is False, "legacy_join": legacy_join,
                 "trained_imgsz_not_640": _num(imgsz) and int(imgsz) != 640, "retracted_claim": bool(rt),
                 "claims": cl + rt, "shortlist": srules.get(mid, []), "notes": list(ct.get("notes") or []),
                 "same_weights_as": m.get("same_weights_as"),
                 "planted_noise": "Bswap" in rid or "Bswap" in json.dumps(pr.get("data") or {}),
                 "inc_kind": pr.get("inc_kind"), "arm": (pr.get("recipe") or {}).get("arm") if isinstance(
                     (pr.get("recipe") or {}).get("arm"), str) else ((pr.get("recipe") or {}).get("arm") or {}).get("id")
                 if isinstance((pr.get("recipe") or {}).get("arm"), dict) else None}
        dev_clean = flags["dev"] == "N" and not dev_sel and not dev_sel_unknown and not dev_gated
        sp12 = species_level and nsp == 12
        fid_bad = fidelity_failed(m, version)
        if not m.get("scorable") or fid_bad:
            section = "unscorable"
        elif species_level:
            section = ("species12" if sp12 else "species_partial") + ("_dev_clean" if dev_clean else "_flagged")
        else:
            section = "agnostic" + ("_dev_clean" if dev_clean else "_flagged")
        row = {
            "model_id": mid, "short_id": mid[:12], "rel": rel, "family": m["family"], "method": m.get("method"),
            "run_dir": m.get("run_dir"), "ckpt_role": m.get("ckpt_role"), "also_role": m.get("also_role") or [],
            "dates": dict(pr.get("dates") or {}, sort=_sortdate(pr, m)),
            "arch": {"model": (m.get("ckpt") or {}).get("train_args", {}).get("model"),
                     "params_m": round(m["params"] / 1e6, 2) if m.get("params") else None,
                     "size_class": m.get("size_class"), "head": (m.get("head") or {}).get("type"),
                     "end2end": (m.get("head") or {}).get("end2end"), "nc_native": (m.get("head") or {}).get("nc")},
            "imgsz_trained": imgsz, "recipe": pr.get("recipe"), "init": pr.get("init"),
            "data": {k: (pr.get("data") or {}).get(k) for k in ("source", "ref", "n_entries", "n_unique", "rating",
                                                                  "reason", "n_unresolved", "n_unreadable")},
            "selected_on": pr.get("selected_on"), "soup_of": [x.get("model_id") or x.get("ref")
                                                              for x in pr.get("soup_of") or []],
            "code": _row_code(pr, cv.get(mid) or cv.get(sid)),
            "class_map": {"rule": cmap.get("rule"), "n_species_channels": nsp, "converted": bool(conv),
                          "converted_sha256": (conv or {}).get("converted_sha256"),
                          "fidelity_ok": (conv or {}).get("ok"),
                          "max_score_diff": ((conv or {}).get("fidelity") or {}).get("max_score_diff"),
                          "reason": cmap.get("reason")},
            "scores": {e: c for e, c in scores.items() if e not in SEALED_EXAMS}, "cols": cols, "external": ext,
            "external_scores": {e: scores[e] for e in SEALED_EXAMS},
            "flags": flags, "contamination": ct.get("counts"), "section": section, "dev_clean": dev_clean,
            "species_level": species_level, "unscorable_reason": m.get("unscorable_reason") or (
                "unscorable_fidelity" if fid_bad else None),
            "rescore_command": "python -m weed_optimizer_framework.tools.inc2.zoo score --version %s --item %s:dev"
                               % (version, sid)}
        row.update(recipe_brief=_recipe_brief(row), init_brief=_init_brief(row["init"], row["soup_of"]))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ select
def _dated_spread(rows, k):
    """k rows evenly spaced by date (selection-free: no score is read)."""
    rows = sorted(rows, key=lambda r: (r["dates"]["sort"], r["model_id"]))
    if len(rows) <= k:
        return rows
    step = (len(rows) - 1) / float(max(1, k - 1))
    return [rows[int(round(i * step))] for i in range(k)]


def select_step(version, conf, conf_sha, shards_c):
    """The shortlist (claims, the top k per family among dev-clean rows by dev,
    filled selection-free when short, the latest per family; at most one per
    run directory; non-INC rows only: an INC row is read on dev alone) and
    stage C's items, cut to what the cap leaves."""
    zd = zoo_dir(version)
    plan = _read_json(zd / "plan.json") or {}
    if plan.get("status") != "plan_ready":
        raise ZooRefused("no plan: plan.json status %r" % plan.get("status"))
    sp = zd / "shortlist.json"
    old = _read_json(sp)
    if isinstance(old, dict) and old.get("format") == SHORTLIST_FORMAT:
        if int(old.get("shards_c") or -1) != int(shards_c):
            raise ZooRefused("shortlist.json was written for --shards-c %s, not %s" % (old.get("shards_c"), shards_c))
        return old
    sl = conf["shortlist"]
    k, cap = int(sl["k"]), int(sl["max"])
    rows = [r for r in build_rows(version, conf) if not is_inc_family(r["family"]) and r["section"] != "unscorable"
            and not r["flags"].get("same_weights_as")]
    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    chosen = collections.OrderedDict()
    used_dirs = set()

    def add(r, rule):
        if r["model_id"] in chosen or r["run_dir"] in used_dirs:
            return False
        chosen[r["model_id"]] = {"rule": rule, "family": r["family"], "row": r}
        used_dirs.add(r["run_dir"])
        return True
    for r in sorted(rows, key=lambda r: r["model_id"]):
        if r["flags"]["claims"]:
            add(r, "claims")

    def key(r):
        if r["section"].startswith("species12"):
            v = r["cols"]["dev_sp"]
        else:
            v = r["cols"]["dev_ag"]
        return (-(v if _num(v) else -1.0), r["model_id"])
    for fam in sorted(by_fam):
        clean = sorted((r for r in by_fam[fam] if r["dev_clean"] and _num(r["cols"]["dev_ag"])), key=key)
        n = 0
        for r in clean:
            if n >= k:
                break
            n += add(r, "top_dev_clean")
        if n < k:
            rest = [r for r in by_fam[fam] if r["model_id"] not in chosen and r["run_dir"] not in used_dirs]
            for r in _dated_spread(rest, k - n):
                n += add(r, "fill_by_date")
    for fam in sorted(by_fam):
        latest = sorted(by_fam[fam], key=lambda r: (r["dates"]["sort"], r["model_id"]))
        for r in reversed(latest):
            if add(r, "latest"):
                break
    for drop_rule in ("latest", "fill_by_date"):
        while len(chosen) > cap:
            fams = collections.Counter(v["family"] for v in chosen.values())
            cand = [(mid, v) for mid, v in chosen.items() if v["rule"] == drop_rule]
            if not cand:
                break
            top = max(fams[v["family"]] for _m, v in cand)
            fam = sorted({v["family"] for _m, v in cand if fams[v["family"]] == top})[0]
            mid = sorted(m_ for m_, v in cand if v["family"] == fam)[-1]
            chosen.pop(mid)
    rule_rank = {"claims": 0, "top_dev_clean": 1, "fill_by_date": 2, "latest": 3}
    exams = read_exams(version)["exams"]
    rates = plan.get("rates") or {}
    margin = float(plan.get("margin") or conf["stages"]["margin"])
    models = {m["model_id"]: m for m in read_models(version)}
    items = []
    order = sorted(chosen.items(), key=lambda kv: (rule_rank[kv[1]["rule"]], kv[1]["family"], kv[0]))
    for mid, v in order:
        m = models[mid]
        got = scorable(m, version)
        if got is None:
            continue
        for e in sl["exam_order"]:
            if record_of(version, mid, e)[0] is not None:
                continue
            r = (rates.get(e) or {}).get(m.get("size_class") or "L") or max(
                list((rates.get(e) or {}).values()) or [0.1])
            items.append({"model_id": mid, "exam": e, "file": got[0], "file_sha256": got[1],
                          "conversion": _conv_brief(got[2]), "size_class": m.get("size_class"),
                          "predicted_s": round(float(r) * int(exams[e]["n_images"]) * margin, 2)})
    spent = ledger_hours(version, own="select")
    budget_c = float(plan.get("max_gpu_hours") or 40) - float(spent.get("total", 0.0)) - \
        float(conf["stages"]["reserve_report"])
    take, tot = [], 0.0
    for it in items:
        h = it["predicted_s"] / 3600.0
        if tot + h <= budget_c:
            take.append(it)
            tot += h
    bins = _lpt(take, shards_c)
    _write_shards(version, "c", bins, shards_c)
    rec = {"format": SHORTLIST_FORMAT, "version": version, "config_sha256": conf_sha, "shards_c": int(shards_c),
           "rules": {r: [mid for mid, v in chosen.items() if v["rule"] == r] for r in rule_rank},
           "ids": list(chosen), "cut": {"max": cap, "kept": len(chosen)}, "budget_c_h": round(budget_c, 4),
           "spent_h": spent, "items_c": len(take), "items_c_not_scored_budget": len(items) - len(take),
           "items_c_dropped": [{"model_id": it["model_id"], "exam": it["exam"], "why": "not_scored_budget"}
                               for it in items if it not in take],
           "created_utc": _utc()}
    _write_json(sp, rec)
    log("select: %d shortlisted, %d stage-C items (%d left out by the cap), budget %.2f GPU-h"
        % (len(chosen), len(take), len(items) - len(take), budget_c))
    return rec


# ------------------------------------------------------------------ report
CSV_COLUMNS = ("short_id", "family", "ckpt", "date_end", "arch", "params_m", "imgsz_trained", "data_ref",
               "n_train_unique", "provenance", "map_rule", "converted", "dev_sp", "dev_ag", "dev_named", "iw_rag",
               "iw_ag", "od1_ag", "t12_sp_desc", "t12_ag_desc", "flag_dev", "flag_dev_sel", "flag_dev_gated",
               "flag_test", "flag_test_sel", "flag_iw", "flag_tv1", "flag_ev1", "legacy_join", "suspect", "section",
               "rank", "method", "recipe", "init", "code", "code_commit", "rel", "model_id")
EXTERNAL_COLUMNS = ("short_id", "family", "ckpt", "date_end", "tv1_ag", "ev1_ag", "flag_tv1", "flag_ev1", "rel",
                    "model_id")
SECTION_ORDER = ("species12_dev_clean", "species_partial_dev_clean", "agnostic_dev_clean", "species12_flagged",
                 "species_partial_flagged", "agnostic_flagged", "unscorable")


def _f(x, d=4):
    return ("%." + str(d) + "f") % x if _num(x) else ""


def _status_cell(row, exam, col):
    c = (row["scores"].get(exam) or row["external_scores"].get(exam) or {})
    v = row["cols"].get(col) if col in row["cols"] else row["external"].get(col)
    if _num(v):
        return v
    return {"nb": "nb", "nt": "nt", "ref": "ref", "err": "err"}.get(c.get("status"), "")


def _sections(rows, conf):
    fo = family_order(conf)
    out = {k: [] for k in SECTION_ORDER}
    for r in rows:
        out.setdefault(r["section"], []).append(r)

    def ranked(rs, keys):
        have = [r for r in rs if _num(r["cols"]["dev_ag"])]
        none = sorted((r for r in rs if not _num(r["cols"]["dev_ag"])), key=lambda r: r["model_id"])
        have.sort(key=lambda r: tuple(-(r["cols"][k] if _num(r["cols"][k]) else -1.0) for k in keys) + (r["model_id"],))
        return have + none

    def unranked(rs):
        have = [r for r in rs if _num(r["cols"]["dev_ag"])]
        none = [r for r in rs if not _num(r["cols"]["dev_ag"])]
        k = lambda r: (fo.get(r["family"], 999), r["dates"]["sort"], r["model_id"])  # noqa: E731
        return sorted(have, key=k) + sorted(none, key=k)
    res = {}
    for name, rs in out.items():
        if name in ("species12_dev_clean", "species_partial_dev_clean"):
            res[name] = ranked(rs, ("dev_sp", "dev_ag"))
        elif name == "agnostic_dev_clean":
            res[name] = ranked(rs, ("dev_ag",))
        elif name == "unscorable":
            res[name] = sorted(rs, key=lambda r: (fo.get(r["family"], 999), r["rel"]))
        else:
            res[name] = unranked(rs)
    return res


def _flag_string(r):
    f = r["flags"]

    def sel(s):
        return "S" if s in TEST_CHOSEN else ""
    s = "dev:%s%s%s test:%s%s iw:%s tv1:%s ev1:%s" % (
        f.get("dev") or "-", "+sel" if f.get("dev_selected") else ("+sel?" if f.get("dev_selected_unknown") else ""),
        "+gated" if f.get("dev_gated") else "",
        sel(f.get("test_selected")) + ("s" if f.get("test_selected_inherited") else ""), f.get("test") or "-",
        f.get("imageweeds") or "-", f.get("test_v1") or "-", f.get("eval_v1") or "-")
    extra = [k for k in ("legacy_join", "class_map_suspect", "species_not_trained", "planted_noise",
                         "retracted_claim") if f.get(k)]
    if f.get("same_weights_as"):
        extra.append("same_weights_as %s" % f["same_weights_as"][:12])
    return s + (" [%s]" % ", ".join(extra) if extra else "")


def _md_table(rows, ranked, with_rank=True):
    head = ("| %sid | family | ckpt | date | arch@imgsz | train imgs (rating) | map | dev sp | dev ag | dev named | "
            "IW Rag | IW ag | OOD ag | cwd12 test sp* | cwd12 test ag* | code | flags |" % ("# | " if with_rank else ""))
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for i, r in enumerate(rows, 1):
        c = r["cols"]
        imgsz = r.get("imgsz_trained")
        dag = "%s%s" % (_status_cell(r, "dev", "dev_ag") if not _num(c["dev_ag"]) else _f(c["dev_ag"]),
                        "" if r["dev_clean"] or not ranked else "")
        out.append("| %s%s | %s | %s | %s | %s@640%s | %s (%s) | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            ("%d | " % i if (ranked and _num(c["dev_ag"])) else "— | ") if with_rank else "",
            r["short_id"], r["family"], r["ckpt_role"] + ("+" + "+".join(r["also_role"]) if r["also_role"] else ""),
            str(r["dates"].get("end") or r["dates"].get("start_utc") or "")[:10],
            (r["arch"].get("head") or "?"), "" if not _num(imgsz) or int(imgsz) == 640 else " †%s" % imgsz,
            "%s%s" % (r["data"].get("n_unique") if r["data"].get("n_unique") is not None else "?",
                      " +%d unread" % (int(r["data"].get("n_unresolved") or 0) + int(r["data"].get("n_unreadable") or 0))
                      if (r["data"].get("n_unresolved") or r["data"].get("n_unreadable")) else ""),
            r["data"].get("rating"),
            r["class_map"]["rule"] or "-", " (conv)" if r["class_map"]["converted"] else "",
            _f(c["dev_sp"]) if _num(c["dev_sp"]) else ("n/i" if r["flags"].get("legacy_join") else ""),
            dag, _f(c["dev_named"]), _f(c["iw_rag"]), _status_cell(r, "imageweeds", "iw_ag") if not _num(c["iw_ag"])
            else _f(c["iw_ag"]), _status_cell(r, EXAM_OOD, "od1_ag") if not _num(c["od1_ag"]) else _f(c["od1_ag"]),
            _f(c["t12_sp_desc"]), _status_cell(r, "test", "t12_ag_desc") if not _num(c["t12_ag_desc"])
            else _f(c["t12_ag_desc"]), _code_cell(r), _flag_string(r)))
    return out


def _csv_row(r, rank):
    c, f = r["cols"], r["flags"]

    def val(x):
        return _f(x) if _num(x) else ("" if x is None else str(x))
    return [r["short_id"], r["family"], r["ckpt_role"], str(r["dates"].get("end") or ""), r["arch"].get("head") or "",
            val(r["arch"].get("params_m")), val(r.get("imgsz_trained")), str(r["data"].get("ref") or ""),
            val(r["data"].get("n_unique")), str(r["data"].get("rating") or ""), str(r["class_map"]["rule"] or ""),
            "1" if r["class_map"]["converted"] else "", val(_status_cell(r, "dev", "dev_sp")),
            val(_status_cell(r, "dev", "dev_ag")), val(c["dev_named"]), val(c["iw_rag"]),
            val(_status_cell(r, "imageweeds", "iw_ag")), val(_status_cell(r, EXAM_OOD, "od1_ag")),
            val(c["t12_sp_desc"]), val(_status_cell(r, "test", "t12_ag_desc")), f.get("dev") or "",
            "1" if f.get("dev_selected") else ("?" if f.get("dev_selected_unknown") else ""),
            "1" if f.get("dev_gated") else "", f.get("test") or "",
            f.get("test_selected") or "", f.get("imageweeds") or "", f.get("test_v1") or "", f.get("eval_v1") or "",
            "1" if f.get("legacy_join") else "", "1" if f.get("class_map_suspect") else "", r["section"],
            str(rank or ""), str(r.get("method") or ""), r.get("recipe_brief") or "", r.get("init_brief") or "",
            str((r.get("code") or {}).get("kind") or ""), str((r.get("code") or {}).get("commit") or ""), r["rel"],
            r["model_id"]]


def _write_csv(path, header, rows):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    return _atomic_bytes(path, buf.getvalue().encode("utf-8"))


def _completion(version):
    """(status, failed shards, items without a final record, counts) of the chain."""
    planned = _planned(version)
    marks = _task_marks(version)
    missing, n = [], collections.Counter()
    for (mid, exam), (stage, idx) in sorted(planned.items()):
        kind, _r = record_of(version, mid, exam)
        if kind is not None:
            n[{"score": "scored"}.get(kind, kind)] += 1
            continue
        mk = marks.get((mid, exam))
        if mk == "not_scored_budget":
            n["not_scored_budget"] += 1
            continue
        n[mk or "missing"] += 1
        missing.append({"model_id": mid, "exam": exam, "stage": stage, "shard": idx, "status": mk or "no record"})
    failed = []
    shards = {"a": {"n": 0, "done": 0, "failed": []}, "c": {"n": 0, "done": 0, "failed": []}}
    for p in sorted((zoo_dir(version) / "shards").glob("*.json")) if (zoo_dir(version) / "shards").is_dir() else []:
        sh = _read_json(p) or {}
        stg = sh.get("stage")
        task = "pilot" if stg == "pilot" else "%s_%03d" % (stg, int(sh.get("index") or 0))
        t = _read_json(zoo_dir(version) / "tasks" / ("%s.json" % task)) or {}
        if stg in shards:
            shards[stg]["n"] += 1
        if t.get("status") == "done" or not sh.get("items"):
            if stg in shards:
                shards[stg]["done"] += 1
        else:
            failed.append(task)
            if stg in shards:
                shards[stg]["failed"].append(int(sh.get("index") or 0))
    status = "complete" if not missing and not failed else "partial"
    return status, failed, missing, dict(n), shards


def report_step(version, conf, conf_sha):
    """report.{json,csv,md} for people, report_external.{json,csv} (the sealed
    test v1 and evaluation-group reads) and the platform record."""
    zd = zoo_dir(version)
    plan = _read_json(zd / "plan.json")
    if not isinstance(plan, dict) or plan.get("status") != "plan_ready":
        update_record(version, status="failed", refusal="no plan: nothing to report")
        raise ZooRefused("no plan.json: nothing to report (record failed)")
    rows = build_rows(version, conf)
    secs = _sections(rows, conf)
    status, failed, missing, ncount, shards = _completion(version)
    inv = _read_json(zd / "inventory" / "summary.json") or {}
    exams = read_exams(version)
    lock_sha = _sha_file(root_dir(version) / "splits" / "v1" / "LOCK.json")
    # meta's decisions plus the convert step's (a failed conversion is unscorable_fidelity, not keep)
    counts = final_counts(version, rows)
    conv_rules = collections.Counter(r["class_map"]["rule"] for r in rows
                                     if r["class_map"]["converted"] and r["class_map"]["fidelity_ok"])
    items = {}
    for e in ALL_EXAMS:
        cnt = collections.Counter()
        for r in rows:
            if r["flags"].get("same_weights_as"):
                continue
            st = (r["scores"].get(e) or r["external_scores"].get(e) or {}).get("status")
            if st:
                cnt[{"nb": "not_scored_budget", "nt": "not_scored_time", "ref": "refused", "err": "error"}.get(st, st)] += 1
        items[e] = dict(cnt)
    fams = []
    fo = family_order(conf)
    for fam in sorted({r["family"] for r in rows}, key=lambda f: fo.get(f, 999)):
        rs = [r for r in rows if r["family"] == fam]

        def stat3(k):
            v = sorted(r["cols"][k] for r in rs if _num(r["cols"].get(k)))
            return {"min": v[0], "median": v[len(v) // 2], "max": v[-1], "n": len(v)} if v else None
        fams.append({"family": fam, "method": family_method(conf, fam), "n_models": len(rs),
                     "date_first": min((r["dates"]["sort"] for r in rs if r["dates"]["sort"]), default=None),
                     "date_last": max((r["dates"]["sort"] for r in rs if r["dates"]["sort"]), default=None),
                     "dev_species": stat3("dev_sp"), "dev_agnostic": stat3("dev_ag"),
                     "flags": {e: dict(collections.Counter(r["flags"].get(e) for r in rs))
                               for e in ("dev", "test", "imageweeds", "test_v1", "eval_v1")}})
    short = _read_json(zd / "shortlist.json") or {}
    codever = _read_json(zd / ("codever_%s.json" % version))
    rep = {"format": REPORT_FORMAT, "version": version, "created_utc": _utc(), "status": status, "usage": USAGE_RULE,
           "inputs": {"config_sha256": conf_sha, "list_sha256": inv.get("list_sha256"), "lock_sha256": lock_sha,
                      "scorer_sha256": (_read_json(root_dir(version) / "splits" / "v1" / "LOCK.json") or {}).get(
                          "scorer_sha256"),
                      "codever_sha256": _sha_file(zd / ("codever_%s.json" % version)) if codever else None,
                      "exams": exams["exams"]},
           "counts": {"files_listed": inv.get("files_listed"), "inc_glob_added": inv.get("inc_glob_added"),
                      "by_decision": counts.get("by_decision"), "skipped_by_reason": counts.get("skipped_by_reason"),
                      "unscorable_by_reason": counts.get("unscorable_by_reason"),
                      "converted_by_rule": dict(conv_rules), "items": items, "completion": ncount,
                      "out_of_scope": conf.get("out_of_scope")},
           "failed_shards": failed, "items_without_record": missing[:500],
           "rows": [dict(r, external=None, external_scores=None) for r in rows],
           "sections": {k: [r["model_id"] for r in v] for k, v in secs.items()},
           "families": fams, "shortlist": {"rules": short.get("rules"), "ids": short.get("ids")},
           "per_source": None}
    _write_json(zd / "report.json", rep)
    ranked_secs = ("species12_dev_clean", "species_partial_dev_clean", "agnostic_dev_clean")
    csv_rows = []
    for name in SECTION_ORDER:
        for i, r in enumerate(secs.get(name) or [], 1):
            csv_rows.append(_csv_row(r, i if name in ranked_secs and _num(r["cols"]["dev_ag"]) else None))
    _write_csv(zd / "report.csv", CSV_COLUMNS, csv_rows)
    ext = {"format": REPORT_FORMAT + "+external", "version": version, "created_utc": _utc(), "usage": USAGE_RULE,
           "sealed": "test v1 and the five test groups stay the frozen test of every model trained after Z1: no "
                     "recipe, data or model choice may cite these values (Amendment Z1, usage rule 4)",
           "rows": [{"model_id": r["model_id"], "rel": r["rel"], "family": r["family"], "external": r["external"],
                     "scores": r["external_scores"], "flags": {k: r["flags"].get(k) for k in ("test_v1", "eval_v1",
                                                                                                "eval_groups")}}
                    for r in rows]}
    _write_json(zd / "report_external.json", ext)
    _write_csv(zd / "report_external.csv", EXTERNAL_COLUMNS,
               [[r["short_id"], r["family"], r["ckpt_role"], str(r["dates"].get("end") or ""),
                 _f(r["external"]["tv1_ag"]) or (r["external_scores"][EXAM_TV1].get("status") or ""),
                 _f(r["external"]["ev1_ag"]) or (r["external_scores"][EXAM_EVG].get("status") or ""),
                 r["flags"].get("test_v1") or "", r["flags"].get("eval_v1") or "", r["rel"], r["model_id"]]
                for r in rows])
    md = _report_md(version, rep, secs, exams, conf)
    _atomic_bytes(zd / "report.md", md.encode("utf-8"))
    rec = platform_counts(version, rows, counts, ncount, shards, plan, lock_sha, conf_sha, inv)
    update_record(version, status=status, **rec)
    log("report: %s; %d rows; %d items without a record; failed shards %s" % (status, len(rows), len(missing),
                                                                              failed[:8]))
    return rep


def platform_counts(version, rows, counts, ncount, shards, plan, lock_sha, conf_sha, inv):
    """The platform record's counts (totals only: never keyed by an exam name)."""
    bd = counts.get("by_decision") or {}
    led = ledger_hours(version)
    return {"counts": {"files_listed": inv.get("files_listed"), "kept": bd.get("keep", 0),
                       "unscorable": bd.get("unscorable", 0), "skipped": bd.get("skip", 0),
                       "skipped_by_reason": counts.get("skipped_by_reason") or {},
                       "unscorable_by_reason": counts.get("unscorable_by_reason") or {},
                       "converted": sum(1 for r in rows if r["class_map"]["converted"]
                                        and r["class_map"]["fidelity_ok"]),
                       "items_planned": sum(ncount.values()), "items_scored": ncount.get("scored", 0) + ncount.get(
                           "pilot", 0), "items_reused": sum(1 for r in rows for c in list(r["scores"].values())
                                                            + list(r["external_scores"].values())
                                                            if c.get("status") == "reused"),
                       "items_refused": ncount.get("refused", 0), "items_error": ncount.get("error", 0)
                       + ncount.get("error_final", 0), "items_not_scored_budget": ncount.get("not_scored_budget", 0)
                       + len(plan.get("items_b_dropped") or []) + len((_read_json(zoo_dir(version) / "shortlist.json")
                                                                       or {}).get("items_c_dropped") or []),
                       "items_not_scored_time": ncount.get("not_scored_time", 0)},
            "shards": shards,
            "gpu_hours": {"cap": plan.get("max_gpu_hours"), "inventory": led.get("inventory", 0.0),
                          "tasks": round(led.get("a", 0.0) + led.get("c", 0.0), 4), "total": led.get("total", 0.0),
                          "predicted": plan.get("predicted_h")},
            "sha256": {"config": conf_sha, "list": inv.get("list_sha256"), "lock": lock_sha,
                       "report_json": _sha_file(zoo_dir(version) / "report.json")}}


def _report_md(version, rep, secs, exams, conf):
    L = ["# Model zoo %s: every detector under one protocol (descriptive)" % version, "",
         "> " + USAGE_RULE, "", "Status: **%s** (%s)." % (rep["status"], rep["created_utc"]), "",
         "## Inputs", "",
         "- config sha256 `%s`, inventory list sha256 `%s`" % (rep["inputs"]["config_sha256"],
                                                             rep["inputs"]["list_sha256"]),
         "- zoo LOCK sha256 `%s`; locked scorer (inc/scorer.py) sha256 `%s`" % (rep["inputs"]["lock_sha256"],
                                                                             rep["inputs"]["scorer_sha256"]),
         "- code versions: %s; each row's commit is in its code column and in Table F" % (
             "codever_%s.json sha256 `%s`" % (version, rep["inputs"]["codever_sha256"])
             if rep["inputs"]["codever_sha256"] else "no codever_%s.json yet (INC rows exact by module sha256s, "
             "others approximate by start date; `zoo codever --git <repo> --report` adds the commits)" % version), "",
         "| exam | images | boxes | role | labels | not read |", "|---|---|---|---|---|---|"]
    for e in ALL_EXAMS:
        x = exams["exams"][e]
        L.append("| %s | %d | %d | %s%s | %s | %s |" % (e, x["n_images"], x["n_boxes"], x["role"],
                                                          ", agnostic only" if x["agnostic_only"] else "", x["labels"],
                                                          "; ".join("%s: %s" % kv for kv in sorted(
                                                              (x.get("not_read") or {}).items())) or "-"))
    c = rep["counts"]
    L += ["", "## What was read", "",
          "Files listed: %s (+%s INC runs the fixed-depth glob found). Decisions: %s." % (
              c["files_listed"], c["inc_glob_added"], json.dumps(c["by_decision"], sort_keys=True)), "",
          "| skipped: reason | files |", "|---|---|"]
    for k in SKIP_REASONS:
        L.append("| %s | %d |" % (k, int((c["skipped_by_reason"] or {}).get(k) or 0)))
    L += ["", "| unscorable: reason | files |", "|---|---|"]
    for k in UNSCORABLE_REASONS:
        L.append("| %s | %d |" % (k, int((c["unscorable_by_reason"] or {}).get(k) or 0)))
    L += ["", "Converted by rule: %s." % (json.dumps(c["converted_by_rule"], sort_keys=True) or "{}"),
          "Out of scope (not .pt checkpoints of the inventory list): %s." % json.dumps(c.get("out_of_scope") or {}),
          ""]
    by = {r["model_id"]: r for r in rep["rows"]}
    titles = {"species12_dev_clean": "Table A. Twelve-species detectors, dev-clean, ordered by dev species mAP50-95",
              "species_partial_dev_clean": "Table A2. Partial-species detectors (fewer than 12 species channels), "
                                           "dev-clean, ordered by dev species mAP50-95 (a missing species scores 0)",
              "species12_flagged": "Table B. Twelve-species detectors whose training could include dev, or that "
                                   "were selected on dev or by dev gates (unranked; family, date)",
              "species_partial_flagged": "Table B2. Partial-species detectors, flagged (unranked)",
              "agnostic_dev_clean": "Table C. Agnostic-only detectors, dev-clean, ordered by dev agnostic mAP50-95",
              "agnostic_flagged": "Table D. Agnostic-only detectors, flagged (unranked)"}
    for name in ("species12_dev_clean", "species_partial_dev_clean", "species12_flagged", "species_partial_flagged",
                 "agnostic_dev_clean", "agnostic_flagged"):
        rows = [by[i] for i in rep["sections"].get(name) or []]
        L += ["", "## %s" % titles[name], ""]
        if not rows:
            L.append("(none)")
            continue
        ranked = name.endswith("dev_clean")
        L += _md_table([dict(r, scores=r["scores"], external_scores={e: {} for e in SEALED_EXAMS}, external={})
                        for r in rows], ranked, with_rank=ranked)
    L += ["", "## Table E. Unscorable detectors", "", "| id | family | rel | reason |", "|---|---|---|---|"]
    for i in rep["sections"].get("unscorable") or []:
        r = by[i]
        L.append("| %s | %s | %s | %s |" % (r["short_id"], r["family"], r["rel"], r.get("unscorable_reason")))
    L += ["", "## Table F. Method, recipe, init and code of every row", "",
          "| id | family | method | ckpt | recipe | init | data | code | rel |", "|---|---|---|---|---|---|---|---|---|"]
    fo = family_order(conf)
    for r in sorted(rep["rows"], key=lambda r: (fo.get(r["family"], 999), r["dates"]["sort"], r["model_id"])):
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["short_id"], r["family"], r.get("method") or "", r["ckpt_role"], r.get("recipe_brief") or "-",
            r.get("init_brief") or "-", "%s (%s)" % (r["data"].get("source") or "-", r["data"].get("rating") or "-"),
            _code_cell(r), r["rel"]))
    L += ["", "## Families", "", "| family | method | models | dates | dev sp (min / median / max) | "
          "dev ag (min / median / max) | dev flags |", "|---|---|---|---|---|---|---|"]
    for f in rep["families"]:
        def s3(x):
            return "%s / %s / %s" % (_f(x["min"]), _f(x["median"]), _f(x["max"])) if x else "-"
        L.append("| %s | %s | %d | %s .. %s | %s | %s | %s |" % (
            f["family"], f["method"], f["n_models"], str(f["date_first"] or "")[:10], str(f["date_last"] or "")[:10],
            s3(f["dev_species"]), s3(f["dev_agnostic"]), json.dumps(f["flags"]["dev"], sort_keys=True)))
    L += ["", "## Sealed reads", "",
          "Test v1 and the five test groups' reads of the shortlist and of every non-INC row are in "
          "report_external.json and report_external.csv only (usage rule 4): they are never quoted, ranked or "
          "used for a choice.", "",
          "## Legend", "",
          "- flags: `dev:Y test:S+Y iw:N tv1:N ev1:P`: Y (an exact or near copy of an exam image in the training "
          "list), P (possible: hits on a list rebuilt after the run or a derived superset, or more than 1 % of the "
          "list unhashed), U (no list, an unknown init, or no hit on a list with training entries it could not "
          "read: `+N unread` beside its image count, relative lines of a .txt list or an unreadable list or "
          "directory), N (a complete, non-empty list with no hit); `+sel` selected on dev (best.pt, or last.pt of "
          "an early stop, on a val set staged from dev or whose list holds dev images), `+sel?` its dev selection "
          "unknown (a val set, its own split or one the zoo cannot place, that could not be listed or read whole, "
          "or a last.pt without results.csv on a val set holding dev images; not dev-clean); `+gated` dev-gated "
          "(its data or init chosen by a "
          "dev gate: an INC candidate, a stream pool, any row initialised from an INC row or from a dev-gated or "
          "dev-selected row); S selected on the cwd12 test (best.pt, an early stop, or a val set holding test "
          "images), s the same through its init.",
          "- `*` cwd12 test is descriptive: the sealed test, the selection set from 03-15 to 09-26.",
          "- `†N`: read at 640, trained at N. The native-resolution reads of the measurement arms (L23N) are in "
          "capacity/native_v1.json and their runs' scores/dev@<imgsz>.json.",
          "- map: R0 the INC class space; R1 cwd12 id order; R2 trainer slots (+ aux -> OtherPlant); R3 eight "
          "trainer slots + novel names; R4 names resolved one by one; R5 every channel read as weed (agnostic "
          "only). `(conv)`: read through a max-merged 13-channel copy of the head (fidelity checked).",
          "- n/i: species columns not interpretable (legacy_join: a pre-2026-09-21 head whose training joined "
          "external or holdout boxes into legacy slots).",
          "- Agnostic AP collapses every class: a multi-class head keeps cross-class duplicates at different anchors "
          "after per-class NMS, which a one-class head does not have; agnostic reads favour one-class heads.",
          "- planted_noise: a pilot run trained on the Bswap increment (40 % of its boxes relabelled on purpose).",
          "- code: `exact <commit>` (an INC row: the commit holding every module sha256 its run recorded), `approx "
          "<commit>` (main's last commit before the run started), or the kind alone before `zoo codever` ran.",
          "", FOOTER, ""]
    return "\n".join(L)


# ------------------------------------------------------------------ status
def status_cmd(version):
    zd = zoo_dir(version)
    out = {"record": read_record(version), "steps": {}, "scores": 0, "tasks": {}}
    d = zd / "steps"
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        out["steps"][p.stem] = (_read_json(p) or {}).get("status")
    sd = zd / "scores"
    out["scores"] = sum(1 for _ in sd.glob("*/*/*.json")) if sd.is_dir() else 0
    td = zd / "tasks"
    for p in sorted(td.glob("*.json")) if td.is_dir() else []:
        t = _read_json(p) or {}
        out["tasks"][p.stem] = {"status": t.get("status"), "counts": t.get("counts")}
    out["gpu_hours"] = ledger_hours(version)
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return out


# ------------------------------------------------------------------ submit
SUBMIT_LOCK_STALE_S = 900


def _exe(var, default):
    return os.environ.get(var) or default


def queued_zoo_jobs(version):
    """[(id, name)] of this user's queued or running zoo jobs of `version`,
    or raises ZooRefused when squeue cannot be read (a duplicate cannot be
    ruled out)."""
    names = {n % version for n in ZOO_JOB_NAMES}
    try:
        p = subprocess.run([_exe("ZOO_SQUEUE", "squeue"), "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j"],
                           capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ZooRefused("squeue could not be run (%s): a zoo job cannot be ruled out" % e)
    if p.returncode != 0:
        raise ZooRefused("squeue exited %d: a zoo job cannot be ruled out" % p.returncode)
    out = []
    for ln in p.stdout.splitlines():
        parts = [x.strip() for x in ln.split("|")]
        if len(parts) >= 2 and parts[1] in names:
            out.append((parts[0], parts[1]))
    return out


def _nested_script():
    return Path(C.REPO) / "weed_llm_benchmark" / "run_inc2_zoo.sh"


def submit_cmd(version, shards_a, shards_c, concurrency, max_gpu_hours):
    """The held-free chain of five GPU-shared jobs (module docstring; the
    amendment's platform flow). Writes capacity/zoo_<version>.json
    (submitting as each id is obtained, then submitted) and prints
    `INCZOO {"job_ids": [...], "names": [...]}` last."""
    if os.environ.get("SLURM_JOB_ID"):
        raise ZooRefused("submit runs on the login node, not inside a Slurm job")
    conf, conf_sha = load_config(version)
    for name, v, lo, hi in (("--shards-a", shards_a, 1, 200), ("--shards-c", shards_c, 1, 200),
                            ("--concurrency", concurrency, 1, 8), ("--max-gpu-hours", max_gpu_hours, 1, 45)):
        if not lo <= int(v) <= hi:
            raise ZooRefused("%s %s is outside %d..%d" % (name, v, lo, hi))
    script = Path(os.environ.get("ZOO_SCRIPT") or _nested_script())
    if not script.is_file() or os.path.realpath(str(script)) != os.path.realpath(str(_nested_script())):
        raise ZooRefused("%s is not the git-tracked copy %s: submit only that script" % (script, _nested_script()))
    q = queued_zoo_jobs(version)
    if q:
        raise ZooRefused("a zoo job of %s is queued or running (%s): nothing submitted" % (version, q[:5]))
    prev = read_record(version)
    resumed = None
    if prev and prev.get("status") == "complete":
        raise ZooRefused("capacity/zoo_%s.json says complete: the audit ran; a new run is a new version" % version)
    if prev and prev.get("status") in LIVE_STATUSES + ("partial", "failed", "refused"):
        resumed = {"status": prev.get("status"), "jobs": prev.get("jobs"), "updated_utc": prev.get("updated_utc")}
    zd = zoo_dir(version)
    plan = _read_json(zd / "plan.json") or {}
    if plan.get("shards_a") not in (None, int(shards_a)):
        raise ZooRefused("plan.json was written for --shards-a %s: resubmit with it" % plan.get("shards_a"))
    sl = _read_json(zd / "shortlist.json") or {}
    if sl.get("shards_c") not in (None, int(shards_c)):
        raise ZooRefused("shortlist.json was written for --shards-c %s: resubmit with it" % sl.get("shards_c"))
    lock = record_path(version).with_name(".zoo_%s.submit.lock" % version)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        age = time.time() - (_mtime(lock) or time.time())
        if age < SUBMIT_LOCK_STALE_S:
            raise ZooRefused("another submission of %s holds %s (%.0f s old)" % (version, lock, age))
        lock.unlink()
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(fd, ("%d %s\n" % (os.getpid(), _utc())).encode("utf-8"))
    os.close(fd)
    old_bytes = record_path(version).read_bytes() if record_path(version).is_file() else None
    S = str(script)
    sb = _exe("ZOO_SBATCH", "sbatch")
    names = [n % version for n in ZOO_JOB_NAMES]
    ids = []
    na, nc, p = int(shards_a), int(shards_c), int(concurrency)
    common = {"decided_by": os.environ.get("INCAP_DECIDED_BY") or None,
              "approval_id": os.environ.get("INCAP_APPROVAL_ID") or None,
              "trigger": [t for t in (os.environ.get("INCAP_TRIGGER") or "").split(",") if t]}
    plan_argv = [
        [sb, "--parsable", "--job-name=%s" % names[0], "-p", "GPU-shared", "--time=12:00:00", S, "inventory",
         "--version", version, "--shards-a", str(na), "--shards-c", str(nc), "--max-gpu-hours", str(max_gpu_hours)],
        [sb, "--parsable", "--job-name=%s" % names[1], "-p", "GPU-shared", "--time=04:00:00",
         "--array=0-%d%%%d" % (na - 1, p), "--dependency=afterok:{0}", "--kill-on-invalid-dep=yes", S, "score",
         "--version", version, "--stage", "a"],
        [sb, "--parsable", "--job-name=%s" % names[2], "-p", "GPU-shared", "--time=01:00:00",
         "--dependency=afterany:{1}", "--kill-on-invalid-dep=yes", S, "select", "--version", version,
         "--shards-c", str(nc)],
        [sb, "--parsable", "--job-name=%s" % names[3], "-p", "GPU-shared", "--time=04:00:00",
         "--array=0-%d%%%d" % (nc - 1, p), "--dependency=afterok:{2}", "--kill-on-invalid-dep=yes", S, "score",
         "--version", version, "--stage", "c"],
        [sb, "--parsable", "--job-name=%s" % names[4], "-p", "GPU-shared", "--time=02:00:00",
         "--dependency=afterany:{3}", "--kill-on-invalid-dep=yes", S, "report", "--version", version]]
    try:
        for i, argv in enumerate(plan_argv):
            argv = [a.format(*ids) if "{" in a else a for a in argv]
            try:
                pr = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise RuntimeError("sbatch of %s could not be run: %s" % (names[i], e))
            jid = ((pr.stdout or "").strip().splitlines() or [""])[-1].split(";")[0].strip()
            if pr.returncode != 0 or not re.fullmatch(r"[0-9]+", jid):
                raise RuntimeError("sbatch of %s exited %d: %s" % (names[i], pr.returncode,
                                                                    (pr.stderr or pr.stdout or "")[-300:]))
            ids.append(jid)
            update_record(version, status="submitting", jobs=dict(zip(("inventory", "score_a", "select", "score_c",
                                                                      "report"), ids)), job_names=names,
                          submitted_utc=_utc(), resumed_from=resumed, **common)
    except RuntimeError as e:
        for j in ids:
            subprocess.run([_exe("ZOO_SCANCEL", "scancel"), j], capture_output=True, text=True, timeout=120)
        if old_bytes is None:
            if record_path(version).exists():
                record_path(version).unlink()
        else:
            _atomic_bytes(record_path(version), old_bytes)
        lock.unlink()
        log("ERROR: %s; cancelled %s; no record written" % (e, ids or "nothing"))
        raise SubmitFailed(str(e))
    rec = update_record(version, status="submitted", jobs=dict(zip(("inventory", "score_a", "select", "score_c",
                                                                    "report"), ids)), job_names=names,
                        submitted_utc=_utc(), resumed_from=resumed,
                        sha256={"config": conf_sha, "list": conf["inventory"]["list_sha256"]}, **common)
    lock.unlink()
    log("submitted %s" % ", ".join("%s=%s" % kv for kv in zip(names, ids)))
    print("INCZOO %s" % json.dumps({"job_ids": ids, "names": names}, sort_keys=True), flush=True)
    return rec


class SubmitFailed(RuntimeError):
    """A failed sbatch: everything submitted was cancelled (exit 1)."""


# --------------------------------------------------------------- preflight
def preflight_cmd(version):
    """Read-only checks before a person or the platform submits."""
    from ..inc import scorer as S
    from . import base3 as B3
    problems = []

    def ok(msg):
        print("ok   %s" % msg, flush=True)

    def bad(msg):
        print("FAIL %s" % msg, flush=True)
        problems.append(msg)
    conf, conf_sha = load_config(version)
    ok("config %s sha256 %s" % (config_path(version), conf_sha))
    lp = Path(conf["inventory"]["list"])
    got = _sha_file(lp)
    (ok if got == conf["inventory"]["list_sha256"] else bad)("inventory list %s sha256 %s (pinned %s)" % (
        lp, got, conf["inventory"]["list_sha256"]))
    src = source_inc()
    lock1 = _read_json(src / "splits" / "v1" / "LOCK.json") or {}
    ss = _sha_file(Path(S.__file__).resolve())
    (ok if ss == lock1.get("scorer_sha256") else bad)("inc/scorer.py %s = LOCK v1's %s" % (
        ss, lock1.get("scorer_sha256")))
    summ = B3.read_summary(src / "splits" / "v3")
    (ok if summ else bad)("splits/v3/summary.json complete")
    mods = (os.environ.get("ZOO_MODULES") or "").split()
    if mods:
        repo = Path(C.REPO)
        drift = [m for m in mods if _sha_file(repo / "weed_optimizer_framework" / m)
                 != _sha_file(repo / "weed_llm_benchmark" / "weed_optimizer_framework" / m)]
        (ok if not drift else bad)("modules outer = nested (%d checked%s)" % (
            len(mods), "; differ: %s" % drift[:6] if drift else ""))
    rec = read_record(version)
    (ok if not rec or rec.get("status") != "complete" else bad)(
        "capacity/zoo_%s.json %s" % (version, "absent" if not rec else "status %s" % rec.get("status")))
    try:
        q = queued_zoo_jobs(version)
        (ok if not q else bad)("no inc_zoo_%s_* job queued%s" % (version, " (%s)" % q[:3] if q else ""))
    except ZooRefused as e:
        bad(str(e))
    if problems:
        raise ZooRefused("preflight: %d problem(s)" % len(problems))
    return 0


# -------------------------------------------------------------- eval-names
def eval_role(name, conf):
    """The role rule of the committed table (fail closed): weed only through
    species_of, the weed keys, suffixes, raw prefixes or weed names; drop
    through the drop keys; anything else, an ambiguous name, a number or an
    empty name, unresolved."""
    cm = conf["class_map"]
    k = name_key(name)
    if not k or k.isdigit() or k in conf["exams"]["eval"].get("ambiguous_keys", []):
        return "unresolved"
    r = name_role(name, cm)
    if r is None:
        return "unresolved"
    return "drop" if r == "drop" else "weed"


def eval_names_cmd(version):
    from . import base3 as B3
    conf, _sha = load_config(version)
    b3, _s = B3.load_config()
    reg = _read_json(B3.registry_path()) or {}
    ents = reg.get("datasets", reg) or {}
    out = {}
    for g, slugs in sorted(((b3.get("evaluation_groups") or {}).get("groups") or {}).items()):
        for slug in slugs:
            e = ents.get(slug)
            if slug in (conf["exams"]["eval"].get("exclude") or []):
                out[slug] = {"group": g, "excluded": True}
                continue
            if not isinstance(e, dict):
                out[slug] = {"group": g, "error": "not in the registry"}
                continue
            names, basis = B3._names_of(e, e.get("local_path") or "")
            out[slug] = {"group": g, "local_path": e.get("local_path"), "names": names, "basis": basis,
                         "roles": {n: eval_role(n, conf) for n in names or []}}
    print(json.dumps(out, indent=1, sort_keys=True))
    return out


# ------------------------------------------------------------------ claims
_CLAIM_RE = re.compile(r"((?:results|runs)/[A-Za-z0-9_./-]+)\.pt")


def claims_cmd(version, docs, list_path=None):
    """Checkpoint paths quoted in the docs, and run-directory tails
    (<project>/<name>) of the inventory list the docs quote, as
    [{match, cite}] for zoo_<version>.json "claims"."""
    tails = set()
    if list_path:
        rows, _b = parse_list_text(Path(list_path).read_text())
        for r in rows:
            rd = run_dir_rel(r["rel"]).split("/")
            if len(rd) >= 2 and not _EPOCH_RE.match(os.path.basename(r["rel"])):
                tails.add("/".join(rd[-2:]))
    out = {}
    for d in docs:
        try:
            lines = Path(d).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines, 1):
            for m in _CLAIM_RE.finditer(ln):
                key = re.escape(m.group(1) + ".pt") + r"\Z"
                out.setdefault(key, "%s:%d" % (d, i))
            for t in tails:
                if t in ln and re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(t), ln):
                    out.setdefault("(?:^|/)%s/weights/(?:best|last)\\.pt\\Z" % re.escape(t), "%s:%d" % (d, i))
    res = [{"match": k, "cite": v} for k, v in sorted(out.items())]
    print(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------- codever
def codever_cmd(version, git_dir, report=False):
    """INC rows' module sha256s mapped to the commits that hold them, and every
    other row's start date to main's last commit before it (git history
    required); writes codever_<version>.json under the zoo."""
    prov = read_provenance(version)

    def git(*a):
        p = subprocess.run(["git", "-C", str(git_dir)] + list(a), capture_output=True, text=True, timeout=300)
        return p.stdout if p.returncode == 0 else ""
    paths = sorted({k for r in prov.values() for k in ((r.get("code") or {}).get("modules") or {})})
    blob = {}
    for p in paths:
        gp = "weed_llm_benchmark/weed_optimizer_framework/%s" % p
        for c in git("log", "--format=%H %cI", "--", gp).split("\n"):
            if not c.strip():
                continue
            h, when = c.split(" ", 1)
            data = subprocess.run(["git", "-C", str(git_dir), "show", "%s:%s" % (h, gp)], capture_output=True,
                                  timeout=120).stdout
            blob.setdefault((p, _sha_bytes(data)), (h, when))
    rows = {}
    for mid, r in prov.items():
        code = r.get("code") or {}
        if code.get("kind") == "exact":
            hits = [blob.get((p, s)) for p, s in sorted((code.get("modules") or {}).items())]
            known = [h for h in hits if h]
            rows[mid] = {"kind": "exact", "commit": max(known, key=lambda h: h[1])[0] if known else None,
                         "modules_unmatched": sum(1 for h in hits if not h)}
        else:
            ds = (r.get("dates") or {}).get("start_utc")
            c = git("log", "-1", "--format=%H", "--before=%s" % ds, "main").strip() if ds else ""
            rows[mid] = {"kind": "approx", "commit": c or None, "before": ds}
    rec = {"format": CODEVER_FORMAT, "version": version, "git": str(git_dir), "rows": rows, "created_utc": _utc()}
    _write_json(zoo_dir(version) / ("codever_%s.json" % version), rec)
    log("codever: %d rows (%d with a commit)" % (len(rows), sum(1 for x in rows.values() if x["commit"])))
    if report:
        conf, sha = load_config(version)
        report_step(version, conf, sha)
    return rec


# --------------------------------------------------------------- inventory
STEPS = ("list", "meta", "provenance", "convert", "exams", "contamination", "pilot", "plan")


def run_step(version, conf, conf_sha, step, args):
    if step != "plan" and step_done(version, step, conf_sha):
        log("%s: kept (done earlier under this config)" % step)
        return "kept"
    if step == "list":
        out = list_step(version, conf, conf_sha)
    elif step == "meta":
        out = meta_step(version, conf, conf_sha)
    elif step == "provenance":
        out = provenance_step(version, conf, conf_sha)
    elif step == "convert":
        out = convert_step(version, conf, conf_sha)
    elif step == "exams":
        out = exams_step(version, conf, conf_sha)
    elif step == "contamination":
        out = contamination_step(version, conf, conf_sha)
    elif step == "pilot":
        out = pilot_step(version, conf, conf_sha)
    else:
        out = plan_step(version, conf, conf_sha, int(args.shards_a), float(args.max_gpu_hours))
    mark_step(version, step, conf_sha)
    return out


def inventory_cmd(version, args):
    conf, conf_sha = load_config(version)
    _once_bytes(zoo_dir(version) / "config.json", config_path(version).read_bytes())
    ledger_touch(version, "inventory")
    update_record(version, status="inventory_running")
    try:
        for step in STEPS:
            run_step(version, conf, conf_sha, step, args)
            ledger_touch(version, "inventory")
    except ZooRefused as e:
        update_record(version, status="refused", refusal=str(e)[:500])
        raise
    finally:
        ledger_touch(version, "inventory", ended=True)
    update_record(version, status="plan_ready")
    return 0


# -------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(prog="inc2.zoo", description="The model-zoo audit (Amendment 2026-10-04, Z1).")
    sub = ap.add_subparsers(dest="verb", required=True)

    def add(name, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("--version", required=True)
        return p
    add("preflight")
    p = add("submit")
    for f in ("--shards-a", "--shards-c", "--concurrency", "--max-gpu-hours"):
        p.add_argument(f, type=int, required=True)
    p = add("inventory")
    p.add_argument("--shards-a", type=int, required=True)
    p.add_argument("--shards-c", type=int, required=True)
    p.add_argument("--max-gpu-hours", type=float, required=True)
    for s in STEPS[:-1]:
        add(s)
    p = add("plan")
    p.add_argument("--shards-a", type=int, required=True)
    p.add_argument("--max-gpu-hours", type=float, required=True)
    for name in ("exam-root", "score"):
        p = add(name)
        p.add_argument("--stage", choices=("a", "c"))
        p.add_argument("--shard", type=int)
        p.add_argument("--pilot", action="store_true")
        if name == "score":
            p.add_argument("--item")
    p = add("select")
    p.add_argument("--shards-c", type=int, required=True)
    add("report")
    add("status")
    add("eval-names")
    p = add("claims")
    p.add_argument("--docs", nargs="+", required=True)
    p.add_argument("--list")
    p = add("codever")
    p.add_argument("--git", required=True)
    p.add_argument("--report", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    v = a.version
    try:
        if a.verb not in ("claims",):
            load_config(v)
        if a.verb == "preflight":
            return preflight_cmd(v)
        if a.verb == "submit":
            submit_cmd(v, a.shards_a, a.shards_c, a.concurrency, a.max_gpu_hours)
            return 0
        if a.verb == "inventory":
            return inventory_cmd(v, a)
        if a.verb in STEPS:
            conf, sha = load_config(v)
            run_step(v, conf, sha, a.verb, a)
            return 0
        if a.verb == "exam-root":
            if not a.pilot and (a.stage is None or (a.shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is None)):
                raise ZooRefused("exam-root needs --pilot, or --stage and --shard (or an array task)")
            shard = a.shard if a.shard is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
            print(exam_root_cmd(v, a.stage, shard, a.pilot), flush=True)
            return 0
        if a.verb == "score":
            if not (a.pilot or a.item or a.stage):
                raise ZooRefused("score needs --pilot, --item ID:EXAM, or --stage (and --shard or an array task)")
            score_cmd(v, a.stage, a.shard, a.pilot, a.item)
            return 0
        if a.verb == "select":
            conf, sha = load_config(v)
            ledger_touch(v, "select")
            try:
                select_step(v, conf, sha, a.shards_c)
            finally:
                ledger_touch(v, "select", ended=True)
            update_record(v, status="selected")
            return 0
        if a.verb == "report":
            conf, sha = load_config(v)
            ledger_touch(v, "report")
            try:
                report_step(v, conf, sha)
            finally:
                ledger_touch(v, "report", ended=True)
            return 0
        if a.verb == "status":
            status_cmd(v)
            return 0
        if a.verb == "eval-names":
            eval_names_cmd(v)
            return 0
        if a.verb == "claims":
            claims_cmd(v, a.docs, a.list)
            return 0
        if a.verb == "codever":
            codever_cmd(v, a.git, a.report)
            return 0
    except ZooRefused as e:
        print("[inc2.zoo] ERROR: %s" % e, file=sys.stderr, flush=True)
        return 2
    except (ZooSystemic, SubmitFailed) as e:
        print("[inc2.zoo] ERROR: %s" % e, file=sys.stderr, flush=True)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
