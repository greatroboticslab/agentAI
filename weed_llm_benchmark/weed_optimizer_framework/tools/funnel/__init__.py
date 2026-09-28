"""The funnel audit engine (docs/FUNNEL_AUDIT.md, runner docs/FUNNEL_AUDIT_RUNNER.md).

What this module holds, shared by every other module of the package:
  * where the audit's files live (FUNNEL_DIR, STEP1_DIR, R1_DIR under INC_DIR);
  * the format id of every file the runner pins (FORMATS);
  * the error classes (FunnelError and one subclass per module);
  * canonical JSON, atomic writers, file records and the freshness check;
  * seeds (common.stable_int) and the one numpy generator per seed text;
  * the header every JSON output carries.

Standard library only at module level: the lab ticker imports this module,
domain.py, ledger.py and claims.py, and numpy is imported inside rng() only.
Nothing here names a domain; everything domain-specific comes from
domains/<domain>.json through domain.py.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import os
import re
from pathlib import Path

from ..inc import common as C

FUNNEL_DIR = C.INC_DIR / "funnel"
STEP1_DIR = C.INC_DIR / "step1"
R1_DIR = C.INC_DIR / "step1_r1"
PKG_DIR = Path(__file__).resolve().parents[1]         # the tools package directory
FUN_DIR = Path(__file__).resolve().parent             # this package

FORMATS = {
    "domain": "funnel-domain/1",
    "prereg": "funnel-prereg/1",
    "census": "funnel-census/1",
    "ledger": "funnel-ledger/1",
    "name_status": "funnel-name-status/2",
    "taxonomy_cache": "funnel-taxonomy-cache/1",
    "known_items": "funnel-known-items/1",
    "cards": "funnel-cards/1",
    "fetch_manifest": "funnel-fetch-manifest/1",
    "refetch_manifest": "funnel-refetch-manifest/1",
    "frames": "funnel-frames/1",
    "board": "funnel-board/1",
    "sheet": "funnel-sheet/1",
    "sheets": "funnel-sheets/1",
    "rl_answers": "funnel-rl-answers/1",
    "leak": "funnel-leak/1",
    "relation_geometry": "funnel-relation-geometry/1",
    "class_maps": "funnel-class-maps/1",
    "relation_audit": "funnel-relation-audit/1",
    "judge_qualification": "funnel-judge-qualification/1",
    "rl_qualification": "funnel-rl-qualification/1",
    "audit": "funnel-audit/1",
    "claims": "funnel-claims/1",
    "prospective_da": "funnel-prospective-da/1",
    "recovery": "funnel-recovery/1",
    "domain_dev": "funnel-domain-dev/1",
    "arms": "funnel-arms/1",
    "panel": "funnel-panel/1",
}
_FORMAT_RE = re.compile(r"^funnel-[a-z0-9][a-z0-9-]*/[0-9]+$")

VOLATILE_KEYS = ("built_utc", "seconds", "hostname", "slurm_job_id")
HEADER_KEYS = ("format", "domain", "built_utc", "prereg", "contract", "domain_config", "code",
               "inputs", "seeds", "testing")


# ------------------------------------------------------------------ errors
class FunnelError(RuntimeError):
    """Any refusal of the funnel engine. The CLI maps it to exit code 2."""


class StaleInput(FunnelError):
    """A recorded input no longer hashes to what its producer recorded."""


class SampleLocked(FunnelError):
    """An output fixed by the sample lock would be rewritten."""


class PreregError(FunnelError):
    pass


class DomainError(FunnelError):
    pass


class LedgerError(FunnelError):
    pass


class StrataError(FunnelError):
    pass


class DrawError(FunnelError):
    pass


class EstimateError(FunnelError):
    pass


class ClaimsError(FunnelError):
    pass


class AdapterError(FunnelError):
    pass


class TaxonomyError(FunnelError):
    pass


class NamesError(FunnelError):
    pass


class RelationError(FunnelError):
    pass


class FetchError(FunnelError):
    pass


class EmbedError(FunnelError):
    pass


class JudgeError(FunnelError):
    pass


class LeakError(FunnelError):
    pass


class LeakCalibrationError(LeakError):
    pass


class SheetError(FunnelError):
    pass


class RLError(FunnelError):
    pass


class QualifyError(FunnelError):
    pass


class DisjointnessError(QualifyError):
    pass


class CircularityError(QualifyError):
    pass


class RecoverError(FunnelError):
    pass


class NeverTrainHit(RecoverError):
    pass


class GateNotMet(RecoverError):
    pass


class CLIError(FunnelError):
    pass


# ------------------------------------------------------------------ basics
def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj):
    """The one canonical form (runner §3.3): sorted keys, no spaces, UTF-8
    text. NaN and infinities are refused: they are not JSON."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False)
    except ValueError as e:
        raise FunnelError("not canonical JSON (%s)" % e)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return C.sha256_file(path)


def _atomic_write_bytes(path, data):
    """Write data to <name>.tmp beside path, then os.replace. On any error the
    temporary file is removed and an existing file at path is left as it was."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return sha256_bytes(data)


def json_text(obj):
    """The file form of a JSON output: sorted keys, indent 1, a final newline."""
    try:
        return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as e:
        raise FunnelError("refusing to write non-finite numbers as JSON (%s)" % e)
    except TypeError as e:
        raise FunnelError("not JSON-serialisable (%s)" % e)


def write_json_atomic(path, obj):
    """Write obj as JSON (json_text), atomically. Returns the file's sha256."""
    return _atomic_write_bytes(path, json_text(obj).encode("utf-8"))


def write_jsonl_atomic(path, rows):
    """One canonical JSON object per line, in the given order. Returns sha256."""
    buf = io.StringIO()
    for r in rows:
        buf.write(canonical_json(r))
        buf.write("\n")
    return _atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def write_csv_atomic(path, header, rows):
    """A CSV with the given header; rows are sequences in header order or dicts
    keyed by the header (a missing key is refused). Returns sha256."""
    header = list(header)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        if isinstance(r, dict):
            missing = [h for h in header if h not in r]
            if missing:
                raise FunnelError("csv row for %s lacks %s" % (Path(path).name, missing))
            r = [r[h] for h in header]
        else:
            r = list(r)
            if len(r) != len(header):
                raise FunnelError("csv row for %s has %d fields, header %d"
                                  % (Path(path).name, len(r), len(header)))
        w.writerow(["" if v is None else v for v in r])
    return _atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def read_json(path):
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise FunnelError("missing file %s" % path)
    except (OSError, ValueError) as e:
        raise FunnelError("unreadable JSON %s (%s)" % (path, e))


def read_jsonl(path):
    path = Path(path)
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError as e:
                        raise FunnelError("%s line %d is not JSON (%s)" % (path, i, e))
    except FileNotFoundError:
        raise FunnelError("missing file %s" % path)
    return rows


def read_csv(path):
    """(header, [dict rows]) of a CSV written by write_csv_atomic."""
    path = Path(path)
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            try:
                header = next(rd)
            except StopIteration:
                raise FunnelError("empty csv %s" % path)
            rows = [dict(zip(header, r)) for r in rd]
    except FileNotFoundError:
        raise FunnelError("missing file %s" % path)
    return header, rows


def file_record(path):
    path = Path(path)
    if not path.is_file():
        raise FunnelError("missing file %s" % path)
    return {"path": str(path), "sha256": C.sha256_file(path), "bytes": path.stat().st_size}


def check_records(records):
    """Re-hash every {"path", "sha256"} record (a dict of name -> record, as a
    header's "inputs"); raise StaleInput naming the first file that is missing
    or changed. Records without a path (in-memory digests) are skipped."""
    for name in sorted(records or {}):
        rec = records[name]
        if not isinstance(rec, dict) or not rec.get("path"):
            continue
        p = Path(rec["path"])
        if not p.is_file():
            raise StaleInput("input %s (%s) is missing" % (name, p))
        got = C.sha256_file(p)
        if got != rec.get("sha256"):
            raise StaleInput("input %s (%s) changed: sha256 %s, recorded %s"
                             % (name, p, got[:12], str(rec.get("sha256"))[:12]))


def check_prereg_core(doc, prereg, what):
    """The prereg core an artifact records must be the current prereg's core
    (runner §3.3: a consumer compares core_sha256). A prereg edited outside
    its amendments after the artifact was made, or an artifact that records
    no prereg, refuses (StaleInput); the sample-lock amendment alone never
    does, because it changes only the amendments. prereg is a loaded Prereg
    or a core sha256."""
    want = getattr(prereg, "core_sha256", prereg)
    rec = doc.get("prereg") if isinstance(doc, dict) else None
    got = rec.get("core_sha256") if isinstance(rec, dict) else None
    if not got:
        raise StaleInput("%s records no prereg core_sha256; it cannot be matched to prereg core %s"
                         % (what, str(want)[:12]))
    if got != want:
        raise StaleInput("%s was made under prereg core %s, the prereg's core is now %s: the prereg was "
                         "edited outside its amendments" % (what, str(got)[:12], str(want)[:12]))


def seed(text):
    return C.stable_int(text)


def rng(text):
    import numpy as np
    return np.random.default_rng(C.stable_int(text))


def _module_path(m):
    if isinstance(m, (str, Path)):
        return Path(m).resolve()
    f = getattr(m, "__file__", None)
    if not f:
        raise FunnelError("module %r has no file" % (m,))
    return Path(f).resolve()


def code_record(*modules):
    """{path relative to the tools package: sha256} of every module given."""
    out = {}
    for m in modules:
        p = _module_path(m)
        try:
            rel = p.relative_to(PKG_DIR).as_posix()
        except ValueError:
            rel = p.as_posix()
        out[rel] = C.sha256_file(p)
    return out


def _named_record(obj, what):
    """{"path", "sha256"} of a loaded config / prereg object or of a dict
    carrying those keys."""
    if obj is None:
        raise FunnelError("header needs the %s" % what)
    if isinstance(obj, dict):
        if "path" in obj and "sha256" in obj:
            return {"path": str(obj["path"]), "sha256": obj["sha256"]}
        raise FunnelError("header: %s record lacks path/sha256" % what)
    path, sha = getattr(obj, "path", None), getattr(obj, "sha256", None)
    if path is None or sha is None:
        raise FunnelError("header: %s object lacks path/sha256" % what)
    return {"path": str(path), "sha256": sha}


def input_records(inputs):
    """Normalise {"name": path | record} to {"name": {"path", "sha256", "bytes"}}.
    A record given as a dict with "sha256" is kept (an in-memory digest may
    have no path)."""
    out = {}
    for name in sorted(inputs or {}):
        v = inputs[name]
        if isinstance(v, dict):
            if "sha256" not in v:
                raise FunnelError("input record %s lacks sha256" % name)
            out[name] = dict(v)
        else:
            out[name] = file_record(v)
    return out


def header(kind, domain, prereg, inputs, seeds=None, modules=(), testing=False):
    """The top-level keys every JSON output carries (runner §1.2).

    kind is a key of FORMATS or a full "funnel-<kind>/<n>" id; domain is a
    loaded domain.Domain (or a {"name"/"domain", "path", "sha256"} dict);
    prereg is a loaded domain.Prereg (or a dict with path, sha256, core_sha256
    and contract); inputs maps logical names to paths or records."""
    if kind in FORMATS:
        fmt = FORMATS[kind]
    elif isinstance(kind, str) and _FORMAT_RE.match(kind):
        fmt = kind
    else:
        raise FunnelError("unknown output kind %r" % (kind,))
    if isinstance(domain, dict):
        dname = domain.get("name") or domain.get("domain")
    else:
        dname = getattr(domain, "name", None)
    if not dname:
        raise FunnelError("header needs the domain name")
    if isinstance(prereg, dict):
        pre = {"path": str(prereg["path"]), "sha256": prereg["sha256"],
               "core_sha256": prereg["core_sha256"]}
        contract = prereg.get("contract")
        if not isinstance(contract, dict) or "path" not in contract or "sha256" not in contract:
            raise FunnelError("header: prereg record lacks the contract record")
        contract = {"path": str(contract["path"]), "sha256": contract["sha256"]}
    else:
        pre = {"path": str(prereg.path), "sha256": prereg.sha256, "core_sha256": prereg.core_sha256}
        contract = {"path": str(prereg.contract_path), "sha256": prereg.contract_sha256}
    return {
        "format": fmt,
        "domain": dname,
        "built_utc": utc(),
        "prereg": pre,
        "contract": contract,
        "domain_config": _named_record(domain, "domain config"),
        "code": code_record(*modules) if modules else {},
        "inputs": input_records(inputs),
        "seeds": dict(seeds or {}),
        "testing": bool(testing),
    }


def strip_volatile(obj):
    """A copy of obj without any VOLATILE_KEYS, at any depth; nothing else changes."""
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    if isinstance(obj, tuple):
        return [strip_volatile(v) for v in obj]
    return obj
