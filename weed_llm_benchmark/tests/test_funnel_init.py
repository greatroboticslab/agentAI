#!/usr/bin/env python3
"""The funnel package's shared helpers (runner §1.2-1.4, §5.1.1).

Pinned:
  * atomic writes: JSON, JSON lines and CSV land whole, leave no <name>.tmp
    behind, and keep the previous file untouched when the write fails midway
    (a non-serialisable value, a NaN, a CSV row of the wrong width);
  * header() carries every key of runner §1.2, hashes the inputs and the
    producing modules, and refuses an unknown output kind;
  * strip_volatile removes exactly VOLATILE_KEYS at any depth and nothing
    else;
  * check_records raises StaleInput naming a changed or missing input;
  * rng(text) is numpy's generator seeded by common.stable_int(text);
  * the error classes form the tree of runner §1.4;
  * __init__, domain, ledger and claims import with numpy blocked (the lab
    ticker imports them), and nothing in them pulls in numpy.

Run:  python3 tests/test_funnel_init.py
"""
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_init_")
check, raises = W.check, W.raises

from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402

print("paths and formats")
check("FUNNEL_DIR under INC_DIR", F.FUNNEL_DIR == pathlib.Path(os.environ["INC_DIR"]) / "funnel")
check("STEP1_DIR and R1_DIR", F.STEP1_DIR.name == "step1" and F.R1_DIR.name == "step1_r1")
for kind, fmt in (("census", "funnel-census/1"), ("ledger", "funnel-ledger/1"), ("name_status", "funnel-name-status/2"),
                  ("frames", "funnel-frames/1"), ("audit", "funnel-audit/1"), ("claims", "funnel-claims/1"),
                  ("rl_qualification", "funnel-rl-qualification/1"), ("recovery", "funnel-recovery/1")):
    check("format %s" % kind, F.FORMATS.get(kind) == fmt, F.FORMATS.get(kind))

print("error tree")
for name in ("StaleInput", "SampleLocked", "PreregError", "DomainError", "LedgerError", "StrataError", "DrawError",
             "EstimateError", "ClaimsError", "AdapterError", "TaxonomyError", "NamesError", "RelationError",
             "FetchError", "EmbedError", "JudgeError", "LeakError", "SheetError", "RLError", "QualifyError",
             "RecoverError", "CLIError"):
    cls = getattr(F, name, None)
    check("%s is a FunnelError" % name, cls is not None and issubclass(cls, F.FunnelError))
check("LeakCalibrationError < LeakError", issubclass(F.LeakCalibrationError, F.LeakError))
check("DisjointnessError, CircularityError < QualifyError",
      issubclass(F.DisjointnessError, F.QualifyError) and issubclass(F.CircularityError, F.QualifyError))
check("NeverTrainHit, GateNotMet < RecoverError",
      issubclass(F.NeverTrainHit, F.RecoverError) and issubclass(F.GateNotMet, F.RecoverError))
check("FunnelError is a RuntimeError", issubclass(F.FunnelError, RuntimeError))

print("atomic writes")
d = TMP / "w"
p = d / "a.json"
sha = F.write_json_atomic(p, {"b": 1, "a": [1, 2]})
check("json written, sha returned", p.exists() and sha == C.sha256_file(p))
check("no tmp left", not (d / "a.json.tmp").exists())
check("sorted keys, newline", p.read_text().startswith('{\n "a"') and p.read_text().endswith("\n"))
before = p.read_bytes()
check("non-serialisable refused", raises(lambda: F.write_json_atomic(p, {"x": object()}), F.FunnelError))
check("NaN refused", raises(lambda: F.write_json_atomic(p, {"x": float("nan")}), F.FunnelError))
check("old file kept after a failed write", p.read_bytes() == before)
check("no tmp after failure", not (d / "a.json.tmp").exists())


class Boom(object):
    def __iter__(self):
        yield {"a": 1}
        raise RuntimeError("disk full")


check("jsonl generator failure propagates", raises(lambda: F.write_jsonl_atomic(d / "b.jsonl", Boom()),
                                                    RuntimeError))
check("no jsonl written on failure", not (d / "b.jsonl").exists() and not (d / "b.jsonl.tmp").exists())
sha = F.write_jsonl_atomic(d / "b.jsonl", [{"z": 1, "a": 2}, {"k": "v"}])
lines = (d / "b.jsonl").read_text().splitlines()
check("jsonl canonical lines", lines == ['{"a":2,"z":1}', '{"k":"v"}'], lines)
check("read_jsonl round trip", F.read_jsonl(d / "b.jsonl") == [{"z": 1, "a": 2}, {"k": "v"}])
F.write_csv_atomic(d / "c.csv", ["x", "y"], [[1, None], {"x": "a,b", "y": 2}])
hdr, rows = F.read_csv(d / "c.csv")
check("csv header and rows (None -> empty, quoting)", hdr == ["x", "y"] and rows == [{"x": "1", "y": ""},
                                                                                     {"x": "a,b", "y": "2"}], rows)
before = (d / "c.csv").read_bytes()
check("csv row of the wrong width refused", raises(lambda: F.write_csv_atomic(d / "c.csv", ["x"], [[1, 2]]),
                                                   F.FunnelError))
check("csv dict row missing a key refused", raises(lambda: F.write_csv_atomic(d / "c.csv", ["x", "y"], [{"x": 1}]),
                                                   F.FunnelError))
check("csv kept after refusals", (d / "c.csv").read_bytes() == before)
check("read_json of a missing file refuses", raises(lambda: F.read_json(d / "nope.json"), F.FunnelError))
(d / "bad.json").write_text("{not json")
check("read_json of malformed JSON refuses", raises(lambda: F.read_json(d / "bad.json"), F.FunnelError))

print("canonical json and records")
check("canonical json", F.canonical_json({"b": [1, {"d": 2, "c": 1}], "a": "é"}) == '{"a":"é","b":[1,{"c":1,"d":2}]}')
check("canonical json refuses NaN (it is not JSON; the core sha256 of a prereg holding one is undefined)",
      raises(lambda: F.canonical_json({"x": float("nan")}), F.FunnelError))
rec = F.file_record(p)
check("file_record", set(rec) == {"path", "sha256", "bytes"} and rec["bytes"] == p.stat().st_size)
F.check_records({"a": rec, "mem": {"sha256": "0" * 64}})
check("check_records passes unchanged files (and skips path-less digests)", True)
p.write_text("{}")
try:
    F.check_records({"a": rec})
    ok = False
except F.StaleInput as e:
    ok = "a" in str(e)
check("check_records names a changed file", ok)
p.unlink()
check("check_records refuses a missing file", raises(lambda: F.check_records({"a": rec}), F.StaleInput))

print("seeds")
import numpy as np  # noqa: E402
check("seed = stable_int", F.seed("funnel/v1/x") == C.stable_int("funnel/v1/x"))
check("rng is the seeded default_rng", F.rng("funnel/v1/x").integers(0, 10 ** 9, 5).tolist()
      == np.random.default_rng(C.stable_int("funnel/v1/x")).integers(0, 10 ** 9, 5).tolist())

print("header")
pre = D.load_prereg(TMP / "inc" / "funnel" / "prereg_v1.json")
dom = D.load("weed")
F.write_json_atomic(d / "in.json", {"q": 1})
h = F.header("audit", dom, pre, {"in": d / "in.json"}, seeds={"s": "funnel/v1/s"}, modules=(F, D), testing=True)
check("every §1.2 key", set(F.HEADER_KEYS) <= set(h), sorted(set(F.HEADER_KEYS) - set(h)))
check("format from the kind", h["format"] == "funnel-audit/1")
check("prereg record with core sha", h["prereg"]["core_sha256"] == pre.core_sha256 and h["prereg"]["sha256"] == pre.sha256)
check("contract record: the contract in force (the prereg's contract amendment records it)",
      h["contract"]["sha256"] == D.contract_sha256_of(pre) == pre.contract_sha256
      and h["contract"]["sha256"] == F.sha256_file(pre.contract_path))
check("domain config record", h["domain_config"]["sha256"] == dom.sha256 and h["domain"] == "weed")
check("inputs hashed", h["inputs"]["in"]["sha256"] == C.sha256_file(d / "in.json"))
check("code relative to the tools package", sorted(h["code"]) == ["funnel/__init__.py", "funnel/domain.py"], h["code"])
check("testing flag", h["testing"] is True)
check("unknown kind refused", raises(lambda: F.header("nonsense", dom, pre, {}), F.FunnelError))
check("explicit format id accepted", F.header("funnel-thing/3", dom, pre, {})["format"] == "funnel-thing/3")

print("strip_volatile")
obj = {"built_utc": 1, "a": {"seconds": 2, "hostname": "h", "keep": [{"slurm_job_id": 3, "x": 4}]}, "b": 5}
check("strips only the volatile keys", F.strip_volatile(obj) == {"a": {"keep": [{"x": 4}]}, "b": 5})
check("VOLATILE_KEYS as pinned", F.VOLATILE_KEYS == ("built_utc", "seconds", "hostname", "slurm_job_id"))

print("light modules import without numpy")
code = r"""
import sys, types, pathlib
sys.modules['numpy'] = None
root = pathlib.Path(sys.argv[1])
for name, path in (('weed_optimizer_framework', root / 'weed_optimizer_framework'),
                   ('weed_optimizer_framework.tools', root / 'weed_optimizer_framework' / 'tools')):
    m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
import importlib
for mod in ('funnel', 'funnel.domain', 'funnel.ledger', 'funnel.claims'):
    importlib.import_module('weed_optimizer_framework.tools.' + mod)
heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('numpy', 'scipy', 'torch', 'sklearn', 'PIL')
               and sys.modules[m] is not None)
print('HEAVY', heavy)
"""
res = subprocess.run([sys.executable, "-c", code, str(W.PKG_ROOT)], capture_output=True, text=True,
                     env=dict(os.environ))
check("imported with numpy blocked", res.returncode == 0 and "HEAVY []" in res.stdout, res.stderr[-800:])

W.finish()
