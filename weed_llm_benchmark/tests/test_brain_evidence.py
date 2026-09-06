#!/usr/bin/env python3
"""Unit tests for the live evidence bundle (no cluster / Mongo / network).

The fake ctx replays recorded gather output, byte for byte in the shape the
remote script emits, so the parser is tested against the transport it will
actually see. Covered:

  * the gather is ONE batched command per build, and the job id, domain and
    step reach the shell as filtered literals (a shell metacharacter cannot)
  * absolute line numbers survive the transport: the 2026-08-29 SKIPPING line
    is still at line 8,123 and a quote resolves to that address
  * an empty section, a section whose command failed, a section that never
    terminated, and payload lines that are delimiter-shaped
  * per-section caps and the priority trimmer: out_tail first, its floor keeps
    every WARN/ERROR/TIMEOUT/SKIPPING line, sacct/strategy/signals untouched,
    and every trim is recorded with what it removed
  * token_estimate is corpus.token_estimate, and build() refuses to hand back a
    bundle above the caller's num_ctx without saying so
  * the bundle has exactly the key set corpus.py writes, and the WP2 harness
    (bench.bundle_lines / ledger_view / a0_decision / resolve_quote) reads it
  * lab->cluster staging stays under the per-command ceiling and round-trips
  * no public entry point raises, whatever it is handed

Run:  python3 tests/test_brain_evidence.py
"""
import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import corpus  # noqa: E402
from weed_optimizer_framework.tools.brain import evidence  # noqa: E402

try:
    from weed_optimizer_framework.tools.brain import bench  # noqa: E402
except Exception:                                  # the harness is optional here
    bench = None

for _k in ("CORPUS_SCRUB_USERS", "CORPUS_SCRUB_LITERALS"):
    os.environ.pop(_k, None)

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


# --------------------------------------------------------------------------
# the recorded gather: the 2026-08-29 walltime TIMEOUT (job 44727703)
# --------------------------------------------------------------------------
DOMAIN, ROUND, STEP, JOB = "weed", 4, "train", "44727703"
EV = "EV" + evidence.nonce(DOMAIN, ROUND, STEP, JOB)

OUT_PATH = "results/framework/m1_merged_rndtrain_s1_44727703.out"
OUT_SHA = "a" * 64
OUT_LINES_TOTAL = 200000
SKIP_LINE = ("[net] WARN: SOCKS proxy via bridges2-login011 failed "
             "(compute->login SSH disabled) - SKIPPING github, using Kaggle/HF only")
# The epoch-closing progress line: the only place this run's iterations per
# epoch is written down, and therefore the only thing a pool_growth finding can
# quote. 8,583 is the 2026-08-29 pool.
POOL_LINE = ("      24/60      12.4G      1.104      0.882      0.913        "
             "118        640: 100%|##########| 8583/8583 [50:12<00:00,  2.85it/s]")
TIMEOUT_LINE = ("slurmstepd: error: *** JOB 44727703 ON v012 CANCELLED AT "
                "2026-08-29T17:52:49 DUE TO TIME LIMIT ***")


def emit(name, rc, lines):
    """One section exactly as the remote helper writes it (awk 1 + wc -c)."""
    payload = "".join(l + "\n" for l in lines)
    return ("%s BEGIN %s\n" % (EV, name) + payload
            + "%s END %s rc=%d bytes=%d\n" % (EV, name, rc,
                                              len(payload.encode("utf-8"))))


def numbered(n, text):
    return "%06d\t%s" % (n, text)


def out_tail_payload(extra=()):
    lines = ["[file] path=%s bytes=23068672 mtime=1756500769 sha256=%s"
             % (OUT_PATH, OUT_SHA), "[block] matches"]
    lines.append(numbered(8123, SKIP_LINE))
    lines.append(numbered(199997, TIMEOUT_LINE))
    lines.append("[block] progress")
    lines.append(numbered(4211, POOL_LINE))
    lines.append("[block] tail")
    for i in range(OUT_LINES_TOTAL - 119, OUT_LINES_TOTAL + 1):
        if i == 199997:
            lines.append(numbered(i, TIMEOUT_LINE))
        else:
            lines.append(numbered(i, "epoch 24/60  box_loss 1.104  time 43102.4"))
    lines.extend(extra)
    lines.append("[lines] %d" % OUT_LINES_TOTAL)
    lines.append("[matches] 2 kept 2")
    return lines


SACCT_HEADER = ("JobID|JobIDRaw|JobName|State|Elapsed|Timelimit|Start|End|"
                "Submit|AllocTRES|ExitCode|NodeList")
SACCT_ROW = ("44727703|44727703|rndtrain|TIMEOUT|12:00:18|12:00:00|"
             "2026-08-29T05:52:31|2026-08-29T17:52:49|2026-08-29T05:50:02|"
             "billing=8,cpu=8,gres/gpu:h100-80=1,mem=63000M,node=1|0:0|v012")

LEDGER = {"rounds": [{"round_num": r, "domain": "weed",
                      "steps": {"train": {"status": "done", "actor":
                                          "round-scheduler"}}}
                     for r in range(1, 4)]
          + [{"round_num": 4, "domain": "weed",
              "steps": {"train": {"status": "failed", "actor": "round-scheduler",
                                  "job": JOB, "detail": "TIMEOUT"}}}],
          "paused_reason": "stop-loss: 2 consecutive train failures",
          "paused_at": 1756500800, "tick_s": 120}

STRATEGY = {"tier": "curated", "seed": 101, "iteration": "rnd4_train_44727703",
            "job_id": JOB, "run_tag": "job44727703", "status": "running",
            "strategy": {"epochs": 60, "imgsz": 640, "patience": 20,
                         "time_h": None, "min_dino_score": 0.5},
            "summary": {"save_dir": "results/framework/mega_iterrnd4_train_"
                                    "44727703/job44727703"}}

TRACE = [{"ts": 1756460000 + i * 1800, "kind": "epoch", "domain": "weed",
          "round": 4, "step": "train", "job_id": JOB, "epoch": i,
          "map50_95": 0.41 + i * 0.001, "elapsed_s": i * 1800.0,
          "eta_total_s": 60 * 1800.0, "walltime_s": 43200.0,
          "sha_prev": "0" * 64} for i in range(1, 25)]

CSV_HEADER = ("epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),time")
CSV_ROWS = ["%d,1.1,0.62,%0.4f,%0.1f" % (i, 0.41 + i * 0.001, i * 1800.0)
            for i in range(1, 25)]

SLUG_SCORES = {"scores": {"slug%02d" % i: 0.4 + i * 0.01 for i in range(40)}}
CORRECTIONS = {"items": [{"seq": 1, "kind": "set_round_param", "author":
                          "human:supervisor", "target": {"key": "epochs",
                                                         "old": 60, "new": 30},
                          "hash": "b" * 64}]}
PLAN = {"version": 1, "hypotheses": ["pool growth outruns the walltime"],
        "ordered_experiments": []}
STAGED_SU = {"round": 24.0, "campaign": 812.5, "envelope": 1500}


def json_block(obj, path, mtime=1756500000, sha="c" * 64):
    body = corpus.canonical_json(obj)
    lines = ["[file] path=%s bytes=%d mtime=%d sha256=%s"
             % (path, len(body.encode("utf-8")), mtime, sha)]
    for i, text in enumerate(body.split("\n"), start=1):
        lines.append(numbered(i, text))
    lines.append("[lines] %d" % len(body.split("\n")))
    return lines


def capture(out_extra=(), overrides=None):
    """The whole recorded gather output, section by section, in script order."""
    over = overrides or {}
    parts = []

    def add(name, rc, lines):
        if name in over:
            parts.append(over[name])
        else:
            parts.append(emit(name, rc, lines))

    add("ledger", 0, json_block(LEDGER, "results/framework/_brain/weed/ledger.json"))
    add("sacct", 0, [numbered(1, SACCT_HEADER), numbered(2, SACCT_ROW)])
    add("out_tail", 0, out_tail_payload(out_extra))
    add("results_csv", 0,
        ["[file] path=results/framework/mega_iterrnd4_train_44727703/"
         "job44727703/results.csv bytes=2048 mtime=1756500700 sha256=%s" % ("d" * 64),
         numbered(1, CSV_HEADER)]
        + [numbered(i + 2, r) for i, r in enumerate(CSV_ROWS)]
        + ["[lines] %d" % (len(CSV_ROWS) + 1)])
    add("strategy", 0, json_block(
        STRATEGY, "results/framework/m1_curated_seed101_44727703.json"))
    add("trace", 0,
        ["[file] path=results/framework/_brain/weed/trace/rnd4_train_44727703"
         "_44727703.jsonl bytes=9000 mtime=1756500760 sha256=%s" % ("e" * 64)]
        + [numbered(i + 1, corpus.canonical_json(r))
           for i, r in enumerate(TRACE)]
        + ["[lines] %d" % len(TRACE)])
    add("slug_scores", 0, json_block(
        SLUG_SCORES, "results/framework/dinov2_curator/slug_scores.json"))
    add("registry_diff", 0,
        ["[file] path=results/framework/dataset_registry.json bytes=4194304 "
         "mtime=1756400000 sha256=%s" % ("f" * 64), "[block] prev",
         "[none] no staged previous-registry summary"])
    add("harvest", 0,
        ["[file] path=results/framework/v3_0_43_brain_harvest_44700001.out "
         "bytes=512000 mtime=1756300000 sha256=%s" % ("9" * 64),
         numbered(77, SKIP_LINE),
         numbered(412, "[src] kaggle: 12 candidates"),
         "[lines] 9000", "[matches] 2 kept 2", "[block] trace",
         "[none] no collect/harvest trace under results/framework/_brain/weed/trace"])
    add("resources", 0,
        [numbered(1, "Filesystem     1024-blocks       Used  Available Capacity Mounted on"),
         numbered(2, "ocean          107374182400 106300000000 1074741824      99% /ocean"),
         "[block] squeue", "[value] squeue_depth=7", "[block] quota"]
        + [numbered(i + 1, t) for i, t in enumerate(
            corpus.canonical_json({"quota_headroom_gb": 980.5}).split("\n"))])
    add("su", 0, json_block(STAGED_SU, "results/framework/_brain/weed/su.json"))
    add("corrections", 0, json_block(
        CORRECTIONS, "results/framework/_brain/weed/corrections.json"))
    add("plan", 0, json_block(PLAN, "results/framework/_brain/weed/plan.json"))
    return "".join(parts)


class Runner(object):
    """A fake remote channel that replays recorded output for the last command."""

    def __init__(self, text, fail=False, returns=None):
        self.text = text
        self.calls = []
        self.fail = fail
        self.returns = returns

    def __call__(self, cmd, timeout=None):
        self.calls.append(cmd)
        if self.fail:
            raise RuntimeError("Connection closed by remote host port 22")
        if self.returns is not None:
            return self.returns
        if "BEGIN" in cmd or "EV" in cmd:
            pass
        return {"ok": True, "stdout": self.text, "stderr": "", "returncode": 0}


# --------------------------------------------------------------------------
print("\n[1] the gather script")
script = evidence.remote_script(DOMAIN, ROUND, STEP, JOB)
ck("remote_script returns one string", isinstance(script, str) and script)
ck("no here-doc anywhere in the gather", "<<" not in script)
ck("no placeholder survives substitution",
   not re.search(r"__[A-Z0-9_]+__", script))
ck("the job id reaches the shell as digits", "\nJ=44727703\n" in script)
ck("every gathered section is emitted exactly once",
   all(script.count("\nS %s $rc\n" % s) == 1 for s in evidence.REMOTE_SECTIONS))
ck("the nonce is in the delimiter", ("EV=%s" % EV) in script)

inj = evidence.remote_script("we;ed $(id)", ROUND, "train;id", "447; rm -rf /")
ck("a shell metacharacter in the job id is filtered out",
   "rm -rf /" not in inj and "\nJ=447\n" in inj)
ck("a shell metacharacter in the domain is filtered out",
   "\nD=weedid\n" in inj and "$(id)" not in inj and ";" not in
   inj.split("\nD=")[1].split("\n")[0])
ck("nonce is deterministic for the same build",
   evidence.nonce(DOMAIN, ROUND, STEP, JOB) == EV[2:])

# --------------------------------------------------------------------------
print("\n[2] one batched command, and the bundle it produces")
runner = Runner(capture())
bundle = evidence.build(DOMAIN, ROUND, STEP, JOB, {"run": runner})
ck("the gather is ONE remote command", len(runner.calls) == 1)
ck("build recorded that call", bundle["export"]["build"]["remote_calls"] == 1)

rep = evidence.validate(bundle)
ck("the bundle validates: " + "; ".join(rep["errors"]), rep["ok"])
ck("all 14 sections are present", set(bundle["sections"]) == set(corpus.SECTIONS))
absent = sorted(n for n in corpus.SECTIONS if bundle["sections"].get(n) is None)
ck("every section this gather carried is filled: " + repr(absent),
   absent in ([], ["signals"]))
sigs = bundle["sections"]["signals"]
ck("signals is either the detector's list or null with a reason, never []",
   (isinstance(sigs, list) and all(isinstance(s, dict) for s in sigs))
   or (sigs is None and bundle["export"]["missing"].get("signals")))
if isinstance(sigs, list):
    fired = {s.get("signal") for s in sigs
             if s.get("severity") in ("warn", "crit")}
    ck("the 2026-08-29 fixture fires walltime_bound: " + repr(sorted(fired)),
       "walltime_bound" in fired)
    ck("no signal fires without evidence it can cite",
       all(s.get("evidence") or s.get("severity") == "unknown" for s in sigs))

ck("sacct parsed the TIMEOUT row",
   bundle["sections"]["sacct"][0]["State"] == "TIMEOUT"
   and bundle["sections"]["sacct"][0]["Elapsed"] == "12:00:18"
   and bundle["sections"]["sacct"][0]["Timelimit"] == "12:00:00")
ck("the strategy JSON is the job's own artifact",
   bundle["sections"]["strategy"]["job_id"] == JOB
   and bundle["sections"]["strategy"]["strategy"]["epochs"] == 60)
ck("results_csv summarises the run",
   bundle["sections"]["results_csv"]["rows"] == 24
   and bundle["sections"]["results_csv"]["epoch_time_s"] == 1800.0)
ck("the trace tail is capped at the record cap",
   len(bundle["sections"]["trace"]) <= bundle["caps"]["trace_records"])
ck("slug_scores became a distribution summary",
   bundle["sections"]["slug_scores"]["n"] == 40
   and bundle["sections"]["slug_scores"]["median"] is not None)
ck("this job's SU is measured from sacct (1 H100 x 12.005 h x 2)",
   abs(bundle["sections"]["su"]["job"] - 24.01) < 0.02)
ck("the staged round/campaign/envelope survive",
   bundle["sections"]["su"]["round"] == 24.0
   and bundle["sections"]["su"]["envelope"] == 1500)
ck("resources reports filesystem free and queue depth separately",
   bundle["sections"]["resources"]["squeue_depth"] == 7
   and abs(bundle["sections"]["resources"]["fs_free_tb"] - 1.0) < 0.01
   and bundle["sections"]["resources"]["quota_headroom_gb"] == 980.5)
ck("harvest per_source is null with a reason, not an invented count",
   bundle["sections"]["harvest"]["per_source"] is None
   and bundle["sections"]["harvest"]["per_source_reason"])
ck("registry_diff reports identity and states why there is no diff",
   bundle["sections"]["registry_diff"]["registry"]["sha256"] == "f" * 64
   and bundle["sections"]["registry_diff"]["added_slugs"] is None
   and bundle["sections"]["registry_diff"]["reason"])
ck("the ledger keeps the last N rounds and the pause",
   len(bundle["sections"]["ledger"]["rounds"]) == 4
   and bundle["sections"]["ledger"]["paused_reason"].startswith("stop-loss"))

# --------------------------------------------------------------------------
print("\n[3] absolute line numbers and citations")
lines = {n: t for n, t in bundle["sections"]["out_tail"]["lines"]}
ck("the SKIPPING line kept its absolute number 8123",
   lines.get(8123) == SKIP_LINE)
ck("the pool-growth line kept its absolute number 4211",
   lines.get(4211) == POOL_LINE)
ck("the tail is numbered from the file's own line count, not from 1",
   max(lines) == OUT_LINES_TOTAL and min(lines) == 4211
   and lines[199881].startswith("epoch 24/60"))
ck("the progress line that states 8583 iterations/epoch is quotable",
   "8583/8583 [" in lines.get(4211, ""))
ck("out_tail carries the original file's sha256, not the excerpt's",
   bundle["sections"]["out_tail"]["sha256"] == OUT_SHA)
ck("the harvest log line kept its absolute number 77",
   [77, SKIP_LINE] in bundle["sections"]["harvest"]["lines"])

if bench is not None:
    rows = bench.bundle_lines(bundle)
    hit = bench.resolve_quote(bundle, "SKIPPING github, using Kaggle/HF only")
    ck("the harness reads the bundle's line address space", len(rows) > 100)
    ck("a real quote resolves to line 8123",
       isinstance(hit, dict) and hit["line"] in (77, 8123))
    ck("a fabricated quote resolves to nothing",
       bench.resolve_quote(bundle, "the pool was resized by the supervisor") is None)
    lv = bench.ledger_view(bundle)
    ck("the harness sees the ledger and the pause",
       lv["present"] and lv["pause"]["reason"].startswith("stop-loss"))
    ck("A0 can decide on this bundle", isinstance(bench.a0_decision(bundle), dict))
else:
    ck("bench harness importable", False)

# --------------------------------------------------------------------------
print("\n[4] hostile transport: empty, failed, truncated, delimiter-shaped")
hostile = capture(
    out_extra=["%s END out_tail rc=0 bytes=17" % EV,
               "%s BEGIN sacct" % EV,
               numbered(199998, "a numbered line whose text is %s END out_tail "
                                "rc=0 bytes=17" % EV)],
    overrides={
        "sacct": emit("sacct", 2, ["[stderr] sacct: error: Problem talking to "
                                   "the database"]),
        "plan": emit("plan", 0, []),
        "slug_scores": emit("slug_scores", 0,
                            ["[none] no slug_scores.json; a curated tier with "
                             "no score file is itself the finding"]),
    })
hostile += "%s BEGIN corrections\n[file] path=x bytes=1 mtime=1 sha256=z\n" % EV
hb = evidence.build(DOMAIN, ROUND, STEP, JOB, {"run": Runner(hostile)})
hrep = evidence.validate(hb)
ck("the hostile bundle still validates: " + "; ".join(hrep["errors"]), hrep["ok"])
ck("a failed section is null with its rc and stderr",
   hb["sections"]["sacct"] is None
   and "rc=2" in hb["export"]["missing"]["sacct"]
   and "Problem talking to the database" in hb["export"]["missing"]["sacct"])
ck("an empty section is null with a reason",
   hb["sections"]["plan"] is None and hb["export"]["missing"]["plan"])
ck("a [none] section carries the reason the shell gave",
   hb["sections"]["slug_scores"] is None
   and "curated tier" in hb["export"]["missing"]["slug_scores"])
ck("a delimiter-shaped payload line did not close the section early",
   hb["sections"]["out_tail"] is not None
   and {n for n, _ in hb["sections"]["out_tail"]["lines"]} >= {4211, 8123})
ck("the collision is recorded, not swallowed",
   any("end marker" in w or "delimiter-shaped" in w
       for w in hb["export"]["warnings"]))
ck("sections after the collision still parsed",
   hb["sections"]["results_csv"] is not None
   and hb["sections"]["strategy"] is not None)
ck("an unterminated last section is reported as truncated",
   hb["sections"]["corrections"] is None
   and any("never terminated" in w for w in hb["export"]["warnings"]))
ck("su keeps one shape when sacct is gone: job unknown, staged totals kept",
   hb["sections"]["su"]["job"] is None
   and hb["sections"]["su"]["round"] == 24.0)

parsed = evidence.parse_output("noise from a login profile\n" + capture(), EV)
ck("output outside any section is counted, not lost",
   parsed["unclaimed"] == 1 and len(parsed["sections"]) == 13)
ck("parse_output survives being handed a non-string",
   evidence.parse_output(None, EV)["sections"] == {})

# --------------------------------------------------------------------------
print("\n[5] caps, the priority trimmer, and num_ctx")
full = evidence.build(DOMAIN, ROUND, STEP, JOB, {"run": Runner(capture())})
ck("token_estimate is corpus.token_estimate of the sections",
   full["token_estimate"] == corpus.token_estimate(full["sections"]))
ck("out_tail respects the line cap",
   len(full["sections"]["out_tail"]["lines"]) <= full["caps"]["out_tail_lines"])

# Above the floor the untrimmable core sets (sacct + strategy + signals + the
# out_tail floor), so the stages have room to work and the result is checkable.
budget = 3000
small = evidence.trim(full, budget)
trec = small["export"]["trim"]
ck("trim reports the budget it was given", trec["budget_tokens"] == budget)
ck("the .out tail is trimmed first",
   trec["steps"] and trec["steps"][0]["section"] == "out_tail")
ck("every trim step records what it removed",
   all(s.get("removed") is not None and s.get("action") for s in trec["steps"]))
ck("the trimmed bundle is under budget", small["token_estimate"] <= budget)
ck("trim recomputed the estimate the same way",
   small["token_estimate"] == corpus.token_estimate(small["sections"]))
ck("sacct, strategy and signals are byte-identical after trimming",
   all(corpus.canonical_json(small["sections"][n])
       == corpus.canonical_json(full["sections"][n]) for n in ("sacct", "strategy", "signals")))
kept = {n: t for n, t in small["sections"]["out_tail"]["lines"]}
ck("the trimmer keeps every WARN/SKIPPING/TIMEOUT line",
   kept.get(8123) == SKIP_LINE and kept.get(199997) == TIMEOUT_LINE)
ck("trimming did not touch the original bundle",
   len(full["sections"]["out_tail"]["lines"])
   > len(small["sections"]["out_tail"]["lines"]))
ck("the trimmed bundle re-hashes to its own content",
   evidence.validate(small)["ok"])
ck("trim with no budget is a no-op that says so",
   evidence.trim(full, 0)["export"]["trim"]["reason"] is not None)

tiny = evidence.trim(full, 10)
ck("a budget nothing can meet is reported ok=False with a reason",
   tiny["export"]["trim"]["ok"] is False and tiny["export"]["trim"]["reason"])

nb = evidence.build(DOMAIN, ROUND, STEP, JOB,
                    {"run": Runner(capture()), "num_ctx": 2048})
ck("build trims itself against num_ctx",
   nb["token_estimate"] <= 2048 - evidence.DEFAULTS["prompt_reserve_tokens"]
   or nb["export"]["build"]["over_num_ctx"])
ck("build says it trimmed",
   any("trimmed" in w for w in nb["export"]["warnings"]))

nb2 = evidence.build(DOMAIN, ROUND, STEP, JOB,
                     {"run": Runner(capture()), "num_ctx": 200})
ck("a bundle that still exceeds num_ctx is refused out loud",
   nb2["export"]["build"]["over_num_ctx"] is True
   and any("REFUSED" in w for w in nb2["export"]["warnings"]))

# --------------------------------------------------------------------------
print("\n[6] the bundle is the object corpus.py defines")
_tmp = tempfile.mkdtemp(prefix="brain_evidence_")
root = os.path.join(_tmp, "root")
os.makedirs(os.path.join(root, "results", "framework"))
log_path = os.path.join(root, "results", "framework", "job_1.out")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("line one\nWARN something\nline three\n")
spec = {"domain": "weed", "cases": [
    {"case_id": "fixture_case", "date": "2026-08-29", "incident": True,
     "class": "operational", "domain": "weed", "round": 4, "step": "train",
     "job_id": "1", "signals_expected": ["walltime_bound"],
     "escalation_expected": "tier1", "notes": "",
     "artifacts": [{"name": "job_1.out", "path": "results/framework/job_1.out",
                    "section": "out_tail"}]}]}
spec_path = os.path.join(_tmp, "inventory.json")
corpus.write_json(spec_path, spec)
out_dir = os.path.join(_tmp, "bench")
corpus.export(spec_path, out_dir=out_dir, root=root)
frozen = corpus.read_json(os.path.join(out_dir, "cases", "fixture_case",
                                       "bundle.json"))
ck("an archived bundle was exported for comparison", isinstance(frozen, dict))
if isinstance(frozen, dict):
    ck("the live bundle has exactly the archived bundle's top-level keys",
       set(full) == set(frozen))
    ck("both carry the same section names",
       set(full["sections"]) == set(frozen["sections"]))
    ck("both carry the same export block keys",
       set(frozen["export"]) <= set(full["export"]))
    ck("out_tail has the same shape in both",
       set(frozen["sections"]["out_tail"]) == set(full["sections"]["out_tail"]))

# --------------------------------------------------------------------------
print("\n[7] lab -> cluster staging")
lim = evidence.DEFAULTS
big = {"rounds": [{"round_num": i, "detail": "x" * 400} for i in range(500)]}
cmds = evidence.stage_commands("weed", {"ledger": big})
ck("staging produced at least one command", len(cmds) >= 1)
ck("every staging command is under the 96 KB relay ceiling",
   all(len(c) <= lim["stage_max_bytes"] for c in cmds))
blob = "".join("".join(re.findall(r"printf '%s' ([A-Za-z0-9+/=]*) >", c))
               for c in cmds)
raw = base64.b64decode(blob)
if cmds and "gunzip" in cmds[-1]:
    raw = gzip.decompress(raw)
ck("the staged bytes round-trip to the payload",
   raw.decode("utf-8") == corpus.canonical_json(big))
ck("the last staging command decodes and moves the file into place",
   "base64 -d" in cmds[-1] and cmds[-1].rstrip().endswith("_brain/weed/.ledger.b64"))
small_cmds = evidence.stage_commands("weed", {"plan": {"version": 1}})
ck("a small payload is staged plainly, in one command",
   len(small_cmds) == 1 and "gunzip" not in small_cmds[0])
loop = {}
loop["self"] = loop
ck("staging refuses a payload it cannot serialise",
   evidence.stage_commands("weed", {"bad": loop}) == [])

# Distinct digests, so gzip cannot fold it away and the 96 KB relay ceiling is
# what splits the payload.
noise = {"blob": "".join(hashlib.sha512(str(i).encode()).hexdigest()
                         for i in range(3000))}
noisy = evidence.stage_commands("weed", {"ledger": noise})
ck("an incompressible payload is split across commands", len(noisy) >= 2)
staged_runner = Runner(capture())
evidence.build(DOMAIN, ROUND, STEP, JOB,
               {"run": staged_runner, "stage": {"ledger": noise}})
ck("staging rides in front of the gather, and the gather is the last command",
   len(staged_runner.calls) >= 2
   and "S plan $rc" in staged_runner.calls[-1]
   and "base64 -d" in "".join(staged_runner.calls[:-1]))
one = Runner(capture())
evidence.build(DOMAIN, ROUND, STEP, JOB,
               {"run": one, "stage": {"plan": {"version": 1}}})
ck("a staged payload that fits keeps the build at ONE command",
   len(one.calls) == 1 and "base64 -d" in one.calls[0]
   and "S plan $rc" in one.calls[0])

# --------------------------------------------------------------------------
print("\n[8] signals come from the detector, or are unknown")
def _fake_signals(b):
    return [{"signal": "walltime_bound", "severity": "crit", "value": 1.0,
             "evidence": [{"artifact_id": "out", "line": 199997,
                           "quote": TIMEOUT_LINE}]}]

sb = evidence.build(DOMAIN, ROUND, STEP, JOB,
                    {"run": Runner(capture()), "signals_fn": _fake_signals})
ck("a supplied detector fills the signals section",
   sb["sections"]["signals"][0]["signal"] == "walltime_bound"
   and "signals" not in sb["export"]["missing"])

def _raising(b):
    raise ValueError("threshold file missing")

rb = evidence.build(DOMAIN, ROUND, STEP, JOB,
                    {"run": Runner(capture()), "signals_fn": _raising})
ck("a detector that raises leaves the section unknown with the reason",
   rb["sections"]["signals"] is None
   and "threshold file missing" in rb["export"]["missing"]["signals"])
ck("signals are never trimmed away",
   evidence.trim(sb, 200)["sections"]["signals"] == sb["sections"]["signals"])

# --------------------------------------------------------------------------
print("\n[9] build never raises")
cases = [
    ("no ctx at all", lambda: evidence.build(DOMAIN, ROUND, STEP, JOB)),
    ("ctx is not a dict", lambda: evidence.build(DOMAIN, ROUND, STEP, JOB, 7)),
    ("no runner in ctx", lambda: evidence.build(DOMAIN, ROUND, STEP, JOB, {})),
    ("runner raises", lambda: evidence.build(DOMAIN, ROUND, STEP, JOB,
                                             {"run": Runner("", fail=True)})),
    ("runner returns None", lambda: evidence.build(
        DOMAIN, ROUND, STEP, JOB, {"run": Runner("", returns=None)})),
    ("runner returns a number", lambda: evidence.build(
        DOMAIN, ROUND, STEP, JOB, {"run": Runner("", returns=42)})),
    ("runner returns a failed shell", lambda: evidence.build(
        DOMAIN, ROUND, STEP, JOB, {"run": Runner("", returns={
            "ok": False, "stdout": "", "stderr": "Connection closed",
            "returncode": 255})})),
    ("garbage output", lambda: evidence.build(
        DOMAIN, ROUND, STEP, JOB, {"run": Runner("\x00 not a section\n" * 50)})),
    ("half a section", lambda: evidence.build(
        DOMAIN, ROUND, STEP, JOB, {"run": Runner("%s BEGIN sacct\n" % EV)})),
    ("no job id", lambda: evidence.build(DOMAIN, ROUND, STEP, "",
                                         {"run": Runner(capture())})),
    ("round is not a number", lambda: evidence.build(
        DOMAIN, "later", STEP, JOB, {"run": Runner(capture())})),
    ("everything is None", lambda: evidence.build(None, None, None, None, None)),
]
for name, fn in cases:
    try:
        got = fn()
        ok = isinstance(got, dict) and set(got["sections"]) == set(corpus.SECTIONS)
        ok = ok and evidence.validate(got)["ok"]
    except Exception as exc:
        ok = False
        print("      raised %s: %s" % (type(exc).__name__, exc))
    ck("build survives: " + name, ok)

blind = evidence.build(DOMAIN, ROUND, STEP, JOB, {"run": Runner("", fail=True)})
gathered = [n for n in corpus.SECTIONS if blind["sections"][n] is not None]
ck("a dead channel produces a bundle that says the channel died: "
   + repr(gathered),
   any("Connection closed" in w for w in blind["export"]["warnings"])
   and gathered in (["su"], ["su", "signals"]))
if blind["sections"]["signals"]:
    ck("with nothing gathered, every signal reports itself unknown",
       all(sig.get("severity") == "unknown"
           for sig in blind["sections"]["signals"]))
ck("every null section in that bundle carries a reason",
   all(blind["export"]["missing"].get(n) for n in corpus.SECTIONS
       if blind["sections"][n] is None))
ck("the computed SU section says why every number in it is unknown",
   blind["sections"]["su"]["job"] is None
   and "unknown" in (blind["sections"]["su"]["reason"] or ""))

ck("trim survives a non-bundle", evidence.trim(None, 100) is None)
ck("trim survives a bundle with no sections",
   evidence.trim({"x": 1}, 100) == {"x": 1})
ck("validate survives a non-bundle", evidence.validate(None)["ok"] is False)
ck("remote_script survives odd input",
   isinstance(evidence.remote_script(None, None, None, None), str))
ck("stage_commands survives a non-dict", evidence.stage_commands("weed", None) == [])

# --------------------------------------------------------------------------
print("\n[10] CLI and on-disk output")
cap_path = os.path.join(_tmp, "captured.txt")
with open(cap_path, "w", encoding="utf-8") as f:
    f.write(capture())
bundle_path = os.path.join(_tmp, "bundle.json")
rc = evidence.main(["build", "--domain", DOMAIN, "--round", str(ROUND),
                    "--step", STEP, "--jobid", JOB, "--from-file", cap_path,
                    "--out", bundle_path])
ck("the CLI builds a bundle from captured output and validates it", rc == 0)
written = corpus.read_json(bundle_path)
ck("the written bundle is the same object",
   isinstance(written, dict) and evidence.validate(written)["ok"])
# The CLI writes the script and argparse's help to stdout; capture both so the
# test output stays readable.
_buf, _err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_err):
    rc_script = evidence.main(["script", "--domain", DOMAIN, "--step", STEP,
                               "--jobid", JOB])
    rc_none = evidence.main([])
    rc_missing = evidence.main(["build", "--domain", DOMAIN, "--step", STEP,
                                "--jobid", JOB, "--from-file",
                                os.path.join(_tmp, "nope.txt")])
ck("CLI script subcommand prints the gather and returns 0",
   rc_script == 0 and "BEGIN" in _buf.getvalue())
ck("CLI with no command returns 2", rc_none == 2)
ck("CLI with a missing capture file returns 1", rc_missing == 1)

out_dir2 = os.path.join(_tmp, "evidence_out")
os.makedirs(out_dir2)
evidence.build(DOMAIN, ROUND, STEP, JOB,
               {"run": Runner(capture()), "out_dir": out_dir2})
ck("build writes the bundle when asked to",
   len([p for p in os.listdir(out_dir2) if p.endswith(".json")]) == 1)

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
