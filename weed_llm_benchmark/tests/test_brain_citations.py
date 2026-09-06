#!/usr/bin/env python3
"""Unit tests for the citation validator (no cluster / Mongo / network).

The fixture is the campaign's own worked example: the 2026-08-29 pair 44727703 /
44767709, whose 12 h training jobs died at the wall because the merged pool had
grown to 8,583 iterations per epoch, and whose harvest log carries the chronic
`SKIPPING github` warning that is a designed control, not an incident.

Covers what a hostile reviewer would attack first:
  * an exact quote resolves to an ABSOLUTE (artifact, line) address, never to an
    index into the trimmed tail the model was shown
  * a reflowed and re-cased quote still resolves to the same address
  * a quote under the pre-registered minimum is refused, with the reason
  * a fabricated quote does not resolve and the finding is rejected
    `rejected_unverifiable`
  * a quote that occurs twice returns the first occurrence and records the count
  * a quote spanning two lines is refused and told to cite each line separately
  * a quote that resolves but is not load-bearing is counted apart from the two
    citation failures - the model read the evidence and drew on the wrong part
  * evidence-hit-rate arithmetic, including distinct addresses and an empty truth
  * every entry point survives a malformed verdict instead of raising
  * the bundle shape agrees with the frozen definition in corpus.py, end to end

Run:  python3 tests/test_brain_citations.py
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import citations  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


# Thresholds must come from the module, not from whatever the operator happens
# to have exported in this shell.
for _k in ("BRAIN_CITATION_MIN_QUOTE_CHARS", "BRAIN_CITATION_SPAN_WINDOW_LINES"):
    os.environ.pop(_k, None)

_tmp = tempfile.mkdtemp(prefix="brain_citations_")


# ---- fixture: the 2026-08-29 bundle ---------------------------------------
# Line numbers are the ones the file itself has. The first kept line is 8118, so
# any implementation that answered with a position inside the `lines` array
# would be caught by the very first assertion.
OUT_ID = "m1_merged_rndtrain_s1_44727703.out"
TIMEOUT_LINE = (
    "slurmstepd: error: *** JOB 44727703 ON v034 CANCELLED AT "
    "2026-08-29T17:53:11 DUE TO TIME LIMIT ***")
ITER_LINE = "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size"
PROGRESS_LINE = "      24/60      21.4G     0.8123     0.5412     0.9011        184        640"
POOL_LINE = "[pool] merged dataset: 8583 iterations/epoch (previous round 6246)"
REPEATED_LINE = "[train] resuming from checkpoint last.pt with the identical command"
HARVEST_ID = "v3_0_43_brain_harvest_45011002.out"
SOCKS_LINE = ("[net] WARN: SOCKS proxy via bridges2-login011 failed "
              "(compute→login SSH disabled) — SKIPPING github, "
              "using Kaggle/HF only")


def make_bundle():
    """A bundle in the shape corpus.py freezes: 14 sections, out_tail line-addressed."""
    sections = {name: None for name in citations.SECTION_ORDER}
    sections["out_tail"] = {
        "artifact_id": OUT_ID,
        "path": "<REPO>/results/framework/" + OUT_ID,
        "sha256": "a" * 64,
        "lines": [
            [8118, ITER_LINE],
            [8119, POOL_LINE],
            [8120, PROGRESS_LINE],
            [8121, REPEATED_LINE],
            [8122, "[train] 24 of 60 epochs done, 11.6 h elapsed"],
            [8123, TIMEOUT_LINE],
            # A second, non-adjacent occurrence of the resume line: an ambiguous
            # quote must return the first and say there were two.
            [8140, REPEATED_LINE],
        ],
    }
    # sacct as corpus.py stores it: parsed rows, no line numbers. Deliberately
    # not citable - there is no address to check a quote against.
    sections["sacct"] = [
        {"JobID": "44727703", "State": "TIMEOUT", "Elapsed": "12:00:18",
         "Timelimit": "12:00:00", "ExitCode": "0:0"},
        {"JobID": "44767709", "State": "TIMEOUT", "Elapsed": "12:00:20",
         "Timelimit": "12:00:00", "ExitCode": "0:0"},
    ]
    # A second line-addressed artifact in another section: the harvest log that
    # carries the chronic degraded-source warning.
    sections["harvest"] = {
        "artifact_id": HARVEST_ID,
        "sha256": "b" * 64,
        "lines": [[41, SOCKS_LINE],
                  [42, "[harvest] kaggle: 37 candidates, huggingface: 12 candidates"]],
    }
    sections["results_csv"] = {"rows": 24, "best": 0.5951, "last": 0.5904,
                               "epoch_time_s": 1741.0, "mtime": 1756500791.0}
    return {"bundle_id": "sched-walltime-double-timeout-20260829", "sha256": "c" * 64,
            "domain": "weed", "round": 4, "step": "train",
            "built_ts": "2026-08-29", "sections": sections,
            "token_estimate": 900, "caps": {"out_tail_lines": 400}}


BUNDLE = make_bundle()
# The truth's load-bearing lines for this case: the cancellation line and the
# pool-growth line. Line 8121 is real and quotable and is NOT load-bearing.
LOAD_BEARING = [{"artifact": OUT_ID, "line": 8123},
                {"artifact": OUT_ID, "line": 8119}]


# ---- the address space ----------------------------------------------------
idx = citations.index_bundle(BUNDLE)
ck("index reads the bundle's sections", idx["source"] == "bundle.sections")
ck("index finds both line-addressed artifacts", len(idx["artifacts"]) == 2)
ck("index finds every citable line", len(idx["lines"]) == 9)
ck("line numbers are absolute, not tail-relative",
   idx["lines"][0]["line"] == 8118 and idx["lines"][0]["artifact_id"] == OUT_ID)
ck("parsed sacct rows contribute no citable line",
   all(r["section"] != "sacct" for r in idx["lines"]))
ck("sections are walked in the frozen order (out_tail before harvest)",
   [r["section"] for r in idx["lines"]][0] == "out_tail"
   and [r["section"] for r in idx["lines"]][-1] == "harvest")

bad_rows = citations.index_bundle({"sections": {"out_tail": {
    "artifact_id": "x.out",
    "lines": [[0, "line number zero is not an address"],
              ["8123", "a string line number is not an address"],
              [8124, None], "not a row", [8125, "kept"]]}}})
ck("a row without a positive integer address is skipped",
   len(bad_rows["lines"]) == 1 and bad_rows["lines"][0]["line"] == 8125)
ck("skipped rows are counted with a reason, never silently dropped",
   bad_rows["skipped"] and bad_rows["skipped"][0]["rows"] == 4
   and "positive integer" in bad_rows["skipped"][0]["reason"])

# The section vocabulary is duplicated in citations.py so it imports without
# corpus.py; this is what stops the copy drifting.
from weed_optimizer_framework.tools.brain import corpus  # noqa: E402
ck("SECTION_ORDER still equals the frozen corpus.SECTIONS",
   citations.SECTION_ORDER == corpus.SECTIONS)


# ---- resolve: exact ---------------------------------------------------------
hit = citations.resolve(BUNDLE, TIMEOUT_LINE)
ck("an exact quote resolves", isinstance(hit, dict))
ck("it resolves to the absolute address",
   hit["artifact_id"] == OUT_ID and hit["line"] == 8123)
ck("it returns the matched line verbatim", hit["matched"] == TIMEOUT_LINE)
ck("an unambiguous match says so", hit["matches"] == 1 and hit["ambiguous"] is False)

part = citations.resolve(BUNDLE, "CANCELLED AT 2026-08-29T17:53:11 DUE TO TIME LIMIT")
ck("a substring of one line resolves to that line",
   part is not None and part["line"] == 8123)

ck("a quote from a second artifact resolves into that artifact",
   (citations.resolve(BUNDLE, "SKIPPING github, using Kaggle/HF only") or {})
   .get("artifact_id") == HARVEST_ID)
ck("resolving the chronic warning is not a verdict about it",
   (citations.resolve(BUNDLE, SOCKS_LINE) or {}).get("line") == 41)


# ---- resolve: whitespace-reflowed and re-cased ------------------------------
reflowed = ("slurmstepd: error: *** JOB 44727703 ON v034\n"
            "    CANCELLED   AT 2026-08-29T17:53:11\tDUE TO TIME LIMIT ***")
hit2 = citations.resolve(BUNDLE, reflowed)
ck("a reflowed quote resolves to the same address",
   hit2 is not None and (hit2["artifact_id"], hit2["line"]) == (OUT_ID, 8123))
hit3 = citations.resolve(BUNDLE, "job 44727703 on v034 cancelled at 2026-08-29t17:53:11")
ck("a re-cased quote resolves to the same address",
   hit3 is not None and hit3["line"] == 8123)
ck("the whole-file collapse is not what matched",
   citations.resolve(BUNDLE, " ".join([POOL_LINE, PROGRESS_LINE])) is None)


# ---- resolve: too short -----------------------------------------------------
ck("the pre-registered minimum is 20 characters",
   citations.threshold("min_quote_chars") == 20)
short = "TIME LIMIT ***"                      # 14 characters, really in the line
ck("a quote under the minimum is refused", citations.resolve(BUNDLE, short) is None)
det_short = citations.resolve_detail(BUNDLE, short)
ck("the refusal says it was too short", det_short["reason"] == "quote_too_short")
ck("the refusal states both lengths",
   "14 characters" in det_short["detail"] and "20" in det_short["detail"])
ck("a 20-character quote is accepted",
   citations.resolve(BUNDLE, "DUE TO TIME LIMIT **") is not None)
ck("a 19-character quote is not",
   citations.resolve(BUNDLE, "DUE TO TIME LIMIT *") is None)


# ---- resolve: does not resolve ---------------------------------------------
fabricated = "CUDA out of memory: tried to allocate 2.00 GiB on device 0"
ck("a fabricated quote does not resolve", citations.resolve(BUNDLE, fabricated) is None)
det_fab = citations.resolve_detail(BUNDLE, fabricated)
ck("the refusal says unresolved with a count of what was searched",
   det_fab["reason"] == "unresolved" and "9 indexed lines" in det_fab["detail"])
ck("a real fact in a non-citable section still does not resolve",
   citations.resolve(BUNDLE, "State TIMEOUT Elapsed 12:00:18 Timelimit 12:00:00") is None)
ck("a non-string quote is refused, not coerced",
   citations.resolve(BUNDLE, 44727703) is None
   and citations.resolve_detail(BUNDLE, None)["reason"] == "no_quote")
ck("a bundle with no line-addressed evidence says so",
   citations.resolve_detail({"sections": {"sacct": []}}, fabricated)["detail"]
   .startswith("the bundle carries no line-addressed evidence"))


# ---- resolve: spanning two lines -------------------------------------------
spanning = "11.6 h elapsed slurmstepd: error: *** JOB 44727703 ON v034 CANCELLED"
det_span = citations.resolve_detail(BUNDLE, spanning)
ck("a quote spanning two lines is refused", det_span["hit"] is None)
ck("and is told to cite each line separately",
   "spans more than one line" in det_span["detail"]
   and "8122" in det_span["detail"] and "8123" in det_span["detail"])
ck("splitting it into two quotes resolves both",
   (citations.resolve(BUNDLE, "24 of 60 epochs done, 11.6 h elapsed") or {}).get("line") == 8122
   and (citations.resolve(BUNDLE, "JOB 44727703 ON v034 CANCELLED AT") or {}).get("line") == 8123)
ck("a span is not claimed across an omitted range",
   "spans more than one line" not in
   citations.resolve_detail(BUNDLE, REPEATED_LINE + " " + SOCKS_LINE)["detail"])


# ---- resolve: ambiguous -----------------------------------------------------
amb = citations.resolve(BUNDLE, REPEATED_LINE)
ck("an ambiguous quote returns the first occurrence in bundle order",
   amb is not None and amb["line"] == 8121)
ck("and records how many places it matched",
   amb["matches"] == 2 and amb["ambiguous"] is True)
det_amb = citations.resolve_detail(BUNDLE, REPEATED_LINE)
ck("both addresses are reported",
   [a["line"] for a in det_amb["addresses"]] == [8121, 8140])


# ---- validate_verdict -------------------------------------------------------
GOOD = {"verdict": "issue",
        "findings": [{"signal": "walltime_bound", "severity": "crit",
                      "quote": TIMEOUT_LINE,
                      "diagnosis": "the job was killed at its 12 h walltime"},
                     {"signal": "pool_growth", "severity": "warn",
                      "quote": POOL_LINE,
                      "diagnosis": "the merged pool grew to 8,583 it/epoch"}],
        "corrections": [], "escalate": {"to": "tier1", "reason": ""},
        "confidence": 0.8}
rep = citations.validate_verdict(BUNDLE, GOOD, load_bearing_lines=LOAD_BEARING)
ck("a fully grounded verdict validates", rep["ok"] is True)
ck("both findings are accepted", len(rep["findings"]) == 2 and rep["rejected"] == [])
ck("each accepted finding carries its address",
   [(f["artifact_id"], f["line"]) for f in rep["findings"]]
   == [(OUT_ID, 8123), (OUT_ID, 8119)])
ck("each accepted finding keeps the model's own object",
   rep["findings"][0]["finding"]["diagnosis"].startswith("the job was killed"))
ck("both cited lines are load-bearing",
   all(f["load_bearing"] is True for f in rep["findings"]))
ck("the stats report the hit", rep["stats"]["load_bearing_hit"] == 2
   and rep["stats"]["not_load_bearing"] == 0
   and rep["stats"]["evidence_hit_rate"] == 1.0)
ck("the stats record which rule they were scored under",
   rep["stats"]["min_quote_chars"] == 20
   and rep["stats"]["tool_version"] == citations.TOOL_VERSION)

# One of each failure, in one verdict: the three are counted apart.
MIXED = {"verdict": "issue", "findings": [
    {"signal": "walltime_bound", "severity": "crit", "quote": TIMEOUT_LINE},
    {"signal": "job_unknown", "severity": "warn", "quote": ""},
    {"signal": "budget", "severity": "warn", "quote": fabricated},
    {"signal": "gate_noop", "severity": "warn", "quote": short},
    {"signal": "plateau", "severity": "warn", "quote": REPEATED_LINE},
]}
mix = citations.validate_verdict(BUNDLE, MIXED, load_bearing_lines=LOAD_BEARING)
st = mix["stats"]
ck("a verdict with any rejected finding is not ok", mix["ok"] is False)
ck("findings with no quote are counted on their own", st["no_quote"] == 1)
ck("findings whose quote does not resolve are counted on their own",
   st["unresolved"] == 1)
ck("a too-short quote is counted apart from a fabricated one",
   st["quote_too_short"] == 1)
ck("a resolved quote that is not load-bearing is NOT a citation failure",
   st["not_load_bearing"] == 1 and st["accepted"] == 2)
ck("that finding is still actionable and marked",
   [f["load_bearing"] for f in mix["findings"]] == [True, False])
ck("every rejected finding is recorded rejected_unverifiable",
   all(r["status"] == "rejected_unverifiable" for r in mix["rejected"])
   and len(mix["rejected"]) == 3)
ck("each rejection names which failure it was",
   sorted(r["reason"] for r in mix["rejected"])
   == ["no_quote", "quote_too_short", "unresolved"])
ck("each rejection keeps the index of the finding it refused",
   [r["index"] for r in mix["rejected"]] == [1, 2, 3])
ck("an ambiguous accepted citation is counted", st["ambiguous"] == 1)
ck("the hit rate counts distinct load-bearing lines cited",
   st["load_bearing_hit"] == 1 and st["evidence_hit_rate"] == 0.5)

blind = citations.validate_verdict(BUNDLE, GOOD)
ck("without truth the load-bearing counts are None, not zero",
   blind["stats"]["load_bearing_hit"] is None
   and blind["stats"]["not_load_bearing"] is None
   and blind["stats"]["evidence_hit_rate"] is None)
ck("and the stats say the truth was not supplied",
   blind["stats"]["load_bearing_source"] == "not supplied")
ck("a live round with no truth still validates its citations",
   blind["ok"] is True and len(blind["findings"]) == 2)
ck("an empty findings list is well-formed",
   citations.validate_verdict(BUNDLE, {"verdict": "ok", "findings": []})["ok"] is True)


# ---- malformed shapes are rejected, never raised ---------------------------
for name, obj in (("None", None), ("a list", [1, 2]), ("a string", "issue"),
                  ("an int", 7), ("findings as a string",
                                  {"verdict": "issue", "findings": "walltime"}),
                  ("findings as an object",
                   {"verdict": "issue", "findings": {"quote": TIMEOUT_LINE}}),
                  ("no findings key", {"verdict": "issue"})):
    r = citations.validate_verdict(BUNDLE, obj, load_bearing_lines=LOAD_BEARING)
    ck("a verdict that is %s is rejected, not raised" % name,
       isinstance(r, dict) and r["ok"] is False and r["stats"]["errors"])

odd = citations.validate_verdict(BUNDLE, {"findings": [
    "not an object", None, {"quote": TIMEOUT_LINE}]}, load_bearing_lines=LOAD_BEARING)
ck("a finding that is not an object is rejected as malformed",
   odd["stats"]["malformed"] == 2 and odd["stats"]["accepted"] == 1)
ck("a malformed finding is recorded, not dropped", len(odd["rejected"]) == 2)

for name, bad in (("None", None), ("a list", []), ("a string", "bundle"),
                  ("sections not an object", {"sections": 3})):
    r = citations.validate_verdict(bad, GOOD, load_bearing_lines=LOAD_BEARING)
    ck("a bundle that is %s is reported, not raised" % name,
       isinstance(r, dict) and r["ok"] is False)
ck("resolve against a broken bundle returns None",
   citations.resolve(None, TIMEOUT_LINE) is None
   and citations.resolve({"sections": 3}, TIMEOUT_LINE) is None)
ck("a truth entry with no line is reported, not raised",
   citations.validate_verdict(BUNDLE, GOOD, load_bearing_lines=[
       {"artifact": OUT_ID}, "not an object"])["stats"]["errors"])


# ---- evidence hit rate ------------------------------------------------------
def verdict_citing(*addresses):
    return {"findings": [{"signal": "s", "artifact_id": a, "line": n}
                         for a, n in addresses]}


ck("citing one of two load-bearing lines is 0.5",
   citations.evidence_hit_rate(verdict_citing((OUT_ID, 8123)), LOAD_BEARING) == 0.5)
ck("citing both is 1.0",
   citations.evidence_hit_rate(verdict_citing((OUT_ID, 8123), (OUT_ID, 8119)),
                               LOAD_BEARING) == 1.0)
ck("citing the same line twice is still 0.5",
   citations.evidence_hit_rate(verdict_citing((OUT_ID, 8123), (OUT_ID, 8123)),
                               LOAD_BEARING) == 0.5)
ck("citing only a line that is not load-bearing is 0.0",
   citations.evidence_hit_rate(verdict_citing((OUT_ID, 8121)), LOAD_BEARING) == 0.0)
ck("citing nothing is 0.0", citations.evidence_hit_rate({"findings": []},
                                                        LOAD_BEARING) == 0.0)
ck("an empty truth is 0.0, never 1.0",
   citations.evidence_hit_rate(verdict_citing((OUT_ID, 8123)), []) == 0.0
   and citations.evidence_hit_rate(verdict_citing((OUT_ID, 8123)), None) == 0.0)
ck("one of three is a third",
   abs(citations.evidence_hit_rate(
       verdict_citing((OUT_ID, 8123)),
       LOAD_BEARING + [{"artifact": HARVEST_ID, "line": 41}]) - 1.0 / 3.0) < 1e-12)
ck("an absolute path in the bundle matches a bare name in the truth",
   citations.evidence_hit_rate(
       verdict_citing(("/ocean/x/results/framework/" + OUT_ID, 8123)),
       LOAD_BEARING) == 0.5)
ck("the rate reads a validated verdict's resolved addresses",
   citations.evidence_hit_rate(rep, LOAD_BEARING) == 1.0)
ck("it reads the signal evidence shape too",
   citations.evidence_hit_rate(
       {"findings": [{"signal": "walltime_bound",
                      "evidence": [{"artifact_id": OUT_ID, "line": 8123,
                                    "quote": TIMEOUT_LINE}]}]}, LOAD_BEARING) == 0.5)
ck("an unresolved quote alone contributes nothing",
   citations.evidence_hit_rate({"findings": [{"quote": fabricated}]},
                               LOAD_BEARING) == 0.0)
ck("the rate agrees with validate_verdict's own stat",
   citations.evidence_hit_rate(mix, LOAD_BEARING) == mix["stats"]["evidence_hit_rate"])
ck("a malformed verdict scores 0.0 instead of raising",
   citations.evidence_hit_rate(None, LOAD_BEARING) == 0.0
   and citations.evidence_hit_rate("issue", LOAD_BEARING) == 0.0
   and citations.evidence_hit_rate({"findings": 5}, LOAD_BEARING) == 0.0)


# ---- thresholds are overridable, and the source is reported -----------------
ck("the default source is the code", citations.thresholds()["sources"]
   ["min_quote_chars"] == "default")
os.environ["BRAIN_CITATION_MIN_QUOTE_CHARS"] = "60"
ck("an environment override takes effect",
   citations.threshold("min_quote_chars") == 60
   and citations.resolve(BUNDLE, POOL_LINE[:40]) is None)
ck("and is reported as the source", citations.thresholds()["sources"]
   ["min_quote_chars"].startswith("env:"))
os.environ["BRAIN_CITATION_MIN_QUOTE_CHARS"] = "not a number"
ck("a nonsense override is ignored and reported",
   citations.threshold("min_quote_chars") == 20
   and any("not a positive integer" in e for e in citations.thresholds()["errors"]))
os.environ.pop("BRAIN_CITATION_MIN_QUOTE_CHARS")
ck("removing the override restores the pre-registered value",
   citations.threshold("min_quote_chars") == 20
   and citations.resolve(BUNDLE, POOL_LINE[:40]) is not None)
ck("every threshold carries its justification",
   all(len(s["why"]) > 80 for s in citations.THRESHOLDS.values()))


# ---- the bench bridge sees the same address --------------------------------
from weed_optimizer_framework.tools.brain import bench  # noqa: E402
ck("bench.resolve_quote (which prefers this module) agrees",
   bench.resolve_quote(BUNDLE, TIMEOUT_LINE) == {"artifact_id": OUT_ID.lower(),
                                                 "line": 8123})
ck("bench and this module agree on the minimum quote length",
   bench.MIN_QUOTE_CHARS == citations.threshold("min_quote_chars"))
ck("bench rejects the fabricated quote too",
   bench.resolve_quote(BUNDLE, fabricated) is None)


# ---- end to end against a bundle corpus.py actually wrote ------------------
ROOT = os.path.join(_tmp, "root")
os.makedirs(os.path.join(ROOT, "results", "framework"))
_lines = ["[train] epoch %d/60 done" % i for i in range(1, 8118)]
_lines += [ITER_LINE, POOL_LINE, PROGRESS_LINE, REPEATED_LINE,
           "[train] 24 of 60 epochs done, 11.6 h elapsed", TIMEOUT_LINE]
OUT_PATH = os.path.join(ROOT, "results", "framework", OUT_ID)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(_lines) + "\n")
ck("the fixture log puts the cancellation on line 8123", len(_lines) == 8123)

SPEC = {"domain": "weed", "cases": [{
    "case_id": "sched-walltime-double-timeout-20260829",
    "date": "2026-08-29", "incident": True, "class": "operational",
    "domain": "weed", "round": 4, "step": "train", "job_id": "44727703",
    "signals_expected": ["walltime_bound", "pool_growth"],
    "load_bearing_lines": [{"artifact": OUT_ID, "line": 8123},
                           {"artifact": OUT_ID, "line": 8119}],
    "escalation_expected": "tier1",
    "artifacts": [{"name": OUT_ID, "section": "out_tail",
                   "path": "results/framework/" + OUT_ID}]}]}
SPEC_PATH = os.path.join(_tmp, "inventory.json")
with open(SPEC_PATH, "w", encoding="utf-8") as f:
    json.dump(SPEC, f)
OUT_DIR = os.path.join(_tmp, "bench")
_rep = corpus.export(SPEC_PATH, out_dir=OUT_DIR, root=ROOT)
ck("corpus exported the case", _rep["ok"] is True)
_case = os.path.join(OUT_DIR, "cases", "sched-walltime-double-timeout-20260829")
with open(os.path.join(_case, "bundle.json"), "r", encoding="utf-8") as f:
    real_bundle = json.load(f)
with open(os.path.join(_case, "truth.json"), "r", encoding="utf-8") as f:
    real_truth = json.load(f)

real_hit = citations.resolve(real_bundle, TIMEOUT_LINE)
ck("a quote resolves inside a bundle corpus.py wrote",
   real_hit is not None and real_hit["line"] == 8123
   and real_hit["artifact_id"] == OUT_ID)
ck("the address is the line number of the ORIGINAL file",
   real_hit["line"] == len(_lines))
real_rep = citations.validate_verdict(
    real_bundle, GOOD, load_bearing_lines=real_truth["load_bearing_lines"])
ck("a grounded verdict validates against the exported case",
   real_rep["ok"] is True and real_rep["stats"]["evidence_hit_rate"] == 1.0)
ck("a fabricated finding is rejected against the exported case",
   citations.validate_verdict(real_bundle, {"findings": [{"quote": fabricated}]},
                              load_bearing_lines=real_truth["load_bearing_lines"]
                              )["stats"]["unresolved"] == 1)
ck("a stored artifact file indexes to the same addresses",
   [r["line"] for r in citations.index_numbered_text(
       OUT_ID, open(os.path.join(_case, "artifacts", OUT_ID + ".txt"),
                    encoding="utf-8").read())][-1] == 8123)


# ---- the command line ------------------------------------------------------
BUNDLE_PATH = os.path.join(_tmp, "bundle.json")
with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
    json.dump(BUNDLE, f)
VERDICT_PATH = os.path.join(_tmp, "verdict.json")
with open(VERDICT_PATH, "w", encoding="utf-8") as f:
    json.dump(GOOD, f)
BAD_VERDICT_PATH = os.path.join(_tmp, "bad_verdict.json")
with open(BAD_VERDICT_PATH, "w", encoding="utf-8") as f:
    json.dump({"findings": [{"quote": fabricated}]}, f)
TRUTH_PATH = os.path.join(_tmp, "truth.json")
with open(TRUTH_PATH, "w", encoding="utf-8") as f:
    json.dump({"load_bearing_lines": LOAD_BEARING}, f)

ck("resolve of a real quote exits 0",
   citations.main(["resolve", "--bundle", BUNDLE_PATH, "--quote", TIMEOUT_LINE]) == 0)
ck("resolve of a fabricated quote exits 1",
   citations.main(["resolve", "--bundle", BUNDLE_PATH, "--quote", fabricated]) == 1)
ck("validate of a grounded verdict exits 0",
   citations.main(["validate", "--bundle", BUNDLE_PATH, "--verdict", VERDICT_PATH,
                   "--truth", TRUTH_PATH]) == 0)
ck("validate of an unverifiable verdict exits 1",
   citations.main(["validate", "--bundle", BUNDLE_PATH, "--verdict",
                   BAD_VERDICT_PATH, "--truth", TRUTH_PATH]) == 1)
ck("index exits 0 on a bundle with citable lines",
   citations.main(["index", "--bundle", BUNDLE_PATH]) == 0)
ck("thresholds exits 0", citations.main(["thresholds", "--json"]) == 0)
ck("a missing bundle exits 1",
   citations.main(["resolve", "--bundle", os.path.join(_tmp, "nope.json"),
                   "--quote", TIMEOUT_LINE]) == 1)
ck("no command exits 2", citations.main([]) == 2)


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
