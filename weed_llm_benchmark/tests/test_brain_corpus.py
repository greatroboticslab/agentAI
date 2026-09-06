#!/usr/bin/env python3
"""Unit tests for the incident-corpus export (no cluster / Mongo / network).

Covers the properties a hostile reviewer would attack first, on synthetic
fixture logs in a temporary directory:
  * absolute line numbers survive the copy ("%06d\\t<line>", awk NR semantics)
  * the recorded sha256 is the ORIGINAL file's, not the numbered copy's
  * scrubbing rewrites usernames, /ocean and /jet/home prefixes, token shapes
    and password-shaped assignments, and counts what it did
  * a scrub target that survives scrubbing REFUSES the case instead of writing it
  * an artifact over 1 MB is stored as a head+tail excerpt that states its
    omitted ranges, keeps a load-bearing line, and records the full original
    sha256 and byte count
  * an incident with no surviving artifacts still exports as record-only, with
    an empty artifacts directory and a stated reason
  * two exports of the same inventory are byte-identical
  * freeze writes split.json once and refuses to overwrite it
  * verify re-hashes stored copies and originals and reports drift
  * no top-level entry point raises, whatever it is handed

Run:  python3 tests/test_brain_corpus.py
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import corpus  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


# The scrub configuration must come from the fixture, not from whatever the
# operator happens to have exported in this shell.
for _k in ("CORPUS_SCRUB_USERS", "CORPUS_SCRUB_LITERALS"):
    os.environ.pop(_k, None)

_tmp = tempfile.mkdtemp(prefix="brain_corpus_")
ROOT = os.path.join(_tmp, "root")
OUT = os.path.join(_tmp, "bench")
OUT2 = os.path.join(_tmp, "bench2")
os.makedirs(os.path.join(ROOT, "logs"))


def _w(rel, text):
    p = os.path.join(ROOT, rel)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---- fixtures -------------------------------------------------------------
# A job log carrying one instance of every scrub target, so the stored copy can
# be checked target by target.
JOB_LOG_LINES = [
    "starting rndtrain on %s/weed_optimizer_framework/tools/mega_trainer.py" % ROOT,
    "HOME=/jet/home/byler models=/ocean/projects/cis240145p/byler/ollama/models",
    "user byler submitted the step",
    "HF_TOKEN=hf_AAAABBBBCCCCDDDDEEEE downloading weights",
    "password=hunter2super api_key: sk-0123456789abcdefghij",
    "epoch 24/60 elapsed 11.4h",
    "slurmstepd: error: *** JOB 100 ON w001 CANCELLED AT 2026-08-29T17:53:00 DUE TO TIME LIMIT ***",
    "TIMEOUT",
]
JOB_LOG = _w("logs/job_100.out", "\n".join(JOB_LOG_LINES) + "\n")
SACCT = _w("logs/sacct_100.tsv",
           "JobID\tState\tElapsed\tAllocTRES\n"
           "100\tTIMEOUT\t12:00:00\tbilling=8,gres/gpu=1\n"
           "100.batch\tCANCELLED\t12:00:00\tcpu=8\n")
RESULTS = _w("logs/results_100.csv",
             "epoch,time,metrics/mAP50-95(B)\n"
             "1,1700.0,0.31\n2,3400.0,0.44\n3,5100.0,0.41\n")
TRACE = _w("logs/trace_100.jsonl",
           json.dumps({"kind": "epoch", "epoch": 1, "map50_95": 0.31}) + "\n" +
           json.dumps({"kind": "epoch", "epoch": 2, "map50_95": 0.44}) + "\n")
STRATEGY = _w("logs/strategy_100.json",
              json.dumps({"job_id": "100", "epochs_requested": 60,
                          "save_dir": "%s/results/framework/mega_iter" % ROOT}))
DIRTY = _w("logs/dirty_101.out",
           "run owned by byler2 on the shared account\nnothing else here\n")

# 40,000 lines of 32 bytes = 1.28 MB: over the excerpt threshold, under the
# read ceiling, so it exercises the head+tail path with a full read.
BIG_LINES = ["line %05d %s" % (i, "x" * 20) for i in range(1, 40001)]
BIG = _w("logs/big_200.out", "\n".join(BIG_LINES) + "\n")
BIG_CITE = 20000

CASE_TIMEOUT = "2026-08-29_train_timeout_100"
CASE_RECORD = "2026-07-01_harvest_booked_failed"
CASE_BIG = "2026-09-02_big_log"
CASE_DIRTY = "2026-08-30_unscrubbable"
CASE_HEALTHY = "2026-06-01_healthy_round1"

SPEC = {
    "root": ROOT,
    "domain": "weed",
    "scrub_users": ["byler"],
    "scrub_literals": ["hunter2super"],
    "cases": [
        {"case_id": CASE_TIMEOUT, "date": "2026-08-29", "incident": True,
         "class": "operational", "round": 4, "step": "train", "job_id": "100",
         "signals_expected": ["walltime_bound", "pool_growth"],
         "load_bearing_lines": [{"artifact": "job_100.out", "line": 7}],
         "acceptable_corrections": [{"action": "set_round_param",
                                     "params_range": {"epochs": [20, 30]},
                                     "risk": "R1"}],
         "escalation_expected": "tier1",
         "labels": {"pre_registered": {"detect": True, "correction_class": "config"}},
         "notes": "60 epochs on a pool that had grown",
         "artifacts": [
             {"name": "job_100.out", "path": "logs/job_100.out", "section": "out_tail"},
             {"name": "sacct_100.tsv", "path": "logs/sacct_100.tsv", "section": "sacct"},
             {"name": "results_100.csv", "path": "logs/results_100.csv",
              "section": "results_csv"},
             {"name": "trace_100.jsonl", "path": "logs/trace_100.jsonl", "section": "trace"},
             {"name": "strategy_100.json", "path": "logs/strategy_100.json",
              "section": "strategy"}],
         "sections": {"su": {"round": 24.0, "campaign": 215.0}}},

        {"case_id": CASE_RECORD, "date": "2026-07-01", "incident": True,
         "class": "code", "job_id": "44322382",
         "signals_expected": ["job_unknown"],
         "escalation_expected": "tier1",
         "notes": "COMPLETED harvest booked failed; the .out did not survive",
         "artifacts": [{"name": "job_44322382.out",
                        "path": "logs/gone_44322382.out", "section": "out_tail"}]},

        {"case_id": CASE_BIG, "date": "2026-09-02", "incident": True, "class": "config",
         "signals_expected": ["gate_noop"],
         "load_bearing_lines": [{"artifact": "big_200.out", "line": BIG_CITE}],
         "escalation_expected": "tier1",
         "artifacts": [{"name": "big_200.out", "path": "logs/big_200.out",
                        "section": "out_tail"}]},

        {"case_id": CASE_DIRTY, "date": "2026-08-30", "incident": True, "class": "code",
         "escalation_expected": "none",
         "artifacts": [{"name": "dirty_101.out", "path": "logs/dirty_101.out",
                        "section": "out_tail"}]},

        {"case_id": CASE_HEALTHY, "date": "2026-06-01", "incident": False,
         "class": "operational", "signals_expected": [], "escalation_expected": "none",
         "notes": "completed round #1 train step",
         "artifacts": [{"name": "job_100.out", "path": "logs/job_100.out",
                        "section": "out_tail"}]},
    ],
}
SPEC_PATH = os.path.join(_tmp, "inventory.json")
with open(SPEC_PATH, "w", encoding="utf-8") as f:
    json.dump(SPEC, f)


# ---- scrub_text / scrub_residue (pure) ------------------------------------
cfg = corpus.scrub_config(SPEC, ROOT)
s, counts = corpus.scrub_text("\n".join(JOB_LOG_LINES), cfg)
ck("scrub replaces the repo prefix", "<REPO>/weed_optimizer_framework" in s)
ck("scrub replaces /jet/home/<user>", "<HOME> models=" in s)
ck("scrub replaces /ocean/projects/<alloc>/<user>", "<HOME>/ollama/models" in s)
ck("scrub replaces a bare username", "user <USER> submitted" in s)
ck("scrub replaces an hf_ token", "hf_AAAABBBBCCCCDDDDEEEE" not in s)
ck("scrub replaces an sk- token", "sk-0123456789abcdefghij" not in s)
ck("scrub replaces a password literal", "hunter2super" not in s)
ck("scrub leaves no username substring", "byler" not in s)
ck("scrub counts every rule it fired",
   counts.get("repo_path") == 1 and counts.get("jet_home") == 1
   and counts.get("ocean_home") == 1 and counts.get("username") == 1
   and counts.get("token_hf") == 1 and counts.get("literal") == 1)
ck("scrub is idempotent", corpus.scrub_text(s, cfg)[0] == s)
ck("scrub leaves no residue on a clean line", corpus.scrub_residue(s, cfg) == [])

res = corpus.scrub_residue("job owned by byler2\nsecond line\n", cfg)
ck("residue check catches a username the word-bounded rule misses",
   len(res) == 1 and res[0]["rule"] == "username" and res[0]["line"] == 1)
ck("residue report carries no matched text",
   all("byler" not in json.dumps(r) for r in res))
ck("scrub keeps a non-secret assignment value alone",
   corpus.scrub_text('"token": null', cfg)[0] == '"token": null')
ck("scrub does not fire on tokens_in / token_estimate",
   corpus.scrub_text('"tokens_in": 1200, "token_estimate": 900', cfg)[0]
   == '"tokens_in": 1200, "token_estimate": 900')


# ---- export ---------------------------------------------------------------
rep = corpus.export(SPEC_PATH, out_dir=OUT, root=ROOT)
by_id = {c["case_id"]: c for c in rep["cases"]}
ck("export reports one row per case", len(rep["cases"]) == 5)
ck("export is not ok while a case is refused", rep["ok"] is False)
ck("export wrote the four exportable cases", rep["counts"]["written"] == 4)
ck("export refused exactly one case", rep["counts"]["refused"] == 1)
ck("export counts raw vs record-only",
   rep["counts"]["raw"] == 3 and rep["counts"]["record_only"] == 1)

CASE_DIR = os.path.join(OUT, "cases", CASE_TIMEOUT)
stored_log = os.path.join(CASE_DIR, "artifacts", "job_100.out.txt")
ck("case directory holds bundle, truth and artifacts",
   sorted(os.listdir(CASE_DIR)) == ["artifacts", "bundle.json", "truth.json"])
ck("stored artifact keeps the original name plus .txt", os.path.exists(stored_log))

bundle = json.loads(_read(os.path.join(CASE_DIR, "bundle.json")))
truth = json.loads(_read(os.path.join(CASE_DIR, "truth.json")))
arts = {a["artifact_id"]: a for a in bundle["export"]["artifacts"]}


# ---- absolute line numbers ------------------------------------------------
lines = _read(stored_log).split("\n")
if lines and lines[-1] == "":
    lines.pop()
ck("stored copy has one line per original line", len(lines) == len(JOB_LOG_LINES))
ck("line 1 is numbered 000001", lines[0].startswith("000001\t"))
ck("line 7 is numbered 000007 and is the line it was",
   lines[6] == "000007\t" + JOB_LOG_LINES[6])
ck("every stored line is numbered or an export marker",
   all(re.match(r"^\d{6}\t", ln) or ln.startswith("[") for ln in lines))
ck("the cited load-bearing line resolves in the stored copy",
   "CANCELLED AT" in lines[6] and int(lines[6][:6]) == 7)
ck("no unresolved citations reported for the timeout case",
   by_id[CASE_TIMEOUT]["unresolved_load_bearing"] == [])


# ---- sha256 of the ORIGINAL ----------------------------------------------
ck("artifact sha256 is the original file's",
   arts["job_100.out"]["sha256"] == _sha_file(JOB_LOG))
ck("artifact sha256 is NOT the numbered copy's",
   arts["job_100.out"]["sha256"] != _sha_file(stored_log))
ck("stored_sha256 is the numbered copy's",
   arts["job_100.out"]["stored_sha256"] == corpus.sha256_str(_read(stored_log)))
ck("artifact byte count is the original's",
   arts["job_100.out"]["bytes"] == os.path.getsize(JOB_LOG))
ck("artifact line count is the original's",
   arts["job_100.out"]["lines"] == len(JOB_LOG_LINES))


# ---- scrubbing of what is written ----------------------------------------
blob = _read(stored_log)
ck("stored copy contains no username", "byler" not in blob)
ck("stored copy contains no absolute cluster path",
   "/jet/home" not in blob and "/ocean/projects" not in blob and ROOT not in blob)
ck("stored copy contains no token", "hf_" not in blob and "sk-0123" not in blob)
ck("stored copy contains no password literal", "hunter2super" not in blob)
ck("stored copy shows the placeholders",
   "<REPO>" in blob and "<HOME>" in blob and "<USER>" in blob
   and "<REDACTED_TOKEN>" in blob)
ck("export reports the substitution count per case",
   by_id[CASE_TIMEOUT]["scrub_total"] >= 6
   and by_id[CASE_TIMEOUT]["scrub"].get("username", 0) >= 1)
ck("the recorded source path is scrubbed too",
   arts["job_100.out"]["path"].startswith("<REPO>/logs/"))
whole_case = "".join(
    _read(os.path.join(dp, fn))
    for dp, _dn, fns in os.walk(CASE_DIR) for fn in fns)
ck("nothing anywhere in the written case carries a scrub target",
   "byler" not in whole_case and "hf_A" not in whole_case
   and "hunter2super" not in whole_case)


# ---- refusal path ---------------------------------------------------------
dirty = by_id[CASE_DIRTY]
ck("a case with surviving residue is refused", dirty["status"] == "refused")
ck("refusal names the rule that fired", "scrub residue" in (dirty["reason"] or "")
   and "username" in (dirty["reason"] or ""))
ck("refusal message does not quote the secret", "byler2" not in (dirty["reason"] or ""))
ck("a refused case is not written",
   not os.path.exists(os.path.join(OUT, "cases", CASE_DIRTY)))


# ---- bundle shape ---------------------------------------------------------
ck("bundle carries every required key",
   set(["bundle_id", "sha256", "domain", "round", "step", "built_ts", "sections",
        "token_estimate", "caps"]).issubset(bundle.keys()))
ck("bundle_id is the case id", bundle["bundle_id"] == CASE_TIMEOUT)
ck("built_ts is the event date, not the export time", bundle["built_ts"] == "2026-08-29")
ck("bundle has all 14 sections as keys",
   sorted(bundle["sections"].keys()) == sorted(corpus.SECTIONS))
ck("out_tail carries artifact_id, path, sha256 and numbered lines",
   bundle["sections"]["out_tail"]["artifact_id"] == "job_100.out"
   and bundle["sections"]["out_tail"]["sha256"] == _sha_file(JOB_LOG)
   and bundle["sections"]["out_tail"]["lines"][0][0] == 1)
ck("out_tail lines are [absolute_line_number, text] pairs",
   all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], int)
       for p in bundle["sections"]["out_tail"]["lines"]))
ck("sacct section parsed into rows",
   isinstance(bundle["sections"]["sacct"], list)
   and bundle["sections"]["sacct"][0]["State"] == "TIMEOUT")
ck("results_csv section summarised",
   bundle["sections"]["results_csv"]["rows"] == 3
   and bundle["sections"]["results_csv"]["best"] == 0.44
   and bundle["sections"]["results_csv"]["last"] == 0.41
   and bundle["sections"]["results_csv"]["epoch_time_s"] == 1700.0)
ck("trace section parsed from jsonl",
   [r["epoch"] for r in bundle["sections"]["trace"]] == [1, 2])
ck("strategy section parsed from json",
   bundle["sections"]["strategy"]["epochs_requested"] == 60)
ck("strategy section was scrubbed before parsing",
   bundle["sections"]["strategy"]["save_dir"].startswith("<REPO>/"))
ck("inline inventory section is used", bundle["sections"]["su"]["round"] == 24.0)
missing = bundle["export"]["missing"]
ck("a section with nothing behind it is null",
   bundle["sections"]["ledger"] is None and bundle["sections"]["harvest"] is None)
ck("every null section carries a reason",
   all(missing.get(k) for k, v in bundle["sections"].items() if v is None))
ck("a filled section carries no missing reason", "out_tail" not in missing)
ck("section_source names the artifact a section came from",
   bundle["export"]["section_source"]["out_tail"] == "job_100.out"
   and bundle["export"]["section_source"]["su"] == "inventory:sections")
ck("token_estimate follows the stated formula",
   bundle["token_estimate"]
   == (len(corpus.canonical_json(bundle["sections"])) + 3) // 4)
ck("bundle sha256 recomputes over the bundle with sha256 blanked",
   bundle["sha256"] == corpus.sha256_str(
       corpus.canonical_json(dict(bundle, sha256=""))))
ck("caps are recorded in the bundle", bundle["caps"] == dict(corpus.CAPS))


# ---- truth.json -----------------------------------------------------------
ck("truth carries every required field",
   set(["case_id", "date", "incident", "class", "signals_expected",
        "load_bearing_lines", "acceptable_corrections", "escalation_expected",
        "provenance", "labels", "notes"]).issubset(truth.keys()))
ck("truth provenance is raw when artifacts survived", truth["provenance"] == "raw")
ck("truth keeps the pre-registered label",
   truth["labels"]["pre_registered"]["detect"] is True)
ck("truth starts with an empty adjudicated label", truth["labels"]["adjudicated"] == {})
ck("truth keeps the expected signals",
   truth["signals_expected"] == ["walltime_bound", "pool_growth"])
healthy = json.loads(_read(os.path.join(OUT, "cases", CASE_HEALTHY, "truth.json")))
ck("a healthy control is exported with incident false", healthy["incident"] is False)


# ---- over-1MB excerpt -----------------------------------------------------
big_dir = os.path.join(OUT, "cases", CASE_BIG)
big_stored = os.path.join(big_dir, "artifacts", "big_200.out.txt")
big_bundle = json.loads(_read(os.path.join(big_dir, "bundle.json")))
big_art = big_bundle["export"]["artifacts"][0]
big_text = _read(big_stored)
big_lines = big_text.split("\n")
if big_lines and big_lines[-1] == "":
    big_lines.pop()
ck("the fixture really is over the 1 MB threshold",
   os.path.getsize(BIG) > corpus.MAX_ARTIFACT_BYTES)
ck("excerpt header states the original sha256 and byte count",
   big_lines[0].startswith("[excerpt]")
   and _sha_file(BIG) in big_lines[0] and str(os.path.getsize(BIG)) in big_lines[0])
ck("excerpt states its omitted line ranges in the file",
   any(ln.startswith("[omitted lines 2001-") for ln in big_lines))
ck("excerpt keeps the head with absolute numbers",
   big_lines[1] == "000001\t" + BIG_LINES[0])
ck("excerpt keeps the tail with absolute numbers",
   big_lines[-1] == "040000\t" + BIG_LINES[-1])
ck("excerpt keeps the load-bearing line",
   ("%06d\t%s" % (BIG_CITE, BIG_LINES[BIG_CITE - 1])) in big_lines)
ck("excerpt reports no unresolved citation",
   by_id[CASE_BIG]["unresolved_load_bearing"] == [])
ck("excerpt is smaller than the original",
   len(big_text.encode("utf-8")) < os.path.getsize(BIG))
ck("manifest records the full original sha256 and bytes",
   big_art["sha256"] == _sha_file(BIG)
   and big_art["bytes"] == os.path.getsize(BIG)
   and big_art["lines"] == len(BIG_LINES))
ck("manifest records the excerpt and what it omitted",
   big_art["excerpt"]["omitted_lines"] > 0
   and big_art["excerpt"]["kept_lines"] + big_art["excerpt"]["omitted_lines"]
   == len(BIG_LINES)
   and "exceeds" in big_art["excerpt"]["reason"])
ck("nothing is truncated without saying so",
   sum(1 for ln in big_lines if ln.startswith("[omitted lines ")) == 2)


# ---- read ceiling (head+tail read, middle never opened) -------------------
# The same excerpt machinery on a file too large to read whole. Exercised by
# lowering the ceilings rather than writing a 64 MB fixture; the arithmetic
# under test is the absolute numbering of a tail read by seek.
OUT_PARTIAL = os.path.join(_tmp, "bench_partial")
_ceilings = (corpus.MAX_READ_BYTES, corpus.EXCERPT_HEAD_BYTES, corpus.EXCERPT_TAIL_BYTES)
corpus.MAX_READ_BYTES = 200000
corpus.EXCERPT_HEAD_BYTES = 50000
corpus.EXCERPT_TAIL_BYTES = 50000
try:
    prep = corpus.export(SPEC_PATH, out_dir=OUT_PARTIAL, root=ROOT, only=[CASE_BIG])
finally:
    (corpus.MAX_READ_BYTES, corpus.EXCERPT_HEAD_BYTES,
     corpus.EXCERPT_TAIL_BYTES) = _ceilings

pb = json.loads(_read(os.path.join(OUT_PARTIAL, "cases", CASE_BIG, "bundle.json")))
pa = pb["export"]["artifacts"][0]
pl = _read(os.path.join(OUT_PARTIAL, "cases", CASE_BIG, "artifacts",
                        "big_200.out.txt")).split("\n")
if pl and pl[-1] == "":
    pl.pop()
ck("a file over the read ceiling is marked partially read", pa["read"] == "partial")
ck("a partial read still records the full original sha256, bytes and line count",
   pa["sha256"] == _sha_file(BIG) and pa["bytes"] == os.path.getsize(BIG)
   and pa["lines"] == len(BIG_LINES))
ck("a partial read says the middle was never read",
   "read ceiling" in pa["excerpt"]["reason"])
ck("partial head keeps absolute numbering from line 1",
   pl[1] == "000001\t" + BIG_LINES[0])
ck("partial tail keeps absolute numbering to the last line",
   pl[-1] == "040000\t" + BIG_LINES[-1])
_tail = [ln for ln in pl if ln.startswith("039")]
ck("every partial tail line's number matches its own text",
   bool(_tail) and all(ln.split("\t", 1)[1] == BIG_LINES[int(ln[:6]) - 1]
                       for ln in _tail))
ck("the omitted middle is stated, not dropped in silence",
   any(ln.startswith("[omitted lines ") for ln in pl))
ck("a citation inside the unread middle is reported, not faked",
   prep["cases"][0]["unresolved_load_bearing"]
   and prep["cases"][0]["unresolved_load_bearing"][0]["line"] == BIG_CITE)


# ---- record-only ----------------------------------------------------------
rec_dir = os.path.join(OUT, "cases", CASE_RECORD)
rec_bundle = json.loads(_read(os.path.join(rec_dir, "bundle.json")))
rec_truth = json.loads(_read(os.path.join(rec_dir, "truth.json")))
ck("an incident with no surviving artifacts is still exported",
   os.path.isdir(rec_dir))
ck("record-only provenance is recorded in truth",
   rec_truth["provenance"] == "record-only")
ck("record-only artifacts directory exists and is empty",
   os.listdir(os.path.join(rec_dir, "artifacts")) == [])
_rec_why = rec_bundle["export"]["provenance_reason"] or ""
ck("record-only states why, naming the artifact and the failure",
   "job_44322382.out" in _rec_why and "could be read" in _rec_why
   and "FileNotFoundError" in _rec_why)
ck("the record-only reason is scrubbed too (an OSError carries an absolute path)",
   ROOT not in _rec_why and "<REPO>/logs/gone_44322382.out" in _rec_why)
ck("record-only sections are all null",
   all(v is None for v in rec_bundle["sections"].values()))
ck("the missing artifact's section says which artifact and why",
   "job_44322382.out" in rec_bundle["export"]["missing"]["out_tail"])
ck("record-only nonetheless keeps its label and notes",
   rec_truth["signals_expected"] == ["job_unknown"]
   and "did not survive" in rec_truth["notes"])
ck("the missing artifact is listed as not present",
   rec_bundle["export"]["artifacts"][0]["present"] is False)


# ---- determinism ----------------------------------------------------------
rep2 = corpus.export(SPEC_PATH, out_dir=OUT2, root=ROOT)
ck("second export reports the same counts", rep2["counts"] == rep["counts"])


def _tree(base):
    out = {}
    for dp, _dn, fns in os.walk(base):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, base)] = f.read()
    return out


t1, t2 = _tree(OUT), _tree(OUT2)
ck("both exports wrote the same file set", sorted(t1) == sorted(t2))
ck("both exports are byte-identical", t1 == t2)

# Re-exporting over an existing case directory must also be stable.
rep3 = corpus.export(SPEC_PATH, out_dir=OUT2, root=ROOT)
ck("re-export in place stays byte-identical", _tree(OUT2) == t1)
ck("re-export leaves no stale files", sorted(_tree(OUT2)) == sorted(t1))


# ---- --only / --dry-run ---------------------------------------------------
OUT3 = os.path.join(_tmp, "bench3")
dry = corpus.export(SPEC_PATH, out_dir=OUT3, root=ROOT, only=[CASE_TIMEOUT],
                    dry_run=True)
ck("--only selects one case", len(dry["cases"]) == 1
   and dry["cases"][0]["case_id"] == CASE_TIMEOUT)
ck("--only skips the rest", dry["counts"]["skipped"] == 4)
ck("--dry-run reports what it would write",
   dry["cases"][0]["status"] == "would-write" and dry["cases"][0]["artifacts"] == 5)
ck("--dry-run writes nothing", not os.path.exists(OUT3))
bad = corpus.export(SPEC_PATH, out_dir=OUT3, root=ROOT, only=["no_such_case"])
ck("--only with an unknown case id is an error, not a crash",
   bad["ok"] is False and "no such case" in " ".join(bad["errors"]))


# ---- list -----------------------------------------------------------------
rows = {r["case_id"]: r for r in corpus.list_cases(OUT)}
ck("list returns one row per exported case", len(rows) == 4)
ck("list reports provenance, class and artifact count",
   rows[CASE_TIMEOUT]["provenance"] == "raw"
   and rows[CASE_TIMEOUT]["class"] == "operational"
   and rows[CASE_TIMEOUT]["artifacts"] == 5
   and rows[CASE_TIMEOUT]["bytes_stored"] > 0)
ck("list reports a record-only case with zero artifacts",
   rows[CASE_RECORD]["provenance"] == "record-only"
   and rows[CASE_RECORD]["artifacts"] == 0)


# ---- verify ---------------------------------------------------------------
v = corpus.verify(OUT, root=ROOT)
ck("verify passes on a fresh export", v["ok"] is True)
ck("verify checked every stored artifact", v["checked"] == 7)

with open(stored_log, "a", encoding="utf-8") as f:
    f.write("000009\tan extra line nobody wrote\n")
v2 = corpus.verify(OUT, case=CASE_TIMEOUT)
ck("verify catches drift in a stored copy",
   v2["ok"] is False and any("drifted" in p for p in v2["cases"][0]["problems"]))
shutil.copyfile(os.path.join(OUT2, "cases", CASE_TIMEOUT, "artifacts",
                             "job_100.out.txt"), stored_log)
ck("verify passes again once the copy is restored",
   corpus.verify(OUT, case=CASE_TIMEOUT)["ok"] is True)

with open(JOB_LOG, "a", encoding="utf-8") as f:
    f.write("a line appended after the export\n")
v3 = corpus.verify(OUT, case=CASE_TIMEOUT, root=ROOT)
ck("verify catches drift in the ORIGINAL when given a root",
   v3["ok"] is False and any("ORIGINAL has changed" in p
                             for p in v3["cases"][0]["problems"]))
ck("verify without a root does not read the originals",
   corpus.verify(OUT, case=CASE_TIMEOUT)["ok"] is True)


# ---- freeze ---------------------------------------------------------------
fz = corpus.freeze(OUT, dry_run=True)
ck("freeze --dry-run computes a split without writing",
   fz["ok"] is True and not os.path.exists(os.path.join(OUT, "split.json")))
ck("freeze warns when rubric.md is missing",
   any("rubric.md" in w for w in fz["warnings"]))

fz = corpus.freeze(OUT)
split = json.loads(_read(os.path.join(OUT, "split.json")))
ck("freeze wrote split.json", fz["ok"] is True)
ck("split has exactly the four registered keys",
   sorted(split.keys()) == ["dev", "rule", "sha256", "test"])
ck("dev holds the pre-cutoff cases and the worked example",
   split["dev"] == sorted([CASE_HEALTHY, CASE_RECORD, CASE_TIMEOUT]))
ck("test holds the later case", split["test"] == [CASE_BIG])
ck("the rule text states the cutoff and the example date",
   "2026-08-25" in split["rule"] and "2026-08-29" in split["rule"])
ck("split sha256 is over the frozen case set",
   split["sha256"] == corpus._corpus_digest(OUT, split["dev"] + split["test"]))

before = _read(os.path.join(OUT, "split.json"))
fz2 = corpus.freeze(OUT)
ck("freeze refuses to overwrite an existing split",
   fz2["ok"] is False and "already exists" in " ".join(fz2["errors"]))
ck("the refused freeze left split.json untouched",
   _read(os.path.join(OUT, "split.json")) == before)
ck("verify re-checks the frozen split hash",
   corpus.verify(OUT)["split_sha256_recomputed"] == split["sha256"])


# ---- an occupied directory is left alone ----------------------------------
OUT_OCCUPIED = os.path.join(_tmp, "bench_occupied")
os.makedirs(os.path.join(OUT_OCCUPIED, "cases", CASE_TIMEOUT))
_w2 = os.path.join(OUT_OCCUPIED, "cases", CASE_TIMEOUT, "somebody_elses_file.txt")
with open(_w2, "w", encoding="utf-8") as f:
    f.write("not mine\n")
occ = corpus.export(SPEC_PATH, out_dir=OUT_OCCUPIED, root=ROOT, only=[CASE_TIMEOUT])
ck("a directory this tool did not write is refused, not deleted",
   occ["cases"][0]["status"] == "refused" and os.path.exists(_w2)
   and not os.path.exists(os.path.join(OUT_OCCUPIED, "cases", CASE_TIMEOUT,
                                       "bundle.json")))


# ---- no entry point raises ------------------------------------------------
ck("export of a missing inventory reports instead of raising",
   corpus.export(os.path.join(_tmp, "nope.json"), out_dir=OUT3)["ok"] is False)
ck("export of a non-inventory JSON reports instead of raising",
   corpus.export(JOB_LOG, out_dir=OUT3)["ok"] is False)
ck("verify of an empty directory reports instead of raising",
   corpus.verify(os.path.join(_tmp, "empty"))["ok"] is False)
ck("list of an empty directory is empty", corpus.list_cases(os.path.join(_tmp, "empty")) == [])
ck("freeze with no cases reports instead of raising",
   corpus.freeze(os.path.join(_tmp, "empty"))["ok"] is False)

OUT4 = os.path.join(_tmp, "bench4")
ck("main export returns 0 on a clean subset",
   corpus.main(["--out", OUT4, "export", "--spec", SPEC_PATH, "--root", ROOT,
                "--only", CASE_TIMEOUT]) == 0)
ck("main export returns 1 when a case is refused",
   corpus.main(["--out", OUT4, "export", "--spec", SPEC_PATH, "--root", ROOT]) == 1)
ck("main verify returns 0 on a clean corpus",
   corpus.main(["--out", OUT4, "verify"]) == 0)
ck("main list returns 0 with cases", corpus.main(["--out", OUT4, "list"]) == 0)
ck("main freeze returns 0 then 1", corpus.main(["--out", OUT4, "freeze"]) == 0
   and corpus.main(["--out", OUT4, "freeze"]) == 1)
ck("main with a missing spec returns 1",
   corpus.main(["export", "--spec", os.path.join(_tmp, "nope.json"),
                "--out", OUT4]) == 1)
ck("main with no command returns 2", corpus.main([]) == 2)


if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)

print("\n-- a stored copy is hashed as bytes, not as decoded text --")
# A job log carries carriage returns from progress bars. Text-mode reading
# normalises them, so hashing the decoded string reported 107 of 390 intact
# artifacts as drifted on the first real export. The check has to survive a log
# that is not newline-clean, or it is a check nobody will believe.
_cr = pathlib.Path(_tmp) / "cr_case"
(_cr / "artifacts").mkdir(parents=True, exist_ok=True)
_raw = b"     1\tepoch 1/60\r     2\tepoch 2/60\r\n     3\tdone\n"
_p = _cr / "artifacts" / "progress.out.txt"
_p.write_bytes(_raw)
ck("the fixture really is not newline-clean",
   _raw != _p.read_text(encoding="utf-8").encode("utf-8"))
ck("bytes hashing is stable across a read-back",
   corpus.sha256_bytes(_p.read_bytes()) == corpus.sha256_bytes(_raw))
ck("text hashing is not, which is why it is no longer used for stored copies",
   corpus.sha256_str(_p.read_text(encoding="utf-8")) != corpus.sha256_bytes(_raw))

print("\nALL PASS")
