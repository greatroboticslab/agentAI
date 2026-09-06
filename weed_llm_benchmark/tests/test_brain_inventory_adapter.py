#!/usr/bin/env python3
"""Unit tests for the inventory -> corpus-case adapter (no cluster / network).

The adapter exists because a dry run of the corpus export over the 162-case
incident inventory refused 162 of 162: the inventory records `correction_class`
(empty for a control) and a flat `artifact_paths_expected` list, the exporter
wants `class`, `escalation_expected` and `artifacts: [{name, path, section}]`.
These tests pin the translation on synthetic entries in a temporary directory:

  * `class` from `correction_class`, and `none` when it is empty
  * `escalation_expected` for every branch of the derivation, with the rule
    written into the notes so a reviewer scores against a stated rule
  * command vs glob vs path classification, including the strings that look like
    commands but are not
  * glob expansion, its newest-first order and its cap, stated in the notes
  * section mapping, including the names that must stay unmapped rather than be
    forced into a wrong section
  * artifact-name collisions disambiguated with a numeric suffix
  * every inventory field the exporter does not consume carried into the notes
  * two runs producing byte-identical output
  * a failing command stored as an artifact rather than raised
  * and last, that the REAL inventory's 162 cases all pass
    `corpus._validate_case` — the check that the mismatch is closed

Run:  python3 tests/test_brain_inventory_adapter.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import corpus  # noqa: E402
from weed_optimizer_framework.tools.brain import inventory_adapter as ia  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


_tmp = tempfile.mkdtemp(prefix="inv_adapter_")
ROOT = os.path.join(_tmp, "repo")
os.makedirs(os.path.join(ROOT, "results", "framework"), exist_ok=True)


def touch(rel, text="hello\nworld\n", mtime=None):
    p = pathlib.Path(ROOT) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "w", encoding="utf-8") as f:
        f.write(text)
    if mtime is not None:
        os.utime(str(p), (mtime, mtime))
    return str(p)


def entry(**kw):
    """An inventory entry with the inventory's own default field set."""
    base = {
        "case_id": "case-1", "date": "2026-08-29", "title": "t",
        "one_line_symptom": "s", "root_cause": "r", "correction_made": "c",
        "correction_class": "config", "actor_who_corrected": "operator",
        "version_tag": "v3.0", "job_ids": [], "artifact_paths_expected": [],
        "detectable_from": [], "signals_expected": [], "healthy_control": False,
        "incident": True, "category": "incident", "scope": "cluster-pipeline",
        "notes": "",
    }
    base.update(kw)
    return base


def adapt_one(e, root=ROOT, **kw):
    case, stats = ia.adapt_case(e, root, **kw)
    return case, stats


# ---- class derivation ----------------------------------------------------
print("class derivation")
for cc in ("code", "config", "design", "plan"):
    ck("correction_class %s -> class %s" % (cc, cc),
       ia.derive_class(entry(correction_class=cc)) == cc)
ck("empty correction_class -> class none",
   ia.derive_class(entry(correction_class="", incident=False)) == "none")
ck("missing correction_class -> class none", ia.derive_class({}) == "none")
ck("whitespace correction_class -> class none",
   ia.derive_class({"correction_class": "   "}) == "none")
ck("every derived class is in corpus.CASE_CLASSES",
   all(ia.derive_class(entry(correction_class=c)) in corpus.CASE_CLASSES
       for c in ("code", "config", "design", "plan", "")))


# ---- escalation derivation -----------------------------------------------
print("escalation derivation")
ck("healthy control -> none",
   ia.derive_escalation(entry(healthy_control=True, correction_class="config"))[0]
   == "none")
ck("incident false -> none",
   ia.derive_escalation(entry(incident=False, correction_class=""))[0] == "none")
ck("code -> human", ia.derive_escalation(entry(correction_class="code"))[0] == "human")
ck("design -> human",
   ia.derive_escalation(entry(correction_class="design"))[0] == "human")
ck("plan -> human", ia.derive_escalation(entry(correction_class="plan"))[0] == "human")
ck("config -> tier1",
   ia.derive_escalation(entry(correction_class="config"))[0] == "tier1")
ck("operational (outside the four edit kinds) -> human, conservatively",
   ia.derive_escalation(entry(correction_class="operational"))[0] == "human")
ck("every derived escalation is in corpus.ESCALATIONS",
   all(ia.derive_escalation(entry(correction_class=c))[0] in corpus.ESCALATIONS
       for c in ("code", "config", "design", "plan", "operational", "")))

_c, _ = adapt_one(entry(correction_class="config"))
ck("escalation lands on the case", _c["escalation_expected"] == "tier1")
ck("escalation rule is recorded in the notes",
   "escalation_expected: tier1 (rule: correction_class=config" in _c["notes"])
ck("class derivation is recorded in the notes",
   _c["notes"].startswith("class: config (from correction_class)"))
_c, _ = adapt_one(entry(correction_class="", incident=False, healthy_control=True))
ck("control notes say the empty class became none",
   "class: none (from correction_class, empty = none)" in _c["notes"])
ck("pre_registered label carries detect + correction_class",
   _c["labels"]["pre_registered"] == {"detect": False, "correction_class": "none"})


# ---- command vs glob vs path ---------------------------------------------
print("entry classification")
ck("sacct is a command", ia.classify_entry("sacct -j 44727703")[0] == "command")
ck("squeue is a command", ia.classify_entry("squeue -u x")[0] == "command")
ck("scontrol is a command", ia.classify_entry("scontrol show job 1")[0] == "command")
ck("a path that merely starts with the word sacct is a path",
   ia.classify_entry("results/framework/sacct_44727703.tsv")[0] == "path")
ck("a journalctl line is a path, not a command (only the three SLURM "
   "accounting programs are commands)",
   ia.classify_entry("lab journalctl --user -u weed-dashboard")[0] == "path")
ck("a star makes a glob", ia.classify_entry("results/framework/*_1.out")[0] == "glob")
ck("a question mark makes a glob", ia.classify_entry("a/b?.out")[0] == "glob")
ck("a bracket makes a glob", ia.classify_entry("a/b[0-9].out")[0] == "glob")
ck("a placeholder in angle brackets is a plain path",
   ia.classify_entry("results/framework/m1_s1_<jobid>.out")[0] == "path")
ck("an empty entry classifies as an empty path",
   ia.classify_entry("   ") == ("path", ""))
ck("a non-string entry classifies as an empty path",
   ia.classify_entry(None) == ("path", ""))

touch("results/framework/real_1.out")
_c, _s = adapt_one(entry(artifact_paths_expected=[
    "results/framework/real_1.out",          # exists
    "results/framework/absent_2.out",        # predicted, not here
    "sacct -j 44727703",                     # command, not collected
]))
ck("existing path counted present", _s["present"] == 1)
ck("absent path counted missing", _s["missing"] == 1)
ck("command skipped without --collect-commands", _s["commands_skipped"] == 1)
ck("skipped command is recorded in the notes",
   "command_not_collected: sacct -j 44727703" in _c["notes"])
ck("skipped command produces no artifact",
   [a["name"] for a in _c["artifacts"]] == ["real_1.out", "absent_2.out"])
ck("an absent path still becomes an artifact, so the exporter can say it is "
   "missing", any(a["path"] == "results/framework/absent_2.out"
                  for a in _c["artifacts"]))


# ---- glob expansion and its cap ------------------------------------------
print("glob expansion")
for i in range(10):
    touch("results/framework/glob/g_%02d.out" % i, mtime=1_700_000_000 + i)
matches, truncated = ia.expand_glob("results/framework/glob/g_*.out", ROOT)
ck("glob capped at GLOB_CAP", len(matches) == ia.GLOB_CAP == 8)
ck("glob reports how many it dropped", truncated == 2)
ck("glob order is newest first",
   [os.path.basename(m) for m in matches[:3]] == ["g_09.out", "g_08.out", "g_07.out"])
ck("glob returns absolute paths", all(os.path.isabs(m) for m in matches))

os.makedirs(os.path.join(ROOT, "results", "framework", "glob", "d_dir.out"),
            exist_ok=True)
m2, _ = ia.expand_glob("results/framework/glob/*.out", ROOT)
ck("a directory never becomes an artifact", all(os.path.isfile(m) for m in m2))

_c, _s = adapt_one(entry(artifact_paths_expected=["results/framework/glob/g_*.out"]))
ck("capped glob yields 8 artifacts", len(_c["artifacts"]) == 8)
ck("the cap is stated in the notes",
   "glob_truncated: results/framework/glob/g_*.out kept 8 newest of 10"
   in _c["notes"])
ck("glob counted as expanded", _s["globs_expanded"] == 1 and _s["globs_capped"] == 1)

_c, _s = adapt_one(entry(artifact_paths_expected=["results/framework/none_*.out"]))
ck("an unmatched glob still yields one artifact", len(_c["artifacts"]) == 1)
ck("an unmatched glob keeps the pattern as the path",
   _c["artifacts"][0]["path"] == "results/framework/none_*.out")
ck("an unmatched glob has a safe name (no star)",
   "*" not in _c["artifacts"][0]["name"])
ck("an unmatched glob is noted", "glob_unmatched: results/framework/none_*.out"
   in _c["notes"])
ck("an unmatched glob counts as a missing path",
   _s["globs_unmatched"] == 1 and _s["missing"] == 1)


# ---- section mapping -----------------------------------------------------
print("section mapping")
ck("*.out -> out_tail", ia.section_for("m1_merged_rndtrain_s1_44727703.out")
   == "out_tail")
ck("*.jsonl -> trace", ia.section_for("cluster_actions.jsonl") == "trace")
ck("results.csv -> results_csv", ia.section_for("results.csv") == "results_csv")
ck("*_results.csv -> results_csv",
   ia.section_for("v3_0_23_mega_iter6_train8_results.csv") == "results_csv")
ck("sacct_<jobid>.txt -> sacct", ia.section_for("sacct_44727703.txt") == "sacct")
ck("slug_scores.json -> slug_scores",
   ia.section_for("slug_scores.json") == "slug_scores")
ck("plan.json -> plan", ia.section_for("plan.json") == "plan")
ck("su.json -> su", ia.section_for("su.json") == "su")
ck("summary.json is NOT the su section", ia.section_for("summary.json") is None)
ck("plain_s102.json is NOT the plan section",
   ia.section_for("plain_s102.json") is None)
ck("a registry snapshot is not a registry diff",
   ia.section_for("dataset_registry.json") is None)
ck("an unmatched name maps to None, never to a wrong section",
   ia.section_for("figures_data.json") is None)
ck("every mapped section is one of the exporter's 14",
   all(sec in corpus.SECTIONS for _, sec in ia.SECTION_RULES))
ck("the table covers all 14 sections",
   {sec for _, sec in ia.SECTION_RULES} == set(corpus.SECTIONS))

touch("results/framework/sec_1.out")
touch("results/framework/pool_report.json", text="{}\n")
_c, _s = adapt_one(entry(artifact_paths_expected=[
    "results/framework/sec_1.out", "results/framework/pool_report.json"]))
_by = {a["name"]: a for a in _c["artifacts"]}
ck("mapped artifact carries its section", _by["sec_1.out"]["section"] == "out_tail")
ck("unmapped artifact carries no section key",
   "section" not in _by["pool_report.json"])
ck("section histogram counts the unmapped one",
   _s["sections"] == {"out_tail": 1, "unmapped": 1})


# ---- names ---------------------------------------------------------------
print("artifact names")
ck("basename keeps the original file visible",
   ia.safe_basename("results/framework/mega_iterX/train/results.csv") == "results.csv")
ck("glob characters are replaced, not kept",
   ia.safe_basename("results/framework/*_44275211.out") == "__44275211.out")
ck("a leading dot is stripped (the exporter refuses a dotfile name)",
   ia.safe_basename("~/.round_scheduler.json") == "round_scheduler.json")
ck("a trailing separator still names the last segment",
   ia.safe_basename("uploads/session/frames/") == "frames")
ck("an unnameable path falls back to a constant",
   ia.safe_basename("///") == "artifact")
ck("spaces become underscores",
   ia.safe_basename("lab journalctl --user -u weed-dashboard")
   == "lab_journalctl_--user_-u_weed-dashboard")

for d in ("a", "b", "c"):
    touch("results/framework/mega_%s/train/results.csv" % d)
_c, _ = adapt_one(entry(artifact_paths_expected=[
    "results/framework/mega_a/train/results.csv",
    "results/framework/mega_b/train/results.csv",
    "results/framework/mega_c/train/results.csv"]))
_names = [a["name"] for a in _c["artifacts"]]
ck("colliding names get a numeric suffix before the extension",
   _names == ["results.csv", "results_2.csv", "results_3.csv"])
ck("names stay unique within the case", len(set(_names)) == len(_names))
ck("the full original path survives beside the name",
   [a["path"] for a in _c["artifacts"]][1].endswith("mega_b/train/results.csv"))
ck("every name is a safe file name",
   all(n and "/" not in n and not n.startswith(".") for n in _names))
ck("the exporter accepts the adapted case", corpus._validate_case(_c) == [])


# ---- notes carry the unconsumed fields -----------------------------------
print("notes")
_e = entry(case_id="notes-case", correction_class="code",
           job_ids=["44727703", "44767709"],
           detectable_from=["sacct State=TIMEOUT"],
           provenance_guess="raw", provenance_reason="September artifacts",
           notes="inventory prose", future_field="a field added later")
_c, _ = adapt_one(_e)
for label in ("title:", "symptom:", "root_cause:", "correction_made:",
              "correction_class: code", "actor_who_corrected: operator",
              "version_tag: v3.0", "category: incident", "scope: cluster-pipeline",
              "job_ids: 44727703; 44767709", "detectable_from: sacct State=TIMEOUT",
              "provenance_guess: raw", "provenance_reason: September artifacts",
              "inventory_notes: inventory prose"):
    ck("notes carry %r" % label.split(":")[0], label in _c["notes"])
ck("notes carry a field this module has never seen",
   "future_field: a field added later" in _c["notes"])
ck("healthy_control false is written, not dropped as falsy",
   "healthy_control: false" in _c["notes"])
ck("job_id is the first recorded job id", _c["job_id"] == "44727703")
ck("a case with no job ids carries no job_id key",
   "job_id" not in adapt_one(entry())[0])
ck("consumed fields are carried through unchanged",
   _c["case_id"] == "notes-case" and _c["date"] == "2026-08-29"
   and _c["incident"] is True)
_c2, _ = adapt_one(entry(signals_expected=["walltime_bound", "pool_growth"]))
ck("signals_expected carried through unchanged",
   _c2["signals_expected"] == ["walltime_bound", "pool_growth"])


# ---- commands ------------------------------------------------------------
print("commands")
ck("command name is the job it asks about",
   ia.command_artifact_name("sacct -j 44727703") == "sacct_44727703.txt")
ck("comma-joined job ids join with underscores",
   ia.command_artifact_name("sacct -j 44727703,44767709")
   == "sacct_44727703_44767709.txt")
ck("an array task keeps its suffix",
   ia.command_artifact_name("sacct -j 44234060_2") == "sacct_44234060_2.txt")
_qname = ia.command_artifact_name("sacct -S 2026-08-24 -u someone --name=brain")
ck("a query with no job id is named by a digest, carrying no free text",
   _qname.startswith("sacct_q") and _qname.endswith(".txt") and len(_qname) == 19)
ck("the command name maps to the sacct section",
   ia.section_for("sacct_44727703.txt") == "sacct")

_body, _ok = ia.run_command("%s -c \"print('one'); print('two')\"" % sys.executable)
ck("a successful command returns its stdout", _ok and "one\ntwo" in _body)
ck("a successful command records the command line", _body.startswith("$ "))
_body, _ok = ia.run_command("%s -c \"import sys; sys.exit(3)\"" % sys.executable)
ck("a non-zero exit is a failure body, not an exception",
   (not _ok) and _body.startswith("[command failed] exit=3"))
_body, _ok = ia.run_command("no_such_program_here --flag")
ck("a missing binary is a failure body, not an exception",
   (not _ok) and _body.startswith("[command failed]"))
_body, _ok = ia.run_command("%s -c \"import time; time.sleep(5)\"" % sys.executable,
                            timeout=1)
ck("a timeout is a failure body, not an exception",
   (not _ok) and "no output within 1 s" in _body)

COLLECT = os.path.join(_tmp, "collected")


def _stub_ok(cmd, timeout=ia.COMMAND_TIMEOUT_S):
    return "$ %s\nJobID State\n44727703 TIMEOUT\n" % cmd, True


def _stub_fail(cmd, timeout=ia.COMMAND_TIMEOUT_S):
    return "[command failed] exit=1\n$ %s\nsacct: not available here\n" % cmd, False


_c, _s = adapt_one(entry(case_id="collect-case",
                         artifact_paths_expected=["sacct -j 44727703"]),
                   collect_dir=COLLECT, runner=_stub_ok)
ck("a collected command becomes an artifact", len(_c["artifacts"]) == 1)
ck("the collected artifact is named after the command",
   _c["artifacts"][0]["name"] == "sacct_44727703.txt")
ck("the collected artifact maps to the sacct section",
   _c["artifacts"][0]["section"] == "sacct")
ck("the collected artifact points at a file that exists",
   os.path.isfile(_c["artifacts"][0]["path"]))
ck("the collected file holds the command output",
   open(_c["artifacts"][0]["path"]).read().endswith("44727703 TIMEOUT\n"))
ck("collection is counted", _s["commands_collected"] == 1)
ck("the collection is noted",
   "command_collected: sacct -j 44727703" in _c["notes"])

_c, _s = adapt_one(entry(case_id="collect-fail",
                         artifact_paths_expected=["sacct -j 44727703"]),
                   collect_dir=COLLECT, runner=_stub_fail)
ck("a failing command still becomes an artifact, so the absence is evidence",
   len(_c["artifacts"]) == 1 and os.path.isfile(_c["artifacts"][0]["path"]))
ck("the failure text is the artifact body",
   open(_c["artifacts"][0]["path"]).read().startswith("[command failed] exit=1"))
ck("the failure is noted as a failure",
   "command_collected (failed): sacct -j 44727703" in _c["notes"])
ck("a failing command is still counted as collected", _s["commands_collected"] == 1)


# ---- determinism ---------------------------------------------------------
print("determinism")
SPEC = os.path.join(_tmp, "spec.json")
with open(SPEC, "w", encoding="utf-8") as f:
    json.dump({"cases": [
        entry(case_id="det-1", artifact_paths_expected=[
            "results/framework/glob/g_*.out",
            "results/framework/real_1.out",
            "results/framework/absent_2.out",
            "sacct -j 44727703"]),
        entry(case_id="det-2", correction_class="", incident=False,
              healthy_control=True, category="healthy-control"),
    ]}, f)

OUT_A = os.path.join(_tmp, "adapted_a.json")
OUT_B = os.path.join(_tmp, "adapted_b.json")
r1 = ia.adapt_file(SPEC, OUT_A, root=ROOT)
r2 = ia.adapt_file(SPEC, OUT_B, root=ROOT)
ck("two runs are byte-identical",
   open(OUT_A, "rb").read() == open(OUT_B, "rb").read())
ck("adapt_file reports ok", r1.get("ok") is True)
ck("report counts the cases", r1["report"]["cases"] == 2)
ck("report counts artifacts present", r1["report"]["artifacts_present"] == 9)
ck("report counts missing paths",
   r1["report"]["paths_missing"] == 1 and r1["report"]["paths_not_a_file"] == 0)
ck("report counts skipped commands", r1["report"]["commands_skipped"] == 1)
ck("report carries a per-section histogram",
   r1["report"]["sections"].get("out_tail") == 10)
_spec_a = corpus.read_json(OUT_A)
ck("adapted spec keeps the case order",
   [c["case_id"] for c in _spec_a["cases"]] == ["det-1", "det-2"])
ck("adapted spec records the adapter version",
   _spec_a["_adapter"]["tool_version"] == ia.TOOL_VERSION)
ck("adapted cases all validate",
   all(corpus._validate_case(c) == [] for c in _spec_a["cases"]))

OUT_C = os.path.join(_tmp, "adapted_c.json")
_rc = ia.adapt_file(SPEC, OUT_C, root=ROOT, collect_commands=True, runner=_stub_ok)
ck("--collect-commands stores output beside the adapted spec",
   _rc["report"]["collect_dir"] == os.path.join(_tmp, "adapted_c_commands"))
ck("--collect-commands collects the one command", _rc["report"]["commands_collected"] == 1
   and _rc["report"]["commands_skipped"] == 0)
ck("the collected file is on disk under the case id",
   os.path.isfile(os.path.join(_tmp, "adapted_c_commands", "det-1",
                               "sacct_44727703.txt")))

with open(os.path.join(_tmp, "scrub_spec.json"), "w", encoding="utf-8") as f:
    json.dump({"scrub_users": ["someone"], "domain": "weed", "cases": []}, f)
_r = ia.adapt(corpus.read_json(os.path.join(_tmp, "scrub_spec.json")), root=ROOT)
ck("scrub configuration is carried through to the exporter",
   _r["spec"]["scrub_users"] == ["someone"] and _r["spec"]["domain"] == "weed")


# ---- entry points never raise --------------------------------------------
print("entry points")
ck("main adapts the synthetic spec and returns 0",
   ia.main(["--spec", SPEC, "--out", os.path.join(_tmp, "cli.json"),
            "--root", ROOT]) == 0)
ck("main --json returns 0",
   ia.main(["--spec", SPEC, "--out", os.path.join(_tmp, "cli2.json"),
            "--root", ROOT, "--json"]) == 0)
ck("main on a missing spec returns 1 without raising",
   ia.main(["--spec", os.path.join(_tmp, "nope.json"),
            "--out", os.path.join(_tmp, "cli3.json")]) == 1)
ck("main on an unwritable out returns 1 without raising",
   ia.main(["--spec", SPEC, "--root", ROOT,
            "--out", os.path.join(ROOT, "results", "framework", "real_1.out",
                                  "x.json")]) == 1)
ck("adapt of a non-object spec returns an empty case list",
   ia.adapt("not a spec", root=ROOT)["report"]["cases"] == 0)
ck("adapt of a bare list of cases works",
   ia.adapt([entry(case_id="bare")], root=ROOT)["report"]["cases"] == 1)
ck("a non-object case entry is skipped and reported",
   ia.adapt({"cases": ["nope"]}, root=ROOT)["report"]["skipped"] == 1)


# ---- the real inventory: the check that the mismatch is closed ------------
print("real inventory")
REAL = pathlib.Path(__file__).resolve().parents[1] / "results" / "framework" / \
    "supervision_bench" / "inventory.json"
_inv = corpus.read_json(REAL)
ck("the real inventory is readable", isinstance(_inv, dict))
if isinstance(_inv, dict):
    REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
    OUT_REAL = os.path.join(_tmp, "adapted_real.json")
    # No --collect-commands: sacct answers only on the cluster, and this check
    # must hold on a laptop.
    _rr = ia.adapt_file(str(REAL), OUT_REAL, root=REPO_ROOT)
    _cases = (corpus.read_json(OUT_REAL) or {}).get("cases") or []
    ck("all 162 inventory cases adapt", len(_cases) == 162)
    _bad = [(c.get("case_id"), corpus._validate_case(c)) for c in _cases
            if corpus._validate_case(c)]
    ck("corpus._validate_case reports no problems for any of the 162 cases "
       "(before this adapter it refused 162 of 162)", not _bad)
    if _bad:
        for cid, why in _bad[:5]:
            print("        %s: %s" % (cid, "; ".join(why)))
    ck("every adapted class is in corpus.CASE_CLASSES",
       all(c["class"] in corpus.CASE_CLASSES for c in _cases))
    ck("every adapted escalation is in corpus.ESCALATIONS",
       all(c["escalation_expected"] in corpus.ESCALATIONS for c in _cases))
    ck("controls escalate to none",
       all(c["escalation_expected"] == "none" for c in _cases if not c["incident"]))
    ck("the 37 sacct commands are recorded, not executed",
       _rr["report"]["commands_skipped"] == 37
       and _rr["report"]["commands_collected"] == 0)
    ck("no artifact name is unsafe",
       all(a["name"] and "/" not in a["name"] and not a["name"].startswith(".")
           for c in _cases for a in c["artifacts"]))
    ck("no artifact carries a section outside the exporter's 14",
       all(a.get("section") in corpus.SECTIONS
           for c in _cases for a in c["artifacts"] if "section" in a))
    print("       real inventory: %d cases, %d artifacts, %d resolved here, "
          "%d not on this machine"
          % (_rr["report"]["cases"], _rr["report"]["artifacts"],
             _rr["report"]["artifacts_present"], _rr["report"]["paths_missing"]))
    print("       sections: %s" % json.dumps(_rr["report"]["sections"],
                                             sort_keys=True))

shutil.rmtree(_tmp, ignore_errors=True)

if _fails:
    print(f"\nFAILED: {len(_fails)} -> {_fails}")
    sys.exit(1)
print("\nALL PASS")
