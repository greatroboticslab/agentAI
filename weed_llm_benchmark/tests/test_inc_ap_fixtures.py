#!/usr/bin/env python3
"""The INC autopilot's replay fixtures are exactly the pinned bytes.

tests/fixtures/inc_replay/ holds frozen copies of the INC artifacts the
replay tests read (docs/INC_AUTOPILOT.md, (f) and (g) step 1): pilot_v1's,
pilot_v2's and pilot_v3's exp.json, report.json, ledger.jsonl and build_summary.json,
b0_v1's exp.json and ledger.jsonl, and base_b_v1's exp.json, report.json
and build_summary.json. MANIFEST.json pins each file's sha256 and size.
A replay case that ran on edited evidence would prove nothing, so:
  * every pinned file exists with its sha256 and size;
  * every file in the fixture tree is pinned (nothing unpinned slips in,
    including a pulled Step 1 or pilot_v2 file that was not added to the
    manifest);
  * synthetic files (MANIFEST.json "synthetic": replay case R7's Step 1
    with a relevance.json whose calibration check failed, and R8's
    relevance.json made for the real select build) are pinned the same
    way, name the Step 1 file each stands for, carry a recorded why, and
    live outside step1/ and every experiment directory, so no loader reads
    them in place of a real file;
  * Step 1 copies (MANIFEST.json "step1_copies": replay case R8's byte
    copies of the local results/framework/inc/step1/select_summary.json and
    admit_summary.json) are pinned the same way, name the results file each
    was copied from and the Step 1 file each stands for, carry a recorded
    why, and live in step1_copies/, outside step1/ (whose cluster pull is
    still pending) and every experiment directory;
  * a pending entry (Step 1: cluster only) is not a pinned file yet, and
    names where it comes from; pilot_v2 is pinned, no longer pending;
  * the evidence loader reads exactly the pinned files of an experiment
    (and nothing else in the tree).
When the source results file is still present locally and differs from the
fixture, that is reported (the results copy moved on), not failed: the
fixture is the pinned record.

Run:  python3 tests/test_inc_ap_fixtures.py
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "inc_replay"
REPO = pathlib.Path(__file__).resolve().parents[1]
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    print("manifest")
    man = json.loads((FIX / "MANIFEST.json").read_text())
    check("the manifest has its format", man.get("format") == "inc-autopilot/replay-fixtures/1", man.get("format"))
    files = man.get("files") or {}
    check("pilot_v1 and b0_v1 are pinned",
          {"pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/ledger.jsonl", "pilot_v1/build_summary.json",
           "b0_v1/exp.json", "b0_v1/ledger.jsonl"} <= set(files), sorted(files))
    check("pilot_v2 (the full rehearsal, gate v1) and base_b_v1 are pinned (replay case R5, the e2e test)",
          {"pilot_v2/exp.json", "pilot_v2/report.json", "pilot_v2/ledger.jsonl", "pilot_v2/build_summary.json",
           "base_b_v1/exp.json", "base_b_v1/report.json", "base_b_v1/build_summary.json"} <= set(files),
          sorted(files))
    check("pilot_v3 (the full rehearsal decided by gate v2) is pinned (replay case R6)",
          {"pilot_v3/exp.json", "pilot_v3/report.json", "pilot_v3/ledger.jsonl", "pilot_v3/build_summary.json"}
          <= set(files) and any(set(a.get("files") or []) >= {"pilot_v3/report.json"} for a in man.get("added") or []),
          sorted(files))
    check("each pinned file names the results file it was copied from",
          all(rec.get("from") == "results/framework/inc/" + name for name, rec in files.items()),
          [n for n, r in files.items() if r.get("from") != "results/framework/inc/" + n])

    print("pinned bytes")
    for name, rec in sorted(files.items()):
        p = FIX / name
        check("%s exists" % name, p.is_file())
        if not p.is_file():
            continue
        check("%s sha256 %s" % (name, rec["sha256"][:12]), sha(p) == rec["sha256"], sha(p))
        check("%s is %d bytes" % (name, rec["bytes"]), p.stat().st_size == rec["bytes"], p.stat().st_size)
        src = REPO / rec["from"]
        if src.is_file() and sha(src) != rec["sha256"]:
            print("  note %s: the results copy %s has moved on from the pinned fixture" % (name, rec["from"]))

    print("synthetic files (replay cases R7 and R8)")
    syn = man.get("synthetic") or {}
    r7 = {"synthetic/step1_calibration_failed/%s" % f for f in
          ("select_summary.json", "admit_summary.json", "select_clusters_by_source.json", "relevance.json")}
    check("R7's synthetic Step 1 is pinned", r7 <= set(syn), sorted(syn))
    check("R8's synthetic relevance.json is pinned",
          "synthetic/step1_real_calibration_failed/relevance.json" in syn, sorted(syn))
    for name, rec in sorted(syn.items()):
        p = FIX / name
        check("%s exists" % name, p.is_file())
        if not p.is_file():
            continue
        check("%s sha256 %s" % (name, rec["sha256"][:12]), sha(p) == rec["sha256"], sha(p))
        check("%s is %d bytes" % (name, rec["bytes"]), p.stat().st_size == rec["bytes"], p.stat().st_size)
        check("%s stands for step1/%s and is no results copy" % (name, p.name),
              rec.get("stands_for") == "step1/" + p.name and "from" not in rec, rec)
    check("synthetic files live under synthetic/, outside step1/ and every experiment directory",
          all(n.startswith("synthetic/") for n in syn) and not (set(syn) & set(files)))
    check("a recorded why covers every synthetic file",
          set(syn) <= {f for a in man.get("synthetic_why") or [] if a.get("why") for f in a.get("files") or []})

    print("Step 1 copies (replay case R8)")
    cps = man.get("step1_copies") or {}
    check("R8's Step 1 copies are pinned: select_summary.json and admit_summary.json",
          set(cps) == {"step1_copies/select_summary.json", "step1_copies/admit_summary.json"}, sorted(cps))
    for name, rec in sorted(cps.items()):
        p = FIX / name
        check("%s exists" % name, p.is_file())
        if not p.is_file():
            continue
        check("%s sha256 %s" % (name, rec["sha256"][:12]), sha(p) == rec["sha256"], sha(p))
        check("%s is %d bytes" % (name, rec["bytes"]), p.stat().st_size == rec["bytes"], p.stat().st_size)
        check("%s is a copy of results/framework/inc/step1/%s and stands for step1/%s" % (name, p.name, p.name),
              rec.get("from") == "results/framework/inc/step1/" + p.name and rec.get("stands_for") == "step1/" + p.name,
              rec)
        src = REPO / rec["from"]
        if src.is_file() and sha(src) != rec["sha256"]:
            print("  note %s: the results copy %s has moved on from the pinned fixture" % (name, rec["from"]))
    check("Step 1 copies live under step1_copies/, outside step1/ and every experiment directory",
          all(n.startswith("step1_copies/") for n in cps) and not (set(cps) & set(files)) and not (set(cps) & set(syn))
          and not (FIX / "step1_copies" / "exp.json").exists())
    check("a recorded why covers every Step 1 copy",
          set(cps) <= {f for a in man.get("step1_copies_why") or [] if a.get("why") for f in a.get("files") or []})

    print("nothing unpinned")
    present = sorted(str(p.relative_to(FIX)) for p in FIX.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
                     and "__pycache__" not in p.parts)
    extra = [p for p in present if p not in files and p not in syn and p not in cps]
    check("every file in the fixture tree is pinned", not extra, extra)
    pending = {k: v for k, v in (man.get("pending") or {}).items() if k != "how"}
    check("the cluster-only inputs are listed as pending",
          {"step1/select_summary.json", "step1/admit_summary.json", "step1/select_clusters_by_source.json",
           "step1/relevance.json"} <= set(pending), sorted(pending))
    check("nothing pinned is still listed as pending", not (set(pending) & set(files)),
          sorted(set(pending) & set(files)))
    check("a pending entry says where it comes from", all(isinstance(v, str) and v for v in pending.values()))
    for name in pending:
        if (FIX / name).is_file():
            check("pulled %s is pinned" % name, name in files)

    print("the loader reads the pinned files")
    ev = E.load_dir(FIX, "pilot_v1")
    touched = set(ev.touched)
    check("every file the loader read is pinned", touched <= set(files), sorted(touched - set(files)))
    check("the loader read every pinned file of pilot_v1, pilot_v2, pilot_v3, b0_v1 and base_b_v1",
          {n for n in files if n.split("/")[0] in ("pilot_v1", "pilot_v2", "pilot_v3", "b0_v1", "base_b_v1")}
          <= touched, sorted(touched))
    check("the loader refused nothing and noted nothing", not ev.refused and not ev.notes, (ev.refused, ev.notes))
    check("provenance hashes are the pinned hashes",
          all(ev.provenance[n]["sha256"] == files[n]["sha256"] for n in touched))

    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
