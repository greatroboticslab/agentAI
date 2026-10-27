#!/usr/bin/env python3
"""The mutation run of the 2026-10-26 amendment (docs/CONTINUOUS_LOOP.md): every mutant of its guards must be killed.

The harness copies the package, the tests and the job scripts into a temporary repository, checks that the unmutated
copy passes the mutant's test file, then for each mutant replaces exactly one text (which must occur exactly once in
its file), runs the test file against the copy and requires it to fail.

  id    file              what the mutant does
  SG1   stream            a done id a diagnosis calls for again is proposed again (no done check)
  SG2   stream            a done id filed as declined stalls as declined
  SG3   stream            the executor's 'already ran' of a done id files it as declined
  SG4   stream            the lane stays wanted for a done id (its stall never ends)
  SG5   stream            no frozen experiment is snapshotted again for its audit
  SG6   stream            the old frozen record is overwritten, not moved aside
  SG7   stream            the audits run before audits_run existed are not read from the ledger
  SG8   diagnose_stream   D31 never reads the audit's record
  SG9   diagnose_stream   D31 calls for L4 although the ticker reports it ran
  SG10  diagnose_stream   an audit that judged no box is read as conclusive
  SG11  diagnose_stream   blames before a person's lift still count toward L24
  SG12  stream            a person's denial of D31's L24 does not keep the source
  SG13  stream            L24 after a person's lift takes the old id
  SG14  stream            the byte plan ignores the plan a refusal raised (the estimate again)
  SG15  stream            a lab fetch's collector refusal is the lane's failure again
  SG16  stream            over_byte_plan never planned again (closed at once)
  SG17  stream            a source closed on a refusal leaves the zero-yield run
  SG18  stream            a byte cap below the plan (disk, envelope) read as the source's size
  SG19  stream            download_failed read as the source's outcome
  SG20  both job scripts  the job's settings copy keeps mlflow as the shared file has it
  SG21  both job scripts  the job's settings are exported whatever the check found
  SG22  run_inc2_job.sh   the block missing from one job script
  SG23  both job scripts  an inherited job copy (a parent job's YOLO_CONFIG_DIR) read as the shared settings
  SG24  both job scripts  the in-job advance hands the job's own copy to the next array
  SG25  stream            a source's repeated failure counts against the lane again
  SG26  diagnose_stream   D20 ranks a failed source first again
  SG27  stream            a source held for a person leaves the zero-yield run (a sweep of every candidate)
  SG28  stream            a refusal naming the machine (ImportError) read as the source's
  SG29  stream            no_data_files closes the source on one refusal
  SG30  stream            the same refusal on every source never holds the lane
  SG31  stream            a person's denial of a fetch leaves its source the top candidate
  SG32  stream            the replan takes the whole per-source room (not bounded, not by the envelope)
  SG33  stream            a Content-Length cap below the plan read as the source's size
  SG34  stream            'fits in X GB' compared to the plan to 1 %, not at its two decimals
  SG35  diagnose_stream   an audit that compared no box with a baseline read as conclusive
  SG36  diagnose_stream   an above-baseline audit of under the share is no finding
  SG37  stream            blames before a person's lift still close the source (D21)
  SG38  stream            a person's lift of a quarantine never reaches the ticker's record
  SG39  diagnose_stream   D21 closes a source a person kept
  SG40  stream            stream reopen never releases a source the stream held for a person
  SG41  stream            over_byte_plan replanned more than once
  SG42  stream            a lift read over a quarantine the ticker took after it

Run:  python3 tests/test_stream_ap_stalls_mutants.py [SG1 SG5 ...]
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
REPO = ROOT.parent
ST = "weed_optimizer_framework/tools/inc_autopilot/stream.py"
DSM = "weed_optimizer_framework/tools/inc_autopilot/diagnose_stream.py"
J1, J2 = "run_inc_job.sh", "run_inc2_job.sh"
STALLS = "tests/test_stream_ap_stalls.py"
MLF = "tests/test_job_mlflow_off.py"
MUTANTS = {
    "SG1": (ST, "            if ran_at:\n                # the same id ran and ended done",
            "            if False:\n                # the same id ran and ended done", STALLS),
    "SG2": (ST, "            if declined and not ran_at:\n", "            if declined:\n", STALLS),
    "SG3": (ST, '            ran_at = self._done_at(p["id"], ledger=True)\n            if ran_at:\n',
            "            ran_at = None\n            if ran_at:\n", STALLS),
    "SG4": (ST, "                want[lane_name] -= 1\n", "                pass\n", STALLS),
    "SG5": (ST, '        return sorted(e for e, r in rf.items() if isinstance(r, dict) and r.get("state") == "pending")\n',
            "        return []\n", STALLS),
    "SG6": (ST, "                    os.replace(str(path), str(aside))\n", "                    pass\n", STALLS),
    "SG7": (ST, '                if rec.get("lever") == "L4" and rec.get("exp"):\n', "                if False:\n",
            STALLS),
    "SG8": (DSM, "        got = _audit_result(v, evs.get(e), e, ss, share, expected)\n", "        got = None\n", STALLS),
    "SG9": (DSM, "        elif isinstance(ran.get(e), dict):\n", "        elif False:\n", STALLS),
    "SG10": (DSM, "conclusive = species_boxes > 0 and judged >= share * species_boxes - EPS and compared",
             "conclusive = species_boxes > 0 and compared", STALLS),
    "SG11": (DSM, '                    if line is None or line > lifted.get(str(s0), -1):\n',
             "                    if True:\n", STALLS),
    "SG12": (ST, '        elif it.get("lever") == "L24" and src and params.get("cite") == "D31":\n',
             "        elif False:\n", STALLS),
    "SG13": (ST, '        attempt = max(int((st.get("attempts") or {}).get(akey) or 0), floor_attempt)\n',
             '        attempt = int((st.get("attempts") or {}).get(akey) or 0)\n', STALLS),
    "SG14": (ST, "        return max(1, min(cap, max(int(_num(est) or default), raised)))\n",
             "        return max(1, min(cap, int(_num(est) or default)))\n", STALLS),
    "SG15": (ST, '            ref = None if res.get("lost") else self._collector_refusal(it, res.get("rc"), '
                 'self._collect_result(res))\n', "            ref = None\n", STALLS),
    "SG16": (ST, "            if new is None:\n                kind = \"close\"\n",
             "            if True:\n                kind = \"close\"\n", STALLS),
    "SG17": (ST, '                    self._zero_yield(src, "refused", reasons={ref["code"]: 1})\n',
             "                    pass\n", STALLS),
    "SG18": (ST, "                if exact and cap < 0.99 * plan:\n", "                if False:\n", STALLS),
    "SG19": (ST, 'SOURCE_REFUSAL_PERSON = ("licence_unresolved",',
             'SOURCE_REFUSAL_PERSON = ("download_failed", "licence_unresolved",', STALLS),
    "SG20": ((J1, J2), 's["mlflow"] = False\n', 's["mlflow"] = s["mlflow"]\n', MLF),
    "SG21": ((J1, J2), "if why:\n    sys.exit", "if False:\n    sys.exit", MLF),
    "SG22": (J2, "# >>> mlflow-off", "# >>> (removed)", MLF),
    "SG23": ((J1, J2), 'case "${YOLO_CONFIG_DIR:-}" in */ultralytics_cfg_*) unset YOLO_CONFIG_DIR ;; esac\n', ":\n",
             MLF),
    "SG24": ((J1, J2), '    (if [ -n "${UL_CALLER_SET:-}" ]; then export YOLO_CONFIG_DIR="$UL_CALLER"; else unset '
                       'YOLO_CONFIG_DIR; fi\n', "    (:\n", MLF),
    "SG25": (ST, '        repeat = bool(src) and ln == "DATA" and str(src) in run\n', "        repeat = False\n",
             STALLS),
    "SG26": (DSM, '    out.sort(key=lambda sc: (int(_num(sc[1].get("failures")) or 0), -sc[0], str(sc[1].get("id"))))\n',
             '    out.sort(key=lambda sc: (-sc[0], str(sc[1].get("id"))))\n', STALLS),
    "SG27": (ST, '            self._zero_yield(src, "held", reasons={ref["code"]: 1})\n', "            pass\n", STALLS),
    "SG28": (ST, "        elif ENVIRONMENT_RE.search(detail):\n", "        elif False:\n", STALLS),
    "SG29": (ST, '        elif action == "close" or (action is None and code in SOURCE_REFUSAL_CLOSE):\n',
             '        elif action == "close" or code in SOURCE_REFUSAL_CLOSE + SOURCE_REFUSAL_RETRY:\n', STALLS),
    "SG30": (ST, '        if len(last) >= REFUSAL_RUN_HOLD and len({r["code"] for r in last}) == 1:\n',
             "        if False:\n", STALLS),
    "SG31": (ST, '        elif it.get("lever") in DENIED_CLOSES and src and it.get("lever") not in REVIEW_LEVERS:\n',
             "        elif False:\n", STALLS),
    "SG32": (ST, "        got = min(want, room) if left is None else min(want, room, left)\n", "        got = room\n",
             STALLS),
    "SG33": (ST, 'BYTE_CAP_FORMS = ((re.compile(r"(\\d+) bytes listed, over the (\\d+)-byte cap"), 1, 2),\n'
                 '                  (re.compile(r"Content-Length (\\d+) over the (\\d+)-byte cap"), 1, 2),\n',
             'BYTE_CAP_FORMS = ((re.compile(r"(\\d+) bytes listed, over the (\\d+)-byte cap"), 1, 2),\n', STALLS),
    "SG34": (ST, '                if not exact and cap < float("%.2f" % (plan / 1e9)) * 1e9 - 1:\n',
             "                if not exact and cap < 0.99 * plan:\n", STALLS),
    "SG35": (DSM, " and compared >= share * judged - EPS\n", "\n", STALLS),
    "SG36": (DSM, '        above = sp.get("above_baseline") is True and compared > 0\n', "        above = False\n",
             STALLS),
    "SG37": (ST, '            st["sources"].setdefault(src, {})["blames"] = int(n if counted is None else counted.get(src) '
                 'or 0)\n', '            st["sources"].setdefault(src, {})["blames"] = int(n)\n', STALLS),
    "SG38": (ST, "        self._resync_lifts()\n        return out\n", "        return out\n", STALLS),
    "SG39": (DSM, '        if int(r.get("blames") or 0) >= blames and not r.get("blame_kept_by"):\n',
             '        if int(r.get("blames") or 0) >= blames:\n', STALLS),
    "SG40": (ST, '            if s and s.get("status") == "held" and s.get("held_by") == "stream" and rec.get("utc") \\\n',
             '            if False and s.get("held_by") == "stream" and rec.get("utc") \\\n', STALLS),
    "SG41": (ST, '            if _num(s.get("byte_plan")):\n', "            if False:\n", STALLS),
    "SG42": (ST, '            if not e.get("utc") or str(e["utc"]) <= str(s.get("quarantined_utc") or ""):\n',
             "            if False:\n", STALLS),
}


def copy_tree(dst):
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    code = dst / "weed_llm_benchmark"
    shutil.copytree(str(ROOT / "weed_optimizer_framework"), str(code / "weed_optimizer_framework"), ignore=ign,
                    symlinks=True)
    shutil.copytree(str(TESTS), str(code / "tests"), ignore=ign)
    for p in ROOT.glob("*.sh"):
        shutil.copy2(str(p), str(code / p.name))
    if (ROOT / "results").exists():
        os.symlink(str(ROOT / "results"), str(code / "results"))
    for name in ("docs", "RESEARCH_LOG.md"):
        if (REPO / name).exists():
            os.symlink(str(REPO / name), str(dst / name))
    return code


def run(code, script):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(code)] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    env.pop("INC_DIR", None)
    try:
        p = subprocess.run([sys.executable, str(code / script)], cwd=str(code), env=env, capture_output=True,
                           text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    return p.returncode, (p.stdout or "")[-1500:] + (p.stderr or "")[-500:]


def main(argv=None):
    ids = list(sys.argv[1:] if argv is None else argv) or list(MUTANTS)
    bad = [i for i in ids if i not in MUTANTS]
    if bad:
        print("unknown mutant(s): %s" % bad)
        return 2
    fails = []
    for mid in ids:
        fs, old, new, _t = MUTANTS[mid]
        for f in (fs if isinstance(fs, tuple) else (fs,)):
            n = (ROOT / f).read_text().count(old)
            if n != 1:
                print("  FAIL %s: its text occurs %d time(s) in %s" % (mid, n, f))
                fails.append(mid)
    if fails:
        print("\n%d failure(s)" % len(fails))
        return 1
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stalls_mutants_"))
    try:
        code = copy_tree(tmp)
        for script in sorted({MUTANTS[m][3] for m in ids}):
            rc, tail = run(code, script)
            if rc != 0:
                print("  FAIL the unmutated copy fails %s (exit %s): %s" % (script, rc, tail[-800:]))
                return 1
            print("  ok   the unmutated copy passes %s" % script)
        for mid in ids:
            fs, old, new, script = MUTANTS[mid]
            saved = {}
            for f in (fs if isinstance(fs, tuple) else (fs,)):
                saved[f] = (code / f).read_text()
                (code / f).write_text(saved[f].replace(old, new, 1))
            try:
                rc, tail = run(code, script)
            finally:
                for f, text in saved.items():
                    (code / f).write_text(text)
            killed = rc not in (0, None)
            last = [x for x in tail.splitlines() if "failure(s)" in x]
            print("  %s %s killed by %s (exit %s; %s)" % ("ok  " if killed else "FAIL", mid, script, rc,
                                                         last[-1].strip() if last else tail[-200:]))
            if not killed:
                fails.append(mid)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d of %d killed; %d failure(s)%s" % (len(ids) - len(fails), len(ids), len(fails),
                                                ": " + ", ".join(fails) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
