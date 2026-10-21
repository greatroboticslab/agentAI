#!/usr/bin/env python3
"""Deviation 4's mutation run (docs/prereg/V3F_dev4_2026-10-20.md): every mutant of the guards that keep the
family's verdict from being decided before every arm's record is complete or failed, of the reopening and of the
continuation's read path and reproduction check, must be killed.

The harness copies the package and the tests into a temporary repository (docs/ linked, read only), checks that
the unmutated copy passes its target test, then for each mutant replaces exactly one text (which must occur exactly
once in its file), runs the target test against the copy and requires it to fail. Targets: tests/test_v3fam_dev4.py
(fast, record-level) for the guards and the reopening; tests/test_inc2_v3fam_verdict.py for the continuation's
reproduction check inside the verb.

  id    file                 what the mutant does
  DM1   diagnose_stream      the data build's state no longer holds the verdict
  DM2   diagnose_stream      'not run (the family's data build failed)' counts as ended (the run of 2026-10-13)
  DM3   diagnose_stream      a pending build no longer holds the verdict
  DM4   diagnose_stream      a pending D6 job no longer holds the verdict
  DM5   diagnose_stream      L23L proposed whether or not the family is ready
  DM6   diagnose_stream      a failed data build counts as ended
  DM7   diagnose_stream      a pending score arm no longer holds the verdict
  DM8   v3fam_verdict        the verb no longer needs the family's data summary
  DM9   v3fam_verdict        an arm that never started no longer refuses
  DM10  v3fam_verdict        BCLEAN excused even when D0-S triggered it
  DM11  v3fam_verdict        the verb skips its guard
  DM12  v3fam_verdict        the reads follow the verdict of 2026-10-13, not its continuation
  DM13  v3fam_verdict        the reproduction check's tolerance 1.0 instead of 1e-9
  DM14  v3fam_verdict        a primary no longer family-confirmed does not refuse
  DM15  v3fam_verdict        an attempt of a build an arm scores does not start the arm
  DM16  v3fam_verdict        a build refused at the family cap does not excuse the arm that scores it
  DM17  stream               the reopening takes items started after the deviation
  DM18  stream               the reopening repeats under the same marker
  DM19  stream               the earlier proposal ids are not retired (the new proposal could take an old id)
  DM20  stream               the reopening without the deviation's file deployed
  DM21  stream               the earlier attempts' spend dropped from the family cap
  DM22  v3fam_verdict        the continuation written without its reproduction check (tests/test_inc2_v3fam_verdict.py)

Run:  python3 tests/test_v3fam_dev4_mutants.py [DM1 DM5 ...]
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
REPO = ROOT.parent
DS = "weed_optimizer_framework/tools/inc_autopilot/diagnose_stream.py"
VR = "weed_optimizer_framework/tools/inc2/v3fam_verdict.py"
ST = "weed_optimizer_framework/tools/inc_autopilot/stream.py"
FAST = "tests/test_v3fam_dev4.py"
FULL = "tests/test_inc2_v3fam_verdict.py"
MUTANTS = {
    "DM1": (DS, '    if fs.get("data") not in V3FAM_DATA_ENDED:\n', '    if False:\n', FAST),
    "DM2": (DS, '        return v3fam_terminal(st) and st != V3FAM_NOT_RUN_DATA\n',
            '        return v3fam_terminal(st)\n', FAST),
    "DM3": (DS, '    pending += ["build:%s" % a for a in fam.get("builds") or [] if not ended(build.get(a))]\n',
            '    pending += []\n', FAST),
    "DM4": (DS, '    pending += ["detr:%s" % k for k in ("smoke", "train") if not ended(det.get(k))]\n',
            '    pending += []\n', FAST),
    "DM5": (DS, '                if ready and _v3fam_due(fs.get("verdict")):\n',
            '                if _v3fam_due(fs.get("verdict")):\n', FAST),
    "DM6": (DS, 'V3FAM_DATA_ENDED = ("done", "not run (family cap)")',
            'V3FAM_DATA_ENDED = ("done", "not run (family cap)", "failed")', FAST),
    "DM7": (DS, '        if not ended(score.get(a)):\n            pending.append(a)\n',
            '        if False:\n            pending.append(a)\n', FAST),
    "DM8": (VR, '    if VD.read_summary() is None and "data" not in refused and not cap_refused_item("data"):\n',
            '    if False:\n', FAST),
    "DM9": (VR, '        if not arm_started(a):\n            never.append(a)\n',
            '        if False:\n            never.append(a)\n', FAST),
    "DM10": (VR, '        if a == "BCLEAN" and not trig.get("bclean"):\n', '        if a == "BCLEAN":\n', FAST),
    "DM11": (VR, '    ok, why = verdict_ready(cap_refused=cap_refused)\n', '    ok, why = True, ""\n', FAST),
    "DM12": (VR, '    return cap_path(CONTINUATION["verdict"]) if is_partial() else verdict_path()\n',
             '    return verdict_path()\n', FAST),
    "DM13": (VR, '                if abs(float(a) - float(b)) > CONT_REPRO_TOL:\n',
             '                if abs(float(a) - float(b)) > 1.0:\n', FAST),
    "DM14": (VR, '    if bad or lost:\n', '    if bad:\n', FAST),
    "DM15": (VR, '            names.append(RC.V3FAM_EXPS[d[len("build:"):]])\n', '            pass\n', FAST),
    "DM16": (VR, '        if refused & ({"score:%s" % a, "data"} | set(SCORE_DEPS.get(a, ()))):\n',
             '        if refused & {"score:%s" % a, "data"}:\n', FAST),
    "DM17": (ST, '            if not ro.get("before_utc") or not rec.get("started_utc") or str(rec["started_utc"]) >= '
                 'ro["before_utc"]:\n                continue\n',
             '            if False:\n                continue\n', FAST),
    "DM18": (ST, 'if not isinstance(rec, dict) or rec.get("reopened") == marker or rec.get("state") not in',
             'if not isinstance(rec, dict) or rec.get("state") not in', FAST),
    "DM19": (ST, '            self.st["failed_ids"] = (failed + [i for i in ids if i and i not in failed])'
                 '[-FAILED_IDS_KEEP:]\n', '            pass\n', FAST),
    "DM20": (ST, '        if not marker or not ro.get("items") or not self._v3fam_deployed(ro.get("file"), '
                 'ro.get("sha256")):\n', '        if not marker or not ro.get("items"):\n', FAST),
    "DM21": (ST, '            self._v3fam_fold(rec)\n            hist = list(rec.get("reopen_history")',
             '            hist = list(rec.get("reopen_history")', FAST),
    "DM22": (VR, '        continues = continuation_check(dev_doc, ext) if cont else None   # before anything is written\n',
             '        continues = dict(verdict=VERDICT, sha256=PARTIAL_VERDICT_SHA256, deviation=DEVIATION_4, '
             'reproduced=[]) if cont else None\n', FULL),
}


def copy_tree(dst):
    """The package and the tests in a temporary repository (docs linked, read only)."""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(str(ROOT / "weed_optimizer_framework"), str(dst / "weed_llm_benchmark" / "weed_optimizer_framework"),
                    ignore=ign, symlinks=True)
    shutil.copytree(str(TESTS), str(dst / "weed_llm_benchmark" / "tests"), ignore=ign)
    os.symlink(str(REPO / "docs"), str(dst / "docs"))
    return dst / "weed_llm_benchmark"


def run(code, target, timeout=1800):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(code / target)], cwd=str(code), capture_output=True, text=True,
                           timeout=timeout)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        rc, out = -9, "timeout"
    fails = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(("FAIL", "FAILED:"))]
    return rc, fails, time.time() - t0


def main(argv):
    want = argv or list(MUTANTS)
    unknown = [m for m in want if m not in MUTANTS]
    if unknown:
        print("unknown mutant(s): %s" % unknown)
        return 2
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="v3fam_dev4_mut_"))
    failures = []
    try:
        code = copy_tree(tmp)
        for target in sorted({MUTANTS[m][3] for m in want}):
            rc, fails, secs = run(code, target)
            print("unmutated %s: rc %d, %d failing line(s) (%.0fs)" % (target, rc, len(fails), secs))
            if rc != 0:
                print("  the unmutated copy must pass its target first: %s" % fails[:5])
                return 1
        for m in want:
            rel, old, new, target = MUTANTS[m]
            path = code / rel
            text = path.read_text()
            n = text.count(old)
            if n != 1:
                print("%s: its text occurs %d times in %s (must be once)" % (m, n, rel))
                failures.append(m)
                continue
            path.write_text(text.replace(old, new, 1))
            try:
                rc, fails, secs = run(code, target)
            finally:
                path.write_text(text)
            killed = rc != 0 and bool(fails)
            print("%s %s (%s, %.0fs): %s" % (m, "killed" if killed else "SURVIVED", pathlib.Path(target).name, secs,
                                             "; ".join(f[:110] for f in fails[:2])))
            if not killed:
                failures.append(m)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d of %d mutant(s) killed; %d failure(s)" % (len(want) - len(failures), len(want), len(failures)))
    for m in failures:
        print("  FAILED: %s" % m)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
