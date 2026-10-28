#!/usr/bin/env python3
"""The mutation run of the 2026-10-28 amendment (docs/CONTINUOUS_LOOP.md): every mutant of its guards must be killed.

The harness copies the package, the tests and the job scripts into a temporary repository, checks that the unmutated
copy passes each mutant's test file, then for each mutant replaces exactly one text (which must occur exactly once in
its file), runs the test file against the copy and requires it to fail.

  id    file              what the mutant does
  IV1   normalize         columnar regions read as VIA 1.x's indexed object (the crash of job 49959269)
  IV2   normalize         a checkbox with one name checked names no class
  IV3   normalize         a checkbox with several names checked names the first
  IV4   normalize         parallel lists of different lengths paired anyway
  IV5   normalize         a VIA 3.x project read, not refused
  IV6   normalize         an image described by two VIA files gets both files' boxes
  IV7   normalize         an option id read as itself, not as its description
  IV8   normalize         a project's typed choice no longer read before a text note
  IV9   normalize         a region naming a class in two attributes not recorded
  IV10  normalize         an ellipse's rotation (theta) ignored
  IV11  normalize         a polyline boxed like a polygon
  IV12  normalize         the first file keeps an image another file describes beside it
  IV13  intake            the exams beyond GuardV2's never loaded
  IV14  intake            a copy of a season exam that nothing else refused is kept
  IV15  intake            a v1 index that misses an exam image accepted
  IV16  intake            a v1 index that does not hash as LOCK v1 records accepted
  IV17  intake            a production LOCK v2 that names no v1 LOCK accepted
  IV18  intake            a copy of the photograph both season exams hold counted for the first exam only
  IV19  intake            the exam index keeps only the first owner of a hash
  IV20  intake            an image with an EXIF orientation kept
  IV21  intake            the EXIF check runs over a guard's refusal (a copy named exif_orientation)
  IV22  intake            an image refused for its EXIF orientation not among the images checked
  IV23  intake            the exam reasons left out of D28's base-copy share
  IV24  intake            the legacy-label copy rule never closes a source
  IV25  intake            a complete recovery record's H10d hold-out never checked
  IV26  weed.json         the never-fetch entry of Zenodo 10762138 gone
  IV27  weed.json         the 2SeasonWeedDet8 title rule gone
  IV28  weed.json         the 'Two-Season' title rule gone
  IV29  approvals         another actor withdraws a request
  IV30  approvals         a settled request withdrawn
  IV31  approvals         the log folds a withdrawal by another actor
  IV32  approvals         a withdrawal folds as a denial
  IV33  stream            a failed intake of a fetch on the cluster makes the source a candidate (D20 fetches again)
  IV34  stream            the third failure says the intake runs again
  IV35  stream            the sweep of sources made candidates before this change never runs
  IV36  stream            a second fetch the DATA lane holds is never withdrawn
  IV37  stream            a staging the intake finds gone still read as on the cluster
  IV38  diagnose_stream   D20's pre-check fetches a source to be intaken again
  IV39  diagnose_stream   a sync recorded before the fetch counts
  IV40  stream            a new fetch keeps the sync of the fetch before it
  IV41  stream            a failure recorded before intake_failed_utc existed (the lab's) not read
  IV42  intake            the season exams the domain names never checked
  IV43  weed.json         the domain config names no season exam

Run:  python3 tests/test_collect_intake_via_mutants.py [IV1 IV5 ...]
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
NZ = "weed_optimizer_framework/tools/collect/normalize.py"
IN = "weed_optimizer_framework/tools/collect/intake.py"
CF = "weed_optimizer_framework/tools/collect/domains/weed.json"
AP = "weed_optimizer_framework/tools/brain/approvals.py"
ST = "weed_optimizer_framework/tools/inc_autopilot/stream.py"
DSM = "weed_optimizer_framework/tools/inc_autopilot/diagnose_stream.py"
NV = "tests/test_collect_normalize_via.py"
EX = "tests/test_collect_intake_exams.py"
TA = "tests/test_approvals.py"
RT = "tests/test_stream_ap_intake_retry.py"
MUTANTS = {
    "IV1": (NZ, "    if any(k in regs for k in VIA_REGION_KEYS):\n", "    if False:\n", NV),
    "IV2": (NZ, "    if len(on) == 1:\n        return _via_option(on[0], d), False\n",
            "    if len(on) == 1:\n        return None, False\n", NV),
    "IV3": (NZ, "    return None, len(on) > 1\n", "    return (on[0] if on else None), False\n", NV),
    "IV4": (NZ, "        if len(ra) != len(sa):\n", "        if False:\n", NV),
    "IV5": (NZ, "    if v3:\n", "    if False:\n", NV),
    "IV6": (NZ, "        if f != win_f:\n", "        if False:\n", NV),
    "IV7": (NZ, "    return desc.strip() if isinstance(desc, str) and desc.strip() else name\n", "    return name\n", NV),
    "IV8": (NZ, "        groups = [choice, [kv for kv in items if kv[0] not in {k for k, _v in choice}]] if choice else "
                "[items]\n", "        groups = [items]\n", NV),
    "IV9": (NZ, "                if len(off) > 1:\n", "                if False:\n", NV),
    "IV10": (NZ, '        t = float(sa.get("theta") or 0.0)\n', "        t = 0.0\n", NV),
    "IV11": (NZ, '    if name in ("polyline", "point"):\n', '    if name in ("point",):\n', NV),
    "IV12": (NZ, "        if cur is None or (beside and not cur[1]):\n", "        if cur is None:\n", NV),
    "IV13": (IN, "            exams, exam_rec = load_exam_guard(lock_path, inc, ",
             '            exams, exam_rec = None, {"checked": False}\n            _x = (', EX),
    "IV14": (IN, "                        reason, match = (H10D_REASON if esplit == \"h10d\" else SEASON_REASON), em\n",
             "                        pass\n", EX),
    "IV15": (IN, "        if not keys or keys != got:\n", "        if False:\n", EX),
    "IV16": (IN, "        if not want or not ntp.is_file() or sha256_file(ntp) != want:\n",
             "        if not ntp.is_file():\n", EX),
    "IV17": (IN, '        if not lock.get("testing"):\n            raise GuardUnavailable("LOCK v2 %s names no v1 LOCK',
             '        if False:\n            raise GuardUnavailable("LOCK v2 %s names no v1 LOCK', EX),
    "IV18": (IN, "                    for sp in sorted({o[0] for o in eowners}):\n",
             "                    for sp in [esplit]:\n", EX),
    "IV19": (IN, "        if who not in self.owners.setdefault(int(h), []):\n",
             "        if not self.owners.setdefault(int(h), []):\n", EX),
    "IV20": (IN, "                    if ori not in (None, 1):\n", "                    if False:\n", EX),
    "IV21": (IN, "                if not reason:\n                    # boxes drawn in a frame",
             "                if True:\n                    # boxes drawn in a frame", EX),
    "IV22": (IN, '("exact_dup", "near_dup_intake", EXIF_REASON))', '("exact_dup", "near_dup_intake"))', EX),
    "IV23": (IN, "sum(reasons.get(k, 0)\n                                                                                  "
                 "for k in EXAM_REASONS)", "0", EX),
    "IV24": (IN, '        if len(legacy_equal) >= int(pf["copy_candidate_min"]):\n', "        if False:\n", EX),
    "IV25": (IN, '    if rec.get("status") != "complete":\n', "    if True:\n", EX),
    "IV26": (CF, '    "ref": "10762138",\n', '    "ref": "10762138-not",\n', EX),
    "IV27": (CF, '   "(?i)(?<![0-9])2\\\\s*-?\\\\s*season\\\\s*weed\\\\s*det",\n', '   "(?i)not-2swd8",\n', EX),
    "IV28": (CF, '   "(?i)two[\\\\s_-]*season[\\\\s_-]*weed"\n', '   "(?i)not-two-season"\n', EX),
    "IV29": (AP, '            if item.get("requested_by") != actor:\n', "            if False:\n", TA),
    "IV30": (AP, '            if item.get("status") != "pending":\n                return {"ok": False, "reason": "item is',
             '            if False:\n                return {"ok": False, "reason": "item is', TA),
    "IV31": (AP, '            if item.get("status") != "pending" or rec.get("withdrawn_by") != item.get("requested_by"):\n',
             '            if item.get("status") != "pending":\n', TA),
    "IV32": (AP, '            item.update(status="withdrawn", ', '            item.update(status="denied", ', TA),
    "IV33": (ST, '            again = lever == "L16I" and s["failures"] < 3 and not held',
             '            again = False and s["failures"] < 3 and not held', RT),
    "IV34": (ST, '            again = lever == "L16I" and s["failures"] < 3 and not held',
             '            again = lever == "L16I" and not held', RT),
    "IV35": (ST, "        self._intake_retry_sweep()\n", "        pass\n", RT),
    "IV36": (ST, '            self._withdraw_refetch("DATA", it, src)\n', "            pass\n", RT),
    "IV37": (ST, '        if STAGING_GONE_RE.search(str(why or "")):\n', "        if False:\n", RT),
    "IV38": (DSM, '    elif cand.get("intake_retry"):\n', "    elif False:\n", RT),
    "IV39": (DSM, '        return bool(s.get("synced")) and (su is None or str(su) >= str(s["fetched_utc"]))\n',
             '        return bool(s.get("synced"))\n', RT),
    "IV40": (ST, '            s.pop("synced", None)\n', "            pass\n", RT),
    "IV42": (IN, "    rec = _exam_season(Path(lock_path or C2.LOCK_PATH), idx, season)\n",
             "    rec = _exam_season(Path(lock_path or C2.LOCK_PATH), idx, ())\n", EX),
    "IV43": (CF, '  "v1_exams": [\n   "ood22",\n   "ood23"\n  ],\n', '  "v1_exams": [],\n', EX),
    "IV41": (ST, '            t = ((self.st.get("step_failures") or {}).get(key) or {}).get("utc")\n',
             "            t = None\n", RT),
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
        f, old, _new, _t = MUTANTS[mid]
        n = (ROOT / f).read_text().count(old)
        if n != 1:
            print("  FAIL %s: its text occurs %d time(s) in %s" % (mid, n, f))
            fails.append(mid)
    if fails:
        print("\n%d failure(s)" % len(fails))
        return 1
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="intake_via_mutants_"))
    try:
        code = copy_tree(tmp)
        for script in sorted({MUTANTS[m][3] for m in ids}):
            rc, tail = run(code, script)
            if rc != 0:
                print("  FAIL the unmutated copy fails %s (exit %s): %s" % (script, rc, tail[-800:]))
                return 1
            print("  ok   the unmutated copy passes %s" % script)
        for mid in ids:
            f, old, new, script = MUTANTS[mid]
            saved = (code / f).read_text()
            (code / f).write_text(saved.replace(old, new, 1))
            try:
                rc, tail = run(code, script)
            finally:
                (code / f).write_text(saved)
            killed = rc not in (0, None)
            last = [x for x in tail.splitlines() if "failure(s)" in x or "FAILED" in x or "Error" in x]
            print("  %s %s killed by %s (exit %s; %s)" % ("ok  " if killed else "FAIL", mid, script, rc,
                                                         last[-1].strip()[:160] if last else tail[-200:]))
            if not killed:
                fails.append(mid)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d of %d killed; %d failure(s)%s" % (len(ids) - len(fails), len(ids), len(fails),
                                                ": " + ", ".join(fails) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
