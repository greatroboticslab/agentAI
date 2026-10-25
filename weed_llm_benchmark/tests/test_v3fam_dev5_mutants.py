#!/usr/bin/env python3
"""Deviation 5's mutation run (docs/prereg/V3F_dev5_2026-10-24.md): every mutant of its guards must be killed.

The harness copies the package, the tests, the job scripts and deploy/ into a temporary repository (docs/ linked,
read only), checks that the unmutated copy passes each target, then for each mutant replaces exactly one text (which
must occur exactly once in its file), runs its target against the copy and requires it to fail. Targets:
tests/test_v3fam_dev5.py (fast, record-level), tests/test_stream_ap_v3fam_dev5.py (the ticker end to end, one
scenario per run: DEV5_WORLD), tests/test_v3fam_dev5_verdict.py (the verb end to end).

  id    file              what the mutant does                                                     target
  EM1   stream            a refusal of the platform's own making counted as the arm's failure      fast
  EM2   stream            held items released while the records are stale                          fast
  EM3   stream            no item ever held for a person (the bound of three refusals gone)        fast
  EM4   stream            a platform refusal recorded failed (it would be passed as --ticker-failed) fast
  EM5   stream            the job follow does not read the provenance's refusal                     world code
  EM6   stream            a provenance refusal read for another job of the item                     fast
  EM7   diagnose_stream   a family score proposed while a record is stale (no pre-score check)     fast
  EM8   diagnose_stream   a held item treated as due                                                fast
  EM9   stream            a record E3RS's job takes, not in place, is not stale                     fast
  EM10  v3fam_verdict     the verb's deviation 5 guard gone (the second continuation before the arms) fast
  EM11  v3fam_verdict     the superseded failure records read as the arms' failures                fast
  EM12  v3fam_verdict     the second continuation does not check the continuation of 2026-10-22     fast
  EM13  v3fam_verdict     the reads follow the continuation of 2026-10-22, not the second           fast
  EM14  v3fam_verdict     D6 reported 'missing', not its smoke's 'not run (walltime)'               fast
  EM15  v3fam_verdict     an attempt's partial records not moved aside (the write-once refuses)     verdict
  EM17  v3fam_supersede   a move cut off after its link never completed                            fast
  EM18  v3fam_supersede   the record left at its path after its copy is made                       fast
  EM19  v3fam_supersede   a file at an entry's path moved whatever its sha256                      fast
  EM20  v3fam_systems     D8's score job does not move the superseded records first                fast
  EM21  v3fam_trained     D4's score job does not move the superseded records first                fast
  EM22  v3fam_verdict     the verb does not move the superseded records first                       fast
  EM23  stream_remote     the deploy check does not compare the code the records pin               fast
  EM24  stream_remote     the deploy check does not compare a pinned file's sha256                 fast
  EM25  stream_remote     step 2's guard refuses nothing                                           fast
  EM27  diagnose_stream   D28 proposes L24 for a source the stream already quarantines             world d28
  EM28  stream            L24's render runs for a source the stream already quarantines            fast
  EM29  stream            an admit done overwrites a quarantine                                    fast
  EM30  stream            the ticker no longer follows the stream ledger's quarantines             world d28
  EM31  stream            only the first reopen block applies (deviation 5's never)                fast
  EM32  stream            a block reopens an item its history already names                        fast
  EM33  stream            the record-only follow does not read the provenance's refusal            fast
  EM34  stream            the ticker's rerun command uses inc2.v3fam verdict                        fast
  EM36  stream_remote     the snapshot reports no record taken by other code                        world code
  EM37  stream_remote     a record a deviation supersedes reported stale                            world live
  (round 2, 2026-10-25: the review's findings)
  EM38  v3fam_verdict     a failure or cap refusal of a reopened arm named with no job since         fast
  EM39  v3fam_verdict     an item named whose own record is complete (E3RS, the data) not refused    fast
  EM40  v3fam_verdict     a failure record written before the reopening read                        fast
  EM41  v3fam_verdict     D6 not among the reopened arms                                            fast
  EM42  v3fam_verdict     D6's superseded smoke read as its status                                  fast
  EM43  stream_remote     the snapshot reports D6's superseded smoke as its status                  fast
  EM44  stream_remote     step 2 refuses a deploy that restores the bytes the records pin           fast
  EM45  stream_remote     the allow flag takes any recorded deviation, not the newest               fast
  EM46  stream_remote     the allow flag takes a deviation file edited after its pin                 fast
  EM47  stream            a provenance refusal cut at 600 characters (its message lost)             fast
  EM48  v3fam_supersede   a copy cut off leaves its partial file                                    fast
  EM49  v3fam_supersede   the score jobs' group moves D6's smoke too (a race with D6's own job)     fast
  EM50  v3fam_systems     an OSError while moving aside is not the platform's refusal               fast
  EM51  stream            an item held for a person never reached by a deviation's reopening        fast
  EM53  stream_remote     deviation 4's block (v3fam.reopen) not read                               fast
  EM54  v3fam_detr        the smoke's projection from the whole fit                                 detr units
  EM55  v3fam_detr        the smoke does not move the record of 2026-10-21 aside first              detr units
  EM56  v3fam_detr        a backend's epoch clock that saw no epoch accepted                        detr units
  EM57  stream_remote     D6's superseded smoke reported (the ticker never reruns the smoke)        world live

Run:  python3 tests/test_v3fam_dev5_mutants.py [EM1 EM5 ...]
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
ST = "weed_optimizer_framework/tools/inc_autopilot/stream.py"
SRM = "weed_optimizer_framework/tools/inc_autopilot/stream_remote.py"
VR = "weed_optimizer_framework/tools/inc2/v3fam_verdict.py"
SUP = "weed_optimizer_framework/tools/inc2/v3fam_supersede.py"
SYS = "weed_optimizer_framework/tools/inc2/v3fam_systems.py"
TRN = "weed_optimizer_framework/tools/inc2/v3fam_trained.py"
DETR = "weed_optimizer_framework/tools/inc2/v3fam_detr.py"
DETR_UNITS = "tests/test_inc2_v3fam_detr.py?V3FAM_ONLY=units"
FAST = "tests/test_v3fam_dev5.py"
W_CODE = "tests/test_stream_ap_v3fam_dev5.py?DEV5_WORLD=code"
W_LIVE = "tests/test_stream_ap_v3fam_dev5.py?DEV5_WORLD=live"
W_D28 = "tests/test_stream_ap_v3fam_dev5.py?DEV5_WORLD=d28"
VERD = "tests/test_v3fam_dev5_verdict.py"
SUPERSEDE_CALL = "    supersede()                              # deviation 5: the records it supersedes moved aside first\n"
MUTANTS = {
    "EM1": (ST, '        if V3FAM_PLATFORM_REFUSAL_RE.search(str(why or "")):\n            return self._v3fam_platform_refusal(',
            '        if False:\n            return self._v3fam_platform_refusal(', FAST),
    "EM2": (ST, '        if (out.get("equivalence") or {}).get("state") != "current":\n            return\n',
            '        if False:\n            return\n', FAST),
    "EM3": (ST, "V3FAM_PLATFORM_MAX = 3\n", "V3FAM_PLATFORM_MAX = 99\n", FAST),
    "EM4": (ST, "        rec.update(state=DS.V3FAM_HELD_PLATFORM, proposal=None, held_utc=self.utc,",
            '        rec.update(state="failed", proposal=None, held_utc=self.utc,', FAST),
    "EM5": (ST, "                    refusal = self._v3fam_prov_refusal(payload, it, ids)\n",
            '                    refusal = ""\n', W_CODE),
    "EM6": (ST, '            if str(rec.get("job_id") or "").split("_")[0] not in {str(j).split("_")[0] for j in ids}:\n'
                '                return ""\n', '            if False:\n                return ""\n', FAST),
    "EM7": (DS, '        if dep == "ok" and stale:\n', '        if False:\n', FAST),
    "EM8": (DS, 'V3FAM_HELD_PLATFORM = "held (platform)"', 'V3FAM_HELD_PLATFORM = "failed_once"', FAST),
    "EM9": (ST, '        missing = sorted(set(er.get("names") or ()) - present) if er.get("after") and sc.get(er["after"]) == '
                '"done" \\\n            else []\n', "        missing = []\n", FAST),
    "EM10": (VR, "        if second and a in DEV5_REOPENED_ARMS:\n", "        if False:\n", FAST),
    "EM11": (VR, "    if not p.exists() or superseded_failure(item, p):\n        return False\n",
             "    if not p.exists():\n        return False\n", FAST),
    "EM12": (VR, "    if is_first_continuation():\n        out.append(", "    if False:\n        out.append(", FAST),
    "EM13": (VR, '    return cap_path(CONTINUATION_2["verdict"]) if is_first_continuation() else '
                 'cap_path(CONTINUATION["verdict"])\n', '    return cap_path(CONTINUATION["verdict"])\n', FAST),
    "EM14": (VR, '        if arm == "D6" and str(d6_smoke_status() or "").startswith("not run"):\n',
             "        if False:\n", FAST),
    "EM15": (VR, "            aside = set_aside_partial(names)\n", "            aside = []\n", VERD),
    "EM17": (SUP, "        if dst.is_symlink() or (dst.exists() and _sha_or_none(dst) != sha):\n",
             "        if dst.is_symlink() or dst.exists():\n", FAST),
    "EM18": (SUP, "        os.unlink(str(src))\n        moved.append(rel)\n", "        moved.append(rel)\n", FAST),
    "EM19": (SUP, "        if _sha_or_none(src) != sha:\n            continue\n        if dst.is_symlink()",
             "        if not src.exists():\n            continue\n        if dst.is_symlink()", FAST),
    "EM20": (SYS, "    supersede()                              # deviation 5: the records it supersedes moved aside first\n",
             "    pass\n", FAST),
    "EM21": (TRN, "    SYS.supersede()                          # deviation 5: the records it supersedes moved aside first\n",
             "    pass\n", FAST),
    "EM22": (VR, SUPERSEDE_CALL, "    pass\n", FAST),
    "EM23": (SRM, "        changed = sorted(k for k in V3FAM_CODE_FILES if now.get(k) != pin.get(k))\n",
             "        changed = []\n", FAST),
    "EM24": (SRM, "        elif got != sha:\n", "        elif False:\n", FAST),
    "EM25": (SRM, "        if not other:\n            return True, (\"pass:", "        if True:\n            return True, (\"pass:",
             FAST),
    "EM27": (DS, '                                                      for h in hits if str(h["source"]) not in qs]\n',
             "                                                      for h in hits]\n", W_D28),
    "EM28": (ST, '            if str(pr["source"]) in self._stream_quarantined():\n', "            if False:\n", FAST),
    "EM29": (ST, '                if s.get("status") not in ("quarantined", "closed"):\n                    # a quarantine',
             "                if True:\n                    # a quarantine", FAST),
    "EM30": (ST, "        self._resync_quarantines()\n        for h in ((d28.get", "        for h in ((d28.get", W_D28),
    "EM31": (ST, "        for ro in self._v3fam_reopen_blocks():\n            out += self._v3fam_reopen_block(ro)\n",
             "        for ro in self._v3fam_reopen_blocks()[:1]:\n            out += self._v3fam_reopen_block(ro)\n", FAST),
    "EM32": (ST, ' or rec.get("reopened") == marker \\\n                    or marker in [h.get("marker") for h in '
                 'rec.get("reopen_history") or [] if isinstance(h, dict)]:\n',
             ' or rec.get("reopened") == marker:\n', FAST),
    "EM33": (ST, "                ref = self._v3fam_prov_refusal(payload, it, None)\n", '                ref = ""\n', FAST),
    "EM34": (ST, 'run_inc2_build.sh inc2.v3fam_verdict verdict%s" % (sb, cr)', 'run_inc2_build.sh inc2.v3fam verdict%s" % '
                                                                         '(sb, cr)', FAST),
    "EM36": (SRM, "        elif diff:\n            stale.append(p.name)\n", "        elif False:\n            stale.append(p.name)\n",
             W_CODE),
    "EM37": (SRM, "        if superseded:\n            sup.append(p.name)\n", "        if False:\n            sup.append(p.name)\n",
             W_LIVE),
    # round 2 (2026-10-25): the review's findings
    "EM38": (VR, "            if any(attempted_since(i, DEV5_REOPEN_UTC) for i in named):\n", "            if named:\n",
             FAST),
    "EM39": (VR, "    bad = contradicted(refused, gone) if second else []\n", "    bad = []\n", FAST),
    "EM40": (VR, '        return isinstance(d, dict) and str(d.get("written_utc") or "") >= DEV5_REOPEN_UTC\n',
             "        return True\n", FAST),
    "EM41": (VR, 'DEV5_REOPENED_ARMS = DEV5_ARMS + ("D6",)', "DEV5_REOPENED_ARMS = DEV5_ARMS", FAST),
    "EM42": (VR, "    if p.is_file() and SUP.is_superseded(SUP.D6_SMOKE, F._sha(p)):\n        return None\n",
             "    if False:\n        return None\n", FAST),
    "EM43": (SRM, "    if sm_sup:\n        sm = None\n", "    if False:\n        sm = None\n", FAST),
    "EM44": (SRM, "        if not other:\n            return True, (\"pass:", "        if False:\n            return True, (\"pass:",
             FAST),
    "EM45": (SRM, '    blk = blocks[-1] if blocks and blocks[-1].get("file") == allow else None\n',
             '    blk = next((b for b in blocks if b.get("file") == allow), None)\n', FAST),
    "EM46": (SRM, '    if got != blk.get("sha256"):\n', "    if False:\n", FAST),
    "EM47": (ST, '        return _refusal_text(rec.get("refusal"))\n', '        return _short(rec.get("refusal"), 600)\n', FAST),
    "EM48": (SUP, "        if os.path.lexists(str(tmp)):\n            os.unlink(str(tmp))",
             "        if False:\n            os.unlink(str(tmp))", FAST),
    "EM49": (SUP, "    return GROUP_D6 if rel == D6_SMOKE else GROUP_SCORES\n", "    return GROUP_SCORES\n", FAST),
    "EM50": (SYS, "    except OSError as e:                     # a full disk, a lost mount: the platform's, never the arm's\n",
             "    except ZeroDivisionError as e:\n", FAST),
    "EM51": (ST, '("failed", "done", DS.V3FAM_HELD_PLATFORM)', '("failed", "done")', FAST),
    "EM53": (SRM, '    for key in ("reopen", "reopens"):\n', '    for key in ("reopens",):\n', FAST),
    "EM54": (DETR, "        secs, measure = smoke_epoch_seconds(rec, backend)\n",
             '        secs, measure = float(rec["fit_seconds"]), "the whole fit"\n', DETR_UNITS),
    "EM55": (DETR, "    supersede_smoke()                        # deviation 5: the smoke of 2026-10-21 moved aside first\n",
             "    pass\n", DETR_UNITS),
    "EM56": (DETR, '    if getattr(backend, "epoch_clock", False):\n        raise', "    if False:\n        raise", DETR_UNITS),
    "EM57": (SRM, "    if sm_sup:\n        sm = None\n", "    if False:\n        sm = None\n", W_LIVE),
}


def copy_tree(dst):
    """The package, the tests, the job scripts and deploy/ in a temporary repository (docs linked, read only)."""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    code = dst / "weed_llm_benchmark"
    shutil.copytree(str(ROOT / "weed_optimizer_framework"), str(code / "weed_optimizer_framework"), ignore=ign,
                    symlinks=True)
    shutil.copytree(str(TESTS), str(code / "tests"), ignore=ign)
    shutil.copytree(str(ROOT / "deploy"), str(code / "deploy"), ignore=ign)
    for p in ROOT.glob("*.sh"):
        shutil.copy2(str(p), str(code / p.name))
    os.symlink(str(REPO / "docs"), str(dst / "docs"))
    return code


def run(code, target, timeout=2400):
    script, _, envs = target.partition("?")
    env = dict(os.environ)
    for kv in [x for x in envs.split("&") if x]:
        k, _, v = kv.partition("=")
        env[k] = v
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(code / script)], cwd=str(code), capture_output=True, text=True,
                           timeout=timeout, env=env)
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
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="v3fam_dev5_mut_"))
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
            print("%s %s (%s, %.0fs): %s" % (m, "killed" if killed else "SURVIVED", target.rsplit("/", 1)[-1], secs,
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
