#!/usr/bin/env python3
"""Deviation 6's mutation run (docs/prereg/V3F_dev6_2026-10-25.md): every mutant of its guards must be killed.

The harness copies the package, the tests, the job scripts and deploy/ into a temporary repository (docs/ linked,
read only), checks that the unmutated copy passes tests/test_v3fam_dev6.py, then for each mutant replaces exactly one
text (which must occur exactly once in its file), runs the test against the copy and requires it to fail.

  id    file              what the mutant does
  FM1   v3fam_loader      the fix never swaps an MPO photo's size (Ultralytics' gate as it was)
  FM2   v3fam_loader      the fix swaps JPEG and MPO whatever Ultralytics returned (a JPEG tagged 6 or 8 swapped twice)
  FM3   v3fam_loader      the fix is never installed
  FM4   v3fam_loader      the fix is not removed when its pass ends (dev passes and later passes run under it)
  FM5   v3fam_loader      the observer does not check that it saw every exam image
  FM6   v3fam_loader      the observer does not compare the changed images with the record's
  FM7   v3fam_loader      the observer does not check the number of builds
  FM8   v3fam_loader      the fix may be nested
  FM9   v3fam_loader      the pre-flight builds its fixed dataset in the unfixed copy (Ultralytics reads that cache back)
  FM10  v3fam_loader      the pre-flight does not compare the fixed size with the loader's frame
  FM11  v3fam_loader      the pre-flight's letterbox scale ignores the batch shape
  FM12  v3fam_loader      the record is read back although a manifest changed
  FM13  v3fam_loader      an identity record without its note is taken on an exam the fix changes
  FM14  v3fam_trained     D1's ood22 and ood23 passes run without the fix
  FM15  v3fam_trained     a kept ood23 pass is not checked for the fix's note
  FM16  v3fam_trained     D1's job skips the pre-flight (an empty record)
  FM17  v3fam_trained     D1's identity reproductions taken without the fix
  FM18  v3fam_supersede   identity_ood23.json of 2026-10-25 not named (another sha256)
  FM19  v3fam_verdict     D1 takes deviation 5's reopening, not deviation 6's
  FM20  v3fam_verdict     a failure record of score:D1 read from deviation 5's reopening on
  FM21  v3fam_verdict     deviation 6's reopening time before D1's attempts of 2026-10-25
  FM22  stream            the loader check's refusal counted as the arm's failure
  FM23  stream            the held card names the reproduction records' remedy for the loader check
  FM24  weed.json         the block's before_utc before D1's last attempt (score:D1 never reopened)
  FM25  weed.json         the block reopens score:D5B too
  FM26  deploy_funnel     the deviation's file not shipped
  FM27  run_inc2_build    the job's module list without inc2/v3fam_loader.py

Run:  python3 tests/test_v3fam_dev6_mutants.py [FM1 FM5 ...]
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
LDR = "weed_optimizer_framework/tools/inc2/v3fam_loader.py"
TRN = "weed_optimizer_framework/tools/inc2/v3fam_trained.py"
SUP = "weed_optimizer_framework/tools/inc2/v3fam_supersede.py"
VR = "weed_optimizer_framework/tools/inc2/v3fam_verdict.py"
ST = "weed_optimizer_framework/tools/inc_autopilot/stream.py"
WJ = "weed_optimizer_framework/tools/inc_autopilot/stream_domains/weed.json"
DF = "deploy/deploy_funnel.sh"
RB = "run_inc2_build.sh"
FAST = "tests/test_v3fam_dev6.py"
MUTANTS = {
    "FM1": (LDR, '    if getattr(img, "format", None) in FIX_FORMATS and s == tuple(img.size) and _tag(img) in SWAP_TAGS:\n',
            "    if False:\n", FAST),
    "FM2": (LDR, '    if getattr(img, "format", None) in FIX_FORMATS and s == tuple(img.size) and _tag(img) in SWAP_TAGS:\n',
            '    if getattr(img, "format", None) in ("JPEG", "MPO") and _tag(img) in SWAP_TAGS:\n', FAST),
    "FM3": (LDR, "    U.exif_size = exif_size\n", "    pass\n", FAST),
    "FM4": (LDR, "        U.exif_size = orig\n", "        pass\n", FAST),
    "FM5": (LDR, "        if seen != keys:\n", "        if False:\n", FAST),
    "FM6": (LDR, "        if set(self.changed) != expected:\n", "        if False:\n", FAST),
    "FM7": (LDR, "        off = sorted(k for k in seen & keys if self.calls[k] != builds)\n", "        off = []\n", FAST),
    "FM8": (LDR, '    if getattr(orig, "_v3fam_dev6", False):\n        refuse("the loader fix is installed already',
            '    if False:\n        refuse("the loader fix is installed already', FAST),
    "FM9": (LDR, '            ds1 = _val_dataset(view1 / "images", Path(tmp) / "runs1")\n',
            '            ds1 = _val_dataset(view / "images", Path(tmp) / "runs1")\n', FAST),
    "FM10": (LDR, "    off = [k for k, r in by.items() if (r[7], r[8]) != (r[9], r[10])]\n", "    off = []\n", FAST),
    "FM11": (LDR, "        out[key] = min(1.0, bh / float(h), bw / float(w))\n", "        out[key] = 1.0\n", FAST),
    "FM12": (LDR, '        if e.get("manifest_sha256") != (C.sha256_file(mp) if mp.is_file() else None):\n',
             "        if False:\n", FAST),
    "FM13": (LDR, "    elif changed:\n", "    elif False:\n", FAST),
    "FM14": (TRN, "                            with LD.deciding(exam) as obs:   # deviation 6: ood22 and ood23 only\n",
             '                            with LD.deciding("dev") as obs:\n', FAST),
    "FM15": (TRN, '                            LD.check_kept(TS._read_json(F.root() / got["file"]), exam, shapes, what)\n',
             "                            pass\n", FAST),
    "FM16": (TRN, "        shapes = LD.preflight()              # deviation 6: every ood22/ood23 frame and letterbox "
                  "proved, once\n", '        shapes = {"exams": {}}\n', FAST),
    "FM17": (TRN, '            LD.identity(exam, "D1", kw, shapes)  # deviation 6: taken under the loader fix when '
                  'missing\n', '            SYS.require_or_make_identity(exam, "D1", kw)\n', FAST),
    "FM18": (SUP, '"3c29764783b8303b0659cfe75ee49b966dff4334394477fba24b2245045ad3ea"', '"%s"' % ("0" * 64), FAST),
    "FM19": (VR, "    if arm in DEV6_REOPENED_ARMS:\n", "    if False:\n", FAST),
    "FM20": (VR, "    return DEV6_REOPEN_UTC if str(item) in DEV6_ITEMS else DEV5_REOPEN_UTC\n",
             "    return DEV5_REOPEN_UTC\n", FAST),
    "FM21": (VR, 'DEV6_REOPEN_UTC = "2026-10-25T17:00:00Z"', 'DEV6_REOPEN_UTC = "2026-10-25T15:00:00Z"', FAST),
    "FM22": (ST, '    r"|deviation 6\'s loader check refuses")', '    r"|deviation 6\'s loader check refuses, never")', FAST),
    "FM23": (ST, '        if V3FAM_LOADER_REFUSAL in str(why or ""):\n', "        if False:\n", FAST),
    "FM24": (WJ, '"before_utc": "2026-10-25T17:00:00Z",', '"before_utc": "2026-10-25T16:00:00Z",', FAST),
    "FM25": (WJ, '     "score:D1"\n    ],\n    "before_utc": "2026-10-25T17:00:00Z"',
             '     "score:D1",\n     "score:D5B"\n    ],\n    "before_utc": "2026-10-25T17:00:00Z"', FAST),
    "FM26": (DF, "            docs/prereg/V3F_dev5_2026-10-24.md docs/prereg/V3F_dev6_2026-10-25.md)",
             "            docs/prereg/V3F_dev5_2026-10-24.md)", FAST),
    "FM27": (RB, "tools/inc2/v3fam_supersede.py tools/inc2/v3fam_loader.py\n",
             "tools/inc2/v3fam_supersede.py\n", FAST),
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
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="v3fam_dev6_mut_"))
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
