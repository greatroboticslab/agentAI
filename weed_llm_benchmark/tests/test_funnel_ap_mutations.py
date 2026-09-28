#!/usr/bin/env python3
"""The funnel audit's mutation harness (docs/FUNNEL_AUDIT.md 8.9; runner 7.6).

Each check below carries the comment "# funnel-mutation: <id>" at the end of
exactly one line, and that line is an "if <condition>:" statement whose body
enforces the check. The harness

  1. copies the package, the tests and the job scripts into a temporary
     repository laid out like this one (docs/ and results/ linked, read only);
  2. rewrites the marked line's condition to False;
  3. runs the owner's test script against the copy (the script puts its own
     copy first on sys.path);
  4. requires the mutation to be killed: a non-zero exit, with more failures
     than the same script has on the unmutated copy (a script that already
     fails there for another reason must fail more, or crash, under the
     mutation; one that crashes unmutated cannot judge anything).

A missing marker, a duplicated one, one outside its owner file, or a marked
line that is not an "if <condition>:" fails the harness.

  id   check switched off                                     owner file              killing test
  M1   D17's conclusion condition (C)                          inc_autopilot/diagnose  test_funnel_ap_replay (negative controls)
  M2   J1 refused as a judge of its own discards / on KT7     funnel/qualify          test_funnel_qualify
  M3   thresholds fitted only on sentinels                    funnel/qualify          test_funnel_qualify
  M4   the sibling guard                                      funnel/recover          test_funnel_recover
  M5   a guard stage marked recoverable refused               funnel/domain           test_funnel_domain
  M6   the RL qualified on the claimed labels it tests        funnel/qualify          test_funnel_qualify
  M7   a kNN bank entry sharing a lab with the query          funnel/judges           test_funnel_judges
  M8   the never-train check on the unmasked image            funnel/recover          test_funnel_recover
  M9   a source with a detected H6 copy excluded as a whole   funnel/recover          test_funnel_recover
  M10  an exam name in the test-blindness list                inc_autopilot/model     test_funnel_ap_replay (R13)

Run:  python3 tests/test_funnel_ap_mutations.py [M1 M10 ...]
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent                     # weed_llm_benchmark/
REPO = ROOT.parent
PKG = "weed_optimizer_framework"
MUTATIONS = {
    "M1": ("weed_optimizer_framework/tools/inc_autopilot/diagnose.py", "tests/test_funnel_ap_replay.py"),
    "M2": ("weed_optimizer_framework/tools/funnel/qualify.py", "tests/test_funnel_qualify.py"),
    "M3": ("weed_optimizer_framework/tools/funnel/qualify.py", "tests/test_funnel_qualify.py"),
    "M4": ("weed_optimizer_framework/tools/funnel/recover.py", "tests/test_funnel_recover.py"),
    "M5": ("weed_optimizer_framework/tools/funnel/domain.py", "tests/test_funnel_domain.py"),
    "M6": ("weed_optimizer_framework/tools/funnel/qualify.py", "tests/test_funnel_qualify.py"),
    "M7": ("weed_optimizer_framework/tools/funnel/judges.py", "tests/test_funnel_judges.py"),
    "M8": ("weed_optimizer_framework/tools/funnel/recover.py", "tests/test_funnel_recover.py"),
    "M9": ("weed_optimizer_framework/tools/funnel/recover.py", "tests/test_funnel_recover.py"),
    "M10": ("weed_optimizer_framework/tools/inc_autopilot/model.py", "tests/test_funnel_ap_replay.py"),
}
MARK_RE = re.compile(r"#\s*funnel-mutation:\s*(M\d+)\b")
LINE_RE = re.compile(r"^(?P<indent>\s*)if (?P<cond>.+?):\s*#\s*funnel-mutation:\s*(?P<id>M\d+)\s*$")
CLOSE_RE = re.compile(r"^(\d+) failure\(s\)")
TIMEOUT = 1800
FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def markers(pkg_root):
    """{id: [(relative file, line number, line)]} of every marker in the package."""
    out = {}
    for p in sorted(pathlib.Path(pkg_root, PKG).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = MARK_RE.search(line)
            if m:
                out.setdefault(m.group(1), []).append((str(p.relative_to(pkg_root)), i, line))
    return out


def copy_repo(tmp):
    """tmp/repo laid out like this repository: the package, tests and job
    scripts copied (the mutation edits the copy), docs/, RESEARCH_LOG.md and
    results/ linked (read only)."""
    repo = tmp / "repo"
    code = repo / "weed_llm_benchmark"
    code.mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(str(ROOT / PKG), str(code / PKG), ignore=ign)
    shutil.copytree(str(ROOT / "tests"), str(code / "tests"), ignore=ign)
    for p in ROOT.glob("*.sh"):
        shutil.copy2(str(p), str(code / p.name))
    os.symlink(str(ROOT / "results"), str(code / "results"))
    for name in ("docs", "RESEARCH_LOG.md"):
        if (REPO / name).exists():
            os.symlink(str(REPO / name), str(repo / name))
    return code


def run_script(code, script):
    """(returncode, failures or None, tail) of one test script in the copy."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code)
    env.pop("INC_DIR", None)
    try:
        p = subprocess.run([sys.executable, str(code / script)], cwd=str(code), env=env, capture_output=True,
                           text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    fails = None
    for line in reversed((p.stdout or "").splitlines()):
        m = CLOSE_RE.match(line.strip())
        if m:
            fails = int(m.group(1))
            break
        if line.strip() == "ALL PASS":
            fails = 0
            break
    return p.returncode, fails, ((p.stdout or "")[-600:] + (p.stderr or "")[-600:])


def main(argv=None):
    ids = list(sys.argv[1:] if argv is None else argv) or sorted(MUTATIONS, key=lambda x: int(x[1:]))
    unknown = [i for i in ids if i not in MUTATIONS]
    if unknown:
        print("unknown mutation id(s): %s" % unknown)
        return 2
    print("the markers")
    found = markers(ROOT)
    for mid in ids:
        owner, _test = MUTATIONS[mid]
        hits = found.get(mid) or []
        check("%s: exactly one marker" % mid, len(hits) == 1, hits)
        if len(hits) != 1:
            continue
        f, _n, line = hits[0]
        check("%s: in its owner file %s" % (mid, owner), f == owner, f)
        check("%s: the marked line is 'if <condition>:'" % mid, LINE_RE.match(line) is not None, line)
    stray = sorted(k for k in found if k not in MUTATIONS)
    check("no marker names an unknown mutation", not stray, stray)
    if FAILURES:
        print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
        return 1
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="funnel_mutations_"))
    try:
        code = copy_repo(tmp)
        print("the killing tests on the unmutated copy")
        base = {}
        for script in sorted({MUTATIONS[m][1] for m in ids}):
            rc, fails, tail = run_script(code, script)
            base[script] = (rc, fails)
            check("%s runs to its closing line unmutated (exit %s, %s failure(s))" % (script, rc, fails),
                  fails is not None, tail)
        print("the mutations")
        for mid in ids:
            owner, script = MUTATIONS[mid]
            f, n, line = found[mid][0]
            path = code / f
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines(True)
            m = LINE_RE.match(lines[n - 1].rstrip("\n"))
            if m is None:
                check("%s: the copy's marked line is an if" % mid, False, lines[n - 1])
                continue
            lines[n - 1] = "%sif False:  # funnel-mutation: %s (switched off)\n" % (m.group("indent"), mid)
            path.write_text("".join(lines), encoding="utf-8")
            try:
                rc, fails, tail = run_script(code, script)
            finally:
                path.write_text(text, encoding="utf-8")
            b_rc, b_fails = base.get(script, (None, None))
            killed = rc not in (0, None) and (b_fails is not None) and (fails is None or fails > b_fails)
            check("%s (%s) is killed by %s: exit %s, %s failure(s) against %s unmutated"
                  % (mid, owner.rsplit("/", 1)[-1], script.rsplit("/", 1)[-1], rc,
                     "a crash" if fails is None else fails, b_fails), killed, tail)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
