#!/usr/bin/env python3
"""The stream diagnoses' mutation harness (docs/CONTINUOUS_LOOP.md 6.8: "a
mutation harness over D20-D33's comparisons, in which every mutant is killed
by some S-case", as tests/test_funnel_ap_mutations.py).

Each comparison below carries "# stream-mutation: SMn" at the end of exactly
one line of inc_autopilot/diagnose_stream.py, and that line is an
"if <condition>:" statement. The harness

  1. copies the package, the tests and the job scripts into a temporary
     repository (results/ and docs/ linked, read only);
  2. rewrites the marked line's condition to False;
  3. runs tests/test_stream_ap_replay.py against the copy;
  4. requires the mutant to be killed: a non-zero exit with more failures than
     the unmutated copy, and names the S-case(s) that killed it.

  id    comparison switched off                                  killed by (expected)
  SM1   D20: eligible target images below the low-water mark 2M  S1
  SM2   D21: a source's yield below floor_gb / floor_su          S2
  SM3   D22: Q >= 4M, or Q >= M with the oldest row 7 days old   S4
  SM4   D24: 4 accepted increments / 3 segments / 30 days since  S17
        the last good milestone (the stream summary's counts)
  SM5   D25: a rollback the stream's recorded 5 v 5 'hurts'      S8
        recommends
  SM6   D25: the boundary check on the summary's means and sd    S8
  SM7   D26: a projected run at >= 0.8 of its walltime           S11
  SM8   D27: /ocean free space under 3 % of the quota            S10
  SM9   D27: the allocation balance under the reserve            S10
  SM10  D28: a source's never-train or base-copy share (the      S14
        intake summary's source_leak)
  SM11  D29: no open candidate and an empty recent discovery     S9
  SM12  D30: half the REJECTs are recipe-caused                  S6
  SM13  D31: a 'data' disposition or a truth 'hurts'             S5, S6, S7
  SM14  D32: 0 ACCEPT in the last 12 decided increments          S6
  SM15  D33: a species fails the guard in half the REJECTs       S6, S19
  SM16  the 'data' disposition of the pinned prior (P_data <=    S6, S19
        p_reject, no helps)
  SM17  the 'species' disposition of the pinned prior            S6, S19
  SM18  D23: a finished segment waits for its commit             S5

Run:  python3 tests/test_stream_ap_mutations.py [SM1 SM5 ...]
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
REPO = ROOT.parent
PKG = "weed_optimizer_framework"
OWNER = "weed_optimizer_framework/tools/inc_autopilot/diagnose_stream.py"
SCRIPT = "tests/test_stream_ap_replay.py"
MUTATIONS = ["SM%d" % i for i in range(1, 19)]
MARK_RE = re.compile(r"#\s*stream-mutation:\s*(SM\d+)\b")
LINE_RE = re.compile(r"^(?P<indent>\s*)if (?P<cond>.+?):\s*#\s*stream-mutation:\s*(?P<id>SM\d+)\s*$")
CLOSE_RE = re.compile(r"^(\d+) failure\(s\)")
CASE_RE = re.compile(r"^case (\S+): fail$")
TIMEOUT = 900
FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def markers(pkg_root):
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
    repo = tmp / "repo"
    code = repo / "weed_llm_benchmark"
    code.mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(str(ROOT / PKG), str(code / PKG), ignore=ign)
    shutil.copytree(str(ROOT / "tests"), str(code / "tests"), ignore=ign)
    for p in ROOT.glob("*.sh"):
        shutil.copy2(str(p), str(code / p.name))
    if (ROOT / "results").exists():
        os.symlink(str(ROOT / "results"), str(code / "results"))
    for name in ("docs", "RESEARCH_LOG.md"):
        if (REPO / name).exists():
            os.symlink(str(REPO / name), str(repo / name))
    return code


def run_script(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code)
    env.pop("INC_DIR", None)
    try:
        p = subprocess.run([sys.executable, str(code / SCRIPT)], cwd=str(code), env=env, capture_output=True,
                           text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, None, [], "timeout"
    fails = None
    for line in reversed((p.stdout or "").splitlines()):
        m = CLOSE_RE.match(line.strip())
        if m:
            fails = int(m.group(1))
            break
    killers = [m.group(1) for m in (CASE_RE.match(x.strip()) for x in (p.stdout or "").splitlines()) if m]
    return p.returncode, fails, killers, ((p.stdout or "")[-600:] + (p.stderr or "")[-600:])


def main(argv=None):
    ids = list(sys.argv[1:] if argv is None else argv) or MUTATIONS
    unknown = [i for i in ids if i not in MUTATIONS]
    if unknown:
        print("unknown mutation id(s): %s" % unknown)
        return 2
    print("the markers")
    found = markers(ROOT)
    for mid in ids:
        hits = found.get(mid) or []
        check("%s: exactly one marker" % mid, len(hits) == 1, hits)
        if len(hits) != 1:
            continue
        f, _n, line = hits[0]
        check("%s: in its owner file" % mid, f == OWNER, f)
        check("%s: the marked line is 'if <condition>:'" % mid, LINE_RE.match(line) is not None, line)
    stray = sorted(k for k in found if k not in MUTATIONS)
    check("no marker names an unknown mutation", not stray, stray)
    if FAILURES:
        print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
        return 1
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stream_mutations_"))
    try:
        code = copy_repo(tmp)
        print("the scenario cases on the unmutated copy")
        b_rc, b_fails, b_kill, tail = run_script(code)
        check("%s runs clean unmutated (exit %s, %s failure(s))" % (SCRIPT, b_rc, b_fails), b_rc == 0 and b_fails == 0,
              tail)
        print("the mutations")
        for mid in ids:
            f, n, _line = found[mid][0]
            path = code / f
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines(True)
            m = LINE_RE.match(lines[n - 1].rstrip("\n"))
            if m is None:
                check("%s: the copy's marked line is an if" % mid, False, lines[n - 1])
                continue
            lines[n - 1] = "%sif False:  # stream-mutation: %s (switched off)\n" % (m.group("indent"), mid)
            path.write_text("".join(lines), encoding="utf-8")
            try:
                rc, fails, killers, tail = run_script(code)
            finally:
                path.write_text(text, encoding="utf-8")
            killed = rc not in (0, None) and b_fails is not None and (fails is None or fails > b_fails)
            check("%s is killed by an S-case (%s): exit %s, %s failure(s) against %s unmutated"
                  % (mid, ", ".join(killers) or "a crash", rc, "a crash" if fails is None else fails, b_fails),
                  killed and (killers or fails is None), tail)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
