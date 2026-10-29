#!/usr/bin/env python3
"""What a deploy ships for DS1 (deploy/deploy_funnel.sh; docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29, DS.9.3):

- the git-root paths hold the amendment's byte copy (docs/prereg/DS1_2026-10-29.md, hashing to the registered
  sha256 that the ticker's gate compares), its two OpenTimestamps proofs and the stamp file the delegate's
  inc2.ds1 timestamp reads on the login node;
- the package paths hold inc2 (DS1's four modules), run_inc2_long.sh (lever L32X) and tests/fixtures/ds1;
- the job scripts hash DS1's modules (their module lists), so a drift between the cluster's two copies refuses;
- the replay set the deploy runs first holds S30 (executor.STREAM_REPLAY_CASES and the replay's cases).

Run:  python3 tests/test_inc2_ds1_deploy.py
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

TESTS = pathlib.Path(__file__).resolve().parent
PKG = TESTS.parent
GIT = PKG.parent
sys.path.insert(0, str(PKG))
FAILURES = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def bash_array(script, name):
    """The elements of a bash array as the script defines it (`NAME=(` and every `NAME+=(` to the line whose closing
    parenthesis ends it, comments dropped), word-split by bash itself; None when the script never defines it."""
    lines = script.read_text().splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"%s\+?=\(" % re.escape(name), l)]
    if not starts:
        return None
    out = []
    for start in starts:
        body = []
        for i, l in enumerate(lines[start:]):
            l = re.sub(r"(^|\s)#.*$", "", l)
            if i == 0:
                l = l[l.index("(") + 1:]
            if l.rstrip().endswith(")"):
                body.append(l.rstrip()[:-1])
                break
            body.append(l)
        p = subprocess.run(["bash", "-c", "A=(%s)\nprintf '%%s\\n' \"${A[@]}\"" % "\n".join(body)],
                           capture_output=True, text=True)
        out += [x for x in p.stdout.splitlines() if x]
    return out


def main():
    from weed_optimizer_framework.tools.inc2 import ds1 as D
    from weed_optimizer_framework.tools.inc_autopilot import executor as X
    from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS
    df = PKG / "deploy" / "deploy_funnel.sh"
    root = bash_array(df, "ROOT_PATHS") or []
    pkg = bash_array(df, "PKG_PATHS") or []
    files = ["docs/prereg/DS1_2026-10-29.md", "docs/prereg/DS1_2026-10-29.md.ots",
             "docs/prereg/DS1_2026-10-29.stamp.txt", "docs/prereg/DS1_2026-10-29.stamp.txt.ots"]
    check("the git-root paths ship the amendment, its two proofs and the stamp file", all(f in root for f in files),
          [f for f in files if f not in root])
    cfg = LS.load_domain("weed")["ds1"]
    amend = GIT / cfg["amendment_file"]
    check("  the amendment's file is the gate's (stream_domains/weed.json) and hashes to the registered sha256",
          cfg["amendment_file"] == files[0] and hashlib.sha256(amend.read_bytes()).hexdigest()
          == cfg["amendment_sha256"] == D.PREREG_SHA256)
    check("  the proofs stamp the amendment and the stamp file (OpenTimestamps headers)",
          D.ots_digest(GIT / files[1]) == D.PREREG_SHA256
          and D.ots_digest(GIT / files[3]) == hashlib.sha256((GIT / files[2]).read_bytes()).hexdigest())
    revs = cfg.get("revisions") or []
    rfiles = [x for r in D.REVISIONS for x in (r["file"], r["file"] + ".ots", r["stamp_file"],
                                               r["stamp_file"] + ".ots")]
    check("the git-root paths ship DS1's Revisions, their proofs and stamp files", revs and all(f in root for f in
                                                                                             rfiles),
          [f for f in rfiles if f not in root])
    check("  each Revision's file hashes to the sha256 the gate pins (stream_domains/weed.json) and inc2.ds1 records; "
          "its proofs stamp it and its stamp file, which names its sha256 and commit",
          [(r["file"], r["sha256"]) for r in revs] == [(r["file"], r["sha256"]) for r in D.REVISIONS] and
          all(hashlib.sha256((GIT / r["file"]).read_bytes()).hexdigest() == r["sha256"] and
              D.ots_digest(GIT / (r["file"] + ".ots")) == r["sha256"] and
              D.ots_digest(GIT / (r["stamp_file"] + ".ots")) == hashlib.sha256(
                  (GIT / r["stamp_file"]).read_bytes()).hexdigest() and
              r["sha256"] in (GIT / r["stamp_file"]).read_text() and r["commit"] in (GIT / r["stamp_file"]).read_text()
              for r in D.REVISIONS))
    check("the package paths ship inc2 (DS1's modules), run_inc2_long.sh and tests/fixtures/ds1",
          "weed_optimizer_framework/tools/inc2" in pkg and "run_inc2_long.sh" in pkg and "tests/fixtures/ds1" in pkg
          and all((PKG / "weed_optimizer_framework" / "tools" / "inc2" / n).is_file()
                  for n in ("ds1.py", "ds1_data.py", "ds1_sheets.py", "ds1_long.py")), pkg[-12:])
    mods = {"ds1.py", "ds1_data.py", "ds1_sheets.py", "ds1_long.py"}
    for sh in ("run_inc2_build.sh", "run_inc2_long.sh"):
        got = set(re.findall(r"tools/inc2/(ds1[a-z_]*\.py)", "\n".join(bash_array(PKG / sh, "MODULES") or [])))
        check("%s hashes DS1's four modules (its module list)" % sh, got == mods, sorted(got))
    pre = bash_array(df, "PREFLIGHT") or []
    sys.path.insert(0, str(TESTS))
    rp = (TESTS / "test_stream_ap_replay.py").read_text()
    check("the deploy's pre-flight runs the stream replay, which holds S30 (executor.STREAM_REPLAY_CASES)",
          "tests/test_stream_ap_replay.py" in pre and "S30" in X.STREAM_REPLAY_CASES and '("S30", s30)' in rp)
    check("  and this check of what ships for DS1, before anything is copied", "tests/test_inc2_ds1_deploy.py" in pre)
    text = df.read_text()
    check("the deploy record names whether HEAD descends from DS1's amendment commit (ds1_amendment_ancestor)",
          "DS1_COMMIT=%s" % D.PREREG_COMMIT in text and '"ds1_amendment_ancestor": $DS1_ANCESTOR' in text)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
