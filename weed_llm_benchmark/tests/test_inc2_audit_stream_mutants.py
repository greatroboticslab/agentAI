#!/usr/bin/env python3
"""The mutation run of the 2026-10-27 amendment (docs/CONTINUOUS_LOOP.md): the label audit with the stream's lookup
(inc2/audit_stream.py, run_inc_audit.sh, deploy_funnel.sh) and D31's coverage reading (diagnose_stream.py). Every
mutant must be killed.

The harness copies the package, the tests, the job scripts and deploy/ into a temporary repository, checks that the
unmutated copy passes each test file used, then for each mutant replaces exactly one text (which must occur exactly
once in its file), runs that mutant's test file against the copy and requires it to fail.

  id     file              what the mutant does
  AS1    audit_stream      a masked admission's label box j read as the image's box j (keep ignored)
  AS2    audit_stream      a crop row taken whatever its geometry
  AS3    audit_stream      a crop with a non-finite embedding taken
  AS4    audit_stream      an image whose unmasked original is trusted not refused
  AS5    audit_stream      a queue row matched on key and image only (other bytes taken)
  AS6    audit_stream      a re-queued image's latest row taken over the one with the manifest's label
  AS7    audit_stream      batch.json's record of the batch's files not checked
  AS8    audit_stream      emb.npz / verdicts.npz of another crops.csv taken
  AS9    audit_stream      a batch embedded by another model taken
  AS10   audit_stream      a batch without batch.json (not committed) read
  AS11   audit_stream      a last queue line without its newline read
  AS12   audit_stream      an output under the stream's directory allowed
  AS13   audit_stream      no stream lookup (inc/audit.py's coverage)
  AS14   audit_stream      crops_info.json of another crops.csv taken
  AS15   audit_stream      a batch without crops.csv counted as no_crop_row, not image_not_in_crops
  AS16   audit_stream      a box the intake left out as small counted as no_crop_row
  AS17   audit_stream      inc/audit.py's in_trusted refusal of a stream image dropped (its path in the trusted set)
  AS18   diagnose_stream   no coverage shortfall ever named
  AS19   diagnose_stream   the judged share read over all boxes, not the species-labelled ones
  AS20   diagnose_stream   no card for a shortfall of a judged audit
  AS21   diagnose_stream   a stream-lookup record read as Step 1's only
  AS22   diagnose_stream   a Step-1-only record never read as predating the stream lookup
  AS23   run_inc_audit.sh  lever L4's job runs inc.audit again
  AS24   deploy_funnel.sh  run_inc_audit.sh not shipped
  AS25   audit_stream      no stream lookup, on s002's inc0005 (the cluster's records)
  AS26   audit_stream      keep ignored, on s002's inc0005 (its masked admissions)
  AS27   diagnose_stream   the judged share over all boxes, on s002's records

Run:  python3 tests/test_inc2_audit_stream_mutants.py [AS1 AS5 ...]
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
AST = "weed_optimizer_framework/tools/inc2/audit_stream.py"
DSM = "weed_optimizer_framework/tools/inc_autopilot/diagnose_stream.py"
JOB = "run_inc_audit.sh"
DEP = "deploy/deploy_funnel.sh"
T1 = "tests/test_inc2_audit_stream.py"
T2 = "tests/test_stream_ap_audit_coverage.py"
T3 = "tests/test_inc2_audit_stream_s002.py"
MUTANTS = {
    "AS1": (AST, "            k = int(keep[j])\n", "            k = j\n", T1),
    "AS2": (AST, "            ok = [i for i, dl in zip(ids, deltas) if dl <= AU.GEOM_TOL]\n",
            "            ok = list(ids)\n", T1),
    "AS3": (AST, "            return (i, None, dl) if b.finite(i) else (i, V.FAILED, dl)\n",
            "            return (i, None, dl)\n", T1),
    "AS4": (AST, "    return img in trusted[0] or img in trusted[2] or sha in trusted[1]\n", "    return False\n", T1),
    "AS5": (AST, '                     if q.get("image") == r["image"] and q.get("sha256") == r["sha256"]]\n',
            '                     if q.get("image") == r["image"]]\n', T1),
    "AS6": (AST, "            pick = same or cands\n", "            pick = cands\n", T1),
    "AS7": (AST, "            if want and want != shas[name]:\n", "            if False:\n", T1),
    "AS8": (AST, '            if meta.get("crops_sha256") != crops.sha:\n', "            if False:\n", T1),
    "AS9": (AST, '        if e and (e.get("embedder") != emb.get("embedder") or int(e.get("dim") or -1) != '
                 'int(emb.get("dim") or -2)):\n', "        if False:\n", T1),
    "AS10": (AST, "        if b is None or not b.committed:\n", "        if b is None:\n", T1),
    "AS11": (AST, '            if not line.endswith(b"\\n"):\n                break\n',
             '            if False:\n                break\n', T1),
    "AS12": (AST, "        if rp == rr or rr in rp.parents:\n", "        if False:\n", T1),
    "AS13": (AST, "    sindex = StreamIndex(root, need)\n", "    sindex = StreamIndex(root, [])\n", T1),
    "AS14": (AST, '        if not info or info.get("crops_sha256") != crops.sha:\n', "        if not info:\n", T1),
    "AS15": (AST, "                S_NO_CROPS: AU.IMAGE_NOT_IN_CROPS,", "                S_NO_CROPS: AU.NO_CROP_ROW,", T1),
    "AS16": (AST, '            return -1, (why if why in ("small", "no_size") else S_UNMAPPED if why == S_UNMAPPED else '
                  'S_NO_CROP_ROW), None\n', "            return -1, S_NO_CROP_ROW, None\n", T1),
    "AS17": (AST, '        refuse = mine and mine[0]["status"] == AU.IN_TRUSTED or in_trusted_stream(q, trusted)\n',
             "        refuse = in_trusted_stream(q, trusted)\n", T1),
    "AS18": (DSM, '    short = [s for s in steps if stream and expected is not None and per[s]["species_boxes"] > 0\n',
             '    short = [s for s in steps if False and per[s]["species_boxes"] > 0\n', T2),
    "AS19": (DSM, "        species_boxes = _species_boxes(a, boxes)\n", "        species_boxes = boxes\n", T2),
    "AS20": (DSM, '(["OP_CARD"] if unjudged or short else [])', '(["OP_CARD"] if unjudged else [])', T2),
    "AS21": (DSM, '    stream = STREAM_LOOKUP in lookup.split("+")\n', "    stream = False\n", T2),
    "AS22": (DSM, "    predates = (not stream) and any(", "    predates = False and any(", T2),
    "AS23": (JOB, 'python -u -m weed_optimizer_framework.tools.inc2.audit_stream "$@"',
             'python -u -m weed_optimizer_framework.tools.inc.audit "$@"', T1),
    "AS24": (DEP, "  run_inc_audit.sh\n", "\n", T1),
    "AS25": (AST, "    sindex = StreamIndex(root, need)\n", "    sindex = StreamIndex(root, [])\n", T3),
    "AS26": (AST, "            k = int(keep[j])\n", "            k = j\n", T3),
    "AS27": (DSM, "        species_boxes = _species_boxes(a, boxes)\n", "        species_boxes = boxes\n", T3),
}


def copy_tree(dst):
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    code = dst / "weed_llm_benchmark"
    shutil.copytree(str(ROOT / "weed_optimizer_framework"), str(code / "weed_optimizer_framework"), ignore=ign,
                    symlinks=True)
    shutil.copytree(str(TESTS), str(code / "tests"), ignore=ign)
    shutil.copytree(str(ROOT / "deploy"), str(code / "deploy"), ignore=ign)
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
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="audit_stream_mutants_"))
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
