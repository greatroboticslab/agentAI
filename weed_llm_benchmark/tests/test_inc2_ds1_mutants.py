#!/usr/bin/env python3
"""DS1's mutation run (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29; docs/prereg/DS1_2026-10-29.md): every mutant of
its guards must be killed.

The harness copies the package, the tests, the job scripts and deploy/ into a temporary repository (docs/ linked,
read only), checks that the unmutated copy passes each target, then for each mutant replaces exactly one text (which
must occur exactly once in its file), runs its target against the copy and requires it to fail.

  id    file              what the mutant does
  DM1   ds1_data          thin(10) keeps every frame
  DM2   ds1_data          thin(10) ignores the frames S0 already holds
  DM3   ds1_data          the 35 % family cap is lifted
  DM4   ds1_data          the dHash guard searches one direction only (rows against the exams)
  DM5   ds1_data          the dHash guard's 6 bits become 0 (exact copies only)
  DM6   ds1_data          a capture relation with an MD or MT-W row is not a hit
  DM7   ds1_data          the DINOv2 detector fails open on a file it cannot describe
  DM8   ds1_data          an EXIF-rotated image (orientation 2-8) is kept
  DM9   ds1_data          an S arm may hold rows of tsw22, tsw23 or session 20220129 (not season-free)
  DM10  ds1_data          an exam's freeze record is written over (not write-once)
  DM11  ds1_data          MT-W v2 is frozen without the person's check
  DM12  ds1_data          the new rows are not guarded against MD v1 and MT-W v2
  DM13  ds1_data          B'o keeps the rows sharing a capture relation with MD or MT-W
  DM14  ds1_data          a source enters a manifest before the owner's audit
  DM15  ds1_data          a held row (F9, join conflict, H6 scan) is eligible
  DM16  ds1_data          a row of a main-test list is eligible
  DM17  ds1_data          a row a stream version consumed is eligible
  DM18  ds1_data          the K arms are built with unequal N_K
  DM19  ds1_data          the cwd12 floor's repeats are 1 whatever the arm's size
  DM20  step1_stream      the cutter's never-cut list does not read DS1's never-train list
  DM21  step1_stream      a never-train list that does not parse is skipped (fail open)
  DC1   ds1               a phase verdict is written over (not write-once)
  DC2   ds1               a sealed read is taken out of DS.7.5's order
  DC3   ds1               a sealed read is taken twice
  DC4   ds1               more than four sealed reads are taken
  DC5   ds1               a sealed exam is scored without score-test's token
  DC6   ds1               the phase reserve's line is not applied (inc2.ds1.cap_check)
  DC7   ds1               the time-stamp record need not name the prereg's sha256
  DC8   ds1               a proof that does not stamp the prereg is taken
  DH1   ds1_sheets        the delegate may decide the owner's audit sheets
  DH2   ds1_sheets        a reversal does not widen the owner's audit
  DH3   ds1_sheets        a names-round source is not drawn for the owner's audit
  DL1   ds1_long          the pair's compute stays at 3.6 M whatever S0x's smoke projects
  DL2   ds1_long          S2x's 36 h re-submission is allowed past 28.8 h
  DL3   ds1_long          a seed trains without a complete smoke
  DL4   run_inc2_long     the job's module list without inc2/ds1_long.py
  DP1   diagnose_stream   the gate opens without the deployed cutter's change
  DP2   diagnose_stream   the gate opens without the time-stamp record
  DP3   stream            the phase reserve's line is not applied by the ticker
  DP4   stream            S2 and S1c-R are not cut first while the long pair's estimate is left
  DP5   stream            the family cap of 260 GPU-h is not applied by the ticker
  DP6   diagnose_stream   a byte-adding item past phase 1's 60 GB keeps only V3F's floor
  DP7   diagnose_stream   an item short of storage is proposed (not a wait)
  DP8   diagnose_stream   training is proposed before RA is recorded
  DP9   diagnose_stream   S1's builds start before MT-W v2 is frozen
  DP10  stream            a failed DS1 item is recorded failed at once (never proposed again)
  DP11  diagnose_stream   the date substitution never applies
  DP12  diagnose_stream   a verdict is proposed while its arms are pending
  DP13  evidence          INC_DIR/ds1 is not a reserved directory
  DP14  weed.json         the gate looks for another marker in the deployed cutter
  DP15  stream_remote     DS1's status reads the default INC_DIR, not the snapshot's
  DP16  stream            a DS1 card waiting for a person is never raised
  DP17  deploy_funnel     the amendment's file is not shipped

Run:  python3 tests/test_inc2_ds1_mutants.py [DM1 DP5 ...]
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
DD = "weed_optimizer_framework/tools/inc2/ds1_data.py"
DC = "weed_optimizer_framework/tools/inc2/ds1.py"
DH = "weed_optimizer_framework/tools/inc2/ds1_sheets.py"
DL = "weed_optimizer_framework/tools/inc2/ds1_long.py"
S1 = "weed_optimizer_framework/tools/inc2/step1_stream.py"
ST = "weed_optimizer_framework/tools/inc_autopilot/stream.py"
DSM = "weed_optimizer_framework/tools/inc_autopilot/diagnose_stream.py"
SRM = "weed_optimizer_framework/tools/inc_autopilot/stream_remote.py"
EV = "weed_optimizer_framework/tools/inc_autopilot/evidence.py"
WJ = "weed_optimizer_framework/tools/inc_autopilot/stream_domains/weed.json"
DF = "deploy/deploy_funnel.sh"
LONGSH = "run_inc2_long.sh"
DATA = "tests/test_inc2_ds1_data.py"
CORE = "tests/test_inc2_ds1.py?DS1_ONLY=units"
SHEETS = "tests/test_inc2_ds1_sheets.py"
LONG = "tests/test_inc2_ds1_long.py"
AP_STATIC = "tests/test_stream_ap_ds1.py?DS1_WORLD=static,plan"
AP_GATE = "tests/test_stream_ap_ds1.py?DS1_WORLD=gate"
AP_PATH = "tests/test_stream_ap_ds1.py?DS1_WORLD=path"
AP_STORAGE = "tests/test_stream_ap_ds1.py?DS1_WORLD=storage"
AP_CAP = "tests/test_stream_ap_ds1.py?DS1_WORLD=cap"
AP_FAIL = "tests/test_stream_ap_ds1.py?DS1_WORLD=fail"
DEPLOY = "tests/test_inc2_ds1_deploy.py"
MUTANTS = {
    "DM1": (DD, "        if d > bits:\n            kept.append(k)\n", "        if True:\n            kept.append(k)\n",
            DATA),
    "DM2": (DD, "    kept_h = [int(h) for h in seeded_hashes if h is not None]\n", "    kept_h = []\n", DATA),
    "DM3": (DD, "    allowed = int(math.floor(cap / (1.0 - cap) * other_count)) - int(base_count)\n",
            "    allowed = len(new_rows)\n", DATA),
    "DM4": (DD, '        T2, Q2, D2_, _w = B3.near_pairs_between(ex["H"], ex["ok"], HA, int(bits))\n',
            "        T2, Q2, D2_ = T1[:0], Q1[:0], D1_[:0]\n", DATA),
    "DM5": (DD, "def dhash_hits(rows, exams, bits=GUARD_BITS):\n", "def dhash_hits(rows, exams, bits=0):\n", DATA),
    "DM6": (DD, "            if k in caps:\n                ex, g = caps[k]\n",
            "            if False:\n                ex, g = caps[k]\n", DATA),
    "DM7": (DD, "    if unscannable:\n        raise DataRefused(", "    if False:\n        raise DataRefused(", DATA),
    "DM8": (DD, '    return None if ori in EXIF_OK else "exif_orientation"\n', "    return None\n", DATA),
    "DM9": (DD, '    if arm.startswith("S"):\n        bad = season_rows(rows)\n',
            '    if False:\n        bad = season_rows(rows)\n', DATA),
    "DM10": (DD, '    write_once(exam_dir(exam) / "frozen.json", doc)\n',
             '    D.write_json(exam_dir(exam) / "frozen.json", doc)\n', DATA),
    "DM11": (DD, '        if write and chk and chk.get("verdict") == "pass" and not frozen(mtw)',
             "        if write and not frozen(mtw)", DATA),
    "DM12": (DD, '    return [r for r in rows if not hits.get(r["key"])]\n', "    return list(rows)\n", DATA),
    "DM13": (DD, '        if gh.get(r["key"]):\n            out["capture_md_mtw"] += 1\n',
             '        if False:\n            out["capture_md_mtw"] += 1\n', DATA),
    "DM14": (DD, '    if d.get("audit_required") and not d.get("audited"):\n        return False, "owner audit pending"\n',
             '    if False:\n        return False, "owner audit pending"\n', DATA),
    "DM15": (DD, '    if held:\n        return "held:%s" % held[0]\n', '    if False:\n        return "held:%s" % held[0]\n',
             DATA),
    "DM16": (DD, '    if row.get("test_v1"):\n        return "test_v1"\n', '    if False:\n        return "test_v1"\n', DATA),
    "DM17": (DD, '    if row["key"] in ctx.get("consumed", ()):\n', "    if False:\n", DATA),
    "DM18": (DD, '        short = {a: v for a, v in kpools["supplies"].items() if a in ("KD", "KT") and v < nk}\n',
             "        short = {}\n", DATA),
    "DM19": (DD, "    return max(1, int(B3.round_half_up((float(n_rows) - c) / (float(n_s0) - c))))\n",
             "    return 1\n", DATA),
    "DM20": (S1, "    ds1 = ds1_never_rows(layout.inc_dir)\n", "    ds1 = []\n", DATA),
    "DM21": (S1, '                raise StreamError("%s line %d does not parse (%s): its rows\' status is unknown" % '
                 '(p, i, e))\n', "                continue\n", DATA),
    "DC1": (DC, '        write_once(capacity("ds1_v%d.json" % int(phase)), json.loads(json.dumps(doc, default=str)))\n',
            '        write_json(capacity("ds1_v%d.json" % int(phase)), json.loads(json.dumps(doc, default=str)))\n', CORE),
    "DC2": (DC, "    if before:\n        raise Ds1Refused(\"the reads are taken in DS.7.5's order",
            "    if False:\n        raise Ds1Refused(\"the reads are taken in DS.7.5's order", CORE),
    "DC3": (DC, "    if (arm, exam) in done:\n", "    if False:\n", CORE),
    "DC4": (DC, '    if sum(1 for v in done.values() if v == "read") >= READ_CAP:\n', "    if False:\n", CORE),
    "DC5": (DC, "        if not any(test_token is t for t in _TEST_TOKENS):\n", "        if False:\n", CORE),
    "DC6": (DC, "    if int(phase) == 1 and not v2_done and not substituted and after > PHASE1_CEILING_GPUH + CAP_EPS:\n",
            "    if False:\n", CORE),
    "DC7": (DC, '    return bool(isinstance(d, dict) and d.get("format") == FMT_TIMESTAMP and d.get("sha256") == '
                'PREREG_SHA256\n',
            '    return bool(isinstance(d, dict) and d.get("format") == FMT_TIMESTAMP\n', CORE),
    "DC8": (DC, "    if ots_digest(proof) != PREREG_SHA256:\n", "    if False:\n", CORE),
    "DH1": (DH, '    if kind == "audit" and decided_by == "delegate":\n', "    if False:\n", SHEETS),
    "DH2": (DH, "    reversed_any = any(_reversed(v) for v in out.values())\n", "    reversed_any = False\n", SHEETS),
    "DH3": (DH, '        v["audit_required"] = bool(reversed_any or s in sample or s in names_round or s == _siu())\n',
            '        v["audit_required"] = bool(reversed_any or s in sample or s == _siu())\n', SHEETS),
    "DL1": (DL, "    return D.PAIR_IMAGE_EPOCHS if float(s0x_projection_3600) <= LINE_H + 1e-9 else "
                "D.PAIR_FALLBACK_IMAGE_EPOCHS\n", "    return D.PAIR_IMAGE_EPOCHS\n", LONG),
    "DL2": (DL, '    if arm in ("S2X", "S1UX") and h <= RESUBMIT_LINE_H + 1e-9:\n', '    if arm in ("S2X", "S1UX"):\n',
            LONG),
    "DL3": (DL, '    if not isinstance(sm, dict) or sm.get("status") != "complete":\n        raise LongRefused("%s has no '
                'complete smoke', '    if False:\n        raise LongRefused("%s has no complete smoke', LONG),
    "DL4": (LONGSH, "tools/inc2/ds1.py tools/inc2/ds1_data.py tools/inc2/ds1_long.py\n",
            "tools/inc2/ds1.py tools/inc2/ds1_data.py\n", LONG),
    "DP1": (DSM, '    if not fs.get("never_cut"):\n        return False, "the deployed cutter',
            '    if False:\n        return False, "the deployed cutter', AP_GATE),
    "DP2": (DSM, '    if not (fs.get("status") or {}).get("timestamp"):\n        return False, "ds1/v1/timestamp.json',
            '    if False:\n        return False, "ds1/v1/timestamp.json', AP_GATE),
    "DP3": (ST, "        elif phase == 1 and not v2 and not sub and after > line + V3FAM_CAP_EPS:\n",
            "        elif False:\n", AP_CAP),
    "DP4": (ST, '        elif key in ("build:S2", "build:S1CR"):\n', "        elif False:\n", AP_CAP),
    "DP5": (ST, '        if after > cap + V3FAM_CAP_EPS:\n            why = "would take DS1 from %.2f past its cap',
            '        if False:\n            why = "would take DS1 from %.2f past its cap', AP_CAP),
    "DP6": (DSM, "    need = v3f if (within or gb <= 0) else max(v3f, data)\n", "    need = v3f\n", AP_STORAGE),
    "DP7": (DSM, '                    if short:\n                        dwait[it["item"]] = "storage: %s" % short\n',
            '                    if False:\n                        dwait[it["item"]] = "storage: %s" % short\n',
            AP_STORAGE),
    "DP8": (DSM, '    train_ok = bool(st.get("ra")) and bool(st.get("power_final")) and bool(md.get("frozen"))\n',
            '    train_ok = bool(st.get("power_final")) and bool(md.get("frozen"))\n', AP_STATIC),
    "DP9": (DSM, '        if arm in ("S1C", "S1U", "S1CR") and blocked_mtw:\n', "        if False:\n", AP_PATH),
    "DP10": (ST, '        last = walltime or rec["fails"] >= int(cfg.get("max_failures") or 2)\n', "        last = True\n",
             AP_FAIL),
    "DP11": (DSM, '    sub = (not arms.get("S2")) and str(fs.get("today") or "") > str(dates.get("s2_freeze_by") or '
                  '"9999")\n', "    sub = False\n", AP_STATIC),
    "DP12": (DSM, '        if not pending:\n            add(maint, {"lever": "L34"',
             '        if True:\n            add(maint, {"lever": "L34"', AP_STATIC),
    "DP13": (EV, '"stream", "step1_stream", "intake", "twostage", "v3fam", "ds1")',
             '"stream", "step1_stream", "intake", "twostage", "v3fam")', AP_STATIC),
    "DP14": (WJ, '"marker": "DS1_NEVER_TRAIN",', '"marker": "DS1_NEVER_TRAIN_LIST",', AP_GATE),
    "DP15": (SRM, "    CM.INC_DIR = Path(inc)\n", "    pass\n", AP_GATE),
    "DP16": (ST, '        if steps:\n            self._ds1_card_once("steps"', '        if False:\n            '
                 'self._ds1_card_once("steps"', AP_PATH),
    "DP17": (DF, "            docs/prereg/DS1_2026-10-29.md docs/prereg/DS1_2026-10-29.md.ots",
             "            docs/prereg/DS1_2026-10-29.md.ots", DEPLOY),
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
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ds1_mut_"))
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
