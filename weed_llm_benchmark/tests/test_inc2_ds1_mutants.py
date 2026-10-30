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
  Revision 1 (2026-10-29) and the fixes made with it:
  DR1   ds1_data          the draw ignores the groups near base B (MT-W v2 not kept apart, MT.2 item 2)
  DR2   ds1_data          a near copy of base B's rows is not looked for at the draw
  DR3   ds1_data          a source with cwd12 capture names is ready when accepted (role C not enforced)
  DR4   ds1_data          the intake's and Step 1's copy refusals never quarantine a source
  DR5   ds1_data          a dHash hit the DINOv2 detector does not confirm quarantines its source
  DR6   ds1_data          a rejected source's pool still enters the manifests
  DR7   ds1_data          MT-W v1.1 only when a label was corrected (a failing source stays in the headline)
  DR8   ds1_data          a screen with no new row writes no revision (the plan proposes it for ever)
  DR9   ds1_data          phase 2 reads every source, not only the named ones
  DR10  ds1_data          traced source classes are ignored (every box keeps its INC class)
  DR11  ds1_data          the never-train list keeps no file (DS1's DINOv2 index misses MD and MT-W)
  DR12  ds1_data          a re-read whose classes do not match the intake label is traced anyway
  DR13  ds1_data          the DINOv2 copies of a built manifest's rows are not left out of the draw
  DR14  ds1_data          a group near a built manifest's row is drawn into the exams
  DR15  ds1_data          an image is described again for every index a job scans it against
  DH4   ds1_sheets        the owner's sample is recomputed on the latest screen only (a drawn source drops out)
  DH5   ds1_sheets        any person's id records an audit sheet
  DH6   ds1_sheets        an LN image not marked checked is recorded
  DH7   ds1_sheets        an MT-W image not answered is recorded
  DH8   ds1_sheets        an MT-W correction keeps only the boxes drawn
  DH9   ds1_sheets        a lab render lists its batch before pulling the cluster's sheets state
  DH10  ds1_sheets        the owner's audit sheet pre-selects the delegate's decision
  DC9   ds1               a sealed read before verdict v2 (the challenger out of DS.7.5's order)
  DC10  ds1               S2x not run (walltime) does not apply the substitution
  DC11  ds1               a verdict is written while an arm of its phase is missing
  DC12  ds1               the 0.90 interval takes the image SE only
  DC13  ds1               a sealed read re-submitted after a partial failure scores its parts again
  DC14  ds1               a rejected source's MD-N is still scored
  DC15  ds1               a time-stamp record without the Revision opens the gate
  DL5   ds1_long          the long path removes its train logs with its data
  DL6   ds1_long          an earlier attempt's work is deleted, not moved aside
  DP18  stream            a DS1 build's blocked runs pause the stream (not a measurement arm)
  DP19  stream            a DS1 build whose runs are blocked stays running (the verdict waits for ever)
  DP20  stream            phase 1's bytes charged on every run of a data step
  DP21  diagnose_stream   SIU is waited on for an exam draw it never gets
  DP22  diagnose_stream   KD and KT hold verdict v1 when the K source is out
  DP23  diagnose_stream   S2x not run (walltime) does not substitute S1ux
  DP24  diagnose_stream   a role C or quarantined source waits for a decision
  DP25  diagnose_stream   the audit batch names no sources (the lab cannot render it)
  DP26  deploy_funnel     the deploy's pre-flight does not check what ships for DS1
  Revision 2 (2026-10-29, before any DS1 training or score):
  DX1   ds1               the weed check compares the key-order arrays with the validator-order AP (the refusal of
                          POWER's pre-flight on tied confidences)
  DX2   ds1               the weed check takes an image's rows without comparing them with the helper's record
  DX3   ds1               the weed check does not compare the ground-truth counts
  DX4   ds1               the weed reading is not tie-broken (it depends on the machine's sort)
  DX5   stream            a finished audit render records the tick's context's set (none: rendered on every tick)
  DX6   diagnose_stream   a set the cluster lists rendered is rendered again (the ticker's memory alone)
  DX7   ds1_sheets        a decisions file naming a superseded sheet is recorded
  DX8   ds1_sheets        a superseded audit batch still counts for its set
  DX9   ds1_data          the screen's audit reads [class, cx, cy, w] as the box
  DX10  ds1_data          audit_rates accepts a box that is not (cx, cy, w, h)
  DX11  ds1_data          screen_records ignores the audit taken again
  DX12  ds1_data          a pool applies masks from an audit taken before Revision 2
  DX13  ds1_data          an 'accept' decision on a partially labelled source enters a pool
  DX14  ds1_data          pools reads a screen's audit without taking it again first
  DX15  ds1               RA's default reports take any *_s<3 digits> folder (b_v2_s640, the pre-fork segments)
  DX16  ds1               a Revision's own time-stamp record naming another sha256 opens the gate
  DX17  ds1               timestamp-revision takes proofs it has not checked
  DX18  ds1_sheets        a sheet rendered after the audit was taken again shows the screen's audit
  DX19  stream            an item attempted after the Revision is reopened by it again
  DX20  stream            an item is reopened before the deployed tree holds the Revision's file
  DX21  stream            the time-stamp card asked for once is never raised again for a later Revision
  DX22  diagnose_stream   an 'accept' on a partially labelled source waits for a pool, not for its sheet
  DX23  ds1_data          the audit is taken again with another predictor than the screen's (E1-A seed 0)
  DX24  ds1_data          a source whose live rows are no longer the screen's is recorded as the screen's
  DX25  stream            the time-stamp card names every Revision, also one the record names (refused)
  DX26  ds1               timestamp_missing lists a Revision its own record names
  DS1 deviation 1 (2026-10-30, after the power pre-flight's b1_ood23 refusals of jobs 50026958 and 50027968):
  DV1   ds1               B'1's P7 constants are the plain reading's again (the text's numbers: the live refusal)
  DV2   ds1               only P7's constant is checked against V3F's record (a wrong plain constant passes)
  DV3   ds1               the refusal does not say a constant is the other reading's value
  DV4   ds1               V3F's records are not checked against the registered constants before the passes
  DV5   ds1               the pass's plain reading is not held to DS.7.1's rule (only P7's difference counts)
  DV6   ds1               the pass's P7 reading is taken from its OtherPlant arrays, not its target ones
  DV7   ds1               V3F's recorded P7 statistic is read from the OtherPlant arrays of its p7 sidecar
  DV8   ds1               the superseded records stay at their paths (require_reproductions refuses every score)
  DV9   ds1               a file at a superseded path with another sha256 is moved aside too
  DV10  ds1               a superseded record is deleted, not moved (no copy kept)
  DV11  ds1               the pre-flight takes its reproductions without moving the superseded records aside
  DV12  stream            the gate opens while the deployed tree lacks the deviation's file
  DV13  weed.json         the reopen block's time precedes the live attempt (the pre-flight is never reopened)
  DV14  deploy_funnel     the deviation's file is not shipped

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
REV2 = "tests/test_inc2_ds1_rev2.py"
REV1 = "tests/test_inc2_ds1_rev1.py"
DEV1 = "tests/test_inc2_ds1_dev1.py"
AP_PLANR1 = "tests/test_stream_ap_ds1.py?DS1_WORLD=plan_rev1"
AP_REVIEW = "tests/test_stream_ap_ds1.py?DS1_WORLD=review"
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
    "DM18": (DD, '    return {a: v for a, v in supplies.items() if a in ("KD", "KT") and v < n_k}\n', "    return {}\n",
             DATA),
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
    "DH1": (DH, '    if kind == "audit" and decided_by != OWNER_HUMAN:\n', '    if kind == "audit" and decided_by not in '
                                                                         '(OWNER_HUMAN, "delegate"):\n', SHEETS),
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
    "DP2": (DSM, '    if not (fs.get("status") or {}).get("timestamp"):\n        return False, ("ds1/v1/timestamp.json',
            '    if False:\n        return False, ("ds1/v1/timestamp.json', AP_GATE),
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
    "DP11": (DSM, '    sub = s2x_wall or ((not arms.get("S2")) and str(fs.get("today") or "") > str(dates.get("s2_freeze_by") '
                  'or "9999"))\n', "    sub = False\n", AP_STATIC),
    "DP12": (DSM, '        if not pending:\n            add(maint, {"lever": "L34"',
             '        if True:\n            add(maint, {"lever": "L34"', AP_STATIC),
    "DP13": (EV, '"stream", "step1_stream", "intake", "twostage", "v3fam", "ds1")',
             '"stream", "step1_stream", "intake", "twostage", "v3fam")', AP_STATIC),
    "DP14": (WJ, '"marker": "DS1_NEVER_TRAIN",', '"marker": "DS1_NEVER_TRAIN_LIST",', AP_GATE),
    "DP15": (SRM, "    CM.INC_DIR = Path(inc)\n", "    pass\n", AP_GATE),
    "DP16": (ST, '        if steps:\n            self._ds1_card_once("steps"', '        if False:\n            '
                 'self._ds1_card_once("steps"', AP_PATH),
    "DP17": (DF, "ROOT_PATHS+=(docs/prereg/DS1_2026-10-29.md docs/prereg/DS1_2026-10-29.md.ots",
             "ROOT_PATHS+=(docs/prereg/DS1_2026-10-29.md.ots", DEPLOY),
    # Revision 1 (2026-10-29) and the fixes made with it
    "DR1": (DD, "                           mtw_excluded_groups=ex_mtw)[s]\n", "                           "
            "mtw_excluded_groups=())[s]\n", DATA),
    "DR2": (DD, '    hit_mtw = set(dhash_hits(rows, _hash_exam("base_b", base))) | {', "    hit_mtw = set() | {", REV1),
    "DR3": (DD, '    if srec.get("role_c") or int((srec.get("capture_names") or {}).get("hits") or 0):\n        return False, '
                '"role C', '    if False:\n        return False, "role C', REV1),
    "DR4": (DD, '        copies = [{"key": k, "where": "ds1_guard"} for k in hits] + refusal_hits.get(s, []) + ih\n',
            '        copies = [{"key": k, "where": "ds1_guard"} for k in hits]\n', REV1),
    "DR5": (DD, "    guard_hits = {k for k, v in eh.items() if v} | (set(dh) if scanner is None else set())\n",
            "    guard_hits = {k for k, v in eh.items() if v} | set(dh)\n", REV1),
    "DR6": (DD, "    return {s: rs for s, rs in pool_rows(phase).items() if source_ready(s, inputs.decisions, "
                "srec.get(s))[0]}\n", "    return pool_rows(phase)\n", REV1),
    "DR7": (DD, "    if corrected or left:\n", "    if corrected:\n", REV1),
    "DR8": (DD, "    if write and (rows or not revs or sorted(seen or []) != sorted(last_seen or [])):\n",
            "    if write and rows:\n", REV1),
    "DR9": (DD, '    rows = [r for r in inputs.candidates if r["key"] not in prior and (int(phase) == 1 or r["source"] in '
                'named)]\n', '    rows = [r for r in inputs.candidates if r["key"] not in prior]\n', REV1),
    "DR10": (DD, '        if r["key"] in traced:\n            r["box_src"] = [str(x) for x in traced[r["key"]]]\n',
             '        if False:\n            r["box_src"] = [str(x) for x in traced[r["key"]]]\n', REV1),
    "DR11": (DD, '                    "image": r.get("train_image") or r.get("image"),\n', "", REV1),
    "DR12": (DD, "        if any(inc_of.get(s) is not None and int(inc_of[s]) != int(b[0]) for s, b in zip(got, "
                 'r["boxes"])):\n', "        if False:\n", REV1),
    "DR13": (DD, "    hit_any |= set(dino)\n", "    hit_any |= set()\n", REV1),
    "DR14": (DD, "    ex = {groups[k] for k in hit_any if k in groups}\n", "    ex = set()\n", REV1),
    "DR15": (DD, "        if k not in memo and k not in seen:\n", "        if k not in seen:\n", REV1),
    "DH4": (DH, "        sample |= sticky_audit_sample(revs) if revs else set(audit_sample(ss))\n",
            "        sample |= set(audit_sample(revs[-1])) if revs else set(audit_sample(ss))\n", REV1),
    "DH5": (DH, "    if kind == \"audit\" and decided_by != OWNER_HUMAN:\n", "    if False:\n", REV1),
    "DH6": (DH, '                if d.get("checked") is not True:\n', "                if False:\n", REV1),
    "DH7": (DH, '                if d.get("verdict") not in MTW_VERDICTS or bool(d.get("error")) != (d.get("verdict") '
                '== "error"):\n', "                if False:\n", REV1),
    "DH8": (DH, '                    row["corrected_boxes"] = [[float(x) for x in b[-4:]] for j, b in enumerate(shown)\n'
                '                                              if j not in removed] + added\n',
            '                    row["corrected_boxes"] = added\n', REV1),
    "DH9": (DH, "    if pull:\n        pull_state(inc, runner)\n", "    if False:\n        pull_state(inc, runner)\n",
            REV1),
    "DH10": (DH, '_radio("dec|" + src, DECISION_LABELS, None))', '_radio("dec|" + src, DECISION_LABELS, '
                                                                 'prev.get("decision")))', REV1),
    "DC9": (DC, "    if not (isinstance(v2, dict) and v2.get(\"status\") == \"decided\"):\n        # no sealed read",
            "    if False:\n        # no sealed read", REV1),
    "DC10": (DC, "    return bool(s2x_walltime) or ((not s2_frozen) and str(on_date) > S2_FREEZE_BY)\n",
             "    return ((not s2_frozen) and str(on_date) > S2_FREEZE_BY)\n", REV1),
    "DC11": (DC, "    if missing:\n        raise Ds1Refused(\"verdict v%d waits for %s", "    if False:\n        raise "
                 "Ds1Refused(\"verdict v%d waits for %s", REV1),
    "DC12": (DC, "        if groups is None:\n            groups = mtw_groups()\n", "        if False:\n            "
                 "groups = mtw_groups()\n", REV1),
    "DC13": (DC, '        return entry(js, "kept") if js.exists() else None\n', "        return None\n", REV1),
    "DC14": (DC, "    gone = DD.sources_out() if d.is_dir() else set()\n", "    gone = set()\n", REV1),
    "DC15": (DC, "                and revisions_ok(d, path or timestamp_path()))\n", "                )\n", CORE),
    "DL5": (DL, '    trash = T.move_aside(work / "data", removed)\n', "    trash = T.move_aside(work, removed)\n", REV1),
    "DL6": (DL, "    os.rename(str(work), str(dst))\n", '    __import__("shutil").rmtree(str(work))\n', REV1),
    "DP18": (ST, '        return out | set((((self.dom or {}).get("ds1") or {}).get("build_exps") or {}).values())\n',
             "        return out\n", AP_REVIEW),
    "DP19": (ST, '                items["build:%s" % arm] = "failed" if exp in blocked else "running"\n',
             '                items["build:%s" % arm] = "running"\n', AP_REVIEW),
    "DP20": (ST, '        if kind and int(p.get("ds1_phase") or 1) == 1 and key not in charged:\n',
             '        if kind and int(p.get("ds1_phase") or 1) == 1:\n', AP_REVIEW),
    "DP21": (DSM, '        if x.get("decision") == "accept" and x.get("exam_draw", True) and not x.get("md_n"):\n',
             '        if x.get("decision") == "accept" and not x.get("md_n"):\n', AP_PLANR1),
    "DP22": (DSM, '                   and not (a in ("KD", "KT") and (st.get("k_not_run") or st.get("k_source_out")))]',
             '                   and not (a in ("KD", "KT") and st.get("k_not_run"))]', AP_PLANR1),
    "DP23": (DSM, '    s2x_wall = str(((long_.get("S2X") or {}).get("smoke")) or "").startswith("not run")\n',
             "    s2x_wall = False\n", AP_PLANR1),
    "DP24": (DSM, '        if x.get("role_c") or x.get("quarantined_whole") or x.get("settled_out"):\n            continue\n',
             "        if False:\n            continue\n", AP_PLANR1),
    "DP25": (DSM, '"item": key, "sources": pend})', '"item": key})', AP_PLANR1),
    "DP26": (DF, "PREFLIGHT+=(tests/test_inc2_ds1_deploy.py)", "PREFLIGHT+=()", DEPLOY),
    # Revision 2 (2026-10-29)
    "DX1": (DC, '    cap = S.collapsed_ap(tp, q, int(v.e3_s1_ngt))',
            '    cap = S.collapsed_ap(weed["tp"], weed["conf"], int(v.e3_s1_ngt))', REV2),
    "DX2": (DC, '        if tp_w.shape != tp_e.shape or not np.array_equal(tp_w, tp_e) or not np.array_equal(q_w, q_e):',
            '        if tp_w.shape != tp_e.shape:', REV2),
    "DX3": (DC, '    if n_gt != int(v.e3_s1_ngt):', '    if False:', REV2),
    "DX4": (DC, '    tb = SC.tie_break(weed)\n', '    tb = weed\n', REV2),
    "DX5": (ST, '            f["audit_render_for"] = ds1_audit_sources((p.get("params") or {}).get("sources"))',
            '            f["audit_render_for"] = []', REV2),
    "DX6": (DSM, '        if due(key) and fs.get("audit_render_for") != pend and pend not in drawn:',
            '        if due(key) and fs.get("audit_render_for") != pend:', REV2),
    "DX7": (DH, '    if sh in sup:\n', '    if False:\n', REV2),
    "DX8": (DH, '            if r.get("kind") == "audit" and "%s_%03d" % (r["kind"], r["batch"]) not in sup}',
            '            if r.get("kind") == "audit"}', REV2),
    "DX9": (DD, '    return audit_verdict(audit_rates([dict(r, boxes=audit_boxes(r)) for r in smp], predict), ref)',
            '    return audit_verdict(audit_rates(smp, predict), ref)', REV2),
    "DX10": (DD, '        if any(len(g) != 4 for g in gt):', '        if False:', REV2),
    "DX11": (DD, '            if s in ra:\n', '            if False:\n', REV2),
    "DX12": (DD, '        if a.get("partial") and not audit_valid(a):', '        if False:', REV2),
    "DX13": (DD, '    if d.get("decision") == "accept" and audit_valid(a) and a.get("partial"):', '    if False:', REV2),
    "DX14": (DD, '    if audit_due(phase):\n        # a screen', '    if False:\n        # a screen', REV2),
    "DX15": (DC, 'RA_SEGMENT_RE = re.compile(r"weed_stream_v1_fork1364_s[0-9]{3}\\Z")',
             'RA_SEGMENT_RE = re.compile(r".*_s[0-9]{3}\\Z")', REV2),
    "DX16": (DC, '    return bool(isinstance(sep, dict) and sep.get("format") == FMT_REV_TIMESTAMP and sep.get("sha256") == '
                 'rv["sha256"]\n', '    return bool(isinstance(sep, dict) and sep.get("format") == FMT_REV_TIMESTAMP and True\n',
             REV2),
    "DX17": (DC, '           "proofs": _proof_pair(proof, stamp_proof, rv["sha256"], rv["commit"], rv["file"]), '
                 '"written_utc": _utc()}', '           "proofs": [{"sha256": "x"}, {"sha256": "y"}], "written_utc": _utc()}',
             REV2),
    "DX18": (DH, '            if ra:\n', '            if False:\n', REV2),
    "DX19": (ST, '                        str(rec["started_utc"]) >= str(ro["before_utc"]):',
             '                        False:', REV2),
    "DX20": (ST, 'or not self._ds1_tree_file(ro.get("file") or "",',
             'or not (lambda *a, **k: True)(ro.get("file") or "",', REV2),
    "DX21": (ST, 'self._ds1_card_once("timestamp", "revisions:%s" % ",".join(',
             'self._ds1_card_once("timestamp", "missing" or ",".join(', REV2),
    "DX22": (DSM, '        if DS1_DECIDE_AGAIN in str(x.get("why") or ""):', '        if False:', REV2),
    # Revision 2 as registered again before its deploy (the review's points)
    "DX23": (DD, '    if (wsha is not None or inputs.production) and wsha != AUDIT_PREDICTOR_SHA256:', '    if False:', REV2),
    "DX24": (DD, '"rows_as_screened": len(rs) == int(v.get("live") or 0),', '"rows_as_screened": True,', REV2),
    "DX25": (ST, '                    int(r["n"]) in [int(x) for x in tm.get("revisions") or []]]',
             '                    True]', REV2),
    "DX26": (DC, '    return {"record": rec, "revisions": [int(rv["n"]) for rv in REVISIONS if not revision_named(rv, d, p)]}',
             '    return {"record": rec, "revisions": [int(rv["n"]) for rv in REVISIONS]}', REV2),
    # DS1 deviation 1 (2026-10-30)
    "DV1": (DC, 'B1_OOD23 = {"p7": (0.1496, 0.1594, 0.1634), "plain": (0.2562, 0.2517, 0.2598)}',
            'B1_OOD23 = {"p7": (0.2562, 0.2517, 0.2598), "plain": (0.2562, 0.2517, 0.2598)}', DEV1),
    "DV2": (DC, '        for name, other in (("p7", "plain"), ("plain", "p7")):',
            '        for name, other in (("p7", "plain"),):', DEV1),
    "DV3": (DC, '" (it is the %s reading\'s value)" % other\n                    if round(float(rec[other]), 4) == reg else ""',
            '""', DEV1),
    "DV4": (DC, '        b1_check_constants(recorded)\n', '', DEV1),
    "DV5": (DC, '"diff": max(r["diff"] for r in rd.values())', '"diff": rd["p7"]["diff"]', DEV1),
    "DV6": (DC, 'got = {"p7": F.statistic_value(ps["p7"]["t"], "agnostic"),',
            'got = {"p7": F.statistic_value(ps["p7"]["o"], "agnostic"),', DEV1),
    "DV7": (DC, 'if k.startswith("t_")}, "agnostic")', 'if k.startswith("o_")}, "agnostic")', DEV1),
    "DV8": (DC, '        os.unlink(str(src))\n', '        pass\n', DEV1),
    "DV9": (DC, '        if got != e["sha256"]:', '        if False:', DEV1),
    "DV10": (DC, '            os.link(str(src), str(dst))', '            pass', DEV1),
    "DV11": (DC, '    supersede()\n', '', DEV1),
    "DV12": (ST, '                       for r in list(cfg.get("revisions") or []) + list(cfg.get("deviations") or [])),',
             '                       for r in list(cfg.get("revisions") or [])),', DEV1),
    "DV13": (WJ, '"before_utc": "DEV1_STAMP_UTC",', '"before_utc": "2026-10-30T03:00:00Z",', DEV1),
    "DV14": (DF, "ROOT_PATHS+=(docs/prereg/DS1_dev1_2026-10-30.md docs/prereg/DS1_dev1_2026-10-30.md.ots",
             "ROOT_PATHS+=(docs/prereg/DS1_dev1_2026-10-30.md.ots", DEPLOY),
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
