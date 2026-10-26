#!/usr/bin/env python3
"""V3F deviation 5 (2026-10-24; docs/prereg/V3F_dev5_2026-10-24.md): inc2/v3fam.py back to the bytes every
reproduction record of 2026-10-12 pins, the verdict's CLI moved out of it, the records a recorded deviation
supersedes moved aside byte for byte, the six arms and the verdict reopened, the second continuation verdict, a
refusal of the platform's own making never counted as an arm's failure, the deploy-time guards, and the ticker's
D28 bookkeeping. Fast, record-level checks on the cluster's own records (tests/fixtures/v3fam_dev5, byte for byte;
no scoring pass, no ticker run), so tests/test_v3fam_dev5_mutants.py runs this file against each of its mutants; the
ticker end to end is tests/test_stream_ap_v3fam_dev5.py, the verb end to end tests/test_v3fam_dev5_verdict.py.

- the code: inc2/v3fam.py hashes 3d776dae... (git b6c4aef); inc2.v3fam.code_shas() equals the code map of each of
  the seven records of 2026-10-12 (E3RS job 49063561), and require_equivalence passes on them; identity_probe.json
  (2026-10-21, D5B job 49570114) differs in v3fam only; the domain's v3fam.equivalence_code is that map; the
  snapshot hashes the same files (stream_remote.V3FAM_CODE_FILES) without importing them;
- which records each of the six arms finds once deviation 5 is deployed (present and passing, or taken in its own
  job);
- the CLI: inc2/v3fam.py has no --ticker-failed; inc2.v3fam_verdict verdict takes --cap-refused and
  --ticker-failed (a refusal exits 2), through L23L's argv, the executor, the policy table, the cluster's grammar,
  run_inc2_build.sh and the ticker's rerun command;
- the records superseded (inc2.v3fam_supersede): identity_probe.json and the six failure records of 2026-10-22
  moved aside byte for byte into a dated folder with a manifest (sha256, job, why, the deviation's sha256); a second
  run changes nothing; a move cut off after its link is completed; another file at the path is left; a destination
  holding other bytes refuses; the failure records are never read as failures, even in place;
- the verb's guard on the world of 2026-10-22 (the cluster's verdicts, failure records and D6's smoke): the reads
  follow the second continuation; it is refused while any of the six arms has no record since the reopening, naming
  them and deviation 5, writing nothing; D6 is 'not run (walltime)' and BCLEAN 'not needed'; ready once each arm has
  a record or a failure the ticker gave after the reopening;
- the second continuation's check on the cluster's own verdicts: P1 (+0.0757, z 7.7723, family-confirmed) and P4
  reproduce, the 19 decided comparisons of each record listed, both records named by sha256; a doctored one refuses;
  its records are written once, an attempt's partial ones moved aside first;
- the reopening on the ticker's live items of 2026-10-24: the six arms and the verdict once, their spend kept, their
  ids retired; deviation 4's block never again; then L23J D8 first and the verdict waiting;
- a refusal of the platform's own making: no fails counted, never recorded failed or passed as --ticker-failed, its
  GPU-h kept, held with one card naming the remedy, released once the records are current, held for a person after
  three; an arm's own refusal still counts; the refusal read from the job's provenance;
- the pre-score check: no L23J while the snapshot reports a record taken by other code or not in place;
- the deploy guards: a changed inc2/v3fam.py, a pin not matching its file, refused (stream_remote.v3fam_deploy_check;
  step 2's v3fam_records_guard while the cluster holds records), the allow flag only with a recorded deviation;
- D28: the stream ledger's quarantines followed by the ticker, an admit never overwriting a quarantine, no L24 for a
  source already quarantined.

Run:  python3 tests/test_v3fam_dev5.py
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_v3fam_verdict as VT  # noqa: E402  (sets INC_DIR / REPO in a temporary directory first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_verdict as VR  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as X  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as S  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream_remote as SR  # noqa: E402

FAILURES = VT.FAILURES
check = VT.check
refused = VT.refused
DOM = LS.load_domain("weed")
FAM = DOM["v3fam"]
TESTS = pathlib.Path(__file__).resolve().parent
PKG = TESTS.parent                                   # weed_llm_benchmark/
GIT = PKG.parent
FIX = TESTS / "fixtures" / "v3fam_dev5"
B6C4AEF = "3d776daed91d487d2817ba6959f9dd832d82c6e0be048b812916b63f22b239b7"
DEV4_V3FAM = "c5e923dcad8f576cdb9761a0433d5c85241def4a87a0a0081063aa57a50e5e78"
SEVEN = ("dev_identity.json", "injection.json", "two_stage_M.json", "two_stage_A.json", "two_stage_B.json",
         "identity_imageweeds.json", "identity_ooddev_v1.json")
SIX = ("D8", "D4F", "D4N", "D4R15", "D1", "D5B")
DEV5_FILE = "docs/prereg/V3F_dev5_2026-10-24.md"
DEV5_MARKER = "deviation 5 (2026-10-24)"
SMOKE_1021 = "1d0691a852f1c4e3345e678e05cdade007b9d5c7de31b8fd52707f4ab597c2b8"   # D6's smoke of 2026-10-21 (job 49542653)
DEV5_ITEMS = ["score:D8", "score:D4F", "score:D4N", "score:D4R15", "score:D1", "score:D5B", "detr:smoke", "verdict"]
# the records each arm's score job rests on (v3fam_systems.score_d8, v3fam_trained.score_d4/score_d1/score_d5b:
# inc2.v3fam.equivalence_names per exam; 'made' are taken in the job by require_or_make_identity when absent)
ARM_RECORDS = {
    "D8": {"present": ("dev_identity.json", "injection.json", "identity_imageweeds.json", "identity_ooddev_v1.json"),
           "made": ()},
    "D4F": {"present": ("dev_identity.json", "injection.json", "two_stage_B.json", "identity_imageweeds.json",
                        "identity_ooddev_v1.json"), "made": ()},
    "D4N": {"present": ("dev_identity.json", "injection.json", "two_stage_B.json", "identity_imageweeds.json",
                        "identity_ooddev_v1.json"), "made": ()},
    "D4R15": {"present": ("dev_identity.json", "injection.json", "two_stage_B.json", "identity_imageweeds.json",
                          "identity_ooddev_v1.json"), "made": ()},
    "D1": {"present": ("dev_identity.json", "injection.json", "identity_ooddev_v1.json"),
           "made": ("identity_ood22.json", "identity_ood23.json")},
    "D5B": {"present": ("dev_identity.json", "injection.json"), "made": ("identity_probe.json",)},
}
REFUSAL = ('  File "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/weed_optimizer_framework/tools/inc2/'
           'v3fam.py", line 1289, in require_equivalence\n    raise V3famRefused("the code changed since %s ran (%s): '
           'it is moved aside and rerun, as a deviation"\nweed_optimizer_framework.tools.inc2.v3fam.V3famRefused: the '
           'code changed since dev_identity.json ran (v3fam): it is moved aside and rerun, as a deviation')


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load(rel):
    return json.loads((FIX / rel).read_text())


def dev5_block():
    blocks = SR.v3fam_reopen_blocks(DOM) if hasattr(SR, "v3fam_reopen_blocks") else [
        b for b in ([FAM.get("reopen")] if isinstance(FAM.get("reopen"), dict) else FAM.get("reopen") or [])]
    return next((b for b in blocks if isinstance(b, dict) and b.get("marker") == DEV5_MARKER), None)


def sup():
    """inc2.v3fam_supersede, or None where it does not exist."""
    try:
        from weed_optimizer_framework.tools.inc2 import v3fam_supersede as SUP
        return SUP
    except ImportError:
        return None


def clean():
    for d in (F.root(), C.INC_DIR / "capacity", C.INC_DIR / "_campaign", C.INC_DIR / "splits" / "v3fam"):
        shutil.rmtree(str(d), ignore_errors=True)


def put_fix(rel_fix, rel_inc):
    dst = C.INC_DIR / rel_inc
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(FIX / rel_fix), str(dst))
    return dst


def put_records(names=SEVEN + ("identity_probe.json",)):
    for n in names:
        put_fix("equivalence/%s" % n, "v3fam/v1/equivalence/%s" % n)


# ------------------------------------------------------------------ the code
def t_code():
    print("the code every reproduction record of 2026-10-12 pins")
    vf = pathlib.Path(F.__file__).resolve()
    now = F.code_shas()
    recs = {n: load("equivalence/%s" % n) for n in SEVEN + ("identity_probe.json",)}
    check("inc2/v3fam.py hashes as git b6c4aef's (3d776dae...), the file the seven records of 2026-10-12 pin",
          sha(vf) == B6C4AEF and now.get("v3fam") == B6C4AEF, sha(vf))
    check("inc2.v3fam.code_shas() equals the code map of each of the seven records of 2026-10-12 (E3RS job 49063561: "
          "dev_identity, injection, two_stage_M/A/B, identity_imageweeds, identity_ooddev_v1)",
          all(recs[n].get("code") == now for n in SEVEN) and all(recs[n]["created_utc"].startswith("2026-10-12")
                                                                  for n in SEVEN),
          {n: sorted(k for k in now if now[k] != (recs[n].get("code") or {}).get(k)) for n in SEVEN})
    pr = recs["identity_probe.json"]
    check("  identity_probe.json (2026-10-21, D5B job 49570114) differs in v3fam only: c5e923dc..., deviation 4's file",
          sorted(k for k in now if now[k] != pr["code"].get(k)) == ["v3fam"] and pr["code"]["v3fam"] == DEV4_V3FAM
          and pr["created_utc"].startswith("2026-10-21T23:45:45"), pr.get("code"))
    clean()
    put_records(SEVEN)
    got = None
    try:
        got = F.require_equivalence(list(SEVEN))
    except F.V3famRefused as e:
        got = e
    check("  inc2.v3fam.require_equivalence passes on the seven records as the cluster holds them (the refusal of "
          "2026-10-21, 'the code changed since dev_identity.json ran', is gone)",
          isinstance(got, dict) and sorted(got) == sorted(SEVEN) and all(v["status"] == "passed" for v in got.values()),
          got)
    clean()
    check("the domain's v3fam.equivalence_code is that code map (the deploy-time check compares the tree with it)",
          FAM.get("equivalence_code") == now, FAM.get("equivalence_code"))
    files = getattr(SR, "V3FAM_CODE_FILES", None)
    code_fn = getattr(SR, "v3fam_code", None)
    tools = pathlib.Path(SR.__file__).resolve().parents[2]
    same = files is not None and set(files) == set(TS.CODE_FILES) | {"v3fam"} and all(
        str((tools / v).resolve()) == str(pathlib.Path(TS.CODE_FILES[k] if k != "v3fam" else F.__file__).resolve())
        for k, v in files.items())
    check("the snapshot hashes the same seven files (stream_remote.V3FAM_CODE_FILES: inc2.twostage.CODE_FILES and "
          "inc2/v3fam.py) without importing them, and gets inc2.v3fam.code_shas()",
          same and code_fn is not None and code_fn()[0] == now, files)


def t_arm_records():
    print("which records each of the six arms finds once deviation 5 is deployed")
    SUP = sup()
    clean()
    put_records()
    moved = SUP.apply(C.INC_DIR)["moved"] if SUP else []
    table, ok = {}, True
    for arm, r in ARM_RECORDS.items():
        try:
            F.require_equivalence(list(r["present"]))
            good = True
        except F.V3famRefused as e:
            good = str(e)[:120]
        absent = [n for n in r["made"] if not (F.equiv_dir() / n).exists()]
        table[arm] = (good, absent)
        ok = ok and good is True and absent == list(r["made"])
    print("    " + "; ".join("%s: %s present and passing, taken in its job: %s" % (
        a, len(ARM_RECORDS[a]["present"]), ", ".join(ARM_RECORDS[a]["made"]) or "none") for a in SIX))
    check("each arm's records in place pass (dev_identity, injection, two_stage_B for D4, identity_imageweeds and "
          "identity_ooddev_v1 for the reported exams); identity_probe.json is moved aside, so D5B's job takes it again; "
          "D1's identity_ood22/ood23 do not exist yet and its job takes them", ok and moved == [
              "v3fam/v1/equivalence/identity_probe.json"], (table, moved))
    clean()


# ------------------------------------------------------------------ the CLI
def t_cli():
    print("the verdict's CLI outside inc2/v3fam.py")
    vtext = pathlib.Path(F.__file__).read_text()
    check("inc2/v3fam.py takes no --ticker-failed (its verdict verb is a person's, --cap-refused only)",
          "--ticker-failed" not in vtext and "ticker_failed" not in vtext)
    main = getattr(VR, "main", None)
    clean()
    rc = main(["verdict", "--ticker-failed", "data"]) if main else None
    env = dict(os.environ, PYTHONPATH=str(PKG))
    p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_verdict", "verdict",
                        "--ticker-failed", "data"], cwd=str(PKG), env=env, capture_output=True, text=True, timeout=300)
    check("inc2.v3fam_verdict verdict --ticker-failed takes the ticker's items; a refusal (the data build named) prints "
          "'[inc2.v3fam_verdict] ERROR' and exits 2, in process and under python -m, writing nothing",
          rc == 2 and p.returncode == 2 and "[inc2.v3fam_verdict] ERROR" in p.stderr and "--ticker-failed" in p.stderr
          and not (C.INC_DIR / "capacity").exists(), (rc, p.returncode, p.stderr[-300:]))
    pp = LS.policy_params("L23L", {"pkg": "inc2", "cap_refused": "score:D9", "ticker_failed": "score:D8"})
    argv = LS.render("L23L", pp)
    act = LS.row("L23L")["policy_action"]
    pol = json.loads((PKG / "weed_optimizer_framework" / "tools" / "brain" / "policy_actions.json").read_text())
    tmpl = ((pol.get("actions") or pol).get(act) or {}).get("template", "")
    req = None
    try:
        req = SR.parse_submit("build", ["inc2.v3fam_verdict", "verdict", "--ticker-failed", "score:D8"])
    except Exception as e:                       # noqa: BLE001
        req = e
    old = None
    try:
        SR.parse_submit("build", ["inc2.v3fam", "verdict", "--ticker-failed", "score:D8"])
    except Exception as e:                       # noqa: BLE001
        old = e
    rib = (PKG / "run_inc2_build.sh").read_text()
    rerun = S.StreamRun._v3fam_rerun(object.__new__(S.StreamRun), "L23L", {"ticker_failed": "score:D8"})
    check("L23L renders inc2.v3fam_verdict verdict --cap-refused ... --ticker-failed ...; the executor reads it back; "
          "the policy template names it; the cluster's grammar admits it (job inc_build_v3fam_v1) and refuses "
          "inc2.v3fam verdict; run_inc2_build.sh names it v3fam_v1; the ticker's rerun command uses it",
          argv[-6:] == ["weed_optimizer_framework.tools.inc2.v3fam_verdict", "verdict", "--cap-refused", "score:D9",
                        "--ticker-failed", "score:D8"] and X.params_from_argv(act, argv) == pp
          and "{pkg}.v3fam_verdict verdict" in tmpl and isinstance(req, dict)
          and SR.job_name(req, {}) == "inc_build_v3fam_v1" and old is not None
          and '"v3fam_verdict verdict") shift 2' in rib and "inc2.v3fam_verdict verdict --ticker-failed score:D8" in rerun,
          (argv[-6:], tmpl[:90], req, rerun[-90:]))
    clean()


# ------------------------------------------------------------------ the records superseded
def t_supersede():
    print("the records deviation 5 supersedes, moved aside byte for byte")
    SUP = sup()
    if SUP is None:
        check("inc2.v3fam_supersede exists", False, "no module")
        return
    clean()
    put_records(("identity_probe.json",) + SEVEN)
    fails = sorted(p.name for p in (FIX / "failed").glob("*.json"))
    for n in fails:
        put_fix("failed/%s" % n, "v3fam/v1/failed/%s" % n)
    put_fix("d6/smoke.json", "v3fam/v1/d6/smoke/smoke.json")
    want = {"v3fam/v1/equivalence/identity_probe.json": sha(FIX / "equivalence/identity_probe.json")}
    want.update({"v3fam/v1/failed/%s" % n: sha(FIX / "failed" / n) for n in fails})
    want["v3fam/v1/d6/smoke/smoke.json"] = sha(FIX / "d6" / "smoke.json")
    st0 = SUP.status(C.INC_DIR)
    check("the entries are the cluster's records by sha256: identity_probe.json (c99a0df0...), the six failure "
          "records of 2026-10-22 and D6's smoke of 2026-10-21 (1d0691a8...); all pending",
          {r: h for r, h, d, _f, _w in SUP.ENTRIES if d == SUP.DEVIATION_5} == want      # (deviation 6 adds its own)
          and sorted(st0["pending"]) == sorted(want)
          and want["v3fam/v1/d6/smoke/smoke.json"] == SMOKE_1021, (st0["pending"], st0["other"]))
    sc = SUP.apply(C.INC_DIR, groups=(SUP.GROUP_SCORES,))
    back = {}
    for r in sc["moved"]:                       # put them back (the same bytes) for the full run below
        os.link(str(SUP.dest_of(C.INC_DIR, r, SUP.FOLDER_5)), str(C.INC_DIR / r))
        os.unlink(str(SUP.dest_of(C.INC_DIR, r, SUP.FOLDER_5)))
        back[r] = True
    for d_ in {SUP.dest_of(C.INC_DIR, r, SUP.FOLDER_5).parent for r in sc["moved"]}:
        (d_ / SUP.MANIFEST).unlink()
        d_.rmdir()
    check("  the score jobs' group moves the six failure records and identity_probe.json, never D6's smoke (D6's own "
          "smoke job, which may run at the same time on TRAIN, moves that one)",
          sorted(sc["moved"]) == sorted(r for r in want if not r.startswith("v3fam/v1/d6/"))
          and (C.INC_DIR / "v3fam/v1/d6/smoke/smoke.json").exists(), sc["moved"])
    repo = VT.TMP / "repo_dev5"
    (repo / "docs" / "prereg").mkdir(parents=True, exist_ok=True)
    (repo / DEV5_FILE).write_text("deviation 5\n")
    os.environ["SLURM_JOB_ID"] = "4242"
    try:
        got = SUP.apply(C.INC_DIR, repo=repo)
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
    folder = {r: SUP.dest_of(C.INC_DIR, r, SUP.FOLDER_5) for r in want}
    man_eq = json.loads((F.equiv_dir() / SUP.FOLDER_5 / SUP.MANIFEST).read_text())
    man_f = json.loads((F.root() / "failed" / SUP.FOLDER_5 / SUP.MANIFEST).read_text())
    man_d6 = json.loads((F.root() / "d6" / "smoke" / SUP.FOLDER_5 / SUP.MANIFEST).read_text())
    ents = dict(man_eq["entries"], **man_f["entries"])
    ents.update(man_d6["entries"])
    check("apply moves each into its dated folder beside it (superseded_2026-10-24), the same bytes, absent at its "
          "path; the folder's MANIFEST.json names each with its sha256, bytes, job, why and the deviation's sha256; the "
          "seven records of 2026-10-12 untouched", sorted(got["moved"]) == sorted(want)
          and all(not (C.INC_DIR / r).exists() and sha(folder[r]) == s for r, s in want.items())
          and all(ents[pathlib.Path(r).name]["sha256"] == s and ents[pathlib.Path(r).name]["job"] == "4242"
                  and ents[pathlib.Path(r).name]["deviation_sha256"] == sha(repo / DEV5_FILE)
                  and ents[pathlib.Path(r).name]["why"] for r, s in want.items())
          and all((F.equiv_dir() / n).exists() for n in SEVEN), (got["moved"], sorted(ents)))
    again = SUP.apply(C.INC_DIR)
    check("  a second run changes nothing; status: all moved", again["moved"] == [] and sorted(
        again["status"]["moved"]) == sorted(want), again["status"])
    # a move cut off after its link: the same bytes at both places
    r0 = "v3fam/v1/failed/score_D8.json"
    os.link(str(folder[r0]), str(C.INC_DIR / r0))
    st1 = SUP.status(C.INC_DIR)
    cut = SUP.apply(C.INC_DIR)
    check("  a move cut off after its link (the same bytes at both places): pending, and the next run completes it",
          r0 in st1["pending"] and cut["moved"] == [r0] and not (C.INC_DIR / r0).exists() and folder[r0].exists(),
          (st1, cut["moved"]))
    # another file at an entry's path: not the record the deviation names
    (C.INC_DIR / r0).write_text('{"item": "score:D8", "status": "failed", "written_utc": "2026-10-30T00:00:00Z"}')
    other = SUP.apply(C.INC_DIR)
    check("  another file at an entry's path (a failure the ticker gave after the reopening) is left in place",
          other["moved"] == [] and (C.INC_DIR / r0).exists() and r0 in other["status"]["other"], other["status"])
    (C.INC_DIR / r0).unlink()
    # a file system that refuses hard links: the copy, and a copy cut off (a full disk)
    clean()
    put_fix("equivalence/identity_probe.json", "v3fam/v1/equivalence/identity_probe.json")
    rp = "v3fam/v1/equivalence/identity_probe.json"
    dst_p = SUP.dest_of(C.INC_DIR, rp, SUP.FOLDER_5)
    real_link, real_fsync = SUP.os.link, SUP.os.fsync

    def no_link(a, b):
        raise PermissionError(1, "Operation not permitted")

    def full(fd):
        raise OSError(28, "No space left on device")
    SUP.os.link, SUP.os.fsync = no_link, full
    e_full = e_sys = None
    try:
        try:
            SUP.apply(C.INC_DIR)
        except OSError as x:
            e_full = x
        e_sys = refused(SYS.supersede)
        left = sorted(os.listdir(str(dst_p.parent))) if dst_p.parent.is_dir() else []
        SUP.os.fsync = real_fsync
        ok_copy = SUP.apply(C.INC_DIR)
    finally:
        SUP.os.link, SUP.os.fsync = real_link, real_fsync
    check("  where hard links are refused the record is copied: a copy cut off (a full disk) leaves no file at the "
          "destination and no partial copy (only its own temporary file, removed), the record in place, and a family "
          "job refused as the platform's ('could not be moved aside'); the next run copies it whole",
          e_full is not None and e_sys is not None and "could not be moved aside" in str(e_sys)
          and S.V3FAM_PLATFORM_REFUSAL_RE.search(str(e_sys)) is not None and left == []
          and ok_copy["moved"] == [rp] and sha(dst_p) == want[rp] and not (C.INC_DIR / rp).exists(),
          (e_full, e_sys, left, ok_copy["moved"]))
    # a destination holding other bytes
    clean()
    put_fix("equivalence/identity_probe.json", "v3fam/v1/equivalence/identity_probe.json")
    dst = SUP.dest_of(C.INC_DIR, "v3fam/v1/equivalence/identity_probe.json", SUP.FOLDER_5)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("something else")
    e = None
    try:
        SUP.apply(C.INC_DIR)
    except SUP.SupersedeRefused as x:
        e = x
    e2 = refused(SYS.supersede) if hasattr(SYS, "supersede") else "no SYS.supersede"
    check("  a destination holding other bytes refuses (never moved over another file), the record left in place; "
          "the family's jobs see it as a family refusal (V3famRefused)", e is not None and "never moved over" in str(e)
          and (F.equiv_dir() / "identity_probe.json").exists() and dst.read_text() == "something else"
          and e2 is not None and "never moved over" in str(e2), (e, e2))
    clean()
    # every family score job of the six arms moves them first (v3fam_systems and v3fam_trained's score_arm)
    from weed_optimizer_framework.tools.inc2 import v3fam_trained as TR
    got_jobs = {}
    for mod, arm in ((SYS, "D8"), (TR, "D4F")):
        clean()
        put_fix("equivalence/identity_probe.json", "v3fam/v1/equivalence/identity_probe.json")
        e_job = refused(mod.score_arm, arm)               # refused later on (no build in this world)
        got_jobs[arm] = (e_job is not None, (F.equiv_dir() / "identity_probe.json").exists(),
                         (F.equiv_dir() / SUP.FOLDER_5 / "identity_probe.json").exists())
    check("  every score job of the six arms moves them first (D8's in v3fam_systems, D4F's in v3fam_trained), before "
          "anything it refuses on", got_jobs == {"D8": (True, False, True), "D4F": (True, False, True)}, got_jobs)
    clean()
    # the failure records are never read as failures, even in place
    for n in fails:
        put_fix("failed/%s" % n, "v3fam/v1/failed/%s" % n)
    st = {a: VR.arm_status(a) for a in SIX}
    VR.record_ticker_failures(["score:D8"])            # in place: kept, and still not read
    st2 = VR.arm_status("D8")
    (F.root() / "failed" / "score_D8.json").unlink()
    VR.record_ticker_failures(["score:D8"])            # a failure the ticker gives after the reopening
    st3 = VR.arm_status("D8")
    check("the six failure records of 2026-10-22, in place, are not read as the arms' failures (missing until "
          "rescored); a failure record written after the reopening is", all(v == "missing" for v in st.values())
          and st2 == "missing" and st3 == "failed (score:D8 failed twice)", (st, st2, st3))
    clean()


# ------------------------------------------------------------------ the verb's guard on the world of 2026-10-22
def oct22_world():
    """The cluster's family records as of 2026-10-24: the verdict of 2026-10-13 and its continuation of 2026-10-22
    with their external records, the six failure records, D6's smoke (not run (walltime)), the data summary, the
    twelve arms scored before the builds; no sweep trigger (BCLEAN not needed)."""
    from weed_optimizer_framework.tools.inc2 import v3fam_data as VD
    clean()
    for n in ("v3fam_v1.json", "v3fam_v1_external.json", "v3fam_v1_cont.json", "v3fam_v1_cont_external.json"):
        put_fix("capacity/%s" % n, "capacity/%s" % n)
    for p in (FIX / "failed").glob("*.json"):
        put_fix("failed/%s" % p.name, "v3fam/v1/failed/%s" % p.name)
    put_fix("d6/smoke.json", "v3fam/v1/d6/smoke/smoke.json")
    TS._write_json(VD._summary_path(), {"format": VD.FORMAT, "status": "complete", "arms": {}})
    for a in ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
              "D3", "D9"):
        TS._write_json(SYS.score_record_path(a), {"format": SYS.FMT_SCORE_RECORD, "arm": a, "status": "complete",
                                                 "exam": "dev", "dev": []})
    pdir = C.INC_DIR / "_campaign" / "provenance"
    pdir.mkdir(parents=True, exist_ok=True)
    for a in SIX + ("D6",):                     # the attempts of 2026-10-21 (the refused score jobs, D6's smoke)
        n = "v3fam_detr_smoke" if a == "D6" else "v3fam_score_%s" % a.lower()
        (pdir / ("%s.json" % n)).write_text(json.dumps({"attempts": [{"job_id": "1", "status": "build_failed",
                                                                      "started_utc": "2026-10-21T18:15:21Z",
                                                                      "finished_utc": "2026-10-21T18:20:07Z"}]}))
    return C.INC_DIR / "capacity"


def attempt_since(name, utc="2026-10-24T05:00:00Z", status="build_failed"):
    """A provenance attempt of the family job `name` started at utc (after deviation 5's reopening by default)."""
    p = C.INC_DIR / "_campaign" / "provenance" / ("%s.json" % name)
    rec = json.loads(p.read_text()) if p.exists() else {"attempts": []}
    rec["attempts"].append({"job_id": "77", "status": status, "started_utc": utc, "finished_utc": utc})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))


def t_verb_guard():
    print("the verb's guard on the world of 2026-10-22")
    cap = oct22_world()
    before = {p.name: p.read_bytes() for p in cap.glob("*.json")}
    due = getattr(VR, "second_continuation_due", lambda: False)()
    path = VR.family_verdict_path().name
    st = {a: VR.arm_status(a) for a in SIX + ("D6", "BCLEAN")}
    ok, why = VR.verdict_ready()
    check("the verdict of 2026-10-13 and the continuation of 2026-10-22 in place (by sha256): the second continuation "
          "is due and the reads follow it (capacity/v3fam_v1_cont2.json)", due and path == "v3fam_v1_cont2.json", path)
    check("D6's smoke of 2026-10-21 (superseded by sha256: its epoch time was the whole fit) is not read: D6 'missing' "
          "until its smoke is taken again; BCLEAN 'not needed' (no D0-S trigger); the six arms 'missing' (their "
          "failure records of 2026-10-22 not read)", st["D6"] == "missing" and st["BCLEAN"] == "not needed"
          and all(st[a] == "missing" for a in SIX), st)
    check("  the verb is not ready: the six arms and D6 are not scored since deviation 5 reopened them, named with its "
          "file", not ok and "not scored since deviation 5 reopened them" in why and DEV5_FILE in why
          and all(a in why for a in SIX + ("D6",)) and "never started" not in why, why)
    e = refused(VR.verdict)
    SUP = sup()
    moved = SUP.status(C.INC_DIR)["moved"] if SUP else []
    check("  the verb refuses, naming them; it writes no verdict (the earlier verdicts unchanged byte for byte); its "
          "first step moved the superseded failure records and D6's smoke of 2026-10-21 aside",
          e is not None and "deviation 5" in str(e)
          and not (cap / "v3fam_v1_cont2.json").exists() and not (cap / "v3fam_v1_cont2_external.json").exists()
          and all((cap / n).read_bytes() == b for n, b in before.items())
          and sorted(moved) == sorted(["v3fam/v1/failed/%s" % p.name for p in (FIX / "failed").glob("*.json")]
                                      + ["v3fam/v1/d6/smoke/smoke.json"]), (e, moved))
    # the lists of 2026-10-22 passed again: the six score items, each with only its attempts of 2026-10-21
    six_failed = ["score:%s" % a for a in SIX]
    ok_s, why_s = VR.verdict_ready(failed=six_failed)
    ok_e, why_e = VR.verdict_ready(failed=["score:E3RS"])
    ok_c, why_c = VR.verdict_ready(cap_refused=["data"])
    ok_c6, why_c6 = VR.verdict_ready(cap_refused=six_failed)
    check("  the ticker's lists of 2026-10-22 passed again (--ticker-failed or --cap-refused naming the six arms' "
          "score items, whose only jobs ran on 2026-10-21) end nothing: refused, naming them; E3RS named failed while "
          "its score record is complete, or the data build named refused while its summary is complete: refused as "
          "contradicted by the cluster's records", not ok_s and "no job of that item since the reopening" in why_s
          and all("score:%s" % a in why_s for a in SIX) and not ok_e and "score:E3RS (its score record is complete)"
          in why_e and not ok_c and "data (its summary is complete)" in why_c and not ok_c6
          and "no job of that item since the reopening" in why_c6, (why_s, why_e, why_c))
    early = F.root() / "failed" / "score_D8.json"
    early.parent.mkdir(parents=True, exist_ok=True)
    early.write_text(json.dumps({"format": "inc2-v3fam-failed/1", "item": "score:D8", "status": "failed",
                                 "written_utc": "2026-10-23T12:00:00Z"}))
    st_early = VR.arm_status("D8")
    early.unlink()
    check("  a failure record written before the reopening (not one deviation 5 names) is not read under the second "
          "continuation either", st_early == "missing", st_early)
    for a in SIX:
        attempt_since("v3fam_score_%s" % a.lower())
    # deviation 6 (2026-10-25): D1's failures count only with a job of it since its own reopening
    attempt_since("v3fam_score_d1", utc=getattr(VR, "DEV6_REOPEN_UTC", "2026-10-24T05:00:00Z"))
    ok_f, why_f = VR.verdict_ready(failed=six_failed)
    attempt_since("v3fam_detr_smoke")
    ok_f6, why_f6 = VR.verdict_ready(failed=six_failed + ["detr:smoke"])
    check("  with a job of each since the reopening (their provenance), the ticker's failures end the six arms; D6 "
          "still waits until its smoke's job since the reopening is named failed too", not ok_f
          and why_f.endswith("D6") and "D8" not in why_f and ok_f6, (why_f, why_f6))
    for a in SIX[:5]:
        TS._write_json(SYS.score_record_path(a), {"format": SYS.FMT_SCORE_RECORD, "arm": a, "status": "complete",
                                                 "exam": "dev", "dev": []})
    sm = F.root() / "d6" / "smoke" / "smoke.json"
    TS._write_json(sm, {"format": "inc2-v3fam-d6-smoke/1", "status": "not run (walltime)", "walltime_ok": False,
                        "epoch_seconds": 2100.0, "projection_h": 29.67})
    ok5, why5 = VR.verdict_ready()
    VR.record_ticker_failures(["score:D5B"])
    ok6, why6 = VR.verdict_ready()
    check("  five arms scored, D6's new smoke 'not run (walltime)' by its own epoch time (a record of its own, not the "
          "superseded one) and D5B still missing: not ready; D5B's failure written after the reopening: ready",
          not ok5 and why5.endswith("D5B") and VR.arm_status("D6") == "not run (walltime)" and ok6, (why5, why6))
    clean()


# ------------------------------------------------------------------ the second continuation's check
def t_cont2_check():
    print("the second continuation's check on the cluster's own verdicts")
    cap = oct22_world()
    cont = load("capacity/v3fam_v1_cont.json")
    ext = load("capacity/v3fam_v1_cont_external.json")
    p1, p4 = cont["primaries"]["P1"]["decision"], cont["primaries"]["P4"]["decision"]
    check("the records are the cluster's (2026-10-13 304bad59..., 2026-10-22 f1a4d60d...): P1 +0.0757 z 7.7723 "
          "family-confirmed, P4 -0.0042 inconclusive", sha(cap / "v3fam_v1.json") == VR.PARTIAL_VERDICT_SHA256
          and sha(cap / "v3fam_v1_cont.json") == getattr(VR, "CONT_SHA256", None)
          and abs(p1["diff"] - 0.07572647529979704) < 1e-15 and abs(p1["z"] - 7.77227630650661) < 1e-12
          and cont["holm"]["P1"]["family_confirmed"] and p4["outcome"] == "inconclusive"
          and abs(p4["diff"] - (-0.004151313826865954)) < 1e-15, (p1, p4))
    rec = {}
    try:
        rec = VR.continuation_check(cont, ext)
    except F.V3famRefused as e:
        rec = {"error": str(e)}
    by = rec.get("reproduced_by_record") or {}
    check("a second continuation deciding the same comparisons passes: the 19 decided comparisons of each earlier "
          "record reproduced (P1, P4, nine D2 challengers and D3, seven Tier S), P1 kept family-confirmed, both records "
          "named by sha256 with deviation 5", len(rec.get("reproduced") or []) == 19 and "P1" in rec["reproduced"]
          and "P4" in rec["reproduced"] and by.get("2026-10-13") == by.get("2026-10-22") == rec["reproduced"]
          and rec.get("family_confirmed_kept") == ["P1"] and rec.get("continuation_sha256") == VR.CONT_SHA256
          and rec.get("sha256") == VR.PARTIAL_VERDICT_SHA256 and rec.get("deviation") == DEV5_FILE
          and rec.get("continuation_external_as_read_on_2026_10_24") is True, rec)
    res = {}
    for tag, fn in (("P1's D 2e-9 off", lambda d: d["primaries"]["P1"]["decision"].update(diff=p1["diff"] + 2e-9)),
                    ("P1's z 0.01 off", lambda d: d["primaries"]["P1"]["decision"].update(z=p1["z"] + 0.01)),
                    ("P4 another outcome", lambda d: d["primaries"]["P4"]["decision"].update(outcome="equivalent")),
                    ("P1 no longer family-confirmed", lambda d: d["holm"]["P1"].update(family_confirmed=False)),
                    ("C_D2CLIP no longer decided", lambda d: d["challengers"]["comparisons"]["D2CLIP"].update(
                        status="not run"))):
        m = json.loads(json.dumps(cont))
        fn(m)
        res[tag] = refused(VR.continuation_check, m, ext)
    check("refused: %s" % "; ".join(res), all(e is not None and "does not reproduce" in str(e) and "2026-10-22" in
                                              str(e) for e in res.values()),
          {k: str(v)[:140] for k, v in res.items()})
    # the continuation of 2026-10-22 doctored where the verdict of 2026-10-13 is not: refused for it alone
    doc = json.loads((cap / "v3fam_v1_cont.json").read_text())
    doc["primaries"]["P4"]["decision"]["z"] += 0.5
    (cap / "v3fam_v1_cont.json").write_text(json.dumps(doc))
    real = getattr(VR, "CONT_SHA256", None)
    VR.CONT_SHA256 = sha(cap / "v3fam_v1_cont.json")
    try:
        e2 = refused(VR.continuation_check, cont, ext)
    finally:
        VR.CONT_SHA256 = real
    check("  a decision only the continuation of 2026-10-22 holds (its P4 z 0.5 off, the verdict of 2026-10-13 "
          "unchanged): refused naming 2026-10-22 and P4", e2 is not None and "P4" in str(e2) and "2026-10-22" in str(e2),
          e2)
    names = getattr(VR, "CONTINUATION_2", {})
    check("its records' names are its own (capacity/v3fam_v1_cont2*.json, v3fam_ext_v1_cont2.json): never an earlier "
          "verdict's", names.get("verdict") == "v3fam_v1_cont2.json" and not {
              names.get(k) for k in ("verdict", "external", "ext")} & {VR.VERDICT, VR.EXTERNAL, VR.EXT,
                                                                      VR.CONTINUATION["verdict"],
                                                                      VR.CONTINUATION["external"],
                                                                      VR.CONTINUATION["ext"]}, names)
    clean()
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "v3fam_v1_cont2_external.json").write_text('{"partial": true}')
    (cap / "v3fam_v1_cont2_report.md").write_text("partial")
    aside = VR.set_aside_partial(VR.CONTINUATION_2) if hasattr(VR, "set_aside_partial") else None
    folders = sorted(p.name for p in cap.glob("v3fam_v1_cont2_partial_*"))
    man = json.loads((cap / folders[0] / "MANIFEST.json").read_text()) if folders else {}
    (cap / "v3fam_v1_cont2.json").write_text("{}")
    e3 = refused(VR.set_aside_partial, VR.CONTINUATION_2) if hasattr(VR, "set_aside_partial") else None
    check("  an attempt cut off before its verdict file: its partial records moved aside byte for byte into "
          "capacity/v3fam_v1_cont2_partial_<utc>/ (a manifest with their sha256s) before the next writes them once; "
          "never once the verdict file exists", aside and len(aside) == 2 and len(folders) == 1
          and not (cap / "v3fam_v1_cont2_external.json").exists()
          and (cap / folders[0] / "v3fam_v1_cont2_external.json").read_text() == '{"partial": true}'
          and len(man.get("records") or []) == 2 and e3 is not None, (aside, folders, e3))
    clean()


# ------------------------------------------------------------------ the ticker
class Paths(object):
    def __init__(self, root):
        self.lab_inc = root / "lab" / "results" / "framework" / "inc"
        self.ledger = root / "ledger.jsonl"


def bind(name, static=False):
    fn = getattr(S.StreamRun, name, None)
    if fn is None:
        return None
    return staticmethod(fn) if static else fn


class Fam(object):
    """S.StreamRun's own family methods on a bare state (those this code has)."""
    _v3fam = bind("_v3fam")
    _v3fam_key = bind("_v3fam_key", static=True)
    _v3fam_fold = bind("_v3fam_fold", static=True)
    _v3fam_item = bind("_v3fam_item")
    _v3fam_reopen = bind("_v3fam_reopen")
    _v3fam_reopen_blocks = bind("_v3fam_reopen_blocks")
    _v3fam_reopen_block = bind("_v3fam_reopen_block")
    _v3fam_deployed = bind("_v3fam_deployed")
    _v3fam_amendment = bind("_v3fam_amendment")
    _v3fam_stage = bind("_v3fam_stage")
    _v3fam_equivalence = bind("_v3fam_equivalence")
    _v3fam_release = bind("_v3fam_release")
    _v3fam_spend = bind("_v3fam_spend")
    _v3fam_report = bind("_v3fam_report")
    _v3fam_params = bind("_v3fam_params")
    _v3fam_failed = bind("_v3fam_failed")
    _v3fam_platform_refusal = bind("_v3fam_platform_refusal")
    _v3fam_prov_refusal = bind("_v3fam_prov_refusal")
    _v3fam_rerun = bind("_v3fam_rerun")
    _failed_ids = bind("_failed_ids")
    _stream_quarantined = bind("_stream_quarantined")
    _resync_quarantines = bind("_resync_quarantines")
    _done = bind("_done")

    def __init__(self, root, st, arts=None, deploy=()):
        self.paths = Paths(root)
        self.dom = DOM
        self.st = st
        self.st.setdefault("failed_ids", [])
        self.st.setdefault("sources", {})
        self.arts = dict(arts or {})
        self.utc = "2026-10-25T03:00:00Z"
        self.sid = "weed_stream_v1_fork1364"
        self.name = "weed_stream_v1"
        self.cards, self.ledger = [], []
        deployed = self.paths.lab_inc.parents[3]
        for f in deploy:
            if not (GIT / f).is_file():
                continue                          # a tree without that file deploys nothing of it
            dst = deployed / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(GIT / f), str(dst))

    def _artifact(self, name):
        return self.arts.get(name)

    def _ledger(self, event, **kw):
        self.ledger.append(dict(kw, event=event))

    def _card(self, kind, title, detail="", lever=None, trigger=None):
        self.cards.append({"kind": kind, "title": title, "detail": detail})

    def _clear(self, ln):
        self.st.setdefault("cleared", []).append(ln)

    def _next_shard(self, src, s):
        pass


def live_items():
    """The family items as the ticker held them on 2026-10-24 (the lab's state.json, read 2026-10-25T02:27Z)."""
    it = {"data": {"state": "done", "fails": 0, "lever": "L23H", "started_utc": "2026-10-21T09:20:33Z",
                   "spend": "settled", "settled": 1.9875, "prior_gpuh": 0.792, "reopened": "deviation 4 (2026-10-20)",
                   "proposal": "68f74474b39d44cca69e9e19adc847f1"},
          "detr:smoke": {"state": "done", "lever": "L23K", "proposal": "a28e626d4ff41c8f2f782834c7273201",
                         "started_utc": "2026-10-21T11:40:33Z", "spend": "settled", "settled": 0.9372},
          "verdict": {"state": "done", "fails": 0, "lever": "L23L", "proposal": "54e3fd24ffbf024da91c6201a33d6b6d",
                      "started_utc": "2026-10-22T00:20:33Z", "spend": "settled", "settled": 0.5778,
                      "prior_gpuh": 0.66, "reopened": "deviation 4 (2026-10-20)"},
          "score:E3RS": {"state": "done", "lever": "L23J", "started_utc": "2026-10-12T13:24:57Z", "spend": "settled",
                         "settled": 1.4797, "proposal": "6a0372feaf4b0a07aa73375ffb0be105"}}
    live = {"D1": ("2026-10-21T23:00:33Z", 0.1203, 0.0831, "59a6a52876f3b3f6203893caab88acbc"),
            "D4F": ("2026-10-21T20:30:33Z", 0.0494, 0.0456, "3d767d9ebb444e30a5263e982e67ee9c"),
            "D4N": ("2026-10-21T21:30:33Z", 0.0422, 0.0053, "6e782a424ae5077fc7ede348e68b5fe4"),
            "D4R15": ("2026-10-21T22:10:33Z", 0.0864, 0.0806, "5f98133609330c21a83ed305d2603ce1"),
            "D5B": ("2026-10-22T00:00:33Z", 0.1194, 0.0928, "85c081b778ded112d6954de86c126e12"),
            "D8": ("2026-10-21T18:30:33Z", 0.0889, 0.06, "5d9648974f559b5a54bf3ade9abb10ed")}
    for a, (t, prior, settled, pid) in live.items():
        it["score:%s" % a] = {"state": "failed", "fails": 2, "lever": "L23J", "proposal": pid, "started_utc": t,
                              "spend": "settled", "settled": settled, "prior_gpuh": prior}
    for b in ("D8M", "D8S", "D4F", "D4N", "D4R15", "D1A1", "D1B1", "D1A12", "D5B0", "D5B25", "D5B100"):
        it["build:%s" % b] = {"state": "done", "lever": "L23I", "spend": "settled", "settled": 1.0,
                              "started_utc": "2026-10-21T12:00:33Z"}
    for a in ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT", "D3", "D9"):
        it["score:%s" % a] = {"state": "done", "lever": "L23J", "spend": "settled", "settled": 0.5,
                              "started_utc": "2026-10-12T16:14:57Z"}
    return it


def status_art(eq="current", stale=(), present=SEVEN + ("identity_probe.json",)):
    """v3fam/v1/status.json as the snapshot ships it after deviation 5's deploy (stream_remote.v3fam_status)."""
    st = {"format": "inc_autopilot.v3fam_status/1", "data": {"complete": True, "arms": {}},
          "timestamp": {"exists": True}, "sweep": {"complete": True}, "exif": {"passed": True},
          "detr": {"smoke": None, "walltime_ok": None, "projection_h": None, "smoke_superseded": True,
                   "smoke_sha256": SMOKE_1021, "trained": []},
          "cap_refused": []}
    if eq == "current" or eq == "stale":
        code = F.code_shas()
        st["equivalence"] = {"code": code, "fingerprint": hashlib.sha256(json.dumps(code, sort_keys=True).encode())
                             .hexdigest(), "records": [{"name": n} for n in present], "stale": list(stale),
                             "superseded": ["identity_probe.json"] if "identity_probe.json" in present else [],
                             "supersede": {"pending": [], "moved": [], "other": [], "absent": []}}
    return st


BUILT = {exp: "done" for exp in FAM["build_exps"].values()}      # the family's build experiments, done on 2026-10-21


def arts(eq="current", stale=(), present=SEVEN + ("identity_probe.json",)):
    a = {"v3fam/v1/status.json": status_art(eq, stale, present)}
    for arm in ("E3RS", "D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT",
                "D3", "D9"):
        a["capacity/v3fam_score_%s.json" % arm.lower()] = {"status": "complete"}
    a["capacity/v3fam_v1_cont.json"] = {"status": "decided"}
    a["splits/v3fam/summary.json"] = {"status": "complete"}
    return a


def t_reopen():
    print("deviation 5's reopening on the ticker's live items of 2026-10-24")
    root = VT.TMP / "reopen5"
    b5 = dev5_block()
    check("the domain's deviation 5 reopening: the six arms' scores, D6's smoke and the verdict, before 2026-10-24, "
          "its file shipped with this sha256; the second continuation the verdict record",
          b5 is not None and b5["items"] == DEV5_ITEMS and b5["before_utc"] == "2026-10-24T00:00:00Z"
          and b5["before_utc"] == VR.DEV5_REOPEN_UTC
          and b5["file"] == DEV5_FILE and (GIT / DEV5_FILE).is_file() and sha(GIT / DEV5_FILE) == b5["sha256"]
          and FAM["verdict_record"] == "capacity/v3fam_v1_cont2.json", b5)
    ro4 = FAM.get("reopen")
    old_reader = None
    try:                                          # what the ticker of deviation 4 runs on the domain file (main 004cdbd)
        ro = FAM.get("reopen") or {}
        old_reader = (ro.get("marker"), ro.get("items"), ro.get("file"), ro.get("sha256"), ro.get("before_utc"))
    except AttributeError as e:
        old_reader = e
    check("  deviation 4's block stays v3fam.reopen, an object byte for byte as deployed on 2026-10-21 (a ticker of "
          "that deploy, still running while a deploy copies this domain file before the restart, reads it with "
          "ro.get(...) and never raises), deviation 5's in v3fam.reopens after it",
          isinstance(ro4, dict) and isinstance(old_reader, tuple)
          and old_reader[:2] == ("deviation 4 (2026-10-20)", ["data", "verdict"])
          and isinstance(FAM.get("reopens"), list) and FAM["reopens"][0] is b5
          and [b.get("marker") for b in SR.v3fam_reopen_blocks(DOM)][:2] == ["deviation 4 (2026-10-20)", DEV5_MARKER],
          old_reader)
    loop = (GIT / "docs" / "CONTINUOUS_LOOP.md").read_bytes()
    check("  the deviation's file is kept byte for byte in docs/CONTINUOUS_LOOP.md",
          (GIT / DEV5_FILE).is_file() and (GIT / DEV5_FILE).read_bytes() in loop)
    f = Fam(root, {"v3fam": {"items": live_items(), "verdict_card": "2026-10-22T01:30:34Z"}},
            arts(), deploy=(DEV5_FILE, "docs/prereg/V3F_dev4_2026-10-20.md"))
    got = f._v3fam_reopen() if f._v3fam_reopen else None
    it = f.st["v3fam"]["items"]
    check("deployed: the six arms (failed twice on 2026-10-21/22), D6's smoke (done 2026-10-21, its epoch time the "
          "whole fit) and the verdict (done 2026-10-22) reopened once: missing, fails 0, the marker recorded, their "
          "earlier records kept; the data build and E3RS not", got == DEV5_ITEMS
          and all(it["score:%s" % a]["state"] == "missing" and it["score:%s" % a]["fails"] == 0
                  and it["score:%s" % a]["reopened"] == DEV5_MARKER
                  and it["score:%s" % a]["reopen_history"][-1]["state"] == "failed" for a in SIX)
          and it["verdict"]["state"] == "missing" and it["data"]["state"] == "done"
          and it["detr:smoke"]["state"] == "missing" and it["detr:smoke"]["reopened"] == DEV5_MARKER
          and abs(float(it["detr:smoke"].get("prior_gpuh") or 0) - 0.9372) < 1e-9
          and it["score:E3RS"]["state"] == "done", (got, {
              k: v.get("state") for k, v in it.items() if not k.startswith("build:")}))
    check("  every attempt stays counted against the family cap: D8's prior 0.0889 + 0.06, the verdict's 0.66 + "
          "0.5778; the earlier ids retired; the verdict's card mark in its history; one card",
          abs(it["score:D8"]["prior_gpuh"] - 0.1489) < 1e-9 and abs(it["verdict"]["prior_gpuh"] - 1.2378) < 1e-9
          and {"5d9648974f559b5a54bf3ade9abb10ed", "54e3fd24ffbf024da91c6201a33d6b6d"} <= set(f.st["failed_ids"])
          and it["verdict"]["reopen_history"][-1].get("verdict_card") == "2026-10-22T01:30:34Z"
          and len([c for c in f.cards if DEV5_MARKER in c["title"]]) == 1, (it["score:D8"], f.cards))
    again = f._v3fam_reopen()
    it["score:D8"]["state"] = "failed"
    third = f._v3fam_reopen()
    check("  once per marker: a second tick reopens nothing, nor an arm that fails again; deviation 4's block never "
          "reopens the verdict or the data build again", again == [] and third == [], (again, third))
    fs = f._v3fam_stage(BUILT)
    it["score:D8"]["state"] = "missing"
    fs = f._v3fam_stage(BUILT)
    plan, waits = DS.v3fam_plan(FAM, fs)
    ready, pending = DS.v3fam_verdict_ready(FAM, fs)
    check("then: the records current (identity_probe.json superseded, not stale), D6's smoke of 2026-10-21 reported "
          "as no smoke (superseded by sha256): L23K's smoke on TRAIN, L23J D8 first on MAINT, the verdict waiting for "
          "the six arms and D6", (fs.get("equivalence") or {}).get("state") == "current"
          and fs["detr"]["smoke"] == "missing" and fs["score"]["D6"] == "missing"
          and plan.get("TRAIN") == {"lever": "L23K", "verb": "smoke", "item": "detr:smoke"}
          and plan.get("MAINT") == {"lever": "L23J", "arm": "D8", "item": "score:D8"} and not ready
          and set(SIX) <= set(pending) and "D6" in pending and "detr:smoke" in pending and fs["verdict"] == "missing",
          (fs.get("equivalence"), fs.get("detr"), plan, pending))
    f.arts = arts(present=tuple(n for n in SEVEN if n != "injection.json"))
    fsm = f._v3fam_stage(BUILT)
    plan_m, waits_m = DS.v3fam_plan(FAM, fsm)
    check("  a record E3RS's job takes not in place (injection.json moved aside, not retaken): stale, no L23J, the wait "
          "names it", fsm["equivalence"]["state"] == "stale" and fsm["equivalence"]["missing"] == ["injection.json"]
          and "MAINT" not in plan_m and "injection.json" in waits_m.get("score:D8", ""), (fsm["equivalence"], plan_m))
    f.arts = arts()
    hist = Fam(root / "hist", {"v3fam": {"items": {"verdict": {
        "state": "failed", "lever": "L23L", "started_utc": "2026-10-19T00:00:00Z", "reopened": DEV5_MARKER,
        "reopen_history": [{"marker": "deviation 4 (2026-10-20)"}, {"marker": DEV5_MARKER}]}}}}, arts(),
        deploy=("docs/prereg/V3F_dev4_2026-10-20.md", DEV5_FILE))
    hist._v3fam_reopen()
    check("  an item whose reopen history holds a marker is never reopened by that block again (deviation 4's, while "
          "its record names deviation 5's)", hist.st["v3fam"]["items"]["verdict"]["state"] == "failed",
          hist.st["v3fam"]["items"]["verdict"])
    held = Fam(root / "held", {"v3fam": {"items": {"score:D8": {
        "state": DS.V3FAM_HELD_PLATFORM, "lever": "L23J", "platform_refusals": 3, "proposal": None,
        "started_utc": "2026-10-23T01:00:00Z", "spend": "settled", "settled": 0.02, "prior_gpuh": 0.04}}}}, arts(),
        deploy=(DEV5_FILE,))
    got_h = held._v3fam_reopen()
    rh = held.st["v3fam"]["items"]["score:D8"]
    check("  an item held for a person after three refusals of the platform's own making is reached by a recorded "
          "deviation's reopening (its sanctioned way out): missing, its refusals count reset, its spend kept",
          got_h == ["score:D8"] and rh["state"] == "missing" and "platform_refusals" not in rh
          and abs(float(rh.get("prior_gpuh") or 0) - 0.06) < 1e-9, rh)
    late = Fam(root / "late", {"v3fam": {"items": dict(live_items(), **{"score:D8": dict(
        live_items()["score:D8"], started_utc="2026-10-24T03:00:00Z")})}}, arts(), deploy=(DEV5_FILE,))
    late._v3fam_reopen()
    none = Fam(root / "none", {"v3fam": {"items": live_items()}}, arts())
    none._v3fam_reopen()
    check("  never reopened: an arm whose last attempt started after 2026-10-24, or any item while the deployed tree "
          "does not hold the deviation's file", late.st["v3fam"]["items"]["score:D8"]["state"] == "failed"
          and all(v["state"] != "missing" for v in none.st["v3fam"]["items"].values()))


def running(arm, est=0.6):
    return {"state": "running", "lever": "L23J", "proposal": "p_%s" % arm, "spend": "open", "estimate": est,
            "started_utc": "2026-10-21T18:30:33Z"}


def item(arm, pid):
    return {"lever": "L23J", "since_utc": "2026-10-21T18:10:33Z",
            "proposal": {"id": pid, "params": {"pkg": "inc2", "arm": arm}, "trigger": ["DR0"]}}


def t_platform_refusal():
    print("a refusal of the platform's own making is never an arm's failure")
    root = VT.TMP / "refusal"
    items = {"score:%s" % a: running(a) for a in SIX}
    f = Fam(root, {"v3fam": {"items": items}, "declined": []}, arts(eq="stale", stale=SEVEN))
    for a in SIX:
        f._v3fam_failed("MAINT", item(a, "p1_%s" % a), "job(s) 4956 ended FAILED; refused: %s" % REFUSAL, True)
    its = f.st["v3fam"]["items"]
    p, _c = f._v3fam_params("L23L", {}, "inc2", {})
    check("the refusals of 2026-10-21 (six arms): no fails counted, nothing failed or declined, each id retired, each "
          "attempt's GPU-h kept in prior_gpuh, each held (held (platform)); L23L would pass no --ticker-failed",
          all(not its["score:%s" % a].get("fails") and its["score:%s" % a]["state"] == DS.V3FAM_HELD_PLATFORM
              and abs(its["score:%s" % a]["prior_gpuh"] - 0.6) < 1e-9 for a in SIX)
          and not f.st["declined"] and len(f.st["failed_ids"]) == 6 and "ticker_failed" not in p
          and len([e for e in f.ledger if e["event"] == "v3fam_platform_refusal"]) == 6
          and not [e for e in f.ledger if e["event"] == "failed"], (its["score:D8"], p))
    c0 = f.cards[0] if f.cards else {}
    check("  one card each, naming the remedy (restore the code the records pin, or move them aside and retake them, "
          "as a recorded deviation)", len(f.cards) == 6 and "not the arm's failure" in c0.get("title", "")
          and "restore the code they pin" in c0.get("detail", "") and "retake" in c0.get("detail", ""), c0)
    held = DS.V3FAM_HELD_PLATFORM
    check("  held is neither due nor ended: the verdict waits for it", not DS.v3fam_terminal(held)
          and not DS._v3fam_due(held), held)
    fs = f._v3fam_stage(BUILT)
    plan, waits = DS.v3fam_plan(FAM, fs)
    check("  while the records are stale nothing is released, no L23J proposed, each arm's wait named",
          its["score:D8"]["state"] == held and fs["equivalence"]["state"] == "stale" and "MAINT" not in plan
          and "held" in waits.get("score:D8", ""), (fs.get("equivalence"), plan, waits.get("score:D8")))
    f.arts = arts()
    fs = f._v3fam_stage(BUILT)
    plan, _w = DS.v3fam_plan(FAM, fs)
    check("  the records current again: every held item released (missing, a ledger event), L23J D8 proposed",
          all(its["score:%s" % a]["state"] == "missing" for a in SIX) and plan.get("MAINT", {}).get("arm") == "D8"
          and [e for e in f.ledger if e["event"] == "v3fam_released"], plan)
    for rnd in (2, 3):
        its["score:D8"].update(state="running", spend="open", estimate=0.6)
        f._v3fam_failed("MAINT", item("D8", "p%d_D8" % rnd), "refused: %s" % REFUSAL, True)
    fs = f._v3fam_stage(BUILT)
    check("  refused a third time although the records were current: held for a person (not released), one "
          "escalation card", its["score:D8"]["state"] == held and its["score:D8"]["platform_refusals"] == 3
          and "held after 3 refusals" in f.cards[-1]["title"] and fs["score"]["D8"] == held, its["score:D8"])
    f2 = Fam(root, {"v3fam": {"items": {"score:D4F": running("D4F")}}, "declined": []}, arts())
    f2._v3fam_failed("MAINT", item("D4F", "q1"), "job(s) 2 ended FAILED; refused: [inc2.v3fam] ERROR: dev identity: "
                                                 "a reproduction differs by 0.004 (> 0.002)", True)
    r2 = f2.st["v3fam"]["items"]["score:D4F"]
    check("an arm's own refusal still counts: failed once, proposed once more", r2.get("fails") == 1
          and r2["state"] == "failed_once" and not r2.get("platform_refusals"), r2)
    for txt in ("V3famRefused: /x/equivalence/injection.json is missing or did not pass: no family score is taken "
                "without it", "the E3-frozen file(s) ['twostage'] no longer hash as V3F.1 records: no family verb runs",
                "the E-series records changed while this family job ran (x): the family writes only",
                "superseded_2026-10-24/identity_probe.json exists: the superseded record /x is never moved over another "
                "file", REFUSAL):
        check("  a platform precondition: %s" % txt.replace("\n", " ")[-70:],
              getattr(S, "V3FAM_PLATFORM_REFUSAL_RE", None) is not None
              and S.V3FAM_PLATFORM_REFUSAL_RE.search(txt) is not None)
    # a submission whose outcome was unknown (the record-only follow): its provenance's refusal routes it the same way
    f3 = Fam(root, {"v3fam": {"items": {"score:D4N": running("D4N")}}, "declined": [], "lanes": {"MAINT": {}}},
             arts(eq="stale", stale=SEVEN))
    f3._record_only_follow = S.StreamRun._record_only_follow.__get__(f3)
    f3._v3fam_record_done = S.StreamRun._v3fam_record_done.__get__(f3)
    f3._failed = S.StreamRun._failed.__get__(f3)
    f3._follow_record_only_ = S.StreamRun._follow_record_only.__get__(f3)
    it3 = dict(item("D4N", "u1"), uncertain=True, lost=S.BUILD_LOST_SNAPSHOTS - 1)
    pay3 = {"status": {"builds": {"v3fam_score_d4n": {"job_id": "49565727", "refusal": REFUSAL,
                                                      "started_utc": "2026-10-21T20:54:54Z"}}}}
    try:
        f3._follow_record_only_("MAINT", it3, set(), set(), {}, payload=pay3)
    except TypeError:                            # a follow that takes no payload
        f3._follow_record_only_("MAINT", it3, set(), set(), {})
    r3 = f3.st["v3fam"]["items"]["score:D4N"]
    check("  after a submission whose outcome was unknown, the provenance's refusal (an attempt started after the "
          "submission) routes it the same way: held, not failed", r3["state"] == DS.V3FAM_HELD_PLATFORM
          and not r3.get("fails"), r3)
    payload = {"status": {"builds": {"v3fam_score_d8": {"job_id": "49561782", "refusal": REFUSAL,
                                                        "started_utc": "2026-10-21T18:55:08Z"}}}}
    got = f._v3fam_prov_refusal(payload, item("D8", "z"), ["49561782"]) if f._v3fam_prov_refusal else ""
    other = f._v3fam_prov_refusal(payload, item("D8", "z"), ["49561022"]) if f._v3fam_prov_refusal else "x"
    unk = f._v3fam_prov_refusal(payload, item("D8", "z"), None) if f._v3fam_prov_refusal else ""
    late = f._v3fam_prov_refusal(payload, dict(item("D8", "z"), since_utc="2026-10-22T00:00:00Z"), None) \
        if f._v3fam_prov_refusal else "x"
    check("the refusal is read from the job's provenance (its last attempt, when it is this job's; after a submission "
          "whose outcome was unknown, when it started after the item): the snapshot carried it on 2026-10-21 and the "
          "ticker never read it", "the code changed since" in got and other == "" and "the code changed since" in unk
          and late == "", (got[-80:], other, late))
    # a refusal whose message is only on the traceback's last line, beyond 600 characters (run_inc2_build.sh keeps the
    # log's last five lines; under python -m a score job's V3famRefused is a traceback): still the platform's
    inc = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc/v3fam/v1/equivalence"
    tb = ('  File "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/weed_optimizer_framework/tools/inc2/'
          'v3fam_trained.py", line 523, in score_arm\n    SYS.supersede()                          # deviation 5: the '
          'records it supersedes moved aside first\n  File "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/'
          'weed_optimizer_framework/tools/inc2/v3fam_systems.py", line 133, in supersede\n    raise '
          'F.V3famRefused(str(e))\nweed_optimizer_framework.tools.inc2.v3fam.V3famRefused: %s/superseded_2026-10-24/'
          'identity_probe.json exists: the superseded record %s/identity_probe.json is never moved over another file'
          % (inc, inc))
    pay_tb = {"status": {"builds": {"v3fam_score_d4r15": {"job_id": "5001", "refusal": tb,
                                                          "started_utc": "2026-10-24T08:00:00Z"}}}}
    got_tb = f._v3fam_prov_refusal(pay_tb, item("D4R15", "tb1"), ["5001"]) if f._v3fam_prov_refusal else ""
    f4 = Fam(root, {"v3fam": {"items": {"score:D4R15": running("D4R15")}}, "declined": []}, arts())
    f4._v3fam_failed("MAINT", item("D4R15", "tb1"), "job(s) 5001 ended FAILED; refused: %s" % got_tb, True)
    r4 = f4.st["v3fam"]["items"]["score:D4R15"]
    check("  a %d-character traceback whose message (a superseded record that cannot be moved) is only on its last "
          "line, past character 600: the ticker keeps it (the tail, and the phrase that classes it), so the arm is "
          "held, not failed" % len(tb), len(tb) > 600 and tb.index("is never moved over") > 600
          and "is never moved over another file" in got_tb and len(got_tb) <= 1000
          and r4["state"] == DS.V3FAM_HELD_PLATFORM and not r4.get("fails"), (len(tb), got_tb[-120:], r4))


def t_pre_score():
    print("the pre-score check")
    sc = {a: "done" for a in FAM["pre_build_scores"] + FAM["post_build_scores"]}
    sc.update({a: "missing" for a in SIX})
    sc.update({"BCLEAN": "not needed", "D6": "not run (a dependency ended without its effect)"})
    b = {a: "done" for a in FAM["builds"]}
    b["D0BCLEAN"] = "not needed"
    base = {"data": "done", "score": sc, "build": b, "detr": {"smoke": "not run (walltime)",
                                                             "train": "not run (walltime)"}, "verdict": "missing"}
    got = {}
    for tag, eq in (("stale", {"state": "stale", "stale": list(SEVEN), "missing": []}),
                    ("a record not in place", {"state": "stale", "stale": [], "missing": ["injection.json"]}),
                    ("current", {"state": "current", "stale": [], "missing": []}),
                    ("unknown (no block)", {"state": "unknown"})):
        plan, waits = DS.v3fam_plan(FAM, dict(base, equivalence=eq))
        got[tag] = ((plan.get("MAINT") or {}).get("lever"), waits.get("score:D8", ""))
    check("no L23J while a record is taken by other code or not in place (the wait names the records and the remedy); "
          "current: L23J; a snapshot without the block: L23J (the job's own check stands)",
          got["stale"][0] is None and "dev_identity.json" in got["stale"][1] and "restores the code" in got["stale"][1]
          and got["a record not in place"][0] is None and "injection.json" in got["a record not in place"][1]
          and got["current"][0] == "L23J" and got["unknown (no block)"][0] == "L23J", got)


# ------------------------------------------------------------------ the deploy guards
def t_deploy():
    print("the deploy guards")
    chk = getattr(SR, "v3fam_deploy_check", None)
    guard = getattr(SR, "v3fam_records_guard", None)
    sh = (PKG / "deploy" / "deploy_funnel.sh").read_text()
    roots = sh.split("ROOT_PATHS=(", 1)[1].split(")", 1)[0].split()
    pkgs = sh.split("PKG_PATHS=(", 1)[1].split("\n)", 1)[0]
    if chk is None or guard is None:
        check("stream_remote.v3fam_deploy_check and v3fam_records_guard exist", False)
        return
    ok, lines = chk(DOM, GIT, roots)
    check("this tree passes: its code is the domain's v3fam.equivalence_code and every file the v3fam block pins is "
          "shipped (ROOT_PATHS) with that sha256", ok and DEV5_FILE in roots and "tests/fixtures/v3fam_dev5" in pkgs,
          lines)
    pkg = VT.TMP / "pkg" / "weed_optimizer_framework"
    for rel in SR.V3FAM_CODE_FILES.values():
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(PKG / "weed_optimizer_framework" / rel), str(dst))
    ok_copy, _l = chk(DOM, GIT, roots, package=pkg)
    (pkg / "tools" / "inc2" / "v3fam.py").write_text((pkg / "tools" / "inc2" / "v3fam.py").read_text()
                                                     + "\n# a CLI change\n")
    bad, blines = chk(DOM, GIT, roots, package=pkg)
    newest = SR.v3fam_reopen_blocks(DOM)[-1]["file"]          # deviation 6's since 2026-10-25
    allow, alines = chk(DOM, GIT, roots, package=pkg, allow=newest)
    nope, _n = chk(DOM, GIT, roots, package=pkg, allow="docs/prereg/none.md")
    log_, _l = chk(DOM, GIT, roots, package=pkg, allow="RESEARCH_LOG.md")
    dev4_, _d = chk(DOM, GIT, roots, package=pkg, allow="docs/prereg/V3F_dev4_2026-10-20.md")
    check("a copy of the seven files passes; with inc2/v3fam.py changed (deviation 4's kind of change) it is refused "
          "naming v3fam and the remedy; --allow-v3fam-code-change with the newest recorded deviation (the domain's "
          "last reopen block's file, shipped and hashing as pinned) passes with a notice; with any other file (one "
          "not shipped, the research log, deviation 4's) it is refused", ok_copy and not bad
          and "v3fam 3d776daed91d" in " ".join(blines) and "--allow-v3fam-code-change" in " ".join(blines) and allow
          and "NOTICE" in " ".join(alines) and not nope and not log_ and not dev4_, (blines[:1], alines[:1]))
    git2 = VT.TMP / "git2"
    for f_ in roots:
        if (GIT / f_).is_file():
            (git2 / f_).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(GIT / f_), str(git2 / f_))
    ok2, _l2 = chk(DOM, git2, roots)
    (git2 / newest).write_text((git2 / newest).read_text() + "\nan edit after the pin\n")
    edited, elines = chk(DOM, git2, roots)
    edited_allow, _ea = chk(DOM, git2, roots, package=pkg, allow=newest)
    check("  --allow-v3fam-code-change naming the newest deviation's file edited after its pin is refused (the file "
          "must hash as its reopen block pins it)", not edited_allow and not SR.v3fam_allow_ok(DOM, git2, newest,
                                                                                             roots)[0], _ea)
    gone, glines = chk(DOM, GIT, [r for r in roots if r != DEV5_FILE])
    check("a deviation's file edited after its pin (its sha256 no longer the domain's) is refused, naming it; a pinned "
          "file left out of the git-root paths is refused", ok2 and not edited and newest in " ".join(elines)
          and not gone and "not a git-root path" in " ".join(glines), (elines[-1:], glines[-1:]))
    changed = ["weed_llm_benchmark/weed_optimizer_framework/tools/inc2/v3fam.py",
               "weed_optimizer_framework/tools/inc2/v3fam.py", "weed_optimizer_framework/tools/inc2/v3fam_verdict.py"]
    # the cluster's records as step 2 fetches them (name, sha256, code map): the seven of 2026-10-12 and
    # identity_probe.json of 2026-10-21, byte for byte
    recs = {"records": [{"name": n, "sha256": sha(FIX / "equivalence" / n), "code": load("equivalence/%s" % n)["code"]}
                        for n in SEVEN + ("identity_probe.json",)]}
    stale_recs = {"records": [dict(r, code=dict(r["code"], v3fam=DEV4_V3FAM)) for r in recs["records"]]}
    kw = {"dom": DOM, "git_root": GIT}
    g = {"restore (this deploy)": guard(changed, recs, None, roots, **kw),
         "restore, as JSON text": guard(changed, json.dumps(recs), None, roots, **kw),
         "records of other code": guard(changed, stale_recs, None, roots, **kw),
         "this tree changed": guard(changed, recs, None, roots, package=pkg, **kw),
         "changed, newest deviation": guard(changed, recs, newest, roots, package=pkg, **kw),
         "changed, deviation 4": guard(changed, recs, "docs/prereg/V3F_dev4_2026-10-20.md", roots, package=pkg, **kw),
         "changed, the research log": guard(changed, recs, "RESEARCH_LOG.md", roots, package=pkg, **kw),
         "8 records of unknown code": guard(changed, 8, None, roots, **kw),
         "no record": guard(changed, {"records": []}, None, roots, **kw),
         "fetch failed": guard(changed, {"error": "the fetch failed"}, None, roots, **kw),
         "unreadable": guard(changed, "not json", None, roots, **kw),
         "other files only": guard(changed[2:], recs, None, roots, **kw),
         "a frozen file, unknown records": guard(["weed_optimizer_framework/tools/inc2/twostage.py"], 8, None, roots,
                                                 **kw)}
    check("step 2 (beside PINNED_RE): a change of inc2/v3fam.py on the cluster passes when every record (but "
          "identity_probe.json, superseded by sha256) pins the code this tree ships: deviation 5's own deploy, which "
          "restores the bytes they pin, needs no flag; records taken by other code than the tree ships, a fetch that "
          "failed or cannot be read, or records of unknown code refuse it (an E3-frozen file too), unless the newest "
          "recorded deviation is named (deviation 4's file or the research log refused); no record, or other files "
          "only, pass", g["restore (this deploy)"][0] and g["restore, as JSON text"][0]
          and "identity_probe.json" in g["restore (this deploy)"][1] and not g["records of other code"][0]
          and not g["this tree changed"][0] and g["changed, newest deviation"][0]
          and not g["changed, deviation 4"][0] and not g["changed, the research log"][0]
          and not g["8 records of unknown code"][0] and g["no record"][0] and not g["fetch failed"][0]
          and not g["unreadable"][0] and g["other files only"][0] and not g["a frozen file, unknown records"][0],
          {k: v[0] for k, v in g.items()})
    check("deploy_funnel.sh runs both: the tree's checks before anything is copied and in --dry-run, step 2's guard "
          "on rsync's list of changes, the flag recorded in stream_deployed.json",
          "SR.v3fam_deploy_check" in sh and "S.v3fam_records_guard" in sh and "--allow-v3fam-code-change=*" in sh
          and "--include='*.json' --exclude='*'" in sh and 'package="weed_optimizer_framework"' in sh
          and '"v3fam_code_change": $V3F_CHANGE' in sh
          and sh.index("v3fam_deploy_check") < sh.index('if [ "$DRY" = 1 ]; then'), None)


# ------------------------------------------------------------------ D6's smoke taken again
def t_d6():
    print("D6's smoke of 2026-10-21 superseded: the snapshot, the verb and the card")
    clean()
    put_fix("d6/smoke.json", "v3fam/v1/d6/smoke/smoke.json")
    rec = load("d6/smoke.json")
    st = SR.v3fam_status(C.INC_DIR)
    det = st.get("detr") or {}
    check("the smoke record of 2026-10-21 (job 49542653: epoch_seconds = fit_seconds = 2,405.583, projection 33.91 h) "
          "is reported by the snapshot as no smoke, superseded by sha256 (so the ticker proposes the smoke again), and "
          "the verb reads no D6 status from it", rec["epoch_seconds"] == rec["attempt"]["fit_seconds"] == 2405.583
          and det.get("smoke") is None and det.get("walltime_ok") is None and det.get("smoke_superseded") is True
          and det.get("smoke_sha256") == SMOKE_1021 and VR.d6_smoke_status() is None, det)
    (F.root() / "d6" / "smoke" / "smoke.json").unlink()
    TS._write_json(F.root() / "d6" / "smoke" / "smoke.json", {
        "format": "inc2-v3fam-d6-smoke/1", "status": "not run (walltime)", "walltime_ok": False,
        "epoch_seconds": 2100.0, "projection_h": 29.67, "epoch_time": "epoch clock: ..."})
    st2 = SR.v3fam_status(C.INC_DIR)
    check("  a smoke record of its own (the epoch timed from its start) is read as it says, by the snapshot and the "
          "verb", (st2.get("detr") or {}).get("smoke") == "not run (walltime)"
          and (st2.get("detr") or {}).get("smoke_superseded") is False
          and (st2.get("detr") or {}).get("epoch_seconds") == 2100.0 and VR.d6_smoke_status() == "not run (walltime)",
          st2.get("detr"))
    clean()


# ------------------------------------------------------------------ D28
def t_d28():
    print("D28: the stream ledger's quarantines followed by the ticker")
    root = VT.TMP / "d28"
    live = json.loads((FIX / "d28_live_2026-10-24.json").read_text())
    st = {"lanes": {"DATA": {}, "STOP": {}}, "sources": json.loads(json.dumps(live["sources"]))}
    f = Fam(root, st, {"stream/%s/queue_summary.json" % live["sid"]: {
        "quarantined_sources": live["quarantined_sources"]}})
    f.sid = live["sid"]
    got = f._resync_quarantines() if f._resync_quarantines else None
    again = f._resync_quarantines() if f._resync_quarantines else None
    leaks = live["d28_leaks"]
    check("the live state of 2026-10-24 (the three D28 leaks of 2026-10-06/07 and weedai_5c78... of 2026-10-01 'held: "
          "quarantine pending' while the stream's ledger quarantines them, seq 18-21): each resynced to quarantined "
          "with the ledger's cite and seq, its pending reason dropped, one ledger event each; once",
          got is not None and sorted(got) == sorted(live["sources"]) and again == []
          and all(st["sources"][s]["status"] == "quarantined"
                  and st["sources"][s]["quarantined_seq"] == live["quarantined_sources"][s]["seq"]
                  and "held_reason" not in st["sources"][s] for s in live["sources"])
          and len([e for e in f.ledger if e["event"] == "source_quarantine_resynced"]) == 4, (got, leaks))
    src = leaks[0]
    st["sources"][src]["status"] = "quarantined"
    batch = st["sources"][src]["batch"]
    f._done("DATA", {"lever": "L17", "proposal": {"id": "adm1", "params": {"verb": "admit", "intake": batch}}})
    check("an admit done after the source's quarantine leaves it quarantined (the race of 2026-10-06/07), its "
          "admit_done_utc recorded", st["sources"][src]["status"] == "quarantined"
          and st["sources"][src].get("admit_done_utc") == f.utc, st["sources"][src].get("status"))
    st["sources"]["fresh_src"] = {"status": "intaken", "batch": "i0099_fresh_src"}
    f._done("DATA", {"lever": "L17", "proposal": {"id": "adm2", "params": {"verb": "admit",
                                                                          "intake": "i0099_fresh_src"}}})
    check("  an ordinary source is still admitted", st["sources"]["fresh_src"]["status"] == "admitted")
    class Ev(object):
        def json(self, name):
            return {}
    st["sources"][leaks[1]] = dict(live["sources"][leaks[1]])         # back to 'held: quarantine pending'
    f.ev, f.pkg, f.th = Ev(), "inc2", {}
    f._arm = lambda: {"name": "m640"}
    prop = S.StreamRun._proposal(f, "L24", {"source": leaks[1], "cite": "D28"}, "D28", {"cites": []})
    check("L24's render for a source the stream's ledger already quarantines: nothing to run (None), the ticker's "
          "record resynced from the ledger", prop is None and st["sources"][leaks[1]]["status"] == "quarantined", prop)
    import inspect
    d28src = inspect.getsource(DS.d28)
    check("D28 proposes L24 only for a leak the stream's queue summary does not already quarantine (D31 likewise)",
          'for h in hits if str(h["source"]) not in qs' in d28src and "if s not in qs]" in inspect.getsource(DS.d31))


def main():
    try:
        for fn in (t_code, t_arm_records, t_cli, t_supersede, t_verb_guard, t_cont2_check, t_reopen,
                   t_platform_refusal, t_pre_score, t_deploy, t_d6, t_d28):
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-1500:]))
    finally:
        shutil.rmtree(str(VT.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
