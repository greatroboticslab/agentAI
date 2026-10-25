"""V3F's system arms: the comparators through the family path (E3RS), the
YOLO11s and YOLO11m 12-class references on cwd12 train_core (D8), and the
helpers the D3 (inc2/v3fam_wbf.py) and D6 (inc2/v3fam_detr.py) verbs share
(docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11): V3F, the v3 experiment
family D0-D9 (pre-registered as one family)", V3F.5 E3RS and D8; V3F.3.2,
V3F.3.5, V3F.3.7).

    python -m weed_optimizer_framework.tools.inc2.v3fam_systems score --arm E3RS|D3|D6|D8        (L23J)
    python -m weed_optimizer_framework.tools.inc2.v3fam_systems build --v3fam D8S|D8M [--no-init]  (L23I's D8)
    python -m weed_optimizer_framework.tools.inc2.v3fam_systems sources --arm D8M|D6     (a reference read's inputs)
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

The platform's verbs are inc2.v3fam score --arm X (which hands E3RS, D3, D6
and D8 to score_arm here) and inc2.baseline build --v3fam D8S|D8M (which takes
its arguments from d8_build_kwargs); the CLI above runs the same functions.

E3RS (one L23J job before any other family score). First the reproductions
of V3F.3.2, each written once by inc2.v3fam (dev identity, injection, E3's
three arms' two-stage files; then the ImageWeeds and ooddev_v1 identities,
whose failure is recorded and fails only those exams' passes). Then the
family-path dev arrays every dev comparison and the selection correction use:
E3-M, E3-A and E3-B (E3's pinned classifier on E3's stage-1 weights, E3's
NMS per arm) and the reference b_v2_m640 (identity), 12 passes; O5's oracle
reading on E3-B (3). Reported: E3-B's three seeds two-stage on ImageWeeds
(the stage-1 rows D3 fuses) and ooddev_v1 for E1-A and E1-B (stage1 mode:
their box sets, the rows D3 fuses) and b_v2_m640 (identity), 9 passes in the
zoo's root, each in a subprocess with INC_DIR there.

D8 (V3F.5 D8). Builds: inc2.baseline build --exp v3_d8_s640_tc|v3_d8_m640_tc
on LOCK v2's train_core (3,048 rows; by sha256), arm s640 or m640, the cold
table (100 epochs), seeds 0-2, role baseline, finals dev and ImageWeeds;
exp.json carries a v3fam record naming the arm. Scoring (L23J, --arm D8):
each seed's base run through the family identity pass on dev (deciding side
of the secondary comparisons; the reference's arrays are E3RS's), its dev
score compared with the run's recorded protocol dev score; ImageWeeds and
ooddev_v1 reported. D8-m has one reference read (V3F.3.6), D8-s none.

Where everything goes (V3F.3.5, V3F.5 "Outputs of every scored arm"):
  v3fam/v1/arms/<dir>/s<k>/<exam>[.<variant>].{json, images.npz, stage1.npz, rows.npz, p7.npz}
      dirs: E3M, E3A, E3B (two-stage; E3B also dev.oracle.*), REF (b_v2_m640, identity), E1A, E1B (ooddev_v1,
      stage1 mode), D8S, D8M (identity); D3 under arms/D3/ens/ (one system, inc2/v3fam_wbf.py); D6
      (inc2/v3fam_detr.py)
  v3fam/v1/reported/<ARM>.json        the reported passes' outcome (names, sha256s, status; overwritten)
  capacity/v3fam_score_<arm>.json     the platform's record: dev files, sha256s, status and counts (overwritten)
Every family score is written once (inc2.v3fam.commit_pass); a rerun keeps a
complete file and takes only what is missing. Every job hashes twostage/e3_v1
and capacity/e{1,2,3}_* before and after and fails if either changed.

Public API: score_arm(arm, **kw); score_e3rs; score_d8; run_source(exp, seed);
d8_build_kwargs(arm, ...), build_d8(arm, ...), check_d8_definition(arm);
family_runs(dir, exam, variant, seeds) (the committed arrays, sha256-checked);
reference_sources(arm); commit(...); score_record(...); exam_rows(exam).

A refusal raises inc2.v3fam.V3famRefused; the CLI prints
"[inc2.v3fam_systems] ERROR: ..." and exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from . import baseline as B
from . import common as C2
from . import recipes as RC
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import twostage as TS
from . import v3fam as F
from . import v3fam_supersede as SUP

# ------------------------------------------------------------------ constants (pre-registered)
SYSTEM_ARMS = ("E3RS", "D3", "D6", "D8")    # the L23J arms this module scores (inc2.v3fam score --arm)
E3_DIRS = {"M": "E3M", "A": "E3A", "B": "E3B"}         # E3's arms through the family two-stage pass
REF_DIR = "REF"                                         # b_v2_m640, the identity pass
E1_DIRS = {"A": "E1A", "B": "E1B"}                      # E1's one-class detectors (ooddev_v1, stage1 mode)
SYSTEM_NAMES = {"E3M": "E3-M", "E3A": "E3-A", "E3B": "E3-B", "REF": "b_v2_m640 (YOLO11m reference)",
                "E1A": "E1-A (one-class boxes)", "E1B": "E1-B (one-class boxes)", "D8S": "D8-s (YOLO11s train_core)",
                "D8M": "D8-m (YOLO11m train_core)", "D3": "D3 (WBF of E3-B's stage-1 seeds)",
                "D6": "D6 (RF-DETR-L 12-class)"}
ORACLE_VARIANT = "oracle"
OODDEV = "ooddev_v1"
E3RS_REPORTED_EXAMS = ("imageweeds", OODDEV)
SCORE_RECORD = "v3fam_score_%s.json"                    # capacity/, lower-case arm
FMT_SCORE_RECORD = "inc2-v3fam-score-record/1"
FMT_REPORTED = "inc2-v3fam-reported/1"
FMT_SOURCES = "inc2-v3fam-reference-sources/1"

# D8 (V3F.5 D8, V3F.6.3 L23I): the builds' experiments, arms and the manifest they train
D8_ARMS = {"D8S": {"exp": "v3_d8_s640_tc", "arm": "s640"}, "D8M": {"exp": "v3_d8_m640_tc", "arm": "m640"}}
D8_ORDER = ("D8S", "D8M")
D8_MANIFEST = "train_core"                  # LOCK v2's
D8_FINALS = ("dev", "imageweeds")
D8_ROLE = "baseline"
D8_REPORTED_EXAMS = ("imageweeds", OODDEV)
D8_READ = "D8M"                              # the reference read (V3F.3.6); D8-s has none
FMT_D8 = "inc2-v3fam-build/1"


def log(msg):
    print("[inc2.v3fam_systems] %s" % msg, flush=True)


# ------------------------------------------------------------------ small helpers
def _utc():
    return TS._utc()


def _read_json(path):
    return TS._read_json(path)


def _sha(path):
    return TS._sha(path)


def supersede(groups=(SUP.GROUP_SCORES,)):
    """The records a recorded deviation supersedes (inc2.v3fam_supersede: deviation 5's identity_probe.json, taken
    under the CLI-only change of inc2/v3fam.py it reverts, and the ticker's failure records of 2026-10-22; deviation
    6's identity_ood23.json, taken while the loader's label cache held the stored size of ood23's MPO photos tagged
    6 or 8; D6's smoke record of 2026-10-21 is its own job's, group d6), moved aside byte for byte into their dated
    folders, each
    named in the folder's manifest, before any family score or verdict reads them. Every family score job (group
    scores) and the verdict's job (every group) run it first; it is a no-op once done. A move that would overwrite a
    file refuses the job."""
    try:
        return SUP.apply(C.INC_DIR, repo=C.REPO, log=log, groups=groups)
    except SUP.SupersedeRefused as e:
        raise F.V3famRefused(str(e))
    except OSError as e:                     # a full disk, a lost mount: the platform's, never the arm's
        raise F.V3famRefused("a superseded record could not be moved aside (%s: %s): it is left in place and "
                             "nothing is moved over another file" % (type(e).__name__, e))


def rel(path):
    """A path under v3fam/v1, as the records name it."""
    return str(Path(path).resolve().relative_to(F.root().resolve()))


def score_record_path(arm):
    return C.INC_DIR / "capacity" / (SCORE_RECORD % arm.lower())


def reported_path(arm):
    return F.root() / "reported" / ("%s.json" % arm)


def system_dir(dirname, seed=None, ens=None):
    """arms/<dir>/s<k> (a seed), or arms/<dir>/<ens> (a one-system arm, D3)."""
    if seed is None:
        if not ens:
            raise F.V3famRefused("a score dir is a seed's or a one-system arm's")
        return F.root() / "arms" / dirname / ens
    return F.arm_dir(dirname, seed)


def exam_rows(exam):
    """The exam's manifest rows sorted by key (the arrays' order): in this process's INC_DIR, or the root of
    ooddev_v1 and the probe (read without entering that root: image paths and keys only, never a label)."""
    if exam in F.ROOT_EXAMS:
        p = F.root_of(exam) / "splits" / "v1" / ("%s.jsonl" % exam)
    else:
        p = C.manifest_path(exam)
    if not p.is_file():
        raise F.V3famRefused("no manifest for %s at %s" % (exam, p))
    return sorted(C.read_manifest(p), key=lambda r: r["key"])


def image_sizes(rows):
    """{key: (W, H)} as each image is seen after EXIF transposition (twostage.header: the header only)."""
    out = {}
    for r in rows:
        try:
            out[r["key"]] = TS.header(r["image"])[1]
        except Exception as e:               # noqa: BLE001 - every exam image was hashed: a defect, refused
            raise F.V3famRefused("cannot read the header of %s: %s" % (r["image"], e))
    return out


# ------------------------------------------------------------------ a family experiment's runs
def run_source(exp, seed, testing_ok=False, read_dev=True):
    """What seed `seed` of a family experiment scores (twostage._source's checks for any experiment): the base
    run's final.pt (a regular file hashing as its run.json says), its final run carrying the same weights, that
    final run's recorded protocol dev score (production unless testing_ok), the research-only flag. read_dev
    False (D5b, Revision 30): the final run's scores/dev.json is never opened and no dev number is returned."""
    seed = int(seed)
    root_ = C.INC_DIR / exp
    defn = _read_json(root_ / "exp.json")
    if not isinstance(defn, dict):
        raise F.V3famRefused("%s has no exp.json" % exp)
    if seed not in [int(s) for s in defn.get("seeds") or []]:
        raise F.V3famRefused("%s has no seed %d" % (exp, seed))
    rid, frid = "base__s%d" % seed, "final__base__s%d" % seed
    rj = _read_json(root_ / "runs" / rid / "run.json")
    if not isinstance(rj, dict) or rj.get("status") != "done":
        raise F.V3famRefused("%s/%s is not done" % (exp, rid))
    w = root_ / "runs" / rid / "weights" / "final.pt"
    if w.is_symlink() or not w.is_file():
        raise F.V3famRefused("%s is missing or a symlink: a scored detector is its base run's own file" % w)
    wsha = C.sha256_file(w)
    if wsha != rj.get("weights_sha256"):
        raise F.V3famRefused("%s hashes to %s, its run.json records %s" % (w, wsha[:12],
                                                                         str(rj.get("weights_sha256"))[:12]))
    fj = _read_json(root_ / "runs" / frid / "run.json")
    if not isinstance(fj, dict) or fj.get("status") != "done" or fj.get("weights_sha256") != wsha:
        raise F.V3famRefused("%s/%s is not done or does not carry %s's weights" % (exp, frid, rid))
    ro = defn.get("research_only")
    flag = ro.get("flag") if isinstance(ro, dict) else ro
    out = {"exp": exp, "seed": seed, "run_id": rid, "final_run_id": frid, "weights": str(w.resolve()),
           "weights_sha256": wsha, "base_run_json_sha256": _sha(root_ / "runs" / rid / "run.json"),
           "final_run_json_sha256": _sha(root_ / "runs" / frid / "run.json"),
           "research_only": True if flag is None else bool(flag), "definition": defn}
    if not read_dev:
        return out
    rp = root_ / "runs" / frid / "scores" / "dev.json"
    rec = _read_json(rp)
    if not isinstance(rec, dict) or rec.get("exam") != "dev":
        raise F.V3famRefused("%s/%s has no recorded protocol dev score" % (exp, frid))
    if rec.get("production") is not True and not testing_ok:
        raise F.V3famRefused("%s is not a production score" % rp)
    if rec.get("weights_sha256") != wsha:
        raise F.V3famRefused("%s names weights %s, not the base run's" % (rp, str(rec.get("weights_sha256"))[:12]))
    out.update(recorded={"file": "%s/%s/scores/dev.json" % (exp, frid), "sha256": _sha(rp),
                         "species_map50_95": rec.get("species_map50_95"),
                         "agnostic_map50_95": rec.get("agnostic_map50_95"), "production": rec.get("production")},
               recorded_doc=rec)
    return out


def source_record(src):
    """The fields of a source a score file carries (no document, no definition)."""
    return {k: src[k] for k in ("exp", "seed", "run_id", "final_run_id", "weights_sha256", "base_run_json_sha256",
                                "final_run_json_sha256", "recorded", "research_only") if k in src}


def e3_source(letter, seed):
    """E3's stage-1 source of arm M, A or B (b_v2_m640, e1_a_m640, e1_b_m640) through twostage._source."""
    return TS._source(letter, seed, testing_ok=TS._testing_ok())


def vs_recorded(new, rec, what):
    """{diff, status} of a family identity score against a recorded locked score of the same weights; refused
    beyond the deviation band (the stamps are checked by inc2.v3fam.compare_with_recorded)."""
    cmp_ = F.compare_with_recorded(new, rec, what)
    st = F.repro_status(cmp_["max_abs_diff"])
    if st == "refused":
        raise F.V3famRefused("%s: the family identity score differs from the recorded one by %.4g (> %g): the "
                             "family path does not score these weights as the locked scorer did"
                             % (what, cmp_["max_abs_diff"], F.DEVIATION_TOL))
    return {"diff": cmp_, "status": st}


# ------------------------------------------------------------------ committing a family score
def commit(arm, dirname, seed, exam, ps, cfg, eq_names, variant="", classifier=None, extra=None, ens=None):
    """Write one family score once (inc2.v3fam.family_doc + commit_pass) under arms/<dir>/s<k> (or
    arms/<dir>/<ens>), resting on the named equivalence records. Returns its entry for the records."""
    eqs = F.require_equivalence(eq_names)
    x = {"system": SYSTEM_NAMES.get(dirname, dirname), "dir": dirname}
    x.update(extra or {})
    doc = F.family_doc(arm, seed, exam, ps, cfg, variant=variant, classifier=classifier, equivalence=eqs, extra=x)
    js = F.commit_pass(system_dir(dirname, seed, ens), exam, doc, ps, variant=variant)
    return entry(js, "written")


def entry(js, status):
    """A score file's entry: name under v3fam/v1, sha256s, and its dev numbers (dev files only)."""
    d = _read_json(js)
    if not isinstance(d, dict) or d.get("format") != F.FMT_SCORE:
        raise F.V3famRefused("%s is not a family score" % js)
    out = {"file": rel(js), "sha256": _sha(js), "images_sha256": (d.get("images") or {}).get("sha256"),
           "status": status, "dir": d.get("dir"), "seed": d.get("seed"), "exam": d.get("exam"),
           "variant": d.get("variant"), "system": d.get("system")}
    if d.get("exam") == "dev":
        out.update(species_map50_95=d.get("species_map50_95"), agnostic_map50_95=d.get("agnostic_map50_95"),
                   map50_95=d.get("map50_95"),
                   stage1_agnostic_map50_95=(d.get("stage1") or {}).get("agnostic_map50_95"))
    return out


def existing(dirname, seed, exam, variant="", ens=None):
    """The entry of a complete family score (kept on a rerun), or None."""
    js = system_dir(dirname, seed, ens) / F.file_names(exam, variant)[0]
    return entry(js, "kept") if js.exists() else None


def family_runs(dirname, exam, variant="", seeds=F.SEEDS, ens=None):
    """[(doc, per-image arrays)] of committed family scores, each JSON and its images npz checked against the
    sha256 the JSON records: what the verdict and D3 read."""
    out = []
    for s in ([None] if ens else seeds):
        d = system_dir(dirname, s, ens)
        jn, nn = F.file_names(exam, variant)[:2]
        doc = _read_json(d / jn)
        if not isinstance(doc, dict) or doc.get("format") != F.FMT_SCORE:
            raise F.V3famRefused("no family score at %s" % (d / jn))
        if _sha(d / nn) != (doc.get("images") or {}).get("sha256"):
            raise F.V3famRefused("%s does not hash as its score records" % (d / nn))
        out.append((doc, SC.load_npz(d / nn)))
    return out


def family_side(dirname, seed, exam, name, variant="", ens=None):
    """A committed score's side npz (stage1_images, rows or p7), checked against its JSON's sha256."""
    d = system_dir(dirname, seed, ens)
    doc = _read_json(d / F.file_names(exam, variant)[0])
    if not isinstance(doc, dict) or doc.get("format") != F.FMT_SCORE:
        raise F.V3famRefused("no family score at %s" % (d / F.file_names(exam, variant)[0]))
    rec = doc.get(name) or {}
    if not rec.get("name") or _sha(d / rec["name"]) != rec.get("sha256"):
        raise F.V3famRefused("%s's %s record is missing or does not hash as recorded" % (rel(d), name))
    return doc, SC.load_npz(d / rec["name"])


def score_record(arm, dev_files, equivalence, extra=None, reported=None, before=None):
    """capacity/v3fam_score_<arm>.json, the platform's record of a family score job: dev file names (relative to
    v3fam/v1), sha256s, status and counts; the dev equivalence records; the reported passes' counts only."""
    rec = {"format": FMT_SCORE_RECORD, "family": F.NAME, "arm": arm, "status": "complete", "exam": "dev",
           "dev": dev_files, "equivalence": {k: {"sha256": v["sha256"], "status": v["status"]}
                                             for k, v in sorted((equivalence or {}).items())},
           "reported": reported or {"status": "not run"},
           "e_trees": None if before is None else {"files": len(before), "sha256": C.sha256_text(
               json.dumps(before, sort_keys=True))},
           "decided_by": F.DECIDED_BY, "code": F.code_shas(), "written_utc": _utc(),
           "note": "dev only; names relative to v3fam/%s; non-dev passes in v3fam/%s/reported/%s.json" % (
               F.NAME, F.NAME, arm)}
    rec.update(extra or {})
    TS._write_json(score_record_path(arm), rec)
    return rec


def write_reported(arm, passes):
    """v3fam/v1/reported/<arm>.json: each reported pass's file, sha256 and status, or its error (overwritten).
    Returns the counts the platform's record carries."""
    TS._write_json(reported_path(arm), {"format": FMT_REPORTED, "arm": arm, "passes": passes,
                                        "written_utc": _utc()})
    return {"status": "complete", "passes": len(passes),
            "failed": sum(1 for p in passes if p.get("status") == "failed")}


def reported_pass(passes, item, fn):
    """Run one reported pass (fn() -> entry), recording a failure instead of raising (it never fails the job)."""
    try:
        e = fn()
        passes.append(dict(item, **{k: e[k] for k in ("file", "sha256", "status")}))
    except Exception as ex:                  # noqa: BLE001 - a reported pass never fails the job
        log("WARNING: reported pass %s failed: %s: %s" % (item, type(ex).__name__, ex))
        passes.append(dict(item, status="failed", error="%s: %s" % (type(ex).__name__, str(ex)[:400])))


def require_or_make_identity(exam, arm, kw):
    """The identity reproduction an exam other than dev rests on (V3F.3.2 item 3), taken if missing."""
    name = "identity_%s.json" % exam
    if not (F.equiv_dir() / name).exists():
        F.equivalence_identity(exam, arm, **kw)
    return F.require_equivalence([name])


def in_root(weights, exam, cfg, arm, kw):
    """A pass of another root (ooddev_v1), in a subprocess with INC_DIR there; the child writes its result into
    a temporary directory outside the root."""
    with tempfile.TemporaryDirectory(prefix="inc2_v3fam_root_") as tmp:
        return F.family_pass_in_root(weights, exam, cfg, arm, Path(tmp) / "out", **kw)


# ------------------------------------------------------------------ E3RS (V3F.5)
def _e3_compare(letter, seed, ps):
    """A family two-stage dev pass of E3 arm `letter` against E3's recorded dev file and its stored stage-1 box
    set (V3F.3.2 item 2's checks, for a pass taken after the reproduction record exists)."""
    d = TS.arm_dir(letter, seed)
    jn, _nn, sn = TS.file_names("dev")
    rec = _read_json(d / jn)
    if not isinstance(rec, dict) or rec.get("format") != TS.FMT_SCORE:
        raise F.V3famRefused("E3-%s seed %d has no recorded dev file" % (letter, seed))
    if _sha(d / sn) != (rec.get("stage1_images") or {}).get("sha256"):
        raise F.V3famRefused("E3-%s seed %d's dev.stage1.npz does not hash as its file records" % (letter, seed))
    s1 = F.stage1_compare(ps["stage1"], SC.load_npz(d / sn))
    cmp_ = F.compare_with_recorded(ps["result"], rec, "E3-%s seed %d dev" % (letter, seed))
    st = F.repro_status(cmp_["max_abs_diff"])
    if st == "refused":
        raise F.V3famRefused("E3-%s seed %d: the family two-stage pass differs from E3's file by %.4g (> %g)"
                             % (letter, seed, cmp_["max_abs_diff"], F.DEVIATION_TOL))
    return {"file": "twostage/%s/arms/%s/s%d/%s" % (TS.E3_NAME, letter, seed, jn), "sha256": _sha(d / jn),
            "species_map50_95": rec.get("species_map50_95"), "diff": cmp_, "status": st, "stage1": s1}


def _two_stage_cfg(ccfg, letter, embedder, wk, mode="two_stage"):
    return dict(ccfg, mode=mode, nms=TS.NMS[letter], embedder=embedder, workers=wk, top_k=TS.TOP_K)


def score_e3rs(embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None, reported=True):
    """E3RS (V3F.5; module docstring): the reproductions, the family-path dev arrays of E3-M, E3-A, E3-B and the
    reference, O5's oracle reading on E3-B, then the reported passes (E3-B on ImageWeeds; ooddev_v1 for E1-A, E1-B
    and b_v2_m640). Every score written once; capacity/v3fam_score_e3rs.json (dev only)."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        # the dev reproductions first (V3F.3.2): no family score is taken without them
        if not (F.equiv_dir() / "dev_identity.json").exists():
            F.equivalence_dev_identity(**kw)
        if not (F.equiv_dir() / "injection.json").exists():
            F.equivalence_injection(**kw)
        ccfg, _crec, shas = TS._clf_cfg()
        dev_files, eq_all = [], {}
        with TS._workers(workers) as wk:
            for letter in TS.ARM_ORDER:
                dirname = E3_DIRS[letter]
                eqn = "two_stage_%s.json" % letter
                passes = {}
                if not (F.equiv_dir() / eqn).exists():
                    embedder = embedder or V.BioclipEmbedder()
                    passes = F.equivalence_two_stage(letter, embedder, workers=wk, **kw)["passes"]
                names = F.equivalence_names("dev", e3_arms=(letter,))
                eq_all.update(F.require_equivalence(names))
                for s in F.SEEDS:
                    got = existing(dirname, s, "dev")
                    if got is not None:
                        dev_files.append(got)
                        continue
                    embedder = embedder or V.BioclipEmbedder()
                    cfg = _two_stage_cfg(ccfg, letter, embedder, wk)
                    ps = passes.get(s) or F._family_pass(e3_source(letter, s)["weights"], "dev", cfg, "E3RS", **kw)
                    src = e3_source(letter, s)
                    e3 = _e3_compare(letter, s, ps)
                    dev_files.append(commit("E3RS", dirname, s, "dev", ps, cfg, names, classifier=shas,
                                            extra={"e3_arm": letter, "stage1_source": source_record(src),
                                                   "nms": TS.NMS[letter], "e3": e3}))
                    log("E3-%s seed %d on dev through the family path: species %.4f (E3's file %.4f, %s)"
                        % (letter, s, ps["result"]["species_map50_95"] or 0.0, e3["species_map50_95"] or 0.0,
                           e3["status"]))
            # the reference, identity (both sides of every dev comparison on one path)
            names = F.equivalence_names("dev")
            eq_all.update(F.require_equivalence(names))
            for s in F.SEEDS:
                got = existing(REF_DIR, s, "dev")
                if got is not None:
                    dev_files.append(got)
                    continue
                src = e3_source("M", s)
                cfg = {"mode": "identity", "nms": "locked"}
                ps = F._family_pass(src["weights"], "dev", cfg, "E3RS", **kw)
                vr = vs_recorded(ps["result"], src["recorded_doc"], "b_v2_m640 seed %d dev" % s)
                dev_files.append(commit("E3RS", REF_DIR, s, "dev", ps, cfg, names,
                                        extra={"source": source_record(src), "vs_recorded": vr}))
            # O5: E3-B's boxes with perfect naming (V3F.5 D2, computed here)
            names = F.equivalence_names("dev", e3_arms=("B",))
            for s in F.SEEDS:
                got = existing(E3_DIRS["B"], s, "dev", ORACLE_VARIANT)
                if got is not None:
                    dev_files.append(got)
                    continue
                embedder = embedder or V.BioclipEmbedder()
                cfg = _two_stage_cfg(ccfg, "B", embedder, wk, mode="oracle")
                src = e3_source("B", s)
                ps = F._family_pass(src["weights"], "dev", cfg, "E3RS", **kw)
                stored = SC.load_npz(TS.arm_dir("B", s) / TS.file_names("dev")[2])
                s1 = F.stage1_compare(ps["stage1"], stored)
                dev_files.append(commit("E3RS", E3_DIRS["B"], s, "dev", ps, cfg, names, variant=ORACLE_VARIANT,
                                        classifier=shas, extra={"e3_arm": "B", "stage1_source": source_record(src),
                                                                "stage1_vs_e3": s1, "reading": "O5 oracle"}))
            rec = score_record("E3RS", dev_files, eq_all, before=before,
                               reported={"status": "pending" if reported else "not run"})
            if reported:
                rep = _e3rs_reported(embedder, ccfg, shas, wk, kw)
                rec = score_record("E3RS", dev_files, eq_all, before=before, reported=rep)
    log("E3RS: %d dev files (%s), %.0fs" % (len(dev_files), ", ".join(sorted(set(
        "%s %s" % (f["dir"], f["status"]) for f in dev_files))), time.time() - t0))
    return rec


def _e3rs_reported(embedder, ccfg, shas, wk, kw):
    """E3-B two-stage on ImageWeeds (3) and ooddev_v1 for E1-A, E1-B and b_v2_m640 (9, in the zoo's root), each
    written once; the identity reproduction of each exam first. A failure is recorded and never raised."""
    passes = []
    # ImageWeeds: E3-B's three seeds two-stage (the comparator row; the stage-1 rows D3 fuses)
    try:
        require_or_make_identity("imageweeds", "E3RS", kw)
        iw_ok = True
    except Exception as e:                   # noqa: BLE001 - a reported exam never fails the job
        log("WARNING: the ImageWeeds identity reproduction failed: %s: %s" % (type(e).__name__, e))
        passes.append({"exam": "imageweeds", "item": "identity_imageweeds.json", "status": "failed",
                       "error": "%s: %s" % (type(e).__name__, str(e)[:400])})
        iw_ok = False
    names = F.equivalence_names("imageweeds", e3_arms=("B",))
    for s in F.SEEDS:
        item = {"dir": E3_DIRS["B"], "seed": s, "exam": "imageweeds"}
        if not iw_ok:
            passes.append(dict(item, status="failed", error="the ImageWeeds identity reproduction failed"))
            continue
        got = existing(E3_DIRS["B"], s, "imageweeds")
        if got is not None:
            passes.append(dict(item, file=got["file"], sha256=got["sha256"], status="kept"))
            continue

        def one(s=s):
            emb = embedder or V.BioclipEmbedder()
            cfg = _two_stage_cfg(ccfg, "B", emb, wk)
            src = e3_source("B", s)
            ps = F._family_pass(src["weights"], "imageweeds", cfg, "E3RS", **kw)
            e3doc = _read_json(TS.arm_dir("B", s) / TS.file_names("imageweeds")[0])
            beside = None
            if isinstance(e3doc, dict):
                try:
                    beside = F.compare_with_recorded(ps["result"], e3doc, "E3-B seed %d imageweeds" % s)
                except F.V3famRefused as e:
                    beside = {"compared": False, "why": str(e)}
            return commit("E3RS", E3_DIRS["B"], s, "imageweeds", ps, cfg, names, classifier=shas,
                          extra={"e3_arm": "B", "stage1_source": source_record(src), "vs_e3_file": beside})
        reported_pass(passes, item, one)
    # ooddev_v1 (agnostic, reported): E1-A's and E1-B's box sets (stage1 mode) and b_v2_m640 (identity)
    try:
        require_or_make_identity(OODDEV, "E3RS", kw)
        od_ok = True
    except Exception as e:                   # noqa: BLE001
        log("WARNING: the ooddev_v1 identity reproduction failed: %s: %s" % (type(e).__name__, e))
        passes.append({"exam": OODDEV, "item": "identity_%s.json" % OODDEV, "status": "failed",
                       "error": "%s: %s" % (type(e).__name__, str(e)[:400])})
        od_ok = False
    names = F.equivalence_names(OODDEV)
    plan = [(E1_DIRS["A"], "A", "stage1"), (E1_DIRS["B"], "B", "stage1"), (REF_DIR, "M", "identity")]
    for dirname, letter, mode in plan:
        for s in F.SEEDS:
            item = {"dir": dirname, "seed": s, "exam": OODDEV}
            if not od_ok:
                passes.append(dict(item, status="failed", error="the ooddev_v1 identity reproduction failed"))
                continue
            got = existing(dirname, s, OODDEV)
            if got is not None:
                passes.append(dict(item, file=got["file"], sha256=got["sha256"], status="kept"))
                continue

            def one(dirname=dirname, letter=letter, mode=mode, s=s):
                src = e3_source(letter, s)
                cfg = {"mode": mode, "nms": "locked"}
                ps = in_root(src["weights"], OODDEV, cfg, "E3RS", kw)
                return commit("E3RS", dirname, s, OODDEV, ps, cfg, names,
                              extra={"source": source_record(src), "statistic": "agnostic (weed boxes only)"})
            reported_pass(passes, item, one)
    return write_reported("E3RS", passes)


# ------------------------------------------------------------------ D8 (V3F.5)
def d8_arm(arm):
    if arm not in D8_ARMS:
        raise F.V3famRefused("D8's builds are %s, not %r" % (list(D8_ORDER), arm))
    return D8_ARMS[arm]


def d8_manifest():
    """(path, sha256) of LOCK v2's train_core, checked against the lock."""
    try:
        sha = C2.verify_manifest_against_lock_v2(D8_MANIFEST)
    except C2.Inc2Error as e:
        raise F.V3famRefused("LOCK v2's %s: %s" % (D8_MANIFEST, e))
    return C2.v2_manifest_path(D8_MANIFEST), sha


def d8_record(arm, sha):
    return {"format": FMT_D8, "family": F.NAME, "arm": arm, "experiment": "D8", "manifest": D8_MANIFEST,
            "manifest_sha256": sha, "locked": "LOCK v2", "recipe": RC.COLD_NAME,
            "decided_by": F.DECIDED_BY}


def d8_build_kwargs(arm, exp=None, seeds=None, role=None, final_exams=None):
    """inc2.baseline build's keyword arguments for a D8 build (L23I: --exp <the arm's> --v3fam D8S|D8M --seeds
    0,1,2 --role baseline): LOCK v2's train_core by sha256, the arm (s640, m640), seeds 0-2, role baseline,
    finals dev and ImageWeeds, the cold table (no cold_budget), exp.json's v3fam record. Anything else refuses."""
    a = d8_arm(arm)
    if exp is not None and exp != a["exp"]:
        raise F.V3famRefused("--v3fam %s builds %s, not %s" % (arm, a["exp"], exp))
    if seeds is not None and [int(s) for s in (seeds.split(",") if isinstance(seeds, str) else seeds)] \
            != list(F.SEEDS):
        raise F.V3famRefused("D8 trains seeds %s" % list(F.SEEDS))
    if role not in (None, D8_ROLE):
        raise F.V3famRefused("D8 builds role %s, not %s" % (D8_ROLE, role))
    if final_exams is not None and list(final_exams) != list(D8_FINALS):
        raise F.V3famRefused("D8's finals are %s (no test: its read is the family's, V3F.3.6)" % list(D8_FINALS))
    path, sha = d8_manifest()
    return {"exp": a["exp"], "manifest": str(path), "seeds": list(F.SEEDS), "arm": a["arm"],
            "final_exams": list(D8_FINALS), "role": D8_ROLE, "extra": {"v3fam": d8_record(arm, sha)}}


def build_d8(arm, testing=False, init=True, backend=None, quiet=False):
    """One D8 build through inc2.baseline.build with d8_build_kwargs (the platform's argv is inc2.baseline build
    --exp E --v3fam ARM ...). Returns (summary, definition, init result)."""
    kw = d8_build_kwargs(arm)
    exp = kw.pop("exp")
    try:
        return B.build(exp, testing=testing, init=init, backend=backend, quiet=quiet, **kw)
    except B.BaselineError as e:
        raise F.V3famRefused("inc2.baseline refuses the D8 build: %s" % e)


def check_d8_definition(arm, testing_ok=False):
    """Refuse unless the D8 experiment of `arm` was built as pre-registered: exp.json's base manifest LOCK v2's
    train_core by sha256, its arm, role baseline, finals dev and ImageWeeds, the cold recipe, seeds 0-2 and the
    v3fam record. Returns the definition."""
    a = d8_arm(arm)
    defn = _read_json(C.INC_DIR / a["exp"] / "exp.json")
    if not isinstance(defn, dict):
        raise F.V3famRefused("%s has no exp.json: D8's build has not run" % a["exp"])
    _path, sha = d8_manifest()
    probs = []
    base = defn.get("base") or {}
    if base.get("manifest_sha256") != sha:
        probs.append("its base manifest is not LOCK v2's train_core")
    if (defn.get("arm") or {}).get("id") != a["arm"]:
        probs.append("arm %r, not %s" % ((defn.get("arm") or {}).get("id"), a["arm"]))
    if defn.get("role") != D8_ROLE:
        probs.append("role %r" % defn.get("role"))
    if list(defn.get("final_exams") or []) != list(D8_FINALS):
        probs.append("finals %s" % defn.get("final_exams"))
    if [int(s) for s in defn.get("seeds") or []] != list(F.SEEDS):
        probs.append("seeds %s" % defn.get("seeds"))
    if defn.get("recipe_name") not in (None, RC.COLD_NAME) or base.get("recipe") != RC.cold(a["arm"]):
        probs.append("not the cold table's recipe")
    v = defn.get("v3fam") or {}
    if v.get("arm") != arm or v.get("manifest_sha256") != sha:
        probs.append("no v3fam record naming %s" % arm)
    if defn.get("testing") and not testing_ok:
        probs.append("a testing build")
    if probs:
        raise F.V3famRefused("%s is not D8's pre-registered build: %s" % (a["exp"], "; ".join(probs)))
    return defn


def score_d8(lock_check=True, batch=S.BATCH, device=None, reported=True):
    """D8's scoring job (L23J --arm D8): each seed of D8-s and D8-m through the family identity pass on dev
    (compared with the run's recorded protocol dev score), then ImageWeeds and ooddev_v1 reported."""
    F.check_frozen()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    testing = TS._testing_ok()
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        srcs = {}
        for arm in D8_ORDER:
            check_d8_definition(arm, testing_ok=testing)
            srcs[arm] = {s: run_source(D8_ARMS[arm]["exp"], s, testing_ok=testing) for s in F.SEEDS}
        names = F.equivalence_names("dev")
        eqs = F.require_equivalence(names)
        dev_files = []
        cfg = {"mode": "identity", "nms": "locked"}
        for arm in D8_ORDER:
            for s in F.SEEDS:
                got = existing(arm, s, "dev")
                if got is not None:
                    dev_files.append(got)
                    continue
                src = srcs[arm][s]
                ps = F._family_pass(src["weights"], "dev", cfg, "D8", **kw)
                vr = vs_recorded(ps["result"], src["recorded_doc"], "%s seed %d dev" % (arm, s))
                dev_files.append(commit("D8", arm, s, "dev", ps, cfg, names,
                                        extra={"source": source_record(src), "vs_recorded": vr,
                                               "d8_arm": D8_ARMS[arm]["arm"]}))
                log("%s seed %d on dev: species %.4f (recorded %s, %s)" % (
                    arm, s, ps["result"]["species_map50_95"] or 0.0, src["recorded"]["species_map50_95"],
                    vr["status"]))
        rec = score_record("D8", dev_files, eqs, before=before,
                           reported={"status": "pending" if reported else "not run"})
        if reported:
            passes = []
            for exam in D8_REPORTED_EXAMS:
                try:
                    require_or_make_identity(exam, "D8", kw)
                    ok, why = True, None
                except Exception as e:       # noqa: BLE001 - a reported exam never fails the job
                    ok, why = False, "%s: %s" % (type(e).__name__, str(e)[:400])
                    passes.append({"exam": exam, "item": "identity_%s.json" % exam, "status": "failed",
                                   "error": why})
                ex_names = F.equivalence_names(exam)
                for arm in D8_ORDER:
                    for s in F.SEEDS:
                        item = {"dir": arm, "seed": s, "exam": exam}
                        if not ok:
                            passes.append(dict(item, status="failed", error="the %s identity reproduction failed"
                                               % exam))
                            continue
                        got = existing(arm, s, exam)
                        if got is not None:
                            passes.append(dict(item, file=got["file"], sha256=got["sha256"], status="kept"))
                            continue

                        def one(arm=arm, s=s, exam=exam, ex_names=ex_names):
                            src = srcs[arm][s]
                            if exam in F.ROOT_EXAMS:
                                ps = in_root(src["weights"], exam, cfg, "D8", kw)
                            else:
                                ps = F._family_pass(src["weights"], exam, cfg, "D8", **kw)
                            return commit("D8", arm, s, exam, ps, cfg, ex_names,
                                          extra={"source": source_record(src), "d8_arm": D8_ARMS[arm]["arm"]})
                        reported_pass(passes, item, one)
            rec = score_record("D8", dev_files, eqs, before=before, reported=write_reported("D8", passes))
    log("D8: %d dev files, %.0fs" % (len(dev_files), time.time() - t0))
    return rec


# ------------------------------------------------------------------ the reference reads' inputs (V3F.3.6)
def reference_sources(arm):
    """What a reference read of `arm` scores (V3F.3.6: D6 and D8-m; score-test pins these by sha256): D8-m's
    three base runs (identity), D6's three exported final EMA weights (injected rows on b_v2_m640 seed 0's
    scaffold). Every other arm refuses."""
    testing = TS._testing_ok()
    if arm == D8_READ:
        check_d8_definition(arm, testing_ok=testing)
        seeds = [source_record(run_source(D8_ARMS[arm]["exp"], s, testing_ok=testing)) for s in F.SEEDS]
        return {"format": FMT_SOURCES, "arm": arm, "mode": "identity", "exp": D8_ARMS[arm]["exp"], "seeds": seeds}
    if arm == "D6":
        from . import v3fam_detr as DT
        return dict(DT.read_sources(), format=FMT_SOURCES, arm=arm)
    raise F.V3famRefused("%s has no reference read: the reference reads are %s (V3F.3.6)"
                         % (arm, list(F.REFERENCE_READS)))


# ------------------------------------------------------------------ dispatch
def score_arm(arm, embedder=None, lock_check=True, batch=S.BATCH, device=None, workers=None, reported=True):
    """inc2.v3fam score --arm X for this module's arms (E3RS, D3, D6, D8)."""
    if arm not in SYSTEM_ARMS:
        raise F.V3famRefused("--arm %r is not one of %s" % (arm, list(SYSTEM_ARMS)))
    supersede()                              # deviation 5: the records it supersedes moved aside first
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    if arm == "E3RS":
        return score_e3rs(embedder=embedder, workers=workers, reported=reported, **kw)
    if arm == "D8":
        return score_d8(reported=reported, **kw)
    if arm == "D3":
        from . import v3fam_wbf as WB
        return WB.score_d3(embedder=embedder, workers=workers, reported=reported, **kw)
    from . import v3fam_detr as DT
    return DT.score_d6(reported=reported, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F's system arms: E3RS, D3, D6, D8 (scores) and D8's builds.")
    ap.add_argument("command", choices=("score", "build", "sources"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--v3fam", default=None, help="D8S or D8M (build)")
    ap.add_argument("--no-init", action="store_true", help="build: write the definition only (no driver init)")
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.v3fam_systems] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    kw = {"lock_check": not a.no_lock_check_for_tests, "batch": a.batch or S.BATCH, "device": a.device}
    try:
        if a.command == "score":
            score_arm(a.arm, **kw)
        elif a.command == "build":
            build_d8(a.v3fam, testing=S.testing(), init=not a.no_init)
        else:
            print(json.dumps(reference_sources(a.arm), indent=1, sort_keys=True, default=str))
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, SN.NativeRefused, V.VerifyError,
            C2.Inc2Error, B.BaselineError, ValueError) as e:
        print("[inc2.v3fam_systems] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
