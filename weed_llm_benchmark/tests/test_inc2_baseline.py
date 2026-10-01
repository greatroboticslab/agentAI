#!/usr/bin/env python3
"""Splits v2 baselines (inc2/baseline.py): B_v2, the capacity arms (L-4), the
canary, B0 u tsw, milestones, and the verdicts that read them
(docs/CONTINUOUS_LOOP.md §4.4, §3.6, §2.6 L-4, §9 group B).

The synthetic world of tests/test_inc2_train.py (v1 and v2 splits, the real
inc2.guard, LOCK v2 marked testing).

What is pinned:
- every build's definition passes the pinned driver's validate_definition
  and check_definition_data, and carries protocol v3 / inc2 / splits v2, the
  arm record with its checkpoint's sha256, init_weights = the arm's
  checkpoint, the arm's Protocol v3 cold recipe (imgsz 640 on every grid
  arm), the role's seeds and final exams, source_locked (the LOCK v2 name of
  the manifest), and an est. cost scaled by the arm's FLOPs and pixels;
- the measurement arms (2026-09-30): m832 (yolo11m.pt, 832 px, batch 16) and
  s1024 (yolo11s.pt, 1024 px, batch 32) resolve with their checkpoint's
  sha256, their GFLOPs are the 640 figure x (imgsz / 640)^2, their cost
  scales by those FLOPs and the pixel ratio (scoring by the 640 FLOPs), a
  recipe at another batch deviates, the grid's recipes are unchanged; built
  on base_v2 without --role they are capacity builds (3 seeds, finals dev,
  imageweeds: built after R0, at no milestone read, they never read test,
  P10), by --arm or by --arch/--imgsz, and a build of one that lists test is
  refused;
- union: the parts' rows in one manifest, each part recorded; overlapping
  parts are refused, and so is a part holding an hflip copy of a test image
  (the guard runs over the union);
- roles and test reads (P10): without --role the role is inferred (LOCK v2's
  base_v2 is b_v2 on n640 and capacity on another arm, train_core is canary,
  anything else baseline, --union union); only b_v2, capacity and milestone
  read test by default or at all; an explicit role that does not match the
  manifest or the arm is refused;
- the autopilot's L23B argv (--arch/--imgsz, no --role): yolo11s at 640 is
  s640 and a capacity build with test; the canary argv reads dev only; an
  arch/imgsz pair outside the table and an --arm that disagrees are refused;
- research_only fails closed: a row outside the provenance, or a provenance
  file that does not hash as LOCK v2 records, makes the flag true;
- refusals before anything is written: an evaluation manifest (v1 or v2), a
  manifest holding an hflip copy of a test image, an ood final exam, both or
  neither of --manifest / --union, an unknown role; a production build with a
  missing arm checkpoint or a testing LOCK v2; INC_JOB_SCRIPT naming another
  script; a second build of a built experiment;
- end to end (FakeBackend, the real inc2.train on the CPU with the recipe table
  shrunk to 1 epoch at imgsz 64): driver init and advance of a baseline run
  its base and final runs to done, every submission carries INC_JOB_SCRIPT =
  run_inc2_job.sh and the SlurmBackend's sbatch argv names it; a 1-step chain
  built the way inc2.stream builds one (recipes.stamp, r0, gate net) is driven
  to done by a runner that checks every spec with inc2.train and writes
  scores; again only run_inc2_job.sh;
- canary-verdict: within one b0_v1 sd passes, outside fails; a test-mode
  run or score, another base manifest or recipe than the reference's, and a
  missing reference exp.json do not pass;
- L-8: the world's v1 train_core holds one row v2 drops; the canary trains
  v2's train_core (v1's minus that row), its exp.json and build summary
  record the dropped row and the list's sha256, and a manifest that is not
  v1's minus the list is refused; its verdict accepts the reduced manifest
  by the reference's sha256 plus the list's sha256 (canary.json names the
  dropped row), and refuses a reduced manifest with no recorded drops,
  drops from another reference, another list than LOCK v2's, or a list
  changed since the build;
- capacity-verdict: an arm beating n640 by more than 2 pooled sd is chosen,
  none beating it keeps n640; truth_every follows the measured rate; the
  decision is identical when every test score is perturbed (test-blind) and
  refuses mixed scorers or a missing arm; the report carries test and the
  gap to 0.90; a measurement arm is refused as a candidate and a grid arm as
  a record; --record lists the measurement arms (dev against n640, finals in
  the report) and leaves the decision what the grid alone gives, even when a
  measurement arm has the best dev mean or an extra seed; a record arm that
  lacks one of the grid's seeds or was trained on another manifest is
  refused; an existing decision file is kept byte for byte, and one that
  disagrees refuses the record-only run;
- secondary: a final spec the v2 executor accepts, the milestone's exams, the
  run_inc2_job.sh argv, the log dir its --output names; a driver run id or
  missing weights refuse;
- estimate prints the cost.

Run:  python3 tests/test_inc2_baseline.py
"""
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO to its own temporary dirs first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402

FAILURES = W.FAILURES
check = W.check
V2 = W.v2_dir


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (B.BaselineError, D.DriverError, RC.RecipeError) as e:
        return e
    return None


@contextlib.contextmanager
def tiny_table():
    """The recipe table shrunk to what a CPU trains in seconds (1 epoch at imgsz
    64, no workers, no cache); restored after."""
    saved = (dict(RC.COMMON), dict(RC.COLD), dict(RC.ARMS["n640"]))
    RC.COMMON.update(workers=0, cache=False, close_mosaic=0, batch=8)
    RC.COLD.update(epochs=1, warmup_epochs=1)
    RC.ARMS["n640"]["imgsz"] = 64
    try:
        yield
    finally:
        RC.COMMON.clear()
        RC.COMMON.update(saved[0])
        RC.COLD.clear()
        RC.COLD.update(saved[1])
        RC.ARMS["n640"].clear()
        RC.ARMS["n640"].update(saved[2])


class RecordingBackend(D.FakeBackend):
    """FakeBackend that records the INC_JOB_SCRIPT every submission carries and
    the argv SlurmBackend would run."""

    def __init__(self, runner=None):
        super().__init__(runner=runner)
        self.scripts, self.argvs = [], []

    def submit(self, list_file, n, exp, job_name, log_dir, env=None, time_limit=None):
        self.scripts.append((env or {}).get("INC_JOB_SCRIPT"))
        self.argvs.append(D.SlurmBackend(user="u").sbatch_argv(list_file, n, exp, job_name, log_dir, time_limit))
        return super().submit(list_file, n, exp, job_name, log_dir, env=env, time_limit=time_limit)


def drive(exp, backend, max_rounds=30):
    for _ in range(max_rounds):
        backend.run_pending()
        res = D.advance(exp, backend=backend, quiet=True)
        if res["done"]:
            return res
    return res


def executor_runner(spec):
    rc = T.execute(pathlib.Path(spec["out_dir"]) / "spec.json")
    return "COMPLETED" if rc == 0 else "FAILED"


# ------------------------------------------------------------------ tests
def test_builds(Wd):
    print("builds (no init)")
    for arm, model in (("s640", "yolo11s.pt"), ("m640", "yolo11m.pt")):
        (C.REPO / model).write_bytes(os.urandom(4096))
    base_v2 = V2() / "base_v2.jsonl"
    prov = [{"key": r["key"], "sha256": r["sha256"], "research_only": r["key"].startswith("tsw22")}
            for r in Wd["base_v2"]]
    (V2() / "base_v2_provenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in prov))
    defn, summ = B.build_definition("b_v2", manifest=base_v2, role="b_v2", testing=True)
    D.validate_definition(json.loads(json.dumps(defn)))
    D.check_definition_data(defn)
    check("b_v2: a baseline the pinned driver accepts; protocol v3, inc2, splits v2; arm n640 with the "
          "checkpoint's sha256; seeds 0-4; finals dev, imageweeds, test; source_locked base_v2",
          defn["type"] == "baseline" and defn["protocol"] == "v3" and defn["protocol_package"] == "inc2"
          and defn["splits_version"] == "v2" and defn["arm"]["id"] == "n640"
          and defn["arm"]["weights_sha256"] == W.sha(C.REPO / "yolo11n.pt") and defn["init_weights"] == "yolo11n.pt"
          and defn["seeds"] == [0, 1, 2, 3, 4] and defn["final_exams"] == ["dev", "imageweeds", "test"]
          and defn["base"]["source_locked"] == "base_v2" and defn["base"]["recipe"] == RC.cold("n640")
          and defn["base"]["n_images"] == len(Wd["base_v2"]), {k: defn.get(k) for k in ("arm", "seeds")})
    check("... the manifest copied byte for byte, the summary with the guard record and every input's sha256",
          W.sha(defn["base"]["manifest"]) == W.sha(base_v2) and summ["manifest"]["guard"]["refused"] == 0
          and summ["splits_v2"]["lock_sha256"] == W.sha(V2() / "LOCK.json") and summ["code"]["tools/inc2/train.py"])
    c = defn["cost_estimate"]
    check("... cost est. from the recorded n640 rates (per run %s GPU-h)" % c["per_run_gpu_h"],
          c["estimate"] and c["arm"] == "n640" and c["seeds"] == 5 and c["flops_factor"] == 1.0)
    s, _ = B.build_definition("b_v2_s640", manifest=base_v2, role="capacity", arm="s640", testing=True)
    m, _ = B.build_definition("b_v2_m640", manifest=base_v2, role="capacity", arm="m640", testing=True)
    check("capacity arms: s640 trains yolo11s.pt at 640, m640 yolo11m.pt at 640, 3 seeds each, pinned shas",
          s["init_weights"] == "yolo11s.pt" and s["base"]["recipe"]["imgsz"] == 640 and s["seeds"] == [0, 1, 2]
          and m["init_weights"] == "yolo11m.pt" and m["base"]["recipe"]["imgsz"] == 640
          and s["arm"]["weights_sha256"] == W.sha(C.REPO / "yolo11s.pt")
          and s["base"]["recipe"] == RC.cold("s640") and not RC.deviations("base", s["base"]["recipe"], "s640"))
    check("... their cost scales by FLOPs (high) and pixels (low, 1 at 640): s640 x%.3f / x%.2f, m640 x%.3f"
          % (s["cost_estimate"]["flops_factor"], s["cost_estimate"]["pixel_factor"], m["cost_estimate"]["flops_factor"]),
          abs(s["cost_estimate"]["flops_factor"] - 3.3427) < 1e-3 and s["cost_estimate"]["pixel_factor"] == 1.0
          and abs(m["cost_estimate"]["flops_factor"] - 10.5733) < 1e-3 and m["cost_estimate"]["pixel_factor"] == 1.0)
    can, can_summ = B.build_definition("canary_v2", manifest=V2() / "train_core.jsonl", role="canary", testing=True)
    check("canary: seed 0 on the v2 train_core copy, final exam dev only",
          can["seeds"] == [0] and can["final_exams"] == ["dev"] and can["base"]["source_locked"] == "train_core")
    vd, drop = can.get("variant_drops") or {}, Wd["variant_drops"][0]
    check("L-8: the canary trains v1's train_core (b0_v1's base) minus the recorded drop: %d of %d images"
          % (can["base"]["n_images"], len(Wd["train_core_v1"])),
          can["base"]["n_images"] == len(Wd["train_core_v1"]) - 1 == len(Wd["train_core"])
          and vd.get("reference_manifest_sha256") == W.sha(C.manifest_path("train_core"))
          and vd.get("manifest_sha256") == can["base"]["manifest_sha256"] != vd.get("reference_manifest_sha256"), vd)
    check("... its exp.json records which rows were dropped (key, image sha256, match) and the list by the sha256 "
          "LOCK v2 records; build_summary.json too",
          vd.get("count") == 1 and [r["key"] for r in vd["rows"]] == [drop["key"]]
          and vd["rows"][0]["sha256"] == drop["sha256"] and vd["rows"][0]["match"]["variant"] == "transverse"
          and vd["sha256"] == W.sha(V2() / "train_core_variant_drops.jsonl")
          == json.loads((V2() / "LOCK.json").read_text())["train_core_variant_drops_sha256"]
          and vd["decided_by"].startswith("L-8") and can_summ.get("variant_drops") == vd, vd)
    check("... only the canary records them", "variant_drops" not in defn and "variant_drops" not in s)
    bad_tc = W.write_manifest("scratch", "train_core_bad", Wd["train_core"][1:])
    e = refused(B.variant_drops_record, W.sha(bad_tc), production=False)
    check("the canary's L-8 record refuses a manifest that is not v1's train_core minus the listed rows",
          e is not None and "would not reproduce b0_v1" in str(e), e)
    mixed = W.write_manifest("scratch", "mixed", Wd["train_core"] + Wd["inc"])
    mx, _ = B.build_definition("b_mixed", manifest=mixed, testing=True)
    check("research_only (8) from base_v2's provenance: b_v2 true (its tsw22 rows), the canary false (every "
          "train_core row known and clear), a base with rows outside the provenance true (fail closed)",
          defn["research_only"]["flag"] is True and defn["research_only"]["research_only_rows"] == len(Wd["tsw22"])
          and can["research_only"]["flag"] is False and mx["research_only"]["flag"] is True
          and mx["research_only"]["basis"] == "rows of unknown licence"
          and mx["research_only"]["unknown_rows"] == len(Wd["inc"]), (defn["research_only"], mx["research_only"]))
    lock_p = V2() / "LOCK.json"
    saved_lock = lock_p.read_bytes()
    lk = json.loads(saved_lock)
    lk["provenance_sha256"] = "0" * 64
    lock_p.write_text(json.dumps(lk, sort_keys=True))
    try:
        ro = B.research_only_record(Wd["train_core"])
    finally:
        lock_p.write_bytes(saved_lock)
    check("a provenance file that does not hash as LOCK v2 records is not read: every row unknown, flag true",
          ro["flag"] is True and ro["unknown_rows"] == len(Wd["train_core"]) and "LOCK v2" in ro["provenance"]["problem"],
          ro)
    check("roles: b_v2, capacity and canary were built as asked; b_mixed (no role) is a baseline without test",
          defn["role"] == "b_v2" and s["role"] == "capacity" and can["role"] == "canary"
          and mx["role"] == "baseline" and mx["final_exams"] == ["dev", "imageweeds"], (mx["role"], mx["final_exams"]))
    inf_b, _ = B.build_definition("inf_b", manifest=base_v2, seeds="0,1,2,3,4", testing=True)
    inf_s, _ = B.build_definition("inf_s", manifest=base_v2, seeds="0,1,2", arch="yolo11s", imgsz=640, testing=True)
    inf_c, _ = B.build_definition("inf_c", manifest=V2() / "train_core.jsonl", seeds="0", testing=True)
    check("no --role: base_v2 on n640 is b_v2 (test read), yolo11s at 640 on base_v2 is capacity s640 (test "
          "read), train_core is the canary (dev only)",
          inf_b["role"] == "b_v2" and inf_b["final_exams"] == ["dev", "imageweeds", "test"]
          and inf_s["role"] == "capacity" and inf_s["arm"]["id"] == "s640" and inf_s["base"]["recipe"]["imgsz"] == 640
          and inf_s["final_exams"] == ["dev", "imageweeds", "test"]
          and inf_c["role"] == "canary" and inf_c["final_exams"] == ["dev"] and inf_c["seeds"] == [0],
          [(d["role"], d["arm"]["id"], d["final_exams"]) for d in (inf_b, inf_s, inf_c)])
    tc = V2() / "train_core.jsonl"
    role_cases = {
        "a baseline that lists test (P10)": dict(manifest=mixed, final_exams=["dev", "test"]),
        "a canary that lists test (P10)": dict(manifest=tc, role="canary", final_exams=["dev", "test"]),
        "a union that lists test (P10)": dict(union=[tc, V2() / "tsw22.jsonl"], final_exams=["dev", "test"]),
        "role canary on base_v2": dict(manifest=base_v2, role="canary"),
        "role b_v2 on train_core": dict(manifest=tc, role="b_v2"),
        "role b_v2 on another arm": dict(manifest=base_v2, role="b_v2", arm="m640"),
        "role capacity on n640": dict(manifest=base_v2, role="capacity"),
        "role capacity on a manifest that is not base_v2": dict(manifest=mixed, role="capacity", arm="s640"),
        "role union without --union": dict(manifest=mixed, role="union"),
        "--union with role b_v2": dict(union=[tc, V2() / "tsw22.jsonl"], role="b_v2"),
        "--arch without --imgsz": dict(manifest=base_v2, arch="yolo11s"),
        "an arch/imgsz pair outside the table": dict(manifest=base_v2, arch="yolo11x", imgsz=640),
        "yolo11s at 832 (a detector of the table at an imgsz no arm has)":
            dict(manifest=base_v2, arch="yolo11s", imgsz=832),
        "yolo11m at 1024 (the measurement arms are m832 and s1024 only)":
            dict(manifest=base_v2, arch="yolo11m", imgsz=1024),
        "--arm and --arch that disagree": dict(manifest=base_v2, arm="m640", arch="yolo11s", imgsz=640)}
    for what, kw in role_cases.items():
        e = refused(B.build_definition, "b_role", testing=True, **kw)
        check("build refuses %s" % what, e is not None, e)
    check("... and none of them wrote an exp.json", not (C.INC_DIR / "b_role" / "exp.json").exists())
    un, us = B.build_definition("b0tsw_v2", union=[V2() / "train_core.jsonl", V2() / "tsw22.jsonl"], role="union",
                                testing=True)
    check("union: train_core u tsw22 in one manifest, both parts recorded with their LOCK names",
          un["base"]["n_images"] == len(Wd["train_core"]) + len(Wd["tsw22"])
          and [p["name"] for p in un["base"]["union"]] == ["train_core", "tsw22"]
          and un["base"]["source_locked"] == ["train_core", "tsw22"])
    e = refused(B.build_definition, "b_bad_u", union=[V2() / "train_core.jsonl", V2() / "base_v2.jsonl"],
                role="union", testing=True)
    check("overlapping union parts are refused", e is not None and "overlap" in str(e), e)
    plant_u = W.planted("te_hflip_u", Wd["test"][3]["image"], "hflip")
    part_u = W.write_manifest("scratch", "part_u", [W.row_for("pl_u", plant_u, [(0, .5, .5, .2, .2)], "x")])
    e = refused(B.build_definition, "b_bad_u2", union=[V2() / "train_core.jsonl", part_u], testing=True)
    check("a union part holding an hflip copy of a test image is refused by the guard over the union",
          e is not None and "never-train guard" in str(e), e)

    plant = W.planted("te_hflip_b", Wd["test"][0]["image"], "hflip")
    bad_m = W.write_manifest("scratch", "bad", Wd["base_v2"] + [W.row_for("pl_b", plant, [(0, .5, .5, .2, .2)], "x")])
    cases = {"the v1 dev manifest": dict(manifest=C.manifest_path("dev")),
             "the v2 imageweeds manifest": dict(manifest=V2() / "imageweeds.jsonl"),
             "a manifest with an hflip copy of a test image": dict(manifest=bad_m),
             "an ood22 final exam": dict(manifest=base_v2, final_exams=["dev", "ood22"]),
             "both --manifest and --union": dict(manifest=base_v2, union=[base_v2]),
             "neither --manifest nor --union": dict(),
             "an unknown role": dict(manifest=base_v2, role="whatever")}
    for what, kw in cases.items():
        e = refused(B.build_definition, "b_refuse", testing=True, **kw)
        check("build refuses %s" % what, e is not None, e)
    check("... and a refused build left no exp.json", not (C.INC_DIR / "b_refuse" / "exp.json").exists())
    (C.REPO / "yolo11m.pt").rename(C.REPO / "yolo11m.pt.away")
    try:
        e = refused(B.build_definition, "b_prod_m", manifest=base_v2, arm="m640", role="capacity")
    finally:
        (C.REPO / "yolo11m.pt.away").rename(C.REPO / "yolo11m.pt")
    check("a production build refuses a missing arm checkpoint (nothing is downloaded)",
          e is not None and "nothing is downloaded" in str(e), e)
    e = refused(B.build_definition, "b_prod", manifest=base_v2, role="b_v2")
    check("a production build refuses a LOCK v2 written by a testing build", e is not None and "testing" in str(e), e)


def test_measure_arms(Wd):
    print("the measurement arms m832 and s1024 (2026-09-30)")
    check("L-4's grid is n640, s640, m640; the measurement arms are m832 and s1024, in the table",
          RC.GRID_ARMS == ("n640", "s640", "m640") and RC.MEASURE_ARMS == ("m832", "s1024")
          and set(RC.ARM_IDS) == set(RC.GRID_ARMS) | set(RC.MEASURE_ARMS), (RC.GRID_ARMS, RC.MEASURE_ARMS))
    check("the grid's recipes are unchanged: 640 px, batch 32, every recipe of the table",
          all(RC.table(a)[r] == dict(RC.COMMON, **(RC.COLD if r == "cold" else RC.INCREMENTAL[r]), imgsz=640)
              and RC.table(a)[r]["batch"] == 32 for a in RC.GRID_ARMS for r in RC.table(a)))
    m, s = RC.table("m832"), RC.table("s1024")
    check("m832 trains yolo11m.pt at 832 px, batch 16, in every recipe; s1024 yolo11s.pt at 1024 px, batch 32",
          RC.ARMS["m832"]["model"] == "yolo11m.pt" and RC.ARMS["s1024"]["model"] == "yolo11s.pt"
          and all(r["imgsz"] == 832 and r["batch"] == 16 for r in m.values())
          and all(r["imgsz"] == 1024 and r["batch"] == 32 for r in s.values())
          and {k: v for k, v in m["cold"].items() if k not in ("imgsz", "batch")}
          == {k: v for k, v in RC.cold("m640").items() if k not in ("imgsz", "batch")}, (m["cold"], s["cold"]))
    check("every arm's GFLOPs are its 640 figure x (imgsz / 640)^2, and the 640 figure is the same model's 640 arm's",
          all(abs(v["gflops"] - round(v["gflops_at_640"] * (v["imgsz"] / 640.0) ** 2, 3)) < 1e-9
              for v in RC.ARMS.values())
          and RC.ARMS["m832"]["gflops_at_640"] == RC.ARMS["m640"]["gflops"]
          and RC.ARMS["s1024"]["gflops_at_640"] == RC.ARMS["s640"]["gflops"],
          {a: (v["gflops"], v["gflops_at_640"]) for a, v in RC.ARMS.items()})
    rm = RC.resolve_arm("m832", repo=C.REPO)
    rs = RC.resolve_arm("s1024", repo=C.REPO)
    check("the arms resolve with their checkpoint's sha256 (yolo11m.pt, yolo11s.pt) and pass check_arm_record",
          rm["weights_sha256"] == W.sha(C.REPO / "yolo11m.pt") and rs["weights_sha256"] == W.sha(C.REPO / "yolo11s.pt")
          and RC.check_arm_record(rm) == "m832" and RC.check_arm_record(rs) == "s1024" and rm["imgsz"] == 832, (rm, rs))
    e = refused(RC.check_arm_record, dict(rm, imgsz=640))
    check("... and a record of m832 at 640 px is refused", e is not None and "imgsz" in str(e), e)
    check("--arch/--imgsz name them: yolo11m at 832 is m832, yolo11s.pt at 1024 is s1024",
          RC.arm_from_arch("yolo11m", 832) == "m832" and RC.arm_from_arch("yolo11s.pt", 1024) == "s1024")
    check("an m832 recipe at batch 32 deviates from the table; its own recipe does not",
          RC.deviations("base", dict(RC.cold("m832"), batch=32), "m832") == ["batch 32 (protocol v3 16)"]
          and not RC.deviations("base", RC.cold("m832"), "m832")
          and RC.match("cand", RC.incremental("r0", "m832"), "m832") == "r0")
    rt = {a: RC.rates(a) for a in ("n640", "m640", "m832", "s1024")}
    check("cost factors: m832 FLOPs x%.3f and pixels x%.2f of n640, s1024 x%.3f and x%.2f; scoring (640 px) at "
          "m640's and s640's FLOPs" % (rt["m832"]["flops_factor"], rt["m832"]["pixel_factor"],
                                         rt["s1024"]["flops_factor"], rt["s1024"]["pixel_factor"]),
          abs(rt["m832"]["flops_factor"] - 17.8689) < 1e-3 and abs(rt["m832"]["pixel_factor"] - 1.69) < 1e-9
          and abs(rt["s1024"]["flops_factor"] - 8.5573) < 1e-3 and abs(rt["s1024"]["pixel_factor"] - 2.56) < 1e-9
          and rt["m832"]["score"] == rt["m640"]["score"] and RC.rates("s1024")["score"] == RC.rates("s640")["score"]
          and rt["m832"]["cold"][0] == round(RC.COLD_MS[0] * 1.69, 4)
          and rt["m832"]["cold"][1] == round(RC.COLD_MS[1] * RC.flops_factor("m832"), 4), rt["m832"])
    base_v2 = V2() / "base_v2.jsonl"
    bm, _ = B.build_definition("b_v2_m832", manifest=base_v2, arm="m832", testing=True)
    bs, _ = B.build_definition("b_v2_s1024", manifest=base_v2, arch="yolo11s", imgsz=1024, testing=True)
    D.validate_definition(json.loads(json.dumps(bm)))
    D.check_definition_data(bm)
    check("on base_v2 without --role they are capacity builds: 3 seeds, finals dev and imageweeds (no test: P10), "
          "the arm's cold recipe and checkpoint, a definition the pinned driver accepts",
          all(d["role"] == "capacity" and d["seeds"] == [0, 1, 2] and d["final_exams"] == ["dev", "imageweeds"]
              for d in (bm, bs))
          and bm["base"]["recipe"] == RC.cold("m832") and bm["init_weights"] == "yolo11m.pt"
          and bs["base"]["recipe"] == RC.cold("s1024") and bs["init_weights"] == "yolo11s.pt"
          and bm["arm"]["id"] == "m832" and bs["arm"]["id"] == "s1024",
          [(d["role"], d["arm"]["id"], d["base"]["recipe"]["imgsz"], d["base"]["recipe"]["batch"]) for d in (bm, bs)])
    for what, kw in (("role capacity", dict(arm="m832", role="capacity")), ("inferred role", dict(arm="s1024"))):
        e = refused(B.build_definition, "b_v2_meas_test", manifest=base_v2, final_exams=["dev", "imageweeds", "test"],
                    testing=True, **kw)
        check("... and a measurement build (%s) that lists test is refused: it is at no milestone read (P10)" % what,
              e is not None and "may not read test" in str(e) and not (C.INC_DIR / "b_v2_meas_test").exists(), e)
    c = bm["cost_estimate"]
    check("... and their est. cost is recorded with the pixel ratio (m832 low bracket %s GPU-h per run)"
          % c["per_run_gpu_h"][0], c["arm"] == "m832" and abs(c["pixel_factor"] - 1.69) < 1e-9 and c["imgsz"] == 832)


def test_e2e_baseline(Wd):
    print("end to end: a baseline through the real v2 executor (FakeBackend)")
    backend = RecordingBackend(runner=executor_runner)
    with tiny_table():
        with W.env(INC_JOB_SCRIPT="/elsewhere/run_inc_job.sh"):
            e = refused(B.build, "canary_e2e", manifest=V2() / "train_core.jsonl", role="canary",
                        testing=W.TESTING, backend=backend)
        check("INC_JOB_SCRIPT naming another script refuses the build before init",
              e is not None and "never run by another executor" in str(e) and not backend.submissions, e)
        shutil.rmtree(C.INC_DIR / "canary_e2e", ignore_errors=True)
        t0 = time.time()
        summ, defn, res = B.build("canary_e2e", manifest=V2() / "train_core.jsonl", role="canary",
                                  testing=W.TESTING, backend=backend, quiet=True)
        res = drive("canary_e2e", backend)
    v2_script = str(B.job_script_path())
    st = json.loads((C.INC_DIR / "canary_e2e" / "state.json").read_text())
    check("driver init + advance run the base and the final run to done (%.0fs)" % (time.time() - t0),
          res["done"] and st["done"] and st["runs"]["base__s0"]["status"] == "complete"
          and st["runs"]["final__base__s0"]["status"] == "complete", D.status("canary_e2e"))
    check("every submission carried INC_JOB_SCRIPT = run_inc2_job.sh, and SlurmBackend's argv names it",
          backend.scripts and all(s == v2_script for s in backend.scripts)
          and all(a[-3] == v2_script for a in backend.argvs), (backend.scripts, backend.argvs[:1]))
    rj = json.loads((C.INC_DIR / "canary_e2e" / "runs" / "base__s0" / "run.json").read_text())
    check("the base run is the v2 executor's (protocol v3, arm pinned by the build, a dev sidecar)",
          rj.get("protocol") == "v3" and rj["arm"] == defn["arm"] and rj["sidecars"].get("dev")
          and rj.get("recipe_name") == "cold", {k: rj.get(k) for k in ("protocol", "recipe_name")})
    e = refused(B.build_definition, "canary_e2e", manifest=V2() / "train_core.jsonl", role="canary", testing=W.TESTING)
    check("a built experiment is not built again", e is not None and "already built" in str(e), e)

    shutil.rmtree(C.INC_DIR / "canary_e2e" / "logs", ignore_errors=True)     # secondary must not rely on it
    out = B.secondary("canary_e2e", C.INC_DIR / "canary_e2e" / "runs" / "base__s0" / "weights" / "final.pt",
                      source="test incumbent")
    sp = json.loads(pathlib.Path(out["spec"]).read_text())
    check("secondary: a final spec on the milestone's exams that the v2 executor accepts, and the run_inc2_job.sh "
          "argv", sp["kind"] == "final" and sp["exams"] == ["dev"] and out["argv"][-3] == v2_script
          and out["argv"][-1] == "canary_e2e" and pathlib.Path(out["list"]).read_text().strip() == out["spec"]
          and out["record"]["weights_sha256"] == W.sha(sp["init"]))
    check("secondary creates the log dir its sbatch --output names (Slurm opens it before the job starts)",
          any(a.startswith("--output=%s/" % (C.INC_DIR / "canary_e2e" / "logs")) for a in out["argv"])
          and (C.INC_DIR / "canary_e2e" / "logs").is_dir())
    check("secondary refuses a driver run id and missing weights",
          refused(B.secondary, "canary_e2e", sp["init"], run_id="final__base__s0") is not None
          and refused(B.secondary, "canary_e2e", W.TMP / "none.pt", run_id="x2") is not None)


def fake_score(value, weights, n=20, species=None):
    per = {s: (species or {}).get(s, value) for s in G.SPECIES}
    return {"exam": "dev", "scorer_sha256": "TEST-" + "s" * 16, "manifest_sha256": "m" * 16,
            "key_order_sha256": "k" * 16, "weights_sha256": weights, "n_images": n, "map50_95": value,
            "map50": value + 0.1, "agnostic_map50_95": value + 0.05, "agnostic_map50": value + 0.15,
            "per_class": per, "n_gt": {c: (40 if c != "OtherPlant" else 0) for c in C.CLASS_NAMES},
            "image_correct": "1" * 12 + "0" * (n - 12), "production": False}


def test_e2e_chain(Wd):
    print("end to end: a 1-step chain as inc2.stream builds one (FakeBackend, checked specs)")
    exp = "seg_e2e"
    arm = RC.resolve_arm("n640", repo=C.REPO)
    inc_m = W.write_manifest(exp, "D1", Wd["inc"])
    defn = {"exp": exp, "type": "chain", "builder": "test (the shape of inc2.stream build)", "testing": W.TESTING,
            "replay_mode": "full", "gate": {"flips_mode": "net"}, "seeds": [0, 1, 2], "decision_exam": "dev",
            "final_exams": ["dev", "imageweeds"],
            "base": {"name": "P0", "manifest": str(V2() / "base_v2.jsonl"), "manifest_sha256": W.sha(V2() / "base_v2.jsonl"),
                     "n_images": len(Wd["base_v2"]), "recipe": RC.cold("n640")},
            "steps": [{"name": "D1", "manifest": str(inc_m), "manifest_sha256": W.sha(inc_m), "n_images": len(Wd["inc"]),
                       "clean": False}],
            "recipes": {"r0": RC.incremental("r0")}, "truth": False}
    defn.update(RC.stamp(arm))
    seen, bad = [], []
    values = {"base": 0.60, "cand": 0.63, "null": 0.60, "soup": 0.64}

    def runner(spec):
        p = pathlib.Path(spec["out_dir"]) / "spec.json"
        try:
            T.validate_spec(spec, p)
            if spec["kind"] in T.TRAIN_KINDS:
                arm_rec, _ = T.experiment_arm(spec["exp"])
                if RC.deviations(spec["kind"], spec["recipe"], arm_rec) or T.init_check(spec["kind"], spec["init"], arm_rec):
                    bad.append((spec["run_id"], "not a production v3 spec"))
        except T.RunError as e:
            bad.append((spec["run_id"], str(e)))
        seen.append(spec["run_id"])
        out = pathlib.Path(spec["out_dir"])
        (out / "weights").mkdir(parents=True, exist_ok=True)
        (out / "weights" / "final.pt").write_bytes(os.urandom(64))
        w = W.sha(out / "weights" / "final.pt")
        kind = spec["kind"]
        v = values.get(kind, 0.61) + 0.001 * (spec.get("recipe") or {}).get("seed", 0)
        for exam in spec["exams"]:
            s = fake_score(v, w)
            s["exam"] = exam
            (out / "scores").mkdir(exist_ok=True)
            (out / "scores" / ("%s.json" % exam)).write_text(json.dumps(s))
        (out / "run.json").write_text(json.dumps({"status": "done", "seconds": 1.0, "weights_sha256": w}))
        return "COMPLETED"

    backend = RecordingBackend(runner=runner)
    B.ensure_job_script(testing=True)
    D.init(defn, backend=backend, quiet=True)
    res = drive(exp, backend)
    st = json.loads((C.INC_DIR / exp / "state.json").read_text())
    led = [json.loads(x) for x in (C.INC_DIR / exp / "ledger.jsonl").read_text().splitlines() if x.strip()]
    gate = [x for x in led if x.get("type") == "gate"]
    check("the chain runs to done: base, cand, null, soup and finals; the step is decided by the pinned gate (net)",
          res["done"] and st["chains"]["r0"]["phase"] == "done" and len(gate) == 1
          and gate[0]["decision"]["config"].get("flips_mode") == "net"
          and gate[0]["decision"]["verdict"] == "ACCEPT" and "r0__s01_D1__soup" in seen, (st["chains"], seen))
    check("every spec the driver wrote is one the v2 executor accepts (exams, v3 recipe at the arm, the arm's "
          "init for cold runs)", not bad, bad)
    v2_script = str(B.job_script_path())
    check("every submission carried INC_JOB_SCRIPT = run_inc2_job.sh; the SlurmBackend argv never names the v1 script",
          backend.scripts and all(s == v2_script for s in backend.scripts)
          and not any("run_inc_job.sh" in " ".join(a) for a in backend.argvs), backend.scripts)


def fake_baseline(exp, arm, dev_values, test_values, rate_ms=None, n_images=7625, scorer="s" * 16, sidecar=True,
                  production=True, manifest_sha="b" * 64, seeds=None):
    """A built baseline's files as inc2.baseline and the driver write them
    (a measurement arm's finals are dev and imageweeds: no test)."""
    root = C.INC_DIR / exp
    seeds = list(range(len(dev_values))) if seeds is None else list(seeds)
    exams = ["dev", "imageweeds"] if RC.arm_id(arm) in RC.MEASURE_ARMS else ["dev", "imageweeds", "test"]
    rec = RC.resolve_arm(arm, require_weights=False)
    defn = {"exp": exp, "type": "baseline", "seeds": seeds, "final_exams": exams,
            "base": {"name": "base_v2", "manifest": "/x/base_v2.jsonl", "manifest_sha256": manifest_sha,
                     "n_images": n_images, "recipe": RC.cold(arm)}}
    defn.update(RC.stamp(rec))
    root.mkdir(parents=True, exist_ok=True)
    (root / "exp.json").write_text(json.dumps(defn))
    for s, dv, tv in zip(seeds, dev_values, test_values):
        r = root / "runs" / ("base__s%d" % s)
        (r / "scores").mkdir(parents=True, exist_ok=True)
        secs = (rate_ms or 6.5) * n_images * 100 / 1000.0
        sc_path = r / "scores" / "dev.sidecar.json"
        sc_path.write_text(json.dumps({"format": "inc2-scorer-sidecar/1"}))
        side = ({"dev": {"path": str(sc_path), "sha256": W.sha(sc_path)}} if sidecar
                else {"dev": {"status": "failed", "error": "broken"}})
        (r / "run.json").write_text(json.dumps({"status": "done", "testing": not production, "train_seconds": secs,
                                                "n_train_images": n_images, "sidecars": side,
                                                "spec": {"recipe": dict(RC.cold(arm), seed=s)}}))
        sc = dict(fake_score(dv, "w%s%d" % (exp, s)), production=production, scorer_sha256=scorer)
        (r / "scores" / "dev.json").write_text(json.dumps(sc))
        f = root / "runs" / ("final__base__s%d" % s) / "scores"
        f.mkdir(parents=True, exist_ok=True)
        for exam, v in (("dev", dv), ("imageweeds", 0.05), ("test", tv)):
            if exam not in exams:
                continue
            (f / ("%s.json" % exam)).write_text(json.dumps(dict(fake_score(v, "w%s%d" % (exp, s)), exam=exam,
                                                                production=True)))


def decision_reads_test(doc):
    """True when any path or key of the decision names the test exam."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "test":
                    found.append(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and (x.endswith("/test.json") or x.endswith("_test.json")):
            found.append(x)
    walk(doc)
    return bool(found)


def test_verdicts(Wd):
    print("canary-verdict, capacity-verdict, estimate")
    rep = C.INC_DIR / "b0_ref" / "report.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"final": [{"model": "base train_core", "exams": {"dev": {"twelve": {
        "mean": 0.8082, "sd": 0.0063, "n": 3}}}}]}))
    e = refused(B.canary_verdict, "none_yet", "b0_ref")
    check("canary-verdict on an experiment that does not exist is refused", e is not None and "exp.json" in str(e), e)
    fake_baseline("can_noref", "n640", [0.8101], [0.85])
    noref = B.canary_verdict("can_noref", "b0_ref", write=False)
    check("a canary cannot pass without the reference's exp.json (what its seed 0 trained)",
          noref["within_one_sd"] and noref["same_run_as_reference"] is None and not noref["passed"], noref)
    (rep.parent / "exp.json").write_text(json.dumps({"exp": "b0_ref", "type": "baseline", "seeds": [0, 1, 2],
                                                     "base": {"manifest_sha256": "b" * 64, "recipe": RC.cold("n640")}}))
    for exp, v, sc, prod, msha in (("can_ok", 0.8101, True, True, "b" * 64), ("can_bad", 0.7950, True, True, "b" * 64),
                                   ("can_nosc", 0.8101, False, True, "b" * 64),
                                   ("can_test", 0.8101, True, False, "b" * 64),
                                   ("can_other", 0.8101, True, True, "c" * 64)):
        fake_baseline(exp, "n640", [v], [0.85], sidecar=sc, production=prod, manifest_sha=msha)
    ok = B.canary_verdict("can_ok", "b0_ref")
    bad = B.canary_verdict("can_bad", "b0_ref")
    nosc = B.canary_verdict("can_nosc", "b0_ref")
    check("canary within one b0 sd passes (0.8101 vs 0.8082 +- 0.0063); 0.7950 fails; canary.json written",
          ok["passed"] and not bad["passed"] and (C.INC_DIR / "can_ok" / "canary.json").is_file()
          and ok["reference_dev"]["sha256"] == W.sha(rep) and ok["production"] is True
          and all(ok["same_run_as_reference"].values()), ok)
    check("a canary within the sd whose sidecar failed does not pass (the sidecar's first cluster reading)",
          nosc["within_one_sd"] and not nosc["sidecar_ok"] and not nosc["passed"])
    tst = B.canary_verdict("can_test", "b0_ref")
    tst_ok = B.canary_verdict("can_test", "b0_ref", write=False, allow_testing=True)
    check("a test-mode canary (run and score not production) does not pass; only allow_testing (tests) lets it",
          tst["within_one_sd"] and tst["sidecar_ok"] and tst["production"] is False and not tst["passed"]
          and tst_ok["passed"], tst)
    oth = B.canary_verdict("can_other", "b0_ref")
    check("a canary trained on another manifest than the reference's seed 0 does not pass",
          oth["within_one_sd"] and oth["same_run_as_reference"]["manifest_sha256"] is False and not oth["passed"], oth)

    print("canary-verdict with the L-8 drops (b0_v1's base minus the recorded rows)")
    ref2 = C.INC_DIR / "b0_ref2"
    ref2.mkdir(parents=True, exist_ok=True)
    (ref2 / "report.json").write_text(rep.read_text())
    v1_tc = C.manifest_path("train_core")
    ref_base = {"manifest": str(ref2 / "manifests" / "x.jsonl"), "source_manifest": str(v1_tc),
                "manifest_sha256": W.sha(v1_tc), "recipe": RC.cold("n640")}
    (ref2 / "exp.json").write_text(json.dumps({"exp": "b0_ref2", "type": "baseline", "seeds": [0, 1, 2],
                                               "base": ref_base}))
    rec = B.variant_drops_record(W.sha(V2() / "train_core.jsonl"), production=False)
    v2_tc = W.sha(V2() / "train_core.jsonl")
    for exp, drops in (("can_l8", rec), ("can_l8_none", None),
                       ("can_l8_other_ref", dict(rec, reference_manifest_sha256="d" * 64)),
                       ("can_l8_other_list", dict(rec, sha256="e" * 64))):
        fake_baseline(exp, "n640", [0.8101], [0.85], manifest_sha=v2_tc)
        d = json.loads((C.INC_DIR / exp / "exp.json").read_text())
        if drops is not None:
            d["variant_drops"] = drops
        (C.INC_DIR / exp / "exp.json").write_text(json.dumps(d))
    l8 = B.canary_verdict("can_l8", "b0_ref2")
    mm = l8.get("manifest_match") or {}
    check("a canary on b0_v1's base minus the recorded L-8 drops passes: accepted by the reference's sha256 and the "
          "list's sha256, canary.json names the dropped row",
          l8["passed"] and l8["same_run_as_reference"]["manifest_sha256"] is True
          and mm.get("how") == "reference minus the recorded L-8 drops"
          and [r["key"] for r in mm.get("dropped", [])] == [Wd["variant_drops"][0]["key"]]
          and mm.get("variant_drops_sha256") == W.sha(V2() / "train_core_variant_drops.jsonl")
          and json.loads((C.INC_DIR / "can_l8" / "canary.json").read_text())["manifest_match"] == mm, mm)
    for exp, why in (("can_l8_none", "records no drops"), ("can_l8_other_ref", "records drops from another reference"),
                     ("can_l8_other_list", "records another L-8 list than LOCK v2's")):
        v = B.canary_verdict(exp, "b0_ref2", write=False)
        check("a canary on the reduced manifest that %s does not pass" % why,
              v["within_one_sd"] and v["same_run_as_reference"]["manifest_sha256"] is False and not v["passed"],
              v.get("manifest_match"))
    lst = V2() / "train_core_variant_drops.jsonl"
    saved = lst.read_bytes()
    lst.write_bytes(saved + b"\n")
    try:
        v = B.canary_verdict("can_l8", "b0_ref2", write=False)
    finally:
        lst.write_bytes(saved)
    check("... nor one whose L-8 list changed since it was built",
          not v["passed"] and v["same_run_as_reference"]["manifest_sha256"] is False, v.get("manifest_match"))
    d = json.loads((C.INC_DIR / "can_l8" / "exp.json").read_text())
    saved_exp = json.dumps(d)
    lst.write_bytes(saved + b"\n")
    d["variant_drops"] = dict(d["variant_drops"], sha256=W.sha(lst))
    (C.INC_DIR / "can_l8" / "exp.json").write_text(json.dumps(d))
    try:
        v = B.canary_verdict("can_l8", "b0_ref2", write=False)
    finally:
        lst.write_bytes(saved)
        (C.INC_DIR / "can_l8" / "exp.json").write_text(saved_exp)
    check("... nor one whose record names a list LOCK v2 does not record, even when the file matches the record",
          not v["passed"] and any("LOCK v2 does not record" in p for p in (v.get("manifest_match") or {}).get(
              "problems", [])), v.get("manifest_match"))
    ok2 = B.canary_verdict("can_ok", "b0_ref2", write=False)
    check("... nor a canary on another manifest against this reference",
          not ok2["passed"] and ok2["same_run_as_reference"]["manifest_sha256"] is False)

    fake_baseline("cap_n", "n640", [0.810, 0.813, 0.808], [0.854, 0.851, 0.858])
    fake_baseline("cap_s", "s640", [0.840, 0.842, 0.838], [0.874, 0.870, 0.877], rate_ms=25.0)
    fake_baseline("cap_m", "m640", [0.812, 0.815, 0.811], [0.860, 0.862, 0.858], rate_ms=18.0)
    dec, repd = B.capacity_verdict("cap_n", ["cap_s", "cap_m"], out_dir=C.INC_DIR / "capacity_t")
    check("s640 beats n640 by more than 2 pooled sd and is chosen; m640 does not qualify",
          dec["chosen_arm"] == "s640" and dec["qualifying"] == ["cap_s"] and not dec["arms"]["cap_m"]["qualifies"]
          and dec["arms"]["cap_s"]["diff_vs_n"] > 2 * dec["arms"]["cap_s"]["pooled_sd"])
    sc = dec["step_cost"]
    check("truth_every from s640's measured 25 ms per image-epoch: one r0 step with truth costs %.1f GPU-h, so "
          "truth runs every %d steps" % (sc["gpu_h"][1], dec["truth_every"]),
          not sc["estimate"] and 25.0 < sc["gpu_h"][1] < 50 and dec["truth_every"] == 2 and dec["m"] == 763
          and "measured" in sc["basis"])
    for e in ("cap_n", "cap_s", "cap_m"):
        for s in range(3):
            p = C.INC_DIR / e / "runs" / ("final__base__s%d" % s) / "scores" / "test.json"
            d = json.loads(p.read_text())
            d["map50_95"] = 0.99 - 0.1 * s
            p.write_text(json.dumps(d))
    dec2, rep2 = B.capacity_verdict("cap_n", ["cap_s", "cap_m"], write=False)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("generated_utc", "out")}  # noqa: E731
    check("test-blind: every test score perturbed, the decision is identical; the report moved",
          strip(dec) == strip(dec2) and rep2["arms"]["cap_s"]["exams"]["test"] != repd["arms"]["cap_s"]["exams"]["test"])
    check("the report (for people) carries dev, imageweeds and test with the gap to 0.90; the decision file "
          "has no test number", abs(repd["arms"]["cap_s"]["gap_to_target"] - (0.90 - (0.874 + 0.870 + 0.877) / 3)) < 1e-9
          and not decision_reads_test(json.load(open(dec["out"])))
          and (C.INC_DIR / "capacity_t" / "capacity_v1_report.md").is_file())
    # the measurement arms: m832 with the best dev mean of all, s1024 below n640
    fake_baseline("cap_m832", "m832", [0.870, 0.872, 0.869], [0.890, 0.888, 0.891], rate_ms=24.0)
    fake_baseline("cap_s1024", "s1024", [0.800, 0.803, 0.799], [0.84] * 3, rate_ms=21.0)
    e = refused(B.capacity_verdict, "cap_n", ["cap_s", "cap_m832"], write=False)
    check("a measurement arm is never a candidate (--arms b_v2_m832 is refused: list it with --record)",
          e is not None and "never a candidate" in str(e) and "--record" in str(e), e)
    e = refused(B.capacity_verdict, "cap_n", ["cap_s"], write=False, record_exps=["cap_m"])
    check("a grid arm is a candidate, not a record", e is not None and "not a record" in str(e), e)
    dec_path = C.INC_DIR / "capacity_t" / "capacity_v1.json"
    dec_bytes = dec_path.read_bytes()
    decr, repr_ = B.capacity_verdict("cap_n", ["cap_s", "cap_m"], out_dir=C.INC_DIR / "capacity_t",
                                     record_exps=["cap_m832", "cap_s1024"])
    core = lambda d: {k: v for k, v in strip(d).items() if k not in ("record_exps", "record", "record_note",  # noqa: E731
                                                                     "decision_kept")}
    rec = decr.get("record") or {}
    check("--record: the decision is the grid's alone (s640 chosen, qualifying [cap_s]) although m832 has the best "
          "dev mean; the measurement arms are listed record only, never in arms, qualifying or chosen",
          core(decr) == core(dec) and decr["chosen_arm"] == "s640" and decr["qualifying"] == ["cap_s"]
          and set(rec) == {"cap_m832", "cap_s1024"} and "cap_m832" not in decr["arms"]
          and rec["cap_m832"]["record_only"] is True and rec["cap_m832"]["above_2_pooled_sd"] is True
          and rec["cap_s1024"]["above_2_pooled_sd"] is False
          and rec["cap_m832"]["mean"] > max(v["mean"] for v in decr["arms"].values())
          and rec["cap_m832"]["trained_imgsz"] == 832 and rec["cap_m832"]["scored_imgsz"] == 640,
          {k: decr.get(k) for k in ("chosen_arm", "qualifying", "record")})
    rows = repr_["arms"]
    check("... an existing decision file is kept byte for byte (the stream adopted it by sha256); the report lists the "
          "measurement arms' finals (dev, imageweeds: no test, so no gap to 0.90), marked record only",
          dec_path.read_bytes() == dec_bytes and decr.get("decision_kept") is True
          and repr_["decision_sha256"] == W.sha(dec_path) and rows["cap_m832"].get("record_only") is True
          and "record_only" not in rows["cap_s"] and sorted(rows["cap_m832"]["exams"]) == ["dev", "imageweeds"]
          and rows["cap_m832"]["exams"]["dev"]["n"] == 3 and rows["cap_m832"]["gap_to_target"] is None
          and rows["cap_s"]["exams"]["test"]["n"] == 3
          and "m832 (record only" in (C.INC_DIR / "capacity_t" / "capacity_v1_report.md").read_text(),
          {k: rows[k].get("record_only") for k in rows})
    fake_baseline("cap_m832_s2", "m832", [0.870, 0.872], [0.0] * 2)
    e = refused(B.capacity_verdict, "cap_n", ["cap_s", "cap_m"], write=False, record_exps=["cap_m832_s2"])
    check("a record arm that lacks one of the grid's seeds is refused (the grid's seeds are never narrowed by it)",
          e is not None and "lacks seeds [2]" in str(e), e)
    fake_baseline("cap_s1024_s4", "s1024", [0.800, 0.803, 0.799, 0.806], [0.0] * 4, seeds=[0, 1, 2, 3])
    dec4, _ = B.capacity_verdict("cap_n", ["cap_s", "cap_m"], write=False, record_exps=["cap_s1024_s4"])
    check("... and one with an extra seed leaves the decision's seeds and outcome the grid's alone, read on those seeds",
          core(dec4) == core(dec) and dec4["seeds"] == [0, 1, 2]
          and dec4["record"]["cap_s1024_s4"]["seeds"] == [0, 1, 2]
          and abs(dec4["record"]["cap_s1024_s4"]["mean"] - (0.800 + 0.803 + 0.799) / 3) < 1e-9,
          {k: dec4.get(k) for k in ("seeds", "chosen_arm", "qualifying")})
    fake_baseline("cap_m832_mf", "m832", [0.870, 0.872, 0.869], [0.0] * 3, manifest_sha="c" * 64)
    e = refused(B.capacity_verdict, "cap_n", ["cap_s", "cap_m"], write=False, record_exps=["cap_m832_mf"])
    check("... and one trained on another manifest is refused", e is not None and "different manifests" in str(e), e)
    other = json.loads(dec_bytes)
    other["chosen_arm"], other["chosen_exp"] = "m640", "cap_m"
    dec_path.write_text(json.dumps(other))
    e = refused(B.capacity_verdict, "cap_n", ["cap_s", "cap_m"], out_dir=C.INC_DIR / "capacity_t",
                record_exps=["cap_m832"])
    check("... and a decision file that disagrees with the recomputed grid refuses the record-only run, untouched",
          e is not None and "never replaces it" in str(e) and json.loads(dec_path.read_text()) == other, e)
    dec_path.write_bytes(dec_bytes)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = B.main(["capacity-verdict", "--n", "cap_n", "--arms", "cap_s,cap_m", "--record", "cap_m832,cap_s1024",
                     "--out-dir", str(C.INC_DIR / "capacity_t")])
    check("the CLI takes --record (exit 0; the decision file unchanged)", rc == 0 and dec_path.read_bytes() == dec_bytes)
    fake_baseline("cap_n2", "n640", [0.810, 0.813, 0.808], [0.85] * 3)
    fake_baseline("cap_m2", "m640", [0.812, 0.816, 0.809], [0.86] * 3)
    dec3, _ = B.capacity_verdict("cap_n2", ["cap_m2"], write=False)
    check("no arm beats n640 by 2 pooled sd: the stream stays on n640, truth every step",
          dec3["chosen_arm"] == "n640" and dec3["qualifying"] == [] and dec3["truth_every"] == 1)
    fake_baseline("cap_x", "s640", [0.840, 0.842, 0.838], [0.87] * 3, scorer="z" * 16)
    e = refused(B.capacity_verdict, "cap_n", ["cap_x"], write=False)
    check("arms scored by different scorers are refused", e is not None and "another scorer" in str(e), e)
    d = json.loads((C.INC_DIR / "cap_x" / "exp.json").read_text())
    d.pop("arm")
    (C.INC_DIR / "cap_x" / "exp.json").write_text(json.dumps(d))
    e = refused(B.capacity_verdict, "cap_n", ["cap_x"], write=False)
    check("an experiment that pins no arm is refused", e is not None and "pins no arm" in str(e), e)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = B.main(["estimate", "--n-images", "7625", "--seeds", "0,1,2", "--arm", "m640"])
    out = json.loads(buf.getvalue())
    check("estimate prints the arm's est. cost", rc == 0 and out["arm"] == "m640" and out["estimate"] is True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = B.main(["estimate", "--n-images", "7625", "--seeds", "0,1,2", "--arch", "yolo11m", "--imgsz", "640"])
    check("estimate takes --arch/--imgsz too", rc == 0 and json.loads(buf.getvalue())["arm"] == "m640")


def test_autopilot_argv():
    print("the autopilot's L23B argv (stream_levers.json: --exp --manifest --seeds [--arch --imgsz], no --role)")
    base_v2 = str(V2() / "base_v2.jsonl")
    tc = str(V2() / "train_core.jsonl")
    cases = [("ap_b_v2", ["--manifest", base_v2, "--seeds", "0,1,2,3,4"], "b_v2", "n640", ["dev", "imageweeds", "test"]),
             ("ap_cap_s", ["--manifest", base_v2, "--seeds", "0,1,2", "--arch", "yolo11s", "--imgsz", "640"],
              "capacity", "s640", ["dev", "imageweeds", "test"]),
             ("ap_cap_m", ["--manifest", base_v2, "--seeds", "0,1,2", "--arch", "yolo11m", "--imgsz", "640"],
              "capacity", "m640", ["dev", "imageweeds", "test"]),
             ("ap_canary", ["--manifest", tc, "--seeds", "0"], "canary", "n640", ["dev"]),
             ("ap_cap_m832", ["--manifest", base_v2, "--seeds", "0,1,2", "--arm", "m832", "--role", "capacity"],
              "capacity", "m832", ["dev", "imageweeds"]),
             ("ap_cap_s1024", ["--manifest", base_v2, "--seeds", "0,1,2", "--arch", "yolo11s", "--imgsz", "1024"],
              "capacity", "s1024", ["dev", "imageweeds"])]
    for exp, args, role, arm, exams in cases:
        rc = B.main(["build", "--exp", exp] + args + ["--testing", "--no-init", "--quiet"])
        summ = json.loads((C.INC_DIR / exp / B.BUILD_SUMMARY).read_text()) if rc == 0 else {}
        check("%s: exit 0, role %s, arm %s, finals %s" % (" ".join(args[2:]) or exp, role, arm, exams),
              rc == 0 and summ.get("role") == role and (summ.get("arm") or {}).get("id") == arm
              and summ.get("final_exams") == exams, (rc, summ.get("role"), summ.get("final_exams")))
    for what, args in (("an arch outside the table", ["--manifest", base_v2, "--seeds", "0,1,2", "--arch", "yolo11x",
                                                      "--imgsz", "640"]),
                       ("--imgsz alone", ["--manifest", base_v2, "--seeds", "0,1,2", "--imgsz", "640"])):
        rc = B.main(["build", "--exp", "ap_bad"] + args + ["--testing", "--no-init", "--quiet"])
        check("the CLI refuses %s (exit 1)" % what, rc == 1)


def main():
    t0 = time.time()
    try:
        Wd = W.build_world()
        test_builds(Wd)
        test_measure_arms(Wd)
        test_e2e_baseline(Wd)
        test_e2e_chain(Wd)
        test_verdicts(Wd)
        test_autopilot_argv()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
