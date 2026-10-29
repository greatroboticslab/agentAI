#!/usr/bin/env python3
"""DS1's long path (inc2/ds1_long.py, lever L32X; run_inc2_long.sh; docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-29): DS1", DS.6, DS.11).

- the recipe at a budget: epochs round_half_up(I / N) with N counting the floor's repeats, warm-up 640 iterations,
  close_mosaic 10 % of the epochs; S0x at 3.6 M about 91 epochs on S0's rows;
- the pair's compute decided once from S0x's smoke: 3.6 M when its projected seed is at most 19.2 h (0.8 x 24),
  else 2.4 M for S0x and S2x alike (pair.json written once); S2x's smoke at the pair's compute: <= 19.2 h 24 h,
  <= 28.8 h submitted once with 36 h, over it 'not run (walltime)'; a moved 1.2 M arm at 1.2 M; S0x over 19.2 h at
  2.4 M not run;
- the smoke times the epoch itself (from its start to its end, the setup apart); a seed trains once, after a complete
  smoke, with the walltime the smoke allows, on the manifest the smoke timed; its weights exported once and checked
  (source: the file's sha256 against train.json);
- run_inc2_long.sh: bash -n; its grammar (smoke --arm, train --arm --seed 0|1|2 [--walltime-h 24|36], any other arm,
  module or verb a usage error); the drift check and the provenance as the other job scripts.

Run:  python3 tests/test_inc2_ds1_long.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_long_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_data as DD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_long as LG  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402

FAILURES = []
DD.PROJECT_LOADER = False


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except D.Ds1Refused as e:
        return e
    return None


def write_arm(arm, n):
    rows = []
    d = TMP / "imgs"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        lab = d / ("%s_%d.txt" % (arm, i))
        lab.write_text("12 0.5 0.5 0.2 0.2\n")
        rows.append({"image": str(d / ("x%d.jpg" % i)), "label": str(lab), "sha256": "%064d" % i,
                     "label_sha256": C.sha256_file(lab), "source": "s", "session": "", "key": "%s_%05d" % (arm, i)})
    return DD.write_arm(arm, rows)


def stub_trainer(epoch_s):
    def train(rows, rcp, out_dir, rec, times):
        times.append((100.0, 100.0 + float(epoch_s)))
        out = pathlib.Path(out_dir) / "weights"
        out.mkdir(parents=True, exist_ok=True)
        w = out / "last.pt"
        w.write_bytes(b"weights %d %d" % (rcp["epochs"], rcp.get("seed") or 0))
        rec["weights_epoch"] = rcp["epochs"] - 1
        return w, out
    return train


def t_recipe():
    print("the recipe at a budget")
    r = LG.recipe(39500, D.PAIR_IMAGE_EPOCHS, seed=1)
    check("S0x at 3.6 M on S0's 39,500 rows: 91 epochs; warm-up 640 iterations; close_mosaic 10 %",
          r["epochs"] == 91 and r["close_mosaic"] == 9 and r["seed"] == 1 and
          abs(r["warmup_epochs"] - RC.budget_warmup_epochs(39500, r["batch"])) < 1e-12, r)
    check("the 2.4 M fallback: 61 epochs", LG.recipe(39500, D.PAIR_FALLBACK_IMAGE_EPOCHS)["epochs"] == 61)
    check("a moved 1.2 M arm keeps 1.2 M", LG.image_epochs("S1C") == D.BUDGET_IMAGE_EPOCHS)
    check("the pair's compute: 3.6 M at most 19.2 h, else 2.4 M", LG.pair_decision(19.2) == D.PAIR_IMAGE_EPOCHS and
          LG.pair_decision(19.21) == D.PAIR_FALLBACK_IMAGE_EPOCHS)
    check("S2x's walltime branches: <= 19.2 h -> 24 h; <= 28.8 h -> 36 h once; over -> not run (walltime)",
          LG.walltime_branch("S2X", 19.0)[0] == 24.0 and LG.walltime_branch("S2X", 25.0)[0] == 36.0 and
          LG.walltime_branch("S2X", 29.0)[0] is None and LG.walltime_branch("S2X", 29.0)[1].startswith(
              "not run (walltime)"))
    check("  S0x and a moved arm have no 36 h branch", LG.walltime_branch("S0X", 25.0)[0] is None and
          LG.walltime_branch("S1C", 20.0)[0] is None)
    check("S2x has no compute before S0x's smoke decides it", refused(LG.image_epochs, "S2X"))


def t_smoke_train():
    print("the smokes, the pair's compute and the seeds (a stub trainer)")
    D.write_json(D.timestamp_path(), {"format": D.FMT_TIMESTAMP, "sha256": D.PREREG_SHA256, "proofs": [
        {"file": "a.ots", "sha256": "a" * 64}, {"file": "b.ots", "sha256": "b" * 64}], "revisions": [
        {"file": r["file"], "sha256": r["sha256"], "proofs": [{"file": "c", "sha256": "c" * 64},
                                                             {"file": "d", "sha256": "d" * 64}]} for r in D.REVISIONS]})
    write_arm("S0", 400)
    check("no seed before its smoke", refused(LG.train, "S0X", 0, trainer=stub_trainer(10)))
    # 400 rows at 3.6 M: 9,000 epochs; an epoch of 10 s projects 25.5 h > 19.2: the pair falls back to 2.4 M
    sm = LG.smoke("S0X", trainer=stub_trainer(10.0))
    pair = json.loads(LG.pair_path().read_text())
    check("S0x's smoke over 19.2 h at 3.6 M: the pair falls back to 2.4 M, recorded once in pair.json",
          pair["image_epochs"] == D.PAIR_FALLBACK_IMAGE_EPOCHS and sm["pair_image_epochs"] == pair["image_epochs"])
    check("  the epoch is timed by itself (10 s), the setup apart", sm["epoch_seconds"] == 10.0)
    check("  at 2.4 M S0x projects 6,000 epochs x 10 s + 0.5 h = 17.2 h: within 24 h",
          abs(sm["projection_h"] - (6000 * 10.0 / 3600 + 0.5)) < 1e-9 and sm["walltime_h"] == 24.0 and
          sm["status"] == "complete")
    check("  a smoke is taken once", refused(LG.smoke, "S0X", trainer=stub_trainer(10.0)))
    tr = LG.train("S0X", 0, trainer=stub_trainer(10.0))
    check("a seed trains at the pair's compute, its weights exported once with train.json", tr["image_epochs"] ==
          D.PAIR_FALLBACK_IMAGE_EPOCHS and (LG.arm_dir("S0X") / "s0" / "final_ema.pt").is_file() and
          refused(LG.train, "S0X", 0, trainer=stub_trainer(10.0)))
    src = LG.source("S0X", 0)
    check("  its source names the file's sha256 as train.json records it", src["weights_sha256"] == C.sha256_file(
        LG.arm_dir("S0X") / "s0" / "final_ema.pt"))
    (LG.arm_dir("S0X") / "s0" / "final_ema.pt").write_bytes(b"other")
    check("  a changed weights file is refused", refused(LG.source, "S0X", 0))
    check("a walltime other than the smoke's is refused", refused(LG.train, "S0X", 1, walltime_h=36,
                                                                  trainer=stub_trainer(10.0)))
    write_arm("S2", 600)
    sm2 = LG.smoke("S2X", trainer=stub_trainer(11.0))
    check("S2x's smoke at the pair's 2.4 M: 4,000 epochs x 11 s + 0.5 = 12.7 h, 24 h", sm2["image_epochs"] ==
          D.PAIR_FALLBACK_IMAGE_EPOCHS and sm2["walltime_h"] == 24.0, sm2)
    write_arm("S1U", 300)
    sm3 = LG.smoke("S1UX", trainer=stub_trainer(10.0))
    check("S1ux's smoke over 19.2 h but within 28.8 h: submitted once with 36 h", sm3["walltime_h"] == 36.0 and
          abs(sm3["projection_h"] - (8000 * 10.0 / 3600 + 0.5)) < 1e-9, sm3)
    tr3 = LG.train("S1UX", 2, walltime_h=36, trainer=stub_trainer(10.0))
    check("  its seed trains with --walltime-h 36", tr3["seed"] == 2)
    write_arm("S1C", 500)
    sm4 = LG.smoke("S1C", trainer=stub_trainer(30.0))
    check("a moved 1.2 M arm at 1.2 M: 2,400 epochs x 30 s + 0.5 = 20.5 h, over 19.2: not run (walltime)",
          sm4["status"] == LG.NOT_RUN_WALLTIME and sm4["walltime_h"] is None and refused(
              LG.train, "S1C", 0, trainer=stub_trainer(30.0)))
    st = LG.status()
    check("status: the pair, each arm's smoke, projection, walltime and trained seeds", st["pair"]["image_epochs"] ==
          D.PAIR_FALLBACK_IMAGE_EPOCHS and st["arms"]["S0X"]["trained"] == [0] and st["arms"]["S1UX"]["walltime_h"] ==
          36.0)


def t_job_script():
    print("run_inc2_long.sh")
    sh = PKG / "run_inc2_long.sh"
    check("bash -n", subprocess.run(["bash", "-n", str(sh)]).returncode == 0)
    text = sh.read_text()
    check("24 h by default; one V100-32; the module list includes DS1's modules; the drift check and provenance",
          "#SBATCH --time=24:00:00" in text and "v100-32:1" in text and "tools/inc2/ds1_long.py" in text and
          "INC_BUILD_ALLOW_DRIFT" in text and "_campaign/provenance" in text)
    env = dict(os.environ, INC_BUILD_REPO=str(TMP / "norepo"))
    for args, ok in ((["inc2.ds1_long", "smoke", "--arm", "S9X"], False), (["inc2.ds1_long", "train", "--arm",
                                                                             "S0X"], False),
                     (["inc2.ds1_long", "train", "--arm", "S0X", "--seed", "3"], False),
                     (["inc2.ds1_long", "train", "--arm", "S0X", "--seed", "1", "--walltime-h", "48"], False),
                     (["inc2.v3fam_detr", "smoke"], False), (["inc2.ds1_long", "verdict", "--arm", "S0X"], False)):
        p = subprocess.run(["bash", str(sh)] + args, env=env, capture_output=True, text=True)
        check("  %s: a usage error (exit 2)" % " ".join(args), p.returncode == 2, (p.returncode, p.stderr[-200:]))
    p = subprocess.run(["bash", str(sh), "inc2.ds1_long", "smoke", "--arm", "S0X"], env=env, capture_output=True,
                       text=True)
    check("  a valid smoke argv passes the grammar (then stops: no repo here)", p.returncode == 1, p.stderr[-200:])


def main():
    for fn in (t_recipe, t_smoke_train, t_job_script):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2000:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
