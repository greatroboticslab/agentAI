#!/usr/bin/env python3
"""The Protocol v3 / splits v2 executor (inc2/train.py), the recipe table
(inc2/recipes.py), the scorer sidecar wiring and run_inc2_job.sh
(docs/CONTINUOUS_LOOP.md §4.3, §5.1, §8, §9 group B).

A synthetic world is built in a temporary INC_DIR and REPO: v1 splits (dev,
test, imageweeds, ood22, ood23 manifests, the materialised exams the locked
scorer reads, the v1 LOCK with scorer.py's sha256, the v1 never-train index)
and splits v2 as group A's inc2.splits lays them out (byte copies of dev, test
and imageweeds, train_core, tsw22, base_v2, the v2 never-train and base-copy
indexes, LOCK v2 marked testing). The real inc2.guard.GuardV2 is used. Runs
train for real on the CPU (1 epoch at imgsz 64, a testing experiment:
INC_SCORER_TESTING=1), and the pinned scorer and the sidecar run as
subprocesses.

What is pinned:
- the v3 table: cold, r0, x1a, x1b on every arm have no deviation; x1a / x1b
  are the contract's values; freeze and lora are refused (L-6); a cold recipe
  on a cand, x1b on a base, imgsz 1024 on n640 or s640 are deviations; every
  arm's cold trains at 640 (capacity only, no resolution arm);
- specs: ood22 / ood23 exams are refused, test only for kind final;
- the evaluation manifests of v1 and v2 are refused by path and by content;
- guard_rows: a planted exact dev copy, a test copy 1-6 dHash bits away, a
  re-encoded test copy, an hflip test copy and a rot90 dev copy are each
  refused (fail closed) and
  the refusal is a failed run.json at stage guard; an unreadable image is
  refused as unhashable; a missing guard module, a missing LOCK v2, an index
  that is not the LOCK's and (production) a testing LOCK each stop the run;
  the index cross-check refuses an hflip copy even when GuardV2 passes it;
  base rows are counted as base_copy, not refused; an image whose bytes are
  an L-5 exclusion (l5_excluded.jsonl, sha256 in LOCK v2) is refused, an L-5
  list that does not hash as LOCK v2 records stops the run, and a production
  LOCK v2 that records no L-5 list is refused; likewise for the L-8 list
  (train_core_variant_drops.jsonl: the world's v1 train_core holds a
  transverse copy of a test image that v2's train_core and base_v2 leave
  out): its bytes re-listed under another key are refused by the list and
  by GuardV2, by the list alone when GuardV2 is patched to pass, a changed
  list stops the run and a production LOCK without the record is refused;
- the arm: no arm means n640 (with a warning) only for yolo11n.pt; a cold run
  whose init is not the arm's checkpoint (name or pinned sha256) departs;
- end to end on the CPU: a base run, a cand from it, a null and a final run
  finish; run.json says protocol v3 / inc2, records the arm, the guard and
  the code of every tools/inc2 module; base and cand runs carry a scorer
  sidecar (JSON + npz, hashes recorded), null and final runs none; a done run
  is a no-op, and a tampered sidecar makes the next attempt re-score, not
  retrain;
- production: a tiny recipe is refused at stage recipe, a cold init whose
  sha256 is not the arm's pin is refused at stage recipe, and the protocol's
  recipe reaches stage device (no CUDA here);
- run_inc2_job.sh: bash -n; exports INC_JOB_SCRIPT as the git-tracked copy of
  itself before anything runs; runs inc2.train --spec, then (INC_JOB_ADVANCE)
  inc.driver advance; flock-check through inc2.train; exits with the
  executor's status; an outer module that differs from the nested copy stops
  it (INC_ALLOW_DRIFT=1 runs it); no git reset, sync or Roboflow in it;
- the pinned INC modules, realloop.py and pilot.py are unchanged against git HEAD.

Run:  python3 tests/test_inc2_train.py
"""
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_inc2_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["INC_SCORER_TESTING"] = "1"
for _k in ("INC_ALLOW_DRIFT", "INC_JOB_ADVANCE", "INC_JOB_SCRIPT", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
           "SLURM_ARRAY_TASK_ID", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_ON_NODE",
           "SLURM_CPUS_PER_TASK"):
    os.environ.pop(_k, None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402

FAILURES = []
EXP = "t2_exp"
EXP_PROD = "t2_prod"
TESTING = {"imgsz": 64, "batch": 8, "device": "cpu"}
RECIPE = {"trainer": "full", "epochs": 1, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01, "momentum": 0.9,
          "weight_decay": 0.0005, "warmup_epochs": 1, "warmup_bias_lr": 0.01, "cos_lr": True, "freeze": None,
          "lora": None, "imgsz": 64, "batch": 8, "seed": 0, "cache": False, "workers": 0, "close_mosaic": 0,
          "deterministic": True}
PINNED = ("tools/inc/__init__.py", "tools/inc/common.py", "tools/inc/driver.py", "tools/inc/gate.py",
          "tools/inc/splits.py", "tools/inc/scorer.py", "tools/inc/lora.py", "tools/inc/train.py",
          "tools/inc/verify.py", "tools/inc/select.py", "tools/inc/relevance.py", "tools/inc/audit.py",
          "tools/cwd12_species.py", "tools/inc/realloop.py", "tools/inc/pilot.py")


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def sha(path):
    return C.sha256_file(path)


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------- world
def make_image(path, seed):
    """96x96 JPEG: a random low-frequency background and 1-2 rectangles."""
    rng = np.random.RandomState(seed)
    low = rng.randint(30, 220, (8, 9)).astype(np.uint8)
    im = Image.fromarray(low).resize((96, 96), Image.BICUBIC).convert("RGB")
    draw = ImageDraw.Draw(im)
    boxes = []
    for _ in range(1 + seed % 2):
        c = int(rng.randint(0, C.NC))
        w, h = rng.randint(20, 40, 2)
        x, y = rng.randint(0, 96 - w), rng.randint(0, 96 - h)
        draw.rectangle([x, y, x + w, y + h], fill=tuple(int(v) for v in rng.randint(0, 255, 3)))
        boxes.append((c, (x + w / 2) / 96, (y + h / 2) / 96, w / 96, h / 96))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return boxes


def row_for(key, image, boxes, source, session=""):
    lab = TMP / "src" / "labels" / (key + ".txt")
    C.write_yolo(lab, boxes)
    return {"image": str(image), "label": str(lab), "sha256": sha(image), "label_sha256": sha(lab),
            "source": source, "session": session, "key": key}


def make_rows(prefix, n, seed0, source, session=True):
    rows = []
    for i in range(n):
        key = "%s_%03d" % (prefix, i)
        img = TMP / "src" / prefix / (key + ".jpg")
        rows.append(row_for(key, img, make_image(img, seed0 + i), source, ("s%d" % (i % 3)) if session else ""))
    return rows


def cold_checkpoint(path, seed=0):
    """yolo11n.yaml at 13 classes, class biases raised so an untrained model
    predicts boxes (the scorer needs predictions)."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(seed)
    net = DetectionModel("yolo11n.yaml", nc=C.NC, verbose=False)
    det = net.model[-1]
    with torch.no_grad():
        for level in range(det.nl):
            det.cv3[level][-1].bias.fill_(-1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net, "train_args": {}}, path)
    return path


def v2_dir():
    return C.INC_DIR / "splits" / "v2"


def write_index(path, entries):
    data = {"entries": entries, "bits": C.HOLDOUT_NEAR_DUP_BITS, "complete": True, "min_expected": len(entries)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
    return sha(path)


def build_world():
    """v1 splits + exams + LOCK, and splits v2 laid out as inc2.splits locks
    them (LOCK v2 marked testing). Returns {name: rows}."""
    W = {"dev": make_rows("dv", 8, 100, "dev", session=False),
         "test": make_rows("te", 6, 200, "test", session=False),
         "imageweeds": make_rows("iw", 4, 300, "imageweeds", session=False),
         "ood22": make_rows("o22", 3, 400, "ood22", session=False),
         "ood23": make_rows("o23", 3, 450, "ood23", session=False),
         "train_core": make_rows("tc", 16, 0, "cottonweeddet12/train"),
         "tsw22": make_rows("tsw22__f", 6, 500, "3seasonweeddet10/data2022"),
         "inc": make_rows("nw", 6, 600, "new_source")}
    # L-8: v1's train_core also holds a transverse copy of a test image (v1 compared the stored dHash only);
    # splits v2 drops it from train_core and base_v2 and lists it in train_core_variant_drops.jsonl
    xv = planted("tc_xv", W["test"][5]["image"], "transverse")
    W["variant_drops"] = [row_for("tc_xv", xv, [(0, 0.5, 0.5, 0.2, 0.2)], "cottonweeddet12/train", "s0")]
    W["train_core_v1"] = W["train_core"] + W["variant_drops"]
    v1 = {}
    for split in ("dev", "test", "imageweeds", "ood22", "ood23"):
        v1[split] = C.write_manifest(C.manifest_path(split), W[split])
    v1["train_core"] = C.write_manifest(C.manifest_path("train_core"), W["train_core_v1"])
    for split in ("dev", "test", "imageweeds"):
        C.materialise(W[split], C.EXAMS_DIR / split)
    nt1 = [[C.dhash(r["image"]), s, r["key"]] for s in ("dev", "test", "ood22", "ood23", "imageweeds") for r in W[s]]
    nt1_sha = write_index(C.NEVER_TRAIN_INDEX, nt1)
    C.LOCK_PATH.write_text(json.dumps({"manifests": v1, "scorer_sha256": sha(pathlib.Path(S.__file__).resolve()),
                                       "nevertrain_sha256": nt1_sha, "splits_version": "v1"}))
    d = v2_dir()
    d.mkdir(parents=True, exist_ok=True)
    v2 = {}
    for split in ("dev", "test", "imageweeds"):
        shutil.copyfile(C.manifest_path(split), d / ("%s.jsonl" % split))
        v2[split] = sha(d / ("%s.jsonl" % split))
    drops = [dict(r, dhash=C.dhash(r["image"]), reason="near_eval_variant",
                  match={"split": "test", "key": W["test"][5]["key"], "variant": "transverse"})
             for r in W["variant_drops"]]
    (d / "train_core_variant_drops.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in drops))
    tc_v1 = C.manifest_path("train_core").read_bytes()
    (d / "train_core.jsonl").write_bytes(b"".join(ln for ln in tc_v1.splitlines(keepends=True)
                                                  if json.loads(ln)["key"] != "tc_xv"))
    v2["train_core"] = sha(d / "train_core.jsonl")
    v2["tsw22"] = C.write_manifest(d / "tsw22.jsonl", W["tsw22"])
    v2["tsw23"] = C.write_manifest(d / "tsw23.jsonl", [])
    W["base_v2"] = W["train_core"] + W["tsw22"]
    v2["base_v2"] = C.write_manifest(d / "base_v2.jsonl", W["base_v2"])
    nt = [[C.dhash(r["image"]), s, r["key"]] for s in ("dev", "test", "imageweeds") for r in W[s]]
    bc = [[C.dhash(r["image"]), "train_core" if r["key"].startswith("tc") else "tsw22", r["key"]]
          for r in W["base_v2"]]
    nt_sha = write_index(d / "nevertrain_dhash.json", nt)
    bc_sha = write_index(d / "base_copies_dhash.json", bc)
    (d / "LOCK.json").write_text(json.dumps({
        "splits_version": "v2", "manifests": v2, "eval_splits": ["dev", "test", "imageweeds"],
        "train_splits": ["train_core", "tsw22", "tsw23"], "base_manifest": "base_v2",
        "final_exams": ["dev", "imageweeds", "test"], "nevertrain_sha256": nt_sha, "nevertrain_entries": len(nt),
        "base_copies_sha256": bc_sha, "base_copies_entries": len(bc),
        "train_core_variant_drops_sha256": sha(d / "train_core_variant_drops.jsonl"),
        "scorer_sha256": sha(pathlib.Path(S.__file__).resolve()), "testing": True}, sort_keys=True))
    cold_checkpoint(C.REPO / "yolo11n.pt")
    for exp, testing in ((EXP, TESTING), (EXP_PROD, None)):
        (C.INC_DIR / exp).mkdir(parents=True, exist_ok=True)
        defn = {"exp": exp}
        if testing:
            defn["testing"] = testing
        (C.INC_DIR / exp / "exp.json").write_text(json.dumps(defn))
    return W


def write_manifest(exp, name, rows):
    path = C.INC_DIR / exp / "manifests" / ("%s.jsonl" % name)
    C.write_manifest(path, rows)
    return path


def spec(run_id, kind, exams=("dev",), exp=EXP, **kw):
    out = C.INC_DIR / exp / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    s = {"exp": exp, "run_id": run_id, "kind": kind, "exams": list(exams), "out_dir": str(out)}
    s.update(kw)
    path = out / "spec.json"
    path.write_text(json.dumps(s, indent=1))
    return path


def recipe(**over):
    return dict(RECIPE, **over)


def run(path, *extra):
    return T.main(["--spec", str(path)] + list(extra))


def run_json(path):
    p = pathlib.Path(path).parent / T.RUN_JSON
    return json.loads(p.read_text()) if p.exists() else {}


def planted(key, src_image, how):
    """A copy of src_image: 'exact' (same bytes), 'reencode' (JPEG q60),
    'hflip', 'rot90', 'transverse'. Returns its path."""
    dst = TMP / "planted" / ("%s.jpg" % key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if how == "exact":
        shutil.copyfile(src_image, dst)
        return dst
    with Image.open(src_image) as im:
        im = im.convert("RGB")
        if how == "hflip":
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif how == "rot90":
            im = im.transpose(Image.Transpose.ROTATE_90)
        elif how == "transverse":
            im = im.transpose(Image.Transpose.TRANSVERSE)
        im.save(dst, quality=60 if how == "reencode" else 95)
    return dst


def bits(a, b):
    return bin(int(a) ^ int(b)).count("1")


def near_copy(key, src_image, lo=1, hi=C.HOLDOUT_NEAR_DUP_BITS):
    """A copy of src_image whose dHash is lo..hi bits from the original's:
    a small patch of the image is shifted in brightness, more each try."""
    ref = C.dhash(src_image)
    dst = TMP / "planted" / ("%s.png" % key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_image) as im:
        base = np.asarray(im.convert("RGB"), dtype=np.int16)
    for k in range(1, 200):
        a = base.copy()
        a[:, : 4 + k % 40] += (k // 40 + 1) * 25
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(dst)
        if lo <= bits(C.dhash(dst), ref) <= hi:
            return dst, bits(C.dhash(dst), ref)
    return None, None


# ------------------------------------------------------------------- tests
def test_pinned_unchanged():
    print("pinned modules")
    pkg = ROOT / "weed_optimizer_framework"
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"] + [str(pkg / m) for m in PINNED],
                       cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode not in (0, 1):
        print("  NOTE git is not usable here (%s); the pinned-module check is skipped" % r.stderr.strip()[:200])
        return
    check("the pinned INC modules, realloop.py and pilot.py are unchanged against git HEAD", r.returncode == 0,
          r.stdout)


def test_recipes():
    print("the Protocol v3 recipe table (inc2.recipes)")
    for arm in RC.ARM_IDS:
        tab = RC.table(arm)
        devs = {("base", arm): RC.deviations("base", tab["cold"], arm),
                ("union", arm): RC.deviations("union", tab["cold"], arm)}
        for n in RC.INCREMENTAL_NAMES:
            for kind in ("cand", "null"):
                devs[(kind, n, arm)] = RC.deviations(kind, dict(tab[n], seed=3, cache=False, workers=0), arm)
        check("arm %s: cold (base, union) and r0 / x1a / x1b (cand, null) have no deviation; seed, cache and "
              "workers are not compared" % arm, not any(devs.values()), {k: v for k, v in devs.items() if v})
    x1a, x1b, r0 = RC.incremental("x1a"), RC.incremental("x1b"), RC.incremental("r0")
    check("x1a is 30 epochs, warmup 3, peak lr0 0.005, cosine to lrf 0.01; x1b 50 epochs, peak lr0 0.01; r0 is "
          "inc.pilot's full recipe; warmup_bias_lr = lr0",
          (x1a["epochs"], x1a["warmup_epochs"], x1a["lr0"], x1a["lrf"], x1a["cos_lr"]) == (30, 3, 0.005, 0.01, True)
          and (x1b["epochs"], x1b["lr0"], x1b["warmup_epochs"]) == (50, 0.01, 3)
          and all(r["warmup_bias_lr"] == r["lr0"] for r in (x1a, x1b, r0)), (x1a, x1b))
    from weed_optimizer_framework.tools.inc import pilot as P
    check("r0 and cold equal the pinned pilot's full and cold recipes (n640)",
          r0 == P.inc_recipes()["full"] and RC.cold("n640") == P.cold_recipe())
    freeze, lora = P.inc_recipes()["freeze"], P.inc_recipes()["lora"]
    dz, dl = RC.deviations("cand", freeze), RC.deviations("cand", lora)
    check("freeze and lora are deviations naming L-6", dz and dl and "L-6" in dz[0] and "L-6" in dl[0], (dz, dl))
    keys = lambda d: {x.split()[0] for x in d if not x.startswith("nearest")}  # noqa: E731
    check("a cold recipe on a cand, x1b on a base, imgsz 1024 on n640 or s640 are deviations; every arm's cold "
          "trains at 640",
          RC.deviations("cand", RC.cold()) and keys(RC.deviations("base", x1b)) >= {"epochs", "warmup_bias_lr"}
          and keys(RC.deviations("cand", dict(r0, imgsz=1024))) == {"imgsz"}
          and all(RC.cold(a)["imgsz"] == 640 and not RC.deviations("base", RC.cold(a), a) for a in RC.ARM_IDS)
          and keys(RC.deviations("base", dict(RC.cold("s640"), imgsz=1024), "s640")) == {"imgsz"})
    check("match names the table entry", RC.match("cand", x1a) == "x1a" and RC.match("base", RC.cold()) == "cold"
          and RC.match("cand", freeze) is None)
    try:
        RC.incremental("lora")
        e = None
    except RC.RecipeError as x:
        e = x
    check("recipes.incremental refuses lora (L-6)", e is not None and "L-6" in str(e), e)
    c = RC.baseline_cost(7625, [0, 1, 2, 3, 4], ["dev", "imageweeds", "test"], "n640")
    s = RC.baseline_cost(7625, [0, 1, 2], ["dev", "imageweeds", "test"], "s640")
    m = RC.baseline_cost(7625, [0, 1, 2], ["dev", "imageweeds", "test"], "m640")
    check("cost: B_v2 at n640 (7,625 images, 5 seeds) is est. about 6.9-7.9 GPU-h as the contract says "
          "(%s); s640 is bracketed by the pixel and FLOPs ratios (%.2f, %.3f) and stays under D26's line; "
          "m640's high bracket crosses it" % (c["total_gpu_h"], s["pixel_factor"], s["flops_factor"]),
          c["estimate"] and 6.6 <= c["total_gpu_h"][0] <= 7.2 and 7.5 <= c["total_gpu_h"][1] <= 8.2
          and s["pixel_factor"] == 1.0 and abs(s["flops_factor"] - 21.574 / 6.454) < 1e-3
          and s["walltime"]["over_d26_line"] == [False, False] and m["walltime"]["over_d26_line"] == [False, True],
          (c, s["walltime"], m["walltime"]))
    step = RC.step_cost(7625, 763, "n640", "r0", truth=True)
    check("a n640 r0 step with its truth arm is est. 7.2-8.4 GPU-h (3.0 chain + 4.2-4.9 truth, 5.6) and runs "
          "truth every step; 61 GPU-h would run it every 3rd", 7.0 <= step["gpu_h"][0] and step["gpu_h"][1] <= 8.5
          and RC.truth_every(step["gpu_h"][1]) == 1 and RC.truth_every(61.0) == 3, step)
    def entry(chain, inc, nulls, p_recipe):
        return {"type": "gate", "chain": chain, "decision": {"inc": inc, "null_values": nulls, "p_recipe": p_recipe}}

    led = [entry("r0", 0.60, [0.58, 0.585, 0.59], 0.0), entry("r0", 0.61, [0.60, 0.60, 0.60], 0.0),
           entry("x1a", 0.60, [0.60, 0.605, 0.598], 0.5), entry("x1a", 0.61, [0.61, 0.612, 0.609], 0.6)]
    ch = RC.stage_b_choice(led, {"r0": "r0", "x1a": "x1a"})
    tie = RC.stage_b_choice([entry("x1a", 0.6, [0.6, 0.6, 0.6], 0.5), entry("r0", 0.6, [0.6, 0.6, 0.6], 0.5)],
                            {"x1a": "x1a", "r0": "r0"})
    flags = RC.stage_b_choice([entry("r0", 0.6, [0.6, 0.6, 0.6], 0.0), entry("x1b", 0.6, [0.6, 0.6, 0.6], 0.5)],
                              {"r0": "r0", "x1b": "x1b"})
    check("Stage B: the smallest median delta_min wins (x1a %.4f vs r0 %.4f); then fewer recipe flags (x1b over "
          "a flagged r0); a full tie goes to r0" % (ch["table"]["x1a"]["median_delta_min"],
                                                    ch["table"]["r0"]["median_delta_min"]),
          ch["chosen"] == "x1a" and abs(RC.delta_min(0.60, [0.58, 0.585, 0.59]) - 0.005) < 1e-12
          and flags["chosen"] == "x1b" and tie["chosen"] == "r0", (ch, flags["table"], tie["order"]))
    rec = RC.resolve_arm("s640", repo=C.REPO, require_weights=False)
    check("an arm record without its checkpoint pins no sha256 (testing); requiring it refuses",
          rec["weights_sha256"] is None and rec["model"] == "yolo11s.pt")
    try:
        RC.resolve_arm("s640", repo=C.REPO)
        e = None
    except RC.RecipeError as x:
        e = x
    check("... 'nothing is downloaded'", e is not None and "nothing is downloaded" in str(e), e)


def test_specs_and_manifests(W):
    print("specs, evaluation manifests, the arm")
    base = dict(init="yolo11n.pt", train_manifest=str(v2_dir() / "base_v2.jsonl"), recipe=recipe())
    cases = [("ood22 exam", spec("r_ood", "base", exams=("dev", "ood22"), **base), "not splits v2 evaluation"),
             ("ood23 on a final", spec("r_ood2", "final", exams=("ood23",), init="yolo11n.pt"), "not splits v2"),
             ("test on a cand", spec("r_test", "cand", exams=("dev", "test"), **base), "sealed test")]
    for name, p, needle in cases:
        rc = run(p)
        rj = run_json(p)
        check("refused at spec: %s" % name, rc == 1 and rj.get("stage") == "spec" and needle in rj.get("error", ""),
              (rc, rj.get("stage"), (rj.get("error") or "")[-300:]))
    ok = T.validate_spec(json.loads(spec("r_fin", "final", exams=("dev", "imageweeds", "test"),
                                         init="yolo11n.pt").read_text()),
                         C.INC_DIR / EXP / "runs" / "r_fin" / "spec.json")
    check("a final spec may list dev, imageweeds and test", ok["exams"] == ["dev", "imageweeds", "test"])
    mods = T.code_modules()
    want = sorted("tools/inc2/%s" % p.name for p in (ROOT / "weed_optimizer_framework" / "tools" / "inc2").glob("*.py"))
    check("code_modules: v1's executor modules, funnel/leak.py and every tools/inc2 module",
          all(m in mods for m in want) and "tools/funnel/leak.py" in mods and "tools/inc/train.py" not in mods
          and "tools/inc/scorer.py" in mods, mods)

    copy = TMP / "elsewhere" / "dev_copy.jsonl"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(C.manifest_path("dev"), copy)
    for what, p in (("v1 dev", C.manifest_path("dev")), ("v1 ood22", C.manifest_path("ood22")),
                    ("v2 dev", v2_dir() / "dev.jsonl"), ("v2 imageweeds", v2_dir() / "imageweeds.jsonl"),
                    ("a copy of dev elsewhere (content)", copy)):
        try:
            T.check_manifest(p)
            e = None
        except T.RunError as x:
            e = x
        check("check_manifest refuses %s" % what, e is not None and e.stage == "manifest"
              and ("never trained on" in str(e)), e)
    rows, dh, info = T.check_manifest(v2_dir() / "base_v2.jsonl")
    check("base_v2 (train_core + tsw22) passes check_manifest", len(rows) == len(W["base_v2"]), info)

    arm, warns = T.experiment_arm(EXP)
    check("an exp.json without 'arm' is the continuity arm n640, with a warning", arm["id"] == "n640" and warns)
    d = C.INC_DIR / "t2_arm"
    d.mkdir(parents=True, exist_ok=True)
    (d / "exp.json").write_text(json.dumps({"exp": "t2_arm", "init_weights": "yolo11s.pt"}))
    try:
        T.experiment_arm("t2_arm")
        e = None
    except T.RunError as x:
        e = x
    check("... but not with another init_weights", e is not None and "must pin its arm" in str(e), e)
    bad = dict(RC.resolve_arm("n640", require_weights=False), imgsz=1024)
    (d / "exp.json").write_text(json.dumps({"exp": "t2_arm", "arm": bad}))
    try:
        T.experiment_arm("t2_arm")
        e = None
    except T.RunError as x:
        e = x
    check("an arm record whose fixed fields differ from the table is refused", e is not None and "imgsz" in str(e), e)
    rec = RC.resolve_arm("n640", repo=C.REPO)
    check("resolve_arm pins the checkpoint's sha256", rec["weights_sha256"] == sha(C.REPO / "yolo11n.pt"))
    check("init_check: a cold run from the arm's checkpoint passes; another name or another sha256 departs; a "
          "cand is not checked",
          T.init_check("base", "yolo11n.pt", rec) == [] and T.init_check("base", "yolo11s.pt", rec)
          and T.init_check("union", "yolo11n.pt", dict(rec, weights_sha256="0" * 64))
          and T.init_check("cand", "/x/y.pt", rec) == [])


def test_guard(W):
    print("the splits v2 never-train guard")
    base = W["base_v2"]
    rows, dh, info = T.check_manifest(v2_dir() / "base_v2.jsonl")
    rec = T.guard_rows(rows, dh, production=False)
    check("a clean base_v2: nothing refused, every base row counted as base_copy",
          rec["refused"] == 0 and rec["crosscheck_hits"] == 0 and rec["reasons"].get("base_copy") == len(base)
          and rec["checked"] == len(base) and rec["index_sha256"] == sha(v2_dir() / "nevertrain_dhash.json"), rec)
    re_img = planted("te_reenc", W["test"][1]["image"], "reencode")
    b = bits(C.dhash(re_img), C.dhash(W["test"][1]["image"]))
    check("fixture: the re-encoded test copy is a different file within 6 dHash bits (%d)" % b,
          sha(re_img) != W["test"][1]["sha256"] and b <= 6)
    nc, nb = near_copy("te_near", W["test"][5]["image"])
    check("fixture: a test copy %s dHash bits from its original (1-6)" % nb, nc is not None)
    if nc is not None:
        m = write_manifest(EXP, "planted_near", base + [row_for("pl_near", nc, [(0, 0.5, 0.5, 0.2, 0.2)], "planted")])
        rows, dh, _ = T.check_manifest(m)
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
        check("guard_rows refuses a planted copy %d bits from a test image (near_eval_v2), fail closed" % nb,
              e is not None and (getattr(e, "guard", {}) or {}).get("reasons", {}).get("near_eval_v2") == 1, e)
    plants = {"exact dev copy": ("dv_exact", W["dev"][0]["image"], "exact", "near_eval_v2"),
              "re-encoded test copy": ("te_reenc", W["test"][1]["image"], "reencode", "near_eval_v2"),
              "hflip test copy": ("te_hflip", W["test"][2]["image"], "hflip", "near_eval_variant"),
              "rot90 dev copy": ("dv_rot90", W["dev"][3]["image"], "rot90", "near_eval_variant")}
    for name, (key, src, how, reason) in plants.items():
        img = planted(key, src, how)
        row = row_for("pl_" + key, img, [(0, 0.5, 0.5, 0.2, 0.2)], "planted")
        m = write_manifest(EXP, "planted_" + key, base + [row])
        rows, dh, _ = T.check_manifest(m)
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
        g = getattr(e, "guard", {}) or {}
        check("guard_rows refuses a planted %s (%s), fail closed" % (name, reason),
              e is not None and e.stage == "guard" and g.get("reasons", {}).get(reason) == 1
              and g.get("refused") == 1, (e, g.get("reasons")))
    p = spec("r_planted", "cand", train_manifest=str(write_manifest(EXP, "planted_run", base + [
        row_for("pl_run", planted("te_hflip2", W["test"][0]["image"], "hflip"), [(1, 0.5, 0.5, 0.3, 0.3)],
                "planted")])), init="yolo11n.pt", recipe=recipe())
    rc = run(p)
    rj = run_json(p)
    check("a run on it fails at stage guard with the guard record in run.json, before any training",
          rc == 1 and rj.get("stage") == "guard" and (rj.get("guard") or {}).get("refused") == 1
          and not rj.get("trained"), (rc, rj.get("stage"), rj.get("guard")))
    junk = TMP / "planted" / "junk.jpg"
    junk.write_bytes(b"not an image at all" * 20)
    m = write_manifest(EXP, "junk", base + [row_for("pl_junk", junk, [(0, 0.5, 0.5, 0.2, 0.2)], "planted")])
    rows, dh, _ = T.check_manifest(m)
    try:
        T.guard_rows(rows, dh, production=False)
        e = None
    except T.RunError as x:
        e = x
    check("an image that cannot be hashed is refused (unhashable)",
          e is not None and (getattr(e, "guard", {}) or {}).get("reasons", {}).get("unhashable") == 1, e)

    rows, dh, _ = T.check_manifest(v2_dir() / "base_v2.jsonl")
    key = "weed_optimizer_framework.tools.inc2.guard"
    saved = sys.modules.get(key)
    sys.modules[key] = None
    try:
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    finally:
        if saved is not None:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)
    check("no guard module: the run stops at stage guard (a guard that is not there clears nothing)",
          e is not None and e.stage == "guard" and "fails closed" in str(e), e)
    try:
        T.guard_rows(rows, dh, production=True)
        e = None
    except T.RunError as x:
        e = x
    check("a production run refuses a LOCK v2 written by a testing build", e is not None and "testing build" in str(e), e)
    idx = v2_dir() / "nevertrain_dhash.json"
    saved_idx = idx.read_bytes()
    idx.write_bytes(saved_idx.replace(b'"complete": true', b'"complete": true ') + b" ")
    try:
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    finally:
        idx.write_bytes(saved_idx)
    check("a never-train index that is not the one LOCK v2 records stops the run", e is not None
          and "LOCK records" in str(e), e)
    lock = v2_dir() / "LOCK.json"
    saved_lock = lock.read_bytes()
    lock.unlink()
    try:
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    finally:
        lock.write_bytes(saved_lock)
    check("no LOCK v2 stops the run", e is not None and "locked splits" in str(e), e)

    l5_rows = make_rows("l5cw", 2, 900, "rf_karthikeya-c8pvy__weed-detection-cwp10")
    l5_path = v2_dir() / "l5_excluded.jsonl"
    l5_path.write_text("".join(json.dumps({"key": r["key"], "source": r["source"], "image": r["image"],
                                           "sha256": r["sha256"]}) + "\n" for r in l5_rows))
    lk = json.loads(saved_lock)
    lock.write_text(json.dumps(dict(lk, l5_excluded_sha256=sha(l5_path)), sort_keys=True))
    relisted = dict(l5_rows[0], key="elsewhere_0001", source="some_reupload")
    m = write_manifest(EXP, "l5", base + [relisted])
    rows_l5, dh_l5, _ = T.check_manifest(m)
    try:
        try:
            T.guard_rows(rows_l5, dh_l5, production=False)
            e = None
        except T.RunError as x:
            e = x
        g = getattr(e, "guard", {}) or {}
        check("an L-5 excluded image re-listed under another key and source is refused (l5_excluded)",
              e is not None and g.get("reasons", {}).get("l5_excluded") == 1 and g.get("refused") == 1
              and (g.get("l5_excluded") or {}).get("images") == 2, (e, g.get("reasons")))
        rec_ok = T.guard_rows(rows, dh, production=False)
        check("... while base_v2 passes, the L-5 list recorded in the guard record",
              rec_ok["refused"] == 0 and rec_ok["l5_excluded"]["sha256"] == sha(l5_path), rec_ok.get("l5_excluded"))
        l5_path.write_text(l5_path.read_text() + "\n")
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
        check("an L-5 list that does not hash as LOCK v2 records stops the run", e is not None
              and "l5_excluded.jsonl" in str(e), e)
        lock.write_text(json.dumps(dict(lk, testing=False), sort_keys=True))
        try:
            T.guard_rows(rows, dh, production=True)
            e = None
        except T.RunError as x:
            e = x
        check("a production LOCK v2 that records no L-5 list is refused", e is not None
              and "no l5_excluded_sha256" in str(e), e)
    finally:
        lock.write_bytes(saved_lock)
        l5_path.unlink()

    test_guard_variant_drops(W, base, rows, dh, lock, saved_lock)
    g2 = T.guard_module()
    img = planted("te_hflip3", W["test"][4]["image"], "hflip")
    m = write_manifest(EXP, "cross", base + [row_for("pl_cross", img, [(0, 0.5, 0.5, 0.2, 0.2)], "planted")])
    rows, dh, _ = T.check_manifest(m)
    with patched(g2.GuardV2, "check", lambda self, d, v=None: (None, None)):
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    g = getattr(e, "guard", {}) or {}
    check("the index cross-check refuses an hflip copy even when GuardV2 passes everything",
          e is not None and g.get("refused") == 0 and g.get("crosscheck_hits") == 1, (e, g))
    with patched(g2.GuardV2, "check", lambda self, d, v=None: ("some_new_reason", None)):
        try:
            T.guard_rows(rows[:1], dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    check("a GuardV2 reason the executor does not know refuses (fail closed)", e is not None
          and "some_new_reason" in str(e), e)


def test_guard_variant_drops(W, base, rows, dh, lock, saved_lock):
    print("the L-8 list (train_core_variant_drops.jsonl)")
    drop = W["variant_drops"][0]
    lst = v2_dir() / T.VARIANT_DROPS_NAME
    check("fixture: the L-8 image is in v1's train_core, not in v2's train_core or base_v2",
          drop["key"] in {r["key"] for r in C.read_manifest(C.manifest_path("train_core"))}
          and drop["sha256"] not in {r["sha256"] for m in ("train_core", "base_v2")
                                     for r in C.read_manifest(v2_dir() / ("%s.jsonl" % m))})
    relisted = dict(drop, key="elsewhere_xv_0001", source="some_reupload")
    m = write_manifest(EXP, "l8", base + [relisted])
    rows_l8, dh_l8, _ = T.check_manifest(m)
    try:
        T.guard_rows(rows_l8, dh_l8, production=False)
        e = None
    except T.RunError as x:
        e = x
    g = getattr(e, "guard", {}) or {}
    check("an L-8 drop re-listed under another key and source is refused by its bytes (train_core_variant_drop) and "
          "by GuardV2 (near_eval_variant)",
          e is not None and e.stage == "guard" and g.get("reasons", {}).get("train_core_variant_drop") == 1
          and g.get("reasons", {}).get("near_eval_variant") == 1 and g.get("refused") == 2
          and (g.get("train_core_variant_drops") or {}).get("keys") == [drop["key"]], (e, g.get("reasons")))
    g2 = T.guard_module()
    with patched(g2.GuardV2, "check", lambda self, d, v=None: (None, None)):
        try:
            T.guard_rows(rows_l8, dh_l8, production=False)
            e = None
        except T.RunError as x:
            e = x
    g = getattr(e, "guard", {}) or {}
    check("... and still refused by the list when GuardV2 passes everything (defence in depth)",
          e is not None and g.get("reasons", {}).get("train_core_variant_drop") == 1, (e, g.get("reasons")))
    rec_ok = T.guard_rows(rows, dh, production=False)
    check("... while base_v2 passes, the L-8 list recorded in the guard record",
          rec_ok["refused"] == 0 and rec_ok["train_core_variant_drops"]["sha256"] == sha(lst), rec_ok)
    saved = lst.read_bytes()
    lst.write_bytes(saved + b"\n")
    try:
        try:
            T.guard_rows(rows, dh, production=False)
            e = None
        except T.RunError as x:
            e = x
    finally:
        lst.write_bytes(saved)
    check("an L-8 list that does not hash as LOCK v2 records stops the run", e is not None and e.stage == "guard"
          and T.VARIANT_DROPS_NAME in str(e), e)
    l5_path = v2_dir() / "l5_excluded.jsonl"
    l5_path.write_text("")
    lk = json.loads(saved_lock)
    try:
        lock.write_text(json.dumps(dict({k: v for k, v in lk.items() if k != T.VARIANT_DROPS_LOCK_KEY},
                                        testing=False, l5_excluded_sha256=sha(l5_path)), sort_keys=True))
        try:
            T.guard_rows(rows, dh, production=True)
            e = None
        except T.RunError as x:
            e = x
        check("a production LOCK v2 that records the L-5 list but no L-8 list is refused", e is not None
              and "no %s" % T.VARIANT_DROPS_LOCK_KEY in str(e), e)
        lock.write_text(json.dumps(dict({k: v for k, v in lk.items() if k != T.VARIANT_DROPS_LOCK_KEY}),
                                   sort_keys=True))
        rec_t = T.guard_rows(rows, dh, production=False)
        check("... a testing LOCK without one checks no L-8 bytes, and says so", rec_t["refused"] == 0
              and rec_t["train_core_variant_drops"]["sha256"] is None, rec_t.get("train_core_variant_drops"))
    finally:
        lock.write_bytes(saved_lock)
        l5_path.unlink()


def test_end_to_end(W):
    print("end to end on the CPU (testing experiment)")
    base_m = str(v2_dir() / "base_v2.jsonl")
    pb = spec("base__s0", "base", init="yolo11n.pt", train_manifest=base_m, recipe=recipe())
    t0 = time.time()
    rc = run(pb)
    rj = run_json(pb)
    out = pb.parent
    sc = (rj.get("sidecars") or {}).get("dev") or {}
    check("a base run finishes (%.0fs)" % (time.time() - t0), rc == 0 and rj.get("status") == "done",
          (rc, rj.get("stage"), (rj.get("error") or "")[-800:]))
    check("run.json: protocol v3, inc2, splits v2, the arm (n640, assumed), testing deviations recorded",
          rj.get("protocol") == "v3" and rj.get("protocol_package") == "inc2" and rj.get("splits_version") == "v2"
          and (rj.get("arm") or {}).get("id") == "n640" and rj.get("recipe_deviations")
          and rj.get("init_check", {}).get("passed") is True
          and any("continuity arm" in w for w in rj.get("warnings", [])), {k: rj.get(k) for k in (
              "protocol", "arm", "recipe_deviations", "init_check")})
    mods = (rj.get("code") or {}).get("modules") or {}
    check("run.json hashes every tools/inc2 module it ran with", mods.get("tools/inc2/train.py") == sha(
        ROOT / "weed_optimizer_framework" / "tools" / "inc2" / "train.py") and "tools/inc2/guard.py" in mods, sorted(mods))
    g = rj.get("guard") or {}
    check("run.json's guard: v2 index, every image checked, base rows counted as base_copy",
          g.get("splits_version") == "v2" and g.get("checked") == len(W["base_v2"]) and g.get("refused") == 0
          and g.get("reasons", {}).get("base_copy") == len(W["base_v2"]), g)
    ok = (sc.get("path") and pathlib.Path(sc["path"]).is_file() and sha(sc["path"]) == sc.get("sha256")
          and pathlib.Path(sc["images"]["path"]).is_file() and sha(sc["images"]["path"]) == sc["images"]["sha256"])
    side = json.loads(pathlib.Path(sc["path"]).read_text()) if ok else {}
    check("the base run carries a dev sidecar (JSON and npz, hashes recorded) made from its dev score",
          ok and side.get("weights_sha256") == rj.get("weights_sha256")
          and side.get("score", {}).get("sha256") == rj["scores"]["dev"]["sha256"]
          and side["species_se"]["resamples"] == 1000 and side["species_se"]["seed_text"] == "inc2/v3/species_se",
          sc)
    check("... whose per-image arrays reproduce the score's per_class exactly",
          side.get("consistency", {}).get("full_recompute_max_abs_diff", 1) <= 1e-9
          and side.get("consistency", {}).get("max_abs_diff_vs_recorded", 1) <= 1e-9, side.get("consistency"))
    rc2 = run(pb)
    check("a done run is a no-op", rc2 == 0 and run_json(pb).get("attempt") == 1)
    pathlib.Path(sc["path"]).write_text(pathlib.Path(sc["path"]).read_text() + " ")
    rc3 = run(pb)
    rj3 = run_json(pb)
    check("a tampered sidecar: the next attempt re-scores the verified final.pt, it does not retrain",
          rc3 == 0 and rj3.get("attempt") == 2 and rj3.get("resumed_from_attempt") == 1
          and rj3.get("weights_sha256") == rj.get("weights_sha256")
          and sha(rj3["sidecars"]["dev"]["path"]) == rj3["sidecars"]["dev"]["sha256"],
          (rc3, rj3.get("attempt"), rj3.get("resumed_from_attempt"), rj3.get("stage")))

    inc_rows = W["base_v2"] + W["inc"]
    cand_m = write_manifest(EXP, "cand", inc_rows)
    pc = spec("r0__s01_X__cand__s0", "cand", init=str(out / "weights" / "final.pt"), train_manifest=str(cand_m),
              recipe=recipe(seed=0))
    pn = spec("r0__s01_X__null__s0", "null", init=str(out / "weights" / "final.pt"), train_manifest=base_m,
              recipe=recipe(seed=0))
    rcc, rcn = run(pc), run(pn)
    jc, jn = run_json(pc), run_json(pn)
    check("a cand from the base finishes with a sidecar; a null finishes without one",
          rcc == 0 and rcn == 0 and (jc.get("sidecars") or {}).get("dev") and not jn.get("sidecars")
          and not (pn.parent / "scores" / "dev.sidecar.json").exists(), (rcc, rcn, jc.get("stage"), jn.get("stage")))
    check("the cand's guard counts the base rows as base_copy and passes the new rows",
          jc.get("guard", {}).get("reasons", {}).get("base_copy") == len(W["base_v2"])
          and jc["guard"]["checked"] == len(inc_rows), jc.get("guard"))
    pf = spec("final__base__s0", "final", exams=("dev", "imageweeds", "test"), init=str(out / "weights" / "final.pt"))
    rcf = run(pf)
    jf = run_json(pf)
    check("a final run scores dev, imageweeds and test through the locked scorer, without a sidecar",
          rcf == 0 and sorted(jf.get("scores") or {}) == ["dev", "imageweeds", "test"] and not jf.get("sidecars"),
          (rcf, jf.get("stage"), (jf.get("error") or "")[-400:]))
    broken = lambda *a, **k: [sys.executable, "-c", "import sys; print('sidecar broken'); sys.exit(2)"]  # noqa: E731
    pf2 = spec("r0__s02_X__cand__s0", "cand", init=str(out / "weights" / "final.pt"), train_manifest=str(cand_m),
               recipe=recipe(seed=1))
    with patched(T, "sidecar_command", broken):
        rcx = run(pf2)
    jx = run_json(pf2)
    check("a sidecar that fails fails the run at stage sidecar outside a baseline (a chain's v3 decision needs it)",
          rcx == 1 and jx.get("stage") == "sidecar" and jx.get("trained") is True
          and "refused" in (jx.get("error") or ""), (rcx, jx.get("stage")))
    rcy = run(pf2)
    jy = run_json(pf2)
    check("... and the next attempt re-scores the trained weights (sidecar included), it does not retrain",
          rcy == 0 and jy.get("resumed_from_attempt") == 1 and jy["sidecars"]["dev"].get("sha256"),
          (rcy, jy.get("stage"), jy.get("resumed_from_attempt")))
    bexp = "t2_base_exp"
    (C.INC_DIR / bexp).mkdir(parents=True, exist_ok=True)
    (C.INC_DIR / bexp / "exp.json").write_text(json.dumps({"exp": bexp, "type": "baseline", "testing": TESTING}))
    pbx = spec("base__s0", "base", exp=bexp, init="yolo11n.pt", train_manifest=base_m, recipe=recipe())
    with patched(T, "sidecar_command", broken):
        rcz = run(pbx)
    jz = run_json(pbx)
    check("in a baseline experiment a failed sidecar is recorded (status failed, a warning) and the run is done",
          rcz == 0 and jz.get("status") == "done" and jz["sidecars"]["dev"].get("status") == "failed"
          and any("sidecar failed" in w for w in jz.get("warnings", [])), (rcz, jz.get("sidecars")))
    return out / "weights" / "final.pt"


def test_production(W):
    print("production: recipe and init checked before the device")
    base_m = str(v2_dir() / "base_v2.jsonl")
    p1 = spec("p_tiny", "base", exp=EXP_PROD, init="yolo11n.pt", train_manifest=base_m, recipe=recipe())
    rc = run(p1)
    rj = run_json(p1)
    check("a production base run with a tiny recipe is refused at stage recipe",
          rc == 1 and rj.get("stage") == "recipe" and "Protocol v3" in (rj.get("error") or ""), (rj.get("stage"),))
    arm = RC.resolve_arm("n640", repo=C.REPO)
    (C.INC_DIR / EXP_PROD / "exp.json").write_text(json.dumps({"exp": EXP_PROD, "arm": dict(arm, weights_sha256="0" * 64),
                                                             "init_weights": "yolo11n.pt"}))
    p2 = spec("p_init", "base", exp=EXP_PROD, init="yolo11n.pt", train_manifest=base_m,
              recipe=dict(RC.cold("n640"), seed=0))
    rc = run(p2)
    rj = run_json(p2)
    check("a production cold run whose checkpoint is not the arm's pinned sha256 is refused at stage recipe",
          rc == 1 and rj.get("stage") == "recipe" and "arm's checkpoint" in (rj.get("error") or ""),
          (rj.get("stage"), (rj.get("error") or "")[-300:]))
    (C.INC_DIR / EXP_PROD / "exp.json").write_text(json.dumps({"exp": EXP_PROD, "arm": arm, "init_weights": "yolo11n.pt"}))
    p3 = spec("p_ok", "base", exp=EXP_PROD, init="yolo11n.pt", train_manifest=base_m,
              recipe=dict(RC.cold("n640"), seed=0))
    import torch
    rc = run(p3)
    rj = run_json(p3)
    if torch.cuda.is_available():
        print("  NOTE a CUDA device is present; the production device refusal is not exercised")
    else:
        check("the protocol's cold recipe from the arm's checkpoint passes stage recipe and stops at device "
              "(no CUDA here)", rc == 1 and rj.get("stage") == "device" and rj.get("init_check", {}).get("passed")
              and rj.get("recipe_name") == "cold", (rj.get("stage"), rj.get("init_check")))


def test_job_script(spec_path):
    print("run_inc2_job.sh")
    src = ROOT / "run_inc2_job.sh"
    r = subprocess.run(["bash", "-n", str(src)], capture_output=True, text=True)
    check("bash -n passes", r.returncode == 0, r.stderr)
    text = src.read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    check("sbatch header: GPU-shared, v100-32:1, 5 CPUs, 45G, 3h, the INC log dir",
          all(s in text for s in ("--partition=GPU-shared", "--gres=gpu:v100-32:1", "--cpus-per-task=5",
                                  "--mem=45G", "--time=03:00:00", "results/framework/inc/logs/%x_%A_%a.out")))
    check("it runs the v2 executor and the pinned driver, never the v1 executor, and never resets, syncs or "
          "uploads", "weed_optimizer_framework.tools.inc2.train --spec" in code
          and "weed_optimizer_framework.tools.inc.driver advance" in code
          and "tools.inc.train --spec" not in code and "tools.inc.train --flock" not in code
          and not any(s in code.lower() for s in ("git reset", "git pull", "git checkout", "rsync", "cp -r",
                                                  "roboflow", "auto_sync")))
    stub = TMP / "job"
    repo = stub / "repo"
    (stub / "bin").mkdir(parents=True)
    (repo / "weed_llm_benchmark").mkdir(parents=True)
    shutil.copyfile(src, repo / "weed_llm_benchmark" / "run_inc2_job.sh")
    pkg = ROOT / "weed_optimizer_framework"
    for m in ("tools/inc/__init__.py", "tools/inc/common.py", "tools/inc/driver.py", "tools/inc/gate.py",
              "tools/inc/splits.py", "tools/inc/scorer.py", "tools/inc/lora.py", "tools/cwd12_species.py",
              "tools/near_dup.py", "tools/mega_trainer.py", "tools/funnel/__init__.py", "tools/funnel/embed.py",
              "tools/funnel/leak.py"):
        for side in ("weed_optimizer_framework", "weed_llm_benchmark/weed_optimizer_framework"):
            (repo / side / m).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pkg / m, repo / side / m)
    for p in (pkg / "tools" / "inc2").glob("*.py"):
        for side in ("weed_optimizer_framework", "weed_llm_benchmark/weed_optimizer_framework"):
            (repo / side / "tools" / "inc2").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, repo / side / "tools" / "inc2" / p.name)
    (stub / "conda.sh").write_text("conda() { :; }\n")
    calls = stub / "calls.txt"
    py = stub / "bin" / "python"
    py.write_text("#!/bin/bash\n"
                  "if [ \"${1:-}\" = -V ]; then echo 'Python stub'; exit 0; fi\n"
                  "if [ \"${1:-}\" = -c ]; then exec %s \"$@\"; fi\n"
                  "echo \"$* | INC_JOB_SCRIPT=${INC_JOB_SCRIPT:-}\" >> %s\n"
                  "for a in \"$@\"; do\n"
                  "  if [ \"$a\" = --flock-check ]; then exit ${STUB_FLOCK_RC:-1}; fi\n"
                  "  if [ \"$a\" = weed_optimizer_framework.tools.inc.driver ]; then exit 7; fi\n"
                  "done\nexit ${STUB_TRAIN_RC:-0}\n" % (sys.executable, calls))
    py.chmod(0o755)
    lst = stub / "list.txt"
    lst.write_text("%s\n" % spec_path)
    base_env = dict(os.environ, PATH="%s:%s" % (stub / "bin", os.environ["PATH"]), INC2_JOB_REPO=str(repo),
                    INC2_JOB_CONDA_SH=str(stub / "conda.sh"))
    base_env.pop("INC_JOB_SCRIPT", None)
    job = repo / "weed_llm_benchmark" / "run_inc2_job.sh"

    def job_run(*args, **extra):
        if calls.exists():
            calls.unlink()
        e = dict(base_env, SLURM_ARRAY_TASK_ID="0", **extra)
        r = subprocess.run(["bash", str(job), str(lst)] + list(args), env=e, capture_output=True, text=True,
                           timeout=300)
        return r, (calls.read_text().splitlines() if calls.exists() else [])

    me = "INC_JOB_SCRIPT=%s" % job
    train = "-u -m weed_optimizer_framework.tools.inc2.train --spec %s | %s" % (spec_path, me)
    adv = "-u -m weed_optimizer_framework.tools.inc.driver advance --exp %s --quiet | %s" % (EXP, me)
    flock = "-u -m weed_optimizer_framework.tools.inc2.train --flock-check %s | %s" % (C.INC_DIR / EXP, me)
    r1, c1 = job_run(EXP, INC_JOB_ADVANCE="1")
    check("INC_JOB_ADVANCE=1: inc2.train --spec, then inc.driver advance, both with INC_JOB_SCRIPT = the "
          "git-tracked run_inc2_job.sh; exit 0 although the driver failed",
          c1 == [train, adv] and r1.returncode == 0 and "INC_JOB_SCRIPT=%s" % job in r1.stdout,
          (c1, r1.returncode, r1.stderr[-500:]))
    r2, c2 = job_run(EXP, STUB_FLOCK_RC="0")
    r3, c3 = job_run(EXP, STUB_FLOCK_RC="1")
    check("auto: the flock check goes through inc2.train; the driver only when it passes",
          c2 == [train, flock, adv] and c3 == [train, flock] and "not advanced from this node" in r3.stderr,
          (c2, c3))
    r4, c4 = job_run(EXP, INC_JOB_ADVANCE="0", STUB_TRAIN_RC="1")
    check("the job exits with the executor's status; INC_JOB_ADVANCE=0 never advances",
          r4.returncode == 1 and c4 == [train], (r4.returncode, c4))
    r5, c5 = job_run(INC_JOB_ADVANCE="1")
    check("without <exp> the exp is read from the spec", c5 == [train, adv], c5)
    with env(INC_JOB_SCRIPT="/elsewhere/run_inc_job.sh"):
        r6, c6 = job_run(EXP, INC_JOB_ADVANCE="1")
    check("an INC_JOB_SCRIPT inherited from the caller is replaced by this script", c6 == [train, adv], c6)
    outer = repo / "weed_optimizer_framework" / "tools" / "inc2" / "train.py"
    saved = outer.read_bytes()
    outer.write_bytes(saved + b"\n# drift\n")
    try:
        r7, c7 = job_run(EXP, INC_JOB_ADVANCE="1")
        r8, c8 = job_run(EXP, INC_JOB_ADVANCE="1", INC_ALLOW_DRIFT="1")
    finally:
        outer.write_bytes(saved)
    check("an outer inc2 module that differs from the nested copy stops the job before the executor (exit 1); "
          "INC_ALLOW_DRIFT=1 runs it with a warning",
          r7.returncode == 1 and c7 == [] and "DIFFERS from nested" in r7.stdout and "FATAL" in r7.stderr
          and c8 == [train, adv] and "INC_ALLOW_DRIFT=1" in r8.stdout, (r7.returncode, c7, c8, r7.stderr[-300:]))
    outer_f = repo / "weed_optimizer_framework" / "tools" / "funnel" / "embed.py"
    saved_f = outer_f.read_bytes()
    outer_f.write_bytes(saved_f + b"\n# drift\n")
    try:
        r10, c10 = job_run(EXP, INC_JOB_ADVANCE="1")
    finally:
        outer_f.write_bytes(saved_f)
    check("the drift check covers the funnel modules leak.py imports (an outer funnel/embed.py that differs stops "
          "the job)", r10.returncode == 1 and c10 == [] and "tools/funnel/embed.py" in r10.stdout, (r10.returncode, c10))
    (repo / "weed_llm_benchmark" / "run_inc2_job.sh").rename(repo / "weed_llm_benchmark" / "moved.sh")
    try:
        e = dict(base_env, SLURM_ARRAY_TASK_ID="0")
        r9 = subprocess.run(["bash", str(repo / "weed_llm_benchmark" / "moved.sh"), str(lst), EXP], env=e,
                            capture_output=True, text=True, timeout=120)
    finally:
        (repo / "weed_llm_benchmark" / "moved.sh").rename(repo / "weed_llm_benchmark" / "run_inc2_job.sh")
    check("without its git-tracked copy the job stops (exit 2): INC_JOB_SCRIPT could not name it",
          r9.returncode == 2 and "missing" in r9.stderr, (r9.returncode, r9.stderr[-300:]))


def main():
    t0 = time.time()
    try:
        check("the test runs on its own INC_DIR and REPO", C.INC_DIR == TMP / "inc" and C.REPO == TMP / "repo")
        test_pinned_unchanged()
        W = build_world()
        test_recipes()
        test_specs_and_manifests(W)
        test_guard(W)
        test_end_to_end(W)
        test_production(W)
        test_job_script(C.INC_DIR / EXP / "runs" / "base__s0" / "spec.json")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
