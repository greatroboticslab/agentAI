#!/usr/bin/env python3
"""The model-zoo audit's inventory, class maps, converter, provenance,
contamination flags, exams, plan, selection, report, submission and job script
(inc2/zoo.py, run_inc2_zoo.sh; docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-04): Z1, the model-zoo audit (pre-registered usage)").

The synthetic world of tests/test_inc2_train.py (v1 exams, LOCK v1 with the
real scorer's sha256, INC_SCORER_TESTING=1), plus a list root TMP/harry holding
one fixture per selection rule (a INC final.pt with a reusable dev score; b a
legacy 12-class run with best and last; c a yolo26n 100-slot end2end head; d an
sp8 + novel head; e one class 'weed'; f names '0','1'; g COCO names; h an
MLflow byte copy of b's last.pt; i a classifier; j a run whose yaml is gone;
j2 a yolo_iter run (derived superset); k a run whose merge was rebuilt; l an
epoch snapshot; m a LoRA last.pt beside last_merged.pt; m2 a foreign module;
n a Mamba pickle; o INC train weights, an undone run and a final link; p a
removed file; q a stock release; r a third-party file; s a test-fixture MLflow
copy; t a child of b), splits v3 test v1 lists and companions, and a registry
with the maize evaluation slug, rf_test-8qezo (names '0','1') and a listed slug
missing. No GPU; the scoring passes are in test_inc2_zoo_score.py.

Run:  python3 tests/test_inc2_zoo.py
"""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools import cwd12_species as SP  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import base3 as B3  # noqa: E402
from weed_optimizer_framework.tools.inc2 import guard as G  # noqa: E402
from weed_optimizer_framework.tools.inc2 import zoo as Z  # noqa: E402

FAILURES = W.FAILURES
check = W.check
TMP = W.TMP
ROOT = W.ROOT
HARRY = TMP / "harry"
FW = HARRY / "weed_llm_benchmark" / "results" / "framework"
INC_REL = "weed_llm_benchmark/results/framework/inc"
LIST = TMP / "allpt_stat.txt"
CONF = TMP / "zoo_v1.json"
V = "v1"
LEGACY_DATE = "2026-05-01T10:00:00.000000"
NEW_DATE = "2026-09-25T10:00:00.000000"
COCO = ["person"] + ["c%02d" % i for i in range(1, 79)] + ["toothbrush"]
os.environ["INC_ZOO_CONFIG"] = str(CONF)
os.environ["INC_ZOO_IMGSZ"] = "64"
os.environ["INC_ZOO_BATCH"] = "8"
os.environ["INC_ZOO_DEVICE"] = "cpu"
os.environ.pop("INC_ZOO_SOURCE", None)
FIX = {}                       # fixture letter -> rel
WORLD = {}


class ForeignAdapter(nn.Module):
    """A module that is neither torch nor Ultralytics (a LoRA adapter stand-in)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


def sha(p):
    return C.sha256_file(p)


@contextlib.contextmanager
def env(**kw):
    with W.env(**kw):
        yield


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except Z.ZooRefused as e:
        return e
    return None


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        out = fn(*a, **k)
    return out, buf.getvalue()


# ------------------------------------------------------------------- world
def det_model(names, cfg="yolo11n.yaml", seed=0):
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(seed)
    net = DetectionModel(cfg, nc=len(names), verbose=False)
    head = net.model[-1]
    with torch.no_grad():
        for br in [head.cv3] + ([head.one2one_cv3] if getattr(head, "one2one_cv3", None) is not None else []):
            for level in range(head.nl):
                br[level][-1].bias.fill_(-1.0)
    net.names = {i: n for i, n in enumerate(names)}
    return net


def save_ckpt(path, net, date=LEGACY_DATE, train_args=None, half=True):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.half() if half else net, "ema": None, "train_args": dict(train_args or {}),
                "date": date, "version": "8.4.22", "epoch": 3, "best_fitness": 0.5}, str(path))
    return path


def img(path, seed, how=None, src=None):
    """A 96x96 JPEG (W.make_image), or a transformed copy of src (vflip, copy)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if src is None:
        W.make_image(path, seed)
    elif how == "copy":
        shutil.copyfile(src, path)
    elif how == "vflip":
        with Image.open(src) as im:
            im.convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path, quality=90)
    return path


def link(path, target):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    os.symlink(str(target), str(path))
    return path


def write_yaml(path, train, val=None, base=None, mtime=None):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if base:
        lines.append("path: %s" % base)
    lines.append("train: %s" % train)
    if val:
        lines.append("val: %s" % val)
    lines.append("nc: 12\nnames: %s" % json.dumps(list(SP.CWD12_LEGACY_LABELS)))
    path.write_text("\n".join(lines) + "\n")
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


def write_args(run_dir, data, model="yolo11n.pt", mtime=None, **extra):
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = {"task": "detect", "model": model, "data": str(data), "epochs": 10, "patience": 3, "imgsz": 640,
            "batch": 16, "seed": 0}
    args.update(extra)
    p = run_dir / "args.yaml"
    p.write_text("".join("%s: %s\n" % (k, v) for k, v in args.items()))
    if mtime:
        os.utime(p, (mtime, mtime))
    return args


def rel_of(path):
    return os.path.relpath(str(path), str(HARRY))


def build(dev_rows, test_rows, iw_rows):
    """The fixtures, splits v3, the registry and the list; returns WORLD."""
    T0 = 1775001600.0                    # 2026-04-01T00:00:00Z: every fixture's merge and args.yaml predate its date
    # the INC tree is reached from the list root through weed_llm_benchmark/results/framework/inc
    FW.mkdir(parents=True, exist_ok=True)
    link(FW / "inc", C.INC_DIR)
    clean = [img(TMP / "clean" / ("c%02d.jpg" % i), 900 + i) for i in range(4)]
    # (a) an INC final.pt: 13 INC names, a done run.json, a 4-row manifest with one dev image, a test dev score
    run_a = C.INC_DIR / "zt_inc" / "runs" / "base__s0"
    fa = save_ckpt(run_a / "weights" / "final.pt", det_model(C.CLASS_NAMES, seed=1), date=NEW_DATE)
    man_rows = [dev_rows[0]] + [W.row_for("zc_%d" % i, clean[i], [(12, 0.5, 0.5, 0.2, 0.2)], "clean") for i in range(3)]
    man = C.INC_DIR / "zt_inc" / "manifests" / "base.jsonl"
    msha = C.write_manifest(man, man_rows)
    (C.INC_DIR / "zt_inc" / "exp.json").write_text(json.dumps({"exp": "zt_inc", "type": "baseline"}))
    run_a.mkdir(parents=True, exist_ok=True)
    rj = {"status": "done", "testing": False, "kind": "base", "spec": {"kind": "base", "init": str(C.REPO / "yolo11n.pt"),
          "train_manifest": str(man), "recipe": {"imgsz": 640, "epochs": 1}},
          "init": str(C.REPO / "yolo11n.pt"), "init_sha256": sha(C.REPO / "yolo11n.pt"), "train_manifest": str(man),
          "train_manifest_sha256": msha, "weights": str(fa), "weights_sha256": sha(fa),
          "started_utc": "2026-09-28T10:00:00Z", "finished_utc": "2026-09-28T11:00:00Z",
          "code": {"modules": {"tools/inc/scorer.py": sha(pathlib.Path(S.__file__).resolve())}},
          "train_class_counts": {s: 3 for s in SP.CWD12_SPECIES}, "ultralytics_version": "8.4.22"}
    (run_a / "run.json").write_text(json.dumps(rj))
    (run_a / "scores").mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        S.score(fa, "dev", run_a / "scores" / "dev.json", imgsz=64, batch=8, device="cpu")
    FIX["a"] = "%s/zt_inc/runs/base__s0/weights/final.pt" % INC_REL
    # (o) INC train weights of an undone run, its glob-found final.pt, and a kind-final link with a test score
    run_o = C.INC_DIR / "zt_inc" / "runs" / "cand__s1"
    save_ckpt(run_o / "train" / "weights" / "best.pt", det_model(C.CLASS_NAMES, seed=2), date=NEW_DATE)
    save_ckpt(run_o / "weights" / "final.pt", det_model(C.CLASS_NAMES, seed=3), date=NEW_DATE)
    FIX["o_train"] = "%s/zt_inc/runs/cand__s1/train/weights/best.pt" % INC_REL
    run_f = C.INC_DIR / "zt_inc" / "runs" / "final__base__s0"
    link(run_f / "weights" / "final.pt", fa)
    (run_f / "run.json").write_text(json.dumps({"status": "done", "testing": False, "kind": "final",
                                                 "spec": {"kind": "final", "init": str(fa)}, "init": str(fa),
                                                 "weights_sha256": sha(fa)}))
    (run_f / "scores").mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        S.score(fa, "test", run_f / "scores" / "test.json", imgsz=64, batch=8, device="cpu")
    # cwd12 roots, a train image of a dev session that is not a dev image, the leave4out supersets
    cwd_train = HARRY / "weed_llm_benchmark" / "downloads" / "cottonweeddet12" / "train" / "images"
    sess_img = img(cwd_train / "dv_950.jpg", 950)
    l4 = HARRY / "weed_llm_benchmark" / "results" / "leave4out"
    l4a = img(l4 / "dataset_8species" / "train" / "images" / "l4a.jpg", 960)
    img(l4 / "dataset_holdout" / "train" / "images" / "l4b.jpg", 961)
    # test v1 rows and companions (splits/v3)
    v3 = C.INC_DIR / "splits" / "v3"
    tv1 = []
    for i in range(6):
        ip = img(v3 / "images" / ("tv%02d.jpg" % i), 700 + i)
        lp = v3 / "labels" / ("tv%02d.txt" % i)
        C.write_yolo(lp, [(12, 0.4, 0.5, 0.3, 0.3)])
        h, var = G.image_hashes(ip)
        tv1.append({"key": "src%d__tv%02d" % (i % 2, i), "source": "src%d" % (i % 2), "kind": "registry",
                    "image": str(ip), "sha256": sha(ip), "label": str(lp), "label_sha256": sha(lp),
                    "original_image": str(ip), "original_sha256": sha(ip), "dhash": int(h),
                    "variants": [int(var[k]) for k in G.VARIANTS], "group": "g%d" % (i // 2), "holdout_v1": True})
    (v3 / "test_v1").mkdir(parents=True, exist_ok=True)
    for s in ("src0", "src1"):
        (v3 / "test_v1" / ("%s.jsonl" % s)).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in tv1
                                                              if r["source"] == s))
    comp_img = img(TMP / "comp" / "comp0.jpg", 760)
    h, var = G.image_hashes(comp_img)
    (v3 / "test_v1_companions").mkdir(parents=True, exist_ok=True)
    (v3 / "test_v1_companions" / "src0.jsonl").write_text(json.dumps({
        "key": "src0__comp0", "source": "src0", "kind": "registry", "companion": True, "reason": "dedupe",
        "original_image": str(comp_img), "original_sha256": sha(comp_img), "sha256": None, "dhash": int(h),
        "variants": [int(var[k]) for k in G.VARIANTS], "group": "g0"}, sort_keys=True) + "\n")
    (v3 / "summary.json").write_text(json.dumps({"format": B3.FORMAT, "status": "complete", "arms": {}}))
    # the registry: the maize evaluation slug, rf_test-8qezo ('0', '1') and nothing else
    maize = TMP / "reg" / "maize"
    mrows = []
    for i in range(8):
        name = "mz%02d_jpg.rf.%s" % (i, hashlib.sha1(str(i).encode()).hexdigest()[:20])
        ip = img(maize / "train" / "images" / ("%s.jpg" % name), 800 + i) if i != 7 else \
            img(maize / "train" / "images" / ("%s.jpg" % name), 0, "copy", dev_rows[1]["image"])
        lp = maize / "train" / "labels" / ("%s.txt" % name)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("0 0.3 0.3 0.2 0.2\n" if i == 6 else "1 0.5 0.5 0.3 0.3\n0 0.2 0.2 0.1 0.1\n")
        mrows.append(ip)
    (maize / "data.yaml").write_text("names: ['maize', 'weed']\nnc: 2\n")
    rft = TMP / "reg" / "rftest"
    img(rft / "train" / "images" / "r0.jpg", 850)
    (rft / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (rft / "train" / "labels" / "r0.txt").write_text("1 0.5 0.5 0.3 0.3\n")
    (rft / "data.yaml").write_text("names: ['0', '1']\nnc: 2\n")
    reg = {"datasets": {"project_agml__maize_weed_detection": {"local_path": str(maize), "status": "ok"},
                        "rf_test-8qezo__weed-detection-ycai2": {"local_path": str(rft), "status": "ok"}}}
    rp = C.REPO / "results" / "framework" / "dataset_registry.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(reg))
    # (b) a legacy 12-class run: best and last; its merged train dir (symlinks) and a holdout val
    merged = FW / "merged_b"
    tr = merged / "train" / "images"
    link(tr / "d0.jpg", dev_rows[0]["image"])
    link(tr / "d0_dup.jpg", dev_rows[0]["image"])
    link(tr / "d1.jpg", dev_rows[2]["image"])
    vf = img(TMP / "copies" / "test_vflip.jpg", 0, "vflip", test_rows[0]["image"])
    link(tr / "tvf.jpg", vf)
    tvc = img(TMP / "copies" / "tv_copy.jpg", 0, "copy", tv1[0]["image"])
    link(tr / "tv1c.jpg", tvc)
    link(tr / "mz.jpg", mrows[0])
    val = HARRY / "weed_llm_benchmark" / "cwd12_holdout" / "images"
    img(val / "v0.jpg", 970)
    yb = write_yaml(merged / "data.yaml", "train/images", str(val), base=str(merged), mtime=T0)
    for e in tr.iterdir():
        os.utime(e, (T0, T0), follow_symlinks=False)
    rb = FW / "legacy12" / "train"
    tab = write_args(rb, yb, mtime=T0 + 3600)
    fb = save_ckpt(rb / "weights" / "best.pt", det_model(SP.CWD12_LEGACY_LABELS, seed=4), train_args=tab)
    fbl = save_ckpt(rb / "weights" / "last.pt", det_model(SP.CWD12_LEGACY_LABELS, seed=5), train_args=tab)
    (rb / "results.csv").write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n" +
                                    "".join("%d,0.5,%.2f\n" % (i + 1, 0.3 + 0.01 * i) for i in range(10)))
    FIX["b_best"], FIX["b_last"] = rel_of(fb), rel_of(fbl)
    # (l) an epoch snapshot beside them (never opened: not even a checkpoint)
    (rb / "weights" / "epoch5.pt").write_bytes(b"not a checkpoint")
    FIX["l"] = rel_of(rb / "weights" / "epoch5.pt")
    # (h) an MLflow byte copy of b's last.pt; (s) a test fixture copy
    mf = HARRY / "weed_llm_benchmark" / "runs" / "mlflow" / "1" / "abc"
    (mf / "artifacts" / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fbl, mf / "artifacts" / "weights" / "last.pt")
    (mf / "params").mkdir(parents=True, exist_ok=True)
    (mf / "params" / "project").write_text(str(FW / "legacy12"))
    (mf / "params" / "name").write_text("train")
    (mf / "meta.yaml").write_text("run_name: train\nstart_time: %d\n" % int(T0 * 1000))
    FIX["h"] = rel_of(mf / "artifacts" / "weights" / "last.pt")
    ms = HARRY / "weed_llm_benchmark" / "runs" / "mlflow" / "2" / "def"
    save_ckpt(ms / "artifacts" / "weights" / "best.pt", det_model(["weed"], seed=6), date=NEW_DATE)
    (ms / "params").mkdir(parents=True, exist_ok=True)
    (ms / "params" / "project").write_text(str(TMP / "fixtures" / "test_x"))
    (ms / "params" / "name").write_text("train")
    FIX["s"] = rel_of(ms / "artifacts" / "weights" / "best.pt")
    # (c) a yolo26n end2end head of 100 classes: 12 trainer slots + aux
    names_c = list(SP.TRAINER_SLOT_LEGACY) + ["aux_%d" % i for i in range(12, 100)]
    FIX["c"] = rel_of(save_ckpt(FW / "slots100" / "train" / "weights" / "best.pt", det_model(names_c, "yolo26n.yaml", 7)))
    # (d) sp8 + a novel class; (e) one class 'weed'; (f) numeric names; (g) COCO names
    FIX["d"] = rel_of(save_ckpt(FW / "sp8" / "weights" / "best.pt",
                                det_model(list(SP.TRAINER_SLOT_LEGACY[:8]) + ["novel_weed"], seed=8)))
    FIX["e"] = rel_of(save_ckpt(FW / "oneclass" / "weights" / "best.pt", det_model(["weed"], seed=9), date=NEW_DATE))
    shutil.copyfile(FW / "oneclass" / "weights" / "best.pt", FW / "oneclass" / "weights" / "last.pt")
    FIX["e_last"] = rel_of(FW / "oneclass" / "weights" / "last.pt")
    # an MLflow copy whose source run file is gone: kept, with its project's family (probe)
    mo = HARRY / "weed_llm_benchmark" / "runs" / "mlflow" / "3" / "ghi"
    save_ckpt(mo / "artifacts" / "weights" / "last.pt", det_model(["weed"], seed=22), date=NEW_DATE)
    (mo / "params").mkdir(parents=True, exist_ok=True)
    (mo / "params" / "project").write_text(str(FW / "mega_itersmoke_timecap"))
    (mo / "params" / "name").write_text("train")
    (mo / "meta.yaml").write_text("start_time: %d\n" % int(T0 * 1000))
    FIX["orphan"] = rel_of(mo / "artifacts" / "weights" / "last.pt")
    FIX["f"] = rel_of(save_ckpt(FW / "numnames" / "weights" / "best.pt", det_model(["0", "1"], seed=10)))
    FIX["g"] = rel_of(save_ckpt(FW / "coco" / "weights" / "best.pt", det_model(COCO, seed=11)))
    # (i) a classifier
    from ultralytics.nn.tasks import ClassificationModel
    cls = ClassificationModel("yolo11n-cls.yaml", nc=3, verbose=False)
    cls.names = {0: "a", 1: "b", 2: "c"}
    FIX["i"] = rel_of(save_ckpt(HARRY / "weed_llm_benchmark" / "runs" / "classify" / "c1" / "weights" / "best.pt", cls))
    # (j) a run whose data yaml is gone; (j2) a yolo_iter run with no yaml (the derived superset)
    rj_ = FW / "noyaml" / "train"
    taj = write_args(rj_, TMP / "gone" / "data.yaml", mtime=T0)
    FIX["j"] = rel_of(save_ckpt(rj_ / "weights" / "best.pt", det_model(["weed"], seed=12), date=NEW_DATE,
                                train_args=taj))
    rj2 = FW / "yolo_iter3"
    taj2 = write_args(rj2, TMP / "gone" / "iter3.yaml", mtime=T0)
    FIX["j2"] = rel_of(save_ckpt(rj2 / "weights" / "best.pt", det_model(["weed"], seed=13), date=NEW_DATE,
                                 train_args=taj2))
    # (k) a merge rebuilt after the run started: its list holds a dev image
    mk = FW / "merged_k"
    link(mk / "train" / "images" / "kd.jpg", dev_rows[3]["image"])
    link(mk / "train" / "images" / "kc.jpg", comp_img)
    yk = write_yaml(mk / "data.yaml", "train/images", base=str(mk), mtime=T0 + 7200)
    rk = FW / "rebuilt" / "train"
    tak = write_args(rk, yk, mtime=T0 + 3600)
    FIX["k"] = rel_of(save_ckpt(rk / "weights" / "best.pt", det_model(["weed"], seed=14), date=NEW_DATE,
                                train_args=tak))
    # (m) a LoRA run: last.pt with a foreign module beside its merged copy; (m2) a foreign module alone
    lm = det_model(["weed"], seed=15)
    lm.model[0] = nn.Sequential(lm.model[0], ForeignAdapter())
    rl = FW / "lora" / "weights"
    save_ckpt(rl / "last.pt", lm, date=NEW_DATE)
    FIX["m_last"] = rel_of(rl / "last.pt")
    FIX["m_merged"] = rel_of(save_ckpt(rl / "last_merged.pt", det_model(["weed"], seed=16), date=NEW_DATE))
    fm = det_model(["weed"], seed=17)
    fm.model[0] = nn.Sequential(fm.model[0], ForeignAdapter())
    FIX["m2"] = rel_of(save_ckpt(FW / "foreign" / "weights" / "best.pt", fm, date=NEW_DATE))
    # (n) a pickle naming ultralytics.nn.modules.mamba_yolo.X (absent from this Ultralytics)
    import types
    mod = types.ModuleType("ultralytics.nn.modules.mamba_yolo")

    class X(nn.Module):
        pass
    X.__module__ = mod.__name__
    X.__qualname__ = "X"
    mod.X = X
    sys.modules[mod.__name__] = mod
    try:
        np_ = FW / "mamba" / "weights" / "best.pt"
        np_.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": X(), "train_args": {}}, str(np_))
    finally:
        del sys.modules[mod.__name__]
    FIX["n"] = rel_of(np_)
    # (p) listed, then removed; (q) a stock release with COCO names; (r) a third-party file
    gone = FW / "removed" / "weights" / "best.pt"
    save_ckpt(gone, det_model(["weed"], seed=18))
    FIX["p"] = rel_of(gone)
    FIX["q"] = rel_of(save_ckpt(HARRY / "weed_llm_benchmark" / "yolo11n.pt", det_model(COCO, seed=19)))
    FIX["r"] = rel_of(save_ckpt(HARRY / "weed_llm_benchmark" / "datasets" / "gh_x" / "models" / "m.pt",
                                det_model(["weed"], seed=20)))
    # (t) a child of b's best.pt on a clean list
    mt = FW / "merged_t"
    link(mt / "train" / "images" / "c3.jpg", clean[3])
    link(mt / "train" / "images" / "sess.jpg", sess_img)
    yt = write_yaml(mt / "data.yaml", "train/images", base=str(mt), mtime=T0)
    for e in (mt / "train" / "images").iterdir():
        os.utime(e, (T0, T0), follow_symlinks=False)
    rt = FW / "child" / "train"
    tat = write_args(rt, yt, model=str(fb), mtime=T0 + 3600)
    FIX["t"] = rel_of(save_ckpt(rt / "weights" / "best.pt", det_model(["weed"], seed=21), date=NEW_DATE,
                                train_args=tat))
    # d, e and f train on the clean list too (their args.yaml beside the weights dir): dev-clean rows
    for x in ("sp8", "oneclass", "numnames"):
        write_args(FW / x, yt, mtime=T0 + 3600)
    # (u) a clean list, an init the zoo cannot resolve: U through inheritance
    ru = FW / "unknowninit" / "train"
    tau = write_args(ru, yt, model="/nowhere/parent.pt", mtime=T0 + 3600)
    FIX["u"] = rel_of(save_ckpt(ru / "weights" / "best.pt", det_model(["weed"], seed=23), date=NEW_DATE,
                                train_args=tau))
    # (w) e's weights saved again with other metadata: another sha256, the same weights digest
    ck = torch.load(str(FW / "oneclass" / "weights" / "best.pt"), map_location="cpu", weights_only=False)
    ck["date"] = "2026-09-30T00:00:00"
    wp = FW / "oneclass_resaved_copy" / "weights" / "best.pt"
    wp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(wp))
    FIX["w"] = rel_of(wp)
    # the list: `size mtime ./rel` of every fixture (lstat, as find prints it), then (p) removed
    lines = []
    for p in sorted(HARRY.rglob("*.pt")):
        if rel_of(p).startswith(INC_REL + "/"):
            continue
        st = os.lstat(p)
        lines.append("%d %d ./%s" % (st.st_size, int(st.st_mtime), rel_of(p)))
    for p in sorted(list(C.INC_DIR.glob("zt_inc/runs/*/weights/final.pt")) +
                    list(C.INC_DIR.glob("zt_inc/runs/*/train/weights/*.pt"))):
        rel = "%s/%s" % (INC_REL, p.relative_to(C.INC_DIR).as_posix())
        if "cand__s1/weights" in rel:
            continue                                      # found by the INC glob only
        st = os.lstat(p)
        lines.append("%d %d ./%s" % (st.st_size, int(st.st_mtime), rel))
    lines.append("12 1700000000 ./weed_llm_benchmark/results/framework/notes.txt")
    LIST.write_text("\n" + "\n".join(lines) + "\n")
    gone.unlink()
    conf = json.loads((ROOT / "weed_optimizer_framework" / "tools" / "inc2" / "zoo_v1.json").read_text())
    conf["inventory"].update(list=str(LIST), list_sha256=sha(LIST), list_root=str(HARRY),
                             fixture_prefixes=[str(TMP / "fixtures")])
    conf["contamination"]["hash_budget_s"] = 600
    CONF.write_text(json.dumps(conf, indent=1))
    WORLD.update(dev=dev_rows, test=test_rows, iw=iw_rows, tv1=tv1, maize=mrows, sess_img=sess_img,
                 vflip=vf, tv_copy=tvc, clean=clean, l4a=l4a, T0=T0)
    return WORLD


def world():
    if not WORLD:
        Wd = W.build_world()
        build(Wd["dev"], Wd["test"], Wd["imageweeds"])
    return WORLD


def conf():
    return Z.load_config(V)


def files_by_rel():
    return {f["rel"]: f for f in Z.read_files(V)}


def models_by_rel():
    return {m["rel"]: m for m in Z.read_models(V)}


def run_steps(*steps):
    c, s = conf()
    for st in steps:
        quiet(Z.run_step, V, c, s, st, None)


# ------------------------------------------------------------------- tests
def test_list():
    print("the inventory: one decision per listed file, by the pre-registered rule, reconciled")
    run_steps("list")
    fs = files_by_rel()
    want = {"a": ("keep", None), "b_best": ("keep", None), "b_last": ("keep", None),
            "h": ("skip", "sha256_duplicate"), "s": ("skip", "test_fixture"), "l": ("skip", "epoch_snapshot"),
            "m_last": ("skip", "lora_unmerged"), "p": ("skip", "gone"), "r": ("skip", "third_party"),
            "o_train": ("skip", "inc_train_weights"), "e_last": ("skip", "sha256_duplicate"),
            "orphan": ("keep", None), "c": ("keep", None), "q": ("keep", None), "n": ("keep", None)}
    got = {k: (fs.get(FIX[k], {}).get("decision"), fs.get(FIX[k], {}).get("reason")) for k in want}
    check("each fixture's list decision and reason (b best/last kept, h a duplicate of b's last, s a test "
          "fixture, l an epoch snapshot, m's last.pt beside last_merged, p gone, r third party, o's train weights)",
          got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]})
    check("h is a duplicate of b's last.pt; e's last.pt of its best.pt, which carries also_role last",
          fs[FIX["h"]]["dup_of"] == FIX["b_last"] and fs[FIX["e_last"]]["dup_of"] == FIX["e"]
          and fs[FIX["e"]]["also_role"] == ["last"], (fs[FIX["h"]]["dup_of"], fs[FIX["e"]]["also_role"]))
    glob = [f for f in fs.values() if f["origin"] == "inc_glob"]
    link_ = [f for f in fs.values() if "final__base__s0" in f["rel"]]
    check("the INC glob adds the undone run's final.pt (inc_not_done); the kind-final symlink is inc_final_link",
          [(f["reason"], "cand__s1" in f["rel"]) for f in glob] == [("inc_not_done", True)]
          and [f["reason"] for f in link_] == ["inc_final_link"], ([f["rel"] for f in glob], link_))
    check("the MLflow orphan (its source run file gone) is kept with its project's family (probe)",
          fs[FIX["orphan"]]["family"] == "probe" and fs[FIX["orphan"]]["decision"] == "keep", fs[FIX["orphan"]])
    check("a line that is not a .pt is not_checkpoint; epoch files are never hashed",
          fs["weed_llm_benchmark/results/framework/notes.txt"]["reason"] == "not_checkpoint"
          and fs[FIX["l"]]["sha256"] is None, fs[FIX["l"]])
    summ = json.loads((Z.zoo_dir(V) / "inventory" / "summary.json").read_text())
    n_lines = len([ln for ln in LIST.read_text().splitlines() if ln.strip()])
    cnt = summ["counts"]
    check("the counts reconcile: %d listed + %d glob hits = %d decisions; every reason has a count (zeros shown)"
          % (n_lines, summ["inc_glob_added"], cnt["reconciled"]["decisions"]),
          summ["files_listed"] == n_lines and summ["blank_lines"] == 1 and cnt["reconciled"]["decisions"] ==
          n_lines + summ["inc_glob_added"] == len(fs) and set(cnt["skipped_by_reason"]) == set(Z.SKIP_REASONS)
          and sum(cnt["by_decision"].values()) == len(fs), summ)
    c, csha = conf()
    bad = json.loads(json.dumps(c))
    bad["inventory"]["list_sha256"] = "0" * 64
    e = refused(quiet, Z.list_step, V, bad, csha)
    check("a list with another sha256 refuses", e is not None and "pinned" in str(e), e)
    rows = Z.read_files(V)
    e2 = refused(Z.reconcile, rows[:-1], summ["files_listed"], summ["inc_glob_added"])
    rows2 = json.loads(json.dumps(rows))
    rows2[0]["decision"] = None
    e3 = refused(Z.reconcile, rows2, summ["files_listed"], summ["inc_glob_added"])
    check("a dropped file or a row without a decision refuses the reconciliation", e2 is not None and e3 is not None,
          (e2, e3))


def test_class_maps():
    print("class maps: R0-R5 over whole names lists")
    c, _s = conf()
    cm = lambda n, legacy=True: Z.class_map(n, c, legacy)  # noqa: E731
    r0, r1, r1s = cm(C.CLASS_NAMES), cm(SP.CWD12_LEGACY_LABELS), cm(SP.CWD12_SPECIES)
    r2 = cm(list(SP.TRAINER_SLOT_LEGACY) + ["aux_%d" % i for i in range(12, 100)])
    r2s = cm(list(SP.TRAINER_SLOT_SPECIES) + ["aux_12"])
    r3 = cm(list(SP.TRAINER_SLOT_LEGACY[:8]) + ["novel_weed"])
    r4, r4c, r5 = cm(["weed"], False), cm(["crop", "weed"], False), cm(["0", "1"])
    check("R0 identity; R1 (legacy and species lists) id i -> INC i, OtherPlant empty",
          r0["rule"] == "R0" and r0["groups"] == [[i] for i in range(13)] and r1["rule"] == r1s["rule"] == "R1"
          and r1["groups"][:12] == [[i] for i in range(12)] and r1["groups"][12] == [], (r0, r1))
    ok = all(r2["groups"][SP.CWD12_SPECIES.index(SP.TRAINER_SLOT_SPECIES[j])] == [j] for j in range(12))
    check("R2 (legacy or species trainer slots + aux): slot j -> its species' INC id, every aux -> 12",
          r2["rule"] == r2s["rule"] == "R2" and ok and r2["groups"][12] == list(range(12, 100)), r2["groups"])
    check("R3: eight trainer slots + a novel name: 8 species channels, the novel one -> 12",
          r3["rule"] == "R3" and r3["n_species_channels"] == 8 and r3["groups"][12] == [8], r3)
    check("R4: ['weed'] one channel -> 12 (no species); ['crop', 'weed'] crop dropped",
          r4["rule"] == "R4" and r4["groups"][12] == [0] and r4["n_species_channels"] == 0
          and r4c["rule"] == "R4" and r4c["dropped"] == [0] and r4c["groups"][12] == [1], (r4, r4c))
    part = cm(["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"], False)
    check("R5 for ['0','1'] (every channel -> 12) and for a partial legacy vocabulary (never Ragweed by species_of)",
          r5["rule"] == "R5" and r5["groups"][12] == [0, 1] and part["rule"] == "R5"
          and "partial legacy" in part["reason"] and not any(part["groups"][:12]), (r5, part))
    rag_old, rag_new = cm(["Ragweed"], True), cm(["Ragweed"], False)
    check("a checkpoint dated before 2026-09-21 naming 'Ragweed' (Sicklepod in the legacy vocabulary) is R5, never "
          "Ragweed; from 2026-09-21 it is R4 Ragweed",
          rag_old["rule"] == "R5" and not rag_old["groups"][5] and rag_new["rule"] == "R4"
          and rag_new["groups"][5] == [0], (rag_old, rag_new))
    check("is_legacy_date: before 2026-09-21 or unknown", Z.is_legacy_date("2026-08-01T00:00:00")
          and Z.is_legacy_date(None) and not Z.is_legacy_date("2026-09-21T00:00:00"))
    check("COCO detection: 80 names, person first, toothbrush last", Z.is_coco(COCO) and not Z.is_coco(COCO[:79]))


def test_meta():
    print("meta: what each kept checkpoint is (task, head, names, class map), reconciled with files.jsonl")
    run_steps("meta")
    meta = {r["rel"]: r for r in json.loads("[" + ",".join(
        (Z.zoo_dir(V) / "inventory" / "meta.jsonl").read_text().split("\n")[:-1]) + "]")}
    want = {"g": ("skip", "stock_coco"), "q": ("skip", "stock_weights"), "i": ("skip", "classifier"),
            "n": ("unscorable", "unscorable_load"), "m2": ("unscorable", "unscorable_foreign")}
    got = {k: (meta[FIX[k]]["decision"], meta[FIX[k]]["reason"]) for k in want}
    check("g COCO names -> stock_coco, q a stock release -> stock_weights, i classifier, n the Mamba pickle -> "
          "unscorable_load, m2 a foreign module -> unscorable_foreign", got == want, got)
    ms = models_by_rel()
    rules = {k: ms[FIX[k]]["class_map"]["rule"] for k in ("a", "b_best", "c", "d", "e", "f")}
    check("rules: a R0, b R1, c R2, d R3, e R4, f R5", rules == {"a": "R0", "b_best": "R1", "c": "R2", "d": "R3",
                                                                 "e": "R4", "f": "R5"}, rules)
    check("c is an end2end head of 100 classes; b's best and last differ (two rows); h is not a row",
          ms[FIX["c"]]["head"]["end2end"] and ms[FIX["c"]]["head"]["nc"] == 100 and FIX["h"] not in ms
          and ms[FIX["b_best"]]["model_id"] != ms[FIX["b_last"]]["model_id"], ms[FIX["c"]]["head"])
    ms_sum = json.loads((Z.zoo_dir(V) / "inventory" / "meta_summary.json").read_text())
    cnt = ms_sum["counts"]
    check("meta's skips and unscorables reconcile over files.jsonl joined with meta.jsonl",
          cnt["skipped_by_reason"]["stock_coco"] == 1 and cnt["skipped_by_reason"]["classifier"] == 1
          and cnt["unscorable_by_reason"]["unscorable_load"] == 1 and cnt["reconciled"]["decisions"]
          == sum(cnt["by_decision"].values()), cnt)
    check("the epoch snapshot was never opened (no meta row)", FIX["l"] not in meta)
    check("(w) holds e's weights under other bytes: one weights digest, w same_weights_as e, scored once (as e)",
          ms[FIX["w"]]["model_id"] != ms[FIX["e"]]["model_id"] and ms[FIX["w"]]["weights_digest"] ==
          ms[FIX["e"]]["weights_digest"] and ms[FIX["w"]].get("same_weights_as") == ms[FIX["e"]]["model_id"]
          and not ms[FIX["e"]].get("same_weights_as"), (ms[FIX["w"]].get("same_weights_as"),))



def test_convert():
    print("convert: heads outside the INC class space rewritten into 13 channels, checked on the reloaded files")
    run_steps("convert")
    ms = models_by_rel()
    recs = {k: Z.conversion_of(V, ms[FIX[k]]["model_id"]) for k in ("b_best", "c", "d", "e", "f")}
    check("b, c, d, e and f are converted and pass the fidelity check (boxes equal, scores within 1e-4, empty "
          "channels near 0) and the scorer's model check",
          all(r and r["ok"] and r["fidelity"]["boxes_equal"] and r["fidelity"]["max_score_diff"] <= 1e-4
              and r["fidelity"]["empty_max_score"] <= 1e-4 and r["fidelity"]["load_check"] == "ok"
              for r in recs.values()), {k: (r or {}).get("fidelity") for k, r in recs.items()})
    cpath = Z.zoo_dir(V) / recs["c"]["converted"]
    yc = S.load_model(cpath)
    head = yc.model.model[-1]
    names = [yc.names[i] for i in sorted(yc.names)]
    check("the converted file reloads (fp16, YOLO) with the INC names, task detect, nc 13, no 13 + 4 reg_max",
          names == C.CLASS_NAMES and yc.task == "detect" and head.nc == 13 and head.no == 13 + 4 * head.reg_max,
          (names[:3], yc.task, head.nc, head.no))
    o2o = getattr(head, "one2one_cv3", None)
    check("c (end2end): one2one_cv3 converted as well (a max-merge of 88 aux slots)",
          o2o is not None and isinstance(o2o[0][-1], nn.Sequential) and o2o[0][-1][0].out_channels == 13 * 88,
          o2o[0][-1] if o2o is not None else None)
    e1 = refused(S._check_model, S.load_model(Z.zoo_dir(V) / recs["b_best"]["converted"]), "x")
    e2 = None
    try:
        S._check_model(S.load_model(HARRY / FIX["b_best"]), "src")
    except S.ScorerRefused as e:
        e2 = e
    check("the scorer accepts the converted file and refuses the source", e1 is None and e2 is not None, (e1, e2))
    before = {k: r["converted_sha256"] for k, r in recs.items()}
    run = Z.convert_step
    c, csha = conf()
    quiet(run, V, c, csha)
    after = {k: Z.conversion_of(V, ms[FIX[k]]["model_id"])["converted_sha256"] for k in recs}
    check("written once: a second convert keeps every file and record", before == after)
    # a planted wrong weight row in the converted head fails the fidelity check
    import copy
    ck = torch.load(str(Z.zoo_dir(V) / recs["e"]["converted"]), map_location="cpu", weights_only=False)
    bad = copy.deepcopy(ck)
    hd = bad["model"].model[-1]
    with torch.no_grad():
        last = hd.cv3[0][-1]
        conv = last[0] if isinstance(last, nn.Sequential) else last
        conv.weight[12].mul_(-3.0)
        conv.bias[12] += 2.0
    bp = TMP / "bad_conv.pt"
    torch.save(bad, str(bp))
    x = Z.fidelity_batch(4, 64)
    fid = Z.fidelity_check(HARRY / FIX["e"], bp, ms[FIX["e"]]["class_map"]["groups"], x, c, "cpu")
    check("a planted wrong weight row in the converted head fails the fidelity check (unscorable_fidelity)",
          not fid["ok"] and fid["max_score_diff"] > 1e-4, fid)
    check("the scorable rows: a, b, c, d, e, f, k, j, j2, t, the orphan and the merged LoRA file; not n, m2, nor w "
          "(scored once, as e)",
          all(Z.scorable(ms[FIX[k]], V) for k in ("a", "b_best", "b_last", "c", "d", "e", "f", "k", "j", "j2", "t",
                                                    "orphan", "m_merged"))
          and not Z.scorable(ms[FIX["n"]], V) and not Z.scorable(ms[FIX["m2"]], V) and not Z.scorable(ms[FIX["w"]], V))


def test_provenance():
    print("provenance: dates, recipe, init, the training list and its rating, selection")
    run_steps("provenance")
    pr = {r["rel"]: r for r in Z._read_jsonl(Z.zoo_dir(V) / "provenance.jsonl")}
    ms = models_by_rel()
    a, b, k, j, j2, t = (pr[FIX[x]] for x in ("a", "b_best", "k", "j", "j2", "t"))
    check("(a) INC: exact (the manifest hashes as run.json records), 4 images, init stock yolo11n.pt, code exact",
          a["data"]["rating"] == "exact" and a["data"]["n_unique"] == 4 and a["init"]["kind"] == "stock"
          and a["code"]["kind"] == "exact" and a["selected_on"] == "train_subset", (a["data"], a["init"]))
    check("(b) listed_exact with n_unique 5 (an oversampled symlink counted once; 6 entries); selected on the "
          "cwd12 test (its val is a cwd12_holdout copy): best.pt test_selected best",
          b["data"]["rating"] == "listed_exact" and b["data"]["n_unique"] == 5 and b["data"]["n_entries"] == 6
          and b["selected_on"] == "cwd12_test" and b["test_selected"] == "best", (b["data"], b["selected_on"]))
    check("(k) listed_after_rebuild (the data yaml is newer than the run); (j) none (yaml gone); (j2) the derived "
          "superset of yolo_iter, selected on cwd12_test_part",
          k["data"]["rating"] == "listed_after_rebuild" and j["data"]["rating"] == "none"
          and j2["data"]["rating"] == "derived_superset" and j2["selected_on"] == "cwd12_test_part",
          (k["data"], j["data"], j2["data"]))
    T0 = WORLD["T0"]
    check("date_start is args.yaml's mtime, date_end the checkpoint's date",
          b["dates"]["start_utc"] == Z._utc(T0 + 3600) and b["dates"]["end"] == LEGACY_DATE, b["dates"])
    check("(t) init resolves to b's best.pt row; a yaml init is scratch",
          t["init"]["kind"] == "row" and t["init"]["model_id"] == ms[FIX["b_best"]]["model_id"]
          and Z._resolve_init("yolo11n.yaml", None, Z.Index([], []), conf()[0])["kind"] == "scratch", t["init"])
    # a changed manifest: rating none, with the reason
    m = dict(ms[FIX["a"]])
    rd = C.INC_DIR / "zt_inc" / "runs" / "chg__s0"
    rd.mkdir(parents=True, exist_ok=True)
    rj = json.loads((C.INC_DIR / "zt_inc" / "runs" / "base__s0" / "run.json").read_text())
    rj["train_manifest_sha256"] = "1" * 64
    (rd / "run.json").write_text(json.dumps(rj))
    m["inc"] = dict(m["inc"], run_dir=str(rd))
    got = Z.provenance_one(m, Z.Index(Z.read_files(V), Z.read_models(V)), conf()[0], Z.Lister(), {"version": V})
    shutil.rmtree(rd)
    check("a manifest that no longer hashes as run.json records: rating none, with the reason",
          got["data"]["rating"] == "none" and "no longer hashes" in got["data"]["reason"], got["data"])
    # listing mirrors Ultralytics: a symlinked subdirectory is followed, a loop ends, hidden names are skipped
    d = TMP / "lister"
    img(d / "a" / "x1.jpg", 1)
    img(d / "a" / ".hidden.jpg", 2)
    (d / "a" / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    img(TMP / "lister_sub" / "y1.png", 3)
    link(d / "a" / "sub", TMP / "lister_sub")
    link(d / "a" / "loop", d / "a")
    got = Z.Lister().entries(str(d / "a"))
    names = sorted(os.path.basename(p) for p, _m in got["entries"])
    check("listing follows a symlinked subdirectory, ends a symlink loop and skips hidden names",
          names == ["x1.jpg", "y1.png"], names)
    rel = TMP / "relyaml" / "data.yaml"
    rel.parent.mkdir(parents=True, exist_ok=True)
    rel.write_text("path: some/relative\ntrain: images\n")
    dd, why = Z.dataset_dirs(str(rel), "train")
    check("a relative dataset path is not resolved (it depends on the training process's working directory)",
          dd is None and "relative" in why, why)
    g = subprocess.run(["git", "-C", str(ROOT.parent), "log", "-1", "--format=%H"], capture_output=True, text=True)
    if g.returncode != 0 or not g.stdout.strip():
        print("  NOTE no git history here; the codever check is skipped")
        return
    cv, _o = quiet(Z.codever_cmd, V, ROOT.parent)
    check("codever maps an INC row's module sha256s to the commit that holds them (git history here)",
          cv["rows"][ms[FIX["a"]]["model_id"]]["commit"] and cv["rows"][ms[FIX["a"]]["model_id"]]["kind"] == "exact",
          cv["rows"].get(ms[FIX["a"]]["model_id"]))



def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            h.update(str(p).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_exams():
    print("exams: the zoo root (copies, test v1, the evaluation-group samples), its LOCK, nothing else touched")
    from weed_optimizer_framework.tools.inc import splits as SPL
    from weed_optimizer_framework.tools.inc_autopilot import remote as R
    before_splits = tree_hash(C.INC_DIR / "splits")
    prior_before = B3.prior_test_lists(C.INC_DIR / "splits")[0]
    run_steps("exams")
    root = Z.root_dir(V)
    lock = json.loads((root / "splits" / "v1" / "LOCK.json").read_text())
    lock1 = json.loads(C.LOCK_PATH.read_text())
    check("the zoo LOCK records inc/scorer.py's sha256 (LOCK v1's) and dev, test, imageweeds as v1's bytes",
          lock["scorer_sha256"] == lock1["scorer_sha256"] == sha(pathlib.Path(S.__file__).resolve())
          and all(lock["manifests"][e] == lock1["manifests"][e] == sha(root / "splits" / "v1" / ("%s.jsonl" % e))
                  for e in ("dev", "test", "imageweeds")), lock["manifests"])
    tv = C.read_manifest(root / "splits" / "v1" / "test_v1.jsonl")
    check("test v1: 6 rows keyed tv1__<key>, every label class 12, per-source manifests",
          len(tv) == 6 and all(r["key"].startswith("tv1__") for r in tv)
          and all(b[0] == 12 for r in tv for b in C.read_yolo(r["label"])) and {r["source"] for r in tv} ==
          {"src0", "src1"}, [r["key"] for r in tv])
    ex = json.loads((Z.zoo_dir(V) / "exams.json").read_text())
    evg = ex["exams"]["evalgroups_v1"]
    maize = evg["groups"]["maize"]
    ev = C.read_manifest(root / "splits" / "v1" / "evalgroups_v1.jsonl")
    labs = [C.read_yolo(r["label"]) for r in ev]
    check("evalgroups_v1: the maize slug is read (crop boxes removed, weed -> 12); its crop-only image is dropped "
          "(no_weed) and its copy of a dev image (near_cwd12_eval)",
          len(ev) == 6 and all(len(b) == 1 and b[0][0] == 12 for b in labs)
          and maize["dropped"].get("no_weed") == 1 and maize["dropped"].get("near_cwd12_eval") == 1,
          (len(ev), maize))
    nr = evg["not_read"]
    check("rf_test-8qezo is not read (no class resolves to weed: '0', '1' unresolved); the unregistered slug is "
          "not read (not in the registry)",
          "no class resolves to weed" in nr.get("rf_test-8qezo__weed-detection-ycai2", "")
          and "not in the registry" in nr.get("kg_vinayakshanawad__weedcrop", ""), nr)
    check("ooddev_v1 reads nothing here (its slugs are not in this registry); the roles and role of each exam are "
          "recorded", ex["exams"]["ooddev_v1"]["n_images"] == 0 and ex["exams"]["test_v1"]["role"] == "sealed"
          and ex["exams"]["dev"]["role"] == "decision" and ex["exams"]["test"]["role"] == "descriptive",
          {e: (x["n_images"], x["role"]) for e, x in ex["exams"].items()})
    c, _s = conf()
    again = Z._eval_exams(V, c, Z.source_inc(), root, B3.hash_matrix(
        Z._copy_hashes(V, Z._copy_exams(V, Z.source_inc(), root)[0], Z.source_inc(), Z.HashCache(V), 2)["dev"]),
        Z.HashCache(V), 2)[0]
    check("the sample is deterministic (a second build gives the same keys)",
          sorted(r["key"] for r in again["evalgroups_v1"]) == sorted(r["key"] for r in ev))
    probs = [SPL.exam_problems(e, C.read_manifest(root / "splits" / "v1" / ("%s.jsonl" % e)),
                               root / "exams" / "v1" / e) for e in ("test_v1", "evalgroups_v1")]
    check("inc.splits.exam_problems finds nothing in the materialised exams", probs == [[], []], probs)
    check("nothing under INC_DIR/splits changed; prior_test_lists reads the same rows",
          tree_hash(C.INC_DIR / "splits") == before_splits
          and B3.prior_test_lists(C.INC_DIR / "splits")[0] == prior_before)
    check("the zoo lies outside every experiment-name glob and remote.status",
          not R.NAME_RE.match("_zoo") and "_zoo" not in [x["exp"] for x in R.status()["experiments"]])
    # a planted label byte change in test v1 refuses
    lp = pathlib.Path(WORLD["tv1"][1]["label"])
    old = lp.read_bytes()
    lp.write_bytes(old + b"12 0.1 0.1 0.05 0.05\n")
    e = refused(Z._test_v1, V, c, Z.source_inc(), root)
    lp.write_bytes(old)
    check("a test v1 label whose bytes changed refuses", e is not None and "no longer hashes" in str(e), e)
    # an exam image whose dHash is not the never-train index's refuses
    idxp = C.NEVER_TRAIN_INDEX
    raw = idxp.read_bytes()
    d = json.loads(raw)
    d["entries"][0][0] = int(d["entries"][0][0]) ^ 0xFF
    idxp.write_text(json.dumps(d))
    copies = Z._copy_exams(V, Z.source_inc(), root)[0]
    e = refused(Z._copy_hashes, V, copies, Z.source_inc(), Z.HashCache(V), 2)
    idxp.write_bytes(raw)
    check("an exam image whose dHash differs from the never-train index refuses",
          e is not None and "never-train index" in str(e), e)


def test_contamination():
    print("contamination: what each training list holds of every exam, the states, inheritance")
    run_steps("contamination")
    ct = Z.read_contamination(V)
    ms = models_by_rel()
    cb = ct[ms[FIX["b_best"]]["model_id"]]
    k = cb["counts"]
    check("(b): dev exact 2, test near 1 (the vflip copy), test v1 exact 1, ImageWeeds 0, the maize group 1",
          k["dev"]["exact"] == 2 and k["test"]["near"] == 1 and k["test"]["exact"] == 0
          and k["test_v1"]["exact"] == 1 and k["imageweeds"]["exact"] + k["imageweeds"]["near"] == 0
          and cb["eval_groups"].get("maize") == 1, (k, cb["eval_groups"]))
    check("(b) states: dev Y, test Y, test_v1 Y, ImageWeeds N; legacy_join (a pre-09-21 R1 head on external images)",
          cb["final"]["dev"] == "Y" and cb["final"]["test"] == "Y" and cb["final"]["test_v1"] == "Y"
          and cb["final"]["imageweeds"] == "N" and cb["legacy_join"], (cb["final"], cb["legacy_join"]))
    meta, arrays = Z.load_exam_hashes(V)
    h, var = G.image_hashes(str(WORLD["vflip"]))
    HE, okE, hasE = arrays["exam_test__H"], arrays["exam_test__ok"], arrays["exam_test__has"]
    d_on, _a = B3.cross_nearest(np.asarray([[h] + [var[x] for x in G.VARIANTS[1:]]], dtype=np.uint64),
                                np.asarray([True]), HE[hasE], okE[hasE], 6)
    d_off, _a = B3.cross_nearest(np.asarray([[h] + [0] * 7], dtype=np.uint64), np.asarray([False]),
                                 HE[hasE], np.zeros(int(hasE.sum()), bool), 6)
    check("the vflip copy is found only through the 8 variants (the dHash alone misses it)",
          int(d_on[0]) <= 6 and int(d_off[0]) > 6, (int(d_on[0]), int(d_off[0])))
    ck, cj, ct_, cj2 = (ct[ms[FIX[x]]["model_id"]] for x in ("k", "j", "t", "j2"))
    check("(k) a rebuilt list with a dev image: P; its test v1 companion is counted; (j) no list: U; (j2) P",
          ck["final"]["dev"] == "P" and ck["counts"]["test_v1"]["companion"] == 1 and cj["final"]["dev"] == "U"
          and cj2["final"]["dev"] == "P", (ck["final"], ck["counts"]["test_v1"], cj["final"], cj2["final"]))
    check("(t) its own list is clean (N) and a cwd12 train image of a dev session counts in session only; it "
          "inherits Y from b's best.pt",
          ct_["own"]["dev"] == "N" and ct_["counts"]["dev"]["session"] == 1 and ct_["final"]["dev"] == "Y"
          and ct_["inherited"]["states"]["dev"] == "Y", (ct_["own"], ct_["counts"]["dev"], ct_["final"]))
    ca = ct[ms[FIX["a"]]["model_id"]]
    check("(a) INC: its manifest holds a dev image: dev Y (exact)", ca["final"]["dev"] == "Y"
          and ca["counts"]["dev"]["exact"] == 1, ca["counts"]["dev"])
    tol = 0.01
    st2 = Z.own_state("listed_exact", {"exact": 0, "near": 0, "unhashed": 2, "n": 100}, 100, tol)
    st05 = Z.own_state("listed_exact", {"exact": 0, "near": 0, "unhashed": 1, "n": 200}, 200, tol)
    st0 = Z.own_state("listed_exact", {"exact": 0, "near": 0, "unhashed": 0, "n": 0}, 0, tol)
    check("2 % unhashed and no hit: P; 0.5 %: N; an empty list: U (N needs a non-empty list)",
          (st2, st05, st0) == ("P", "N", "U"), (st2, st05, st0))
    cu = ct[ms[FIX["u"]]["model_id"]]
    check("(u) a clean list but an init the zoo cannot resolve: own N, inherited U, final U",
          cu["own"]["dev"] == "N" and cu["inherited"]["states"]["dev"] == "U" and cu["final"]["dev"] == "U",
          (cu["own"]["dev"], cu["inherited"], cu["final"]["dev"]))



def plant(mid, exam, ag, sp=None, origin="scored", per_class=None):
    rec = {"format": Z.SCORE_FORMAT, "model_id": mid, "exam": exam, "origin": origin,
           "result": {"agnostic_map50_95": ag, "agnostic_map50": ag, "species_map50_95": sp, "map50_95": sp,
                      "per_class": per_class or {}, "n_images": 8, "seconds": 1.0},
           "production": False, "scorer_stamp": "TEST-ZOO-x", "created_utc": Z._utc()}
    return Z._link_once(Z.score_path(V, mid, exam), rec)


def test_plan():
    print("the plan: reused INC records, stage A whole, stage B cut to the cap, LPT shards, written once")
    c, csha = conf()
    quiet(Z.pilot_step, V, c, csha, False)
    pilot = json.loads((Z.zoo_dir(V) / "pilot.json").read_text())
    picks = pilot["models"]
    ms = models_by_rel()
    inc_ids = {m["model_id"] for m in ms.values() if Z.is_inc_family(m["family"])}
    check("the pilot picks non-INC rows only (one per size class), and writes every exam for them",
          picks and not (set(picks.values()) & inc_ids) and len(Z.read_shard(V, "pilot")["items"]) ==
          len(picks) * len(Z.ALL_EXAMS), picks)
    rates = {e: {"S": 1.0, "M": 1.0, "L": 1.0} for e in Z.ALL_EXAMS}
    pilot["rates"] = rates
    (Z.zoo_dir(V) / "pilot.json").write_text(json.dumps(pilot))
    e = refused(quiet, Z.plan_step, V, c, csha, 4, 9.5, 0.0)
    rec = Z.read_record(V) or {}
    check("stage A over the cap: refused, the record says refused, no shard written",
          e is not None and "stage A alone" in str(e) and rec.get("status") == "refused"
          and not list((Z.zoo_dir(V) / "shards").glob("a_*.json")), (e, rec.get("status")))
    c2 = json.loads(json.dumps(c))
    c2["stages"]["budget_b_gpu_hours"] = 0.01
    plan, _o = quiet(Z.plan_step, V, c2, csha, 4, 40, 0.0)
    a_id = ms[FIX["a"]]["model_id"]
    kd, rd = Z.record_of(V, a_id, "dev")
    kt, _rt = Z.record_of(V, a_id, "test")
    check("(a)'s recorded dev score (and its kind-final run's test score) are reused, never re-scored",
          kd == "reused" and rd["from"]["name"].endswith("base__s0/scores/dev.json") and kt == "reused"
          and plan["reused"].get("dev") == 1 and plan["reused"].get("test") == 1, (kd, kt, plan["reused"]))
    shards = [json.loads(p.read_text()) for p in sorted((Z.zoo_dir(V) / "shards").glob("a_*.json"))]
    items = [it for sh in shards for it in sh["items"]]
    tv1 = [it for it in items if it["exam"] == "test_v1"]
    devs = [it for it in items if it["exam"] == "dev"]
    check("stage A whole (dev for every scorable row without a record, never an INC row's test v1); stage B cut "
          "to its budget, the rest listed not_scored_budget",
          len(devs) == plan["items_a"] and a_id not in {it["model_id"] for it in items}
          and len(tv1) == plan["items_b"] < len(tv1) + len(plan["items_b_dropped"])
          and all(d["why"] == "not_scored_budget" for d in plan["items_b_dropped"]), plan)
    s_ = [sh["predicted_s"] for sh in shards]
    check("exactly --shards-a shard files, LPT-balanced (max - min <= one item)",
          len(shards) == 4 and max(s_) - min(s_) <= max(it["predicted_s"] for it in items) + 1e-6, s_)
    again, _o = quiet(Z.plan_step, V, c2, csha, 4, 40, 0.0)
    e = refused(quiet, Z.plan_step, V, c2, csha, 5, 40, 0.0)
    check("written once: a second plan keeps it; another --shards-a refuses",
          again["created_utc"] == plan["created_utc"] and e is not None, e)
    Z.mark_step(V, "pilot", csha)
    rc, _o = quiet(Z.main, ["inventory", "--version", V, "--shards-a", "4", "--shards-c", "2", "--max-gpu-hours", "40"])
    check("the inventory verb keeps every finished step and ends plan_ready (record plan_ready, a ledger entry)",
          rc == 0 and Z.read_record(V)["status"] == "plan_ready"
          and list((Z.zoo_dir(V) / "ledger").glob("inventory__*.json")), rc)
    WORLD["plan"] = plan


def test_select():
    print("select: the shortlist from dev and provenance only, the cap, stage C's items within what is left")
    c, csha = conf()
    ms = models_by_rel()
    vals = {"d": 0.5, "e": 0.7, "f": 0.6, "b_best": 0.9, "b_last": 0.85, "c": 0.4, "k": 0.3, "j": 0.2, "j2": 0.25,
            "t": 0.95, "orphan": 0.1, "m_merged": 0.15}
    for k, v in vals.items():
        plant(ms[FIX[k]]["model_id"], "dev", v)
    ld = Z.zoo_dir(V) / "ledger"
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "a__999_0.json").write_text(json.dumps({"format": Z.LEDGER_FORMAT, "kind": "a", "job": "999", "task": "0",
                                                 "started_s": 1000.0, "updated_s": 1000.0 + 7200,
                                                 "ended_s": 1000.0 + 7200}))
    c2 = json.loads(json.dumps(c))
    c2["shortlist"]["claims"] = [{"match": "results/framework/sp8/weights/best\\.pt\\Z", "cite": "docs/x.md:1"}]
    c2["shortlist"]["max"] = 5
    sl, _o = quiet(Z.select_step, V, c2, csha, 2)
    ids = sl["rules"]
    mid = lambda k: ms[FIX[k]]["model_id"]  # noqa: E731
    check("rule 1: the claim (d's path)", ids["claims"] == [mid("d")], ids)
    check("rule 3: the top 3 of family 'other' among dev-clean rows by dev (e 0.7, f 0.6; d already a claim), "
          "never a flagged row by its dev; a family with no clean row is filled by date (yolo_iter: j2)",
          set(ids["top_dev_clean"]) == {mid("e"), mid("f")} and mid("j2") in ids["fill_by_date"]
          and mid("t") not in ids["top_dev_clean"], ids)
    check("the cap (5) drops the 'latest' entries first; at most one row per run directory; INC rows never "
          "shortlisted", len(sl["ids"]) <= 5 and not ids["latest"] and ms[FIX["a"]]["model_id"] not in sl["ids"],
          sl["ids"])
    sh = [json.loads(p.read_text()) for p in sorted((Z.zoo_dir(V) / "shards").glob("c_*.json"))]
    items = [it for s_ in sh for it in s_["items"]]
    order = c["shortlist"]["exam_order"]
    per = {}
    for it in items:
        per.setdefault(it["model_id"], []).append(it["exam"])
    check("stage C: the shortlist on test v1 (when B did not), the evaluation groups, OOD dev, ImageWeeds, cwd12 "
          "test; two LPT shards; budget_c = cap - spent (2 h of stage A, planted) - the report reserve",
          len(sh) == 2 and all(set(v) <= set(order) for v in per.values())
          and abs(sl["budget_c_h"] - (40 - sl["spent_h"]["total"] - 1.0)) < 1e-6 and sl["spent_h"]["total"] >= 2.0,
          (sl["budget_c_h"], sl["spent_h"]))
    e2 = refused(quiet, Z.select_step, V, c2, csha, 3)
    check("a second select with other --shards-c refuses (written once)", e2 is not None, e2)
    pj = Z.zoo_dir(V) / "plan.json"
    raw = pj.read_bytes()
    pj.unlink()
    (Z.zoo_dir(V) / "shortlist.json").rename(Z.zoo_dir(V) / "shortlist.keep")
    e3 = refused(quiet, Z.select_step, V, c2, csha, 2)
    pj.write_bytes(raw)
    (Z.zoo_dir(V) / "shortlist.keep").rename(Z.zoo_dir(V) / "shortlist.json")
    check("no plan: select refuses (exit 2)", e3 is not None and "no plan" in str(e3), e3)


def plant_all(skip=()):
    """A score record for every planned item (but `skip`) and a done task record for every shard."""
    for (mid, exam), (stage, idx) in Z._planned(V).items():
        if (mid, exam) in skip or Z.record_of(V, mid, exam)[0] is not None:
            continue
        plant(mid, exam, 0.42, 0.33, origin="pilot" if stage == "pilot" else "scored",
              per_class={"Ragweed": 0.2, "Waterhemp": 0.4})
    for p in sorted((Z.zoo_dir(V) / "shards").glob("*.json")):
        sh = json.loads(p.read_text())
        task = "pilot" if sh["stage"] == "pilot" else "%s_%03d" % (sh["stage"], sh["index"])
        bad = any((it["model_id"], it["exam"]) in skip for it in sh["items"])
        Z._write_json(Z.zoo_dir(V) / "tasks" / ("%s.json" % task),
                      {"format": Z.TASK_FORMAT, "task": task, "status": "systemic" if bad else "done",
                       "items": [{"model_id": it["model_id"], "exam": it["exam"], "status": "scored"}
                                 for it in sh["items"]]})


def test_report():
    print("report: the table, its sections and legend, partial vs complete, the platform record")
    from weed_optimizer_framework.tools.inc_autopilot import evidence as E
    from weed_optimizer_framework.tools.inc_autopilot import remote as R
    c, csha = conf()
    sh = sorted((Z.zoo_dir(V) / "shards").glob("c_*.json"))
    one = json.loads(sh[0].read_text())["items"][0]
    plant_all(skip={(one["model_id"], one["exam"])})
    rep, _o = quiet(Z.report_step, V, c, csha)
    check("partial while an item has no record; the failed shard is named", rep["status"] == "partial"
          and "c_000" in rep["failed_shards"] and any(m["model_id"] == one["model_id"] for m in
                                                       rep["items_without_record"]), rep["failed_shards"])
    plant(one["model_id"], one["exam"], 0.4)
    plant_all()
    rep, _o = quiet(Z.report_step, V, c, csha)
    check("complete once every planned item has a record and every task record is done", rep["status"] ==
          "complete", (rep["status"], rep["failed_shards"], rep["items_without_record"][:3]))
    md = (Z.zoo_dir(V) / "report.md").read_text()
    ms = models_by_rel()
    secs = rep["sections"]
    rows = {r["model_id"]: r for r in rep["rows"]}
    clean = secs["agnostic_dev_clean"]
    devs = [rows[i]["cols"]["dev_ag"] for i in clean if rows[i]["cols"]["dev_ag"] is not None]
    check("report.md carries the usage rule verbatim, every skip reason with its count, and the footer",
          Z.USAGE_RULE in md and all("| %s |" % r in md for r in Z.SKIP_REASONS) and Z.FOOTER in md)
    check("dev-clean agnostic rows ranked by dev agnostic (descending); flagged rows unranked (family, date); "
          "the unscorable section holds n and m2",
          devs == sorted(devs, reverse=True) and ms[FIX["n"]]["model_id"] in secs["unscorable"]
          and ms[FIX["m2"]]["model_id"] in secs["unscorable"] and ms[FIX["t"]]["model_id"] not in clean, secs)
    hdr = (Z.zoo_dir(V) / "report.csv").read_text().splitlines()[0].split(",")
    check("report.csv columns as specified", tuple(hdr) == Z.CSV_COLUMNS, hdr)
    ext = json.loads((Z.zoo_dir(V) / "report_external.json").read_text())
    check("test v1 and evaluation-group values only in report_external (never in report.json's rows or report.md)",
          all(r["external"] is None and not any(e in r["scores"] for e in Z.SEALED_EXAMS) for r in rep["rows"])
          and "tv1" not in md.split("## Legend")[0].lower().replace("tv1:", "") and len(ext["rows"]) == len(rep["rows"]),
          [k for k in rep["rows"][0]["scores"]])
    b = rows[ms[FIX["b_best"]]["model_id"]]
    check("b: legacy_join (species columns not interpretable), dev Y, selected on the cwd12 test",
          b["flags"]["legacy_join"] and b["cols"]["dev_sp"] is None and b["flags"]["dev"] == "Y"
          and b["flags"]["test_selected"] == "best", b["flags"])
    rec = Z.read_record(V)
    check("the platform record: complete, counts and sha256s only (no non-dev key, no score path)",
          rec["status"] == "complete" and R.non_dev_keys(rec) == [] and not E._NON_DEV_SCORE.search(json.dumps(rec))
          and rec["counts"]["files_listed"] == rep["counts"]["files_listed"], rec)



def fake_bin(fail_at=None):
    """sbatch, scontrol, scancel and squeue stand-ins logging their argv."""
    b = TMP / "bin"
    b.mkdir(exist_ok=True)
    log = TMP / "slurm_calls.jsonl"
    (b / "sbatch").write_text("#!/bin/bash\nn=$(cat %s/sb_n 2>/dev/null || echo 0); n=$((n+1)); echo $n > %s/sb_n\n"
                              "python3 -c 'import json,sys; print(json.dumps([\"sbatch\"]+sys.argv[1:]))' \"$@\" >> %s\n"
                              "if [ -n \"${FAIL_AT:-}\" ] && [ \"$n\" = \"$FAIL_AT\" ]; then echo 'socket timed out' >&2;"
                              " exit 1; fi\necho \"$((1000+n));cluster\"\n" % (TMP, TMP, log))
    for name in ("scontrol", "scancel"):
        (b / name).write_text("#!/bin/bash\npython3 -c 'import json,sys; print(json.dumps([\"%s\"]+sys.argv[1:]))' "
                              "\"$@\" >> %s\n" % (name, log))
    (b / "squeue").write_text("#!/bin/bash\ncat %s/squeue.txt 2>/dev/null\nexit 0\n" % TMP)
    for x in b.iterdir():
        os.chmod(x, 0o755)
    for f in (log, TMP / "sb_n", TMP / "squeue.txt"):
        if f.exists():
            f.unlink()
    return {"ZOO_SBATCH": str(b / "sbatch"), "ZOO_SCONTROL": str(b / "scontrol"), "ZOO_SCANCEL": str(b / "scancel"),
            "ZOO_SQUEUE": str(b / "squeue"), "FAIL_AT": str(fail_at) if fail_at else None, "SLURM_JOB_ID": None,
            "USER": "tester"}


def calls():
    p = TMP / "slurm_calls.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def test_submit():
    print("submit: five jobs without a hold, the record, the INCZOO line; refusals")
    script = C.REPO / "weed_llm_benchmark" / "run_inc2_zoo.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "run_inc2_zoo.sh", script)
    rp = Z.record_path(V)
    args = ["submit", "--version", V, "--shards-a", "4", "--shards-c", "2", "--concurrency", "6",
            "--max-gpu-hours", "40"]
    with env(**fake_bin()):
        rc, _o = quiet(Z.main, args)
    check("a complete record refuses a second submission (exit 2)", rc == 2 and not calls(), rc)
    rp.unlink()
    with env(**fake_bin()):
        (TMP / "squeue.txt").write_text("77|inc_zoo_v1_score_a\n")
        rc, _o = quiet(Z.main, args)
    check("a queued zoo job refuses (exit 2), nothing submitted", rc == 2 and not calls() and not rp.exists(), rc)
    with env(**dict(fake_bin(), SLURM_JOB_ID="5")):
        rc, _o = quiet(Z.main, args)
    check("inside a Slurm job: refused (exit 2)", rc == 2 and not calls(), rc)
    with env(**fake_bin()):
        rc, _o = quiet(Z.main, args[:4] + ["32"] + args[5:])
    check("--shards-a other than the existing plan's refuses", rc == 2 and not calls(), rc)
    with env(**dict(fake_bin(), INCAP_DECIDED_BY="human:x@y.z", INCAP_TRIGGER="Z1")):
        rc, out = quiet(Z.main, args)
    cl = calls()
    sb = [c[1:] for c in cl if c[0] == "sbatch"]
    S_ = str(script)
    ok = len(sb) == 5 and "--hold" not in sum(sb, []) and all("-p" in a and a[a.index("-p") + 1] == "GPU-shared"
                                                              for a in sb)
    ok = ok and "--array=0-3%6" in sb[1] and "--dependency=afterok:1001" in sb[1] and "--array=0-1%6" in sb[3] \
        and "--dependency=afterany:1002" in sb[2] and "--dependency=afterok:1003" in sb[3] \
        and "--dependency=afterany:1004" in sb[4] and sum("--kill-on-invalid-dep=yes" in a for a in sb) == 4 \
        and [a[a.index(S_) + 1] for a in sb] == ["inventory", "score", "select", "score", "report"]
    rec = Z.read_record(V) or {}
    line = [ln for ln in out.splitlines() if ln.startswith("INCZOO ")]
    check("five sbatch argvs in order (no --hold; afterok/afterany chain; arrays 0-3%6 and 0-1%6; GPU-shared; "
          "--kill-on-invalid-dep on four)", rc == 0 and ok, sb)
    check("the record says submitted with five ids, who decided and the trigger; the last line is INCZOO",
          rec.get("status") == "submitted" and [rec["jobs"][k] for k in ("inventory", "score_a", "select", "score_c",
                                                                           "report")] == ["1001", "1002", "1003",
                                                                                          "1004", "1005"]
          and rec.get("decided_by") == "human:x@y.z" and rec.get("trigger") == ["Z1"] and line
          and json.loads(line[-1][7:])["job_ids"] == ["1001", "1002", "1003", "1004", "1005"], (rec, line))
    rp.write_text(json.dumps(dict(rec, status="plan_ready")))
    with env(**dict(fake_bin(fail_at=3))):
        rc, _o = quiet(Z.main, args)
    cl = calls()
    check("a resubmission of a stale record (nothing queued) runs; the third sbatch failing cancels the two "
          "submitted and leaves the earlier record", rc == 1 and [c for c in cl if c[0] == "scancel"] ==
          [["scancel", "1001"], ["scancel", "1002"]] and (Z.read_record(V) or {}).get("status") == "plan_ready",
          (rc, cl, Z.read_record(V)))
    code = ("import sys; sys.path.insert(0, %r); from weed_optimizer_framework.tools.inc2 import zoo; "
            "rc = zoo.main(%r); print('TORCH', 'torch' in sys.modules, rc)" % (str(ROOT), args))
    e = dict(os.environ, **{k: v for k, v in fake_bin().items() if v is not None})
    e.pop("SLURM_JOB_ID", None)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)
    check("submit never imports torch", "TORCH False 0" in p.stdout, (p.stdout[-300:], p.stderr[-500:]))


def test_script():
    print("run_inc2_zoo.sh: the job script")
    sh = ROOT / "run_inc2_zoo.sh"
    text = sh.read_text()
    r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    check("bash -n passes; GPU-shared with one v100-32; a literal log path", r.returncode == 0
          and "#SBATCH --partition=GPU-shared" in text and "#SBATCH --gres=gpu:v100-32:1" in text
          and "#SBATCH --output=/ocean/" in text, r.stderr)
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    check("no git reset, rsync or copy of the package", "git reset" not in body and "rsync" not in body
          and not any(ln.strip().startswith("cp ") for ln in body.splitlines()))
    repo = TMP / "repo2"
    (repo / "weed_llm_benchmark").mkdir(parents=True, exist_ok=True)
    for d in (repo / "weed_optimizer_framework", repo / "weed_llm_benchmark" / "weed_optimizer_framework"):
        if not d.exists():
            os.symlink(str(ROOT / "weed_optimizer_framework"), str(d))
    stub = TMP / "stubbin"
    stub.mkdir(exist_ok=True)
    (stub / "python").write_text(
        "#!/bin/bash\nif [ \"$1\" = \"-\" ]; then cat >/dev/null; exit 0; fi\n"
        "echo \"$@ INC_DIR=$INC_DIR SOURCE=${INC_ZOO_SOURCE:-}\" >> %s/py_calls.txt\n"
        "case \"$*\" in *exam-root*) echo %s/localroot ;; esac\nexit 0\n" % (TMP, TMP))
    os.chmod(stub / "python", 0o755)
    conda = TMP / "conda.sh"
    conda.write_text("conda() { :; }\n")
    base = {"INC_BUILD_REPO": str(repo), "INC_BUILD_CONDA_SH": str(conda), "INC_DIR": str(TMP / "incroot"),
            "PATH": "%s:%s" % (stub, os.environ["PATH"]), "SLURM_JOB_ID": None, "SLURM_ARRAY_JOB_ID": None,
            "SLURM_ARRAY_TASK_ID": None, "LOCAL": None}
    with env(**base):
        r1 = subprocess.run(["bash", str(sh), "inventory", "--version", V], capture_output=True, text=True)
    with env(**dict(base, SLURM_JOB_ID="9")):
        r2 = subprocess.run(["bash", str(sh), "submit", "--version", V], capture_output=True, text=True)
    check("job modes refuse outside a job, login modes inside one (exit 2)", r1.returncode == 2
          and r2.returncode == 2, (r1.stderr[-200:], r2.stderr[-200:]))
    (TMP / "py_calls.txt").unlink() if (TMP / "py_calls.txt").exists() else None
    with env(**dict(base, SLURM_JOB_ID="77", SLURM_ARRAY_JOB_ID="70", SLURM_ARRAY_TASK_ID="3")):
        r3 = subprocess.run(["bash", str(sh), "score", "--version", V, "--stage", "a"], capture_output=True,
                            text=True)
    pc = [ln for ln in ((TMP / "py_calls.txt").read_text().splitlines() if (TMP / "py_calls.txt").exists() else [])
          if "weed_optimizer_framework.tools.inc2.zoo" in ln]
    check("score mode: exam-root (INC_DIR the INC tree), then score with INC_DIR at the printed root and "
          "INC_ZOO_SOURCE the INC tree", r3.returncode == 0 and len(pc) == 2 and "exam-root" in pc[0]
          and "--shard 3" in pc[0] and "INC_DIR=%s" % (TMP / "incroot") in pc[0]
          and "score --version v1 --stage a --shard 3" in pc[1] and "INC_DIR=%s/localroot" % TMP in pc[1]
          and "SOURCE=%s" % (TMP / "incroot") in pc[1], (r3.stdout[-500:], r3.stderr[-500:], pc))
    check("the job's provenance and lock live under _zoo/v1 (never _campaign/)",
          (TMP / "incroot" / "_zoo" / V / "provenance").is_dir() and (TMP / "incroot" / "_zoo" / V / "locks").is_dir()
          and not (TMP / "incroot" / "_campaign").exists() and not (TMP / "localroot").exists())
    # the import closure of a full run: every weed_optimizer_framework module and config it opened is drift-checked
    mods = set(text.split("MODULES=(", 1)[1].split(")", 1)[0].split())
    code = r"""
import builtins, io, json, os, sys
sys.path.insert(0, %r)
opened = set()
_open = builtins.open
def rec_open(f, *a, **k):
    try:
        opened.add(os.path.realpath(str(f)))
    except Exception:
        pass
    return _open(f, *a, **k)
builtins.open = rec_open
io.open = rec_open
from weed_optimizer_framework.tools.inc2 import zoo as Z
conf, sha = Z.load_config("v1")
for st in ("list", "meta", "provenance", "convert", "exams", "contamination"):
    os.environ["INC_ZOO_FORCE"] = "1"
    Z.run_step("v1", conf, sha, st, None) if not Z.step_done("v1", st, sha) else None
for p in Z.step_marker("v1", "x").parent.glob("*.json"):
    pass
m = next(x for x in Z.read_models("v1") if Z.scorable(x, "v1") and x["class_map"]["rule"] != "R0")
got = Z.scorable(m, "v1")
it = {"model_id": m["model_id"], "exam": "test_v1", "file": got[0], "file_sha256": got[1]}
Z.score_item("v1", conf, sha, it, "closure")
pkg = os.path.realpath(%r)
out = sorted({os.path.relpath(os.path.realpath(mod.__file__), pkg) for mod in list(sys.modules.values())
              if getattr(mod, "__file__", None) and os.path.realpath(mod.__file__).startswith(pkg + os.sep)})
out += sorted({os.path.relpath(p, pkg) for p in opened if p.startswith(pkg + os.sep) and p.endswith(".json")})
print("CLOSURE " + json.dumps(out))
""" % (str(ROOT), str(ROOT / "weed_optimizer_framework"))
    for st in ("list", "meta", "provenance", "exams", "contamination"):
        mk = Z.step_marker(V, st)
        if mk.exists():
            mk.unlink()
    e = dict(os.environ, INC_DIR=str(Z.root_dir(V)), INC_ZOO_SOURCE=str(C.INC_DIR))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, timeout=900)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("CLOSURE ")]
    closure = set(json.loads(line[-1][8:])) if line else None
    closure = {x for x in closure or () if not x.startswith("tools/inc_autopilot/")} if closure is not None else None
    missing = sorted(closure - mods) if closure is not None else None
    check("MODULES lists every weed_optimizer_framework module and config a full run (every inventory step and a "
          "score) imports or opens", closure is not None and not missing,
          (missing, p.stdout[-300:], p.stderr[-1500:]))


def test_pinned_unchanged():
    print("pinned modules")
    pkg = ROOT / "weed_optimizer_framework" / "tools"
    paths = [str(p) for p in sorted((pkg / "inc").glob("*.py"))] + [str(pkg / "cwd12_species.py")]
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"] + paths, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode not in (0, 1):
        print("  NOTE git is not usable here (%s); the pinned-module check is skipped" % r.stderr.strip()[:200])
        return
    check("inc/*.py and cwd12_species.py are unchanged against git HEAD", r.returncode == 0)
    check("the zoo's job names and the scorer's pin agree with stream_remote and inc/scorer.py",
          Z.PINNED_ULTRALYTICS == S.PINNED_ULTRALYTICS)


def main():
    world()
    for t in (test_list, test_class_maps, test_meta, test_convert, test_provenance, test_exams,
              test_contamination, test_plan, test_select, test_report, test_submit, test_script,
              test_pinned_unchanged):
        t()
    print("\n%d failure(s)" % len(FAILURES))
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
