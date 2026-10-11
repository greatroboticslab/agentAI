#!/usr/bin/env python3
"""V3F D6: RF-DETR-L 12-class on base A under the protocol (inc2/v3fam_detr.py;
docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-11): V3F, the v3 experiment family
D0-D9", V3F.5 D6, V3F.6.5).

rfdetr is never imported: a fake backend (FakeBackend) stands for its train,
export, load, predict and YOLO-reader calls, and writes the files rfdetr writes
in its output directory (last.ckpt, the three checkpoint_best_* files with
other weights than the EMA's, metrics.csv). Two groups.

UNITS (the synthetic world's splits, no scoring pass; V3FAM_ONLY=units):
- the recipe (50 epochs, batch 2 x 8, lr 1e-4 / 1.5e-4, EMA 0.993 / 100,
  run_test false, checkpoint_interval 1000 > 50 epochs, 704 px), the walltime
  rule (0.8 x 36 h, epoch x 50 + 0.5 h), valid = round(0.02 x 6,811) = 136,
  EXIF_STAGING unset until the rehearsal; the six E3-frozen sha256s;
- the 0-12 class map round trip (data.yaml's names -> INC ids) and its
  refusals; the converter on planted predictions (background 13 dropped and
  counted, 14 / -1 / 2.5 / NaN / a score above 1 refused);
- staging: refused without the rehearsal's outcome, refused on a planted
  holdout row (and nothing left behind), on a reader that does not turn a
  tag-6 image (symlink branch); written once with valid a subset of train and
  never dev, no test/ folder, labels as their verified bytes; the transpose
  branch on a tag-6 image (a transposed JPEG without a tag, linked from train/);
  a planted test/ folder and a planted holdout link refused on verification;
- the attempt's clean-up: its own files removed, a file it did not write (old,
  a link, another name) refused; the trainer's facts (run_test, the archive
  interval, every epoch, no archive file) checked; the rehearsal writes nothing
  under INC_DIR;
- smoke and train with the fake backend: the walltime refusal (not run
  (walltime), train refused); the smoke's checks (export equal to the
  in-process EMA, the 20-image predictions, the clean-up, its export removed);
  each seed's export equal to the EMA weights and never to the planted best
  file's, written once; an archive checkpoint refused (nothing exported,
  cleaned); SIGTERM in training cleans and exports nothing; another seed refused.
WORLD (real Ultralytics CPU passes, E3's records as in tests/test_inc2_v3fam.py,
a zoo root holding ooddev_v1): score_d6 with predictions planted as
b_v2_m640 seed 0's own identity rows reproduces that seed's family identity
dev score exactly through inject_rows (the converter, the class map and the
injection end to end), background rows dropped; ImageWeeds and ooddev_v1
reported; the reference read's inputs; a rerun keeps every file; the CLI.

Run:  python3 tests/test_inc2_v3fam_detr.py          (V3FAM_ONLY=units: the units only)
"""
import hashlib
import json
import os
import pathlib
import pickle
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
import test_inc2_e3 as E3T  # noqa: E402
import test_inc2_v3fam as FT  # noqa: E402
import test_inc2_v3fam_systems as ST  # noqa: E402

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_detr as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = E3T.patched
moved = E3T.moved
KW = FT.KW
WORLD = {}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused) as e:
        return e
    return None


# ------------------------------------------------------------------ the fake RF-DETR
class FakeModel(object):
    def __init__(self, state):
        self.state = state


def pixel_key(pil):
    return hashlib.sha256(np.asarray(pil.convert("RGB")).tobytes()).hexdigest()


class FakeBackend(object):
    """rfdetr's calls, faked: train writes what rfdetr writes in its output directory and returns EMA weights that
    depend on the seed; predict returns planted rows (by the image's pixels) or one box from the weights."""
    name = "fake"

    def __init__(self, fit_seconds=60.0, archive=False, run_test=False, short=False, term=False, transposes=True,
                 planted=None):
        self.fit_seconds, self.archive, self.run_test, self.short, self.term = fit_seconds, archive, run_test, short, \
            term
        self.transposes, self.planted, self.trained = transposes, planted or {}, []

    def versions(self):
        return {"fake": "1"}

    def train(self, stage, out, seed, epochs, pretrain):
        out = pathlib.Path(out)
        self.trained.append((str(stage), str(out), int(seed), int(epochs)))
        (out / "last.ckpt").write_bytes(os.urandom(256))
        (out / "metrics.csv").write_text("epoch,val\n0,0.1\n")
        other = {"w": np.full(4, -7.0), "b": np.zeros(3)}
        for n in ("checkpoint_best_regular.pth", "checkpoint_best_ema.pth", "checkpoint_best_total.pth"):
            with open(out / n, "wb") as fh:
                pickle.dump({"model": other, "args": {}, "epoch": 0}, fh)
        if self.archive:
            (out / "checkpoint_0.ckpt").write_bytes(b"archive")
        if self.term:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        ema = {"w": np.full(4, seed + 0.5), "b": np.arange(3.0) + epochs}
        return {"ema_state": ema, "inprocess_state": {k: v.copy() for k, v in ema.items()},
                "args": {"epochs": epochs, "seed": seed, "class_names": list(D.CLASS_NAMES)},
                "epoch": epochs - 1 if self.short else epochs, "max_epochs": epochs, "interrupted": False,
                "callbacks": {"run_test": self.run_test, "archive_every_n_epochs": D.RECIPE["checkpoint_interval"],
                              "ema": True},
                "seconds": self.fit_seconds, "model": FakeModel(ema), "class_names": list(D.CLASS_NAMES),
                "seed_mechanism": "fake"}

    def save(self, payload, path):
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    def load_state(self, path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def states_equal(self, a, b):
        return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)

    def load_model(self, path):
        return FakeModel(self.load_state(path)["model"])

    def model_class_names(self, model):
        return list(D.CLASS_NAMES)

    def predict(self, model, pils, threshold):
        out = []
        for p in pils:
            got = self.planted.get(pixel_key(p))
            if got is not None:
                out.append(got)
                continue
            w, h = p.size
            s = float(model.state["w"][0])
            out.append((np.array([[0.1 * w, 0.1 * h, 0.5 * w, 0.6 * h]]), np.array([min(0.9, 0.1 + s / 10)]),
                        np.array([int(s) % 12])))
        return out

    def read_yolo(self, split_dir, data_yaml):
        """OpenCV's frame when transposes (EXIF applied), PIL's raw frame otherwise; the labels scaled to it."""
        from PIL import Image, ImageOps
        d = pathlib.Path(split_dir)
        out = []
        for f in sorted(os.listdir(str(d / "images"))):
            p = d / "images" / f
            with Image.open(p) as im0:
                im = ImageOps.exif_transpose(im0) if self.transposes else im0.copy()
                a = np.asarray(im.convert("RGB"))
            h, w = a.shape[:2]
            labs = D.label_boxes_px(d / "labels" / (os.path.splitext(f)[0] + ".txt"), w, h)
            out.append((str(p), a, np.asarray([x[1:] for x in labs], dtype=np.float64).reshape(-1, 4),
                        np.asarray([x[0] for x in labs])))
        return out


def plant_pretrain():
    """A small file standing for rf-detr-large-2026.pth (its size and MD5 patched for the test)."""
    p = D.pretrain_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"rf-detr-large-2026 stand-in " * 64)
    b = p.read_bytes()
    return len(b), hashlib.md5(b).hexdigest()


# ================================================================== UNITS
def u_constants():
    print("D6's constants (pre-registered)")
    r = D.RECIPE
    check("the recipe: 50 epochs at 704, batch 2 x 8 (16), lr 1e-4, lr_encoder 1.5e-4, weight decay 1e-4, EMA 0.993 / "
          "100, no early stopping, run_test false, checkpoint_interval 1000 > 50, eval every epoch, 4 workers, no "
          "tensorboard; rfdetr 1.6.5.post0, rf-detr-large-2026.pth 135,954,129 bytes",
          D.EPOCHS == 50 and D.RESOLUTION == 704 and r["batch_size"] == 2 and r["grad_accum_steps"] == 8
          and r["lr"] == 1e-4 and r["lr_encoder"] == 1.5e-4 and r["weight_decay"] == 1e-4 and r["use_ema"] is True
          and r["ema_decay"] == 0.993 and r["ema_tau"] == 100 and r["early_stopping"] is False
          and r["run_test"] is False and r["checkpoint_interval"] == 1000 > D.EPOCHS and r["eval_interval"] == 1
          and r["num_workers"] == 4 and r["tensorboard"] is False and D.RFDETR_VERSION == "1.6.5.post0"
          and D.PRETRAIN == "rf-detr-large-2026.pth" and D.PRETRAIN_BYTES == 135954129)
    check("the walltime rule: 36 h, seeds only if epoch x 50 + 0.5 h <= 28.8 h (2,030 s per epoch passes, 2,100 s "
          "does not); valid = round(0.02 x 6,811) = 136; threshold 0.001, the top 300, background index 13",
          D.WALLTIME_H == 36.0 and abs(D.walltime_cap_h() - 28.8) < 1e-12 and D.projection_h(2030) <= 28.8
          and D.projection_h(2100) > 28.8 and D.valid_count(6811) == 136 and D.valid_count(22) == 1
          and D.THRESHOLD == 0.001 and D.NUM_SELECT == 300 and D.BACKGROUND_ID == 13 == C.NC)
    check("EXIF_STAGING is unset until the read-only rehearsal records symlink or transpose",
          D.EXIF_STAGING is None and D.EXIF_MODES == ("symlink", "transpose") and D.ROTATED_TAGS == (3, 6, 8))
    check("the six E3-frozen sha256s of V3F.1", F.FROZEN_SHA256 == ST.FROZEN and {
        k: C.sha256_file(pathlib.Path(v).resolve()) for k, v in TS.CODE_FILES.items()} == ST.FROZEN)


def u_class_map_converter():
    print("the class map (0-12 round trip) and the converter")
    import yaml
    (W.TMP / "cmap").mkdir(parents=True, exist_ok=True)
    yml = D.write_data_yaml(W.TMP / "cmap" / "data.yaml", W.TMP / "cmap")
    names = yaml.safe_load(yml.read_text())["names"]
    cmap = D.class_map(names)
    lab = W.TMP / "cmap" / "l.txt"
    C.write_yolo(lab, [(c, 0.5, 0.5, 0.1, 0.1) for c in range(13)])
    back = [C.CLASS_NAMES[int(cmap[c])] for c, *_ in C.read_yolo(lab)]
    check("data.yaml's names -> INC ids is the identity; a label of INC id c reads back as INC class c (13 classes)",
          list(cmap) == list(range(13)) and back == list(C.CLASS_NAMES) and yaml.safe_load(yml.read_text())["nc"] == 13
          and "test" not in yaml.safe_load(yml.read_text()))
    out = {"permuted names": refused(D.class_map, list(reversed(C.CLASS_NAMES))),
           "twelve names": refused(D.class_map, list(C.CLASS_NAMES)[:12]),
           "COCO-style names": refused(D.class_map, ["background"] + list(C.CLASS_NAMES))}
    check("class maps refused: %s" % ", ".join(out), all(e is not None for e in out.values()))
    xy = np.array([[1, 2, 30, 40], [5, 5, 9, 9], [0, 0, 10, 10], [3, 3, 4, 4.0]])
    rows, c = D.convert(xy, [0.9, 0.5, 0.2, 0.01], [0, 5, 13, 12])
    check("planted predictions: background (13) dropped and counted, classes as INC ids, scores float32, boxes as "
          "given (original pixels)", c["background"] == 1 and list(rows["cls"]) == [0, 5, 12]
          and rows["conf"].dtype == np.float32 and np.array_equal(rows["xyxy"], xy[[0, 1, 3]]))
    bad = {"class 14": ([[1, 1, 2, 2]], [0.5], [14]), "class -1": ([[1, 1, 2, 2]], [0.5], [-1]),
           "class 2.5": ([[1, 1, 2, 2]], [0.5], [2.5]), "a NaN box": ([[np.nan, 1, 2, 2]], [0.5], [1]),
           "a score of 1.5": ([[1, 1, 2, 2]], [1.5], [1]), "two boxes, one score": ([[1, 1, 2, 2]] * 2, [0.5], [1, 1])}
    got = {k: refused(D.convert, *v) for k, v in bad.items()}
    check("the converter refuses: %s" % ", ".join(got), all(e is not None for e in got.values()),
          {k: str(v)[:60] for k, v in got.items()})


def tag6_row():
    """A tag-6 training image (raw 60 x 40, seen 40 x 60) with its label in the transposed frame."""
    p = FT.tagged_jpeg(W.TMP / "src" / "tag6" / "tsw23x_t6.jpg", 6, size=(60, 40), seed=7)
    return W.row_for("tsw23x_t6", p, [(3, 0.5, 0.25, 0.4, 0.3)], "3seasonweeddet10/data2023")


def u_staging():
    print("staging (no test/ folder, no holdout image, valid in train, the EXIF branch)")
    import yaml
    rows = list(WORLD["rows"]["base_v2"]) + [tag6_row()]
    WORLD["stage_rows"] = rows
    e0 = refused(D.stage, rows=rows, backend=FakeBackend())
    dev = WORLD["rows"]["dev"][0]
    hold = D.holdout_index()
    test_row = WORLD["rows"]["test"][0]
    moved_dev = dict(dev, key="renamed_dev", image=str(W.TMP / "src" / "elsewhere.jpg"))
    dev_key = dict(rows[0], key=dev["key"])
    check("the holdout index (dev, test, ImageWeeds, ood22, ood23; the probe when it exists) finds a dev row, a test "
          "row, a renamed copy of a dev image (by its sha256), a base image under a dev key (by key), and no base row",
          D.holdout_hits(rows + [dev, test_row, moved_dev, dev_key], hold) == [dev["key"], test_row["key"],
                                                                              "renamed_dev", dev["key"]]
          and D.holdout_hits(rows, hold) == [] and set(hold["counts"]) == {"dev", "test", "imageweeds", "ood22",
                                                                            "ood23", "probe"}, hold["counts"])
    e1 = refused(D.stage, exif="symlink", rows=rows + [dev], backend=FakeBackend())
    e2 = refused(D.stage, exif="symlink", rows=rows, backend=FakeBackend(transposes=False))
    left = [p.name for p in D.d6_dir().iterdir()] if D.d6_dir().exists() else []
    check("refused: without the rehearsal's outcome, a planted dev row, a reader that does not turn the tag-6 image "
          "(symlink branch); nothing left behind", e0 is not None and "rehearsal" in str(e0) and e1 is not None
          and "holdout" in str(e1) and e2 is not None and "frame" in str(e2) and not left, (e0, e1, e2, left))
    with moved(D.d6_dir()):
        st = D.stage(exif="transpose", rows=rows, backend=FakeBackend())
        t = D.stage_dir() / "transposed" / "tsw23x_t6.jpg"
        lk = D.stage_dir() / "train" / "images" / "tsw23x_t6.jpg"
        from PIL import Image
        with Image.open(t) as im:
            size, tag = im.size, im.getexif().get(0x0112)
        check("the transpose branch: the tag-6 image written once, transposed (40 x 60, no tag), JPEG q100 4:4:4, linked "
              "(relative) from train/; the others symlinked to their files", size == (40, 60) and tag is None
              and lk.is_symlink() and not os.path.isabs(os.readlink(str(lk))) and os.path.realpath(str(lk))
              == os.path.realpath(str(t)) and st["record"]["exif"]["transposed"] == {"tsw23x_t6": W.sha(t)}
              and st["record"]["exif"]["mode"] == "transpose", (size, tag))
    st = D.stage(exif="symlink", rows=rows, backend=FakeBackend())
    rec = st["record"]
    tr = sorted(os.path.splitext(f)[0] for f in os.listdir(str(D.stage_dir() / "train" / "images")))
    va = sorted(os.path.splitext(f)[0] for f in os.listdir(str(D.stage_dir() / "valid" / "images")))
    devk = {r["key"] for r in WORLD["rows"]["dev"]}
    tests = [dp for dp, dns, _ in os.walk(str(D.d6_dir())) for x in dns if x.lower() == "test"]
    by_rank = sorted(rows, key=lambda r: (C.stable_int("inc2/v3fam/d6/valid/%s" % r["key"]), r["key"]))
    labs_ok = all((D.stage_dir() / "train" / "labels" / ("%s.txt" % r["key"])).read_bytes()
                  == pathlib.Path(r["label"]).read_bytes() for r in rows)
    check("staged once (symlink branch, the reader checked on the tag-6 image): train/ is every base row (%d), valid/ "
          "the %d smallest stable_int keys, a subset of train and never dev; no test/ folder; labels their verified "
          "bytes; data.yaml without a test split" % (len(tr), len(va)), tr == sorted(r["key"] for r in rows)
          and va == rec["valid"]["keys"] and set(va) <= set(tr) and not (set(va) | set(tr)) & devk
          and len(va) == D.valid_count(len(rows)) and not tests and labs_ok and rec["test_folder"] is False
          and rec["exif"]["reader_check"]["failed"] == [] and "test" not in yaml.safe_load(
              (D.stage_dir() / "data.yaml").read_text())
          and va == D.valid_keys(rows) == sorted(r["key"] for r in by_rank[:D.valid_count(len(rows))]), rec["valid"])
    check("  stage() again only verifies (stage.json is written once)", D.stage()["sha256"] == st["sha256"])
    (D.stage_dir() / "train" / "test").mkdir()
    e3 = refused(D.verify_stage)
    (D.stage_dir() / "train" / "test").rmdir()
    planted = D.stage_dir() / "valid" / "images" / (dev["key"] + ".jpg")
    os.symlink(dev["image"], str(planted))
    e4 = refused(D.verify_stage)
    planted.unlink()
    check("verification refuses a planted test/ folder and a planted link to a dev image; passes again after",
          e3 is not None and "test" in str(e3) and e4 is not None and D.verify_stage()["sha256"] == st["sha256"],
          (e3, e4))


def u_attempt():
    print("an attempt's clean-up and the trainer's facts")
    att = D.Attempt(W.TMP / "att")
    for n in ("last.ckpt", "checkpoint_best_ema.pth", "metrics.csv"):
        (att.dir / n).write_bytes(b"x")
    old = att.dir / "checkpoint_best_regular.pth"
    old.write_bytes(b"y")
    os.utime(str(old), (time.time() - 3600, time.time() - 3600))
    link = att.dir / "checkpoint_best_total.pth"
    os.symlink(str(W.TMP / "elsewhere.pth"), str(link))
    out = {"an old file (not this attempt's)": refused(att.remove, old),
           "a link": refused(att.remove, link),
           "another name": refused(att.remove, att.dir / "metrics.csv"),
           "a file outside the attempt": refused(att.remove, W.TMP / "elsewhere.pth")}
    att.remove(att.dir / "last.ckpt")
    att.remove(att.dir / "checkpoint_best_ema.pth")
    check("removed: its own last.ckpt and checkpoint_best_ema.pth; refused: %s" % ", ".join(out),
          all(e is not None for e in out.values()) and not (att.dir / "last.ckpt").exists()
          and old.exists() and (att.dir / "metrics.csv").exists() and len(att.removed) == 2,
          {k: str(v)[:60] for k, v in out.items()})
    att2 = D.Attempt(W.TMP / "att")
    check("  each attempt is a fresh directory (attempt_1, attempt_2)", att2.dir.name == "attempt_2"
          and att.dir.name == "attempt_1")
    base = FakeBackend().train(W.TMP, att2.dir, 0, 3, None)
    ok = D._check_trained(base, 3, att2.dir)
    bad = {"run_test true": dict(base, callbacks=dict(base["callbacks"], run_test=True)),
           "an archive interval of 3": dict(base, callbacks=dict(base["callbacks"], archive_every_n_epochs=3)),
           "an epoch short": dict(base, epoch=2), "interrupted": dict(base, interrupted=True),
           "other classes": dict(base, class_names=list(C.CLASS_NAMES)[:12])}
    got = {k: refused(D._check_trained, v, 3, att2.dir) for k, v in bad.items()}
    (att2.dir / "checkpoint_2.ckpt").write_bytes(b"a")
    got["an archive checkpoint written"] = refused(D._check_trained, base, 3, att2.dir)
    check("the trainer's facts pass (%s); refused: %s" % (ok, ", ".join(got)), ok["run_test"] is False
          and all(e is not None for e in got.values()), {k: str(v)[:60] for k, v in got.items()})
    e = refused(D.rehearse_exif, C.INC_DIR / "rehearsal", backend=FakeBackend(), rows=WORLD["stage_rows"])
    doc = D.rehearse_exif(W.TMP / "rehearsal", backend=FakeBackend(), rows=WORLD["stage_rows"])
    doc2 = D.rehearse_exif(W.TMP / "rehearsal2", backend=FakeBackend(transposes=False), rows=WORLD["stage_rows"])
    check("the rehearsal: refused under INC_DIR; on the tagged images a turning reader suggests symlink, a raw one "
          "transpose; written outside INC_DIR only", e is not None and doc["tagged"] == 1
          and doc["suggests"] == "symlink" and doc2["suggests"] == "transpose"
          and (W.TMP / "rehearsal" / "d6_exif_rehearsal.json").is_file() and not (C.INC_DIR / "rehearsal").exists())


def u_smoke_train():
    print("smoke and train with the fake backend")
    size, md5 = plant_pretrain()
    with patched(D, "PRETRAIN_BYTES", size), patched(D, "PRETRAIN_MD5", md5):
        slow = D.smoke(backend=FakeBackend(fit_seconds=2100.0))
        e = refused(D.train, 0, backend=FakeBackend())
        check("the walltime rule: one epoch of 2,100 s projects %.2f h > 28.8 h: the smoke records 'not run "
              "(walltime)' and train refuses" % slow["projection_h"], slow["status"] == D.NOT_RUN_WALLTIME
              and not slow["walltime_ok"] and e is not None and "walltime" in str(e), e)
        shutil.rmtree(str(D.smoke_dir()))
        fb = FakeBackend(fit_seconds=900.0)
        sm = D.smoke(backend=fb)
        att = sm["attempt"]
        kept = sorted(os.listdir(str(D.smoke_dir() / att["attempt"])))
        check("the smoke: one epoch on the staging, projection %.2f h within the rule; the export equals the in-process "
              "EMA weights and predicts as the in-process model on the training images; its last.ckpt and best files "
              "removed, metrics kept; its export removed after the checks" % sm["projection_h"],
              sm["status"] == "complete" and sm["walltime_ok"] and fb.trained[-1][3] == 1
              and att["export"]["equals_inprocess_ema"] and att["export_predictions"]["ok"]
              and att["export_predictions"]["images"] == min(D.SMOKE_PRED_IMAGES, len(WORLD["stage_rows"]))
              and sorted(r["file"] for r in att["cleanup"]["removed"]) == sorted(D.OWN_FILES)
              and kept == ["metrics.csv"] and not (D.smoke_dir() / D.FINAL).exists()
              and att["trainer"]["run_test"] is False, att)
        e = refused(D.smoke, backend=fb)
        check("  the smoke runs once", e is not None)
        t0 = D.train(0, backend=fb)
        st = fb.load_state(D.final_path(0))
        check("seed 0: 50 epochs; final_ema.pth in {model, args, epoch} equals the EMA weights (seed 0), not the planted "
              "best file's other weights; train.json written once with its sha256; the attempt cleaned",
              fb.trained[-1][2:] == (0, 50) and set(st) == {"model", "args", "epoch"}
              and np.array_equal(st["model"]["w"], np.full(4, 0.5)) and st["epoch"] == 50
              and t0["final"]["sha256"] == W.sha(D.final_path(0))
              and sorted(os.listdir(str(D.seed_dir(0) / t0["attempt"]["attempt"]))) == ["metrics.csv"])
        out = {"seed 0 again": refused(D.train, 0, backend=fb), "seed 3": refused(D.train, 3, backend=fb),
               "an archive checkpoint": refused(D.train, 1, backend=FakeBackend(archive=True)),
               "run_test true": refused(D.train, 1, backend=FakeBackend(run_test=True)),
               "an epoch short": refused(D.train, 1, backend=FakeBackend(short=True))}
        atts = sorted(p.name for p in D.seed_dir(1).iterdir() if p.name.startswith("attempt_"))
        failed = [json.loads((D.seed_dir(1) / a / "attempt.json").read_text()) for a in atts]
        check("refused: %s; the failed attempts exported nothing, recorded attempt.json and removed what they wrote"
              % ", ".join(out), all(e is not None for e in out.values()) and not D.final_path(1).exists()
              and len(failed) == 3 and all(f.get("failed") and not f.get("export") and len(
                  f["cleanup"]["removed"]) == 4 for f in failed), {k: str(v)[:60] for k, v in out.items()})
        try:
            D.train(1, backend=FakeBackend(term=True))
            term = None
        except SystemExit as x:
            term = x.code
        last = sorted(p.name for p in D.seed_dir(1).iterdir() if p.name.startswith("attempt_"))[-1]
        lt = json.loads((D.seed_dir(1) / last / "attempt.json").read_text())
        check("SIGTERM during training: SystemExit(143), the attempt cleans what it wrote, nothing exported",
              term == 143 and not D.final_path(1).exists() and len(lt["cleanup"]["removed"]) == 4
              and "SystemExit" in lt["failed"], (term, lt.get("failed")))
        D.train(1, backend=fb)
        D.train(2, backend=fb)
    check("seeds 0, 1, 2 trained: their final weights differ by seed and hash as train.json records",
          all(D.final_record(s)["sha256"] == W.sha(D.final_path(s)) for s in F.SEEDS)
          and len({D.final_record(s)["sha256"] for s in F.SEEDS}) == 3)


UNITS = ("u_constants", "u_class_map_converter", "u_staging", "u_attempt", "u_smoke_train")


# ================================================================== WORLD
def w_setup():
    print("the world: E3's records, the zoo root, the reproductions D6's scores rest on")
    FT.WORLD["rows"] = WORLD["rows"]
    FT.w_setup()
    ST.build_ooddev_root()
    F.equivalence_dev_identity(**KW)
    F.equivalence_injection(**KW)
    F.equivalence_identity("imageweeds", "D6", **KW)
    F.equivalence_identity("ooddev_v1", "D6", **KW)
    check("the dev identity, injection and identity reproductions on ImageWeeds and ooddev_v1 passed",
          all(json.loads((F.equiv_dir() / n).read_text())["status"] == "passed" for n in (
              "dev_identity.json", "injection.json", "identity_imageweeds.json", "identity_ooddev_v1.json")))


def w_score():
    print("score_d6: predictions planted as b_v2_m640 s0's own rows, through the converter and inject_rows")
    from PIL import Image, ImageOps
    src = SYS.e3_source("M", 0)
    ident = F._family_pass(src["weights"], "dev", {"mode": "identity", "nms": "locked"}, "E3RS", **KW)
    inj = F.inject_from_rows(ident["rows"])
    planted = {}
    n_bg = 0
    for r in SYS.exam_rows("dev"):
        with Image.open(r["image"]) as im0:
            pk = pixel_key(ImageOps.exif_transpose(im0))
        v = inj[r["key"]]
        xy = np.concatenate([v["xyxy"], [[1.0, 1.0, 5.0, 5.0]]])
        cf = np.concatenate([v["conf"].astype(np.float64), [0.3]])
        cl = np.concatenate([v["cls"], [D.BACKGROUND_ID]])
        planted[pk] = (xy, cf, cl)
        n_bg += 1
    fb = FakeBackend(planted=planted)
    t0 = time.time()
    rec = D.score_d6(backend=fb, **KW)
    docs = [json.loads((F.arm_dir("D6", s) / "dev.json").read_text()) for s in F.SEEDS]
    r0 = ident["result"]
    check("every seed's dev score equals b_v2_m640 s0's identity score exactly (the converter, the class map and the "
          "injection end to end); %d planted background rows dropped per seed (%.0fs)" % (n_bg, time.time() - t0),
          all(d["species_map50_95"] == r0["species_map50_95"] and d["per_class"] == r0["per_class"]
              and d["settings"]["mode"] == "inject_rows" and d["detr"]["converter"]["background_dropped"] == n_bg
              and d["scaffold"]["exp"] == "b_v2_m640" for d in docs), [d["species_map50_95"] for d in docs])
    rep = json.loads(SYS.reported_path("D6").read_text())
    cap = json.loads(SYS.score_record_path("D6").read_text())
    check("reported: ImageWeeds and ooddev_v1 (the zoo's root) for each seed; the platform's record dev only, with the "
          "three final weights' sha256s", sorted((p["seed"], p["exam"], p["status"]) for p in rep["passes"])
          == sorted((s, e, "written") for s in F.SEEDS for e in ("imageweeds", "ooddev_v1"))
          and all(f["exam"] == "dev" for f in cap["dev"]) and len(cap["dev"]) == 3
          and [x["sha256"] for x in cap["finals"]] == [W.sha(D.final_path(s)) for s in F.SEEDS], rep["passes"])
    rs = SYS.reference_sources("D6")
    check("the reference read's inputs: the three final EMA files by sha256 on b_v2_m640 s0's scaffold",
          rs["mode"] == "inject_rows" and [x["seed"] for x in rs["seeds"]] == [0, 1, 2]
          and rs["scaffold"]["exp"] == "b_v2_m640", rs)
    rec2 = D.score_d6(backend=fb, **KW)
    check("a rerun keeps every file", all(f["status"] == "kept" for f in rec2["dev"]))
    del rec


def w_cli():
    print("the CLI")
    env = dict(os.environ)
    pkg = str(pathlib.Path(F.__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def cli(*a):
        p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_detr"] + list(a),
                           env=env, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    rc1, _o, e1 = cli("train")
    rc2, _o, e2 = cli("rehearse-exif")
    rc3, out3, _e = cli("sources")
    check("usage errors exit 2 with [inc2.v3fam_detr] ERROR; sources prints the read's inputs", rc1 == 2
          and "--seed" in e1 and rc2 == 2 and "--out" in e2 and rc3 == 0 and len(json.loads(out3)["seeds"]) == 3,
          (rc1, e1[-200:], rc2, e2[-200:], rc3))


WORLD_TESTS = ("w_setup", "w_score", "w_cli")


def main():
    t0 = time.time()
    only = os.environ.get("V3FAM_ONLY")
    try:
        WORLD["rows"] = W.build_world()
        for name in UNITS + (() if only == "units" else WORLD_TESTS):
            globals()[name]()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
