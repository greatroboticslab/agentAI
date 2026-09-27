#!/usr/bin/env python3
"""The INC scorer is the only producer of metrics, so what it refuses and what it
reports are pinned here (docs/INCREMENTAL_PROTOCOL.md, "Scorer").

Refusals: an exam manifest or a scorer.py that no longer hashes to LOCK.json; an
exam dir with a key missing or extra; an exam label or image that differs from
its manifest, wherever the image sits in the exam (every image is hashed, not a
sample); weights whose classes are not the INC class space; a LoRA last.pt that
still holds its adapters (the merged checkpoint is what is scored). A refusal,
or an error, leaves no score.json at the output path, not even one an earlier
run wrote there.

Protocol settings: without INC_SCORER_TESTING=1 the scorer refuses anything but
a protocol score (LOCK.json checked, imgsz 640, batch 32, fp16 on CUDA, the
pinned Ultralytics). This test needs imgsz 64 on a CPU, so it runs with the
variable set: every score it takes must say production=false, list its
deviations and carry scorer_sha256 "TEST-" + sha256, a stamp no protocol score
shares. With the deviations taken away the stamp is the bare sha256.

Metrics: the 12-class numbers equal a plain Ultralytics model.val on the exam's
own data.yaml; the class-agnostic score is unchanged when only the class names
of the GT change, and is >= the 12-class score when the names are wrong; a box
the validator emits under two classes counts once, with its most confident
class, in the agnostic score and in the per-image bit; the per-image bit is
right on hand-made cases, and through a real validation pass it equals the bits
built into an exam where some images are fully right and each of the others
misses, misnames or adds one box. Scoring leaves the exam dir, the source photos
and the temp dir as they were, including a JPEG with bytes after its EOI marker,
which plain Ultralytics val rewrites in place.

The detector is a YOLO11n built from yolo11n.yaml and trained one epoch on
coloured shapes (nothing is downloaded). One epoch teaches it nothing, so its
Detect head's class biases are raised to make it confident, one class per
stride, and the exam labels are its own predictions: the 12-class score is then
far from zero, and giving the same boxes other class ids makes an exam where
localisation is right and every name is wrong.

Run:  python3 tests/test_inc_scorer.py
"""
import contextlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_inc_scorer_"))
# common reads INC_DIR at import: this test never sees the machine's real one.
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["INC_SCORER_TESTING"] = "1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402

FAILURES = []
IMGSZ, BATCH = 64, 8
LEVEL_LOGITS = ({0: 4.0}, {5: 4.0, 12: 3.0}, {})  # class logits of the confident head at stride 8, 16, 32
CONFUSE = {0: 5, 5: 0, 12: 0}       # each GT box renamed to a class the model never puts on a box its size
COLOURS = {0: (220, 40, 40), 5: (40, 190, 40), 12: (40, 70, 220)}


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def sha(path):
    return C.sha256_file(path)


@contextlib.contextmanager
def production_mode():
    saved = os.environ.pop(S.TEST_ENV, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[S.TEST_ENV] = saved


# ---------------------------------------------------------------- fixtures
def make_images(out_dir, prefix, n, seed):
    """n 96x96 JPEGs with 1-2 coloured shapes; returns [(path, yolo boxes)]."""
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(n):
        im = Image.new("RGB", (96, 96), (rng.randint(90, 150),) * 3)
        draw = ImageDraw.Draw(im)
        boxes = []
        for c in rng.sample(sorted(COLOURS), rng.randint(1, 2)):
            w, h = rng.randint(22, 40), rng.randint(22, 40)
            x, y = rng.randint(0, 95 - w), rng.randint(0, 95 - h)
            (draw.ellipse if c == 12 else draw.rectangle)([x, y, x + w, y + h], fill=COLOURS[c])
            boxes.append((c, (x + w / 2) / 96, (y + h / 2) / 96, w / 96, h / 96))
        path = out_dir / ("%s_%03d.jpg" % (prefix, i))
        im.save(path, quality=92)
        made.append((path, boxes))
    return made


def rows_for(images, label_dir, labels, source):
    rows = []
    for path, _ in images:
        key = path.stem
        lab = label_dir / (key + ".txt")
        C.write_yolo(lab, labels[key])
        rows.append({"image": str(path), "label": str(lab), "sha256": sha(path),
                     "label_sha256": sha(lab), "source": source, "session": "", "key": key})
    return rows


def make_exam(name, rows):
    msha = C.write_manifest(C.manifest_path(name), rows)
    C.materialise(rows, C.EXAMS_DIR / name)
    return msha


def write_lock(manifests, scorer_sha=None):
    C.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.LOCK_PATH, "w") as fh:
        json.dump({"manifests": manifests,
                   "scorer_sha256": scorer_sha or sha(pathlib.Path(S.__file__).resolve())}, fh)


def _net(ck):
    return ck["ema"] if ck.get("ema") is not None else ck["model"]


def confident_copy(src, dst, names=None):
    """The trained checkpoint made confident: at strides 8 and 16 every anchor
    predicts the classes in LEVEL_LOGITS with a box two strides wide centred on
    it; stride 32 predicts nothing."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    net = _net(ck)
    det = net.model[-1]
    for level, logits in enumerate(LEVEL_LOGITS):
        cls_conv, box_conv = det.cv3[level][-1], det.cv2[level][-1]
        cls_conv.bias.data[:] = -12.0
        for c, b in logits.items():
            cls_conv.bias.data[c] = b
        assert box_conv.out_channels == 4 * det.reg_max, box_conv
        box_conv.weight.data.zero_()
        box_conv.bias.data.zero_()
        box_conv.bias.data[1::det.reg_max] = 10.0   # DFL bin 1 on all four sides: one stride
    if names is not None:
        net.names = names
    torch.save(ck, dst)
    return dst


def tree_state(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        for n in sorted(dirnames + filenames):
            p = os.path.join(dirpath, n)
            st = os.lstat(p)
            out.append((os.path.relpath(p, root), st.st_size, st.st_mtime_ns))
    return sorted(out)


STALE = '{"stale": "an earlier run left this score here"}\n'


def refused(weights, exam, **kw):
    """The refusal message, or None if the scorer scored. An earlier run's
    score.json is planted at the output path first; a refusal that leaves it
    (or writes one) fails the test."""
    out = TMP / "out" / "refused.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(STALE)
    for k, v in (("imgsz", IMGSZ), ("batch", BATCH), ("device", "cpu")):
        kw.setdefault(k, v)
    try:
        S.score(weights, exam, out, **kw)
    except S.ScorerRefused as e:
        if out.exists():
            out.unlink()
            return "REFUSED BUT LEFT %s" % out
        return str(e)
    out.unlink()
    return None


def wired_detector(path):
    """A detector whose every prediction is known: layer 0 averages the input
    over 8x8 cells, and at stride 8 Detect's class logit is a linear function of
    the cell's mean colour. A red cell is Waterhemp (0), also emitted as
    OtherPlant (12) a little less confidently, the multi-label NMS twin the
    scorer must count once; a green cell is Ragweed (5); a grey cell, like the
    letterbox padding, predicts nothing above conf 0.001. Every box is the cell
    centre +- one stride. Strides 16 and 32 predict nothing. It is built only
    from torch and Ultralytics modules, so plain YOLO() loads it.

    Unlike the confident fixture it predicts nothing in the half-stride border
    that Ultralytics' rect validation pads every image with, so an image can be
    entirely right."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    cfg = {"nc": C.NC, "backbone": [[-1, 1, "nn.AvgPool2d", [8, 8]], [-1, 1, "nn.AvgPool2d", [2, 2]],
                                    [-1, 1, "nn.AvgPool2d", [2, 2]]],
           "head": [[[0, 1, 2], 1, "Detect", [C.NC]]]}
    net = DetectionModel(cfg, ch=3, nc=C.NC, verbose=False)
    det = net.model[-1]
    for level in range(3):
        branch, mixed = det.cv3[level], False
        # every Conv+BN of the class branch passes channels through at its centre
        # tap; the first dense one turns RGB into (R - G, G - R) on channels 0, 1
        for m in [m for m in branch.modules() if hasattr(m, "conv") and hasattr(m, "bn")]:
            w = torch.zeros_like(m.conv.weight)
            k = w.shape[-1] // 2
            if m.conv.groups > 1:
                w[:, 0, k, k] = 1.0
            elif not mixed:
                assert m.conv.in_channels == 3, m
                w[0, 0, k, k], w[0, 1, k, k], w[1, 1, k, k], w[1, 0, k, k] = 10.0, -10.0, 10.0, -10.0
                mixed = True
            else:
                for o in range(min(m.conv.out_channels, m.conv.in_channels)):
                    w[o, o, k, k] = 1.0
            m.conv.weight.data.copy_(w)
            m.bn.weight.data.fill_(1.0)
            m.bn.bias.data.zero_()
            m.bn.running_mean.zero_()
            m.bn.running_var.fill_(1.0 - m.bn.eps)
        assert mixed, branch
        last = branch[-1]
        last.weight.data.zero_()
        last.bias.data.fill_(-12.0)
        if level == 0:
            for c, ch, b in ((0, 0, -9.0), (12, 0, -11.0), (5, 1, -9.0)):
                last.weight.data[c, ch] = 3.0
                last.bias.data[c] = b
        box = det.cv2[level][-1]
        box.weight.data.zero_()
        box.bias.data.zero_()
        box.bias.data[1::det.reg_max] = 10.0        # DFL bin 1 on all four sides: one stride
    net.names = {i: n for i, n in enumerate(C.CLASS_NAMES)}
    torch.save({"model": net, "train_args": {"task": "detect"}}, path)
    return path


# The bits exam for the wired detector: 96x96 images on a 12-px grid, which the
# letterbox maps onto its 8x8 cells exactly. A square is (colour, cell x, cell
# y); the GT is the squares' cells, unless it departs from them.
SQUARES = {"red": ((220, 40, 40), 0), "green": ((40, 190, 40), 5)}
BITS_SPECS = [
    ([("red", 2, 2)], None),
    ([("red", 2, 2), ("green", 4, 5)], None),
    ([("red", 3, 3), ("green", 5, 2)], "drop"),       # a predicted box has no GT
    ([("green", 1, 6)], "rename"),                    # found, but the GT names another class
    ([], None),                                       # nothing there, nothing predicted
    ([("red", 6, 6)], "shift"),                       # GT one cell off: IoU 1/3
    ([("red", 1, 1), ("red", 2, 1)], None),           # neighbours, boxes overlapping
    ([("green", 3, 4)], "extra"),                     # a GT box the detector cannot see
    ([("red", 5, 5)], None),
    ([], "extra"),
]


def cell_box(cls, qx, qy):
    return (cls, (12 * qx + 6) / 96, (12 * qy + 6) / 96, 24 / 96, 24 / 96)


def make_bits_images(out_dir):
    """[(path, GT boxes)] and the expected bit string of BITS_SPECS."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    made, want = [], ""
    for i, (squares, departs) in enumerate(BITS_SPECS):
        im = Image.new("RGB", (96, 96), (96 + 5 * i,) * 3)
        draw = ImageDraw.Draw(im)
        gt = []
        for colour, qx, qy in squares:
            draw.rectangle([12 * qx, 12 * qy, 12 * qx + 11, 12 * qy + 11], fill=SQUARES[colour][0])
            gt.append(cell_box(SQUARES[colour][1], qx, qy))
        if departs == "drop":
            gt = gt[:-1]
        elif departs == "rename":
            gt[0] = ((gt[0][0] + 1) % C.NC,) + gt[0][1:]
        elif departs == "shift":
            gt[0] = gt[0][:1] + (gt[0][1] + 12 / 96,) + gt[0][2:]
        elif departs == "extra":
            gt.append(cell_box(0, 6, 1))
        path = out_dir / ("b_%02d.png" % i)
        im.save(path)
        made.append((path, gt))
        want += "0" if departs else "1"
    return made, want


def one_per_box_rows(lines):
    """Test-side reduction of saved predictions ('c x y w h conf') to one row
    per box, the most confident class of each."""
    best = {}
    for ln in lines:
        c, x, y, w, h, conf = (float(t) for t in ln.split())
        if (x, y, w, h) not in best or conf > best[(x, y, w, h)][0]:
            best[(x, y, w, h)] = (conf, int(c))
    return [(c, x, y, w, h, conf) for (x, y, w, h), (conf, c) in sorted(best.items())]


def cells(boxes):
    """(class, cell x, cell y) of each 24-px box centred on a 12-px cell."""
    return sorted((int(b[0]), round((b[1] * 96 - 6) / 12), round((b[2] * 96 - 6) / 12)) for b in boxes)


# ------------------------------------------------------------------- tests
def test_image_correct():
    ic = S.image_correct
    g = [[0, 0, 10, 10], [20, 20, 40, 40]]
    check("image_correct: every GT matched by its class, nothing extra -> 1",
          ic([1, 2], g, [2, 1], [[21, 21, 40, 40], [0, 0, 10, 11]], [0.9, 0.3]) == 1)
    check("image_correct: a box found with the wrong class -> 0",
          ic([1, 2], g, [1, 1], [[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.8]) == 0)
    check("image_correct: an extra confident prediction -> 0",
          ic([1], g[:1], [1, 1], [[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.5]) == 0)
    check("image_correct: predictions below conf 0.25 are ignored",
          ic([1], g[:1], [1, 3], [[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.2]) == 1)
    check("image_correct: a missed GT box -> 0",
          ic([1, 2], g, [1], [[0, 0, 10, 10]], [0.9]) == 0)
    check("image_correct: a duplicate on one GT box -> 0",
          ic([1], g[:1], [1, 1], [[0, 0, 10, 10], [0, 0, 10, 9]], [0.9, 0.8]) == 0)
    check("image_correct: IoU just under 0.5 -> 0, at 0.5 -> 1",
          ic([1], [[0, 0, 10, 10]], [1], [[0, 0, 10, 4.9]], [0.9]) == 0
          and ic([1], [[0, 0, 10, 10]], [1], [[0, 0, 10, 5]], [0.9]) == 1)
    check("image_correct: empty image with no confident prediction -> 1, with one -> 0",
          ic([], [], [4], [[0, 0, 5, 5]], [0.1]) == 1 and ic([], [], [4], [[0, 0, 5, 5]], [0.3]) == 0)
    # the most confident prediction claims the GT box it overlaps most, even when
    # that leaves the next prediction without a partner a global matching would find
    check("image_correct: greedy by confidence, as specified",
          ic([1, 1], [[0, 0, 10, 10], [5, 0, 15, 10]], [1, 1],
             [[3, 0, 13, 10], [6, 0, 16, 10]], [0.9, 0.8]) == 0)


def test_one_per_box():
    b = [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 11], [5, 5, 9, 9], [0, 0, 10, 10]]
    keep = S.one_per_box(b, [0.3, 0.9, 0.8, 0.1, 0.9])
    check("one_per_box: identical boxes keep their most confident row (first of equals); "
          "boxes that differ at all are kept", list(keep) == [1, 2, 3], list(keep))
    check("one_per_box: 0 and 1 predictions", list(S.one_per_box([], [])) == []
          and list(S.one_per_box([[0, 0, 1, 1]], [0.5])) == [0])


def test_validator_hook():
    """The hook on synthetic predictions: boxes right, names swapped; and a box
    that multi-label NMS emits under two classes."""
    import numpy as np
    import torch
    from ultralytics.utils.metrics import ap_per_class
    T = torch.tensor
    v = S.validator_class()(save_dir=TMP / "hook", args=dict(conf=0.001, iou=0.7, plots=False))
    v.inc_reset()
    gt_a = T([[10., 10., 30., 30.], [40., 40., 60., 60.]])
    gt_b = T([[5., 5., 50., 50.]])
    tp_a = v._process_batch({"bboxes": gt_a.clone(), "conf": T([0.9, 0.8]), "cls": T([5., 0.])},
                            {"cls": T([0., 5.]), "bboxes": gt_a, "im_file": "/x/images/a.jpg"})["tp"]
    tp_b = v._process_batch({"bboxes": gt_b.clone(), "conf": T([0.7]), "cls": T([12.])},
                            {"cls": T([12.]), "bboxes": gt_b, "im_file": "/x/images/b.jpg"})["tp"]
    ap12 = ap_per_class(np.concatenate([tp_a, tp_b]), np.array([0.9, 0.8, 0.7]),
                        np.array([5., 0., 12.]), np.array([0., 5., 12.]))[5].mean()
    res = v.inc_results()
    check("hook: class-confused boxes score 0 in 12-class AP for those classes",
          not tp_a.any() and tp_b.all(), (tp_a, tp_b))
    check("hook: agnostic mAP50-95 >= 12-class on the confusion fixture (%.3f >= %.3f)"
          % (res["agnostic_map50_95"], ap12),
          res["agnostic_map50_95"] >= ap12 and res["agnostic_map50_95"] > 0.99)
    check("hook: per-image bits keyed by image stem", res["bits"] == {"a": 0, "b": 1}, res["bits"])
    check("hook: collapsed GT count", res["n_gt"] == 3, res["n_gt"])
    check("collapsed_ap: no GT -> 0", S.collapsed_ap(np.zeros((2, 10), bool), [0.5, 0.4], 0) == (0.0, 0.0))
    check("collapsed_ap: no prediction at all -> 0, not a crash (a collapsed model is scored, then rejected)",
          S.collapsed_ap(np.zeros((0, 10), bool), [], 5) == (0.0, 0.0))

    # 20 images, one GT box each, predicted at exactly that box with the right
    # class (conf 0.95 down to 0.57); then the same box also as class 5, the way
    # multi-label NMS repeats a box for every class above conf 0.001
    def run(second):
        h = S.validator_class()(save_dir=TMP / "hook2", args=dict(conf=0.001, iou=0.7, plots=False))
        h.inc_reset()
        for i in range(20):
            box = T([[10. + i, 10., 40. + i, 40.]])
            primary = 0.95 - 0.02 * i
            if second is None:
                preds = {"bboxes": box.clone(), "conf": T([primary]), "cls": T([0.])}
            else:
                preds = {"bboxes": torch.cat([box, box]), "conf": T([primary, second]), "cls": T([0., 5.])}
            h._process_batch(preds, {"cls": T([0.]), "bboxes": box, "im_file": "/x/images/im%02d.jpg" % i})
        r = h.inc_results()
        bits = "".join(str(r["bits"][k]) for k in sorted(r["bits"]))
        return r["agnostic_map50_95"], r["agnostic_map50"], bits

    alone, low, high = run(None), run(0.002), run(0.30)
    top = run(0.60)
    check("hook: a second class on a well-placed box leaves the agnostic score as it was "
          "(%.4f; with it at 0.002 / 0.30 / 0.60: %.4f / %.4f / %.4f)" % (alone[0], low[0], high[0], top[0]),
          alone[:2] == low[:2] == high[:2] == top[:2], (alone, low, high, top))
    check("hook: a second, less confident class leaves every image correct",
          alone[2] == low[2] == high[2] == "1" * 20, (alone[2], low[2], high[2]))
    check("hook: a box whose most confident class is wrong makes its image incorrect",
          top[2] == "1" * 18 + "00", top[2])


def test_deviations():
    d = S.deviations
    check("deviations: the protocol's settings are a protocol score",
          d(True, 640, 32, True, S.PINNED_ULTRALYTICS) == [])
    each = [d(False, 640, 32, True, S.PINNED_ULTRALYTICS), d(True, 320, 32, True, S.PINNED_ULTRALYTICS),
            d(True, 640, 8, True, S.PINNED_ULTRALYTICS), d(True, 640, 32, False, S.PINNED_ULTRALYTICS, "cpu"),
            d(True, 640, 32, True, "8.4.22")]
    check("deviations: lock bypass, imgsz, batch, fp32 and another Ultralytics are each one",
          [len(x) for x in each] == [1] * 5 and "LOCK.json" in each[0][0] and "imgsz 320" in each[1][0]
          and "batch 8" in each[2][0] and "fp32 on device cpu" in each[3][0] and "8.4.22" in each[4][0], each)


def test_every_image_hashed():
    """100 images, more than any sample the scorer once took: an altered image
    is refused wherever it sits."""
    from PIL import Image
    src = TMP / "src" / "many"
    src.mkdir(parents=True)
    rows = []
    for i in range(100):
        p = src / ("m%03d.png" % i)
        Image.new("RGB", (8, 8), (i, (7 * i) % 256, 50)).save(p)
        lab = src / ("m%03d.txt" % i)
        C.write_yolo(lab, [(0, .5, .5, .5, .5)])
        rows.append({"image": str(p), "label": str(lab), "sha256": sha(p), "label_sha256": sha(lab),
                     "source": "synthetic", "session": "", "key": "m%03d" % i})
    make_exam("t_many", rows)
    rows = sorted(C.read_manifest(C.manifest_path("t_many")), key=lambda r: r["key"])
    files, labels = S.check_exam("t_many", rows)
    check("check_exam: an intact 100-image exam passes", len(files) == len(labels) == 100)
    missed = []
    for i in (0, 37, 64, 99):
        p = src / ("m%03d.png" % i)
        orig = p.read_bytes()
        p.write_bytes(orig + b"\0")
        try:
            S.check_exam("t_many", rows)
            missed.append(i)
        except S.ScorerRefused as e:
            if "m%03d" % i not in str(e) or "images that differ" not in str(e):
                missed.append((i, str(e)))
        finally:
            p.write_bytes(orig)
    check("check_exam: an altered image is refused at any position (0, 37, 64, 99 of 100)",
          not missed, missed)


def test_end_to_end():
    import torch
    from ultralytics import YOLO

    # --- a 13-class checkpoint and self-labelled exams ------------------------
    t0 = time.time()
    train = make_images(TMP / "src" / "train", "tr", 16, seed=0)
    train_rows = rows_for(train, TMP / "src" / "train_labels",
                          {p.stem: b for p, b in train}, "synthetic_train")
    train_yaml = C.materialise(train_rows, TMP / "train_ds")
    YOLO("yolo11n.yaml").train(data=str(train_yaml), epochs=1, imgsz=IMGSZ, batch=BATCH,
                               device="cpu", workers=0, plots=False, val=False, amp=False,
                               seed=0, deterministic=True, verbose=False,
                               project=str(TMP / "runs"), name="train", exist_ok=True)
    W = confident_copy(TMP / "runs" / "train" / "weights" / "last.pt", TMP / "confident.pt")
    print("  (trained + made confident in %.0fs)" % (time.time() - t0))

    exam_imgs = make_images(TMP / "src" / "exam", "ex", 20, seed=1)
    probe_rows = rows_for(exam_imgs, TMP / "src" / "probe_labels",
                          {p.stem: [] for p, _ in exam_imgs}, "synthetic_exam")
    probe_yaml = C.materialise(probe_rows, TMP / "probe")
    with warnings.catch_warnings():     # an exam without labels: numpy warns inside Ultralytics' AP
        warnings.simplefilter("ignore", RuntimeWarning)
        YOLO(str(W)).val(data=str(probe_yaml), imgsz=IMGSZ, batch=BATCH, conf=0.001, iou=0.7,
                         device="cpu", half=False, plots=False, verbose=False, save_txt=True,
                         save_conf=True, project=str(TMP / "runs"), name="probe", exist_ok=True)
    probe = {}
    for path, _ in exam_imgs:
        pred_file = TMP / "runs" / "probe" / "labels" / (path.stem + ".txt")
        probe[path.stem] = pred_file.read_text().splitlines() if pred_file.exists() else []
    # GT: every predicted box the image border did not clip (Ultralytics saves
    # clipped boxes): the 36 inner stride-8 boxes as class 0, and the 4 inner
    # stride-16 boxes, predicted as both 5 and 12, alternately as 5 and as 12
    self_labels = {}
    for path, _ in exam_imgs:
        by_cls = {}
        for ln in probe[path.stem]:
            c, x, y, w, h, _conf = (float(t) for t in ln.split())
            if min(x - w / 2, y - h / 2) > 1e-3 and max(x + w / 2, y + h / 2) < 1 - 1e-3:
                by_cls.setdefault(int(c), []).append((x, y, w, h))
        self_labels[path.stem] = [(0,) + b for b in by_cls.get(0, [])] + [
            ((5, 12)[j % 2],) + b for j, b in enumerate(sorted(by_cls.get(5, [])))]
    n_gt = sum(len(v) for v in self_labels.values())
    classes = {b[0] for v in self_labels.values() for b in v}
    check("fixture: the confident model labels its own exam (%d boxes, classes %s)"
          % (n_gt, sorted(classes)), n_gt == 20 * (36 + 4) and classes == {0, 5, 12})

    # The bits exam, scored with the wired detector. Independently of the
    # scorer, a plain val's saved predictions (one per box, conf >= 0.25) say
    # which images are right: those whose predicted cells and classes are
    # exactly their GT's.
    WIRED = wired_detector(TMP / "wired.pt")
    bits_imgs, want_bits = make_bits_images(TMP / "src" / "bits")
    bits_labels = {p.stem: gt for p, gt in bits_imgs}
    bits_rows = rows_for(bits_imgs, TMP / "src" / "bits_labels", bits_labels, "synthetic_bits")
    YOLO(str(WIRED)).val(data=str(C.materialise(bits_rows, TMP / "bits_plain")), imgsz=IMGSZ,
                         batch=BATCH, conf=0.001, iou=0.7, device="cpu", half=False, plots=False,
                         verbose=False, save_txt=True, save_conf=True, project=str(TMP / "runs"),
                         name="bits_plain", exist_ok=True)
    saved, raw = {}, {}
    for path, _ in bits_imgs:
        pred_file = TMP / "runs" / "bits_plain" / "labels" / (path.stem + ".txt")
        raw[path.stem] = pred_file.read_text().splitlines() if pred_file.exists() else []
        saved[path.stem] = [b for b in one_per_box_rows(raw[path.stem]) if b[5] >= S.CORRECT_CONF]
    saved_bits = "".join("1" if cells(saved[p.stem]) == cells(bits_labels[p.stem]) else "0"
                         for p, _ in bits_imgs)
    drawn = {p.stem: cells([cell_box(SQUARES[c][1], qx, qy) for c, qx, qy in spec[0]])
             for (p, _), spec in zip(bits_imgs, BITS_SPECS)}
    check("fixture: the wired detector predicts exactly the squares drawn, each red one "
          "also as OtherPlant", all(cells(saved[k]) == drawn[k] for k in saved)
          and all(sum(ln.startswith("12 ") for ln in raw[k]) == sum(c == 0 for c, _, _ in drawn[k])
                  for k in raw), {k: (cells(saved[k]), drawn[k]) for k in saved})
    check("fixture: the bits exam has right and wrong images (%s), and the saved predictions "
          "say the same" % want_bits, "0" in want_bits and "1" in want_bits and saved_bits == want_bits,
          saved_bits)

    self_rows = rows_for(exam_imgs, TMP / "src" / "self_labels", self_labels, "synthetic_exam")
    perm_rows = rows_for(exam_imgs, TMP / "src" / "perm_labels",
                         {k: [(CONFUSE[b[0]],) + b[1:] for b in v] for k, v in self_labels.items()},
                         "synthetic_exam")
    # a photo with bytes after its EOI marker, as some cameras write
    trail = TMP / "src" / "trail" / "ex_000.jpg"
    trail.parent.mkdir(parents=True)
    trail.write_bytes(exam_imgs[0][0].read_bytes() + b"\x00\x00")
    trail_rows = [dict(r) for r in self_rows]
    trail_rows[0].update(image=str(trail), sha256=sha(trail))
    manifests = {n: make_exam(n, r) for n, r in
                 (("t_self", self_rows), ("t_perm", perm_rows), ("t_bits", bits_rows),
                  ("t_trail", trail_rows))}
    write_lock(manifests)
    shutil.copyfile(C.manifest_path("t_self"), C.manifest_path("t_unlocked"))
    scorer_sha = sha(pathlib.Path(S.__file__).resolve())

    # --- a locked score (test mode: imgsz 64 on the CPU) ------------------------
    sys_tmp = TMP / "systmp"
    sys_tmp.mkdir()
    tempfile.tempdir = str(sys_tmp)
    exam_before = tree_state(C.EXAMS_DIR / "t_self")
    src_before = {p: sha(p) for p, _ in exam_imgs}
    s = S.score(W, "t_self", TMP / "out" / "self.json", imgsz=IMGSZ, batch=BATCH, device="cpu")
    tempfile.tempdir = None
    on_disk = json.loads((TMP / "out" / "self.json").read_text())
    keys = sorted(r["key"] for r in self_rows)
    check("lock check passes and is recorded", on_disk["lock_checked"] is True)
    check("score.json holds what score() returned",
          all(on_disk[k] == s[k] for k in on_disk), sorted(k for k in on_disk if on_disk[k] != s[k]))
    check("score.json is stamped with the manifest, scorer and weights hashes, the scorer's as TEST-",
          on_disk["manifest_sha256"] == manifests["t_self"]
          and on_disk["scorer_sha256"] == S.TEST_PREFIX + scorer_sha
          and on_disk["weights_sha256"] == sha(W), on_disk["scorer_sha256"])
    check("a test-mode score says production=false and lists its deviations (%s)"
          % "; ".join(on_disk["deviations"]),
          on_disk["production"] is False and on_disk["protocol_settings"] is False
          and any("imgsz 64" in x for x in on_disk["deviations"])
          and any("fp32 on device cpu" in x for x in on_disk["deviations"])
          and on_disk["settings"]["half"] is False)
    required = ("map50_95", "map50", "per_class", "per_class_ap50", "n_gt", "species_map50_95",
                "agnostic_map50_95", "agnostic_map50", "image_correct", "key_order_sha256",
                "n_images", "n_boxes", "seconds", "production", "deviations", "protocol_settings",
                "ultralytics_version", "torch_version", "device", "weights", "weights_sha256",
                "exam", "manifest_sha256", "scorer_sha256", "lock_checked", "created_utc")
    check("score.json has every field", all(k in on_disk for k in required),
          [k for k in required if k not in on_disk])
    check("n_images, n_boxes match the exam; every image and label was hashed",
          s["n_images"] == 20 and s["n_boxes"] == n_gt and s["images_checked"] == s["labels_checked"] == 20,
          (s["n_images"], s["n_boxes"], n_gt, s["images_checked"]))
    check("image_correct: one 0/1 per image, in key order",
          len(s["image_correct"]) == s["n_images"] and set(s["image_correct"]) <= {"0", "1"}
          and s["key_order_sha256"] == C.sha256_text("\n".join(keys)))
    check("the fixture is not trivial: 12-class mAP50-95 %.3f > 0.1" % s["map50_95"],
          s["map50_95"] > 0.1)
    present = {C.CLASS_NAMES[c] for c in classes}
    check("per-class AP for exactly the classes with GT; n_gt counts all 13 classes",
          set(s["per_class"]) == set(s["per_class_ap50"]) == present
          and list(s["n_gt"]) == C.CLASS_NAMES and sum(s["n_gt"].values()) == n_gt
          and all((s["n_gt"][n] > 0) == (n in present) for n in C.CLASS_NAMES), s["n_gt"])
    species = [s["per_class"][n] for n in present if n != "OtherPlant"]
    check("species score leaves OtherPlant out",
          abs(s["species_map50_95"] - sum(species) / len(species)) < 1e-12)
    check("scoring leaves the exam dir untouched (no labels.cache)",
          tree_state(C.EXAMS_DIR / "t_self") == exam_before)
    check("scoring leaves no temp dir behind", not os.listdir(sys_tmp), os.listdir(sys_tmp))

    # --- equal to a plain Ultralytics val --------------------------------------
    plain = YOLO(str(W)).val(data=str(C.EXAMS_DIR / "t_self" / "data.yaml"), imgsz=IMGSZ,
                             batch=BATCH, conf=0.001, iou=0.7, device="cpu", half=False,
                             plots=False, verbose=False, project=str(TMP / "runs"),
                             name="plain", exist_ok=True)
    plain_cls = {C.CLASS_NAMES[int(c)]: (float(plain.box.ap[i]), int(plain.nt_per_class[int(c)]))
                 for i, c in enumerate(plain.box.ap_class_index)}
    ours_cls = {n: (ap, s["n_gt"][n]) for n, ap in s["per_class"].items()}
    check("12-class mAP50-95 and mAP50 equal plain model.val (%.6f, %.6f)"
          % (plain.box.map, plain.box.map50),
          s["map50_95"] == float(plain.box.map) and s["map50"] == float(plain.box.map50),
          (s["map50_95"], s["map50"]))
    check("per-class AP50-95 and GT counts equal plain model.val", ours_cls == plain_cls,
          (ours_cls, plain_cls))

    # --- class-agnostic ---------------------------------------------------------
    p = S.score(W, "t_perm", TMP / "out" / "perm.json", imgsz=IMGSZ, batch=BATCH, device="cpu")
    check("agnostic score ignores GT class names (%.4f == %.4f)"
          % (p["agnostic_map50_95"], s["agnostic_map50_95"]),
          p["agnostic_map50_95"] == s["agnostic_map50_95"]
          and p["agnostic_map50"] == s["agnostic_map50"])
    check("class-confusion exam: agnostic %.3f >= 12-class %.3f, and 12-class drops from %.3f"
          % (p["agnostic_map50_95"], p["map50_95"], s["map50_95"]),
          p["agnostic_map50_95"] >= p["map50_95"] and p["map50_95"] < s["map50_95"])

    # --- the per-image bit through a real validation pass -------------------------
    b = S.score(WIRED, "t_bits", TMP / "out" / "bits.json", imgsz=IMGSZ, batch=BATCH, device="cpu")
    check("image_correct through val equals the bits built into the exam (%s)" % b["image_correct"],
          b["image_correct"] == want_bits and b["n_images_correct"] == want_bits.count("1"),
          (b["image_correct"], want_bits))

    # --- a JPEG Ultralytics would rewrite ---------------------------------------
    trail_sha = sha(trail)
    t = S.score(W, "t_trail", TMP / "out" / "trail.json", imgsz=IMGSZ, batch=BATCH, device="cpu")
    check("a JPEG with bytes after EOI is scored from a private copy; the source is unchanged",
          t["jpegs_copied_not_linked"] == 1 and sha(trail) == trail_sha)
    YOLO(str(W)).val(data=str(C.EXAMS_DIR / "t_trail" / "data.yaml"), imgsz=IMGSZ, batch=BATCH,
                     device="cpu", plots=False, verbose=False, project=str(TMP / "runs"),
                     name="plain_trail", exist_ok=True)
    check("(why) plain Ultralytics val rewrites that source photo in place",
          sha(trail) != trail_sha)
    check("scoring left the source photos unchanged",
          all(sha(q) == h for q, h in src_before.items()))

    # --- refusals ------------------------------------------------------------------
    man = C.manifest_path("t_self")
    orig = man.read_bytes()
    man.write_bytes(orig.replace(b'"session": ""', b'"session": "x"', 1))
    msg = refused(W, "t_self")
    man.write_bytes(orig)
    check("a manifest changed after locking is refused, and the earlier score.json at the "
          "output path is gone", msg and "changed since it was locked" in msg, msg)

    lock = json.loads(C.LOCK_PATH.read_text())
    write_lock(manifests, scorer_sha="0" * 64)
    msg = refused(W, "t_self")
    check("a scorer.py that differs from the lock is refused", msg and "scorer.py changed" in msg, msg)
    C.LOCK_PATH.write_text(json.dumps(lock))
    msg = refused(W, "t_unlocked")
    check("an exam that is not in LOCK.json is refused", msg and "not in LOCK.json" in msg, msg)

    img_dir, lab_dir = C.EXAMS_DIR / "t_self" / "images", C.EXAMS_DIR / "t_self" / "labels"
    shutil.move(str(img_dir / "ex_003.jpg"), str(TMP / "moved.jpg"))
    msg = refused(W, "t_self")
    shutil.move(str(TMP / "moved.jpg"), str(img_dir / "ex_003.jpg"))
    check("an exam dir missing a key is refused", msg and "1 missing ['ex_003']" in msg, msg)

    shutil.copyfile(exam_imgs[1][0], img_dir / "stray.jpg")
    msg = refused(W, "t_self")
    (img_dir / "stray.jpg").unlink()
    check("an exam dir with an extra image is refused", msg and "1 extra ['stray']" in msg, msg)

    lab = lab_dir / "ex_004.txt"
    lab_orig = lab.read_bytes()
    lab.write_bytes(lab_orig.replace(b"0.", b"1.", 1) if b"0." in lab_orig else b"5 0.5 0.5 0.2 0.2\n")
    msg = refused(W, "t_self")
    lab.write_bytes(lab_orig)
    check("an exam label that differs from the manifest is refused",
          msg and "labels that differ" in msg and "ex_004" in msg, msg)

    link = img_dir / "ex_005.jpg"
    target = os.readlink(link)
    link.unlink()
    os.symlink(exam_imgs[6][0], link)
    msg = refused(W, "t_self")
    link.unlink()
    os.symlink(target, link)
    check("an exam image that differs from the manifest is refused",
          msg and "images that differ" in msg and "ex_005" in msg, msg)

    bad_names = confident_copy(TMP / "runs" / "train" / "weights" / "last.pt", TMP / "names.pt",
                               names={i: "slot%d" % i for i in range(13)})
    msg = refused(bad_names, "t_self")
    check("weights whose classes are not the INC class space are refused",
          msg and "not the INC class space" in msg, msg)
    msg = refused(TMP / "missing.pt", "t_self")
    check("missing weights are refused", msg and "weights not found" in msg, msg)
    check("the exam still scores after the tampering was undone", refused(W, "t_self") is None)

    out = TMP / "out" / "garbage.json"
    out.write_text(STALE)
    (TMP / "garbage.pt").write_bytes(b"not a checkpoint")
    try:
        S.score(TMP / "garbage.pt", "t_self", out, imgsz=IMGSZ, batch=BATCH, device="cpu")
        err = None
    except Exception as e:          # an error, not a refusal: whatever Ultralytics raises
        err = e
    check("weights that cannot be loaded are an error, and the earlier score.json is gone",
          err is not None and not isinstance(err, S.ScorerRefused) and not out.exists(),
          "%s: %s" % (type(err).__name__, err))

    # --- protocol settings and the test mode ---------------------------------------
    with production_mode():
        msg = refused(W, "t_self")
        msg_cpu = refused(W, "t_self", imgsz=S.IMGSZ, batch=S.BATCH)
        msg_lock = refused(W, "t_self", lock_check=False, imgsz=S.IMGSZ, batch=S.BATCH)
    check("outside test mode a non-protocol score is refused, naming each deviation",
          msg and "not a protocol score" in msg and "imgsz 64" in msg and "batch 8" in msg
          and "fp32 on device cpu" in msg and S.TEST_ENV in msg, msg)
    check("outside test mode, protocol imgsz/batch on the CPU is still refused (fp32)",
          msg_cpu and "not a protocol score" in msg_cpu and "fp32 on device cpu" in msg_cpu
          and "imgsz" not in msg_cpu, msg_cpu)
    check("outside test mode the lock bypass is refused",
          msg_lock and "LOCK.json not checked" in msg_lock, msg_lock)
    if __import__("ultralytics").__version__ != S.PINNED_ULTRALYTICS:
        check("outside test mode an Ultralytics other than the pinned %s is refused"
              % S.PINNED_ULTRALYTICS, msg and "not the pinned" in msg, msg)

    saved = S.deviations
    try:
        S.deviations = lambda *a, **k: []
        with production_mode():
            prod = S.score(W, "t_self", TMP / "out" / "prod.json", imgsz=IMGSZ, batch=BATCH, device="cpu")
    finally:
        S.deviations = saved
    check("a score with no deviation is stamped with the bare scorer sha256, production=true",
          prod["scorer_sha256"] == scorer_sha and prod["production"] is True
          and prod["deviations"] == [] and prod["map50_95"] == s["map50_95"])

    C.LOCK_PATH.rename(TMP / "LOCK.json.away")
    msg = refused(W, "t_self")
    check("without LOCK.json the scorer refuses", msg and "lock check failed" in msg, msg)
    by = S.score(W, "t_self", TMP / "out" / "bypass.json", lock_check=False, imgsz=IMGSZ,
                 batch=BATCH, device="cpu")
    (TMP / "LOCK.json.away").rename(C.LOCK_PATH)
    check("the test-only bypass scores, records lock_checked=false and is stamped TEST-",
          by["lock_checked"] is False and by["map50_95"] == s["map50_95"]
          and by["scorer_sha256"] == S.TEST_PREFIX + scorer_sha
          and "LOCK.json not checked" in by["deviations"])

    # --- LoRA: last.pt holds adapters; the merged checkpoint is what is scored ------
    from weed_optimizer_framework.tools.inc import lora as L
    ck = torch.load(W, map_location="cpu", weights_only=False)
    wrapped = L.inject(_net(ck))        # B starts at zero: the network computes what it did
    torch.save(ck, TMP / "lora_last.pt")
    msg = refused(TMP / "lora_last.pt", "t_self")
    check("a LoRA last.pt with %d ConvLoRA adapters is refused, pointing at last_merged.pt"
          % len(wrapped), wrapped and msg and "ConvLoRA" in msg and "last_merged.pt" in msg, msg)
    L.merge(_net(ck))
    torch.save(ck, TMP / "lora_merged.pt")
    lo = S.score(TMP / "lora_merged.pt", "t_self", TMP / "out" / "lora.json", imgsz=IMGSZ,
                 batch=BATCH, device="cpu")
    check("its merged checkpoint scores like the network it came from",
          lo["map50_95"] == s["map50_95"] and lo["image_correct"] == s["image_correct"])

    # --- the CLI ------------------------------------------------------------------
    env = dict(os.environ, INC_DIR=str(C.INC_DIR), YOLO_VERBOSE="False")
    base = [sys.executable, "-m", "weed_optimizer_framework.tools.inc.scorer",
            "--imgsz", str(IMGSZ), "--batch", str(BATCH), "--device", "cpu", "--weights"]
    out_dir = TMP / "out" / "cli"
    run = subprocess.run(base + [str(WIRED), "--exam", "t_bits", "--out", str(out_dir)], cwd=str(ROOT),
                         env=env, capture_output=True, text=True, timeout=600)
    cli = json.loads((out_dir / "score.json").read_text()) if (out_dir / "score.json").exists() else {}
    check("CLI: exit 0, --out DIR gets score.json, same numbers and bits as in-process",
          run.returncode == 0 and cli.get("map50_95") == b["map50_95"]
          and cli.get("image_correct") == b["image_correct"] == want_bits,
          (run.returncode, run.stderr[-800:]))
    stale = TMP / "out" / "cli2.json"
    stale.write_text(STALE)
    run = subprocess.run(base + [str(W), "--exam", "t_unlocked", "--out", str(stale)],
                         cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    check("CLI: a refusal exits 2 and leaves no score.json, not even an earlier one",
          run.returncode == 2 and "REFUSED" in run.stderr and not stale.exists(),
          (run.returncode, run.stderr[-800:]))
    env.pop(S.TEST_ENV)
    stale.write_text(STALE)
    run = subprocess.run(base + [str(W), "--exam", "t_self", "--out", str(stale)],
                         cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    check("CLI: without %s a non-protocol score exits 2" % S.TEST_ENV,
          run.returncode == 2 and "not a protocol score" in run.stderr and not stale.exists(),
          (run.returncode, run.stderr[-800:]))


def main():
    t0 = time.time()
    try:
        check("the test runs on its own INC_DIR", C.INC_DIR == TMP / "inc", C.INC_DIR)
        test_image_correct()
        test_one_per_box()
        test_validator_hook()
        test_deviations()
        test_every_image_hashed()
        test_end_to_end()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
