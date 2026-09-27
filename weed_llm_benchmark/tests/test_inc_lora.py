#!/usr/bin/env python3
"""INC LoRA (Step 0.5): adapters train on a pretrained, frozen base and ship as a plain checkpoint.

Audit of tools/lora_yolo.py, reproduced here on yolo11n.yaml with nc 13 in both
the source weights and the data (Ultralytics 8.4.22 / torch 2.10 locally; the
test re-establishes each fact on whatever version runs it). The source network
has every tensor perturbed, so "equal to the source" means "transferred".
- lora_yolo wraps convs of YOLO.model and calls YOLO.train(). Model.train
  rebuilds the network from its yaml (trainer.get_model(weights=self.model,
  cfg=self.model.yaml)) and copies the old state_dict by key and shape.
- The network that trains has no ConvLoRA module and no lora_A / lora_B
  parameter: no adapter is ever trained.
- target_layers='head' wraps 5 convs (model.20.conv and four 3x3 convs in
  model.22). 494 of 499 state entries are transferred (499 of 499 without
  injection); the 5 missing ones are exactly the wrapped conv weights, stored
  under '<conv>.original_conv.weight', so those convs keep a random init.
  With freeze=22 (what lora_yolo passes in that mode) model.20.conv is then
  frozen at that random init.
- 'hybrid' (train_yolo_with_lora's default), 'backbone' and 'all' wrap
  model.0.conv, and DetectionModel.load raises KeyError('model.0.conv.weight')
  before training starts.
- 'hybrid' also wraps the Detect layer's twelve 3x3 convs: lora_yolo's layer
  numbers are YOLOv8's, and in YOLO11n Detect is layer 23.

What is pinned for tools/inc/lora.py, the replacement:
- ConvLoRA is the identity at init; merge() folds B A into the conv exactly
  and leaves the original architecture's state keys;
- inject() wraps only dense 3x3 convs outside the Detect layer, and
  unwrapped_3x3() names the 3x3 convs it skips (in YOLO11n, C2PSA's depthwise
  positional-encoding conv), which lora.json records;
- a 1-epoch CPU training changes only adapters and the Detect layer: every
  frozen tensor, including frozen BatchNorm running statistics, is
  bit-identical afterwards, and the optimizer holds only trainable
  parameters;
- the merged checkpoint loads with plain YOLO() in a process that cannot
  import this repository, and matches the unmerged model to < 1e-4;
- save_model follows Ultralytics' 8.4.37 contract, emulated on older
  versions. When BaseTrainer.save_model returns False (EMA NaN/Inf after the
  first epoch; last.pt not rewritten), last_merged.pt is left byte-identical
  and save_model returns False. A normal epoch returns True, which 8.4.37's
  loop needs to run on_model_save. A non-finite merged checkpoint is never
  written, and check_saved() refuses a merged file that is non-finite or
  from another epoch than last.pt;
- train_lora refuses **extra keys it sets itself (model=, data=, ...), so
  lora.json always names the weights and data the run started from.

Run:  python3 tests/test_inc_lora.py   (CPU, offline, about a minute)
"""
import copy
import hashlib
import inspect
import os
import pathlib
import subprocess
import sys
import tempfile

# Before ultralytics is imported: no network checks, quieter logs.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_VERBOSE", "false")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.trainer import BaseTrainer  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from weed_optimizer_framework.tools import lora_yolo  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import lora as L  # noqa: E402

FAILURES = []
IMGSZ = 64


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# ------------------------------------------------------------------ fixtures
def make_dataset(root, n=16, size=96, seed=0):
    """Coloured rectangles on noise, one or two boxes per image, INC ids."""
    rng = np.random.RandomState(seed)
    src = pathlib.Path(root) / "src"
    src.mkdir(parents=True)
    rows = []
    for i in range(n):
        im = rng.randint(0, 60, (size, size, 3)).astype(np.uint8)
        boxes = []
        for _ in range(1 + i % 2):
            w, h = rng.randint(size // 5, size // 2, 2)
            x0, y0 = rng.randint(0, size - w), rng.randint(0, size - h)
            im[y0:y0 + h, x0:x0 + w] = rng.randint(80, 255, 3)
            boxes.append((rng.randint(0, C.NC), (x0 + w / 2) / size, (y0 + h / 2) / size, w / size, h / size))
        img, lab = src / ("im%02d.jpg" % i), src / ("im%02d.txt" % i)
        Image.fromarray(im).save(img)
        C.write_yolo(lab, boxes)
        rows.append({"image": str(img), "label": str(lab), "sha256": "", "label_sha256": "",
                     "source": "synthetic", "session": "", "key": "im%02d" % i})
    return str(C.materialise(rows, pathlib.Path(root) / "data"))


def perturbed_net(seed=0):
    """yolo11n.yaml at nc 13 with every tensor perturbed: 'equal to the source'
    can then only mean 'copied from it', and deep activations are not ~0 as
    they are in a freshly initialised network."""
    torch.manual_seed(seed)
    net = DetectionModel("yolo11n.yaml", nc=C.NC, verbose=False)
    with torch.no_grad():
        for k, v in net.state_dict().items():
            if ".dfl" in k:
                continue  # the fixed DFL projection, identical in every checkpoint
            if k.endswith("running_var"):
                v.uniform_(0.5, 1.5)
            elif k.endswith("num_batches_tracked"):
                v.fill_(7)
            else:
                v.add_(torch.randn_like(v) * 0.05)
    return net.eval()


def make_source(path, seed=0):
    """perturbed_net saved as an Ultralytics checkpoint; returns its state."""
    net = perturbed_net(seed)
    torch.save({"model": net, "train_args": {}}, path)
    return {k: v.clone() for k, v in net.state_dict().items()}


def _flat(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, dict):
        return [t for k in sorted(o) for t in _flat(o[k])]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in _flat(v)]
    return []


def net_out(model, x):
    """(decoded predictions, [raw head outputs]) of an eval-mode forward."""
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (list, tuple)) and len(y) == 2 and torch.is_tensor(y[0]):
        return y[0], _flat(y[1])
    return _flat(y)[0], _flat(y)[1:]


def out_diff(a, b):
    """max |diff| over the raw head outputs, and over the decoded predictions
    relative to their largest value. Decoded boxes are pixel coordinates up to
    ~stride * reg_max (hundreds), where one fp32 ulp is already ~3e-5, so they
    get a relative bound; the raw outputs get the absolute 1e-4."""
    raw = max([float((p - q).abs().max()) for p, q in zip(a[1], b[1])] or [0.0])
    dec = float((a[0] - b[0]).abs().max()) / max(1.0, float(a[0].abs().max()))
    return raw, dec


def fresh_net(seed=0):
    torch.manual_seed(seed)
    return DetectionModel("yolo11n.yaml", nc=C.NC, verbose=False).eval()


# --------------------------------------------------------------------- audit
class _Stop(Exception):
    pass


class _Capture(DetectionTrainer):
    """Model.train up to the first batch: Model.train's get_model, then
    _setup_train (freeze flags, optimizer); keeps the network that would train."""
    seen = None

    def train(self):
        self._setup_train()
        _Capture.seen = self
        raise _Stop()


def run_lora_yolo_path(src_pt, data_yaml, tmp, mode, freeze):
    yolo = YOLO(src_pt)
    wrapped = []
    if mode:
        wrapped = ["%s.%s" % pa for pa in
                   lora_yolo.inject_lora_into_yolo(yolo, target_layers=mode, rank=16, alpha=32.0)]
    try:
        yolo.train(trainer=_Capture, data=data_yaml, epochs=1, imgsz=IMGSZ, batch=4, device="cpu",
                   workers=0, project=tmp, name="audit_%s" % mode, exist_ok=True, plots=False,
                   val=False, freeze=freeze, lr0=0.0005, patience=15)
    except _Stop:
        return wrapped, _Capture.seen.model, None
    except KeyError as e:
        return wrapped, None, e
    raise AssertionError("Model.train returned without reaching the trainer")


def test_audit(src, src_pt, data_yaml, tmp):
    print("audit of lora_yolo (ultralytics %s)" % L.ULTRALYTICS_VERSION)
    _, control, _ = run_lora_yolo_path(src_pt, data_yaml, tmp, None, 22)
    csd = control.state_dict()
    n_total = len(csd)
    n_ctrl = sum(torch.equal(v, src[k]) for k, v in csd.items())
    print("       control (no injection): %d/%d state entries equal the source" % (n_ctrl, n_total))
    check("without injection, Model.train transfers every tensor", n_ctrl == n_total)

    wrapped, trained, err = run_lora_yolo_path(src_pt, data_yaml, tmp, "head", 22)
    check("'head' mode reaches training", err is None, repr(err))
    if trained is not None:
        sd = trained.state_dict()
        n_lora = sum(1 for k, _ in trained.named_parameters() if "lora_" in k)
        n_mod = sum(1 for m in trained.modules() if isinstance(m, lora_yolo.ConvLoRA))
        differing = sorted(k for k, v in sd.items() if not torch.equal(v, src[k]))
        n_eq = n_total - len(differing)
        frozen_random = [w for w in wrapped if not trained.get_parameter(w + ".weight").requires_grad]
        print("       head: wrapped %d %s; %d/%d transferred; %d lora params, %d ConvLoRA in the "
              "trained net; frozen at random init: %s"
              % (len(wrapped), wrapped, n_eq, n_total, n_lora, n_mod, frozen_random))
        check("the network that trains has no adapter parameter", n_lora == 0 and n_mod == 0)
        check("exactly the wrapped conv weights are not transferred",
              differing == sorted(w + ".weight" for w in wrapped), differing)
        check("transferred = all - wrapped", n_eq == n_total - len(wrapped))
        check("with freeze=22 a wrapped conv trains nothing but its random init stays frozen",
              len(frozen_random) >= 1)

    for mode, freeze in (("hybrid", 20), ("backbone", 22), ("all", 22)):
        wrapped, trained, err = run_lora_yolo_path(src_pt, data_yaml, tmp, mode, freeze)
        det = "model.%d." % (len(control.model) - 1)
        in_det = [w for w in wrapped if w.startswith(det)]
        print("       %s: wrapped %d (%d in Detect %s); outcome %s"
              % (mode, len(wrapped), len(in_det), det, repr(err) if err else "trained"))
        check("%s wraps model.0.conv" % mode, "model.0.conv" in wrapped)
        if err is not None:
            check("%s fails before training on the renamed first conv" % mode,
                  "model.0.conv.weight" in str(err), repr(err))
        else:  # a later Ultralytics may drop the fallback; the adapters must still be gone
            check("%s: the network that trains has no adapter" % mode,
                  not any("lora_" in k for k, _ in trained.named_parameters()))
            check("%s: model.0.conv is not the source's" % mode,
                  not torch.equal(trained.state_dict()["model.0.conv.weight"], src["model.0.conv.weight"]))
        if mode == "hybrid":
            check("hybrid also wraps the Detect layer's 3x3 convs (YOLOv8 numbering)", len(in_det) == 12,
                  in_det)


# ---------------------------------------------------------------- unit tests
def test_convlora():
    print("ConvLoRA / merge / inject")
    torch.manual_seed(1)
    conv = nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=True)
    lo = L.ConvLoRA(conv, rank=4, alpha=8)
    x = torch.randn(2, 8, 17, 17)
    check("ConvLoRA is the identity at init", torch.equal(lo(x), conv(x)))
    check("A copies stride/padding/dilation, B is 1x1 zero, scaling = alpha/rank",
          lo.lora_A.stride == conv.stride and lo.lora_A.padding == conv.padding
          and lo.lora_A.dilation == conv.dilation and lo.lora_A.groups == 1
          and lo.lora_B.kernel_size == (1, 1) and float(lo.lora_B.weight.detach().abs().sum()) == 0
          and lo.scaling == 2.0)
    try:
        L.ConvLoRA(nn.Conv2d(8, 8, 3, groups=8), rank=4)
        check("a depthwise conv is refused", False)
    except ValueError:
        check("a depthwise conv is refused", True)

    for kw in ({"stride": 2, "padding": 1}, {"stride": 1, "padding": 2, "dilation": 2}):
        torch.manual_seed(2)
        seq = nn.Sequential(L.ConvLoRA(nn.Conv2d(8, 16, 3, bias=True, **kw), rank=4, alpha=8))
        with torch.no_grad():
            seq[0].lora_A.weight.normal_(0, 0.2)
            seq[0].lora_B.weight.normal_(0, 0.2)
        bias = seq[0].base.bias.clone()
        y1 = seq(x)
        names = L.merge(seq)
        y2 = seq(x)
        err = float((y1 - y2).abs().max())
        check("merge is exact (%s): max |diff| %.2e" % (kw, err),
              names == ["0"] and type(seq[0]) is nn.Conv2d and err < 1e-5
              and torch.equal(seq[0].bias, bias))

    # on the whole network
    net = perturbed_net()
    ref = perturbed_net()
    xi = torch.rand(2, 3, IMGSZ, IMGSZ)
    y0 = net_out(net, xi)
    skipped = L.unwrapped_3x3(net)
    names = L.inject(net, rank=16, alpha=32)
    y0b = net_out(net, xi)
    check("a freshly injected YOLO11n computes exactly what it did",
          torch.equal(y0b[0], y0[0]) and all(torch.equal(p, q) for p, q in zip(y0b[1], y0[1])))

    det_i = len(net.model) - 1
    det = "model.%d." % det_i
    expect = sorted(n for n, m in ref.named_modules()
                    if isinstance(m, nn.Conv2d) and not n.startswith(det)
                    and m.kernel_size == (3, 3) and m.groups == 1 and m.dilation == (1, 1))
    print("       wrapped %d convs in layers 0..%d" % (len(names), det_i - 1))
    check("inject wraps exactly the dense 3x3 convs outside Detect", sorted(names) == expect)
    check("no Detect-layer conv is wrapped",
          not any(n.startswith(det) for n in names)
          and not any(isinstance(m, L.ConvLoRA) for m in net.model[det_i].modules()))
    check("depthwise and 1x1 convs are left alone",
          all(not isinstance(net.get_submodule(n), L.ConvLoRA) for n, m in ref.named_modules()
              if isinstance(m, nn.Conv2d) and (m.groups != 1 or m.kernel_size != (3, 3))))
    all_3x3 = sorted(n for n, m in ref.named_modules()
                     if isinstance(m, nn.Conv2d) and not n.startswith(det) and m.kernel_size == (3, 3))
    print("       3x3 convs outside Detect left without an adapter: %s" % skipped)
    check("unwrapped_3x3 = the 3x3 convs outside Detect that inject skips (YOLO11n: C2PSA's depthwise pe conv)",
          sorted(skipped + names) == all_3x3 and not set(skipped) & set(names)
          and "model.10.m.0.attn.pe.conv" in skipped
          and all(ref.get_submodule(n).groups != 1 or ref.get_submodule(n).dilation != (1, 1) for n in skipped),
          skipped)
    try:
        L.unwrapped_3x3(net)
        check("unwrapped_3x3 after inject is refused", False)
    except ValueError:
        check("unwrapped_3x3 after inject is refused", True)
    try:
        L.inject(net)
        check("injecting twice is refused", False)
    except ValueError:
        check("injecting twice is refused", True)
    sub = L.inject(fresh_net(), targets=[0, 1])
    check("targets as layer indices", sub == ["model.0.conv", "model.1.conv"], sub)
    try:
        L.inject(fresh_net(), targets=[det_i])
        check("the Detect layer cannot be targeted", False)
    except ValueError:
        check("the Detect layer cannot be targeted", True)

    counts = L.set_trainable(net)
    wrong = [n for n, p in net.named_parameters()
             if p.requires_grad != (L.is_adapter_name(n) or (n.startswith(det) and ".dfl" not in n))]
    check("set_trainable: adapters + Detect (except DFL) and nothing else", not wrong, wrong[:5])
    check("parameter counts add up",
          counts["trainable_params"] == counts["adapter_params"] + counts["head_params"]
          and counts["total_params"] == sum(p.numel() for p in net.parameters())
          and counts["base_params"] == sum(p.numel() for p in ref.parameters()), counts)

    with torch.no_grad():
        for n, p in net.named_parameters():
            if L.is_adapter_name(n):
                p.normal_(0, 0.1)
    y1 = net_out(net, xi)
    effect = out_diff(y0, y1)[0]
    check("random adapters change the raw head outputs (max |diff| %.2e)" % effect, effect > 1e-2)
    merged = L.merge(net)
    raw, dec = out_diff(y1, net_out(net, xi))
    check("merging a YOLO11n with random adapters is exact: raw %.2e, decoded rel %.2e" % (raw, dec),
          sorted(merged) == sorted(names) and raw < 1e-4 and dec < 1e-6)
    check("the merged network has the original architecture's state keys and shapes",
          {k: v.shape for k, v in net.state_dict().items()} == {k: v.shape for k, v in ref.state_dict().items()})


# ------------------------------------------------------------------ training
TRAIN_KW = dict(epochs=1, imgsz=IMGSZ, batch=4, nbs=4, workers=0, device="cpu", optimizer="SGD",
                lr0=0.01, warmup_epochs=0, close_mosaic=0, val=False, plots=False, seed=0,
                deterministic=True, amp=False, exist_ok=True)

CHILD = r'''
import importlib.abc, sys, torch
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "weed_optimizer_framework":
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
from ultralytics import YOLO
m = YOLO(sys.argv[1]).model.float().eval()
x = torch.load(sys.argv[2])
with torch.no_grad():
    y = m(x)
torch.save(y, sys.argv[3])
foreign = [type(q).__module__ for q in m.modules()
           if not type(q).__module__.startswith(("torch.", "ultralytics."))]
assert not foreign, foreign
assert not [k for k in sys.modules if k.startswith("weed_optimizer_framework")]
print("child loaded %s, %d params" % (type(m).__name__, sum(p.numel() for p in m.parameters())))
'''


def load_plain(merged, x, tmp):
    """Run the merged checkpoint in a fresh interpreter that cannot import
    this repository; returns its output (or None) and the child's log."""
    inp, out = os.path.join(tmp, "x.pt"), os.path.join(tmp, "y.pt")
    torch.save(x, inp)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", CHILD, merged, inp, out], cwd=tmp, env=env,
                       capture_output=True, text=True, timeout=600)
    log = (r.stdout + r.stderr).strip().splitlines()[-3:]
    if r.returncode:
        return None, log
    y = torch.load(out)
    return (y[0], _flat(y[1])) if isinstance(y, (list, tuple)) and len(y) == 2 else (_flat(y)[0], _flat(y)[1:]), log


def test_training(src, src_pt, data_yaml, tmp):
    print("1-epoch LoRADetectionTrainer run")
    tr = L.LoRADetectionTrainer(overrides=dict(TRAIN_KW, model=src_pt, data=data_yaml, project=tmp,
                                               name="trainer"),
                                lora={"rank": 4, "alpha": 8})
    snap = {}

    def on_setup(t):
        snap["state"] = {k: v.detach().clone() for k, v in t.model.state_dict().items()}

    tr.add_callback("on_pretrain_routine_end", on_setup)
    tr.train()
    model = tr.model
    det = L.detect_prefix(model)
    after = model.state_dict()
    before = snap["state"]

    def trains(k):
        return L.is_adapter_name(k) or (k.startswith(det) and ".dfl" not in k)

    base_key = {k: k.replace(".base.", ".") for k in after if not L.is_adapter_name(k)}
    check("the frozen base started as the source (all tensors outside Detect)",
          all(torch.equal(before[k], src[base_key[k]]) for k in base_key if not k.startswith(det)))
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    in_opt = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
    names_tr = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    check("the optimizer holds exactly the trainable parameters", in_opt == trainable)
    check("trainable = adapters + Detect (except DFL)",
          names_tr == sorted(n for n, _ in model.named_parameters() if trains(n)))

    frozen = [k for k in after if not trains(k)]
    moved = [k for k in frozen if not torch.equal(after[k], before[k])]
    check("every frozen tensor is bit-identical after training (%d tensors)" % len(frozen), not moved, moved[:5])
    wrapped_base = [k for k in frozen if ".base.weight" in k]
    check("frozen wrapped conv weights included (%d)" % len(wrapped_base),
          len(wrapped_base) == len(tr.lora_wrapped) > 0)
    bn_frozen = [k for k in frozen if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    bn_head = [k for k in after if k.startswith(det) and k.endswith("running_mean")]
    check("frozen BN running stats unchanged (%d buffers)" % len(bn_frozen),
          bn_frozen and not [k for k in bn_frozen if not torch.equal(after[k], before[k])])
    check("the test can see BN stats move: Detect BN running means changed",
          any(not torch.equal(after[k], before[k]) for k in bn_head))
    b_keys = [k for k in after if ".lora_B." in k]
    a_keys = [k for k in after if ".lora_A." in k]
    check("every lora_B moved off zero", all(float(after[k].abs().sum()) > 0 for k in b_keys))
    check("lora_A changed", any(not torch.equal(after[k], before[k]) for k in a_keys))
    check("the Detect head changed",
          any(not torch.equal(after[k], before[k]) for k in after if k.startswith(det) and k.endswith("weight")))

    check("weights: last.pt and last_merged.pt kept, best.pt removed",
          tr.last.exists() and tr.merged.exists() and not tr.best.exists())
    torch.manual_seed(3)
    x = torch.rand(2, 3, IMGSZ, IMGSZ)
    y_unmerged = net_out(tr.ema.ema.float().eval(), x)
    y_plain, log = load_plain(str(tr.merged), x, tmp)
    print("       " + " | ".join(log))
    check("plain YOLO() loads the merged checkpoint without this repository", y_plain is not None, log)
    if y_plain is not None:
        raw, dec = out_diff(y_unmerged, y_plain)
        check("merged matches the unmerged EMA model: raw max |diff| %.2e, decoded rel %.2e" % (raw, dec),
              raw < 1e-4 and dec < 1e-6)
    ck = torch.load(tr.merged, map_location="cpu", weights_only=False)
    msd = ck["model"].state_dict()
    check("merged checkpoint: fp32, original architecture keys, lora metadata",
          all(v.dtype != torch.float16 for v in msd.values())
          and set(msd) == set(src) and ck["lora"]["rank"] == 4 and len(ck["lora"]["merged"]) == len(tr.lora_wrapped))
    far = max(float((msd[k].float() - src[k].float()).abs().max()) for k in msd
              if not k.startswith(det) and k.replace(".weight", "") not in tr.lora_wrapped)
    check("merged: unwrapped frozen tensors equal the source up to EMA rounding (max %.1e)" % far, far < 1e-5)


def test_train_lora(src_pt, data_yaml, tmp):
    print("train_lora entry point")
    kw = {k: v for k, v in TRAIN_KW.items() if k not in ("epochs", "imgsz", "batch", "workers", "device",
                                                          "lr0", "seed")}
    path = L.train_lora(src_pt, data_yaml, project=tmp, name="entry", epochs=1, lr0=0.01, seed=0,
                        imgsz=IMGSZ, batch=4, device="cpu", workers=0, rank=16, alpha=32, **kw)
    wdir = pathlib.Path(path).parent
    check("returns weights/last_merged.pt; last.pt kept",
          pathlib.Path(path).name == L.MERGED_NAME and pathlib.Path(path).exists()
          and (wdir / "last.pt").exists() and not (wdir / "best.pt").exists(), path)
    rep = json.load(open(wdir.parent / L.LORA_JSON))
    print("       lora.json: %d wrapped, trainable %d / total %d (adapters %d, head %d), transferred %d/%d"
          % (rep["n_wrapped"], rep["trainable_params"], rep["total_params"], rep["adapter_params"],
             rep["head_params"], rep["transferred"], rep["state_entries"]))
    check("lora.json has trainable / total parameter counts",
          0 < rep["trainable_params"] < rep["total_params"]
          and rep["trainable_params"] == rep["adapter_params"] + rep["head_params"]
          and rep["rank"] == 16 and rep["alpha"] == 32.0)
    check("the whole source transferred (same nc), nothing re-initialised",
          rep["transferred"] == rep["state_entries"] and rep["detect_reinit"] == [])
    check("lora.json records the skipped 3x3 convs and where the merged file comes from",
          rep["unwrapped_3x3"] == ["model.10.m.0.attn.pe.conv"] and rep["epochs"] == 1
          and rep["merged_from_epoch"] == 0 and rep["skipped_saves"] == []
          and rep["init_weights"] == src_pt and rep["data"] == data_yaml,
          {k: rep[k] for k in ("unwrapped_3x3", "epochs", "merged_from_epoch", "skipped_saves")})
    m = YOLO(path)
    check("YOLO() loads it with the INC class names", list(m.names.values()) == C.CLASS_NAMES)
    for bad, key in (({"optimizer": "auto"}, "optimizer auto"), ({"freeze": 10}, "freeze"),
                     ({"model": "other.pt"}, "model= in **extra (would override init_weights)"),
                     ({"data": "other.yaml"}, "data= in **extra (would override data_yaml)"),
                     ({"task": "segment"}, "task= in **extra")):
        try:
            L.train_lora(src_pt, data_yaml, project=tmp, name="refused", epochs=1, imgsz=IMGSZ, batch=4,
                         device="cpu", workers=0, **{**kw, **bad})
            check("%s is refused" % key, False)
        except ValueError:
            check("%s is refused" % key, True)


# ----------------------------------------------------- save_model contract
def save_model_8437(orig):
    """BaseTrainer.save_model under the Ultralytics 8.4.37 contract
    (engine/trainer.py at v8.4.37): after the first epoch a non-finite EMA
    writes nothing and returns False; otherwise the checkpoint is written and
    True returned. On 8.4.37+ this repeats the installed check."""
    def save_model(self):
        ema = copy.deepcopy(self.ema.ema).half()
        if (not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor))
                and self.epoch > self.start_epoch):
            return False
        orig(self)
        return True
    return save_model


class _Recorder(L.LoRADetectionTrainer):
    """Records what save_model returns and last_merged.pt's hash after each call."""

    def save_model(self):
        r = super().save_model()
        self.returns.append((self.epoch, r))
        self.merged_sha.append(sha256(self.merged))
        return r


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() if pathlib.Path(path).exists() else None


def test_save_contract(src_pt, data_yaml, tmp):
    print("save_model when Ultralytics skips a save (8.4.37 contract, NaN in the EMA at the last epoch)")
    fired = []

    def poison(t):
        if t.epoch == t.epochs - 1:
            with torch.no_grad():
                p = [p for n, p in t.ema.ema.named_parameters() if ".lora_B." in n][0]
                p[0, 0, 0, 0] = float("nan")

    orig = BaseTrainer.save_model
    BaseTrainer.save_model = save_model_8437(orig)
    try:
        tr = _Recorder(overrides=dict(TRAIN_KW, model=src_pt, data=data_yaml, project=tmp, name="skip",
                                      epochs=2), lora={"rank": 4, "alpha": 8})
        tr.returns, tr.merged_sha = [], []
        tr.add_callback("on_train_epoch_end", poison)
        tr.add_callback("on_model_save", lambda t: fired.append(t.epoch))
        tr.train()
    finally:
        BaseTrainer.save_model = orig

    check("save_model returns True on a saved epoch and False on the skipped one",
          tr.returns == [(0, True), (1, False)], tr.returns)
    gated = "and self.save_model()" in inspect.getsource(BaseTrainer._do_train)
    print("       this Ultralytics runs on_model_save only on a true save_model: %s; it ran for epochs %s"
          % (gated, fired))
    check("on_model_save ran for the saved epoch (and, where gated, not for the skipped one)",
          fired == ([0] if gated else [0, 1]), fired)
    check("the skipped epoch left last_merged.pt byte-identical",
          len(tr.merged_sha) == 2 and tr.merged_sha[0] is not None and tr.merged_sha[1] == tr.merged_sha[0])
    ck = torch.load(tr.merged, map_location="cpu", weights_only=False)
    last = torch.load(tr.last, map_location="cpu", weights_only=False)
    check("last_merged.pt is from epoch 0 and finite; last.pt is finite",
          ck["lora"]["from_epoch"] == 0 and not L.nonfinite_tensors(ck["model"])
          and not L.nonfinite_tensors(last["model"]) and tr.lora_skipped_saves == [1]
          and tr.lora_saved_epoch == 0)
    try:
        ep = L.check_saved(tr.merged, tr.last)
        check("check_saved: merged and last.pt both from epoch 0 (last.pt stripped, epoch %r)" % last["epoch"],
              ep == 0, ep)
    except RuntimeError as e:
        check("check_saved: merged and last.pt both from epoch 0", False, repr(e))

    before = sha256(tr.merged)
    try:
        tr._save_merged()
        check("a non-finite EMA is never written as a merged checkpoint", False)
    except RuntimeError as e:
        check("a non-finite EMA is never written as a merged checkpoint",
              "NaN/Inf" in str(e) and sha256(tr.merged) == before, repr(e))

    wrong = os.path.join(tmp, "wrong_epoch.pt")
    torch.save({**ck, "lora": {**ck["lora"], "from_epoch": 1}}, wrong)
    nan = os.path.join(tmp, "nan.pt")
    m = copy.deepcopy(ck["model"])
    with torch.no_grad():
        next(m.parameters()).view(-1)[0] = float("inf")
    torch.save({**ck, "model": m}, nan)
    for path, what in ((wrong, "a merged file from another epoch than last.pt"), (nan, "a non-finite merged file")):
        try:
            L.check_saved(path, tr.last)
            check("check_saved refuses %s" % what, False)
        except RuntimeError:
            check("check_saved refuses %s" % what, True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = make_dataset(tmp)
        src_pt = os.path.join(tmp, "source.pt")
        src = make_source(src_pt)
        test_audit(src, src_pt, data_yaml, tmp)
        test_convlora()
        test_training(src, src_pt, data_yaml, tmp)
        test_save_contract(src_pt, data_yaml, tmp)
        test_train_lora(src_pt, data_yaml, tmp)

    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
