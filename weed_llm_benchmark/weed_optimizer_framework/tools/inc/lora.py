"""LoRA for YOLO11 detection in INC (Step 0.5): adapters that really train, merged on save.

Audit of tools/lora_yolo.py. tests/test_inc_lora.py reproduces its code path on
yolo11n.yaml (nc 13 in both the source weights and the data) under Ultralytics
8.4.22 / torch 2.10, and re-checks these facts on whatever version runs it:
- lora_yolo wraps convs of YOLO.model in its ConvLoRA and then calls
  YOLO.train(). Model.train (engine/model.py) builds the trainer and then
  replaces the network: trainer.model = trainer.get_model(weights=self.model,
  cfg=self.model.yaml). DetectionTrainer.get_model builds a new DetectionModel
  from the yaml, and DetectionModel.load copies the old state_dict into it by
  key and shape (utils/torch_utils.intersect_dicts).
- Adapters: the network that trains has no ConvLoRA module and no lora_A or
  lora_B parameter. No adapter is ever trained.
- Wrapped convs: their pretrained weights sit under
  '<conv>.original_conv.weight', a key the rebuilt network does not have, so
  each wrapped conv keeps the rebuilt network's random init. With
  target_layers='head', 5 convs are wrapped (model.20.conv and four 3x3 convs
  in model.22). 494 of 499 state entries are transferred, against 499 of 499
  without injection, and none of the 5 wrapped weights equals its source.
  freeze=22, which lora_yolo passes in that mode, then leaves model.20.conv
  frozen at its random init.
- 'hybrid' (the default of train_yolo_with_lora), 'backbone' and 'all' also
  wrap model.0.conv. DetectionModel.load's first-conv fallback then reads the
  source's 'model.0.conv.weight', which the rename removed, and raises
  KeyError before training starts.
- lora_yolo's layer numbers are YOLOv8's (head = 20-22). In YOLO11n the Detect
  layer is 23, so 'hybrid' also wraps the Detect layer's twelve 3x3 convs, six
  of them depthwise.

What this module does instead:
- LoRADetectionTrainer.get_model lets Ultralytics build the network and load
  the weights, checks that every tensor outside the Detect layer arrived
  verbatim (a LoRA base that is partly random is the bug above), and only then
  injects the adapters.
- Adapters go on every dense 3x3 conv (groups 1, dilation 1) in the backbone
  and neck. This narrows the protocol's "every 3x3 conv": a dense B A update
  cannot be folded into a grouped conv's weight, so a grouped 3x3 conv (or a
  dilated one; YOLO11 has none) gets no adapter and stays frozen. In YOLO11n
  the only one is C2PSA's depthwise positional-encoding conv,
  model.10.m.0.attn.pe.conv. lora.json lists the skipped convs under
  'unwrapped_3x3'.
- Trainable = adapters + the Detect layer (except its fixed DFL conv). The
  flags are set in build_optimizer because _setup_train re-enables
  requires_grad on every parameter outside args.freeze right before it builds
  the optimizer; the optimizer receives only trainable parameters.
- BatchNorm layers outside the Detect layer stay in eval mode during training,
  so their running statistics do not move.
- On every save: weights/last.pt is Ultralytics' own checkpoint (the EMA in
  fp16, with the adapters; unpickling it needs this module).
  weights/last_merged.pt is the EMA in fp32 with every adapter folded into
  its conv (plain nn.Conv2d, W + scaling * B A, computed in float64 and
  rounded once); plain YOLO() loads it without this module, and it is the
  file INC scores. best.pt is removed: INC never uses it, and Ultralytics'
  final_eval would fuse it, which fails on ConvLoRA.
- Ultralytics can skip a save. From 8.4.37 on, BaseTrainer.save_model returns
  False and writes nothing when the EMA holds NaN/Inf after the first epoch,
  so last.pt stays at the last finite epoch. save_model then leaves
  last_merged.pt as it was, so it still matches last.pt, and returns False
  too. On a save it returns True; 8.4.37's loop runs the on_model_save
  callbacks only on a true return. A merged checkpoint with a non-finite
  tensor is never written. train_lora checks that the file it returns is
  finite and comes from the epoch that last.pt holds.

Numerical facts worth knowing when comparing checkpoints:
- Ultralytics' EMA averages every float tensor, frozen ones included, so a
  frozen weight in the EMA differs from the source by float rounding only (a
  few fp32 ulps).
- On CUDA with AMP the final-epoch validation casts the EMA to fp16 and back,
  so there last_merged.pt is last.pt with the adapters folded in. On CPU the
  EMA stays fp32, and last.pt is its fp16 rounding.
"""
from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from ultralytics import __version__ as ULTRALYTICS_VERSION
from ultralytics.engine.model import Model
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER

ADAPTER_KEYS = (".lora_A.", ".lora_B.")
LORA_JSON = "lora.json"
MERGED_NAME = "last_merged.pt"

# The protocol's incremental LoRA recipe (docs/INCREMENTAL_PROTOCOL.md); any
# key can be overridden through train_lora(**extra).
INC_LORA_DEFAULTS = {
    "optimizer": "SGD",
    "cos_lr": True,
    "lrf": 0.01,
    "warmup_epochs": 1,
    "warmup_bias_lr": 0.01,
    "val": False,
    "plots": False,
    "deterministic": True,
}


# ------------------------------------------------------------------ adapter
class ConvLoRA(nn.Module):
    """y = base(x) + scaling * B(A(x)), scaling = alpha / rank.

    A is a k x k conv with base's stride, padding and dilation (groups 1), B is
    1x1 and starts at zero, so a freshly wrapped network computes exactly what
    it computed before. Because B is 1x1, B(A(x)) is one k x k conv whose
    weight is einsum(B[:, :, 0, 0], A), which is what merge() folds into base.
    requires_grad is left alone; set_trainable decides it."""

    def __init__(self, conv, rank=16, alpha=32):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("ConvLoRA wraps nn.Conv2d, got %s" % type(conv).__name__)
        if conv.groups != 1:
            raise ValueError("ConvLoRA needs groups == 1 (got %d)" % conv.groups)
        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = conv
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        w = conv.weight
        self.lora_A = nn.Conv2d(conv.in_channels, rank, conv.kernel_size, stride=conv.stride,
                                padding=conv.padding, dilation=conv.dilation, groups=1,
                                bias=False, padding_mode=conv.padding_mode).to(w.device, w.dtype)
        self.lora_B = nn.Conv2d(rank, conv.out_channels, 1, bias=False).to(w.device, w.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    @torch.no_grad()
    def merged_weight(self):
        """base.weight + scaling * B A, computed in float64 and rounded once."""
        a = self.lora_A.weight.double()
        b = self.lora_B.weight[:, :, 0, 0].double()
        delta = torch.einsum("or,rikl->oikl", b, a) * self.scaling
        return (self.base.weight.double() + delta).to(self.base.weight.dtype)


def is_adapter_name(name):
    return any(k in name for k in ADAPTER_KEYS)


def _detection_model(model):
    """The DetectionModel inside a YOLO wrapper, or the model itself; its last
    layer must be a Detect head (YOLO11 layout)."""
    net = model.model if isinstance(model, Model) else model
    layers = getattr(net, "model", None)
    if not isinstance(layers, nn.Sequential) or not isinstance(layers[-1], Detect):
        raise TypeError("expected a YOLO detection model whose last layer is Detect")
    return net


def detect_prefix(model):
    """'model.<i>.' of the Detect layer, i.e. the last module of model.model."""
    net = _detection_model(model)
    return "model.%d." % (len(net.model) - 1)


def _is_3x3(m):
    return isinstance(m, nn.Conv2d) and tuple(m.kernel_size) == (3, 3)


def _wrappable(m):
    return _is_3x3(m) and m.groups == 1 and tuple(m.dilation) == (1, 1)


def _target_convs(model, targets):
    """(name, conv) of every 3x3 conv in the target layers, before injection."""
    net = _detection_model(model)
    det = len(net.model) - 1
    if targets == "backbone_neck":
        idxs = list(range(det))
    elif isinstance(targets, str):
        raise ValueError("unknown LoRA targets %r" % targets)
    else:
        idxs = sorted(set(int(i) for i in targets))
        if any(i < 0 or i >= det for i in idxs):
            raise ValueError("LoRA targets must be layer indices in 0..%d (Detect is %d)" % (det - 1, det))
    if any(isinstance(m, ConvLoRA) for m in net.modules()):
        raise ValueError("model already holds ConvLoRA adapters")
    return [("model.%d.%s" % (i, sub), m) for i in idxs
            for sub, m in net.model[i].named_modules() if sub and _is_3x3(m)]


def unwrapped_3x3(model, targets="backbone_neck"):
    """3x3 convs in the target layers that inject() leaves without an adapter
    (grouped or dilated), so they stay frozen. Call it before inject()."""
    return [n for n, m in _target_convs(model, targets) if not _wrappable(m)]


def inject(model, rank=16, alpha=32, targets="backbone_neck"):
    """Wrap every dense 3x3 conv (groups 1, dilation 1) in the chosen layers in
    ConvLoRA and return the wrapped names (e.g. 'model.0.conv').

    targets: 'backbone_neck' = every layer except the Detect layer, or an
    iterable of layer indices (the Detect layer is never allowed: it is
    trained in full)."""
    net = _detection_model(model)
    names = [n for n, m in _target_convs(net, targets) if _wrappable(m)]
    for name in names:
        parent, attr = name.rsplit(".", 1)
        holder = net.get_submodule(parent)
        setattr(holder, attr, ConvLoRA(getattr(holder, attr), rank=rank, alpha=alpha))
    return names


@torch.no_grad()
def merge(model):
    """Replace every ConvLoRA in model (any nn.Module) by a plain nn.Conv2d with
    weight W + scaling * B A and the base's bias; returns the merged names."""
    if isinstance(model, ConvLoRA):
        raise TypeError("merge replaces ConvLoRA modules inside a model; pass the container")
    names = [n for n, m in model.named_modules() if isinstance(m, ConvLoRA)]
    for name in names:
        parent, attr = name.rsplit(".", 1) if "." in name else ("", name)
        holder = model.get_submodule(parent) if parent else model
        lora = getattr(holder, attr)
        conv = copy.deepcopy(lora.base)
        conv.weight.copy_(lora.merged_weight())
        setattr(holder, attr, conv)
    return names


def set_trainable(model):
    """requires_grad True for adapter parameters and the Detect layer (except
    its DFL conv, which Ultralytics keeps fixed), False for everything else.
    Returns parameter counts."""
    net = _detection_model(model)
    head = detect_prefix(net)
    counts = {"trainable_params": 0, "total_params": 0, "adapter_params": 0, "head_params": 0}
    for n, p in net.named_parameters():
        adapter = is_adapter_name(n)
        in_head = n.startswith(head) and ".dfl" not in n
        p.requires_grad_(adapter or in_head)
        counts["total_params"] += p.numel()
        if adapter or in_head:
            counts["trainable_params"] += p.numel()
        counts["adapter_params"] += p.numel() if adapter else 0
        counts["head_params"] += p.numel() if in_head else 0
    counts["base_params"] = counts["total_params"] - counts["adapter_params"]
    return counts


def frozen_bn_names(model):
    """BatchNorm modules outside the Detect layer."""
    net = _detection_model(model)
    head = detect_prefix(net)
    return [n for n, m in net.named_modules()
            if isinstance(m, nn.BatchNorm2d) and not (n + ".").startswith(head)]


def check_transfer(model, weights):
    """Every state tensor outside the Detect layer must have arrived from
    weights verbatim; the Detect layer may be partly new (a different nc
    changes its class branch). Raises otherwise."""
    src = weights["model"] if isinstance(weights, dict) else weights
    ssd = src.state_dict()
    head = detect_prefix(model)
    bad, reinit, n = [], [], 0
    for k, v in model.state_dict().items():
        ok = k in ssd and ssd[k].shape == v.shape and torch.equal(ssd[k].to(v.device, v.dtype), v)
        n += ok
        if not ok:
            (reinit if k.startswith(head) else bad).append(k)
    if bad:
        raise RuntimeError("LoRA base is not the source network: %d tensor(s) outside the Detect "
                           "layer were not transferred, first %s" % (len(bad), bad[:5]))
    return {"transferred": n, "total": n + len(reinit), "detect_reinit": reinit}


def nonfinite_tensors(model):
    """State keys of floating tensors holding NaN or Inf."""
    return [k for k, v in model.state_dict().items()
            if v.is_floating_point() and not bool(torch.isfinite(v).all())]


# ------------------------------------------------------------------ trainer
# The Ultralytics methods LoRADetectionTrainer overrides. If one is renamed in
# a later release the override would silently never run, so import fails.
_REQUIRED = ("get_model", "build_optimizer", "_setup_train", "_model_train", "save_model")


def _check_internals(trainer_cls):
    missing = [m for m in _REQUIRED if not callable(getattr(trainer_cls, m, None))]
    if missing:
        raise RuntimeError("inc.lora relies on DetectionTrainer.%s, missing in ultralytics %s"
                           % (", ".join(missing), ULTRALYTICS_VERSION))


_check_internals(DetectionTrainer)


class LoRADetectionTrainer(DetectionTrainer):
    """DetectionTrainer that trains ConvLoRA adapters + the Detect layer.

    lora: {"rank": 16, "alpha": 32, "targets": "backbone_neck"}. Single device
    only (a DDP subprocess would rebuild the trainer without the lora
    settings); no resume, no compile, no args.freeze, and no 'auto' optimizer
    (it ignores lr0)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, lora=None):
        lora = dict(lora or {})
        self.lora_cfg = {"rank": int(lora.pop("rank", 16)), "alpha": float(lora.pop("alpha", 32)),
                         "targets": lora.pop("targets", "backbone_neck")}
        if lora:
            raise ValueError("unknown lora keys %s" % sorted(lora))
        super().__init__(cfg, overrides, _callbacks)
        a = self.args
        if a.resume:
            raise ValueError("LoRA runs do not resume")
        if a.compile:
            raise ValueError("LoRA runs do not support compile")
        if a.freeze not in (None, 0, []):
            raise ValueError("LoRA defines its own trainable set; freeze must be unset")
        if str(a.optimizer).lower() == "auto":
            raise ValueError("optimizer='auto' ignores lr0; pass it explicitly (e.g. 'SGD')")
        if self.ddp or self.world_size > 1:
            raise ValueError("LoRA runs are single-device")
        self.merged = self.wdir / MERGED_NAME
        self.lora_wrapped = []
        self.lora_unwrapped = []
        self.lora_saved_epoch = None
        self.lora_skipped_saves = []
        self.lora_transfer = None
        self.lora_params = None
        self._frozen_bn = set()
        self._bn_epoch = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise ValueError("LoRA needs pretrained weights (got none)")
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.lora_transfer = check_transfer(model, weights)
        self.lora_unwrapped = unwrapped_3x3(model, targets=self.lora_cfg["targets"])
        self.lora_wrapped = inject(model, rank=self.lora_cfg["rank"], alpha=self.lora_cfg["alpha"],
                                   targets=self.lora_cfg["targets"])
        if not self.lora_wrapped:
            raise RuntimeError("no conv matched the LoRA targets")
        self._frozen_bn = set(frozen_bn_names(model))
        return model

    def build_optimizer(self, model, *args, **kwargs):
        # First hook after _setup_train's loop that re-enables requires_grad on
        # everything outside args.freeze; also re-run if Ultralytics rebuilds
        # the pipeline after a CUDA OOM.
        self.lora_params = set_trainable(model)
        opt = super().build_optimizer(model, *args, **kwargs)
        for g in opt.param_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        return opt

    def _setup_train(self):
        super()._setup_train()
        trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        in_opt = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        if not self.lora_wrapped or self.lora_params is None or in_opt != trainable:
            raise RuntimeError("LoRA setup did not take effect (adapters %d, optimizer params %d, "
                               "trainable %d)" % (len(self.lora_wrapped), len(in_opt), len(trainable)))

    def _model_train(self):
        super()._model_train()
        for n, m in self.model.named_modules():
            if n in self._frozen_bn:
                m.eval()
        self._bn_epoch = self.epoch

    def save_model(self):
        """Ultralytics' save, then the merged checkpoint. Returns False when
        Ultralytics skipped the save (8.4.37+: non-finite EMA), True otherwise."""
        if self._bn_epoch != self.epoch:
            raise RuntimeError("Ultralytics did not call _model_train this epoch; frozen BN "
                               "statistics were not protected")
        ok = super().save_model()
        # 8.4.21/8.4.22 return None and always write; 8.4.37 returns a bool.
        if ok is not None and not isinstance(ok, bool):
            raise RuntimeError("BaseTrainer.save_model returned %r in ultralytics %s; inc.lora expects "
                               "None or a bool" % (ok, ULTRALYTICS_VERSION))
        if ok is False:
            # last.pt was not rewritten, so the merged file from the same epoch stays.
            self.lora_skipped_saves.append(self.epoch)
            return False
        if self.best.exists():
            self.best.unlink()
        self._save_merged()
        self.lora_saved_epoch = self.epoch
        return True

    def _save_merged(self):
        m = copy.deepcopy(self.ema.ema).to("cpu").float()
        merged = merge(m)
        foreign = sorted({type(x).__module__ for x in m.modules()
                          if not type(x).__module__.startswith(("torch.", "ultralytics."))})
        if foreign:
            raise RuntimeError("merged model still holds modules from %s" % foreign)
        bad = nonfinite_tensors(m)
        if bad:
            raise RuntimeError("epoch %d: the merged EMA holds NaN/Inf in %d tensor(s), first %s; "
                               "%s was not written" % (self.epoch, len(bad), bad[:5], self.merged))
        if hasattr(m, "args"):
            m.args = dict(m.args)
        m.criterion = None
        for p in m.parameters():
            p.requires_grad_(False)
        args = {**DEFAULT_CFG_DICT, **vars(self.args)}
        ckpt = {
            "date": datetime.now().isoformat(),
            "version": ULTRALYTICS_VERSION,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "epoch": -1,
            "best_fitness": None,
            "model": m,
            "ema": None,
            "updates": None,
            "optimizer": None,
            "scaler": None,
            "train_args": {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS},
            "train_metrics": {**(self.metrics or {}), "fitness": self.fitness},
            "lora": {**self.lora_cfg, "merged": merged, "from_epoch": self.epoch,
                     **(self.lora_params or {})},
        }
        tmp = self.merged.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, self.merged)


# -------------------------------------------------------------- entry point
def _checkpoint_epoch(ckpt, path):
    """The training epoch (0-based) an Ultralytics checkpoint was saved at.
    final_eval's strip_optimizer sets 'epoch' to -1, so a stripped last.pt is
    read through train_results, results.csv as of that save, whose 'epoch'
    column is 1-based."""
    epoch = ckpt.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch
    col = (ckpt.get("train_results") or {}).get("epoch") or []
    if not col:
        raise RuntimeError("cannot tell which epoch %s holds: no 'epoch' and no train_results" % path)
    return int(round(float(max(col)))) - 1


def check_saved(merged, last):
    """The merged checkpoint must be finite and come from the epoch that
    Ultralytics' last.pt holds. Raises otherwise; returns that epoch."""
    ck = torch.load(merged, map_location="cpu", weights_only=False)
    bad = nonfinite_tensors(ck["model"])
    if bad:
        raise RuntimeError("%s holds NaN/Inf in %d tensor(s), first %s" % (merged, len(bad), bad[:5]))
    from_epoch = (ck.get("lora") or {}).get("from_epoch")
    if not isinstance(from_epoch, int):
        raise RuntimeError("%s has no lora.from_epoch" % merged)
    last_epoch = _checkpoint_epoch(torch.load(last, map_location="cpu", weights_only=False), last)
    if from_epoch != last_epoch:
        raise RuntimeError("%s is from epoch %d but %s from epoch %d" % (merged, from_epoch, last, last_epoch))
    return from_epoch


def train_lora(init_weights, data_yaml, project, name, epochs=30, lr0=0.01, seed=0, imgsz=640,
               batch=32, device=None, workers=8, rank=16, alpha=32, **extra):
    """Train LoRA adapters + the Detect layer from init_weights and return the
    path of the merged checkpoint (weights/last_merged.pt, loadable with plain
    YOLO()). The unmerged weights/last.pt is kept. <save_dir>/lora.json records
    the adapter settings, the wrapped and the skipped 3x3 convs, the weight
    transfer, the trainable / total parameter counts and the epoch the merged
    checkpoint comes from.

    extra: any other Ultralytics train argument (the INC_LORA_DEFAULTS keys
    included). Keys this function sets from its own arguments (model, data,
    project, name, task, mode, ...) are refused, so the run cannot start from
    weights or data other than the ones lora.json records."""
    for k in ("lora", "freeze", "resume"):
        if extra.get(k):
            raise ValueError("train_lora does not take %r" % k)
    own = dict(model=str(init_weights), data=str(data_yaml), project=str(project), name=name,
               epochs=epochs, lr0=lr0, seed=seed, imgsz=imgsz, batch=batch, workers=workers,
               task="detect", mode="train")
    clash = sorted(set(extra) & (set(own) | {"device"}))
    if clash:
        raise ValueError("train_lora sets %s from its own arguments; do not pass them in **extra" % clash)
    overrides = dict(INC_LORA_DEFAULTS)
    overrides.update(own)
    if device is not None:
        overrides["device"] = device
    overrides.update(extra)

    trainer = LoRADetectionTrainer(overrides=overrides, lora={"rank": rank, "alpha": alpha})
    trainer.train()
    if not trainer.merged.exists():
        raise RuntimeError("training finished without %s" % trainer.merged)
    from_epoch = check_saved(trainer.merged, trainer.last)
    if from_epoch != trainer.lora_saved_epoch:
        raise RuntimeError("%s is from epoch %d, the trainer last saved epoch %s"
                           % (trainer.merged, from_epoch, trainer.lora_saved_epoch))
    if from_epoch != trainer.epochs - 1:
        LOGGER.warning("inc.lora: Ultralytics skipped the save of epoch(s) %s; %s and last.pt are from "
                       "epoch %d of %d" % (trainer.lora_skipped_saves, MERGED_NAME, from_epoch + 1,
                                           trainer.epochs))

    report = {
        "init_weights": str(init_weights),
        "data": str(data_yaml),
        "rank": trainer.lora_cfg["rank"],
        "alpha": trainer.lora_cfg["alpha"],
        "targets": trainer.lora_cfg["targets"],
        "n_wrapped": len(trainer.lora_wrapped),
        "wrapped": trainer.lora_wrapped,
        "unwrapped_3x3": trainer.lora_unwrapped,
        "epochs": trainer.epochs,
        "merged_from_epoch": from_epoch,
        "skipped_saves": trainer.lora_skipped_saves,
        **trainer.lora_params,
        "transferred": trainer.lora_transfer["transferred"],
        "state_entries": trainer.lora_transfer["total"],
        "detect_reinit": trainer.lora_transfer["detect_reinit"],
        "last": str(trainer.last),
        "merged": str(trainer.merged),
        "merged_dtype": "float32",
        "ultralytics_version": ULTRALYTICS_VERSION,
        "torch_version": torch.__version__,
    }
    out = Path(trainer.save_dir) / LORA_JSON
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, out)
    return str(trainer.merged)
