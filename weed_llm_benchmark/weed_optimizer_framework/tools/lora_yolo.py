"""
LoRA for YOLO — Low-Rank Adaptation for Conv2d layers.

Real LoRA implementation for YOLO11n's Conv2d layers in the detection head.
Original Conv weights stay frozen; small low-rank adapters are trained.

This is the closest open-source implementation to professor's request.
Caveats:
- Vanilla LoRA was designed for Linear layers (transformers)
- For Conv2d we use the "ConvLoRA" decomposition: factor 4D weight as 2D
- Only the detection head Conv layers get LoRA (backbone+neck stay frozen)
- Falls back to head-only training if LoRA injection fails

Reference: Wang et al. "Low-rank adaptation for edge AI" Nature Sci Reports 2025
"""

import os
import gc
import logging
import torch
import torch.nn as nn
from pathlib import Path
from ..config import Config

logger = logging.getLogger(__name__)


class ConvLoRA(nn.Module):
    """Wraps a Conv2d layer with a low-rank adapter (parallel path).

    Forward: y = original_conv(x) + lora_B(lora_A(x))
    Original conv is frozen; only lora_A and lora_B are trained.
    """

    def __init__(self, original_conv: nn.Conv2d, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original_conv = original_conv
        # Freeze original
        for p in self.original_conv.parameters():
            p.requires_grad = False

        in_ch = original_conv.in_channels
        out_ch = original_conv.out_channels
        kernel = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding

        # LoRA: down-project (in_ch → rank), up-project (rank → out_ch)
        # Use 1x1 convs for the rank projection (smallest spatial extent)
        self.lora_A = nn.Conv2d(in_ch, rank, kernel_size=kernel, stride=stride,
                                padding=padding, bias=False)
        self.lora_B = nn.Conv2d(rank, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

        # Initialize: A random Gaussian, B zero (so initial LoRA contribution = 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / rank
        self.rank = rank

    def forward(self, x):
        return self.original_conv(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora_into_yolo(yolo_model, target_layers="head", rank=16, alpha=32.0):
    """Inject LoRA adapters into YOLO Conv2d layers.

    Args:
        yolo_model: ultralytics.YOLO instance
        target_layers:
            "head" — only detection head (layers 20-22)
            "backbone" — backbone + neck (layers 0-21), LoRA preserves old knowledge
            "hybrid" — LoRA on backbone+neck, head stays fully trainable (Gemini recommendation)
            "all" — all conv layers
        rank: LoRA rank (smaller = fewer params, more conservative)
        alpha: LoRA scaling factor

    Returns:
        list of (parent_module, attr_name) for injected modules
    """
    torch_model = yolo_model.model  # underlying nn.Module
    injected = []

    # Find Conv2d modules to wrap
    for name, module in list(torch_model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue

        # Filter by target strategy
        if target_layers == "head":
            if not any(p in name for p in [".22.", ".21.", ".20."]):
                continue
        elif target_layers == "backbone":
            # LoRA on backbone+neck (layers 0-21), skip head
            if any(p in name for p in [".22."]):
                continue
        elif target_layers == "hybrid":
            # LoRA ONLY on backbone+neck (layers 0-21)
            # Head layers are left as-is (fully trainable via freeze < 22)
            if any(p in name for p in [".22.", ".21.", ".20."]):
                continue
        # "all" = no filter

        # Skip 1x1 convs (already small)
        if module.kernel_size == (1, 1):
            continue

        # Wrap in LoRA
        try:
            lora_conv = ConvLoRA(module, rank=rank, alpha=alpha)
            # Replace the module in its parent
            parent_path = name.rsplit(".", 1)
            if len(parent_path) == 2:
                parent_name, attr = parent_path
                parent = torch_model.get_submodule(parent_name)
                setattr(parent, attr, lora_conv)
                injected.append((parent_name, attr))
        except Exception as e:
            logger.warning(f"Could not inject LoRA into {name}: {e}")
            continue

    logger.info(f"[LoRA] Injected adapters into {len(injected)} Conv2d layers (rank={rank})")
    return injected


def count_trainable_params(model):
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_yolo_with_lora(strategy, label_dir, iteration):
    """Train YOLO with LoRA adapters injected into the detection head.

    This is the LoRA-style approach: most weights frozen, only small
    rank-r adapters are trained. Should preserve old species knowledge
    while allowing new species adaptation.
    """
    from ultralytics import YOLO
    from .yolo_trainer import _assemble_dataset, _count_nonempty_labels

    n_labels = _count_nonempty_labels(label_dir)
    if n_labels < Config.MIN_CONSENSUS_BOXES:
        raise ValueError(f"Too few labels ({n_labels} < {Config.MIN_CONSENSUS_BOXES})")

    ds_dir, data_yaml, ds_stats = _assemble_dataset(strategy, label_dir, iteration)

    base_weights = Config.YOLO_8SP_WEIGHTS
    if not os.path.exists(base_weights):
        raise FileNotFoundError(f"Base weights not found: {base_weights}")

    model = YOLO(base_weights)

    # LoRA strategy
    rank = strategy.get("lora_rank", 64)
    alpha = strategy.get("lora_alpha", 128.0)
    lora_mode = strategy.get("lora_mode", "hybrid")  # "head", "backbone", "hybrid"

    try:
        injected = inject_lora_into_yolo(model, target_layers=lora_mode, rank=rank, alpha=alpha)
        if injected:
            trainable, total = count_trainable_params(model.model)
            logger.info(f"[LoRA] Trainable params: {trainable:,}/{total:,} "
                        f"({100*trainable/total:.2f}%)")
            mode = f"lora_rank{rank}"
        else:
            logger.warning("[LoRA] No layers injected, falling back to head-only training")
            mode = "head_only_fallback"
    except Exception as e:
        logger.warning(f"[LoRA] Injection failed: {e}, falling back to head-only freeze")
        mode = "head_only_fallback"

    # Train (Ultralytics handles the loop)
    project_dir = os.path.join(Config.FRAMEWORK_DIR, f"yolo_lora_iter{iteration}")
    logger.info(f"Training YOLO with {mode}: lr={strategy.get('lr', 0.001)}, "
                f"epochs={strategy.get('epochs', 50)}")

    try:
        # For LoRA: low LR (LoRA params are randomly initialized for A, zero for B)
        # Use freeze=22 to ensure backbone+neck stays frozen even if injection partial
        # Hybrid mode: freeze=20 (backbone+neck frozen via LoRA, head fully trainable)
        # Head-only mode: freeze=22 (everything frozen except LoRA adapters in head)
        freeze_val = 20 if lora_mode == "hybrid" else 22

        model.train(
            data=data_yaml,
            epochs=strategy.get("epochs", 50),
            batch=strategy.get("batch_size", -1),
            device="cuda" if torch.cuda.is_available() else "cpu",
            project=project_dir,
            name="train",
            patience=strategy.get("patience", 15),
            lr0=strategy.get("lr", 0.0005),
            freeze=freeze_val,
            workers=4,
            verbose=False,
        )
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        import shutil
        shutil.rmtree(ds_dir, ignore_errors=True)

    best_pt = os.path.join(project_dir, "train", "weights", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"LoRA training failed: {best_pt} not found")

    logger.info(f"[LoRA] Training complete: {best_pt}")
    return best_pt
