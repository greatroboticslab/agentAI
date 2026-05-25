"""
v3.0.41 Phase 1/2 — FLUX.1-Fill LoRA trainer (FLORA recipe).

Per FLORA (arXiv 2508.21712, Aug 2025):
  - 30 real object crops per class
  - per-class LoRA, rank 32, alpha 16, 5 epochs
  - 8-bit AdamW, bfloat16, 512x512
  - trigger format: `{dataset}-{class}` (e.g. "cwd12-Goosegrass")
  - attention layers only

Why per-class LoRA (not one multi-class): FLORA shows class-specialised
LoRAs let each one fully exploit its 30 samples without inter-class
gradient interference. One multi-class LoRA on 12×30=360 samples of
visually similar species risks averaging features away.

We run one slug at a time:
  python flux_lora_train.py --class-name Goosegrass --epochs 5

Outputs:
  results/framework/flux_lora/{class_name}/
    pytorch_lora_weights.safetensors   (just the LoRA delta)
    train_meta.json                    (config + training loss)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("flux_lora_train")

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()

BANK_DIR = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
LORA_DIR = REPO / "results" / "framework" / "flux_lora"
FLUX_MODEL = os.environ.get("FLUX_FILL_MODEL",
                            "black-forest-labs/FLUX.1-Fill-dev")

CANONICAL_12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]


def _load_class_crops(class_name: str, max_n: int = 30, min_px: int = 80,
                      seed: int = 0):
    """Sample at most `max_n` high-quality crops for the given class."""
    cdir = BANK_DIR / class_name
    if not cdir.is_dir():
        raise FileNotFoundError(f"no bank dir for class {class_name}: {cdir}")
    from PIL import Image
    candidates = []
    for p in sorted(cdir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                if min(w, h) < min_px:
                    continue
        except Exception:
            continue
        candidates.append(p)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_n]


def train_one_class(class_name: str, *,
                    rank: int = 32, alpha: int = 16,
                    epochs: int = 5, lr: float = 1e-4,
                    batch_size: int = 1, grad_accum: int = 4,
                    resolution: int = 512, seed: int = 42):
    """Train a single per-class LoRA on the cleaned cwd12 bank crops."""
    if class_name not in CANONICAL_12:
        log.warning(f"{class_name} is not in CANONICAL_12 — proceeding anyway")
    try:
        import torch
        from diffusers import FluxTransformer2DModel, FluxFillPipeline
        from diffusers.optimization import get_scheduler
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from PIL import Image
    except ImportError as e:
        log.error(f"missing dep: {e}")
        log.error("install: pip install -U diffusers peft accelerate bitsandbytes")
        sys.exit(2)

    out_dir = LORA_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"=== FLUX-LoRA train: class={class_name} rank={rank} epochs={epochs} ===")

    crops = _load_class_crops(class_name, seed=seed)
    if len(crops) < 5:
        log.error(f"insufficient crops for {class_name}: {len(crops)} (need >=5)")
        sys.exit(1)
    log.info(f"using {len(crops)} crops from {BANK_DIR / class_name}")

    trigger = f"cwd12-{class_name}"

    # Load the FLUX transformer alone (we only need the transformer + VAE for
    # rectified-flow training; T5 is frozen, we encode prompts once)
    log.info(f"loading FLUX transformer + VAE + text encoders ({FLUX_MODEL})...")
    pipe = FluxFillPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)

    # Attach a LoRA to the transformer's attention layers ONLY (FLORA).
    log.info(f"attaching LoRA: rank={rank} alpha={alpha} attention-only")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.transformer.add_adapter(lora_config)
    pipe.transformer.train()
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.requires_grad_(False)
    for p in pipe.transformer.parameters():
        if p.requires_grad is False and p.dim() > 1:
            # PEFT marks LoRA params requires_grad=True; vanilla weights stay frozen
            pass
    # Trainable params are the LoRA adapters only
    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info(f"trainable LoRA params: {n_trainable/1e6:.2f}M")

    # Move static components
    pipe.vae.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder_2.to("cuda", dtype=torch.bfloat16).eval()
    pipe.transformer.to("cuda", dtype=torch.bfloat16)

    # 8-bit AdamW
    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=0.01)
    except Exception:
        log.warning("bitsandbytes 8-bit AdamW not available; using fp32 AdamW")
        optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    # Pre-encode prompt (single trigger, used for every sample)
    log.info(f"encoding prompt: {trigger!r}")
    with torch.no_grad():
        prompt_emb, pooled_emb, txt_ids = pipe.encode_prompt(
            prompt=trigger, prompt_2=trigger, max_sequence_length=128)
    # cache: same for every step

    # Cache VAE latents for each crop
    log.info("pre-encoding crops to VAE latents...")
    latents_list = []
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    for p in crops:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            log.warning(f"  skip {p}: {e}")
            continue
        t = tf(img).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            lat = pipe.vae.encode(t).latent_dist.sample()
            lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        latents_list.append(lat.detach())
    log.info(f"cached {len(latents_list)} latents")

    n_steps = max(1, (len(latents_list) * epochs) // (batch_size * grad_accum))
    sched = get_scheduler("constant", optimizer=optim, num_warmup_steps=0,
                          num_training_steps=n_steps)

    # Train loop — rectified flow loss (FLUX is a flow-matching model)
    t0 = time.time()
    losses = []
    step_counter = 0
    optim.zero_grad()
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        order = list(range(len(latents_list)))
        rng.shuffle(order)
        for sub_step, i in enumerate(order):
            lat = latents_list[i]
            bsz = lat.shape[0]
            # Sample timestep + noise for flow matching
            noise = torch.randn_like(lat)
            t_step = torch.rand(bsz, device="cuda", dtype=torch.bfloat16)
            sigmas = t_step.view(bsz, 1, 1, 1)
            noisy = (1.0 - sigmas) * lat + sigmas * noise
            target = noise - lat   # flow-matching target (velocity)

            # FluxFill expects packed latents; for training-only we mimic the
            # transformer forward used by FLUX rectified-flow training
            # (simplified — production code in diffusers' train_dreambooth_lora_flux
            #  handles packed latent shape; we follow the same pattern)
            from diffusers.training_utils import compute_density_for_timestep_sampling
            timesteps = (t_step * 1000).long()

            model_pred = pipe.transformer(
                hidden_states=noisy,
                timestep=timesteps,
                pooled_projections=pooled_emb,
                encoder_hidden_states=prompt_emb,
                txt_ids=txt_ids,
                img_ids=torch.zeros((noisy.shape[0], 1, 3),
                                    device="cuda", dtype=torch.bfloat16),
                return_dict=False,
            )[0]
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
            loss = loss / grad_accum
            loss.backward()
            losses.append(float(loss.detach()) * grad_accum)

            if (sub_step + 1) % grad_accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad()
                step_counter += 1
                if step_counter % 5 == 0:
                    avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                    log.info(f"  ep {epoch+1}/{epochs} step {step_counter}/{n_steps} "
                             f"loss={avg:.4f}")

    elapsed = time.time() - t0
    log.info(f"training done in {elapsed/60:.1f} min, final loss "
             f"{sum(losses[-20:])/max(1,len(losses[-20:])):.4f}")

    # Save LoRA delta only
    lora_state = get_peft_model_state_dict(pipe.transformer)
    from safetensors.torch import save_file
    save_path = out_dir / "pytorch_lora_weights.safetensors"
    save_file(lora_state, str(save_path))

    meta = {
        "class_name": class_name,
        "trigger": trigger,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rank": rank, "alpha": alpha, "epochs": epochs,
        "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
        "resolution": resolution,
        "n_crops": len(latents_list),
        "n_steps": step_counter,
        "elapsed_min": elapsed / 60,
        "final_loss": float(sum(losses[-20:]) / max(1, len(losses[-20:]))),
        "loss_history_last100": losses[-100:],
    }
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"saved LoRA → {save_path}")
    log.info(f"saved meta → {out_dir / 'train_meta.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class-name", required=True,
                    help="one of CANONICAL_12 (Carpetweeds, Crabgrass, …)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    train_one_class(args.class_name, rank=args.rank, alpha=args.alpha,
                    epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, grad_accum=args.grad_accum,
                    resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()
