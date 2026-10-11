"""V3F stage 2: D2, the stage-2 backbone ablation on E3-B's boxes, and D9, the
classifier's fit-sample sensitivity (docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered as one
family)", V3F.5 D2 and D9; kept byte for byte as docs/prereg/V3F_2026-10-11.md).

    python -m weed_optimizer_framework.tools.inc2.v3fam_embed score --arm D2LAION|D2CLIP|D2DINOL|D2DINOB|D2SIGLIP|
                                                                          D2RN50|D2BC1|D2BC2640|D2FT|D9
        (L23J: inc2.v3fam score --arm X)
    python -m weed_optimizer_framework.tools.inc2.v3fam_embed plan                            (the delegate, step 0c)
    python -m weed_optimizer_framework.tools.inc2.v3fam_embed downloads --commands LOG       (the delegate, step 0c)
    python -m weed_optimizer_framework.tools.inc2.v3fam_embed status
        [--batch N --device D --no-lock-check-for-tests]   (test mode only, INC_SCORER_TESTING=1)

The embedder registry (BACKBONES). Each entry pins a Hugging Face repository,
its revision and the sha256 of the weight file its loader opens, as V3F.5 D2
records them (W1, W2); DINOv2-L, DINOv2-B and ResNet-50 a1 have no revision in
the amendment and take theirs from v3fam/v1/downloads.json, the delegate's
record of the default cache after the downloads (record_downloads). At load
every pinned file of the snapshot is hashed; a loader that resolves by
refs/main (open_clip) needs refs/main at the pinned revision; the weight file
the loader would open (open_clip 3.3.0: open_clip_model.safetensors when it
exists and safetensors is importable, else open_clip_pytorch_model.bin;
transformers: model.safetensors first) must be a pinned one; anything else
refuses, and the arm is "not run" with no substitute. Every embedder takes
E3's 224-px crop as it is (no resize, no centre crop), with its own
normalisation, fp16 autocast on CUDA, and exposes name, dim and
__call__(pils), the interface E3's validator duck-types; E3's path
L2-normalises the features (verify._norm):
  BC2     BioCLIP-2, the reference: verify.BioclipEmbedder itself (E3's)
  LAION   CLIP ViT-L/14 LAION-2B, BioCLIP-2's starting checkpoint (P1's
          control): through semisup_labeler._load_backbone and _embed_batch,
          as E3's embedder (OpenClipEmbedder)
  CLIP    CLIP ViT-L/14 OpenAI: the projected image embedding
  DINOL   DINOv2-L, DINOB DINOv2-B: the CLS token
  SIGLIP  SigLIP2 so400m/14 at 224: the pooled image embedding
  RN50    ResNet-50 a1: the global-average-pooled features
  BC1     BioCLIP-1 (ViT-B/16), as LAION
The arms (ARM_BACKBONE): D2LAION, D2CLIP, D2DINOL, D2DINOB, D2SIGLIP, D2RN50,
D2BC1 (frozen backbone + E3's LR protocol on E3's crops); D2BC2640 (BioCLIP-2
on crops cut from the 640-px view stage 1 sees, view640, for the training
crops and the detected boxes alike; a new LR fit); D2FT (ResNet-50 a1 fine-
tuned end to end, classifier seed s paired with stage-1 seed s).

D2's job (score_d2, one per arm): E3's 25,697 ground-truth crops
(twostage/e3_v1/classifier/train_crops.csv by the sha256 E3's pin records,
base_v2 through E3's own never-train checks), cut by verify._cut_task
(view640: cut640_task) and embedded; the 13-class logistic regression by E3's
protocol (twostage.cross_validate, twostage._fit_lr: lbfgs, max_iter 3000, no
class weights, C by 5-fold GroupKFold by session on pooled log-loss, ties to
the smaller C), fitted and pinned once (capacity/v3fam_d2_<key>_pin.json)
before any of the arm's dev passes; then E3-B's seeds 0-2 through the family
two-stage pass (view640 for D2BC2640), each stage-1 box set equal to E3's
stored arms/B/s<k>/dev.stage1.npz or refused, written once; the dev
ground-truth accuracy (family gt mode); capacity/v3fam_score_<arm>.json
(dev file names, sha256s, status and counts; overwritten). D2FT trains
FT-ResNet-50 per seed (train_ft: AdamW lr 3e-4, weight decay 0.05, 1 warm-up
epoch then cosine to 0, 20 epochs, batch 128, label smoothing 0.1, AMP;
flips, RandomResizedCrop(224, 0.7-1.0, 3/4-4/3), ColorJitter(0.2, 0.2, 0.2);
the crops cut once per job into memory, never written; final-epoch weights
written once), pins the three, and names boxes by the network's softmax: its
embedder returns the centred logits as the first 13 entries of a unit vector
(FTEmbedder), so E3's normalisation leaves it unchanged and E3's proba with
W = FT_K [I | 0], b = 0 returns that softmax to float32 rounding.

D9's job (score_d9): D9a, E3's own twostage.sensitivity (the five fold
classifiers at the chosen C on seed 0 of each E3 arm; it writes
twostage/e3_v1/sensitivity.json once, E3's tree's one permitted family
write); D9b, 30 session-stratified half-sample refits of the 13-class LR at
C = 30 on E3's stored fp32 features (half_sample: strata by the rarest class a
session holds, half of each stratum's sessions drawn, odd counts rounding up
and down alternately from a seeded start, a draw missing a class redrawn and
counted), each applied to E3-B's seeds 0-2 through full family passes; the sd
across refits per seed and pooled, beside E3-B's dev seed sd, E3's D's and the
SESOI, and the sentence V3F.5 fixes (v3fam/v1/d9/record.json, written once;
capacity/v3fam_score_d9.json).

Every job checks the six E3-frozen files (v3fam.check_frozen), rests on the
family's equivalence records (v3fam.require_equivalence: dev identity,
injection, two_stage_B) and runs inside v3fam.e_trees_unchanged: nothing is
written under twostage/e3_v1 but D9a's sensitivity.json. Only dev is read.

Layout (INC_DIR/v3fam/v1, written once each unless said):
  downloads.json                          the delegate's record of the cache (step 0c)
  d2/<KEY>/classifier/classifier.json, classifier.npz, train_emb.npz
  d2/<KEY>/dev_gt.json (D2FT: dev_gt.s<k>.json)
  d2/FT/s<k>/ft_resnet50.pt, train.json
  arms/<ARM>/s<k>/dev.json, dev.images.npz, dev.stage1.npz
  arms/D9/s<k>/dev.half<b>.json, .images.npz, .stage1.npz (no per-box features)
  d9/refits/half<b>.npz, half<b>.json; d9/record.json
and in INC_DIR/capacity: v3fam_d2_<key>_pin.json, v3fam_score_<arm>.json
(overwritten).

Public API: BACKBONES, D2_ARMS, ARM_BACKBONE, ARM_KEY, ARM_CUT, FT_ARM, FT_CFG,
FT_K, D9_ARM, D9_C, D9_REFITS; hub_cache, checkpoint, backbone_status,
load_embedder, OpenClipEmbedder, TensorEmbedder, FTEmbedder,
ft_classifier_cfg, view640, cut640_task, e3_cut640, embed_crops, cut_crops,
training_crops, e3_train_features, fit_lr_protocol, load_d2_classifier,
train_ft, half_sample, d9_required, score_d2, score_d9, score,
record_downloads, download_plan, code_shas.

A refusal raises v3fam.V3famRefused; the CLI prints "[inc2.v3fam_embed]
ERROR: ..." and exits 2. Any other error exits 1.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

from ..inc import common as C
from ..inc import scorer as S
from ..inc import verify as V
from .. import semisup_labeler as SL
from . import common as C2
from . import scorer_native as SN
from . import scorer_sidecar as SC
from . import train as T
from . import twostage as TS
from . import v3fam as F

Refused = F.V3famRefused

# ------------------------------------------------------------------ the registry (V3F.5 D2; W1, W2)
NAME = F.NAME
CROP_PX = SL.CROP_PX                         # 224: every backbone is a 224-px model and takes the crop as it is
IMGSZ = TS.IMGSZ                             # the 640-px view stage 1 sees
OPEN_CLIP_WEIGHTS = ("open_clip_model.safetensors", "open_clip_pytorch_model.bin")    # open_clip 3.3.0's order
HF_WEIGHTS = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
TIMM_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
WEIGHT_NAMES = frozenset(OPEN_CLIP_WEIGHTS + HF_WEIGHTS + TIMM_WEIGHTS + ("open_clip_pytorch_model.safetensors",
                                                                          "tf_model.h5", "flax_model.msgpack"))
TIMM_NAME = "resnet50.a1_in1k"
BACKBONES = {
    "BC2": {"repo": "imageomics/bioclip-2", "loader": "open_clip", "dim": 768,
            "revision": "2957b322090f9cb17ae72c71981c7218a28d81e0",
            "weights": {"open_clip_model.safetensors":
                        "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
                        "open_clip_pytorch_model.bin":
                        "035208dd616321f6c220384efaa1f10e9fcd8e1f2998ddefc88687718d186a85"},
            "configs": {"open_clip_config.json": "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8"},
            "pooling": "open_clip encode_image (projected, 768-d)"},
    "LAION": {"repo": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "loader": "open_clip", "dim": 768,
              "revision": "1627032197142fbe2a7cfec626f4ced3ae60d07a",
              "weights": {"open_clip_pytorch_model.bin":
                          "5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b"},
              "configs": {"open_clip_config.json": "de8f2f846f38ccb247b9648512c31ea79b51fde6a623a8d5e85d96cfe10d943a"},
              "pooling": "open_clip encode_image (projected, 768-d)"},
    "CLIP": {"repo": "openai/clip-vit-large-patch14", "loader": "hf_clip", "dim": 768,
             "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
             "weights": {"model.safetensors": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"},
             "configs": {}, "pooling": "CLIPVisionModelWithProjection image_embeds (projected, 768-d)"},
    "DINOL": {"repo": "facebook/dinov2-large", "loader": "hf_dinov2", "dim": 1024, "revision": None, "weights": None,
              "configs": {}, "pooling": "last_hidden_state[:, 0] (CLS, 1,024-d)"},
    "DINOB": {"repo": "facebook/dinov2-base", "loader": "hf_dinov2", "dim": 768, "revision": None, "weights": None,
              "configs": {}, "pooling": "last_hidden_state[:, 0] (CLS, 768-d)"},
    "SIGLIP": {"repo": "google/siglip2-so400m-patch14-224", "loader": "hf_siglip", "dim": 1152,
               "revision": "78e403963a4f6a3640d07803284752326fdf4edf",
               "weights": {"model.safetensors": "2041e5330b3db96299b26bc3a6506c7fa8d5e61d5c94f267dd8aa4cfb12c1050"},
               "configs": {}, "pooling": "SiglipVisionModel pooler_output (attention pooling, 1,152-d)"},
    "RN50": {"repo": "timm/resnet50.a1_in1k", "loader": "timm", "dim": 2048, "revision": None, "weights": None,
             "configs": {}, "pooling": "timm forward with num_classes 0 (global average pool, 2,048-d)"},
    "BC1": {"repo": "imageomics/bioclip", "loader": "open_clip", "dim": 512,
            "revision": "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
            "weights": {"open_clip_pytorch_model.bin":
                        "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"},
            "configs": {}, "pooling": "open_clip encode_image (projected, 512-d)"}}
REFERENCE = "BC2"
# the files the delegate fetches for each backbone (plus configs and processor files), in the default cache
DOWNLOAD_FILES = {
    "LAION": ("open_clip_pytorch_model.bin", "open_clip_config.json"),
    "CLIP": ("model.safetensors", "config.json", "preprocessor_config.json"),
    "SIGLIP": ("model.safetensors", "config.json", "preprocessor_config.json"),
    "BC1": ("open_clip_pytorch_model.bin", "open_clip_config.json"),
    "RN50": ("model.safetensors", "config.json")}

D2_ARMS = ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640", "D2FT")
ARM_BACKBONE = {"D2LAION": "LAION", "D2CLIP": "CLIP", "D2DINOL": "DINOL", "D2DINOB": "DINOB", "D2SIGLIP": "SIGLIP",
                "D2RN50": "RN50", "D2BC1": "BC1", "D2BC2640": "BC2", "D2FT": "RN50"}
ARM_KEY = {a: a[2:] for a in D2_ARMS}       # the classifier's directory and pin name: LAION, ..., BC2640, FT
ARM_CUT = {a: ("view640" if a == "D2BC2640" else "original") for a in D2_ARMS}
SYSTEM_NAMES = {"D2LAION": "E3-B's boxes + a LAION-2B CLIP ViT-L/14 classifier",
                "D2CLIP": "E3-B's boxes + an OpenAI CLIP ViT-L/14 classifier",
                "D2DINOL": "E3-B's boxes + a DINOv2-L classifier", "D2DINOB": "E3-B's boxes + a DINOv2-B classifier",
                "D2SIGLIP": "E3-B's boxes + a SigLIP2 so400m/14 classifier",
                "D2RN50": "E3-B's boxes + a ResNet-50 classifier", "D2BC1": "E3-B's boxes + a BioCLIP-1 classifier",
                "D2BC2640": "E3-B's boxes + BioCLIP-2 on the 640-px view (a new classifier)",
                "D2FT": "E3-B's boxes + FT-ResNet-50", "D9": "E3-B's boxes + a D9b half-sample BioCLIP-2 refit"}
FT_ARM = "D2FT"
E3_ARM = "B"                                # D2 and D9 name E3-B's boxes
SEEDS = F.SEEDS
GT_DETECTOR = ("M", 0)                       # the detector E3's dev ground-truth pass ran (twostage.dev_gt_accuracy)

# FT-ResNet-50 (V3F.5 D2, backbone 10)
FT_CFG = {"epochs": 20, "warmup_epochs": 1, "batch": 128, "lr": 3e-4, "weight_decay": 0.05, "label_smoothing": 0.1,
          "amp": True, "hflip": 0.5, "vflip": 0.5, "rrc_scale": (0.7, 1.0), "rrc_ratio": (3.0 / 4.0, 4.0 / 3.0),
          "jitter": (0.2, 0.2, 0.2), "schedule": "linear warm-up for warmup_epochs, then cosine to 0, per step"}
FT_SEED_TEXT = "inc2/v3fam/d2/ft/s%d"
FT_K = 1.0e4                                 # the logit encoding's norm: a crop's centred logits must stay below it
FT_WEIGHTS = "ft_resnet50.pt"

# D9 (V3F.5 D9)
D9_ARM = "D9"
D9_C = 30.0                                  # E3's chosen C, fixed
D9_REFITS = 30
D9_SEED_TEXT = "inc2/v3fam/d9/half/%d"
D9_MAX_REDRAWS = 10000
D9_SENTENCE = {
    "exceeds": "the pooled refit sd exceeds E3-B's dev seed sd: E3's dev margins understate uncertainty",
    "within": "the classifier's fit sample adds less spread than the seeds"}

FMT_DOWNLOADS = "inc2-v3fam-downloads/1"
FMT_CLASSIFIER = "inc2-v3fam-d2-classifier/1"
FMT_PIN = "inc2-v3fam-d2-pin/1"
FMT_FT_TRAIN = "inc2-v3fam-d2-ft-train/1"
FMT_FT_PIN = "inc2-v3fam-d2-ft-pin/1"
FMT_DEV_GT = "inc2-v3fam-d2-dev-gt/1"
FMT_SCORE_RECORD = "inc2-v3fam-score-record/1"
FMT_D9_REFIT = "inc2-v3fam-d9-refit/1"
FMT_D9 = "inc2-v3fam-d9/1"
NOT_RUN_DOWNLOAD = "not run (download)"


def log(msg):
    print("[inc2.v3fam_embed] %s" % msg, flush=True)


# ------------------------------------------------------------------ paths and records
def root():
    return F.root()


def d2_dir(key):
    return root() / "d2" / key


def clf_dir(key):
    return d2_dir(key) / "classifier"


def pin_path(key):
    return C.INC_DIR / "capacity" / ("v3fam_d2_%s_pin.json" % key.lower())


def score_record_path(arm):
    return C.INC_DIR / "capacity" / ("v3fam_score_%s.json" % arm.lower())


def downloads_path():
    return root() / "downloads.json"


def d9_dir():
    return root() / "d9"


def _rel(path):
    """A path relative to v3fam/v1 (records name files, never absolute paths)."""
    return str(Path(path).relative_to(root()))


def code_shas():
    """The family's code (v3fam.code_shas: the six E3-frozen files and v3fam.py) and this module."""
    out = F.code_shas()
    out["v3fam_embed"] = C.sha256_file(Path(__file__).resolve())
    return out


def _npz_bytes(arrays):
    return TS._npz_bytes(arrays)


# ------------------------------------------------------------------ the cache and the pins
def hub_cache():
    """The Hugging Face cache the loaders read (huggingface_hub's HF_HUB_CACHE: the bench env's default
    ~/.cache/huggingface/hub on the cluster)."""
    try:
        from huggingface_hub import constants as HC
        return Path(HC.HF_HUB_CACHE)
    except Exception:                        # noqa: BLE001 - no huggingface_hub: its own default rule
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"])
        home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        return Path(home) / "hub"


def repo_dir(repo):
    return hub_cache() / ("models--%s" % repo.replace("/", "--"))


def _refs_main(repo):
    p = repo_dir(repo) / "refs" / "main"
    try:
        return p.read_text().strip() or None
    except OSError:
        return None


def _snapshot_files(snap):
    """{relative name: resolved path} of every file of a snapshot (the cache's symlinks followed)."""
    out = {}
    for p in sorted(Path(snap).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(snap))] = p.resolve()
    return out


def _has_safetensors():
    return importlib.util.find_spec("safetensors") is not None


def loader_weight_file(loader, names):
    """The weight file a loader opens among a snapshot's file names: open_clip 3.3.0 opens
    open_clip_model.safetensors when it exists and safetensors is importable, else open_clip_pytorch_model.bin;
    transformers model.safetensors first; timm (loaded here from an explicit file) model.safetensors first."""
    order = {"open_clip": OPEN_CLIP_WEIGHTS if _has_safetensors() else OPEN_CLIP_WEIGHTS[1:],
             "timm": TIMM_WEIGHTS}.get(loader, HF_WEIGHTS)
    for n in order:
        if n in names:
            return n
    return None


def _open_clip_resolves(repo):
    """The file open_clip itself resolves for 'hf-hub:<repo>' offline (open_clip.pretrained.
    download_pretrained_from_hf, the call its factory makes), or None when open_clip is not installed."""
    try:
        from open_clip import pretrained as OP
    except Exception:                        # noqa: BLE001 - not installed (tests, the lab): not cross-checked
        return None
    return Path(OP.download_pretrained_from_hf(repo, cache_dir=str(hub_cache()))).resolve()


def read_downloads():
    """v3fam/v1/downloads.json, or None."""
    d = TS._read_json(downloads_path())
    return d if isinstance(d, dict) and d.get("format") == FMT_DOWNLOADS else None


def pins_for(bb, downloads=None):
    """{repo, loader, revision, weights {file: sha256}, files {file: sha256}, required, source} of a backbone:
    the amendment's revision and weight hashes where it gives them, otherwise downloads.json's. Every other file
    downloads.json recorded for the snapshot (configs, processor files) is pinned and required too, and so are
    the amendment's config hashes; a weight file nobody pinned is left out (the loader must not open it,
    checkpoint). Every backbone but the reference (BioCLIP-2, which the amendment and E3 pin fully) needs
    downloads.json's record; one recorded as anything but ok is not run; a record that contradicts the amendment
    refuses."""
    if bb not in BACKBONES:
        raise Refused("backbone %r is not one of %s" % (bb, sorted(BACKBONES)))
    e = BACKBONES[bb]
    dl = downloads if downloads is not None else read_downloads()
    rec = ((dl or {}).get("backbones") or {}).get(bb)
    if rec is not None and rec.get("status") != "ok":
        raise Refused("%s: %s %s: downloads.json records it %r (%s)" % (NOT_RUN_DOWNLOAD, bb, e["repo"],
                                                                       rec.get("status"), rec.get("why")))
    if rec is None and bb != REFERENCE:
        raise Refused("%s: %s %s has no record in %s (the delegate's step 0c)" % (NOT_RUN_DOWNLOAD, bb, e["repo"],
                                                                                 downloads_path()))
    out = {"backbone": bb, "repo": e["repo"], "loader": e["loader"], "dim": e["dim"]}
    if e["revision"]:
        if rec is not None and rec.get("revision") != e["revision"]:
            raise Refused("%s: downloads.json records revision %s, the amendment pins %s" % (bb, rec.get("revision"),
                                                                                           e["revision"]))
        out.update(revision=e["revision"], weights=dict(e["weights"]), source="amendment")
    else:
        wf = rec.get("weight_file")
        out.update(revision=rec.get("revision"), weights={wf: ((rec.get("files") or {}).get(wf) or {}).get("sha256")},
                   source="downloads.json")
    files, required = {}, set()
    for n, v in sorted(((rec or {}).get("files") or {}).items()):
        if n in WEIGHT_NAMES and n not in out["weights"]:
            continue
        files[n] = v.get("sha256")
        required.add(n)
    for n, sha in sorted((e.get("configs") or {}).items()):
        if n in files and files[n] != sha:
            raise Refused("%s: downloads.json records %s as %s, the amendment pins %s" % (bb, n, files[n][:12], sha[:12]))
        files[n] = sha
        required.add(n)
    for n, sha in sorted(out["weights"].items()):
        if n in files and files[n] != sha:
            raise Refused("%s: downloads.json records %s as %s, the amendment pins %s" % (bb, n, str(files[n])[:12],
                                                                                         str(sha)[:12]))
        files[n] = sha
    out.update(files=files, required=sorted(required))
    return out


def checkpoint(bb, downloads=None):
    """Refuse unless backbone bb's snapshot is the pinned one (module docstring); returns its record: repo,
    revision, snapshot, the weight file the loader opens and its sha256, every pinned file's sha256."""
    p = pins_for(bb, downloads)
    rd = repo_dir(p["repo"])
    snap = rd / "snapshots" / str(p["revision"])
    if not snap.is_dir():
        raise Refused("%s: %s has no snapshot %s in %s" % (NOT_RUN_DOWNLOAD, p["repo"], p["revision"], hub_cache()))
    present = _snapshot_files(snap)
    wf = loader_weight_file(p["loader"], present)
    if wf is None:
        raise Refused("%s: %s@%s holds no weight file the %s loader opens" % (NOT_RUN_DOWNLOAD, p["repo"],
                                                                            p["revision"][:12], p["loader"]))
    if wf not in p["weights"]:
        raise Refused("%s@%s: the %s loader would open %s, which is not a pinned weight file (%s)"
                      % (p["repo"], p["revision"][:12], p["loader"], wf, sorted(p["weights"])))
    refs = _refs_main(p["repo"])
    if p["loader"] == "open_clip" and refs != p["revision"]:
        raise Refused("%s: refs/main is %s, the pinned revision %s: open_clip resolves 'hf-hub:' names by refs/main, "
                      "so it would load another snapshot" % (p["repo"], refs, p["revision"]))
    got = {}
    for n, want in sorted(p["files"].items()):
        if n not in present:
            if n == wf or n in p["required"]:
                raise Refused("%s@%s lacks the pinned file %s" % (p["repo"], p["revision"][:12], n))
            continue                         # the reference's other pinned weight file, which nothing opens
        got[n] = C.sha256_file(present[n])
        if got[n] != want:
            raise Refused("%s@%s: %s hashes to %s, pinned %s" % (p["repo"], p["revision"][:12], n, got[n][:12],
                                                                str(want)[:12]))
    if p["loader"] == "open_clip":
        try:
            oc = _open_clip_resolves(p["repo"])
        except Exception as e:               # noqa: BLE001 - open_clip cannot resolve it offline: refused
            raise Refused("%s: open_clip cannot resolve its weight file offline: %s: %s" % (p["repo"], type(e).__name__,
                                                                                          e))
        if oc is not None and oc != present[wf]:
            raise Refused("%s: open_clip resolves %s, not the pinned %s" % (p["repo"], oc, present[wf]))
    return {"backbone": bb, "repo": p["repo"], "loader": p["loader"], "revision": p["revision"],
            "snapshot": str(snap), "refs_main": refs, "weight_file": wf, "weight_sha256": got[wf],
            "files": got, "pins_from": p["source"]}


def backbone_status(arm, downloads=None):
    """{status ok | not run, why} of a D2 arm's backbone from the pins and downloads.json alone (no file read):
    what the gate may consult before proposing the arm."""
    if arm not in D2_ARMS:
        raise Refused("%s is not a D2 arm" % arm)
    try:
        pins_for(ARM_BACKBONE[arm], downloads)
    except Refused as e:
        return {"arm": arm, "status": "not run", "why": str(e)}
    return {"arm": arm, "status": "ok", "why": None}


# ------------------------------------------------------------------ the embedders
class OpenClipEmbedder(object):
    """An open_clip 'hf-hub:' backbone through semisup_labeler's loader and batch embedder, exactly as
    verify.BioclipEmbedder embeds BioCLIP-2 (its own open_clip preprocessing; fp16 autocast on CUDA)."""

    def __init__(self, name, amp=True):
        from PIL import Image
        self.device = SL._device()
        self.amp = bool(amp) and self.device == "cuda"
        self.base_name = name
        self.name = name + ("+fp16" if self.amp else "")
        log("loading %s on %s" % (name, self.device))
        self.model, self.proc = SL._load_backbone(name, self.device)
        grey = Image.new("RGB", (CROP_PX, CROP_PX), (124, 124, 124))
        self.dim = int(self([grey]).shape[1])

    def __call__(self, pils):
        import torch
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext())
        with ctx:
            return SL._embed_batch(self.model, self.proc, pils, ("clip",), self.device)["clip"]


def _pool_clip(model, x):
    return model(pixel_values=x).image_embeds


def _pool_siglip(model, x):
    return model(pixel_values=x).pooler_output


def _pool_dinov2(model, x):
    return model(pixel_values=x).last_hidden_state[:, 0]


def _pool_timm(model, x):
    return model(x)


POOLS = {"hf_clip": _pool_clip, "hf_siglip": _pool_siglip, "hf_dinov2": _pool_dinov2, "timm": _pool_timm}


def crop_tensor(pils, mean, std, device="cpu"):
    """[n, 3, 224, 224] float32: each crop as it is (224 px; no resize, no centre crop), /255, then the model's
    own (x - mean) / std."""
    import numpy as np
    import torch
    a = np.stack([np.asarray(p.convert("RGB"), dtype=np.uint8) for p in pils])
    if a.shape[1:] != (CROP_PX, CROP_PX, 3):
        raise RuntimeError("crops of %s, not %dx%d: a backbone takes E3's crop as it is" % (a.shape[1:], CROP_PX,
                                                                                            CROP_PX))
    t = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (t - m) / s


class TensorEmbedder(object):
    """A transformers or timm backbone on E3's crops as they are (crop_tensor), fp16 autocast on CUDA; the
    features POOLS[loader] reads."""

    def __init__(self, name, model, mean, std, pool, device, amp=True):
        from PIL import Image
        self.model, self.mean, self.std, self.pool, self.device = model, list(mean), list(std), pool, device
        self.amp = bool(amp) and device == "cuda"
        self.name = name + ("+fp16" if self.amp else "")
        grey = Image.new("RGB", (CROP_PX, CROP_PX), (124, 124, 124))
        self.dim = int(self([grey]).shape[1])

    def __call__(self, pils):
        import torch
        x = crop_tensor(pils, self.mean, self.std, self.device)
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext())
        with torch.no_grad(), ctx:
            out = self.pool(self.model, x)
        return out.float().cpu().numpy()


def _hf_norm(snapshot):
    """(mean, std) of a transformers snapshot's own preprocessor_config.json (rescale 1/255, normalise)."""
    d = TS._read_json(Path(snapshot) / "preprocessor_config.json")
    if not isinstance(d, dict) or len(d.get("image_mean") or []) != 3 or len(d.get("image_std") or []) != 3:
        raise Refused("%s has no preprocessor_config.json with image_mean and image_std" % snapshot)
    if d.get("do_normalize") is False or abs(float(d.get("rescale_factor", 1 / 255.0)) - 1 / 255.0) > 1e-12:
        raise Refused("%s's processor does not rescale by 1/255 and normalise" % snapshot)
    return [float(x) for x in d["image_mean"]], [float(x) for x in d["image_std"]]


def _load_hf_model(loader, snapshot, device):
    """The transformers model of a checked snapshot, loaded from its directory (never resolved on the hub)."""
    import transformers as TF
    cfg = TS._read_json(Path(snapshot) / "config.json") or {}
    if loader == "hf_clip":
        cls = TF.CLIPVisionModelWithProjection
    elif loader == "hf_siglip":
        if cfg.get("model_type") != "siglip":
            raise Refused("%s is model_type %r, not the fixed-resolution SigLIP architecture" % (snapshot,
                                                                                                cfg.get("model_type")))
        cls = TF.SiglipVisionModel
    else:
        cls = TF.AutoModel
    return cls.from_pretrained(str(snapshot), local_files_only=True).to(device).eval()


def _load_timm(weight_file, device, num_classes=0):
    """ResNet-50 a1 built by timm (pretrained_cfg: its mean and std) with the pinned file's weights loaded
    explicitly; the 1,000-class head dropped (num_classes 0: pooled features; 13: a fresh head, seeded by the
    caller). Returns (model, mean, std)."""
    import timm
    import torch
    m = timm.create_model(TIMM_NAME, pretrained=False, num_classes=num_classes)
    wf = str(weight_file)
    if wf.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(wf)
    else:
        sd = torch.load(wf, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected or set(missing) - {"fc.weight", "fc.bias"}:
        raise Refused("%s does not fit %s (missing %s, unexpected %s)" % (wf, TIMM_NAME, missing[:3], unexpected[:3]))
    pc = getattr(m, "pretrained_cfg", None) or {}
    if len(pc.get("mean") or ()) != 3 or len(pc.get("std") or ()) != 3:
        raise Refused("timm's %s has no mean and std" % TIMM_NAME)
    return m.to(device).eval(), [float(x) for x in pc["mean"]], [float(x) for x in pc["std"]]


def load_embedder(bb, amp=True, downloads=None):
    """(embedder, checkpoint record) of backbone bb, after checkpoint(bb). BC2 is verify.BioclipEmbedder itself."""
    ck = checkpoint(bb, downloads)
    e = BACKBONES[bb]
    tag = "%s@%s" % (e["repo"], ck["revision"][:12])
    if bb == REFERENCE:
        emb = V.BioclipEmbedder(amp=amp)
    elif e["loader"] == "open_clip":
        emb = OpenClipEmbedder("hf-hub:%s" % e["repo"], amp=amp)
    elif e["loader"] == "timm":
        device = SL._device()
        model, mean, std = _load_timm(Path(ck["snapshot"]) / ck["weight_file"], device, num_classes=0)
        emb = TensorEmbedder("timm:%s" % tag, model, mean, std, POOLS["timm"], device, amp=amp)
    else:
        device = SL._device()
        mean, std = _hf_norm(ck["snapshot"])
        model = _load_hf_model(e["loader"], ck["snapshot"], device)
        emb = TensorEmbedder("hf:%s" % tag, model, mean, std, POOLS[e["loader"]], device, amp=amp)
    if int(emb.dim) != int(e["dim"]):
        raise Refused("%s embeds %d-d features, the amendment's %s is %d-d" % (bb, emb.dim, e["repo"], e["dim"]))
    emb.checkpoint = ck
    ck = dict(ck, embedder=emb.name, dim=int(emb.dim), normalisation={"mean": getattr(emb, "mean", None),
                                                                      "std": getattr(emb, "std", None)}
              if hasattr(emb, "mean") else "the backbone's own open_clip preprocessing", pooling=e["pooling"])
    return emb, ck


# ------------------------------------------------------------------ FT-ResNet-50 as an embedder
def ft_encode(logits):
    """float32 [n, 14]: each row's centred logits / FT_K, then the entry that makes the row a unit vector, so that
    E3's L2-normalisation leaves it as it is and W = FT_K [I | 0], b = 0 gives back the logits (to float32
    rounding; softmax is invariant to the centring). A row whose centred logits reach FT_K refuses."""
    import numpy as np
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    r = z / FT_K
    n2 = (r * r).sum(axis=1)
    if not np.isfinite(z).all() or (n2 >= 1.0).any():
        raise RuntimeError("a crop's centred logits are not finite or reach FT_K = %g" % FT_K)
    return np.concatenate([r, np.sqrt(1.0 - n2)[:, None]], axis=1).astype(np.float32)


def ft_classifier_cfg(prior):
    """E3's classifier fields for FT's encoding: W = FT_K [I_13 | 0], b = 0, every class, the given prior."""
    import numpy as np
    W = np.concatenate([np.eye(C.NC), np.zeros((C.NC, 1))], axis=1) * FT_K
    return {"W": W, "b": np.zeros(C.NC), "classes": np.arange(C.NC, dtype=np.int64),
            "prior": np.asarray(prior, dtype=np.float64)}


class FTEmbedder(object):
    """A fine-tuned 13-class network as E3's validator's embedder (ft_encode of its logits; fp16 autocast on CUDA)."""
    dim = C.NC + 1

    def __init__(self, model, mean, std, device, name, amp=True):
        self.model, self.mean, self.std, self.device = model.eval(), list(mean), list(std), device
        self.amp = bool(amp) and device == "cuda"
        self.name = name + ("+fp16" if self.amp else "")

    def logits(self, pils):
        import torch
        x = crop_tensor(pils, self.mean, self.std, self.device)
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext())
        with torch.no_grad(), ctx:
            return self.model(x).float().cpu().numpy()

    def __call__(self, pils):
        return ft_encode(self.logits(pils))


# ------------------------------------------------------------------ the 640-px view (D2BC2640)
def view640(image):
    """(RGB uint8 array of the 640-px view, (w0, h0) after EXIF transposition, (w, h) of the view): the image as
    every E3 crop reads it (PIL, exif_transpose), resized as Ultralytics' validator loader resizes it
    (BaseDataset.load_image, rect mode: long side IMGSZ, short side min(ceil(side x r), IMGSZ),
    cv2.INTER_LINEAR; no resize when r = 1)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(image) as im0:
        a = np.asarray(ImageOps.exif_transpose(im0).convert("RGB"))
    h0, w0 = a.shape[:2]
    r = IMGSZ / max(h0, w0)
    if r != 1:
        w, h = (min(math.ceil(w0 * r), IMGSZ), min(math.ceil(h0 * r), IMGSZ))
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(a), (w0, h0), (int(a.shape[1]), int(a.shape[0]))


def cut640_task(task):
    """verify._cut_task's contract on the 640-px view: the normalised boxes (the EXIF-transposed frame's) mapped
    into the view by its per-axis gains, then E3's square crop, margin, grey padding and 224-px bicubic resize
    (semisup_labeler._cut). Worker: (image, crop ids, [uint8 crops] or None, error)."""
    import numpy as np
    from PIL import Image
    image, rows = task
    ids = [r["crop_id"] for r in rows]
    try:
        a, _wh0, (w, h) = view640(image)
        im = Image.fromarray(a)
        return image, ids, [np.asarray(SL._cut(im, dict(r, W=w, H=h))) for r in rows], None
    except Exception as e:                   # noqa: BLE001 - reported to the caller, which refuses
        return image, ids, None, "%s: %s" % (type(e).__name__, e)


def e3_cut640(task):
    """twostage._e3_cut's contract on the 640-px view (the family view640 pass's cut function)."""
    image, rows = task
    try:
        hd = TS.header(image)
    except Exception as e:                   # noqa: BLE001 - reported to the caller, which refuses
        return image, [r["crop_id"] for r in rows], None, "%s: %s" % (type(e).__name__, e), None
    if not rows:
        return image, [], [], None, hd
    img, ids, arrays, err = cut640_task((image, rows))
    return img, ids, arrays, err, hd


class _CutWorkers(object):
    """Workers whose E3 cut (twostage._e3_cut) is replaced by another (a gt pass on the 640-px view)."""

    def __init__(self, inner, cut):
        self.inner, self.cut = inner, cut

    def imap(self, fn, items, chunksize=8):
        return self.inner.imap(self.cut if fn is TS._e3_cut else fn, items, chunksize)


# ------------------------------------------------------------------ E3's training crops
def _read_crops_csv(path):
    rows = []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        if tuple(head) != TS.TRAIN_CROP_FIELDS:
            raise Refused("%s has columns %s, not E3's %s" % (path, head, list(TS.TRAIN_CROP_FIELDS)))
        for v in rd:
            r = dict(zip(head, v))
            rows.append({"crop_id": int(r["crop_id"]), "key": r["key"], "image": r["image"], "box": int(r["box"]),
                         "label": int(r["label"]), "session": r["session"], "cx": float(r["cx"]),
                         "cy": float(r["cy"]), "w": float(r["w"]), "h": float(r["h"])})
    return rows


def _e3_classifier():
    """(E3's pin, classifier record, {json_sha256, npz_sha256}, fitted classes): the files hashing as the pin."""
    try:
        pin = TS.read_pin()
        _W, _b, classes, _p, crec, shas = TS.load_classifier()
    except TS.E3Refused as e:
        raise Refused("E3's pinned classifier: %s" % e)
    return pin, crec, shas, [int(c) for c in classes]


def training_crops(testing=None, images=True):
    """E3's ground-truth training crops, exactly (V3F.5 D2): twostage/e3_v1/classifier/train_crops.csv in crop-id
    order, hashing as E3's pin and classifier.json record it; base_v2 under LOCK v2 the manifest E3 fitted on,
    every crop's key and image a base_v2 row's, no row an evaluation manifest's; and (images) base_v2 through
    inc2.train.check_manifest (every image's sha256), the v2 never-train guard and the test v1 lists, as E3's
    fit checked it. Returns (rows, record)."""
    testing = S.testing() if testing is None else bool(testing)
    pin, crec, shas, classes = _e3_classifier()
    tc = TS.classifier_dir() / "train_crops.csv"
    sha = TS._sha(tc)
    if sha is None or sha != pin.get("train_crops_sha256") or sha != (crec.get("files") or {}).get("train_crops_sha256"):
        raise Refused("%s does not hash as E3's pin and classifier.json record it" % tc)
    rows = _read_crops_csv(tc)
    if [r["crop_id"] for r in rows] != list(range(len(rows))) or len(rows) != int(crec.get("train_boxes") or -1):
        raise Refused("train_crops.csv is not E3's %s training boxes in crop-id order" % crec.get("train_boxes"))
    if any(not 0 <= r["label"] < C.NC for r in rows):
        raise Refused("train_crops.csv holds a label outside 0..%d" % (C.NC - 1))
    try:
        path = C2.v2_manifest_path("base_v2")
        msha = C2.verify_manifest_against_lock_v2("base_v2")
    except C2.Inc2Error as e:
        raise Refused("LOCK v2's base_v2: %s" % e)
    if msha != (crec.get("manifest") or {}).get("sha256"):
        raise Refused("base_v2 (%s) is not the manifest E3's classifier was fitted on" % msha[:12])
    rec = {"train_crops": {"file": "twostage/%s/classifier/train_crops.csv" % TS.E3_NAME, "sha256": sha,
                           "n": len(rows)},
           "e3_classifier": shas, "e3_pin_sha256": TS._sha(TS.pin_path()), "base_v2_sha256": msha,
           "classes": classes}
    if images:
        try:
            brows, dhashes, summ = T.check_manifest(path)
            guard = T.guard_rows(brows, dhashes, production=not testing)
        except T.RunError as e:
            raise Refused("base_v2 refused at %s: %s" % (e.stage, e))
        if summ.get("train_manifest_sha256") != msha:
            raise Refused("base_v2 changed while it was checked")
        hits, prior_rec = TS._prior_test_hits(brows)
        if hits:
            raise Refused("%d base_v2 row(s) are in a test v1 list or its companions (e.g. %s)" % (len(hits), hits[:3]))
        rec["never_train"] = {"check_manifest": {k: summ.get(k) for k in ("n_train_images", "n_train_boxes")},
                              "guard": {k: guard.get(k) for k in ("lock_sha256", "index_sha256", "checked", "refused",
                                                                  "crosscheck_hits")},
                              "test_v1": {"hits": 0, "files": prior_rec.get("files"), "rows": prior_rec.get("rows")}}
    else:
        brows = C.read_manifest(path)
    by_key = {r["key"]: r for r in brows}
    bad = [r["key"] for r in rows if r["key"] not in by_key or by_key[r["key"]]["image"] != r["image"]]
    if bad:
        raise Refused("%d training crop(s) name no base_v2 row's key and image (e.g. %s)" % (len(bad), bad[:3]))
    ev_keys, ev_shas, ev_imgs = set(), set(), set()
    for _ver, split, p in T.eval_manifest_paths():
        # the never-train exams (dev, test, imageweeds): base_v2's tsw rows are ood22's and ood23's images by design
        if split in C2.EVAL_SPLITS and Path(p).is_file():
            for r in C.read_manifest(p):
                ev_keys.add(r["key"])
                ev_shas.add(r.get("sha256"))
                ev_imgs.add(r.get("image"))
    used = {r["key"] for r in rows}
    ev = sorted(k for k in used if k in ev_keys or by_key[k].get("sha256") in ev_shas or by_key[k]["image"] in ev_imgs)
    if ev:
        raise Refused("%d training crop image(s) are evaluation images (e.g. %s): never trained" % (len(ev), ev[:3]))
    rec["evaluation_overlap"] = 0
    return rows, rec


def e3_train_features():
    """(F float32 [n, d], rows, record): E3's stored training features (train_emb.npz, hashing as E3's pin and
    classifier.json record it) with E3's training crops' labels and sessions (training_crops without images)."""
    import numpy as np
    rows, rec = training_crops(images=False)
    pin, crec, _shas, _cl = _e3_classifier()
    p = TS.classifier_dir() / "train_emb.npz"
    sha = TS._sha(p)
    if sha is None or sha != pin.get("train_emb_sha256") or sha != (crec.get("files") or {}).get("train_emb_sha256"):
        raise Refused("%s does not hash as E3's pin and classifier.json record it" % p)
    z = SC.load_npz(p)
    if not np.array_equal(z["crop_id"], np.arange(len(rows))) or not np.isfinite(z["F"]).all():
        raise Refused("train_emb.npz does not hold one finite feature per training crop in crop-id order")
    rec = dict(rec, train_emb={"file": "twostage/%s/classifier/train_emb.npz" % TS.E3_NAME, "sha256": sha,
                               "dim": int(z["F"].shape[1])})
    return np.asarray(z["F"], dtype=np.float32), rows, rec


def d9_design():
    """(X float64 [n, d], y, groups, record): E3's stored features L2-normalised in fp32 and taken to float64, the
    design E3's fit built (twostage.fit_classifier), with the crops' labels and sessions (NDSU's rows, which carry
    no session, are one group)."""
    import numpy as np
    Fm, rows, rec = e3_train_features()
    X = V._norm(Fm).astype(np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    groups = np.array([r["session"] for r in rows])
    return X, y, groups, rec


# ------------------------------------------------------------------ cutting and embedding
def embed_crops(images, table, embedder, workers, batch=TS.EMBED_BATCH, cut=None):
    """verify._embed_images with the cut function a parameter (default verify._cut_task: the same calls and
    the same rows): (crop ids, X float32, stats)."""
    import numpy as np
    from PIL import Image
    cut = cut or V._cut_task
    ids = [i for _img, cs in images for i in cs]
    pos = {c: k for k, c in enumerate(ids)}
    X = np.full((len(ids), int(embedder.dim)), np.nan, dtype=np.float32)
    stats = collections.Counter()
    buf, buf_ids = [], []

    def flush():
        Fm = V._embed_safely(embedder, buf, stats)
        for c, v in zip(buf_ids, Fm):
            X[pos[c]] = v

    tasks = [(img, [table.row(i) for i in cs]) for img, cs in images]
    for image, cids, arrays, err in workers.imap(cut, tasks, chunksize=4):
        if err is not None:
            stats["failed_images"] += 1
            stats["failed_crops"] += len(cids)
            if stats["failed_images"] <= 5:
                log("  WARNING: cannot cut %s: %s" % (image, err))
            continue
        for c, a in zip(cids, arrays):
            buf.append(Image.fromarray(a))
            buf_ids.append(c)
            if len(buf) >= batch:
                flush()
                buf, buf_ids = [], []
    if buf:
        flush()
    return np.array(ids, dtype=np.int64), X, stats


def cut_crops(images, table, workers, cut=None):
    """uint8 [n, 224, 224, 3] of every crop in crop-id order (verify._cut_task by default), held in memory and
    never written (FT); a crop that cannot be cut refuses."""
    import numpy as np
    cut = cut or V._cut_task
    n = sum(len(cs) for _img, cs in images)
    out = np.zeros((n, CROP_PX, CROP_PX, 3), dtype=np.uint8)
    seen = np.zeros(n, dtype=bool)
    tasks = [(img, [table.row(i) for i in cs]) for img, cs in images]
    for image, cids, arrays, err in workers.imap(cut, tasks, chunksize=4):
        if err is not None:
            raise Refused("cannot cut the crops of %s: %s (every image was hashed: a defect)" % (image, err))
        for c, a in zip(cids, arrays):
            out[int(c)] = a
            seen[int(c)] = True
    if not seen.all():
        raise Refused("%d crop(s) were not cut" % int((~seen).sum()))
    return out


# ------------------------------------------------------------------ E3's LR protocol on another backbone
def fit_lr_protocol(X32, y, groups):
    """E3's fit (twostage.fit_classifier's lines): X = the L2-normalised fp32 features in float64; C by
    twostage.cross_validate; the final fit by twostage._fit_lr; softmax(X W' + b) checked against scikit-learn's
    predict_proba; prior the training class frequencies. Returns (arrays {W, b, classes, prior, fold<f>_*},
    record {cv, fit})."""
    import numpy as np
    X = np.asarray(X32, dtype=np.float32).astype(np.float64)
    y = np.asarray(y, dtype=np.int64)
    try:
        chosen, cv, fold_models = TS.cross_validate(X, y, groups)
    except (TS.E3Refused, V.VerifyError) as e:
        raise Refused("E3's cross-validation: %s" % e)
    clf, conv = TS._fit_lr(X, y, chosen)
    if not conv:
        raise Refused("the final fit at C=%g did not converge in %d iterations" % (chosen, TS.MAX_ITER))
    W, b, classes = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64), clf.classes_.astype(np.int64)
    if W.shape[0] != len(classes):
        raise Refused("the fit's coefficients are not one row per class (%s for %d classes)" % (W.shape, len(classes)))
    Pn, Ps = TS.proba(X, W, b, classes), TS._clf_proba(clf, X)
    pdiff = float(np.abs(Pn - Ps).max()) if len(X) else 0.0
    if not pdiff <= TS.PROBA_TOL:
        raise Refused("softmax(X W' + b) differs from scikit-learn's predict_proba by %.3g" % pdiff)
    counts = np.bincount(y, minlength=C.NC).astype(np.float64)
    arrays = {"W": W, "b": b, "classes": classes, "prior": counts / counts.sum()}
    for f, m in sorted(fold_models.items()):
        arrays["fold%d_W" % f] = m.coef_.astype(np.float64)
        arrays["fold%d_b" % f] = m.intercept_.astype(np.float64)
        arrays["fold%d_classes" % f] = m.classes_.astype(np.int64)
    rec = {"cv": cv, "fit": {"C": chosen, "solver": "lbfgs", "max_iter": TS.MAX_ITER, "class_weight": TS.CLASS_WEIGHT,
                             "random_state": C.stable_int(TS.CLF_SEED_TEXT), "seed_text": TS.CLF_SEED_TEXT,
                             "n_iter": int(np.max(clf.n_iter_)), "converged": True,
                             "classes": [int(c) for c in classes], "proba_max_abs_diff": pdiff,
                             "train_top1": float((np.argmax(Pn, axis=1) == y).mean()),
                             "features": "L2-normalised fp32 features (verify._norm), fitted in float64"},
           "per_class": {C.CLASS_NAMES[k]: int(counts[k]) for k in range(C.NC)},
           "sessions": int(len(set(str(g) for g in groups)))}
    return arrays, rec


def _exists_after_fit(arm):
    key = ARM_KEY[arm]
    found = [p for p in (root() / "arms" / arm, d2_dir(key) / "dev_gt.json") if p.exists()]
    found += sorted(d2_dir(key).glob("dev_gt.s*.json")) if d2_dir(key).is_dir() else []
    return [_rel(p) for p in found]


def load_d2_classifier(key):
    """(classifier cfg {W, b, classes, prior}, classifier.json record, {pin_sha256, json_sha256, npz_sha256,
    train_emb_sha256}) of a D2 classifier: its files hashing as its pin records."""
    import numpy as np
    pin = TS._read_json(pin_path(key))
    if not isinstance(pin, dict) or pin.get("format") != FMT_PIN or pin.get("key") != key:
        raise Refused("%s is missing or not the %s classifier's pin" % (pin_path(key), key))
    d = clf_dir(key)
    got = {"json_sha256": TS._sha(d / "classifier.json"), "npz_sha256": TS._sha(d / "classifier.npz"),
           "train_emb_sha256": TS._sha(d / "train_emb.npz")}
    bad = [k for k, v in got.items() if v is None or v != pin.get(k)]
    if bad:
        raise Refused("the %s classifier's %s do not hash as its pin records: it is fitted once and never refitted"
                      % (key, bad))
    rec = TS._read_json(d / "classifier.json")
    with np.load(str(d / "classifier.npz"), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    cfg = {"W": a["W"], "b": a["b"], "classes": a["classes"], "prior": a["prior"]}
    return cfg, rec, dict(got, pin_sha256=TS._sha(pin_path(key)), key=key)


def _fit_d2(arm, embedder, ck, workers, testing):
    """Fit and pin arm's classifier once (V3F.5 D2): E3's crops (cut by verify._cut_task, or cut640_task for
    D2BC2640), embedded, E3's LR protocol. Refuses while anything made after the fit exists."""
    import numpy as np
    key = ARM_KEY[arm]
    d = clf_dir(key)
    after = _exists_after_fit(arm)
    if (d / "classifier.json").exists() or after:
        raise Refused("the %s classifier was fitted already or %s has read dev (%s): it is fitted once, before any "
                      "of the arm's dev passes" % (key, arm, ", ".join(after or [_rel(d / "classifier.json")])))
    t0 = time.time()
    rows, crec = training_crops(testing)
    images = TS._group(rows)
    t1 = time.time()
    cut = cut640_task if ARM_CUT[arm] == "view640" else None
    if cut is None:
        ids, Fm, stats = V._embed_images(images, TS._Table(rows), embedder, workers=workers, batch=TS.EMBED_BATCH)
    else:
        ids, Fm, stats = embed_crops(images, TS._Table(rows), embedder, workers, batch=TS.EMBED_BATCH, cut=cut)
    if list(ids) != list(range(len(rows))) or not np.isfinite(Fm).all():
        raise Refused("%d training crop(s) could not be cut or embedded (%s)" % (int((~np.isfinite(Fm).all(axis=1)).sum()),
                                                                                 dict(stats)))
    embed_s = time.time() - t1
    Fm = np.asarray(Fm, dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    groups = np.array([r["session"] for r in rows])
    arrays, frec = fit_lr_protocol(V._norm(Fm), y, groups)
    after = _exists_after_fit(arm)
    if after:
        raise Refused("%s appeared while the classifier was fitted" % after)
    sh = {"train_emb_sha256": F.write_once(d / "train_emb.npz", _npz_bytes(
        {"crop_id": np.arange(len(rows), dtype=np.int64), "F": Fm})),
          "npz_sha256": F.write_once(d / "classifier.npz", _npz_bytes(arrays))}
    rec = {"format": FMT_CLASSIFIER, "family": NAME, "arm": arm, "key": key, "decided_by": F.DECIDED_BY,
           "backbone": ck, "cut": ARM_CUT[arm], "crops": crec,
           "files": dict(sh, train_emb="train_emb.npz", npz="classifier.npz"),
           "embed": {"seconds": round(embed_s, 3), "stats": {k: int(v) for k, v in dict(stats).items()},
                     "dim": int(Fm.shape[1])},
           "protocol": "E3's: twostage.cross_validate (5-fold GroupKFold by session, pooled log-loss, ties to the smaller "
                       "C) and twostage._fit_lr (lbfgs, max_iter %d, no class weights)" % TS.MAX_ITER,
           "versions": TS._versions(), "code": code_shas(), "testing": bool(testing),
           "seconds": round(time.time() - t0, 3), "written_utc": TS._utc()}
    rec.update(frec)
    jsha = F.write_once(d / "classifier.json", rec)
    F.write_once(pin_path(key), {"format": FMT_PIN, "key": key, "arm": arm, "json_sha256": jsha,
                                 "npz_sha256": sh["npz_sha256"], "train_emb_sha256": sh["train_emb_sha256"],
                                 "train_crops_sha256": crec["train_crops"]["sha256"],
                                 "checkpoint": {k: ck.get(k) for k in ("repo", "revision", "weight_file",
                                                                       "weight_sha256")},
                                 "written_utc": rec["written_utc"],
                                 "note": "fitted once, before any of the arm's dev passes; never rewritten"})
    log("%s classifier: %d crops, C=%g by CV, train top-1 %.4f, %.0fs" % (key, len(rows), frec["fit"]["C"],
                                                                        frec["fit"]["train_top1"], time.time() - t0))
    return load_d2_classifier(key)


# ------------------------------------------------------------------ FT-ResNet-50
def ft_lr_lambda(step, steps_per_epoch, cfg=None):
    """The learning-rate factor of a step: linear warm-up over warmup_epochs, then cosine to 0."""
    cfg = cfg or FT_CFG
    total = int(cfg["epochs"]) * int(steps_per_epoch)
    warm = int(cfg["warmup_epochs"]) * int(steps_per_epoch)
    if step < warm:
        return float(step + 1) / float(warm)
    return 0.5 * (1.0 + math.cos(math.pi * float(step - warm) / float(max(1, total - warm))))


def _ft_augment(cfg):
    from torchvision.transforms import v2
    return v2.Compose([v2.RandomResizedCrop(CROP_PX, scale=tuple(cfg["rrc_scale"]), ratio=tuple(cfg["rrc_ratio"]),
                                            antialias=True),
                       v2.RandomHorizontalFlip(cfg["hflip"]), v2.RandomVerticalFlip(cfg["vflip"]),
                       v2.ColorJitter(*cfg["jitter"])])


def train_ft(crops, y, seed, init, device=None, cfg=None):
    """Fine-tune end to end (V3F.5 D2, backbone 10) on uint8 crops [n, 224, 224, 3] with labels y: torch seeded by
    stable_int(FT_SEED_TEXT % seed) before init() builds the network (its fresh 13-class head) and draws every
    augmentation; the order of each epoch drawn from a generator seeded the same way; final-epoch weights.
    init() -> (model, mean, std). Returns (model, record)."""
    import numpy as np
    import torch
    cfg = dict(FT_CFG, **(cfg or {}))
    device = device or SL._device()
    sint = C.stable_int(FT_SEED_TEXT % int(seed))
    torch.manual_seed(sint)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, mean, std = init()
    model = model.to(device).train()
    n = int(len(crops))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.int64))
    steps = int(math.ceil(n / float(cfg["batch"])))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: ft_lr_lambda(i, steps, cfg))
    amp = bool(cfg["amp"]) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if amp else None
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    aug = _ft_augment(cfg)
    m_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    gen = torch.Generator().manual_seed(sint)
    hist, t0 = [], time.time()
    for ep in range(int(cfg["epochs"])):
        te = time.time()
        perm = torch.randperm(n, generator=gen)
        tot, cnt = 0.0, 0
        for i in range(steps):
            idx = perm[i * cfg["batch"]:(i + 1) * cfg["batch"]]
            xb = torch.from_numpy(np.ascontiguousarray(crops[idx.numpy()])).to(device).permute(0, 3, 1, 2)
            xb = torch.stack([aug(x) for x in xb]).float().div_(255.0)
            xb = (xb - m_t) / s_t
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_fn(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            tot += float(loss.detach().float()) * len(idx)
            cnt += len(idx)
        hist.append({"epoch": ep + 1, "loss": tot / max(1, cnt), "lr_end": float(opt.param_groups[0]["lr"]),
                     "seconds": round(time.time() - te, 3)})
        log("FT seed %d epoch %d/%d: loss %.4f" % (int(seed), ep + 1, int(cfg["epochs"]), hist[-1]["loss"]))
    model.eval()
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=C.NC)
    rec = {"seed": int(seed), "seed_text": FT_SEED_TEXT % int(seed), "seed_int": sint, "cfg": cfg, "n_train": n,
           "steps_per_epoch": steps, "steps": steps * int(cfg["epochs"]), "epochs": hist, "device": str(device),
           "amp": amp, "mean": list(mean), "std": list(std),
           "class_counts": {C.CLASS_NAMES[k]: int(counts[k]) for k in range(C.NC)},
           "seconds": round(time.time() - t0, 3)}
    return model, rec


def _state_bytes(model):
    import torch
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getvalue()


def _ft_init(ck, device, num_classes=C.NC):
    def init():
        return _load_timm(Path(ck["snapshot"]) / ck["weight_file"], device, num_classes=num_classes)
    return init


def _ft_pin():
    p = TS._read_json(pin_path("FT"))
    if not isinstance(p, dict) or p.get("format") != FMT_FT_PIN:
        raise Refused("%s is missing or not FT's pin" % pin_path("FT"))
    for s in SEEDS:
        r = (p.get("seeds") or {}).get(str(s)) or {}
        d = d2_dir("FT") / ("s%d" % s)
        if TS._sha(d / FT_WEIGHTS) != r.get("weights_sha256") or TS._sha(d / "train.json") != r.get("train_sha256"):
            raise Refused("FT seed %d's files do not hash as its pin records: FT is trained once" % s)
    return p


def _ft_classifiers(ck, workers, testing, cfg=None):
    """FT's three networks (V3F.5 D2, backbone 10): each seed trained once (its weights and record written
    once), the crops cut once per job into memory; the pin written once after all three. Returns the pin."""
    import numpy as np
    if pin_path("FT").exists():
        return _ft_pin()
    after = _exists_after_fit(FT_ARM)
    if after:
        raise Refused("%s has read dev (%s) but FT's pin is missing: FT is trained once, before any of its dev passes"
                      % (FT_ARM, ", ".join(after)))
    device = SL._device()
    need = [s for s in SEEDS if not (d2_dir("FT") / ("s%d" % s) / "train.json").exists()]
    rows, crec = training_crops(testing)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    crops = cut_crops(TS._group(rows), TS._Table(rows), workers) if need else None
    for s in SEEDS:
        d = d2_dir("FT") / ("s%d" % s)
        if s not in need:
            continue
        if (d / FT_WEIGHTS).exists():        # a crashed attempt's weights without their record: trained again
            (d / FT_WEIGHTS).unlink()
        model, rec = train_ft(crops, y, s, _ft_init(ck, device), device=device, cfg=cfg)
        wsha = F.write_once(d / FT_WEIGHTS, _state_bytes(model))
        rec.update(format=FMT_FT_TRAIN, family=NAME, decided_by=F.DECIDED_BY, init=ck, crops=crec,
                   weights={"file": FT_WEIGHTS, "sha256": wsha}, versions=TS._versions(), code=code_shas(),
                   testing=bool(testing), written_utc=TS._utc())
        F.write_once(d / "train.json", rec)
        del model
    seeds = {}
    for s in SEEDS:
        d = d2_dir("FT") / ("s%d" % s)
        tr = TS._read_json(d / "train.json") or {}
        if (tr.get("weights") or {}).get("sha256") != TS._sha(d / FT_WEIGHTS):
            raise Refused("FT seed %d's weights do not hash as its training record says" % s)
        seeds[str(s)] = {"weights_sha256": TS._sha(d / FT_WEIGHTS), "train_sha256": TS._sha(d / "train.json")}
    F.write_once(pin_path("FT"), {"format": FMT_FT_PIN, "key": "FT", "arm": FT_ARM, "seeds": seeds,
                                  "init": {k: ck.get(k) for k in ("repo", "revision", "weight_file", "weight_sha256")},
                                  "train_crops_sha256": crec["train_crops"]["sha256"], "written_utc": TS._utc(),
                                  "note": "trained once, before any of the arm's dev passes; never rewritten"})
    return _ft_pin()


def _ft_embedder(s, ck, amp=True):
    """FTEmbedder of FT seed s: the network rebuilt and its pinned final weights loaded."""
    import torch
    pin = _ft_pin()
    d = d2_dir("FT") / ("s%d" % s)
    device = SL._device()
    model, mean, std = _load_timm(Path(ck["snapshot"]) / ck["weight_file"], device, num_classes=C.NC)
    sd = torch.load(str(d / FT_WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    emb = FTEmbedder(model.to(device), mean, std, device, "FT-%s s%d %s" % (TIMM_NAME, s,
                                                                           pin["seeds"][str(s)]["weights_sha256"][:12]),
                     amp=amp)
    return emb, {"seed": s, "file": _rel(d / FT_WEIGHTS), "sha256": pin["seeds"][str(s)]["weights_sha256"],
                 "pin_sha256": TS._sha(pin_path("FT"))}


# ------------------------------------------------------------------ the passes
def _e3b_stage1(s):
    """E3-B seed s's stored dev.stage1.npz (hashing as its dev file records)."""
    d = TS.arm_dir(E3_ARM, s)
    jn, _nn, sn = TS.file_names("dev")
    rec = TS._read_json(d / jn)
    if not isinstance(rec, dict) or rec.get("format") != TS.FMT_SCORE:
        raise Refused("E3-%s seed %d has no recorded dev file" % (E3_ARM, s))
    if TS._sha(d / sn) != (rec.get("stage1_images") or {}).get("sha256"):
        raise Refused("E3-%s seed %d's dev.stage1.npz does not hash as its file records" % (E3_ARM, s))
    return SC.load_npz(d / sn), {"file": "twostage/%s/arms/%s/s%d/%s" % (TS.E3_NAME, E3_ARM, s, sn),
                                 "sha256": TS._sha(d / sn)}


def _file_entry(arm, s, js, status):
    """A dev score's entry (the fields inc2.v3fam_systems' records carry): its name under v3fam/v1, sha256s and
    dev numbers."""
    dd = TS._read_json(js) or {}
    if dd.get("format") != F.FMT_SCORE:
        raise Refused("%s is not a family score" % js)
    return {"file": _rel(js), "sha256": TS._sha(js), "images_sha256": (dd.get("images") or {}).get("sha256"),
            "status": status, "dir": dd.get("dir"), "seed": int(s), "exam": dd.get("exam"),
            "variant": dd.get("variant"), "system": dd.get("system"), "species_map50_95": dd.get("species_map50_95"),
            "agnostic_map50_95": dd.get("agnostic_map50_95"), "map50_95": dd.get("map50_95"),
            "stage1_agnostic_map50_95": (dd.get("stage1") or {}).get("agnostic_map50_95")}


def _two_stage_pass(arm, s, cfg, src, stored, eqs, clf, extra, kw, variant="", keep_features=True):
    """One family pass of E3-B seed s's boxes named by cfg's classifier, its stage-1 box set checked against E3's
    stored one, written once (kept when it exists). Returns the score record's entry."""
    d = F.arm_dir(arm, s)
    js = d / F.file_names("dev", variant)[0]
    if js.exists():
        return _file_entry(arm, s, js, "kept")
    ps = F._family_pass(src["weights"], "dev", cfg, arm, **kw)
    s1 = F.stage1_compare(ps["stage1"], stored[0])
    if not keep_features and ps.get("stage1") is not None:
        ps = dict(ps, stage1={k: v for k, v in ps["stage1"].items() if k != "feat"})
    ex = {"system": SYSTEM_NAMES[arm], "dir": arm,
          "stage1_source": {"e3_arm": E3_ARM, "exp": src["exp"], "run_id": src["run_id"],
                            "weights_sha256": src["weights_sha256"], "nms": cfg.get("nms"),
                            "vs_e3_stage1": dict(s1, file=stored[1]["file"], sha256=stored[1]["sha256"])},
          "code_arm": code_shas()}
    ex.update(extra or {})
    doc = F.family_doc(arm, s, "dev", ps, cfg, variant=variant, extra=ex, classifier=clf, equivalence=eqs)
    F.commit_pass(d, "dev", doc, ps, variant=variant)
    log("%s seed %d%s on dev: species %.4f, %d crops, %.0fs" % (arm, s, " %s" % variant if variant else "",
                                                                 doc.get("species_map50_95") or 0.0,
                                                                 doc["stage2"]["n_crops"], ps["seconds"]))
    return _file_entry(arm, s, js, "written")


def gt_record(gt, gt_small):
    """top-1 / top-3, per class and the confusion of a gt pass's [(true, top1, top3)] (twostage.dev_gt_accuracy's)."""
    import numpy as np
    conf = np.zeros((C.NC, C.NC), dtype=np.int64)
    per = {}
    for t, p1, p3 in gt:
        conf[t, p1] += 1
        e = per.setdefault(C.CLASS_NAMES[t], {"n": 0, "top1": 0, "top3": 0})
        e["n"] += 1
        e["top1"] += int(p1 == t)
        e["top3"] += int(t in p3)
    n = len(gt)
    return {"n": n, "gt_small_left_out": int(gt_small),
            "top1": (sum(1 for t, p1, _ in gt if p1 == t) / n) if n else None,
            "top3": (sum(1 for t, _p, p3 in gt if t in p3) / n) if n else None,
            "per_class": {k: dict(v, top1=v["top1"] / v["n"], top3=v["top3"] / v["n"]) for k, v in sorted(per.items())},
            "confusion": conf.tolist()}


def _gt_pass(arm, out, cfg, clf, kw):
    """The classifier on dev's ground-truth crops (family gt mode; for D2BC2640 the crops cut from the 640-px
    view), written once to out (kept when it exists)."""
    if out.exists():
        return {"file": _rel(out), "sha256": TS._sha(out), "status": "kept"}
    src = TS._source(GT_DETECTOR[0], GT_DETECTOR[1], testing_ok=S.testing())
    gcfg = dict(cfg, mode="gt")
    gcfg.pop("cut", None)
    if ARM_CUT.get(arm) == "view640":
        gcfg["workers"] = _CutWorkers(cfg["workers"], e3_cut640)
    ps = F._family_pass(src["weights"], "dev", gcfg, arm, **kw)
    rec = dict(gt_record(ps["gt"], ps["stats"].get("gt_small_left_out", 0)), format=FMT_DEV_GT, arm=arm, exam="dev",
               classifier=clf, cut=ARM_CUT.get(arm, "original"),
               embedder=getattr(cfg.get("embedder"), "name", None),
               detector={k: src[k] for k in ("exp", "run_id", "weights_sha256")},
               checks={k: ps["stats"].get(k) for k in ("orientations", "gt_roundtrip_max_diff")},
               code=F.code_shas(), code_arm=code_shas(), decided_by=F.DECIDED_BY, created_utc=TS._utc(),
               note="reported only: the classifier on dev's ground-truth crops, never deciding")
    F.write_once(out, rec)
    log("%s dev ground truth: top-1 %s, top-3 %s on %d crops" % (arm, rec["top1"], rec["top3"], rec["n"]))
    return {"file": _rel(out), "sha256": TS._sha(out), "status": "written", "top1": rec["top1"], "top3": rec["top3"]}


def _write_score_record(arm, rec, before=None):
    """capacity/v3fam_score_<arm>.json (overwritten), in inc2.v3fam_systems' record format: dev file names
    (relative to v3fam/v1), sha256s, status and counts; the equivalence records it rests on; the E-series trees
    it ran over."""
    eq = rec.get("equivalence")
    rec = dict(rec, format=FMT_SCORE_RECORD, family=NAME, arm=arm, exam="dev", decided_by=F.DECIDED_BY,
               equivalence={k: {"sha256": v.get("sha256"), "status": v.get("status")} for k, v in sorted(eq.items())}
               if isinstance(eq, dict) else None,
               e_trees=None if before is None else {"files": len(before), "sha256": C.sha256_text(
                   json.dumps(before, sort_keys=True))},
               code=F.code_shas(), code_arm=code_shas(), written_utc=TS._utc())
    TS._write_json(score_record_path(arm), rec)
    return rec


def _not_run(arm, why):
    _write_score_record(arm, {"status": "not run", "reason": NOT_RUN_DOWNLOAD, "why": str(why)[:600], "dev": [],
                              "note": "a download that failed or hashes otherwise leaves the backbone not run, with "
                                      "no substitute (V3F.5 D2); the arm keeps its place in C"})


# ------------------------------------------------------------------ D2 (L23J)
def score_d2(arm, embedder=None, workers=None, lock_check=True, batch=S.BATCH, device=None, ft_cfg=None):
    """D2's job for one arm (module docstring). embedder: an already-loaded embedder (tests); otherwise the
    registry's, after its pins (a backbone that is not run is recorded so and refused)."""
    import numpy as np
    if arm not in D2_ARMS:
        raise Refused("--arm %r is not one of D2's %s" % (arm, list(D2_ARMS)))
    F.check_frozen()
    testing = S.testing()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    t0 = time.time()
    with F.e_trees_unchanged() as before:
        eqs = F.require_equivalence(F.equivalence_names("dev", e3_arms=(E3_ARM,)))
        bb = ARM_BACKBONE[arm]
        try:
            if arm == FT_ARM:
                ck = checkpoint(bb)
            elif embedder is None:
                embedder, ck = load_embedder(bb)
            else:
                ck = dict(getattr(embedder, "checkpoint", None) or {"backbone": bb, "given": True},
                          embedder=embedder.name, dim=int(embedder.dim))
        except Refused as e:
            if NOT_RUN_DOWNLOAD in str(e):
                _not_run(arm, e)
            raise
        srcs = {s: TS._source(E3_ARM, s, testing_ok=testing) for s in SEEDS}       # every source before any pass
        stored = {s: _e3b_stage1(s) for s in SEEDS}
        files, gts = [], []
        with TS._workers(workers) as wk:
            if arm == FT_ARM:
                pin = _ft_classifiers(ck, wk, testing, cfg=ft_cfg)
                rows, _r = training_crops(testing, images=False)
                counts = np.bincount([r["label"] for r in rows], minlength=C.NC).astype(np.float64)
                for s in SEEDS:
                    emb, finfo = _ft_embedder(s, ck)
                    cfg = dict(ft_classifier_cfg(counts / counts.sum()), mode="two_stage", nms=TS.NMS[E3_ARM],
                               embedder=emb, workers=wk, top_k=TS.TOP_K)
                    clf = {"key": "FT", "pin_sha256": finfo["pin_sha256"], "weights_sha256": finfo["sha256"],
                           "classifier_seed": s, "encoding": "ft_encode (FT_K %g)" % FT_K}
                    extra = {"d2": {"backbone": "FT", "checkpoint": ck, "cut": "original", "classifier_seed": s,
                                    "ft": finfo}}
                    files.append(_two_stage_pass(arm, s, cfg, srcs[s], stored[s], eqs, clf, extra, kw))
                    gts.append(_gt_pass(arm, d2_dir("FT") / ("dev_gt.s%d.json" % s), cfg, clf, kw))
                clf_rec = {"key": "FT", "pin_sha256": TS._sha(pin_path("FT")),
                           "seeds": {k: v["weights_sha256"] for k, v in pin["seeds"].items()}}
            else:
                key = ARM_KEY[arm]
                if pin_path(key).exists():
                    ccfg, _crec, cshas = load_d2_classifier(key)
                else:
                    ccfg, _crec, cshas = _fit_d2(arm, embedder, ck, wk, testing)
                mode = "view640" if ARM_CUT[arm] == "view640" else "two_stage"
                cfg = dict(ccfg, mode=mode, nms=TS.NMS[E3_ARM], embedder=embedder, workers=wk, top_k=TS.TOP_K)
                if mode == "view640":
                    cfg["cut"] = e3_cut640
                extra = {"d2": {"backbone": bb, "checkpoint": ck, "cut": ARM_CUT[arm]}}
                for s in SEEDS:
                    files.append(_two_stage_pass(arm, s, cfg, srcs[s], stored[s], eqs, cshas, extra, kw))
                gts.append(_gt_pass(arm, d2_dir(key) / "dev_gt.json", cfg, cshas, kw))
                clf_rec = cshas
    rec = _write_score_record(arm, {
        "status": "complete", "seeds": list(SEEDS), "dev": files, "dev_gt": gts, "classifier": clf_rec,
        "backbone": {k: ck.get(k) for k in ("backbone", "repo", "revision", "weight_file", "weight_sha256",
                                             "embedder", "dim")},
        "cut": ARM_CUT[arm], "equivalence": eqs, "stage1": {"e3_arm": E3_ARM, "reproduced": True},
        "counts": {"dev_passes": len(files) + len(gts), "written": sum(1 for f in files + gts
                                                                       if f.get("status") == "written")},
        "note": "dev only; names relative to v3fam/%s, sha256s, no path" % NAME}, before=before)
    log("%s scored on dev (%s), %.0fs" % (arm, ", ".join("s%d %s" % (f["seed"], f["status"]) for f in files),
                                          time.time() - t0))
    return rec


# ------------------------------------------------------------------ D9 (L23J)
def d9_required(classes):
    """The classes every D9b refit must hold: E3's fitted classes (all 13 in production)."""
    cl = sorted(int(c) for c in classes)
    if not S.testing() and cl != list(range(C.NC)):
        raise Refused("E3's classifier was fitted on %d classes, not %d: D9b keeps every class" % (len(cl), C.NC))
    return tuple(cl)


def half_sample(groups, y, b, required, max_redraws=D9_MAX_REDRAWS):
    """D9b's draw b (V3F.5 D9): every session (group) placed in the stratum of the rarest class it holds (the
    fewest training boxes of y among the classes present; ties to the lower class id); within each stratum, in
    class order, half of its sessions drawn without replacement by numpy.random.default_rng(stable_int(
    D9_SEED_TEXT % b)), an odd count rounding up and down alternately from a seeded start; a draw in which a
    required class has no box is redrawn (the same generator, counted). Returns (bool mask, record)."""
    import numpy as np
    g = np.asarray([str(x) for x in groups])
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=C.NC)
    strata = collections.defaultdict(list)
    for s in sorted(set(g.tolist())):
        present = sorted(set(y[g == s].tolist()))
        strata[min(present, key=lambda c: (int(counts[c]), c))].append(s)
    text = D9_SEED_TEXT % int(b)
    rng = np.random.default_rng(C.stable_int(text))
    redraws = 0
    while True:
        start = int(rng.integers(0, 2))
        drawn, odd, sizes = [], 0, {}
        for c in sorted(strata):
            ss = strata[c]
            if len(ss) % 2:
                k = (len(ss) + 1) // 2 if (odd + start) % 2 == 0 else len(ss) // 2
                odd += 1
            else:
                k = len(ss) // 2
            sizes[C.CLASS_NAMES[c]] = [len(ss), k]
            if k:
                drawn += [str(x) for x in rng.choice(np.asarray(ss), size=k, replace=False)]
        sel = np.isin(g, drawn)
        missing = sorted(set(int(c) for c in required) - set(y[sel].tolist()))
        if not missing:
            break
        redraws += 1
        if redraws > max_redraws:
            raise Refused("half-sample %d: no draw in %d held every class (missing %s)"
                          % (b, max_redraws, [C.CLASS_NAMES[c] if 0 <= c < C.NC else c for c in missing]))
    got = np.bincount(y[sel], minlength=C.NC)
    return sel, {"b": int(b), "seed_text": text, "seed": C.stable_int(text), "redraws": redraws, "start": start,
                 "strata": sizes, "sessions": sorted(drawn), "n_sessions": len(drawn),
                 "sessions_sha256": C.sha256_text("\n".join(sorted(drawn))), "n_crops": int(sel.sum()),
                 "class_counts": {C.CLASS_NAMES[k]: int(got[k]) for k in range(C.NC)}}


def _d9a(embedder, wk, kw, e3_shas):
    """D9a: E3's own sensitivity (twostage.sensitivity), written once under twostage/e3_v1; kept when it exists."""
    p = TS.root() / "sensitivity.json"
    if p.exists():
        d = TS._read_json(p)
        if not isinstance(d, dict) or d.get("format") != TS.FMT_SENSITIVITY or d.get("classifier") != e3_shas:
            raise Refused("%s exists but is not E3's sensitivity of the pinned classifier" % p)
        status = "kept"
    else:
        try:
            d = TS.sensitivity(embedder=embedder, workers=wk, **kw)
        except TS.E3Refused as e:
            raise Refused("D9a (E3's sensitivity): %s" % e)
        status = "written"
    return {"file": "twostage/%s/sensitivity.json" % TS.E3_NAME, "sha256": TS._sha(p), "status": status,
            "arms": {a: {k: (v or {}).get(k) for k in ("values", "mean", "sd", "full_classifier")}
                     for a, v in (d.get("arms") or {}).items()}}


def _d9_refit(b, X, y, groups, required):
    """D9b refit b: its half sample, the 13-class LR at D9_C on it (twostage._fit_lr), written once
    (d9/refits/half<b>.npz and .json); a rerun reads the stored refit back."""
    import numpy as np
    d = d9_dir() / "refits"
    npz, js = d / ("half%02d.npz" % b), d / ("half%02d.json" % b)
    if js.exists():
        rec = TS._read_json(js) or {}
        if TS._sha(npz) != rec.get("npz_sha256"):
            raise Refused("%s does not hash as its record says" % npz)
        a = SC.load_npz(npz)
        return {"W": a["W"], "b": a["b"], "classes": a["classes"], "prior": a["prior"]}, rec, "kept"
    sel, hrec = half_sample(groups, y, b, required)
    clf, conv = TS._fit_lr(X[sel], y[sel], D9_C)
    W, bb, classes = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64), clf.classes_.astype(np.int64)
    counts = np.bincount(y[sel], minlength=C.NC).astype(np.float64)
    arrays = {"W": W, "b": bb, "classes": classes, "prior": counts / counts.sum(), "selected": sel}
    nsha = F.write_once(npz, _npz_bytes(arrays))
    rec = {"format": FMT_D9_REFIT, "b": int(b), "C": D9_C, "half_sample": hrec, "converged": bool(conv),
           "n_iter": int(np.max(clf.n_iter_)), "classes": [int(c) for c in classes], "npz_sha256": nsha,
           "solver": "lbfgs", "max_iter": TS.MAX_ITER, "class_weight": TS.CLASS_WEIGHT,
           "random_state": C.stable_int(TS.CLF_SEED_TEXT), "written_utc": TS._utc()}
    F.write_once(js, rec)
    return {"W": W, "b": bb, "classes": classes, "prior": arrays["prior"]}, rec, "written"


def _sd(xs):
    xs = [float(x) for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def score_d9(embedder=None, workers=None, refits=D9_REFITS, lock_check=True, batch=S.BATCH, device=None):
    """D9's job (module docstring): D9a, then D9b's refits each through E3-B's seeds 0-2."""
    F.check_frozen()
    testing = S.testing()
    kw = {"lock_check": lock_check, "batch": batch, "device": device}
    t0 = time.time()
    out = d9_dir() / "record.json"
    with F.e_trees_unchanged() as before:
        eqs = F.require_equivalence(F.equivalence_names("dev", e3_arms=(E3_ARM,)))
        _pin, crec, e3_shas, classes = _e3_classifier()
        if embedder is None:
            embedder, ck = load_embedder(REFERENCE)
        else:
            ck = dict(getattr(embedder, "checkpoint", None) or {"backbone": REFERENCE, "given": True},
                      embedder=embedder.name)
        try:
            _vp, verdict = TS._verdict_doc(testing_ok=testing)
        except TS.E3Refused as e:
            raise Refused("D9 needs E3's decided verdict: %s" % e)
        srcs = {s: TS._source(E3_ARM, s, testing_ok=testing) for s in SEEDS}
        stored = {s: _e3b_stage1(s) for s in SEEDS}
        X, y, groups, frec = d9_design()
        required = d9_required(classes)
        files, refit_recs = [], []
        with TS._workers(workers) as wk:
            d9a = _d9a(embedder, wk, kw, e3_shas)
            for b in range(int(refits)):
                rcfg, rrec, rstatus = _d9_refit(b, X, y, groups, required)
                refit_recs.append(dict(rrec, status=rstatus))
                cfg = dict(rcfg, mode="two_stage", nms=TS.NMS[E3_ARM], embedder=embedder, workers=wk, top_k=TS.TOP_K)
                clf = {"refit": int(b), "npz_sha256": rrec["npz_sha256"],
                       "record_sha256": TS._sha(d9_dir() / "refits" / ("half%02d.json" % b))}
                for s in SEEDS:
                    e = _two_stage_pass(D9_ARM, s, cfg, srcs[s], stored[s], eqs, clf,
                                        {"d9": {"refit": int(b), "C": D9_C, "checkpoint": ck}}, kw,
                                        variant="half%02d" % b, keep_features=False)
                    files.append(dict(e, refit=int(b)))
        per_seed = {}
        for s in SEEDS:
            vals = [f["species_map50_95"] for f in files if f["seed"] == s]
            per_seed[str(s)] = {"values": vals, "mean": statistics.fmean(vals) if vals else None, "sd": _sd(vals)}
        var = [v["sd"] ** 2 for v in per_seed.values() if v["sd"] is not None]
        pooled = math.sqrt(statistics.fmean(var)) if var else None
        e3b = [(TS._read_json(TS.arm_dir(E3_ARM, s) / "dev.json") or {}).get("species_map50_95") for s in SEEDS]
        e3b_sd = _sd(e3b)
        diffs = {a: ((verdict.get("arms") or {}).get(a) or {}).get("diff") for a in TS.ARM_ORDER}
        diffs["attribution"] = (verdict.get("attribution") or {}).get("diff")
        sentence = None
        if pooled is not None and e3b_sd is not None:
            sentence = D9_SENTENCE["exceeds"] if pooled > e3b_sd else D9_SENTENCE["within"]
        summary = {"per_seed": per_seed, "pooled_sd": pooled,
                   "all_sd": _sd([f["species_map50_95"] for f in files]),
                   "e3b_dev_seed_values": e3b, "e3b_dev_seed_sd": e3b_sd, "e3_diffs": diffs, "sesoi": F.SESOI,
                   "sentence": sentence, "rule": "pooled refit sd > E3-B's dev seed sd (unrounded, from E3's dev "
                                                 "files): '%s'; otherwise '%s'" % (D9_SENTENCE["exceeds"],
                                                                                 D9_SENTENCE["within"]),
                   "redraws": sum(r["half_sample"]["redraws"] for r in refit_recs),
                   "not_converged": sum(1 for r in refit_recs if not r["converged"])}
        doc = {"format": FMT_D9, "family": NAME, "decided_by": F.DECIDED_BY, "d9a": d9a,
               "d9b": {"refits": int(refits), "C": D9_C, "seed_text": D9_SEED_TEXT, "required_classes": list(required),
                       "features": frec, "refit_records": [{k: r.get(k) for k in ("b", "converged", "n_iter", "npz_sha256",
                                                                                 "status")}
                                                           for r in refit_recs],
                       "passes": files, "summary": summary},
               "checkpoint": ck, "equivalence": eqs, "code": F.code_shas(), "code_arm": code_shas(),
               "testing": bool(testing), "note": "descriptive: no decision; dev only"}
        if out.exists():
            old = TS._read_json(out) or {}
            if [p.get("sha256") for p in ((old.get("d9b") or {}).get("passes") or [])] != [p["sha256"] for p in files]:
                raise Refused("%s exists and names other passes: written once" % out)
            status = "kept"
        else:
            doc["written_utc"] = TS._utc()
            F.write_once(out, doc)
            status = "written"
    rec = _write_score_record(D9_ARM, {
        "status": "complete", "seeds": list(SEEDS), "dev": files, "d9a": d9a,
        "record": {"file": _rel(out), "sha256": TS._sha(out), "status": status},
        "counts": {"refits": int(refits), "dev_passes": len(files), "d9a_passes": sum(
            len(v.get("values") or []) for v in d9a["arms"].values())},
        "summary": {k: summary[k] for k in ("pooled_sd", "e3b_dev_seed_sd", "sentence", "redraws")},
        "equivalence": eqs,
        "note": "dev only; names relative to v3fam/%s (D9a's file to INC_DIR), sha256s, no path" % NAME},
        before=before)
    log("D9: pooled refit sd %s vs E3-B's dev seed sd %s (SESOI %g; E3's D's %s): %s; %.0fs"
        % (pooled, e3b_sd, F.SESOI, diffs, sentence, time.time() - t0))
    return rec


def score(arm, **kw):
    """inc2.v3fam score --arm X for D2's nine arms and D9."""
    if arm == D9_ARM:
        return score_d9(**kw)
    return score_d2(arm, **kw)


# ------------------------------------------------------------------ the downloads (V3F.5 D2, step 0c)
DATE_CMD = "date -u +%Y-%m-%dT%H:%M:%SZ"
UTC_LINE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s*$", re.M)


def download_plan():
    """The delegate's download commands (login node, into the default cache), run with their output logged (V3F.5
    D2: the commands and times go into downloads.json): the UTC time before and after each download, the files of
    DOWNLOAD_FILES at the amendment's revisions (ResNet-50 a1: the revision the other cache holds), then
    `downloads --commands` with that log."""
    lines = ["# V3F step 0c (login node): run these with their output logged, e.g. bash -x plan.sh 2>&1 | tee "
             "downloads.log; fetch at the pinned revisions into %s, then record them:" % hub_cache()]
    for bb, files in DOWNLOAD_FILES.items():
        e = BACKBONES[bb]
        if e["loader"] == "open_clip":
            # open_clip resolves 'hf-hub:' names by refs/main, which only a download by branch writes
            rev = "main   # refs/main must then name %s; if main has moved, download with --revision %s and " \
                  "write that sha into refs/main" % (e["revision"], e["revision"])
        else:
            rev = e["revision"] or "<the revision /ocean/projects/cis240145p/byler/hf_cache/hub holds>"
        lines.append("echo '%s start'; %s" % (bb, DATE_CMD))
        lines.append("HF_HUB_OFFLINE=0 huggingface-cli download %s %s --revision %s" % (e["repo"], " ".join(files), rev))
        lines.append("echo '%s end'; %s" % (bb, DATE_CMD))
    lines.append("python -m weed_optimizer_framework.tools.inc2.v3fam_embed downloads --commands downloads.log")
    return "\n".join(lines)


def command_log(commands):
    """{file, sha256, text, times: [first, last], n_times} of the delegate's download log (V3F.5 D2: the commands
    and times), refused when it is missing, empty or records no UTC time line (DATE_CMD's output)."""
    if not commands:
        raise Refused("downloads needs --commands <the download log> (V3F.5 D2: the commands and times are logged in "
                      "downloads.json, which is written once)")
    p = Path(commands)
    if not p.is_file() or not p.read_text().strip():
        raise Refused("the download log %s is missing or empty" % p)
    text = p.read_text()
    times = UTC_LINE_RE.findall(text)
    if not times:
        raise Refused("the download log %s records no UTC time (download_plan's '%s' lines): the times are part of "
                      "the record" % (p, DATE_CMD))
    return {"file": p.name, "sha256": C.sha256_file(p), "text": text, "times": [times[0], times[-1]],
            "n_times": len(times)}


def record_downloads(commands=None):
    """v3fam/v1/downloads.json, written once (the delegate, step 0c): for every backbone of the registry, the
    default cache's revision (the amendment's where it pins one; else refs/main, or the one snapshot), every
    file of that snapshot with its sha256 and size, the weight file its loader opens, and its status: ok, or
    missing / mismatch with why (that backbone is then not run, with no substitute); and the delegate's command
    log with its first and last UTC time (command_log: required, so the record is never written without it)."""
    out = downloads_path()
    if out.exists():
        raise Refused("%s exists: written once" % out)
    cmd_log = command_log(commands)
    bbs = {}
    for bb, e in sorted(BACKBONES.items()):
        rd = repo_dir(e["repo"])
        refs = _refs_main(e["repo"])
        snaps = sorted(p.name for p in (rd / "snapshots").iterdir() if p.is_dir()) if (rd / "snapshots").is_dir() \
            else []
        rev = e["revision"] or refs or (snaps[0] if len(snaps) == 1 else None)
        r = {"repo": e["repo"], "loader": e["loader"], "revision": rev, "refs_main": refs, "snapshots": snaps,
             "files": {}, "weight_file": None, "status": "ok", "why": None}
        if rev is None or rev not in snaps:
            r.update(status="missing", why="no snapshot %s in the cache (%s)" % (rev, snaps))
            bbs[bb] = r
            continue
        present = _snapshot_files(rd / "snapshots" / rev)
        r["files"] = {n: {"sha256": C.sha256_file(p), "bytes": int(p.stat().st_size)} for n, p in present.items()}
        wf = loader_weight_file(e["loader"], present)
        r["weight_file"] = wf
        why = []
        if wf is None:
            why.append("no weight file the %s loader opens" % e["loader"])
        elif e["weights"] and wf not in e["weights"]:
            why.append("the loader would open %s, not a pinned weight file %s" % (wf, sorted(e["weights"])))
        for n, sha in list((e["weights"] or {}).items()) + list((e.get("configs") or {}).items()):
            if n in r["files"] and r["files"][n]["sha256"] != sha:
                why.append("%s hashes to %s, pinned %s" % (n, r["files"][n]["sha256"][:12], sha[:12]))
        for n in DOWNLOAD_FILES.get(bb, ()):
            if n not in r["files"]:
                why.append("%s was not fetched" % n)
        if e["loader"] == "open_clip" and refs != rev:
            why.append("refs/main is %s, not %s (open_clip resolves refs/main)" % (refs, rev))
        if why:
            r.update(status="mismatch" if r["files"] else "missing", why="; ".join(why))
        bbs[bb] = r
    doc = {"format": FMT_DOWNLOADS, "cache": str(hub_cache()), "backbones": bbs, "decided_by": F.DECIDED_BY,
           "code": code_shas(), "created_utc": TS._utc()}
    doc["commands"] = cmd_log
    F.write_once(out, doc)
    log("downloads: %s" % {k: v["status"] for k, v in bbs.items()})
    return doc


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F stage 2: D2 (stage-2 backbones on E3-B's boxes) and D9 (the "
                                             "classifier's fit-sample sensitivity); dev only.")
    ap.add_argument("command", choices=("score", "plan", "downloads", "status"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--commands", default=None, help="downloads (required): the delegate's command log with its "
                                                     "UTC times, kept in the record")
    ap.add_argument("--batch", type=int, default=None, help="protocol %d; another only with %s=1" % (S.BATCH,
                                                                                                    S.TEST_ENV))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lock-check-for-tests", action="store_true")
    a = ap.parse_args(argv)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if (a.batch is not None or a.device is not None or a.no_lock_check_for_tests) and not S.testing():
        print("[inc2.v3fam_embed] ERROR: --batch, --device and --no-lock-check-for-tests need %s=1" % S.TEST_ENV,
              file=sys.stderr)
        return 2
    kw = {"lock_check": not a.no_lock_check_for_tests, "batch": a.batch or S.BATCH, "device": a.device}
    try:
        if a.command == "score":
            if a.arm not in D2_ARMS + (D9_ARM,):
                raise Refused("score needs --arm %s" % "|".join(D2_ARMS + (D9_ARM,)))
            score(a.arm, **kw)
        elif a.arm is not None:
            raise Refused("%s takes no --arm" % a.command)
        elif a.command == "plan":
            print(download_plan())
        elif a.command == "downloads":
            record_downloads(commands=a.commands)
        else:
            print(json.dumps({arm: backbone_status(arm) for arm in D2_ARMS}, indent=1, sort_keys=True))
    except (Refused, TS.E3Refused, S.ScorerRefused, SC.SidecarError, SN.NativeRefused, V.VerifyError, C2.Inc2Error,
            T.RunError, ValueError) as e:
        print("[inc2.v3fam_embed] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
