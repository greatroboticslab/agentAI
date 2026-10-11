#!/usr/bin/env python3
"""V3F stage 2: D2, the stage-2 backbone ablation on E3-B's boxes, and D9, the
classifier's fit-sample sensitivity (inc2/v3fam_embed.py; docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered as
one family)", V3F.5 D2 and D9).

No model is downloaded and none of open_clip, transformers or timm is imported:
tiny deterministic fakes stand in for every backbone (semisup_labeler's
open_clip loader, the transformers and timm loaders of v3fam_embed), and a fake
Hugging Face cache in the layout huggingface_hub writes (refs, snapshots,
blobs) stands in for the cluster's, with the registry's pins pointed at its
files (the real pins are asserted literally first).

Two groups. UNITS need no scoring pass (V3FAM_EMBED_ONLY=units runs only them):
- the constants: the nine D2 arms (v3fam's first nine challengers) and their
  checkpoints (D2LAION the LAION-2B CLIP ViT-L/14, D2CLIP OpenAI's, ...), every
  pinned revision and file sha256 of V3F.5 D2 (W1, W2), the dims, FT's recipe,
  D9's C, refits and seed text; the six E3-frozen sha256s of V3F.1;
- the pins: a snapshot at the pinned revision loads; refs/main at another
  snapshot refuses an open_clip backbone; a transformers backbone without its
  pinned snapshot refuses; the LAION-2B entry's weight file by its pinned
  sha256 (another .bin refuses, a planted open_clip_model.safetensors in its
  snapshot refuses); BioCLIP-2's weight file by open_clip's rule; a missing
  pinned config refuses; open_clip resolving another file refuses;
  downloads.json written once, its statuses, a backbone it records as missing
  "not run", one it does not record "not run" (but the reference), a record
  contradicting the amendment refused; backbone_status reads no file;
- each embedder on a stub: dimension, normalisation (the model receives
  (pixel / 255 - mean) / std of its own config), no resize and no centre crop
  (a crop of another size refuses); the open_clip entries are BioCLIP-2's own
  embedder's calls (equal features for the same name); a wrong dimension
  refuses;
- FT's logit encoding through E3's unchanged normalisation and proba equals
  the softmax; FT's schedule (warm-up, cosine to 0) and its determinism under a
  seed (another seed differs);
- the view640 geometry against Ultralytics' BaseDataset.load_image on a
  3024 x 4032 image (the view's size equal, a planted box within 1 px, the
  pixels equal), a landscape and an upscaled image; cut640_task's crops are
  semisup_labeler._cut's on the view;
- D9's half sample: strata by the rarest class, half of each stratum with odd
  counts alternating from a seeded start, every class present, redraws
  counted, determinism, a draw that cannot hold a class refused.
WORLD runs real Ultralytics CPU passes in the synthetic world of
tests/test_inc2_e3.py, with E3's classifier fitted by verify.BioclipEmbedder
on the fake BioCLIP-2:
- BioCLIP-2's reproduction: the registry's reference embedder gives E3's stored
  training features exactly, E3's LR protocol on them (fit_lr_protocol) gives
  E3's pinned classifier exactly, and E3-B's boxes with E3's classifier
  through the family pass reproduce E3-B's dev files (the family's
  two_stage_B equivalence, taken with the registry's embedder);
- D2 end to end (LAION, CLIP, RN50, BC2640 on the 640-px view, FT): the
  classifier fitted and pinned once before any dev pass, three passes on
  E3-B's boxes written once, the dev ground-truth accuracy, the score record;
  a rerun keeps every file; a refit refused; a changed classifier file
  refused; a stored stage-1 box set that differs refuses the pass (nothing
  written); a backbone not downloaded recorded "not run" and refused;
  nothing written under twostage/e3_v1;
- D9 end to end (two refits): D9a's sensitivity.json the only file written
  under twostage/e3_v1, and only once; the refits stored once, every class
  present, six passes on E3-B's boxes; the record and the sentence rule; a
  rerun keeps everything;
- the CLI.

Run:  python3 tests/test_inc2_v3fam_embed.py      (V3FAM_EMBED_ONLY=units: the units only)
"""
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
import test_inc2_e3 as E3T  # noqa: E402

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools import semisup_labeler as SL  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_embed as EM  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = E3T.patched
moved = E3T.moved
KW = {"batch": 32, "device": "cpu", "lock_check": True}
REAL_BACKBONES = copy.deepcopy(EM.BACKBONES)
WORLD = {}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused) as e:
        return e
    return None


def safe(fn, *a, **k):
    """fn's value, or the exception it raised (so a check fails rather than the file crashing)."""
    try:
        return fn(*a, **k)
    except Exception as e:                       # noqa: BLE001 - reported by the check that reads it
        return e


def raised(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as e:                       # noqa: BLE001 - the check reads it
        return e
    return None


def tree(root):
    root = pathlib.Path(root)
    return {str(p.relative_to(root)): W.sha(p) for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


# ------------------------------------------------------------------ fakes (no model is downloaded)
OPENAI_MEAN, OPENAI_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
HALF = (0.5, 0.5, 0.5)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OPEN_CLIP_NORM = {"imageomics/bioclip-2": (OPENAI_MEAN, OPENAI_STD),
                  "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": (HALF, HALF),
                  "imageomics/bioclip": (OPENAI_MEAN, OPENAI_STD)}
HF_NORM = {"openai/clip-vit-large-patch14": (OPENAI_MEAN, OPENAI_STD, "clip"),
           "google/siglip2-so400m-patch14-224": (HALF, HALF, "siglip"),
           "facebook/dinov2-large": (IMAGENET_MEAN, IMAGENET_STD, "dinov2"),
           "facebook/dinov2-base": (IMAGENET_MEAN, IMAGENET_STD, "dinov2")}
LOADS = []


def _seed_of(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _quad16(t):
    """16 statistics of [n, 3, H, W]: the four quadrants' channel means, the channel sds, a constant."""
    import torch
    h, w = t.shape[2] // 2, t.shape[3] // 2
    q = [t[:, :, :h, :w], t[:, :, :h, w:], t[:, :, h:, :w], t[:, :, h:, w:]]
    return torch.cat([x.mean((2, 3)) for x in q] + [t.std((2, 3)), torch.ones(t.shape[0], 1, dtype=t.dtype)], 1)


def fake_open_clip(name, device):
    """semisup_labeler._load_backbone's contract for the open_clip names: (model with encode_image, preprocess)."""
    import torch
    repo = name[len("hf-hub:"):]
    dim = {"imageomics/bioclip-2": 768, "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": 768, "imageomics/bioclip": 512}[repo]
    mean, std = OPEN_CLIP_NORM[repo]
    P = torch.randn(16, dim, generator=torch.Generator().manual_seed(_seed_of(repo)))
    LOADS.append(name)

    class M(torch.nn.Module):
        def encode_image(self, t):
            return _quad16(t) @ P

    def pre(p):
        a = torch.from_numpy(np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (a - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    return M().to(device).eval(), pre


class FakeHF(object):
    """A transformers vision model's call (pixel_values=x) -> its output fields; it keeps the last input."""

    def __init__(self, repo, dim):
        import torch
        self.P = torch.randn(16, dim, generator=torch.Generator().manual_seed(_seed_of(repo)))
        self.last = None

    def __call__(self, pixel_values):
        import torch
        self.last = pixel_values.detach().clone()
        out = _quad16(pixel_values) @ self.P
        return types.SimpleNamespace(image_embeds=out, pooler_output=out, last_hidden_state=torch.stack([out, -out], 1))

    def to(self, _d):
        return self

    def eval(self):
        return self


def _repo_of_snapshot(snapshot):
    return pathlib.Path(snapshot).parent.parent.name[len("models--"):].replace("--", "/")


def fake_load_hf(loader, snapshot, device):
    repo = _repo_of_snapshot(snapshot)
    bb = next(k for k, v in REAL_BACKBONES.items() if v["repo"] == repo)
    LOADS.append(repo)
    return FakeHF(repo, REAL_BACKBONES[bb]["dim"])


def fake_timm_net(num_classes):
    """ResNet-50's stand-in: a small conv trunk (its 'pretrained' weights fixed), pooled features projected to
    2,048 (num_classes 0) or a fresh head (drawn from the caller's torch seed)."""
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(7)
            self.conv = torch.nn.Conv2d(3, 8, 7, stride=8)
            with torch.no_grad():
                self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * 0.1)
                self.conv.bias.zero_()
            self.register_buffer("proj", torch.randn(8, 2048, generator=g))
            self.fc = torch.nn.Linear(8, num_classes) if num_classes else None
            self.pretrained_cfg = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
            self.last = None

        def forward(self, x):
            self.last = x.detach().clone()
            f = torch.relu(self.conv(x)).mean((2, 3))
            return self.fc(f) if self.fc is not None else f @ self.proj
    return Net()


def fake_load_timm(weight_file, device, num_classes=0):
    LOADS.append("timm:%s" % pathlib.Path(weight_file).name)
    return fake_timm_net(num_classes).to(device).eval(), list(IMAGENET_MEAN), list(IMAGENET_STD)


def put(snap_root, repo, rev, files, refs_main=True):
    """A repository in the layout huggingface_hub writes: blobs/<sha256>, snapshots/<rev>/<name> -> the blob,
    refs/main. Returns {name: sha256}."""
    rd = snap_root / ("models--%s" % repo.replace("/", "--"))
    (rd / "blobs").mkdir(parents=True, exist_ok=True)
    snap = rd / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    out = {}
    for n, data in files.items():
        data = data if isinstance(data, bytes) else data.encode()
        sha = hashlib.sha256(data).hexdigest()
        (rd / "blobs" / sha).write_bytes(data)
        link = snap / n
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(os.path.relpath(rd / "blobs" / sha, link.parent), link)
        out[n] = sha
    if refs_main:
        (rd / "refs").mkdir(parents=True, exist_ok=True)
        (rd / "refs" / "main").write_text(rev)
    return out


def fake_rev(bb):
    return hashlib.sha1(("rev " + bb).encode()).hexdigest()


def fake_files(bb):
    e = REAL_BACKBONES[bb]
    out = {}
    if e["loader"] == "open_clip":
        out["open_clip_config.json"] = json.dumps({"model_cfg": {"embed_dim": e["dim"]}, "bb": bb})
        names = list(e["weights"]) if e["weights"] else ["open_clip_pytorch_model.bin"]
    elif e["loader"] == "timm":
        out["config.json"] = json.dumps({"architecture": "resnet50"})
        names = ["model.safetensors"]
    else:
        mean, std, mt = HF_NORM[e["repo"]]
        out["config.json"] = json.dumps({"model_type": mt})
        out["preprocessor_config.json"] = json.dumps({"image_mean": mean, "image_std": std, "do_normalize": True,
                                                      "rescale_factor": 1 / 255.0})
        names = ["model.safetensors"]
    for n in names:
        out[n] = ("fake weights %s %s" % (bb, n)).encode()
    return out


def fake_cache(root):
    """Every backbone of the registry in a fake cache at root; returns the registry pointed at it (the
    amendment's revisions and hashes replaced by the fake files', the same backbones pinned or not)."""
    reg = copy.deepcopy(REAL_BACKBONES)
    for bb, e in reg.items():
        rev = fake_rev(bb)
        got = put(root, e["repo"], rev, fake_files(bb))
        if e["revision"]:
            e["revision"] = rev
            e["weights"] = {n: got[n] for n in e["weights"]}
            e["configs"] = {n: got[n] for n in e["configs"]}
    return reg


@__import__("contextlib").contextmanager
def fake_world_models(cache, reg):
    with patched(EM, "hub_cache", lambda: cache), patched(EM, "BACKBONES", reg), \
            patched(EM, "_load_hf_model", fake_load_hf), patched(EM, "_load_timm", fake_load_timm), \
            patched(EM, "_open_clip_resolves", lambda repo: None), patched(EM, "_has_safetensors", lambda: True):
        yield


# ================================================================== UNITS
def u_constants():
    print("the constants (V3F.5 D2, D9; V3F.1)")
    check("D2's nine arms are v3fam's first nine challengers and score arms, in the amendment's job order",
          EM.D2_ARMS == F.CHALLENGERS[:9] and all(a in F.SCORE_ARMS for a in EM.D2_ARMS) and "D9" in F.SCORE_ARMS
          and EM.D2_ARMS == ("D2LAION", "D2CLIP", "D2DINOL", "D2DINOB", "D2SIGLIP", "D2RN50", "D2BC1", "D2BC2640",
                             "D2FT"))
    rb = REAL_BACKBONES
    repo = {a: rb[EM.ARM_BACKBONE[a]]["repo"] for a in EM.D2_ARMS}
    check("each arm's checkpoint: D2LAION the LAION-2B CLIP ViT-L/14 (P1's control), D2CLIP OpenAI's, D2DINOL/B "
          "DINOv2-L/B, D2SIGLIP SigLIP2 so400m/14 at 224, D2RN50 and D2FT ResNet-50 a1, D2BC1 BioCLIP-1, D2BC2640 "
          "BioCLIP-2 on the 640-px view",
          repo == {"D2LAION": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "D2CLIP": "openai/clip-vit-large-patch14",
                   "D2DINOL": "facebook/dinov2-large", "D2DINOB": "facebook/dinov2-base",
                   "D2SIGLIP": "google/siglip2-so400m-patch14-224", "D2RN50": "timm/resnet50.a1_in1k",
                   "D2BC1": "imageomics/bioclip", "D2BC2640": "imageomics/bioclip-2", "D2FT": "timm/resnet50.a1_in1k"}
          and EM.ARM_CUT["D2BC2640"] == "view640" and all(EM.ARM_CUT[a] == "original" for a in EM.D2_ARMS
                                                           if a != "D2BC2640")
          and EM.REFERENCE == "BC2" and EM.E3_ARM == "B", repo)
    want = {"BC2": ("2957b322090f9cb17ae72c71981c7218a28d81e0",
                    {"open_clip_model.safetensors": "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
                     "open_clip_pytorch_model.bin": "035208dd616321f6c220384efaa1f10e9fcd8e1f2998ddefc88687718d186a85"}),
            "LAION": ("1627032197142fbe2a7cfec626f4ced3ae60d07a",
                      {"open_clip_pytorch_model.bin": "5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b"}),
            "CLIP": ("32bd64288804d66eefd0ccbe215aa642df71cc41",
                     {"model.safetensors": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"}),
            "SIGLIP": ("78e403963a4f6a3640d07803284752326fdf4edf",
                       {"model.safetensors": "2041e5330b3db96299b26bc3a6506c7fa8d5e61d5c94f267dd8aa4cfb12c1050"}),
            "BC1": ("ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
                    {"open_clip_pytorch_model.bin": "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"})}
    check("the pinned revisions and weight sha256s of V3F.5 D2 (W1, W2), literally",
          all(rb[k]["revision"] == r and rb[k]["weights"] == w for k, (r, w) in want.items())
          and rb["LAION"]["configs"] == {"open_clip_config.json":
                                         "de8f2f846f38ccb247b9648512c31ea79b51fde6a623a8d5e85d96cfe10d943a"}
          and rb["BC2"]["configs"] == {"open_clip_config.json":
                                       "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8"}
          and all(rb[k]["revision"] is None and rb[k]["weights"] is None for k in ("DINOL", "DINOB", "RN50")))
    cites = {"LAION": ("16270321", "d07a", "5ddb4733", "7e2b"), "CLIP": ("32bd6428", "cc41", "a2bf730a", "6dcb"),
             "SIGLIP": ("78e40396", "4edf", "2041e533", "1050"), "BC1": ("ce901ab3", "f01c", "e380384f", "0ef4")}
    check("  each agrees with the amendment's own citation (prefix ... suffix)",
          all(rb[k]["revision"].startswith(a) and rb[k]["revision"].endswith(b)
              and list(rb[k]["weights"].values())[0].startswith(c) and list(rb[k]["weights"].values())[0].endswith(d)
              for k, (a, b, c, d) in cites.items()) and rb["BC2"]["revision"].startswith("2957b322"))
    check("the dims: BioCLIP-2, LAION and OpenAI CLIP-L/14 768, DINOv2-L 1,024, DINOv2-B 768, SigLIP2 1,152, "
          "ResNet-50 2,048, BioCLIP-1 512; open_clip for the 'hf-hub:' ones",
          {k: v["dim"] for k, v in rb.items()} == {"BC2": 768, "LAION": 768, "CLIP": 768, "DINOL": 1024, "DINOB": 768,
                                                   "SIGLIP": 1152, "RN50": 2048, "BC1": 512}
          and {k for k, v in rb.items() if v["loader"] == "open_clip"} == {"BC2", "LAION", "BC1"})
    f = EM.FT_CFG
    check("FT: AdamW lr 3e-4, weight decay 0.05, 1 warm-up epoch, 20 epochs, batch 128, label smoothing 0.1, AMP, "
          "flips, RandomResizedCrop 0.7-1.0 / 3/4-4/3, ColorJitter 0.2", f["lr"] == 3e-4 and f["weight_decay"] == 0.05
          and f["warmup_epochs"] == 1 and f["epochs"] == 20 and f["batch"] == 128 and f["label_smoothing"] == 0.1
          and f["amp"] is True and f["hflip"] == f["vflip"] == 0.5 and tuple(f["rrc_scale"]) == (0.7, 1.0)
          and abs(f["rrc_ratio"][0] - 0.75) < 1e-12 and abs(f["rrc_ratio"][1] - 4 / 3.0) < 1e-12
          and tuple(f["jitter"]) == (0.2, 0.2, 0.2))
    check("D9: C = 30, 30 refits, seed text inc2/v3fam/d9/half/<b>; the sentences fixed",
          EM.D9_C == 30.0 and EM.D9_REFITS == 30 and EM.D9_SEED_TEXT % 7 == "inc2/v3fam/d9/half/7"
          and "understate" in EM.D9_SENTENCE["exceeds"] and "less spread" in EM.D9_SENTENCE["within"])
    want6 = {"twostage": "8fe2edc0e0d2a00d07182c3746407281a112c9e50c76662bc9f48a558b328ddc",
             "scorer": "18c008371cd2c021104d034d06b837564016f4509e8a969c2a3f850b09dd2de4",
             "scorer_sidecar": "de7aacda903780b32944866be0b4daa5c6b1f76e2c86965f7a99d85f443d5e52",
             "scorer_native": "ba9eeb928a2b5a0756ae8ac0144b54df2fdfb3570fd011f66b4a8022039d6b7d",
             "verify": "602eba3c935f375179d6f922dce28bbf89006054e5bcec5894f0c9cc7b87aa10",
             "semisup_labeler": "03b28c50699b30a32720e383ff2b713a2638926650007acb92a367d1f450458c"}
    now = {k: C.sha256_file(pathlib.Path(v).resolve()) for k, v in TS.CODE_FILES.items()}
    check("the six E3-frozen files of V3F.1 hash as recorded (literally), and v3fam.FROZEN_SHA256 holds them",
          now == want6 and F.FROZEN_SHA256 == want6, {k: now.get(k, "")[:12] for k in want6})
    cs = EM.code_shas()
    check("code_shas: the family's and this module's", set(cs) == set(F.code_shas()) | {"v3fam_embed"})


def u_pins():
    print("the pins: revision, weight file, configs, downloads.json")
    cache = W.TMP / "hub_units"
    reg = fake_cache(cache)
    with patched(EM, "hub_cache", lambda: cache), patched(EM, "BACKBONES", reg), \
            patched(EM, "_open_clip_resolves", lambda repo: None), patched(EM, "_has_safetensors", lambda: True):
        ck = EM.checkpoint("BC2", downloads={})
        check("the reference loads on the amendment's pins alone: its revision, open_clip's choice of weight file "
              "(safetensors importable: open_clip_model.safetensors), every pinned file hashed",
              ck["revision"] == fake_rev("BC2") and ck["weight_file"] == "open_clip_model.safetensors"
              and set(ck["files"]) == {"open_clip_config.json", "open_clip_model.safetensors",
                                       "open_clip_pytorch_model.bin"} and ck["pins_from"] == "amendment", ck)
        with patched(EM, "_has_safetensors", lambda: False):
            ck2 = EM.checkpoint("BC2", downloads={})
        check("  without safetensors open_clip opens open_clip_pytorch_model.bin, also pinned",
              ck2["weight_file"] == "open_clip_pytorch_model.bin", ck2["weight_file"])
        e = refused(EM.checkpoint, "LAION", downloads={})
        check("every other backbone needs downloads.json's record: without it LAION is not run",
              e is not None and EM.NOT_RUN_DOWNLOAD in str(e), e)
        with moved(EM.downloads_path()):
            dl = EM.record_downloads()
            e_once = refused(EM.record_downloads)
            st = {k: v["status"] for k, v in dl["backbones"].items()}
            check("downloads.json: every backbone ok in the fake cache, its revision, files and weight file; written "
                  "once", set(st.values()) == {"ok"} and dl["backbones"]["RN50"]["revision"] == fake_rev("RN50")
                  and dl["backbones"]["RN50"]["weight_file"] == "model.safetensors" and e_once is not None, st)
            for bb in ("LAION", "CLIP", "DINOL", "DINOB", "SIGLIP", "RN50", "BC1"):
                ok = EM.checkpoint(bb)
                if ok["revision"] != fake_rev(bb):
                    check("checkpoint %s" % bb, False, ok)
            check("with it every backbone's snapshot checks: DINOv2-L/B and ResNet-50 pinned by the record",
                  EM.checkpoint("DINOL")["pins_from"] == "downloads.json"
                  and EM.checkpoint("CLIP")["pins_from"] == "amendment")
            st0 = {a: EM.backbone_status(a)["status"] for a in EM.D2_ARMS}
            check("backbone_status: every arm ok (no file opened beyond downloads.json)", set(st0.values()) == {"ok"},
                  st0)
            rd = cache / "models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K"
            (rd / "refs" / "main").write_text("f" * 40)
            e1 = refused(EM.checkpoint, "LAION")
            (rd / "refs" / "main").write_text(fake_rev("LAION"))
            check("refs/main at another snapshot refuses an open_clip backbone (open_clip resolves refs/main)",
                  e1 is not None and "refs/main" in str(e1), e1)
            snap = rd / "snapshots" / fake_rev("LAION")
            planted = snap / "open_clip_model.safetensors"
            planted.write_bytes(b"other weights")
            e2 = raised(EM.checkpoint, "LAION")
            planted.unlink()
            check("a planted open_clip_model.safetensors in LAION's snapshot refuses: open_clip would open it, and only "
                  "open_clip_pytorch_model.bin is pinned", isinstance(e2, F.V3famRefused)
                  and "not a pinned weight file" in str(e2), e2)
            binp = snap / "open_clip_pytorch_model.bin"
            target = binp.resolve()
            raw = target.read_bytes()
            target.write_bytes(raw + b"x")
            e3 = refused(EM.checkpoint, "LAION")
            target.write_bytes(raw)
            check("LAION's .bin by its pinned sha256: other bytes refuse", e3 is not None and "hashes to" in str(e3), e3)
            cfgp = snap / "open_clip_config.json"
            keep = cfgp.resolve().read_bytes()
            cfgp.unlink()
            e4 = refused(EM.checkpoint, "LAION")
            put(cache, "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", fake_rev("LAION"), {"open_clip_config.json": keep})
            check("a missing pinned config refuses", e4 is not None and "lacks the pinned file" in str(e4), e4)
            cr = cache / "models--openai--clip-vit-large-patch14"
            put(cache, "openai/clip-vit-large-patch14", "2" * 40, {"model.safetensors": b"another snapshot",
                                                                   "config.json": "{}"})
            ck_c = EM.checkpoint("CLIP")
            (cr / "refs" / "main").write_text(fake_rev("CLIP"))
            check("a transformers backbone loads its pinned snapshot by revision, whatever refs/main names",
                  ck_c["snapshot"].endswith(fake_rev("CLIP")) and ck_c["weight_sha256"]
                  == reg["CLIP"]["weights"]["model.safetensors"], ck_c)
            sp = cache / "models--openai--clip-vit-large-patch14" / "snapshots"
            os.replace(str(sp / fake_rev("CLIP")), str(sp / ("0" * 40)))
            e5 = refused(EM.checkpoint, "CLIP")
            os.replace(str(sp / ("0" * 40)), str(sp / fake_rev("CLIP")))
            check("a transformers backbone without its pinned snapshot refuses (another snapshot is never loaded)",
                  e5 is not None and "no snapshot" in str(e5), e5)
            with patched(EM, "_open_clip_resolves", lambda repo: pathlib.Path("/elsewhere/weights.bin")):
                e6 = refused(EM.checkpoint, "BC1")
            check("open_clip resolving another file than the pinned one refuses", e6 is not None
                  and "open_clip resolves" in str(e6), e6)
            bad = copy.deepcopy(dl)
            bad["backbones"]["SIGLIP"].update(status="missing", why="the download failed")
            bad["backbones"]["CLIP"]["revision"] = "1" * 40
            e7 = refused(EM.checkpoint, "SIGLIP", downloads=bad)
            e8 = refused(EM.checkpoint, "CLIP", downloads=bad)
            stb = EM.backbone_status("D2SIGLIP", downloads=bad)
            check("a backbone downloads.json records as missing is not run (backbone_status says so); a record "
                  "contradicting the amendment's revision refuses", e7 is not None and EM.NOT_RUN_DOWNLOAD in str(e7)
                  and stb["status"] == "not run" and e8 is not None and "the amendment pins" in str(e8), (e7, e8, stb))
        with moved(EM.downloads_path()):
            e9 = refused(EM.checkpoint, "DINOB")
        check("DINOv2-B without downloads.json: not run (the amendment pins no revision for it)", e9 is not None
              and EM.NOT_RUN_DOWNLOAD in str(e9), e9)
        cache2 = W.TMP / "hub_units_missing"
        put(cache2, reg["BC2"]["repo"], fake_rev("BC2"), fake_files("BC2"))
        with patched(EM, "hub_cache", lambda: cache2), moved(EM.downloads_path()):
            d2 = EM.record_downloads()
            (EM.downloads_path()).unlink()
        check("record_downloads on a cache holding only BioCLIP-2: the others missing", d2["backbones"]["BC2"]["status"]
              == "ok" and all(d2["backbones"][k]["status"] == "missing" for k in reg if k != "BC2"),
              {k: v["status"] for k, v in d2["backbones"].items()})
    WORLD["unit_cache"] = (cache, reg)


def u_embedders():
    print("each embedder on a stub: dimension, normalisation, no resize and no centre crop")
    from PIL import Image
    cache, reg = WORLD["unit_cache"]
    rng = np.random.default_rng(3)
    crops = [Image.fromarray((rng.random((224, 224, 3)) * 255).astype(np.uint8)) for _ in range(5)]
    with patched(SL, "_load_backbone", fake_open_clip), patched(SL, "_device", lambda: "cpu"), \
            fake_world_models(cache, reg), moved(EM.downloads_path()):
        EM.record_downloads()
        ref = V.BioclipEmbedder()
        bc2, ck = EM.load_embedder("BC2")
        oc = EM.OpenClipEmbedder("hf-hub:imageomics/bioclip-2")
        check("the reference is verify.BioclipEmbedder itself (E3's embedder: same name, dim, features exactly); an "
              "OpenClipEmbedder of the same name makes the same calls (same features)",
              type(bc2) is V.BioclipEmbedder and bc2.name == ref.name == "hf-hub:imageomics/bioclip-2"
              and bc2.dim == 768 and np.array_equal(bc2(crops), ref(crops)) and np.array_equal(oc(crops), ref(crops))
              and ck["weight_file"] == "open_clip_model.safetensors", (bc2.name, ref.name))
        dims, norm_ok, names = {}, {}, {}
        for bb in ("LAION", "CLIP", "DINOL", "DINOB", "SIGLIP", "RN50", "BC1"):
            emb, ck = EM.load_embedder(bb)
            out = emb(crops)
            dims[bb] = (out.shape, emb.dim, out.dtype == np.float32)
            names[bb] = emb.name
            if reg[bb]["loader"] != "open_clip":
                model = emb.model
                mean, std = (IMAGENET_MEAN, IMAGENET_STD) if bb == "RN50" else HF_NORM[reg[bb]["repo"]][:2]
                direct = np.stack([np.asarray(c, dtype=np.float32) / 255.0 for c in crops]).transpose(0, 3, 1, 2)
                want = (direct - np.asarray(mean, np.float32).reshape(1, 3, 1, 1)) / np.asarray(std, np.float32).reshape(
                    1, 3, 1, 1)
                got = model.last.numpy()
                norm_ok[bb] = got.shape == (5, 3, 224, 224) and float(np.abs(got - want).max()) < 1e-5
        check("every backbone's dimension as registered, float32 rows, one per crop",
              all(s == (5, d) and emb_d == d and f32 for (s, emb_d, f32), d in
                  ((dims[k], reg[k]["dim"]) for k in dims)), dims)
        check("the transformers and timm backbones receive each 224-px crop pixel for pixel (no resize, no centre "
              "crop), as (x / 255 - mean) / std with their own config's mean and std (DINOv2 and ResNet-50 ImageNet's, "
              "OpenAI CLIP's, SigLIP2's 0.5)", norm_ok and all(norm_ok.values()), norm_ok)
        check("open_clip backbones through semisup_labeler's loader by their 'hf-hub:' names (LAION's is the LAION-2B "
              "checkpoint)", "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K" in LOADS
              and "hf-hub:imageomics/bioclip" in LOADS and names["LAION"].startswith("hf-hub:laion/"), names)
        emb, _ck = EM.load_embedder("DINOB")
        e = raised(emb, [Image.new("RGB", (256, 256))])
        check("a crop of another size refuses (the backbone takes E3's 224-px crop as it is)", e is not None
              and "as it is" in str(e), e)
        reg2 = copy.deepcopy(reg)
        reg2["DINOB"]["dim"] = 1024
        with patched(EM, "BACKBONES", reg2):
            e2 = refused(EM.load_embedder, "DINOB")
        check("an embedder whose dimension is not the registry's refuses", e2 is not None and "-d" in str(e2), e2)
        sp = cache / "models--facebook--dinov2-base" / "snapshots" / fake_rev("DINOB")
        check("_hf_norm reads the snapshot's preprocessor_config.json", EM._hf_norm(sp) == (list(IMAGENET_MEAN),
                                                                                          list(IMAGENET_STD)))
        EM.downloads_path().unlink()


def u_ft_encode():
    print("FT's logits through E3's normalisation and proba")
    rng = np.random.default_rng(5)
    z = rng.normal(size=(200, C.NC)) * 6 + rng.normal(size=(200, 1)) * 40
    cfg = EM.ft_classifier_cfg(np.ones(C.NC) / C.NC)
    P = TS.proba(V._norm(EM.ft_encode(z)), cfg["W"], cfg["b"], cfg["classes"])
    e = np.exp(z - z.max(axis=1, keepdims=True))
    Ps = e / e.sum(axis=1, keepdims=True)
    rel = float((np.abs(P - Ps) / Ps).max())
    enc = EM.ft_encode(z)
    check("verify._norm leaves the encoding a unit vector and twostage.proba with W = FT_K [I | 0], b = 0 returns the "
          "network's softmax (max relative difference %.2g, float32 rounding)" % rel,
          rel < 1e-5 and np.allclose(np.linalg.norm(enc, axis=1), 1.0, atol=1e-6) and enc.shape == (200, C.NC + 1)
          and EM.FTEmbedder.dim == C.NC + 1 and list(cfg["classes"]) == list(range(C.NC)))
    e = raised(EM.ft_encode, np.full((1, C.NC), 0.0) + np.eye(C.NC)[0] * 2 * EM.FT_K)
    check("centred logits that reach FT_K refuse", e is not None, e)


def u_view640():
    print("the 640-px view against Ultralytics' BaseDataset.load_image")
    import cv2
    from PIL import Image
    from ultralytics.data.base import BaseDataset

    def loader(path):
        ds = BaseDataset.__new__(BaseDataset)
        ds.ims, ds.im_files, ds.npy_files = [None], [str(path)], [pathlib.Path(str(path) + ".npy")]
        ds.im_hw0, ds.im_hw, ds.buffer, ds.max_buffer_length, ds.cache = [None], [None], [], 0, None
        ds.imgsz, ds.cv2_flag, ds.augment, ds.prefix = 640, cv2.IMREAD_COLOR, False, ""
        im, hw0, hw = ds.load_image(0)
        return np.ascontiguousarray(im[..., ::-1]), hw0, hw

    def located(a, colour):
        m = np.all(np.abs(a.astype(np.int16) - np.asarray(colour, np.int16)) <= 40, axis=2)
        ys, xs = np.nonzero(m)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)

    res = {}
    for name, (w0, h0), box in (("portrait", (3024, 4032), (1000, 1500, 1800, 2600)),
                                ("landscape", (4032, 3024), (2500, 400, 3300, 1300)),
                                ("upscaled", (500, 300), (100, 60, 300, 200)),
                                ("ceil", (1000, 777), (200, 150, 600, 500))):
        yy, xx = np.mgrid[0:h0, 0:w0]
        a = np.stack([(xx * 200 // w0), (yy * 200 // h0), np.full_like(xx, 60)], axis=2).astype(np.uint8)
        a[box[1]:box[3], box[0]:box[2]] = (250, 20, 200)
        p = W.TMP / "view640" / ("%s.png" % name)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(a).save(p)
        mine, wh0, wh = EM.view640(p)
        theirs, hw0, hw = loader(p)
        lm, lt = located(mine, (250, 20, 200)), located(theirs, (250, 20, 200))
        mapped = np.asarray(box, np.float64) * np.array([wh[0] / w0, wh[1] / h0, wh[0] / w0, wh[1] / h0])
        res[name] = {"size": (wh, (hw[1], hw[0])), "orig": (wh0, (hw0[1], hw0[0])),
                     "located": float(np.abs(lm - lt).max()), "mapped": float(np.abs(mapped - lt).max()),
                     "mad": float(np.abs(mine.astype(np.int16) - theirs.astype(np.int16)).mean())
                     if mine.shape == theirs.shape else None}
    check("the view's size is load_image's (long side 640, short side ceil(side x r), upscaling too); the planted box "
          "lies within 1 px of load_image's, mapped by the per-axis gains; the pixels equal",
          all(v["size"][0] == v["size"][1] and v["orig"][0] == v["orig"][1] and v["located"] <= 1.0
              and v["mapped"] <= 1.0 and v["mad"] == 0.0 for v in res.values())
          and res["portrait"]["size"][0] == (480, 640) and res["upscaled"]["size"][0] == (640, 384)
          and res["ceil"]["size"][0] == (640, 498), res)
    tagged, _boxes = E3T.exif_image("view640_tag6", 6, (240, 30, 30))
    tv, twh0, twh = EM.view640(tagged)
    seen = np.asarray(Image.open(tagged).transpose(Image.Transpose.ROTATE_270).convert("RGB"))
    check("an image tagged 6 is viewed in its EXIF-transposed frame (its labels' frame, 80 x 120 seen as stored 120 x "
          "80): view %s" % (twh,), twh0 == TS.header(str(tagged))[1] == (80, 120) and twh == (427, 640)
          and seen.shape[:2] == (120, 80), (twh0, twh))
    p = W.TMP / "view640" / "portrait.png"
    rows = [{"crop_id": 0, "cx": 0.46, "cy": 0.5, "w": 0.26, "h": 0.27}, {"crop_id": 1, "cx": 0.1, "cy": 0.9,
                                                                         "w": 0.05, "h": 0.04}]
    img, ids, arrays, err = EM.cut640_task((str(p), rows))
    view, _wh0, (w, h) = EM.view640(p)
    want = [np.asarray(SL._cut(Image.fromarray(view), dict(r, W=w, H=h))) for r in rows]
    _i, _d, a640, e640, hd = EM.e3_cut640((str(p), rows))
    check("cut640_task: semisup_labeler._cut on the view (224 px, the boxes in the view's pixels); e3_cut640 the same "
          "crops with the image header", err is None and ids == [0, 1] and all(np.array_equal(x, y) for x, y in
                                                                              zip(arrays, want))
          and arrays[0].shape == (224, 224, 3) and e640 is None and all(np.array_equal(x, y) for x, y in zip(a640, want))
          and hd == TS.header(str(p)), err)
    shutil.rmtree(str(W.TMP / "view640"), ignore_errors=True)


def u_ft_schedule():
    print("FT's schedule and its determinism under a seed")
    lam = [EM.ft_lr_lambda(i, 10) for i in range(200)]
    check("1 warm-up epoch rising linearly to the full rate, then cosine to 0 (per step)",
          abs(lam[0] - 0.1) < 1e-12 and abs(lam[9] - 1.0) < 1e-12 and abs(lam[10] - 1.0) < 1e-12
          and lam[105] < lam[50] and lam[-1] < 1e-3 and all(b <= a + 1e-12 for a, b in zip(lam[10:], lam[11:])))
    rng = np.random.default_rng(11)
    crops = (rng.random((20, 224, 224, 3)) * 255).astype(np.uint8)
    y = rng.integers(0, C.NC, 20)
    init = lambda: (fake_timm_net(C.NC), list(IMAGENET_MEAN), list(IMAGENET_STD))  # noqa: E731
    cfg = {"epochs": 2, "batch": 8}
    m1, r1 = EM.train_ft(crops, y, 0, init, device="cpu", cfg=cfg)
    m2, _r2 = EM.train_ft(crops, y, 0, init, device="cpu", cfg=cfg)
    m3, _r3 = EM.train_ft(crops, y, 1, init, device="cpu", cfg=cfg)
    s1, s2, s3 = m1.state_dict(), m2.state_dict(), m3.state_dict()
    same = all(np.array_equal(s1[k].numpy(), s2[k].numpy()) for k in s1)
    diff = any(not np.array_equal(s1[k].numpy(), s3[k].numpy()) for k in s1)
    check("the same seed gives the same final weights (head, data order and augmentations from stable_int(%r)); "
          "another seed other weights; the record (3 steps an epoch, AMP off on the CPU)" % (EM.FT_SEED_TEXT % 0),
          same and diff and r1["steps_per_epoch"] == 3 and r1["steps"] == 6 and r1["amp"] is False
          and r1["seed_int"] == C.stable_int("inc2/v3fam/d2/ft/s0") and len(r1["epochs"]) == 2, r1["epochs"])


def u_half_sample():
    print("D9b's half sample (strata by the rarest class)")
    rng = np.random.default_rng(2)
    groups, y = [], []
    for i in range(41):
        s = "sess%02d" % i
        labels = list(rng.integers(0, 9, 12))
        if i in (3, 17, 29):
            labels.append(11)                    # class 11: three sessions
        if i in (5, 6, 7, 8, 30):
            labels.append(10)                    # class 10: five sessions
        if i == 12:
            labels.append(9)                     # class 9: one session
        if i in (1, 2):
            labels.append(12)
        groups += [s] * len(labels)
        y += labels
    groups, y = np.asarray(groups), np.asarray(y)
    required = tuple(sorted(set(y.tolist())))
    sel, rec = EM.half_sample(groups, y, 0, required)
    sel2, rec2 = EM.half_sample(groups, y, 0, required)
    _sel3, rec3 = EM.half_sample(groups, y, 1, required)
    counts = np.bincount(y, minlength=C.NC)
    strata = {}
    for s in sorted(set(groups.tolist())):
        present = sorted(set(y[groups == s].tolist()))
        strata.setdefault(min(present, key=lambda c: (counts[c], c)), []).append(s)
    sizes_ok = all((rec["strata"].get(C.CLASS_NAMES[c]) or [None])[0] == len(ss) for c, ss in strata.items())
    halves = [rec["strata"].get(C.CLASS_NAMES[c]) or [0, -1] for c in sorted(strata)]
    odd_k = [k for n, k in halves if n % 2]
    odd_n = [n for n, k in halves if n % 2]
    alternate = all((k == (n + 1) // 2) == ((i + rec["start"]) % 2 == 0) for i, (n, k) in enumerate(zip(odd_n, odd_k)))
    even_ok = all(k == n // 2 for n, k in halves if n % 2 == 0)
    drawn_by = {c: [s for s in rec["sessions"] if s in ss] for c, ss in strata.items()}
    check("each session in the stratum of the rarest class it holds; half of every even stratum, odd strata rounding "
          "up and down alternately from the seeded start; the sessions drawn come from their strata in those numbers",
          sizes_ok and alternate and even_ok and len(odd_k) >= 2
          and all(len(drawn_by[c]) == (rec["strata"].get(C.CLASS_NAMES[c]) or [0, -1])[1] for c in strata), halves)
    check("every class present in the draw; the mask is the drawn sessions' crops; deterministic under stable_int(%r); "
          "another b draws other sessions" % (EM.D9_SEED_TEXT % 0),
          set(y[sel].tolist()) == set(required) and np.array_equal(sel, sel2) and rec == rec2
          and rec3["sessions"] != rec["sessions"] and rec["seed"] == C.stable_int("inc2/v3fam/d9/half/0")
          and np.array_equal(sel, np.isin(groups, rec["sessions"])) and rec["n_crops"] == int(sel.sum()))
    redraws = [EM.half_sample(groups, y, b, required)[1]["redraws"] for b in range(30)]
    check("a draw missing a class (here class 9's single session left out) is redrawn and counted: %d redraws over 30 "
          "draws, every final draw holding all classes" % sum(redraws), sum(redraws) > 0)
    e = refused(EM.half_sample, groups, y, 0, required + (C.NC - 1,) if C.NC - 1 not in required else required + (99,),
                max_redraws=3)
    check("a class no draw can hold refuses after the redraw cap", e is not None and "no draw" in str(e), e)
    with patched(S, "testing", lambda: False):
        ok = EM.d9_required(range(C.NC))
        e2 = refused(EM.d9_required, range(C.NC - 1))
    check("production: every refit keeps all 13 classes (E3's fit held 13; a classifier of 12 is refused)",
          ok == tuple(range(C.NC)) and e2 is not None, e2)


def u_cli():
    print("the CLI (units)")
    import contextlib
    import io as _io

    def cli(*a):
        err = _io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(_io.StringIO()):
            try:
                rc = EM.main(list(a))
            except SystemExit as e:
                rc = e.code
        return rc, err.getvalue()
    out = [cli("score"), cli("score", "--arm", "D2NOPE"), cli("plan", "--arm", "D2LAION"), cli("nonsense")]
    check("score without a D2 arm or D9, a verb with --arm, an unknown verb: exit 2",
          all(rc == 2 for rc, _e in out) and "[inc2.v3fam_embed] ERROR" in out[0][1], out)


UNITS = ("u_constants", "u_pins", "u_embedders", "u_ft_encode", "u_view640", "u_ft_schedule", "u_half_sample", "u_cli")


# ================================================================== WORLD
def w_setup():
    print("the world: E3's fixtures and records with BioCLIP-2 = the fake, the fake cache, the family's equivalences")
    E3T.setup_world()
    t0 = time.time()
    cache = W.TMP / "hub"
    reg = fake_cache(cache)
    WORLD.update(cache=cache, reg=reg)
    bc2 = V.BioclipEmbedder()
    for arm in ("M", "A", "B"):
        TS.score_arm(arm, embedder=bc2, reported=False, **KW)
    check("E3's records exist, its classifier fitted by verify.BioclipEmbedder (%s) (%.0fs)"
          % (json.loads((TS.classifier_dir() / "classifier.json").read_text())["embedder"]["name"], time.time() - t0),
          all((TS.arm_dir(a, s) / "dev.json").exists() for a in "MAB" for s in (0, 1, 2)))
    with fake_world_models(cache, reg):
        EM.record_downloads()
        emb, ck = EM.load_embedder("BC2")
        WORLD["bc2"] = emb
        F.equivalence_dev_identity(**KW)
        t2 = F.equivalence_two_stage("B", emb, **KW)
        F.equivalence_injection(**KW)
    per = t2["record"]["per"]
    check("BioCLIP-2's reproduction of E3-B's files: E3-B's boxes and E3's pinned classifier, named by the registry's "
          "reference embedder through the family pass, reproduce E3-B's dev files (diff %s) and stage-1 records "
          "(top classes equal)" % [v["diff"]["max_abs_diff"] for v in per.values()],
          t2["record"]["status"] == "passed" and all(v["diff"]["max_abs_diff"] == 0.0 and v["stage1"]["top_cls_equal"]
                                                     for v in per.values()), per)
    TS.verdict(out_dir=C.INC_DIR / "cap_e3", testing_ok=True, resamples=30)
    E3T.fabricate_verdict(qualifying=("M", "A", "B"), chosen="B")


def w_reference():
    print("the reference: E3's features and E3's classifier reproduced exactly")
    rows, rec = EM.training_crops()
    with fake_world_models(WORLD["cache"], WORLD["reg"]):
        emb = WORLD["bc2"]
        images = TS._group(rows)
        ids, Fm, _st = V._embed_images(images, TS._Table(rows), emb, workers=V._Workers(1), batch=TS.EMBED_BATCH)
        ids2, F2, _st2 = EM.embed_crops(images, TS._Table(rows), emb, V._Workers(1))
    stored = SC.load_npz(TS.classifier_dir() / "train_emb.npz")
    check("E3's %d training crops (train_crops.csv by its pinned sha256, base_v2 through E3's never-train checks, no "
          "evaluation image) embedded by the registry's reference give train_emb.npz's features exactly; embed_crops "
          "with verify._cut_task is verify._embed_images" % len(rows),
          np.array_equal(Fm, stored["F"]) and np.array_equal(ids, stored["crop_id"]) and np.array_equal(F2, Fm)
          and np.array_equal(ids2, ids) and rec["never_train"]["guard"]["checked"] and rec["evaluation_overlap"] == 0,
          float(np.abs(Fm - stored["F"]).max()))
    y = np.array([r["label"] for r in rows])
    g = np.array([r["session"] for r in rows])
    arrays, frec = EM.fit_lr_protocol(V._norm(Fm), y, g)
    with np.load(str(TS.classifier_dir() / "classifier.npz")) as z:
        e3 = {k: z[k] for k in z.files}
    crec = json.loads((TS.classifier_dir() / "classifier.json").read_text())
    check("E3's LR protocol on them (fit_lr_protocol) gives E3's pinned classifier exactly: W, b, classes, prior, "
          "every fold model, C = %g" % frec["fit"]["C"],
          set(arrays) == set(e3) and all(np.array_equal(arrays[k], e3[k]) for k in e3)
          and frec["fit"]["C"] == crec["cv"]["chosen_C"] and frec["cv"]["per_c"] == crec["cv"]["per_c"],
          {k: (arrays[k].shape, e3[k].shape) for k in e3 if k in arrays})
    X, yd, gd, _drec = EM.d9_design()
    clf, conv = TS._fit_lr(X, yd, crec["cv"]["chosen_C"])
    check("D9's design (E3's stored features, normalised as E3's fit did) refitted on every crop at E3's C gives E3's "
          "classifier exactly", conv and np.array_equal(clf.coef_, e3["W"]) and np.array_equal(clf.intercept_, e3["b"])
          and np.array_equal(yd, y) and np.array_equal(gd, g))
    planted = dict(rows[0], key=json.loads(C.manifest_path("dev").read_text().splitlines()[0])["key"])
    with patched(EM, "_read_crops_csv", lambda p: [planted] + rows[1:]):
        e = refused(EM.training_crops, images=False)
    check("a training crop naming a dev image is refused", e is not None, e)
    base = C.read_manifest(EM.C2.v2_manifest_path("base_v2"))
    fake_dev = W.TMP / "planted_eval" / "dev.jsonl"
    C.write_manifest(fake_dev, [next(r for r in base if r["key"] == rows[5]["key"])])
    ev0 = EM.T.eval_manifest_paths
    with patched(EM.T, "eval_manifest_paths", lambda: list(ev0()) + [("v2", "dev", fake_dev)]):
        e2 = refused(EM.training_crops, images=False)
    check("a training image that an evaluation manifest (dev, test or imageweeds) lists is refused", e2 is not None
          and "evaluation images" in str(e2), e2)
    fake_ood = W.TMP / "planted_eval" / "ood22.jsonl"
    C.write_manifest(fake_ood, [next(r for r in base if r["key"] == rows[5]["key"])])
    with patched(EM.T, "eval_manifest_paths", lambda: list(ev0()) + [("v1", "ood22", fake_ood)]):
        ok = safe(EM.training_crops, images=False)
    check("  one that only ood22 or ood23 lists is not (base_v2's tsw rows are those exams' images by design)",
          isinstance(ok, tuple) and len(ok[0]) == len(rows), ok)
    tc = TS.classifier_dir() / "train_crops.csv"
    raw = tc.read_bytes()
    tc.write_bytes(raw + b"\n")
    e3x = refused(EM.training_crops, images=False)
    tc.write_bytes(raw)
    check("a train_crops.csv that does not hash as E3's pin records is refused", e3x is not None
          and "does not hash" in str(e3x), e3x)


def _d2(arm, **kw):
    with fake_world_models(WORLD["cache"], WORLD["reg"]):
        return EM.score_d2(arm, workers=V._Workers(1), **dict(KW, **kw))


def w_d2():
    print("D2 end to end: D2LAION")
    before = tree(TS.root())
    t0 = time.time()
    rec = _d2("D2LAION")
    key = "LAION"
    pin = json.loads(EM.pin_path(key).read_text())
    clf = json.loads((EM.clf_dir(key) / "classifier.json").read_text())
    docs = [json.loads((F.arm_dir("D2LAION", s) / "dev.json").read_text()) for s in (0, 1, 2)]
    check("the classifier fitted on E3's crops with LAION's embedder by E3's protocol and pinned once before any dev "
          "pass; three passes on E3-B's boxes written once (stage-1 reproduced), the dev ground-truth accuracy, the "
          "score record (%.0fs)" % (time.time() - t0),
          rec["status"] == "complete" and [f["status"] for f in rec["dev"]] == ["written"] * 3
          and rec["dev_gt"][0]["status"] == "written" and pin["json_sha256"] == W.sha(EM.clf_dir(key) / "classifier.json")
          and clf["backbone"]["repo"] == "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" and clf["cut"] == "original"
          and clf["crops"]["train_crops"]["sha256"] == W.sha(TS.classifier_dir() / "train_crops.csv")
          and all(d["created_utc"] > pin["written_utc"] for d in docs)
          and all(d["stage1_source"]["vs_e3_stage1"]["rows"] > 0 and d["arm"] == "D2LAION" for d in docs)
          and all(d["stage2"]["embedder"].startswith("hf-hub:laion/") for d in docs)
          and docs[0]["scorer_sha256"].startswith("TEST-V3F-")
          and set(docs[0]["equivalence"]) == {"dev_identity.json", "injection.json", "two_stage_B.json"},
          (rec.get("dev"), pin))
    s1 = SC.load_npz(F.arm_dir("D2LAION", 0) / "dev.stage1.npz")
    e3s1 = SC.load_npz(TS.arm_dir("B", 0) / "dev.stage1.npz")
    check("  paired on boxes: the arm's stage-1 rows are E3-B's (xyxy, q); its per-box features 768-d on dev",
          np.array_equal(s1["xyxy"], e3s1["xyxy"]) and np.array_equal(s1["q"], e3s1["q"]) and s1["feat"].shape[1] == 768)
    sr = json.loads(EM.score_record_path("D2LAION").read_text())
    check("capacity/v3fam_score_d2laion.json: dev file names (relative), sha256s, status and counts; no path",
          sr["exam"] == "dev" and sr["counts"]["dev_passes"] == 4 and all(not f["file"].startswith("/")
                                                                           for f in sr["dev"])
          and str(C.INC_DIR) not in json.dumps(sr))
    pin0 = W.sha(EM.pin_path(key))
    rec2 = safe(_d2, "D2LAION")
    check("a rerun keeps every file (no refit, no pass)", isinstance(rec2, dict)
          and [f["status"] for f in rec2["dev"]] == ["kept"] * 3 and rec2["dev_gt"][0]["status"] == "kept"
          and W.sha(EM.pin_path(key)) == pin0, rec2)
    with fake_world_models(WORLD["cache"], WORLD["reg"]):
        emb, ck = EM.load_embedder("LAION")
        e = refused(EM._fit_d2, "D2LAION", emb, ck, V._Workers(1), True)
    check("a refit is refused (fitted once, before any of the arm's dev passes)", e is not None
          and "fitted already" in str(e), e)
    npz = EM.clf_dir(key) / "classifier.npz"
    raw = npz.read_bytes()
    npz.write_bytes(raw + b"\0")
    e2 = refused(_d2, "D2LAION")
    npz.write_bytes(raw)
    check("a changed classifier file is refused at load", e2 is not None and "pin" in str(e2), e2)
    check("nothing written under twostage/e3_v1", tree(TS.root()) == before)


def w_d2_more():
    print("D2: CLIP and RN50 (transformers, timm), BC2640 (the 640-px view), the box reproduction, not run")
    before = tree(TS.root())
    r_clip = _d2("D2CLIP")
    r_rn = _d2("D2RN50")
    check("D2CLIP (OpenAI CLIP-L/14, 768-d) and D2RN50 (ResNet-50 a1, 2,048-d) end to end",
          r_clip["status"] == r_rn["status"] == "complete" and r_clip["backbone"]["dim"] == 768
          and r_rn["backbone"]["dim"] == 2048 and r_clip["backbone"]["repo"] == "openai/clip-vit-large-patch14",
          (r_clip.get("backbone"), r_rn.get("backbone")))
    calls = []
    cut0 = EM.e3_cut640

    def counted(task):
        calls.append(task[0])
        return cut0(task)
    with patched(EM, "e3_cut640", counted):
        r640 = _d2("D2BC2640")
    n_dev = len(C.read_manifest(C.manifest_path("dev")))
    check("D2BC2640 cuts every dev image from the 640-px view in each of its three passes and in its ground-truth "
          "pass (%d calls)" % len(calls), len(calls) == 4 * n_dev, len(calls))
    clf = json.loads((EM.clf_dir("BC2640") / "classifier.json").read_text())
    rows, _r = EM.training_crops(images=False)
    with fake_world_models(WORLD["cache"], WORLD["reg"]):
        _i, F640, _s = EM.embed_crops(TS._group(rows), TS._Table(rows), WORLD["bc2"], V._Workers(1),
                                      cut=EM.cut640_task)
    stored = SC.load_npz(EM.clf_dir("BC2640") / "train_emb.npz")
    e3f = SC.load_npz(TS.classifier_dir() / "train_emb.npz")
    d0 = json.loads((F.arm_dir("D2BC2640", 0) / "dev.json").read_text())
    gt = json.loads((EM.d2_dir("BC2640") / "dev_gt.json").read_text())
    check("D2BC2640: BioCLIP-2 on the training crops cut from the 640-px view (its features those of cut640_task, not "
          "E3's), a new LR fit, its dev passes in view640 mode and its ground-truth crops from the view too",
          r640["status"] == "complete" and clf["cut"] == "view640" and np.array_equal(stored["F"], F640)
          and not np.array_equal(stored["F"], e3f["F"]) and d0["settings"]["mode"] == "view640" and gt["cut"] == "view640"
          and gt["n"] > 0, (clf.get("cut"), d0.get("settings")))
    load0 = SC.load_npz

    def shifted(p):
        x = load0(p)
        if str(p).endswith("dev.stage1.npz") and "/twostage/" in str(p):
            x["q"] = x["q"] + 2e-4
        return x
    with patched(SC, "load_npz", shifted):
        e = refused(_d2, "D2DINOB")
    check("a stored stage-1 box set that differs (q 2e-4 off) refuses the arm's first pass: no arm file written",
          e is not None and "stage-1" in str(e) and not (F.root() / "arms" / "D2DINOB").exists(), e)
    dl = json.loads(EM.downloads_path().read_text())
    bad = copy.deepcopy(dl)
    bad["backbones"]["SIGLIP"].update(status="missing", why="the download failed")
    with patched(EM, "read_downloads", lambda: bad):
        e2 = refused(_d2, "D2SIGLIP")
    sr = json.loads(EM.score_record_path("D2SIGLIP").read_text())
    check("a backbone not downloaded: D2SIGLIP recorded 'not run (download)' and refused; no pass taken",
          e2 is not None and sr["status"] == "not run" and sr["reason"] == EM.NOT_RUN_DOWNLOAD
          and not (F.root() / "arms" / "D2SIGLIP").exists(), (e2, sr))
    check("nothing written under twostage/e3_v1", tree(TS.root()) == before)
    stray = TS.root() / "stray.json"
    gt0 = EM._gt_pass

    def writes_into_e3(*a, **k):
        stray.write_text("{}")
        return gt0(*a, **k)
    with patched(EM, "_gt_pass", writes_into_e3):
        e3 = refused(_d2, "D2DINOL")
    stray.unlink()
    check("the job runs inside v3fam.e_trees_unchanged: a file appearing under twostage/e3_v1 during it fails it",
          e3 is not None and "changed" in str(e3), e3)


def w_ft():
    print("D2FT: FT-ResNet-50 per seed, trained once, its softmax naming E3-B's boxes")
    before = tree(TS.root())
    t0 = time.time()
    rec = _d2("D2FT", ft_cfg={"epochs": 2, "batch": 8})
    pin = json.loads(EM.pin_path("FT").read_text())
    trs = [json.loads((EM.d2_dir("FT") / ("s%d" % s) / "train.json").read_text()) for s in (0, 1, 2)]
    docs = [json.loads((F.arm_dir("D2FT", s) / "dev.json").read_text()) for s in (0, 1, 2)]
    check("three networks trained once (seeds 0-2, final weights written once and pinned), classifier seed s with "
          "stage-1 seed s, three passes and three ground-truth records (%.0fs)" % (time.time() - t0),
          rec["status"] == "complete" and [f["status"] for f in rec["dev"]] == ["written"] * 3
          and sorted(pin["seeds"]) == ["0", "1", "2"] and len({v["weights_sha256"] for v in pin["seeds"].values()}) == 3
          and [t["seed_text"] for t in trs] == ["inc2/v3fam/d2/ft/s%d" % s for s in (0, 1, 2)]
          and all(d["d2"]["classifier_seed"] == s and d["stage2"]["classifier"]["classifier_seed"] == s
                  and d["stage2"]["classifier"]["weights_sha256"] == pin["seeds"][str(s)]["weights_sha256"]
                  for s, d in zip((0, 1, 2), docs)) and len(rec["dev_gt"]) == 3
          and all(d["created_utc"] > pin["written_utc"] for d in docs), rec.get("dev"))
    s1 = SC.load_npz(F.arm_dir("D2FT", 0) / "dev.stage1.npz")
    check("  its per-box records hold the encoding (14-d) and top-3 probabilities from the softmax",
          s1["feat"].shape[1] == C.NC + 1 and np.all(s1["top_p"][:, 0] >= s1["top_p"][:, 1]))
    rec2 = safe(_d2, "D2FT", ft_cfg={"epochs": 2, "batch": 8})
    check("a rerun trains nothing and keeps every pass", isinstance(rec2, dict)
          and [f["status"] for f in rec2["dev"]] == ["kept"] * 3 and json.loads(EM.pin_path("FT").read_text()) == pin,
          rec2)
    check("nothing written under twostage/e3_v1", tree(TS.root()) == before)


def w_d9():
    print("D9: D9a (E3's sensitivity) and two D9b refits")
    before = tree(TS.root())
    e3_trees = F.e_trees()
    t0 = time.time()
    with patched(EM, "d9_required", lambda classes: ()), fake_world_models(WORLD["cache"], WORLD["reg"]):
        rec = EM.score_d9(workers=V._Workers(1), refits=2, **KW)
    after = tree(TS.root())
    new = sorted(set(after) - set(before))
    changed = sorted(k for k in before if before[k] != after.get(k))
    sens = json.loads((TS.root() / "sensitivity.json").read_text())
    check("D9a: E3's own sensitivity.json, the only file written under twostage/e3_v1 (%.0fs)" % (time.time() - t0),
          new == ["sensitivity.json"] and not changed and sens["format"] == TS.FMT_SENSITIVITY
          and rec["d9a"]["status"] == "written" and F.e_trees() == e3_trees, (new, changed))
    doc = json.loads((EM.d9_dir() / "record.json").read_text())
    passes = doc["d9b"]["passes"]
    refits = [json.loads((EM.d9_dir() / "refits" / ("half%02d.json" % b)).read_text()) for b in (0, 1)]
    check("D9b: two refits at C = 30 stored once (their half samples by stable_int, sessions and crops), each through "
          "E3-B's seeds 0-2 (six passes, stage-1 reproduced, no per-box features kept)",
          len(passes) == 6 and all(r["C"] == 30.0 and r["half_sample"]["seed_text"] == "inc2/v3fam/d9/half/%d" % b
                                   for b, r in enumerate(refits))
          and all("feat" not in SC.load_npz(F.arm_dir("D9", s) / ("dev.half%02d.stage1.npz" % b))
                  for b in (0, 1) for s in (0, 1, 2)), [p["file"] for p in passes])
    sm = doc["d9b"]["summary"]
    e3b = [json.loads((TS.arm_dir("B", s) / "dev.json").read_text())["species_map50_95"] for s in (0, 1, 2)]
    want = EM.D9_SENTENCE["exceeds"] if sm["pooled_sd"] > np.std(e3b, ddof=1) else EM.D9_SENTENCE["within"]
    check("the record: per seed and pooled sd across refits, beside E3-B's dev seed sd, E3's D's and the SESOI; the "
          "sentence by the fixed rule (%s)" % sm["sentence"], sm["sentence"] == want and sm["sesoi"] == 0.005
          and abs(sm["e3b_dev_seed_sd"] - float(np.std(e3b, ddof=1))) < 1e-12 and set(sm["e3_diffs"]) >= {"M", "A", "B"}
          and len(sm["per_seed"]) == 3, sm)
    with patched(EM, "d9_required", lambda classes: ()), fake_world_models(WORLD["cache"], WORLD["reg"]):
        rec2 = safe(EM.score_d9, workers=V._Workers(1), refits=2, **KW)
    check("a rerun keeps D9a, the refits, the passes and the record", isinstance(rec2, dict)
          and rec2["d9a"]["status"] == "kept"
          and rec2["record"]["status"] == "kept" and all(p["status"] == "kept" for p in rec2["dev"])
          and tree(TS.root()) == after)
    with fake_world_models(WORLD["cache"], WORLD["reg"]):
        e = refused(TS.sensitivity, embedder=WORLD["bc2"], workers=V._Workers(1), **KW)
    check("sensitivity.json is written only once", e is not None and "written once" in str(e), e)


def w_cli():
    print("the CLI")
    env = dict(os.environ)
    pkg = str(pathlib.Path(EM.__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["HF_HUB_CACHE"] = str(WORLD["cache"])

    def cli(*a):
        p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_embed"] + list(a), env=env,
                           capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    rc, out, _err = cli("status")
    st = json.loads(out) if rc == 0 else {}
    check("status prints every D2 arm's backbone status from downloads.json (exit 0)", rc == 0
          and set(st) == set(EM.D2_ARMS), (rc, out[:300]))
    rc, _o, err = cli("score")
    rc2, _o2, err2 = cli("downloads")
    check("score without --arm and a second downloads record exit 2 with [inc2.v3fam_embed] ERROR",
          rc == 2 and "--arm" in err and rc2 == 2 and "written once" in err2, (err[-200:], err2[-200:]))


WORLD_TESTS = ("w_setup", "w_reference", "w_d2", "w_d2_more", "w_ft", "w_d9", "w_cli")


def main():
    """UNITS, then (unless V3FAM_EMBED_ONLY=units) WORLD_TESTS in order (each builds on the files the earlier ones
    wrote); V3FAM_EMBED_UPTO=<name> stops after it."""
    t0 = time.time()
    only = os.environ.get("V3FAM_EMBED_ONLY")
    upto = os.environ.get("V3FAM_EMBED_UPTO")
    try:
        WORLD["rows"] = W.build_world()
        with patched(SL, "_load_backbone", fake_open_clip), patched(SL, "_device", lambda: "cpu"):
            for name in UNITS + (() if only == "units" else WORLD_TESTS):
                globals()[name]()
                if name == upto:
                    break
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
