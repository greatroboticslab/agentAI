#!/usr/bin/env python3
"""V3F D7: the two-stage system's latency on the lab RTX 3060 (docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-11): V3F, the v3 experiment family D0-D9 (pre-registered as
one family)", V3F.5 D7; decision 13). Not a decision arm: no frame budget is
set, so no pass or fail is stated.

    python two_stage_latency.py --dev-manifest splits/v1/dev.jsonl [--path-map OLD=NEW ...] \\
        --stage1 e1_b_m640/runs/base__s0/weights/final.pt \\
        [--stage1-seeds e1_b_m640/runs/base__s1/weights/final.pt,e1_b_m640/runs/base__s2/weights/final.pt] \\
        --reference b_v2_m640/runs/base__s0/weights/final.pt \\
        --classifier twostage/e3_v1/classifier/classifier.npz \\
        --stored-stage1 twostage/e3_v1/arms/B/s0/dev.stage1.npz \\
        --hf-cache ~/v3fam_latency/hf [--out ~/v3fam_latency] [--max-wait-min 120]

A person's (the delegate's) step on the lab, never a lever, outside the family's
GPU-h cap. Standalone: run as a plain file in its own venv pinned to
Ultralytics 8.4.37 and open_clip 3.3.0; it imports nothing of the platform
(its few helpers are E3's, copied: the crop, the classifier's probabilities,
one box per location, the inverse letterbox; the platform's tests compare
them with E3's). The weights, the classifier and E3-B s0's stage-1 record come
from the P0 backup on the lab and must hash as pinned (PINNED); BioCLIP-2 is
read from the venv's own Hugging Face cache at snapshot 2957b322 (offline).

What it reads: the dev manifest's key, image and sha256 fields (never a label
path, never a label, never a score); the dev images' pixels, each checked
against its manifest sha256. What it writes: <out>/v3fam_lab3060.json (once)
and v3fam_lab3060.md, nothing else. It never stops another process: it waits
until nvidia-smi lists no compute process on the GPU (recorded before and
after) and gives up, writing nothing, after --max-wait-min.

Protocol (fixed by V3F.5 D7):
  inputs   550 dev images ranked by stable_int("inc2/v3fam/d7/<key>") (ties by key): the first 50 warm up,
           the next 500 are timed;
  timing   batch 1, FP16, torch.cuda.synchronize() around each stage, stages timed separately:
           decode (OpenCV) and preprocess (letterbox to 640, the tensor on the GPU); stage 1 (E1-B s0,
           YOLO11m at 640: forward, NMS at conf 0.25 = the deployment threshold q >= 0.25, IoU 0.7,
           max_det 300, one box per location, boxes to original pixels); crops of the boxes kept (E3's:
           square, 10 % margin, grey padding, 224 px bicubic); BioCLIP-2 ViT-L/14 on the image's crops as
           one batch (fp16 autocast, L2-normalised); the LR (E3's classifier.npz);
  reference YOLO11m single-stage (b_v2_m640 s0) at 640 with conf 0.25: decode and preprocess, forward and NMS;
  beside   tau in {0.1, 0.5} for stage 1; D3's 3 x stage 1 (E1-B s0, s1, s2) and the WBF of their boxes; the
           per-crop cost of each D2 backbone the venv's cache holds (FT-ResNet-50 costs as ResNet-50,
           BioCLIP-2@640 as BioCLIP-2);
  reported median, p90 and p99 ms per image of every stage and of the total, boxes per image, per-crop ms;
  software the stage-1 boxes of 20 dev images (the first 20 timed) at conf 0.001 through Ultralytics' own
           validator path (a label-free view of the 617 dev images, batch 32, rect, FP16, IoU 0.7,
           max_det 300; one box per location; the inverse letterbox) against E3-B s0's stored
           dev.stage1.npz within 1 px: otherwise the run is labelled "different software".
The Jetson repeats the protocol later (TensorRT FP16 there; PyTorch here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ------------------------------------------------------------------ the protocol (pre-registered, V3F.5 D7)
SEED_TEXT = "inc2/v3fam/d7/%s"
WARMUP = 50
TIMED = 500
IMGSZ = 640
STRIDE = 32
CONF = 0.25                                  # the deployment threshold, q >= 0.25
TAUS = (0.1, 0.5)                            # beside
NMS_IOU = 0.7
MAX_DET = 300
CHECK_IMAGES = 20
CHECK_CONF = 0.001
CHECK_BATCH = 32                             # the locked scorer's batch: the rect letterbox of each image
CHECK_TOL_PX = 1.0
CROP_PX = 224                                # E3's crop (semisup_labeler: CROP_PX, MARGIN, the grey)
MARGIN = 0.10
GREY = (124, 124, 124)
BIOCLIP = "hf-hub:imageomics/bioclip-2"
BIOCLIP_REPO = "imageomics/bioclip-2"
BIOCLIP_SNAPSHOT = "2957b322090f9cb17ae72c71981c7218a28d81e0"
ULTRALYTICS = "8.4.37"
OPEN_CLIP = "3.3.0"
NC = 13
PINNED = {                                   # sha256, from the E-series records (P0 backup on the lab)
    "stage1": "d84aa23553c381cc6caa40cfe1c806a2876844dd53c9f78efa1ed800b0befc60",       # E1-B s0 final.pt
    "stage1_s1": "048d7ce9e713583c4c9d934947b963bc9f2c928cd499ff71ab04baaf664e4bdc",    # E1-B s1
    "stage1_s2": "3d5850adbb73e2588f011bdf9699c5855b0e84cdefa0782892bb3a5ae412a4db",    # E1-B s2
    "reference": "1d9027f6ca41211369f7d48d045ce86bfe33fb7dfb497b68500d2230f0712708",    # b_v2_m640 s0
    "classifier": "4905e4a67c07597146fcea62785666e65cc5ce3a3e8e35b62282bfd5d1c5f42b"}   # E3's classifier.npz
D2_BACKBONES = (                             # (name, loader, id): D2's frozen backbones (V3F.5 D2)
    ("LAION-2B CLIP ViT-L/14", "open_clip", "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"),
    ("OpenAI CLIP ViT-L/14", "hf_clip", "openai/clip-vit-large-patch14"),
    ("DINOv2-L", "hf_cls", "facebook/dinov2-large"),
    ("DINOv2-B", "hf_cls", "facebook/dinov2-base"),
    ("SigLIP2 so400m/14 224", "hf_siglip", "google/siglip2-so400m-patch14-224"),
    ("ResNet-50 a1", "timm", "hf-hub:timm/resnet50.a1_in1k"),
    ("BioCLIP-1", "open_clip", "hf-hub:imageomics/bioclip"))
WBF = {"iou_thr": 0.55, "skip_box_thr": 0.001, "conf_type": "avg", "weights": [1, 1, 1]}    # D3's setting
OUT_JSON = "v3fam_lab3060.json"
OUT_MD = "v3fam_lab3060.md"
FORMAT = "inc2-v3fam-d7-latency/1"
DIFFERENT = "different software"


class LatencyRefused(RuntimeError):
    """The run is not the pre-declared protocol (or the GPU never became free)."""


def log(msg):
    print("[v3fam.latency] %s" % msg, flush=True)


# ------------------------------------------------------------------ E3's helpers, copied (the tests compare them)
def stable_int(text, mod=2 ** 31 - 1):
    """inc.common.stable_int."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % mod


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def square_box(cx, cy, w, h, W, H):
    """semisup_labeler._square_box: the crop's square, normalised box -> (x0, y0, side) in pixels."""
    side = max(w * W, h * H) * (1 + 2 * MARGIN)
    return cx * W - side / 2, cy * H - side / 2, side


def cut(im, cx, cy, w, h):
    """semisup_labeler._cut: the square crop padded with neutral grey off-frame, 224 px bicubic."""
    from PIL import Image
    W, H = im.size
    x0, y0, side = square_box(cx, cy, w, h, W, H)
    s = int(round(side))
    canvas = Image.new("RGB", (s, s), GREY)
    ix0, iy0 = int(round(x0)), int(round(y0))
    box = (max(0, ix0), max(0, iy0), min(W, ix0 + s), min(H, iy0 + s))
    if box[2] > box[0] and box[3] > box[1]:
        canvas.paste(im.crop(box), (box[0] - ix0, box[1] - iy0))
    return canvas.resize((CROP_PX, CROP_PX), Image.BICUBIC)


def degenerate(cx, cy, w, h, W, H):
    """twostage.degenerate: the square crop's side rounds below 1 px."""
    return int(round(square_box(cx, cy, w, h, W, H)[2])) < 1


def normalise(xyxy, W, H):
    """twostage.normalise: original-pixel xyxy -> (cx, cy, w, h) in [0, 1]."""
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(a)
    out[:, 0] = (a[:, 0] + a[:, 2]) / 2.0 / W
    out[:, 1] = (a[:, 1] + a[:, 3]) / 2.0 / H
    out[:, 2] = (a[:, 2] - a[:, 0]) / W
    out[:, 3] = (a[:, 3] - a[:, 1]) / H
    return out


def crops_for(im, xyxy):
    """E3's crops of one image's boxes (original pixels; a degenerate box is skipped, as E3 skips it)."""
    W, H = im.size
    out = []
    for cx, cy, w, h in normalise(xyxy, W, H):
        if not degenerate(cx, cy, w, h, W, H):
            out.append(cut(im, cx, cy, w, h))
    return out


def proba(X, W, b, classes):
    """twostage.proba: softmax(X W' + b) over the fitted classes as [n, 13]."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    Z = X @ np.asarray(W, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
    Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    Pk = E / E.sum(axis=1, keepdims=True)
    P = np.zeros((len(X), NC), dtype=np.float64)
    P[:, np.asarray(classes, dtype=np.int64)] = Pk
    return P


def l2(X):
    """inc.verify._norm."""
    import numpy as np
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def one_per_box(xyxy, conf):
    """inc.scorer.one_per_box: indices (ascending) left when rows at identical coordinates keep their most
    confident one."""
    import numpy as np
    b = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    c = np.asarray(conf, dtype=float).reshape(-1)
    if len(c) < 2:
        return np.arange(len(c))
    order = np.argsort(-c, kind="stable")
    first = np.unique(b[order], axis=0, return_index=True)[1]
    return np.sort(order[first])


def to_original(xyxy, ratio_pad, ori_shape):
    """twostage.to_original: the validator's letterbox inverted per axis, clipped to the image."""
    import numpy as np
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    (gh, gw), (px, py) = ratio_pad
    gh, gw, px, py = float(gh), float(gw), float(px), float(py)
    h0, w0 = float(ori_shape[0]), float(ori_shape[1])
    out = np.empty_like(a)
    out[:, [0, 2]] = np.clip((a[:, [0, 2]] - px) / gw, 0.0, w0)
    out[:, [1, 3]] = np.clip((a[:, [1, 3]] - py) / gh, 0.0, h0)
    return out


def summary(ms):
    """{n, median, p90, p99, mean} of per-image milliseconds (numpy's linear percentiles)."""
    import numpy as np
    a = np.asarray(ms, dtype=np.float64)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "median": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


# ------------------------------------------------------------------ inputs (dev pixels only)
def map_path(p, maps):
    for old, new in maps:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def dev_rows(manifest, maps=()):
    """[{key, image, sha256}] of the dev manifest, sorted by key: the key, image and sha256 fields only (a
    row's label path is dropped unread)."""
    rows = []
    with open(manifest) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            rows.append({"key": str(r["key"]), "image": map_path(str(r["image"]), maps), "sha256": r.get("sha256")})
    if len({r["key"] for r in rows}) != len(rows):
        raise LatencyRefused("the dev manifest holds duplicate keys")
    return sorted(rows, key=lambda r: r["key"])


def ranked(rows):
    """(warm-up, timed): the first WARMUP and the next TIMED rows by stable_int('inc2/v3fam/d7/<key>'), key."""
    order = sorted(rows, key=lambda r: (stable_int(SEED_TEXT % r["key"]), r["key"]))
    if len(order) < WARMUP + TIMED:
        raise LatencyRefused("the dev manifest lists %d images; the protocol takes %d" % (len(order), WARMUP + TIMED))
    return order[:WARMUP], order[WARMUP:WARMUP + TIMED]


def check_images(rows):
    """Each image's bytes against its manifest sha256 (the pixels the exam holds)."""
    bad = [r["key"] for r in rows if not r.get("sha256") or sha256_file(r["image"]) != r["sha256"]]
    if bad:
        raise LatencyRefused("%d image(s) do not hash as the dev manifest records (e.g. %s): another file, or the "
                             "path map is wrong" % (len(bad), bad[:3]))
    return len(rows)


def check_pinned(files):
    """{name: sha256} of the given files, each equal to PINNED."""
    out = {}
    for name, path in files.items():
        if path is None:
            continue
        got = sha256_file(path)
        if got != PINNED[name]:
            raise LatencyRefused("%s (%s) hashes to %s, not the pinned %s" % (name, path, got[:12], PINNED[name][:12]))
        out[name] = got
    return out


def bioclip_snapshot(hf_cache):
    """The snapshot refs/main names in the venv's cache for BioCLIP-2; refused unless it is the pinned one."""
    ref = Path(hf_cache) / ("models--%s" % BIOCLIP_REPO.replace("/", "--")) / "refs" / "main"
    got = ref.read_text().strip() if ref.is_file() else None
    if got != BIOCLIP_SNAPSHOT:
        raise LatencyRefused("BioCLIP-2's cache %s names snapshot %r, not %s" % (ref, got, BIOCLIP_SNAPSHOT))
    return got


# ------------------------------------------------------------------ the GPU (wait, never stop anything)
def gpu_apps(run=subprocess.run):
    """[(pid, name, MiB)] of the compute processes nvidia-smi lists."""
    p = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
    if p.returncode != 0:
        raise LatencyRefused("nvidia-smi failed: %s" % (p.stderr or "").strip()[:200])
    out = []
    for ln in (p.stdout or "").splitlines():
        t = [x.strip() for x in ln.split(",")]
        if len(t) >= 3 and t[0]:
            out.append((int(t[0]), t[1], t[2]))
    return out


def gpu_state(run=subprocess.run):
    p = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,"
             "clocks.sm,power.draw", "--format=csv,noheader"], capture_output=True, text=True)
    return {"gpu": (p.stdout or "").strip(), "apps": [list(a) for a in gpu_apps(run)], "utc": _utc()}


def wait_gpu_free(max_wait_s, poll_s=60.0, run=subprocess.run, sleep=time.sleep, own_pid=None):
    """Wait until no compute process but this one holds the GPU; never stop one. Returns the seconds waited."""
    waited = 0.0
    while True:
        others = [a for a in gpu_apps(run) if a[0] != (own_pid or os.getpid())]
        if not others:
            return waited
        if waited >= max_wait_s:
            raise LatencyRefused("the GPU stayed busy for %.0f min (%s): nothing timed, nothing written"
                                 % (waited / 60.0, others))
        log("the GPU is busy (%s): waiting" % others)
        sleep(poll_s)
        waited += poll_s


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ models
class Stage1(object):
    """An Ultralytics detector at batch 1, FP16, as the predictor runs it: letterbox to 640 (minimal rectangle,
    stride 32), forward, NMS, boxes to original pixels, one box per location."""

    def __init__(self, path, device):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        self.torch = torch
        self.device = torch.device(device)
        self.net = AutoBackend(str(path), device=self.device, fp16=True, fuse=True)
        self.net.eval()
        self.net.warmup(imgsz=(1, 3, IMGSZ, IMGSZ))

    def preprocess(self, bgr):
        import numpy as np
        from ultralytics.data.augment import LetterBox
        lb = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=True, stride=STRIDE)
        im = lb(image=bgr)
        im = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))
        t = self.torch.from_numpy(im).to(self.device).half() / 255.0
        return t[None]

    def detect(self, t, shape0, conf):
        import numpy as np
        try:
            from ultralytics.utils import nms as N
            nmsf = N.non_max_suppression
        except ImportError:                  # older layouts keep it in ops
            from ultralytics.utils import ops as N
            nmsf = N.non_max_suppression
        from ultralytics.utils import ops
        with self.torch.no_grad():
            preds = self.net(t)
        out = nmsf(preds, conf, NMS_IOU, max_det=MAX_DET)[0]
        boxes = ops.scale_boxes(t.shape[2:], out[:, :4].clone(), shape0).float().cpu().numpy()
        q = out[:, 4].float().cpu().numpy()
        keep = one_per_box(boxes, q)
        return np.asarray(boxes[keep], dtype=np.float64), np.asarray(q[keep], dtype=np.float64)


def sync():
    import torch
    torch.cuda.synchronize()


def load_bioclip(device):
    import open_clip
    model, _train, pre = open_clip.create_model_and_transforms(BIOCLIP)
    return model.to(device).eval(), pre


def embedder_for(kind, ident, device):
    """f(pils) -> features on device for a D2 backbone (fp16 autocast), offline from the venv's cache."""
    import torch
    if kind == "open_clip":
        import open_clip
        model, _t, pre = open_clip.create_model_and_transforms(ident)
        model = model.to(device).eval()

        def f(pils):
            t = torch.stack([pre(p) for p in pils]).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return model.encode_image(t)
        return f
    if kind == "timm":
        import timm
        model = timm.create_model(ident, pretrained=True, num_classes=0).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        tf = timm.data.create_transform(**cfg)

        def f(pils):
            t = torch.stack([tf(p) for p in pils]).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return model(t)
        return f
    from transformers import AutoImageProcessor, AutoModel
    model = AutoModel.from_pretrained(ident).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(ident)

    def f(pils):
        px = proc(images=pils, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if kind in ("hf_clip", "hf_siglip"):
                return model.get_image_features(pixel_values=px)
            return model(pixel_values=px).last_hidden_state[:, 0]
    return f


# ------------------------------------------------------------------ the software check (V3F.5 D7)
def stage1_validator_boxes(weights, rows, device, keys):
    """{key: (xyxy original px, q)} of the stage-1 box sets Ultralytics' own validator path gives on a label-free
    view of the dev images (batch 32, rect, FP16, conf 0.001, IoU 0.7, max_det 300; one box per location; the
    inverse letterbox), for `keys`."""
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    got = {}

    class Capture(DetectionValidator):
        def update_metrics(self, preds, batch):
            for si, p in enumerate(preds):
                pb = self._prepare_batch(si, batch)
                key = Path(pb["im_file"]).stem
                if key in keys:
                    b = p["bboxes"].float().cpu().numpy()
                    q = p["conf"].float().cpu().numpy()
                    k = one_per_box(b, q)
                    got[key] = (to_original(b[k], pb["ratio_pad"], pb["ori_shape"]), np.asarray(q[k], np.float64))
            return super().update_metrics(preds, batch)

    with tempfile.TemporaryDirectory(prefix="v3fam_d7_view_") as tmp:
        view = Path(tmp) / "view"
        (view / "images").mkdir(parents=True)
        (view / "labels").mkdir()
        for r in rows:
            ext = os.path.splitext(r["image"])[1].lower() or ".jpg"
            os.symlink(os.path.realpath(r["image"]), str(view / "images" / (r["key"] + ext)))
        names = "".join("  - c%d\n" % i for i in range(NC))
        (view / "data.yaml").write_text("path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n%s"
                                        % (view, NC, names))
        YOLO(str(weights)).val(validator=Capture, data=str(view / "data.yaml"), imgsz=IMGSZ, batch=CHECK_BATCH,
                               conf=CHECK_CONF, iou=NMS_IOU, max_det=MAX_DET,
                               half=not str(device).startswith("cpu"), device=device,
                               plots=False, save_json=False, verbose=False, project=str(Path(tmp) / "runs"),
                               name="val", exist_ok=True)
    return got


def compare_stage1(got, stored, keys):
    """Each key's box set against the stored dev.stage1.npz rows: the same count and, sorted by (-q, x1, y1),
    every coordinate within CHECK_TOL_PX. Returns {matched, per_key, max_diff_px}."""
    import numpy as np
    skeys = [str(k) for k in stored["keys"]]
    img = np.asarray(stored["img"], dtype=np.int64)
    per, worst, ok = {}, 0.0, True
    for k in keys:
        if k not in skeys or k not in got:
            per[k] = {"ok": False, "why": "missing"}
            ok = False
            continue
        sel = img == skeys.index(k)
        sx, sq = np.asarray(stored["xyxy"], np.float64)[sel], np.asarray(stored["q"], np.float64)[sel]
        gx, gq = got[k]
        if len(sx) != len(gx):
            per[k] = {"ok": False, "stored": int(len(sx)), "now": int(len(gx))}
            ok = False
            continue
        so = np.lexsort((sx[:, 1], sx[:, 0], -sq))
        go = np.lexsort((gx[:, 1], gx[:, 0], -gq))
        d = float(np.abs(sx[so] - gx[go]).max()) if len(sx) else 0.0
        worst = max(worst, d)
        per[k] = {"ok": bool(d <= CHECK_TOL_PX), "boxes": int(len(sx)), "max_diff_px": d}
        ok = ok and d <= CHECK_TOL_PX
    return {"matched": bool(ok), "per_key": per, "max_diff_px": worst, "tolerance_px": CHECK_TOL_PX}


# ------------------------------------------------------------------ the timing
def _clock(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def _decode(det, path):
    """(the BGR frame, the detector's input tensor): OpenCV's decode and the letterbox."""
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise LatencyRefused("OpenCV read nothing from %s" % path)
    return bgr, det.preprocess(bgr)


def time_two_stage(rows, s1, clip, pre, clf, device, conf, record=True):
    """Per image: decode+preprocess, stage 1, crops, embedding, LR (ms) and the boxes kept."""
    import numpy as np
    import torch
    from PIL import Image
    W, b, classes = clf
    out = {k: [] for k in ("decode_preprocess", "stage1", "crops", "embed", "lr", "total", "boxes")}
    for r in rows:
        (bgr, t), m_dec = _clock(lambda: _decode(s1, r["image"]))
        (xy, q), m_s1 = _clock(lambda: s1.detect(t, bgr.shape[:2], conf))
        if len(q):
            pils, m_cut = _clock(lambda: crops_for(Image.fromarray(np.ascontiguousarray(bgr[..., ::-1])), xy))

            def emb():
                x = torch.stack([pre(p) for p in pils]).to(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    return l2(clip.encode_image(x).float().cpu().numpy())
            X, m_emb = _clock(emb) if pils else (None, 0.0)
            _P, m_lr = _clock(lambda: proba(X, W, b, classes)) if pils else (None, 0.0)
        else:
            m_cut = m_emb = m_lr = 0.0
        if record:
            for k, v in (("decode_preprocess", m_dec), ("stage1", m_s1), ("crops", m_cut), ("embed", m_emb),
                         ("lr", m_lr), ("total", m_dec + m_s1 + m_cut + m_emb + m_lr), ("boxes", len(q))):
                out[k].append(v)
    return out


def time_single(rows, det, conf):
    """The single-stage reference per image: decode+preprocess, forward+NMS (ms)."""
    out = {"decode_preprocess": [], "detect": [], "total": [], "boxes": []}
    for r in rows:
        (bgr, t), m_dec = _clock(lambda: _decode(det, r["image"]))
        (_xy, q), m_det = _clock(lambda: det.detect(t, bgr.shape[:2], conf))
        out["decode_preprocess"].append(m_dec)
        out["detect"].append(m_det)
        out["total"].append(m_dec + m_det)
        out["boxes"].append(len(q))
    return out


def time_d3(rows, dets, conf):
    """D3's 3 x stage 1 per image (three detectors on one preprocessed frame) and the WBF of their boxes."""
    import numpy as np
    try:
        from ensemble_boxes import weighted_boxes_fusion as wbf
    except ImportError:
        wbf = None
    out = {"stage1_x3": [], "wbf": [], "boxes": []}
    for r in rows:
        bgr, t = _decode(dets[0], r["image"])
        res, m = _clock(lambda: [d.detect(t, bgr.shape[:2], conf) for d in dets])
        out["stage1_x3"].append(m)
        if wbf is not None:
            h, w = bgr.shape[:2]
            s = np.asarray([w, h, w, h], dtype=np.float64)
            t0 = time.perf_counter()
            fb, _fs, _fl = wbf([xy / s for xy, _q in res], [q for _xy, q in res],
                               [np.zeros(len(q)) for _xy, q in res], **WBF)
            out["wbf"].append((time.perf_counter() - t0) * 1000.0)
            out["boxes"].append(len(fb))
    return out


def time_backbone(rows, crops, f):
    """Per image, the backbone on the image's crops as one batch (ms), and the crop count."""
    ms, n = [], []
    for r in rows:
        pils = crops.get(r["key"]) or []
        if not pils:
            continue
        _o, m = _clock(lambda: f(pils))
        ms.append(m)
        n.append(len(pils))
    per_crop = [m / c for m, c in zip(ms, n)]
    return {"per_image": summary(ms), "per_crop": summary(per_crop), "crops": int(sum(n)),
            "per_crop_ms_pooled": (sum(ms) / float(sum(n))) if n else None}


def stats(d):
    out = {k: summary(v) for k, v in d.items() if k != "boxes"}
    if "boxes" in d:
        out["boxes_per_image"] = summary(d["boxes"])
    return out


def table(doc):
    """The markdown table of the record's medians, p90s and p99s."""
    lines = ["# D7: two-stage latency on the lab RTX 3060 (%s)" % doc["created_utc"], "",
             "Software: %s. Batch 1, FP16, %d timed dev images after %d warm-up." % (
                 doc["software_label"], TIMED, WARMUP), "",
             "| Pipeline | Stage | median ms | p90 ms | p99 ms |", "|---|---|---:|---:|---:|"]
    for name, block in doc["timings"].items():
        if not isinstance(block, dict):
            continue
        for stage, s in block.items():
            if isinstance(s, dict) and s.get("n"):
                lines.append("| %s | %s | %.2f | %.2f | %.2f |" % (name, stage, s["median"], s["p90"], s["p99"]))
    lines += ["", "| D2 backbone | per-crop ms (median) | per-image ms (median) |", "|---|---:|---:|"]
    for name, r in doc.get("backbones", {}).items():
        if r.get("per_crop", {}).get("n"):
            lines.append("| %s | %.3f | %.2f |" % (name, r["per_crop"]["median"], r["per_image"]["median"]))
        else:
            lines.append("| %s | %s | |" % (name, r.get("status", "not available")))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ the run
def main(argv=None):
    ap = argparse.ArgumentParser(description="V3F D7: two-stage latency on the lab RTX 3060 (pre-declared protocol).")
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--path-map", action="append", default=[], help="OLD=NEW prefix map for the manifest's paths")
    ap.add_argument("--stage1", required=True, help="E1-B s0 final.pt")
    ap.add_argument("--stage1-seeds", default=None, help="E1-B s1 and s2 final.pt, comma-separated (D3's 3 x)")
    ap.add_argument("--reference", required=True, help="b_v2_m640 s0 final.pt")
    ap.add_argument("--classifier", required=True, help="E3's classifier.npz")
    ap.add_argument("--stored-stage1", required=True, help="E3-B s0 dev.stage1.npz")
    ap.add_argument("--hf-cache", required=True, help="this venv's own Hugging Face hub cache")
    ap.add_argument("--out", default=str(Path.home() / "v3fam_latency"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-wait-min", type=float, default=120.0)
    a = ap.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(Path(a.hf_cache).expanduser())
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        return run(a)
    except LatencyRefused as e:
        print("[v3fam.latency] ERROR: %s" % e, file=sys.stderr)
        return 2


def run(a):
    import numpy as np
    import open_clip
    import torch
    import ultralytics
    out = Path(a.out).expanduser()
    if (out / OUT_JSON).exists():
        raise LatencyRefused("%s exists: the lab run is recorded once" % (out / OUT_JSON))
    if ultralytics.__version__ != ULTRALYTICS or open_clip.__version__ != OPEN_CLIP:
        raise LatencyRefused("ultralytics %s and open_clip %s: the timing venv pins %s and %s"
                             % (ultralytics.__version__, open_clip.__version__, ULTRALYTICS, OPEN_CLIP))
    if not torch.cuda.is_available():
        raise LatencyRefused("no CUDA device: the protocol times FP16 on the GPU")
    maps = [tuple(m.split("=", 1)) for m in a.path_map]
    seeds = [s for s in (a.stage1_seeds or "").split(",") if s]
    if seeds and len(seeds) != 2:
        raise LatencyRefused("--stage1-seeds takes E1-B s1 and s2")
    shas = check_pinned({"stage1": a.stage1, "reference": a.reference, "classifier": a.classifier,
                         "stage1_s1": seeds[0] if seeds else None, "stage1_s2": seeds[1] if seeds else None})
    snap = bioclip_snapshot(os.environ["HF_HUB_CACHE"])
    rows = dev_rows(a.dev_manifest, maps)
    warm, timed = ranked(rows)
    check_images(rows)
    before = gpu_state()
    waited = wait_gpu_free(a.max_wait_min * 60.0)
    dev = torch.device(a.device)
    with np.load(a.classifier, allow_pickle=False) as z:
        clf = (z["W"], z["b"], z["classes"])
    with np.load(a.stored_stage1, allow_pickle=False) as z:
        stored = {k: z[k] for k in ("keys", "img", "xyxy", "q")}
    check_keys = [r["key"] for r in timed[:CHECK_IMAGES]]
    got = stage1_validator_boxes(a.stage1, rows, a.device, set(check_keys))
    sw = compare_stage1(got, stored, check_keys)
    label = "pinned (Ultralytics %s, open_clip %s; stage-1 boxes reproduce E3-B s0's within %g px)" % (
        ULTRALYTICS, OPEN_CLIP, CHECK_TOL_PX) if sw["matched"] else DIFFERENT
    s1 = Stage1(a.stage1, dev)
    clip, pre = load_bioclip(dev)
    timings = {}
    time_two_stage(warm, s1, clip, pre, clf, dev, CONF, record=False)
    timings["two-stage q>=0.25"] = stats(time_two_stage(timed, s1, clip, pre, clf, dev, CONF))
    for tau in TAUS:
        timings["two-stage q>=%g" % tau] = stats(time_two_stage(timed, s1, clip, pre, clf, dev, tau))
    ref = Stage1(a.reference, dev)
    time_single(warm, ref, CONF)
    timings["single-stage YOLO11m q>=0.25"] = stats(time_single(timed, ref, CONF))
    del ref
    if seeds:
        dets = [s1, Stage1(seeds[0], dev), Stage1(seeds[1], dev)]
        time_d3(warm, dets, CONF)
        timings["D3 3 x stage 1 q>=0.25"] = stats(time_d3(timed, dets, CONF))
        del dets
    else:
        timings["D3 3 x stage 1 q>=0.25"] = "not run (--stage1-seeds not given)"
    # per-crop costs: the crops of stage 1 at q >= 0.25, as the pipeline cuts them
    from PIL import Image
    crops = {}
    for r in timed:
        bgr, t = _decode(s1, r["image"])
        xy, _q = s1.detect(t, bgr.shape[:2], CONF)
        crops[r["key"]] = crops_for(Image.fromarray(np.ascontiguousarray(bgr[..., ::-1])), xy) if len(xy) else []

    def bio(pils):
        x = torch.stack([pre(p) for p in pils]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return clip.encode_image(x)
    backbones = {"BioCLIP-2 ViT-L/14 (reference)": dict(time_backbone(timed, crops, bio), status="timed")}
    for name, kind, ident in D2_BACKBONES:
        try:
            f = embedder_for(kind, ident, dev)
            time_backbone(warm, crops, f)
            backbones[name] = dict(time_backbone(timed, crops, f), status="timed", id=ident)
            del f
            torch.cuda.empty_cache()
        except Exception as e:               # noqa: BLE001 - a backbone the venv lacks is recorded, not fatal
            backbones[name] = {"status": "not available", "id": ident, "error": "%s: %s" % (type(e).__name__,
                                                                                          str(e)[:300])}
    after = gpu_state()
    doc = {"format": FORMAT, "protocol": "V3F.5 D7", "hardware": "lab RTX 3060 12 GB", "software_label": label,
           "software_check": sw,
           "versions": {"ultralytics": ultralytics.__version__, "open_clip": open_clip.__version__,
                        "torch": torch.__version__, "python": sys.version.split()[0]},
           "inputs": {"warmup": WARMUP, "timed": TIMED, "seed_text": SEED_TEXT % "<key>",
                      "timed_keys_sha256": hashlib.sha256("\n".join(r["key"] for r in timed).encode()).hexdigest(),
                      "dev_images": len(rows)},
           "pinned": shas, "bioclip_snapshot": snap, "settings": {"imgsz": IMGSZ, "conf": CONF, "taus": list(TAUS),
                                                                  "nms_iou": NMS_IOU, "max_det": MAX_DET,
                                                                  "fp16": True, "batch": 1},
           "gpu_before": before, "gpu_after": after, "waited_s": waited, "timings": timings, "backbones": backbones,
           "note": "FT-ResNet-50 costs as ResNet-50 a1; BioCLIP-2@640 as BioCLIP-2; no frame budget, no pass or fail",
           "created_utc": _utc()}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (".%s.%d.tmp" % (OUT_JSON, os.getpid()))
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    try:
        os.link(str(tmp), str(out / OUT_JSON))
    finally:
        tmp.unlink()
    (out / OUT_MD).write_text(table(doc))
    log("written %s (%s)" % (out / OUT_JSON, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
