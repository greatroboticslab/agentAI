"""Per-box admission with masking (decision D-B), for the incremental Step 1
(docs/CONTINUOUS_LOOP.md §2 D-B, §3.3 stage 6).

Two pieces, both pure (no state, no network):

decide(labels, verdicts, boxes)
    The admission of one image from its box verdicts, first rule that applies:
      whole            verify.image_verdict == admitted (the v1 image rule,
                       unchanged);
      masked           not admitted, and at least one target box (INC id 0-11)
                       is VERIFIED. Kept: the VERIFIED target boxes and the
                       OtherPlant boxes judged OTHER_OK or SMALL. Masked:
                       every other box (target CONFLICT, UNKNOWN, FAILED or
                       SMALL; OtherPlant CONFLICT or FAILED, and the rare
                       OtherPlant UNKNOWN of an out-of-fold probe without
                       OtherPlant; every unmapped box, INC id 13);
      refused_overlap  a masked box has IoU >= REFUSE_IOU with a kept box:
                       one plant would carry two contradicting labels, so the
                       image goes to the human queue;
      not_admitted     everything else (no verified target box).

mask_except(image, mask_boxes, keep_boxes, out_dir, key)
    A lossless PNG of the EXIF-transposed image in which
    (masked rectangles rounded outward) minus (kept rectangles rounded outward)
    holds the image's mean RGB (rounded, taken before any fill). With no
    overlap between the two sets of rectangles the file is byte-identical to
    funnel.recover.mask (same rounding, same mean, same PNG encoder call).
    recover.mask itself is not used: it fills every masked rectangle whole,
    so it erases verified pixels where a masked box overlaps a kept one
    (funnel/recover.py:111-144). Named <key>.<sha16>.png, written once,
    atomically. Returns the record the queue row carries: path, sha256,
    masked_area_frac, and for each masked box the share of its pixels that
    stays visible inside kept boxes.

Box coordinates are normalised YOLO (cx, cy, w, h), with or without a leading
class id. Nothing here names a domain.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path

from ..inc import verify as V

REFUSE_IOU = 0.5            # contract §3.3: a masked box with IoU >= 0.5 with a kept box refuses the image
UNMAPPED_ID = 13            # intake class id of a name the class map could not resolve (never trained on)
UNMAPPED = "unmapped"       # the verdict of an unmapped box (it is never embedded)
WHOLE, MASKED, REFUSED_OVERLAP, NOT_ADMITTED = "whole", "masked", "refused_overlap", "not_admitted"
ADMISSIONS = (WHOLE, MASKED, REFUSED_OVERLAP, NOT_ADMITTED)
N_TARGET = V.OTHER           # INC ids 0..11 are the targets, 12 is OtherPlant


class MaskError(RuntimeError):
    """A mask that must not be written (unreadable image, a refused overlap)."""


def _xywh(b):
    b = tuple(float(v) for v in b)
    return b[1:5] if len(b) == 5 else b[:4]


def pixel_rect(box, W, H):
    """(x0, y0, x1, y1) of a normalised box rounded outward to whole pixels and
    clipped to the frame: funnel.recover.mask's rounding."""
    cx, cy, w, h = _xywh(box)
    x0 = max(0, int(math.floor((cx - w / 2.0) * W)))
    x1 = min(W, int(math.ceil((cx + w / 2.0) * W)))
    y0 = max(0, int(math.floor((cy - h / 2.0) * H)))
    y1 = min(H, int(math.ceil((cy + h / 2.0) * H)))
    return x0, y0, x1, y1


def iou(a, b):
    """IoU of two normalised boxes (cx, cy, w, h), with or without a class id."""
    ax, ay, aw, ah = _xywh(a)
    bx, by, bw, bh = _xywh(b)
    ix = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
    iy = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def overlaps(mask_boxes, keep_boxes, thr=REFUSE_IOU):
    """[(masked index, kept index, iou)] of every pair at IoU >= thr."""
    out = []
    for i, m in enumerate(mask_boxes):
        for j, k in enumerate(keep_boxes):
            u = iou(m, k)
            if u >= thr:
                out.append((i, j, round(u, 6)))
    return out


def is_kept(label, verdict):
    """The D-B keep rule for one box."""
    label = int(label)
    if label < N_TARGET:
        return verdict == V.VERIFIED
    if label == V.OTHER:
        return verdict in (V.OTHER_OK, V.SMALL)
    return False                                   # unmapped (13): always masked


def decide(labels, verdicts, boxes, thr=REFUSE_IOU):
    """The admission of one image (module docstring). labels: INC ids 0..13;
    verdicts: verify verdict names (UNMAPPED for id 13); boxes: normalised
    boxes in the same order. Returns {"admission", "image_verdict", "keep",
    "mask", "overlaps"} with box indices."""
    labels = [int(x) for x in labels]
    verdicts = list(verdicts)
    if not (len(labels) == len(verdicts) == len(boxes)):
        raise ValueError("labels, verdicts and boxes differ in length (%d, %d, %d)"
                         % (len(labels), len(verdicts), len(boxes)))
    bad = [x for x in labels if not 0 <= x <= UNMAPPED_ID]
    if bad:
        raise ValueError("labels outside 0..%d: %s" % (UNMAPPED_ID, bad[:3]))
    iv = V.image_verdict(labels, verdicts)
    n = len(labels)
    if iv == V.ADMITTED:
        return {"admission": WHOLE, "image_verdict": iv, "keep": list(range(n)), "mask": [], "overlaps": []}
    keep = [i for i in range(n) if is_kept(labels[i], verdicts[i])]
    verified_target = [i for i in keep if labels[i] < N_TARGET]
    if not verified_target:
        return {"admission": NOT_ADMITTED, "image_verdict": iv, "keep": [], "mask": [], "overlaps": []}
    mask = [i for i in range(n) if i not in set(keep)]
    ov = [(mask[a], keep[b], u) for a, b, u in overlaps([boxes[i] for i in mask], [boxes[i] for i in keep], thr)]
    if ov:
        return {"admission": REFUSED_OVERLAP, "image_verdict": iv, "keep": keep, "mask": mask,
                "overlaps": [list(x) for x in ov]}
    return {"admission": MASKED, "image_verdict": iv, "keep": keep, "mask": mask, "overlaps": []}


def _open_rgb(image_path):
    import numpy as np
    from PIL import Image, ImageOps
    try:
        with Image.open(image_path) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")
    except Exception as e:
        raise MaskError("cannot open %s to mask it (%s: %s)" % (image_path, type(e).__name__, e))
    return np.asarray(im).copy()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def masked_array(a, mask_boxes, keep_boxes):
    """(the filled copy of the HxWx3 uint8 array a, the filled-pixel mask, the
    kept-pixel mask, the mean RGB)."""
    import numpy as np
    H, W = a.shape[:2]
    mean = np.rint(a.reshape(-1, 3).astype(np.float64).mean(axis=0)).astype(np.uint8)
    fill = np.zeros((H, W), dtype=bool)
    kept = np.zeros((H, W), dtype=bool)
    for b in mask_boxes:
        x0, y0, x1, y1 = pixel_rect(b, W, H)
        if x1 > x0 and y1 > y0:
            fill[y0:y1, x0:x1] = True
    for b in keep_boxes:
        x0, y0, x1, y1 = pixel_rect(b, W, H)
        if x1 > x0 and y1 > y0:
            kept[y0:y1, x0:x1] = True
    region = fill & ~kept
    out = a.copy()
    out[region] = mean
    return out, region, kept, mean


def mask_except(image_path, mask_boxes, keep_boxes, out_dir, key=None, refuse_iou=REFUSE_IOU):
    """Write the masked PNG (module docstring) and return its record. Raises
    MaskError on an unreadable image, and on a masked box with IoU >=
    refuse_iou with a kept box (decide() refuses such an image first; this is
    the fail-closed second check)."""
    import numpy as np
    from PIL import Image
    mask_boxes, keep_boxes = list(mask_boxes), list(keep_boxes)
    if refuse_iou is not None:
        ov = overlaps(mask_boxes, keep_boxes, refuse_iou)
        if ov:
            raise MaskError("%s: masked box %d overlaps kept box %d at IoU %.3f >= %.2f"
                            % (image_path, ov[0][0], ov[0][1], ov[0][2], refuse_iou))
    a = _open_rgb(image_path)
    H, W = a.shape[:2]
    out, region, kept, mean = masked_array(a, mask_boxes, keep_boxes)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", optimize=False)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    name = "%s.%s.png" % (key or Path(image_path).stem, sha[:16])
    path = Path(out_dir) / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        _atomic_write(path, data)
    visible = []
    for b in mask_boxes:
        x0, y0, x1, y1 = pixel_rect(b, W, H)
        area = max(0, x1 - x0) * max(0, y1 - y0)
        visible.append(round(float(kept[y0:y1, x0:x1].sum()) / area, 6) if area else 0.0)
    return {"path": str(path), "sha256": sha, "W": int(W), "H": int(H),
            "masked_px": int(region.sum()), "masked_area_frac": round(float(region.sum()) / float(W * H), 6),
            "visible_in_kept": visible, "mean_rgb": [int(x) for x in np.asarray(mean).tolist()]}
