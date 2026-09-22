"""Weed detection service — the campaign's model, applied to live robot frames (S6).

Closes the loop the platform was built for: data collected by the robots and the
harvest agent trains a model, and that model comes back to annotate what the robots
are seeing right now.

  GET /api/detect/model                 what is loaded, and what it is known to be bad at
  GET /api/detect/frame/{sid}.jpg       the newest frame from a live robot session,
                                        with boxes drawn — a drop-in replacement for
                                        /api/robot/frame_latest/{sid}.jpg
  POST /api/detect                      raw JPEG body -> JSON detections (no image back)

Deliberate properties:
  * the model is loaded **lazily and once**; a dashboard restart must not pay for it
    and a machine without the weights must not fail to boot.
  * inference never blocks the event loop — it runs in a threadpool.
  * per-species reliability from the model card travels **with the predictions**: the
    three weak species are flagged in every response, because a detection of
    Carpetweed (0.7324 mAP50-95) does not mean what a detection of Sicklepod (0.9767)
    means, and a laser-weeding system downstream should be able to tell.
  * detections name the species. The served checkpoint was trained with this
    project's legacy cwd12 labels (model.names 'Nutsedge' is ragweed, 'Ragweed' is
    sicklepod); they are translated once at load, in memory, and the .pt is left
    as it is.
  * the frame endpoint degrades to the plain frame if inference fails, so the live
    view keeps working when detection does not.
"""
import io
import json
import os
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .cwd12_species import (CWD12_SPECIES, is_legacy_label_list, legacy_to_species,
                            species_names_for)

router = APIRouter()
_CTX = {}
_LOCK = threading.Lock()
_MODEL = {"obj": None, "err": None, "loaded_at": None, "device": None,
          "names": None, "legacy_names": None, "card": False}

WEIGHTS = os.path.expanduser(
    os.environ.get("WEED_MODEL", "~/models/cwd12_yolo11n_s102.pt"))
CONF = float(os.environ.get("WEED_CONF", "0.25"))
IMGSZ = int(os.environ.get("WEED_IMGSZ", "640"))

# From docs/BEST_MODEL_CARD.md — independent re-evaluation, job 44454237, n=3 seeds.
# v3.60.0: the card measured each class id and named it with the legacy cwd12
# label; the numbers stay, the keys are translated to the species of that id.
_CARD_PER_LEGACY_LABEL = {
    "Ragweed": 0.9767, "Purslane": 0.9276, "PalmerAmaranth": 0.9163,
    "Crabgrass": 0.9157, "Carpetweeds": 0.9151, "PricklySida": 0.9118,
    "SpottedSpurge": 0.8818, "Nutsedge": 0.8585, "Sicklepod": 0.8555,
    "Eclipta": 0.8219, "Goosegrass": 0.7973, "Morningglory": 0.7324,
}
PER_SPECIES = {legacy_to_species(k): v for k, v in _CARD_PER_LEGACY_LABEL.items()}
WEAK = [k for k, v in PER_SPECIES.items() if v < 0.83]      # the card's three
# v3.60.0: the card measured one checkpoint. Its figures are attached only when
# that checkpoint is served and its head names the twelve cwd12 species by id;
# any other weights get null reliability rather than borrowed numbers.
CARD_CHECKPOINT = "cwd12_yolo11n_s102.pt"
MODEL_META = {
    "name": "cwd12 YOLO11n (COCO-pretrained, seed 102)",
    "holdout_map50_95": 0.8759, "holdout_std": 0.0030, "n_seeds": 3,
    "trained_on": "CottonWeedDet12 train split (3,671 images, 12 species)",
    "card": "docs/BEST_MODEL_CARD.md",
    "known_weak_species": WEAK,
    "domain_gap_warning": ("trained on close-range handheld field photography. "
                           "Measured 2026-08-26: zero-shot transfer to a clean "
                           "greenhouse-seedling weed dataset collapses (class-agnostic "
                           "mAP50-95 0.873 in-domain -> 0.100) -- treat any new "
                           "camera/domain as unmeasured until evaluated in it; "
                           "robot-frame recall is still pending an outdoor run"),
}


def _card_applies(names):
    """True when the model card's per-species figures describe the loaded model."""
    return (os.path.basename(WEIGHTS) == CARD_CHECKPOINT
            and dict(names) == dict(enumerate(CWD12_SPECIES)))


def _load():
    """Load once, remember the failure if it fails (never retry-storm)."""
    if _MODEL["obj"] is not None or _MODEL["err"]:
        return _MODEL
    with _LOCK:
        if _MODEL["obj"] is not None or _MODEL["err"]:
            return _MODEL
        try:
            if not os.path.isfile(WEIGHTS):
                raise FileNotFoundError(WEIGHTS)
            from ultralytics import YOLO
            import torch
            m = YOLO(WEIGHTS)
            # v3.60.0: legacy-labelled checkpoints name species through the
            # whole-list translation; set on the model before the first predict
            # so res.plot() draws the species too. Species lists pass through.
            raw = dict(m.names) if isinstance(m.names, dict) else dict(enumerate(m.names))
            names = species_names_for(raw)
            if names != raw:
                m.model.names = names
            dev = 0 if torch.cuda.is_available() else "cpu"
            m.predict(imgsz=IMGSZ, device=dev, verbose=False,
                      source=__import__("numpy").zeros((IMGSZ, IMGSZ, 3), dtype="uint8"))
            _MODEL.update(obj=m, device=str(dev), loaded_at=time.time(), names=names,
                          legacy_names=raw if is_legacy_label_list(raw) else None,
                          card=_card_applies(names))
            _CTX["log"].info("[detect] model loaded from %s on device %s"
                             % (WEIGHTS, dev))
        except Exception as e:
            _MODEL["err"] = "%s: %s" % (type(e).__name__, str(e)[:200])
            _CTX["log"].warning("[detect] model unavailable — %s" % _MODEL["err"])
    return _MODEL


def _predict(img_bytes, annotate):
    m = _load()
    if m["obj"] is None:
        raise RuntimeError(m["err"] or "model not loaded")
    from PIL import Image
    import numpy as np
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    res = m["obj"].predict(np.array(im), imgsz=IMGSZ, conf=CONF,
                           device=m["device"], verbose=False)[0]
    names = m["names"] or m["obj"].names
    legacy = m["legacy_names"]
    card = m["card"]
    dets = []
    if res.boxes is not None and len(res.boxes):
        for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                           res.boxes.conf.tolist()):
            sp = names[int(c)]
            d = {"species": sp, "conf": round(float(s), 3),
                 "box_xyxy": [round(float(v), 1) for v in b],
                 "species_holdout_map50_95": PER_SPECIES.get(sp) if card else None,
                 "low_reliability_species": card and sp in WEAK}
            if legacy:
                # v3.60.0: the checkpoint's own label, for clients still keyed by it.
                d["legacy_label"] = legacy.get(int(c))
            dets.append(d)
    out = None
    if annotate:
        buf = io.BytesIO()
        Image.fromarray(res.plot()[:, :, ::-1]).save(buf, "JPEG", quality=85)
        out = buf.getvalue()
    return dets, out


@router.get("/api/detect/model")
def detect_model(request: Request):
    _ = _CTX["actor"](request)
    m = _load()
    return JSONResponse({"ok": m["obj"] is not None, "error": m["err"],
                         "weights": WEIGHTS, "device": m["device"],
                         "class_names": m["names"],
                         "checkpoint_labels_legacy": m["legacy_names"] is not None,
                         "conf": CONF, "imgsz": IMGSZ, **MODEL_META,
                         "card_applies": m["card"],
                         "known_weak_species": WEAK if m["card"] else []})


@router.post("/api/detect")
async def detect_post(request: Request):
    _ = _CTX["actor"](request)
    body = await request.body()
    if body[:2] != b"\xff\xd8":
        return JSONResponse({"ok": False, "error": "send a JPEG body"}, status_code=400)
    t0 = time.time()
    try:
        dets, _img = await run_in_threadpool(_predict, body, False)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:200]}, status_code=503)
    return JSONResponse({"ok": True, "detections": dets, "n": len(dets),
                         "ms": round((time.time() - t0) * 1000, 1),
                         "model": MODEL_META["name"],
                         "known_weak_species": WEAK if _MODEL["card"] else []})


@router.get("/api/detect/frame/{sid}.jpg")
async def detect_frame(request: Request, sid: str, cam: str = ""):
    """Latest live robot frame with detections drawn.

    Falls back to the unannotated frame when the model is unavailable — a live view
    that keeps showing the robot is worth more than one that errors out because a
    detector could not load.
    """
    _ = _CTX["actor"](request)
    try:
        # v3.24.9: cam passthrough — '' keeps the down-first default, so the
        # overlay annotates the scientific view unless the viewer picks another.
        raw = _CTX["latest_frame"](sid, cam)
    except TypeError:
        raw = _CTX["latest_frame"](sid)      # older ingest module signature
    except Exception:
        raw = None
    if not raw:
        return JSONResponse({"ok": False, "error": "no frame for that session"},
                            status_code=404)
    try:
        dets, img = await run_in_threadpool(_predict, raw, True)
        hdrs = {"Cache-Control": "no-store", "X-Detections": str(len(dets)),
                "X-Detect-Species": ",".join(sorted({d["species"] for d in dets}))[:200]}
        return Response(img, media_type="image/jpeg", headers=hdrs)
    except Exception as e:
        _CTX["log"].warning("[detect] frame passthrough (%s)" % str(e)[:120])
        return Response(raw, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Detect-Error": "1"})


def mount(app, ctx: dict):
    _CTX.update(ctx)
    app.include_router(router)
