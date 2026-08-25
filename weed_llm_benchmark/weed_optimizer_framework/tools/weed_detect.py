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
    Morningglory (0.7324 mAP50-95) does not mean what a detection of Ragweed (0.9767)
    means, and a laser-weeding system downstream should be able to tell.
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

router = APIRouter()
_CTX = {}
_LOCK = threading.Lock()
_MODEL = {"obj": None, "err": None, "loaded_at": None, "device": None}

WEIGHTS = os.path.expanduser(
    os.environ.get("WEED_MODEL", "~/models/cwd12_yolo11n_s102.pt"))
CONF = float(os.environ.get("WEED_CONF", "0.25"))
IMGSZ = int(os.environ.get("WEED_IMGSZ", "640"))

# From docs/BEST_MODEL_CARD.md — independent re-evaluation, job 44454237, n=3 seeds.
PER_SPECIES = {
    "Ragweed": 0.9767, "Purslane": 0.9276, "PalmerAmaranth": 0.9163,
    "Crabgrass": 0.9157, "Carpetweeds": 0.9151, "PricklySida": 0.9118,
    "SpottedSpurge": 0.8818, "Nutsedge": 0.8585, "Sicklepod": 0.8555,
    "Eclipta": 0.8219, "Goosegrass": 0.7973, "Morningglory": 0.7324,
}
WEAK = [k for k, v in PER_SPECIES.items() if v < 0.83]      # the card's three
MODEL_META = {
    "name": "cwd12 YOLO11n (COCO-pretrained, seed 102)",
    "holdout_map50_95": 0.8759, "holdout_std": 0.0030, "n_seeds": 3,
    "trained_on": "CottonWeedDet12 train split (3,671 images, 12 species)",
    "card": "docs/BEST_MODEL_CARD.md",
    "known_weak_species": WEAK,
    "domain_gap_warning": ("trained on close-range handheld field photography; robot "
                           "camera frames are a different distribution and field "
                           "accuracy is unmeasured"),
}


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
            dev = 0 if torch.cuda.is_available() else "cpu"
            m.predict(imgsz=IMGSZ, device=dev, verbose=False,
                      source=__import__("numpy").zeros((IMGSZ, IMGSZ, 3), dtype="uint8"))
            _MODEL.update(obj=m, device=str(dev), loaded_at=time.time())
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
    names = m["obj"].names
    dets = []
    if res.boxes is not None and len(res.boxes):
        for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                           res.boxes.conf.tolist()):
            sp = names[int(c)]
            dets.append({"species": sp, "conf": round(float(s), 3),
                         "box_xyxy": [round(float(v), 1) for v in b],
                         "species_holdout_map50_95": PER_SPECIES.get(sp),
                         "low_reliability_species": sp in WEAK})
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
                         "conf": CONF, "imgsz": IMGSZ, **MODEL_META})


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
                         "known_weak_species": WEAK})


@router.get("/api/detect/frame/{sid}.jpg")
async def detect_frame(request: Request, sid: str):
    """Latest live robot frame with detections drawn.

    Falls back to the unannotated frame when the model is unavailable — a live view
    that keeps showing the robot is worth more than one that errors out because a
    detector could not load.
    """
    _ = _CTX["actor"](request)
    try:
        raw = _CTX["latest_frame"](sid)
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
