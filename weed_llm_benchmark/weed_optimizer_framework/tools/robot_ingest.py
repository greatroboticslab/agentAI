"""Robot live ingest — the platform half of the robot cloud uplink.

Contract: 241robot repo, docs/CLOUD_UPLINK_PLAN.md rev 1.1 (amendments in
CLOUD_UPLINK_PLATFORM_REVIEW.md). Endpoints under /api/robot/*:

  GET  /api/robot/ping                 contract version + server time (skew check)
  POST /api/robot/session/start        open a live session -> {session_id}
  POST /api/robot/ingest               batched telemetry samples (gzip JSON, seq-deduped)
  POST /api/robot/frame                one JPEG camera frame (raw body)
  POST /api/robot/session/stop         finalize -> registered platform dataset
  GET  /api/robot/sessions             recent sessions (live first) for the live view
  GET  /api/robot/live/{sid}           latest per-source snapshot + frame pointer
  GET  /api/robot/frame_latest/{sid}.jpg   the newest received frame
  GET  /robot/live                     minimal polling live-view page

Auth is the dashboard's global middleware (X-API-Key or a signed-in user); this
module only attributes actors. Storage is one directory per session under
uploads/<slug>/ : append-only <source>.jsonl + frames/<ts>.jpg + manifest.json +
robot_session.json (live state snapshot). The directory is registered as a normal
platform dataset at session START so a running session is visible mid-flight;
platform-native CSVs (gps/imu/telemetry/control, shared absolute timestamps) are
materialized every 5 minutes and at stop. A reaper thread finalizes sessions idle
longer than 10 minutes (the robot sees 410 and re-opens — contract behavior), and
sessions found still "live" at server start are finalized as interrupted.
"""
import gzip
import json
import re
import secrets
import threading
import time
from bisect import bisect_left
from collections import deque
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

router = APIRouter()
_CTX = {}                      # helpers injected by dashboard_server.mount()
_SESS = {}                     # sid -> session state dict (single-process server)
_DONE = {}                     # sid -> slug of finalized sessions (idempotent stop)
_LOCK = threading.Lock()       # guards _SESS/_DONE membership
_RATE = {}                     # (actor, kind) -> deque of request timestamps

CONTRACT = 1
MAX_BATCH_WIRE = 1_000_000     # hard 413 on the post-gzip body (contract §review 2.6)
MAX_BATCH_RAW = 8_000_000      # decompressed safety cap
ADVERT_BATCH_KB = 256          # soft target advertised in session/start
MIN_FRAME_INTERVAL_S = 0.2     # advertised; matches the robot UI's 0.2-5 fps range
MAX_FRAME_BYTES = 300_000
SESSION_DISK_CAP = 5 * 1024 ** 3   # frames stop with 507 above this; telemetry continues
INGEST_RPS, FRAME_RPS = 4, 6       # per-actor token buckets (contract §review 2.6)
IDLE_CLOSE_S = 600                 # auto-finalize after 10 min silence
CSV_ROLL_S = 300                   # re-materialize CSVs every 5 min while live

_SID_RE = re.compile(r"^rs_[a-f0-9]{16}$")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _log():
    return _CTX["log"]


def _uploads_root() -> Path:
    return Path(_CTX["repo"]) / "uploads"


def _actor(request) -> str:
    try:
        return _CTX["actor"](request)
    except Exception:
        return "?"


def _rate_ok(actor: str, kind: str, limit: int) -> bool:
    """Sliding 1-second window per (actor, kind). Burst = 2x the steady rate."""
    now = time.time()
    with _LOCK:
        dq = _RATE.setdefault((actor, kind), deque(maxlen=limit * 4))
        while dq and now - dq[0] > 1.0:
            dq.popleft()
        if len(dq) >= limit * 2:
            return False
        dq.append(now)
        return True


def _err(status: int, msg: str, **hdrs):
    return JSONResponse({"ok": False, "error": msg}, status_code=status,
                        headers=(hdrs or None))


def _gone():
    return _err(410, "session closed")


def _session(sid: str):
    with _LOCK:
        return _SESS.get(sid)


def _state_snapshot(s: dict) -> dict:
    return {k: s[k] for k in s if k not in ("lock",)}


def _persist_state(s: dict):
    try:
        p = Path(s["dir"]) / "robot_session.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(_state_snapshot(s), default=str))
        tmp.replace(p)
    except Exception as e:
        _log().warning(f"[robot] state persist failed for {s.get('slug')}: {e}")


# --------------------------------------------------------------------------- #
# dataset registration (mirrors the manual-upload record so every existing
# platform feature — gallery, analysis, chat, training — sees the session)
# --------------------------------------------------------------------------- #
def _register(s: dict, finalize: bool = False):
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    fields = {
        "status": "downloaded",
        "source": "robot_live",
        "domain": s["domain"],
        "modality": "sensor",
        "format": "sensor",
        "uploaded_by": s["actor"],
        "uploaded_at": s["registered_at"],
        "downloaded_at": now,
        "local_path": s["dir"],
        "local_images": s["frame_count"],
        "n_local_files": s["n_files"],
        "n_local_labels": 0,
        "class_names": [],
        "harvest_round": _CTX["current_round"](),
        "display_name": s["name"],
        "goal": s["goal"],
        "license": "unspecified",
        "version": s["version"],
        "provenance": {"source": "robot_live", "robot_id": s["robot_id"],
                       "uploaded_by": s["actor"], "uploaded_at": s["registered_at"],
                       "license": "unspecified", "version": s["version"],
                       "session_id": s["sid"], "live": (not finalize)},
    }
    try:
        _CTX["append_manual_upload"](s["slug"], fields)
    except Exception as e:
        _log().warning(f"[robot] manual_uploads write failed: {e}")
    try:
        from . import db as _db
        _db.upsert_slug(s["slug"], fields, actor=s["actor"])
    except Exception as e:
        _log().warning(f"[robot] registry upsert failed: {e}")


# --------------------------------------------------------------------------- #
# CSV materialization (contract §4.5 — the shapes the sensor analyzer expects)
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path):
    rows = []
    try:
        with open(path, "r") as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass                    # tolerate a partial trailing line
    except FileNotFoundError:
        pass
    rows.sort(key=lambda r: r.get("ts") or 0)
    return rows


def _nearest(ts_list, rows, t, tol):
    """rows sorted by ts; return the row nearest t within tol, else None."""
    if not rows:
        return None
    i = bisect_left(ts_list, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(rows):
            d = abs(ts_list[j] - t)
            if d <= tol and (best is None or d < best[0]):
                best = (d, rows[j])
    return best[1] if best else None


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        # repr = shortest round-trip form. NOT "%.6g": that collapsed epoch
        # timestamps (1786829160.35 -> 1.78683e+09), which zeroed the shared-clock
        # span and truncated lat/lon to ~10 m. Found on the first real live run.
        return repr(v)
    return str(v)


def _write_csv(path: Path, header, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(_fmt(x) for x in r) + "\n")
    tmp.replace(path)


_IMU_COLS = [("ax", "accX"), ("ay", "accY"), ("az", "accZ"), ("gyro_z", "gyroZ"),
             ("gyro_x", "gyroX"), ("gyro_y", "gyroY"), ("heading", "heading"),
             ("mag_x", "magX"), ("mag_y", "magY"), ("mag_z", "magZ")]
_TEL_COLS = [("enc_l", "encL"), ("enc_r", "encR"), ("speed_l_cps", "speedL"),
             ("speed_r_cps", "speedR"), ("current_l", "currentL"),
             ("current_r", "currentR"), ("temp_c", "temp"), ("voltage", "voltage"),
             ("battery_pct", "percent")]
_CTL_COLS = [("throttle", "throttle"), ("steer", "steer"), ("applied_a", "applied_a"),
             ("applied_b", "applied_b"), ("cmd", "last_cmd")]
_KNOWN = {"imu", "gps", "telemetry", "control"}


def _materialize(s: dict):
    """(Re)generate the platform-native CSVs from the JSONL ground truth."""
    d = Path(s["dir"])
    imu = _read_jsonl(d / "imu.jsonl")
    tel = _read_jsonl(d / "telemetry.jsonl")
    gps = _read_jsonl(d / "gps.jsonl")
    ctl = _read_jsonl(d / "control.jsonl")
    imu_ts = [r.get("ts") or 0 for r in imu]
    tel_ts = [r.get("ts") or 0 for r in tel]
    mpc = s.get("meters_per_count") or 0.001

    if imu:
        _write_csv(d / "imu.csv", ["timestamp"] + [c for c, _ in _IMU_COLS],
                   [[r.get("ts")] + [(r.get("data") or {}).get(k) for _, k in _IMU_COLS]
                    for r in imu])
    if tel:
        _write_csv(d / "telemetry.csv", ["timestamp"] + [c for c, _ in _TEL_COLS],
                   [[r.get("ts")] + [(r.get("data") or {}).get(k) for _, k in _TEL_COLS]
                    for r in tel])
    if ctl:
        _write_csv(d / "control.csv", ["timestamp"] + [c for c, _ in _CTL_COLS],
                   [[r.get("ts")] + [(r.get("data") or {}).get(k) for _, k in _CTL_COLS]
                    for r in ctl])
    if gps:
        rows = []
        for r in gps:
            g = r.get("data") or {}
            t = r.get("ts")
            lat = g.get("pi_lat") if g.get("pi_lat") is not None else g.get("board_lat")
            lon = g.get("pi_lon") if g.get("pi_lon") is not None else g.get("board_lon")
            spd = hdg = None
            tr = _nearest(tel_ts, tel, t, 1.0)
            if tr:
                td = tr.get("data") or {}
                try:
                    spd = (abs(float(td.get("speedL") or 0)) +
                           abs(float(td.get("speedR") or 0))) / 2.0 * mpc
                except Exception:
                    spd = None
            ir = _nearest(imu_ts, imu, t, 0.5)
            if ir:
                hdg = (ir.get("data") or {}).get("heading")
            rows.append([t, lat, lon, spd, hdg])
        _write_csv(d / "gps.csv",
                   ["timestamp", "lat", "lon", "speed_mps", "heading_deg"], rows)

    # unknown sources (e.g. the lasercar's `laser` events): best-effort generic CSV
    for src in list(s["counts"].keys()):
        if src in _KNOWN:
            continue
        rows = _read_jsonl(d / (src + ".jsonl"))
        if not rows:
            continue
        keys = sorted({k for r in rows for k in (r.get("data") or {})
                       if isinstance((r.get("data") or {}).get(k), (int, float, str, bool))})
        _write_csv(d / (src + ".csv"), ["timestamp"] + keys,
                   [[r.get("ts")] + [(r.get("data") or {}).get(k) for k in keys]
                    for r in rows])

    s["n_files"] = len([p for p in d.iterdir() if p.is_file()])
    s["last_csv"] = time.time()


def _finalize(s: dict, robot_counts=None, interrupted=False):
    with s["lock"]:
        if s["status"] != "live":
            return
        s["status"] = "done"
    try:
        _materialize(s)
    except Exception as e:
        _log().warning(f"[robot] materialize failed for {s['slug']}: {e}")
    try:
        d = Path(s["dir"])
        (d / "manifest.json").write_text(json.dumps({
            "robot_id": s["robot_id"], "name": s["name"], "goal": s["goal"],
            "domain": s["domain"], "sources": s["sources"], "contract": s["contract"],
            "started": s["started"], "ended": time.time(), "counts": s["counts"],
            "dropped_reported": s["dropped"], "frame_count": s["frame_count"],
            "frame_bytes": s["frame_bytes"], "meters_per_count": s.get("meters_per_count"),
            "robot_counts": robot_counts, "interrupted": interrupted}, indent=1))
    except Exception:
        pass
    _register(s, finalize=True)
    _persist_state(s)
    with _LOCK:
        _SESS.pop(s["sid"], None)
        _DONE[s["sid"]] = s["slug"]
        if len(_DONE) > 200:                       # bounded idempotency memory
            for k in list(_DONE)[:-100]:
                _DONE.pop(k, None)
    _log().info(f"[robot] session {s['sid']} finalized -> {s['slug']} "
                f"(counts={s['counts']} frames={s['frame_count']}"
                f"{' INTERRUPTED' if interrupted else ''})")


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@router.get("/api/robot/ping")
def robot_ping(request: Request):
    return JSONResponse({"ok": True, "contract": CONTRACT, "server_ts": time.time()})


@router.post("/api/robot/session/start")
async def robot_session_start(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _err(400, "JSON body required")
    robot_id = re.sub(r"[^A-Za-z0-9_-]", "", str(body.get("robot_id") or ""))[:40]
    domain = re.sub(r"[^a-z0-9_]", "", str(body.get("domain") or "").lower())[:40]
    if not robot_id or not domain:
        return _err(400, "robot_id and domain required")
    try:
        from . import db as _db
        if not _db.get_domain(domain):
            return _err(404, "unknown domain '%s' — create the project first" % domain)
    except Exception:
        pass                                        # registry down: accept anyway
    name = str(body.get("name") or "").strip()[:80] or time.strftime("run-%Y%m%d-%H%M%S")
    goal = str(body.get("goal") or "").strip()[:1000]
    sources = [re.sub(r"[^a-z0-9_-]", "", str(x).lower())[:24]
               for x in (body.get("sources") or []) if str(x).strip()][:16]
    mpc = body.get("meters_per_count")
    cpm = body.get("counts_per_meter")
    try:
        mpc = float(mpc) if mpc else (1.0 / float(cpm) if cpm else None)
    except Exception:
        mpc = None

    sid = "rs_" + secrets.token_hex(8)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")[:40] or "run"
    slug = "rl_%s_%s" % (safe, secrets.token_hex(4))
    d = _uploads_root() / slug
    (d / "frames").mkdir(parents=True, exist_ok=True)
    try:
        prior = [v for v in _CTX["read_manual_uploads"]().values()
                 if v.get("domain") == domain
                 and str(v.get("display_name") or "").strip().lower() == name.lower()]
        version = len(prior) + 1                   # 410-reopened sessions keep the
    except Exception:                              # name; version tells them apart
        version = 1
    s = {"sid": sid, "slug": slug, "dir": str(d), "robot_id": robot_id,
         "domain": domain, "name": name, "goal": goal, "sources": sources,
         "contract": int(body.get("contract") or 1), "meters_per_count": mpc,
         "actor": _actor(request), "version": version,
         "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "started": time.time(), "last_seen": time.time(), "last_seq": 0,
         "counts": {}, "dropped": 0, "frame_count": 0, "frame_bytes": 0,
         "latest": {}, "latest_frame_ts": None, "last_csv": 0.0, "n_files": 0,
         "status": "live", "lock": threading.Lock()}
    with _LOCK:
        _SESS[sid] = s
    _register(s)                                   # visible mid-flight
    _persist_state(s)
    _log().info(f"[robot] session START {sid} robot={robot_id} domain={domain} "
                f"name='{name}' by={s['actor']} -> {slug}")
    return JSONResponse({"ok": True, "session_id": sid,
                         "max_batch_kb": ADVERT_BATCH_KB,
                         "min_frame_interval_s": MIN_FRAME_INTERVAL_S})


@router.post("/api/robot/ingest")
async def robot_ingest(request: Request):
    actor = _actor(request)
    if not _rate_ok(actor, "ingest", INGEST_RPS):
        return _err(429, "ingest rate limit (%d/s)" % INGEST_RPS, **{"Retry-After": "1"})
    raw = await request.body()
    if len(raw) > MAX_BATCH_WIRE:
        return _err(413, "batch too large (> %d bytes on the wire)" % MAX_BATCH_WIRE)
    if raw[:2] == b"\x1f\x8b" or "gzip" in (request.headers.get("content-encoding") or ""):
        try:
            raw = gzip.decompress(raw)
        except Exception:
            return _err(400, "bad gzip body")
        if len(raw) > MAX_BATCH_RAW:
            return _err(413, "batch too large decompressed")
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return _err(400, "bad JSON body")
    sid = str(body.get("session_id") or "")
    s = _session(sid)
    if not s or s["status"] != "live":
        return _gone()
    seq = int(body.get("seq") or 0)
    samples = body.get("samples") or []
    with s["lock"]:
        if seq <= s["last_seq"]:                    # retry of an acked batch
            return JSONResponse({"ok": True, "received": 0, "dup": True})
        s["last_seq"] = seq
        s["last_seen"] = time.time()
        try:
            s["dropped"] += max(0, int(body.get("dropped_since_last") or 0))
        except Exception:
            pass
        by_src, n = {}, 0
        for sm in samples:
            src = re.sub(r"[^a-z0-9_-]", "", str(sm.get("source") or "").lower())[:24]
            if not src:
                continue
            by_src.setdefault(src, []).append(sm)
            n += 1
        d = Path(s["dir"])
        for src, rows in by_src.items():
            with open(d / (src + ".jsonl"), "a") as f:
                for sm in rows:
                    f.write(json.dumps({"ts": sm.get("ts"), "mono": sm.get("mono"),
                                        "data": sm.get("data") or {}}) + "\n")
            s["counts"][src] = s["counts"].get(src, 0) + len(rows)
            last = rows[-1]
            s["latest"][src] = {"ts": last.get("ts"), "data": last.get("data") or {}}
    return JSONResponse({"ok": True, "received": n})


@router.post("/api/robot/frame")
async def robot_frame(request: Request):
    actor = _actor(request)
    if not _rate_ok(actor, "frame", FRAME_RPS):
        return _err(429, "frame rate limit (%d/s)" % FRAME_RPS, **{"Retry-After": "1"})
    qp = request.query_params
    s = _session(str(qp.get("session_id") or ""))
    if not s or s["status"] != "live":
        return _gone()
    body = await request.body()
    if len(body) > MAX_FRAME_BYTES:
        return _err(413, "frame too large (> %d bytes)" % MAX_FRAME_BYTES)
    if body[:2] != b"\xff\xd8":
        return _err(400, "not a JPEG")
    if s["frame_bytes"] + len(body) > SESSION_DISK_CAP:
        return _err(507, "session frame quota exceeded — telemetry still accepted")
    try:
        ts = float(qp.get("ts") or time.time())
    except Exception:
        ts = time.time()
    with s["lock"]:
        (Path(s["dir"]) / "frames" / ("%.3f.jpg" % ts)).write_bytes(body)
        s["frame_count"] += 1
        s["frame_bytes"] += len(body)
        s["latest_frame_ts"] = ts
        s["last_seen"] = time.time()
    return JSONResponse({"ok": True})


@router.post("/api/robot/session/stop")
async def robot_session_stop(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    sid = str(body.get("session_id") or "")
    s = _session(sid)
    if s:
        _finalize(s, robot_counts=body.get("counts"))
        return JSONResponse({"ok": True, "dataset_slug": s["slug"],
                             "gallery_url": "/gallery/" + s["slug"]})
    with _LOCK:
        slug = _DONE.get(sid)
    if slug:                                        # idempotent re-stop
        return JSONResponse({"ok": True, "dataset_slug": slug, "already": True})
    return _gone()


# --------------------------------------------------------------------------- #
# live view (poll-based, matching the platform's no-WebSocket style)
# --------------------------------------------------------------------------- #
@router.get("/api/robot/sessions")
def robot_sessions(request: Request, limit: int = 20, domain: str = ""):
    """Recent sessions, live first. `domain` scopes it to one project (the
    project page's live card polls with its own domain)."""
    dom = re.sub(r"[^a-z0-9_]", "", domain.lower())[:40]
    out = []
    with _LOCK:
        live = list(_SESS.values())
    for s in live:
        if dom and s.get("domain") != dom:
            continue
        out.append({k: s[k] for k in ("sid", "slug", "robot_id", "domain", "name",
                                      "status", "started", "last_seen", "counts",
                                      "dropped", "frame_count")})
    try:                                            # recent finished, newest first
        dirs = sorted(_uploads_root().glob("rl_*/robot_session.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for p in dirs:
            if len(out) >= limit:
                break
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            if j.get("status") == "live":           # stale file of a live session
                continue
            if dom and j.get("domain") != dom:
                continue
            out.append({k: j.get(k) for k in ("sid", "slug", "robot_id", "domain",
                                              "name", "status", "started", "last_seen",
                                              "counts", "dropped", "frame_count")})
    except Exception:
        pass
    return JSONResponse({"ok": True, "sessions": out[:limit]})


@router.get("/api/robot/live/{sid}")
def robot_live(request: Request, sid: str):
    if not _SID_RE.match(sid):
        return _err(400, "bad session id")
    s = _session(sid)
    if not s:
        with _LOCK:
            slug = _DONE.get(sid)
        if slug:
            return JSONResponse({"ok": True, "status": "done", "slug": slug,
                                 "gallery_url": "/gallery/" + slug})
        return _gone()
    with s["lock"]:
        return JSONResponse({
            "ok": True, "status": "live", "sid": sid, "slug": s["slug"],
            "robot_id": s["robot_id"], "domain": s["domain"], "name": s["name"],
            "started": s["started"], "age_s": round(time.time() - s["started"], 1),
            "last_seen_s": round(time.time() - s["last_seen"], 1),
            "counts": s["counts"], "dropped": s["dropped"], "seq": s["last_seq"],
            "frame_count": s["frame_count"], "frame_bytes": s["frame_bytes"],
            "latest": s["latest"], "latest_frame_ts": s["latest_frame_ts"],
            "frame_url": ("/api/robot/frame_latest/%s.jpg" % sid
                          if s["latest_frame_ts"] else None)})


@router.get("/api/robot/frame_latest/{sid}.jpg")
def robot_frame_latest(request: Request, sid: str):
    if not _SID_RE.match(sid):
        return _err(400, "bad session id")
    s = _session(sid)
    if not s or not s["latest_frame_ts"]:
        return _err(404, "no frame yet")
    p = Path(s["dir"]) / "frames" / ("%.3f.jpg" % s["latest_frame_ts"])
    try:
        return Response(p.read_bytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except Exception:
        return _err(404, "frame unreadable")


@router.get("/robot/live")
def robot_live_page(request: Request):
    return HTMLResponse(_LIVE_PAGE)


_LIVE_PAGE = r'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Live</title><style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;padding:14px;max-width:900px}
 .card{border:1px solid #4443;border-radius:10px;padding:12px;margin:10px 0}
 .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 .kv{font-size:13px;opacity:.85}.kv b{font-size:16px}
 #frame{max-width:100%;border-radius:8px;background:#0002;min-height:120px}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#888;margin-right:6px}
 .dot.live{background:#e33;animation:p 1.2s infinite}@keyframes p{50%{opacity:.35}}
 select{padding:6px;border-radius:8px}
 a{color:inherit}
</style></head><body>
<h2 style="margin:4px 0"><span class="dot" id="dot"></span>Robot Live</h2>
<div class="row"><select id="pick"></select>
 <span class="kv" id="meta"></span></div>
<div class="card"><img id="frame" alt="waiting for the first camera frame…"></div>
<div class="card"><div class="row" id="stats"></div></div>
<div class="card kv" id="latest" style="white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px"></div>
<script>
let SID=new URLSearchParams(location.search).get('session')||'';
async function sessions(){
  const r=await fetch('/api/robot/sessions',{credentials:'include'}); const d=await r.json();
  const el=document.getElementById('pick'); el.innerHTML='';
  (d.sessions||[]).forEach(s=>{const o=document.createElement('option');
    o.value=s.sid||''; o.dataset.slug=s.slug||'';
    o.textContent=(s.status==='live'?'● LIVE  ':'')+(s.name||s.slug)+'  ['+(s.robot_id||'?')+']';
    el.appendChild(o);});
  if(!SID&&d.sessions&&d.sessions.length) SID=d.sessions[0].sid||'';
  if(SID) el.value=SID;
  el.onchange=()=>{SID=el.value; history.replaceState(null,'','?session='+SID);};
}
function kv(l,v){return '<span class="kv">'+l+' <b>'+v+'</b></span>';}
async function tick(){
  if(!SID){return}
  try{
    const r=await fetch('/api/robot/live/'+SID,{credentials:'include'});
    if(r.status===410){document.getElementById('meta').textContent='session closed';
      document.getElementById('dot').className='dot';return}
    const d=await r.json(); if(!d.ok) return;
    document.getElementById('dot').className='dot'+(d.status==='live'?' live':'');
    document.getElementById('meta').innerHTML=(d.status==='live'
      ? 'streaming — last packet '+d.last_seen_s+'s ago'
      : 'finished') + (d.slug?' · <a href="/gallery/'+d.slug+'">dataset ↗</a>':'');
    if(d.status!=='live'){return}
    const c=d.counts||{};
    document.getElementById('stats').innerHTML=
      kv('seq',d.seq)+kv('frames',d.frame_count)+kv('dropped',d.dropped)+
      Object.keys(c).map(k=>kv(k,c[k])).join('');
    if(d.frame_url){document.getElementById('frame').src=d.frame_url+'?t='+Date.now();}
    const L=d.latest||{};
    document.getElementById('latest').textContent=Object.keys(L).map(
      k=>k.padEnd(10)+JSON.stringify(L[k].data)).join('\n');
  }catch(e){}
}
sessions(); setInterval(sessions,10000); setInterval(tick,1000); tick();
</script></body></html>'''


# --------------------------------------------------------------------------- #
# reaper + recovery
# --------------------------------------------------------------------------- #
def _reaper_loop():
    while True:
        time.sleep(30)
        try:
            now = time.time()
            with _LOCK:
                live = list(_SESS.values())
            for s in live:
                if now - s["last_seen"] > IDLE_CLOSE_S:
                    _log().info(f"[robot] session {s['sid']} idle "
                                f"{int(now - s['last_seen'])}s — auto-finalizing")
                    _finalize(s)
                elif now - s["last_csv"] > CSV_ROLL_S:
                    try:
                        _materialize(s)
                        _persist_state(s)
                    except Exception as e:
                        _log().warning(f"[robot] rolling CSV failed: {e}")
        except Exception as e:
            try:
                _log().warning(f"[robot] reaper error: {e}")
            except Exception:
                pass


def _recover():
    """Finalize sessions the previous server run left live (contract: the robot
    gets 410 on its next post and re-opens a fresh session)."""
    try:
        for p in _uploads_root().glob("rl_*/robot_session.json"):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            if j.get("status") != "live":
                continue
            j["dir"] = str(p.parent)
            j["lock"] = threading.Lock()
            j.setdefault("counts", {})
            j.setdefault("dropped", 0)
            j.setdefault("frame_count", 0)
            j.setdefault("frame_bytes", 0)
            j.setdefault("n_files", 0)
            _log().info(f"[robot] recovering interrupted session {j.get('sid')}")
            with _LOCK:
                _SESS[j["sid"]] = j
            _finalize(j, interrupted=True)
    except Exception as e:
        _log().warning(f"[robot] recovery scan failed: {e}")


def mount(app, ctx: dict):
    """Called once by dashboard_server after its helpers exist."""
    _CTX.update(ctx)
    _recover()
    threading.Thread(target=_reaper_loop, daemon=True).start()
    app.include_router(router)
