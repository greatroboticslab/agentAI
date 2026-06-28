"""
v3.0.30 — Live dashboard FastAPI server (Cloudflare Tunnel friendly).

Hosts the autonomous-weed-detection dashboard from the cluster, serving:
  - GET /                       → smart index (auto-redirects to /dashboard/)
  - GET /dashboard/{page}       → static HTML pages
  - GET /api/state              → JSON of current registry state (cached 60s)
  - GET /api/sample/{slug}/{file} → on-demand bbox-rendered thumbnail
  - GET /api/img/{slug}/{file}  → original full-resolution image
  - GET /healthz                → liveness probe

Image rendering is on-demand with a filesystem cache so repeated requests are
fast and we never preload all 372 (eventually 50K+) thumbnails into RAM.

Run via:
  uvicorn weed_optimizer_framework.tools.dashboard_server:app --host 0.0.0.0 --port 8080

Environment:
  REPO_ROOT  — defaults to /ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("dash")

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
)).resolve()

REGISTRY_PATH = REPO / "results" / "framework" / "dataset_registry.json"
DOCS_DIR      = REPO / "docs" / "dashboard"
CACHE_DIR     = REPO / "dashboard_cache"
JOBD_DIR      = REPO / "results" / "framework" / "jobd_runs"
PYCOCO_GLOB   = REPO / "results" / "framework"
FLAGS_PATH    = REPO / "results" / "framework" / "dataset_flags.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)

NEVER_TRAIN = {"cottonweeddet12", "weedsense", "francesco__weed_crop_aerial"}
CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]

# Reuse the static HTML generator's slug helpers
from weed_optimizer_framework.tools.dashboard_generator import (  # noqa: E402
    slug_source, slug_crop, annotation_type_human,
)

app = FastAPI(title="Autonomous Weed Detection Dashboard")
# Allow GitHub Pages and any browser to call the API.
# v3.0.71.7 (2026-06-01): OPTIONS added so preflight succeeds for
# cross-origin fetches with Authorization header (which github.io
# password prompt sends). Without this, browser sees preflight 401,
# returns TypeError: Failed to fetch, password page can't auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["WWW-Authenticate"],
    max_age=3600,
)

# ============================================================================
# v3.0.70 (2026-05-31) — HTTP Basic auth + IP rate-limit
#
# Threat model: tunnel URL is on a public github.io page (necessary for the
# redirect to work). Without auth, anyone who finds the github profile can
# trigger SLURM jobs / read all data. This middleware adds:
#
#   - HTTP Basic auth with password loaded from /jet/home/byler/.dashpass
#     (read at startup; never logged; never returned in responses).
#   - Per-IP failed-attempt counter: 5 failures → 1h lockout.
#   - Exempt paths: /healthz (so cloudflared/uvicorn keep-alive works), and
#     /tunnel_url.json (read by 404.html — not a secret).
#
# Username is fixed to "harry" (single-tenant research dashboard). Password
# in plain-text file is acceptable for a research cluster — encrypted disk +
# UNIX permissions (0600) are the access control.
# ============================================================================
import base64 as _base64
import secrets as _secrets
from collections import defaultdict as _defaultdict

# v3.0.99.35: portable secret paths. os.path.expanduser("~") = /jet/home/byler on
# the cluster (unchanged) and /home/lab on the lab server → works in BOTH places.
# Env override DASHPASS_FILE / ROBOFLOW_KEY_FILE for non-standard locations.
_DASHPASS_FILE = os.environ.get("DASHPASS_FILE", os.path.expanduser("~/.dashpass"))
_ROBOFLOW_KEY_FILE = os.environ.get("ROBOFLOW_KEY_FILE", os.path.expanduser("~/.roboflow_key"))
_AUTH_USER = os.environ.get("DASH_USER", "harry")  # env-configurable for testing
_AUTH_PASS = None
try:
    with open(_DASHPASS_FILE, "r") as _f:
        _AUTH_PASS = _f.read().strip() or None
    if _AUTH_PASS:
        log.info(f"[auth] HTTP Basic auth ENABLED (password from {_DASHPASS_FILE})")
    else:
        log.warning(f"[auth] {_DASHPASS_FILE} empty — auth DISABLED")
except FileNotFoundError:
    log.warning(f"[auth] {_DASHPASS_FILE} missing — auth DISABLED. "
                f"Run: echo '<password>' > {_DASHPASS_FILE} && "
                f"chmod 600 {_DASHPASS_FILE}")
except Exception as _e:
    log.error(f"[auth] could not read {_DASHPASS_FILE}: {_e}; auth DISABLED")

# Per-IP attempt tracker. {ip: (failed_count, lockout_until_epoch)}
# In-memory: forgets on uvicorn restart, which is fine — that means a
# legitimate user who got locked just needs to wait for the next dashboard
# refresh (≤48h auto-resubmit, or restart_dashboard button).
_AUTH_FAIL = _defaultdict(lambda: (0, 0.0))
_AUTH_LOCK_THRESHOLD = 5      # 5 failures
_AUTH_LOCK_SECONDS = 3600     # locks for 1h

# --------------------------------------------------------------------------- #
# v3.0.130 (Z3) — OPTIONAL Google OAuth login (Prof Zhang: students sign in with
# their own account; we save users + attribute uploads). Enabled ONLY when the
# OAuth env vars are all set; otherwise the dashboard stays exactly on Basic auth
# (1/1). A signed (HMAC, stdlib — no new deps) session cookie carries the logged
# in user. Basic auth always remains valid (curl -u, the cluster-control path).
#
# To ENABLE Google login the operator must (one-time):
#   1. Google Cloud Console → APIs & Services → Credentials → Create OAuth client
#      ID (type: Web application).
#   2. Authorized redirect URI = <public dashboard URL>/auth/google/callback
#      (e.g. https://lab-b660m-c.tailfa6424.ts.net/auth/google/callback).
#   3. Put the values in the dashboard service env (deploy/run_dashboard_labserver.sh):
#        export GOOGLE_CLIENT_ID=...    export GOOGLE_CLIENT_SECRET=...
#        export OAUTH_REDIRECT_BASE=https://lab-b660m-c.tailfa6424.ts.net
#      then restart weed-dashboard. /login then shows "Sign in with Google".
# --------------------------------------------------------------------------- #
import hmac as _hmac
_GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
_GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
_OAUTH_REDIRECT_BASE = os.environ.get("OAUTH_REDIRECT_BASE", "").rstrip("/")
_GOOGLE_ENABLED = bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET and _OAUTH_REDIRECT_BASE)
_SESSION_COOKIE = "agentai_session"
_SESSION_TTL = 7 * 24 * 3600


def _load_session_secret() -> bytes:
    k = os.environ.get("SESSION_SECRET", "").strip()
    if k:
        return k.encode()
    p = os.path.expanduser("~/.dash_session_key")
    try:
        if os.path.isfile(p):
            return open(p, "rb").read()
        kb = _secrets.token_bytes(32)
        with open(p, "wb") as f:
            f.write(kb)
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
        return kb
    except Exception:
        return (_AUTH_PASS or "agentai-session-fallback").encode()


_SESSION_SECRET = _load_session_secret()


def _b64u(b: bytes) -> str:
    return _base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64u_dec(s: str) -> bytes:
    return _base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign_session(payload: dict) -> str:
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64u(_hmac.new(_SESSION_SECRET, body.encode(), hashlib.sha256).digest())
    return body + "." + sig


def _verify_session(cookie: str):
    try:
        body, _, sig = (cookie or "").partition(".")
        if not body or not sig:
            return None
        exp_sig = _b64u(_hmac.new(_SESSION_SECRET, body.encode(),
                                  hashlib.sha256).digest())
        if not _hmac.compare_digest(sig, exp_sig):
            return None
        data = json.loads(_b64u_dec(body))
        if float(data.get("exp", 0)) < time.time():
            return None
        return data
    except Exception:
        return None


def _session_user(request):
    try:
        c = request.cookies.get(_SESSION_COOKIE)
    except Exception:
        c = None
    if not c:
        return None
    return _verify_session(c)


# v3.0.140 — OUR-OWN API keys (no paid/Google needed): scripts/robots/other
# machines authenticate by sending  X-API-Key: ak_...  The server stores only the
# sha256 hash. A key authenticates as a member-level identity (upload/read); it
# does NOT grant cluster GPU access (that stays admin/granted). Keys live in
# ~/.dash_api_keys.json {hash: {label, created, owner}} (chmod 600, not in git).
_API_KEYS_FILE = os.path.expanduser("~/.dash_api_keys.json")


def _read_api_keys() -> dict:
    try:
        if os.path.isfile(_API_KEYS_FILE):
            return json.load(open(_API_KEYS_FILE)) or {}
    except Exception:
        pass
    return {}


def _write_api_keys(d: dict) -> None:
    tmp = _API_KEYS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, _API_KEYS_FILE)
    try:
        os.chmod(_API_KEYS_FILE, 0o600)
    except Exception:
        pass


def _check_api_key(request):
    """Return the key record {label,...} if a valid X-API-Key is present, else None."""
    try:
        k = (request.headers.get("x-api-key") or "").strip()
    except Exception:
        k = ""
    if not k:
        return None
    h = hashlib.sha256(k.encode()).hexdigest()
    rec = _read_api_keys().get(h)
    return rec if rec else None


# Public paths (no auth required)
_AUTH_EXEMPT_PATHS = {
    "/healthz",                      # cloudflared probe
    "/tunnel_url.json",              # not a secret, github.io reads it
    "/login",                        # v3.0.130: login page
    "/logout",                       # clear session
    "/auth/google/start",            # OAuth redirect to Google
    "/auth/google/callback",         # OAuth return
}
_AUTH_EXEMPT_PREFIXES = (
    "/static/",                      # static assets if any
)


def _client_ip(request) -> str:
    """Best-effort client IP — through cloudflared the real IP is in headers."""
    hdr = request.headers
    for k in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        v = hdr.get(k, "").split(",")[0].strip()
        if v:
            return v
    return getattr(request.client, "host", "?") or "?"


@app.middleware("http")
async def _auth_and_rate_limit(request, call_next):
    """Gate every non-exempt request behind HTTP Basic auth.

    Rate limit: 5 failed attempts from an IP → 1h lockout (HTTP 429)."""
    if _AUTH_PASS is None:
        # Auth not configured; let everything through (logged at startup).
        return await call_next(request)

    # v3.0.71.7: CORS preflight (OPTIONS) must pass without auth. Otherwise
    # cross-origin browser fetches with Authorization header fail with
    # 'TypeError: Failed to fetch' — user-reported bug 2026-06-01 ~03:30.
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS or any(
        path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES
    ):
        return await call_next(request)

    ip = _client_ip(request)
    n_fail, locked_until = _AUTH_FAIL[ip]
    now = time.time()
    if locked_until > now:
        remaining = int(locked_until - now)
        return JSONResponse(
            status_code=429,
            content={"error": "rate-limited",
                     "msg": f"too many failed auth attempts; locked for "
                            f"{remaining // 60}m {remaining % 60}s"},
            headers={"Retry-After": str(remaining)},
        )

    # v3.0.130: a valid signed session cookie (Google login) authenticates first.
    if _session_user(request) is not None:
        if n_fail > 0:
            _AUTH_FAIL[ip] = (0, 0.0)
        return await call_next(request)

    # v3.0.140: our-own API key (X-API-Key) — for scripts / robots / other machines.
    if _check_api_key(request) is not None:
        if n_fail > 0:
            _AUTH_FAIL[ip] = (0, 0.0)
        return await call_next(request)

    auth_hdr = request.headers.get("authorization", "")
    ok = False
    if auth_hdr.startswith("Basic "):
        try:
            decoded = _base64.b64decode(auth_hdr[6:]).decode("utf-8")
            user, sep, passwd = decoded.partition(":")
            ok = (sep == ":"
                  and _secrets.compare_digest(user, _AUTH_USER)
                  and _secrets.compare_digest(passwd, _AUTH_PASS))
        except Exception:
            ok = False

    if not ok:
        # v3.0.130: when Google login is enabled and a browser navigates here
        # with NO credentials at all, send them to /login instead of a Basic
        # popup. Requests that DO present a (wrong) Basic header still count as
        # failed attempts; curl -u and API calls are unaffected.
        if (_GOOGLE_ENABLED and not auth_hdr and request.method == "GET"
                and "text/html" in request.headers.get("accept", "")):
            return RedirectResponse(url="/login", status_code=302)
        n_fail += 1
        if n_fail >= _AUTH_LOCK_THRESHOLD:
            _AUTH_FAIL[ip] = (n_fail, now + _AUTH_LOCK_SECONDS)
            log.warning(
                f"[auth] IP {ip} reached {n_fail} failed attempts → "
                f"locked for {_AUTH_LOCK_SECONDS}s"
            )
        else:
            _AUTH_FAIL[ip] = (n_fail, 0.0)
            log.info(f"[auth] failed attempt from {ip} ({n_fail}/{_AUTH_LOCK_THRESHOLD})")
        return Response(
            status_code=401,
            content=b'{"error":"unauthorized","msg":"Basic auth required"}',
            headers={"WWW-Authenticate": 'Basic realm="weed-dashboard"',
                     "Content-Type": "application/json"},
        )

    # Reset on success
    if n_fail > 0:
        _AUTH_FAIL[ip] = (0, 0.0)
    return await call_next(request)


# v3.0.99.45: global RESPONSIVE CSS — inject once into every HTML page so the
# dashboard renders well on BOTH phone and desktop (pages already have viewport +
# auto-fill grids; this adds: fluid images, mobile-scrollable tables, and a
# <=768px breakpoint that shrinks padding/fonts). Conservative — doesn't override
# existing layouts. Marker id="_rwd" prevents double-injection.
_RESPONSIVE_CSS = (
    '<style id="_rwd">'
    'img,svg,canvas,video{max-width:100%;height:auto}'
    'table{max-width:100%}'
    '@media (max-width:768px){'
    'html{-webkit-text-size-adjust:100%}'
    'body{font-size:14px}'
    'table{display:block;overflow-x:auto;white-space:nowrap;-webkit-overflow-scrolling:touch}'
    '.hero,.nav,header,section,main,.wrap,.container{padding-left:.6rem!important;padding-right:.6rem!important}'
    'h1{font-size:1.3rem}h2{font-size:1.1rem}'
    '}'
    '</style>'
)

# v3.0.134: global "signed in as X · Logout" badge, injected before </body> on
# every HTML page (one place). Fills itself from /api/me. Skipped on /login.
_LOGIN_BADGE = (
    '<div id="_authbadge" style="position:fixed;top:8px;right:10px;z-index:99999;'
    'font:12px -apple-system,BlinkMacSystemFont,sans-serif"></div>'
    '<script>(function(){fetch("/api/me",{credentials:"include"})'
    '.then(function(r){return r.json();}).then(function(m){'
    'if(!m||!m.ok)return;var el=document.getElementById("_authbadge");if(!el)return;'
    'var who=m.user||"?";var star=m.is_admin?"\\u2605 ":"";var dot="";'
    'if(m.is_admin&&m.pending_requests>0){dot="<a href=\\"/users\\" title=\\"pending cluster-access requests\\" '
    'style=\\"background:#dc2626;color:#fff;padding:3px 8px;border-radius:20px;text-decoration:none;'
    'font-weight:700;margin-right:6px\\">"+m.pending_requests+" \\u2691</a>";}'
    'el.innerHTML=dot+"<span style=\\"background:#0f172a;color:#fff;padding:4px 10px;'
    'border-radius:20px;box-shadow:0 1px 4px rgba(0,0,0,.25);opacity:.93\\">"+star+who'
    '+" \\u00b7 <a href=\\"/logout\\" style=\\"color:#93c5fd;text-decoration:none\\">Logout</a></span>";'
    '}).catch(function(){});})();</script>'
)


@app.on_event("startup")
def _warm_cluster_master():
    """v3.0.99.46: in lab-control mode, eagerly open + keep alive the SSH
    ControlMaster to the cluster so user actions hit a WARM connection (~1-2s)
    instead of a cold password handshake (~15s → felt like 'buttons hang')."""
    if not _CLUSTER_SSH:
        return
    import threading, time as _t

    def warmer():
        while True:
            try:
                _shell(_ssh_cluster_prefix() + ["true"], timeout=45)
            except Exception:
                pass
            _t.sleep(240)   # ControlPersist=12h; refresh well within it
    threading.Thread(target=warmer, daemon=True,
                     name="cluster_master_warmer").start()
    log.info("[cluster] ControlMaster warmer started")


@app.middleware("http")
async def _inject_responsive_css(request: Request, call_next):
    resp = await call_next(request)
    ctype = (resp.headers.get("content-type", "") or "").lower()
    if "text/html" not in ctype:
        return resp
    body = b""
    async for chunk in resp.body_iterator:
        body += chunk
    try:
        text = body.decode("utf-8", "replace")
        if "_rwd" not in text:
            if "</head>" in text:
                text = text.replace("</head>", _RESPONSIVE_CSS + "</head>", 1)
            else:
                text = _RESPONSIVE_CSS + text
        # v3.0.134: login badge on every page except the login page itself.
        if ("_authbadge" not in text and request.url.path != "/login"
                and "</body>" in text):
            text = text.replace("</body>", _LOGIN_BADGE + "</body>", 1)
        body = text.encode("utf-8")
    except Exception:
        pass
    headers = dict(resp.headers)
    headers.pop("content-length", None)
    return Response(content=body, status_code=resp.status_code, headers=headers)


# v3.0.43.8: pre-warm /classes thumbnail cache on startup in a background
# thread, so the user's first hit lands on a hot cache (otherwise the
# 350-class first-load times out the cloudflare tunnel at ~30s).
@app.on_event("startup")
def _prewarm_classes_cache():
    import threading
    # v3.0.43.22: uvicorn runs --workers 2, so this startup hook fires in
    # BOTH processes → two prewarm threads racing on the same cache dir
    # (duplicate work + Lustre contention). Guard with an atomic lock file
    # so only the first worker prewarms; the second exits immediately.
    lock_fp = _pool_cache_dir / ".prewarm.lock"
    try:
        _pool_cache_dir.mkdir(parents=True, exist_ok=True)
        # O_CREAT|O_EXCL is atomic across processes on the same node.
        fd = os.open(str(lock_fp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        # Stale lock (>2h) → take over; else another worker owns prewarm.
        try:
            if time.time() - lock_fp.stat().st_mtime > 7200:
                lock_fp.unlink(missing_ok=True)
                os.close(os.open(str(lock_fp), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            else:
                log.info("[prewarm] another worker owns the lock — skipping")
                return
        except Exception:
            log.info("[prewarm] lock contention — skipping")
            return
    except Exception as e:
        log.debug(f"[prewarm] lock setup failed ({e}) — proceeding anyway")
    def worker():
        try:
            import time as _t
            _t.sleep(8)  # give uvicorn time to start; serve real requests first
            log.info("[prewarm] starting /classes thumb cache warmer")
            # Call _class_summary_landing for every class — same work
            # /classes does, but on a background thread.
            classes = _all_known_classes()
            n_total = len(classes)
            t0 = _t.time()
            n_done = 0
            for cls in classes:
                try:
                    _class_summary_landing(cls)
                    n_done += 1
                    if n_done % 50 == 0:
                        log.info(f"[prewarm] {n_done}/{n_total} done "
                                  f"({_t.time()-t0:.1f}s elapsed)")
                except Exception as e:
                    log.warning(f"[prewarm] {cls}: {e}")
            log.info(f"[prewarm] complete: {n_done}/{n_total} in "
                      f"{_t.time()-t0:.1f}s — /classes is now hot")
        except Exception as e:
            log.warning(f"[prewarm] worker error: {e}")
    threading.Thread(target=worker, daemon=True, name="thumb_prewarm").start()
    log.info("[prewarm] scheduled thumb-cache warmer in background")

# Mount static HTML if present (generated by dashboard_generator.py at boot)
if DOCS_DIR.is_dir():
    app.mount("/dashboard/static", StaticFiles(directory=DOCS_DIR),
              name="dashboard_static")


# ---------- state caching ----------
_state_cache: dict = {"data": None, "ts": 0.0}
STATE_TTL = 60  # seconds


def _build_state() -> dict:
    """Rebuild full state from disk. Cached for STATE_TTL seconds."""
    state: dict = {
        "generated_at": time.time(),
        "registry_path": str(REGISTRY_PATH),
        "totals": {},
        "datasets": [],
        "crop_counts": {},
        "source_counts": {},
        "annotation_counts": {},
        "jobd_runs": [],
        "pyco_history": [],
        "twelve_class_gt": {},
    }
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
    except Exception as e:
        log.error(f"failed loading registry: {e}")
        return state

    flags = _load_flags()
    state["flags"] = flags

    datasets = reg.get("datasets", {})
    crop_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    real_slugs = 0
    autolabel_slugs = 0
    downloaded_imgs = 0

    for slug, info in datasets.items():
        if not isinstance(info, dict):
            continue
        ann = info.get("annotation") or info.get("status", "")
        ann_h = annotation_type_human(ann)
        annotation_counts[ann_h] = annotation_counts.get(ann_h, 0) + 1
        src = slug_source(slug)
        source_counts[src] = source_counts.get(src, 0) + 1
        crops = slug_crop(slug, info)
        for c in crops:
            crop_counts[c] = crop_counts.get(c, 0) + 1

        # Real image count: walk local_path and count
        n_imgs = 0
        local = info.get("local_path")
        if local and os.path.isdir(local):
            for root, _, files in os.walk(local):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        n_imgs += 1
                # cap walk for huge slugs to keep state build fast
                if n_imgs > 100000:
                    break
        downloaded_imgs += n_imgs
        if "real" in ann_h.lower() or "human" in ann_h.lower():
            real_slugs += 1
        if "AI-labeled" in ann_h:
            autolabel_slugs += 1

        state["datasets"].append({
            "slug": slug,
            "source": src,
            "n_imgs": n_imgs,
            "annotation": ann,
            "annotation_h": ann_h,
            "crops": crops,
            "description": info.get("description") or "",
            "never_train": slug in NEVER_TRAIN,
            "class_names": info.get("class_names") or [],
            "flag": flags.get(slug, {}).get("flag"),
            "flag_reason": flags.get(slug, {}).get("reason", ""),
            "has_local_data": bool(info.get("local_path")
                                   and os.path.isdir(info.get("local_path") or "")),
        })

    state["datasets"].sort(key=lambda d: -d["n_imgs"])
    state["totals"] = {
        "slugs": len(datasets),
        "real_bbox_slugs": real_slugs,
        "autolabel_slugs": autolabel_slugs,
        "downloaded_imgs": downloaded_imgs,
    }
    state["crop_counts"] = crop_counts
    state["source_counts"] = source_counts
    state["annotation_counts"] = annotation_counts

    # Latest pycocotools eval
    import glob
    for p in sorted(glob.glob(str(PYCOCO_GLOB / "*pycoco*summary.json")),
                    key=os.path.getmtime):
        try:
            with open(p) as f:
                d = json.load(f)
            label = Path(p).stem.replace("_pycoco_summary", "").replace("v3_0_29_", "")
            state["pyco_history"].append({
                "label": label,
                "mAP50_95": float(d.get("mAP50_95", 0)),
                "mAP50": float(d.get("mAP50", 0)),
                "mtime": os.path.getmtime(p),
            })
        except Exception:
            pass

    # Rich per-result metadata (training context + per-class AP)
    state["result_metas"] = []
    for p in sorted((PYCOCO_GLOB).glob("*pycoco*meta.json"),
                    key=lambda x: x.stat().st_mtime):
        try:
            state["result_metas"].append(json.loads(p.read_text()))
        except Exception:
            pass

    # Per-Job-D run summaries
    if JOBD_DIR.is_dir():
        for p in sorted(JOBD_DIR.glob("*.json")):
            try:
                with open(p) as f:
                    state["jobd_runs"].append(json.load(f))
            except Exception:
                pass

    return state


def get_state() -> dict:
    now = time.time()
    if _state_cache["data"] is None or (now - _state_cache["ts"]) > STATE_TTL:
        log.info("[state] rebuilding from disk")
        _state_cache["data"] = _build_state()
        _state_cache["ts"] = now
    return _state_cache["data"]


# ---------- image rendering ----------

def parse_yolo_labels(label_path: Path):
    out = []
    try:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            out.append((cid, [cx, cy, bw, bh]))
    except Exception:
        pass
    return out


def find_label_for_image(img_path: Path, ds_root: Path) -> Optional[Path]:
    stem = img_path.stem
    if "images" in img_path.parts:
        cand = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if cand.exists():
            return cand
    same = img_path.with_suffix(".txt")
    if same.exists():
        return same
    for parent in img_path.parents:
        cand = parent / "labels" / (stem + ".txt")
        if cand.exists():
            return cand
        if parent == ds_root:
            break
    return None


def find_image_in_slug(slug: str, filename: str) -> Optional[tuple[Path, Path]]:
    """Return (img_path, slug_local_root) — searches the slug's local_path tree."""
    state = get_state()
    info = next((d for d in state["datasets"] if d["slug"] == slug), None)
    if info is None:
        return None
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        local = reg["datasets"][slug].get("local_path")
    except Exception:
        return None
    if not local or not os.path.isdir(local):
        return None
    local_p = Path(local)

    # Disallow path traversal
    if "/" in filename or ".." in filename or filename.startswith("."):
        return None

    # Try common subpaths first
    candidates = list(local_p.rglob(filename))
    # Limit search depth for huge slugs
    if not candidates:
        return None
    return candidates[0], local_p


def render_with_bbox(img_path: Path, label_path: Optional[Path],
                     out_path: Path, slug: str, max_width: int = 600,
                     class_names: list[str] = ()) -> bool:
    """Render image with bboxes drawn, save JPEG."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]

    if label_path is not None:
        # Color: real-bbox slug → green, autolabel → orange, never_train → red
        is_never_train = slug in NEVER_TRAIN
        if is_never_train:
            box_color = (0, 0, 220)  # red (BGR)
        else:
            # need annotation to decide green/orange
            try:
                with open(REGISTRY_PATH) as f:
                    reg = json.load(f)
                ann = reg["datasets"][slug].get("annotation") or ""
            except Exception:
                ann = ""
            if ann == "yolo_autolabel":
                box_color = (0, 140, 240)  # orange
            else:
                box_color = (0, 200, 0)    # green

        for cid, (cx, cy, bw, bh) in parse_yolo_labels(label_path):
            x1 = int(max(0, (cx - bw / 2) * w))
            y1 = int(max(0, (cy - bh / 2) * h))
            x2 = int(min(w, (cx + bw / 2) * w))
            y2 = int(min(h, (cy + bh / 2) * h))
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, max(2, w // 250))
            lbl_str = class_names[cid] if 0 <= cid < len(class_names) else f"cls{cid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, lbl_str, (x1, max(15, y1 - 6)), font,
                        max(0.4, w / 1200), box_color, max(1, w // 600),
                        cv2.LINE_AA)

    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return True


# ---------- endpoints ----------

@app.get("/healthz")
def healthz():
    return {"ok": True, "repo": str(REPO), "ts": time.time()}


@app.get("/api/db_status")
def api_db_status():
    """MongoDB migration Phase 1 probe. Reports whether tools.db has a live
    Mongo connection or is serving from the JSON fallback. Never raises."""
    try:
        from . import db as _db
        info = _db.ping()
        info["backend"] = "mongo" if info.get("available") else "json-fallback"
        # Cross-check: does get_registry actually return data via the live path?
        reg = _db.get_registry()
        info["registry_datasets"] = len(reg.get("datasets", {}))
        return info
    except Exception as e:
        return {"available": False, "backend": "json-fallback",
                "error": f"db module error: {e}"}


@app.get("/api/domains")
def api_domains():
    """v3.0.82 multi-domain: list dataset-collection-agent domains + per-domain
    slug counts. Each domain is one collection agent (weed is the first; future
    pest/crop-disease agents appear here with no schema change). Never raises."""
    try:
        from . import db as _db
        domains = _db.get_domains()
        out = []
        for d in domains:
            dom_id = d.get("_id")
            out.append({
                "domain": dom_id,
                "display_name": d.get("display_name", dom_id),
                "taxonomy": d.get("taxonomy"),
                "status": d.get("status"),
                "target_metric": d.get("target_metric"),
                "n_slugs": len(_db.list_slugs(domain=dom_id)),
            })
        return {"available": _db.available(), "n_domains": len(out),
                "domains": out}
    except Exception as e:
        return {"available": False, "n_domains": 0, "domains": [],
                "error": f"db module error: {e}"}


# ---------- flag API: user-driven REQ-3 feedback loop ----------

def _load_flags() -> dict:
    if FLAGS_PATH.exists():
        try:
            return json.loads(FLAGS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_flags(flags: dict) -> None:
    tmp = FLAGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(flags, indent=1))
    os.replace(tmp, FLAGS_PATH)


@app.get("/api/flags")
def api_flags():
    """Return the full flag state.
    {slug: {flag: "garbage"|"good"|"unsure", reason: str, ts: float}, ...}"""
    return _load_flags()


@app.post("/api/flag/{slug}")
def api_set_flag(slug: str, body: dict = Body(default=None)):
    """Mark a dataset slug. Persists to dataset_flags.json on cluster.
    mega_trainer.py reads this file at merge time and skips garbage slugs."""
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    if body is None:
        body = {}
    flag = (body.get("flag") or "").strip()
    reason = (body.get("reason") or "")[:500]
    if flag not in ("garbage", "good", "unsure", "clear"):
        raise HTTPException(400, "flag must be one of: garbage/good/unsure/clear")
    flags = _load_flags()
    if flag == "clear":
        flags.pop(slug, None)
        action = "cleared"
    else:
        flags[slug] = {
            "flag": flag,
            "reason": reason,
            "ts": time.time(),
            "ts_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        action = f"set to {flag}"
    _save_flags(flags)
    log.info(f"[flag] {slug} {action}: {reason[:80]}")
    # invalidate state cache so next /api/state reflects the flag
    _state_cache["data"] = None
    return {"ok": True, "slug": slug, "action": action,
            "current": flags.get(slug)}


@app.get("/", response_class=HTMLResponse)
def root():
    """v3.0.100 — Agent Launcher (clean entry page).

    Big "Weed Detection" agent card + a "+" to create new domain agents.
    Pure presentation; links to /agent/weed (mission control) and /console
    (classic advanced view). NO backend logic touched. English-only UI.

    v3.0.107: data-driven — built-in Weed card + any domains created via the
    "+" (db.create_domain → COLL_DOMAINS) render automatically."""
    _extra = ""
    try:
        import html as _htmlmod
        _esc = _htmlmod.escape
        from . import db as _db
        for _d in (_db.get_domains() or []):
            _did = str(_d.get("_id") or "")
            if not _did or _did == "weed":
                continue
            _nm = _esc(str(_d.get("display_name") or _did))
            _st = _esc(str(_d.get("status") or "created"))
            _extra += (
                f'<a class="agent" href="/agent/{_esc(_did)}">'
                f'<div class="ic">&#129516;</div><div class="nm">{_nm}</div>'
                f'<div class="ds">New collection agent. Status: {_st}. '
                f'Harvest pipeline not wired yet.</div>'
                f'<div class="badge" style="background:#2a2a16;color:#d9c95f;'
                f'border-color:#544a1d">&#9679; {_st}</div></a>')
    except Exception:
        _extra = ""
    html = '''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Launcher</title>
<style>
 *{box-sizing:border-box}
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;
   min-height:100vh;background:linear-gradient(160deg,#1b2433 0%,#0f1218 100%);color:#e7eaf0;
   display:flex;flex-direction:column;align-items:center;padding:52px 20px}
 .brand{font-size:12px;letter-spacing:2.5px;text-transform:uppercase;color:#7b8aa5;margin-bottom:8px}
 h1{font-size:30px;margin:0 0 8px;font-weight:700;text-align:center}
 .tag{color:#9aa7bd;font-size:15px;margin-bottom:42px;text-align:center;max-width:580px;line-height:1.55}
 .agents{display:grid;grid-template-columns:repeat(auto-fit,minmax(258px,300px));gap:22px;
   justify-content:center;width:100%;max-width:960px}
 .agent{background:linear-gradient(160deg,#243049,#1a2230);border:1px solid #2c3a52;border-radius:16px;
   padding:26px;text-decoration:none;color:inherit;transition:.15s;display:block}
 .agent:hover{transform:translateY(-3px);border-color:#3b82f6;box-shadow:0 14px 34px rgba(0,0,0,.45)}
 .agent .ic{font-size:42px;margin-bottom:14px}
 .agent .nm{font-size:20px;font-weight:700;margin-bottom:7px}
 .agent .ds{font-size:13.5px;color:#9aa7bd;line-height:1.5}
 .agent .badge{display:inline-block;margin-top:15px;font-size:11px;padding:3px 11px;border-radius:20px;
   background:#16351f;color:#5fd98a;border:1px solid #1f5132}
 .agent.add{border-style:dashed;display:flex;flex-direction:column;align-items:center;justify-content:center;
   text-align:center;color:#8b9bb5;cursor:pointer;background:transparent;min-height:210px}
 .agent.add .plus{font-size:50px;line-height:1;margin-bottom:10px;color:#5b6c8a}
 .agent.add:hover{color:#cdd6e6;border-color:#3b82f6}
 #createPanel{display:none;margin-top:28px;background:#1a2230;border:1px solid #2c3a52;border-radius:14px;
   padding:24px;width:100%;max-width:470px}
 #createPanel h3{margin:0 0 6px;font-size:17px}
 #createPanel p.h{margin:0 0 14px;font-size:12.5px;color:#7b8aa5}
 #createPanel label{display:block;font-size:12px;color:#9aa7bd;margin:13px 0 5px}
 #createPanel input,#createPanel select{width:100%;padding:10px 12px;border-radius:8px;border:1px solid #2c3a52;
   background:#11161f;color:#e7eaf0;font-size:14px}
 .row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 .btn{margin-top:18px;width:100%;padding:12px;border:0;border-radius:9px;background:#2563eb;color:#fff;
   font-size:14px;font-weight:600;cursor:pointer}
 .note{font-size:12px;color:#7b8aa5;margin-top:11px;text-align:center;line-height:1.5}
 .foot{margin-top:48px;color:#5b6c8a;font-size:12px;text-align:center}
 .foot a{color:#93a3bd}
</style></head><body>
 <div class="brand">Greater Robotics Lab</div>
 <h1>Research Projects</h1>
 <div class="tag">Self-driving agents that collect, human-review, filter and train on real-world
   datasets &mdash; compounding over weeks and months.</div>
 <div class="agents">
   <a class="agent" href="/agent/weed">
     <div class="ic">&#127806;</div>
     <div class="nm">Weed Detection</div>
     <div class="ds">Harvests weed &amp; crop imagery, human-reviews labels, and trains detection
       models toward field-ready accuracy.</div>
     <div class="badge">&#9679; Active</div>
   </a>
   __EXTRA_AGENTS__
   <div class="agent add" onclick="var p=document.getElementById('createPanel');p.style.display='block';p.scrollIntoView({behavior:'smooth'})">
     <div class="plus">+</div>
     <div class="nm">New Project</div>
     <div class="ds">Create a project for any research domain &mdash; upload data, add agents later.</div>
   </div>
 </div>
 <div id="createPanel">
   <h3>Create a new project</h3>
   <p class="h">A project is a research workspace (any field, any data type). Upload datasets and add agents (collect / filter / label / train) any time &mdash; or none at all.</p>
   <label>Project name</label>
   <input id="agName" placeholder="e.g. Crop Disease, Warehouse Robot, Coral Reef Survey">
   <label>Research field <span style="color:#94a3b8;font-weight:400">(free text, any domain)</span></label>
   <input id="agField" placeholder="e.g. agriculture, robotics, marine biology, materials science">
   <label>Data types this project will hold</label>
   <div style="display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 12px;font-size:13px">
     <label style="font-weight:400"><input type="checkbox" class="agMod" value="image" checked> Image</label>
     <label style="font-weight:400"><input type="checkbox" class="agMod" value="video"> Video</label>
     <label style="font-weight:400"><input type="checkbox" class="agMod" value="sensor"> Sensor (GPS/IMU/LiDAR)</label>
     <label style="font-weight:400"><input type="checkbox" class="agMod" value="pointcloud"> Point cloud</label>
     <label style="font-weight:400"><input type="checkbox" class="agMod" value="audio"> Audio</label>
     <label style="font-weight:400"><input type="checkbox" class="agMod" value="text"> Text</label>
   </div>
   <button class="btn" id="createBtn" onclick="createAgent()">Create project</button>
   <div class="note" id="createNote">Creates an empty research workspace. Upload datasets right away, and add agents (collect / filter / label / train) whenever you want &mdash; any number, any mix, or none.</div>
 </div>
 <div class="foot">Lab server &middot; MongoDB &middot; cluster GPU compute &nbsp;|&nbsp; <a href="/console">Advanced console &rarr;</a></div>
 <script>
  function createAgent(){
    var name=(document.getElementById('agName').value||'').trim();
    var note=document.getElementById('createNote'), btn=document.getElementById('createBtn');
    if(!name){note.textContent='Please enter a project name.';return;}
    var mods=Array.prototype.slice.call(document.querySelectorAll('.agMod:checked')).map(function(c){return c.value;});
    if(!mods.length){note.textContent='Pick at least one data type.';return;}
    btn.disabled=true;note.textContent='\\u23f3 Creating\\u2026';
    fetch('/api/agent/create',{method:'POST',credentials:'include',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:name,modality:mods,
        research_field:(document.getElementById('agField').value||'').trim()})})
     .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
     .then(function(res){
        if(res.ok&&res.j.ok){note.textContent='\\u2705 Created \"'+res.j.display_name+'\". Opening\\u2026';setTimeout(function(){location.href='/agent/'+res.j.domain;},800);}
        else{note.textContent='\\u274c '+((res.j&&res.j.detail)||'Create failed');btn.disabled=false;}
     })
     .catch(function(e){note.textContent='\\u274c Error: '+e;btn.disabled=false;});
  }
 </script>
</body></html>'''
    return HTMLResponse(html.replace("__EXTRA_AGENTS__", _extra))


@app.get("/agent/weed", response_class=HTMLResponse)
def agent_weed():
    """v3.0.100 — Mission Control for the Weed Detection agent.

    Clean overview: pipeline strip + the two agents (Collector / Trainer) +
    quick links to existing data / review / results pages + advanced console.
    Pure presentation linking to existing routes; NO backend logic touched.
    Live cluster status pulled from the existing /api/cluster_status. English."""
    return HTMLResponse('''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weed Detection &mdash; Mission Control</title>
<style>
 *{box-sizing:border-box}
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;
   background:#f2f4f8;color:#1a1a1d;padding:0 0 48px}
 .top{background:#fff;border-bottom:1px solid #e3e7ef;padding:14px 22px;display:flex;align-items:center;
   gap:14px;flex-wrap:wrap}
 .top a.bc{color:#64748b;text-decoration:none;font-size:13px}
 .top a.bc:hover{color:#2563eb}
 .top h1{font-size:18px;margin:0;font-weight:700}
 .dot{margin-left:auto;font-size:12.5px;color:#475569;display:flex;align-items:center;gap:7px}
 .dot .c{width:9px;height:9px;border-radius:50%;background:#94a3b8}
 .dot .c.run{background:#16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.18)}
 .wrap{max-width:1080px;margin:22px auto;padding:0 18px}
 .pipe{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}
 .pipe .st{flex:1;min-width:120px;background:#fff;border:1px solid #e3e7ef;border-radius:10px;padding:12px 14px;
   text-align:center;font-size:13px;color:#475569;position:relative}
 .pipe .st b{display:block;font-size:14px;color:#0f172a;margin-top:3px}
 .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:22px}
 .agent{background:#fff;border:1px solid #e3e7ef;border-radius:14px;padding:20px 22px}
 .agent .hd{display:flex;align-items:center;gap:12px;margin-bottom:6px}
 .agent .hd .ic{font-size:28px}
 .agent .hd .t{font-size:16px;font-weight:700}
 .agent .hd .t small{display:block;font-weight:500;color:#64748b;font-size:12px}
 .agent .stat{font-size:13px;color:#475569;margin:10px 0 4px}
 .agent .stat b{color:#0f172a}
 .acts{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}
 .acts a,.acts button{flex:1;text-align:center;text-decoration:none;font-size:13px;font-weight:600;padding:9px 12px;
   border-radius:8px;background:#2563eb;color:#fff;white-space:nowrap;border:0;cursor:pointer;font-family:inherit}
 .acts a.sec{flex:0 0 auto;background:#eef2ff;color:#2563eb}
 .acts button:disabled{opacity:.55;cursor:default}
 #toast{display:none;margin:18px 0 0;padding:12px 15px;border-radius:9px;background:#0f172a;color:#e7eaf0;
   font-size:13px;line-height:1.5;word-break:break-word}
 .quick{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:13px}
 .quick a{background:#fff;border:1px solid #e3e7ef;border-radius:12px;padding:16px;text-decoration:none;
   color:inherit;transition:.12s}
 .quick a:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,0,0,.07);border-color:#c7d2fe}
 .quick .ic{font-size:24px}
 .quick .nm{font-weight:600;margin:8px 0 3px;font-size:14.5px}
 .quick .ds{font-size:12.5px;color:#64748b;line-height:1.45}
 .sec-h{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:#94a3b8;margin:4px 0 11px}
</style></head><body>
 <div class="top">
   <a class="bc" href="/">&larr; Agents</a>
   <h1>&#127806; Weed Detection</h1>
   <div class="dot"><span class="c" id="cdot"></span><span id="ctxt">cluster: checking&hellip;</span></div>
 </div>
 <div class="wrap">
   <div class="pipe">
     <div class="st">1<b>Collect</b></div>
     <div class="st">2<b>Review &amp; Label</b></div>
     <div class="st">3<b>Filter</b></div>
     <div class="st">4<b>Train</b></div>
     <div class="st">5<b>Results</b></div>
   </div>

   <div class="sec-h">Agents</div>
   <div class="grid2">
     <div class="agent">
       <div class="hd"><span class="ic">&#129302;</span><span class="t">Agent 1 &middot; Collector<small>Brain harvester &mdash; finds &amp; pulls datasets</small></span></div>
       <div class="stat">Runs autonomous search + harvest on the cluster, then syncs results to this server.</div>
       <div class="acts"><button onclick="runAction(this,'harvest_full_round_e2e','Harvest','This launches a ~4h GPU harvest round on the cluster (find + pull new datasets, then auto-sync to this server). Continue?')">&#9654; Start harvest</button><a class="sec" href="/slugs">Datasets</a><a class="sec" href="/console">Configure</a></div>
     </div>
     <div class="agent">
       <div class="hd"><span class="ic">&#128640;</span><span class="t">Agent 2 &middot; Trainer<small>Trains detection on reviewed data</small></span></div>
       <div class="stat">Trains YOLO on human-verified data and evaluates against the held-out gold set.</div>
       <div class="acts"><button onclick="runAction(this,'clean_train_d','Training','This launches a GPU training job on the cluster (clean-subset training, evaluated on the held-out gold set). Continue?')">&#128640; Train</button><a class="sec" href="/rounds">Results</a><a class="sec" href="/console">Configure</a></div>
     </div>
   </div>

   <div id="toast"></div>

   <div class="sec-h">Upload a dataset (.zip)</div>
   <div style="background:#fff;border:1px dashed #cbd5e1;border-radius:12px;padding:16px">
     <div style="font-size:13px;color:#475569;margin-bottom:10px">Add data manually: a <b>.zip</b> of images (optionally with YOLO <code>labels/</code> + <code>data.yaml</code>) registers as a dataset for this agent and appears under Datasets. Future: automatic upload + open community contributions.</div>
     <input id="wds-name" type="text" placeholder="Dataset name (e.g. field-photos-2026-06)" style="width:100%;padding:9px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px;margin-bottom:8px">
     <input id="wds-file" type="file" accept=".zip,application/zip" style="font-size:13px;margin-bottom:10px;display:block">
     <button id="wds-up" onclick="uploadWeedDataset()" style="border:0;cursor:pointer;background:#2563eb;color:#fff;font-weight:600;font-size:13px;padding:10px 16px;border-radius:9px">&#11014; Upload dataset</button>
     <span id="wds-toast" style="margin-left:12px;font-size:13px"></span>
   </div>
   <div id="wul-wrap" style="margin-top:14px"></div>

   <div class="sec-h">Roboflow push</div>
   <div style="background:#fff;border:1px solid #e3e7ef;border-radius:12px;padding:16px">
     <div style="font-size:13px;color:#475569;margin-bottom:10px">The agent uploads collected datasets to Roboflow for human labeling. Set the <b>upper limit</b> — the most images the agent pushes per dataset (already-labeled images are pushed with their annotations too).</div>
     <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
       <label style="font-size:13px;color:#334">Agent push cap (images / dataset):</label>
       <input id="wcap" type="number" min="1" max="2000" style="width:100px;padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px">
       <button id="wcap-save" onclick="saveWeedCap()" style="border:0;cursor:pointer;background:#0e7c66;color:#fff;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">Save</button>
       <span id="wcap-msg" style="font-size:13px;color:#475569"></span>
     </div>
     <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">
       <a href="/labeling" style="text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">Open Labeling Console</a>
       <a href="https://app.roboflow.com/a-test-of-will" target="_blank" style="text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">Adjust labels in Roboflow &#8599;</a>
     </div>
   </div>

   <div class="sec-h">Workspace</div>
   <div class="quick">
     <a href="/classes"><div class="ic">&#127793;</div><div class="nm">Browse Data</div><div class="ds">All collected species &amp; classes, with thumbnails.</div></a>
     <a href="/slugs"><div class="ic">&#128451;</div><div class="nm">Datasets</div><div class="ds">Every harvested dataset, sample previews.</div></a>
     <a href="/labeling"><div class="ic">&#9989;</div><div class="nm">Review &amp; Label</div><div class="ds">Human-in-the-loop labeling queue.</div></a>
     <a href="/rounds"><div class="ic">&#128200;</div><div class="nm">Rounds &amp; Results</div><div class="ds">Harvest rounds and training metrics.</div></a>
     <a href="/users"><div class="ic">&#128100;</div><div class="nm">Users</div><div class="ds">Who signed in &amp; uploaded what.</div></a>
     <a href="/models"><div class="ic">&#129504;</div><div class="nm">Models</div><div class="ds">Pick the LLM/VLM per agent role.</div></a>
     <a href="/manual"><div class="ic">&#128214;</div><div class="nm">Manual</div><div class="ds">Architecture &amp; full pipeline docs.</div></a>
     <a href="/console"><div class="ic">&#9881;</div><div class="nm">Advanced Console</div><div class="ds">All raw controls &amp; cluster ops.</div></a>
   </div>
 </div>
 <script>
  fetch('/api/cluster_status').then(r=>r.json()).then(d=>{
    var jobs=(d&&(d.running!=null?d.running:(d.n_running!=null?d.n_running:(Array.isArray(d.jobs)?d.jobs.length:null))));
    var dot=document.getElementById('cdot'),txt=document.getElementById('ctxt');
    if(jobs&&jobs>0){dot.className='c run';txt.textContent='cluster: '+jobs+' job'+(jobs>1?'s':'')+' running';}
    else{dot.className='c';txt.textContent='cluster: idle';}
  }).catch(function(){document.getElementById('ctxt').textContent='cluster: status unavailable';});

  // v3.0.133 (RBAC): cluster (GPU) jobs are admin/granted-only. For members,
  // disable the Harvest/Train buttons and explain — they can still upload/browse.
  fetch('/api/me',{credentials:'include'}).then(r=>r.json()).then(function(me){
    if(me&&me.ok&&!me.can_use_cluster){
      document.querySelectorAll('button').forEach(function(b){
        var oc=b.getAttribute('onclick')||'';
        if(oc.indexOf('runAction')>=0){
          b.disabled=true;b.title='Cluster jobs are limited to administrators';
          b.style.opacity='0.5';b.style.cursor='not-allowed';
        }
      });
      var t=document.getElementById('toast');
      if(t){t.style.display='block';t.style.background='#fef9c3';t.style.color='#854d0e';
        var reqd=me.cluster_requested;
        t.innerHTML='\\u2139 Harvest \\u0026 Train run on the shared GPU cluster and are limited to administrators. You can still upload datasets, browse, and review. '
          +(reqd?'<b>Cluster access requested \\u2014 waiting for an admin.</b>'
                :'<button id="reqcl" style="border:0;cursor:pointer;background:#0e7c66;color:#fff;font-weight:600;font-size:12px;padding:6px 12px;border-radius:7px;margin-left:6px">Request cluster access</button>');
        var rb=document.getElementById('reqcl');
        if(rb)rb.onclick=function(){rb.disabled=true;rb.textContent='\\u2026';
          fetch('/api/cluster/request',{method:'POST',credentials:'include'}).then(function(r){return r.json();})
           .then(function(d){rb.outerHTML=(d&&d.ok)?'<b>'+(d.already?'You already have access.':'Request sent to admins.')+'</b>':'\\u274c failed';})
           .catch(function(e){rb.disabled=false;rb.textContent='Request cluster access';});};
      }
    }
  }).catch(function(){});

  function runAction(btn,name,label,confirmMsg){
    if(!confirm(confirmMsg))return;
    var t=document.getElementById('toast');
    t.style.display='block';t.textContent='\\u23f3 Submitting '+label+'\\u2026';
    btn.disabled=true;
    fetch('/api/cluster_action/'+name,{method:'POST',credentials:'include'})
     .then(function(r){return r.json();})
     .then(function(d){
        if(d&&d.ok){t.textContent='\\u2705 '+label+' submitted to cluster. '+(d.msg||d.stdout||'');}
        else{t.textContent='\\u274c '+label+' failed: '+((d&&(d.msg||d.stderr))||'unknown error');}
     })
     .catch(function(e){t.textContent='\\u274c '+label+' error: '+e;})
     .finally(function(){btn.disabled=false;});
  }
  async function uploadWeedDataset(){
   var nameEl=document.getElementById('wds-name'),fileEl=document.getElementById('wds-file'),
       t=document.getElementById('wds-toast'),btn=document.getElementById('wds-up');
   var name=(nameEl.value||'').trim();
   if(!name){t.textContent='\\u26a0 enter a dataset name';return;}
   if(!fileEl.files||!fileEl.files[0]){t.textContent='\\u26a0 choose a .zip file';return;}
   var f=fileEl.files[0];
   if(!/\\.zip$/i.test(f.name)){t.textContent='\\u26a0 must be a .zip file';return;}
   btn.disabled=true;t.textContent='\\u23f3 uploading '+(f.size/1048576).toFixed(1)+' MB\\u2026';
   try{
    var url='/api/dataset/upload?domain=weed&name='+encodeURIComponent(name);
    var r=await fetch(url,{method:'POST',credentials:'include',headers:{'Content-Type':'application/zip'},body:f});
    var d=await r.json();
    if(r.ok&&d&&d.ok){t.innerHTML='\\u2705 '+d.images+' images registered as <b>'+d.slug+'</b> \\u00b7 <a href="'+d.gallery_url+'" target="_blank">view</a>';nameEl.value='';fileEl.value='';loadWeedUploads();}
    else{t.textContent='\\u274c '+((d&&(d.detail||d.msg))||('HTTP '+r.status));}
   }catch(e){t.textContent='\\u274c '+e;}finally{btn.disabled=false;}
  }
  async function loadWeedUploads(){
   var w=document.getElementById('wul-wrap');if(!w)return;
   try{
    var d=await (await fetch('/api/dataset/uploads?domain=weed',{credentials:'include'})).json();
    var rows=(d&&d.uploads)||[];
    if(!rows.length){w.innerHTML='';return;}
    var h='<div style="font-size:12px;color:#94a3b8;margin-bottom:6px">Your uploads ('+rows.length+')</div><div style="border:1px solid #e3e7ef;border-radius:10px;overflow:hidden;background:#fff">';
    rows.forEach(function(u,i){
     h+='<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 12px;'+(i?'border-top:1px solid #eef1f6;':'')+'font-size:13px">'
       +'<span><a href="/gallery/'+encodeURIComponent(u.slug)+'" target="_blank" style="color:#2563eb;text-decoration:none;font-weight:600">'+(u.name||u.slug)+'</a> <span style="color:#94a3b8">\\u00b7 '+u.images+' imgs \\u00b7 by '+(u.uploaded_by||'?')+'</span></span>'
       +'<button onclick="deleteWeedUpload(\\''+u.slug+'\\',this)" style="border:1px solid #fecaca;background:#fff;color:#dc2626;font-size:12px;padding:5px 10px;border-radius:7px;cursor:pointer">Delete</button></div>';
    });
    h+='</div>';w.innerHTML=h;
   }catch(e){w.innerHTML='';}
  }
  async function deleteWeedUpload(slug,btn){
   if(!confirm('Delete uploaded dataset '+slug+'? This removes its images and registration.'))return;
   btn.disabled=true;btn.textContent='\\u2026';
   try{
    var d=await (await fetch('/api/dataset/delete',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({slug:slug})})).json();
    if(d&&d.ok){loadWeedUploads();}else{btn.disabled=false;btn.textContent='Delete';alert((d&&(d.detail||d.msg))||'delete failed');}
   }catch(e){btn.disabled=false;btn.textContent='Delete';alert(''+e);}
  }
  loadWeedUploads();
  async function loadWeedCap(){
   try{var d=await (await fetch('/api/domain/push_cap?domain=weed',{credentials:'include'})).json();
    if(d&&d.ok){document.getElementById('wcap').value=d.cap;}}catch(e){}
  }
  async function saveWeedCap(){
   var el=document.getElementById('wcap'),m=document.getElementById('wcap-msg'),b=document.getElementById('wcap-save');
   var cap=parseInt(el.value,10);if(!cap||cap<1){m.textContent='\\u26a0 enter a number';return;}
   b.disabled=true;m.textContent='\\u23f3 saving\\u2026';
   try{var d=await (await fetch('/api/domain/push_cap',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'weed',cap:cap})})).json();
    if(d&&d.ok){el.value=d.cap;m.textContent='\\u2705 saved (cap '+d.cap+')';}else{m.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}
   }catch(e){m.textContent='\\u274c '+e;}finally{b.disabled=false;}
  }
  loadWeedCap();
 </script>
</body></html>''')


@app.post("/api/agent/create")
async def api_agent_create(request: Request):
    """v3.0.107: register a new collection-agent domain (Prof multi-domain
    design). Persists to COLL_DOMAINS via db.create_domain. Additive — does NOT
    touch the weed agent and does NOT yet wire the new domain's harvest pipeline."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (str(body.get("name") or "")).strip()
    if not name:
        raise HTTPException(400, "name required")
    domain_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    if not domain_id:
        raise HTTPException(400, "name must contain letters or digits")
    try:
        queries = [q.strip() for q in str(body.get("queries") or "").split(",") if q.strip()]
        n = int(body.get("n_subagents") or 2)
    except Exception:
        queries, n = [], 2
    # v3.0.137: generalized agent fields (task / modality / model)
    task = str(body.get("task") or body.get("type") or "detection").strip().lower()
    model = str(body.get("model") or "auto").strip()
    mod_in = body.get("modality")
    if isinstance(mod_in, str):
        mod_in = [m.strip() for m in mod_in.split(",") if m.strip()]
    modality = [str(m).strip().lower() for m in (mod_in or ["image"])]
    research_field = str(body.get("research_field") or "").strip()[:80]
    from . import db as _db
    actor = _actor_from_request(request)
    res = _db.create_domain(domain_id, name, harvest_queries=queries, n_subagents=n,
                            task=task, modality=modality, model=model, owner=actor,
                            research_field=research_field)
    if res == "exists":
        raise HTTPException(409, f"agent '{domain_id}' already exists")
    if res is None:
        raise HTTPException(503, "database unavailable (Mongo down) — cannot create agent")
    try:
        _db.upsert_user(actor, auth_provider="basic")
    except Exception:
        pass
    return {"ok": True, "domain": domain_id, "display_name": name,
            "task": res.get("task"), "modality": res.get("modality"),
            "model": res.get("model")}


def _can_manage_agent(actor: str, dom: dict) -> bool:
    """Owner of the agent, or any admin, may edit/delete it. The flagship 'weed'
    agent is admin-only (never owner-deletable)."""
    if _is_admin(actor):
        return True
    if not dom or dom.get("_id") == "weed":
        return False
    return bool(dom.get("owner") and dom.get("owner") == actor)


@app.post("/api/agent/delete")
async def api_agent_delete(request: Request):
    """Delete an agent (domain). Owner or admin only; 'weed' is protected."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain_id = re.sub(r"[^a-z0-9_]+", "", str(body.get("domain") or "").lower())[:40]
    if not domain_id:
        raise HTTPException(400, "domain required")
    if domain_id == "weed":
        raise HTTPException(403, "the weed agent cannot be deleted")
    from . import db as _db
    dom = _db.get_domain(domain_id)
    if not dom:
        raise HTTPException(404, f"no agent '{domain_id}'")
    actor = _actor_from_request(request)
    if not _can_manage_agent(actor, dom):
        raise HTTPException(403, "only the agent's owner or an admin can delete it")
    # v3.0.145: COMPREHENSIVE delete — cascade the agent's own datasets (files +
    # registry + Mongo + manual_uploads). The shared weed pool (no domain tag) is
    # never matched, so this only removes THIS agent's data.
    purged = []
    for ds in _datasets_for_domain(domain_id):
        purged.append(_purge_dataset(ds, actor=actor))
    ok = _db.delete_domain(domain_id, actor=actor)
    log.info(f"[agent] deleted {domain_id} (+{len(purged)} datasets) by {actor}")
    return JSONResponse({"ok": bool(ok), "domain": domain_id,
                         "datasets_removed": len(purged)})


@app.post("/api/agent/update")
async def api_agent_update(request: Request):
    """Edit an agent's config (display_name / task / modality / model / queries).
    Owner or admin only."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain_id = re.sub(r"[^a-z0-9_]+", "", str(body.get("domain") or "").lower())[:40]
    if not domain_id:
        raise HTTPException(400, "domain required")
    from . import db as _db
    dom = _db.get_domain(domain_id)
    if not dom:
        raise HTTPException(404, f"no agent '{domain_id}'")
    actor = _actor_from_request(request)
    if not _can_manage_agent(actor, dom):
        raise HTTPException(403, "only the agent's owner or an admin can edit it")
    fields = {}
    if "display_name" in body:
        fields["display_name"] = str(body["display_name"]).strip()[:80]
    if "task" in body:
        fields["task"] = str(body["task"]).strip().lower()
    if "model" in body:
        fields["model"] = str(body["model"]).strip()[:40]
    if "queries" in body:
        fields["harvest_queries"] = [q.strip() for q in str(body["queries"]).split(",") if q.strip()]
    if "modality" in body and isinstance(body["modality"], list):
        fields["modality"] = [str(m).strip().lower() for m in body["modality"]]
    updated = _db.update_domain(domain_id, fields, actor=actor)
    if updated is None:
        raise HTTPException(503, "database unavailable")
    return JSONResponse({"ok": True, "domain": domain_id})


# v3.0.146 — agents are COMPONENTS inside a project (freely composed, 0..N).
_AGENT_TYPES = {
    "collector": "Collector — autonomously finds & pulls datasets",
    "filter": "Filter / QC — quality-scores & prunes data",
    "labeler": "Labeler — auto / assisted annotation",
    "trainer": "Trainer — trains models (supervised / self-sup / RL / hybrid)",
    "evaluator": "Evaluator — benchmarks & reports metrics",
    "custom": "Custom — a component you define later",
}


@app.get("/api/agent_types")
def api_agent_types():
    return JSONResponse({"ok": True, "types": _AGENT_TYPES})


@app.post("/api/project/agent/add")
async def api_project_agent_add(request: Request):
    """Add an agent component to a project (owner or admin)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain_id = re.sub(r"[^a-z0-9_]+", "", str(body.get("project") or body.get("domain") or "").lower())[:40]
    atype = str(body.get("type") or "").strip().lower()
    name = str(body.get("name") or "").strip()[:60]
    if not domain_id:
        raise HTTPException(400, "project required")
    if atype not in _AGENT_TYPES:
        raise HTTPException(400, f"type must be one of {list(_AGENT_TYPES)}")
    from . import db as _db
    dom = _db.get_domain(domain_id)
    if not dom:
        raise HTTPException(404, f"no project '{domain_id}'")
    actor = _actor_from_request(request)
    if not _can_manage_agent(actor, dom):
        raise HTTPException(403, "only the project's owner or an admin can add agents")
    a = _db.add_project_agent(domain_id, atype, name or atype, actor=actor)
    if a is None:
        raise HTTPException(503, "database unavailable")
    return JSONResponse({"ok": True, "agent": a})


@app.post("/api/project/agent/remove")
async def api_project_agent_remove(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain_id = re.sub(r"[^a-z0-9_]+", "", str(body.get("project") or body.get("domain") or "").lower())[:40]
    agent_id = re.sub(r"[^a-f0-9]+", "", str(body.get("agent_id") or ""))[:16]
    if not domain_id or not agent_id:
        raise HTTPException(400, "project + agent_id required")
    from . import db as _db
    dom = _db.get_domain(domain_id)
    if not dom:
        raise HTTPException(404, f"no project '{domain_id}'")
    actor = _actor_from_request(request)
    if not _can_manage_agent(actor, dom):
        raise HTTPException(403, "only the project's owner or an admin can remove agents")
    ok = _db.remove_project_agent(domain_id, agent_id, actor=actor)
    return JSONResponse({"ok": bool(ok), "agent_id": agent_id})


# ===========================================================================
# v3.0.125 (Prof Zhang platform expansion, Z1) — MANUAL DATASET UPLOAD.
# Every domain/dataset gets a manual ZIP-upload interface (future: auto-upload
# + open to the community). The .zip is POSTed as the raw request body (no
# python-multipart dependency); query params carry domain + name.
#
# Durability note: the lab dataset_registry.json is a sync-DOWN mirror of the
# cluster (overwritten ≤30min). So a manual upload is recorded in BOTH the live
# registry (immediate visibility) AND results/framework/manual_uploads.json
# (lab-local, never synced); deploy/fix_local_paths.py re-injects the latter
# into the registry after every cluster pull so uploads survive.
# ===========================================================================
_UPLOAD_DIR = REPO / "uploads"
_MANUAL_UPLOADS_FILE = REPO / "results" / "framework" / "manual_uploads.json"
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB
_MAX_UPLOAD_FILES = 60000
_UPLOAD_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# v3.0.137: per-modality accepted file extensions so non-image agents (video /
# robot sensor / point cloud) can upload too — "nothing gets rejected".
_MODALITY_EXT = {
    "image": _UPLOAD_IMG_EXT,
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"},
    "sensor": {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".npy", ".npz",
               ".bag", ".mcap", ".txt", ".log", ".gpx", ".nmea"},
    "pointcloud": {".pcd", ".ply", ".las", ".laz", ".bin", ".npy"},
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a"},
    "text": {".txt", ".json", ".jsonl", ".csv", ".md"},
}
# sidecars always allowed alongside any modality (labels / metadata / config)
_UPLOAD_SIDECAR_EXT = {".txt", ".json", ".xml", ".yaml", ".yml", ".csv"}


def _actor_from_request(request) -> str:
    """Best-effort uploader identity: the signed Google session user if present,
    else an X-User header, the Basic-auth user, or 'admin'."""
    try:
        su = _session_user(request)
        if su and su.get("uid"):
            return str(su["uid"])[:80]
    except Exception:
        pass
    try:
        ak = _check_api_key(request)
        if ak:
            return ("key:" + str(ak.get("label") or "api"))[:80]
    except Exception:
        pass
    try:
        xu = (request.headers.get("x-user") or "").strip()
        if xu:
            return xu[:80]
        h = request.headers.get("authorization", "")
        if h.startswith("Basic "):
            u = _base64.b64decode(h[6:]).decode("utf-8").partition(":")[0]
            # the shared admin Basic login (user '1') maps to 'admin'
            if u and u != _AUTH_USER:
                return ("user:" + u)[:80]
    except Exception:
        pass
    return "admin"


# ---- v3.0.133 (RBAC) — admin / cluster-access checks ----------------------- #
# Admins: the Basic-auth operator ("admin"), any email seeded in ~/.dash_admins
# (bootstrap, can't be locked out), or any user whose DB role is "admin".
# Cluster (GPU) jobs require admin OR a per-user can_use_cluster grant.
_ADMIN_FILE = os.path.expanduser("~/.dash_admins")
_admin_cache = {"mtime": -1.0, "set": set()}


def _admin_emails() -> set:
    try:
        mt = os.path.getmtime(_ADMIN_FILE)
    except Exception:
        return _admin_cache["set"]
    if mt == _admin_cache["mtime"]:
        return _admin_cache["set"]
    s = set()
    try:
        for line in open(_ADMIN_FILE):
            e = line.strip().lower()
            if e and not e.startswith("#"):
                s.add(e)
    except Exception:
        pass
    _admin_cache["set"] = s
    _admin_cache["mtime"] = mt
    return s


def _is_admin(actor: str) -> bool:
    if not actor:
        return False
    if actor == "admin":          # shared Basic-auth operator
        return True
    if actor.lower() in _admin_emails():
        return True
    try:
        from . import db as _dbu
        u = _dbu.get_user(actor)
        if u and u.get("role") == "admin":
            return True
    except Exception:
        pass
    return False


def _can_use_cluster(actor: str) -> bool:
    if _is_admin(actor):
        return True
    try:
        from . import db as _dbu
        u = _dbu.get_user(actor)
        return bool(u and u.get("can_use_cluster"))
    except Exception:
        return False


# ---- v3.0.134 — members can request cluster access; admins see + grant ------ #
_CLUSTER_REQ_FILE = REPO / "results" / "framework" / "cluster_requests.json"


def _read_cluster_requests() -> dict:
    try:
        if _CLUSTER_REQ_FILE.is_file():
            return json.load(open(_CLUSTER_REQ_FILE)) or {}
    except Exception:
        pass
    return {}


def _write_cluster_requests(d: dict) -> None:
    _CLUSTER_REQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(_CLUSTER_REQ_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, _CLUSTER_REQ_FILE)


@app.get("/api/me")
def api_me(request: Request):
    actor = _actor_from_request(request)
    reqs = _read_cluster_requests()
    is_admin = _is_admin(actor)
    return JSONResponse({"ok": True, "user": actor,
                         "is_admin": is_admin,
                         "can_use_cluster": _can_use_cluster(actor),
                         "cluster_requested": actor in reqs,
                         # admins see how many access requests are pending (red dot)
                         "pending_requests": (len(reqs) if is_admin else 0)})


@app.post("/api/cluster/request")
async def api_cluster_request(request: Request):
    """A signed-in member asks an admin for cluster (GPU) access."""
    actor = _actor_from_request(request)
    if _can_use_cluster(actor):
        return JSONResponse({"ok": True, "already": True,
                             "msg": "You already have cluster access."})
    try:   # ensure the requester is in the users table so admins see them in /users
        from . import db as _dbu
        _dbu.upsert_user(actor, auth_provider="basic")
    except Exception:
        pass
    reqs = _read_cluster_requests()
    reqs[actor] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        _write_cluster_requests(reqs)
    except Exception as e:
        raise HTTPException(500, f"could not record request: {e}")
    log.info(f"[cluster] access requested by {actor}")
    return JSONResponse({"ok": True, "msg": "Request sent to administrators."})


# ===========================================================================
# v3.0.138 — Model registry: pick which LLM/VLM each agent role uses (local
# Ollama or DeepSeek/GLM/OpenAI/Anthropic API or a cluster vLLM endpoint).
# Flexible switching done right (see llm_providers.py + docs/PLATFORM_ROADMAP).
# ===========================================================================
_MODEL_CONFIG_FILE = REPO / "results" / "framework" / "model_config.json"
_MODEL_ROLES = {
    "brain": "Agent reasoning / harvest decisions",
    "curation": "Dataset quality judging",
    "labeling_vlm": "VLM captioning / label assist",
}
_DEFAULT_MODELS = {"brain": "ollama:gemma2", "curation": "ollama:gemma2",
                   "labeling_vlm": "ollama:llava"}
_SUGGESTED_MODELS = [
    "ollama:gemma2:9b", "ollama:qwen2.5:7b", "ollama:llava",
    "deepseek:deepseek-chat", "deepseek:deepseek-reasoner",
    "glm:glm-4-flash", "glm:glm-4.6",
    "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-20241022",
    "vllm@http://NODE:8000/v1:deepseek-ai/DeepSeek-V3",
]


def _read_model_config() -> dict:
    cfg = {"roles": dict(_DEFAULT_MODELS)}
    try:
        if _MODEL_CONFIG_FILE.is_file():
            saved = json.load(open(_MODEL_CONFIG_FILE)) or {}
            cfg["roles"].update(saved.get("roles") or {})
    except Exception:
        pass
    return cfg


def _write_model_config(cfg: dict) -> None:
    _MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(_MODEL_CONFIG_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, _MODEL_CONFIG_FILE)


@app.get("/api/models")
def api_models():
    try:
        from . import llm_providers as _llm
        providers = _llm.provider_status()
    except Exception as e:
        providers = {"error": str(e)}
    cfg = _read_model_config()
    roles = {r: {"model": cfg["roles"].get(r, _DEFAULT_MODELS.get(r, "")),
                 "desc": _MODEL_ROLES[r]} for r in _MODEL_ROLES}
    return JSONResponse({"ok": True, "roles": roles, "providers": providers,
                         "suggested": _SUGGESTED_MODELS})


@app.post("/api/models/role")
async def api_set_model_role(request: Request):
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    role = str(body.get("role") or "")
    model = str(body.get("model") or "").strip()
    if role not in _MODEL_ROLES:
        raise HTTPException(400, f"role must be one of {list(_MODEL_ROLES)}")
    if not model or ":" not in model:
        raise HTTPException(400, "model must look like provider:name")
    cfg = _read_model_config()
    cfg["roles"][role] = model
    _write_model_config(cfg)
    log.info(f"[models] {role} -> {model} by {_actor_from_request(request)}")
    return JSONResponse({"ok": True, "role": role, "model": model})


@app.post("/api/models/test")
async def api_test_model(request: Request):
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = str(body.get("model") or "").strip()
    prompt = str(body.get("prompt") or "Reply with exactly: OK")
    if not model or ":" not in model:
        raise HTTPException(400, "model must look like provider:name")
    try:
        from . import llm_providers as _llm
        res = _llm.chat(model, prompt, max_tokens=64, timeout=45)
    except Exception as e:
        res = {"ok": False, "error": str(e), "model": model}
    return JSONResponse(res)


# ===========================================================================
# v3.0.142 (roadmap #3) — generic TRAINING submission. Stage an uploaded dataset
# lab→cluster, then submit the whitelisted generic Ultralytics template. On-demand
# (the job runs only while queued), same pattern as harvest. Cluster-gated.
# ===========================================================================
def _stage_dataset_to_cluster(slug: str) -> dict:
    """rsync an uploaded dataset (lab uploads/<slug>) to the cluster shared FS so a
    GPU job can read it. Returns {ok, dest, error}."""
    import subprocess as _sp
    local = _UPLOAD_DIR / slug
    if not local.is_dir():
        return {"ok": False, "error": f"no local dataset at uploads/{slug}"}
    if not _CLUSTER_SSH:
        return {"ok": False, "error": "cluster not configured"}
    rel = f"uploads/{slug}"
    _slurm(["mkdir", "-p", rel], timeout=30)   # login node, shared FS
    data_host = os.environ.get("CLUSTER_DATA_SSH", "byler@data.bridges2.psc.edu")
    rsh = "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
    dest = f"{data_host}:{_CLUSTER_REPO}/{rel}/"
    try:
        r = _sp.run(["rsync", "-az", "-e", rsh, str(local) + "/", dest],
                    env=os.environ.copy(), capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return {"ok": False, "error": "rsync: " + (r.stderr or "")[-200:]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "dest": f"{_CLUSTER_REPO}/{rel}"}


@app.post("/api/train/submit")
async def api_train_submit(request: Request):
    actor = _actor_from_request(request)
    if not _can_use_cluster(actor):
        raise HTTPException(403, "Cluster (GPU) jobs are restricted to admins / granted users.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain = _norm_domain(body.get("domain") or "weed")
    slug = str(body.get("slug") or "").strip()
    if not re.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400, "bad/missing dataset slug")
    from . import db as _db
    dd = _db.get_domain(domain) or {}
    task = str(body.get("task") or dd.get("task") or "detection")
    ultra = {"detection": "detect", "segmentation": "segment",
             "classification": "classify"}.get(task, task)
    if ultra not in ("detect", "segment", "classify"):
        raise HTTPException(400, "task must be detection / classification / segmentation")
    model = str(body.get("model") or dd.get("model") or "auto")
    if not model.endswith(".pt"):
        model = "auto"   # let the template pick the right yolo11n-* per task
    try:
        epochs = max(1, min(int(body.get("epochs") or 20), 300))
    except (TypeError, ValueError):
        epochs = 20
    # 1) stage the dataset to the cluster
    st = _stage_dataset_to_cluster(slug)
    if not st.get("ok"):
        raise HTTPException(502, "data staging to cluster failed: " + st.get("error", ""))
    base = st["dest"]
    data_path = base + "/images" if ultra == "classify" else base + "/data.yaml"
    # 2) make sure the whitelisted template is on the cluster outer root
    _slurm(["bash", "-lc",
            "git fetch origin >/dev/null 2>&1; "
            "git checkout origin/main -- weed_llm_benchmark/run_train_generic.sh 2>/dev/null; "
            "cp -f weed_llm_benchmark/run_train_generic.sh run_train_generic.sh 2>/dev/null; true"],
           timeout=60)
    # 3) submit
    jobtag = "j" + time.strftime("%m%d%H%M%S")
    export = ",".join(["ALL", f"TRAIN_TASK={ultra}", f"TRAIN_MODEL={model}",
                       f"TRAIN_EPOCHS={epochs}", f"TRAIN_DATA={data_path}",
                       f"TRAIN_DOMAIN={domain}", f"TRAIN_JOBTAG={jobtag}"])
    r = _slurm(["sbatch", f"--export={export}", "run_train_generic.sh"], timeout=25)
    ok = bool(r["ok"]) and "Submitted batch job" in (r.get("stdout") or "")
    result = {"ok": ok, "domain": domain, "task": ultra, "model": model,
              "epochs": epochs, "data": data_path, "jobtag": jobtag,
              "msg": (r.get("stdout") or r.get("stderr") or "").strip()}
    _log_action("train_generic", result)
    return JSONResponse(result)


# ===========================================================================
# v3.0.143 — self-hosted model GATEWAY (on-demand). An authenticated caller (our
# own API key / session) submits a prompt; the lab server queues a cluster job
# that runs OUR model and writes the answer to a file the server polls. No paid
# API, no persistent server — the user's exact architecture.
# ===========================================================================
@app.post("/api/llm/infer")
async def api_llm_infer(request: Request):
    """Submit an on-demand inference job. Returns a jobtag to poll."""
    actor = _actor_from_request(request)   # any authenticated identity (incl. api key)
    try:
        body = await request.json()
    except Exception:
        body = {}
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt required")
    if len(prompt) > 8000:
        raise HTTPException(400, "prompt too long (max 8000 chars)")
    model = re.sub(r"[^A-Za-z0-9_.:-]", "", str(body.get("model") or "gemma4"))[:60] or "gemma4"
    import base64 as _b64
    pb64 = _b64.b64encode(prompt.encode()).decode()
    jobtag = "i" + time.strftime("%m%d%H%M%S") + _secrets.token_hex(2)
    # write the prompt to the cluster + ensure the job script is present
    stage = ("mkdir -p results/framework/llm_infer; "
             f"echo {pb64} | base64 -d > results/framework/llm_infer/{jobtag}.prompt; "
             "git fetch origin >/dev/null 2>&1; "
             "git checkout origin/main -- weed_llm_benchmark/run_llm_infer.sh 2>/dev/null; "
             "cp -f weed_llm_benchmark/run_llm_infer.sh run_llm_infer.sh 2>/dev/null; true")
    _slurm(["bash", "-lc", stage], timeout=60)
    export = ",".join(["ALL", f"LLM_MODEL={model}", f"LLM_JOBTAG={jobtag}"])
    r = _slurm(["sbatch", f"--export={export}", "run_llm_infer.sh"], timeout=25)
    ok = bool(r["ok"]) and "Submitted batch job" in (r.get("stdout") or "")
    jobid = (r.get("stdout") or "").split()[-1] if ok else None
    log.info(f"[llm_infer] {jobtag} model={model} by {actor} ok={ok}")
    return JSONResponse({"ok": ok, "jobtag": jobtag, "job_id": jobid, "model": model,
                         "msg": (r.get("stdout") or r.get("stderr") or "").strip(),
                         "poll": f"/api/llm/infer/result?jobtag={jobtag}"})


@app.get("/api/llm/infer/result")
def api_llm_infer_result(jobtag: str):
    if not re.match(r"^[A-Za-z0-9]+$", jobtag):
        raise HTTPException(400, "bad jobtag")
    r = _slurm(["bash", "-lc",
                f"cat results/framework/llm_infer/{jobtag}.json 2>/dev/null"], timeout=20)
    txt = (r.get("stdout") or "").strip()
    if txt:
        try:
            d = json.loads(txt)
            return JSONResponse({"status": "done" if d.get("ok") else "failed",
                                 "text": d.get("text"), "error": d.get("error"),
                                 "model": d.get("model")})
        except Exception:
            pass
    return JSONResponse({"status": "pending"})


@app.get("/models", response_class=HTMLResponse)
def models_page():
    """v3.0.138 — choose the model/provider per agent role (admins edit, all view)."""
    html = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Models</title><style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#1a1a1d}
 .top{background:#0b1220;padding:10px 16px}.top a{display:inline-block;text-decoration:none;background:#1e293b;color:#93c5fd;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px}
 .hero{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:1.3rem 2rem}.hero h1{margin:0 0 .3rem}.hero .sub{opacity:.85;font-size:13px;line-height:1.5}
 .wrap{padding:1.2rem 2rem;max-width:920px;margin:0 auto}
 .card{background:#fff;border:1px solid #e3e7ef;border-radius:12px;padding:16px 18px;margin-bottom:14px}
 .card h3{margin:0 0 2px;font-size:15px}.card .d{color:#64748b;font-size:12px;margin-bottom:10px}
 input,select{padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px}
 input{width:340px;max-width:100%}
 button{border:0;cursor:pointer;background:#0e7c66;color:#fff;font-weight:600;font-size:13px;padding:8px 14px;border-radius:8px}
 button.sec{background:#eef2ff;color:#2563eb}
 .prov{display:flex;gap:8px;flex-wrap:wrap;margin:.4rem 0}
 .pill{font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px}
 .pill.on{background:#dcfce7;color:#166534}.pill.off{background:#f1f5f9;color:#64748b}
 .msg{font-size:12px;color:#475569;margin-left:8px}
 code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
</style></head><body>
<div class="top"><a href="/">&larr; Projects</a></div>
<div class="hero"><h1>&#129504; Models</h1><div class="sub">Which model each agent role uses. Mix local (Ollama) and APIs (DeepSeek / GLM / OpenAI / Anthropic) or a cluster vLLM endpoint &mdash; switch any time. Model id = <code>provider:name</code>.</div></div>
<div class="wrap">
 <div class="card"><h3>Providers</h3><div class="d">Green = ready to use now. API keys live in <code>~/.llm_keys</code> on the server (DEEPSEEK_API_KEY, ZHIPU_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY). Ollama = a local model server.</div><div class="prov" id="prov">loading…</div></div>
 <div id="roles"></div>
 <div class="card"><h3>Test a model</h3><div class="d">Send a tiny prompt to verify a model id works (admin).</div>
   <input id="tm" placeholder="e.g. deepseek:deepseek-chat or ollama:gemma2"> <button onclick="testModel()">Test</button>
   <div class="msg" id="tmsg"></div></div>
 <div class="card"><h3>On-demand cluster inference (our model, no paid key)</h3><div class="d">Submit a prompt &rarr; the server queues a cluster GPU job that runs our own model (ollama, default <code>gemma4</code>) &rarr; answer comes back. This is the self-hosted gateway; first response waits for the GPU queue + model load.</div>
   <input id="ip" placeholder="prompt, e.g. Say hello in one word" style="width:340px;max-width:100%"> <button onclick="clusterInfer()">Run on cluster</button>
   <div class="msg" id="imsg"></div></div>
 <div class="card" style="background:#fffbeb;border-color:#fde68a"><h3>How we serve models (self-hosted, no paid keys)</h3><div class="d" style="color:#92400e">Our plan: don&rsquo;t buy external API keys. We deploy our OWN models on the cluster <b>on-demand</b> (a batch job, same pattern as the harvest/Gemma agent &mdash; the cluster is never always-on). This server (Unie) is the always-on broker: it holds our OWN api key, other machines call the server, and the server submits a queued cluster job to run the model and returns the result. In this page that maps to a <code>cluster:&lt;model&gt;</code> / <code>vllm@&lt;url&gt;:&lt;model&gt;</code> id. Bridges-2 has H100-80GB nodes that can host DeepSeek-V3/V4 &amp; GLM-4.6 for such jobs. Paid APIs (DeepSeek/GLM/OpenAI/Anthropic) remain available as an option but are not required.</div></div>
</div>
<script>
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
var ME={is_admin:false},DATA={};
async function load(){
 try{ME=await (await fetch('/api/me',{credentials:'include'})).json();}catch(e){}
 DATA=await (await fetch('/api/models',{credentials:'include'})).json();
 var pv=document.getElementById('prov');pv.innerHTML='';
 var P=DATA.providers||{};
 Object.keys(P).forEach(function(k){var p=P[k];if(p&&typeof p==='object'){
   pv.innerHTML+='<span class="pill '+(p.configured?'on':'off')+'">'+esc(k)+(p.configured?' \\u2713':' \\u2014 '+esc(p.needs||''))+'</span>';}});
 var rs=document.getElementById('roles');rs.innerHTML='';
 var sug=(DATA.suggested||[]).map(function(s){return '<option value="'+esc(s)+'">'+esc(s)+'</option>';}).join('');
 Object.keys(DATA.roles||{}).forEach(function(role){
  var r=DATA.roles[role];
  var ctl=ME.is_admin
    ? '<input id="in_'+role+'" value="'+esc(r.model)+'" list="sug"> <button onclick="saveRole(\\''+role+'\\')">Save</button> <button class="sec" onclick="testRole(\\''+role+'\\')">Test</button> <span class="msg" id="m_'+role+'"></span>'
    : '<code>'+esc(r.model)+'</code>';
  rs.innerHTML+='<div class="card"><h3>'+esc(role)+'</h3><div class="d">'+esc(r.desc)+'</div>'+ctl+'</div>';
 });
 rs.innerHTML+='<datalist id="sug">'+sug+'</datalist>';
}
async function saveRole(role){
 var m=document.getElementById('in_'+role).value.trim(),el=document.getElementById('m_'+role);
 el.textContent='\\u23f3';
 try{var d=await (await fetch('/api/models/role',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({role:role,model:m})})).json();
  el.textContent=d&&d.ok?'\\u2705 saved':'\\u274c '+((d&&(d.detail||d.msg))||'failed');}catch(e){el.textContent='\\u274c '+e;}
}
async function testRole(role){var m=document.getElementById('in_'+role).value.trim();_test(m,document.getElementById('m_'+role));}
async function testModel(){_test(document.getElementById('tm').value.trim(),document.getElementById('tmsg'));}
async function clusterInfer(){
 var p=document.getElementById('ip').value.trim(),el=document.getElementById('imsg');
 if(!p){el.textContent='enter a prompt';return;}
 el.textContent='\\u23f3 submitting cluster job\\u2026';
 try{
  var d=await (await fetch('/api/llm/infer',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p})})).json();
  if(!d||!d.ok){el.textContent='\\u274c '+((d&&(d.detail||d.msg))||'submit failed');return;}
  el.textContent='\\u23f3 queued (job '+(d.job_id||'?')+') \\u2014 waiting for GPU + model load\\u2026';
  var tag=d.jobtag,tries=0;
  var iv=setInterval(async function(){
   tries++;
   try{var rr=await (await fetch('/api/llm/infer/result?jobtag='+encodeURIComponent(tag),{credentials:'include'})).json();
    if(rr.status==='done'){clearInterval(iv);el.innerHTML='\\u2705 '+esc((rr.text||'').slice(0,400));}
    else if(rr.status==='failed'){clearInterval(iv);el.textContent='\\u274c '+esc(rr.error||'failed');}
    else{el.textContent='\\u23f3 running\\u2026 ('+tries+')';}
   }catch(e){}
   if(tries>120){clearInterval(iv);el.textContent='\\u23f3 still running \\u2014 check back (poll /api/llm/infer/result?jobtag='+tag+')';}
  },10000);
 }catch(e){el.textContent='\\u274c '+e;}
}
async function _test(model,el){
 if(!model){el.textContent='enter a model id';return;}el.textContent='\\u23f3 testing\\u2026';
 try{var d=await (await fetch('/api/models/test',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model})})).json();
  el.textContent=d&&d.ok?('\\u2705 '+esc((d.text||'').slice(0,80))):'\\u274c '+esc(d&&(d.error||d.detail||'failed'));}catch(e){el.textContent='\\u274c '+e;}
}
load();
</script></body></html>'''
    return HTMLResponse(html)


def _read_manual_uploads() -> dict:
    try:
        if _MANUAL_UPLOADS_FILE.is_file():
            return json.load(open(_MANUAL_UPLOADS_FILE)) or {}
    except Exception:
        pass
    return {}


def _append_manual_upload(slug: str, fields: dict) -> None:
    data = _read_manual_uploads()
    data[slug] = fields
    _MANUAL_UPLOADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(_MANUAL_UPLOADS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, _MANUAL_UPLOADS_FILE)


def _registry_current_round() -> int:
    try:
        with open(REGISTRY_PATH) as f:
            return int(json.load(f).get("current_round", 1) or 1)
    except Exception:
        return 1


def _extract_anno_classes(root: Path):
    """v3.0.139 (roadmap #2): scan an extracted dataset for COCO or Pascal-VOC
    annotations and return (sorted class names, format). YOLO has no class names
    in the labels themselves (needs data.yaml), so this covers the other two
    common formats. Bounded: scans up to a few files, caps json size."""
    names = set()
    fmt = ""
    try:
        # COCO: a *.json with a top-level "categories":[{name},...]
        for p in list(root.rglob("*.json"))[:40]:
            try:
                if p.stat().st_size > 80 * 1024 * 1024:
                    continue
                d = json.load(open(p))
            except Exception:
                continue
            cats = d.get("categories") if isinstance(d, dict) else None
            if isinstance(cats, list) and cats:
                for c in cats:
                    if isinstance(c, dict) and c.get("name"):
                        names.add(str(c["name"]))
                if names:
                    fmt = "coco"
                    break
        # Pascal VOC: *.xml with <object><name>...</name>
        if not names:
            import xml.etree.ElementTree as _ET
            for p in list(root.rglob("*.xml"))[:200]:
                try:
                    r = _ET.parse(p).getroot()
                except Exception:
                    continue
                for obj in r.findall(".//object/name"):
                    if obj.text:
                        names.add(obj.text.strip())
                if names:
                    fmt = "voc"
    except Exception:
        pass
    return sorted(n for n in names if n), fmt


def _extract_video_frames(files_dir: Path, img_dir: Path, max_frames: int = 8) -> int:
    """v3.0.139 (roadmap #2): grab a few evenly-spaced preview frames from each
    uploaded video (cv2) so video datasets show thumbnails in the gallery."""
    try:
        import cv2  # available on lab venv
    except Exception:
        return 0
    if not files_dir.is_dir():
        return 0
    vexts = _MODALITY_EXT["video"]
    n = 0
    for vp in sorted(files_dir.rglob("*")):
        if vp.suffix.lower() not in vexts:
            continue
        try:
            cap = cv2.VideoCapture(str(vp))
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if tot <= 0:
                cap.release(); continue
            step = max(1, tot // max_frames)
            grabbed = 0
            for i in range(0, tot, step):
                if grabbed >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    continue
                out = img_dir / f"{vp.stem}_f{i:06d}.jpg"
                if cv2.imwrite(str(out), frame):
                    grabbed += 1; n += 1
            cap.release()
        except Exception:
            continue
    return n


@app.post("/api/dataset/upload")
async def api_dataset_upload(request: Request):
    """Manual dataset upload. POST the .zip as the raw request body with
    ?domain=<id>&name=<dataset name>. Images (+ optional YOLO labels / data.yaml)
    are extracted to uploads/<slug>/ and the dataset is registered (status
    downloaded, source manual_upload, attributed to the uploader)."""
    import zipfile
    import shutil
    import tempfile
    qp = request.query_params
    domain = re.sub(r"[^a-z0-9_]+", "", (qp.get("domain") or "weed").lower())[:40] or "weed"
    name = (qp.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "missing dataset name (?name=)")
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:60] or "dataset"

    # v3.0.139 (roadmap #2): STREAM the body to a temp file instead of loading the
    # whole zip in memory — large robot/video datasets no longer blow up RAM.
    # Size cap enforced incrementally; sha1 (for the slug) computed on the fly.
    _h = hashlib.sha1()
    total = 0
    tmpf = tempfile.NamedTemporaryFile(prefix="upload_", suffix=".zip", delete=False)
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_UPLOAD_BYTES:
                tmpf.close(); os.unlink(tmpf.name)
                raise HTTPException(413, f"zip too large (> {_MAX_UPLOAD_BYTES // (1024*1024)} MB)")
            _h.update(chunk)
            tmpf.write(chunk)
        tmpf.close()
    except HTTPException:
        raise
    except Exception as e:
        try:
            tmpf.close(); os.unlink(tmpf.name)
        except Exception:
            pass
        raise HTTPException(400, f"upload read failed: {type(e).__name__}")
    if total == 0:
        os.unlink(tmpf.name)
        raise HTTPException(400, "empty body — POST the .zip file as the request body")
    zip_path = tmpf.name
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        os.unlink(zip_path)
        raise HTTPException(400, "not a valid .zip file")
    members = [m for m in zf.infolist() if not m.is_dir()]
    if not members:
        zf.close(); os.unlink(zip_path)
        raise HTTPException(400, "zip is empty")
    if len(members) > _MAX_UPLOAD_FILES:
        zf.close(); os.unlink(zip_path)
        raise HTTPException(413, f"too many files in zip (> {_MAX_UPLOAD_FILES})")

    actor = _actor_from_request(request)
    try:
        from . import db as _dbu
        _dbu.upsert_user(actor, auth_provider="basic")
    except Exception:
        pass
    short = _h.hexdigest()[:8]
    slug = f"ul_{base}_{short}"
    dest = _UPLOAD_DIR / slug
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir = dest / "labels"
    files_dir = dest / "files"   # non-image payloads land here

    # v3.0.137: modality — explicit ?modality= wins, else the agent's configured
    # modality, else image. Determines which file types we accept (so video /
    # sensor / point-cloud agents can upload too).
    modality = re.sub(r"[^a-z]", "", (qp.get("modality") or "").lower())
    if modality not in _MODALITY_EXT:
        try:
            from . import db as _dbm
            _dd = _dbm.get_domain(domain) or {}
            _mlist = _dd.get("modality") or ["image"]
            modality = _mlist[0] if _mlist and _mlist[0] in _MODALITY_EXT else "image"
        except Exception:
            modality = "image"
    accept_ext = _MODALITY_EXT.get(modality, _UPLOAD_IMG_EXT)

    n_img = n_lbl = n_file = n_skipped = 0
    data_yaml_bytes = None
    for m in members:
        nm = m.filename
        parts = Path(nm).parts
        if nm.startswith("/") or ".." in parts:    # path-traversal guard
            n_skipped += 1
            continue
        safe_name = Path(nm).name
        if not safe_name:
            continue
        ext = Path(safe_name).suffix.lower()
        if ext in _UPLOAD_IMG_EXT and modality == "image":
            try:
                # v3.0.142: preserve the zip's directory structure (train/<class>/
                # img.jpg) — needed for classification ImageFolder + det/seg splits.
                # The gallery still finds images (it rglobs local_path).
                rel = "/".join(p for p in parts if p not in ("..", ""))[:200] or safe_name
                outp = img_dir / rel
                outp.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(outp, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                n_img += 1
            except Exception:
                n_skipped += 1
        elif ext == ".txt" and "label" in nm.lower():
            label_dir.mkdir(exist_ok=True)
            try:
                with zf.open(m) as src, open(label_dir / safe_name, "wb") as out:
                    shutil.copyfileobj(src, out)
                n_lbl += 1
            except Exception:
                n_skipped += 1
        elif safe_name.lower() in ("data.yaml", "data.yml", "classes.txt"):
            try:
                data_yaml_bytes = zf.read(m)
            except Exception:
                pass
        elif ext in accept_ext or ext in _UPLOAD_SIDECAR_EXT:
            # non-image modality payload (or sidecar): keep folder structure
            files_dir.mkdir(exist_ok=True)
            try:
                rel = "/".join(p for p in parts if p not in ("..", ""))[:180] or safe_name
                outp = files_dir / rel
                outp.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(outp, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                n_file += 1
            except Exception:
                n_skipped += 1
        else:
            n_skipped += 1

    try:
        zf.close()
    except Exception:
        pass
    try:
        os.unlink(zip_path)   # temp upload zip no longer needed
    except Exception:
        pass

    if n_img == 0 and n_file == 0:
        shutil.rmtree(dest, ignore_errors=True)
        _hint = ", ".join(sorted(accept_ext))
        raise HTTPException(400, f"no recognized {modality} files in zip "
                                 f"(accepted: {_hint})")

    class_names = []
    detected_fmt = ""
    if data_yaml_bytes:
        try:
            import yaml as _yaml
            y = _yaml.safe_load(data_yaml_bytes) or {}
            names = y.get("names") if isinstance(y, dict) else None
            if isinstance(names, dict):
                class_names = [str(names[k]) for k in sorted(names)]
            elif isinstance(names, list):
                class_names = [str(x) for x in names]
            if class_names:
                detected_fmt = "yolo"
        except Exception:
            class_names = []
    # v3.0.139 (roadmap #2): COCO/VOC import — pull class names + format from
    # annotation files when there's no data.yaml (so non-YOLO datasets aren't blind).
    if not class_names:
        cn2, fmt2 = _extract_anno_classes(dest)
        if cn2:
            class_names, detected_fmt = cn2, fmt2
    # v3.0.139: video → extract a few preview frames (cv2) so the gallery shows it.
    if modality == "video":
        try:
            n_frames = _extract_video_frames(files_dir, img_dir, max_frames=8)
            n_img += n_frames
        except Exception as e:
            log.warning(f"[upload] video frame extract failed: {e}")

    # coarse format tag
    if detected_fmt:
        fmt = detected_fmt
    elif modality == "image":
        fmt = "yolo" if n_lbl > 0 else "images"
    else:
        fmt = modality
    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    fields = {
        "status": "downloaded",
        "source": "manual_upload",
        "domain": domain,
        "modality": modality,
        "format": fmt,
        "uploaded_by": actor,
        "uploaded_at": uploaded_at,
        "downloaded_at": uploaded_at,
        # image datasets point local_path at images/ (gallery walks it); non-image
        # datasets point at the dataset root so files/ is browsable later.
        "local_path": str(img_dir if modality == "image" else dest),
        "local_images": n_img,
        "n_local_files": n_file,
        "n_local_labels": n_lbl,
        "class_names": class_names,
        "harvest_round": _registry_current_round(),
        "display_name": name,
    }
    # durable record first (survives cluster→lab sync), then live registry+Mongo
    try:
        _append_manual_upload(slug, fields)
    except Exception as e:
        log.warning(f"[upload] manual_uploads.json write failed: {e}")
    wrote = {"mongo": False, "json": False}
    try:
        from . import db as _db
        wrote = _db.upsert_slug(slug, fields, actor=actor)
    except Exception as e:
        log.warning(f"[upload] registry upsert failed: {e}")
    log.info(f"[upload] {slug} domain={domain} modality={modality} imgs={n_img} "
             f"files={n_file} labels={n_lbl} by={actor}")
    return JSONResponse({
        "ok": True, "slug": slug, "domain": domain, "modality": modality,
        "format": fmt, "images": n_img, "files": n_file,
        "labels": n_lbl, "skipped": n_skipped, "class_names": class_names,
        "uploaded_by": actor, "registered": wrote,
        "gallery_url": f"/gallery/{slug}",
    })


# ---- v3.0.145 — shared dataset purge + domain scoping (comprehensive deletion)
def _purge_dataset(slug: str, actor: str = "system") -> dict:
    """Fully remove ONE dataset everywhere: manual_uploads.json, the registry,
    the Mongo slug doc, and uploads/<slug> files on disk. Returns what was removed.
    The single source of truth for dataset deletion (per-dataset + agent cascade)."""
    import shutil
    res = {"slug": slug, "manual": False, "registry": False, "files": False}
    manual = _read_manual_uploads()
    if slug in manual:
        manual.pop(slug, None)
        try:
            _MANUAL_UPLOADS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(_MANUAL_UPLOADS_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(manual, f, indent=2, default=str)
            os.replace(tmp, _MANUAL_UPLOADS_FILE)
            res["manual"] = True
        except Exception as e:
            log.warning(f"[purge] manual_uploads write: {e}")
    try:
        from .registry_lock import atomic_write_json
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        if slug in reg.get("datasets", {}):
            reg["datasets"].pop(slug, None)
            reg["total_downloaded"] = sum(
                1 for v in reg["datasets"].values() if v.get("status") == "downloaded")
            atomic_write_json(REGISTRY_PATH, reg)
            res["registry"] = True
    except Exception as e:
        log.warning(f"[purge] registry write: {e}")
    try:
        from . import db as _db
        dbh = _db._get_db()
        if dbh is not None:
            dbh[_db.COLL_SLUGS].delete_one({"_id": slug})
    except Exception:
        pass
    d = _UPLOAD_DIR / slug
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        res["files"] = True
    log.info(f"[purge] {slug} by {actor}: {res}")
    return res


def _is_manual_dataset(slug: str) -> bool:
    if slug in _read_manual_uploads():
        return True
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)["datasets"].get(slug, {}).get("source") == "manual_upload"
    except Exception:
        return False


def _datasets_for_domain(domain: str) -> list:
    """All dataset slugs scoped to a domain (manual_uploads + registry domain tag).
    The shared weed harvest pool has no domain tag, so it is never matched by a
    non-weed agent — deleting a robot/student agent only touches ITS own data."""
    out = set()
    for s, v in _read_manual_uploads().items():
        if (v.get("domain") or "weed") == domain:
            out.add(s)
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        for s, v in reg.get("datasets", {}).items():
            if v.get("domain") == domain:
                out.add(s)
    except Exception:
        pass
    return sorted(out)


@app.post("/api/dataset/delete")
async def api_dataset_delete(request: Request):
    """Delete a MANUALLY-uploaded dataset (manual_uploads + registry + Mongo +
    files). Safety: only source=manual_upload slugs are deletable here — the
    harvested/cluster weed pool is never removable from the UI."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    slug = str((body.get("slug") if isinstance(body, dict) else "")
               or request.query_params.get("slug") or "").strip()
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug) or slug in (".", ".."):
        raise HTTPException(400, "bad slug")
    if not _is_manual_dataset(slug):
        raise HTTPException(403, "only manually-uploaded datasets can be deleted here")
    res = _purge_dataset(slug, actor=_actor_from_request(request))
    return JSONResponse({"ok": True, "slug": slug, "removed": res})


@app.get("/api/dataset/uploads")
def api_dataset_uploads(domain: str = ""):
    """List manually-uploaded datasets (optionally filtered by domain) so the UI
    can show + manage them."""
    dom = re.sub(r"[^a-z0-9_]+", "", (domain or "").lower())[:40]
    manual = _read_manual_uploads()
    rows = []
    for slug, info in manual.items():
        if dom and (info.get("domain") or "weed") != dom:
            continue
        rows.append({
            "slug": slug,
            "name": info.get("display_name") or slug,
            "images": info.get("local_images", 0),
            "uploaded_by": info.get("uploaded_by"),
            "uploaded_at": info.get("uploaded_at"),
            "domain": info.get("domain") or "weed",
        })
    rows.sort(key=lambda r: (r.get("uploaded_at") or ""), reverse=True)
    return JSONResponse({"ok": True, "n": len(rows), "uploads": rows})


# ===========================================================================
# v3.0.128 (Z4) — user-controlled Roboflow push CAP. Prof Zhang: the agent may
# push to Roboflow, but the USER sets the upper limit (max images per dataset
# the agent uploads). Stored per-domain in a lab-local push_caps.json (durable,
# never synced/clobbered). Enforced on BOTH the manual push (/api/labeling/push
# clamps to the cap) and the autonomous harvest auto-sync (PUSH_CAP env →
# run_v3_0_43_brain_harvest_oneshot.sh --cap-per-slug).
# ===========================================================================
_PUSH_CAPS_FILE = REPO / "results" / "framework" / "push_caps.json"
_DEFAULT_PUSH_CAP = 100
_MAX_PUSH_CAP = 2000


def _norm_domain(domain: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", (domain or "weed").lower())[:40] or "weed"


def _read_push_caps() -> dict:
    try:
        if _PUSH_CAPS_FILE.is_file():
            return json.load(open(_PUSH_CAPS_FILE)) or {}
    except Exception:
        pass
    return {}


def _get_push_cap(domain: str) -> int:
    try:
        return int(_read_push_caps().get(_norm_domain(domain), _DEFAULT_PUSH_CAP))
    except Exception:
        return _DEFAULT_PUSH_CAP


def _set_push_cap(domain: str, cap: int) -> int:
    d = _norm_domain(domain)
    cap = max(1, min(int(cap), _MAX_PUSH_CAP))
    caps = _read_push_caps()
    caps[d] = cap
    _PUSH_CAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(_PUSH_CAPS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(caps, f, indent=2)
    os.replace(tmp, _PUSH_CAPS_FILE)
    return cap


@app.get("/api/domain/push_cap")
def api_get_push_cap(domain: str = "weed"):
    return JSONResponse({"ok": True, "domain": _norm_domain(domain),
                         "cap": _get_push_cap(domain),
                         "default": _DEFAULT_PUSH_CAP, "max": _MAX_PUSH_CAP})


@app.post("/api/domain/push_cap")
async def api_set_push_cap_ep(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    domain = str((body.get("domain") if isinstance(body, dict) else "") or "weed")
    try:
        cap = int(body.get("cap"))
    except (TypeError, ValueError):
        raise HTTPException(400, "cap must be an integer")
    newcap = _set_push_cap(domain, cap)
    try:
        from . import db as _dbu
        _dbu.upsert_user(_actor_from_request(request), auth_provider="basic")
    except Exception:
        pass
    log.info(f"[push_cap] {_norm_domain(domain)} -> {newcap} by {_actor_from_request(request)}")
    return JSONResponse({"ok": True, "domain": _norm_domain(domain), "cap": newcap,
                         "max": _MAX_PUSH_CAP})


# ===========================================================================
# v3.0.129 (Z2) — users + upload attribution. Prof Zhang: students log in with
# their own account; track who uploaded what. Users live in Mongo COLL_USERS;
# uploads already carry uploaded_by. This page joins the two.
# ===========================================================================
@app.get("/api/users")
def api_users():
    """List users + how many datasets / images each has uploaded."""
    try:
        from . import db as _dbu
        _dbu.ensure_default_admin()
        users = _dbu.list_users()
    except Exception:
        users = []
    # aggregate manual uploads by uploader
    agg: dict = {}
    for slug, info in _read_manual_uploads().items():
        who = info.get("uploaded_by") or "admin"
        a = agg.setdefault(who, {"datasets": 0, "images": 0, "slugs": []})
        a["datasets"] += 1
        a["images"] += int(info.get("local_images", 0) or 0)
        a["slugs"].append(slug)
    seeds = _admin_emails()
    creq = _read_cluster_requests()
    rows = []
    seen = set()
    for u in users:
        uid = u.get("_id")
        seen.add(uid)
        a = agg.get(uid, {"datasets": 0, "images": 0})
        eff_admin = (u.get("role") == "admin") or (str(uid).lower() in seeds)
        rows.append({
            "user_id": uid, "name": u.get("name") or uid,
            "email": u.get("email") or "", "role": "admin" if eff_admin else "member",
            "auth_provider": u.get("auth_provider") or "basic",
            "created_at": _iso(u.get("created_at")), "last_seen": _iso(u.get("last_seen")),
            "uploads": a["datasets"], "images": a["images"],
            "can_use_cluster": bool(eff_admin or u.get("can_use_cluster")),
            "is_seed_admin": str(uid).lower() in seeds,
            "requested_cluster": uid in creq,
        })
    # uploaders that aren't in the users table yet (e.g. Mongo was down on upload)
    for who, a in agg.items():
        if who not in seen:
            eff_admin = str(who).lower() in seeds
            rows.append({"user_id": who, "name": who, "email": "",
                         "role": "admin" if eff_admin else "member",
                         "auth_provider": "unknown", "created_at": "", "last_seen": "",
                         "uploads": a["datasets"], "images": a["images"],
                         "can_use_cluster": eff_admin, "is_seed_admin": eff_admin,
                         "requested_cluster": who in creq})
    # requesters not otherwise present (so admins can always act on a request)
    for who in creq:
        if who not in seen and who not in agg:
            rows.append({"user_id": who, "name": who, "email": "", "role": "member",
                         "auth_provider": "unknown", "created_at": "", "last_seen": "",
                         "uploads": 0, "images": 0, "can_use_cluster": False,
                         "is_seed_admin": False, "requested_cluster": True})
    rows.sort(key=lambda r: (-(1 if r.get("requested_cluster") else 0),
                             -(r["uploads"]), r["user_id"]))
    return JSONResponse({"ok": True, "n": len(rows), "users": rows,
                         "mongo": (len(users) > 0)})


@app.post("/api/users/role")
async def api_set_user_role(request: Request):
    """Admin-only: promote/demote a user (admin|member)."""
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    uid = str((body.get("user_id") if isinstance(body, dict) else "") or "").strip()
    role = str(body.get("role") or "")
    if not uid:
        raise HTTPException(400, "user_id required")
    if role not in ("admin", "member"):
        raise HTTPException(400, "role must be admin or member")
    from . import db as _dbu
    ok = _dbu.set_user_role(uid, role, actor=_actor_from_request(request))
    if not ok:
        raise HTTPException(503, "database unavailable")
    return JSONResponse({"ok": True, "user_id": uid, "role": role})


@app.get("/api/keys")
def api_list_keys(request: Request):
    """Admin: list API keys (labels + metadata, never the secret)."""
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    out = []
    for h, rec in _read_api_keys().items():
        out.append({"label": rec.get("label"), "created": rec.get("created"),
                    "owner": rec.get("owner"), "hash_prefix": h[:10]})
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    return JSONResponse({"ok": True, "keys": out})


@app.post("/api/keys")
async def api_create_key(request: Request):
    """Admin: generate a new API key. The plaintext is returned ONCE."""
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(body.get("label") or "").strip())[:40]
    if not label:
        raise HTTPException(400, "label required")
    secret = "ak_" + _secrets.token_urlsafe(30)
    h = hashlib.sha256(secret.encode()).hexdigest()
    keys = _read_api_keys()
    keys[h] = {"label": label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "owner": _actor_from_request(request)}
    _write_api_keys(keys)
    log.info(f"[apikey] created '{label}' by {_actor_from_request(request)}")
    # returned once — not retrievable later
    return JSONResponse({"ok": True, "label": label, "api_key": secret,
                         "note": "Save this now — it is shown only once."})


@app.post("/api/keys/revoke")
async def api_revoke_key(request: Request):
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    label = str(body.get("label") or "")
    keys = _read_api_keys()
    removed = [h for h, r in keys.items() if r.get("label") == label]
    for h in removed:
        keys.pop(h, None)
    if removed:
        _write_api_keys(keys)
    return JSONResponse({"ok": True, "revoked": len(removed), "label": label})


@app.post("/api/users/cluster_access")
async def api_set_user_cluster_access(request: Request):
    """Admin-only: grant/revoke a member's permission to launch cluster GPU jobs."""
    if not _is_admin(_actor_from_request(request)):
        raise HTTPException(403, "Administrators only.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    uid = str((body.get("user_id") if isinstance(body, dict) else "") or "").strip()
    allow = bool(body.get("allow"))
    if not uid:
        raise HTTPException(400, "user_id required")
    from . import db as _dbu
    ok = _dbu.set_user_cluster_access(uid, allow, actor=_actor_from_request(request))
    if not ok:
        raise HTTPException(503, "database unavailable")
    # clear any pending access request once handled
    try:
        reqs = _read_cluster_requests()
        if uid in reqs:
            reqs.pop(uid, None)
            _write_cluster_requests(reqs)
    except Exception:
        pass
    return JSONResponse({"ok": True, "user_id": uid, "can_use_cluster": allow})


def _iso(v):
    try:
        return v.strftime("%Y-%m-%d %H:%M") if hasattr(v, "strftime") else (str(v)[:16] if v else "")
    except Exception:
        return ""


@app.get("/users", response_class=HTMLResponse)
def users_page():
    """v3.0.129 — admin view of users + per-user upload attribution (English)."""
    html = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Users</title><style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#1a1a1d}
 .top{background:#0b1220;padding:10px 16px}
 .top a{display:inline-block;text-decoration:none;background:#1e293b;color:#93c5fd;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px}
 .hero{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:1.3rem 2rem}
 .hero h1{margin:0 0 .3rem}.hero .sub{opacity:.85;font-size:13px}
 .wrap{padding:1.2rem 2rem;max-width:1000px;margin:0 auto}
 .note{background:#fff;border:1px solid #e3e7ef;border-radius:10px;padding:.7rem 1rem;font-size:13px;color:#475569;margin-bottom:14px}
 .tblwrap{overflow-x:auto;border-radius:10px}
 table{width:100%;min-width:640px;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 th,td{padding:.6rem .8rem;text-align:left;font-size:13px;border-bottom:1px solid #f0f2f5}
 th{background:#0f172a;color:#fff;font-size:12px}
 .role{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px}
 .role.admin{background:#fee2e2;color:#991b1b}.role.member{background:#e0f2fe;color:#075985}
 .pill{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px}
 .pill.yes{background:#dcfce7;color:#166534}.pill.no{background:#f1f5f9;color:#64748b}
 .num{font-family:ui-monospace,monospace;text-align:right}
 .act{border:1px solid #cbd5e1;background:#fff;border-radius:7px;padding:4px 9px;font-size:12px;cursor:pointer;margin:2px 2px 0 0}
 .act.warn{border-color:#fecaca;color:#dc2626}
</style></head><body>
<div class="top"><a href="/">&larr; Projects</a></div>
<div class="hero"><h1>&#128100; Users &amp; access</h1><div class="sub">Everyone who has signed in or uploaded data. Per-user attribution + role-based access (admins can grant cluster use).</div></div>
<div class="wrap">
 <div class="note" id="note">loading…</div>
 <div class="tblwrap"><table id="tbl"><thead><tr><th>User</th><th>Email</th><th>Role</th><th>Cluster</th><th>Uploads</th><th>Images</th><th>Last seen</th><th id="acth" style="display:none">Manage</th></tr></thead><tbody></tbody></table></div>
 <div id="keys-card" style="display:none;background:#fff;border:1px solid #e3e7ef;border-radius:12px;padding:16px;margin-top:16px">
   <h3 style="margin:0 0 2px;font-size:15px">&#128273; API keys (our own &mdash; for scripts &amp; robots)</h3>
   <div style="color:#64748b;font-size:12px;margin-bottom:10px">Other machines authenticate by sending <code>X-API-Key: ak_...</code> &mdash; no Google/paid key needed. Member-level (upload/read); does not grant cluster GPU.</div>
   <input id="kl" placeholder="key label (e.g. humanoid-laptop)" style="padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px">
   <button onclick="createKey()" style="border:0;cursor:pointer;background:#0e7c66;color:#fff;font-weight:600;font-size:13px;padding:8px 14px;border-radius:8px">Generate key</button>
   <div id="knew" style="font-size:12px;margin-top:8px"></div>
   <div id="klist" style="font-size:13px;margin-top:10px"></div>
 </div>
</div>
<script>
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
var ME={is_admin:false};
async function load(){
 var note=document.getElementById('note');
 try{ME=await (await fetch('/api/me',{credentials:'include'})).json();}catch(e){}
 let d;
 try{const r=await fetch('/api/users',{credentials:'include'});if(!r.ok){note.textContent='HTTP '+r.status;return;}d=await r.json();}
 catch(e){note.textContent='could not load: '+esc(e);return;}
 if(!d.ok){note.textContent=esc(d.error||'error');return;}
 var adminNote=ME.is_admin?' · you are an <b>admin</b> — you can change roles &amp; cluster access':' · you are a <b>member</b>';
 note.innerHTML='<b>'+d.n+'</b> user(s)'+adminNote+(d.mongo?'':' · <span style="color:#d97706">Mongo offline</span>');
 if(ME.is_admin){document.getElementById('acth').style.display='';document.getElementById('keys-card').style.display='block';loadKeys();}
 var tb=document.querySelector('#tbl tbody');tb.innerHTML='';
 if(!d.users.length){tb.innerHTML='<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:1.4rem">No users yet.</td></tr>';return;}
 d.users.forEach(function(u){
  var tr=document.createElement('tr');
  var cluster=u.can_use_cluster?'<span class="pill yes">allowed</span>':(u.requested_cluster?'<span class="pill" style="background:#fef3c7;color:#92400e">\\u23f3 requested</span>':'<span class="pill no">no</span>');
  var html='<td><b>'+esc(u.name)+'</b><br><span style="color:#94a3b8;font-size:11px">'+esc(u.user_id)+'</span></td>'
   +'<td>'+esc(u.email||'\\u2014')+'</td>'
   +'<td><span class="role '+(u.role==='admin'?'admin':'member')+'">'+esc(u.role)+'</span></td>'
   +'<td>'+cluster+'</td>'
   +'<td class="num">'+u.uploads+'</td><td class="num">'+(u.images||0).toLocaleString()+'</td>'
   +'<td>'+esc(u.last_seen||'\\u2014')+'</td>';
  if(ME.is_admin){
   var btns='';
   if(u.is_seed_admin){btns='<span style="color:#94a3b8;font-size:11px">seed admin</span>';}
   else{
    if(u.role==='admin') btns+='<button class="act" onclick="setRole(\\''+esc(u.user_id)+'\\',\\'member\\')">Make member</button>';
    else btns+='<button class="act" onclick="setRole(\\''+esc(u.user_id)+'\\',\\'admin\\')">Make admin</button>';
    if(u.can_use_cluster) btns+='<button class="act warn" onclick="setCluster(\\''+esc(u.user_id)+'\\',false)">Revoke cluster</button>';
    else btns+='<button class="act" onclick="setCluster(\\''+esc(u.user_id)+'\\',true)">Grant cluster</button>';
   }
   html+='<td>'+btns+'</td>';
  }
  tr.innerHTML=html;
  tb.appendChild(tr);
 });
}
async function setRole(uid,role){
 if(!confirm('Set '+uid+' role to '+role+'?'))return;
 try{var d=await (await fetch('/api/users/role',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:uid,role:role})})).json();
  if(d&&d.ok){load();}else{alert((d&&(d.detail||d.msg))||'failed');}}catch(e){alert(''+e);}
}
async function setCluster(uid,allow){
 if(!confirm((allow?'Grant':'Revoke')+' cluster (GPU) access for '+uid+'?'))return;
 try{var d=await (await fetch('/api/users/cluster_access',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:uid,allow:allow})})).json();
  if(d&&d.ok){load();}else{alert((d&&(d.detail||d.msg))||'failed');}}catch(e){alert(''+e);}
}
async function loadKeys(){
 try{var d=await (await fetch('/api/keys',{credentials:'include'})).json();var el=document.getElementById('klist');
  if(!d||!d.ok){el.textContent='';return;}
  if(!d.keys.length){el.innerHTML='<span style="color:#94a3b8">no keys yet</span>';return;}
  el.innerHTML=d.keys.map(function(k){return '<div style="display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid #eef1f6"><span><b>'+esc(k.label)+'</b> <span style="color:#94a3b8">'+esc(k.created||'')+' \\u00b7 '+esc(k.hash_prefix)+'\\u2026</span></span><button onclick="revokeKey(\\''+esc(k.label)+'\\')" style="border:1px solid #fecaca;background:#fff;color:#dc2626;font-size:12px;padding:4px 9px;border-radius:7px;cursor:pointer">Revoke</button></div>';}).join('');
 }catch(e){}
}
async function createKey(){
 var lbl=(document.getElementById('kl').value||'').trim(),nw=document.getElementById('knew');
 if(!lbl){nw.textContent='enter a label';return;}nw.textContent='\\u23f3';
 try{var d=await (await fetch('/api/keys',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:lbl})})).json();
  if(d&&d.ok){nw.innerHTML='\\u2705 <b>Save now (shown once):</b> <code style="background:#f1f5f9;padding:2px 6px;border-radius:4px">'+esc(d.api_key)+'</code>';document.getElementById('kl').value='';loadKeys();}
  else{nw.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}}catch(e){nw.textContent='\\u274c '+e;}
}
async function revokeKey(label){
 if(!confirm('Revoke API key "'+label+'"? Machines using it will stop working.'))return;
 try{await fetch('/api/keys/revoke',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:label})});loadKeys();}catch(e){}
}
load();
</script></body></html>'''
    return HTMLResponse(html)


# ---- v3.0.130 (Z3) — login / Google OAuth routes (all auth-exempt) ----
@app.get("/login", response_class=HTMLResponse)
def login_page():
    if _GOOGLE_ENABLED:
        body = ('<a class="gbtn" href="/auth/google/start">'
                '<span style="font-weight:700">G</span>&nbsp; Sign in with Google</a>'
                '<p class="muted">Sign in with your institutional Google account. '
                'Your uploads and labeling actions are recorded under your name.</p>')
    else:
        body = ('<p class="muted">Google sign-in is not configured on this server yet. '
                'Use the shared access credentials in the browser prompt, or ask the '
                'administrator to enable Google login.</p>'
                '<details style="margin-top:14px;text-align:left">'
                '<summary style="cursor:pointer;color:#2563eb;font-weight:600">Enable Google login (administrator)</summary>'
                '<ol style="color:#475569;font-size:13px;line-height:1.7;margin-top:8px">'
                '<li>Google Cloud Console &rarr; APIs &amp; Services &rarr; Credentials &rarr; '
                'Create <b>OAuth client ID</b> (type: Web application).</li>'
                '<li>Add an Authorized redirect URI: '
                '<code>&lt;dashboard URL&gt;/auth/google/callback</code>.</li>'
                '<li>In <code>deploy/run_dashboard_labserver.sh</code> set '
                '<code>GOOGLE_CLIENT_ID</code>, <code>GOOGLE_CLIENT_SECRET</code>, and '
                '<code>OAUTH_REDIRECT_BASE</code> (the dashboard&rsquo;s public URL), then restart.</li>'
                '</ol></details>')
    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in — AgentAI dataset platform</title><style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;
   min-height:100vh;display:flex;align-items:center;justify-content:center;
   background:linear-gradient(135deg,#0f172a,#1e293b);color:#1a1a1d}}
 .card{{background:#fff;border-radius:16px;padding:36px 30px;max-width:380px;width:92%;
   text-align:center;box-shadow:0 12px 40px rgba(0,0,0,.3)}}
 h1{{margin:0 0 4px;font-size:20px}} .sub{{color:#64748b;font-size:13px;margin-bottom:22px}}
 .muted{{color:#64748b;font-size:13px;line-height:1.6;margin-top:16px}}
 .gbtn{{display:inline-flex;align-items:center;justify-content:center;gap:6px;
   text-decoration:none;background:#fff;color:#3c4043;border:1px solid #dadce0;
   font-weight:600;font-size:14px;padding:11px 18px;border-radius:9px;width:100%}}
 .gbtn:hover{{background:#f8fafc}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body>
 <div class="card">
   <h1>&#129302; AgentAI Dataset Platform</h1>
   <div class="sub">Autonomous data-collection agents</div>
   {body}
 </div>
</body></html>'''
    return HTMLResponse(html)


@app.get("/auth/google/start")
def google_start():
    if not _GOOGLE_ENABLED:
        return RedirectResponse(url="/login", status_code=302)
    import urllib.parse as _up
    state = _secrets.token_urlsafe(24)
    params = {
        "client_id": _GOOGLE_CLIENT_ID,
        "redirect_uri": _OAUTH_REDIRECT_BASE + "/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + _up.urlencode(params)
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie("oauth_state", state, max_age=600, httponly=True, samesite="lax")
    return resp


@app.get("/auth/google/callback")
def google_callback(request: Request, code: str = "", state: str = ""):
    if not _GOOGLE_ENABLED:
        return RedirectResponse(url="/login", status_code=302)
    saved = None
    try:
        saved = request.cookies.get("oauth_state")
    except Exception:
        saved = None
    if not code or not state or not saved or not _secrets.compare_digest(state, saved):
        return HTMLResponse("<h3>Login failed: invalid state.</h3>"
                            "<a href='/login'>Try again</a>", status_code=400)
    import urllib.request as _ur
    import urllib.parse as _up
    try:
        data = _up.urlencode({
            "code": code,
            "client_id": _GOOGLE_CLIENT_ID,
            "client_secret": _GOOGLE_CLIENT_SECRET,
            "redirect_uri": _OAUTH_REDIRECT_BASE + "/auth/google/callback",
            "grant_type": "authorization_code",
        }).encode()
        with _ur.urlopen(_ur.Request("https://oauth2.googleapis.com/token", data=data),
                         timeout=20) as r:
            tok = json.load(r)
        access = tok.get("access_token")
        if not access:
            raise RuntimeError("no access_token")
        with _ur.urlopen(_ur.Request("https://www.googleapis.com/oauth2/v2/userinfo",
                                     headers={"Authorization": "Bearer " + access}),
                         timeout=20) as r:
            ui = json.load(r)
    except Exception as e:
        return HTMLResponse(f"<h3>Login failed: {type(e).__name__}</h3>"
                            "<a href='/login'>Try again</a>", status_code=502)
    email = (ui.get("email") or "").strip()
    name = (ui.get("name") or email or "user").strip()
    uid = email or ui.get("id") or ("g_" + _secrets.token_hex(6))
    try:
        from . import db as _dbu
        _dbu.upsert_user(uid, email=email, name=name,
                         auth_provider="google", role="member")
    except Exception:
        pass
    payload = {"uid": uid, "email": email, "name": name,
               "exp": time.time() + _SESSION_TTL}
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(_SESSION_COOKIE, _sign_session(payload),
                    max_age=_SESSION_TTL, httponly=True, samesite="lax")
    resp.delete_cookie("oauth_state")
    log.info(f"[auth] google login: {uid}")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(_SESSION_COOKIE)
    return resp


@app.get("/agent/{domain_id}", response_class=HTMLResponse)
def agent_generic(domain_id: str):
    """v3.0.107: mission control for a created (non-weed) domain agent. Honest:
    a fresh domain has no data and its harvest pipeline isn't wired yet. The
    specific /agent/weed route above takes precedence for the weed agent."""
    if domain_id == "weed":
        return agent_weed()
    if not re.match(r"^[a-z0-9_]{1,40}$", domain_id):
        raise HTTPException(400, "bad agent id")
    from . import db as _db
    d = _db.get_domain(domain_id)
    if not d:
        raise HTTPException(404, f"no agent '{domain_id}'")
    import html as _h
    nm = _h.escape(str(d.get("display_name") or domain_id))
    st = _h.escape(str(d.get("status") or "created"))
    qs = d.get("harvest_queries") or []
    qline = _h.escape(", ".join(qs)) if qs else "&mdash; none set &mdash;"
    # v3.0.137: generalized agent attributes (default to image/detection/yolo)
    _task = _h.escape(str(d.get("task") or "detection"))
    _mods = d.get("modality") or ["image"]
    _modline = _h.escape(", ".join(str(m) for m in _mods))
    _model = _h.escape(str(d.get("model") or "auto"))
    _metric = _h.escape(str(d.get("target_metric") or "mAP50-95"))
    _owner = _h.escape(str(d.get("owner") or ""))
    # v3.0.146: agents are components inside this PROJECT (0..N, freely composed)
    _agents = d.get("agents") or []
    if _agents:
        _arows = "".join(
            ('<div style="display:flex;align-items:center;gap:10px;padding:9px 12px;'
             + ('border-top:1px solid #eef1f6;' if i else '') + 'font-size:13px">'
             + '<span style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;'
               'background:#eef2ff;color:#1d4ed8">' + _h.escape(str(a.get("type", ""))) + '</span>'
             + '<b>' + _h.escape(str(a.get("name") or a.get("type", ""))) + '</b>'
             + '<span style="color:#94a3b8;margin-left:auto">' + _h.escape(str(a.get("status", "idle"))) + '</span>'
             + '<button class="agdel" data-id="' + _h.escape(str(a.get("id", ""))) + '" '
               'style="display:none;border:1px solid #fecaca;background:#fff;color:#dc2626;font-size:12px;'
               'padding:4px 9px;border-radius:7px;cursor:pointer" '
               'onclick="removeAgent(this.getAttribute(\'data-id\'))">remove</button></div>')
            for i, a in enumerate(_agents))
    else:
        _arows = ('<div style="color:#64748b;font-size:13px;padding:4px 0">No agents yet &mdash; '
                  'this project is a pure dataset workspace. Add an agent when you want automation '
                  '(collect / filter / label / train / &hellip;).</div>')
    if qs:
        _hbtn = ('<button class="btn" id="hbtn" onclick="startHarvest()" '
                 'style="border:0;cursor:pointer">&#9654; Start harvest</button>'
                 '<span id="htoast" style="display:none;margin-left:12px;font-size:13px"></span>')
        _note = ("Harvest is wired (v3.0.108): it uses this domain's seed queries "
                 "+ topic vocabulary and tags every collected dataset with "
                 f"<code>{_h.escape(domain_id)}</code>.")
    else:
        _hbtn = ''
        _note = ("Add seed search queries (re-create the agent with queries) to "
                 "enable harvesting for this domain.")
    page = (f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{nm} &mdash; Project</title>
<style>
 *{{box-sizing:border-box}}
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f2f4f8;color:#1a1a1d}}
 .top{{background:#fff;border-bottom:1px solid #e3e7ef;padding:14px 22px;display:flex;align-items:center;gap:14px}}
 .top a.bc{{color:#64748b;text-decoration:none;font-size:13px}} .top h1{{font-size:18px;margin:0}}
 .wrap{{max-width:760px;margin:34px auto;padding:0 18px}}
 .card{{background:#fff;border:1px solid #e3e7ef;border-radius:14px;padding:24px}}
 .badge{{display:inline-block;font-size:12px;padding:3px 11px;border-radius:20px;background:#fef9e7;color:#946c00;border:1px solid #f3e1a0}}
 .row{{margin:14px 0;font-size:14px;color:#334}} .row b{{color:#0f172a}}
 .steps{{margin:14px 0 0;padding-left:20px;color:#475569;font-size:13.5px;line-height:1.7}}
 a.btn{{display:inline-block;margin-top:18px;text-decoration:none;background:#2563eb;color:#fff;font-weight:600;font-size:13px;padding:10px 16px;border-radius:9px}}
</style></head><body>
 <div class="top"><a class="bc" href="/">&larr; Agents</a><h1>&#129516; {nm}</h1></div>
 <div class="wrap"><div class="card">
   <span class="badge">&#9679; {st}</span>
   <span class="badge" style="background:#eef2ff;color:#1d4ed8;border-color:#bfdbfe">task: {_task}</span>
   <span class="badge" style="background:#ecfdf5;color:#047857;border-color:#a7f3d0">modality: {_modline}</span>
   <span class="badge" style="background:#fdf4ff;color:#a21caf;border-color:#f5d0fe">model: {_model}</span>
   {('<span class="badge" style="background:#f0fdfa;color:#0f766e;border-color:#99f6e4">field: ' + _h.escape(str(d.get("research_field"))) + '</span>') if d.get("research_field") else ''}
   <div id="manage" data-owner="{_owner}" style="display:none;margin-top:12px;padding-top:10px;border-top:1px solid #eef1f6">
     <button onclick="renameAgent()" style="border:1px solid #cbd5e1;background:#fff;color:#334;font-size:12px;padding:6px 11px;border-radius:7px;cursor:pointer">&#9998; Rename</button>
     <button onclick="editQueries()" style="border:1px solid #cbd5e1;background:#fff;color:#334;font-size:12px;padding:6px 11px;border-radius:7px;cursor:pointer">&#9998; Edit seed queries</button>
     <button onclick="deleteAgent()" style="border:1px solid #fecaca;background:#fff;color:#dc2626;font-size:12px;padding:6px 11px;border-radius:7px;cursor:pointer">&#128465; Delete agent</button>
     <span id="mng-msg" style="font-size:12px;color:#475569;margin-left:6px"></span>
   </div>
   <div class="row" style="margin-top:16px">A <b>project</b> is your research workspace. Upload datasets (any kind) below, and add <b>agents</b> to automate collection, filtering, labeling, or training &mdash; any number, any mix. Target metric: <b>{_metric}</b>.</div>

   <div style="margin-top:20px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8">Agents in this project</div>
   <div style="margin-top:10px;border:1px solid #e3e7ef;border-radius:10px;overflow:hidden">{_arows}</div>
   <div id="agent-add" style="display:none;margin-top:10px;display:none;gap:8px;flex-wrap:wrap;align-items:center">
     <select id="ag-type" style="padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px">
       <option value="collector">Collector (auto-collect datasets)</option>
       <option value="filter">Filter / QC</option>
       <option value="labeler">Labeler</option>
       <option value="trainer">Trainer</option>
       <option value="evaluator">Evaluator</option>
       <option value="custom">Custom</option>
     </select>
     <input id="ag-name" placeholder="name (optional)" style="padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px">
     <button onclick="addAgent()" style="border:0;cursor:pointer;background:#2563eb;color:#fff;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">&#43; Add agent</button>
     <span id="ag-msg" style="font-size:12px;color:#475569"></span>
   </div>
   <div style="margin-top:14px">{_hbtn}</div>
   <div style="margin-top:22px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8">Upload a dataset (.zip)</div>
   <div style="margin-top:10px;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px;padding:14px">
     <div style="font-size:13px;color:#475569;margin-bottom:10px">Drop a <b>.zip</b> of images (optionally with YOLO <code>labels/</code> + <code>data.yaml</code>). It registers as a dataset for this agent and appears under Datasets. Future: automatic upload + open community contributions.</div>
     <input id="ds-name" type="text" placeholder="Dataset name (e.g. lab-run-2026-06)" style="width:100%;padding:9px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px;margin-bottom:8px">
     <input id="ds-file" type="file" accept=".zip,application/zip" style="font-size:13px;margin-bottom:10px;display:block">
     <button id="ds-up" onclick="uploadDataset()" class="btn" style="margin-top:0;border:0;cursor:pointer">&#11014; Upload dataset</button>
     <span id="ds-toast" style="margin-left:12px;font-size:13px"></span>
   </div>
   <div id="ul-wrap" style="margin-top:14px"></div>
   <div style="margin-top:22px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8">Roboflow push</div>
   <div style="margin-top:10px;background:#f8fafc;border:1px solid #e3e7ef;border-radius:10px;padding:14px">
     <div style="font-size:13px;color:#475569;margin-bottom:10px">Set the <b>upper limit</b> — the most images the agent pushes per dataset to Roboflow for labeling.</div>
     <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
       <label style="font-size:13px;color:#334">Agent push cap:</label>
       <input id="cap" type="number" min="1" max="2000" style="width:100px;padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px">
       <button id="cap-save" onclick="saveCap()" style="border:0;cursor:pointer;background:#0e7c66;color:#fff;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">Save</button>
       <span id="cap-msg" style="font-size:13px;color:#475569"></span>
     </div>
     <div style="margin-top:12px"><a href="https://app.roboflow.com/a-test-of-will" target="_blank" style="text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">Adjust labels in Roboflow &#8599;</a></div>
   </div>
   <div style="margin-top:22px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8">Train a model</div>
   <div style="margin-top:10px;background:#f8fafc;border:1px solid #e3e7ef;border-radius:10px;padding:14px">
     <div style="font-size:13px;color:#475569;margin-bottom:10px">Train this agent&rsquo;s <b>{_task}</b> task on one of its uploaded datasets. Runs on the cluster GPU on-demand (queued); the result mAP/accuracy is written back here.</div>
     <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
       <select id="tr-ds" style="padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px;min-width:200px"><option value="">&mdash; choose an uploaded dataset &mdash;</option></select>
       <label style="font-size:13px;color:#334">epochs <input id="tr-ep" type="number" value="20" min="1" max="300" style="width:70px;padding:8px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px"></label>
       <button id="tr-go" onclick="submitTrain()" style="border:0;cursor:pointer;background:#7c3aed;color:#fff;font-weight:600;font-size:13px;padding:9px 14px;border-radius:8px">&#128640; Train</button>
       <span id="tr-msg" style="font-size:13px;color:#475569"></span>
     </div>
   </div>
   <div style="margin-top:22px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8">Workspace (scoped to this agent)</div>
   <div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap">
     <a class="btn" style="margin-top:0;background:#eef2ff;color:#2563eb" href="/classes?domain=__DOM__">Browse data</a>
     <a class="btn" style="margin-top:0;background:#eef2ff;color:#2563eb" href="/slugs?domain=__DOM__">Datasets</a>
   </div>
   <a class="btn" href="/" style="margin-top:20px">&larr; Back to agents</a>
 </div></div>
''' + '''<script>
function startHarvest(){
 if(!confirm('Launch a harvest on the cluster for this domain (uses its seed queries)? This starts a GPU job.'))return;
 var t=document.getElementById('htoast'),b=document.getElementById('hbtn');
 t.style.display='inline';t.textContent='\\u23f3 Submitting\\u2026';b.disabled=true;
 fetch('/api/cluster_action/brain_harvest',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'__DOM__'})})
  .then(function(r){return r.json();}).then(function(d){t.textContent=(d&&d.ok?'\\u2705 Harvest submitted to cluster.':'\\u274c '+((d&&(d.msg||d.stderr))||'failed'));})
  .catch(function(e){t.textContent='\\u274c '+e;}).finally(function(){b.disabled=false;});
}
async function uploadDataset(){
 var nameEl=document.getElementById('ds-name'),fileEl=document.getElementById('ds-file'),
     t=document.getElementById('ds-toast'),btn=document.getElementById('ds-up');
 var name=(nameEl.value||'').trim();
 if(!name){t.textContent='\\u26a0 enter a dataset name';return;}
 if(!fileEl.files||!fileEl.files[0]){t.textContent='\\u26a0 choose a .zip file';return;}
 var f=fileEl.files[0];
 if(!/\\.zip$/i.test(f.name)){t.textContent='\\u26a0 must be a .zip file';return;}
 btn.disabled=true;t.textContent='\\u23f3 uploading '+(f.size/1048576).toFixed(1)+' MB\\u2026';
 try{
  var url='/api/dataset/upload?domain=__DOM__&name='+encodeURIComponent(name);
  var r=await fetch(url,{method:'POST',credentials:'include',headers:{'Content-Type':'application/zip'},body:f});
  var d=await r.json();
  if(r.ok&&d&&d.ok){t.innerHTML='\\u2705 '+d.images+' images registered as <b>'+d.slug+'</b> \\u00b7 <a href="'+d.gallery_url+'" target="_blank">view</a>';nameEl.value='';fileEl.value='';loadUploads();}
  else{t.textContent='\\u274c '+((d&&(d.detail||d.msg))||('HTTP '+r.status));}
 }catch(e){t.textContent='\\u274c '+e;}finally{btn.disabled=false;}
}
async function loadUploads(){
 var w=document.getElementById('ul-wrap');if(!w)return;
 try{
  var d=await (await fetch('/api/dataset/uploads?domain=__DOM__',{credentials:'include'})).json();
  var rows=(d&&d.uploads)||[];
  if(!rows.length){w.innerHTML='';return;}
  var h='<div style="font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8;margin-bottom:6px">Your uploads ('+rows.length+')</div>';
  h+='<div style="border:1px solid #e3e7ef;border-radius:10px;overflow:hidden">';
  rows.forEach(function(u,i){
   h+='<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 12px;'+(i?'border-top:1px solid #eef1f6;':'')+'font-size:13px">'
     +'<span><a href="/gallery/'+encodeURIComponent(u.slug)+'" target="_blank" style="color:#2563eb;text-decoration:none;font-weight:600">'+(u.name||u.slug)+'</a>'
     +' <span style="color:#94a3b8">\\u00b7 '+u.images+' imgs \\u00b7 by '+(u.uploaded_by||'?')+'</span></span>'
     +'<button onclick="deleteUpload(\\''+u.slug+'\\',this)" style="border:1px solid #fecaca;background:#fff;color:#dc2626;font-size:12px;padding:5px 10px;border-radius:7px;cursor:pointer">Delete</button>'
     +'</div>';
  });
  h+='</div>';w.innerHTML=h;
 }catch(e){w.innerHTML='';}
}
async function deleteUpload(slug,btn){
 if(!confirm('Delete uploaded dataset '+slug+'? This removes its images and registration.'))return;
 btn.disabled=true;btn.textContent='\\u2026';
 try{
  var d=await (await fetch('/api/dataset/delete',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({slug:slug})})).json();
  if(d&&d.ok){loadUploads();}else{btn.disabled=false;btn.textContent='Delete';alert((d&&(d.detail||d.msg))||'delete failed');}
 }catch(e){btn.disabled=false;btn.textContent='Delete';alert(''+e);}
}
loadUploads();
loadTrainDatasets();
(async function(){try{
  var me=await (await fetch('/api/me',{credentials:'include'})).json();
  var mng=document.getElementById('manage');if(!mng)return;
  var owner=mng.getAttribute('data-owner')||'';
  if(me&&me.ok&&(me.is_admin||(owner&&owner===me.user))){
    mng.style.display='block';
    var aa=document.getElementById('agent-add');if(aa)aa.style.display='flex';
    Array.prototype.forEach.call(document.querySelectorAll('.agdel'),function(b){b.style.display='inline-block';});
  }
}catch(e){}})();
async function addAgent(){
 var t=document.getElementById('ag-type').value,nm=document.getElementById('ag-name').value.trim(),m=document.getElementById('ag-msg');
 m.textContent='\\u23f3';
 try{var d=await (await fetch('/api/project/agent/add',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({project:'__DOM__',type:t,name:nm})})).json();
  if(d&&d.ok){m.textContent='\\u2705 added';setTimeout(function(){location.reload();},600);}else{m.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}}catch(e){m.textContent='\\u274c '+e;}
}
async function removeAgent(aid){
 if(!confirm('Remove this agent from the project?'))return;
 try{var d=await (await fetch('/api/project/agent/remove',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({project:'__DOM__',agent_id:aid})})).json();
  if(d&&d.ok){location.reload();}else{alert((d&&(d.detail||d.msg))||'failed');}}catch(e){alert(''+e);}
}
async function renameAgent(){
 var nm=prompt('New display name for this agent:');if(!nm||!nm.trim())return;
 var m=document.getElementById('mng-msg');m.textContent='\\u23f3';
 try{var d=await (await fetch('/api/agent/update',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'__DOM__',display_name:nm.trim()})})).json();
  if(d&&d.ok){m.textContent='\\u2705 renamed';setTimeout(function(){location.reload();},700);}else{m.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}}catch(e){m.textContent='\\u274c '+e;}
}
async function editQueries(){
 var q=prompt('Seed search queries (comma-separated):');if(q===null)return;
 var m=document.getElementById('mng-msg');m.textContent='\\u23f3';
 try{var d=await (await fetch('/api/agent/update',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'__DOM__',queries:q})})).json();
  if(d&&d.ok){m.textContent='\\u2705 saved';setTimeout(function(){location.reload();},700);}else{m.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}}catch(e){m.textContent='\\u274c '+e;}
}
async function deleteAgent(){
 var n=0;try{var u=await (await fetch('/api/dataset/uploads?domain=__DOM__',{credentials:'include'})).json();n=(u.uploads||[]).length;}catch(e){}
 var warn='Delete this agent? This permanently removes the agent';
 warn += n>0 ? (' AND its '+n+' uploaded dataset(s) — files, registry, and database records. This cannot be undone.') : '. This cannot be undone.';
 if(!confirm(warn))return;
 var m=document.getElementById('mng-msg');m.textContent='\\u23f3 deleting agent + data\\u2026';
 try{var d=await (await fetch('/api/agent/delete',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'__DOM__'})})).json();
  if(d&&d.ok){m.textContent='\\u2705 deleted ('+(d.datasets_removed||0)+' datasets removed)';setTimeout(function(){location.href='/';},900);}else{m.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}}catch(e){m.textContent='\\u274c '+e;}
}
async function loadCap(){
 try{var d=await (await fetch('/api/domain/push_cap?domain=__DOM__',{credentials:'include'})).json();
  if(d&&d.ok){document.getElementById('cap').value=d.cap;}}catch(e){}
}
async function loadTrainDatasets(){
 try{
  var me=await (await fetch('/api/me',{credentials:'include'})).json();
  var sel=document.getElementById('tr-ds'),msg=document.getElementById('tr-msg'),go=document.getElementById('tr-go');
  if(me&&me.ok&&!me.can_use_cluster){go.disabled=true;sel.disabled=true;msg.textContent='\\u2139 training is admin/granted-only';return;}
  var d=await (await fetch('/api/dataset/uploads?domain=__DOM__',{credentials:'include'})).json();
  (d.uploads||[]).forEach(function(u){var o=document.createElement('option');o.value=u.slug;o.textContent=u.name+' ('+u.images+' imgs)';sel.appendChild(o);});
 }catch(e){}
}
async function submitTrain(){
 var sel=document.getElementById('tr-ds'),ep=document.getElementById('tr-ep'),msg=document.getElementById('tr-msg'),go=document.getElementById('tr-go');
 var slug=sel.value;if(!slug){msg.textContent='\\u26a0 choose a dataset';return;}
 if(!confirm('Stage "'+slug+'" to the cluster and submit a GPU training job?'))return;
 go.disabled=true;msg.textContent='\\u23f3 staging + submitting\\u2026';
 try{var d=await (await fetch('/api/train/submit',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'__DOM__',slug:slug,epochs:parseInt(ep.value||'20')})})).json();
  msg.textContent=(d&&d.ok)?('\\u2705 submitted ('+d.task+', '+d.epochs+'ep) \\u2014 '+(d.msg||'')):'\\u274c '+((d&&(d.detail||d.msg))||'failed');}
 catch(e){msg.textContent='\\u274c '+e;}finally{go.disabled=false;}
}
async function saveCap(){
 var el=document.getElementById('cap'),m=document.getElementById('cap-msg'),b=document.getElementById('cap-save');
 var cap=parseInt(el.value,10);if(!cap||cap<1){m.textContent='\\u26a0 enter a number';return;}
 b.disabled=true;m.textContent='\\u23f3 saving\\u2026';
 try{var d=await (await fetch('/api/domain/push_cap',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:'__DOM__',cap:cap})})).json();
  if(d&&d.ok){el.value=d.cap;m.textContent='\\u2705 saved (cap '+d.cap+')';}else{m.textContent='\\u274c '+((d&&(d.detail||d.msg))||'failed');}
 }catch(e){m.textContent='\\u274c '+e;}finally{b.disabled=false;}
}
loadCap();
</script></body></html>''')
    page = page.replace("__DOM__", domain_id)
    return HTMLResponse(page)


@app.get("/console", response_class=HTMLResponse)
def console_page():
    """v3.0.66 — unified single-page command center.

    v3.0.100: moved from "/" to "/console". "/" is now the Agent Launcher
    (clean entry) and /agent/weed is the per-agent Mission Control. This page
    is preserved BYTE-FOR-BYTE as the advanced/classic console — all existing
    action buttons + logic intact (user: do NOT change functionality).

    User feedback 2026-05-31: previously / was a hub of card-links to
    other pages. User clicking those from github.io got 404 because
    github.io has no routes for subpaths. Plus they wanted everything
    'on ONE page'. Now / loads ALL pipeline state in panels with
    auto-refresh:

      - Top banner: live agent + registry + Roboflow + disk in one row
      - Action grid: every cluster_action button inline (no /control hop)
      - Live SLURM queue
      - Recent agent task log (from logs/agent_tasks/ via /api/recent_jobs)
      - Roboflow workspace state (from /api/roboflow_status)
      - CWD12 per-species snapshot

    Drill-down /classes /slugs /roboflow stay accessible from header nav."""
    return HTMLResponse('''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🌱 Weed-detection — Framework Controller</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif;
         max-width: 1400px; margin: 16px auto; padding: 1rem; color: #1a1a1d;
         background: #f2f3f7; }
  h1 { margin: 0 0 6px 0; font-size: 22px; }
  .sub { color: #666; margin-bottom: 22px; font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 14px; margin-bottom: 22px; }
  .card { background: #fff; padding: 16px 18px; border-radius: 10px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06);
          text-decoration: none; color: inherit; transition: transform 0.1s; }
  .card:hover { transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.10); }
  .card .icon { font-size: 28px; margin-bottom: 8px; }
  .card .title { font-size: 16px; font-weight: 600; margin-bottom: 6px;
                 color: #06c; }
  .card .desc { font-size: 13px; color: #555; line-height: 1.45; }
  .card.new { border-left: 4px solid #38a169; }
  .card.old { border-left: 4px solid #888; }
  .card.api { border-left: 4px solid #06c; background: #f6f9ff; }
  .section-h { color: #555; font-size: 13px; text-transform: uppercase;
               letter-spacing: 0.5px; margin: 18px 0 8px 0; }
  .footer { color: #999; font-size: 12px; margin-top: 30px; text-align: center; }
  .live-banner { background: linear-gradient(90deg, #2a7 0%, #28a 100%);
                 color: #fff; padding: 10px 16px; border-radius: 8px;
                 margin-bottom: 14px; font-size: 13px;
                 display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .live-banner.idle { background: linear-gradient(90deg, #888 0%, #aaa 100%); }
  .live-banner .pulse { width: 8px; height: 8px; background: #fff;
                        border-radius: 50%; animation: pulse 1.5s infinite; }
  .live-banner.idle .pulse { animation: none; opacity: 0.5; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .live-banner a { color: #fff; text-decoration: underline; font-weight: 600; }
  /* v3.0.66 unified-controller styles */
  .stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
    gap:10px;margin-bottom:14px}
  .stat{background:#fff;border-radius:8px;padding:10px 14px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  .stat .l{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px}
  .stat .v{font-size:20px;font-weight:600;color:#000;margin-top:2px}
  .stat .v.green{color:#2a7} .stat .v.red{color:#c44}
  .panels{display:grid;grid-template-columns:1.4fr 1fr;gap:14px;margin-bottom:14px}
  @media (max-width:1100px){.panels{grid-template-columns:1fr}}
  .panel{background:#fff;border-radius:10px;padding:14px 16px;
    box-shadow:0 1px 3px rgba(0,0,0,.06);overflow:hidden}
  .panel h2{font-size:14px;margin:0 0 10px 0;color:#444;
    display:flex;align-items:center;justify-content:space-between}
  .panel h2 a{font-size:11px;color:#06c;text-decoration:none;font-weight:400}
  .actions{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}
  .act{background:#f7f7f9;border:1px solid #e0e0e6;padding:8px 10px;border-radius:6px;
    cursor:pointer;font-size:12px;line-height:1.3;transition:all .12s;text-align:left;
    font-family:inherit;color:#222}
  .act:hover{background:#eef4ff;border-color:#06c}
  .act .nm{font-weight:600;color:#06c;font-size:13px}
  .act .ds{color:#666;margin-top:3px;font-size:11px}
  .ord{display:inline-block;min-width:16px;height:16px;line-height:16px;text-align:center;
       background:#0e7c66;color:#fff;border-radius:9px;font-size:10px;font-weight:700;
       margin-right:4px;padding:0 4px}
  .ord.alt{background:#fff;color:#94a3b8;border:1px solid #cbd5e1}
  .act-status{margin-right:3px;font-size:11px}
  .act.dangerous{background:#fff3f0}
  .act.dangerous .nm{color:#c44}
  .log{font-family:ui-monospace,monospace;font-size:11px;background:#fafafa;
    border:1px solid #eee;border-radius:6px;padding:8px;max-height:160px;overflow:auto;
    white-space:pre-wrap;word-break:break-all;color:#333}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;color:#888;font-weight:500;padding:4px 8px 4px 0;font-size:11px;text-transform:uppercase}
  td{padding:5px 8px 5px 0;border-top:1px solid #f0f0f0}
  td.r{text-align:right;font-variant-numeric:tabular-nums}
  .state-R{color:#2a7;font-weight:600}.state-PD{color:#c70}.state-CD{color:#06c}
  .nav{display:flex;gap:10px;font-size:13px;margin-bottom:12px;flex-wrap:wrap}
  .nav a{color:#06c;text-decoration:none;padding:4px 10px;border-radius:5px;background:#fff}
  .navhelp{color:#16a34a;font-size:11px;font-weight:700;cursor:help;position:relative;
           border:1px solid #bbf7d0;border-radius:50%;padding:0 4px;margin-left:2px;
           background:#f0fdf4}
  .navhelp:hover::after{content:attr(data-tip);position:absolute;left:0;top:150%;
           width:320px;max-width:80vw;white-space:normal;text-align:left;
           background:#0f172a;color:#fff;font-size:12px;font-weight:400;line-height:1.55;
           padding:9px 11px;border-radius:7px;box-shadow:0 6px 20px rgba(0,0,0,.3);
           z-index:1000;pointer-events:none}
  .navhelp:hover::before{content:"";position:absolute;left:8px;top:135%;
           border:6px solid transparent;border-bottom-color:#0f172a;z-index:1000}
  .nav a:hover{background:#eef4ff}
  .live-banner{background:linear-gradient(90deg,#2a7 0%,#28a 100%);color:#fff;
    padding:10px 16px;border-radius:8px;margin-bottom:14px;font-size:13px;
    display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  .live-banner.idle{background:linear-gradient(90deg,#888 0%,#aaa 100%)}
  .live-banner .pulse{width:8px;height:8px;background:#fff;border-radius:50%;
    animation:pulse 1.5s infinite}
  .live-banner.idle .pulse{animation:none;opacity:.5}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  /* ============================================================
     v3.0.72 (2026-06-01) — Phase E polish, additive overrides.
     Goal: 'super clean, professional + slick' single-page controller. Keeps
     the same selectors so existing JS works, just upgrades visuals.
  ============================================================ */
  body{background:linear-gradient(180deg,#f6f8fb 0%,#e9eef6 100%) !important;
       min-height:100vh;max-width:none !important;margin:0 !important;padding:0 !important}
  .body-inner{max-width:1400px;margin:0 auto;padding:1rem}
  /* Hero banner replaces plain h1 */
  .hero-bar{background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
            color:#fff;padding:1.6rem 2rem 1.4rem;
            box-shadow:0 4px 30px rgba(0,0,0,.15);margin-bottom:0;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem}
  .hero-bar h1{margin:0;color:#fff;font-size:1.75rem;font-weight:700;letter-spacing:-.5px}
  .hero-bar h1 span.v{color:#7dd3c0 !important;font-size:.85rem;font-weight:500;
        background:rgba(125,211,192,.15);padding:.2rem .55rem;border-radius:4px;
        margin-left:.5rem;letter-spacing:.5px}
  .hero-bar .live-indicator{display:inline-flex;align-items:center;gap:.5rem;
        background:rgba(255,255,255,.08);padding:.4rem .9rem;border-radius:20px;font-size:.85rem;color:#9aa5b8}
  .hero-bar .live-indicator .dot{width:8px;height:8px;background:#0a9b7a;
        border-radius:50%;box-shadow:0 0 8px rgba(10,155,122,.5);animation:pulse 1.8s infinite}
  /* Pill nav style */
  .nav{justify-content:center !important;padding:.6rem .8rem !important;
       background:transparent !important;border:none !important;margin-top:.8rem !important}
  .nav a{background:#fff !important;border:1px solid #e2e8f0 !important;
         padding:.4rem .85rem !important;border-radius:18px !important;
         color:#0e7c66 !important;font-weight:500 !important;font-size:13px !important;
         transition:all .15s !important;text-decoration:none !important}
  .nav a:hover{background:#0e7c66 !important;color:#fff !important;
               border-color:#0e7c66 !important;transform:translateY(-1px);
               box-shadow:0 4px 8px rgba(14,124,102,.2) !important}
  /* Stat cards with accent gradient on hover */
  .stat{transition:transform .12s,box-shadow .12s,border-color .12s;
        border-left:3px solid transparent;background:#fff !important;
        box-shadow:0 1px 4px rgba(0,0,0,.05) !important}
  .stat:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(0,0,0,.08) !important;
              border-left-color:#0e7c66}
  .stat .v{font-family:ui-monospace,Menlo,monospace !important;font-size:24px !important;
           color:#0f172a !important;font-weight:700 !important;letter-spacing:-.5px}
  .stat .v.green{color:#0e7c66 !important}
  /* Panels: deeper shadow + rounded edges */
  .panel{background:#fff !important;border-radius:14px !important;
         box-shadow:0 2px 8px rgba(15,23,42,.05),0 1px 3px rgba(15,23,42,.04) !important;
         border:1px solid #f0f4f8}
  .panel h2{font-size:15px !important;color:#0f172a !important;
            font-weight:600 !important;border-bottom:1px solid #f0f4f8;
            padding-bottom:.5rem !important;margin-bottom:.8rem !important}
  .panel h2 a{color:#0e7c66 !important;font-weight:500 !important}
  /* Action buttons: brighter on hover, gradient on dangerous */
  .act{border-radius:8px !important;transition:all .15s !important;
       border:1px solid #e2e8f0 !important;background:#f7f9fc !important}
  .act:hover{background:#fff !important;border-color:#0e7c66 !important;
             box-shadow:0 4px 12px rgba(14,124,102,.12) !important;
             transform:translateY(-1px)}
  .act .nm{color:#0e7c66 !important;font-weight:600 !important}
  .act.dangerous{background:linear-gradient(135deg,#fff5f5 0%,#fee 100%) !important;
                 border-color:#fecaca !important}
  .act.dangerous .nm{color:#c53030 !important}
  /* Live banner deeper gradient */
  .live-banner{background:linear-gradient(135deg,#0e7c66 0%,#0a9b7a 100%) !important;
               box-shadow:0 4px 12px rgba(14,124,102,.25) !important;
               border-radius:10px !important}
  .live-banner.idle{background:linear-gradient(135deg,#64748b 0%,#94a3b8 100%) !important;
                    box-shadow:0 4px 12px rgba(100,116,139,.2) !important}
  /* SLURM state badges */
  .state-R{background:#f0fff4 !important;color:#0e7c66 !important;
           padding:1px 7px !important;border-radius:4px !important;
           font-size:11px !important;font-weight:600 !important}
  .state-PD{background:#fff8e0 !important;color:#c70 !important;
            padding:1px 7px !important;border-radius:4px !important;
            font-size:11px !important;font-weight:600 !important}
  /* Smooth fade-in for table updates */
  table tbody tr{transition:background-color .2s}
  /* Log block monospace polish */
  .log{font-size:12px !important;border-radius:8px !important;
       background:#0f172a !important;color:#7dd3c0 !important;
       border:1px solid #1e293b !important;padding:12px !important}
  /* Footer */
  .footer{padding:1.5rem 1rem 2rem;text-align:center;font-size:12px;color:#94a3b8}
  .footer a{color:#0e7c66 !important;text-decoration:none;margin:0 .35rem}
  /* Hover ripple on stats and panels */
  .panel,.stat{will-change:transform,box-shadow}
  /* Count-up animation hook */
  .stat .v.flash{animation:flash .4s ease-out}
  @keyframes flash{0%{transform:scale(1)}50%{transform:scale(1.08);color:#0a9b7a !important}100%{transform:scale(1)}}
</style>
</head><body>
<div class="hero-bar">
  <h1>🌱 Weed-detection framework controller <span class="v">v3.0.79 · Mongo Phase 1</span></h1>
  <div class="live-indicator"><span class="dot"></span> auto-refresh ON</div>
</div>
<div class="body-inner">
<div class="nav">
  <a href="/">🏠 hub <span class="navhelp" data-tip="Home command center. Stat cards (slugs/images/Roboflow/storage backend) + 27 agent action buttons + live SLURM queue + Roboflow panel + per-species pipeline stats, all auto-refreshing. The entry page for every operation.">ⓘ</span></a>
  <a href="/manual" style="background:linear-gradient(135deg,#fef3c7,#fde68a) !important;color:#c70 !important;border-color:#fbbf24 !important;font-weight:600 !important">📖 manual <span class="navhelp" data-tip="Manual page. Dual-agent architecture diagram + the full 8-stage data pipeline + what each button does + current real progress + an honest data status (e.g. 8/12 species missing training data). Docs for new users and the professor.">ⓘ</span></a>
  <a href="/rounds" style="background:linear-gradient(135deg,#dbeafe,#bfdbfe) !important;color:#1d4ed8 !important;border-color:#60a5fa !important;font-weight:600 !important">🔄 rounds <span class="navhelp" data-tip="Review by harvest round. Each harvest round = a versioned data snapshot; the round's datasets are grouped by class name / bbox status, and you triage each one with ✓ keep / ✗ delete / 🔄 relabel. Humans promote good data to gold.">ⓘ</span></a>
  <a href="/classes">📋 classes <span class="navhelp" data-tip="Weed class browser. One card per class name + real image thumbnails; filter & search by topic (weed/disease/pest/crop); click a card for detail to see all images of the class, change its topic, or mark images as exemplars. The only page where you review images.">ⓘ</span></a>
  <a href="/slugs">📦 slugs <span class="navhelp" data-tip="Dataset-level cleanup. One row per dataset (slug) showing class names / image count / status; mark ✓ keep / ✗ junk / 🤔 unsure for whole-dataset triage. ✗ datasets are hidden from /classes and excluded from training. Faster than per-image review.">ⓘ</span></a>
  <a href="/roboflow">📊 roboflow <span class="navhelp" data-tip="Our Roboflow project status (shows only our 4: cwd12 gold + agent v1/v2/v3; unrelated old projects filtered out). Images/boxes/classes per project + a link to Roboflow to draw boxes for precise labeling.">ⓘ</span></a>
  <a href="/annotate">🏷️ annotate <span class="navhelp" data-tip="Labeling guide: for each collected dataset → whether its classes are real names / numeric placeholders / generic / CWD12, and how to label it (just verify / re-upload with names / judge species per box / label each image). Answers 'how do I label this messy data, and what's in v2/v3/v4'.">ⓘ</span></a>
  <a href="/labeling">🎯 labeling <span class="navhelp" data-tip="Labeling console (the professor's human-in-the-loop): for each dataset you decide how many images to push to Roboflow for human labeling (agent recommends a sample) → label → export back to the cluster → delete to save quota → push more. Mongo tracks counts (collected / agent-labeled / human-labeled / human-verified) + history.">ⓘ</span></a>
  <a href="/api/cluster_status">📥 JSON <span class="navhelp" data-tip="Raw status API (machine-readable JSON) that the dashboard panels poll: job queue, registry stats, ollama, verdict counts, etc. For developer debugging; not needed in normal use.">ⓘ</span></a>
</div>

<div class="live-banner idle" id="live-banner">
  <span class="pulse"></span>
  <span id="live-banner-text">loading status…</span>
</div>

<!-- ───── STAT ROW ───── -->
<div class="stat-row">
  <div class="stat"><div class="l">📦 Registry slugs</div><div class="v" id="stat-slugs">…</div></div>
  <div class="stat"><div class="l">⬇️ Downloaded</div><div class="v green" id="stat-downloaded">…</div></div>
  <div class="stat"><div class="l">🖼️ Total images</div><div class="v" id="stat-imgs">…</div></div>
  <div class="stat"><div class="l">🏷️ Topic overrides</div><div class="v" id="stat-topic">…</div></div>
  <div class="stat"><div class="l">📡 Roboflow imgs</div><div class="v green" id="stat-rf-imgs">…</div></div>
  <div class="stat"><div class="l">📐 Roboflow boxes</div><div class="v green" id="stat-rf-boxes">…</div></div>
  <div class="stat" title="MongoDB migration Phase 1 — is tools.db serving from Mongo or the JSON fallback?">
    <div class="l">🗄️ Storage backend</div>
    <div class="v" id="stat-db-backend">…</div>
    <div class="l" id="stat-db-detail" style="margin-top:2px;font-size:.7rem">…</div>
  </div>
</div>

<!-- ───── ACTIONS ───── -->
<div class="panel" style="margin-bottom:14px">
  <h2>🤖 Agent actions <span><a href="/control">/control full controls →</a></span></h2>
  <div style="font-size:11px;color:#666;margin:-4px 0 8px;line-height:1.7">
    <span class="ord">1</span> green = <b>recommended order</b> (click by number) ·
    <span class="ord alt">1</span> gray = alternatives/variants ·
    <b>no number</b> = tools · <b>④</b> = human (draw boxes in Roboflow) ·
    status ⏳ running ✅ success ❌ failed
  </div>
  <div class="actions" id="actions">loading…</div>
  <h2 style="margin-top:10px">📜 last action output</h2>
  <div class="log" id="action-log">click an action above to fire it</div>
</div>

<!-- ───── PANELS: SLURM | Roboflow ───── -->
<div class="panels">
  <div class="panel">
    <h2>📋 SLURM queue <span id="squeue-time">…</span></h2>
    <table>
      <thead><tr><th>jobid</th><th>name</th><th>st</th><th>time</th><th>reason</th></tr></thead>
      <tbody id="squeue-body"><tr><td colspan="5" style="color:#888">loading…</td></tr></tbody>
    </table>
  </div>
  <div class="panel">
    <h2>📡 Roboflow workspace <span><a href="/roboflow">details →</a></span></h2>
    <div id="rf-summary">loading…</div>
  </div>
</div>

<div class="panel" style="margin-bottom:14px">
  <h2>🌿 CWD12 species snapshot (Roboflow boxes per class)</h2>
  <div id="cwd12-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px">loading…</div>
</div>

<!-- v3.0.72 Phase D — per-species pipeline stats panel -->
<div class="panel" style="margin-bottom:14px">
  <h2>📊 Per-species pipeline state <span><a href="/api/per_species_stats">JSON →</a></span></h2>
  <div style="font-size:11px;color:#888;margin-bottom:6px">
    auto-refresh every 30s · gold = Roboflow cwd12 verified · auto = downloaded with YOLO labels ·
    unlabeled = single-class slug with no label files · owl = OWL red proposals · exemplars = object_bank samples
  </div>
  <div id="per-species" style="overflow-x:auto">loading…</div>
</div>

<div class="panel">
  <h2>🐛 Recent agent task runs <span><a href="/api/recent_jobs">JSON →</a></span></h2>
  <div id="recent-jobs">loading…</div>
</div>

<script>
function fmtN(n){return (n==null)?'?':n.toLocaleString()}

async function loadStatus(){
  try{
    const d = await (await fetch('/api/cluster_status')).json();
    const reg = d.registry || {};
    document.getElementById('stat-slugs').textContent = fmtN(reg.n_slugs);
    document.getElementById('stat-downloaded').textContent = fmtN(reg.n_downloaded);
    document.getElementById('stat-imgs').textContent = fmtN(reg.total_imgs);
    document.getElementById('stat-topic').textContent = fmtN(d.n_topic_overrides);
    // live banner
    const jobs = d.jobs || [];
    const agentJob = jobs.find(j => j.state==='RUNNING' &&
      !['v3030_dS'].includes(j.name||''));
    const banner = document.getElementById('live-banner');
    const txt = document.getElementById('live-banner-text');
    if(agentJob){
      banner.classList.remove('idle');
      txt.innerHTML = `🤖 <strong>${agentJob.name}</strong> RUNNING (${agentJob.time}) on ${agentJob.nodelist||'?'}`;
    }else{
      banner.classList.add('idle');
      txt.innerHTML = `💤 idle (no agent jobs) — registry ${reg.n_downloaded}/${reg.n_slugs} downloaded`;
    }
    // squeue table
    const tb = document.getElementById('squeue-body');
    if(!jobs.length){
      tb.innerHTML = '<tr><td colspan="5" style="color:#888">empty</td></tr>';
    }else{
      tb.innerHTML = jobs.map(j =>
        `<tr><td>${j.jobid}</td><td>${j.name||'?'}</td>`
        +`<td class="state-${j.state||'?'}">${j.state||'?'}</td>`
        +`<td>${j.time||'?'}</td><td>${j.nodelist||'?'}</td></tr>`).join('');
    }
    document.getElementById('squeue-time').textContent =
      new Date().toLocaleTimeString();
  }catch(e){
    document.getElementById('live-banner-text').innerHTML = 'cluster_status err: ' + e;
  }
}

// v3.0.96: canonical pipeline. Each action → {n: step label, k: 'p'(primary,
// green = the recommended one button for that step) | 'a'(alt/variant, gray)}.
// Sub-steps use letters (5a→5b→5c, 7a→7b). Utilities have no entry. Step ④ is
// human (Roboflow labeling) — no button. See /manual.
const ACTION_ORDER = {
  brain_harvest:{n:'1',k:'p'},            // ① collect (primary)
  start_new_round:{n:'1',k:'a'},          //   variant: just create round/RF project
  harvest_full_round_e2e:{n:'1·3',k:'a'}, //   variant: one-click collect+sync
  audit_registry_garbage:{n:'2',k:'p'},   // ② clean (dry-run primary)
  audit_registry_garbage_APPLY:{n:'2',k:'a'}, //   variant: actually delete
  build_buckets:{n:'2',k:'a'},            //   variant: bucket audit
  sync_newest_slugs:{n:'3',k:'p'},        // ③ upload to Roboflow (primary)
  sync_all_to_roboflow:{n:'3',k:'a'},
  roboflow_sync_cwd12_v1:{n:'3',k:'a'},
  roboflow_sync_agent_v1:{n:'3',k:'a'},
  roboflow_move_agent_to_folder:{n:'3',k:'a'},
  export_owl_exemplars:{n:'5a',k:'p'},    // ⑤ OWL auto-label (3 sub-steps)
  owl_preannotate_one:{n:'5b',k:'p'},
  owl_upload_proposals:{n:'5c',k:'p'},
  dinov2_curate_registry:{n:'6',k:'p'},   // ⑥ DINOv2 quality (primary)
  dinov2_filter_round_1:{n:'6',k:'a'},
  dinov2_route_classes:{n:'6',k:'a'},
  roboflow_generate_versions:{n:'7a',k:'p'}, // ⑦ pull back (2 sub-steps)
  roboflow_download_merge:{n:'7b',k:'p'},
  train_yolo_round_1:{n:'8',k:'p'},       // ⑧ train
};
function ordBadge(k){
  const e = ACTION_ORDER[k];
  if(!e) return '';
  return `<span class="ord${e.k==='a'?' alt':''}">${e.n}</span>`;
}
function stSpan(k){ return `<span class="act-status" data-st-for="${k}"></span>`; }

async function loadActions(){
  try{
    const acts = await (await fetch('/api/cluster_actions')).json();
    const html = Object.entries(acts).map(([k,v]) => {
      const danger = k.includes('restart') || k.includes('download_known') ? ' dangerous' : '';
      const label = (v.label||'').slice(0,90);
      // v3.0.68: brain_harvest gets an inline form (time / strict / max_new)
      if(k === 'brain_harvest'){
        return `<div class="act" data-action="brain_harvest" style="background:#fff8e0;border-color:#c70;padding:10px">
          <div class="nm" style="color:#c70">${ordBadge('brain_harvest')}${stSpan('brain_harvest')}🧠 brain_harvest</div>
          <div class="ds" style="margin-bottom:6px">${label}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;font-size:11px;color:#444;align-items:center">
            <label>Duration:
              <select id="bh-time" style="font-size:11px">
                <option value="1">1h</option>
                <option value="2">2h</option>
                <option value="4" selected>4h</option>
                <option value="6">6h</option>
              </select>
            </label>
            <label>max_new:
              <input id="bh-maxnew" type="number" value="5" min="1" max="50" style="width:48px;font-size:11px">
            </label>
            <label style="display:flex;align-items:center;gap:3px">
              <input id="bh-strict" type="checkbox" checked> strict weed/crop
            </label>
            <button onclick="triggerBrain()" style="background:#c70;color:#fff;border:0;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px">▶ run</button>
          </div></div>`;
      }
      return `<button class="act${danger}" data-action="${k}" onclick="trigger('${k}')">
                <div class="nm">${ordBadge(k)}${stSpan(k)}${k}</div><div class="ds">${label}</div></button>`;
    }).join('');
    document.getElementById('actions').innerHTML = html;
    loadActionStatus();  // paint real per-button status right after render
  }catch(e){
    document.getElementById('actions').innerHTML = 'actions err: ' + e;
  }
}

async function trigger(name, body){
  if(name === 'restart_dashboard' &&
     !confirm('Restart dashboard? This page will reconnect in ~90s (github.io auto-redirects to the new URL).')) return;
  const log = document.getElementById('action-log');
  // v3.0.99.47: per-button feedback so SSH-backed actions (2-5s) never feel "hung".
  const btn = document.querySelector('[data-action="'+name+'"]');
  let orig = null;
  if(btn && !btn.dataset.busy){
    orig = btn.textContent; btn.dataset.busy='1';
    btn.disabled = true; btn.style.opacity='0.6'; btn.textContent='⏳ Running…';
  }
  log.textContent = `→ triggering ${name} …`;
  try{ log.scrollIntoView({behavior:'smooth', block:'nearest'}); }catch(e){}
  const restore = (mark)=>{ if(btn && orig!==null){ btn.textContent=mark+orig;
    setTimeout(()=>{ btn.textContent=orig; btn.disabled=false; btn.style.opacity='';
      delete btn.dataset.busy; }, 2500); } };
  try{
    const opts = {method:'POST'};
    if(body){
      opts.headers = {'Content-Type':'application/json'};
      opts.body = JSON.stringify(body);
    }
    const r = await fetch('/api/cluster_action/' + name, opts);
    const d = await r.json();
    log.textContent = JSON.stringify(d, null, 2);
    restore(d && d.ok===false ? '❌ ' : '✅ ');
    if(name !== 'restart_dashboard') loadStatus();
  }catch(e){ log.textContent = 'error: ' + e; restore('❌ '); }
}

// v3.0.68: brain_harvest form-driven trigger
async function triggerBrain(){
  const body = {
    time_h: parseInt(document.getElementById('bh-time').value, 10),
    max_new: parseInt(document.getElementById('bh-maxnew').value, 10),
    strict: document.getElementById('bh-strict').checked,
  };
  return trigger('brain_harvest', body);
}

async function loadRoboflow(){
  try{
    const d = await (await fetch('/api/roboflow_status')).json();
    if(!d.ok){
      document.getElementById('rf-summary').innerHTML =
        '<span style="color:#c44">RF err: '+d.error+'</span>';
      return;
    }
    const master = d.projects.find(p => p.role === 'cwd12_master');
    if(master){
      document.getElementById('stat-rf-imgs').textContent = fmtN(master.images);
      document.getElementById('stat-rf-boxes').textContent = fmtN(master.boxes_total);
    }
    let html = `<div style="font-size:12px;color:#666;margin-bottom:8px">`
      +`workspace <code>${d.workspace}</code> · ${d.n_projects} projects (weed_crop_agent_dataset)</div>`;
    if(master){
      const ann = master.annotated||0, tot = master.total||0;
      const annPct = Math.max(0,Math.min(100,master.annotated_pct||0));
      html += `<div style="background:#f4faf4;border-left:3px solid #38a169;padding:8px 12px;border-radius:4px;font-size:13px">`
        + `<strong>${master.slug}</strong> (main project)<br>`
        + `📐 ${master.n_classes} classes · 📦 ${master.boxes_total} boxes<br>`
        + `🏷️ annotated ${ann}/${tot} (${annPct}%) · ⏳ ${master.pending||0} pending · 🗂️ ${master.versions} versions`
        + `</div>`;
    }
    // v3.0.91: list ALL our projects (incl. agent v1/v2/v3) so the folder's
    // contents are visible — unrelated workspace projects are already filtered.
    const roleLbl = {cwd12_master:'🥇 gold', agent:'🤖 agent', cwd12_species:'species', other:''};
    html += `<div style="margin-top:8px;font-size:12px">`
      + d.projects.map(p => `<div style="display:flex;justify-content:space-between;`
        + `padding:4px 8px;border-bottom:1px solid #f0f0f0">`
        + `<span>${p.error?'⚠️ ':''}<strong>${p.slug}</strong> `
        + `<span style="color:#999">${roleLbl[p.role]||''}</span></span>`
        + `<span style="color:#666">${p.error?p.error:('📷 '+(p.images||0)+' · 📦 '+(p.boxes_total||0)+' boxes')}</span>`
        + `</div>`).join('')
      + `</div>`;
    // CWD12 per-class breakdown
    if(master && master.boxes_per_class){
      const cwd12 = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass","Morningglory","Nutsedge",
        "PalmerAmaranth","PricklySida","Purslane","Ragweed","Sicklepod","SpottedSpurge"];
      const cls = master.boxes_per_class || {};
      const max = Math.max(...Object.values(cls).concat([1]));
      const cwd12html = cwd12.map(sp => {
        const n = cls[sp]||0; const pct = (100*n/max).toFixed(0);
        return `<div style="background:#fff;border:1px solid #eee;border-radius:6px;padding:6px 10px;font-size:11px">
                  <div style="font-weight:600">${sp}</div>
                  <div style="color:#666">${n} boxes</div>
                  <div style="background:#e0e0e6;height:4px;border-radius:2px;margin-top:4px">
                    <div style="background:#38a169;width:${pct}%;height:100%;border-radius:2px"></div>
                  </div>
                </div>`;
      }).join('');
      document.getElementById('cwd12-row').innerHTML = cwd12html;
    }
    document.getElementById('rf-summary').innerHTML = html;
  }catch(e){
    document.getElementById('rf-summary').innerHTML = 'RF err: ' + e;
  }
}

async function loadRecentJobs(){
  try{
    const d = await (await fetch('/api/recent_jobs?n=12')).json();
    const jobs = d.jobs || [];
    if(!jobs.length){
      document.getElementById('recent-jobs').innerHTML =
        '<div style="color:#888;font-size:12px">no jobs yet</div>';
      return;
    }
    document.getElementById('recent-jobs').innerHTML =
      '<table><thead><tr><th>jobid</th><th>name</th><th>finished</th><th class="r">size</th><th>log</th></tr></thead><tbody>'
      + jobs.map(j =>
        `<tr><td>${j.jobid}</td><td>${j.name}</td><td>${j.mtime_h}</td>`
        +`<td class="r">${(j.size/1024).toFixed(1)} KB</td>`
        +`<td><a href="/api/job_log/${j.jobid}?tail=200" target="_blank">📜</a></td></tr>`).join('')
      + '</tbody></table>';
  }catch(e){
    document.getElementById('recent-jobs').innerHTML = 'recent-jobs err: ' + e;
  }
}

// v3.0.72 Phase D — per-species pipeline state
async function loadPerSpecies(){
  try {
    const r = await fetch('/api/per_species_stats');
    const d = await r.json();
    const per = d.per_species || {};
    const totals = d.totals || {};
    const cwd12 = ["Carpetweeds","Crabgrass","Eclipta","Goosegrass","Morningglory","Nutsedge",
                   "PalmerAmaranth","PricklySida","Purslane","Ragweed","Sicklepod","SpottedSpurge"];
    // Find max for bar scaling
    const allVals = [];
    for(const sp of cwd12){
      const r = per[sp] || {};
      allVals.push(r.gold||0, r.auto||0, r.unlabeled||0, r.owl||0);
    }
    const maxVal = Math.max(...allVals, 1);
    const bar = (v,color) => {
      const pct = (100*v/maxVal).toFixed(1);
      return `<div style="display:flex;align-items:center;gap:4px">
        <span style="min-width:32px;font-variant-numeric:tabular-nums;font-size:11px;color:#333">${v}</span>
        <div style="flex:1;background:#eef2f6;height:5px;border-radius:3px;min-width:24px">
          <div style="background:${color};width:${pct}%;height:100%;border-radius:3px"></div>
        </div></div>`;
    };
    let html = '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    html += '<thead><tr style="background:#f7f9fc;color:#555">'
         + '<th style="text-align:left;padding:6px 10px;font-weight:600">Species</th>'
         + '<th style="text-align:left;padding:6px 10px;font-weight:600">🟢 gold</th>'
         + '<th style="text-align:left;padding:6px 10px;font-weight:600">⚙️ auto-on-download</th>'
         + '<th style="text-align:left;padding:6px 10px;font-weight:600">⚪ unlabeled</th>'
         + '<th style="text-align:left;padding:6px 10px;font-weight:600">🎯 OWL proposed</th>'
         + '<th style="text-align:left;padding:6px 10px;font-weight:600">🧬 exemplars</th>'
         + '</tr></thead><tbody>';
    for(const sp of cwd12){
      const r = per[sp] || {gold:0,auto:0,unlabeled:0,owl:0,exemplars:0};
      html += `<tr style="border-top:1px solid #eee">
        <td style="padding:5px 10px;font-weight:600;color:#333">${sp}</td>
        <td style="padding:5px 10px">${bar(r.gold||0,'#38a169')}</td>
        <td style="padding:5px 10px">${bar(r.auto||0,'#3b82f6')}</td>
        <td style="padding:5px 10px">${bar(r.unlabeled||0,'#9ca3af')}</td>
        <td style="padding:5px 10px">${bar(r.owl||0,'#ef4444')}</td>
        <td style="padding:5px 10px">${bar(r.exemplars||0,'#a855f7')}</td>
      </tr>`;
    }
    html += `<tr style="background:#f7f9fc;font-weight:700;border-top:2px solid #ddd">
      <td style="padding:6px 10px">Total</td>
      <td style="padding:6px 10px;color:#38a169">${totals.gold||0}</td>
      <td style="padding:6px 10px;color:#3b82f6">${totals.auto||0}</td>
      <td style="padding:6px 10px;color:#888">${totals.unlabeled||0}</td>
      <td style="padding:6px 10px;color:#ef4444">${totals.owl||0}</td>
      <td style="padding:6px 10px;color:#a855f7">${totals.exemplars||0}</td>
    </tr>`;
    html += '</tbody></table>';
    document.getElementById('per-species').innerHTML = html;
  } catch(e){
    document.getElementById('per-species').innerHTML =
      '<span style="color:#c44">per_species err: '+e+'</span>';
  }
}

// v3.0.79 — MongoDB Phase 1 storage-backend card
async function loadDbStatus(){
  const elB = document.getElementById('stat-db-backend');
  const elD = document.getElementById('stat-db-detail');
  if(!elB) return;
  try {
    const r = await fetch('/api/db_status');
    const d = await r.json();
    if(d.available){
      elB.innerHTML = '<span style="color:#16a34a">🟢 Mongo</span>';
      const c = d.counts || {};
      elD.textContent = `${d.registry_datasets??c.slugs??0} slugs · ${c.classes??0} classes`;
      elD.title = d.url || '';
    } else {
      elB.innerHTML = '<span style="color:#d97706">🟡 JSON</span>';
      elD.textContent = `${d.registry_datasets??0} slugs (fallback)`;
      elD.title = d.error || 'mongo unavailable';
    }
  } catch(e){
    elB.innerHTML = '<span style="color:#c44">err</span>';
    elD.textContent = String(e).slice(0,40);
  }
}

// v3.0.95: paint each action button with its REAL last-run status (from P0
// /api/action_history): ⏳ running / ✅ succeeded / ❌ failed / 🚀 launched.
async function loadActionStatus(){
  try{
    const d = await (await fetch('/api/action_history?n=80')).json();
    const latest = {};
    for(const ev of (d.history||[])){ if(ev.action) latest[ev.action] = ev.status || 'unknown'; }
    const icon = {succeeded:'✅', running:'⏳', failed:'❌', launched:'🚀', unknown:''};
    document.querySelectorAll('.act-status').forEach(el=>{
      const a = el.dataset.stFor; const st = latest[a];
      el.textContent = st ? (icon[st]||'') : '';
      el.title = st ? ('Last run: '+st) : 'never run';
    });
  }catch(e){}
}

// Initial load + poll
loadStatus(); loadActions(); loadRoboflow(); loadRecentJobs(); loadPerSpecies(); loadDbStatus();
setInterval(loadStatus, 6000);
setInterval(loadRecentJobs, 15000);
setInterval(loadRoboflow, 60000);
setInterval(loadPerSpecies, 30000);
setInterval(loadDbStatus, 20000);
setInterval(loadActionStatus, 12000);
</script>

</div><!-- /.body-inner -->
<div class="footer">
  <div style="margin-bottom:.4rem">
    v3.0.72 · all panels auto-refresh · 📖 see
    <a href="/manual">/manual</a> for the full pipeline guide
  </div>
  <div style="opacity:.7">
    Legacy:
    <a href="/dashboard/index.html">stats</a> ·
    <a href="/dashboard/datasets.html">datasets</a> ·
    <a href="/dashboard/categories.html">categories</a> ·
    <a href="/dashboard/progress.html">progress</a>
    &nbsp;|&nbsp; API:
    <a href="/api/state">state</a> ·
    <a href="/api/per_species_stats">per_species_stats</a> ·
    <a href="/api/exemplars_export">exemplars</a> ·
    <a href="/api/slug_verdicts">verdicts</a> ·
    <a href="/healthz">healthz</a>
  </div>
</div>
</body></html>''')


@app.get("/dashboard/{page}")
def dashboard_page(page: str):
    if not re.match(r"^[A-Za-z0-9_.-]+$", page):
        raise HTTPException(400, "bad page name")
    fp = DOCS_DIR / page
    if not fp.is_file():
        # try regenerate on the fly
        try:
            from weed_optimizer_framework.tools.dashboard_generator import (
                build_index, build_datasets, build_categories, build_progress,
            )
            state = get_state()
            mapping = {
                "index.html":      ("Home",  build_index),
                "datasets.html":   ("Datasets", build_datasets),
                "categories.html": ("Categories", build_categories),
                "progress.html":   ("Progress", build_progress),
            }
            if page in mapping:
                _, fn = mapping[page]
                html = fn(state)
                # Rewrite static image links to /api/sample/{slug}/{file}
                html = html.replace('src="samples/', 'src="/api/sample/')
                html = html.replace('href="samples/', 'href="/api/sample/')
                return HTMLResponse(html)
        except Exception as e:
            log.error(f"page regen failed: {e}")
        raise HTTPException(404, f"page not found: {page}")
    text = fp.read_text()
    # Rewrite static image paths to live endpoint
    text = text.replace('src="samples/', 'src="/api/sample/')
    text = text.replace('href="samples/', 'href="/api/sample/')
    return HTMLResponse(text)


@app.get("/api/state")
def api_state():
    return JSONResponse(get_state())


@app.get("/api/sample/{slug}/{filename}")
def api_sample(slug: str, filename: str):
    """On-demand bbox-rendered thumbnail. Disk-cached."""
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    if not re.match(r"^[A-Za-z0-9_. -]+\.(jpg|jpeg|png|JPG|JPEG|PNG)$", filename):
        raise HTTPException(400, "bad filename")

    # Cache key includes slug + filename so updates don't collide
    key = hashlib.sha1(f"{slug}/{filename}".encode()).hexdigest()
    cache_p = CACHE_DIR / f"{key}.jpg"
    if cache_p.is_file():
        # Serve from cache; cluster-side TTL via mtime check could be added later
        return FileResponse(cache_p, media_type="image/jpeg")

    # Find the actual image
    found = find_image_in_slug(slug, filename)
    if not found:
        raise HTTPException(404, f"not found: {slug}/{filename}")
    img_path, slug_root = found
    label_path = find_label_for_image(img_path, slug_root)

    # Class names for this slug
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        info = reg["datasets"].get(slug, {})
    except Exception:
        info = {}
    if slug in ("cottonweed_sp8", "cottonweed_holdout", "cottonweeddet12"):
        class_names = ["Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass",
                       "Morningglory", "Nutsedge", "PalmerAmaranth",
                       "PricklySida", "Purslane", "Ragweed", "Sicklepod",
                       "SpottedSpurge"]
    else:
        class_names = info.get("class_names") or ["target"]

    ok = render_with_bbox(img_path, label_path, cache_p, slug,
                          max_width=600, class_names=class_names)
    if not ok:
        # v3.0.126: degrade gracefully instead of 500. Community/manual uploads
        # may carry unusual formats/sizes that the bbox renderer can't process;
        # show the raw image (no overlay) rather than a broken thumbnail.
        if os.path.isfile(img_path):
            ext = os.path.splitext(img_path)[1].lower()
            mt = "image/png" if ext == ".png" else "image/jpeg"
            return FileResponse(img_path, media_type=mt)
        raise HTTPException(500, "render failed")
    return FileResponse(cache_p, media_type="image/jpeg")


@app.get("/api/img/{slug}/{filename}")
def api_img(slug: str, filename: str):
    """Original full-resolution image (no bbox overlay). For zoom-in.

    v3.0.54 (auto-loop iter 14 / Phase E3): refactored to use the
    storage abstraction (`tools/storage.py`). LustreBackend mirrors the
    previous find_image_in_slug behavior exactly, so this is behavior-
    preserving — but path resolution is now in one place, ready for the
    Uni-server migration (per Prof Zhang directive, see
    `docs/mongodb_schema.md`)."""
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    if not re.match(r"^[A-Za-z0-9_. -]+\.(jpg|jpeg|png|JPG|JPEG|PNG)$", filename):
        raise HTTPException(400, "bad filename")
    # v3.0.57 (auto-loop iter 17): LustreBackend now reads registry's
    # local_path (see storage.py v3.0.56), so it handles canonical slugs
    # without the find_image_in_slug bare-fallback. We keep the fallback
    # only for the catastrophic case of the storage module being
    # unimportable — any backend-level miss now means "not found", not
    # "try the legacy path".
    try:
        from weed_optimizer_framework.tools.storage import default_backend
        img_path = default_backend().get_image_path(slug, filename)
    except Exception as e:
        log.warning(f"/api/img storage backend unavailable: {e!r} — fallback")
        found = find_image_in_slug(slug, filename)
        img_path = str(found[0]) if found else None
    if not img_path:
        raise HTTPException(404)
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/api/slug/{slug}/samples")
def api_slug_samples(slug: str, n: int = 12, offset: int = 0):
    """Return list of filenames for this slug's sample images.

    v3.0.35.1: added `offset` for pagination + raised `n` cap (was 12).
    `n` clamped at 1000 to prevent huge JSON. For all-image listing use the
    /gallery/{slug} HTML endpoint which paginates server-side.
    """
    n = min(max(int(n), 1), 1000)
    offset = max(int(offset), 0)
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        local = reg["datasets"][slug].get("local_path")
    except Exception:
        raise HTTPException(404)
    if not local or not os.path.isdir(local):
        raise HTTPException(404)
    files = []
    skipped = 0
    seen = set()
    for p in sorted(Path(local).rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            # Dedup by filename (cwd12 has copies in train/test/valid subdirs)
            if p.name in seen:
                continue
            seen.add(p.name)
            if skipped < offset:
                skipped += 1
                continue
            files.append(p.name)
            if len(files) >= n:
                break
    return {"slug": slug, "samples": files, "offset": offset,
            "returned": len(files)}


@app.get("/api/slug/{slug}/count")
def api_slug_count(slug: str):
    """Total unique image count for this slug (for gallery pagination)."""
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        local = reg["datasets"][slug].get("local_path")
    except Exception:
        raise HTTPException(404)
    if not local or not os.path.isdir(local):
        raise HTTPException(404)
    seen = set()
    for p in Path(local).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            seen.add(p.name)
    return {"slug": slug, "total_unique": len(seen)}


@app.get("/gallery/{slug}", response_class=HTMLResponse)
def gallery(slug: str, page: int = 1, per_page: int = 24):
    """v3.0.35.1: Full-gallery view of all images in a slug, paginated.

    Each thumbnail is the bbox-rendered version (uses /api/sample/...).
    Click → opens the original full-res image via /api/img/...
    """
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    per_page = min(max(int(per_page), 1), 200)
    page = max(int(page), 1)
    offset = (page - 1) * per_page

    # Get registry info
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        info = reg["datasets"].get(slug, {})
        local = info.get("local_path")
    except Exception:
        raise HTTPException(404)
    if not local or not os.path.isdir(local):
        raise HTTPException(404, f"slug {slug} has no local_path")

    # Collect all unique filenames
    all_files = []
    seen = set()
    for p in sorted(Path(local).rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            if p.name not in seen:
                seen.add(p.name)
                all_files.append(p.name)
    total = len(all_files)
    n_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, n_pages)
    offset = (page - 1) * per_page
    page_files = all_files[offset:offset + per_page]

    # Build HTML
    annot = info.get("annotation", "?")
    src = info.get("source", "?")
    cards = []
    for fname in page_files:
        thumb_url = f"/api/sample/{slug}/{fname}"
        full_url  = f"/api/img/{slug}/{fname}"
        cards.append(f'''
        <div class="card">
          <a href="{full_url}" target="_blank">
            <img src="{thumb_url}" alt="{fname}" loading="lazy"/>
          </a>
          <div class="caption">{fname}</div>
        </div>''')

    # Pagination links
    def page_link(p):
        return f'<a href="/gallery/{slug}?page={p}&per_page={per_page}">{p}</a>'
    nav_pages = []
    # Show: 1, prev, current-2..current+2, next, last
    show_set = {1, n_pages, page}
    for p in range(max(1, page-2), min(n_pages, page+2)+1):
        show_set.add(p)
    sorted_pages = sorted(show_set)
    prev_p = None
    nav = []
    for p in sorted_pages:
        if prev_p is not None and p - prev_p > 1:
            nav.append('<span class="gap">…</span>')
        if p == page:
            nav.append(f'<span class="current">{p}</span>')
        else:
            nav.append(page_link(p))
        prev_p = p
    nav_html = ' '.join(nav)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{slug} — gallery (page {page}/{n_pages})</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f7; color: #333; }}
  header {{ background: #fff; padding: 16px 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  header h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  header .meta {{ color: #666; font-size: 14px; }}
  header a {{ color: #06c; text-decoration: none; }}
  header a:hover {{ text-decoration: underline; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: transform 0.15s; }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.12); }}
  .card img {{ width: 100%; height: 180px; object-fit: cover; display: block; background: #eee; }}
  .card .caption {{ padding: 8px 10px; font-size: 11px; color: #888; word-break: break-all; }}
  nav.pagination {{ text-align: center; margin: 24px 0; }}
  nav.pagination a, nav.pagination span {{ display: inline-block; padding: 6px 12px; margin: 0 3px; border-radius: 6px; font-size: 14px; }}
  nav.pagination a {{ background: #fff; color: #06c; text-decoration: none; border: 1px solid #ddd; }}
  nav.pagination a:hover {{ background: #06c; color: #fff; }}
  nav.pagination .current {{ background: #06c; color: #fff; font-weight: 600; }}
  nav.pagination .gap {{ color: #999; }}
</style>
</head>
<body>
<div style="margin:0 0 12px"><a href="/slugs" style="display:inline-block;text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Back to Datasets</a></div>
<header>
  <h1>📂 {slug}</h1>
  <div class="meta">
    {total} unique images · page {page}/{n_pages} ({per_page}/page)
    · annotation: <code>{annot}</code>
    · source: <code>{src}</code>
    · <a href="/slugs">← back to all datasets</a>
  </div>
</header>

<nav class="pagination">{nav_html}</nav>

<div class="grid">
  {''.join(cards) if cards else '<p>No images on this page.</p>'}
</div>

<nav class="pagination">{nav_html}</nav>
</body>
</html>'''
    return HTMLResponse(html)


# ----------------------------------------------------------------------------
# v3.0.39.x: synthetic-image galleries — direct file serving (no registry slug)
# Serves images generated by synth_cutpaste.py and synth_diffusion.py so the
# user can visually evaluate synthetic quality directly from the dashboard.
# ----------------------------------------------------------------------------
_SYNTH_KINDS = {
    "flux":      REPO / "results" / "framework" / "synth_diffusion" / "images",
    "cutpaste":  REPO / "results" / "framework" / "synth_cutpaste"  / "images",
}


@app.get("/synth/raw/{kind}/{filename}")
def synth_raw_image(kind: str, filename: str):
    """Serve a single synthetic image file (full-resolution)."""
    if kind not in _SYNTH_KINDS:
        raise HTTPException(404, f"unknown synth kind {kind!r}")
    if not re.match(r"^[A-Za-z0-9_.-]+\.(jpg|jpeg|png|JPG|JPEG|PNG)$", filename):
        raise HTTPException(400, "bad filename")
    p = _SYNTH_KINDS[kind] / filename
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/synth/{kind}", response_class=HTMLResponse)
def synth_gallery(kind: str, page: int = 1, per_page: int = 36):
    """Paginated gallery of synthetic images (FLUX or cut-paste)."""
    if kind not in _SYNTH_KINDS:
        raise HTTPException(404, f"unknown synth kind {kind!r} — "
                                  f"try {list(_SYNTH_KINDS)}")
    per_page = min(max(int(per_page), 1), 200)
    page = max(int(page), 1)
    src_dir = _SYNTH_KINDS[kind]
    if not src_dir.is_dir():
        return HTMLResponse(
            f"<h2>No synthetic images for {kind} yet</h2>"
            f"<p>Expected directory: <code>{src_dir}</code></p>"
            f"<p>Once the corresponding job (v3.0.39 FLUX or v3.0.38 cut-paste) "
            f"writes images here, refresh this page.</p>")

    files = sorted(
        (p.name for p in src_dir.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")),
    )
    total = len(files)
    n_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, n_pages)
    offset = (page - 1) * per_page
    page_files = files[offset:offset + per_page]

    # also surface the corresponding YOLO labels count (sanity check that
    # synth has GT, not just images) and a sample montage if present
    labels_dir = src_dir.parent / "labels"
    n_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.is_dir() else 0
    montage = src_dir.parent / "sample_montage.jpg"
    montage_card = ""
    if montage.is_file() and offset == 0:
        montage_card = f'''
        <div class="card montage">
          <a href="/synth/raw/{kind}/../sample_montage.jpg" target="_blank">
            <img src="/synth/montage/{kind}" alt="contact sheet"/>
          </a>
          <div class="caption">📊 sample montage (6×6 contact sheet)</div>
        </div>'''

    cards = []
    for fname in page_files:
        url = f"/synth/raw/{kind}/{fname}"
        cards.append(f'''
        <div class="card">
          <a href="{url}" target="_blank">
            <img src="{url}" alt="{fname}" loading="lazy"/>
          </a>
          <div class="caption">{fname}</div>
        </div>''')

    # pagination
    def page_link(p):
        return f'<a href="/synth/{kind}?page={p}&per_page={per_page}">{p}</a>'
    show = sorted({1, n_pages, page} |
                  set(range(max(1, page-2), min(n_pages, page+2)+1)))
    nav = []
    prev = None
    for p in show:
        if prev is not None and p - prev > 1:
            nav.append('<span class="gap">…</span>')
        nav.append(f'<span class="current">{p}</span>' if p == page
                   else page_link(p))
        prev = p
    nav_html = ' '.join(nav)

    kind_pretty = {"flux": "FLUX.1-Fill (v3.0.39 layout-conditioned)",
                   "cutpaste": "Cut-paste (v3.0.38 GT-anchored)"}[kind]
    other = "cutpaste" if kind == "flux" else "flux"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>synth · {kind} (page {page}/{n_pages})</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f7; color: #333; }}
  header {{ background: #fff; padding: 16px 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  header h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  header .meta {{ color: #666; font-size: 14px; }}
  header a {{ color: #06c; text-decoration: none; }}
  header a:hover {{ text-decoration: underline; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: transform 0.15s; }}
  .card.montage {{ grid-column: 1 / -1; }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.12); }}
  .card img {{ width: 100%; height: 240px; object-fit: cover; display: block; background: #eee; }}
  .card.montage img {{ height: auto; max-height: 800px; object-fit: contain; }}
  .card .caption {{ padding: 8px 10px; font-size: 11px; color: #666; word-break: break-all; }}
  nav.pagination {{ text-align: center; margin: 24px 0; }}
  nav.pagination a, nav.pagination span {{ display: inline-block; padding: 6px 12px; margin: 0 3px; border-radius: 6px; font-size: 14px; }}
  nav.pagination a {{ background: #fff; color: #06c; text-decoration: none; border: 1px solid #ddd; }}
  nav.pagination a:hover {{ background: #06c; color: #fff; }}
  nav.pagination .current {{ background: #06c; color: #fff; font-weight: 600; }}
  nav.pagination .gap {{ color: #999; }}
</style>
</head>
<body>
<header>
  <h1>🧪 Synthetic images · {kind_pretty}</h1>
  <div class="meta">
    <strong>{total}</strong> images · {n_labels} YOLO label files · page {page}/{n_pages}
    · <a href="/synth/{other}">switch to {other}</a>
    · <a href="/">← dashboard home</a>
  </div>
</header>

<nav class="pagination">{nav_html}</nav>

<div class="grid">
  {montage_card}{''.join(cards) if cards else '<p>No images on this page.</p>'}
</div>

<nav class="pagination">{nav_html}</nav>
</body>
</html>'''
    return HTMLResponse(html)


@app.get("/synth/montage/{kind}")
def synth_montage(kind: str):
    """Serve the 6x6 sample_montage.jpg produced by the synth scripts."""
    if kind not in _SYNTH_KINDS:
        raise HTTPException(404)
    p = _SYNTH_KINDS[kind].parent / "sample_montage.jpg"
    if not p.is_file():
        raise HTTPException(404, "no montage yet")
    return FileResponse(str(p), media_type="image/jpeg")




# ----------------------------------------------------------------------------
# v3.0.41 Phase 0 — /audit gallery (REDESIGNED 2026-05-25 per user feedback)
# The original /audit served pre-baked 6×N montage JPEGs which crushed
# per-image quality and lacked per-class navigation. This rewrite serves
# every image individually at native resolution and organises navigation
# around the 12 cwd12 species so a reviewer can answer, for each species:
#   1. What does the ORIGINAL cwd12 labelled data look like?
#   2. What did we DO to FLUX for this class (prompt, training, etc.)?
#   3. What did FLUX OUTPUT for this class? Does it match the species?
# ----------------------------------------------------------------------------
import re as _re_audit
from typing import List as _List_audit, Tuple as _Tuple_audit

# v3.0.99.52: class names legitimately contain spaces / dashes / dots / parens
# ("Carpet weed", "Crab Grass", "grass weeds - v2 release"). The old
# ^[A-Za-z0-9_]+$ guard 400'd every such class → the WHOLE view+review workflow
# (detail page /classes/{cls}, /thumb_reg, exemplar POST, exemplars_export) was
# unreachable for multi-word weed classes the agent actually collected — they
# showed on the landing but clicking them returned 400. Relax to an allowlist
# that still bars the only dangerous chars: path separators (/ \) and HTML/quote
# chars (< > "). cls is used ONLY as a single path component (dir/{cls}.jsonl),
# so with no separator no traversal is possible; a bare '..' is a harmless stem.
_RE_CLS_OK = _re_audit.compile(r"^[\w .()\-'&,+]{1,128}$")


def _cls_ok(cls: str) -> bool:
    """Validate a class-name path param (allows spaces; bars separators/HTML)."""
    return bool(cls) and _RE_CLS_OK.match(cls) is not None

_CWD12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]
_CWD12_ZH = {
    "Carpetweeds": "毯草", "Crabgrass": "马唐", "Eclipta": "鳢肠",
    "Goosegrass": "蟋蟀草", "Morningglory": "牵牛花", "Nutsedge": "莎草",
    "PalmerAmaranth": "苋菜 / 帕氏苋", "PricklySida": "刺苋",
    "Purslane": "马齿苋", "Ragweed": "豚草", "Sicklepod": "决明",
    "SpottedSpurge": "斑地锦",
}

# ----------------- registry class index (canonical -> [(slug, cid, raw)]) ----
import re as _re_canon

# v3.0.113: explicit same-species synonyms (alnum-lowercased key → canonical
# class). Consolidates near-duplicates the harvester created from differently
# worded source labels, e.g. "Carpet weed" vs CWD12 "Carpetweeds". Keep this
# list HIGH-CONFIDENCE only (true same species), not fuzzy guesses.
_CLASS_SYNONYMS = {
    "carpetweed": "Carpetweeds",          # → CWD12 canonical
    "weeds": "Weed",                       # generic plural → generic singular
    "grassweedsv2release": "GrassWeeds",   # Roboflow version-suffix leaked into name
    "partheniumhysterophorous": "Parthenium",
}


def _canon_class(raw: str) -> str:
    """Normalize species name from registry to canonical form.
    Matches CWD12 case+punctuation-insensitive; else PascalCase the input."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    alnum = _re_canon.sub(r'[^A-Za-z0-9]', '', raw).lower()
    if not alnum:
        return ""
    if alnum in _CLASS_SYNONYMS:           # v3.0.113 same-species merge
        return _CLASS_SYNONYMS[alnum]
    for c12 in _CWD12:
        if _re_canon.sub(r'[^A-Za-z0-9]', '', c12).lower() == alnum:
            return c12
    parts = _re_canon.split(r'[^A-Za-z0-9]+', raw)
    return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)


# v3.0.43.23 (user request): pseudo-classes that are NOT real species —
# they're PlantVillage image-VARIANT folders (color/grayscale/segmented) or
# junk folder names mis-extracted as class_names by the harvester. They have
# no meaningful per-class thumbnail and pollute /classes. Filtered out of the
# class index entirely. Matched on lowercased-alphanumeric form so case and
# punctuation don't matter. Source data stays on disk; only the bogus
# "class" is hidden.
_JUNK_CLASS_ALNUM = frozenset({
    "color", "grayscale", "segmented", "nonsegmentedv2",
    "cropimages", "somemoreimages", "testcropimage",
})


def _is_junk_class(canon: str) -> bool:
    """True if `canon` is a known non-species pseudo-class to hide."""
    if not canon:
        return False
    # v3.0.113: pure-numeric / single-char names are leftover YOLO class indices
    # (e.g. "0","1","2","3") — never a real species. Hide them from /classes.
    s = canon.strip()
    if s.isdigit() or len(s) <= 1:
        return True
    return _re_canon.sub(r'[^A-Za-z0-9]', '', canon).lower() in _JUNK_CLASS_ALNUM


_registry_index_cache: dict = {}   # v3.0.109: per-domain {domain: {ts, index, empty_slugs}}
_REG_INDEX_TTL = float(os.environ.get("REG_INDEX_TTL_SEC", "15"))

def _load_registry_index(domain: str = "weed") -> dict:
    """Returns {canon_class_name: [(slug, class_id, raw_name), ...]} for `domain`.
    Side-effect populates the per-domain cache's 'empty_slugs' (slugs with local
    data but missing class_names) — surfaces metadata-gap.

    v3.0.109: domain-scoped + per-domain cache (default "weed" = unchanged).
    Source is `db.get_registry(domain=...)` (Mongo authoritative, JSON fallback
    treats untagged slugs as the weed default). Short-TTL cached per domain so a
    single /classes render never re-queries 355×."""
    now = time.time()
    ce = _registry_index_cache.get(domain)
    if ce and (now - ce["ts"]) < _REG_INDEX_TTL and ce["index"]:
        return ce["index"]
    try:
        from . import db as _db
        reg = _db.get_registry(domain=domain)
    except Exception:
        # last-resort direct file read (should rarely happen — db has its own
        # JSON fallback already)
        try:
            with open(REGISTRY_PATH) as f:
                reg = json.load(f)
        except Exception:
            return (ce or {}).get("index", {})
    idx: dict = {}
    empty: list = []
    # v3.0.114: don't nag about junk-verdicted slugs in the metadata-gap banner
    # (a slug the user marked ✗ junk shouldn't demand a backfill).
    try:
        _vd = _slug_verdict_state()
    except Exception:
        _vd = {}
    for slug, info in (reg.get("datasets") or {}).items():
        if _vd.get(slug) == "junk":
            continue
        cn = info.get("class_names") or []
        lp = info.get("local_path") or ""
        has_local = bool(lp and os.path.isdir(lp))
        if not cn:
            if has_local:
                empty.append(slug)
            continue
        if not has_local:
            continue
        for cid, raw in enumerate(cn):
            canon = _canon_class(raw)
            if not canon:
                continue
            if _is_junk_class(canon):   # v3.0.43.23: hide non-species pseudo-classes
                continue
            idx.setdefault(canon, []).append((slug, cid, raw))
    _registry_index_cache[domain] = {"ts": now, "index": idx, "empty_slugs": empty}
    return idx


def _registry_empty_slugs(domain: str = "weed") -> list:
    _load_registry_index(domain)
    return list((_registry_index_cache.get(domain) or {}).get("empty_slugs", []))


_pool_cache_dir = REPO / "results" / "framework" / "cache" / "class_pool"
_pool_cache_dir.mkdir(parents=True, exist_ok=True)


def _scandir_subdirs(d: Path, cap: int = 600) -> list:
    """Return subdirectories of `d`, scanning at most `cap` entries. A flat
    image dir (no subdirs) bails after `cap` os.scandir entries instead of
    enumerating tens of thousands of files — this is the bound that keeps
    folder discovery from stat-storming on Lustre."""
    subs: list = []
    try:
        with os.scandir(d) as it:
            n = 0
            for e in it:
                n += 1
                if n > cap:
                    break
                try:
                    if e.is_dir():
                        subs.append(Path(e.path))
                except Exception:
                    pass
    except Exception:
        pass
    return subs


def _find_class_folder(lpp: Path, raw: str, max_depth: int = 3,
                       max_visit: int = 400):
    """Bounded BFS for a directory named `raw` (case-insensitive) under lpp.
    v3.0.43.22e: replaces the brittle hardcoded-wrapper list (Dataset/,
    PlantVillage/, data/train/, Rice_Image_Dataset/, …). Real slugs nest
    class folders under arbitrary wrapper names (PlantDiseasesDataset/Apple,
    Agricultural-crops/almond, BangladeshiCrops/BangladeshiCrops/Crop___Disease/
    Potato). BFS descends only into container dirs via _scandir_subdirs (so a
    flat image dir is cheap), capped at max_visit dir-stats total."""
    raw_l = raw.lower()
    from collections import deque
    q = deque([(lpp, 0)])
    visited = 0
    while q and visited < max_visit:
        d, depth = q.popleft()
        subs = _scandir_subdirs(d)
        for sd in subs:
            visited += 1
            if sd.name.lower() == raw_l:
                return sd
            if visited >= max_visit:
                return None
        if depth < max_depth:
            # descend breadth-first into the first 40 container dirs
            for sd in subs[:40]:
                q.append((sd, depth + 1))
    return None


def _find_label_dirs(local_p: Path, max_dirs: int = 64) -> list:
    """Return YOLO `labels/` directories under local_p WITHOUT a full-tree
    rglob. v3.0.43.22: the old code did `local_p.rglob('*.txt')` which on
    classification slugs (108K-434K images, NO labels/ at all) walked the
    entire Lustre tree and took 70-85s PER CLASS — prewarm of 267 classes
    took 5-6 hours. Here we probe only the common bounded locations:

        local_p/labels
        local_p/<split>/labels          (train/valid/test/...)
        local_p/<wrapper>/<split>/labels (2-level nesting)

    A classification slug returns [] instantly, so the caller skips it."""
    # v3.0.43.22b: NEVER iterate a dir full of image files. The first cut
    # descended into every depth-1 child, so a flat {lp}/{class}/*.jpg
    # classification slug triggered tens of thousands of .is_dir() stats on
    # Lustre (AppleAppleScab took 5 min). Now: depth-1 checks only
    # {child}/labels (one stat each), and we descend a level deeper ONLY
    # into known split/wrapper dirs (train/valid/data/dataset/…) which hold
    # a handful of subdirs, never raw images.
    _SPLIT_HINTS = {
        "train", "training", "valid", "validation", "val", "test", "testing",
        "images", "data", "dataset", "yolo", "splits", "obj_train_data",
        "obj_val_data", "obj_test_data",
    }
    found: list = []
    d0 = local_p / "labels"
    if d0.is_dir():
        found.append(d0)
    try:
        for child in local_p.iterdir():
            if not child.is_dir() or child.name == "labels":
                continue
            d1 = child / "labels"
            if d1.is_dir():
                found.append(d1)
                if len(found) >= max_dirs:
                    return found
            elif child.name.lower() in _SPLIT_HINTS:
                # cheap descent: split dirs contain few subdirs, not raw imgs
                try:
                    for gchild in child.iterdir():
                        if gchild.is_dir():
                            d2 = gchild / "labels"
                            if d2.is_dir():
                                found.append(d2)
                                if len(found) >= max_dirs:
                                    return found
                except Exception:
                    pass
    except Exception:
        pass
    return found


def _reg_pool_for_class(cls: str, per_slug_cap: int = 200,
                         include_junk: bool = False,
                         max_scan_per_slug: int = 4000) -> list:
    """For each slug containing `cls`, sample up to per_slug_cap images whose
    label has a bbox of that class_id. Cached on disk by registry mtime.

    v3.0.43: by default exclude slugs marked '✗ junk' via /slugs UI.
    Pass include_junk=True to override.

    v3.0.43.22c: max_scan_per_slug bounds how many label .txt files we READ
    per slug before giving up — a per-class-folder detection slug
    (kg_karagwaanntreasure: 23 folders × ~1-2K txt = 20K+ files) otherwise
    made every disease class read tens of thousands of files. The landing
    thumb no longer calls this at all (it uses the class-folder name); this
    cap only protects the class-detail view."""
    idx = _load_registry_index()
    entries = idx.get(cls, [])
    if not entries:
        return []
    if not include_junk:
        verdicts = _slug_verdict_state()
        entries = [(s, c, r) for s, c, r in entries if verdicts.get(s) != "junk"]
        if not entries:
            return []
    try:
        reg_mtime = int(REGISTRY_PATH.stat().st_mtime) if REGISTRY_PATH.exists() else 0
    except Exception:
        reg_mtime = 0
    cache_fp = _pool_cache_dir / f"{cls}.json"
    if cache_fp.is_file():
        try:
            cached = json.loads(cache_fp.read_text())
            if cached.get("reg_mtime") == reg_mtime and cached.get("cap") == per_slug_cap:
                return cached.get("entries", [])
        except Exception:
            pass
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
    except Exception:
        return []
    out: list = []
    for slug, cid, _raw in entries:
        info = (reg.get("datasets") or {}).get(slug) or {}
        lp = info.get("local_path")
        if not lp or not os.path.isdir(lp):
            continue
        local_p = Path(lp)
        n_added = 0
        # v3.0.43.22: bounded label-dir probe — NO full-tree rglob. A
        # classification slug (no labels/ anywhere) yields [] instantly.
        label_dirs = _find_label_dirs(local_p)
        if not label_dirs:
            continue
        n_scanned = 0
        try:
            for ldir in label_dirs:
                if n_added >= per_slug_cap or n_scanned >= max_scan_per_slug:
                    break
                try:
                    lbls = sorted(ldir.glob("*.txt"))
                except Exception:
                    continue
                for lbl in lbls:
                    n_scanned += 1
                    if n_scanned >= max_scan_per_slug:
                        break
                    try:
                        txt = lbl.read_text(errors="ignore")
                    except Exception:
                        continue
                    has = False
                    for line in txt.splitlines():
                        p = line.split()
                        if p and p[0].isdigit() and int(p[0]) == cid:
                            has = True
                            break
                    if not has:
                        continue
                    img_path = None
                    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
                        cand = Path(str(lbl).replace("/labels/", "/images/")).with_suffix(ext)
                        if cand.is_file():
                            img_path = cand
                            break
                    if img_path is None:
                        continue
                    out.append({
                        "kind": "reg", "slug": slug,
                        "fname": img_path.name, "cid": cid,
                    })
                    n_added += 1
                    if n_added >= per_slug_cap:
                        break
        except Exception as e:
            log.warning(f"reg pool walk fail {slug}/{cls}: {e}")
    try:
        cache_fp.write_text(json.dumps({
            "reg_mtime": reg_mtime, "cap": per_slug_cap, "entries": out
        }))
    except Exception:
        pass
    return out

# FLUX setup as actually used in v3.0.39 (vanilla text-to-image).
# Mirrors weed_optimizer_framework/tools/synth_diffusion.SPECIES_PROMPT.
_FLUX_SPECIES_PROMPT = {
    "Carpetweeds":    "a carpetweed plant, small green sprawling weed",
    "Crabgrass":      "a crabgrass plant, spreading grassy weed",
    "Eclipta":        "an eclipta weed plant, green leaves",
    "Goosegrass":     "a goosegrass plant, flat rosette grassy weed",
    "Morningglory":   "a morningglory weed, heart-shaped leaves vine",
    "Nutsedge":       "a nutsedge plant, upright grass-like weed",
    "PalmerAmaranth": "a palmer amaranth pigweed plant, broadleaf weed",
    "PricklySida":    "a prickly sida weed plant, broadleaf",
    "Purslane":       "a purslane plant, fleshy red-stemmed weed",
    "Ragweed":        "a ragweed plant, lobed green leaves",
    "Sicklepod":      "a sicklepod weed plant, paired oval leaflets",
    "SpottedSpurge":  "a spotted spurge plant, low mat-forming weed",
}
_FLUX_PROMPT_SUFFIX = (", top-down view, cotton field soil background, daylight, "
                       "photorealistic, sharp focus")

# Filesystem roots
_BANK_DIR = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
_FLUX_IMG_DIR = REPO / "results" / "framework" / "synth_diffusion" / "images"
_FLUX_LBL_DIR = REPO / "results" / "framework" / "synth_diffusion" / "labels"


def _bank_files(cls: str) -> _List_audit[str]:
    d = _BANK_DIR / cls
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def _flux_files_for_class(cls: str) -> _List_audit[_Tuple_audit[str, list]]:
    """List of (filename, [list of (cx,cy,bw,bh) bboxes of this class])."""
    if cls not in _CWD12 or not _FLUX_LBL_DIR.is_dir():
        return []
    cid = _CWD12.index(cls)
    out = []
    for lbl in sorted(_FLUX_LBL_DIR.glob("*.txt")):
        try:
            content = lbl.read_text().splitlines()
        except Exception:
            continue
        boxes = []
        for line in content:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                lcid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            if lcid == cid:
                boxes.append((cx, cy, bw, bh))
        if boxes:
            img_name = lbl.stem + ".jpg"
            if (_FLUX_IMG_DIR / img_name).is_file():
                out.append((img_name, boxes))
    return out


# ---------------------------------------------------------------- raw serving
@app.get("/audit/raw/bank/{cls}/{filename}")
def audit_raw_bank(cls: str, filename: str):
    if not _cls_ok(cls):
        raise HTTPException(400, "bad class name")
    if not _re_audit.match(r"^[A-Za-z0-9_.-]+\.(png|jpg|jpeg|PNG|JPG|JPEG)$", filename):
        raise HTTPException(400, "bad filename")
    p = _BANK_DIR / cls / filename
    if not p.is_file():
        raise HTTPException(404)
    ext = p.suffix.lower()
    media = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(str(p), media_type=media)


@app.get("/audit/raw/flux/{filename}")
def audit_raw_flux(filename: str):
    if not _re_audit.match(r"^[A-Za-z0-9_.-]+\.(jpg|jpeg|JPG|JPEG)$", filename):
        raise HTTPException(400, "bad filename")
    p = _FLUX_IMG_DIR / filename
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="image/jpeg")


# ---------------------------------------------------------------- HTML shell
_AUDIT_CSS = """
  body { font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
         margin: 0; padding: 24px; background: #f4f4f7; color: #1c1c1e; }
  header { background: #fff; padding: 20px 24px; border-radius: 12px;
           margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }
  header h1 { margin: 0 0 4px 0; font-size: 22px; }
  header .sub { color: #666; font-size: 14px; }
  header a { color: #06c; text-decoration: none; }
  header a:hover { text-decoration: underline; }
  .crumbs { color: #888; font-size: 13px; margin-bottom: 10px; }
  .crumbs a { color: #06c; text-decoration: none; }
  section { background: #fff; padding: 20px 24px; border-radius: 12px;
            margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }
  section h2 { margin: 0 0 6px 0; font-size: 18px; }
  section .desc { color: #555; font-size: 13px; margin: 0 0 14px 0; }
  .grid-classes { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                  gap: 14px; }
  .class-card { background: #fafafa; border-radius: 10px; overflow: hidden;
                box-shadow: 0 1px 2px rgba(0,0,0,0.06); transition: transform 0.15s;
                text-decoration: none; color: inherit; display: block; }
  .class-card:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.12); }
  .class-card img { width: 100%; height: 160px; object-fit: cover; display: block; background: #eee; }
  .class-card .name { padding: 8px 12px 4px; font-weight: 600; font-size: 14px; }
  .class-card .zh { padding: 0 12px; color: #666; font-size: 12px; }
  .class-card .counts { padding: 4px 12px 10px; color: #888; font-size: 11px; font-family: ui-monospace, monospace; }
  .grid-imgs { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
               gap: 12px; }
  .img-card { background: #fafafa; border-radius: 8px; overflow: hidden;
              box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
  .img-card a { display: block; }
  .img-card img { width: 100%; height: 240px; object-fit: contain; background: #fff;
                  display: block; cursor: zoom-in; }
  .img-card .meta { padding: 6px 10px; color: #888; font-size: 11px;
                    font-family: ui-monospace, monospace; word-break: break-all; }
  .method-box { background: #fdf6e3; border-left: 4px solid #d4a017;
                padding: 14px 18px; border-radius: 6px; margin-bottom: 16px; }
  .method-box code { background: #fff7e0; padding: 2px 6px; border-radius: 3px;
                     font-size: 13px; }
  .method-row { display: flex; gap: 12px; margin-top: 6px; font-size: 14px; }
  .method-row .k { color: #666; min-width: 110px; }
  .method-row .v { color: #222; font-family: ui-monospace, monospace; }
"""


# ---------------------------------------------------------------- /audit landing
@app.get("/audit", response_class=HTMLResponse)
def audit_landing():
    cards = []
    total_bank = 0
    total_flux = 0
    for cls in _CWD12:
        bank = _bank_files(cls)
        flux = _flux_files_for_class(cls)
        total_bank += len(bank)
        total_flux += len(flux)
        # thumbnail = first bank image if available, else placeholder
        if bank:
            thumb = f"/audit/raw/bank/{cls}/{bank[0]}"
        else:
            thumb = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='160'%3E%3Crect width='220' height='160' fill='%23ccc'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23666' font-size='14'%3Eno samples%3C/text%3E%3C/svg%3E"
        zh = _CWD12_ZH.get(cls, "")
        cards.append(f'''
        <a class="class-card" href="/audit/class/{cls}">
          <img src="{thumb}" alt="{cls}" loading="lazy"/>
          <div class="name">{cls}</div>
          <div class="zh">{zh}</div>
          <div class="counts">bank {len(bank)}  ·  flux {len(flux)}</div>
        </a>''')

    html = f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audit — cwd12 数据 + FLUX 输出</title>
<style>{_AUDIT_CSS}</style>
</head><body>
<header>
  <h1>🔬 数据审计 — cwd12 真实数据 + FLUX 合成输出</h1>
  <div class="sub">
    <strong>{total_bank}</strong> 张真实 crop · <strong>{total_flux}</strong> 张 FLUX 合成
    含目标类 · <strong>{len(_CWD12)}</strong> 个物种
    · <a href="/audit/method">📘 方法论说明(FLUX 配置 / 训练 / 当前状态)</a>
    · <a href="/">← dashboard 首页</a>
  </div>
</header>
<section>
  <h2>按类浏览</h2>
  <p class="desc">点任一类卡片进入该类详情:左列是真实 cwd12 crop,右列是 FLUX 合成在该类的输出,
     底部说明对该类使用的 prompt 和方法。每张图都是原生分辨率,可点击放大。</p>
  <div class="grid-classes">{''.join(cards)}</div>
</section>
</body></html>'''
    return HTMLResponse(html)


# ---------------------------------------------------------------- /audit/method
@app.get("/audit/method", response_class=HTMLResponse)
def audit_method():
    rows = []
    for cls in _CWD12:
        prompt = _FLUX_SPECIES_PROMPT.get(cls, "") + _FLUX_PROMPT_SUFFIX
        rows.append(f"<tr><td><strong>{cls}</strong> ({_CWD12_ZH.get(cls,'')})</td>"
                    f"<td><code>{prompt}</code></td></tr>")
    table = "<table>" + "".join(rows) + "</table>"

    html = f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FLUX 方法论 / Methodology</title>
<style>{_AUDIT_CSS}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  td:first-child {{ width: 200px; }}
  code {{ background: #f4f4f7; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
  .ko {{ color: #c00; }}
  .ok {{ color: #080; }}
</style>
</head><body>
<header>
  <h1>📘 FLUX 方法论 / 当前状态</h1>
  <div class="sub"><a href="/audit">← back to audit</a></div>
</header>

<section>
  <h2>项目 FLUX 阶段汇总</h2>
  <p class="desc">截至当前所有 FLUX 尝试,以及对应 outcome。诚实记录,不掩盖失败。</p>
  <ul>
    <li><strong>v3.0.39</strong> vanilla text-to-image 文生图(本页 prompt 列表)—
        <span class="ko">视觉证伪</span>:文字 prompt 不足以让 FLUX 生成正确的
        cwd12 物种(Goosegrass 出仙人掌,Ragweed 出粉花等)。</li>
    <li><strong>v3.0.39.1</strong> 修 CUDA OOM,enable_model_cpu_offload —
        <span class="ko">速度 5min/张</span>,600 张不可行,killed @ 42 张。</li>
    <li><strong>v3.0.39.2</strong> BioCLIP 2 backbone 测试 —
        <span class="ko">分类头准确率 0.615</span>,反而比通用 DINOv2 (0.667) 差,假设证伪。</li>
    <li><strong>v3.0.39.3</strong> 2× V100 multi-GPU —
        <span class="ko">9 分钟失败</span>(multi-GPU placement)。</li>
    <li><strong>v3.0.39.4</strong> bank canonical mapping fix —
        <span class="ok">bank 12 类齐全</span>,但 Goosegrass 视觉仍可疑(用户审计中)。</li>
    <li><strong>v3.0.41 Phase 0</strong>(当前)bank rebuild SLURM job —
        <span class="ok">已完成</span>;Phase 1 (LoRA 微调)<strong>等用户审完 bank
        正确性后再启动</strong>。</li>
  </ul>
</section>

<section>
  <h2>FLUX 当前配置(v3.0.39 系列实际跑过的)</h2>
  <div class="method-box">
    <div class="method-row"><span class="k">模型:</span>
        <span class="v">black-forest-labs/FLUX.1-Fill-dev (bf16, ~24GB)</span></div>
    <div class="method-row"><span class="k">推理模式:</span>
        <span class="v">text-to-image with mask(伪 inpainting,mask 区域内重画)</span></div>
    <div class="method-row"><span class="k">LoRA 微调:</span>
        <span class="v ko">无(vanilla,即裸模型;这是为什么物种不正确)</span></div>
    <div class="method-row"><span class="k">分辨率:</span>
        <span class="v">768 × 768</span></div>
    <div class="method-row"><span class="k">推理步数:</span>
        <span class="v">28</span></div>
    <div class="method-row"><span class="k">guidance_scale:</span>
        <span class="v">30.0</span></div>
    <div class="method-row"><span class="k">类采样:</span>
        <span class="v">原计划弱类偏置(cwd12_class_counts.json),实测未生效 → 接近均匀</span></div>
  </div>
</section>

<section>
  <h2>当前每类使用的 prompt</h2>
  <p class="desc">每个 species 都拼接以下 suffix:<code>{_FLUX_PROMPT_SUFFIX}</code></p>
  {table}
</section>

<section>
  <h2>计划中(Phase 1 LoRA 微调,等用户审计 bank 通过后启动)</h2>
  <p>FLORA 论文 (arXiv 2508.21712) 的 recipe:</p>
  <ul>
    <li>每类 30 张干净 crop 作为 LoRA 训练集</li>
    <li>rank 32, alpha 16, 5 epochs, attention layers only</li>
    <li>8-bit AdamW, bfloat16, 512²,gradient checkpointing</li>
    <li>trigger 词:<code>cwd12-Goosegrass</code> 等(class-specific token)</li>
    <li>推理:在真实 cwd12 图上 mask 一个真实 bbox,LoRA + FLUX inpaint
        → 物种正确 + bbox 像素精确</li>
  </ul>
  <p><em>这一步还没启动,等用户在 /audit 上确认 bank 真实性正确后再做。</em></p>
</section>
</body></html>'''
    return HTMLResponse(html)


# ---------------------------------------------------------------- /audit/class/{name}
@app.get("/audit/class/{cls}", response_class=HTMLResponse)
def audit_class(cls: str):
    if cls not in _CWD12:
        raise HTTPException(404, f"unknown cwd12 class {cls!r}")
    bank = _bank_files(cls)
    flux = _flux_files_for_class(cls)
    cid = _CWD12.index(cls)
    zh = _CWD12_ZH.get(cls, "")
    prompt = _FLUX_SPECIES_PROMPT.get(cls, "") + _FLUX_PROMPT_SUFFIX

    bank_cards = "".join(
        f'<div class="img-card"><a href="/audit/raw/bank/{cls}/{fn}" target="_blank">'
        f'<img src="/audit/raw/bank/{cls}/{fn}" loading="lazy"/></a>'
        f'<div class="meta">{fn}</div></div>'
        for fn in bank)

    flux_cards = "".join(
        f'<div class="img-card"><a href="/audit/raw/flux/{fn}" target="_blank">'
        f'<img src="/audit/raw/flux/{fn}" loading="lazy"/></a>'
        f'<div class="meta">{fn} · {len(boxes)} bbox(es) of {cls}</div></div>'
        for fn, boxes in flux)

    html = f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cls} — audit</title>
<style>{_AUDIT_CSS}</style>
</head><body>
<header>
  <div class="crumbs"><a href="/audit">← 全部类</a></div>
  <h1>{cls} <span style="color:#888;font-size:18px;">/ {zh}</span></h1>
  <div class="sub">
    cwd12 class_id <code>{cid}</code>
    · <strong>{len(bank)}</strong> 张真实 crop
    · <strong>{len(flux)}</strong> 张 FLUX 输出含此类
  </div>
</header>

<section>
  <h2>① 原始 cwd12 数据(真实标注 bbox crop)</h2>
  <p class="desc">从 cwd12 真实标注 bbox 抠出的 {len(bank)} 张本类 crop。
     这是 LoRA 微调的训练源、分类头的训练源、以及合成对比的"ground truth"参考。
     点任一张可看原生 PNG。</p>
  <div class="grid-imgs">{bank_cards or '<p style="color:#888">(尚无该类 crop)</p>'}</div>
</section>

<section>
  <h2>② 对 FLUX 做了什么(本类配置)</h2>
  <div class="method-box">
    <div class="method-row"><span class="k">阶段:</span>
        <span class="v">v3.0.39 vanilla text-to-image(无 LoRA 微调)</span></div>
    <div class="method-row"><span class="k">FLUX 知道这个物种吗:</span>
        <span class="v">通用文生图,<strong>FLUX 没有在 cwd12 上微调过</strong> ——
        靠 prompt 文字推断。这是 v3.0.39 失败的根因。</span></div>
    <div class="method-row"><span class="k">所用 prompt:</span></div>
  </div>
  <pre style="background:#f4f4f7;padding:12px 16px;border-radius:6px;font-size:13px;overflow-x:auto;">{prompt}</pre>
  <p class="desc">下一步(Phase 1)计划:用上面这一段 ① 的真实 crop(30 张)训练
     class-specific LoRA,trigger token <code>cwd12-{cls}</code>,把 FLUX 教会这个物种。
     <a href="/audit/method">完整方法论 →</a></p>
</section>

<section>
  <h2>③ FLUX 输出(含本类 bbox 的合成图)</h2>
  <p class="desc">v3.0.39 阶段生成的 42 张合成图中,含 <strong>{cls}</strong> bbox 的有
     {len(flux)} 张。每张点开是原生分辨率,可视觉判断 FLUX 画出来的是不是真的 {cls}。</p>
  <div class="grid-imgs">{flux_cards or '<p style="color:#888">(无 FLUX 输出含本类 ——「弱类偏置」配置未生效,导致部分类未覆盖到)</p>'}</div>
</section>
</body></html>'''
    return HTMLResponse(html)


# ----------------------------------------------------------------------------
# v3.0.42 — /classes Roboflow-Lite human verification UI (2026-05-26)
#
# Per user direction:
#   "我们其实不管做FLUX还是其他的 最重要的问题是先确保该分类的数据集没问题…
#    每次新采集的数据集都按类分类… 可以点进类里面来人眼核实数据集的质量…
#    可以自己选择标注box… 一个小的roboflow私人版本… 最重要的就是这些榜样
#    数据集 因为如果没有确定的高质量对应分类数据集以及标注框的话我们后面的
#    所有都是白做"
#
# v1 (this commit) ships:
#   GET  /classes                — landing page, all known species
#   GET  /classes/{cls}          — per-class viewer with ✓ ✗ buttons + filters
#   POST /api/exemplar/{cls}     — record a verification verdict
#   GET  /api/exemplar/{cls}     — return current verdict map for class
#   GET  /thumb/{kind}/{cls}/{file}?w=256
#                                — cached thumbnail (huge speedup vs raw)
#
# v2 (later) will add in-browser bbox draw/correct.
# ----------------------------------------------------------------------------

import hashlib as _hl
import json as _json
import re as _re_cls
import time as _time_cls
from typing import Dict as _Dict_cls, List as _List_cls

_CLS_EXEMPLAR_DIR = REPO / "results" / "framework" / "class_exemplars"
_CLS_EXEMPLAR_DIR.mkdir(parents=True, exist_ok=True)

# v3.0.43: slug-level verdict log (separate from class-level exemplar log).
# Users can mark whole slugs as ✓ keep / ✗ junk / 🤔 unsure — used to hide
# garbage from /classes by default and speed up audit.
_SLUG_VERDICT_FILE = REPO / "results" / "framework" / "slug_verdicts.jsonl"
_SLUG_VERDICT_FILE.parent.mkdir(parents=True, exist_ok=True)


def _slug_verdict_state() -> dict:
    """Replay slug_verdicts.jsonl → {slug: latest_verdict}.
    verdict ∈ {keep, junk, unsure}. 'clear' removes the slug entry."""
    state: dict = {}
    if not _SLUG_VERDICT_FILE.is_file():
        return state
    try:
        for line in _SLUG_VERDICT_FILE.read_text().splitlines():
            if not line.strip():
                continue
            ev = _json.loads(line)
            slug = ev.get("slug", "")
            v = ev.get("verdict", "")
            if not slug or not v:
                continue
            if v == "clear":
                state.pop(slug, None)
            else:
                state[slug] = v
    except Exception:
        pass
    return state

_THUMB_DIR = REPO / "results" / "framework" / "cache" / "thumbs"
_THUMB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- thumbnails
def _thumb_path_for(kind: str, cls: str, fname: str, w: int) -> Path:
    safe = _re_cls.sub(r'[^A-Za-z0-9_.-]', '_', f"{kind}__{cls}__{fname}_{w}")
    return _THUMB_DIR / (safe + ".jpg")


def _source_for(kind: str, cls: str, fname: str) -> Optional[Path]:
    """Resolve the underlying source file for a thumbnail request.
    kind = 'bank'  → synth_cutpaste/object_bank/{cls}/{fname}
    kind = 'flux'  → synth_diffusion/images/{fname}  (cls ignored — just locates file)
    """
    if kind == "bank":
        p = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank" / cls / fname
        return p if p.is_file() else None
    if kind == "flux":
        p = REPO / "results" / "framework" / "synth_diffusion" / "images" / fname
        return p if p.is_file() else None
    return None


@app.get("/thumb/{kind}/{cls}/{filename}")
def thumb_serve(kind: str, cls: str, filename: str, w: int = 256):
    """Serve a cached small thumbnail (default 256px). Huge speedup vs raw on
    /ocean. Cache key includes mtime so source changes invalidate."""
    if kind not in ("bank", "flux"):
        raise HTTPException(400)
    if not _cls_ok(cls):
        raise HTTPException(400)
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+\.(png|jpg|jpeg|PNG|JPG|JPEG)$', filename):
        raise HTTPException(400)
    if not 64 <= w <= 1024:
        w = 256
    src = _source_for(kind, cls, filename)
    if src is None:
        raise HTTPException(404)
    cache = _thumb_path_for(kind, cls, filename, w)
    if (not cache.is_file()) or (cache.stat().st_mtime < src.stat().st_mtime):
        try:
            from PIL import Image as _Im
            im = _Im.open(src).convert("RGB")
            im.thumbnail((w, w), _Im.LANCZOS)
            tmp = cache.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=86, optimize=True)
            os.replace(tmp, cache)
        except Exception as e:
            log.warning(f"thumb generate fail {src}: {e}")
            return FileResponse(str(src), media_type="image/jpeg")
    return FileResponse(
        str(cache), media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},
    )


# -------- registry-source thumb / raw with target-class bbox highlighted -----
def _render_class_thumb(img_path: Path, label_path: Optional[Path],
                        target_cid: int, out: Path, max_width: int) -> bool:
    """Draw target class bboxes in red, others in dim gray, then thumbnail."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if label_path is not None:
        for cid, (cx, cy, bw, bh) in parse_yolo_labels(label_path):
            x1 = int(max(0, (cx - bw / 2) * w))
            y1 = int(max(0, (cy - bh / 2) * h))
            x2 = int(min(w, (cx + bw / 2) * w))
            y2 = int(min(h, (cy + bh / 2) * h))
            if cid == target_cid:
                color = (0, 0, 220); thick = max(3, w // 150)
            else:
                color = (140, 140, 140); thick = max(1, w // 400)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if w > max_width * 2:
        scale = (max_width * 2) / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    tmp = out.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    try:
        from PIL import Image as _Im
        im = _Im.open(tmp).convert("RGB")
        im.thumbnail((max_width, max_width), _Im.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, "JPEG", quality=86, optimize=True)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
    return True


@app.get("/thumb_reg/{slug}/{cls}/{filename}")
def thumb_reg_serve(slug: str, cls: str, filename: str, w: int = 256):
    """Cached thumb of a registry-slug image with target-class bbox in red."""
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400, "bad slug")
    if not _cls_ok(cls):
        raise HTTPException(400, "bad cls")
    if "/" in filename or ".." in filename or filename.startswith("."):
        raise HTTPException(400, "bad fname")
    if not 64 <= w <= 1024:
        w = 256
    idx = _load_registry_index()
    pair = next(((s, c) for s, c, _ in idx.get(cls, []) if s == slug), None)
    if pair is None:
        raise HTTPException(404, "class not in this slug")
    _, target_cid = pair
    found = find_image_in_slug(slug, filename)
    if found is None:
        raise HTTPException(404, "image not found")
    img_path, local_p = found
    safe = _re_cls.sub(r'[^A-Za-z0-9_.-]', '_', f"reg__{slug}__{cls}__{filename}_{w}")
    cache = _THUMB_DIR / (safe + ".jpg")
    if (not cache.is_file()) or (cache.stat().st_mtime < img_path.stat().st_mtime):
        lbl = find_label_for_image(img_path, local_p)
        ok = _render_class_thumb(img_path, lbl, target_cid, cache, w)
        if not ok:
            return FileResponse(str(img_path), media_type="image/jpeg")
    return FileResponse(
        str(cache), media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/raw_reg/{slug}/{cls}/{filename}")
def raw_reg_serve(slug: str, cls: str, filename: str):
    """Full-res image with target-class bbox highlighted (cached on disk)."""
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400)
    if not _cls_ok(cls):
        raise HTTPException(400)
    if "/" in filename or ".." in filename or filename.startswith("."):
        raise HTTPException(400)
    idx = _load_registry_index()
    pair = next(((s, c) for s, c, _ in idx.get(cls, []) if s == slug), None)
    if pair is None:
        raise HTTPException(404)
    _, target_cid = pair
    found = find_image_in_slug(slug, filename)
    if found is None:
        raise HTTPException(404)
    img_path, local_p = found
    lbl = find_label_for_image(img_path, local_p)
    out_dir = REPO / "results" / "framework" / "cache" / "reg_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _re_cls.sub(r'[^A-Za-z0-9_.-]', '_', f"{slug}__{cls}__{filename}")
    out = out_dir / (safe + ".jpg")
    if (not out.is_file()) or (out.stat().st_mtime < img_path.stat().st_mtime):
        ok = _render_class_thumb(img_path, lbl, target_cid, out, max_width=1600)
        if not ok:
            return FileResponse(str(img_path), media_type="image/jpeg")
    return FileResponse(str(out), media_type="image/jpeg")


# ---------------------------------------------------------------- exemplar log
def _exemplar_file(cls: str) -> Path:
    return _CLS_EXEMPLAR_DIR / f"{cls}.jsonl"


def _exemplar_state(cls: str) -> _Dict_cls[str, str]:
    """Replay the jsonl log -> {img_key: latest verdict}."""
    p = _exemplar_file(cls)
    state: dict[str, str] = {}
    if not p.is_file():
        return state
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = _json.loads(line)
                    if "img" in ev and "verdict" in ev:
                        state[ev["img"]] = ev["verdict"]
                except Exception:
                    continue
    except Exception:
        pass
    return state


@app.get("/api/exemplar/{cls}")
def api_exemplar_get(cls: str):
    if not _cls_ok(cls):
        raise HTTPException(400)
    return JSONResponse(_exemplar_state(cls))


@app.post("/api/exemplar/{cls}")
async def api_exemplar_post(cls: str, payload: dict = Body(...)):
    if not _cls_ok(cls):
        raise HTTPException(400)
    img = payload.get("img", "")
    verdict = payload.get("verdict", "")
    if not isinstance(img, str) or not isinstance(verdict, str):
        raise HTTPException(400, "missing img/verdict")
    if verdict not in ("exemplar", "bad", "rebox", "clear"):
        raise HTTPException(400, "verdict must be exemplar|bad|rebox|clear")
    if "/" in img and not _re_cls.match(r'^[A-Za-z0-9_./-]+$', img):
        raise HTTPException(400, "bad img path chars")
    ev = {
        "img": img, "verdict": verdict,
        "ts": _time_cls.time(),
        "ts_h": _time_cls.strftime("%Y-%m-%d %H:%M:%S UTC", _time_cls.gmtime()),
    }
    fp = _exemplar_file(cls)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "a") as f:
        f.write(_json.dumps(ev) + "\n")
    return JSONResponse({"ok": True, **ev})


@app.post("/api/exemplar_bulk/{cls}")
async def api_exemplar_bulk(cls: str, payload: dict = Body(...)):
    """v3.0.111: apply one verdict to MANY images at once (one-click select-all
    for a clean class). Body: {verdict, imgs:[key,...]}. Appends one event per
    img to the same jsonl the per-image endpoint uses (latest-wins replay)."""
    if not _cls_ok(cls):
        raise HTTPException(400)
    verdict = payload.get("verdict", "")
    imgs = payload.get("imgs", [])
    if verdict not in ("exemplar", "bad", "rebox", "clear"):
        raise HTTPException(400, "verdict must be exemplar|bad|rebox|clear")
    if not isinstance(imgs, list) or not imgs:
        raise HTTPException(400, "imgs must be a non-empty list")
    if len(imgs) > 20000:
        raise HTTPException(400, "too many imgs")
    ts = _time_cls.time()
    tsh = _time_cls.strftime("%Y-%m-%d %H:%M:%S UTC", _time_cls.gmtime())
    fp = _exemplar_file(cls)
    fp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(fp, "a") as f:
        for img in imgs:
            if not isinstance(img, str) or not img:
                continue
            if "/" in img and not _re_cls.match(r'^[A-Za-z0-9_./-]+$', img):
                continue
            f.write(_json.dumps({"img": img, "verdict": verdict,
                                 "ts": ts, "ts_h": tsh}) + "\n")
            n += 1
    return JSONResponse({"ok": True, "verdict": verdict, "count": n})


@app.post("/api/exemplar_markall/{cls}")
async def api_exemplar_markall(cls: str, payload: dict = Body(...)):
    """v3.0.118: mark the ENTIRE class pool with one verdict, server-side.
    Unlike /api/exemplar_bulk (which the client builds from rendered cards),
    this enumerates the full _class_image_pool on the server, so the detail
    page never has to render hundreds of cards just to "select all" — fixes the
    lag the user saw on big classes and guarantees the progress bar reaches
    100%. Body: {verdict}. Returns count actually written."""
    if not _cls_ok(cls):
        raise HTTPException(400)
    verdict = payload.get("verdict", "")
    if verdict not in ("exemplar", "bad", "clear"):
        raise HTTPException(400, "verdict must be exemplar|bad|clear")
    pool = _class_image_pool(cls)
    ts = _time_cls.time()
    tsh = _time_cls.strftime("%Y-%m-%d %H:%M:%S UTC", _time_cls.gmtime())
    fp = _exemplar_file(cls)
    fp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(fp, "a") as f:
        for entry in pool:
            try:
                key = _pool_entry_urls(entry, cls)[0]
            except Exception:
                key = None
            if not key or not isinstance(key, str):
                continue
            f.write(_json.dumps({"img": key, "verdict": verdict,
                                 "ts": ts, "ts_h": tsh}) + "\n")
            n += 1
    return JSONResponse({"ok": True, "verdict": verdict, "count": n})


# ---- exemplar EXPORT: closes the human-in-loop circle ----
# Human ✓ marks → exportable manifest → downstream LoRA / curator training.

def _exemplar_export_entry(cls: str, img_key: str) -> dict:
    """Resolve an exemplar key into a usable training entry.
    img_key formats:
      'bank/{cls}/{fname}'      → cropped object on transparent bg
      'flux/{fname}'            → FLUX synthetic full scene
      'reg/{slug}/{fname}'      → full real-bbox image
    Returns dict with kind, source_path (absolute), bbox_class_id (if reg),
    and direct URLs for the dashboard."""
    parts = img_key.split("/", 2)
    if len(parts) < 2:
        return {"key": img_key, "error": "malformed_key"}
    kind = parts[0]
    if kind == "bank" and len(parts) == 3:
        fn = parts[2]
        p = (REPO / "results" / "framework" / "synth_cutpaste" /
             "object_bank" / cls / fn)
        return {
            "key": img_key, "kind": "bank", "class": cls, "fname": fn,
            "path": str(p) if p.is_file() else None,
            "thumb_url": f"/thumb/bank/{cls}/{fn}?w=256",
            "raw_url": f"/audit/raw/bank/{cls}/{fn}",
        }
    if kind == "flux" and len(parts) >= 2:
        fn = parts[1] if len(parts) == 2 else parts[2]
        p = REPO / "results" / "framework" / "synth_diffusion" / "images" / fn
        return {
            "key": img_key, "kind": "flux", "class": cls, "fname": fn,
            "path": str(p) if p.is_file() else None,
            "thumb_url": f"/thumb/flux/{cls}/{fn}?w=256",
            "raw_url": f"/audit/raw/flux/{fn}",
        }
    if kind == "reg" and len(parts) == 3:
        slug = parts[1]
        fn = parts[2]
        # resolve class_id from registry index
        idx = _load_registry_index()
        pair = next(((s, c) for s, c, _ in idx.get(cls, []) if s == slug), None)
        cid = pair[1] if pair else None
        return {
            "key": img_key, "kind": "reg", "class": cls,
            "slug": slug, "fname": fn, "class_id_in_slug": cid,
            "thumb_url": f"/thumb_reg/{slug}/{cls}/{fn}?w=256",
            "raw_url": f"/raw_reg/{slug}/{cls}/{fn}",
        }
    return {"key": img_key, "error": "unknown_kind", "raw_kind": kind}


@app.get("/api/exemplars_export/{cls}")
def api_exemplars_export_class(cls: str):
    """Return all ✓ exemplar entries for a class as a usable manifest."""
    if not _cls_ok(cls):
        raise HTTPException(400)
    state = _exemplar_state(cls)
    entries = [
        _exemplar_export_entry(cls, k)
        for k, v in state.items() if v == "exemplar"
    ]
    return JSONResponse({
        "class": cls,
        "n_exemplars": len(entries),
        "n_bad": sum(1 for v in state.values() if v == "bad"),
        "n_rebox": sum(1 for v in state.values() if v == "rebox"),
        "exported_at": _time_cls.strftime(
            "%Y-%m-%d %H:%M:%S UTC", _time_cls.gmtime()),
        "entries": entries,
    })


@app.get("/api/exemplars_export")
def api_exemplars_export_all():
    """Full manifest: every class with at least one ✓ exemplar."""
    out: dict = {"by_class": {}, "total_exemplars": 0,
                 "exported_at": _time_cls.strftime(
                     "%Y-%m-%d %H:%M:%S UTC", _time_cls.gmtime())}
    for cls in _all_known_classes():
        state = _exemplar_state(cls)
        ex_keys = [k for k, v in state.items() if v == "exemplar"]
        if not ex_keys:
            continue
        out["by_class"][cls] = {
            "n_exemplars": len(ex_keys),
            "entries": [_exemplar_export_entry(cls, k) for k in ex_keys],
        }
        out["total_exemplars"] += len(ex_keys)
    out["n_classes_with_exemplars"] = len(out["by_class"])
    return JSONResponse(out)


# ---- slug-level verdicts: ✓ keep / ✗ junk / 🤔 unsure ----
@app.get("/api/slug_verdicts")
def api_slug_verdicts_all():
    """Return full {slug: verdict} map."""
    state = _slug_verdict_state()
    counts = {"keep": 0, "junk": 0, "unsure": 0}
    for v in state.values():
        counts[v] = counts.get(v, 0) + 1
    return JSONResponse({"verdicts": state, "counts": counts})


@app.post("/api/slug_verdict/{slug}")
async def api_slug_verdict_post(slug: str, payload: dict = Body(...)):
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400, "bad slug chars")
    verdict = payload.get("verdict", "")
    if verdict not in ("keep", "junk", "unsure", "clear"):
        raise HTTPException(400, "verdict must be keep|junk|unsure|clear")
    note = payload.get("note", "")
    if not isinstance(note, str):
        note = ""
    ev = {
        "slug": slug, "verdict": verdict,
        "note": note[:200],
        "ts": _time_cls.time(),
        "ts_h": _time_cls.strftime("%Y-%m-%d %H:%M:%S UTC", _time_cls.gmtime()),
    }
    with open(_SLUG_VERDICT_FILE, "a") as f:
        f.write(_json.dumps(ev) + "\n")
    # v3.0.116: lab-side writes get clobbered by the cluster→lab sync (which
    # pulls the cluster's slug_verdicts.jsonl). Mirror the verdict to the
    # cluster — where the harvest guard reads it (so junk isn't re-harvested)
    # AND where the sync pulls from — so it persists. Fire-and-forget so the
    # verdict button stays instant (don't block on the SSH).
    if _CLUSTER_SSH:
        try:
            import base64 as _b64, subprocess as _sp2
            _b = _b64.b64encode((_json.dumps(ev) + "\n").encode()).decode()
            # ONE remote command string (ssh joins extra argv with spaces, which
            # would break `bash -lc <cmd>`); mirror how _slurm passes it.
            _cmd = ("cd %s 2>/dev/null; echo %s | base64 -d >> results/framework/slug_verdicts.jsonl"
                    % (_shlex.quote(_CLUSTER_REPO), _b))
            _sp2.Popen(_ssh_cluster_prefix() + [_cmd],
                       stdout=_sp2.DEVNULL, stderr=_sp2.DEVNULL,
                       stdin=_sp2.DEVNULL, start_new_session=True)
        except Exception:
            pass
    # Verdict changed → invalidate per-class pool caches so junk slugs vanish
    # from /classes on next load.
    try:
        for p in _pool_cache_dir.glob("*.json"):
            try: p.unlink()
            except Exception: pass
    except Exception:
        pass
    return JSONResponse({"ok": True, **ev})


# ---- topic override: user/Brain corrects misclassified species ----
@app.get("/api/class_topic")
def api_class_topic_list():
    """Return the full override map."""
    return JSONResponse({
        "overrides": _load_topic_overrides(),
        "file": str(_CLASS_TOPIC_OVERRIDES_FILE),
        "valid_topics": ["cwd12", "weed", "disease", "pest", "crop", "other"],
    })


@app.post("/api/class_topic/{cls}")
async def api_class_topic_set(cls: str, payload: dict = Body(...)):
    """Set topic for a specific class.
    Topic must be one of: cwd12 | weed | disease | pest | crop | other | _clear_.
    _clear_ removes the override (falls back to keyword heuristic).

    Future autonomous agent: after Brain LLM classifies a new species, it
    POSTs here to persist. User can override too via /classes UI."""
    if not _cls_ok(cls):
        raise HTTPException(400, "bad class name chars")
    topic = payload.get("topic", "")
    if topic not in ("cwd12", "weed", "disease", "pest", "crop", "other", "_clear_"):
        raise HTTPException(400, "invalid topic")
    ok = _save_topic_override(cls, topic)
    if not ok:
        raise HTTPException(500, "save failed")
    return JSONResponse({
        "ok": True, "cls": cls, "topic": topic,
        "effective_topic": _class_topic(cls),
    })


# ---- refresh: invalidate all caches so next /classes load re-scans -------
# ===========================================================
# /control — operator-facing cluster control panel (v3.0.43.4)
# ===========================================================
# User asked: don't make me depend on Claude to deploy / restart / browse
# cluster state. Surface everything as a dashboard page that I can hit
# directly with a button.
# ===========================================================

def _shell(cmd: list, timeout: int = 15) -> dict:
    """Run a command, return {ok, stdout, stderr, returncode}."""
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": r.returncode == 0, "stdout": r.stdout,
                "stderr": r.stderr, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "TIMEOUT",
                "returncode": -1}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e),
                "returncode": -2}


# v3.0.99.40: lab-server CONTROL mode. When the dashboard runs on the lab server
# (no SLURM), set CLUSTER_SSH so SLURM/cluster commands (sbatch/squeue/sacct/
# scancel) + sbatch scripts + job .out reads run ON the cluster over an SSH key.
# When unset (dashboard hosted on the cluster itself), everything runs locally.
import shlex as _shlex
_CLUSTER_SSH = os.environ.get("CLUSTER_SSH", "")  # e.g. byler@bridges2.psc.edu
_CLUSTER_REPO = os.environ.get(
    "CLUSTER_REPO", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark")
_CLUSTER_KEY = os.environ.get(
    "CLUSTER_SSH_KEY", os.path.expanduser("~/.ssh/id_lab2cluster"))


def _ssh_cluster_prefix() -> list:
    # v3.0.99.41: Bridges-2 does NOT honor user-added authorized_keys (resets them)
    # → use PASSWORD auth via SSH_ASKPASS (env set in the service) + ControlMaster
    # multiplexing: the FIRST connection authenticates (password), all subsequent
    # commands reuse the persistent socket → no repeated auth → no login throttle
    # + fast. No -i/IdentitiesOnly/BatchMode (those would block password fallback).
    cm = os.path.expanduser("~/.ssh/cm-%r@%h:%p")
    return ["ssh", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=25", "-o", "ControlMaster=auto",
            "-o", f"ControlPath={cm}", "-o", "ControlPersist=12h", _CLUSTER_SSH]


def _slurm(cmd: list, timeout: int = 15) -> dict:
    """Run a SLURM/cluster command locally, or ON the cluster via SSH key when
    CLUSTER_SSH is set (lab-server control mode). cmd is an argv list."""
    if not _CLUSTER_SSH:
        return _shell(cmd, timeout)
    remote = "cd %s 2>/dev/null; %s" % (
        _shlex.quote(_CLUSTER_REPO),
        " ".join(_shlex.quote(a) for a in cmd))
    return _shell(_ssh_cluster_prefix() + [remote], timeout=max(timeout + 20, 35))


@app.post("/api/cancel_job/{jobid}")
def api_cancel_job(jobid: str):
    """Cancel a SLURM job by ID. Only allow cancelling job names we recognize
    as 'safe to interrupt' (agent jobs, not the dashboard itself)."""
    if not _re_cls.match(r'^[0-9]+$', jobid):
        raise HTTPException(400, "bad jobid")
    # Check job name to prevent self-killing
    sq = _slurm(["squeue", "-j", jobid, "-h", "-o", "%j"], timeout=8)
    name = (sq["stdout"] or "").strip()
    if not name:
        return JSONResponse({"ok": False, "msg": f"job {jobid} not in queue"})
    # Whitelist: agent jobs only. Don't cancel the dashboard from /control
    # (use restart_dashboard action for that instead).
    SAFE_PREFIXES = ("dl_known", "brain_hrv", "topic_bf", "smoke", "lora_")
    if not any(name.startswith(p) for p in SAFE_PREFIXES):
        return JSONResponse({"ok": False, "msg":
            f"refuse to cancel {name!r} — only agent jobs cancellable from UI"})
    r = _slurm(["scancel", jobid], timeout=8)
    return JSONResponse({"ok": r["ok"], "jobid": jobid, "name": name,
                          "msg": r["stdout"] or r["stderr"] or "cancelled"})


# Cluster-actions history log (append-only)
_ACTIONS_LOG = REPO / "results" / "framework" / "cluster_actions.jsonl"
_ACTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_action(action: str, result: dict):
    """Persist a /control action invocation for the history panel."""
    try:
        ev = {
            "ts": time.time(),
            "ts_h": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "action": action,
            "result": result,
        }
        with open(_ACTIONS_LOG, "a") as f:
            f.write(_json.dumps(ev) + "\n")
    except Exception as e:
        log.warning(f"action log fail: {e}")


# v3.0.84 (P0 — honesty instrumentation): resolve the REAL outcome of an action,
# not just "launched ok". sbatch → sacct/squeue; subprocess → pid liveness + log
# failure markers; refresh/restart → the immediate ok flag.
_action_status_cache: dict = {}   # jobid → terminal status (never changes once terminal)
_SACCT_TERMINAL = {
    "COMPLETED": "succeeded", "FAILED": "failed", "TIMEOUT": "failed",
    "CANCELLED": "failed", "OUT_OF_MEMORY": "failed", "NODE_FAIL": "failed",
    "DEADLINE": "failed", "PREEMPTED": "failed", "BOOT_FAIL": "failed",
}
_SACCT_ACTIVE = {"RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING"}
_LOG_FAIL_MARKERS = ("Traceback (most recent", "No module named", "command not found",
                     "FATAL", "FAILED", " Error:", "Killed", "OOM",
                     "raise ", "Exception:", "Errno")


def _sacct_state(jobid: str) -> str:
    """Main job-step State from sacct (e.g. COMPLETED/FAILED/TIMEOUT). '' if unknown."""
    r = _slurm(["sacct", "-j", jobid, "--format=State", "--noheader", "-P"], timeout=10)
    if not r["ok"]:
        return ""
    for ln in r["stdout"].splitlines():
        tok = ln.strip().split()
        if tok:
            return tok[0].upper()   # "CANCELLED by 123" → CANCELLED
    return ""


def _batch_sacct(jobids: list) -> dict:
    """v3.0.99.46: ONE sacct call for MANY jobids → {jobid: status}. Replaces the
    per-action SSH loop that made /api/action_history take ~19s in lab-control mode
    (30 actions × per-job SSH). status ∈ {succeeded, failed, running}; jobs not in
    sacct output are omitted (caller treats as 'launched')."""
    ids = sorted({str(j) for j in jobids if j})
    if not ids:
        return {}
    r = _slurm(["sacct", "-j", ",".join(ids),
                "--format=JobID,State", "--noheader", "-P"], timeout=20)
    if not r["ok"]:
        return {}
    out: dict = {}
    for ln in r["stdout"].splitlines():
        parts = ln.split("|")
        if len(parts) < 2:
            continue
        jid = parts[0].split(".")[0].strip()   # main step (drop .batch/.extern)
        if jid in out:
            continue
        state = parts[1].strip().upper().split()[0] if parts[1].strip() else ""
        if state in _SACCT_TERMINAL:
            out[jid] = _SACCT_TERMINAL[state]
        elif state in _SACCT_ACTIVE:
            out[jid] = "running"
    return out


def _pid_alive_nonzombie(pid) -> bool:
    """True only if pid exists AND is not a zombie. On Linux read /proc/<pid>/stat
    (state field after the ')'); 'Z' = zombie (exited, un-reaped) → treat as done.
    Falls back to os.kill(pid,0) where /proc is unavailable."""
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
        state = st.rsplit(") ", 1)[1].split(" ", 1)[0]
        return state != "Z"
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def _resolve_action_real_status(result: dict, state_map: dict = None) -> str:
    """One of: launched | running | succeeded | failed | unknown. Real, not just ok.
    state_map (v3.0.99.46): optional {jobid: status} from a batched sacct so the
    sbatch branch needs ZERO per-call SSH (avoids the 19s action_history hang)."""
    if not isinstance(result, dict):
        return "unknown"
    # --- subprocess: pid + log_path ---
    pid = result.get("pid")
    log_path = result.get("log_path")
    if pid and log_path:
        tail = ""
        try:
            p = Path(log_path)
            if p.is_file():
                sz = p.stat().st_size
                with open(p, "rb") as f:
                    f.seek(max(0, sz - 8192))
                    tail = f.read().decode("utf-8", "replace")
        except Exception:
            pass
        # v3.0.99: authoritative exit-code marker appended by the bash wrapper.
        mrc = re.search(r"__ACTION_RC__=(-?\d+)", tail)
        if mrc:
            return "succeeded" if mrc.group(1) == "0" else "failed"
        # No marker yet → still going IF the pid is alive AND not a zombie.
        # (Un-reaped children show as zombies; os.kill(zombie,0) lies "alive".)
        if _pid_alive_nonzombie(pid):
            return "running"
        # pid gone, no RC marker (killed / pre-v3.0.99 launch) → fall back to markers
        if any(m in tail for m in _LOG_FAIL_MARKERS):
            return "failed"
        return "succeeded"
    # --- sbatch: parse "Submitted batch job N" ---
    text = f"{result.get('stdout','')} {result.get('msg','')}"
    m = re.search(r"Submitted batch job (\d+)", text)
    if m:
        jid = m.group(1)
        if jid in _action_status_cache:
            return _action_status_cache[jid]
        # v3.0.99.46: use the batched sacct map first → no per-action SSH.
        if state_map is not None and jid in state_map:
            s = state_map[jid]
            if s in ("succeeded", "failed"):
                _action_status_cache[jid] = s
            return s
        if state_map is not None:
            # batched but this jobid wasn't in sacct (purged/too new) → don't SSH again
            return "launched"
        st = _sacct_state(jid)
        if st in _SACCT_TERMINAL:
            _action_status_cache[jid] = _SACCT_TERMINAL[st]
            return _action_status_cache[jid]
        if st in _SACCT_ACTIVE:
            return "running"
        sq = _slurm(["squeue", "-j", jid, "-h", "-o", "%T"], timeout=6)
        if sq["ok"] and sq["stdout"].strip():
            return "running"
        return "launched"             # submitted but unresolvable (purged/too new)
    # --- refresh / restart_self / immediate ---
    if "ok" in result:
        return "succeeded" if result.get("ok") else "failed"
    return "unknown"


@app.get("/api/action_history")
def api_action_history(n: int = 50, resolve: int = 1):
    """Last N action invocations (from cluster_actions.jsonl). v3.0.84: each row
    gets a real `status` (launched/running/succeeded/failed) unless resolve=0."""
    if not 1 <= n <= 500:
        n = 50
    if not _ACTIONS_LOG.is_file():
        return JSONResponse({"history": []})
    try:
        lines = _ACTIONS_LOG.read_text().splitlines()
        events = []
        for ln in lines[-n:]:
            try: events.append(_json.loads(ln))
            except Exception: pass
        if resolve:
            recent = events[-30:]
            # v3.0.99.46: ONE batched sacct for all sbatch jobids (was 1 SSH/action
            # → ~19s in lab-control mode). Subprocess actions resolve from local logs.
            jids = []
            for ev in recent:
                res = ev.get("result") or {}
                txt = f"{res.get('stdout','')} {res.get('msg','')}"
                mm = re.search(r"Submitted batch job (\d+)", txt)
                if mm and mm.group(1) not in _action_status_cache:
                    jids.append(mm.group(1))
            state_map = _batch_sacct(jids) if jids else {}
            for ev in recent:
                try:
                    ev["status"] = _resolve_action_real_status(
                        ev.get("result") or {}, state_map)
                except Exception:
                    ev["status"] = "unknown"
        return JSONResponse({"history": events})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/disk_usage")
def api_disk_usage():
    """`df` for /ocean (the data root). Cheap. Lustre du is too slow."""
    r = _shell(["df", "-h", str(REPO)], timeout=6)
    if not r["ok"]:
        return JSONResponse({"error": r["stderr"][:120]}, status_code=500)
    lines = r["stdout"].strip().split("\n")
    if len(lines) < 2:
        return JSONResponse({"raw": r["stdout"]})
    parts = lines[1].split()
    return JSONResponse({
        "filesystem": parts[0] if len(parts) > 0 else "",
        "size":       parts[1] if len(parts) > 1 else "",
        "used":       parts[2] if len(parts) > 2 else "",
        "avail":      parts[3] if len(parts) > 3 else "",
        "use_pct":    parts[4] if len(parts) > 4 else "",
        "mount":      parts[5] if len(parts) > 5 else "",
        "raw": r["stdout"][:400],
    })


@app.get("/api/agent_progress")
def api_agent_progress():
    """Parse the latest agent job (dl_known, brain_hrv, topic_bf) log file
    for progress markers like '[dl_known] [3/8] === deepweeds ==='.

    Returns a structured 'what is the agent currently doing' summary —
    surfaced on /control as a stat card."""
    pat_dir = REPO / "results" / "framework"
    if not pat_dir.is_dir():
        return JSONResponse({"no_agent_jobs": True})

    # Only look at agent-launched jobs (not dashboard self-runs)
    agent_patterns = ("v3_0_43_dl_known", "v3_0_43_brain_harvest",
                       "v3_0_43_topic_backfill", "v3_0_41_brain_harvest")
    candidates = []
    for pat in agent_patterns:
        candidates.extend(pat_dir.glob(f"{pat}_*.out"))
    if not candidates:
        return JSONResponse({"no_agent_jobs": True})
    latest = max(candidates, key=lambda p: p.stat().st_mtime)

    import re as _re
    try:
        size = latest.stat().st_size
        # Tail 64KB to find progress markers
        with open(latest, "rb") as f:
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return JSONResponse({"error": str(e)})

    # Extract progress markers
    m_jobid = _re.search(r"_(\d+)\.out$", latest.name)
    jobid = m_jobid.group(1) if m_jobid else ""
    # name part = filename minus _jobid.out
    name_part = latest.name[: m_jobid.start()] if m_jobid else latest.stem

    # Try job state
    state = ""
    sq = _slurm(["squeue", "-j", jobid, "-h", "-o", "%T"], timeout=6)
    if sq["ok"] and sq["stdout"].strip():
        state = sq["stdout"].strip()

    # Parse progress like `[dl_known] [3/8] === deepweeds ===`
    progress = None
    for line in tail.splitlines()[::-1]:
        m = _re.search(r"\[(\d+)/(\d+)\][^=]*=== ([^=]+?) ===", line)
        if m:
            progress = {
                "current": int(m.group(1)),
                "total": int(m.group(2)),
                "current_item": m.group(3).strip(),
                "raw_line": line.strip()[:160],
            }
            break

    # Last few non-empty lines for log_tail
    log_tail = [l for l in tail.splitlines()[-15:] if l.strip()][-8:]

    return JSONResponse({
        "job_name": name_part,
        "jobid": jobid,
        "state": state or "FINISHED",
        "log_file": str(latest),
        "log_size": size,
        "progress": progress,
        "log_tail": log_tail,
        "mtime": latest.stat().st_mtime,
    })


_squeue_cache: dict = {"ts": 0.0, "sq": None}   # v3.0.99.46: TTL cache for /control auto-refresh


@app.get("/api/cluster_status")
def api_cluster_status():
    """Return a structured snapshot of cluster state — what /control polls."""
    out: dict = {"generated_at": time.time()}

    # SLURM job queue for our user. v3.0.99.46: 12s TTL cache so the /control
    # auto-refresh (every ~5-30s) doesn't SSH the cluster on every hit.
    _now = time.time()
    if _squeue_cache["sq"] is not None and _now - _squeue_cache["ts"] < 12:
        sq = _squeue_cache["sq"]
    else:
        sq = _slurm(["squeue", "-u", "byler",
                     "-o", "%i\t%j\t%T\t%M\t%R\t%C\t%m"], timeout=10)
        if sq["ok"]:
            _squeue_cache["sq"] = sq
            _squeue_cache["ts"] = _now
    jobs = []
    if sq["ok"]:
        for line in sq["stdout"].strip().split("\n")[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) >= 5:
                jobs.append({
                    "jobid": parts[0].strip(),
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "time": parts[3].strip(),
                    "reason": parts[4].strip(),
                    "cpus": parts[5].strip() if len(parts) > 5 else "",
                    "mem": parts[6].strip() if len(parts) > 6 else "",
                })
    out["jobs"] = jobs
    out["squeue_ok"] = sq["ok"]

    # Current dashboard job (this process)
    out["my_slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")

    # Tunnel URL: read from logs/tunnel_${SLURM_JOB_ID}.log (this job)
    jid = out["my_slurm_job_id"]
    tunnel_url = ""
    if jid:
        tlog = REPO / "logs" / f"tunnel_{jid}.log"
        if tlog.exists():
            try:
                content = tlog.read_text()
                import re as _re
                matches = _re.findall(r"https://[a-z][a-z0-9-]+\.trycloudflare\.com",
                                       content)
                # Filter out api.trycloudflare.com (cloudflared metric)
                matches = [m for m in matches if "//api." not in m]
                if matches:
                    tunnel_url = matches[0]
            except Exception:
                pass
    out["tunnel_url"] = tunnel_url

    # Ollama state
    ol_bin = "/ocean/projects/cis240145p/byler/ollama/bin/ollama"
    if os.path.isfile(ol_bin):
        # Check if reachable
        try:
            import urllib.request as _urlreq
            r = _urlreq.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            ol_data = _json.loads(r.read())
            out["ollama"] = {
                "running": True,
                "models": [m.get("name", "?") for m in ol_data.get("models", [])][:10],
            }
        except Exception as e:
            out["ollama"] = {"running": False, "error": str(e)[:80]}
    else:
        out["ollama"] = {"running": False, "error": "binary not found"}

    # Registry stats
    try:
        with open(REGISTRY_PATH) as f:
            reg = _json.load(f)
        ds = reg.get("datasets", {})
        n_with_classnames = sum(
            1 for v in ds.values() if (v.get("class_names") or []))
        n_downloaded = sum(1 for v in ds.values() if v.get("status") == "downloaded")
        out["registry"] = {
            "n_slugs": len(ds),
            "n_downloaded": n_downloaded,
            "n_with_classnames": n_with_classnames,
            "total_imgs": sum(v.get("local_images", 0) for v in ds.values()),
        }
    except Exception as e:
        out["registry"] = {"error": str(e)[:80]}

    # Topic overrides count
    try:
        out["n_topic_overrides"] = len(_load_topic_overrides())
    except Exception:
        out["n_topic_overrides"] = 0

    # Exemplar marks count
    try:
        n_ex = 0; n_bad = 0
        if _CLS_EXEMPLAR_DIR.is_dir():
            for fp in _CLS_EXEMPLAR_DIR.glob("*.jsonl"):
                try:
                    cls = fp.stem
                    st = _exemplar_state(cls)
                    n_ex += sum(1 for v in st.values() if v == "exemplar")
                    n_bad += sum(1 for v in st.values() if v == "bad")
                except Exception:
                    pass
        out["exemplars"] = {"n_keep": n_ex, "n_bad": n_bad}
    except Exception:
        out["exemplars"] = {"n_keep": 0, "n_bad": 0}

    # Slug verdicts
    try:
        sv = _slug_verdict_state()
        cnt = {}
        for v in sv.values():
            cnt[v] = cnt.get(v, 0) + 1
        out["slug_verdicts"] = {"counts": cnt, "total": len(sv)}
    except Exception:
        out["slug_verdicts"] = {"counts": {}, "total": 0}

    return JSONResponse(out)


# Map of action_id → (script_path, allowed?). Whitelist.
_CLUSTER_ACTIONS = {
    "restart_dashboard": {
        "type": "restart_self",
        "label": "Restart dashboard server (cancels current job + sbatch new)",
    },
    "brain_harvest": {
        "type": "sbatch",
        "script": "run_v3_0_43_brain_harvest_oneshot.sh",
        "label": "Brain one round: harvest_new_datasets — find + pull NEW datasets (~30 min)",
    },
    "download_known_slugs": {
        "type": "sbatch",
        "script": "run_v3_0_43_download_known_slugs.sh",
        "label": "Download all status=known HF slugs to the /ocean cluster (~30min-2hr)",
    },
    "topic_backfill": {
        "type": "sbatch",
        "script": "run_v3_0_43_topic_backfill.sh",
        "label": "Ollama + topic_backfill_all (1 GPU, ~15 min)",
    },
    "refresh_registry": {
        "type": "refresh",
        "label": "wipe class-pool cache + reload registry index",
    },
    # v3.0.45 (auto-loop iter 2): subprocess actions — local to dashboard
    # node, no sbatch. Logs to shared FS so visible across login nodes.
    # v3.0.60 (2026-05-30): per-species projects deleted (Public Plan cap +
    # user shifted to school workspace a-test-of-will). Replaced with a
    # single multi-class upload to cwd12-multiclass-v1.
    "roboflow_sync_cwd12_v1": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "bulk-upload",
            "--images", "downloads/cottonweeddet12/train/images",
            "--labels", "downloads/cottonweeddet12/train/labels",
            "--split", "train", "--batch", "green",
            "--workers", "8", "--per-species", "50",
            "--project", "cwd12-multiclass-v1",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Roboflow multi-class upload → cwd12-multiclass-v1 (frozen 598-img benchmark, ~10min)",
    },
    # v3.0.67 (2026-05-31): new agent-harvested target project per user
    # directive — keep frozen cwd12 benchmark separate from autonomous
    # harvest, route all brain-harvested goal-relevant data to
    # weed-crop-agent-dataset (project lives in a-test-of-will workspace;
    # user may drag into "weed_crop_agent_dataset" Project Folder in UI).
    "roboflow_sync_agent_v1": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "bulk-upload",
            "--images", "downloads/cottonweeddet12/train/images",
            "--labels", "downloads/cottonweeddet12/train/labels",
            "--split", "train", "--batch", "green",
            "--workers", "8", "--per-species", "50",
            "--project", "weed-crop-agent-dataset",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Roboflow upload → weed-crop-agent-dataset (new agent collection pool, ~10min)",
    },
    # v3.0.46 (auto-loop iter 4 / Phase D1): bucket-audit CLI.
    "build_buckets": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.bucketer",
            "--out", "results/framework/buckets.json",
        ],
        "label": "Bucket audit: each downloaded slug → A/B/C buckets + cwd12 species coverage (~30s)",
    },
    # v3.0.47 (auto-loop iter 5 / Phase C3 skeleton): Roboflow workspace audit.
    "roboflow_state_audit": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.merge_roboflow_projects",
            "--out", "results/framework/roboflow_state.json",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Roboflow 13-project status audit (imgs/boxes/versions per cwd12-<sp>, ~10s)",
    },
    # v3.0.50 (auto-loop iter 8 / Phase D3): OWL pre-annotate sbatch.
    # Needs an exemplar-config JSON at the default path or env override —
    # the script fails gracefully with usage hint if config missing.
    "owl_preannotate_one": {
        "type": "sbatch",
        "script": "run_v3_0_50_owl_preannotate.sh",
        "label": "OWL pre-annotate 1 species — produces red bbox proposals (~10-30min, needs exemplar JSON)",
    },
    # v3.0.99.30 (D): clean-subset training — quality>quantity probe. Trains on
    # high-DINO clean data only (min_dino_score gate, no 175K loose auto-labels)
    # with cwd12 gold val, then auto-evals mAP50-95 vs the 0.67 noisy baseline.
    # HTTP-triggerable so D can launch without SSH (login-throttle workaround).
    "clean_train_d": {
        "type": "sbatch",
        "script": "run_v3_0_99_clean_train.sh",
        "label": "D clean-subset training (quality>quantity probe: DINO gate + cwd12 gold val + auto-eval, GPU)",
    },
    # v3.0.62 (button-test iter 9, Phase T3): export ✓/object_bank exemplars
    # to JSON configs that owl_preannotate.py consumes. Without this, OWL
    # button just FATALs on missing exemplar config.
    "export_owl_exemplars": {
        "type": "subprocess",
        "needs_cluster": True,  # v3.0.99.49: reads object_bank (cluster-only, not migrated)
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.export_owl_exemplars",
            "--source", "bank", "--per-species", "5",
        ],
        "label": "Generate per-species exemplar JSON for OWL (reads object_bank, ~3s, 12 files)",
    },
    # v3.0.74 (2026-06-01): round tracking — start a new harvest round.
    # Subsequent brain_harvest calls tag downloaded slugs with the new round.
    "start_new_round": {
        "type": "subprocess",
        # v3.0.99.50: mutates the AUTHORITATIVE registry (current_round). On
        # lab that write is clobbered by the next cluster→lab sync (harvest
        # runs on cluster → cluster registry is current_round's source of
        # truth) → silently reverted + misleads. Route to cluster.
        "needs_cluster": True,
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.rounds", "start-new",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "▶ Start NEW harvest round (v{N} → v{N+1}); creates RF project but no data yet",
    },
    # v3.0.77 (2026-06-01): ONE-CLICK new harvest round end-to-end.
    # Triggers brain_harvest SBATCH with env vars that:
    #   1. ROUND_BUMP=1 → start_new_round at job-start (new RF project)
    #   2. brain_harvest collects new datasets (4h, loosened strict)
    #   3. AUTO_SYNC=1 → upload survivors to the new round's RF project
    # User clicks ONCE, walks away, comes back to populated v{N+1}.
    "harvest_full_round_e2e": {
        "type": "sbatch",
        "script": "run_v3_0_43_brain_harvest_oneshot.sh",
        "sbatch_extra_args": [
            "--time=04:00:00",
            "--export=ALL,ROUND_BUMP=1,AUTO_SYNC=1,BRAIN_STRICT=1,BRAIN_STRICT_MIN_LABELS=50,BRAIN_MAX_NEW=8,BRAIN_MAX_IMGS=2000",
        ],
        "label": "🚀 ONE-CLICK new round: bump round → harvest 4h → auto-sync to weed-crop-agent-v{N+1}",
    },
    "backfill_round_1": {
        "type": "subprocess",
        "needs_cluster": True,  # v3.0.99.50: mutates authoritative registry (round tags) — cluster is truth
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.rounds", "backfill",
        ],
        "label": "Backfill harvest_round=1 on pre-v3.0.74 slugs (idempotent, ~3s)",
    },
    # v3.0.74 Stage 4 — DINOv2 filters round N's un-verified slugs and
    # uploads survivors as agent-v{N}-dinov2-v{X.Y}. Auto-bumps sub-version
    # on re-run so you can iterate (v1.0 → v1.1 → v1.2 ...).
    "dinov2_filter_round_1": {
        "type": "subprocess",
        "needs_cluster": True,  # v3.0.99.49: torch/DINOv2 GPU — runs on cluster
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.dinov2_round_filter",
            "--round", "1", "--threshold", "0.6",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "🧠 DINOv2 filter ROUND 1 → upload survivors as agent-v1-dinov2-v{X.Y} (re-clickable, ~10min)",
    },
    # v3.0.74 Stage 5 — placeholder send-to-training trigger
    "train_yolo_round_1": {
        "type": "subprocess",
        "needs_cluster": True,  # v3.0.99.49: YOLO training GPU — runs on cluster
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.train_yolo_on_verified",
            "--round", "1",
        ],
        "label": "🚀 Send round 1 verified ✓ → YOLO trainer (PLACEHOLDER, queues 10min stub)",
    },
    # v3.0.71 (2026-05-31): retroactive registry garbage audit.
    # Applies v3.0.68 strict filter to pre-strict slugs (ibm CIF etc.).
    # Two-button design: first DRY-RUN (default), then APPLY confirms.
    "audit_registry_garbage": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.audit_registry_garbage",
        ],
        "label": "Retroactive registry audit — list slugs with labeled<100/classes=0 (dry-run, ~5s)",
    },
    "audit_registry_garbage_APPLY": {
        "type": "subprocess",
        # v3.0.99.50: --apply pops slugs from the authoritative registry + deletes
        # disk files. On lab both are futile: registry write clobbered by sync,
        # and deleted disk files re-appear on next rsync from cluster. Route to
        # cluster (the dry-run variant above stays lab-local — it's read-only).
        "needs_cluster": True,
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.audit_registry_garbage",
            "--apply",
        ],
        "label": "⚠️ APPLY: delete all garbage slugs listed by the dry-run above + their disk files",
    },
    # v3.0.71: sync newest brain-harvested slugs → weed-crop-agent-dataset
    # + auto-place into weed_crop_agent_dataset folder. THIS is the
    # brain_harvest → Roboflow visible loop the user wants to see.
    "sync_newest_slugs": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "sync-newest-slugs",
            "--project", "weed-crop-agent-dataset",
            "--folder", "weed_crop_agent_dataset",
            "--cap-per-slug", "100",
            "--skip-baselines",  # default: don't push cwd12 here (use sync_all_to_roboflow for that)
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Roboflow upload — skip cwd12 baseline (agent-harvested only, ~10min/slug)",
    },
    # v3.0.75 (2026-06-01): user wants ALL data in Roboflow for fast browsing.
    # This variant uploads EVERYTHING including cwd12 baselines.
    "sync_all_to_roboflow": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "sync-newest-slugs",
            "--project", "weed-crop-agent-dataset",
            "--folder", "weed_crop_agent_dataset",
            "--cap-per-slug", "100",
            # NO --skip-baselines: cwd12 baselines (sp8/holdout) included
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "🚀 Sync ALL (incl. cwd12 baselines) → Roboflow for fast review (~10min/slug)",
    },
    # v3.0.71.3: OWL red proposals → Roboflow upload.
    # OWL only ran on Goosegrass against cottonweed_holdout/test/images; its
    # output is 50 .txt files in results/framework/owl_red_proposals/Goosegrass/.
    # We upload the source images + OWL-proposed labels via bulk-upload to
    # weed-crop-agent-dataset, tagged batch=red so human reviewer sees them.
    "owl_upload_proposals": {
        "type": "subprocess",
        "needs_cluster": True,  # v3.0.99.49: reads OWL proposals produced on cluster
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.owl_upload_proposals",
            "--species", "Goosegrass",
            "--project", "weed-crop-agent-dataset",
            "--per-species", "50",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Upload OWL red proposals → weed-crop-agent-dataset (precision-gated; rejects upload below threshold, OWL_UPLOAD_FORCE=1 to force)",
    },
    # v3.0.71: DINOv2 dataset-quality curator (full pipeline as one button)
    "dinov2_curate_registry": {
        "type": "sbatch",
        "script": "run_v3_0_36_dinov2_curator.sh",
        "label": "DINOv2 reference-pool curator (~4h GPU): build ref + score all slugs + ranking report",
    },
    # v3.0.69 (2026-05-31): Roboflow Project Folder ops via /groups REST.
    # Earlier-session conclusion "no API support" was wrong — the path uses
    # internal name /groups while UI says Folders. Verified PATCH 204 OK on
    # free-tier school workspace. See memory/feedback_roboflow_folder_api.md.
    "roboflow_list_folders": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "list-folders",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Roboflow folders list (all folders in workspace + member projects, ~3s)",
    },
    "roboflow_move_agent_to_folder": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "move-to-folder",
            "--project", "weed-crop-agent-dataset",
            "--folder", "weed_crop_agent_dataset",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Move the weed-crop-agent-dataset project into the weed_crop_agent_dataset folder (idempotent, ~3s)",
    },
    # v3.0.51 (auto-loop iter 9 / Phase D4): Roboflow Version generation + pull-back.
    # Generate is RATE-LIMITED on free tier (~10/proj/month) — skips by default
    # if Versions already exist. Use --force in manual sbatch to regenerate.
    "roboflow_generate_versions": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.merge_roboflow_projects",
            "generate-versions",
            "--project", "weed-crop-agent-dataset",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "Roboflow version generate → weed-crop-agent-dataset (skips if existing, ~30s)",
    },
    "roboflow_download_merge": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m",
            "weed_optimizer_framework.tools.merge_roboflow_projects",
            "download-merge",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "⬇️ Export labels back to cluster: Roboflow-labeled data → cwd12 multi-class training set (quality-kept + ground-truth written to disk)",
    },
    # v3.0.99.21: delete junk-verdict datasets' images from Roboflow (storage is a
    # recurring monthly cost → after review, delete junk, keep only the clean library).
    # Dry-run first (counts), then APPLY. Ground truth is already exported to cluster.
    "roboflow_delete_junk_dryrun": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "delete-junk",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "🗑️ Junk delete (dry-run): count how many images of junk datasets would be removed on Roboflow (no real delete)",
    },
    "roboflow_delete_junk_apply": {
        "type": "subprocess",
        "argv": [
            "python", "-u", "-m", "weed_optimizer_framework.tools.roboflow_sync",
            "delete-junk", "--apply",
        ],
        "env_secret_files": {"ROBOFLOW_API_KEY": _ROBOFLOW_KEY_FILE},
        "label": "🗑️ Junk delete (apply): remove verdict=junk datasets from Roboflow to save monthly storage (ground-truth already on cluster)",
    },
    # v3.0.52 (auto-loop iter 11 / Phase D2 sbatch): DINOv2 routing job.
    "dinov2_route_classes": {
        "type": "sbatch",
        "script": "run_v3_0_52_dinov2_route.sh",
        "label": "DINOv2 routing (weed-vs-not-weed + species classification, 1×V100 ~30min)",
    },
}


# Shared FS for agent-task logs (login-node /tmp is session-local and
# Bridges-2 kills login-node nohup processes — see [[project_classes_thumb_perf]]).
_AGENT_LOG_DIR = REPO / "logs" / "agent_tasks"


@app.get("/api/job_log/{jobid}")
def api_job_log(jobid: str, tail: int = 200):
    """Return the last `tail` lines of the SLURM output file for jobid.

    SLURM writes to results/framework/<name>_<jobid>.out (with SBATCH --output).
    We glob for *_{jobid}.out to find the file regardless of name."""
    if not _re_cls.match(r'^[0-9_]+$', jobid):
        raise HTTPException(400, "bad jobid chars")
    if not 1 <= tail <= 2000:
        tail = 200

    # v3.0.99.40: lab-control mode → the .out lives on the cluster; find newest
    # matching file there + tail it over SSH.
    if _CLUSTER_SSH:
        script = (f"f=$(ls -t results/framework/*_{jobid}.out "
                  f"results/*_{jobid}.out logs/*{jobid}*.out 2>/dev/null | head -1); "
                  f'if [ -z "$f" ]; then echo __NOFILE__; '
                  f'else echo "__FILE__:$f"; tail -n {tail} "$f"; fi')
        r = _slurm(["bash", "-lc", script], timeout=45)
        out = r.get("stdout", "")
        if (not r["ok"]) or "__NOFILE__" in out:
            return JSONResponse({"ok": False, "jobid": jobid, "remote": True,
                "msg": f"no output file for {jobid} on cluster"}, status_code=404)
        lines = out.splitlines()
        fpath = ""
        if lines and lines[0].startswith("__FILE__:"):
            fpath = lines[0][len("__FILE__:"):]
            lines = lines[1:]
        return JSONResponse({"ok": True, "jobid": jobid, "remote": True,
            "file": fpath, "lines_returned": len(lines),
            "content": "\n".join(lines)})

    # Search standard locations
    candidates = []
    search_dirs = [
        REPO / "results" / "framework",
        REPO / "results",
        REPO / "logs",
    ]
    for d in search_dirs:
        if d.is_dir():
            candidates.extend(d.glob(f"*_{jobid}.out"))
            candidates.extend(d.glob(f"*_{jobid}.log"))
            candidates.extend(d.glob(f"*{jobid}*.out"))

    # De-dup + sort by mtime
    seen: set = set()
    files: list = []
    for p in candidates:
        rp = str(p.resolve())
        if rp in seen: continue
        seen.add(rp)
        files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    if not files:
        return JSONResponse({
            "ok": False, "jobid": jobid,
            "msg": f"no output file found for job {jobid} in standard locations",
            "searched": [str(d) for d in search_dirs],
        }, status_code=404)

    main = files[0]
    try:
        size = main.stat().st_size
        # Efficient tail read for large files: seek to end, read last ~64KB
        max_bytes = min(size, 256 * 1024)  # cap at 256KB
        with open(main, "rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        # If we truncated, prepend marker
        lines = data.splitlines()
        if size > max_bytes:
            lines = ["… (file truncated, showing last %d bytes) …" % max_bytes] + lines
        if len(lines) > tail:
            lines = lines[-tail:]
    except Exception as e:
        raise HTTPException(500, f"read fail: {e}")

    return JSONResponse({
        "ok": True, "jobid": jobid,
        "file": str(main),
        "file_size_bytes": size,
        "mtime": main.stat().st_mtime,
        "lines_returned": len(lines),
        "tail_requested": tail,
        "content": "\n".join(lines),
        "other_files": [str(f) for f in files[1:6]],  # show up to 5 alternates
    })


@app.get("/api/recent_jobs")
def api_recent_jobs(n: int = 20):
    """List the N most recent job .out files (by mtime). Useful for the
    /control 'old jobs' picker — view logs from already-finished jobs."""
    if not 1 <= n <= 100:
        n = 20
    pat_dir = REPO / "results" / "framework"
    if not pat_dir.is_dir():
        return JSONResponse({"jobs": []})
    files = sorted(pat_dir.glob("*.out"),
                    key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    import re as _re
    out = []
    for f in files:
        m = _re.search(r"_(\d+)\.out$", f.name)
        if not m: continue
        jid = m.group(1)
        name_part = f.name[: m.start()].rstrip("_")
        try:
            mt = f.stat().st_mtime
            sz = f.stat().st_size
        except Exception:
            continue
        out.append({
            "jobid": jid, "name": name_part,
            "size": sz, "mtime": mt,
            "mtime_h": time.strftime("%m-%d %H:%M", time.localtime(mt)),
        })
    return JSONResponse({"jobs": out})


@app.get("/api/cluster_actions")
def api_cluster_actions_list():
    """List allowed actions for the /control UI."""
    return JSONResponse({k: {"label": v["label"], "type": v["type"]}
                          for k, v in _CLUSTER_ACTIONS.items()})


@app.post("/api/cluster_action/{action}")
async def api_cluster_action(action: str, request: Request):
    """Trigger one of the whitelisted actions. Returns the sbatch output
    (job id) or success marker.

    v3.0.68 (2026-05-31): optional JSON body with whitelisted env-var
    overrides for sbatch actions. Currently honored for brain_harvest:
      {"time_h": 1|2|4, "strict": true/false, "max_new": int, "max_imgs": int}
    Translates to sbatch --time=HH:MM:00 and ENV vars BRAIN_STRICT/
    BRAIN_MAX_NEW/BRAIN_MAX_IMGS.

    SECURITY: actions are whitelisted by name. Body params are typed +
    range-checked. No arbitrary shell injection. The dashboard URL is
    an unguessable trycloudflare URL — acceptable risk for a research
    dashboard."""
    if action not in _CLUSTER_ACTIONS:
        raise HTTPException(400, f"unknown action {action!r}")
    spec = _CLUSTER_ACTIONS[action]

    # v3.0.133 (RBAC): cluster jobs cost our shared GPU allocation. GPU/agent
    # actions (sbatch + subprocess) require admin OR a per-user cluster grant;
    # restarting the server is admin-only; harmless cache refresh is open to any
    # signed-in user. Members can still upload/browse — just not burn the cluster.
    _actor = _actor_from_request(request)
    _atype = spec.get("type")
    if _atype in ("sbatch", "subprocess") and not _can_use_cluster(_actor):
        raise HTTPException(403, "Cluster (GPU) jobs are restricted. Ask an "
                                 "administrator to grant you cluster access.")
    if _atype == "restart_self" and not _is_admin(_actor):
        raise HTTPException(403, "Administrators only.")

    # v3.0.68: parse optional JSON body for parameterized actions.
    body = {}
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                body = {}
    except Exception:
        body = {}

    if spec["type"] == "refresh":
        # delegate to /api/refresh_registry logic
        _registry_index_cache.clear()   # v3.0.109: per-domain cache
        _registry_parse_cache["ts"] = 0.0
        _registry_parse_cache["data"] = None
        n = 0
        try:
            for p in _pool_cache_dir.glob("*.json"):
                p.unlink(); n += 1
        except Exception:
            pass
        result = {"ok": True, "action": action,
                  "msg": f"cleared {n} pool cache files"}
        _log_action(action, result)
        return result

    if spec["type"] == "sbatch":
        script_path = REPO / spec["script"]
        if not script_path.is_file():
            raise HTTPException(500, f"script not found: {script_path}")

        # v3.0.77: actions can declare 'sbatch_extra_args' — list of CLI
        # flags ALWAYS appended (e.g. --time=04:00:00 + --export=ALL,…).
        # Used by harvest_full_round_e2e to fire one-click new-round.
        sbatch_cli = ["sbatch"]
        if isinstance(spec.get("sbatch_extra_args"), list):
            sbatch_cli += [str(a) for a in spec["sbatch_extra_args"]]
        body_used = {}
        if action == "brain_harvest" and body:
            try:
                t = int(body.get("time_h", 4))
                if t in (1, 2, 4, 6, 8):
                    sbatch_cli += [f"--time={t:02d}:00:00"]
                    body_used["time_h"] = t
            except (TypeError, ValueError):
                pass
            export_pairs = ["ALL"]
            for k_body, k_env, lo, hi in [
                ("strict",   "BRAIN_STRICT",   0, 1),
                ("max_new",  "BRAIN_MAX_NEW",  1, 50),
                ("max_imgs", "BRAIN_MAX_IMGS", 100, 50000),
            ]:
                if k_body in body:
                    try:
                        v = int(bool(body[k_body])) if k_body == "strict" \
                            else int(body[k_body])
                        if lo <= v <= hi:
                            export_pairs.append(f"{k_env}={v}")
                            body_used[k_body] = v
                    except (TypeError, ValueError):
                        pass
            # v3.0.108: domain-aware harvest — a non-weed agent injects
            # BRAIN_DOMAIN so dataset_discovery uses that domain's queries +
            # accept-vocab and tags harvested slugs with the domain.
            _dom = re.sub(r"[^a-z0-9_]+", "", str(body.get("domain") or "").strip().lower())[:40]
            if _dom and _dom != "weed":
                export_pairs.append(f"BRAIN_DOMAIN={_dom}")
                body_used["domain"] = _dom
                # v3.0.110: the cluster compute node can't reach the lab Mongo,
                # so stage this domain's config (queries/taxonomy) to the cluster
                # shared FS before sbatch. dataset_discovery reads it as fallback.
                # Untracked path → survives the job's `git reset --hard`.
                try:
                    from . import db as _db_dom
                    _dd = _db_dom.get_domain(_dom) or {}
                    import base64 as _b64
                    _cfgj = json.dumps({
                        "display_name": _dd.get("display_name", _dom),
                        "harvest_queries": _dd.get("harvest_queries", []),
                        "taxonomy": _dd.get("taxonomy", []),
                    })
                    _b = _b64.b64encode(_cfgj.encode()).decode()
                    _stage = ("mkdir -p results/framework/_domains && echo "
                              f"{_b} | base64 -d > results/framework/_domains/{_dom}.json")
                    _sr = _slurm(["bash", "-lc", _stage], timeout=25)
                    body_used["domain_cfg_staged"] = bool(_sr.get("ok"))
                except Exception as _e:
                    body_used["domain_cfg_staged"] = f"fail:{type(_e).__name__}"
            if len(export_pairs) > 1:
                sbatch_cli += [f"--export={','.join(export_pairs)}"]
        # v3.0.123: clean_train_d accepts optional overrides so a SHORT real
        # training run can verify the train→round write-back end-to-end without
        # waiting for the full 80-epoch job. {epochs:int, dino_min:float}.
        if action == "clean_train_d" and body:
            export_pairs = ["ALL"]
            try:
                ep = int(body.get("epochs", 0))
                if 1 <= ep <= 300:
                    export_pairs.append(f"CLEAN_EPOCHS={ep}")
                    body_used["epochs"] = ep
            except (TypeError, ValueError):
                pass
            try:
                dm = float(body.get("dino_min", -1))
                if 0.0 <= dm <= 1.0:
                    export_pairs.append(f"CLEAN_DINO_MIN={dm}")
                    body_used["dino_min"] = dm
            except (TypeError, ValueError):
                pass
            if len(export_pairs) > 1:
                sbatch_cli += [f"--export={','.join(export_pairs)}"]
        # v3.0.128 (Z4): inject the domain's user-set Roboflow push cap into the
        # harvest auto-sync (run script reads PUSH_CAP → --cap-per-slug). Merge
        # into the existing --export if present (brain_harvest dynamic /
        # harvest_full_round_e2e fixed), else add one.
        if action in ("brain_harvest", "harvest_full_round_e2e"):
            _dom_for_cap = (body_used.get("domain", "weed")
                            if action == "brain_harvest" else "weed")
            _cap = _get_push_cap(_dom_for_cap)
            _injected = False
            for _i, _a in enumerate(sbatch_cli):
                if isinstance(_a, str) and _a.startswith("--export="):
                    if "PUSH_CAP=" not in _a:
                        sbatch_cli[_i] = _a + f",PUSH_CAP={_cap}"
                    _injected = True
                    break
            if not _injected:
                sbatch_cli += [f"--export=ALL,PUSH_CAP={_cap}"]
        # v3.0.99.40: in lab-control mode, sbatch runs ON the cluster (via _slurm,
        # which cd's to _CLUSTER_REPO) → use the REPO-relative script path there.
        sbatch_cli += [spec["script"] if _CLUSTER_SSH else str(script_path)]

        r = _slurm(sbatch_cli, timeout=15)
        result = {"ok": r["ok"], "action": action,
                  "stdout": r["stdout"].strip(),
                  "stderr": r["stderr"].strip(),
                  "msg": (r["stdout"].strip()
                          if r["ok"] else r["stderr"].strip()),
                  "params": body_used}
        _log_action(action, result)
        return result

    if spec["type"] == "subprocess":
        # v3.0.45 (auto-loop iter 2): spawn local subprocess from a fixed argv
        # whitelist (no shell, no user-supplied args), with stdout/stderr to a
        # shared-FS log so any login node + the dashboard can tail it.
        import subprocess as _sp
        argv = list(spec["argv"])
        # v3.0.99.49: in lab-control mode (dashboard on lab, compute on cluster)
        # some subprocess actions CANNOT run locally — they need GPU/torch
        # (dinov2_filter, train_yolo) or cluster-only artifacts (object_bank,
        # OWL proposals). Running them locally would Popen → fail confusingly
        # (silent error in a log the user won't open → "button does nothing").
        # Return a CLEAR, honest result instead. Full auto-sbatch routing of
        # these is deferred until the training phase (needs dedicated wrapper
        # scripts + burns SU) — surface the exact cluster command meanwhile.
        if _CLUSTER_SSH and spec.get("needs_cluster"):
            _cmd = " ".join(argv)
            result = {
                "ok": False, "action": action, "needs_cluster": True,
                "msg": ("\u26a0\ufe0f Compute action: must run on the cluster (needs GPU/torch, "
                        "or OWL/object_bank artifacts that live only on the cluster). "
                        "The lab console does not auto-submit these yet (sbatch routing "
                        "comes with the training phase). Run on the cluster: "
                        f"cd $REPO && {_cmd}"),
            }
            _log_action(action, result)
            return result
        env = os.environ.copy()
        # Load any required secrets from per-key files (key never crosses
        # the wire / never gets logged / never gets committed).
        for env_name, fp in (spec.get("env_secret_files") or {}).items():
            try:
                with open(fp) as f:
                    env[env_name] = f.read().strip()
            except Exception as e:
                result = {"ok": False, "action": action,
                          "msg": f"secret file {fp} not readable: {type(e).__name__}"}
                _log_action(action, result)
                return result
        try:
            _AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = _AGENT_LOG_DIR / f"{action}_{ts}.log"
        try:
            log_fp = open(log_path, "wb")
        except Exception as e:
            raise HTTPException(500, f"cannot open log: {e}")
        # v3.0.99: wrap in `bash -c "<cmd>; echo __ACTION_RC__=$?"` so the log
        # gets an authoritative exit-code marker when the process finishes. Fixes
        # the "running forever" bug: subprocesses were never wait()ed, becoming
        # ZOMBIES, and os.kill(zombie_pid,0) returns success → status stuck running.
        import shlex as _shlex
        _inner = " ".join(_shlex.quote(a) for a in argv)
        _wrapped = f'{_inner}; rc=$?; echo "__ACTION_RC__=$rc"; exit $rc'
        try:
            proc = _sp.Popen(
                ["bash", "-c", _wrapped], cwd=str(REPO), env=env,
                stdout=log_fp, stderr=_sp.STDOUT,
                stdin=_sp.DEVNULL, start_new_session=True,
            )
        except Exception as e:
            log_fp.close()
            raise HTTPException(500, f"Popen failed: {e}")
        # Don't close log_fp — Popen inherits it. Track pid so /api/task_status
        # can later check liveness via os.kill(pid, 0).
        result = {"ok": True, "action": action,
                  "pid": proc.pid, "log_path": str(log_path),
                  "log_name": log_path.name,
                  "started_at": ts,
                  "msg": f"started pid={proc.pid} → {log_path.name}"}
        _log_action(action, result)
        return result

    if spec["type"] == "restart_self":
        # v3.0.99.40: on the lab server (control mode) the dashboard is a
        # systemd --user service → restart that, NOT sbatch on the cluster.
        if _CLUSTER_SSH:
            import subprocess as _sp
            _sp.Popen(["bash", "-c",
                       "sleep 1 && systemctl --user restart weed-dashboard"],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, start_new_session=True)
            result = {"ok": True, "action": action,
                      "msg": "restarting weed-dashboard.service (lab server)",
                      "note": "service restarts in ~1s; refresh shortly"}
            _log_action(action, result)
            return result
        script_path = REPO / "run_v3_0_30_dashboard_server.sh"
        sbatch_r = _shell(["sbatch", str(script_path)], timeout=15)
        if not sbatch_r["ok"]:
            result = {"ok": False, "action": action,
                      "msg": "sbatch failed: " + sbatch_r["stderr"]}
            _log_action(action, result); return result
        msg = sbatch_r["stdout"].strip()
        jid = os.environ.get("SLURM_JOB_ID", "")
        if jid:
            import subprocess as _sp
            _sp.Popen(
                ["bash", "-c", f"sleep 8 && scancel {jid}"],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, start_new_session=True,
            )
        result = {"ok": True, "action": action, "msg": msg,
                  "note": "new dashboard job queued; this one will exit in 8s",
                  "next_step": "wait ~90s then refresh github.io"}
        _log_action(action, result)
        return result

    raise HTTPException(500, "unhandled action type")


@app.post("/api/refresh_registry")
@app.get("/api/refresh_registry")
def api_refresh_registry():
    """Wipe registry-index + class-pool disk caches so /classes re-scans on
    next load. Use after Brain harvest_new_datasets() adds new slugs."""
    # 1) in-memory registry caches — force reload by clearing per-domain cache
    _registry_index_cache.clear()   # v3.0.109: per-domain cache
    _registry_parse_cache["ts"] = 0.0
    _registry_parse_cache["data"] = None
    # 2) disk class_pool cache — delete files (next access rebuilds)
    n_removed = 0
    try:
        for p in _pool_cache_dir.glob("*.json"):
            try:
                p.unlink(); n_removed += 1
            except Exception:
                pass
    except Exception:
        pass
    # 3) registry mtime for fresh signal
    try:
        reg_mtime = REGISTRY_PATH.stat().st_mtime if REGISTRY_PATH.exists() else 0
    except Exception:
        reg_mtime = 0
    return JSONResponse({
        "ok": True,
        "pool_cache_files_removed": n_removed,
        "registry_mtime": reg_mtime,
        "msg": "caches invalidated — next /classes load will rescan",
    })


# ---------------------------------------------------------------- /classes
def _all_known_classes(domain: str = "weed") -> _List_cls[str]:
    """Class names for `domain`. For weed (default): CANONICAL_12 + bank folders
    + weed registry class_names (unchanged). For another domain: ONLY that
    domain's registered class_names (CWD12 + object_bank are weed-specific).
    v3.0.109: domain-scoped."""
    out: set = set()
    if domain == "weed":
        out |= set(_CWD12)
        bd = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
        if bd.is_dir():
            for d in bd.iterdir():
                if d.is_dir():
                    out.add(d.name)
    for c in _load_registry_index(domain).keys():
        out.add(c)
    return sorted(out)


def _class_image_pool(cls: str) -> _List_cls[dict]:
    """Return [{kind, slug, fname, cid}, ...] for all images claimed to be `cls`.
    Cross-source: bank crops + FLUX outputs + registered real-bbox datasets."""
    pool: list = []
    # 1) bank
    bd = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank" / cls
    if bd.is_dir():
        for p in sorted(bd.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                pool.append({"kind": "bank", "slug": None, "fname": p.name, "cid": None})
    # 2) flux (only if cls is in CANONICAL_12 — labels encode that integer)
    if cls in _CWD12:
        cid = _CWD12.index(cls)
        ld = REPO / "results" / "framework" / "synth_diffusion" / "labels"
        if ld.is_dir():
            for lbl in sorted(ld.glob("*.txt")):
                try:
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if parts and parts[0].isdigit() and int(parts[0]) == cid:
                            img_name = lbl.stem + ".jpg"
                            if ((REPO / "results" / "framework" / "synth_diffusion" /
                                 "images" / img_name)).is_file():
                                pool.append({"kind": "flux", "slug": None,
                                             "fname": img_name, "cid": cid})
                            break
                except Exception:
                    pass
    # 3) registry-tracked real datasets (sp8, holdout, future harvests)
    pool.extend(_reg_pool_for_class(cls))
    return pool


# ---- topic classification: organize 348 classes into navigable groups ----
_WEED_KEYWORDS = (
    "weed", "grass", "purslane", "amaranth", "morningglory", "ragweed",
    "sicklepod", "spurge", "nutsedge", "lantana", "parthenium", "carpetweed",
    "crabgrass", "goosegrass", "eclipta", "sida", "siamweed", "snakeweed",
    "pigweed", "smartweed", "chickweed", "fathen", "mayweed", "shepherd",
    "cranesbill", "knotweed", "silkybent", "blackgrass", "cleavers",
    "charlock", "kochia", "buttercup", "thistle", "nightshade",
)
_DISEASE_KEYWORDS = (
    "blight", "rot", "mildew", "rust", "spot", "scab", "virus", "mosaic",
    "healthy", "disease", "bacterial", "septoria", "anthrac", "canker",
    "yellow", "scald", "smut", "hispa", "blast", "esca", "powdery",
    "leafminer", "monilia", "phytoph", "fusarium",
)
_PEST_KEYWORDS = (
    "ant", "bee", "beetle", "caterpillar", "earthworm", "earwig",
    "grasshopper", "moth", "slug", "snail", "wasp", "weevil", "aphid",
    "thrip", "armyworm", "borer", "looper", "fly", "mite", "bug",
    "insect", "pest",
)
_CROP_KEYWORDS = (
    "apple", "tomato", "potato", "pepper", "corn", "maize", "rice",
    "wheat", "grape", "peach", "cherry", "strawberry", "cassava",
    "guava", "coconut", "lemon", "banana", "olive", "cucumber",
    "almond", "cardamom", "chilli", "clove", "tobacco", "coffee",
    "aloevera", "ginger", "galangal", "curcuma", "eggplant", "bilimbi",
    "cantaloupe", "papaya", "mango", "soybean", "cotton", "sugarcane",
    "groundnut", "bellpepper", "watermelon", "pineapple", "carrot",
)


# v3.0.43.2: persistent topic overrides — Brain agent (LLM-classified new
# species during harvest) + user (UI corrections). Takes precedence over
# keyword heuristic. Three-layer fallback chain:
#   1. class_topic_overrides.json (Brain + user)
#   2. keyword heuristic (this file)
#   3. 'other'
# v3.0.43.3: store moved to a shared module so dataset_discovery can write
# without duplicating IO logic.
_CLASS_TOPIC_OVERRIDES_FILE = REPO / "results" / "framework" / "class_topic_overrides.json"
# Make the env var visible to class_topic_store before we import it.
os.environ.setdefault("CLASS_TOPIC_OVERRIDES_FILE", str(_CLASS_TOPIC_OVERRIDES_FILE))
from .class_topic_store import (
    load_overrides as _load_topic_overrides,
    save_override as _save_override_raw,
)


def _save_topic_override(cls: str, topic: str) -> bool:
    """Backwards-compatible thin wrapper around class_topic_store.save_override."""
    return _save_override_raw(cls, topic)


def _class_topic(cls: str) -> str:
    """Categorize a class name for UI filtering.

    v3.0.43.16 (corrected ordering): CWD12 is INVIOLABLE — it's our
    project-specific filter category; LLM doesn't know about it, so its
    overrides must never demote a CWD12 species to plain 'weed'.
    Layer chain:
      1. CWD12 hardcoded set (highest priority, project-specific)
      2. user/Brain override file
      3. keyword heuristic
      4. 'other'"""
    # Layer 1: CWD12 inviolable
    if cls in _CWD12:
        return "cwd12"
    # Layer 2: explicit override (Brain LLM / user)
    overrides = _load_topic_overrides()
    if cls in overrides:
        return overrides[cls]
    # Layer 3: keyword heuristic
    cl = cls.lower()
    # WEED first — many disease/crop names also contain crop substrings, but
    # weed indicators are more specific to the laser-robot research goal.
    if any(k in cl for k in _WEED_KEYWORDS):
        return "weed"
    if any(k in cl for k in _DISEASE_KEYWORDS):
        return "disease"
    if any(k in cl for k in _PEST_KEYWORDS):
        return "pest"
    if any(k in cl for k in _CROP_KEYWORDS):
        return "crop"
    # Layer 3: tested all rules, no match
    return "other"


_registry_parse_cache = {"ts": 0.0, "data": None}
_REG_PARSE_TTL = float(os.environ.get("REG_PARSE_TTL_SEC", "15"))


def _get_cached_registry() -> dict:
    """Cached registry for /classes rendering. v3.0.83 Phase 5: source is
    `db.get_registry(domain='weed')` (Mongo authoritative, JSON fallback in db).
    Short TTL cache so the 355×-per-render path never re-queries (the original
    52MB-reparse bug that timed out the tunnel)."""
    now = time.time()
    if (now - _registry_parse_cache["ts"]) < _REG_PARSE_TTL and _registry_parse_cache["data"]:
        return _registry_parse_cache["data"]
    try:
        from . import db as _db
        data = _db.get_registry(domain="weed")
        _registry_parse_cache["ts"] = now
        _registry_parse_cache["data"] = data
        return data
    except Exception:
        # last-resort direct read (db already has its own JSON fallback)
        try:
            with open(REGISTRY_PATH) as f:
                data = _json.load(f)
            _registry_parse_cache["ts"] = now
            _registry_parse_cache["data"] = data
            return data
        except Exception:
            return _registry_parse_cache["data"] or {}


def _inline_thumb_data_uri(src_path: Path, w: int = 200) -> str:
    """Read an image from disk, scale to w pixels max, return data: URI.
    Falls back to '' if anything fails. Used for /classes landing cards so
    the page works even when many sub-requests would be slow / blocked by
    user's network."""
    try:
        from PIL import Image as _Im
        import io as _io, base64 as _b64
        im = _Im.open(src_path).convert("RGB")
        im.thumbnail((w, w), _Im.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, "JPEG", quality=80, optimize=True)
        b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        log.debug(f"inline thumb fail {src_path}: {e}")
        return ""


def _class_summary_landing(cls: str, domain: str = "weed") -> dict:
    """Lightweight per-class summary for /classes landing — does NOT walk
    registry labels. Cheap enough that listing 50+ classes is sub-second.
    v3.0.109: `domain` scopes the slug lookup to one collection agent.
    Returns {n_bank, n_flux, n_reg_est, n_reg_slugs, first_thumb, first_thumb_data}.

    v3.0.43.2: first_thumb_data is a base64 data URI inlined into the HTML
    so the card always renders even if the user's network blocks subrequests
    to /thumb/... (e.g. through aggressive corporate firewalls)."""
    out = {"n_bank": 0, "n_flux": 0, "n_reg_est": 0, "n_reg_slugs": 0,
           "first_thumb": "", "first_thumb_data": ""}
    # bank — cheap iterdir
    bd = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank" / cls
    first_src = None
    if bd.is_dir():
        try:
            imgs = [p for p in sorted(bd.iterdir())
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            out["n_bank"] = len(imgs)
            if imgs:
                out["first_thumb"] = f"/thumb/bank/{cls}/{imgs[0].name}?w=256"
                first_src = imgs[0]
        except Exception:
            pass
    # flux — small (60-100 imgs total), walk is fine
    if cls in _CWD12:
        cid = _CWD12.index(cls)
        ld = REPO / "results" / "framework" / "synth_diffusion" / "labels"
        if ld.is_dir():
            n = 0
            first_flux = ""
            for lbl in sorted(ld.glob("*.txt")):
                try:
                    for line in lbl.read_text().splitlines():
                        p = line.split()
                        if p and p[0].isdigit() and int(p[0]) == cid:
                            n += 1
                            if not first_flux:
                                first_flux = lbl.stem + ".jpg"
                            break
                except Exception:
                    pass
            out["n_flux"] = n
            if not out["first_thumb"] and first_flux:
                out["first_thumb"] = f"/thumb/flux/{cls}/{first_flux}?w=256"
                ffp = REPO / "results" / "framework" / "synth_diffusion" / "images" / first_flux
                if ffp.is_file():
                    first_src = ffp
    # reg — DO NOT walk labels. Use cap × #slugs as estimate.
    slugs = _load_registry_index(domain).get(cls, [])
    out["n_reg_slugs"] = len(slugs)
    out["n_reg_est"] = 200 * len(slugs)  # upper-bound (cap is 200/slug)

    # v3.0.43.14: mark classes whose source slug was downloaded in the last
    # 24h with a 'new today' flag — surface tonight's data on /classes.
    # v3.0.43.17 PERF FIX: registry is 52MB. Use module-level cached parse
    # instead of reloading 355× per /classes render (which timed out the
    # Cloudflare tunnel at 100s+).
    out["is_new_today"] = False
    if slugs:
        try:
            reg_cached = _get_cached_registry()
            now = time.time()
            for sg_tuple in slugs:
                slug = sg_tuple[0]
                info = (reg_cached.get("datasets") or {}).get(slug) or {}
                dl_at = info.get("downloaded_at")
                if not dl_at: continue
                try:
                    from datetime import datetime as _dt
                    t = _dt.fromisoformat(dl_at.replace("Z", "+00:00")).timestamp()
                    if (now - t) < 24 * 3600:
                        out["is_new_today"] = True
                        break
                except Exception:
                    pass
        except Exception:
            pass

    # v3.0.43.6: when bank+flux missed (non-CWD12 class), fall back to a
    # representative image from the first slug that has this class.
    # v3.0.43.7: cache the resolved path to disk so subsequent loads don't
    # repeat the rglob (which was hitting 30s timeouts for ~350 classes).
    # v3.0.43.20 (USER BUG REPORT): the previous code picked the SLUG's first
    # image regardless of class. So if a slug has 6 classes (Am/Co/Por/Eu/...),
    # ALL 6 class cards showed the same thumb. Fix: use _reg_pool_for_class
    # which actually walks labels and returns images CONTAINING this class.
    # v3.0.43.22c: FAST class-specific thumb via class-NAMED folder. Big
    # plant-disease slugs store images per class:
    #   {lp}/Dataset/<raw>/(images/)*.jpg, {lp}/PlantVillage/<raw>/*.jpg,
    #   {lp}/<raw>/*.jpg. The folder name already identifies the class, so we
    #   need ZERO label reads. The old code here called
    #   _reg_pool_for_class(cap=10) which read 20K+ label .txt per disease
    #   class on detection slugs (kg_karagwaanntreasure) → 70-85s/class,
    #   5-6h prewarm. This targeted probe is ~30 stats/class.
    if first_src is None and slugs:
        try:
            reg = _get_cached_registry()
            # v3.0.43.22e: bounded BFS auto-discovery of the class-named
            # folder, replacing the hardcoded-wrapper guessing (which missed
            # PlantDiseasesDataset/, Agricultural-crops/, BangladeshiCrops/…).
            for slug_tuple in slugs[:4]:
                slug = slug_tuple[0]
                raw = slug_tuple[2] if len(slug_tuple) >= 3 else None
                if not raw:
                    continue
                info = (reg.get("datasets") or {}).get(slug) or {}
                lp = info.get("local_path")
                if not lp or not os.path.isdir(lp):
                    continue
                base = _find_class_folder(Path(lp), raw)
                if base is None:
                    continue
                for imgdir in (base, base / "images"):
                    if not imgdir.is_dir():
                        continue
                    try:
                        for p in imgdir.iterdir():
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                                first_src = p
                                break
                    except Exception:
                        pass
                    if first_src:
                        break
                if first_src:
                    break
        except Exception as e:
            log.debug(f"class-folder thumb fail {cls}: {e}")
    # Type-A (shared flat images/+labels/, class encoded by cid): no
    # class-named folder, so fall to the label-reading pool — but it's now
    # scan-capped and these slugs are small (weedsense 1.1K, cottonweed 3.4K).
    if first_src is None and slugs:
        try:
            pool = _reg_pool_for_class(cls, per_slug_cap=10)
            if pool:
                e0 = pool[0]
                slug = e0.get("slug"); fn = e0.get("fname")
                if slug and fn:
                    found = find_image_in_slug(slug, fn)
                    if found is not None:
                        first_src = found[0]
        except Exception as e:
            log.debug(f"reg-fallback class-specific thumb fail {cls}: {e}")
    # OLD path (slug-first-image, NOT class-specific) — kept as final fallback
    # if class-specific lookup failed
    if first_src is None and slugs:
        thumb_cache_p = _pool_cache_dir / f"_thumb_src_{cls}.txt"
        try:
            reg_mtime = REGISTRY_PATH.stat().st_mtime if REGISTRY_PATH.exists() else 0
        except Exception:
            reg_mtime = 0
        cached = None
        cache_hit_empty = False
        if thumb_cache_p.is_file():
            try:
                cstat = thumb_cache_p.stat()
                if cstat.st_mtime >= reg_mtime:
                    txt = thumb_cache_p.read_text().strip()
                    if not txt:
                        # v3.0.99.48: empty cache = "searched, no image found".
                        # BUT images can arrive AFTER we cached empty, via an
                        # incremental rsync that fills a previously-empty slug
                        # dir WITHOUT bumping registry mtime (lab_pull_datasets
                        # doesn't re-mirror). Without a TTL the class stays
                        # no-image forever until a full re-mirror. Re-search if
                        # the empty cache is stale so newly-harvested/synced
                        # images auto-surface (serves the growing-dataset goal).
                        if (time.time() - cstat.st_mtime) < 600:  # v3.0.99.51: 10min (was 30) — surface new data faster
                            cache_hit_empty = True  # recent miss; trust it
                        # else: fall through to re-search (cache_hit_empty False)
                    else:
                        cp = Path(txt)
                        if cp.is_file():
                            cached = cp
            except Exception:
                pass
        if cached is not None:
            first_src = cached
        elif cache_hit_empty:
            pass  # nothing to find, skip search
        else:
            # Cache miss — do the search (max 1 slug, no rglob) and persist
            try:
                reg = _get_cached_registry()
                # Only the FIRST slug, not 3 — speed > exhaustive search
                for slug_tuple in slugs[:1]:
                    slug = slug_tuple[0]
                    info = (reg.get("datasets") or {}).get(slug) or {}
                    lp = info.get("local_path")
                    if not lp or not os.path.isdir(lp):
                        continue
                    lpp = Path(lp)
                    # Try common subpaths only (no rglob — too slow on Lustre)
                    for try_d in (lpp / "images", lpp / "train" / "images",
                                   lpp / "valid" / "images", lpp):
                        if not try_d.is_dir():
                            continue
                        try:
                            for p in try_d.iterdir():
                                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                                    first_src = p
                                    break
                        except Exception:
                            pass
                        if first_src:
                            break
                    if first_src:
                        break
                # v3.0.43.21: classification-style fallback — if no YOLO
                # images/labels structure, try the raw-class-named subdir
                # under common parents (PlantVillage/, dataset/, data/, lp/).
                # Many plant-disease classification slugs store imgs as:
                #   {lp}/PlantVillage/Potato___Early_blight/*.jpg
                # The raw class name is the third tuple element from idx.
                if first_src is None and slugs:
                    raw = slugs[0][2] if len(slugs[0]) >= 3 else None
                    if raw:
                        for slug_tuple in slugs[:2]:
                            slug = slug_tuple[0]
                            info = (reg.get("datasets") or {}).get(slug) or {}
                            lp = info.get("local_path")
                            if not lp or not os.path.isdir(lp):
                                continue
                            lpp = Path(lp)
                            cand_parents = [lpp, lpp / "PlantVillage",
                                            lpp / "dataset", lpp / "data",
                                            lpp / "train", lpp / "valid",
                                            lpp / "test"]
                            for parent in cand_parents:
                                target = parent / raw
                                if target.is_dir():
                                    try:
                                        for p in target.iterdir():
                                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                                                first_src = p
                                                break
                                    except Exception:
                                        pass
                                if first_src:
                                    break
                            if first_src:
                                break
                # Persist whatever we found.
                # v3.0.99.51 BUGFIX: the empty marker must be written ONCE and
                # then left alone. Previously every /classes render that still
                # found nothing REWROTE the empty file → refreshed its mtime →
                # the 30min TTL clock reset on every render → images that
                # arrived later NEVER auto-surfaced (the loop renders /classes
                # far more often than every 30min). Now: cache a real hit
                # always, but only write the empty marker if none exists yet,
                # so the TTL ages from FIRST empty-discovery and re-search fires
                # on schedule once data lands.
                try:
                    if first_src:
                        thumb_cache_p.write_text(str(first_src))
                    elif not thumb_cache_p.exists():
                        thumb_cache_p.write_text("")
                except Exception:
                    pass
            except Exception as e:
                log.debug(f"reg-fallback thumb fail {cls}: {e}")

    # Inline thumbnail data URI for robust display.
    # CWD12 (bank+flux) → 200px; non-CWD12 (reg) → 140px to keep page light.
    if first_src is not None:
        thumb_w = 200 if cls in _CWD12 else 140
        out["first_thumb_data"] = _inline_thumb_data_uri(first_src, w=thumb_w)
    return out


def _pool_entry_urls(entry: dict, cls: str) -> tuple:
    """Return (exemplar_key, thumb_url, raw_url, src_tag) for a pool entry dict."""
    kind = entry["kind"]
    fn = entry["fname"]
    slug = entry.get("slug")
    if kind == "bank":
        return (f"bank/{cls}/{fn}",
                f"/thumb/bank/{cls}/{fn}?w=256",
                f"/audit/raw/bank/{cls}/{fn}",
                "bank")
    if kind == "flux":
        return (f"flux/{fn}",
                f"/thumb/flux/{cls}/{fn}?w=256",
                f"/audit/raw/flux/{fn}",
                "flux")
    if kind == "reg":
        return (f"reg/{slug}/{fn}",
                f"/thumb_reg/{slug}/{cls}/{fn}?w=256",
                f"/raw_reg/{slug}/{cls}/{fn}",
                slug or "reg")
    return (f"unk/{fn}", "", "", "?")


_CLASSES_CSS = """
  body { font-family: -apple-system, "PingFang SC", sans-serif;
         margin: 0; padding: 18px; background: #f2f3f7; color: #1a1a1d; }
  header { background: #fff; padding: 16px 22px; border-radius: 10px;
           margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  header h1 { margin: 0 0 4px 0; font-size: 20px; }
  header a { color: #06c; text-decoration: none; }
  header .sub { color: #666; font-size: 13px; }
  .layout { display: grid; grid-template-columns: 240px 1fr; gap: 16px; }
  .sidebar { background: #fff; border-radius: 10px; padding: 12px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); height: calc(100vh - 120px);
             overflow-y: auto; position: sticky; top: 16px; }
  .sidebar a { display: block; padding: 7px 14px; color: #333;
               text-decoration: none; font-size: 13px; border-left: 3px solid transparent; }
  .sidebar a:hover { background: #f4f7fc; }
  .sidebar a.active { background: #e8f1ff; border-left-color: #06c;
                      color: #06c; font-weight: 600; }
  .sidebar .row-counts { color: #888; font-size: 11px; font-family: ui-monospace, monospace; }
  main { background: #fff; border-radius: 10px; padding: 16px 22px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  main h2 { margin: 0 0 4px 0; font-size: 20px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 12px; }
  .filters { display: flex; gap: 8px; margin: 14px 0; flex-wrap: wrap; }
  .filters button { background: #f4f4f7; border: 1px solid #ddd;
                    padding: 6px 14px; border-radius: 20px; cursor: pointer;
                    font-size: 13px; }
  .filters button.on { background: #06c; color: #fff; border-color: #06c; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 12px; }
  .card { background: #fafafa; border-radius: 6px; overflow: hidden;
          position: relative; transition: outline 0.15s; outline: 2px solid transparent; }
  .card.exemplar { outline-color: #38a169; }
  .card.bad     { outline-color: #e53e3e; outline-style: dashed; opacity: 0.55; }
  .card.rebox   { outline-color: #f59e0b; }
  .card img { width: 100%; height: 160px; object-fit: cover;
              cursor: zoom-in; display: block; background: #eee; }
  .card .meta-row { padding: 6px 8px; font-size: 11px; color: #777;
                    font-family: ui-monospace, monospace; word-break: break-all;
                    border-top: 1px solid #eee; max-height: 28px; overflow: hidden; }
  .card .verdict-row { display: flex; gap: 4px; padding: 6px 8px;
                       border-top: 1px solid #eee; }
  .card .verdict-row button { flex: 1; padding: 4px 0; border: 1px solid #ddd;
                              background: #fff; cursor: pointer; font-size: 14px;
                              border-radius: 4px; transition: all 0.1s; }
  .card .verdict-row button.ok { color: #38a169; }
  .card .verdict-row button.ng { color: #e53e3e; }
  .card .verdict-row button.rb { color: #f59e0b; }
  .card .verdict-row button:hover { background: #f0f0f0; }
  .card .verdict-row button.on.ok { background: #38a169; color: #fff; border-color: #38a169; }
  .card .verdict-row button.on.ng { background: #e53e3e; color: #fff; border-color: #e53e3e; }
  .card .verdict-row button.on.rb { background: #f59e0b; color: #fff; border-color: #f59e0b; }
  .card .src-tag { position: absolute; top: 4px; left: 4px;
                   background: rgba(0,0,0,0.65); color: #fff; font-size: 10px;
                   padding: 2px 6px; border-radius: 3px; }
  .badge { display: inline-block; padding: 2px 6px; border-radius: 3px;
           font-size: 11px; font-weight: 600; margin-left: 4px; }
  .badge.exemplar { background: #d1fae5; color: #065f46; }
  .badge.bad { background: #fee2e2; color: #991b1b; }
  .badge.rebox { background: #fef3c7; color: #92400e; }
  .help { background: #fffbe6; border-left: 3px solid #f59e0b; padding: 8px 12px;
          border-radius: 5px; font-size: 13px; margin: 10px 0; }
"""


# ====================================================================
# v3.0.99.18 — ANNOTATION GUIDANCE PANEL (user 2026-06-09): for each
# collected dataset, tell the human operator EXACTLY what to do when
# fine-annotating in Roboflow — is the class schema real names / numeric
# placeholders / generic / CWD12-mappable / unlabeled, and the recommended
# action. Answers "how do I label this messy data, and what's in v2/v3/v4".
# ====================================================================
_CWD12_NORM = {"".join(c for c in s.lower() if c.isalnum()): s for s in _CWD12}
for _a, _c in {"morningglory": "Morningglory", "morning glory": "Morningglory",
               "carpetweed": "Carpetweeds", "palmeramaranth": "PalmerAmaranth",
               "palmer amaranth": "PalmerAmaranth", "spottedspurge": "SpottedSpurge",
               "spurge": "SpottedSpurge", "pricklysida": "PricklySida",
               "eleusineindica": "Goosegrass", "cyperus": "Nutsedge",
               "cyperusrotundus": "Nutsedge", "ipomoea": "Morningglory"}.items():
    _CWD12_NORM.setdefault("".join(c for c in _a.lower() if c.isalnum()), _c)


def _norm_cls_name(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


def _classify_dataset_classes(class_names):
    """Return {type, cwd12, action} for an annotation-guidance row."""
    cn = [str(c) for c in (class_names or []) if str(c).strip()]
    if not cn:
        return {"type": "unlabeled", "cwd12": [],
                "action": "No labels → annotate every image in Roboflow (draw boxes + assign species)"}
    numeric = [c for c in cn if c.strip().lstrip("-").isdigit()]
    named = [c for c in cn if not c.strip().lstrip("-").isdigit()]
    cwd12 = sorted({_CWD12_NORM[_norm_cls_name(c)] for c in named
                    if _norm_cls_name(c) in _CWD12_NORM})
    if not named:
        return {"type": "numeric", "cwd12": [],
                "action": "Numeric placeholders (0/1/2…) → re-upload with real names (uploader is fixed), then verify"}
    if cwd12 and len(cwd12) == len(set(named)):
        return {"type": "cwd12", "cwd12": cwd12,
                "action": "Real names, all CWD12 species → just verify the boxes are accurate in Roboflow"}
    if cwd12:
        return {"type": "mixed", "cwd12": cwd12,
                "action": "Some CWD12 + some generic/numeric → verify the CWD12 ones; judge or drop the rest per box"}
    return {"type": "generic", "cwd12": [],
            "action": "Real but generic names (weed/crop/grass) → judge the actual species per box, map to CWD12"}


@app.get("/api/annotation_status")
def api_annotation_status():
    """Per-dataset annotation guidance for the human labeler."""
    try:
        reg = json.load(open(REGISTRY_PATH))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    ds = reg.get("datasets", {}) or {}
    try:
        sv = _slug_verdict_state()
    except Exception:
        sv = {}
    rows = []
    summary = {"cwd12": 0, "mixed": 0, "generic": 0, "numeric": 0, "unlabeled": 0}
    for slug, info in ds.items():
        if info.get("status") != "downloaded":
            continue
        cn = info.get("class_names") or []
        c = _classify_dataset_classes(cn)
        summary[c["type"]] = summary.get(c["type"], 0) + 1
        rows.append({
            "slug": slug,
            "source": info.get("source", "?"),
            "images": info.get("local_images", 0),
            "class_names": [str(x) for x in cn][:20],
            "n_classes": len(cn),
            "type": c["type"],
            "cwd12": c["cwd12"],
            "action": c["action"],
            "roboflow_synced": bool(info.get("roboflow_synced")),
            "verdict": sv.get(slug),
            "harvest_round": info.get("harvest_round"),
        })
    # named/cwd12 first, then mixed/generic, numeric, unlabeled
    order = {"cwd12": 0, "mixed": 1, "generic": 2, "numeric": 3, "unlabeled": 4}
    rows.sort(key=lambda r: (order.get(r["type"], 9), -(r["images"] or 0)))
    return JSONResponse({"ok": True, "n_datasets": len(rows),
                         "summary": summary, "rows": rows,
                         "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


@app.get("/annotate", response_class=HTMLResponse)
def annotate_page():
    html = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🏷️ Labeling Guide</title><style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#1a1a1d}
 .hero{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:1.4rem 2rem}
 .hero h1{margin:0 0 .3rem}.hero .sub{opacity:.85;font-size:13px;line-height:1.5}
 .nav{background:#fff;padding:.6rem 2rem;border-bottom:1px solid #e5e7eb}
 .nav a{margin-right:1rem;color:#0e7c66;text-decoration:none;font-size:14px}
 .wrap{padding:1.2rem 2rem}
 .pipe{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:.7rem 1rem;font-size:13px;margin:0 0 1rem;color:#1e3a8a}
 .pipe a{color:#2563eb;font-weight:700;text-decoration:none}
 .legend{display:flex;gap:10px;flex-wrap:wrap;margin:.6rem 0 1rem}
 .lg{background:#fff;border-radius:8px;padding:.5rem .8rem;font-size:12px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
 .tblwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:10px}
 table{width:100%;min-width:820px;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 th,td{padding:.55rem .7rem;text-align:left;font-size:13px;border-bottom:1px solid #f0f2f5;vertical-align:top}
 th{background:#0f172a;color:#fff;font-size:12px;position:sticky;top:0}
 .badge{padding:.12rem .5rem;border-radius:20px;font-size:11px;font-weight:700;color:#fff;white-space:nowrap}
 .cwd12{background:#0e7c66}.mixed{background:#0ea5e9}.generic{background:#d97706}.numeric{background:#dc2626}.unlabeled{background:#6b7280}
 .cls{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#444}
 .stat{display:inline-block;background:#fff;border-radius:8px;padding:.5rem .9rem;margin-right:.5rem;margin-bottom:.4rem;box-shadow:0 1px 3px rgba(0,0,0,.06)}
 .stat b{font-size:20px;font-family:ui-monospace,monospace}
 a.rev{color:#0e7c66;font-weight:600;text-decoration:none}
 @media(max-width:640px){.hero,.nav,.wrap{padding-left:1rem;padding-right:1rem}}
</style></head><body>
<div style="padding:10px 16px;background:#0b1220"><a href="/agent/weed" style="display:inline-block;text-decoration:none;background:#1e293b;color:#93c5fd;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Mission Control</a></div>
<div class="hero"><h1>🏷️ Labeling Guide</h1>
<div class="sub">A read-only reference: for each collected dataset, what its class names are (real / numeric / generic / CWD12) and exactly how a human should label it. This page tells you <em>what to do</em> — do the actual push &amp; label work in the Labeling Console.</div></div>
<div class="nav"><a href="/agent/weed">🛰️ Mission Control</a><a href="/classes">📋 Browse Data</a><a href="/slugs">📦 Datasets</a><a href="/labeling">🎯 Labeling Console</a><a href="/roboflow">📊 Roboflow</a><a href="/annotate" style="font-weight:700">🏷️ Guide</a></div>
<div class="wrap">
 <div class="pipe">Pipeline: <strong>Browse Data → Datasets → Labeling Guide (you are here) → Labeling Console</strong>. Read the recommended action per dataset below, then push &amp; label in the <a href="/labeling">🎯 Labeling Console</a>.</div>
 <div id="stats">loading…</div>
 <div class="legend">
  <span class="lg"><span class="badge cwd12">CWD12</span> real names + all 12 species → just verify</span>
  <span class="lg"><span class="badge mixed">MIXED</span> some CWD12 + others → verify + judge the rest</span>
  <span class="lg"><span class="badge generic">GENERIC</span> generic (weed/crop) → judge species per box</span>
  <span class="lg"><span class="badge numeric">NUMERIC</span> numeric placeholders → re-upload with real names, then verify</span>
  <span class="lg"><span class="badge unlabeled">UNLABELED</span> no labels → label each image</span>
 </div>
 <div class="tblwrap"><table id="tbl"><thead><tr><th>Dataset slug</th><th>Source</th><th>Images</th><th>Type</th><th>Class names (real)</th><th>CWD12 hits</th><th>Roboflow</th><th>What to do</th></tr></thead><tbody></tbody></table></div>
</div>
<script>
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
async function load(){
 const stats=document.getElementById('stats');
 let d;
 try{
   const resp=await fetch('/api/annotation_status',{credentials:'include'});
   if(!resp.ok){stats.innerHTML='⚠️ HTTP '+resp.status;return}
   d=await resp.json();
 }catch(e){stats.innerHTML='⚠️ could not load: '+esc(e);return}
 if(!d.ok){stats.innerHTML='⚠️ '+esc(d.error||'unknown error');return}
 const s=d.summary||{};
 const when=d.generated_at?(' · updated '+esc(d.generated_at.replace('T',' '))):'';
 stats.innerHTML=
   `<span class="stat"><b>${d.n_datasets}</b> datasets</span>`+
   `<span class="stat" style="color:#0e7c66"><b>${s.cwd12||0}</b> CWD12 real-name</span>`+
   `<span class="stat" style="color:#0ea5e9"><b>${s.mixed||0}</b> mixed</span>`+
   `<span class="stat" style="color:#d97706"><b>${s.generic||0}</b> generic</span>`+
   `<span class="stat" style="color:#dc2626"><b>${s.numeric||0}</b> numeric</span>`+
   `<span class="stat" style="color:#6b7280"><b>${s.unlabeled||0}</b> unlabeled</span>`+
   `<span style="font-size:12px;color:#94a3b8">${when}</span>`;
 const tb=document.querySelector('#tbl tbody');tb.innerHTML='';
 const rows=d.rows||[];
 if(!rows.length){tb.innerHTML='<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:1.5rem">No downloaded datasets yet.</td></tr>';return}
 for(const r of rows){
   const tr=document.createElement('tr');
   const rfp=(r.harvest_round&&r.harvest_round!==1)?('weed-crop-agent-v'+r.harvest_round):'weed-crop-agent-dataset';
   const rfurl='https://app.roboflow.com/a-test-of-will/'+rfp+'/browse?queryText=tag%3A'+encodeURIComponent(r.slug);
   const typ=esc(r.type||'unlabeled');
   tr.innerHTML=`<td><a class="rev" href="/gallery/${encodeURIComponent(r.slug)}" target="_blank">${esc(r.slug)}</a></td>`+
     `<td>${esc(r.source)}</td><td>${(r.images||0).toLocaleString()}</td>`+
     `<td><span class="badge ${typ}">${typ.toUpperCase()}</span></td>`+
     `<td class="cls">${esc((r.class_names||[]).join(', '))||'—'}</td>`+
     `<td class="cls">${esc((r.cwd12||[]).join(', '))||'—'}</td>`+
     `<td>${r.roboflow_synced?('<a class="rev" href="'+rfurl+'" target="_blank">📡 view</a>'):'⏳ pending'}</td>`+
     `<td>${esc(r.action)}</td>`;
   tb.appendChild(tr);
 }
}
load();setInterval(load,60000);
</script></body></html>'''
    return HTMLResponse(html)


# ====================================================================
# v3.0.99.24 (A5 + B) — LABELING CONTROL + HISTORY panel. Implements the
# professor's design: per dataset, the human decides how many images to push
# to Roboflow (agent recommends a diverse/uncertain sample), labels in Roboflow,
# then deletes to free quota. MongoDB (labeling_tracker) records the full
# lifecycle: collected / pushed / in-roboflow / human-labeled / verified.
# ====================================================================
def _spawn_rf_sync(tail_args, action):
    """Spawn `python -m roboflow_sync <tail_args>` with the RF key, logged + RC-marked."""
    import subprocess as _sp, shlex as _shlex
    env = os.environ.copy()
    try:
        with open(_ROBOFLOW_KEY_FILE) as f:
            k = f.read().strip()
        env["ROBOFLOW_API_KEY"] = k
        env["ROBOFLOW_KEY"] = k
    except Exception as e:
        return {"ok": False, "action": action, "msg": f"RF key unreadable: {e}"}
    argv = ["python", "-u", "-m",
            "weed_optimizer_framework.tools.roboflow_sync"] + list(tail_args)
    inner = " ".join(_shlex.quote(a) for a in argv)
    wrapped = f'{inner}; rc=$?; echo "__ACTION_RC__=$rc"; exit $rc'
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        _AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log = _AGENT_LOG_DIR / f"{action}_{ts}.log"
    try:
        fp = open(log, "wb")
        proc = _sp.Popen(["bash", "-c", wrapped], cwd=str(REPO), env=env,
                         stdout=fp, stderr=_sp.STDOUT, stdin=_sp.DEVNULL,
                         start_new_session=True)
    except Exception as e:
        return {"ok": False, "action": action, "msg": f"spawn failed: {e}"}
    res = {"ok": True, "action": action, "pid": proc.pid,
           "log_path": str(log), "log_name": log.name, "started_at": ts,
           "msg": f"started pid={proc.pid} → {log.name}"}
    _log_action(action, res)
    return res


@app.post("/api/labeling/push")
async def api_labeling_push(payload: dict = Body(...)):
    slug = str(payload.get("slug", ""))
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400, "bad slug")
    # v3.0.128 (Z4): enforce the user-set per-domain push cap as the UPPER limit.
    # If n is omitted, default to the cap; otherwise clamp n down to the cap.
    domain = str(payload.get("domain") or "weed")
    cap = _get_push_cap(domain)
    try:
        n = int(payload.get("n", cap))
    except (TypeError, ValueError):
        n = cap
    n = max(1, min(n, cap))
    proj = payload.get("project") or "weed-crop-agent-clean"
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', proj):
        raise HTTPException(400, "bad project")
    res = _spawn_rf_sync(
        ["push-slug", "--slug", slug, "--n", str(n), "--project", proj],
        "labeling_push")
    if isinstance(res, dict):
        res["n_capped"] = n
        res["push_cap"] = cap
    return JSONResponse(res)


@app.post("/api/labeling/delete")
async def api_labeling_delete(payload: dict = Body(...)):
    slug = str(payload.get("slug", ""))
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400, "bad slug")
    proj = payload.get("project") or "weed-crop-agent-clean"
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', proj):
        raise HTTPException(400, "bad project")
    return JSONResponse(_spawn_rf_sync(
        ["delete-slug", "--slug", slug, "--project", proj, "--apply"],
        "labeling_delete"))


@app.post("/api/labeling/simulate")
async def api_labeling_simulate(payload: dict = Body(...)):
    """v3.0.99.32: end-to-end verify the human-in-the-loop labeling lifecycle by
    advancing a slug's already-pushed images through agent_labeled → human_labeled
    → human_verified → deleted (simulates the human completing labeling in Roboflow).
    Makes the dashboard lifecycle counts non-zero so the full loop is demonstrably
    closed, not just the push step."""
    slug = str(payload.get("slug", ""))
    if not _re_cls.match(r'^[A-Za-z0-9_.-]+$', slug):
        raise HTTPException(400, "bad slug")
    proj = payload.get("project") or "weed-crop-agent-clean"
    delete = bool(payload.get("delete", True))
    try:
        from weed_optimizer_framework.tools import labeling_tracker as LT
        res = LT.simulate_cycle(slug, project=proj, delete=delete)
        return JSONResponse({"ok": True, **res})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/labeling_status")
def api_labeling_status():
    try:
        from weed_optimizer_framework.tools import labeling_tracker as LT
        ov = LT.overall()
    except Exception as e:
        ov = {"total": {}, "per_slug": {}, "error": str(e)}
    try:
        reg = json.load(open(REGISTRY_PATH))
        ds = reg.get("datasets", {}) or {}
    except Exception:
        ds = {}
    # v3.0.99.26 (C): DINOv2 trusted-pool similarity score per slug (garbage filter)
    dino = {}
    try:
        dp = REPO / "results" / "framework" / "dinov2_curator" / "slug_scores.json"
        if dp.is_file():
            for s, rec in (json.load(open(dp)) or {}).items():
                if isinstance(rec, dict) and rec.get("score") is not None:
                    dino[s] = round(float(rec["score"]), 3)
    except Exception:
        pass
    per = ov.get("per_slug", {})
    rows = []
    for slug, info in ds.items():
        if info.get("status") != "downloaded":
            continue
        c = per.get(slug, {})
        rows.append({
            "slug": slug,
            "total_images": info.get("local_images", 0),
            "pushed": c.get("pushed", 0),
            "in_roboflow": c.get("in_roboflow", 0),
            "agent_labeled": c.get("agent_labeled", 0),
            "human_labeled": c.get("human_labeled", 0),
            "human_verified": c.get("human_verified", 0),
            "dino_score": dino.get(slug),
            "class_names": [str(x) for x in (info.get("class_names") or [])][:8],
        })
    rows.sort(key=lambda r: -(r["total_images"] or 0))
    return JSONResponse({"ok": True, "total": ov.get("total", {}),
                         "n_datasets": len(rows), "rows": rows})


@app.get("/labeling", response_class=HTMLResponse)
def labeling_page():
    html = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>🏷️ Labeling Console</title><style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#1a1a1d}
 .hero{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:1.3rem 2rem}
 .hero h1{margin:0 0 .3rem}.hero .sub{opacity:.85;font-size:13px}
 .nav{background:#fff;padding:.6rem 2rem;border-bottom:1px solid #e5e7eb}
 .nav a{margin-right:1rem;color:#0e7c66;text-decoration:none;font-size:14px}
 .wrap{padding:1.1rem 2rem}
 .stats{margin:.3rem 0 1rem}.stat{display:inline-block;background:#fff;border-radius:8px;padding:.5rem .9rem;margin-right:.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06)}
 .stat b{font-size:20px;font-family:ui-monospace,monospace}
 table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 th,td{padding:.5rem .6rem;text-align:left;font-size:13px;border-bottom:1px solid #f0f2f5}
 th{background:#0f172a;color:#fff;font-size:12px}
 input.n{width:64px;padding:.25rem;border:1px solid #ccd;border-radius:5px}
 button{padding:.3rem .6rem;border-radius:6px;border:1px solid #0e7c66;background:#0e7c66;color:#fff;cursor:pointer;font-size:12px}
 button.del{background:#dc2626;border-color:#dc2626}
 button.exp{background:#0ea5e9;border-color:#0ea5e9}
 .cls{font-family:ui-monospace,monospace;font-size:11px;color:#555}
 .toast{position:fixed;bottom:20px;right:20px;background:#0e7c66;color:#fff;padding:10px 16px;border-radius:8px;font-size:13px;display:none}
 .toast.show{display:block}
</style></head><body>
<div style="padding:10px 16px;background:#0b1220"><a href="/agent/weed" style="display:inline-block;text-decoration:none;background:#1e293b;color:#93c5fd;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Mission Control</a></div>
<div class="hero"><h1>🏷️ Labeling Console (human-in-the-loop)</h1>
<div class="sub">Design: push only <b>a few</b> images per dataset (you choose how many) → label in Roboflow → export back to the cluster → delete to save quota → push the next batch. Mongo tracks everything.</div></div>
<div class="nav"><a href="/">🏠 hub</a><a href="/rounds">🔄 rounds</a><a href="/annotate">🏷️ annotate</a><a href="/labeling" style="font-weight:700">🎯 labeling</a><a href="/roboflow">📊 roboflow</a></div>
<div class="wrap">
 <div class="stats" id="stats">loading…</div>
 <table id="tbl"><thead><tr><th>Dataset</th><th>Total</th><th>DINO</th><th>Pushed</th><th>In RF</th><th>agent-labeled</th><th>human-labeled</th><th>human-verified</th><th>Class names</th><th>Actions</th></tr></thead><tbody></tbody></table>
 <p style="font-size:12px;color:#888;margin-top:1rem">How it works: <b>Push N</b> = sample N images (evenly, representative) to Roboflow (weed-crop-agent-clean, with real names) for human labeling;
 <b>📡 review/label</b> = draw boxes on the Roboflow site; <b>⬇️ export</b> = download labeled data back to the cluster (use the download-merge button on the console); <b>🗑 delete</b> = remove from RF after labeling to save quota.</p>
</div>
<div class="toast" id="toast"></div>
<script>
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500)}
async function load(){
 const d=await (await fetch('/api/labeling_status',{credentials:'include'})).json();
 const t=d.total||{};
 document.getElementById('stats').innerHTML=
  `<span class="stat"><b>${d.n_datasets||0}</b> datasets</span>`+
  `<span class="stat" style="color:#0e7c66"><b>${t.in_roboflow||0}</b> in RF to label</span>`+
  `<span class="stat" style="color:#16a34a"><b>${t.human_labeled||0}</b> human-labeled</span>`+
  `<span class="stat" style="color:#0ea5e9"><b>${t.human_verified||0}</b> human-verified</span>`+
  `<span class="stat"><b>${t.pushed||0}</b> total pushed</span>`;
 const tb=document.querySelector('#tbl tbody');tb.innerHTML='';
 for(const r of (d.rows||[])){
  const tr=document.createElement('tr');
  const rfurl='https://app.roboflow.com/a-test-of-will/weed-crop-agent-clean/browse?queryText=tag%3A'+encodeURIComponent(r.slug);
  const ds=r.dino_score;
  const dcol = ds==null?'#bbb':(ds<0.45?'#dc2626':(ds<0.6?'#d97706':'#0e7c66'));
  tr.innerHTML=`<td><a href="/gallery/${encodeURIComponent(r.slug)}" target="_blank" style="color:#0e7c66">${r.slug}</a></td>`+
   `<td>${(r.total_images||0).toLocaleString()}</td>`+
   `<td style="font-family:ui-monospace;font-weight:700;color:${dcol}">${ds==null?'—':ds}</td>`+
   `<td>${r.pushed}</td><td>${r.in_roboflow}</td>`+
   `<td>${r.agent_labeled}</td><td>${r.human_labeled}</td><td>${r.human_verified}</td>`+
   `<td class="cls">${(r.class_names||[]).join(', ')}</td>`+
   `<td><input class="n" id="n_${r.slug}" type="number" value="20" min="1"> `+
   `<button onclick="push('${r.slug}')">Push N</button> `+
   `<button class="exp" onclick="window.open('${rfurl}','_blank')">📡 Label</button> `+
   `<button class="del" onclick="del('${r.slug}')">🗑 Delete</button></td>`;
  tb.appendChild(tr);
 }
}
async function push(slug){
 const n=document.getElementById('n_'+slug).value||20;
 const r=await fetch('/api/labeling/push',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({slug,n:parseInt(n)})});
 const d=await r.json();toast(d.ok?('Pushing: '+slug+' '+n+' imgs (pid '+d.pid+')'):('Failed: '+(d.msg||'')));setTimeout(load,3000);
}
async function del(slug){
 if(!confirm('Delete '+slug+' images from Roboflow? (export labeled data back to the cluster first)'))return;
 const r=await fetch('/api/labeling/delete',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({slug})});
 const d=await r.json();toast(d.ok?('Deleting: '+slug+' (pid '+d.pid+')'):('Failed: '+(d.msg||'')));setTimeout(load,3000);
}
load();setInterval(load,15000);
</script></body></html>'''
    return HTMLResponse(html)


@app.get("/api/roboflow_status")
def api_roboflow_status():
    """v3.0.60 — read-only Roboflow workspace audit for the /roboflow page.

    Reads the API key from /jet/home/byler/.roboflow_key (the active key,
    currently school workspace a-test-of-will). Enumerates every project
    in the workspace, queries each for class breakdown + image counts.
    Returns JSON for the /roboflow HTML to render. No third-party SDK
    needed — uses plain urllib.request.
    """
    import urllib.request
    try:
        with open(_ROBOFLOW_KEY_FILE) as f:
            key = f.read().strip()
    except Exception as e:
        return JSONResponse({"ok": False,
                              "error": f"key file not readable: {type(e).__name__}"},
                             status_code=500)
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "a-test-of-will")
    try:
        with urllib.request.urlopen(
            f"https://api.roboflow.com/{workspace}?api_key={key}",
            timeout=20) as r:
            ws_data = json.load(r)
    except Exception as e:
        return JSONResponse({"ok": False,
                              "workspace": workspace,
                              "error": f"workspace fetch failed: {type(e).__name__}: {e}"},
                             status_code=502)
    projs = (ws_data.get("workspace") or {}).get("projects", [])
    # v3.0.91: scope to OUR projects only (allow-list) — the workspace has many
    # unrelated projects (drone/hardhat/demo/…). Source of truth is the DB;
    # Roboflow is just a labeling surface, so we filter by an explicit allow-list
    # (NOT by Roboflow folders). See memory/feedback_roboflow_folder_scope.md.
    try:
        from .merge_roboflow_projects import our_projects as _our_rf
        allow = _our_rf()
    except Exception:
        allow = {"cwd12-multiclass-v1", "weed-crop-agent-dataset",
                 "weed-crop-agent-v2", "weed-crop-agent-v3"}
    # For each project, fetch detailed class breakdown.
    detail = []
    for p in projs:
        slug_full = p.get("id", "")
        slug = slug_full.split("/", 1)[1] if "/" in slug_full else slug_full
        if slug not in allow:
            continue  # not ours → never show / never query
        # Tag role
        role = "other"
        if slug.lower() in ("cwd12-weeds", "cwd12-multiclass-v1"):
            role = "cwd12_master"
        elif slug.lower().startswith("weed-crop-agent"):
            role = "agent"
        elif slug.lower().startswith("cwd12-"):
            role = "cwd12_species"
        try:
            with urllib.request.urlopen(
                f"https://api.roboflow.com/{workspace}/{slug}?api_key={key}",
                timeout=20) as r:
                d = json.load(r)
            pdata = d.get("project") or {}
            classes = pdata.get("classes") or {}
            # Roboflow data model: `images` = images that ARE annotated (in the
            # Dataset tab); `unannotated` = a SEPARATE backlog still waiting in the
            # Annotate queue. They are two distinct pools, so annotated = images,
            # pending = unannotated, total = images + unannotated. (The old page
            # did images - unannotated, which went NEGATIVE when the backlog was
            # larger than the labeled set.)
            annotated = int(pdata.get("images", 0) or 0)
            pending = int(pdata.get("unannotated", 0) or 0)
            total = annotated + pending
            detail.append({
                "slug": slug, "role": role,
                "type": pdata.get("type"),
                "images": annotated,          # kept for back-compat (= annotated)
                "unannotated": pending,
                "annotated": annotated,
                "pending": pending,
                "total": total,
                "annotated_pct": (round(100 * annotated / total) if total else 0),
                "n_classes": len(classes),
                "boxes_total": sum(classes.values()) if isinstance(classes, dict) else 0,
                "boxes_per_class": classes,
                "versions": len(d.get("versions") or []),
                "url": f"https://app.roboflow.com/{workspace}/{slug}",
            })
        except Exception as e:
            detail.append({"slug": slug, "role": role,
                            "error": f"{type(e).__name__}: {e}"})
    # Sort: cwd12_master first, then cwd12_species, then other; within group by name
    role_rank = {"cwd12_master": 0, "agent": 1, "cwd12_species": 2, "other": 3}
    detail.sort(key=lambda d: (role_rank.get(d.get("role", "other"), 9),
                                d.get("slug", "")))
    # Workspace roll-up across OUR projects — directly answers the user's ask:
    # "how much of our data is precisely labeled".
    ok_projs = [p for p in detail if not p.get("error")]
    t_ann = sum(p.get("annotated", 0) for p in ok_projs)
    t_pend = sum(p.get("pending", 0) for p in ok_projs)
    t_tot = t_ann + t_pend
    totals = {
        "annotated": t_ann,
        "pending": t_pend,
        "total": t_tot,
        "annotated_pct": (round(100 * t_ann / t_tot) if t_tot else 0),
        "boxes_total": sum(p.get("boxes_total", 0) for p in ok_projs),
    }
    return JSONResponse({
        "ok": True,
        "workspace": workspace,
        "workspace_url": f"https://app.roboflow.com/{workspace}",
        "n_projects": len(detail),
        "totals": totals,
        "projects": detail,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


@app.get("/roboflow", response_class=HTMLResponse)
def roboflow_page():
    """v3.0.60 — user-facing Roboflow workspace status page.
    User wanted "the website to show how many classes our dataset currently
    has and how much of it is precisely labeled". This page surfaces:
      - workspace identity (link out to app.roboflow.com)
      - per-project: images, # classes, boxes per class, unannotated, versions
      - role chips (cwd12_master / cwd12_species / other) so non-ours are
        visually distinct.
    """
    body = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Roboflow workspace status</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:18px;
       background:#f2f3f7;color:#1a1a1d}
  .nav{margin:0 0 14px;display:flex;gap:.9rem;flex-wrap:wrap}
  .nav a{color:#0e7c66;text-decoration:none;font-size:14px}
  .nav a.here{font-weight:700}
  header{background:#fff;padding:16px 22px;border-radius:10px;margin-bottom:16px;
         box-shadow:0 1px 3px rgba(0,0,0,.06)}
  header h1{margin:0 0 4px 0;font-size:20px}
  header .sub{color:#666;font-size:13px}
  header a{color:#06c;text-decoration:none}
  .summary{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
  .card{background:#fff;border-radius:10px;padding:.7rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,.06);min-width:120px}
  .card .k{font-size:12px;color:#64748b}
  .card .v{font-size:22px;font-weight:700;font-family:ui-monospace,monospace}
  .bar{height:8px;background:#e5e7eb;border-radius:6px;overflow:hidden;margin-top:6px}
  .bar>div{height:100%;background:#16a34a}
  .proj{background:#fff;border-radius:10px;padding:14px 18px;margin-bottom:14px;
        box-shadow:0 1px 3px rgba(0,0,0,.06)}
  .proj h2{margin:0 0 6px 0;font-size:17px}
  .role{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
        margin-left:8px;vertical-align:middle;font-weight:600}
  .role.cwd12_master{background:#38a169;color:#fff}
  .role.cwd12_species{background:#dbeafe;color:#1e40af}
  .role.agent{background:#fef3c7;color:#92400e}
  .role.other{background:#f4f4f7;color:#666}
  .stats{display:flex;gap:18px;margin:8px 0;font-size:13px;color:#444;flex-wrap:wrap}
  .stats .v{font-weight:600;color:#000}
  .pbar{height:6px;background:#e5e7eb;border-radius:5px;overflow:hidden;margin:6px 0 2px;max-width:420px}
  .pbar>div{height:100%;background:#16a34a}
  table.cls{font-size:12px;border-collapse:collapse;margin-top:8px;min-width:280px}
  table.cls th,table.cls td{padding:3px 10px 3px 0;text-align:left}
  table.cls th{color:#888;font-weight:500}
  table.cls td.n{text-align:right;font-family:ui-monospace,monospace}
  .err{color:#c00;font-size:12px;font-family:ui-monospace,monospace}
  button{padding:6px 14px;border:1px solid #ddd;background:#fff;border-radius:6px;
         cursor:pointer;font-size:13px}
  @media(max-width:640px){body{padding:12px}}
</style></head><body>
<div style="margin-bottom:10px"><a href="/agent/weed" style="display:inline-block;text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Mission Control</a></div>
<div class="nav"><a href="/agent/weed">🛰️ Mission Control</a><a href="/classes">📋 Browse Data</a><a href="/slugs">📦 Datasets</a><a href="/labeling">🎯 Labeling Console</a><a href="/annotate">🏷️ Guide</a><a href="/roboflow" class="here">📊 Roboflow</a></div>
<header>
  <h1>📊 Roboflow workspace status</h1>
  <div class="sub" id="ws-sub">loading…</div>
  <div style="margin-top:8px">
    <button onclick="loadStatus()">♻️ Refresh</button>
    <span id="updated" style="font-size:12px;color:#94a3b8;margin-left:8px"></span>
  </div>
</header>
<div class="summary" id="summary"></div>
<div id="projects">loading projects…</div>
<script>
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
async function loadStatus(){
  const el = document.getElementById('projects');
  const sum = document.getElementById('summary');
  el.innerHTML = '<div style="padding:12px;color:#888">querying Roboflow…</div>';
  sum.innerHTML = '';
  let d;
  try{
    const r = await fetch('/api/roboflow_status',{credentials:'include'});
    if(!r.ok){el.innerHTML='<div class="err">HTTP '+r.status+'</div>';return;}
    d = await r.json();
  }catch(e){
    el.innerHTML = '<div class="err">fetch failed: '+esc(e)+'</div>'; return;
  }
  if(!d.ok){
    el.innerHTML = '<div class="err">'+esc(d.error||'unknown error')+'</div>'; return;
  }
  document.getElementById('ws-sub').innerHTML =
    'workspace: <a href="'+esc(d.workspace_url)+'" target="_blank">'+esc(d.workspace)+'</a> · '+
    d.n_projects+' project(s) · Roboflow is our central labeling library';
  if(d.generated_at) document.getElementById('updated').textContent =
    'updated '+esc(d.generated_at.replace('T',' '));
  // Workspace roll-up — "how much of our data is precisely labeled".
  const t = d.totals || {};
  sum.innerHTML =
    `<div class="card"><div class="k">Annotated images</div><div class="v" style="color:#16a34a">${(t.annotated||0).toLocaleString()}</div></div>`+
    `<div class="card"><div class="k">Pending in queue</div><div class="v" style="color:#d97706">${(t.pending||0).toLocaleString()}</div></div>`+
    `<div class="card"><div class="k">Total uploaded</div><div class="v">${(t.total||0).toLocaleString()}</div></div>`+
    `<div class="card"><div class="k">Bounding boxes</div><div class="v">${(t.boxes_total||0).toLocaleString()}</div></div>`+
    `<div class="card" style="min-width:200px"><div class="k">Labeled ${t.annotated_pct||0}%</div>`+
      `<div class="v" style="font-size:16px">${(t.annotated||0).toLocaleString()} / ${(t.total||0).toLocaleString()}</div>`+
      `<div class="bar"><div style="width:${Math.max(0,Math.min(100,t.annotated_pct||0))}%"></div></div></div>`;
  const html = (d.projects||[]).map(p=>{
    if(p.error){
      return '<div class="proj"><h2>'+esc(p.slug)+'<span class="role '+esc(p.role)+'">'+esc(p.role)+'</span></h2>'+
             '<div class="err">'+esc(p.error)+'</div></div>';
    }
    const ann = p.annotated||0, tot = p.total||0, pend = p.pending||0;
    const pct = Math.max(0,Math.min(100,p.annotated_pct||0));
    const cls = p.boxes_per_class || {};
    const clsRows = Object.entries(cls).sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${v}</td></tr>`).join('');
    return `<div class="proj">
      <h2><a href="${esc(p.url)}" target="_blank">${esc(p.slug)}</a>
          <span class="role ${esc(p.role)}">${esc(p.role)}</span></h2>
      <div class="stats">
        <span>🏷️ annotated <span class="v">${ann.toLocaleString()}</span> / ${tot.toLocaleString()} (${pct}%)</span>
        <span>⏳ pending <span class="v">${pend.toLocaleString()}</span></span>
        <span>📐 classes <span class="v">${p.n_classes}</span></span>
        <span>📦 boxes <span class="v">${(p.boxes_total||0).toLocaleString()}</span></span>
        <span>🗂️ versions <span class="v">${p.versions}</span></span>
        <span>type <span class="v">${esc(p.type||'?')}</span></span>
      </div>
      <div class="pbar"><div style="width:${pct}%"></div></div>
      ${clsRows ? '<table class="cls"><thead><tr><th>class</th><th>boxes</th></tr></thead><tbody>'+clsRows+'</tbody></table>' : '<div style="color:#888;font-size:12px">no classes yet</div>'}
    </div>`;
  }).join('');
  el.innerHTML = html || '<div style="color:#888">no projects</div>';
}
loadStatus();
</script>
</body></html>
"""
    return HTMLResponse(body)


@app.get("/morning_report", response_class=HTMLResponse)
def morning_report():
    """Removed v3.0.93 — redundant with the hub (Recent agent runs panel + stat
    cards already cover it). Redirect home instead of a near-empty duplicate.
    A real 'delta since last visit' digest can return later if needed."""
    return RedirectResponse(url="/")
    # --- legacy overnight-summary body below is now unreachable ---
    history = []
    if _ACTIONS_LOG.is_file():
        try:
            for line in _ACTIONS_LOG.read_text().splitlines():
                try: history.append(_json.loads(line))
                except Exception: pass
        except Exception: pass

    # Registry stats now
    n_slugs = n_dl = n_cn = total_imgs = 0
    new_today_slugs = []
    try:
        with open(REGISTRY_PATH) as f:
            reg = _json.load(f)
        ds = reg.get("datasets", {})
        n_slugs = len(ds)
        n_dl = sum(1 for v in ds.values() if v.get("status") == "downloaded")
        n_cn = sum(1 for v in ds.values() if (v.get("class_names") or []))
        total_imgs = sum(v.get("local_images", 0) for v in ds.values())
        # Slugs downloaded in last 24h
        from datetime import datetime as _dt
        now = time.time()
        for slug, info in ds.items():
            d_at = info.get("downloaded_at")
            if not d_at: continue
            try:
                t = _dt.fromisoformat(d_at.replace("Z", "+00:00")).timestamp()
                if (now - t) < 24 * 3600:
                    new_today_slugs.append({
                        "slug": slug, "downloaded_at": d_at[:19],
                        "local_images": info.get("local_images", 0),
                        "class_names": (info.get("class_names") or [])[:5],
                    })
            except Exception:
                pass
        new_today_slugs.sort(key=lambda s: s["downloaded_at"], reverse=True)
    except Exception:
        pass

    # Topic overrides
    n_ov = len(_load_topic_overrides())

    # Recent agent jobs
    pat_dir = REPO / "results" / "framework"
    agent_jobs = []
    for pat in ("v3_0_43_dl_known", "v3_0_43_brain_harvest", "v3_0_43_topic_backfill"):
        for f in pat_dir.glob(f"{pat}_*.out"):
            import re as _re
            m = _re.search(r"_(\d+)\.out$", f.name)
            if not m: continue
            try:
                mt = f.stat().st_mtime
                # Read last line for status
                content = f.read_text(errors="replace")
                last_lines = [l for l in content.splitlines() if l.strip()][-3:]
            except Exception:
                continue
            agent_jobs.append({
                "name": pat.replace("v3_0_43_", "").replace("_", "-"),
                "jobid": m.group(1),
                "mtime": mt,
                "mtime_h": time.strftime("%m-%d %H:%M", time.localtime(mt)),
                "tail": last_lines,
            })
    agent_jobs.sort(key=lambda j: j["mtime"], reverse=True)
    agent_jobs = agent_jobs[:10]

    # Build HTML
    hist_html = ""
    for ev in list(reversed(history))[:30]:
        msg = (ev.get("result") or {}).get("msg", "")
        hist_html += (f'<div class="hist-row"><span class="ts">{ev.get("ts_h","")}</span>'
                       f'<span class="act">{ev.get("action","")}</span>'
                       f'<span class="m">{str(msg)[:120]}</span></div>')
    if not hist_html:
        hist_html = "<div style='color:#888'>no actions logged yet</div>"

    new_slugs_html = ""
    for s in new_today_slugs[:20]:
        cn = ", ".join(s["class_names"][:3]) or "<em>empty</em>"
        new_slugs_html += (f'<div class="slug-row">'
                            f'<span class="slug-name">{s["slug"]}</span>'
                            f'<span class="slug-imgs">{s["local_images"]} imgs</span>'
                            f'<span class="slug-cls">{cn}</span>'
                            f'<span class="slug-ts">{s["downloaded_at"]}</span>'
                            f'</div>')
    if not new_slugs_html:
        new_slugs_html = "<div style='color:#888'>no new slugs in last 24h</div>"

    jobs_html = ""
    for j in agent_jobs:
        tail = "<br>".join(t[:100].replace("<","&lt;") for t in j["tail"])
        jobs_html += (f'<div class="job-row">'
                       f'<span class="job-name">{j["name"]}</span>'
                       f'<span class="job-id">{j["jobid"]}</span>'
                       f'<span class="job-ts">{j["mtime_h"]}</span>'
                       f'<div class="job-tail">{tail}</div>'
                       f'</div>')
    if not jobs_html:
        jobs_html = "<div style='color:#888'>no agent jobs found</div>"

    return HTMLResponse(f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>☀️ Morning Report</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", sans-serif;
         max-width: 1100px; margin: 30px auto; padding: 1rem;
         background: #f2f3f7; color: #1a1a1d; }}
  h1 {{ font-size: 24px; margin: 0 0 8px 0; }}
  .sub {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px; margin-bottom: 20px; }}
  .stat {{ background: #fff; padding: 12px 14px; border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center; }}
  .stat .v {{ font-size: 22px; font-weight: 600; color: #06c; }}
  .stat .l {{ font-size: 11px; color: #666; }}
  .section {{ background: #fff; padding: 14px 18px; border-radius: 8px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 16px; }}
  .section h3 {{ margin: 0 0 12px 0; font-size: 14px; color: #555;
                text-transform: uppercase; letter-spacing: 0.5px; }}
  .hist-row, .slug-row, .job-row {{ display: grid; gap: 8px; padding: 6px 0;
                                     border-bottom: 1px solid #f0f0f0; font-size: 12px; }}
  .hist-row {{ grid-template-columns: 160px 160px 1fr; }}
  .slug-row {{ grid-template-columns: 1fr 80px 1fr 140px; }}
  .job-row {{ grid-template-columns: 140px 90px 90px; align-items: start;
              padding: 8px 0; }}
  .job-tail {{ grid-column: 1 / -1; font-family: ui-monospace, monospace;
               font-size: 10px; color: #555; background: #fafafa;
               padding: 5px 8px; border-radius: 4px; margin-top: 4px; }}
  .ts, .slug-ts {{ font-family: ui-monospace, monospace; color: #888; font-size: 11px; }}
  .act, .slug-name, .job-name {{ font-weight: 600; color: #06c; font-family: ui-monospace, monospace; }}
  .slug-imgs, .job-id {{ text-align: right; font-family: ui-monospace, monospace; color: #555; }}
  .slug-cls {{ color: #666; word-break: break-all; }}
  .nav {{ margin-bottom: 20px; font-size: 13px; }}
  .nav a {{ color: #06c; margin-right: 12px; }}
</style>
</head><body>
<h1>☀️ Morning Report</h1>
<div class="sub">Snapshot of cluster state and overnight progress.</div>
<div class="nav">
  <a href="/">🏠 hub</a>
  <a href="/control">🎛️ control</a>
  <a href="/classes">📋 classes</a>
  <a href="/slugs">📦 slugs</a>
</div>

<div class="stats">
  <div class="stat"><div class="v">{n_slugs}</div><div class="l">total slugs</div></div>
  <div class="stat"><div class="v">{n_dl}</div><div class="l">downloaded</div></div>
  <div class="stat"><div class="v">{n_cn}</div><div class="l">with class_names</div></div>
  <div class="stat"><div class="v">{total_imgs:,}</div><div class="l">total imgs</div></div>
  <div class="stat"><div class="v">{n_ov}</div><div class="l">topic overrides</div></div>
  <div class="stat"><div class="v">{len(new_today_slugs)}</div><div class="l">new today (24h)</div></div>
</div>

<div class="section">
  <h3>🆕 new slugs (last 24h)</h3>
  {new_slugs_html}
</div>

<div class="section">
  <h3>📚 action history (cluster_actions.jsonl)</h3>
  {hist_html}
</div>

<div class="section">
  <h3>🤖 recent agent jobs (last 3 lines of each)</h3>
  {jobs_html}
</div>

<div class="sub">
  Auto-refresh: <a href="/morning_report">🔄 reload</a> ·
  raw JSON: <a href="/api/action_history">history</a> /
  <a href="/api/cluster_status">cluster</a>
</div>
</body></html>''')


@app.get("/control", response_class=HTMLResponse)
def control_page():
    """Operator-facing control panel. Lives at /control. Polls /api/cluster_status
    every 5 s; action buttons POST /api/cluster_action/{name}."""
    return HTMLResponse('''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎛️ Cluster Control Panel</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif;
         max-width: 1200px; margin: 20px auto; padding: 1rem; color: #1a1a1d;
         background: #f2f3f7; }
  h1 { margin: 0 0 6px 0; font-size: 22px; }
  .sub { color: #666; margin-bottom: 22px; font-size: 14px; }
  .nav { font-size: 13px; margin-bottom: 12px; }
  .nav a { color: #06c; margin-right: 12px; }
  .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 10px; margin-bottom: 16px; }
  .stat { background: #fff; padding: 12px 14px; border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .stat .v { font-size: 22px; font-weight: 600; color: #06c; }
  .stat .v.warn { color: #c70; }
  .stat .v.bad { color: #c00; }
  .stat .v.ok { color: #2a7; }
  .stat .l { font-size: 12px; color: #666; }
  .section { background: #fff; padding: 12px 18px; border-radius: 8px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 14px; }
  .section h3 { font-size: 14px; margin: 0 0 10px 0; color: #555;
                text-transform: uppercase; letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #f0f0f0; }
  th { color: #666; font-weight: 600; font-size: 11px; }
  td.mono { font-family: ui-monospace, monospace; font-size: 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
           font-size: 11px; font-weight: 600; }
  .badge.R { background: #d1fae5; color: #065f46; }
  .badge.PD { background: #fef3c7; color: #92400e; }
  .badge.CD, .badge.F { background: #fee2e2; color: #991b1b; }
  .actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
             gap: 10px; }
  .act-btn { background: #fff; padding: 12px 14px; border-radius: 8px;
             border: 1px solid #ddd; cursor: pointer; font-size: 13px;
             text-align: left; transition: all 0.1s; }
  .act-btn:hover { border-color: #06c; background: #f0f8ff; }
  .act-btn .l { font-weight: 600; color: #06c; margin-bottom: 4px; }
  .act-btn .d { font-size: 11px; color: #666; }
  .act-btn.danger:hover { border-color: #c00; background: #fff5f5; }
  .act-btn.danger .l { color: #c00; }
  .log-out { font-family: ui-monospace, monospace; font-size: 11px;
             background: #1a1a1d; color: #d4d4d4; padding: 10px;
             border-radius: 5px; max-height: 200px; overflow-y: auto;
             white-space: pre-wrap; }
  .refresh-info { font-size: 11px; color: #999; margin-top: 6px; }
  .log-btn { background: #06c; color: #fff; border: none;
             padding: 3px 9px; border-radius: 3px; font-size: 11px;
             cursor: pointer; }
  .log-btn:hover { background: #048; }
  .kill-btn { background: #c00; color: #fff; border: none;
              padding: 3px 9px; border-radius: 3px; font-size: 11px;
              cursor: pointer; margin-left: 4px; }
  .kill-btn:hover { background: #900; }
  .history-item { display: grid; grid-template-columns: 140px 160px 1fr;
                   gap: 8px; padding: 5px 0; border-bottom: 1px solid #f0f0f0;
                   font-size: 12px; }
  .history-item:last-child { border-bottom: none; }
  .history-item .ts { font-family: ui-monospace, monospace; color: #888; }
  .history-item .act { font-weight: 600; color: #06c; }
  .history-item .msg { color: #555; word-break: break-all; }
  .modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,0.55);
              display: none; align-items: center; justify-content: center;
              z-index: 100; }
  .modal-bg.show { display: flex; }
  .modal-box { background: #1a1a1d; color: #d4d4d4;
               width: min(95vw, 1100px); height: min(85vh, 700px);
               border-radius: 8px; display: flex; flex-direction: column;
               box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
  .modal-head { padding: 10px 16px; background: #2a2a2d;
                border-radius: 8px 8px 0 0; color: #fff;
                font-family: ui-monospace, monospace; font-size: 12px;
                display: flex; align-items: center; gap: 10px; }
  .modal-head .title { flex: 1; }
  .modal-head .close-btn { background: #c00; color: #fff; border: none;
                            padding: 4px 12px; border-radius: 3px;
                            cursor: pointer; font-size: 12px; }
  .modal-head .auto-poll { color: #4f4; font-size: 11px; }
  .modal-body { flex: 1; overflow-y: auto; padding: 10px 16px;
                font-family: ui-monospace, monospace; font-size: 11px;
                white-space: pre-wrap; word-break: break-all;
                background: #1a1a1d; color: #d4d4d4; }
  .recent-jobs { display: grid;
                 grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                 gap: 6px; }
  .recent-job { background: #fafafa; padding: 7px 10px; border-radius: 5px;
                font-size: 12px; cursor: pointer; transition: all 0.1s;
                border-left: 3px solid #06c; }
  .recent-job:hover { background: #e8f1ff; }
  .recent-job .n { font-family: ui-monospace, monospace; color: #06c; }
  .recent-job .t { color: #888; font-size: 10px; }
</style>
</head><body>
<h1>🎛️ Cluster Control Panel</h1>
<div class="sub">Live monitoring + one-click actions</div>
<div class="nav">
  <a href="/">🏠 hub</a>
  <a href="/classes">📋 classes</a>
  <a href="/slugs">📦 slugs</a>
  <a href="/roboflow">📊 roboflow</a>
  <a href="/api/cluster_status">📥 JSON</a>
</div>

<div class="grid-stats" id="stats"></div>

<div class="section" id="agent-section">
  <h3>🤖 Current agent activity (latest dl_known / brain_harvest / topic_backfill)</h3>
  <div id="agent-progress">
    <span style="color:#999">loading…</span>
  </div>
</div>

<div class="section">
  <h3>🔘 Actions — click to trigger (backend sbatch / cache clear)</h3>
  <div class="actions" id="actions"></div>
  <div class="refresh-info" id="last-action"></div>
</div>

<div class="section">
  <h3>📋 SLURM queue (squeue)</h3>
  <table id="jobs-table">
    <thead><tr><th>jobid</th><th>name</th><th>state</th><th>time</th><th>cpus</th><th>mem</th><th>reason</th><th>log</th></tr></thead>
    <tbody><tr><td colspan="8" style="color:#999;text-align:center">loading…</td></tr></tbody>
  </table>
  <div class="refresh-info" id="last-refresh"></div>
</div>

<div class="section">
  <h3>📁 recent .out files (clickable — view log)</h3>
  <div class="recent-jobs" id="recent-jobs"><div style="color:#999">loading…</div></div>
</div>

<div class="section">
  <h3>📚 action history (overnight + ongoing)</h3>
  <div id="action-history"><div style="color:#999">loading…</div></div>
</div>

<div class="section">
  <h3>📜 latest action output</h3>
  <pre class="log-out" id="action-log">(no action yet)</pre>
</div>

<!-- Log viewer modal -->
<div class="modal-bg" id="log-modal">
  <div class="modal-box">
    <div class="modal-head">
      <div class="title" id="log-modal-title">job log</div>
      <span class="auto-poll" id="log-modal-status">●</span>
      <button class="close-btn" onclick="closeLog()">✗ close (esc)</button>
    </div>
    <div class="modal-body" id="log-modal-body">loading…</div>
  </div>
</div>

<script>
async function poll() {
  try {
    const r = await fetch('/api/cluster_status');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    renderStats(d);
    renderJobs(d.jobs || []);
    document.getElementById('last-refresh').textContent =
      'last poll: ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('last-refresh').textContent =
      'poll error: ' + e.message;
  }
}

async function pollAgentProgress() {
  try {
    const r = await fetch('/api/agent_progress');
    const d = await r.json();
    const el = document.getElementById('agent-progress');
    if (d.no_agent_jobs) {
      el.innerHTML = '<span style="color:#888">no agent jobs yet — click 🤖 brain_harvest or 📥 download_known_slugs</span>';
      return;
    }
    const stateBadge = `<span class="badge ${d.state}">${d.state}</span>`;
    let progressHtml = '';
    if (d.progress) {
      const p = d.progress;
      const pct = Math.round(p.current * 100 / p.total);
      progressHtml = `
        <div style="margin:8px 0">
          <div style="font-weight:600">🌱 currently: ${p.current_item}</div>
          <div style="margin:6px 0">step ${p.current} / ${p.total}
            (${pct}%)</div>
          <div style="height:6px; background:#eee; border-radius:3px; overflow:hidden">
            <div style="height:100%; background:#06c; width:${pct}%"></div>
          </div>
        </div>`;
    }
    const tailHtml = (d.log_tail||[]).slice(-5).map(l =>
      '<div>' + l.replace(/[<>]/g, c => ({'<':'&lt;','>':'&gt;'}[c])) + '</div>'
    ).join('');
    el.innerHTML = `
      <div style="display:flex; gap:14px; align-items:center; font-size:13px">
        <span><strong>${d.job_name}</strong></span>
        <span class="mono">${d.jobid}</span>
        ${stateBadge}
        <span style="color:#888">log: ${(d.log_size/1024).toFixed(1)}KB</span>
        <button class="log-btn" onclick="openLog('${d.jobid}','${d.job_name}')">📜 full log</button>
      </div>
      ${progressHtml}
      <pre style="background:#1a1a1d; color:#d4d4d4; padding:8px 10px;
                  border-radius:5px; font-size:11px; margin:6px 0 0 0;
                  white-space:pre-wrap; max-height:120px; overflow-y:auto">
${tailHtml}
      </pre>`;
  } catch (e) {
    document.getElementById('agent-progress').innerHTML =
      '<span style="color:#c00">agent_progress fetch err: ' + e + '</span>';
  }
}

async function pollDiskUsage() {
  // Lightweight; piggyback on stats render
  try {
    const r = await fetch('/api/disk_usage');
    const d = await r.json();
    return d;
  } catch (e) {
    return null;
  }
}

let _diskUsage = null;
async function refreshDiskUsage() {
  _diskUsage = await pollDiskUsage();
}

function renderStats(d) {
  const reg = d.registry || {};
  const ex = d.exemplars || {};
  const sv = d.slug_verdicts || {counts:{}};
  const ol = d.ollama || {};
  const jobs = d.jobs || [];
  const nR = jobs.filter(j => j.state === 'RUNNING').length;
  const nPD = jobs.filter(j => j.state === 'PENDING').length;
  const du = _diskUsage || {};
  const stats = [
    { l: 'tunnel URL',  v: d.tunnel_url ? '✓ live' : '? no tunnel',
      cls: d.tunnel_url ? 'ok' : 'bad' },
    { l: 'this dashboard job', v: d.my_slurm_job_id || '?',
      cls: d.my_slurm_job_id ? 'ok' : 'bad' },
    { l: 'jobs running',  v: nR, cls: nR > 0 ? 'ok' : '' },
    { l: 'jobs pending',  v: nPD, cls: nPD > 0 ? 'warn' : '' },
    { l: 'ollama',  v: ol.running ? '✓' : '✗',
      cls: ol.running ? 'ok' : 'bad',
      sub: ol.running ? (ol.models||[]).join(',').slice(0,40) : (ol.error||'') },
    { l: 'registry slugs', v: reg.n_slugs || '?',
      sub: 'with class_names: ' + (reg.n_with_classnames||0) },
    { l: 'total images',  v: (reg.total_imgs||0).toLocaleString() },
    { l: 'topic overrides', v: d.n_topic_overrides || 0,
      sub: 'Brain + human classified' },
    { l: '✓ exemplars (all classes)',  v: ex.n_keep || 0,
      cls: (ex.n_keep||0) > 0 ? 'ok' : '' },
    { l: '✗ bad (all classes)',  v: ex.n_bad || 0,
      cls: (ex.n_bad||0) > 0 ? 'bad' : '' },
    { l: '✓ keep slugs', v: sv.counts.keep || 0 },
    { l: '✗ junk slugs', v: sv.counts.junk || 0,
      cls: (sv.counts.junk||0) > 0 ? 'bad' : '' },
    { l: '/ocean used',  v: du.used || '?',
      sub: 'avail: ' + (du.avail||'?') + ' (' + (du.use_pct||'?') + ')' },
  ];
  const html = stats.map(s =>
    `<div class="stat"><div class="v ${s.cls||''}">${s.v}</div>
     <div class="l">${s.l}</div>
     ${s.sub ? `<div class="l" style="font-size:10px;color:#aaa;margin-top:3px">${s.sub}</div>` : ''}
     </div>`
  ).join('');
  document.getElementById('stats').innerHTML = html;
}

function renderJobs(jobs) {
  const tbody = document.querySelector('#jobs-table tbody');
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="color:#999">no jobs</td></tr>';
    return;
  }
  // Whitelist of cancellable job-name prefixes
  const cancellable = ['dl_known','brain_hrv','topic_bf','smoke','lora_'];
  tbody.innerHTML = jobs.map(j => {
    const canKill = cancellable.some(p => (j.name||'').startsWith(p));
    return `<tr>
      <td class="mono">${j.jobid}</td>
      <td class="mono">${j.name}</td>
      <td><span class="badge ${j.state}">${j.state}</span></td>
      <td class="mono">${j.time}</td>
      <td class="mono">${j.cpus||'?'}</td>
      <td class="mono">${j.mem||'?'}</td>
      <td class="mono" style="color:#888">${j.reason}</td>
      <td><button class="log-btn" onclick="openLog('${j.jobid}','${j.name}')">📜 log</button>
        ${canKill ? `<button class="kill-btn" onclick="cancelJob('${j.jobid}','${j.name}')">✗ kill</button>` : ''}
      </td>
    </tr>`;
  }).join('');
}

async function cancelJob(jobid, name) {
  if (!confirm(`Cancel job ${jobid} (${name})? It will lose all progress.`)) return;
  try {
    const r = await fetch('/api/cancel_job/' + jobid, {method:'POST'});
    const data = await r.json();
    alert(JSON.stringify(data, null, 2));
    poll();
  } catch (e) {
    alert('cancel failed: ' + e.message);
  }
}

async function loadActionHistory() {
  try {
    const r = await fetch('/api/action_history?n=30');
    const d = await r.json();
    const events = (d.history || []).reverse(); // newest first
    if (!events.length) {
      document.getElementById('action-history').innerHTML =
        '<div style="color:#999">no actions yet</div>';
      return;
    }
    const stColor = {succeeded:'#16a34a', running:'#2563eb', failed:'#dc2626',
                     launched:'#d97706', unknown:'#9ca3af'};
    const stIcon = {succeeded:'✅', running:'⏳', failed:'❌',
                    launched:'🚀', unknown:'❔'};
    document.getElementById('action-history').innerHTML = events.map(e => {
      const msg = (e.result && (e.result.msg || JSON.stringify(e.result))) || '';
      const t = e.ts_h ? e.ts_h.replace('UTC','') : '?';
      const st = e.status || 'unknown';
      const badge = `<span class="hstatus" title="real outcome (sacct/log), not just launched-ok" `
        + `style="color:${stColor[st]||'#9ca3af'};font-weight:600;min-width:90px;display:inline-block">`
        + `${stIcon[st]||'❔'} ${st}</span>`;
      return `<div class="history-item">
        <span class="ts">${t}</span>
        ${badge}
        <span class="act">${e.action}</span>
        <span class="msg">${msg.toString().replace(/[<>]/g, c => ({'<':'&lt;','>':'&gt;'}[c]))}</span>
      </div>`;
    }).join('');
  } catch (e) {
    document.getElementById('action-history').innerHTML =
      '<div style="color:#c00">history fetch err: ' + e + '</div>';
  }
}

async function loadRecentJobs() {
  try {
    const r = await fetch('/api/recent_jobs?n=15');
    const d = await r.json();
    const jobs = d.jobs || [];
    if (!jobs.length) {
      document.getElementById('recent-jobs').innerHTML =
        '<div style="color:#999">no recent jobs</div>';
      return;
    }
    const html = jobs.map(j =>
      `<div class="recent-job" onclick="openLog('${j.jobid}','${j.name}')">
         <div class="n">${j.jobid} · ${j.name}</div>
         <div class="t">${j.mtime_h} · ${(j.size/1024).toFixed(1)}KB</div>
       </div>`
    ).join('');
    document.getElementById('recent-jobs').innerHTML = html;
  } catch (e) {
    document.getElementById('recent-jobs').innerHTML =
      '<div style="color:#c00">recent jobs load fail: ' + e + '</div>';
  }
}

// ============ Log viewer modal ============
let _logPollTimer = null;
let _logCurrentJob = null;

async function openLog(jobid, name) {
  _logCurrentJob = {jobid, name};
  document.getElementById('log-modal-title').textContent = `${jobid} · ${name||'?'}`;
  document.getElementById('log-modal-body').textContent = 'loading…';
  document.getElementById('log-modal').classList.add('show');
  await refreshLog();
  if (_logPollTimer) clearInterval(_logPollTimer);
  _logPollTimer = setInterval(refreshLog, 3000);
}

async function refreshLog() {
  if (!_logCurrentJob) return;
  const {jobid} = _logCurrentJob;
  const status = document.getElementById('log-modal-status');
  const body = document.getElementById('log-modal-body');
  status.style.color = '#ff4';
  status.textContent = '⟳ fetching…';
  try {
    const r = await fetch(`/api/job_log/${jobid}?tail=500`);
    if (!r.ok) {
      const err = await r.json().catch(()=>({msg:r.statusText}));
      body.textContent = `HTTP ${r.status}: ${err.msg || err.detail || ''}`;
      status.style.color = '#f44';
      status.textContent = '✗ error';
      return;
    }
    const d = await r.json();
    // Set scroll to bottom after content updates (tail-follow)
    const wasNearBottom = (body.scrollTop + body.clientHeight + 50)
                          >= body.scrollHeight;
    const header = `# file: ${d.file}\n# size: ${(d.file_size_bytes/1024).toFixed(1)}KB  `
                   + `lines: ${d.lines_returned}/${d.tail_requested}  `
                   + `mtime: ${new Date(d.mtime*1000).toLocaleTimeString()}\n`
                   + `# (auto-refresh 3s — close to stop)\n${'─'.repeat(80)}\n`;
    body.textContent = header + d.content;
    if (wasNearBottom) body.scrollTop = body.scrollHeight;
    status.style.color = '#4f4';
    status.textContent = `● live · ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    body.textContent = 'fetch error: ' + e;
    status.style.color = '#f44';
    status.textContent = '✗ error';
  }
}

function closeLog() {
  document.getElementById('log-modal').classList.remove('show');
  if (_logPollTimer) {
    clearInterval(_logPollTimer);
    _logPollTimer = null;
  }
  _logCurrentJob = null;
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLog();
});

// Close modal when clicking outside the box
document.getElementById('log-modal').addEventListener('click', e => {
  if (e.target.id === 'log-modal') closeLog();
});

async function loadActions() {
  try {
    const r = await fetch('/api/cluster_actions');
    const acts = await r.json();
    const html = Object.entries(acts).map(([k, v]) => {
      const danger = k.includes('restart') ? ' danger' : '';
      return `<div class="act-btn${danger}" onclick="trigger('${k}')">
                <div class="l">${k}</div>
                <div class="d">${v.label}</div>
              </div>`;
    }).join('');
    document.getElementById('actions').innerHTML = html;
  } catch (e) {
    document.getElementById('actions').innerHTML =
      '<div style="color:#c00">failed to load actions: ' + e + '</div>';
  }
}

async function trigger(action) {
  if (action === 'restart_dashboard') {
    if (!confirm('Restart dashboard? This page will reconnect in ~90s (refresh github.io for the new URL)')) return;
  }
  const log = document.getElementById('action-log');
  const li = document.getElementById('last-action');
  log.textContent = `→ triggering ${action}…`;
  try {
    const r = await fetch('/api/cluster_action/' + action, {method:'POST'});
    const data = await r.json();
    log.textContent = JSON.stringify(data, null, 2);
    li.textContent = `last action: ${action} @ ${new Date().toLocaleTimeString()}`;
    if (action !== 'restart_dashboard') poll();
  } catch (e) {
    log.textContent = 'error: ' + e.message;
  }
}

loadActions();
loadRecentJobs();
loadActionHistory();
refreshDiskUsage();
pollAgentProgress();
poll();
setInterval(poll, 5000);
setInterval(loadRecentJobs, 30000);
setInterval(loadActionHistory, 15000);
setInterval(refreshDiskUsage, 60000);
setInterval(pollAgentProgress, 8000);
</script>
</body></html>''')


@app.get("/slugs", response_class=HTMLResponse)
def slugs_landing(domain: str = "weed"):
    """Slug-level cleanup: ✓ keep / ✗ junk / 🤔 unsure on whole slugs.
    Faster than per-image audit when a slug is obviously garbage (plant
    disease imported by accident, etc.). v3.0.109: ?domain= scopes to one agent."""
    if not re.match(r"^[a-z0-9_]{1,40}$", domain or ""):
        domain = "weed"
    # v3.0.83 Phase 5: read via Mongo; db.get_registry falls back to
    # dataset_registry.json automatically when Mongo is down.
    from . import db as _db
    reg = _db.get_registry(domain=domain)
    datasets = reg.get("datasets", {})
    if not datasets and not REGISTRY_PATH.exists():
        return HTMLResponse("<h1>no registry</h1>", status_code=404)
    verdicts = _slug_verdict_state()

    # Build rows. Sort: junk last, then unsure, then keep, then unverified.
    rows = []
    for slug, info in datasets.items():
        cn = info.get("class_names") or []
        lp = info.get("local_path") or ""
        st = info.get("status", "?")
        n_imgs = info.get("local_images") or info.get("images") or 0
        used = info.get("used_for_training", False)
        verdict = verdicts.get(slug, "")
        rows.append({
            "slug": slug, "cn": cn, "lp": lp, "status": st,
            "n_imgs": n_imgs, "used": used, "verdict": verdict,
            "has_local": bool(lp and os.path.isdir(lp)),
        })
    # Sort by verdict (unverified first, junk last)
    def sort_key(r):
        order = {"": 0, "keep": 1, "unsure": 2, "junk": 3}
        return (order.get(r["verdict"], 0), -r["n_imgs"])
    rows.sort(key=sort_key)

    # Render table
    tr_html = []
    for r in rows:
        cn_str = ", ".join(r["cn"][:4]) + (f" (+{len(r['cn'])-4})" if len(r["cn"]) > 4 else "")
        if not r["cn"]:
            cn_str = '<span style="color:#c00">empty</span>'
        used_badge = '<span class="badge badge-used">TRAINED</span>' if r["used"] else ""
        local_badge = '' if r["has_local"] else '<span class="badge badge-no-local">no local</span>'
        v_class = f" verdict-{r['verdict']}" if r["verdict"] else ""
        v_buttons = "".join(
            f'<button class="vbtn {k}{"" if r["verdict"]!=k else " on"}" '
            f'data-v="{k}" title="{lbl}">{sym}</button>'
            for k, sym, lbl in [
                ("keep", "✓", "Keep (good slug)"),
                ("junk", "✗", "Delete (garbage slug)"),
                ("unsure", "🤔", "Unsure"),
            ]
        )
        tr_html.append(f'''
        <tr class="srow{v_class}" data-slug="{r["slug"]}" data-verdict="{r["verdict"]}">
          <td class="slug-col">
            <a href="/gallery/{r["slug"]}" target="_blank" title="View this dataset's images (to judge keep / junk)">{r["slug"]}</a>
            <a href="/gallery/{r["slug"]}" target="_blank" title="View images" style="text-decoration:none;margin-left:6px">🖼</a>
            {used_badge} {local_badge}
          </td>
          <td>{r["status"]}</td>
          <td class="num">{r["n_imgs"]}</td>
          <td class="num">{len(r["cn"])}</td>
          <td class="cn-col">{cn_str}</td>
          <td class="verdict-col">{v_buttons}</td>
        </tr>''')

    n_keep = sum(1 for v in verdicts.values() if v == "keep")
    n_junk = sum(1 for v in verdicts.values() if v == "junk")
    n_unsure = sum(1 for v in verdicts.values() if v == "unsure")
    n_unverified = len(datasets) - n_keep - n_junk - n_unsure

    html = f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Slugs — registry cleanup</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", sans-serif;
         margin: 0; padding: 16px; background: #f2f3f7; color: #1a1a1d; }}
  header {{ background: #fff; padding: 14px 22px; border-radius: 10px;
           margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  header h1 {{ margin: 0 0 4px 0; font-size: 20px; }}
  header .sub {{ color: #666; font-size: 13px; }}
  header a {{ color: #06c; text-decoration: none; }}
  .summary {{ background: #fff; padding: 10px 16px; border-radius: 8px;
              margin-bottom: 12px; display: flex; gap: 18px; flex-wrap: wrap;
              font-size: 14px; }}
  .filter-bar {{ background: #fff; padding: 10px 16px; border-radius: 8px;
                margin-bottom: 12px; display: flex; gap: 6px; flex-wrap: wrap; }}
  .filter-bar button {{ background: #f4f4f7; border: 1px solid #ddd;
                       padding: 5px 12px; border-radius: 16px; cursor: pointer;
                       font-size: 13px; }}
  .filter-bar button.on {{ background: #06c; color: #fff; border-color: #06c; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
          border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); font-size: 13px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
  th {{ background: #fafafa; font-weight: 600; font-size: 12px; color: #555; }}
  td.num {{ text-align: right; font-family: ui-monospace, monospace; }}
  td.slug-col {{ font-family: ui-monospace, monospace; word-break: break-all;
                 max-width: 360px; }}
  td.cn-col {{ color: #666; font-size: 12px; max-width: 320px; word-break: break-all; }}
  td.verdict-col {{ white-space: nowrap; }}
  .vbtn {{ background: #fff; border: 1px solid #ccc; padding: 4px 10px;
          margin: 0 2px; border-radius: 4px; cursor: pointer; font-size: 14px; }}
  .vbtn.keep.on   {{ background: #38a169; color: #fff; border-color: #38a169; }}
  .vbtn.junk.on   {{ background: #e53e3e; color: #fff; border-color: #e53e3e; }}
  .vbtn.unsure.on {{ background: #f59e0b; color: #fff; border-color: #f59e0b; }}
  .srow.verdict-junk td {{ opacity: 0.55; text-decoration: line-through; }}
  .srow.verdict-keep {{ background: #f0fff4; }}
  .srow.verdict-unsure {{ background: #fffbeb; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
           font-size: 10px; font-weight: 600; margin-left: 4px;
           vertical-align: middle; }}
  .badge-used {{ background: #dbeafe; color: #1e40af; }}
  .badge-no-local {{ background: #fee2e2; color: #991b1b; }}
</style>
</head><body>
<div style="margin:0 0 12px"><a href="/agent/{domain}" style="display:inline-block;text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Mission Control</a></div>
<header>
  <h1>📦 Slugs — registry-level cleanup</h1>
  <div class="sub">
    One slug per row — ✓ Keep / ✗ Delete (garbage) / 🤔 Unsure.
    Slugs marked ✗ are hidden from /classes by default.
    · <a href="/classes">/classes class-level audit</a>
    · <a href="/">dashboard home</a>
    · <a href="/api/slug_verdicts">📥 JSON</a>
  </div>
</header>
<div class="summary">
  <div>📊 Total <strong>{len(datasets)}</strong> slugs</div>
  <div>✓ keep: <strong>{n_keep}</strong></div>
  <div>🤔 unsure: <strong>{n_unsure}</strong></div>
  <div>✗ junk: <strong>{n_junk}</strong></div>
  <div>Unreviewed: <strong>{n_unverified}</strong></div>
</div>
<div class="filter-bar" id="filter-bar">
  <button class="on" data-f="all">All {len(datasets)}</button>
  <button data-f="unverified">Unreviewed {n_unverified}</button>
  <button data-f="keep">✓ keep {n_keep}</button>
  <button data-f="unsure">🤔 unsure {n_unsure}</button>
  <button data-f="junk">✗ junk {n_junk}</button>
  <button data-f="has_classnames">has class_names</button>
  <button data-f="empty_classnames">no class_names</button>
</div>
<table>
<thead>
<tr><th>Slug</th><th>Status</th><th class="num"># imgs</th><th class="num"># classes</th><th>class_names preview</th><th>verdict</th></tr>
</thead>
<tbody id="slugs-tbody">
{''.join(tr_html)}
</tbody>
</table>
<script>
async function setSlugVerdict(row, verdict) {{
  const slug = row.dataset.slug;
  row.classList.remove('verdict-keep','verdict-junk','verdict-unsure');
  if (verdict !== 'clear') row.classList.add('verdict-' + verdict);
  row.dataset.verdict = verdict === 'clear' ? '' : verdict;
  row.querySelectorAll('.vbtn').forEach(b => b.classList.remove('on'));
  if (verdict !== 'clear') {{
    const b = row.querySelector(`.vbtn.${{verdict}}`);
    if (b) b.classList.add('on');
  }}
  try {{
    await fetch('/api/slug_verdict/' + encodeURIComponent(slug), {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{verdict}}),
    }});
  }} catch (e) {{
    console.error('slug verdict save failed', e);
  }}
}}
document.querySelectorAll('.vbtn').forEach(b => {{
  b.addEventListener('click', () => {{
    const row = b.closest('.srow');
    const v = b.dataset.v;
    const isOn = b.classList.contains('on');
    setSlugVerdict(row, isOn ? 'clear' : v);
  }});
}});
// Filter buttons
document.querySelectorAll('.filter-bar button').forEach(b => {{
  b.addEventListener('click', () => {{
    document.querySelectorAll('.filter-bar button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    const f = b.dataset.f;
    document.querySelectorAll('.srow').forEach(r => {{
      let show = false;
      if (f === 'all') show = true;
      else if (f === 'unverified') show = !r.dataset.verdict;
      else if (f === 'has_classnames') {{
        const cnCol = r.querySelector('.cn-col');
        show = cnCol && !cnCol.innerHTML.includes('empty');
      }}
      else if (f === 'empty_classnames') {{
        const cnCol = r.querySelector('.cn-col');
        show = cnCol && cnCol.innerHTML.includes('empty');
      }}
      else show = r.dataset.verdict === f;
      r.style.display = show ? '' : 'none';
    }});
  }});
}});
</script>
</body></html>'''
    return HTMLResponse(html)


@app.get("/classes", response_class=HTMLResponse)
def classes_landing(domain: str = "weed"):
    # v3.0.109: ?domain= scopes /classes to one collection agent (default weed).
    if not re.match(r"^[a-z0-9_]{1,40}$", domain or ""):
        domain = "weed"
    rows = []
    topic_counts: dict = {"all": 0, "cwd12": 0, "weed": 0, "disease": 0,
                          "pest": 0, "crop": 0, "other": 0}
    for cls in _all_known_classes(domain):
        summary = _class_summary_landing(cls, domain)
        state = _exemplar_state(cls)
        n_bank = summary["n_bank"]
        n_flux = summary["n_flux"]
        n_reg_slugs = summary["n_reg_slugs"]
        n_reg_est = summary["n_reg_est"]
        n_total_est = n_bank + n_flux + n_reg_est
        n_ex = sum(1 for v in state.values() if v == "exemplar")
        n_bad = sum(1 for v in state.values() if v == "bad")
        # v3.0.118: n_reg_est above is a cap-based OVER-estimate (200×slugs), so
        # "463/600 reviewed" looked incomplete even after marking everything. For
        # REVIEWED classes (few), use the ACTUAL pool size as the denominator so
        # the fraction is truthful; unreviewed classes keep the cheap estimate.
        if (n_ex + n_bad) > 0:
            try:
                actual = len(_class_image_pool(cls))
                if actual >= (n_ex + n_bad):
                    n_total_est = actual
            except Exception:
                pass
        zh = _CWD12_ZH.get(cls, "")
        # v3.0.43.2: prefer inline data URI (works even with network blocking
        # subrequests); fall back to /thumb/ URL.
        thumb_inline = summary.get("first_thumb_data", "")
        thumb_url = summary["first_thumb"]
        thumb_src = thumb_inline if thumb_inline else thumb_url
        # Whether the topic came from override (Brain/user) or keyword heuristic
        is_override = cls in _load_topic_overrides()
        topic = _class_topic(cls)
        is_new_today = summary.get("is_new_today", False)
        topic_counts["all"] += 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        rows.append((cls, zh, n_total_est, n_bank, n_flux,
                     n_reg_slugs, n_reg_est, n_ex, n_bad, thumb_src, topic, is_override, is_new_today))

    # Sort: new-today first, then CWD12, then by name
    def sort_key(r):
        cls = r[0]
        is_new = r[12]
        is_cwd12 = cls in _CWD12
        return (0 if is_new else (1 if is_cwd12 else 2), cls.lower())
    rows.sort(key=sort_key)

    cards = "".join(
        f'''
        <a class="class-card{' new-today' if is_new_today else ''}" href="/classes/{cls}" data-topic="{topic}" data-name="{cls.lower()}" data-cls="{cls}" data-new="{int(is_new_today)}">
          {(f'<img src="{thumb}" loading="lazy" alt="{cls}"/>' if thumb
            else f'<div class="no-thumb">no image</div>')}
          <div class="name">{cls}
            {('<span class="new-badge">🆕 NEW</span>' if is_new_today else '')}
            <span class="topic-tag tag-{topic}" title="{'manual override' if is_override else 'keyword heuristic'}">{topic}{'★' if is_override else ''}</span>
            {(f'<span class="badge exemplar">✓ {n_ex}</span>' if n_ex else '')}
            {(f'<span class="badge bad">✗ {n_bad}</span>' if n_bad else '')}
          </div>
          <div class="counts">bank {n_bank} · flux {n_flux} · real ≤{n_reg_est} ({n_reg_slugs} slugs)</div>
          <div style="height:5px;background:#e5e7eb;border-radius:3px;overflow:hidden;margin:5px 0">
            <div style="height:100%;width:{min(100, round(100*(n_ex+n_bad)/max(1,n_total_est)))}%;background:{'#16a34a' if (n_ex+n_bad) else '#cbd5e1'}"></div>
          </div>
          <div class="counts">{n_ex+n_bad} / {n_total_est} reviewed{(' ✓' if n_total_est and (n_ex+n_bad)>=n_total_est else '')}</div>
        </a>'''
        for cls, zh, n_total_est, n_bank, n_flux, n_reg_slugs, n_reg_est, n_ex, n_bad, thumb, topic, is_override, is_new_today in rows
    )

    # Build filter bar + search input
    tab_def = [
        ("cwd12", "🌟 CWD12", "primary 12 species (real bbox in sp8+holdout)"),
        ("weed",  "🌱 Other weeds", "Lantana, Parthenium, weeds/grasses, etc."),
        ("disease", "🦠 Disease", "leaf disease classes from PlantVillage etc."),
        ("pest", "🐛 Pest", "insect/pest species"),
        ("crop", "🌾 Crop", "crop / produce classes"),
        ("other", "❓ Other", "everything else"),
        ("all",  "📋 All", "no filter"),
    ]
    tab_html = "".join(
        f'<button class="filter-tab{(" on" if k=="all" else "")}" '
        f'data-topic="{k}" title="{desc}">{label} '
        f'<span class="tab-count">{topic_counts.get(k,0)}</span></button>'
        for k, label, desc in tab_def
    )

    # ----- metadata-gap surface: slugs with local data but no class_names -----
    empty_slugs = _registry_empty_slugs(domain)
    banner = ""
    if empty_slugs:
        sample = ", ".join(empty_slugs[:6])
        more = f" (+{len(empty_slugs)-6} more)" if len(empty_slugs) > 6 else ""
        banner = f'''
  <div class="help" style="background:#fee;border-left-color:#c00;">
    ⚠️ <strong>Registry metadata gap</strong>: {len(empty_slugs)} downloaded slugs have <strong>empty class_names</strong>,
    their images will <strong>not</strong> appear in any class below. Run the backfill tool first to fill metadata.<br>
    e.g. <code>{sample}</code>{more}
  </div>'''

    html = f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classes — human verification</title>
<style>{_AUDIT_CSS}
  .help {{ background: #fffbe6; border-left: 3px solid #f59e0b;
          padding: 8px 12px; border-radius: 5px; font-size: 13px;
          margin: 10px 0; }}
  .filter-bar {{ background: #fff; padding: 12px 16px; border-radius: 10px;
                margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
  .filter-tab {{ background: #f4f4f7; border: 1px solid #ddd;
                padding: 6px 14px; border-radius: 20px; cursor: pointer;
                font-size: 13px; transition: all 0.1s; }}
  .filter-tab:hover {{ background: #e8e8ec; }}
  .filter-tab.on {{ background: #06c; color: #fff; border-color: #06c; }}
  .filter-tab .tab-count {{ background: rgba(0,0,0,0.1); padding: 1px 7px;
                            border-radius: 10px; margin-left: 4px;
                            font-size: 11px; font-family: ui-monospace, monospace; }}
  .filter-tab.on .tab-count {{ background: rgba(255,255,255,0.25); }}
  .filter-search {{ flex: 1; min-width: 160px; max-width: 280px;
                    padding: 7px 12px; border: 1px solid #ddd; border-radius: 16px;
                    font-size: 13px; outline: none; margin-left: auto; }}
  .filter-search:focus {{ border-color: #06c; }}
  .filter-empty-note {{ width: 100%; color: #888; font-size: 12px;
                        padding: 30px; text-align: center; display: none; }}
  .topic-tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
               font-size: 10px; font-weight: 600; margin-left: 4px;
               vertical-align: middle; opacity: 0.8; }}
  .tag-cwd12   {{ background: #fde68a; color: #92400e; }}
  .tag-weed    {{ background: #bbf7d0; color: #065f46; }}
  .tag-disease {{ background: #fecaca; color: #991b1b; }}
  .tag-pest    {{ background: #ddd6fe; color: #5b21b6; }}
  .tag-crop    {{ background: #c7d2fe; color: #3730a3; }}
  .tag-other   {{ background: #e5e7eb; color: #4b5563; }}
  .no-thumb {{ width: 100%; height: 180px; background: #f0f0f0; color: #aaa;
              display: flex; align-items: center; justify-content: center;
              font-size: 12px; }}
  .class-card.new-today {{ box-shadow: 0 0 0 3px #2a7;
                            animation: pulse-new 2s infinite; }}
  @keyframes pulse-new {{
    0%, 100% {{ box-shadow: 0 0 0 3px #2a7; }}
    50% {{ box-shadow: 0 0 0 3px #4c9; }}
  }}
  .new-badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
                background: #2a7; color: #fff; font-size: 10px;
                font-weight: 700; margin-left: 4px; }}
</style>
</head><body>
<div style="margin:0 0 12px"><a href="/agent/{domain}" style="display:inline-block;text-decoration:none;background:#eef2ff;color:#2563eb;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Mission Control</a></div>
<header>
  <h1>📋 Classes — human-in-the-loop class-level data audit</h1>
  <div class="sub">
    Click any class to review image-by-image: ✓ exemplar / ✗ mislabeled.
    Approved images join that class's "exemplar set" as a trusted source for LoRA / curator / training.
    · <a href="/">dashboard home</a>
    · <a href="/api/exemplars_export">📥 Export exemplars</a>
    · <a href="/slugs">📦 slug-level cleanup</a>
    · <a href="javascript:void(0)" onclick="refreshRegistry()">♻️ Refresh registry</a>
  </div>
</header>
{banner}
<div class="filter-bar" id="filter-bar">
  {tab_html}
  <input class="filter-search" id="filter-search" type="search"
         placeholder="🔎 Search class name (e.g. Goose, weed, tomato)" autocomplete="off"/>
</div>
<section>
  <div class="grid-classes" id="grid-classes">{cards}</div>
  <div class="filter-empty-note" id="filter-empty">No matching classes.</div>
</section>
<script>
async function refreshRegistry() {{
  const ok = confirm('Wipe cached registry index + class pool? Page will reload.');
  if (!ok) return;
  try {{
    const r = await fetch('/api/refresh_registry', {{method: 'POST'}});
    const data = await r.json();
    alert('Refresh OK. Removed ' + data.pool_cache_files_removed + ' cache files. Reloading.');
    location.reload();
  }} catch (e) {{
    alert('Refresh failed: ' + e);
  }}
}}

const tabs = document.querySelectorAll('.filter-tab');
const search = document.getElementById('filter-search');
const cards = document.querySelectorAll('.class-card');
const empty = document.getElementById('filter-empty');
let currentTopic = 'all';
let currentQuery = '';

function applyFilter() {{
  let nShown = 0;
  cards.forEach(c => {{
    const topic = c.dataset.topic;
    const name = c.dataset.name || '';
    const topicOk = (currentTopic === 'all') || (topic === currentTopic);
    const queryOk = !currentQuery || name.includes(currentQuery);
    const ok = topicOk && queryOk;
    c.style.display = ok ? '' : 'none';
    if (ok) nShown++;
  }});
  empty.style.display = nShown === 0 ? 'block' : 'none';
}}

tabs.forEach(t => t.addEventListener('click', () => {{
  tabs.forEach(x => x.classList.remove('on'));
  t.classList.add('on');
  currentTopic = t.dataset.topic;
  applyFilter();
}}));
search.addEventListener('input', e => {{
  currentQuery = e.target.value.toLowerCase().trim();
  applyFilter();
}});

// Preserve filter state via URL hash so refresh keeps view
function hashState() {{
  const params = new URLSearchParams();
  if (currentTopic !== 'all') params.set('t', currentTopic);
  if (currentQuery) params.set('q', currentQuery);
  history.replaceState(null, '', '#' + params.toString());
}}
const obs = new MutationObserver(hashState);
[document].forEach(d => d.addEventListener('input', hashState));
tabs.forEach(t => t.addEventListener('click', hashState));

// Restore state from URL
const initParams = new URLSearchParams(window.location.hash.slice(1));
if (initParams.get('t')) {{
  const tab = document.querySelector(`.filter-tab[data-topic="${{initParams.get('t')}}"]`);
  if (tab) tab.click();
}}
if (initParams.get('q')) {{
  search.value = initParams.get('q');
  search.dispatchEvent(new Event('input'));
}}
</script>
</body></html>'''
    return HTMLResponse(html)


@app.get("/classes/{cls}", response_class=HTMLResponse)
def classes_detail(cls: str):
    if not _cls_ok(cls):
        raise HTTPException(400)
    pool = _class_image_pool(cls)
    if not pool and cls not in _CWD12:
        raise HTTPException(404, f"unknown class {cls!r}")
    state = _exemplar_state(cls)
    # v3.0.111: removed the all-classes sidebar (cluttered + it ran
    # _class_summary_landing for every class on every detail load = slow).
    cards = []
    n_ex = n_bad = n_rb = n_un = 0
    src_breakdown: dict = {}
    # v3.0.118: cap how many cards we RENDER (big classes had 400-600 thumbnails
    # → laggy page). Counts/badges below still reflect the FULL pool; the
    # "Mark ALL in class" button labels everything server-side without needing
    # every card in the DOM. Unreviewed cards render first so review is useful.
    RENDER_CAP = 180
    pool_sorted = sorted(pool, key=lambda e: 0 if not state.get(
        _pool_entry_urls(e, cls)[0], "") else 1)
    for entry in pool_sorted:
        key, thumb_url, full_url, src_tag = _pool_entry_urls(entry, cls)
        kind = entry["kind"]
        fn = entry["fname"]
        src_breakdown[src_tag] = src_breakdown.get(src_tag, 0) + 1
        verdict = state.get(key, "")
        if verdict == "exemplar":  n_ex += 1
        elif verdict == "bad":     n_bad += 1
        elif verdict == "rebox":   n_rb += 1
        else:                      n_un += 1
        if len(cards) >= RENDER_CAP:
            continue
        klass = " " + verdict if verdict else " unverified"
        cards.append(f'''
        <div class="card{klass}" data-img="{key}" data-kind="{kind}" data-src="{src_tag}">
          <span class="src-tag">{src_tag}</span>
          <a href="{full_url}" target="_blank">
            <img src="{thumb_url}" loading="lazy" alt="{fn}"/>
          </a>
          <div class="meta-row">{fn}</div>
          <div class="verdict-row">
            <button class="ok{(' on' if verdict=='exemplar' else '')}"
              title="Exemplar (1)" data-v="exemplar">✓</button>
            <button class="ng{(' on' if verdict=='bad' else '')}"
              title="Mislabeled (2)" data-v="bad">✗</button>
            <button class="rb{(' on' if verdict=='rebox' else '')}"
              title="bbox off (3)" data-v="rebox">🔄</button>
          </div>
        </div>''')
    src_summary = " · ".join(f"{k} {v}" for k, v in sorted(src_breakdown.items()))
    n_shown = len(cards)
    cap_note = ("" if n_shown >= len(pool) else
                f'<div class="help" style="background:#fef9c3;border-color:#fde047">'
                f'Showing the first <strong>{n_shown}</strong> of {len(pool)} candidates '
                f'(unreviewed first) to keep the page fast. Use '
                f'<strong>“Mark ALL {len(pool)} in class”</strong> above to label every '
                f'image at once — it works on the whole class, not just what is shown.</div>')

    html = f'''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cls} — verify</title>
<style>{_CLASSES_CSS}
 .backbar{{margin:0 0 10px}}
 .backbtn{{display:inline-block;text-decoration:none;background:#eef2ff;color:#2563eb;
   font-weight:600;font-size:13px;padding:8px 14px;border-radius:8px}}
 .bulk{{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:0 0 12px}}
 .bulk button{{font-size:13px;font-weight:600;padding:8px 13px;border:0;border-radius:8px;cursor:pointer}}
 .bulk .bulk-ex{{background:#16a34a;color:#fff}} .bulk .bulk-bad{{background:#dc2626;color:#fff}}
 .bulk .bulk-clear{{background:#e5e7eb;color:#374151}}
 .bulk .bulk-msg{{font-size:13px;color:#475569}}
</style>
</head><body>
<header>
  <div class="backbar"><a class="backbtn" href="/classes">&larr; Back to Browse Data</a></div>
  <h1>📋 Class audit · {cls}</h1>
  <div class="sub">
    <strong>{len(pool)}</strong> candidates ({src_summary}) ·
    <span class="badge exemplar">✓ exemplar {n_ex}</span>
    <span class="badge bad">✗ mislabeled {n_bad}</span>
    <span class="badge rebox">🔄 bbox to fix {n_rb}</span>
    · unreviewed {n_un}
  </div>
</header>
<div class="help">
  Shortcuts: hover a thumbnail and press <kbd>1</kbd>=exemplar ✓, <kbd>2</kbd>=mislabeled ✗, <kbd>3</kbd>=bbox to fix 🔄, <kbd>0</kbd>=clear.
  Click a thumbnail to open the full image in a new window. Filter by status below.
</div>
<main>
    {cap_note}
    <div class="bulk">
      <button class="bulk-ex" id="markall-ex" style="font-size:14px">✓ Mark ALL {len(pool)} in class as exemplar</button>
      <button class="bulk-bad" id="markall-bad" style="font-size:14px">✗ Mark ALL {len(pool)} mislabeled</button>
      <span class="bulk-msg" id="bulk-msg"></span>
    </div>
    <div class="bulk" style="margin-top:-4px">
      <span class="bulk-msg">Or only the cards matching the active filter:</span>
      <button class="bulk-ex" id="bulk-ex">✓ shown → exemplar</button>
      <button class="bulk-bad" id="bulk-bad">✗ shown → mislabeled</button>
      <button class="bulk-clear" id="bulk-clear">Clear shown</button>
    </div>
    <div class="filters">
      <button class="on" data-f="all">All {len(pool)}</button>
      <button data-f="unverified">Unreviewed {n_un}</button>
      <button data-f="exemplar">Exemplar ✓ {n_ex}</button>
      <button data-f="bad">Mislabeled ✗ {n_bad}</button>
      <button data-f="rebox">bbox fix 🔄 {n_rb}</button>
    </div>
    <div class="grid" id="grid">{''.join(cards)}</div>
</main>
<script>
const CLS = {_json.dumps(cls)};

async function setVerdict(card, verdict) {{
  const img = card.dataset.img;
  // optimistic UI
  for (const c of ['exemplar','bad','rebox','unverified']) card.classList.remove(c);
  card.classList.add(verdict === 'clear' ? 'unverified' : verdict);
  card.querySelectorAll('.verdict-row button').forEach(b => b.classList.remove('on'));
  if (verdict !== 'clear') {{
    const target = card.querySelector(`.verdict-row button[data-v="${{verdict}}"]`);
    if (target) target.classList.add('on');
  }}
  try {{
    await fetch(`/api/exemplar/${{CLS}}`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{img, verdict}}),
    }});
  }} catch (e) {{
    console.error('save failed', e);
    card.style.outlineColor = 'magenta';
  }}
}}

document.querySelectorAll('.card .verdict-row button').forEach(btn => {{
  btn.addEventListener('click', e => {{
    e.preventDefault();
    const card = btn.closest('.card');
    const v = btn.dataset.v;
    const isOn = btn.classList.contains('on');
    setVerdict(card, isOn ? 'clear' : v);
  }});
}});

document.querySelectorAll('.filters button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    const f = btn.dataset.f;
    document.querySelectorAll('#grid .card').forEach(card => {{
      let show = false;
      if (f === 'all') show = true;
      else if (f === 'unverified') show = card.classList.contains('unverified');
      else show = card.classList.contains(f);
      card.style.display = show ? '' : 'none';
    }});
  }});
}});

// v3.0.111: bulk select-all → one verdict for every SHOWN card (respects the
// active filter, e.g. filter to Unreviewed then mark all as exemplar).
function visibleCards() {{
  return Array.from(document.querySelectorAll('#grid .card'))
    .filter(c => c.style.display !== 'none');
}}
async function bulkMark(verdict) {{
  const cards = visibleCards();
  if (!cards.length) return;
  const label = verdict==='exemplar'?'exemplar':(verdict==='bad'?'mislabeled':'cleared');
  if (!confirm('Mark ' + cards.length + ' shown image(s) as ' + label + '?')) return;
  const msg = document.getElementById('bulk-msg');
  msg.textContent = '⏳ Saving ' + cards.length + '…';
  const imgs = cards.map(c => c.dataset.img);
  try {{
    const r = await fetch(`/api/exemplar_bulk/${{CLS}}`, {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{verdict, imgs}}),
    }});
    const d = await r.json();
    if (d.ok) {{
      cards.forEach(card => {{
        for (const k of ['exemplar','bad','rebox','unverified']) card.classList.remove(k);
        card.classList.add(verdict==='clear'?'unverified':verdict);
        card.querySelectorAll('.verdict-row button').forEach(b=>b.classList.remove('on'));
        if (verdict!=='clear') {{
          const t = card.querySelector(`.verdict-row button[data-v="${{verdict}}"]`);
          if (t) t.classList.add('on');
        }}
      }});
      msg.textContent = '✅ ' + d.count + ' marked ' + label + ' — refresh to update counts';
    }} else {{ msg.textContent = '❌ ' + (d.detail || 'failed'); }}
  }} catch(e) {{ msg.textContent = '❌ ' + e; }}
}}
document.getElementById('bulk-ex').addEventListener('click', () => bulkMark('exemplar'));
document.getElementById('bulk-bad').addEventListener('click', () => bulkMark('bad'));
document.getElementById('bulk-clear').addEventListener('click', () => bulkMark('clear'));

// v3.0.118: mark the ENTIRE class server-side (no dependence on rendered cards)
// — this is the fast path for a clean class and always reaches 100%.
async function markAll(verdict) {{
  const label = verdict==='exemplar'?'exemplar':'mislabeled';
  if (!confirm('Mark the ENTIRE class ({len(pool)} images) as ' + label + '? This labels every candidate, not just what is shown.')) return;
  const msg = document.getElementById('bulk-msg');
  msg.textContent = '⏳ Marking the whole class…';
  try {{
    const r = await fetch(`/api/exemplar_markall/${{CLS}}`, {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{verdict}}),
    }});
    const d = await r.json();
    if (d.ok) {{
      document.querySelectorAll('#grid .card').forEach(card => {{
        for (const k of ['exemplar','bad','rebox','unverified']) card.classList.remove(k);
        card.classList.add(verdict);
        card.querySelectorAll('.verdict-row button').forEach(b=>b.classList.remove('on'));
        const t = card.querySelector(`.verdict-row button[data-v="${{verdict}}"]`);
        if (t) t.classList.add('on');
      }});
      msg.textContent = '✅ Entire class (' + d.count + ') marked ' + label + ' — refresh to update counts';
    }} else {{ msg.textContent = '❌ ' + (d.detail || 'failed'); }}
  }} catch(e) {{ msg.textContent = '❌ ' + e; }}
}}
document.getElementById('markall-ex').addEventListener('click', () => markAll('exemplar'));
document.getElementById('markall-bad').addEventListener('click', () => markAll('bad'));

// keyboard shortcuts when hovering a card
let hovered = null;
document.querySelectorAll('.card').forEach(c => {{
  c.addEventListener('mouseenter', () => hovered = c);
  c.addEventListener('mouseleave', () => {{ if (hovered === c) hovered = null; }});
}});
document.addEventListener('keydown', e => {{
  if (!hovered) return;
  if (e.key === '1') setVerdict(hovered, 'exemplar');
  else if (e.key === '2') setVerdict(hovered, 'bad');
  else if (e.key === '3') setVerdict(hovered, 'rebox');
  else if (e.key === '0') setVerdict(hovered, 'clear');
}});
</script>
</body></html>'''
    return HTMLResponse(html)


# ===========================================================================
# v3.0.72 (2026-06-01) — /api/per_species_stats + /manual
# ===========================================================================

@app.get("/api/per_species_stats")
def api_per_species_stats():
    """v3.0.72 — Per CWD12 species: count of images in each pipeline stage.

    Stages:
      - gold: images in Roboflow cwd12-multiclass-v1 (frozen human-verified
              benchmark). Source: /api/roboflow_status master.boxes_per_class.
      - auto: images that came pre-labeled at harvest time (registry slug has
              local YOLO .txt files matched to images on disk, primary species
              parsed from first non-empty line).
      - unlabeled: registry-known images with no labels on disk.
      - owl: OWL red proposals (results/framework/owl_red_proposals/<species>/
             *.txt count).
      - exemplars: bank exemplar count per species (object_bank/<sp>/).

    Returns per-species dict. Used by the dashboard's per-species stats panel
    (Phase D of overnight loop)."""
    out = {sp: {"gold": 0, "auto": 0, "unlabeled": 0,
                "owl": 0, "exemplars": 0} for sp in _CWD12}

    # gold: from Roboflow master
    try:
        from weed_optimizer_framework.tools.roboflow_status import (
            get_roboflow_status,
        )
    except ImportError:
        get_roboflow_status = None
    try:
        # reuse the same data /api/roboflow_status pulls
        import urllib.request
        key_path = _ROBOFLOW_KEY_FILE
        if os.path.isfile(key_path):
            with open(key_path) as f:
                rf_key = f.read().strip()
            url = (f"https://api.roboflow.com/a-test-of-will/"
                   f"cwd12-multiclass-v1?api_key={rf_key}")
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            cls = (d.get("project", {}) or {}).get("classes", {}) or {}
            for sp in _CWD12:
                out[sp]["gold"] = int(cls.get(sp, 0))
    except Exception as e:
        log.warning(f"[per_species_stats] gold fetch failed: {e!r}")

    # auto + unlabeled: walk registry slugs
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        for slug, info in (reg.get("datasets") or {}).items():
            if info.get("status") != "downloaded":
                continue
            lp = info.get("local_path", "")
            if not lp or not os.path.isdir(lp):
                continue
            # Walk for image-label pairs (cap 5000 imgs per slug for speed)
            n_imgs = 0
            sp_counts = {sp: 0 for sp in _CWD12}
            n_unlabeled = 0
            img_exts = (".jpg", ".jpeg", ".png")
            for img_p in Path(lp).rglob("*"):
                if img_p.suffix.lower() not in img_exts:
                    continue
                n_imgs += 1
                if n_imgs > 5000:
                    break
                # find matching .txt
                txt_p = img_p.with_suffix(".txt")
                # also check parallel labels/ dir
                if not txt_p.is_file():
                    parts = list(img_p.parts)
                    if "images" in parts:
                        idx = parts.index("images")
                        parts[idx] = "labels"
                        cand = Path(*parts).with_suffix(".txt")
                        if cand.is_file():
                            txt_p = cand
                if txt_p.is_file() and txt_p.stat().st_size > 0:
                    # Parse first line for primary class — see if any cwd12
                    # class_name field on the slug maps the index
                    class_names = info.get("class_names") or []
                    try:
                        first_line = txt_p.read_text().split("\n")[0]
                        cid = int(first_line.split()[0])
                        if 0 <= cid < len(class_names):
                            cname = class_names[cid]
                            if cname in sp_counts:
                                sp_counts[cname] += 1
                                continue
                        # Unknown class index — count as auto-labeled to "other"
                    except Exception:
                        pass
                    # Has txt but no recognized CWD12 species — skip
                else:
                    n_unlabeled += 1
            for sp, c in sp_counts.items():
                out[sp]["auto"] += c
            # Unlabeled gets distributed pro-rata? Simpler: just add to
            # "unlabeled" total per-slug; we'll surface aggregate separately.
            # For now, attach to slug's primary species if class_names has
            # exactly one entry; otherwise drop to "unlabeled" aggregate.
            if n_unlabeled and info.get("class_names"):
                cn = info.get("class_names", [])
                if len(cn) == 1 and cn[0] in out:
                    out[cn[0]]["unlabeled"] += n_unlabeled
    except Exception as e:
        log.warning(f"[per_species_stats] registry walk failed: {e!r}")

    # OWL proposals: count .txt files in results/framework/owl_red_proposals/<sp>/
    owl_root = REPO / "results" / "framework" / "owl_red_proposals"
    if owl_root.is_dir():
        for sp in _CWD12:
            d = owl_root / sp
            if d.is_dir():
                out[sp]["owl"] = sum(1 for p in d.iterdir()
                                     if p.suffix == ".txt"
                                     and p.stat().st_size > 0)

    # exemplars: object_bank/<sp>/* count
    # v3.0.86 (P2) fix: was REPO/"object_bank" (wrong path → always 0). The real
    # bank is results/framework/synth_cutpaste/object_bank (= _BANK_DIR).
    bank_root = _BANK_DIR
    if bank_root.is_dir():
        for sp in _CWD12:
            d = bank_root / sp
            if d.is_dir():
                out[sp]["exemplars"] = sum(
                    1 for p in d.rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )

    # Aggregate row
    agg = {k: sum(out[sp][k] for sp in _CWD12)
           for k in ("gold", "auto", "unlabeled", "owl", "exemplars")}
    return JSONResponse({"per_species": out, "totals": agg,
                          "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


@app.get("/manual", response_class=HTMLResponse)
def manual_page():
    """v3.0.72 — User manual: dual-agent architecture + button reference.

    Single-page documentation explaining what the dashboard does, the full
    data-collection pipeline, the role of each cluster_action button, and
    the recommended daily workflow. Linked from the unified controller's
    header nav. Audience: the user (Harry) + reviewers / professor."""
    html = '''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>📖 Manual — Weed-detection framework</title>
<style>
  :root {
    --bg: #0f172a; --bg-soft: #1e293b; --bg-card: #ffffff;
    --accent: #0e7c66; --accent-grad: linear-gradient(135deg,#0e7c66 0%,#0a9b7a 100%);
    --text: #1a1a1d; --text-soft: #555; --text-faint: #888;
    --border: #e2e8f0; --highlight: #fff8e0;
    --danger: #c53030; --warn: #c70; --ok: #2f855a;
  }
  body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",
         sans-serif; color: var(--text);
         background: linear-gradient(180deg,#f6f8fb 0%,#e9eef6 100%);
         margin: 0; padding: 0; line-height: 1.55; }
  .hero { background: var(--bg); color: #fff; padding: 3rem 2rem 2.5rem;
          text-align: center; box-shadow: 0 4px 30px rgba(0,0,0,.15); }
  .hero h1 { margin: 0 0 .5rem 0; font-size: 2.2rem; font-weight: 700; }
  .hero .sub { color: #9aa5b8; margin-top: .3rem; font-size: 1.05rem; }
  .nav { background: #fff; border-bottom: 1px solid var(--border);
         padding: .7rem 1rem; display: flex; gap: 14px; font-size: 14px;
         justify-content: center; flex-wrap: wrap; position: sticky; top: 0;
         z-index: 10; }
  .nav a { color: var(--accent); text-decoration: none; padding: 4px 10px;
           border-radius: 6px; }
  .nav a:hover { background: #eef4ff; }
  .container { max-width: 980px; margin: 0 auto; padding: 2rem 1.2rem 4rem; }
  h2 { color: var(--accent); margin-top: 2.5rem; font-size: 1.4rem;
       border-bottom: 2px solid var(--accent); padding-bottom: .35rem;
       display: inline-block; }
  h3 { color: var(--text); font-size: 1.05rem; margin-top: 1.6rem;
       margin-bottom: .4rem; }
  p { color: var(--text-soft); }
  .card { background: var(--bg-card); border-radius: 12px; padding: 1.5rem;
          box-shadow: 0 1px 3px rgba(0,0,0,.06); margin: 1rem 0; }
  .pipeline { display: grid; grid-template-columns: 1fr;
              gap: 12px; margin: 1.5rem 0; }
  .stage { background: var(--bg-card); border-radius: 10px;
           padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
           border-left: 4px solid var(--accent); position: relative; }
  .stage .num { position: absolute; right: 16px; top: 14px;
                color: var(--text-faint); font-weight: 600; font-size: 13px; }
  .stage h4 { margin: 0 0 .3rem 0; color: var(--accent); font-size: 1rem; }
  .stage .who { display: inline-block; background: #eef4ff;
                color: var(--accent); padding: 1px 8px; border-radius: 10px;
                font-size: 11px; font-weight: 600; margin-right: 6px; }
  .stage .who.bot { background: #fff8e0; color: var(--warn); }
  .stage .who.h { background: #f0fff4; color: var(--ok); }
  .stage .desc { color: var(--text-soft); font-size: 14px; }
  .stage .btns { margin-top: .4rem; font-size: 12px; color: var(--text-faint); }
  .stage .btns code { background: #f4f6fb; padding: .1rem .35rem;
                       border-radius: 3px; color: var(--accent);
                       font-family: ui-monospace,Menlo,monospace; font-size: 11px; }
  .arrow { text-align: center; color: var(--text-faint); font-size: 22px;
           margin: -4px 0; }
  table.btn-table { width: 100%; border-collapse: collapse; font-size: 13px;
                    margin: .5rem 0; }
  table.btn-table th { text-align: left; background: #f4f6fb;
                       padding: 8px 10px; font-weight: 600; color: var(--text); }
  table.btn-table td { padding: 8px 10px; border-top: 1px solid var(--border);
                       vertical-align: top; }
  table.btn-table td.btn-name { font-family: ui-monospace,Menlo,monospace;
                                 color: var(--accent); white-space: nowrap;
                                 font-size: 12px; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 3px;
           font-size: 10px; font-weight: 600; text-transform: uppercase;
           letter-spacing: .3px; }
  .badge.sb { background: #fce4ec; color: #d62a55; }
  .badge.sub { background: #e7f5ff; color: #06c; }
  .badge.refresh { background: #f0fff4; color: var(--ok); }
  .badge.danger { background: #fff5f5; color: var(--danger); }
  .stat-row { display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr));
              gap: 10px; margin: 1rem 0; }
  .stat { background: var(--bg-card); border-radius: 8px; padding: 14px;
          text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .stat .lbl { color: var(--text-faint); font-size: 12px;
               text-transform: uppercase; letter-spacing: .4px; }
  .stat .val { color: var(--accent); font-size: 26px; font-weight: 700;
               margin-top: 4px; font-family: ui-monospace,Menlo,monospace; }
  ol, ul { color: var(--text-soft); }
  .warn-box { background: #fff8e0; border-left: 4px solid var(--warn);
              padding: 12px 16px; border-radius: 6px; margin: 1rem 0;
              color: #856404; font-size: 14px; }
  .ok-box { background: #f0fff4; border-left: 4px solid var(--ok);
            padding: 12px 16px; border-radius: 6px; margin: 1rem 0;
            color: #2d6f4d; font-size: 14px; }
</style></head><body>
<div class="hero">
  <h1>📖 Weed-detection — User Manual</h1>
  <div class="sub">Dual-agent autonomous data collection + training pipeline · Bridges-2 cluster · MongoDB-backed · v3.0.89 (2026-06)</div>
</div>
<div class="nav">
  <a href="/">🏠 Dashboard</a>
  <a href="/manual">📖 Manual</a>
  <a href="/classes">📋 Classes</a>
  <a href="/slugs">📦 Slugs</a>
  <a href="/roboflow">📊 Roboflow</a>
  <a href="#status">↓ Status</a>
  <a href="#workflow">↓ Workflow</a>
  <a href="#buttons">↓ Buttons</a>
  <a href="#daily">↓ Daily</a>
</div>
<div class="container">

<h2 id="overview">1. The big picture</h2>
<div class="card">
<p>This framework is an <b>autonomous data-collection-and-training agent system</b>
targeting the <b>cottonweed-detection benchmark (cwd12)</b>.
Two agents collaborate via a shared dataset registry + Roboflow workspace:</p>

<div class="stat-row">
  <div class="stat"><div class="lbl">🤖 Agent 1</div><div class="val" style="font-size:18px">Data Collector</div></div>
  <div class="stat"><div class="lbl">🧠 Quality</div><div class="val" style="font-size:18px">DINOv2 Curator</div></div>
  <div class="stat"><div class="lbl">🎯 Auto-label</div><div class="val" style="font-size:18px">OWLv2</div></div>
  <div class="stat"><div class="lbl">🤖 Agent 2</div><div class="val" style="font-size:18px">Trainer (v1 coded)</div></div>
</div>

<p><b>Metric — two levels:</b> <i>interim baseline</i> = cwd12 holdout mAP50-95 ≥ <b>0.90</b>
(fixed 12-class benchmark, comparable to SOTA). <i>True target</i> = mAP50-95 ≥ 0.90 on the
<b>overall, all-species/all-domain, hand-verified, never-train test set</b> that grows as
collection matures. The dataset is the lever that lifts the overall number.</p>
</div>

<h2 id="status">1b. Current status (2026-06, honest)</h2>
<div class="card">
<p><b>✅ Built &amp; verified:</b></p>
<ul>
  <li><b>MongoDB backend</b> — cross-node, authenticated (SCRAM), live on the dashboard node.
      9 dataset slugs / 381 classes / 1 domain loaded. Dashboard <code>/classes</code> +
      <code>/slugs</code> read from Mongo; new harvest dual-writes to Mongo automatically.</li>
  <li><b>Multi-domain extensibility</b> (per Prof's "future flexibility") — every slug/class is
      scoped to a <code>domain</code>; "weed" is domain #1, a new agent (pest, crop-disease…) is an
      additive config entry, <b>no schema migration</b>. See <code>/api/domains</code>.</li>
  <li><b>Roboflow integration</b> — upload / folder mgmt / version gen / download-merge (pull
      labeled ground truth) all wired &amp; run.</li>
  <li><b>Roboflow→training closed loop is coded end-to-end</b> (<code>train_from_roboflow.py</code> +
      <code>run_v3_0_89_roboflow_loop.sh</code>): pull labeled data → train a real YOLO
      (class-remapped to canonical order) → eval on the cwd12 holdout (with stem-level leak
      guard) → report mAP. First real mAP run is pending cluster access.</li>
  <li><b>Honest per-action status</b> — buttons show real succeeded/running/failed (from sacct +
      logs), not just "launched". Holdout leak-protection verified (1,977 cwd12 stems).</li>
</ul>
<p><b>⚠️ Known gaps (being worked):</b></p>
<ul>
  <li><b>Only 8 of 12 weed species have training data</b> — Eclipta, Goosegrass, Morningglory,
      Nutsedge currently exist only in the eval holdout (zero train data). Closing this is the
      data priority.</li>
  <li><b>~5,928 real weed-bbox vs a 50K target</b> (gap ~44K) — needs more harvesting biased to weeds.</li>
  <li><b>OWL auto-label</b> was over-proposing (~600 boxes/img, precision ≈0); fixed with NMS +
      per-image top-k (re-verification pending).</li>
  <li><b>First end-to-end training mAP</b> not yet produced (closed-loop coded, awaiting a clean
      cluster run).</li>
</ul>
</div>

<h2 id="workflow">2. The pipeline (8 stages)</h2>

<div class="pipeline">
  <div class="stage">
    <span class="num">stage 1</span>
    <h4>🌐 Discover</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Brain searches HuggingFace, Kaggle, GitHub, Roboflow Universe for new
      weed/crop datasets. Topic classifier filters out off-goal (disease,
      pest, tree, etc.) BEFORE downloading.
    </p>
    <div class="btns">Buttons: <code>brain_harvest</code> (configurable duration + strict mode), <code>refresh_registry</code></div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 2</span>
    <h4>⬇️ Download</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Selected slugs downloaded to <code>datasets/&lt;slug&gt;/</code> on the cluster.
      Auto-classify topic + auto-detect any existing labels.
    </p>
    <div class="btns">Buttons: <code>download_known_slugs</code> (heavy)</div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 3</span>
    <h4>🧹 Garbage audit</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Drop slugs with <code>&lt;100 labels</code> OR <code>classes=0</code>. Disk-first
      check (counts only .txt files with matching image stems + non-empty),
      so brain's fuzzy metadata can't cause false positives.
    </p>
    <div class="btns">Buttons: <code>audit_registry_garbage</code> (dry-run), <code>audit_registry_garbage_APPLY</code> (deletes files)</div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 4</span>
    <h4>🔍 Bucket + classify</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Per-slug bucket assignment (A=detection-ready, B=class-only, C=unknown)
      + DINOv2-based CWD12 species routing.
    </p>
    <div class="btns">Buttons: <code>build_buckets</code>, <code>dinov2_route_classes</code>, <code>topic_backfill</code></div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 5</span>
    <h4>🧠 DINOv2 quality filter</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Reference pool built from trusted bbox slugs (cwd12 family +
      weedsense + grass_weeds + ...). Score every new slug's mean cosine
      similarity. Below threshold → reject.
    </p>
    <div class="btns">Buttons: <code>dinov2_curate_registry</code> (4h GPU)</div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 6</span>
    <h4>🎯 OWL auto-label</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Image-conditioned OWLv2 detection. Uses object_bank exemplars as
      visual queries → proposes bboxes on unlabeled targets. All proposals
      tagged <b>red</b> (= needs human approval).
    </p>
    <div class="btns">Buttons: <code>export_owl_exemplars</code>, <code>owl_preannotate_one</code> (sbatch GPU), <code>owl_upload_proposals</code></div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 7</span>
    <h4>📡 Sync to Roboflow</h4>
    <p class="desc">
      <span class="who bot">AGENT 1</span>
      Upload (image + .txt YOLO labels) to <code>weed-crop-agent-dataset</code>
      project inside <code>weed_crop_agent_dataset</code> folder. Auto-skips
      garbage candidates. Folder placement is idempotent (PATCH /groups).
    </p>
    <div class="btns">Buttons: <code>sync_newest_slugs</code>, <code>roboflow_sync_cwd12_v1</code>, <code>roboflow_sync_agent_v1</code>, <code>roboflow_state_audit</code>, <code>roboflow_list_folders</code>, <code>roboflow_move_agent_to_folder</code>, <code>roboflow_generate_versions</code>, <code>roboflow_download_merge</code></div>
  </div>
  <div class="arrow">↓</div>

  <div class="stage">
    <span class="num">stage 8</span>
    <h4>👤 Human refinement</h4>
    <p class="desc">
      <span class="who h">HUMAN</span>
      Open Roboflow web UI. Red boxes → approve / correct. Approved boxes
      flip from red to green (= gold). The dashboard <code>/classes</code> page
      gives an alternative ✓/✗ approval flow with side-by-side originals.
    </p>
    <div class="btns">Dashboard pages: <code>/classes</code>, <code>/slugs</code></div>
  </div>
  <div class="arrow">↓ (FUTURE)</div>

  <div class="stage" style="border-left-color:#888;opacity:.85">
    <span class="num">stage 9</span>
    <h4>🚀 Agent 2: Trainer (future)</h4>
    <p class="desc">
      <span class="who bot" style="background:#f4f6fb;color:#888">AGENT 2</span>
      Pulls Roboflow Version of the curated agent project + cwd12
      benchmark. Fine-tunes YOLO/RF-DETR. Reports mAP back to dashboard.
      Targets ≥ 0.90 cwd12 mAP50-95.
    </p>
    <div class="btns">Not yet implemented. Planned: SBATCH script + new cluster_action.</div>
  </div>
</div>

<h2 id="buttons">3. Every dashboard button explained</h2>
<div class="card">
<p>21 cluster actions, grouped by phase. Click <code>/</code> to see them inline
with one-click triggers.</p>

<h3>Discovery + collection</h3>
<table class="btn-table">
  <tr><th>Button</th><th>Type</th><th>What it does</th></tr>
  <tr><td class="btn-name">brain_harvest</td><td><span class="badge sb">SBATCH 1-8h</span></td>
      <td>One round of dataset hunting on HF + Kaggle + GitHub + Roboflow.
          Form on dashboard: <b>time_h</b> (1/2/4/6h), <b>strict</b>
          (post-download quality gate: reject if labeled&lt;100 OR
          classes=0), <b>max_new</b> (cap per round).</td></tr>
  <tr><td class="btn-name">download_known_slugs</td><td><span class="badge sb danger">SBATCH heavy</span></td>
      <td>Bulk-download every <code>status=known</code> slug. Bandwidth-heavy.
          Rarely needed; brain_harvest covers the common case.</td></tr>
  <tr><td class="btn-name">refresh_registry</td><td><span class="badge refresh">refresh</span></td>
      <td>Wipe class-pool cache + reload registry index. Click after manual
          edits or to force re-fetch.</td></tr>
</table>

<h3>Quality + classification</h3>
<table class="btn-table">
  <tr><td class="btn-name">audit_registry_garbage</td><td><span class="badge sub">subprocess</span></td>
      <td>Dry-run scan: list slugs with labeled&lt;100 OR classes=0.
          Protects cwd12 baselines by default. Reports what would drop.</td></tr>
  <tr><td class="btn-name">audit_registry_garbage_APPLY</td><td><span class="badge sub danger">subprocess</span></td>
      <td>Actually drop the slugs the dry-run listed. Removes registry
          entries AND <code>rmtree</code>'s the downloaded files on disk.</td></tr>
  <tr><td class="btn-name">build_buckets</td><td><span class="badge sub">subprocess ~30s</span></td>
      <td>Audit each slug → A (detection-ready) / B (class-only) / C
          (unknown). Outputs <code>buckets.json</code>.</td></tr>
  <tr><td class="btn-name">topic_backfill</td><td><span class="badge sb">SBATCH 15min</span></td>
      <td>Ollama + topic classifier on slugs missing the topic_override
          field. Lets the harvest filter pre-reject off-goal slugs.</td></tr>
  <tr><td class="btn-name">dinov2_route_classes</td><td><span class="badge sb">SBATCH 1×V100 ~30min</span></td>
      <td>DINOv2 routes unknown bucket images to a CWD12 species via
          nearest-neighbor against the object_bank exemplars.</td></tr>
  <tr><td class="btn-name">dinov2_curate_registry</td><td><span class="badge sb">SBATCH 1×V100 ~4h</span></td>
      <td>Build DINOv2 reference pool from trusted bbox slugs, then score
          every registry slug. Ranked report at
          <code>results/framework/dinov2_curator/slug_scores.json</code>.</td></tr>
</table>

<h3>OWL auto-label</h3>
<table class="btn-table">
  <tr><td class="btn-name">export_owl_exemplars</td><td><span class="badge sub">subprocess ~3s</span></td>
      <td>Read object_bank/&lt;sp&gt;/* → write per-species exemplar JSON to
          <code>results/framework/owl_exemplars/&lt;sp&gt;.json</code>. Required
          before owl_preannotate.</td></tr>
  <tr><td class="btn-name">owl_preannotate_one</td><td><span class="badge sb">SBATCH 1×V100 ~10-30min</span></td>
      <td>OWLv2-large image-conditioned detection. Reads exemplars +
          target images, writes YOLO .txt labels marked
          <code>red conf=… src=owlv2</code> in
          <code>results/framework/owl_red_proposals/&lt;sp&gt;/</code>.</td></tr>
  <tr><td class="btn-name">owl_upload_proposals</td><td><span class="badge sub">subprocess ~5min</span></td>
      <td>Upload OWL-proposed (image, .txt) pairs to Roboflow tagged
          <code>batch=red</code> for human review.</td></tr>
</table>

<h3>Roboflow sync</h3>
<table class="btn-table">
  <tr><td class="btn-name">sync_newest_slugs</td><td><span class="badge sub">subprocess</span></td>
      <td>Iterate registry, upload any <code>status=downloaded</code> slug
          not yet <code>roboflow_synced</code>. Skips cwd12 baselines (frozen)
          + audit-garbage candidates + slug_verdicts.junk. After loop:
          place target project into the agent folder (idempotent PATCH).</td></tr>
  <tr><td class="btn-name">roboflow_sync_cwd12_v1</td><td><span class="badge sub">subprocess ~10min</span></td>
      <td>Upload the cwd12 frozen benchmark (598 images, 822 boxes,
          12 classes) to <code>cwd12-multiclass-v1</code>. Dedup-safe.</td></tr>
  <tr><td class="btn-name">roboflow_sync_agent_v1</td><td><span class="badge sub">subprocess ~10min</span></td>
      <td>Upload the same payload but to <code>weed-crop-agent-dataset</code>.
          Mostly for early testing; in steady state <code>sync_newest_slugs</code>
          replaces it.</td></tr>
  <tr><td class="btn-name">roboflow_state_audit</td><td><span class="badge sub">subprocess ~3s</span></td>
      <td>List every project in workspace + image / box / version counts.
          Writes <code>roboflow_state.json</code>.</td></tr>
  <tr><td class="btn-name">roboflow_generate_versions</td><td><span class="badge sub">subprocess ~30s</span></td>
      <td>Generate a Roboflow Version for the agent project (Public-tier
          generation can stall — paid tier is faster). Idempotent.</td></tr>
  <tr><td class="btn-name">roboflow_download_merge</td><td><span class="badge sub">subprocess ~5min</span></td>
      <td>Pull the latest Version back as YOLO zip, merge with cwd12 to
          form a unified multi-class training set. Requires a generated
          version.</td></tr>
  <tr><td class="btn-name">roboflow_list_folders</td><td><span class="badge sub">subprocess</span></td>
      <td>List all folders (groups) in workspace + their member projects.
          Verifies <code>weed-crop-agent-dataset</code> is inside
          <code>weed_crop_agent_dataset</code> folder.</td></tr>
  <tr><td class="btn-name">roboflow_move_agent_to_folder</td><td><span class="badge sub">subprocess</span></td>
      <td>PATCH the agent project into its folder. Idempotent — clicking
          when already in folder is a no-op (HTTP 204).</td></tr>
</table>

<h3>System</h3>
<table class="btn-table">
  <tr><td class="btn-name">restart_dashboard</td><td><span class="badge sb danger">restart_self</span></td>
      <td>Cancel + sbatch the dashboard server SLURM job. ~90s downtime,
          new tunnel URL minted. github.io tunnel_url.json auto-updates.
          Use sparingly — browser auth cache lost per restart.</td></tr>
</table>

<h3>Added since v3.0.72 (rounds, full-loop, training, all-sync)</h3>
<table class="btn-table">
  <tr><td class="btn-name">start_new_round</td><td><span class="badge sub">subprocess</span></td>
      <td>Begin a new harvest round v{N}→v{N+1} (creates the RF project; no data yet).</td></tr>
  <tr><td class="btn-name">harvest_full_round_e2e</td><td><span class="badge sb">sbatch 4h</span></td>
      <td>One-click round: bump round → harvest 4h → auto-sync survivors to the round's RF project.</td></tr>
  <tr><td class="btn-name">backfill_round_1</td><td><span class="badge sub">subprocess</span></td>
      <td>Stamp harvest_round=1 on pre-v3.0.74 slugs (idempotent).</td></tr>
  <tr><td class="btn-name">dinov2_filter_round_1</td><td><span class="badge sub">subprocess</span></td>
      <td>DINOv2-filter a round's unverified slugs → upload survivors as agent-v1-dinov2-v{X.Y}.</td></tr>
  <tr><td class="btn-name">sync_all_to_roboflow</td><td><span class="badge sub">subprocess</span></td>
      <td>Upload ALL slugs (incl. cwd12 baselines) to Roboflow for fast review.</td></tr>
  <tr><td class="btn-name">train_yolo_round_1</td><td><span class="badge sub">subprocess</span></td>
      <td>Send round-1 verified ✓ set to the trainer. NOTE: this button is still the
          v3.0.74 <b>placeholder stub</b>; the REAL training is the closed-loop
          <code>run_v3_0_89_roboflow_loop.sh</code> / <code>train_from_roboflow.py</code>
          (pull labeled RF data → train YOLO → eval cwd12 holdout → mAP).</td></tr>
</table>
<p style="color:#555"><b>Roboflow closed loop (top priority, professor-facing):</b> the real
training path is <code>train_from_roboflow.py</code> — pull human-labeled Roboflow ground truth →
train a real YOLO (classes remapped to canonical order) → eval on the cwd12 holdout with a
stem-level leak guard → report mAP50-95 + gap to 0.90. Run via
<code>run_v3_0_89_roboflow_loop.sh</code> (RF_DOWNLOAD=1).</p>

<p style="color:#555"><b>Storage:</b> the whole registry + classes + per-image metadata live in
<b>MongoDB</b> (cross-node, authenticated). The dashboard's 🗄️ card shows whether it's serving
from Mongo (🟢) or the JSON fallback. <code>/api/domains</code> lists collection-agent domains.</p>
</div>

<h2 id="daily">4. Recommended daily workflow</h2>
<div class="card">
<ol>
  <li><b>Morning</b>: Open <a href="/">🏠 dashboard</a> → check the live banner + the "Recent agent task runs" panel for overnight progress.</li>
  <li><b>Harvest</b>: Click <code>brain_harvest</code> with <b>strict ON</b>, <b>4h</b> duration. Walk away for a few hours.</li>
  <li><b>Check results</b>: when SBATCH brain_harvest job completes, return. Look at the stat-row TOTAL IMAGES — should have grown.</li>
  <li><b>Audit</b>: click <code>audit_registry_garbage</code> (dry-run first). Read the result. If happy → click <code>audit_registry_garbage_APPLY</code> to delete garbage.</li>
  <li><b>Quality score</b>: click <code>dinov2_curate_registry</code> for a fresh quality ranking (4h GPU; can run overnight).</li>
  <li><b>Sync to Roboflow</b>: click <code>sync_newest_slugs</code> → uploads survivors to the agent folder.</li>
  <li><b>Refine</b>: open Roboflow → <code>weed-crop-agent-dataset</code> project → annotate red boxes. Approve good ones (they flip green).</li>
  <li><b>Auto-label more</b>: for the next batch of unlabeled, click <code>export_owl_exemplars</code> then <code>owl_preannotate_one</code> then <code>owl_upload_proposals</code>.</li>
  <li><b>Repeat</b> until you have enough labeled data per species for Agent 2 to train.</li>
</ol>

<div class="ok-box"><b>📊 Watch the dashboard stats</b> while the loop runs —
the SLURM queue, Roboflow workspace panel, and CWD12 species snapshot all
auto-refresh.</div>
</div>

<h2 id="trouble">5. Troubleshooting</h2>
<div class="card">
<h3>"Cluster unreachable: TypeError: Failed to fetch"</h3>
<p>Fixed in v3.0.71.7 (CORS preflight). If you still see this, the
dashboard SLURM job may have just restarted — wait 90s and refresh.</p>

<h3>"Wrong password" after typing in github.io prompt</h3>
<p>Most common cause: trailing whitespace on copy-paste. The v3.0.71.5 page
strips whitespace automatically. If still rejected, verify the exact
characters match what's in <code>/jet/home/byler/.dashpass</code>.</p>

<h3>"5 wrong → IP locked 1h"</h3>
<p>Per-IP rate limit. Wait 1 hour OR restart the dashboard
(<code>restart_dashboard</code>) which clears the in-memory counter.</p>

<h3>Tunnel URL changed</h3>
<p>Quick cloudflared tunnels rotate per SLURM restart. The stable
github.io URL <code>harry567566.github.io/weed-dashboard/</code> reads
<code>tunnel_url.json</code> and redirects to the current URL. Always
bookmark the github.io URL, never the raw trycloudflare one.</p>

<h3>A button POSTs but the subprocess fails</h3>
<p>Click into the dashboard's recent agent task runs (📜 icon) to read the
log file for the action. Common issues: wrong path in argv (registry's
local_path != the path I assumed), missing exemplars, Roboflow API rate
limit.</p>
</div>

<div style="text-align:center;color:#999;font-size:12px;margin-top:3rem">
  v3.0.72 · 2026-06-01 · <a href="/" style="color:var(--accent)">back to dashboard</a>
</div>
</div>
</body></html>'''
    return HTMLResponse(html)


# ===========================================================================
# v3.0.74 (2026-06-01) — Round tracking + /rounds review UI
# ===========================================================================

@app.get("/api/rounds_state")
def api_rounds_state():
    """Return registry's round tracking: current_round + per-round slug list
    + per-slug class info + Other-bucket detection. Used by /rounds UI."""
    try:
        from weed_optimizer_framework.tools.rounds import status
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})
    s = status()
    # Enrich per_round_slugs with each slug's class_names + verdict state
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
    except Exception:
        reg = {"datasets": {}}
    ds = reg.get("datasets", {}) or {}

    # Load existing slug + class verdicts
    sv = _slug_verdict_state()  # {slug: latest_verdict}
    # exemplar verdicts are per-image; we summarize per-class
    cv = {}  # {slug: {class: count_verified}}
    try:
        ev_path = REPO / "results" / "framework" / "exemplar_verdicts.jsonl"
        if ev_path.exists():
            for line in ev_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    sl = e.get("slug") or e.get("source_slug")
                    cl = e.get("class") or e.get("cls")
                    v = e.get("verdict")
                    if sl and cl and v == "exemplar":
                        cv.setdefault(sl, {}).setdefault(cl, 0)
                        cv[sl][cl] += 1
                except Exception:
                    continue
    except Exception:
        pass

    # v3.0.76.1: include EMPTY rounds too. The /rounds JS iterates `rounds`
    # and didn't show round 2/3 when they were just-created with 0 slugs —
    # user couldn't tell start_new_round had worked. Union of rounds_meta
    # keys and per_round_slugs ensures every round announced via
    # start_new_round shows up immediately.
    all_round_keys = set(s["per_round_slugs"].keys()) | set(
        str(k) for k in (s.get("rounds_meta") or {}).keys())
    enriched = {}
    for round_num in all_round_keys:
        slugs = s["per_round_slugs"].get(round_num, [])
        enriched[round_num] = []
        for slug in slugs:
            info = ds.get(slug, {})
            cn = info.get("class_names") or []
            # Detect "Other" bucket — numeric-only or empty class_names
            is_other = (
                len(cn) == 0
                or all(str(x).isdigit() for x in cn if x is not None)
            )
            enriched[round_num].append({
                "slug": slug,
                "local_images": info.get("local_images", 0),
                "source": info.get("source", "?"),
                "class_names": cn,
                "is_other": is_other,
                "slug_verdict": sv.get(slug),
                "class_verdicts": cv.get(slug, {}),
                "roboflow_synced": info.get("roboflow_synced", False),
            })
    return JSONResponse({
        "ok": True,
        "current_round": s["current_round"],
        "rounds_meta": s["rounds_meta"],
        "per_round_counts": s["per_round_counts"],
        "rounds": enriched,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


@app.get("/rounds", response_class=HTMLResponse)
def rounds_page():
    """v3.0.74 — per-round review hub.

    User vision: each harvest round = a versioned snapshot. Show every slug
    in the round grouped by:
      • Has class names + bbox → 'Categorized'
      • Has class names only (digits or empty) → 'Other' bucket
      • Garbage slug (slug_verdict=junk) → already filtered out

    For each class within a slug: ✓ / ✗ / 🔄 (manual relabel).
    Click anywhere → AJAX POST verdict, panel updates immediately."""
    html = '''<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>🔄 Rounds — review per harvest version</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:#1a1a1d;background:linear-gradient(180deg,#f6f8fb 0%,#e9eef6 100%);
       margin:0;padding:0;min-height:100vh}
  .hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
        color:#fff;padding:1.6rem 2rem 1.4rem;box-shadow:0 4px 30px rgba(0,0,0,.15)}
  .hero h1{margin:0;font-size:1.55rem;font-weight:700}
  .hero .sub{color:#9aa5b8;margin-top:.3rem;font-size:.9rem}
  .nav{background:#fff;border-bottom:1px solid #e2e8f0;padding:.7rem 1rem;
       display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
  .nav a{color:#0e7c66;text-decoration:none;padding:4px 12px;
         border-radius:18px;border:1px solid #e2e8f0;font-size:13px;
         transition:all .15s;background:#fff}
  .nav a:hover{background:#0e7c66;color:#fff;transform:translateY(-1px)}
  .container{max-width:1200px;margin:0 auto;padding:1.5rem 1rem 4rem}
  .round-card{background:#fff;border-radius:14px;padding:1.2rem 1.5rem;
              margin:1rem 0;box-shadow:0 2px 8px rgba(15,23,42,.05),
              0 1px 3px rgba(15,23,42,.04);border:1px solid #f0f4f8}
  .round-card.current{border-left:4px solid #0e7c66;background:linear-gradient(180deg,#fff 0%,#f4faf8 100%)}
  .round-header{display:flex;align-items:center;justify-content:space-between;
                margin-bottom:1rem;border-bottom:1px solid #f0f4f8;padding-bottom:.6rem}
  .round-header h2{margin:0;font-size:1.2rem;color:#0f172a}
  .round-meta{font-size:.8rem;color:#888}
  .badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;
         font-weight:600;background:#eef4ff;color:#0e7c66;margin-left:.4rem}
  .badge.cur{background:#0e7c66;color:#fff}
  .slug-row{background:#f7f9fc;border:1px solid #e8eef5;border-radius:8px;
            padding:10px 14px;margin:.5rem 0;display:grid;
            grid-template-columns:1fr auto auto;gap:10px;align-items:start}
  .slug-row.other{background:linear-gradient(135deg,#fff8e0,#fef3c7);
                  border-color:#fbbf24}
  .slug-row.junk{opacity:.5;background:#fef2f2;text-decoration:line-through}
  /* v3.0.76.1: visual feedback for kept slugs */
  .slug-row.kept{background:linear-gradient(135deg,#f0fff4,#dcfce7);
                 border-color:#22c55e;border-left:4px solid #0e7c66}
  .slug-row.kept .slug-name::before{content:"✓ ";color:#0e7c66;font-weight:700}
  .slug-name{font-weight:600;color:#0f172a;font-size:14px}
  .slug-name code{background:#fff;padding:.1rem .35rem;border-radius:3px;
                  font-family:ui-monospace,Menlo,monospace;font-size:11px;
                  color:#0e7c66}
  .slug-meta{font-size:11px;color:#888;margin-top:3px}
  .classes-list{margin-top:6px;display:flex;flex-wrap:wrap;gap:6px}
  .cls-chip{background:#fff;border:1px solid #d8e2eb;border-radius:14px;
            padding:3px 10px;font-size:11px;color:#444;
            display:inline-flex;align-items:center;gap:4px;cursor:pointer;
            transition:all .12s}
  .cls-chip:hover{background:#eef4ff;border-color:#0e7c66}
  .cls-chip.ok{background:linear-gradient(135deg,#f0fff4,#dcfce7);
               border-color:#22c55e;color:#0e7c66;font-weight:600}
  .cls-chip.ok::before{content:"✓ "}
  .cls-chip.bad{background:linear-gradient(135deg,#fee,#fecaca);
                border-color:#c44;color:#c44}
  .cls-chip.bad::before{content:"✗ "}
  .cls-chip.uncategorized{background:#fff8e0;border-color:#fbbf24;color:#c70}
  .cls-chip.uncategorized::before{content:"🔶 "}
  .actions{display:flex;flex-direction:column;gap:4px;align-items:flex-end}
  button.act{background:#fff;border:1px solid #d8e2eb;border-radius:6px;
             padding:3px 10px;font-size:11px;cursor:pointer;font-family:inherit}
  button.act.ok{background:#22c55e;color:#fff;border-color:#22c55e}
  button.act.bad{background:#c44;color:#fff;border-color:#c44}
  button.act:hover{transform:translateY(-1px);box-shadow:0 2px 6px rgba(0,0,0,.1)}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
         gap:10px;margin:1rem 0}
  .stat{background:#fff;border-radius:8px;padding:10px 14px;
        box-shadow:0 1px 3px rgba(0,0,0,.05);text-align:center}
  .stat .lbl{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.4px}
  .stat .val{font-size:22px;font-weight:700;color:#0f172a;
             font-family:ui-monospace,Menlo,monospace}
  .rev-thumbs{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;align-items:center}
  .rev-thumbs img{width:84px;height:64px;object-fit:cover;border-radius:5px;
        border:1px solid #d7dde6;background:#eef1f5;cursor:zoom-in;transition:transform .12s}
  .rev-thumbs img:hover{transform:scale(1.04);border-color:#0e7c66}
  .rev-thumbs .th-load,.rev-thumbs .th-empty{font-size:11px;color:#94a3b8}
  .act.view{background:#0e7c66;color:#fff;border-color:#0e7c66;text-decoration:none;
        display:inline-block}
  .toast{position:fixed;bottom:20px;right:20px;background:#0e7c66;color:#fff;
         padding:10px 16px;border-radius:8px;font-size:13px;display:none;
         box-shadow:0 4px 12px rgba(14,124,102,.3);z-index:100}
  .toast.show{display:block;animation:slidein .2s ease-out}
  @keyframes slidein{from{transform:translateY(20px);opacity:0}to{transform:none;opacity:1}}
</style></head><body>
<div style="padding:10px 16px;background:#0b1220"><a href="/agent/weed" style="display:inline-block;text-decoration:none;background:#1e293b;color:#93c5fd;font-weight:600;font-size:13px;padding:7px 13px;border-radius:8px">&larr; Mission Control</a></div>
<div class="hero">
  <h1>🔄 Harvest Rounds — version-by-version review</h1>
  <div class="sub">Each round = a snapshot of what the agent collected. Click ✓/✗ on each class to flag for DINOv2.</div>
</div>
<div class="nav">
  <a href="/">🏠 Dashboard</a>
  <a href="/manual">📖 Manual</a>
  <a href="/rounds" style="background:#0e7c66;color:#fff">🔄 Rounds</a>
  <a href="/classes">📋 Classes</a>
  <a href="/slugs">📦 Slugs</a>
  <a href="/roboflow">📊 Roboflow</a>
</div>
<div class="container">
  <div style="background:linear-gradient(135deg,#dbeafe,#bfdbfe);border-left:4px solid #1d4ed8;
              padding:.9rem 1.2rem;border-radius:8px;margin-bottom:1rem;font-size:13px;color:#1e3a8a">
    <strong>💡 Workflow:</strong> harvest → sync to Roboflow → <b>review HERE</b>.
    Each slug now shows an inline <b>boxed preview</b> (YOLO boxes drawn server-side);
    click a thumb or <strong style="background:#fff;padding:.1rem .4rem;border-radius:3px;color:#0e7c66">🖼️ View all images + boxes</strong>
    for the full boxed gallery, judge quality, then <b>✓ keep / ✗ junk</b> right here — no need to
    leave for Roboflow. (📡 Roboflow ↗ still available for the web labeler / faster CDN.)
  </div>
  <div class="stats" id="stats">loading…</div>
  <div id="rounds-content">loading…</div>
</div>
<div class="toast" id="toast">saved</div>
<script>
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),1500)}

async function loadRounds(){
  const r = await fetch('/api/rounds_state', {credentials:'include'});
  const d = await r.json();
  if(!d.ok){document.getElementById('rounds-content').innerHTML = 'err: '+(d.error||'?');return}
  const cur = d.current_round;
  // Stats row
  const totalSlugs = Object.values(d.per_round_counts||{}).reduce((a,b)=>a+b,0);
  const totalRounds = Object.keys(d.rounds||{}).length;
  let other = 0, categorized = 0;
  for(const r of Object.values(d.rounds||{}))
    for(const s of r){ if(s.is_other) other++; else categorized++; }
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="lbl">Current Round</div><div class="val">v${cur}</div></div>
    <div class="stat"><div class="lbl">Total Rounds</div><div class="val">${totalRounds}</div></div>
    <div class="stat"><div class="lbl">Total Slugs</div><div class="val">${totalSlugs}</div></div>
    <div class="stat"><div class="lbl">Categorized</div><div class="val" style="color:#0e7c66">${categorized}</div></div>
    <div class="stat"><div class="lbl">🔶 Other</div><div class="val" style="color:#c70">${other}</div></div>
  `;
  // Render rounds, newest first
  const sorted = Object.keys(d.rounds||{}).map(Number).sort((a,b)=>b-a);
  let html = '';
  for(const rn of sorted){
    const slugs = d.rounds[rn] || [];
    const meta = (d.rounds_meta||{})[String(rn)] || {};
    const isCur = (rn === cur);
    // v3.0.76: each round → its own RF project (round 1 legacy name)
    const projName = (rn === 1) ? 'weed-crop-agent-dataset' : ('weed-crop-agent-v' + rn);
    const projUrl = `https://app.roboflow.com/a-test-of-will/${projName}/browse`;
    html += `<div class="round-card${isCur?' current':''}">
      <div class="round-header">
        <h2>Round v${rn}${isCur?' <span class="badge cur">CURRENT</span>':''}
          <a href="${projUrl}" target="_blank" style="font-size:.7em;color:#0e7c66;text-decoration:none;margin-left:.6rem;font-weight:500">
            📡 ${projName} ↗
          </a>
        </h2>
        <div class="round-meta">started: ${meta.started_at||'?'} · ${slugs.length} slugs · RF project: <code style="background:#f4f6fb;padding:.05rem .3rem;border-radius:3px">${projName}</code></div>
        ${(function(){
          const tr = meta.train_results || {};
          const m = (tr.map50_95!=null?tr.map50_95:(tr['mAP50-95']!=null?tr['mAP50-95']:meta.map50_95));
          const subv = meta.dinov2_subversions || [];
          let trainTxt;
          if(meta.trained){
            trainTxt = '<span style="color:#0e7c66;font-weight:600">✅ trained</span>';
            if(m!=null){
              trainTxt += ' · cwd12 mAP50-95 <b>'+m+'</b>';
              // research goal is locked at >= 0.90 — always show the gap
              const gap = (tr.gap_to_0_90!=null)?tr.gap_to_0_90:Math.round((0.90-m)*1e4)/1e4;
              if(gap>0) trainTxt += ' <span style="color:#dc2626">(gap to 0.90: '+gap.toFixed(4)+')</span>';
              else trainTxt += ' <span style="color:#16a34a">(✓ goal met)</span>';
              const parts=[];
              if(tr.cwd12_test_map50_95!=null) parts.push('test '+tr.cwd12_test_map50_95);
              if(tr.cwd12_valid_map50_95!=null) parts.push('valid '+tr.cwd12_valid_map50_95);
              if(parts.length) trainTxt += ' <span style="color:#64748b;font-size:.9em">['+parts.join(' · ')+']</span>';
              if(tr.model_label) trainTxt += ' <span style="color:#94a3b8;font-size:.9em">'+tr.model_label+'</span>';
            } else { trainTxt += ' · mAP pending'; }
            if(meta.trained_at) trainTxt += ' <span style="color:#94a3b8;font-size:.85em">@ '+meta.trained_at+'</span>';
          } else {
            trainTxt = '<span style="color:#94a3b8">⚪ not trained yet</span>';
          }
          const dinoTxt = subv.length ? (' · DINOv2: '+subv.join(', ')) : '';
          return '<div class="round-meta">Results: '+trainTxt+dinoTxt+'</div>';
        })()}
      </div>`;
    if(slugs.length === 0){
      html += '<div style="color:#888;font-size:13px">no slugs in this round yet — fire brain_harvest from the dashboard</div>';
    } else {
      // Sort: categorized first, then 🔶 Other
      slugs.sort((a,b)=> (a.is_other?1:0) - (b.is_other?1:0));
      for(const s of slugs){
        const cls_other = s.is_other ? ' other' : '';
        const cls_junk = (s.slug_verdict === 'junk') ? ' junk' : '';
        // v3.0.76.1: add visual kept styling so user sees ✓ flag immediately
        const cls_kept = (s.slug_verdict === 'keep') ? ' kept' : '';
        // v3.0.76: each round = its own Roboflow project, NOT batch tags.
        // Round 1 = legacy 'weed-crop-agent-dataset', round 2+ = 'weed-crop-agent-v{N}'.
        const projName = (rn === 1) ? 'weed-crop-agent-dataset' : ('weed-crop-agent-v' + rn);
        const rfBatchUrl = s.roboflow_synced
          ? `https://app.roboflow.com/a-test-of-will/${projName}/browse?queryText=tag%3A${encodeURIComponent(s.slug)}`
          : null;
        html += `<div class="slug-row${cls_other}${cls_junk}${cls_kept}">
          <div>
            <div class="slug-name">${s.is_other?'🔶 ':''}<code>${s.slug}</code></div>
            <div class="slug-meta">${s.local_images.toLocaleString()} imgs · ${s.source} ·
              ${s.roboflow_synced
                ? '<a href="'+rfBatchUrl+'" target="_blank" style="color:#0e7c66;font-weight:600">📡 on Roboflow ↗</a>'
                : '<span style="color:#c70">⚠ NOT on Roboflow yet — click sync_all_to_roboflow on dashboard</span>'
              }
            </div>
            <div class="classes-list">`;
        if(s.class_names && s.class_names.length){
          for(const cn of s.class_names){
            const verified = (s.class_verdicts||{})[cn] || 0;
            const cls = verified>0 ? 'ok' : '';
            html += `<span class="cls-chip ${cls}" title="${verified} verified" onclick="markClass('${s.slug}','${cn}','exemplar')">${cn}${verified>0?' ('+verified+')':''}</span>`;
          }
        } else {
          html += '<span class="cls-chip uncategorized">no class_names — needs manual labeling in Roboflow</span>';
        }
        html += `</div>
            <div class="rev-thumbs" id="th_${s.slug}" data-slug="${s.slug}"><span class="th-load">🖼️ Loading boxed previews…</span></div>
          </div>
          <div class="actions">
            <a class="act view" href="/gallery/${encodeURIComponent(s.slug)}" target="_blank">🖼️ View all images + boxes</a>
            <button class="act ok" onclick="markSlug('${s.slug}','keep')">✓ keep slug</button>
            <button class="act bad" onclick="markSlug('${s.slug}','junk')">✗ junk slug</button>
            ${s.roboflow_synced
              ? `<button class="act" onclick="window.open('${rfBatchUrl}','_blank')" style="background:#0e7c66;color:#fff;border-color:#0e7c66">📡 Roboflow ↗</button>`
              : `<button class="act" disabled title="Not synced yet" style="opacity:.5">⚠ Not synced</button>`}
          </div>
          <div></div>
        </div>`;
      }
    }
    html += '</div>';
  }
  document.getElementById('rounds-content').innerHTML = html || '<div class="round-card">no rounds yet — fire brain_harvest from the dashboard</div>';
  loadThumbs();
}

// v3.0.99: inline boxed-thumbnail preview per slug, so the human can VISUALLY
// review each round's images (with bboxes) WITHOUT leaving for Roboflow. Thumbs
// come from /api/sample (server renders YOLO boxes); cached so the 30s refresh
// doesn't refetch. Click a thumb → full boxed gallery /gallery/{slug}.
const thumbCache = {};
async function loadThumbs(){
  const boxes = document.querySelectorAll('.rev-thumbs[data-slug]');
  for(const box of boxes){
    const slug = box.dataset.slug;
    try{
      let files = thumbCache[slug];
      if(files === undefined){
        const r = await fetch('/api/slug/'+encodeURIComponent(slug)+'/samples?n=4', {credentials:'include'});
        const d = await r.json();
        files = d.samples || [];
        thumbCache[slug] = files;
      }
      if(!files.length){ box.innerHTML = '<span class="th-empty">— no local images —</span>'; continue; }
      box.innerHTML = files.map(f =>
        '<a href="/gallery/'+encodeURIComponent(slug)+'" target="_blank" title="'+f+' — click for full boxed gallery">'+
        '<img loading="lazy" src="/api/sample/'+encodeURIComponent(slug)+'/'+encodeURIComponent(f)+'"/></a>'
      ).join('');
    }catch(e){ box.innerHTML = '<span class="th-empty">preview failed</span>'; }
  }
}

// v3.0.75.1 (2026-06-01): real-test caught both AJAX calls were broken.
//   - slug_verdict endpoint is POST /api/slug_verdict/{slug} (path param)
//     with JSON body {verdict, note}. NOT query string ?slug=...&verdict=...
//   - exemplar endpoint requires 'img' (not 'image') as the key.
// Bugs found by curl-testing my own page and getting 404.
async function markSlug(slug, verdict){
  const note = verdict==='junk' ? (prompt('Reason (optional, why junk?)', '') || '') : '';
  if(verdict==='junk' && note === null) return;
  try {
    const r = await fetch('/api/slug_verdict/' + encodeURIComponent(slug), {
      method:'POST', credentials:'include',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({verdict: verdict, note: note}),
    });
    if(r.ok){ toast('✓ saved: '+slug+' → '+verdict); loadRounds(); }
    else { const t = await r.text(); toast('err HTTP '+r.status+': '+t.slice(0,80)); }
  } catch(e){ toast('err: '+e) }
}

async function markClass(slug, cls, verdict){
  try {
    const r = await fetch('/api/exemplar/' + encodeURIComponent(cls), {
      method:'POST', credentials:'include',
      headers:{'Content-Type':'application/json'},
      // Server requires 'img' key (NOT 'image'), and 'verdict' in
      // (exemplar|bad|rebox|clear). Img is a path/identifier; we use
      // the slug as a stable identifier for round-level flags.
      body: JSON.stringify({img: 'rounds:'+slug+':'+cls, verdict: verdict}),
    });
    if(r.ok){ toast('✓ saved: '+slug+'/'+cls+' → '+verdict); loadRounds(); }
    else { const t = await r.text(); toast('err HTTP '+r.status+': '+t.slice(0,80)); }
  } catch(e){ toast('err: '+e) }
}

loadRounds();
setInterval(loadRounds, 30000);  // 30s auto-refresh
</script>
</body></html>'''
    return HTMLResponse(html)
