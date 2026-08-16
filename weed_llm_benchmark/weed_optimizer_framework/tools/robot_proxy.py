"""Platform-relayed robot driving (uplink contract P6 / §8.2).

The robot's console lives on a tailnet address (100.x) that only tailnet members
can reach. The platform server IS a tailnet member, so it relays: any signed-in
platform user opens /drive/<robot>/m in a plain browser and the platform proxies
every request to the robot's console — no Tailscale on the viewer's device. Two
gates remain: the platform's own login (global auth middleware) and the robot's
console password page (proxied through untouched).

Targets come from ~/.robot_targets.json:
    {"robot241": {"base": "http://100.97.4.109:5014",
                  "domain": "241_robot", "label": "241 Robot"}}
Only listed robots are reachable — the proxy never fetches arbitrary URLs.

Proxy behavior (the contract the robot side codes against):
- The prefix is stripped: the robot sees /m, /status, … plus
  `X-Forwarded-Prefix: /drive/<robot>` so it can emit prefixed URLs itself.
- 30x `Location: /...` answers are re-prefixed; `Set-Cookie` paths are scoped to
  /drive/<robot> so the console cookie only rides on proxied requests.
- The platform's own session cookie is stripped before forwarding (the robot has
  no business seeing it); every other cookie (console_auth) passes through.
- Responses stream through in chunks — the MJPEG camera works. HTML is passed
  byte-for-byte; dashboard_server exempts /drive/ from its global HTML injection.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

router = APIRouter()
_CTX = {}

_TARGETS_FILE = os.path.expanduser("~/.robot_targets.json")
_DEFAULT_TARGETS = {
    "robot241": {"base": "http://100.97.4.109:5014",
                 "domain": "241_robot", "label": "241 Robot"},
}
_cache = {"mtime": None, "targets": None}

MAX_BODY = 20 * 1024 * 1024
CONNECT_TIMEOUT = 30            # also the max idle gap inside a stream
CHUNK = 8192

# hop-by-hop headers never forwarded in either direction
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}
_PLATFORM_COOKIE = "agentai_session"


def _targets() -> dict:
    try:
        mt = os.path.getmtime(_TARGETS_FILE)
        if _cache["mtime"] != mt:
            _cache["targets"] = json.loads(Path(_TARGETS_FILE).read_text())
            _cache["mtime"] = mt
    except FileNotFoundError:
        return _DEFAULT_TARGETS
    except Exception:
        return _cache["targets"] or _DEFAULT_TARGETS
    return _cache["targets"] or _DEFAULT_TARGETS


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):        # let 30x pass through untouched
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _filtered_cookies(raw: str) -> str:
    keep = []
    for part in (raw or "").split(";"):
        if part.strip().split("=", 1)[0].strip() != _PLATFORM_COOKIE:
            if part.strip():
                keep.append(part.strip())
    return "; ".join(keep)


def _open_upstream(method: str, url: str, body, headers: dict):
    """Blocking: returns the live upstream response (HTTPError IS a response)."""
    req = urllib.request.Request(url, data=(body or None), method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        return _OPENER.open(req, timeout=CONNECT_TIMEOUT)
    except urllib.error.HTTPError as e:
        return e                                 # non-2xx still carries the page


def _offline_page(robot: str, err: str):
    return HTMLResponse(
        '<div style="font-family:system-ui;max-width:560px;margin:80px auto;'
        'text-align:center"><div style="font-size:44px">&#128268;</div>'
        "<h2>Robot &ldquo;%s&rdquo; is offline</h2>"
        "<p style='color:#64748b'>The platform could not reach the robot &mdash; it is "
        "probably powered off or its internet link is down (it self-heals within "
        "~2 minutes when powered). Technical detail: <code>%s</code></p>"
        "<p><a href='javascript:location.reload()'>Try again</a></p></div>"
        % (robot, err[:160]), status_code=502)


@router.get("/api/robot/drive_targets")
def drive_targets(request: Request, domain: str = ""):
    """Which drivable robots belong to this project (for the Drive button)."""
    out = [{"robot": rid, "label": t.get("label") or rid}
           for rid, t in _targets().items()
           if not domain or t.get("domain") == domain]
    return JSONResponse({"ok": True, "targets": out})


@router.get("/drive/{robot}")
def drive_root(robot: str):
    if robot not in _targets():
        return JSONResponse({"ok": False, "error": "unknown robot"}, status_code=404)
    return RedirectResponse("/drive/%s/m" % robot)


@router.api_route("/drive/{robot}/{path:path}",
                  methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def drive_proxy(request: Request, robot: str, path: str):
    t = _targets().get(robot)
    if not t:
        return JSONResponse({"ok": False, "error": "unknown robot"}, status_code=404)
    prefix = "/drive/" + robot
    url = t["base"].rstrip("/") + "/" + path
    if request.url.query:
        url += "?" + request.url.query

    body = await request.body()
    if len(body) > MAX_BODY:
        return JSONResponse({"ok": False, "error": "body too large"}, status_code=413)

    up_headers = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in _HOP:
            continue
        if lk == "cookie":
            v = _filtered_cookies(v)
            if not v:
                continue
        up_headers[k] = v
    up_headers["X-Forwarded-Prefix"] = prefix
    up_headers["X-Forwarded-Proto"] = "https"
    up_headers["X-Forwarded-Host"] = request.headers.get("host", "")
    client_ip = getattr(request.client, "host", "") or ""
    if client_ip:
        up_headers["X-Forwarded-For"] = client_ip

    t0 = time.time()
    try:
        up = await run_in_threadpool(_open_upstream, request.method, url, body, up_headers)
    except Exception as e:
        _CTX.get("log") and _CTX["log"].info(
            f"[drive] {robot} unreachable after {time.time()-t0:.1f}s: {e}")
        return _offline_page(robot, "%s: %s" % (type(e).__name__, e))

    status = getattr(up, "status", None) or getattr(up, "code", 502)
    down_headers = []
    for k, v in (up.headers.items() if up.headers else []):
        lk = k.lower()
        if lk in _HOP:
            continue
        if lk == "location" and v.startswith("/"):
            v = prefix + v                        # robot redirects stay inside /drive
        if lk == "set-cookie":
            # scope the console cookie to this robot's proxied path
            parts = [p for p in v.split(";") if p.strip().lower()[:5] != "path="]
            v = "; ".join(p.strip() for p in parts) + "; Path=" + prefix
        down_headers.append((k, v))

    # read1() returns whatever bytes are available (<= CHUNK) instead of
    # blocking until a full CHUNK accumulates — read() throttled slow streams
    # (an MJPEG frame would sit in the buffer until 8 KB piled up).
    _read = getattr(up, "read1", None) or up.read

    def _stream():
        try:
            while True:
                chunk = _read(CHUNK)
                if not chunk:
                    break
                yield chunk
        except Exception:
            pass                                  # viewer left / robot went away
        finally:
            try:
                up.close()
            except Exception:
                pass

    media = up.headers.get("Content-Type") if up.headers else None
    return StreamingResponse(_stream(), status_code=status,
                             headers=dict(down_headers), media_type=media)


def mount(app, ctx: dict):
    _CTX.update(ctx)
    if not os.path.exists(_TARGETS_FILE):
        try:
            Path(_TARGETS_FILE).write_text(json.dumps(_DEFAULT_TARGETS, indent=1))
            os.chmod(_TARGETS_FILE, 0o600)
        except Exception:
            pass
    app.include_router(router)
