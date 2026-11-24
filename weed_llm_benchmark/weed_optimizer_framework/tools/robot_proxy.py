"""Platform-relayed robot driving (uplink contract P6 / §8.2).

The robot's console lives on a tailnet address (100.x) that only tailnet members
can reach. The platform server IS a tailnet member, so it relays: an approved
platform user opens /drive/<robot>/m in a plain browser and the platform proxies
every request to the robot's console — no Tailscale on the viewer's device. Three
gates: the platform's own login, the driving allow-list below, and the robot's
console password page (proxied through untouched).

Targets come from ~/.robot_targets.json:
    {"robot241": {"base": "http://<tailnet address>:5014",
                  "domain": "241_robot", "label": "241 Robot"}}
Only listed robots are reachable — the proxy never fetches arbitrary URLs.

Who may drive comes from ~/.drive_access.json (lab-only, never committed):
    {"users": ["<email>", ...],
     "keys": {"<api key label>": {"robots": ["r1"], "until_utc": "2026-11-25T12:00:00Z"}}}
A signed-in user drives every listed robot when their email (or uid) is in
"users". An API key drives only the robots its grant names, and only until the
grant's until_utc; a grant without a valid until_utc grants nothing. A missing or
unreadable file lets nobody drive. Signing in is not enough: the platform admits
any Google account, and an API key is issued for uploading, not for moving a
robot.

Proxy behavior (the contract the robot side codes against):
- The prefix is stripped: the robot sees /m, /status, … plus
  `X-Forwarded-Prefix: /drive/<robot>` so it can emit prefixed URLs itself;
  also X-Forwarded-Proto: https, X-Forwarded-Host (the Host the viewer's browser
  used) and X-Forwarded-For (the viewer's address: the TCP peer, or behind the local
  Funnel proxy the last X-Forwarded-For entry, the one that proxy wrote).
- 30x `Location: /...` answers are re-prefixed; `Set-Cookie` paths are scoped to
  /drive/<robot> so the console cookie only rides on proxied requests (every
  other attribute — Secure, HttpOnly, SameSite, Max-Age — passes unchanged, and
  several Set-Cookie headers stay several).
- The platform's own credentials (its session cookie, an X-API-Key) are stripped
  before forwarding; every other cookie (console_auth) passes through.
- Responses stream through in chunks — the MJPEG camera works. CONNECT_TIMEOUT
  bounds the connect and each read, not the whole response. HTML is passed
  byte-for-byte; dashboard_server exempts /drive/ from its global HTML injection.
- WebSocket: an Upgrade request to /drive/<robot>/<path> is relayed to
  ws://<base>/<path> with the same header handling (Origin passed as sent).
  Text and binary frames go both ways; when either side closes the other is
  closed. No idle limit: a quiet socket stays open, and a peer that stops
  answering protocol pings for WS_PING_TIMEOUT seconds is closed. The handshake
  is checked here because the platform's login middleware handles HTTP only.
"""
import asyncio
import calendar
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

router = APIRouter()
_CTX = {}

_TARGETS_FILE = os.path.expanduser("~/.robot_targets.json")
_ACCESS_FILE = os.path.expanduser("~/.drive_access.json")
_DEFAULT_TARGETS = {}
_cache = {"mtime": None, "targets": None}

MAX_BODY = 20 * 1024 * 1024
CONNECT_TIMEOUT = 30            # the connect, and each read inside a stream
CHUNK = 8192
WS_PING_INTERVAL = 20           # protocol pings to the robot, seconds
WS_PING_TIMEOUT = 60            # a robot that answers none for this long is gone
WS_MAX_MESSAGE = 16 * 1024 * 1024

# hop-by-hop headers never forwarded in either direction
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}
# the WebSocket client library writes these itself
_WS_OWN = {"sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
           "sec-websocket-protocol", "sec-websocket-accept", "origin"}
# the platform's own credentials; the robot has no business seeing them
_PLATFORM_COOKIE = "agentai_session"
_PLATFORM_HEADERS = {"x-api-key"}
# set by dashboard_server's proxy chain; replaced by ours
_FORWARDED = {"x-forwarded-prefix", "x-forwarded-proto", "x-forwarded-host", "x-forwarded-for"}


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


def _utc_ts(s):
    try:
        return float(calendar.timegm(time.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def _access() -> dict:
    try:
        a = json.loads(Path(_ACCESS_FILE).read_text())
        return a if isinstance(a, dict) else {}
    except Exception:
        return {}


def _driver(conn, robot: str):
    """Who is asking to drive `robot` through this request or WebSocket handshake:
    'user:<email>' or 'key:<label>' when allowed, else None."""
    acc = _access()
    users = {str(u).strip().lower() for u in (acc.get("users") or []) if str(u).strip()}
    su = _CTX.get("session_user")
    sess = None
    try:
        sess = su(conn) if su else None
    except Exception:
        sess = None
    if sess:
        for who in (sess.get("email"), sess.get("uid")):
            if who and str(who).strip().lower() in users:
                return "user:" + str(who)
    ak = _CTX.get("api_key")
    rec = None
    try:
        rec = ak(conn) if ak else None
    except Exception:
        rec = None
    if rec:
        g = (acc.get("keys") or {}).get(rec.get("label") or "")
        if isinstance(g, dict) and robot in (g.get("robots") or []):
            until = _utc_ts(g.get("until_utc"))
            if until is not None and time.time() <= until:
                return "key:" + str(rec.get("label"))
    return None


def _log(msg):
    lg = _CTX.get("log")
    if lg:
        lg.info(msg)


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


def _upstream_headers(conn, prefix: str, skip=frozenset()) -> dict:
    """The viewer's headers as the robot should see them (shared by HTTP and WebSocket)."""
    out = {}
    for k, v in conn.headers.items():
        lk = k.lower()
        if lk in _HOP or lk in _PLATFORM_HEADERS or lk in _FORWARDED or lk in skip:
            continue
        if lk == "cookie":
            v = _filtered_cookies(v)
            if not v:
                continue
        out[k] = v
    out["X-Forwarded-Prefix"] = prefix
    out["X-Forwarded-Proto"] = "https"
    out["X-Forwarded-Host"] = conn.headers.get("host", "")
    peer = getattr(conn.client, "host", "") or ""
    # Behind Tailscale Funnel the TCP peer is the local proxy, and the viewer's address
    # is the LAST entry of X-Forwarded-For: that one the proxy wrote, while anything
    # before it came from the viewer and can be forged (the robot rate-limits its
    # password page by this address).
    if peer in ("127.0.0.1", "::1") and conn.headers.get("x-forwarded-for"):
        last = conn.headers["x-forwarded-for"].split(",")[-1].strip()
        if last:
            peer = last
    if peer:
        out["X-Forwarded-For"] = peer
    return out


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


def _denied_page(robot: str):
    return HTMLResponse(
        '<div style="font-family:system-ui;max-width:560px;margin:80px auto;'
        'text-align:center"><h2>Driving &ldquo;%s&rdquo; is limited to approved accounts</h2>'
        "<p style='color:#64748b'>You are signed in, but this account is not on the "
        "platform's driving list. Ask the platform admin to add it.</p></div>" % robot,
        status_code=403)


@router.get("/api/robot/drive_targets")
def drive_targets(request: Request, domain: str = ""):
    """Which robots of this project the caller may drive (for the Drive button)."""
    out = [{"robot": rid, "label": t.get("label") or rid}
           for rid, t in _targets().items()
           if (not domain or t.get("domain") == domain) and _driver(request, rid)]
    return JSONResponse({"ok": True, "targets": out})


@router.get("/drive/{robot}")
def drive_root(request: Request, robot: str):
    if robot not in _targets():
        return JSONResponse({"ok": False, "error": "unknown robot"}, status_code=404)
    if not _driver(request, robot):
        return _denied_page(robot)
    return RedirectResponse("/drive/%s/m" % robot)


@router.api_route("/drive/{robot}/{path:path}",
                  methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def drive_proxy(request: Request, robot: str, path: str):
    t = _targets().get(robot)
    if not t:
        return JSONResponse({"ok": False, "error": "unknown robot"}, status_code=404)
    if not _driver(request, robot):
        return _denied_page(robot)
    prefix = "/drive/" + robot
    url = t["base"].rstrip("/") + "/" + path
    if request.url.query:
        url += "?" + request.url.query

    body = await request.body()
    if len(body) > MAX_BODY:
        return JSONResponse({"ok": False, "error": "body too large"}, status_code=413)

    up_headers = _upstream_headers(request, prefix)
    t0 = time.time()
    try:
        up = await run_in_threadpool(_open_upstream, request.method, url, body, up_headers)
    except Exception as e:
        _log(f"[drive] {robot} unreachable after {time.time()-t0:.1f}s: {e}")
        return _offline_page(robot, "%s: %s" % (type(e).__name__, e))

    status = getattr(up, "status", None) or getattr(up, "code", 502)
    media = up.headers.get("Content-Type") if up.headers else None
    down_headers = []
    for k, v in (up.headers.items() if up.headers else []):
        lk = k.lower()
        if lk in _HOP or lk == "content-type":
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

    resp = StreamingResponse(_stream(), status_code=status, media_type=media)
    for k, v in down_headers:                     # append, so repeated Set-Cookie survive
        resp.headers.append(k, v)
    return resp


@router.websocket("/drive/{robot}/{path:path}")
async def drive_ws(websocket: WebSocket, robot: str, path: str):
    t = _targets().get(robot)
    who = _driver(websocket, robot) if t else None
    if not t or not who:
        await websocket.close(code=1008)          # before accept: the handshake gets a 403
        return
    from websockets.asyncio.client import connect

    prefix = "/drive/" + robot
    base = t["base"].rstrip("/")
    url = ("wss://" + base[8:]) if base.startswith("https://") else ("ws://" + base.split("://", 1)[-1])
    url += "/" + path
    if websocket.url.query:
        url += "?" + websocket.url.query
    protos = [p.strip() for p in (websocket.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()]
    t0 = time.time()
    try:
        up = await connect(url, additional_headers=_upstream_headers(websocket, prefix, _WS_OWN),
                           origin=websocket.headers.get("origin") or None,
                           subprotocols=protos or None, open_timeout=CONNECT_TIMEOUT,
                           ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT,
                           max_size=WS_MAX_MESSAGE, compression=None)
    except Exception as e:
        _log(f"[drive] {robot} ws /{path} unreachable after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
        await websocket.accept()
        await websocket.close(code=1011, reason="robot unreachable")
        return
    await websocket.accept(subprotocol=up.subprotocol)
    n = {"up": 0, "down": 0}

    async def viewer_to_robot():
        while True:
            m = await websocket.receive()
            if m["type"] == "websocket.disconnect":
                return
            if m.get("text") is not None:
                await up.send(m["text"])
            elif m.get("bytes") is not None:
                await up.send(m["bytes"])
            n["up"] += 1

    async def robot_to_viewer():
        async for m in up:
            if isinstance(m, str):
                await websocket.send_text(m)
            else:
                await websocket.send_bytes(m)
            n["down"] += 1

    tasks = [asyncio.ensure_future(viewer_to_robot()), asyncio.ensure_future(robot_to_viewer())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for tk in tasks:
            tk.cancel()
        # the robot side is closed first, and dropped outright if the closing handshake
        # cannot finish (this handler cancelled, the robot silent): a robot must never be
        # left holding a session the viewer has left
        try:
            await asyncio.wait_for(up.close(), 5)
        except asyncio.CancelledError:
            up.transport.abort()
            raise
        except Exception:
            up.transport.abort()
        await asyncio.gather(*tasks, return_exceptions=True)
        code = up.close_code if up.close_code and 1000 <= up.close_code <= 4999 and up.close_code not in (1005, 1006) else 1000
        try:
            await websocket.close(code=code)
        except Exception:
            pass                                  # the viewer already left
        _log(f"[drive] {robot} ws /{path} by {who} closed after {time.time()-t0:.0f}s "
             f"({n['up']} up, {n['down']} down, robot code {up.close_code})")


def mount(app, ctx: dict):
    """ctx: log, and the platform's session_user(conn) and api_key(conn) checks."""
    _CTX.update(ctx)
    app.include_router(router)
