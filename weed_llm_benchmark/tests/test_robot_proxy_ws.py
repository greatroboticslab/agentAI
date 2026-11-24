#!/usr/bin/env python3
"""robot_proxy: the /drive/<robot>/* WebSocket relay and the driving allow-list.

A robot console that talks over /ws could not be driven through the platform: the HTTP relay drops the Upgrade
header. And any account the platform admits (any Google account, any API key) could open any robot's console.
These checks run the proxy router in a bare FastAPI app against a local WebSocket server and a local HTTP server
standing in for the robot. No dashboard, no Mongo, no network beyond 127.0.0.1.

Run:  python -m pytest tests/test_robot_proxy_ws.py     (or) python tests/test_robot_proxy_ws.py
"""
import asyncio
import http.server
import json
import pathlib
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

from weed_optimizer_framework.tools import robot_proxy as RP  # noqa: E402

_fails = []
SEEN = {"ws": [], "closed": [], "http": []}


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


async def _ws_handler(conn):
    SEEN["ws"].append({"path": conn.request.path, "headers": dict(conn.request.headers.raw_items()),
                       "proto": conn.subprotocol})
    try:
        async for m in conn:
            if m == "close-me":
                await conn.close(code=4001, reason="bye")
                return
            if m == "bin?":
                await conn.send(b"\x00\x01robot")
                continue
            await conn.send(("echo:" + m) if isinstance(m, str) else (b"echo:" + m))
    finally:
        SEEN["closed"].append(time.time())


def _start_ws():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    box = {}

    async def main():
        # picks r1console when the client offers it, and lets a client that offers none in
        pick = lambda conn, offered: "r1console" if "r1console" in offered else None  # noqa: E731
        async with serve(_ws_handler, "127.0.0.1", 0, select_subprotocol=pick) as srv:
            box["port"] = srv.sockets[0].getsockname()[1]
            ready.set()
            await asyncio.Future()

    threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True).start()
    ready.wait(10)
    return box["port"]


class _Http(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        SEEN["http"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
        self.send_response(302)
        self.send_header("Location", "/m?x=1")
        self.send_header("Set-Cookie", "console_auth=abc; Path=/; HttpOnly; Secure; SameSite=Strict")
        self.send_header("Set-Cookie", "remote_auth=def; Path=/; Secure")
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def _start_http():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Http)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def _session_user(conn):
    c = conn.cookies.get("agentai_session")
    return {"uid": c + "@lab.example", "email": c + "@lab.example"} if c else None


def _api_key(conn):
    return {"ak_r1": {"label": "unitree_r1"}, "ak_241": {"label": "robot241"}}.get(conn.headers.get("x-api-key"))


def _refused(client, path, **kw):
    try:
        with client.websocket_connect(path, **kw) as ws:
            ws.send_text("hi")
            ws.receive_text()
        return False
    except Exception:
        return True


def main():
    wsport, httpport = _start_ws(), _start_http()
    tmp = tempfile.mkdtemp()
    RP._TARGETS_FILE = tmp + "/targets.json"
    RP._ACCESS_FILE = tmp + "/access.json"
    RP._cache.update(mtime=None, targets=None)
    pathlib.Path(RP._TARGETS_FILE).write_text(json.dumps({
        "r1": {"base": "http://127.0.0.1:%d" % wsport, "domain": "unitree_r1", "label": "Unitree R1"},
        "h1": {"base": "http://127.0.0.1:%d" % httpport, "domain": "unitree_r1", "label": "HTTP robot"},
        "gone": {"base": "http://127.0.0.1:9", "domain": "x", "label": "offline"}}))
    future = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
    past = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    access = {"users": ["Harry@Lab.example"],
              "keys": {"unitree_r1": {"robots": ["r1", "h1"], "until_utc": future},
                       "robot241": {"robots": ["r1"], "until_utc": past}}}
    pathlib.Path(RP._ACCESS_FILE).write_text(json.dumps(access))

    app = FastAPI()
    RP.mount(app, {"session_user": _session_user, "api_key": _api_key})
    client = TestClient(app)
    harry = {"cookie": "agentai_session=harry; console_auth=abc", "origin": "https://platform.example"}

    # --- the WebSocket relay
    with client.websocket_connect("/drive/r1/ws?client=7", headers=harry, subprotocols=["r1console"]) as ws:
        ws.send_text("hello")
        ck("text frames go both ways", ws.receive_text() == "echo:hello")
        ws.send_bytes(b"\x01\x02")
        ck("binary frames go both ways", ws.receive_bytes() == b"echo:\x01\x02")
        ws.send_text("bin?")
        ck("binary from the robot reaches the viewer", ws.receive_bytes() == b"\x00\x01robot")
        ck("the subprotocol the robot chose reaches the viewer", ws.accepted_subprotocol == "r1console")
    seen = SEEN["ws"][-1]
    h = {k.lower(): v for k, v in seen["headers"].items()}
    ck("the robot sees the path without the prefix, query kept", seen["path"] == "/ws?client=7")
    ck("the platform's session cookie is stripped, the console cookie kept",
       "agentai_session" not in h.get("cookie", "") and "console_auth=abc" in h.get("cookie", ""))
    ck("Origin is passed as sent", h.get("origin") == "https://platform.example")
    ck("X-Forwarded-Prefix / Proto / Host / For are set",
       h.get("x-forwarded-prefix") == "/drive/r1" and h.get("x-forwarded-proto") == "https"
       and h.get("x-forwarded-host") == "testserver" and h.get("x-forwarded-for") == "testclient")
    t_end = time.time() + 5
    while not SEEN["closed"] and time.time() < t_end:
        time.sleep(0.05)
    ck("the viewer closing closes the robot side", len(SEEN["closed"]) >= 1)

    with client.websocket_connect("/drive/r1/ws", headers=harry) as ws:
        ws.send_text("close-me")
        try:
            ws.receive_text()
            code = None
        except Exception as e:
            code = getattr(e, "code", None)
        ck("the robot closing closes the viewer side, with the robot's code", code == 4001)

    # --- who may open it
    n0 = len(SEEN["ws"])
    ck("no login: refused", _refused(client, "/drive/r1/ws"))
    ck("a signed-in account not on the list: refused",
       _refused(client, "/drive/r1/ws", headers={"cookie": "agentai_session=stranger"}))
    ck("unknown robot: refused", _refused(client, "/drive/nope/ws", headers=harry))
    ck("an expired key grant: refused", _refused(client, "/drive/r1/ws", headers={"x-api-key": "ak_241"}))
    ck("a key granted r1 used on another robot: refused",
       _refused(client, "/drive/gone/ws", headers={"x-api-key": "ak_r1"}))
    ck("no refused handshake reached the robot", len(SEEN["ws"]) == n0)
    with client.websocket_connect("/drive/r1/ws", headers={"x-api-key": "ak_r1"}) as ws:
        ws.send_text("k")
        ck("a live key grant for this robot: allowed", ws.receive_text() == "echo:k")
    ck("the API key is not forwarded to the robot", "x-api-key" not in {k.lower() for k in SEEN["ws"][-1]["headers"]})
    ck("an offline robot: accepted, then closed 1011",
       _refused(client, "/drive/gone/ws", headers=harry))
    pathlib.Path(RP._ACCESS_FILE).write_text("{not json")
    ck("an unreadable access file lets nobody drive", _refused(client, "/drive/r1/ws", headers=harry))
    pathlib.Path(RP._ACCESS_FILE).write_text(json.dumps(access))

    # --- the viewer's address: behind the local proxy, the last X-Forwarded-For entry (the one the proxy wrote)
    behind = TestClient(app, client=("127.0.0.1", 50123))
    with behind.websocket_connect("/drive/r1/ws", headers=dict(harry, **{"x-forwarded-for": "1.2.3.4, 100.64.0.9"})) as ws:
        ws.send_text("a")
        ws.receive_text()
    ck("behind the proxy: a forged first X-Forwarded-For entry is not passed as the viewer",
       {k.lower(): v for k, v in SEEN["ws"][-1]["headers"].items()}.get("x-forwarded-for") == "100.64.0.9")
    behind.get("/drive/h1/login", headers=dict(harry, **{"x-forwarded-for": "1.2.3.4, 100.64.0.9"}),
               follow_redirects=False)
    ck("behind the proxy, HTTP: the same", SEEN["http"][-1]["headers"].get("x-forwarded-for") == "100.64.0.9")
    client.get("/drive/h1/login", headers=dict(harry, **{"x-forwarded-for": "1.2.3.4"}), follow_redirects=False)
    ck("not behind the proxy: a sent X-Forwarded-For is ignored, the TCP peer is passed",
       SEEN["http"][-1]["headers"].get("x-forwarded-for") == "testclient")

    # --- the HTTP relay
    r = client.get("/drive/h1/login", headers=harry, follow_redirects=False)
    ck("HTTP: a listed user is relayed", r.status_code == 302)
    ck("HTTP: a relative Location is re-prefixed", r.headers.get("location") == "/drive/h1/m?x=1")
    cookies = r.headers.get_list("set-cookie")
    ck("HTTP: two Set-Cookie headers stay two", len(cookies) == 2)
    ck("HTTP: Set-Cookie Path is scoped, Secure/HttpOnly/SameSite kept",
       any(c.startswith("console_auth=abc") and "HttpOnly" in c and "Secure" in c and "SameSite=Strict" in c
           and c.endswith("Path=/drive/h1") and "Path=/;" not in c for c in cookies))
    hh = SEEN["http"][-1]["headers"]
    ck("HTTP: session cookie and API key stripped",
       "agentai_session" not in hh.get("cookie", "") and "x-api-key" not in hh)
    r = client.get("/drive/h1/login", headers={"cookie": "agentai_session=stranger"}, follow_redirects=False)
    ck("HTTP: an account not on the list gets 403", r.status_code == 403)
    r = client.get("/api/robot/drive_targets?domain=unitree_r1", headers={"cookie": "agentai_session=stranger"})
    ck("drive_targets lists nothing for an account not on the list", r.json()["targets"] == [])
    r = client.get("/api/robot/drive_targets?domain=unitree_r1", headers=harry)
    ck("drive_targets lists the project's robots for a listed user",
       sorted(t["robot"] for t in r.json()["targets"]) == ["h1", "r1"])
    return 0 if not _fails else 1


def test_robot_proxy_ws():
    assert main() == 0, _fails


if __name__ == "__main__":
    rc = main()
    print("%d check(s) failed" % len(_fails) if _fails else "all checks passed")
    sys.exit(rc)
