#!/usr/bin/env python3
"""
Humanoid-robot laptop auto-upload client (v3.0.135).

Runs on a laptop next to the robot and pushes datasets to the AgentAI dashboard
via its public HTTP API (POST /api/dataset/upload). Zero extra dependencies —
standard library only. Two modes:

  one-shot : zip a folder (or send an existing .zip) and upload once.
  --watch  : poll a folder; every NEW subfolder that appears is zipped and
             uploaded automatically (the "agent on the laptop" the professor
             asked for — humanoid data flows in without a human clicking).

The upload is attributed to --as-user (sent as the X-User header) so the
dashboard's /users page shows who/what produced the data. Auth uses the
dashboard's HTTP Basic credentials (the laptop is a trusted device).

Examples
--------
  # one-shot: upload a folder of images as one dataset
  python humanoid_auto_upload.py --name "kitchen-run-12" --path ./captures/run12

  # watch a capture root; auto-upload each new run subfolder
  python humanoid_auto_upload.py --name "kitchen" --path ./captures --watch

Config via flags or env: AGENTAI_URL, AGENTAI_USER, AGENTAI_PASS, AGENTAI_AS_USER.
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".txt", ".yaml", ".yml")


def make_zip(path: str) -> bytes:
    """Return zip bytes for a folder of images (+ optional labels/yaml), or pass
    through an existing .zip."""
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        with open(path, "rb") as f:
            return f.read()
    if not os.path.isdir(path):
        raise SystemExit(f"path not found (need a folder or .zip): {path}")
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(_EXTS):
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.relpath(fp, path))
                    n += 1
    if n == 0:
        raise SystemExit(f"no images found under {path}")
    return buf.getvalue()


def upload(base, domain, name, data, user, password, as_user):
    url = base.rstrip("/") + "/api/dataset/upload?" + urllib.parse.urlencode(
        {"domain": domain, "name": name})
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/zip")
    if user:
        tok = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    if as_user:
        req.add_header("X-User", as_user)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description="Humanoid robot auto-upload client")
    ap.add_argument("--url", default=os.environ.get(
        "AGENTAI_URL", "https://lab-b660m-c.tailfa6424.ts.net"))
    ap.add_argument("--domain", default="humanoid_robot")
    ap.add_argument("--name", required=True, help="dataset name (watch mode appends the subfolder)")
    ap.add_argument("--path", required=True, help="folder of images / capture root / a .zip")
    ap.add_argument("--user", default=os.environ.get("AGENTAI_USER", "1"))
    ap.add_argument("--password", default=os.environ.get("AGENTAI_PASS", "1"))
    ap.add_argument("--as-user", default=os.environ.get("AGENTAI_AS_USER", "humanoid-lab-agent"),
                    help="attribution identity shown in /users")
    ap.add_argument("--watch", action="store_true",
                    help="poll --path; auto-upload each NEW subfolder")
    ap.add_argument("--interval", type=int, default=60, help="watch poll seconds")
    a = ap.parse_args()

    if not a.watch:
        res = upload(a.url, a.domain, a.name, make_zip(a.path),
                     a.user, a.password, a.as_user)
        print(json.dumps(res, indent=2))
        return

    seen = set()
    print(f"[watch] {a.path} every {a.interval}s → {a.url} (domain={a.domain})")
    while True:
        try:
            for d in sorted(os.listdir(a.path)):
                sub = os.path.join(a.path, d)
                if os.path.isdir(sub) and sub not in seen:
                    seen.add(sub)
                    try:
                        res = upload(a.url, a.domain, f"{a.name}-{d}",
                                     make_zip(sub), a.user, a.password, a.as_user)
                        print(f"[uploaded] {d}: {res.get('images')} imgs → {res.get('slug')}")
                    except SystemExit as e:
                        print(f"[skip] {d}: {e}")
                    except Exception as e:
                        seen.discard(sub)  # retry next round
                        print(f"[error] {d}: {e}")
        except Exception as e:
            print(f"[watch error] {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
