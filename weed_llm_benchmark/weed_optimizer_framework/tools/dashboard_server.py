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
from fastapi import FastAPI, HTTPException, Response, Body
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
# Allow GitHub Pages and any browser to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
    """Hub page — links to all dashboard tools, old + new.
    v3.0.43.1: previously redirected to the old static index.html which had
    no links to /classes or /slugs. Users (incl. the project owner) couldn't
    discover them. This hub fixes discoverability."""
    return HTMLResponse('''<!DOCTYPE html><html lang="zh"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>autonomous weed detection — hub</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif;
         max-width: 900px; margin: 40px auto; padding: 1rem; color: #1a1a1d;
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
</style>
</head><body>
<h1>🌱 Autonomous Weed Detection — dashboard hub</h1>
<div class="sub">
  Track A (data harvest) + Track B (training) — pick a tool below.
</div>

<div class="live-banner idle" id="live-banner">
  <span class="pulse"></span>
  <span id="live-banner-text">loading status…</span>
</div>

<script>
  // Refresh live banner with current cluster status
  async function updateBanner() {
    try {
      const r = await fetch('/api/cluster_status');
      const d = await r.json();
      const jobs = d.jobs || [];
      const reg = d.registry || {};
      const agentRunning = jobs.find(j =>
        ['dl_known','brain_hrv','topic_bf','smoke','lora_'].some(p => (j.name||'').startsWith(p))
        && j.state === 'RUNNING');
      const banner = document.getElementById('live-banner');
      const txt = document.getElementById('live-banner-text');
      if (agentRunning) {
        banner.classList.remove('idle');
        txt.innerHTML = `🤖 <strong>${agentRunning.name}</strong> RUNNING (${agentRunning.time}) — `
          + `<a href="/control">monitor 📺</a> · `
          + `registry ${reg.n_downloaded||'?'}/${reg.n_slugs||'?'} slugs, `
          + `${(reg.total_imgs||0).toLocaleString()} imgs · `
          + `${d.n_topic_overrides||0} topic overrides`;
      } else {
        banner.classList.add('idle');
        txt.innerHTML = `💤 idle — no agent jobs · registry ${reg.n_downloaded||'?'}/${reg.n_slugs||'?'} slugs, `
          + `${(reg.total_imgs||0).toLocaleString()} imgs · `
          + `<a href="/control">trigger agent ▶️</a>`;
      }
    } catch(e) {
      document.getElementById('live-banner-text').innerHTML =
        'error fetching status: ' + e;
    }
  }
  updateBanner();
  setInterval(updateBanner, 7000);
</script>

<div class="section-h">🎛️ Cluster control (v3.0.43)</div>
<div class="grid">
  <a class="card new" href="/morning_report" style="border-left-color:#2a7;">
    <div class="icon">☀️</div>
    <div class="title">/morning_report — overnight 总结</div>
    <div class="desc">一页看 overnight progress:新 slugs、action history、agent jobs 最新输出。
    早上一杯咖啡时间看完。</div>
  </a>
  <a class="card new" href="/control" style="border-left-color:#c70;">
    <div class="icon">🎛️</div>
    <div class="title">/control — 控制台</div>
    <div class="desc">实时 squeue / ollama / registry 状态 + 一键
    重启 dashboard / Brain harvest / topic backfill /缓存清理。
    <strong>你不再需要叫我做这些。</strong></div>
  </a>
</div>

<div class="section-h">🆕 Human-in-the-loop verification (v3.0.42–43)</div>
<div class="grid">
  <a class="card new" href="/classes">
    <div class="icon">📋</div>
    <div class="title">/classes — 类级人工审计</div>
    <div class="desc">每个类(CWD12 + 348 总类)逐张点 ✓ 榜样 / ✗ 错标 / 🔄 bbox 修。
    7 个 filter tab + 搜索框。三源混排:bank/flux/真实 reg。</div>
  </a>
  <a class="card new" href="/slugs">
    <div class="icon">📦</div>
    <div class="title">/slugs — slug 级清理</div>
    <div class="desc">80 个数据集 slug 列表,批量 ✓ keep / ✗ junk / 🤔 unsure。
    ✗ 后自动从 /classes 隐藏 — 不用逐张点垃圾类。</div>
  </a>
  <a class="card new" href="/classes/Goosegrass">
    <div class="icon">🌾</div>
    <div class="title">直接进 Goosegrass</div>
    <div class="desc">最难类 — 看 bank + flux + holdout 真图(红框 bbox 标本类)
    混排,审定后写到 exemplar 集供训练。</div>
  </a>
</div>

<div class="section-h">📊 旧版静态 dashboard</div>
<div class="grid">
  <a class="card old" href="/dashboard/index.html">
    <div class="icon">🏠</div>
    <div class="title">主页(stats)</div>
    <div class="desc">总览:n_datasets / n_imgs / latest mAP. 由 dashboard_generator
    定期 regenerate(可能滞后几天)。</div>
  </a>
  <a class="card old" href="/dashboard/datasets.html">
    <div class="icon">📑</div>
    <div class="title">Datasets 列表</div>
    <div class="desc">所有 slug 的元数据列表(只读)。新功能 /slugs 是它的可写版。</div>
  </a>
  <a class="card old" href="/dashboard/categories.html">
    <div class="icon">🏷️</div>
    <div class="title">Categories</div>
    <div class="desc">按 crop/topic 分桶统计。</div>
  </a>
  <a class="card old" href="/dashboard/progress.html">
    <div class="icon">📈</div>
    <div class="title">Progress</div>
    <div class="desc">训练历史 / mAP 时序。</div>
  </a>
  <a class="card old" href="/audit">
    <div class="icon">🔍</div>
    <div class="title">/audit(旧)</div>
    <div class="desc">v3.0.41 之前的 per-class 大图浏览,被 /classes 取代。</div>
  </a>
</div>

<div class="section-h">🔌 API endpoints (JSON)</div>
<div class="grid">
  <a class="card api" href="/api/exemplars_export">
    <div class="icon">📥</div>
    <div class="title">/api/exemplars_export</div>
    <div class="desc">所有 ✓ 榜样图导出 manifest。供 LoRA / training agent 读取。</div>
  </a>
  <a class="card api" href="/api/slug_verdicts">
    <div class="icon">📥</div>
    <div class="title">/api/slug_verdicts</div>
    <div class="desc">所有 slug ✓/✗/🤔 判定 JSON。</div>
  </a>
  <a class="card api" href="/api/state">
    <div class="icon">📊</div>
    <div class="title">/api/state</div>
    <div class="desc">整体 registry 状态(60s cache)。</div>
  </a>
  <a class="card api" href="/api/refresh_registry">
    <div class="icon">♻️</div>
    <div class="title">/api/refresh_registry</div>
    <div class="desc">harvest 后强制清缓存。</div>
  </a>
  <a class="card api" href="/healthz">
    <div class="icon">❤️</div>
    <div class="title">/healthz</div>
    <div class="desc">liveness probe.</div>
  </a>
</div>

<div class="footer">
  v3.0.43.1 hub · server pid自启 · for direct 路径:
  /classes · /slugs · /classes/{species} · /dashboard/{page}.html
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
        raise HTTPException(500, "render failed")
    return FileResponse(cache_p, media_type="image/jpeg")


@app.get("/api/img/{slug}/{filename}")
def api_img(slug: str, filename: str):
    """Original full-resolution image (no bbox overlay). For zoom-in."""
    if not re.match(r"^[A-Za-z0-9_.-]+$", slug):
        raise HTTPException(400, "bad slug")
    if not re.match(r"^[A-Za-z0-9_. -]+\.(jpg|jpeg|png|JPG|JPEG|PNG)$", filename):
        raise HTTPException(400, "bad filename")
    found = find_image_in_slug(slug, filename)
    if not found:
        raise HTTPException(404)
    img_path, _ = found
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
<header>
  <h1>📂 {slug}</h1>
  <div class="meta">
    {total} unique images · page {page}/{n_pages} ({per_page}/page)
    · annotation: <code>{annot}</code>
    · source: <code>{src}</code>
    · <a href="/dashboard/datasets.html">← back to all datasets</a>
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

def _canon_class(raw: str) -> str:
    """Normalize species name from registry to canonical form.
    Matches CWD12 case+punctuation-insensitive; else PascalCase the input."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    alnum = _re_canon.sub(r'[^A-Za-z0-9]', '', raw).lower()
    if not alnum:
        return ""
    for c12 in _CWD12:
        if _re_canon.sub(r'[^A-Za-z0-9]', '', c12).lower() == alnum:
            return c12
    parts = _re_canon.split(r'[^A-Za-z0-9]+', raw)
    return "".join(p[:1].upper() + p[1:].lower() for p in parts if p)


_registry_index_cache: dict = {"mtime": 0.0, "index": {}, "empty_slugs": []}

def _load_registry_index() -> dict:
    """Returns {canon_class_name: [(slug, class_id, raw_name), ...]}.
    Side-effect populates _registry_index_cache['empty_slugs'] for slugs
    with local data but missing class_names — surfaces metadata-gap."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        mtime = REGISTRY_PATH.stat().st_mtime
    except Exception:
        mtime = 0.0
    if _registry_index_cache["mtime"] == mtime:
        return _registry_index_cache["index"]
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
    except Exception:
        return {}
    idx: dict = {}
    empty: list = []
    for slug, info in (reg.get("datasets") or {}).items():
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
            idx.setdefault(canon, []).append((slug, cid, raw))
    _registry_index_cache.update({"mtime": mtime, "index": idx, "empty_slugs": empty})
    return idx


def _registry_empty_slugs() -> list:
    _load_registry_index()
    return list(_registry_index_cache.get("empty_slugs", []))


_pool_cache_dir = REPO / "results" / "framework" / "cache" / "class_pool"
_pool_cache_dir.mkdir(parents=True, exist_ok=True)


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
                         include_junk: bool = False) -> list:
    """For each slug containing `cls`, sample up to per_slug_cap images whose
    label has a bbox of that class_id. Cached on disk by registry mtime.

    v3.0.43: by default exclude slugs marked '✗ junk' via /slugs UI.
    Pass include_junk=True to override."""
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
        try:
            for ldir in label_dirs:
                if n_added >= per_slug_cap:
                    break
                try:
                    lbls = sorted(ldir.glob("*.txt"))
                except Exception:
                    continue
                for lbl in lbls:
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
    if not _re_audit.match(r"^[A-Za-z0-9_]+$", cls):
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

    html = f'''<!DOCTYPE html><html lang="zh"><head>
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

    html = f'''<!DOCTYPE html><html lang="zh"><head>
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

    html = f'''<!DOCTYPE html><html lang="zh"><head>
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
        raise HTTPException(400)
    return JSONResponse(_exemplar_state(cls))


@app.post("/api/exemplar/{cls}")
async def api_exemplar_post(cls: str, payload: dict = Body(...)):
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
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


@app.post("/api/cancel_job/{jobid}")
def api_cancel_job(jobid: str):
    """Cancel a SLURM job by ID. Only allow cancelling job names we recognize
    as 'safe to interrupt' (agent jobs, not the dashboard itself)."""
    if not _re_cls.match(r'^[0-9]+$', jobid):
        raise HTTPException(400, "bad jobid")
    # Check job name to prevent self-killing
    sq = _shell(["squeue", "-j", jobid, "-h", "-o", "%j"], timeout=8)
    name = (sq["stdout"] or "").strip()
    if not name:
        return JSONResponse({"ok": False, "msg": f"job {jobid} not in queue"})
    # Whitelist: agent jobs only. Don't cancel the dashboard from /control
    # (use restart_dashboard action for that instead).
    SAFE_PREFIXES = ("dl_known", "brain_hrv", "topic_bf", "smoke", "lora_")
    if not any(name.startswith(p) for p in SAFE_PREFIXES):
        return JSONResponse({"ok": False, "msg":
            f"refuse to cancel {name!r} — only agent jobs cancellable from UI"})
    r = _shell(["scancel", jobid], timeout=8)
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


@app.get("/api/action_history")
def api_action_history(n: int = 50):
    """Last N action invocations (from cluster_actions.jsonl)."""
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
    sq = _shell(["squeue", "-j", jobid, "-h", "-o", "%T"], timeout=6)
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


@app.get("/api/cluster_status")
def api_cluster_status():
    """Return a structured snapshot of cluster state — what /control polls."""
    out: dict = {"generated_at": time.time()}

    # SLURM job queue for our user
    sq = _shell(["squeue", "-u", "byler",
                 "-o", "%i\t%j\t%T\t%M\t%R\t%C\t%m"], timeout=10)
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
        "label": "重启 dashboard server (cancels current job + sbatch new)",
    },
    "brain_harvest": {
        "type": "sbatch",
        "script": "run_v3_0_43_brain_harvest_oneshot.sh",
        "label": "Brain 一轮 harvest_new_datasets — 找 + 抓 NEW 数据集 (~30 min)",
    },
    "download_known_slugs": {
        "type": "sbatch",
        "script": "run_v3_0_43_download_known_slugs.sh",
        "label": "下载所有 status=known 的 HF slug 到 /ocean cluster (~30min-2hr)",
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
}


@app.get("/api/job_log/{jobid}")
def api_job_log(jobid: str, tail: int = 200):
    """Return the last `tail` lines of the SLURM output file for jobid.

    SLURM writes to results/framework/<name>_<jobid>.out (with SBATCH --output).
    We glob for *_{jobid}.out to find the file regardless of name."""
    if not _re_cls.match(r'^[0-9_]+$', jobid):
        raise HTTPException(400, "bad jobid chars")
    if not 1 <= tail <= 2000:
        tail = 200

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
def api_cluster_action(action: str):
    """Trigger one of the whitelisted actions. Returns the sbatch output
    (job id) or success marker.

    SECURITY: actions are whitelisted by name. No arbitrary shell injection.
    The dashboard URL is unguessable trycloudflare URL — acceptable risk
    for a research dashboard."""
    if action not in _CLUSTER_ACTIONS:
        raise HTTPException(400, f"unknown action {action!r}")
    spec = _CLUSTER_ACTIONS[action]

    if spec["type"] == "refresh":
        # delegate to /api/refresh_registry logic
        _registry_index_cache["mtime"] = 0.0
        _registry_index_cache["index"] = {}
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
        r = _shell(["sbatch", str(script_path)], timeout=15)
        result = {"ok": r["ok"], "action": action,
                  "stdout": r["stdout"].strip(),
                  "stderr": r["stderr"].strip(),
                  "msg": (r["stdout"].strip()
                          if r["ok"] else r["stderr"].strip())}
        _log_action(action, result)
        return result

    if spec["type"] == "restart_self":
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
    # 1) in-memory registry index — force reload by zeroing mtime cache
    _registry_index_cache["mtime"] = 0.0
    _registry_index_cache["index"] = {}
    _registry_index_cache["empty_slugs"] = []
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
def _all_known_classes() -> _List_cls[str]:
    """Union of CANONICAL_12 + bank folders + registry slugs' class_names.
    Auto-grows: when a slug with new class_names is registered, that class
    appears here automatically (next /classes load picks up registry mtime change)."""
    out: set = set(_CWD12)
    bd = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
    if bd.is_dir():
        for d in bd.iterdir():
            if d.is_dir():
                out.add(d.name)
    for c in _load_registry_index().keys():
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


_registry_parse_cache = {"mtime": 0.0, "data": None}


def _get_cached_registry() -> dict:
    """Module-level cached parse of dataset_registry.json (52MB).
    mtime-invalidated. Use instead of opening + json.load every call —
    355× per /classes render was timing out the cloudflare tunnel."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        mt = REGISTRY_PATH.stat().st_mtime
    except Exception:
        return _registry_parse_cache["data"] or {}
    if _registry_parse_cache["mtime"] == mt and _registry_parse_cache["data"]:
        return _registry_parse_cache["data"]
    try:
        with open(REGISTRY_PATH) as f:
            data = _json.load(f)
        _registry_parse_cache["mtime"] = mt
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


def _class_summary_landing(cls: str) -> dict:
    """Lightweight per-class summary for /classes landing — does NOT walk
    registry labels. Cheap enough that listing 50+ classes is sub-second.
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
    slugs = _load_registry_index().get(cls, [])
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
    if first_src is None and slugs:
        try:
            # Use the proper class-specific pool builder (disk-cached).
            # First entry is an image whose label contains this class's bbox.
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
                        cache_hit_empty = True  # we already searched; no thumb
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
                # Persist whatever we found (or empty if nothing)
                try:
                    thumb_cache_p.write_text(str(first_src) if first_src else "")
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


@app.get("/morning_report", response_class=HTMLResponse)
def morning_report():
    """One-page 'what happened overnight' summary. Read once with morning coffee."""
    # Action history
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

    return HTMLResponse(f'''<!DOCTYPE html><html lang="zh"><head>
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
    return HTMLResponse('''<!DOCTYPE html><html lang="zh"><head>
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
<div class="sub">实时监控 + 一键操作 — 你不需要叫我做这些</div>
<div class="nav">
  <a href="/">🏠 hub</a>
  <a href="/classes">📋 classes</a>
  <a href="/slugs">📦 slugs</a>
  <a href="/api/cluster_status">📥 JSON</a>
</div>

<div class="grid-stats" id="stats"></div>

<div class="section" id="agent-section">
  <h3>🤖 agent 当前活动(latest dl_known / brain_harvest / topic_backfill)</h3>
  <div id="agent-progress">
    <span style="color:#999">loading…</span>
  </div>
</div>

<div class="section">
  <h3>🔘 actions — 点击触发(后端 sbatch / 缓存清理)</h3>
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
      sub: 'Brain + 人工已分类' },
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
    document.getElementById('action-history').innerHTML = events.map(e => {
      const msg = (e.result && (e.result.msg || JSON.stringify(e.result))) || '';
      const t = e.ts_h ? e.ts_h.replace('UTC','') : '?';
      return `<div class="history-item">
        <span class="ts">${t}</span>
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
    if (!confirm('重启 dashboard? 当前页面 ~90 秒后会重连(刷新 github.io 看新 URL)')) return;
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
def slugs_landing():
    """Slug-level cleanup: ✓ keep / ✗ junk / 🤔 unsure on whole slugs.
    Faster than per-image audit when a slug is obviously garbage (plant
    disease imported by accident, etc.)."""
    # Load registry once
    if not REGISTRY_PATH.exists():
        return HTMLResponse("<h1>no registry</h1>", status_code=404)
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    datasets = reg.get("datasets", {})
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
                ("keep", "✓", "保留 (good slug)"),
                ("junk", "✗", "删除 (garbage slug)"),
                ("unsure", "🤔", "存疑"),
            ]
        )
        tr_html.append(f'''
        <tr class="srow{v_class}" data-slug="{r["slug"]}" data-verdict="{r["verdict"]}">
          <td class="slug-col">
            <a href="/classes#q={r["slug"][:10]}" title="filter /classes by this slug">{r["slug"]}</a>
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

    html = f'''<!DOCTYPE html><html lang="zh"><head>
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
<header>
  <h1>📦 Slugs — registry-level 清理</h1>
  <div class="sub">
    每行一个 slug — ✓ 保留 / ✗ 删除(垃圾) / 🤔 存疑。
    标 ✗ 的 slug 默认从 /classes 隐藏。
    · <a href="/classes">/classes 类级审计</a>
    · <a href="/">dashboard 首页</a>
    · <a href="/api/slug_verdicts">📥 JSON</a>
  </div>
</header>
<div class="summary">
  <div>📊 总 <strong>{len(datasets)}</strong> 个 slugs</div>
  <div>✓ keep: <strong>{n_keep}</strong></div>
  <div>🤔 unsure: <strong>{n_unsure}</strong></div>
  <div>✗ junk: <strong>{n_junk}</strong></div>
  <div>未审: <strong>{n_unverified}</strong></div>
</div>
<div class="filter-bar" id="filter-bar">
  <button class="on" data-f="all">全部 {len(datasets)}</button>
  <button data-f="unverified">未审 {n_unverified}</button>
  <button data-f="keep">✓ keep {n_keep}</button>
  <button data-f="unsure">🤔 unsure {n_unsure}</button>
  <button data-f="junk">✗ junk {n_junk}</button>
  <button data-f="has_classnames">有 class_names</button>
  <button data-f="empty_classnames">无 class_names</button>
</div>
<table>
<thead>
<tr><th>Slug</th><th>状态</th><th class="num"># 图</th><th class="num"># 类</th><th>class_names 预览</th><th>verdict</th></tr>
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
def classes_landing():
    rows = []
    topic_counts: dict = {"all": 0, "cwd12": 0, "weed": 0, "disease": 0,
                          "pest": 0, "crop": 0, "other": 0}
    for cls in _all_known_classes():
        summary = _class_summary_landing(cls)
        state = _exemplar_state(cls)
        n_bank = summary["n_bank"]
        n_flux = summary["n_flux"]
        n_reg_slugs = summary["n_reg_slugs"]
        n_reg_est = summary["n_reg_est"]
        n_total_est = n_bank + n_flux + n_reg_est
        n_ex = sum(1 for v in state.values() if v == "exemplar")
        n_bad = sum(1 for v in state.values() if v == "bad")
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
            <span class="topic-tag tag-{topic}" title="{'手动覆盖' if is_override else '关键词启发式'}">{topic}{'★' if is_override else ''}</span>
            {(f'<span class="badge exemplar">✓ {n_ex}</span>' if n_ex else '')}
            {(f'<span class="badge bad">✗ {n_bad}</span>' if n_bad else '')}
          </div>
          <div class="zh">{zh}</div>
          <div class="counts">bank {n_bank} · flux {n_flux} · real ≤{n_reg_est} ({n_reg_slugs} slugs)</div>
          <div class="counts">已审 {n_ex+n_bad}</div>
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
    empty_slugs = _registry_empty_slugs()
    banner = ""
    if empty_slugs:
        sample = ", ".join(empty_slugs[:6])
        more = f" (+{len(empty_slugs)-6} 更多)" if len(empty_slugs) > 6 else ""
        banner = f'''
  <div class="help" style="background:#fee;border-left-color:#c00;">
    ⚠️ <strong>Registry metadata gap</strong>:{len(empty_slugs)} 个已下载的 slugs <strong>class_names 为空</strong>,
    其图片<strong>不会</strong>出现在下面任何类里。需要先跑 backfill 工具补元数据。<br>
    举例:<code>{sample}</code>{more}
  </div>'''

    html = f'''<!DOCTYPE html><html lang="zh"><head>
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
<header>
  <h1>📋 Classes — human-in-the-loop 类级数据审计</h1>
  <div class="sub">
    点任一类进入,逐张人眼判定 ✓ 榜样 / ✗ 标错。
    通过的图自动加入该类"榜样集",作为 LoRA / curator / 训练的可信源。
    · <a href="/audit">← 旧 /audit 视图</a>
    · <a href="/">dashboard 首页</a>
    · <a href="/api/exemplars_export">📥 导出榜样集</a>
    · <a href="/slugs">📦 slug 级清理</a>
    · <a href="javascript:void(0)" onclick="refreshRegistry()">♻️ 刷新 registry</a>
  </div>
</header>
{banner}
<div class="filter-bar" id="filter-bar">
  {tab_html}
  <input class="filter-search" id="filter-search" type="search"
         placeholder="🔎 类名搜索 (e.g. Goose, weed, tomato)" autocomplete="off"/>
</div>
<section>
  <div class="grid-classes" id="grid-classes">{cards}</div>
  <div class="filter-empty-note" id="filter-empty">没有匹配的类。</div>
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
    if not _re_cls.match(r'^[A-Za-z0-9_]+$', cls):
        raise HTTPException(400)
    pool = _class_image_pool(cls)
    if not pool and cls not in _CWD12:
        raise HTTPException(404, f"unknown class {cls!r}")
    state = _exemplar_state(cls)
    zh = _CWD12_ZH.get(cls, "")

    # sidebar of all classes — uses cheap estimate (don't walk labels per class)
    sidebar_rows = []
    for c in _all_known_classes():
        st = _exemplar_state(c) if c != cls else state
        if c == cls:
            n_total = len(pool)
        else:
            sm = _class_summary_landing(c)
            n_total = sm["n_bank"] + sm["n_flux"] + sm["n_reg_est"]
        n_ex = sum(1 for v in st.values() if v == "exemplar")
        cls_attr = ' class="active"' if c == cls else ""
        sidebar_rows.append(
            f'<a href="/classes/{c}"{cls_attr}>'
            f'<div>{c}</div>'
            f'<div class="row-counts">{n_total} · ✓{n_ex}</div>'
            f'</a>')
    sidebar = "".join(sidebar_rows)

    cards = []
    n_ex = n_bad = n_rb = n_un = 0
    src_breakdown: dict = {}
    for entry in pool:
        key, thumb_url, full_url, src_tag = _pool_entry_urls(entry, cls)
        kind = entry["kind"]
        fn = entry["fname"]
        src_breakdown[src_tag] = src_breakdown.get(src_tag, 0) + 1
        verdict = state.get(key, "")
        if verdict == "exemplar":  n_ex += 1
        elif verdict == "bad":     n_bad += 1
        elif verdict == "rebox":   n_rb += 1
        else:                      n_un += 1
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
              title="榜样 (1)" data-v="exemplar">✓</button>
            <button class="ng{(' on' if verdict=='bad' else '')}"
              title="错标 (2)" data-v="bad">✗</button>
            <button class="rb{(' on' if verdict=='rebox' else '')}"
              title="bbox 不准 (3)" data-v="rebox">🔄</button>
          </div>
        </div>''')
    src_summary = " · ".join(f"{k} {v}" for k, v in sorted(src_breakdown.items()))

    html = f'''<!DOCTYPE html><html lang="zh"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cls} — verify</title>
<style>{_CLASSES_CSS}</style>
</head><body>
<header>
  <h1>📋 类审计 · {cls} <span style="color:#888;font-weight:normal;">/ {zh}</span></h1>
  <div class="sub">
    共 <strong>{len(pool)}</strong> 张候选 ({src_summary}) ·
    <span class="badge exemplar">✓ 榜样 {n_ex}</span>
    <span class="badge bad">✗ 错标 {n_bad}</span>
    <span class="badge rebox">🔄 bbox 待修 {n_rb}</span>
    · 未审 {n_un}
    · <a href="/classes">← 所有类</a>
    · <a href="/audit/class/{cls}">旧视图</a>
  </div>
</header>
<div class="help">
  快捷键:鼠标悬停在缩略图上,按 <kbd>1</kbd>=榜样 ✓,<kbd>2</kbd>=错标 ✗,<kbd>3</kbd>=bbox 待修 🔄,<kbd>0</kbd>=清除。
  点击缩略图弹出原图新窗口。下方过滤器按状态筛选。
</div>
<div class="layout">
  <aside class="sidebar">{sidebar}</aside>
  <main>
    <div class="filters">
      <button class="on" data-f="all">全部 {len(pool)}</button>
      <button data-f="unverified">未审 {n_un}</button>
      <button data-f="exemplar">榜样 ✓ {n_ex}</button>
      <button data-f="bad">错标 ✗ {n_bad}</button>
      <button data-f="rebox">bbox 修 🔄 {n_rb}</button>
    </div>
    <div class="grid" id="grid">{''.join(cards)}</div>
  </main>
</div>
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
