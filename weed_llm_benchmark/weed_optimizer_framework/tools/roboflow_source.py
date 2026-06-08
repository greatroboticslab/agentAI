"""
Roboflow Universe as a weed/crop dataset source.

Why: HF yields ~1-3 usable bbox datasets/round. Roboflow Universe has thousands
of YOLO-exportable weed/crop/plant projects, each typically 1K-10K bbox images.
One harvest round through Roboflow should add 20K+ images to the registry —
matching the user's v3.0 north-star of 50K+ real bbox labels.

Key:
  Looked up from `/Users/.../.roboflow_key` (local) or env WEED_ROBOFLOW_KEY.
  `reference_plantid_api.md` memory mentions a saved key; never hardcode it.

Gracefully degrades when:
  * roboflow package not installed
  * API key not found
  * Universe search returns nothing
"""

import os
import json
import shutil
import logging
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

# v3.0.9: CURATED_PROJECTS removed per autonomy principle — no human-seeded slugs.
# Roboflow Universe has no public search API (probed 2026-04-18: 403/401 on all
# candidate endpoints), so Brain cannot autonomously discover projects here.
# Until Roboflow exposes a search API or user provides workspace URLs at runtime,
# this source is a no-op — keeping the code so it's easy to re-enable later.


def load_api_key():
    """Load Roboflow API key from env or ~/.roboflow_key or project root."""
    env = os.environ.get("WEED_ROBOFLOW_KEY") or os.environ.get("ROBOFLOW_KEY")
    if env:
        return env.strip()
    # Search common locations
    candidates = [
        Path.home() / ".roboflow_key",
        Path.cwd() / ".roboflow_key",
    ]
    # Walk up 4 levels from this file
    here = Path(__file__).resolve()
    for up in range(5):
        candidates.append(here.parents[up] / ".roboflow_key")
    for p in candidates:
        try:
            if p.exists():
                text = p.read_text(errors="ignore").strip()
                if text:
                    return text.splitlines()[0].strip()
        except Exception:
            continue
    return None


# v3.0.99.8 (2026-06-08): Universe search API WORKS now — the official endpoint is
#   GET https://api.roboflow.com/universe/search?q=<query>&api_key=<KEY>
# (the 2026-04-18 probe failed because it used ?query= and no api_key). Each result
# has: name, url (…/<workspace>/<project>), workspace.{name,url}, type, classes[],
# classCount, images, latestVersion, license, etc. This re-enables Roboflow Universe
# as a real source (the only one that has the missing CWD12 species — Kaggle/HF/github
# came up empty 2026-06-08). See project_overnight_loop_2026_06_07.

# class-name vocab to drop off-topic Universe hits (disease/PPE/vehicle/etc.)
_RF_CLS_REJECT = (
    "disease", "blight", "rust", "mildew", "lesion", "leaf-spot", "leafspot",
    "boots", "cig", "sunglass", "helmet", "hardhat", "ppe", "person",
    "vehicle", "car", "license", "plate", "face", "x-ray", "blood", "cell",
    "drone", "aerial-vehicle", "pest", "insect", "fruit", "flower",
)


def _ws_proj_from_result(r):
    """Extract (workspace_slug, project_slug) from a Universe search result."""
    ws = ((r.get("workspace") or {}) or {}).get("url") or ""
    url = (r.get("url") or "").rstrip("/")
    proj = url.split("/")[-1] if url else ""
    if not ws and "/" in url:
        # fallback: …/<ws>/<proj>
        parts = url.split("/")
        if len(parts) >= 2:
            ws = parts[-2]
    return ws, proj


def search_roboflow_universe(query, max_results=30, od_only=True, min_images=50):
    """Search Roboflow Universe (official API) for projects matching query.

    GET https://api.roboflow.com/universe/search?q=<query>&api_key=<KEY>
    Returns list of dicts: {workspace, project, images, classCount, classes,
    version, url, name, type}. Filters to object-detection + min_images and drops
    obviously off-topic class vocab (disease/PPE/vehicle/…).
    """
    key = load_api_key()
    if not key:
        logger.info("[Roboflow] no API key — cannot search Universe")
        return []
    # v3.0.99.10: paginate (API returns ~12/page) until max_results or a short page.
    # The Universe search API rate-limits (HTTP 429) — back off + retry, and pace
    # pages so a bulk multi-query sweep doesn't get cut off.
    import time as _t
    items = []
    q = urllib.parse.quote(query)
    for page in range(1, 11):
        data = None
        for attempt in range(3):
            try:
                url = (f"https://api.roboflow.com/universe/search?q={q}"
                       f"&api_key={key}&page={page}")
                req = urllib.request.Request(url, headers={
                    "User-Agent": "weed-llm-benchmark", "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    _t.sleep(4 * (attempt + 1))
                    continue
                logger.warning(f"[Roboflow] search '{query}' p{page}: {e}")
                break
            except Exception as e:
                logger.warning(f"[Roboflow] search '{query}' p{page}: {e}")
                break
        if data is None:
            break
        batch = data.get("results") or []
        items.extend(batch)
        if len(items) >= max_results or len(batch) < int(data.get("page_size", 12) or 12):
            break
        _t.sleep(0.6)   # pace pages to avoid 429

    results = []
    for item in items[:max_results]:
        if od_only and item.get("type") != "object-detection":
            continue
        ws, proj = _ws_proj_from_result(item)
        if not ws or not proj:
            continue
        images = item.get("images", 0) or 0
        if images < min_images:
            continue
        classes = [str(c) for c in (item.get("classes") or [])]
        blob = (item.get("name", "") + " " + " ".join(classes) + " "
                + (item.get("annotation") or "")).lower()
        if any(bad in blob for bad in _RF_CLS_REJECT):
            continue
        results.append({
            "workspace": ws, "project": proj,
            "images": images,
            "classCount": item.get("classCount"),
            "classes": classes[:12],
            "version": item.get("latestVersion"),
            "url": item.get("url"),
            "name": item.get("name"),
            "type": item.get("type"),
            "description": (item.get("description") or "")[:200],
        })
    return results


def _find_yolo_dataset_root(root):
    """Find the directory containing train/valid YOLO data under root."""
    root = Path(root)
    # v3.0.99.10: LENIENT — a Roboflow yolov8 export may have only valid/ or test/
    # (not train/), or images directly. tuuf/goosegrass failed 'no_yolo_structure'
    # because we required train/. Accept data.yaml beside ANY split dir; then any
    # dir with an images/ subdir; last resort any data.yaml parent.
    for yaml in root.rglob("data.yaml"):
        p = yaml.parent
        if any((p / s).is_dir() for s in ("train", "valid", "test", "images")):
            return p
    for imgs in root.rglob("images"):
        if imgs.is_dir():
            return imgs.parent
    for yaml in root.rglob("data.yaml"):
        return yaml.parent
    return None


def _count_images(path):
    return sum(1 for p in Path(path).rglob("*")
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def _count_labels(path):
    return sum(1 for _ in Path(path).rglob("*.txt"))


def download_roboflow_project(api_key, workspace, project, version, dest_dir):
    """Download one Roboflow project version as YOLOv8 format.

    Returns (local_path, stats) or (None, {"status": "failed", ...}).
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.info("[Roboflow] roboflow package missing — install with `pip install roboflow`")
        return None, {"status": "no_package"}

    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
    except Exception as e:
        logger.warning(f"[Roboflow] cannot access {workspace}/{project}: {e}")
        return None, {"status": "not_found", "error": str(e)[:200]}

    # v3.0.99.10: resolve a downloadable Version robustly. Order: explicit int →
    # versions()[-1] → try v1/v2/v3. A Universe project with NO generated version
    # genuinely can't be downloaded (Roboflow needs a version snapshot) → no_versions.
    version_obj = None
    tried = []
    explicit = []
    if isinstance(version, int) or (isinstance(version, str) and version.isdigit()):
        explicit = [int(version)]
    try:
        vs = project_obj.versions()
        if vs:
            version_obj = vs[-1]
    except Exception as e:
        tried.append(f"versions():{str(e)[:50]}")
    if version_obj is None:
        for vn in (explicit or [1, 2, 3]):
            try:
                version_obj = project_obj.version(int(vn))
                break
            except Exception as e:
                tried.append(f"v{vn}:{str(e)[:40]}")
    if version_obj is None:
        return None, {"status": "no_versions", "tried": tried[:5]}

    # v3.0.99.10: download to a UNIQUE ABSOLUTE location — no os.chdir (that caused
    # the 'weed-detection-1/roboflow.zip No such file' relative-path failure when
    # pulling several in a row). Mirrors merge_roboflow_projects' download pattern.
    slug = f"rf_{workspace.replace('/', '_')}__{project}".lower()
    loc = os.path.join(dest_dir, "_rf_dl", slug)
    shutil.rmtree(loc, ignore_errors=True)
    os.makedirs(loc, exist_ok=True)
    try:
        version_obj.download("yolov8", location=loc)
    except Exception as e:
        shutil.rmtree(loc, ignore_errors=True)
        logger.warning(f"[Roboflow] download {workspace}/{project} failed: {e}")
        return None, {"status": "download_failed", "error": str(e)[:200]}

    root = _find_yolo_dataset_root(loc)
    if root is None:
        shutil.rmtree(loc, ignore_errors=True)
        return None, {"status": "no_yolo_structure"}

    final = os.path.join(dest_dir, slug)
    if os.path.exists(final):
        shutil.rmtree(final, ignore_errors=True)
    shutil.move(str(root), final)
    shutil.rmtree(loc, ignore_errors=True)

    return final, {
        "status": "downloaded",
        "images": _count_images(final),
        "labeled": _count_labels(final),
        "workspace": workspace, "project": project,
    }


def harvest_roboflow_datasets(data_dir, queries, already_known_cb, max_new=10):
    """Find + download Roboflow Universe weed/crop projects.

    Strategy:
      1. Try API search for each query (up to 30 results/query).
      2. Add curated projects as fallback seeds.
      3. Dedup + try each until max_new downloaded.

    Returns list of {slug, info, stats, hf_id, reason} (hf_id used as display).
    """
    api_key = load_api_key()
    if not api_key:
        logger.info("[Roboflow] no API key found — skipping")
        return []
    try:
        import roboflow  # noqa
    except ImportError:
        logger.info("[Roboflow] roboflow package missing — skipping")
        return []

    # Autonomous search only — no curated seeds (per v3.0.9 autonomy principle).
    seen = set()
    candidates = []
    for q in queries:
        for r in search_roboflow_universe(q, max_results=30):
            key = (r["workspace"], r["project"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "workspace": r["workspace"], "project": r["project"],
                "version": "latest", "est_images": r.get("images", 0),
                "description": r.get("description", ""), "source_query": q,
            })

    if not candidates:
        logger.info("[Roboflow] 0 candidates (public search unavailable); "
                    "skipping Roboflow this round")
        return []

    # Sort by est_images desc
    candidates.sort(key=lambda c: -c["est_images"])

    logger.info(f"[Roboflow] {len(candidates)} candidates (search + curated)")
    results = []
    for c in candidates:
        if len(results) >= max_new:
            break
        slug = f"rf_{c['workspace'].replace('/', '_')}__{c['project']}".lower()
        if already_known_cb(slug):
            continue
        logger.info(f"[Roboflow] trying {c['workspace']}/{c['project']} "
                    f"(~{c['est_images']} imgs, query={c['source_query']})")
        local_path, stats = download_roboflow_project(
            api_key, c["workspace"], c["project"], c["version"], data_dir,
        )
        if stats.get("status") != "downloaded":
            logger.info(f"[Roboflow] skip {c['workspace']}/{c['project']}: {stats.get('status')}")
            continue
        if stats.get("images", 0) < 50 or stats.get("labeled", 0) == 0:
            logger.debug(f"[Roboflow] {slug}: too small, removing")
            shutil.rmtree(local_path, ignore_errors=True)
            continue

        logger.info(f"[Roboflow] ✓ {slug}: {stats['images']} imgs, {stats['labeled']} labels")
        info = {
            "source": "roboflow", "hf_id": None,
            "roboflow_workspace": c["workspace"],
            "roboflow_project": c["project"],
            "roboflow_version": c["version"],
            "images": stats["images"], "classes": "?",
            "annotation": "yolo", "format": "yolo",
            "description": f"Roboflow Universe: {c['description']}"[:300],
            "status": "downloaded", "local_path": local_path,
            "local_images": stats["images"], "class_names": [],
            "downloaded_at": None, "used_for_training": False,
            "training_runs": [], "harvest_reason": f"roboflow:{c['source_query']}",
        }
        results.append({
            "slug": slug, "info": info, "stats": {
                "status": "downloaded", "images": stats["images"],
                "labeled": stats["labeled"], "annotation_kind": "yolo",
            },
            "hf_id": f"{c['workspace']}/{c['project']}",
            "reason": f"roboflow:{c['source_query']}",
        })
    return results


def _read_yaml_classes(local_path):
    """Read class names from a downloaded Roboflow yolov8 data.yaml."""
    try:
        import yaml
        dy = Path(local_path) / "data.yaml"
        if dy.is_file():
            d = yaml.safe_load(dy.read_text(errors="ignore")) or {}
            nm = d.get("names")
            if isinstance(nm, dict):
                return [nm[k] for k in sorted(nm, key=lambda x: int(x))]
            if isinstance(nm, list):
                return nm
    except Exception:
        pass
    return []


# --------------------------------------------------------------------------
# v3.0.99.8 CLI — MANUAL, human-driven Universe search + pull (user 2026-06-08:
# "接 Roboflow Universe 搜索, 手动拉缺失物种数据集"). This is a deliberate human-in-
# the-loop tool, separate from the autonomous Brain (which stays seed-free).
#   search "<q1>" "<q2>" …      → ranked candidates (review before pulling)
#   pull <workspace> <project> [version]  → download + register + dual-write Mongo
# --------------------------------------------------------------------------
def cmd_search(queries, min_images=50):
    seen = set()
    rows = []
    for q in queries:
        for r in search_roboflow_universe(q, max_results=30, min_images=min_images):
            k = (r["workspace"], r["project"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    rows.sort(key=lambda r: -(r.get("images") or 0))
    print(f"\n=== {len(rows)} Universe object-detection candidates "
          f"(queries={queries}) ===")
    for r in rows:
        print(f"  {r['images']:>6} imgs  {r['workspace']}/{r['project']}  "
              f"v{r.get('version')}  classes({r.get('classCount')})={r.get('classes')}")
    return rows


def cmd_pull(workspace, project, version="latest"):
    key = load_api_key()
    if not key:
        print("FATAL: no Roboflow API key")
        return 2
    # lazy import to avoid circular import (dataset_discovery imports this module)
    from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery
    d = DatasetDiscovery()
    slug = f"rf_{workspace}__{project}".replace("/", "_").lower()
    if slug in d.registry["datasets"]:
        print(f"[pull] {slug} already in registry — skipping")
        return 0
    print(f"[pull] downloading {workspace}/{project} (v{version}) …")
    local, stats = download_roboflow_project(key, workspace, project, version, d.data_dir)
    if stats.get("status") != "downloaded":
        print(f"[pull] FAILED: {stats}")
        return 2
    classes = _read_yaml_classes(local)
    cur_round = int(d.registry.get("current_round", 1))
    info = {
        "source": "roboflow_universe", "hf_id": None,
        "roboflow_workspace": workspace, "roboflow_project": project,
        "roboflow_version": version,
        "images": stats["images"], "classes": len(classes) or "?",
        "class_names": classes,
        "annotation": "yolo", "format": "yolo",
        "description": f"Roboflow Universe (manual pull): {workspace}/{project}",
        "status": "downloaded", "local_path": local,
        "local_images": stats["images"], "harvest_round": cur_round,
        "harvest_reason": "manual:roboflow_universe",
        "used_for_training": False, "training_runs": [],
    }
    d.registry["datasets"][slug] = info
    d._save_registry()   # dual-writes to Mongo via db.mirror_registry_to_mongo
    print(f"[pull] ✓ registered {slug}: {stats['images']} imgs, "
          f"{stats['labeled']} labels, classes={classes}")
    print(f"[pull] local_path={local}")
    return 0


_BULK_QUERIES = [
    "weed detection", "weed", "crop weed detection", "weed segmentation",
    "agriculture weed", "field weed detection", "weed yolo", "broadleaf weed",
    "grass weed detection", "crop and weed", "weeds in field", "robot weeding",
    "cotton weed", "soybean weed", "corn weed", "rice weed", "sugar beet weed",
    "lettuce weed", "carrot weed", "weed species detection", "smart farming weed",
]


def cmd_bulk(target_images=20000, max_pulls=40, min_images=100, queries=None):
    """Search Universe broadly + pull fresh weed datasets toward an image target.
    Human-driven bulk grow (user 2026-06-08: 批量拉更多 Universe 杂草集冲 50K)."""
    from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery
    key = load_api_key()
    if not key:
        print("FATAL: no Roboflow API key"); return 2
    d = DatasetDiscovery()
    queries = queries or _BULK_QUERIES
    seen, cands = set(), []
    for q in queries:
        for r in search_roboflow_universe(q, max_results=60, min_images=min_images):
            if r.get("version") is None:          # no downloadable version snapshot
                continue
            k = (r["workspace"], r["project"])
            if k in seen:
                continue
            seen.add(k)
            slug = f"rf_{r['workspace']}__{r['project']}".replace("/", "_").lower()
            if slug in d.registry["datasets"]:    # already have it
                continue
            cands.append(r)
    cands.sort(key=lambda r: -(r.get("images") or 0))
    print(f"[bulk] {len(cands)} fresh candidates (over {len(queries)} queries); "
          f"target +{target_images} imgs / max {max_pulls} pulls")
    added = added_imgs = 0
    for r in cands:
        if added >= max_pulls or added_imgs >= target_images:
            break
        ws, proj = r["workspace"], r["project"]
        print(f"[bulk] pull {ws}/{proj} (~{r.get('images')} imgs, classes={r.get('classes')})")
        local, stats = download_roboflow_project(key, ws, proj, r.get("version") or "latest", d.data_dir)
        if stats.get("status") != "downloaded":
            print(f"  skip: {stats.get('status')}")
            continue
        if stats.get("images", 0) < min_images or stats.get("labeled", 0) == 0:
            print(f"  skip: too small/no labels ({stats.get('images')}/{stats.get('labeled')})")
            shutil.rmtree(local, ignore_errors=True)
            continue
        classes = _read_yaml_classes(local)
        slug = f"rf_{ws}__{proj}".replace("/", "_").lower()
        d.registry["datasets"][slug] = {
            "source": "roboflow_universe", "hf_id": None,
            "roboflow_workspace": ws, "roboflow_project": proj,
            "roboflow_version": r.get("version") or "latest",
            "images": stats["images"], "classes": len(classes) or "?",
            "class_names": classes, "annotation": "yolo", "format": "yolo",
            "description": f"Roboflow Universe (bulk): {ws}/{proj}",
            "status": "downloaded", "local_path": local,
            "local_images": stats["images"],
            "harvest_round": int(d.registry.get("current_round", 1)),
            "harvest_reason": "bulk:roboflow_universe",
            "used_for_training": False, "training_runs": [],
        }
        d._save_registry()
        added += 1; added_imgs += stats["images"]
        print(f"  ✓ {slug}: {stats['images']} imgs, {stats['labeled']} labels "
              f"(running total +{added_imgs} imgs / {added} datasets)")
    print(f"[bulk] DONE: added {added} datasets, +{added_imgs} images")
    return 0


def main():
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = sys.argv[1:]
    if not args:
        print("usage: roboflow_source.py search '<q>' …  |  pull <ws> <proj> [version]  "
              "|  bulk [target_images] [max_pulls]")
        return 2
    cmd = args[0]
    if cmd == "search":
        cmd_search(args[1:] or ["weed detection"])
        return 0
    if cmd == "pull":
        if len(args) < 3:
            print("usage: pull <workspace> <project> [version]")
            return 2
        return cmd_pull(args[1], args[2], args[3] if len(args) > 3 else "latest")
    if cmd == "bulk":
        tgt = int(args[1]) if len(args) > 1 else 20000
        mx = int(args[2]) if len(args) > 2 else 40
        return cmd_bulk(target_images=tgt, max_pulls=mx)
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
