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
from pathlib import Path

logger = logging.getLogger(__name__)

# Curated Roboflow Universe projects — verified weed/crop detection with bbox.
# Each entry: (workspace, project_id, version, est_images, description).
# `version` can be an int or "latest".
# This list is the safety net when search returns nothing.
CURATED_PROJECTS = [
    # These slugs are illustrative — search fallback will recover if any 404.
    ("augmented-startups", "weed-detection-dlodq", "latest", 4000, "weed detection"),
    ("roboflow-universe-projects", "weeds-nxe1w", "latest", 3000, "weeds"),
    ("weed-detection-ros4r", "weed-detection-ros4r", "latest", 2500, "weed ROS"),
    ("ws-1", "crop-and-weed-detection-xzlff", "latest", 2000, "crop vs weed"),
    ("agroai", "agricultural-weeds-detection-yz4iu", "latest", 3500, "agricultural weeds"),
    ("iitrpr", "crop-weed-cws4c", "latest", 2200, "crop-weed IITR"),
]


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


def search_roboflow_universe(query, max_results=30):
    """Search Roboflow Universe public API for projects matching query.

    Endpoint is the Universe search backend used by universe.roboflow.com.
    Returns list of {workspace, project, images, classes, url}.
    """
    try:
        q = urllib.parse.quote(query)
        url = f"https://api.roboflow.com/universe/search?query={q}&limit={max_results}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "weed-llm-benchmark",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        logger.warning(f"[Roboflow] search '{query}' failed: {e}")
        return []

    results = []
    for item in data.get("items", data.get("results", []))[:max_results]:
        ws = item.get("workspace") or item.get("universe", {}).get("workspace")
        proj = item.get("id") or item.get("slug") or item.get("project")
        images = item.get("images", item.get("image_count", 0))
        if not ws or not proj:
            continue
        results.append({
            "workspace": ws, "project": proj,
            "images": images,
            "classes": item.get("classes", item.get("class_count", "?")),
            "description": (item.get("description") or "")[:200],
        })
    return results


def _find_yolo_dataset_root(root):
    """Find the directory containing train/valid YOLO data under root."""
    root = Path(root)
    # Roboflow export usually has: {export_name}/data.yaml + train/ + valid/
    for yaml in root.rglob("data.yaml"):
        if (yaml.parent / "train").is_dir():
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

    try:
        if version == "latest":
            versions = project_obj.versions()
            if not versions:
                return None, {"status": "no_versions"}
            version_obj = versions[-1]
        else:
            version_obj = project_obj.version(int(version))
    except Exception as e:
        logger.warning(f"[Roboflow] version fetch {workspace}/{project} failed: {e}")
        return None, {"status": "version_fetch_failed", "error": str(e)[:200]}

    # Download into a scratch subdir inside dest_dir
    scratch = os.path.join(dest_dir, "_rf_tmp")
    os.makedirs(scratch, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(scratch)
        version_obj.download("yolov8")
    except Exception as e:
        os.chdir(prev_cwd)
        shutil.rmtree(scratch, ignore_errors=True)
        logger.warning(f"[Roboflow] download {workspace}/{project} failed: {e}")
        return None, {"status": "download_failed", "error": str(e)[:200]}
    finally:
        os.chdir(prev_cwd)

    # Find the actual export dir
    root = _find_yolo_dataset_root(scratch)
    if root is None:
        shutil.rmtree(scratch, ignore_errors=True)
        return None, {"status": "no_yolo_structure"}

    # Move out of scratch to dest
    slug = f"rf_{workspace.replace('/', '_')}__{project}".lower()
    final = os.path.join(dest_dir, slug)
    if os.path.exists(final):
        shutil.rmtree(final, ignore_errors=True)
    shutil.move(str(root), final)
    shutil.rmtree(scratch, ignore_errors=True)

    img_count = _count_images(final)
    lbl_count = _count_labels(final)
    return final, {
        "status": "downloaded",
        "images": img_count,
        "labeled": lbl_count,
        "workspace": workspace,
        "project": project,
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

    # Build candidate pool
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
    for ws, proj, ver, est, desc in CURATED_PROJECTS:
        key = (ws, proj)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "workspace": ws, "project": proj, "version": ver,
            "est_images": est, "description": desc, "source_query": "curated",
        })

    # Sort: API-search results first (likely more relevant), then by est_images desc
    candidates.sort(key=lambda c: (c["source_query"] == "curated", -c["est_images"]))

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
