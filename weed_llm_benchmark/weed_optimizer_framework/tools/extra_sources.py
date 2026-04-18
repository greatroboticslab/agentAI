"""
Extra dataset sources beyond HuggingFace: GitHub + Kaggle.

Why:
- HuggingFace object-detection pool for weed/crop is thin (~1 usable/round).
- Professor direction: "agent browses GitHub weed-detection repos, clones/trains autonomously".
- Accumulation target: 100K+ real-bbox images needs more sources.

Each source is optional — missing deps/creds log + skip, never raise.
"""

import os
import json
import shutil
import subprocess
import tempfile
import logging
import urllib.request
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)


# =========================================================
# GITHUB
# =========================================================

def search_github_repos(query, max_results=10):
    """Search GitHub for repos via public API (60 req/hr unauth).

    Returns [{full_name, clone_url, stars, description}].
    """
    try:
        q = urllib.parse.quote(f"{query} yolo dataset")
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&per_page={max_results}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "weed-llm-benchmark",
            "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        results = []
        for item in data.get("items", []):
            results.append({
                "full_name": item["full_name"],
                "clone_url": item["clone_url"],
                "stars": item.get("stargazers_count", 0),
                "description": (item.get("description") or "")[:300],
            })
        return results
    except Exception as e:
        logger.warning(f"[GitHub] search '{query}' failed: {e}")
        return []


def scan_for_yolo_dataset(repo_dir):
    """Find YOLO-format dataset roots inside a cloned repo.

    Indicators (any one is enough):
      * data.yaml file
      * a dir containing both `images/` and `labels/` subdirs with ≥10 .txt labels
    """
    repo = Path(repo_dir)
    candidates = set()

    for yaml in repo.rglob("data.yaml"):
        candidates.add(yaml.parent)
    for yaml in repo.rglob("*.yaml"):
        # Roboflow exports sometimes name it `train.yaml` etc.
        try:
            text = yaml.read_text(errors="ignore")[:2000].lower()
            if "train:" in text and ("names:" in text or "nc:" in text):
                candidates.add(yaml.parent)
        except Exception:
            continue

    for img_dir in repo.rglob("images"):
        if not img_dir.is_dir():
            continue
        lbl_dir = img_dir.parent / "labels"
        if lbl_dir.is_dir() and sum(1 for _ in lbl_dir.rglob("*.txt")) >= 10:
            candidates.add(img_dir.parent)

    return list(candidates)


def _count_images(root):
    return sum(1 for p in Path(root).rglob("*")
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def _count_labels(root):
    return sum(1 for _ in Path(root).rglob("*.txt"))


def clone_github_repo(clone_url, dest_dir, depth=1, timeout=300):
    try:
        r = subprocess.run(
            ["git", "clone", "--depth", str(depth), clone_url, dest_dir],
            capture_output=True, timeout=timeout,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"[GitHub] clone failed {clone_url}: {e}")
        return False


def harvest_github_datasets(data_dir, queries, already_known_cb, max_new=3):
    """Search GitHub, clone repos, extract YOLO datasets, return registry entries.

    Returns list of {slug, info_dict, stats} — caller writes to registry.

    already_known_cb(slug_or_full_name) -> bool: True if already in registry.
    """
    if shutil.which("git") is None:
        logger.info("[GitHub] git not available, skipping GitHub source")
        return []

    results = []
    seen_repos = set()

    for q in queries:
        if len(results) >= max_new:
            break
        repos = search_github_repos(q, max_results=10)
        for repo in repos:
            if len(results) >= max_new:
                break
            full_name = repo["full_name"]
            if full_name in seen_repos:
                continue
            seen_repos.add(full_name)
            slug = "gh_" + full_name.replace("/", "__").lower()
            if already_known_cb(slug):
                continue

            with tempfile.TemporaryDirectory() as tmp:
                if not clone_github_repo(repo["clone_url"], tmp):
                    continue
                candidates = scan_for_yolo_dataset(tmp)
                if not candidates:
                    logger.debug(f"[GitHub] {full_name}: no YOLO dataset detected")
                    continue
                best = max(candidates, key=_count_images)
                img_count_src = _count_images(best)
                if img_count_src < 50:
                    logger.debug(f"[GitHub] {full_name}: too few images ({img_count_src})")
                    continue

                local_path = os.path.join(data_dir, slug)
                if os.path.exists(local_path):
                    shutil.rmtree(local_path, ignore_errors=True)
                try:
                    shutil.copytree(best, local_path)
                except Exception as e:
                    logger.warning(f"[GitHub] copy failed {full_name}: {e}")
                    continue

                img_count = _count_images(local_path)
                lbl_count = _count_labels(local_path)
                if lbl_count == 0:
                    shutil.rmtree(local_path, ignore_errors=True)
                    logger.debug(f"[GitHub] {full_name}: no .txt labels, discarded")
                    continue

                logger.info(
                    f"[GitHub] ★{repo['stars']} {full_name}: "
                    f"{img_count} images, {lbl_count} labels → {local_path}"
                )
                info = {
                    "source": "github", "hf_id": None,
                    "github_url": repo["clone_url"],
                    "github_name": full_name,
                    "images": img_count, "classes": "?",
                    "annotation": "yolo" if lbl_count > 0 else "image_only",
                    "format": "yolo",
                    "description": f"GitHub ★{repo['stars']}: {repo['description']}"[:300],
                    "status": "downloaded", "local_path": local_path,
                    "local_images": img_count, "class_names": [],
                    "downloaded_at": None, "used_for_training": False,
                    "training_runs": [], "harvest_reason": f"github:{q}",
                    "gh_stars": repo["stars"],
                }
                results.append({
                    "slug": slug, "info": info,
                    "stats": {"status": "downloaded", "images": img_count,
                              "labeled": lbl_count, "annotation_kind": "yolo"},
                    "hf_id": full_name, "reason": f"github:{q}",
                })
    return results


# =========================================================
# KAGGLE
# =========================================================

def _kaggle_cli_search(query, max_results=5):
    try:
        r = subprocess.run(
            ["kaggle", "datasets", "list", "-s", query],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            return []
        results = []
        for line in r.stdout.decode("utf-8", errors="ignore").splitlines()[2:]:
            parts = line.split()
            if len(parts) >= 1 and "/" in parts[0]:
                results.append({"ref": parts[0], "title": " ".join(parts[1:])[:80]})
                if len(results) >= max_results:
                    break
        return results
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# Curated Kaggle weed/crop bbox datasets — known to have YOLO labels.
# Added to every harvest run as seeds (in addition to CLI search results).
CURATED_KAGGLE = [
    # ref, est_images, description
    ("ravirajsinh45/crop-and-weed-detection", 1300, "Crop-weed YOLO bbox (ravirajsinh45)"),
    ("gavinarmstrong/open-sprayer-images", 6000, "Open Sprayer weed images"),
    ("jaidalmotra/cotton-weed-dataset", 4000, "Cotton weed"),
    ("fpeccia/weed-detection-in-soybean-crops", 15000, "Soybean weed (large)"),
    ("vbookshelf/v2-plant-seedlings-dataset", 5500, "V2 Plant seedlings"),
]


def harvest_kaggle_datasets(data_dir, queries, already_known_cb, max_new=3):
    """Search + download Kaggle datasets via kagglehub + kaggle CLI.

    Strategy: merge CURATED_KAGGLE seeds with `kaggle datasets list -s` results.
    Needs ~/.kaggle/kaggle.json or env creds. Silently skips if missing.
    """
    try:
        import kagglehub  # noqa
    except ImportError:
        logger.info("[Kaggle] kagglehub not installed — skipping")
        return []
    # Verify creds present — try a dummy listing
    if shutil.which("kaggle") is None:
        logger.info("[Kaggle] `kaggle` CLI missing, using curated-only mode")

    # Build candidate list
    seen = set()
    candidates = []

    # 1. Curated seeds (always tried)
    for ref, est, desc in CURATED_KAGGLE:
        if ref in seen:
            continue
        seen.add(ref)
        candidates.append({"ref": ref, "title": desc, "source": "curated"})

    # 2. Search results (if CLI available)
    if shutil.which("kaggle") is not None:
        for q in queries:
            for kr in _kaggle_cli_search(q, max_results=5):
                if kr["ref"] in seen:
                    continue
                seen.add(kr["ref"])
                candidates.append({**kr, "source": f"search:{q}"})

    logger.info(f"[Kaggle] {len(candidates)} candidates (curated + search)")

    results = []
    for c in candidates:
        if len(results) >= max_new:
            break
        ref = c["ref"]
        slug = "kg_" + ref.replace("/", "__").lower()
        if already_known_cb(slug):
            continue

        try:
            path = kagglehub.dataset_download(ref)
        except Exception as e:
            logger.debug(f"[Kaggle] {ref}: download failed ({str(e)[:100]})")
            continue
        if not os.path.isdir(path):
            continue

        img_count = _count_images(path)
        lbl_count = _count_labels(path)
        if img_count < 50:
            logger.debug(f"[Kaggle] {ref}: too few images ({img_count}), skip")
            continue
        # Allow label-less datasets through (some cotton-weed Kaggle sets are image-only);
        # mega_trainer will ignore them via the annotation filter.

        local_path = os.path.join(data_dir, slug)
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        try:
            shutil.copytree(path, local_path)
        except Exception as e:
            logger.warning(f"[Kaggle] copy {ref} failed: {e}")
            continue

        logger.info(f"[Kaggle] ✓ {ref} ({c['source']}): {img_count} imgs, "
                    f"{lbl_count} labels → {slug}")
        info = {
            "source": "kaggle", "hf_id": None, "kaggle_ref": ref,
            "images": img_count, "classes": "?",
            "annotation": "yolo" if lbl_count > 0 else "image_only",
            "format": "yolo", "description": c["title"][:300],
            "status": "downloaded", "local_path": local_path,
            "local_images": img_count, "class_names": [],
            "downloaded_at": None, "used_for_training": False,
            "training_runs": [], "harvest_reason": f"kaggle:{c['source']}",
        }
        results.append({
            "slug": slug, "info": info,
            "stats": {"status": "downloaded", "images": img_count,
                      "labeled": lbl_count, "annotation_kind": "yolo"},
            "hf_id": ref, "reason": f"kaggle:{c['source']}",
        })
    return results
