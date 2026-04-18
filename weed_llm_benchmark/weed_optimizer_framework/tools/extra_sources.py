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

def _kaggle_token():
    """Pick up Kaggle v2 token from env. Returns None if unset."""
    return os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")


def _kaggle_http_search(query, token, max_results=20):
    """Autonomous search via Kaggle v1 datasets/list REST API (v2 bearer).

    Returns list of {ref, title, size_mb, downloads, votes}.
    """
    try:
        import urllib.parse, urllib.request, json
        url = (f"https://www.kaggle.com/api/v1/datasets/list"
               f"?search={urllib.parse.quote(query)}&page=1&sortBy=hottest")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "weed-llm-benchmark",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        logger.warning(f"[Kaggle] HTTP search '{query}' failed: {e}")
        return []
    out = []
    for d in data:
        ref = d.get("ref") or f"{d.get('ownerName','?')}/{d.get('datasetSlug','?')}"
        if "/" not in ref:
            continue
        out.append({
            "ref": ref,
            "title": d.get("title", ""),
            "size_mb": d.get("totalBytes", 0) // (1024 * 1024) if d.get("totalBytes") else 0,
            "downloads": d.get("downloadCount", 0),
            "votes": d.get("voteCount", 0),
        })
        if len(out) >= max_results:
            break
    return out


def harvest_kaggle_datasets(data_dir, queries, already_known_cb, max_new=5):
    """Autonomously search Kaggle + download matching weed/crop bbox datasets.

    Uses KAGGLE_API_TOKEN env (v2 bearer auth). No hardcoded slugs — Brain's
    query terms drive discovery. Ranks by downloads, skips already-known slugs.
    Filters by name for agriculture keywords before downloading.
    """
    token = _kaggle_token()
    if not token:
        logger.info("[Kaggle] KAGGLE_API_TOKEN not set — skipping")
        return []
    try:
        import kagglehub  # noqa
    except ImportError:
        logger.info("[Kaggle] kagglehub not installed — skipping")
        return []

    # v3.0.11: agriculture filter only. Classification datasets are kept now —
    # autolabel_dataset will convert them to YOLO bbox via OWLv2 (class-known
    # → high-quality localization). Dropping the detection-hint requirement is
    # what unlocks the 380K+ plant-classification pool that's the only way to
    # hit the user's "几万到几十万" target.
    AG_VOCAB = ("weed", "crop", "plant", "leaf", "fruit", "rice", "wheat",
                "corn", "cotton", "soybean", "agri", "farm", "pest", "seedling",
                "tomato", "potato", "disease")
    seen = set()
    candidates = []
    for q in queries:
        for kr in _kaggle_http_search(q, token, max_results=30):
            ref = kr["ref"]
            if ref in seen:
                continue
            seen.add(ref)
            haystack = f"{ref} {kr.get('title','')}".lower()
            if not any(v in haystack for v in AG_VOCAB):
                continue
            candidates.append({**kr, "source_query": q})

    # Rank by downloads desc
    candidates.sort(key=lambda c: -c.get("downloads", 0))
    logger.info(f"[Kaggle] {len(candidates)} autonomous candidates from "
                f"{len(queries)} search queries")

    results = []
    for c in candidates:
        if len(results) >= max_new:
            break
        ref = c["ref"]
        slug = "kg_" + ref.replace("/", "__").lower()
        if already_known_cb(slug):
            continue
        try:
            # kagglehub picks up KAGGLE_API_TOKEN automatically
            path = kagglehub.dataset_download(ref)
        except Exception as e:
            logger.debug(f"[Kaggle] {ref}: download failed ({str(e)[:100]})")
            continue
        if not os.path.isdir(path):
            continue

        img_count = _count_images(path)
        lbl_count = _count_labels(path)
        if img_count < 50:
            logger.debug(f"[Kaggle] {ref}: too few images ({img_count}) — skip")
            continue
        # v3.0.11: classification-only sets are now KEPT and marked for auto-label.
        # OWLv2 will generate pseudo-bboxes in a later step. Converts 380K+
        # plant-classification images into usable training data — matches the user's
        # "几万到几十万" target that real-bbox searches alone can't hit.
        is_autolabel_candidate = (lbl_count == 0)

        local_path = os.path.join(data_dir, slug)
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=True)
        try:
            shutil.copytree(path, local_path)
        except Exception as e:
            logger.warning(f"[Kaggle] copy {ref} failed: {e}")
            continue

        annotation = "needs_autolabel" if is_autolabel_candidate else "yolo"
        logger.info(f"[Kaggle] ✓ {ref} (q={c['source_query']}, dl={c['downloads']}): "
                    f"{img_count} imgs, {lbl_count} labels, annotation={annotation}")
        info = {
            "source": "kaggle", "hf_id": None, "kaggle_ref": ref,
            "images": img_count, "classes": "?",
            "annotation": annotation,
            "format": "yolo", "description": c["title"][:300],
            "status": "downloaded", "local_path": local_path,
            "local_images": img_count, "class_names": [],
            "downloaded_at": None, "used_for_training": False,
            "training_runs": [], "harvest_reason": f"kaggle:{c['source_query']}",
        }
        results.append({
            "slug": slug, "info": info,
            "stats": {"status": "downloaded", "images": img_count,
                      "labeled": lbl_count, "annotation_kind": "yolo"},
            "hf_id": ref, "reason": f"kaggle:{c['source_query']}",
        })
    return results
