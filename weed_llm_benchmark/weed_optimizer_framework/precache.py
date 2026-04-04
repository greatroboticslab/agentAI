#!/usr/bin/env python3
"""
Pre-cache — Run external API calls locally before cluster job.

Bridges-2 blocks external HTTPS. This script runs on a machine WITH internet
and caches all API results to JSON. The cluster reads from cache.

Usage:
    # Run locally (has internet access):
    export PLANT_API_KEY=your_key
    python -m weed_optimizer_framework.precache --images-dir /path/to/test/images

    # Upload cache to cluster, then run framework (reads from cache)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from .config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(Config.FRAMEWORK_DIR, "api_cache.json")


def load_cache():
    """Load existing cache."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {"plant_id": {}, "huggingface_search": {}}


def save_cache(cache):
    """Save cache with atomic write."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, CACHE_PATH)
    logger.info(f"Cache saved to {CACHE_PATH}")


def cache_plant_id(images_dir, max_images=20):
    """Run plant.id on images and cache results."""
    from .tools.web_identifier import WebIdentifier

    api_key = os.environ.get("PLANT_API_KEY", "")
    if not api_key:
        logger.error("Set PLANT_API_KEY environment variable")
        return {}

    identifier = WebIdentifier(api_key=api_key)
    if not identifier.api_available:
        logger.error("plant.id API not available")
        return {}

    logger.info(f"Running plant.id on {images_dir} (max {max_images} images)...")
    results = identifier.identify_batch(images_dir, max_images=max_images, use_api=True)
    logger.info(f"Identified {len(results)} images, "
                f"{sum(1 for r in results.values() if r.get('is_weed'))} weeds")
    logger.info(f"API usage: {identifier.get_usage_info()}")
    return results


def cache_hf_search():
    """Search HuggingFace for weed detection models and cache."""
    from .tools.model_discovery import ModelDiscovery
    discovery = ModelDiscovery()

    queries = ["weed detection", "weed segmentation", "crop weed"]
    results = {}
    for q in queries:
        models = discovery.search_huggingface(q, max_results=10)
        results[q] = models
        logger.info(f"HuggingFace '{q}': {len(models)} models")
    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-cache external API results")
    parser.add_argument("--images-dir", default=None,
                        help="Directory of images for plant.id")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Max images to identify (default 20, costs 20 credits)")
    parser.add_argument("--skip-plantid", action="store_true",
                        help="Skip plant.id caching")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace search caching")
    args = parser.parse_args()

    cache = load_cache()

    if not args.skip_plantid and args.images_dir:
        plant_results = cache_plant_id(args.images_dir, args.max_images)
        cache["plant_id"].update(plant_results)

    if not args.skip_hf:
        hf_results = cache_hf_search()
        cache["huggingface_search"].update(hf_results)

    save_cache(cache)
    logger.info(f"Cache: {len(cache['plant_id'])} plant IDs, "
                f"{len(cache['huggingface_search'])} HF searches")


if __name__ == "__main__":
    main()
