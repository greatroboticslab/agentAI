#!/usr/bin/env python3
"""Quick test: verify plant.id API works on our weed images.

Uses 5 images from CottonWeedDet12 test set (costs 5 credits out of 50).
"""

import os
import sys
import json

sys.path.insert(0, ".")
from weed_optimizer_framework.tools.web_identifier import WebIdentifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES = os.path.join(BASE_DIR, "results", "leave4out", "dataset_8species", "test", "images")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "plantid_test.json")


def main():
    api_key = os.environ.get("PLANT_API_KEY", "")
    if not api_key:
        print("ERROR: Set PLANT_API_KEY environment variable")
        sys.exit(1)

    identifier = WebIdentifier(api_key=api_key)
    print(f"plant.id API ready. Credits: {identifier.usage_limit}")

    # Test with 5 images
    results = identifier.identify_batch(TEST_IMAGES, max_images=5, use_api=True)

    print("\n=== Results ===")
    for stem, r in results.items():
        species = r.get("species", "?")
        conf = r.get("confidence", 0)
        is_weed = r.get("is_weed", False)
        print(f"  {stem}: {species} ({conf:.1%}) weed={is_weed}")
        if r.get("all_suggestions"):
            for s in r["all_suggestions"][:3]:
                print(f"    -> {s['species']} ({s['confidence']:.1%})")

    print(f"\nAPI usage: {identifier.get_usage_info()}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump({"results": results, "usage": identifier.get_usage_info()}, f, indent=2)
    print(f"Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
