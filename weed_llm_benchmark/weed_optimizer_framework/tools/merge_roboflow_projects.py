"""
merge_roboflow_projects.py — Roboflow state audit + scaffold for pulling
verified annotations from the 12 cwd12-<species> projects and merging
into a multi-class YOLO dataset for cluster training.

This module is the cluster-side pull-back half of the active-learning
pipeline:

    Roboflow (per-species, human-verified bboxes, single-class cid=0)
                   │
                   ▼          ← THIS MODULE
    cwd12 multi-class YOLO dataset (cid remapped back to CWD12 index)
                   │
                   ▼
    cluster training (cwd12 mAP50-95 ≥ 0.90)

See `docs/roboflow_workspace.md` for the project layout.

Iter 5 scope (skeleton): audit each of 12 cwd12-<species> projects via
Roboflow REST API. Report image count, class count, last update. Output
JSON to results/framework/roboflow_state.json. NO data download yet —
that comes in a later iter once the SDK call for "export labeled data
as YOLO zip" is wired (Roboflow `Version` workflow).

SECURITY: ROBOFLOW_API_KEY from env only. Never logged in detail.

Wired as cluster_action 'roboflow_state_audit' (type=subprocess, ~10s).

Usage:
    export ROBOFLOW_API_KEY=...
    python -m weed_optimizer_framework.tools.merge_roboflow_projects
    python -m weed_optimizer_framework.tools.merge_roboflow_projects \\
        --out results/framework/roboflow_state.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
DEFAULT_OUT = REPO / "results" / "framework" / "roboflow_state.json"

WORKSPACE = "research-lhi4x"
CWD12 = (
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
)


def _key() -> str:
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not k:
        print("FATAL: ROBOFLOW_API_KEY not set in env", file=sys.stderr)
        sys.exit(2)
    return k


def _query_project(project_slug: str, key: str) -> dict:
    """REST API: GET /workspace/project — returns {project, versions, ...}.
    No SDK dependency for read-only audit."""
    import urllib.request
    url = f"https://api.roboflow.com/{WORKSPACE}/{project_slug}?api_key={key}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        return {"ok": True, "raw": data}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def audit_all_species(key: str) -> dict:
    rows: list = []
    totals = {"images": 0, "boxes": 0, "missing": 0}
    cwd12_full_too = {"slug": "cwd12-weeds", "purpose": "combined-multiclass"}
    # include the unified project at the top for context
    for slug in ["cwd12-weeds"] + [f"cwd12-{sp.lower()}" for sp in CWD12]:
        species = (slug.split("-", 1)[1].title()
                    if slug.startswith("cwd12-") and slug != "cwd12-weeds"
                    else None)
        # title() over a lowercase species: "carpetweeds" → "Carpetweeds"
        # but PalmerAmaranth and SpottedSpurge are camelcase. Map explicitly:
        if species:
            for sp in CWD12:
                if sp.lower() == slug.split("-", 1)[1]:
                    species = sp
                    break
        r = _query_project(slug, key)
        row = {"slug": slug, "species": species}
        if not r["ok"]:
            row.update({"status": "error", "error": r["error"]})
            totals["missing"] += 1
        else:
            p = (r["raw"].get("project") or {})
            classes = p.get("classes") or {}
            n_boxes = sum(classes.values()) if isinstance(classes, dict) else 0
            row.update({
                "status": "ok",
                "images": p.get("images", 0),
                "unannotated": p.get("unannotated", 0),
                "type": p.get("type"),
                "n_classes": len(classes) if isinstance(classes, dict) else None,
                "boxes_total": n_boxes,
                "boxes_per_class": classes if isinstance(classes, dict) else None,
                "versions": len((r["raw"].get("versions") or [])),
            })
            totals["images"] += row["images"]
            totals["boxes"] += row["boxes_total"]
        rows.append(row)
    return {"workspace": WORKSPACE, "rows": rows, "totals": totals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t0 = time.time()
    key = _key()
    audit = audit_all_species(key)
    audit["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    audit["elapsed_sec"] = round(time.time() - t0, 1)

    # Compact stdout summary
    print(f"\n=== Roboflow workspace audit ({audit['elapsed_sec']}s) ===")
    print(f"workspace: {WORKSPACE}")
    for r in audit["rows"]:
        if r["status"] == "ok":
            v = r.get("versions", 0)
            print(f"  {r['slug']:24s} imgs={r['images']:>5} "
                  f"boxes={r['boxes_total']:>6}  versions={v}")
        else:
            print(f"  {r['slug']:24s} ERROR: {r.get('error')}")
    print(f"\nTOTAL imgs={audit['totals']['images']}  "
          f"boxes={audit['totals']['boxes']}  "
          f"missing={audit['totals']['missing']}")

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    os.replace(tmp, out_path)
    print(f"\nWROTE: {out_path} ({out_path.stat().st_size} bytes)")

    # Roadmap for downstream merge (NOT implemented yet):
    print("""
NEXT STEPS (not yet implemented in this skeleton):
  • For each cwd12-<species> project: pull the human-verified annotations
    as YOLO txt. Roboflow exposes this via Versions — needs us to
    Generate a version then download (`roboflow` SDK
    Project.version(n).download('yolov8')).
  • Remap each downloaded label's class_id 0 → CWD12.index(<species>).
  • Merge into a single train/labels + train/images dataset under
    results/framework/datasets/cwd12_active_learning/.
  • Then feed into cluster training (sbatch).
""")


if __name__ == "__main__":
    main()
