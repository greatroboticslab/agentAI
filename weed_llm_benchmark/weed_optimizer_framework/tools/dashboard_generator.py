"""
v3.0.30 — Public dashboard generator (static HTML for GitHub Pages).

Pulls live state from:
  - results/framework/dataset_registry.json        (slug metadata)
  - results/framework/jobd_runs/*.json             (per-Job-D run summaries)
  - results/framework/*pycoco*summary.json         (canonical mAP results)
  - results/framework/v3_0_29_curated_imgs.json    (green-pixel curate stats)

Produces:
  docs/dashboard/index.html     — landing page: totals + growth chart
  docs/dashboard/datasets.html  — searchable table of all slugs
  docs/dashboard/categories.html — class distribution + crop types
  docs/dashboard/progress.html  — mAP timeline + accuracy vs scale
  docs/dashboard/data.json      — full JSON for client-side reuse

GitHub Pages serves docs/ at /agentAI/ — accessible publicly without auth.

Pure stdlib (no Flask/Jinja). Self-contained HTML with inline CSS + minimal JS.
"""
from __future__ import annotations

import argparse
import glob
import html as html_lib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_12 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge",
                "Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]

# Source detection from slug prefix
def slug_source(slug: str) -> str:
    if slug.startswith("kg_"):    return "Kaggle"
    if slug.startswith("gh_"):    return "GitHub"
    if slug.startswith("hf_"):    return "HuggingFace"
    if slug.startswith("francesco"): return "Roboflow"
    if slug in ("cottonweeddet12", "cottonweed_sp8", "cottonweed_holdout",
                "weedsense", "deepweeds", "crop_weed_research", "grass_weeds",
                "weed_crop_aerial", "rice_weeds_ph", "weeds7kpd"):
        return "Curated (seed)"
    return "Other"


# Crop / class detection from slug + description
CROP_KEYWORDS = {
    "Cotton":     ["cotton", "cottonweed"],
    "Rice":       ["rice", "padi"],
    "Wheat":      ["wheat"],
    "Corn/Maize": ["corn", "maize"],
    "Potato":     ["potato"],
    "Tomato":     ["tomato"],
    "Sugar beet": ["sugar beet", "sugarbeet"],
    "Lettuce":    ["lettuce"],
    "Soybean":    ["soybean", "soy"],
    "Coconut":    ["coconut"],
    "Guava":      ["guava"],
    "Mixed crops":["crop", "agriculture", "farmland", "field"],
    "Plant disease (general)": ["plantdisease", "plantvillage", "plant-village", "leaf-disease", "leaf disease", "plant disease"],
    "Pest/Insect":["pest", "insect", "bug", "weevil"],
    "Generic weed":["weed-detection", "weed_detection", "weeddet", "weed-yolo"],
    "Non-target (drop)":["recycling", "waste", "beehive", "warp"],
}

def slug_crop(slug: str, info: dict) -> list[str]:
    blob = (slug + " " + (info.get("description", "") or "") + " " +
            (info.get("name", "") or "")).lower()
    hits = []
    for crop, kws in CROP_KEYWORDS.items():
        if any(k in blob for k in kws):
            hits.append(crop)
    return hits or ["Unclassified"]


def annotation_type_human(ann: str) -> str:
    return {
        "bbox": "Real bbox (human)",
        "bbox+segmentation": "Real bbox+seg (human)",
        "yolo": "Real YOLO (human)",
        "yolo_autolabel": "AI-labeled (OWLv2)",
        "classification": "Classification only (no bbox)",
    }.get(ann or "", ann or "unknown")


def page_template(title: str, body: str, nav_active: str = "") -> str:
    nav_items = [("index", "Home"), ("datasets", "Datasets"),
                 ("categories", "Categories"), ("progress", "Progress")]
    nav_html = " · ".join(
        f'<a href="{n}.html" class="{ "active" if n == nav_active else ""}">{label}</a>'
        for n, label in nav_items
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_lib.escape(title)} — Autonomous Weed Detection</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  nav {{ background: #fafafa; padding: 0.7rem 1rem; border-radius: 8px;
         margin-bottom: 1.5rem; font-size: 0.95rem; }}
  nav a {{ text-decoration: none; color: #444; margin-right: 0.5rem; }}
  nav a.active {{ font-weight: 600; color: #2a7; }}
  h1 {{ margin-top: 0.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; cursor: pointer; }}
  tr:hover {{ background: #fafafa; }}
  .stat {{ display: inline-block; background: #f3f9f5; padding: 0.6rem 1rem;
           border-radius: 6px; margin-right: 0.7rem; margin-bottom: 0.6rem; }}
  .stat b {{ font-size: 1.4rem; color: #2a7; }}
  .badge {{ display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px;
            font-size: 0.78rem; margin-right: 0.2rem; }}
  .badge-real {{ background: #d4f0d4; color: #285; }}
  .badge-ai {{ background: #ffe5b4; color: #864; }}
  .badge-cls {{ background: #eee; color: #777; }}
  .badge-drop {{ background: #ffd4d4; color: #844; }}
  .small {{ color: #888; font-size: 0.85rem; }}
  input.search {{ padding: 0.4rem 0.6rem; font-size: 0.95rem;
                  border: 1px solid #ccc; border-radius: 4px; width: 320px; }}
  footer {{ margin-top: 3rem; font-size: 0.8rem; color: #888;
            border-top: 1px solid #eee; padding-top: 1rem; }}
</style></head><body>
<nav>{nav_html} <span class="small">— autonomous weed detection · greatroboticslab</span></nav>
<h1>{html_lib.escape(title)}</h1>
{body}
<footer>Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} ·
  cwd12 holdout test set (1,977 hand-labeled imgs) is the NEVER_TRAIN evaluation reference ·
  <a href="https://github.com/greatroboticslab/agentAI">repo</a></footer>
</body></html>"""


def build_index(state: dict) -> str:
    s = state
    total_slugs = s["totals"]["slugs"]
    real_slugs = s["totals"]["real_bbox_slugs"]
    autolabel_slugs = s["totals"]["autolabel_slugs"]
    total_imgs = s["totals"]["downloaded_imgs"]
    crops = s["crop_counts"]
    sources = s["source_counts"]

    pyco = s.get("latest_pyco")
    pyco_block = ""
    if pyco:
        pyco_block = f"""
        <div class="stat"><b>{pyco['mAP50_95']:.4f}</b><br><span class="small">latest cwd12 mAP50-95<br>(pycocotools, {pyco['model_label']})</span></div>
        <div class="stat"><b>{pyco['mAP50']:.4f}</b><br><span class="small">mAP50</span></div>
        """

    crop_rows = "".join(
        f"<tr><td>{html_lib.escape(c)}</td><td>{n}</td></tr>"
        for c, n in sorted(crops.items(), key=lambda kv: -kv[1])
    )
    source_rows = "".join(
        f"<tr><td>{html_lib.escape(c)}</td><td>{n}</td></tr>"
        for c, n in sorted(sources.items(), key=lambda kv: -kv[1])
    )

    body = f"""
<p>This dashboard tracks an <b>autonomous data-discovery pipeline</b> for weed detection.
A "Brain" agent searches Kaggle, HuggingFace, GitHub, and Roboflow for relevant
datasets; downloads what it finds; auto-labels via OWLv2 where ground-truth boxes
are unavailable; and feeds everything into a continually-growing training corpus.
Concurrently a training pipeline finetunes YOLO/RF-DETR models on the corpus and
evaluates on a <b>fixed human-labeled holdout (cwd12 test+valid, 1977 imgs)</b>
that <b>never enters training</b>.</p>

<h2>Current scale</h2>
<div>
  <div class="stat"><b>{total_slugs}</b><br><span class="small">datasets discovered</span></div>
  <div class="stat"><b>{total_imgs:,}</b><br><span class="small">raw images on disk</span></div>
  <div class="stat"><b>{real_slugs}</b><br><span class="small">slugs with real bboxes</span></div>
  <div class="stat"><b>{autolabel_slugs}</b><br><span class="small">slugs autolabeled (OWLv2)</span></div>
  {pyco_block}
</div>

<h2>Crops & domain coverage</h2>
<table><thead><tr><th>Crop / topic</th><th>#slugs</th></tr></thead>
<tbody>{crop_rows}</tbody></table>

<h2>Sources</h2>
<table><thead><tr><th>Source</th><th>#slugs</th></tr></thead>
<tbody>{source_rows}</tbody></table>

<h2>How autonomy works</h2>
<ul>
<li><b>Brain LLM</b> (Gemma 4 via Ollama on cluster) writes search queries on each
  iteration and parses dataset metadata returned by Kaggle / HuggingFace APIs.</li>
<li><b>Quality filters</b>: cross-dataset dHash dedup → green-pixel HSV check →
  per-image stem-level holdout filter (defense against test-set leak) → OWLv2
  confidence threshold for autolabel.</li>
<li><b>NEVER_TRAIN</b>: cwd12 test+valid (1,977 imgs) is guarded at slug-level
  AND per-image stem-level. After the v3.0.27 incident where two slugs holding
  cwd12 copies bypassed the slug-level guard, the per-image stem filter was
  added as defense in depth.</li>
<li><b>Job-D continuous</b>: self-chaining SLURM job (afterany dependency)
  runs harvest+autolabel for months at a time. Idle guard pauses when Brain
  finds zero new slugs across 20 consecutive iterations.</li>
</ul>
"""
    return page_template("Home", body, nav_active="index")


def build_datasets(state: dict) -> str:
    rows = []
    for d in state["datasets_for_table"]:
        label_badge = {
            "Real bbox (human)":        '<span class="badge badge-real">real bbox</span>',
            "Real bbox+seg (human)":    '<span class="badge badge-real">real bbox+seg</span>',
            "Real YOLO (human)":        '<span class="badge badge-real">real YOLO</span>',
            "AI-labeled (OWLv2)":       '<span class="badge badge-ai">AI label</span>',
            "Classification only (no bbox)": '<span class="badge badge-cls">cls only</span>',
        }.get(d["annotation_h"], f'<span class="badge">{html_lib.escape(d["annotation_h"])}</span>')
        crops = " ".join(f'<span class="badge">{html_lib.escape(c)}</span>'
                         for c in d["crops"])
        never = ' <span class="badge badge-drop">NEVER_TRAIN</span>' if d["never_train"] else ""
        rows.append(
            f'<tr><td><code>{html_lib.escape(d["slug"])}</code>{never}</td>'
            f'<td>{html_lib.escape(d["source"])}</td>'
            f'<td>{d["n_imgs"]:,}</td>'
            f'<td>{label_badge}</td>'
            f'<td>{crops}</td>'
            f'<td class="small">{html_lib.escape(d["description"][:120])}</td></tr>'
        )

    body = f"""
<p><span class="small">All slugs Brain has discovered. Real-bbox slugs feed training directly;
AI-labeled slugs go to auxiliary class slots; classification-only slugs are not used
unless they pass autolabel. NEVER_TRAIN slugs (cwd12, weedsense, francesco) are blocked
at merge time. Click column headers to sort.</span></p>
<p><input type="text" id="q" class="search" placeholder="search slug / crop / source…"
   onkeyup="filter()"></p>
<table id="t"><thead>
  <tr><th>Slug</th><th>Source</th><th>#imgs</th><th>Annotation</th>
      <th>Crop / domain</th><th>Description</th></tr>
</thead><tbody>
{"".join(rows)}
</tbody></table>
<script>
function filter() {{
  var q = document.getElementById('q').value.toLowerCase();
  var rows = document.querySelectorAll('#t tbody tr');
  for (var r of rows) {{ r.style.display = r.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none'; }}
}}
</script>
"""
    return page_template("Datasets", body, nav_active="datasets")


def build_categories(state: dict) -> str:
    crops = state["crop_counts"]
    sources = state["source_counts"]
    ann_counts = state["annotation_counts"]

    crop_rows = "".join(f"<tr><td>{html_lib.escape(c)}</td><td>{n}</td></tr>"
                        for c, n in sorted(crops.items(), key=lambda kv: -kv[1]))
    source_rows = "".join(f"<tr><td>{html_lib.escape(c)}</td><td>{n}</td></tr>"
                          for c, n in sorted(sources.items(), key=lambda kv: -kv[1]))
    ann_rows = "".join(f"<tr><td>{html_lib.escape(c)}</td><td>{n}</td></tr>"
                       for c, n in sorted(ann_counts.items(), key=lambda kv: -kv[1]))

    twelve = state.get("twelve_class_gt", {})
    twelve_rows = "".join(
        f"<tr><td>{html_lib.escape(c)}</td><td>{n}</td></tr>"
        for c in CANONICAL_12 for n in [twelve.get(c, 0)]
    )

    body = f"""
<h2>Annotation type</h2>
<table><thead><tr><th>Type</th><th>#slugs</th></tr></thead><tbody>{ann_rows}</tbody></table>

<h2>Crop / topic</h2>
<table><thead><tr><th>Crop / topic</th><th>#slugs</th></tr></thead><tbody>{crop_rows}</tbody></table>

<h2>Source</h2>
<table><thead><tr><th>Source</th><th>#slugs</th></tr></thead><tbody>{source_rows}</tbody></table>

<h2>12 cottonweed classes — ground-truth count in cwd12 holdout (1977 imgs)</h2>
<table><thead><tr><th>Class</th><th>#instances</th></tr></thead><tbody>{twelve_rows}</tbody></table>
"""
    return page_template("Categories", body, nav_active="categories")


def build_progress(state: dict) -> str:
    runs = state.get("jobd_runs", [])
    run_rows = "".join(
        f'<tr><td>{html_lib.escape(str(r["slurm_job_id"]))}</td>'
        f'<td>{r["iters"]}</td>'
        f'<td>{r["total_new_slugs"]}</td>'
        f'<td>{r["elapsed_h"]:.1f}h</td>'
        f'<td>{"yes" if r.get("exhausted") else "no"}</td></tr>'
        for r in runs
    )

    pyco_history = state.get("pyco_history", [])
    pyco_rows = "".join(
        f'<tr><td>{html_lib.escape(p["label"])}</td>'
        f'<td>{p["mAP50_95"]:.4f}</td>'
        f'<td>{p["mAP50"]:.4f}</td></tr>'
        for p in pyco_history
    )

    body = f"""
<h2>Goal</h2>
<p>Research goal: <b>cwd12 mAP50-95 ≥ 0.90 (pycocotools canonical)</b>.
Above the v3.0.6 published baseline (~0.71 pyco / 0.865 ultralytics) and the
2026 DINOv3+YOLO26 in-domain published result (0.723 pyco).</p>

<h2>Canonical mAP history (pycocotools)</h2>
<table><thead><tr><th>Model / run</th><th>mAP50-95</th><th>mAP50</th></tr></thead>
<tbody>{pyco_rows}</tbody></table>

<h2>Job-D continuous harvest runs</h2>
<p><span class="small">Each row is one 48h Job-D iteration. "new_slugs" is how many
NEW dataset slugs Brain discovered in that window. Goal: never-ending growth.</span></p>
<table><thead><tr><th>SLURM job</th><th>iters</th><th>new_slugs</th><th>elapsed</th><th>exhausted</th></tr></thead>
<tbody>{run_rows}</tbody></table>
"""
    return page_template("Progress", body, nav_active="progress")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir",
                    default="results/framework",
                    help="dir containing pycoco summaries + jobd_runs/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.registry) as f:
        reg = json.load(f)
    datasets = reg.get("datasets", {})

    NEVER_TRAIN = {"cottonweeddet12", "weedsense", "francesco__weed_crop_aerial"}

    crop_counts = Counter()
    source_counts = Counter()
    annotation_counts = Counter()
    real_slugs = 0
    autolabel_slugs = 0
    downloaded_imgs = 0
    table_rows = []
    for slug, info in datasets.items():
        if not isinstance(info, dict):
            continue
        ann = info.get("annotation") or info.get("status", "")
        ann_h = annotation_type_human(ann)
        annotation_counts[ann_h] += 1
        source = slug_source(slug)
        source_counts[source] += 1
        crops = slug_crop(slug, info)
        for c in crops:
            crop_counts[c] += 1
        n_imgs = (info.get("unique_imgs") or info.get("n_images")
                  or info.get("img_count") or 0)
        if not isinstance(n_imgs, int):
            try: n_imgs = int(n_imgs)
            except: n_imgs = 0
        downloaded_imgs += n_imgs
        if "real" in ann_h.lower() or "human" in ann_h.lower():
            real_slugs += 1
        if "AI-labeled" in ann_h:
            autolabel_slugs += 1
        table_rows.append({
            "slug": slug,
            "source": source,
            "n_imgs": n_imgs,
            "annotation_h": ann_h,
            "crops": crops,
            "description": info.get("description", "") or "",
            "never_train": slug in NEVER_TRAIN,
        })
    table_rows.sort(key=lambda r: -r["n_imgs"])

    # Find latest pycoco summary
    latest_pyco = None
    pyco_history = []
    for p in sorted(glob.glob(f"{args.results_dir}/*pycoco*summary.json"),
                    key=os.path.getmtime):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        label = Path(p).stem.replace("_pycoco_summary", "").replace("v3_0_29_", "")
        entry = {
            "label": label,
            "mAP50_95": float(d.get("mAP50_95", 0)),
            "mAP50": float(d.get("mAP50", 0)),
            "model_label": label,
        }
        pyco_history.append(entry)
        latest_pyco = entry

    # Per-Job-D run summaries
    jobd_runs = []
    for p in sorted(glob.glob(f"{args.results_dir}/jobd_runs/*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            jobd_runs.append(d)
        except Exception:
            pass

    # 12-class GT count in cwd12 holdout (from pycoco_gt.json if available)
    twelve_class_gt = {}
    gt_paths = sorted(glob.glob(f"{args.results_dir}/*pycoco_gt.json"),
                      key=os.path.getmtime, reverse=True)
    if gt_paths:
        try:
            with open(gt_paths[0]) as f:
                gt = json.load(f)
            for c in gt.get("categories", []):
                twelve_class_gt[c["name"]] = 0
            for a in gt.get("annotations", []):
                cid = a["category_id"]
                if 0 <= cid < len(CANONICAL_12):
                    twelve_class_gt[CANONICAL_12[cid]] = twelve_class_gt.get(CANONICAL_12[cid], 0) + 1
        except Exception:
            pass

    state = {
        "totals": {
            "slugs": len(datasets),
            "real_bbox_slugs": real_slugs,
            "autolabel_slugs": autolabel_slugs,
            "downloaded_imgs": downloaded_imgs,
        },
        "crop_counts": dict(crop_counts),
        "source_counts": dict(source_counts),
        "annotation_counts": dict(annotation_counts),
        "datasets_for_table": table_rows,
        "latest_pyco": latest_pyco,
        "pyco_history": pyco_history,
        "jobd_runs": jobd_runs,
        "twelve_class_gt": twelve_class_gt,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    (out_dir / "data.json").write_text(json.dumps(state, indent=1))
    (out_dir / "index.html").write_text(build_index(state))
    (out_dir / "datasets.html").write_text(build_datasets(state))
    (out_dir / "categories.html").write_text(build_categories(state))
    (out_dir / "progress.html").write_text(build_progress(state))
    print(f"[dashboard] wrote 4 pages + data.json → {out_dir}")
    print(f"[dashboard]   slugs={len(datasets)} imgs={downloaded_imgs} "
          f"crops={len(crop_counts)} sources={len(source_counts)}")


if __name__ == "__main__":
    main()
