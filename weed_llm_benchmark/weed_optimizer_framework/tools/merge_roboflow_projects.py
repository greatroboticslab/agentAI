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

# v3.0.59 (2026-05-30): workspace env-configurable. School account active
# (a-test-of-will); personal (research-lhi4x) kept as snapshot.
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "a-test-of-will")
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


def _list_workspace_projects(key: str) -> list:
    """v3.0.59: enumerate projects in current workspace via REST. Replaces
    the hardcoded 13-project list — works regardless of which workspace
    the key points at (a-test-of-will, research-lhi4x, future ones)."""
    import urllib.request
    url = f"https://api.roboflow.com/{WORKSPACE}?api_key={key}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
        return (d.get("workspace") or {}).get("projects", [])
    except Exception as e:
        print(f"[list_projects] {WORKSPACE}: {type(e).__name__}: {e}")
        return []


def audit_all_species(key: str) -> dict:
    """v3.0.59: workspace-aware. Lists every project in the workspace
    (not just our hardcoded list), then queries each for full details.
    Tags each project with its inferred role:
      - 'cwd12_master': our multi-class detection target
      - 'cwd12_species': legacy per-species (now removed in a-test-of-will)
      - 'other': pre-existing or unrelated projects in the workspace."""
    rows: list = []
    totals = {"images": 0, "boxes": 0, "missing": 0}

    projs = _list_workspace_projects(key)
    if not projs:
        return {"workspace": WORKSPACE, "rows": rows, "totals": totals,
                "error": "could not list workspace projects"}

    cwd12_species_lower = {sp.lower() for sp in CWD12}

    for p in projs:
        slug_full = p.get("id", "")
        # API returns 'a-test-of-will/<slug>' — drop the workspace prefix
        slug = slug_full.split("/", 1)[1] if "/" in slug_full else slug_full
        # Classify role
        if slug.lower() in ("cwd12-weeds", "cwd12-multiclass-v1"):
            role = "cwd12_master"
            species = None
        elif slug.startswith("cwd12-") and slug.split("-", 1)[1] in cwd12_species_lower:
            role = "cwd12_species"
            sp_lower = slug.split("-", 1)[1]
            species = next((sp for sp in CWD12 if sp.lower() == sp_lower), None)
        else:
            role = "other"
            species = None

        r = _query_project(slug, key)
        row = {"slug": slug, "slug_full": slug_full,
               "role": role, "species": species}
        if not r["ok"]:
            row.update({"status": "error", "error": r["error"]})
            totals["missing"] += 1
        else:
            pdata = r["raw"].get("project") or {}
            classes = pdata.get("classes") or {}
            n_boxes = sum(classes.values()) if isinstance(classes, dict) else 0
            row.update({
                "status": "ok",
                "images": pdata.get("images", 0),
                "unannotated": pdata.get("unannotated", 0),
                "type": pdata.get("type"),
                "n_classes": len(classes) if isinstance(classes, dict) else None,
                "boxes_total": n_boxes,
                "boxes_per_class": classes if isinstance(classes, dict) else None,
                "versions": len(r["raw"].get("versions") or []),
            })
            if role in ("cwd12_master", "cwd12_species"):
                totals["images"] += row["images"]
                totals["boxes"] += row["boxes_total"]
        rows.append(row)
    return {"workspace": WORKSPACE, "rows": rows, "totals": totals}


def cmd_audit(args):
    """Read-only workspace audit. Writes roboflow_state.json."""
    t0 = time.time()
    key = _key()
    audit = audit_all_species(key)
    audit["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    audit["elapsed_sec"] = round(time.time() - t0, 1)

    print(f"\n=== Roboflow workspace audit ({audit['elapsed_sec']}s) ===")
    print(f"workspace: {WORKSPACE}")
    for r in audit["rows"]:
        if r["status"] == "ok":
            print(f"  {r['slug']:24s} imgs={r['images']:>5} "
                  f"boxes={r['boxes_total']:>6}  versions={r.get('versions', 0)}")
        else:
            print(f"  {r['slug']:24s} ERROR: {r.get('error')}")
    print(f"\nTOTAL imgs={audit['totals']['images']}  "
          f"boxes={audit['totals']['boxes']}  "
          f"missing={audit['totals']['missing']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    os.replace(tmp, out_path)
    print(f"\nWROTE: {out_path} ({out_path.stat().st_size} bytes)")


def cmd_generate_versions(args):
    """Call Roboflow SDK to GENERATE a Version snapshot for one or more
    projects. Free-tier quota guard: skip if Versions already exist
    (unless --force).

    Targets, in priority order:
      1. --project NAME (explicit, single project) — added v3.0.61 after
         button-test loop iter 2 found the legacy CWD12-species iteration
         was hitting 12 dead per-species projects (deleted 2026-05-30).
      2. --species SPECIES (single legacy per-species project).
      3. Default: ['cwd12-multiclass-v1'] (the current master). Iterates
         CWD12 per-species names ONLY if explicitly opted in via
         --legacy-per-species.
    """
    key = _key()
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    ws = rf.workspace()

    if args.project:
        targets = [args.project]
    elif args.species:
        targets = [f"cwd12-{args.species.lower()}"]
    elif args.legacy_per_species:
        targets = [f"cwd12-{sp.lower()}" for sp in CWD12]
    else:
        targets = ["cwd12-multiclass-v1"]
    species_filter = targets  # naming kept for compat below

    state = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "requests": []}
    for entry in species_filter:
        # v3.0.61: entries now ARE the project names already (not species
        # names that need cwd12-<sp> prefix). Old per-species path uses
        # full project names from the legacy block above.
        proj_name = entry
        sp = entry  # reused for logging
        try:
            proj = ws.project(proj_name)
        except Exception as e:
            print(f"[skip] {proj_name}: project lookup fail {e}")
            state["requests"].append({"species": sp, "ok": False,
                                       "reason": str(e)})
            continue

        existing = []
        try:
            existing = proj.versions()
        except Exception:
            existing = []
        if existing and not args.force:
            print(f"[skip] {proj_name}: already has {len(existing)} version(s); "
                  f"--force to regenerate")
            state["requests"].append({"species": sp, "ok": True,
                                       "skipped": True,
                                       "existing_versions": len(existing)})
            continue

        try:
            v = proj.generate_version(settings={
                "augmentation": {}, "preprocessing": {},
            })
            print(f"[gen]  {proj_name}: generate_version → {v}")
            state["requests"].append({"species": sp, "ok": True,
                                       "version_id": str(v),
                                       "skipped": False})
        except Exception as e:
            print(f"[FAIL] {proj_name}: {type(e).__name__}: {e}")
            state["requests"].append({"species": sp, "ok": False,
                                       "error": f"{type(e).__name__}: {e}"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, out_path)
    print(f"\nWROTE: {out_path}")


def cmd_download_merge(args):
    """For each cwd12-<species> project with a Version: download YOLO zip,
    remap labels (cid 0 → CWD12.index(species)), and merge into a single
    multi-class YOLO dataset suitable for cluster training.

    Output layout:
      results/framework/datasets/cwd12_active_learning/
        train/images/<species>_<orig_name>.jpg
        train/labels/<species>_<orig_name>.txt
        data.yaml

    Splits: we put everything from Roboflow into TRAIN. The training-time
    eval uses cottonweeddet12/valid + test (untouched holdout). This
    ensures the active-learning loop never contaminates evaluation.
    """
    key = _key()
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    ws = rf.workspace()

    species_filter = (args.species,) if args.species else CWD12
    out_root = Path(args.out_dir)
    img_dir = out_root / "train" / "images"
    lbl_dir = out_root / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    import zipfile, shutil
    cwd12_index = {sp: i for i, sp in enumerate(CWD12)}

    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "per_species": []}
    for sp in species_filter:
        proj_name = f"cwd12-{sp.lower()}"
        try:
            proj = ws.project(proj_name)
            vs = proj.versions()
        except Exception as e:
            print(f"[skip] {proj_name}: {type(e).__name__}: {e}")
            summary["per_species"].append({"species": sp, "ok": False,
                                            "error": str(e)})
            continue
        if not vs:
            print(f"[skip] {proj_name}: no versions yet — run generate-versions first")
            summary["per_species"].append({"species": sp, "ok": False,
                                            "reason": "no_versions"})
            continue

        v = vs[-1]  # latest
        print(f"[download] {proj_name} version {v.version} ...")
        try:
            dl = v.download("yolov8", location=str(out_root / "_dl" / sp))
        except Exception as e:
            print(f"[FAIL download] {proj_name}: {type(e).__name__}: {e}")
            summary["per_species"].append({"species": sp, "ok": False,
                                            "error": f"download: {e}"})
            continue

        loc = Path(getattr(dl, "location", "") or out_root / "_dl" / sp)
        if not loc.is_dir():
            print(f"[FAIL] download location not found: {loc}")
            summary["per_species"].append({"species": sp, "ok": False,
                                            "error": "no_location"})
            continue

        cid_target = cwd12_index[sp]
        n_imgs = 0; n_lbls = 0
        # Roboflow yolov8 layout: train/images, train/labels, valid/, test/, data.yaml
        for split in ("train", "valid", "test"):
            si = loc / split / "images"
            sl = loc / split / "labels"
            if not si.is_dir() or not sl.is_dir():
                continue
            for img in si.iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl = sl / (img.stem + ".txt")
                new_stem = f"{sp}_{img.stem}"
                new_img = img_dir / (new_stem + img.suffix)
                new_lbl = lbl_dir / (new_stem + ".txt")
                try:
                    shutil.copy2(img, new_img)
                except Exception:
                    continue
                n_imgs += 1
                # Remap label cid 0 → cid_target
                lines_out = []
                if lbl.is_file():
                    try:
                        for line in lbl.read_text(errors="ignore").splitlines():
                            p = line.split()
                            if p and p[0].lstrip("-").isdigit():
                                p[0] = str(cid_target)
                                lines_out.append(" ".join(p))
                    except Exception:
                        pass
                with open(new_lbl, "w") as f:
                    f.write("\n".join(lines_out))
                n_lbls += 1

        print(f"  [{sp}] images={n_imgs} labels={n_lbls}")
        summary["per_species"].append({"species": sp, "ok": True,
                                        "images": n_imgs, "labels": n_lbls,
                                        "cid_target": cid_target})

    # Write data.yaml
    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("# auto-generated by merge_roboflow_projects download-merge\n")
        f.write(f"nc: {len(CWD12)}\n")
        f.write("names: [" + ", ".join(repr(c) for c in CWD12) + "]\n")
        f.write("train: train/images\n")
        f.write("val: train/images   # placeholder — real eval uses cottonweeddet12/valid\n")

    out_state = out_root / "_merge_summary.json"
    with open(out_state, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWROTE: {out_state}")
    print(f"      dataset: {out_root}")
    print(f"      data.yaml: {data_yaml}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p_audit = sub.add_parser("audit", help="read-only workspace audit")
    p_audit.add_argument("--out", default=str(DEFAULT_OUT))

    p_gen = sub.add_parser("generate-versions",
                            help="trigger Roboflow Version generation")
    p_gen.add_argument("--project", default="",
                        help="single project name (e.g. cwd12-multiclass-v1)")
    p_gen.add_argument("--species", default="",
                        help="legacy: a single CWD12 species → cwd12-<species>")
    p_gen.add_argument("--legacy-per-species", action="store_true",
                        help="legacy: iterate all 12 cwd12-<species> projects (deleted 2026-05-30)")
    p_gen.add_argument("--force", action="store_true",
                        help="re-generate even if Versions exist")
    p_gen.add_argument("--out",
                        default=str(REPO / "results" / "framework" / "roboflow_versions.json"))

    p_dl = sub.add_parser("download-merge",
                           help="download labeled YOLO data and merge to multi-class")
    p_dl.add_argument("--species", default="")
    p_dl.add_argument("--out-dir",
                       default=str(REPO / "results" / "framework" / "datasets" / "cwd12_active_learning"))

    # Backward compat: bare `--out ...` with no subcommand → treat as audit.
    ap.add_argument("--out", default=None,
                    help="(legacy) implies audit cmd if used without subcommand")
    args = ap.parse_args()

    if args.cmd is None:
        # legacy invocation: roboflow_state_audit action runs without subcmd
        ns = argparse.Namespace(out=args.out or str(DEFAULT_OUT))
        cmd_audit(ns)
        return

    {"audit": cmd_audit,
     "generate-versions": cmd_generate_versions,
     "download-merge": cmd_download_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
