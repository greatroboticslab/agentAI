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

try:
    from .cwd12_species import (CWD12_SPECIES, CWD12_LEGACY_LABELS,
                                species_of, legacy_to_species,
                                uploaded_label_species)
except ImportError:  # run as a plain script from tools/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cwd12_species import (CWD12_SPECIES, CWD12_LEGACY_LABELS,
                               species_of, legacy_to_species,
                               uploaded_label_species)

# v3.60.0: the merged set is written in cwd12 id space, named by species. The
# old table joined each Roboflow class name to the invented alphabetical list
# by string, so a person's real 'Ragweed' box went to id 9 (Sicklepod),
# 'Crabgrass' to id 1 (MorningGlory) and waterhemp was dropped.


def _read_yaml_names(loc: Path) -> list:
    names = []
    dy = loc / "data.yaml"
    if dy.is_file():
        try:
            import yaml  # ultralytics env ships pyyaml
            d = yaml.safe_load(dy.read_text(errors="ignore")) or {}
            nm = d.get("names")
            if isinstance(nm, dict):
                names = [nm[k] for k in sorted(nm, key=lambda x: int(x))]
            elif isinstance(nm, list):
                names = nm
        except Exception:
            pass
    return names


def _multiclass_remap(names):
    """{project class id: cwd12 id} for the boxes a person drew, and
    {unmapped name: class id}. v3.60.0: every name is read as a real name
    (species_of), in legacy projects too: the boxes that carry legacy labels
    are the ones our uploader wrote on cwd12 photographs, and those images
    are dropped before this map is used (cwd12_photo_index). On any other
    photograph roboflow_sync writes names that read back through species_of
    (roboflow_sync._class_name). A name that is not a cwd12 species is not
    joined to any id."""
    remap, unknown = {}, {}
    for cid, nm in enumerate(names):
        sp = species_of(nm)
        if sp is None:
            unknown[str(nm)] = cid
        else:
            remap[cid] = CWD12_SPECIES.index(sp)
    return remap, unknown


def _build_multiclass_remap(loc: Path):
    """Read a downloaded Roboflow yolov8 data.yaml `names` and map each project
    class-id → cwd12 id by species. Returns (remap {src_cid: cwd12 id},
    names list, unmapped names list)."""
    names = _read_yaml_names(loc)
    remap, unknown = _multiclass_remap(names)
    return remap, names, list(unknown)


# ---------------------------------------------------------------------------
# v3.60.0: cwd12 photographs never come back from Roboflow. The holdout must
# never train, and a cwd12 train photograph's labels come from the verified
# cwd12 copies (boxes our uploader wrote on it carry legacy labels, which a
# name lookup would read as other plants). A downloaded image within
# near_dup.NEAR_DUP_BITS of either (HOLDOUT_NEAR_DUP_BITS of a holdout image)
# is dropped, by mega_trainer._dhash.
# ---------------------------------------------------------------------------
_CWD12_ROOTS = (REPO / "downloads" / "cottonweeddet12",
                Path("downloads") / "cottonweeddet12")
_DHASH_CACHE = REPO / "results" / "framework" / "roboflow_cwd12_dhash_cache.json"


def _cwd12_ref_images():
    """(kind, path) for every cwd12 holdout and train image on this machine;
    kind is 'holdout' or 'cwd12_train'. Deduplicated by resolved path."""
    from .mega_trainer import _iter_holdout_images, _cwd12_train_label_dir
    seen = set()

    def _take(kind, p):
        rp = str(p.resolve())
        if rp in seen:
            return None
        seen.add(rp)
        return kind, p

    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
    for p in _iter_holdout_images():
        t = _take("holdout", p)
        if t:
            yield t
    for root in _CWD12_ROOTS:
        for split in ("test", "valid"):
            d = root / split / "images"
            if d.is_dir():
                for ext in exts:
                    for p in d.glob(ext):
                        t = _take("holdout", p)
                        if t:
                            yield t
    train_dirs = [root / "train" / "images" for root in _CWD12_ROOTS]
    ld = _cwd12_train_label_dir()
    if ld is not None:
        train_dirs.append(ld.parent / "images")
    for d in train_dirs:
        if d.is_dir():
            for ext in exts:
                for p in d.glob(ext):
                    t = _take("cwd12_train", p)
                    if t:
                        yield t


def cwd12_photo_index(cache_path: Path = _DHASH_CACHE):
    """NearHashIndex of every cwd12 holdout + train photograph (owner = kind).

    Raises RuntimeError when no holdout or no train image is found: without
    them nothing stops a holdout photograph from reaching training. Hashes are
    cached by (path, size, mtime) in `cache_path`."""
    from .mega_trainer import _dhash
    from .near_dup import NearHashIndex
    cache = {}
    try:
        cache = json.load(open(cache_path))
    except Exception:
        cache = {}
    idx = NearHashIndex()
    counts = {"holdout": 0, "cwd12_train": 0}
    dirty = False
    for kind, p in _cwd12_ref_images():
        try:
            st = p.stat()
        except OSError:
            continue
        key = str(p.resolve())
        sig = [st.st_size, int(st.st_mtime)]
        rec = cache.get(key)
        if isinstance(rec, list) and len(rec) == 3 and rec[:2] == sig:
            h = rec[2]
        else:
            h = _dhash(p)
            if h is None:
                continue
            cache[key] = sig + [h]
            dirty = True
        idx.add(h, kind)
        counts[kind] += 1
    if not counts["holdout"] or not counts["cwd12_train"]:
        raise RuntimeError(
            f"cwd12 photographs not found (holdout={counts['holdout']}, "
            f"train={counts['cwd12_train']}; looked under "
            f"{[str(r) for r in _CWD12_ROOTS]}). Refusing to build a training "
            f"set that cannot be checked for holdout photographs.")
    if dirty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(cache, f)
            os.replace(tmp, cache_path)
        except Exception:
            pass
    print(f"[cwd12-guard] {counts['holdout']} holdout + {counts['cwd12_train']} "
          f"train photographs indexed (near = within NEAR_DUP_BITS, "
          f"HOLDOUT_NEAR_DUP_BITS for holdout)")
    return idx


def cwd12_photo_kind(index, img_path):
    """'holdout' / 'cwd12_train' when img_path is a cwd12 photograph, else None.
    An image that cannot be hashed counts as 'unhashable' and is dropped too."""
    from .mega_trainer import _dhash
    h = _dhash(img_path)
    if h is None:
        return "unhashable"
    hits = index.matches(h)
    if not hits:
        return None
    kinds = {k for k, _bits in hits}
    return "holdout" if "holdout" in kinds else "cwd12_train"

# v3.0.91: OUR-projects ALLOW-LIST. The workspace contains many unrelated
# projects (drone/hardhat/demo/…). The dashboard must show ONLY our pipeline's
# projects. Source of truth = Mongo/DB; Roboflow is just a labeling surface, so
# we do NOT scope by Roboflow folders — we keep an explicit allow-list here.
# Resolution: env AGENTAI_RF_PROJECTS (comma-sep) → config json → default.
# See memory/feedback_roboflow_folder_scope.md.
_RF_PROJECTS_CONFIG = REPO / "results" / "framework" / "roboflow_projects.json"
_DEFAULT_RF_PROJECTS = [
    "cwd12-multiclass-v1",        # 598-img human-labeled train gold
    "weed-crop-agent-dataset",    # agent harvest pool (v1)
    "weed-crop-agent-v2",
    "weed-crop-agent-v3",
    # v3.0.99.17: round-N projects must be in the allow-list or the dashboard's
    # /api/roboflow_status undercounts (v4 held the bulk Universe upload — 2175+
    # imgs — but was invisible because it wasn't listed). Cover future rounds too.
    "weed-crop-agent-v4",
    "weed-crop-agent-v5",
    "weed-crop-agent-v6",
    "weed-crop-agent-v7",
    "weed-crop-agent-v8",
]


def our_projects() -> set:
    env = os.environ.get("AGENTAI_RF_PROJECTS", "").strip()
    if env:
        return {s.strip() for s in env.split(",") if s.strip()}
    try:
        if _RF_PROJECTS_CONFIG.is_file():
            d = json.load(open(_RF_PROJECTS_CONFIG))
            if isinstance(d, list) and d:
                return set(d)
            if isinstance(d, dict) and d.get("projects"):
                return set(d["projects"])
    except Exception:
        pass
    return set(_DEFAULT_RF_PROJECTS)


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

    # v3.0.91: keep ONLY our pipeline's projects (allow-list). Unrelated
    # workspace projects (drone/hardhat/demo/…) are dropped before querying.
    allow = our_projects()
    # v3.60.0: per-species projects are cwd12-<species> (create-species-projects);
    # the legacy-named ones were deleted 2026-05-30.
    cwd12_species_lower = {sp.lower(): sp for sp in CWD12_SPECIES}

    for p in projs:
        slug_full = p.get("id", "")
        # API returns 'a-test-of-will/<slug>' — drop the workspace prefix
        slug = slug_full.split("/", 1)[1] if "/" in slug_full else slug_full
        if slug not in allow:
            continue  # not ours → never show / never query
        # Classify role
        if slug.lower() in ("cwd12-weeds", "cwd12-multiclass-v1"):
            role = "cwd12_master"
            species = None
        elif slug.startswith("weed-crop-agent"):
            role = "agent"
            species = None
        elif slug.startswith("cwd12-") and slug.split("-", 1)[1] in cwd12_species_lower:
            role = "cwd12_species"
            species = cwd12_species_lower[slug.split("-", 1)[1]]
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
            # v3.60.0: species of each class name as our uploader wrote it
            # (legacy labels in LEGACY_ROBOFLOW_PROJECTS). A class a person
            # created or drew into may hold another plant; counts only.
            if isinstance(classes, dict):
                cs = {c: uploaded_label_species(slug, c) for c in classes}
                row["class_species"] = cs
                bps = {}
                for c, n in classes.items():
                    if cs[c] is not None:
                        bps[cs[c]] = bps.get(cs[c], 0) + (n or 0)
                row["boxes_per_species"] = bps
            # v3.0.91: all kept rows are ours (allow-listed) → count all.
            totals["images"] += row["images"]
            totals["boxes"] += row["boxes_total"]
        rows.append(row)
    return {"workspace": WORKSPACE, "rows": rows, "totals": totals,
            "allow_list": sorted(allow), "n_our_projects": len(rows)}


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
        targets = [_species_project(_species_arg(args.species))]
    elif args.legacy_per_species:
        # the deleted per-species projects were named by the legacy labels
        targets = [f"cwd12-{lb.lower()}" for lb in CWD12_LEGACY_LABELS]
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


def _species_arg(name):
    """--species takes a cwd12 species (v3.60.0), e.g. SpottedSpurge."""
    sp = species_of(name)
    if sp is None:
        print(f"FATAL: --species {name!r} is not a cwd12 species "
              f"({', '.join(CWD12_SPECIES)})", file=sys.stderr)
        sys.exit(2)
    return sp


def _species_project(sp):
    """`cwd12-<species>` for --species. v3.60.0: the per-species projects made
    before v3.60.0 were named by the legacy labels, and 10 of those slugs are
    the slug of a different species now (cwd12-goosegrass held SpottedSpurge).
    Such a project is refused; --legacy-per-species reads them by label."""
    proj = f"cwd12-{sp.lower()}"
    for i, lb in enumerate(CWD12_LEGACY_LABELS):
        if f"cwd12-{lb.lower()}" == proj and CWD12_SPECIES[i] != sp:
            print(f"FATAL: --species {sp} resolves to {proj}, which before "
                  f"v3.60.0 was the per-species project of {CWD12_SPECIES[i]} "
                  f"(legacy label {lb}). Pass --project {proj} if it was created "
                  f"after v3.60.0, or --legacy-per-species for the old projects.",
                  file=sys.stderr)
            sys.exit(2)
    return proj


def merge_project_dir(loc: Path, proj_name: str, remap: dict, names: list,
                      photo_kind, img_dir: Path, lbl_dir: Path) -> dict:
    """Copy one downloaded project's images into the merged set with labels
    remapped to cwd12 ids. `photo_kind(img)` says whether an image is a cwd12
    photograph ('holdout' / 'cwd12_train' / 'unhashable') or not (None); such
    images are dropped whole. Returns counts, with each reason separate."""
    import shutil
    st = {"images": 0, "boxes": 0, "dropped_oov_boxes": 0,
          "dropped_oov_by_name": {}, "dropped_cwd12_photos": 0,
          "dropped_cwd12_by_kind": {}, "dropped_no_cwd12_box": 0}
    for split in ("train", "valid", "test"):
        si = loc / split / "images"
        sl = loc / split / "labels"
        if not si.is_dir() or not sl.is_dir():
            continue
        for img in si.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            kind = photo_kind(img)
            if kind is not None:
                st["dropped_cwd12_photos"] += 1
                st["dropped_cwd12_by_kind"][kind] = \
                    st["dropped_cwd12_by_kind"].get(kind, 0) + 1
                continue
            lbl = sl / (img.stem + ".txt")
            lines_out = []
            if lbl.is_file():
                try:
                    for line in lbl.read_text(errors="ignore").splitlines():
                        p = line.split()
                        if not p or not p[0].lstrip("-").isdigit():
                            continue
                        src = int(p[0])
                        if src not in remap:
                            st["dropped_oov_boxes"] += 1
                            nm = str(names[src]) if 0 <= src < len(names) else f"id{src}"
                            st["dropped_oov_by_name"][nm] = \
                                st["dropped_oov_by_name"].get(nm, 0) + 1
                            continue
                        p[0] = str(remap[src])
                        lines_out.append(" ".join(p))
                except Exception:
                    pass
            # keep only images with ≥1 box of a cwd12 species
            if not lines_out:
                st["dropped_no_cwd12_box"] += 1
                continue
            new_stem = f"{proj_name}_{img.stem}"
            try:
                shutil.copy2(img, img_dir / (new_stem + img.suffix))
            except Exception:
                continue
            with open(lbl_dir / (new_stem + ".txt"), "w") as f:
                f.write("\n".join(lines_out) + "\n")  # trailing NL: clean cat/tools
            st["images"] += 1
            st["boxes"] += len(lines_out)
    return st


def _resolve_dl_targets(args):
    """Decide which Roboflow projects to pull and how to remap each.
    Returns list of (project_name, ('multiclass', None) | ('species', sp)).
    Default = allow-list MULTI-CLASS projects (our_projects); the legacy
    per-species `cwd12-<sp>` projects were deleted 2026-05-30.
    v3.60.0: `sp` is the species. The legacy per-species projects were named
    and labelled by our uploader with the legacy labels."""
    if getattr(args, "species", ""):
        sp = _species_arg(args.species)
        return [(_species_project(sp), ("species", sp))]
    if getattr(args, "legacy_per_species", False):
        return [(f"cwd12-{lb.lower()}", ("species", legacy_to_species(lb)))
                for lb in CWD12_LEGACY_LABELS]
    if getattr(args, "project", ""):
        return [(args.project, ("multiclass", None))]
    return [(p, ("multiclass", None)) for p in sorted(our_projects())]


def cmd_download_merge(args):
    """Download labeled YOLO data from our Roboflow projects and merge into a
    single multi-class CWD12 YOLO dataset for cluster training.

    v3.0.98 FIX: default now pulls the allow-list MULTI-CLASS projects
    (cwd12-multiclass-v1, weed-crop-agent-*) and remaps each box's class via
    the downloaded data.yaml names → CWD12 index. The old code iterated 12
    per-species `cwd12-<species>` projects that were DELETED 2026-05-30, so
    every download 404'd and the merged set was empty. Legacy single-class
    behavior is still available via --species / --legacy-per-species.

    Output layout:
      <out-dir>/train/images/<project>_<orig>.jpg
      <out-dir>/train/labels/<project>_<orig>.txt
      <out-dir>/data.yaml
    Everything goes to TRAIN; eval uses untouched cottonweeddet12/valid+test
    holdout so the active-learning loop never contaminates evaluation.

    v3.60.0: ids and data.yaml names are cwd12 ids named by species
    (CWD12_SPECIES). cwd12 photographs are dropped (cwd12_photo_index), the
    rest are resolved by species (_multiclass_remap), and each dropped box or
    image is counted with its reason in _merge_summary.json.
    """
    key = _key()
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    ws = rf.workspace()

    out_root = Path(args.out_dir)
    img_dir = out_root / "train" / "images"
    lbl_dir = out_root / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    targets = _resolve_dl_targets(args)
    photo_index = cwd12_photo_index()
    print(f"[download-merge] {len(targets)} target project(s): "
          f"{[t[0] for t in targets]}")

    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "targets": [t[0] for t in targets], "per_project": []}
    tot_imgs = tot_lbls = tot_boxes = 0

    for proj_name, mode in targets:
        try:
            proj = ws.project(proj_name)
            vs = proj.versions()
        except Exception as e:
            print(f"[skip] {proj_name}: {type(e).__name__}: {e}")
            summary["per_project"].append({"project": proj_name, "ok": False,
                                           "error": str(e)})
            continue
        if not vs:
            print(f"[skip] {proj_name}: no versions yet — run generate-versions first")
            summary["per_project"].append({"project": proj_name, "ok": False,
                                           "reason": "no_versions"})
            continue

        v = vs[-1]  # latest
        print(f"[download] {proj_name} version {v.version} ...")
        try:
            dl = v.download("yolov8", location=str(out_root / "_dl" / proj_name))
        except Exception as e:
            print(f"[FAIL download] {proj_name}: {type(e).__name__}: {e}")
            summary["per_project"].append({"project": proj_name, "ok": False,
                                           "error": f"download: {e}"})
            continue

        loc = Path(getattr(dl, "location", "") or out_root / "_dl" / proj_name)
        if not loc.is_dir():
            print(f"[FAIL] download location not found: {loc}")
            summary["per_project"].append({"project": proj_name, "ok": False,
                                           "error": "no_location"})
            continue

        # Per-project cid → cwd12-id remap, by species.
        names = _read_yaml_names(loc)
        if mode[0] == "species":
            remap, unknown = {0: CWD12_SPECIES.index(mode[1])}, {}
            names = names or [mode[1]]
        else:
            remap, unknown = _multiclass_remap(names)
            if not remap:
                print(f"  [WARN] {proj_name}: no class name is a cwd12 species "
                      f"(names={names}) — 0 boxes will be kept")

        st = merge_project_dir(
            loc, proj_name, remap, names,
            lambda p: cwd12_photo_kind(photo_index, p), img_dir, lbl_dir)
        n_imgs, n_boxes = st["images"], st["boxes"]
        print(f"  [{proj_name}] images={n_imgs} boxes={n_boxes} "
              f"dropped_oov_boxes={st['dropped_oov_boxes']} "
              f"dropped_cwd12_photos={st['dropped_cwd12_photos']}")
        tot_imgs += n_imgs; tot_lbls += n_imgs; tot_boxes += n_boxes
        rec = {"project": proj_name, "ok": True, "mode": mode[0],
               "label_vocabulary": "real names (species_of); cwd12 photographs dropped"}
        rec.update(st)
        if mode[0] == "multiclass" and unknown:
            rec["unmapped_class_names"] = sorted(unknown)
        summary["per_project"].append(rec)

    # Write data.yaml
    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("# auto-generated by merge_roboflow_projects download-merge\n")
        f.write(f"nc: {len(CWD12_SPECIES)}\n")
        f.write("names: [" + ", ".join(repr(c) for c in CWD12_SPECIES) + "]\n")
        f.write("train: train/images\n")
        f.write("val: train/images   # placeholder — real eval uses cottonweeddet12/valid\n")

    summary["totals"] = {"images": tot_imgs, "labels": tot_lbls, "boxes": tot_boxes}
    out_state = out_root / "_merge_summary.json"
    with open(out_state, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWROTE: {out_state}")
    print(f"      dataset: {out_root}  (images={tot_imgs}, boxes={tot_boxes})")
    print(f"      data.yaml: {data_yaml}")
    if tot_imgs == 0:
        print("WARNING: 0 images merged — check project names/versions/perms above.")


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
                        help="a single cwd12 species (e.g. PricklySida) → cwd12-<species>; "
                             "refused where that slug was a legacy project of another species")
    p_gen.add_argument("--legacy-per-species", action="store_true",
                        help="legacy: iterate all 12 cwd12-<species> projects (deleted 2026-05-30)")
    p_gen.add_argument("--force", action="store_true",
                        help="re-generate even if Versions exist")
    p_gen.add_argument("--out",
                        default=str(REPO / "results" / "framework" / "roboflow_versions.json"))

    p_dl = sub.add_parser("download-merge",
                           help="download labeled YOLO data and merge to multi-class")
    p_dl.add_argument("--project", default="",
                       help="single project to pull (default: allow-list multi-class projects)")
    p_dl.add_argument("--species", default="",
                       help="single cwd12 species (e.g. PricklySida) → cwd12-<species> (single-class); "
                            "refused where that slug was a legacy project of another species")
    p_dl.add_argument("--legacy-per-species", action="store_true",
                       help="legacy: iterate 12 cwd12-<species> projects (deleted 2026-05-30)")
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
