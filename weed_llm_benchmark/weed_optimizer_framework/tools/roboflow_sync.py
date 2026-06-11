"""
roboflow_sync.py — push curated weed datasets to a Roboflow project for
bbox ground-truth labeling, and pull verified annotations back.

Part of the Roboflow-integrated active-learning labeling pipeline
(see memory project_roboflow_pipeline_plan). Division of labor:
  our site = high-level dataset SELECTION  →  Roboflow = bbox labeling
  → ground truth  →  cluster = training.

SECURITY: the API key is read from env ROBOFLOW_API_KEY only. NEVER hard-
code it, never print it, never commit it. (Same rule as the GitHub PAT.)

Project structure (user decision 2026-05-28): ONE multi-class object-
detection project `cwd12-weeds` with the 12 CottonWeedDet12 classes.
Seed = cottonweeddet12/train (gold, all 12 species). NEVER upload the
valid/test split (eval contamination).

Usage (run on cluster where the images live; key in env):
  export ROBOFLOW_API_KEY=...        # do NOT echo
  python -m weed_optimizer_framework.tools.roboflow_sync whoami
  python -m weed_optimizer_framework.tools.roboflow_sync create-project
  python -m weed_optimizer_framework.tools.roboflow_sync upload \
       --images downloads/cottonweeddet12/train/images \
       --labels downloads/cottonweeddet12/train/labels \
       --split train --batch green --limit 5      # test first!
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CWD12 = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
]
# v3.0.99.20: normalized CWD12 names + aliases, for --require-cwd12 filtering
# (sync only the clean datasets whose class names actually map to CWD12 species).
_CWD12_NORM = {"".join(c for c in s.lower() if c.isalnum()) for s in CWD12}
for _a in ("morningglory", "carpetweed", "palmeramaranth", "spottedspurge",
           "spurge", "pricklysida", "eleusineindica", "cyperus", "ipomoea"):
    _CWD12_NORM.add(_a)


def _slug_has_cwd12(class_names):
    """True if ≥1 of the slug's class names maps to a CWD12 species."""
    for c in (class_names or []):
        if "".join(ch for ch in str(c).lower() if ch.isalnum()) in _CWD12_NORM:
            return True
    return False
# v3.0.58 (2026-05-30): project name configurable. Defaults to
# 'cwd12-weeds' for backward compat with personal workspace
# (research-lhi4x). Override via env ROBOFLOW_PROJECT or per-call
# `--project <name>` flag — e.g., a-test-of-will/cwd12-multiclass-v1.
PROJECT_NAME = os.environ.get("ROBOFLOW_PROJECT", "cwd12-weeds")


def _resolve_project(args) -> str:
    """Pick project: CLI --project > env ROBOFLOW_PROJECT > module default."""
    return getattr(args, "project", None) or PROJECT_NAME


def _key() -> str:
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not k:
        print("FATAL: ROBOFLOW_API_KEY not set in env", file=sys.stderr)
        sys.exit(2)
    return k


def _rf():
    from roboflow import Roboflow
    return Roboflow(api_key=_key())


def _workspace():
    return _rf().workspace()


def cmd_whoami(args):
    ws = _workspace()
    # ws.url / ws.project_list — print non-secret identifiers only
    print("workspace:", getattr(ws, "url", "?"))
    try:
        projs = ws.projects()
        print("projects:", projs)
    except Exception as e:
        print("projects(): ", type(e).__name__, e)


def _ws_name() -> str:
    """Resolve the workspace slug (a-test-of-will or override)."""
    return os.environ.get("ROBOFLOW_WORKSPACE", "a-test-of-will")


def _list_folders() -> list:
    """v3.0.69 (2026-05-31): Roboflow Project Folders REST API.

    Path uses internal name `/groups` (NOT /folders), which is why earlier
    probes against /folders + /project_folders found nothing. Confirmed
    working on free tier (school workspace a-test-of-will, returned 200
    + folder data). Docs page incorrectly says "Enterprise only" — empirically
    the basic CRUD works on free; only RBAC/SSO-restricted folders need Enterprise.

    See memory/feedback_roboflow_folder_api.md for the full investigation."""
    import urllib.request, json as _json
    url = f"https://api.roboflow.com/{_ws_name()}/groups?api_key={_key()}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return _json.load(r).get("data", []) or []
    except Exception as e:
        print(f"[folders] list FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return []


def _resolve_folder_id(folder_ref: str) -> str:
    """Accept either an explicit folder id OR a folder name; return the id.
    Returns '' if not found."""
    if not folder_ref:
        return ""
    # 20-char ids; names are arbitrary — try as id first
    for f in _list_folders():
        if folder_ref == f.get("id") or folder_ref == f.get("name"):
            return f.get("id", "")
    return ""


def _add_project_to_folder(project_name: str, folder_id: str) -> bool:
    """PATCH /:ws/groups/:folder_id/projects. Returns True on 200/204."""
    import urllib.request, urllib.error, json as _json
    if not folder_id:
        return False
    url = (f"https://api.roboflow.com/{_ws_name()}/groups/{folder_id}"
           f"/projects?api_key={_key()}")
    body = _json.dumps({"projects": [project_name]}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code = r.status
        ok = code in (200, 204)
        print(f"[folders] PATCH {project_name} → folder {folder_id}: HTTP {code}")
        return ok
    except urllib.error.HTTPError as e:
        print(f"[folders] PATCH FAIL HTTP {e.code}: {e.read()[:200]!r}",
              file=sys.stderr)
        return False
    except Exception as e:
        print(f"[folders] PATCH FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def cmd_create_project(args):
    ws = _workspace()
    name = _resolve_project(args)
    try:
        proj = ws.create_project(
            project_name=name,
            project_type="object-detection",
            project_license="MIT",
            annotation=name,
        )
        print(f"CREATED project {name}: {proj}")
    except Exception as e:
        # Already exists is fine — report and continue
        print(f"create_project({name}): {type(e).__name__}: {e}")
        print("(if it already exists, that's OK — proceed to upload)")

    # v3.0.69: optionally place into a folder. ROBOFLOW_FOLDER (env) or
    # --folder (CLI) can be either a folder-id OR a folder-name.
    folder_ref = (getattr(args, "folder", None)
                  or os.environ.get("ROBOFLOW_FOLDER", ""))
    if folder_ref:
        fid = _resolve_folder_id(folder_ref)
        if fid:
            ok = _add_project_to_folder(name, fid)
            if ok:
                print(f"[folders] placed {name} into folder {folder_ref} ({fid})")
            else:
                print(f"[folders] could not place {name} in {folder_ref}")
        else:
            print(f"[folders] folder ref {folder_ref!r} not found in workspace; "
                  f"project lives at root level. List: "
                  f"{[f.get('name') for f in _list_folders()]}")


def cmd_sync_newest_slugs(args):
    """v3.0.71 / v3.0.76 — iterate registry; for each slug with status='downloaded'
    AND not yet marked roboflow_synced=true, upload its images to a target
    Roboflow project, then mark synced. Optionally place into a folder.

    v3.0.76 (2026-06-01): the target project is now PER-ROUND. Each slug's
    harvest_round determines the destination project name:
      - round 1 → weed-crop-agent-dataset (legacy v1)
      - round N → weed-crop-agent-v{N}
    The --project arg now means 'default project if slug has no round info'.

    Skips:
      - slugs not on disk (local_path missing)
      - cwd12 baselines (cottonweed_*, cottonweeddet12) — those belong to
        the frozen benchmark project, not the agent collection pool
      - slugs the user has flagged junk in slug_verdicts.jsonl

    This is what closes the brain_harvest → Roboflow visible loop.
    """
    import json as _json
    from pathlib import Path as _Path

    default_project = getattr(args, "project", None) or PROJECT_NAME
    folder_ref = (getattr(args, "folder", None)
                  or os.environ.get("ROBOFLOW_FOLDER", ""))
    cap_per_slug = int(getattr(args, "cap_per_slug", 0) or 0)
    # v3.0.99.20: clean-only sync. --require-cwd12 → upload only datasets whose class
    # names map to ≥1 CWD12 species (skip generic 'weed'/'crop', numeric, off-topic).
    # --force-project → route ALL to one project (override per-round) so the clean set
    # lands in a single pristine project, not mixed with the numeric v4.
    require_cwd12 = (getattr(args, "require_cwd12", False)
                     or os.environ.get("SYNC_REQUIRE_CWD12") == "1")
    force_project = (getattr(args, "force_project", None)
                     or os.environ.get("SYNC_FORCE_PROJECT", "") or "")

    # v3.0.76: helper to map round → project name
    try:
        from weed_optimizer_framework.tools.rounds import round_project_name
    except Exception:
        def round_project_name(n):
            return "weed-crop-agent-dataset" if int(n) == 1 else f"weed-crop-agent-v{int(n)}"

    reg_path = _Path(os.environ.get(
        "REPO_ROOT",
        "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
    )) / "results" / "framework" / "dataset_registry.json"
    if not reg_path.exists():
        print(f"FATAL: registry missing {reg_path}", file=sys.stderr)
        sys.exit(2)
    reg = _json.load(open(reg_path))
    ds = reg.get("datasets", {}) or {}

    CWD12_BASELINES = {"cottonweed_holdout", "cottonweed_sp8",
                       "cottonweeddet12"}

    # slug_verdicts.jsonl: {slug: latest_verdict}
    sv_path = reg_path.parent / "slug_verdicts.jsonl"
    junk_slugs = set()
    if sv_path.exists():
        try:
            for line in sv_path.read_text().splitlines():
                if not line.strip():
                    continue
                e = _json.loads(line)
                if e.get("verdict") == "junk":
                    junk_slugs.add(e.get("slug"))
        except Exception:
            pass

    # v3.0.71.3: also skip audit-flagged garbage slugs. Last bug-test showed
    # sync iterating dict-insertion order and hitting ibm_research__cif_dataset
    # FIRST — that slug's .txt files are placeholders Roboflow rejects with
    # 'AnnotationSaveError: Unrecognized annotation format'. Skipping garbage
    # candidates means we never even try them.
    garbage_slugs = set()
    try:
        from weed_optimizer_framework.tools.audit_registry_garbage import (
            _slug_garbage_reason as _audit_reason,
        )
        for s, info in ds.items():
            if _audit_reason(s, info, 100):
                garbage_slugs.add(s)
        if garbage_slugs:
            print(f"  [audit] skipping {len(garbage_slugs)} garbage slugs: "
                  f"{sorted(garbage_slugs)}")
    except Exception as e:
        print(f"  [audit] could not run garbage filter: {e}")

    # v3.0.75 (2026-06-01): user feedback — "我希望所有采集的数据集都会上传到
    # roboflow上先 不然访问速度非常慢". cwd12 baselines (cottonweed_sp8/holdout)
    # are part of "all collected data" too — review should happen in Roboflow's
    # fast CDN, not Lustre-served dashboard. So we no longer skip them by
    # default. To opt out, set --skip-baselines or env SYNC_SKIP_BASELINES=1.
    skip_baselines = (getattr(args, "skip_baselines", False)
                      or os.environ.get("SYNC_SKIP_BASELINES") == "1")

    pending = []
    for slug, info in ds.items():
        if info.get("status") != "downloaded":
            continue
        if skip_baselines and slug in CWD12_BASELINES:
            continue
        if slug in junk_slugs:
            continue
        if slug in garbage_slugs:
            continue
        if info.get("roboflow_synced"):
            continue
        if require_cwd12 and not _slug_has_cwd12(info.get("class_names")):
            continue
        lp = info.get("local_path", "")
        if not lp or not os.path.isdir(lp):
            continue
        pending.append((slug, info, lp))

    # v3.0.77.2: was `{project}` (NameError after v3.0.76 rename to
    # default_project). Caught by clicking sync from /rounds after
    # round 2 backfill — sync silently died with NameError.
    print(f"=== sync-newest-slugs → default project {default_project} (per-round routing applies) ===")
    print(f"  registry: {len(ds)} slugs total")
    print(f"  pending sync: {len(pending)}")

    if not pending:
        print("  nothing to do — all eligible slugs already synced.")
        return

    # Resolve folder once
    fid = ""
    if folder_ref:
        fid = _resolve_folder_id(folder_ref)
        if not fid:
            print(f"  WARN folder {folder_ref!r} not found — skip folder step")

    # For each slug: walk imgs + (optional) labels, upload, then mark
    from roboflow import Roboflow as _RF
    # v3.0.76: per-round project resolution — derive target project from
    # each slug's harvest_round. Open projects lazily and cache.
    rf = _RF(api_key=_key())
    ws = rf.workspace(_ws_name())

    n_uploaded_total = 0
    n_synced_slugs = 0
    EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    proj_cache = {}  # name → SDK Project object
    folders_visited = set()  # so we don't PATCH-into-folder per slug

    for slug, info, lp in pending:
        # Resolve THIS slug's target project from its harvest_round
        # (v3.0.99.20: --force-project overrides per-round routing).
        h_round = int(info.get("harvest_round", 0) or 0)
        slug_proj_name = force_project or (round_project_name(h_round)
                                           if h_round > 0 else default_project)
        if slug_proj_name not in proj_cache:
            try:
                proj_cache[slug_proj_name] = ws.project(slug_proj_name)
                print(f"\n=== Project: {slug_proj_name} (round {h_round}) ===")
            except Exception as e:
                print(f"  SKIP {slug}: cannot open project "
                      f"{slug_proj_name!r}: {type(e).__name__}: "
                      f"{str(e)[:80]}. Run start_new_round to create it.",
                      file=sys.stderr)
                proj_cache[slug_proj_name] = None
        proj = proj_cache[slug_proj_name]
        if proj is None:
            continue

        print(f"\n--- {slug} ---")
        # Look for images in common subpaths
        # v3.0.77.3: GitHub/Kaggle slugs have varied layouts:
        #   - HF:        <lp>/images/*.jpg
        #   - YOLO:      <lp>/train/images/*.jpg
        #   - this one:  <lp>/data/images/*.jpg (yolov9-beehive)
        #   - kaggle:    <lp>/agri_data/data/*.jpeg (ravirajsinh45)
        # Add common 2-level patterns + fall back to scanning any dir
        # with enough imgs.
        cand_img_dirs = [_Path(lp) / s for s in
                        ("images", "train/images", "valid/images",
                         "test/images", "data/images", "data",
                         "agri_data/data", "raw_data", "train",
                         "valid", "test")]
        cand_img_dirs.append(_Path(lp))
        img_dir = next((d for d in cand_img_dirs if d.is_dir()
                        and any(p.suffix in EXTS for p in d.iterdir())), None)
        if img_dir is None:
            # Fallback: recursive scan, pick dir with most images
            counts = {}
            for p in _Path(lp).rglob("*"):
                if p.suffix in EXTS and p.is_file():
                    counts[p.parent] = counts.get(p.parent, 0) + 1
                    if sum(counts.values()) > 5000:
                        break
            if counts:
                img_dir = max(counts.items(), key=lambda kv: kv[1])[0]
                print(f"  (rglob fallback) → {img_dir}  ({counts[img_dir]} imgs)")
        if img_dir is None:
            print(f"  SKIP no image dir under {lp}")
            continue
        # Find labels in matching subpath
        lbl_dir = None
        for s in ("labels", "train/labels", "valid/labels", "test/labels"):
            if (_Path(lp) / s).is_dir():
                lbl_dir = _Path(lp) / s
                break

        imgs = sorted(p for p in img_dir.iterdir() if p.suffix in EXTS)
        if cap_per_slug:
            imgs = imgs[:cap_per_slug]
        print(f"  imgs: {len(imgs)} from {img_dir}")
        print(f"  labels: {lbl_dir or '(none)'}")

        # v3.0.74: round-aware batch_name. registry has current_round +
        # each slug has harvest_round. Tag the upload so Roboflow UI can
        # filter by round. Fall back to slug-only if metadata missing.
        h_round = info.get("harvest_round")
        batch_name = (f"agent-v{int(h_round)}-{slug}"
                      if h_round else f"agent-{slug}")
        # v3.0.99.18: label-index → REAL class name from the registry. Without it
        # every dataset's raw indices upload as 0/1/2 and COLLIDE in the shared
        # project (one set's "0"=crop, another's "0"=eclipta → merged/corrupted).
        # With per-slug names the project gets distinct real classes the human can
        # actually review. Slug name itself is prepended so cross-dataset same-name
        # classes (generic "weed") stay traceable to their source.
        _cn = info.get("class_names") or []
        labelmap = {}
        for _i, _n in enumerate(_cn):
            _n = str(_n).strip()
            if _n and not _n.isdigit():        # skip placeholder-numeric names
                labelmap[_i] = _n
        ok = 0
        fail = 0
        # v3.0.75.2: progress logging every 10 imgs so silent hangs are
        # visible. The Roboflow SDK upload can stall on network hiccups
        # and the previous code printed nothing for ~3 min stretches.
        import time as _t
        last_print = _t.time()
        for i, img in enumerate(imgs, 1):
            try:
                kw = dict(
                    image_path=str(img),
                    split="train",
                    batch_name=batch_name,
                    tag_names=["green", "brain-harvest", slug,
                               f"round-{h_round}" if h_round else "round-?"],
                    num_retry_uploads=1,
                )
                if lbl_dir:
                    lbl = lbl_dir / (img.stem + ".txt")
                    if lbl.is_file():
                        kw["annotation_path"] = str(lbl)
                        # v3.0.99.18: map indices → real names (see above).
                        if labelmap:
                            kw["annotation_labelmap"] = labelmap
                proj.single_upload(**kw)
                ok += 1
            except Exception as e:
                fail += 1
                if fail < 5:
                    print(f"    [{i}/{len(imgs)}] FAIL {img.name}: "
                          f"{type(e).__name__}: {str(e)[:120]}",
                          flush=True)
            # v3.0.75.2: heartbeat every 10 imgs OR every 30s — without
            # this, a stuck upload looked exactly like "still running".
            if i % 10 == 0 or _t.time() - last_print > 30:
                print(f"    progress: {i}/{len(imgs)} ok={ok} fail={fail}",
                      flush=True)
                last_print = _t.time()
        n_uploaded_total += ok
        print(f"  uploaded {ok} ok, {fail} fail", flush=True)

        # Mark synced in registry
        if ok > 0:
            info["roboflow_synced"] = True
            info["roboflow_synced_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            info["roboflow_synced_count"] = ok
            n_synced_slugs += 1

    # Save registry mid-flight so progress isn't lost
    tmp = str(reg_path) + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(reg, f, indent=2)
    os.replace(tmp, reg_path)
    print(f"\n=== TOTAL: synced {n_synced_slugs}/{len(pending)} slugs, "
          f"uploaded {n_uploaded_total} imgs ===")

    # v3.0.77.3: was `project` (NameError, second instance from same
    # v3.0.76 rename). PATCH all projects we touched into folder.
    if fid and proj_cache:
        for proj_name in proj_cache:
            if proj_cache[proj_name] is None:
                continue
            ok = _add_project_to_folder(proj_name, fid)
            print(f"[folders] place {proj_name} → {folder_ref}: "
                  f"{'ok' if ok else 'FAIL'}")


def cmd_move_to_folder(args):
    """Standalone: move an existing project into a folder by name or id."""
    project = getattr(args, "project", None) or _resolve_project(args)
    folder_ref = getattr(args, "folder", None) or os.environ.get(
        "ROBOFLOW_FOLDER", "")
    if not project or not folder_ref:
        print("FATAL: --project and --folder required (or set ROBOFLOW_FOLDER)",
              file=sys.stderr)
        sys.exit(2)
    fid = _resolve_folder_id(folder_ref)
    if not fid:
        print(f"FATAL: folder {folder_ref!r} not found. Available: "
              f"{[(f.get('name'), f.get('id')) for f in _list_folders()]}",
              file=sys.stderr)
        sys.exit(3)
    ok = _add_project_to_folder(project, fid)
    sys.exit(0 if ok else 4)


def cmd_list_folders(args):
    """Pretty-print all folders in the workspace + their members."""
    folders = _list_folders()
    print(f"workspace {_ws_name()}: {len(folders)} folder(s)")
    for f in folders:
        ps = f.get("projects") or []
        print(f"  - {f.get('name','?')!r}  id={f.get('id','?')}  "
              f"({len(ps)} projects)")
        for p in ps:
            print(f"      ↳ {p}")


def cmd_upload(args):
    """Upload images + YOLO .txt annotations to the project.
    `batch` tags provenance: green=human/gold-trusted, red=model-proposed."""
    images = Path(args.images)
    labels = Path(args.labels)
    if not images.is_dir():
        print(f"FATAL: images dir not found: {images}", file=sys.stderr)
        sys.exit(2)
    ws = _workspace()
    proj = ws.project(PROJECT_NAME)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    imgs = sorted(p for p in images.iterdir() if p.suffix in exts)
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"uploading {len(imgs)} images  split={args.split} batch={args.batch}")

    ok = fail = noann = 0
    t0 = time.time()
    for i, img in enumerate(imgs, 1):
        lbl = labels / (img.stem + ".txt") if labels else None
        has_ann = lbl is not None and lbl.is_file()
        if not has_ann:
            noann += 1
        try:
            kw = dict(
                image_path=str(img),
                split=args.split,
                batch_name=f"{args.batch}-{args.split}",
                tag_names=[args.batch],
                num_retry_uploads=2,
            )
            if has_ann:
                kw["annotation_path"] = str(lbl)
                # YOLO txt needs the class index→name map
                kw["annotation_labelmap"] = {i: n for i, n in enumerate(CWD12)}
            proj.single_upload(**kw)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  [{i}] FAIL {img.name}: {type(e).__name__}: {e}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(imgs)} ok={ok} fail={fail} "
                  f"({time.time()-t0:.0f}s)")
    print(f"DONE upload: ok={ok} fail={fail} no_annotation={noann} "
          f"in {time.time()-t0:.0f}s")


def _primary_species(lbl: Path):
    """Read a YOLO .txt and return the CWD12 name of its most-frequent class.
    None if no valid boxes."""
    try:
        lines = lbl.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    from collections import Counter
    c = Counter()
    for ln in lines:
        p = ln.split()
        if p and p[0].lstrip("-").isdigit():
            cid = int(p[0])
            if 0 <= cid < len(CWD12):
                c[cid] += 1
    if not c:
        return None
    return CWD12[c.most_common(1)[0][0]]


def cmd_bulk_upload(args):
    """Upload images grouped BY SPECIES into per-species Roboflow batches
    (user request: 'different classes in different places'), in parallel.

    One project (cwd12-weeds), but batch_name = green-<species> so the
    Annotate view groups/filters by species. Parallel workers measure
    whether 57s/img was per-call overhead (parallel helps) or global
    free-tier rate limiting (parallel won't help → wait for paid tier)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    images = Path(args.images)
    labels = Path(args.labels)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_imgs = sorted(p for p in images.iterdir() if p.suffix in exts)

    # group by primary species; cap per species for testing via --per-species
    by_sp: dict = {}
    for img in all_imgs:
        lbl = labels / (img.stem + ".txt")
        sp = _primary_species(lbl) if lbl.is_file() else None
        sp = sp or "Unlabeled"
        by_sp.setdefault(sp, []).append((img, lbl if lbl.is_file() else None))
    # build work list with per-species cap
    work = []
    for sp, items in sorted(by_sp.items()):
        take = items[: args.per_species] if args.per_species else items
        for img, lbl in take:
            work.append((sp, img, lbl))
    print(f"species present: { {k: len(v) for k, v in sorted(by_sp.items())} }")
    print(f"uploading {len(work)} images across {len(by_sp)} species, "
          f"workers={args.workers}")

    ws = _workspace()
    proj_name = _resolve_project(args)
    proj = ws.project(proj_name)
    print(f"target project: {proj_name}")
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}
    labelmap = {i: n for i, n in enumerate(CWD12)}

    def _one(task):
        sp, img, lbl = task
        t0 = time.time()
        try:
            kw = dict(image_path=str(img), split=args.split,
                      batch_name=f"{args.batch}-{sp}",
                      tag_names=[args.batch, sp], num_retry_uploads=1)
            if lbl is not None:
                kw["annotation_path"] = str(lbl)
                kw["annotation_labelmap"] = labelmap
            proj.single_upload(**kw)
            with lock:
                counters["ok"] += 1
            return (sp, img.name, time.time() - t0, None)
        except Exception as e:
            with lock:
                counters["fail"] += 1
            return (sp, img.name, time.time() - t0, f"{type(e).__name__}: {e}")

    t0 = time.time()
    per_times = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, t) for t in work]
        for i, f in enumerate(as_completed(futs), 1):
            sp, name, dt, err = f.result()
            per_times.append(dt)
            if err:
                print(f"  FAIL [{sp}] {name}: {err}")
            if i % 10 == 0:
                print(f"  ... {i}/{len(work)} ok={counters['ok']} "
                      f"fail={counters['fail']} ({time.time()-t0:.0f}s)")
    wall = time.time() - t0
    avg = sum(per_times) / len(per_times) if per_times else 0
    print(f"DONE bulk: ok={counters['ok']} fail={counters['fail']} "
          f"wall={wall:.0f}s  per-img(in-thread)avg={avg:.1f}s  "
          f"effective={wall/max(1,len(work)):.1f}s/img")


def cmd_create_species_projects(args):
    """v3.0.44.2 — user feedback: Roboflow has NO folders within a project,
    only filters. To literally separate species, create ONE project per
    species (`cwd12-<species>`). Workspace home then shows 12 distinct
    project tiles = the 'different folders' UX the user wants."""
    ws = _workspace()
    for sp in CWD12:
        name = f"cwd12-{sp.lower()}"
        try:
            proj = ws.create_project(
                project_name=name,
                project_type="object-detection",
                project_license="MIT",
                annotation=name,
            )
            print(f"CREATED {name}")
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {e}")


def cmd_species_upload(args):
    """Upload to per-species projects: each image of species X goes to
    project `cwd12-<x>` with its labels remapped to single-class (cid=0).
    The annotation is filtered to keep ONLY this species' boxes (other
    species in mixed-species images are dropped — single-class project
    semantics)."""
    import threading, tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    images = Path(args.images); labels = Path(args.labels)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_imgs = sorted(p for p in images.iterdir() if p.suffix in exts)

    by_sp: dict = {}
    for img in all_imgs:
        lbl = labels / (img.stem + ".txt")
        if not lbl.is_file():
            continue
        sp = _primary_species(lbl)
        if sp is None:
            continue
        by_sp.setdefault(sp, []).append((img, lbl))
    sp2id = {sp: i for i, sp in enumerate(CWD12)}
    print(f"species available: { {k: len(v) for k, v in sorted(by_sp.items())} }")

    ws = _workspace()
    for sp in CWD12:
        items = by_sp.get(sp, [])
        if args.per_species:
            items = items[: args.per_species]
        if not items:
            print(f"=== {sp}: 0 items, skip ===")
            continue
        proj_name = f"cwd12-{sp.lower()}"
        try:
            proj = ws.project(proj_name)
        except Exception as e:
            print(f"=== {sp}: project {proj_name} not found ({e}) — skip ===")
            continue

        cid = sp2id[sp]
        labelmap = {0: sp}
        lock = threading.Lock()
        counters = {"ok": 0, "fail": 0, "no_box": 0}

        def _one(item):
            img, lbl = item
            t0 = time.time()
            try:
                txt = lbl.read_text(errors="ignore")
                kept = []
                for ln in txt.splitlines():
                    p = ln.split()
                    if p and p[0].lstrip("-").isdigit() and int(p[0]) == cid:
                        kept.append("0 " + " ".join(p[1:]))
                if not kept:
                    with lock:
                        counters["no_box"] += 1
                    return (img.name, time.time() - t0, "no_box_of_species")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                                  delete=False) as tf:
                    tf.write("\n".join(kept))
                    tmp = tf.name
                try:
                    proj.single_upload(
                        image_path=str(img), annotation_path=tmp,
                        annotation_labelmap=labelmap,
                        split=args.split, batch_name=f"{args.batch}-{sp}",
                        tag_names=[args.batch, sp], num_retry_uploads=1,
                    )
                    with lock:
                        counters["ok"] += 1
                    return (img.name, time.time() - t0, None)
                finally:
                    try: os.unlink(tmp)
                    except Exception: pass
            except Exception as e:
                with lock:
                    counters["fail"] += 1
                return (img.name, time.time() - t0, f"{type(e).__name__}: {e}")

        t0 = time.time()
        print(f"=== {sp} → {proj_name}: uploading {len(items)} imgs"
              f" (workers={args.workers}) ===")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, x) for x in items]
            for f in as_completed(futs):
                _, _, err = f.result()
                if err and err != "no_box_of_species":
                    print(f"  FAIL: {err}")
        wall = time.time() - t0
        print(f"=== {sp}: ok={counters['ok']} fail={counters['fail']} "
              f"no_box={counters['no_box']} wall={wall:.0f}s ===")


# ---------------------------------------------------------------------------
# v3.0.99.21 — delete junk images from Roboflow (user: 垃圾删除 + 优质保留 +
# 标注导出回集群). Storage on Roboflow is a recurring per-month cost, so after
# human review we DELETE the junk-verdict datasets' images (keep only clean).
#   search:  POST /{ws}/{proj}/search  body {tag, offset, limit}  → results[].id
#   delete:  DELETE /{ws}/{proj}/images  body {images:[ids]}      → 204
# Ground-truth labels are exported to the cluster separately (download-merge),
# so deleting from Roboflow never loses data.
# ---------------------------------------------------------------------------
def _rf_api(method, path, body=None, timeout=30):
    import urllib.request, urllib.error, json as _j
    url = f"https://api.roboflow.com/{_ws_name()}/{path}?api_key={_key()}"
    data = _j.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode("utf-8", "replace")
        return r.status, (_j.loads(txt) if txt.strip() else {})


def cmd_delete_junk(args):
    """Delete junk-verdict datasets' images from ALL our Roboflow projects.
    Latest verdict per slug wins; dry-run unless --apply."""
    import json as _j
    from pathlib import Path as _P
    reg_path = _P(os.environ.get(
        "REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark")
    ) / "results" / "framework" / "dataset_registry.json"
    sv = reg_path.parent / "slug_verdicts.jsonl"
    latest = {}
    if sv.exists():
        for line in sv.read_text().splitlines():
            if line.strip():
                try:
                    e = _j.loads(line)
                    if e.get("slug"):
                        latest[e["slug"]] = e.get("verdict")
                except Exception:
                    pass
    junk = sorted(s for s, v in latest.items() if v == "junk")
    if not junk:
        print("no junk-verdict slugs — nothing to delete")
        return 0
    apply = getattr(args, "apply", False)
    print(f"=== delete-junk ({'APPLY' if apply else 'DRY-RUN'}) ===")
    print(f"junk slugs ({len(junk)}): {junk}")
    try:
        from .merge_roboflow_projects import our_projects
        projects = sorted(our_projects())
    except Exception:
        projects = ["weed-crop-agent-dataset", "weed-crop-agent-clean"]
    total = 0
    for proj in projects:
        for slug in junk:
            ids, offset = [], 0
            while True:
                try:
                    st, res = _rf_api("POST", f"{proj}/search",
                                      {"tag": slug, "offset": offset, "limit": 250})
                except Exception:
                    break
                batch = [x.get("id") for x in (res.get("results") or []) if x.get("id")]
                ids += batch
                if len(batch) < 250:
                    break
                offset += 250
            if not ids:
                continue
            print(f"  {proj}/{slug}: {len(ids)} junk images"
                  + ("" if apply else " (dry-run, not deleted)"))
            if not apply:
                total += len(ids)
                continue
            for i in range(0, len(ids), 250):
                chunk = ids[i:i + 250]
                try:
                    st, _ = _rf_api("DELETE", f"{proj}/images", {"images": chunk})
                    total += len(chunk)
                    print(f"    deleted {len(chunk)} (HTTP {st})")
                except Exception as e:
                    print(f"    delete FAILED: {type(e).__name__}: {str(e)[:100]}")
    print(f"DONE: {'deleted' if apply else 'would delete'} {total} junk images")
    return 0


# ---------------------------------------------------------------------------
# v3.0.99.23 (A2/A3) — PER-DATASET push of a FEW sampled images (prof: only a few
# images per dataset, human decides how many). push-slug uploads N diverse images
# of ONE slug to a project (real names via labelmap) + records to labeling_tracker;
# delete-slug removes that slug's images from the project (free Roboflow quota).
# ---------------------------------------------------------------------------
_PUSH_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _find_slug_image_dir(lp):
    from pathlib import Path as _P
    cands = [_P(lp) / s for s in
             ("images", "train/images", "valid/images", "test/images",
              "data/images", "data", "agri_data/data", "raw_data",
              "train", "valid", "test")]
    cands.append(_P(lp))
    for d in cands:
        if d.is_dir() and any(p.suffix in _PUSH_EXTS for p in d.iterdir()):
            return d
    counts = {}
    for p in _P(lp).rglob("*"):
        if p.suffix in _PUSH_EXTS and p.is_file():
            counts[p.parent] = counts.get(p.parent, 0) + 1
            if sum(counts.values()) > 5000:
                break
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None


def _labelmap_for(info):
    lm = {}
    for i, n in enumerate(info.get("class_names") or []):
        n = str(n).strip()
        if n and not n.isdigit():
            lm[i] = n
    return lm


def _registry():
    import json as _j
    from pathlib import Path as _P
    rp = _P(os.environ.get(
        "REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark")
    ) / "results" / "framework" / "dataset_registry.json"
    return _j.load(open(rp))["datasets"]


def cmd_push_slug(args):
    """Push N sampled (diverse, evenly-spaced) images of ONE slug to a project for
    human labeling. Records 'pushed' events to labeling_tracker."""
    from pathlib import Path as _P
    from weed_optimizer_framework.tools import labeling_tracker as LT
    slug = args.slug
    n = int(args.n)
    project = getattr(args, "project", None) or "weed-crop-agent-clean"
    ds = _registry()
    if slug not in ds:
        print(f"FATAL: {slug} not in registry"); return 2
    info = ds[slug]
    lp = info.get("local_path", "")
    img_dir = _find_slug_image_dir(lp) if lp and os.path.isdir(lp) else None
    if not img_dir:
        print(f"FATAL: no image dir for {slug}"); return 2
    lbl_dir = None
    for s in ("labels", "train/labels", "valid/labels", "test/labels"):
        if (_P(lp) / s).is_dir():
            lbl_dir = _P(lp) / s; break
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix in _PUSH_EXTS)
    if not imgs:
        print(f"FATAL: 0 images under {img_dir}"); return 2
    # diverse sample: evenly spaced across the dataset (v1 recommendation heuristic)
    if n < len(imgs):
        step = len(imgs) / n
        picked = [imgs[int(i * step)] for i in range(n)]
    else:
        picked = imgs
    labelmap = _labelmap_for(info)
    ws = _workspace()
    try:
        proj = ws.project(project)
    except Exception as e:
        print(f"FATAL: cannot open project {project}: {e}"); return 2
    batch = f"label-{slug}"
    ok = []
    print(f"[push-slug] {slug}: pushing {len(picked)}/{len(imgs)} sampled imgs → {project}")
    for i, img in enumerate(picked, 1):
        kw = dict(image_path=str(img), split="train", batch_name=batch,
                  tag_names=["needs-label", slug], num_retry_uploads=1)
        if lbl_dir and (lbl_dir / (img.stem + ".txt")).is_file():
            kw["annotation_path"] = str(lbl_dir / (img.stem + ".txt"))
            if labelmap:
                kw["annotation_labelmap"] = labelmap
        try:
            proj.single_upload(**kw)
            ok.append(img.name)
        except Exception as e:
            if len(ok) < 3:
                print(f"  fail {img.name}: {str(e)[:80]}")
        if i % 10 == 0:
            print(f"  {i}/{len(picked)} ok={len(ok)}", flush=True)
    LT.record_push(slug, ok, project=project, batch=batch)
    print(f"[push-slug] done: pushed {len(ok)} imgs, recorded to labeling_tracker")
    return 0


def cmd_delete_slug(args):
    """Delete ONE slug's images from a Roboflow project (frees quota after labeling).
    Records 'deleted' events."""
    from weed_optimizer_framework.tools import labeling_tracker as LT
    slug = args.slug
    project = getattr(args, "project", None) or "weed-crop-agent-clean"
    ids, offset = [], 0
    while True:
        try:
            st, res = _rf_api("POST", f"{project}/search",
                              {"tag": slug, "offset": offset, "limit": 250})
        except Exception as e:
            print(f"search failed: {e}"); break
        batch = [x.get("id") for x in (res.get("results") or []) if x.get("id")]
        ids += batch
        if len(batch) < 250:
            break
        offset += 250
    if not ids:
        print(f"[delete-slug] {slug}: 0 images in {project}"); return 0
    if not getattr(args, "apply", False):
        print(f"[delete-slug] DRY-RUN: would delete {len(ids)} imgs of {slug} from {project}")
        return 0
    deleted = 0
    for i in range(0, len(ids), 250):
        chunk = ids[i:i + 250]
        try:
            _rf_api("DELETE", f"{project}/images", {"images": chunk})
            deleted += len(chunk)
        except Exception as e:
            print(f"  delete fail: {str(e)[:80]}")
    LT.record_delete(slug, [f"id:{x}" for x in ids[:deleted]], project=project)
    print(f"[delete-slug] {slug}: deleted {deleted} imgs from {project}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    cp = sub.add_parser("create-project")
    cp.add_argument("--project", default=None,
                    help="project name (overrides env ROBOFLOW_PROJECT)")
    cp.add_argument("--folder", default=None,
                    help="v3.0.69: after create, place project into this "
                         "folder (name OR id). Also reads env ROBOFLOW_FOLDER.")
    # v3.0.69: standalone folder ops
    lf = sub.add_parser("list-folders")
    mv = sub.add_parser("move-to-folder")
    mv.add_argument("--project", default=None,
                    help="project name to move (overrides env)")
    mv.add_argument("--folder", default=None,
                    help="folder name or id (also reads env ROBOFLOW_FOLDER)")
    # v3.0.71: sync newest unsynced slugs from registry
    sn = sub.add_parser("sync-newest-slugs")
    sn.add_argument("--project", default=None,
                    help="destination project (default weed-crop-agent-dataset)")
    sn.add_argument("--folder", default=None,
                    help="folder name or id to place project (idempotent)")
    sn.add_argument("--cap-per-slug", type=int, default=0,
                    help="cap images per slug (0=all, for testing)")
    sn.add_argument("--skip-baselines", action="store_true",
                    help="skip cwd12_sp8/holdout/det12 baselines (default: include them)")
    sn.add_argument("--require-cwd12", action="store_true",
                    help="only sync datasets whose class names map to a CWD12 species (clean subset)")
    sn.add_argument("--force-project", default=None,
                    help="route ALL synced slugs to this one project (override per-round routing)")
    sub.add_parser("create-species-projects")
    up = sub.add_parser("upload")
    up.add_argument("--images", required=True)
    up.add_argument("--labels", default="")
    up.add_argument("--split", default="train", choices=["train", "valid", "test"])
    up.add_argument("--batch", default="green",
                    help="provenance tag: green=human/gold, red=model-proposed")
    up.add_argument("--limit", type=int, default=0)
    bu = sub.add_parser("bulk-upload")
    bu.add_argument("--images", required=True)
    bu.add_argument("--labels", required=True)
    bu.add_argument("--split", default="train", choices=["train", "valid", "test"])
    bu.add_argument("--batch", default="green")
    bu.add_argument("--workers", type=int, default=8)
    bu.add_argument("--per-species", type=int, default=0,
                    help="cap images per species (0=all). For testing.")
    bu.add_argument("--project", default=None,
                    help="project name (overrides env ROBOFLOW_PROJECT)")
    dj = sub.add_parser("delete-junk")
    dj.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run = just count)")
    ps = sub.add_parser("push-slug")
    ps.add_argument("--slug", required=True)
    ps.add_argument("--n", type=int, default=20, help="how many sampled images to push")
    ps.add_argument("--project", default=None, help="target project (default weed-crop-agent-clean)")
    ds_ = sub.add_parser("delete-slug")
    ds_.add_argument("--slug", required=True)
    ds_.add_argument("--project", default=None)
    ds_.add_argument("--apply", action="store_true")
    su = sub.add_parser("species-upload")
    su.add_argument("--images", required=True)
    su.add_argument("--labels", required=True)
    su.add_argument("--split", default="train", choices=["train", "valid", "test"])
    su.add_argument("--batch", default="green")
    su.add_argument("--workers", type=int, default=8)
    su.add_argument("--per-species", type=int, default=0)
    args = ap.parse_args()

    {"whoami": cmd_whoami,
     "create-project": cmd_create_project,
     "list-folders": cmd_list_folders,
     "move-to-folder": cmd_move_to_folder,
     "sync-newest-slugs": cmd_sync_newest_slugs,
     "create-species-projects": cmd_create_species_projects,
     "upload": cmd_upload,
     "bulk-upload": cmd_bulk_upload,
     "delete-junk": cmd_delete_junk,
     "push-slug": cmd_push_slug,
     "delete-slug": cmd_delete_slug,
     "species-upload": cmd_species_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
