"""
active_learning_round.py — orchestrates one round of the active-learning loop.

This is the CONDUCTOR. It doesn't run models itself; it chains existing
tools/sbatch actions via SLURM `--dependency=afterok:JOBID` so each stage
fires when the previous one succeeds. The pipeline per species per round:

  1. gather_green_exemplars: read confirmed exemplars from object_bank
     and the Roboflow cwd12-<species> project (verified=green).
  2. sbatch run_v3_0_50_owl_preannotate.sh
       OWL_SPECIES=<species>
       OWL_TARGET_DIR=<bucket-C or bucket-B-filtered-by-DINOv2>
       OWL_EXEMPLAR_CONFIG=<written from step 1>
       OWL_OUT_DIR=<results/framework/owl_red/<species>/round_<n>/labels>
     → produces red YOLO labels.
  3. (chained) sbatch run_v3_0_52_dinov2_route.sh on the SAME target dir
       — flags which images aren't even cwd12 (drops red proposals on
       those). Improves precision before human review.
  4. (chained) upload red proposals to Roboflow cwd12-<species> with
       batch_name=red-round-<n>.

Then humans review on Roboflow → approve red → flip to green → next round.

This skeleton (auto-loop iter 15) sets up the CLI + the per-step function
shape. The actual SLURM dependency chain and the gathering logic land in
subsequent iters once the user has had a chance to label seed data.

See:
  - memory/project_roboflow_pipeline_plan.md (active-learning section)
  - tools/owl_preannotate.py
  - tools/dinov2_route.py
  - tools/roboflow_sync.py species-upload --batch red

CLI:
  python -m weed_optimizer_framework.tools.active_learning_round \\
      --species Goosegrass \\
      --target-dir downloads/cottonweeddet12/valid/images \\
      --round 1 [--dry-run]

A whole-cwd12 round (12 species in parallel):
  python -m weed_optimizer_framework.tools.active_learning_round all \\
      --target-dir <unlabeled new harvest> --round 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))

CWD12 = (
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge",
)
BANK_DIR = REPO / "results" / "framework" / "synth_cutpaste" / "object_bank"
ROUND_DIR = REPO / "results" / "framework" / "active_learning_rounds"


def gather_green_exemplars(species: str, max_per_species: int = 10) -> list:
    """Read existing green exemplars (confirmed gold) for this species.

    Source order (most-trusted first):
      1. results/framework/synth_cutpaste/object_bank/<species>/*  (CWD12 cut-paste bank)
      2. results/framework/exemplar_sets/<species>/*  (user ✓-marked in /classes UI)
      3. (future) human-uploaded approved boxes from cwd12-<species> Roboflow project.

    Returns list of {image: abs_path, bbox_yolo: [cx, cy, w, h]} — for
    object_bank entries, the crop IS already the box so we use [0.5, 0.5,
    1.0, 1.0] (whole crop = the object). For real-image exemplars with
    bbox metadata, those should pass through unchanged.
    """
    out = []
    # object_bank/<species>/<crop>.jpg — each file is a tight crop
    bank_sp = BANK_DIR / species
    if bank_sp.is_dir():
        for p in sorted(bank_sp.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                out.append({"image": str(p), "bbox_yolo": [0.5, 0.5, 1.0, 1.0]})
            if len(out) >= max_per_species:
                break
    # TODO: read from exemplar_sets/<species> when E2/E3 lands
    # TODO: pull human-approved bboxes from Roboflow cwd12-<species>
    return out


def write_exemplar_config(species: str, exemplars: list, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"{species}.json"
    with open(cfg_path, "w") as f:
        json.dump({"species": species, "exemplars": exemplars,
                   "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   f, indent=2)
    return cfg_path


def sbatch_owl(species: str, target_dir: Path, exemplar_cfg: Path,
                out_dir: Path, dry_run: bool = False) -> Optional[str]:
    """Sbatch run_v3_0_50_owl_preannotate.sh with env overrides. Returns
    SLURM job ID (or None on dry-run / failure)."""
    env_overrides = {
        "OWL_SPECIES": species,
        "OWL_TARGET_DIR": str(target_dir),
        "OWL_EXEMPLAR_CONFIG": str(exemplar_cfg),
        "OWL_OUT_DIR": str(out_dir),
    }
    if dry_run:
        print(f"  [dry-run] sbatch run_v3_0_50_owl_preannotate.sh")
        for k, v in env_overrides.items():
            print(f"    {k}={v}")
        return None
    env = os.environ.copy(); env.update(env_overrides)
    r = subprocess.run(
        ["sbatch", "--parsable", str(REPO / "run_v3_0_50_owl_preannotate.sh")],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  [FAIL] sbatch OWL: {r.stderr.strip()}")
        return None
    job_id = r.stdout.strip()
    print(f"  sbatch OWL job_id={job_id}")
    return job_id


def sbatch_dinov2(target_dir: Path, dep_jobid: Optional[str],
                   dry_run: bool = False) -> Optional[str]:
    """Optional DINOv2 quality pass to drop red proposals on non-cwd12 imgs."""
    env_overrides = {
        "DINOV2_TARGET_DIR": str(target_dir),
        "DINOV2_OUT": str(target_dir / "_dinov2_routing.json"),
    }
    if dry_run:
        print(f"  [dry-run] sbatch run_v3_0_52_dinov2_route.sh")
        for k, v in env_overrides.items():
            print(f"    {k}={v}")
        return None
    args = ["sbatch", "--parsable"]
    if dep_jobid:
        args.append(f"--dependency=afterok:{dep_jobid}")
    args.append(str(REPO / "run_v3_0_52_dinov2_route.sh"))
    env = os.environ.copy(); env.update(env_overrides)
    r = subprocess.run(args, cwd=str(REPO), env=env,
                        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  [FAIL] sbatch DINOv2: {r.stderr.strip()}")
        return None
    job_id = r.stdout.strip()
    print(f"  sbatch DINOv2 job_id={job_id}")
    return job_id


def run_round_one_species(species: str, target_dir: Path, round_n: int,
                           dry_run: bool = False) -> dict:
    """One species, one round. Returns summary dict for the round log."""
    print(f"\n=== active-learning round {round_n}: {species} ===")
    round_out = ROUND_DIR / f"round_{round_n:03d}" / species
    round_out.mkdir(parents=True, exist_ok=True)
    cfg_dir = round_out / "exemplar_config"
    owl_out = round_out / "owl_red_labels"

    exemplars = gather_green_exemplars(species)
    print(f"  green exemplars: {len(exemplars)} (from object_bank)")
    if not exemplars:
        return {"species": species, "ok": False,
                "reason": "no_green_exemplars"}

    cfg_path = write_exemplar_config(species, exemplars, cfg_dir)
    owl_jid = sbatch_owl(species, target_dir, cfg_path, owl_out, dry_run=dry_run)
    if not dry_run and not owl_jid:
        return {"species": species, "ok": False, "reason": "owl_sbatch_failed"}

    dinov2_jid = sbatch_dinov2(target_dir, dep_jobid=owl_jid, dry_run=dry_run)

    summary = {
        "species": species, "round": round_n,
        "target_dir": str(target_dir),
        "exemplar_config": str(cfg_path),
        "n_exemplars": len(exemplars),
        "owl_out_dir": str(owl_out),
        "owl_job_id": owl_jid,
        "dinov2_job_id": dinov2_jid,
        "scheduled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # write per-species summary
    with open(round_out / "round_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", help="species name (CWD12) or 'all'")
    ap.add_argument("--target-dir", required=True,
                    help="dir of unlabeled images to annotate this round")
    ap.add_argument("--round", type=int, default=1,
                    help="round number (auto-bumped if conflict)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan without sbatch-ing anything")
    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        print(f"FATAL: target-dir not found: {target_dir}", file=sys.stderr)
        sys.exit(2)

    targets = list(CWD12) if args.species.lower() == "all" else [args.species]
    for sp in targets:
        if sp not in CWD12:
            print(f"FATAL: {sp} not in CWD12", file=sys.stderr)
            sys.exit(2)

    print(f"=== active-learning round {args.round} ({len(targets)} species) ===")
    print(f"  target: {target_dir}")
    print(f"  dry-run: {args.dry_run}")

    results = []
    for sp in targets:
        results.append(run_round_one_species(sp, target_dir, args.round,
                                              dry_run=args.dry_run))

    summary = {
        "round": args.round,
        "target_dir": str(target_dir),
        "scheduled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_species": results,
    }
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    out = ROUND_DIR / f"round_{args.round:03d}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWROTE: {out}")
    ok = sum(1 for r in results if r.get("ok") is not False)
    print(f"\nScheduled {ok}/{len(results)} species. "
          f"Watch SLURM: squeue -u $USER")


if __name__ == "__main__":
    main()
