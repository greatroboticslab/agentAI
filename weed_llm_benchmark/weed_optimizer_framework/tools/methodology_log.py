"""
methodology_log.py — append-only structured logger for paper-grade
metrics of the active-learning loop.

The active-learning loop's paper contribution rests on quantitative
claims like:

    "Across N rounds, OWLv2-bootstrapped auto-labels reached 87% precision
     against human spot-check, with median human-review time dropping
     from 12s to 4s per image as the trained YOLOv8 took over from OWL."

To make that claim, we need a precision/effort log per round per species.
This module owns that log.

Output: a JSONL file at `results/framework/methodology/al_rounds.jsonl`.
Each line = one (round, species) outcome:

    {
      "ts": "2026-05-30T11:00:00",
      "round": 1,
      "species": "Goosegrass",
      "model_id": "google/owlv2-large-patch14-ensemble",
      "n_target_imgs": 200,
      "n_red_proposed": 312,
      "n_human_approved_green": 261,
      "n_human_rejected": 51,
      "auto_label_precision": 0.837,                       # approved / proposed
      "n_exemplars_in": 12,
      "median_human_review_sec": 4.1,                       # optional
      "total_human_review_sec": 850,
      "notes": ""
    }

Wired by:
- active_learning_round.py — calls `record_round_start(...)` at sbatch time.
- (future) the Roboflow approval webhook / poll → calls
  `record_round_complete(...)` after humans finish a batch.

CLI:
    python -m weed_optimizer_framework.tools.methodology_log show [--last N]
    python -m weed_optimizer_framework.tools.methodology_log summary
        # aggregates: precision over time, effort decay
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

REPO = Path(os.environ.get(
    "REPO_ROOT",
    "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark",
))
LOG_PATH = REPO / "results" / "framework" / "methodology" / "al_rounds.jsonl"


def _append(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def record_round_start(round_n: int, species: str, model_id: str,
                        n_target_imgs: int, n_exemplars_in: int,
                        owl_job_id: Optional[str] = None,
                        notes: str = "") -> None:
    """Called by active_learning_round.py when a per-species round is sbatched."""
    _append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": "round_start",
        "round": round_n,
        "species": species,
        "model_id": model_id,
        "n_target_imgs": n_target_imgs,
        "n_exemplars_in": n_exemplars_in,
        "owl_job_id": owl_job_id,
        "notes": notes,
    })


def record_round_owl_done(round_n: int, species: str,
                           n_red_proposed: int) -> None:
    """Called when OWL finishes; we know how many red proposals it made."""
    _append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": "round_owl_done",
        "round": round_n,
        "species": species,
        "n_red_proposed": n_red_proposed,
    })


def record_round_complete(round_n: int, species: str,
                           n_red_proposed: int,
                           n_human_approved_green: int,
                           n_human_rejected: int,
                           median_human_review_sec: Optional[float] = None,
                           total_human_review_sec: Optional[float] = None,
                           notes: str = "") -> None:
    """Called after the human-review step on Roboflow completes."""
    n_total = max(1, n_red_proposed)
    precision = n_human_approved_green / n_total
    _append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": "round_complete",
        "round": round_n,
        "species": species,
        "n_red_proposed": n_red_proposed,
        "n_human_approved_green": n_human_approved_green,
        "n_human_rejected": n_human_rejected,
        "auto_label_precision": round(precision, 4),
        "median_human_review_sec": median_human_review_sec,
        "total_human_review_sec": total_human_review_sec,
        "notes": notes,
    })


def _read_rows() -> list:
    if not LOG_PATH.is_file():
        return []
    out = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def cmd_show(args):
    rows = _read_rows()
    if args.last:
        rows = rows[-args.last:]
    for r in rows:
        print(json.dumps(r, default=str))


def cmd_summary(args):
    rows = _read_rows()
    # Group by (round, species), prefer most-recent complete event
    by_key: dict = {}
    for r in rows:
        if r.get("event") != "round_complete":
            continue
        key = (r["round"], r["species"])
        by_key[key] = r  # keep last

    if not by_key:
        print("No completed rounds yet.")
        return

    print(f"{'round':<6} {'species':<18} {'proposed':>9} {'approved':>9} "
          f"{'rejected':>9} {'precision':>10}  ts")
    for (rd, sp), r in sorted(by_key.items()):
        print(f"{rd:<6} {sp:<18} {r['n_red_proposed']:>9} "
              f"{r['n_human_approved_green']:>9} {r['n_human_rejected']:>9} "
              f"{r['auto_label_precision']:>10.3f}  {r['ts']}")

    # Roll-up: mean precision per round (across species)
    print()
    print("per-round mean precision:")
    by_round: dict = {}
    for (rd, _sp), r in by_key.items():
        by_round.setdefault(rd, []).append(r["auto_label_precision"])
    for rd in sorted(by_round):
        ps = by_round[rd]
        print(f"  round {rd:>3}: {sum(ps)/len(ps):.3f}  (n_species={len(ps)})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_show = sub.add_parser("show", help="dump rows (newest last)")
    p_show.add_argument("--last", type=int, default=0)
    sub.add_parser("summary", help="aggregate metrics per round")
    args = ap.parse_args()
    {"show": cmd_show, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
