"""
rounds.py — v3.0.74 (2026-06-01)

Round tracking for the data-collection pipeline. Every slug downloaded
by brain_harvest is tagged with the current harvest round (v1, v2, ...).
This module bumps the round counter + backfills missing tags on already-
downloaded slugs.

User vision (2026-06-01):
  Round 1 = agent harvests → human reviews each slug+class → Roboflow
            batch name "agent-v1-{slug}"
  Round 2 = DINOv2 filters round 1's verified set → uploads filtered
            subset as "agent-v1-dinov2-v{X.Y}"
  Round 3 = human satisfied → "send to training"
  Round N+1 starts when user clicks `start_new_round`

CLI:
    python -m weed_optimizer_framework.tools.rounds status
    python -m weed_optimizer_framework.tools.rounds start-new
    python -m weed_optimizer_framework.tools.rounds backfill
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
REGISTRY = REPO / "results" / "framework" / "dataset_registry.json"


def _load() -> dict:
    if not REGISTRY.exists():
        print(f"FATAL: registry missing at {REGISTRY}", file=sys.stderr)
        sys.exit(2)
    return json.load(open(REGISTRY))


def _save(reg: dict) -> None:
    tmp = str(REGISTRY) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, REGISTRY)


def status(reg: dict | None = None) -> dict:
    """Return current_round + per-round slug counts."""
    reg = reg or _load()
    cur = int(reg.get("current_round", 1))
    rounds_meta = reg.get("rounds", {}) or {}
    ds = reg.get("datasets", {}) or {}
    # Tally per-round
    per_round = {}
    for slug, info in ds.items():
        if info.get("status") != "downloaded":
            continue
        r = int(info.get("harvest_round", 0) or 0)
        per_round.setdefault(r, []).append(slug)
    return {
        "current_round": cur,
        "rounds_meta": rounds_meta,
        "per_round_slugs": {str(k): v for k, v in sorted(per_round.items())},
        "per_round_counts": {str(k): len(v) for k, v in sorted(per_round.items())},
        "total_downloaded": sum(len(v) for v in per_round.values()),
    }


def start_new_round() -> dict:
    """Increment current_round. Subsequent brain_harvest tags slugs
    with the new value. Doesn't touch existing slugs."""
    reg = _load()
    cur = int(reg.get("current_round", 1))
    new = cur + 1
    reg["current_round"] = new
    rounds_meta = reg.setdefault("rounds", {})
    rounds_meta[str(new)] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at_ts": int(time.time()),
        "dinov2_subversions": [],
        "trained": False,
    }
    # Close previous round
    if str(cur) in rounds_meta and not rounds_meta[str(cur)].get("ended_at"):
        rounds_meta[str(cur)]["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    elif str(cur) not in rounds_meta:
        rounds_meta[str(cur)] = {
            "started_at": "unknown (round predates start_new_round)",
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dinov2_subversions": [],
            "trained": False,
        }
    _save(reg)
    return {
        "ok": True,
        "previous_round": cur,
        "new_round": new,
        "note": f"Round {cur} closed. Next brain_harvest call will tag slugs as round {new}.",
    }


def backfill_missing(default_round: int = 1) -> dict:
    """For slugs lacking harvest_round, set it to default_round (typically 1
    since v3.0.74 ships with current_round = 1 by default).

    Also ensures rounds[1] meta exists."""
    reg = _load()
    ds = reg.get("datasets", {}) or {}
    n_backfilled = 0
    for slug, info in ds.items():
        if info.get("status") != "downloaded":
            continue
        if "harvest_round" not in info or info["harvest_round"] is None:
            info["harvest_round"] = default_round
            info["harvest_round_ts"] = info.get("downloaded_at_ts",
                                                 int(time.time()))
            n_backfilled += 1
    if "current_round" not in reg:
        reg["current_round"] = default_round
    rounds_meta = reg.setdefault("rounds", {})
    if str(default_round) not in rounds_meta:
        rounds_meta[str(default_round)] = {
            "started_at": "backfilled — pre-v3.0.74 round",
            "started_at_ts": int(time.time()),
            "dinov2_subversions": [],
            "trained": False,
        }
    if n_backfilled or "current_round" not in reg:
        _save(reg)
    return {
        "ok": True,
        "backfilled_slugs": n_backfilled,
        "current_round_after": reg["current_round"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["status", "start-new", "backfill"])
    args = ap.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "start-new":
        print(json.dumps(start_new_round(), indent=2))
    elif args.command == "backfill":
        print(json.dumps(backfill_missing(default_round=1), indent=2))


if __name__ == "__main__":
    main()
