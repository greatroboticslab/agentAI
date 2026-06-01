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


# v3.0.76 (2026-06-01): per-round Roboflow PROJECT (not just batch_name).
# User vision: weed_crop_agent_dataset folder contains v1, v2, v3 as
# separately-clickable projects in the Roboflow UI. Round 1 maps to the
# pre-existing 'weed-crop-agent-dataset' project (legacy naming we don't
# rename to avoid disrupting 897 imgs already uploaded). Round 2+ uses
# 'weed-crop-agent-v{N}'. Free-tier cap = 10 projects total → ~7 rounds
# of capacity (account for cwd12-multiclass-v1 and a few others).
def round_project_name(round_n: int) -> str:
    """Derive Roboflow project slug for a given harvest round."""
    if int(round_n) == 1:
        return "weed-crop-agent-dataset"  # legacy v1
    return f"weed-crop-agent-v{int(round_n)}"


def round_project_url(workspace: str, round_n: int) -> str:
    """Browse URL for the round-N project."""
    return (f"https://app.roboflow.com/{workspace}/"
            f"{round_project_name(round_n)}/browse")


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


def _create_rf_project_and_assign(project_name: str, folder_name: str = "weed_crop_agent_dataset") -> dict:
    """v3.0.76 — create a new Roboflow project + assign to a folder.
    Uses the SDK (already installed in bench env) and the /groups REST API
    (verified working on free tier — see feedback_roboflow_folder_api memory).
    Returns dict with status info; never raises."""
    import urllib.request, urllib.error
    res = {"project": project_name, "folder": folder_name}
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        kfile = "/jet/home/byler/.roboflow_key"
        if os.path.isfile(kfile):
            key = open(kfile).read().strip()
    if not key:
        res["error"] = "no ROBOFLOW_API_KEY"
        return res
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "a-test-of-will")
    res["workspace"] = workspace

    # Step 1: Create project via SDK
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key)
        ws = rf.workspace(workspace)
        try:
            proj = ws.create_project(
                project_name=project_name,
                project_type="object-detection",
                project_license="MIT",
                annotation=project_name,
            )
            res["create_status"] = "created"
        except Exception as e:
            # Already exists is fine
            res["create_status"] = f"exists_or_err: {type(e).__name__}: {str(e)[:80]}"
    except Exception as e:
        res["create_status"] = f"SDK_FAIL: {type(e).__name__}: {e}"
        return res

    # Step 2: Find folder id + PATCH project into it
    try:
        url = f"https://api.roboflow.com/{workspace}/groups?api_key={key}"
        with urllib.request.urlopen(url, timeout=20) as r:
            folders = json.load(r).get("data", []) or []
        fid = next((f.get("id") for f in folders
                    if f.get("name") == folder_name or f.get("id") == folder_name), "")
        if not fid:
            res["folder_status"] = f"folder {folder_name!r} not found"
            return res
        res["folder_id"] = fid

        body = json.dumps({"projects": [project_name]}).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.roboflow.com/{workspace}/groups/{fid}/projects?api_key={key}",
            data=body, method="PATCH",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            res["folder_status"] = f"PATCH ok (HTTP {r.status})"
    except urllib.error.HTTPError as e:
        res["folder_status"] = f"PATCH HTTP {e.code}: {e.read()[:120]!r}"
    except Exception as e:
        res["folder_status"] = f"{type(e).__name__}: {e}"
    return res


def start_new_round(create_rf_project: bool = True) -> dict:
    """Increment current_round. Subsequent brain_harvest tags slugs
    with the new value. v3.0.76: also create a new Roboflow project
    'weed-crop-agent-v{N+1}' and assign to weed_crop_agent_dataset folder
    so the round's data has its own dedicated project."""
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

    rf_result = None
    if create_rf_project:
        proj_name = round_project_name(new)
        rf_result = _create_rf_project_and_assign(proj_name)
        rounds_meta[str(new)]["roboflow_project"] = proj_name
        rounds_meta[str(new)]["roboflow_setup"] = rf_result

    _save(reg)
    return {
        "ok": True,
        "previous_round": cur,
        "new_round": new,
        "roboflow_setup": rf_result,
        "note": f"Round {cur} closed. Next brain_harvest call will tag slugs as round {new}. "
                f"New project: {round_project_name(new)}",
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
