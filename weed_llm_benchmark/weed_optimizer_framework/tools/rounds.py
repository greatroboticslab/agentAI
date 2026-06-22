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


def record_train_result(eval_json, round_n: int | None = None,
                        model_label: str = "yolo-clean") -> dict:
    """v3.0.122 — write a training/eval result back into a round's meta.

    This closes the train→round loop: after a clean-subset training job evals
    on the cwd12 gold holdout (eval_v3_0_23.py → results/v3_0_23_eval/
    v3_0_23_eval.json), stamp the round so the /rounds page shows a real mAP
    instead of "mAP pending".

    Runs on the CLUSTER at end-of-training, writing into the cluster registry
    (the source of truth). The next cluster→lab sync mirrors it to the
    dashboard — so the result is durable (a lab-side write would be clobbered).

    `eval_json` is a path to, or an already-parsed dict of, the eval JSON shape
    {"cwd12_test": {"mAP50_95", "mAP50", "n_classes_with_data", ...},
     "cwd12_valid": {...}}. `round_n` defaults to the registry's current_round
    (training uses the cumulative clean set = the latest snapshot). Idempotent:
    re-recording overwrites the round's train_results (latest wins)."""
    reg = _load()
    if round_n is None:
        round_n = int(reg.get("current_round", 1))
    round_n = int(round_n)

    if isinstance(eval_json, str):
        with open(eval_json) as f:
            ev = json.load(f)
        eval_src = eval_json
    else:
        ev = dict(eval_json or {})
        eval_src = "inline"

    def _m(d, k):
        try:
            return round(float(d.get(k)), 4)
        except (TypeError, ValueError, AttributeError):
            return None

    test = ev.get("cwd12_test") or {}
    valid = ev.get("cwd12_valid") or {}
    m_test = _m(test, "mAP50_95")
    m_valid = _m(valid, "mAP50_95")
    cands = [x for x in (m_test, m_valid) if x is not None]
    # headline = paper-grade holdout test mAP50-95; fall back to valid, then mean
    if m_test is not None:
        headline = m_test
    elif cands:
        headline = round(sum(cands) / len(cands), 4)
    else:
        headline = None

    train_results = {
        "map50_95": headline,
        "mAP50_95": headline,  # alias (the /rounds JS also checks this key)
        "map50": _m(test, "mAP50") if test else _m(valid, "mAP50"),
        "cwd12_test_map50_95": m_test,
        "cwd12_valid_map50_95": m_valid,
        "n_classes_with_data": (test.get("n_classes_with_data")
                                if test else valid.get("n_classes_with_data")),
        "model_label": model_label,
        # research goal is locked at cwd12 mAP50-95 >= 0.90 — surface the gap
        "gap_to_0_90": (round(0.90 - headline, 4) if headline is not None else None),
        "eval_source": eval_src,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    rounds_meta = reg.setdefault("rounds", {})
    meta = rounds_meta.setdefault(str(round_n), {
        "started_at": "unknown (predates record_train_result)",
        "started_at_ts": int(time.time()),
        "dinov2_subversions": [],
    })
    meta["trained"] = True
    meta["trained_at"] = train_results["recorded_at"]
    meta["train_results"] = train_results
    _save(reg)
    return {"ok": True, "round": round_n, "train_results": train_results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["status", "start-new", "backfill", "record-train"])
    ap.add_argument("--eval", help="path to eval JSON (record-train)")
    ap.add_argument("--round", type=int, default=None,
                    help="round number (record-train; default current_round)")
    ap.add_argument("--model-label", default="yolo-clean",
                    help="model label stored with the result (record-train)")
    args = ap.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "start-new":
        print(json.dumps(start_new_round(), indent=2))
    elif args.command == "backfill":
        print(json.dumps(backfill_missing(default_round=1), indent=2))
    elif args.command == "record-train":
        if not args.eval:
            print("record-train requires --eval <path>", file=sys.stderr)
            sys.exit(2)
        print(json.dumps(record_train_result(
            args.eval, round_n=args.round, model_label=args.model_label),
            indent=2))


if __name__ == "__main__":
    main()
