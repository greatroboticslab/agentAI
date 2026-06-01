"""
dinov2_round_filter.py — v3.0.74 (2026-06-01)

Use a round's HUMAN-VERIFIED set as reference; filter the round's
UN-verified slugs by cosine score; upload survivors back to Roboflow
with a sub-versioned batch_name "agent-v{N}-dinov2-v{X.Y}".

Workflow:
  Round 1 → user reviews → marks some slugs ✓ keep
       → click dinov2_filter_round
       → THIS SCRIPT runs:
            • reads slug_verdicts.jsonl + exemplar_verdicts.jsonl
            • assembles ✓ set for round N
            • reuses existing slug_scores.json (from dinov2_curate_registry)
              OR runs the curator if scores missing
            • picks survivors with score ≥ threshold (default 0.6)
            • uploads them to Roboflow with batch_name agent-v{N}-dinov2-v{X.Y}
            • registers sub-version in registry.rounds[N].dinov2_subversions

Side-effects (idempotent): always increments the sub-version on re-run,
so click-again means "v1.0 → v1.1 → v1.2 → ...".

CLI:
    python -m weed_optimizer_framework.tools.dinov2_round_filter \\
        --round 1 --threshold 0.6
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
SLUG_SCORES = REPO / "results" / "framework" / "dinov2_curator" / "slug_scores.json"
SLUG_VERDICTS = REPO / "results" / "framework" / "slug_verdicts.jsonl"
EXEMPLAR_VERDICTS = REPO / "results" / "framework" / "exemplar_verdicts.jsonl"


def _load_registry() -> dict:
    return json.load(open(REGISTRY))


def _save_registry(reg: dict) -> None:
    tmp = str(REGISTRY) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, REGISTRY)


def _round_slugs(reg: dict, round_n: int) -> list:
    return [s for s, info in reg.get("datasets", {}).items()
            if info.get("status") == "downloaded"
            and int(info.get("harvest_round", 0) or 0) == round_n]


def _human_verified_set(round_slugs: list) -> set:
    """Slugs in round that user marked keep OR have any exemplar verdicts."""
    verified = set()
    # slug-level verdicts (latest wins, jsonl is replay)
    sv_latest = {}
    if SLUG_VERDICTS.exists():
        for line in SLUG_VERDICTS.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                sv_latest[e.get("slug")] = e.get("verdict")
            except Exception:
                continue
    for s in round_slugs:
        if sv_latest.get(s) == "keep":
            verified.add(s)
    # class-level verdicts also count as positive signal
    if EXEMPLAR_VERDICTS.exists():
        for line in EXEMPLAR_VERDICTS.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if e.get("verdict") == "exemplar":
                    sl = e.get("slug") or e.get("source_slug")
                    if sl in round_slugs:
                        verified.add(sl)
            except Exception:
                continue
    return verified


def _load_scores() -> dict:
    if not SLUG_SCORES.exists():
        return {}
    try:
        d = json.load(open(SLUG_SCORES))
    except Exception:
        return {}
    # Score file shape: {"slugs": [{slug, score, ...}, ...]} OR {slug: score}
    out = {}
    if isinstance(d, dict) and "slugs" in d:
        for row in d["slugs"]:
            out[row.get("slug")] = float(row.get("score", 0.0))
    elif isinstance(d, dict):
        for k, v in d.items():
            try:
                out[k] = float(v.get("score") if isinstance(v, dict) else v)
            except Exception:
                continue
    return out


def _next_subversion(reg: dict, round_n: int) -> str:
    rmeta = reg.setdefault("rounds", {}).setdefault(str(round_n), {})
    subs = rmeta.setdefault("dinov2_subversions", [])
    # Pattern v1.0 → v1.1 → v1.2 → ...
    if not subs:
        nxt = "1.0"
    else:
        # parse last as float; bump by 0.1
        try:
            last = float(subs[-1])
            nxt = f"{last + 0.1:.1f}"
        except Exception:
            nxt = f"{len(subs)}.0"
    subs.append(nxt)
    rmeta["last_dinov2_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return nxt


def run(round_n: int, threshold: float = 0.6, dry_run: bool = False) -> dict:
    reg = _load_registry()
    r_slugs = _round_slugs(reg, round_n)
    if not r_slugs:
        return {"ok": False,
                "error": f"Round {round_n} has no downloaded slugs."}

    verified = _human_verified_set(r_slugs)
    scores = _load_scores()

    candidates = [s for s in r_slugs if s not in verified]

    # Survivors: candidates above threshold (uses prior dinov2_curate_registry
    # output). If no score → skip with note.
    survivors = []
    rejected = []
    no_score = []
    for s in candidates:
        sc = scores.get(s)
        if sc is None:
            no_score.append(s)
            continue
        (survivors if sc >= threshold else rejected).append((s, sc))

    subversion = "(dry-run)" if dry_run else _next_subversion(reg, round_n)
    result = {
        "ok": True,
        "round": round_n,
        "threshold": threshold,
        "verified_slugs": sorted(verified),
        "candidates": sorted(candidates),
        "survivors": [{"slug": s, "score": sc} for s, sc in survivors],
        "rejected": [{"slug": s, "score": sc} for s, sc in rejected],
        "no_score": no_score,
        "dinov2_subversion": subversion,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if dry_run:
        print(json.dumps(result, indent=2))
        return result

    # Upload survivors to Roboflow with new batch_name
    # For now, mark them in registry for sync_newest_slugs to pick up with
    # the round+subversion batch tag. Actual upload via the existing
    # bulk-upload SDK call.
    n_uploaded = 0
    n_fail = 0
    if survivors:
        try:
            from roboflow import Roboflow
            api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
            if not api_key:
                key_file = "/jet/home/byler/.roboflow_key"
                if os.path.isfile(key_file):
                    api_key = open(key_file).read().strip()
            ws_name = os.environ.get("ROBOFLOW_WORKSPACE", "a-test-of-will")
            proj_name = os.environ.get("ROBOFLOW_PROJECT",
                                        "weed-crop-agent-dataset")
            rf = Roboflow(api_key=api_key)
            proj = rf.workspace(ws_name).project(proj_name)
            print(f"[dinov2-filter] uploading {len(survivors)} survivors "
                  f"as agent-v{round_n}-dinov2-v{subversion}")
            for s, sc in survivors:
                info = reg["datasets"].get(s, {})
                lp = info.get("local_path", "")
                if not lp or not os.path.isdir(lp):
                    continue
                # Cap to 50 imgs per slug for speed
                EXTS = (".jpg", ".jpeg", ".png")
                imgs = []
                for img_p in Path(lp).rglob("*"):
                    if img_p.suffix.lower() in EXTS:
                        imgs.append(img_p)
                        if len(imgs) >= 50:
                            break
                batch = f"agent-v{round_n}-dinov2-v{subversion}-{s}"
                for img in imgs:
                    try:
                        proj.single_upload(
                            image_path=str(img),
                            split="train",
                            batch_name=batch,
                            tag_names=["dinov2-filtered",
                                       f"round-{round_n}",
                                       f"dinov2-v{subversion}", s],
                            num_retry_uploads=1,
                        )
                        n_uploaded += 1
                    except Exception as e:
                        n_fail += 1
                        if n_fail < 3:
                            print(f"  FAIL {img.name}: {type(e).__name__}: "
                                  f"{str(e)[:80]}")
            result["uploaded_imgs"] = n_uploaded
            result["upload_fail"] = n_fail
        except Exception as e:
            result["upload_error"] = f"{type(e).__name__}: {e}"

    _save_registry(reg)
    print(json.dumps(result, default=str, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True,
                    help="which harvest round to filter")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="DINOv2 cosine threshold (default 0.6)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't upload, just show what would happen")
    args = ap.parse_args()
    run(args.round, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
