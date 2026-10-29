"""DS1's long path (lever L32X, lane TRAIN; docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29, DS.6, DS.9.1): the
scaled pair S0x and S2x (or S1ux under the substitution of DS.7.2) at the pair's compute, and any 1.2 M arm whose
loader projection is over the cold line (0.8 x 8 h), moved here at the same 1.2 M. Like V3F's v3fam_detr (L23K): a
one-epoch smoke first, then the seeds; run by run_inc2_long.sh (24 h walltime; 36 h only for DS.6's one
re-submission of S2x).

    python -m weed_optimizer_framework.tools.inc2.ds1_long smoke --arm S0X|S2X|S1UX|S1C|S1U|S2
    python -m weed_optimizer_framework.tools.inc2.ds1_long train --arm ARM --seed 0|1|2 [--walltime-h 24|36]
    python -m weed_optimizer_framework.tools.inc2.ds1_long status                       (read-only)

The pair's compute is fixed once, from S0x's smoke (DS.6): 3.6 M image-epochs if the projected seed is at most
19.2 h (0.8 x 24 h); otherwise 2.4 M for both S0x and S2x, recorded (ds1/v1/long/pair.json, written once). S2x's
own smoke must then project at most 19.2 h at the pair's compute; if it does not, S2x is submitted once with 36 h;
if it projects over 28.8 h it is recorded 'not run (walltime)' and the substitution of DS.7.2 applies. A moved
1.2 M arm's smoke must project at most 19.2 h at 1.2 M.

The smoke times the epoch itself: from the first training epoch's start to its end (Ultralytics' callbacks), the
one-time setup before it recorded apart (V3F deviation 5's lesson). Projection = epochs x epoch seconds + the
fixed overhead (0.5 h: materialising, the final export).

Training (one seed): the arm's manifest (S0x: S0's; S2x: S2's; S1ux: S1u's; a moved arm its own), checked and
guarded as inc2.train checks every run (check_manifest, guard_rows, materialise_run), YOLO11m from the arm's pinned
checkpoint, the cold recipe with the budget's epochs (round_half_up(I / N), N with the cwd12 floor's repeats),
warm-up fixed at 640 iterations and close_mosaic 10 % of the epochs (inc2.recipes' budget functions), seed k,
final-epoch weights (inc2.train.train_full's checks), exported once as ds1/v1/long/<ARM>/s<k>/final_ema.pt with
train.json (sha256, epochs, image-epochs, seconds). No checkpoint selection, no evaluation image read. Afterwards the
run's work directory keeps its train logs (results.csv, args.yaml) and manifest; its materialised data and
Ultralytics' .pt checkpoints are removed, as inc2.train._finish removes them, and listed in the record
(cleanup_removed). An earlier attempt's work directory is moved aside (<work>.attempt<k>), never deleted.

A refusal raises LongRefused; the CLI prints "[inc2.ds1_long] ERROR: ..." and exits 2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from ..inc import common as C
from . import common as C2
from . import ds1 as D
from . import recipes as RC

FMT_SMOKE = "inc2-ds1-long-smoke/1"
FMT_PAIR = "inc2-ds1-pair/1"
FMT_TRAIN = "inc2-ds1-long-train/1"
ARMS = D.LONG_ARMS + ("S1C", "S1U", "S2", "S1CR", "S0", "KD", "KT", "KF")
PAIR = D.LONG_ARMS
MOVED = tuple(a for a in ARMS if a not in PAIR)
LINE_H = D.WALLTIME_SHARE * D.LONG_WALLTIME_H          # 19.2
RESUBMIT_LINE_H = D.WALLTIME_SHARE * D.RESUBMIT_WALLTIME_H   # 28.8
FIXED_OVERHEAD_H = 0.5
DETECTOR = "m640"
NOT_RUN_WALLTIME = "not run (walltime)"


class LongRefused(D.Ds1Refused):
    pass


def log(msg):
    print("[inc2.ds1_long] %s" % msg, flush=True)


def long_dir():
    return D.root() / "long"


def arm_dir(arm):
    return long_dir() / arm


def pair_path():
    return long_dir() / "pair.json"


def manifest_arm(arm):
    """The manifest a long arm trains: S0x S0's, S2x S2's, S1ux S1u's; a moved 1.2 M arm its own."""
    if arm not in ARMS:
        raise LongRefused("--arm %r is not one of the long path's %s" % (arm, list(ARMS)))
    return D.LONG_PARENT.get(arm, arm)


def image_epochs(arm, pair=None):
    """The arm's compute: the pair's (3.6 M or 2.4 M, decided once from S0x's smoke) for S0x, S2x and S1ux; 1.2 M for
    a moved arm."""
    if arm in MOVED:
        return D.BUDGET_IMAGE_EPOCHS
    p = pair if pair is not None else D._read_json(pair_path())
    if arm != "S0X" and not isinstance(p, dict):
        raise LongRefused("the pair's compute is not decided yet: S0x's smoke decides it (DS.6)")
    if isinstance(p, dict):
        return int(p["image_epochs"])
    return D.PAIR_IMAGE_EPOCHS


def recipe(n_images, budget, seed=None):
    """The cold recipe at a budget of image-epochs (inc2.recipes' budget functions: epochs round_half_up(I / N),
    warm-up 640 iterations, close_mosaic 10 % of the epochs)."""
    r = RC.cold(DETECTOR)
    ep = RC.budget_epochs(int(n_images), budget=int(budget))
    r.update(epochs=ep, warmup_epochs=RC.budget_warmup_epochs(int(n_images), r["batch"]),
             close_mosaic=RC.budget_close_mosaic(ep))
    if seed is not None:
        r["seed"] = int(seed)
    return r


def projection(epoch_seconds, epochs):
    """Hours of a seed: epochs x the epoch's own seconds + the fixed overhead."""
    return float(epochs) * float(epoch_seconds) / 3600.0 + FIXED_OVERHEAD_H


def pair_decision(s0x_projection_3600):
    """DS.6: 3.6 M when S0x's projected seed is at most 19.2 h, else 2.4 M for both members of the pair."""
    return D.PAIR_IMAGE_EPOCHS if float(s0x_projection_3600) <= LINE_H + 1e-9 else D.PAIR_FALLBACK_IMAGE_EPOCHS


def walltime_branch(arm, proj_h):
    """(walltime hours or None, why): S2x (S1ux) at most 19.2 h -> 24 h; at most 28.8 h -> submitted once with 36 h;
    over 28.8 h -> not run (walltime) and the substitution of DS.7.2; S0x and a moved arm need 19.2 h (else not
    run (walltime))."""
    h = float(proj_h)
    if h <= LINE_H + 1e-9:
        return D.LONG_WALLTIME_H, "within 0.8 x 24 h"
    if arm in ("S2X", "S1UX") and h <= RESUBMIT_LINE_H + 1e-9:
        return D.RESUBMIT_WALLTIME_H, "over 19.2 h, within 28.8 h: submitted once with 36 h"
    return None, "%s: projected %.2f h, over %.1f h" % (NOT_RUN_WALLTIME, h, RESUBMIT_LINE_H if arm in (
        "S2X", "S1UX") else LINE_H)


def _rows(arm):
    from . import ds1_data as DD
    rows, rec = DD.arm_rows(manifest_arm(arm))
    return rows, rec


def tidy(work):
    """What a run's work directory holds that is reproducible and large, removed as inc2.train._finish removes it:
    the materialised data (moved aside, then purged) and Ultralytics' .pt checkpoints (the final weights are
    exported before this); the train logs (results.csv, args.yaml, the per-epoch metrics) and the manifest stay.
    Returns the paths removed (recorded in the smoke's or the seed's record)."""
    from . import train as T
    work = Path(work)
    removed = []
    if not work.is_dir():
        return removed
    trash = T.move_aside(work / "data", removed)
    for wd in sorted(p for p in work.rglob("weights") if p.is_dir()):
        T.drop_train_weights(wd.parent, removed)
    if trash is not None:
        T.purge_dir(trash)
    return removed


def set_aside(work):
    """An earlier attempt's work directory (a seed proposed again after a failure): tidied and renamed
    <work>.attempt<k>, its logs kept for the diagnosis, never deleted. None when there is none."""
    work = Path(work)
    if not work.exists():
        return None
    removed = tidy(work)
    k = 1
    while work.with_name("%s.attempt%d" % (work.name, k)).exists():
        k += 1
    dst = work.with_name("%s.attempt%d" % (work.name, k))
    os.rename(str(work), str(dst))
    return {"moved_to": str(dst), "removed": removed}


def _train_one(rows, rcp, out_dir, rec, epoch_times=None):
    """inc2.train's library steps on one seed: check_manifest, guard_rows, materialise_run, choose_cache and
    train_full (YOLO(init).train with the final-epoch checks); epoch_times gets each epoch's (start, end) monotonic
    times through Ultralytics' default callbacks, added for the call and removed after it."""
    from ultralytics.utils import callbacks as CB
    from . import train as T
    arm_rec = RC.resolve_arm(DETECTOR, repo=C.REPO, require_weights=True)
    init = Path(C.REPO) / arm_rec["model"]
    man = Path(out_dir) / "manifest.jsonl"
    man.parent.mkdir(parents=True, exist_ok=True)
    C.write_manifest(man, rows)
    checked, dh, info = T.check_manifest(man)
    rec["manifest"] = {k: info[k] for k in ("train_manifest_sha256", "n_train_images", "n_train_boxes")}
    rec["guard"] = T.guard_rows(checked, dh, production=not T.S.testing())
    data_yaml, rec["materialised"] = T.materialise_run(checked, Path(out_dir) / "data")
    cache = T.choose_cache(rcp, checked, rec)
    times = epoch_times if epoch_times is not None else []
    state = {}

    def on_start(_trainer):
        state["t"] = time.monotonic()

    def on_end(_trainer):
        times.append((state.get("t"), time.monotonic()))
    CB.default_callbacks["on_train_epoch_start"].append(on_start)
    CB.default_callbacks["on_train_epoch_end"].append(on_end)
    try:
        return T.train_full(init, data_yaml, rcp, Path(out_dir), 0, len(checked), rec, cache)
    finally:
        CB.default_callbacks["on_train_epoch_start"].remove(on_start)
        CB.default_callbacks["on_train_epoch_end"].remove(on_end)


def smoke(arm, trainer=None):
    """One epoch of the arm's manifest, its epoch timed; the projection at the arm's compute; for S0x, the pair's
    compute decided once (pair.json). ds1/v1/long/<ARM>/smoke.json written once. trainer(rows, recipe, out_dir, rec,
    epoch_times) replaces the GPU training (tests)."""
    D.check_frozen()
    D.require_timestamp()
    sp = arm_dir(arm) / "smoke.json"
    if sp.exists():
        raise LongRefused("%s exists: a smoke is taken once" % sp)
    rows, arec = _rows(arm)
    n = len(rows)
    t0 = time.monotonic()
    rec = {"warnings": []}
    times = []
    one = dict(recipe(n, D.BUDGET_IMAGE_EPOCHS, seed=0), epochs=1, close_mosaic=0)
    work = arm_dir(arm) / "smoke_work"
    earlier = set_aside(work)
    (trainer or _train_one)(rows, one, work, rec, times)
    total = time.monotonic() - t0
    if not times or times[0][0] is None:
        raise LongRefused("the smoke timed no epoch")
    ep_s = float(times[0][1] - times[0][0])
    out = {"format": FMT_SMOKE, "arm": arm, "manifest_arm": manifest_arm(arm), "images": n,
           "manifest_sha256": arec["sha256"], "epoch_seconds": ep_s, "setup_seconds": round(total - ep_s, 1),
           "work": str(work), "earlier_attempt": earlier, "decided_by": D.DECIDED_BY, "written_utc": D._utc()}
    if arm == "S0X":
        p36 = projection(ep_s, recipe(n, D.PAIR_IMAGE_EPOCHS)["epochs"])
        ie = pair_decision(p36)
        out.update(projection_h_3600k=p36, pair_image_epochs=ie)
        if not pair_path().exists():
            D.write_once(pair_path(), {"format": FMT_PAIR, "image_epochs": ie, "s0x_projection_h_3600k": p36,
                                       "line_h": LINE_H, "rule": "DS.6: 3.6 M if S0x's projected seed <= 19.2 h, "
                                       "else 2.4 M for both", "decided_utc": D._utc()})
    budget = image_epochs(arm)
    proj = projection(ep_s, recipe(n, budget)["epochs"])
    wall, why = walltime_branch(arm, proj)
    out.update(image_epochs=budget, epochs=recipe(n, budget)["epochs"], projection_h=proj, walltime_h=wall,
               status="complete" if wall is not None else NOT_RUN_WALLTIME, why=why, cleanup_removed=tidy(work))
    D.write_once(sp, out)
    log("%s smoke: epoch %.0f s, %d epochs at %d image-epochs -> %.2f h (%s)" % (
        arm, ep_s, out["epochs"], budget, proj, why))
    return out


def train(arm, seed, walltime_h=None, trainer=None):
    """One seed of a long arm at its compute, after its smoke (within the line, or S2x's one 36 h re-submission);
    the final weights exported once with train.json."""
    D.check_frozen()
    D.require_timestamp()
    seed = int(seed)
    if seed not in D.SEEDS:
        raise LongRefused("--seed %d is not one of %s" % (seed, list(D.SEEDS)))
    sm = D._read_json(arm_dir(arm) / "smoke.json")
    if not isinstance(sm, dict) or sm.get("status") != "complete":
        raise LongRefused("%s has no complete smoke within its walltime rule: no seed is trained" % arm)
    if walltime_h is not None and float(walltime_h) != float(sm["walltime_h"]):
        raise LongRefused("%s's smoke allows %s h, not %s h" % (arm, sm["walltime_h"], walltime_h))
    d = arm_dir(arm) / ("s%d" % seed)
    if (d / "train.json").exists():
        raise LongRefused("%s seed %d is trained: a seed is trained once" % (arm, seed))
    rows, arec = _rows(arm)
    if arec["sha256"] != sm["manifest_sha256"]:
        raise LongRefused("%s's manifest changed since its smoke" % arm)
    budget = image_epochs(arm)
    rcp = recipe(len(rows), budget, seed=seed)
    work = d / "work"
    earlier = set_aside(work)
    rec = {"warnings": []}
    t0 = time.time()
    weights, _sd = (trainer or _train_one)(rows, rcp, work, rec, [])
    dst = d / "final_ema.pt"
    from . import train as T
    sha = T.install_weights(Path(weights), dst)
    doc = {"format": FMT_TRAIN, "arm": arm, "seed": seed, "manifest_arm": manifest_arm(arm),
           "manifest_sha256": arec["sha256"], "images": len(rows), "image_epochs": budget, "recipe": rcp,
           "weights": str(dst), "weights_sha256": sha, "seconds": round(time.time() - t0, 1),
           "weights_epoch": rec.get("weights_epoch"), "warnings": rec.get("warnings"),
           "research_only": bool(arec.get("research_only")), "work": str(work), "earlier_attempt": earlier,
           "cleanup_removed": tidy(work), "decided_by": D.DECIDED_BY, "written_utc": D._utc()}
    D.write_once(d / "train.json", doc)
    log("%s seed %d trained: %s (%d epochs, %.0f s)" % (arm, seed, dst, rcp["epochs"], doc["seconds"]))
    return doc


def source(arm, seed):
    """What inc2.ds1's scores read of a long arm's seed: its exported weights, checked against train.json."""
    d = arm_dir(arm) / ("s%d" % int(seed))
    tj = D._read_json(d / "train.json")
    if not isinstance(tj, dict) or tj.get("format") != FMT_TRAIN:
        raise LongRefused("%s seed %s is not trained" % (arm, seed))
    w = d / "final_ema.pt"
    if w.is_symlink() or not w.is_file() or C.sha256_file(w) != tj["weights_sha256"]:
        raise LongRefused("%s does not hash as its train.json records" % w)
    return {"arm": arm, "seed": int(seed), "weights": str(w.resolve()), "weights_sha256": tj["weights_sha256"],
            "train_json_sha256": C.sha256_file(d / "train.json"), "research_only": bool(tj.get("research_only"))}


def status():
    """{pair, arms: {ARM: {smoke, trained}}} (read-only; stream_remote's DS1 status reads the same files)."""
    out = {"pair": D._read_json(pair_path()), "arms": {}}
    for a in ARMS:
        sm = D._read_json(arm_dir(a) / "smoke.json")
        out["arms"][a] = {"smoke": (sm or {}).get("status") if isinstance(sm, dict) else None,
                          "projection_h": (sm or {}).get("projection_h") if isinstance(sm, dict) else None,
                          "walltime_h": (sm or {}).get("walltime_h") if isinstance(sm, dict) else None,
                          "trained": sorted(int(p.parent.name[1:]) for p in arm_dir(a).glob("s[0-9]/train.json"))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="DS1's long path (L32X).")
    ap.add_argument("command", choices=("smoke", "train", "status"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--walltime-h", type=float, default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    try:
        if a.command == "smoke":
            smoke(a.arm)
        elif a.command == "train":
            if a.seed is None:
                raise LongRefused("train needs --seed")
            train(a.arm, a.seed, a.walltime_h)
        else:
            print(json.dumps(status(), indent=1, sort_keys=True))
    except (D.Ds1Refused, C2.Inc2Error, ValueError) as e:
        print("[inc2.ds1_long] ERROR: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
