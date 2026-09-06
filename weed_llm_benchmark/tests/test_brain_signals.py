#!/usr/bin/env python3
r"""WP3 deterministic-signal tests — no cluster, no Mongo, no model.

The fixtures are the campaign's own incidents, with the numbers the artifacts
carry:

  * **The dev case, 2026-08-29.** Jobs 44727703 / 44767709, both `--time=12:00:00`,
    hit the walltime at epoch 24 and 16 of 60 because the merged pool had grown
    to 8,583 iterations per epoch. `sacct` says State TIMEOUT, Elapsed 12:00:18
    against Timelimit 12:00:00. The bundle carries no job-scoped strategy
    artifact, because in that era the writer only ran at the end — which is part
    of what made the case hard, and is reproduced here rather than papered over.
    The corpus labels it `walltime_bound` + `pool_growth`; this asserts exactly
    that, with evidence, and nothing else at or above `warn`.
  * **The designed control inside that same bundle.** `[net] WARN: SOCKS proxy
    ... SKIPPING github, using Kaggle/HF only` is in every harvest log since
    June: GitHub is unreachable from the compute nodes and the collector is
    designed to skip it. Its correct verdict is "no incident". An arm that
    escalates it produces a false alarm on the very case it is meant to solve,
    so `source_degraded` must report it at `info` and never at `warn`.
  * **The failure v3.25.0 created.** A run under an Ultralytics `time=` cap ends
    early, writes a valid best.pt and reports COMPLETED. 24 of 60 epochs must
    read as `epochs_truncated` and must NOT read as `walltime_bound`: the wall
    was never reached, and counting it twice would inflate the class the
    benchmark reports.

Two invariants are asserted everywhere rather than in one place: a signal at or
above `info` always carries evidence with an artifact and a line, and a check
that cannot run reports `unknown` with a reason instead of disappearing or
raising.

The fixtures below are hand-built from the numbers the record states, not read
off the cluster, and three of them are worth confirming against the artifacts
themselves before any of this is quoted in a result:

  VERIFY: sacct -j 44727703,44767709 --format=JobID,JobName,State,Elapsed,\
          Timelimit,Start,End,ExitCode -P
          -> State TIMEOUT, Elapsed 12:00:18 and 12:00:20, Timelimit 12:00:00.
  VERIFY: grep -nE "^ +[0-9]+/60 .*[0-9]+/[0-9]+ \[" \
          results/framework/m1_merged_rndtrain_s1_44727703.out | tail -3
          -> the epoch-closing progress bar, and the 8,583 iterations per epoch
             the pool_growth fixture quotes. If the surviving log does not carry
             that shape, pool_growth has no line to cite on the dev case and the
             bundle builder has to stage the count instead.
  VERIFY: grep -c "SKIPPING github" results/framework/v3_0_43_brain_harvest_*.out
          -> non-zero in every harvest log, which is what makes the line a
             chronic environment condition rather than a degradation.

Run:  python3 tests/test_brain_signals.py
 (or) python -m pytest tests/test_brain_signals.py
"""
import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import signals  # noqa: E402

_fails = []


def ck(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        _fails.append(name)


def ck_eq(name, got, want, tol=1e-9):
    if isinstance(want, float) and isinstance(got, (int, float)):
        cond = abs(float(got) - want) <= tol
    else:
        cond = got == want
    print(("  ok   " if cond else "  FAIL ") + name
          + ("" if cond else "  (got %r, want %r)" % (got, want)))
    if not cond:
        _fails.append(name)


def ep(iso):
    """Epoch seconds for a UTC ISO stamp, so fixtures date consistently."""
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()


def by_name(rows):
    return {r["signal"]: r for r in rows}


def fired(rows, at_least="warn"):
    floor = signals._SEV_RANK[at_least]
    return {r["signal"] for r in rows if signals._SEV_RANK[r["severity"]] >= floor}


# ---------------------------------------------------------------- fixtures --
# The two lines of job 44727703's own log that decide the case, at the absolute
# line numbers a 22 MB `.out` puts them at, plus the chronic harvest line that
# travels with it.
DEV_OUT = [
    [8104, "[net] WARN: SOCKS proxy via bridges2-login011 failed "
           "(compute->login SSH disabled) - SKIPPING github, using Kaggle/HF only"],
    [8112, "      22/60      13.8G      1.043      0.771      0.918        139"
           "        640: 100%|##########| 8583/8583 [29:52<00:00,  4.79it/s]"],
    [8119, "                 Class     Images  Instances      Box(P          R"
           "      mAP50  mAP50-95): 100%|##########| 118/118 [00:41<00:00,  2.84it/s]"],
    [8123, "      24/60      13.8G      1.021      0.735      0.912        142"
           "        640: 100%|##########| 8583/8583 [29:41<00:00,  4.82it/s]"],
    [8128, "slurmstepd: error: *** JOB 44727703 ON v012 CANCELLED AT "
           "2026-08-29T17:52:41 DUE TO TIME LIMIT ***"],
]

WEED_CONFIG = {
    # The sealed per-recipe seed standard deviations, as db.DEFAULT_DOMAIN_CONFIG
    # carries them. A signal reads them from here and never from its own code.
    "noise_floor": {"merged_curated": 0.005, "merged_raw": 0.009,
                    "cwd12_core": 0.006},
    "budget": {"su_envelope": 4000, "per_round_cap": 60},
    "target_metric": "mAP50-95",
}


def dev_bundle():
    """The 2026-08-29 double TIMEOUT, as an evidence bundle."""
    return {
        "bundle_id": "sched-walltime-double-timeout-20260829",
        "domain": "weed", "round": 4, "step": "train", "built_ts": "2026-08-29",
        "sections": {
            "ledger": {
                "tick_s": 120, "mongo_ok": True, "config": WEED_CONFIG,
                "rounds": [
                    {"round_num": 1, "metrics": {"mAP50-95": 0.6019}, "steps": {}},
                    {"round_num": 2, "metrics": {"mAP50-95": 0.5919}, "steps": {}},
                    {"round_num": 3, "metrics": {"mAP50-95": 0.5951}, "steps": {
                        "train": {"status": "done", "at": "2026-08-27T21:10:04Z",
                                  "job": "44688201",
                                  "params": {"iterations_per_epoch": 6246,
                                             "epochs": 60, "tier": "curated"}}}},
                    {"round_num": 4, "metrics": {}, "steps": {
                        "collect": {"status": "done", "at": "2026-08-29T04:41:12Z",
                                    "job": "44720118"},
                        "train": {"status": "failed", "at": "2026-08-29T17:53:02Z",
                                  "job": "44727703", "detail": "TIMEOUT",
                                  "attempts": [{"status": "running", "job": "44727703",
                                                "at": "2026-08-29T05:52:41Z"}]}}},
                ],
            },
            "sacct": [
                {"JobID": "44727703", "JobName": "rndtrain", "State": "TIMEOUT",
                 "Elapsed": "12:00:18", "Timelimit": "12:00:00",
                 "Start": "2026-08-29T05:52:23", "End": "2026-08-29T17:52:41",
                 "ExitCode": "0:0"},
                {"JobID": "44727703.batch", "JobName": "batch", "State": "CANCELLED",
                 "Elapsed": "12:00:20", "Timelimit": "",
                 "Start": "2026-08-29T05:52:23", "End": "2026-08-29T17:52:43",
                 "ExitCode": "0:15"},
            ],
            "out_tail": {"artifact_id": "job_44727703.out",
                         "path": "<REPO>/results/framework/"
                                 "m1_merged_rndtrain_s1_44727703.out",
                         "sha256": "d0" * 32, "lines": DEV_OUT},
            "results_csv": {"rows": 24, "best": 0.5951, "last": 0.5904,
                            "epoch_time_s": 1789.4,
                            "mtime": ep("2026-08-29T17:50:00")},
            # The job-scoped artifact does not exist for these two runs.
            "strategy": None,
            "trace": [],
            "slug_scores": None,
            "registry_diff": {"slugs_added": 2, "images_added": 2337},
            "harvest": {"per_source": {"kaggle": 12, "huggingface": 7, "github": 0},
                        "previous_per_source": {"kaggle": 9, "huggingface": 11,
                                                "github": 0}},
            "resources": {"df_project": "5.9T of 6.9T used",
                          "quota_headroom_gb": 1024.0, "fs_free_tb": 1.0,
                          "squeue_depth": 8},
            "su": {"round": 24.0, "campaign": 215.0, "envelope": 4000.0},
            "corrections": [],
            "plan": None,
            "signals": None,
        },
        "export": {
            "case_id": "sched-walltime-double-timeout-20260829",
            "job_id": "44727703",
            "artifacts": [
                {"artifact_id": "job_44727703.out", "section": "out_tail",
                 "present": True, "mtime": ep("2026-08-29T17:52:41")},
                {"artifact_id": "m1_curated_seed1_44727703.json",
                 "section": "strategy", "present": False,
                 "reason": "no such file: the job-scoped writer only ran at the end"},
            ],
            "missing": {"strategy": "artifact m1_curated_seed1_44727703.json "
                                    "not readable",
                        "slug_scores": "no artifact mapped to this section and no "
                                       "inventory value given"},
        },
    }


def capped_bundle():
    """A post-v3.25.0 run: 60 epochs asked for, 24 ran under a 10.8 h cap."""
    return {
        "bundle_id": "capped-run", "domain": "weed", "round": 5, "step": "train",
        "sections": {
            "ledger": {"config": WEED_CONFIG, "mongo_ok": True, "rounds": [
                {"round_num": 5, "metrics": {}, "steps": {
                    "collect": {"status": "done", "at": "2026-09-05T09:00:00Z",
                                "job": "45300000"},
                    "train": {"status": "done", "at": "2026-09-05T20:50:00Z",
                              "job": "45301111",
                              "attempts": [{"status": "running", "job": "45301111",
                                            "at": "2026-09-05T10:00:00Z"}]}}}]},
            "sacct": [{"JobID": "45301111", "State": "COMPLETED",
                       "Elapsed": "10:49:33", "Timelimit": "12:00:00",
                       "Start": "2026-09-05T10:00:12", "End": "2026-09-05T20:49:45"}],
            "out_tail": {"artifact_id": "job_45301111.out", "sha256": "aa" * 32,
                         "lines": [
                             [201, "ultralytics: Starting training for 10.8 hours..."],
                             [980, "      24/60      13.7G      1.001      0.702"
                                   "      0.905        140        640: 100%|"
                                   "##########| 8583/8583 [26:14<00:00, 5.45it/s]"],
                             [999, "Stopping training early as time limit of 10.8 "
                                   "hours has been reached."]]},
            "results_csv": {"rows": 24, "best": 0.6042, "last": 0.6001,
                            "epoch_time_s": 1601.0,
                            "mtime": ep("2026-09-05T20:49:00")},
            "strategy": {"job_id": "45301111", "status": "done", "tier": "curated",
                         "epochs_requested": 60, "epochs_completed": 24,
                         "time_h": 10.8, "iterations_per_epoch": 8583,
                         "previous_iterations_per_epoch": 8583},
            "trace": [
                {"kind": "epoch", "epoch": 22, "elapsed_s": 35000.0,
                 "eta_total_s": 95454.0, "walltime_s": 43200.0},
                {"kind": "epoch", "epoch": 23, "elapsed_s": 36600.0,
                 "eta_total_s": 95478.0, "walltime_s": 43200.0},
                {"kind": "epoch", "epoch": 24, "elapsed_s": 38200.0,
                 "eta_total_s": 95500.0, "walltime_s": 43200.0},
                {"kind": "end", "ok": True, "epochs_requested": 60,
                 "epochs_completed": 24, "time_h": 10.8},
            ],
            "slug_scores": {"n": 61, "unscored": 0, "median": 0.55,
                            "mtime": ep("2026-09-05T09:30:00")},
            "registry_diff": {}, "plan": None, "signals": None,
            "harvest": {"per_source": {"kaggle": 4, "huggingface": 3, "github": 0},
                        "previous_per_source": {"kaggle": 6, "huggingface": 2,
                                                "github": 0}},
            "resources": {"quota_headroom_gb": 980.0, "fs_free_tb": 0.98},
            "su": {"round": 21.6, "campaign": 260.0, "envelope": 4000.0},
            "corrections": {"mirror_sha256": "c" * 64, "ledger_sha256": "c" * 64,
                            "chain_ok": True},
        },
        "export": {"job_id": "45301111", "artifacts": [], "missing": {}},
    }


def healthy_bundle():
    """A completed round with nothing wrong with it — the false-alarm control."""
    return {
        "bundle_id": "healthy-round-3", "domain": "weed", "round": 3,
        "step": "train",
        "sections": {
            "ledger": {"config": WEED_CONFIG, "mongo_ok": True, "tick_s": 120,
                       "rounds": [
                           {"round_num": 1, "metrics": {"mAP50-95": 0.5702},
                            "steps": {"train": {
                                "status": "done", "job": "44100001",
                                "at": "2026-08-20T10:00:00Z",
                                "params": {"iterations_per_epoch": 8100}}}},
                           {"round_num": 2, "metrics": {"mAP50-95": 0.5915},
                            "steps": {"train": {
                                "status": "done", "job": "44200001",
                                "at": "2026-08-23T10:00:00Z",
                                "params": {"iterations_per_epoch": 8210}}}},
                           {"round_num": 3, "metrics": {"mAP50-95": 0.6104},
                            "steps": {
                                "collect": {"status": "done", "job": "44300000",
                                            "at": "2026-08-26T06:00:00Z"},
                                "train": {"status": "done", "job": "44300111",
                                          "at": "2026-08-26T20:00:00Z",
                                          "attempts": [
                                              {"status": "running", "job": "44300111",
                                               "at": "2026-08-26T09:00:00Z"}]}}},
                       ]},
            "sacct": [{"JobID": "44300111", "State": "COMPLETED",
                       "Elapsed": "10:40:02", "Timelimit": "12:00:00",
                       "Start": "2026-08-26T09:00:11", "End": "2026-08-26T19:40:13"}],
            "out_tail": {"artifact_id": "job_44300111.out", "sha256": "bb" * 32,
                         "lines": [
                             [77, "[net] WARN: SOCKS proxy via bridges2-login011 "
                                  "failed (compute->login SSH disabled) - SKIPPING "
                                  "github, using Kaggle/HF only"],
                             [412, "[merge] min_dino_score=0.50 kept 55 of 80 "
                                   "scored slugs"],
                             [980, "      60/60      13.7G      0.902      0.611"
                                   "      0.877        141        640: 100%|"
                                   "##########| 8430/8430 [10:31<00:00, 13.3it/s]"],
                             [999, "60 epochs completed in 10.640 hours."]]},
            "results_csv": {"rows": 60, "best": 0.6104, "last": 0.6088,
                            "epoch_time_s": 638.0,
                            "mtime": ep("2026-08-26T19:40:00")},
            "strategy": {"job_id": "44300111", "status": "done", "tier": "curated",
                         "epochs_requested": 60, "epochs_completed": 60,
                         "iterations_per_epoch": 8430, "datasets_used": ["s"] * 55,
                         "mtime": ep("2026-08-26T19:40:10")},
            "trace": [
                {"kind": "epoch", "epoch": 58, "elapsed_s": 37000.0,
                 "eta_total_s": 38275.0, "walltime_s": 43200.0},
                {"kind": "epoch", "epoch": 59, "elapsed_s": 37640.0,
                 "eta_total_s": 38278.0, "walltime_s": 43200.0},
                {"kind": "epoch", "epoch": 60, "elapsed_s": 38280.0,
                 "eta_total_s": 38280.0, "walltime_s": 43200.0},
                {"kind": "end", "ok": True, "epochs_requested": 60,
                 "epochs_completed": 60},
            ],
            "slug_scores": {"n": 80, "unscored": 0, "p25": 0.42, "median": 0.57,
                            "p75": 0.71, "mtime": ep("2026-08-26T08:10:00")},
            "registry_diff": {"slugs_added": 3, "images_added": 1204},
            "harvest": {"per_source": {"kaggle": 8, "huggingface": 5, "github": 0},
                        "previous_per_source": {"kaggle": 6, "huggingface": 9,
                                                "github": 0}},
            "resources": {"quota_headroom_gb": 1024.0, "fs_free_tb": 1.0,
                          "squeue_depth": 4},
            "su": {"round": 21.3, "campaign": 190.0, "envelope": 4000.0},
            "corrections": {"mirror_sha256": "e" * 64, "ledger_sha256": "e" * 64,
                            "chain_ok": True},
            "plan": None, "signals": None,
        },
        "export": {"job_id": "44300111", "artifacts": [], "missing": {}},
    }


print("\n-- the 2026-08-29 dev case: two 12 h jobs killed at the wall --")
DEV = signals.detect(dev_bundle())
DEVX = by_name(DEV)
ck_eq("exactly walltime_bound and pool_growth fire at or above warn",
      fired(DEV), {"walltime_bound", "pool_growth"})
ck_eq("walltime_bound is crit", DEVX["walltime_bound"]["severity"], "crit")
ck_eq("its value is Elapsed over Timelimit (12:00:18 / 12:00:00)",
      DEVX["walltime_bound"]["value"], 1.0004)
ck("its reason names the sacct state and both times",
   all(s in DEVX["walltime_bound"]["reason"]
       for s in ("TIMEOUT", "12:00:18", "12:00:00", "44727703")))
ck_eq("pool_growth is warn", DEVX["pool_growth"]["severity"], "warn")
ck_eq("its value is the growth 6,246 -> 8,583 iterations per epoch",
      DEVX["pool_growth"]["value"], 0.3742)
ck("its reason carries both iteration counts",
   "8583" in DEVX["pool_growth"]["reason"] and "6246" in DEVX["pool_growth"]["reason"])

_bundle_lines = {(r["artifact_id"], r["line"]): r["text"]
                 for r in signals._Bundle(dev_bundle()).lines()}
ck("walltime_bound's first evidence is a real line of the job log",
   _bundle_lines.get((DEVX["walltime_bound"]["evidence"][0]["artifact_id"],
                      DEVX["walltime_bound"]["evidence"][0]["line"]))
   == DEVX["walltime_bound"]["evidence"][0]["quote"])
ck("and that line is the one SLURM wrote when it killed the job",
   "DUE TO TIME LIMIT" in DEVX["walltime_bound"]["evidence"][0]["quote"])
ck("pool_growth's first evidence is the log's own progress line",
   "8583/8583" in DEVX["pool_growth"]["evidence"][0]["quote"]
   and DEVX["pool_growth"]["evidence"][0]["artifact_id"] == "job_44727703.out")
ck("pool_growth also cites where the previous count came from",
   any(e["artifact_id"] == "section:ledger" and "6246" in e["quote"]
       for e in DEVX["pool_growth"]["evidence"]))
ck("epochs_truncated does not claim a truncated recipe on a run that was killed",
   DEVX["epochs_truncated"]["severity"] == "unknown")
ck("and says why: this era's job-scoped artifact was written only at the end",
   "epochs_requested" in DEVX["epochs_truncated"]["reason"])
ck("plateau stays unknown without a recipe rather than picking a seed std",
   DEVX["plateau"]["severity"] == "unknown"
   and "sealed seed std" in DEVX["plateau"]["reason"])

print("\n-- the designed control that travels inside the same bundle --")
SRC = DEVX["source_degraded"]
ck_eq("the chronic SKIPPING github line is info, not warn", SRC["severity"], "info")
ck("its reason says it is an environment fact, not a new degradation",
   "environment fact" in SRC["reason"])
ck("it still cites the line itself, so the verdict can be checked",
   any("SKIPPING github" in e["quote"] for e in SRC["evidence"]))
ck("and it is below the benchmark's detection severity",
   signals._SEV_RANK[SRC["severity"]] < signals._SEV_RANK["warn"])

print("\n-- the failure the v3.25.0 time cap created --")
CAP = signals.detect(capped_bundle())
CAPX = by_name(CAP)
ck_eq("a capped 24-of-60 run fires epochs_truncated",
      CAPX["epochs_truncated"]["severity"], "warn")
ck_eq("its value is the fraction of the recipe that ran",
      CAPX["epochs_truncated"]["value"], 0.4)
ck("its reason names the cap that ended it",
   "10.8" in CAPX["epochs_truncated"]["reason"])
_cap_ev = [e["quote"] for e in CAPX["epochs_truncated"]["evidence"]]
ck("and it carries the two epoch counts as evidence",
   any("epochs_requested=60" in q for q in _cap_ev)
   and any("epochs_completed=24" in q for q in _cap_ev))
ck("walltime_bound does NOT fire on it: the wall was never reached",
   "walltime_bound" not in CAPX)
ck("the check still ran, and detect_all reports what it saw",
   by_name(signals.detect_all(capped_bundle()))["walltime_bound"]["severity"] == "ok")
ck("its reason points at the signal that owns the shortfall",
   "epochs_truncated" in by_name(signals.detect_all(capped_bundle()))
   ["walltime_bound"]["reason"])
ck_eq("nothing else on a capped run reaches warn", fired(CAP), {"epochs_truncated"})

print("\n-- a healthy round is quiet --")
HEALTHY = signals.detect(healthy_bundle())
ck_eq("no healthy signal reaches warn", fired(HEALTHY), set())
ck("the only thing it reports is the chronic environment line",
   {r["signal"] for r in HEALTHY if r["severity"] == "info"} == {"source_degraded"})
HALL = by_name(signals.detect_all(healthy_bundle()))
for _name in ("walltime_bound", "epochs_truncated", "pool_growth", "gate_noop",
              "stale_artifact", "plateau", "budget", "disk_low",
              "ownership_violation", "mongo_down", "job_unknown"):
    ck("healthy: %s ran and found nothing" % _name, HALL[_name]["severity"] == "ok")
ck("a filter artifact written before the train step is not called stale",
   "results_csv" in HALL["stale_artifact"]["reason"]
   and "slug_scores" not in HALL["stale_artifact"]["reason"])

print("\n-- every remaining signal fires on its own evidence --")


def sig_of(bundle, want):
    return by_name(signals.detect(bundle)).get(want)


GATE = sig_of({"domain": "weed", "round": 7, "step": "train", "sections": {
    "ledger": {"rounds": [{"round_num": 7, "metrics": {}, "steps": {
        "collect": {"status": "done", "at": "2026-09-06T01:30:00Z", "job": "45399000"},
        "train": {"status": "running", "at": "2026-09-06T02:00:00Z",
                  "job": "45400001"}}}]},
    "sacct": [{"JobID": "45400001", "State": "RUNNING", "Elapsed": "01:10:00",
               "Timelimit": "12:00:00", "Start": "2026-09-06T02:00:12"}],
    "strategy": {"job_id": "45400001", "tier": "curated",
                 "datasets_used": ["a", "b", "c", "d"]},
    "slug_scores": {"n": 4, "unscored": 0, "median": 0.61,
                    "mtime": ep("2026-09-06T00:10:00")},
    "out_tail": {"artifact_id": "job_45400001.out", "lines": [
        [12, "[merge] min_dino_score=0.50 kept 4 of 4 scored slugs"]]},
}, "export": {"job_id": "45400001"}}, "gate_noop")
ck("gate_noop fires when the curated gate kept every slug it scored",
   GATE is not None and GATE["severity"] == "warn")
ck("and names both rules it broke",
   "older than" in GATE["reason"] and "removed nothing" in GATE["reason"])
ck("quoting the merge line the job itself printed",
   any("kept 4 of 4" in e["quote"] for e in GATE["evidence"]))

GATE0 = sig_of({"domain": "weed", "round": 7, "step": "train", "sections": {
    "strategy": {"tier": "curated", "datasets_used": ["a", "b", "c", "d"]},
    "slug_scores": {"n": 4, "unscored": 4, "mtime": ep("2026-09-06T00:10:00")}}},
    "gate_noop")
ck("gate_noop also catches a gate that scored nothing at all",
   GATE0 is not None and GATE0["severity"] == "warn" and GATE0["value"] == 4.0)
ck("and says the merge used slugs the gate never scored",
   "scored none of them" in GATE0["reason"])

_EMPTY_SRC = by_name(signals.detect({"sections": {"harvest": {"per_source": {}}}}))
ck("an empty per-source report is unknown, not a clean round",
   _EMPTY_SRC["source_degraded"]["severity"] == "unknown")

STALE = sig_of({"domain": "weed", "round": 8, "step": "train", "sections": {
    "ledger": {"rounds": [{"round_num": 8, "metrics": {}, "steps": {"train": {
        "status": "done", "at": "2026-09-06T06:00:00Z", "job": "45400002"}}}]},
    "sacct": [{"JobID": "45400002", "State": "COMPLETED", "Elapsed": "04:00:00",
               "Timelimit": "12:00:00", "Start": "2026-09-06T02:00:12"}],
    "results_csv": {"rows": 60, "best": 0.61, "last": 0.60, "epoch_time_s": 200.0,
                    "mtime": ep("2026-08-14T09:00:00")},
    "strategy": {"job_id": "45400002", "epochs_requested": 60,
                 "epochs_completed": 60},
}, "export": {"job_id": "45400002"}}, "stale_artifact")
ck("stale_artifact fires on a metric read from a 23-day-old results.csv",
   STALE is not None and STALE["severity"] == "warn" and STALE["value"] > 1.9e6)
ck("and dates the step from the cluster's own clock",
   "sacct Start" in STALE["reason"])

PLATEAU_CFG = {"noise_floor": {"merged_curated": 0.005}, "target_metric": "mAP50-95"}
PLAT = sig_of({"domain": "weed", "round": 9, "step": "train", "sections": {
    "ledger": {"config": PLATEAU_CFG, "rounds": [
        {"round_num": 6, "metrics": {"mAP50-95": 0.5921}, "steps": {}},
        {"round_num": 7, "metrics": {"mAP50-95": 0.5934}, "steps": {}},
        {"round_num": 8, "metrics": {"mAP50-95": 0.5928}, "steps": {}}]},
    "strategy": {"tier": "curated"}}}, "plateau")
ck("plateau fires when three rounds sit inside the recipe's seed noise",
   PLAT is not None and PLAT["severity"] == "warn")
ck_eq("its value is the spread of the three", PLAT["value"], 0.0013)
ck("its reason quotes the sealed std it read from the domain config",
   "0.0100" in PLAT["reason"] and "merged_curated" in PLAT["reason"])

# The campaign's own three scripted rounds land exactly on the boundary
# (spread 0.0100 against 2 x 0.005). The rule is a strict "below", so they do
# not fire — and the comparison is rounded before it is made, so the verdict on
# real data cannot turn on binary floating-point noise.
BOUNDARY = by_name(signals.detect_all({
    "domain": "weed", "round": 4, "step": "train", "sections": {
        "ledger": {"config": PLATEAU_CFG, "rounds": [
            {"round_num": 1, "metrics": {"mAP50-95": 0.6019}, "steps": {}},
            {"round_num": 2, "metrics": {"mAP50-95": 0.5919}, "steps": {}},
            {"round_num": 3, "metrics": {"mAP50-95": 0.5951}, "steps": {}}]},
        "strategy": {"tier": "curated"}}}))["plateau"]
ck("the scripted rounds sit exactly on the plateau boundary and do not fire",
   BOUNDARY["severity"] == "ok" and "0.0100" in BOUNDARY["reason"])

SRCD = sig_of({"domain": "weed", "round": 5, "step": "collect", "sections": {
    "harvest": {"per_source": {"kaggle": 0, "huggingface": 6, "github": 0},
                "previous_per_source": {"kaggle": 14, "huggingface": 5,
                                        "github": 0}},
    "out_tail": {"artifact_id": "job_45320000.out", "lines": [
        [88, "[net] WARN: SOCKS proxy via bridges2-login011 failed "
             "(compute->login SSH disabled) - SKIPPING github, using Kaggle/HF only"],
        [140, "[kaggle] HTTP 403 for every download: the job has no credentials, "
              "kaggle contributed 0 candidates"]]},
}}, "source_degraded")
ck("source_degraded fires on a source that yielded last round and not now",
   SRCD is not None and SRCD["severity"] == "warn")
ck("it names kaggle and its previous count, not the chronic github line",
   "kaggle returned 14" in SRCD["reason"] and "github" not in SRCD["reason"])
ck("and quotes the kaggle line, not the line that merely mentions kaggle",
   SRCD["evidence"][0]["line"] == 140)

JU = sig_of({"domain": "weed", "round": 6, "step": "train", "sections": {
    "ledger": {"rounds": [{"round_num": 6, "metrics": {}, "steps": {"train": {
        "status": "running", "at": "2026-09-06T04:00:00Z", "job": "45410000",
        "detail": "sacct state UNKNOWN", "attempts": [
            {"status": "running", "at": "2026-09-06T03:52:00Z", "job": "45410000",
             "detail": "state UNKNOWN"},
            {"status": "running", "at": "2026-09-06T03:54:00Z", "job": "45410000",
             "detail": "state UNKNOWN"},
            {"status": "running", "at": "2026-09-06T03:56:00Z", "job": "45410000",
             "detail": "state UNKNOWN"},
            {"status": "running", "at": "2026-09-06T03:58:00Z", "job": "45410000",
             "detail": "state UNKNOWN"}]}}}]}}}, "job_unknown")
ck("job_unknown fires after more than three consecutive UNKNOWN polls",
   JU is not None and JU["severity"] == "warn" and JU["value"] == 5.0)

BUD = sig_of({"domain": "weed", "round": 10, "step": "train", "sections": {
    "ledger": {"config": {"budget": {"su_envelope": 4000, "per_round_cap": 60}}},
    "su": {"round": 96.0, "campaign": 3400.0}}}, "budget")
ck("budget fires on a round past its cap",
   BUD is not None and BUD["severity"] == "warn")
ck_eq("and reports the worse of the two ratios as its value", BUD["value"], 1.6)
ck("naming both the round overrun and the campaign's 85 per cent",
   "per-round cap" in BUD["reason"] and "85%" in BUD["reason"])

DISK = sig_of({"domain": "weed", "round": 10, "step": "train", "sections": {
    "resources": {"df_project": "6.8T of 6.9T used", "quota_headroom_gb": 80.0,
                  "fs_free_tb": 0.1}}}, "disk_low")
ck("disk_low is crit when the project quota is nearly gone",
   DISK is not None and DISK["severity"] == "crit" and DISK["value"] == 80.0)
ck("and it says plainly which of the two numbers binds",
   "not the binding one" in DISK["reason"])
DISKFS = sig_of({"domain": "weed", "round": 10, "step": "train", "sections": {
    "resources": {"quota_headroom_gb": 900.0, "fs_free_tb": 0.2}}}, "disk_low")
ck("a low filesystem with healthy quota cannot raise it above info",
   DISKFS is not None and DISKFS["severity"] == "info")

OWN = sig_of({"domain": "weed", "round": 10, "step": "train", "sections": {
    "corrections": {"mirror_sha256": "a" * 64, "ledger_sha256": "b" * 64,
                    "chain_ok": True},
    "registry_diff": {"supervisory_reverts": [
        {"slug": "weeds-x9", "field": "quarantine", "was": True, "now": False}]}}},
    "ownership_violation")
ck("ownership_violation is crit when the mirror does not hash to the ledger",
   OWN is not None and OWN["severity"] == "crit")
ck("and it counts the supervisory revert as a second divergence", OWN["value"] == 2.0)

MONGO = sig_of({"domain": "weed", "round": 10, "step": "train", "sections": {
    "ledger": {"mongo_ok": False, "mongo_last_error_ts": ep("2026-09-06T03:00:00"),
               "ts": ep("2026-09-06T04:00:00"), "rounds": []}}}, "mongo_down")
ck("mongo_down is crit when the heartbeat says ledger writes are failing",
   MONGO is not None and MONGO["severity"] == "crit")
ck("and reports how long they have been failing", "3600 s" in MONGO["reason"])

print("\n-- a check that cannot run says so, and says why --")
EMPTY = signals.detect({})
ck_eq("an empty bundle produces one row per signal", len(EMPTY), len(signals.SIGNALS))
ck("every one of them is unknown",
   all(r["severity"] == "unknown" for r in EMPTY))
ck("every one carries a reason", all(len(r["reason"]) > 20 for r in EMPTY))
ck("and none carries evidence it does not have",
   all(r["evidence"] == [] for r in EMPTY))
ck_eq("the rows come back in the corpus's weight order",
      [r["signal"] for r in EMPTY], list(signals.SIGNALS))
for _shape in (None, [], "sections", 17, {"sections": "not an object"},
               {"sections": {"sacct": "rows", "ledger": 3, "trace": {"a": 1},
                             "harvest": [], "su": 9, "resources": None}}):
    rows = signals.detect(_shape)
    _label = "a bad-shape bundle" if isinstance(_shape, dict) else repr(_shape)
    ck("detect survives %s" % _label,
       len(rows) == len(signals.SIGNALS)
       and all(r["severity"] in signals.SEVERITIES for r in rows))

print("\n-- the evidence invariant --")
for _label, _bundle in (("dev case", dev_bundle()), ("capped", capped_bundle()),
                        ("healthy", healthy_bundle())):
    for row in signals.detect_all(_bundle):
        if signals._SEV_RANK[row["severity"]] >= signals._SEV_RANK["info"]:
            ck("%s: %s fired at %s and carries evidence"
               % (_label, row["signal"], row["severity"]), bool(row["evidence"]))
            ck("%s: %s evidence rows are addressed" % (_label, row["signal"]),
               all(isinstance(e.get("artifact_id"), str) and e["artifact_id"]
                   and isinstance(e.get("line"), int) and e["line"] >= 1
                   and isinstance(e.get("quote"), str) and e["quote"]
                   for e in row["evidence"]))
        else:
            ck("%s: %s is %s and claims no evidence"
               % (_label, row["signal"], row["severity"]), row["evidence"] == [])

# A check that fires without an address is downgraded rather than published: the
# invariant holds structurally, not by inspection of the twelve.
_saved = signals._CHECKS["mongo_down"]
try:
    signals._CHECKS["mongo_down"] = lambda b: signals._sig(
        "mongo_down", "crit", 1.0, "fired with nothing to point at", [])
    _guard = by_name(signals.detect({}))["mongo_down"]
    ck("a signal that fires with no evidence is downgraded to unknown",
       _guard["severity"] == "unknown")
    ck("and the downgrade says what it would have claimed",
       "not actionable" in _guard["reason"]
       and "fired with nothing" in _guard["reason"])
    signals._CHECKS["mongo_down"] = lambda b: (_ for _ in ()).throw(ValueError("boom"))
    _raised = by_name(signals.detect({}))["mongo_down"]
    ck("a check that raises becomes an unknown, not an exception",
       _raised["severity"] == "unknown" and "ValueError" in _raised["reason"])
finally:
    signals._CHECKS["mongo_down"] = _saved

print("\n-- thresholds live in one file, and only there --")
TH = signals.thresholds()
ck("thresholds.json parses and declares values", len(TH["values"]) >= 19)
ck("it reports no errors of its own", TH["errors"] == [])
ck("every threshold carries a why", all(TH["why"][k] for k in TH["values"]))
ck("every why is a sentence, not a label",
   all(len(TH["why"][k]) > 80 for k in TH["values"]))
ck("every value came from the file", set(TH["sources"].values()) == {"thresholds.json"})
_used = set()
_src = pathlib.Path(signals.__file__).read_text(encoding="utf-8")
for _k in TH["values"]:
    if '"%s"' % _k in _src:
        _used.add(_k)
ck("every declared threshold is read by a check", _used == set(TH["values"]))

ck_eq("a caller override wins over the file",
      signals.thresholds({"pool_growth": {"growth_fraction": 0.9}})
      ["values"]["pool_growth.growth_fraction"], 0.9)
ck("with the override's source recorded",
   signals.thresholds({"pool_growth": {"growth_fraction": 0.9}})
   ["sources"]["pool_growth.growth_fraction"] == "caller")
ck("raising the growth trigger silences pool_growth on the dev case",
   "pool_growth" not in fired(signals.detect(
       dev_bundle(), {"pool_growth": {"growth_fraction": 0.9}})))
ck("a flat override key works too",
   signals.thresholds({"plateau.window": 4})["values"]["plateau.window"] == 4)
ck("an override of a key the file does not declare is refused and reported",
   any("not declared" in e
       for e in signals.thresholds({"pool_growth.invented": 1})["errors"]))

os.environ["BRAIN_SIGNAL_POOL_GROWTH_GROWTH_FRACTION"] = "0.95"
try:
    _env = signals.thresholds({"pool_growth": {"growth_fraction": 0.9}})
    ck_eq("the environment wins over both the file and the caller",
          _env["values"]["pool_growth.growth_fraction"], 0.95)
    ck("and says so", _env["sources"]["pool_growth.growth_fraction"]
       == "env:BRAIN_SIGNAL_POOL_GROWTH_GROWTH_FRACTION")
    ck("so the dev case's pool growth no longer clears the trigger",
       "pool_growth" not in fired(signals.detect(dev_bundle())))
    os.environ["BRAIN_SIGNAL_POOL_GROWTH_GROWTH_FRACTION"] = "not a number"
    ck("a mistyped override is ignored and reported, never applied",
       signals.thresholds()["values"]["pool_growth.growth_fraction"] == 0.2
       and any("ignored" in e for e in signals.thresholds()["errors"]))
finally:
    os.environ.pop("BRAIN_SIGNAL_POOL_GROWTH_GROWTH_FRACTION", None)

_tmp = tempfile.mkdtemp(prefix="brain_signals_")
os.environ["BRAIN_SIGNAL_THRESHOLDS"] = os.path.join(_tmp, "absent.json")
try:
    _gone = by_name(signals.detect(dev_bundle()))
    ck("with no threshold file every check reports unknown rather than guessing",
       all(r["severity"] == "unknown" for r in _gone.values()))
    ck("and each names the key it could not find",
       "is not declared in" in _gone["walltime_bound"]["reason"])
finally:
    os.environ.pop("BRAIN_SIGNAL_THRESHOLDS", None)

print("\n-- the file does not shadow the other module that reads it --")
# citations.py will read a `citations` block from this same file if one is
# there. None is: that module owns its two numbers in its own table, and a
# duplicate here would silently win over them. One number, one owner.
_raw = json.loads((pathlib.Path(signals.__file__).parent
                   / "thresholds.json").read_text(encoding="utf-8"))
ck("thresholds.json declares the signal block and nothing it does not own",
   set(_raw) == {"_meta", "signals"})
try:
    from weed_optimizer_framework.tools.brain import citations
    _c = citations.thresholds()
    ck("so the citation validator still uses its own committed value",
       _c["values"]["min_quote_chars"] == 20
       and _c["sources"]["min_quote_chars"] == "default")
    ck("and reads nothing malformed here", _c["errors"] == [])
except ImportError:
    ck("citations.py is importable", False)

print("\n-- the public surface --")
ck_eq("SIGNALS is the twelve, in the corpus's weight order", list(signals.SIGNALS),
      ["walltime_bound", "epochs_truncated", "pool_growth", "gate_noop",
       "stale_artifact", "plateau", "source_degraded", "job_unknown", "budget",
       "disk_low", "ownership_violation", "mongo_down"])
ck("every signal has a check", set(signals._CHECKS) == set(signals.SIGNALS))
ck("every signal has a rationale paragraph",
   all(len(signals.explain(n)) > 200 for n in signals.SIGNALS))
ck("explain_all covers them in order",
   [r["signal"] for r in signals.explain_all()] == list(signals.SIGNALS))
ck("explain of an unknown name answers instead of raising",
   "not one of the signals" in signals.explain("no_such_signal"))
ck("explain of nothing at all still answers", isinstance(signals.explain(None), str))
ck("the rationale for source_degraded states the chronic-line rule",
   "SKIPPING github" in signals.explain("source_degraded"))
ck("the rationale for epochs_truncated says why it is the twelfth",
   "time cap" in signals.explain("epochs_truncated"))

_summary = signals.summary(dev_bundle())
ck("summary carries the bundle id, the signals and the rules they ran under",
   _summary["bundle_id"] == "sched-walltime-double-timeout-20260829"
   and _summary["checked"] == list(signals.SIGNALS)
   and _summary["thresholds"]["values"]["walltime_bound.eta_fraction"] == 0.95)
ck("summary is JSON-serialisable, which is how it reaches a results file",
   isinstance(json.dumps(_summary), str))

def cli(bundle, name):
    """Run the CLI over a bundle written to disk; return (exit code, stdout)."""
    path = os.path.join(_tmp, name)
    if bundle is not None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = signals.main(["detect", path])
    return code, buf.getvalue()


_code, _text = cli(dev_bundle(), "bundle.json")
ck("the CLI exits 1 on a bundle carrying a warn-or-worse signal", _code == 1)
ck("and prints the signals as JSON a script can read",
   [r["signal"] for r in json.loads(_text)] == [r["signal"] for r in DEV])
ck("the CLI exits 0 on a healthy bundle", cli(healthy_bundle(), "healthy.json")[0] == 0)
ck("the CLI reports a bundle it cannot read instead of raising",
   cli(None, "nope.json")[0] == 2)

shutil.rmtree(_tmp, ignore_errors=True)

if _fails:
    print("\nFAILED: %d -> %s" % (len(_fails), _fails))
    sys.exit(1)
print("\nALL PASS")
