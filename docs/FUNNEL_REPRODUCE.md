# Reproducing the funnel audit on the platform

This document explains how to deploy the funnel audit and have the platform run it from a commit, and what "reproduce" means for each result. The contract is [FUNNEL_AUDIT.md](FUNNEL_AUDIT.md), which pre-registers the hypotheses, gates and decisions. The runner contract is [FUNNEL_AUDIT_RUNNER.md](FUNNEL_AUDIT_RUNNER.md), which pins the modules, formats and commands.

## 1. What is reproducible, and at which level

- **Decisions and audit numbers.** Every step of F3 to F9 runs as a versioned command of the platform (`python -m weed_optimizer_framework.tools.funnel <verb>`, or the `run_inc_funnel.sh` job wrapper). Seeds come from fixed text (`common.stable_int("funnel/v1/...")`). Every output records the sha256 of its inputs, of the code that made it, of the pre-registration core and of the domain config. From the same inputs, a rerun gives the same census, strata, sample, sheets, estimates and recovery overlay. A step refuses to run on inputs that do not match what it recorded.
- **External answers.** GBIF, the dataset cards (Mendeley, Hugging Face, Zenodo, papers), the Roboflow class lists and the iNaturalist known-truth photos are fetched once and cached, each with its sha256:
  - `funnel/taxonomy_cache.json`;
  - `funnel/cards/index.json`;
  - `funnel/fetch_manifest.json`;
  - the KT7 manifest.

  A reproduction reads the caches. A fresh fetch may differ, because these services change, and the cache hash then shows the difference.
- **Training metrics.** YOLO training on GPUs is not bit-exact across runs. No decision reads a single run: every comparison uses several seeds and the pinned rules (`gate.truth_detail`, the 5-seed permutation test of H10a).

## 2. Machines

| Machine | Role |
|---|---|
| Lab server | The dashboard and the campaign ticker (the autopilot), fetches from the network, the RL-A answer files, the claims register, the funnel sync to the cluster. |
| Cluster (Bridges-2) | Every computation: census, leak detection, judges, RL-B, estimates, recovery, training. Jobs are submitted by the platform with `sbatch`. |
| Workstation | A git checkout; `deploy/deploy_funnel.sh` runs here. |

## 3. Prerequisites

**Lab** (the dashboard setup is `deploy/lab_server_setup.sh` and `deploy/weed-dashboard.service`):
- The package tree at `~/weed_llm_benchmark`, with its `.venv`. This is a copy, not a git checkout.
- `~/.cluster_askpass.sh` (mode 600): the password helper for the cluster's data-transfer node, used by rsync.
- `~/.ssh/id_lab2cluster`: the key for the cluster login node (`deploy/lab_setup_cluster_key.sh`).
- `~/.roboflow_key` (mode 600): the Roboflow key the dashboard already reads. The funnel reads it through the `api_key_file` of the Roboflow specs in `funnel/domains/weed.json`.
- The environment of `deploy/run_dashboard_labserver.sh`: `CLUSTER_SSH` and `CLUSTER_REPO`. `CLUSTER_DATA_SSH` defaults to `byler@data.bridges2.psc.edu`.

**Cluster:**
- For the continuous loop's capacity arms (docs/CONTINUOUS_LOOP.md L-4): `yolo11s.pt` (sha256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`) and `yolo11m.pt` (`d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`) in `$REPO`, from the Ultralytics assets release v8.3.0 (placed 2026-09-28).
- The conda env `bench`. Checked 2026-09-28: torch 2.5.1, torchvision 0.20.1, timm 1.0.25, open_clip 3.3.0, transformers 5.8.0, scikit-learn 1.7.2, scipy 1.15.3, Pillow 12.0.0, OpenCV 4.10.0.
- The Hugging Face cache with `facebook/dinov2-base` and BioCLIP-2 (`imageomics/bioclip-2`).
- The ollama store `/ocean/projects/cis240145p/byler/ollama/models` with `qwen3.8:27b` (RL-B and the planner) and `glm-4.7-flash` / `gemma4` (the adversary).
- The INC splits v1 with `LOCK.json`, and the Step 1 artifacts under `INC_DIR/step1/`, built by `inc.verify` and `inc.select` (docs/INCREMENTAL_PROTOCOL.md, Step 1).

## 4. Deploy

```
weed_llm_benchmark/deploy/deploy_funnel.sh              # deploy, verify, record the replay, restart the dashboard
weed_llm_benchmark/deploy/deploy_funnel.sh --dry-run    # list what would be copied
```

**What it copies:**
- *Package paths* go to the lab tree and to both cluster copies. They include the continuous loop's packages (`tools/inc2/`, `tools/collect/`, the stream-mode autopilot and their tests, listed in the script): the nested git copy `$CLUSTER_REPO/weed_llm_benchmark/`, and the outer copy `$CLUSTER_REPO/` that the INC job scripts import. The paths are:
  - `tools/funnel/`, `tools/inc_autopilot/`, `tools/inc/`, `tools/model_router.py`, `tools/cwd12_species.py`;
  - the policy table (`tools/brain/policy_actions.json`, `approvals.py`);
  - the job scripts;
  - `results/framework/inc/funnel/prereg_v1.json` and `census_v0.json`;
  - the funnel and INC tests with their fixtures. The lab's replay gate runs these.
- *Git-root files* go beside the lab tree (`~/`) and to `$CLUSTER_REPO/`: the contract, the runner, this document, `RESEARCH_LOG.md`, and `docs/poster/figures_data.json`. The claims fixtures read the last two.

**What it checks:**
- *Before copying.* An `rsync --dry-run --checksum` against both cluster copies lists every file the deploy would change. The cluster's data-transfer node runs rsync only, so it cannot run commands. If a module INC experiments pin would change (`cwd12_species.py`, `inc/{driver,gate,splits,common,scorer,lora,train,verify,select,relevance,audit}.py`), the deploy refuses unless `--allow-pinned-change` is given.
- *After copying.* Every file must hash the same on the lab (sha256), and the same dry run against both cluster copies and the git-root files must list nothing. The script lists and fails on any mismatch.
- It then records the replay gate on the lab (`executor record-replay`; envelope autonomy needs a pass for the code in place) and restarts the dashboard.
- It writes `results/framework/inc/funnel/deployed.json` on the lab with the git HEAD it deployed.

**Before a deploy:** every file it replaces on the lab is backed up under `.deploy_backups/<ts>/`.

## 5. Start the platform

The campaign runs inside the dashboard's round scheduler, and its configuration is `~/.round_scheduler.json` on the lab. A person enables it:

```
python -m weed_optimizer_framework.tools.inc_autopilot.campaign enable --name weed_inc_v1 --by human:<actor> \
    --autonomy envelope --envelope-su 300 --daily-cap-su 150 --brain on
python -m weed_optimizer_framework.tools.inc_autopilot.campaign status --name weed_inc_v1
```

`funnel` defaults to on. Health is at `/api/health/inc`, and the event log is `results/framework/_brain/weed/inc/inc_campaign.jsonl`.

## 6. What the platform does, in order

The ticker decides each step from the evidence alone: diagnoses D17, D18 and D19; levers L10 to L14. A lever runs only when its inputs exist on the cluster, one item at a time, and each funnel step runs once per campaign (its lineage key).

| Step | Lever / verb | Runs on | Writes | Acceptance (FUNNEL_AUDIT.md §10) |
|---|---|---|---|---|
| F2 | D17 and D19 fire; the adversary pass is staged | lab; adversary job on the cluster | campaign ledger; `funnel/prospective_da.json` | the diagnoses cite resolvable evidence; the DA reply is validated before any estimate |
| — | L12 `fetch --what taxonomy` | lab | `funnel/taxonomy_cache.json` | every census name resolves or is recorded as unresolved |
| — | L11a `fetch --what cards` | lab | `funnel/cards/`, `fetch_manifest.json` | every file hashed; refusals listed; a stale index is fetched again once per domain config |
| F3 | L10 `census` | cluster | `funnel/census_v1.json`, `ledger.jsonl`, `funnel_ledger.json`, `name_status_v2.json` | the sums reconcile with Step 1: 545,318 crops; 2,049 / 8,732 / 2,756 / 531,781 verdicts; 1,060 admitted; 989 veto losses |
| F4 | L10 `leak`; L11a `kt7`; L11 `map --part geometry` | cluster; lab | `leak_v1.json`, `kt7/`, `relation_geometry_v1.json` | detector calibration (recall ≥ 0.95 per family, FPR ≤ 1 %) before any scan |
| F5 | L10 `embed-judges`, `qualify` | cluster | `judges/`, `judge_qualification.json` | the disjointness and circularity rules pass |
| F6 | L10 `draw` | cluster | `sample_v1.csv`; the sample lock in the prereg's amendments | frame sizes match the ledger |
| F7 | L10 `sheets`, `rl-b`; RL-A answers; `ingest`; `qualify --rl` | cluster; lab | `sheets_v1/`, `rl_answers/`, `gold_v1.csv`, `rl_qualification.json` | the labeller level is fixed from sentinels before any estimate |
| F8 | L10 `estimate` | cluster | `audit_v1.json` | H0 passes, or the campaign stops |
| F9 | L13 `recover` | cluster | `step1_r1/` | no guard stage touched; never-train checks on unmasked and masked images |
| F10 | L2 realloop_v2 and the panel arms | cluster | `realloop_v2/`, `rv2_*`, `panel.json` | as FUNNEL_AUDIT.md §9 |

## 7. What is not platform-native

- **RL-A** is an external model (DEC-1). Its answers enter as files under `funnel/rl_answers/RL-A/`. Without it, the platform has RL-B (qwen3.8:27b on the cluster) and a person's verify queue (L14). Which labeller serves each hypothesis is decided by sentinel qualification (FUNNEL_AUDIT.md §4.3), not by preference.
- **Decisions made by a person, or under the owner's delegation, are recorded, not re-derived:**
  - DEC-1 to DEC-10 (FUNNEL_AUDIT.md §13);
  - F1-R1 to F1-R4 (FUNNEL_AUDIT_RUNNER.md, "Decisions after F1"), including the name overrides in `funnel/domains/weed.json`;
  - the final claim status (card X12) belongs to the project owner.

## 8. Verification

- **Local test suite** (from `weed_llm_benchmark/`):
  - `for f in tests/test_funnel_*.py tests/test_inc_*.py; do python3 $f; done`
  - Replay cases R1 to R14, the mutation tests (M1 to M10) and the end-to-end pipeline test on a synthetic world are among them.
- **On the lab:** `executor record-replay`.
- **On the cluster:** the census reconciliation (§6, F3) is the first real-data check of the adapter.

**Differences between the test world and the live environment**, each fixed with a test that fails without the fix:
- The lab resolved the contract path against the cluster's REPO. `domain.contract_path` now searches up from the prereg (commit d94b1c6).
- The Mendeley file listing answers 206 and cannot be paged. A partially listed folder is recorded, and a fetch refuses one inside its scope (d94b1c6).
- The cluster login node has no rsync. The funnel sync now copies through the data-transfer node (d94b1c6).
- The Roboflow key lives in the dashboard's key file. Licences come from the fetched cards, and a stale cards index is fetched again (e16b9e1).
