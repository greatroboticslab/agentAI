# Development guide

How to run, test, and contribute to the platform. For what it *is* and a visual
tour, see the top-level [README](../README.md).

## Run the dashboard locally

The dashboard is a server-rendered FastAPI app and boots without the heavy ML
stack — you can click through the whole UI on a laptop.

```bash
cd weed_llm_benchmark
pip install -r requirements-dev.txt      # light: no torch/ultralytics
echo mypassword > ~/.dashpass            # auth fails CLOSED without this
DASH_USER=me uvicorn weed_optimizer_framework.tools.dashboard_server:app --port 8000
# open http://localhost:8000  (log in: me / mypassword)
```

Creating a project needs MongoDB. For a throwaway local Mongo, either run
`mongod` yourself and `export AGENTAI_MONGO_URL=mongodb://127.0.0.1:27017/agentai`,
or the dashboard runs read-only against the JSON registry fallback without it.

## Run the tests

```bash
cd weed_llm_benchmark
pip install -r requirements-dev.txt
python tests/test_domain_config.py       # registry merge, per-domain config, round ledger, EDA
```

No Mongo, cluster, or network needed — these are pure-logic tests over the code
that mutates `dataset_registry.json` (the source of truth).

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push and PR to `main`:

| Job | What it guards |
|-----|----------------|
| **compile** | `py_compile` on every tracked `.py` — catches syntax/indentation breakage across all 180+ files |
| **unit-tests** | the pure-logic suite above (no heavy deps) |
| **lint** | `ruff` — informational, non-blocking |

CI intentionally does **not** install torch/ultralytics (minutes of setup for
little added safety). Training/eval correctness is validated on the cluster.

## Dependencies

- `requirements-dev.txt` — light set to run the dashboard + tests (this is what CI uses).
- `requirements.txt` — full runtime for training/inference on the cluster.

For a reproducible env (papers, lab-server rebuilds), lock from the working
cluster env: `pip freeze > requirements.lock`, then install from that. Don't
hand-guess exact ML pins.

## Repository layout

```
weed_llm_benchmark/weed_optimizer_framework/
  tools/dashboard_server.py   # the FastAPI dashboard (large; being de-monolithed)
  tools/db.py                 # Mongo + JSON registry (source of truth)
  tools/registry_lock.py      # advisory lock + atomic writes for the registry
  tools/mega_trainer.py       # dataset merge + YOLO training (holdout guards live here)
  orchestrator.py             # Brain harvest → autolabel loop
  config.py                   # env-overridable paths & settings
```

## Known engineering debt (see `CHANGELOG.md` v3.1.0 + the audit)

Tracked, not yet addressed — contributions welcome:

- `dashboard_server.py` is a ~13k-line monolith; extract routers incrementally
  (one small, tested extraction at a time — `dataset_eda.py` was the first).
- ~47 files hardcode the `/ocean/...` cluster path; route everything through
  `config.Config` so the lab-server migration is a config change, not a sweep.
- Broaden test coverage on `mega_trainer` merge and `roboflow_sync`.

## Conventions

- Product surface (UI, code, prompts, docs) is **English only**.
- Every change that ships must update `CHANGELOG.md`; experiment results also
  update `RESEARCH_LOG.md`.
- Never commit secrets — API keys/tokens load from untracked files
  (`~/.roboflow_key`, `~/.kaggle_token`, `~/.mongo_url`) or env vars.
