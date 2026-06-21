#!/usr/bin/env bash
# Start the dashboard on the lab server (foreground; wrap with setsid/tmux/systemd).
cd "$HOME/weed_llm_benchmark"
source .venv/bin/activate
export REPO_ROOT="$HOME/weed_llm_benchmark"
export DASH_PORT="${DASH_PORT:-8000}"
# TESTING creds (user=1, pass=~/.dashpass=1). ⚠️ CHANGE before real deployment.
export DASH_USER="${DASH_USER:-1}"
# v3.0.99.40: lab-control mode — SLURM (sbatch/squeue/job_log) runs ON the cluster
# over an SSH key (lab = website + control + storage; cluster = compute only).
export CLUSTER_SSH="${CLUSTER_SSH:-byler@bridges2.psc.edu}"
export CLUSTER_REPO="${CLUSTER_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
export CLUSTER_SSH_KEY="${CLUSTER_SSH_KEY:-$HOME/.ssh/id_lab2cluster}"
# Bridges-2 = password auth (no user keys) → ssh reads pw from askpass non-interactively.
export SSH_ASKPASS="$HOME/.cluster_askpass.sh"
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0
exec .venv/bin/uvicorn weed_optimizer_framework.tools.dashboard_server:app \
  --host 0.0.0.0 --port "$DASH_PORT" --workers 1
