#!/usr/bin/env bash
# Start the dashboard on the lab server (foreground; wrap with setsid/tmux/systemd).
cd "$HOME/weed_llm_benchmark"
source .venv/bin/activate
export REPO_ROOT="$HOME/weed_llm_benchmark"
export DASH_PORT="${DASH_PORT:-8000}"
exec .venv/bin/uvicorn weed_optimizer_framework.tools.dashboard_server:app \
  --host 0.0.0.0 --port "$DASH_PORT" --workers 1
