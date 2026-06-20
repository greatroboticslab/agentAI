#!/usr/bin/env bash
# lab_server_setup.sh — one-time setup of the dashboard + MongoDB on the lab
# Ubuntu server (Unie). Run ON the lab server with sudo. Idempotent-ish.
#
#   bash deploy/lab_server_setup.sh
#
# CONFIRM/EDIT these before running:
REPO_DIR="${REPO_DIR:-$HOME/weed_llm_benchmark}"        # where the repo lives on lab server
REPO_GIT="${REPO_GIT:-https://github.com/harry567566/weed_llm_benchmark.git}"  # TODO confirm remote
DASH_PORT="${DASH_PORT:-8000}"
DASH_USER="${DASH_USER:-harry}"
PYTHON="${PYTHON:-python3}"
set -euo pipefail

echo "==[1/5] system packages =="
sudo apt-get update -y
sudo apt-get install -y git curl gnupg rsync "${PYTHON}" "${PYTHON}-venv" "${PYTHON}-pip"

echo "==[2/5] MongoDB Community (Ubuntu) =="
if ! command -v mongod >/dev/null 2>&1; then
  # Ubuntu 22.04 (jammy). For 24.04 use 'noble'. Confirm your release: lsb_release -cs
  REL="$(lsb_release -cs || echo jammy)"
  curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
    sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
  echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu ${REL}/mongodb-org/7.0 multiverse" | \
    sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
  sudo apt-get update -y
  sudo apt-get install -y mongodb-org
fi
sudo systemctl enable --now mongod
echo "  mongod status: $(systemctl is-active mongod)"

echo "==[3/5] repo + python venv =="
if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_GIT" "$REPO_DIR"; fi
cd "$REPO_DIR"
git pull --ff-only || true
$PYTHON -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
# dashboard core deps (training deps NOT needed here — training stays on cluster)
pip install fastapi "uvicorn[standard]" pymongo pillow opencv-python-headless numpy requests

echo "==[4/5] dashpass + env =="
# reuse the existing dashboard password (already shared w/ prof)
echo -n "dmz2l8L5CNOOSI6Q" > "$HOME/.dashpass" && chmod 600 "$HOME/.dashpass"
# point the app at the lab repo + local mongo
ENVF="$REPO_DIR/.env.labserver"
cat > "$ENVF" <<EOF
REPO_ROOT=$REPO_DIR
MONGO_URL=mongodb://127.0.0.1:27017
DASH_PORT=$DASH_PORT
EOF
echo "  wrote $ENVF (edit if Mongo auth/host differs)"

echo "==[5/5] systemd service (always-on dashboard, no GPU, no tunnel) =="
SVC=/etc/systemd/system/weed-dashboard.service
sudo tee "$SVC" >/dev/null <<EOF
[Unit]
Description=Weed dataset dashboard (FastAPI)
After=network-online.target mongod.service
Wants=network-online.target

[Service]
User=$(whoami)
WorkingDirectory=$REPO_DIR
EnvironmentFile=$ENVF
ExecStart=$REPO_DIR/.venv/bin/uvicorn weed_optimizer_framework.tools.dashboard_server:app --host 0.0.0.0 --port $DASH_PORT --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now weed-dashboard
echo "  dashboard: $(systemctl is-active weed-dashboard) on port $DASH_PORT"
echo ""
echo "DONE. Access on LAN: http://<lab-server-ip>:$DASH_PORT  (user $DASH_USER)"
echo "Next: migrate Mongo (mongorestore) + rsync image library from cluster — see MIGRATION_TO_LAB_SERVER.md"
