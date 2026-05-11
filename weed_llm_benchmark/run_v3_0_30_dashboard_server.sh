#!/bin/bash
#SBATCH --job-name=v3030_dS
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_30_dashboard_server_%j.out

# v3.0.30 Job-S — live dashboard server with public tunnel.
#
# Components:
#   1. FastAPI server (uvicorn) on localhost:8080
#      Serves dashboard HTML + on-demand bbox-rendered samples from /ocean
#   2. Tunnel to expose to public internet — falls back through providers:
#      a. cloudflared (preferred, stable)
#      b. localhost.run via ssh -R (zero install, free, URL rotates)
#   3. Push the current public URL to harry567566/weed-dashboard repo's
#      tunnel_url.json so GitHub Pages can JS-redirect users to the live URL
#
# Self-chain: 48h walltime, then afterany resubmits.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH

echo "=== v3.0.30 Job-S (dashboard server + public tunnel) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"

# Kill switch
if [ -f "$REPO/.stop_dashserver" ]; then
    echo "[chain] .stop_dashserver present — exit, do NOT chain"
    exit 0
fi

# --- 1. start uvicorn ---
mkdir -p "$REPO/logs"
LOG_UVICORN="$REPO/logs/uvicorn_$SLURM_JOB_ID.log"
echo "[uvicorn] starting on localhost:8080 (log → $LOG_UVICORN)"
nohup python -m uvicorn \
    weed_optimizer_framework.tools.dashboard_server:app \
    --host 127.0.0.1 --port 8080 --workers 2 \
    > "$LOG_UVICORN" 2>&1 &
UVICORN_PID=$!
echo "[uvicorn] PID=$UVICORN_PID"

# Wait for uvicorn to be live
for i in $(seq 1 30); do
    if curl -sS --max-time 2 http://127.0.0.1:8080/healthz | grep -q '"ok": true'; then
        echo "[uvicorn] ready ($i sec)"
        break
    fi
    sleep 1
done

# Sanity check
curl -sS --max-time 5 http://127.0.0.1:8080/healthz | head -3

# --- 2. start tunnel ---
TUNNEL_URL=""
LOG_TUNNEL="$REPO/logs/tunnel_$SLURM_JOB_ID.log"

# Method A: cloudflared quick tunnel
if [ -x /ocean/projects/cis240145p/byler/harry/bin/cloudflared ]; then
    echo "[tunnel] trying cloudflared quick tunnel"
    nohup /ocean/projects/cis240145p/byler/harry/bin/cloudflared tunnel --no-autoupdate \
        --url http://127.0.0.1:8080 > "$LOG_TUNNEL" 2>&1 &
    CF_PID=$!
    # cloudflared prints "https://....trycloudflare.com" on stderr
    for i in $(seq 1 30); do
        URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_TUNNEL" | head -1)
        if [ -n "$URL" ]; then
            TUNNEL_URL="$URL"
            echo "[tunnel] cloudflared URL: $TUNNEL_URL"
            break
        fi
        sleep 1
    done
    if [ -z "$TUNNEL_URL" ]; then
        echo "[tunnel] cloudflared failed, killing PID=$CF_PID"
        kill $CF_PID 2>/dev/null
    fi
fi

# Method B: localhost.run via SSH (fallback)
if [ -z "$TUNNEL_URL" ]; then
    echo "[tunnel] trying localhost.run via SSH"
    nohup ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
        -R 80:localhost:8080 nokey@localhost.run \
        > "$LOG_TUNNEL" 2>&1 &
    SSH_TUN_PID=$!
    for i in $(seq 1 30); do
        URL=$(grep -oE 'https://[a-z0-9-]+\.(lhr\.life|localhost\.run)' "$LOG_TUNNEL" | head -1)
        if [ -n "$URL" ]; then
            TUNNEL_URL="$URL"
            echo "[tunnel] localhost.run URL: $TUNNEL_URL"
            break
        fi
        sleep 1
    done
fi

if [ -z "$TUNNEL_URL" ]; then
    echo "[tunnel] ALL TUNNEL METHODS FAILED — server running locally only"
    TUNNEL_URL="http://localhost-only-failed-to-expose"
fi

# --- 3. push URL to GitHub Pages redirect ---
echo "[git] pushing tunnel URL to harry567566/weed-dashboard"
TUNNEL_JSON='{"url":"'$TUNNEL_URL'","updated_at":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","slurm_job_id":"'$SLURM_JOB_ID'"}'
WD=$(mktemp -d)
GH_TOKEN_FILE=/jet/home/byler/.gh_pat
if [ ! -f "$GH_TOKEN_FILE" ]; then
    echo "[git] no PAT at $GH_TOKEN_FILE — cannot push. Manually run:"
    echo "      echo 'YOUR_PAT' > $GH_TOKEN_FILE && chmod 600 $GH_TOKEN_FILE"
    echo "      Tunnel URL: $TUNNEL_URL"
else
    GH_TOKEN=$(cat "$GH_TOKEN_FILE")
    git clone --depth 1 "https://$GH_TOKEN@github.com/harry567566/weed-dashboard.git" "$WD" 2>&1 | tail -3
    echo "$TUNNEL_JSON" > "$WD/tunnel_url.json"
    cd "$WD"
    git config user.email "byler@bridges2.psc.edu"
    git config user.name  "cluster-bot"
    git add tunnel_url.json
    git commit -m "update tunnel URL ($SLURM_JOB_ID)" 2>&1 | tail -2
    git push origin main 2>&1 | tail -2
    cd "$REPO"
fi

echo
echo "============================================================"
echo "  PUBLIC DASHBOARD URL: $TUNNEL_URL"
echo "  (always accessible via: https://harry567566.github.io/weed-dashboard/)"
echo "============================================================"
echo

# --- 4. block forever (until walltime kills us) ---
echo "[server] running until walltime"
# Use a loop that also notices if uvicorn dies
while kill -0 $UVICORN_PID 2>/dev/null; do
    sleep 60
done
echo "[server] uvicorn died"

# --- 5. self-chain ---
if [ -f "$REPO/.stop_dashserver" ]; then
    echo "[chain] kill switch present — not chaining"
else
    echo "[chain] scheduling next dashboard-server via afterany:$SLURM_JOB_ID"
    sbatch --dependency=afterany:$SLURM_JOB_ID \
        "$REPO/run_v3_0_30_dashboard_server.sh" || echo "[chain] sbatch failed"
fi

echo "=== Done $(date) ==="
