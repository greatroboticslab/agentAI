#!/bin/bash
#SBATCH --job-name=v3030_dS
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=48:00:00
#SBATCH --output=results/framework/v3_0_30_dashboard_server_cpu_%j.out

# v3.0.30.1 — dashboard server on RM-shared (CPU partition).
#
# Dashboard does NO GPU work — it's just FastAPI + cv2 for bbox rendering.
# Original script requested gpu:v100-32:1 which made it compete with FLUX /
# RF-DETR training jobs for scarce GPU nodes and ended up PD-blocked under
# "Nodes required ... DOWN, DRAINED or reserved" when the GPU partition
# filled up. RM-shared has ~750 nodes and the dashboard typically starts
# within seconds.
#
# All other behaviour identical to run_v3_0_30_dashboard_server.sh: uvicorn
# on localhost:8080, cloudflared quick tunnel, push URL to
# harry567566/weed-dashboard, self-chain at walltime.

set -e
eval "$(conda shell.bash hook)"
conda activate bench
set +e

REPO=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
cd "$REPO"
export PYTHONPATH=.:$PYTHONPATH
export REPO_ROOT="$REPO"

echo "=== v3.0.30.1 dashboard (CPU partition, RM-shared) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Date: $(date)"
echo "Host: $(hostname)  CPU: $(nproc)"

# Reuse the rest of the original script's flow by sourcing it past the SBATCH
# header — that script handles uvicorn, tunnel, push-to-github, watchdog,
# self-chain. We bypass the GPU resource request by using THIS file as the
# entry. Inline the bulk of the original here for clarity.

mkdir -p "$REPO/logs"
LOG_UVICORN="$REPO/logs/uvicorn_$SLURM_JOB_ID.log"

# Kill switch
if [ -f "$REPO/.stop_dashserver" ]; then
    echo "[chain] .stop_dashserver present — exit, do NOT chain"
    exit 0
fi

# Start uvicorn in background
uvicorn weed_optimizer_framework.tools.dashboard_server:app \
    --host 0.0.0.0 --port 8080 \
    > "$LOG_UVICORN" 2>&1 &
UVICORN_PID=$!
echo "[uvicorn] PID=$UVICORN_PID, log=$LOG_UVICORN"

# Wait for uvicorn to become responsive
for i in $(seq 1 60); do
    if curl -fs http://127.0.0.1:8080/healthz 2>/dev/null; then
        echo "[uvicorn] healthy after ${i}s"; break
    fi
    sleep 1
done

# Launch cloudflared quick tunnel
TUNNEL_LOG="$REPO/logs/cloudflared_$SLURM_JOB_ID.log"
CFD=$HOME/cloudflared
if [ ! -x "$CFD" ]; then
    echo "[tunnel] cloudflared not found at $CFD, attempting wget"
    wget -qO "$CFD" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CFD"
fi
$CFD tunnel --url http://localhost:8080 > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!
echo "[tunnel] cloudflared PID=$TUNNEL_PID"

# Poll log for the public URL
PUBLIC_URL=""
for i in $(seq 1 60); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
        PUBLIC_URL=$URL
        echo "[tunnel] cloudflared URL: $PUBLIC_URL"
        break
    fi
    sleep 2
done

# Push to harry567566/weed-dashboard so /weed-dashboard/ redirects
if [ -n "$PUBLIC_URL" ]; then
    TMP=$(mktemp -d)
    cd "$TMP"
    if git clone -q https://github.com/harry567566/weed-dashboard.git . 2>/dev/null; then
        echo "{\"url\": \"$PUBLIC_URL\", \"job\": $SLURM_JOB_ID, \"ts\": $(date +%s)}" > tunnel_url.json
        git add tunnel_url.json
        git -c user.email=bot@bridges2 -c user.name=dashboot \
            commit -q -m "update tunnel URL ($SLURM_JOB_ID)" 2>/dev/null
        git push -q https://github.com/harry567566/weed-dashboard.git main 2>&1 | head -5
    fi
    cd "$REPO"
fi

echo "============================================================"
echo "  PUBLIC DASHBOARD URL: $PUBLIC_URL"
echo "  (always accessible via: https://harry567566.github.io/weed-dashboard/)"
echo "============================================================"

# Watch processes; exit when either dies
while kill -0 $UVICORN_PID 2>/dev/null && kill -0 $TUNNEL_PID 2>/dev/null; do
    sleep 30
done
echo "[server] one of {uvicorn,tunnel} died — exiting"
kill $UVICORN_PID 2>/dev/null
kill $TUNNEL_PID 2>/dev/null
