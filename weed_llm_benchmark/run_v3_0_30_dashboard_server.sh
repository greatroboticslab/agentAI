#!/bin/bash
#SBATCH --job-name=v3030_dS
# v3.0.99.19: REVERT to GPU-shared. The v3.0.99.2 "move to RM-shared" was wrong for
# THIS allocation: account cis240145p is GPU-only (RM-shared → "mem-per-core higher
# than 2000M/core" + Invalid qos), so RM submit FAILS and the dashboard can't start.
# The dashboard is CPU work but GPU-shared is the only partition this account can use.
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

# v3.0.99.29: SELF-SYNC code to origin/main on every (re)start. Without this, the
# restart_dashboard HTTP button (restart_self → sbatch this script) silently ran
# whatever was checked out on the cluster, so "edit locally → push → restart" kept
# running STALE code (bit us 2026-06-11 deploying the DINO column under SSH login-
# throttle). fetch+reset --hard guarantees the nested git tree == latest pushed
# code before the nested→outer mirror below. reset --hard only touches tracked
# files; results/ + downloads/ (gitignored) are preserved.
echo "[sync] git fetch + reset --hard origin/main ..."
git fetch origin 2>&1 | tail -2
git reset --hard origin/main 2>&1 | tail -2
echo "[sync] HEAD now: $(git log --oneline -1 2>/dev/null)"

# v3.0.66 / v3.0.67: prevent nested/outer dashboard_server.py drift.
# Git tracks the nested copy at weed_llm_benchmark/weed_optimizer_framework/
# but the outer weed_optimizer_framework/ package is what Python imports
# because cwd is $REPO. Without this sync, a git pull updates code that
# never actually runs. Mirror nested → outer at every job start so the
# import always picks up the latest git HEAD.
#
# v3.0.67 (debugging 2026-05-31): Bridges-2 doesn't have rsync installed
# (verified `which rsync` returns nothing); previous attempt with
# `rsync -a --delete` silently no-op'd. Use `cp -ar` (force-overwrite)
# preceded by `rm -rf` of the destination — same effective semantics as
# rsync --delete (orphan files in outer get cleaned).
NESTED_PKG="$REPO/weed_llm_benchmark/weed_optimizer_framework"
OUTER_PKG="$REPO/weed_optimizer_framework"
if [ -d "$NESTED_PKG" ]; then
    rm -rf "$OUTER_PKG" 2>/dev/null
    if cp -ar "$NESTED_PKG" "$OUTER_PKG"; then
        echo "[sync] nested → outer weed_optimizer_framework ok ($(find $OUTER_PKG -type f | wc -l) files)"
    else
        echo "[sync] WARN cp failed (job continues with whatever code is on disk)"
    fi
else
    echo "[sync] WARN nested package not found at $NESTED_PKG"
fi

# v3.0.75.2: also mirror new run_*.sh scripts. v3.0.74 train_yolo placeholder
# silently failed because run_v3_0_74_yolo_trainer_placeholder.sh was only at
# the nested git-tracked path while subprocess actions look at outer ($REPO).
# Symlinks would be cleaner but cp keeps behavior consistent with above.
NESTED_SH_DIR="$REPO/weed_llm_benchmark"
if [ -d "$NESTED_SH_DIR" ]; then
    n_sh=0
    for sh in "$NESTED_SH_DIR"/run_v3_0_*.sh; do
        [ -f "$sh" ] || continue
        bn=$(basename "$sh")
        if [ ! -f "$REPO/$bn" ] || [ "$sh" -nt "$REPO/$bn" ]; then
            cp -p "$sh" "$REPO/$bn" && chmod +x "$REPO/$bn" && n_sh=$((n_sh+1))
        fi
    done
    # v3.0.78: also mirror run_mongo_node.sh (no v3_0 prefix → not caught above).
    if [ -f "$NESTED_SH_DIR/run_mongo_node.sh" ]; then
        cp -p "$NESTED_SH_DIR/run_mongo_node.sh" "$REPO/run_mongo_node.sh" \
            && chmod +x "$REPO/run_mongo_node.sh"
    fi
    echo "[sync] nested → outer run_v3_0_*.sh: $n_sh new/updated"
fi

echo "=== v3.0.30 Job-S (dashboard server + public tunnel) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Date: $(date)"

# Kill switch
if [ -f "$REPO/.stop_dashserver" ]; then
    echo "[chain] .stop_dashserver present — exit, do NOT chain"
    exit 0
fi

# --- 0. stand up co-located MongoDB (Phase 1) ---
# Bring up a single-instance user-space mongod on THIS node (binds 127.0.0.1),
# co-located with uvicorn so dashboard_server / tools.db connect via localhost.
# Fully guarded: if the binary isn't downloaded yet or the node has no internet,
# this fails softly and the app keeps running in JSON-fallback (tools.db handles
# the missing connection transparently). Pre-download once on a login node:
#   bash run_mongo_node.sh download
if [ -f "$REPO/.stop_mongo" ]; then
    echo "[mongo] .stop_mongo present — skipping Mongo startup (JSON-fallback)"
elif [ -f "$REPO/run_mongo_node.sh" ]; then
    echo "[mongo] attempting co-located mongod startup"
    REPO_ROOT="$REPO" bash "$REPO/run_mongo_node.sh" up \
        && echo "[mongo] up — db.py will use Mongo" \
        || echo "[mongo] WARN startup failed — app continues in JSON-fallback"
else
    echo "[mongo] run_mongo_node.sh not found — JSON-fallback"
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
    # cloudflared prints "https://....trycloudflare.com" on stderr.
    # IMPORTANT: filter out api.trycloudflare.com — that's Cloudflare's own
    # API endpoint, which appears in error messages when the tunnel HTTP POST
    # times out. Without this filter, a failed tunnel silently publishes
    # api.trycloudflare.com to GitHub Pages and the dashboard 404s.
    for i in $(seq 1 30); do
        URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_TUNNEL" \
              | grep -v '//api\.' | head -1)
        if [ -n "$URL" ]; then
            TUNNEL_URL="$URL"
            echo "[tunnel] cloudflared URL: $TUNNEL_URL"
            break
        fi
        # also detect explicit cloudflared failure to skip waiting
        if grep -q 'failed to request quick Tunnel\|context deadline exceeded' "$LOG_TUNNEL" 2>/dev/null; then
            echo "[tunnel] cloudflared upstream API failed; falling back"
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
