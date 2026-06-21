#!/usr/bin/env bash
# Pull the MISSING dataset dirs (datasets/ + results/leave4out/) cluster->lab.
# These hold the actual registry slug images (downloads/ was the wrong dir).
CL=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
LAB="$HOME/weed_llm_benchmark"
HOST=byler@data.bridges2.psc.edu
[ -f ~/.cluster_askpass.sh ] || { printf '#!/bin/bash\necho "Robotics!!!"\n' > ~/.cluster_askpass.sh; chmod 700 ~/.cluster_askpass.sh; }
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
LOG="$HOME/datasets_pull.log"; exec >>"$LOG" 2>&1
echo "[pull-ds $(date -u +%FT%TZ)] START"
mkdir -p "$LAB/datasets" "$LAB/results/leave4out"
setsid -w rsync -az --partial --info=progress2 -e "$RSH" "$HOST:$CL/datasets/" "$LAB/datasets/" && echo "  datasets/ done: $(du -sh $LAB/datasets 2>/dev/null|cut -f1)"
setsid -w rsync -az --partial --info=progress2 -e "$RSH" "$HOST:$CL/results/leave4out/" "$LAB/results/leave4out/" && echo "  results/leave4out/ done: $(du -sh $LAB/results/leave4out 2>/dev/null|cut -f1)"
echo "[pull-ds $(date -u +%FT%TZ)] DONE"
