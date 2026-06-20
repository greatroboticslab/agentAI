#!/usr/bin/env bash
# Runs on lab server (via systemd-run --user). Pulls the 55G image library from
# the Bridges-2 data node. Resumable (--partial). Logs to ~/img_rsync.log.
CL=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
LAB="$HOME/weed_llm_benchmark"
HOST=byler@data.bridges2.psc.edu
export SSH_ASKPASS="$HOME/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
mkdir -p "$LAB/downloads"
echo "[img-pull $(date -u +%FT%TZ)] start rsync downloads/ (55G) ..."
setsid -w rsync -az --partial --info=progress2 -e "$RSH" "$HOST:$CL/downloads/" "$LAB/downloads/"
echo "[img-pull $(date -u +%FT%TZ)] rsync exit=$?  local size: $(du -sh $LAB/downloads 2>/dev/null | cut -f1)"
