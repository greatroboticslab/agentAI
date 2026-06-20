#!/usr/bin/env bash
# Runs ON the lab server. Gauges cluster data size + pulls SMALL metadata
# (registry, labeling events, dinov2 scores) from the Bridges-2 data node.
set -uo pipefail
CL=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
LAB="$HOME/weed_llm_benchmark"
export SSH_ASKPASS="$HOME/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=no"
HOST=byler@data.bridges2.psc.edu

echo "=== GAUGE: cluster data sizes ==="
setsid -w $RSH "$HOST" "du -sh $CL/downloads 2>/dev/null; echo '---n_datasets:'; ls $CL/downloads 2>/dev/null | wc -l; du -sh $CL/results/framework 2>/dev/null; ls -la $CL/results/framework/dataset_registry.json $CL/results/framework/labeling_events.jsonl 2>/dev/null" 2>&1 | grep -vE "Warning|Permanently|W A R N|PSC|Bridges|policies|LOG OFF|consent|help@|authorized use|^\s*$"

echo "=== PULL small metadata ==="
mkdir -p "$LAB/results/framework/dinov2_curator"
for f in results/framework/dataset_registry.json results/framework/labeling_events.jsonl results/framework/slug_verdicts.jsonl results/framework/dinov2_curator/slug_scores.json; do
  setsid -w rsync -az -e "$RSH" "$HOST:$CL/$f" "$LAB/$f" 2>/dev/null && echo "  pulled: $f" || echo "  (skip/absent: $f)"
done
echo "=== registry 内容概览 ==="
python3 -c "import json;d=json.load(open('$LAB/results/framework/dataset_registry.json'));ds=d.get('datasets',{});print('datasets:',len(ds),' total_imgs:',sum(v.get('local_images',0) for v in ds.values()))" 2>&1 | tail -2
echo "=== DONE ==="
