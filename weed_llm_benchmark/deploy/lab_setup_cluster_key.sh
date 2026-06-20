#!/usr/bin/env bash
# Runs ON lab. Sets up passwordless SSH lab->Bridges-2 so the dashboard can
# remote-sbatch/squeue. Adds a key via the data node (shared /jet/home), then
# tests SLURM on the login node. Needs ~/askpass.sh (cluster pw) for the one-time key add.
set -uo pipefail
export SSH_ASKPASS="$HOME/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
DATA=byler@data.bridges2.psc.edu
LOGIN=byler@bridges2.psc.edu

[ -f ~/.ssh/id_lab2cluster ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_lab2cluster -C "labserver-to-bridges2" >/dev/null 2>&1
PUB=$(cat ~/.ssh/id_lab2cluster.pub)

echo "=== 1) 通过 data 节点把 key 加到 cluster authorized_keys ==="
setsid -w $RSH "$DATA" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qF '$PUB' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; echo KEY_ADDED" 2>&1 | grep -E "KEY_ADDED|denied|error" | head -2

echo "=== 2) 用 KEY 测登录节点 + SLURM(无密码)==="
ssh -i ~/.ssh/id_lab2cluster -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 "$LOGIN" "echo LOGIN_OK via key; whoami; echo sbatch=\$(command -v sbatch); echo squeue=\$(command -v squeue); squeue -u byler -h | wc -l" 2>&1 | grep -E "LOGIN_OK|byler|sbatch=|squeue=|^[0-9]" | head -6
