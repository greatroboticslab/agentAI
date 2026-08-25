#!/usr/bin/env bash
# Runs ON lab (via weed-sync.timer). Pulls NEW harvested data + records from the
# Bridges-2 data node (incremental rsync) and refreshes the lab MongoDB.
# Auth: password via ~/.cluster_askpass.sh + ControlMaster reuse. lab->cluster only.
#
# EVERY step here is bounded and every outcome is recorded. Twice this unit hung
# on a blocked ssh and stayed `activating` forever; systemd will not start a
# second run while the first lives, so the timer stopped firing and the platform
# served weeks-old data with nothing in any log. Two independent guards now make
# that impossible: each transfer has a wall clock (`timeout`), and each run
# writes a heartbeat that weed_optimizer_framework.tools.sync_health turns into a
# site-wide banner when it goes stale. The heartbeat is written even when this
# script is killed, via the EXIT trap.
set -uo pipefail
CL=/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark
LAB="$HOME/weed_llm_benchmark"
HOST=byler@data.bridges2.psc.edu
LOG="$HOME/sync_from_cluster.log"
HB="$LAB/results/framework/sync_status.json"
RUN_TS=$(date +%s)
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# ROOT CAUSE of both silent hangs (found 2026-08-25): SSH_AUTH_SOCK pointed at
# /run/user/1000/keyring/ssh, a gnome-keyring agent socket left over from a
# session where the ssh component was enabled. gnome-keyring now runs with
# --components=pkcs11,secrets, so the file exists but nothing accepts on it, and
# every ssh blocked forever in connect() before authentication ever started
# (wchan unix_wait_for_peer). ConnectTimeout does not cover agent sockets. This
# connection authenticates by password through the askpass and has no use for an
# agent at all, so the dependency is removed outright rather than bounded.
unset SSH_AUTH_SOCK

# IdentityAgent=none is the root-cause fix above. The rest is defence in depth,
# because the next wedge will be something else: ServerAlive* bounds a peer that
# stops answering, one password prompt bounds a credential that stopped working,
# and the outer `timeout` bounds anything neither of those anticipated.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 \
-o IdentityAgent=none -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
-o ConnectionAttempts=1 -o NumberOfPasswordPrompts=1 -o ControlMaster=auto \
-o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=3m"
RSH="ssh $SSH_OPTS"
META_TIMEOUT=${SYNC_META_TIMEOUT:-300}      # per small metadata file
DATA_TIMEOUT=${SYNC_DATA_TIMEOUT:-1800}     # per dataset tree (--partial resumes)
exec >>"$LOG" 2>&1

# --- heartbeat -------------------------------------------------------------
# NOTE: results/framework is deliberately NOT directory-rsynced below (only four
# named files are pulled), so this file survives a sync. Keep it that way.
_hb() {   # _hb <state> [stage-or-detail]
  RUN_TS="$RUN_TS" HB="$HB" ST="$1" DETAIL="${2:-}" python3 - <<'PY' 2>/dev/null || true
import json, os, time
p, st, detail = os.environ["HB"], os.environ["ST"], os.environ["DETAIL"]
try:
    cur = json.load(open(p))
except Exception:
    cur = {}
now = time.time()
cur.update(run_started_ts=float(os.environ["RUN_TS"]), updated_ts=now, stage=detail)
if st == "running":
    cur["in_flight"] = True
else:
    cur.update(in_flight=False, last_end_ts=now, last_outcome=st, detail=detail)
    if st == "ok":
        cur["last_success_ts"] = now
os.makedirs(os.path.dirname(p), exist_ok=True)
tmp = p + ".tmp"
json.dump(cur, open(tmp, "w"), indent=1)
os.replace(tmp, p)
PY
}
STAGE="starting"
FAIL=""
_on_exit() {
  rc=$?
  if [ -n "$FAIL" ]; then _hb failed "$FAIL"
  elif [ "$rc" -ne 0 ]; then _hb killed "killed during: $STAGE (rc=$rc)"
  fi
  echo "[sync $(date -u +%FT%TZ)] EXIT rc=$rc stage=$STAGE ${FAIL:+fail=$FAIL}"
}
trap _on_exit EXIT
trap 'echo "[sync] received TERM/INT during $STAGE"; exit 143' TERM INT

echo "[sync $(date -u +%FT%TZ)] START"
_hb running "starting"

# 0) A ControlMaster socket whose master has died makes every later ssh block
#    forever. Prove it answers, or remove it and let this run open a fresh one.
STAGE="controlmaster check"; _hb running "$STAGE"
CM="$HOME/.ssh/cm-byler@data.bridges2.psc.edu:22"
if [ -S "$CM" ]; then
  if timeout 20 ssh -o ControlPath="$CM" -O check "$HOST" 2>/dev/null; then
    echo "  ControlMaster alive"
  else
    echo "  ControlMaster socket is stale -> removing"; rm -f "$CM"
  fi
fi

# 1) small metadata (registry / labeling events / dinov2 scores / verdicts)
mkdir -p "$LAB/results/framework/dinov2_curator"
for f in results/framework/dataset_registry.json results/framework/labeling_events.jsonl results/framework/slug_verdicts.jsonl results/framework/dinov2_curator/slug_scores.json; do
  STAGE="pull $f"; _hb running "$STAGE"
  if setsid -w timeout -k 30 "$META_TIMEOUT" rsync -az -e "$RSH" "$HOST:$CL/$f" "$LAB/$f"; then
    echo "  pulled $f"
  else
    rc=$?
    echo "  FAILED $f (rc=$rc$([ $rc -eq 124 ] && echo ' TIMEOUT'))"
    # The registry is the one file the dashboard cannot be correct without.
    case "$f" in *dataset_registry.json)
      FAIL="registry pull failed (rc=$rc)"; echo "[sync] registry pull failed — aborting run"; exit 1;;
    esac
  fi
done

# 1.5) v3.0.110 FIX: rewrite local_path (cluster /ocean -> lab) + re-mirror Mongo
# IMMEDIATELY after pulling the registry, BEFORE the slow dataset rsync below.
# Previously fix ran only at the end (step 3), so during the long rsync the
# dashboard saw cluster paths -> 0 images on /classes. Run it here too (it is
# idempotent) to keep the broken window near-zero.
STAGE="fix_local_paths (early)"; _hb running "$STAGE"
cd "$LAB"; source .venv/bin/activate 2>/dev/null
REPO_ROOT="$LAB" CLUSTER_REPO="$CL" PYTHONPATH="$LAB" python deploy/fix_local_paths.py | tail -2

# 2) ALL dataset dirs the registry references (incremental: only new/changed).
#    Registry slugs live in datasets/ + results/leave4out/ + downloads/ — sync all 3.
for d in downloads datasets results/leave4out; do
  STAGE="sync $d"; _hb running "$STAGE"
  mkdir -p "$LAB/$d"
  if setsid -w timeout -k 60 "$DATA_TIMEOUT" rsync -az --partial -e "$RSH" "$HOST:$CL/$d/" "$LAB/$d/"; then
    echo "  synced $d ($(du -sh "$LAB/$d" 2>/dev/null | cut -f1))"
  else
    rc=$?
    echo "  FAILED $d (rc=$rc$([ $rc -eq 124 ] && echo ' TIMEOUT'))"
    # --partial keeps what arrived; a later run resumes. Record it, keep going:
    # a stalled image tree must not block the metadata refresh below.
    FAIL="${FAIL:+$FAIL; }$d rsync rc=$rc"
  fi
done

# 3) fix local_path (cluster prefix -> lab REPO_ROOT) THEN re-mirror Mongo + import events.
#    fix_local_paths.py rewrites the just-pulled registry so the dashboard finds images
#    at lab paths, then mirrors to Mongo (db.get_registry reads Mongo).
STAGE="fix_local_paths + mongo"; _hb running "$STAGE"
cd "$LAB"; source .venv/bin/activate
REPO_ROOT="$LAB" CLUSTER_REPO="$CL" PYTHONPATH="$LAB" python deploy/fix_local_paths.py | tail -2
REPO_ROOT="$LAB" PYTHONPATH="$LAB" python -m weed_optimizer_framework.tools.mongo_import_events | tail -1

STAGE="done"
if [ -n "$FAIL" ]; then
  echo "[sync $(date -u +%FT%TZ)] DONE WITH ERRORS: $FAIL"
else
  _hb ok "completed"
  echo "[sync $(date -u +%FT%TZ)] DONE"
fi
