#!/usr/bin/env bash
# Deploys the funnel audit, the INC autopilot and the continuous loop (stream
# mode: tools/inc2, tools/collect, the stream job scripts) from a git checkout
# to the lab dashboard tree and to both cluster copies, then verifies what
# arrived. Contracts: docs/FUNNEL_REPRODUCE.md ("Deploy") and
# docs/CONTINUOUS_LOOP.md (6.5, "Verification"). Runs on a workstation with ssh
# access to the lab; the lab reaches the cluster (docs/FUNNEL_REPRODUCE.md,
# "Machines").
#
#   deploy/deploy_funnel.sh            pre-flight, deploy, verify, record the replay, restart the dashboard
#   deploy/deploy_funnel.sh --no-restart
#   deploy/deploy_funnel.sh --dry-run  list what would be copied; copy nothing
#
# Why the stream ships here: tools/inc_autopilot is one package, and its replay
# gate (executor.REPLAY_SCRIPTS) runs the stream's scenario and mutation cases,
# which import tools/inc2 and tools/collect. A deploy of inc_autopilot without
# them fails `record-replay` on the lab, and a failed replay record switches off
# envelope autonomy for every campaign (the live funnel audit included).
#
# Where things go:
#   * package paths (relative to weed_llm_benchmark/) -> the lab tree $LAB_TREE,
#     the cluster's nested git copy $CLUSTER_REPO/weed_llm_benchmark/ and its
#     outer copy $CLUSTER_REPO/ (INC job scripts import the outer copy; the
#     funnel and plan jobs the nested one; both must hold the same bytes);
#   * git-root paths (the contract, the runner, RESEARCH_LOG.md, the poster data
#     the claims fixtures read) -> the lab tree's parent (the lab keeps its
#     package tree under ~/weed_llm_benchmark and these beside it) and
#     $CLUSTER_REPO/.
# The lab tree is a copy, not a git checkout: every file this script replaces is
# first backed up under $LAB_TREE/.deploy_backups/<ts>/.
#
# Credentials: none in this script. The lab reaches the cluster's data-transfer
# node with the password its askpass helper supplies (~/.cluster_askpass.sh on
# the lab, mode 600) and the login node with its cluster key (CLUSTER_SSH_KEY).
set -euo pipefail

LAB_SSH="${LAB_SSH:-lab@lab-b660m-c}"
LAB_SSH_KEY="${LAB_SSH_KEY:-$HOME/.ssh/id_ed25519_lab}"
LAB_TREE_REL="${LAB_TREE_REL:-weed_llm_benchmark}"          # relative to the lab user's home
CLUSTER_REPO="${CLUSTER_REPO:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
CLUSTER_DATA_SSH="${CLUSTER_DATA_SSH:-byler@data.bridges2.psc.edu}"
CLUSTER_LOGIN_SSH="${CLUSTER_LOGIN_SSH:-byler@bridges2.psc.edu}"
CLUSTER_SSH_KEY_ON_LAB="${CLUSTER_SSH_KEY_ON_LAB:-.ssh/id_lab2cluster}"   # relative to the lab user's home

RESTART=1
DRY=0
ALLOW_PINNED=0
ALLOW_SHARED=0
for a in "$@"; do
  case "$a" in
    --no-restart) RESTART=0 ;;
    --dry-run) DRY=1 ;;
    --allow-pinned-change) ALLOW_PINNED=1 ;;
    --allow-shared-change) ALLOW_SHARED=1 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done
# Modules INC experiments pin by sha256 (the driver refuses to advance an experiment whose
# pinned module changed): a deploy that would change one on the cluster is refused unless
# --allow-pinned-change (and then every running experiment is checked by hand first).
PINNED_RE='weed_optimizer_framework/tools/(cwd12_species|inc/(driver|gate|splits|common|scorer|lora|train|verify|select|relevance|audit))\.py$'
# Libraries the live harvest, trainer and scheduler also run, shipped only so that the stream's job
# scripts find the outer and nested copies identical (they hash them and refuse on drift): a deploy that
# would change one on the cluster is refused unless --allow-shared-change.
SHARED_RE='weed_optimizer_framework/tools/(near_dup|semisup_labeler|registry_lock|mega_trainer|license_audit|brain/(su_ledger|policy))\.py$'
PY="${PYTHON:-python3}"

GIT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$GIT_ROOT/weed_llm_benchmark"
cd "$PKG"

# The package paths the funnel audit and the INC autopilot need (docs/FUNNEL_REPRODUCE.md, "What is deployed").
PKG_PATHS=(
  weed_optimizer_framework/tools/funnel
  weed_optimizer_framework/tools/inc_autopilot
  weed_optimizer_framework/tools/inc
  weed_optimizer_framework/tools/model_router.py
  # the /inc page and /api/health/inc (their health rows read the stream state)
  weed_optimizer_framework/tools/inc_dashboard.py weed_optimizer_framework/tools/inc_dashboard_page.py
  weed_optimizer_framework/tools/cwd12_species.py
  weed_optimizer_framework/tools/brain/policy_actions.json
  weed_optimizer_framework/tools/brain/approvals.py
  run_inc_funnel.sh run_inc_plan.sh run_inc_job.sh run_inc_build.sh run_inc_verify.sh
  results/framework/inc/funnel/prereg_v1.json
  results/framework/inc/funnel/census_v0.json
  tests/fixtures/funnel tests/fixtures/inc_replay
  # the continuous loop (docs/CONTINUOUS_LOOP.md 3, 9): the protocol package, the collector with its domain
  # configs (governance files), the scheduler guard, the SU rates, the stream job scripts
  weed_optimizer_framework/tools/inc2
  weed_optimizer_framework/tools/collect
  weed_optimizer_framework/tools/round_scheduler.py
  weed_optimizer_framework/tools/brain/su_rates.json
  run_inc2_build.sh run_inc2_zoo.sh run_inc2_job.sh run_inc2_splits.sh run_inc2_stream.sh run_inc_collect.sh
  tests/fixtures/collect
  # shared libraries the stream job scripts hash (SHARED_RE: never changed by this deploy)
  weed_optimizer_framework/tools/near_dup.py weed_optimizer_framework/tools/semisup_labeler.py
  weed_optimizer_framework/tools/registry_lock.py weed_optimizer_framework/tools/mega_trainer.py
  weed_optimizer_framework/tools/license_audit.py
  weed_optimizer_framework/tools/brain/su_ledger.py weed_optimizer_framework/tools/brain/policy.py
)
# Files only the lab runs, never copied to the cluster: db.py's DEFAULT_DOMAIN_CONFIG is the domain budget that
# inc_autopilot.budget.domain_budget() reads where the ticker runs (docs/CONTINUOUS_LOOP.md 6.6, "Amendment
# (2026-10-04)": no default daily cap). The cluster's copy belongs to the paused harvest and round scheduler, so
# this deploy leaves it alone, and the shared-module guard (SHARED_RE) is never needed for it.
LAB_ONLY_PATHS=(weed_optimizer_framework/tools/db.py)
while IFS= read -r t; do PKG_PATHS+=("$t"); done < <(ls tests/test_funnel_*.py tests/funnel_*.py tests/test_inc_*.py \
  tests/test_inc2_*.py tests/test_collect_*.py tests/test_stream_*.py tests/test_round_scheduler_stream_guard.py 2>/dev/null)
ROOT_PATHS=(docs/FUNNEL_AUDIT.md docs/FUNNEL_AUDIT_RUNNER.md docs/FUNNEL_REPRODUCE.md RESEARCH_LOG.md
            docs/poster/figures_data.json
            docs/CONTINUOUS_LOOP.md docs/INCREMENTAL_PROTOCOL.md docs/INCREMENTAL_PROTOCOL_RUNNER.md docs/INC_AUTOPILOT.md)
# The replay set envelope autonomy needs a pass for (executor.REPLAY_SCRIPTS: inc, funnel and stream,
# with both mutation harnesses) plus the stream's end-to-end pipeline: all must pass locally before any
# copy (docs/CONTINUOUS_LOOP.md 6.5: a failing S-case blocks the funnel's autonomy too). Since 2026-10-04 it
# also runs test_stream_ap_no_throttles.py: a tree that declares a default daily cap, monthly window or per-day
# lever count again (a merge that took an older db.py, say) is refused before anything is copied (6.6,
# "Amendment (2026-10-04)").
PREFLIGHT=(tests/test_inc_ap_replay.py tests/test_inc_ap_governance.py tests/test_funnel_ap_replay.py
           tests/test_funnel_ap_mutations.py tests/test_funnel_domain_free.py tests/test_stream_ap_replay.py
           tests/test_stream_ap_mutations.py tests/test_stream_pipeline.py tests/test_stream_ap_no_throttles.py)

for p in "${PKG_PATHS[@]}" "${LAB_ONLY_PATHS[@]}"; do [ -e "$p" ] || { echo "missing package path: $p" >&2; exit 1; }; done
for p in "${ROOT_PATHS[@]}"; do [ -e "$GIT_ROOT/$p" ] || { echo "missing git-root path: $p" >&2; exit 1; }; done
if ! git -C "$GIT_ROOT" diff --quiet HEAD -- "${PKG_PATHS[@]}" 2>/dev/null; then
  echo "WARNING: the deployed package paths differ from HEAD (uncommitted changes); the deploy record names HEAD" >&2
fi
HEAD_SHA="$(git -C "$GIT_ROOT" rev-parse HEAD)"

if [ "$DRY" = 1 ]; then
  printf 'package: %s\n' "${PKG_PATHS[@]}"
  printf 'lab-only: %s\n' "${LAB_ONLY_PATHS[@]}"
  printf 'git root: %s\n' "${ROOT_PATHS[@]}"
  printf 'pre-flight: %s\n' "${PREFLIGHT[@]}"
  exit 0
fi

LSSH=(ssh -o ConnectTimeout=30 -i "$LAB_SSH_KEY" "$LAB_SSH")
RSYNC_LAB=(rsync -aR --exclude __pycache__ -e "ssh -o ConnectTimeout=30 -i $LAB_SSH_KEY")
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="funnel_stage_$TS"

echo "== 0) pre-flight: the replay set (inc, funnel, stream) and the stream pipeline pass locally"
for t in "${PREFLIGHT[@]}"; do
  log="/tmp/deploy_preflight_$(basename "$t" .py)_$TS.log"
  if ! "$PY" "$t" > "$log" 2>&1; then
    echo "refused: $t fails locally (log $log); nothing was copied" >&2
    tail -5 "$log" >&2
    exit 6
  fi
  echo "  pass: $t"
done
# S23: the sha256 of every stream module this deploy ships, recorded beside the HEAD it names
"$PY" -c 'import json; from weed_optimizer_framework.tools.inc_autopilot import stream_remote as S; print(json.dumps(S.module_hashes(), sort_keys=True))' \
  > "/tmp/deploy_stream_modules_$TS.json"

echo "== 1) stage on the lab (~/$STAGE)"
"${LSSH[@]}" "mkdir -p ~/$STAGE/pkg ~/$STAGE/root ~/$STAGE/labonly"
"${RSYNC_LAB[@]}" "${PKG_PATHS[@]}" "$LAB_SSH:$STAGE/pkg/"
"${RSYNC_LAB[@]}" "${LAB_ONLY_PATHS[@]}" "$LAB_SSH:$STAGE/labonly/"
(cd "$GIT_ROOT" && "${RSYNC_LAB[@]}" "${ROOT_PATHS[@]}" "$LAB_SSH:$STAGE/root/")

echo "== 2) pre-check: what would change on the cluster (rsync --dry-run --checksum; the data-transfer node runs rsync only)"
"${LSSH[@]}" bash -s -- "$STAGE" "$CLUSTER_REPO" "$CLUSTER_DATA_SSH" > /tmp/deploy_funnel_pre_$TS.txt <<'LAB'
set -uo pipefail
STAGE="$HOME/$1"; CL="$2"; DATA="$3"
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
unset SSH_AUTH_SOCK
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o IdentityAgent=none -o NumberOfPasswordPrompts=1"
for dest in "weed_llm_benchmark/" ""; do
  setsid -w rsync -anc --itemize-changes -e "$RSH" "$STAGE/pkg/" "$DATA:$CL/$dest" < /dev/null \
    | awk -v d="$dest" '$1 ~ /^[<>c]f/ {print d $2}'
done
setsid -w rsync -anc --itemize-changes -e "$RSH" "$STAGE/root/" "$DATA:$CL/" < /dev/null | awk '$1 ~ /^[<>c]f/ {print "root:" $2}'
LAB
echo "  $(wc -l < /tmp/deploy_funnel_pre_$TS.txt | tr -d ' ') file(s) would change on the cluster"
if grep -Eq "$PINNED_RE" /tmp/deploy_funnel_pre_$TS.txt && [ "$ALLOW_PINNED" != 1 ]; then
  echo "refused: the deploy would change pinned INC module(s) on the cluster:" >&2
  grep -E "$PINNED_RE" /tmp/deploy_funnel_pre_$TS.txt >&2
  "${LSSH[@]}" "rm -rf ~/$STAGE"
  exit 3
fi
if grep -Eq "$SHARED_RE" /tmp/deploy_funnel_pre_$TS.txt && [ "$ALLOW_SHARED" != 1 ]; then
  echo "refused: the deploy would change shared library module(s) the live harvest, trainer or scheduler run:" >&2
  grep -E "$SHARED_RE" /tmp/deploy_funnel_pre_$TS.txt >&2
  "${LSSH[@]}" "rm -rf ~/$STAGE"
  exit 3
fi

echo "== 3) copy: lab tree (backup first), both cluster copies, the git-root files"
"${LSSH[@]}" bash -s -- "$STAGE" "$LAB_TREE_REL" "$CLUSTER_REPO" "$CLUSTER_DATA_SSH" "$TS" <<'LAB'
set -euo pipefail
STAGE="$HOME/$1"; TREE="$HOME/$2"; CL="$3"; DATA="$4"; TS="$5"
for part in pkg labonly; do
  cd "$STAGE/$part"
  find . -type f | while read -r f; do
    if [ -e "$TREE/$f" ]; then mkdir -p "$TREE/.deploy_backups/$TS/$(dirname "$f")"; cp -a "$TREE/$f" "$TREE/.deploy_backups/$TS/$f"; fi
  done
done
rsync -a "$STAGE/pkg/" "$TREE/"
rsync -a "$STAGE/labonly/" "$TREE/"
rsync -a "$STAGE/root/" "$HOME/"
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
unset SSH_AUTH_SOCK
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o IdentityAgent=none -o NumberOfPasswordPrompts=1"
setsid -w rsync -a -e "$RSH" "$STAGE/pkg/" "$DATA:$CL/weed_llm_benchmark/" < /dev/null
setsid -w rsync -a -e "$RSH" "$STAGE/pkg/" "$DATA:$CL/" < /dev/null
setsid -w rsync -a -e "$RSH" "$STAGE/root/" "$DATA:$CL/" < /dev/null
echo "  copied $(find "$STAGE" -type f | wc -l) file(s); lab backup .deploy_backups/$TS"
LAB

echo "== 3b) verify: nothing differs any more (lab by sha256, cluster by rsync --checksum)"
(cd "$PKG" && find "${PKG_PATHS[@]}" -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 shasum -a 256) \
  > /tmp/deploy_funnel_local_$TS.txt
"${LSSH[@]}" bash -s -- "$STAGE" "$LAB_TREE_REL" "$CLUSTER_REPO" "$CLUSTER_DATA_SSH" > /tmp/deploy_funnel_post_$TS.txt <<'LAB'
set -uo pipefail
STAGE="$HOME/$1"; TREE="$HOME/$2"; CL="$3"; DATA="$4"
for part in pkg labonly; do
  cd "$STAGE/$part" && find . -type f | sort | while read -r f; do
    a=$(sha256sum < "$f" | cut -c1-64); b=$(sha256sum < "$TREE/$f" 2>/dev/null | cut -c1-64)
    [ "$a" = "$b" ] || echo "lab:${f#./}"
  done
done
cd "$STAGE/root" && find . -type f | while read -r f; do
  a=$(sha256sum < "$f" | cut -c1-64); b=$(sha256sum < "$HOME/$f" 2>/dev/null | cut -c1-64)
  [ "$a" = "$b" ] || echo "lab-root:${f#./}"
done
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
unset SSH_AUTH_SOCK
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o IdentityAgent=none -o NumberOfPasswordPrompts=1"
for dest in "weed_llm_benchmark/" ""; do
  setsid -w rsync -anc --itemize-changes -e "$RSH" "$STAGE/pkg/" "$DATA:$CL/$dest" < /dev/null \
    | awk -v d="$dest" '$1 ~ /^[<>c]f/ {print "cluster:" d $2}'
done
setsid -w rsync -anc --itemize-changes -e "$RSH" "$STAGE/root/" "$DATA:$CL/" < /dev/null | awk '$1 ~ /^[<>c]f/ {print "cluster-root:" $2}'
LAB
N=$(wc -l < /tmp/deploy_funnel_post_$TS.txt | tr -d ' ')
echo "  verified $(wc -l < /tmp/deploy_funnel_local_$TS.txt | tr -d ' ') package file(s) on the lab and in both cluster copies; $N mismatch(es)"
if [ "$N" != 0 ]; then head -40 /tmp/deploy_funnel_post_$TS.txt | sed 's/^/  MISMATCH /' >&2; exit 4; fi

echo "== 4) the replay gate on the lab (envelope autonomy needs a pass for this code, for every campaign)"
REPLAY_RC=0
"${LSSH[@]}" "cd ~/$LAB_TREE_REL && .venv/bin/python -m weed_optimizer_framework.tools.inc_autopilot.executor record-replay" \
  > /tmp/deploy_replay_$TS.txt 2>&1 || REPLAY_RC=$?
grep -E '"status"|"code_hash"|": "fail"' /tmp/deploy_replay_$TS.txt | tail -5 || true
if [ "$REPLAY_RC" != 0 ] || ! grep -q '"status": "pass"' /tmp/deploy_replay_$TS.txt; then
  echo "WARNING: record-replay did not pass on the lab (/tmp/deploy_replay_$TS.txt): envelope autonomy is off for" \
       "every campaign until a passing record-replay" >&2
  if [ "$REPLAY_RC" = 0 ]; then REPLAY_RC=5; fi
fi

if [ "$RESTART" = 1 ]; then
  echo "== 5) restart the dashboard (the campaign ticker loads the new code)"
  "${LSSH[@]}" "systemctl --user restart weed-dashboard && sleep 15 && systemctl --user is-active weed-dashboard"
fi
"${LSSH[@]}" "rm -rf ~/$STAGE; mkdir -p ~/$LAB_TREE_REL/results/framework/inc/funnel && echo '{\"deployed_utc\": \"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\", \"git_head\": \"$HEAD_SHA\", \"script\": \"deploy/deploy_funnel.sh\"}' > ~/$LAB_TREE_REL/results/framework/inc/funnel/deployed.json"
# the stream's own record: the module hashes the snapshot's S23 check compares with the cluster's copy
"${LSSH[@]}" "cat > ~/$LAB_TREE_REL/results/framework/inc/stream_deployed.json" <<REC
{"deployed_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "git_head": "$HEAD_SHA", "script": "deploy/deploy_funnel.sh",
 "replay_rc": $REPLAY_RC, "stream_module_sha256": $(cat /tmp/deploy_stream_modules_$TS.json)}
REC
echo "done: HEAD $HEAD_SHA"
exit "$REPLAY_RC"
