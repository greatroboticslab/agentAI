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
#   deploy/deploy_funnel.sh --dry-run  list what would be copied and run the V3F deploy checks; copy nothing
#   deploy/deploy_funnel.sh --allow-older-head   (only to put back an older tree on purpose: see DESCENDS below)
#   deploy/deploy_funnel.sh --allow-v3fam-code-change=docs/prereg/<deviation>.md
#                                      (only with a recorded deviation, the file of one of the domain's reopen
#                                      blocks, that moves V3F's reproduction records aside and retakes them: see
#                                      V3F_CODE below; a deploy that restores the bytes they pin needs none)
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
ALLOW_V3F=""
ALLOW_OLDER=0
for a in "$@"; do
  case "$a" in
    --no-restart) RESTART=0 ;;
    --dry-run) DRY=1 ;;
    --allow-pinned-change) ALLOW_PINNED=1 ;;
    --allow-shared-change) ALLOW_SHARED=1 ;;
    --allow-older-head) ALLOW_OLDER=1 ;;
    --allow-v3fam-code-change=*) ALLOW_V3F="${a#*=}" ;;
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
  # V3F (2026-10-11): D6's RF-DETR job script (lever L23K)
  run_inc2_detr.sh
  tests/fixtures/collect
  # deviation 5 (2026-10-24): the cluster's own V3F records the shipped test_stream_ap_v3fam_dev5.py reads
  tests/fixtures/v3fam_dev5
  # 2026-10-26: the lab's records the shipped test_stream_ap_stalls.py replays (D31's L4, the DATA lane's hold)
  tests/fixtures/platform_stalls
  # 2026-10-26: the check of the job scripts' mlflow-off block (MLFLOW_OFF_REAL=1 on the cluster's bench env)
  tests/test_job_mlflow_off.py
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
            docs/CONTINUOUS_LOOP.md docs/INCREMENTAL_PROTOCOL.md docs/INCREMENTAL_PROTOCOL_RUNNER.md docs/INC_AUTOPILOT.md
            docs/prereg/V3F_2026-10-11.md docs/prereg/V3F_rev29-31_2026-10-11.md docs/prereg/V3F_dev4_2026-10-20.md
            docs/prereg/V3F_dev5_2026-10-24.md docs/prereg/V3F_dev6_2026-10-25.md)
# V3F (2026-10-11): the ticker proposes the family's levers only once the deployed tree holds the amendment's byte
# copy (docs/prereg/V3F_2026-10-11.md) and its Revisions 29-31's (docs/prereg/V3F_rev29-31_2026-10-11.md), each by
# its sha256; it reopens the items a recorded deviation names only once the deployed tree holds that deviation's
# file by its sha256 (the domain's v3fam.reopen: docs/prereg/V3F_dev4_2026-10-20.md, deviation 4;
# docs/prereg/V3F_dev5_2026-10-24.md, deviation 5; docs/prereg/V3F_dev6_2026-10-25.md, deviation 6); the deploy record
# names whether HEAD descends from the amendment's commit
V3F_COMMIT=55cc2d97cd2147d01644c1e193098b73aeba0a06
# V3F_CODE (deviation 5, 2026-10-24): every reproduction record of the family (the cluster's v3fam/v1/equivalence)
# pins the sha256 of the code that took it (inc2.v3fam.code_shas: inc2/v3fam.py and the six E3-frozen files), and
# inc2.v3fam.require_equivalence refuses every family score resting on a record taken by other code; on 2026-10-21 a
# CLI-only change of inc2/v3fam.py refused twelve score jobs of six arms. Before anything is copied (and in --dry-run)
# stream_remote.v3fam_deploy_check refuses a tree whose code is not the domain's v3fam.equivalence_code, or that would
# ship a file the domain's v3fam block pins by sha256 (the amendment, its Revisions, each reopen block's deviation)
# with another sha256; step 2 refuses, as for PINNED_RE, a change of those files on the cluster that would leave a
# reproduction record taken by other code than the tree ships (stream_remote.v3fam_records_guard reads each record's
# code map: a change that restores the bytes the records pin passes, as deviation 5's deploy does).
# --allow-v3fam-code-change=<file> passes both only when <file> is the deviation file of one of the domain's reopen
# blocks, among ROOT_PATHS and hashing as that block pins it (stream_remote.v3fam_allow_ok), and the deploy record
# names it with its sha256.
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
# (the package paths are relative to $PKG, the working directory: git reads pathspecs relative to it)
if ! git diff --quiet HEAD -- "${PKG_PATHS[@]}" 2>/dev/null \
    || ! (cd "$GIT_ROOT" && git diff --quiet HEAD -- "${ROOT_PATHS[@]}" 2>/dev/null); then
  echo "WARNING: the deployed paths differ from HEAD (uncommitted changes); the deploy record names HEAD" >&2
fi
if [ -n "$(git status --porcelain --untracked-files=all -- "${PKG_PATHS[@]}" 2>/dev/null | grep '^??' || true)" ]; then
  echo "WARNING: the deployed package paths hold untracked files (shipped from the working tree, absent from HEAD):" >&2
  git status --porcelain --untracked-files=all -- "${PKG_PATHS[@]}" 2>/dev/null | grep '^??' | head -20 >&2
fi
HEAD_SHA="$(git -C "$GIT_ROOT" rev-parse HEAD)"
if git -C "$GIT_ROOT" merge-base --is-ancestor "$V3F_COMMIT" HEAD 2>/dev/null; then V3F_ANCESTOR=true; else V3F_ANCESTOR=false; fi
if [ -n "$ALLOW_V3F" ]; then
  case " ${ROOT_PATHS[*]} " in *" $ALLOW_V3F "*) ;; *)
    echo "refused: --allow-v3fam-code-change names $ALLOW_V3F, not a git-root path this deploy ships" >&2; exit 3 ;; esac
  V3F_CHANGE="{\"file\": \"$ALLOW_V3F\", \"sha256\": \"$(shasum -a 256 "$GIT_ROOT/$ALLOW_V3F" | cut -c1-64)\"}"
else
  V3F_CHANGE=null
fi
# V3F_CODE: the deploy-time checks on this tree (no network)
if ! V3F_CHECK="$("$PY" - "$GIT_ROOT" "$ALLOW_V3F" "${ROOT_PATHS[@]}" <<'PYCHK' 2>&1
import sys
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS, stream_remote as SR
ok, lines = SR.v3fam_deploy_check(LS.load_domain("weed"), sys.argv[1], sys.argv[3:], allow=sys.argv[2] or None)
print("\n".join("v3fam check: %s" % x for x in lines))
sys.exit(0 if ok else 1)
PYCHK
)"; then
  echo "$V3F_CHECK" >&2
  echo "refused: the V3F deploy checks failed; nothing was copied" >&2
  exit 3
fi

if [ "$DRY" = 1 ]; then
  printf 'package: %s\n' "${PKG_PATHS[@]}"
  printf 'lab-only: %s\n' "${LAB_ONLY_PATHS[@]}"
  printf 'git root: %s\n' "${ROOT_PATHS[@]}"
  printf 'pre-flight: %s\n' "${PREFLIGHT[@]}"
  echo "$V3F_CHECK"
  exit 0
fi
echo "$V3F_CHECK"

LSSH=(ssh -o ConnectTimeout=30 -i "$LAB_SSH_KEY" "$LAB_SSH")
RSYNC_LAB=(rsync -aR --exclude __pycache__ -e "ssh -o ConnectTimeout=30 -i $LAB_SSH_KEY")
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="funnel_stage_$TS"

# DESCENDS (V3F deviation 6, 2026-10-25): the head this deploy ships must descend from the head deployed now (the
# lab's stream_deployed.json): the copies below are rsync -a without --delete, so a deploy from a branch cut before
# the deployed commit would put that commit's older files back (a recorded deviation's code, its reopen block, its
# ticker rules) with every other check passing. Refused before anything is copied, unless --allow-older-head.
echo "== 0a) the deployed head is an ancestor of HEAD"
DEPLOYED_REC="$("${LSSH[@]}" "cat ~/$LAB_TREE_REL/results/framework/inc/stream_deployed.json 2>/dev/null" || true)"
if ! DESCENDS="$(printf '%s' "$DEPLOYED_REC" | "$PY" -c 'import sys
from weed_optimizer_framework.tools.inc_autopilot import stream_remote as S
ok, why = S.deploy_descends(sys.argv[1], sys.stdin.read())
print(why)
sys.exit(0 if ok else 1)' "$GIT_ROOT" 2>&1)"; then
  if [ "$ALLOW_OLDER" != 1 ]; then
    echo "refused: $DESCENDS; nothing was copied" >&2
    exit 3
  fi
  echo "  --allow-older-head: $DESCENDS" >&2
else
  echo "  $DESCENDS"
fi

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
# V3F_CODE: a change of a file the reproduction records pin, while the cluster holds records: the records (top-level
# *.json of v3fam/v1/equivalence) are copied through the data-transfer node into a temporary folder on the lab, and
# each one's name, sha256 and code map read there; the change passes when every record but those a recorded
# deviation supersedes pins the code this tree ships (a restore of the bytes they pin, as deviation 5's deploy is), or
# with --allow-v3fam-code-change naming a reopen block's deviation; a failed fetch reads as records present
if grep -Eq "$("$PY" -c 'from weed_optimizer_framework.tools.inc_autopilot import stream_remote as S; print(S.V3FAM_CODE_RE.pattern)')" /tmp/deploy_funnel_pre_$TS.txt; then
  "${LSSH[@]}" bash -s -- "$CLUSTER_REPO" "$CLUSTER_DATA_SSH" > /tmp/deploy_funnel_eq_$TS.json <<'LAB' || echo '{"error": "the lab step failed"}' > /tmp/deploy_funnel_eq_$TS.json
set -uo pipefail
CL="$1"; DATA="$2"
export SSH_ASKPASS="$HOME/.cluster_askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
unset SSH_AUTH_SOCK
RSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o IdentityAgent=none -o NumberOfPasswordPrompts=1"
T="$(mktemp -d)"
if setsid -w rsync -a --include='*.json' --exclude='*' -e "$RSH" "$DATA:$CL/results/framework/inc/v3fam/v1/equivalence/" "$T/" < /dev/null; then
  python3 - "$T" <<'PY'
import hashlib, json, os, sys
d, out = sys.argv[1], []
for n in sorted(os.listdir(d)):
    p = os.path.join(d, n)
    if not n.endswith(".json") or not os.path.isfile(p) or os.path.islink(p):
        continue
    b = open(p, "rb").read()
    try:
        code = json.loads(b.decode("utf-8")).get("code")
    except (ValueError, AttributeError, UnicodeDecodeError):
        code = None
    out.append({"name": n, "sha256": hashlib.sha256(b).hexdigest(), "code": code if isinstance(code, dict) else None})
print(json.dumps({"records": out}))
PY
else
  echo '{"error": "the fetch failed"}'
fi
rm -rf "$T"
LAB
  if ! V3F_GUARD="$("$PY" - /tmp/deploy_funnel_pre_$TS.txt /tmp/deploy_funnel_eq_$TS.json "$GIT_ROOT" "$ALLOW_V3F" "${ROOT_PATHS[@]}" <<'PYG'
import sys
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS, stream_remote as S
ok, msg = S.v3fam_records_guard(open(sys.argv[1]).read().split(), open(sys.argv[2]).read(), sys.argv[4] or None,
                                sys.argv[5:], dom=LS.load_domain("weed"), git_root=sys.argv[3],
                                package="weed_optimizer_framework")
print(msg)
sys.exit(0 if ok else 1)
PYG
)"; then
    echo "$V3F_GUARD" >&2
    "${LSSH[@]}" "rm -rf ~/$STAGE"
    exit 3
  fi
  echo "  $V3F_GUARD" >&2
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
 "replay_rc": $REPLAY_RC, "v3fam_amendment_ancestor": $V3F_ANCESTOR, "v3fam_code_change": $V3F_CHANGE,
 "stream_module_sha256": $(cat /tmp/deploy_stream_modules_$TS.json)}
REC
echo "done: HEAD $HEAD_SHA"
exit "$REPLAY_RC"
