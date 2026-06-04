#!/bin/bash
# run_mongo_node.sh — stand up a single-instance MongoDB WITHOUT root, for the
# MongoDB migration Phase 1 (see docs/mongodb_schema.md). Bridges-2 has no root
# and no system mongod, so we run a user-space mongod from a downloaded tarball,
# with the data files on /ocean (Lustre) so they survive node/job restarts.
#
# Scenario (a) from the schema doc: "labeler stays on cluster, uses MongoDB
# locally." mongod binds 127.0.0.1 only and is meant to be CO-LOCATED on the
# same SLURM node as the dashboard (uvicorn), which connects via localhost.
# Login-node daemons get killed by Bridges-2 policy, so do NOT start mongod on
# a login node — only download there (login nodes have outbound internet).
#
# Usage:
#   # one-time, on a LOGIN node (has internet) — just fetch the binary:
#   bash run_mongo_node.sh download
#
#   # inside the dashboard SBATCH (compute node), bring mongod up + write secret:
#   bash run_mongo_node.sh start
#
#   # both (download-if-needed then start) — safe to call repeatedly:
#   bash run_mongo_node.sh up
#
#   bash run_mongo_node.sh status   # is it reachable?
#   bash run_mongo_node.sh stop     # clean shutdown (keeps data)
#
# Env overrides:
#   REPO_ROOT          repo root (default: the cluster path)
#   MONGO_PORT         default 27017
#   MONGO_TARBALL_URL  override the download (e.g. different distro/version)
#
# Idempotent: skips download if the binary exists; skips start if the port is
# already serving; re-writes ~/.mongo_url every time so db.py picks it up.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark}"
MONGO_PORT="${MONGO_PORT:-27017}"
# rhel80 build matches Bridges-2 (RHEL 8 / glibc 2.28). NOTE: MongoDB renamed
# the target from "rhel80" to "rhel8" at 7.0.13+, so pin 7.0.12 (rhel80, verified
# downloadable). Override MONGO_TARGET=rhel8 + MONGO_VER for a newer build, or
# MONGO_TARBALL_URL for a fully custom one. Binary either runs or the version
# check below reports the glibc error clearly.
MONGO_VER="${MONGO_VER:-7.0.12}"
MONGO_TARGET="${MONGO_TARGET:-rhel80}"
MONGO_DISTRO="mongodb-linux-x86_64-${MONGO_TARGET}-${MONGO_VER}"
MONGO_TARBALL_URL="${MONGO_TARBALL_URL:-https://fastdl.mongodb.org/linux/${MONGO_DISTRO}.tgz}"

MONGO_HOME="$REPO_ROOT/.mongodb"
MONGO_BIN="$MONGO_HOME/${MONGO_DISTRO}/bin/mongod"
DBPATH="$REPO_ROOT/mongo_data"
LOGPATH="$REPO_ROOT/results/framework/mongod.log"
PIDFILE="$REPO_ROOT/results/framework/mongod.pid"
SECRET_FILE="$HOME/.mongo_url"
# v3.0.80: bind on the node's hostname (not just 127.0.0.1) so OTHER jobs
# (brain_harvest, trainer) on different compute nodes can reach this Mongo —
# required for the multi-agent pipeline to share one DB. --bind_ip_all also
# listens on 127.0.0.1 so co-located processes (dashboard) still reach it.
MONGO_HOST="${MONGO_HOST:-$(hostname -f 2>/dev/null || hostname)}"

# v3.0.81: authentication. Since the DB binds on a shared compute node, enable
# auth (SCRAM). Password lives in a 600 secret file (same pattern as the
# Roboflow key / GH PAT), NEVER in git. The connection string written to
# ~/.mongo_url embeds user:pass; db.py reads it as-is and redacts on display.
# Disable with MONGO_AUTH=0 (e.g. for a throwaway local test).
MONGO_AUTH="${MONGO_AUTH:-1}"
MONGO_USER="${MONGO_USER:-agentai_admin}"
MONGO_PASS_FILE="${MONGO_PASS_FILE:-$HOME/.mongo_pass}"
MONGO_PASS=""        # filled by ensure_password
MONGO_URL=""         # filled by build_url
LOCAL_URL=""         # 127.0.0.1 cred URL for same-node admin ops

log() { echo "[mongo] $*"; }

ensure_password() {
    if [ ! -s "$MONGO_PASS_FILE" ]; then
        local p
        p=$(openssl rand -hex 24 2>/dev/null) || \
            p=$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)
        printf '%s\n' "$p" > "$MONGO_PASS_FILE"
        chmod 600 "$MONGO_PASS_FILE"
        log "generated new Mongo password → $MONGO_PASS_FILE (chmod 600)"
    fi
    MONGO_PASS=$(cat "$MONGO_PASS_FILE")
}

build_url() {
    if [ "$MONGO_AUTH" = "1" ]; then
        [ -z "$MONGO_PASS" ] && ensure_password
        MONGO_URL="mongodb://${MONGO_USER}:${MONGO_PASS}@${MONGO_HOST}:${MONGO_PORT}/agentai?authSource=admin"
        LOCAL_URL="mongodb://${MONGO_USER}:${MONGO_PASS}@127.0.0.1:${MONGO_PORT}/agentai?authSource=admin"
    else
        MONGO_URL="mongodb://${MONGO_HOST}:${MONGO_PORT}/agentai"
        LOCAL_URL="mongodb://127.0.0.1:${MONGO_PORT}/agentai"
    fi
}

do_download() {
    if [ -x "$MONGO_BIN" ]; then
        log "binary already present: $MONGO_BIN"
        return 0
    fi
    mkdir -p "$MONGO_HOME"
    local tgz="$MONGO_HOME/mongodb-${MONGO_VER}.tgz"
    log "downloading $MONGO_TARBALL_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$tgz" "$MONGO_TARBALL_URL" || { log "ERROR download failed (curl). On a compute node? run 'download' on a LOGIN node first."; return 1; }
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$tgz" "$MONGO_TARBALL_URL" || { log "ERROR download failed (wget)."; return 1; }
    else
        log "ERROR neither curl nor wget available"; return 1
    fi
    log "extracting into $MONGO_HOME"
    tar -xzf "$tgz" -C "$MONGO_HOME" || { log "ERROR extract failed"; return 1; }
    rm -f "$tgz"
    if [ -x "$MONGO_BIN" ]; then
        log "binary ready: $MONGO_BIN"
        "$MONGO_BIN" --version 2>&1 | head -1 | sed 's/^/[mongo] /'
        return 0
    fi
    log "ERROR binary not found after extract: $MONGO_BIN"
    return 1
}

_ping() {
    # 0 if a mongod answers `ping` at the given URL. `ping` is allowed even
    # when --auth is on and the URL has no creds, so this tests "is it up".
    python3 - "$1" <<'PY' 2>/dev/null
import sys
try:
    from pymongo import MongoClient
    MongoClient(sys.argv[1], serverSelectionTimeoutMS=2000).admin.command("ping")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

port_listening() { _ping "mongodb://127.0.0.1:${MONGO_PORT}/"; }   # is mongod up?

auth_works() {
    # 0 if the credentialed URL can do an AUTHENTICATED op (listCollections).
    [ "$MONGO_AUTH" = "1" ] || { port_listening; return; }
    python3 - "$LOCAL_URL" <<'PY' 2>/dev/null
import sys
try:
    from pymongo import MongoClient
    MongoClient(sys.argv[1], serverSelectionTimeoutMS=2500)["agentai"].list_collection_names()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

create_user_if_needed() {
    # Idempotent: if auth already works, the user exists. Otherwise use the
    # MongoDB localhost exception (first-user-on-localhost is allowed even with
    # --auth) to create the root user. Runs on the dashboard node (127.0.0.1).
    if auth_works; then
        log "auth user already present + working"
        return 0
    fi
    log "creating Mongo user '$MONGO_USER' via localhost exception"
    python3 - "$MONGO_PORT" "$MONGO_USER" "$MONGO_PASS" <<'PY'
import sys
from pymongo import MongoClient
port, user, pw = sys.argv[1], sys.argv[2], sys.argv[3]
c = MongoClient(f"mongodb://127.0.0.1:{port}/", serverSelectionTimeoutMS=3000)
try:
    c.admin.command("createUser", user, pwd=pw,
                    roles=[{"role": "root", "db": "admin"}])
    print("[mongo] user created")
except Exception as e:
    # already exists (race / re-run) is fine; anything else is a real error
    if "already exists" in str(e):
        print("[mongo] user already exists")
    else:
        print(f"[mongo] ERROR createUser: {e}"); sys.exit(1)
PY
}

ensure_pymongo() {
    python3 -c "import pymongo" 2>/dev/null && return 0
    log "pymongo missing in this env — installing"
    python3 -m pip install --quiet pymongo || { log "WARN pip install pymongo failed; db.py will stay in JSON-fallback"; return 1; }
}

write_secret() {
    printf '%s\n' "$MONGO_URL" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
    log "wrote connection string to $SECRET_FILE"
}

do_start() {
    ensure_pymongo
    [ "$MONGO_AUTH" = "1" ] && ensure_password
    build_url
    if auth_works; then
        log "mongod already serving + auth OK on ${MONGO_HOST}:${MONGO_PORT} — reusing"
        write_secret
        return 0
    fi
    if port_listening; then
        # mongod is up but auth doesn't work yet — likely first-time enable of
        # auth on an existing no-auth dbpath. Create the user via localhost
        # exception, then verify.
        log "mongod up but auth not yet set — bootstrapping user"
        create_user_if_needed || return 1
        if auth_works; then write_secret; return 0; fi
        log "ERROR auth still failing after user creation"; return 1
    fi
    if [ ! -x "$MONGO_BIN" ]; then
        log "binary not present; attempting download"
        do_download || { log "ERROR cannot start without binary"; return 1; }
    fi
    mkdir -p "$DBPATH" "$(dirname "$LOGPATH")"
    local auth_flag=""
    [ "$MONGO_AUTH" = "1" ] && auth_flag="--auth"
    log "starting mongod  dbpath=$DBPATH  port=$MONGO_PORT  (bind_ip_all → $MONGO_HOST + 127.0.0.1, auth=$MONGO_AUTH)"
    "$MONGO_BIN" \
        --dbpath "$DBPATH" \
        --bind_ip_all \
        --port "$MONGO_PORT" \
        --logpath "$LOGPATH" \
        --pidfilepath "$PIDFILE" \
        --wiredTigerCacheSizeGB 2 \
        $auth_flag \
        --fork
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "ERROR mongod failed to start (rc=$rc). Last log lines:"
        tail -n 20 "$LOGPATH" 2>/dev/null | sed 's/^/[mongo]   /'
        return 1
    fi
    # Wait for mongod to listen.
    local up=0
    for i in $(seq 1 20); do
        if port_listening; then up=1; log "mongod is up (after ${i}s)"; break; fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        log "ERROR mongod started but never listened. Last log lines:"
        tail -n 20 "$LOGPATH" 2>/dev/null | sed 's/^/[mongo]   /'
        return 1
    fi
    # First-run: create the auth user (localhost exception).
    if [ "$MONGO_AUTH" = "1" ]; then
        create_user_if_needed || return 1
        if ! auth_works; then
            log "ERROR auth not working after start+createUser"; return 1
        fi
        log "mongod up and AUTHENTICATED (user $MONGO_USER)"
    fi
    write_secret
    return 0
}

do_status() {
    if ! port_listening; then
        log "NOT reachable on 127.0.0.1:${MONGO_PORT}"
        return 1
    fi
    if [ "$MONGO_AUTH" = "1" ] && ! auth_works; then
        log "mongod up but AUTH FAILED (wrong/no creds in $MONGO_PASS_FILE?)"
        return 1
    fi
    log "REACHABLE + auth=$MONGO_AUTH on ${MONGO_HOST}:${MONGO_PORT}"
    python3 - "$LOCAL_URL" <<'PY' 2>/dev/null
import sys
from pymongo import MongoClient
db = MongoClient(sys.argv[1], serverSelectionTimeoutMS=2500)["agentai"]
print("[mongo]   db:", db.name, "collections:", db.list_collection_names())
for c in ("slugs","classes"):
    try: print(f"[mongo]   {c}: {db[c].estimated_document_count()} docs")
    except Exception: pass
PY
    return 0
}

do_stop() {
    if [ ! -x "$MONGO_BIN" ]; then log "no binary; nothing to stop"; return 0; fi
    log "shutting down mongod cleanly (data kept at $DBPATH)"
    # --shutdown is a local admin op on the dbpath; works without auth creds.
    "$MONGO_BIN" --dbpath "$DBPATH" --shutdown 2>/dev/null \
        && log "stopped" || log "shutdown returned nonzero (may already be down)"
}

# Resolve creds + URLs once before dispatching (status/stop need them too).
[ "$MONGO_AUTH" = "1" ] && [ -s "$MONGO_PASS_FILE" ] && MONGO_PASS=$(cat "$MONGO_PASS_FILE")
build_url

case "${1:-up}" in
    download) do_download ;;
    start)    do_start ;;
    up)       do_download && do_start ;;
    status)   do_status ;;
    stop)     do_stop ;;
    *) echo "usage: $0 {download|start|up|status|stop}"; exit 2 ;;
esac
