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
MONGO_URL="mongodb://127.0.0.1:${MONGO_PORT}/agentai"

log() { echo "[mongo] $*"; }

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

port_alive() {
    # 0 if something is serving Mongo on the port (python+pymongo ping).
    python3 - "$MONGO_URL" <<'PY' 2>/dev/null
import sys
try:
    from pymongo import MongoClient
    MongoClient(sys.argv[1], serverSelectionTimeoutMS=1500).admin.command("ping")
    sys.exit(0)
except Exception:
    sys.exit(1)
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
    if port_alive; then
        log "mongod already serving on 127.0.0.1:${MONGO_PORT} — reusing"
        write_secret
        return 0
    fi
    if [ ! -x "$MONGO_BIN" ]; then
        log "binary not present; attempting download"
        do_download || { log "ERROR cannot start without binary"; return 1; }
    fi
    mkdir -p "$DBPATH" "$(dirname "$LOGPATH")"
    log "starting mongod  dbpath=$DBPATH  port=$MONGO_PORT  (bind 127.0.0.1 only)"
    "$MONGO_BIN" \
        --dbpath "$DBPATH" \
        --bind_ip 127.0.0.1 \
        --port "$MONGO_PORT" \
        --logpath "$LOGPATH" \
        --pidfilepath "$PIDFILE" \
        --wiredTigerCacheSizeGB 2 \
        --fork
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "ERROR mongod failed to start (rc=$rc). Last log lines:"
        tail -n 20 "$LOGPATH" 2>/dev/null | sed 's/^/[mongo]   /'
        return 1
    fi
    # Wait for it to accept connections.
    for i in $(seq 1 20); do
        if port_alive; then
            log "mongod is up and reachable (after ${i}s)"
            write_secret
            return 0
        fi
        sleep 1
    done
    log "ERROR mongod started but never became reachable. Last log lines:"
    tail -n 20 "$LOGPATH" 2>/dev/null | sed 's/^/[mongo]   /'
    return 1
}

do_status() {
    if port_alive; then
        log "REACHABLE on 127.0.0.1:${MONGO_PORT}  url=$MONGO_URL"
        python3 - "$MONGO_URL" <<'PY' 2>/dev/null
import sys
from pymongo import MongoClient
db = MongoClient(sys.argv[1], serverSelectionTimeoutMS=1500).get_default_database()
print("[mongo]   db:", db.name, "collections:", db.list_collection_names())
for c in ("slugs","classes"):
    try: print(f"[mongo]   {c}: {db[c].estimated_document_count()} docs")
    except Exception: pass
PY
        return 0
    fi
    log "NOT reachable on 127.0.0.1:${MONGO_PORT}"
    return 1
}

do_stop() {
    if [ ! -x "$MONGO_BIN" ]; then log "no binary; nothing to stop"; return 0; fi
    log "shutting down mongod cleanly (data kept at $DBPATH)"
    "$MONGO_BIN" --dbpath "$DBPATH" --shutdown 2>/dev/null \
        && log "stopped" || log "shutdown returned nonzero (may already be down)"
}

case "${1:-up}" in
    download) do_download ;;
    start)    do_start ;;
    up)       do_download && do_start ;;
    status)   do_status ;;
    stop)     do_stop ;;
    *) echo "usage: $0 {download|start|up|status|stop}"; exit 2 ;;
esac
