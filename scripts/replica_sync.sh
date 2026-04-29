#!/usr/bin/env bash
# =============================================================================
# Weaviate Replica Sync — pulls a backup from the primary into the local Pi.
#
# Designed for the Phase 3 Pi 5 → neworleans cutover. The local Pi runs as a
# read-replica: every 6 hours it triggers a fresh backup on the primary,
# rsyncs the backup files over, and restores them locally.
#
# Cron usage:
#   0 */6 * * * /mnt/shanebrain-raid/shanebrain-core/mcp-server/scripts/replica_sync.sh
#
# Required env (sourced from .claude/infra.env):
#   WEAVIATE_REPLICA_PRIMARY_HOST   # primary host (e.g. neworleans tailscale ip)
#   WEAVIATE_REPLICA_PRIMARY_PORT   # primary http port (default 8080)
#   WEAVIATE_HOST                   # local replica host (usually localhost)
#   WEAVIATE_PORT                   # local replica http port (default 8080)
#
# Required env (this file's defaults — override only if your layout differs):
#   PRIMARY_SSH_USER          ssh user on primary (for rsync), default: shane
#   PRIMARY_BACKUP_DIR        absolute backup dir on primary
#   LOCAL_BACKUP_DIR          absolute backup dir on local (must be the path
#                             Weaviate's container has mounted as the backups
#                             volume — same as weaviate_backup.sh)
#
# Prereqs:
#   - SSH key auth from the Pi to the primary (no password prompts)
#   - rsync installed on both ends
#   - Both Weaviate instances configured with the `backup-filesystem` module
#     pointing at their respective backup dirs
# =============================================================================

set -euo pipefail

INFRA_ENV="${INFRA_ENV:-/mnt/shanebrain-raid/shanebrain-core/mcp-server/.claude/infra.env}"
if [ -r "$INFRA_ENV" ]; then
    set -a; source "$INFRA_ENV"; set +a
fi

PRIMARY_HOST="${WEAVIATE_REPLICA_PRIMARY_HOST:-}"
PRIMARY_PORT="${WEAVIATE_REPLICA_PRIMARY_PORT:-8080}"
LOCAL_HOST="${WEAVIATE_HOST:-localhost}"
LOCAL_PORT="${WEAVIATE_PORT:-8080}"

PRIMARY_SSH_USER="${PRIMARY_SSH_USER:-shane}"
PRIMARY_BACKUP_DIR="${PRIMARY_BACKUP_DIR:-/mnt/shanebrain-raid/shanebrain-core/weaviate-config/backups}"
LOCAL_BACKUP_DIR="${LOCAL_BACKUP_DIR:-/mnt/shanebrain-raid/shanebrain-core/weaviate-config/backups}"

BACKUP_ID="replica-$(date -u +%Y-%m-%dT%H%M%SZ)"
LOG_TAG="weaviate-replica-sync"
WAIT_TIMEOUT_SECS=180

log()  { logger -t "$LOG_TAG" "$1"; echo "[$(date -u '+%FT%TZ')] $1"; }
fail() { log "ERROR: $1"; exit 1; }

[ -n "$PRIMARY_HOST" ] || fail "WEAVIATE_REPLICA_PRIMARY_HOST is empty. Source .claude/infra.env first."

primary_url="http://${PRIMARY_HOST}:${PRIMARY_PORT}"
local_url="http://${LOCAL_HOST}:${LOCAL_PORT}"

# 1. Sanity-check both ends are reachable
curl -sf "${primary_url}/v1/.well-known/ready" > /dev/null \
    || fail "primary $primary_url not ready"
curl -sf "${local_url}/v1/.well-known/ready" > /dev/null \
    || fail "local $local_url not ready"

# 2. Trigger backup on primary
log "Triggering backup $BACKUP_ID on primary $PRIMARY_HOST"
resp=$(curl -s -X POST "${primary_url}/v1/backups/filesystem" \
    -H "Content-Type: application/json" \
    -d "{\"id\": \"$BACKUP_ID\"}")

echo "$resp" | grep -q '"status"' \
    || fail "primary refused backup request: $resp"

# 3. Poll primary until backup is SUCCESS or FAILED
status="UNKNOWN"
for i in $(seq 1 $((WAIT_TIMEOUT_SECS / 5))); do
    sleep 5
    status=$(curl -sf "${primary_url}/v1/backups/filesystem/$BACKUP_ID" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" \
        2>/dev/null || echo "UNKNOWN")
    case "$status" in
        SUCCESS) break ;;
        FAILED)  fail "primary backup $BACKUP_ID FAILED" ;;
    esac
done
[ "$status" = "SUCCESS" ] || fail "primary backup $BACKUP_ID timed out (last status: $status)"
log "Primary backup $BACKUP_ID complete"

# 4. rsync the backup directory to the local Pi
log "rsync ${PRIMARY_SSH_USER}@${PRIMARY_HOST}:${PRIMARY_BACKUP_DIR}/${BACKUP_ID}/ -> ${LOCAL_BACKUP_DIR}/${BACKUP_ID}/"
mkdir -p "${LOCAL_BACKUP_DIR}/${BACKUP_ID}"
rsync -az --delete \
    "${PRIMARY_SSH_USER}@${PRIMARY_HOST}:${PRIMARY_BACKUP_DIR}/${BACKUP_ID}/" \
    "${LOCAL_BACKUP_DIR}/${BACKUP_ID}/" \
    || fail "rsync failed"

# 5. Trigger restore on local
log "Triggering restore of $BACKUP_ID on local $LOCAL_HOST"
resp=$(curl -s -X POST "${local_url}/v1/backups/filesystem/$BACKUP_ID/restore" \
    -H "Content-Type: application/json" -d "{}")

echo "$resp" | grep -q '"status"' \
    || fail "local refused restore: $resp"

# 6. Poll local restore status
status="UNKNOWN"
for i in $(seq 1 $((WAIT_TIMEOUT_SECS / 5))); do
    sleep 5
    status=$(curl -sf "${local_url}/v1/backups/filesystem/$BACKUP_ID/restore" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" \
        2>/dev/null || echo "UNKNOWN")
    case "$status" in
        SUCCESS) break ;;
        FAILED)  fail "local restore $BACKUP_ID FAILED" ;;
    esac
done
[ "$status" = "SUCCESS" ] || fail "local restore $BACKUP_ID timed out (last status: $status)"

log "Replica sync of $BACKUP_ID complete."
