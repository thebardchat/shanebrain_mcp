#!/usr/bin/env bash
# =============================================================================
# Weaviate Automated Backup Script
# Runs daily via cron, keeps last 7 backups, prunes older ones.
# Backups stored in: weaviate-config/backups/ (mounted into container)
# =============================================================================

set -euo pipefail

WEAVIATE_URL="http://localhost:8080"
BACKUP_ID="backup-$(date +%Y-%m-%d)"
BACKUP_DIR="/mnt/shanebrain-raid/shanebrain-core/weaviate-config/backups"
KEEP_DAYS=7
LOG_TAG="weaviate-backup"

log() { logger -t "$LOG_TAG" "$1"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# Check Weaviate is up
if ! curl -sf "$WEAVIATE_URL/v1/.well-known/ready" > /dev/null 2>&1; then
    log "ERROR: Weaviate not ready, skipping backup"
    exit 1
fi

# Trigger backup
log "Starting backup: $BACKUP_ID"
RESPONSE=$(curl -s -X POST "$WEAVIATE_URL/v1/backups/filesystem" \
    -H "Content-Type: application/json" \
    -d "{\"id\": \"$BACKUP_ID\"}" 2>&1)

# Check if backup already exists (ran twice today)
if echo "$RESPONSE" | grep -q "already exists"; then
    log "Backup $BACKUP_ID already exists, skipping"
    exit 0
fi

# Check for other errors (no "status" field means failure)
if ! echo "$RESPONSE" | grep -q '"status"'; then
    log "ERROR: Failed to start backup: $RESPONSE"
    exit 1
fi

# Wait for completion (max 2 minutes)
for i in $(seq 1 24); do
    sleep 5
    STATUS=$(curl -sf "$WEAVIATE_URL/v1/backups/filesystem/$BACKUP_ID" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    if [ "$STATUS" = "SUCCESS" ]; then
        log "Backup $BACKUP_ID completed successfully"
        break
    elif [ "$STATUS" = "FAILED" ]; then
        log "ERROR: Backup $BACKUP_ID failed"
        exit 1
    fi
done

if [ "$STATUS" != "SUCCESS" ]; then
    log "ERROR: Backup $BACKUP_ID timed out (status: $STATUS)"
    exit 1
fi

# Prune old backups (keep last N days)
if [ -d "$BACKUP_DIR" ]; then
    PRUNED=0
    for dir in "$BACKUP_DIR"/backup-*; do
        [ -d "$dir" ] || continue
        DIR_DATE=$(basename "$dir" | sed 's/backup-//')
        # Skip if not a valid date format
        echo "$DIR_DATE" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' || continue
        AGE_DAYS=$(( ($(date +%s) - $(date -d "$DIR_DATE" +%s 2>/dev/null || echo 0)) / 86400 ))
        if [ "$AGE_DAYS" -gt "$KEEP_DAYS" ]; then
            rm -rf "$dir"
            PRUNED=$((PRUNED + 1))
            log "Pruned old backup: $(basename "$dir") ($AGE_DAYS days old)"
        fi
    done
    [ "$PRUNED" -gt 0 ] && log "Pruned $PRUNED old backups"
fi

log "Backup complete. Current backups:"
ls -d "$BACKUP_DIR"/backup-* 2>/dev/null | while read -r d; do
    SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
    log "  $(basename "$d"): $SIZE"
done
