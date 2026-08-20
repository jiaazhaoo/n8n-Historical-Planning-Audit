#!/usr/bin/env bash
# Cold-backup the n8n data volume, encrypt it, and push the archive to GitHub.
#
# n8n is stopped for the duration of the tar so SQLite's write-ahead log is
# checkpointed into the database and the archive is internally consistent.
# Typical downtime is a few seconds.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="${N8N_COMPOSE_DIR:-/env/code/n8n}"
VOLUME="${N8N_VOLUME:-n8n_n8n_data}"
RECIPIENT="${N8N_BACKUP_RECIPIENT:-n8n-backup@localhost}"
BACKUP_DIR="$REPO_DIR/backups"
KEEP="${N8N_BACKUP_KEEP:-14}"
STAMP="$(date +%Y-%m-%dT%H%M%S)"

mkdir -p "$BACKUP_DIR"

# storage/ holds regenerable execution binary artifacts and is ~100x the size of
# everything else; the event logs and crash journal are runtime noise. Excluding
# them keeps the archive small enough to version. Everything that defines the
# instance -- database.sqlite, config (the encryption key), nodes/ -- is kept.
TAR_EXCLUDES='--exclude=./storage --exclude=./n8nEventLog*.log --exclude=./crash.journal'

STAGE="$(mktemp -d)"
chmod 700 "$STAGE"

cd "$COMPOSE_DIR"
docker compose stop n8n >/dev/null
# Restart n8n even if the archive step fails, so a failed backup never leaves
# the service down.
trap 'docker compose -f "$COMPOSE_DIR/compose.yaml" start n8n >/dev/null' EXIT

docker run --rm -v "$VOLUME:/data:ro" -v "$STAGE:/out" \
  --user "$(id -u):$(id -g)" alpine \
  sh -c "tar czf /out/n8n.tgz -C /data $TAR_EXCLUDES ." >/dev/null

docker compose start n8n >/dev/null
trap - EXIT

ARCHIVE="$BACKUP_DIR/n8n-$STAMP.tgz.gpg"
gpg --batch --yes --trust-model always --encrypt --recipient "$RECIPIENT" \
  --output "$ARCHIVE" "$STAGE/n8n.tgz"
# The plaintext archive contains the n8n encryption key, so the staging copy is
# removed as soon as the encrypted one exists.
rm -rf "$STAGE"

# Guard against ever committing a readable archive.
if tar tzf "$ARCHIVE" >/dev/null 2>&1; then
  echo "ERROR: $ARCHIVE is not encrypted; refusing to commit" >&2
  rm -f "$ARCHIVE"
  exit 1
fi

ls -1t "$BACKUP_DIR"/n8n-*.tgz.gpg 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f

cd "$REPO_DIR"
git add -A backups
if git diff --cached --quiet; then
  echo "backup unchanged: $(basename "$ARCHIVE")"
  exit 0
fi
git commit -q -m "backup: n8n $STAMP"
git push -q origin HEAD
echo "backed up and pushed: $(basename "$ARCHIVE") ($(du -h "$ARCHIVE" | cut -f1))"
