#!/usr/bin/env bash
# Restore an encrypted backup archive into the n8n data volume.
#
# Requires the backup GPG private key in the local keyring. Without it the
# archives in backups/ cannot be read at all -- keep an independent copy.
#
#   ./scripts/restore.sh backups/n8n-2026-08-20T033000.tgz.gpg
set -euo pipefail

ARCHIVE="${1:?usage: restore.sh <backups/n8n-....tgz.gpg>}"
COMPOSE_DIR="${N8N_COMPOSE_DIR:-/env/code/n8n}"
VOLUME="${N8N_VOLUME:-n8n_n8n_data}"

[ -f "$ARCHIVE" ] || { echo "no such archive: $ARCHIVE" >&2; exit 1; }
ARCHIVE="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"

STAGE="$(mktemp -d)"
chmod 700 "$STAGE"
trap 'rm -rf "$STAGE"' EXIT

gpg --batch --yes --decrypt --output "$STAGE/n8n.tgz" "$ARCHIVE"
tar tzf "$STAGE/n8n.tgz" >/dev/null || { echo "archive is corrupt" >&2; exit 1; }

echo "This replaces the contents of volume $VOLUME. Existing data is lost."
read -r -p "Type the volume name to confirm: " confirm
[ "$confirm" = "$VOLUME" ] || { echo "aborted"; exit 1; }

cd "$COMPOSE_DIR"
docker compose stop n8n >/dev/null

docker run --rm -v "$VOLUME:/data" -v "$STAGE:/in:ro" alpine \
  sh -c 'find /data -mindepth 1 -delete && tar xzf /in/n8n.tgz -C /data'

docker compose start n8n >/dev/null
echo "restored $(basename "$ARCHIVE"); n8n restarted"
echo
echo "storage/ (execution binary artifacts) is not part of the backup and will"
echo "be missing for past executions. Workflows and credentials are unaffected."
