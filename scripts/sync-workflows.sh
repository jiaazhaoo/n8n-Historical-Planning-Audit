#!/usr/bin/env bash
# Export every workflow from the running n8n container, strip instance-local
# and personal fields, and commit the result to workflows/.
#
# Never exports credentials: `n8n export:credentials` writes decrypted secrets
# and must not be used here.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${N8N_CONTAINER:-n8n}"
OUT_DIR="$REPO_DIR/workflows"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

docker exec "$CONTAINER" sh -c \
  'rm -rf /tmp/n8n-wf-sync && mkdir -p /tmp/n8n-wf-sync &&
   n8n export:workflow --all --pretty --separate --output=/tmp/n8n-wf-sync/' >/dev/null
docker cp "$CONTAINER:/tmp/n8n-wf-sync/." "$STAGE/"
docker exec "$CONTAINER" rm -rf /tmp/n8n-wf-sync

python3 "$REPO_DIR/scripts/sanitize_workflows.py" "$STAGE" "$OUT_DIR"

cd "$REPO_DIR"
if git diff --quiet -- workflows && git diff --cached --quiet -- workflows &&
   [ -z "$(git ls-files --others --exclude-standard -- workflows)" ]; then
  echo "no workflow changes"
  exit 0
fi

git add -A workflows
git commit -q -m "chore: sync n8n workflows"
git push -q origin HEAD
echo "pushed: $(git log --oneline -1)"
