# n8n Historical Planning Audit

Private, encrypted backup repository for the local n8n instance.

## Safety rules

- Never commit `.env`, `N8N_ENCRYPTION_KEY`, n8n's `config` file, or an unencrypted database/archive.
- Keep the encryption key only on the n8n host and in an independent password manager or recovery location.
- Store only encrypted backup archives in `backups/`.
- Verify restoration after changing the backup procedure.

## Versioned automation source

- `workflows/` contains secret-free n8n workflow definitions.
- `infrastructure/` contains the secret-free Docker Compose and user-service definitions.
- `mapping-service/` contains the n8n bridge, isolated capture-rule compiler integration, and focused tests.

Runtime mapping jobs and outputs remain under `/data`; OAuth credentials and API keys remain only on the local host.

## Workflow sync

`scripts/sync-workflows.sh` exports every workflow from the running n8n
container into `workflows/`, then commits and pushes any change:

```bash
./scripts/sync-workflows.sh
```

The export is normalised by `scripts/sanitize_workflows.py` before it is
written. It drops instance-local bookkeeping (timestamps, version counters) so
an unchanged workflow produces no diff, and drops the `shared` block, which
names the owning user and their email address. Node `credentials` entries are
kept — they hold only an id and display name, which a restore needs to relink
against locally stored credentials.

The sanitizer refuses to write a file matching a secret pattern (email address,
AWS key id, API key, GitHub token, private key), so a leak fails the sync rather
than reaching a commit. Credentials themselves are never exported: n8n's
`export:credentials` writes them decrypted and must not be used here.

The encryption key stays on the host in the n8n volume's `config` file. Nothing
in this repository can decrypt a credential without it, so keep an independent
copy of it — losing the host disk otherwise means losing every credential.
