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
