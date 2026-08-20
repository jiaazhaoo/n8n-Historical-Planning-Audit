# n8n Historical Planning Audit

Private, encrypted backup repository for the local n8n instance.

## Safety rules

- Never commit `.env`, `N8N_ENCRYPTION_KEY`, n8n's `config` file, or an unencrypted database/archive.
- Keep the encryption key only on the n8n host and in an independent password manager or recovery location.
- Store only encrypted backup archives in `backups/`.
- Verify restoration after changing the backup procedure.
