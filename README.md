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

## Encrypted backups

`scripts/backup.sh` stops n8n for a few seconds, tars the data volume, encrypts
the archive to the `n8n-backup` GPG key, restarts n8n, and commits the result to
`backups/`:

```bash
./scripts/backup.sh
```

The archive excludes `storage/`, which holds regenerable execution binary
artifacts and is roughly 100x the size of everything else (49 MB vs 508 KB).
What it keeps is what defines the instance: `database.sqlite` (workflows,
credentials, execution metadata), `config` (the n8n encryption key), and
`nodes/`.

n8n is stopped during the tar so SQLite checkpoints its write-ahead log and the
archive is internally consistent. A `trap` restarts n8n even if the archive step
fails. Before committing, the script re-reads the archive and aborts if it is
still readable as a tar, so an unencrypted archive cannot reach a commit.

Local retention is 14 archives (`N8N_BACKUP_KEEP`); every archive is also pushed
to this repository.

### Restoring

```bash
./scripts/restore.sh backups/n8n-2026-08-20T033000.tgz.gpg
```

Verified end to end on 2026-08-20: encrypt, decrypt, extract, `PRAGMA
integrity_check` = ok, with all 3 workflows, 2 credentials and 18 execution
records intact.

### The backup key

`scripts/backup-public-key.asc` is the public half and is safe to commit. The
private half lives at `~/.config/n8n-backup/private-key.asc` (mode 600) and in
the local GPG keyring.

**Nothing in this repository can be read without that private key.** It is the
single artifact protecting every backup, and it is currently on the same disk as
the data it protects -- store an independent copy in a password manager. Because
the archive contains n8n's own `config`, saving this one key also covers the n8n
encryption key.

## Scheduling

Both jobs run from cron. `sudo` needs a password here and systemd *user* timers
do not run while the user is logged out (`Linger=no`), so cron is the option
that works unattended:

```cron
30 3 * * * /env/code/n8n-Historical-Planning-Audit/scripts/backup.sh >> /home/rmsi/.local/log/n8n-backup.log 2>&1
0  4 * * * /env/code/n8n-Historical-Planning-Audit/scripts/sync-workflows.sh >> /home/rmsi/.local/log/n8n-sync.log 2>&1
```
