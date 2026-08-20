from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


FILE_BROWSER_DATA_ROOT = Path("/data/file-browser-data")
ISOLATION_ENV = "FILE_MATCHING_FILE_BROWSER_DATA_ISOLATED"
READ_ONLY_JOB_ISOLATION_ENV = "FILE_MATCHING_READ_ONLY_JOB_ISOLATED"


class FileBrowserDataAccessDenied(PermissionError):
    """Raised when mapping or audit code targets the runtime file-browser tree."""


class FileBrowserIsolationUnavailable(RuntimeError):
    """Raised when the fail-closed process isolation layer cannot be started."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def require_unprotected_path(path: str | Path, *, operation: str) -> Path:
    """Reject the protected root, descendants, and paths that symlink into it."""

    resolved = _resolved(path)
    protected = _resolved(FILE_BROWSER_DATA_ROOT)
    if resolved == protected or protected in resolved.parents:
        raise FileBrowserDataAccessDenied(
            f"{operation} access to {resolved} is denied: autonomous mapping and audit "
            f"must not read or write {protected}"
        )
    return resolved


def require_unprotected_paths(
    paths: Iterable[tuple[str, str | Path]],
    *,
    operation: str,
) -> dict[str, Path]:
    return {
        label: require_unprotected_path(path, operation=f"{operation} ({label})")
        for label, path in paths
    }


def require_unprotected_url(url: str, *, operation: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return
    if parsed.netloc not in {"", "localhost"}:
        raise FileBrowserDataAccessDenied(
            f"{operation} access through a remote-host file URL is denied: {url}"
        )
    require_unprotected_path(Path(unquote(parsed.path)), operation=operation)


def policy_record() -> dict[str, object]:
    return {
        "policy": "deny_file_browser_data_read_write",
        "protected_roots": [str(FILE_BROWSER_DATA_ROOT)],
        "symlink_aliases_denied": True,
        "process_isolation": "bubblewrap_empty_read_only_mount",
        "fail_closed": True,
    }


def process_isolation_active() -> bool:
    """Verify mount metadata without listing or otherwise reading the protected tree."""

    if os.environ.get(ISOLATION_ENV) != "1":
        return False
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return False
    protected = str(FILE_BROWSER_DATA_ROOT)
    for line in mountinfo.splitlines():
        try:
            mount, filesystem = line.split(" - ", 1)
        except ValueError:
            continue
        fields = mount.split()
        filesystem_fields = filesystem.split()
        if len(fields) < 6 or not filesystem_fields:
            continue
        mountpoint = fields[4].replace("\\040", " ")
        mount_options = set(fields[5].split(","))
        if mountpoint == protected and "ro" in mount_options and filesystem_fields[0] == "tmpfs":
            return True
    return False


def read_only_job_isolation_active() -> bool:
    """True only inside the stricter mapping-job sandbox used by n8n.

    The outer sandbox makes the root filesystem read-only, exposes one
    explicit job root as writable, and hides the protected runtime tree. In
    that environment Codex must not start a second bubblewrap namespace.
    """

    if os.environ.get(READ_ONLY_JOB_ISOLATION_ENV) != "1" or not process_isolation_active():
        return False
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return False
    for line in mountinfo.splitlines():
        try:
            mount, _ = line.split(" - ", 1)
        except ValueError:
            continue
        fields = mount.split()
        if len(fields) >= 6 and fields[4] == "/":
            return "ro" in set(fields[5].split(","))
    return False


def file_browser_isolated_command(command: list[str]) -> list[str]:
    """Hide the real file-browser tree from a subprocess and make its mount read-only."""

    if process_isolation_active():
        return command
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise FileBrowserIsolationUnavailable(
            "bubblewrap is required for autonomous mapping/audit isolation; refusing to run "
            "without a hard /data/file-browser-data boundary"
        )
    protected = str(FILE_BROWSER_DATA_ROOT)
    return [
        bubblewrap,
        "--die-with-parent",
        "--dev-bind",
        "/",
        "/",
        "--dir",
        protected,
        "--tmpfs",
        protected,
        "--remount-ro",
        protected,
        "--setenv",
        ISOLATION_ENV,
        "1",
        "--",
        *command,
    ]


def read_only_job_isolated_command(
    command: list[str],
    *,
    writable_root: str | Path,
    codex_home: str | Path | None = None,
    codex_auth: str | Path | None = None,
) -> list[str]:
    """Run a mapping job with a read-only host view and one writable job root."""

    root = require_unprotected_path(writable_root, operation="use writable mapping-job root")
    root.mkdir(parents=True, exist_ok=True)
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise FileBrowserIsolationUnavailable(
            "bubblewrap is required for the read-only mapping-job boundary"
        )
    protected = str(FILE_BROWSER_DATA_ROOT)
    wrapped = [
        bubblewrap,
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        str(root),
        str(root),
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        protected,
        "--remount-ro",
        protected,
        "--setenv",
        ISOLATION_ENV,
        "1",
        "--setenv",
        READ_ONLY_JOB_ISOLATION_ENV,
        "1",
    ]
    if codex_home is not None or codex_auth is not None:
        if codex_home is None or codex_auth is None:
            raise ValueError("codex_home and codex_auth must be supplied together")
        runtime_home = require_unprotected_path(codex_home, operation="use isolated Codex home")
        auth = require_unprotected_path(codex_auth, operation="mount Codex OAuth credential")
        if root != runtime_home and root not in runtime_home.parents:
            raise ValueError("isolated Codex home must be inside the writable mapping-job root")
        if not auth.is_file():
            raise FileNotFoundError(f"Codex OAuth credential does not exist: {auth}")
        runtime_home.mkdir(parents=True, exist_ok=True)
        auth_target = runtime_home / "auth.json"
        auth_target.touch(exist_ok=True)
        wrapped.extend(
            [
                "--ro-bind",
                str(auth),
                str(auth_target),
                "--setenv",
                "CODEX_HOME",
                str(runtime_home),
            ]
        )
    return [*wrapped, "--", *command]
