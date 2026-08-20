"""Normalise n8n workflow exports into committable, diff-stable JSON.

Raw exports carry instance-local bookkeeping (timestamps, version counters) that
produces a diff on every run, and a `shared` block naming the owning user and
their email address. Both are stripped here.
"""
import json
import pathlib
import re
import sys

# Dropped because they change without the workflow changing, or identify the
# local instance/owner rather than the automation itself.
DROP_KEYS = {
    "updatedAt",
    "createdAt",
    "versionId",
    "activeVersionId",
    "versionCounter",
    "triggerCount",
    "isArchived",
    "shared",
    "sourceWorkflowId",
    "versionMetadata",
}

# A committed workflow must never carry a live secret. Node `credentials` blocks
# hold only an id and a display name, so anything matching here is a leak.
SECRET_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "email address"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "API key"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "GitHub token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "workflow"


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in DROP_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def main(stage_dir: str, out_dir: str) -> int:
    stage = pathlib.Path(stage_dir)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = set()
    for src in sorted(stage.glob("*.json")):
        workflow = scrub(json.loads(src.read_text()))
        # Keep the id in the filename so a rename does not orphan the history.
        dest = out / f"{slugify(workflow['name'])}.{workflow['id']}.json"
        body = json.dumps(workflow, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

        for pattern, label in SECRET_PATTERNS:
            hit = pattern.search(body)
            if hit:
                print(
                    f"refusing to write {dest.name}: found {label} ({hit.group(0)})",
                    file=sys.stderr,
                )
                return 1

        dest.write_text(body)
        written.add(dest.name)

    # Workflows deleted in n8n should disappear from the repo too.
    for stale in out.glob("*.json"):
        if stale.name not in written:
            stale.unlink()
            print(f"removed {stale.name}")

    print(f"synced {len(written)} workflows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
