"""Build the scan inventory a mapping joins against, and check it is complete.

Nothing upstream produces this. Exeter's index was built by a one-off script and
Sheffield had none, so the first attempt at Sheffield listed one of its three
scan trees and missed 38% of the files. Downstream that read as "the mapping
cannot find a candidate" rather than as "the evidence is incomplete", and the
join rate had to be reverse-engineered to discover why.

Three things are council knowledge rather than anything stated in the capture
rules or the source table, and each of them was got wrong once:

  which trees hold the scans -- Sheffield's are split across two buckets;
  what a record is -- a Fiche record is a folder of page images while a U-Drive
      record is a single PDF, so listing both at object level buries the folders
      under 704,000 pages;
  what is not a record -- inventory spreadsheets and Thumbs.db sit in the same
      trees.

So they are declared per council, once, and the build is checked against the
delivered mapping when one exists: a prefix that appears in delivered paths but
in no declared source is a missing tree, and saying so here is far cheaper than
inferring it from a low join rate later.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


class InventoryError(RuntimeError):
    pass


INVENTORY_FIELDS = (
    "source",
    "bucket",
    "parent_prefix",
    "folder",
    "filename",
    "amazons3_path",
    "size_bytes",
    "inventory_evidence",
)

# Files that live in the scan trees without being scans.
DEFAULT_EXCLUDE = (r"(?i)^thumbs\.db$", r"(?i)^\.ds_store$", r"(?i)inventory\.xlsx?$")


@dataclass(frozen=True)
class InventorySource:
    """One tree of scans, and what a record looks like inside it."""

    name: str
    bucket: str
    prefix: str
    # "folder": a record is a directory of page images, as on Fiche.
    # "file": a record is a single document, as on U-Drive.
    granularity: str = "folder"
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE

    def validate(self) -> None:
        if self.granularity not in {"folder", "file"}:
            raise InventoryError(
                f"source {self.name!r}: granularity must be 'folder' or 'file', "
                f"got {self.granularity!r}"
            )
        if not self.bucket or not self.prefix:
            raise InventoryError(f"source {self.name!r} needs a bucket and a prefix")

    def excluded(self, name: str) -> bool:
        return any(re.search(pattern, name) for pattern in self.exclude)


def load_sources(path: Path) -> tuple[InventorySource, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("sources") if isinstance(payload, dict) else payload
    if not entries:
        raise InventoryError(f"No inventory sources declared in {path}")
    sources = []
    for entry in entries:
        source = InventorySource(
            name=str(entry["name"]),
            bucket=str(entry["bucket"]),
            prefix=str(entry["prefix"]),
            granularity=str(entry.get("granularity", "folder")),
            exclude=tuple(entry.get("exclude") or DEFAULT_EXCLUDE),
        )
        source.validate()
        sources.append(source)
    return tuple(sources)


def build_rows(
    sources: Sequence[InventorySource],
    *,
    lister: Callable[[str, str], Iterable[dict[str, Any]]],
) -> list[dict[str, str]]:
    """List every declared tree into one table of records."""
    rows: list[dict[str, str]] = []
    for source in sources:
        seen_folders: set[str] = set()
        for obj in lister(source.bucket, source.prefix):
            key = str(obj["Key"])
            if key.endswith("/"):
                continue
            name = key.rsplit("/", 1)[-1]
            if source.excluded(name):
                continue
            parent = key[: len(key) - len(name)]
            if source.granularity == "folder":
                # The record is the directory holding the pages, listed once.
                directory = parent.rstrip("/")
                if not directory or directory in seen_folders:
                    continue
                seen_folders.add(directory)
                rows.append(
                    {
                        "source": source.name,
                        "bucket": source.bucket,
                        "parent_prefix": directory.rsplit("/", 1)[0] + "/",
                        "folder": directory.rsplit("/", 1)[-1],
                        "filename": "",
                        "amazons3_path": f"s3://{source.bucket}/{directory}",
                        "size_bytes": "",
                        "inventory_evidence": "list_objects_v2",
                    }
                )
                continue
            rows.append(
                {
                    "source": source.name,
                    "bucket": source.bucket,
                    "parent_prefix": parent,
                    "folder": name.rsplit(".", 1)[0] if "." in name else name,
                    "filename": name,
                    "amazons3_path": f"s3://{source.bucket}/{key}",
                    "size_bytes": str(obj.get("Size", "")),
                    "inventory_evidence": "list_objects_v2",
                }
            )
    if not rows:
        raise InventoryError("No scan records were listed from the declared sources")
    return rows


def delivered_prefixes(paths: Iterable[str], *, depth: int = 5) -> set[str]:
    """The distinct S3 trees a delivered mapping actually points into."""
    prefixes = set()
    for path in paths:
        value = (path or "").strip()
        if not value.startswith("s3://"):
            continue
        parts = value[len("s3://") :].split("/")
        if len(parts) > 1:
            prefixes.add("/".join(parts[: min(depth, len(parts) - 1)]))
    return prefixes


def missing_trees(
    sources: Sequence[InventorySource], delivered: Iterable[str], *, depth: int = 5
) -> list[str]:
    """Delivered trees no declared source covers.

    A mapping that was delivered from a tree the inventory never lists cannot be
    reproduced, and the shortfall shows up downstream as unmatched cases.
    """
    covered = [f"{source.bucket}/{source.prefix}".rstrip("/") for source in sources]
    missing = []
    for prefix in sorted(delivered_prefixes(delivered, depth=depth)):
        if not any(prefix.startswith(item) or item.startswith(prefix) for item in covered):
            missing.append(prefix)
    return missing


def write_inventory(rows: Sequence[dict[str, str]], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INVENTORY_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def summarise(rows: Sequence[dict[str, str]], sources: Sequence[InventorySource]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    return {
        "records": len(rows),
        "by_source": counts,
        "sources": [
            {
                "name": source.name,
                "bucket": source.bucket,
                "prefix": source.prefix,
                "granularity": source.granularity,
            }
            for source in sources
        ],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
