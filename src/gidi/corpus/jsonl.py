"""JSONL read/write. Writes never replace an existing file unless explicitly asked."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-empty line; errors report the 1-based line number."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}") from e
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            yield record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(
    path: str | Path, records: Iterable[dict[str, Any]], *, overwrite: bool = False
) -> int:
    """Write records atomically as UTF-8 JSONL (Vietnamese kept unescaped).

    Refuses to replace an existing file unless ``overwrite=True`` so raw corpus files are
    never clobbered by review or transformation steps. Returns the number of records.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
                count += 1
        if overwrite:
            os.replace(tmp, path)
        else:
            # link() fails if the target appeared meanwhile, keeping the no-overwrite guarantee.
            os.link(tmp, path)
            os.unlink(tmp)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return count
