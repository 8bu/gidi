"""The release manifest (``manifest.json``) and helpers shared by the release modules.

The manifest is the one machine-readable description of a release directory: which files it
holds (with sha256 and size), which runtime interface the model has, where the bits came from,
and which archives were derived from them. It never records archive hashes (the archives embed
the manifest, so that would be circular; ``checksums.txt`` covers them) and never an absolute
path. The serialized form is canonical: fixed key order, ``indent=2``, ``ensure_ascii=False``
and one trailing newline, so the same inputs always produce the same bytes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from gidi.inference.bundle import sha256_file

RELEASE_FORMAT = 1
MANIFEST_FILE = "manifest.json"
CHECKSUMS_FILE = "checksums.txt"
README_FILE = "README.md"
ARCHIVES_DIR = "archives"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

_TOP_KEYS = (
    "release_format",
    "model_version",
    "release_version",
    "git",
    "build_timestamp",
    "source",
    "runtime",
    "requirements",
    "artifacts",
    "archives",
)
_GIT_KEYS = ("commit", "dirty")
_SOURCE_KEYS = (
    "experiment",
    "seed",
    "experiment_protocol_sha256",
    "deployment_protocol_sha256",
    "bundle_manifest_sha256",
    "checkpoint_sha256",
    "spec_sha256",
)
_RUNTIME_KEYS = (
    "onnx_inputs",
    "onnx_outputs",
    "opset",
    "max_length",
    "types",
    "tags",
    "value_labels",
    "value_decoding",
    "prediction_keys",
    "schema_version",
)
# A bundle without a value head whose value span comes from the rule parser (gidi-finance-v3)
# also records the decoding rules; ``value_labels`` and ``value_decoding`` are then ``null``.
_RUNTIME_RULE_KEYS = ("value_source", "value_parser", "target_snap")


def runtime_keys(runtime: Any) -> tuple[str, ...]:
    """The exact key tuple of a manifest ``runtime`` block (the rule keys only when present)."""
    if isinstance(runtime, dict) and "value_source" in runtime:
        return (*_RUNTIME_KEYS, *_RUNTIME_RULE_KEYS)
    return _RUNTIME_KEYS


_ARTIFACT_KEYS = ("path", "kind", "sha256", "size_bytes")
_ARCHIVE_KEYS = ("path", "variant", "root", "members")


class ReleaseError(Exception):
    """A release cannot be built, read or trusted."""


def safe_relative_path(value: Any, what: str) -> str:
    """``value`` as a POSIX relative path without ``..`` or empty parts; else ``ReleaseError``."""
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"{what}: path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseError(f"{what}: {value!r} is not a plain relative POSIX path")
    if path.as_posix() != value:
        raise ReleaseError(f"{what}: {value!r} is not normalized")
    return value


def _require_keys(block: Any, keys: tuple[str, ...], what: str) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ReleaseError(f"manifest {what} must be an object")
    if tuple(block) != keys:
        raise ReleaseError(f"manifest {what} keys must be exactly {list(keys)}, got {list(block)}")
    return block


def _require_sha256(value: Any, what: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReleaseError(f"manifest {what} must be a lowercase sha256 hex digest")


@dataclass(frozen=True)
class Manifest:
    """In-memory form of ``manifest.json``; ``parse(m.serialize()) == m``."""

    model_version: str
    release_version: str
    git: dict[str, Any]
    build_timestamp: str | None
    source: dict[str, Any]
    runtime: dict[str, Any]
    requirements: dict[str, Any]
    artifacts: list[dict[str, Any]]
    archives: list[dict[str, Any]]
    release_format: int = RELEASE_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_format": self.release_format,
            "model_version": self.model_version,
            "release_version": self.release_version,
            "git": {key: self.git[key] for key in _GIT_KEYS},
            "build_timestamp": self.build_timestamp,
            "source": {key: self.source[key] for key in _SOURCE_KEYS},
            "runtime": {key: self.runtime[key] for key in runtime_keys(self.runtime)},
            "requirements": self.requirements,
            "artifacts": [{key: item[key] for key in _ARTIFACT_KEYS} for item in self.artifacts],
            "archives": [{key: item[key] for key in _ARCHIVE_KEYS} for item in self.archives],
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> Manifest:
        """Validate the schema of a decoded manifest; ``ReleaseError`` on any deviation."""
        top = _require_keys(data, _TOP_KEYS, "top level")
        if top["release_format"] != RELEASE_FORMAT:
            raise ReleaseError(f"unsupported release_format {top['release_format']!r}")
        for key in ("model_version", "release_version"):
            if not isinstance(top[key], str) or not top[key]:
                raise ReleaseError(f"manifest {key} must be a non-empty string")
        git = _require_keys(top["git"], _GIT_KEYS, "git")
        if git["commit"] is not None and not (
            isinstance(git["commit"], str) and _COMMIT.fullmatch(git["commit"])
        ):
            raise ReleaseError("manifest git.commit must be a 40-hex commit id or null")
        if git["dirty"] is not None and not isinstance(git["dirty"], bool):
            raise ReleaseError("manifest git.dirty must be a boolean or null")
        timestamp = top["build_timestamp"]
        if timestamp is not None and not (
            isinstance(timestamp, str) and _TIMESTAMP.fullmatch(timestamp)
        ):
            raise ReleaseError("manifest build_timestamp must be 'YYYY-MM-DDTHH:MM:SSZ' or null")
        source = _require_keys(top["source"], _SOURCE_KEYS, "source")
        if not isinstance(source["experiment"], str) or not isinstance(source["seed"], int):
            raise ReleaseError("manifest source.experiment/seed have the wrong type")
        for key in _SOURCE_KEYS[2:]:
            _require_sha256(source[key], f"source.{key}")
        runtime_block = top["runtime"]
        runtime = _require_keys(runtime_block, runtime_keys(runtime_block), "runtime")
        if not isinstance(top["requirements"], dict):
            raise ReleaseError("manifest requirements must be an object")
        artifacts = top["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ReleaseError("manifest artifacts must be a non-empty list")
        for item in artifacts:
            _require_keys(item, _ARTIFACT_KEYS, "artifact")
            safe_relative_path(item["path"], "manifest artifact")
            if not isinstance(item["kind"], str) or not item["kind"]:
                raise ReleaseError(f"manifest artifact {item['path']}: kind must be a string")
            _require_sha256(item["sha256"], f"artifact {item['path']} sha256")
            size = item["size_bytes"]
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ReleaseError(f"manifest artifact {item['path']}: bad size_bytes")
        paths = [item["path"] for item in artifacts]
        if paths != sorted(set(paths)):
            raise ReleaseError("manifest artifacts must be unique and sorted by path")
        archives = top["archives"]
        if not isinstance(archives, list):
            raise ReleaseError("manifest archives must be a list")
        for item in archives:
            _require_keys(item, _ARCHIVE_KEYS, "archive")
            safe_relative_path(item["path"], "manifest archive")
            safe_relative_path(item["root"], "manifest archive root")
            members = item["members"]
            if not isinstance(members, dict) or not members:
                raise ReleaseError(f"manifest archive {item['path']}: members must be an object")
            for name, target in members.items():
                safe_relative_path(name, "manifest archive member")
                safe_relative_path(target, "manifest archive member target")
        return cls(
            model_version=top["model_version"],
            release_version=top["release_version"],
            git=git,
            build_timestamp=timestamp,
            source=source,
            runtime=runtime,
            requirements=top["requirements"],
            artifacts=artifacts,
            archives=archives,
        )

    @classmethod
    def parse(cls, text: str) -> Manifest:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ReleaseError(f"manifest is not valid JSON: {error}") from error
        return cls.from_dict(data)


def load_manifest(release_dir: str | Path) -> dict[str, Any]:
    """The decoded ``manifest.json`` of a release directory (schema-validated)."""
    path = Path(release_dir) / MANIFEST_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError(f"cannot read {path}: {error}") from error
    return Manifest.parse(text).to_dict()


def manifest_sha256(release_dir: str | Path) -> str:
    """sha256 of the ``manifest.json`` bytes as they are on disk."""
    path = Path(release_dir) / MANIFEST_FILE
    if not path.is_file():
        raise ReleaseError(f"missing {path}")
    return sha256_file(path)


def snapshot(release_dir: str | Path) -> dict[str, tuple[str, int]]:
    """Every file under ``release_dir`` as ``{posix relative path: (sha256, size)}``.

    Used to prove a release was not changed between two points in time.
    """
    root = Path(release_dir)
    if not root.is_dir():
        raise ReleaseError(f"release directory not found: {root}")
    result: dict[str, tuple[str, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = (sha256_file(path), path.stat().st_size)
    return result


def release_dir_for(
    model_version: str, release_version: str, dist_root: Path = Path("dist")
) -> Path:
    """``<dist_root>/releases/<model_version>/<release_version>``."""
    return Path(dist_root) / "releases" / model_version / release_version
