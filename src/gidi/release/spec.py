"""The committed release spec (``releases/<model>.json``): the reviewed, hash-pinned input.

A release is assembled only from files the spec names, and every one of them is pinned by sha256
and size, so a build can neither pick up a stale model nor silently change one. Nothing in the
spec is generated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gidi.inference.bundle import sha256_file
from gidi.release.manifest import ARCHIVES_DIR, README_FILE, ReleaseError, safe_relative_path

SPEC_FORMAT = 1
# Names a variant member may map to besides a spec file: both are generated at build time.
GENERATED_MEMBER_TARGETS = ("README.md", "manifest.json")
SMOKE_EXPECTED_KEYS = ("type", "target", "value_text")

_SEMVER = re.compile(
    r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)
_MODEL_MAJOR = re.compile(r".*-v(\d+)$")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPEC_KEYS = {
    "spec_format",
    "model_version",
    "source",
    "bundle_check",
    "files",
    "readmes",
    "variants",
    "smoke",
}


def repo_root() -> Path:
    """The repository root this package was imported from (``src/gidi/release/spec.py``)."""
    return Path(__file__).resolve().parents[3]


def default_spec_path(model_version: str, root: Path) -> Path:
    return Path(root) / "releases" / f"{model_version}.json"


def validate_release_version(model_version: str, version: str) -> None:
    """``MAJOR.MINOR.PATCH[-prerelease]`` whose MAJOR equals the model's trailing ``-vN``."""
    match = _SEMVER.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise ReleaseError(f"release version {version!r} is not MAJOR.MINOR.PATCH[-prerelease]")
    model_match = _MODEL_MAJOR.fullmatch(model_version)
    if model_match is None:
        raise ReleaseError(f"model version {model_version!r} does not end in -v<MAJOR>")
    if int(match["major"]) != int(model_match[1]):
        raise ReleaseError(
            f"release version {version} has major {match['major']} but {model_version} "
            f"needs major {model_match[1]}"
        )


@dataclass(frozen=True)
class SpecFile:
    source: str
    path: str
    kind: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class Variant:
    name: str
    members: dict[str, str]  # archive member name -> release path
    model_file: str  # member name of the ONNX file


@dataclass(frozen=True)
class SmokeCase:
    text: str
    expected: dict[str, Any]


@dataclass(frozen=True)
class ReleaseSpec:
    path: Path
    sha256: str  # of the spec file bytes
    model_version: str
    experiment: str
    seed: int
    experiment_protocol: tuple[str, str]  # (path, sha256)
    deployment_protocol: tuple[str, str]
    bundle_path: str
    bundle_manifest_sha256: str
    checkpoint_sha256: str
    bundle_check: dict[str, str] | None
    files: tuple[SpecFile, ...]
    readmes: dict[str, str]  # release path (README.md first) -> template path
    variants: tuple[Variant, ...]
    smoke: tuple[SmokeCase, ...]

    def file_for_kind(self, kind: str) -> SpecFile:
        for item in self.files:
            if item.kind == kind:
                return item
        raise ReleaseError(f"spec {self.model_version} has no file of kind {kind!r}")


def _object(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"spec {what} must be an object")
    return value


def _string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"spec {what} must be a non-empty string")
    return value


def _sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReleaseError(f"spec {what} must be a lowercase sha256 hex digest")
    return value


def _pinned_path(block: Any, what: str) -> tuple[str, str]:
    block = _object(block, what)
    return (
        safe_relative_path(block.get("path"), f"spec {what}"),
        _sha(block.get("sha256"), f"{what}.sha256"),
    )


def _parse_files(raw: Any) -> tuple[SpecFile, ...]:
    if not isinstance(raw, list) or not raw:
        raise ReleaseError("spec files must be a non-empty list")
    files = []
    for item in raw:
        item = _object(item, "files[]")
        if set(item) != {"source", "path", "kind", "sha256", "size_bytes"}:
            raise ReleaseError(f"spec files[] keys are wrong: {sorted(item)}")
        size = item["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReleaseError(f"spec file {item['path']}: bad size_bytes")
        path = safe_relative_path(item["path"], "spec file")
        if path.split("/")[0] in (ARCHIVES_DIR, "manifest.json", "checksums.txt", "README.md"):
            raise ReleaseError(f"spec file path {path!r} collides with a generated release path")
        files.append(
            SpecFile(
                source=safe_relative_path(item["source"], "spec file source"),
                path=path,
                kind=_string(item["kind"], "file kind"),
                sha256=_sha(item["sha256"], f"file {path} sha256"),
                size_bytes=size,
            )
        )
    paths = [item.path for item in files]
    if len(set(paths)) != len(paths):
        raise ReleaseError("spec files have duplicate release paths")
    return tuple(files)


def _parse_readmes(raw: Any) -> dict[str, str]:
    """``{release path: template}``; ``README.md`` (the default card) is required."""
    raw = _object(raw, "readmes")
    if README_FILE not in raw:
        raise ReleaseError(f"spec readmes must include {README_FILE}")
    readmes: dict[str, str] = {}
    for path, template in raw.items():
        path = safe_relative_path(path, "spec readmes path")
        if "/" in path or not path.endswith(".md"):
            raise ReleaseError(f"spec readmes path {path!r} must be a top-level .md file")
        readmes[path] = safe_relative_path(template, f"spec readmes.{path}")
    return readmes


def _parse_variants(
    raw: Any, files: tuple[SpecFile, ...], readmes: dict[str, str]
) -> tuple[Variant, ...]:
    raw = _object(raw, "variants")
    if not raw:
        raise ReleaseError("spec variants must not be empty")
    valid_targets = {item.path for item in files} | set(GENERATED_MEMBER_TARGETS) | set(readmes)
    variants = []
    for name, block in raw.items():
        _string(name, "variant name")
        if re.fullmatch(r"[a-z0-9]+", name) is None:
            raise ReleaseError(f"spec variant name {name!r} must be lowercase alphanumeric")
        block = _object(block, f"variants.{name}")
        if set(block) != {"members", "model_file"}:
            raise ReleaseError(f"spec variants.{name} keys must be members and model_file")
        members = _object(block["members"], f"variants.{name}.members")
        for member, target in members.items():
            if not member or "/" in member or member in (".", ".."):
                raise ReleaseError(f"spec variant {name}: bad member name {member!r}")
            if target not in valid_targets:
                raise ReleaseError(
                    f"spec variant {name}: member {member} maps to unknown {target!r}"
                )
        model_file = _string(block["model_file"], f"variants.{name}.model_file")
        for required in (model_file, "config.json", "tokenizer.json"):
            if required not in members:
                raise ReleaseError(f"spec variant {name} has no member {required!r}")
        variants.append(Variant(name=name, members=dict(members), model_file=model_file))
    return tuple(variants)


def _parse_smoke(raw: Any) -> tuple[SmokeCase, ...]:
    if not isinstance(raw, list):
        raise ReleaseError("spec smoke must be a list")
    cases = []
    for item in raw:
        item = _object(item, "smoke[]")
        expected = _object(item.get("expected"), "smoke[].expected")
        if set(item) != {"text", "expected"} or tuple(sorted(expected)) != tuple(
            sorted(SMOKE_EXPECTED_KEYS)
        ):
            raise ReleaseError(f"spec smoke case keys are wrong: {item}")
        cases.append(SmokeCase(text=_string(item["text"], "smoke text"), expected=expected))
    return tuple(cases)


def load_spec(path: str | Path) -> ReleaseSpec:
    """Read and validate a release spec; every structural problem is a ``ReleaseError``."""
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read release spec {path}: {error}") from error
    data = _object(data, "root")
    unknown = set(data) - _SPEC_KEYS
    missing = _SPEC_KEYS - {"bundle_check"} - set(data)
    if unknown or missing:
        raise ReleaseError(f"spec keys wrong: unknown {sorted(unknown)}, missing {sorted(missing)}")
    if data["spec_format"] != SPEC_FORMAT:
        raise ReleaseError(f"unsupported spec_format {data['spec_format']!r}")
    source = _object(data["source"], "source")
    bundle = _object(source.get("bundle"), "source.bundle")
    bundle_check = data.get("bundle_check")
    if bundle_check is not None:
        bundle_check = _object(bundle_check, "bundle_check")
        if set(bundle_check) != {"script", "bundle", "protocol"}:
            raise ReleaseError("spec bundle_check keys must be script, bundle, protocol")
        for key, value in bundle_check.items():
            safe_relative_path(value, f"spec bundle_check.{key}")
    seed = source.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ReleaseError("spec source.seed must be an integer")
    files = _parse_files(data["files"])
    readmes = _parse_readmes(data["readmes"])
    if set(readmes) & {item.path for item in files}:
        raise ReleaseError("spec readmes paths collide with spec files")
    return ReleaseSpec(
        path=path,
        sha256=sha256_file(path),
        model_version=_string(data["model_version"], "model_version"),
        experiment=_string(source.get("experiment"), "source.experiment"),
        seed=seed,
        experiment_protocol=_pinned_path(source.get("experiment_protocol"), "experiment_protocol"),
        deployment_protocol=_pinned_path(source.get("deployment_protocol"), "deployment_protocol"),
        bundle_path=safe_relative_path(bundle.get("path"), "spec source.bundle"),
        bundle_manifest_sha256=_sha(bundle.get("manifest_sha256"), "bundle.manifest_sha256"),
        checkpoint_sha256=_sha(source.get("checkpoint_sha256"), "source.checkpoint_sha256"),
        bundle_check=bundle_check,
        files=files,
        readmes=readmes,
        variants=_parse_variants(data["variants"], files, readmes),
        smoke=_parse_smoke(data["smoke"]),
    )
