"""Build a deterministic release directory from a hash-pinned spec.

A build copies the pinned files byte for byte (no export, quantization or regeneration), renders
the README from a template, writes ``manifest.json``, the deterministic archives and
``checksums.txt``, and makes every file read-only. The same inputs always give the same bytes:
timestamps come only from ``SOURCE_DATE_EPOCH`` or the git HEAD commit time, never from the
clock. A finished release is never overwritten; rebuilding into an existing directory is a no-op
when the bytes match and an error when they do not.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gidi.inference.bundle import sha256_file
from gidi.release.manifest import (
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    README_FILE,
    Manifest,
    ReleaseError,
    snapshot,
)
from gidi.release.package import archive_path, archive_root, member_sources, write_archive
from gidi.release.spec import ReleaseSpec, validate_release_version

# ``Prediction.to_dict()`` keys of a bundle with a value head.
PREDICTION_KEYS = (
    "type",
    "type_confidence",
    "target",
    "target_span",
    "target_confidence",
    "value_text",
    "value_span",
    "value_confidence",
    "truncated",
    "model_version",
)
READ_ONLY = 0o444
_CHUNK = 1 << 20
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class GitState:
    commit: str | None  # None outside a git repository
    dirty: bool | None
    commit_epoch: int | None


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def git_state(root: Path) -> GitState:
    """HEAD commit, working-tree dirtiness and HEAD commit time; all ``None`` without git."""
    head = _git(root, "rev-parse", "HEAD")
    if head is None or not _COMMIT.fullmatch(head.strip()):
        return GitState(None, None, None)
    status = _git(root, "status", "--porcelain")
    epoch = _git(root, "log", "-1", "--format=%ct")
    return GitState(
        commit=head.strip(),
        dirty=None if status is None else bool(status.strip()),
        commit_epoch=int(epoch.strip()) if epoch and epoch.strip().isdigit() else None,
    )


def build_epoch(git: GitState) -> int | None:
    """``SOURCE_DATE_EPOCH`` if set, else the git HEAD commit time, else ``None``."""
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is not None and value != "":
        if not value.isdigit():
            raise ReleaseError(f"SOURCE_DATE_EPOCH must be an integer, got {value!r}")
        return int(value)
    return git.commit_epoch


def iso_timestamp(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def runtime_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest ``runtime`` block, derived only from the bundle ``config.json``."""
    try:
        onnx = config["onnx"]
        return {
            "onnx_inputs": list(onnx["inputs"]),
            "onnx_outputs": list(onnx["outputs"]),
            "opset": onnx["opset"],
            "max_length": config["max_length"],
            "types": list(config["types"]),
            "tags": list(config["tags"]),
            "value_labels": list(config["value_labels"]),
            "value_decoding": config["value_decoding"]["method"],
            "prediction_keys": list(PREDICTION_KEYS),
            "schema_version": None,
        }
    except (KeyError, TypeError) as error:
        raise ReleaseError(f"runtime config.json lacks the interface fields: {error!r}") from error


def readme_front_matter_problems(text: str, *, license_file: bool) -> list[str]:
    """Problems with the README YAML front matter (language vi, onnx library, license).

    The front matter declares ``license:`` exactly when the release ships a ``license`` file.
    """
    lines = text.split("\n")
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        return ["README.md has no YAML front matter"]
    block = lines[1 : lines.index("---", 1)]
    keys = {line.split(":", 1)[0].strip() for line in block if ":" in line and line[:1] != " "}
    problems = []
    if "language: vi" not in block:
        problems.append("README front matter must declare 'language: vi'")
    if "library_name: onnx" not in block:
        problems.append("README front matter must declare 'library_name: onnx'")
    if license_file and "license" not in keys:
        problems.append("README front matter must declare the license of the shipped LICENSE")
    if not license_file and "license" in keys:
        problems.append("README front matter declares a license but the release has no LICENSE")
    return problems


def render_readme(template: str, values: Mapping[str, Any]) -> str:
    """Replace ``{name}`` placeholders; an unknown placeholder is an error, never left behind."""

    def substitute(match: re.Match[str]) -> str:
        key = match[1]
        if key not in values:
            raise ReleaseError(f"README template uses unknown placeholder {{{key}}}")
        return str(values[key])

    return _PLACEHOLDER.sub(substitute, template)


def _size_mb(size: int) -> str:
    return f"{size / 1_000_000:.1f} MB"


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read {what} {path}: {error}") from error
    if not isinstance(data, dict):
        raise ReleaseError(f"{what} {path} must be a JSON object")
    return data


def _require_pinned(path: Path, sha256: str, what: str, size: int | None = None) -> None:
    if not path.is_file():
        raise ReleaseError(f"{what} not found: {path}")
    if size is not None and path.stat().st_size != size:
        raise ReleaseError(f"{what} size is {path.stat().st_size}, spec pins {size}: {path}")
    if sha256_file(path) != sha256:
        raise ReleaseError(f"{what} sha256 differs from the spec pin: {path}")


def _check_sources(spec: ReleaseSpec, root: Path) -> dict[str, Any]:
    """Pinned protocols and bundle manifest; returns the decoded bundle manifest."""
    _require_pinned(root / spec.experiment_protocol[0], spec.experiment_protocol[1], "protocol")
    _require_pinned(root / spec.deployment_protocol[0], spec.deployment_protocol[1], "protocol")
    manifest_path = root / spec.bundle_path / MANIFEST_FILE
    _require_pinned(manifest_path, spec.bundle_manifest_sha256, "bundle manifest")
    bundle = _read_json(manifest_path, "bundle manifest")
    sources = bundle.get("sources", {})
    expected = {
        "bundle protocol": (bundle.get("protocol", {}).get("sha256"), spec.deployment_protocol[1]),
        "bundle experiment protocol": (
            sources.get("source_experiment_protocol", {}).get("sha256"),
            spec.experiment_protocol[1],
        ),
        "bundle checkpoint": (
            sources.get("checkpoint_model_safetensors", {}).get("sha256"),
            spec.checkpoint_sha256,
        ),
    }
    for what, (actual, pinned) in expected.items():
        if actual != pinned:
            raise ReleaseError(f"{what} sha256 in the bundle manifest differs from the spec pin")
    if not isinstance(bundle.get("runtime_requirements"), dict):
        raise ReleaseError("bundle manifest has no runtime_requirements")
    prefix = f"{spec.bundle_path}/"
    listed = bundle.get("files", {})
    for item in spec.files:
        if item.source.startswith(prefix) and "/" not in item.source[len(prefix) :]:
            recorded = listed.get(item.source[len(prefix) :], {}).get("sha256")
        elif item.kind == "onnx-fp32":
            recorded = sources.get("onnx_fp32", {}).get("sha256")
        else:
            continue
        if recorded != item.sha256:
            raise ReleaseError(f"{item.source}: bundle manifest records a different sha256")
    return bundle


def _run_bundle_check(spec: ReleaseSpec, root: Path) -> None:
    check = spec.bundle_check
    assert check is not None
    command = [
        sys.executable,
        check["script"],
        "--out",
        check["bundle"],
        "--protocol",
        check["protocol"],
        "--check",
    ]
    done = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        tail = (done.stdout + done.stderr).strip().splitlines()[-8:]
        raise ReleaseError(
            f"bundle {check['bundle']} is not a byte-identical rebuild "
            f"(exit {done.returncode}): " + " | ".join(tail)
        )


def _copy_verified(source: Path, dest: Path, sha256: str, size: int) -> None:
    """Copy ``source`` to ``dest`` byte for byte; the copy must match the spec pin."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader, dest.open("wb") as writer:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
            writer.write(chunk)
    if digest.hexdigest() != sha256 or dest.stat().st_size != size:
        raise ReleaseError(f"{source} does not match the spec pin (sha256/size)")


def _readme_values(
    spec: ReleaseSpec, version: str, config: Mapping[str, Any], requirements: Mapping[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_version": spec.model_version,
        "release_version": version,
        "experiment": spec.experiment,
        "seed": spec.seed,
        "max_length": config["max_length"],
        "types": ", ".join(config["types"]),
        "python_requirement": requirements.get("python", ""),
        "onnxruntime_requirement": requirements.get("onnxruntime", ""),
        "tokenizers_requirement": requirements.get("tokenizers", ""),
        "numpy_requirement": requirements.get("numpy", ""),
    }
    for item in spec.files:
        if item.kind.startswith("onnx-"):
            name = item.kind.removeprefix("onnx-")
            values[f"{name}_sha256"] = item.sha256
            values[f"{name}_size_mb"] = _size_mb(item.size_bytes)
    for variant in spec.variants:
        values[f"{variant.name}_archive"] = Path(
            archive_path(spec.model_version, version, variant.name)
        ).name
    return values


def _assemble(
    spec: ReleaseSpec, version: str, root: Path, tmp: Path, git: GitState, bundle: dict[str, Any]
) -> None:
    for item in spec.files:
        _copy_verified(root / item.source, tmp / item.path, item.sha256, item.size_bytes)
    config_file = spec.file_for_kind("runtime-config")
    config = _read_json(tmp / config_file.path, "runtime config")
    if config.get("model_version") != spec.model_version:
        raise ReleaseError(
            f"{config_file.source} model_version {config.get('model_version')!r} "
            f"!= spec {spec.model_version!r}"
        )
    runtime = runtime_from_config(config)
    requirements = bundle["runtime_requirements"]

    try:
        template = (root / spec.readme_template).read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError(
            f"cannot read README template {spec.readme_template}: {error}"
        ) from error
    readme = render_readme(template, _readme_values(spec, version, config, requirements))
    problems = readme_front_matter_problems(
        readme, license_file=any(item.kind == "license" for item in spec.files)
    )
    if problems:
        raise ReleaseError("; ".join(problems))
    (tmp / README_FILE).write_text(readme, encoding="utf-8")

    artifacts = [
        {"path": item.path, "kind": item.kind, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in spec.files
    ]
    readme_path = tmp / README_FILE
    artifacts.append(
        {
            "path": README_FILE,
            "kind": "readme",
            "sha256": sha256_file(readme_path),
            "size_bytes": readme_path.stat().st_size,
        }
    )
    artifacts.sort(key=lambda item: item["path"])
    archives = [
        {
            "path": archive_path(spec.model_version, version, variant.name),
            "variant": variant.name,
            "root": archive_root(spec.model_version, version, variant.name),
            "members": dict(variant.members),
        }
        for variant in spec.variants
    ]
    epoch = build_epoch(git)
    manifest = Manifest(
        model_version=spec.model_version,
        release_version=version,
        git={"commit": git.commit, "dirty": git.dirty},
        build_timestamp=iso_timestamp(epoch),
        source={
            "experiment": spec.experiment,
            "seed": spec.seed,
            "experiment_protocol_sha256": spec.experiment_protocol[1],
            "deployment_protocol_sha256": spec.deployment_protocol[1],
            "bundle_manifest_sha256": spec.bundle_manifest_sha256,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "spec_sha256": spec.sha256,
        },
        runtime=runtime,
        requirements=dict(requirements),
        artifacts=artifacts,
        archives=archives,
    )
    (tmp / MANIFEST_FILE).write_text(manifest.serialize(), encoding="utf-8")

    (tmp / "archives").mkdir()
    for archive in archives:
        write_archive(
            tmp / archive["path"], archive["root"], member_sources(tmp, archive), epoch or 0
        )

    write_checksums(tmp)
    for path in tmp.rglob("*"):
        if path.is_file():
            path.chmod(READ_ONLY)


def write_checksums(release_dir: Path) -> None:
    """``checksums.txt`` in ``sha256sum`` format for every other file, sorted by path."""
    lines = []
    for path in sorted(release_dir.rglob("*")):
        if path.is_file() and path != release_dir / CHECKSUMS_FILE:
            lines.append(f"{sha256_file(path)}  {path.relative_to(release_dir).as_posix()}\n")
    (release_dir / CHECKSUMS_FILE).write_text("".join(lines), encoding="utf-8")


def _differences(built: Path, existing: Path) -> list[str]:
    new, old = snapshot(built), snapshot(existing)
    problems = [f"missing in existing: {name}" for name in sorted(new.keys() - old.keys())]
    problems += [f"unexpected in existing: {name}" for name in sorted(old.keys() - new.keys())]
    problems += [
        f"bytes differ: {name}"
        for name in sorted(new.keys() & old.keys())
        if new[name] != old[name]
    ]
    return problems


def build_release(
    spec: ReleaseSpec,
    version: str,
    *,
    root: Path,
    out_dir: Path,
    allow_unversioned: bool = False,
    check_bundle: bool = True,
    check: bool = False,
) -> Path:
    """Build ``spec`` as release ``version`` into ``out_dir`` (see the module docstring).

    ``out_dir`` missing: the release is built next to it and moved into place. ``out_dir``
    present: the rebuild must be byte-identical (no-op) or this raises; nothing is overwritten.
    ``check=True`` never creates ``out_dir``: it requires it to exist and be identical.
    """
    validate_release_version(spec.model_version, version)
    root, out_dir = Path(root), Path(out_dir)
    git = git_state(root)
    if not allow_unversioned:
        if git.commit is None:
            raise ReleaseError("not a git checkout with a commit; use --allow-unversioned")
        if git.dirty is not False:
            raise ReleaseError("git working tree is dirty; commit first or use --allow-unversioned")
    if check and not out_dir.is_dir():
        raise ReleaseError(f"--check needs an existing release directory: {out_dir}")
    bundle = _check_sources(spec, root)
    if check_bundle and spec.bundle_check is not None:
        _run_bundle_check(spec, root)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / f".{out_dir.name}.building-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        _assemble(spec, version, root, tmp, git, bundle)
        if out_dir.exists():
            problems = _differences(tmp, out_dir)
            if problems:
                raise ReleaseError(
                    f"{out_dir} differs from a fresh build (not overwriting): "
                    + "; ".join(problems)
                )
        else:
            tmp.rename(out_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_dir
