"""Verify a release directory without trusting anything in it.

Every check re-derives its answer from the bytes on disk: file hashes against the manifest and
``checksums.txt``, the file set against the manifest (nothing unlisted, nothing forbidden),
versions against each other, artifacts against the committed spec's pins, archives against the
release files, and (with ``smoke``) both ONNX variants loaded from the extracted archives and
run on the spec's smoke notes. A failing check never stops the others unless later checks depend
on it.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gidi.inference.bundle import BundleError, load_config, sha256_file
from gidi.release.build import readme_front_matter_problems, runtime_from_config
from gidi.release.manifest import (
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    README_FILE,
    Manifest,
    ReleaseError,
)
from gidi.release.package import (
    archive_path,
    archive_root,
    epoch_from_manifest,
    extract_archive,
    inspect_archive,
)
from gidi.release.spec import (
    ReleaseSpec,
    default_spec_path,
    load_spec,
    repo_root,
    validate_release_version,
)

FORBIDDEN_PATTERNS = (
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.jsonl",
    "*.db",
    "*.log",
    "*.pyc",
    "*.tmp",
    ".DS_Store",
    "__pycache__",
)
ABSOLUTE_PATH_MARKERS = ("/Users/", "/home/", "/root/", "C:\\")
TEXT_SUFFIXES = (".json", ".md", ".txt")
_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class VerifyReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.ok]

    def add(self, name: str, problems: list[str], ok_detail: str = "") -> bool:
        """Record a check that passes iff ``problems`` is empty."""
        if problems:
            self.checks.append(Check(name, False, "; ".join(problems)))
        else:
            self.checks.append(Check(name, True, ok_detail))
        return not problems


def _is_forbidden(relative: str) -> bool:
    parts = relative.split("/")
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in FORBIDDEN_PATTERNS)


def _scan_files(release_dir: Path) -> tuple[list[str], list[str]]:
    """(regular files, non-regular entries) below ``release_dir`` as POSIX relative paths."""
    regular, other = [], []
    for current, dirs, names in os.walk(release_dir, followlinks=False):
        base = Path(current)
        for name in [*dirs, *names]:
            path = base / name
            relative = path.relative_to(release_dir).as_posix()
            if path.is_symlink():
                other.append(relative)
            elif path.is_file():
                regular.append(relative)
            elif not path.is_dir():
                other.append(relative)
    return sorted(regular), sorted(other)


def _file_problems(release_dir: Path, manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(missing files, unlisted/forbidden files) against the manifest."""
    expected = {item["path"] for item in manifest["artifacts"]}
    expected |= {MANIFEST_FILE, CHECKSUMS_FILE}
    expected |= {item["path"] for item in manifest["archives"]}
    regular, other = _scan_files(release_dir)
    missing = [f"missing {path}" for path in sorted(expected - set(regular))]
    extra = []
    for path in [*regular, *other]:
        if path in expected and path not in other:
            continue
        if path in other:
            extra.append(f"non-regular file {path}")
        elif _is_forbidden(path):
            extra.append(f"forbidden file {path}")
        else:
            extra.append(f"unlisted file {path}")
    return missing, extra


def _hash_problems(release_dir: Path, manifest: dict[str, Any]) -> list[str]:
    problems = []
    for item in manifest["artifacts"]:
        path = release_dir / item["path"]
        if not path.is_file():
            continue
        if path.stat().st_size != item["size_bytes"]:
            problems.append(f"{item['path']}: size {path.stat().st_size} != {item['size_bytes']}")
        elif sha256_file(path) != item["sha256"]:
            problems.append(f"{item['path']}: sha256 differs from manifest")
    return problems


def _checksum_problems(release_dir: Path, manifest: dict[str, Any]) -> list[str]:
    path = release_dir / CHECKSUMS_FILE
    if not path.is_file():
        return [f"missing {CHECKSUMS_FILE}"]
    entries: dict[str, str] = {}
    order = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            return [f"{CHECKSUMS_FILE}:{number}: malformed line"]
        entries[match[2]] = match[1]
        order.append(match[2])
    problems = []
    if order != sorted(set(order)):
        problems.append("entries are not unique and sorted by path")
    regular, _ = _scan_files(release_dir)
    expected = {name for name in regular if name != CHECKSUMS_FILE}
    for name in sorted(expected - set(entries)):
        problems.append(f"{name} not listed")
    for name in sorted(set(entries) - expected):
        problems.append(f"{name} listed but absent")
    for name in sorted(expected & set(entries)):
        if sha256_file(release_dir / name) != entries[name]:
            problems.append(f"{name}: sha256 differs")
    listed_in_manifest = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    for name, digest in listed_in_manifest.items():
        if name in entries and entries[name] != digest:
            problems.append(f"{name}: checksums.txt disagrees with manifest")
    return problems


def _version_problems(release_dir: Path, manifest: dict[str, Any]) -> list[str]:
    model, version = manifest["model_version"], manifest["release_version"]
    problems = []
    try:
        validate_release_version(model, version)
    except ReleaseError as error:
        problems.append(str(error))
    if release_dir.name != version:
        problems.append(f"directory name {release_dir.name!r} != release_version {version!r}")
    if release_dir.parent.name != model:
        problems.append(f"parent directory {release_dir.parent.name!r} != model_version {model!r}")
    configs = [item for item in manifest["artifacts"] if item["kind"] == "runtime-config"]
    if len(configs) != 1:
        problems.append(f"expected exactly one runtime-config artifact, found {len(configs)}")
    else:
        try:
            config = json.loads((release_dir / configs[0]["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"cannot read {configs[0]['path']}: {error}")
        else:
            if config.get("model_version") != model:
                problems.append(
                    f"{configs[0]['path']} model_version {config.get('model_version')!r} "
                    f"!= manifest {model!r}"
                )
            if manifest["runtime"] != runtime_from_config(config):
                problems.append("manifest runtime differs from runtime config.json")
    for item in manifest["archives"]:
        if item["path"] != archive_path(model, version, item["variant"]):
            problems.append(f"archive path {item['path']} does not match model/version/variant")
        if item["root"] != archive_root(model, version, item["variant"]):
            problems.append(f"archive root {item['root']} does not match model/version/variant")
    return problems


def _config_problems(release_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """CRF/value-head metadata parses with the runtime's own loader."""
    for item in manifest["artifacts"]:
        if item["kind"] == "runtime-config":
            with tempfile.TemporaryDirectory(prefix="gidi-config-") as tmp:
                (Path(tmp) / "config.json").write_bytes((release_dir / item["path"]).read_bytes())
                try:
                    load_config(tmp)
                except (BundleError, OSError) as error:
                    return [f"runtime config rejected by load_config: {error}"]
            return []
    return ["no runtime-config artifact"]


def _spec_problems(spec: ReleaseSpec, manifest: dict[str, Any]) -> list[str]:
    problems = []
    if spec.model_version != manifest["model_version"]:
        problems.append(
            f"spec is for {spec.model_version}, release for {manifest['model_version']}"
        )
    pinned = {
        item.path: {
            "path": item.path,
            "kind": item.kind,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in spec.files
    }
    actual = {item["path"]: item for item in manifest["artifacts"] if item["kind"] != "readme"}
    for path in sorted(pinned.keys() - actual.keys()):
        problems.append(f"spec file {path} absent from manifest")
    for path in sorted(actual.keys() - pinned.keys()):
        problems.append(f"manifest artifact {path} is not in the spec")
    for path in sorted(pinned.keys() & actual.keys()):
        if pinned[path] != actual[path]:
            problems.append(f"{path}: differs from the spec pin")
    source = manifest["source"]
    expected_source = {
        "experiment": spec.experiment,
        "seed": spec.seed,
        "experiment_protocol_sha256": spec.experiment_protocol[1],
        "deployment_protocol_sha256": spec.deployment_protocol[1],
        "bundle_manifest_sha256": spec.bundle_manifest_sha256,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "spec_sha256": spec.sha256,
    }
    for key, value in expected_source.items():
        if source[key] != value:
            problems.append(f"manifest source.{key} differs from the spec")
    variants = {variant.name: variant.members for variant in spec.variants}
    archives = {item["variant"]: item["members"] for item in manifest["archives"]}
    if archives != variants:
        problems.append("manifest archives differ from the spec variants")
    return problems


def _readme_problems(release_dir: Path, manifest: dict[str, Any]) -> list[str]:
    path = release_dir / README_FILE
    if not path.is_file():
        return [f"missing {README_FILE}"]
    license_file = any(item["kind"] == "license" for item in manifest["artifacts"])
    return readme_front_matter_problems(path.read_text(encoding="utf-8"), license_file=license_file)


def _absolute_path_problems(release_dir: Path, root: Path) -> list[str]:
    markers = {*ABSOLUTE_PATH_MARKERS, str(root), str(root.resolve())}
    problems = []
    regular, _ = _scan_files(release_dir)
    for name in regular:
        if not name.endswith(TEXT_SUFFIXES):
            continue
        text = (release_dir / name).read_text(encoding="utf-8", errors="replace")
        problems += [f"{name} contains {marker!r}" for marker in sorted(markers) if marker in text]
    return problems


def _archive_problems(release_dir: Path, manifest: dict[str, Any]) -> list[str]:
    mtime = epoch_from_manifest(manifest)
    problems: list[str] = []
    for archive in manifest["archives"]:
        problems += inspect_archive(release_dir / archive["path"], release_dir, archive, mtime)
    return problems


def _smoke(report: VerifyReport, release_dir: Path, manifest: dict[str, Any], spec: ReleaseSpec):
    import onnxruntime

    from gidi.inference.predictor import GidiPredictor

    runtime = manifest["runtime"]
    with tempfile.TemporaryDirectory(prefix="gidi-release-verify-") as tmp:
        for archive in manifest["archives"]:
            variant = archive["variant"]
            try:
                bundle = extract_archive(
                    release_dir / archive["path"], Path(tmp) / variant, archive["root"]
                )
                models = [name for name in archive["members"] if name.endswith(".onnx")]
                if len(models) != 1:
                    raise ReleaseError(f"expected one .onnx member, found {models}")
                session = onnxruntime.InferenceSession(
                    str(bundle / models[0]), providers=["CPUExecutionProvider"]
                )
                inputs = [node.name for node in session.get_inputs()]
                outputs = [node.name for node in session.get_outputs()]
                mismatches = []
                if inputs != runtime["onnx_inputs"]:
                    mismatches.append(f"inputs {inputs} != {runtime['onnx_inputs']}")
                if outputs != runtime["onnx_outputs"]:
                    mismatches.append(f"outputs {outputs} != {runtime['onnx_outputs']}")
                del session
                report.add(f"onnx-load:{variant}", mismatches, f"{models[0]} loads")
                predictor = GidiPredictor.from_bundle(bundle, model_path=bundle / models[0])
                problems = []
                for case in spec.smoke:
                    got = predictor.predict(case.text).to_dict()
                    if list(got) != runtime["prediction_keys"]:
                        problems.append(f"{case.text!r}: output keys {list(got)}")
                    if got["model_version"] != manifest["model_version"]:
                        problems.append(f"{case.text!r}: model_version {got['model_version']!r}")
                    problems += [
                        f"{case.text!r}: {key} {got[key]!r} != {want!r}"
                        for key, want in case.expected.items()
                        if got[key] != want
                    ]
                report.add(f"smoke:{variant}", problems, f"{len(spec.smoke)} notes match")
            except Exception as error:  # a broken archive or model must fail, not crash, the report
                report.checks.append(
                    Check(f"smoke:{variant}", False, f"{type(error).__name__}: {error}")
                )


def verify_release(
    release_dir: str | Path,
    *,
    spec: ReleaseSpec | None = None,
    root: Path | None = None,
    smoke: bool = True,
) -> VerifyReport:
    """Run every release check; ``smoke=False`` skips ONNX loading and inference."""
    release_dir = Path(release_dir)
    root = repo_root() if root is None else Path(root)
    report = VerifyReport()
    try:
        manifest = Manifest.parse(
            (release_dir / MANIFEST_FILE).read_text(encoding="utf-8")
        ).to_dict()
    except (OSError, ReleaseError) as error:
        report.add("manifest", [str(error)])
        return report
    report.add("manifest", [])

    missing, extra = _file_problems(release_dir, manifest)
    report.add("files-present", missing, f"{len(manifest['artifacts'])} artifacts")
    report.add("files-unlisted", extra, "no unlisted or forbidden files")
    report.add("artifact-hashes", _hash_problems(release_dir, manifest), "sha256 and sizes match")
    report.add("checksums", _checksum_problems(release_dir, manifest), f"{CHECKSUMS_FILE} matches")
    report.add("versions", _version_problems(release_dir, manifest), "model/release versions agree")
    report.add("runtime-config", _config_problems(release_dir, manifest), "load_config accepts it")
    report.add("readme", _readme_problems(release_dir, manifest), "front matter ok")

    if spec is None:
        spec_path = default_spec_path(manifest["model_version"], root)
        try:
            spec = load_spec(spec_path)
        except ReleaseError as error:
            report.add("spec", [str(error)])
    if spec is not None:
        report.add("spec", _spec_problems(spec, manifest), f"matches {spec.path.name}")

    report.add("no-absolute-paths", _absolute_path_problems(release_dir, root), "none found")
    report.add("archives", _archive_problems(release_dir, manifest), "members and metadata exact")
    if smoke and spec is not None and not missing:
        _smoke(report, release_dir, manifest, spec)
    elif smoke:
        report.add("smoke", ["skipped: needs the spec and a complete release"])
    return report
