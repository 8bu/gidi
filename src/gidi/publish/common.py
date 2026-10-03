"""Shared publish machinery: provider protocol, staging, credential probes, receipts, orchestrator.

A provider turns a verified release directory into a :class:`Plan` (pure: no network, no
writes), the orchestrator stages that plan as fresh byte copies, prints it, and, unless
``dry_run``, runs ``provider.execute(plan)``. The release directory is snapshotted before and
after and must not change. Secrets are only ever probed for availability; their values are never
read into output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from gidi.inference.bundle import sha256_file
from gidi.release.manifest import (
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    load_manifest,
    manifest_sha256,
    snapshot,
)
from gidi.release.spec import ReleaseSpec
from gidi.release.verify import verify_release

RECEIPT_FORMAT = 1
# Environment variables whose values must never be printed (scrubbed from error text).
SECRET_ENV_VARS = (
    "HF_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_API_TOKEN",
)


class PublishError(Exception):
    """A publish step cannot proceed (bad config, no credentials/tools, drift, remote error)."""


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class PlanFile:
    """One file in the staging directory.

    ``path`` is relative to the staging dir. ``source`` is the release-relative path it is a
    byte copy of, or ``None`` for a generated file whose bytes are in ``Plan.generated``.
    """

    path: str
    sha256: str
    size_bytes: int
    source: str | None = None

    def to_receipt(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass
class Plan:
    provider: str
    release_dir: Path
    staging_dir: Path
    model_version: str
    release_version: str
    target: dict[str, Any]
    files: list[PlanFile]
    generated: dict[str, bytes]
    actions: list[str]
    # Human description of where the model card / release notes come from.
    notes_source: str
    # Provider-specific parameters ``execute`` needs (JSON-serialisable).
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


@dataclass(frozen=True)
class Auth:
    """Credential availability. ``source`` names where it was found, never its value."""

    available: bool
    source: str | None = None

    def to_receipt(self) -> dict[str, Any]:
        return {"available": self.available, "source": self.source}


class Provider(Protocol):
    name: str

    def plan(
        self, release_dir: Path, manifest: dict[str, Any], config: dict[str, Any], staging_dir: Path
    ) -> Plan:
        """Pure: compute what would be published. No network, no filesystem writes."""
        ...

    def auth(self, env: Mapping[str, str]) -> Auth:
        """Report credential availability and its source name (never the value)."""
        ...

    def preflight(self, plan: Plan) -> list[str]:
        """Problems that would make a real execution fail (missing CLI/library, bad config)."""
        ...

    def execute(self, plan: Plan) -> dict[str, Any]:
        """Perform the real publication from the staged files; return remote info."""
        ...


# --------------------------------------------------------------------------- config helpers


def load_providers_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PublishError(f"providers config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PublishError(f"providers config is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PublishError(f"providers config must be a JSON object: {path}")
    return data


def require_keys(config: Mapping[str, Any], provider: str, keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in config]
    if missing:
        raise PublishError(f"providers config for {provider} is missing: {', '.join(missing)}")


def render(template: str, manifest: Mapping[str, Any]) -> str:
    """Fill ``{model_version}`` / ``{release_version}`` in a providers-config template."""
    try:
        return template.format(
            model_version=manifest["model_version"], release_version=manifest["release_version"]
        )
    except (KeyError, IndexError) as exc:
        raise PublishError(f"unknown placeholder {exc} in template {template!r}") from exc


# --------------------------------------------------------------------------- release files


def release_digests(release_dir: Path, manifest: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    """``release path -> (sha256, size)`` for every file the release lists.

    Artifacts come from the manifest; ``manifest.json``, ``checksums.txt`` and the archives are
    hashed from disk (the manifest deliberately carries no hash for them).
    """
    digests = {a["path"]: (a["sha256"], int(a["size_bytes"])) for a in manifest["artifacts"]}
    extra = [MANIFEST_FILE, CHECKSUMS_FILE, *(a["path"] for a in manifest.get("archives", []))]
    for rel in extra:
        path = release_dir / rel
        if not path.is_file():
            raise PublishError(f"release file missing: {rel}")
        digests[rel] = (sha256_file(path), path.stat().st_size)
    return digests


def plan_file(source: str, path: str, digests: Mapping[str, tuple[str, int]]) -> PlanFile:
    if source not in digests:
        raise PublishError(f"release file not described by the release: {source}")
    sha, size = digests[source]
    return PlanFile(path=path, sha256=sha, size_bytes=size, source=source)


def generated_file(path: str, data: bytes) -> PlanFile:
    return PlanFile(path=path, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


# --------------------------------------------------------------------------- staging


def _copy_bytes(src: Path, dst: Path, expected_sha256: str) -> None:
    """Fresh byte copy (never a link); the copied bytes must hash to ``expected_sha256``."""
    digest = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as reader, open(dst, "xb") as writer:
        for block in iter(lambda: reader.read(1 << 20), b""):
            digest.update(block)
            writer.write(block)
    if digest.hexdigest() != expected_sha256:
        raise PublishError(f"release file changed while staging: {src}")


def wipe_dir(path: Path) -> None:
    if path.is_symlink():
        raise PublishError(f"refusing to wipe a symlinked staging dir: {path}")
    if path.exists():
        shutil.rmtree(path)


def stage(plan: Plan) -> None:
    """Wipe and recreate ``plan.staging_dir`` holding exactly the planned files."""
    release = plan.release_dir.resolve()
    staging = plan.staging_dir.resolve()
    if staging == release or release in staging.parents or staging in release.parents:
        raise PublishError(f"staging dir {plan.staging_dir} overlaps the release dir")
    wipe_dir(plan.staging_dir)
    plan.staging_dir.mkdir(parents=True)
    for item in plan.files:
        dst = plan.staging_dir / item.path
        if item.source is None:
            data = plan.generated[item.path]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        else:
            _copy_bytes(plan.release_dir / item.source, dst, item.sha256)


# --------------------------------------------------------------------------- credentials


def env_source(env: Mapping[str, str], *names: str) -> str | None:
    """First ``env:NAME`` among ``names`` that is set and non-empty (value never returned)."""
    for name in names:
        if env.get(name):
            return f"env:{name}"
    return None


def file_source(label: str, path: Path) -> str | None:
    return f"file:{label}" if path.is_file() else None


def redact(text: str, env: Mapping[str, str]) -> str:
    """Scrub known secret values out of ``text`` (for subprocess output in error messages)."""
    for name in SECRET_ENV_VARS:
        value = env.get(name)
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def run_command(
    argv: Sequence[str], env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a provider CLI (real executions only); failure raises with redacted output."""
    env = os.environ if env is None else env
    try:
        proc = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise PublishError(f"command not found: {argv[0]}") from exc
    if proc.returncode != 0:
        detail = redact((proc.stderr or proc.stdout or "").strip(), env)
        raise PublishError(f"`{shell_join(argv)}` failed (exit {proc.returncode}): {detail}")
    return proc


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


# --------------------------------------------------------------------------- receipts


def receipt_path(
    receipts_root: Path, model: str, version: str, provider: str, dry_run: bool
) -> Path:
    suffix = ".dry-run.json" if dry_run else ".json"
    return receipts_root / model / version / f"{provider}{suffix}"


def write_receipt(
    plan: Plan,
    *,
    receipts_root: Path,
    dry_run: bool,
    auth: Auth,
    remote: dict[str, Any] | None,
    manifest_digest: str,
    checksums_digest: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    target = dict(plan.target)
    if remote:
        for key in ("revision", "url"):
            if remote.get(key) is not None:
                target[key] = remote[key]
    receipt = {
        "receipt_format": RECEIPT_FORMAT,
        "provider": plan.provider,
        "dry_run": dry_run,
        "model_version": plan.model_version,
        "release_version": plan.release_version,
        "manifest_sha256": manifest_digest,
        "checksums_sha256": checksums_digest,
        "timestamp": now().isoformat(timespec="seconds"),
        "target": target,
        "remote": None if dry_run else remote,
        "files": [f.to_receipt() for f in plan.files],
        "actions": plan.actions,
        "auth": auth.to_receipt(),
    }
    path = receipt_path(
        receipts_root, plan.model_version, plan.release_version, plan.provider, dry_run
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- printing


def format_plan(plan: Plan, auth: Auth, *, dry_run: bool) -> str:
    lines = [
        f"== {plan.provider}: {'DRY RUN' if dry_run else 'PUBLISH'} "
        f"{plan.model_version} {plan.release_version} =="
    ]
    for key, value in plan.target.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    lines.append(f"  staging: {plan.staging_dir}")
    lines.append(f"  notes/model card: {plan.notes_source}")
    source = f" ({auth.source})" if auth.source else ""
    lines.append(f"  auth: {'available' if auth.available else 'NOT available'}{source}")
    lines.append(f"  files ({len(plan.files)}, {plan.total_bytes} bytes total):")
    for item in plan.files:
        lines.append(f"    {item.size_bytes:>12}  {item.sha256}  {item.path}")
    verb = "actions" if not dry_run else "actions that WOULD run"
    lines.append(f"  {verb}:")
    lines.extend(f"    - {action}" for action in plan.actions)
    return "\n".join(lines)


# --------------------------------------------------------------------------- orchestrator


def publish_release(
    providers: Sequence[Provider],
    release_dir: Path,
    *,
    configs: Mapping[str, dict[str, Any]],
    dry_run: bool,
    confirm: bool = False,
    staging_root: Path,
    receipts_root: Path,
    smoke: bool = True,
    spec: ReleaseSpec | None = None,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
) -> list[Path]:
    """verify -> snapshot -> plan+stage+print (all providers) -> [creds + execute] -> receipts.

    Raises :class:`PublishError` (or ``ReleaseError``) on any problem; nothing is staged and no
    receipt is written when the release does not verify. With several providers, every
    credential/tool check runs before the first real execution.
    """
    if not dry_run and not confirm:
        raise PublishError(
            "a real publish needs explicit confirmation (--confirm-publish); "
            "use --dry-run to only plan. Nothing was probed or executed."
        )
    env = os.environ if env is None else env
    release_dir = Path(release_dir)
    report = verify_release(release_dir, spec=spec, root=root, smoke=smoke)
    if not report.ok:
        details = "; ".join(f"{c.name}: {c.detail}" for c in report.failures)
        raise PublishError(f"release does not verify, refusing to publish: {details}")
    before = snapshot(release_dir)
    manifest = load_manifest(release_dir)
    manifest_digest = manifest_sha256(release_dir)
    checksums_digest = sha256_file(release_dir / CHECKSUMS_FILE)
    model, version = manifest["model_version"], manifest["release_version"]

    staged: list[tuple[Provider, Plan, Auth]] = []
    for provider in providers:
        if provider.name not in configs:
            raise PublishError(f"providers config has no section for {provider.name}")
        staging_dir = Path(staging_root) / provider.name / model / version
        plan = provider.plan(release_dir, manifest, configs[provider.name], staging_dir)
        stage(plan)
        auth = provider.auth(env)
        out(format_plan(plan, auth, dry_run=dry_run))
        if dry_run:
            # Local checks only (CLI on PATH, config); surfaced so a dry run shows real blockers.
            for problem in provider.preflight(plan):
                out(f"  WOULD BLOCK a real publish: {problem}")
        staged.append((provider, plan, auth))

    if not dry_run:
        problems: list[str] = []
        for provider, plan, auth in staged:
            if not auth.available:
                problems.append(
                    f"{provider.name}: no credentials found ({_AUTH_HINTS[provider.name]})"
                )
            problems.extend(f"{provider.name}: {p}" for p in provider.preflight(plan))
        if problems:
            raise PublishError("cannot publish, nothing was executed:\n  " + "\n  ".join(problems))

    receipts: list[Path] = []
    for provider, plan, auth in staged:
        remote = None
        if not dry_run:
            try:
                remote = provider.execute(plan)
            except PublishError as exc:
                raise PublishError(redact(str(exc), env)) from None
        if snapshot(release_dir) != before:
            raise PublishError(f"release directory changed while publishing {provider.name}")
        receipts.append(
            write_receipt(
                plan,
                receipts_root=Path(receipts_root),
                dry_run=dry_run,
                auth=auth,
                remote=remote,
                manifest_digest=manifest_digest,
                checksums_digest=checksums_digest,
            )
        )
        out(f"  receipt: {receipts[-1]}")
    return receipts


_AUTH_HINTS = {
    "huggingface": "set HF_TOKEN or run `hf auth login`",
    "github": "set GH_TOKEN/GITHUB_TOKEN or run `gh auth login`",
    "kaggle": "set KAGGLE_USERNAME and KAGGLE_KEY or create ~/.kaggle/kaggle.json",
}
