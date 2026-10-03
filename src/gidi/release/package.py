"""Deterministic ``tar.gz`` archives of a release.

Same files in, same bytes out: members are sorted, owner/mode/mtime are fixed, the gzip header
carries no mtime and no file name, and the compression level is pinned. An archive is a loadable
runtime bundle (``GidiPredictor.from_bundle(<root>)``) whose member bytes equal the release files
the manifest maps them to.
"""

from __future__ import annotations

import calendar
import gzip
import hashlib
import tarfile
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from gidi.inference.bundle import sha256_file
from gidi.release.manifest import ARCHIVES_DIR, ReleaseError, load_manifest

FILE_MODE = 0o644
DIR_MODE = 0o755
COMPRESS_LEVEL = 9
_CHUNK = 1 << 20


def archive_root(model_version: str, release_version: str, variant: str) -> str:
    return f"{model_version}-{release_version}-{variant}"


def archive_path(model_version: str, release_version: str, variant: str) -> str:
    return f"{ARCHIVES_DIR}/{archive_root(model_version, release_version, variant)}.tar.gz"


def write_archive(dest: Path, root_name: str, members: Mapping[str, Path], mtime: int) -> None:
    """Write ``dest`` as a deterministic tar.gz: ``<root_name>/`` holding ``members`` by name."""
    with (
        dest.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=COMPRESS_LEVEL, mtime=0
        ) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar,
    ):
        root = tarfile.TarInfo(root_name)
        root.type = tarfile.DIRTYPE
        root.mode = DIR_MODE
        root.mtime = mtime
        tar.addfile(root)
        for name in sorted(members):
            source = members[name]
            info = tarfile.TarInfo(f"{root_name}/{name}")
            info.size = source.stat().st_size
            info.mode = FILE_MODE
            info.mtime = mtime
            with source.open("rb") as handle:
                tar.addfile(info, handle)


def epoch_from_manifest(manifest: Mapping[str, Any]) -> int:
    """The archive mtime the build used: the manifest's ``build_timestamp`` as epoch, else 0."""
    timestamp = manifest["build_timestamp"]
    if timestamp is None:
        return 0
    return calendar.timegm(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ"))


def member_sources(release_dir: Path, archive: Mapping[str, Any]) -> dict[str, Path]:
    """``{member name: release file}`` for one manifest archive entry."""
    return {name: release_dir / target for name, target in archive["members"].items()}


def _hash_stream(handle: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(_CHUNK):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _gzip_header_problems(path: Path) -> Iterator[str]:
    with path.open("rb") as handle:
        header = handle.read(10)
    if len(header) < 10 or header[:2] != b"\x1f\x8b" or header[2] != 8:
        yield "not a gzip stream"
        return
    if header[3] != 0:
        yield "gzip header has optional fields (file name/comment/extra)"
    if header[4:8] != b"\0\0\0\0":
        yield "gzip header mtime is not 0"


def inspect_archive(
    path: Path, release_dir: Path, archive: Mapping[str, Any], mtime: int
) -> list[str]:
    """Problems of an archive against its manifest entry and the release files; [] if exact."""
    if not path.is_file():
        return [f"{path.name}: missing"]
    problems = [f"{path.name}: {item}" for item in _gzip_header_problems(path)]
    root = archive["root"]
    sources = member_sources(release_dir, archive)
    expected_names = [root, *(f"{root}/{name}" for name in sorted(sources))]
    try:
        with tarfile.open(path, "r:gz") as tar:
            entries = tar.getmembers()
            names = [entry.name for entry in entries]
            if names != expected_names:
                return [*problems, f"{path.name}: members {names} != expected {expected_names}"]
            for entry in entries:
                label = f"{path.name}:{entry.name}"
                is_root = entry.name == root
                if entry.isdir() != is_root or (not is_root and not entry.isreg()):
                    problems.append(f"{label}: unexpected entry type")
                    continue
                expected_mode = DIR_MODE if is_root else FILE_MODE
                if entry.mode != expected_mode:
                    problems.append(f"{label}: mode {entry.mode:o} != {expected_mode:o}")
                if (entry.uid, entry.gid, entry.uname, entry.gname) != (0, 0, "", ""):
                    problems.append(f"{label}: owner is not normalized to 0/0/''")
                if entry.mtime != mtime:
                    problems.append(f"{label}: mtime {entry.mtime} != {mtime}")
                if is_root:
                    continue
                source = sources[entry.name.removeprefix(f"{root}/")]
                if not source.is_file():
                    problems.append(f"{label}: source {source.name} missing in release")
                    continue
                handle = tar.extractfile(entry)
                if handle is None:
                    problems.append(f"{label}: unreadable member")
                    continue
                with handle:
                    digest, size = _hash_stream(handle)
                if size != source.stat().st_size or digest != sha256_file(source):
                    problems.append(f"{label}: bytes differ from release file")
    except (tarfile.TarError, OSError, EOFError) as error:
        problems.append(f"{path.name}: cannot read archive: {error}")
    return problems


def extract_archive(path: Path, dest: Path, root: str) -> Path:
    """Extract ``path`` below ``dest`` (safe ``data`` filter) and return the bundle directory."""
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    bundle = dest / root
    if not bundle.is_dir():
        raise ReleaseError(f"{path.name} has no top directory {root!r}")
    return bundle


def check_archives(release_dir: str | Path) -> list[str]:
    """Re-create every archive of a release in a temp dir; list those whose bytes differ."""
    release_dir = Path(release_dir)
    manifest = load_manifest(release_dir)
    mtime = epoch_from_manifest(manifest)
    problems = []
    with tempfile.TemporaryDirectory(prefix="gidi-package-check-") as tmp:
        for archive in manifest["archives"]:
            existing = release_dir / archive["path"]
            if not existing.is_file():
                problems.append(f"{archive['path']}: missing")
                continue
            rebuilt = Path(tmp) / Path(archive["path"]).name
            sources = member_sources(release_dir, archive)
            missing = [
                str(path.relative_to(release_dir))
                for path in sources.values()
                if not path.is_file()
            ]
            if missing:
                problems.append(f"{archive['path']}: release files missing: {missing}")
                continue
            write_archive(rebuilt, archive["root"], sources, mtime)
            if sha256_file(rebuilt) != sha256_file(existing):
                problems.append(f"{archive['path']}: differs from a fresh deterministic rebuild")
    return problems
