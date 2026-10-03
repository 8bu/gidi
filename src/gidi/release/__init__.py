"""Deterministic release pipeline: spec -> release directory -> verification.

The pipeline copies hash-pinned model files and never exports, quantizes or trains.
"""

from gidi.release.build import build_release
from gidi.release.manifest import (
    Manifest,
    ReleaseError,
    load_manifest,
    manifest_sha256,
    release_dir_for,
    snapshot,
)
from gidi.release.package import check_archives, write_archive
from gidi.release.spec import ReleaseSpec, default_spec_path, load_spec, validate_release_version
from gidi.release.verify import Check, VerifyReport, verify_release

__all__ = [
    "Check",
    "Manifest",
    "ReleaseError",
    "ReleaseSpec",
    "VerifyReport",
    "build_release",
    "check_archives",
    "default_spec_path",
    "load_manifest",
    "load_spec",
    "manifest_sha256",
    "release_dir_for",
    "snapshot",
    "validate_release_version",
    "verify_release",
    "write_archive",
]
