#!/usr/bin/env python
"""Build, verify and re-check a deterministic release of a Gidi model.

    uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0
    uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0 --check
    uv run python scripts/release.py verify --release dist/releases/gidi-finance-v2/2.0.0
    uv run python scripts/release.py package --release dist/releases/gidi-finance-v2/2.0.0 --check

``build`` copies the files pinned by ``releases/<model>.json`` (never exporting or quantizing
anything) into ``<out>/releases/<model>/<version>/``. An existing release is never overwritten;
``build --check`` rebuilds into a temp directory and requires the existing one to be
byte-identical. Nothing here publishes anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gidi.release import (  # noqa: E402
    ReleaseError,
    build_release,
    check_archives,
    default_spec_path,
    load_manifest,
    load_spec,
    release_dir_for,
    verify_release,
)


def _cmd_build(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec or default_spec_path(args.model, ROOT))
    if spec.model_version != args.model:
        raise ReleaseError(f"spec {spec.path} is for {spec.model_version}, not {args.model}")
    out_dir = release_dir_for(args.model, args.version, args.out)
    build_release(
        spec,
        args.version,
        root=ROOT,
        out_dir=out_dir,
        allow_unversioned=args.allow_unversioned,
        check=args.check,
    )
    manifest = load_manifest(out_dir)
    verb = "reproduced byte-identically" if args.check else "ready"
    print(f"release {verb}: {out_dir}")
    for item in manifest["artifacts"]:
        print(f"  {item['size_bytes']:>12}  {item['path']}")
    for item in manifest["archives"]:
        print(f"  {(out_dir / item['path']).stat().st_size:>12}  {item['path']}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec) if args.spec else None
    report = verify_release(args.release, spec=spec, root=ROOT, smoke=not args.no_smoke)
    for check in report.checks:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")
    print("verified" if report.ok else f"{len(report.failures)} check(s) failed")
    return 0 if report.ok else 1


def _cmd_package(args: argparse.Namespace) -> int:
    problems = check_archives(args.release)
    for problem in problems:
        print(f"FAIL {problem}")
    print("archives reproduce byte-identically" if not problems else "archives differ")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build (or --check) a release directory")
    build.add_argument("--model", required=True, help="model version, e.g. gidi-finance-v2")
    build.add_argument("--version", required=True, help="release version MAJOR.MINOR.PATCH[-pre]")
    build.add_argument("--out", type=Path, default=ROOT / "dist", help="dist root (default dist/)")
    build.add_argument("--spec", type=Path, help="release spec (default releases/<model>.json)")
    build.add_argument("--check", action="store_true", help="rebuild and require identical bytes")
    build.add_argument(
        "--allow-unversioned",
        action="store_true",
        help="allow a dirty tree or a checkout without git (records commit/dirty as observed)",
    )
    build.set_defaults(handler=_cmd_build)

    verify = commands.add_parser("verify", help="verify a release directory")
    verify.add_argument("--release", required=True, type=Path)
    verify.add_argument("--spec", type=Path, help="release spec (default releases/<model>.json)")
    verify.add_argument("--no-smoke", action="store_true", help="skip ONNX loading and inference")
    verify.set_defaults(handler=_cmd_verify)

    package = commands.add_parser("package", help="re-create archives and compare bytes")
    package.add_argument("--release", required=True, type=Path)
    package.add_argument("--check", action="store_true", required=True)
    package.set_defaults(handler=_cmd_package)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
