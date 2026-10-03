"""Publish a verified release to Hugging Face, GitHub Releases and/or Kaggle.

Usage::

    uv run python scripts/publish.py {huggingface,github,kaggle,all} --release DIR [--dry-run]

The release is verified first (exit 1 on failure). ``--dry-run`` stages the exact files, prints
the plan and writes ``<provider>.dry-run.json`` receipts, but never contacts a provider. Without
``--dry-run`` credentials and CLIs must be present, otherwise the command exits 1 before
executing anything. Receipts land in ``dist/publications/<model>/<version>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gidi.publish import ALL_PROVIDERS, PROVIDERS, publish
from gidi.publish.common import PublishError
from gidi.release.manifest import ReleaseError
from gidi.release.spec import load_spec

DEFAULT_CONFIG = Path("releases/providers.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("provider", choices=[*PROVIDERS, "all"], help=f"all = {ALL_PROVIDERS}")
    parser.add_argument("--release", type=Path, required=True, help="release directory")
    parser.add_argument("--dry-run", action="store_true", help="stage and plan only")
    parser.add_argument(
        "--confirm-publish",
        action="store_true",
        help="required for a real publication (without --dry-run)",
    )
    parser.add_argument("--staging-root", type=Path, default=Path("dist/staging"))
    parser.add_argument("--receipts-root", type=Path, default=Path("dist/publications"))
    parser.add_argument("--providers-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spec", type=Path, help="release spec (default releases/<model>.json)")
    parser.add_argument("--no-smoke", action="store_true", help="skip ONNX/smoke verify checks")
    parser.add_argument("--hf-repo-id", help="override huggingface repo_id")
    parser.add_argument("--github-repo", help="override github repo (OWNER/NAME)")
    parser.add_argument("--kaggle-owner", help="override kaggle owner")
    parser.add_argument("--kaggle-model-slug", help="override kaggle model_slug")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "huggingface": {"repo_id": args.hf_repo_id},
        "github": {"repo": args.github_repo},
        "kaggle": {"owner": args.kaggle_owner, "model_slug": args.kaggle_model_slug},
    }
    try:
        publish(
            args.provider,
            args.release,
            providers_config=args.providers_config,
            dry_run=args.dry_run,
            confirm=args.confirm_publish,
            staging_root=args.staging_root,
            receipts_root=args.receipts_root,
            smoke=not args.no_smoke,
            spec=load_spec(args.spec) if args.spec else None,
            overrides=overrides,
        )
    except (PublishError, ReleaseError) as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
