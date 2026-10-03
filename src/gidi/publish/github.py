"""GitHub Releases provider: archives + checksums.txt + manifest.json via ``gh release create``."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from gidi.publish.common import (
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    Auth,
    Plan,
    PublishError,
    env_source,
    generated_file,
    plan_file,
    release_digests,
    render,
    require_keys,
    run_command,
    shell_join,
)

NAME = "github"
ASSETS_DIR = "assets"
NOTES_FILE = "release-notes.md"


class GitHubProvider:
    name = NAME

    def plan(
        self, release_dir: Path, manifest: dict[str, Any], config: dict[str, Any], staging_dir: Path
    ) -> Plan:
        require_keys(config, NAME, ["repo", "tag", "title", "draft"])
        repo = config["repo"]
        tag = render(config["tag"], manifest)
        title = render(config["title"], manifest)
        draft = bool(config["draft"])
        commit = (manifest.get("git") or {}).get("commit")
        digests = release_digests(release_dir, manifest)

        sources = [a["path"] for a in manifest.get("archives", [])] + [
            CHECKSUMS_FILE,
            MANIFEST_FILE,
        ]
        if len(manifest.get("archives", [])) != 2:
            raise PublishError("GitHub release expects exactly the int8 and fp32 archives")
        files = [
            plan_file(src, f"{ASSETS_DIR}/{PurePosixPath(src).name}", digests) for src in sources
        ]
        notes = _release_notes(manifest, {src: digests[src][0] for src in sources})
        generated = {NOTES_FILE: notes.encode("utf-8")}
        files.append(generated_file(NOTES_FILE, generated[NOTES_FILE]))

        details = {
            "repo": repo,
            "tag": tag,
            "title": title,
            "draft": draft,
            "target_commit": commit,
            "assets": [f.path for f in files if f.path.startswith(f"{ASSETS_DIR}/")],
            "notes_file": NOTES_FILE,
        }
        plan = Plan(
            provider=NAME,
            release_dir=release_dir,
            staging_dir=staging_dir,
            model_version=manifest["model_version"],
            release_version=manifest["release_version"],
            target={
                "repo": repo,
                "tag": tag,
                "title": title,
                "draft": draft,
                "target_commit": commit,
                "revision": commit,
                "url": f"https://github.com/{repo}/releases/tag/{tag}",
            },
            files=files,
            generated=generated,
            actions=[],
            notes_source=f"{NOTES_FILE} generated from manifest.json (staged, not a release file)",
            details=details,
        )
        plan.actions.append(shell_join(create_command(plan)))
        return plan

    def auth(self, env: Mapping[str, str]) -> Auth:
        source = env_source(env, "GH_TOKEN", "GITHUB_TOKEN")
        if source is None:
            # Local status probe only: exit code, output discarded; never contacts a release API.
            try:
                probe = subprocess.run(
                    ["gh", "auth", "token"],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                probe = None
            if probe is not None and probe.returncode == 0:
                source = "gh auth token"
        return Auth(available=source is not None, source=source)

    def preflight(self, plan: Plan) -> list[str]:
        if shutil.which("gh") is None:
            return ["`gh` CLI is not installed or not on PATH"]
        return []

    def execute(self, plan: Plan) -> dict[str, Any]:
        proc = run_command(create_command(plan))
        url = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
        return {
            "repo": plan.details["repo"],
            "tag": plan.details["tag"],
            "draft": plan.details["draft"],
            "revision": plan.details["target_commit"],
            "url": url or plan.target["url"],
        }


def create_command(plan: Plan) -> list[str]:
    """The exact ``gh release create`` invocation (printed in dry-run, run on execute)."""
    d = plan.details
    argv = ["gh", "release", "create", d["tag"]]
    argv += [str(plan.staging_dir / asset) for asset in d["assets"]]
    argv += [
        "--repo",
        d["repo"],
        "--title",
        d["title"],
        "--notes-file",
        str(plan.staging_dir / d["notes_file"]),
    ]
    if d["draft"]:
        argv.append("--draft")
    if d["target_commit"]:
        argv += ["--target", d["target_commit"]]
    return argv


def _release_notes(manifest: Mapping[str, Any], archive_sha256: Mapping[str, str]) -> str:
    git = manifest.get("git") or {}
    lines = [
        f"# {manifest['model_version']} {manifest['release_version']}",
        "",
        "Deterministic release of the Gidi on-device classifier for short Vietnamese "
        "personal-finance notes. Archives are loadable runtime bundles "
        "(`GidiPredictor.from_bundle`).",
        "",
        f"- commit: {git.get('commit') or 'unknown'}",
        f"- experiment: {manifest['source']['experiment']} (seed {manifest['source']['seed']})",
        f"- build timestamp: {manifest.get('build_timestamp') or 'unknown'}",
        "",
        "## Assets",
        "",
        "| file | sha256 |",
        "| --- | --- |",
    ]
    for path, sha in archive_sha256.items():
        lines.append(f"| `{PurePosixPath(path).name}` | `{sha}` |")
    lines += [
        "",
        "Extract an archive and check its members against `manifest.json`, or verify the "
        "assets against `checksums.txt` (paths in it are relative to the release directory).",
        "",
    ]
    return "\n".join(lines)
