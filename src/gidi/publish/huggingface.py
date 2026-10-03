"""Hugging Face Hub provider: upload the release files (minus ``archives/``) byte-identically."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gidi.publish.common import (
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    Auth,
    Plan,
    PublishError,
    env_source,
    file_source,
    plan_file,
    release_digests,
    render,
    require_keys,
)

NAME = "huggingface"
_URL_PREFIX = {"model": "", "dataset": "datasets/", "space": "spaces/"}


class HuggingFaceProvider:
    name = NAME

    def plan(
        self, release_dir: Path, manifest: dict[str, Any], config: dict[str, Any], staging_dir: Path
    ) -> Plan:
        require_keys(config, NAME, ["repo_id", "repo_type", "private", "tag"])
        repo_type = config["repo_type"]
        if repo_type not in _URL_PREFIX:
            raise PublishError(f"huggingface repo_type must be one of {sorted(_URL_PREFIX)}")
        repo_id = config["repo_id"]
        tag = render(config["tag"], manifest)
        digests = release_digests(release_dir, manifest)
        archives = {a["path"] for a in manifest.get("archives", [])}
        exclude_kinds = config.get("exclude_kinds", [])
        if not isinstance(exclude_kinds, list) or not all(
            isinstance(k, str) for k in exclude_kinds
        ):
            raise PublishError("huggingface exclude_kinds must be a list of artifact kinds")
        excluded = {a["path"] for a in manifest["artifacts"] if a["kind"] in exclude_kinds}
        paths = sorted(p for p in digests if p not in archives and p not in excluded)
        for required in ("README.md", MANIFEST_FILE, CHECKSUMS_FILE):
            if required not in paths:
                raise PublishError(f"release has no {required} to upload")
        kinds = {a["path"]: a["kind"] for a in manifest["artifacts"]}
        if not any(kinds.get(p, "").startswith("onnx-") for p in paths):
            raise PublishError("huggingface upload would contain no ONNX model")
        files = [plan_file(p, p, digests) for p in paths]
        private = bool(config["private"])
        target = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private,
            "tag": tag,
            "revision": None,
            "url": f"https://huggingface.co/{_URL_PREFIX[repo_type]}{repo_id}",
        }
        message = _commit_message(manifest)
        actions = [
            f"HfApi().create_repo(repo_id={repo_id!r}, repo_type={repo_type!r}, "
            f"private={private}, exist_ok=True)",
            f"HfApi().upload_folder(repo_id={repo_id!r}, repo_type={repo_type!r}, "
            f"folder_path={str(staging_dir)!r}, commit_message={message!r})",
            f"HfApi().create_tag(repo_id={repo_id!r}, repo_type={repo_type!r}, tag={tag!r}, "
            "revision=<commit created by upload_folder>)",
        ]
        return Plan(
            provider=NAME,
            release_dir=release_dir,
            staging_dir=staging_dir,
            model_version=manifest["model_version"],
            release_version=manifest["release_version"],
            target=target,
            files=files,
            generated={},
            actions=actions,
            notes_source="README.md from the release, uploaded unchanged as the model card",
            details={
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "tag": tag,
                "commit_message": message,
                "url": target["url"],
                "exclude_kinds": list(exclude_kinds),
            },
        )

    def auth(self, env: Mapping[str, str]) -> Auth:
        source = env_source(env, "HF_TOKEN")
        if source is None:
            hf_home = env.get("HF_HOME")
            if hf_home:
                source = file_source("$HF_HOME/token", Path(hf_home) / "token")
            else:
                source = file_source(
                    "~/.cache/huggingface/token", Path.home() / ".cache" / "huggingface" / "token"
                )
        return Auth(available=source is not None, source=source)

    def preflight(self, plan: Plan) -> list[str]:
        if importlib.util.find_spec("huggingface_hub") is None:
            return ["python package huggingface_hub is not installed"]
        return []

    def execute(self, plan: Plan) -> dict[str, Any]:
        from huggingface_hub import HfApi  # lazy: dry-run must never reach the network library

        d = plan.details
        api = HfApi()  # token resolved by the library from HF_TOKEN / token file
        try:
            api.create_repo(
                repo_id=d["repo_id"], repo_type=d["repo_type"], private=d["private"], exist_ok=True
            )
            commit = api.upload_folder(
                repo_id=d["repo_id"],
                repo_type=d["repo_type"],
                folder_path=str(plan.staging_dir),
                commit_message=d["commit_message"],
            )
            revision = str(commit.oid)
            api.create_tag(
                repo_id=d["repo_id"], repo_type=d["repo_type"], tag=d["tag"], revision=revision
            )
        except Exception as exc:  # huggingface_hub raises many error types
            raise PublishError(f"huggingface upload failed: {type(exc).__name__}: {exc}") from exc
        return {
            "repo_id": d["repo_id"],
            "repo_type": d["repo_type"],
            "tag": d["tag"],
            "revision": revision,
            "url": d["url"],
        }


def _commit_message(manifest: Mapping[str, Any]) -> str:
    return f"{manifest['model_version']} {manifest['release_version']}"
