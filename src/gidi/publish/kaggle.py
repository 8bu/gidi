"""Kaggle Models provider (optional): one model, one instance per archive variant."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gidi.publish.common import (
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

NAME = "kaggle"
MODEL_DIR = "model"
INSTANCES_DIR = "instances"
MODEL_METADATA = "model-metadata.json"
INSTANCE_METADATA = "model-instance-metadata.json"


class KaggleProvider:
    name = NAME

    def plan(
        self, release_dir: Path, manifest: dict[str, Any], config: dict[str, Any], staging_dir: Path
    ) -> Plan:
        require_keys(config, NAME, ["owner", "model_slug", "framework", "variations"])
        owner, slug, framework = config["owner"], config["model_slug"], config["framework"]
        license_name = config.get("license")
        private = bool(config.get("private", True))
        model, version = manifest["model_version"], manifest["release_version"]
        digests = release_digests(release_dir, manifest)
        archives = manifest.get("archives", [])
        if not archives:
            raise PublishError("release has no archives to derive Kaggle variants from")

        files = []
        generated: dict[str, bytes] = {}
        variants: list[dict[str, Any]] = []
        commit = (manifest.get("git") or {}).get("commit")
        notes = f"{model} {version}" + (f" ({commit[:12]})" if commit else "")

        model_meta = {
            "ownerSlug": owner,
            "title": slug,
            "slug": slug,
            "subtitle": "Tiny on-device classifier for short Vietnamese personal-finance notes",
            "isPrivate": private,
            "description": _description(manifest),
            "provenanceSources": "",
        }
        model_meta_path = f"{MODEL_DIR}/{MODEL_METADATA}"
        generated[model_meta_path] = _dump(model_meta)
        files.append(generated_file(model_meta_path, generated[model_meta_path]))

        for archive in archives:
            variant = archive["variant"]
            if variant not in config["variations"]:
                raise PublishError(f"kaggle variations has no entry for variant {variant!r}")
            variation = render(config["variations"][variant], manifest)
            base = f"{INSTANCES_DIR}/{variant}"
            for member, release_path in archive["members"].items():
                files.append(plan_file(release_path, f"{base}/{member}", digests))
            instance_meta = {
                "ownerSlug": owner,
                "modelSlug": slug,
                "instanceSlug": variation,
                "framework": framework,
                "overview": f"{model} {version} ({variant}) ONNX runtime bundle",
                "usage": "Load with gidi.inference.GidiPredictor.from_bundle(<extracted dir>); "
                "see README.md in this instance.",
                "licenseName": license_name,
                "fineTunable": False,
                "trainingData": [],
                "modelInstanceType": "Unspecified",
            }
            meta_path = f"{base}/{INSTANCE_METADATA}"
            generated[meta_path] = _dump(instance_meta)
            files.append(generated_file(meta_path, generated[meta_path]))
            variants.append({"variant": variant, "variation": variation, "dir": base})

        plan = Plan(
            provider=NAME,
            release_dir=release_dir,
            staging_dir=staging_dir,
            model_version=model,
            release_version=version,
            target={
                "model": f"{owner}/{slug}",
                "framework": framework,
                "variations": {v["variant"]: v["variation"] for v in variants},
                "private": private,
                "license": license_name,
                "revision": None,
                "url": f"https://www.kaggle.com/models/{owner}/{slug}",
            },
            files=files,
            generated=generated,
            actions=[],
            notes_source=f"{MODEL_METADATA} / {INSTANCE_METADATA} generated from manifest.json; "
            "instance files are the archive members",
            details={
                "owner": owner,
                "slug": slug,
                "framework": framework,
                "license": license_name,
                "notes": notes,
                "model_dir": MODEL_DIR,
                "variants": variants,
            },
        )
        plan.actions.extend(_describe_actions(plan))
        return plan

    def auth(self, env: Mapping[str, str]) -> Auth:
        if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
            return Auth(True, "env:KAGGLE_USERNAME+KAGGLE_KEY")
        config_dir = env.get("KAGGLE_CONFIG_DIR")
        if config_dir:
            found, label = Path(config_dir) / "kaggle.json", "file:$KAGGLE_CONFIG_DIR/kaggle.json"
        else:
            found, label = Path.home() / ".kaggle" / "kaggle.json", "file:~/.kaggle/kaggle.json"
        if found.is_file():
            return Auth(True, label)
        return Auth(False, env_source(env, "KAGGLE_API_TOKEN"))

    def preflight(self, plan: Plan) -> list[str]:
        problems = []
        if shutil.which("kaggle") is None:
            problems.append("`kaggle` CLI is not installed or not on PATH")
        if not plan.details["license"]:
            problems.append("kaggle license is not set in providers config (licenseName required)")
        return problems

    def execute(self, plan: Plan) -> dict[str, Any]:
        d = plan.details
        model_ref = f"{d['owner']}/{d['slug']}"
        created_model = False
        if not _exists(["kaggle", "models", "get", model_ref]):
            run_command(_model_create(plan))
            created_model = True
        instances = {}
        for variant in d["variants"]:
            ref = _instance_ref(plan, variant)
            if _exists(["kaggle", "models", "instances", "get", ref]):
                run_command(_version_create(plan, variant))
                instances[variant["variant"]] = {"ref": ref, "action": "new-version"}
            else:
                run_command(_instance_create(plan, variant))
                instances[variant["variant"]] = {"ref": ref, "action": "created"}
        return {
            "model": model_ref,
            "created_model": created_model,
            "instances": instances,
            "url": plan.target["url"],
        }


def _dump(data: Mapping[str, Any]) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _description(manifest: Mapping[str, Any]) -> str:
    return (
        f"{manifest['model_version']} {manifest['release_version']}: Gidi dual-encoder "
        "classifier (transaction type, counterparty and value spans) for short Vietnamese "
        "personal-finance notes, exported to ONNX (int8 and fp32 variants)."
    )


def _instance_ref(plan: Plan, variant: Mapping[str, Any]) -> str:
    d = plan.details
    return f"{d['owner']}/{d['slug']}/{d['framework']}/{variant['variation']}"


def _model_create(plan: Plan) -> list[str]:
    return ["kaggle", "models", "create", "-p", str(plan.staging_dir / plan.details["model_dir"])]


def _instance_create(plan: Plan, variant: Mapping[str, Any]) -> list[str]:
    return ["kaggle", "models", "instances", "create", "-p", str(plan.staging_dir / variant["dir"])]


def _version_create(plan: Plan, variant: Mapping[str, Any]) -> list[str]:
    return [
        "kaggle",
        "models",
        "instances",
        "versions",
        "create",
        _instance_ref(plan, variant),
        "-p",
        str(plan.staging_dir / variant["dir"]),
        "-n",
        plan.details["notes"],
    ]


def _describe_actions(plan: Plan) -> list[str]:
    d = plan.details
    actions = [
        f"kaggle models get {d['owner']}/{d['slug']}  (probe; exit 0 = model exists)",
        f"if missing: {shell_join(_model_create(plan))}",
    ]
    for variant in d["variants"]:
        ref = _instance_ref(plan, variant)
        actions += [
            f"kaggle models instances get {ref}  (probe; exit 0 = instance exists)",
            f"if missing: {shell_join(_instance_create(plan, variant))}",
            f"if exists: {shell_join(_version_create(plan, variant))}",
        ]
    return actions


def _exists(argv: list[str]) -> bool:
    """Probe with a ``kaggle ... get`` command: exit 0 means the resource exists."""
    try:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise PublishError("command not found: kaggle") from exc
    return proc.returncode == 0
