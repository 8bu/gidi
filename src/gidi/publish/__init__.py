"""Publish a verified release directory to HF / GitHub / Kaggle (dry-run by default in CI)."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from gidi.publish.common import (
    Auth,
    Plan,
    PlanFile,
    Provider,
    PublishError,
    load_providers_config,
    publish_release,
)
from gidi.publish.github import GitHubProvider
from gidi.publish.huggingface import HuggingFaceProvider
from gidi.publish.kaggle import KaggleProvider
from gidi.release.manifest import load_manifest
from gidi.release.spec import ReleaseSpec

PROVIDERS: dict[str, Provider] = {
    "huggingface": HuggingFaceProvider(),
    "github": GitHubProvider(),
    "kaggle": KaggleProvider(),
}
ALL_PROVIDERS = ("huggingface", "github", "kaggle")


def model_family(model_version: str) -> str:
    """``gidi-finance`` for ``gidi-finance-v4``: the model name without its ``-vN`` suffix."""
    return re.sub(r"-v\d+$", "", model_version)


def resolve_configs(
    names: tuple[str, ...],
    base: Mapping[str, Any],
    env: Mapping[str, str],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    family: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-provider config: providers.json, then ``GITHUB_REPOSITORY``, then explicit overrides.

    ``families[<family>][<provider>]`` in providers.json replaces the top-level section of that
    provider for one model family (never merged, so no field leaks from another family); ``null``
    switches the provider off for the family.
    """
    families = base.get("families", {})
    if not isinstance(families, dict):
        raise PublishError("providers config 'families' must be an object")
    own = families.get(family, {}) if family is not None else {}
    configs: dict[str, dict[str, Any]] = {}
    for name in names:
        section = own.get(name, base.get(name)) if isinstance(own, dict) else base.get(name)
        if not isinstance(section, dict):
            where = f" for the {family} family" if name in own else ""
            raise PublishError(f"providers config has no section for {name}{where}")
        config = dict(section)
        if name == "github" and env.get("GITHUB_REPOSITORY"):
            config["repo"] = env["GITHUB_REPOSITORY"]
        config.update({k: v for k, v in (overrides or {}).get(name, {}).items() if v is not None})
        configs[name] = config
    return configs


def publish(
    provider: str,
    release_dir: Path,
    *,
    providers_config: Path,
    dry_run: bool,
    confirm: bool = False,
    staging_root: Path,
    receipts_root: Path,
    smoke: bool = True,
    spec: ReleaseSpec | None = None,
    root: Path | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    env: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
) -> list[Path]:
    """Publish (or dry-run) ``provider`` (a name or ``"all"``); returns the receipt paths."""
    env = os.environ if env is None else env
    if provider == "all":
        names = ALL_PROVIDERS
    elif provider in PROVIDERS:
        names = (provider,)
    else:
        raise PublishError(f"unknown provider {provider!r}; choose from {[*PROVIDERS, 'all']}")
    family = model_family(load_manifest(release_dir)["model_version"])
    configs = resolve_configs(
        names, load_providers_config(providers_config), env, overrides, family
    )
    return publish_release(
        [PROVIDERS[n] for n in names],
        release_dir,
        configs=configs,
        dry_run=dry_run,
        confirm=confirm,
        staging_root=staging_root,
        receipts_root=receipts_root,
        smoke=smoke,
        spec=spec,
        root=root,
        env=env,
        out=out,
    )


__all__ = [
    "ALL_PROVIDERS",
    "PROVIDERS",
    "Auth",
    "Plan",
    "PlanFile",
    "Provider",
    "PublishError",
    "publish",
    "resolve_configs",
]
