"""Publish pipeline: dry-run planning/staging/receipts, refusal paths, mocked real execution."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from gidi.publish import PROVIDERS, publish
from gidi.publish.common import PublishError
from gidi.release import build_release, load_manifest, load_spec, snapshot

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_JSON = ROOT / "releases" / "providers.json"
MODEL = "gidi-fake-v1"
VERSION = "1.0.0"
TOKEN = "hf_SUPERSECRETTOKEN0123456789"
SECRETS = {
    "HF_TOKEN": TOKEN,
    "GH_TOKEN": "ghp_SUPERSECRETGITHUBTOKEN",
    "KAGGLE_USERNAME": "supersecretuser",
    "KAGGLE_KEY": "supersecretkagglekey",
}
_REAL_RUN = subprocess.run

_SCRIPT = ROOT / "scripts" / "publish.py"
_module = importlib.util.spec_from_file_location("publish_cli", _SCRIPT)
cli = importlib.util.module_from_spec(_module)
sys.modules["publish_cli"] = cli
_module.loader.exec_module(cli)


# ------------------------------------------------------------------------------ fixtures


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_json() -> dict:
    return {
        "model_version": MODEL,
        "types": ["expense", "income"],
        "tags": ["O", "B-TARGET", "I-TARGET"],
        "value_labels": ["O", "B-VALUE", "I-VALUE"],
        "value_decoding": {
            "method": "crf_viterbi",
            "start_transitions": [0.0, 0.1, 0.2],
            "end_transitions": [0.0, 0.1, 0.2],
            "transitions": [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
            "span_selection": "highest_confidence",
        },
        "max_length": 32,
        "onnx": {
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["type_logits", "tag_logits", "value_logits"],
            "opset": 17,
        },
    }


@dataclass
class Fake:
    root: Path
    spec_path: Path
    release: Path

    @property
    def spec(self):
        return load_spec(self.spec_path)


def _make_fake_release(tmp_path: Path) -> Fake:
    """A tiny but real release: fake model bytes, built by ``build_release`` (no ONNX checks)."""
    root = tmp_path / "repo"
    blobs = {
        "models/fake/model.int8.onnx": b"int8-model-bytes",
        "models/fake-onnx/model.onnx": b"fp32-model-bytes" * 8,
        "models/fake/tokenizer.json": b'{"tokenizer": true}\n',
        "models/fake/tokenizer_config.json": b"{}\n",
        "models/fake/vocab_map.json": b'{"a": 1}\n',
        "models/fake/config.json": (json.dumps(_config_json(), indent=2) + "\n").encode(),
        "experiments/e/protocol.json": b'{"experiment": 1}\n',
        "experiments/d/protocol.json": b'{"deployment": 1}\n',
    }
    for rel, data in blobs.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    kinds = [
        ("models/fake/model.int8.onnx", "model/int8/model.int8.onnx", "onnx-int8"),
        ("models/fake-onnx/model.onnx", "model/fp32/model.onnx", "onnx-fp32"),
        ("models/fake/tokenizer.json", "tokenizer/tokenizer.json", "tokenizer"),
        (
            "models/fake/tokenizer_config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer-config",
        ),
        ("models/fake/vocab_map.json", "tokenizer/vocab_map.json", "vocab-map"),
        ("models/fake/config.json", "runtime/config.json", "runtime-config"),
    ]
    files = [
        {
            "source": src,
            "path": dst,
            "kind": kind,
            "sha256": _sha(blobs[src]),
            "size_bytes": len(blobs[src]),
        }
        for src, dst, kind in kinds
    ]
    bundle_files = {
        Path(src).name: {"sha256": _sha(blobs[src]), "bytes": len(blobs[src])}
        for src, _, _ in kinds
        if src.startswith("models/fake/")
    }
    bundle_manifest = {
        "protocol": {"sha256": _sha(blobs["experiments/d/protocol.json"])},
        "sources": {
            "source_experiment_protocol": {"sha256": _sha(blobs["experiments/e/protocol.json"])},
            "checkpoint_model_safetensors": {"sha256": _sha(b"checkpoint")},
            "onnx_fp32": {"sha256": _sha(blobs["models/fake-onnx/model.onnx"])},
        },
        "files": bundle_files,
        "runtime_requirements": {"python": ">=3.12", "onnxruntime": ">=1.30"},
    }
    manifest_bytes = (json.dumps(bundle_manifest, indent=2) + "\n").encode()
    (root / "models/fake/manifest.json").write_bytes(manifest_bytes)

    common_members = {
        "config.json": "runtime/config.json",
        "tokenizer.json": "tokenizer/tokenizer.json",
        "tokenizer_config.json": "tokenizer/tokenizer_config.json",
        "vocab_map.json": "tokenizer/vocab_map.json",
        "README.md": "README.md",
        "release-manifest.json": "manifest.json",
    }
    spec = {
        "spec_format": 1,
        "model_version": MODEL,
        "source": {
            "experiment": "fake-experiment",
            "seed": 1,
            "experiment_protocol": {
                "path": "experiments/e/protocol.json",
                "sha256": _sha(blobs["experiments/e/protocol.json"]),
            },
            "deployment_protocol": {
                "path": "experiments/d/protocol.json",
                "sha256": _sha(blobs["experiments/d/protocol.json"]),
            },
            "bundle": {"path": "models/fake", "manifest_sha256": _sha(manifest_bytes)},
            "checkpoint_sha256": _sha(b"checkpoint"),
        },
        "files": files,
        "readmes": {"README.md": "releases/templates/fake.README.md"},
        "variants": {
            "int8": {
                "members": {"model.int8.onnx": "model/int8/model.int8.onnx", **common_members},
                "model_file": "model.int8.onnx",
            },
            "fp32": {
                "members": {"model.onnx": "model/fp32/model.onnx", **common_members},
                "model_file": "model.onnx",
            },
        },
        "smoke": [],
    }
    template = root / "releases/templates/fake.README.md"
    template.parent.mkdir(parents=True)
    template.write_text(
        "---\nlanguage: vi\nlibrary_name: onnx\ntags:\n  - finance\n---\n"
        "# {model_version} {release_version}\n\nTypes: {types}\n"
    )
    spec_path = root / "releases" / f"{MODEL}.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    release = build_release(
        load_spec(spec_path),
        VERSION,
        root=root,
        out_dir=tmp_path / "dist" / "releases" / MODEL / VERSION,
        allow_unversioned=True,
        check_bundle=False,
    )
    return Fake(root=root, spec_path=spec_path, release=release)


@pytest.fixture
def fake(tmp_path) -> Fake:
    return _make_fake_release(tmp_path)


class Runner:
    """Stands in for ``subprocess.run`` for ``gh`` / ``kaggle``; everything else runs for real."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.respond = lambda argv: SimpleNamespace(returncode=1, stdout="", stderr="")

    def __call__(self, argv, **kwargs):
        if argv[0] not in ("gh", "kaggle"):
            return _REAL_RUN(argv, **kwargs)
        self.calls.append(list(argv))
        return self.respond(argv)


class FakeHfApi:
    instances: list[FakeHfApi] = []

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, dict]] = []
        FakeHfApi.instances.append(self)

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        # the staged tree is only valid while we are called: capture what would be uploaded
        folder = Path(kwargs["folder_path"])
        self.uploaded = sorted(
            p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
        )
        return SimpleNamespace(oid="a" * 40)

    def create_tag(self, **kwargs):
        self.calls.append(("create_tag", kwargs))


@pytest.fixture
def world(monkeypatch, tmp_path):
    """No network, no real CLIs, no real credentials; records every provider interaction."""
    import huggingface_hub

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    runner = Runner()
    monkeypatch.setattr(subprocess, "run", runner)
    FakeHfApi.instances = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in (*SECRETS, "GITHUB_TOKEN", "GITHUB_REPOSITORY", "HF_HOME", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(runner=runner, home=home, hf=FakeHfApi)


def run_publish(
    fake,
    tmp_path,
    provider,
    *,
    dry_run=True,
    env=None,
    overrides=None,
    confirm=None,
    providers_config=PROVIDERS_JSON,
):
    lines: list[str] = []
    receipts = publish(
        provider,
        fake.release,
        providers_config=providers_config,
        dry_run=dry_run,
        confirm=not dry_run if confirm is None else confirm,
        staging_root=tmp_path / "staging",
        receipts_root=tmp_path / "pubs",
        smoke=False,
        spec=fake.spec,
        root=fake.root,
        overrides=overrides,
        env=env or {},
        out=lines.append,
    )
    return receipts, "\n".join(lines)


def staged_files(tmp_path, provider) -> dict[str, Path]:
    base = tmp_path / "staging" / provider / MODEL / VERSION
    return {p.relative_to(base).as_posix(): p for p in sorted(base.rglob("*")) if p.is_file()}


def read_receipt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def release_files(fake) -> dict[str, Path]:
    return {
        p.relative_to(fake.release).as_posix(): p
        for p in sorted(fake.release.rglob("*"))
        if p.is_file()
    }


def hf_uploads(name):
    """Release paths the Hugging Face provider uploads: no archives, no FP32 reference model."""
    return not name.startswith("archives/") and name != "model/fp32/model.onnx"


def assert_no_provider_calls(world):
    assert world.runner.calls in ([], [["gh", "auth", "token"]])
    assert world.hf.instances == []


# ------------------------------------------------------------------------------ dry-run plans


def test_huggingface_dry_run_stages_release_files_except_archives_and_fp32(fake, world, tmp_path):
    receipts, output = run_publish(fake, tmp_path, "huggingface")

    expected = {name for name in release_files(fake) if hf_uploads(name)}
    assert "model/int8/model.int8.onnx" in expected
    staged = staged_files(tmp_path, "huggingface")
    assert set(staged) == expected
    assert {"README.md", "manifest.json", "checksums.txt", "runtime/config.json"} <= expected
    for name, path in staged.items():
        original = fake.release / name
        assert path.read_bytes() == original.read_bytes()
        assert path.stat().st_nlink == 1 and not path.is_symlink()
        assert path.stat().st_ino != original.stat().st_ino
    receipt = read_receipt(receipts[0])
    assert {f["path"] for f in receipt["files"]} == expected
    assert "huggingface" in output and "repo_id: x8bu/gidi-finance\n" in output
    assert f"tag: {MODEL}-{VERSION}" in output
    assert "README.md from the release" in output
    assert_no_provider_calls(world)


def test_github_dry_run_plans_two_archives_checksums_manifest_and_notes(fake, world, tmp_path):
    receipts, output = run_publish(fake, tmp_path, "github")

    staged = staged_files(tmp_path, "github")
    archives = sorted(p.name for p in (fake.release / "archives").iterdir())
    assert len(archives) == 2
    assert set(staged) == {
        *(f"assets/{name}" for name in archives),
        "assets/checksums.txt",
        "assets/manifest.json",
        "release-notes.md",
    }
    for name in (*archives, "checksums.txt", "manifest.json"):
        source = fake.release / ("archives" if name in archives else "") / name
        assert staged[f"assets/{name}"].read_bytes() == source.read_bytes()
    notes = staged["release-notes.md"].read_text(encoding="utf-8")
    assert f"{MODEL} {VERSION}" in notes
    for name in archives:
        sha = hashlib.sha256((fake.release / "archives" / name).read_bytes()).hexdigest()
        assert sha in notes
    receipt = read_receipt(receipts[0])
    assert receipt["target"]["tag"] == f"{MODEL}-{VERSION}"
    command = next(a for a in receipt["actions"] if a.startswith("gh release create"))
    assert "--draft" not in command and "--repo 8bu/gidi" in command
    assert f"--title '{MODEL} {VERSION}'" in command and "--notes-file" in command
    assert "gh release create" in output and "release-notes.md" in output
    assert_no_provider_calls(world)


def test_github_plan_targets_the_manifest_commit_when_known(fake, tmp_path):
    manifest = copy.deepcopy(load_manifest(fake.release))
    manifest["git"] = {"commit": "c" * 40, "dirty": False}
    config = json.loads(PROVIDERS_JSON.read_text())["github"]

    plan = PROVIDERS["github"].plan(fake.release, manifest, config, tmp_path / "stage")

    assert plan.actions[0].endswith(f"--target {'c' * 40}")
    assert not (tmp_path / "stage").exists()  # plan() is pure


def test_kaggle_dry_run_stages_archive_members_per_variant_plus_metadata(fake, world, tmp_path):
    receipts, output = run_publish(fake, tmp_path, "kaggle")

    manifest = load_manifest(fake.release)
    expected = {"model/model-metadata.json"}
    for archive in manifest["archives"]:
        variant = archive["variant"]
        expected |= {f"instances/{variant}/{member}" for member in archive["members"]}
        expected.add(f"instances/{variant}/model-instance-metadata.json")
    staged = staged_files(tmp_path, "kaggle")
    assert set(staged) == expected
    for archive in manifest["archives"]:
        for member, source in archive["members"].items():
            path = staged[f"instances/{archive['variant']}/{member}"]
            assert path.read_bytes() == (fake.release / source).read_bytes()
    meta = json.loads(staged["instances/int8/model-instance-metadata.json"].read_text())
    assert meta["instanceSlug"] == f"{MODEL}-int8" and meta["framework"] == "onnx"
    assert meta["ownerSlug"] == "8bu" and meta["modelSlug"] == "gidi-finance"
    model_meta = json.loads(staged["model/model-metadata.json"].read_text())
    assert model_meta["slug"] == "gidi-finance"
    actions = "\n".join(read_receipt(receipts[0])["actions"])
    assert "kaggle models create -p" in actions
    assert "kaggle models instances create -p" in actions
    assert f"kaggle models instances versions create 8bu/gidi-finance/onnx/{MODEL}-fp32" in actions
    assert "kaggle models create" in output
    assert_no_provider_calls(world)


def test_plan_printout_lists_target_files_sizes_hashes_total_and_auth(fake, world, tmp_path):
    _, output = run_publish(fake, tmp_path, "huggingface", env={"HF_TOKEN": TOKEN})

    release = release_files(fake)
    total = sum(p.stat().st_size for n, p in release.items() if hf_uploads(n))
    assert f"{total} bytes total" in output
    config = release["runtime/config.json"]
    assert hashlib.sha256(config.read_bytes()).hexdigest() in output
    assert str(config.stat().st_size) in output
    assert "auth: available (env:HF_TOKEN)" in output
    assert "repo_type: model" in output and "actions that WOULD run" in output
    assert "upload_folder" in output and "create_tag" in output


def test_all_runs_providers_in_order_and_writes_one_dry_run_receipt_each(fake, world, tmp_path):
    receipts, output = run_publish(fake, tmp_path, "all")

    assert [r.name for r in receipts] == [
        "huggingface.dry-run.json",
        "github.dry-run.json",
        "kaggle.dry-run.json",
    ]
    positions = [output.index(f"== {p}:") for p in ("huggingface", "github", "kaggle")]
    assert positions == sorted(positions)
    assert_no_provider_calls(world)


# ------------------------------------------------------------------------------ receipts


def test_dry_run_receipt_is_marked_and_describes_the_release(fake, world, tmp_path):
    receipts, _ = run_publish(fake, tmp_path, "github", env={"GH_TOKEN": SECRETS["GH_TOKEN"]})

    assert receipts[0] == tmp_path / "pubs" / MODEL / VERSION / "github.dry-run.json"
    assert not (tmp_path / "pubs" / MODEL / VERSION / "github.json").exists()
    receipt = read_receipt(receipts[0])
    assert receipt["receipt_format"] == 1 and receipt["dry_run"] is True
    assert receipt["provider"] == "github" and receipt["remote"] is None
    assert receipt["model_version"] == MODEL and receipt["release_version"] == VERSION
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256((fake.release / "manifest.json").read_bytes()).hexdigest()
    )
    assert (
        receipt["checksums_sha256"]
        == hashlib.sha256((fake.release / "checksums.txt").read_bytes()).hexdigest()
    )
    assert receipt["auth"] == {"available": True, "source": "env:GH_TOKEN"}
    assert {"path", "sha256", "size_bytes"} == set(receipt["files"][0])
    assert receipt["timestamp"]


def test_release_directory_is_unchanged_by_publishing(fake, world, tmp_path):
    before = snapshot(fake.release)
    modes = {p: p.stat().st_mode for p in fake.release.rglob("*")}

    run_publish(fake, tmp_path, "all")

    assert snapshot(fake.release) == before
    assert {p: p.stat().st_mode for p in fake.release.rglob("*")} == modes


def test_staging_is_wiped_and_recreated(fake, world, tmp_path):
    stale = tmp_path / "staging" / "huggingface" / MODEL / VERSION / "stale.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"left over")

    run_publish(fake, tmp_path, "huggingface")

    assert not stale.exists()
    assert "stale.bin" not in staged_files(tmp_path, "huggingface")


# ------------------------------------------------------------------------------ refusals


def test_tampered_release_is_refused_without_staging_or_receipt(fake, world, tmp_path, capsys):
    target = fake.release / "model" / "int8" / "model.int8.onnx"
    target.chmod(0o644)
    target.write_bytes(b"tampered")

    code = cli.main(
        [
            "all",
            "--release",
            str(fake.release),
            "--dry-run",
            "--no-smoke",
            "--spec",
            str(fake.spec_path),
            "--staging-root",
            str(tmp_path / "staging"),
            "--receipts-root",
            str(tmp_path / "pubs"),
        ]
    )

    assert code == 1
    assert "does not verify" in capsys.readouterr().err
    assert not (tmp_path / "staging").exists()
    assert not (tmp_path / "pubs").exists()
    assert_no_provider_calls(world)


def test_cli_dry_run_exits_zero_and_prints_the_plan(fake, world, tmp_path, capsys):
    code = cli.main(
        [
            "github",
            "--release",
            str(fake.release),
            "--dry-run",
            "--no-smoke",
            "--spec",
            str(fake.spec_path),
            "--staging-root",
            str(tmp_path / "staging"),
            "--receipts-root",
            str(tmp_path / "pubs"),
            "--github-repo",
            "someone/else",
        ]
    )

    assert code == 0
    assert "--repo someone/else" in capsys.readouterr().out
    assert (tmp_path / "pubs" / MODEL / VERSION / "github.dry-run.json").is_file()


def test_github_repository_env_overrides_config_and_flag_overrides_env(fake, world, tmp_path):
    _, output = run_publish(fake, tmp_path, "github", env={"GITHUB_REPOSITORY": "ci/repo"})
    assert "--repo ci/repo" in output
    _, output = run_publish(
        fake,
        tmp_path,
        "github",
        env={"GITHUB_REPOSITORY": "ci/repo"},
        overrides={"github": {"repo": "flag/repo"}},
    )
    assert "--repo flag/repo" in output


@pytest.mark.parametrize("provider", ["huggingface", "github", "kaggle", "all"])
def test_real_publish_without_credentials_fails_before_executing(fake, world, tmp_path, provider):
    with pytest.raises(PublishError, match="no credentials found"):
        run_publish(fake, tmp_path, provider, dry_run=False)

    assert_no_provider_calls(world)
    assert not (tmp_path / "pubs").exists()


def test_real_publish_without_confirmation_is_refused_before_any_probe(fake, world, tmp_path):
    with pytest.raises(PublishError, match="--confirm-publish"):
        run_publish(fake, tmp_path, "all", dry_run=False, confirm=False, env={"HF_TOKEN": TOKEN})

    assert world.runner.calls == []  # not even the `gh auth token` probe
    assert world.hf.instances == []
    assert not (tmp_path / "staging").exists() and not (tmp_path / "pubs").exists()
    argv = ["github", "--release", str(fake.release), "--spec", str(fake.spec_path)]
    assert cli.main(argv) == 1


def test_real_publish_without_cli_fails_before_executing(fake, world, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(PublishError, match="`gh` CLI is not installed"):
        run_publish(fake, tmp_path, "github", dry_run=False, env={"GH_TOKEN": "x" * 10})

    assert world.runner.calls == []
    assert not (tmp_path / "pubs").exists()


def test_kaggle_real_publish_requires_a_license(fake, world, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    env = {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k" * 8}
    config = json.loads(PROVIDERS_JSON.read_text())
    config["kaggle"]["license"] = None
    unlicensed = tmp_path / "providers.json"
    unlicensed.write_text(json.dumps(config))

    with pytest.raises(PublishError, match="license"):
        run_publish(
            fake,
            tmp_path,
            "kaggle",
            dry_run=False,
            env=env,
            providers_config=unlicensed,
        )

    assert world.runner.calls == []


def test_all_checks_every_provider_before_executing_any(fake, world, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    env = {"HF_TOKEN": TOKEN, "GH_TOKEN": SECRETS["GH_TOKEN"]}  # kaggle credentials missing

    with pytest.raises(PublishError, match="kaggle: no credentials"):
        run_publish(fake, tmp_path, "all", dry_run=False, env=env)

    assert world.hf.instances == []
    assert world.runner.calls == []


# ------------------------------------------------------------------------------ secrets


def test_secret_values_never_appear_in_output_or_receipts(fake, world, tmp_path, capsys):
    receipts, output = run_publish(fake, tmp_path, "all", env=dict(SECRETS))

    assert "env:HF_TOKEN" in output
    blobs = [output, capsys.readouterr().out + capsys.readouterr().err]
    blobs += [r.read_text() for r in receipts]
    for blob in blobs:
        for value in SECRETS.values():
            assert value not in blob


def test_secret_values_are_scrubbed_from_failed_command_errors(fake, world, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    secret = SECRETS["GH_TOKEN"]
    world.runner.respond = lambda argv: SimpleNamespace(
        returncode=1, stdout="", stderr=f"auth failed for token {secret}"
    )

    with pytest.raises(PublishError) as raised:
        run_publish(fake, tmp_path, "github", dry_run=False, env={"GH_TOKEN": secret})

    assert secret not in str(raised.value) and "***" in str(raised.value)
    assert not (tmp_path / "pubs").exists()


# ------------------------------------------------------------------------------ real execution


def test_huggingface_execute_creates_repo_uploads_staging_and_tags(fake, world, tmp_path):
    receipts, _ = run_publish(fake, tmp_path, "huggingface", dry_run=False, env={"HF_TOKEN": TOKEN})

    (api,) = world.hf.instances
    assert [name for name, _ in api.calls] == ["create_repo", "upload_folder", "create_tag"]
    create, upload, tag = (kwargs for _, kwargs in api.calls)
    assert create == {
        "repo_id": "x8bu/gidi-finance",
        "repo_type": "model",
        "private": False,
        "exist_ok": True,
    }
    staging = tmp_path / "staging" / "huggingface" / MODEL / VERSION
    assert Path(upload["folder_path"]) == staging
    assert upload["repo_id"] == "x8bu/gidi-finance" and upload["repo_type"] == "model"
    assert api.uploaded == sorted(n for n in release_files(fake) if hf_uploads(n))
    assert tag["tag"] == f"{MODEL}-{VERSION}" and tag["revision"] == "a" * 40
    assert world.runner.calls == []
    receipt = read_receipt(receipts[0])
    assert receipts[0].name == "huggingface.json" and receipt["dry_run"] is False
    assert receipt["remote"]["revision"] == "a" * 40
    assert receipt["target"]["revision"] == "a" * 40


def test_github_execute_runs_gh_release_create_with_staged_assets(
    fake, world, tmp_path, monkeypatch
):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")

    def respond(argv):
        return SimpleNamespace(
            returncode=0, stdout="https://github.com/8bu/gidi/releases/tag/x\n", stderr=""
        )

    world.runner.respond = respond

    receipts, _ = run_publish(fake, tmp_path, "github", dry_run=False, env={"GH_TOKEN": "t" * 12})

    (argv,) = world.runner.calls
    staging = tmp_path / "staging" / "github" / MODEL / VERSION
    assert argv[:4] == ["gh", "release", "create", f"{MODEL}-{VERSION}"]
    assets = [a for a in argv[4 : argv.index("--repo")]]
    assert sorted(Path(a).name for a in assets) == sorted(
        [*(p.name for p in (fake.release / "archives").iterdir()), "checksums.txt", "manifest.json"]
    )
    assert all(Path(a).parent == staging / "assets" and Path(a).is_file() for a in assets)
    flags = argv[argv.index("--repo") :]
    assert flags[:2] == ["--repo", "8bu/gidi"]
    assert flags[flags.index("--title") + 1] == f"{MODEL} {VERSION}"
    assert Path(flags[flags.index("--notes-file") + 1]) == staging / "release-notes.md"
    assert "--draft" not in flags
    receipt = read_receipt(receipts[0])
    assert receipt["dry_run"] is False
    assert receipt["remote"]["url"] == "https://github.com/8bu/gidi/releases/tag/x"


def test_github_failed_command_is_an_error_without_receipt(fake, world, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    world.runner.respond = lambda argv: SimpleNamespace(returncode=1, stdout="", stderr="exists")

    with pytest.raises(PublishError, match="exists"):
        run_publish(fake, tmp_path, "github", dry_run=False, env={"GH_TOKEN": "t" * 12})

    assert not (tmp_path / "pubs").exists()


def test_github_auth_falls_back_to_gh_auth_token_status(fake, world, tmp_path):
    world.runner.respond = lambda argv: SimpleNamespace(returncode=0, stdout="SECRET", stderr="")

    receipts, output = run_publish(fake, tmp_path, "github")

    assert world.runner.calls == [["gh", "auth", "token"]]
    assert read_receipt(receipts[0])["auth"] == {"available": True, "source": "gh auth token"}
    assert "SECRET" not in output


def test_kaggle_execute_creates_missing_model_and_versions_existing_instance(
    fake, world, tmp_path, monkeypatch
):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    ok = SimpleNamespace(returncode=0, stdout="", stderr="")
    missing = SimpleNamespace(returncode=1, stdout="", stderr="not found")

    def respond(argv):
        if argv[:3] == ["kaggle", "models", "get"]:
            return missing
        if argv[:4] == ["kaggle", "models", "instances", "get"]:
            return ok if argv[4].endswith("-fp32") else missing
        return ok

    world.runner.respond = respond
    env = {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k" * 8}

    receipts, _ = run_publish(
        fake,
        tmp_path,
        "kaggle",
        dry_run=False,
        env=env,
        overrides={"kaggle": {"license": "Apache 2.0"}},
    )

    staging = tmp_path / "staging" / "kaggle" / MODEL / VERSION
    int8_ref = f"8bu/gidi-finance/onnx/{MODEL}-int8"
    fp32_ref = f"8bu/gidi-finance/onnx/{MODEL}-fp32"
    assert world.runner.calls == [
        ["kaggle", "models", "get", "8bu/gidi-finance"],
        ["kaggle", "models", "create", "-p", str(staging / "model")],
        ["kaggle", "models", "instances", "get", int8_ref],
        ["kaggle", "models", "instances", "create", "-p", str(staging / "instances" / "int8")],
        ["kaggle", "models", "instances", "get", fp32_ref],
        [
            *["kaggle", "models", "instances", "versions", "create", fp32_ref],
            *["-p", str(staging / "instances" / "fp32"), "-n", f"{MODEL} {VERSION}"],
        ],
    ]
    receipt = read_receipt(receipts[0])
    assert receipt["dry_run"] is False
    assert receipt["remote"]["created_model"] is True
    assert receipt["remote"]["instances"]["fp32"]["action"] == "new-version"
    assert receipt["remote"]["instances"]["int8"]["action"] == "created"
    meta = json.loads((staging / "instances" / "int8" / "model-instance-metadata.json").read_text())
    assert meta["licenseName"] == "Apache 2.0"
