"""Release pipeline: deterministic build, strict verification, immutability."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from release_fakes import MODEL, make_fake_project

from gidi.release import (
    Manifest,
    ReleaseError,
    check_archives,
    load_manifest,
    load_spec,
    snapshot,
    validate_release_version,
    verify_release,
)

EPOCH = 1_000_000_000  # 2001-09-09T01:46:40Z


@pytest.fixture(autouse=True)
def _no_source_date_epoch(monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)


@pytest.fixture
def project(tmp_path):
    return make_fake_project(tmp_path)


@pytest.fixture
def release(project):
    return project.build("1.0.0")


def failed(report) -> dict[str, str]:
    return {check.name: check.detail for check in report.failures}


def verify(project, release_dir, **kwargs):
    return verify_release(release_dir, spec=project.spec, root=project.root, smoke=False, **kwargs)


def writable(path: Path) -> Path:
    path.chmod(0o644)
    return path


# --- versions --------------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["2.0.0", "2.0.0-rc.1", "2.13.7-beta"])
def test_valid_release_versions(version):
    validate_release_version("gidi-finance-v2", version)


@pytest.mark.parametrize(
    "version", ["2.0", "v2.0.0", "2.0.0-", "02.0.0", "2.0.0+build", "", "2.0.0 ", "latest"]
)
def test_malformed_release_versions_are_rejected(version):
    with pytest.raises(ReleaseError, match="MAJOR.MINOR.PATCH"):
        validate_release_version("gidi-finance-v2", version)


def test_release_major_must_match_model_major():
    with pytest.raises(ReleaseError, match="major"):
        validate_release_version("gidi-finance-v2", "1.9.0")
    with pytest.raises(ReleaseError, match="-v<MAJOR>"):
        validate_release_version("gidi-finance", "1.0.0")


# --- spec ------------------------------------------------------------------------------------


def test_spec_rejects_a_variant_member_that_maps_to_no_file(project):
    def mutate(data):
        data["variants"]["int8"]["members"]["config.json"] = "runtime/nope.json"

    with pytest.raises(ReleaseError, match="unknown"):
        load_spec(project.edit_spec(mutate))


def test_spec_rejects_paths_that_escape_the_release(project):
    def mutate(data):
        data["files"][0]["path"] = "../model.onnx"

    with pytest.raises(ReleaseError, match="relative"):
        load_spec(project.edit_spec(mutate))


# --- manifest --------------------------------------------------------------------------------


def test_manifest_round_trips_byte_for_byte(release):
    text = (release / "manifest.json").read_text(encoding="utf-8")
    assert text.endswith("}\n") and not text.endswith("\n\n")
    manifest = Manifest.parse(text)
    assert manifest.serialize() == text
    assert Manifest.parse(manifest.serialize()) == manifest
    assert [item["path"] for item in manifest.artifacts] == sorted(
        item["path"] for item in manifest.artifacts
    )


def test_manifest_rejects_unknown_or_malformed_fields(release):
    data = load_manifest(release)
    with pytest.raises(ReleaseError, match="keys"):
        Manifest.from_dict({**data, "hf_repo": "x"})
    with pytest.raises(ReleaseError, match="sha256"):
        Manifest.from_dict({**data, "artifacts": [{**data["artifacts"][0], "sha256": "xyz"}]})
    with pytest.raises(ReleaseError, match="not valid JSON"):
        Manifest.parse("{")


def test_manifest_records_artifacts_requirements_and_no_archive_hashes(release):
    manifest = load_manifest(release)
    kinds = {item["path"]: item["kind"] for item in manifest["artifacts"]}
    assert kinds["README.md"] == "readme"
    assert kinds["model/fp32/model.onnx"] == "onnx-fp32"
    assert "manifest.json" not in kinds and "checksums.txt" not in kinds
    assert manifest["requirements"] == {"python": ">=3.12", "onnxruntime": ">=1.30.0"}
    assert manifest["runtime"]["value_decoding"] == "crf_viterbi"
    assert {item["variant"] for item in manifest["archives"]} == {"int8", "fp32"}
    assert all(set(item) == {"path", "variant", "root", "members"} for item in manifest["archives"])
    assert "/" not in manifest["source"]["experiment"]


def test_snapshot_covers_every_file_with_hash_and_size(release):
    shot = snapshot(release)
    assert set(shot) >= {
        "manifest.json",
        "checksums.txt",
        "README.md",
        "model/int8/model.int8.onnx",
    }
    digest, size = shot["runtime/config.json"]
    assert size == (release / "runtime/config.json").stat().st_size and len(digest) == 64


# --- determinism -----------------------------------------------------------------------------


def test_two_builds_are_byte_identical_including_archives(project, tmp_path):
    first = project.build("1.0.0", dist=tmp_path / "a")
    second = project.build("1.0.0", dist=tmp_path / "b")
    assert snapshot(first) == snapshot(second)
    assert any(name.endswith(".tar.gz") for name in snapshot(first))


def test_build_is_identical_across_checkouts_of_the_same_inputs(tmp_path):
    one = make_fake_project(tmp_path / "one")
    two = make_fake_project(tmp_path / "two")
    assert snapshot(one.build("1.0.0")) == snapshot(two.build("1.0.0"))


def test_source_date_epoch_drives_timestamp_and_archive_mtimes(project, tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(EPOCH))
    release = project.build("1.0.0")
    manifest = load_manifest(release)
    assert manifest["build_timestamp"] == "2001-09-09T01:46:40Z"
    for item in manifest["archives"]:
        with tarfile.open(release / item["path"]) as tar:
            assert {member.mtime for member in tar.getmembers()} == {EPOCH}
    assert check_archives(release) == []
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(EPOCH + 1))
    other = project.build("1.0.0", dist=tmp_path / "other")
    assert snapshot(other) != snapshot(release)


def test_without_any_time_source_timestamp_is_null_and_mtime_zero(release):
    manifest = load_manifest(release)
    assert manifest["build_timestamp"] is None
    with tarfile.open(release / manifest["archives"][0]["path"]) as tar:
        assert {member.mtime for member in tar.getmembers()} == {0}


def test_archives_have_normalized_tar_and_gzip_metadata(release):
    manifest = load_manifest(release)
    for item in manifest["archives"]:
        path = release / item["path"]
        header = path.read_bytes()[:10]
        assert header[:2] == b"\x1f\x8b" and header[3] == 0 and header[4:8] == b"\0\0\0\0"
        with tarfile.open(path) as tar:
            members = tar.getmembers()
            root, files = members[0], members[1:]
            assert root.isdir() and root.name == item["root"] and root.mode == 0o755
            assert [m.name for m in files] == sorted(m.name for m in files)
            assert {m.name for m in files} == {f"{item['root']}/{n}" for n in item["members"]}
            for member in members:
                assert (member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "")
            assert all(m.isreg() and m.mode == 0o644 for m in files)
            for member in files:
                name = member.name.removeprefix(item["root"] + "/")
                source = release / item["members"][name]
                assert tar.extractfile(member).read() == source.read_bytes()
        assert gzip.decompress(path.read_bytes())  # a valid single gzip member


def test_release_files_are_read_only(release):
    files = [path for path in release.rglob("*") if path.is_file()]
    assert files and all(path.stat().st_mode & 0o222 == 0 for path in files)


def test_checksums_file_is_sha256sum_format_and_covers_everything(release):
    lines = (release / "checksums.txt").read_text(encoding="utf-8").splitlines()
    names = [line.split("  ", 1)[1] for line in lines]
    assert names == sorted(names)
    on_disk = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
    assert set(names) == on_disk - {"checksums.txt"}
    if shutil.which("shasum"):
        subprocess.run(["shasum", "-a", "256", "-c", "checksums.txt"], cwd=release, check=True)


def test_readme_is_rendered_from_the_template(release, project):
    text = (release / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\nlanguage: vi\nlibrary_name: onnx\n")
    assert f"# {MODEL} 1.0.0" in text and "{" not in text
    assert "license" not in text.split("---")[1]


def test_readme_with_unknown_placeholder_fails_the_build(tmp_path):
    project = make_fake_project(tmp_path, template="---\nlanguage: vi\n---\n{nope}\n")
    with pytest.raises(ReleaseError, match="nope"):
        project.build("1.0.0")
    assert not project.release_dir("1.0.0").exists()


def test_readme_must_not_declare_a_license(tmp_path):
    template = "---\nlanguage: vi\nlibrary_name: onnx\nlicense: mit\n---\n# x\n"
    project = make_fake_project(tmp_path, template=template)
    with pytest.raises(ReleaseError, match="license"):
        project.build("1.0.0")


def test_readme_must_declare_the_license_when_a_license_file_ships():
    from gidi.release.build import readme_front_matter_problems

    bare = "---\nlanguage: vi\nlibrary_name: onnx\n---\n# x\n"
    declared = "---\nlanguage: vi\nlibrary_name: onnx\nlicense: apache-2.0\n---\n# x\n"
    assert readme_front_matter_problems(declared, license_file=True) == []
    assert any("license" in p for p in readme_front_matter_problems(bare, license_file=True))


# --- build inputs and immutability ----------------------------------------------------------


def test_build_rejects_a_source_that_differs_from_the_pin(project):
    path = writable(project.root / "models" / MODEL / "tokenizer.json")
    path.write_bytes(b"tampered")
    with pytest.raises(ReleaseError, match="tokenizer.json"):
        project.build("1.0.0")
    assert not project.release_dir("1.0.0").exists()
    assert not any(project.release_dir("1.0.0").parent.glob(".*")), "temp dir left behind"


def test_build_rejects_a_bundle_manifest_that_differs_from_the_pin(project):
    writable(project.root / "models" / MODEL / "manifest.json").write_text("{}\n")
    with pytest.raises(ReleaseError, match="bundle manifest"):
        project.build("1.0.0")


def test_build_rejects_bad_versions_before_touching_disk(project):
    with pytest.raises(ReleaseError):
        project.build("1.0")
    with pytest.raises(ReleaseError, match="major"):
        project.build("2.0.0")
    assert not project.dist.exists()


def test_identical_rebuild_is_a_no_op(project, release):
    before = {p: p.stat().st_mtime_ns for p in release.rglob("*") if p.is_file()}
    assert project.build("1.0.0") == release
    assert {p: p.stat().st_mtime_ns for p in release.rglob("*") if p.is_file()} == before


def test_differing_rebuild_never_overwrites_a_release(project, release):
    before = snapshot(release)
    template = project.root / "releases" / "templates" / f"{MODEL}.README.md"
    template.write_text(template.read_text(encoding="utf-8") + "\nmore\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="not overwriting"):
        project.build("1.0.0")
    assert snapshot(release) == before


def test_build_check_needs_an_identical_existing_release(project, release):
    assert project.build("1.0.0", check=True) == release
    with pytest.raises(ReleaseError, match="existing"):
        project.build("1.0.1", check=True)
    assert not project.release_dir("1.0.1").exists()
    writable(release / "README.md").write_text("changed")
    with pytest.raises(ReleaseError, match="README.md"):
        project.build("1.0.0", check=True)


# --- git ------------------------------------------------------------------------------------


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(root: Path, *args: str, date: str | None = None) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    done = subprocess.run(
        ["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@needs_git
def test_build_outside_git_needs_allow_unversioned(project):
    with pytest.raises(ReleaseError, match="allow-unversioned"):
        project.build("1.0.0", allow_unversioned=False)
    manifest = load_manifest(project.build("1.0.0"))
    assert manifest["git"] == {"commit": None, "dirty": None}


@needs_git
def test_build_records_commit_and_commit_time_from_a_clean_checkout(project):
    git(project.root, "init", "-q")
    git(project.root, "add", "-A")
    git(project.root, "commit", "-q", "-m", "x", date=f"{EPOCH} +0000")
    release = project.build("1.0.0", allow_unversioned=False)
    manifest = load_manifest(release)
    assert manifest["git"] == {"commit": git(project.root, "rev-parse", "HEAD"), "dirty": False}
    assert manifest["build_timestamp"] == "2001-09-09T01:46:40Z"
    assert check_archives(release) == []


@needs_git
def test_dirty_checkout_is_refused_unless_allowed(project):
    git(project.root, "init", "-q")
    git(project.root, "add", "-A")
    git(project.root, "commit", "-q", "-m", "x")
    (project.root / "untracked.txt").write_text("x")
    with pytest.raises(ReleaseError, match="dirty"):
        project.build("1.0.0", allow_unversioned=False)
    manifest = load_manifest(project.build("1.0.0", allow_unversioned=True))
    assert manifest["git"]["dirty"] is True and len(manifest["git"]["commit"]) == 40


# --- verify: success and rejections ----------------------------------------------------------


def test_verify_passes_for_a_fresh_release(project, release):
    report = verify(project, release)
    assert report.ok, report.failures
    assert {check.name for check in report.checks} >= {
        "manifest",
        "files-present",
        "files-unlisted",
        "artifact-hashes",
        "checksums",
        "versions",
        "spec",
        "no-absolute-paths",
        "archives",
    }


def test_verify_finds_the_default_spec_under_root(project, release):
    report = verify_release(release, root=project.root, smoke=False)
    assert report.ok, report.failures


def test_verify_rejects_a_tampered_artifact(project, release):
    path = writable(release / "model/int8/model.int8.onnx")
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    problems = failed(verify(project, release))
    assert "model/int8/model.int8.onnx" in problems["artifact-hashes"]
    assert "model/int8/model.int8.onnx" in problems["checksums"]
    assert "bytes differ from release file" in problems["archives"]


def test_verify_rejects_a_tampered_checksums_file(project, release):
    path = writable(release / "checksums.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = "0" * 64 + lines[0][64:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = failed(verify(project, release))
    assert set(problems) == {"checksums"}


def test_verify_rejects_a_missing_artifact_or_archive(project, release):
    writable(release / "tokenizer/vocab_map.json").unlink()
    (release / "archives").chmod(0o755)
    next((release / "archives").glob("*-int8.tar.gz")).unlink()
    problems = failed(verify(project, release))
    assert "tokenizer/vocab_map.json" in problems["files-present"]
    assert "int8.tar.gz" in problems["files-present"]
    assert "archives" in problems


def test_verify_rejects_a_manifest_that_disagrees_with_the_files(project, release):
    path = writable(release / "manifest.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model_version"] = "fake-v2"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    problems = failed(verify(project, release))
    assert "model_version" in problems["versions"]
    assert "spec" in problems


def test_verify_rejects_a_release_in_the_wrong_directory(project, release, tmp_path):
    moved = tmp_path / "elsewhere" / "9.9.9"
    shutil.copytree(release, moved)
    assert "directory name" in failed(verify(project, moved))["versions"]


def test_verify_rejects_a_malformed_manifest(project, release):
    writable(release / "manifest.json").write_text("{not json")
    report = verify(project, release)
    assert [check.name for check in report.failures] == ["manifest"]


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("model.safetensors", "forbidden"),
        ("train.jsonl", "forbidden"),
        ("corpus.quet.db", "forbidden"),
        (".DS_Store", "forbidden"),
        ("__pycache__/x.pyc", "forbidden"),
        ("debug.log", "forbidden"),
        ("notes.txt", "unlisted"),
    ],
)
def test_verify_rejects_extra_files(project, release, name, label):
    extra = release / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"x")
    problem = failed(verify(project, release))["files-unlisted"]
    assert f"{label} file {name}" in problem


def test_verify_rejects_symlinks(project, release):
    os.symlink(release / "README.md", release / "link.md")
    assert "non-regular file link.md" in failed(verify(project, release))["files-unlisted"]


@pytest.mark.parametrize(
    ("leak", "marker"),
    [
        ("/Users/someone/work", "/Users/"),
        ("/home/ci/work", "/home/"),
        ("/root/work", "/root/"),
        ("C:\\build", "C:\\"),
        ("{root}", "{root}"),
    ],
)
def test_verify_rejects_absolute_path_leaks(tmp_path, leak, marker):
    base = make_fake_project(tmp_path)
    leaky = f"---\nlanguage: vi\nlibrary_name: onnx\n---\n# {{model_version}}\nbuilt in {leak}\n"
    project = make_fake_project(
        tmp_path / "leaky", template=leaky.replace("{root}", str(base.root))
    )
    release = project.build("1.0.0")
    marker = marker if marker != "{root}" else str(base.root)
    report = verify_release(release, spec=project.spec, root=base.root, smoke=False)
    assert marker in failed(report)["no-absolute-paths"]


def test_verify_rejects_artifacts_that_do_not_match_the_spec_pin(project, release):
    def mutate(data):
        data["files"][2]["sha256"] = "0" * 64

    other = load_spec(project.edit_spec(mutate))
    report = verify_release(release, spec=other, root=project.root, smoke=False)
    assert "tokenizer/tokenizer.json" in failed(report)["spec"]


def test_verify_rejects_an_archive_with_unnormalized_metadata(project, release):
    from gidi.release import write_archive
    from gidi.release.package import member_sources

    manifest = load_manifest(release)
    archive = manifest["archives"][0]
    path = release / archive["path"]
    (release / "archives").chmod(0o755)
    path.unlink()
    write_archive(path, archive["root"], member_sources(release, archive), mtime=12345)
    problems = failed(verify(project, release))
    assert "mtime 12345" in problems["archives"]
    assert check_archives(release), "package --check also sees the difference"


def test_verify_rejects_an_archive_whose_member_bytes_differ(project, release):
    from gidi.release import write_archive
    from gidi.release.package import member_sources

    archive = load_manifest(release)["archives"][0]
    other = project.root / "other-readme.md"
    other.write_text("not the release README")
    members = {**member_sources(release, archive), "README.md": other}
    path = release / archive["path"]
    (release / "archives").chmod(0o755)
    path.unlink()
    write_archive(path, archive["root"], members, mtime=0)
    assert "README.md: bytes differ" in failed(verify(project, release))["archives"]


def test_package_check_detects_changed_archives(release):
    assert check_archives(release) == []
    path = next((release / "archives").glob("*-fp32.tar.gz"))
    (release / "archives").chmod(0o755)
    data = writable(path).read_bytes()
    path.write_bytes(data[:-8] + bytes(8))
    assert any("fp32.tar.gz" in problem for problem in check_archives(release))


# --- smoke inference (real ONNX from the throwaway value-head bundle) -------------------------


def test_smoke_runs_both_variants_from_the_extracted_archives(tmp_path, v2_bundle):
    project = make_fake_project(tmp_path, onnx_bundle=v2_bundle)
    release = project.build("1.0.0")
    report = verify_release(release, spec=project.spec, root=project.root)
    assert report.ok, report.failures
    names = {check.name for check in report.checks}
    assert {"onnx-load:int8", "smoke:int8", "onnx-load:fp32", "smoke:fp32"} <= names

    def mutate(data):
        data["smoke"][0]["expected"]["value_text"] = "definitely not this"

    wrong = load_spec(project.edit_spec(mutate))
    problems = failed(verify_release(release, spec=wrong, root=project.root))
    assert "definitely not this" in problems["smoke:int8"]
    assert "definitely not this" in problems["smoke:fp32"]


def test_smoke_flag_off_skips_model_loading(tmp_path, v2_bundle):
    project = make_fake_project(tmp_path, onnx_bundle=v2_bundle)
    release = project.build("1.0.0")
    report = verify_release(release, spec=project.spec, root=project.root, smoke=False)
    assert report.ok and not any("smoke" in check.name for check in report.checks)
