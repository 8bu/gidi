# Releasing

A release is a deterministic, hash-pinned directory built from the frozen model files. Publishing
only copies a verified release to providers. All logic is in `src/gidi/release/`,
`src/gidi/publish/`, `scripts/release.py` and `scripts/publish.py`; CI only runs those commands.

Inputs are pinned by the committed spec `releases/gidi-finance-v2.json` (paths and sha256 of the
bundle, FP32 model, protocols, checkpoint). Provider metadata (repo ids, tags) is in
`releases/providers.json` and is not part of the release.

All commands below use `gidi-finance-v2` and `2.0.0`.

## 1. Build locally

Needs the model files in `models/` (gitignored) and a clean git tree.

```bash
uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0
```

Output: `dist/releases/gidi-finance-v2/2.0.0/` (all files `0444`). Re-running with identical
inputs is a no-op; a different result for an existing directory is an error, never an overwrite.
For a dirty tree (local experiments only; records commit `null`):

```bash
uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0 --allow-unversioned
```

Use `--out DIR` to build elsewhere and `--spec PATH` for another spec.

## 2. Verify locally

```bash
uv run python scripts/release.py verify --release dist/releases/gidi-finance-v2/2.0.0
uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0 --check
uv run python scripts/release.py package --release dist/releases/gidi-finance-v2/2.0.0 --check
```

`verify` exits 0/1 and prints each check (hashes, files, versions, absolute paths, archives,
ONNX load, smoke predictions for INT8 and FP32). `--no-smoke` skips the model runs.
`build --check` rebuilds in a temp dir and requires byte identity; `package --check` re-creates the
archives and compares bytes.

## 3. Inspect the release

```bash
cd dist/releases/gidi-finance-v2/2.0.0
find . -type f | sort
sha256sum -c checksums.txt          # macOS: shasum -a 256 -c checksums.txt
python -m json.tool manifest.json
```

```text
README.md  manifest.json  checksums.txt
model/int8/model.int8.onnx   model/fp32/model.onnx
tokenizer/{tokenizer.json,tokenizer_config.json,vocab_map.json}
runtime/config.json
archives/gidi-finance-v2-2.0.0-{int8,fp32}.tar.gz
```

`manifest.json` records git commit and dirty flag, source hashes, runtime contract, artifact
hashes and sizes, and archive members. `checksums.txt` covers every file except itself.

## 4. Provider dry-runs

Dry-runs verify the release, stage fresh copies, print the plan and write a dry-run receipt. They
contact no provider and need no credentials.

```bash
uv run python scripts/publish.py huggingface --release dist/releases/gidi-finance-v2/2.0.0 --dry-run
uv run python scripts/publish.py github --release dist/releases/gidi-finance-v2/2.0.0 --dry-run
uv run python scripts/publish.py kaggle --release dist/releases/gidi-finance-v2/2.0.0 --dry-run
uv run python scripts/publish.py all --release dist/releases/gidi-finance-v2/2.0.0 --dry-run
```

Staging goes to `dist/staging/<provider>/<model>/<version>/` (`--staging-root`). Without
`--dry-run` the same command performs the real publication, after a credential check, but only
with the explicit `--confirm-publish` flag; without it (and without `--dry-run`) the command exits 1
before probing credentials or contacting any provider. The publish workflow adds the flag only when
`dry_run` is unticked.

## 5. Release CI

Workflow `release` (manual dispatch, `.github/workflows/release.yml`). Inputs: `release_version`
(required), `model_version` (default `gidi-finance-v2`).

- Runs on a self-hosted runner labelled `gidi-models`. Model binaries are gitignored, so they are
  not in the checkout. The runner must set `GIDI_MODELS_DIR` to a directory holding
  `gidi-finance-v2/` and `gidi-finance-v2-onnx/`; the job fails if it is unset.
- A step symlinks its entries into `models/`. The runner cannot substitute different bytes: the
  committed spec pins every input by sha256 and the build rejects mismatches.
- Steps: checkout exact sha, `uv sync --frozen`, link models, `build`, `verify`, `build --check`,
  `package --check`, upload artifact `release-<model>-<version>` (the release directory).
- It never publishes.

## 6. Publish CI

Workflow `publish` (manual dispatch, `.github/workflows/publish.yml`). Inputs: `release_run_id`
(the run ID of the `release` workflow run, required), `release_version` (required),
`model_version`, `providers` (`huggingface`, `github`, `kaggle`, `all`), `dry_run` (default
`true`).

- Downloads artifact `release-<model>-<version>` from that run by run ID, then verifies it before
  doing anything else (a failed verify stops the run).
- Run it from the same commit the release was built from (the manifest records it).
- Artifacts expire (90 days here, or sooner by repository policy). If the artifact is gone, rerun
  `release` from the same commit; the build is reproducible (see section 10). Use the new run ID.
- Uploads the receipts as artifact `publication-receipts-<model>-<version>-<run id>`.
- Always run with `dry_run: true` first. Untick it only for the real publication.

## 7. Required secrets

Repository secrets, passed only to the publish step:

| secret | provider |
|---|---|
| `HF_TOKEN` | Hugging Face (write access to the target repo) |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle (optional provider) |

GitHub uses the built-in `GITHUB_TOKEN` (passed as `GH_TOKEN`); the workflow grants
`contents: write` and `actions: read` (needed to download artifacts from another run). Secret
values are never printed.

## 8. Receipts

`dist/publications/<model>/<version>/<provider>.json` (real) or `<provider>.dry-run.json`
(`"dry_run": true`). Each records the manifest and checksums hashes, target, per-file hashes,
planned actions and whether auth was available. Locally they stay under `dist/`; in CI they are
the receipts artifact.

## 9. Model version vs release version

The model version (`gidi-finance-v2`) names the trained, frozen model. The release version
(`2.0.0`) names a packaging of it, semver `MAJOR.MINOR.PATCH[-prerelease]`. MAJOR must equal the
model's `vN`: `gidi-finance-v2` releases are `2.x.y`. A packaging-only fix (README, archive
layout) bumps MINOR or PATCH; a new model bumps MAJOR with a new model version.

## 10. Reproducing an older release

```bash
git checkout <commit-from-manifest.json>
# provision the same models dir (gidi-finance-v2/, gidi-finance-v2-onnx/); the spec pins sha256s
uv sync --frozen
uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0 --check   # vs existing dist/
# or rebuild to another directory and compare
uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.0 --out /tmp/rebuild
diff -r dist/releases/gidi-finance-v2/2.0.0 /tmp/rebuild
```

Compare `checksums.txt`. Gzip output depends on the zlib version, so archive bytes can differ
between machines. The release directory and its `manifest.json` are the source of truth; the
non-archive files must always match.

## 11. Rules and open items

- Publish never rebuilds, exports, quantizes or trains, and never modifies the release. It
  verifies the release, snapshots it, and fails if the snapshot changes.
- The Hugging Face repo (`x8bu/gidi-finance`, one repo for all model versions, one tag per
  release such as `gidi-finance-v2-2.0.0`) defaults to private (`private: true`) and gets the
  INT8 build only (`exclude_kinds: ["onnx-fp32"]`). The FP32 reference ships as the `-fp32`
  archive on the GitHub release.
- The GitHub release is created as a draft.
- License: Apache-2.0 (`releases/licenses/Apache-2.0.txt`, shipped as `LICENSE`; the README
  front matter declares `license: apache-2.0`).
