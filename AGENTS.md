# AGENTS.md

Gidi is a tiny on-device classifier for short Vietnamese personal-finance notes. It predicts the
transaction type and the counterparty span. Python 3.12 via `uv`.

## Ground rules

- After code changes, run `uv run pytest -q`, `uv run ruff check .`, and `uv run ruff format .`.
  The line length is 100.
- Never modify `corpus/raw/`, `corpus/reviewed/`, or Quet review state (`*.quet.db`) unless
  explicitly asked.
- Never auto-generate or overwrite human labels.
- The annotation contract lives in `configs/annotation-v1.yaml` and `docs/annotation-v1.md`.
  Changing the taxonomy or its meaning means a new annotation version.
- `datasets/probe-*` are diagnostic, evaluation-only sets. Never train on them, and never copy
  or paraphrase them into augmentation. The held-out test split
  (`datasets/annotation-v1/splits/test.jsonl`) is likewise never a source for training data.

## Investigation journal

`docs/investigation-journal.md` is the single canonical, compact record of major model/data
investigations (question → experiment → result → decision → next direction).

- Update it only when explicitly told to (e.g. "update the investigation journal", "record this
  in the journal"). Routine work and new experiments do not update it automatically.
- When updating: read the journal and the relevant report first, append or refine only the
  needed section, bump "Last updated", and keep the "Current status" line accurate.
- Detailed metrics belong in `experiments/*/report.md`, not the journal.
- History is append-only: never silently rewrite earlier decisions; record what a later finding
  changed. Do not keep copies of the journal elsewhere.

## Quet integration

Quet (Quick Utility for Evaluating Text) is Gidi's canonical human-in-the-loop tool for corpus
review and annotation.

Repository: https://github.com/8bu/quet

Quet is a separate open-source project and must stay outside Gidi.

**Gidi owns:**
- corpus schemas, ingestion and validation;
- the annotation taxonomy, its semantic rules, and the annotation config;
- annotation queue generation and annotation validation;
- consistency analysis and re-check candidate generation;
- building the train/validation/test splits;
- model training, evaluation and export.

**Quet owns:**
- the review workflow: interactive corpus review, the approve / reject / needs-review UX, manual
  corpus edits during review, and manual flags;
- the annotation workflow: the annotation UI, target-span selection, revisiting and correcting
  existing labels, and persisting annotations;
- keyboard navigation and workflow UX.

**Boundaries:**
- Do not vendor Quet into Gidi, and do not add it as a Python dependency.
- Do not build a second review or annotation TUI inside Gidi.
- Do not move Gidi's taxonomy or semantic rules into Quet. Gidi's schema and config files are
  the source of truth; Quet consumes the queues and schemas Gidi generates.
- Human annotations are authoritative. Automated analysis may flag suspicious labels, but it
  must never silently overwrite them.
- For Quet behavior or bugs, point to the external repository instead of implementing the
  feature in Gidi.

`quet annotate` has two modes. Pick by whether the queue owns the labels file:

- **`--out` (strict, normal annotation):** the labels file belongs to the active queue, and
  labels whose ids are outside the queue are rejected. Use it for full annotation jobs.
- **`--labels` (merge, re-check):** the labels file must already exist, and the queue may be a
  subset of it. Quet loads the existing annotations for the subset ids, edits only those, and
  preserves every other label. Use it for re-check queues. Never point `--out` at the canonical
  labels file with a subset queue.
- **`--proposals` (advisory, with `--labels`):** pre-fills the editor from a proposals file
  (e.g. `ai-proposals.jsonl`). Proposals are never labels; only the reviewer's confirmed or edited
  decision is saved. After such a pass, mark the reviewed ids in `provenance.jsonl` as
  `{"annotator":"human","reviewed":"ai-proposal","proposal_confidence":…,"proposal_kept":…}`.

Normal annotation:

```sh
quet annotate datasets/annotation-v1/queue.jsonl \
  --schema configs/annotation-v1.yaml \
  --out datasets/annotation-v1/labels.jsonl
```

Re-check (current pilot pass; queue built by `scripts/build_recheck_queue.py`):

```sh
quet annotate datasets/annotation-v1/recheck-pilot-01.jsonl \
  --schema configs/annotation-v1.yaml \
  --labels datasets/annotation-v1/labels.jsonl
```

To validate after annotating:

```sh
uv run python scripts/validate_annotations.py datasets/annotation-v1/labels.jsonl
```

The workflow:

```
raw generation
  -> Gidi ingest + validate
  -> Quet review -> reviewed corpus (corpus/reviewed/)
  -> Gidi builds annotation queue (scripts/build_annotation_queue.py)
  -> Quet annotate
  -> Gidi validates + analyzes labels (e.g. experiments/annotation-v1-pilot.md)
  -> Gidi generates re-check candidates + subset queue (scripts/build_recheck_queue.py)
  -> Quet manual correction (quet annotate <subset-queue> --labels <canonical labels>)
  -> training dataset
```
