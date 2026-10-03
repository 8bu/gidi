# Experiments

Records of what was tried and what was learned. One directory (or one markdown file) per
experiment, dated, self-contained.

## Convention

Name it `<YYYY-MM-DD>-<short-slug>`, for example `2026-09-29-tokenizer-audit-phobert`. Inside,
record these sections, in this order:

- **Question** — the single thing the experiment answers. If it cannot be answered by the
  measurement described, it is not an experiment.
- **Setup** — data used (path and whether raw, reviewed, or derived), tokenizer or model
  candidate, device, versions, and any configuration that affects the result.
- **Commands** — the exact commands run, copy-pasteable.
- **Results** — raw numbers and observations, including the ones that contradict the
  hypothesis. Link or embed the artifacts rather than describing them loosely.
- **Conclusion** — what is now believed, and what it implies for the next step. If the result
  is inconclusive, say so.

Keep it short. The point is that a past decision can be re-checked without re-running anything.

## First expected experiments

Tokenizer audits. Before committing to an encoder or building a labeled dataset, audit the
candidate tokenizers on the reviewed corpus: how Vietnamese text is segmented, how much of the
vocabulary the reviewed notes actually use, per-token fragmentation of finance terms and
amounts such as `45k` and `15tr`, and how much the resulting sequences cost in tokens. Tokenizer
choice constrains everything downstream, so it is measured first.

Experiment records are notes and small artifacts, not model checkpoints. Local model files
belong in `models/`, which is gitignored.

## Tokenizer audit

`scripts/audit_tokenizers.py` measures candidate tokenizers on a JSONL corpus before an encoder
is chosen. Per tokenizer it reports vocabulary size, tokens per note (mean/p50/p95/max), tokens
per word, the share of words fragmented into more than one token, `[UNK]` rate plus which
characters collapse to `[UNK]` (diacritic coverage), exact-match encode/decode round trips with
failing examples, and the token cost of amount expressions — `AMOUNT_PATTERN` in
`src/gidi/tokenizer/audit.py` covers `45k`, `50K`, `45.000đ`, `1tr5`, `1,5tr`, `200 nghìn`,
`200 ngàn`, `2 triệu`, `500đ`.

```bash
uv run python scripts/audit_tokenizers.py \
    --corpus corpus/reviewed/<file>.jsonl \
    --tokenizer vinai/phobert-base FPTAI/vibert-base-cased \
    --out experiments/<YYYY-MM-DD>-tokenizer-audit/report.json --examples 20
```

Tokenizer arguments go to `AutoTokenizer.from_pretrained`, so Hub names download on the first run
(the only network access in this workflow) and local directories are used as is. An existing
`--out` file is refused unless `--overwrite` is passed; the JSON is the artifact, the printed
table is the comparison.

Candidate names — verify each on the Hub, they are not hardcoded defaults:

- `vinai/phobert-base` — Vietnamese BPE, but it expects word-segmented input (syllables joined
  with `_`). Gidi notes are written unsegmented, so auditing raw notes understates it; record the
  segmented comparison separately.
- `FPTAI/vibert-base-cased` — Vietnamese, no word segmentation required.
- `bert-base-multilingual-cased`, `xlm-roberta-base`, `google/mt5-small` — multilingual
  references with very different vocabularies.
- `microsoft/Multilingual-MiniLM-L12-H384` — small multilingual encoder, an on-device candidate.
