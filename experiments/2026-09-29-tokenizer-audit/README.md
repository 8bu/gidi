# 2026-09-29 tokenizer audit

## Question

Which pretrained tokenizer candidates encode Gidi's short Vietnamese finance notes losslessly
and cheaply enough to carry forward as encoder baselines?

## Setup

- Data: `corpus/reviewed/baseline-01.jsonl` (reviewed; 1,000 Claude-generated notes, all
  approved in Quet — 258 by a human, 742 accepted from AI suggestions without per-record check).
- Candidates: `vinai/phobert-base`, `FPTAI/vibert-base-cased`, `bert-base-multilingual-cased`,
  `xlm-roberta-base`, `google/mt5-small`, `microsoft/Multilingual-MiniLM-L12-H384`.
- PhoBERT was fed raw, unsegmented notes (no `_` word segmentation), which understates it.
- `sentencepiece` and `protobuf` supplied ad hoc via `uv run --with` (mt5 needs them; not
  project dependencies).

## Commands

```bash
quet export corpus/raw/baseline-01.jsonl -o corpus/reviewed/baseline-01.jsonl --with-review
uv run --with sentencepiece --with protobuf python scripts/audit_tokenizers.py \
    --corpus corpus/reviewed/baseline-01.jsonl \
    --tokenizer vinai/phobert-base FPTAI/vibert-base-cased bert-base-multilingual-cased \
      xlm-roberta-base google/mt5-small microsoft/Multilingual-MiniLM-L12-H384 \
    --out experiments/2026-09-29-tokenizer-audit/report.json --examples 20
```

## Results

Full report: [`report.json`](report.json).

| tokenizer | vocab | tok/note | p95 | tok/word | % words split | unk% | round-trip % | tok/amount |
|---|---|---|---|---|---|---|---|---|
| vinai/phobert-base | 64,000 | 7.42 | 10 | 1.40 | 31.9 | 0.0 | 100.0 | 2.30 |
| FPTAI/vibert-base-cased | 38,168 | 8.38 | 12 | 1.59 | 49.1 | 0.0 | 32.0 | 2.39 |
| bert-base-multilingual-cased | 119,547 | 8.23 | 12 | 1.56 | 44.5 | 0.0 | 91.1 | 2.36 |
| xlm-roberta-base | 250,002 | 7.43 | 10 | 1.41 | 34.9 | 0.0 | 100.0 | 2.22 |
| google/mt5-small | 250,100 | 9.66 | 15 | 1.83 | 64.5 | 0.0 | 100.0 | 2.25 |
| Multilingual-MiniLM-L12-H384 | 250,002 | 7.43 | 10 | 1.41 | 34.9 | 0.0 | 100.0 | 2.22 |

948 amount expressions matched. Observed round-trip failures:

- viBERT strips diacritics and lowercases despite the `-cased` name
  (`bún bò sáng nay 40k` → `bun bo sang nay 40k`) and splits formatted amounts
  (`80.000đ` → `80. 000đ`).
- mBERT splits formatted amounts on decode (`347,000` → `347, 000`).

## Conclusion

- viBERT is out: lossy on diacritics, which carry meaning in Vietnamese.
- mBERT is weaker (91% round trip, more splitting) with no offsetting advantage.
- PhoBERT, XLM-R, and Multilingual-MiniLM tie on cost (~7.4 tokens/note, p95 10) and are lossless.
  MiniLM shares XLM-R's tokenizer and is the smallest encoder, so it is the leading on-device
  candidate; PhoBERT stays in as the Vietnamese-specific reference.
- mT5 is lossless but ~30% more tokens; drop unless needed.
- Caveats: one generated batch, short notes, PhoBERT unsegmented. Re-run on later batches before
  committing.
