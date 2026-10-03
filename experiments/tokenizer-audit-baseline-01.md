# Tokenizer audit — baseline-01

Machine-readable report: [`tokenizer-audit-baseline-01.json`](tokenizer-audit-baseline-01.json).
No model was trained.

## Setup

- **Corpus:** `corpus/reviewed/baseline-01.jsonl`. This is the Quet export, with all 1,000 records
  approved: 258 checked by a human, and 742 accepted from AI suggestions without a per-record
  check. Raw and rejected records were not used.
- **Command:**
  ```bash
  uv run python scripts/audit_tokenizers.py --corpus corpus/reviewed/baseline-01.jsonl \
      --tokenizer vinai/phobert-base bert-base-multilingual-cased FacebookAI/xlm-roberta-base \
        models/tokenizers/BamiBERT \
      --out experiments/tokenizer-audit-baseline-01.json
  ```
- **PhoBERT input:** notes were fed raw, without word segmentation (see tradeoffs).
- **BamiBERT loading:** `Qualcomm-AI-Research/BamiBERT` declares `XLMRobertaTokenizer` in
  `tokenizer_config.json`, but its `tokenizer.json` is byte-level BPE, and Transformers 5.17
  fails to load it. `models/tokenizers/BamiBERT` is the same `tokenizer.json` re-saved as
  `PreTrainedTokenizerFast`. The vocabulary and merges are unchanged, and the round trip is
  100%.

## Candidate 4: why BamiBERT

| Candidate | Vietnamese | Params (embeddings) | Vocab | License | Notes |
|---|---|---|---|---|---|
| **Qualcomm-AI-Research/BamiBERT** | native, 129 GB vi corpus, raw input (no segmenter) | 103M (15.7M) | 20,480 | BSD-3-Clause-Clear + Qualcomm Responsible AI license | RoBERTa arch (easy ONNX); SOTA among base vi encoders (arXiv 2607.02259, Jul 2026) |
| jhu-clsp/mmBERT-small | multilingual | ~140M, embedding-heavy | ~256k | MIT | modern, but its vocabulary is 12× BamiBERT's |
| Fsoft-AIC/videberta-xsmall | native | 22M backbone + 48M embeddings | 128k | none stated on the card | smallest backbone; licence unclear, so rejected |
| microsoft/Multilingual-MiniLM-L12-H384 | multilingual | ~118M, mostly embeddings | 250k | MIT | same tokenizer as XLM-R, so it adds no new tokenizer data |

BamiBERT is the only candidate with a Vietnamese-native vocabulary that is also small (20k), does
not require word segmentation, and has an open licence. Its whole model is still base-sized: 12
layers × 768 hidden. What makes it attractive is the vocabulary, which is cheap to carry into a
distilled student. **Licence caveat:** the Qualcomm Responsible AI license adds use restrictions
on top of BSD. Read it before shipping anything derived from the model.

## Aggregate results

| tokenizer | vocab | tok/note | p50 | p95 | fertility | % words split | UNK | round trip | tok/amount | tok/slang word | % slang split | emb. params @768 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PhoBERT (unsegmented) | 64,000 | 7.42 | 7 | 10 | 1.40 | 31.9 | 0.0% | 100% | 2.30 | 1.88 | 64.0 | 49M |
| mBERT cased | 119,547 | 8.23 | 8 | 12 | 1.56 | 44.5 | 0.0% | 91.1% | 2.36 | 1.92 | 70.5 | 92M |
| XLM-R base | 250,002 | 7.43 | 7 | 10 | 1.41 | 34.9 | 0.0% | 100% | 2.22 | 1.81 | 69.4 | 192M |
| BamiBERT | 20,480 | 7.99 | 8 | 11 | 1.51 | 65.0* (≈34.9) | 0.0% | 100% | 2.36 | 2.13 | 80.2 | 16M |

- **Counts:** 957 amount expressions (regex now includes `N củ`) and 5,283 words.
- **Slang metric:** covers words from the spec's shorthand list (`ck`, `cf`, `vs`, `ko`, `tk`,
  `thg`, `ship`, `momo`, `grab`, ...).
- **\* BamiBERT's word-split figure is inflated.** The audit tokenizes each word on its own,
  without a leading space. BamiBERT's byte-level vocabulary stores word-initial pieces as
  `Ġword`, so a bare word fragments more than it does inside a sentence. Re-measured with a
  leading space, 34.9% of words split, the same as XLM-R. Per-note token counts are not affected
  by this and are the fair comparison.

## Domain probes

Tokens are shown decoded; `·` marks a word-initial piece that carries a leading space. **No
tokenizer produced UNK on any probe.**

| probe | PhoBERT | mBERT | XLM-R | BamiBERT |
|---|---|---|---|---|
| `1tr5` | 1@@ tr@@ 5 (3) | 1 ##tr ##5 (3) | 1 tr 5 (3) | 1 tr 5 (3) |
| `2tr` | 2@@ tr (2) | 2 ##tr (2) | 2 tr (2) | 2 tr (2) |
| `2 củ` | 2 củ (2) | 2 c ##ủ (3) | 2 ·củ (2) | 2 ·củ (2) |
| `5 lít` | 5 lít (2) | 5 l ##ít (3) | 5 ·lí t (3) | 5 ·lít (2) |
| `45k` | 45@@ k (2) | 45 ##k (2) | 45 k (2) | 45 k (2) |
| `45.000đ` | 45@@ .000@@ đ (3) | 45 . 000 ##đ (4) | 4 5.000 đ (3) | 45 . 000đ (3) |
| `ck` | ck (1) | c ##k (2) | ▁ ck (2) | ck (1) |
| `thg` | th@@ g (2) | th ##g (2) | th g (2) | th g (2) |
| `tien nha` | ti@@ en nha (3) | tien n ##ha (3) | tien ·nha (2) | t ien ·nha (3) |
| `an trua` | an tr@@ ua (3) | an tr ##ua (3) | an ·tru a (3) | an ·tr ua (3) |
| `shope` | sho@@ pe (2) | shop ##e (2) | shop e (2) | sh ope (2) |
| `refund` | ref@@ und (2) | ref ##und (2) | refund (1) | ref und (2) |
| `refun` | re@@ fun (2) | ref ##un (2) | refu n (2) | ref un (2) |
| `grab` | gra@@ b (2) | gra ##b (2) | grab (1) | gr ab (2) |
| `khach tra not` | kha@@ ch tra not (4) | k ##ha ##ch tra not (5) | kha ch ·tra ·not (4) | kh ach ·tra ·not (4) |

Observations:

- **Compact amounts:** every tokenizer splits `1tr5`, `2tr` and `45k` the same way, into number
  + unit + number. Amounts cost about 2.2–2.4 tokens each everywhere, so amount handling does not
  separate the candidates. The classifier has to learn the `tr` / `k` unit tokens whatever
  tokenizer is chosen.
- **Formatted amounts:** `45.000đ` breaks into an inconsistent number piece in every tokenizer.
  XLM-R even splits it as `4 | 5.000`, which destroys the digit grouping. mBERT also fails the
  round trip on formatted amounts (`80.000đ` → `80. 000đ`), which is where its 91% round-trip
  score comes from.
- **Unaccented Vietnamese** (`tien nha`, `an trua`, `khach tra not`) fragments in all four.
  Vietnamese vocabularies (PhoBERT, BamiBERT) are trained on accented text, so unaccented
  syllables become sub-syllable pieces. Around 40% of the corpus is unaccented, so this matters
  more than any single slang item.
- **Brands and loanwords:** only XLM-R keeps `grab` and `refund` whole, thanks to its large
  multilingual vocabulary. The Vietnamese vocabularies split them into two pieces.
- **Vietnamese syllables and units:** PhoBERT and BamiBERT keep `ck`, `củ` and `lít` whole.
  mBERT splits Vietnamese syllables into characters (`c ##ủ`, `l ##ít`).

## Tradeoffs

- **Word segmentation (PhoBERT):** PhoBERT was pretrained on text segmented with VnCoreNLP
  (`tiền_nhà`). Its numbers here come from unsegmented input, so they undersell the model:
  its embeddings never saw raw syllable sequences. Using it properly means shipping a Java
  segmenter (or a port) in the pipeline and on the device, and segmenters trained on formal text
  are unreliable on unaccented teencode. XLM-R, mBERT and BamiBERT take raw input.
- **Vocabulary cost on mobile:** at 768 dimensions the embedding matrix alone is 192M parameters
  for XLM-R and 92M for mBERT, against 49M for PhoBERT and 16M for BamiBERT. A tiny student
  (e.g. 256–384 hidden) with the XLM-R vocabulary would still spend most of its size on
  embeddings. Most of those 250k rows are never used by Vietnamese finance notes, so XLM-R is only
  practical with vocabulary pruning, which in turn costs compatibility with the teacher.
- **Compatibility with pretrained encoders:** each tokenizer only makes sense together with its
  own pretrained encoder, and no candidate should be combined with a different encoder. Of the
  four, only BamiBERT pairs a native Vietnamese encoder with a small vocabulary, so it is the
  cleanest teacher, or tokenizer donor, for a distilled student. XLM-R shares its tokenizer with
  Multilingual-MiniLM (a smaller student already exists), but at the embedding cost above.
- **Is it practical for a tiny final model?**
  - **BamiBERT:** yes. Its 20k vocabulary, raw input and RoBERTa architecture keep ONNX export
    simple, at the cost of about 0.5 extra tokens per note (8.0 against 7.4).
  - **XLM-R and MiniLM:** only with vocabulary pruning.
  - **PhoBERT:** only if on-device segmentation is acceptable.
  - **mBERT:** no. It has more tokens, a large vocabulary, lossy round trips on amounts, and
    character-level Vietnamese.

## Summary

The token counts barely differ: 7.4–8.2 tokens per note, with p95 between 10 and 12. None of the
four produces UNK, and none handles amounts better than the others. The choice should therefore be
made on vocabulary cost, segmentation requirements and licence, not on aggregate token count.
**BamiBERT** fits a mobile target best, provided its licence is acceptable. **XLM-R/MiniLM** is
the strongest multilingual fallback, but needs vocabulary pruning. **PhoBERT** stays as the
segmented Vietnamese reference. **mBERT** can be dropped.

## Caveats

- One generated batch of short notes. 742 of its records were approved from AI suggestions, not
  checked per record by a human.
- PhoBERT was not re-run with VnCoreNLP segmentation.
- The per-word split metric is biased against byte-level BPE vocabularies (see above).
