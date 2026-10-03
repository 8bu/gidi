# Encoder shortlist

Date: 2026-09-29. This is research only: nothing was trained and no model weights were downloaded.
I only downloaded tokenizer files, to measure how each candidate tokenizes the approved corpus.

## Method

- **Model metadata:** taken from the Hugging Face Hub API (`/api/models/<id>?blobs=true`) and
  each model's own `config.json`, not from model-card claims. That covers the licence tag, last
  change date, weight file sizes, layers, hidden size and vocabulary size.
- **Parameter counts:** read from the Hub's `safetensors.total` field where it exists. Otherwise
  they are computed as weight file size ÷ 4 bytes for FP32 checkpoints, and marked `≈`.
- **Licences:** checked against the model card, the licence of the authors' GitHub repository,
  and, for BamiBERT, the full Qualcomm licence text.
- **Tokenization:** every candidate that is not a duplicate was run through the
  existing `scripts/audit_tokenizers.py` on `corpus/reviewed/baseline-01.jsonl` (1,000 notes).
  PhoBERT, mBERT, XLM-R and BamiBERT figures come from
  [tokenizer-audit-baseline-01](tokenizer-audit-baseline-01.md). ViSoBERT, ViDeBERTa-xsmall and
  mmBERT-small were audited for this shortlist with the same script.
- **Size estimates:** 4 bytes per parameter for FP32, 2 for FP16, and about 1 for INT8 with
  weights-only dynamic quantization. Real files add a few MB of overhead.

## The size target

No pretrained encoder is anywhere near the long-term 10–30 MB INT8 target. The smallest
realistic candidate is about 100 MB in INT8. **The shipped model will be a distilled student**,
so what matters in a candidate is:

1. whether it is a good **teacher** on noisy Vietnamese finance notes; and
2. whether its **tokenizer** can be carried over to the student.

The vocabulary decides how small the student can be. A student with 4 layers × 256 hidden units
has about 3.2M transformer parameters. Its embedding table adds 5M parameters with a 20k
vocabulary, but 64M with a 250k vocabulary. The first gives a student of about 9 MB in INT8; the
second about 67 MB. A Vietnamese tokenizer with 15–20k entries is therefore worth more than a few
points of teacher accuracy.

## Comparison

| Model | Params | Layers | Hidden | Vocab | Raw VN | License | FP32 / FP16 / INT8 est. | Mobile fit | Notes |
|---|---|---|---|---|---|---|---|---|---|
| [Qualcomm-AI-Research/BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) | 103.0M | 12 | 768 | 20,481 | yes | BSD-3-Clause-Clear **+ Qualcomm RAIL** → *commercial with conditions* | 412 / 206 / 103 MB | good teacher, ideal tokenizer donor | RoBERTa arch; byte-level BPE; 7.99 tok/note, 100% round trip; released 2026-07 by the PhoBERT authors ([arXiv 2607.02259](https://arxiv.org/abs/2607.02259)); ships `tokenizer_class: XLMRobertaTokenizer` but a BPE `tokenizer.json` (Transformers 5.17 needs `PreTrainedTokenizerFast`) |
| [uitnlp/visobert](https://huggingface.co/uitnlp/visobert) | ≈97.6M | 12 | 768 | 15,004 | yes | HF card: none; authors' [GitHub](https://github.com/qnamng/ViSoBERT) MIT → *unclear, needs legal review* | 390 / 195 / 98 MB | good teacher, smallest vocab | XLM-R arch + SentencePiece; **pretrained on Vietnamese social media** ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.315/)); 9.84 tok/note but slang is cheap (1.45 tok/shorthand word; `ck`, `thg` are single tokens); `.bin` only; stable since 2024-06 |
| [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384) | ≈117.7M | 12 | 384 | 250,037 | yes | MIT → *clearly suitable* | 470 / 235 / 118 MB | body tiny (21M), embeddings 96M | BERT body + XLM-R SentencePiece tokenizer (card says load with `XLMRobertaTokenizer`); 7.43 tok/note; stable/unchanged since 2022; strong ONNX tooling |
| [vinai/phobert-base](https://huggingface.co/vinai/phobert-base) | ≈135M | 12 | 768 | 64,001 | **no** (needs VnCoreNLP word segmentation) | MIT → *clearly suitable* | 540 / 270 / 135 MB | poor as-is (Java segmenter) | 7.42 tok/note even unsegmented; max 256 positions (fine for notes); card now points users to BamiBERT |
| [Fsoft-AIC/videberta-xsmall](https://huggingface.co/Fsoft-AIC/videberta-xsmall) | ≈70M (card: 22M body + 48M emb) | 12 | 384 | 128,000 | yes | HF card: none; [GitHub](https://github.com/HySonLab/ViDeBERTa) MIT (code) → *unclear* | 280 / 140 / 70 MB | smallest body, but DeBERTa attention is awkward on mobile | best token efficiency measured (6.76 tok/note, 22.9% words split, `2tr` one token); 78 downloads, no updates since 2023-03 |
| [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | ≈141M | 22 | 384 | 256,000 | yes | MIT → *clearly suitable* | 564 / 282 / 141 MB | poor: large vocab, 22 layers | ModernBERT, 2025; **digit-level numbers** (`45.000đ` → 8 tokens, 4.50 tok/amount); 9.61 tok/note |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 117.7M | 12 | 384 | 250,037 | yes | MIT → *clearly suitable* | 470 / 235 / 118 MB | same as MiniLM | Multilingual-MiniLM retrained contrastively for embeddings; ships official ONNX incl. INT8 (118 MB); actively updated (2026-04) |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 278.9M | 12 | 768 | 250,002 | yes | MIT → *clearly suitable* | 1,116 / 558 / 279 MB | teacher/reference only | 192M of the 279M parameters are embeddings |

Licence classes, as requested:

- **Clearly suitable for commercial use:** PhoBERT-base (MIT), Multilingual-MiniLM (MIT),
  multilingual-e5-small (MIT), mmBERT (MIT), XLM-R (MIT).
- **Commercial use with conditions:** BamiBERT (BSD-3-Clause-Clear + Qualcomm RAIL, see below).
- **Unclear, needs legal review:** ViSoBERT and ViDeBERTa. The Hub model cards declare no licence.
  The authors' GitHub repositories are MIT, but it is not stated anywhere that the MIT licence
  covers the released weights.
- **Unsuitable:** see Excluded.

### BamiBERT licence detail

The [model card](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) states: "released under the
BSD 3-Clause Clear license and the Qualcomm responsible AI license". The
[Qualcomm RAIL](https://www.qualcomm.com/site/responsible-ai-license) (revision of 2026-02-17)
applies to the models themselves and to "any quantized versions and/or derivatives thereof".
A distilled student therefore inherits it. Summary of the parts that matter:

- **§1(b), prohibited uses (20 items):**
  - military use;
  - criminal use and predictive policing;
  - uses that violate law or third-party licences;
  - exploiting or harming people;
  - generating false information, or undisclosed machine-generated content;
  - "fully automated decision making that adversely impacts an individual's legal rights";
  - social scoring;
  - discrimination;
  - exploiting vulnerable groups;
  - biometric uses and emotion recognition.
- **§3, high-risk applications:** these include "access to and enjoyment of essential private
  services … and benefits" and "any other AI system that … makes, or is a substantial factor in
  making, a consequential decision".
- **§4, indemnity:** if you use the model for a high-risk application (or violate the licence),
  you must indemnify Qualcomm.
- **§2, monitoring:** "Qualcomm reserves the right to monitor your account and your use of the
  Resources".

Gidi's use (a private, on-device classifier of the user's own notes) does not match any
prohibited use. Two things still need legal review before production:

- **Finance features:** if Gidi's output ever feeds credit, lending or eligibility decisions,
  §3 applies.
- **Inheritance:** every derivative must carry the RAIL terms, including a distilled student.

## Recommended benchmark set

The candidates are not ranked. Each one tests a different hypothesis.

1. **BamiBERT: the strongest Vietnamese-specific baseline that takes raw text.**
   - It is a native Vietnamese encoder from the PhoBERT group that needs no segmenter.
   - Its 20k byte-level BPE vocabulary never produces UNK and is the right size for a student.
   - The RoBERTa architecture exports to ONNX cleanly.
   - **Risk:** the RAIL conditions travel with every derivative.
2. **ViSoBERT: pretraining on noisy text in the same domain.**
   - It is the only candidate pretrained on Vietnamese social-media text, the closest register to
     `ck mẹ 2tr` and `khach tra not 8cu`.
   - Its 15k vocabulary is the smallest shortlisted, and it keeps teencode whole: `ck` and `thg`
     are single tokens, at 1.45 tokens per shorthand word against 1.8–2.1 for the others.
   - It pays for that on unaccented words and English loanwords: 9.8 tokens/note.
   - The benchmark shows whether pretraining on this register beats token efficiency.
   - **Blocker for production:** the weights licence must be confirmed with the authors.
3. **Multilingual-MiniLM-L12-H384: the multilingual fallback with the clearest licence and best
   deployment tooling.**
   - MIT licence, a 384-hidden body of 21M parameters, and mature ONNX and INT8 tooling (a sibling
     model, e5-small, ships official INT8 ONNX at 118 MB).
   - It keeps English merchant and platform names whole (`grab`, `refund`).
   - It is the safe baseline if both Vietnamese candidates fail on licence.
   - Getting it to the target size requires pruning its 250k vocabulary.
4. **PhoBERT-base (optional reference): the established Vietnamese baseline, MIT.**
   - Run it twice, on raw notes and on VnCoreNLP-segmented notes. That measures what segmentation
     is worth on this data, which decides whether the Java preprocessing step ever needs to exist.
   - Keep it as a teacher and reference only; it is not a candidate for shipping.

Together these cover native vs. multilingual pretraining, general vs. social-media text, and
three licence situations: conditions, unclear, and MIT.

## Excluded candidates

| Model | Reason |
|---|---|
| [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2) | **AGPL-3.0**, unsuitable for a closed mobile app; still needs segmentation |
| [FPTAI/vibert-base-cased](https://huggingface.co/FPTAI/vibert-base-cased) | no licence; strips diacritics despite "cased" (32% round trip in the earlier audit); stale since 2021 |
| [NlpHUST/vibert4news-base-cased](https://huggingface.co/NlpHUST/vibert4news-base-cased) | no licence; news-domain; stale since 2023 |
| [Fsoft-AIC/videberta-xsmall](https://huggingface.co/Fsoft-AIC/videberta-xsmall) | unclear weights licence; almost no usage (78 downloads) and no maintenance; DeBERTa-v2 relative attention exports to ONNX but is costly and fiddly on mobile runtimes. Tokenizer is the most efficient measured, so revisit only if its licence is clarified |
| [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 256k vocab plus 22 layers; splits numbers into single digits (4.5 tok/amount), bad for numeric-heavy notes |
| [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384) siblings: [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | same architecture and tokenizer as the shortlisted MiniLM; they are tuned for sentence embeddings, not token-level span extraction. e5-small is the drop-in swap if MiniLM underperforms |
| [nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large](https://huggingface.co/nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large) | no licence on the upload (Microsoft's [unilm](https://github.com/microsoft/unilm) repo is MIT, but provenance of the upload is unstated); same 250k vocab, so still ~107 MB INT8 |
| [distilbert-base-multilingual-cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased) | mBERT tokenizer (character-level Vietnamese, lossy amount round trip) |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 279M params (69% embeddings); useful only as a teacher, and MiniLM covers the same tokenizer |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | 305M; custom architecture (`model_type: new`, remote code), a poor fit for mobile export |
| [uitnlp/CafeBERT](https://huggingface.co/uitnlp/CafeBERT) | 560M (XLM-R-large base); teacher-only at best |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | Gemma licence and gated access; 300M; derived from a decoder |

## Next experiment

This is the smallest fine-tuning experiment that can tell the shortlisted candidates apart. It has
not been implemented.

1. **Data prerequisite: labels.** The corpus has none yet.
   - Label about 600 of the approved notes with transaction type (the spec's six categories) and
     counterparty spans.
   - Split 70/15/15, stratified by type and by whether the note has diacritics, so the
     unaccented 40% is measured separately.
   - Freeze the split, so every candidate sees identical data.
2. **Task heads:** the same two heads on every encoder: a sequence classifier from the pooled first
   token, and a BIO token tagger for spans. Train them jointly. Multi-label tags wait until type
   and spans work.
3. **Training budget:**
   - max_length 32 (p95 is 10–13 tokens);
   - learning rate chosen from {2e-5, 5e-5}, 10 epochs with early stopping on the dev split;
   - 3 seeds per configuration;
   - MPS device.
4. **Encoders:** BamiBERT, ViSoBERT and Multilingual-MiniLM. Add PhoBERT on raw and on segmented
   input if the time is available.
5. **Metrics:**
   - macro-F1 for type;
   - exact-match span F1 for counterparty;
   - both split into accented and unaccented notes;
   - the mean and spread across seeds;
   - for every model, FP32 ONNX export success, INT8 file size and CPU latency per note, to check
     early that it can be deployed.
6. **Decision rule:**
   - If the Vietnamese candidates do not clearly beat MiniLM on unaccented notes and span F1, the
     vocabulary argument alone decides the choice of teacher.
   - Either way, the winning tokenizer becomes the student's tokenizer for the distillation step.

## Sources

- Hub model API and `config.json` for every model listed above (links in the tables), fetched
  2026-09-29.
- Qualcomm Responsible AI License, revision 2026-02-17:
  <https://www.qualcomm.com/site/responsible-ai-license>
- BamiBERT paper: <https://arxiv.org/abs/2607.02259>
- ViSoBERT paper: <https://aclanthology.org/2023.emnlp-main.315/>; code and licence:
  <https://github.com/qnamng/ViSoBERT> (MIT, © 2023 Nguyen Quoc Nam)
- ViDeBERTa code and licence: <https://github.com/HySonLab/ViDeBERTa> (MIT)
- MiniLM code and licence: <https://github.com/microsoft/unilm> (MIT)
- Tokenizer measurements: [tokenizer-audit-baseline-01.md](tokenizer-audit-baseline-01.md), and
  `scripts/audit_tokenizers.py` run on `uitnlp/visobert`, `Fsoft-AIC/videberta-xsmall` and
  `jhu-clsp/mmBERT-small`.
