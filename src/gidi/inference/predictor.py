"""``GidiPredictor``: text in, type + target span (+ value span) out, using only the bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gidi.inference.bundle import MODEL_FILE, TOKENIZER_FILE, BundleConfig, load_config
from gidi.inference.crf import CRFTransitions, viterbi_masked
from gidi.inference.decode import (
    TAG_O,
    DecodedSpan,
    decode_first_span,
    decode_value_crf,
    snap_span_to_words,
    target_confidence,
    type_prediction,
)
from gidi.inference.runner import OnnxRunner
from gidi.inference.schema import EmptyInputError, Prediction, RawOutput, ValuePrediction
from gidi.inference.text import NormalizedText, normalize_nfc
from gidi.inference.tokenizer import BundleTokenizer, TokenizedText


class GidiPredictor:
    """Single-note CPU classifier. Deterministic: one thread, no sampling, no state."""

    def __init__(
        self,
        config: BundleConfig,
        tokenizer: BundleTokenizer,
        runner: OnnxRunner,
        *,
        snap_words: bool = False,
    ) -> None:
        # ``snap_words`` extends the decoded target span to whole-word boundaries (off by
        # default: the deployed web parity is unchanged).
        self._snap_words = snap_words
        self._config = config
        self._tokenizer = tokenizer
        self._runner = runner
        self.model_version: str = config.model_version
        self.has_value_head: bool = config.value_labels is not None
        self._crf: CRFTransitions | None = None
        if config.value_decoding is not None:
            decoding = config.value_decoding
            self._crf = CRFTransitions(
                np.asarray(decoding.start, dtype=np.float64),
                np.asarray(decoding.end, dtype=np.float64),
                np.asarray(decoding.transitions, dtype=np.float64),
            )

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        model_path: str | Path | None = None,
        intra_op_threads: int = 1,
        snap_words: bool = False,
    ) -> GidiPredictor:
        """Load ``bundle_dir``; ``model_path`` swaps the ONNX file (e.g. the FP32 export).

        ``snap_words`` extends the target span to whole-word boundaries (see ``__init__``).
        """
        root = Path(bundle_dir)
        config = load_config(root)
        tokenizer = BundleTokenizer(root / TOKENIZER_FILE, config.max_length)
        runner = OnnxRunner(
            model_path if model_path is not None else root / MODEL_FILE,
            config.input_names,
            config.output_names,
            intra_op_threads,
        )
        return cls(config, tokenizer, runner, snap_words=snap_words)

    def run(self, text: str) -> RawOutput:
        """Tokenize and run the model; see ``RawOutput`` for the fields."""
        normalized, tokens, type_logits, tag_logits, value_logits = self._infer(text)
        original_offsets = tuple(
            (0, 0) if special else normalized.span_to_original(s, e)
            for (s, e), special in zip(tokens.offsets, tokens.special_tokens_mask, strict=True)
        )
        value_tags = None
        if value_logits is not None:
            real = [not special for special in tokens.special_tokens_mask]
            value_tags = tuple(viterbi_masked(value_logits, real, self._crf, fill=TAG_O))
        return RawOutput(
            type_logits=type_logits,
            tag_logits=tag_logits,
            input_ids=tokens.ids,
            offsets=original_offsets,
            truncated=tokens.truncated,
            normalized_text=normalized.text,
            normalized_offsets=tokens.offsets,
            special_tokens_mask=tokens.special_tokens_mask,
            value_logits=value_logits,
            value_tags=value_tags,
        )

    def predict(self, text: str) -> Prediction:
        normalized, tokens, type_logits, tag_logits, value_logits = self._infer(text)

        type_index, type_conf = type_prediction(type_logits)
        real = [not special for special in tokens.special_tokens_mask]
        tag_ids = np.argmax(tag_logits, axis=-1)
        target_decoded = decode_first_span(tokens.offsets, tag_ids, normalized.text)
        if self._snap_words:
            target_decoded = snap_span_to_words(target_decoded, normalized.text)
        target_conf = target_confidence(tag_logits, tag_ids, target_decoded, real)
        target, target_span = self._slice_original(text, normalized, target_decoded)
        value = None
        if value_logits is not None:
            value_decoded, value_conf = decode_value_crf(
                value_logits, tokens.offsets, normalized.text, real, self._crf
            )
            value_text, value_span = self._slice_original(text, normalized, value_decoded)
            value = ValuePrediction(text=value_text, span=value_span, confidence=value_conf)
        return Prediction(
            type=self._config.types[type_index],
            type_confidence=type_conf,
            target=target,
            target_span=target_span,
            target_confidence=target_conf,
            truncated=tokens.truncated,
            model_version=self.model_version,
            value=value,
        )

    def _infer(
        self, text: str
    ) -> tuple[NormalizedText, TokenizedText, np.ndarray, np.ndarray, np.ndarray | None]:
        """One normalization, one tokenization, one ONNX run for ``text``."""
        normalized = self._normalize(text)
        tokens = self._tokenizer.encode(normalized.text)
        type_logits, tag_logits, *rest = self._runner(tokens.ids)
        return normalized, tokens, type_logits, tag_logits, rest[0] if rest else None

    @staticmethod
    def _slice_original(
        text: str, normalized: NormalizedText, span: DecodedSpan | None
    ) -> tuple[str | None, tuple[int, int] | None]:
        """``(text sliced from the caller's string, span in its code points)`` of a decoded span."""
        if span is None:
            return None, None
        original = normalized.span_to_original(span.start, span.end)
        return text[original[0] : original[1]], original

    @staticmethod
    def _normalize(text: str) -> NormalizedText:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, not {type(text).__name__}")
        if not text.strip():
            raise EmptyInputError("text is empty or whitespace-only")
        return normalize_nfc(text)
