"""CPU ONNX Runtime session for one note at a time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxRunner:
    def __init__(
        self,
        model_path: str | Path,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        intra_op_threads: int = 1,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        session_inputs = tuple(i.name for i in self._session.get_inputs())
        session_outputs = tuple(o.name for o in self._session.get_outputs())
        if session_inputs != input_names or session_outputs != output_names:
            raise ValueError(
                f"{model_path} exposes inputs {session_inputs} / outputs {session_outputs}, "
                f"bundle config expects {input_names} / {output_names}"
            )
        self._input_names = input_names
        self._output_names = list(output_names)

    def __call__(self, input_ids: tuple[int, ...]) -> tuple[np.ndarray, ...]:
        """Logits for one sequence, in ``output_names`` order.

        ``(type_logits [num_types], tag_logits [tokens, num_tags])``, plus
        ``value_logits [tokens, 3]`` for a bundle with a value head.
        """
        ids = np.asarray([input_ids], dtype=np.int64)
        mask = np.ones_like(ids)
        outputs = self._session.run(
            self._output_names, dict(zip(self._input_names, (ids, mask), strict=True))
        )
        return tuple(output[0] for output in outputs)
