"""Deployment runtime: bundle directory + onnxruntime + tokenizers + numpy, nothing else.

This package never imports torch, transformers, ``gidi.modeling`` or ``gidi.export``.
"""

from gidi.inference.bundle import BundleError, verify_bundle
from gidi.inference.predictor import GidiPredictor
from gidi.inference.schema import EmptyInputError, Prediction, RawOutput, ValuePrediction

__all__ = [
    "BundleError",
    "EmptyInputError",
    "GidiPredictor",
    "Prediction",
    "RawOutput",
    "ValuePrediction",
    "verify_bundle",
]
