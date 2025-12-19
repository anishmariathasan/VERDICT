"""Baseline hallucination mitigation methods for VERDICT project."""

from .attention_calibration import UnifiedAttentionCalibration, DenseAttentionCalibration
from .visual_evidence import VisualEvidencePrompter
from .utils import compute_hallucination_rate

__all__ = [
    "UnifiedAttentionCalibration",
    "DenseAttentionCalibration",
    "VisualEvidencePrompter",
    "compute_hallucination_rate",
]
