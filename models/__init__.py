"""Model implementations for VERDICT project."""

from .maira2_wrapper import (
    MAIRA2Model,
    MAIRA2Config,
    MAIRA2Output,
    MAIRA2_IMAGE_SIZE,
    MAIRA2_PATCH_SIZE,
    MAIRA2_NUM_PATCHES,
    MAIRA2_NUM_VISION_TOKENS,
    MAIRA2_NUM_VISION_LAYERS,
    MAIRA2_VISION_HIDDEN_DIM,
    MAIRA2_LLM_HIDDEN_DIM,
    MAIRA2_NUM_LLM_LAYERS,
)
from .vision_encoder import VisionEncoderWrapper, extract_vision_features
from .language_decoder import LanguageDecoderWrapper, extract_language_features

__all__ = [
    # MAIRA-2 Model
    "MAIRA2Model",
    "MAIRA2Config",
    "MAIRA2Output",
    # MAIRA-2 Architecture Constants
    "MAIRA2_IMAGE_SIZE",
    "MAIRA2_PATCH_SIZE",
    "MAIRA2_NUM_PATCHES",
    "MAIRA2_NUM_VISION_TOKENS",
    "MAIRA2_NUM_VISION_LAYERS",
    "MAIRA2_VISION_HIDDEN_DIM",
    "MAIRA2_LLM_HIDDEN_DIM",
    "MAIRA2_NUM_LLM_LAYERS",
    # Vision Encoder
    "VisionEncoderWrapper",
    "extract_vision_features",
    # Language Decoder
    "LanguageDecoderWrapper",
    "extract_language_features",
]
