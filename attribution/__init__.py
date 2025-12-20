"""Attribution methods for VERDICT project.

This module provides attribution methods adapted for Large Vision-Language Models,
including the CoIBA (Comprehensive Information Bottleneck Attribution) adapter
for attributing generated text tokens to image regions.

Reference:
    Hong et al. "Comprehensive Information Bottleneck for Unveiling Universal 
    Attribution to Interpret Vision Transformers" CVPR 2025 (Highlight)
"""

from .coiba_adapter import (
    CoIBAForLVLM,
    AttributionOutput,
    WelfordEstimator,
    overlay_attribution_on_image,
    save_attribution_visualisation,
    get_vit_layer_names,
    postprocess_heatmap,
)
from .layer_attribution import LayerAttribution
from .token_attribution import TokenAttribution
from .visualisation import visualise_attribution, plot_attribution_grid

__all__ = [
    # CoIBA Adapter (main class)
    "CoIBAForLVLM",
    "AttributionOutput",
    "WelfordEstimator",
    # CoIBA Utilities
    "get_vit_layer_names",
    "postprocess_heatmap",
    # CoIBA Visualisation
    "overlay_attribution_on_image",
    "save_attribution_visualisation",
    # Layer Attribution
    "LayerAttribution",
    # Token Attribution
    "TokenAttribution",
    # General Visualisation
    "visualise_attribution",
    "plot_attribution_grid",
]
