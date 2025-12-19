"""Attribution methods for VERDICT project.

This module provides attribution methods adapted for Large Vision-Language Models,
including the CoIBA (Contextual Interpretability for Biological Applications)
adapter for attributing generated text tokens to image regions.
"""

from .coiba_adapter import (
    CoIBAForLVLM,
    AttributionOutput,
    TokenAttributionResult,
    overlay_attribution_on_image,
    save_attribution_visualisation,
    create_token_attribution_grid,
)
from .layer_attribution import LayerAttribution
from .token_attribution import TokenAttribution
from .visualisation import visualise_attribution, plot_attribution_grid

__all__ = [
    # CoIBA Adapter
    "CoIBAForLVLM",
    "AttributionOutput",
    "TokenAttributionResult",
    # CoIBA Visualisation
    "overlay_attribution_on_image",
    "save_attribution_visualisation",
    "create_token_attribution_grid",
    # Layer Attribution
    "LayerAttribution",
    # Token Attribution
    "TokenAttribution",
    # General Visualisation
    "visualise_attribution",
    "plot_attribution_grid",
]
