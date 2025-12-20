"""Layer-wise attribution methods for VERDICT project.

This module provides methods for computing attribution at intermediate layers
of the model, useful for understanding feature importance at different levels of abstraction.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients, LayerGradCam

logger = logging.getLogger(__name__)


class LayerAttribution:
    """
    Compute layer-wise attributions for LVLMs.
    
    Supports Layer Integrated Gradients and Layer GradCAM to analyse
    feature importance within specific layers of the vision encoder or language decoder.
    
    Attributes:
        model: The model to analyse.
        layer: The specific layer module to attribute to.
    """
    
    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        """
        Initialise LayerAttribution.
        
        Args:
            model: The full model.
            layer: The specific layer module to analyse.
        """
        self.model = model
        self.layer = layer
        self.lig = LayerIntegratedGradients(self._model_forward_wrapper, layer)
        self.grad_cam = LayerGradCam(self._model_forward_wrapper, layer)
    
    def _model_forward_wrapper(
        self,
        inputs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
    ) -> torch.Tensor:
        """
        Wrapper for model forward pass.
        
        Args:
            inputs: Image tensor (pixel_values).
            input_ids: Token IDs.
            attention_mask: Attention mask.
            target_token_idx: Index of token to compute attribution for.
        
        Returns:
            Logits for the target token.
        """
        # MAIRA-2 uses pixel_values, not images
        outputs = self.model(
            pixel_values=inputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        # Get logits for the last generated position
        last_token_logits = logits[:, -1, :]
        return last_token_logits[:, target_token_idx]

    def compute_layer_integrated_gradients(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        target_token_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        n_steps: int = 50,
    ) -> torch.Tensor:
        """
        Compute Layer Integrated Gradients.
        
        Args:
            image: Input image.
            input_ids: Input token IDs.
            target_token_idx: Target token index.
            attention_mask: Attention mask.
            n_steps: Number of steps.
        
        Returns:
            Attribution tensor for the layer.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        additional_args = (input_ids, attention_mask, target_token_idx)
        
        attributions = self.lig.attribute(
            inputs=image,
            additional_forward_args=additional_args,
            n_steps=n_steps,
        )
        return attributions

    def compute_layer_grad_cam(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        target_token_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Layer GradCAM.
        
        Args:
            image: Input image.
            input_ids: Input token IDs.
            target_token_idx: Target token index.
            attention_mask: Attention mask.
        
        Returns:
            GradCAM attribution map.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        additional_args = (input_ids, attention_mask, target_token_idx)
        
        attributions = self.grad_cam.attribute(
            inputs=image,
            additional_forward_args=additional_args,
        )
        return attributions
