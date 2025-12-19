"""Vision encoder utilities for VERDICT project.

This module provides utilities for working with vision encoders in
Large Vision-Language Models, particularly for feature extraction
and analysis in the context of attribution methods.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class VisionEncoderWrapper:
    """
    Wrapper for vision encoder components of LVLMs.
    
    Provides utilities for feature extraction, layer access,
    and attention analysis for vision encoders.
    
    Attributes:
        encoder: The underlying vision encoder module.
        layer_names: Names of encoder layers.
    
    Example:
        >>> wrapper = VisionEncoderWrapper(model.get_vision_encoder())
        >>> features = wrapper.extract_features(image)
        >>> print(features["layer_12"].shape)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        layer_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Initialise vision encoder wrapper.
        
        Args:
            encoder: Vision encoder module.
            layer_indices: Specific layer indices to track.
                If None, extracts from all layers.
        """
        self.encoder = encoder
        self.layer_indices = layer_indices
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        
        # Identify layer structure
        self.layer_names = self._identify_layers()
    
    def _identify_layers(self) -> List[str]:
        """Identify transformer layers in the encoder."""
        layer_names = []
        
        for name, module in self.encoder.named_modules():
            # Common patterns for transformer layers
            if any(pattern in name.lower() for pattern in ["block", "layer", "encoder"]):
                if isinstance(module, nn.Module) and list(module.children()):
                    layer_names.append(name)
        
        return layer_names
    
    def register_hooks(
        self,
        layers: Optional[List[Union[str, int]]] = None,
        capture_gradients: bool = False,
    ) -> None:
        """
        Register forward/backward hooks to capture activations.
        
        Args:
            layers: Layer names or indices to hook. If None, hooks all.
            capture_gradients: Whether to also capture gradients.
        """
        self.remove_hooks()
        self._activations.clear()
        self._gradients.clear()
        
        # Determine which layers to hook
        if layers is None:
            target_layers = self.layer_names
        elif all(isinstance(l, int) for l in layers):
            target_layers = [self.layer_names[i] for i in layers if i < len(self.layer_names)]
        else:
            target_layers = [l for l in layers if l in self.layer_names]
        
        for layer_name in target_layers:
            module = dict(self.encoder.named_modules())[layer_name]
            
            # Forward hook
            def forward_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    self._activations[name] = output.detach()
                return hook
            
            handle = module.register_forward_hook(forward_hook(layer_name))
            self._hooks.append(handle)
            
            # Backward hook for gradients
            if capture_gradients:
                def backward_hook(name):
                    def hook(module, grad_input, grad_output):
                        if isinstance(grad_output, tuple):
                            grad_output = grad_output[0]
                        self._gradients[name] = grad_output.detach()
                    return hook
                
                handle = module.register_backward_hook(backward_hook(layer_name))
                self._hooks.append(handle)
        
        logger.debug(f"Registered hooks on {len(target_layers)} layers")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
    
    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            images: Image tensor of shape (B, C, H, W).
            layers: Layer indices to extract from.
        
        Returns:
            Dictionary mapping layer names to feature tensors.
        
        Example:
            >>> features = wrapper.extract_features(images, layers=[-4, -2, -1])
            >>> for name, feat in features.items():
            ...     print(f"{name}: {feat.shape}")
        """
        self.register_hooks(layers)
        
        with torch.no_grad():
            _ = self.encoder(images)
        
        features = self._activations.copy()
        self.remove_hooks()
        
        return features
    
    def extract_attention_maps(
        self,
        images: torch.Tensor,
        aggregate_heads: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps from self-attention layers.
        
        Args:
            images: Image tensor of shape (B, C, H, W).
            aggregate_heads: Whether to average across attention heads.
        
        Returns:
            Dictionary mapping layer names to attention tensors.
            Shape: (B, H, N, N) or (B, N, N) if aggregated.
        """
        attention_maps = {}
        
        # Register hooks specifically for attention modules
        def attention_hook(name):
            def hook(module, input, output):
                # Most attention modules return (output, attention_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]
                    if aggregate_heads and attn.dim() == 4:
                        attn = attn.mean(dim=1)
                    attention_maps[name] = attn.detach()
            return hook
        
        handles = []
        for name, module in self.encoder.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(attention_hook(name))
                handles.append(handle)
        
        with torch.no_grad():
            _ = self.encoder(images)
        
        for handle in handles:
            handle.remove()
        
        return attention_maps
    
    def get_patch_embeddings(
        self,
        images: torch.Tensor,
        include_cls: bool = False,
    ) -> torch.Tensor:
        """
        Get patch embeddings from the vision encoder.
        
        Args:
            images: Image tensor of shape (B, C, H, W).
            include_cls: Whether to include the CLS token.
        
        Returns:
            Patch embeddings of shape (B, N, D) where N is number of patches.
        """
        with torch.no_grad():
            outputs = self.encoder(images, output_hidden_states=True)
        
        # Get the last hidden state
        if hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if not include_cls and embeddings.shape[1] > 0:
            # Assume first token is CLS
            embeddings = embeddings[:, 1:, :]
        
        return embeddings
    
    def compute_patch_similarity(
        self,
        images: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between patches and a query embedding.
        
        Args:
            images: Image tensor of shape (B, C, H, W).
            query: Query embedding of shape (B, D) or (D,).
        
        Returns:
            Similarity scores of shape (B, N).
        """
        patch_embeddings = self.get_patch_embeddings(images)
        
        # Normalise embeddings
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = F.normalize(query, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.einsum("bnd,bd->bn", patch_embeddings, query)
        
        return similarity
    
    def reshape_to_spatial(
        self,
        features: torch.Tensor,
        patch_size: int = 14,
        image_size: int = 518,
    ) -> torch.Tensor:
        """
        Reshape flat patch features to spatial grid.
        
        Args:
            features: Features of shape (B, N, D) or (B, N).
            patch_size: Size of each patch in pixels.
            image_size: Original image size.
        
        Returns:
            Spatial features of shape (B, H, W, D) or (B, H, W).
        """
        num_patches = image_size // patch_size
        B = features.shape[0]
        
        if features.dim() == 2:
            # Shape (B, N) -> (B, H, W)
            spatial = features.view(B, num_patches, num_patches)
        else:
            # Shape (B, N, D) -> (B, H, W, D)
            D = features.shape[-1]
            spatial = features.view(B, num_patches, num_patches, D)
        
        return spatial
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


def extract_vision_features(
    model: nn.Module,
    images: torch.Tensor,
    layers: Optional[List[int]] = None,
    return_attention: bool = False,
) -> Dict[str, Any]:
    """
    Extract vision features from a model.
    
    Convenience function for one-off feature extraction without
    creating a wrapper instance.
    
    Args:
        model: Model containing vision encoder.
        images: Image tensor of shape (B, C, H, W).
        layers: Layer indices to extract from.
        return_attention: Whether to also return attention maps.
    
    Returns:
        Dictionary containing:
            - features: Dictionary of layer features
            - attention: Attention maps (if requested)
    
    Example:
        >>> result = extract_vision_features(model, images, layers=[-1])
        >>> final_features = result["features"]["layer_23"]
    """
    # Find vision encoder in model
    vision_encoder = None
    
    for name in ["vision_tower", "vision_encoder", "visual"]:
        if hasattr(model, name):
            vision_encoder = getattr(model, name)
            break
    
    if vision_encoder is None:
        raise ValueError("Could not find vision encoder in model")
    
    wrapper = VisionEncoderWrapper(vision_encoder, layers)
    
    result = {"features": wrapper.extract_features(images, layers)}
    
    if return_attention:
        result["attention"] = wrapper.extract_attention_maps(images)
    
    return result


def compute_grad_cam(
    model: nn.Module,
    images: torch.Tensor,
    target_layer: str,
    target_class: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Grad-CAM attribution for vision encoder.
    
    Args:
        model: Model containing vision encoder.
        images: Image tensor of shape (B, C, H, W).
        target_layer: Name of layer to compute Grad-CAM for.
        target_class: Target class index. If None, uses predicted class.
    
    Returns:
        Grad-CAM attribution map of shape (B, H, W).
    """
    # Find vision encoder
    vision_encoder = None
    for name in ["vision_tower", "vision_encoder", "visual"]:
        if hasattr(model, name):
            vision_encoder = getattr(model, name)
            break
    
    if vision_encoder is None:
        raise ValueError("Could not find vision encoder in model")
    
    # Storage for activations and gradients
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations["value"] = output
    
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]
        gradients["value"] = grad_output
    
    # Register hooks
    target_module = dict(vision_encoder.named_modules())[target_layer]
    fwd_handle = target_module.register_forward_hook(forward_hook)
    bwd_handle = target_module.register_full_backward_hook(backward_hook)
    
    # Forward pass
    images.requires_grad_(True)
    outputs = model(images)
    
    # Get target for backward
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs
    
    if target_class is None:
        target_class = logits.argmax(dim=-1)
    
    # Backward pass
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, target_class.unsqueeze(1), 1)
    logits.backward(gradient=one_hot, retain_graph=True)
    
    # Compute Grad-CAM
    grads = gradients["value"]
    acts = activations["value"]
    
    # Global average pooling of gradients
    weights = grads.mean(dim=(2, 3) if grads.dim() == 4 else 1, keepdim=True)
    
    # Weighted combination
    cam = (weights * acts).sum(dim=-1 if acts.dim() == 3 else 1)
    cam = F.relu(cam)
    
    # Normalise
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Clean up
    fwd_handle.remove()
    bwd_handle.remove()
    
    return cam
