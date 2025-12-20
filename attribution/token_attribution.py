"""Token-level attribution methods for VERDICT project.

This module provides methods for attributing model outputs to specific input tokens
or image patches, focusing on the language generation aspect.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TokenAttribution:
    """
    Compute token-level attributions for LVLMs.
    
    Analyses how much each input token (text) or image patch contributes
    to the generation of a specific output token.
    
    Supports multiple attribution methods:
        - Attention-based: Uses attention weights directly
        - Gradient-based: Computes gradients w.r.t. embeddings
        - Rollout: Aggregates attention across layers
    
    Attributes:
        model: The LVLM model.
    """
    
    def __init__(self, model: nn.Module) -> None:
        """
        Initialise TokenAttribution.
        
        Args:
            model: The vision-language model.
        """
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def compute_attention_attribution(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_token_idx: int = -1,
        layer_idx: int = -1,
        aggregate_heads: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribution based on attention weights.
        
        Args:
            input_ids: Input token IDs [B, L].
            pixel_values: Image tensor [B, C, H, W] (for VLMs).
            attention_mask: Attention mask [B, L].
            target_token_idx: Index of target token position.
            layer_idx: Layer index to extract attention from (-1 for last).
            aggregate_heads: Whether to average across attention heads.
        
        Returns:
            Dictionary containing:
                - 'attention': Attention weights for target token [B, L]
                - 'cross_attention': Cross-attention to image (if available)
        """
        with torch.no_grad():
            # Prepare inputs
            model_kwargs = {
                "input_ids": input_ids,
                "output_attentions": True,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            if pixel_values is not None:
                model_kwargs["pixel_values"] = pixel_values
            
            outputs = self.model(**model_kwargs)
        
        result = {}
        
        # Extract self-attention from the specified layer
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions[layer_idx]
            # Shape: [B, H, L, L]
            
            if aggregate_heads:
                attentions = attentions.mean(dim=1)  # [B, L, L]
            
            # Get attention TO the target token (what it attends to)
            target_attention = attentions[:, target_token_idx, :]  # [B, L]
            result["attention"] = target_attention
        
        # Extract cross-attention if available (for VLMs)
        if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions is not None:
            cross_attn = outputs.cross_attentions[layer_idx]
            if aggregate_heads:
                cross_attn = cross_attn.mean(dim=1)
            result["cross_attention"] = cross_attn[:, target_token_idx, :]
        
        return result
    
    def compute_gradient_attribution(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_token_idx: int = -1,
        target_vocab_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribution based on gradients w.r.t. input embeddings.
        
        Args:
            input_ids: Input token IDs [B, L].
            pixel_values: Image tensor [B, C, H, W].
            attention_mask: Attention mask [B, L].
            target_token_idx: Position in sequence for target token.
            target_vocab_idx: Vocabulary index of target token.
                If None, uses the predicted token.
        
        Returns:
            Dictionary containing:
                - 'text_attribution': Gradient norms for text tokens [B, L]
                - 'image_attribution': Gradient norms for image patches [B, N]
        """
        result = {}
        
        # Get text embeddings
        embed_layer = self._get_embedding_layer()
        if embed_layer is None:
            logger.warning("Could not find embedding layer")
            return result
        
        text_embeddings = embed_layer(input_ids)
        text_embeddings.requires_grad_(True)
        
        # Forward pass
        model_kwargs = {"inputs_embeds": text_embeddings}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        
        # Handle image inputs for VLMs
        if pixel_values is not None:
            pixel_values = pixel_values.clone().requires_grad_(True)
            model_kwargs["pixel_values"] = pixel_values
        
        outputs = self.model(**model_kwargs)
        logits = outputs.logits
        
        # Get target logit
        if target_vocab_idx is None:
            target_vocab_idx = logits[:, target_token_idx, :].argmax(dim=-1)
        
        # Select target logits
        batch_size = logits.shape[0]
        target_logits = logits[
            torch.arange(batch_size, device=logits.device),
            target_token_idx,
            target_vocab_idx if isinstance(target_vocab_idx, int) else target_vocab_idx
        ]
        
        # Backward pass
        target_logits.sum().backward()
        
        # Compute gradient norms
        if text_embeddings.grad is not None:
            text_attribution = text_embeddings.grad.norm(dim=-1)
            result["text_attribution"] = text_attribution.detach()
        
        if pixel_values is not None and pixel_values.grad is not None:
            # For image gradients, reshape to patch grid
            img_grad = pixel_values.grad
            # Sum over channels and compute per-patch norms
            result["image_attribution"] = img_grad.abs().mean(dim=1).detach()
        
        return result
    
    def compute_attention_rollout(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_token_idx: int = -1,
        start_layer: int = 0,
        discard_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute attention rollout across layers.
        
        Tracks information flow by multiplying attention matrices
        across layers, accounting for residual connections.
        
        Args:
            input_ids: Input token IDs [B, L].
            pixel_values: Image tensor [B, C, H, W].
            attention_mask: Attention mask [B, L].
            target_token_idx: Target token position.
            start_layer: Layer to start rollout from.
            discard_ratio: Ratio of lowest attention weights to discard.
        
        Returns:
            Rollout attention from target token to all inputs [B, L].
        """
        with torch.no_grad():
            model_kwargs = {
                "input_ids": input_ids,
                "output_attentions": True,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            if pixel_values is not None:
                model_kwargs["pixel_values"] = pixel_values
            
            outputs = self.model(**model_kwargs)
        
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise ValueError("Model does not output attention weights")
        
        attentions = outputs.attentions[start_layer:]
        
        # Aggregate attention heads (mean)
        attention_matrices = []
        for attn in attentions:
            # [B, H, L, L] -> [B, L, L]
            attn_avg = attn.mean(dim=1)
            
            # Optionally discard low attention weights
            if discard_ratio > 0:
                flat = attn_avg.view(attn_avg.size(0), -1)
                threshold = flat.quantile(discard_ratio, dim=-1, keepdim=True)
                threshold = threshold.view(attn_avg.size(0), 1, 1)
                attn_avg = torch.where(attn_avg < threshold, torch.zeros_like(attn_avg), attn_avg)
            
            # Re-normalise
            attn_avg = attn_avg / (attn_avg.sum(dim=-1, keepdim=True) + 1e-10)
            attention_matrices.append(attn_avg)
        
        # Compute rollout
        batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device
        
        # Start with identity (residual connection)
        rollout = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        for attn in attention_matrices:
            # Add residual connection
            attn_with_residual = 0.5 * attn + 0.5 * torch.eye(seq_len, device=device)
            rollout = torch.bmm(attn_with_residual, rollout)
        
        # Get attention from target token to all positions
        target_rollout = rollout[:, target_token_idx, :]
        
        return target_rollout
    
    def _get_embedding_layer(self) -> Optional[nn.Module]:
        """Find the token embedding layer in the model."""
        # Try common patterns
        patterns = [
            "get_input_embeddings",  # HuggingFace standard
            "embed_tokens",
            "word_embeddings",
            "wte",
        ]
        
        for pattern in patterns:
            if hasattr(self.model, pattern):
                attr = getattr(self.model, pattern)
                if callable(attr):
                    return attr()
                return attr
        
        # Search in submodules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and "word" in name.lower() or "token" in name.lower():
                return module
        
        return None
    
    def __del__(self):
        """Clean up hooks on deletion."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
