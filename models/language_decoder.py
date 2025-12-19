"""Language decoder utilities for VERDICT project.

This module provides utilities for working with language model decoders
in Large Vision-Language Models, particularly for feature extraction,
attention analysis, and token-level attribution.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LanguageDecoderWrapper:
    """
    Wrapper for language model decoder components of LVLMs.
    
    Provides utilities for feature extraction, attention analysis,
    and token-level analysis for autoregressive language models.
    
    Attributes:
        decoder: The underlying language model module.
        layer_names: Names of decoder layers.
    
    Example:
        >>> wrapper = LanguageDecoderWrapper(model.get_language_model())
        >>> features = wrapper.extract_features(input_ids)
        >>> print(features["layer_16"].shape)
    """
    
    def __init__(
        self,
        decoder: nn.Module,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """
        Initialise language decoder wrapper.
        
        Args:
            decoder: Language model decoder module.
            tokenizer: Optional tokenizer for text processing.
        """
        self.decoder = decoder
        self.tokenizer = tokenizer
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        
        # Identify layer structure
        self.layer_names = self._identify_layers()
    
    def _identify_layers(self) -> List[str]:
        """Identify transformer layers in the decoder."""
        layer_names = []
        
        for name, module in self.decoder.named_modules():
            # Common patterns for transformer decoder layers
            if any(pattern in name.lower() for pattern in ["layer", "block", "decoder"]):
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
            layers: Layer names or indices to hook.
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
            module = dict(self.decoder.named_modules())[layer_name]
            
            # Forward hook
            def forward_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    self._activations[name] = output.detach()
                return hook
            
            handle = module.register_forward_hook(forward_hook(layer_name))
            self._hooks.append(handle)
            
            # Backward hook
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden state features from specified layers.
        
        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            layers: Layer indices to extract from.
        
        Returns:
            Dictionary mapping layer names to feature tensors.
        """
        self.register_hooks(layers)
        
        with torch.no_grad():
            _ = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        features = self._activations.copy()
        self.remove_hooks()
        
        return features
    
    def extract_attention_patterns(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aggregate_heads: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract self-attention patterns from decoder layers.
        
        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            aggregate_heads: Whether to average across attention heads.
        
        Returns:
            Dictionary mapping layer names to attention tensors.
        """
        attention_patterns = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]
                    if aggregate_heads and attn.dim() == 4:
                        attn = attn.mean(dim=1)
                    attention_patterns[name] = attn.detach()
            return hook
        
        handles = []
        for name, module in self.decoder.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(attention_hook(name))
                handles.append(handle)
        
        with torch.no_grad():
            _ = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        
        for handle in handles:
            handle.remove()
        
        return attention_patterns
    
    def compute_token_embeddings(
        self,
        input_ids: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Get token embeddings from a specific layer.
        
        Args:
            input_ids: Token IDs of shape (B, L).
            layer: Layer index to extract embeddings from.
        
        Returns:
            Token embeddings of shape (B, L, D).
        """
        with torch.no_grad():
            outputs = self.decoder(
                input_ids=input_ids,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states
        return hidden_states[layer]
    
    def compute_attention_rollout(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_layer: int = 0,
        head_fusion: str = "mean",
    ) -> torch.Tensor:
        """
        Compute attention rollout across layers.
        
        Attention rollout tracks information flow through the
        transformer by multiplying attention matrices.
        
        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            start_layer: Layer to start rollout from.
            head_fusion: How to combine heads - 'mean', 'max', or 'min'.
        
        Returns:
            Rollout attention of shape (B, L, L).
        """
        with torch.no_grad():
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        
        attentions = outputs.attentions[start_layer:]
        
        # Fuse attention heads
        if head_fusion == "mean":
            fused_attentions = [attn.mean(dim=1) for attn in attentions]
        elif head_fusion == "max":
            fused_attentions = [attn.max(dim=1)[0] for attn in attentions]
        elif head_fusion == "min":
            fused_attentions = [attn.min(dim=1)[0] for attn in attentions]
        else:
            raise ValueError(f"Unknown head fusion method: {head_fusion}")
        
        # Compute rollout
        batch_size, seq_len = input_ids.shape
        rollout = torch.eye(seq_len, device=input_ids.device).unsqueeze(0)
        rollout = rollout.expand(batch_size, -1, -1)
        
        for attention in fused_attentions:
            # Add residual connection
            attention = attention + torch.eye(seq_len, device=attention.device)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            rollout = torch.bmm(attention, rollout)
        
        return rollout
    
    def get_cross_attention_to_image(
        self,
        input_ids: torch.Tensor,
        image_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract cross-attention weights from language to image.
        
        Args:
            input_ids: Token IDs of shape (B, L).
            image_embeddings: Image embeddings of shape (B, N, D).
            attention_mask: Attention mask.
        
        Returns:
            Dictionary mapping layer names to cross-attention weights.
        """
        cross_attentions = {}
        
        def cross_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    cross_attentions[name] = output[1].detach()
            return hook
        
        handles = []
        for name, module in self.decoder.named_modules():
            if "cross" in name.lower() and "attention" in name.lower():
                handle = module.register_forward_hook(cross_attention_hook(name))
                handles.append(handle)
        
        with torch.no_grad():
            # Note: This requires the model to accept cross-attention inputs
            # The exact API depends on the model architecture
            _ = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeddings,
                output_attentions=True,
            )
        
        for handle in handles:
            handle.remove()
        
        return cross_attentions
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


def extract_language_features(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    return_attention: bool = False,
) -> Dict[str, Any]:
    """
    Extract language features from a model.
    
    Convenience function for one-off feature extraction.
    
    Args:
        model: Model containing language decoder.
        input_ids: Token IDs of shape (B, L).
        attention_mask: Attention mask.
        layers: Layer indices to extract from.
        return_attention: Whether to return attention patterns.
    
    Returns:
        Dictionary containing features and optionally attention.
    """
    # Find language decoder
    language_model = None
    
    for name in ["language_model", "model", "decoder"]:
        if hasattr(model, name):
            language_model = getattr(model, name)
            break
    
    if language_model is None:
        raise ValueError("Could not find language model in model")
    
    wrapper = LanguageDecoderWrapper(language_model)
    
    result = {
        "features": wrapper.extract_features(input_ids, attention_mask, layers)
    }
    
    if return_attention:
        result["attention"] = wrapper.extract_attention_patterns(
            input_ids, attention_mask
        )
    
    return result


def compute_token_importance(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_positions: List[int],
    method: str = "attention",
) -> torch.Tensor:
    """
    Compute importance scores for each input token.
    
    Args:
        model: Language model.
        input_ids: Token IDs of shape (B, L).
        target_positions: Positions of target tokens to analyse.
        method: Importance method - 'attention', 'gradient', or 'erasure'.
    
    Returns:
        Importance scores of shape (B, L).
    """
    if method == "attention":
        # Use attention-based importance
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
        
        # Aggregate attention to target positions
        attentions = torch.stack(outputs.attentions)
        attentions = attentions.mean(dim=0).mean(dim=1)  # Average layers and heads
        
        importance = torch.zeros(input_ids.shape, device=input_ids.device)
        for pos in target_positions:
            importance += attentions[:, pos, :]
        importance = importance / len(target_positions)
        
    elif method == "gradient":
        # Use gradient-based importance
        input_ids.requires_grad_(False)
        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits
        
        # Sum logits at target positions
        target_logits = sum(logits[:, pos, :].sum() for pos in target_positions)
        target_logits.backward()
        
        importance = embeddings.grad.norm(dim=-1)
        
    elif method == "erasure":
        # Use leave-one-out erasure
        with torch.no_grad():
            baseline_outputs = model(input_ids=input_ids)
            baseline_logits = baseline_outputs.logits
            baseline_probs = F.softmax(baseline_logits, dim=-1)
            
            importance = torch.zeros(input_ids.shape, device=input_ids.device)
            
            for i in range(input_ids.shape[1]):
                # Create masked input
                masked_ids = input_ids.clone()
                masked_ids[:, i] = model.config.pad_token_id
                
                outputs = model(input_ids=masked_ids)
                masked_probs = F.softmax(outputs.logits, dim=-1)
                
                # Measure change at target positions
                for pos in target_positions:
                    diff = (baseline_probs[:, pos] - masked_probs[:, pos]).abs().sum(dim=-1)
                    importance[:, i] += diff
            
            importance = importance / len(target_positions)
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    return importance


def identify_hallucination_tokens(
    generated_ids: torch.Tensor,
    vision_attention: torch.Tensor,
    threshold: float = 0.1,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Identify potentially hallucinated tokens based on vision attention.
    
    Tokens with low attention to image features may indicate
    hallucinations driven by language priors rather than visual evidence.
    
    Args:
        generated_ids: Generated token IDs of shape (B, L).
        vision_attention: Cross-attention to image of shape (B, L, N).
        threshold: Minimum vision attention for grounded tokens.
        tokenizer: Optional tokenizer for decoding tokens.
    
    Returns:
        Dictionary containing:
            - hallucination_mask: Boolean mask of potential hallucinations
            - attention_scores: Per-token vision attention scores
            - hallucination_tokens: List of potentially hallucinated tokens
    """
    # Sum attention across image patches
    attention_scores = vision_attention.sum(dim=-1)
    
    # Normalise attention scores
    attention_scores = attention_scores / (attention_scores.max(dim=-1, keepdim=True)[0] + 1e-8)
    
    # Identify low-attention tokens
    hallucination_mask = attention_scores < threshold
    
    result = {
        "hallucination_mask": hallucination_mask,
        "attention_scores": attention_scores,
    }
    
    if tokenizer is not None:
        hallucination_tokens = []
        for batch_idx in range(generated_ids.shape[0]):
            batch_hallucinations = []
            for token_idx in range(generated_ids.shape[1]):
                if hallucination_mask[batch_idx, token_idx]:
                    token = tokenizer.decode([generated_ids[batch_idx, token_idx]])
                    batch_hallucinations.append({
                        "position": token_idx,
                        "token": token,
                        "attention": attention_scores[batch_idx, token_idx].item(),
                    })
            hallucination_tokens.append(batch_hallucinations)
        
        result["hallucination_tokens"] = hallucination_tokens
    
    return result
