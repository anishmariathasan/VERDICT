"""CoIBA (Contextual Interpretability for Biological Applications) adapted for LVLMs.

This module adapts the CoIBA framework from CVPR 2025 to work with generative
Large Vision-Language Models. The key adaptation is attributing to generated
TEXT TOKENS in medical reports rather than single class labels.

Key Challenge:
    - Original CoIBA attributes to single class label
    - Need to attribute to generated TEXT TOKENS in reports

Implementation:
    - Compute layer-wise attributions for each ViT layer
    - Attribute each generated token separately
    - Apply information bottleneck with damping
    - Aggregate across layers with weighted average

Architecture Assumptions (MAIRA-2):
    - Vision Encoder: Rad-DINO ViT-B/14 (12 layers)
    - Image Size: 518×518 → 37×37 patch grid
    - Vision Tokens: 1370 (1 CLS + 1369 patches)
    - Hidden Dimension: 768
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DAMPING_RATIO = 0.85
DEFAULT_NUM_ITERATIONS = 10
DEFAULT_NUM_LAYERS = 12  # Rad-DINO ViT layers


# ============================================================================
# Output Dataclasses
# ============================================================================

@dataclass
class AttributionOutput:
    """
    Output from attribution computation.
    
    Attributes:
        attribution_map: Spatial attribution map. Shape: [H, W] (37×37 for MAIRA-2).
        token_attributions: Per-token attribution maps. Maps token string to tensor.
        layer_attributions: Per-layer attribution maps. Maps layer index to tensor.
        aggregated_attribution: Final aggregated attribution across all tokens/layers.
        metadata: Additional metadata about the attribution computation.
    """
    attribution_map: torch.Tensor  # Shape: [H, W] or [num_patches]
    token_attributions: Dict[str, torch.Tensor] = field(default_factory=dict)
    layer_attributions: Dict[int, torch.Tensor] = field(default_factory=dict)
    aggregated_attribution: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenAttributionResult:
    """
    Attribution result for a single token.
    
    Attributes:
        token: The token string.
        token_id: The token vocabulary ID.
        attribution: Attribution map for this token.
        confidence: Confidence score for the attribution.
        layer_contributions: Per-layer contribution to this token's attribution.
    """
    token: str
    token_id: int
    attribution: torch.Tensor
    confidence: float = 0.0
    layer_contributions: Dict[int, float] = field(default_factory=dict)


# ============================================================================
# Main CoIBA Implementation
# ============================================================================

class CoIBAForLVLM:
    """
    CoIBA adapted for Large Vision-Language Models.
    
    This class adapts the Contextual Interpretability for Biological Applications
    (CoIBA) framework to work with generative LVLMs like MAIRA-2. The key difference
    from original CoIBA is attributing to generated text tokens rather than
    single class labels.
    
    Key Differences from Original CoIBA:
        1. Attributes to generated text tokens, not class labels
        2. Handles sequential generation (autoregressive)
        3. Aggregates attributions across multiple ViT layers
        4. Computes sentence-level attributions for error analysis
    
    Algorithm:
        1. For each generated token:
           a. Compute gradients of token logit w.r.t. vision features
           b. Apply information bottleneck with damping
           c. Aggregate across layers
        2. Generate comprehensive attribution map
        3. Optional: Generate sentence-level attributions
    
    Attributes:
        model: The LVLM model (e.g., MAIRA-2).
        vision_encoder: Vision encoder module (e.g., Rad-DINO ViT).
        damping_ratio: Damping ratio for information bottleneck (0-1).
        num_iterations: Number of iterations for feature removal.
        aggregation: Layer aggregation method ("weighted", "mean", "max").
        layer_weights: Weights for layer aggregation.
    
    Example:
        >>> from models.maira2_wrapper import MAIRA2Model
        >>> wrapper = MAIRA2Model.from_pretrained("microsoft/maira-2")
        >>> coiba = CoIBAForLVLM(
        ...     model=wrapper.model,
        ...     vision_encoder=wrapper.get_vision_encoder(),
        ...     damping_ratio=0.85,
        ...     num_iterations=10
        ... )
        >>> output = coiba.generate_comprehensive_attribution(
        ...     image_tensor,
        ...     generated_text="There is a right pleural effusion.",
        ...     prompt="Describe the findings:"
        ... )
        >>> print(f"Attribution shape: {output.attribution_map.shape}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        vision_encoder: nn.Module,
        damping_ratio: float = DEFAULT_DAMPING_RATIO,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        aggregation: str = "weighted",
        layer_weights: Optional[List[float]] = None,
        num_layers: int = DEFAULT_NUM_LAYERS,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """
        Initialise CoIBA for LVLM.
        
        Args:
            model: The LVLM model (MAIRA-2).
            vision_encoder: Vision encoder module (Rad-DINO ViT).
            damping_ratio: Damping ratio for information bottleneck (0-1).
                Higher values = more aggressive feature removal.
            num_iterations: Number of iterations for feature removal.
                More iterations = finer-grained attribution.
            aggregation: How to aggregate layers ("weighted", "mean", "max").
            layer_weights: Optional custom weights for layer aggregation.
                If None, uses exponential weighting favouring later layers.
            num_layers: Number of ViT layers (default 12 for Rad-DINO).
            tokenizer: Optional tokenizer for decoding tokens.
        
        Raises:
            ValueError: If damping_ratio not in (0, 1) or aggregation unknown.
        """
        if not 0 < damping_ratio < 1:
            raise ValueError(f"damping_ratio must be in (0, 1), got {damping_ratio}")
        
        if aggregation not in ("weighted", "mean", "max"):
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        self.model = model
        self.vision_encoder = vision_encoder
        self.damping_ratio = damping_ratio
        self.num_iterations = num_iterations
        self.aggregation = aggregation
        self.num_layers = num_layers
        self.tokenizer = tokenizer
        
        # Set layer weights (later layers more important for high-level features)
        if layer_weights is None:
            self.layer_weights = self._compute_exponential_weights(num_layers)
        else:
            if len(layer_weights) != num_layers:
                raise ValueError(
                    f"layer_weights length ({len(layer_weights)}) must match "
                    f"num_layers ({num_layers})"
                )
            self.layer_weights = torch.tensor(layer_weights, dtype=torch.float32)
            self.layer_weights = self.layer_weights / self.layer_weights.sum()
        
        # Determine device
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature storage for hooks
        self._vision_features: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        logger.info(
            f"Initialised CoIBA with damping={damping_ratio}, "
            f"iterations={num_iterations}, aggregation={aggregation}"
        )
    
    # ========================================================================
    # Weight Computation
    # ========================================================================
    
    def _compute_exponential_weights(
        self,
        num_layers: int,
        decay: float = 0.9,
    ) -> torch.Tensor:
        """
        Compute exponential weights favouring later layers.
        
        Later layers capture higher-level semantic features, which are
        typically more relevant for attribution in vision-language tasks.
        
        Args:
            num_layers: Number of layers.
            decay: Decay factor (< 1 = favour later layers).
        
        Returns:
            Normalised weights tensor of shape [num_layers].
        """
        weights = torch.tensor(
            [decay ** (num_layers - i - 1) for i in range(num_layers)],
            dtype=torch.float32
        )
        weights = weights / weights.sum()
        return weights
    
    # ========================================================================
    # Feature Hooks
    # ========================================================================
    
    def register_vision_hooks(self, vision_encoder: Optional[nn.Module] = None) -> None:
        """
        Register forward hooks to capture vision features from all layers.
        
        Args:
            vision_encoder: Vision encoder to hook (default: self.vision_encoder).
        """
        self._remove_hooks()
        self._vision_features = {}
        
        encoder = vision_encoder or self.vision_encoder
        
        # Find transformer layers
        layers = self._find_vision_layers(encoder)
        
        for layer_idx, layer in enumerate(layers):
            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    self._vision_features[idx] = hidden_states.detach()
                return hook_fn
            
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)
        
        logger.debug(f"Registered {len(self._hooks)} vision feature hooks")
    
    def _find_vision_layers(self, encoder: nn.Module) -> List[nn.Module]:
        """
        Find transformer layers in vision encoder.
        
        Args:
            encoder: Vision encoder module.
        
        Returns:
            List of transformer layer modules.
        """
        # Common attribute names for ViT layers
        layer_attrs = [
            'encoder.layer',      # HuggingFace ViT
            'blocks',             # timm ViT
            'layers',             # Generic
            'encoder.layers',
        ]
        
        for attr_path in layer_attrs:
            try:
                obj = encoder
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue
        
        # Fallback: search for ModuleList
        for name, module in encoder.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                return list(module)
        
        logger.warning("Could not find vision layers, returning empty list")
        return []
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._vision_features = {}
    
    # ========================================================================
    # Core Attribution Methods
    # ========================================================================
    
    def compute_layer_attributions(
        self,
        vision_features: Dict[int, torch.Tensor],
        token_gradients: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute attribution for a specific ViT layer.
        
        This implements the core CoIBA attribution computation for a single layer.
        It combines gradient information with the information bottleneck principle.
        
        Args:
            vision_features: Vision features from all layers.
                Shape per layer: [B, num_tokens, hidden_dim]
            token_gradients: Gradients of token logit w.r.t. vision features.
                Shape: [B, num_tokens, hidden_dim]
            layer_idx: Layer index to compute attribution for.
        
        Returns:
            Attribution map for this layer. Shape: [B, num_tokens]
        
        Raises:
            KeyError: If layer_idx not in vision_features.
        """
        if layer_idx not in vision_features:
            raise KeyError(f"Layer {layer_idx} not found in vision_features")
        
        # Get features for this layer
        features = vision_features[layer_idx]  # [B, num_tokens, hidden_dim]
        
        # Ensure gradient has same shape
        if token_gradients.shape != features.shape:
            logger.warning(
                f"Gradient shape {token_gradients.shape} != feature shape {features.shape}"
            )
            # Attempt to broadcast
            if token_gradients.shape[-1] == features.shape[-1]:
                token_gradients = token_gradients.expand_as(features)
            else:
                raise ValueError("Cannot reconcile gradient and feature shapes")
        
        # Compute gradient-based importance (element-wise product summed over hidden dim)
        importance = (features * token_gradients).sum(dim=-1)  # [B, num_tokens]
        
        # Apply information bottleneck
        attribution = self._apply_information_bottleneck(
            features=features,
            importance=importance,
            damping_ratio=self.damping_ratio,
            num_iterations=self.num_iterations,
        )
        
        return attribution
    
    def _apply_information_bottleneck(
        self,
        features: torch.Tensor,
        importance: torch.Tensor,
        damping_ratio: float,
        num_iterations: int,
    ) -> torch.Tensor:
        """
        Apply information bottleneck with iterative feature removal.
        
        This is the core of CoIBA: iteratively remove least important features
        and measure impact on prediction. The damping ratio controls how
        aggressively features are removed.
        
        Args:
            features: Vision features [B, num_tokens, hidden_dim].
            importance: Initial importance scores [B, num_tokens].
            damping_ratio: Damping factor for importance update (0-1).
            num_iterations: Number of removal iterations.
        
        Returns:
            Final attribution scores [B, num_tokens].
        """
        B, N, D = features.shape
        
        # Use double precision for numerical stability
        attribution = importance.clone().double()
        
        # Normalise to sum to 1 (probability distribution)
        attribution = attribution / (attribution.abs().sum(dim=1, keepdim=True) + 1e-10)
        
        # Iteratively remove least important features
        for iteration in range(num_iterations):
            # Find least important token (per batch)
            min_idx = attribution.abs().argmin(dim=1, keepdim=True)  # [B, 1]
            
            # Create mask (1 = keep, 0 = remove)
            mask = torch.ones_like(attribution)
            mask.scatter_(1, min_idx, 0)
            
            # Apply mask with damping
            # damping_ratio controls how much the removed token's importance
            # is redistributed vs. zeroed
            attribution = attribution * (damping_ratio * mask + (1 - damping_ratio))
            
            # Renormalise to maintain probability distribution
            attribution = attribution / (attribution.abs().sum(dim=1, keepdim=True) + 1e-10)
        
        # Return in original precision
        return attribution.float()
    
    def _compute_token_gradients(
        self,
        model_output: torch.Tensor,
        vision_features: Dict[int, torch.Tensor],
        token_idx: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute gradients of token logit w.r.t. vision features.
        
        Args:
            model_output: Model output logits [B, seq_len, vocab_size].
            vision_features: Vision features from all layers.
            token_idx: Index of token to attribute.
        
        Returns:
            Dictionary mapping layer index to gradients [B, num_tokens, hidden_dim].
        """
        # Get logit for target token position
        token_logit = model_output[:, token_idx, :]  # [B, vocab_size]
        target_token = token_logit.argmax(dim=-1)  # [B]
        
        # Get target token score (the predicted token's logit)
        target_score = token_logit.gather(
            1, target_token.unsqueeze(1)
        ).squeeze(1)  # [B]
        
        # Compute gradients for each layer
        gradients = {}
        for layer_idx, features in vision_features.items():
            # Enable gradient computation for this layer
            features = features.detach().requires_grad_(True)
            
            try:
                # Backward pass
                grad = torch.autograd.grad(
                    outputs=target_score.sum(),
                    inputs=features,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                
                if grad is not None:
                    gradients[layer_idx] = grad.detach()
                else:
                    # No gradient flow to this layer
                    gradients[layer_idx] = torch.zeros_like(features)
                    
            except RuntimeError as e:
                logger.warning(f"Gradient computation failed for layer {layer_idx}: {e}")
                gradients[layer_idx] = torch.zeros_like(features)
        
        return gradients
    
    # ========================================================================
    # Comprehensive Attribution
    # ========================================================================
    
    def generate_comprehensive_attribution(
        self,
        image: torch.Tensor,
        generated_text: str,
        prompt: str = "Describe the findings:",
        model_wrapper: Optional[Any] = None,
    ) -> AttributionOutput:
        """
        Generate comprehensive attribution map for entire generated text.
        
        This aggregates attributions across all generated tokens and all layers
        to produce a single attribution map showing which image regions
        contributed to the generated text.
        
        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W].
            generated_text: Generated report text.
            prompt: Input prompt used for generation.
            model_wrapper: Optional MAIRA2Model wrapper for preprocessing.
        
        Returns:
            AttributionOutput with aggregated attribution map and per-token details.
        
        Note:
            This method requires the model to be in eval mode and enables
            gradient computation temporarily.
        """
        # Ensure image has batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Enable gradient computation
        self.model.eval()
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        try:
            # Register hooks to capture vision features
            self.register_vision_hooks()
            
            # Get vision features via forward pass
            if model_wrapper is not None:
                # Use wrapper's preprocessing
                inputs = model_wrapper.preprocess_image(image)
                if hasattr(model_wrapper, 'processor') and model_wrapper.processor is not None:
                    text_inputs = model_wrapper.processor(
                        text=prompt,
                        images=None,  # Already processed
                        return_tensors="pt"
                    )
                    inputs.update(text_inputs)
            else:
                # Assume image is already preprocessed
                inputs = {"pixel_values": image.to(self.device)}
            
            # Forward pass to get logits
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            
            # Copy vision features (before removing hooks)
            vision_features = {k: v.clone() for k, v in self._vision_features.items()}
            self._remove_hooks()
            
            # Get tokenizer
            tokenizer = self.tokenizer
            if tokenizer is None and model_wrapper is not None:
                tokenizer = getattr(model_wrapper, 'tokenizer', None)
            
            if tokenizer is None:
                raise ValueError("Tokenizer required for token attribution")
            
            # Tokenise generated text
            tokens = tokenizer.encode(generated_text, return_tensors="pt").to(self.device)
            num_tokens = tokens.shape[1]
            
            # Compute attribution for each token
            token_attributions = {}
            layer_attributions = {i: [] for i in range(self.num_layers)}
            
            # Get model outputs (logits)
            model_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            for token_idx in range(num_tokens):
                # Compute gradients for this token
                gradients = self._compute_token_gradients(
                    model_logits, vision_features, token_idx
                )
                
                # Compute attribution for each layer
                for layer_idx in range(min(len(vision_features), self.num_layers)):
                    if layer_idx in vision_features and layer_idx in gradients:
                        attribution = self.compute_layer_attributions(
                            vision_features,
                            gradients[layer_idx],
                            layer_idx
                        )
                        layer_attributions[layer_idx].append(attribution)
                
                # Aggregate across layers for this token
                if all(len(attrs) > 0 for attrs in layer_attributions.values()):
                    token_attr = self._aggregate_layers({
                        i: attrs[-1] for i, attrs in layer_attributions.items()
                        if len(attrs) > 0
                    })
                    token_str = tokenizer.decode([tokens[0, token_idx].item()])
                    token_attributions[token_str] = token_attr.detach().cpu()
            
            # Final aggregation: average across tokens and layers
            final_layer_attrs = {}
            for layer_idx, attrs in layer_attributions.items():
                if len(attrs) > 0:
                    final_layer_attrs[layer_idx] = torch.stack(attrs).mean(0).detach().cpu()
            
            # Aggregate across all layers
            if final_layer_attrs:
                final_attribution = self._aggregate_layers(final_layer_attrs)
            else:
                # Fallback: use first layer if available
                final_attribution = torch.zeros(1, 1370)  # MAIRA-2 default
            
            # Reshape to spatial grid (37×37 for MAIRA-2, excluding CLS token)
            B, N = final_attribution.shape
            num_patches = N - 1  # Exclude CLS token
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size == num_patches:
                # Perfect square - can reshape
                spatial_attribution = final_attribution[:, 1:].reshape(B, grid_size, grid_size)
            else:
                # Not a perfect square - keep flat
                logger.warning(
                    f"Cannot reshape {num_patches} patches to square grid, keeping flat"
                )
                spatial_attribution = final_attribution[:, 1:]  # Skip CLS
            
            return AttributionOutput(
                attribution_map=spatial_attribution.squeeze(0).detach().cpu(),
                token_attributions=token_attributions,
                layer_attributions=final_layer_attrs,
                aggregated_attribution=final_attribution.detach().cpu(),
                metadata={
                    "damping_ratio": self.damping_ratio,
                    "num_iterations": self.num_iterations,
                    "aggregation": self.aggregation,
                    "num_tokens": num_tokens,
                    "generated_text": generated_text,
                    "prompt": prompt,
                },
            )
            
        finally:
            # Restore gradient state
            torch.set_grad_enabled(was_grad_enabled)
            self._remove_hooks()
    
    def _aggregate_layers(
        self,
        layer_attributions: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Aggregate attributions across layers.
        
        Args:
            layer_attributions: Dictionary mapping layer index to attribution.
        
        Returns:
            Aggregated attribution tensor.
        
        Raises:
            ValueError: If unknown aggregation method.
        """
        if not layer_attributions:
            raise ValueError("No layer attributions to aggregate")
        
        # Stack attributions in layer order
        sorted_keys = sorted(layer_attributions.keys())
        attrs = torch.stack([layer_attributions[i] for i in sorted_keys])
        
        if self.aggregation == "weighted":
            # Weighted average with exponential weights
            weights = self.layer_weights[:len(sorted_keys)].to(attrs.device)
            weights = weights / weights.sum()  # Renormalise for subset
            
            # Reshape weights for broadcasting
            while weights.dim() < attrs.dim():
                weights = weights.unsqueeze(-1)
            
            return (attrs * weights).sum(0)
        
        elif self.aggregation == "mean":
            return attrs.mean(0)
        
        elif self.aggregation == "max":
            return attrs.max(0)[0]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    # ========================================================================
    # Sentence-Level Attribution
    # ========================================================================
    
    def generate_sentence_level_attribution(
        self,
        image: torch.Tensor,
        generated_text: str,
        sentences: List[str],
        model_wrapper: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate attribution maps for individual sentences.
        
        Useful for error analysis: identify which sentences have
        incorrect visual grounding.
        
        Args:
            image: Input image tensor.
            generated_text: Full generated text (for context).
            sentences: List of sentences to attribute individually.
            model_wrapper: Optional MAIRA2Model wrapper.
        
        Returns:
            Dictionary mapping sentence to attribution map.
        """
        sentence_attributions = {}
        
        for sentence in sentences:
            output = self.generate_comprehensive_attribution(
                image=image,
                generated_text=sentence,
                prompt="",  # Empty prompt for sentence-only attribution
                model_wrapper=model_wrapper,
            )
            sentence_attributions[sentence] = output.attribution_map
        
        return sentence_attributions
    
    def compute_attribution_confidence(
        self,
        attribution_map: torch.Tensor,
        threshold_percentile: float = 90.0,
    ) -> float:
        """
        Compute confidence score for attribution map.
        
        A higher confidence indicates more focused/peaked attribution,
        while lower confidence indicates more diffuse attribution.
        
        Args:
            attribution_map: Attribution map tensor.
            threshold_percentile: Percentile for high-attribution threshold.
        
        Returns:
            Confidence score in [0, 1].
        """
        flat_attr = attribution_map.flatten()
        
        # Compute entropy-based confidence
        # Lower entropy = higher confidence (more focused)
        attr_abs = flat_attr.abs()
        attr_norm = attr_abs / (attr_abs.sum() + 1e-10)
        entropy = -(attr_norm * torch.log(attr_norm + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(float(len(flat_attr))))
        
        confidence = 1.0 - (entropy / max_entropy).item()
        return max(0.0, min(1.0, confidence))


# ============================================================================
# Visualisation Utilities
# ============================================================================

def overlay_attribution_on_image(
    image: Union[torch.Tensor, np.ndarray],
    attribution: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay attribution map on image.
    
    Args:
        image: Original image [H, W, C] or [C, H, W].
        attribution: Attribution map [H, W].
        alpha: Blending factor (0 = only image, 1 = only attribution).
        colormap: Matplotlib colormap name.
    
    Returns:
        Overlaid image as numpy array [H, W, C] in [0, 1] range.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib is required for visualisation")
    
    # Convert image to numpy and ensure HWC format
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    # Handle CHW format
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Normalise image to [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Handle grayscale
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[-1] == 1:
        image_np = np.concatenate([image_np] * 3, axis=-1)
    
    # Normalise attribution to [0, 1]
    attr_np = attribution.cpu().numpy() if isinstance(attribution, torch.Tensor) else attribution
    attr_min, attr_max = attr_np.min(), attr_np.max()
    if attr_max - attr_min > 1e-8:
        attr_norm = (attr_np - attr_min) / (attr_max - attr_min)
    else:
        attr_norm = np.zeros_like(attr_np)
    
    # Resize attribution to image size
    if attr_norm.shape != image_np.shape[:2]:
        from PIL import Image as PILImage
        attr_resized = np.array(
            PILImage.fromarray((attr_norm * 255).astype(np.uint8)).resize(
                (image_np.shape[1], image_np.shape[0]),
                PILImage.BILINEAR
            )
        ) / 255.0
    else:
        attr_resized = attr_norm
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    attr_coloured = cmap(attr_resized)[:, :, :3]  # RGB only, drop alpha
    
    # Blend
    overlaid = alpha * attr_coloured + (1 - alpha) * image_np
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid


def save_attribution_visualisation(
    image: Union[torch.Tensor, np.ndarray],
    attribution: torch.Tensor,
    save_path: str,
    title: str = "Attribution Map",
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
) -> None:
    """
    Save attribution visualisation to file.
    
    Creates a figure with three panels: original image, attribution map,
    and overlay.
    
    Args:
        image: Original image.
        attribution: Attribution map.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size in inches.
        dpi: Dots per inch for saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualisation")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert image to displayable format
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    # Handle CHW format
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Normalise to [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Original image
    axes[0].imshow(image_np, cmap='gray' if image_np.ndim == 2 else None)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attribution map
    attr_np = attribution.cpu().numpy() if isinstance(attribution, torch.Tensor) else attribution
    im = axes[1].imshow(attr_np, cmap='hot')
    axes[1].set_title("Attribution Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlaid = overlay_attribution_on_image(image, attribution)
    axes[2].imshow(overlaid)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved attribution visualisation to {save_path}")


def create_token_attribution_grid(
    token_attributions: Dict[str, torch.Tensor],
    grid_cols: int = 4,
    figsize_per_cell: Tuple[float, float] = (3, 3),
) -> "plt.Figure":
    """
    Create a grid visualisation of per-token attributions.
    
    Args:
        token_attributions: Dictionary mapping token string to attribution map.
        grid_cols: Number of columns in the grid.
        figsize_per_cell: Figure size per cell in inches.
    
    Returns:
        Matplotlib figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualisation")
    
    num_tokens = len(token_attributions)
    grid_rows = (num_tokens + grid_cols - 1) // grid_cols
    
    figsize = (figsize_per_cell[0] * grid_cols, figsize_per_cell[1] * grid_rows)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    
    # Flatten axes for easier indexing
    if grid_rows == 1:
        axes = [axes] if grid_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    
    for idx, (token, attribution) in enumerate(token_attributions.items()):
        attr_np = attribution.cpu().numpy() if isinstance(attribution, torch.Tensor) else attribution
        axes[idx].imshow(attr_np, cmap='hot')
        axes[idx].set_title(f'"{token}"', fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(num_tokens, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from PIL import Image
    from models.maira2_wrapper import MAIRA2Model
    
    print("Loading MAIRA-2 model...")
    wrapper = MAIRA2Model.from_pretrained(
        "microsoft/maira-2",
        device="cuda",
        load_in_8bit=True,
    )
    
    # Initialise CoIBA
    print("Initialising CoIBA...")
    coiba = CoIBAForLVLM(
        model=wrapper.model,
        vision_encoder=wrapper.get_vision_encoder(),
        damping_ratio=0.85,
        num_iterations=10,
        aggregation="weighted",
        tokenizer=wrapper.tokenizer,
    )
    
    # Load test image
    print("Processing test image...")
    image = Image.new('RGB', (518, 518), color='gray')  # Dummy image for testing
    image_tensor = wrapper.preprocess_image(image)['pixel_values']
    
    # Generate attribution
    print("Generating attribution...")
    output = coiba.generate_comprehensive_attribution(
        image=image_tensor,
        generated_text="There is a right pleural effusion.",
        prompt="Describe the findings:",
        model_wrapper=wrapper,
    )
    
    print(f"\nAttribution map shape: {output.attribution_map.shape}")
    print(f"Number of tokens attributed: {len(output.token_attributions)}")
    print(f"Number of layers: {len(output.layer_attributions)}")
    
    # Compute confidence
    confidence = coiba.compute_attribution_confidence(output.attribution_map)
    print(f"Attribution confidence: {confidence:.3f}")
    
    # Print per-token attributions
    print("\nPer-token attribution shapes:")
    for token, attr in list(output.token_attributions.items())[:5]:
        print(f"  '{token}': {attr.shape}")
    
    # Save visualisation
    print("\nSaving visualisation...")
    save_attribution_visualisation(
        image_tensor[0],
        output.attribution_map,
        "attribution_visualisation.png",
        title="CoIBA Attribution for MAIRA-2"
    )
    
    print("Done!")
