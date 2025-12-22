"""CoIBA (Comprehensive Information Bottleneck Attribution) adapted for LVLMs.

This module adapts the CoIBA framework from CVPR 2025 to work with generative
Large Vision-Language Models. The key adaptation is attributing to generated
TEXT TOKENS in medical reports rather than single class labels.

Reference:
    Hong et al. "Comprehensive Information Bottleneck for Unveiling Universal 
    Attribution to Interpret Vision Transformers" CVPR 2025 (Highlight)
    https://github.com/KU-HJH/CoIBA

Key Challenge:
    - Original CoIBA attributes to single class label using cross-entropy loss
    - Need to attribute to generated TEXT TOKENS in reports

Implementation Details (matching original CoIBA):
    1. Estimate mean/variance of features across a calibration dataset
    2. Learn shared alpha parameter via optimisation
    3. Apply information bottleneck: z = λ*x + (1-λ)*ε where ε ~ N(μ, σ²)
    4. Minimise: model_loss + β * information_loss (KL divergence)
    5. Aggregate attributions across multiple ViT layers

Architecture Assumptions (MAIRA-2):
    - Vision Encoder: Rad-DINO ViT-B/14 (12 layers)
    - Image Size: 518×518 → 37×37 patch grid
    - Vision Tokens: 1370 (1 CLS + 1369 patches)
    - Hidden Dimension: 768
"""

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BETA = 10.0  # Weight for information loss (from CoIBA paper)
DEFAULT_OPTIMISATION_STEPS = 10
DEFAULT_LR = 1.0  # High LR since few iterations
DEFAULT_BATCH_SIZE = 10
DEFAULT_INITIAL_ALPHA = 5.0
DEFAULT_MIN_STD = 0.01
DEFAULT_NUM_LAYERS = 12  # Rad-DINO ViT layers
DEFAULT_START_LAYER = 4  # CoIBA default: start from layer 4


# ============================================================================
# Welford Estimator for Running Mean/Variance
# ============================================================================

class WelfordEstimator(nn.Module):
    """
    Online estimation of mean and variance using Welford's algorithm.
    
    This matches the TorchWelfordEstimator from the original CoIBA implementation.
    Used to estimate feature distribution statistics before running attribution.
    """
    
    def __init__(self) -> None:
        """Initialise the Welford estimator."""
        super().__init__()
        self.shape: Optional[torch.Size] = None
        self.device: Optional[torch.device] = None
        self._n_samples: int = 0
        self.register_buffer('m', None)  # Running mean
        self.register_buffer('s', None)  # Running sum of squared differences
    
    def _init(self, shape: torch.Size, device: torch.device) -> None:
        """Initialise buffers with given shape."""
        self.shape = shape
        self.device = device
        self.m = torch.zeros(shape, device=device, dtype=torch.float64)
        self.s = torch.zeros(shape, device=device, dtype=torch.float64)
        self._n_samples = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Update estimates with new samples.
        
        Args:
            x: Feature tensor of shape [B, ...] where B is batch size.
        
        Returns:
            Input tensor unchanged.
        """
        x = x.detach().double()
        
        if self.shape is None:
            self._init(x.shape[1:], x.device)
        
        for xi in x:
            self._n_samples += 1
            delta = xi - self.m
            self.m = self.m + delta / self._n_samples
            delta2 = xi - self.m
            self.s = self.s + delta * delta2
        
        return x
    
    def mean(self) -> torch.Tensor:
        """Return the estimated mean."""
        if self.m is None:
            raise RuntimeError("No samples processed yet")
        return self.m.float()
    
    def var(self) -> torch.Tensor:
        """Return the estimated variance."""
        if self.s is None or self._n_samples < 2:
            raise RuntimeError("Need at least 2 samples for variance")
        return (self.s / (self._n_samples - 1)).float()
    
    def std(self) -> torch.Tensor:
        """Return the estimated standard deviation."""
        return torch.sqrt(self.var() + 1e-10)
    
    def n_samples(self) -> int:
        """Return number of samples processed."""
        return self._n_samples
    
    def reset(self) -> None:
        """Reset the estimator."""
        self.shape = None
        self.m = None
        self.s = None
        self._n_samples = 0


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
        capacity: Information capacity (bits) - measures how much info passes through.
        metadata: Additional metadata about the attribution computation.
    """
    attribution_map: torch.Tensor
    token_attributions: Dict[str, torch.Tensor] = field(default_factory=dict)
    layer_attributions: Dict[int, torch.Tensor] = field(default_factory=dict)
    capacity: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Forward Hook for IBA
# ============================================================================

class _IBAForwardHook:
    """Forward hook for Information Bottleneck Attribution."""
    
    def __init__(self, iba: "CoIBAForLVLM", layer_idx: int) -> None:
        self.iba = iba
        self.layer_idx = layer_idx
    
    def __call__(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Process the layer output through IBA."""
        return self.iba._forward_iba(outputs, self.layer_idx)


# ============================================================================
# Main CoIBA Implementation
# ============================================================================

class CoIBAForLVLM(nn.Module):
    """
    CoIBA (Comprehensive Information Bottleneck Attribution) for LVLMs.
    
    This implementation closely follows the original CoIBA paper and code,
    adapted for attributing to generated text tokens instead of class labels.
    
    Key Components (from original CoIBA):
        1. **Estimation Phase**: Estimate μ and σ of features across dataset
        2. **Shared Alpha**: Single learnable damping parameter across layers
        3. **Information Bottleneck**: z = λ*x + (1-λ)*ε where λ = sigmoid(α)
        4. **KL Divergence Loss**: Measures information flow through bottleneck
        5. **Cross-Layer Attribution**: Aggregate attributions from multiple layers
    
    Adaptation for LVLMs:
        - Uses token generation loss instead of cross-entropy classification
        - Attributes to each generated token separately
        - Aggregates across all tokens for sentence-level attribution
    
    Example:
        >>> # Setup
        >>> model = MAIRA2Model.from_pretrained("microsoft/maira-2")
        >>> coiba = CoIBAForLVLM(
        ...     model=model.model,
        ...     target_layers=["encoder.layer.4", "encoder.layer.8", "encoder.layer.11"],
        ...     beta=10.0,
        ... )
        >>> 
        >>> # Estimate feature statistics (do once)
        >>> coiba.estimate(model.model, dataloader, n_samples=1000)
        >>> 
        >>> # Generate attribution
        >>> attribution = coiba.analyze(
        ...     image_tensor,
        ...     generated_tokens,
        ...     model_loss_fn=lambda x, t: cross_entropy(model(x), t)
        ... )
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        target_layers: Optional[List[str]] = None,
        beta: float = DEFAULT_BETA,
        optimisation_steps: int = DEFAULT_OPTIMISATION_STEPS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        initial_alpha: float = DEFAULT_INITIAL_ALPHA,
        min_std: float = DEFAULT_MIN_STD,
        start_token_idx: int = 1,  # Skip CLS token
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialise CoIBA for LVLM.
        
        Args:
            model: The vision-language model.
            target_layers: List of layer names to apply IBA to.
                For ViT: ["blocks.4", "blocks.8", "blocks.11"] or similar.
            beta: Weight for information loss term (higher = more compression).
            optimisation_steps: Number of optimisation iterations per sample.
            lr: Learning rate for alpha optimisation.
            batch_size: Batch size for optimisation (input is expanded).
            initial_alpha: Initial value for alpha parameter.
            min_std: Minimum standard deviation to prevent division by zero.
            start_token_idx: Index to start from (1 = skip CLS token).
            device: Device to use.
        """
        super().__init__()
        
        self.model = model
        self.target_layers = target_layers or []
        self.beta = beta
        self.optimisation_steps = optimisation_steps
        self.lr = lr
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.min_std = min_std
        self.start_token_idx = start_token_idx  # st parameter in original
        
        # Device
        if device is not None:
            self.device = device
        elif model is not None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Alpha parameter (shared across layers) - initialised in _build()
        self.alpha: Optional[nn.Parameter] = None
        
        # Per-layer estimators for mean/variance
        self.layer_estimator: Dict[str, WelfordEstimator] = {}
        for layer_name in self.target_layers:
            self.layer_estimator[layer_name] = WelfordEstimator()
        
        # Hook management
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        
        # State flags
        self._restrict_flow = False
        self._estimate = False
        self._buffer_capacity: Optional[torch.Tensor] = None
        
        # Feature storage during forward pass
        self._layer_features: Dict[str, torch.Tensor] = {}
        self._layer_idx = 0
        
        logger.info(
            f"Initialised CoIBA with beta={beta}, "
            f"optimisation_steps={optimisation_steps}, lr={lr}"
        )
    
    # ========================================================================
    # Model Setup
    # ========================================================================
    
    def setup_model(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Setup IBA hooks on target layers.
        
        Args:
            model: The model to attach hooks to.
            target_layers: Layer names to hook. If None, uses self.target_layers.
        
        Returns:
            List of layer names that were hooked.
        """
        self.model = model
        
        if target_layers is not None:
            self.target_layers = target_layers
        
        # Remove existing hooks
        self.detach()
        
        # Find and hook target layers
        hooked_layers = []
        for layer_name in self.target_layers:
            module = self._get_module_by_name(model, layer_name)
            if module is not None:
                hook = _IBAForwardHook(self, layer_name)
                handle = module.register_forward_hook(hook)
                self._hook_handles.append(handle)
                hooked_layers.append(layer_name)
                
                # Create estimator if not exists
                if layer_name not in self.layer_estimator:
                    self.layer_estimator[layer_name] = WelfordEstimator()
            else:
                logger.warning(f"Layer '{layer_name}' not found in model")
        
        logger.info(f"Hooked {len(hooked_layers)} layers: {hooked_layers}")
        return hooked_layers
    
    def _get_module_by_name(
        self,
        model: nn.Module,
        name: str,
    ) -> Optional[nn.Module]:
        """Get a module by its name path (e.g., 'encoder.layer.4')."""
        try:
            module = model
            for part in name.split('.'):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, KeyError):
            return None
    
    def detach(self) -> None:
        """Remove all hooks to restore original model."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    # ========================================================================
    # Alpha Parameter
    # ========================================================================
    
    def _build(self) -> None:
        """
        Initialise alpha parameter based on estimated feature shape.
        
        Called after estimation to create the learnable alpha with correct shape.
        """
        if not self.layer_estimator:
            raise RuntimeError("No layer estimators - call setup_model first")
        
        # Get shape from first estimator
        first_key = next(iter(self.layer_estimator))
        estimator = self.layer_estimator[first_key]
        
        if estimator.n_samples() <= 0:
            raise RuntimeError(
                "Must estimate feature distribution before using bottleneck. "
                "Call estimate() first."
            )
        
        shape = estimator.shape
        device = estimator.device
        
        # Alpha shape: [num_tokens - start_idx, 1] for per-token attribution
        # This allows different importance for each spatial position
        if len(shape) == 2:
            # Shape: [num_tokens, hidden_dim]
            num_tokens = shape[0] - self.start_token_idx
            self.alpha = nn.Parameter(
                torch.full((num_tokens, 1), self.initial_alpha, device=device),
                requires_grad=True
            )
        elif len(shape) == 3:
            # Shape: [H, W, C] for CNN-style features
            self.alpha = nn.Parameter(
                torch.full((1, *shape[:-1], 1), self.initial_alpha, device=device),
                requires_grad=True
            )
        else:
            # Fallback
            self.alpha = nn.Parameter(
                torch.full(shape, self.initial_alpha, device=device),
                requires_grad=True
            )
        
        logger.info(f"Alpha initialised with shape: {self.alpha.shape}")
    
    def _reset_alpha(self) -> None:
        """Reset alpha to initial value for new sample."""
        if self.alpha is not None:
            self.alpha.data.fill_(self.initial_alpha)
    
    # ========================================================================
    # Estimation Phase
    # ========================================================================
    
    @contextmanager
    def enable_estimation(self) -> Iterator[None]:
        """Context manager to enable feature estimation mode."""
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False
    
    def reset_estimate(self) -> None:
        """Reset all estimators."""
        for estimator in self.layer_estimator.values():
            estimator.reset()
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 10000,
        device: Optional[torch.device] = None,
        reset: bool = True,
    ) -> None:
        """
        Estimate mean and variance of features using dataset.
        
        This must be called before using the bottleneck for attribution.
        Typically 10,000 samples gives good estimates.
        
        Args:
            model: The model to estimate features from.
            dataloader: DataLoader yielding (images, ...) batches.
            n_samples: Number of samples to use for estimation.
            device: Device to run on.
            reset: Whether to reset existing estimates.
        """
        if device is None:
            device = self.device
        
        if reset:
            self.reset_estimate()
        
        # Ensure hooks are set up
        if not self._hook_handles:
            self.setup_model(model, self.target_layers)
        
        model.eval()
        
        pbar = tqdm(dataloader, desc="Estimating feature statistics")
        
        with torch.no_grad(), self.enable_estimation():
            for batch in pbar:
                # Check if we have enough samples
                first_key = next(iter(self.layer_estimator))
                if self.layer_estimator[first_key].n_samples() >= n_samples:
                    break
                
                # Get images from batch
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Forward pass to collect features
                model(images.to(device))
                
                pbar.set_postfix(
                    samples=self.layer_estimator[first_key].n_samples()
                )
        
        # Build alpha parameter now that we know the feature shape
        if self.alpha is None:
            self._build()
        
        logger.info(
            f"Estimation complete: {self.layer_estimator[first_key].n_samples()} samples"
        )
    
    # ========================================================================
    # Information Bottleneck Forward
    # ========================================================================
    
    def _forward_iba(
        self,
        x: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Process features through information bottleneck.
        
        Args:
            x: Feature tensor from the layer.
            layer_name: Name of the layer.
        
        Returns:
            Processed features (potentially with noise added).
        """
        # Estimation mode: just collect statistics
        if self._estimate:
            self.layer_estimator[layer_name](x)
            return x
        
        # Restrict flow mode: apply information bottleneck
        if self._restrict_flow:
            return self._do_restrict_information(x, layer_name)
        
        return x
    
    def _do_restrict_information(
        self,
        x: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Apply information bottleneck to features.
        
        This implements the core IBA operation:
            z = λ * x + (1 - λ) * ε
        where:
            - λ = sigmoid(α) is the learned mask
            - ε ~ N(μ, σ²) is noise sampled from estimated distribution
        
        IMPORTANT: According to CoIBA paper (Equation 8, Section 3.4), the 
        variational upper bound is optimised by computing mutual information
        ONLY for the FIRST targeted layer (I[R_1; Z_1]), while applying noise
        to ALL targeted layers via the shared alpha parameter.
        
        Paper quote: "To compress the subsequent layers, the simplified objective 
        necessitates calculating only the mutual information of the first layer 
        I[R_1; Z_1]."
        
        Args:
            x: Input features [B, N, D] where N is num_tokens, D is hidden_dim.
            layer_name: Name of the layer for getting μ and σ.
        
        Returns:
            Noised features with same shape as input.
        """
        if self.alpha is None:
            raise RuntimeError("Alpha not initialised. Call _build() or estimate() first.")
        
        # Store original for potential CLS token preservation
        orig_x = x
        st = self.start_token_idx
        
        # Skip CLS token if configured
        if st != 0:
            x = x[:, st:, :] if x.dim() == 3 else x[:, st:, :, :]
        
        # Store features for this layer (needed for CoIBA's cross-layer aggregation)
        self._layer_features[layer_name] = x
        
        # Get estimated mean and std for this layer
        μ = self.layer_estimator[layer_name].mean()[st:]
        σ = self.layer_estimator[layer_name].std()[st:]
        σ = torch.clamp(σ, min=self.min_std)
        
        # Compute lambda (information mask) - shared across all layers
        λ = torch.sigmoid(self.alpha)
        
        # Expand lambda to match batch size
        if x.dim() == 3:
            λ = λ.expand(x.shape[0], -1, -1)
        else:
            λ = λ.expand(x.shape[0], -1, -1, -1)
        
        # Compute KL divergence ONLY for the FIRST targeted layer
        # This is the key insight from CoIBA: optimise I[R_1; Z_1] only
        # while applying noise to all layers via shared alpha
        if self.target_layers and layer_name == self.target_layers[0]:
            kl = self._kl_div(x, λ, μ, σ)
            self._buffer_capacity = kl  # Set (not accumulate) from first layer only
        
        # Sample noise from estimated distribution
        ε = torch.randn_like(x) * σ + μ
        
        # Apply bottleneck: z = λ * x + (1 - λ) * ε
        # This is applied to ALL layers
        z = λ * x + (1 - λ) * ε
        
        # Reattach CLS token if we skipped it
        if st != 0:
            if orig_x.dim() == 3:
                z = torch.cat([orig_x[:, :st, :], z], dim=1)
            else:
                z = torch.cat([orig_x[:, :st, :, :], z], dim=1)
        
        return z
    
    @staticmethod
    def _kl_div(
        x: torch.Tensor,
        λ: torch.Tensor,
        μ: torch.Tensor,
        σ: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence for information bottleneck.
        
        Measures the information capacity: KL(P(Z|X) || Q(Z))
        where Q(Z) = N(0, 1) and P(Z|X) is the bottleneck distribution.
        
        Args:
            x: Input features [B, N, D].
            λ: Information mask [B, N, 1] in [0, 1].
            μ: Estimated mean [N, D].
            σ: Estimated std [N, D].
        
        Returns:
            KL divergence tensor [B, N, D].
        """
        # Normalise input
        x_norm = (x - μ) / σ
        
        # Compute mean and variance of Z given X
        μ_z = λ * x_norm
        var_z = (1 - λ) ** 2
        log_var_z = torch.log(var_z + 1e-10)
        
        # KL divergence: KL(N(μ_z, var_z) || N(0, 1))
        # = 0.5 * (var_z + μ_z² - 1 - log(var_z))
        kl = 0.5 * (var_z + μ_z ** 2 - 1 - log_var_z)
        
        return kl
    
    def capacity(self) -> torch.Tensor:
        """
        Get the information capacity from last forward pass.
        
        Returns:
            Capacity tensor averaged over batch, shape depends on features.
        """
        if self._buffer_capacity is None:
            raise RuntimeError("No capacity computed. Run forward pass first.")
        return self._buffer_capacity.mean(dim=0)
    
    # ========================================================================
    # Analysis / Attribution
    # ========================================================================
    
    @contextmanager
    def restrict_flow(self) -> Iterator[None]:
        """Context manager to enable information restriction mode."""
        self._restrict_flow = True
        self._buffer_capacity = None
        self._layer_features = {}
        self._layer_idx = 0
        try:
            yield
        finally:
            self._restrict_flow = False
    
    def analyze(
        self,
        image: torch.Tensor,
        target_token_idx: Union[int, torch.Tensor],
        model_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        beta: Optional[float] = None,
        optimisation_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate attribution map for a given image and target token.
        
        This is the main entry point for computing attributions.
        
        Args:
            image: Input image tensor [1, C, H, W].
            target_token_idx: Index of target token to attribute to.
            model_loss_fn: Loss function taking (images, target) and returning loss.
                For LVLMs, this should compute the negative log-likelihood of
                generating the target token.
            beta: Override default beta (information loss weight).
            optimisation_steps: Override default optimisation steps.
            lr: Override default learning rate.
            batch_size: Override default batch size.
        
        Returns:
            Attribution map (saliency) with shape matching input spatial dims.
        """
        assert image.shape[0] == 1, "Can only analyse one sample at a time"
        
        # Use defaults if not specified
        beta = beta if beta is not None else self.beta
        optimisation_steps = optimisation_steps if optimisation_steps is not None else self.optimisation_steps
        lr = lr if lr is not None else self.lr
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Ensure alpha is built
        if self.alpha is None:
            self._build()
        
        # Move to device
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        image = image.to(self.device)
        
        if not isinstance(target_token_idx, torch.Tensor):
            target_token_idx = torch.tensor(target_token_idx)
        target_token_idx = target_token_idx.to(self.device)
        
        # Expand image for batch optimisation
        batch = image.expand(batch_size, -1, -1, -1)
        target = target_token_idx.expand(batch_size) if target_token_idx.dim() == 0 else target_token_idx
        
        # Reset alpha for this sample
        self._reset_alpha()
        
        # Setup optimiser
        optimiser = torch.optim.Adam([self.alpha], lr=lr)
        
        # Check estimation
        first_key = next(iter(self.layer_estimator))
        if self.layer_estimator[first_key].n_samples() < 1000:
            warnings.warn(
                f"Estimator only fitted on {self.layer_estimator[first_key].n_samples()} "
                f"samples. We recommend at least 1000, ideally 10000."
            )
        
        # Optimisation loop
        with self.restrict_flow():
            for step in range(optimisation_steps):
                optimiser.zero_grad()
                self._buffer_capacity = None  # Reset capacity
                
                # Forward pass with bottleneck active
                model_loss = model_loss_fn(batch, target)
                
                # Information loss (how much info passes through bottleneck)
                information_loss = self.capacity().mean()
                
                # Total loss
                loss = model_loss + beta * information_loss
                
                # Backward and update
                loss.backward()
                optimiser.step()
        
        # Extract saliency from capacity
        saliency = self._get_saliency(shape=image.shape[2:])
        
        return saliency
    
    def _get_saliency(
        self,
        mode: str = "saliency",
        shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Convert capacity to saliency map.
        
        Args:
            mode: "saliency" (scaled bits) or "capacity" (raw bits).
            shape: Optional output shape (H, W) for scaling.
        
        Returns:
            Saliency map tensor.
        """
        capacity = self.capacity().detach()
        
        # Sum over hidden dimension if present
        if capacity.dim() > 2:
            capacity = capacity.sum(dim=-1)
        
        # Convert to bits
        saliency = capacity / float(np.log(2))
        
        if mode == "saliency" and shape is not None:
            # Scale bits to pixel area
            h_out, w_out = saliency.shape[-2:] if saliency.dim() >= 2 else (int(np.sqrt(saliency.shape[0])),) * 2
            h_in, w_in = shape
            saliency = saliency * (h_out * w_out) / (h_in * w_in)
        
        return saliency
    
    # ========================================================================
    # High-Level API for LVLMs
    # ========================================================================
    
    def generate_token_attribution(
        self,
        image: torch.Tensor,
        generated_ids: torch.Tensor,
        model: nn.Module,
        tokenizer: Any,
        prompt_ids: Optional[torch.Tensor] = None,
    ) -> AttributionOutput:
        """
        Generate attribution for each token in generated sequence.
        
        This is the main API for LVLM attribution.
        
        Args:
            image: Input image [1, C, H, W].
            generated_ids: Generated token IDs [1, seq_len].
            model: The LVLM model.
            tokenizer: Tokenizer for decoding.
            prompt_ids: Optional prompt token IDs (to exclude from attribution).
        
        Returns:
            AttributionOutput with per-token and aggregated attribution maps.
        """
        # Ensure hooks are set up
        if not self._hook_handles:
            self.setup_model(model, self.target_layers)
        
        # Ensure estimation is done
        first_key = next(iter(self.layer_estimator))
        if self.layer_estimator[first_key].n_samples() < 100:
            raise RuntimeError(
                "Feature statistics not estimated. Call estimate() first with a dataloader."
            )
        
        # Determine which tokens to attribute (skip prompt)
        start_idx = prompt_ids.shape[1] if prompt_ids is not None else 0
        tokens_to_attribute = generated_ids[0, start_idx:]
        
        token_attributions = {}
        layer_attributions = {name: [] for name in self.target_layers}
        
        for idx, token_id in enumerate(tokens_to_attribute):
            # Create loss function for this token
            def model_loss_fn(images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                """Compute loss for generating the target token."""
                # This should be adapted to your specific model
                outputs = model(pixel_values=images)
                logits = outputs.logits[:, start_idx + idx, :]
                return F.cross_entropy(logits, target)
            
            # Generate attribution for this token
            target = token_id.expand(self.batch_size)
            saliency = self.analyze(image, target, model_loss_fn)
            
            # Store per-token attribution
            token_str = tokenizer.decode([token_id.item()])
            token_attributions[token_str] = saliency.cpu()
            
            # Store per-layer attributions
            for name in self.target_layers:
                if name in self._layer_features:
                    layer_attributions[name].append(
                        self._layer_features[name].mean(0).cpu()
                    )
        
        # Aggregate across tokens
        all_attrs = list(token_attributions.values())
        if all_attrs:
            aggregated = torch.stack(all_attrs).mean(0)
        else:
            aggregated = torch.zeros(37, 37)  # MAIRA-2 default
        
        # Reshape to spatial grid
        if aggregated.dim() == 1:
            grid_size = int(np.sqrt(len(aggregated)))
            if grid_size * grid_size == len(aggregated):
                aggregated = aggregated.reshape(grid_size, grid_size)
        
        return AttributionOutput(
            attribution_map=aggregated,
            token_attributions=token_attributions,
            layer_attributions={
                name: torch.stack(attrs).mean(0) if attrs else torch.zeros(1)
                for name, attrs in layer_attributions.items()
            },
            capacity=self._buffer_capacity.cpu() if self._buffer_capacity is not None else None,
            metadata={
                "beta": self.beta,
                "optimisation_steps": self.optimisation_steps,
                "n_tokens": len(tokens_to_attribute),
                "target_layers": self.target_layers,
            },
        )


# ============================================================================
# Utility Functions
# ============================================================================

def get_vit_layer_names(
    num_layers: int = 12,
    start_layer: int = 4,
    end_layer: int = 12,
    pattern: str = "blocks.{i}",
) -> List[str]:
    """
    Generate layer names for ViT models.
    
    Args:
        num_layers: Total number of layers.
        start_layer: First layer to include (CoIBA default: 4).
        end_layer: Last layer to include (exclusive).
        pattern: Pattern for layer names with {i} placeholder.
    
    Returns:
        List of layer names.
    """
    return [pattern.format(i=i) for i in range(start_layer, min(end_layer, num_layers))]


def postprocess_heatmap(
    heatmap: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Post-process attribution heatmap.
    
    Args:
        heatmap: Raw heatmap tensor.
        target_size: Optional size to resize to.
    
    Returns:
        Processed heatmap.
    """
    # Ensure positive values
    heatmap = heatmap - heatmap.min()
    
    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Resize if needed
    if target_size is not None:
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False,
        ).squeeze()
    
    return heatmap


# ============================================================================
# Visualisation
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
        from matplotlib import cm
    except ImportError:
        raise ImportError("matplotlib is required for visualisation")
    
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    # Handle CHW format
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Normalise image
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Handle grayscale
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    
    # Process attribution
    attr_np = attribution.cpu().numpy() if isinstance(attribution, torch.Tensor) else attribution
    attr_np = postprocess_heatmap(torch.from_numpy(attr_np)).numpy()
    
    # Resize to image size
    if attr_np.shape != image_np.shape[:2]:
        from PIL import Image as PILImage
        attr_np = np.array(
            PILImage.fromarray((attr_np * 255).astype(np.uint8)).resize(
                (image_np.shape[1], image_np.shape[0]),
                PILImage.BILINEAR
            )
        ) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    attr_coloured = cmap(attr_np)[:, :, :3]
    
    # Blend
    overlaid = alpha * attr_coloured + (1 - alpha) * image_np
    return np.clip(overlaid, 0, 1)


def save_attribution_visualisation(
    image: Union[torch.Tensor, np.ndarray],
    attribution: torch.Tensor,
    save_path: str,
    title: str = "CoIBA Attribution",
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
) -> None:
    """
    Save attribution visualisation to file.
    
    Args:
        image: Original image.
        attribution: Attribution map.
        save_path: Path to save figure.
        title: Figure title.
        figsize: Figure size in inches.
        dpi: Resolution.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualisation")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert image
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
        image_np = np.transpose(image_np, (1, 2, 0))
    
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


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CoIBA for LVLM - Example Usage")
    print("=" * 50)
    print("""
    # 1. Setup CoIBA
    coiba = CoIBAForLVLM(
        target_layers=["blocks.4", "blocks.8", "blocks.11"],
        beta=10.0,
    )
    
    # 2. Attach to model
    coiba.setup_model(model, target_layers)
    
    # 3. Estimate feature statistics (REQUIRED - do once)
    coiba.estimate(model, train_dataloader, n_samples=10000)
    
    # 4. Generate attribution
    attribution = coiba.analyze(
        image=image_tensor,
        target_token_idx=token_id,
        model_loss_fn=lambda x, t: model.compute_loss(x, t)
    )
    
    # 5. Visualise
    save_attribution_visualisation(image, attribution, "coiba_result.png")
    """)
