"""Tests for CoIBA adapter.

Tests for the CoIBA (Comprehensive Information Bottleneck Attribution)
adapter for Large Vision-Language Models.

Reference:
    Hong et al. "Comprehensive Information Bottleneck for Unveiling Universal 
    Attribution to Interpret Vision Transformers" CVPR 2025
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for test execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn
import numpy as np

from attribution.coiba_adapter import (
    CoIBAForLVLM,
    AttributionOutput,
    WelfordEstimator,
    overlay_attribution_on_image,
    get_vit_layer_names,
    postprocess_heatmap,
    DEFAULT_BETA,
    DEFAULT_OPTIMISATION_STEPS,
    DEFAULT_LR,
    DEFAULT_INITIAL_ALPHA,
)

# Optional import for MAIRA-2 constants (allow tests to run without full model)
try:
    from models.maira2_wrapper import (
        MAIRA2_NUM_VISION_TOKENS,
        MAIRA2_VISION_HIDDEN_DIM,
        MAIRA2_NUM_VISION_LAYERS,
    )
except ImportError:
    # Default values for MAIRA-2
    MAIRA2_NUM_VISION_TOKENS = 1370
    MAIRA2_VISION_HIDDEN_DIM = 768
    MAIRA2_NUM_VISION_LAYERS = 12


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Create a mock LVLM model with named modules."""
    model = MagicMock(spec=nn.Module)
    
    # Create mock transformer blocks
    blocks = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])
    model.blocks = blocks
    
    # Make parameters() work
    model.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])
    
    # Mock forward pass
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 10, 32000)
    model.return_value = mock_output
    
    return model


@pytest.fixture
def simple_model():
    """Create a simple real model for testing hooks."""
    class SimpleViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Linear(768, 768) for _ in range(12)
            ])
        
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x
    
    return SimpleViT()


@pytest.fixture
def coiba_instance():
    """Create a CoIBA instance with default settings."""
    return CoIBAForLVLM(
        target_layers=["blocks.4", "blocks.8", "blocks.11"],
        beta=10.0,
        optimisation_steps=10,
    )


# ============================================================================
# WelfordEstimator Tests
# ============================================================================

class TestWelfordEstimator:
    """Test the Welford online estimator."""
    
    def test_init(self):
        """Test estimator initialisation."""
        estimator = WelfordEstimator()
        assert estimator.shape is None
        assert estimator.n_samples() == 0
    
    def test_single_sample(self):
        """Test with a single sample."""
        estimator = WelfordEstimator()
        x = torch.randn(1, 100, 768)
        
        estimator(x)
        
        assert estimator.n_samples() == 1
        assert estimator.shape == (100, 768)
    
    def test_mean_estimation(self):
        """Test mean estimation accuracy."""
        estimator = WelfordEstimator()
        
        # Generate samples with known mean
        true_mean = 5.0
        samples = torch.randn(100, 50, 10) + true_mean
        
        for sample in samples:
            estimator(sample.unsqueeze(0))
        
        estimated_mean = estimator.mean()
        
        # Mean should be close to true_mean
        assert torch.allclose(
            estimated_mean.mean(),
            torch.tensor(true_mean),
            atol=0.5
        )
    
    def test_variance_estimation(self):
        """Test variance estimation accuracy."""
        estimator = WelfordEstimator()
        
        # Generate samples with known variance
        true_std = 2.0
        samples = torch.randn(100, 50, 10) * true_std
        
        for sample in samples:
            estimator(sample.unsqueeze(0))
        
        estimated_std = estimator.std()
        
        # Std should be close to true_std
        assert torch.allclose(
            estimated_std.mean(),
            torch.tensor(true_std),
            atol=0.5
        )
    
    def test_reset(self):
        """Test estimator reset."""
        estimator = WelfordEstimator()
        x = torch.randn(10, 100, 768)
        
        for xi in x:
            estimator(xi.unsqueeze(0))
        
        assert estimator.n_samples() == 10
        
        estimator.reset()
        
        assert estimator.n_samples() == 0
        assert estimator.shape is None


# ============================================================================
# CoIBAForLVLM Initialisation Tests
# ============================================================================

class TestCoIBAInitialisation:
    """Test CoIBA initialisation."""
    
    def test_default_init(self):
        """Test default initialisation."""
        coiba = CoIBAForLVLM()
        
        assert coiba.beta == DEFAULT_BETA
        assert coiba.optimisation_steps == DEFAULT_OPTIMISATION_STEPS
        assert coiba.lr == DEFAULT_LR
        assert coiba.alpha is None  # Not built yet
    
    def test_custom_init(self):
        """Test custom initialisation."""
        coiba = CoIBAForLVLM(
            target_layers=["blocks.4", "blocks.8"],
            beta=5.0,
            optimisation_steps=20,
            lr=0.5,
        )
        
        assert coiba.beta == 5.0
        assert coiba.optimisation_steps == 20
        assert coiba.lr == 0.5
        assert len(coiba.target_layers) == 2
    
    def test_layer_estimators_created(self):
        """Test that layer estimators are created for each target layer."""
        target_layers = ["blocks.4", "blocks.8", "blocks.11"]
        coiba = CoIBAForLVLM(target_layers=target_layers)
        
        assert len(coiba.layer_estimator) == 3
        for layer_name in target_layers:
            assert layer_name in coiba.layer_estimator
            assert isinstance(coiba.layer_estimator[layer_name], WelfordEstimator)


# ============================================================================
# Hook Management Tests
# ============================================================================

class TestHookManagement:
    """Test hook registration and removal."""
    
    def test_setup_model(self, simple_model):
        """Test setting up hooks on model."""
        coiba = CoIBAForLVLM(target_layers=["blocks.4", "blocks.8"])
        
        hooked = coiba.setup_model(simple_model, ["blocks.4", "blocks.8"])
        
        assert len(hooked) == 2
        assert len(coiba._hook_handles) == 2
    
    def test_detach(self, simple_model):
        """Test removing hooks."""
        coiba = CoIBAForLVLM(target_layers=["blocks.4"])
        coiba.setup_model(simple_model)
        
        assert len(coiba._hook_handles) == 1
        
        coiba.detach()
        
        assert len(coiba._hook_handles) == 0
    
    def test_invalid_layer_warning(self, simple_model, caplog):
        """Test warning for invalid layer names."""
        coiba = CoIBAForLVLM(target_layers=["invalid.layer"])
        
        hooked = coiba.setup_model(simple_model)
        
        assert len(hooked) == 0
        assert "not found" in caplog.text


# ============================================================================
# KL Divergence Tests
# ============================================================================

class TestKLDivergence:
    """Test KL divergence computation."""
    
    def test_kl_div_shape(self):
        """Test KL divergence output shape."""
        x = torch.randn(2, 100, 768)
        λ = torch.sigmoid(torch.randn(2, 100, 1))
        μ = torch.randn(100, 768)
        σ = torch.abs(torch.randn(100, 768)) + 0.01
        
        kl = CoIBAForLVLM._kl_div(x, λ, μ, σ)
        
        assert kl.shape == x.shape
    
    def test_kl_div_positive(self):
        """Test that KL divergence is non-negative."""
        x = torch.randn(2, 100, 768)
        λ = torch.sigmoid(torch.randn(2, 100, 1))
        μ = torch.randn(100, 768)
        σ = torch.abs(torch.randn(100, 768)) + 0.01
        
        kl = CoIBAForLVLM._kl_div(x, λ, μ, σ)
        
        # KL divergence should be non-negative
        assert (kl >= -1e-5).all()  # Small tolerance for numerical issues
    
    def test_kl_div_zero_lambda(self):
        """Test KL when lambda is 0 (no information passes)."""
        x = torch.randn(1, 100, 768)
        λ = torch.zeros(1, 100, 1)  # No information passes
        μ = torch.zeros(100, 768)
        σ = torch.ones(100, 768)
        
        kl = CoIBAForLVLM._kl_div(x, λ, μ, σ)
        
        # When λ=0, μ_z=0 and var_z=1, so KL should be ~0
        assert kl.mean() < 0.1


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_vit_layer_names_default(self):
        """Test default ViT layer name generation."""
        names = get_vit_layer_names()
        
        assert len(names) == 8  # 4 to 12 (exclusive)
        assert names[0] == "blocks.4"
        assert names[-1] == "blocks.11"
    
    def test_get_vit_layer_names_custom(self):
        """Test custom layer name generation."""
        names = get_vit_layer_names(
            num_layers=6,
            start_layer=0,
            end_layer=6,
            pattern="encoder.layer.{i}"
        )
        
        assert len(names) == 6
        assert names[0] == "encoder.layer.0"
        assert names[-1] == "encoder.layer.5"
    
    def test_postprocess_heatmap_normalisation(self):
        """Test heatmap normalisation."""
        heatmap = torch.randn(37, 37) * 10 - 5  # Range roughly [-5, 5]
        
        processed = postprocess_heatmap(heatmap)
        
        assert processed.min() >= 0
        assert processed.max() <= 1
    
    def test_postprocess_heatmap_resize(self):
        """Test heatmap resizing."""
        heatmap = torch.randn(37, 37)
        
        processed = postprocess_heatmap(heatmap, target_size=(224, 224))
        
        assert processed.shape == (224, 224)


# ============================================================================
# Attribution Output Tests
# ============================================================================

class TestAttributionOutput:
    """Test AttributionOutput dataclass."""
    
    def test_basic_output(self):
        """Test basic output creation."""
        output = AttributionOutput(
            attribution_map=torch.randn(37, 37),
        )
        
        assert output.attribution_map.shape == (37, 37)
        assert output.token_attributions == {}
        assert output.layer_attributions == {}
        assert output.capacity is None
    
    def test_full_output(self):
        """Test output with all fields."""
        output = AttributionOutput(
            attribution_map=torch.randn(37, 37),
            token_attributions={"the": torch.randn(37, 37)},
            layer_attributions={"blocks.4": torch.randn(100, 768)},
            capacity=torch.randn(100, 768),
            metadata={"beta": 10.0, "n_tokens": 5},
        )
        
        assert "the" in output.token_attributions
        assert "blocks.4" in output.layer_attributions
        assert output.capacity is not None
        assert output.metadata["beta"] == 10.0


# ============================================================================
# Constants Tests
# ============================================================================

class TestConstants:
    """Test that constants match CoIBA paper defaults."""
    
    def test_default_beta(self):
        """Beta should be 10 as in CoIBA paper."""
        assert DEFAULT_BETA == 10.0
    
    def test_default_optimisation_steps(self):
        """Optimisation steps should be 10."""
        assert DEFAULT_OPTIMISATION_STEPS == 10
    
    def test_default_lr(self):
        """Learning rate should be 1.0 (high for few iterations)."""
        assert DEFAULT_LR == 1.0
    
    def test_default_initial_alpha(self):
        """Initial alpha should be 5.0."""
        assert DEFAULT_INITIAL_ALPHA == 5.0


# ============================================================================
# Visualisation Tests
# ============================================================================

class TestVisualisation:
    """Test visualisation utilities."""
    
    def test_overlay_shape(self):
        """Test overlay output shape."""
        image = torch.randn(100, 100, 3).abs()
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        assert overlaid.shape == (100, 100, 3)
    
    def test_overlay_range(self):
        """Test overlay values in valid range."""
        image = torch.randn(100, 100, 3).abs()
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        assert overlaid.min() >= 0.0
        assert overlaid.max() <= 1.0
    
    def test_overlay_chw_format(self):
        """Test with CHW format input."""
        image = torch.randn(3, 100, 100).abs()
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        assert overlaid.shape == (100, 100, 3)


# ============================================================================
# Context Manager Tests
# ============================================================================

class TestContextManagers:
    """Test context managers."""
    
    def test_enable_estimation(self):
        """Test estimation context manager."""
        coiba = CoIBAForLVLM()
        
        assert coiba._estimate is False
        
        with coiba.enable_estimation():
            assert coiba._estimate is True
        
        assert coiba._estimate is False
    
    def test_restrict_flow(self):
        """Test restrict flow context manager."""
        coiba = CoIBAForLVLM()
        
        assert coiba._restrict_flow is False
        
        with coiba.restrict_flow():
            assert coiba._restrict_flow is True
        
        assert coiba._restrict_flow is False


# ============================================================================
# Integration Tests (Require Model)
# ============================================================================

@pytest.mark.skip(reason="Requires full model and GPU")
class TestCoIBAIntegration:
    """Integration tests requiring actual model."""
    
    def test_full_pipeline(self):
        """Test complete CoIBA pipeline."""
        from models.maira2_wrapper import MAIRA2Model
        
        # Load model
        model = MAIRA2Model.from_pretrained(
            "microsoft/maira-2",
            device="cuda",
            load_in_8bit=True,
        )
        
        # Setup CoIBA
        coiba = CoIBAForLVLM(
            target_layers=get_vit_layer_names(12, 4, 12, "blocks.{i}"),
            beta=10.0,
        )
        coiba.setup_model(model.model)
        
        # Would need dataloader for estimation
        # coiba.estimate(model.model, dataloader, n_samples=1000)
        
        # Then analyze
        # attribution = coiba.analyze(image, token_id, loss_fn)
    
    def test_attribution_shape(self):
        """Test attribution output shapes match MAIRA-2."""
        # Attribution map should be 37x37 for MAIRA-2 patch grid
        pass


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_estimation_error(self):
        """Test error when using bottleneck without estimation."""
        coiba = CoIBAForLVLM(target_layers=["blocks.4"])
        
        with pytest.raises(RuntimeError, match="Must estimate"):
            coiba._build()
    
    def test_empty_target_layers(self):
        """Test with no target layers."""
        coiba = CoIBAForLVLM(target_layers=[])
        
        assert len(coiba.layer_estimator) == 0
    
    def test_capacity_without_forward(self):
        """Test capacity() without forward pass."""
        coiba = CoIBAForLVLM()
        
        with pytest.raises(RuntimeError, match="No capacity"):
            coiba.capacity()
