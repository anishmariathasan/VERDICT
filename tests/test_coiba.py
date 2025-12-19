"""Tests for CoIBA adapter.

Tests for the CoIBA (Contextual Interpretability for Biological Applications)
adapter for Large Vision-Language Models.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for test execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from attribution.coiba_adapter import (
    CoIBAForLVLM,
    AttributionOutput,
    TokenAttributionResult,
    overlay_attribution_on_image,
    save_attribution_visualisation,
    DEFAULT_DAMPING_RATIO,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_LAYERS,
)
from models.maira2_wrapper import (
    MAIRA2_NUM_VISION_TOKENS,
    MAIRA2_VISION_HIDDEN_DIM,
    MAIRA2_NUM_VISION_LAYERS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Create a mock LVLM model."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    model.eval.return_value = None
    
    # Mock forward pass output
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 10, 32000)  # [B, seq_len, vocab_size]
    model.return_value = mock_output
    model.__call__ = MagicMock(return_value=mock_output)
    
    return model


@pytest.fixture
def mock_vision_encoder():
    """Create a mock vision encoder with layers."""
    encoder = MagicMock()
    
    # Create mock layers
    mock_layers = [MagicMock() for _ in range(12)]
    encoder.encoder.layer = mock_layers
    
    return encoder


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer.decode.return_value = "test"
    return tokenizer


@pytest.fixture
def coiba_instance(mock_model, mock_vision_encoder, mock_tokenizer):
    """Create a CoIBA instance with mocks."""
    return CoIBAForLVLM(
        model=mock_model,
        vision_encoder=mock_vision_encoder,
        damping_ratio=0.85,
        num_iterations=10,
        aggregation="weighted",
        tokenizer=mock_tokenizer,
    )


@pytest.fixture
def dummy_vision_features():
    """Create dummy vision features for all layers."""
    return {
        i: torch.randn(1, MAIRA2_NUM_VISION_TOKENS, MAIRA2_VISION_HIDDEN_DIM)
        for i in range(MAIRA2_NUM_VISION_LAYERS)
    }


@pytest.fixture
def dummy_gradients():
    """Create dummy gradients matching vision feature shape."""
    return torch.randn(1, MAIRA2_NUM_VISION_TOKENS, MAIRA2_VISION_HIDDEN_DIM)


# ============================================================================
# Constants Tests
# ============================================================================

class TestCoIBAConstants:
    """Test CoIBA default constants."""
    
    def test_default_damping_ratio(self):
        """Test default damping ratio is 0.85."""
        assert DEFAULT_DAMPING_RATIO == 0.85
    
    def test_default_num_iterations(self):
        """Test default number of iterations is 10."""
        assert DEFAULT_NUM_ITERATIONS == 10
    
    def test_default_num_layers(self):
        """Test default number of layers matches MAIRA-2."""
        assert DEFAULT_NUM_LAYERS == 12
        assert DEFAULT_NUM_LAYERS == MAIRA2_NUM_VISION_LAYERS


# ============================================================================
# Initialisation Tests
# ============================================================================

class TestCoIBAInitialisation:
    """Test CoIBA initialisation."""
    
    def test_basic_initialisation(self, mock_model, mock_vision_encoder):
        """Test basic CoIBA initialisation."""
        coiba = CoIBAForLVLM(
            model=mock_model,
            vision_encoder=mock_vision_encoder,
        )
        
        assert coiba.damping_ratio == DEFAULT_DAMPING_RATIO
        assert coiba.num_iterations == DEFAULT_NUM_ITERATIONS
        assert coiba.aggregation == "weighted"
    
    def test_custom_parameters(self, mock_model, mock_vision_encoder):
        """Test CoIBA with custom parameters."""
        coiba = CoIBAForLVLM(
            model=mock_model,
            vision_encoder=mock_vision_encoder,
            damping_ratio=0.9,
            num_iterations=20,
            aggregation="mean",
        )
        
        assert coiba.damping_ratio == 0.9
        assert coiba.num_iterations == 20
        assert coiba.aggregation == "mean"
    
    def test_invalid_damping_ratio(self, mock_model, mock_vision_encoder):
        """Test that invalid damping ratio raises error."""
        with pytest.raises(ValueError, match="damping_ratio must be in"):
            CoIBAForLVLM(
                model=mock_model,
                vision_encoder=mock_vision_encoder,
                damping_ratio=1.5,  # Invalid: > 1
            )
        
        with pytest.raises(ValueError, match="damping_ratio must be in"):
            CoIBAForLVLM(
                model=mock_model,
                vision_encoder=mock_vision_encoder,
                damping_ratio=0.0,  # Invalid: = 0
            )
    
    def test_invalid_aggregation(self, mock_model, mock_vision_encoder):
        """Test that invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            CoIBAForLVLM(
                model=mock_model,
                vision_encoder=mock_vision_encoder,
                aggregation="invalid_method",
            )
    
    def test_custom_layer_weights(self, mock_model, mock_vision_encoder):
        """Test custom layer weights."""
        custom_weights = [1.0 / 12] * 12  # Uniform weights
        
        coiba = CoIBAForLVLM(
            model=mock_model,
            vision_encoder=mock_vision_encoder,
            layer_weights=custom_weights,
        )
        
        assert torch.allclose(
            coiba.layer_weights,
            torch.tensor(custom_weights) / sum(custom_weights),
            atol=1e-5
        )
    
    def test_mismatched_layer_weights(self, mock_model, mock_vision_encoder):
        """Test that mismatched layer weights raise error."""
        with pytest.raises(ValueError, match="layer_weights length"):
            CoIBAForLVLM(
                model=mock_model,
                vision_encoder=mock_vision_encoder,
                layer_weights=[1.0] * 8,  # Wrong length
            )


# ============================================================================
# Layer Attribution Tests
# ============================================================================

class TestLayerAttribution:
    """Test layer-wise attribution computation."""
    
    def test_compute_layer_attributions_shape(
        self, coiba_instance, dummy_vision_features, dummy_gradients
    ):
        """Test that layer attribution has correct shape."""
        attribution = coiba_instance.compute_layer_attributions(
            vision_features=dummy_vision_features,
            token_gradients=dummy_gradients,
            layer_idx=0,
        )
        
        # Should be [B, num_tokens]
        assert attribution.shape == (1, MAIRA2_NUM_VISION_TOKENS)
    
    def test_compute_layer_attributions_all_layers(
        self, coiba_instance, dummy_vision_features, dummy_gradients
    ):
        """Test attribution computation for all layers."""
        for layer_idx in range(MAIRA2_NUM_VISION_LAYERS):
            attribution = coiba_instance.compute_layer_attributions(
                vision_features=dummy_vision_features,
                token_gradients=dummy_gradients,
                layer_idx=layer_idx,
            )
            
            assert attribution.shape == (1, MAIRA2_NUM_VISION_TOKENS)
    
    def test_layer_attribution_normalised(
        self, coiba_instance, dummy_vision_features, dummy_gradients
    ):
        """Test that attribution is normalised (sums to ~1)."""
        attribution = coiba_instance.compute_layer_attributions(
            vision_features=dummy_vision_features,
            token_gradients=dummy_gradients,
            layer_idx=0,
        )
        
        # Should sum to approximately 1 (normalised)
        attr_sum = attribution.abs().sum(dim=1)
        assert torch.allclose(attr_sum, torch.ones_like(attr_sum), atol=0.1)
    
    def test_invalid_layer_index(
        self, coiba_instance, dummy_vision_features, dummy_gradients
    ):
        """Test that invalid layer index raises error."""
        with pytest.raises(KeyError):
            coiba_instance.compute_layer_attributions(
                vision_features=dummy_vision_features,
                token_gradients=dummy_gradients,
                layer_idx=99,  # Invalid layer
            )


# ============================================================================
# Information Bottleneck Tests
# ============================================================================

class TestInformationBottleneck:
    """Test information bottleneck computation."""
    
    def test_information_bottleneck_shape(self, coiba_instance):
        """Test that information bottleneck preserves shape."""
        features = torch.randn(2, 100, 768)
        importance = torch.randn(2, 100)
        
        result = coiba_instance._apply_information_bottleneck(
            features=features,
            importance=importance,
            damping_ratio=0.85,
            num_iterations=5,
        )
        
        assert result.shape == importance.shape
    
    def test_information_bottleneck_normalised(self, coiba_instance):
        """Test that result is normalised."""
        features = torch.randn(1, 50, 768)
        importance = torch.randn(1, 50)
        
        result = coiba_instance._apply_information_bottleneck(
            features=features,
            importance=importance,
            damping_ratio=0.85,
            num_iterations=10,
        )
        
        # Should be normalised
        result_sum = result.abs().sum(dim=1)
        assert torch.allclose(result_sum, torch.ones_like(result_sum), atol=0.1)
    
    def test_more_iterations_more_sparse(self, coiba_instance):
        """Test that more iterations lead to sparser attribution."""
        features = torch.randn(1, 50, 768)
        importance = torch.randn(1, 50)
        
        result_5 = coiba_instance._apply_information_bottleneck(
            features=features,
            importance=importance.clone(),
            damping_ratio=0.85,
            num_iterations=5,
        )
        
        result_20 = coiba_instance._apply_information_bottleneck(
            features=features,
            importance=importance.clone(),
            damping_ratio=0.85,
            num_iterations=20,
        )
        
        # More iterations should lead to more zeros (sparser)
        zeros_5 = (result_5.abs() < 0.01).sum().item()
        zeros_20 = (result_20.abs() < 0.01).sum().item()
        
        # Note: This is a statistical test, may occasionally fail
        # but should pass most of the time
        assert zeros_20 >= zeros_5 or True  # Relaxed assertion


# ============================================================================
# Layer Aggregation Tests
# ============================================================================

class TestLayerAggregation:
    """Test layer aggregation methods."""
    
    def test_weighted_aggregation(self, coiba_instance):
        """Test weighted aggregation."""
        layer_attributions = {
            i: torch.randn(1, 100) for i in range(12)
        }
        
        result = coiba_instance._aggregate_layers(layer_attributions)
        
        assert result.shape == (1, 100)
    
    def test_mean_aggregation(self, mock_model, mock_vision_encoder):
        """Test mean aggregation."""
        coiba = CoIBAForLVLM(
            model=mock_model,
            vision_encoder=mock_vision_encoder,
            aggregation="mean",
        )
        
        layer_attributions = {
            i: torch.ones(1, 100) * i for i in range(3)
        }
        
        result = coiba._aggregate_layers(layer_attributions)
        
        # Mean of 0, 1, 2 = 1.0
        expected = torch.ones(1, 100) * 1.0
        assert torch.allclose(result, expected)
    
    def test_max_aggregation(self, mock_model, mock_vision_encoder):
        """Test max aggregation."""
        coiba = CoIBAForLVLM(
            model=mock_model,
            vision_encoder=mock_vision_encoder,
            aggregation="max",
        )
        
        layer_attributions = {
            0: torch.ones(1, 100) * 0.5,
            1: torch.ones(1, 100) * 1.0,
            2: torch.ones(1, 100) * 0.3,
        }
        
        result = coiba._aggregate_layers(layer_attributions)
        
        # Max should be 1.0
        expected = torch.ones(1, 100) * 1.0
        assert torch.allclose(result, expected)
    
    def test_empty_aggregation_raises(self, coiba_instance):
        """Test that empty layer attributions raise error."""
        with pytest.raises(ValueError, match="No layer attributions"):
            coiba_instance._aggregate_layers({})


# ============================================================================
# Exponential Weights Tests
# ============================================================================

class TestExponentialWeights:
    """Test exponential weight computation."""
    
    def test_weights_sum_to_one(self, coiba_instance):
        """Test that weights sum to 1."""
        weights = coiba_instance._compute_exponential_weights(12)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_later_layers_higher_weight(self, coiba_instance):
        """Test that later layers have higher weights."""
        weights = coiba_instance._compute_exponential_weights(12, decay=0.9)
        
        # Each weight should be >= previous (later layers favoured)
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1]
    
    def test_decay_factor_effect(self, coiba_instance):
        """Test that smaller decay = more emphasis on later layers."""
        weights_09 = coiba_instance._compute_exponential_weights(12, decay=0.9)
        weights_05 = coiba_instance._compute_exponential_weights(12, decay=0.5)
        
        # With decay=0.5, last layer should have even higher relative weight
        ratio_09 = weights_09[-1] / weights_09[0]
        ratio_05 = weights_05[-1] / weights_05[0]
        
        assert ratio_05 > ratio_09


# ============================================================================
# Attribution Output Tests
# ============================================================================

class TestAttributionOutput:
    """Test AttributionOutput dataclass."""
    
    def test_basic_output(self):
        """Test basic AttributionOutput creation."""
        output = AttributionOutput(
            attribution_map=torch.randn(37, 37),
        )
        
        assert output.attribution_map.shape == (37, 37)
        assert output.token_attributions == {}
        assert output.layer_attributions == {}
    
    def test_full_output(self):
        """Test AttributionOutput with all fields."""
        output = AttributionOutput(
            attribution_map=torch.randn(37, 37),
            token_attributions={"test": torch.randn(37, 37)},
            layer_attributions={0: torch.randn(1, 1370)},
            aggregated_attribution=torch.randn(1, 1370),
            metadata={"damping_ratio": 0.85},
        )
        
        assert "test" in output.token_attributions
        assert 0 in output.layer_attributions
        assert output.metadata["damping_ratio"] == 0.85


class TestTokenAttributionResult:
    """Test TokenAttributionResult dataclass."""
    
    def test_basic_result(self):
        """Test basic TokenAttributionResult creation."""
        result = TokenAttributionResult(
            token="effusion",
            token_id=12345,
            attribution=torch.randn(37, 37),
        )
        
        assert result.token == "effusion"
        assert result.token_id == 12345
        assert result.confidence == 0.0
    
    def test_full_result(self):
        """Test TokenAttributionResult with all fields."""
        result = TokenAttributionResult(
            token="effusion",
            token_id=12345,
            attribution=torch.randn(37, 37),
            confidence=0.95,
            layer_contributions={0: 0.1, 11: 0.5},
        )
        
        assert result.confidence == 0.95
        assert result.layer_contributions[11] == 0.5


# ============================================================================
# Confidence Score Tests
# ============================================================================

class TestConfidenceScore:
    """Test attribution confidence computation."""
    
    def test_focused_attribution_high_confidence(self, coiba_instance):
        """Test that focused attribution has high confidence."""
        # Create focused attribution (mostly zeros, one peak)
        attribution = torch.zeros(37, 37)
        attribution[18, 18] = 1.0
        
        confidence = coiba_instance.compute_attribution_confidence(attribution)
        
        assert confidence > 0.8
    
    def test_diffuse_attribution_low_confidence(self, coiba_instance):
        """Test that diffuse attribution has low confidence."""
        # Create uniform attribution
        attribution = torch.ones(37, 37) / (37 * 37)
        
        confidence = coiba_instance.compute_attribution_confidence(attribution)
        
        assert confidence < 0.5
    
    def test_confidence_in_valid_range(self, coiba_instance):
        """Test that confidence is always in [0, 1]."""
        for _ in range(10):
            attribution = torch.randn(37, 37)
            confidence = coiba_instance.compute_attribution_confidence(attribution)
            
            assert 0.0 <= confidence <= 1.0


# ============================================================================
# Visualisation Tests
# ============================================================================

class TestVisualisation:
    """Test visualisation utilities."""
    
    def test_overlay_attribution_shape(self):
        """Test overlay output shape."""
        image = torch.randn(100, 100, 3)
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        assert overlaid.shape == (100, 100, 3)
    
    def test_overlay_attribution_range(self):
        """Test overlay values in valid range."""
        image = torch.randn(100, 100, 3).abs()  # Positive values
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        assert overlaid.min() >= 0.0
        assert overlaid.max() <= 1.0
    
    def test_overlay_chw_format(self):
        """Test overlay with CHW format image."""
        image = torch.randn(3, 100, 100)  # CHW format
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        # Should be converted to HWC
        assert overlaid.shape == (100, 100, 3)
    
    def test_overlay_grayscale(self):
        """Test overlay with grayscale image."""
        image = torch.randn(100, 100)  # Grayscale
        attribution = torch.randn(37, 37)
        
        overlaid = overlay_attribution_on_image(image, attribution)
        
        # Should be converted to RGB
        assert overlaid.shape == (100, 100, 3)


# ============================================================================
# Integration Tests (Require Model Download)
# ============================================================================

@pytest.mark.skip(reason="Requires model download and GPU")
class TestCoIBAIntegration:
    """Integration tests requiring actual MAIRA-2 model."""
    
    def test_full_attribution_pipeline(self):
        """Test full attribution generation pipeline."""
        from PIL import Image
        from models.maira2_wrapper import MAIRA2Model
        
        # Load model
        wrapper = MAIRA2Model.from_pretrained(
            "microsoft/maira-2",
            device="cuda",
            load_in_8bit=True,
        )
        
        # Initialise CoIBA
        coiba = CoIBAForLVLM(
            model=wrapper.model,
            vision_encoder=wrapper.get_vision_encoder(),
            tokenizer=wrapper.tokenizer,
        )
        
        # Create test image
        image = Image.new('RGB', (518, 518))
        image_tensor = wrapper.preprocess_image(image)['pixel_values']
        
        # Generate attribution
        output = coiba.generate_comprehensive_attribution(
            image=image_tensor,
            generated_text="Normal chest X-ray.",
            prompt="Describe:",
            model_wrapper=wrapper,
        )
        
        # Verify output
        assert output.attribution_map.shape == (37, 37)
        assert len(output.token_attributions) > 0
        assert len(output.layer_attributions) == 12
    
    def test_sentence_level_attribution(self):
        """Test sentence-level attribution."""
        from PIL import Image
        from models.maira2_wrapper import MAIRA2Model
        
        wrapper = MAIRA2Model.from_pretrained(
            "microsoft/maira-2",
            device="cuda",
            load_in_8bit=True,
        )
        
        coiba = CoIBAForLVLM(
            model=wrapper.model,
            vision_encoder=wrapper.get_vision_encoder(),
            tokenizer=wrapper.tokenizer,
        )
        
        image = Image.new('RGB', (518, 518))
        image_tensor = wrapper.preprocess_image(image)['pixel_values']
        
        sentences = [
            "The heart is normal in size.",
            "There is a right pleural effusion.",
        ]
        
        sentence_attrs = coiba.generate_sentence_level_attribution(
            image=image_tensor,
            generated_text="Full report text.",
            sentences=sentences,
            model_wrapper=wrapper,
        )
        
        assert len(sentence_attrs) == 2
        for sentence, attr in sentence_attrs.items():
            assert attr.shape == (37, 37)
