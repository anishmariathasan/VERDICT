"""Tests for models.

Tests for MAIRA-2 model wrapper and related components.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for test execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from models import (
    MAIRA2Config,
    MAIRA2Model,
    MAIRA2Output,
    MAIRA2_IMAGE_SIZE,
    MAIRA2_NUM_VISION_TOKENS,
    MAIRA2_NUM_VISION_LAYERS,
    MAIRA2_VISION_HIDDEN_DIM,
)


# ============================================================================
# MAIRA-2 Architecture Constant Tests
# ============================================================================

class TestMAIRA2Constants:
    """Test MAIRA-2 architecture constants."""
    
    def test_image_size_is_518(self):
        """CRITICAL: MAIRA-2 uses 518x518, NOT 224x224."""
        assert MAIRA2_IMAGE_SIZE == 518, (
            f"Expected 518, got {MAIRA2_IMAGE_SIZE}. "
            "MAIRA-2 uses 518x518 images, NOT 224x224!"
        )
    
    def test_vision_tokens_count(self):
        """Test vision tokens = 1 CLS + 37*37 patches = 1370."""
        expected_patches = 37 * 37  # 1369 patches
        expected_total = 1 + expected_patches  # 1370 tokens
        assert MAIRA2_NUM_VISION_TOKENS == expected_total, (
            f"Expected {expected_total}, got {MAIRA2_NUM_VISION_TOKENS}. "
            "Should be 1 CLS + 1369 patch tokens = 1370"
        )
    
    def test_vision_layers_count(self):
        """Test that Rad-DINO uses 12 ViT layers."""
        assert MAIRA2_NUM_VISION_LAYERS == 12, (
            f"Expected 12 layers, got {MAIRA2_NUM_VISION_LAYERS}"
        )
    
    def test_vision_hidden_dimension(self):
        """Test Rad-DINO hidden dimension is 768."""
        assert MAIRA2_VISION_HIDDEN_DIM == 768, (
            f"Expected 768, got {MAIRA2_VISION_HIDDEN_DIM}"
        )
    
    def test_patch_grid_calculation(self):
        """Test that 518/14 = 37 patches per dimension."""
        patch_size = 14
        num_patches_per_dim = MAIRA2_IMAGE_SIZE // patch_size
        assert num_patches_per_dim == 37, (
            f"Expected 37, got {num_patches_per_dim}"
        )


# ============================================================================
# MAIRA2Config Tests
# ============================================================================

class TestMAIRA2Config:
    """Test MAIRA2Config dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MAIRA2Config()
        assert config.checkpoint == "microsoft/maira-2"
        assert config.max_new_tokens == 512
        assert config.device == "cuda"
        assert config.load_in_8bit is False
        assert config.use_flash_attention is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MAIRA2Config(
            checkpoint="test/checkpoint",
            device="cpu",
            load_in_8bit=True,
            max_new_tokens=256,
        )
        assert config.checkpoint == "test/checkpoint"
        assert config.device == "cpu"
        assert config.load_in_8bit is True
        assert config.max_new_tokens == 256
    
    def test_config_prompts(self):
        """Test that default prompts are defined."""
        config = MAIRA2Config()
        assert "findings" in config.prompts
        assert "impression" in config.prompts
        assert "full_report" in config.prompts


# ============================================================================
# MAIRA2Output Tests
# ============================================================================

class TestMAIRA2Output:
    """Test MAIRA2Output dataclass."""
    
    def test_output_creation(self):
        """Test creating MAIRA2Output."""
        output = MAIRA2Output(
            generated_text="Test findings.",
            generated_ids=torch.tensor([1, 2, 3]),
        )
        assert output.generated_text == "Test findings."
        assert output.vision_features is None
        assert output.attention_maps is None
    
    def test_output_with_features(self):
        """Test MAIRA2Output with vision features."""
        vision_features = {
            0: torch.randn(1, MAIRA2_NUM_VISION_TOKENS, MAIRA2_VISION_HIDDEN_DIM),
            11: torch.randn(1, MAIRA2_NUM_VISION_TOKENS, MAIRA2_VISION_HIDDEN_DIM),
        }
        
        output = MAIRA2Output(
            generated_text="Test findings.",
            generated_ids=torch.tensor([1, 2, 3]),
            vision_features=vision_features,
        )
        
        assert output.vision_features is not None
        assert len(output.vision_features) == 2
        assert output.vision_features[0].shape == (1, 1370, 768)


# ============================================================================
# MAIRA2Model Tests
# ============================================================================

class TestMAIRA2Model:
    """Test MAIRA2Model wrapper."""
    
    def test_model_init(self):
        """Test model initialisation without loading."""
        model = MAIRA2Model()
        assert model.config.checkpoint == "microsoft/maira-2"
        assert model._is_loaded is False
    
    def test_model_repr(self):
        """Test model string representation."""
        model = MAIRA2Model()
        repr_str = repr(model)
        assert "MAIRA2Model" in repr_str
        assert "518x518" in repr_str
        assert "1370" in repr_str
    
    def test_model_not_loaded_error(self):
        """Test that methods raise error if model not loaded."""
        model = MAIRA2Model()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model._check_loaded()
    
    @patch('models.maira2_wrapper.AutoModelForCausalLM')
    @patch('models.maira2_wrapper.AutoProcessor')
    def test_model_loading_mock(self, mock_processor, mock_model):
        """Test model loading with mocked transformers."""
        # Setup mocks
        mock_model.from_pretrained.return_value = MagicMock()
        mock_processor.from_pretrained.return_value = MagicMock()
        
        model = MAIRA2Model(MAIRA2Config(device="cpu"))
        
        # This would normally load the model
        # Full test requires actual model or more sophisticated mocking


# ============================================================================
# Feature Extraction Tests
# ============================================================================

class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_expected_feature_shapes(self):
        """Test expected shapes for extracted features."""
        batch_size = 2
        
        # Expected vision feature shape per layer
        expected_shape = (batch_size, MAIRA2_NUM_VISION_TOKENS, MAIRA2_VISION_HIDDEN_DIM)
        
        # Create mock features
        features = torch.randn(*expected_shape)
        
        assert features.shape[1] == 1370, "Should have 1370 vision tokens"
        assert features.shape[2] == 768, "Hidden dim should be 768"
    
    def test_all_layers_extractable(self):
        """Test that all 12 layers can be indexed."""
        mock_features = {
            i: torch.randn(1, MAIRA2_NUM_VISION_TOKENS, MAIRA2_VISION_HIDDEN_DIM)
            for i in range(MAIRA2_NUM_VISION_LAYERS)
        }
        
        assert len(mock_features) == 12
        for i in range(12):
            assert i in mock_features


# ============================================================================
# Integration Tests (Require Model Download)
# ============================================================================

@pytest.mark.skip(reason="Requires model download and GPU")
class TestMAIRA2Integration:
    """Integration tests requiring actual MAIRA-2 model."""
    
    def test_load_model(self):
        """Test loading MAIRA-2 from HuggingFace."""
        model = MAIRA2Model.from_pretrained(
            "microsoft/maira-2",
            device="cuda",
            load_in_8bit=True,
        )
        assert model._is_loaded
    
    def test_generate_report(self):
        """Test report generation."""
        from PIL import Image
        
        model = MAIRA2Model.from_pretrained(
            "microsoft/maira-2",
            device="cuda",
            load_in_8bit=True,
        )
        
        # Create test image
        image = Image.new('RGB', (512, 512), color='gray')
        
        output = model.generate_report(image, extract_features=True)
        
        assert output.generated_text
        assert output.vision_features is not None
        
        # Verify feature shapes
        for layer_idx, features in output.vision_features.items():
            assert features.shape[1] == MAIRA2_NUM_VISION_TOKENS
            assert features.shape[2] == MAIRA2_VISION_HIDDEN_DIM
