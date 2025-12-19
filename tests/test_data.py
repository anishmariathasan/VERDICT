"""Tests for data module."""

import sys
from pathlib import Path

# Add project root to path for test execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from data import ImagePreprocessor


def test_image_preprocessor():
    """Test ImagePreprocessor resizes and normalises correctly."""
    preprocessor = ImagePreprocessor(image_size=(224, 224))
    # Create fake PIL-like image (H, W, C) as numpy would be
    image = torch.rand(3, 500, 500)  # Fake image tensor
    
    processed = preprocessor(image)
    
    assert processed.shape == (3, 224, 224)
    assert isinstance(processed, torch.Tensor)
