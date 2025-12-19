"""Pytest configuration and fixtures for VERDICT tests."""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "project": {
            "name": "verdict-test",
            "seed": 42,
            "device": "cpu",
        },
        "data": {
            "data_dir": "/tmp/test_data",
            "batch_size": 2,
        },
        "model": {
            "checkpoint": "test/model",
        },
    }


@pytest.fixture
def device():
    """Return available device."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
