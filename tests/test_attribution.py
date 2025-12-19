"""Tests for attribution methods."""

import sys
from pathlib import Path

# Add project root to path for test execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn
from attribution import CoIBAForLVLM

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, images, input_ids, attention_mask):
        # Mock output
        batch_size = images.shape[0]
        logits = torch.randn(batch_size, 5, 100) # (B, Seq, Vocab)
        return type('Output', (), {'logits': logits})()

def test_coiba_initialization():
    model = MockModel()
    coiba = CoIBAForLVLM(model)
    assert coiba.model == model
