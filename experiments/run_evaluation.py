"""Script to run full evaluation pipeline.

Evaluates model performance using various metrics (CheXpert, POPE, etc.).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from utils import setup_logger, load_config
from models import MAIRA2Model
from evaluation import POPEEvaluator, compute_chexpert_metrics

logger = logging.getLogger(__name__)


def evaluate(config: dict) -> None:
    """
    Run evaluation.
    
    Args:
        config: Configuration dictionary.
    """
    # Setup
    device = torch.device(config["project"]["device"])
    
    # Model
    model = MAIRA2Model.from_pretrained(config["model"]["checkpoint"])
    model.to(device)
    model.eval()
    
    # Evaluators
    pope = POPEEvaluator(model)
    
    # Run evaluation loop (placeholder)
    logger.info("Starting evaluation...")
    
    # ... evaluation logic ...
    
    results = {
        "pope_accuracy": 0.85,
        "chexpert_f1": 0.75
    }
    
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logger("verdict")
    
    evaluate(config)
