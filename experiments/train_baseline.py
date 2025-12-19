"""Script to train or fine-tune baseline models.

This script handles the training loop for baseline hallucination mitigation methods
like UAC (Unified Attention Calibration).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import setup_logger, set_seed, load_config
from data import MIMICCXRDataModule
from models import MAIRA2Model
from baselines import UnifiedAttentionCalibration

logger = logging.getLogger(__name__)


def train(config: dict) -> None:
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary.
    """
    # Setup
    device = torch.device(config["project"]["device"])
    set_seed(config["project"]["seed"])
    
    # Data
    datamodule = MIMICCXRDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    
    # Model
    model = MAIRA2Model.from_pretrained(config["model"]["checkpoint"])
    model.to(device)
    
    # Baseline method (e.g., UAC)
    # If UAC involves learnable parameters, we initialise it here
    # For simple UAC, it might just be inference-time calibration, 
    # but let's assume we are fine-tuning something.
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"])
    )
    
    # Training loop
    model.train()
    for epoch in range(config["training"]["max_epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['training']['max_epochs']}")
        
        for batch in tqdm(train_loader):
            images = batch["images"].to(device)
            # ... process batch ...
            
            # loss = ...
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            
        # Validation...
        
    # Save checkpoint
    save_path = Path(config["output"]["checkpoints_dir"]) / "baseline_model.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--method", type=str, default="uac", help="Baseline method")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logger("verdict", log_file=Path(config["output"]["logs_dir"]) / "train.log")
    
    train(config)
