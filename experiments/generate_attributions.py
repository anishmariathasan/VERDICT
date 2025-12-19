"""Script to generate attribution maps for a dataset.

Generates and saves attribution maps using specified methods (e.g., CoIBA, Integrated Gradients).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from utils import setup_logger, load_config
from data import MIMICCXRDataModule
from models import MAIRA2Model
from attribution import CoIBAForLVLM, visualise_attribution

logger = logging.getLogger(__name__)


def generate(config: dict) -> None:
    """
    Generate attributions.
    
    Args:
        config: Configuration dictionary.
    """
    device = torch.device(config["project"]["device"])
    
    # Data
    datamodule = MIMICCXRDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=1, # Process one by one for analysis
        max_samples=10 # Limit for demo
    )
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()
    
    # Model
    model = MAIRA2Model.from_pretrained(config["model"]["checkpoint"])
    model.to(device)
    model.eval()
    
    # Attribution method
    coiba = CoIBAForLVLM(model, config.get("coiba", {}))
    
    output_dir = Path(config["output"]["attributions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, batch in enumerate(tqdm(loader)):
        image = batch["images"].to(device)
        # Dummy input_ids for now, in reality would be generated or ground truth
        input_ids = torch.randint(0, 1000, (1, 10)).to(device) 
        
        # Generate attribution for a specific token (e.g., last one)
        attr = coiba.generate_attribution(
            image=image,
            input_ids=input_ids,
            target_token_idx=0, # Placeholder
            n_steps=10
        )
        
        # Visualise and save
        save_path = output_dir / f"sample_{i}_attr.png"
        visualise_attribution(
            image=image[0],
            attribution_map=attr[0],
            save_path=str(save_path),
            show=False
        )
        
    logger.info(f"Attributions saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attributions")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logger("verdict")
    
    generate(config)
