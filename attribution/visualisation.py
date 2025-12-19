"""Visualisation utilities for attribution maps in VERDICT project.

This module provides functions to visualise attribution maps overlaid on
medical images, supporting various colourmaps and plotting styles."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


def visualise_attribution(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    attribution_map: Union[torch.Tensor, np.ndarray],
    title: Optional[str] = None,
    cmap: str = "jet",
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualise an attribution map overlaid on the original image.
    
    Args:
        image: Original input image.
        attribution_map: Attribution map (heatmap).
        title: Plot title.
        cmap: Colourmap for the heatmap.
        alpha: Transparency of the heatmap overlay.
        save_path: Path to save the figure.
        show: Whether to display the plot.
    
    Returns:
        Matplotlib Figure object.
    """
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().detach().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Normalise image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
        
    # Process attribution map
    if isinstance(attribution_map, torch.Tensor):
        attribution_map = attribution_map.cpu().detach().numpy()
    
    # If attribution map has channels, aggregate them (e.g., mean or max)
    if attribution_map.ndim == 3:
        attribution_map = np.mean(attribution_map, axis=0)
        
    # Resize attribution map to match image size if needed
    if attribution_map.shape != image.shape[:2]:
        if cv2 is not None:
            attribution_map = cv2.resize(
                attribution_map, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # Fallback using scipy or simple scaling if cv2 missing
            pass

    # Normalise attribution map for visualisation
    attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    ax.imshow(attribution_map, cmap=cmap, alpha=alpha)
    
    if title:
        ax.set_title(title)
    ax.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig


def plot_attribution_grid(
    images: List[Union[torch.Tensor, np.ndarray]],
    attributions: List[Union[torch.Tensor, np.ndarray]],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a grid of attribution visualisations.
    
    Args:
        images: List of images.
        attributions: List of corresponding attribution maps.
        titles: List of titles for each subplot.
        cols: Number of columns in the grid.
        save_path: Path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        
        # Prepare image
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().detach().numpy()
        if img.max() > 1.0:
            img = img / 255.0
            
        # Prepare attribution
        attr = attributions[i]
        if isinstance(attr, torch.Tensor):
            attr = attr.cpu().detach().numpy()
        if attr.ndim == 3:
            attr = np.mean(attr, axis=0)
            
        # Resize attr if needed (simplified)
        
        # Normalise attr
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        
        ax.imshow(img, cmap="gray")
        ax.imshow(attr, cmap="jet", alpha=0.5)
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis("off")
        
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig
