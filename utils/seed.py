"""Reproducibility utilities for VERDICT project.

This module provides functions to ensure reproducible experiments
by setting random seeds across all relevant libraries.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    Optionally enables deterministic algorithms for full reproducibility.
    
    Args:
        seed: Random seed value. Default is 42.
        deterministic: If True, enables deterministic algorithms in PyTorch.
            This may impact performance but ensures reproducibility.
    
    Note:
        Setting deterministic=True may slow down training significantly
        for some operations. Consider disabling for hyperparameter search.
    
    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    # Python random module
    random.seed(seed)
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic algorithms
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass


def get_random_state() -> dict:
    """
    Capture current random state for checkpointing.
    
    Saves the random state from Python, NumPy, and PyTorch to allow
    exact resumption of training from a checkpoint.
    
    Returns:
        Dictionary containing random states from all libraries.
    
    Example:
        >>> state = get_random_state()
        >>> # Save state with checkpoint
        >>> torch.save({"random_state": state, "model": model.state_dict()}, "checkpoint.pt")
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state from checkpoint.
    
    Restores the random state to Python, NumPy, and PyTorch from a
    previously saved state dictionary.
    
    Args:
        state: Dictionary containing random states, as returned by
            get_random_state().
    
    Example:
        >>> checkpoint = torch.load("checkpoint.pt")
        >>> set_random_state(checkpoint["random_state"])
        >>> # Training will resume with exact same random sequence
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class SeedContext:
    """Context manager for temporarily setting a different seed."""
    
    def __init__(self, seed: int) -> None:
        """
        Initialise seed context.
        
        Args:
            seed: Temporary seed to use within context.
        """
        self.seed = seed
        self.original_state: Optional[dict] = None
    
    def __enter__(self) -> None:
        """Enter context and set temporary seed."""
        self.original_state = get_random_state()
        set_seed(self.seed)
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original random state."""
        if self.original_state is not None:
            set_random_state(self.original_state)
