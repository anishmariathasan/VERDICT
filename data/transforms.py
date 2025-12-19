"""Data augmentation and transformation utilities for VERDICT project.

This module provides PyTorch-compatible data transformations for
training, validation, and testing of medical imaging models.
"""

import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except ImportError:
    T = None
    TF = None


def get_train_transforms(
    image_size: Tuple[int, int] = (518, 518),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    augmentation_strength: str = "light",
) -> Callable:
    """
    Get training data transformations with augmentation.
    
    Provides a transform pipeline suitable for training medical
    imaging models with appropriate augmentations.
    
    Args:
        image_size: Target (height, width) for output images.
        mean: Normalisation mean values.
        std: Normalisation standard deviation values.
        augmentation_strength: Level of augmentation - 'none', 'light', or 'strong'.
    
    Returns:
        Callable transform pipeline.
    
    Example:
        >>> transform = get_train_transforms(image_size=(518, 518))
        >>> augmented_image = transform(pil_image)
    """
    if T is None:
        raise ImportError("torchvision is required for transforms")
    
    transforms_list = []
    
    # Resize and pad to target size
    transforms_list.append(ResizeAndPad(image_size))
    
    # Augmentations based on strength
    if augmentation_strength in ("light", "strong"):
        # Random horizontal flip (common in chest X-rays)
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        
        # Slight rotation
        transforms_list.append(T.RandomRotation(degrees=5))
        
        # Colour jitter (subtle for medical images)
        transforms_list.append(
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
        )
    
    if augmentation_strength == "strong":
        # Random affine transformations
        transforms_list.append(
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5,
            )
        )
        
        # Random erasing (simulate occlusions)
        transforms_list.append(T.RandomErasing(p=0.1, scale=(0.02, 0.1)))
    
    # Convert to tensor and normalise
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    return T.Compose(transforms_list)


def get_val_transforms(
    image_size: Tuple[int, int] = (518, 518),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Callable:
    """
    Get validation data transformations (no augmentation).
    
    Args:
        image_size: Target (height, width) for output images.
        mean: Normalisation mean values.
        std: Normalisation standard deviation values.
    
    Returns:
        Callable transform pipeline.
    """
    if T is None:
        raise ImportError("torchvision is required for transforms")
    
    return T.Compose([
        ResizeAndPad(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = (518, 518),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Callable:
    """
    Get test data transformations (identical to validation).
    
    Args:
        image_size: Target (height, width) for output images.
        mean: Normalisation mean values.
        std: Normalisation standard deviation values.
    
    Returns:
        Callable transform pipeline.
    """
    return get_val_transforms(image_size, mean, std)


class ResizeAndPad:
    """
    Transform to resize image and pad to target size.
    
    Maintains aspect ratio by resizing based on shortest edge,
    then centre-padding to achieve target dimensions.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int],
        pad_value: int = 0,
        resize_mode: str = "shortest_edge",
    ) -> None:
        """
        Initialise transform.
        
        Args:
            target_size: Target (height, width).
            pad_value: Pixel value for padding.
            resize_mode: 'shortest_edge' or 'longest_edge'.
        """
        self.target_size = target_size
        self.pad_value = pad_value
        self.resize_mode = resize_mode
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply resize and pad transformation."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        target_h, target_w = self.target_size
        orig_w, orig_h = image.size
        
        # Calculate scale factor
        if self.resize_mode == "shortest_edge":
            scale = min(target_w / orig_w, target_h / orig_h)
        else:
            scale = max(target_w / orig_w, target_h / orig_h)
        
        # Resize
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Create padded image
        padded = Image.new("RGB", (target_w, target_h), (self.pad_value,) * 3)
        
        # Centre paste
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(image, (paste_x, paste_y))
        
        return padded


class RandomGaussianBlur:
    """Apply Gaussian blur with random probability."""
    
    def __init__(self, p: float = 0.1, radius_range: Tuple[float, float] = (0.5, 1.5)) -> None:
        """
        Initialise transform.
        
        Args:
            p: Probability of applying blur.
            radius_range: Range for blur radius.
        """
        self.p = p
        self.radius_range = radius_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random Gaussian blur."""
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image


class RandomHistogramEqualisation:
    """Apply histogram equalisation with random probability."""
    
    def __init__(self, p: float = 0.2) -> None:
        """
        Initialise transform.
        
        Args:
            p: Probability of applying equalisation.
        """
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random histogram equalisation."""
        if random.random() < self.p:
            image = ImageOps.equalize(image)
        return image


class GammaCorrection:
    """Apply gamma correction to adjust image brightness."""
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        """
        Initialise transform.
        
        Args:
            gamma_range: Range for gamma value.
        """
        self.gamma_range = gamma_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply gamma correction."""
        gamma = random.uniform(*self.gamma_range)
        
        # Convert to numpy for gamma correction
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)


class SimulateLowQuality:
    """Simulate low-quality imaging conditions for robustness training."""
    
    def __init__(
        self,
        p: float = 0.1,
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        jpeg_quality_range: Tuple[int, int] = (50, 90),
    ) -> None:
        """
        Initialise transform.
        
        Args:
            p: Probability of applying degradation.
            noise_std_range: Range for Gaussian noise standard deviation.
            jpeg_quality_range: Range for JPEG compression quality.
        """
        self.p = p
        self.noise_std_range = noise_std_range
        self.jpeg_quality_range = jpeg_quality_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random quality degradation."""
        if random.random() < self.p:
            # Add Gaussian noise
            img_array = np.array(image, dtype=np.float32) / 255.0
            noise_std = random.uniform(*self.noise_std_range)
            noise = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        return image


def denormalise_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Reverse normalisation for visualisation.
    
    Args:
        tensor: Normalised image tensor of shape (C, H, W) or (B, C, H, W).
        mean: Original normalisation mean.
        std: Original normalisation standard deviation.
    
    Returns:
        Denormalised tensor with values in [0, 1].
    """
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    
    if tensor.dim() == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    return tensor * std_tensor + mean_tensor


def tensor_to_pil(
    tensor: torch.Tensor,
    denormalise: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor of shape (C, H, W).
        denormalise: Whether to reverse normalisation.
        mean: Normalisation mean (if denormalising).
        std: Normalisation standard deviation (if denormalising).
    
    Returns:
        PIL Image.
    """
    if denormalise:
        tensor = denormalise_tensor(tensor, mean, std)
    
    tensor = tensor.clamp(0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(array)
