"""Image and report preprocessing utilities for VERDICT project.

This module provides preprocessing pipelines for chest X-ray images
and radiology reports, tailored for use with MAIRA-2 and other LVLMs.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocessing pipeline for chest X-ray images.
    
    Handles resizing, normalisation, and format conversion for
    medical imaging models, particularly MAIRA-2.
    
    Attributes:
        image_size: Target image size (height, width).
        normalise: Whether to apply normalisation.
        mean: Normalisation mean values.
        std: Normalisation standard deviation values.
    
    Example:
        >>> preprocessor = ImagePreprocessor(image_size=(518, 518))
        >>> processed_image = preprocessor(pil_image)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (518, 518),
        normalise: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        resize_mode: str = "shortest_edge",
        pad_to_square: bool = True,
        pad_value: int = 0,
    ) -> None:
        """
        Initialise image preprocessor.
        
        Args:
            image_size: Target (height, width) for output images.
            normalise: Whether to normalise pixel values.
            mean: Mean values for normalisation (per channel).
            std: Standard deviation for normalisation (per channel).
            resize_mode: Resize strategy - 'shortest_edge', 'longest_edge', or 'exact'.
            pad_to_square: Whether to pad image to square after resize.
            pad_value: Pixel value used for padding.
        """
        self.image_size = image_size
        self.normalise = normalise
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.resize_mode = resize_mode
        self.pad_to_square = pad_to_square
        self.pad_value = pad_value
    
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess an image.
        
        Args:
            image: Input image as PIL Image, NumPy array, or torch Tensor.
        
        Returns:
            Preprocessed image tensor of shape (3, H, W).
        """
        # Convert to PIL Image if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize
        image = self._resize(image)
        
        # Pad to square if requested
        if self.pad_to_square:
            image = self._pad_to_square(image)
        
        # Convert to tensor
        image_tensor = self._to_tensor(image)
        
        # Normalise
        if self.normalise:
            image_tensor = self._normalise(image_tensor)
        
        return image_tensor
    
    def _resize(self, image: Image.Image) -> Image.Image:
        """Resize image according to specified mode."""
        original_width, original_height = image.size
        target_height, target_width = self.image_size
        
        if self.resize_mode == "exact":
            return image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        
        elif self.resize_mode == "shortest_edge":
            # Scale so shortest edge matches target
            scale = min(target_width / original_width, target_height / original_height)
            
        elif self.resize_mode == "longest_edge":
            # Scale so longest edge matches target
            scale = max(target_width / original_width, target_height / original_height)
        
        else:
            raise ValueError(f"Unknown resize mode: {self.resize_mode}")
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    def _pad_to_square(self, image: Image.Image) -> Image.Image:
        """Pad image to square dimensions."""
        width, height = image.size
        target_height, target_width = self.image_size
        
        if width == target_width and height == target_height:
            return image
        
        # Create padded image
        padded = Image.new("RGB", (target_width, target_height), (self.pad_value,) * 3)
        
        # Centre the original image
        paste_x = (target_width - width) // 2
        paste_y = (target_height - height) // 2
        
        padded.paste(image, (paste_x, paste_y))
        
        return padded
    
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to torch Tensor."""
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor
    
    def _normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalisation to tensor."""
        return (tensor - self.mean) / self.std
    
    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation for visualisation."""
        return tensor * self.std + self.mean
    
    def to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image for visualisation."""
        if self.normalise:
            tensor = self.denormalise(tensor)
        
        # Clamp and convert
        tensor = tensor.clamp(0, 1)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        return Image.fromarray(array)


class ReportPreprocessor:
    """
    Preprocessing pipeline for radiology reports.
    
    Handles text cleaning, section extraction, and tokenisation
    preparation for radiology reports.
    
    Example:
        >>> preprocessor = ReportPreprocessor()
        >>> cleaned_report = preprocessor.clean_report(raw_report)
        >>> findings = preprocessor.extract_findings(raw_report)
    """
    
    # Standard section headers in radiology reports
    SECTION_HEADERS = [
        "FINDINGS",
        "IMPRESSION",
        "INDICATION",
        "TECHNIQUE",
        "COMPARISON",
        "HISTORY",
        "CLINICAL HISTORY",
        "CLINICAL INFORMATION",
    ]
    
    # Common abbreviations in radiology
    ABBREVIATIONS = {
        "w/": "with",
        "w/o": "without",
        "b/l": "bilateral",
        "r/o": "rule out",
        "h/o": "history of",
        "c/w": "consistent with",
        "s/p": "status post",
        "d/t": "due to",
    }
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        expand_abbreviations: bool = True,
        remove_headers: bool = False,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Initialise report preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation marks.
            expand_abbreviations: Whether to expand common abbreviations.
            remove_headers: Whether to remove section headers.
            max_length: Maximum character length (truncate if exceeded).
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.expand_abbreviations = expand_abbreviations
        self.remove_headers = remove_headers
        self.max_length = max_length
    
    def __call__(self, report: str) -> str:
        """
        Preprocess a radiology report.
        
        Args:
            report: Raw report text.
        
        Returns:
            Preprocessed report text.
        """
        return self.clean_report(report)
    
    def clean_report(self, report: str) -> str:
        """
        Clean and normalise report text.
        
        Args:
            report: Raw report text.
        
        Returns:
            Cleaned report text.
        """
        if not report:
            return ""
        
        # Remove extra whitespace
        report = " ".join(report.split())
        
        # Expand abbreviations
        if self.expand_abbreviations:
            report = self._expand_abbreviations(report)
        
        # Remove section headers if requested
        if self.remove_headers:
            report = self._remove_section_headers(report)
        
        # Lowercase if requested
        if self.lowercase:
            report = report.lower()
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            report = re.sub(r"[^\w\s]", "", report)
        
        # Truncate if max_length specified
        if self.max_length and len(report) > self.max_length:
            report = report[:self.max_length]
        
        return report.strip()
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common radiology abbreviations."""
        for abbrev, expansion in self.ABBREVIATIONS.items():
            text = text.replace(abbrev, expansion)
        return text
    
    def _remove_section_headers(self, text: str) -> str:
        """Remove section header labels from text."""
        for header in self.SECTION_HEADERS:
            text = re.sub(rf"{header}[:\s]*", "", text, flags=re.IGNORECASE)
        return text
    
    def extract_section(self, report: str, section: str) -> str:
        """
        Extract a specific section from the report.
        
        Args:
            report: Full report text.
            section: Section name to extract (e.g., 'FINDINGS', 'IMPRESSION').
        
        Returns:
            Extracted section text, or empty string if not found.
        """
        section = section.upper()
        
        # Find section start
        pattern = rf"{section}[:\s]*"
        match = re.search(pattern, report, re.IGNORECASE)
        
        if not match:
            return ""
        
        start_idx = match.end()
        
        # Find section end (next header or end of document)
        end_idx = len(report)
        for header in self.SECTION_HEADERS:
            if header != section:
                header_match = re.search(rf"\n{header}[:\s]*", report[start_idx:], re.IGNORECASE)
                if header_match:
                    end_idx = min(end_idx, start_idx + header_match.start())
        
        return report[start_idx:end_idx].strip()
    
    def extract_findings(self, report: str) -> str:
        """Extract the findings section from a report."""
        return self.extract_section(report, "FINDINGS")
    
    def extract_impression(self, report: str) -> str:
        """Extract the impression section from a report."""
        return self.extract_section(report, "IMPRESSION")
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        Args:
            text: Input text.
        
        Returns:
            List of sentences.
        """
        # Simple sentence splitting (medical text often has unusual patterns)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_findings_list(self, report: str) -> List[str]:
        """
        Extract individual findings as a list.
        
        Attempts to identify discrete findings in the report,
        handling both sentence-based and list-based formats.
        
        Args:
            report: Report text (preferably findings section).
        
        Returns:
            List of individual findings.
        """
        findings = []
        
        # Try numbered list format (1., 2., etc.)
        numbered = re.findall(r"\d+\.\s*(.+?)(?=\d+\.|$)", report, re.DOTALL)
        if numbered:
            findings.extend([f.strip() for f in numbered if f.strip()])
            return findings
        
        # Try bullet format
        bulleted = re.findall(r"[-•]\s*(.+?)(?=[-•]|$)", report, re.DOTALL)
        if bulleted:
            findings.extend([f.strip() for f in bulleted if f.strip()])
            return findings
        
        # Fall back to sentence splitting
        return self.extract_sentences(report)


def apply_windowing(
    image: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Apply windowing to a medical image (typically from DICOM).
    
    Windowing adjusts the display of pixel values to enhance
    visibility of specific structures.
    
    Args:
        image: Input image array.
        window_center: Centre of the window (level).
        window_width: Width of the window.
    
    Returns:
        Windowed image normalised to [0, 255].
    """
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    windowed = np.clip(image, min_value, max_value)
    windowed = (windowed - min_value) / (max_value - min_value) * 255
    
    return windowed.astype(np.uint8)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to enhance contrast.
    
    Args:
        image: Input image (grayscale or RGB).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalisation.
    
    Returns:
        Contrast-enhanced image.
    
    Raises:
        ImportError: If OpenCV is not installed.
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for contrast enhancement")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    # Convert back to RGB if input was RGB
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced
