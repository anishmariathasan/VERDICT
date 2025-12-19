"""MIMIC-CXR dataset loader for VERDICT project.

This module provides data loading utilities for the MIMIC-CXR dataset,
including image loading, report parsing, and split management.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    import pydicom
except ImportError:
    pydicom = None

logger = logging.getLogger(__name__)


class MIMICCXRDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-CXR chest X-ray images and reports.
    
    Supports loading from both DICOM and JPEG formats. Handles the
    standard MIMIC-CXR split files and provides flexible filtering.
    
    Attributes:
        data_dir: Root directory of MIMIC-CXR dataset.
        reports_dir: Directory containing radiology reports.
        split: Dataset split ('train', 'validate', 'test').
        transform: Optional image transformations.
    
    Example:
        >>> dataset = MIMICCXRDataset(
        ...     data_dir="/path/to/mimic-cxr-jpg",
        ...     reports_dir="/path/to/mimic-cxr-reports",
        ...     split="train",
        ...     transform=get_train_transforms()
        ... )
        >>> sample = dataset[0]
        >>> print(sample["image"].shape, sample["report"][:50])
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        reports_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_format: str = "jpg",
        max_samples: Optional[int] = None,
        include_labels: bool = True,
        section: str = "findings",
    ) -> None:
        """
        Initialise MIMIC-CXR dataset.
        
        Args:
            data_dir: Path to MIMIC-CXR image directory.
            reports_dir: Path to radiology reports. If None, uses data_dir.
            split: Dataset split - 'train', 'validate', or 'test'.
            transform: Optional callable for image transformations.
            image_format: Image format - 'jpg' or 'dcm' (DICOM).
            max_samples: Maximum number of samples to load (for debugging).
            include_labels: Whether to load CheXpert labels.
            section: Report section to use - 'findings', 'impression', or 'full'.
        
        Raises:
            ValueError: If split is not valid or data directory not found.
        """
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir) if reports_dir else self.data_dir
        self.split = split
        self.transform = transform
        self.image_format = image_format.lower()
        self.max_samples = max_samples
        self.include_labels = include_labels
        self.section = section
        
        # Validate inputs
        if split not in ["train", "validate", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validate', or 'test'.")
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.samples = self._prepare_samples()
        
        logger.info(f"Loaded MIMIC-CXR {split} split with {len(self.samples)} samples")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load MIMIC-CXR metadata and split information."""
        # Try to load split file
        split_file = self.data_dir / "mimic-cxr-2.0.0-split.csv"
        
        if split_file.exists():
            splits_df = pd.read_csv(split_file)
        else:
            # Create placeholder if split file not found
            logger.warning(f"Split file not found: {split_file}. Using all available images.")
            splits_df = None
        
        # Load study metadata
        metadata_file = self.data_dir / "mimic-cxr-2.0.0-metadata.csv"
        
        if metadata_file.exists():
            metadata_df = pd.read_csv(metadata_file)
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
            metadata_df = pd.DataFrame()
        
        # Merge with splits
        if splits_df is not None and not metadata_df.empty:
            metadata_df = metadata_df.merge(splits_df, on=["subject_id", "study_id", "dicom_id"])
        
        return metadata_df
    
    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """Prepare list of samples for the specified split."""
        samples = []
        
        if self.metadata.empty:
            # Fallback: scan directory for images
            logger.warning("No metadata found. Scanning directory for images.")
            return self._scan_directory_for_samples()
        
        # Filter by split
        if "split" in self.metadata.columns:
            split_df = self.metadata[self.metadata["split"] == self.split]
        else:
            split_df = self.metadata
        
        for _, row in split_df.iterrows():
            sample = {
                "subject_id": row.get("subject_id"),
                "study_id": row.get("study_id"),
                "dicom_id": row.get("dicom_id"),
                "image_path": self._get_image_path(row),
            }
            
            if sample["image_path"].exists():
                samples.append(sample)
        
        # Apply max samples limit
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        return samples
    
    def _scan_directory_for_samples(self) -> List[Dict[str, Any]]:
        """Scan directory for images when metadata is not available."""
        samples = []
        pattern = f"**/*.{self.image_format}"
        
        for image_path in self.data_dir.glob(pattern):
            samples.append({
                "subject_id": None,
                "study_id": None,
                "dicom_id": image_path.stem,
                "image_path": image_path,
            })
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        return samples
    
    def _get_image_path(self, row: pd.Series) -> Path:
        """Construct image path from metadata row."""
        subject_id = row["subject_id"]
        study_id = row["study_id"]
        dicom_id = row["dicom_id"]
        
        # MIMIC-CXR-JPG structure: files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        subject_prefix = f"p{str(subject_id)[:2]}"
        
        image_path = (
            self.data_dir
            / "files"
            / subject_prefix
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.{self.image_format}"
        )
        
        return image_path
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load image from file."""
        if self.image_format == "dcm":
            if pydicom is None:
                raise ImportError("pydicom is required for DICOM loading")
            
            dcm = pydicom.dcmread(str(image_path))
            image_array = dcm.pixel_array
            
            # Normalise to 8-bit
            image_array = (
                (image_array - image_array.min())
                / (image_array.max() - image_array.min())
                * 255
            ).astype(np.uint8)
            
            image = Image.fromarray(image_array)
        else:
            image = Image.open(image_path)
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def _load_report(self, subject_id: int, study_id: int) -> str:
        """Load radiology report for a study."""
        if subject_id is None or study_id is None:
            return ""
        
        # Report file structure: files/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt
        subject_prefix = f"p{str(subject_id)[:2]}"
        
        report_path = (
            self.reports_dir
            / "files"
            / subject_prefix
            / f"p{subject_id}"
            / f"s{study_id}.txt"
        )
        
        if not report_path.exists():
            logger.debug(f"Report not found: {report_path}")
            return ""
        
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        
        # Extract specific section if requested
        report_text = self._extract_section(report_text, self.section)
        
        return report_text
    
    def _extract_section(self, report: str, section: str) -> str:
        """Extract specific section from report."""
        if section == "full":
            return report
        
        section_markers = {
            "findings": ["FINDINGS:", "FINDINGS :", "Findings:"],
            "impression": ["IMPRESSION:", "IMPRESSION :", "Impression:"],
        }
        
        markers = section_markers.get(section, [])
        
        for marker in markers:
            if marker in report:
                start_idx = report.find(marker) + len(marker)
                # Find end of section (next section marker or end)
                end_idx = len(report)
                
                for other_section, other_markers in section_markers.items():
                    if other_section != section:
                        for other_marker in other_markers:
                            if other_marker in report[start_idx:]:
                                potential_end = report.find(other_marker, start_idx)
                                if potential_end < end_idx:
                                    end_idx = potential_end
                
                return report[start_idx:end_idx].strip()
        
        return report
    
    def _load_labels(self, subject_id: int, study_id: int) -> Dict[str, int]:
        """Load CheXpert labels for a study."""
        # Placeholder - implement based on available label files
        return {}
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dictionary containing:
                - image: Image tensor or PIL Image
                - report: Report text
                - subject_id: Subject identifier
                - study_id: Study identifier
                - labels: CheXpert labels (if include_labels=True)
        """
        sample_info = self.samples[idx]
        
        # Load image
        image = self._load_image(sample_info["image_path"])
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Load report
        report = self._load_report(
            sample_info["subject_id"],
            sample_info["study_id"]
        )
        
        # Prepare output
        output = {
            "image": image,
            "report": report,
            "subject_id": sample_info["subject_id"],
            "study_id": sample_info["study_id"],
            "dicom_id": sample_info["dicom_id"],
            "image_path": str(sample_info["image_path"]),
        }
        
        # Load labels if requested
        if self.include_labels:
            output["labels"] = self._load_labels(
                sample_info["subject_id"],
                sample_info["study_id"]
            )
        
        return output


class MIMICCXRDataModule:
    """
    Data module for managing MIMIC-CXR dataset splits and loaders.
    
    Provides a unified interface for creating train, validation, and
    test data loaders with consistent preprocessing.
    
    Example:
        >>> datamodule = MIMICCXRDataModule(
        ...     data_dir="/path/to/mimic-cxr-jpg",
        ...     batch_size=8
        ... )
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        reports_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Initialise data module.
        
        Args:
            data_dir: Path to MIMIC-CXR image directory.
            reports_dir: Path to radiology reports.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            train_transform: Transforms for training data.
            val_transform: Transforms for validation data.
            test_transform: Transforms for test data.
            max_samples: Maximum samples per split (for debugging).
        """
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir) if reports_dir else self.data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.max_samples = max_samples
        
        self.train_dataset: Optional[MIMICCXRDataset] = None
        self.val_dataset: Optional[MIMICCXRDataset] = None
        self.test_dataset: Optional[MIMICCXRDataset] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for each stage.
        
        Args:
            stage: Optional stage ('fit', 'validate', 'test', or None for all).
        """
        if stage in (None, "fit"):
            self.train_dataset = MIMICCXRDataset(
                data_dir=self.data_dir,
                reports_dir=self.reports_dir,
                split="train",
                transform=self.train_transform,
                max_samples=self.max_samples,
            )
            
            self.val_dataset = MIMICCXRDataset(
                data_dir=self.data_dir,
                reports_dir=self.reports_dir,
                split="validate",
                transform=self.val_transform,
                max_samples=self.max_samples,
            )
        
        if stage in (None, "validate"):
            if self.val_dataset is None:
                self.val_dataset = MIMICCXRDataset(
                    data_dir=self.data_dir,
                    reports_dir=self.reports_dir,
                    split="validate",
                    transform=self.val_transform,
                    max_samples=self.max_samples,
                )
        
        if stage in (None, "test"):
            self.test_dataset = MIMICCXRDataset(
                data_dir=self.data_dir,
                reports_dir=self.reports_dir,
                split="test",
                transform=self.test_transform,
                max_samples=self.max_samples,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before creating dataloaders")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before creating dataloaders")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before creating dataloaders")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for variable-length reports."""
        images = torch.stack([sample["image"] for sample in batch])
        reports = [sample["report"] for sample in batch]
        
        collated = {
            "images": images,
            "reports": reports,
            "subject_ids": [sample["subject_id"] for sample in batch],
            "study_ids": [sample["study_id"] for sample in batch],
            "dicom_ids": [sample["dicom_id"] for sample in batch],
        }
        
        # Include labels if present
        if "labels" in batch[0]:
            collated["labels"] = [sample["labels"] for sample in batch]
        
        return collated
