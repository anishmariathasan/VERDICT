"""VERDICT: Vision & language Error Reasoning, Diagnosis, and Classification with Technical improvements.

A research project investigating attribution methods for Large Vision-Language Models
(LVLMs) in medical imaging, specifically for chest X-ray report generation.
"""

__version__ = "0.1.0"
__author__ = "Anish"
__email__ = "your.email@imperial.ac.uk"

# Lazy imports to avoid circular dependencies
# Users should import directly from submodules:
#   from verdict.utils import set_seed, setup_logger, load_config
#   from verdict.models import MAIRA2Model
#   from verdict.data import MIMICCXRDataset

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
