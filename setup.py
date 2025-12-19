"""Setup script for VERDICT package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="verdict",
    version="0.1.0",
    description="Attribution methods for medical vision-language models - VERDICT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anish",
    author_email="your.email@imperial.ac.uk",
    url="https://github.com/yourusername/verdict",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    python_requires=">=3.10",
    install_requires=[
        # Core deep learning
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        # Attribution and interpretability
        "captum>=0.6.0",
        # Data handling
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "pydicom>=2.4.0",
        "SimpleITK>=2.3.0",
        # Visualisation
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        # Utilities
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "wandb>=0.15.0",
        # ML utilities
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            # Code quality
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            # Jupyter
            "jupyter>=1.0.0",
            "ipython>=8.12.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.1.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords=[
        "medical imaging",
        "vision-language models",
        "attribution",
        "interpretability",
        "chest x-ray",
        "radiology",
    ],
)
