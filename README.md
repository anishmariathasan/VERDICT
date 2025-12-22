# VERDICT: Vision & language Error Reasoning, Diagnosis, and Classification with Technical improvements

This repository contains the codebase for my Master's thesis project "Applying attribution methods to Large Vision-Language Models (LVLMs) for chest X-ray report generation".

The project focuses on adapting the CoIBA (Contextual Interpretability for Biological Applications) feature attribution method to LVLMs like MAIRA-2 to identify errors in X-Ray report generation, then use methods appropriate for the error type to improve results and compare it to our own method.

## Project Structure

```
VERDICT/
├── attribution/        # Attribution methods (CoIBA, Layer-wise, Token-level)
├── baselines/          # Baseline hallucination mitigation methods (UAC, VEP)
├── configs/            # Configuration files (YAML)
├── data/               # Data loading and preprocessing (MIMIC-CXR)
├── evaluation/         # Evaluation metrics (CheXpert, POPE, etc.)
├── experiments/        # Experiment scripts
├── models/             # Model wrappers (MAIRA-2)
├── notebooks/          # Jupyter notebooks for analysis
├── scripts/            # Utility scripts (setup, download)
├── tests/              # Unit tests
├── utils/              # Core utilities (logging, config, seed)
├── pyproject.toml      # Project metadata and dependencies
├── requirements.txt    # Python dependencies
└── setup.py            # Package setup script
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd VERDICT
    ```

2.  **Set up the environment:**
    ```bash
    ./scripts/setup_environment.sh
    ```
    Or manually:
    ```bash
    conda create -n verdict python=3.10 -y
    conda activate verdict
    pip install -e ".[dev]"
    ```

3.  **Download Data:**
    Follow the instructions in `scripts/download_data.sh` to download the MIMIC-CXR dataset from PhysioNet.

## Usage

### Configuration
The project uses a hierarchical configuration system. The base configuration is in `configs/base_config.yaml`. You can override settings for specific experiments using other config files (e.g., `configs/maira2_config.yaml`).

### Running Experiments

To generate attributions:
```bash
python experiments/generate_attributions.py --config configs/maira2_config.yaml
```

To run the full evaluation pipeline:
```bash
python experiments/run_evaluation.py --config configs/maira2_config.yaml
```

### Notebooks
- `notebooks/01_data_exploration.ipynb`: Explore the MIMIC-CXR dataset.
- `notebooks/02_model_inference.ipynb`: Run inference with MAIRA-2.
- `notebooks/03_attribution_analysis.ipynb`: Visualise attribution maps.

## Attribution Methods
This project implements:
- **CoIBA Adapter**: Adapts CoIBA for LVLMs.
- **Layer-wise Attribution**: Integrated Gradients and GradCAM for intermediate layers.
- **Token-level Attribution**: Attention-based and gradient-based token importance.