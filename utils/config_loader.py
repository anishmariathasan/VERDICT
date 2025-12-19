"""Configuration loading utilities for VERDICT project.

This module provides functions for loading, merging, and validating
YAML configuration files used throughout the project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Configuration dictionary parsed from YAML.
    
    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    
    Example:
        >>> config = load_config("configs/base_config.yaml")
        >>> print(config["model"]["name"])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path to save the YAML configuration file.
    
    Example:
        >>> config = {"model": {"name": "maira-2"}}
        >>> save_config(config, "outputs/experiment_config.yaml")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Values in the override dictionary take precedence over base values.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
    
    Returns:
        Merged configuration dictionary.
    
    Example:
        >>> base = {"model": {"name": "maira-2", "size": "base"}}
        >>> override = {"model": {"size": "large"}}
        >>> merged = merge_configs(base, override)
        >>> print(merged)
        {'model': {'name': 'maira-2', 'size': 'large'}}
    """
    merged = base.copy()
    
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def load_config_with_overrides(
    base_config_path: Union[str, Path],
    override_config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load base configuration with optional overrides.
    
    Loads a base configuration file and optionally merges it with
    an override configuration file and/or command-line overrides.
    Priority (lowest to highest): base < override file < CLI overrides.
    
    Args:
        base_config_path: Path to base configuration file.
        override_config_path: Optional path to override configuration file.
        cli_overrides: Optional dictionary of CLI overrides.
    
    Returns:
        Merged configuration dictionary.
    
    Example:
        >>> config = load_config_with_overrides(
        ...     "configs/base_config.yaml",
        ...     "configs/maira2_config.yaml",
        ...     {"training.learning_rate": 1e-5}
        ... )
    """
    # Load base configuration
    config = load_config(base_config_path)
    
    # Merge override configuration file if provided
    if override_config_path is not None:
        override_config = load_config(override_config_path)
        config = merge_configs(config, override_config)
    
    # Apply CLI overrides if provided
    if cli_overrides is not None:
        config = merge_configs(config, _flatten_to_nested(cli_overrides))
    
    return config


def _flatten_to_nested(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat dot-notation keys to nested dictionary.
    
    Args:
        flat_dict: Dictionary with dot-notation keys.
    
    Returns:
        Nested dictionary.
    
    Example:
        >>> _flatten_to_nested({"model.name": "maira-2"})
        {'model': {'name': 'maira-2'}}
    """
    nested: Dict[str, Any] = {}
    
    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return nested


def resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in configuration values.
    
    Replaces ${VAR_NAME} patterns with corresponding environment
    variable values.
    
    Args:
        config: Configuration dictionary with potential env var references.
    
    Returns:
        Configuration with environment variables resolved.
    
    Example:
        >>> os.environ["DATA_DIR"] = "/data/mimic"
        >>> config = {"data": {"path": "${DATA_DIR}/images"}}
        >>> resolved = resolve_env_vars(config)
        >>> print(resolved["data"]["path"])
        '/data/mimic/images'
    """
    resolved = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_env_vars(value)
        elif isinstance(value, str):
            # Replace ${VAR_NAME} patterns
            resolved[key] = _resolve_env_string(value)
        else:
            resolved[key] = value
    
    return resolved


def _resolve_env_string(value: str) -> str:
    """Resolve environment variables in a string value."""
    import re
    
    pattern = r"\$\{([^}]+)\}"
    
    def replace_env(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    
    return re.sub(pattern, replace_env, value)
