"""Configuration loading utilities."""

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str, overrides: dict | None = None) -> dict:
    """Load a YAML config file and optionally merge overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional dict of overrides to merge on top.

    Returns:
        Merged configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dir = config_path.parent.resolve()
    project_root = config_dir.parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        config = _deep_merge(config, overrides)

    # Resolve relative paths in dataset.root relative to project root
    if "dataset" in config and "root" in config["dataset"]:
        dataset_root = config["dataset"]["root"]
        if not Path(dataset_root).is_absolute():
            config["dataset"]["root"] = str(project_root / dataset_root)

    # Resolve output.dir relative to project root
    if "output" in config and "dir" in config["output"]:
        output_dir = config["output"]["dir"]
        if not Path(output_dir).is_absolute():
            config["output"]["dir"] = str(project_root / output_dir)

    # Resolve device
    if config.get("device") == "auto":
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config
