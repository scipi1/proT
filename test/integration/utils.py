"""
Shared utilities for proT tests.

This module contains common functions used across multiple test files,
particularly for protocol experiment discovery and configuration.
"""

import pytest
from pathlib import Path

# Constants
ROOT_DIR = Path(__file__).parent.parent.parent  # Go up 3 levels to project root
PROTOCOL_DIR = ROOT_DIR / "experiments" / "training" / "tests" / "protocol"
DATA_DIR = ROOT_DIR / "data" / "input"


def discover_protocol_configs():
    """
    Auto-discover all config files in protocol experiment folders.
    
    Returns:
        list: List of pytest.param objects with custom IDs for clean display
    """
    configs = []
    
    if not PROTOCOL_DIR.exists():
        pytest.skip(f"Protocol directory not found: {PROTOCOL_DIR}", allow_module_level=True)
        return configs
    
    # Walk through protocol directory
    for experiment_dir in PROTOCOL_DIR.iterdir():
        if experiment_dir.is_dir():
            # Look for config files in each experiment directory
            for config_file in experiment_dir.glob("config*.yaml"):
                # Use only the experiment folder name for cleaner test output
                experiment_name = experiment_dir.name
                # Use pytest.param with id to control display name
                configs.append(pytest.param(experiment_name, str(config_file), id=experiment_name))
    
    if not configs:
        pytest.skip("No protocol experiment configs found", allow_module_level=True)
    
    return configs


def modify_config_for_fast_testing(config, max_epochs=1, k_fold=None, max_data_size=10):
    """
    Override config parameters for fast testing.
    
    Args:
        config: OmegaConf config object
        max_epochs: Maximum number of epochs to train (default: 1)
        k_fold: Number of k-folds (default: None, keeps original)
        max_data_size: Maximum data size (default: 10)
        
    Returns:
        OmegaConf: Modified config for fast testing
    """
    # Make a copy to avoid modifying original
    config_dev = config.copy()
    
    # Override training parameters for speed
    config_dev["training"]["max_epochs"] = max_epochs
    config_dev["training"]["save_ckpt_every_n_epochs"] = 999  # Disable checkpointing
    
    # Override k_fold if specified
    if k_fold is not None:
        config_dev["training"]["k_fold"] = k_fold
    
    # Limit data size for fast execution
    config_dev["data"]["max_data_size"] = max_data_size
    
    # Disable test_ds_idx to avoid index out of bounds with limited dataset
    # The trainer will use random splits instead
    config_dev["data"]["test_ds_ixd"] = None
    
    return config_dev
