"""
Protocol Experiment Compatibility Tests

Tests that all protocol experiments in experiments/training/tests/protocol/
can successfully compile and run at least one training batch.

This ensures that code changes don't break the compatibility with existing
experiment templates.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

from proT.training.trainer import trainer
from proT.training.experiment_control import update_config

# Constants
ROOT_DIR = Path(__file__).parent.parent
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
        pytest.skip(f"Protocol directory not found: {PROTOCOL_DIR}")
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
        pytest.skip("No protocol experiment configs found")
    
    return configs


def modify_config_for_dev_mode(config):
    """
    Override config parameters for fast dev mode testing.
    
    Args:
        config: OmegaConf config object
        
    Returns:
        OmegaConf: Modified config for dev mode
    """
    # Make a copy to avoid modifying original
    config_dev = config.copy()
    
    # Override training parameters for speed
    config_dev["training"]["max_epochs"] = 1
    config_dev["training"]["save_ckpt_every_n_epochs"] = 999  # Disable checkpointing
    
    # Limit data size for fast execution
    config_dev["data"]["max_data_size"] = 10
    
    # Disable test_ds_idx to avoid index out of bounds with limited dataset
    # The trainer will use random splits instead
    config_dev["data"]["test_ds_ixd"] = None
    
    return config_dev


@pytest.fixture
def temp_output_dir():
    """
    Create temporary directory for test outputs, cleanup after test.
    
    Yields:
        str: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="protocol_test_")
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.mark.parametrize("experiment_name,config_path", discover_protocol_configs())
def test_protocol_experiment_compatibility(experiment_name, config_path, temp_output_dir):
    """
    Test that protocol experiment can compile and run one training batch.
    
    This test ensures code changes don't break experiment compatibility by:
    1. Loading the experiment config
    2. Running one epoch with minimal data
    3. Verifying the model compiles and runs without errors
    
    Args:
        experiment_name: Name of the experiment being tested
        config_path: Path to the experiment config file
        temp_output_dir: Temporary directory for test outputs
    """
    # 1. Load config
    config = OmegaConf.load(config_path)
    
    # 2. Modify for dev mode (fast execution)
    config_dev = modify_config_for_dev_mode(config)
    
    # 3. Update config (handle placeholders, etc.)
    config_updated = update_config(config_dev)
    
    # 4. Run trainer
    try:
        result_df = trainer(
            config=config_updated,
            data_dir=str(DATA_DIR),
            save_dir=temp_output_dir,
            cluster=False,
            experiment_tag=f"protocol_test_{experiment_name}",
            resume_ckpt=None,
            plot_pred_check=False,
            debug=False,
            best=False,
        )
        
        # 5. Validate results
        assert result_df is not None, "Trainer should return a DataFrame"
        assert isinstance(result_df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert len(result_df) > 0, "Result DataFrame should not be empty"
        
        # Check that basic metrics exist
        assert "val_loss" in result_df.columns or "val_mae" in result_df.columns, \
            "Result should contain validation metrics"
        
    except Exception as e:
        pytest.fail(f"Experiment '{experiment_name}' failed to run: {str(e)}")


# Optional: Mark tests as slow for CI filtering
pytestmark = pytest.mark.integration


if __name__ == "__main__":
    """
    Allow running this test file directly for quick debugging.
    Usage: python test/test_protocol_compatibility.py
    """
    # Discover and print available experiments
    configs = discover_protocol_configs()
    print(f"\nFound {len(configs)} protocol experiments:")
    for name, path in configs:
        print(f"  - {name}")
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
