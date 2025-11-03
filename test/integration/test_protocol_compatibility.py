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
from omegaconf import OmegaConf
import pandas as pd

from proT.training.trainer import trainer
from proT.training.experiment_control import update_config

# Import shared test utilities
from .utils import (
    discover_protocol_configs,
    modify_config_for_fast_testing,
    DATA_DIR
)


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
    config_dev = modify_config_for_fast_testing(config, max_epochs=1)
    
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
# pytestmark = pytest.mark.integration


if __name__ == "__main__":
    """
    Allow running this test file directly for quick debugging.
    Usage: python test/test_protocol_compatibility.py
    """
    # Discover and print available experiments
    configs = discover_protocol_configs()
    print(f"\nFound {len(configs)} protocol experiments:")
    for param in configs:
        print(f"  - {param.id}")
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
