"""
Protocol Optuna Compatibility Tests

Tests that all protocol experiments in experiments/training/tests/protocol/
can successfully run Optuna hyperparameter optimization.

This ensures that:
1. All model sampling functions work correctly
2. The Optuna workflow (create/resume/summary) executes without errors
3. Code changes don't break Optuna integration


Examples

One experiment, e.g. test_forecaster_proT_adaptive_random
pytest test/integration/test_optuna_protocol.py::test_optuna_workflow_compatibility[test_forecaster_proT_adaptive_random] -v -s


"""

import pytest
import os
import tempfile
import shutil
from omegaconf import OmegaConf
import yaml
import platform
import gc
import time

from proT.euler_optuna.optuna_opt import OptunaStudy

# Import shared test utilities
from .utils import (
    discover_protocol_configs,
    modify_config_for_fast_testing,
    DATA_DIR
)


def create_optuna_settings_for_testing():
    """
    Create minimal Optuna settings for fast testing.
    
    Returns:
        dict: Optuna settings with minimal trials for testing
    """
    return {
        "n_trials": 3,  # Fast testing - just verify it works
        "direction": "minimize",
        "sampler": {
            "name": "sobol"
        },
        "pruner": "none"
    }


def get_sampling_function_for_model(model_object, config):
    """
    Get the appropriate sampling function for a given model type.
    Uses the dispatcher from CLI module to handle all model types correctly.
    
    Args:
        model_object: Model type string (e.g., "proT", "MLP", "LSTM")
        config: OmegaConf configuration object (needed for proT variants)
        
    Returns:
        callable: Sampling function for the model that takes a trial object
    """
    from proT.euler_optuna.cli import sample_params_for_optuna
    
    # Return a lambda that calls the dispatcher with both trial and config
    return lambda trial: sample_params_for_optuna(trial, config)


def get_training_and_metrics_functions():
    """
    Get training and metrics extraction functions from CLI module.
    
    Returns:
        tuple: (train_function, get_metrics_function)
    """
    from proT.euler_optuna.cli import (
        train_function_for_optuna,
        get_metrics_for_optuna
    )
    
    return train_function_for_optuna, get_metrics_for_optuna


@pytest.fixture
def temp_output_dir(request):
    """
    Create temporary directory for test outputs, cleanup after test.
    Uses test name to ensure uniqueness across parallel tests.
    Implements robust Windows-compatible cleanup.
    
    Args:
        request: pytest fixture providing test context
        
    Yields:
        str: Path to temporary directory
    """
    # Get test name for unique directory
    test_name = request.node.name
    # Create unique temp directory based on test name
    temp_dir = tempfile.mkdtemp(prefix=f"optuna_test_{test_name}_")
    yield temp_dir
    
    # Cleanup after test - platform-specific handling
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        # Windows needs more aggressive cleanup due to file locking
        time.sleep(1)  # Initial wait for file handles to be released
        
        # Try cleanup with retries
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                break  # Success!
            except PermissionError as e:
                if attempt < max_retries - 1:
                    # Wait progressively longer
                    wait_time = (attempt + 1) * 2
                    print(f"\nWindows file lock detected, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed - log but don't fail the test
                    print(f"\nWarning: Could not clean up temp directory {temp_dir}")
                    print(f"This is a Windows file locking issue and won't affect test results.")
                    print(f"Directory will be cleaned up on next system restart.")
    else:
        # Unix systems - normal cleanup
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"\nWarning: Could not clean up temp directory {temp_dir}: {e}")


@pytest.mark.parametrize("experiment_name,config_path", discover_protocol_configs())
def test_optuna_workflow_compatibility(experiment_name, config_path, temp_output_dir):
    """
    Test that Optuna workflow works for each protocol experiment.
    
    This test ensures code changes don't break Optuna integration by:
    1. Loading the experiment config
    2. Creating optuna_settings.yaml for testing
    3. Running Optuna workflow: create → resume (3 trials) → summary
    4. Validating results
    
    Args:
        experiment_name: Name of the experiment being tested
        config_path: Path to the experiment config file
        temp_output_dir: Temporary directory for test outputs
    """
    # 1. Load and modify config
    config = OmegaConf.load(config_path)
    config_dev = modify_config_for_fast_testing(config, max_epochs=2, k_fold=2)
    
    # Save modified config to temp directory
    temp_config_path = os.path.join(temp_output_dir, "config.yaml")
    OmegaConf.save(config_dev, temp_config_path)
    
    # 2. Create optuna_settings.yaml
    optuna_settings = create_optuna_settings_for_testing()
    optuna_settings_path = os.path.join(temp_output_dir, "optuna_settings.yaml")
    with open(optuna_settings_path, 'w') as f:
        yaml.dump(optuna_settings, f)
    
    # 3. Get model-specific sampling function
    model_object = config_dev["model"]["model_object"]
    sample_params_fn = get_sampling_function_for_model(model_object, config_dev)
    
    # 4. Get training and metrics functions
    train_fn, get_metrics_fn = get_training_and_metrics_functions()
    
    # 5. Create OptunaStudy
    study_name = f"test_study_{experiment_name}"
    
    try:
        optuna_study = OptunaStudy(
            exp_dir=temp_output_dir,
            data_dir=str(DATA_DIR),
            cluster=False,
            study_name=study_name,
            manifest_tag=f"optuna_test_{experiment_name}",
            study_path=temp_output_dir,
            sample_params_fn=sample_params_fn,
            train_fn=train_fn,
            get_metrics_fn=get_metrics_fn,
            optimization_metric="val_mae_mean",
            optimization_direction="minimize"
        )
        
        # 6. Create study
        optuna_study.create()
        
        # Verify study database was created
        study_db_path = os.path.join(temp_output_dir, "study.db")
        assert os.path.exists(study_db_path), "Study database should be created"
        
        # 7. Run optimization (3 trials)
        optuna_study.resume()
        
        # 8. Generate summary
        optuna_study.summary()
        
        # 9. Explicitly cleanup OptunaStudy object to close database connections
        # This is critical for Windows to release file locks
        del optuna_study
        gc.collect()  # Force garbage collection
        time.sleep(0.5)  # Give OS time to release file handles
        
        # 10. Validate results
        best_trial_path = os.path.join(temp_output_dir, "best_trial.yaml")
        assert os.path.exists(best_trial_path), "best_trial.yaml should be created"
        
        # Load and validate best_trial.yaml
        with open(best_trial_path, 'r') as f:
            best_trial = yaml.safe_load(f)
        
        # Check required fields
        assert "trial_number" in best_trial, "best_trial should contain trial_number"
        assert "optimization_metric" in best_trial, "best_trial should contain optimization_metric"
        assert "optimization_value" in best_trial, "best_trial should contain optimization_value"
        assert "params" in best_trial, "best_trial should contain params"
        assert "metrics" in best_trial, "best_trial should contain metrics"
        
        # Check that metrics contain expected keys
        metrics = best_trial["metrics"]
        expected_metrics = ["val_mae_mean", "val_mae_std", "test_mae_mean", "test_mae_std"]
        for metric in expected_metrics:
            assert metric in metrics, f"Metrics should contain {metric}"
        
        # Check that params is not empty
        assert len(best_trial["params"]) > 0, "params should not be empty"
        
        # Check that optimization_value is a valid number
        opt_value = best_trial["optimization_value"]
        assert isinstance(opt_value, (int, float)), "optimization_value should be a number"
        
    except Exception as e:
        pytest.fail(f"Optuna workflow for '{experiment_name}' failed: {str(e)}")
    finally:
        # Ensure cleanup even if test fails
        try:
            del optuna_study
        except:
            pass
        gc.collect()


if __name__ == "__main__":
    """
    Allow running this test file directly for quick debugging.
    Usage: python test/test_optuna_compatibility.py
    """
    # Discover and print available experiments
    configs = discover_protocol_configs()
    print(f"\nFound {len(configs)} protocol experiments for Optuna testing:")
    for param in configs:
        print(f"  - {param.id}")
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
