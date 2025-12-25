"""
Protocol Sweep Compatibility Tests

Tests that the euler_sweep framework works correctly with protocol experiments.
Unlike Optuna which has model-specific sampling, the sweeper is model-agnostic
and works by applying parameter combinations to the base config.

This ensures that:
1. Independent sweep mode works correctly (vary one param at a time)
2. Combination sweep mode works correctly (all combinations)
3. The sweep workflow executes without errors
4. Code changes don't break sweep integration
"""

import pytest
import os
import tempfile
import shutil
import yaml
import platform
import gc
import time
from pathlib import Path
from omegaconf import OmegaConf
from click.testing import CliRunner

from proT.euler_sweep.cli import cli

# Import shared test utilities
from .utils import (
    discover_protocol_configs,
    modify_config_for_fast_testing,
    DATA_DIR
)


# Default protocol for testing (can be overridden via pytest args)
DEFAULT_PROTOCOL = "test_baseline_proT_ishigami_cat"


@pytest.fixture
def protocol_config(request):
    """
    Load protocol config for testing.
    Can be overridden via --protocol command line argument.
    
    Args:
        request: pytest fixture providing command line args
        
    Returns:
        tuple: (protocol_name, config_path)
    """
    protocol_name = request.config.getoption("--protocol")
    
    # Find the config file for this protocol
    protocol_dir = Path(__file__).parent.parent.parent / "experiments" / "training" / "tests" / "protocol"
    experiment_dir = protocol_dir / protocol_name
    
    if not experiment_dir.exists():
        pytest.skip(f"Protocol experiment not found: {protocol_name}")
    
    # Find config file in experiment directory
    config_files = list(experiment_dir.glob("config*.yaml"))
    if not config_files:
        pytest.skip(f"No config file found in {experiment_dir}")
    
    config_path = config_files[0]
    return protocol_name, str(config_path)


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
    temp_dir = tempfile.mkdtemp(prefix=f"sweep_test_{test_name}_")
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


def create_sweep_config_for_testing(max_values_per_param=2):
    """
    Create minimal sweep configuration for fast testing.
    
    For nested parameters like model.kwargs.n_heads, we need to use
    dot notation in the parameter name: "kwargs.n_heads"
    
    Args:
        max_values_per_param: Maximum number of values per parameter (default: 2)
        
    Returns:
        dict: Sweep configuration with minimal parameter combinations
    """
    return {
        "training": {
            "lr": [0.001, 0.01][:max_values_per_param],
            "batch_size": [32, 64][:max_values_per_param]
        }
    }


def count_parameters_in_sweep(sweep_config):
    """
    Count total number of parameters being swept.
    
    Args:
        sweep_config: Sweep configuration dictionary
        
    Returns:
        int: Total number of parameters
    """
    count = 0
    for category in sweep_config:
        count += len(sweep_config[category])
    return count


def calculate_expected_runs(sweep_config, sweep_mode):
    """
    Calculate expected number of runs for a sweep configuration.
    
    Args:
        sweep_config: Sweep configuration dictionary
        sweep_mode: "independent" or "combination"
        
    Returns:
        int: Expected number of runs
    """
    if sweep_mode == "independent":
        # Independent: sum of all parameter values (one at a time)
        total = 0
        for category in sweep_config:
            for param_name in sweep_config[category]:
                total += len(sweep_config[category][param_name])
        return total
    
    elif sweep_mode == "combination":
        # Combination: product of all parameter values
        total = 1
        for category in sweep_config:
            for param_name in sweep_config[category]:
                total *= len(sweep_config[category][param_name])
        return total
    
    else:
        raise ValueError(f"Unknown sweep_mode: {sweep_mode}")


def validate_sweep_results(sweep_dir, sweep_mode, expected_runs):
    """
    Validate that sweep produced expected directory structure and results.
    
    Args:
        sweep_dir: Directory containing sweep results
        sweep_mode: "independent" or "combination"
        expected_runs: Expected number of run directories
        
    Raises:
        AssertionError: If validation fails
    """
    # Determine base directory based on mode
    if sweep_mode == "independent":
        base_dir = os.path.join(sweep_dir, "sweeper", "runs", "sweeps")
    else:
        base_dir = os.path.join(sweep_dir, "sweeper", "runs", "combinations")
    
    assert os.path.exists(base_dir), f"Sweep results directory not found: {base_dir}"
    
    # Count run directories
    if sweep_mode == "independent":
        # For independent, runs are nested: sweeps/sweep_paramname/sweep_paramname_value/
        run_dirs = []
        for param_dir in os.listdir(base_dir):
            param_path = os.path.join(base_dir, param_dir)
            if os.path.isdir(param_path):
                for run_dir in os.listdir(param_path):
                    run_path = os.path.join(param_path, run_dir)
                    if os.path.isdir(run_path):
                        run_dirs.append(run_path)
    else:
        # For combination, runs are direct children: combinations/combo_...
        run_dirs = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    
    actual_runs = len(run_dirs)
    assert actual_runs == expected_runs, \
        f"Expected {expected_runs} runs, found {actual_runs}"
    
    # Validate each run has a config file
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "config.yaml")
        assert os.path.exists(config_path), \
            f"Config file not found in {run_dir}"
        
        # Load and validate config can be parsed
        config = OmegaConf.load(config_path)
        assert config is not None, f"Failed to load config from {config_path}"
    
    print(f"\n✓ Validated {actual_runs} sweep runs in {sweep_mode} mode")


@pytest.mark.parametrize("sweep_mode", ["independent", "combination"])
def test_sweep_workflow_compatibility(sweep_mode, temp_output_dir, protocol_config):
    """
    Test that sweep workflow works for a protocol experiment via CLI.
    
    This test ensures code changes don't break sweep integration by:
    1. Loading a protocol experiment config
    2. Creating minimal sweep.yaml for testing
    3. Running sweep workflow via CLI (independent or combination mode)
    4. Validating results
    
    Args:
        sweep_mode: "independent" or "combination"
        temp_output_dir: Temporary directory for test outputs
        protocol_config: Protocol configuration fixture
    """
    protocol_name, config_path = protocol_config
    
    print(f"\n{'='*60}")
    print(f"Testing {sweep_mode} sweep with protocol: {protocol_name}")
    print(f"{'='*60}")
    
    # 1. Load and modify config for fast testing
    config = OmegaConf.load(config_path)
    config_dev = modify_config_for_fast_testing(config, max_epochs=1, k_fold=2)
    
    # 2. Create experiment directory structure in experiments/training/
    # Use unique experiment ID to avoid conflicts
    import uuid
    exp_id = f"sweep_test_{protocol_name}_{sweep_mode}_{uuid.uuid4().hex[:8]}"
    
    # Create experiment directory
    root_dir = Path(__file__).parent.parent.parent
    exp_dir = root_dir / "experiments" / "training" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save modified config to experiment directory
        config_path_in_exp = exp_dir / "config.yaml"
        OmegaConf.save(config_dev, config_path_in_exp)
        
        # 3. Create sweep.yaml with minimal parameters
        sweep_config = create_sweep_config_for_testing(max_values_per_param=2)
        num_params = count_parameters_in_sweep(sweep_config)
        
        # Safety check: Skip combination mode if too many parameters
        if sweep_mode == "combination" and num_params > 2:
            print(f"\n⚠ Skipping combination mode: too many parameters ({num_params} > 2)")
            print("This would create too many combinations for fast testing.")
            pytest.skip(f"Too many parameters ({num_params}) for combination mode in testing")
        
        # Create sweeper directory
        sweeper_dir = exp_dir / "sweeper"
        sweeper_dir.mkdir(exist_ok=True)
        
        # Save sweep.yaml
        sweep_yaml_path = sweeper_dir / "sweep.yaml"
        with open(sweep_yaml_path, 'w') as f:
            yaml.dump(sweep_config, f)
        
        print(f"\n✓ Created experiment structure at {exp_dir}")
        print(f"✓ Created sweep config with {num_params} parameters")
        
        # 4. Calculate expected number of runs
        expected_runs = calculate_expected_runs(sweep_config, sweep_mode)
        print(f"✓ Expecting {expected_runs} runs in {sweep_mode} mode")
        
        # 5. Run sweep via CLI
        print(f"\n→ Starting {sweep_mode} sweep via CLI...")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'sweep',
            '--exp_id', exp_id,
            '--sweep_mode', sweep_mode,
            '--exp_tag', f'cli_test_{protocol_name}'
        ])
        
        # Check CLI execution
        if result.exit_code != 0:
            print(f"\n✗ CLI execution failed with exit code {result.exit_code}")
            print(f"Output: {result.output}")
            if result.exception:
                print(f"Exception: {result.exception}")
                import traceback
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
            pytest.fail(f"CLI sweep command failed with exit code {result.exit_code}")
        
        print(f"✓ CLI sweep completed successfully")
        
        # 6. Validate results
        try:
            validate_sweep_results(str(exp_dir), sweep_mode, expected_runs)
            print(f"✓ All validations passed")
            
        except AssertionError as e:
            pytest.fail(f"Validation failed for {sweep_mode} sweep: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"✓ Test completed successfully: {sweep_mode} sweep")
        print(f"{'='*60}\n")
    
    finally:
        # Cleanup: Remove experiment directory
        print(f"\n→ Cleaning up experiment directory: {exp_dir}")
        if exp_dir.exists():
            try:
                shutil.rmtree(exp_dir)
                print(f"✓ Cleanup successful")
            except Exception as e:
                print(f"⚠ Warning: Could not clean up {exp_dir}: {e}")


# Mark tests as slow for CI filtering
pytestmark = pytest.mark.slow


if __name__ == "__main__":
    """
    Allow running this test file directly for quick debugging.
    Usage: 
        python test/integration/test_sweep_compatibility.py
        python test/integration/test_sweep_compatibility.py --protocol=test_baseline_MLP_ishigami_sum
    """
    print(f"\nSweep Compatibility Test")
    print(f"Default protocol: {DEFAULT_PROTOCOL}")
    print(f"Override with: pytest {__file__} --protocol=<experiment_name>")
    print("=" * 60)
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
