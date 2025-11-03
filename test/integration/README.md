# proT Test Suite

This directory contains the test suite for the proT project, ensuring code quality and compatibility across different model variants and configurations.

## Test Files

### `test_protocol_compatibility.py`
Tests that all protocol experiments can compile and run at least one training batch.

**Purpose:**
- Ensures code changes don't break existing experiment templates
- Validates model compilation and basic training functionality
- Fast smoke tests for model configurations

**Usage:**
```bash
# Run all protocol compatibility tests
pytest proT/test/test_protocol_compatibility.py -v

# Run specific test
pytest proT/test/test_protocol_compatibility.py::test_protocol_experiment_compatibility[test_proT_ishigami] -v -s
```

### `test_optuna_compatibility.py`
Tests that all protocol experiments work with Optuna hyperparameter optimization.

**Purpose:**
- Validates Optuna integration across all model types
- Tests complete optimization workflow: create → resume → summary
- Ensures sampling functions work for all model variants

**Features:**
- ✅ Minimal trials (3) for fast testing
- ✅ Windows-compatible cleanup (handles SQLite file locks)
- ✅ Platform-specific retry logic
- ✅ Comprehensive result validation

**Usage:**
```bash
# Run all Optuna compatibility tests
pytest proT/test/test_optuna_compatibility.py -v -s

# Run specific model test
pytest proT/test/test_optuna_compatibility.py::test_optuna_workflow_compatibility[test_adaptive_deterministic] -v -s

# Skip slow tests
pytest proT/test/ -v -m "not slow"
```

### `utils.py`
Shared utilities for test files.

**Contents:**
- `discover_protocol_configs()` - Auto-discovers protocol experiment configs
- `modify_config_for_fast_testing()` - Modifies configs for fast execution
- Constants: `ROOT_DIR`, `PROTOCOL_DIR`, `DATA_DIR`

## Supported Model Types

The test suite covers all proT model variants:

| Model Type | Test Coverage |
|------------|---------------|
| **proT** | ✅ Protocol + Optuna |
| **proT_sim** | ✅ Protocol + Optuna |
| **proT_adaptive** | ✅ Protocol + Optuna |
| **MLP** | ✅ Protocol + Optuna |
| **LSTM** | ✅ Protocol + Optuna |
| **GRU** | ✅ Protocol + Optuna |
| **TCN** | ✅ Protocol + Optuna |

## Running Tests

### Quick Start
```bash
# Run all tests
pytest proT/test/ -v

# Run with output
pytest proT/test/ -v -s

# Run only fast tests
pytest proT/test/test_protocol_compatibility.py -v
```

### Advanced Usage
```bash
# Stop on first failure
pytest proT/test/ -v -x

# Run in parallel (if pytest-xdist installed)
pytest proT/test/ -v -n auto

# Show coverage
pytest proT/test/ --cov=proT --cov-report=html

# Quiet mode (just pass/fail)
pytest proT/test/ -q
```

## Test Configuration

### Protocol Experiments Location
```
proT/experiments/training/tests/protocol/
├── test_MLP_ishigami/
├── test_proT_ishigami/
├── test_RNN_ishigami/
├── test_TCN_ishigami/
├── test_adaptive_deterministic/
├── test_adaptive_random/
└── test_simulator/
```

### Fast Testing Parameters
- `max_epochs`: 1-2 (instead of full training)
- `k_fold`: 2 (instead of 5+)
- `max_data_size`: 10 samples (instead of full dataset)
- `n_trials`: 3 (for Optuna tests)

## Troubleshooting

### Windows PermissionError
The test suite handles SQLite file locking on Windows automatically:
- Implements retry logic with progressive backoff
- Gracefully handles cleanup failures
- Logs warnings without failing tests

If you see cleanup warnings, they're informational only and don't affect test results.

### Test Discovery Issues
If tests aren't discovered:
```bash
# Make sure test/ is a package
ls proT/test/__init__.py

# Run from project root
cd proT
pytest test/ -v
```

### Import Errors
If you get import errors:
```bash
# Install proT in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Adding New Tests

### 1. Protocol Compatibility Test
Simply add a new experiment config to:
```
proT/experiments/training/tests/protocol/your_new_test/
└── config_your_model.yaml
```

The test will be auto-discovered!

### 2. Optuna Sampling Function
If adding a new model type, add its sampling function to:
```python
# proT/proT/euler_optuna/cli.py

def your_model_sample_params(trial):
    return {
        "model.param1": trial.suggest_int("param1", 64, 256),
        "training.lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }

# Then add to dispatcher
params_select = {
    ...
    "YourModel": your_model_sample_params,
}
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Run fast tests
        run: pytest proT/test/test_protocol_compatibility.py -v
      - name: Run Optuna tests
        run: pytest proT/test/test_optuna_compatibility.py -v -m slow
```

## Best Practices

1. **Run tests before committing**
   ```bash
   pytest proT/test/test_protocol_compatibility.py -v
   ```

2. **Add protocol tests for new models**
   - Create config in `experiments/training/tests/protocol/`
   - Tests auto-discover and run

3. **Test Optuna integration for new models**
   - Add sampling function to `cli.py`
   - Update dispatcher in `sample_params_for_optuna()`

4. **Keep tests fast**
   - Use minimal epochs/folds/data
   - Mark slow tests with `@pytest.mark.slow`

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [proT Optuna Integration](../proT/euler_optuna/README.md)
