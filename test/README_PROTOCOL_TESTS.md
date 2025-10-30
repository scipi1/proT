# Protocol Experiment Compatibility Tests

## Overview

This test suite verifies that all protocol experiment templates in `experiments/training/tests/protocol/` remain compatible with the codebase as it evolves. Each test runs a training session in "dev mode" (1 epoch, minimal data) to ensure the model compiles and runs without errors.

## Installation

### Install pytest (required)

```bash
pip install pytest
```

### Optional: Add to requirements.txt

If you want to include pytest in your project dependencies, add this line to `requirements.txt`:

```
pytest==8.0.0
```

## Running the Tests

### Run all protocol tests

```bash
pytest test/test_protocol_compatibility.py -v
```

### Run a specific experiment test

```bash
pytest test/test_protocol_compatibility.py -k "adaptive_deterministic" -v
```

### Run with detailed output (for debugging)

```bash
pytest test/test_protocol_compatibility.py -v -s
```

### Run from within the test directory

```bash
cd test
pytest test_protocol_compatibility.py -v
```

## How It Works

### Auto-Discovery

The test automatically discovers all config files in the protocol directory:
- `experiments/training/tests/protocol/test_adaptive_deterministic/config_*.yaml`
- `experiments/training/tests/protocol/test_adaptive_random/config_*.yaml`
- `experiments/training/tests/protocol/test_simulator/config_*.yaml`

### Dev Mode Configuration

For fast execution, the test overrides these config parameters:
- `max_epochs = 1` (only one training epoch)
- `max_data_size = 10` (only 10 samples)
- `save_ckpt_every_n_epochs = 999` (disable checkpointing)
- Original `k_fold` value is preserved (sklearn KFold requires minimum 2 splits)

### Validation

Each test verifies:
1. Config loads successfully
2. Model instantiates without errors
3. Training completes one epoch
4. Returns valid metrics DataFrame
5. No runtime exceptions occur

## Adding New Protocol Experiments

Simply add a new directory under `experiments/training/tests/protocol/` with a config file:

```
experiments/training/tests/protocol/
├── test_adaptive_deterministic/
│   └── config_proT_dyconex_v5_1.yaml
├── test_adaptive_random/
│   └── config_proT_dyconex_v5_1.yaml
├── test_simulator/
│   └── config_proT_dyconex_v5_1.yaml
└── your_new_experiment/          # <-- Add here
    └── config_your_experiment.yaml
```

The test will automatically discover and run your new experiment.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Protocol Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest test/test_protocol_compatibility.py -v
```

## Troubleshooting

### ModuleNotFoundError: No module named 'proT'

If running directly with `python test/test_protocol_compatibility.py`, make sure your package is installed:

```bash
pip install -e .
```

Or use pytest which handles imports correctly:

```bash
pytest test/test_protocol_compatibility.py -v
```

### Test takes too long

- Check that `max_epochs=1` and `max_data_size=10` are being applied
- Verify data is available at `data/input/`
- Consider reducing k_fold in your config (minimum 2)

### Data not found errors

Ensure your data directory exists and contains the required dataset:
```
data/input/ds_dx_250806_panel_200_pad/
├── X.npy
└── Y.npy
```

## Test Markers

The test file includes the `@pytest.mark.integration` marker for filtering:

```bash
# Run only integration tests
pytest -m integration -v

# Skip integration tests
pytest -m "not integration" -v
