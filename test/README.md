# proT Test Suite

Comprehensive test suite for the proT project, organized by test type.

## Directory Structure

```
proT/test/
├── unit/                    # Core model component tests (fast)
│   ├── test_attention.py
│   ├── test_decoder.py
│   ├── test_embedding.py
│   ├── test_encoder.py
│   ├── test_model.py
│   ├── test_modular_embedding.py
│   └── test_predicted_mask.py
│
└── integration/             # Application & workflow tests (slower)
    ├── test_protocol_compatibility.py
    ├── test_optuna_compatibility.py
    ├── utils.py
    └── README.md           # Detailed integration test documentation
```

## Quick Start

### Run All Tests
```bash
pytest proT/test/ -v
```

### Run Unit Tests Only (Fast)
```bash
# All unit tests
pytest proT/test/unit/ -v

# Specific component
pytest proT/test/unit/test_attention.py -v
```

### Run Integration Tests Only
```bash
# All integration tests
pytest proT/test/integration/ -v

# Protocol compatibility (fast smoke tests)
pytest proT/test/integration/test_protocol_compatibility.py -v

# Optuna integration (slower, marked as slow)
pytest proT/test/integration/test_optuna_compatibility.py -v
```

### Skip Slow Tests
```bash
pytest proT/test/ -v -m "not slow"
```

## Test Categories

### Unit Tests (`unit/`)
Fast, focused tests for individual model components:
- **Purpose**: Verify core model functionality
- **Speed**: Fast (< 1s per test typically)
- **Isolation**: Test components independently
- **Run before**: Every commit

**Components tested:**
- Attention mechanisms (scaled dot-product, masking)
- Encoder/Decoder layers
- Embedding layers (value, variable, position, time)
- Complete model architecture
- Predicted masks

### Integration Tests (`integration/`)
End-to-end workflow and application tests:
- **Purpose**: Verify complete workflows and integrations
- **Speed**: Slower (minutes for full suite)
- **Scope**: Test full training pipelines, optimization workflows
- **Run before**: Major changes, releases

**Workflows tested:**
- Protocol experiment compatibility (all model variants)
- Optuna hyperparameter optimization
- Training pipeline end-to-end

## Supported Model Types

| Model Type | Unit Tests | Integration Tests |
|------------|------------|-------------------|
| **proT** | ✅ | ✅ |
| **proT_sim** | ✅ | ✅ |
| **proT_adaptive** | ✅ | ✅ |
| **MLP** | ✅ | ✅ |
| **LSTM** | ✅ | ✅ |
| **GRU** | ✅ | ✅ |
| **TCN** | ✅ | ✅ |

## CI/CD Workflow

### Recommended Test Strategy
```bash
# 1. Fast feedback: Run unit tests locally
pytest proT/test/unit/ -v

# 2. Before commit: Run non-slow tests
pytest proT/test/ -v -m "not slow"

# 3. Before push: Run full suite
pytest proT/test/ -v
```

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest proT/test/unit/ -v
  
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: pytest proT/test/integration/ -v
```

## Development Guidelines

### Adding Unit Tests
Place in `unit/` directory:
```python
# proT/test/unit/test_my_component.py
import pytest
from proT.models import MyComponent

def test_my_component_basic():
    component = MyComponent()
    result = component.forward(input_data)
    assert result.shape == expected_shape
```

### Adding Integration Tests
Place in `integration/` directory:
```python
# proT/test/integration/test_my_workflow.py
import pytest
from proT.training import train_model

@pytest.mark.slow  # Mark if test takes > 5 seconds
def test_my_workflow():
    config = load_config("my_config.yaml")
    result = train_model(config)
    assert result.metrics["val_loss"] < threshold
```

## Troubleshooting

### Import Errors
```bash
# Install proT in development mode
pip install -e .
```

### Tests Not Discovered
```bash
# Ensure __init__.py files exist
ls proT/test/__init__.py
ls proT/test/unit/__init__.py
ls proT/test/integration/__init__.py
```

### Windows File Locking
Integration tests handle SQLite file locking automatically. If you see cleanup warnings, they're informational only.

## More Information

- **Integration Test Details**: See `integration/README.md`
- **Optuna Integration**: See `../proT/euler_optuna/`
- **pytest Documentation**: https://docs.pytest.org/

---

**Quick Tips:**
- Run `pytest proT/test/unit/ -v` for fast feedback
- Mark slow tests with `@pytest.mark.slow`
- Keep unit tests focused and fast
- Use integration tests for end-to-end validation
