# ProT Forecasters Guide

## Overview

This guide explains the different forecaster modules available in ProT and how to use them.

## Available Forecasters

### 1. SimpleForecaster (Recommended for most use cases)

**File:** `simple_forecaster.py`

**Description:** Clean, simple forecaster with AdamW optimizer. This is the recommended starting point for most projects.

**Features:**
- ✅ Standard PyTorch Lightning training loop
- ✅ AdamW optimizer (simple and effective)
- ✅ MSE loss with NaN handling
- ✅ Comprehensive metrics (MAE, RMSE, R²)
- ✅ Attention entropy logging
- ✅ No complex optimization schemes

**Configuration Requirements:**
```yaml
training:
  loss_fn: "mse"
  lr: 0.0001              # Learning rate for AdamW
  weight_decay: 0.01      # Optional, default: 0.01

data:
  val_idx: 0              # Index of value feature in target
```

**Usage:**
```python
from proT.training.forecasters import SimpleForecaster

model = SimpleForecaster(config)
```

---

### 2. SimulatorForecaster (Physics-Informed)

**File:** `simulator_forecaster.py`

**Description:** Extends SimpleForecaster with physics-based trajectory simulation. Use this when you have known physical dynamics.

**Features:**
- ✅ All features from SimpleForecaster
- ✅ Integrates ISTSimulator for decoder input generation
- ✅ Post-processes predictions through trajectory simulator
- ✅ Useful for physics-informed neural networks (PINNs)

**Configuration Requirements:**
```yaml
training:
  loss_fn: "mse"
  lr: 0.0001
  pinn: true                    # Enable PINN mode
  simulator_model: "F"          # Simulator model name (default: "F")

data:
  val_idx: 0
```

**Usage:**
```python
from proT.training.forecasters import SimulatorForecaster

model = SimulatorForecaster(config)
```

**When to use:**
- You have known physical equations/dynamics
- Want to constrain predictions to be physically plausible
- Working with time-series that follow known trajectories

---

### 3. OnlineTargetForecaster (Curriculum Learning)

**File:** `online_target_forecaster.py`

**Description:** Extends SimpleForecaster with progressive target revelation during training (curriculum learning).

**Features:**
- ✅ All features from SimpleForecaster
- ✅ Progressively reveals target sequence during training
- ✅ Two modes: "fixed" or "random"
- ✅ Helps with long sequence training
- ✅ Simulates online/streaming scenarios

**Configuration Requirements:**
```yaml
training:
  loss_fn: "mse"
  lr: 0.0001
  target_show_mode: "random"           # or "fixed"
  epoch_show_trg: 10                   # Start showing targets at epoch 10
  show_trg_max_idx: 50                 # For "fixed" mode
  show_trg_max_idx_upper_bound: 100    # For "random" mode

data:
  val_idx: 0
  pos_idx: 3              # Index of position feature in target
```

**Usage:**
```python
from proT.training.forecasters import OnlineTargetForecaster

model = OnlineTargetForecaster(config)
```

**When to use:**
- Training on very long sequences
- Want to teach model to use partial observations
- Simulating real-time/streaming prediction scenarios
- Need curriculum learning for stability

---

### 4. TransformerForecaster (Legacy/Complex)

**File:** `transformer_forecaster.py`

**Description:** Original forecaster with 7 complex optimization schemes. Use only if you specifically need one of these optimization strategies.

**Features:**
- Manual optimization control
- 7 different optimizer configurations
- Parameter group splitting (embeddings vs model)
- Scheduler combinations
- All features from other forecasters combined

**Note:** This will be simplified in Phase 3 after the new forecasters are tested.

---

## Testing Your Forecasters

### Test 1: Import Test

```python
# Test imports
from proT.training.forecasters import (
    SimpleForecaster,
    SimulatorForecaster,
    OnlineTargetForecaster,
    TransformerForecaster
)

print("✅ All forecasters imported successfully!")
```

### Test 2: SimpleForecaster Test

```python
from proT.training.forecasters import SimpleForecaster
from proT.training.dataloader import ProcessDataModule
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Load your config
config = OmegaConf.load("path/to/your/config.yaml")

# Make sure config has required fields
config["training"]["lr"] = 0.0001
config["training"]["loss_fn"] = "mse"

# Create data module
dm = ProcessDataModule(
    data_dir="path/to/data",
    input_file="X.npy",
    target_file="Y.npy",
    batch_size=32,
    num_workers=4,
    data_format="float32",
    seed=42
)

# Create model
model = SimpleForecaster(config)

# Train
trainer = pl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(model, dm)

print("✅ SimpleForecaster training completed!")
```

### Test 3: SimulatorForecaster Test

```python
from proT.training.forecasters import SimulatorForecaster

# Ensure config has simulator settings
config["training"]["pinn"] = True
config["training"]["simulator_model"] = "F"

# Create model
model = SimulatorForecaster(config)

# Train (same as above)
trainer = pl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(model, dm)

print("✅ SimulatorForecaster training completed!")
```

### Test 4: OnlineTargetForecaster Test

```python
from proT.training.forecasters import OnlineTargetForecaster

# Ensure config has online target settings
config["training"]["target_show_mode"] = "random"
config["training"]["epoch_show_trg"] = 2
config["training"]["show_trg_max_idx_upper_bound"] = 50
config["data"]["pos_idx"] = 3

# Create model
model = OnlineTargetForecaster(config)

# Train (same as above)
trainer = pl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(model, dm)

print("✅ OnlineTargetForecaster training completed!")
```

---

## Comparison: SimpleForecaster vs TransformerForecaster

| Feature | SimpleForecaster | TransformerForecaster |
|---------|------------------|----------------------|
| Optimizer | AdamW (simple) | 7 complex schemes |
| Automatic optimization | ✅ Yes (PyTorch Lightning) | ❌ No (manual) |
| Code complexity | Low (~150 lines) | High (~500+ lines) |
| Parameter splitting | ❌ No | ✅ Yes (7 ways) |
| Schedulers | Simple | Complex sequences |
| Maintenance | Easy | Difficult |
| Recommended for | Most projects | Specific research needs |

---

## Migration Path

### From TransformerForecaster to SimpleForecaster

**Old config:**
```yaml
training:
  optimization: 1  # or 2, 3, 4, 5, 6, 7
  base_lr: 0.0001
  emb_lr: 0.001
  switch_epoch: 50
  switch_step: 100
  warmup_steps: 1000
```

**New config:**
```yaml
training:
  lr: 0.0001           # Just one learning rate!
  weight_decay: 0.01   # Optional
```

**Code change:**
```python
# Old
from proT.training.forecasters import TransformerForecaster
model = TransformerForecaster(config)

# New
from proT.training.forecasters import SimpleForecaster
model = SimpleForecaster(config)
```

---

## Best Practices

### 1. Start Simple
- Begin with `SimpleForecaster`
- Only move to specialized forecasters if you have specific needs

### 2. Configuration
- Keep configs minimal
- Only add complexity when necessary
- Document why you chose a specific forecaster

### 3. Testing
- Test each forecaster independently
- Use small epochs (5-10) for initial testing
- Verify metrics are logged correctly

### 4. Choosing a Forecaster

**Use SimpleForecaster if:**
- ✅ Standard time series forecasting
- ✅ No special physical constraints
- ✅ Want simple, maintainable code

**Use SimulatorForecaster if:**
- ✅ Have known physics/dynamics
- ✅ Want physically plausible predictions
- ✅ Working with PINN-style problems

**Use OnlineTargetForecaster if:**
- ✅ Training very long sequences
- ✅ Need curriculum learning
- ✅ Simulating streaming predictions

**Use TransformerForecaster if:**
- ✅ Specifically need one of the 7 optimization schemes
- ✅ Have existing configs that depend on it
- ⚠️ Consider migrating to simpler alternatives

---

## Troubleshooting

### Issue: Import Error
```python
ModuleNotFoundError: No module named 'proT.training.forecasters.simple_forecaster'
```

**Solution:** Make sure you've installed proT in editable mode:
```bash
pip install -e .
```

### Issue: Missing Config Keys
```python
KeyError: 'lr'
```

**Solution:** Add required keys to your config:
```yaml
training:
  lr: 0.0001
  loss_fn: "mse"
```

### Issue: Simulator Not Found
```python
ModuleNotFoundError: No module named 'proT.proj_specific.simulator'
```

**Solution:** Only use `SimulatorForecaster` if you have the simulator module. Otherwise use `SimpleForecaster`.

---

## Summary

- ✅ **SimpleForecaster**: Clean, recommended for most use cases
- ✅ **SimulatorForecaster**: For physics-informed neural networks
- ✅ **OnlineTargetForecaster**: For curriculum learning with long sequences
- ⚠️ **TransformerForecaster**: Legacy, will be simplified

**Recommendation:** Start with `SimpleForecaster` and only move to specialized versions if you have specific requirements.
