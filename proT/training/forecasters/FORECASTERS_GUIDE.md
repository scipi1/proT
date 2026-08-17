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

### 4. EncoderForecaster (Encoder-only proT)

**File:** `encoder_forecaster.py`

**Description:** Encoder-only proT variant (`proT_enc`). The transformer encoder processes the input sequence; a learnable query token pools the encoder outputs via attention into a context vector. Two head variants are supported:

- **Scalar head** (`head_type: "scalar"`): predicts a D-dimensional vector of scalar targets per sample (e.g. IST slope, fitting coefficients). The dataset provides the scalar targets directly; the model output is `(B, D, out_dim)`.
- **GRU head** (`head_type: "gru"`): the context vector initializes the hidden state of a GRU, which is unrolled over the target positions. Per-step inputs are the target covariates (position, variable, value, given-indicator) embedded with the existing `ModularEmbedding`. The value channel is fed with the real target where given (teacher forcing / "real IST" at inference) and with the predicted mean elsewhere (autoregressive rollout).

**Features:**
- ✅ Probabilistic losses (`PositionWeightedMSE` / `CensoredGaussianNLL`) unchanged
- ✅ Scheduled curriculum (annealed forcing bound) as in `OnlineTargetForecaster`
- ✅ "Real IST" at inference via `show_trg_max_idx`
- ✅ Attention pooling over the encoder outputs (no decoder self/cross attention)
- ✅ Compatible with the existing trainer, predictor and plotting infrastructure

**Configuration Requirements:**
```yaml
model:
  model_object: "proT_enc"
  kwargs:
    head_type: "gru"        # "scalar" | "gru"
    output_dim: null        # scalar head: number of scalar targets (D)
    d_hidden: 128           # GRU hidden dimension
    rnn_n_layers: 1
    rnn_dropout: 0.0
    dec_val_idx: 3          # value channel in the target tensor
    dec_pos_idx: 1          # position channel in the target tensor
```

**Usage:**
```python
from proT.training.forecasters.encoder_forecaster import EncoderForecaster

model = EncoderForecaster(config)
```

---

### 5. TransformerForecaster (Legacy/Complex)

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

---

## Optional Learning Controls

Three optional, config-driven training controls are available for all proT
forecasters (`EntropyRegularizedForecaster`, `SimpleForecaster`,
`SimulatorForecaster`, `OnlineTargetForecaster`). All default to off, so old
configs are unaffected.

```yaml
model:
  kwargs:
    # 1) freeze decoder input embeddings to force learning at the encoder
    freeze_dec_embedding: false     # true = frozen for the whole run

training:
  # 2) delayed Q/K learning (hard freeze OR linear LR ramp; ramp takes precedence)
  qk_freeze_scope: "all"            # "all" | "self" | "cross"
  qk_freeze_epochs: 0               # hard-freeze Q/K projections for N epochs (0 = off)
  qk_lr_ramp: false                 # linear LR ramp on Q/K projections
  qk_lr_ramp_start_factor: 1e-4     # Q/K LR starts at lr * start_factor
  qk_lr_ramp_end_frac: 0.95         # Q/K LR reaches full lr at 95% of max_epochs

  # 3) positional loss weighting (later target positions weigh more)
  loss_pos_weight: "none"           # "none" | "linear" | "exp"
  loss_pos_weight_alpha: 2.0        # sharpness for "exp"
```

**Notes:**
- `freeze_dec_embedding` freezes the whole decoder embedding module (variable,
  time2vec, value bias). Attention projections, FF layers and the encoder stay
  trainable; the decoder self-attention keeps working on the frozen inputs.
- The Q/K hard freeze is implemented as the `QKFreezeScheduler` callback
  (registered automatically in `trainer.py`); the LR ramp is a parameter-group
  `LambdaLR` built in the forecaster's `configure_optimizers`.
- The MSE loss (`proT/training/losses.py`, `PositionWeightedMSE`) always
  excludes NaN targets (previously they entered the loss as zeros) and applies
  mean-normalized position weights, so the loss scale stays comparable to the
  unweighted case. Metrics (mae/rmse/r2) remain unweighted for comparability.
- Unit tests: `test/unit/test_learning_options.py`.
