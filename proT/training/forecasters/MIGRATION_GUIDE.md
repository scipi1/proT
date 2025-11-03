# Migration Guide: TransformerForecaster to New Forecasters

## Overview

The functionality from `transformer_forecaster.py` has been split into four specialized forecasters, each designed for specific use cases. This document explains the migration path and helps you choose the right forecaster.

## New Forecaster Architecture

```
SimpleForecaster (base)
├── SimulatorForecaster (adds physics simulation)
└── OnlineTargetForecaster (adds curriculum learning)
    └── OptimizationForecaster (adds advanced optimization)
```

## Forecaster Selection Guide

### 1. **SimpleForecaster** - Start Here
**Use when:** You need a clean, straightforward transformer forecaster with standard training.

**Features:**
- ✅ Standard MSE loss
- ✅ AdamW optimizer
- ✅ Basic metrics (MAE, RMSE, R2)
- ✅ Entropy logging
- ✅ Clean, maintainable code

**Config Requirements:**
```yaml
training:
  loss_fn: "mse"
  lr: 0.001
  weight_decay: 0.01  # optional
data:
  val_idx: 0  # index of value feature in target
```

**Recommendation:** Use this for most standard forecasting tasks.

---

### 2. **SimulatorForecaster** - Physics-Informed
**Use when:** You have physics-based constraints or dynamics in your problem.

**Features:**
- ✅ All features from SimpleForecaster
- ✅ Physics-informed decoder input generation
- ✅ Trajectory simulation post-processing
- ✅ Integration with ISTSimulator

**Config Requirements:**
```yaml
training:
  pinn: true
  simulator_model: "F"  # optional, default: "F"
  # ... plus all SimpleForecaster config
```

**Recommendation:** Use for physics-based problems (e.g., trajectory prediction, dynamic systems).

---

### 3. **OnlineTargetForecaster** - Curriculum Learning
**Use when:** You want progressive target revelation during training.

**Features:**
- ✅ All features from SimpleForecaster
- ✅ Progressive target revelation (curriculum learning)
- ✅ Fixed or random target showing modes
- ✅ BCE loss for NaN regions (2-channel output)
- ✅ Entropy regularization
- ✅ MSE masking based on position
- ✅ **Given indicator feature** (new addition)

**Config Requirements:**
```yaml
training:
  loss_fn: "mse"
  lr: 0.001
  lam: 1.0  # BCE loss weight (optional)
  gamma: 0.1  # entropy regularization weight (optional)
  entropy_regularizer: false  # optional
  
  # Curriculum learning settings
  target_show_mode: "fixed"  # or "random"
  epoch_show_trg: 10  # when to start showing targets
  show_trg_max_idx: 50  # max position to reveal (fixed mode)
  show_trg_upper_bound_max: 100  # upper bound (random mode)
  
data:
  val_idx: 0  # index of value feature
  pos_idx: 3  # index of position feature
```

**Recommendation:** Use for tasks where gradual exposure to targets improves learning.

---

### 4. **OptimizationForecaster** - Advanced Research
**Use when:** You need fine-grained control over the optimization process.

**Features:**
- ✅ All features from OnlineTargetForecaster
- ✅ 7 different optimization modes
- ✅ Manual optimization control
- ✅ Parameter splitting (embeddings vs model)
- ✅ Mid-training optimizer switching
- ✅ Learning rate schedulers with warmup
- ✅ Optional gradient clipping

**Config Requirements:**
```yaml
training:
  optimization: 1  # or 2, 3, 4, 5, 6, 7
  switch_epoch: 50
  switch_step: 1000
  base_lr: 0.001
  emb_lr: 0.0001  # modes 2-7
  emb_start_lr: 0.00001  # modes 3-7
  warmup_steps: 100  # modes 5-7
  # ... plus all OnlineTargetForecaster config
```

**Optimization Modes:**
1. **Mode 1:** Same AdamW for all parameters
2. **Mode 2:** Adam with different LRs (embedding vs model)
3. **Mode 3:** Adam + SparseAdam combination
4. **Mode 4:** Adam + Adagrad combination
5. **Mode 5:** SGD → Adam switch with Adagrad embedding + schedulers
6. **Mode 6:** SGD → Adam switch with Adagrad embedding (alt config) + schedulers
7. **Mode 7:** SGD → Adam switch (learn embeddings then model) + schedulers

**Recommendation:** Use only for research experiments requiring complex optimization strategies.

---

## Migration from TransformerForecaster

### Basic Setup (No Special Features)
**Before:**
```python
from proT.training.forecasters import TransformerForecaster

forecaster = TransformerForecaster(config)
```

**After:**
```python
from proT.training.forecasters import SimpleForecaster

forecaster = SimpleForecaster(config)
```

---

### With Physics Simulation
**Before:**
```python
config["training"]["pinn"] = True
forecaster = TransformerForecaster(config)
```

**After:**
```python
forecaster = SimulatorForecaster(config)
```

---

### With Curriculum Learning
**Before:**
```python
config["training"]["target_show_mode"] = "random"
config["training"]["epoch_show_trg"] = 10
forecaster = TransformerForecaster(config)
```

**After:**
```python
forecaster = OnlineTargetForecaster(config)
```

---

### With Advanced Optimization
**Before:**
```python
config["training"]["optimization"] = 5
config["training"]["switch_epoch"] = 50
forecaster = TransformerForecaster(config)
```

**After:**
```python
forecaster = OptimizationForecaster(config)
```

---

## Key Differences and Improvements

### 1. Given Indicator Feature (New in OnlineTargetForecaster)
The decoder input now includes a "given indicator" feature that signals which target values are revealed:
- `1.0` where targets are given
- `0.0` where targets are not given

This helps the model distinguish between observed and missing data.

### 2. Cleaner Code Structure
Each forecaster has a single, focused purpose:
- Easier to understand
- Easier to maintain
- Easier to extend
- Better documentation

### 3. Automatic Optimization (Most Forecasters)
- SimpleForecaster, SimulatorForecaster, OnlineTargetForecaster use automatic optimization
- Only OptimizationForecaster uses manual optimization
- Simpler training loops for most use cases

### 4. No Simulator in OnlineTargetForecaster
If you need both simulation AND curriculum learning, you have two options:
1. Choose which is more important for your task
2. Create a custom forecaster extending SimulatorForecaster with curriculum learning features

---

## Complete Functionality Coverage

✅ **All functionalities from TransformerForecaster are covered:**

| Feature | Covered By |
|---------|------------|
| Basic model setup | All forecasters |
| MSE loss | All forecasters |
| Metrics (MAE, RMSE, R2) | All forecasters |
| Entropy logging | All forecasters |
| Simple AdamW optimizer | SimpleForecaster |
| Physics simulation | SimulatorForecaster |
| Curriculum learning | OnlineTargetForecaster, OptimizationForecaster |
| BCE loss for NaN | OnlineTargetForecaster, OptimizationForecaster |
| Entropy regularization | OnlineTargetForecaster, OptimizationForecaster |
| 7 optimization modes | OptimizationForecaster |
| Manual optimization | OptimizationForecaster |
| Optimizer switching | OptimizationForecaster |
| LR schedulers | OptimizationForecaster |
| Gradient clipping | OptimizationForecaster |

---

## Recommendations

### For Production/Standard Use:
→ **SimpleForecaster** or **OnlineTargetForecaster**

### For Physics-Based Problems:
→ **SimulatorForecaster**

### For Research/Experimentation:
→ **OptimizationForecaster**

---

## Questions?

If you're unsure which forecaster to use:
1. Start with **SimpleForecaster**
2. Add features incrementally as needed
3. Only use **OptimizationForecaster** if experiments show it's necessary

The new architecture is designed to be:
- **Simple by default**
- **Powerful when needed**
- **Easy to understand and maintain**
