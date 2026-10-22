# Optuna Sampling Bounds Documentation

**Last Updated:** November 2, 2025  
**Profile:** baseline  
**File Reference:** `proT/euler_optuna/cli.py`

## Overview

This document provides comprehensive documentation of the hyperparameter sampling bounds used in Optuna optimization studies for proT models. The sampling bounds are centrally defined in `BASELINE_SAMPLING_BOUNDS` dictionary and used across all model architectures to ensure consistent and fair hyperparameter optimization.

## Profile System

The sampling system supports multiple profiles for different optimization scenarios:

```python
SAMPLING_PROFILES = {
    "baseline": BASELINE_SAMPLING_BOUNDS,   # Default profile for fair model comparison
    # Future profiles can be added here
}
```

### Selecting a Profile

Profiles are selected via CLI:
```bash
python cli.py paramsopt --exp_id my_exp --mode create --sampling_profile baseline
```

The default profile is **baseline**.

---

## Baseline Sampling Bounds

### Common Parameters

These parameters are shared across multiple model architectures:

| Parameter | Low | High | Step | Log Scale | Description |
|-----------|-----|------|------|-----------|-------------|
| `d_model_set` | 64 | 512 | 16 | No | Embedding dimension for baseline models |
| `d_hidden_set` | 64 | 512 | 16 | No | Hidden layer dimension |
| `dropout` | 0.0 | 0.3 | - | No | Standard dropout rate |
| `lr` | 1e-4 | 1e-3 | - | Yes | Learning rate (log scale) |
| `lr_stepped` | 1e-4 | 1e-3 | 1e-4 | No | Learning rate with fixed step (for proT) |

### RNN-Specific Parameters

| Parameter | Low | High | Step | Description |
|-----------|-----|------|------|-------------|
| `n_layers` | 1 | 4 | 1 | Number of recurrent layers (LSTM/GRU) |

### proT Embedding Dimensions

| Parameter | Low | High | Step | Description |
|-----------|-----|------|------|-------------|
| `embedding_dim_standard` | 50 | 200 | 10 | Standard embedding dimensions (position, variable, value) |
| `embedding_dim_time` | 10 | 100 | 10 | Temporal embedding dimensions |
| `embedding_dim_adaptive` | 30 | 100 | 10 | Adaptive target embedding (proT_adaptive only) |

### proT Architecture Parameters

| Parameter | Low | High | Step | Description |
|-----------|-----|------|------|-------------|
| `n_heads` | 1 | 3 | 1 | Number of attention heads |
| `d_ff` | 200 | 600 | 100 | Feed-forward network dimension |
| `d_qk` | 100 | 200 | 50 | Query/Key dimension for attention |

### proT Dropout Parameters

All proT dropout parameters use the same bounds with fine-grained control:

| Parameter | Low | High | Step | Description |
|-----------|-----|------|------|-------------|
| `dropout_fine` | 0.0 | 0.3 | 0.1 | Fine-grained dropout for all transformer components |

**Applied to:**
- `dropout_emb` - Embedding layer dropout
- `dropout_data` - Input data dropout
- `dropout_attn_out` - Attention output dropout
- `dropout_ff` - Feed-forward network dropout
- `enc_dropout_qkv` - Encoder Q/K/V dropout
- `enc_attention_dropout` - Encoder attention weights dropout
- `dec_self_dropout_qkv` - Decoder self-attention Q/K/V dropout
- `dec_self_attention_dropout` - Decoder self-attention weights dropout
- `dec_cross_dropout_qkv` - Decoder cross-attention Q/K/V dropout
- `dec_cross_attention_dropout` - Decoder cross-attention weights dropout

---

## Model-Specific Sampling

### MLP (Multi-Layer Perceptron)

**Sampled Parameters:**
- `mlp_arch` - Architecture selection (categorical)
  - "mlp_s-128x3": [128, 128, 128]
  - "mlp_m-256x3": [256, 256, 256]
  - "mlp_m-512x3": [512, 512, 512]
- `d_model_set` - Embedding dimension (from common params)
- `d_hidden_set` - Hidden dimension (from common params)
- `lr` - Learning rate (from common params)
- `dropout` - Dropout rate (from common params)

**Total parameters optimized:** 5 (1 categorical + 4 continuous/integer)

### RNN (LSTM/GRU)

**Sampled Parameters:**
- `n_layers_set` - Number of recurrent layers (from RNN params)
- `d_model_set` - Embedding dimension (from common params)
- `d_hidden_set` - Hidden dimension (from common params)
- `lr` - Learning rate (from common params)
- `dropout` - Dropout rate (from common params)

**Total parameters optimized:** 5 (all integer/continuous)

### TCN (Temporal Convolutional Network)

**Sampled Parameters:**
- `tcn_arch` - Architecture selection (categorical)
  - "tcn_s-32-32-64": [32, 32, 64]
  - "tcn_s-64-64-128": [64, 64, 128]
  - "tcn_m-64-128-256": [64, 128, 256]
  - "tcn_m-128-128-128": [128, 128, 128]
  - "tcn_l-128-256-256": [128, 256, 256]
- `d_model_set` - Embedding dimension (from common params)
- `lr` - Learning rate (from common params)
- `dropout` - Dropout rate (from common params)

**Total parameters optimized:** 4 (1 categorical + 3 continuous/integer)

### proT (Transformer)

proT supports two optimization modes based on configuration:

#### Benchmark Mode
**Enabled when:**
- `use_uniform_embedding_dims: true` in config, OR
- Using `summation` embedding composition

**Sampled Parameters:**
- `d_model_set` - Single embedding dimension applied to all embeddings (from proT params)
- `n_heads` - Number of attention heads (from proT architecture params)
- `d_ff` - Feed-forward dimension (from proT architecture params)
- `d_qk` - Query/Key dimension (from proT architecture params)
- 10× `dropout_*` - All dropout parameters (from proT dropout params)
- `lr` - Learning rate (from lr_stepped)

**Total parameters optimized:** 15 (1 embedding + 3 architecture + 10 dropout + 1 lr)

**Embedding dimensions (all set to d_model_set):**
- enc_val_emb_hidden
- enc_var_emb_hidden
- enc_pos_emb_hidden
- enc_time_emb_hidden
- dec_val_emb_hidden
- dec_var_emb_hidden
- dec_pos_emb_hidden
- dec_time_emb_hidden

#### Research Mode
**Enabled when:**
- `use_uniform_embedding_dims: false` in config, AND
- Using `concat` embedding composition

**Sampled Parameters:**
- 8× Independent embedding dimensions (from proT embedding params)
  - 6× standard embeddings (enc_val, enc_var, enc_pos, dec_val, dec_var, dec_pos)
  - 2× time embeddings (enc_time, dec_time)
- `n_heads` - Number of attention heads (from proT architecture params)
- `d_ff` - Feed-forward dimension (from proT architecture params)
- `d_qk` - Query/Key dimension (from proT architecture params)
- 10× `dropout_*` - All dropout parameters (from proT dropout params)
- `lr` - Learning rate (from lr_stepped)

**Total parameters optimized:** 22 (8 embeddings + 3 architecture + 10 dropout + 1 lr)

### proT_adaptive (Adaptive Transformer)

Same as proT Research Mode, but with an additional embedding parameter:

**Additional Parameter:**
- `dec_val_given_emb_hidden` - Adaptive target embedding (from embedding_dim_adaptive)

**Total parameters optimized:** 23 (9 embeddings + 3 architecture + 10 dropout + 1 lr)

---

## Rationale for Bounds

### Embedding Dimensions
- **d_model_set (64-512, step 16):** Balances model capacity with computational efficiency for baseline models
- **embedding_dim_standard (50-200, step 10):** Appropriate range for proT embeddings to capture feature relationships
- **embedding_dim_time (10-100, step 10):** Narrower range as temporal features typically need less capacity

### Architecture
- **n_layers (1-4):** More layers may lead to overfitting on small datasets
- **n_heads (1-3):** Limited heads for smaller datasets; higher values tested for larger datasets
- **d_ff (200-600, step 100):** Standard transformer feed-forward expansion
- **d_qk (100-200, step 50):** Query/key dimension for attention computation

### Dropout
- **dropout (0.0-0.3):** Standard range; values >0.3 may hurt performance
- **dropout_fine (0.0-0.3, step 0.1):** Fine-grained control with discrete steps

### Learning Rate
- **lr (1e-4 to 1e-3, log scale):** Standard range for Adam optimizer
- **lr_stepped (1e-4 to 1e-3, step 1e-4):** Discrete sampling for reproducibility

---

## Usage Examples

### Running with Default Profile

```bash
# Create study
python cli.py paramsopt --exp_id baseline_LSTM_ishigami_cat --mode create

# Resume optimization  
python cli.py paramsopt --exp_id baseline_LSTM_ishigami_cat --mode resume

# View results
python cli.py paramsopt --exp_id baseline_LSTM_ishigami_cat --mode summary
```

### Running with Explicit Profile

```bash
python cli.py paramsopt \
    --exp_id baseline_proT_ishigami_sum \
    --mode create \
    --sampling_profile baseline
```

---

## Extending the System

### Adding a New Profile

1. **Define new bounds dictionary:**
```python
EXTENDED_SAMPLING_BOUNDS = {
    "d_model_set": {"low": 32, "high": 1024, "step": 32},
    "d_hidden_set": {"low": 32, "high": 1024, "step": 32},
    # ... other parameters
}
```

2. **Register in SAMPLING_PROFILES:**
```python
SAMPLING_PROFILES = {
    "baseline": BASELINE_SAMPLING_BOUNDS,
    "extended": EXTENDED_SAMPLING_BOUNDS,
}
```

3. **Update CLI choices:**
```python
@click.option("--sampling_profile", default="baseline",
              type=click.Choice(['baseline', 'extended']),
              help="Sampling bounds profile to use")
```

4. **Use new profile:**
```bash
python cli.py paramsopt --exp_id my_exp --mode create --sampling_profile extended
```

### Modifying Bounds

To adjust bounds globally, simply edit `BASELINE_SAMPLING_BOUNDS`:

```python
BASELINE_SAMPLING_BOUNDS = {
    # Increase embedding dimension range
    "d_model_set": {"low": 64, "high": 1024, "step": 32},  # Changed from 512 to 1024
    # ...
}
```

This change will automatically apply to all models using the baseline profile.

---

## Best Practices

1. **Profile Selection:**
   - Use `baseline` for fair model comparison with tested bounds
   - Create custom profiles for specific research questions

2. **Bound Tuning:**
   - Start with baseline bounds
   - Analyze optimization results to identify if bounds are limiting
   - Expand bounds gradually if needed

3. **Step Sizes:**
   - Smaller steps provide finer granularity but slower optimization
   - Larger steps speed up optimization but may miss optimal values
   - Balance based on computational budget

4. **Documentation:**
   - Document rationale when changing bounds
   - Track which profile was used for each experiment
   - Compare results only within the same profile

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-02 | Initial documentation with baseline profile |

---

## References

- Optuna documentation: https://optuna.readthedocs.io/
- proT configuration: `proT/training/experiment_control.py`
- Experiment configs: `experiments/training/baseline_optuna/`
