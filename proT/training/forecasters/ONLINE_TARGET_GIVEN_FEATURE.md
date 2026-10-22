# Online Target Forecaster - "Given" Feature Guide

## Overview

The `OnlineTargetForecaster` automatically appends a **"given indicator"** feature to the decoder input. This feature informs the model which parts of the target sequence are given (known) vs. which parts need to be predicted.

## How It Works

### 1. **Automatic Feature Appending**

During the forward pass, the forecaster:
1. Determines which target positions are "given" based on `show_trg_max_idx`
2. Creates a binary indicator: `1.0` = given, `0.0` = not given
3. Appends this indicator to the decoder input on the last dimension

**Example:**
- Original decoder input shape: `(B, L, 5)` - 5 features from dataset
- After appending: `(B, L, 6)` - 6 features (5 original + 1 given indicator)

### 2. **Curriculum Learning Behavior**

The amount of "given" information increases during training:

```python
Epoch 0-9:   All positions marked as 0 (not given)
Epoch 10:    Positions <= show_trg_max_idx marked as 1 (given)
Epoch 11+:   In "random" mode, randomly varies how much is shown
```

### 3. **What Gets Appended**

```python
# Position 0-50: given (value shown to decoder)
# Position 51-100: not given (value zeroed out)

given_indicator:
[1.0, 1.0, 1.0, ..., 1.0,  # positions 0-50
 0.0, 0.0, 0.0, ..., 0.0]  # positions 51-100
```

## Configuration Requirements

### Decoder Embedding Config

Your decoder embedding configuration **must** include an embedding for the appended "given" feature at index `N+1` (where N is the last dataset feature index).

**Example Configuration:**

```yaml
model:
  ds_embed_dec:
    setting:
      d_model: 128
      d_time: 32
      d_value: 1
    modules:
      # Original dataset features (indices 0-4)
      - idx: 0
        embed: "value"
        label: "value"
        kwargs: null
      
      - idx: 1
        embed: "nn_embedding"
        label: "variable"
        kwargs:
          num_embeddings: 50
          embedding_dim: 64
      
      - idx: 2
        embed: "nn_embedding"
        label: "position"
        kwargs:
          num_embeddings: 200
          embedding_dim: 64
      
      - idx: 3
        embed: "time2vec"
        label: "time"
        kwargs:
          kernel_size: 32
      
      - idx: 4
        embed: "identity"
        label: "mask"
        kwargs: null
      
      # NEW: Given indicator (appended by forecaster at idx 5)
      - idx: 5  # N+1, where N=4 (last dataset feature index)
        embed: "nn_embedding"
        label: "given"
        kwargs:
          num_embeddings: 2  # 0 or 1
          embedding_dim: 64
          sparse_grad: false  # optional
```

### Key Points

1. **Index Calculation**: If your dataset has features 0 through N, the given indicator will be at index N+1
2. **num_embeddings**: Must be `2` (for values 0 and 1)
3. **embedding_dim**: Should match your architectural design (typically same as other categorical embeddings)
4. **Label**: Use `"given"` for clarity

## Training Configuration

```yaml
training:
  target_show_mode: "random"  # or "fixed"
  epoch_show_trg: 10  # Start showing targets at epoch 10
  show_trg_max_idx: 50  # For "fixed" mode
  show_trg_upper_bound_max: 100  # For "random" mode
```

### Modes

**Fixed Mode (`target_show_mode: "fixed"`)**:
- Always shows targets up to `show_trg_max_idx`
- Consistent behavior across epochs

**Random Mode (`target_show_mode: "random"`)**:
- Randomly varies `show_trg_max_idx` between 0 and `show_trg_upper_bound_max`
- More diverse training signal
- Better generalization

## Implementation Details

### Code Flow

```python
# In OnlineTargetForecaster.forward()

# 1. Create given indicator
if show_trg_max_idx > 0:
    # Positions <= show_trg_max_idx are GIVEN
    given_indicator = (positions <= show_trg_max_idx).float()
else:
    # Nothing is given
    given_indicator = torch.zeros(...)

# 2. Append to decoder input
dec_input = torch.cat([dec_input, given_indicator], dim=-1)

# 3. Pass to model (embedding system handles it automatically)
model_output = self.model.forward(..., target_tensor=dec_input, ...)
```

### Embedding System Handling

The `ModularEmbedding` class automatically:
1. Recognizes the appended feature at index N+1
2. Looks up the embedding configuration for that index
3. Embeds it using the specified embedding type (nn_embedding)
4. Combines it with other embeddings according to `comps` strategy

## Example Use Cases

### 1. **Online Prediction**
Simulate real-time prediction where only past values are known:
```yaml
training:
  target_show_mode: "fixed"
  epoch_show_trg: 0  # Start immediately
  show_trg_max_idx: 0  # Only show first position
```

### 2. **Progressive Learning**
Gradually increase difficulty:
```yaml
training:
  target_show_mode: "random"
  epoch_show_trg: 10
  show_trg_upper_bound_max: 100  # Varies 0-100
```

### 3. **Partial Observation**
Train on partially observed sequences:
```yaml
training:
  target_show_mode: "fixed"
  epoch_show_trg: 5
  show_trg_max_idx: 50  # Always show first 50 positions
```

## Troubleshooting

### Issue: Index Mismatch Error

**Error**: `AssertionError: Invalid embedding selected!` or index out of bounds

**Solution**: Ensure your decoder embedding config has an entry for index N+1:
- Count your dataset features (0, 1, 2, ..., N)
- Add given embedding at index N+1

### Issue: Unexpected Tensor Shape

**Error**: Shape mismatch in embedding

**Solution**: 
- Check that `num_embeddings: 2` (not more, not less)
- Verify the given indicator is being created correctly (values are 0.0 or 1.0)

### Issue: Given Indicator Not Learning

**Symptom**: Model ignores given/not-given information

**Solutions**:
1. Increase `embedding_dim` for given indicator
2. Check that `epoch_show_trg` is set appropriately
3. Verify random mode is varying the amount shown
4. Ensure the given embedding is being combined properly (check `comps` mode)

### Issue: Values Don't Match Expected

**Symptom**: Given indicator has values other than 0.0 or 1.0

**Check**: 
- The `given_indicator` is created as float: `.float()`
- No operations are modifying it after creation

## Summary

The `OnlineTargetForecaster` automatically handles the "given" feature:

✅ **Automatic**: No manual feature engineering required
✅ **Curriculum Learning**: Progressive difficulty increase
✅ **Flexible**: Works with any decoder embedding configuration
✅ **Transparent**: Just append embedding config at index N+1

**Key Requirement**: Your decoder embedding config must include an `nn_embedding` at index N+1 with `num_embeddings: 2`.
