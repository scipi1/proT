# Mamba NaN Diagnostic Guide

## What Was Added

Comprehensive diagnostic logging has been added to trace where NaN values originate in the S6/Mamba model.

### Files Modified
1. `proT/baseline/baseline_models.py` - S6 model forward pass
2. `proT/baseline/s6/s6_wrapper.py` - BiMamba forward pass

## Diagnostic Logging Points

The code now prints debug messages at every stage:

### 1. Input/Target Embeddings (in S6.forward)
```
[S6 DEBUG] NaN detected in input embeddings!
[S6 DEBUG] NaN detected in target embeddings!
```

### 2. BiMamba Processing (in BiMamba.forward)
```
[BiMamba DEBUG] NaN in input!
[BiMamba DEBUG] NaN after forward layer {i}!
[BiMamba DEBUG] NaN after backward layer {i}!
[BiMamba DEBUG] NaN in final bidirectional output!
```

### 3. Context and Output (in S6.forward)
```
[S6 DEBUG] NaN detected in BiMamba context!
[S6 DEBUG] BiMamba context OK - ... (if no NaN)
[S6 DEBUG] NaN detected after concatenation!
[S6 DEBUG] NaN detected in final output!
[S6 DEBUG] Final output OK - ... (if no NaN)
```

## How to Test

1. **Run your training on the cluster** with the modified code
2. **Check the console output** for these debug messages
3. **Identify the first occurrence** of NaN - this tells us exactly where the problem starts

## What to Look For

### Scenario 1: NaN in Embeddings
If you see:
```
[S6 DEBUG] NaN detected in input embeddings!
```
or
```
[S6 DEBUG] NaN detected in target embeddings!
```

**Issue**: Problem with ModularEmbedding or input data
**Solution**: Check embedding configuration and input data preprocessing

### Scenario 2: NaN After First Mamba Layer
If you see:
```
[BiMamba DEBUG] NaN after forward layer 0!
```

**Issue**: Mamba layer itself is producing NaN (most likely)
**Possible causes**:
- Numerical instability in SSM computation
- Bad initialization in Mamba layers
- Input dimension mismatch

### Scenario 3: NaN After Several Layers
If NaN appears after layer 2 or 3:
```
[BiMamba DEBUG] NaN after forward layer 2!
```

**Issue**: Accumulated numerical error
**Solution**: 
- Add layer normalization between Mamba layers
- Reduce number of layers
- Check if gradient clipping helps

### Scenario 4: NaN in Final Output Only
If all intermediate outputs are OK but final output has NaN:
```
[S6 DEBUG] Final output OK - ... (all values look normal)
```
Then in validation logs you see NaN.

**Issue**: Problem in loss computation or metric calculation
**Solution**: Already handled in `baseline_pl_modules.py`

## Expected DebugOutput

When everything works correctly, you should see messages like:
```
[S6 DEBUG] BiMamba context OK - Shape: torch.Size([32, 128]), min=-2.3456, max=1.9876, mean=0.0123
[S6 DEBUG] Final output OK - Shape: torch.Size([32, 50]), min=-0.5234, max=0.8765, mean=0.0432
```

## Next Steps Based on Results

**If NaN is in Mamba layers (most likely)**:
1. Check if using mamba-ssm or pytorch implementation
2. Try reducing `n_layers` from 4 to 2
3. Try reducing `d_model` (d_hidden) from 64 to 32
4. Add layer normalization after each Mamba layer
5. Initialize Mamba weights more carefully

**If NaN is NOT appearing in diagnostics but still in loss**:
- Issue is in the pl_module's `_step` method
- Already handled with NaN detection code

## Removing Diagnostics

Once the issue is fixed, you can remove the diagnostic prints by searching for `[S6 DEBUG]` and `[BiMamba DEBUG]` in the code and deleting those lines.
