# Mamba NaN Issue - Root Cause and Solution

## Problem Identified

**Diagnostic Output:**
```
[BiMamba DEBUG] NaN after forward layer 2! Shape: torch.Size([50, 1023, 100])
[BiMamba DEBUG] Stats: min=-3.108360862507917e+26, max=3.9612992013788104e+26
```

**Root Cause:** Numerical explosion in Mamba layers. Values were growing exponentially through each layer (reaching 10^26 magnitude) until they overflowed to NaN after the 3rd layer.

## Why This Happened

1. **Deep Mamba stacking** (4 layers) without normalization
2. **SSM (State Space Model) accumulation** - each Mamba layer processes temporal dependencies, and without normalization, activations compound across layers
3. **No regularization** between layers to constrain activation magnitudes

## Solution Implemented

### Added Layer Normalization After Each Mamba Layer

**File Modified:** `proT/baseline/s6/s6_wrapper.py`

**Changes:**
1. Added `forward_norms` - LayerNorm modules for forward direction (one per layer)
2. Added `backward_norms` - LayerNorm modules for backward direction (one per layer)
3. Applied LayerNorm immediately after each Mamba layer, before dropout

**Code Pattern:**
```python
for i, (layer, norm) in enumerate(zip(self.forward_layers, self.forward_norms)):
    h_forward = layer(h_forward)
    h_forward = norm(h_forward)  # Normalize activations
    h_forward = self.dropout(h_forward)
```

### Why Layer Normalization Works

- **Normalizes activations** to have mean=0 and variance=1 at each layer
- **Prevents exponential growth** of activation magnitudes
- **Stabilizes deep architectures** by ensuring each layer receives normalized inputs
- **Standard practice** in transformer and deep sequential models

## Expected Behavior After Fix

With LayerNorm, you should see diagnostic output like:
```
[BiMamba DEBUG] Forward layer 0 OK - min=-2.3456, max=1.9876, mean=0.0123
[BiMamba DEBUG] Forward layer 1 OK - min=-1.8765, max=2.1234, mean=-0.0045
[BiMamba DEBUG] Forward layer 2 OK - min=-2.0123, max=1.7654, mean=0.0089
[BiMamba DEBUG] Forward layer 3 OK - min=-1.9432, max=2.0567, mean=0.0012
```

Notice values stay in reasonable ranges (around -2 to +2) instead of exploding to 10^26.

## Additional Safeguards Already in Place

From previous fixes in `baseline_pl_modules.py`:
1. **Gradient clipping** (max_norm=1.0)
2. **NaN detection** in predictions, loss, and gradients
3. **Value clamping** to prevent extreme outliers
4. **Gradient monitoring** to detect issues early

## Testing Instructions

1. Run training on cluster with the updated code
2. Check console for `[BiMamba DEBUG]` messages
3. Verify all layers show "OK" with reasonable min/max values
4. Confirm training proceeds without NaN

## If Issues Persist

If you still see NaN after this fix:

### Option 1: Reduce number of layers
```python
n_layers: 2  # Instead of 4
```

### Option 2: Add residual connections
Modify BiMamba to add skip connections:
```python
h_forward = layer(h_forward) + h_forward  # Residual
h_forward = norm(h_forward)
```

### Option 3: Reduce hidden dimension
```python
d_hidden: 32  # Instead of 64
```

## Performance Impact

Layer Normalization adds:
- **Minimal computational cost** (~2-3% overhead)
- **Improved stability** (much more important)
- **Better generalization** (often improves model performance)

## Files Modified Summary

1. `proT/baseline/s6/s6_wrapper.py` - Added LayerNorm
2. `proT/baseline/baseline_pl_modules.py` - Gradient clipping + NaN handling
3. `proT/baseline/baseline_models.py` - Diagnostic logging

## Removing Diagnostic Logs

Once confirmed working, remove diagnostic prints by searching for:
- `[BiMamba DEBUG]`
- `[S6 DEBUG]`

And deleting those print statements.
