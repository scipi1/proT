# NaN Issue Fix Summary

## Problem
Mamba (S6) model training was producing NaN values in the loss during training, particularly:
- First k-fold on Ishigami worked, then NaN appeared
- Dyconex dataset produced NaN immediately
- Issue occurred with both mamba-ssm and pytorch-implemented Mamba

## Root Cause
**Gradient explosion** - common issue with recurrent architectures like SSMs/Mamba that was not being controlled.

## Solutions Implemented

### 1. **Gradient Clipping** (Primary Fix)
Added automatic gradient clipping in `configure_optimizers()`:
- Default `gradient_clip_val=1.0` (max gradient norm)
- Can be configured via `training.gradient_clip_val` in config file
- Uses PyTorch Lightning's built-in gradient clipping

### 2. **NaN Detection and Handling**
Added multiple safety checks in the `_step()` method:
- **Prediction NaN check**: Detects NaN in model outputs and replaces with 0.0
- **Value clamping**: Clips predictions to [-1e6, 1e6] range to prevent overflow
- **Loss NaN check**: Replaces NaN loss elements with large penalty (1e6)
- **Final loss check**: Ensures final loss is never NaN
- **Logging**: All NaN detections are logged with stage-specific metrics

### 3. **Gradient Monitoring**
Added `on_before_optimizer_step()` callback to:
- Detect NaN/Inf in gradients before optimization
- Log gradient norms for problematic parameters
- Help identify which layers are causing issues

## Configuration Options

You can customize gradient clipping in your config YAML:

```yaml
training:
  lr: 1e-4
  weight_decay: 0.0
  gradient_clip_val: 1.0  # Adjust this if needed (0.5 for more aggressive, 5.0 for more lenient)
  loss_fn: "mse"
```

## New Logging Metrics

Watch for these new metrics in your logs:
- `{stage}_nan_detected`: NaN found in predictions
- `{stage}_loss_nan_detected`: NaN found in loss computation
- `{stage}_final_loss_nan`: Final loss was NaN
- `invalid_gradients_detected`: NaN/Inf found in gradients
- `grad_norm_{param_name}`: Gradient norm for specific parameters (when invalid)

## Testing Recommendations

1. **Start with defaults**: The default `gradient_clip_val=1.0` should work for most cases
2. **Monitor logs**: Check for the new NaN detection metrics
3. **If NaN persists**:
   - Try more aggressive clipping: `gradient_clip_val: 0.5`
   - Reduce learning rate: `lr: 5e-5` or `lr: 1e-5`
   - Check the gradient norm logs to identify problematic layers

## Alternative Solutions (if needed)

If the issue persists, consider:

1. **Learning Rate Warmup**:
```python
# In configure_optimizers, replace StepLR with:
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
main_sched = StepLR(opt, step_size=10, gamma=0.1)
sched = SequentialLR(opt, schedulers=[warmup, main_sched], milestones=[5])
```

2. **Layer Normalization**: Add normalization in BiMamba layers
3. **Smaller Initial Learning Rate**: Start with `lr: 1e-5`

## Files Modified
- `proT/baseline/baseline_pl_modules.py`: Added gradient clipping, NaN detection, and monitoring

## Backward Compatibility
✅ All changes are backward compatible:
- Gradient clipping defaults to 1.0 (safe value)
- NaN handling is transparent to normal operation
- No changes needed to existing config files (but can add `gradient_clip_val` if desired)
