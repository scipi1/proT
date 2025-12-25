# Plot Predictions Functions - Usage Guide

This document explains how to use the plotting functions in `plot_predictions_fixed.py`.

## Functions Available

### 1. `plot_predictions()` - Original Fixed Version
The original function that plots all variables for a sample, with the bug fix for metric computation.

### 2. `plot_predictions_enhanced()` - New Enhanced Version
Enhanced version with:
- Variable selection (plot only specific variable)
- MAE distribution inset (shows where sample falls in test set distribution)

## Usage Examples

### Basic Usage - Plot All Variables

```python
from plot_predictions_fixed import plot_predictions

plot_predictions(
    sample_id=idx_med,
    var_index=config.data.features.Y.variable,
    x_index=config.data.features.Y.position,
    val_index=config.data.features.Y.value,
    target_array=results.targets,
    output_array=results.outputs
)
```

### Enhanced - Select Specific Variable

```python
from plot_predictions_fixed import plot_predictions_enhanced

# Plot only variable ID 1.0 (or whatever your variable IDs are)
plot_predictions_enhanced(
    sample_id=idx_med,
    var_index=config.data.features.Y.variable,
    x_index=config.data.features.Y.position,
    val_index=config.data.features.Y.value,
    target_array=results.targets,
    output_array=results.outputs,
    selected_var=1.0  # Only plot this variable
)
```

### Enhanced - With MAE Distribution Inset

```python
from plot_predictions_fixed import plot_predictions_enhanced

# Assuming you have computed df_metrics with MAE values
plot_predictions_enhanced(
    sample_id=idx_med,
    var_index=config.data.features.Y.variable,
    x_index=config.data.features.Y.position,
    val_index=config.data.features.Y.value,
    target_array=results.targets,
    output_array=results.outputs,
    selected_var=1.0,
    mae_df=df_metrics,  # DataFrame with columns ['index', 'MAE']
    inset_position='upper right'  # Options: 'upper right', 'upper left', 'lower right', 'lower left'
)
```

### Complete Example with All Options

```python
from plot_predictions_fixed import plot_predictions_enhanced

# Complete example with all optional parameters
plot_predictions_enhanced(
    sample_id=sorted_idx_df[0],
    var_index=config.data.features.Y.variable,
    x_index=config.data.features.Y.position,
    val_index=config.data.features.Y.value,
    target_array=results.targets,
    output_array=results.outputs,
    selected_var=1.0,
    mae_df=df_metrics,
    inset_position='upper left',
    title_map={1.0: "Variable 1", 2.0: "Variable 2"},  # Custom titles
    save_dir="./output/figures",
    tag="analysis_v1"
)
```

## Parameters

### Required Parameters
- `sample_id`: Index of the sample to plot
- `var_index`: Index in target_array for variable ID
- `x_index`: Index in target_array for x-axis (position)
- `val_index`: Index in target_array for values
- `target_array`: Target data (numpy array)
- `output_array`: Model predictions (numpy array)

### Optional Parameters (Enhanced Version)
- `selected_var`: Variable ID to plot (if None, plots all)
- `mae_df`: DataFrame with MAE distribution data
  - Must have columns: `['index', 'MAE']`
  - `index` should match your sample indices
- `inset_position`: Position for MAE KDE plot
  - Options: `'upper right'`, `'upper left'`, `'lower right'`, `'lower left'`
  - Default: `'upper right'`
- `title_map`: Dictionary mapping variable IDs to custom titles
- `save_dir`: Directory to save figure (if provided)
- `tag`: Tag to add to filename

## MAE Distribution DataFrame Format

The `mae_df` parameter expects a pandas DataFrame with at least these columns:

```python
# Example structure
mae_df = pd.DataFrame({
    'index': [0, 1, 2, 3, ...],  # Sample indices
    'MAE': [0.15, 0.23, 0.11, ...],  # MAE values
    # ... other columns are ignored
})
```

You can use the DataFrame returned by `compute_prediction_metrics()`:

```python
from proT.evaluation.metrics import compute_prediction_metrics

df_metrics = compute_prediction_metrics(
    results, 
    target_feature_idx=config.data.features.Y.value
)

# df_metrics already has 'index' and 'MAE' columns
plot_predictions_enhanced(..., mae_df=df_metrics)
```

## Output Files

When `save_dir` is provided, files are saved with this naming convention:
- Basic: `prediction_sample_{sample_id}_{tag}.pdf`
- With selected_var: `prediction_sample_{sample_id}_var{selected_var}_{tag}.pdf`

## Notes

- Both functions compute metrics on **normalized data** (matching `compute_prediction_metrics`)
- Plots show **de-normalized data** (multiplied by 10 for visualization)
- MAE inset is minimal and unobtrusive
- Variable selection uses variable **ID** (the actual value), not the array index
