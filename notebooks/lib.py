"""
Fixed version of plot_predictions function.

The bug was: metrics were computed on de-normalized data (multiplied by 10),
while compute_prediction_metrics uses normalized data.
This caused a 10x difference in MAE values.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import makedirs
from os.path import join
from torchmetrics.regression import R2Score, MeanSquaredError, MeanAbsoluteError


@torch.no_grad()
def eval_df_torchmetrics(
    y_pred: torch.Tensor,          # (B, S, D)
    y_true: torch.Tensor,          # (B, S, D)
    mask: torch.Tensor | None = None,  # (B, S) bool, True = valid
    feature_names: list[str] | None = None,
):
    import pandas as pd
    
    if not isinstance(y_pred, torch.Tensor):
        print("Converted output to torch tensor")
        y_pred = torch.tensor(y_pred)
        
    if not isinstance(y_true, torch.Tensor):
        print("Converted target to torch tensor")
        y_true = torch.tensor(y_true)
    
    if y_pred.shape != y_true.shape or y_true.ndim != 3:
        print(f"Found output shape {y_pred.shape} and target shape {y_true.shape}")
        raise ValueError("Expected matching (B,S,D) tensors.")
    B, S, D = y_true.shape
    device = y_true.device

    if feature_names is not None and len(feature_names) != D:
        raise ValueError(f"feature_names length {len(feature_names)} must equal D={D}")
    
    if D > 1:
        labels = feature_names or [f"f{d}" for d in range(D)]
    else:
        labels = None

    # Per-sample metrics (over S), returned per-feature (D)
    r2_metric  = lambda: R2Score(multioutput="raw_values").to(device)
    mse_metric = lambda: MeanSquaredError().to(device)
    mae_metric = lambda: MeanAbsoluteError().to(device)

    rows = []
    has_mask = mask is not None
    if has_mask:
        if mask.shape != (B, S):
            raise ValueError(f"mask must be (B,S); got {mask.shape}")

    for b in range(B):
        idx = mask[b] if has_mask else slice(None)
        yt = y_true[b, idx, :]   # (S_b, D)
        yp = y_pred[b, idx, :]   # (S_b, D)

        if yt.numel() == 0:  # no valid steps → fill zeros
            r2 = torch.zeros(D, device=device)
            mse = torch.zeros(D, device=device)
            mae = torch.zeros(D, device=device)
        else:
            r2  = r2_metric()(yp, yt)   # (D,)
            mse = mse_metric()(yp, yt)  # (D,)
            mae = mae_metric()(yp, yt)  # (D,)

        r2  = r2.detach().cpu().tolist()
        mse = mse.detach().cpu().tolist()
        mae = mae.detach().cpu().tolist()

        if D > 1 and labels is not None:
            for d, name in enumerate(labels):
                rows.append({"index": b, "feature": name, "R2": r2[d], "MSE": mse[d], "MAE": mae[d]})
        else:
            rows.append({"index": b, "R2": r2, "MSE": mse, "MAE": mae})

    return pd.DataFrame(rows)


def plot_predictions(
    sample_id: int, var_index: int, x_index: int, val_index: int, target_array: np.ndarray, 
    output_array: np.ndarray, title_map: dict=None, 
    save_dir: str=None, tag: str=None, plot_target: bool=True):
    """
    Plot predictions for a given sample, split by variables.
    
    Args:
        sample_id: Index of the sample to plot
        var_index: Index in target_array for variable ID
        x_index: Index in target_array for x-axis (position)
        val_index: Index in target_array for values
        target_array: Target data array (used for variable/position info even if not plotting target)
        output_array: Model predictions array
        title_map: Optional mapping from variable ID to title string
        save_dir: Optional directory to save figure
        tag: Optional tag for filename
        plot_target: If True (default), plots target alongside prediction. 
                     If False, only plots prediction (useful for external inference).
    """
    vars = np.unique(target_array[:,:,var_index])
    vars = vars[~np.isnan(vars)]
    num_vars = len(vars)
    
    fig = plt.figure(figsize=(6*num_vars, 6))
    gs = gridspec.GridSpec(1, num_vars, wspace=0.3)

    
    if len(output_array.shape) > 2:
        value = output_array[:,:,0].copy()
    else:
        value = output_array.copy()
    
    target = target_array.copy()
    
    # SAVE NORMALIZED COPIES FOR METRICS (FIX: compute metrics BEFORE de-normalizing)
    value_normalized = value.copy()
    target_normalized = target.copy()
    
    # de-normalize FOR PLOTTING ONLY
    value *= 10
    target[:, :, val_index] *= 10
    
    # get min and max value for y-axis limits
    if plot_target:
        c = np.concatenate([value[sample_id], target[sample_id, :, val_index]])
    else:
        c = value[sample_id]
    d = c[~(np.isnan(c))]
    min_val = np.min(d)
    max_val = np.max(d)
    
    for i,var in enumerate(vars):
        var_mask = target_normalized[sample_id, :, var_index] == var
        x = target[sample_id, :, x_index][var_mask]
        
        # For plotting: use de-normalized values
        y_out = value[sample_id][var_mask]
        y_trg = target[sample_id, :, val_index][var_mask]
        
        # For metrics: use NORMALIZED values (FIX!)
        y_out_norm = value_normalized[sample_id][var_mask]
        y_trg_norm = target_normalized[sample_id, :, val_index][var_mask]
        
        ax = fig.add_subplot(gs[i])
        ax.plot(x, y_out, label="prediction")
        if plot_target:
            ax.plot(x, y_trg, label="target")
        ax.set_ylabel(r"$\Delta R [\%]$")
        ax.set_xlabel("Position")
        ax.set_xlim(0,x[~np.isnan(x)][-1])
        ax.set_ylim(min_val, max_val)
        
        # Set title (with or without metrics depending on plot_target)
        title = (title_map[var]) if title_map is not None else var
        
        if plot_target:
            # Compute metrics on NORMALIZED data (matching compute_prediction_metrics)
            mask = torch.logical_not(torch.tensor(y_trg_norm).isnan())
            df_met = eval_df_torchmetrics(
                y_pred=torch.tensor(y_out_norm).unsqueeze(0).unsqueeze(-1), 
                y_true=torch.tensor(y_trg_norm).unsqueeze(0).unsqueeze(-1), 
                mask=mask.unsqueeze(0))
            mae = df_met.loc[0,"MAE"]
            r2 = df_met.loc[0,"R2"]
            ax.set_title(f"{title}: MAE={mae:.2f}, R2={r2:.2f}")
        else:
            ax.set_title(f"{title}")
        
        ax.legend()
    
    # optional export
    filename = f"prediction_sample_{sample_id}_{tag}.pdf"
    if save_dir is not None:
        makedirs(save_dir, exist_ok=True)
        out_path = join(save_dir, filename)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
    
    plt.show()


def plot_predictions_enhanced(
    sample_id: int, 
    var_index: int, 
    x_index: int, 
    val_index: int, 
    target_array: np.ndarray, 
    output_array,  # Can be np.ndarray or list of (np.ndarray, str) tuples
    selected_var=None,
    mae_df=None,
    inset_position='upper right',
    legend_position='best',
    show_metrics_in_legend=True,
    make_title: bool = False,
    title_map: dict=None, 
    save_dir: str=None, 
    tag: str=None):
    """
    Enhanced version of plot_predictions with variable selection and MAE distribution inset.
    
    Args:
        sample_id: Index of the sample to plot
        var_index: Index in target_array for variable ID
        x_index: Index in target_array for x-axis (position)
        val_index: Index in target_array for values
        target_array: Target data array
        output_array: Either:
                      - Single np.ndarray for one model's predictions
                      - List of (np.ndarray, str) tuples for multiple models: [(output1, "Model A"), (output2, "Model B")]
        selected_var: Optional. If provided, plots only this variable ID. If None, plots all variables.
        mae_df: Optional. DataFrame with MAE distribution. Must have columns ['index', 'MAE'].
                If provided, adds a KDE inset showing where this sample falls in the distribution.
        inset_position: Position for MAE inset. Options: 'upper right', 'upper left', 'lower right', 'lower left'
        legend_position: Position for legend. Options: 'best', 'upper right', 'upper left', 'lower right', 'lower left', etc.
        show_metrics_in_legend: If True (default), shows MAE and R² in legend labels when multiple models are provided.
        title_map: Optional mapping from variable ID to title string
        save_dir: Optional directory to save figure
        tag: Optional tag for filename
    """
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Get all available variables
    all_vars = np.unique(target_array[:,:,var_index])
    all_vars = all_vars[~np.isnan(all_vars)]
    
    # Filter to selected variable if provided
    if selected_var is not None:
        if selected_var not in all_vars:
            raise ValueError(f"selected_var {selected_var} not found in available variables: {all_vars}")
        vars = np.array([selected_var])
    else:
        vars = all_vars
    
    num_vars = len(vars)
    
    fig = plt.figure(figsize=(6*num_vars, 6))
    gs = gridspec.GridSpec(1, num_vars, wspace=0.3)

    # Parse output_array: can be single array or list of (array, label) tuples
    if isinstance(output_array, list) and len(output_array) > 0 and isinstance(output_array[0], tuple):
        # Multiple models: list of (array, label) tuples
        model_outputs = []
        for arr, label in output_array:
            if len(arr.shape) > 2:
                model_outputs.append((arr[:,:,0].copy(), label))
            else:
                model_outputs.append((arr.copy(), label))
    else:
        # Single model: wrap in list for uniform handling
        if len(output_array.shape) > 2:
            value = output_array[:,:,0].copy()
        else:
            value = output_array.copy()
        model_outputs = [(value, "prediction")]
    
    target = target_array.copy()
    
    # SAVE NORMALIZED COPIES FOR METRICS (FIX: compute metrics BEFORE de-normalizing)
    model_outputs_normalized = [(arr.copy(), label) for arr, label in model_outputs]
    target_normalized = target.copy()
    
    # de-normalize FOR PLOTTING ONLY
    model_outputs = [(arr * 10, label) for arr, label in model_outputs]
    target[:, :, val_index] *= 10
    
    # get min and max value across all models and target
    all_values = [target[sample_id, :, val_index]]
    for arr, _ in model_outputs:
        all_values.append(arr[sample_id])
    c = np.concatenate(all_values)
    d = c[~(np.isnan(c))]
    min_val = np.min(d)
    max_val = np.max(d)
    
    for i, var in enumerate(vars):
        var_mask = target_normalized[sample_id, :, var_index] == var
        x = target[sample_id, :, x_index][var_mask]
        
        # For plotting: use de-normalized target
        y_trg = target[sample_id, :, val_index][var_mask]
        y_trg_norm = target_normalized[sample_id, :, val_index][var_mask]
        
        ax = fig.add_subplot(gs[i])
        
        # Plot target first
        ax.plot(x, y_trg, label="target", color='black', linewidth=1)
        
        # Check if multiple models and metrics should be shown
        multiple_models = len(model_outputs) > 1
        mask = torch.logical_not(torch.tensor(y_trg_norm).isnan())
        
        # Plot all model predictions
        for idx, (arr, label) in enumerate(model_outputs):
            y_out = arr[sample_id][var_mask]
            
            # Add metrics to legend if requested and multiple models
            if show_metrics_in_legend and multiple_models:
                arr_norm, _ = model_outputs_normalized[idx]
                y_out_norm = arr_norm[sample_id][var_mask]
                
                df_met = eval_df_torchmetrics(
                    y_pred=torch.tensor(y_out_norm).unsqueeze(0).unsqueeze(-1), 
                    y_true=torch.tensor(y_trg_norm).unsqueeze(0).unsqueeze(-1), 
                    mask=mask.unsqueeze(0))
                mae = df_met.loc[0, "MAE"]
                r2 = df_met.loc[0, "R2"]
                label = f"{label} (MAE={mae:.2f}, R²={r2:.2f})"
            
            ax.plot(x, y_out, label=label)
        
        ax.set_ylabel(r"$\Delta R [\%]$")
        ax.set_xlabel("Position")
        ax.set_xlim(0, x[~np.isnan(x)][-1])
        
        # Hide y-axis ticks/scale if single variable is selected
        if selected_var is None:
            ax.set_ylim(min_val, max_val)
        
        # Compute title metrics (for single model or when not in legend)
        if (not multiple_models or not show_metrics_in_legend) and make_title:
            arr_norm, _ = model_outputs_normalized[0]
            y_out_norm = arr_norm[sample_id][var_mask]
            
            df_met = eval_df_torchmetrics(
                y_pred=torch.tensor(y_out_norm).unsqueeze(0).unsqueeze(-1), 
                y_true=torch.tensor(y_trg_norm).unsqueeze(0).unsqueeze(-1), 
                mask=mask.unsqueeze(0))
            mae = df_met.loc[0, "MAE"]
            r2 = df_met.loc[0, "R2"]
            
            title = (title_map[var]) if title_map is not None else var
            ax.set_title(f"{title}: MAE={mae:.2f}, R2={r2:.2f}")
        
        
        ax.legend(frameon=False, loc=legend_position)
        
        # Add MAE distribution inset if mae_df is provided
        if mae_df is not None:
            # Determine inset location
            loc_map = {
                'upper right': 1,
                'upper left': 2,
                'lower left': 3,
                'lower right': 4
            }
            loc_code = loc_map.get(inset_position.lower(), 1)
            
            # Create inset axes (30% width, 25% height)
            ax_inset = inset_axes(ax, width="30%", height="25%", loc=loc_code, 
                                 borderpad=2.5)
            
            # Get sample MAE
            sample_mae = mae_df.loc[mae_df['index'] == sample_id, 'MAE'].values
            if len(sample_mae) == 0:
                print(f"Warning: sample_id {sample_id} not found in mae_df")
                sample_mae_val = mae  # Use computed MAE as fallback
            else:
                sample_mae_val = sample_mae[0]
            
            # Plot KDE
            # sns.kdeplot(
            #     data=mae_df,
            #     x="MAE",
            #     color="black",
            #     #fill=True,
            #     #alpha=0.1,
            #     linewidth=1,
            #     bw_adjust=0.5,
            #     ax=ax_inset
            # )
            
            sns.histplot(
                data=mae_df,
                x="MAE",
                #kde=True,
                color="gray",
                edgecolor=None,
                alpha=0.7,
                linewidth=0,
                ax=ax_inset
            )
            
            # Add vertical line for current sample (solid line)
            ax_inset.axvline(x=sample_mae_val, color="black", linewidth=.75)
            
            # Minimal styling
            ax_inset.set_xlabel("MAE")
            ax_inset.set_ylabel("")
            ax_inset.set_yscale("log")
            ax_inset.set_yticks([])
            ax_inset.set_xticklabels([])  # Remove x-axis labels
            ax_inset.spines['top'].set_visible(False)
            ax_inset.spines['right'].set_visible(False)
            ax_inset.spines['left'].set_visible(False)
    
    # optional export
    tag_str = f"_{tag}" if tag is not None else ""
    var_str = f"_var{selected_var}" if selected_var is not None else ""
    filename = f"prediction_sample_{sample_id}{var_str}{tag_str}.pdf"
    if save_dir is not None:
        makedirs(save_dir, exist_ok=True)
        out_path = join(save_dir, filename)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
    
    plt.show()


def plot_predictions_adaptive(
    sample_id: int, 
    var_index: int, 
    x_index: int, 
    val_index: int, 
    target_array: np.ndarray, 
    output_array,  # Shape: (B, S, 2) - index 0: curve, index 1: failure signal
    selected_var=None,
    mae_df=None,
    inset_position='upper right',
    legend_position='best',
    show_metrics_in_legend=True,
    make_title: bool = False,
    title_map: dict=None, 
    save_dir: str=None, 
    tag: str=None,
    show_failure: bool = True,  # Toggle failure region visualization
    show_trg_max_idx: int | None = None,  # If provided, show grey region for x < show_trg_max_idx
    failure_color: str = 'red',  # Color for failure regions
    failure_alpha: float = 0.2,  # Transparency for failure regions
    trg_reveal_color: str = 'grey',  # Color for target revelation region
    trg_reveal_alpha: float = 0.15  # Transparency for target revelation region
):
    """
    Adaptive version of plot_predictions_enhanced for plotting results from predict_test_from_ckpt_adaptive.
    
    This function handles outputs with two features:
    - Feature 0: Main prediction curve (same as in plot_predictions_enhanced)
    - Feature 1: Failure signal (converted to binary: signal > 0 indicates failure)
    
    Args:
        sample_id: Index of the sample to plot
        var_index: Index in target_array for variable ID
        x_index: Index in target_array for x-axis (position)
        val_index: Index in target_array for values
        target_array: Target data array (single feature, same as plot_predictions_enhanced)
        output_array: Either:
                      - Single np.ndarray with shape (B, S, 2) where last dim is [curve, failure_signal]
                      - List of (np.ndarray, str) tuples for multiple models: [(output1, "Model A"), ...]
        selected_var: Optional. If provided, plots only this variable ID. If None, plots all variables.
        mae_df: Optional. DataFrame with MAE distribution. Must have columns ['index', 'MAE'].
        inset_position: Position for MAE inset. Options: 'upper right', 'upper left', 'lower right', 'lower left'
        legend_position: Position for legend. Options: 'best', 'upper right', etc.
        show_metrics_in_legend: If True (default), shows MAE and R² in legend labels when multiple models provided.
        make_title: If True, add title with metrics
        title_map: Optional mapping from variable ID to title string
        save_dir: Optional directory to save figure
        tag: Optional tag for filename
        show_failure: If True (default), visualize failure regions with background coloring
        show_trg_max_idx: If provided, show grey region for x-axis values < show_trg_max_idx
        failure_color: Color for failure region background (default: 'red')
        failure_alpha: Transparency for failure regions (default: 0.2)
        trg_reveal_color: Color for target revelation region (default: 'grey')
        trg_reveal_alpha: Transparency for target revelation region (default: 0.15)
    """
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # === Input Validation ===
    # Check if output_array has 2 features in last dimension
    if isinstance(output_array, list) and len(output_array) > 0 and isinstance(output_array[0], tuple):
        # Multiple models case
        for arr, label in output_array:
            if len(arr.shape) < 3 or arr.shape[-1] != 2:
                raise ValueError(
                    f"Model '{label}': output_array must have shape (B, S, 2) for adaptive plotting. "
                    f"Got shape {arr.shape}. Last dimension should be [curve, failure_signal]."
                )
    else:
        # Single model case
        if len(output_array.shape) < 3 or output_array.shape[-1] != 2:
            raise ValueError(
                f"output_array must have shape (B, S, 2) for adaptive plotting. "
                f"Got shape {output_array.shape}. Last dimension should be [curve, failure_signal]."
            )
    
    # Get all available variables
    all_vars = np.unique(target_array[:,:,var_index])
    all_vars = all_vars[~np.isnan(all_vars)]
    
    # Filter to selected variable if provided
    if selected_var is not None:
        if selected_var not in all_vars:
            raise ValueError(f"selected_var {selected_var} not found in available variables: {all_vars}")
        vars = np.array([selected_var])
    else:
        vars = all_vars
    
    num_vars = len(vars)
    
    fig = plt.figure(figsize=(6*num_vars, 6))
    gs = gridspec.GridSpec(1, num_vars, wspace=0.3)

    # === Parse output_array ===
    # Extract curve (index 0) and failure signal (index 1)
    if isinstance(output_array, list) and len(output_array) > 0 and isinstance(output_array[0], tuple):
        # Multiple models: list of (array, label) tuples
        model_outputs = []
        model_failure_signals = []
        for arr, label in output_array:
            model_outputs.append((arr[:,:,0].copy(), label))  # Main curve
            model_failure_signals.append((arr[:,:,1].copy(), label))  # Failure signal
    else:
        # Single model: wrap in list for uniform handling
        model_outputs = [(output_array[:,:,0].copy(), "prediction")]
        model_failure_signals = [(output_array[:,:,1].copy(), "prediction")]
    
    target = target_array.copy()
    
    # SAVE NORMALIZED COPIES FOR METRICS (compute metrics BEFORE de-normalizing)
    model_outputs_normalized = [(arr.copy(), label) for arr, label in model_outputs]
    target_normalized = target.copy()
    
    # de-normalize FOR PLOTTING ONLY
    model_outputs = [(arr * 10, label) for arr, label in model_outputs]
    target[:, :, val_index] *= 10
    
    # get min and max value across all models and target
    all_values = [target[sample_id, :, val_index]]
    for arr, _ in model_outputs:
        all_values.append(arr[sample_id])
    c = np.concatenate(all_values)
    d = c[~(np.isnan(c))]
    min_val = np.min(d)
    max_val = np.max(d)
    
    for i, var in enumerate(vars):
        var_mask = target_normalized[sample_id, :, var_index] == var
        x = target[sample_id, :, x_index][var_mask]
        
        # For plotting: use de-normalized target
        y_trg = target[sample_id, :, val_index][var_mask]
        y_trg_norm = target_normalized[sample_id, :, val_index][var_mask]
        
        ax = fig.add_subplot(gs[i])
        
        # === Add target revelation region (lowest z-order) ===
        if show_trg_max_idx is not None:
            # Find x-value corresponding to show_trg_max_idx
            if show_trg_max_idx < len(x):
                x_threshold = x[show_trg_max_idx]
                ax.axvspan(0, x_threshold, color=trg_reveal_color, alpha=trg_reveal_alpha, zorder=-2)
                # Optional: add vertical line at boundary
                ax.axvline(x_threshold, color=trg_reveal_color, linestyle='--', alpha=0.5, linewidth=1, zorder=-1)
        
        # === Add failure regions (second-lowest z-order) ===
        if show_failure and len(model_failure_signals) > 0:
            # Use failure signal from first model
            failure_signal, _ = model_failure_signals[0]
            failure_signal_sample = failure_signal[sample_id][var_mask]
            
            # Convert to boolean: failure occurs where signal > 0
            failure_mask = failure_signal_sample > 0
            
            # Find contiguous failure regions
            # diff will be 1 at start of failure region, -1 at end
            failure_changes = np.diff(np.concatenate(([False], failure_mask, [False])).astype(int))
            failure_starts = np.where(failure_changes == 1)[0]
            failure_ends = np.where(failure_changes == -1)[0]
            
            # Plot each failure region
            for start_idx, end_idx in zip(failure_starts, failure_ends):
                if start_idx < len(x) and end_idx <= len(x):
                    x_start = x[start_idx]
                    # Handle edge case where end_idx equals length
                    x_end = x[end_idx - 1] if end_idx < len(x) else x[-1]
                    ax.axvspan(x_start, x_end, color=failure_color, alpha=failure_alpha, zorder=-1)
        
        # === Plot curves ===
        # Plot target first
        ax.plot(x, y_trg, label="target", color='black', linewidth=1)
        
        # Check if multiple models and metrics should be shown
        multiple_models = len(model_outputs) > 1
        mask = torch.logical_not(torch.tensor(y_trg_norm).isnan())
        
        # Plot all model predictions
        for idx, (arr, label) in enumerate(model_outputs):
            y_out = arr[sample_id][var_mask]
            
            # Add metrics to legend if requested and multiple models
            if show_metrics_in_legend and multiple_models:
                arr_norm, _ = model_outputs_normalized[idx]
                y_out_norm = arr_norm[sample_id][var_mask]
                
                df_met = eval_df_torchmetrics(
                    y_pred=torch.tensor(y_out_norm).unsqueeze(0).unsqueeze(-1), 
                    y_true=torch.tensor(y_trg_norm).unsqueeze(0).unsqueeze(-1), 
                    mask=mask.unsqueeze(0))
                mae = df_met.loc[0, "MAE"]
                r2 = df_met.loc[0, "R2"]
                label = f"{label} (MAE={mae:.2f}, R²={r2:.2f})"
            
            ax.plot(x, y_out, label=label)
        
        ax.set_ylabel(r"$\Delta R [\%]$")
        ax.set_xlabel("Position")
        ax.set_xlim(0, x[~np.isnan(x)][-1])
        
        # Set y-limits
        if selected_var is None:
            ax.set_ylim(min_val, max_val)
        
        # Compute title metrics (for single model or when not in legend)
        if (not multiple_models or not show_metrics_in_legend) and make_title:
            arr_norm, _ = model_outputs_normalized[0]
            y_out_norm = arr_norm[sample_id][var_mask]
            
            df_met = eval_df_torchmetrics(
                y_pred=torch.tensor(y_out_norm).unsqueeze(0).unsqueeze(-1), 
                y_true=torch.tensor(y_trg_norm).unsqueeze(0).unsqueeze(-1), 
                mask=mask.unsqueeze(0))
            mae = df_met.loc[0, "MAE"]
            r2 = df_met.loc[0, "R2"]
            
            title = (title_map[var]) if title_map is not None else var
            ax.set_title(f"{title}: MAE={mae:.2f}, R2={r2:.2f}")
        
        ax.legend(frameon=False, loc=legend_position)
        
        # === Add MAE distribution inset if mae_df is provided ===
        if mae_df is not None:
            # Determine inset location
            loc_map = {
                'upper right': 1,
                'upper left': 2,
                'lower left': 3,
                'lower right': 4
            }
            loc_code = loc_map.get(inset_position.lower(), 1)
            
            # Create inset axes (30% width, 25% height)
            ax_inset = inset_axes(ax, width="30%", height="25%", loc=loc_code, 
                                 borderpad=2.5)
            
            # Get sample MAE
            sample_mae = mae_df.loc[mae_df['index'] == sample_id, 'MAE'].values
            if len(sample_mae) == 0:
                print(f"Warning: sample_id {sample_id} not found in mae_df")
                # Use computed MAE as fallback
                arr_norm, _ = model_outputs_normalized[0]
                y_out_norm = arr_norm[sample_id][var_mask]
                df_met = eval_df_torchmetrics(
                    y_pred=torch.tensor(y_out_norm).unsqueeze(0).unsqueeze(-1), 
                    y_true=torch.tensor(y_trg_norm).unsqueeze(0).unsqueeze(-1), 
                    mask=mask.unsqueeze(0))
                sample_mae_val = df_met.loc[0, "MAE"]
            else:
                sample_mae_val = sample_mae[0]
            
            # Plot histogram
            sns.histplot(
                data=mae_df,
                x="MAE",
                color="gray",
                edgecolor=None,
                alpha=0.7,
                linewidth=0,
                ax=ax_inset
            )
            
            # Add vertical line for current sample
            ax_inset.axvline(x=sample_mae_val, color="black", linewidth=.75)
            
            # Minimal styling
            ax_inset.set_xlabel("MAE")
            ax_inset.set_ylabel("")
            ax_inset.set_yticks([])
            ax_inset.set_xticklabels([])
            ax_inset.spines['top'].set_visible(False)
            ax_inset.spines['right'].set_visible(False)
            ax_inset.spines['left'].set_visible(False)
    
    # === Optional export ===
    tag_str = f"_{tag}" if tag is not None else ""
    var_str = f"_var{selected_var}" if selected_var is not None else ""
    filename = f"prediction_adaptive_sample_{sample_id}{var_str}{tag_str}.pdf"
    if save_dir is not None:
        makedirs(save_dir, exist_ok=True)
        out_path = join(save_dir, filename)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
    
    plt.show()


def plot_attention_heatmap(
    sample_id: int,
    cross_attention: np.ndarray,
    input_array: np.ndarray,
    val_index: int,
    title: str = "Cross-Attention",
    sum_heads: bool = True,
    cmap: str = 'viridis',
    save_dir: str = None,
    tag: str = None
):
    """
    Plot cross-attention heatmap for a single sample with input missing mask.
    
    Creates a two-part figure:
    - Top: Main attention heatmap (decoder queries x encoder keys)
    - Bottom: Input missing values mask strip
    
    Args:
        sample_id: Index of the sample to plot
        cross_attention: Attention weights array from results.attention_weights['cross']
                        Shape can be (B, N, M) or (B, H, N, M) for multi-head
        input_array: Input data array (B, L, F) for computing missing mask
        val_index: Feature index to check for missing values (NaN)
        title: Title for the plot
        sum_heads: If True and multi-head attention, sum across heads. 
                   If False, uses first head only.
        cmap: Colormap for attention heatmap (default: 'viridis')
        save_dir: Optional directory to save figure
        tag: Optional tag for filename
    """
    # Handle attention array dimensions
    # Expected: (B, N, M) for single-head or (B, H, N, M) for multi-head
    att = cross_attention.copy()
    
    # Ensure at least 3D
    if len(att.shape) == 2:
        # (N, M) -> assume single sample, add batch dim
        att = att[np.newaxis, :, :]
    
    # Extract sample
    if len(att.shape) == 4:
        # Multi-head: (B, H, N, M)
        if sum_heads:
            # Sum across head dimension
            att_sample = att[sample_id].sum(axis=0)  # (N, M)
            title_suffix = " (Summed Across Heads)"
        else:
            # Use first head
            att_sample = att[sample_id, 0, :, :]  # (N, M)
            title_suffix = " (Head 0)"
    elif len(att.shape) == 3:
        # Single head or pre-aggregated: (B, N, M)
        att_sample = att[sample_id]  # (N, M)
        title_suffix = ""
    else:
        raise ValueError(f"Unexpected attention shape: {att.shape}. Expected 3D or 4D array.")
    
    N, M = att_sample.shape
    
    # Compute input missing mask
    # input_array shape: (B, L, F)
    input_sample = input_array[sample_id, :, val_index]  # (L,)
    input_miss_bool = np.isnan(input_sample)
    input_miss = input_miss_bool[np.newaxis, :].astype(int)  # (1, M)
    
    # Create figure with gridspec (10:1 height ratio)
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.3)
    
    # === Main Attention Heatmap ===
    ax0 = fig.add_subplot(gs[0])
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    
    im0 = ax0.imshow(att_sample, cmap=cmap, aspect='auto', origin='upper')
    fig.colorbar(im0, cax=cax0, label='Attention Weight')
    
    ax0.set_xticks([])  # Hide x-ticks (shared with mask below)
    ax0.set_ylabel("Output Position (Decoder)")
    ax0.set_title(f"{title}{title_suffix}")
    
    # === Input Missing Mask ===
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    
    im1 = ax1.imshow(input_miss, cmap='Greys', aspect='auto', origin='upper', vmin=0, vmax=1)
    cbar1 = fig.colorbar(im1, cax=cax1, ticks=[0, 1])
    cbar1.ax.set_yticklabels(['Valid', 'Missing'])
    
    ax1.set_yticks([])
    ax1.set_xlabel("Input Position (Encoder)")
    
    # Set x-ticks
    num_labels = min(M, 10)
    step = max(M // num_labels, 1)
    ax1.set_xticks(np.arange(0, M, step))
    ax1.set_xticklabels(np.arange(0, M, step))
    
    plt.tight_layout()
    
    # Optional export
    if save_dir is not None:
        makedirs(save_dir, exist_ok=True)
        tag_str = f"_{tag}" if tag is not None else ""
        filename = f"attention_sample_{sample_id}{tag_str}.pdf"
        out_path = join(save_dir, filename)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print(f"Saved attention heatmap to {out_path}")
    
    plt.show()




import re
import os
from omegaconf import OmegaConf
import json
def get_config_and_best_checkpoint_from_experiment(exp_path):
    
    # get config
    config_regex = re.compile("config")
    
    config_list = []
    for file in os.listdir(exp_path):
        
        if config_regex.match(file):
            config_list.append(file)
    
    if len(config_list) != 1:
        raise ValueError(f"More (or none) than one config found! {config_list}")
    else:
        config = OmegaConf.load(os.path.join(exp_path, config_list[0]))
    
    with open(os.path.join(exp_path, "kfold_summary.json")) as f:
        kfold_summary = json.load(f)
    
    
    best_checkpoint_path = os.path.join(
        exp_path,
        f"k_{kfold_summary["best_fold"]["fold_number"]}",
        "checkpoints",
        "best_checkpoint.ckpt"
    )
        
        
    return config, best_checkpoint_path