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
    tag: str=None,
    ax=None,
    show_inset: bool=True,
    scale: float=10.0,
    ylabel: str=r"$\Delta R [\%]$",
    xlabel: str="Position",
    color_map: dict=None,
    selected_var2=None,
    var1_label: str="Sense A",
    var2_label: str="Sense B",
    ylim: tuple=None):


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
        selected_var2: Optional second variable ID plotted on the same axes with dashed lines
                       (same colors as the main variable). Default None.
        var1_label: Legend suffix of the main variable when selected_var2 is used (default "Sense A").
        var2_label: Legend suffix of selected_var2 (default "Sense B").
        ylim: Optional (ymin, ymax) tuple applied to the axes, e.g. to share the y scale
              across two plots. Default None (autoscale / data range).
        title_map: Optional mapping from variable ID to title string
        save_dir: Optional directory to save figure (ignored when ax is provided)
        tag: Optional tag for filename (ignored when ax is provided)
        ax: Optional matplotlib Axes to plot on. When provided, the function plots onto this axes
            instead of creating a new figure. Figure saving and plt.show() are skipped.
            Only supported when selected_var selects exactly one variable.
        show_inset: If True (default) and mae_df is provided, shows the MAE distribution inset.
                    Set to False to suppress the inset even when mae_df is given.
    
    Returns:
        ax: The matplotlib Axes object used for plotting.
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
    
    if selected_var2 is not None and selected_var2 not in all_vars:
        raise ValueError(f"selected_var2 {selected_var2} not found in available variables: {all_vars}")
    
    num_vars = len(vars)
    
    # Determine if we use an externally provided axes
    _external_ax = ax is not None
    if _external_ax:
        if num_vars != 1:
            raise ValueError("When passing an external `ax`, selected_var must select exactly one variable.")
        fig = ax.get_figure()
    else:
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
    model_outputs = [(arr * scale, label) for arr, label in model_outputs]
    target[:, :, val_index] *= scale
    
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
        
        if _external_ax:
            cur_ax = ax
        else:
            cur_ax = fig.add_subplot(gs[i])
        
        # Plot target first
        trg_label = "target" if selected_var2 is None else f"target ({var1_label})"
        cur_ax.plot(x, y_trg, label=trg_label, color='black', linewidth=1)

        
        # Check if multiple models and metrics should be shown
        multiple_models = len(model_outputs) > 1
        mask = torch.logical_not(torch.tensor(y_trg_norm).isnan())
        
        # Plot all model predictions
        model_colors = []
        model_labels = []
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
            
            # optional explicit color per model label (keys use the raw labels)
            raw_label = model_outputs_normalized[idx][1]
            color = color_map.get(raw_label) if color_map is not None else None
            model_labels.append(label)
            legend_label = f"{label} ({var1_label})" if selected_var2 is not None else label
            line, = cur_ax.plot(x, y_out, label=legend_label, color=color)
            model_colors.append(line.get_color())
        
        # Optional second variable on the same axes: same colors, dashed lines
        x_max = x[~np.isnan(x)][-1]
        if selected_var2 is not None:
            var2_mask = target_normalized[sample_id, :, var_index] == selected_var2
            x2 = target[sample_id, :, x_index][var2_mask]
            y_trg2 = target[sample_id, :, val_index][var2_mask]
            cur_ax.plot(x2, y_trg2, label=f"target ({var2_label})", color='black', linewidth=1, linestyle='--')
            for idx, (arr, _) in enumerate(model_outputs):
                cur_ax.plot(x2, arr[sample_id][var2_mask], label=f"{model_labels[idx]} ({var2_label})",
                            color=model_colors[idx], linestyle='--')
            if np.any(~np.isnan(x2)):
                x_max = max(x_max, x2[~np.isnan(x2)][-1])
        
        cur_ax.set_ylabel(ylabel)

        cur_ax.set_xlabel(xlabel)
        cur_ax.set_xlim(0, x_max)
        
        # y-axis limits: explicit input wins, otherwise autoscale (single variable) or data range
        if ylim is not None:
            cur_ax.set_ylim(ylim)
        elif selected_var is None:
            cur_ax.set_ylim(min_val, max_val)
        
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
            cur_ax.set_title(f"{title}: MAE={mae:.2f}, R2={r2:.2f}")
        
        
        cur_ax.legend(frameon=False, loc=legend_position)
        
        # Add MAE distribution inset if mae_df is provided and show_inset is True
        if mae_df is not None and show_inset:
            # Determine inset location
            loc_map = {
                'upper right': 1,
                'upper left': 2,
                'lower left': 3,
                'lower right': 4
            }
            loc_code = loc_map.get(inset_position.lower(), 1)
            
            # Create inset axes (30% width, 25% height), close to the axes edges
            ax_inset = inset_axes(cur_ax, width="30%", height="25%", loc=loc_code, 
                                 borderpad=0.8)
            
            # Get sample MAE
            sample_mae = mae_df.loc[mae_df['index'] == sample_id, 'MAE'].values
            if len(sample_mae) == 0:
                print(f"Warning: sample_id {sample_id} not found in mae_df")
                sample_mae_val = mae  # Use computed MAE as fallback
            else:
                sample_mae_val = sample_mae[0]
            
            sns.histplot(
                data=mae_df,
                x="MAE",
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
    
    # When using external ax, skip saving and showing — caller handles that
    if not _external_ax:
        # optional export
        tag_str = f"_{tag}" if tag is not None else ""
        var_str = f"_var{selected_var}" if selected_var is not None else ""
        filename = f"prediction_sample_{sample_id}{var_str}{tag_str}.pdf"
        if save_dir is not None:
            makedirs(save_dir, exist_ok=True)
            out_path = join(save_dir, filename)
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
        
        plt.show()
    
    return cur_ax


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
    trg_reveal_alpha: float = 0.15,  # Transparency for target revelation region
    ax=None,
    show_inset: bool=True
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
        save_dir: Optional directory to save figure (ignored when ax is provided)
        tag: Optional tag for filename (ignored when ax is provided)
        show_failure: If True (default), visualize failure regions with background coloring
        show_trg_max_idx: If provided, show grey region for x-axis values < show_trg_max_idx
        failure_color: Color for failure region background (default: 'red')
        failure_alpha: Transparency for failure regions (default: 0.2)
        trg_reveal_color: Color for target revelation region (default: 'grey')
        trg_reveal_alpha: Transparency for target revelation region (default: 0.15)
        ax: Optional matplotlib Axes to plot on. When provided, the function plots onto this axes
            instead of creating a new figure. Figure saving and plt.show() are skipped.
            Only supported when selected_var selects exactly one variable.
        show_inset: If True (default) and mae_df is provided, shows the MAE distribution inset.
                    Set to False to suppress the inset even when mae_df is given.
    
    Returns:
        ax: The matplotlib Axes object used for plotting.
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
    
    # Determine if we use an externally provided axes
    _external_ax = ax is not None
    if _external_ax:
        if num_vars != 1:
            raise ValueError("When passing an external `ax`, selected_var must select exactly one variable.")
        fig = ax.get_figure()
    else:
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
        
        if _external_ax:
            cur_ax = ax
        else:
            cur_ax = fig.add_subplot(gs[i])
        
        # === Add target revelation region (lowest z-order) ===
        if show_trg_max_idx is not None:
            # Find x-value corresponding to show_trg_max_idx
            if show_trg_max_idx < len(x):
                x_threshold = x[show_trg_max_idx]
                cur_ax.axvspan(0, x_threshold, color=trg_reveal_color, alpha=trg_reveal_alpha, zorder=-2)
                # Optional: add vertical line at boundary
                cur_ax.axvline(x_threshold, color=trg_reveal_color, linestyle='--', alpha=0.5, linewidth=1, zorder=-1)
        
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
                    cur_ax.axvspan(x_start, x_end, color=failure_color, alpha=failure_alpha, zorder=-1)
        
        # === Plot curves ===
        # Plot target first
        cur_ax.plot(x, y_trg, label="target", color='black', linewidth=1)
        
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
            
            cur_ax.plot(x, y_out, label=label)
        
        cur_ax.set_ylabel(r"$\Delta R [\%]$")
        cur_ax.set_xlabel("IST cycle")
        cur_ax.set_xlim(0, x[~np.isnan(x)][-1])
        
        # Set y-limits
        if selected_var is None:
            cur_ax.set_ylim(min_val, max_val)
        
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
            cur_ax.set_title(f"{title}: MAE={mae:.2f}, R2={r2:.2f}")
        
        cur_ax.legend(frameon=False, loc=legend_position)
        
        # === Add MAE distribution inset if mae_df is provided and show_inset is True ===
        if mae_df is not None and show_inset:
            # Determine inset location
            loc_map = {
                'upper right': 1,
                'upper left': 2,
                'lower left': 3,
                'lower right': 4
            }
            loc_code = loc_map.get(inset_position.lower(), 1)
            
            # Create inset axes (30% width, 25% height)
            ax_inset = inset_axes(cur_ax, width="30%", height="25%", loc=loc_code, 
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
    
    # When using external ax, skip saving and showing — caller handles that
    if not _external_ax:
        # === Optional export ===
        tag_str = f"_{tag}" if tag is not None else ""
        var_str = f"_var{selected_var}" if selected_var is not None else ""
        filename = f"prediction_adaptive_sample_{sample_id}{var_str}{tag_str}.pdf"
        if save_dir is not None:
            makedirs(save_dir, exist_ok=True)
            out_path = join(save_dir, filename)
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
        
        plt.show()
    
    return cur_ax


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


import glob


def get_config_and_best_checkpoint(exp_path):
    """
    Resolve config and best checkpoint of an optuna experiment folder.

    Expects the folder layout:
        exp_path/config_*.yaml
        exp_path/best_trial.yaml           (trial_number)
        exp_path/optuna/run_<trial>/kfold_summary.json
        exp_path/optuna/run_<trial>/k_<fold>/checkpoints/best_checkpoint.ckpt

    Returns:
        (config, best_checkpoint_path)
    """
    config_regex = re.compile("config")
    best_trial_regex = re.compile("best_trial")

    config_list = []
    summary_list = []
    for file in os.listdir(exp_path):
        if config_regex.match(file):
            config_list.append(file)
        elif best_trial_regex.match(file):
            summary_list.append(file)

    if len(config_list) != 1:
        raise ValueError(f"More (or none) than one config found! {config_list}")
    config = OmegaConf.load(os.path.join(exp_path, config_list[0]))

    if len(summary_list) != 1:
        raise ValueError(f"More (or none) than one best_trial summary found! {summary_list}")
    summary = OmegaConf.load(os.path.join(exp_path, summary_list[0]))

    optuna_dir = os.path.join(exp_path, "optuna")
    if not os.path.exists(optuna_dir):
        raise FileNotFoundError(f"Optuna folder doesn't exist in {exp_path}")

    run_dir = os.path.join(optuna_dir, f"run_{summary.trial_number}")

    with open(os.path.join(run_dir, "kfold_summary.json")) as f:
        kfold_summary = json.load(f)

    best_checkpoint_path = os.path.join(
        run_dir,
        f"k_{kfold_summary['best_fold']['fold_number']}",
        "checkpoints",
        "best_checkpoint.ckpt",
    )

    return config, best_checkpoint_path


def find_experiment_folder(baseline_dir: str, model: str, dataset: str, fusion: str = "sum"):
    """
    Find a baseline experiment folder matching model/dataset/fusion.

    The folders are named e.g. `baseline_GRU_ishigami_sum_49813798`.

    Args:
        baseline_dir: directory holding the baseline experiment folders
        model: model tag (e.g. "GRU", "LSTM", "proT")
        dataset: dataset tag (e.g. "ishigami", "dyconex")
        fusion: fusion tag (default "sum")

    Returns:
        Full path to the matching folder.
    """
    pattern = f"baseline_{model}_{dataset}_{fusion}_*"
    matches = sorted(glob.glob(os.path.join(baseline_dir, pattern)))
    if len(matches) == 0:
        raise FileNotFoundError(f"No experiment folder matching {pattern} in {baseline_dir}")
    if len(matches) > 1:
        print(f"Warning: multiple folders match {pattern}, using {matches[0]}")
    return matches[0]


def select_best_sample(metrics_dfs: dict, metric: str = "MAE", reference_models=None):
    """
    Select the test sample where a reference model performs best.

    Args:
        metrics_dfs: mapping {model_name: metrics DataFrame with columns ['index', metric]}
        metric: metric column used for the selection (lower is better)
        reference_models: optional list of model names to consider. If None, all are used.

    Returns:
        (sample_id, model_name, metric_value) of the global minimum.
    """
    best = None
    models = reference_models if reference_models is not None else list(metrics_dfs.keys())
    for model in models:
        df = metrics_dfs[model]
        row = df.loc[df[metric].idxmin()]
        if best is None or row[metric] < best[2]:
            best = (int(row["index"]), model, float(row[metric]))
    return best


def select_median_sample(metrics_df, metric: str = "MAE"):
    """
    Select the test sample whose metric is closest to the median of the distribution.

    Args:
        metrics_df: metrics DataFrame with columns ['index', metric]
        metric: metric column used for the selection

    Returns:
        (sample_id, metric_value) of the median-like sample.
    """
    median_val = metrics_df[metric].median()
    row = metrics_df.loc[(metrics_df[metric] - median_val).abs().idxmin()]
    return int(row["index"]), float(row[metric])


def select_percentile_sample(
    metrics_df,
    metric: str = "MAE",
    q_low: float = 0.5,
    q_high: float = 0.75,
    rank: int = 0,
):
    """
    Select a sample inside a percentile window of the metric distribution.

    Args:
        metrics_df: metrics DataFrame with columns ['index', metric]
        metric: metric column used for the selection
        q_low: lower quantile of the window (0-1)
        q_high: upper quantile of the window (0-1)
        rank: which sample of the window to return (0 = lowest metric in window)

    Returns:
        (sample_id, metric_value)
    """
    lo = metrics_df[metric].quantile(q_low)
    hi = metrics_df[metric].quantile(q_high)
    window = metrics_df[(metrics_df[metric] >= lo) & (metrics_df[metric] <= hi)]
    window = window.sort_values(metric).reset_index(drop=True)
    if len(window) == 0:
        raise ValueError(f"No sample found in the [{q_low}, {q_high}] quantile window")
    row = window.iloc[min(rank, len(window) - 1)]
    return int(row["index"]), float(row[metric])


def rank_samples_by_separation(
    metrics_dfs: dict,
    ref_model: str,
    other_models=None,
    metric: str = "MAE",
    q_low: float = 0.5,
    q_high: float = 0.75,
    top_n: int = 15,
):
    """
    Rank candidate samples by how much the reference model outperforms the others.

    Only samples inside the [q_low, q_high] quantile window of the reference model
    metric distribution are considered. Candidates are sorted by descending
    "separation" = mean(other models metric) - ref model metric.

    Args:
        metrics_dfs: mapping {model_name: metrics DataFrame with ['index', metric]}
        ref_model: model that should look best (e.g. "proT")
        other_models: models to compare against. If None, all except ref_model.
        metric: metric column (lower is better)
        q_low: lower quantile of the window (0-1)
        q_high: upper quantile of the window (0-1)
        top_n: number of candidates returned

    Returns:
        DataFrame with columns ['index', metric+'_ref', metric+'_others', 'separation',
        'ratio'] sorted by descending separation.
    """
    import pandas as pd

    if other_models is None:
        other_models = [m for m in metrics_dfs if m != ref_model]

    df_ref = metrics_dfs[ref_model][["index", metric]].rename(
        columns={metric: f"{metric}_ref"})

    lo = df_ref[f"{metric}_ref"].quantile(q_low)
    hi = df_ref[f"{metric}_ref"].quantile(q_high)
    df_ref = df_ref[(df_ref[f"{metric}_ref"] >= lo) & (df_ref[f"{metric}_ref"] <= hi)]

    df_oth = None
    for m in other_models:
        d = metrics_dfs[m][["index", metric]].rename(columns={metric: m})
        df_oth = d if df_oth is None else df_oth.merge(d, on="index", how="inner")

    out = df_ref.merge(df_oth, on="index", how="inner")
    out[f"{metric}_others"] = out[list(other_models)].mean(axis=1)
    out["separation"] = out[f"{metric}_others"] - out[f"{metric}_ref"]
    out["ratio"] = out[f"{metric}_others"] / out[f"{metric}_ref"]
    out = out.sort_values("separation", ascending=False).reset_index(drop=True)

    cols = ["index", f"{metric}_ref", f"{metric}_others", "separation", "ratio"] + list(other_models)
    return out[cols].head(top_n)


def last_value_df(
    target_array: np.ndarray,
    outputs: dict,
    var_index: int,
    x_index: int,
    val_index: int,
    selected_var=None,
    scale: float = 10.0,
):
    """
    Extract the last (largest position) target/prediction value of each sample.

    For every sample the valid steps of `selected_var` are taken (target value not NaN)
    and the step with the largest position is selected. Values are multiplied by `scale`
    (de-normalization).

    Args:
        target_array: target array (B, S, F)
        outputs: mapping {model_name: prediction array (B, S) or (B, S, D)}
        var_index: feature index of the variable id in target_array
        x_index: feature index of the position in target_array
        val_index: feature index of the value in target_array
        selected_var: variable id to use. If None, the first available one is used.
        scale: de-normalization factor applied to both target and predictions

    Returns:
        DataFrame with columns ['index', 'model', 'true', 'pred'].
    """
    import pandas as pd

    all_vars = np.unique(target_array[:, :, var_index])
    all_vars = all_vars[~np.isnan(all_vars)]
    if selected_var is None:
        selected_var = all_vars[0]

    # flatten predictions to (B, S)
    preds = {}
    for name, arr in outputs.items():
        preds[name] = arr[:, :, 0] if arr.ndim > 2 else arr

    B = target_array.shape[0]
    rows = []

    for b in range(B):
        var_mask = target_array[b, :, var_index] == selected_var
        val = target_array[b, :, val_index]
        pos = target_array[b, :, x_index]
        valid = var_mask & ~np.isnan(val) & ~np.isnan(pos)

        if not valid.any():
            continue

        idx_valid = np.flatnonzero(valid)
        last_idx = idx_valid[np.argmax(pos[idx_valid])]

        for name, arr in preds.items():
            rows.append({
                "index": b,
                "model": name,
                "position": pos[last_idx],
                "true": val[last_idx] * scale,
                "pred": arr[b, last_idx] * scale,
            })

    return pd.DataFrame(rows)


def plot_last_value_parity(
    df_last,
    model_order=None,
    color_map=None,

    ax=None,
    xlabel: str = "True last value",
    ylabel: str = "Predicted last value",
    show_metrics_in_legend: bool = True,
    legend_position: str = "best",
    marker_size: float = 22.0,
    alpha: float = 0.65,
    log_scale: bool = False,
):
    """
    Parity plot (true vs predicted) of the last value, one colour per model.

    Args:
        df_last: DataFrame from `last_value_df` with columns ['model', 'true', 'pred']
        model_order: plotting order of the models. Defaults to the order of appearance.
        color_map: mapping {model_name: color}
        ax: optional matplotlib Axes; created if None
        xlabel, ylabel: axis labels
        show_metrics_in_legend: append MAE / R2 to the legend labels
        legend_position: matplotlib legend `loc`
        marker_size: scatter marker size
        alpha: scatter transparency
        log_scale: use log scale on both axes

    Returns:
        ax: the matplotlib Axes used.
    """
    if ax is None:
        _, ax = plt.subplots()

    if model_order is None:
        model_order = list(dict.fromkeys(df_last["model"].tolist()))

    lo = min(df_last["true"].min(), df_last["pred"].min())
    hi = max(df_last["true"].max(), df_last["pred"].max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad

    # diagonal
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", zorder=1)

    for model in model_order:
        sub = df_last[df_last["model"] == model]
        if len(sub) == 0:
            continue

        label = model
        if show_metrics_in_legend:
            err = sub["pred"].to_numpy() - sub["true"].to_numpy()
            mae = np.mean(np.abs(err))
            var = np.var(sub["true"].to_numpy())
            r2 = 1.0 - np.mean(err ** 2) / var if var > 0 else float("nan")
            label = f"{model} (MAE={mae:.2f}, R2={r2:.2f})"

        color = color_map.get(model) if color_map is not None else None
        ax.scatter(
            sub["true"], sub["pred"],
            s=marker_size, alpha=alpha, label=label,
            color=color, edgecolors="none", zorder=2,
        )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc=legend_position)

    return ax


def sample_mae_per_model(
    sample_id: int,
    target_array: np.ndarray,
    outputs: dict,
    var_index: int,
    val_index: int,
    selected_var=None,
):
    """
    Per-sample MAE of each model on the (normalized) values of one variable.

    Args:
        sample_id: sample index
        target_array: target array (B, S, F)
        outputs: mapping {model_name: prediction array (B, S) or (B, S, D)}
        var_index: feature index of the variable id
        val_index: feature index of the value
        selected_var: variable id to use. If None, the first available one is used.

    Returns:
        dict {model_name: MAE}
    """
    all_vars = np.unique(target_array[:, :, var_index])
    all_vars = all_vars[~np.isnan(all_vars)]
    if selected_var is None:
        selected_var = all_vars[0]

    var_mask = target_array[sample_id, :, var_index] == selected_var
    y_true = target_array[sample_id, :, val_index]
    valid = var_mask & ~np.isnan(y_true)

    mae = {}
    for name, arr in outputs.items():
        pred = arr[:, :, 0] if arr.ndim > 2 else arr
        mae[name] = float(np.mean(np.abs(pred[sample_id][valid] - y_true[valid])))
    return mae


def plot_benchmark_panel(
    sample_id: int,
    var_index: int,
    x_index: int,
    val_index: int,
    target_array: np.ndarray,
    outputs: dict,
    ref_model: str = "proT",
    selected_var=None,
    scale: float = 10.0,
    ax=None,
    color_map=None,
    ref_color=None,
    bench_alpha: float = 0.9,
    bench_linewidth: float = 1.0,
    ref_linewidth: float = 2.0,
    target_linewidth: float = 1.5,
    xlabel: str = "Position",
    ylabel: str = r"$\Delta R\ [\%]$",
    show_legend: bool = True,
    legend_position: str = "best",
    target_label: str = "target",
    model_order=None,
):
    """
    Single panel comparing one reference model against several benchmarks.

    Every benchmark is drawn as a thin coloured line, the reference model is
    highlighted with a thicker line, and the target is drawn in black.
    Colours are taken from `color_map` so that several panels can share one
    common colour legend (see `show_legend=False` + `fig.legend`).

    Args:
        sample_id: sample index to plot
        var_index: feature index of the variable id in target_array
        x_index: feature index of the position in target_array
        val_index: feature index of the value in target_array
        target_array: target array (B, S, F)
        outputs: mapping {model_name: prediction array (B, S) or (B, S, D)}
        ref_model: key of `outputs` to highlight
        selected_var: variable id to plot. If None, the first available one is used.
        scale: de-normalization factor for plotting (metrics use normalized values)
        ax: optional matplotlib Axes; created if None
        color_map: mapping {model_name: colour}. Missing models fall back to the
                   current property cycle.
        ref_color: colour override for the reference model
        bench_alpha, bench_linewidth: styling of the benchmark lines
        ref_linewidth, target_linewidth: line widths of reference and target
        xlabel, ylabel: axis labels
        show_legend: draw the legend on this axes (set False for a shared figure legend)
        legend_position: matplotlib legend `loc`
        target_label: legend label of the target curve
        model_order: plotting / legend order (default: benchmarks then ref)

    Returns:
        (ax, mae_dict): the Axes used and the per-sample MAE of every model.
    """
    if ax is None:

        _, ax = plt.subplots()

    if ref_model not in outputs:
        raise ValueError(f"ref_model '{ref_model}' not found in outputs: {list(outputs)}")

    all_vars = np.unique(target_array[:, :, var_index])
    all_vars = all_vars[~np.isnan(all_vars)]
    if selected_var is None:
        selected_var = all_vars[0]

    var_mask = target_array[sample_id, :, var_index] == selected_var
    x = target_array[sample_id, :, x_index][var_mask]
    y_trg = target_array[sample_id, :, val_index][var_mask]

    # per-sample metrics on normalized values
    mae = sample_mae_per_model(
        sample_id, target_array, outputs, var_index, val_index, selected_var
    )

    bench_models = [m for m in outputs if m != ref_model]
    if model_order is None:
        model_order = bench_models + [ref_model]

    # resolve colours: color_map first, then property cycle
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    colors = {}
    for i, name in enumerate(model_order):
        if color_map is not None and name in color_map:
            colors[name] = color_map[name]
        else:
            colors[name] = cycle[i % len(cycle)]
    if ref_color is not None:
        colors[ref_model] = ref_color

    # target (black, drawn first so the models sit on top)
    ax.plot(x, y_trg * scale, color="black", linewidth=target_linewidth,
            label=target_label, zorder=2)

    # benchmarks: thin coloured lines
    for name in model_order:
        if name == ref_model:
            continue
        arr = outputs[name]
        pred = arr[:, :, 0] if arr.ndim > 2 else arr
        ax.plot(
            x, pred[sample_id][var_mask] * scale,
            color=colors[name], alpha=bench_alpha, linewidth=bench_linewidth,
            label=name, zorder=3,
        )

    # reference model: highlighted
    arr = outputs[ref_model]
    pred = arr[:, :, 0] if arr.ndim > 2 else arr
    ax.plot(x, pred[sample_id][var_mask] * scale, color=colors[ref_model],
            linewidth=ref_linewidth, label=ref_model, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x_valid = x[~np.isnan(x)]
    ax.set_xlim(x_valid.min(), x_valid.max())

    if show_legend:
        ax.legend(frameon=False, loc=legend_position)

    return ax, mae


def benchmark_legend_handles(
    model_order,
    color_map,
    ref_model: str = None,
    bench_linewidth: float = 1.0,
    ref_linewidth: float = 2.0,
    target_label: str = "target",
    target_linewidth: float = 1.5,
    include_target: bool = True,
):
    """
    Build matplotlib handles/labels for a shared colour legend of the model curves.

    Args:
        model_order: order of the model names in the legend
        color_map: mapping {model_name: colour}
        ref_model: model drawn with `ref_linewidth` (all others use `bench_linewidth`)
        bench_linewidth, ref_linewidth: line widths mirrored from the panels
        target_label: label of the black target line
        target_linewidth: line width of the target entry
        include_target: prepend the target entry

    Returns:
        (handles, labels) ready for `fig.legend(handles, labels, ...)`
    """
    from matplotlib.lines import Line2D

    handles, labels = [], []

    if include_target:
        handles.append(Line2D([0], [0], color="black", linewidth=target_linewidth))
        labels.append(target_label)

    for name in model_order:
        lw = ref_linewidth if name == ref_model else bench_linewidth
        handles.append(Line2D([0], [0], color=color_map[name], linewidth=lw))
        labels.append(name)

    return handles, labels


def load_trial_history(exp_path, metric="val_mae"):
    """
    Per-epoch metric history of the best optuna trial, aggregated over folds.

    Reads every optuna/run_<best_trial>/k_*/logs/csv/version_*/metrics.csv and
    returns a DataFrame with columns [epoch, mean, std] for the requested metric.
    """
    import pandas as pd

    best_trial_regex = re.compile("best_trial")
    summary_list = [f for f in os.listdir(exp_path) if best_trial_regex.match(f)]
    if len(summary_list) != 1:
        raise ValueError(f"More (or none) than one best_trial summary found! {summary_list}")
    summary = OmegaConf.load(os.path.join(exp_path, summary_list[0]))

    run_dir = os.path.join(exp_path, "optuna", f"run_{summary.trial_number}")
    pattern = os.path.join(run_dir, "k_*", "logs", "csv", "version_*", "metrics.csv")
    csv_files = sorted(glob.glob(pattern))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No metrics.csv found with pattern {pattern}")

    frames = [pd.read_csv(p, usecols=["epoch", metric]).dropna() for p in csv_files]
    df_all = pd.concat(frames, ignore_index=True)
    return df_all.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()


def load_best_checkpoint_epoch(exp_path, optuna=True):
    """
    Return (best_epoch, val_mae) of the best-fold checkpoint used for prediction.

    Reads kfold_summary.json to find the best fold, then the fold's best_metrics.json
    (written by BestCheckpointCallback) for the epoch/val_mae of its best checkpoint.
    optuna=True: exp_path is an optuna experiment (best run via best_trial.yaml).
    optuna=False: exp_path is a plain training run directory.
    """
    if optuna:
        best_trial_regex = re.compile("best_trial")
        summary_list = [f for f in os.listdir(exp_path) if best_trial_regex.match(f)]
        if len(summary_list) != 1:
            raise ValueError(f"More (or none) than one best_trial summary found! {summary_list}")
        summary = OmegaConf.load(os.path.join(exp_path, summary_list[0]))
        run_dir = os.path.join(exp_path, "optuna", f"run_{summary.trial_number}")
    else:
        run_dir = exp_path

    with open(os.path.join(run_dir, "kfold_summary.json")) as f:
        kfold_summary = json.load(f)
    best_fold = kfold_summary["best_fold"]["fold_number"]

    with open(os.path.join(run_dir, f"k_{best_fold}", "best_metrics.json")) as f:
        best_metrics = json.load(f)

    return int(best_metrics["best_epoch"]), float(best_metrics["val_mae"])


def final_value_table(target_array, var_idx, val_idx, selected_vars=(1.0, 2.0)):
    """Per-sample criterion: mean of the final target values of the given variables."""
    import pandas as pd

    final_vals = np.full(target_array.shape[0], np.nan)
    for b in range(target_array.shape[0]):
        per_var = []
        for var in selected_vars:
            mask = target_array[b, :, var_idx] == var
            vals = target_array[b, :, val_idx][mask]
            vals = vals[~np.isnan(vals)]
            if vals.size:
                per_var.append(vals[-1])
        if per_var:
            final_vals[b] = np.mean(per_var)
    return pd.DataFrame({"index": np.arange(target_array.shape[0]), "final_value": final_vals})


def pick_final_value_sample(target_array, var_idx, val_idx, selected_vars=(1.0, 2.0), mode="min"):
    """
    Return (sample_index, criterion); criterion = mean of per-variable final values.

    min mode uses the lowest positive criterion (negative values excluded).
    """
    final_vals = final_value_table(target_array, var_idx, val_idx, selected_vars)["final_value"].to_numpy()
    if mode == "min":
        final_vals = np.where(final_vals > 0, final_vals, np.nan)
        idx = int(np.nanargmin(final_vals))
    else:
        idx = int(np.nanargmax(final_vals))
    return idx, float(final_vals[idx])


def print_percentile_candidates(target_array, var_idx, val_idx, percentiles,
                                selected_vars=(1.0, 2.0), positive_only=False, mae_df=None):
    """
    Print the sample closest to each percentile of the mean-final-value criterion.

    percentiles: iterable of percentiles in [0, 100], e.g. [1, 5, 50, 95, 99].
    positive_only: restrict to samples with final_value > 0 (useful for the low end).
    mae_df: optional metrics DataFrame with ['index', 'MAE'], appended to the printout.
    """
    df_crit = final_value_table(target_array, var_idx, val_idx, selected_vars).dropna()
    if positive_only:
        df_crit = df_crit[df_crit["final_value"] > 0]
    if mae_df is not None:
        df_crit = df_crit.merge(mae_df[["index", "MAE"]], on="index", how="left")
    print(f"{'pct':>6} {'index':>6} {'final_value':>12}" + (f" {'MAE':>10}" if mae_df is not None else ""))
    for q in percentiles:
        q_val = df_crit["final_value"].quantile(q / 100)
        row = df_crit.loc[df_crit["final_value"].sub(q_val).abs().idxmin()]
        line = f"{q:>6.1f} {int(row['index']):>6} {row['final_value']:>12.4f}"
        if mae_df is not None:
            line += f" {row['MAE']:>10.4f}"
        print(line)


def shared_ylim(sample_ids, target_array, output_array, var_idx, val_idx,
                selected_vars=(1.0, 2.0), scale=10.0, margin=0.05):
    """Common y-limits spanning targets and predictions of the given samples/variables."""
    outputs = output_array
    if outputs.ndim > 2:
        outputs = outputs[:, :, 0]  # flatten (B, S, 1) -> (B, S)
    vals = []
    for sid in sample_ids:
        for var in selected_vars:
            m = target_array[sid, :, var_idx] == var
            vals.append(target_array[sid, :, val_idx][m] * scale)
            vals.append(outputs[sid][m] * scale)
    vals = np.concatenate(vals)
    vals = vals[~np.isnan(vals)]
    pad = margin * (vals.max() - vals.min())
    return vals.min() - pad, vals.max() + pad






