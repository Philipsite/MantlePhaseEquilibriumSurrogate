using JLD2
using CairoMakie
using Sprout

#= ══════════════════════════════════════════════════════════════════════════════
   CONFIGURATION & CONSTANTS
   ══════════════════════════════════════════════════════════════════════════════ =#

# Hyperparameter grid
const N_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
const N_NEURONS = [50, 100, 150, 200, 250, 300, 350, 400]

# Figure dimensions (A4 paper)
const FIGURE_DPI = 72
const FIGURE_WIDTH_MM = 210.0
const FIGURE_HEIGHT_MM = 290.0  # Full A4 height for 3 panels

# Paths
const HPT_RESULTS_DIR = "data/hpt_results/regressor"

# Experiment directories (full paths) - PLACEHOLDER: Update these paths
const EXPERIMENT_DIRS = [
    # pretrained classifier model with frozen classifier layers
    # Backbone fraction = 1/2, batch sizes 1, 2, 3
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_frozen_fbl1_bs1_2026Jan09_1039"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_frozen_fbl1_bs2_2026Jan10_1127"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_frozen_fbl1_bs3_2026Jan11_0453"),
    # Backbone fraction = 2/3, batch sizes 1, 2, 3
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_frozen_fbl2_bs1_2026Jan12_0217"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_frozen_fbl2_bs2_2026Jan13_0222"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_frozen_fbl2_bs3_2026Jan13_1910"),
    # pretrained classifier model with adjustable classifier layers
    # Backbone fraction = 1/2, batch sizes 1, 2, 3
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_fbl1_bs1_2026Jan09_1041"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_fbl1_bs2_2026Jan10_0750"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_fbl1_bs3_2026Jan10_2347"),
    # Backbone fraction = 2/3, batch sizes 1, 2, 3
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_fbl2_bs1_2026Jan11_2324"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_fbl2_bs2_2026Jan12_1922"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningptClas_fbl2_bs3_2026Jan13_1113"),
    # simultaneous training of classifier and regressor > common backbone
    # Backbone fraction = 1/2, batch sizes 1, 2, 3
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningcBack_fbl1_bs1_2026Jan09_1046"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningcBack_fbl1_bs2_2026Jan10_1305"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningcBack_fbl1_bs3_2026Jan11_1133"),
    # Backbone fraction = 2/3, batch sizes 1, 2, 3
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningcBack_fbl2_bs1_2026Jan12_1827"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningcBack_fbl2_bs2_2026Jan13_1757"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningcBack_fbl2_bs2_2026Jan13_1757"), #FIXME - duplicate as placeholder; update when available
]

# Experiment metadata (aligned with EXPERIMENT_DIRS)
const BACKBONE_FRACTIONS = ["1/2", "1/2", "1/2", "2/3", "2/3", "2/3"]
const EXPERIMENT_BATCH_SIZES = [4096, 25000, 100000]  # Unique batch sizes for labels
# Full batch sizes for all 18 experiments (repeats 3x for each training setup)
const EXPERIMENT_BATCH_SIZES_FULL = repeat([4096, 25000, 100000, 4096, 25000, 100000], 3)

# Training setup labels for each panel
const TRAINING_SETUP_LABELS = [
    "Pretrained Classifier (Frozen)",
    "Pretrained Classifier (Adjustable)",
    "Common Backbone",
]

#= ══════════════════════════════════════════════════════════════════════════════
   THEME SETUP
   ══════════════════════════════════════════════════════════════════════════════ =#

function create_figure_theme()
    Theme(
        # Global font settings
        font = "Helvetica",
        fontsize = 10,

        # Figure background
        figure_padding = 10,
        backgroundcolor = RGBf(0.98, 0.98, 0.98),

        # Axis defaults
        Axis = (
            xgridvisible = false,
            ygridvisible = true,
            spinewidth = 0.5,
            xlabelpadding = 5,
            ylabelpadding = 5,
        ),

        # Heatmap defaults
        Heatmap = (
            interpolate = false,
        ),

        # Colorbar defaults
        Colorbar = (
            size = 8,
            ticklabelsize = 9,
            labelsize = 10,
            spinewidth = 0.5,
        ),
    )
end

#= ══════════════════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════════════════ =#

"""
Load all experiment data, returning metric arrays and learning curves:
- mae_𝑣: 3D array (n_layers × n_neurons × n_experiments) of minimum MAE for 𝑣
- mae_𝐗: 3D array (n_layers × n_neurons × n_experiments) of minimum MAE for 𝐗
- curves_𝑣: 3D array of mae_𝑣 learning curve vectors
- curves_𝐗: 3D array of mae_𝐗 learning curve vectors

Arguments:
- experiment_dirs: Vector of paths to experiment directories
- n_layers: Vector of layer counts in the hyperparameter grid
- n_neurons: Vector of neuron counts in the hyperparameter grid
"""
function load_all_experiments(experiment_dirs::Vector{String},
                               n_layers::Vector{Int}, n_neurons::Vector{Int})
    mae_𝑣_slices = Vector{Matrix{Float64}}()
    mae_𝐗_slices = Vector{Matrix{Float64}}()
    curves_𝑣_slices = Vector{Matrix{Vector{Float64}}}()
    curves_𝐗_slices = Vector{Matrix{Vector{Float64}}}()

    for path in experiment_dirs
        logs = load_hyperparam_tuning_results(path, n_layers, n_neurons)

        # Extract minimum MAE for each model configuration
        push!(mae_𝑣_slices, minimum.(getfield.(logs, :mae_𝑣)))
        push!(mae_𝐗_slices, minimum.(getfield.(logs, :mae_𝐗)))
        # Extract full learning curves
        push!(curves_𝑣_slices, getfield.(logs, :mae_𝑣))
        push!(curves_𝐗_slices, getfield.(logs, :mae_𝐗))
    end

    mae_𝑣 = cat(mae_𝑣_slices..., dims=3)
    mae_𝐗 = cat(mae_𝐗_slices..., dims=3)
    curves_𝑣 = cat(curves_𝑣_slices..., dims=3)
    curves_𝐗 = cat(curves_𝐗_slices..., dims=3)

    return mae_𝑣, mae_𝐗, curves_𝑣, curves_𝐗
end

# Convenience method using default paths from constants
function load_all_experiments()
    load_all_experiments(EXPERIMENT_DIRS, N_LAYERS, N_NEURONS)
end

#= ══════════════════════════════════════════════════════════════════════════════
   PLOTTING UTILITIES
   ══════════════════════════════════════════════════════════════════════════════ =#

"""Convert mm to pixels at given DPI."""
mm_to_px(mm, dpi=FIGURE_DPI) = round(Int, mm * (1/25.4) * dpi)

"""Create a heatmap axis with standard hyperparameter tuning configuration."""
function create_hpt_axis!(layout, row, col; xlabel="n.o. hidden layers", ylabel="n.o. neurons")
    ax = Axis(layout[row, col],
        aspect = 1.0,
        xlabel = xlabel,
        ylabel = ylabel,
        xticks = (N_LAYERS, string.(N_LAYERS)),
        yticks = (N_NEURONS, string.(N_NEURONS)),
    )
    return ax
end

"""Plot a heatmap on the given axis."""
function plot_hpt_heatmap!(ax, data::Matrix; colormap, colorrange)
    heatmap!(ax, N_LAYERS, N_NEURONS, data; colormap, colorrange)
end

"""Add an optimal model marker to the axis."""
function mark_optimal_model!(ax, idx::CartesianIndex;
                             color=:red, marker=:star5, markersize=8)
    scatter!(ax, N_LAYERS[idx[1]], N_NEURONS[idx[2]];
             color, marker, markersize)
end

"""Compute model size matrix (n_layers × n_neurons)."""
function compute_model_sizes()
    N_LAYERS .* N_NEURONS'
end

"""Plot training curves from a 3D array of loss vectors."""
function plot_training_curves!(ax, curves::Array{Vector{Float64},3}, exp_indices::UnitRange,
                                model_size::Matrix; colormap, colorrange)
    for exp_idx in exp_indices
        for j in eachindex(view(curves, :, :, exp_idx))
            lines!(ax, curves[:, :, exp_idx][j];
                   linewidth = 0.3,
                   color = model_size[j],
                   colormap, colorrange)
        end
    end
end

"""Highlight a specific model's training curve."""
function highlight_training_curve!(ax, loss_vec::Vector;
                                    color=:red, linewidth=1.0,
                                    label=nothing, label_position=:top)
    lines!(ax, loss_vec; color, linewidth)

    if !isnothing(label)
        min_epoch = argmin(loss_vec)
        min_loss = minimum(loss_vec)
        valign = label_position == :top ? :bottom : :top
        text!(ax, Float32(min_epoch), min_loss;
              text = "$label $(round(min_loss, digits=4))",
              align = (:center, valign),
              fontsize = 8,
              color)
    end
end

#= ══════════════════════════════════════════════════════════════════════════════
   PANEL BUILDERS
   ══════════════════════════════════════════════════════════════════════════════ =#

"""Build a panel: 2×6 grid of heatmaps showing two metrics for 6 experiments.
Grid layout:
- Row 1: Panel title (spanning columns 3-8)
- Rows 2-3: Heatmaps (2 backbone fractions)
- Columns 1-2: Backbone fraction labels
- Columns 3-5: mae_𝑣 heatmaps (3 batch sizes)
- Columns 6-8: mae_𝐗 heatmaps (3 batch sizes)

Arguments:
- global_opt_𝑣_idx: Global optimal model index for mae_𝑣 (across all experiments)
- global_opt_𝐗_idx: Global optimal model index for mae_𝐗 (across all experiments)
- panel_exp_range: Range of experiment indices this panel covers (e.g., 1:6)
"""
function build_heatmap_panel!(grid, mae_𝑣::Array{Float64,3}, mae_𝐗::Array{Float64,3},
                               panel_title::String;
                               colorrange_𝑣, colorrange_𝐗,
                               global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range)

    colormap_mae_𝑣 = cgrad(:acton, rev=true)
    colormap_mae_𝐗 = cgrad(:lipari, rev=true)

    # Create 2×3 grid of axes for mae_𝑣 at rows 2-3, columns 3-5
    axs_𝑣 = Matrix{Axis}(undef, 2, 3)
    for row in 1:2, col in 1:3
        axs_𝑣[row, col] = create_hpt_axis!(grid, row + 1, col + 2)
    end

    # Create 2×3 grid of axes for mae_𝐗 at rows 2-3, columns 6-8
    axs_𝐗 = Matrix{Axis}(undef, 2, 3)
    for row in 1:2, col in 1:3
        axs_𝐗[row, col] = create_hpt_axis!(grid, row + 1, col + 5)
    end

    # Plot heatmaps for mae_𝑣
    for i in 1:6
        row = (i <= 3) ? 1 : 2
        col = ((i - 1) % 3) + 1
        plot_hpt_heatmap!(axs_𝑣[row, col], mae_𝑣[:, :, i]; colormap=colormap_mae_𝑣, colorrange=colorrange_𝑣)
    end

    # Plot heatmaps for mae_𝐗
    for i in 1:6
        row = (i <= 3) ? 1 : 2
        col = ((i - 1) % 3) + 1
        plot_hpt_heatmap!(axs_𝐗[row, col], mae_𝐗[:, :, i]; colormap=colormap_mae_𝐗, colorrange=colorrange_𝐗)
    end

    # Clean up axis decorations for mae_𝑣
    hidedecorations!.([axs_𝑣[1, 2], axs_𝑣[1, 3]], ticks=false, grid=false)
    axs_𝑣[1, 1].xlabelvisible = false
    axs_𝑣[1, 1].xticklabelsvisible = false
    axs_𝑣[2, 2].ylabelvisible = false
    axs_𝑣[2, 2].yticklabelsvisible = false
    axs_𝑣[2, 3].ylabelvisible = false
    axs_𝑣[2, 3].yticklabelsvisible = false

    # Clean up axis decorations for mae_𝐗 (hide all y-axis labels)
    for row in 1:2, col in 1:3
        axs_𝐗[row, col].ylabelvisible = false
        axs_𝐗[row, col].yticklabelsvisible = false
    end
    hidedecorations!.([axs_𝐗[1, 1], axs_𝐗[1, 2], axs_𝐗[1, 3]], ticks=false, grid=false)

    # Helper to convert experiment index (1-6) to grid position (row, col)
    function exp_to_grid(exp_idx)
        row = exp_idx <= 3 ? 1 : 2
        col = ((exp_idx - 1) % 3) + 1
        return row, col
    end

    # Mark global optimal model for mae_𝑣 (only if it's in this panel)
    if global_opt_𝑣_idx[3] in panel_exp_range
        local_exp_idx = global_opt_𝑣_idx[3] - first(panel_exp_range) + 1
        𝑣_row, 𝑣_col = exp_to_grid(local_exp_idx)
        local_idx = CartesianIndex(global_opt_𝑣_idx[1], global_opt_𝑣_idx[2])
        mark_optimal_model!(axs_𝑣[𝑣_row, 𝑣_col], local_idx; color=:red, marker=:star5, markersize=8)
        # Add text with metric value
        opt_val = mae_𝑣[local_idx[1], local_idx[2], local_exp_idx]
        text!(axs_𝑣[𝑣_row, 𝑣_col], N_LAYERS[local_idx[1]], N_NEURONS[local_idx[2]];
              text=string(round(opt_val, digits=4)), fontsize=8, color=:red,
              align=(:left, :bottom), offset=(5, -5))
    end

    # Mark global optimal model for mae_𝐗 (only if it's in this panel)
    if global_opt_𝐗_idx[3] in panel_exp_range
        local_exp_idx = global_opt_𝐗_idx[3] - first(panel_exp_range) + 1
        X_row, X_col = exp_to_grid(local_exp_idx)
        local_idx = CartesianIndex(global_opt_𝐗_idx[1], global_opt_𝐗_idx[2])
        mark_optimal_model!(axs_𝐗[X_row, X_col], local_idx; color=:red, marker=:star5, markersize=8)
        # Add text with metric value
        opt_val = mae_𝐗[local_idx[1], local_idx[2], local_exp_idx]
        text!(axs_𝐗[X_row, X_col], N_LAYERS[local_idx[1]], N_NEURONS[local_idx[2]];
              text=string(round(opt_val, digits=4)), fontsize=8, color=:red,
              align=(:left, :bottom), offset=(5, -5))
    end

    # Panel title at row 1 (spanning both heatmap column groups)
    Label(grid[1, 3:8], panel_title; fontsize=11, font=:bold, tellwidth=false)

    # Backbone fraction labels (columns 1-2)
    Label(grid[2:3, 1], "Backbone fraction"; rotation=π/2, fontsize=10, font=:bold, tellheight=false)
    Label(grid[2, 2], "1/2"; rotation=π/2, fontsize=10, tellheight=false)
    Label(grid[3, 2], "2/3"; rotation=π/2, fontsize=10, tellheight=false)

    return axs_𝑣, axs_𝐗
end

#= ══════════════════════════════════════════════════════════════════════════════
   MAIN FIGURE ASSEMBLY
   ══════════════════════════════════════════════════════════════════════════════ =#

function create_hpt_regressor_figure()
    # Load all data at once (18 experiments) - two metrics and curves
    mae_𝑣, mae_𝐗, _, _ = load_all_experiments(EXPERIMENT_DIRS, N_LAYERS, N_NEURONS)

    # Split into 3 panels (6 experiments each)
    mae_𝑣_a = mae_𝑣[:, :, 1:6]    # Pretrained Classifier (Frozen)
    mae_𝑣_b = mae_𝑣[:, :, 7:12]   # Pretrained Classifier (Adjustable)
    mae_𝑣_c = mae_𝑣[:, :, 13:18]  # Common Backbone

    mae_𝐗_a = mae_𝐗[:, :, 1:6]
    mae_𝐗_b = mae_𝐗[:, :, 7:12]
    mae_𝐗_c = mae_𝐗[:, :, 13:18]

    # Global colorranges for comparability across all panels
    colorrange_𝑣 = (minimum(mae_𝑣), maximum(mae_𝑣))
    colorrange_𝐗 = (minimum(mae_𝐗), maximum(mae_𝐗))

    # Find global optimal models across ALL experiments
    global_opt_𝑣_idx = argmin(mae_𝑣)
    global_opt_𝐗_idx = argmin(mae_𝐗)

    # Create figure with theme
    with_theme(create_figure_theme()) do
        fig = Figure(size = (mm_to_px(FIGURE_WIDTH_MM), mm_to_px(FIGURE_HEIGHT_MM)))

        # Create 3 panels vertically
        grid_a = fig[1, 1] = GridLayout()
        grid_b = fig[2, 1] = GridLayout()
        grid_c = fig[3, 1] = GridLayout()
        # Create 4th panel for labels and colorbars
        grid_d = fig[4, 1] = GridLayout()

        # Build panels with shared colorranges and global optimal markers
        build_heatmap_panel!(grid_a, mae_𝑣_a, mae_𝐗_a, TRAINING_SETUP_LABELS[1];
                             colorrange_𝑣, colorrange_𝐗,
                             global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range=1:6)
        build_heatmap_panel!(grid_b, mae_𝑣_b, mae_𝐗_b, TRAINING_SETUP_LABELS[2];
                             colorrange_𝑣, colorrange_𝐗,
                             global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range=7:12)
        build_heatmap_panel!(grid_c, mae_𝑣_c, mae_𝐗_c, TRAINING_SETUP_LABELS[3];
                             colorrange_𝑣, colorrange_𝐗,
                             global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range=13:18)

        # Add spacer box to grid_d columns 1-2 to maintain alignment
        Box(grid_d[1:3, 1:2]; color=:transparent, strokevisible=false)

        # Set fixed widths for label columns (1-2) in all grids
        label_col_width = 5
        for grid in [grid_a, grid_b, grid_c, grid_d]
            colsize!(grid, 1, Fixed(label_col_width))
            colsize!(grid, 2, Fixed(label_col_width))
        end

        # Add batch size labels for mae_𝑣 columns (3-5)
        Label(grid_d[1, 3], "$(EXPERIMENT_BATCH_SIZES[1])"; fontsize=10, tellwidth=false)
        Label(grid_d[1, 4], "$(EXPERIMENT_BATCH_SIZES[2])"; fontsize=10, tellwidth=false)
        Label(grid_d[1, 5], "$(EXPERIMENT_BATCH_SIZES[3])"; fontsize=10, tellwidth=false)
        Label(grid_d[2, 3:5], "Batch size"; fontsize=10, font=:bold, tellwidth=false)

        # Add batch size labels for mae_𝐗 columns (6-8)
        Label(grid_d[1, 6], "$(EXPERIMENT_BATCH_SIZES[1])"; fontsize=10, tellwidth=false)
        Label(grid_d[1, 7], "$(EXPERIMENT_BATCH_SIZES[2])"; fontsize=10, tellwidth=false)
        Label(grid_d[1, 8], "$(EXPERIMENT_BATCH_SIZES[3])"; fontsize=10, tellwidth=false)
        Label(grid_d[2, 6:8], "Batch size"; fontsize=10, font=:bold, tellwidth=false)

        # Colorbar for mae_𝑣
        colormap_mae_𝑣 = cgrad(:acton, rev=true)
        Colorbar(grid_d[3, 3:5]; colormap=colormap_mae_𝑣, limits=colorrange_𝑣,
                 label=L"\textrm{Mean Absolute Error 𝑣 [molmol^{-1}]}", vertical=false)

        # Colorbar for mae_𝐗
        colormap_mae_𝐗 = cgrad(:lipari, rev=true)
        Colorbar(grid_d[3, 6:8]; colormap=colormap_mae_𝐗, limits=colorrange_𝐗,
                 label=L"\textrm{Mean Absolute Error 𝐗_{ss} [molmol^{-1}]}", vertical=false)

        return fig
    end
end

"""Build a learning curves panel: two columns showing mae_𝑣 and mae_𝐗 curves."""
function build_learning_curves_panel!(grid, curves_𝑣::Array{Vector{Float64},3},
                                       curves_𝐗::Array{Vector{Float64},3},
                                       panel_title::String, exp_indices::UnitRange;
                                       model_size, colormap, colorrange,
                                       global_opt_𝑣_idx=nothing, global_opt_𝐗_idx=nothing,
                                       panel_exp_range=nothing)
    x_limits = (9, 1001)
    y_limits = (-0.001, 0.051)

    # mae_𝑣 curves (left column)
    ax_𝑣 = Axis(grid[1, 1];
        xscale = log10,
        xlabel = "n.o. epochs",
        ylabel = L"\textrm{Mean Absolute Error 𝑣 [molmol^{-1}]}",
    )
    xlims!(ax_𝑣, x_limits...)
    ylims!(ax_𝑣, y_limits...)

    # mae_𝐗 curves (right column)
    ax_𝐗 = Axis(grid[1, 2];
        xscale = log10,
        xlabel = "n.o. epochs",
        ylabel = L"\textrm{Mean Absolute Error 𝐗_{ss} [molmol^{-1}]}",
    )
    xlims!(ax_𝐗, x_limits...)
    ylims!(ax_𝐗, y_limits...)

    # Plot all training curves
    plot_training_curves!(ax_𝑣, curves_𝑣, exp_indices, model_size; colormap, colorrange)
    plot_training_curves!(ax_𝐗, curves_𝐗, exp_indices, model_size; colormap, colorrange)

    # Highlight optimal model curves if indices are provided
    if !isnothing(global_opt_𝑣_idx) && !isnothing(panel_exp_range)
        global_exp_idx_𝑣 = global_opt_𝑣_idx[3]
        if global_exp_idx_𝑣 in panel_exp_range
            local_exp_idx = global_exp_idx_𝑣 - first(panel_exp_range) + 1
            curve_𝑣 = curves_𝑣[global_opt_𝑣_idx[1], global_opt_𝑣_idx[2], local_exp_idx]
            highlight_training_curve!(ax_𝑣, curve_𝑣; label="Optimal MAE v")
        end
    end

    if !isnothing(global_opt_𝐗_idx) && !isnothing(panel_exp_range)
        global_exp_idx_𝐗 = global_opt_𝐗_idx[3]
        if global_exp_idx_𝐗 in panel_exp_range
            local_exp_idx = global_exp_idx_𝐗 - first(panel_exp_range) + 1
            curve_𝐗 = curves_𝐗[global_opt_𝐗_idx[1], global_opt_𝐗_idx[2], local_exp_idx]
            highlight_training_curve!(ax_𝐗, curve_𝐗; label="Optimal MAE 𝐗")
        end
    end

    # Panel title spanning both columns
    Label(grid[0, 1:2], panel_title; fontsize=11, font=:bold, tellwidth=false)

    return ax_𝑣, ax_𝐗
end

"""Create the learning curves figure showing training progression for all models."""
function create_learning_curves_figure()
    # Load all data
    mae_𝑣, mae_𝐗, curves_𝑣, curves_𝐗 = load_all_experiments(EXPERIMENT_DIRS, N_LAYERS, N_NEURONS)

    # Find global optimal models across ALL experiments
    global_opt_𝑣_idx = argmin(mae_𝑣)
    global_opt_𝐗_idx = argmin(mae_𝐗)

    # Compute model sizes for color coding
    model_size = compute_model_sizes()
    colormap = cgrad(:bamako, rev=true)
    colorrange = (minimum(model_size), maximum(model_size))

    # Create figure with theme
    with_theme(create_figure_theme()) do
        fig = Figure(size = (mm_to_px(FIGURE_WIDTH_MM), mm_to_px(FIGURE_HEIGHT_MM)))

        # Create 3 panels vertically for each training setup
        grid_a = fig[1, 1] = GridLayout()
        grid_b = fig[2, 1] = GridLayout()
        grid_c = fig[3, 1] = GridLayout()
        # Create 4th panel for colorbar
        grid_d = fig[4, 1] = GridLayout()

        # Build panels - each training setup has 6 experiments
        # Panel A: Pretrained Classifier (Frozen) - experiments 1-6
        build_learning_curves_panel!(grid_a, curves_𝑣[:, :, 1:6], curves_𝐗[:, :, 1:6],
                                      TRAINING_SETUP_LABELS[1], 1:6;
                                      model_size, colormap, colorrange,
                                      global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range=1:6)

        # Panel B: Pretrained Classifier (Adjustable) - experiments 7-12
        build_learning_curves_panel!(grid_b, curves_𝑣[:, :, 7:12], curves_𝐗[:, :, 7:12],
                                      TRAINING_SETUP_LABELS[2], 1:6;
                                      model_size, colormap, colorrange,
                                      global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range=7:12)

        # Panel C: Common Backbone - experiments 13-18
        build_learning_curves_panel!(grid_c, curves_𝑣[:, :, 13:18], curves_𝐗[:, :, 13:18],
                                      TRAINING_SETUP_LABELS[3], 1:6;
                                      model_size, colormap, colorrange,
                                      global_opt_𝑣_idx, global_opt_𝐗_idx, panel_exp_range=13:18)

        # Horizontal colorbar at the bottom
        Colorbar(grid_d[1, 1]; colormap, limits=colorrange,
                 label = "← smaller models / larger models →",
                 ticksvisible = false,
                 ticklabelsvisible = false,
                 vertical = false)

        return fig
    end
end

#= ══════════════════════════════════════════════════════════════════════════════
   SCRIPT EXECUTION
   ══════════════════════════════════════════════════════════════════════════════ =#

function main()
    # Create heatmap figure
    fig_heatmaps = create_hpt_regressor_figure()
    save(joinpath(@__DIR__, "hpt_regressor.pdf"), fig_heatmaps; dpi=300)

    # Create learning curves figure
    fig_curves = create_learning_curves_figure()
    save(joinpath(@__DIR__, "hpt_regressor_curves.pdf"), fig_curves; dpi=300)

    return fig_heatmaps, fig_curves
end

# Only run when script is executed directly (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
