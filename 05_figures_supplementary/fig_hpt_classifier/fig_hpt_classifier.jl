using JLD2
using CairoMakie
using Sprout

#= ══════════════════════════════════════════════════════════════════════════════
   CONFIGURATION & CONSTANTS
   ══════════════════════════════════════════════════════════════════════════════ =#

# Hyperparameter grid
const N_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
const N_NEURONS = [50, 100, 150, 200, 250, 300, 350, 400]

# Threshold for selecting "fast and precise-enough" models
const MINIMAL_METRIC_THRESHOLD = 0.015

# Figure dimensions (A4 paper)
const FIGURE_DPI = 72
const FIGURE_WIDTH_MM = 210.0
const FIGURE_HEIGHT_MM = 290.0

# Paths
const HPT_RESULTS_DIR = "data/hpt_results/classifier"
const INF_TIMES_PATH = joinpath(HPT_RESULTS_DIR, "inference_times_hpt_classifier.jld2")

# Experiment directories (full paths)
const EXPERIMENT_DIRS = [
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningbs1_bce_2025Dec22_1229"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningbs2_bce_2025Dec23_0027"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningbs3_bce_2025Dec23_1440"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningbs1_bfl_2025Dec24_1133"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningbs2_bfl_2025Dec24_2340"),
    joinpath(HPT_RESULTS_DIR, "hyperparam_tuningbs3_bfl_2025Dec25_1638"),
]

# Experiment metadata (aligned with EXPERIMENT_DIRS)
const LOSS_METHODS = [:BCE, :BCE, :BCE, :BFL, :BFL, :BFL]
const EXPERIMENT_BATCH_SIZES = [4096, 25000, 100000, 4096, 25000, 100000]

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

        # Lines defaults
        Lines = (
            linewidth = 0.3,
        ),
    )
end

#= ══════════════════════════════════════════════════════════════════════════════
   DATA LOADING
   ══════════════════════════════════════════════════════════════════════════════ =#

"""
Load all experiment data, returning only the needed arrays:
- qasm_loss: 3D array (n_layers × n_neurons × n_experiments) of minimum Qasm loss
- loss_curves: 3D array of loss curve vectors for training visualization
- inf_times: 3D array of inference times

Arguments:
- experiment_dirs: Vector of paths to experiment directories
- inf_times_path: Path to the inference times JLD2 file
- n_layers: Vector of layer counts in the hyperparameter grid
- n_neurons: Vector of neuron counts in the hyperparameter grid
"""
function load_all_experiments(experiment_dirs::Vector{String}, inf_times_path::String,
                               n_layers::Vector{Int}, n_neurons::Vector{Int})
    qasm_slices = Vector{Matrix{Float64}}()
    curve_slices = Vector{Matrix{Vector{Float64}}}()

    for path in experiment_dirs
        logs = load_hyperparam_tuning_results(path, n_layers, n_neurons)

        # Extract only what we need, then discard logs
        push!(qasm_slices, minimum.(getfield.(logs, :loss_asm)))
        push!(curve_slices, getfield.(logs, :loss_asm))
    end

    qasm_loss = cat(qasm_slices..., dims=3)
    loss_curves = cat(curve_slices..., dims=3)

    # Load inference times
    @load inf_times_path t_EXP1 t_EXP2 t_EXP3 t_EXP4 t_EXP5 t_EXP6
    inf_times = Float64.(cat(t_EXP1, t_EXP2, t_EXP3, t_EXP4, t_EXP5, t_EXP6, dims=3))

    return qasm_loss, loss_curves, inf_times
end


#= ══════════════════════════════════════════════════════════════════════════════
   ANALYSIS UTILITIES
   ══════════════════════════════════════════════════════════════════════════════ =#

"""Compute model size matrix (n_layers × n_neurons)."""
function compute_model_sizes()
    N_LAYERS .* N_NEURONS'
end

"""Find optimal model index across all experiments."""
function find_optimal_model(qasm_loss::Array{Float64,3})
    argmin(qasm_loss)
end

"""Find optimal model among BCE experiments only (first 3)."""
function find_optimal_bce_model(qasm_loss::Array{Float64,3})
    argmin(qasm_loss[:, :, 1:3])
end

"""Find fastest model that meets the quality threshold."""
function find_fastest_qualified_model(qasm_loss::Array{Float64,3}, inf_times::Array{Float64,3};
                                       threshold=MINIMAL_METRIC_THRESHOLD, bce_only=false)
    if bce_only
        subset = qasm_loss[:, :, 1:3]
        ids = findall(subset .< threshold)
        ids[argmin(inf_times[:, :, 1:3][ids])]
    else
        ids = findall(qasm_loss .< threshold)
        ids[argmin(inf_times[ids])]
    end
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

"""Plot a heatmap on the given axis with qasm loss data."""
function plot_hpt_heatmap!(ax, data::Matrix; colormap, colorrange)
    heatmap!(ax, N_LAYERS, N_NEURONS, data; colormap, colorrange)
end

"""Add an optimal model marker to the axis."""
function mark_optimal_model!(ax, idx::CartesianIndex;
                             color=:red, marker=:star5, markersize=12)
    scatter!(ax, N_LAYERS[idx[1]], N_NEURONS[idx[2]];
             color, marker, markersize)
end

"""Add inference time annotation to a marker."""
function annotate_inference_time!(ax, idx::CartesianIndex, time_ms::Real;
                                   color=:red, align=(:right, :bottom), offset=(5, 5))
    text!(ax, N_LAYERS[idx[1]], N_NEURONS[idx[2]];
          text = "$(round(time_ms, digits=0)) ms",
          align, color, offset)
end

"""Plot training curves from a 3D array of loss vectors."""
function plot_training_curves!(ax, loss_curves::Array{Vector{Float64},3}, exp_indices::UnitRange,
                                model_size::Matrix; colormap, colorrange)
    for exp_idx in exp_indices
        for j in eachindex(view(loss_curves, :, :, exp_idx))
            lines!(ax, loss_curves[:, :, exp_idx][j];
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
        valign = label_position == :top ? :top : :bottom
        text!(ax, Float32(min_epoch), 0.0;
              text = "$label $(round(min_loss, digits=4))",
              align = (:center, valign),
              color)
    end
end

#= ══════════════════════════════════════════════════════════════════════════════
   PANEL BUILDERS
   ══════════════════════════════════════════════════════════════════════════════ =#

"""Build Panel A: 2×3 grid of heatmaps showing Qasm loss for all experiments."""
function build_panel_a!(grid, qasm_loss::Array{Float64,3};
                        opt_idx, opt_bce_idx, opt_fast_idx, opt_fast_bce_idx)

    colormap = cgrad(:acton, rev=true)
    colorrange = (minimum(qasm_loss), maximum(qasm_loss))

    # Create 2×3 grid of axes
    axs = Matrix{Axis}(undef, 2, 3)
    for row in 1:2, col in 1:3
        axs[row, col] = create_hpt_axis!(grid, row, col)
    end

    # Plot heatmaps
    for i in 1:6
        row = (i <= 3) ? 1 : 2
        col = ((i - 1) % 3) + 1
        plot_hpt_heatmap!(axs[row, col], qasm_loss[:, :, i]; colormap, colorrange)
    end

    # Clean up axis decorations
    hidedecorations!.([axs[1, 2], axs[1, 3]], ticks=false, grid=false)
    axs[1, 1].xlabelvisible = false
    axs[1, 1].xticklabelsvisible = false
    axs[2, 2].ylabelvisible = false
    axs[2, 2].yticklabelsvisible = false
    axs[2, 3].ylabelvisible = false
    axs[2, 3].yticklabelsvisible = false

    # Add optimal model markers
    mark_optimal_model!(axs[opt_idx[3] <= 3 ? 1 : 2, ((opt_idx[3]-1) % 3) + 1], opt_idx;
                        color=:red, marker=:star5, markersize=12)
    mark_optimal_model!(axs[1, opt_bce_idx[3]], opt_bce_idx;
                        color=:red, marker=:diamond, markersize=10)

    # Add fast model markers (if in BFL experiments)
    if opt_fast_idx[3] > 3
        mark_optimal_model!(axs[2, ((opt_fast_idx[3]-1) % 3) + 1], opt_fast_idx;
                            color=:blue, marker=:star5, markersize=10)
    end
    mark_optimal_model!(axs[1, opt_fast_bce_idx[3]], opt_fast_bce_idx;
                        color=:blue, marker=:diamond, markersize=8)

    # Add colorbar
    Colorbar(grid[1:2, 4]; colormap, limits=colorrange,
             label=L"1 - \textrm{Q_{asm}}")

    # Add row/column labels
    # "Batch size" header spanning all columns
    Label(grid[-1, 1:3], "Batch size"; fontsize=10, font=:bold, tellwidth=false)
    # Individual batch size values below
    Label(grid[0, 1], "$(EXPERIMENT_BATCH_SIZES[1])"; fontsize=10, tellwidth=false)
    Label(grid[0, 2], "$(EXPERIMENT_BATCH_SIZES[2])"; fontsize=10, tellwidth=false)
    Label(grid[0, 3], "$(EXPERIMENT_BATCH_SIZES[3])"; fontsize=10, tellwidth=false)
    # Row labels for loss functions
    Label(grid[1, 0], "Binary Cross-Entropy Loss"; rotation=π/2, fontsize=10, font=:bold, tellheight=false)
    Label(grid[2, 0], "Binary Focal Loss"; rotation=π/2, fontsize=10, font=:bold, tellheight=false)

    return axs
end

"""Build Panel B: Training curves comparison."""
function build_panel_b!(grid, loss_curves::Array{Vector{Float64},3};
                        opt_idx, opt_bce_idx, opt_fast_idx, opt_fast_bce_idx)

    model_size = compute_model_sizes()
    colormap = cgrad(:bamako, rev=true)
    colorrange = (minimum(model_size), maximum(model_size))

    # BCE training curves (top)
    ax_bce = Axis(grid[1, 1];
        xscale = log10,
        title = "Models trained with Binary Cross-Entropy Loss",
        xlabel = "n.o. epochs (iteration over training set)",
        ylabel = L"1 - \textrm{Q_{asm}}",
    )
    xlims!(ax_bce, 9, nothing)
    ylims!(ax_bce, nothing, 0.25)

    # BFL training curves (bottom)
    ax_bfl = Axis(grid[2, 1];
        xscale = log10,
        title = "Models trained with Binary Focal Loss",
        xlabel = "n.o. epochs (iteration over training set)",
        ylabel = L"1 - \textrm{Q_{asm}}",
    )
    xlims!(ax_bfl, 9, nothing)
    ylims!(ax_bfl, nothing, 0.25)

    # Plot all training curves (BCE: experiments 1-3, BFL: experiments 4-6)
    plot_training_curves!(ax_bce, loss_curves, 1:3, model_size; colormap, colorrange)
    plot_training_curves!(ax_bfl, loss_curves, 4:6, model_size; colormap, colorrange)

    # Highlight optimal models
    # Overall optimal (if in BFL)
    if opt_idx[3] > 3
        highlight_training_curve!(ax_bfl, loss_curves[opt_idx]; color=:red, label="Minimal loss")
    end

    # Optimal BCE
    highlight_training_curve!(ax_bce, loss_curves[opt_bce_idx]; color=:red, label="Minimal loss BCE")

    # Fast optimal (if in BFL)
    if opt_fast_idx[3] > 3
        highlight_training_curve!(ax_bfl, loss_curves[opt_fast_idx]; color=:blue,
                                   label="Minimal loss quick model", label_position=:bottom)
    end

    # Fast BCE
    highlight_training_curve!(ax_bce, loss_curves[opt_fast_bce_idx]; color=:blue,
                               label="Minimal loss quick BCE", label_position=:bottom)

    # Colorbar
    Colorbar(grid[1:2, 2]; colormap, limits=colorrange,
             label = "← smaller models / larger models →",
             ticksvisible = false,
             ticklabelsvisible = false)

    return (ax_bce, ax_bfl)
end

"""Build Panel C: Inference time heatmaps for best experiment configurations."""
function build_panel_c!(grid, inf_times::Array{Float64,3};
                        opt_idx, opt_bce_idx, opt_fast_idx, opt_fast_bce_idx)

    colormap = cgrad(:lapaz, rev=true)
    colorrange = (minimum(inf_times), maximum(inf_times))

    # BCE inference times (top)
    ax_bce = create_hpt_axis!(grid, 1, 1)
    ax_bce.title = "BCE & batch size = $(EXPERIMENT_BATCH_SIZES[opt_bce_idx[3]])"
    plot_hpt_heatmap!(ax_bce, inf_times[:, :, opt_bce_idx[3]]; colormap, colorrange)

    # BFL inference times (bottom) - only if optimal is in BFL
    ax_bfl = create_hpt_axis!(grid, 2, 1)
    if opt_idx[3] > 3
        ax_bfl.title = "BFL & batch size = $(EXPERIMENT_BATCH_SIZES[opt_idx[3]])"
        plot_hpt_heatmap!(ax_bfl, inf_times[:, :, opt_idx[3]]; colormap, colorrange)
    else
        ax_bfl.title = "BFL & batch size = 25k"
        plot_hpt_heatmap!(ax_bfl, inf_times[:, :, 5]; colormap, colorrange)  # fallback
    end

    # Add markers and annotations for BCE panel
    mark_optimal_model!(ax_bce, opt_bce_idx; color=:red, marker=:diamond, markersize=10)
    annotate_inference_time!(ax_bce, opt_bce_idx, inf_times[opt_bce_idx]; color=:red)
    mark_optimal_model!(ax_bce, opt_fast_bce_idx; color=:blue, marker=:diamond, markersize=8)
    annotate_inference_time!(ax_bce, opt_fast_bce_idx, inf_times[opt_fast_bce_idx];
                             color=:blue, align=(:left, :bottom))

    # Add markers and annotations for BFL panel
    if opt_idx[3] > 3
        mark_optimal_model!(ax_bfl, opt_idx; color=:red, marker=:star5, markersize=12)
        annotate_inference_time!(ax_bfl, opt_idx, inf_times[opt_idx]; color=:red)
    end
    if opt_fast_idx[3] > 3
        mark_optimal_model!(ax_bfl, opt_fast_idx; color=:blue, marker=:star5, markersize=10)
        annotate_inference_time!(ax_bfl, opt_fast_idx, inf_times[opt_fast_idx];
                                 color=:blue, align=(:left, :bottom))
    end

    # Horizontal colorbar at bottom
    Colorbar(grid[3, 1]; colormap, limits=colorrange,
             label = "Inference time /ms",
             vertical = false)

    return (ax_bce, ax_bfl)
end

#= ══════════════════════════════════════════════════════════════════════════════
   MAIN FIGURE ASSEMBLY
   ══════════════════════════════════════════════════════════════════════════════ =#

function create_hpt_classifier_figure()
    # Load all data at once
    qasm_loss, loss_curves, inf_times = load_all_experiments(EXPERIMENT_DIRS, INF_TIMES_PATH, N_LAYERS, N_NEURONS)

    # Find optimal models
    opt_idx = find_optimal_model(qasm_loss)
    opt_bce_idx = find_optimal_bce_model(qasm_loss)
    opt_fast_idx = find_fastest_qualified_model(qasm_loss, inf_times)
    opt_fast_bce_idx = find_fastest_qualified_model(qasm_loss, inf_times; bce_only=true)

    # Create figure with theme
    with_theme(create_figure_theme()) do
        fig = Figure(size = (mm_to_px(FIGURE_WIDTH_MM), mm_to_px(FIGURE_HEIGHT_MM)))

        # Set up main grid layout
        grid_a = fig[1, 1:3] = GridLayout()
        grid_b = fig[2, 1:2] = GridLayout()
        grid_c = fig[2, 3] = GridLayout()

        # Build panels
        build_panel_a!(grid_a, qasm_loss;
                       opt_idx, opt_bce_idx, opt_fast_idx, opt_fast_bce_idx)

        build_panel_b!(grid_b, loss_curves;
                       opt_idx, opt_bce_idx, opt_fast_idx, opt_fast_bce_idx)

        build_panel_c!(grid_c, inf_times;
                       opt_idx, opt_bce_idx, opt_fast_idx, opt_fast_bce_idx)

        return fig
    end
end

#= ══════════════════════════════════════════════════════════════════════════════
   SCRIPT EXECUTION
   ══════════════════════════════════════════════════════════════════════════════ =#

function main()
    fig = create_hpt_classifier_figure()
    save(joinpath(@__DIR__, "hpt_classifier.pdf"), fig; dpi=300)
    return fig
end

# Only run when script is executed directly (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
