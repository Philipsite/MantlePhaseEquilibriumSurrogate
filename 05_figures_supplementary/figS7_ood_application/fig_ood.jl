using JLD2, CSV, DataFrames, FileIO
using Statistics
using Flux
using Sprout
using Sprout.misfit: loss_asm, fraction_mismatched_asm, fraction_mismatched_phases
using Sprout.misfit: me_no_zeros, me_trivial_zeros, mae_no_zeros, mae_trivial_zeros, re_no_zeros, re_trivial_zeros, mre_no_zeros, mre_trivial_zeros
using Sprout.misfit: closure_condition, mass_balance_abs_misfit, mass_balance_rel_misfit, mass_residual
using CairoMakie

phase_names = [p for (i, p) in enumerate(vcat(PP, SS)) if i ∉ Sprout.IDX_OF_PHASES_NEVER_STABLE];
ss_names = phase_names[7:end];

# LOAD DATA
#-----------------------------------------------------------------------
DATA_DIR = joinpath("data", "ood_dataset");
x = CSV.read(joinpath(DATA_DIR, "AOC_test_data_x.csv"), DataFrame);
y = CSV.read(joinpath(DATA_DIR, "AOC_test_data_y.csv"), DataFrame);
x, 𝑣, 𝐗_ss, _, _, _ = preprocess_data(x, y);

# load normalisers
@load joinpath("models", "surrogate", "normalisers.jld2") xNorm 𝐗Scale 𝑣Scale;

x_norm = xNorm(x);

# LOAD MODEL
#-----------------------------------------------------------------------
masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));
# Load CLASSIFIER
m_classifier = create_classifier_model(2, 200, 8, 20);
model_state = JLD2.load(joinpath("models", "classifier", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m_classifier, model_state);

# Load REGRESSOR
m = create_model_pretrained_classifier(1//2, 4, 400, masking_f, m_classifier);
model_state = JLD2.load(joinpath("models", "surrogate", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m, model_state);

# extract only the classifier part for assemblage predictions
m_classifier = m.layers[1];

# PREDICT
#-----------------------------------------------------------------------
(𝑣_ŷ_, 𝐗_ŷ_) = m(x_norm);

# DESCALING
𝑣_ŷ, 𝐗_ŷ = descale(𝑣Scale, 𝑣_ŷ_), descale(𝐗Scale, 𝐗_ŷ_);

AOC_KE03 = Float32[0.5344, 0.1515, 0.0770, 0.1120, 0.1006, 0.0242];

asm_grid, var_vec_grid = generate_mineral_assemblage_diagram((10., 400.), (500., 2500.), AOC_KE03, 1000, m_classifier, xNorm);

ŷ_asm = m_classifier(x_norm) .|> >=(0.5);
y_asm = 𝑣 .> 0.;

# CALCULATE METRICS
#-----------------------------------------------------------------------
# Classifier metrics
mean_loss_asm = loss_asm(ŷ_asm, y_asm);
println("Mean (1-Q_{asm}): ", "$mean_loss_asm")
tp = vec(sum((ŷ_asm .== 1) .& (y_asm .== 1), dims=3));
fp = vec(sum((ŷ_asm .== 1) .& (y_asm .== 0), dims=3));
tn = vec(sum((ŷ_asm .== 0) .& (y_asm .== 0), dims=3));
fn = vec(sum((ŷ_asm .== 0) .& (y_asm .== 1), dims=3));

n_samples = size(y_asm, 3);
accuracy_phasewise = (tp .+ tn) ./ n_samples .* 100;
recall_phasewise = tp ./ (tp .+ fn) .* 100;

# Regressor metrics for 𝑣
median_ae_𝑣 = mae_no_zeros(𝑣_ŷ, 𝑣, agg=median);
median_re_𝑣 = mre_no_zeros(𝑣_ŷ, 𝑣, agg=median);
println("MAE 𝑣: $median_ae_𝑣");
println("MRE 𝑣: $(median_re_𝑣 * 100)%");

# a bit an awkward if else to avoid issues with phases that are never predicted..
med_ae_𝑣 = [me_no_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=x -> length(x) > 0 ? median(abs.(x)) : missing) for i in 1:20];
med_re_𝑣 = [re_no_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=x -> length(x) > 0 ? median(abs.(x)) : missing) * 100 for i in 1:20];
med_ae_𝑣_trivial = [me_trivial_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=x -> length(x) > 0 ? median(abs.(x)) : missing) for i in 1:20];
med_re_𝑣_trivial = [re_trivial_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=x -> length(x) > 0 ? median(abs.(x)) : missing) * 100 for i in 1:20];
n_nonzero_𝑣 = [sum(𝑣[i, :, :] .!= 0.0) for i in 1:20];

# compare non zero with trivial zeros
println("Median AE 𝑣 (non-zero):")
[println("$p : $err") for (p, err) in zip(phase_names, med_ae_𝑣)];
println("Median AE 𝑣 (trivial zeros):")
[println("$p : $err") for (p, err) in zip(phase_names, med_ae_𝑣_trivial)];

# Error distributions for violin plots
err_𝑣 = [me_no_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=identity) for i in 1:20];
err_trivial_𝑣 = [me_trivial_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=identity) for i in 1:20];
rel_err_𝑣 = [re_no_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=identity) .* 100 for i in 1:20];
rel_trivial_𝑣 = [re_trivial_zeros(𝑣_ŷ[i, :, :], 𝑣[i, :, :], agg=identity) .* 100 for i in 1:20];

# Metrics for 𝐗
mae_𝐗 = mae_no_zeros(𝐗_ŷ, 𝐗_ss);
mre_𝐗 = mre_no_zeros(𝐗_ŷ, 𝐗_ss);
median_ae_𝐗 = mae_no_zeros(𝐗_ŷ, 𝐗_ss, agg=median);
median_re_𝐗 = mre_no_zeros(𝐗_ŷ, 𝐗_ss, agg=median);
println("MAE 𝐗: $mae_𝐗");
println("Median AE 𝐗: $median_ae_𝐗");
println("MRE 𝐗: $mre_𝐗%");
println("Median RE 𝐗: $(median_re_𝐗 * 100)%");

# Metrics for 𝐗 phase-wise
err_𝐗 = [me_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg=identity) for i in 1:14]
rel_err_𝐗 = [re_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg=identity) * 100 for i in 1:14]

med_ae_𝐗 = [me_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg=x -> length(x) > 0 ? median(abs.(x)) : missing) for i in 1:14];
med_re_𝐗 = [re_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg=x -> length(x) > 0 ? median(abs.(x)) : missing) * 100 for i in 1:14];
# PLOTTING
#-----------------------------------------------------------------------
# Figure dimensions (A4 paper)
const FIGURE_DPI = 72;
const FIGURE_WIDTH_MM = 210.0;
const FIGURE_HEIGHT_MM = 290.0;
const HEATMAP_DPI = 600;
mm_to_px(mm, dpi=FIGURE_DPI) = round(Int, mm * (1/25.4) * dpi);

"""
Save the mineral assemblage diagram heatmap as a high-resolution PNG file.
Returns the path to the saved image.
"""
function save_heatmap_raster(asm_grid, var_vec_grid, P_bounds, T_bounds, colormap_sym;
                              output_path, dpi=HEATMAP_DPI, size_px=(1200, 1200))
    # Create a standalone figure with just the heatmap (no axis decorations)
    fig_hm = Figure(size=size_px, figure_padding=0)
    ax_hm = Axis(fig_hm[1, 1], aspect=1.0)
    hidedecorations!(ax_hm)
    hidespines!(ax_hm)

    palette = cgrad(colormap_sym)
    n = size(asm_grid)[1]
    P = range(P_bounds[1], P_bounds[2], length=n)
    T = range(T_bounds[1], T_bounds[2], length=n)
    P_rev = reverse(P)

    heatmap!(ax_hm, T, P_rev, var_vec_grid'; colormap=palette, colorrange=(2, 7), interpolate=false)

    save(output_path, fig_hm; px_per_unit=dpi/300)
    return output_path
end

# Colors for TP, FP, TN, FN
pie_colors = [:green, :red, :gray, :orange];
pie_colors = [RGBf(131/255, 175/255, 116/255), RGBf(147/255, 75/255, 75/255),RGBf(125/255, 125/255, 125/255), RGBf(147/255, 126/255, 75/255)];

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
            spinewidth = 0.5,
            xlabelpadding = 5,
            ylabelpadding = 5,
        )
    )
end


function pie_charts(grid_layout, tp, fp, tn, fn, accuracy_phasewise, recall_phasewise, phase_names, pie_colors)
    for idx in 1:20
        row = div(idx - 1, 4) + 1
        col = mod(idx - 1, 4) + 1

        ax = Axis(grid_layout[row, col],
                  title="$(phase_names[idx])",
                  titlesize=10,
                  aspect=DataAspect(),
                  backgroundcolor = RGBf(0.98, 0.98, 0.98))
        hidedecorations!(ax)
        hidespines!(ax)


        vals = [tp[idx], fp[idx], tn[idx], fn[idx]]

        if sum(vals) > 0
            pie!(ax, vals, color=pie_colors,
                 radius=1.0,
                 inner_radius=0.5,  # donut style
                 strokewidth=1,
                 strokecolor=:white)

            # Add accuracy and recall in center
            text!(ax, 0, 0, text="$(round(Int, accuracy_phasewise[idx]))%",
                  align=(:center, :bottom), fontsize=8)
            if !isnan(recall_phasewise[idx])
                text!(ax, 0, 0, text="$(round(Int, recall_phasewise[idx]))%",
                    align=(:center, :top), fontsize=8)
            else
                text!(ax, 0, 0, text="N/A",
                    align=(:center, :top), fontsize=8)
            end
        end
    end

    colgap!(grid_layout, 1)
    rowgap!(grid_layout, 3)
end


function mineral_assemblage_diagram(grid_layout, asm_grid, var_vec_grid, P_bounds, T_bounds;
                                    heatmap_path, heatmap_dpi=HEATMAP_DPI)
    # Save the heatmap as a high-resolution raster
    save_heatmap_raster(asm_grid, var_vec_grid, P_bounds, T_bounds, :acton;
                        output_path=heatmap_path, dpi=heatmap_dpi)

    # Create axis with proper labels and limits
    ax = Axis(grid_layout[1, 1],
              aspect=1.0,
              xlabel=L"Temperature\ [°C]",
              ylabel=L"Pressure\ [GPa]")

    # Load the raster image
    heatmap_img = load(heatmap_path)
    image!(ax, T_bounds[1]..T_bounds[2], P_bounds[1]..P_bounds[2], rotr90(heatmap_img))

    # Add assemblage labels at centroids
    n = size(asm_grid)[1]
    P = range(P_bounds[1], P_bounds[2], length=n)
    T = range(T_bounds[1], T_bounds[2], length=n)
    P_rev = reverse(P)
    P_grid = Matrix{Float32}(repeat(P_rev, 1, n))
    T_grid = Matrix{Float32}(repeat(T', n, 1))

    unique_values = unique(vec(asm_grid))
    for value in unique_values
        indices = findall(x -> x == value, asm_grid)
        mean_i = mean([idx[1] for idx in indices])
        mean_j = mean([idx[2] for idx in indices])
        centroid_x = T_grid[1, Int(round(mean_j))]
        centroid_y = P_grid[Int(round(mean_i)), 1]
        text!(ax, centroid_x, centroid_y, text=value,
              align=(:center, :center), fontsize=8, color=:white)
    end

    # Set axis labels to GPa values
    ax.yticks = ([10, 100, 200, 300, 400], ["1", "10", "20", "30", "40"])

    return ax
end


function pie_chart_legend(grid_layout, pie_colors)
    # Create a legend pie chart with example values
    ax_legend1 = Axis(grid_layout[1, 1],
                      aspect=DataAspect(),
                      backgroundcolor=RGBf(0.98, 0.98, 0.98))
    hidedecorations!(ax_legend1)
    hidespines!(ax_legend1)

    # Example pie chart with representative segments for legend
    legend_vals = [2//8, 1//8, 4//8, 1//8]
    pie!(ax_legend1, legend_vals, color=pie_colors,
         radius=1.0,
         inner_radius=0.5,
         strokewidth=1,
         strokecolor=:white)

    # Add "Accuracy" and "Recall" labels in center
    text!(ax_legend1, 0, 0, text="Accuracy",
          align=(:center, :bottom), fontsize=8)
    text!(ax_legend1, 0, 0, text="Recall",
          align=(:center, :top), fontsize=8)

    # Create axis for category labels
    ax_legend2 = Axis(grid_layout[1, 2],
                      backgroundcolor=RGBf(0.98, 0.98, 0.98))
    hidedecorations!(ax_legend2)
    hidespines!(ax_legend2)

    # Add category labels
    legend_labels = ["True Positive", "False Positive", "True Negative", "False Negative"]
    for (i, label) in enumerate(legend_labels)
        text!(ax_legend2, 0., 0.6 - (i-1)*0.4, text=label,
              align=(:left, :center), fontsize=8, color=pie_colors[i])
    end

    return (ax_legend1, ax_legend2)
end


function violin_plots_𝑣(grid_layout, err_𝑣, rel_err_𝑣, med_ae_𝑣, med_re_𝑣, phase_names)
    ax1 = Axis(grid_layout[1, 1], xlabel=L"Error\ in\ 𝑣\ [molmol^{-1}]")

    # Flatten error distributions and create position vectors for violin plots
    err = vcat(err_𝑣...)
    rel_err = vcat(rel_err_𝑣...)
    pos = vcat([ones(length(err_𝑣[i])) .* i for i in 1:20]...)
    colors = Makie.cgrad(:bamO, 20, categorical = true).colors
    color = [colors[Int(p)] for p in pos]

    vspan!(ax1, -0.01, 0.01, color = RGBf(0.95, 0.95, 0.95))
    text!(ax1, "-0.01", position = (-0.012, 0.2), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax1, "+0.01", position = (0.012, 0.2), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax1, [0.0], color = :black, linewidth = 1, linestyle = :dash)

    violin!(ax1, pos .* (-2/5), err,
            boundary = (-0.1, 0.1),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color,
            orientation=:horizontal)
    text!(ax1, "MAE", position = (0.09, 0.2), align = (:right, :center), fontsize = 10)
    for i in 1:20
        text!(ax1, phase_names[i],
              position = (-0.09, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:left, :center),
              fontsize = 10)
        text!(ax1, "$(round(med_ae_𝑣[i], sigdigits=1))",
              position = (0.09, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:right, :center),
              fontsize = 10)
    end
    hideydecorations!(ax1)
    ax1.xgridvisible = true
    ax1.xminorticksvisible = true
    ax1.xminorticks = -0.1:0.01:0.1
end



function violin_plots_𝐗(grid_layout, err_𝐗, rel_err_𝐗, med_ae_𝐗, med_re_𝐗, ss_names)
    ax1 = Axis(grid_layout[1, 2], xlabel=L"Error\ in\ 𝐗\ [molmol^{-1}]")

    # Flatten error distributions and create position vectors for violin plots
    err = vcat(err_𝐗...)
    rel_err = vcat(rel_err_𝐗...)
    pos = vcat([ones(length(err_𝐗[i])) .* i for i in 1:14]...)
    colors = Makie.cgrad(:lipari, 16, categorical = true).colors
    color = [colors[Int(p) + 1] for p in pos]

    vspan!(ax1, -0.01, 0.01, color = RGBf(0.95, 0.95, 0.95))
    text!(ax1, "-0.01", position = (-0.012, 0.2), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax1, "+0.01", position = (0.012, 0.2), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax1, [0.0], color = :black, linewidth = 1, linestyle = :dash)

    violin!(ax1, pos .* (-2/5), err,
            boundary = (-0.1, 0.1),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color,
            orientation=:horizontal)
    text!(ax1, "MAE", position = (0.1, 0.2), align = (:right, :center), fontsize = 10)
    for i in 1:14
        text!(ax1, ss_names[i],
              position = (-0.1, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:left, :center),
              fontsize = 10)
        text!(ax1, "$(round(med_ae_𝐗[i], sigdigits=1))",
              position = (0.1, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:right, :center),
              fontsize = 10)
    end
    hideydecorations!(ax1)
    ax1.xgridvisible = true
    ax1.xminorticksvisible = true
    ax1.xticks = [-0.1, -0.05, 0.0, 0.05, 0.1]
    ax1.xminorticks = -0.1:0.01:0.1
end

with_theme(create_figure_theme()) do
    fig = Figure(size = (mm_to_px(FIGURE_WIDTH_MM), mm_to_px(FIGURE_HEIGHT_MM)))

    # Create panels
    grid_left    = fig[1, 1]   = GridLayout()
    grid_PIES    = fig[1, 2] = GridLayout()
    grid_𝑣_error = fig[2, 1:2]   = GridLayout()

    # Split left column into MAD (top) and legend (bottom)
    grid_MAD    = grid_left[1, 1] = GridLayout()
    grid_legend = grid_left[2, 1] = GridLayout()
    rowsize!(grid_left, 2, Auto(0.1))  # Legend takes less vertical space

    # MINERAL ASSEMBLAGE DIAGRAM
    heatmap_path = joinpath("05_figures_supplementary", "figS7_ood_application", "fig_heatmap.png")
    mineral_assemblage_diagram(grid_MAD, asm_grid, var_vec_grid, (10., 400.), (500., 2500.);
                               heatmap_path=heatmap_path, heatmap_dpi=HEATMAP_DPI)

    # PIE CHART LEGEND
    pie_chart_legend(grid_legend, pie_colors)

    # PIE CHARTS OF CLASSIFIER METRICS
    pie_charts(grid_PIES, tp, fp, tn, fn, accuracy_phasewise, recall_phasewise, phase_names, pie_colors)

    # VIOLIN PLOTS OF ERRORS IN 𝑣 PREDICTIONS
    violin_plots_𝑣(grid_𝑣_error, err_𝑣, rel_err_𝑣, med_ae_𝑣, med_re_𝑣, phase_names)
    # VIOLIN PLOTS OF ERRORS IN 𝐗 PREDICTIONS
    violin_plots_𝐗(grid_𝑣_error, err_𝐗, rel_err_𝐗, med_ae_𝐗, med_re_𝐗, ss_names)

    save(joinpath("05_figures_supplementary", "figS7_ood_application", "fig_ood.pdf"), fig)
    fig
end
