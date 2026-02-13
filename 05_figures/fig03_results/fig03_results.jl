using JLD2, CSV, DataFrames
using Statistics
using Flux
using Sprout
using Sprout.misfit: me_no_zeros, me_trivial_zeros, mae_no_zeros, mae_trivial_zeros, re_no_zeros, re_trivial_zeros, mre_no_zeros, mre_trivial_zeros
using Sprout.misfit: closure_condition, mass_balance_abs_misfit, mass_balance_rel_misfit, mass_residual
using CairoMakie

phase_names = [p for (i, p) in enumerate(vcat(PP, SS)) if i ∉ Sprout.IDX_OF_PHASES_NEVER_STABLE];
ss_names = phase_names[7:end];

# LOAD DATA
#-----------------------------------------------------------------------
DATA_DIR = joinpath("data", "generated_dataset");
x = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_test_x.csv"), DataFrame);
y = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_test_y.csv"), DataFrame);
x, 𝑣, 𝐗_ss, _, _, _ = preprocess_data(x, y);

# load normalisers
@load joinpath("models", "surrogate", "normalisers.jld2") xNorm 𝐗Scale 𝑣Scale;

x_norm = xNorm(x);

# LOAD MODEL
#-----------------------------------------------------------------------
masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));
m_classifier = create_classifier_model(2, 200, 8, 20);
# Load REGRESSOR
m = create_model_pretrained_classifier(1//2, 4, 400, masking_f, m_classifier);
model_state = JLD2.load(joinpath("models", "surrogate", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m, model_state);


# PREDICT
#-----------------------------------------------------------------------
(𝑣_ŷ_, 𝐗_ŷ_) = m(x_norm);

# DESCALING
𝑣_ŷ, 𝐗_ŷ = descale(𝑣Scale, 𝑣_ŷ_), descale(𝐗Scale, 𝐗_ŷ_);

# CALCULATE METRICS
#-----------------------------------------------------------------------
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

med_ae_𝐗 = [me_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg=x -> median(abs.(x))) for i in 1:14];
med_re_𝐗 = [re_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg=x -> median(abs.(x))) * 100 for i in 1:14];

# element-wise metrics for 𝐗 in garnet
idx_grt = findfirst(ss_names .== "gtmj")
el_wise_err_𝐗 = [me_no_zeros(𝐗_ŷ[i, idx_grt, :], 𝐗_ss[i, idx_grt, :], agg=identity) for i in 1:6]

# Mass-balance metrics
mass_balance_misfit = Sprout.misfit.recalculate_bulk((𝑣_ŷ, 𝐗_ŷ)) .- x[3:end, :, :]

# PLOTTING
#-----------------------------------------------------------------------
# Figure dimensions (A4 paper)
const FIGURE_DPI = 72;
const FIGURE_WIDTH_MM = 210.0;
const FIGURE_HEIGHT_MM = 290.0 * 0.6;
const HEATMAP_DPI = 600;
mm_to_px(mm, dpi=FIGURE_DPI) = round(Int, mm * (1/25.4) * dpi);


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


function violin_plots_𝐗(grid_layout, err_𝐗, rel_err_𝐗, med_ae_𝐗, med_re_𝐗, ss_names)
    ax1 = Axis(grid_layout[1, 1], xlabel=L"Error\ in\ 𝐗\ [molmol^{-1}]")

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
            boundary = (-0.06, 0.06),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color,
            orientation=:horizontal)
    text!(ax1, "MAE", position = (0.06, 0.2), align = (:right, :center), fontsize = 10)
    for i in 1:14
        text!(ax1, ss_names[i],
              position = (-0.06, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:left, :center),
              fontsize = 10)
        text!(ax1, "$(round(med_ae_𝐗[i], sigdigits=1))",
              position = (0.06, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:right, :center),
              fontsize = 10)
    end
    hideydecorations!(ax1)
    ax1.xgridvisible = true
    ax1.xminorticksvisible = true
    ax1.xticks = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
    ax1.xminorticks = -0.06:0.01:0.06


    ax2 = Axis(grid_layout[1, 2], xlabel=L"Relative\ Error\ in\ 𝐗 [\%]")

    vspan!(ax2, -5.0, 5.0, color = RGBf(0.95, 0.95, 0.95))
    text!(ax2, "-5%", position = (-5.5, 0.2), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax2, "+5%", position = (5.5, 0.2), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax2, [0.0], color = :black, linewidth = 1, linestyle = :dash)

    violin!(ax2, pos .* (-2/5), rel_err,
            boundary = (-100.0, 100.0),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color,
            orientation=:horizontal)
    text!(ax2, "MRE", position = (29.0, 0.2), align = (:right, :center), fontsize = 10)
    for i in 1:14
        text!(ax2, "$(round(med_re_𝐗[i], sigdigits=1))%",
              position = (29.0, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:right, :center),
              fontsize = 10)
    end

    hideydecorations!(ax2)
    ax2.xgridvisible = true
    ax2.xminorticksvisible = true
    xlims!(ax2, -30.0, 30.0)
    ax2.xticks = -30:10:30
    ax2.xminorticks = -30:5:30
end

function plot_mass_balance_misfit(grid_layout, mass_balance_misfit)
    mb_misfit = vcat(mass_balance_misfit...)
    pos = vcat([ones(size(mass_balance_misfit, 3)) .* i for i in 1:6]...)
    colors = Makie.cgrad(:bamako, 6, categorical = true).colors
    color = [colors[Int(p)] for p in pos]

    ax1 = Axis(grid_layout[1, 1], xlabel=L"Mass\ Balance\ Misfit\ [molmol^{-1}]")

    vspan!(ax1, -0.002, 0.002, color = RGBf(0.95, 0.95, 0.95))
    text!(ax1, "-0.002", position = (-0.0022, 0.2), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax1, "+0.002", position = (0.0022, 0.2), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax1, [0.0], color = :black, linewidth = 1, linestyle = :dash)

    violin!(ax1, pos .* (-2/5), mb_misfit,
            boundary = (-0.01, 0.01),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color,
            orientation=:horizontal)

    for i in 1:6
        text!(ax1, [L"SiO_2"; L"CaO"; L"Al_2O_3"; L"FeO"; L"MgO"; L"Na_2O"][i],
              position = (-0.01, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:left, :center),
              fontsize = 10)
    end

    hideydecorations!(ax1)
    # ax1.xgridvisible = true
    # ax1.xminorticksvisible = true
    # ax1.xticks = [-0.02, -0.01, 0.0, 0.01, 0.02]
    # ax1.xminorticks = -0.03:0.005:0.03
end


function plot_elementwise_misfit_in_𝐗(grid_layout, el_wise_err_𝐗)
    ax1 = Axis(grid_layout[1, 1], xlabel=L"Error\ in\ 𝐗^{Grt}\ [molmol^{-1}]")

    # Flatten error distributions and create position vectors for violin plots
    err = vcat(el_wise_err_𝐗...)
    pos = vcat([ones(length(el_wise_err_𝐗[i])) .* i for i in 1:6]...)
    colors = Makie.cgrad(:bamako, 6, categorical = true).colors
    color = [colors[Int(p)] for p in pos]

    vspan!(ax1, -0.002, 0.002, color = RGBf(0.95, 0.95, 0.95))
    text!(ax1, "-0.002", position = (-0.0022, 0.2), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax1, "+0.002", position = (0.0022, 0.2), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax1, [0.0], color = :black, linewidth = 1, linestyle = :dash)

    violin!(ax1, pos .* (-2/5), err,
            boundary = (-0.01, 0.01),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color,
            orientation=:horizontal)

    for i in 1:6
        text!(ax1, [L"SiO_2"; L"CaO"; L"Al_2O_3"; L"FeO"; L"MgO"; L"Na_2O"][i],
              position = (-0.01, -i/5 * 2 + 0.2),
              color = colors[i],
              align = (:left, :center),
              fontsize = 10)
    end
    hideydecorations!(ax1)
end


with_theme(create_figure_theme()) do
    fig = Figure(size = (mm_to_px(FIGURE_WIDTH_MM), mm_to_px(FIGURE_HEIGHT_MM)))

    # Create panels
    grid_upper         = fig[1:2, 1:2]   = GridLayout()
    grid_lower_left    = fig[3, 1]   = GridLayout()
    grid_lower_right   = fig[3, 2]   = GridLayout()

    # VIOLIN PLOTS OF ERRORS IN 𝐗 PREDICTIONS
    violin_plots_𝐗(grid_upper, err_𝐗, rel_err_𝐗, med_ae_𝐗, med_re_𝐗, ss_names)

    # VIOLIN PLOTS OF MASS BALANCE MISFITS
    plot_mass_balance_misfit(grid_lower_left, mass_balance_misfit)

    # VIOLIN PLOTS OF ELEMENT-WISE ERRORS IN GARNET
    plot_elementwise_misfit_in_𝐗(grid_lower_right, el_wise_err_𝐗)

    save(joinpath("05_figures", "fig03_results", "fig03_results.pdf"), fig)
    fig
end
