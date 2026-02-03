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
err = []
rel_err = []
pos = []
dodge = []

for i in 1:14
    for j in 1:6
         push!(err, me_no_zeros(𝐗_ŷ[j, i, :], 𝐗_ss[j, i, :], agg=identity))
         push!(rel_err, re_no_zeros(𝐗_ŷ[j, i, :], 𝐗_ss[j, i, :], agg=identity) .* 100)
         push!(pos, ones(length(err[end])) .* i)
         push!(dodge, ones(length(err[end])) .* (7 - j))
    end
end
err = vcat(err...)
rel_err = vcat(rel_err...)
pos = vcat(pos...)
dodge = Int.(vcat(dodge...))
colors_el = Makie.cgrad(:bamako, 6, categorical = true).colors
color_el = [colors_el[Int(d)] for d in dodge]
colors_phase = Makie.cgrad(:lipari, 16, categorical = true).colors


# PLOTTING
#-----------------------------------------------------------------------
# Figure dimensions (A4 paper)
const FIGURE_DPI = 72;
const FIGURE_WIDTH_MM = 210.0;
const FIGURE_HEIGHT_MM = 290.0;
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



with_theme(create_figure_theme()) do
    fig = Figure(size = (mm_to_px(FIGURE_WIDTH_MM), mm_to_px(FIGURE_HEIGHT_MM)))
    ax_left = Axis(fig[1, 1], xlabel=L"Error\ in\ 𝐗\ [molmol^{-1}]")
    ax_right = Axis(fig[1, 2], xlabel=L"Relative\ Error\ in\ 𝐗 [\%]")

    # VIOLIN PLOTS OF ERRORS IN 𝐗 PREDICTIONS
    vspan!(ax_left, -0.01, 0.01, color = RGBf(0.95, 0.95, 0.95))
    text!(ax_left, "-0.01", position = (-0.012, -0.1), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax_left, "+0.01", position = (0.012, -0.1), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax_left, [0.0], color = :black, linewidth = 1, linestyle = :dash)
    violin!(ax_left, pos .* (-2/5), err,
            dodge = dodge,
            boundary = (-0.06, 0.06),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color_el,
            orientation=:horizontal)

    for i in 1:14
        text!(ax_left, "$(ss_names[i])",
          position = (-0.059, -i/5 * 2 + 0.2),
          color = colors_phase[i],
          align = (:left, :center),
          fontsize = 10)
        text!(ax_left, "MAE = $(me_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg= (x -> median(abs.(x)))) |> x -> round(x, digits=3))",
              position = (0.059, -i/5 * 2 + 0.2),
              color = colors_phase[i],
              align = (:right, :center),
              fontsize = 10)
    end

    for el_idx in 1:6
        text!(ax_left, [L"\textrm{SiO_2}"; L"\textrm{CaO}"; L"\textrm{Al_2O_3}"; L"\textrm{FeO}"; L"\textrm{MgO}"; L"\textrm{Na_2O}"][el_idx],
              position = (-0.045, -0.15-(el_idx*0.4/6)),
              color = colors_el[7 - el_idx],
              align = (:left, :center),
              fontsize = 8)
    end

    hideydecorations!(ax_left)
    ax_left.xgridvisible = true
    ax_left.xminorticksvisible = true
    ax_left.xticks = [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]
    ax_left.xminorticks = -0.06:0.01:0.06

    ylims!(ax_left, -5.8, 0.0)

    # VIOLIN PLOTS OF RELATIVE ERRORS IN 𝐗 PREDICTIONS
    vspan!(ax_right, -5.0, 5.0, color = RGBf(0.95, 0.95, 0.95))
    text!(ax_right, "-5%", position = (-5.5, -0.1), align = (:right, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    text!(ax_right, "+5%", position = (5.5, -0.1), align = (:left, :center), fontsize = 8, color = RGBf(0.75, 0.75, 0.75))
    vlines!(ax_right, [0.0], color = :black, linewidth = 1, linestyle = :dash)
    violin!(ax_right, pos .* (-2/5), rel_err,
            dodge = dodge,
            boundary = (-50.0, 50.0),
            scale = :width,
            npoints = 10000,
            width = 0.4,
            color = color_el,
            orientation=:horizontal)

    for i in 1:14
        text!(ax_right, "MRE = $(re_no_zeros(𝐗_ŷ[:, i, :], 𝐗_ss[:, i, :], agg= (x -> median(abs.(x)))) .* 100 |> x -> round(x, digits=2))%",
              position = (49.0, -i/5 * 2 + 0.2),
              color = colors_phase[i],
              align = (:right, :center),
              fontsize = 10)
    end

    hideydecorations!(ax_right)
    ax_right.xgridvisible = true
    ax_right.xminorticksvisible = true
    ax_right.xticks = -50:10:50
    ax_right.xminorticks = -50:5:50

    ylims!(ax_right, -5.8, 0.0)

    fig
    save(joinpath("05_figures_supplementary", "fig_elementwise_misfit", "fig_elementwise_misfit.pdf"), fig)
end
