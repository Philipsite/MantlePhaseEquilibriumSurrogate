using JLD2, CSV, DataFrames
using Statistics
using Flux
using Sprout
using CairoMakie

const HEATMAP_DPI = 600;
colormap = :acton
# LOAD MODEL
#-----------------------------------------------------------------------
masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));
m_classifier = create_classifier_model(2, 200, 8, 20);
m = create_model_pretrained_classifier(1//2, 4, 400, masking_f, m_classifier);
model_state = JLD2.load(joinpath("models", "surrogate", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m, model_state);

# extract only the classifier part for assemblage predictions
m_classifier = m.layers[1];

# load normalisers
@load joinpath("models", "surrogate", "normalisers.jld2") xNorm 𝐗Scale 𝑣Scale;


# Compositions after Xu et al. (2008)
Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
HARZBURGITE  = Float32[36.04, 0.79, 0.65, 5.97, 56.54, 0.00]  # Modified Harzburgite from Xu et al. (2008)
x_Hrz = HARZBURGITE ./ sum(HARZBURGITE)
BASALT = Float32[51.75, 13.88, 10.19, 7.06, 14.94, 2.18]
x_Bas = BASALT ./ sum(BASALT)

P_range_kbar = (10., 400.)
T_range_C = (500., 2500.)

asm_grid_Hrz, var_vec_grid_Hrz = generate_mineral_assemblage_diagram(P_range_kbar, T_range_C, x_Hrz, 1000, m_classifier, xNorm)
asm_grid_Bas, var_vec_grid_Bas = generate_mineral_assemblage_diagram(P_range_kbar, T_range_C, x_Bas, 1000, m_classifier, xNorm)

"""
Save the mineral assemblage diagram heatmap as a high-resolution PNG file.
Returns the path to the saved image.
"""
function save_heatmap_raster(asm_grid, var_vec_grid, P_bounds, T_bounds, colormap;
                              output_path, dpi=HEATMAP_DPI, size_px=(1200, 1200))
    # Create a standalone figure with just the heatmap (no axis decorations)
    fig_hm = Figure(size=size_px, figure_padding=0)
    ax_hm = Axis(fig_hm[1, 1], aspect=1.0)
    hidedecorations!(ax_hm)
    hidespines!(ax_hm)

    palette = cgrad(colormap)
    n = size(asm_grid)[1]
    P = range(P_bounds[1], P_bounds[2], length=n)
    T = range(T_bounds[1], T_bounds[2], length=n)
    P_rev = reverse(P)

    heatmap!(ax_hm, T, P_rev, var_vec_grid'; colormap=palette, colorrange=(2, 7), interpolate=false)

    save(output_path, fig_hm; px_per_unit=dpi/300)
    return output_path
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

fig = Figure(size = (400, 800))
gt = fig[1, 1] = GridLayout()
gb = fig[2, 1] = GridLayout()

hrz_mad = joinpath("05_figures_supplementary", "fig_mineral_assemblage diagrams", "hrz_mad.png")
bas_mad = joinpath("05_figures_supplementary", "fig_mineral_assemblage diagrams", "bas_mad.png")

hrz_ax = mineral_assemblage_diagram(gt, asm_grid_Hrz, var_vec_grid_Hrz, P_range_kbar, T_range_C;
                                    heatmap_path = hrz_mad, heatmap_dpi=HEATMAP_DPI)
hrz_ax.title = "Harzburgite (Xu et al., 2008)"

bas_ax = mineral_assemblage_diagram(gb, asm_grid_Bas, var_vec_grid_Bas, P_range_kbar, T_range_C;
                                    heatmap_path = bas_mad, heatmap_dpi=HEATMAP_DPI)
bas_ax.title = "Basalt (Xu et al., 2008)"

fig
save(joinpath("05_figures_supplementary", "fig_mineral_assemblage diagrams","mineral_assemblage_diagrams.pdf"), fig)
