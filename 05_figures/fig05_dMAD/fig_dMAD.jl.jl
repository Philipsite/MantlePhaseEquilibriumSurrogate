using JLD2
using LinearAlgebra
using CairoMakie
using FileIO

# load arrays of Jacobians
@load joinpath("data", "jacobians_classifier", "Js_PYR.jld2") Js_PYR
@load joinpath("data", "jacobians_classifier", "Js_HRZ.jld2") Js_HRZ
@load joinpath("data", "jacobians_classifier", "Js_BAS.jld2") Js_BAS


# DEFINE COMPOSITIONS (after Xu et al. 2008) and P-T SPACE
#-----------------------------------------------------------------------
PYR_XU08 = Float32[0.3871, 0.0294, 0.0222, 0.0617, 0.4985, 0.0011];
HRZ_XU08 = Float32[0.3604, 0.0079, 0.0065, 0.0597, 0.5654, 0.0000];
BAS_XU08 = Float32[0.5175, 0.1388, 0.1019, 0.0706, 0.1494, 0.0218];

bulk_var_v = (BAS_XU08 - HRZ_XU08)
norm_bulk_var_vec = norm(bulk_var_v)
bulk_var_vec = bulk_var_v / norm_bulk_var_vec

# P-T space
n = 1000
P_bounds = (10.0f0, 400.0f0)    # GPa
T_bounds = (500.0f0, 2500.0f0)  # °C
P = range(P_bounds[1], P_bounds[2], length=n)
T = range(T_bounds[1], T_bounds[2], length=n)
# Reverse P for desired orientation
P_rev = reverse(P)


# SCALE JACOBIANS
#-------------------------------------------------------------------------
# Define a scale vector to compare partial derivatives
# Each component is the change associated with a "characteristic" parameter step
# As a reference a ∆T of 1 K is taken.
# Analogous, both pressure and the bulk compositional change are scaled to reflect
# a change in 2000 steps over the attained range, respectively.

T_range = T_bounds[2] - T_bounds[1]
steps = T_range / 1
P_range = P_bounds[2] - P_bounds[1]
COMP_range = norm_bulk_var_vec .* bulk_var_vec

norm_vec = abs.([P_range/steps, T_range/steps, COMP_range./steps...])

# turn this into a symmetrical diagonal scaling of the Jacobian rows
norm_mat = Diagonal(norm_vec)

Js_PYR_scaled = Array{Float32, 3}(undef, n^2, 20, 8)
for i in 1:n^2
    Js_PYR_scaled[i, :, :] .= Js_PYR[i, :, :] * norm_mat
end

# EXTRACT NORMS OF RELEVANT DIRECTIONAL DERIVATIVES
#-------------------------------------------------------------------------
extract_directional_norm = (x, v) -> reshape([norm(x[i, :, :] * v) for i in 1:size(x, 1)], n, n)

dfdp = extract_directional_norm(Js_PYR_scaled, [1, 0, 0, 0, 0, 0, 0, 0])
dfdt = extract_directional_norm(Js_PYR_scaled, [0, 1, 0, 0, 0, 0, 0, 0])
dfdv = extract_directional_norm(Js_PYR_scaled, [0, 0, bulk_var_vec...])

dfdSi = extract_directional_norm(Js_PYR_scaled, [0, 0, 1, 0, 0, 0, 0, 0])
dfdCa = extract_directional_norm(Js_PYR_scaled, [0, 0, 0, 1, 0, 0, 0, 0])
dfdAl = extract_directional_norm(Js_PYR_scaled, [0, 0, 0, 0, 1, 0, 0, 0])
dfdFe = extract_directional_norm(Js_PYR_scaled, [0, 0, 0, 0, 0, 1, 0, 0])
dfdMg = extract_directional_norm(Js_PYR_scaled, [0, 0, 0, 0, 0, 0, 1, 0])
dfdNa = extract_directional_norm(Js_PYR_scaled, [0, 0, 0, 0, 0, 0, 0, 1])

min_val = minimum([minimum(dfdp), minimum(dfdt), minimum(dfdv)])
max_val = maximum([maximum(dfdp), maximum(dfdt), maximum(dfdv)])
min_val_comp = minimum([minimum(dfdSi), minimum(dfdCa), minimum(dfdAl), minimum(dfdFe), minimum(dfdMg), minimum(dfdNa)])
max_val_comp = maximum([maximum(dfdSi), maximum(dfdCa), maximum(dfdAl), maximum(dfdFe), maximum(dfdMg), maximum(dfdNa)])
# adjust max val as to not have a few outliers dominate the color range
max_val_adj = 0.1
max_val_comp_adj = 0.05

# PLOTTING
#-------------------------------------------------------------------------
const HEATMAP_DPI = 600
const COLORMAP      = :lipari
const COLORMAP_COMP = :navia

function save_heatmap_raster(x, y, z; colormap, colorrange, output_path, dpi=HEATMAP_DPI, size_px=(1800, 1800))
    # Render a standalone high-resolution heatmap raster without axis decorations.
    fig_hm = Figure(size=size_px, figure_padding=0)
    ax_hm = Axis(fig_hm[1, 1], aspect=1.0)
    hidedecorations!(ax_hm)
    hidespines!(ax_hm)

    heatmap!(ax_hm, x, y, z; colormap=colormap, colorscale=sqrt, colorrange=colorrange)
    save(output_path, fig_hm; px_per_unit=dpi/300)
    return output_path
end

function axis_with_heatmap(grid_pos, x, y, z; title=nothing, colormap, colorrange, raster_name="heatmap", xlabel=nothing, ylabel=nothing)
    raster_path = joinpath("05_figures", "fig05_dMAD", "hm_rasters", "$(raster_name).png")
    save_heatmap_raster(x, y, z; colormap=colormap, colorrange=colorrange, output_path=raster_path)

    ax = Axis(grid_pos, aspect = 1.0)
    # Keep an invisible heatmap as color source for Colorbar.
    hm = heatmap!(ax, x, y, z; colormap=colormap, colorscale=sqrt, colorrange=colorrange, visible=false)
    raster_img = load(raster_path)
    image!(ax, minimum(x)..maximum(x), minimum(y)..maximum(y), rotr90(raster_img))
    if !isnothing(title)
        ax.title = title
    end
    if !isnothing(xlabel)
        ax.xlabel = xlabel
    end
    if !isnothing(ylabel)
        ax.ylabel = ylabel
    end
    return ax, hm
end


fig = Figure(size=(1000, 900))

mkpath(joinpath("05_figures", "fig05_dMAD", "hm_rasters"))

ax1, hm1 = axis_with_heatmap(fig[1, 1], T, P_rev, dfdp';
    # title = L"\text{\Vert \mathbf{J}_f v_P \Vert_2}",
    colormap = COLORMAP,
    colorrange = (min_val, max_val_adj),
    raster_name = "dfdp",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax2, hm2 = axis_with_heatmap(fig[1, 2], T, P_rev, dfdt';
    # title = L"\text{\Vert \mathbf{J}_f \hat{e}_T \Vert_2}",
    colormap = COLORMAP,
    colorrange = (min_val, max_val_adj),
    raster_name = "dfdt",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax3, hm3 = axis_with_heatmap(fig[1, 3], T, P_rev, dfdv';
    # title = L"\text{\Vert \mathbf{J}_f v_{bulk} \Vert_2}",
    colormap = COLORMAP,
    colorrange = (min_val, max_val_adj),
    raster_name = "dfdv_bulk",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax4, hm4 = axis_with_heatmap(fig[2, 1], T, P_rev, dfdSi';
    # title = L"\text{\Vert \mathbf{J}_f v_{Si} \Vert_2}",
    colormap = COLORMAP_COMP,
    colorrange = (min_val_comp, max_val_comp_adj),
    raster_name = "dfdSi",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax5, hm5 = axis_with_heatmap(fig[2, 2], T, P_rev, dfdCa';
    # title = L"\text{\Vert \mathbf{J}_f v_{Ca} \Vert_2}",
    colormap = COLORMAP_COMP,
    colorrange = (min_val_comp, max_val_comp_adj),
    raster_name = "dfdCa",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax6, hm6 = axis_with_heatmap(fig[2, 3], T, P_rev, dfdAl';
    # title = L"\text{\Vert \mathbf{J}_f v_{Al} \Vert_2}",
    colormap = COLORMAP_COMP,
    colorrange = (min_val_comp, max_val_comp_adj),
    raster_name = "dfdAl",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax7, hm7 = axis_with_heatmap(fig[3, 1], T, P_rev, dfdFe';
    # title = L"\text{\Vert \mathbf{J}_f v_{Fe} \Vert_2}",
    colormap = COLORMAP_COMP,
    colorrange = (min_val_comp, max_val_comp_adj),
    raster_name = "dfdFe",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax8, hm8 = axis_with_heatmap(fig[3, 2], T, P_rev, dfdMg';
    # title = L"\text{\Vert \mathbf{J}_f v_{Mg} \Vert_2}",
    colormap = COLORMAP_COMP,
    colorrange = (min_val_comp, max_val_comp_adj),
    raster_name = "dfdMg",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax9, hm9 = axis_with_heatmap(fig[3, 3], T, P_rev, dfdNa';
    # title = L"\text{\Vert \mathbf{J}_f v_{Na} \Vert_2}",
    colormap = COLORMAP_COMP,
    colorrange = (min_val_comp, max_val_comp_adj),
    raster_name = "dfdNa",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)

# Add geotherms
geotherm_P = range(P_bounds[1], P_bounds[2], length=100)

geotherms = (
    cold_subduction = (a=0.95, b=580.0),
    warm_subduction = (a=1.3, b=800),
    mantle_adiabat  = (a=0.8, b=1520),
)
geotherm_T(P, g) = P .* g.a .+ g.b

lines!(ax3, geotherm_T(geotherm_P, geotherms.cold_subduction), geotherm_P, color=:white, alpha=0.2, linewidth=20, label="Cold subduction")
lines!(ax3, geotherm_T(geotherm_P, geotherms.warm_subduction), geotherm_P, color=:white, alpha=0.2, linewidth=20, label="Warm subduction")
lines!(ax3, geotherm_T(geotherm_P, geotherms.mantle_adiabat), geotherm_P, color=:white, alpha=0.2, linewidth=20, label="Mantle adiabat")

# add labels to geotherm lines
text!(ax3, "1", position = (geotherm_T(11., geotherms.cold_subduction), 11.), align = (:center, :bottom), color=:white, fontsize=12)
text!(ax3, "2", position = (geotherm_T(11., geotherms.warm_subduction), 11.), align = (:center, :bottom), color=:white, fontsize=12)
text!(ax3, "3", position = (geotherm_T(11., geotherms.mantle_adiabat), 11.), align = (:center, :bottom), color=:white, fontsize=12)

# Show labels only on outer axes for the 3x3 panel layout.
axes = [ax1 ax2 ax3; ax4 ax5 ax6; ax7 ax8 ax9]
for i in 1:3, j in 1:3
    ax = axes[i, j]
    show_x = (i == 3)
    show_y = (j == 1)

    ax.xlabelvisible = show_x
    ax.xticklabelsvisible = show_x
    ax.xticksvisible = show_x

    ax.ylabelvisible = show_y
    ax.yticklabelsvisible = show_y
    ax.yticksvisible = show_y
end

# add subfigure labels and titles
label_fontsize = 18
label_offset = (550, 395)
title_offset = (2450, 395)
labels = ["(a)" "(b)" "(c)";
          "(d)" "(e)" "(f)";
          "(g)" "(h)" "(i)"]
titles = [L"\text{\Vert \mathbf{J}_f v_P \Vert_2}" L"\text{\Vert \mathbf{J}_f \hat{e}_T \Vert_2}" L"\text{\Vert \mathbf{J}_f v_{BAS-HRZ} \Vert_2}";
          L"\text{\Vert \mathbf{J}_f v_{Si} \Vert_2}" L"\text{\Vert \mathbf{J}_f v_{Ca} \Vert_2}" L"\text{\Vert \mathbf{J}_f v_{Al} \Vert_2}";
          L"\text{\Vert \mathbf{J}_f v_{Fe} \Vert_2}" L"\text{\Vert \mathbf{J}_f v_{Mg} \Vert_2}" L"\text{\Vert \mathbf{J}_f v_{Na} \Vert_2}"]
for i in 1:3, j in 1:3
    ax = axes[i, j]
    label = labels[i, j]
    title = titles[i, j]

    text!(ax, label, position = label_offset, align = (:left, :top), color = :white, font = :bold, fontsize = label_fontsize)
    text!(ax, title, position = title_offset, align = (:right, :top), color = :white, fontsize = label_fontsize)
end



# Add colorbars
Colorbar(fig[1, 4], hm1, height = Relative(1.0), ticks = ([0, 0.005, 0.025, 0.05, 0.1], ["0.0", "0.5", "2.5", "5.0", "10.0"]), label = L"\text{Sensitivity [10^{-2} a.u.]}")
Colorbar(fig[2:3, 4], hm4, height = Relative(1.0), ticks = ([0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1], ["0.0", "0.1", "0.5", "1.0", "2.5", "5.0", "10.0"]), label = L"\text{Sensitivity [10^{-2} a.u.]}")

colgap!(fig.layout, 30)
rowgap!(fig.layout, 30)
fig

save(joinpath("05_figures", "fig05_dMAD", "fig_dMAD.pdf"), fig)
