using JLD2
using LinearAlgebra
using CairoMakie
using FileIO
using Sprout

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
phase_names = [p for (i, p) in enumerate(vcat(PP, SS)) if i ∉ Sprout.IDX_OF_PHASES_NEVER_STABLE];
wa_idx = findfirst(phase_names .== "wa")
ri_idx = findfirst(phase_names .== "ri")
pv_idx = findfirst(phase_names .== "pv")
gt_idx = findfirst(phase_names .== "gtmj")
ak_idx = findfirst(phase_names .== "ak")
nal_idx = findfirst(phase_names .== "nal")

extract_directional_derivatives = (x, ph_idx, v) -> reshape([(x[i, ph_idx:ph_idx, :] * v)[1] for i in 1:size(x, 1)], n, n)

∇_v_wa = extract_directional_derivatives(Js_PYR_scaled, wa_idx, [0, 0, bulk_var_vec...])
∇_v_ri = extract_directional_derivatives(Js_PYR_scaled, ri_idx, [0, 0, bulk_var_vec...])
∇_v_pv = extract_directional_derivatives(Js_PYR_scaled, pv_idx, [0, 0, bulk_var_vec...])
∇_v_gt = extract_directional_derivatives(Js_PYR_scaled, gt_idx, [0, 0, bulk_var_vec...])
∇_v_ak = extract_directional_derivatives(Js_PYR_scaled, ak_idx, [0, 0, bulk_var_vec...])
∇_v_nal = extract_directional_derivatives(Js_PYR_scaled, nal_idx, [0, 0, bulk_var_vec...])

max_abs_val = maximum(abs.(vcat(vec(∇_v_wa), vec(∇_v_ri), vec(∇_v_pv), vec(∇_v_gt), vec(∇_v_ak), vec(∇_v_nal))))
max_abs_val_adj = 0.01 + eps(Float32)

# PLOTTING
#-------------------------------------------------------------------------
const HEATMAP_DPI = 600
const COLORMAP    = Reverse(:berlin)
squeeze0(x) = sign(x) * abs(x)^(1//3)

function save_heatmap_raster(x, y, z; colormap, colorrange, colorscale, output_path, dpi=HEATMAP_DPI, size_px=(1800, 1800))
    # Render a standalone high-resolution heatmap raster without axis decorations.
    fig_hm = Figure(size=size_px, figure_padding=0)
    ax_hm = Axis(fig_hm[1, 1], aspect=1.0)
    hidedecorations!(ax_hm)
    hidespines!(ax_hm)

    heatmap!(ax_hm, x, y, z; colormap=colormap, colorrange=colorrange, colorscale=colorscale)
    save(output_path, fig_hm; px_per_unit=dpi/300)
    return output_path
end

function axis_with_heatmap(grid_pos, x, y, z; title=nothing, colormap, colorrange, colorscale, raster_name="heatmap", xlabel=nothing, ylabel=nothing)
    raster_path = joinpath("05_figures", "fig06_dPhaseStability", "hm_rasters", "$(raster_name).png")
    save_heatmap_raster(x, y, z; colormap=colormap, colorrange=colorrange, colorscale=colorscale, output_path=raster_path)

    ax = Axis(grid_pos, aspect = 1.0)
    # Keep an invisible heatmap as color source for Colorbar.
    hm = heatmap!(ax, x, y, z; colormap=colormap, colorrange=colorrange, colorscale=colorscale, visible=false)
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

    ax.xticks = ([500, 1000, 1500, 2000, 2500], ["500", "1000", "1500", "2000", "2500"])
    ax.yticks = ([10, 100, 200, 300, 400], ["1", "10", "20", "30", "40"])
    return ax, hm
end


fig = Figure(size=(1000, 600))

mkpath(joinpath("05_figures", "fig06_dPhaseStability", "hm_rasters"))

ax1, hm1 = axis_with_heatmap(fig[1, 1], T, P_rev, ∇_v_wa';
    colormap = COLORMAP,
    colorrange = (-max_abs_val_adj, max_abs_val_adj),
    colorscale = squeeze0,
    raster_name = "dwadv",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax2, hm2 = axis_with_heatmap(fig[1, 2], T, P_rev, ∇_v_ri';
    colormap = COLORMAP,
    colorrange = (-max_abs_val_adj, max_abs_val_adj),
    colorscale = squeeze0,
    raster_name = "dridv",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax3, hm3 = axis_with_heatmap(fig[1, 3], T, P_rev, ∇_v_pv';
    colormap = COLORMAP,
    colorrange = (-max_abs_val_adj, max_abs_val_adj),
    colorscale = squeeze0,
    raster_name = "dpvdv",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax4, hm4 = axis_with_heatmap(fig[2, 1], T, P_rev, ∇_v_gt';
    colormap = COLORMAP,
    colorrange = (-max_abs_val_adj, max_abs_val_adj),
    colorscale = squeeze0,
    raster_name = "dgtmjdv",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax5, hm5 = axis_with_heatmap(fig[2, 2], T, P_rev, ∇_v_ak';
    colormap = COLORMAP,
    colorrange = (-max_abs_val_adj, max_abs_val_adj),
    colorscale = squeeze0,
    raster_name = "dakdv",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax6, hm6 = axis_with_heatmap(fig[2, 3], T, P_rev, ∇_v_nal';
    colormap = COLORMAP,
    colorrange = (-max_abs_val_adj, max_abs_val_adj),
    colorscale = squeeze0,
    raster_name = "dnaldv",
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)


# Add phase stability markers
text!(ax1, "+wa", position = (1960, 155), color = :white)
text!(ax2, "+ri", position = (1750, 200), color = :white)
text!(ax3, "+pv", position = (2250, 240), color = :white)
text!(ax4, "+gt", position = (2250, 235), color = :white)
text!(ax5, "+ak", position = (920, 280), color = :white)
text!(ax6, "+nal", position = (900, 280), color = :white)

# Show labels only on outer axes for the 2x3 panel layout.
axes = [ax1 ax2 ax3; ax4 ax5 ax6]
for i in 1:2, j in 1:3
    ax = axes[i, j]
    show_x = (i == 2)
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
          "(d)" "(e)" "(f)"]

titles = [L"\text{\nabla_{\mathbf{v}} wa}" L"\text{\nabla_{\mathbf{v}} ri}" L"\text{\nabla_{\mathbf{v}} pv}";
          L"\text{\nabla_{\mathbf{v}} gt}" L"\text{\nabla_{\mathbf{v}} ak}" L"\text{\nabla_{\mathbf{v}} nal}"]

for i in 1:2, j in 1:3
    ax = axes[i, j]
    label = labels[i, j]
    title = titles[i, j]

    text!(ax, label, position = label_offset, align = (:left, :top), color = :white, font = :bold, fontsize = label_fontsize)
    text!(ax, title, position = title_offset, align = (:right, :top), color = :white, fontsize = label_fontsize)
end

# Add colorbars
Colorbar(fig[1:2, 4], hm1,
    height = Relative(1.0),
    ticks = ([-0.01, -0.001, 0, 0.001, 0.01], ["-1.0", "-0.1", "0", "0.1", "1.0"]),
    label = L"\textrm{\leftarrow Stabilised by depletion} \qquad \nabla_{\mathbf{v}}\ f \text{ [10^{-2} a.u.]} \qquad \textrm{Stabilised by enrichment} \rightarrow")

colgap!(fig.layout, 30)
rowgap!(fig.layout, 30)
fig

save(joinpath("05_figures", "fig06_dPhaseStability", "fig_dPhaseStability.pdf"), fig)
