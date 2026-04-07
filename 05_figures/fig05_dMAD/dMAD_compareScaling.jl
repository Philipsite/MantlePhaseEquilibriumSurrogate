"""
Script to illustrate the importance of scaling the obtained Jacobian matrices.

Plot the partial derivatives of the classifier output with respect to pressure,
temperature, and directional derivative with respect to the bulk compositional change
between pyrolite and harzburgite.

As all these partial derivatives are on different scales, it is important to scale
them appropriately. Instead as a norm vector, each gradient should be scaled to reflect
the change associated with a "characteristic" parameter step. This allows comparing the
relative sensitivity with respect to a change in pressure, temperature, or composition
not only within but also across parameters.
"""

using JLD2
using LinearAlgebra
using CairoMakie

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
dfdp_scaled = [norm(Js_PYR_scaled[i, :, 1]) for i in 1:size(Js_PYR_scaled, 1)]
dfdp_scaled = reshape(dfdp_scaled, n, n)
dfdt_scaled = [norm(Js_PYR_scaled[i, :, 2]) for i in 1:size(Js_PYR_scaled, 1)]
dfdt_scaled = reshape(dfdt_scaled, n, n)
dfdv_scaled = [norm(Js_PYR_scaled[i, :, :] * [0, 0, bulk_var_vec...]) for i in 1:size(Js_PYR_scaled, 1)]
dfdv_scaled = reshape(dfdv_scaled, n, n)

dfdp = [norm(Js_PYR[i, :, 1]) for i in 1:size(Js_PYR, 1)]
dfdp = reshape(dfdp, n, n)
dfdt = [norm(Js_PYR[i, :, 2]) for i in 1:size(Js_PYR, 1)]
dfdt = reshape(dfdt, n, n)
dfdv = [Js_PYR[i, :, :] * [0, 0, bulk_var_vec...] for i in 1:size(Js_PYR, 1)]
dfdv = [norm(dfdv[i]) for i in 1:length(dfdv)]
dfdv = reshape(dfdv, n, n)

unscaled_min = minimum(vcat(vec(dfdp), vec(dfdt), vec(dfdv)))
unscaled_max = maximum(vcat(vec(dfdp), vec(dfdt), vec(dfdv)))
scaled_min = minimum(vcat(vec(dfdp_scaled), vec(dfdt_scaled), vec(dfdv_scaled)))
scaled_max = maximum(vcat(vec(dfdp_scaled), vec(dfdt_scaled), vec(dfdv_scaled)))

function axis_with_heatmap(grid_pos, x, y, z; title, colorrange, xlabel=nothing, ylabel=nothing)
    ax = Axis(grid_pos, title = title)
    hm = heatmap!(ax, x, y, z; colormap=:batlow, colorscale=sqrt, colorrange=colorrange)
    if !isnothing(xlabel)
        ax.xlabel = xlabel
    end
    if !isnothing(ylabel)
        ax.ylabel = ylabel
    end
    return ax, hm
end

# compare scaled and unscaled
fig = Figure(size=(1200, 800))

ax1, hm1 = axis_with_heatmap(fig[1, 1], T, P_rev, dfdp';
    title = L"\text{\Vert \mathbf{J}_f \hat{e}_P \Vert_2}",
    colorrange = (unscaled_min, unscaled_max),
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax2, hm2 = axis_with_heatmap(fig[2, 1], T, P_rev, dfdp_scaled';
    title = L"\text{\Vert \mathbf{J}_f \Delta$x_P$ \Vert_2}",
    colorrange = (scaled_min, scaled_max),
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax3, hm3 = axis_with_heatmap(fig[1, 2], T, P_rev, dfdt';
    title = L"\text{\Vert \mathbf{J}_f \hat{e}_T \Vert_2}",
    colorrange = (unscaled_min, unscaled_max),
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax4, hm4 = axis_with_heatmap(fig[2, 2], T, P_rev, dfdt_scaled';
    title = L"\text{\Vert \mathbf{J}_f \Delta$x_T$ \Vert_2}",
    colorrange = (scaled_min, scaled_max),
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax5, hm5 = axis_with_heatmap(fig[1, 3], T, P_rev, dfdv';
    title = L"\text{\Vert \mathbf{J}_f \hat{v} \Vert_2}",
    colorrange = (unscaled_min, unscaled_max),
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)
ax6, hm6 = axis_with_heatmap(fig[2, 3], T, P_rev, dfdv_scaled';
    title = L"\text{\Vert \mathbf{J}_f \Delta$x_v$ \Vert_2}",
    colorrange = (scaled_min, scaled_max),
    xlabel = L"\text{Temperature [°C]}", ylabel = L"\text{Pressure [GPa]}"
)

# add colorbars
Colorbar(fig[1, 4], hm1, ticks = [0, 5, 50, 200, 400], label = L"\text{Unscaled sensitivity [a.u.]}")
Colorbar(fig[2, 4], hm2, ticks = [0, 0.07, 0.14, 0.27, 0.55], label = L"\text{Parameter scale$\endash$adjusted sensitivity [a.u.]}")

fig
