using CSV, DataFrames
using CairoMakie, LaTeXStrings, Printf

# dataset
DATA_DIR = "data/generated_dataset/"

x_train = CSV.read(DATA_DIR * "sb21_02Oct25_train_x.csv", DataFrame);
x_names = names(x_train);
x_train = Matrix(Matrix{Float32}(x_train)');

# (1) PLOT the bulk composition (molar)
# all compositional vectors follow: ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
# and are renormalised to a sum of 1.0
MOLAR_MASS = [60.08, 56.08, 101.96, 71.84, 40.30, 61.98] # g/mol

# Pyrolite after Green (1979)
PYR_wt = [45, 3.4, 4.4, 7.6, 38.8,  0.34]
PYR = PYR_wt ./ MOLAR_MASS
PYR ./= sum(PYR)

# Primitive upper mantle after Sun and McDonough (1989)
PUM_SD89_wt = [44.9, 3.54, 4.44, 8.03, 37.7, 0.36]
PUM_SD89 = PUM_SD89_wt ./ MOLAR_MASS
PUM_SD89 ./= sum(PUM_SD89)

# Primitive upper mantle after Palme and O'Neill (2004)
PUM_PO04_wt = [45.40, 3.65, 4.49, 8.10, 36.77, 0.2590]
PUM_PO04 = PUM_PO04_wt ./ MOLAR_MASS
PUM_PO04 ./= sum(PUM_PO04)

# Depleted MORB Mantle after Workman and Hart (2005)
DMM_wt = [44.71, 3.17, 3.98, 8.18, 38.73, 0.13]
DMM = DMM_wt ./ MOLAR_MASS
DMM ./= sum(DMM)

# Altered Oceanic Crust after Kelley et al. (2003)
AOC_wt = [49.23, 13.03, 12.05, 13.72 * 0.8998, 6.22, 2.30]
AOC = AOC_wt ./ MOLAR_MASS
AOC ./= sum(AOC)

# Compositions after Xu et al. (2008)
# Pyrolite
PYR_Xu_mol = [38.71, 2.94, 2.22, 6.17, 49.85, 0.11]
PYR_Xu_mol ./= sum(PYR_Xu_mol)
# Basalt
BAS_Xu_mol = [51.75, 13.88, 10.19, 7.06, 14.94, 2.18]
BAS_Xu_mol ./= sum(BAS_Xu_mol)
# Modified Harzburgite
HAR_Xu_mol = [36.04, 0.79, 0.65, 5.97, 56.54, 0.00]
HAR_Xu_mol ./= sum(HAR_Xu_mol)

function x_Mg(bulk_mol)
    return bulk_mol[findfirst(oxides .== "MgO"), :] ./ (bulk_mol[findfirst(oxides .== "MgO"), :] .+ bulk_mol[findfirst(oxides .== "FeO"), :])
end
function ratio_AlSi(bulk_mol)
    return 2. * bulk_mol[findfirst(oxides .== "Al2O3"), :] ./ bulk_mol[findfirst(oxides .== "SiO2"), :]
end
function totalNaCA(bulk_mol)
    return bulk_mol[findfirst(oxides .== "Na2O"), :] .+ bulk_mol[findfirst(oxides .== "CaO"), :]
end

oxides = x_names[3:end]
bulk_train = x_train[3:end, :]

col_map = :bamako
col_range = (minimum(totalNaCA(bulk_train)), maximum(totalNaCA(bulk_train)))

fig = Figure(; size = (400, 300))

ax1 = Axis(fig[1,1],
           xgridvisible=false, ygridvisible=false,
           xlabel = L"X_{\mathrm{Mg}}", ylabel = L"\frac{Al}{Si}")
s1 = scatter!(x_Mg(bulk_train[:, 1:200:end]), ratio_AlSi(bulk_train[:, 1:200:end]), color=totalNaCA(bulk_train[:, 1:200:end]), markersize=3, colormap=col_map, colorrange=col_range)

Colorbar(fig[1, 2], s1, label = L"\mathrm{Na_2O + CaO [molmol^{-1}]}")

text!(ax1, 0.55, 0.045, text=@sprintf("n = %0.1e", size(x_train)[2]))

s1_3 = scatter!(x_Mg(PYR_Xu_mol), ratio_AlSi(PYR_Xu_mol), markersize=22, marker = :circle, color=:white)
s1_3 = scatter!(x_Mg(PYR_Xu_mol), ratio_AlSi(PYR_Xu_mol), markersize=20, marker = :circle, color=totalNaCA(PYR_Xu_mol), colormap=col_map, colorrange=col_range)
text!(ax1, x_Mg(PYR_Xu_mol) .- 0.01, ratio_AlSi(PYR_Xu_mol), text="Pyrolite (Xu)", align=(:right, :center), fontsize=10)
s1_4 = scatter!(x_Mg(BAS_Xu_mol), ratio_AlSi(BAS_Xu_mol), markersize=22, marker = :circle, color=:white)
s1_4 = scatter!(x_Mg(BAS_Xu_mol), ratio_AlSi(BAS_Xu_mol), markersize=20, marker = :circle, color=totalNaCA(BAS_Xu_mol), colormap=col_map, colorrange=col_range)
text!(ax1, x_Mg(BAS_Xu_mol) .+ 0.01, ratio_AlSi(BAS_Xu_mol), text="Basalt (Xu)", align=(:left, :center), fontsize=10)
s1_5 = scatter!(x_Mg(HAR_Xu_mol), ratio_AlSi(HAR_Xu_mol), markersize=22, marker = :circle, color=:white)
s1_5 = scatter!(x_Mg(HAR_Xu_mol), ratio_AlSi(HAR_Xu_mol), markersize=20, marker = :circle, color=totalNaCA(HAR_Xu_mol), colormap=col_map, colorrange=col_range)
text!(ax1, x_Mg(HAR_Xu_mol) .- 0.01, ratio_AlSi(HAR_Xu_mol), text="Harzburgite (Xu)", align=(:right, :center), fontsize=10)

s2_1 = scatter!(x_Mg(PUM_PO04), ratio_AlSi(PUM_PO04), markersize=17, marker = :diamond, color=:white)
s2_1 = scatter!(x_Mg(PUM_PO04), ratio_AlSi(PUM_PO04), markersize=15, marker = :diamond, color=totalNaCA(PUM_PO04), colormap=col_map, colorrange=col_range)
text!(ax1, x_Mg(PUM_PO04) .- 0.002, ratio_AlSi(PUM_PO04), text="PM Palme & O'Neill (2004)", align=(:right, :bottom), fontsize=8)

s2_2 = scatter!(x_Mg(DMM), ratio_AlSi(DMM), markersize=17, marker = :diamond, color=:white)
s2_2 = scatter!(x_Mg(DMM), ratio_AlSi(DMM), markersize=15, marker = :diamond, color=totalNaCA(DMM), colormap=col_map, colorrange=col_range)
text!(ax1, x_Mg(DMM) .- 0.002, ratio_AlSi(DMM), text="DMM Workman & Hart (2005)", align=(:right, :top), fontsize=8)

# s2_3 = scatter!(x_Mg(AOC), ratio_AlSi(AOC), markersize=15, marker = :diamond, color=totalNaCA(AOC), colormap=col_map, colorrange=col_range)
# text!(ax1, x_Mg(AOC) .+ 0.002, ratio_AlSi(AOC), text="AOC Kelley et al. (2003)", align=(:left, :center), fontsize=8)

fig
save(joinpath("05_figures", "fig01_results", "fig01_dataset.pdf"), fig)
