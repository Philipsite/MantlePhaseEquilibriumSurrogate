using CSV, DataFrames
using CairoMakie, LaTeXStrings, Printf

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


#=
LOAD DATASET
=#
DATA_DIR = "data/generated_dataset/"

x_train = CSV.read(DATA_DIR * "sb21_02Oct25_train_x.csv", DataFrame);
bulk_train = x_train[3:end, :]
oxides = [L"SiO_2", L"CaO", L"Al_2O_3", L"FeO", L"MgO", L"Na_2O"];
x_train = Matrix(Matrix{Float32}(x_train)');

x_val = CSV.read(DATA_DIR * "sb21_02Oct25_val_x.csv", DataFrame);
x_val = Matrix(Matrix{Float32}(x_val)');

#=
PLOT
=#
# reuse functions from Fig 1 to match colors for PYR; BAS; HAR markers
function totalNaCA(bulk_mol)
    return bulk_mol[end, :] .+ bulk_mol[2, :]
end

col_map = :bamako
col_range = (minimum(totalNaCA(x_train[3:end, :])), maximum(totalNaCA(x_train[3:end, :])))


fig = Figure(; size = (500, 500))
ax1 = Axis(fig[1,1],
           xgridvisible=true, ygridvisible=true,
           ylabel = L"\text{Oxide molar fraction [molmol^{-1}]}",
           yticks = Vector(0:0.1:0.7),
           xticklabelrotation = π/4,
           xticks = (1:length(oxides), oxides))

for i in 3:8
    pos = repeat([i], size(x_train)[2])  .- 2
    violin!(ax1, pos, x_train[i, :], color="darkgrey", orientation = :vertical, side=:left)
    pos = repeat([i], size(x_val)[2])  .- 2
    violin!(ax1, pos, x_val[i, :], color="lightgrey", orientation = :vertical, side=:right)
end

scatter!(ax1, 1:6, PYR_Xu_mol, markersize=14, marker = :circle, color=:white)
scatter!(ax1, 1:6, PYR_Xu_mol, markersize=12, marker = :circle, color=totalNaCA(PYR_Xu_mol)[1], colormap=col_map, colorrange=col_range)
scatter!(ax1, 1:6, BAS_Xu_mol, markersize=14, marker = :rect, color=:white)
scatter!(ax1, 1:6, BAS_Xu_mol, markersize=12, marker = :rect, color=totalNaCA(BAS_Xu_mol)[1], colormap=col_map, colorrange=col_range)
scatter!(ax1, 1:6, HAR_Xu_mol, markersize=14, marker = :utriangle, color=:white)
scatter!(ax1, 1:6, HAR_Xu_mol, markersize=12, marker = :utriangle, color=totalNaCA(HAR_Xu_mol)[1], colormap=col_map, colorrange=col_range)

m1 = MarkerElement(marker = :rect, color = :darkgrey, label = "Training dataset")
m2 = MarkerElement(marker = :rect, color = :lightgrey, label = "Validation dataset")

s1 = MarkerElement(marker = :utriangle, color = totalNaCA(HAR_Xu_mol)[1], colormap=col_map, colorrange=col_range)
s2 = MarkerElement(marker = :circle, color = totalNaCA(PYR_Xu_mol)[1], colormap=col_map, colorrange=col_range)
s3 = MarkerElement(marker = :rect, color = totalNaCA(BAS_Xu_mol)[1], colormap=col_map, colorrange=col_range)

Legend(
    fig[2,1],
    [[m1, m2], [s1, s2, s3]],
    [["Training dataset", "Validation dataset"], ["Harzburgite", "Pyrolite", "Basalt"]],
    [nothing, nothing],
    framevisible = false, orientation = :vertical, nbanks = 3, align = :left, tellheight = true)

fig
save(joinpath("05_figures_supplementary/figS1_training_data_distribution", "training_data_distribution.pdf"), fig)
