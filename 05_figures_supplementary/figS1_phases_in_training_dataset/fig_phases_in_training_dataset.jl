using CSV, DataFrames
using CairoMakie, LaTeXStrings, Printf

#=
LOAD DATASET
=#
DATA_DIR = "data/generated_dataset/"

x_train = CSV.read(DATA_DIR * "sb21_02Oct25_train_x.csv", DataFrame);
x_names = names(x_train);
x_train = Matrix(Matrix{Float32}(x_train)');

y_train = CSV.read(DATA_DIR * "sb21_02Oct25_train_y.csv", DataFrame);
y_names = names(y_train);
y_train = Matrix(Matrix{Float32}(y_train)');
# Convert phase fractions into one-hot vec for phase stability
y_train_asm = y_train[1:22,:] .!= 0.0;

x_val = CSV.read(DATA_DIR * "sb21_02Oct25_val_x.csv", DataFrame);
x_val = Matrix(Matrix{Float32}(x_val)');

y_val = CSV.read(DATA_DIR * "sb21_02Oct25_val_y.csv", DataFrame);
y_val = Matrix(Matrix{Float32}(y_val)');
y_val_asm = y_val[1:22,:] .!= 0.0;

phase_names = [split(n, "_")[1] for n in y_names[1:22]]
vol_train = y_train[1:22, :]
vol_val = y_val[1:22, :]

#=
PLOT
=#
fig = Figure(; size = (600, 500))
ax1 = Axis(fig[1,1],
           xgridvisible=true, ygridvisible=true,
           ylabel = L"\text{Molar fraction } \mathbf{v}\text{ [molmol^{-1}]}",
           yticks = Vector(0:0.2:1.0),
           xticklabelrotation = π/4,
           xticks = (1:length(phase_names), phase_names))

# compute max cts, used to scale the width of each violin plot
max_cts_train = maximum([length(filter(x -> x != 0, vol_mineral)) for vol_mineral in eachrow(vol_train)])
max_cts_val = maximum([length(filter(x -> x != 0, vol_mineral)) for vol_mineral in eachrow(vol_val)])
# compute number of "simulated rocks" to report "n" as fractions of data, where phase is present
n_train = length(y_train[1,:])
n_val = length(y_val[1,:])

for i in 1:22
    vol_f = filter(x -> x != 0, vol_train[i, :])
    n = length(vol_f)

    if n > 0
        pos = repeat([i], n)
        side = repeat([:left], n)
        violin!(ax1, pos, vol_f, boundary = (0., 1.), color="darkgrey", orientation = :vertical, side=side, width = (1. * n / max_cts_train) + 0.2)
    else
        lines!(ax1, [i, i], [0.0, 1.0], color="red", linestyle=:dot)
    end

    text!(ax1, i, 1.02, text=@sprintf("%0.2f%%", n/n_train * 100), rotation = π/2, align=(:left, :bottom), fontsize=10)

    vol_f = filter(x -> x != 0, vol_val[i, :])
    n = length(vol_f)

    if n > 0
        pos = repeat([i], n)
        side = repeat([:right], n)
        violin!(ax1, pos, vol_f, boundary = (0., 1.), color="lightgrey", orientation = :vertical, side=side, width = (1. * n / max_cts_val) + 0.1)
    end

    # text!(ax1, i, 1.05, text=@sprintf("%0.2f%%", n/n_val * 100), rotation = π/2, align=(:left, :top), fontsize=9)

end

ylims!(ax1, (-0.1, 1.2))

m1 = MarkerElement(marker = :rect, color = :darkgrey, label = "Training dataset")
m2 = MarkerElement(marker = :rect, color = :lightgrey, label = "Validation dataset")
Legend(fig[2,1], [m1, m2], ["Training dataset", "Validation dataset"], framevisible = false, orientation = :horizontal, align = :left)

fig
save(joinpath("05_figures_supplementary/figS1_phases_in_training_dataset", "phase_modes_in_training_dataset.pdf"), fig)
