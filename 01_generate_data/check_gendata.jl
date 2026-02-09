using CSV, DataFrames
using Sprout
using CairoMakie


DATA_DIR = joinpath("data", "generated_dataset");
REGEN_DATA_DIR = joinpath("data", "regenerated_dataset");

x = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_test_x.csv"), DataFrame);
y = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_test_y.csv"), DataFrame);
x, 𝑣, 𝐗_ss, _, _, _ = preprocess_data(x, y);
p = x[1, 1, :]
t = x[2, 1, :]

x_rg = CSV.read(joinpath(REGEN_DATA_DIR, "sb21_09Feb25_test_x.csv"), DataFrame);
y_rg = CSV.read(joinpath(REGEN_DATA_DIR, "sb21_09Feb25_test_y.csv"), DataFrame);
x_rg, 𝑣_rg, 𝐗_ss_rg, _, _, _ = preprocess_data(x_rg, y_rg);
p_rg = x_rg[1, 1, :]
t_rg = x_rg[2, 1, :]

P_range_kbar = (10., 400.)
T_range_C = (500., 2500.)

fig = Figure()

gl = fig[1, 1] = GridLayout()
gr = fig[1, 2] = GridLayout()

for (grid_layout, p, t) in zip((gl, gr), (p, p_rg), (t, t_rg))
    ax_top = Axis(grid_layout[1, 1])
    ax_center = Axis(grid_layout[2, 1])
    ax_left = Axis(grid_layout[2, 2])

    hist!(ax_top, p; bins = 1000, color=(:orangered, 0.5))
    scatter!(ax_center, p, t; color=:orangered, markersize=0.5, strokewidth=0)
    hist!(ax_left, t; direction=:x, bins = 1000, color=(:orangered, 0.5))
    xlims!(ax_top, P_range_kbar...)
    limits!(ax_center, P_range_kbar..., T_range_C...)
    ylims!(ax_left, T_range_C...)
    hideydecorations!(ax_left, ticks=false, grid=false)
    hidexdecorations!(ax_top, ticks=false, grid=false)

end

fig
