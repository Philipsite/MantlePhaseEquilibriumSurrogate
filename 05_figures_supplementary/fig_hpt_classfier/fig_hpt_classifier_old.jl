using JLD2
using CairoMakie
using Sprout

set_theme!(Theme(font = "Helvetica", fontsize = 10))

#=
Hyperparameter tuning experiments
EXP1 > BCE loss + batch size of 4096 + no lr schedule
EXP2 > BCE loss + batch size of 25000 + no lr schedule
EXP3 > BCE loss + batch size of 100000 + no lr schedule
EXP4 > BFL loss + batch size of 4096 + no lr schedule
EXP5 > BFL loss + batch size of 25000 + no lr schedule
EXP6 > BFL loss + batch size of 100000 + no lr schedule
=#

# Set-up all paths
hpt_results_classifier_dir = "data/hpt_results/classifier"
path_EXP1 = joinpath(hpt_results_classifier_dir, "hyperparam_tuningbs1_bce_2025Dec22_1229")
path_EXP2 = joinpath(hpt_results_classifier_dir, "hyperparam_tuningbs2_bce_2025Dec23_0027")
path_EXP3 = joinpath(hpt_results_classifier_dir, "hyperparam_tuningbs3_bce_2025Dec23_1440")
path_EXP4 = joinpath(hpt_results_classifier_dir, "hyperparam_tuningbs1_bfl_2025Dec24_1133")
path_EXP5 = joinpath(hpt_results_classifier_dir, "hyperparam_tuningbs2_bfl_2025Dec24_2340")
path_EXP6 = joinpath(hpt_results_classifier_dir, "hyperparam_tuningbs3_bfl_2025Dec25_1638")

# All experiments use the same grid for n_layers, n_neurons
n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9];
n_neurons = [50, 100, 150, 200, 250, 300, 350, 400];
batch_size = [4096, 25000, 100000];

# load hyperparam tuning results
log_EXP1 = load_hyperparam_tuning_results(path_EXP1, n_layers, n_neurons)
log_EXP2 = load_hyperparam_tuning_results(path_EXP2, n_layers, n_neurons)
log_EXP3 = load_hyperparam_tuning_results(path_EXP3, n_layers, n_neurons)
log_EXP4 = load_hyperparam_tuning_results(path_EXP4, n_layers, n_neurons)
log_EXP5 = load_hyperparam_tuning_results(path_EXP5, n_layers, n_neurons)
log_EXP6 = load_hyperparam_tuning_results(path_EXP6, n_layers, n_neurons)

logs = [log_EXP1, log_EXP2, log_EXP3, log_EXP4, log_EXP5, log_EXP6]

qasm_loss_EXP1 = minimum.(getfield.(log_EXP1, :loss_asm))
qasm_loss_EXP2 = minimum.(getfield.(log_EXP2, :loss_asm))
qasm_loss_EXP3 = minimum.(getfield.(log_EXP3, :loss_asm))
qasm_loss_EXP4 = minimum.(getfield.(log_EXP4, :loss_asm))
qasm_loss_EXP5 = minimum.(getfield.(log_EXP5, :loss_asm))
qasm_loss_EXP6 = minimum.(getfield.(log_EXP6, :loss_asm))

qasm_loss = cat(qasm_loss_EXP1, qasm_loss_EXP2, qasm_loss_EXP3, qasm_loss_EXP4, qasm_loss_EXP5, qasm_loss_EXP6, dims=3)

# Load inference times
@load joinpath(hpt_results_classifier_dir, "inference_times_hpt_classifier.jld2") t_EXP1 t_EXP2 t_EXP3 t_EXP4 t_EXP5 t_EXP6

inf_times = cat(t_EXP1, t_EXP2, t_EXP3, t_EXP4, t_EXP5, t_EXP6, dims=3)


# Get global limits
MIN_QASM = minimum(qasm_loss)
MAX_QASM = maximum(qasm_loss)

# find "THE OPTIMAL MODEL"
opt_m_idx = argmin(qasm_loss)
opt_m_idx_bce = argmin(qasm_loss[:, :, 1:3])

colormap = cgrad(:acton, rev = true)
colorrange = (MIN_QASM, MAX_QASM)

# Define a "model size" parameter to color-code model training curves
model_size = n_layers .* n_neurons'
colorrange_size = (minimum(model_size), maximum(model_size))
colormap_size = cgrad(:bamako, rev = true)

# inf_times colors
MIN_INF_TIME = minimum(inf_times)
MAX_INF_TIME = maximum(inf_times)

# find the quickest model among all experiments with (1 - Qasm) < MINIMAL METRIC THRESHOLD
MINIMAL_METRIC_THRESHOLD = 0.015
ids = findall(qasm_loss .< MINIMAL_METRIC_THRESHOLD)
opt_m_idx_inf_time = ids[argmin(inf_times[ids])]

# find the quickest model among all experiments with (1 - Qasm) < MINIMAL METRIC THRESHOLD for BCE models
ids_bce = findall(qasm_loss[:, :, 1:3] .< MINIMAL_METRIC_THRESHOLD)
opt_m_idx_inf_time_bce = ids_bce[argmin(inf_times[ids_bce])]

colormap_inf_time = cgrad(:lapaz, rev = true)
colorrange_inf_time = (MIN_INF_TIME, MAX_INF_TIME)

# Figure set-up
dpi = 72
w_mm, h_mm = 210.0, 290.
mm_to_in = 1/25.4
w_px = Int(round(w_mm * mm_to_in * dpi))
h_px = Int(round(h_mm * mm_to_in * dpi))

fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (w_px, h_px))

# set-up grid
grid_a = fig[1, 1:3] = GridLayout()
grid_b = fig[2, 1:2] = GridLayout()
grid_c = fig[2, 3] = GridLayout()

# PANEL A
ax1 = Axis(grid_a[1, 1], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax2 = Axis(grid_a[1, 2], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax3 = Axis(grid_a[1, 3], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax4 = Axis(grid_a[2, 1], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax5 = Axis(grid_a[2, 2], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax6 = Axis(grid_a[2, 3], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")

axs = [ax1, ax2, ax3, ax4, ax5, ax6]

for i in eachindex(axs)
    ax = axs[i]
    ax.xticks = (n_layers, string.(n_layers))
    ax.yticks = (n_neurons, string.(n_neurons))

    hm = heatmap!(ax, n_layers, n_neurons, qasm_loss[:, :, i], colormap = colormap, colorrange = colorrange)
end

hidedecorations!.([ax2, ax3], ticks=false, grid=false)
ax1.xlabelvisible = false
ax1.xticklabelsvisible = false
ax5.ylabelvisible = false
ax5.yticklabelsvisible = false
ax6.ylabelvisible = false
ax6.yticklabelsvisible = false
# plot a marker for the optimal model
scatter!(axs[opt_m_idx[3]], n_layers[opt_m_idx[1]], n_neurons[opt_m_idx[2]], color=:red, marker=:star5, markersize=12)
# plot a marker for the optimal model among BCE models
scatter!(axs[opt_m_idx_bce[3]], n_layers[opt_m_idx_bce[1]], n_neurons[opt_m_idx_bce[2]], color=:red, marker=:diamond, markersize=10)
# plot a marker for the fastest model among all models with (1 - Qasm) < 0.01, if this is NOT identical to the fastet model using BCE
if opt_m_idx_inf_time[3] > 3
    scatter!(axs[opt_m_idx_inf_time[3]], n_layers[opt_m_idx_inf_time[1]], n_neurons[opt_m_idx_inf_time[2]], color=:blue, marker=:star5, markersize=10)
end
# plot a marker for the fastest BCE model among all models with (1 - Qasm) < 0.01
scatter!(axs[opt_m_idx_inf_time_bce[3]], n_layers[opt_m_idx_inf_time_bce[1]], n_neurons[opt_m_idx_inf_time_bce[2]], color=:blue, marker=:diamond, markersize=8)

Colorbar(grid_a[1:2, 4], colormap = colormap, limits = colorrange, label=L"1 - \textrm{Q_{asm}}", size=8)

# add labels for loss functions and batch sizes
label_bs1 = Label(grid_a[0, 1], "Batch size = $(batch_size[1])", fontsize=10, tellwidth=false)
label_bs2 = Label(grid_a[0, 2], "Batch size = $(batch_size[2])", fontsize=10, tellwidth=false)
label_bs3 = Label(grid_a[0, 3], "Batch size = $(batch_size[3])", fontsize=10, tellwidth=false)
label_loss_bce = Label(grid_a[1, 0], "Binary Cross-Entropy Loss", rotation = pi/2, fontsize=10, tellheight=false)
label_loss_bfl = Label(grid_a[2, 0], "Binary Focal Loss", rotation = pi/2, fontsize=10, tellheight=false)

# PANEL B
ax1 = Axis(grid_b[1, 1], xscale = log10, ygridvisible=true, xgridvisible=false)
ax1.title = "Models trained with Binary Cross-Entropy Loss"
ax1.xlabel = "n.o. epochs (iteration over training set)"
ax1.ylabel = L"1 - \textrm{Q_{asm}}"
xlims!(ax1, 9, nothing)
ylims!(ax1, nothing, 0.25)
for i in 1:3
    loss_vecs = getfield.(logs[i], :loss_asm)
    for j in eachindex(loss_vecs)
        lines!(ax1, loss_vecs[j], linewidth=.3, color=model_size[j], colormap = colormap_size, colorrange = colorrange_size)
    end
end
ax2 = Axis(grid_b[2, 1], xscale = log10, ygridvisible=true, xgridvisible=false)
ax2.title = "Models trained with Binary Focal Loss"
ax2.xlabel = "n.o. epochs (iteration over training set)"
ax2.ylabel = L"1 - \textrm{Q_{asm}}"
xlims!(ax2, 9, nothing)
ylims!(ax2, nothing, 0.25)
for i in 4:6
    loss_vecs = getfield.(logs[i], :loss_asm)
    for j in eachindex(loss_vecs)
        lines!(ax2, loss_vecs[j], linewidth=.3, color=model_size[j], colormap = colormap_size, colorrange = colorrange_size)
    end
end

# plot optimal model training curve on top
loss_opt = logs[opt_m_idx[3]][opt_m_idx[1], opt_m_idx[2]][:loss_asm]
if opt_m_idx[3] <= 3
    lines!(ax1, loss_opt, color=:red, linewidth=1)
    text!(ax1, Float32(argmin(loss_opt)), 0.0, text="Minimal loss $(round(minimum(loss_opt), digits=4))", align = (:center, :top), color=:red)
else
    lines!(ax2, loss_opt, color=:red, linewidth=1)
    text!(ax2, Float32(argmin(loss_opt)), 0.0, text="Minimal loss $(round(minimum(loss_opt), digits=4))", align = (:center, :top), color=:red)
end
loss_opt_inf_time = logs[opt_m_idx_inf_time[3]][opt_m_idx_inf_time[1], opt_m_idx_inf_time[2]][:loss_asm]
if opt_m_idx_inf_time[3] > 3
    lines!(ax2, loss_opt_inf_time, color=:blue, linewidth=1)
    text!(ax2, Float32(argmin(loss_opt_inf_time)), 0.0, text="Minimal loss quick model $(round(minimum(loss_opt_inf_time), digits=4))", align = (:center, :bottom), color=:blue)
end
loss_opt_bce = logs[opt_m_idx_bce[3]][opt_m_idx_bce[1], opt_m_idx_bce[2]][:loss_asm]
lines!(ax1, loss_opt_bce, color=:red, linewidth=1)
text!(ax1, Float32(argmin(loss_opt_bce)), 0.0, text="Minimal loss BCE $(round(minimum(loss_opt_bce), digits=4))", align = (:center, :top), color=:red)

loss_opt_inf_time_bce = logs[opt_m_idx_inf_time_bce[3]][opt_m_idx_inf_time_bce[1], opt_m_idx_inf_time_bce[2]][:loss_asm]
lines!(ax1, loss_opt_inf_time_bce, color=:blue, linewidth=1)
text!(ax1, Float32(argmin(loss_opt_inf_time_bce)), 0.0, text="Minimal loss quick BCE $(round(minimum(loss_opt_inf_time_bce), digits=4))", align = (:center, :bottom), color=:blue)

Colorbar(grid_b[1:2, 2], colormap = colormap_size, limits = colorrange_size, label="← smaller models / larger models →", size=8, ticksvisible=false, ticklabelsvisible=false)

# PANEL C
ax_c2 = Axis(grid_c[1, 1],aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax_c2.title = "BCE & batch size = 4096"
ax_c2.xticks = (n_layers, string.(n_layers))
ax_c2.yticks = (n_neurons, string.(n_neurons))
hm_inf_time_bce = heatmap!(ax_c2, n_layers, n_neurons, inf_times[:, :, opt_m_idx_bce[3]], colormap = colormap_inf_time, colorrange = colorrange_inf_time)

ax_c1 = Axis(grid_c[2, 1],aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons")
ax_c1.title = "BFL & batch size = 25k"
ax_c1.xticks = (n_layers, string.(n_layers))
ax_c1.yticks = (n_neurons, string.(n_neurons))
hm_inf_time = heatmap!(ax_c1, n_layers, n_neurons, inf_times[:, :, opt_m_idx[3]], colormap = colormap_inf_time, colorrange = colorrange_inf_time)

# Plot markers for optimal models
scatter!(ax_c2, n_layers[opt_m_idx_bce[1]], n_neurons[opt_m_idx_bce[2]], color=:red, marker=:diamond, markersize=10)
text!(ax_c2, n_layers[opt_m_idx_bce[1]], n_neurons[opt_m_idx_bce[2]], text="$(round(inf_times[opt_m_idx_bce], digits=0)) ms", align = (:right, :bottom), color=:red, offset = (5, 5))
scatter!(ax_c2, n_layers[opt_m_idx_inf_time_bce[1]], n_neurons[opt_m_idx_inf_time_bce[2]], color=:blue, marker=:diamond, markersize=8)
text!(ax_c2, n_layers[opt_m_idx_inf_time_bce[1]], n_neurons[opt_m_idx_inf_time_bce[2]], text="$(round(inf_times[opt_m_idx_inf_time_bce], digits=0)) ms", align = (:left, :bottom), color=:blue, offset = (5, 5))
if opt_m_idx[3] > 3
    scatter!(ax_c1, n_layers[opt_m_idx[1]], n_neurons[opt_m_idx[2]], color=:red, marker=:star5, markersize=12)
    text!(ax_c1, n_layers[opt_m_idx[1]], n_neurons[opt_m_idx[2]], text="$(round(inf_times[opt_m_idx], digits=0)) ms", align = (:right, :bottom), color=:red, offset = (5, 5))
end
if opt_m_idx_inf_time[3] > 3
    scatter!(ax_c1, n_layers[opt_m_idx_inf_time[1]], n_neurons[opt_m_idx_inf_time[2]], color=:blue, marker=:star5, markersize=10)
    text!(ax_c1, n_layers[opt_m_idx_inf_time[1]], n_neurons[opt_m_idx_inf_time[2]], text="$(round(inf_times[opt_m_idx_inf_time], digits=0)) ms", align = (:left, :bottom), color=:blue, offset = (5, 5))
end
Colorbar(grid_c[3, 1], colormap = colormap_inf_time, limits = colorrange_inf_time, label="Inference time /ms", size=8, vertical = false)
fig
save(joinpath(@__DIR__, "hpt_classifier.pdf"), fig; dpi = 300)
