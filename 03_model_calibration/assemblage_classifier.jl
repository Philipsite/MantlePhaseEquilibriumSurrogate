using CSV, DataFrames
using JLD2
using Flux
using Sprout
using CairoMakie

# LOAD DATA
#-----------------------------------------------------------------------
DATA_DIR = joinpath("data", "generated_dataset")
x_train = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_train_x.csv"), DataFrame);
y_train = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_train_y.csv"), DataFrame);
x_val = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_val_x.csv"), DataFrame);
y_val = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_val_y.csv"), DataFrame);

x_train, 𝑣_train, _, _, _, _ = preprocess_data(x_train, y_train);
y_train = one_hot_phase_stability(𝑣_train);
x_val, 𝑣_val, _, _, _, _ = preprocess_data(x_val, y_val);
y_val = one_hot_phase_stability(𝑣_val);

# Normalise inputs
xNorm = Norm(x_train);
x_train = xNorm(x_train);
x_val = xNorm(x_val);

# Setup DataLoader
loader = Flux.DataLoader((x_train, y_train), batchsize=4096, shuffle=true);

# SETUP MODEL
#-----------------------------------------------------------------------
n_layers = 2;
n_neurons = 200;
input_dim = size(x_train, 1);
output_dim = size(y_train, 1);

m = create_classifier_model(n_layers, n_neurons, input_dim, output_dim) |> gpu_device();

opt_state = Flux.setup(Flux.Adam(0.001), m);
# Optional Learning Rate Scheduler and Early Stopping; can be passed to train_loop as kwargs
# lr_schedule = CosAnneal(λ0 = 1e-5, λ1 = 1e-3, period = 50, restart=true)
# setup early_stopping_condition
early_stopping = Flux.early_stopping((val_loss) -> val_loss, 10, init_score=Inf32)

# TRAIN MODEL
#-----------------------------------------------------------------------
model, opt_state, logs_t, dir = train_loop(
    m,
    loader,
    opt_state,
    (x_val, y_val),
    Flux.Losses.binarycrossentropy,
    5000,
    metrics = [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
    early_stopping_condition = early_stopping,
    save_to_subdir = joinpath("classifier"),
    gpu_device = gpu_device()
);

# save Norm
@save joinpath(dir, "xNorm.jld2") xNorm

# POST-TRAINING PLOTS
fig = post_training_plots(logs_t, dir);

# update metrics plot
ax2 = fig.content[5];
xlims!(ax2, (nothing, nothing));
ylims!(ax2, (nothing, 0.10));

save(dir * "/train_log_.pdf", fig)
