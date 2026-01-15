
using Flux
using CSV, DataFrames
using Sprout
using CairoMakie

n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9];
n_neurons = [50, 100, 150, 200, 250, 300, 350, 400];
batch_size = [4096, 25000, 100000];

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


# TUNE IT
#-----------------------------------------------------------------------
hpt_classifier(n_layers, n_neurons, batch_size[1], Flux.Losses.binarycrossentropy,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=false,
               subdir_appendix="bs1_bce")

hpt_classifier(n_layers, n_neurons, batch_size[2], Flux.Losses.binarycrossentropy,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=false,
               subdir_appendix="bs2_bce")

hpt_classifier(n_layers, n_neurons, batch_size[3], Flux.Losses.binarycrossentropy,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=false,
               subdir_appendix="bs3_bce")

hpt_classifier(n_layers, n_neurons, batch_size[1], misfit.binary_focal_loss,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=false,
               subdir_appendix="bs1_bfl")

hpt_classifier(n_layers, n_neurons, batch_size[2], misfit.binary_focal_loss,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=false,
               subdir_appendix="bs2_bfl")

hpt_classifier(n_layers, n_neurons, batch_size[3], misfit.binary_focal_loss,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=false,
               subdir_appendix="bs3_bfl")

println("Hyperparameter tuning complete!")
