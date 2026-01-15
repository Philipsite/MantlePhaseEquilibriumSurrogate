using JLD2, CSV, DataFrames
using CairoMakie
using Sprout

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

DATA_DIR = joinpath("data", "generated_dataset")
x_train = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_train_x.csv"), DataFrame);
y_train = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_train_y.csv"), DataFrame);
x_val = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_val_x.csv"), DataFrame);
y_val = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_val_y.csv"), DataFrame);

x_train, _, _, _, _, _ = preprocess_data(x_train, y_train);
x_val, 𝑣_val, _, _, _, _ = preprocess_data(x_val, y_val);
y_val = one_hot_phase_stability(𝑣_val);

# Normalise inputs
xNorm = Norm(x_train);
x_val = xNorm(x_val);

t_EXP1 = estimate_inference_time(path_EXP1, n_layers, n_neurons, (x_val, y_val))
println("Estimated inference time for EXP1; DONE!")
t_EXP2 = estimate_inference_time(path_EXP2, n_layers, n_neurons, (x_val, y_val))
println("Estimated inference time for EXP2; DONE!")
t_EXP3 = estimate_inference_time(path_EXP3, n_layers, n_neurons, (x_val, y_val))
println("Estimated inference time for EXP3; DONE!")
t_EXP4 = estimate_inference_time(path_EXP4, n_layers, n_neurons, (x_val, y_val))
println("Estimated inference time for EXP4; DONE!")
t_EXP5 = estimate_inference_time(path_EXP5, n_layers, n_neurons, (x_val, y_val))
println("Estimated inference time for EXP5; DONE!")
t_EXP6 = estimate_inference_time(path_EXP6, n_layers, n_neurons, (x_val, y_val))
println("Estimated inference time for EXP6; DONE!")

# save inference times
@save joinpath(hpt_results_classifier_dir, "inference_times_hpt_classifier.jld2") t_EXP1 t_EXP2 t_EXP3 t_EXP4 t_EXP5 t_EXP6
