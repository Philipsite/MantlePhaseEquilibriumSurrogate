
using Flux
using CSV, DataFrames
using JLD2
using Sprout
using CairoMakie

n_layers = [2, 3, 4, 5, 6, 7, 8, 9];
n_neurons = [50, 100, 150, 200, 250, 300, 350, 400];
fraction_backbone_layers = [1//2, 2//3];
batch_size = [4096, 25000, 100000];

masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));

# LOAD DATA
#-----------------------------------------------------------------------
DATA_DIR = joinpath("data", "generated_dataset")
x_train = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_train_x.csv"), DataFrame);
y_train = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_train_y.csv"), DataFrame);
x_val = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_val_x.csv"), DataFrame);
y_val = CSV.read(joinpath(DATA_DIR, "sb21_02Oct25_val_y.csv"), DataFrame);

x_train, 𝑣_train, 𝐗_ss_train, ρ_train, Κ_train, μ_train = preprocess_data(x_train, y_train);
x_val, 𝑣_val, 𝐗_ss_val, ρ_val, Κ_val, μ_val = preprocess_data(x_val, y_val);

# Normalise inputs
xNorm = Norm(x_train);
x_train = xNorm(x_train);
x_val = xNorm(x_val);

# Scale outputs
𝐗Scale = MinMaxScaler(𝐗_ss_train);
𝐗_ss_train = 𝐗Scale(𝐗_ss_train);
𝐗_ss_val = 𝐗Scale(𝐗_ss_val);

𝑣Scale = MinMaxScaler(𝑣_train);
𝑣_train = 𝑣Scale(𝑣_train);
𝑣_val = 𝑣Scale(𝑣_val);

# SETUP LOSS & METRICS
#----------------------------------------------------------------------
# Normalisation/scaling structures must live on the same device as the model is trained on
# for training on GPU move normalisers/scalers/pure_phase_comp to GPU; e.g. xNorm_gpu = xNorm |> gpu
xNorm_gpu = xNorm |> gpu;
𝑣Scale_gpu = 𝑣Scale |> gpu;
𝐗Scale_gpu = 𝐗Scale |> gpu;
pp_mat_gpu = reshape(PP_COMP_adj, 6, :) |> gpu;

function loss((𝑣_ŷ, 𝐗_ŷ), (𝑣, 𝐗), x)
    return sum(abs2, 𝑣_ŷ .- 𝑣) + sum(abs2, 𝐗_ŷ .- 𝐗) + misfit.mass_balance_abs_misfit((descale(𝑣Scale_gpu, 𝑣_ŷ), descale(𝐗Scale_gpu, 𝐗_ŷ)), denorm(xNorm_gpu, x)[3:end,:,:], agg=sum, pure_phase_comp=pp_mat_gpu) + misfit.closure_condition((descale(𝑣Scale_gpu, 𝑣_ŷ), descale(𝐗Scale_gpu, 𝐗_ŷ)), (𝑣, 𝐗), agg=sum)
end
# Metrics (for validation only, must follow signature (ŷ, y) -> Real)
function mass_balance_metric((𝑣_ŷ, 𝐗_ŷ), (_, _))
    return misfit.mass_balance_abs_misfit((descale(𝑣Scale, 𝑣_ŷ), descale(𝐗Scale, 𝐗_ŷ)), denorm(xNorm, x_val)[3:end,:,:], agg=mean)
end
function mae_𝑣(ŷ, y)
    return misfit.mae_no_zeros(descale(𝑣Scale, ŷ[1]), descale(𝑣Scale, y[1]))
end
function mae_𝐗(ŷ, y)
    return misfit.mae_no_zeros(descale(𝐗Scale, ŷ[2]), descale(𝐗Scale, y[2]))
end
function closure_metric((𝑣_ŷ, 𝐗_ŷ), y)
    return misfit.closure_condition((descale(𝑣Scale, 𝑣_ŷ), descale(𝐗Scale, 𝐗_ŷ)), y, agg=mean)
end

metrics = [mass_balance_metric, mae_𝑣, mae_𝐗, closure_metric];

# TUNE IT
#-----------------------------------------------------------------------
hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers[1], batch_size[1], loss,
                              (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
                              masking_f,
                              1000, metrics,
                              lr_schedule = false,
                              subdir_appendix = "cBack_fbl1_bs1")

hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers[1], batch_size[2], loss,
                              (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
                              masking_f,
                              1000, metrics,
                              lr_schedule = false,
                              subdir_appendix = "cBack_fbl1_bs2")

hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers[1], batch_size[3], loss,
                              (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
                              masking_f,
                              1000, metrics,
                              lr_schedule = false,
                              subdir_appendix = "cBack_fbl1_bs3")

hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers[2], batch_size[1], loss,
                              (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
                              masking_f,
                              1000, metrics,
                              lr_schedule = false,
                              subdir_appendix = "cBack_fbl2_bs1")

hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers[2], batch_size[2], loss,
                              (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
                              masking_f,
                              1000, metrics,
                              lr_schedule = false,
                              subdir_appendix = "cBack_fbl2_bs2")

hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers[2], batch_size[3], loss,
                              (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
                              masking_f,
                              1000, metrics,
                              lr_schedule = false,
                              subdir_appendix = "cBack_fbl2_bs3")

println("Hyperparameter tuning complete!")
