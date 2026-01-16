
using Flux
using CSV, DataFrames
using JLD2
using Sprout
using CairoMakie


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

# Setup DataLoader
batch_size = 4096;
loader = Flux.DataLoader((x_train, (𝑣_train, 𝐗_ss_train)), batchsize=batch_size, shuffle=true);


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


# SETUP MODEL
#----------------------------------------------------------------------
n_layers = 4;
n_neurons = 400;
fraction_backbone_layers = 1//2;

masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));
# Load CLASSIFIER
m_classifier = create_classifier_model(2, 200, 8, 20);
model_state = JLD2.load(joinpath("models", "classifier", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m_classifier, model_state);

model = create_model_pretrained_classifier(fraction_backbone_layers, n_layers, n_neurons,
                                           masking_f, m_classifier;
                                           out_dim_𝑣 = 20, out_dim_𝐗 = (6, 14), scaled_FC = true) |> gpu_device()
opt_state = Flux.setup(Flux.Adam(0.001), model)

early_stopping = Flux.early_stopping((val_loss) -> val_loss, 10, init_score=Inf32)


# TRAIN MODEL
#-----------------------------------------------------------------------
model, opt_state, logs_t, dir = train_loop(
    model,
    loader,
    opt_state,
    (x_val, (𝑣_val, 𝐗_ss_val)),
    loss,
    1000,
    metrics = metrics,
    early_stopping_condition=early_stopping,
    gpu_device = gpu_device(),
    save_to_subdir = "surrogate"
);

# save Norm
@save joinpath(dir, "normalisers.jld2") xNorm 𝐗Scale 𝑣Scale

# POST-TRAINING PLOTS
fig = post_training_plots(logs_t, dir)
