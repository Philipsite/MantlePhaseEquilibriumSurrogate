using JLD2
using ProgressMeter
using Flux, Zygote
using Sprout

# LOAD MODEL
#-----------------------------------------------------------------------
# load normalisers
@load joinpath("models", "surrogate", "normalisers.jld2") xNorm 𝐗Scale 𝑣Scale;
masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));

# Load CLASSIFIER
m_classifier = create_classifier_model(2, 200, 8, 20);
model_state = JLD2.load(joinpath("models", "classifier", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m_classifier, model_state);

# Load REGRESSOR
m = create_model_pretrained_classifier(1//2, 4, 400, masking_f, m_classifier);
model_state = JLD2.load(joinpath("models", "surrogate", "saved_model.jld2"), "model_state");
Flux.loadmodel!(m, model_state);

# extract only the classifier part for assemblage predictions
m_classifier = m.layers[1];


# DEFINE COMPOSITIONS (after Xu et al. 2008)
#-----------------------------------------------------------------------
PYR_XU08 = Float32[0.3871, 0.0294, 0.0222, 0.0617, 0.4985, 0.0011];
HRZ_XU08 = Float32[0.3604, 0.0079, 0.0065, 0.0597, 0.5654, 0.0000];
BAS_XU08 = Float32[0.5175, 0.1388, 0.1019, 0.0706, 0.1494, 0.0218];


# COMPUTE JACOBIAN arrays (P-T) for various compositions
#-----------------------------------------------------------------------
# P-T space
n = 1000
P_bounds = (10.0f0, 400.0f0)    # GPa
T_bounds = (500.0f0, 2500.0f0)  # °C
P = range(P_bounds[1], P_bounds[2], length=n)
T = range(T_bounds[1], T_bounds[2], length=n)

# Reverse P for desired orientation
P_rev = reverse(P)

# Create 2D grid
P_grid = Matrix{Float32}(repeat(P_rev, 1, n))
T_grid = Matrix{Float32}(repeat(T', n, 1))

P_flat = vec(P_grid)
T_flat = vec(T_grid)


Js_PYR = Array{Float32, 3}(undef, n^2, 20, 8)
@showprogress for i in 1:n^2
    x = [P_flat[i], T_flat[i], PYR_XU08...]
    J_i, = Zygote.jacobian(x -> m_classifier(xNorm(x)), x)
    Js_PYR[i, :, :] .= J_i
end

Js_HRZ = Array{Float32, 3}(undef, n^2, 20, 8)
@showprogress for i in 1:n^2
    x = [P_flat[i], T_flat[i], HRZ_XU08...]
    J_i, = Zygote.jacobian(x -> m_classifier(xNorm(x)), x)
    Js_HRZ[i, :, :] .= J_i
end

Js_BAS = Array{Float32, 3}(undef, n^2, 20, 8)
@showprogress for i in 1:n^2
    x = [P_flat[i], T_flat[i], BAS_XU08...]
    J_i, = Zygote.jacobian(x -> m_classifier(xNorm(x)), x)
    Js_BAS[i, :, :] .= J_i
end

# Save Jacobian arrays
data_dir = joinpath("data", "jacobians_classifier")
if !isdir(data_dir)
    mkdir(data_dir)
end
@save joinpath(pwd(), data_dir, "Js_PYR.jld2") Js_PYR
@save joinpath(pwd(), data_dir, "Js_HRZ.jld2") Js_HRZ
@save joinpath(pwd(), data_dir, "Js_BAS.jld2") Js_BAS
