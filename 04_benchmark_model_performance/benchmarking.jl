using CSV, DataFrames
using JLD2
using Flux, CUDA
using Sprout
using MAGEMin_C
using BenchmarkTools
using Hwloc


# LOAD MODELS AND NORMS/SCALERS
#-----------------------------------------------------------------------
MODEL_DIR = "models/surrogate";
@load joinpath(MODEL_DIR, "normalisers.jld2") xNorm 𝐗Scale 𝑣Scale

n_layers = 4;
n_neurons = 400;
fraction_backbone_layers = 1//2;

masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));
# Load CLASSIFIER
m_classifier = create_classifier_model(2, 200, 8, 20);
model = create_model_pretrained_classifier(fraction_backbone_layers, n_layers, n_neurons,
                                           masking_f, m_classifier;
                                           out_dim_𝑣 = 20, out_dim_𝐗 = (6, 14), scaled_FC = true);

@load joinpath(MODEL_DIR, "saved_model.jld2") model_state;
Flux.loadmodel!(model, model_state);

model_gpu = gpu(model);

# LOAD DATA
#-----------------------------------------------------------------------
x_val = CSV.read("data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("data/sb21_02Oct25_val_y.csv", DataFrame);

x_val, 𝑣_val, 𝐗_ss_val, ρ_val, Κ_val, μ_val = preprocess_data(x_val, y_val);

# setup MAGEMin input
MAGEMin_DB = Initialize_MAGEMin("sb21", verbose=false);
pressures = Float64.(x_val[1, 1, :]);
temperatures = Float64.(x_val[2, 1, :]);
Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
X = Float64.(x_val[3:end, 1, :]);
# convert matrix to vector of vectors
X = [X[:, i] for i in 1:size(X, 2)];
sys_in = "mol";

# set-up model input (normalization)
x = xNorm(x_val)
x_gpu = gpu(x)

# BENCHMARKING
#-----------------------------------------------------------------------
N_TRIALS = 5
N_SAMPLES = [100, 1000, 10000, 100000]

benchmark_MAGEMin_inf = (n) -> begin
    out = multi_point_minimization(pressures[1:n], temperatures[1:n], MAGEMin_DB, X=X[1:n], Xoxides=Xoxides, sys_in=sys_in, progressbar=false);
    return nothing
end
benchmark_model_inference = (n) -> begin
    𝑣, 𝐗_ss = model(x[:,:,1:n])
    return nothing
end
benchmark_model_inference_gpu = (n) -> begin
    CUDA.synchronize()
    𝑣, 𝐗_ss = model_gpu(x_gpu[:,:,1:n])
    return nothing
end

# Min_time_per_point_μs = Float64[], N_Samples = Int[], device = String[]

# HARDWARE INFORMATION
println("="^60)
println("HARDWARE INFORMATION")
println("="^60)
println("CPU Threads available: ", Threads.nthreads())
println("CPU Physical cores: ", Hwloc.num_physical_cores())
println("CPU Logical cores: ", Hwloc.num_virtual_cores())
if CUDA.functional()
    println("GPU Device: ", CUDA.name(CUDA.device()))
    println("GPU Memory: ", round(CUDA.totalmem(CUDA.device()) / 1e9, digits=2), " GB")
    println("CUDA version: ", CUDA.runtime_version())
else
    println("GPU: Not available")
end
println("="^60)

benchmark_data = DataFrame(Method = String[], N_minimizations = Int[], trial_time = Float64[], N_Samples = Int[], device = String[])

# (1) Benchmark MAGEMin inference
println("Benchmarking MAGEMin inference...");

for n in N_SAMPLES
    for t in 1:N_TRIALS
        GC.gc()  # Clear memory before benchmark
        benchmark_MAGEMin_inf(n)  # Warm-up

        res_MAGEMin = @benchmark $benchmark_MAGEMin_inf($n) samples = 1
        trial_time = res_MAGEMin.times[1]
        push!(benchmark_data, ( "MAGEMin", n, trial_time, length(res_MAGEMin.times), "CPU" ))
    end
end

Finalize_MAGEMin(MAGEMin_DB)

# (2) Benchmark Model inference
println("Benchmarking Model inference...");

for n in N_SAMPLES
    for t in 1:N_TRIALS
        GC.gc()  # Clear memory before benchmark
        benchmark_model_inference(n)  # Warm-up

        res_model = @benchmark $benchmark_model_inference($n) samples = 1
        trial_time = res_model.times[1]
        push!(benchmark_data, ( "Surrogate Model", n, trial_time, length(res_model.times), "CPU"))
    end
end


# (3) Benchmark Model inference on GPU
println("Benchmarking Model inference on GPU...");

for n in N_SAMPLES
    for t in 1:N_TRIALS
        GC.gc(); CUDA.reclaim()  # Clear CPU and GPU memory
        benchmark_model_inference_gpu(n)  # Warm-up (includes sync)

        res_model_gpu = @benchmark $benchmark_model_inference_gpu($n) samples = 1
        trial_time = res_model_gpu.times[1]
        push!(benchmark_data, ( "Surrogate Model", n, trial_time, length(res_model_gpu.times), "GPU"))
    end
end

@save joinpath("04_benchmark_model_performance", "benchmark_results.jld2") benchmark_data

println("Benchmarking complete. Results saved to benchmark_results.jld2")
