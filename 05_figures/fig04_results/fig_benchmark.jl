    using JLD2, CSV, DataFrames
    using CairoMakie

    @load joinpath("04_benchmark_model_performance", "benchmark_results.jld2") benchmark_data
    benchmark_data[!, :trial_time_s] = benchmark_data.trial_time ./ 1e9  # Convert from nanoseconds to seconds
    # add trial number for plotting purposes
    benchmark_data[!, :trial_number] = repeat(1:5, 12)


    fig = Figure(size = (600, 400))
    ax = Axis(fig[1, 1]; xlabel = L"\textrm{Phase equilibrium predictions}", ylabel = L"\textrm{Wall time [s]}", yscale = log10, xscale = log10)
    methods = unique(benchmark_data.Method)
    device = unique(benchmark_data.device)
    trial_numbers = unique(benchmark_data.trial_number)
    colors = [:royalblue3, :deeppink4]
    marker = [:circle, :diamond]

    # Reference lines for Big O scaling
    n_ref = 10 .^ range(log10(minimum(benchmark_data.N_minimizations)),
                        log10(maximum(benchmark_data.N_minimizations)), length=100)

    # Choose a reference point to anchor the scaling lines (adjust t0 to position vertically)
    t0 = 8e-4  # base time at n=1 for scaling reference
    n0 = 1.0   # reference n

    # O(n) - linear scaling
    lines!(ax, n_ref, t0 .* (n_ref ./ n0); color = :gray, linestyle = :dot, linewidth = 2, label = "O(n)")

    # O(1) - constant scaling
    lines!(ax, n_ref, t0 .* ones(length(n_ref)); color = :gray, linestyle = :dash, linewidth = 2, label = "O(1)")

    scatter!(ax,
        benchmark_data.N_minimizations,
        benchmark_data.trial_time_s;
        color = [colors[findfirst(==(m), methods)] for m in benchmark_data.Method],
        marker = [marker[findfirst(==(d), device)] for d in benchmark_data.device],
        markersize = 12,
        label = ""
    )

    for (m, d) in Iterators.product(methods, device)
        for tn in trial_numbers
            pts = (benchmark_data.Method .== m) .&
                (benchmark_data.device .== d) .&
                (benchmark_data.trial_number .== tn)
            lines!(ax,
                benchmark_data.N_minimizations[pts],
                benchmark_data.trial_time_s[pts];
                color = colors[findfirst(==(m), methods)],
                linestyle = :solid,
                alpha = 0.7
            )
        end
    end

    l1 = LineElement(color = :royalblue3, linestyle = :solid, linewidth = 2, label = "MAGEMin")
    l2 = LineElement(color = :deeppink4, linestyle = :solid, linewidth = 2, label = "Surrogate Model")
    m1 = MarkerElement(marker = :circle, color = :black, label = "CPU")
    m2 = MarkerElement(marker = :diamond, color = :black, label = "GPU")
    s1 = LineElement(color = :gray, linestyle = :dot, linewidth = 2, label = "O(n)")
    s2 = LineElement(color = :gray, linestyle = :dash, linewidth = 2, label = "O(1)")

    ax.yticks = (10. .^ collect(-3:1:3), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"])
    ax.yminorticksvisible = true
    ax.yminorticks = vcat(collect(0.001:0.001:0.01),
                        collect(0.01:0.01:0.1),
                        collect(0.1:0.1:1.0),
                        collect(1.0:1.0:10.0),
                        collect(10.0:10.0:100.0),
                        collect(100.0:100.0:1000.0))

    Legend(fig[1, 2], [[l1, l2], [m1, m2], [s1, s2]],
        [["MAGEMin", "Surrogate"], ["CPU (12 threads)", "GPU (3'072 cores)"], [L"\mathcal{O}(n)", L"\mathcal{O}(1)"]],
        ["Method", "Device", "Theoretical"];
        framevisible = false, orientation = :vertical, valign = :top, titlehalign = :left, gridshalign = :left)

    fig
    save(joinpath("04_figures", "fig04_results", "benchmark_results.pdf"), fig)
