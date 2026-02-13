    using JLD2, CSV, DataFrames
    using CairoMakie

    @load joinpath("04_benchmark_model_performance", "benchmark_results.jld2") benchmark_data
    benchmark_data[!, :trial_time_s] = benchmark_data.trial_time ./ 1e9  # Convert from nanoseconds to seconds
    # add trial number for plotting purposes
    benchmark_data[!, :trial_number] = repeat(1:5, 12)

    # calculate time per prediction
    benchmark_data[!, :time_per_prediction_s] = benchmark_data.trial_time_s ./ benchmark_data.N_minimizations

    # extract minimal times for each method and device combination
    min_times_µs = combine(groupby(benchmark_data, [:Method, :device]), :time_per_prediction_s => (x -> round(minimum(x) * 10^6, sigdigits=3)) => :min_time_µs)

    fig = Figure(size = (800, 500))
    ga = fig[1, 1] = GridLayout()
    gb = fig[1, 2] = GridLayout()
    g_legend = fig[2, 1:2] = GridLayout()

    ax1 = Axis(ga[1, 1]; xlabel = L"\textrm{Phase equilibrium predictions}", ylabel = L"\textrm{Wall time [s]}", yscale = log10, xscale = log10)
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
    lines!(ax1, n_ref, t0 .* (n_ref ./ n0); color = :gray, linestyle = :dot, linewidth = 2, label = "O(n)")

    # O(1) - constant scaling
    lines!(ax1, n_ref, t0 .* ones(length(n_ref)); color = :gray, linestyle = :dash, linewidth = 2, label = "O(1)")

    scatter!(ax1,
        benchmark_data.N_minimizations,
        benchmark_data.trial_time_s;
        color = [colors[findfirst(==(m), methods)] for m in benchmark_data.Method],
        marker = [marker[findfirst(==(d), device)] for d in benchmark_data.device],
        markersize = 12
    )

    for (m, d) in Iterators.product(methods, device)
        for tn in trial_numbers
            pts = (benchmark_data.Method .== m) .&
                (benchmark_data.device .== d) .&
                (benchmark_data.trial_number .== tn)
            lines!(ax1,
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

    ax1.yticks = (10. .^ collect(-3:1:3), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"])
    ax1.yminorticksvisible = true
    ax1.yminorticks = vcat(collect(0.001:0.001:0.01),
                        collect(0.01:0.01:0.1),
                        collect(0.1:0.1:1.0),
                        collect(1.0:1.0:10.0),
                        collect(10.0:10.0:100.0),
                        collect(100.0:100.0:1000.0))

    ax2 = Axis(gb[1, 1]; xlabel = L"\textrm{Phase equilibrium predictions}", ylabel = L"\textrm{Time per prediction [s]}", yscale = log10, xscale = log10)

    scatter!(ax2,
        benchmark_data.N_minimizations,
        benchmark_data.time_per_prediction_s;
        color = [colors[findfirst(==(m), methods)] for m in benchmark_data.Method],
        marker = [marker[findfirst(==(d), device)] for d in benchmark_data.device],
        markersize = 12
    )

    for (m, d) in Iterators.product(methods, device)
        for tn in trial_numbers
            pts = (benchmark_data.Method .== m) .&
                (benchmark_data.device .== d) .&
                (benchmark_data.trial_number .== tn)
            lines!(ax2,
                benchmark_data.N_minimizations[pts],
                benchmark_data.time_per_prediction_s[pts];
                color = colors[findfirst(==(m), methods)],
                linestyle = :solid,
                alpha = 0.7
            )
        end
    end

    # plot text for minimal times per prediction
    for (m, d) in Iterators.product(methods, device)
        pts = (min_times_µs.Method .== m) .& (min_times_µs.device .== d)
        if any(pts)
            x = maximum(benchmark_data.N_minimizations)
            y = min_times_µs.min_time_µs[pts][1] / 1e6 # convert back to seconds for plotting
            text!(ax2, string(min_times_µs.min_time_µs[pts][1], " µs") , position = (x, y), align = (:right, :top))
        end
    end

    ylims!(ax2, 1.5*1e-8, nothing)

    Legend(g_legend[1, 1], [[l1, l2], [m1, m2], [s1, s2]],
        [[L"\mathcal{O}(n)", L"\mathcal{O}(1)"],["MAGEMin", "Surrogate"], ["CPU (12 threads)", "GPU (3'072 cores)"]],
        ["Theoretical complexity", "Method", "Device"];
        framevisible = false, orientation = :horizontal, valign = :top, titlehalign = :left, gridshalign = :left)

    for (label, layout) in zip(["(a)", "(b)"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
            fontsize = 16,
            font = :bold,
            padding = (0, 5, 5, 0),
            halign = :right)
    end

    fig
    save(joinpath("05_figures", "fig04_results", "benchmark_results.pdf"), fig)
