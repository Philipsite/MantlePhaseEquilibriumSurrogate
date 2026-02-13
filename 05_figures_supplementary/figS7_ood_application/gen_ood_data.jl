using Sprout

DIR = joinpath(pwd(), "data", "ood_dataset")
mkpath(DIR)

# Altered Oceanic Crust after Kelley et al. (2003)
Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
MOLAR_MASS = [60.0843; 56.0774; 101.9613; 71.8444; 40.3044; 61.9789]
AOC_wt = [49.23, 13.03, 12.05, 13.72 * 0.8998, 6.22, 2.30]
AOC = AOC_wt ./ MOLAR_MASS
AOC ./= sum(AOC)

P_range_kbar = (10., 400.)
T_range_C = (500., 2500.)

n_test = 100000
filename_base_test = joinpath(DIR, "AOC_test_data_")
x_test, y_test = generate_dataset(n_test, filename_base_test;
                                    database              = "sb21",
                                    Xoxides               = Xoxides,
                                    sys_in                = "mol",
                                    pressure_range_kbar   = P_range_kbar,
                                    temperature_range_C   = T_range_C,
                                    bulk_em_1             = AOC,
                                    bulk_em_2             = AOC,
                                    noisy_bulk            = true,
                                    λ_dirichlet           = 600,
                                    phase_list            = [PP..., SS...],
                                    save_to_csv           = true)
