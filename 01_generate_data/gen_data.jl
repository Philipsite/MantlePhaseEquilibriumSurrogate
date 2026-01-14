#=
This script re-generates datasets that can be used for the training, validation, and testing of the ML surrogates.

However, instead of re-generating the datasets from scratch, it is recommended to download the pre-generated
datasets from Zenodo using the fetch.jl script.
=#
using Sprout

DIR = joinpath(pwd(), "data", "regenerated_dataset")
mkpath(DIR)

# Compositions after Xu et al. (2008)
Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
HARZBURGITE  = [36.04, 0.79, 0.65, 5.97, 56.54, 0.00]  # Modified Harzburgite from Xu et al. (2008)
BASALT       = [51.75, 13.88, 10.19, 7.06, 14.94, 2.18]

P_range_kbar = (10., 400.)
T_range_C = (500., 2500.)


n_test = 10
filename_base_test = joinpath(DIR, "sb21_02Oct25_t_")
x_test, y_test = generate_dataset(n_test, filename_base_test;
                                    database              = "sb21",
                                    Xoxides               = Xoxides,
                                    sys_in                = "mol",
                                    pressure_range_kbar   = P_range_kbar,
                                    temperature_range_C   = T_range_C,
                                    bulk_em_1             = HARZBURGITE,
                                    bulk_em_2             = BASALT,
                                    noisy_bulk            = true,
                                    λ_dirichlet           = 600,
                                    phase_list            = [PP..., SS...],
                                    save_to_csv           = true)

n_train = 2000000
filename_base_train = joinpath(DIR, "sb21_02Oct25_train_")
x_train, y_train = generate_dataset(n_train, filename_base_train;
                                    database              = "sb21",
                                    Xoxides               = Xoxides,
                                    sys_in                = "mol",
                                    pressure_range_kbar   = P_range_kbar,
                                    temperature_range_C   = T_range_C,
                                    bulk_em_1             = HARZBURGITE,
                                    bulk_em_2             = BASALT,
                                    noisy_bulk            = true,
                                    λ_dirichlet           = 600,
                                    phase_list            = [PP..., SS...],
                                    save_to_csv           = true)

n_val = 100000
filename_base_val = joinpath(DIR, "sb21_02Oct25_val_")
x_val, y_val = generate_dataset(n_val, filename_base_val;
                                    database              = "sb21",
                                    Xoxides               = Xoxides,
                                    sys_in                = "mol",
                                    pressure_range_kbar   = P_range_kbar,
                                    temperature_range_C   = T_range_C,
                                    bulk_em_1             = HARZBURGITE,
                                    bulk_em_2             = BASALT,
                                    noisy_bulk            = true,
                                    λ_dirichlet           = 600,
                                    phase_list            = [PP..., SS...],
                                    save_to_csv           = true)

n_test = 100000
filename_base_test = joinpath(DIR, "sb21_02Oct25_test_")
x_test, y_test = generate_dataset(n_test, filename_base_test;
                                    database              = "sb21",
                                    Xoxides               = Xoxides,
                                    sys_in                = "mol",
                                    pressure_range_kbar   = P_range_kbar,
                                    temperature_range_C   = T_range_C,
                                    bulk_em_1             = HARZBURGITE,
                                    bulk_em_2             = BASALT,
                                    noisy_bulk            = true,
                                    λ_dirichlet           = 600,
                                    phase_list            = [PP..., SS...],
                                    save_to_csv           = true)
