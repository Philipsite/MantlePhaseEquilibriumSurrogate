# MantlePhaseEquilibriumSurrogate


> **Publication in preparation** — This repository contains the code and data accompanying the manuscript "A machine-learning surrogate for Gibbs Free Energy minimization-based phase equilibrium modelling of mantle rocks on a planetary scale, Hartmeier and Lanari (in prep.)".

This contribution provides a machine-learning surrogate model for predicting mantle phase equilibria. The surrogate replaces computationally expensive Gibbs energy minimization (GEM) calculations with fast neural network inference, enabling rapid predictions of stable mineral assemblages, phase proportions, and compositions over a wide range of mantle *P*–*T*-*X* conditions.

The model is trained on data generated using [MAGEMin](https://github.com/ComputationalThermodynamics/MAGEMin) ([Riel et al., 2022](https://doi.org/10.1029/2022GC010427)) with the thermodynamic database of [Stixrude and Lithgow-Bertelloni (2022)](https://doi.org/10.1093/gji/ggab394). The machine-learning surrogate is built using [Sprout.jl](https://github.com/Philipsite/Sprout.jl) :seedling:, a Julia package for phase equilibrium surrogate modeling built using [Flux.jl](https://github.com/FluxML/Flux.jl).

---

## Installation

1. Clone this repository:
   ```zsh
   git clone https://github.com/Philipsite/MantlePhaseEquilibriumSurrogate.git
   cd MantlePhaseEquilibriumSurrogate
   ```

2. Start Julia in the repository root (multi-threaded for faster model predictions)
   ```zsh
   julia -t auto
   ```

3. Activate and instantiate the project environment:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

This will install all required dependencies as specified in the `Project.toml`  and `Manifest.toml` files.

---

## Download Datasets and Trained Models

To run the scripts in this repository, you need to download the generated datasets and trained models from the accompanying [Zenodo repository](https://zenodo.org/records/18154881).

Either download the archives yourself and place them into `data/` and `models/`, or simply run the [fetch.jl](fetch.jl) script from the repository root to do this automatically:

> [!CAUTION]
> This will download approximately 2.5GB of data. Make sure you have sufficient disk space and a stable internet connection.

```julia
include("fetch.jl")
```

---

## Repository Contents

### MantlePhaseEquilibriumSurrogate

- **01 Generate Data**: Scripts to regenerate training, validation, and test datasets using MAGEMin.
- **02 Hyperparameter Tuning**: Scripts for hyperparameter optimization of the surrogate model.
- **03 Model Calibration**: Scripts for training the phase assemblage classifier and the full surrogate model.
- **04 Benchmark Model Performance**: Script for benchmarking surrogate performance against MAGEMin on CPU and GPU.
- **05 Figures**: Scripts for generating the main manuscript figures.
- **05 Figures Supplementary**: Scripts for generating supplementary figures.

---
(created by fetch.jl download script:)
- **data/**: Contains the generated datasets, hyperparameter tuning results, and out-of-distribution test data.
- **models/**: Contains the trained phase assemblage classifier and surrogate models.


## Acknowledgements
### References

- [Innes et al. (2018)](https://arxiv.org/abs/1811.01457): Fashionable Modelling with Flux, arXiv:1811.01457.
- [Riel et al. (2022)](https://doi.org/10.1029/2022GC010427): MAGEMin,an efficient Gibbs energy minimizer: Application to igneous systems. Geochemistry, Geophysics, Geosystems.
- [Stixrude, L. and Lithgow-Bertelloni, C. (2022)](https://doi.org/10.1093/gji/ggab394): Thermal expansivity, heat capacity and bulk modulus of the mantle, Geophysical Journal International.

### Funding
This work was supported by the Swiss National Science Foundation (SNSF) trough the ThermoFORGE project under grant number [10007192](https://data.snf.ch/grants/grant/10007192).