# MantlePhaseEquilibriumSurrogate


## Download Datasets and Trained Models

To run the scripts in this repository, you need to download the generated datasets and trained models from the accompanying [Zenodo repository](https://zenodo.org/records/18154882).

Either download the archives yourself and place them into `data/` and `models/` or simply run the [fetch.jl](fetch.jl) script from the repository root to do this automatically:

> [!CAUTION]
> This will download approximately 2.5GB of data. Make sure you have sufficient disk space and a stable internet connection.


```julia
include("fetch.jl")
```