# Comrade M87 Stokes I Imaging Pipeline for EHT observations in April 2018

Authors: The Event Horizon Telescope Collaboration et al. <br>
Date: January 18, 2024  <br>
Primary Reference: The Event Horizon Telescope Collaboration, et al. 2024, A&A, 681, A79  <br>
Data Product Code: 2024-D01-01  <br>

Brief Description:
The pipeline reconstructs an image from uvfits files simultaneously
released in the EHT website (data release ID: 2024-D01-01) using Comrade,
a Bayesian modeling package, targeted for very-long-baseline interferometry (VLBI) 
and written in the Julia programming language (Tiede, P., (2022). JOSS, 7(76), 4457).

Additional References:
 - EHT Collaboration Data Portal Website:
   https://eventhorizontelescope.org/for-astronomers/data
 - The Event Horizon Telescope Collaboration, et al. 2024, A&A, 681, A79
 - Comrade: https://github.com/ptiede/Comrade.jl 
 - Tiede, P., (2022), Journal of Open Source Software, 7(76), 4457, https://doi.org/10.21105/joss.04457

# Pre-requisites

## 1. Install Julia
1. Install julia using [juliaup](https://github.com/JuliaLang/juliaup)
2. Install Julia version 1.8.2 (or lower) using 
```
juliaup add 1.8.2
juliaup default 1.8.2
```

## 2. Install ehtim
Follow the instructions here: https://github.com/achael/eht-imaging to install eht-imaging.

## 3. Clone this repository
```
git clone https://github.com/rohandahale/EHT2018M87_Comrade.git
```

## 4. Setup the project environment
`Manifest.toml` and `Project.toml` files already have necessary package information required to run this pipeline.

When runing this pipeline for the first time:
1. `cd EHT2018M87_Comrade`
2. In Julia REPL,
```
julia> using Pkg;Pkg.activate(@__DIR__)
julia> Pkg.instantiate()
``````


This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.


## 4. Build PyCall
Comrade uses python library `eht-imaging` (https://github.com/achael/eht-imaging) to
load the uvfits data. We use `PyCall.jl` to access our pre-installed python packages.

When runing this pipeline for the first time:
1. `cd EHT2018M87_Comrade`
2. In Julia REPL,
```
julia> ENV["PYTHON"]="<your python path>"
julia> using Pkg;Pkg.activate(@__DIR__)
julia> Pkg.build("PyCall")
```
3. Restart julia


# Running the pipeline
To run the pipeline, specify the input uvfits data file name without the path. 
It is assumed that all uvfits are in "./data/" folder. Additional you have to specify,
epoch as "3644" (April 21) or "3647" (April 25) and band from one of "b1", "b2, "b3", "b4".
Epoch and Band is required only to choose the hyperparameters like number of pixels and field
of fiew which depend on the uv-coverage.

Example call:

```
julia comrade_2018m87_pipeline.jl --uvfits "hops_3644_M87_b3.netcal_10s_StokesI.uvfits" --epoch "3644" --band "b3"
```
