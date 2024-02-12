"""
Comrade M87 Stokes I Imaging Pipeline for EHT observations in April 2018

Authors: The Event Horizon Telescope Collaboration et al.
Date: January 18, 2024
Primary Reference: The Event Horizon Telescope Collaboration, et al. 2024, A&A, 681, A79
Data Product Code: 2024-D01-01

Brief Description:
The pipeline reconstructs an image from uvfits files simultaneously
released in the EHT website (data release ID: 2024-D01-01) using Comrade,
a Bayesian modeling package, targeted for very-long-baseline interferometry (VLBI) 
and written in the Julia1 programming language (Tiede, P., (2022). JOSS, 7(76), 4457).

To run the pipeline, specify the input uvfits data file name without the path. 
It is assumed that all uvfits are in "./data/" folder. Additional you have to specify,
epoch as "3644" (April 21) or "3647" (April 25) and band from one of "b1", "b2, "b3", "b4".
Epoch and Band is required only to choose the hyperparameters like number of pixels and field
of fiew which depend on the uv-coverage.

Example call:
julia comrade_2018m87_pipeline.jl --uvfits "hops_3644_M87_b3.netcal_10s_StokesI.uvfits" --epoch "3644" --band "b3"

Additional References:
 - EHT Collaboration Data Portal Website:
   https://eventhorizontelescope.org/for-astronomers/data
 - The Event Horizon Telescope Collaboration, et al. 2024, A&A, 681, A79
 - Comrade: https://github.com/ptiede/Comrade.jl 
 - Tiede, P., (2022), Journal of Open Source Software, 7(76), 4457, https://doi.org/10.21105/joss.04457
"""

"""
Julia Version 1.8.2
Commit 36034abf260 (2022-09-29 15:21 UTC)

  Status `~/Project.toml`
  [c7e460c6] ArgParse v1.1.4
⌃ [6e4b80f9] BenchmarkTools v1.3.2
⌃ [336ed68f] CSV v0.10.9
  [13f3f980] CairoMakie v0.11.6
⌅ [99d987ce] Comrade v0.6.10
⌃ [a4336a5c] ComradeAHMC v0.2.2
⌃ [26988f03] ComradeOptimization v0.1.2
⌃ [a93c6f00] DataFrames v1.3.6
⌃ [31c24e10] Distributions v0.25.80
⌃ [ced4e74d] DistributionsAD v0.6.43
⌃ [be115224] MCMCDiagnosticTools v0.2.1
  [0a4f8689] MathTeXEngine v0.5.7
⌃ [eff96d63] Measurements v2.8.0
  [d9ec5142] NamedTupleTools v0.14.3
⌅ [36348300] OptimizationOptimJL v0.1.5
⌃ [b1d3bc72] Pathfinder v0.6.2
⌅ [91a5bcdd] Plots v1.38.4
⌃ [438e738f] PyCall v1.95.1
  [860ef19b] StableRNGs v1.0.1
⌅ [2913bbd2] StatsBase v0.33.21
⌃ [bd369af6] Tables v1.10.0
⌃ [9d95f2ec] TypedTables v1.4.1
⌃ [b1ba175b] VLBIImagePriors v0.1.0
⌃ [e88e6eb3] Zygote v0.6.55
  [de0858da] Printf
Info Packages marked with ⌃ and ⌅ have new versions available, 
but those with ⌅ are restricted by compatibility constraints from upgrading. 
To see why use `status --outdated`
"""

# Activate the project environment and load the packages
using Pkg;Pkg.activate(@__DIR__)
#When runing this script for the first time
# Pkg.instantiate()
using ArgParse
using Comrade
using Distributions
using ComradeOptimization
using ComradeAHMC
using StatsBase
using OptimizationOptimJL
using VLBIImagePriors
using DistributionsAD
using NamedTupleTools
using Zygote
using DataFrames
using CSV
using Pathfinder
using LinearAlgebra
using Serialization
using BenchmarkTools
# Plotting Setup
import CairoMakie as CM
using MathTeXEngine # required for texfont
textheme = CM.Theme(fonts=(; regular=texfont(:text),
                        bold=texfont(:bold),
                        italic=texfont(:italic),
                        bold_italic=texfont(:bolditalic)))
ENV["GKSwstype"] = "nul"
using Plots
using Tables
using Printf
using Measurements
using TypedTables
using MCMCDiagnosticTools
#For reproducibility
using StableRNGs
rng = StableRNG(123)

#Load ehtim
"""
Comrade uses python library eht-imaging (https://github.com/achael/eht-imaging) to
load the uvfits data. We use PyCall.jl to access our pre-installed python packages.

When runing this script for the first time:
1. Start julia in terminal
2. ENV["PYTHON"]="<your python path>"
3. using Pkg;Pkg.activate(@__DIR__)
4. Pkg.build("PyCall")
5. Restart julia
"""

#Load eht-imaging
using PyCall
load_ehtim()

# Include our fit functions
include("./src/ampcp_fit.jl")

#Parsing arguments function
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--uvfits"
            help = "uvfits file name to read"
            arg_type = String
            required = true
        "--epoch"
             help = "111 or 115 as a string"
             arg_type = String
             default = "111"
             required = true
        "--band"
             help = "Band name b1,b2,b3,b4 as a string"
             arg_type = String
             default = "b3"
             required = true
    end
    return parse_args(s)
end

# Main function
function main()
    #Parse the command line arguments
    parsed_args = parse_commandline()
    #Assign the arguments to specific variables
    file = parsed_args["uvfits"]
    epoch = parsed_args["epoch"]
    band = parsed_args["band"]
    println("Using options: ")
    println("uvfits file: $file, ")
    println("epoch: $epoch")
    println("band: $band")
    
    # Set output directories
    outbase = joinpath(@__DIR__, "results", chop(file,tail=7), chop(file,tail=7))
    outdir = joinpath(@__DIR__, "results", chop(file,tail=7), "samples", chop(file,tail=7))
    
    # hyperparameters for different uv-coverages
    # nx = number of pixels in x-direction
    # fovx = field of view in uas in x-direction
    # fovy = field of view in uas in y-direction
    if epoch=="111"
        if band=="b3" || band=="b4"
            nx=12 
            fovx=90.0
            fovy=90.0
        else
            nx=10 
            fovx=75.0
            fovy=75.0
        end
    else
        if band=="b3" || band=="b4"
            nx=9 
            fovx=67.5
            fovy=67.5
        else
            nx=9 
            fovx=67.5
            fovy=67.5
        end
    end
    
    # Make directories for results
    if !isdir(joinpath(@__DIR__,"results"))
        mkdir(joinpath(@__DIR__,"results"))
        if !isdir(joinpath(@__DIR__,"results", chop(file,tail=7)))
            mkdir(joinpath(@__DIR__,"results", chop(file,tail=7)))
            mkdir(joinpath(@__DIR__,"results", chop(file,tail=7), "samples"))
        end
    end
    
    println("Starting fit")
    image_data("./data/"*file, epoch, band, outbase, outdir; nx=nx, fovx=fovx, fovy=fovy)

    done = joinpath(@__DIR__, "results", chop(file,tail=7))
    println("Done! Check $done folder")
    return 0
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end