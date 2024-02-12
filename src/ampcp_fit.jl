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
using PyCall
using Measurements
using TypedTables
using MCMCDiagnosticTools
using StableRNGs
load_ehtim()


# Function to load uvfits file and extract visibilty amplitudes and closure phases
function load_data(file)
    obs = ehtim.obsdata.load_uvfits(file)
    # Add systematic noise of 1% and scan average the data
    obsavg = scan_average(obs.add_fractional_noise(0.01))
    # extract visibilty amplitudes
    damp = extract_amp(obsavg)
    # extract closure phases and remove trivial trinagles and data with SNR<=3
    dcp  = extract_cphase(obsavg, cut_trivial=true, snrcut=3)
    return obsavg, damp, dcp
end


# Define model as raster + large gaussian + amplitude gains
function model(θ, metadata)
    (;f, fg, c, lgamp) = θ #model parameters
    (; fovx, fovy, cache, gcache, pulse) = metadata
    # Construct the image model
    # Produce Stokes I images from parameters
    # f = total flux density
    # fg = fraction of flux of a Gaussian that models extended flux
	img = IntensityMap((f*(1-fg))*c, fovx, fovy, pulse)
	m1 = modelimage(img, cache)
    # Add a Gaussian of 1mas to the raster to model extended flux
    g1 = (f*fg)*stretched(Gaussian(), μas2rad(1000.0), μas2rad(1000.0))
    m = m1+g1
	# Now corrupt the model with gains
	g = exp.(lgamp)
	Comrade.GainModel(gcache, g, m)
end

# Define prior distributions for amplitude gains
# We model the gains for all scans and stations.
# The Gaussian widths are different for different days
# Normal(0.0, 0.1) == gains with no offset but 10% deviations since σ=0.1
# Note that it kept different for different band because GLT is only for bands 3 & 4.
function create_calib_priors(damp,epoch,band)
    if epoch=="111"
        if band=="b3" || band=="b4"
            distamp = (AA = Normal(0.0, 0.1),
                    AX = Normal(0.0, 0.1),
                    LM = Normal(0.0, 0.3),
                    SW = Normal(0.0, 0.1),
                    MM = Normal(0.0, 0.1),
                    PV = Normal(0.0, 0.1),
                    MG = Normal(0.0, 0.1),
                    GL = Normal(0.0, 1.0)
                    )
        else
            distamp = (AA = Normal(0.0, 0.1),
                    AX = Normal(0.0, 0.1),
                    LM = Normal(0.0, 0.3),
                    SW = Normal(0.0, 0.1),
                    MM = Normal(0.0, 0.1),
                    PV = Normal(0.0, 0.1),
                    MG = Normal(0.0, 0.1)
                    )
        end
    else
        if band=="b3" || band=="b4"
            distamp = (AA = Normal(0.0, 0.1),
                    AX = Normal(0.0, 0.1),
                    SW = Normal(0.0, 0.1),
                    MM = Normal(0.0, 0.1),
                    PV = Normal(0.0, 1.0),
                    MG = Normal(0.0, 0.1),
                    GL = Normal(0.0, 1.0)
                    )
        else
            distamp = (AA = Normal(0.0, 0.1),
                    AX = Normal(0.0, 0.1),
                    SW = Normal(0.0, 0.1),
                    MM = Normal(0.0, 0.1),
                    PV = Normal(0.0, 1.0),
                    MG = Normal(0.0, 0.1)
                    )
        end
    end
    return NamedTupleTools.select(distamp, stations(damp))
end

# Prior distributions for all model parameters
function create_priors(damp, dcp, X, Y, epoch, band, alpha=1.0)
    # Gain priors
    distamp = create_calib_priors(damp,epoch,band)
    nx = length(X)
    ny = length(Y)
    prior = (
             c = ImageDirichlet(alpha, ny, nx), # Raster Pixels 
             f = Uniform(0.0, 1.5),             # Total flux density
             fg = Uniform(0.0,1.0),             # Fraction of flux of the Gaussian
	         lgamp = Comrade.GainPrior(distamp, scantable(damp))
             )
    return prior
end

# function for metadata of the model
function create_metadata(damp, fovx, fovy, nx, ny, pulse)
    buffer = IntensityMap(zeros(ny, nx), fovx, fovy, pulse)
    cache = create_cache(DFTAlg(damp), buffer)
    gcache = GainCache(scantable(damp))
    return  (;fovx, fovy, cache, gcache, pulse)
end

# Function to find the maximum a posteriori image
function best_image(tpost, ntrials=25, maxiters=10_000)
    # Dimensions of the transformed posterior
    ndim = dimension(tpost)
    # Optimization problem to find minima of the transformed posterior
    f = OptimizationFunction(tpost, Optimization.AutoZygote())
    # Run optimizer many times to find the minima
    # We use a LBFGS optimizer
    sols = map(1:ntrials) do i
        prob = OptimizationProblem(f, rand(rng,ndim) .- 0.5, nothing)
        sol = solve(prob, LBFGS(); maxiters=maxiters÷2, g_tol=1e-1)
        @info "Preliminary image $i/$(ntrials) done: minimum: $(sol.minimum)"

        prob = OptimizationProblem(f, sol.u .+ randn(rng,ndim)*0.05, nothing)
        sol = solve(prob, LBFGS(); maxiters, g_tol=1e-1)
        @info "Best image $i/$(ntrials) done: minimum: $(sol.minimum)"
        return sol
    end
    lmaps = logdensityof.(Ref(tpost), sols)
    # Sort the solutions based on log density of the posterior
    inds = sortperm(lmaps, rev=true)
    return sols[inds], lmaps[inds]
end

# For creating effective sample size map from MCMC chain
function ess_map(chain::Table)
    pname = propertynames(chain)
    esses = map(p->ess_map(getproperty(chain, p)), pname)
    return NamedTuple{pname}(esses)
end

function ess_map(chain::AbstractVector{<:Real})
    return ess_rhat(reshape(chain, :, 1, 1)) |> first
end

function ess_map(chain::AbstractVector{<:AbstractArray})
    cstack = reduce(vcat, reshape.(chain, 1, :))
    return ess_rhat(reshape(cstack, size(cstack, 1), 1, size(cstack, 2)), ) |> first
end

# Actual fit function
# A function to run all above functions, sample the posterior, save results
function image_data(fname, epoch, band, outbase, outdir;
                  nx=12, fovx=90.0, fovy=90.0, alpha=1.0,
                  maxiters=10_000, ntrials=20,
                  nsample = 12_000,
                  nadapt = 10_000,
                  nsamples = 500
                  )
    # extract visibilty amplitudes and closure phases
    obs, damp, dcphase = load_data(fname)

    @info "There are $(length(damp)) visibilities"

    # Set hyperparameters and metadata
    ny = floor(Int, fovy/fovx*nx)
    fovxuas = μas2rad(fovx)
    fovyuas = μas2rad(fovy)
    metadata = create_metadata(damp, fovxuas, fovyuas, nx, ny, BSplinePulse{3}())

    #Define posterior
    X, Y = imagepixels(fovxuas, fovyuas, nx, ny)
    prior = create_priors(damp, dcphase, X, Y, epoch, band, alpha)
    lklhd = RadioLikelihood(damp, dcphase)
    post = Posterior(lklhd, prior, model, metadata)

    # transform the posterior to a flat space (-∞,∞) : faster for sampling
    tpost = asflat(post)
    ndim = dimension(tpost) # Dimensions of the transformed posterior
    ℓ = logdensityof(tpost) # Log density of the transformed posterior

    @info "Benchmarking logdensity"
    @time ℓ(rand(ndim))
    @time ℓ(rand(ndim))

    @info "Benchmarking logdensity gradient"
    @time Zygote.gradient(ℓ, ((rand(ndim))))
    @time Zygote.gradient(ℓ, ((rand(ndim))))

    # Finding the maximum a posteriori image (MAP)
    sols, ℓopt = best_image(tpost, ntrials, maxiters)

    # save 5 least negative log density images
    for i in 1:(min(5, length(ℓopt)))
        @info "Optimimum $i: $(ℓopt[i])"
        x = Comrade.transform(tpost, sols[i])
        img = intensitymap(model(x, metadata), fovxuas, fovyuas, 16*nx, 16*ny)
        Comrade.save(outbase*"_optimum_image_$i.fits", img)
    end

    # Get the parameters of the least negative log density image
    xopt = Comrade.transform(tpost, sols[begin])

    # Save optimum caltables
    # Plot the calibration tables for gains
    gcache = metadata.gcache
    gL = Comrade.caltable(gcache, exp.(xopt.lgamp))
    CSV.write(outbase*"_gains_optimum.csv", gL)

    # combined chi-square of visibility amplitude and closure phases of the MAP
    c2 = χ²(model(xopt, metadata), damp, dcphase)

    @info "chi-square: $(c2/(length(damp) + length(dcphase)))"

    # Plot the residuals for the MAP
    p1 = residual(model(xopt, metadata), damp, dpi=500)
    savefig(p1, outbase*"_residuals_amp.png")
    p1b = residual(model(xopt, metadata), dcphase, dpi=500)
    savefig(p1b, outbase*"_residuals_cp.png")

    # Plot the MAP and save it
    img = intensitymap(model(xopt, metadata), fovxuas, fovxuas, nx*16, nx*16)
    CM.with_theme(textheme) do
        fig = CM.Figure(;size=(800, 800))
        CM.image(fig[1,1], img, axis=(xreversed=true, aspect=1, title="MAP Image"), colormap=:afmhot)
        CM.save(outbase*"_MAP.png", fig)
    end

    Comrade.save(outbase*"_map.fits", img)

    # Save the parameters of the MAP
    serialize(outbase*"_optimum_allres.jls",
                Dict(:xopt=>xopt,
                     :lopt=>logdensityof(tpost, sols[1].u),
                     :model=>model,
                     :metadata=>metadata,
                     :chi2=>c2))

    # Based on the MAP location in the negative log posterior define a starting location for sampling it.
    res = pathfinder(
        ℓ, ℓ';
        init=sols[begin].u .+ 0.05*randn(rng,ndim),
        dim = ndim,
        optimizer=LBFGS(m=6),
        g_tol=1e-1,
        maxiters=1000,
        )

    x0 = Comrade.transform(tpost, res.draws[:,1])
    metric = DiagEuclideanMetric(diag(res.fit_distribution.Σ))
    # HMC sampling of 12000 steps with 10000 adpatation steps.
    # We use a AHMC sampler based on the NUTS sampler.
    hchain, stats = StatsBase.sample(post, AHMC(;metric, autodiff=AD.ZygoteBackend()), nsample; nadapts=nadapt, init_params=x0, progress=true)

    # Save the chain, model and data
    serialize(outbase*"_optimum_chain.jls",
        Dict(:chain=>hchain,
            :stats=>stats,
            :model=>model,
            :metadata=>metadata,
            :damp => damp,
            :dcphase => dcphase)
            )
    CSV.write(outbase*"_chain.csv", hchain)

    # Remove the adpatation steps from the chain
    results = deserialize(outbase*"_optimum_chain.jls")
    hchain = results[:chain][nadapt:end]

    #Plot the gain table with error bars
    gamps = exp.(hcat(hchain.lgamp...))
    mga = mean(gamps, dims=2)
    sga = std(gamps, dims=2)
    gmeas = measurement.(mga, sga)
    ctable = caltable(gcache, vec(gmeas))
    p5 = plot(ctable, layout=(3,3), size=(600, 500), datagains=true, dpi=500)
    display(p5)
    savefig(outbase*"_gains.png")
		
    # Export gain mean and std in CSV files
    gamps = exp.(hcat(hchain.lgamp...))
    mga_CSV = mean(gamps, dims=2)
    sga_CSV = std(gamps, dims=2)
    ctable_m = caltable(gcache, vec(mga_CSV))
    ctable_s = caltable(gcache, vec(sga_CSV))
    CSV.write(outbase*"_gains_mean.csv", ctable_m)
    CSV.write(outbase*"_gains_std.csv", ctable_s)
    
    # Save sampled images 
    chain = sample(results[:chain][nadapt:end], nsamples)
    for i in 1:nsamples
        smodel = model(chain[i], metadata)
        img = intensitymap(smodel, fovxuas, fovxuas, nx*16, ny*16)

        outim = @sprintf "_image_%03d.fits" i
        Comrade.save(outdir*outim, img, damp)
    end
    
    # Save Mean, Std, Error fits file and plots
    results = deserialize(outbase*"_optimum_chain.jls")
    hchain = results[:chain][nadapt:end]
    samples = model.(sample(hchain, nsamples), Ref(metadata))
    imgs = intensitymap.(samples, fovxuas, fovxuas, nx*16, ny*16)

    # Align the sampled images using their centriod 
    shifted_imgs = map(similar, imgs)
    for i in eachindex(shifted_imgs)
            xc = centroid(imgs[i])
            smodel = shifted(samples[i].model, -xc[1], -xc[2])
            sftimg = intensitymap(smodel, imgs[i].fovx, imgs[i].fovy, size(imgs[i],2), size(imgs[i],1))
            shifted_imgs[i] .= sftimg .+ 1e-10 # add a floor to prevent zero flux
    end

    mimg, simg = mean_and_std(shifted_imgs)
    
    save(outbase*"_mean.fits", mimg)
    save(outbase*"_std.fits", simg)
    save(outbase*"_error.fits", simg./(mimg))

    #Plot for image statistics 
    CM.with_theme(textheme) do
        fig = CM.Figure(;size=(800, 800))
        CM.image(fig[1,1], mimg,
                        axis=(xreversed=true, aspect=1, title="Mean Image"),
                        colormap=:afmhot)
        CM.image(fig[1,2], simg./(max.(mimg, 1e-5)),
                        axis=(xreversed=true, aspect=1, title="1/SNR",),
                        colormap=:afmhot)
        CM.image(fig[2,1], imgs[1],
                        axis=(xreversed=true, aspect=1,title="Draw 1"),
                        colormap=:afmhot)
        CM.image(fig[2,2], imgs[end],
                        axis=(xreversed=true, aspect=1,title="Draw 2"),
                        colormap=:afmhot)
        fig
        CM.save(outbase*"_imstats.png", fig)
    end

    #  Residuals for 10 sampled images
    p = plot();
    for s in sample(hchain, 10)
        residual!(p, model(s,metadata), damp, dpi=500)
    end
    p
    savefig(p, outbase*"_residuals_amp_sampled.png")

    p = plot();
    for s in sample(hchain, 10)
        residual!(p, model(s,metadata), dcphase, dpi=500)
    end
    p
    savefig(p, outbase*"_residuals_cp_sampled.png")

    # Serialize the ess value of chain (adpatations removed)
    results = deserialize(outbase*"_optimum_chain.jls")
    hchain = results[:chain][nadapt:end]
    ess = ess_map(hchain)
    serialize(outbase*"_optimum_ess_chain.jls", Dict(:ess=>ess))

    res = nothing
    hchain = nothing
    stats = nothing
    GC.gc()
    return nothing
end


