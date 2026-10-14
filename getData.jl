module getData

using JuMP, Gurobi, CSV, DataFrames, LinearAlgebra, StatsBase

export Data

    mutable struct data
        Tf::Int64 # Number of periods
        Ig::Int64 # Number of generators
        Tl::Int64 # Number of lines
        bus_renewable_generation::Array{Int64,1} # Index set for buses with renewable generation injection
        ind_renewable_generation::Dict{Int64, Int64} # Dictionary for positions within the index set of buses with renewable generation injection
        D::Array{Array{Float64,1},1} # Load demand
        F::Array{Float64,2} # Line-bus incidence matrix
        B::Array{Float64,2} # Thermal generator-bus incidence matrix
        S::Array{Float64,2} # Angle-to-flow matrix
        Cu::Array{Float64,1} # Vector of fixed cost for thermal units
        Csu::Array{Float64,1} # Start-up cost vector for thermal units
        Csd::Array{Float64,1} # Shut-down cost vector for thermal units
        Cp::Array{Float64,1} # Vector of fuel costs for thermal units
        RU::Array{Float64,1} # Ramp-up limit vector for thermal units
        RD::Array{Float64,1} # Ramp-down limit vector for thermal units
        Pmin::Array{Float64,1} # Vector of minimum generation limits for thermal units
        Pmax::Array{Float64,1} # Vector of maximum generation limits for thermal units
        Fmax::Array{Float64,1} # Vector of line capacities
        Cdef::Int64 # load shedding cost
        ξ::Array{Array{Array{Float64,1},1},1} # Scenarios for load demand minus renewable generation at buses with renewable generation injection
        z::Array{Array{Array{Float64,1},1},1} # Scenarios for features
        zstar::Array{Array{Float64,1},1} # Contextual observation
        Y_b::Array{Array{Float64,2},1} # Matrix scenarios for load demand
        X_lag_b::Array{Array{Float64,2},1} # Matrix scenarios for features
        N::Int64 # Number of scenarios
    end

    syst = "118_syst"

    generators = CSV.read(joinpath(pwd(), syst, "generators.csv"), DataFrame; header=false)
    lines = CSV.read(joinpath(pwd(), syst, "lines.csv"), DataFrame; header=false)
    maximum_load = CSV.read(joinpath(pwd(), syst, "maximum_load.csv"), DataFrame; header=false)
    load_distribution_profile = CSV.read(joinpath(pwd(), syst, "load_distribution_profile.csv"), DataFrame; header=false)[:,2]/100

    """Scenario"""
    renewable_generation_profile = Matrix(CSV.read(joinpath(pwd(), syst, "renewable_generation_profile.csv"), DataFrame;header=true))[:,2:end]
    S,  = size(renewable_generation_profile)
    maximum_renewable_generation = CSV.read(joinpath(pwd(), syst, "maximum_renewable_generation.csv"), DataFrame;header=false)
    bus_renewable_generation = maximum_renewable_generation[:,1]
    N = 50 # Number of scenarios

    Tf = 24 
    Ig = Int(generators[end,1]) 
    Tl = Int(lines[end,1]) 
    Total_Buses = Int(maximum_load[end,1]) 

    D = []
    for b = 1:Total_Buses
        load = maximum_load[b,2]*load_distribution_profile
        push!(D, load)
    end

    ind_renewable_generation = Dict()
    for b in bus_renewable_generation
        b_ = findfirst(bus_renewable_generation.== b)
        ind_renewable_generation[b] = b_
    end

    ξ = []
    z = []
    zstar = []
    Y_b = []
    X_lag_b = []
    k = 1 # lag
    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        renewable_generation_b = reshape(renewable_generation_profile[:,b_]*maximum_renewable_generation[b_,2],24,Int(S/24))
        # push!(ξ, [max.(D[b] - s[:],zeros(Tf)) for s in eachcol(renewable_generation_b)][1:N])
        Days = size(renewable_generation_b, 2) - 1
        target_range = (Days - N + 1):Days
        lag_range = (Days - N + 1 - k):(Days - k)
        Y_b_ = renewable_generation_b[:, target_range]
        X_lag_b_ = renewable_generation_b[:,lag_range]
        push!(ξ, [s[:] for s in eachcol(Y_b_)])
        push!(z, [s[:] for s in eachcol(X_lag_b_)])
        push!(Y_b, Y_b_)
        push!(X_lag_b, X_lag_b_)
        push!(zstar, trunc.(renewable_generation_b[:, end], digits=2))
    end

    """ Net Power Flow """

    F = zeros(Total_Buses,Tl)
    From_bus_To_bus = Matrix(lines[:,2:3])
    for i=1:Total_Buses, l=1:Tl
        if From_bus_To_bus[l,1] == i
            F[i,l] = 1
        end
        if From_bus_To_bus[l,2] == i
            F[i,l] = -1
        end
    end

    Bus = generators[:,2]
    B = zeros(Total_Buses,Ig)
    for i=1:Total_Buses, j=1:Ig
        if Bus[j] == i
            B[i,j] = 1
        end
    end

    Reactance = lines[:,4]
    Aux = (1 ./ Reactance).*Diagonal(ones(Tl))
    S = Matrix((F*Aux)')

    Cu = generators[:,3]
    Cp = generators[:,4]
    Pmin = generators[:,6]
    Pmax = generators[:,5]
    Fmax = lines[:,end]
    RU = generators[:,7]
    RD = generators[:,7]
    Cup = generators[:,9]
    Cdn = generators[:,8]
    Csu = generators[:,10]
    Csd = generators[:,10]
    Cdef = 1e5

    Data = data(Tf, Ig, Tl, bus_renewable_generation, ind_renewable_generation, D, F, B, S, Cu, Csu, Csd, Cp, RU, RD, Pmin, Pmax, Fmax, Cdef, ξ, z, zstar, Y_b, X_lag_b, N)

end