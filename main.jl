using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP, Gurobi, JSON, Distributions, Polyhedra, CDDLib, Dualization, LinearAlgebra, Statistics, Plots, LaTeXStrings
using GLM, DataFrames, Random, NearestNeighbors, StatsPlots
using Plots.Measures
const GUROBI_ENV = Gurobi.Env()

include(joinpath(pwd(),"getData.jl"))

using Main.getData

function save_json(output_dict::Dict, path::String, output_name::String="output")
    jD = JSON.json(output_dict)

    open(joinpath(path, "$output_name.json"), "w") do j
        write(j, jD)
    end

    return
end

function open_json(input_file::String, path::String)
    file_data = JSON.parsefile(joinpath(path, input_file))
    return file_data
end

function scenarios(Data)

    Y_b = Data.Y_b
    X_lag_b = Data.X_lag_b
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    Models = Vector{Any}(undef, length(bus_renewable_generation))
    errors = []

    for b in bus_renewable_generation

        b_ = ind_renewable_generation[b]

        y_vector = vec(Y_b[b_])
        x_vector = vec(X_lag_b[b_])

        df_regresion = DataFrame(Target = y_vector, Lag1 = x_vector)

        model = lm(@formula(Target ~ Lag1), df_regresion)
        
        Models[b_] = model

        forcast = predict(model)

        aux = reshape(df_regresion.Target .- forcast,24,Int(length(y_vector)/24))

        push!(errors, [s[:] for s in eachcol(aux)])

    end

    return Models, errors
   
end

function forecast(Data, zstar, Models)

    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    PredictionStar = []

    for b in bus_renewable_generation

        b_ = ind_renewable_generation[b]

        PredictionStar_ = predict(
            Models[b_],
            DataFrame(Lag1 = zstar[b_])
        )

        push!(PredictionStar, PredictionStar_)

    end

    return PredictionStar

end

function uncertainty_support(Data, zstar, errors)

    z = Data.z
    Tf = Data.Tf
    N = Data.N
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb = length(bus_renewable_generation)

    # Ξ_y = []

    # for b in bus_renewable_generation

    #     b_ = ind_renewable_generation[b]

    #     Z = [z[b_][n][i] for n ∈ sample, i ∈ collect(1:Tf)]
    #     errors = [errors[b_][n][i] for n ∈ sample, i ∈ collect(1:Tf)]

    #     tree = KDTree(permutedims(Z))

    #     k = 3

    #     idxs, dists = knn(tree,zstar[b_],k,true)

    #     Localerrors = errors[idxs,:]

    #     # Contextual dependent uncertainty set Ξ_y(z^*)

    #     Lower = zeros(Tf)

    #     Upper = zeros(Tf)

    #     α = 0.05

    #     for t in 1:Tf

    #         Lower[t] = quantile(Localerrors[:,t],α)

    #         Upper[t] = quantile(Localerrors[:,t],1-α)

    #     end

    #     push!(Ξ_y, Matrix([Lower[:] Upper[:]]))

    # end

    global_residual_scenarios = zeros(Tf * Tb, N)
    global_contextual_scenarios =  zeros(Tf * Tb, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    zstar_global = vcat(zstar...)
    Z = Matrix(global_contextual_scenarios')
    errors = Matrix(global_residual_scenarios')

    tree = KDTree(permutedims(Z))

    k = Data.N

    idxs, dists = knn(tree,zstar_global,k,true)

    Localerrors = errors[idxs,:]

    # Contextual dependent uncertainty set Ξ_y(z^*)

    Lower = zeros(Tf * Tb)

    Upper = zeros(Tf * Tb)

    α = 0.05

    for t in 1:(Tf * Tb)

        Lower[t] = quantile(Localerrors[:,t],α)

        Upper[t] = quantile(Localerrors[:,t],1-α)

    end

    Ξ_y_global = Matrix([Lower[:] Upper[:]])

    # Ξ_y = @views [ Ξ_y_global[((b - 1) * 24 + 1):(b * 24), :] for b in 1:Tb]

    return Ξ_y_global
    
end

function economic_dispatch_problem(Data, u, v, w)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    Tl = Data.Tl
    bus_renewable_generation = Data.bus_renewable_generation
    Cp = Data.Cp
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Fmax = Data.Fmax
    Cdef = Data.Cdef
    D = Data.D
    F = Data.F
    B = Data.B
    S = Data.S
    Tb, = size(B) # Total buses
    Bus = setdiff(collect(1:Tb), bus_renewable_generation) # Buses
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    L = collect(1:Tl) # transmission lines

    # Ignitial state
    p0 = zeros(number_of_generators)
    f0 = zeros(Tl)
    u0 = zeros(number_of_generators)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Method" => 2, "Crossover" => 0))
    @variable(model, β[1:Tb, 1:T[end]]) # phase
    @variable(model, f[1:L[end], 1:T[end]]) # power flow
    @variable(model, p[1:Ig[end], 1:T[end]] >= 0) # generation
    @variable(model, pdef[1:Tb, 1:T[end]] >= 0) # deficit generation
    @variable(model, s[bus_renewable_generation, 1:T[end]] >= 0) # spillage
    @variable(model, s_positivo[b in 1:Tb, t in T] >= 0)
    @variable(model, s_negativo[b in 1:Tb, t in T] >= 0)

    @constraint(model, [t in T], f[:, t] .== S*β[:, t])
    @constraint(model, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s_positivo[b, t] + s_negativo[b, t] == D[b][t]) # Power balance
    @constraint(model, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
    @constraint(model, π[b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] - s_positivo[b, t] + s_negativo[b, t] == 0.0) # Uncertainty
    @constraint(model, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
    @constraint(model, π_s[b in bus_renewable_generation, t in T], s[b, t] <=  0) # Uncertainty
    @constraint(model, [l in L, t in T], f[l,t] <= Fmax[l])
    @constraint(model, [l in L, t in T], f[l,t] >= -Fmax[l])
    @constraint(model, x1[i in Ig, t in T], p[i,t] <= Pmax[i]*u[i, t])
    @constraint(model, x2[i in Ig, t in T], p[i,t] >= Pmin[i]*u[i, t])

    @constraint(model, x3[i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*u[i, t-1] + Pmax[i]*v[i, t])
    @constraint(model, x4[i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t])

    # Ignitial state constraints
    @constraint(model, x5[i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i])
    @constraint(model, x6[i in Ig, t in [1]], -p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i])

    M_penalizacion = 1e7

    @objective(model, Min, sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) + sum(0.1*Cdef*s[b,t] for b in bus_renewable_generation, t in T) + sum(M_penalizacion * (s_positivo[b,t] + s_negativo[b,t]) for b ∈ Bus, t ∈ T))

    return model
end

function create_oracle_problem(Data)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    bus_renewable_generation = Data.bus_renewable_generation

    u = zeros(number_of_generators, Tf)
    v = zeros(number_of_generators, Tf)
    w = zeros(number_of_generators, Tf)

    m = economic_dispatch_problem(Data, u, v, w)
    m_oracle = dualize(m; dual_names = DualNames())
   
    @variable(m_oracle, ηpos[b in bus_renewable_generation, t in T], Bin)
    @variable(m_oracle, ηneg[b in bus_renewable_generation, t in T], Bin)
    @constraint(m_oracle, [b in bus_renewable_generation, t in T], ηpos[b, t] + ηneg[b, t] ≤ 1)
    @constraint(m_oracle, positive_forcast[b in bus_renewable_generation, t in T], ηpos[b, t] + ηneg[b, t] ≥ 0)
    set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    obj = objective_function(m_oracle)

    return m_oracle, obj

end

function modify_oracle(Data, m_oracle, obj, y, Ξ_y, n, u ,v, w, λ, PredictionStar)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    D = Data.D
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    RD = Data.RD
    RU = Data.RU
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    T = collect(1:Tf)
    Ig = collect(1:number_of_generators)
    Tb_renewable = length(bus_renewable_generation)

    y_n = y[n]

    lower = Ξ_y[:,1]
    upper = Ξ_y[:,2]

    yproj_aux = clamp.(y_n, lower, upper)
    Δpos_aux = upper - yproj_aux
    Δneg_aux = yproj_aux - lower

    yproj = @views [yproj_aux[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]

    Δpos = @views [Δpos_aux[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]
    Δneg = @views [Δneg_aux[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]

    ex = sum(m_oracle[Symbol("π[$b,$t]")] *( D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - yproj[ind_renewable_generation[b]][t]) -Δpos[ind_renewable_generation[b]][t] * m_oracle[Symbol("π[$b,$t]")] * m_oracle[:ηpos][b, t] + Δneg[ind_renewable_generation[b]][t] * m_oracle[Symbol("π[$b,$t]")] * m_oracle[:ηneg][b, t] + m_oracle[Symbol("π_s[$b,$t]")] *(PredictionStar[ind_renewable_generation[b]][t] + yproj[ind_renewable_generation[b]][t]) + Δpos[ind_renewable_generation[b]][t] * m_oracle[Symbol("π_s[$b,$t]")] * m_oracle[:ηpos][b, t] - Δneg[ind_renewable_generation[b]][t] * m_oracle[Symbol("π[$b,$t]")] * m_oracle[:ηneg][b, t] - λ*Δpos[ind_renewable_generation[b]][t] * m_oracle[:ηpos][b, t] - λ*Δneg[ind_renewable_generation[b]][t] * m_oracle[:ηpos][b, t] for b ∈ bus_renewable_generation, t ∈ T)

    # ex = sum(m_oracle[Symbol("π[$b,$t]")] * ( D[b][t] - PredictionStar[ind_renewable_generation[b]][t]) - m_oracle[Symbol("π[$b,$t]")] * yproj[ind_renewable_generation[b]][t]  + m_oracle[Symbol("π_s[$b,$t]")] *PredictionStar[ind_renewable_generation[b]][t] + m_oracle[Symbol("π_s[$b,$t]")] *yproj[ind_renewable_generation[b]][t]  for b ∈ bus_renewable_generation, t ∈ T)
    set_objective_function(m_oracle, obj + ex - λ * norm(yproj_aux - y_n, 1))

    # y_opt = @views [yproj[((b - 1) * Tf + 1):(b * Tf)] for b in 1:Tb_renewable]
    # model = economic_dispatch_problem(Data, u, v, w)

    # for b ∈ bus_renewable_generation, t ∈ T

    #     set_normalized_rhs(model[:π][b, t], D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - y_opt[ind_renewable_generation[b]][t])
    #     set_normalized_rhs(model[:π_s][b, t], PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t])

    # end
    # m_oracle = dualize(model; dual_names = DualNames())

    # π, π_s, x1, x2, x3, x4, x5, x6, const_term, obj_aux = dual_vertices(Data, u, v, w, y_opt, PredictionStar)

    for t in T

        for b ∈ bus_renewable_generation

            # fix(m_oracle[Symbol("π[$b,$t]")], π["($b, $t)"], force = true)
            # fix(m_oracle[Symbol("π_s[$b,$t]")], π_s["($b, $t)"], force = true)
            set_normalized_coefficient(m_oracle[:positive_forcast][b, t], m_oracle[:ηpos][b, t], Δpos[ind_renewable_generation[b]][t])
            set_normalized_coefficient(m_oracle[:positive_forcast][b, t], m_oracle[:ηneg][b, t],- Δneg[ind_renewable_generation[b]][t])
            set_normalized_rhs(m_oracle[:positive_forcast][b, t], - PredictionStar[ind_renewable_generation[b]][t] - yproj[ind_renewable_generation[b]][t])

        end

        for i in Ig

            set_objective_coefficient(m_oracle, m_oracle[Symbol("x1[$i,$t]")], Pmax[i]*u[i, t])
            set_objective_coefficient(m_oracle, m_oracle[Symbol("x2[$i,$t]")], Pmin[i]*u[i, t])
            # fix(m_oracle[Symbol("x1[$i,$t]")], x1["($i, $t)"], force = true)
            # fix(m_oracle[Symbol("x2[$i,$t]")], x2["($i, $t)"], force = true)

            if t == 1

                set_objective_coefficient(m_oracle, m_oracle[Symbol("x5[$i,$t]")], RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i])
                set_objective_coefficient(m_oracle, m_oracle[Symbol("x6[$i,$t]")], RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i])
                # fix(m_oracle[Symbol("x5[$i,$t]")], x5["($i, $t)"], force = true)
                # fix(m_oracle[Symbol("x6[$i,$t]")], x6["($i, $t)"], force = true)
                

            else

                set_objective_coefficient(m_oracle, m_oracle[Symbol("x3[$i,$t]")], RU[i]*u[i, t-1] + Pmax[i]*v[i, t])
                set_objective_coefficient(m_oracle, m_oracle[Symbol("x4[$i,$t]")], RD[i]*u[i, t] + Pmax[i]*w[i, t])
                # fix(m_oracle[Symbol("x3[$i,$t]")], x3["($i, $t)"], force = true)
                # fix(m_oracle[Symbol("x1[$i,$t]")], x4["($i, $t)"], force = true)

            end
        end
    end

    # set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Method" => 2, "Crossover" => 0))
    # optimize!(m_oracle)
    # termination_status(m_oracle)
    # objective_value(m_oracle)


end

function solve_oracle_problem(Data, m_oracle, obj, z, y, Ξ_y, u, v, w, λ, μ, θ, oracle_dict, zstar, PredictionStar, initial_cuts, idx)
    
    N = Data.N
    Tf = Data.Tf
    bus_renewable_generation = Data.bus_renewable_generation
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig


    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    T = collect(1:Tf)   
    a = Ξ_y[:,1]
    b = Ξ_y[:,2]
    zstar_global = vcat(zstar...)

    violation = [true for n ∈ 1:N]

    for n ∈ 1:N
    
        val, time_ = @timed(begin
            modify_oracle(Data, m_oracle, obj, y, Ξ_y, n, u ,v, w, λ, PredictionStar)
            optimize!(m_oracle)
            status = termination_status(m_oracle)
            if status != MOI.OPTIMAL
                error(" Status: $(status)")
            end
        end)
        obj_val = objective_value(m_oracle)
        viol = μ[n] + θ - obj_val + λ*norm(zstar_global - z[n],1)
        yproj = clamp.(y[n], a, b)  
        Δpos = b .- yproj
        Δneg = yproj .- a
        ηpos_val = [value.(m_oracle[:ηpos])[b, t] for b ∈ bus_renewable_generation, t ∈ T]
        ηpos = vec(ηpos_val')
        ηneg_val = [value.(m_oracle[:ηneg])[b, t] for b ∈ bus_renewable_generation, t ∈ T]
        ηneg = vec(ηneg_val')
        y_opt = yproj .+ ηpos .* Δpos .- ηneg .* Δneg
        oracle_dict["y"]["$n"] = @views [y_opt[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]

        obj_funcion = objective_function(m_oracle)

       const_term_x1, const_term_x2, const_term_x3, const_term_x4, const_term_x5, const_term_x6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for t ∈ T

            for b in bus_renewable_generation

                oracle_dict["π"]["$n"]["($b, $t)"] = value(m_oracle[Symbol("π[$b,$t]")])
                oracle_dict["π_s"]["$n"]["($b, $t)"] = value(m_oracle[Symbol("π_s[$b,$t]")])

            end

            for i in Ig

                oracle_dict["x1"]["$n"]["($i, $t)"] = value(m_oracle[Symbol("x1[$i,$t]")])
                oracle_dict["x2"]["$n"]["($i, $t)"] = value(m_oracle[Symbol("x2[$i,$t]")])
                const_term_x1 += Pmax[i]*u[i, t]  * value(m_oracle[Symbol("x1[$i,$t]")])
                const_term_x2 += Pmin[i]*u[i, t]  * value(m_oracle[Symbol("x2[$i,$t]")])

                if t == 1

                    oracle_dict["x5"]["$n"]["($i, $t)"] = value(m_oracle[Symbol("x5[$i,$t]")])
                    oracle_dict["x6"]["$n"]["($i, $t)"] = value(m_oracle[Symbol("x6[$i,$t]")])
                    const_term_x5 += (RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i]) * value(m_oracle[Symbol("x5[$i,$t]")])
                    const_term_x6 += (RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i]) * value(m_oracle[Symbol("x6[$i,$t]")])

                else

                    oracle_dict["x3"]["$n"]["($i, $t)"] = value(m_oracle[Symbol("x3[$i,$t]")])
                    oracle_dict["x4"]["$n"]["($i, $t)"] = value(m_oracle[Symbol("x4[$i,$t]")])
                    const_term_x3 += (RU[i]*u[i, t-1] + Pmax[i]*v[i, t])* value(m_oracle[Symbol("x3[$i,$t]")])
                    const_term_x4 += (RD[i]*u[i, t] + Pmax[i]*w[i, t])* value(m_oracle[Symbol("x4[$i,$t]")])

                end

            end

        end

        oracle_dict["const"]["$n"] = copy(objective_value(m_oracle) - (const_term_x1 + const_term_x2 + const_term_x3 + const_term_x4 + const_term_x5 + const_term_x6) - λ*norm(y_opt - y[n],1))
        oracle_dict["time"]["$n"] = time_

        oracle_dict["ϑ"][n] = objective_value(m_oracle) - λ*norm(zstar_global - z[n],1)

        if viol ≥ -1e-2
            violation[n] = false
        else
            violation[n] = true
        end

    end

    if sum(violation) < N 
        print("feasibility for $(N - sum(violation)) subproblems\n")
    end

    convergence = ~Base.any(violation)

    return convergence
    
end

function create_master_problem(Data, ρ)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    N = Data.N
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd

    # Ignitial state
    u0 = zeros(number_of_generators)


    m_master = Model(
    optimizer_with_attributes(
            () -> Gurobi.Optimizer(GUROBI_ENV),
            "NumericFocus" => 3,
            "ScaleFlag" => 2,
            "Method" => 1,
            "OutputFlag" => 0
        )
    )
    @variable(m_master, 0 <= v[1:Ig[end], 1:T[end]] <= 1) # start-up
    @variable(m_master, 0 <= w[1:Ig[end], 1:T[end]] <= 1) # shutdown
    @variable(m_master, u[1:Ig[end], 1:T[end]], Bin) # commit
    @variables(m_master, begin
        μ[1:N] ≥ 0
        θ
        λ ≥ 0
    end)

    @constraint(m_master, [i in Ig, t in T[2:end]], v[i, t] - w[i, t] == u[i, t] - u[i, t-1])
    @constraint(m_master, [i in Ig, t in T], v[i, t] <= u[i, t])
    @constraint(m_master, [i in Ig, t in T], w[i, t] <= 1 - u[i, t])

    # Ignitial state constraint
    @constraint(m_master, [i in Ig, t in [1]], v[i, t] - w[i, t] == u[i, t] - u0[i])

    @objective(m_master, Min, sum(Cu[i]*m_master[:u][i, t] for i in Ig, t in T) + sum(Csu[i]*m_master[:v][i, t] for i in Ig, t in T) +  sum(Csd[i]*m_master[:w][i, t] for i in Ig, t in T) + m_master[:λ]*ρ + m_master[:θ] + (1/(N*α))*sum(m_master[:μ]))

    return m_master
end

function update_master_CCG(Data, m_master, oracle_dict, zstar, y, PredictionStar, cuts_CCG)

    N = Data.N
    Tf = Data.Tf
    number_of_generators = Data.Ig
    Tl = Data.Tl
    bus_renewable_generation = Data.bus_renewable_generation
    Cp = Data.Cp
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Fmax = Data.Fmax
    Cdef = Data.Cdef
    D = Data.D
    F = Data.F
    B = Data.B
    S = Data.S
    Tb, = size(B) # Total buses
    Bus = setdiff(collect(1:Tb), bus_renewable_generation) # Buses
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    L = collect(1:Tl) # transmission lines

    # Ignitial state
    p0 = zeros(number_of_generators)
    f0 = zeros(Tl)
    u0 = zeros(number_of_generators)

    for n ∈ 1:N

        norm_l1_z = norm(vcat(zstar...) - z[n], 1)

        y_opt = oracle_dict["y"]["$n"]

        signature = Tuple(round.(vcat(y_opt...); digits = 8))

        if !(signature in cuts_CCG["$n"])

            push!(cuts_CCG["$n"], signature)

            norm_l1_y = norm(vcat(y_opt...) - y[n], 1)

            β = @variable(m_master, [1:Tb, 1:T[end]]) # phase
            f = @variable(m_master, [1:L[end], 1:T[end]]) # power flow
            p = @variable(m_master, [1:Ig[end], 1:T[end]], lower_bound = 0) # generation
            pdef = @variable(m_master, [1:Tb, 1:T[end]], lower_bound = 0) # deficit generation
            s = @variable(m_master, [bus_renewable_generation, 1:T[end]], lower_bound = 0) # spillage

            @constraint(m_master, [t in T], f[:, t] .== S*β[:, t])
            @constraint(m_master, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] >= D[b][t]) # Power balance
            @constraint(m_master, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
            @constraint(m_master, [b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] >= D[b][t] - maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0])) # Uncertainty
            @constraint(m_master, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
            @constraint(m_master, [b in bus_renewable_generation, t in T], s[b, t] <=  maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0])) # Uncertainty
            @constraint(m_master, [l in L, t in T], f[l,t] <= Fmax[l])
            @constraint(m_master, [l in L, t in T], f[l,t] >= -Fmax[l])
            @constraint(m_master, [i in Ig, t in T], p[i,t] <= Pmax[i]*m_master[:u][i, t])
            @constraint(m_master, [i in Ig, t in T], p[i,t] >= Pmin[i]*m_master[:u][i, t])

            @constraint(m_master, [i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*m_master[:u][i, t-1] + Pmax[i]*m_master[:v][i, t])
            @constraint(m_master, [i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*m_master[:u][i, t] + Pmax[i]*m_master[:w][i, t])

            # Ignitial state constraints
            @constraint(m_master, [i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*m_master[:v][i, t] + p0[i])
            @constraint(m_master, [i in Ig, t in [1]], -p[i, t] <= RD[i]*m_master[:u][i, t] + Pmax[i]*m_master[:w][i, t] - p0[i])

            @constraint(m_master, sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) + sum(0.1*Cdef*s[b,t] for b in bus_renewable_generation, t in T) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

        end

    end

    return nothing
    
end

function update_master_Benders(Data, m_master, z, zstar, y, oracle_dict, cuts_Benders, cuts)

    N = Data.N
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig
    D = Data.D

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    for n ∈ 1:N

        ex = @expression(m_master, AffExpr())

        y_opt = oracle_dict["y"]["$n"]
        π = oracle_dict["π"]["$n"]
        π_s = oracle_dict["π_s"]["$n"]
        x1 = oracle_dict["x1"]["$n"]
        x2 = oracle_dict["x2"]["$n"]
        x3 = oracle_dict["x3"]["$n"]
        x4 = oracle_dict["x4"]["$n"]
        x5 = oracle_dict["x5"]["$n"]
        x6 = oracle_dict["x6"]["$n"]
        const_term = oracle_dict["const"]["$n"]

        δ = norm(vcat(y_opt...) - y[n], 1) 

        γ = const_term

        signature = (
            Tuple(round.(values(x1); digits = 8)),
            Tuple(round.(values(x2); digits = 8)),
            Tuple(round.(values(x3); digits = 8)),
            Tuple(round.(values(x4); digits = 8)),
            Tuple(round.(values(x5); digits = 8)),
            Tuple(round.(values(x6); digits = 8)),
            round(δ; digits = 8),
            round(γ; digits = 8),
        )

        if !(signature in cuts_Benders["$n"]["signatures"])

            push!(cuts_Benders["$n"]["signatures"], signature)

            δ += norm(vcat(zstar...) - z[n], 1)

            for t in T

                for i in Ig

                    add_to_expression!(ex, Pmax[i]*x1["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmin[i]*x2["($i, $t)"], m_master[:u][i, t])

                    if t == 1

                        add_to_expression!(ex, RU[i]*u0[i]*x5["($i, $t)"] + p0[i]*x5["($i, $t)"])
                        add_to_expression!(ex, Pmax[i]*x5["($i, $t)"], m_master[:v][i, t])

                        add_to_expression!(ex, RD[i]*x6["($i, $t)"], m_master[:u][i, t])
                        add_to_expression!(ex, Pmax[i]*x6["($i, $t)"], m_master[:w][i, t])
                        add_to_expression!(ex, -p0[i]*x6["($i, $t)"])

                    else
                        add_to_expression!(ex, RU[i]*x3["($i, $t)"], m_master[:u][i, t-1])
                        add_to_expression!(ex, Pmax[i]*x3["($i, $t)"], m_master[:v][i, t])

                        add_to_expression!(ex, RD[i]*x4["($i, $t)"], m_master[:u][i, t])
                        add_to_expression!(ex, Pmax[i]*x4["($i, $t)"], m_master[:w][i, t])
                    end
                end
            end

            push!(cuts_Benders["$n"]["π"], round.(values(π); digits = 8))
            push!(cuts_Benders["$n"]["π_s"], round.(values(π_s); digits = 8))
            push!(cuts_Benders["$n"]["y_opt"], vcat(y_opt...))
            push!(cuts_Benders["$n"]["const"], γ - sum(π["($b, $t)"]*(D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - y_opt[ind_renewable_generation[b]][t]) + π_s["($b, $t)"]*(PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t]) for b ∈ bus_renewable_generation, t ∈ T))

            cut = @constraint(m_master, ex - δ*m_master[:λ] + γ ≤ m_master[:μ][n] + m_master[:θ])
            push!(cuts["$n"], cut)

        end

    end
    
end

function initilization_restricted_master_CCG(Data, m_master, z, zstar, y, errors, Ξ_y, PredictionStar, α)

    N = Data.N
    Tf = Data.Tf
    number_of_generators = Data.Ig
    Tl = Data.Tl
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Cp = Data.Cp
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Fmax = Data.Fmax
    Cdef = Data.Cdef
    D = Data.D
    F = Data.F
    B = Data.B
    S = Data.S
    Tb, = size(B) # Total buses
    Bus = setdiff(collect(1:Tb), bus_renewable_generation) # Buses
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    L = collect(1:Tl) # transmission lines

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    zstar_global = vcat(zstar...)

    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    m = ceil(Int, N*α)

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar_global, 1)

        dy = 0.0
        for d in eachindex(a)
            if y[n][d] < a[d]
                dy += a[d] - y[n][d]
            elseif y[n][d] > b[d]
                dy += y[n][d] - b[d]
            end
        end

        dist[n] = dz + dy

    end

    idx = sortperm(dist)

    for k in 1:m

        n = idx[k]

        yproj = clamp.(y[n], a, b)

        norm_l1_z = norm(zstar_global - z[n], 1)
        norm_l1_y = norm(yproj - y[n], 1)

        β = @variable(m_master, [1:Tb, 1:T[end]]) # phase
        f = @variable(m_master, [1:L[end], 1:T[end]]) # power flow
        p = @variable(m_master, [1:Ig[end], 1:T[end]], lower_bound = 0) # generation
        pdef = @variable(m_master, [1:Tb, 1:T[end]], lower_bound = 0) # deficit generation
        s = @variable(m_master, [bus_renewable_generation, 1:T[end]], lower_bound = 0) # spillage

        @constraint(m_master, [t in T], f[:, t] .== S*β[:, t])
        @constraint(m_master, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] >= D[b][t]) # Power balance
        @constraint(m_master, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
        @constraint(m_master, [b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] >= D[b][t] - PredictionStar[ind_renewable_generation[b]][t] - errors[ind_renewable_generation[b]][n][t]) # Uncertainty
        @constraint(m_master, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
        @constraint(m_master, [b in bus_renewable_generation, t in T], s[b, t] <=  PredictionStar[ind_renewable_generation[b]][t] + errors[ind_renewable_generation[b]][n][t]) # Uncertainty
        @constraint(m_master, [l in L, t in T], f[l,t] <= Fmax[l])
        @constraint(m_master, [l in L, t in T], f[l,t] >= -Fmax[l])
        @constraint(m_master, [i in Ig, t in T], p[i,t] <= Pmax[i]*m_master[:u][i, t])
        @constraint(m_master, [i in Ig, t in T], p[i,t] >= Pmin[i]*m_master[:u][i, t])

        @constraint(m_master, [i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*m_master[:u][i, t-1] + Pmax[i]*m_master[:v][i, t])
        @constraint(m_master, [i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*m_master[:u][i, t] + Pmax[i]*m_master[:w][i, t])

        # Ignitial state constraints
        @constraint(m_master, [i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*m_master[:v][i, t] + p0[i])
        @constraint(m_master, [i in Ig, t in [1]], -p[i, t] <= RD[i]*m_master[:u][i, t] + Pmax[i]*m_master[:w][i, t] - p0[i])
        @constraint(m_master, sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

    end

    return nothing
 
end

function deterministic_equivalent_problem(Data, y_list, PredictionStar)
    
    D = Data.D
    F = Data.F
    B = Data.B
    S = Data.S
    Tf = Data.Tf
    Tl = Data.Tl
    Cp = Data.Cp
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Fmax = Data.Fmax
    Cdef = Data.Cdef
    number_of_generators = Data.Ig
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators
    Tb, = size(B) # Total buses
    L = collect(1:Tl) # transmission lines
    Bus = setdiff(collect(1:Tb), bus_renewable_generation) # Buses
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd


    # Ignitial state
    p0 = zeros(number_of_generators)
    f0 = zeros(Tl)
    u0 = zeros(number_of_generators)

    m = length(y_list)

    model =  Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variable(model, cost_recourse[1:m])

    @variable(model, 0 <= v[1:Ig[end], 1:T[end]] <= 1) # start-up
    @variable(model, 0 <= w[1:Ig[end], 1:T[end]] <= 1) # shutdown
    @variable(model, u[1:Ig[end], 1:T[end]], Bin) # commit

    @constraint(model, [i in Ig, t in T[2:end]], v[i, t] - w[i, t] == u[i, t] - u[i, t-1])
    @constraint(model, [i in Ig, t in T], v[i, t] <= u[i, t])
    @constraint(model, [i in Ig, t in T], w[i, t] <= 1 - u[i, t])

    # Ignitial state constraint
    @constraint(model, [i in Ig, t in [1]], v[i, t] - w[i, t] == u[i, t] - u0[i])

    for k in 1:m

        y_opt = y_list[k]

        β = @variable(model, [1:Tb, 1:T[end]]) # phase
        f = @variable(model, [1:L[end], 1:T[end]]) # power flow
        p = @variable(model, [1:Ig[end], 1:T[end]], lower_bound = 0) # generation
        pdef = @variable(model, [1:Tb, 1:T[end]], lower_bound = 0) # deficit generation
        s = @variable(model, [bus_renewable_generation, 1:T[end]], lower_bound = 0) # spillage

        @constraint(model, [t in T], f[:, t] .== S*β[:, t])
        @constraint(model, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] == D[b][t]) # Power balance
        @constraint(model, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
        @constraint(model, [b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] == D[b][t] - maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0])) # Uncertainty
        @constraint(model, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
        @constraint(model, [b in bus_renewable_generation, t in T], s[b, t] <=  maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0])) # Uncertainty
        @constraint(model, [l in L, t in T], f[l,t] <= Fmax[l])
        @constraint(model, [l in L, t in T], f[l,t] >= -Fmax[l])
        @constraint(model, [i in Ig, t in T], p[i,t] <= Pmax[i]*u[i, t])
        @constraint(model, [i in Ig, t in T], p[i,t] >= Pmin[i]*u[i, t])

        @constraint(model, [i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*u[i, t-1] + Pmax[i]*v[i, t])
        @constraint(model, [i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t])

        # Ignitial state constraints
        @constraint(model, [i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i])
        @constraint(model, [i in Ig, t in [1]], -p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i])

        @constraint(model, cost_recourse[k] == sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T) + sum(0.1*Cdef*s[b,t] for b in bus_renewable_generation, t in T))

    end

    @objective(model, Min, sum(Cu[i]*model[:u][i, t] for i in Ig, t in T) + sum(Csu[i]*model[:v][i, t] for i in Ig, t in T) +  sum(Csd[i]*model[:w][i, t] for i in Ig, t in T) + (1/m)*sum(cost_recourse[k] for k in 1:m))
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        error(" Status: $(status)")
    end

    u = value.(model[:u])
    v = value.(model[:v])
    w = value.(model[:w])

    return u, v, w
   
end

function dual_vertices(Data, u, v, w, y_opt, PredictionStar) 

    D = Data.D
    Tf = Data.Tf
    T = collect(1:Tf)
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig


    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)


    number_of_generators = Data.Ig
    Ig = collect(1:number_of_generators)
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    model = economic_dispatch_problem(Data, u, v, w)

    for b ∈ bus_renewable_generation, t ∈ T

        set_normalized_rhs(model[:π][b, t], D[b][t] - maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0]))
        set_normalized_rhs(model[:π_s][b, t], maximum([PredictionStar[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0]))

    end
    m_oracle = dualize(model; dual_names = DualNames())
    set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Method" => 2, "Crossover" => 0))
    optimize!(m_oracle)
    status = termination_status(m_oracle)
    if status != MOI.OPTIMAL
        error(" Status: $(status)")
    end

    π = Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    π_s = Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    x1 = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x2 = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x3 = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x4 = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x5 = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    x6 = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    const_term = 0.0

    obj_funcion = objective_function(m_oracle)

    const_term_x1, const_term_x2, const_term_x3, const_term_x4, const_term_x5, const_term_x6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for t ∈ T

        for b in bus_renewable_generation

            π["($b, $t)"] = value(m_oracle[Symbol("π[$b,$t]")])
            π_s["($b, $t)"] = value(m_oracle[Symbol("π_s[$b,$t]")])

        end

        for i in Ig

            x1["($i, $t)"] = value(m_oracle[Symbol("x1[$i,$t]")])
            x2["($i, $t)"] = value(m_oracle[Symbol("x2[$i,$t]")])
            const_term_x1 += Pmax[i]*u[i, t] * value(m_oracle[Symbol("x1[$i,$t]")])
            const_term_x2 += Pmin[i]*u[i, t] * value(m_oracle[Symbol("x2[$i,$t]")])

            if t == 1

                x5["($i, $t)"] = value(m_oracle[Symbol("x5[$i,$t]")])
                x6["($i, $t)"] = value(m_oracle[Symbol("x6[$i,$t]")])
                const_term_x5 += (RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i]) * value(m_oracle[Symbol("x5[$i,$t]")])
                const_term_x6 += (RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i]) * value(m_oracle[Symbol("x6[$i,$t]")])

            else

                x3["($i, $t)"] = value(m_oracle[Symbol("x3[$i,$t]")])
                x4["($i, $t)"] = value(m_oracle[Symbol("x4[$i,$t]")])
                const_term_x3 += (RU[i]*u[i, t-1] + Pmax[i]*v[i, t])* value(m_oracle[Symbol("x3[$i,$t]")])
                const_term_x4 += (RD[i]*u[i, t] + Pmax[i]*w[i, t]) * value(m_oracle[Symbol("x4[$i,$t]")])


            end

        end

    end


    const_term = objective_value(m_oracle) - (const_term_x1 + const_term_x2 + const_term_x3 + const_term_x4 + const_term_x5 + const_term_x6)

    return π, π_s, x1, x2, x3, x4, x5, x6, const_term, objective_value(m_oracle)
end

function initilization_restricted_master_Benders(Data, m_master, z, zstar, y, Ξ_y, PredictionStar, α)

    N = Data.N
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    Tf = Data.Tf   
    number_of_generators = Data.Ig 

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    y_list = []
    idx = collect(1:N)

    for  n ∈ 1:N

        yproj = clamp.(y[n], a, b)
        push!(y_list, @views [yproj[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]) 

    end   

    y_mean = mean(collect(y_list))

    u , v , w = deterministic_equivalent_problem(Data, [y_mean], PredictionStar)

    initial_cuts = Dict{String, Any}()

    for n ∈ 1:N


        yproj = y_list[n]

        π, π_s, x1, x2, x3, x4, x5, x6, const_term = dual_vertices(Data, u, v, w, yproj, PredictionStar)

        ex = @expression(m_master, AffExpr())


        δ = norm(vcat(yproj...) - y[n], 1) +
            norm(vcat(zstar...) - z[n], 1)

        γ = const_term

        for t in T

            for i in Ig

                add_to_expression!(ex, Pmax[i]*x1["($i, $t)"], m_master[:u][i, t])
                add_to_expression!(ex, Pmin[i]*x2["($i, $t)"], m_master[:u][i, t])

                if t == 1

                    add_to_expression!(ex, RU[i]*u0[i]*x5["($i, $t)"] + p0[i]*x5["($i, $t)"])
                    add_to_expression!(ex, Pmax[i]*x5["($i, $t)"], m_master[:v][i, t])

                    add_to_expression!(ex, RD[i]*x6["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmax[i]*x6["($i, $t)"], m_master[:w][i, t])
                    add_to_expression!(ex, -p0[i]*x6["($i, $t)"])

                else
                    add_to_expression!(ex, RU[i]*x3["($i, $t)"], m_master[:u][i, t-1])
                    add_to_expression!(ex, Pmax[i]*x3["($i, $t)"], m_master[:v][i, t])

                    add_to_expression!(ex, RD[i]*x4["($i, $t)"], m_master[:u][i, t])
                    add_to_expression!(ex, Pmax[i]*x4["($i, $t)"], m_master[:w][i, t])
                end
            end
        end

       con = @constraint(m_master, ex - δ*m_master[:λ] + γ ≤ m_master[:μ][n] + m_master[:θ])

       initial_cuts["$n"] = con

    end

    return initial_cuts, idx
 
end


function warm_start_solution_CCG(data, instalations, demand, G, D, ξ_dict, Ξ_y, cuts_CCG, old_cuts, zstar, PredictionStar, ρ)
    
    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    y_scen = ξ_dict["y"]
    z = ξ_dict["z"]

    m_master = create_master_problem(data, G, ρ_new)

    initilization_restricted_master_CCG(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y, zstar, PredictionStar, α)
    
    N = data["sample_size"]

    for n ∈ collect(1:N)

        for signature ∈ cuts_CCG["$n"]
            y_opt = collect(signature)
            norm_l1_z = norm(zstar_new - z[n], 1)
            norm_l1_y = norm(y_opt - y_scen[n], 1)

            y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
            u = @variable(m_master, [1:D], lower_bound = 0)
            v = @variable(m_master, [1:G], lower_bound = 0)
            @constraints(m_master, begin
                        [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
                        [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar[d] + y_opt[d]
            end)
            @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])
        end
        # y_opt = old_cuts["$n"]["best_vertice"]
        # norm_l1_z = norm(zstar_new - z[n], 1)
        # norm_l1_y = norm(y_opt - y_scen[n], 1)

        # y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
        # u = @variable(m_master, [1:D], lower_bound = 0)
        # v = @variable(m_master, [1:G], lower_bound = 0)
        # @constraints(m_master, begin
        #             [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
        #             [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar_new[d] + y_opt[d]
        # end)
        # @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])


    end

    optimize!(m_master)
    status = termination_status(m_master)
    if status != MOI.OPTIMAL
        error("problem $status")
    end

    x_warm = value.(m_master[:x])
    λ_warm= value.(m_master[:λ])

    μ_warm = [-Inf for n ∈ 1:N]
    θ_warm = -Inf

    return x_warm, λ_warm, μ_warm, θ_warm, objective_value(m_master)
 
end

function warm_start_solution_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new, ρ_new, α, cuts_Benders)

    m_master_aux = create_master_problem(Data, ρ_new)

    initilization_restricted_master_Benders(Data, m_master_aux, z, zstar_new, y, Ξ_y_new, PredictionStar_new, α)

    N = Data.N
    RU = Data.RU
    RD = Data.RD
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    number_of_generators = Data.Ig
    Tf = Data.Tf
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation

    # Ignitial state
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)

    sample = collect(1:N)

    π_dict = Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample)
    π_s_dict = Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample)
    x1_dict = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x2_dict = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x3_dict = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x4_dict = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x5_dict = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    x6_dict = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])

    cuts_warm = Dict("$n" => [] for n ∈ 1:N)

    for n ∈ 1:N

        i = 1

        for signature ∈ cuts_Benders["$n"]["signatures"]

            ex = @expression(m_master_aux, AffExpr())

            x1_val = signature[1]
            x2_val = signature[2]
            x3_val = signature[3]
            x4_val = signature[4]
            x5_val = signature[5]
            x6_val = signature[6]
            π_val = cuts_Benders["$n"]["π"][i]
            π_s_val = cuts_Benders["$n"]["π_s"][i]
            x1 = reconstruct_sub_dict(x1_dict, x1_val)
            x2 = reconstruct_sub_dict(x2_dict, x2_val)
            x3 = reconstruct_sub_dict(x3_dict, x3_val)
            x4 = reconstruct_sub_dict(x4_dict, x4_val)
            x5 = reconstruct_sub_dict(x5_dict, x5_val)
            x6 = reconstruct_sub_dict(x6_dict, x6_val)
            π = reconstruct_sub_dict(π_dict["$n"], π_val)
            π_s = reconstruct_sub_dict(π_s_dict["$n"], π_s_val)

            δ = signature[7]
            δ += norm(vcat(zstar...) - z[n], 1)
            γ = cuts_Benders["$n"]["const"][i] # corrected to include the constant term from the cuts_Benders dictionary
            yopt = cuts_Benders["$n"]["y_opt"][i]
            y_opt = @views [yopt[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]

            γ += sum(π["($b, $t)"]*maximum([PredictionStar_new[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0]) + π_s["($b, $t)"]*maximum([PredictionStar_new[ind_renewable_generation[b]][t] + y_opt[ind_renewable_generation[b]][t], 0.0]) for b ∈ bus_renewable_generation, t ∈ T)
            γ = signature[8]

            for t in T

                for i in Ig

                    add_to_expression!(ex, Pmax[i]*x1["($i, $t)"], m_master_aux[:u][i, t])
                    add_to_expression!(ex, Pmin[i]*x2["($i, $t)"], m_master_aux[:u][i, t])

                    if t == 1

                        add_to_expression!(ex, RU[i]*u0[i]*x5["($i, $t)"] + p0[i]*x5["($i, $t)"])
                        add_to_expression!(ex, Pmax[i]*x5["($i, $t)"], m_master_aux[:v][i, t])

                        add_to_expression!(ex, RD[i]*x6["($i, $t)"], m_master_aux[:u][i, t])
                        add_to_expression!(ex, Pmax[i]*x6["($i, $t)"], m_master_aux[:w][i, t])
                        add_to_expression!(ex, -p0[i]*x6["($i, $t)"])

                    else
                        add_to_expression!(ex, RU[i]*x3["($i, $t)"], m_master_aux[:u][i, t-1])
                        add_to_expression!(ex, Pmax[i]*x3["($i, $t)"], m_master_aux[:v][i, t])

                        add_to_expression!(ex, RD[i]*x4["($i, $t)"], m_master_aux[:u][i, t])
                        add_to_expression!(ex, Pmax[i]*x4["($i, $t)"], m_master_aux[:w][i, t])
                    end
                end
            end

            con = @constraint(m_master_aux, ex - δ*m_master_aux[:λ] + γ ≤ m_master_aux[:μ][n] + m_master_aux[:θ])

            push!(cuts_warm["$n"], con)

            i += 1

        end

    end

    val, time_ = @timed(begin
        optimize!(m_master_aux)
        status = termination_status(m_master_aux)
        if status != MOI.OPTIMAL
            error("problem $status")
        end
    end)

    u_warm = value.(m_master_aux[:u])
    v_warm = value.(m_master_aux[:v])
    w_warm = value.(m_master_aux[:w])
    λ_warm = value.(m_master_aux[:λ])

    initial_cuts_warm, idx_warm = initilization_restricted_master_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new, α)
    optimize!(m_master_warm)
    status = termination_status(m_master_warm)
    if status != MOI.OPTIMAL
        error("problem $status")
    end
    μ_warm = value.(m_master_warm[:μ])
    θ_warm = value(m_master_warm[:θ])

    return u_warm, v_warm, w_warm, λ_warm, μ_warm, θ_warm, objective_value(m_master_aux), time_, initial_cuts_warm, idx_warm
 
end

function solve_master_problem(m_master)

    val, time_master = @timed(begin
        optimize!(m_master)
        status = termination_status(m_master)
        if status != MOI.OPTIMAL
            error("problem $status")
        end
    end) 

    # Solution
    u = value.(m_master[:u])
    v = value.(m_master[:v])
    w = value.(m_master[:w])
    λ = value.(m_master[:λ])

    μ = value.(m_master[:μ])
    θ = value(m_master[:θ])

    return u, v, w, λ, μ, θ, objective_value(m_master), time_master
    
end

function critical_radius(Data, errors, Ξ_y, α, zstar; p = 1)

    N = Data.N
    z_aux = Data.z
    Tf = Data.Tf
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb = length(bus_renewable_generation)

    global_residual_scenarios = zeros(Tf * Tb, N)
    global_contextual_scenarios =  zeros(Tf * Tb, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z_aux[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    zstar_global = vcat(zstar...)
    y = [s[:] for s in eachcol(global_residual_scenarios)]
    z = [s[:] for s in eachcol(global_contextual_scenarios)]

    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar_global, 1)

        dy = 0.0
        for d in eachindex(a)
            if y[n][d] < a[d]
                dy += a[d] - y[n][d]
            elseif y[n][d] > b[d]
                dy += y[n][d] - b[d]
            end
        end

        dist[n] = dz + dy
    end

    sort!(dist)

    if α == 0
        return dist[1]
    end

    m = floor(Int, N*α)
    frac = N*α - m

    value = sum(dist[k] for k in 1:m)

    if frac > 0
        value += frac*dist[m+1]
    end

    return value/(N*α)

end

function CVaR(α, vec)

    S = collect(1:length(vec))
    p = (1/length(vec))*ones(length(vec))

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variable(model, θ)
    @variable(model, z[s in S] >= 0)

    @constraint(model, [s in S],
    z[s] >= vec[s] - θ)

    @objective(model, Min, θ + (1/α) * sum(p[s] * z[s] for s in S))

    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        error("problem $status")
    end

    CVaR_value = objective_value(model)

    return CVaR_value

end

function reconstruct_sub_dict(dict, values_tuple)

    sub_dict = Dict{String, Float64}()
    for (idx, key) in enumerate(collect(keys(dict)))
        sub_dict[key] = values_tuple[idx]
    end

    return sub_dict

end

function Upper_Bound(Data, u, v, w, λ, vec, α)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators

    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd

    UB = sum(Cu[i]*u[i, t] for i in Ig, t in T) + sum(Csu[i]*v[i, t] for i in Ig, t in T) +  sum(Csd[i]*w[i, t] for i in Ig, t in T) + ρ*λ + CVaR(α, vec)

    return UB

end

function CCG_algorithm(Data, errors, Ξ_y, zstar, PredictionStar, ρ, α)
    
    N = Data.N
    z_aux = Data.z
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb_renewable = length(bus_renewable_generation)
    number_of_generators = Data.Ig
    Tf = Data.Tf
    Ig = collect(1:number_of_generators)
    T = collect(1:Tf)
    sample = collect(1:N)

    global_residual_scenarios = zeros(Tf * Tb_renewable, N)
    global_contextual_scenarios =  zeros(Tf * Tb_renewable, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z_aux[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    y = [s[:] for s in eachcol(global_residual_scenarios)]
    z = [s[:] for s in eachcol(global_contextual_scenarios)]

    y_val = Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample)
    π = Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample)
    π_s = Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample)
    x1 = Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample)
    x2 = Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample)
    x3 = Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample)
    x4 = Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample)
    x5 = Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample)
    x6 = Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample)
    const_term = Dict("$n" => 0.0 for n ∈ sample)

    oracle_dict = Dict("y" => y_val, "π" => π, "π_s" => π_s, "x1" => x1, "x2" => x2, "x3" => x3, "x4" => x4, "x5" => x5, "x6" => x6, "const" => const_term)

    cuts_CCG = Dict("$n" => Set{Tuple{Vararg{Float64}}}() for n ∈ sample)
    iter = 1
    m_master = create_master_problem(Data, ρ)
    initilization_restricted_master_CCG(Data, m_master, z, zstar, y, errors, Ξ_y, PredictionStar, α)
    m_oracle, obj  = create_oracle_problem(Data)
    convergence = false
    while ~convergence && iter ≤ 20
        global u_CCG, v_CCG, w_CCG, λ_CCG, μ_CCG, θ_CCG, LB_CCG = solve_master_problem(m_master)
        convergence = solve_oracle_problem(Data, m_oracle, obj, z, y, Ξ_y, u_CCG, v_CCG, w_CCG, λ_CCG, μ_CCG, θ_CCG, oracle_dict, zstar, PredictionStar)
        update_master_CCG(Data, m_master, oracle_dict, zstar, y, PredictionStar, cuts_CCG)
        println("================================================")
        println("                                                         CCG                                                        ")
        println("                                                     Iter= $iter                                                   ")
        println("================================================")
        iter +=  1
    end

    return u_CCG, v_CCG, w_CCG, λ_CCG, μ_CCG, θ_CCG, LB_CCG, iter - 1, cuts_CCG

end

function Benders_algorithm(Data, Ξ_y, zstar, PredictionStar, ρ, α)

    zstar = Data.zstar

    Models, errors = scenarios(Data)

    Ξ_y = uncertainty_support(Data, zstar, errors)

    PredictionStar = forecast(Data, zstar, Models)


    # Trimming parameters

    α = 0.2
    ρ = 1.1*critical_radius(Data, errors, Ξ_y, α, zstar) 


    N = Data.N
    z_aux = Data.z
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb_renewable = length(bus_renewable_generation)
    number_of_generators = Data.Ig
    Tf = Data.Tf
    Ig = collect(1:number_of_generators)
    T = collect(1:Tf)
    sample = collect(1:N)

    global_residual_scenarios = zeros(Tf * Tb_renewable, N)
    global_contextual_scenarios =  zeros(Tf * Tb_renewable, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z_aux[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    y = [s[:] for s in eachcol(global_residual_scenarios)]
    z = [s[:] for s in eachcol(global_contextual_scenarios)]


    oracle_dict = Dict("y" => Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample), "π" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "π_s" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "x1" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x2" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x3" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x4" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x5" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "x6" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "const" => Dict("$n" => 0.0 for n ∈ sample), "time" => Dict("$n" => 0.0 for n ∈ sample), "ϑ" => zeros(N))
    cuts_Benders = Dict("$n" => Dict("signatures" => Set{Tuple{Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}},Float64,Float64}}(), "π" => Array{Array{Float64},1}(), "π_s" => Array{Array{Float64},1}(), "y_opt" => Array{Array{Float64},1}(), "const" => Array{Float64,1}()) for n in sample)
    iter = 1

    LowerBound = []
    UpperBound = []
    Time = []
    cuts = Dict("$n" => [] for n ∈ sample)

    m_master = create_master_problem(Data, ρ)
    initial_cuts, idx = initilization_restricted_master_Benders(Data, m_master, z, zstar, y, Ξ_y, PredictionStar, α)
    # initial_cuts, idx = [], []
    # @constraint(m_master, m_master[:θ] ≥ - 1e10)
    m_oracle, obj  = create_oracle_problem(Data)
    convergence = false
    Max_iter = 250
    cumulative_time = 0.0
    UB_Benders = Inf

    gap = Inf

    while ~convergence && iter ≤ Max_iter  #&& gap ≥ 5e-2

        global u_Benders, v_Benders, w_Benders, λ_Benders, μ_Benders, θ_Benders, LB_Benders, time_master_Benders = solve_master_problem(m_master)
        convergence = solve_oracle_problem(Data, m_oracle, obj, z, y, Ξ_y, u_Benders, v_Benders, w_Benders, λ_Benders, μ_Benders, θ_Benders, oracle_dict, zstar, PredictionStar, initial_cuts, idx)
        UB_Benders = minimum([UB_Benders, Upper_Bound(Data, u_Benders, v_Benders, w_Benders, λ_Benders, oracle_dict["ϑ"], α)])
        update_master_Benders(Data, m_master, z, zstar, y, oracle_dict, cuts_Benders, cuts)
        total_time_oracle = sum(oracle_dict["time"]["$n"][end] for n ∈ sample)
        cumulative_time += time_master_Benders + total_time_oracle
        total_time = time_master_Benders + total_time_oracle
        println("================================================")
        println(" Lower Bound: $LB_Benders\n") 
        println(" Upper Bound: $UB_Benders\n") 
        println(" λ: $λ_Benders\n") 
        println(" θ: $θ_Benders\n") 
        println(" μ: $(μ_Benders')\n") 
        println(" Iteration: $iter")
        println("================================================") 
        iter +=  1
        push!(LowerBound, LB_Benders)
        push!(UpperBound, UB_Benders)
        push!(Time,(time_master_Benders, total_time_oracle,total_time))

        gap = (UB_Benders - LB_Benders)/UB_Benders

    end

    return u_Benders, v_Benders, w_Benders, λ_Benders, μ_Benders, θ_Benders, LB_Benders, iter - 1, cuts_Benders
    
end

function CCG_with_warm_start_algorithm(data, ξ_dict, Ξ_y, zstar, PredictionStar, ρ, α, cuts_CCG, old_cuts)

    sample = collect(1:data["sample_size"])

    y_dict_CCG = Dict("$n" => zeros(D) for n ∈ sample)
    π_dict_CCG = Dict("$n" => zeros(D) for n ∈ sample) 
    σ_dict_CCG = Dict("$n" => zeros(G) for n ∈ sample)
    old_cuts_CCG = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
    cuts_CCG = Dict("$n" => Set{Tuple{Vararg{Float64}}}() for n ∈ sample)
    iter = 1
    obj_dict_CCG = Dict("$n" => [] for n ∈ sample) 
    η_dict_CCG = Dict("$n" => Dict("ηpos" => [], "ηneg" => []) for n ∈ sample) 
    m_master = create_master_problem(data, G, ρ)
    initilization_restricted_master_CCG(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y, zstar, PredictionStar, α)
    m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
    convergence = false
    while ~convergence && iter ≤ 20
        if iter == 1
            global x_CCG, λ_CCG, μ_CCG, θ_CCG, LB_CCG = warm_start_solution_CCG(data, instalations, demand, G, D, ξ_dict, Ξ_y, cuts_CCG, old_cuts, zstar, PredictionStar, ρ)
        else
            global x_CCG, λ_CCG, μ_CCG, θ_CCG, LB_CCG = solve_master_problem(m_master)
        end
        convergence = solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y, x_CCG, λ_CCG, μ_CCG, θ_CCG, G, D, y_dict_CCG, π_dict_CCG, σ_dict_CCG, old_cuts_CCG, zstar, PredictionStar, obj_dict_CCG, η_dict_CCG)
        update_master_CCG(data, m_master, ξ_dict, y_dict_CCG, G, D, zstar, PredictionStar, cuts_CCG)
        iter +=  1
    end

    return x_CCG, λ_CCG, μ_CCG, θ_CCG, LB_CCG, iter - 1

end

function Benders_with_warm_start(Data, Ξ_y_new, zstar_new, PredictionStar, ρ_new, α, cuts_Benders)

    # New context
    ϵ = 0.1
    zstar_new_ = vcat(zstar...) .+ (rand(length(vcat(zstar...))) .- 0.5) .* (2 * ϵ)
    zstar_new = @views [zstar_new_[((b - 1) * Tf + 1):(b * Tf)] for b in 1:length(bus_renewable_generation)]

    Ξ_y_new = uncertainty_support(Data, zstar_new, errors)

    PredictionStar_new = forecast(Data, zstar_new, Models)

    ρ_new = 1.1*critical_radius(Data, errors, Ξ_y_new, α, zstar_new) 

    N = Data.N
    z_aux = Data.z
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Tb_renewable = length(bus_renewable_generation)
    number_of_generators = Data.Ig
    Tf = Data.Tf
    Ig = collect(1:number_of_generators)
    T = collect(1:Tf)
    sample = collect(1:N)

    global_residual_scenarios = zeros(Tf * Tb_renewable, N)
    global_contextual_scenarios =  zeros(Tf * Tb_renewable, N)

    for b in bus_renewable_generation
        b_ = ind_renewable_generation[b]
        row_range = ((b_ - 1) * Tf + 1):(b_ * Tf)
    
        for n in 1:N
            vector_residual_24h = errors[b_][n] 
            vector_contextual_24h = z_aux[b_][n]
            global_residual_scenarios[row_range, n] = vector_residual_24h
            global_contextual_scenarios[row_range, n] = vector_contextual_24h
        end
    end

    y = [s[:] for s in eachcol(global_residual_scenarios)]
    z = [s[:] for s in eachcol(global_contextual_scenarios)]

    oracle_dict_warm = Dict("y" => Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample), "π" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "π_s" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "x1" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x2" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x3" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x4" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x5" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "x6" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "const" => Dict("$n" => 0.0 for n ∈ sample), "time" => Dict("$n" => 0.0 for n ∈ sample), "ϑ" => zeros(N))
    cuts_Benders_warm = Dict("$n" => Dict("signatures" => Set{Tuple{Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}},Float64,Float64}}(), "π" => Array{Array{Float64},1}(), "π_s" => Array{Array{Float64},1}(), "y_opt" => Array{Array{Float64},1}(), "const" => Array{Float64,1}()) for n in sample)

    iter = 1

    LowerBound_warm = []
    UpperBound_warm = []
    Time_warm = []
    cuts_warm = Dict("$n" => [] for n ∈ sample)

    m_master_warm = create_master_problem(Data, ρ_new)
    m_oracle, obj  = create_oracle_problem(Data)
    convergence_warm = false
    Max_iter = 250
    cumulative_time_warm = 0.0
    UB_Benders_warm = Inf
    gap_warm =   Inf

    while ~convergence_warm && iter ≤ Max_iter && gap_warm ≥ 1e-1

        if iter == 1
            global u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, μ_Benders_warm, θ_Benders_warm, LB_Benders_warm, time_master_Benders_warm, initial_cuts_warm, idx_warm = warm_start_solution_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new, ρ_new, α, cuts_Benders)
        else                                                                
            global u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, μ_Benders_warm, θ_Benders_warm, LB_Benders_warm, time_master_Benders_warm = solve_master_problem(m_master_warm)
        end
        convergence_warm = solve_oracle_problem(Data, m_oracle, obj, z, y, Ξ_y_new, u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, μ_Benders_warm, θ_Benders_warm, oracle_dict_warm, zstar_new, PredictionStar_new, initial_cuts_warm, idx_warm)
        UB_Benders_warm = minimum([UB_Benders_warm, Upper_Bound(Data, u_Benders_warm, v_Benders_warm, w_Benders_warm, λ_Benders_warm, oracle_dict_warm["ϑ"], α)])
        update_master_Benders(Data, m_master_warm, z, zstar_new, y, oracle_dict_warm, cuts_Benders_warm, cuts_warm)
        total_time_oracle_warm = sum(oracle_dict_warm["time"]["$n"][end] for n ∈ sample)
        cumulative_time_warm += time_master_Benders_warm + total_time_oracle_warm
        total_time_warm = time_master_Benders_warm + total_time_oracle_warm
        println("================================================")
        println(" Lower Bound: $LB_Benders_warm\n") 
        println(" Iteration: $iter")
        println("================================================") 
        push!(LowerBound_warm, LB_Benders_warm)
        push!(UpperBound_warm, UB_Benders_warm)
        push!(Time_warm, (time_master_Benders_warm, total_time_oracle_warm, total_time_warm))

        if iter > 1
            gap_warm =( UB_Benders_warm - LB_Benders_warm)/UB_Benders_warm
        end

        iter +=  1

    end

    oracle_dict_new = Dict("y" => Dict("$n" => [zeros(Tf) for b ∈ collect(1:Tb_renewable)] for n ∈ sample), "π" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "π_s" => Dict("$n" => Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T) for n ∈ sample), "x1" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x2" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T) for n ∈ sample), "x3" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x4" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end]) for n ∈ sample), "x5" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "x6" => Dict("$n" => Dict("($i, $t)" => 0.0 for i in Ig, t in [1]) for n ∈ sample), "const" => Dict("$n" => 0.0 for n ∈ sample), "time" => Dict("$n" => 0.0 for n ∈ sample), "ϑ" => [-Inf for n ∈ sample])
    cuts_Benders_new = Dict("$n" => Dict("signatures" => Set{Tuple{Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}}, Tuple{Vararg{Float64}},Float64,Float64}}(), "π" => Array{Array{Float64},1}(), "π_s" => Array{Array{Float64},1}(), "y_opt" => Array{Array{Float64},1}(), "const" => Array{Float64,1}()) for n in sample)
    iter = 1

    LowerBound_new = []
    UpperBound_new = []
    Time_new = []
    cuts_new = Dict("$n" => [] for n ∈ sample)

    m_master_new = create_master_problem(Data, ρ_new)
    initial_cuts_new, idx_new = initilization_restricted_master_Benders(Data, m_master_new, z, zstar_new, y, Ξ_y_new, PredictionStar_new, α)
    m_oracle, obj  = create_oracle_problem(Data)
    convergence_new = false
    Max_iter = 250
    cumulative_time_new = 0.0
    UB_Benders_new = Inf
    gap_new =   Inf

    while ~convergence_new && iter ≤ Max_iter # && gap_new ≥ 1e-1

        global u_Benders_new, v_Benders_new, w_Benders_new, λ_Benders_new, μ_Benders_new, θ_Benders_new, LB_Benders_new, time_master_Benders_new = solve_master_problem(m_master_new)

        println("================================================")
        println(" Lower Bound: $LB_Benders_new\n") 
        println(" Iteration: $iter")
        println("================================================")                                        

        convergence_new = solve_oracle_problem(Data, m_oracle, obj, z, y, Ξ_y_new, u_Benders_new, v_Benders_new, w_Benders_new, λ_Benders_new, μ_Benders_new, θ_Benders_new, oracle_dict_new, zstar_new, PredictionStar_new, initial_cuts_new, idx_new)
        UB_Benders_new = minimum([UB_Benders_new, Upper_Bound(Data, u_Benders_new, v_Benders_new, w_Benders_new, λ_Benders_new, oracle_dict_new["ϑ"], α)])
        update_master_Benders(Data, m_master_new, z, zstar_new, y, oracle_dict_new, cuts_Benders_new, cuts_new)
        total_time_oracle_new = sum(oracle_dict_new["time"]["$n"][end] for n ∈ sample)
        cumulative_time_new += time_master_Benders_new + total_time_oracle_new
        total_time_new = time_master_Benders_new + total_time_oracle_new
        push!(LowerBound_new, LB_Benders_new)
        push!(UpperBound_new, UB_Benders_new)
        push!(Time_new, (time_master_Benders_new, total_time_oracle_new, total_time_new))

        gap_new =( UB_Benders_new - LB_Benders_new)/UB_Benders_new
        iter +=  1

    end


    return x_Benders_new, λ_Benders_new, μ_Benders_new, θ_Benders_new, LB_Benders_new, iter - 1
   
end

LowerBound_dict = load( joinpath(@__DIR__, "results", "LowerBound.jld2"))
UpperBound_dict = load( joinpath(@__DIR__, "results", "UpperBound.jld2"))
LowerBound = LowerBound_dict["LowerBound"]
UpperBound = UpperBound_dict["UpperBound"]

initial_iter = 50
step = 2

xmin, xmax = initial_iter, length(LowerBound)
ymin, ymax = LowerBound[initial_iter]/1e5-10, UpperBound[initial_iter]/1e5
ymin, ymax = LowerBound[initial_iter]/1e5, LowerBound[end]/1e5


plot(collect(initial_iter:step:length(LowerBound)), LowerBound[initial_iter:step:end]/1e5, 
    lw = 2, 
    xlims = (xmin, xmax),
    ylims = (ymin, ymax),
    xlabel = L"Iterations", 
    ylabel = L"Cost", 
    label = L"LB", 
    yformatter = :plain,
    top_margin = 12mm,    
    left_margin = 10mm )
annotate!(
    xmin - 5,                        
    ymax,                   
    text("x "*L"10^5",           
         :left,                    
         :center,               
         rotation = 0,          
         10                       
    )
)
plot!(collect(initial_iter:step:length(UpperBound)), UpperBound[initial_iter:step:end]/1e5, lw = 2, label = L"UB")



savefig("images/bounds.png")
GAP = (UpperBound - LowerBound)./UpperBound


plot(collect(50:Max_iter), 100*GAP[50:end], 
    lw = 2, 
    color = :black,
    linestyle = :dash,
    xlabel = L"Iterations", 
    ylabel = L"GAP"*" (%)", 
    label = "", 
    yformatter = :plain
  )

t_master_vec = [t[1] for t in Time]
t_oracle_vec = [t[2] for t in Time]
matriz_tiempos = [t_master_vec t_oracle_vec]

p = plot(collect(1:Max_iter), t_master_vec,
    lw = 3,
    color = :blue,
    label = "Tiempo Máster",
    xlabel = "Iteración",
    ylabel = "Tiempo Máster (segundos)",
    legend = :topleft,
    right_margin = 15mm # Margen extra a la derecha para que quepa el segundo eje
)

plot!(twinx(p), collect(1:Max_iter), t_oracle_vec,
    lw = 3,
    color = :orange,
    label = "Tiempo Oráculo",
    ylabel = "Tiempo Oráculo (segundos)",
    box = :off, # Evita que las líneas de la caja se dupliquen de forma extraña
    legend = :topright
)

m_master_warm = create_master_problem(Data, ρ_new)
initilization_restricted_master_Benders(Data, m_master_warm, z, zstar_new, y, Ξ_y_new, PredictionStar_new, α)


vars_m1 = all_variables(m_master_new)
vars_m2 = all_variables(m_master_warm)

mapeo = Dict(vars_m1[i] => vars_m2[i] for i in collect(1:length(vars_m1)))


function traducir_restriccion_rapida(m_master_warm, con_m1, mapeo)

    obj_m1 = constraint_object(con_m1)
    lhs_m2 = @expression(m_master_warm, AffExpr())
    add_to_expression!(lhs_m2, obj_m1.func.constant)
    
    for (coef, v_m1) in JuMP.linear_terms(obj_m1.func)
        v_m2 = mapeo[v_m1] # Acceso en memoria instantáneo
        add_to_expression!(lhs_m2, coef, v_m2)
    end
    set_m1 = obj_m1.set
    set_m2 = if set_m1 isa MOI.LessThan
        MOI.LessThan(set_m1.upper) # Extrae el número límite superior
    elseif set_m1 isa MOI.GreaterThan
        MOI.GreaterThan(set_m1.lower) # Extrae el número límite inferior
    elseif set_m1 isa MOI.EqualTo
        MOI.EqualTo(set_m1.value) # Extrae el valor de la igualdad
    else
        # Si es un conjunto complejo (cónico o indicador), arrojar error descriptivo
        error("El conjunto $(typeof(set_m1)) es complejo y requiere traducción especial.")
    end

    add_constraint(m_master_warm, ScalarConstraint(lhs_m2, set_m2)) 
    
    return nothing
end

for n ∈ 1:N 

    for con_original in cuts_new["$n"]

        traducir_restriccion_rapida(m_master_warm, con_original, mapeo)

    end

end

optimize!(m_master_warm)


todas_las_cons = Set(all_constraints(m_master, AffExpr, MOI.LessThan{Float64}[end]))


generated_cons = Set(vcat([cuts["$n"][i] for i ∈ eachindex(cuts["$n"]), n ∈ 1:N]...))
total_cons = Set(all_constraints(m_master, AffExpr, MOI.LessThan{Float64}))
diiif_con = setdiff(total_cons, generated_cons)


initial_cons = []    

for i ∈ Ig, t ∈ T 
    lhs_1 = AffExpr()
    add_to_expression!(lhs_1, 1.0, m_master[:v][i,t])
    add_to_expression!(lhs_1, -1.0, m_master[:u][i,t])
    rhs_set_1 = MOI.LessThan(0.0)
    push!(initial_cons, ScalarConstraint(lhs_1, rhs_set_1))
end 

initial_con_ref = []

for con_ref ∈ initial_cons

    for con in total_cons
        obj_existente = constraint_object(con)
        
        # Comparamos el Lado Izquierdo (LHS) con tolerancia
        lhs_igual = con_ref.func == obj_existente.func
        
        # Comparamos el Lado Derecho (RHS) con tolerancia
        rhs_igual = isapprox(con_ref.set.upper, obj_existente.set.upper, atol = 1e-7)
        
        # Si ambos coinciden, hemos encontrado la restricción en el modelo
        if lhs_igual && rhs_igual
            push!(initial_con_ref, con)
        end
    end

end


initial_cons_2 = []    

for i ∈ Ig, t ∈ T 
    lhs_2 = AffExpr()
    add_to_expression!(lhs_2, 1.0, m_master[:w][i,t])
    add_to_expression!(lhs_2, 1.0, m_master[:u][i,t])
    rhs_set_2 = MOI.LessThan(1.0)
    push!(initial_cons_2, ScalarConstraint(lhs_2, rhs_set_2))
end 

initial_con_ref_2 = []

for con_ref ∈ initial_cons_2

    for con in total_cons
        obj_existente = constraint_object(con)
        
        # Comparamos el Lado Izquierdo (LHS) con tolerancia
        lhs_igual = con_ref.func == obj_existente.func
        
        # Comparamos el Lado Derecho (RHS) con tolerancia
        rhs_igual = isapprox(con_ref.set.upper, obj_existente.set.upper, atol = 1e-7)
        
        # Si ambos coinciden, hemos encontrado la restricción en el modelo
        if lhs_igual && rhs_igual
            push!(initial_con_ref_2, con)
        end
    end

end

set_initial_cons = Set(initial_con_ref)∪Set(initial_con_ref_2)

different_cons = setdiff(collect(diiif_con), collect(set_initial_cons))

all_constraints(m_master_warm, AffExpr, MOI.LessThan{Float64})[end]


for i ∈ Ig, t ∈ T 
    fix(m_master_new[:u][i, t], u_Benders_warm[i, t], force = true)
    fix(m_master_new[:v][i, t], v_Benders_warm[i, t], force = true)
    fix(m_master_new[:w][i, t], w_Benders_warm[i, t], force = true)
end

update_master_Benders(Data, m_master_new, z, zstar_new, y, oracle_dict_warm, cuts_Benders_new, cuts_new)


optimize!(m_master_new)
objective_value(m_master_new)


UB = Upper_Bound(Data, u_Benders_new, v_Benders_warm, w_Benders_warm, λ_Benders_warm, oracle_dict_warm["ϑ"], α)

oracle_dict_warm["ϑ"] == oracle_dict_new["ϑ"]
μ_Benders_new
θ_Benders_new

fix(m_master_new[:θ], oracle_dict_new["ϑ"][n], force = true)
optimize!(m_master_new)
termination_status(m_master_new) 
n = 10

val = [value(constraint_object(cuts_new["$n"][i]).func) + θ_Benders_new - constraint_object(cuts_new["$n"][i]).set.upper for i ∈ eachindex(cuts_new["$n"])]
val_initial = [value(constraint_object(initial_cuts["$n"]).func) + θ_Benders_new - constraint_object(initial_cuts["$n"]).set.upper ]

path  = joinpath(@__DIR__, "results", "cuts_warm.jld2")
jldsave(path;  cuts_warm)

write_to_file(m_master_warm, joinpath(@__DIR__, "results", "m_master_warm.mps"))

sort(findall([oracle_dict_new["ϑ"][n] for n in sample] .!= oracle_dict_new["ϑ"][1])) ==sort(idx_new)

oracle_dict_new["ϑ"][87]
