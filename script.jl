using Pkg
Pkg.activate(".")
Pkg.instantiate()

include(joinpath(pwd(),"getData.jl"))

using JuMP, Gurobi, JSON, Distributions, LazySets, Polyhedra, CDDLib, Dualization, LinearAlgebra, Statistics
using Plots, LaTeXStrings, Colors, ColorSchemes, StatsPlots
using Main.getData
const GUROBIg_ENV = Gurobi.Env()


function save_json(output_dict::Dict, path::String, output_name::String="output")
    # Adicionando o argumento que permite o símbolo Inf
    jD = JSON.json(output_dict, allownan=true)

    open(joinpath(path, "$output_name.json"), "w") do j
        write(j, jD)
    end

    return
end

function open_json(input_file::String, path::String)
    file_data = JSON.parsefile(joinpath(path, input_file))
    return file_data
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

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBIg_ENV), "OutputFlag" => 0))
    @variable(model, α[1:Tb, 1:T[end]]) # phase
    @variable(model, f[1:L[end], 1:T[end]]) # power flow
    @variable(model, p[1:Ig[end], 1:T[end]] >= 0) # generation
    @variable(model, pdef[1:Tb, 1:T[end]] >= 0) # deficit generation
    @variable(model, s[bus_renewable_generation, 1:T[end]] >= 0) # spillage

    @constraint(model, [t in T], f[:, t] .== S*α[:, t])
    @constraint(model, [b in Bus, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] == D[b][t]) # Power balance
    #  @constraint(model, [b in Bus, t in T],  pdef[b, t] ≤ D[b][t]) 
    @constraint(model, θ[b in bus_renewable_generation, t in T], B[b,:]'*p[:, t] + F[b,:]'*f[:, t] + pdef[b, t] - s[b, t] == D[b][t]) # Uncertainty
    # @constraint(model, [b in bus_renewable_generation, t in T], pdef[b, t] ≤ D[b][t]) 
    @constraint(model, θ_s[b in bus_renewable_generation, t in T], s[b, t] <=  0) # Uncertainty
    @constraint(model, [l in L, t in T], f[l,t] <= Fmax[l])
    @constraint(model, [l in L, t in T], f[l,t] >= -Fmax[l])
    @constraint(model, x1[i in Ig, t in T], p[i,t] <= Pmax[i]*u[i, t])
    @constraint(model, x2[i in Ig, t in T], p[i,t] >= Pmin[i]*u[i, t])

    @constraint(model, x3[i in Ig, t in T[2:end]], p[i, t] - p[i, t-1] <= RU[i]*u[i, t-1] + Pmax[i]*v[i, t])
    @constraint(model, x4[i in Ig, t in T[2:end]], p[i, t-1] - p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t])

    # Ignitial state constraints
    @constraint(model, x5[i in Ig, t in [1]], p[i, t] <= RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i])
    @constraint(model, x6[i in Ig, t in [1]], -p[i, t] <= RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i])

    @objective(model, Min, sum(Cp[i]*p[i,t] for i in Ig, t in T) + sum(Cdef*pdef[b,t] for b in 1:Tb, t in T))

    return model
end

function oracle(Data)

    M = Data.M
    Tf = Data.Tf
    number_of_generators = Data.Ig
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    Ξ_b = Data.Ξ_b
    ξ = Data.ξ
    T = collect(1:Tf)
    u = zeros(number_of_generators, Tf)
    v = zeros(number_of_generators, Tf)
    w = zeros(number_of_generators, Tf)
    m = economic_dispatch_problem(Data, u, v, w)
    m_oracle = dualize(m; dual_names = DualNames())
    @variables(m_oracle, begin
        β[b in bus_renewable_generation, t in T] ≥ 0
        z[b in bus_renewable_generation, t in T], Bin
        α ≥ 0
        t ≥ 0
        γ
    end)
    @constraints(m_oracle, begin
        [b in bus_renewable_generation, t in T], β[b, t] ≥- m_oracle[Symbol("θ[$b,$t]")] + m_oracle[Symbol("θ_s[$b,$t]")]
        [b in bus_renewable_generation, t in T], β[b, t] ≥ m_oracle[Symbol("θ[$b,$t]")] - m_oracle[Symbol("θ_s[$b,$t]")]
        [b in bus_renewable_generation, t in T], m_oracle[Symbol("θ[$b,$t]")] + m_oracle[Symbol("θ_s[$b,$t]")] ≤ M*z[b, t]
        [b in bus_renewable_generation, t in T], m_oracle[Symbol("θ[$b,$t]")] + m_oracle[Symbol("θ_s[$b,$t]")] ≥- M*(1 - z[b, t])
        t ≥ α*γ_ub + γ*α_ub - α_ub*γ_ub
        t ≤ α*γ_ub
        t ≤ γ*α_ub
        # t ≤ α*γ
    end)
    @constraint(m_oracle, norm_l1, γ == sum(β) )
    @constraint(m_oracle, con[b in bus_renewable_generation, t in T], α ≤ z[b, t]*(Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][1][t]) + (1 - z[b, t])*(ξ[ind_renewable_generation[b]][1][t] - Ξ_b[ind_renewable_generation[b]][:,1][t]))
    # set_optimizer(m_oracle,optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBIg_ENV), "OutputFlag" => 0, "NonConvex" => 2, "Crossover"=> 1, "NumericFocus"=> 2))
    set_optimizer(m_oracle,optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBIg_ENV), "OutputFlag" => 0))

    obj = objective_function(m_oracle)

    return m_oracle, obj
  
end

function modify_oracle_model(Data, model, obj, u ,v, w, λ, n)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    RD = Data.RD
    RU = Data.RU
    ξ = Data.ξ
    D = Data.D
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)
    Ξ_b = Data.Ξ_b
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation
    T = collect(1:Tf)
    Ig = collect(1:number_of_generators)
    ex1 = AffExpr()
    ex2 = AffExpr()
    for t in T
        for b in bus_renewable_generation
            set_normalized_coefficient(model[:con][b, t], model[:z][b, t],-Ξ_b[ind_renewable_generation[b]][:,1][t] - Ξ_b[ind_renewable_generation[b]][:,2][t] + 2*ξ[ind_renewable_generation[b]][n][t])
            set_normalized_rhs(model[:con][b, t], ξ[ind_renewable_generation[b]][n][t] - Ξ_b[ind_renewable_generation[b]][:,1][t])
            add_to_expression!(ex1, -ξ[ind_renewable_generation[b]][n][t], model[Symbol("θ[$b,$t]")])
            add_to_expression!(ex1, ξ[ind_renewable_generation[b]][n][t], model[Symbol("θ_s[$b,$t]")])
        end
        for i in Ig
            add_to_expression!(ex2, Pmax[i]*u[i, t], model[Symbol("x1[$i,$t]")])
            add_to_expression!(ex2, Pmin[i]*u[i, t], model[Symbol("x2[$i,$t]")])
            if t == 1
                add_to_expression!(ex2, RU[i]*u0[i] + Pmax[i]*v[i, t] + p0[i], model[Symbol("x5[$i,$t]")])
                add_to_expression!(ex2, RD[i]*u[i, t] + Pmax[i]*w[i, t] - p0[i], model[Symbol("x6[$i,$t]")])
            else
                add_to_expression!(ex2, RU[i]*u[i, t-1] + Pmax[i]*v[i, t], model[Symbol("x3[$i,$t]")])
                add_to_expression!(ex2, RD[i]*u[i, t] + Pmax[i]*w[i, t], model[Symbol("x4[$i,$t]")])
            end
        end
    end
    set_normalized_rhs(model[:norm_l1],- λ)
    # @constraint(model, obj + ex1 + ex2 + model[:t] ≤ 1000)
    @objective(model, Max, obj + ex1 + ex2 + model[:t])

    return ex1, ex2

end

function call_oracle( Data, model_oracle, obj, tol, u, v, w, λ, s, oracle_dict, α_ub, γ_ub)

    N = Data.N
    Tf = Data.Tf
    number_of_generators = Data.Ig
    T = collect(1:Tf) 
    Ig = collect(1:number_of_generators) 
    bus_renewable_generation = Data.bus_renewable_generation

    θ_val = Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    θ_s_val = Dict("($b, $t)" => 0.0  for b in bus_renewable_generation, t in T)
    x1_val = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x2_val = Dict("($i, $t)" => 0.0 for i in Ig, t in T)
    x3_val = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x4_val = Dict("($i, $t)" => 0.0 for i in Ig, t in T[2:end])
    x5_val = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])
    x6_val = Dict("($i, $t)" => 0.0 for i in Ig, t in [1])

    for n in 1:N 

        # n = 2

        if tol[n]

           ex1, ex2 =  modify_oracle_model(Data, model_oracle, obj, u ,v, w, λ, n)

            # Optimization
            val, time_oracle = @timed(begin
                optimize!(model_oracle)
                status = termination_status(model_oracle)
                if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED 
                    error("problem $status \n scenario n = $n")
                end
            end)

            # Solution
            for t in T
                for b in bus_renewable_generation
                    θ_val["($b, $t)"] = value.(model_oracle[Symbol("θ[$b,$t]")])
                    θ_s_val["($b, $t)"] = value.(model_oracle[Symbol("θ_s[$b,$t]")])
                end
                for i in Ig
                    x1_val["($i, $t)"] = value.(model_oracle[Symbol("x1[$i,$t]")])
                    x2_val["($i, $t)"] = value.(model_oracle[Symbol("x2[$i,$t]")])
                    if t == 1
                        x5_val["($i, $t)"] = value.(model_oracle[Symbol("x5[$i,$t]")])
                        x6_val["($i, $t)"] = value.(model_oracle[Symbol("x6[$i,$t]")])
                    else
                        x3_val["($i, $t)"] = value.(model_oracle[Symbol("x3[$i,$t]")])
                        x4_val["($i, $t)"] = value.(model_oracle[Symbol("x4[$i,$t]")])
                    end
                end
            end

            obj_val = objective_value(model_oracle)
            push!(oracle_dict["ρ"][n], obj_val)
            push!(oracle_dict["time_oracle"][n], time_oracle)
            const_val = obj_val - value(ex1) - value(ex2) - value(model_oracle[:t])
            if  obj_val - s[n] > 1e-2 
                ℓ = length(oracle_dict["K"][n]) + 1
                push!(oracle_dict["K"][n], ℓ)
                push!(oracle_dict["Θ"][n], θ_val)
                push!(oracle_dict["Θ_s"][n], θ_s_val)
                push!(oracle_dict["X1"][n], x1_val)
                push!(oracle_dict["X2"][n], x2_val)
                push!(oracle_dict["X3"][n], x3_val)
                push!(oracle_dict["X4"][n], x4_val)
                push!(oracle_dict["X5"][n], x5_val)
                push!(oracle_dict["X6"][n], x6_val)
                push!(oracle_dict["const"][n], const_val)
                push!(oracle_dict["tol"][n], obj_val - s[n] )
                push!(oracle_dict["status"][n], status)
            else
                tol[n] = false
            end
        end
    end

    return nothing

end

function master(Data, δ)
    Tf = Data.Tf
    number_of_generators = Data.Ig
    N = Data.N
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators

    # Ignitial state
    u0 = zeros(number_of_generators)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBIg_ENV), "OutputFlag" => 0, "DualReductions"=> 0))
    @variable(model, 0 <= v[1:Ig[end], 1:T[end]] <= 1) # start-up
    @variable(model, 0 <= w[1:Ig[end], 1:T[end]] <= 1) # shutdown
    @variable(model, u[1:Ig[end], 1:T[end]], Bin) # commit
    @variable(model, s[1:N])
    @variable(model, λ >= 0)
    @constraint(model, [i in Ig, t in T[2:end]], v[i, t] - w[i, t] == u[i, t] - u[i, t-1])
    @constraint(model, [i in Ig, t in T], v[i, t] <= u[i, t])
    @constraint(model, [i in Ig, t in T], w[i, t] <= 1 - u[i, t])

    # Ignitial state constraint
    @constraint(model, [i in Ig, t in [1]], v[i, t] - w[i, t] == u[i, t] - u0[i])

    return model

end

function update_master(Data, model_master,  oracle_dict, tol)
    
    Tf = Data.Tf
    number_of_generators = Data.Ig
    Pmin = Data.Pmin
    Pmax = Data.Pmax
    RD = Data.RD
    RU = Data.RU
    ξ = Data.ξ
    Ξ_b = Data.Ξ_b
    p0 = zeros(number_of_generators)
    u0 = zeros(number_of_generators)
    bus_renewable_generation = Data.bus_renewable_generation
    ind_renewable_generation = Data.ind_renewable_generation    
    T = collect(1:Tf)
    Ig = collect(1:number_of_generators)
    N = Data.N
    M = Data.M

    # n = 1
    # for b in bus_renewable_generation, t in T
    #     print("ξ(b = $b, n = $n, t = $t) = $(Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][n][t]) \n")
    # end

    # for b in bus_renewable_generation, t in T
    #     print("θ(b = $b, t = $t) = $(θ["($b, $t)"]) \n")
    # end

    # δ = [θ["($b, $t)"] > 0 ? Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][n][t] : θ["($b, $t)"] < 0 ?  ξ[ind_renewable_generation[b]][n][t] - Ξ_b[ind_renewable_generation[b]][:,1][t] : Inf for b in bus_renewable_generation, t in T ]

    # b , t = 1, 21

    for n in 1:N 

        # n = 1

        if tol[n]
            # α  = (θ, θ_s) -> minimum([θ["($b, $t)"] + θ_s["($b, $t)"]  > 0 ? ( Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][n][t] == 0 ? Inf : Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][n][t]) : θ["($b, $t)"] + θ_s["($b, $t)"]  < 0 ?  (ξ[ind_renewable_generation[b]][n][t] - Ξ_b[ind_renewable_generation[b]][:,1][t] == 0 ? Inf : ξ[ind_renewable_generation[b]][n][t] - Ξ_b[ind_renewable_generation[b]][:,1][t]) :  Inf for b in bus_renewable_generation, t in T ])
            α  = (θ, θ_s) -> minimum([-θ["($b, $t)"] + θ_s["($b, $t)"]  > 0 ? Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][n][t] : - θ["($b, $t)"] + θ_s["($b, $t)"]  < 0 ?  ξ[ind_renewable_generation[b]][n][t] - Ξ_b[ind_renewable_generation[b]][:,1][t] :  Inf for b in bus_renewable_generation, t in T ])
            θ = oracle_dict["Θ"][n][end]
            θ_s = oracle_dict["Θ_s"][n][end]
            x1 = oracle_dict["X1"][n][end]
            x2 = oracle_dict["X2"][n][end]
            x3 = oracle_dict["X3"][n][end]
            x4 = oracle_dict["X4"][n][end]
            x5 = oracle_dict["X5"][n][end]
            x6 = oracle_dict["X6"][n][end]
            const_term = oracle_dict["const"][n][end]
            # β = @variable(model, [b in bus_renewable_generation, t in T], lower_bound = 0)
            # z = @variable(model, [b in bus_renewable_generation, t in T], Bin)
            # α = @variable(model, lower_bound = 0)
            t = @variable(model_master, lower_bound = 0)
            γ = @variable(model_master)

            @constraints(model_master, begin
                γ == norm([θ["($b, $t)"] + θ_s["($b, $t)"] for b in bus_renewable_generation, t in T] ,1) - model_master[:λ]
                # [b in bus_renewable_generation, t in T], β[b, t] ≥ θ["($b, $t)"]
                # [b in bus_renewable_generation, t in T], β[b, t] ≥ -θ["($b, $t)"]
                # [b in bus_renewable_generation, t in T], θ["($b, $t)" ]≤ M*z[b, t]
                # [b in bus_renewable_generation, t in T], θ["($b, $t)"]≥- M*(1 - z[b,t])
                # [b in bus_renewable_generation, t in T], α ≤ z[b, t]*(Ξ_b[ind_renewable_generation[b]][:,2][t] - ξ[ind_renewable_generation[b]][n][t]) + (1 - z[b, t])*(ξ[ind_renewable_generation[b]][n][t] - Ξ_b[ind_renewable_generation[b]][:,1][t])
                t ≥ α(θ, θ_s)*γ
            end)   
                            
            ex1 = AffExpr()
            for t in T
                for i in Ig
                    add_to_expression!(ex1, x1["($i, $t)"], Pmax[i]*model_master[:u][i, t])
                    add_to_expression!(ex1, x2["($i, $t)"],  Pmin[i]*model_master[:u][i, t])
                    if t == 1
                        add_to_expression!(ex1, x5["($i, $t)"], RU[i]*u0[i] + Pmax[i]*model_master[:v][i, t] + p0[i])
                        add_to_expression!(ex1, x6["($i, $t)"], RD[i]*model_master[:u][i, t] + Pmax[i]*model_master[:w][i, t] - p0[i])
                    else
                        add_to_expression!(ex1, x3["($i, $t)"], RU[i]*model_master[:u][i, t-1] + Pmax[i]*model_master[:v][i, t])
                        add_to_expression!(ex1, x4["($i, $t)"], RD[i]*model_master[:u][i, t] + Pmax[i]*model_master[:w][i, t])
                    end
                end
            end
            @constraint(model_master, const_term + ex1 - sum(θ["($b, $t)"]*ξ[ind_renewable_generation[b]][n][t] for b in bus_renewable_generation, t in T ) + sum(θ_s["($b, $t)"]*ξ[ind_renewable_generation[b]][n][t] for b in bus_renewable_generation, t in T ) + t ≤ model_master[:s][n])    
            # @constraint(model, con, sum(θ["($b, $t)"]*ξ[ind_renewable_generation[b]][n][t] for b in bus_renewable_generation, t in T ) + t ≤ model[:s][n])           
        end
    end

    return nothing
end

function solve_master(Data, model, iter, u_iter, v_iter, w_iter; regularized = false)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators

    N = Data.N
    sample = 1:N

    if iter == 1
            @objective(model, Min, sum(Cu[i]*model[:u][i, t] for i in Ig, t in T) + sum(Csu[i]*model[:v][i, t] for i in Ig, t in T) +  sum(Csd[i]*model[:w][i, t] for i in Ig, t in T) + model[:λ]*δ)
    else
            @objective(model, Min, sum(Cu[i]*model[:u][i, t] for i in Ig, t in T) + sum(Csu[i]*model[:v][i, t] for i in Ig, t in T) +  sum(Csd[i]*model[:w][i, t] for i in Ig, t in T) + model[:λ]*δ + (1/N)* sum(model[:s][n] for n ∈ sample))
            if regularized
                    u_pos = @variable(model_master, [1:Ig[end], 1:T[end]], lower_bound = 0)
                    v_pos = @variable(model_master, [1:Ig[end], 1:T[end]], lower_bound = 0)
                    w_pos = @variable(model_master, [1:Ig[end], 1:T[end]], lower_bound = 0)  
                    @constraint(model_master, model_master[:u] - u_iter .<= u_pos)
                    @constraint(model_master, -u_pos .<= model_master[:u] - u_iter)
                    @constraint(model_master, model_master[:v] - v_iter .<= v_pos)
                    @constraint(model_master, -v_pos .<= model_master[:v] - v_iter)
                    @constraint(model_master, model_master[:w] - w_iter .<= w_pos)
                    @constraint(model_master, -w_pos .<= model_master[:w] - w_iter)
                    obj_fun = objective_function(model_master)
                    @objective(model_master, Min, obj_fun + sum(u_pos) + sum(v_pos) + sum(w_pos))
            end

    end

    val, time_master = @timed(begin
        optimize!(model)
        status = termination_status(model)
        if status != MOI.OPTIMAL
            error("problem $status")
        end
    end)   

    # Solution
    u = value.(model[:u])
    v = value.(model[:v])
    w = value.(model[:w])
    λ = value.(model[:λ])
    if iter == 1
        s = [-Inf for n ∈ sample]
    else
        s = value.(model[:s])
    end

    return u, v, w, λ, s, time_master

end

function regularized(Data, model_master, u_iter, v_iter, w_iter)
    Tf = Data.Tf
    number_of_generators = Data.Ig
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators

    # Initial state
    u_pos = @variable(model_master, [1:Ig[end], 1:T[end]], lower_bound = 0)
    v_pos = @variable(model_master, [1:Ig[end], 1:T[end]], lower_bound = 0)
    w_pos = @variable(model_master, [1:Ig[end], 1:T[end]], lower_bound = 0)  
    @constraint(model_master, model_master[:u] - u_iter .<= u_pos)
    @constraint(model_master, -u_pos .<= model_master[:u] - u_iter)
    @constraint(model_master, model_master[:v] - v_iter .<= v_pos)
    @constraint(model_master, -v_pos .<= model_master[:v] - v_iter)
    @constraint(model_master, model_master[:w] - w_iter .<= w_pos)
    @constraint(model_master, -w_pos .<= model_master[:w] - w_iter)
    obj_fun = objective_function(model_master)
    @objective(model_master, Min, obj_fun + sum(u_pos) + sum(v_pos) + sum(w_pos))
    return nothing 
end

function upper_boun_vale(Data, u, v, w, λ, δ, oracle_dict)

    Tf = Data.Tf
    number_of_generators = Data.Ig
    Cu = Data.Cu
    Csu = Data.Csu
    Csd = Data.Csd
    T = collect(1:Tf) # time periods
    Ig = collect(1:number_of_generators) # generators

    N = Data.N
    sample = 1:N
    UB  = sum(Cu[i]*u[i, t] for i in Ig, t in T) + sum(Csu[i]*v[i, t] for i in Ig, t in T) +  sum(Csd[i]*w[i, t] for i in Ig, t in T) + λ*δ + (1/N)* sum(oracle_dict["ρ"][n][end] for n ∈ sample)

    return UB

end

Max_iter = 100
sample = 1:Data.N
oracle_dict = Dict("K" =>  Dict(n => [] for n ∈ sample), "Θ" =>  Dict(n => [] for n ∈ sample), "Θ_s" =>  Dict(n => [] for n ∈ sample), "X1" =>  Dict(n => [] for n ∈ sample), "X2" =>  Dict(n => [] for n ∈ sample), "X3" =>  Dict(n => [] for n ∈ sample), "X4" =>  Dict(n => [] for n ∈ sample), "X5" =>  Dict(n => [] for n ∈ sample), "X6" =>  Dict(n => [] for n ∈ sample), "const" =>  Dict(n => [] for n ∈ sample), "tol" => Dict(n => [] for n ∈ sample), "ρ" => Dict(n => [] for n ∈ sample), "time_oracle" => Dict(n => [] for n ∈ sample), "status" => Dict(n => [] for n ∈ sample))
time_dict = Dict("time_master" => Dict(iter => [] for iter ∈ 1:Max_iter), "time_oracle" => Dict(iter => [] for iter ∈ 1:Max_iter), "total_time" => Dict(iter => [] for iter ∈ 1:Max_iter))
bounds_dict = Dict("LB" => Dict(iter => [] for iter ∈ 1:Max_iter), "UB" => Dict(iter => [] for iter ∈ 1:Max_iter), "tol" => Dict(iter => [] for iter ∈ 1:Max_iter))

tol = [true for n ∈ sample]
iter = 1
δ = 0.5
α_ub = 1e4
γ_ub = 1e4
model_master = master(Data, δ)
model_oracle, obj = oracle(Data)
u, v, w = NaN, NaN, NaN
UB, LB = Inf, -Inf

cumulative_time = 0.0

while  UB - LB ≥ 0.05*UB && iter ≤ Max_iter  #Base.any(tol) && iter ≤ 50
   global u, v, w, λ, s, time_master = solve_master(Data, model_master, iter, u, v, w)
   LB = objective_value(model_master)
    call_oracle( Data, model_oracle, obj, tol, u, v, w, λ, s, oracle_dict, α_ub, γ_ub)
    UB = minimum([UB, upper_boun_vale(Data, u, v, w, λ, δ, oracle_dict)])
    total_time_oracle = sum(oracle_dict["time_oracle"][n][end] for n ∈ sample)
    cumulative_time += time_master + total_time_oracle
    push!(time_dict["time_master"][iter], time_master)
    push!(time_dict["time_oracle"][iter], total_time_oracle)
    push!(time_dict["total_time"][iter], time_master + total_time_oracle)
    push!(bounds_dict["LB"][iter], LB)
    push!(bounds_dict["UB"][iter], UB)
    push!(bounds_dict["tol"][iter], (UB- LB)/UB)
    update_master(Data, model_master,  oracle_dict, tol)
    println("===============================================")
    println("                                              WASSERSTEIN                                                ")
    println("                         Iter= $iter , tol = $(oracle_dict["tol"][1][end])                     ")
    println("         time_oracle= $total_time_oracle , time_master = $time_master        ")
    println("                    gap= $((UB - LB)/UB) , time = $cumulative_time                    ")
    println("===============================================")
    iter +=  1
end

save_json(oracle_dict, joinpath(pwd(), "results"), "oracle_dict")
save_json(time_dict, joinpath(pwd(), "results"), "time_dict")
save_json(bounds_dict, joinpath(pwd(), "results"), "bounds_dict")

