################ Imports ################

using LinearAlgebra
using Exceptions
include("schemes_nd.jl")

################ User-defined exceptions ################

mutable struct NonConvergenceError <: Exception
end
mutable struct NonInvertibilityError <: Exception
end

################ Hamiltonian function and gradients ################

function H(
    q, p,
    V::Function,
    D::Function,
    ln_det_D::Function
)::Float64
    return V(q) - ln_det_D(q) / 2 + dot(p, D(q), p) / 2
end

function ∇qH(
    q, p,
    ∇V::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Float64}

    ∇V_q = ∇V(q)
    ∇D_q = ∇D(q)
    inv_D_q = inv_D(q)

    a = Vector{Float64}(undef, length(q))
    for i in eachindex(q)
        ∇D_q_i = @view ∇D_q[Block(i, 1)]
        a[i] = ∇V_q[i] + (dot(p, ∇D_q_i, p) - tr(inv_D_q * ∇D_q_i)) / 2
    end

    return a
end

function ∇pH(
    q, p,
    D::Function
)::Vector{Float64}
    return D(q) * p
end

function ∇pqH(
    q, p,
    ∇D::Function
)::Matrix{Float64}

    ∇D_q = ∇D(q)

    A = zeros(length(q), length(q))
    for i in 1:length(q)
        A[i, :] = ∇D_q[Block(i, 1)] * p
    end

    return A
end

function ∇p2H(
    q, p,
    D::Function
)::Matrix{Float64}
    return D(q)
end

function ∇q2H(
    q, p,
    ∇2V::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Matrix{Float64}

    ∇2V_q = ∇2V(q)
    ∇D_q = ∇D(q)
    ∇2D_q = ∇2D(q)
    inv_D_q = inv_D(q)

    A = zeros(length(q), length(q))
    for i in 1:length(q), j = i:length(q)
        ∇2D_q_ij = @view ∇2D_q[Block(i, j)]
        ∇D_q_i = @view ∇D_q[Block(i, 1)]
        ∇D_q_j = @view ∇D_q[Block(j, 1)]
        A[i, j] = (dot(p, ∇2D_q_ij, p) + tr(inv_D_q * ∇2D_q_ij - inv_D_q * ∇D_q_j * inv_D_q * ∇D_q_i)) / 2 + ∇2V_q[i, j]
        A[j, i] = A[i, j]
    end

    return A
end

################ Sampling momenta for RMHMC ################

function sample_momenta_RMHMC(
    q,
    inv_D_sqrt::Function
)::Vector{Float64}
    return inv_D_sqrt(q) * randn(length(q))
end

################ RMHMC algorithm ################

function RMHMC(
    q0, p0,
    n_iterations,
    numerical_scheme,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    eta_rev,
    V::Function,
    ∇V::Function,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function,
    inv_D_sqrt::Function,
    ln_det_D::Function,
)

    d = 2 * length(q0)
    mc_trajectory = Array{Float64}(undef, n_iterations + 1, d)
    mc_trajectory[1, :] = [q0..., p0...]
    q = copy(q0)
    p = copy(p0)

    for n in 1:n_iterations
        if mod(n, 1000) == 0
            println("Iteration $(n)/$(n_iterations)")
        end

        # Sample momenta according to N(0,D(q)^-1)
        p = sample_momenta_RMHMC(q, inv_D_sqrt)

        # One step of the Hamiltonian dynamics with momentum reversal and S-reversibility check
        new_q, new_p = psi_rev(q, p, numerical_scheme, Δt, n_newton, eta_newton, eta_newton_tilde, eta_rev, ∇V, ∇2V, D, ∇D, ∇2D, inv_D)

        # Metropolis-Hastings accept/reject procedure
        if log(rand()) <= (H(q, p, V, D, ln_det_D) - H(new_q, new_p, V, D, ln_det_D))
            q = copy(new_q)
            p = copy(new_p)
        end

        # Set Markov Chain state
        mc_trajectory[n+1, :] = [q..., p...]
    end

    return mc_trajectory
end

################ Sampling momenta for RMGHMC ################

function sample_momenta_RMGHMC(
    q, p,
    Δt,
    γ,
    D::Function,
    inv_D::Function
)::Vector{Float64}

    α_q = exp(-γ * D(q) * Δt)

    return α_q * p + sqrt((Matrix{Float64}(I,length(q),length(q)) - α_q^2) * inv_D(q)) * randn(length(q))
end

################ RMGHMC algorithm ################

function RMGHMC(
    q0, p0,
    n_iterations,
    numerical_scheme,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    eta_rev,
    V::Function,
    ∇V::Function,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function,
    ln_det_D::Function,
)

    d = 2 * length(q0)
    mc_trajectory = Array{Float64}(undef, n_iterations + 1, d)
    mc_trajectory[1, :] = [q0..., p0...]
    q = copy(q0)
    p = copy(p0)

    for n in 1:n_iterations
        if mod(n, 1000) == 0
            println("Iteration $(n)/$(n_iterations)")
        end

        # Evolve momenta by integrating the fluctuation-dissipation part of the Langevin dynamics with time step Δt / 2
        p = sample_momenta_RMGHMC(q, p, Δt / 2, γ, D, inv_D)

        # One step of the Hamiltonian dynamics with momentum reversal and S-reversibility check
        new_q, new_p = psi_rev(q, p, numerical_scheme, Δt, n_newton, eta_newton, eta_newton_tilde, eta_rev, ∇V, ∇2V, D, ∇D, ∇2D, inv_D)

        # Metropolis-Hastings accept/reject procedure
        if log(rand()) <= (H(q, p, V, D, ln_det_D) - H(new_q, new_p, V, D, ln_det_D))
            q = copy(new_q)
            p = copy(new_p)
        end

        # Reverse momenta
        p = -p

        # Evolve momenta by integrating the fluctuation-dissipation part of the Langevin dynamics with time step Δt / 2
        p = sample_momenta_RMGHMC(q, p, Δt / 2, γ, D, inv_D)

        # Set Markov Chain state
        mc_trajectory[n+1, :] = [q..., p...]
    end

    return mc_trajectory
end

################ psi_rev implementation ################

function psi_rev(
    q, p,
    numerical_scheme,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    eta_rev,
    ∇V::Function,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    if numerical_scheme == "GSV"
        integrator = varphi_GSV
    elseif numerical_scheme == "IMR"
        integrator = varphi_IMR
    else
        error("Choose an integrator betwen GSV and IMR")
    end

    tilde_q = copy(q)
    tilde_p = copy(p)
    try
        tilde_q, tilde_p = integrator(q, p, Δt, n_newton, eta_newton, eta_newton_tilde, ∇V, ∇2V, D, ∇D, ∇2D, inv_D)
    catch e
        if isa(e, NonConvergenceError) || isa(e, NonInvertibilityError)
            return [q, p]
        end
    end

    tilde_tilde_q = copy(q)
    tilde_tilde_p = copy(p)
    try
        tilde_tilde_q, tilde_tilde_p = integrator(tilde_q, -tilde_p, Δt, n_newton, eta_newton, eta_newton_tilde, ∇V, ∇2V, D, ∇D, ∇2D, inv_D)
    catch e
        if isa(e, NonConvergenceError) || isa(e, NonInvertibilityError)
            return [q, p]
        end
    end

    if norm([(tilde_tilde_q .- q)..., (-tilde_tilde_p .- p)...]) < eta_rev * norm([q..., p...])
        return [tilde_q, -tilde_p]
    else
        return [q, p]
    end

end