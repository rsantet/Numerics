using Cubature, Plots
include("main.jl")

Δt = 0.1
γ = 1.0
n_iterations = 100000
numerical_scheme = "IMR"
n_newton = 100
eta_newton = 1e-8
eta_newton_tilde = 1e-8
eta_rev = 1e-6
q0 = 0.
p0 = 0.

function V(q)
    return cos(2 * π * q)
end
function ∇V(q)
    return -2 * π * sin(2 * π * q)
end
function ∇2V(q)
    return -4 * π^2 * cos(2 * π * q)
end
#= function D(q)
    return 2 + cos(2 * π * q)
end
function ∇D(q)
    return -2 * π * sin(2 * π * q)
end
function ∇2D(q)
    return -4 * π^2 * cos(2 * π * q)
end =#
function D(q)
    return 1
end
function ∇D(q)
    return 0
end
function ∇2D(q)
    return 0
end
function inv_D(q)
    return inv(D(q))
end
function inv_D_sqrt(q)
    return sqrt(inv_D(q))
end
function inv_D_2(q)
    return inv_D(q)^2
end
function ln_det_D(q)
    return logdet(D(q))
end
function mu(q)
    return exp(-V(q))
end
Z = hquadrature(mu, 0,1)[1]
function pi_inv(q)
    return mu(q) / Z
end

function test_hmc()
    
    mc_traj = RMHMC(
        q0, p0,
        n_iterations,
        numerical_scheme,
        Δt,
        n_newton,
        eta_newton,
        eta_newton_tilde,
        eta_rev,
        V,
        ∇V,
        ∇2V,
        D,
        ∇D,
        ∇2D,
        inv_D,
        inv_D_2,
        inv_D_sqrt,
        ln_det_D,
    )

    mc_traj_q = mc_traj[:,1]
    X = LinRange(0,1,1000)
    histogram(
        mod.(mc_traj_q,1),
        normalize=:pdf,
        label="Histogram"
    )
    plot!(
        X,
        pi_inv.(X),
        label="Target distribution",
        linewidth=3
    )
    mkpath("./img/")
    savefig("./img/rmhmc_$(numerical_scheme)_histogram.png")
    plot(
        0:n_iterations,
        mc_traj_q,
        label="Trajectory"
    )
    savefig("./img/rmhmc_$(numerical_scheme)_trajectory.png")
end

function test_rmghmc()
    mc_traj = RMGHMC(
        q0, p0,
        n_iterations,
        numerical_scheme,
        Δt,
        n_newton,
        eta_newton,
        eta_newton_tilde,
        eta_rev,
        V,
        ∇V,
        ∇2V,
        D,
        ∇D,
        ∇2D,
        inv_D,
        inv_D_2,
        ln_det_D,
    )

    mc_traj_q = mc_traj[:,1]
    X = LinRange(0,1,1000)
    histogram(
        mod.(mc_traj_q,1),
        normalize=:pdf,
        label="Histogram"
    )
    plot!(
        X,
        pi_inv.(X),
        label="Target distribution",
        linewidth=3
    )

    mkpath("./img/")
    savefig("./img/rmghmc_$(numerical_scheme)_histogram.png")
    plot(
        0:n_iterations,
        mc_traj_q,
        label="Trajectory"
    )
    savefig("./img/rmghmc_$(numerical_scheme)_trajectory.png")
end

test_hmc()
test_rmghmc()