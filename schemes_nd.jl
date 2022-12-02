using BlockArrays

################ GSV numerical scheme ################

function phi_euler_B(
    q, p, q1, p1,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    return [
        q1 - q - Δt * ∇pH(q, p1, D),
        p1 - p + Δt * ∇qH(q, p1, ∇V, ∇D, inv_D)
    ]

end

function ∇phi_euler_B(
    q, p, q1, p1,
    Δt,
    D::Function,
    ∇D::Function
)::Matrix{Float64}

    A = BlockArray{Float64}(undef, [length(q), length(q)], [length(q), length(q)])
    A[Block(1, 1)] = Matrix{Float64}(I, length(q), length(q))
    A[Block(1, 2)] = -Δt * ∇p2H(q, p1, D)
    A[Block(2, 1)] = zeros(length(q), length(q))
    A[Block(2, 2)] = Matrix{Float64}(I, length(q), length(q)) + Δt * ∇pqH(q, p1, ∇D)
    return Array(A)

end

function phi_euler_A(
    q, p, q1, p1,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    return [
        q1 - q - Δt * ∇pH(q1, p, D),
        p1 - p + Δt * ∇qH(q1, p, ∇V, ∇D, inv_D)
    ]

end

function ∇phi_euler_A(
    q, p, q1, p1,
    Δt,
    ∇2V::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Matrix{Float64}

    A = BlockArray{Float64}(undef, [length(q), length(q)], [length(q), length(q)])
    A[Block(1, 1)] = Matrix{Float64}(I, length(q), length(q)) - Δt * ∇pqH(q1, p, ∇D)
    A[Block(1, 2)] = zeros(length(q), length(q))
    A[Block(2, 1)] = Δt * ∇q2H(q1, p, ∇2V, ∇D, ∇2D, inv_D)
    A[Block(2, 2)] = Matrix{Float64}(I, length(q), length(q))
    return Array(A)

end

################ GSV numerical flow ################

function varphi_euler_B(
    q, p,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    q1, p1 = explicit_euler_euler_B(q, p, Δt, ∇V, D, ∇D, inv_D)
    phi_0_q, phi_0_p = phi_euler_B(q, p, q1, p1, Δt, ∇V, D, ∇D, inv_D)
    phi_cur = [phi_0_q..., phi_0_p...]
    phi_check = norm(phi_cur) * eta_newton
    q_cur = copy(q1)
    p_cur = copy(p1)

    for i in 0:n_newton-1
        gradient = ∇phi_euler_B(q, p, q_cur, p_cur, Δt, D, ∇D)
        if rank(gradient) != 2 * length(q)
            throw(NonInvertibilityError())
        end
        # solve linear system (∇Phi)(y^{n+1}-y^n)=-Phi
        sol = gradient \ (-phi_cur)
        # update newton iterate
        q_cur .+= sol[1:length(q)]
        p_cur .+= sol[length(q)+1:2*length(q)]

        phi_cur_q, phi_cur_p = phi_euler_B(q, p, q_cur, p_cur, Δt, ∇V, D, ∇D, inv_D)
        phi_cur = [phi_cur_q..., phi_cur_p...]
        if norm(phi_cur) < phi_check || norm([(q_cur .- q1)..., (p_cur .- p1)...]) < eta_newton_tilde * norm([q1..., p1...])
            return [q_cur, p_cur]
        else
            q1 = copy(q_cur)
            p1 = copy(p_cur)
        end
        if i == n_newton - 1
            throw(NonConvergenceError())
        end
    end

end

function explicit_euler_euler_B(
    q, p,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    return [
        q + Δt * ∇pH(q, p, D),
        p - Δt * ∇qH(q, p, ∇V, ∇D, inv_D)
    ]

end

function varphi_euler_A(
    q, p,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    ∇V::Function,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    q1, p1 = explicit_euler_euler_A(q, p, Δt, ∇V, D, ∇D, inv_D)
    phi_q_0, phi_p_0 = phi_euler_A(q, p, q1, p1, Δt, ∇V, D, ∇D, inv_D)
    phi_cur = [phi_q_0..., phi_p_0...]
    phi_check = norm(phi_cur) * eta_newton
    q_cur = copy(q1)
    p_cur = copy(p1)

    for i in 0:n_newton-1
        gradient = ∇phi_euler_A(q, p, q_cur, p_cur, Δt, ∇2V, ∇D, ∇2D, inv_D)
        if rank(gradient) != 2 * length(q)
            throw(NonInvertibilityError())
        end
        # solve linear system (∇Phi)(y^{n+1}-y^n)=-Phi
        sol = gradient \ (-phi_cur)
        # update newton iterate
        q_cur .+= sol[1:length(q)]
        p_cur .+= sol[length(q)+1:2*length(q)]

        phi_cur_q, phi_cur_p = phi_euler_A(q, p, q_cur, p_cur, Δt, ∇V, D, ∇D, inv_D)
        phi_cur = [phi_cur_q..., phi_cur_p...]
        if norm(phi_cur) < phi_check || norm([(q_cur .- q1)..., (p_cur .- p1)...]) < eta_newton_tilde * norm([q1..., p1...])
            return [q_cur, p_cur]
        else
            q1 = copy(q_cur)
            p1 = copy(p_cur)
        end
        if i == n_newton - 1
            throw(NonConvergenceError())
        end
    end
end

function explicit_euler_euler_A(
    q, p,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    return [
        q + Δt * ∇pH(q, p, D),
        p - Δt * ∇qH(q, p, ∇V, ∇D, inv_D)
    ]
end

function varphi_GSV(
    q, p,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    ∇V::Function,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    # Euler B with time step Δt/2
    q1, p1 = varphi_euler_B(q, p, Δt / 2, n_newton, eta_newton, eta_newton_tilde, ∇V, D, ∇D, inv_D)

    # Euler A with time step Δt/2
    q2, p2 = varphi_euler_A(q1, p1, Δt / 2, n_newton, eta_newton, eta_newton_tilde, ∇V, ∇2V, D, ∇D, ∇2D, inv_D)

    return [q2, p2]
end

################ IMR numerical scheme ################

function phi_IMR(
    q, p, q1, p1,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}

    q_mid = @. (q + q1) / 2
    p_mid = @. (p + p1) / 2

    return [
        q1 - q - Δt * ∇pH(q_mid, p_mid, D),
        p1 - p + Δt * ∇qH(q_mid, p_mid, ∇V, ∇D, inv_D)
    ]
end

function ∇phi_IMR(
    q, p, q1, p1,
    Δt,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Matrix{Float64}

    q_mid = @. (q + q1) / 2
    p_mid = @. (p + p1) / 2
    A = BlockArray{Float64}(undef, [length(q), length(q)], [length(q), length(q)])
    A[Block(1, 1)] = Matrix{Float64}(I, length(q), length(q)) - Δt * ∇pqH(q_mid, p_mid, ∇D) / 2
    A[Block(1, 2)] = -Δt * ∇p2H(q_mid, p_mid, D) / 2
    A[Block(2, 1)] = Δt * ∇q2H(q_mid, p_mid, ∇2V, ∇D, ∇2D, inv_D) / 2
    A[Block(2, 2)] = Matrix{Float64}(I, length(q), length(q)) + Δt * ∇pqH(q_mid, p_mid, ∇D) / 2

    return Array(A)
end

################ IMR numerical flow ################

function varphi_IMR(
    q, p,
    Δt,
    n_newton,
    eta_newton,
    eta_newton_tilde,
    ∇V::Function,
    ∇2V::Function,
    D::Function,
    ∇D::Function,
    ∇2D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}
    q1, p1 = explicit_euler_IMR(q, p, Δt, ∇V, D, ∇D, inv_D)
    phi_q_0, phi_p_0 = phi_IMR(q, p, q1, p1, Δt, ∇V, D, ∇D, inv_D)
    phi_cur = [phi_q_0..., phi_p_0...]
    phi_check = norm(phi_cur) * eta_newton
    q_cur = copy(q1)
    p_cur = copy(p1)

    for i in 0:n_newton-1
        gradient = ∇phi_IMR(q, p, q_cur, p_cur, Δt, ∇2V, D, ∇D, ∇2D, inv_D)
        if rank(gradient) != 2 * length(q)
            throw(NonInvertibilityError())
        end
        # solve linear system (∇Phi)(y^{n+1}-y^n)=-Phi
        sol = gradient \ (-phi_cur)
        # update newton iterate
        q_cur .+= sol[1:length(q)]
        p_cur .+= sol[length(q)+1:2*length(q)]

        phi_cur_q, phi_cur_p = phi_IMR(q, p, q_cur, p_cur, Δt, ∇V, D, ∇D, inv_D)
        phi_cur = [phi_cur_q..., phi_cur_p...]
        if norm(phi_cur) < phi_check || norm([(q_cur .- q1)..., (p_cur .- p1)...]) < eta_newton_tilde * norm([q1..., p1...])
            return [q_cur, p_cur]
        else
            q1 = copy(q_cur)
            p1 = copy(p_cur)
        end
        if i == n_newton - 1
            throw(NonConvergenceError())
        end
    end
end

function explicit_euler_IMR(
    q, p,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)::Vector{Vector{Float64}}
    return [
        q + Δt * ∇pH(q, p, D),
        p - Δt * ∇qH(q, p, ∇V, ∇D, inv_D)
    ]
end