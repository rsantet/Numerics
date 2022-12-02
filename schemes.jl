################ GSV numerical scheme ################

function phi_euler_B(
    q, p, q1, p1,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)
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
)
    return [
        1 -Δt*∇p2H(q, p1, D)
        0 1+Δt*∇pqH(q, p1, ∇D)
    ]
end

function phi_euler_A(
    q, p, q1, p1,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)
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
    inv_D::Function,
    inv_D_2::Function
)
    return [
        1-Δt*∇pqH(q1, p, ∇D) 0
        Δt*∇q2H(q1, p, ∇2V, ∇D, ∇2D, inv_D, inv_D_2) 1
    ]
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
)
    q1, p1 = explicit_euler_euler_B(q, p, Δt, ∇V, D, ∇D, inv_D)
    phi_0 = phi_euler_B(q, p, q1, p1, Δt, ∇V, D, ∇D, inv_D)
    phi_check = norm(phi_0) * eta_newton
    phi_cur = phi_0
    q_cur = q1
    p_cur = p1

    for i in 0:n_newton-1
        gradient = ∇phi_euler_B(q, p, q_cur, p_cur, Δt, D, ∇D)
        if rank(gradient) != size(gradient)[1]
            throw(NonInvertibilityError())
        end
        # solve linear system (∇Phi)(y^{n+1}-y^n)=-Phi
        sol = gradient \ (-phi_cur)
        # update newton iterate
        q_cur += sol[1]
        p_cur += sol[2]

        phi_cur = phi_euler_B(q, p, q_cur, p_cur, Δt, ∇V, D, ∇D, inv_D)
        if norm(phi_cur) < phi_check || norm([q_cur - q1, p_cur - p1]) < eta_newton_tilde * norm([q1, p1])
            return q_cur, p_cur
        else
            q1 = q_cur
            p1 = p_cur
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
)
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
    inv_D::Function,
    inv_D_2::Function
)
    q1, p1 = explicit_euler_euler_A(q, p, Δt, ∇V, D, ∇D, inv_D)
    phi_0 = phi_euler_A(q, p, q1, p1, Δt, ∇V, D, ∇D, inv_D)
    phi_check = norm(phi_0) * eta_newton
    phi_cur = phi_0
    q_cur = q1
    p_cur = p1

    for i in 0:n_newton-1
        gradient = ∇phi_euler_A(q, p, q_cur, p_cur, Δt, ∇2V, ∇D, ∇2D, inv_D, inv_D_2)
        if rank(gradient) != size(gradient)[1]
            throw(NonInvertibilityError())
        end
        # solve linear system (∇Phi)(y^{n+1}-y^n)=-Phi
        sol = gradient \ (-phi_cur)
        # update newton iterate
        q_cur += sol[1]
        p_cur += sol[2]

        phi_cur = phi_euler_A(q, p, q_cur, p_cur, Δt, ∇V, D, ∇D, inv_D)
        if norm(phi_cur) < phi_check || norm([q_cur - q1, p_cur - p1]) < eta_newton_tilde * norm([q1, p1])
            return q_cur, p_cur
        else
            q1 = q_cur
            p1 = p_cur
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
)
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
    inv_D::Function,
    inv_D_2::Function
)

    # Euler B with time step Δt/2
    q1, p1 = varphi_euler_B(q, p, Δt / 2, n_newton, eta_newton, eta_newton_tilde, ∇V, D, ∇D, inv_D)

    # Euler A with time step Δt/2
    q2, p2 = varphi_euler_A(q1, p1, Δt / 2, n_newton, eta_newton, eta_newton_tilde, ∇V, ∇2V, D, ∇D, ∇2D, inv_D, inv_D_2)

    return q2, p2
end

################ IMR numerical scheme ################

function phi_IMR(
    q, p, q1, p1,
    Δt,
    ∇V::Function,
    D::Function,
    ∇D::Function,
    inv_D::Function
)
    q_mid = (q + q1) / 2
    p_mid = (p + p1) / 2
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
    inv_D::Function,
    inv_D_2::Function
)
    q_mid = (q + q1) / 2
    p_mid = (p + p1) / 2
    return [
        1-Δt*∇pqH(q_mid, p_mid, ∇D)/2 -Δt*∇p2H(q_mid, p_mid, D)/2
        Δt*∇q2H(q_mid, p_mid, ∇2V, ∇D, ∇2D, inv_D, inv_D_2)/2 1+Δt*∇pqH(q_mid, p_mid, ∇D)/2
    ]
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
    inv_D::Function,
    inv_D_2::Function,
)
    q1, p1 = explicit_euler_IMR(q, p, Δt, ∇V, D, ∇D, inv_D)
    phi_0 = phi_IMR(q, p, q1, p1, Δt, ∇V, D, ∇D, inv_D)
    phi_check = norm(phi_0) * eta_newton
    phi_cur = phi_0
    q_cur = q1
    p_cur = p1

    for i in 0:n_newton-1
        gradient = ∇phi_IMR(q, p, q_cur, p_cur, Δt, ∇2V, D, ∇D, ∇2D, inv_D, inv_D_2)
        if rank(gradient) != size(gradient)[1]
            throw(NonInvertibilityError())
        end
        # solve linear system (∇Phi)(y^{n+1}-y^n)=-Phi
        sol = gradient \ (-phi_cur)
        # update newton iterate
        q_cur += sol[1]
        p_cur += sol[2]

        phi_cur = phi_IMR(q, p, q_cur, p_cur, Δt, ∇V, D, ∇D, inv_D)
        if norm(phi_cur) < phi_check || norm([q_cur - q1, p_cur - p1]) < eta_newton_tilde * norm([q1, p1])
            return q_cur, p_cur
        else
            q1 = q_cur
            p1 = p_cur
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
)
    return [
        q + Δt * ∇pH(q, p, D),
        p - Δt * ∇qH(q, p, ∇V, ∇D, inv_D)
    ]
end