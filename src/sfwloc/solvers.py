"""Optimization primitives: MCP proximal, FISTA, conjugate gradient, and Levenberg-Marquardt."""

import mlx.core as mx

from .model import EPS, forward_model, poisson_nll


# ── MCP proximal helpers ──────────────────────────────────────────────────────


def mcp_penalty(a, lam, gamma):
    """MCP penalty value (vectorized, for a ≥ 0).

    p(a; λ, γ) = λa − a²/(2γ)   for 0 ≤ a ≤ γλ
               = γλ²/2           for a > γλ
    """
    p_inner = lam * a - a * a / (2.0 * gamma)
    p_cap = 0.5 * gamma * lam * lam
    return mx.where(a <= gamma * lam, p_inner, p_cap)


def prox_mcp_nonneg(u, lam, gamma, step):
    """Proximal operator for step*MCP with non-negativity (element-wise).

    Closed-form: argmin_a { (a−u)²/(2t) + MCP(a; λ, γ) }, a ≥ 0

        region 1  u ≤ t·λ         →  a = 0
        region 2  t·λ < u ≤ γ·λ   →  a = (u − t·λ) / (1 − t/γ)
        region 3  u > γ·λ          →  a = u   (no shrinkage)

    Requires step < gamma (always true for typical step sizes).
    """
    a_inner = (u - step * lam) / (1.0 - step / gamma)
    a = mx.where(
        u > gamma * lam,
        u,
        mx.where(u > step * lam, a_inner, mx.zeros_like(u)),
    )
    return mx.maximum(a, 0.0)


# ── FISTA amplitude solver ────────────────────────────────────────────────────


def fista_amplitudes(a0, H_basis, y_flat, bg, lam, gamma=3.0, n_iter=50):
    """FISTA with backtracking for Poisson NLL, non-negative MCP proximal.

    Parameters
    ----------
    a0 : (K,) initial amplitudes
    H_basis : (N, K) precomputed basis
    y_flat : (N,) observed data (flattened)
    bg : scalar background
    lam : regularisation weight
    gamma : MCP shape parameter γ > 1 (default 3.0)
    n_iter : int

    Returns
    -------
    (K,) optimised amplitudes
    """
    a = a0
    z = a0  # momentum variable
    t = 1.0

    def grad_smooth(a_):
        model = H_basis @ a_ + bg
        residual = 1.0 - y_flat / (model + EPS)  # (N,)
        return H_basis.T @ residual

    step = 1.0  # initial step size

    for _ in range(n_iter):
        g = grad_smooth(z)
        fz_model = H_basis @ z + bg
        fz = poisson_nll(fz_model, y_flat)

        # Backtracking line search
        for _bt in range(20):
            a_cand = prox_mcp_nonneg(z - step * g, lam, gamma, step)
            fa_model = H_basis @ a_cand + bg
            fa = poisson_nll(fa_model, y_flat)
            diff = a_cand - z
            quad_bound = fz + mx.sum(g * diff) + 0.5 / step * mx.sum(diff * diff)
            mx.eval(fa, quad_bound)
            if fa.item() <= quad_bound.item() + 1e-12:
                break
            step *= 0.5

        a_new = a_cand
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = a_new + ((t - 1.0) / t_new) * (a_new - a)

        a = a_new
        t = t_new
        step = min(step * 1.5, 1.0)  # cautiously grow step
        mx.eval(a, z)

    return a


# ── Conjugate gradient solver ─────────────────────────────────────────────────


def cg_solve(matvec, rhs, n_iter=20, tol=1e-6):
    """Conjugate gradient for Ax = b with A given by matvec."""
    x = mx.zeros_like(rhs)
    r = rhs  # r = b - A@0 = b
    p = r
    rr = mx.sum(r * r)
    mx.eval(x, r, p, rr)

    for _ in range(n_iter):
        Ap = matvec(p)
        pAp = mx.sum(p * Ap)
        alpha = rr / (pAp + EPS)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = mx.sum(r * r)
        mx.eval(rr_new)
        if rr_new.item() < tol * tol:
            mx.eval(x)
            break
        beta = rr_new / (rr + EPS)
        p = r + beta * p
        rr = rr_new
        mx.eval(x, r, p, rr)

    return x


# ── Coleman-Li scaling ────────────────────────────────────────────────────────


def coleman_li_scaling(params, grad, lower, upper):
    """Branchless Coleman-Li diagonal scaling for box constraints.

    Returns a vector v where v_i reflects distance to the active bound,
    slowing steps near boundaries.
    """
    v = mx.where(
        grad > 0,
        params - lower,
        mx.where(grad < 0, upper - params, mx.minimum(params - lower, upper - params)),
    )
    return mx.maximum(v, 1e-10)


# ── Levenberg-Marquardt solvers (block-coordinate) ────────────────────────────


def lm_amplitudes(
    amplitudes,
    positions,
    y,
    grid_y,
    grid_x,
    sigma,
    bg,
    amp_upper=1e5,
    n_outer=10,
    n_cg=15,
):
    """Refine amplitudes via bounded Levenberg-Marquardt with Coleman-Li scaling.

    Positions and sigma are held fixed. Only amplitudes (K params) are
    optimised with bounds [0, amp_upper].
    """
    K = amplitudes.shape[0]
    if K == 0:
        return amplitudes

    a = amplitudes

    # Bounds
    lower = mx.zeros(K)
    upper = mx.full((K,), amp_upper)
    mx.eval(lower, upper)

    mu = 1.0

    def nll_of_amp(a_):
        model = forward_model(a_, positions, grid_y, grid_x, sigma, bg)
        return poisson_nll(model, y)

    def model_of_amp(a_):
        return forward_model(a_, positions, grid_y, grid_x, sigma, bg)

    for _ in range(n_outer):
        loss, grad = mx.value_and_grad(nll_of_amp)(a)
        mx.eval(loss, grad)

        # Coleman-Li scaling
        v = coleman_li_scaling(a, grad, lower, upper)
        D = mx.sqrt(v)
        mx.eval(D)

        # Fisher weights
        model_img = model_of_amp(a)
        inv_model = 1.0 / (model_img + EPS)
        mx.eval(model_img, inv_model)

        # CG in scaled space: (D J^T W J D + μI) δ̂ = -D g
        scaled_grad = D * grad

        def scaled_matvec(d_hat):
            d = D * d_hat
            _, Jd = mx.jvp(model_of_amp, [a], [d])
            Jd_img = Jd[0]
            weighted = inv_model * Jd_img
            _, JTw_list = mx.vjp(model_of_amp, [a], [weighted])
            JTw = JTw_list[0]
            return D * JTw + mu * d_hat

        step_hat = cg_solve(scaled_matvec, -scaled_grad, n_iter=n_cg)
        step = D * step_hat
        mx.eval(step)

        # Clamp trial point to bounds
        a_new = mx.clip(a + step, lower, upper)
        loss_new = nll_of_amp(a_new)
        mx.eval(loss_new)

        if loss_new.item() < loss.item():
            a = a_new
            mu = max(mu * 0.5, 1e-8)
        else:
            mu = min(mu * 4.0, 1e8)

        mx.eval(a)

    return a


def lm_positions(
    amplitudes,
    positions,
    y,
    grid_y,
    grid_x,
    sigma,
    bg,
    pos_bound=1.0,
    rep_strength=0.0,
    rep_sigma=None,
    n_outer=10,
    n_cg=15,
):
    """Refine positions via bounded Levenberg-Marquardt with Coleman-Li scaling.

    Amplitudes and sigma are held fixed. Only positions (2K params) are
    optimised, giving a smaller, better-conditioned system than the joint
    approach.

    rep_strength : Gaussian repulsion coefficient between emitters (default 0).
        Adds rep_strength * Σ_{i<j} exp(−d²_{ij} / (2 rep_sigma²)) to the
        objective, pushing nearby atoms apart.  Useful for crowded scenes to
        prevent two atoms from collapsing to the same position.
    rep_sigma : width of the repulsion kernel in pixels (default: PSF sigma).
    """
    K = positions.shape[0]
    if K == 0:
        return positions

    _rep_sigma = rep_sigma if rep_sigma is not None else sigma

    pos = positions
    pos_flat = pos.reshape(-1)

    # Bounds: ±pos_bound from initial positions
    lower = pos_flat - pos_bound
    upper = pos_flat + pos_bound
    mx.eval(lower, upper)

    mu = 1.0

    def nll_of_pos(p):
        model = forward_model(amplitudes, p, grid_y, grid_x, sigma, bg)
        nll = poisson_nll(model, y)
        if rep_strength > 0.0 and K > 1:
            # Gaussian pairwise repulsion: Σ_{i<j} exp(−d²/(2 rep_sigma²))
            diff = p[:, None, :] - p[None, :, :]  # (K, K, 2)
            d2 = mx.sum(diff * diff, axis=-1)  # (K, K)
            # 0.5 * (sum_all - K diagonal) to count each pair once
            rep = 0.5 * rep_strength * (
                mx.sum(mx.exp(-d2 / (2.0 * _rep_sigma * _rep_sigma))) - K
            )
            nll = nll + rep
        return nll

    def model_of_pos(p):
        return forward_model(amplitudes, p, grid_y, grid_x, sigma, bg)

    for _ in range(n_outer):
        pf = pos.reshape(-1)
        loss, grad_pos = mx.value_and_grad(nll_of_pos)(pos)
        grad_flat = grad_pos.reshape(-1)
        mx.eval(loss, grad_flat)

        # Coleman-Li scaling
        v = coleman_li_scaling(pf, grad_flat, lower, upper)
        D = mx.sqrt(v)
        mx.eval(D)

        # Fisher weights
        model_img = model_of_pos(pos)
        inv_model = 1.0 / (model_img + EPS)
        mx.eval(model_img, inv_model)

        # CG in scaled space
        scaled_grad = D * grad_flat

        def scaled_matvec(d_hat):
            d = D * d_hat
            d_shaped = d.reshape(pos.shape)
            _, Jd = mx.jvp(model_of_pos, [pos], [d_shaped])
            Jd_img = Jd[0]
            weighted = inv_model * Jd_img
            _, JTw_list = mx.vjp(model_of_pos, [pos], [weighted])
            JTw = JTw_list[0].reshape(-1)
            return D * JTw + mu * d_hat

        step_hat = cg_solve(scaled_matvec, -scaled_grad, n_iter=n_cg)
        step = D * step_hat
        mx.eval(step)

        # Clamp trial point to bounds
        pf_new = mx.clip(pf + step, lower, upper)
        pos_new = pf_new.reshape(pos.shape)
        loss_new = nll_of_pos(pos_new)
        mx.eval(loss_new)

        if loss_new.item() < loss.item():
            pos = pos_new
            mu = max(mu * 0.5, 1e-8)
        else:
            mu = min(mu * 4.0, 1e8)

        mx.eval(pos)

    return pos


# ── Joint bounded LM (positions + amplitudes + sigma) ────────────────────────


def _pack_params(positions, amplitudes, sigma):
    """Pack (positions, amplitudes, sigma) into flat vector [y1,x1,...,yK,xK,a1,...,aK,sigma]."""
    return mx.concatenate([positions.reshape(-1), amplitudes, mx.array([sigma])])


def _unpack_params(theta, K):
    """Unpack flat vector into (positions (K,2), amplitudes (K,), sigma scalar)."""
    positions = theta[: 2 * K].reshape(K, 2)
    amplitudes = theta[2 * K : 3 * K]
    sigma = theta[3 * K]
    return positions, amplitudes, sigma


def joint_lm_refine(
    amplitudes,
    positions,
    sigma,
    y,
    grid_y,
    grid_x,
    bg,
    pos_bound=2.0,
    sigma_bound=0.2,
    amp_upper=1e5,
    n_outer=15,
    n_cg=20,
    a_expected=None,
    sigma_a=None,
):
    """Joint bounded Levenberg-Marquardt over positions, amplitudes, and sigma.

    Minimises the Poisson NLL, optionally with a Gaussian amplitude prior:
        F(θ) = Poisson_NLL(model, y) + (1/(2σₐ²))·Σ(aₖ - a_expected)²

    When a_expected and sigma_a are both provided, the prior is active.
    Otherwise, pure NLL refinement.

    Box constraints (via Coleman-Li scaling) keep parameters in range.
    """
    K = amplitudes.shape[0]
    if K == 0:
        return amplitudes, positions, sigma

    use_prior = a_expected is not None and sigma_a is not None

    theta = _pack_params(positions, amplitudes, sigma)
    mx.eval(theta)

    # Build bounds
    pos_flat = positions.reshape(-1)
    lower_pos = pos_flat - pos_bound
    upper_pos = pos_flat + pos_bound
    lower_amp = mx.zeros(K)
    upper_amp = mx.full((K,), amp_upper)
    lower_sig = mx.array([sigma - sigma_bound])
    upper_sig = mx.array([sigma + sigma_bound])
    lower = mx.concatenate([lower_pos, lower_amp, lower_sig])
    upper = mx.concatenate([upper_pos, upper_amp, upper_sig])
    mx.eval(lower, upper)

    if use_prior:
        inv_sigma_a2 = 1.0 / (sigma_a * sigma_a)

    def objective(th):
        pos, amp, sig = _unpack_params(th, K)
        model = forward_model(amp, pos, grid_y, grid_x, sig, bg)
        nll = poisson_nll(model, y)
        if use_prior:
            nll = nll + 0.5 * inv_sigma_a2 * mx.sum((amp - a_expected) ** 2)
        return nll

    def model_of(th):
        pos, amp, sig = _unpack_params(th, K)
        return forward_model(amp, pos, grid_y, grid_x, sig, bg)

    mu = 1.0

    for _ in range(n_outer):
        loss, grad = mx.value_and_grad(objective)(theta)
        mx.eval(loss, grad)

        # Coleman-Li scaling: D = diag(sqrt(v))
        v = coleman_li_scaling(theta, grad, lower, upper)
        D = mx.sqrt(v)
        mx.eval(D)

        # Current model for Fisher weights
        model_img = model_of(theta)
        inv_model = 1.0 / (model_img + EPS)
        mx.eval(model_img, inv_model)

        # CG in scaled space: (D J^T W J D + μ I) δ̂ = -D g
        # then unscale: δ = D δ̂
        scaled_grad = D * grad

        def scaled_matvec(d_hat):
            d = D * d_hat  # unscale direction
            # J d via forward-mode
            _, Jd = mx.jvp(model_of, [theta], [d])
            Jd_img = Jd[0]
            # Fisher weighting
            weighted = inv_model * Jd_img
            # J^T weighted via reverse-mode
            _, JTw_list = mx.vjp(model_of, [theta], [weighted])
            JTw = JTw_list[0]
            return D * JTw + mu * d_hat  # scale back + isotropic damping

        step_hat = cg_solve(scaled_matvec, -scaled_grad, n_iter=n_cg)
        step = D * step_hat  # unscale to original space
        mx.eval(step)

        # Clamp trial point to bounds
        theta_new = mx.clip(theta + step, lower, upper)
        loss_new = objective(theta_new)
        mx.eval(loss_new)

        if loss_new.item() < loss.item():
            theta = theta_new
            mu = max(mu * 0.5, 1e-8)
        else:
            mu = min(mu * 4.0, 1e8)

        mx.eval(theta)

    positions_out, amplitudes_out, sigma_out = _unpack_params(theta, K)
    mx.eval(positions_out, amplitudes_out, sigma_out)
    return amplitudes_out, positions_out, sigma_out.item()
