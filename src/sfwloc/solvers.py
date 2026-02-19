"""Optimization primitives for BLASSO + joint local refinement."""

import mlx.core as mx

from .model import EPS, forward_model, poisson_nll


def prox_nonneg_l1(u, thresh):
    """Prox of thresh * ||a||_1 + I_{a>=0}: positive soft-threshold."""
    if thresh <= 0.0:
        return mx.maximum(u, 0.0)
    return mx.maximum(u - thresh, 0.0)


def fista_amplitudes(a0, H_basis, y_flat, bg, lam, n_iter=20, rel_tol=1e-5):
    """Monotone FISTA for Poisson NLL + lam * sum(a), with a >= 0.

    Solves:
        min_a  PoissonNLL(Ha + bg, y) + lam * sum(a)
        s.t.   a >= 0
    """
    if lam is None or lam <= 0.0:
        raise ValueError("lam must be > 0.")

    a = mx.maximum(a0, 0.0)
    z = a
    t = 1.0
    step = 1.0

    def smooth_obj(a_):
        model = H_basis @ a_ + bg
        return poisson_nll(model, y_flat)

    def composite_obj(a_):
        return smooth_obj(a_) + lam * mx.sum(mx.maximum(a_, 0.0))

    prev_obj = composite_obj(a)
    mx.eval(prev_obj)

    for _ in range(n_iter):
        fz, g = mx.value_and_grad(smooth_obj)(z)
        mx.eval(fz, g)

        # Backtracking on composite majorizer.
        for _bt in range(20):
            a_new = prox_nonneg_l1(z - step * g, step * lam)
            f_new = smooth_obj(a_new)
            d = a_new - z
            quad = fz + mx.sum(g * d) + 0.5 / step * mx.sum(d * d)
            obj_new = f_new + lam * mx.sum(a_new)
            bound = quad + lam * mx.sum(a_new)
            mx.eval(obj_new, bound)
            if obj_new.item() <= bound.item() + 1e-12:
                break
            step *= 0.5

        obj_curr = composite_obj(a)
        mx.eval(obj_curr)

        # Monotone restart for stability.
        if obj_new.item() > obj_curr.item() + 1e-12:
            z = a
            t = 1.0
            step *= 0.5
            mx.eval(z)
            continue

        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = a_new + ((t - 1.0) / t_new) * (a_new - a)
        a = a_new
        t = t_new
        step = min(step * 1.5, 1.0)

        denom = abs(prev_obj.item()) + 1e-12
        rel_drop = abs(prev_obj.item() - obj_new.item()) / denom
        prev_obj = obj_new
        mx.eval(a, z, prev_obj)
        if rel_drop < rel_tol:
            break

    return a


def cg_solve(matvec, rhs, n_iter=20, tol=1e-6):
    """Conjugate gradient for Ax = b with A given by matvec."""
    x = mx.zeros_like(rhs)
    r = rhs
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


def coleman_li_scaling(params, grad, lower, upper):
    """Branchless Coleman-Li diagonal scaling for box constraints."""
    v = mx.where(
        grad > 0,
        params - lower,
        mx.where(
            grad < 0,
            upper - params,
            mx.minimum(params - lower, upper - params),
        ),
    )
    return mx.maximum(v, 1e-10)


def _pack_params(positions, amplitudes, sigma=None, include_sigma=True):
    if include_sigma:
        return mx.concatenate([positions.reshape(-1), amplitudes, mx.array([sigma])])
    return mx.concatenate([positions.reshape(-1), amplitudes])


def _unpack_params(theta, K, sigma_fixed=None, include_sigma=True):
    positions = theta[: 2 * K].reshape(K, 2)
    amplitudes = theta[2 * K : 3 * K]
    if include_sigma:
        sigma = theta[3 * K]
    else:
        sigma = mx.array(sigma_fixed, dtype=theta.dtype)
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
    sigma_bound=0.0,
    amp_upper=1e5,
    n_outer=6,
    n_cg=20,
):
    """Joint bounded LM over positions, amplitudes, and optionally sigma.

    When sigma_bound <= 0, sigma is kept fixed for a smaller/faster system.
    """
    K = amplitudes.shape[0]
    if K == 0:
        return amplitudes, positions, sigma

    optimize_sigma = sigma_bound > 0.0
    theta = _pack_params(positions, amplitudes, sigma=sigma, include_sigma=optimize_sigma)
    mx.eval(theta)

    pos_flat = positions.reshape(-1)
    lower_pos = pos_flat - pos_bound
    upper_pos = pos_flat + pos_bound
    lower_amp = mx.zeros(K)
    upper_amp = mx.full((K,), amp_upper)

    if optimize_sigma:
        lower_sig = mx.array([sigma - sigma_bound])
        upper_sig = mx.array([sigma + sigma_bound])
        lower = mx.concatenate([lower_pos, lower_amp, lower_sig])
        upper = mx.concatenate([upper_pos, upper_amp, upper_sig])
    else:
        lower = mx.concatenate([lower_pos, lower_amp])
        upper = mx.concatenate([upper_pos, upper_amp])
    mx.eval(lower, upper)

    def objective(th):
        pos, amp, sig = _unpack_params(
            th, K, sigma_fixed=sigma, include_sigma=optimize_sigma
        )
        model = forward_model(amp, pos, grid_y, grid_x, sig, bg)
        return poisson_nll(model, y)

    def model_of(th):
        pos, amp, sig = _unpack_params(
            th, K, sigma_fixed=sigma, include_sigma=optimize_sigma
        )
        return forward_model(amp, pos, grid_y, grid_x, sig, bg)

    mu = 1.0

    for _ in range(n_outer):
        loss, grad = mx.value_and_grad(objective)(theta)
        mx.eval(loss, grad)

        v = coleman_li_scaling(theta, grad, lower, upper)
        D = mx.sqrt(v)
        mx.eval(D)

        model_img = model_of(theta)
        inv_model = 1.0 / (model_img + EPS)
        mx.eval(model_img, inv_model)

        scaled_grad = D * grad

        def scaled_matvec(d_hat):
            d = D * d_hat
            _, Jd = mx.jvp(model_of, [theta], [d])
            Jd_img = Jd[0]
            weighted = inv_model * Jd_img
            _, JTw_list = mx.vjp(model_of, [theta], [weighted])
            JTw = JTw_list[0]
            return D * JTw + mu * d_hat

        step_hat = cg_solve(scaled_matvec, -scaled_grad, n_iter=n_cg)
        step = D * step_hat
        mx.eval(step)

        theta_new = mx.clip(theta + step, lower, upper)
        loss_new = objective(theta_new)
        mx.eval(loss_new)

        if loss_new.item() < loss.item():
            theta = theta_new
            mu = max(mu * 0.5, 1e-8)
        else:
            mu = min(mu * 4.0, 1e8)

        mx.eval(theta)

    positions_out, amplitudes_out, sigma_out = _unpack_params(
        theta, K, sigma_fixed=sigma, include_sigma=optimize_sigma
    )
    mx.eval(positions_out, amplitudes_out, sigma_out)
    return amplitudes_out, positions_out, sigma_out.item()
