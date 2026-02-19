"""Sliding Frank-Wolfe (SFW) for gridless sparse spike recovery.

Standard mode:
- Positive BLASSO amplitudes (Poisson NLL + lam * sum(a), a >= 0)
- Joint local LM refinement of amplitudes and positions
- Optional lam continuation schedule (high -> low) with warm starts
"""

import mlx.core as mx
import numpy as np

from .model import (
    EPS,
    build_basis,
    compute_certificate,
    find_new_spike,
    forward_model,
    refine_new_spike,
)
from .solvers import fista_amplitudes, joint_lm_refine
from .utils import prune


def sfw_poisson(
    y,
    sigma,
    bg,
    lam=0.15,
    lam_schedule=None,
    n_iter=50,
    fista_iter=20,
    n_refine=2,
    source_refine_steps=25,
    source_refine_step=1.0,
    pos_bound=1.0,
    joint_lm_iter=6,
    joint_lm_sigma_bound=0.0,
    joint_lm_amp_upper=1e5,
    joint_lm_cg_iter=20,
    prune_tol=1e-4,
    cert_tol=1e-3,
    verbose=True,
):
    """Sliding Frank-Wolfe with Poisson noise model.

    Parameters
    ----------
    y : (H, W) observed image (MLX array)
    sigma : PSF width
    bg : background level (scalar)
    lam : positive BLASSO/LASSO weight (ignored when lam_schedule is provided)
    lam_schedule : optional continuation schedule (list/tuple), high -> low
    n_iter : max number of outer iterations across all continuation stages
    fista_iter : max FISTA iterations per local amplitude subproblem
    n_refine : local correction sweeps per outer iteration
    source_refine_steps : local-search steps for continuous source selection
    source_refine_step : initial step size for source-selection local search
    pos_bound : position box half-width for local LM updates (pixels)
    joint_lm_iter : LM outer iterations in each local joint refinement
    joint_lm_sigma_bound : sigma box half-width (0 keeps sigma fixed)
    joint_lm_amp_upper : amplitude upper bound used by joint LM
    joint_lm_cg_iter : CG iterations inside each LM step
    prune_tol : amplitude pruning threshold after each outer iteration
    cert_tol : certificate tolerance, stop when (-min eta / lam) <= 1 + cert_tol

    Returns
    -------
    amplitudes : (K,)
    positions : (K, 2)  — row, col
    energies : list of mean Poisson deviance
    """
    if lam_schedule is None:
        if lam is None or lam <= 0.0:
            raise ValueError("lam must be > 0 when lam_schedule is not provided.")
        lam_values = [float(lam)]
    else:
        lam_values = [float(v) for v in lam_schedule]
        if len(lam_values) == 0:
            raise ValueError("lam_schedule must contain at least one value.")
        if any(v <= 0.0 for v in lam_values):
            raise ValueError("All lam_schedule values must be > 0.")
        if verbose and any(
            lam_values[i + 1] > lam_values[i] for i in range(len(lam_values) - 1)
        ):
            print("  Note: lam_schedule is usually provided high-to-low.")

    H, W = y.shape
    gy = mx.arange(H, dtype=mx.float32)
    gx = mx.arange(W, dtype=mx.float32)
    grid_y, grid_x = mx.meshgrid(gy, gx, indexing="ij")

    amplitudes = mx.array([], dtype=mx.float32)
    positions = mx.zeros((0, 2), dtype=mx.float32)

    if len(lam_values) > n_iter:
        lam_values = lam_values[:n_iter]
    n_stages = len(lam_values)
    base = n_iter // n_stages
    rem = n_iter % n_stages
    stage_iters = [base + (1 if s < rem else 0) for s in range(n_stages)]

    energies = []
    finished = False
    global_it = 0

    for stage_idx, lam_stage in enumerate(lam_values):
        if verbose and n_stages > 1:
            print(f"  stage {stage_idx+1}/{n_stages} | lam={lam_stage:.6g}")

        for _ in range(stage_iters[stage_idx]):
            if amplitudes.size == 0:
                model_img = mx.full(y.shape, bg, dtype=mx.float32)
            else:
                model_img = forward_model(amplitudes, positions, grid_y, grid_x, sigma, bg)
            mx.eval(model_img)

            eta = compute_certificate(y, model_img, sigma, grid_y, grid_x)
            coarse_pos, _ = find_new_spike(eta)
            new_pos, min_eta = refine_new_spike(
                coarse_pos,
                y,
                model_img,
                sigma,
                grid_y,
                grid_x,
                n_steps=source_refine_steps,
                step0=source_refine_step,
            )
            mx.eval(min_eta)

            N = y.shape[0] * y.shape[1]
            cert_ratio = -min_eta / lam_stage
            mx.eval(cert_ratio)

            deviance = 2.0 * mx.sum(
                model_img - y - y * mx.log((model_img + EPS) / (y + EPS))
            )
            mean_dev = deviance / N
            mx.eval(mean_dev)

            if verbose:
                K = amplitudes.size
                print(
                    f"  iter {global_it:3d} | K={K:3d} | min(eta)={min_eta.item():.4f} "
                    f"| mean deviance={mean_dev.item():.4f} "
                    f"| cert ratio={cert_ratio.item():.4f}"
                )
            energies.append(mean_dev.item())
            global_it += 1

            if cert_ratio.item() <= (1.0 + cert_tol):
                if verbose:
                    print("  Certificate condition reached.")
                if stage_idx == n_stages - 1:
                    finished = True
                break

            if amplitudes.size == 0:
                amplitudes = mx.array([1.0], dtype=mx.float32)
                positions = new_pos
            else:
                amplitudes = mx.concatenate([amplitudes, mx.array([1.0], dtype=mx.float32)])
                positions = mx.concatenate([positions, new_pos], axis=0)
            mx.eval(amplitudes, positions)

            pos_before = positions
            for _ in range(n_refine):
                y_flat = y.reshape(-1)
                H_basis = build_basis(positions, grid_y, grid_x, sigma)
                mx.eval(H_basis)

                amplitudes = fista_amplitudes(
                    amplitudes,
                    H_basis,
                    y_flat,
                    bg,
                    lam=lam_stage,
                    n_iter=fista_iter,
                )
                mx.eval(amplitudes)

                amplitudes, positions = prune(amplitudes, positions, tol=prune_tol)
                mx.eval(amplitudes, positions)
                if amplitudes.size == 0:
                    break

                amplitudes, positions, _ = joint_lm_refine(
                    amplitudes,
                    positions,
                    sigma,
                    y,
                    grid_y,
                    grid_x,
                    bg,
                    pos_bound=pos_bound,
                    sigma_bound=joint_lm_sigma_bound,
                    amp_upper=float(joint_lm_amp_upper),
                    n_outer=int(joint_lm_iter),
                    n_cg=int(joint_lm_cg_iter),
                )
                mx.eval(amplitudes, positions)

            if verbose and pos_before.shape[0] > 0 and amplitudes.size > 0:
                shifts = mx.sqrt(mx.sum((positions - pos_before) ** 2, axis=1))
                mx.eval(shifts)
                shifts_np = np.array(shifts)
                print(
                    f"    LM shift — mean: {shifts_np.mean():.3f}, max: {shifts_np.max():.3f} px"
                )

            if amplitudes.size == 0:
                if verbose:
                    print("  All spikes pruned — stopping.")
                finished = True
                break

        if finished:
            break

    return amplitudes, positions, energies
