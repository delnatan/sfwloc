"""Sliding Frank-Wolfe (SFW) for gridless sparse spike recovery.

Poisson noise model with joint bounded Levenberg-Marquardt refinement
(positions, amplitudes, sigma) using Coleman-Li scaling. Built on Apple MLX.

Reference: Denoyelle et al., "Sliding Frank-Wolfe Algorithm" (2019)

Overview
--------
The outer loop alternates three steps until the Frank-Wolfe certificate η
satisfies min(η) ≥ −1 everywhere (optimality condition):

  1. Certificate  — compute η = (1/λ) Φ*(1 − y/model); its minimum
                    identifies the pixel where adding a new emitter would
                    most improve the Poisson log-likelihood.
  2. FISTA        — update all amplitudes jointly with an MCP penalty
                    (non-convex, reduces L1 bias for bright emitters).
  3. LM refinement — alternate Levenberg-Marquardt steps on amplitudes
                    and positions (block-coordinate, Coleman-Li bounds).

Key parameters
--------------
lam       : sparsity weight.  Controls how negative η must be before a new
            emitter is accepted (threshold = −1 regardless of lam; lam
            scales the certificate).  Larger lam → fewer, brighter emitters.

S_min     : minimum signal in photons.  Sets *both* the MCP penalty
            threshold (γ = S_min/lam) and the hard prune cutoff.
            Rule of thumb: S_min ≈ 3–10 × per-pixel background level.
            Amplitudes above S_min are recovered without bias; those below
            are driven to zero and pruned.

Super-resolved (crowded) mode — amp_cap and rep_strength
---------------------------------------------------------
When emitters are separated by less than ~2 σ the standard solver merges
nearby pairs into a single over-bright spike.  Two complementary parameters
address this:

amp_cap     Amplitude ceiling applied after every FISTA step (photons).
            Limits how much flux any one emitter can absorb, leaving
            structured residual that keeps η < −1 near the second emitter
            so the certificate can propose it on the next iteration.
            The certificate mask radius is automatically tightened from
            1 σ to 0.5 σ when amp_cap is active, allowing proposals
            within one PSF width of an existing emitter.

            Rule of thumb: amp_cap ≈ 0.5 × A_max, where A_max is the
            brightest single-emitter amplitude you expect.  Setting it too
            high (≥ A_single) leaves too little residual; too low
            (< 0.3 × A_single) fragments bright isolated emitters.

rep_strength  Gaussian pairwise repulsion added to the LM position
            objective: R = Σ_{i<j} exp(−d²_{ij} / 2σ²).  Prevents the
            LM step from collapsing two nearby atoms back to their centroid
            after the certificate correctly places them apart.

            Rule of thumb: rep_strength ≈ 0.1–0.2 × A_single.
            For A_single ~ 1000 ph, start with rep_strength = 100–200.
            Too large (> 0.5 × A_single) over-repels and inflates position
            errors; zero leaves pairs co-located at the centroid.

The two parameters are complementary and both are needed for sub-2σ
recovery: amp_cap creates the residual the certificate needs; rep_strength
ensures the proposed second atom stays where the certificate placed it.

Typical call for a crowded fluorescence image (σ ≈ 2 px, bg ≈ 10 ph/px,
single-emitter brightness ≈ 1000 ph):

    amps, pos, energies = sfw_poisson(
        y, sigma=2.0, bg=10.0, lam=1.0,
        S_min=10.0,
        amp_cap=500.0,      # ~0.5 × A_single
        rep_strength=150.0, # ~0.15 × A_single
    )
"""

import mlx.core as mx
import numpy as np

from .model import (
    EPS,
    build_basis,
    compute_certificate,
    find_new_spike,
    forward_model,
    poisson_nll,
)
from .solvers import fista_amplitudes, lm_amplitudes, lm_positions
from .utils import prune


def sfw_poisson(
    y,
    sigma,
    bg,
    lam,
    n_iter=20,
    fista_iter=50,
    lm_iter=10,
    n_refine=3,
    S_min=10.0,
    pos_bound=1.0,
    amp_cap=None,
    rep_strength=0.0,
    verbose=True,
):
    """Sliding Frank-Wolfe with Poisson noise model.

    Parameters
    ----------
    y : (H, W) observed image (MLX array)
    sigma : PSF width
    bg : background level (scalar)
    lam : sparsity regularisation
    n_iter : max outer iterations
    fista_iter : FISTA iterations for amplitude update (with MCP penalty)
    lm_iter : LM iterations per block in refinement
    n_refine : number of alternating LM sweeps (amp → pos) per SFW iter
    S_min : minimum signal threshold in photons (default 10.0).
        Controls both the MCP penalty and hard pruning:
          • MCP shape γ = S_min / lam  — amplitudes above S_min are
            recovered without bias; those below are driven to zero.
          • prune threshold = S_min    — any spike below S_min is
            discarded after each SFW iteration.
        Set S_min ≈ your noise-floor photon count (e.g. a few times the
        per-pixel background).  No need to tune prune_tol separately.
    pos_bound : position box constraint (±pixels)
    rep_strength : Gaussian repulsion strength between emitters (default 0).
        Prevents nearby atoms from collapsing to the same position during LM
        refinement.  Set to O(100–1000) for typical photon counts; scale
        roughly with expected single-emitter amplitude.  Pairs with amp_cap
        to resolve closely-spaced emitters: amp_cap creates the residual the
        certificate needs to propose a second atom; rep_strength keeps the
        two atoms from collapsing back together after the proposal.
    amp_cap : per-emitter amplitude upper bound for LM refinement
        (default None = unconstrained).  When set, LM cannot assign more
        than amp_cap photons to any single emitter.  Excess flux stays in
        the residual, letting the certificate propose a second nearby atom
        on the next SFW iteration — useful for resolving closely-spaced
        emitter pairs.  The certificate mask radius is automatically
        tightened to 0.5 σ (from 1 σ) so that the residual candidate is
        not suppressed even if it falls close to an existing emitter.

    Returns
    -------
    amplitudes : (K,)
    positions : (K, 2)  — row, col
    energies : list of mean Poisson deviance
    """
    mcp_gamma = S_min / lam
    H, W = y.shape
    gy = mx.arange(H, dtype=mx.float32)
    gx = mx.arange(W, dtype=mx.float32)
    grid_y, grid_x = mx.meshgrid(gy, gx, indexing="ij")

    amplitudes = mx.array([], dtype=mx.float32)
    positions = mx.zeros((0, 2), dtype=mx.float32)

    energies = []
    stable_count = 0
    prev_k = 0

    for it in range(n_iter):
        # Current model
        if amplitudes.size == 0:
            model_img = mx.full(y.shape, bg, dtype=mx.float32)
        else:
            model_img = forward_model(amplitudes, positions, grid_y, grid_x, sigma, bg)
        mx.eval(model_img)

        # Certificate — find most negative eta, masking near existing spikes.
        # With amp_cap active we tighten the exclusion zone to 0.5 σ so that
        # a residual peak right next to a capped emitter is still proposable.
        _mask_r = (0.5 * sigma) ** 2 if amp_cap is not None else sigma**2
        eta = compute_certificate(y, model_img, sigma, lam, grid_y, grid_x)
        if amplitudes.size > 0:
            for k in range(positions.shape[0]):
                dy = grid_y - positions[k, 0]
                dx = grid_x - positions[k, 1]
                mask = (dy * dy + dx * dx) < _mask_r
                eta = mx.where(mask, mx.array(0.0), eta)
            mx.eval(eta)
        new_pos, min_eta = find_new_spike(eta)
        mx.eval(min_eta)

        if verbose:
            K = amplitudes.size
            # Poisson deviance: 2 * sum(model - y - y * log(model/y))
            deviance = 2.0 * mx.sum(
                model_img - y - y * mx.log((model_img + EPS) / (y + EPS))
            )
            N = y.shape[0] * y.shape[1]
            mean_dev = deviance / N
            mx.eval(mean_dev)
            print(
                f"  iter {it:3d} | K={K:3d} | min(eta)={min_eta.item():.4f} "
                f"| mean deviance={mean_dev.item():.4f}"
            )
            energies.append(mean_dev.item())

        if min_eta.item() >= -1.0:
            if verbose:
                print("  Certificate >= -1 — converged.")
            break

        # Append new spike — merge_nearby will collapse duplicates
        if amplitudes.size == 0:
            amplitudes = mx.array([1.0])
            positions = new_pos
        else:
            amplitudes = mx.concatenate([amplitudes, mx.array([1.0])])
            positions = mx.concatenate([positions, new_pos], axis=0)
        mx.eval(amplitudes, positions)

        # FISTA with MCP penalty for sparse amplitude recovery
        y_flat = y.reshape(-1)
        H_basis = build_basis(positions, grid_y, grid_x, sigma)
        mx.eval(H_basis)
        amplitudes = fista_amplitudes(
            amplitudes, H_basis, y_flat, bg, lam=lam, gamma=mcp_gamma, n_iter=fista_iter
        )
        if amp_cap is not None:
            amplitudes = mx.clip(amplitudes, 0.0, amp_cap)
        mx.eval(amplitudes)

        # Block-coordinate LM refinement (pure NLL, no penalty):
        # alternate LM-amplitudes → LM-positions
        _amp_upper = amp_cap if amp_cap is not None else 1e5
        pos_before = positions
        for _ref in range(n_refine):
            amplitudes = lm_amplitudes(
                amplitudes,
                positions,
                y,
                grid_y,
                grid_x,
                sigma,
                bg,
                amp_upper=_amp_upper,
                n_outer=lm_iter,
            )
            mx.eval(amplitudes)
            positions = lm_positions(
                amplitudes,
                positions,
                y,
                grid_y,
                grid_x,
                sigma,
                bg,
                pos_bound=pos_bound,
                rep_strength=rep_strength,
                n_outer=lm_iter,
            )
            mx.eval(positions)

        if verbose and pos_before.shape[0] > 0:
            shifts = mx.sqrt(mx.sum((positions - pos_before) ** 2, axis=1))
            mx.eval(shifts)
            shifts_np = np.array(shifts)
            print(
                f"    LM shift — mean: {shifts_np.mean():.3f}, max: {shifts_np.max():.3f} px"
            )

        # Prune only — no merging, let LM + bounds handle separation
        amplitudes, positions = prune(amplitudes, positions, tol=S_min)
        mx.eval(amplitudes, positions)

        if amplitudes.size == 0:
            if verbose:
                print("  All spikes pruned — stopping.")
            break

        # Early stopping: emitter count stable (new spike merged away)
        K = amplitudes.size
        if K == prev_k:
            stable_count += 1
        else:
            stable_count = 0
        prev_k = K
        if stable_count >= 4 and it > 5:
            if verbose:
                print(f"  Emitter count stable at K={K} for 4 iters — stopping.")
            break

    return amplitudes, positions, energies
