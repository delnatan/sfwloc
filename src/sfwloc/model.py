"""PSF, forward model, Poisson negative log-likelihood, and Frank-Wolfe certificate."""

import mlx.core as mx

EPS = 1e-8


# ── Forward model ─────────────────────────────────────────────────────────────


def gaussian_psf(positions, grid_y, grid_x, sigma):
    """Evaluate normalized Gaussian PSF for each emitter at every pixel.

    Parameters
    ----------
    positions : (K, 2)  — row, col coordinates (continuous)
    grid_y, grid_x : (H, W) coordinate grids
    sigma : scalar

    Returns
    -------
    (K, H, W) PSF images, each normalized to sum to 1.
    """

    def _single(pos):
        dy = grid_y - pos[0]
        dx = grid_x - pos[1]
        unnorm = mx.exp(-(dy * dy + dx * dx) / (2.0 * sigma * sigma))
        return unnorm / (unnorm.sum() + EPS)

    return mx.vmap(_single)(positions)


def forward_model(amplitudes, positions, grid_y, grid_x, sigma, bg):
    """Differentiable image model: sum_k a_k * PSF_k + bg.

    Returns (H, W) predicted image.
    """
    psfs = gaussian_psf(positions, grid_y, grid_x, sigma)  # (K, H, W)
    img = mx.tensordot(amplitudes, psfs, axes=[[0], [0]])  # (H, W)
    return img + bg


def build_basis(positions, grid_y, grid_x, sigma):
    """Precompute PSF basis matrix H with positions frozen.

    Returns (N, K) where N = H*W and column k is the flattened PSF for
    emitter k.
    """
    psfs = gaussian_psf(positions, grid_y, grid_x, sigma)  # (K, H, W)
    K = psfs.shape[0]
    return psfs.reshape(K, -1).T  # (N, K)


# ── Poisson NLL & certificate ─────────────────────────────────────────────────


def poisson_nll(model, y):
    """Poisson negative log-likelihood (up to constant)."""
    return mx.sum(model - y * mx.log(model + EPS))


def compute_certificate(y, model_img, sigma, lam, grid_y, grid_x):
    """Dual certificate eta = (1/lam) * Phi^*(1 - y/model).

    Uses mx.conv2d with the PSF kernel (symmetric, so no flip needed) to
    compute the adjoint operator.

    Returns (H, W) certificate image.
    """
    w = 1.0 - y / (model_img + EPS)  # (H, W) weighted residual

    # Build small PSF kernel centred at (0,0)
    H, W = grid_y.shape
    ksize = int(6 * sigma) + 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    kc = ksize // 2
    ky = mx.arange(ksize, dtype=mx.float32) - kc
    kx = mx.arange(ksize, dtype=mx.float32) - kc
    kgy, kgx = mx.meshgrid(ky, kx, indexing="ij")
    kernel = mx.exp(-(kgy * kgy + kgx * kgx) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + EPS)

    # mx.conv2d: input (N, H, W, C_in), weight (C_out, kH, kW, C_in)
    w_4d = w[None, :, :, None]  # (1, H, W, 1)
    k_4d = kernel[None, :, :, None]  # (1, kH, kW, 1)
    pad = kc
    adj = mx.conv2d(w_4d, k_4d, padding=pad)  # (1, H, W, 1)
    adj = adj[0, :H, :W, 0]

    return adj / lam


def find_new_spike(eta):
    """Find pixel location with most negative eta (strongest certificate violation).

    For Poisson NLL + non-negative L1: adding a spike is beneficial when
    eta(x) < -1.  We find argmin(eta) and return (position, min_val).
    """
    flat_idx = mx.argmin(eta.reshape(-1))
    mx.eval(flat_idx)
    idx = flat_idx.item()
    H, W = eta.shape
    row, col = idx // W, idx % W
    min_val = eta.reshape(-1)[flat_idx]
    return mx.array([[float(row), float(col)]]), min_val
