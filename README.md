# sfwloc

Gridless sparse point source localization using the **Sliding Frank-Wolfe (SFW)** algorithm, built on [Apple MLX](https://github.com/ml-explore/mlx).

Designed for fluorescence microscopy: recovers emitter positions (continuous, sub-pixel) and amplitudes from blurry images with Poisson noise, without restricting emitters to a pixel grid.

## Features

- **Poisson noise model** — native Poisson NLL (not Gaussian MSE approximation)
- **MCP penalty** — non-convex Minimax Concave Penalty for unbiased amplitude recovery above a threshold
- **Gridless sub-pixel accuracy** — positions are continuous, optimized via Levenberg-Marquardt
- **Coleman-Li scaling** — adaptive box constraints for bounded LM steps
- **Super-resolution mode** — `amp_cap` + `rep_strength` for resolving sub-2σ emitter pairs

## Installation

```bash
pip install -e ".[test]"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[test]"
```

## Quick start

```python
import mlx.core as mx
from sfwloc import sfw_poisson

# y: (H, W) MLX array of observed photon counts
amps, positions, energies = sfw_poisson(
    y,
    sigma=2.0,   # PSF width in pixels
    bg=10.0,     # background level (photons/pixel)
    lam=1.0,     # sparsity regularization
    S_min=10.0,  # minimum signal (photons); sets MCP threshold and prune cutoff
)

# positions: (K, 2) array of (row, col) coordinates
# amps: (K,) array of amplitudes in photons
```

### Crowded / super-resolution mode

For emitters separated by less than ~2σ:

```python
amps, positions, energies = sfw_poisson(
    y, sigma=2.0, bg=10.0, lam=1.0,
    S_min=10.0,
    amp_cap=500.0,       # ~0.5 × expected single-emitter amplitude
    rep_strength=150.0,  # ~0.15 × expected single-emitter amplitude
)
```

## Algorithm

See [`docs/algorithm.md`](docs/algorithm.md) for a full description of the SFW algorithm, the MCP penalty, and the Levenberg-Marquardt refinement.

## Reference

Denoyelle, Q., Duval, V., Peyré, G., & Soubies, E. (2019). *The Sliding Frank-Wolfe Algorithm and its Application to Super-Resolution Microscopy*. Inverse Problems.
