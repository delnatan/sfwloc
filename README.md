# sfwloc

Gridless sparse point source localization using the **Sliding Frank-Wolfe (SFW)** algorithm, built on [Apple MLX](https://github.com/ml-explore/mlx).

Designed for fluorescence microscopy: recovers emitter positions (continuous, sub-pixel) and amplitudes from blurry images with Poisson noise, without restricting emitters to a pixel grid.

## Features

- **Poisson noise model** — native Poisson NLL (not Gaussian MSE approximation)
- **BLASSO amplitudes** — projected-FISTA for `a >= 0` with L1 penalty `lam * sum(a)`
- **Continuous source selection** — coarse grid certificate + local refinement
- **Joint local refinement** — bounded LM jointly updates amplitudes and positions
- **Lam continuation** — optional warm-started `lam_schedule` (high → low)
- **Coleman-Li scaling** — adaptive box constraints for bounded LM steps

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
    lam=0.15,    # BLASSO weight for sum(a), a>=0
    prune_tol=1e-4,  # prune near-zero weights after each iteration
)

# positions: (K, 2) array of (row, col) coordinates
# amps: (K,) array of amplitudes in photons
```

Optional continuation:

```python
amps, positions, energies = sfw_poisson(
    y, sigma=2.0, bg=10.0,
    lam_schedule=[0.30, 0.20, 0.15],  # high -> low
)
```

## Algorithm

See [`docs/algorithm.md`](docs/algorithm.md) for a full description of the SFW BLASSO loop and joint Levenberg-Marquardt refinement.

## Visual Stress Test

Run the primary visual benchmark:

```bash
MPLBACKEND=Agg python tests/test_stress_grid.py
```

This generates `test_stress_grid_result.png` for a dense close-pair grid
(36 emitters, pair separation = 1.5σ) and reports TP/FP/FN, recall, precision,
and complete-pair recovery.

## Reference

Denoyelle, Q., Duval, V., Peyré, G., & Soubies, E. (2019). *The Sliding Frank-Wolfe Algorithm and its Application to Super-Resolution Microscopy*. Inverse Problems.
