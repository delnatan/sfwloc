"""sfwloc: Sliding Frank-Wolfe for gridless sparse point source localization.

Built on Apple MLX with a Poisson noise model and MCP amplitude penalty.

Primary entry point
-------------------
>>> from sfwloc import sfw_poisson
>>> amps, positions, energies = sfw_poisson(y, sigma=2.0, bg=10.0, lam=1.0)
"""

from .algorithm import sfw_poisson
from .model import (
    build_basis,
    compute_certificate,
    find_new_spike,
    forward_model,
    gaussian_psf,
    poisson_nll,
)
from .solvers import (
    fista_amplitudes,
    joint_lm_refine,
    lm_amplitudes,
    lm_positions,
)
from .utils import merge_nearby, prune

__version__ = "0.1.0"

__all__ = [
    "sfw_poisson",
    # model
    "gaussian_psf",
    "forward_model",
    "build_basis",
    "poisson_nll",
    "compute_certificate",
    "find_new_spike",
    # solvers
    "fista_amplitudes",
    "lm_amplitudes",
    "lm_positions",
    "joint_lm_refine",
    # utils
    "prune",
    "merge_nearby",
]
