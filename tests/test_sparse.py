#!/usr/bin/env python3
"""Test SFW on well-separated emitters (easy case).

8 emitters spread across a 64x64 field, all >5σ apart.
Expected: all recovered with sub-pixel accuracy.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from scipy.spatial.distance import cdist

from sfwloc import forward_model, sfw_poisson

# ── Parameters ───────────────────────────────────────────────────────────────

SIGMA = 2.0
BG = 10.0
IMG_SHAPE = (64, 64)
SEED = 123
AMP_RANGE = (800.0, 2000.0)
READOUT_STD = 2.0
LAM = 1.0

# ── Generate data ────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
H, W = IMG_SHAPE

# Place emitters on a grid-like layout so they're well separated
gt_pos = np.array(
    [
        [12.3, 12.7],
        [12.5, 50.2],
        [32.1, 20.4],
        [32.8, 45.6],
        [48.0, 10.9],
        [48.5, 35.3],
        [48.2, 55.1],
        [28.0, 55.8],
    ],
    dtype=np.float32,
)
gt_amp = rng.uniform(*AMP_RANGE, size=len(gt_pos)).astype(np.float32)
n_gt = len(gt_pos)

gt_pos_mx = mx.array(gt_pos)
gt_amp_mx = mx.array(gt_amp)

gy = mx.arange(H, dtype=mx.float32)
gx = mx.arange(W, dtype=mx.float32)
grid_y, grid_x = mx.meshgrid(gy, gx, indexing="ij")

clean = forward_model(gt_amp_mx, gt_pos_mx, grid_y, grid_x, SIGMA, BG)
mx.eval(clean)

clean_np = np.array(clean)
noisy = rng.poisson(np.maximum(clean_np, 0)).astype(np.float32)
noisy += rng.normal(0, READOUT_STD, size=noisy.shape).astype(np.float32)
noisy = np.maximum(noisy, 0)
y = mx.array(noisy)

# ── Run solver ───────────────────────────────────────────────────────────────

print(f"=== Sparse test: {n_gt} well-separated emitters ===")
print(f"  Image: {H}x{W}, sigma={SIGMA}, bg={BG}, lam={LAM}")
print(f"  Amplitude range: {gt_amp.min():.0f} – {gt_amp.max():.0f}")
print()

amplitudes, positions, energies = sfw_poisson(
    y,
    sigma=SIGMA,
    bg=BG,
    lam=LAM,
    n_iter=25,
    lm_iter=15,
    prune_tol=0.5,
    verbose=True,
)

# ── Results ──────────────────────────────────────────────────────────────────

rec_pos = np.array(positions) if amplitudes.size > 0 else np.zeros((0, 2))
rec_amp = np.array(amplitudes) if amplitudes.size > 0 else np.array([])
K_rec = len(rec_amp)

print(f"\n=== Results: recovered {K_rec} / {n_gt} emitters ===")

if K_rec > 0:
    dists = cdist(rec_pos, gt_pos)
    matched_gt = set()
    pos_errors = []
    for i in range(K_rec):
        j = dists[i].argmin()
        d = dists[i, j]
        matched_gt.add(j)
        pos_errors.append(d)
        print(
            f"  rec {i}: ({rec_pos[i,0]:.2f}, {rec_pos[i,1]:.2f}) "
            f"<-> GT {j} ({gt_pos[j,0]:.2f}, {gt_pos[j,1]:.2f})  "
            f"dist={d:.3f} px  amp={rec_amp[i]:.0f} (GT {gt_amp[j]:.0f})"
        )
    print(f"\n  Mean position error: {np.mean(pos_errors):.3f} px")
    print(f"  Max position error:  {np.max(pos_errors):.3f} px")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.imshow(clean_np, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=8, mew=2, label="GT")
ax.set_title("Clean image + GT")
ax.legend(fontsize=8)

ax = axes[1]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.set_title("Observed (Poisson + readout)")

ax = axes[2]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=10, mew=2, label="GT")
if K_rec > 0:
    ax.plot(rec_pos[:, 1], rec_pos[:, 0], "c+", ms=10, mew=2, label="Recovered")
ax.set_title(f"Recovery: {K_rec}/{n_gt} emitters")
ax.legend(fontsize=8)

for ax in axes:
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

fig.suptitle("Sparse case — well-separated emitters", fontweight="bold")
plt.tight_layout()
plt.savefig("test_sparse_result.png", dpi=150)
print(f"\nPlot saved to test_sparse_result.png")
plt.show()
