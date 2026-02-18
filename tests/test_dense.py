#!/usr/bin/env python3
"""Test SFW on a dense field of 40 emitters.

Random placement with a minimum separation of 2σ enforced via rejection
sampling. This mimics a moderately dense SMLM field where some emitters
are close but not completely unresolvable.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from scipy.spatial.distance import cdist

from sfwloc import forward_model, sfw_poisson

# ── Parameters ───────────────────────────────────────────────────────────────

N_EMITTERS = 80
SIGMA = 1.6
BG = 10.0
IMG_SHAPE = (64, 64)
SEED = 7
AMP_RANGE = (500.0, 2000.0)
READOUT_STD = 2.0
LAM = 1e-2
MIN_SEP = 2.0 * SIGMA  # minimum separation between emitters

# ── Generate data ────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
H, W = IMG_SHAPE
margin = 3 * SIGMA

# Rejection sampling to enforce minimum separation
gt_pos_list = []
while len(gt_pos_list) < N_EMITTERS:
    cand = rng.uniform([margin, margin], [H - margin, W - margin]).astype(
        np.float32
    )
    if len(gt_pos_list) == 0:
        gt_pos_list.append(cand)
        continue
    existing = np.array(gt_pos_list)
    dists = np.linalg.norm(existing - cand, axis=1)
    if dists.min() >= MIN_SEP:
        gt_pos_list.append(cand)

gt_pos = np.array(gt_pos_list, dtype=np.float32)
gt_amp = rng.uniform(*AMP_RANGE, size=N_EMITTERS).astype(np.float32)
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

# ── Print ground truth summary ───────────────────────────────────────────────

print(
    f"=== Dense test: {n_gt} emitters (min sep = {MIN_SEP:.1f} px = {MIN_SEP / SIGMA:.1f}σ) ==="
)
print(f"  Image: {H}x{W}, sigma={SIGMA}, bg={BG}, lam={LAM}")
print(f"  Amplitude range: {gt_amp.min():.0f} – {gt_amp.max():.0f}")

# Compute nearest-neighbor distances
gt_dists = cdist(gt_pos, gt_pos)
np.fill_diagonal(gt_dists, np.inf)
nn_dists = gt_dists.min(axis=1)
print(
    f"  Nearest-neighbor distances: min={nn_dists.min():.2f}, "
    f"median={np.median(nn_dists):.2f}, max={nn_dists.max():.2f} px"
)
print()

# ── Run solver ───────────────────────────────────────────────────────────────

amplitudes, positions, energies = sfw_poisson(
    y,
    sigma=SIGMA,
    bg=BG,
    lam=LAM,
    n_iter=100,
    fista_iter=100,
    lm_iter=25,
    prune_tol=400.0,
    verbose=True,
)

# ── Results ──────────────────────────────────────────────────────────────────

rec_pos = np.array(positions) if amplitudes.size > 0 else np.zeros((0, 2))
rec_amp = np.array(amplitudes) if amplitudes.size > 0 else np.array([])
K_rec = len(rec_amp)

print(f"\n=== Results: recovered {K_rec} / {n_gt} emitters ===")

if K_rec > 0:
    dists = cdist(rec_pos, gt_pos)

    # Greedy nearest-neighbor matching
    used_rec = set()
    used_gt = set()
    matches = []
    dist_flat = [
        (dists[i, j], i, j) for i in range(K_rec) for j in range(n_gt)
    ]
    dist_flat.sort()
    for d, i, j in dist_flat:
        if i in used_rec or j in used_gt:
            continue
        if d > 3.0:
            break
        matches.append((i, j, d))
        used_rec.add(i)
        used_gt.add(j)

    pos_errors = [d for _, _, d in matches]
    amp_ratios = [rec_amp[i] / gt_amp[j] for i, j, _ in matches]
    n_matched = len(matches)
    n_spurious = K_rec - len(used_rec)
    n_missed = n_gt - len(used_gt)

    print(f"  Matched:  {n_matched}/{n_gt}")
    print(f"  Missed:   {n_missed}")
    print(f"  Spurious: {n_spurious}")
    if pos_errors:
        print(
            f"  Position error — mean: {np.mean(pos_errors):.3f}, "
            f"median: {np.median(pos_errors):.3f}, max: {np.max(pos_errors):.3f} px"
        )
        print(
            f"  Amplitude ratio (rec/GT) — mean: {np.mean(amp_ratios):.3f}, "
            f"std: {np.std(amp_ratios):.3f}"
        )

    # Show missed emitters
    if n_missed > 0:
        print(f"\n  Missed GT emitters:")
        for j in range(n_gt):
            if j not in used_gt:
                print(
                    f"    GT {j}: ({gt_pos[j, 0]:.2f}, {gt_pos[j, 1]:.2f})  "
                    f"amp={gt_amp[j]:.0f}  nn_dist={nn_dists[j]:.2f} px"
                )

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

ax = axes[0, 0]
ax.imshow(clean_np, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=6, mew=1.5)
ax.set_title(f"Clean image + {n_gt} GT emitters")

ax = axes[0, 1]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.set_title("Observed (Poisson + readout)")

ax = axes[1, 0]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=8, mew=1.5, label="GT")
if K_rec > 0:
    ax.plot(
        rec_pos[:, 1], rec_pos[:, 0], "c+", ms=8, mew=1.5, label="Recovered"
    )
ax.set_title(f"Recovery: {K_rec}/{n_gt}")
ax.legend(fontsize=8)

ax = axes[1, 1]
if energies:
    ax.plot(energies, "o-", ms=4)
    ax.set_xlabel("SFW iteration")
    ax.set_ylabel("Mean Poisson deviance")
    ax.set_title("Convergence")
    ax.axhline(1.0, color="k", ls="--", alpha=0.4, label="ideal (deviance=1)")
    ax.legend(fontsize=8)

for ax in axes.flat[:3]:
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

fig.suptitle(
    f"Dense field — {n_gt} emitters, min sep {MIN_SEP:.0f} px ({MIN_SEP / SIGMA:.0f}σ)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("test_dense_result.png", dpi=150)
print(f"\nPlot saved to test_dense_result.png")
plt.show()
