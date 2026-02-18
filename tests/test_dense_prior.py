#!/usr/bin/env python3
"""Test amplitude prior on a dense 40-emitter field.

Amplitudes drawn from N(μ, σ²) so the prior is well-matched.
Minimum separation 2σ via rejection sampling — some neighbors are close
enough to interact but not completely unresolvable.

Runs solver without and with amplitude prior for comparison.
"""

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from scipy.spatial.distance import cdist

from sfwloc import forward_model, sfw_poisson

# ── Parameters ───────────────────────────────────────────────────────────────

N_EMITTERS = 40
SIGMA = 2.0
BG = 10.0
IMG_SHAPE = (64, 64)
SEED = 7
READOUT_STD = 2.0
LAM = 0.1
MIN_SEP = 2.0 * SIGMA

# Amplitude distribution
MU_A = 1000.0
SIGMA_A = 200.0

# ── Generate data ────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
H, W = IMG_SHAPE
margin = 3 * SIGMA

# Rejection sampling to enforce minimum separation
gt_pos_list = []
while len(gt_pos_list) < N_EMITTERS:
    cand = rng.uniform([margin, margin], [H - margin, W - margin]).astype(np.float32)
    if len(gt_pos_list) == 0:
        gt_pos_list.append(cand)
        continue
    existing = np.array(gt_pos_list)
    dists = np.linalg.norm(existing - cand, axis=1)
    if dists.min() >= MIN_SEP:
        gt_pos_list.append(cand)

gt_pos = np.array(gt_pos_list, dtype=np.float32)

# Draw amplitudes from the known distribution
gt_amp = rng.normal(MU_A, SIGMA_A, size=N_EMITTERS).astype(np.float32)
gt_amp = np.maximum(gt_amp, 100.0)  # floor to avoid near-zero
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

gt_dists = cdist(gt_pos, gt_pos)
np.fill_diagonal(gt_dists, np.inf)
nn_dists = gt_dists.min(axis=1)

print(f"=== Dense prior test: {n_gt} emitters ===")
print(f"  Image: {H}x{W}, sigma={SIGMA}, bg={BG}, lam={LAM}")
print(f"  Min separation: {MIN_SEP:.1f} px ({MIN_SEP/SIGMA:.1f}σ)")
print(f"  GT amps ~ N({MU_A:.0f}, {SIGMA_A:.0f}²): "
      f"mean={gt_amp.mean():.0f}, std={gt_amp.std():.0f}, "
      f"range=[{gt_amp.min():.0f}, {gt_amp.max():.0f}]")
print(f"  NN distances: min={nn_dists.min():.2f}, "
      f"median={np.median(nn_dists):.2f}, max={nn_dists.max():.2f} px")


# ── Helper: run and evaluate ─────────────────────────────────────────────────


def run_and_evaluate(label, lam_override=None, **extra_kwargs):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    amplitudes, positions, energies = sfw_poisson(
        y,
        sigma=SIGMA,
        bg=BG,
        lam=lam_override if lam_override is not None else LAM,
        n_iter=60,
        lm_iter=15,
        prune_tol=0.5,
        verbose=True,
        **extra_kwargs,
    )

    rec_pos = np.array(positions) if amplitudes.size > 0 else np.zeros((0, 2))
    rec_amp = np.array(amplitudes) if amplitudes.size > 0 else np.array([])
    K_rec = len(rec_amp)

    # Greedy matching
    matches = []
    used_rec, used_gt = set(), set()
    if K_rec > 0:
        dists = cdist(rec_pos, gt_pos)
        dist_flat = sorted(
            [(dists[i, j], i, j) for i in range(K_rec) for j in range(n_gt)]
        )
        for d, i, j in dist_flat:
            if i in used_rec or j in used_gt:
                continue
            if d > 3.0:
                break
            matches.append((i, j, d))
            used_rec.add(i)
            used_gt.add(j)

    n_matched = len(matches)
    n_spurious = K_rec - len(used_rec)
    n_missed = n_gt - len(used_gt)
    pos_errors = [d for _, _, d in matches]
    amp_ratios = [rec_amp[i] / gt_amp[j] for i, j, _ in matches]
    amp_errors = [abs(rec_amp[i] - gt_amp[j]) for i, j, _ in matches]

    print(f"\n  Results: {K_rec} recovered, {n_matched}/{n_gt} matched, "
          f"{n_missed} missed, {n_spurious} spurious")
    if pos_errors:
        print(f"  Position — mean: {np.mean(pos_errors):.3f}, "
              f"median: {np.median(pos_errors):.3f}, max: {np.max(pos_errors):.3f} px")
        print(f"  Amp ratio (rec/GT) — mean: {np.mean(amp_ratios):.3f}, "
              f"std: {np.std(amp_ratios):.3f}")
        print(f"  Amp error — mean: {np.mean(amp_errors):.0f}, "
              f"median: {np.median(amp_errors):.0f}, max: {np.max(amp_errors):.0f}")

    if n_missed > 0:
        print(f"\n  Missed GT emitters:")
        for j in range(n_gt):
            if j not in used_gt:
                print(f"    GT {j}: ({gt_pos[j,0]:.2f}, {gt_pos[j,1]:.2f})  "
                      f"amp={gt_amp[j]:.0f}  nn_dist={nn_dists[j]:.2f} px")

    return K_rec, rec_pos, rec_amp, matches, energies


# ── Run both ─────────────────────────────────────────────────────────────────

K1, pos1, amp1, m1, e1 = run_and_evaluate(f"lam={LAM}")
K2, pos2, amp2, m2, e2 = run_and_evaluate("lam=0.05", lam_override=0.05)

# ── Comparison plot ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Row 0: images
ax = axes[0, 0]
ax.imshow(clean_np, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=6, mew=1.5)
ax.set_title(f"Clean + {n_gt} GT emitters")

ax = axes[0, 1]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=8, mew=1.5, label="GT")
if K1 > 0:
    ax.plot(pos1[:, 1], pos1[:, 0], "c+", ms=8, mew=1.5, label="Recovered")
ax.set_title(f"lam=0.1: {K1} rec, {len(m1)}/{n_gt} matched")
ax.legend(fontsize=7)

ax = axes[0, 2]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=8, mew=1.5, label="GT")
if K2 > 0:
    ax.plot(pos2[:, 1], pos2[:, 0], "c+", ms=8, mew=1.5, label="Recovered")
ax.set_title(f"lam=0.05: {K2} rec, {len(m2)}/{n_gt} matched")
ax.legend(fontsize=7)

# Row 1: quantitative
ax = axes[1, 0]
if e1:
    ax.plot(e1, "o-", ms=3, label="lam=0.1")
if e2:
    ax.plot(e2, "s-", ms=3, label="lam=0.05")
ax.axhline(1.0, color="k", ls="--", alpha=0.4)
ax.set_xlabel("SFW iteration")
ax.set_ylabel("Mean Poisson deviance")
ax.set_title("Convergence")
ax.legend(fontsize=8)

ax = axes[1, 1]
if m1:
    gt_m1 = [gt_amp[j] for _, j, _ in m1]
    rc_m1 = [amp1[i] for i, _, _ in m1]
    ax.scatter(gt_m1, rc_m1, marker="o", s=30, alpha=0.7,
               edgecolors="C0", facecolors="none", label="lam=0.1")
if m2:
    gt_m2 = [gt_amp[j] for _, j, _ in m2]
    rc_m2 = [amp2[i] for i, _, _ in m2]
    ax.scatter(gt_m2, rc_m2, marker="s", s=30, alpha=0.7,
               edgecolors="C1", facecolors="none", label="lam=0.05")
lims = [0, max(gt_amp) * 1.5]
ax.plot(lims, lims, "k--", alpha=0.3)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("GT amplitude")
ax.set_ylabel("Recovered amplitude")
ax.set_title("Amplitude accuracy")
ax.legend(fontsize=8)
ax.set_aspect("equal")

ax = axes[1, 2]
if m1:
    pe1 = sorted([d for _, _, d in m1])
    ax.plot(pe1, np.linspace(0, 1, len(pe1)), "-", label="lam=0.1", lw=2)
if m2:
    pe2 = sorted([d for _, _, d in m2])
    ax.plot(pe2, np.linspace(0, 1, len(pe2)), "--", label="lam=0.05", lw=2)
ax.set_xlabel("Position error (px)")
ax.set_ylabel("CDF")
ax.set_title("Position error CDF")
ax.legend(fontsize=8)

for r in range(2):
    for c in range(3):
        if r == 0:
            axes[r, c].set_xlabel("x (col)")
            axes[r, c].set_ylabel("y (row)")

fig.suptitle(
    f"Dense field ({n_gt} emitters, min sep {MIN_SEP:.0f}px={MIN_SEP/SIGMA:.0f}σ) — "
    f"amps ~ N({MU_A:.0f}, {SIGMA_A:.0f}²)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("test_dense_prior_result.png", dpi=150)
print(f"\nPlot saved to test_dense_prior_result.png")
plt.show()
