#!/usr/bin/env python3
"""Test the effect of a Gaussian amplitude prior on recovery.

Ground-truth amplitudes are drawn from N(μ, σ²) so the prior is well-matched.
Scenario: 4 isolated + 3 close pairs at 1.5 px (0.75σ) separation.
Runs the solver twice — without and with the amplitude prior — for comparison.
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
SEED = 99
READOUT_STD = 2.0
LAM = 1.0

# Amplitude distribution: N(mu_a, sigma_a^2)
MU_A = 1000.0
SIGMA_A = 200.0

# ── Generate data ────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
H, W = IMG_SHAPE

# 4 isolated emitters
isolated = np.array(
    [
        [10.0, 10.0],
        [10.0, 50.0],
        [50.0, 10.0],
        [50.0, 50.0],
    ],
    dtype=np.float32,
)

# 3 close pairs at 1.5 px separation
pair_sep = 1.5
pairs_center = np.array(
    [
        [20.0, 32.0],
        [40.0, 25.0],
        [32.0, 50.0],
    ],
    dtype=np.float32,
)

close_pos = []
for center in pairs_center:
    angle = rng.uniform(0, 2 * np.pi)
    offset = pair_sep / 2 * np.array([np.cos(angle), np.sin(angle)])
    close_pos.extend([center + offset, center - offset])
close_pos = np.array(close_pos, dtype=np.float32)

gt_pos = np.concatenate([isolated, close_pos], axis=0)
n_gt = len(gt_pos)

# Draw ALL amplitudes from N(mu_a, sigma_a^2) — the prior's assumed distribution
gt_amp = rng.normal(MU_A, SIGMA_A, size=n_gt).astype(np.float32)
gt_amp = np.maximum(gt_amp, 100.0)  # floor at 100 to avoid near-zero

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

# ── Print ground truth ───────────────────────────────────────────────────────

print(f"=== Amplitude prior test ===")
print(f"  {n_gt} emitters: {len(isolated)} isolated + 3 close pairs (sep={pair_sep} px = {pair_sep/SIGMA:.2f}σ)")
print(f"  GT amplitudes drawn from N({MU_A:.0f}, {SIGMA_A:.0f}²)")
print(f"  Actual: mean={gt_amp.mean():.0f}, std={gt_amp.std():.0f}, "
      f"range=[{gt_amp.min():.0f}, {gt_amp.max():.0f}]")
print()
print("  Ground truth:")
for i in range(n_gt):
    tag = "isolated" if i < len(isolated) else f"pair {(i - len(isolated)) // 2 + 1}"
    print(f"    {i:2d}: ({gt_pos[i,0]:.2f}, {gt_pos[i,1]:.2f})  amp={gt_amp[i]:.0f}  [{tag}]")


# ── Helper: run solver and evaluate ──────────────────────────────────────────


def run_and_evaluate(label, lam_override=None, **extra_kwargs):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    amplitudes, positions, energies = sfw_poisson(
        y,
        sigma=SIGMA,
        bg=BG,
        lam=lam_override if lam_override is not None else LAM,
        n_iter=30,
        lm_iter=20,
        prune_tol=0.5,
        verbose=True,
        **extra_kwargs,
    )

    rec_pos = np.array(positions) if amplitudes.size > 0 else np.zeros((0, 2))
    rec_amp = np.array(amplitudes) if amplitudes.size > 0 else np.array([])
    K_rec = len(rec_amp)

    print(f"\n  Results: recovered {K_rec} / {n_gt}")

    # Greedy matching
    matches = []
    if K_rec > 0:
        dists = cdist(rec_pos, gt_pos)
        used_rec, used_gt = set(), set()
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
        amp_errors = [abs(rec_amp[i] - gt_amp[j]) for i, j, _ in matches]
        amp_ratios = [rec_amp[i] / gt_amp[j] for i, j, _ in matches]

        # Separate isolated vs pair stats
        iso_pos, pair_pos = [], []
        iso_amp, pair_amp = [], []
        for i, j, d in matches:
            if j < len(isolated):
                iso_pos.append(d)
                iso_amp.append(abs(rec_amp[i] - gt_amp[j]))
            else:
                pair_pos.append(d)
                pair_amp.append(abs(rec_amp[i] - gt_amp[j]))

        print(f"  Matched: {n_matched}, Missed: {n_missed}, Spurious: {n_spurious}")
        if pos_errors:
            print(f"  Position — mean: {np.mean(pos_errors):.3f}, max: {np.max(pos_errors):.3f} px")
            print(f"  Amplitude ratio (rec/GT) — mean: {np.mean(amp_ratios):.3f}, std: {np.std(amp_ratios):.3f}")
        if iso_pos:
            print(f"  Isolated: {len(iso_pos)}/4 matched, pos err={np.mean(iso_pos):.3f} px, "
                  f"amp err={np.mean(iso_amp):.0f}")
        n_pair_total = len(close_pos)
        if pair_pos:
            print(f"  Pairs: {len(pair_pos)}/{n_pair_total} matched, pos err={np.mean(pair_pos):.3f} px, "
                  f"amp err={np.mean(pair_amp):.0f}")
        else:
            print(f"  Pairs: 0/{n_pair_total} matched")

        # Print per-emitter details
        for i, j, d in matches:
            tag = "isolated" if j < len(isolated) else f"pair {(j - len(isolated)) // 2 + 1}"
            print(f"    rec {i}: ({rec_pos[i,0]:.2f}, {rec_pos[i,1]:.2f}) "
                  f"<-> GT {j} ({gt_pos[j,0]:.2f}, {gt_pos[j,1]:.2f})  "
                  f"dist={d:.3f} px  amp={rec_amp[i]:.0f} (GT {gt_amp[j]:.0f})  [{tag}]")

        unmatched_gt = set(range(n_gt)) - used_gt
        for j in sorted(unmatched_gt):
            tag = "isolated" if j < len(isolated) else f"pair {(j - len(isolated)) // 2 + 1}"
            print(f"    GT {j}: ({gt_pos[j,0]:.2f}, {gt_pos[j,1]:.2f})  "
                  f"amp={gt_amp[j]:.0f}  [MISSED, {tag}]")

    return K_rec, rec_pos, rec_amp, matches, energies


# ── Run both configurations ──────────────────────────────────────────────────

K1, pos1, amp1, m1, e1 = run_and_evaluate(f"lam={LAM}")
K2, pos2, amp2, m2, e2 = run_and_evaluate(
    f"With lower lam (0.1)",
    lam_override=0.1,
)

# ── Comparison plot ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Top row: without prior
ax = axes[0, 0]
ax.imshow(clean_np, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=8, mew=2)
for center in pairs_center:
    circ = plt.Circle((center[1], center[0]), pair_sep, fill=False,
                       edgecolor="yellow", linewidth=1.5, linestyle="--")
    ax.add_patch(circ)
ax.set_title("Clean + GT")

ax = axes[0, 1]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=10, mew=2, label="GT")
if K1 > 0:
    ax.plot(pos1[:, 1], pos1[:, 0], "c+", ms=10, mew=2, label="Recovered")
for center in pairs_center:
    circ = plt.Circle((center[1], center[0]), pair_sep, fill=False,
                       edgecolor="yellow", linewidth=1.5, linestyle="--")
    ax.add_patch(circ)
ax.set_title(f"lam={LAM}: {K1}/{n_gt}")
ax.legend(fontsize=8)

ax = axes[0, 2]
if e1:
    ax.plot(e1, "o-", ms=3, label=f"lam={LAM}")
if e2:
    ax.plot(e2, "s-", ms=3, label="lam=0.1")
ax.axhline(1.0, color="k", ls="--", alpha=0.4)
ax.set_xlabel("SFW iteration")
ax.set_ylabel("Mean Poisson deviance")
ax.set_title("Convergence")
ax.legend(fontsize=8)

# Bottom row: with prior
ax = axes[1, 0]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.set_title("Observed (Poisson + readout)")

ax = axes[1, 1]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=10, mew=2, label="GT")
if K2 > 0:
    ax.plot(pos2[:, 1], pos2[:, 0], "c+", ms=10, mew=2, label="Recovered")
for center in pairs_center:
    circ = plt.Circle((center[1], center[0]), pair_sep, fill=False,
                       edgecolor="yellow", linewidth=1.5, linestyle="--")
    ax.add_patch(circ)
ax.set_title(f"lam=0.1: {K2}/{n_gt}")
ax.legend(fontsize=8)

# Amplitude comparison
ax = axes[1, 2]
if m1:
    gt_matched1 = [gt_amp[j] for _, j, _ in m1]
    rec_matched1 = [amp1[i] for i, _, _ in m1]
    ax.scatter(gt_matched1, rec_matched1, marker="o", s=40, alpha=0.7,
               edgecolors="C0", facecolors="none", label=f"lam={LAM}")
if m2:
    gt_matched2 = [gt_amp[j] for _, j, _ in m2]
    rec_matched2 = [amp2[i] for i, _, _ in m2]
    ax.scatter(gt_matched2, rec_matched2, marker="s", s=40, alpha=0.7,
               edgecolors="C1", facecolors="none", label="lam=0.1")
lims = [0, max(gt_amp) * 1.5]
ax.plot(lims, lims, "k--", alpha=0.3)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("GT amplitude")
ax.set_ylabel("Recovered amplitude")
ax.set_title("Amplitude accuracy")
ax.legend(fontsize=8)
ax.set_aspect("equal")

for ax in axes.flat:
    if ax.get_xlabel() == "":
        ax.set_xlabel("x (col)")
    if ax.get_ylabel() == "" or ax.get_ylabel() == "y (row)":
        ax.set_ylabel("y (row)")

fig.suptitle(
    f"Lam comparison — close pairs at {pair_sep} px ({pair_sep/SIGMA:.1f}σ), "
    f"amps ~ N({MU_A:.0f}, {SIGMA_A:.0f}²)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("test_prior_result.png", dpi=150)
print(f"\nPlot saved to test_prior_result.png")
plt.show()
