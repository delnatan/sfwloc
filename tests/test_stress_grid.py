#!/usr/bin/env python3
"""Visual stress test: dense grid of close pairs at 1.5*sigma separation.

This is the primary visual benchmark for BLASSO + joint LM.
"""

import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from scipy.spatial.distance import cdist

from sfwloc import forward_model, sfw_poisson

# --- Scene parameters -------------------------------------------------------

SIGMA = 2.0
BG = 10.0
IMG_SHAPE = (96, 96)
READOUT_STD = 2.0
SEED = 20260219
PAIR_SEP = 1.5 * SIGMA

# 6 x 3 center grid => 18 close pairs => 36 emitters
GRID_Y = np.linspace(12, 84, 6, dtype=np.float32)
GRID_X = np.linspace(14, 82, 3, dtype=np.float32)

# --- Solver parameters ------------------------------------------------------

SOLVER_KWARGS = dict(
    sigma=SIGMA,
    bg=BG,
    lam=0.15,
    n_iter=50,
    fista_iter=20,
    n_refine=2,
    joint_lm_iter=6,
    joint_lm_sigma_bound=0.0,
    prune_tol=1e-4,
    cert_tol=1e-3,
    verbose=True,
)

MATCH_THRESH = 4.0

# --- Generate ground truth --------------------------------------------------

rng = np.random.default_rng(SEED)
centers = np.array([(y, x) for y in GRID_Y for x in GRID_X], dtype=np.float32)

gt_pos = []
gt_amp = []
for c in centers:
    ang = rng.uniform(0, 2 * np.pi)
    off = (PAIR_SEP / 2) * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
    gt_pos.extend([c + off, c - off])

    base_amp = rng.uniform(900.0, 1500.0)
    gt_amp.extend([base_amp, base_amp * rng.uniform(0.75, 1.25)])

gt_pos = np.array(gt_pos, dtype=np.float32)
gt_amp = np.array(gt_amp, dtype=np.float32)
n_gt = len(gt_pos)
pair_idx = [(2 * i, 2 * i + 1) for i in range(len(centers))]

H, W = IMG_SHAPE
grid_y, grid_x = mx.meshgrid(
    mx.arange(H, dtype=mx.float32),
    mx.arange(W, dtype=mx.float32),
    indexing="ij",
)

clean = forward_model(mx.array(gt_amp), mx.array(gt_pos), grid_y, grid_x, SIGMA, BG)
mx.eval(clean)
clean_np = np.array(clean)

obs = rng.poisson(np.maximum(clean_np, 0.0)).astype(np.float32)
obs += rng.normal(0.0, READOUT_STD, size=obs.shape).astype(np.float32)
obs = np.maximum(obs, 0.0)
y = mx.array(obs)

print("=== Stress test: BLASSO + joint LM on close-pair grid ===")
print(f"  Image: {H}x{W}")
print(f"  Sigma: {SIGMA}")
print(f"  Pair separation: {PAIR_SEP:.2f} px ({PAIR_SEP/SIGMA:.2f} sigma)")
print(f"  Emitters: {n_gt} ({len(pair_idx)} close pairs)")
print(f"  Amplitude range: {gt_amp.min():.0f} - {gt_amp.max():.0f}")
print()

# --- Run solver -------------------------------------------------------------

t0 = time.perf_counter()
amplitudes, positions, energies = sfw_poisson(y, **SOLVER_KWARGS)
elapsed = time.perf_counter() - t0
mx.eval(amplitudes, positions)

rec_pos = np.array(positions) if amplitudes.size > 0 else np.zeros((0, 2), dtype=np.float32)
rec_amp = np.array(amplitudes) if amplitudes.size > 0 else np.zeros((0,), dtype=np.float32)
K_rec = len(rec_amp)

# --- Greedy matching --------------------------------------------------------

def greedy_match(rec_pos_, gt_pos_, thresh):
    if len(rec_pos_) == 0 or len(gt_pos_) == 0:
        return []
    d = cdist(rec_pos_, gt_pos_)
    used_rec, used_gt = set(), set()
    matches = []
    for _ in range(min(len(rec_pos_), len(gt_pos_))):
        best = (np.inf, -1, -1)
        for i in range(len(rec_pos_)):
            if i in used_rec:
                continue
            for j in range(len(gt_pos_)):
                if j in used_gt:
                    continue
                if d[i, j] < best[0]:
                    best = (d[i, j], i, j)
        if best[0] < thresh:
            used_rec.add(best[1])
            used_gt.add(best[2])
            matches.append(best)
    return matches

matches = greedy_match(rec_pos, gt_pos, MATCH_THRESH)
matched_rec = {i for _, i, _ in matches}
matched_gt = {j for _, _, j in matches}

TP = len(matches)
FP = max(0, K_rec - TP)
FN = max(0, n_gt - TP)

pair_complete = sum(1 for a, b in pair_idx if a in matched_gt and b in matched_gt)
pair_complete_frac = pair_complete / len(pair_idx)
emit_recall = TP / n_gt if n_gt > 0 else 0.0
precision = TP / K_rec if K_rec > 0 else 0.0

print("=== Results ===")
print(f"  Runtime:    {elapsed:.2f} s")
print(f"  Recovered:  {K_rec}")
print(f"  TP / FP / FN: {TP} / {FP} / {FN}")
print(f"  Recall:     {emit_recall:.3f}")
print(f"  Precision:  {precision:.3f}")
print(f"  Complete close-pairs: {pair_complete}/{len(pair_idx)} ({pair_complete_frac:.3f})")

if matches:
    pos_err = np.array([d for d, _, _ in matches], dtype=float)
    print(f"  Position error mean/median/max: {pos_err.mean():.3f} / {np.median(pos_err):.3f} / {pos_err.max():.3f} px")

# --- Plot -------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

ax = axes[0]
ax.imshow(clean_np, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=5, mew=1.5, label="GT")
ax.set_title("Clean + GT")
ax.legend(fontsize=7)

ax = axes[1]
ax.imshow(obs, cmap="gray", origin="lower")
ax.set_title("Observed")

ax = axes[2]
ax.imshow(obs, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=5, mew=1.5, label="GT")
if K_rec > 0:
    ax.plot(rec_pos[:, 1], rec_pos[:, 0], "c+", ms=6, mew=1.5, label="Recovered")
ax.set_title(f"Recovered (K={K_rec})")
ax.legend(fontsize=7)

ax = axes[3]
if energies:
    ax.plot(energies, "o-", ms=3)
ax.set_title("Outer Convergence")
ax.set_xlabel("Outer iteration")
ax.set_ylabel("Mean deviance")
ax.grid(alpha=0.25)

for ax in axes[:3]:
    for c in centers:
        circ = plt.Circle(
            (c[1], c[0]),
            PAIR_SEP,
            fill=False,
            edgecolor="yellow",
            linewidth=0.8,
            linestyle="--",
        )
        ax.add_patch(circ)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

fig.suptitle(
    (
        f"Stress grid (36 emitters, sep={PAIR_SEP:.1f}px={PAIR_SEP/SIGMA:.1f}σ) | "
        f"TP/FP/FN={TP}/{FP}/{FN} | pair-complete={pair_complete}/{len(pair_idx)}"
    ),
    fontweight="bold",
)
fig.tight_layout()
out_path = "test_stress_grid_result.png"
fig.savefig(out_path, dpi=170)
print(f"\nPlot saved to {out_path}")
plt.show()
