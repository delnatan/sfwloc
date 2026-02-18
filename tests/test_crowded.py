#!/usr/bin/env python3
"""Test SFW on closely-spaced emitters (hard case).

4 isolated emitters + 3 close pairs (~1.5 px separation = 0.75σ).
Compares a low S_min (weak penalty, nearly unbiased) vs a high S_min
(strong penalty, L1-like) to demonstrate the MCP amplitude threshold.
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

SOLVER_KWARGS = dict(
    sigma=SIGMA,
    bg=BG,
    lam=LAM,
    n_iter=30,
    lm_iter=20,
    verbose=True,
)

# ── Generate data ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
H, W = IMG_SHAPE

isolated = np.array(
    [[10.0, 10.0], [10.0, 50.0], [50.0, 10.0], [50.0, 50.0]], dtype=np.float32
)
isolated_amp = np.array([1200.0, 900.0, 1500.0, 1100.0], dtype=np.float32)

pair_sep = SIGMA * 1.0  # pixels — 1 sigma separation
pairs_center = np.array(
    [[20.0, 32.0], [40.0, 25.0], [32.0, 50.0]], dtype=np.float32
)

close_pos, close_amp = [], []
for center in pairs_center:
    angle = rng.uniform(0, 2 * np.pi)
    offset = pair_sep / 2 * np.array([np.cos(angle), np.sin(angle)])
    close_pos.extend([center + offset, center - offset])
    base_amp = rng.uniform(800.0, 1400.0)
    close_amp.extend([base_amp, base_amp * rng.uniform(0.7, 1.3)])

close_pos = np.array(close_pos, dtype=np.float32)
close_amp = np.array(close_amp, dtype=np.float32)

gt_pos = np.concatenate([isolated, close_pos], axis=0)
gt_amp = np.concatenate([isolated_amp, close_amp])
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

# ── Print ground truth ────────────────────────────────────────────────────────

print(f"=== Crowded test: {n_gt} emitters ({len(isolated)} isolated + 3 close pairs) ===")
print(f"  Image: {H}x{W}, sigma={SIGMA}, bg={BG}, lam={LAM}")
print(f"  Pair separation: {pair_sep} px ({pair_sep/SIGMA:.2f} sigma)")
print(f"  Amplitude range: {gt_amp.min():.0f} – {gt_amp.max():.0f}")
print()
print("  Ground truth:")
for i in range(n_gt):
    tag = "isolated" if i < len(isolated) else f"pair {(i - len(isolated)) // 2 + 1}"
    print(f"    {i:2d}: ({gt_pos[i,0]:.2f}, {gt_pos[i,1]:.2f})  amp={gt_amp[i]:.0f}  [{tag}]")
print()


# ── Matching helper ───────────────────────────────────────────────────────────

def match_and_report(rec_pos, rec_amp, label):
    K_rec = len(rec_amp)
    print(f"\n{'='*60}")
    print(f"  {label}  —  recovered {K_rec} / {n_gt} emitters")
    print(f"{'='*60}")

    if K_rec == 0:
        print("  No emitters recovered.")
        return 0, 0

    dists = cdist(rec_pos, gt_pos)
    used_rec, used_gt, matches = set(), set(), []
    for _ in range(min(K_rec, n_gt)):
        best_d, best_ij = np.inf, (-1, -1)
        for i in range(K_rec):
            if i in used_rec:
                continue
            for j in range(n_gt):
                if j in used_gt:
                    continue
                if dists[i, j] < best_d:
                    best_d, best_ij = dists[i, j], (i, j)
        if best_d < 5.0:
            i, j = best_ij
            matches.append((i, j, best_d))
            used_rec.add(i)
            used_gt.add(j)

    iso_errors, pair_errors = [], []
    for i, j, d in matches:
        tag = "isolated" if j < len(isolated) else f"pair {(j - len(isolated)) // 2 + 1}"
        print(
            f"  rec {i}: ({rec_pos[i,0]:.2f}, {rec_pos[i,1]:.2f}) "
            f"<-> GT {j} ({gt_pos[j,0]:.2f}, {gt_pos[j,1]:.2f})  "
            f"dist={d:.3f} px  amp={rec_amp[i]:.0f} (GT {gt_amp[j]:.0f})  [{tag}]"
        )
        (iso_errors if j < len(isolated) else pair_errors).append(d)

    unmatched_rec = set(range(K_rec)) - used_rec
    unmatched_gt  = set(range(n_gt))  - used_gt
    for i in unmatched_rec:
        print(f"  rec {i}: ({rec_pos[i,0]:.2f}, {rec_pos[i,1]:.2f})  amp={rec_amp[i]:.0f}  [spurious]")
    for j in unmatched_gt:
        tag = "isolated" if j < len(isolated) else f"pair {(j - len(isolated)) // 2 + 1}"
        print(f"  GT {j}: ({gt_pos[j,0]:.2f}, {gt_pos[j,1]:.2f})  amp={gt_amp[j]:.0f}  [missed, {tag}]")

    if iso_errors:
        print(f"\n  Isolated : {len(iso_errors)}/{len(isolated)} matched, mean err {np.mean(iso_errors):.3f} px")
    if pair_errors:
        print(f"  Close pairs: {len(pair_errors)}/{len(close_pos)} emitters matched"
              + f", mean err {np.mean(pair_errors):.3f} px")
    else:
        print(f"  Close pairs: 0/{len(close_pos)} emitters matched")

    return len(iso_errors), len(pair_errors)


# ── Run both modes ────────────────────────────────────────────────────────────

# amp_cap ≈ 0.5–0.6 × expected single-emitter brightness.
# Capping at this level leaves enough residual flux at the second emitter
# location for the certificate to cross -1, while the tightened 0.5σ mask
# prevents the residual from being suppressed by the existing spike.
AMP_CAP = 600.0

REP = 200.0

print("\n" + "#"*60)
print("# Baseline: no cap, no repulsion")
print("#"*60)
amp_base, pos_base, _ = sfw_poisson(y, S_min=10.0, **SOLVER_KWARGS)
rec_pos_base = np.array(pos_base) if amp_base.size > 0 else np.zeros((0, 2))
rec_amp_base = np.array(amp_base) if amp_base.size > 0 else np.array([])
iso_base, pair_base = match_and_report(rec_pos_base, rec_amp_base, "no cap, no rep")

print("\n" + "#"*60)
print(f"# amp_cap={AMP_CAP:.0f} + rep_strength={REP:.0f}")
print("#"*60)
amp_cap_r, pos_cap_r, _ = sfw_poisson(
    y, S_min=10.0, amp_cap=AMP_CAP, rep_strength=REP, **SOLVER_KWARGS)
rec_pos_cap = np.array(pos_cap_r) if amp_cap_r.size > 0 else np.zeros((0, 2))
rec_amp_cap = np.array(amp_cap_r) if amp_cap_r.size > 0 else np.array([])
iso_cap, pair_cap = match_and_report(
    rec_pos_cap, rec_amp_cap, f"amp_cap={AMP_CAP:.0f} + rep={REP:.0f}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  {'':30s}  baseline    cap+rep")
print(f"  {'isolated matched':30s}  {iso_base}/{len(isolated)}           {iso_cap}/{len(isolated)}")
print(f"  {'close-pair emitters matched':30s}  {pair_base}/{len(close_pos)}           {pair_cap}/{len(close_pos)}")
print(f"  {'total recovered':30s}  {len(rec_amp_base)}/{n_gt}          {len(rec_amp_cap)}/{n_gt}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

ax = axes[0]
ax.imshow(clean_np, cmap="gray", origin="lower")
ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=8, mew=2, label="GT")
for center in pairs_center:
    circ = plt.Circle((center[1], center[0]), pair_sep, fill=False,
                       edgecolor="yellow", linewidth=1.5, linestyle="--")
    ax.add_patch(circ)
ax.set_title("Clean + GT")
ax.legend(fontsize=8)

ax = axes[1]
ax.imshow(np.array(y), cmap="gray", origin="lower")
ax.set_title("Observed (Poisson + readout)")

for ax, rec_pos, rec_amp, title in [
    (axes[2], rec_pos_base, rec_amp_base, f"baseline\n({len(rec_amp_base)}/{n_gt})"),
    (axes[3], rec_pos_cap,  rec_amp_cap,  f"cap={AMP_CAP:.0f}+rep={REP:.0f}\n({len(rec_amp_cap)}/{n_gt})"),
]:
    ax.imshow(np.array(y), cmap="gray", origin="lower")
    ax.plot(gt_pos[:, 1], gt_pos[:, 0], "rx", ms=10, mew=2, label="GT")
    if len(rec_amp) > 0:
        ax.plot(rec_pos[:, 1], rec_pos[:, 0], "c+", ms=10, mew=2, label="Recovered")
    for center in pairs_center:
        circ = plt.Circle((center[1], center[0]), pair_sep, fill=False,
                           edgecolor="yellow", linewidth=1.5, linestyle="--")
        ax.add_patch(circ)
    ax.set_title(title)
    ax.legend(fontsize=8)

for ax in axes:
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

fig.suptitle(
    f"Crowded case — 3 close pairs at {pair_sep} px ({pair_sep/SIGMA:.1f}σ), σ={SIGMA}",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("test_crowded_result.png", dpi=150)
print(f"\nPlot saved to test_crowded_result.png")
plt.show()
