"""Bookkeeping utilities: pruning and merging of emitters."""

import mlx.core as mx
import numpy as np


def prune(amplitudes, positions, tol=1e-2):
    """Remove emitters with amplitude below tol."""
    mx.eval(amplitudes)
    mask_np = np.array(amplitudes) > tol
    if not mask_np.any():
        return mx.array([], dtype=mx.float32), mx.zeros((0, 2), dtype=mx.float32)
    idx = np.where(mask_np)[0]
    return amplitudes[mx.array(idx)], positions[mx.array(idx)]


def merge_nearby(amplitudes, positions, min_dist=1.0):
    """Merge emitters closer than min_dist by amplitude-weighted averaging."""
    mx.eval(amplitudes, positions)
    amp_np = np.array(amplitudes)
    pos_np = np.array(positions)
    K = len(amp_np)
    if K <= 1:
        return amplitudes, positions

    merged = np.ones(K, dtype=bool)
    new_amps, new_poss = [], []

    for i in range(K):
        if not merged[i]:
            continue
        # Find all unmerged spikes within min_dist of spike i
        group = [i]
        for j in range(i + 1, K):
            if not merged[j]:
                continue
            d = np.linalg.norm(pos_np[i] - pos_np[j])
            if d < min_dist:
                group.append(j)
                merged[j] = False

        total_amp = amp_np[group].sum()
        avg_pos = (amp_np[group, None] * pos_np[group]).sum(axis=0) / total_amp
        new_amps.append(total_amp)
        new_poss.append(avg_pos)

    return mx.array(np.array(new_amps, dtype=np.float32)), mx.array(
        np.array(new_poss, dtype=np.float32)
    )
