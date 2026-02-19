# Sliding Frank-Wolfe (SFW) for Gridless Sparse Spike Recovery

Based on Denoyelle et al., "Sliding Frank-Wolfe Algorithm" (2019).
See also: https://github.com/XeBasTeX/SFW-python

## The Problem

You observe a blurry 2D image `y(x)` on an N × N pixel grid. This image is a
sum of unknown point sources convolved with a known Gaussian PSF:

```
y(x) = Σᵢ aᵢ · h(x − xᵢ) + noise
```

where `h(x) = exp(−|x|² / (2σ²)) / Z` is the normalized Gaussian kernel.

The goal is to recover the number of sources, their positions `{xᵢ}` (which
are **not** restricted to the pixel grid), and their amplitudes `{aᵢ}`.

The optimization problem used here is:

```
minimize over a, x:  PoissonNLL(Σᵢ aᵢ h(· − xᵢ) + bg, y) + λ Σᵢ aᵢ
subject to          aᵢ ≥ 0
```

where `m = Σᵢ aᵢ δ(x − xᵢ)` is a discrete measure (a list of
amplitude–position pairs), and `λ` controls sparsity/complexity.

**This implementation** uses a Poisson negative log-likelihood with a
non-negative L1 amplitude step (proximal/FISTA), plus local bounded
Levenberg-Marquardt refinement of support locations.

---

## Key Objects

**Measure m** — a list of `(amplitude, position)` pairs `{(a₁, x₁), …, (aₖ, xₖ)}`.
Positions `xᵢ` are 2D continuous coordinates; amplitudes `aᵢ` are scalars.

**Forward operator Φ(m)** — predicts the blurry image:
```
Φ(m)(x) = Σᵢ aᵢ · exp(−|x − xᵢ|² / (2σ²)) / Z
```
evaluated at every pixel. Produces an H × W image.

**Adjoint operator Φ\*(r)** — given a residual image `r`, computes the
cross-correlation of `r` with the PSF `h` (implemented as `mx.conv2d`).
Intuitively: `Φ*(r)(x)` tells you how much the residual "looks like" a point
source at location `x`.

**Certificate η** — the dual variable for optimality testing:
```
η(x) = Φ*(1 − y / model)
```
In nonnegative BLASSO mode, one adds spikes while `min_x η(x) < -λ`
(equivalently `-min_x η(x)/λ > 1`).

---

## Algorithm (one iteration)

Start with a measure `mₖ` (initially empty).

1. **Source selection** — compute `η = Φ*(1 − y/model)`, take the
   best grid point, then run local continuous refinement from that start.

2. **Check stopping** — stop when the certificate condition is met.

3. **Add spike** — append `x*` with amplitude 1.0 to the measure.

4. **Projected FISTA** — optimize amplitudes with positions frozen under
   `a >= 0` with L1 penalty `λ * sum(a)`.

5. **Local descent** — bounded joint Levenberg-Marquardt on amplitudes and
   positions (Coleman-Li box constraints).

6. **Prune** — remove emitters with near-zero amplitude (`prune_tol`).

7. **Repeat** until the certificate condition is satisfied or `n_iter` is reached.

---

## Amplitude Step

Given current support, amplitudes are updated by solving:

```
minimize_a  PoissonNLL(Ha + bg, y) + λ * sum(a)
subject to  a >= 0
```
This uses positive soft-threshold proximal steps (monotone FISTA).

## Lam Continuation

For crowded scenes, use `lam_schedule=[lam_1, lam_2, ..., lam_S]` with
`lam_1 >= ... >= lam_S`:

1. Run SFW at `lam_1` from empty support.
2. Warm-start the next stage at `lam_2` from the previous stage solution.
3. Continue until `lam_S`.

This often reduces over-spawning early while still allowing support splitting
at the final, lower-regularized stage.

---

## Covariance Variant (reference)

An alternative formulation for fluorescence microscopy with blinking matches
the empirical covariance matrix `Rᵧ` instead of the mean image. The forward
operator becomes:

```
Λ(m) = Σᵢ aᵢ · vec(hᵢ) · vec(hᵢ)ᵀ
```

This is useful when individual source locations are better encoded in the
covariance than in the time-averaged image. Not currently implemented in
`sfwloc`.
