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

The optimization problem is:

```
minimize over m:  T_λ(m) = λ · Σ|aᵢ| + ½ · ‖y − Σᵢ aᵢ h(· − xᵢ)‖²
```

where `m = Σᵢ aᵢ δ(x − xᵢ)` is a discrete measure (a list of
amplitude–position pairs), and `λ` is a regularization parameter controlling
sparsity.

**This implementation** replaces the L1 penalty with an MCP (Minimax Concave
Penalty) and uses a Poisson negative log-likelihood instead of the squared
residual, making it appropriate for photon-counting fluorescence microscopy.

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
η(x) = (1/λ) · Φ*(1 − y / model)
```
The optimality condition is `η(x) ≥ −1` everywhere. If this holds, the
current measure is optimal and the algorithm stops.

---

## Algorithm (one iteration)

Start with a measure `mₖ` (initially empty).

1. **Certificate** — compute `η = (1/λ) Φ*(1 − y/model)` and find `x* = argmin η`.

2. **Check stopping** — if `min(η) ≥ −1`, the measure is optimal. Stop.

3. **Add spike** — append `x*` with amplitude 1.0 to the measure.

4. **FISTA** — optimize all amplitudes with positions frozen, using the MCP
   penalty (non-convex, reduces L1 bias for bright emitters).

5. **Levenberg-Marquardt refinement** — alternating block-coordinate LM:
   - LM on amplitudes (positions fixed, Coleman-Li box constraints)
   - LM on positions (amplitudes fixed, optional pairwise Gaussian repulsion)

6. **Prune** — remove emitters with amplitude below `S_min`.

7. **Repeat** — or stop early if emitter count is stable for 4 iterations.

---

## Super-resolved (Crowded) Mode

When emitters are separated by less than ~2σ, the standard solver merges
nearby pairs into a single over-bright spike. Two complementary parameters
address this:

**`amp_cap`** — amplitude ceiling applied after every FISTA step.
Limits per-emitter flux, leaving structured residual so the certificate can
propose a second nearby atom. The certificate exclusion zone is automatically
tightened from 1σ to 0.5σ when `amp_cap` is active.

Rule of thumb: `amp_cap ≈ 0.5 × A_max`

**`rep_strength`** — Gaussian pairwise repulsion added to the LM position
objective:
```
R = Σᵢ<ⱼ exp(−dᵢⱼ² / (2σ²))
```
Prevents LM from collapsing two nearby atoms back to their centroid.

Rule of thumb: `rep_strength ≈ 0.1–0.2 × A_single`

Both parameters are needed: `amp_cap` creates the residual; `rep_strength`
keeps the proposed atoms separated.

---

## MCP Penalty

The Minimax Concave Penalty (MCP) is used instead of L1 to recover unbiased
amplitude estimates above a threshold:

```
p(a; λ, γ) = λa − a²/(2γ)   for 0 ≤ a ≤ γλ
           = γλ²/2           for a > γλ
```

The shape parameter `γ = S_min / λ`. Amplitudes above `S_min` are recovered
without bias; those below are driven to zero and pruned.

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
