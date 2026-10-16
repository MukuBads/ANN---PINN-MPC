"""
Where do initial profiles for training come from?

The PINN must work for any state the closed loop can visit, so each collocation instance
needs an initial profile (2K Legendre coefficients) and a jacket temperature u.
No transient trajectory and no label is used. Initial profiles are drawn from the plant's
*operating envelope*: equilibrium (steady-state) profiles of the same PDE across the input
range, blended pairwise and perturbed, plus a share drawn uniformly from a widened box
around that envelope.
"""
from __future__ import annotations

import numpy as np

from basis import StateEncoder
from reactor import FDReactor


class InstanceSampler:
    def __init__(self, plant: FDReactor, enc: StateEncoder, n_ss: int = 41,
                 widen: float = 0.5, sigma: float = 0.15, p_box: float = 0.25, seed: int = 0):
        p = plant.p
        self.u_lo, self.u_hi = p.u_min, p.u_max
        self.u_grid = np.linspace(p.u_min, p.u_max, n_ss)
        xs, guess = [], None
        for u in self.u_grid:
            guess = plant.steady_state(u, guess)
            xs.append(guess)
        xs = np.array(xs)
        C, T = plant.split(xs)
        self.enc = enc
        self.A_ss = enc.encode(C, T)                             # (n_ss, 2K)
        lo, hi = self.A_ss.min(0), self.A_ss.max(0)
        span = np.maximum(hi - lo, 0.02)                         # floor so tiny modes still vary
        self.box_lo = lo - widen * span
        self.box_hi = hi + widen * span
        self.span = span
        self.sigma = sigma
        self.p_box = p_box
        self.rng = np.random.default_rng(seed)

    # normalisation used for the network inputs
    @property
    def a_center(self):
        return 0.5 * (self.box_lo + self.box_hi)

    @property
    def a_half(self):
        return 0.5 * (self.box_hi - self.box_lo)

    def ss_coeffs(self, u):
        """Steady-state coefficients at arbitrary u (linear interpolation of the table)."""
        u = np.atleast_1d(u)
        return np.stack([np.interp(u, self.u_grid, self.A_ss[:, j]) for j in range(self.A_ss.shape[1])], -1)

    def sample(self, n):
        rng = self.rng
        ua = rng.uniform(self.u_lo, self.u_hi, n)
        ub = rng.uniform(self.u_lo, self.u_hi, n)
        lam = rng.uniform(0, 1, (n, 1))
        a = (1 - lam) * self.ss_coeffs(ua) + lam * self.ss_coeffs(ub)
        a += self.sigma * self.span * rng.uniform(-1, 1, a.shape)
        box = rng.uniform(self.box_lo, self.box_hi, a.shape)
        use_box = rng.uniform(0, 1, (n, 1)) < self.p_box
        a = np.where(use_box, box, a)
        a = self.enc.project(a)          # v2: every training IC satisfies the BCs (no-op in v1)
        u = rng.uniform(self.u_lo, self.u_hi, n)
        return a, u
