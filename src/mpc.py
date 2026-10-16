"""
Nonlinear MPC for outlet-concentration tracking with jacket temperature u.

At each sampling instant (one hop, dt = 0.25 residence times):
    min_{u_0..u_{Nc-1}}  sum_{j=1..Np} ((y_j + d - r)/y_s)^2  +  w_du * sum_{j} (du_j/u_s)^2
    s.t.  u_min <= u_j <= u_max,  u_j = u_{Nc-1} for j >= Nc  (move blocking)
where y_j is the predicted outlet concentration after hop j and d an optional output bias.

Two predictors share this exact problem:
  * PINNPredictor  : chained frozen PINN, exact gradients by autodiff (jax.grad through the chain)
  * FD model       : the finite-difference reactor, gradients by finite differences
                     (what you would do without a PINN; also the "perfect-model" reference)
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

Y_SCALE = 0.1   # outlet-concentration error scale
U_SCALE = 0.2   # input-move scale (= width of the input range)


class PINNMPC:
    def __init__(self, pred, cfg, u_lo, u_hi):
        self.pred, self.cfg = pred, cfg
        self.bounds = [(u_lo, u_hi)] * cfg.Nc
        Np, Nc, q, w = cfg.Np, cfg.Nc, cfg.q, cfg.r

        def J(u_free, a0, r, u_prev, d):
            u_seq = jnp.concatenate([u_free, jnp.repeat(u_free[-1], Np - Nc)])
            y = pred.rollout_cout(a0, u_seq) + d
            du = jnp.diff(jnp.concatenate([jnp.atleast_1d(u_prev), u_free]))
            return q * jnp.sum(((y - r) / Y_SCALE) ** 2) + w * jnp.sum((du / U_SCALE) ** 2)

        self.vg = jax.jit(jax.value_and_grad(J))
        self.n_evals = 0

    def solve(self, a0, r, u_prev, d=0.0, warm=None):
        x0 = np.full(self.cfg.Nc, u_prev) if warm is None else warm
        self.n_evals = 0

        def f(u):
            self.n_evals += 1
            v, g = self.vg(jnp.asarray(u), jnp.asarray(a0), r, u_prev, d)
            return float(v), np.asarray(g, dtype=float)

        t0 = time.perf_counter()
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=self.bounds,
                       options=dict(maxiter=100, ftol=1e-12, gtol=1e-8))
        return res.x, time.perf_counter() - t0, self.n_evals

    def predict(self, a0, u_free):
        cfg = self.cfg
        u_seq = np.concatenate([u_free, np.repeat(u_free[-1], cfg.Np - cfg.Nc)])
        return np.asarray(self.pred.rollout_cout(jnp.asarray(a0), jnp.asarray(u_seq)))


class FDMPC:
    def __init__(self, plant, cfg, dt):
        self.plant, self.cfg, self.dt = plant, cfg, dt
        self.bounds = [(plant.p.u_min, plant.p.u_max)] * cfg.Nc
        self.n_evals = 0

    def _traj(self, x0, u_free):
        cfg = self.cfg
        u_seq = np.concatenate([u_free, np.repeat(u_free[-1], cfg.Np - cfg.Nc)])
        y, x = [], x0
        for u in u_seq:
            x = self.plant.step(x, u, self.dt, rtol=1e-9, atol=1e-11)[-1]
            y.append(x[self.plant.N - 1])
        return np.array(y)

    def solve(self, x0, r, u_prev, d=0.0, warm=None):
        cfg = self.cfg
        x_init = np.full(cfg.Nc, u_prev) if warm is None else warm
        self.n_evals = 0

        def J(u):
            self.n_evals += 1
            y = self._traj(x0, u) + d
            du = np.diff(np.concatenate([[u_prev], u]))
            return cfg.q * np.sum(((y - r) / Y_SCALE) ** 2) + cfg.r * np.sum((du / U_SCALE) ** 2)

        t0 = time.perf_counter()
        res = minimize(J, x_init, method="L-BFGS-B", bounds=self.bounds,
                       options=dict(maxiter=100, ftol=1e-12, gtol=1e-8, eps=1e-6))
        return res.x, time.perf_counter() - t0, self.n_evals

    def predict(self, x0, u_free):
        return self._traj(x0, u_free)
