"""
Finite-difference model of the non-isothermal axial-dispersion tubular reactor.

This is the *ground-truth plant*. It is used for three things only:
  1. as the "real" process inside the closed loop,
  2. as the reference that PINN predictions are scored against,
  3. as the slow model-based MPC baseline (to show why the PINN is needed).
It is never used to produce training labels for the PINN.

Dimensionless model on z in [0, 1], velocity v = 1 (time unit = residence time):

    dC/dt = -dC/dz + (1/Pe_m) d2C/dz2 - k(T) C
    dT/dt = -dT/dz + (1/Pe_h) d2T/dz2 + beta k(T) C + gamma (u - T)
    k(T)  = k0 exp(-E / T)            with k0 = Da * exp(E)  (so k(1) = Da)

Danckwerts boundary conditions
    z = 0 :  C - (1/Pe_m) dC/dz = C_in ,   T - (1/Pe_h) dT/dz = T_in
    z = 1 :  dC/dz = 0 ,                   dT/dz = 0
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ReactorParams:
    Pe_m: float = 10.0      # mass Peclet number   (D = 1/Pe_m)
    Pe_h: float = 10.0      # heat Peclet number   (alpha = 1/Pe_h)
    Da: float = 0.8         # Damkohler number, k(T=1)
    E: float = 15.0         # dimensionless activation energy  E/(R T_ref)
    beta: float = 0.15      # adiabatic temperature-rise coefficient
    gamma: float = 3.0      # jacket heat-transfer coefficient
    C_in: float = 1.0
    T_in: float = 1.0
    u_min: float = 0.95     # jacket temperature bounds
    u_max: float = 1.15

    @property
    def D(self) -> float:
        return 1.0 / self.Pe_m

    @property
    def alpha(self) -> float:
        return 1.0 / self.Pe_h

    @property
    def k0(self) -> float:
        return self.Da * np.exp(self.E)

    def k(self, T):
        return self.Da * np.exp(self.E * (1.0 - 1.0 / T))


class FDReactor:
    """Method-of-lines discretisation with second-order central differences and
    ghost nodes for the Danckwerts conditions. State vector x = [C_0..C_{N-1}, T_0..T_{N-1}]."""

    def __init__(self, p: ReactorParams = ReactorParams(), N: int = 101):
        self.p = p
        self.N = N
        self.z = np.linspace(0.0, 1.0, N)
        self.h = self.z[1] - self.z[0]
        self._build_linear_operators()

    # ------------------------------------------------------------------ operators
    def _lin_op(self, diff: float):
        """Linear part  -d/dz + diff d2/dz2  with Danckwerts ghost nodes.
        Returns (sparse matrix A, inlet vector b) such that L[f] = A f + b * f_in."""
        N, h = self.N, self.h
        conv = 1.0 / (2 * h)
        dif = diff / h**2
        lo = dif + conv           # coefficient of f_{i-1}
        mid = -2 * dif            # coefficient of f_i
        hi = dif - conv           # coefficient of f_{i+1}
        A = sparse.lil_matrix((N, N))
        for i in range(1, N - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = lo, mid, hi
        # inlet ghost: f_{-1} = f_1 - 2h (f_0 - f_in)/diff
        g = 2 * h / diff
        A[0, 0] = mid - lo * g
        A[0, 1] = hi + lo
        b = np.zeros(N)
        b[0] = lo * g
        # outlet ghost: f_N = f_{N-2}
        A[N - 1, N - 2] = lo + hi
        A[N - 1, N - 1] = mid
        return A.tocsr(), b

    def _build_linear_operators(self):
        self.Ac, self.bc = self._lin_op(self.p.D)
        self.At, self.bt = self._lin_op(self.p.alpha)
        N = self.N
        self._jac_pattern = sparse.bmat(
            [[self.Ac != 0, sparse.eye(N)], [sparse.eye(N), (self.At != 0) + sparse.eye(N)]]
        ).astype(bool)

    # ------------------------------------------------------------------ dynamics
    def rhs(self, t, x, u):
        p, N = self.p, self.N
        C, T = x[:N], x[N:]
        r = p.k(T) * C
        dC = self.Ac @ C + self.bc * p.C_in - r
        dT = self.At @ T + self.bt * p.T_in + p.beta * r + p.gamma * (u - T)
        return np.concatenate([dC, dT])

    def jac(self, t, x, u):
        p, N = self.p, self.N
        C, T = x[:N], x[N:]
        k = p.k(T)
        dk = k * p.E / T**2
        J = sparse.bmat(
            [
                [self.Ac - sparse.diags(k), sparse.diags(-dk * C)],
                [sparse.diags(p.beta * k), self.At + sparse.diags(p.beta * dk * C - p.gamma)],
            ]
        )
        return J.tocsc()

    def step(self, x0, u, dt, n_out: int = 2, rtol=1e-7, atol=1e-9):
        """Integrate with constant jacket temperature u for dt. Returns states at n_out times."""
        t_eval = np.linspace(0.0, dt, n_out)
        sol = solve_ivp(self.rhs, (0.0, dt), x0, method="BDF", t_eval=t_eval,
                        args=(u,), jac=self.jac, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y.T  # (n_out, 2N)

    def simulate(self, x0, u_seq, dt):
        """Piecewise-constant input sequence; returns array of states at hop boundaries."""
        xs = [np.asarray(x0, float)]
        for u in u_seq:
            xs.append(self.step(xs[-1], u, dt)[-1])
        return np.array(xs)

    def steady_state(self, u, x_guess=None, t_final=30.0):
        """Steady state by integrating to t_final (robust for this stable plant)."""
        if x_guess is None:
            x_guess = np.concatenate([np.full(self.N, self.p.C_in), np.full(self.N, self.p.T_in)])
        return self.step(x_guess, u, t_final)[-1]

    def split(self, x):
        return x[..., : self.N], x[..., self.N:]

    def outlet_C(self, x):
        return x[..., self.N - 1]


if __name__ == "__main__":
    plant = FDReactor()
    t0 = time.perf_counter()
    for u in np.linspace(plant.p.u_min, plant.p.u_max, 9):
        xs = plant.steady_state(u)
        C, T = plant.split(xs)
        print(f"u={u:.3f}  C_out={C[-1]:.4f}  T_max={T.max():.4f} at z={plant.z[T.argmax()]:.2f}  T_out={T[-1]:.4f}")
    print(f"time {time.perf_counter()-t0:.2f}s")
