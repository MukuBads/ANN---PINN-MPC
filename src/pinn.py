"""
Control-oriented, physics-informed network for the tubular reactor (JAX, float64).

    inputs  : z, tau = t/dt, u, a  (a = 2K Legendre coefficients of the initial C and T profiles)
    outputs : C(z, t), T(z, t) on one hop t in [0, dt]

    C = C0(z; a) + sC * N_C(z, tau, u, a)          (output = initial profile + learned change)
    T = T0(z; a) + sT * N_T(z, tau, u, a)

The initial condition is still *soft*: the IC loss drives N(z, 0) -> 0. The skip term only gives
the network the initial profile for free, so its capacity goes to the dynamics. Uniform and
causal runs use the identical architecture.

Loss (per instance b, time slice i, collocation point p):
    L_pde[b, i] = mean_p ( rC^2 + rT^2 )                  scaled residuals
    uniform :  L_pde = mean_{b,i}  L_pde[b, i]
    causal  :  L_pde = mean_{b,i}  w[b, i] L_pde[b, i],   w[b, i] = exp(-eps * sum_{j<i} L_pde[b, j])
                                                         (stop-gradient on w; Wang et al. 2024)
    total   =  L_pde + lam_ic L_ic + lam_bc L_bc
Causal weights are computed per instance (per initial profile + input), because causality is a
property of each trajectory, not of the batch.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------------- network
def init_params(key, sizes):
    """Xavier/Glorot-normal weights, zero biases."""
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        std = jnp.sqrt(2.0 / (din + dout))
        params.append((std * jax.random.normal(sub, (din, dout), dtype=jnp.float64),
                       jnp.zeros(dout, dtype=jnp.float64)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def legendre(z, K):
    """Shifted Legendre P_k(2z-1), k < K, by the three-term recurrence (differentiable in z)."""
    x = 2.0 * z - 1.0
    P = [jnp.ones_like(x), x]
    for n in range(1, K - 1):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return jnp.stack(P[:K], axis=-1)


@dataclass(frozen=True)
class Physics:
    D: float
    alpha: float
    Da: float
    E: float
    beta: float
    gamma: float
    C_in: float
    T_in: float
    u_lo: float
    u_hi: float


@dataclass(frozen=True)
class Scales:
    K: int
    dt: float
    sC: float
    sT: float


class Norm:
    """Input normalisation for coefficients (fixed by the sampler's box)."""

    def __init__(self, a_center, a_half):
        self.a_center = jnp.asarray(a_center)
        self.a_half = jnp.asarray(a_half)


# ------------------------------------------------------------------------- field
def make_field(phys: Physics, sc: Scales, norm: Norm):
    K = sc.K
    u_c = 0.5 * (phys.u_lo + phys.u_hi)
    u_h = 0.5 * (phys.u_hi - phys.u_lo)

    def field(params, z, tau, u, a):
        """Single point -> array([C, T])."""
        feat = jnp.concatenate([jnp.stack([2 * z - 1, 2 * tau - 1, (u - u_c) / u_h]),
                                (a - norm.a_center) / norm.a_half])
        n = mlp(params, feat)
        P = legendre(z, K)
        C0 = P @ a[:K]
        T0 = P @ a[K:]
        return jnp.stack([C0 + sc.sC * n[0], T0 + sc.sT * n[1]])

    return field


def make_losses(phys: Physics, sc: Scales, norm: Norm):
    field = make_field(phys, sc, norm)
    scale = jnp.array([sc.sC, sc.sT])

    def k_of(T):
        return phys.Da * jnp.exp(phys.E * (1.0 - 1.0 / T))

    def residual_point(params, z, tau, u, a):
        # forward-mode autodiff: one nested jvp gives (f, f_z, f_zz); one jvp gives f_t
        f_tau = jax.jvp(lambda t_: field(params, z, t_, u, a), (tau,), (1.0,))[1]
        fz = lambda z_: jax.jvp(lambda zz: field(params, zz, tau, u, a), (z_,), (1.0,))
        (CT, f_z), (_, f_zz) = jax.jvp(fz, (z,), (1.0,))
        C, T = CT[0], CT[1]
        r = k_of(T) * C
        dC_dt = f_tau[0] / sc.dt
        dT_dt = f_tau[1] / sc.dt
        rC = dC_dt - (-f_z[0] + phys.D * f_zz[0] - r)
        rT = dT_dt - (-f_z[1] + phys.alpha * f_zz[1] + phys.beta * r + phys.gamma * (u - T))
        # scale: residual in units of (normalised output) per (normalised time)
        return jnp.stack([rC, rT]) * sc.dt / scale

    def bc_point(params, tau, u, a):
        """Danckwerts inlet and zero-gradient outlet residuals (scaled)."""
        f0 = field(params, 0.0, tau, u, a)
        f0z = jax.jvp(lambda zz: field(params, zz, tau, u, a), (0.0,), (1.0,))[1]
        f1z = jax.jvp(lambda zz: field(params, zz, tau, u, a), (1.0,), (1.0,))[1]
        inlet = jnp.stack([f0[0] - phys.D * f0z[0] - phys.C_in,
                           f0[1] - phys.alpha * f0z[1] - phys.T_in])
        outlet = f1z
        return jnp.concatenate([inlet / scale, outlet / scale])

    def ic_point(params, z, u, a):
        P = legendre(z, sc.K)
        f = field(params, z, 0.0, u, a)
        return (f - jnp.stack([P @ a[:sc.K], P @ a[sc.K:]])) / scale

    # vmaps: instances (B) x points
    res_inst = jax.vmap(residual_point, in_axes=(None, 0, 0, None, None))       # over points
    res_batch = jax.vmap(res_inst, in_axes=(None, 0, 0, 0, 0))                 # over instances
    bc_batch = jax.vmap(jax.vmap(bc_point, in_axes=(None, 0, None, None)), in_axes=(None, 0, 0, 0))
    ic_batch = jax.vmap(jax.vmap(ic_point, in_axes=(None, 0, None, None)), in_axes=(None, 0, 0, 0))

    def slice_losses(params, batch, M):
        """L_pde[b, i] for B instances and M time slices."""
        z, tau, u, a = batch["z"], batch["tau"], batch["u"], batch["a"]   # z,tau: (B, M*P)
        r = res_batch(params, z, tau, u, a)                                  # (B, M*P, 2)
        B = z.shape[0]
        sq = jnp.sum(r**2, axis=-1).reshape(B, M, -1)
        return jnp.mean(sq, axis=-1)                                         # (B, M)

    def causal_weights(L, eps):
        cum = jnp.cumsum(L, axis=1) - L                                      # exclusive sum_{j<i}
        return jax.lax.stop_gradient(jnp.exp(-eps * cum))

    def total_loss(params, batch, M, eps, lam_ic, lam_bc, causal, W_fixed=None):
        L = slice_losses(params, batch, M)
        if causal:
            W = causal_weights(L, eps) if W_fixed is None else W_fixed
        else:
            W = jnp.ones_like(L)
        L_pde_w = jnp.mean(W * L)
        L_ic = jnp.mean(ic_batch(params, batch["z_ic"], batch["u"], batch["a"]) ** 2)
        L_bc = jnp.mean(bc_batch(params, batch["tau_bc"], batch["u"], batch["a"]) ** 2)
        total = L_pde_w + lam_ic * L_ic + lam_bc * L_bc
        aux = dict(L_pde=jnp.mean(L), L_pde_w=L_pde_w, L_ic=L_ic, L_bc=L_bc,
                   w_min=jnp.min(jnp.mean(W, axis=0)), w_mean_slices=jnp.mean(W, axis=0),
                   L_slices=jnp.mean(L, axis=0))
        return total, aux

    return field, total_loss, slice_losses, causal_weights


# ------------------------------------------------------------------------- batches
def make_batch(rng: np.random.Generator, sampler, B, M, P, n_ic, n_bc):
    a, u = sampler.sample(B)
    z = rng.uniform(0, 1, (B, M * P))
    slice_idx = np.repeat(np.arange(M), P)[None, :]
    tau = (slice_idx + rng.uniform(0, 1, (B, M * P))) / M
    return dict(z=jnp.asarray(z), tau=jnp.asarray(tau), u=jnp.asarray(u), a=jnp.asarray(a),
                z_ic=jnp.asarray(rng.uniform(0, 1, (B, n_ic))),
                tau_bc=jnp.asarray(rng.uniform(0, 1, (B, n_bc))))


# ------------------------------------------------------------------------- io
def save_model(path, params, meta):
    with open(path, "wb") as f:
        pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), meta=meta), f)


def load_model(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, d["params"]), d["meta"]
