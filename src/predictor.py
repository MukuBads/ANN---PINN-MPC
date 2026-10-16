"""
The frozen PINN used as an MPC predictor (right half of the Review-1 flowchart).

One hop:  coefficients a_k, input u_k  ->  PINN at tau = 1 on a z-grid  ->  C, T profiles
          ->  least-squares re-encode  ->  a_{k+1}
Chaining hops with lax.scan gives a prediction of any length; the whole chain is
differentiable, so the MPC gets exact gradients of its cost w.r.t. the input sequence.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pinn import load_model, make_field


class PINNPredictor:
    def __init__(self, model_path, phys, sc, norm, enc):
        self.params, self.meta = load_model(model_path)
        self.field = make_field(phys, sc, norm)
        self.K = sc.K
        self.enc = enc
        self.zg = jnp.asarray(enc.basis.z)
        MC, mC = jnp.asarray(enc.MC), jnp.asarray(enc.mC)
        MT, mT = jnp.asarray(enc.MT), jnp.asarray(enc.mT)
        field, params, zg = self.field, self.params, self.zg

        def profiles(a, u, tau, params=params):
            out = jax.vmap(lambda z: field(params, z, tau, u, a))(zg)
            return out[:, 0], out[:, 1]

        def hop(a, u):
            C, T = profiles(a, u, 1.0)
            return jnp.concatenate([MC @ C + mC, MT @ T + mT]), (C[-1], C, T)

        def rollout(a0, u_seq):
            aN, (c_out, C, T) = jax.lax.scan(hop, a0, u_seq)
            return c_out, C, T

        def rollout_cout(a0, u_seq):
            return rollout(a0, u_seq)[0]

        self.profiles = jax.jit(profiles)
        self.hop = jax.jit(hop)
        self.rollout = jax.jit(rollout)
        self.rollout_cout = jax.jit(rollout_cout)

    def encode(self, C, T):
        return self.enc.encode(C, T)
