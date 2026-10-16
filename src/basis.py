"""
Profile encoding: a spatial profile f(z) on [0, 1] is represented by K shifted-Legendre
coefficients,  f(z) ~ sum_k a_k P_k(2z - 1).

Why this exists: the deck's network takes an "initial-profile parameter y0". After one hop the
reactor state is two whole curves C(z), T(z), so y0 has to be a vector that can describe any
reachable profile. Encoding -> PINN hop -> decode on a grid -> re-encode is what makes
chaining possible.

Two encoders:
  * plain least squares (v1)
  * BC-consistent least squares (v2, default): the fit is constrained so the encoded profile
    satisfies the Danckwerts inlet condition  f(0) - D f'(0) = f_in  and the outlet condition
    f'(1) = 0 exactly. Both are linear in the coefficients, so this is an equality-constrained
    least-squares problem with a closed-form (KKT) solution, and the encoder stays linear:
    a = M f + m.  Without it, initial profiles fed to the PINN contradict its own boundary
    conditions at the inlet corner (see README, Section 7).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as npleg


class LegendreBasis:
    def __init__(self, K: int, n_grid: int = 101):
        self.K = K
        self.z = np.linspace(0.0, 1.0, n_grid)
        self.Phi = self.eval(self.z)                    # (n_grid, K)
        self.pinv = np.linalg.pinv(self.Phi)            # least-squares projector (K, n_grid)

    def eval(self, z):
        """Basis matrix  Phi[i, k] = P_k(2 z_i - 1)."""
        x = 2.0 * np.asarray(z) - 1.0
        return np.stack([npleg.legval(x, np.eye(self.K)[k]) for k in range(self.K)], axis=-1)

    def eval_dz(self, z):
        """d/dz of each basis function (chain rule factor 2)."""
        x = 2.0 * np.asarray(z) - 1.0
        return np.stack([2.0 * npleg.legval(x, npleg.legder(np.eye(self.K)[k])) for k in range(self.K)], -1)

    def encode(self, f_grid):
        """Profile values on self.z (..., n_grid) -> coefficients (..., K)."""
        return f_grid @ self.pinv.T

    def decode(self, a, z=None):
        Phi = self.Phi if z is None else self.eval(z)
        return a @ Phi.T

    # ---------------------------------------------------------- BC-consistent encoder
    def bc_constraints(self, D, f_in):
        """G a = g  <=>  f(0) - D f'(0) = f_in  and  f'(1) = 0."""
        P0, dP0 = self.eval(0.0), self.eval_dz(0.0)
        dP1 = self.eval_dz(1.0)
        G = np.stack([P0 - D * dP0, dP1])
        g = np.array([f_in, 0.0])
        return G, g

    def bc_encoder(self, D, f_in):
        """Returns (M, m) with a = M f + m the BC-constrained least-squares fit."""
        G, g = self.bc_constraints(D, f_in)
        K = self.K
        KKT = np.block([[2 * self.Phi.T @ self.Phi, G.T], [G, np.zeros((2, 2))]])
        Kinv = np.linalg.inv(KKT)
        M = Kinv[:K, :K] @ (2 * self.Phi.T)
        m = Kinv[:K, K:] @ g
        return M, m

    def bc_project(self, a, D, f_in):
        """Minimum-norm projection of coefficient vectors onto {G a = g}."""
        G, g = self.bc_constraints(D, f_in)
        corr = (a @ G.T - g) @ np.linalg.inv(G @ G.T).T
        return a - corr @ G


class StateEncoder:
    """Encodes (C, T) profiles to 2K coefficients, plain or BC-consistent."""

    def __init__(self, basis: LegendreBasis, bc: bool, D=None, Cin=None, alpha=None, Tin=None):
        self.basis, self.bc, self.K = basis, bc, basis.K
        if bc:
            self.MC, self.mC = basis.bc_encoder(D, Cin)
            self.MT, self.mT = basis.bc_encoder(alpha, Tin)
            self._proj = (D, Cin, alpha, Tin)
        else:
            self.MC = self.MT = basis.pinv
            self.mC = self.mT = np.zeros(basis.K)

    def encode(self, C, T):
        return np.concatenate([C @ self.MC.T + self.mC, T @ self.MT.T + self.mT], axis=-1)

    def decode(self, a, z=None):
        K = self.K
        return self.basis.decode(a[..., :K], z), self.basis.decode(a[..., K:], z)

    def project(self, a):
        if not self.bc:
            return a
        D, Cin, alpha, Tin = self._proj
        K = self.K
        return np.concatenate([self.basis.bc_project(a[..., :K], D, Cin),
                               self.basis.bc_project(a[..., K:], alpha, Tin)], axis=-1)


def encode_state(basis: LegendreBasis, C, T):
    return np.concatenate([basis.encode(C), basis.encode(T)], axis=-1)


def decode_state(basis: LegendreBasis, coeffs, z=None):
    K = basis.K
    return basis.decode(coeffs[..., :K], z), basis.decode(coeffs[..., K:], z)
