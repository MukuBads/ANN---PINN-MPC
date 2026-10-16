"""All experiment settings in one place. Uniform and causal runs differ ONLY in `weighting`."""
from dataclasses import dataclass, field, asdict

from reactor import ReactorParams


@dataclass(frozen=True)
class PINNConfig:
    # profile encoding: K shifted-Legendre coefficients per field (C and T)
    K: int = 6
    # network: inputs = z, tau, u, 2K coefficients  -> 3 + 2K = 15
    hidden: int = 64
    depth: int = 5
    # hop length (one MPC sampling period), in residence times
    dt: float = 0.25
    # output scaling: C = C0(z) + sC * N_C,  T = T0(z) + sT * N_T
    sC: float = 0.3
    sT: float = 0.1
    # v2: encode profiles with Danckwerts/outlet BCs imposed exactly (v1 = plain least squares)
    bc_encoding: bool = True


@dataclass(frozen=True)
class TrainConfig:
    weighting: str = "uniform"     # "uniform" | "causal"
    seed: int = 0
    # collocation batch = B instances x M time slices x P points  (resampled every step)
    B: int = 32
    M: int = 16
    P: int = 8
    n_ic: int = 32                 # IC points per instance
    n_bc: int = 16                 # boundary points per instance per boundary
    lam_ic: float = 10.0
    lam_bc: float = 10.0
    eps: float = 1.0               # causality parameter epsilon
    adam_iters: int = 8000
    lr: float = 1e-3
    lr_decay_every: int = 1500
    lr_decay: float = 0.7
    lbfgs_iters: int = 600
    lbfgs_refresh: int = 200       # causal weights frozen within each L-BFGS block
    log_every: int = 100


@dataclass(frozen=True)
class MPCConfig:
    Np: int = 8                    # prediction horizon (hops)
    Nc: int = 3                    # free moves (move blocking: last move held)
    q: float = 100.0               # tracking weight on outlet concentration
    r: float = 1.0                 # move-suppression weight on (du)^2
    bias_correction: bool = False  # output-disturbance (offset-free) correction
    bias_filter: float = 0.3       # first-order filter gain on the disturbance estimate


def as_dict(*cfgs):
    out = {}
    for c in cfgs:
        out[type(c).__name__] = asdict(c)
    return out


REACTOR = ReactorParams()
