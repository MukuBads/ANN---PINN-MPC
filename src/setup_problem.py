"""Builds the shared objects (plant, basis, sampler, physics) identically for every script."""
from basis import LegendreBasis, StateEncoder
from config import PINNConfig, REACTOR
from pinn import Norm, Physics, Scales
from reactor import FDReactor
from sampling import InstanceSampler


def build(pcfg: PINNConfig = PINNConfig(), seed: int = 0, N: int = 101):
    plant = FDReactor(REACTOR, N=N)
    basis = LegendreBasis(pcfg.K, n_grid=N)
    p = REACTOR
    enc = StateEncoder(basis, pcfg.bc_encoding, D=p.D, Cin=p.C_in, alpha=p.alpha, Tin=p.T_in)
    sampler = InstanceSampler(plant, enc, seed=seed)
    phys = Physics(D=p.D, alpha=p.alpha, Da=p.Da, E=p.E, beta=p.beta, gamma=p.gamma,
                   C_in=p.C_in, T_in=p.T_in, u_lo=p.u_min, u_hi=p.u_max)
    sc = Scales(K=pcfg.K, dt=pcfg.dt, sC=pcfg.sC, sT=pcfg.sT)
    norm = Norm(sampler.a_center, sampler.a_half)
    return plant, enc, sampler, phys, sc, norm


def pcfg_for(model_meta):
    """PINNConfig a saved model was trained with (v1 models predate bc_encoding)."""
    from dataclasses import replace
    c = model_meta["config"]["PINNConfig"]
    return replace(PINNConfig(), **{**c, "bc_encoding": c.get("bc_encoding", False)})
