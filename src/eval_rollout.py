"""
Open-loop accuracy tests against the finite-difference ground truth.

  (1) within-hop error vs normalised time tau   -> the direct test of the causality hypothesis
  (2) chained-rollout error vs number of hops   -> drift that the MPC predictor experiences
  (3) encoding-only floor: FD model that is projected onto the Legendre basis after every hop,
      i.e. the error a *perfect* PINN would still have because of the state encoding

Test initial states are genuine transients (not steady states), produced by a random warm-up
under a test seed that the training sampler never sees.

    python eval_rollout.py v2 uniform_s0 causal_s0
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from config import PINNConfig
from pinn import load_model
from predictor import PINNPredictor
from setup_problem import build, pcfg_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_test_set(plant, n_traj=40, H=20, warmup=3, seed=12345):
    rng = np.random.default_rng(seed)
    p, dt = plant.p, PINNConfig().dt
    tests = []
    for _ in range(n_traj):
        x = plant.steady_state(rng.uniform(p.u_min, p.u_max))
        for _ in range(warmup):
            x = plant.step(x, rng.uniform(p.u_min, p.u_max), dt)[-1]
        u_seq = [rng.uniform(p.u_min, p.u_max)]
        for _ in range(H - 1):
            u_seq.append(u_seq[-1] if rng.uniform() < 0.5 else rng.uniform(p.u_min, p.u_max))
        u_seq = np.array(u_seq)
        X = plant.simulate(x, u_seq, dt)                     # (H+1, 2N)
        tests.append(dict(x0=x, u=u_seq, X=X))
    return tests


def main(names, out_tag):
    pcfg = pcfg_for(load_model(os.path.join(ROOT, "results", "models", f"{names[0]}.pkl"))[1])
    plant, enc, sampler, phys, sc, norm = build(pcfg)
    N, dt, n_tau = plant.N, pcfg.dt, 9
    taus = np.linspace(0, 1, n_tau)
    t0 = time.perf_counter()
    tests = make_test_set(plant)
    H = len(tests[0]["u"])
    print(f"test set: {len(tests)} trajectories x {H} hops ({time.perf_counter()-t0:.1f}s)")

    # ---------- encoding-only floor (FD dynamics + projection each hop)
    enc_err = np.zeros((len(tests), H))
    for i, tr in enumerate(tests):
        x = tr["x0"]
        for h in range(H):
            Cd, Td = enc.decode(enc.encode(*plant.split(x)))
            x = plant.step(np.concatenate([Cd, Td]), tr["u"][h], dt)[-1]
            enc_err[i, h] = abs(x[N - 1] - tr["X"][h + 1, N - 1])

    # ---------- within-hop truth: FD from the *decoded* IC (isolates PINN error from encoding)
    pairs = []
    for tr in tests:
        for h in range(0, H, 4):
            pairs.append((tr["X"][h], tr["u"][h]))
    within_truth, within_a = [], []
    for x, u in pairs:
        a = enc.encode(*plant.split(x))
        xd = np.concatenate(enc.decode(a))
        within_truth.append(plant.step(xd, u, dt, n_out=n_tau))
        within_a.append(a)
    within_truth = np.array(within_truth)                    # (n_pairs, n_tau, 2N)

    results = dict(H=H, dt=dt, taus=taus.tolist(), n_traj=len(tests), n_pairs=len(pairs),
                   encoding_only=dict(cout_abs_err_mean=enc_err.mean(0).tolist()), models={})

    for name in names:
        pred = PINNPredictor(os.path.join(ROOT, "results", "models", f"{name}.pkl"), phys, sc, norm, enc)
        # (1) within-hop
        eC = np.zeros((len(pairs), n_tau)); eT = np.zeros((len(pairs), n_tau))
        for j, ((x, u), a) in enumerate(zip(pairs, within_a)):
            for t_i, tau in enumerate(taus):
                C, T = pred.profiles(a, u, float(tau))
                Ct, Tt = plant.split(within_truth[j, t_i])
                eC[j, t_i] = np.sqrt(np.mean((np.asarray(C) - Ct) ** 2))
                eT[j, t_i] = np.sqrt(np.mean((np.asarray(T) - Tt) ** 2))
        # (2) chained rollout
        cout_err = np.zeros((len(tests), H)); prof_err = np.zeros((len(tests), H))
        for i, tr in enumerate(tests):
            C0, T0 = plant.split(tr["x0"])
            a0 = pred.encode(C0, T0)
            c_out, C, T = pred.rollout(a0, tr["u"])
            Ct, Tt = plant.split(tr["X"][1:])
            cout_err[i] = np.abs(np.asarray(c_out) - Ct[:, -1])
            prof_err[i] = np.sqrt(np.mean((np.asarray(C) - Ct) ** 2, axis=1))
        # timing: one Np=8 prediction
        a0 = pred.encode(*plant.split(tests[0]["x0"]))
        pred.rollout_cout(a0, tests[0]["u"][:8]).block_until_ready()
        t1 = time.perf_counter()
        for _ in range(200):
            pred.rollout_cout(a0, tests[0]["u"][:8]).block_until_ready()
        t_pinn = (time.perf_counter() - t1) / 200
        results["models"][name] = dict(
            within_rmse_C=eC.mean(0).tolist(), within_rmse_T=eT.mean(0).tolist(),
            rollout_cout_abs_err_mean=cout_err.mean(0).tolist(),
            rollout_cout_abs_err_p90=np.percentile(cout_err, 90, axis=0).tolist(),
            rollout_profile_rmse_C_mean=prof_err.mean(0).tolist(),
            time_8hop_prediction_s=t_pinn)
        print(f"{name}: within-hop RMSE_C tau=0 {eC.mean(0)[0]:.2e} tau=0.5 {eC.mean(0)[n_tau//2]:.2e} tau=1 {eC.mean(0)[-1]:.2e} | "
              f"C_out err hop1 {cout_err.mean(0)[0]:.2e} hop8 {cout_err.mean(0)[7]:.2e} hop20 {cout_err.mean(0)[-1]:.2e} | "
              f"8-hop prediction {t_pinn*1e3:.2f} ms")

    # FD timing for the same 8-hop prediction
    t1 = time.perf_counter()
    for _ in range(10):
        plant.simulate(tests[0]["x0"], tests[0]["u"][:8], dt)
    results["time_8hop_prediction_FD_s"] = (time.perf_counter() - t1) / 10
    print(f"encoding-only floor: C_out err hop1 {enc_err.mean(0)[0]:.2e} hop20 {enc_err.mean(0)[-1]:.2e}; "
          f"FD 8-hop prediction {results['time_8hop_prediction_FD_s']*1e3:.1f} ms")

    with open(os.path.join(ROOT, "results", f"rollout_{out_tag}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    # python eval_rollout.py <out_tag> <model> [<model> ...]
    main(sys.argv[2:], sys.argv[1])
