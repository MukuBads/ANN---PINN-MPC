"""
Closed-loop experiment (right half of the Review-1 flowchart), on the finite-difference plant.

    measure state -> MPC proposes input sequence -> roll predictor forward -> score vs setpoint
    -> apply first move only -> next sample

    python run_closed_loop.py --controller pinn --model uniform_s0
    python run_closed_loop.py --controller pinn --model causal_s0
    python run_closed_loop.py --controller fd                  # perfect-model reference (slow)
    add --bias for offset-free (output-disturbance) correction
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np

from config import MPCConfig, PINNConfig, as_dict
from mpc import FDMPC, PINNMPC
from setup_problem import build, pcfg_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_STEPS = 64          # 64 x 0.25 = 16 residence times


def setpoint(k):
    if k < 4:
        return 0.354
    if k < 20:
        return 0.15
    if k < 36:
        return 0.45
    if k < 52:
        return 0.08
    return 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["pinn", "fd"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--bias", action="store_true")
    args = ap.parse_args()

    pcfg = PINNConfig()
    if args.controller == "pinn":
        from pinn import load_model
        pcfg = pcfg_for(load_model(os.path.join(ROOT, "results", "models", f"{args.model}.pkl"))[1])
    mcfg = replace(MPCConfig(), bias_correction=args.bias)
    plant, enc, sampler, phys, sc, norm = build(pcfg)
    N, dt = plant.N, pcfg.dt

    if args.controller == "pinn":
        from predictor import PINNPredictor
        pred = PINNPredictor(os.path.join(ROOT, "results", "models", f"{args.model}.pkl"), phys, sc, norm, enc)
        ctrl = PINNMPC(pred, mcfg, plant.p.u_min, plant.p.u_max)
        label = f"pinn_{args.model}"
    else:
        ctrl = FDMPC(plant, mcfg, dt)
        label = "fd"
    label += "_bias" if args.bias else ""

    x = plant.steady_state(1.0)
    u_prev, warm, d = 1.0, None, 0.0
    y_pred_next = None
    log = dict(t=[], y=[], r=[], u=[], solve_time=[], n_evals=[], pred_err_mean=[], d=[])
    for k in range(N_STEPS):
        r = setpoint(k)
        C, T = plant.split(x)
        y_meas = C[-1]
        if args.bias and y_pred_next is not None:
            # filtered output-disturbance estimate (raw one-step model error is noisy during
            # transients; unfiltered it drove the input into a bound-to-bound limit cycle)
            a_f = mcfg.bias_filter
            d = (1 - a_f) * d + a_f * (y_meas - y_pred_next)
        state = enc.encode(C, T) if args.controller == "pinn" else x
        u_free, t_solve, n_ev = ctrl.solve(state, r, u_prev, d, warm)
        # in-loop predictor quality: predicted vs true response to the chosen plan
        y_hat = ctrl.predict(state, u_free)
        if args.controller == "pinn":
            u_seq = np.concatenate([u_free, np.repeat(u_free[-1], mcfg.Np - mcfg.Nc)])
            y_true = plant.simulate(x, u_seq, dt)[1:, N - 1]
            log["pred_err_mean"].append(float(np.mean(np.abs(y_hat - y_true))))
        y_pred_next = float(y_hat[0])
        u = float(u_free[0])
        x = plant.step(x, u, dt)[-1]
        warm = np.concatenate([u_free[1:], u_free[-1:]])
        u_prev = u
        log["t"].append((k + 1) * dt); log["y"].append(float(x[N - 1])); log["r"].append(r)
        log["u"].append(u); log["solve_time"].append(t_solve); log["n_evals"].append(n_ev)
        log["d"].append(float(d))
        print(f"[{label}] k={k:2d}  r={r:.3f}  y={x[N-1]:.4f}  u={u:.4f}  solve {t_solve*1e3:7.1f} ms  evals {n_ev}", flush=True)

    y, rr = np.array(log["y"]), np.array([setpoint(k) for k in range(N_STEPS)])
    err = y - rr
    summary = dict(IAE=float(np.sum(np.abs(err)) * dt), ISE=float(np.sum(err**2) * dt),
                   mean_solve_time_s=float(np.mean(log["solve_time"])),
                   max_solve_time_s=float(np.max(log["solve_time"])),
                   mean_inloop_pred_err=float(np.mean(log["pred_err_mean"])) if log["pred_err_mean"] else None,
                   total_variation_u=float(np.sum(np.abs(np.diff([1.0] + log["u"])))))
    # steady-state offset: mean |error| over last 4 samples of each setpoint segment
    seg_ends = [19, 35, 51, 63]
    summary["offset_mean"] = float(np.mean([np.mean(np.abs(err[e - 3:e + 1])) for e in seg_ends]))
    print(label, json.dumps(summary))
    os.makedirs(os.path.join(ROOT, "results", "closed_loop"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "closed_loop", f"{label}.json"), "w") as f:
        json.dump(dict(label=label, summary=summary, log=log, config=as_dict(mcfg)), f)


if __name__ == "__main__":
    main()
