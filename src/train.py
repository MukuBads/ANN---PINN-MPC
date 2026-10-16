"""
Offline loop (left half of the Review-1 flowchart):
sample collocation points -> forward pass -> autodiff PDE residual -> (causal) weights
-> Adam -> L-BFGS (float64) -> freeze weights.

    python train.py --weighting uniform --seed 0
    python train.py --weighting causal  --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import PINNConfig, TrainConfig, as_dict
from pinn import init_params, make_batch, make_losses, save_model
from setup_problem import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weighting", default="uniform", choices=["uniform", "causal"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam_iters", type=int, default=None)
    ap.add_argument("--lbfgs_iters", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--v1", action="store_true", help="plain (non-BC) profile encoding, as in the first runs")
    args = ap.parse_args()

    pcfg = PINNConfig(bc_encoding=not args.v1)
    tcfg = TrainConfig(weighting=args.weighting, seed=args.seed)
    if args.adam_iters is not None:
        tcfg = replace(tcfg, adam_iters=args.adam_iters)
    if args.lbfgs_iters is not None:
        tcfg = replace(tcfg, lbfgs_iters=args.lbfgs_iters)
    if args.eps is not None:
        tcfg = replace(tcfg, eps=args.eps)
    causal = tcfg.weighting == "causal"
    name = f"{tcfg.weighting}_s{tcfg.seed}{args.tag}"
    out_dir = os.path.join(ROOT, "results", "models")
    os.makedirs(out_dir, exist_ok=True)

    plant, enc, sampler, phys, sc, norm = build(pcfg, seed=tcfg.seed)
    field, total_loss, slice_losses, causal_weights = make_losses(phys, sc, norm)

    n_in = 3 + 2 * pcfg.K
    sizes = [n_in] + [pcfg.hidden] * pcfg.depth + [2]
    params = init_params(jax.random.PRNGKey(tcfg.seed), sizes)
    n_params = sum(int(W.size + b.size) for W, b in params)
    print(f"[{name}] network {sizes}  trainable parameters = {n_params}")

    rng = np.random.default_rng(1000 + tcfg.seed)
    M = tcfg.M

    # ---------------------------------------------------------------- Adam
    sched = optax.exponential_decay(tcfg.lr, tcfg.lr_decay_every, tcfg.lr_decay, staircase=True)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    @jax.jit
    def adam_step(params, opt_state, batch, eps):
        (loss, aux), g = jax.value_and_grad(
            lambda p: total_loss(p, batch, M, eps, tcfg.lam_ic, tcfg.lam_bc, causal),
            has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss, aux

    hist = dict(it=[], loss=[], L_pde=[], L_ic=[], L_bc=[], w_min=[], w_slices=[], L_slices=[], time=[], eps=[], eps_change=[])
    t0 = time.perf_counter()

    def log(it, loss, aux, phase):
        hist["it"].append(it)
        hist["loss"].append(float(loss))
        for k in ("L_pde", "L_ic", "L_bc", "w_min"):
            hist[k].append(float(aux[k]))
        hist["w_slices"].append(np.asarray(aux["w_mean_slices"]).tolist())
        hist["L_slices"].append(np.asarray(aux["L_slices"]).tolist())
        hist["time"].append(time.perf_counter() - t0)
        hist["eps"].append(eps_list[eps_i])
        print(f"[{name}] {phase} it {it:6d}  loss {float(loss):.3e}  L_pde {float(aux['L_pde']):.3e}  "
              f"L_ic {float(aux['L_ic']):.3e}  L_bc {float(aux['L_bc']):.3e}  w_min {float(aux['w_min']):.3f}  "
              f"t {time.perf_counter()-t0:.0f}s", flush=True)

    # epsilon annealing (Wang et al. 2024): move to a stricter epsilon once every
    # time slice is switched on (min weight > 0.99), so causality keeps acting late in training
    eps_list = [e * tcfg.eps for e in (1.0, 10.0, 100.0)]
    eps_i = 0
    for it in range(tcfg.adam_iters):
        batch = make_batch(rng, sampler, tcfg.B, M, tcfg.P, tcfg.n_ic, tcfg.n_bc)
        params, opt_state, loss, aux = adam_step(params, opt_state, batch, eps_list[eps_i])
        if causal and float(aux["w_min"]) > 0.99 and eps_i < len(eps_list) - 1:
            eps_i += 1
            print(f"[{name}] it {it}: all slices active -> epsilon = {eps_list[eps_i]}", flush=True)
            hist["eps_change"].append([it, eps_list[eps_i]])
        if it % tcfg.log_every == 0 or it == tcfg.adam_iters - 1:
            log(it, loss, aux, "adam ")
        if it > 0 and it % 1000 == 0:
            save_model(os.path.join(out_dir, f"{name}_ckpt.pkl"), params, dict(it=it, sizes=sizes))

    # ---------------------------------------------------------------- L-BFGS
    # Full-batch quasi-Newton on a fixed, larger collocation set. Causal weights are
    # frozen within each block (L-BFGS needs a fixed objective) and refreshed between blocks.
    if tcfg.lbfgs_iters > 0:
        big = make_batch(rng, sampler, 4 * tcfg.B, M, tcfg.P, tcfg.n_ic, tcfg.n_bc)
        it_global = tcfg.adam_iters
        n_blocks = int(np.ceil(tcfg.lbfgs_iters / tcfg.lbfgs_refresh))
        for blk in range(n_blocks):
            if causal:
                W = causal_weights(slice_losses(params, big, M), eps_list[eps_i])
            else:
                W = None

            def f(p):
                return total_loss(p, big, M, eps_list[eps_i], tcfg.lam_ic, tcfg.lam_bc, causal, W_fixed=W)[0]

            lb = optax.lbfgs(memory_size=20)
            st = lb.init(params)
            vg = optax.value_and_grad_from_state(f)

            @jax.jit
            def lb_step(params, st):
                val, g = vg(params, state=st)
                upd, st = lb.update(g, st, params, value=val, grad=g, value_fn=f)
                return optax.apply_updates(params, upd), st, val

            for k in range(tcfg.lbfgs_refresh):
                params_new, st_new, val = lb_step(params, st)
                if not np.isfinite(float(val)):
                    print("L-BFGS produced non-finite loss; stopping block")
                    break
                params, st = params_new, st_new
                it_global += 1
                if k % tcfg.log_every == 0 or k == tcfg.lbfgs_refresh - 1:
                    loss, aux = total_loss(params, big, M, eps_list[eps_i], tcfg.lam_ic, tcfg.lam_bc, causal)
                    log(it_global, loss, aux, "lbfgs")

    train_time = time.perf_counter() - t0
    meta = dict(name=name, sizes=sizes, n_params=n_params, train_time_s=train_time, eps_final=eps_list[eps_i],
                config=as_dict(pcfg, tcfg),
                norm=dict(a_center=np.asarray(norm.a_center).tolist(), a_half=np.asarray(norm.a_half).tolist()))
    save_model(os.path.join(out_dir, f"{name}.pkl"), params, meta)
    with open(os.path.join(out_dir, f"{name}_history.json"), "w") as f:
        json.dump(dict(meta=meta, hist=hist), f)
    print(f"[{name}] done in {train_time/60:.1f} min")


if __name__ == "__main__":
    main()
