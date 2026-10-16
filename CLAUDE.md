# Project handoff: Causality-aware PINN-MPC for a tubular reactor

ANN course project (ICE, NIT Trichy): Mukund Dinesh Badsewal + Revant Sinha.
**Review 2 is on 16 Oct 2026.** It needs a working model, plus the progress made and the
difficulties faced. Review 1 (research only) proposed putting a control-oriented PINN inside MPC
and comparing **uniform vs causal temporal weighting** (Wang et al. 2024) on closed-loop tracking
error.

Stack: Python + JAX (float64) + optax + scipy. PyTorch was unavailable in the build environment,
and the reference causal-PINN code (jaxpi) is JAX. `pip install -r requirements.txt`, then run
everything from `src/`. On the current Mac only `python3` is on PATH (no `python`); a venv in
`.venv` is recommended because the global env has an unrelated numpy<2 conflict (opensourceleg).

## What exists and works
- `src/reactor.py`: finite-difference plant (101 nodes, Danckwerts BCs, BDF + sparse Jacobian).
  Parameters: Pe=10, Da=0.8, E=15, beta=0.15, gamma=3, u in [0.95, 1.15]. Control target: outlet C.
- `src/basis.py`: Legendre profile encoder, K=6 per field (12 state inputs). The **v2** encoder is
  BC-consistent: an equality-constrained least-squares (KKT) fit that imposes Danckwerts at z=0
  and zero gradient at z=1. `PINNConfig.bc_encoding=True` is the default; pass `--v1` to
  `train.py` for the old plain encoder.
- `src/sampling.py`: training ICs come from the steady-state envelope (blend + perturb) plus 25%
  box samples, projected onto the BC constraint set in v2. No labels are used.
- `src/pinn.py`: 15 -> 5x64 tanh -> 2 network, 17,794 params. Output = IC profile + scaled
  correction (soft IC). Per-instance causal weights. Batch = 32 instances x 16 slices x 8 points.
- `src/train.py`: Adam 8000 (lr 1e-3, x0.7 every 1500) then L-BFGS 600 (causal weights frozen
  per 200-step block). eps annealing 1 -> 10 -> 100 when w_min > 0.99. **5-6 min per model on the
  current Mac** (~37 min on the original build machine). Checkpoint every 1000 iterations.
- `src/predictor.py`: frozen PINN chained with lax.scan; differentiable.
- `src/mpc.py`: PINN-MPC (jax.grad gradients, L-BFGS-B) and FD-MPC (finite-difference gradients).
  Np=8 hops of dt=0.25, Nc=3, q=100, r=1, Y_SCALE=0.1, U_SCALE=0.2. Optional filtered output-bias
  correction (`--bias`, filter gain 0.3).
- `src/run_closed_loop.py`: 64 steps; setpoints 0.354 -> 0.15 -> 0.45 -> 0.08 -> 0.30.
- `src/eval_rollout.py <out_tag> <models...>`: within-hop error vs tau, chained drift over 20 hops,
  encoding-only floor, timing.
- `src/make_figures.py`: figures. `rollout` now plots mean over seeds with a min-max band.
- `README.md`: article-style write-up, **fully filled** (results Sections 5-6, v1/v2 table 7.1).

## Results (v2 = BC-consistent encoder, seeds 0-2, mean ± sd)
| | uniform v2 | causal v2 | FD-MPC (perfect model) |
|---|---|---|---|
| Within-hop RMSE of C at tau=1 | 2.75e-3 ± 0.57e-3 | 2.89e-3 ± 0.37e-3 | - |
| Outlet-C error after 8 chained hops | 5.1e-3 ± 2.0e-3 | 5.1e-3 ± 0.9e-3 | - |
| Closed-loop IAE, no bias correction | 0.326 ± 0.005 | 0.322 ± 0.004 | 0.306 |
| Closed-loop IAE, filtered bias | 0.318 ± 0.004 | 0.318 ± 0.002 | - |
| Mean offset, filtered bias | 5.1e-4 | 5.2e-4 | ~0 |
| Mean MPC solve time | ~0.035-0.038 s | ~0.035-0.038 s | 8.15 s |

- **Conclusion: causal and uniform are statistically tied.** Seed 0 alone favoured causal by
  20-30% open-loop; seeds 1-2 reversed it. Causal has 2-6x smaller seed-to-seed spread (hint only).
- Causal w_min reaches ~0.94 (v1: last slice 0.75) but never passes 0.99, so eps stays at 1.
- Causal ends with higher unweighted L_pde (4.4e-3 vs 3.1e-3) under equal budget.
- 8-hop prediction 1.0 ms (PINN) vs 35 ms (FD). Encoding-only floor 3.9e-5 (v1 encoder: ~3e-6).
- v1 (plain encoder, seed 0) for reference: IAE 0.477 / 0.605 (uniform / causal), 8-hop error
  2.47e-2 / 3.25e-2. Cause of v1 causal under-performance: training ICs violated the Danckwerts
  inlet BC, so slice 1 kept the largest residual and gated later slices.
- Data: `results/rollout_v2.json` (seed 0), `results/rollout_v2_seeds.json` (all six),
  `results/closed_loop/pinn_{uniform,causal}_s{0,1,2}[_bias].json`, logs `results/logs/train_*_v2.out`.
- Figures: `results/figures/fig1..fig4*.png` (v2), v1 in `results/figures/v1/`.

## Next steps
1. Review 2 deck (in progress as a claude.ai Slides artifact): working model, progress,
   difficulties. Review 1 style: navy/orange, section labels.
2. Optional: eps sweep / larger eps (annealing never triggers); 5-10 seeds to test the
   variance-reduction hint; equal-residual comparison; stiffer kinetics (higher E, Pe).

## Notes
- Reference 6 (Patel, Bhartiya & Gudi, JPC 128, 2023) has an unconfirmed title. Check it.
- The float64 claim cites Xu et al., "FP64 is All You Need", NeurIPS 2025, arXiv:2505.10949.
