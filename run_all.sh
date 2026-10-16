#!/usr/bin/env bash
# Full pipeline. Training is the slow part (~30-40 min per model on one CPU core);
# pretrained models are already in results/models, so you can skip straight to step 2.
set -e
cd "$(dirname "$0")/src"

if [ "$1" == "--train" ]; then
  # 1. offline training (uniform and causal differ only in --weighting)
  python train.py --weighting uniform --seed 0
  python train.py --weighting causal  --seed 0
fi

# 2. open-loop accuracy: within-hop error, chained rollout drift, encoding floor
python eval_rollout.py v2 uniform_s0 causal_s0

# 3. closed loop on the finite-difference plant
python run_closed_loop.py --controller pinn --model uniform_s0
python run_closed_loop.py --controller pinn --model causal_s0
python run_closed_loop.py --controller pinn --model uniform_s0 --bias
python run_closed_loop.py --controller pinn --model causal_s0  --bias
[ -f ../results/closed_loop/fd.json ] || python run_closed_loop.py --controller fd   # slow perfect-model reference

# 4. figures
python make_figures.py plant
python make_figures.py training uniform_s0 causal_s0
python make_figures.py rollout ../results/rollout_v2.json
python make_figures.py closed fig4_closed_loop.png fd pinn_uniform_s0 pinn_causal_s0
python make_figures.py closed fig5_closed_loop_bias.png fd pinn_uniform_s0_bias pinn_causal_s0_bias
