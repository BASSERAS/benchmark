#!/usr/bin/env bash
set -euo pipefail

source /home/samer/venvs/mfc/bin/activate

run_dir="/home/samer/scenarios/deep-mkv-gen-path-dt/runs/garch_tree_guyon_yfree_seed0_20260809"

python /home/samer/scenarios/deep-mkv-gen-path-dt/experiments/scripts/run_production_garch_tree.py \
  --phase exact \
  --run-dir "${run_dir}" \
  --device cuda:0 \
  --seed 0 \
  --horizon 6 \
  --lambda-scale 50 \
  --kappa-scale 100 \
  --eta 1 \
  --reference-kind guyon \
  --max-iterations 600 \
  --swd-projections 512

python /home/samer/scenarios/deep-mkv-gen-path-dt/experiments/scripts/run_production_garch_tree.py \
  --phase deep-mkv \
  --run-dir "${run_dir}" \
  --device cuda:0 \
  --seed 0 \
  --horizon 6 \
  --lambda-scale 50 \
  --kappa-scale 100 \
  --eta 1 \
  --reference-kind guyon \
  --adjoint-weight 0 \
  --adjoint-noise-weight 1 \
  --steps 3000 \
  --hidden-dim 96 \
  --log-every 100 \
  --exact-tree-replay \
  --oracle-ce-targets \
  --swd-projections 512
