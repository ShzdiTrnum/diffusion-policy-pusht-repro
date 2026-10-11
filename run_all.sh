#!/usr/bin/env bash
# Full pipeline on the GPU server. Run inside tmux:  bash run_all.sh 2>&1 | tee run_all.log
# Steps are skipped if their output already exists, so you can re-run after an interruption
# (an interrupted training seed restarts from epoch 1; delete its folder first).
set -euo pipefail
DATA=data/pusht/pusht_cchi_v7_replay.zarr
SEEDS="${SEEDS:-0 1 2}"

python -m pytest -q tests

if [ ! -d "$DATA" ]; then
  mkdir -p data && (cd data && wget -q https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && unzip -q pusht.zip && rm pusht.zip)
fi
[ -d "$DATA" ] || { echo "dataset not found at $DATA - check the unzipped folder name"; exit 1; }

python scripts/baseline_hold.py
for s in $SEEDS; do
  [ -f runs/seed$s/ckpt_epoch04500.pt ] || python scripts/train.py --data $DATA --out runs/seed$s --seed $s --official-split
  [ -f results/seed${s}_eval.json ] || python scripts/evaluate.py --run runs/seed$s --last-k 10 --out results/seed${s}_eval.json
done

CK=runs/seed0/ckpt_epoch04500.pt
[ -f results/ablation_steps.json ] || python scripts/ablate_inference_steps.py --ckpt $CK --out results/ablation_steps.json
[ -f results/uncertainty.json ] || python scripts/uncertainty_failure.py --ckpt $CK -K 16 --episodes 200 --out results/uncertainty.json
python scripts/evaluate.py --ckpt $CK --episodes 4 --videos 4 --video-dir results/videos
python scripts/make_figures.py --runs $(for s in $SEEDS; do printf "runs/seed%s " $s; done) \
  --evals $(for s in $SEEDS; do printf "results/seed%s_eval.json " $s; done) \
  --ablation results/ablation_steps.json --uncertainty results/uncertainty.json --out results/figures
echo "DONE. See results/figures/results_table.md"
