# Diffusion Policy on Push-T: a from-scratch reproduction + two extensions

Re-implementation of the CNN-based **Diffusion Policy** (Chi et al., RSS 2023) for the simulated
**Push-T** task, evaluated with the paper's protocol, plus two small studies:

1. **Sampler steps vs. latency** - how far can DDIM cut the 100-step DDPM sampler before task score drops?
2. **Action-sample dispersion as a failure signal** - does the spread of K sampled action chunks predict a failed episode?

> **Result:** 0.866 +/- 0.038 average target coverage on simulated Push-T (3 seeds x 10 checkpoints x 50 episodes),
> vs. 0.91 reported in the paper for the state-based CNN policy - close to, but below, the reported score.
> All numbers below were measured with the code in this repository on 2x NVIDIA RTX 6000 Ada (October 2026).

## Paper

- Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song.
  *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Official code (MIT): <https://github.com/real-stanford/diffusion_policy> - project page: <https://diffusion-policy.cs.columbia.edu/>

## What is reproduced, and what is not

| Item | Paper | This repo |
|---|---|---|
| Policy | CNN (1D temporal U-Net, FiLM) | same architecture, re-implemented (65.35M params) |
| Horizons To / Tp / Ta | 2 / 16 / 8 | 2 / 16 / 8 |
| Diffusion | 100 train steps, squaredcos schedule, eps-prediction | same |
| Optimizer | AdamW lr 1e-4, wd 1e-6, cosine, 500 warmup, EMA | same |
| Data | `pusht_cchi_v7_replay.zarr` (206 episodes in the file; official config: 2% val, 90 random train episodes, seed 42) | same file and split (`--official-split`) |
| Observations (Table I "state") | 9 block keypoints + agent xy (20-D) | **5-D state** (agent xy, block xy, angle), as in the authors' Colab |
| Simulator | authors' Push-T env | `gym-pusht` 0.1.6 port (pymunk 6) |
| Metric | max coverage reward over episode, 50 inits, max/avg of last 10 ckpts, 3 seeds | same protocol (`scripts/evaluate.py`) |

Because the observation space and simulator port differ slightly, treat the paper's number as a reference,
not a target to match exactly. For an exact, paper-config reproduction, also run the official repo (Track A below).

## Results

### Reproduction (paper protocol)

| Run | Push-T max (last 10 ckpts) | Push-T avg (last 10 ckpts) |
|---|---|---|
| Seed 0 | 0.863 | 0.854 |
| Seed 1 | 0.923 | 0.909 |
| Seed 2 | 0.846 | 0.835 |
| **This repo, mean +/- std (3 seeds)** | **0.877 +/- 0.040** | **0.866 +/- 0.038** |
| DiffusionPolicy-C, paper (Table I, state) | 0.95 | 0.91 |
| IBC, paper (Table I, state) | 0.90 | 0.84 |
| Hold-position floor (measured) | - | 0.139 |

Each seed: 4,500 epochs (~5.8 h on one RTX 6000 Ada), then 10 checkpoints x 50 test initializations (seeds 100000-100049).
The gap to the paper (~0.04 avg, ~0.07 max) is about one seed-standard-deviation. Possible causes, **not tested here**:
the 5-D state observation instead of 20-D keypoints, and the `gym-pusht` port instead of the authors' environment.

### Extension 1 - sampler steps vs latency (seed 0, epoch 4500, same 50 episodes per setting)

| Sampler | Score | Diff vs DDPM-100 [95% paired bootstrap CI] | Median latency / replan |
|---|---|---|---|
| DDPM-100 (paper) | 0.863 | - | 491.7 ms |
| DDIM-32 | 0.861 | -0.002 [-0.035, +0.029] | 155.6 ms |
| DDIM-16 | 0.859 | -0.004 [-0.039, +0.027] | 78.1 ms |
| DDIM-8 | 0.852 | -0.011 [-0.046, +0.020] | 39.6 ms |
| DDIM-4 | 0.820 | -0.043 [-0.081, -0.011] | 20.4 ms |
| DDIM-2 | 0.842 | -0.021 [-0.071, +0.028] | 10.8 ms |
| DDIM-1 | 0.152 | -0.711 [-0.797, -0.613] | 5.7 ms |

DDIM with 2, 8, 16 or 32 steps is not distinguishable from DDPM-100 at this sample size, at 3-45x lower latency;
one step fails (score near the 0.139 floor). The DDIM-4 drop is not monotonic with step count and was not investigated
(6 comparisons; could be chance). Limits: one checkpoint, one seed, 50 episodes (resolution ~+/-0.04).

### Extension 2 - does action-sample dispersion predict failure? (preliminary)

K = 16 action chunks per replan (DDIM-16), 200 held-out episodes (seeds 200000+), failure = score < 0.9 (43 failures).

| Episode statistic of dispersion | AUROC [95% bootstrap CI] |
|---|---|
| Mean | 0.487 [0.381, 0.590] |
| Max | 0.393 [0.297, 0.491] |
| **Median** | **0.791 [0.703, 0.875]** |
| Mean of first 5 replans | 0.574 [0.486, 0.665] |

Typical (median) dispersion is higher in failed episodes, while dispersion **peaks** are higher in successful ones,
consistent with peaks reflecting multimodal choices (pushing the T from either side) rather than uncertainty.
**Not yet established:** this is one checkpoint; failed episodes are longer than successful ones (possible confound);
and the episode median is only known after the episode, so it is not yet a runtime monitor.
`scripts/online_monitor.py` tests the signal using only replans seen so far; replication on seeds 1-2 is the next step.

Figures in `results/figures/`: `training_curves.png`, `ablation_steps.png`, `uncertainty_vs_score.png`.

## Setup

Quick path on a GPU server: `bash check_server.sh`, create the environment below, then `bash run_all.sh` inside tmux.
Checkpoints store EMA weights only (~261 MB each) and only the newest 10 are kept per seed (`--keep-last`), so plan for ~3 GB of disk per seed.

```bash
conda create -n dpr python=3.11 -y && conda activate dpr
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the CUDA build for your driver
pip install -r requirements.txt
python -m pytest -q tests            # 7 tests, < 10 s
```

Known pitfalls (found while building this repo):
- `gym-pusht` 0.1.6 crashes with **pymunk 7** (`add_collision_handler` removed) -> pin `pymunk<7`.
- Its keypoint observation mode calls `np.row_stack`, missing in recent NumPy 2.x -> this repo uses `obs_type="state"`.
- The official zarr is zarr-v2 format -> pin `zarr<3`.

## Data

```bash
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm pusht.zip && cd ..
# -> data/pusht/pusht_cchi_v7_replay.zarr
```

## Track B - this repo (from scratch)

```bash
# Smoke test (CPU, ~1 min): dummy data, tiny net. Scores are meaningless.
python scripts/make_dummy_dataset.py --out data/dummy.zarr
python scripts/train.py --data data/dummy.zarr --out runs/smoke --epochs 2 --small --workers 0

# Full run, 3 seeds (GPU). 4500 epochs as in the paper; log time per epoch and extrapolate first.
for s in 0 1 2; do python scripts/train.py --data data/pusht/pusht_cchi_v7_replay.zarr --out runs/seed$s --seed $s --official-split; done

# Paper protocol: last 10 checkpoints x 50 inits per seed
for s in 0 1 2; do python scripts/evaluate.py --run runs/seed$s --last-k 10 --out results/seed${s}_eval.json; done
python scripts/baseline_hold.py
```

## Track A - official code (exact paper config)

```bash
git clone https://github.com/real-stanford/diffusion_policy && cd diffusion_policy
mamba env create -f conda_environment.yaml && conda activate robodiff
# data as above, inside the official repo's data/ folder
wget -O image_pusht_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0
```

## Extensions

```bash
CK=runs/seed0/ckpt_epoch04500.pt
python scripts/ablate_inference_steps.py --ckpt $CK --ddim-steps 1 2 4 8 16 32 --out results/ablation_steps.json
python scripts/uncertainty_failure.py --ckpt $CK -K 16 --episodes 200 --fail-below 0.9 --out results/uncertainty.json
python scripts/make_figures.py --runs runs/seed0 runs/seed1 runs/seed2 \
  --evals results/seed0_eval.json results/seed1_eval.json results/seed2_eval.json \
  --ablation results/ablation_steps.json --uncertainty results/uncertainty.json --out results/figures
python scripts/evaluate.py --ckpt $CK --videos 4 --episodes 4 --video-dir results/videos
python scripts/summarize_extensions.py   # paired CIs + AUROC table
python scripts/online_monitor.py         # dispersion signal using only replans seen so far
```

The uncertainty study uses seeds 200000+ so it never touches the 100000+ test seeds.
Push-T is multimodal (the T can be pushed from either side), so high dispersion may reflect
multimodality instead of uncertainty; the script reports mean, max, median and early-window statistics with bootstrap CIs.

## Repository layout

```
dprepro/            model.py (U-Net), diffusion.py (DDPM/DDIM), dataset.py, policy.py (EMA, I/O), rollout.py (eval loop)
configs/            pusht_state.json
scripts/            train, evaluate, baseline_hold, ablate_inference_steps, uncertainty_failure, make_figures, make_dummy_dataset
tests/              unit tests (shapes, DDIM oracle recovery, padding, normalizer, AUROC)
results/            JSON results and figures; runs/*/ holds training logs and configs (checkpoints are not committed)
```

## Acknowledgements

Architecture, hyperparameters and the sequence-sampling scheme follow the paper and the authors'
MIT-licensed code and Colab notebook; this repository is an independent re-implementation.
