"""Evaluate checkpoints with the paper's protocol.

Paper protocol (Chi et al. 2023, Tables I/II): 50 environment initializations per checkpoint,
score = mean over episodes of the per-episode max reward (target-area coverage / 0.95, clipped),
report "max / average of the last 10 checkpoints", averaged over 3 training seeds.

Examples:
    # last 10 checkpoints of one run (paper-style)
    python scripts/evaluate.py --run runs/seed0 --last-k 10 --out results/seed0_eval.json
    # one checkpoint, DDIM with 10 steps, save 4 videos
    python scripts/evaluate.py --ckpt runs/seed0/ckpt_epoch04500.pt --method ddim --steps 10 --videos 4
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dprepro.policy import DiffusionPolicy  # noqa: E402
from dprepro.rollout import run_episode  # noqa: E402


def evaluate_ckpt(path, n_episodes, method, steps, num_samples, device, videos=0, video_dir=None,
                  start_seed=100000):
    policy, extra = DiffusionPolicy.load(path, device=device, use_ema=True)
    eps = []
    for i in range(n_episodes):
        render = i < videos
        r = run_episode(policy, seed=start_seed + i, method=method, num_inference_steps=steps,
                        num_samples=num_samples, render=render, sample_seed=i)
        if render and video_dir:
            import imageio
            os.makedirs(video_dir, exist_ok=True)
            imageio.mimsave(os.path.join(video_dir, f"{os.path.basename(path)[:-3]}_seed{start_seed+i}.mp4"),
                            r["frames"], fps=10)
        r.pop("frames")
        eps.append(r)
        print(f"  {os.path.basename(path)} seed={start_seed+i} score={r['max_reward']:.3f}", flush=True)
    lat = np.concatenate([e["latency_per_replan_s"] for e in eps])
    return {"ckpt": path, "epoch": extra.get("epoch"), "method": method, "steps": steps,
            "num_samples": num_samples, "n_episodes": n_episodes,
            "mean_score": float(np.mean([e["max_reward"] for e in eps])),
            "success_rate_cov95": float(np.mean([e["max_reward"] >= 1.0 for e in eps])),
            "latency_ms_median": float(np.median(lat) * 1000), "episodes": eps}


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--run", help="run directory; evaluates the last --last-k checkpoints")
    p.add_argument("--last-k", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--method", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--steps", type=int, default=None, help="DDIM inference steps")
    p.add_argument("--num-samples", type=int, default=1, help=">1 also records action-sample dispersion")
    p.add_argument("--videos", type=int, default=0)
    p.add_argument("--video-dir", default="results/videos")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpts = [args.ckpt] if args.ckpt else sorted(glob.glob(os.path.join(args.run, "ckpt_epoch*.pt")))[-args.last_k:]
    results = [evaluate_ckpt(c, args.episodes, args.method, args.steps, args.num_samples, args.device,
                             args.videos, args.video_dir) for c in ckpts]
    scores = [r["mean_score"] for r in results]
    summary = {"n_ckpts": len(results), "max_over_ckpts": float(np.max(scores)),
               "avg_over_ckpts": float(np.mean(scores)), "per_ckpt": results}
    print(f"\nmax/avg over {len(results)} ckpts: {summary['max_over_ckpts']:.3f} / {summary['avg_over_ckpts']:.3f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
