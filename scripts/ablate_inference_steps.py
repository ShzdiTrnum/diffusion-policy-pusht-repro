"""Extension 1: accuracy vs. latency of the denoising sampler.

Evaluates one checkpoint with full DDPM (100 steps, the paper's sim setting) and DDIM with
k steps, on the same 50 test seeds, and records mean score and median replanning latency.

    python scripts/ablate_inference_steps.py --ckpt runs/seed0/ckpt_epoch04500.pt \
        --ddim-steps 1 2 4 8 16 32 --out results/ablation_steps.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import evaluate_ckpt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ddim-steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--skip-ddpm", action="store_true")
    p.add_argument("--out", default="results/ablation_steps.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    settings = ([] if args.skip_ddpm else [("ddpm", None)]) + [("ddim", k) for k in args.ddim_steps]
    rows = []
    for method, k in settings:
        r = evaluate_ckpt(args.ckpt, args.episodes, method, k, 1, args.device)
        rows.append({"method": method, "steps": k if k else 100, "mean_score": r["mean_score"],
                     "success_rate_cov95": r["success_rate_cov95"], "latency_ms_median": r["latency_ms_median"],
                     "per_episode": [e["max_reward"] for e in r["episodes"]]})
        print(rows[-1]["method"], rows[-1]["steps"], rows[-1]["mean_score"], rows[-1]["latency_ms_median"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"ckpt": args.ckpt, "device": args.device, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
