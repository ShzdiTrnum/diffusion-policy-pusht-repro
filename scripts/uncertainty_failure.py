"""Extension 2 (research question): does the spread of sampled action chunks predict failure?

At every replanning step the policy draws K action chunks from independent noise; sample 0 is
executed and the dispersion (pixels) of all K over the executed window is logged. We then ask
whether episode-level dispersion statistics separate failed from successful episodes (AUROC),
and whether the first N replanning steps already give an early warning.

Caveat built into the analysis: Push-T is multimodal (the agent may go around the T on either
side), so high dispersion can reflect legitimate multimodality rather than uncertainty. We
therefore report both the raw statistic and a mode-robust variant (median over replans).

    python scripts/uncertainty_failure.py --ckpt runs/seed0/ckpt_epoch04500.pt -K 16 \
        --episodes 200 --fail-below 0.9 --out results/uncertainty.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dprepro.policy import DiffusionPolicy  # noqa: E402
from dprepro.rollout import run_episode  # noqa: E402


def auroc(scores, labels):
    """Probability that a random positive (failure) scores higher than a random negative.
    Mann-Whitney U formulation with tie handling; returns nan if one class is empty."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def bootstrap_ci(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores), np.asarray(labels)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(scores), len(scores))
        v = auroc(scores[idx], labels[idx])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("-K", "--num-samples", type=int, default=16)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--start-seed", type=int, default=200000, help="disjoint from the 100000+ test seeds")
    p.add_argument("--fail-below", type=float, default=0.9, help="episode score below this = failure")
    p.add_argument("--early-replans", type=int, default=5)
    p.add_argument("--method", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--out", default="results/uncertainty.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    policy, _ = DiffusionPolicy.load(args.ckpt, device=args.device)
    eps = []
    for i in range(args.episodes):
        r = run_episode(policy, seed=args.start_seed + i, method=args.method, num_inference_steps=args.steps,
                        num_samples=args.num_samples, sample_seed=i)
        r.pop("frames")
        d = np.asarray(r["dispersion_per_replan"])
        eps.append({"seed": r["seed"], "score": r["max_reward"], "disp": d.tolist(),
                    "disp_mean": float(d.mean()), "disp_max": float(d.max()), "disp_median": float(np.median(d)),
                    "disp_early_mean": float(d[: args.early_replans].mean())})
        print(f"seed={r['seed']} score={r['max_reward']:.3f} disp_mean={d.mean():.2f}", flush=True)

    labels = np.array([e["score"] < args.fail_below for e in eps])
    report = {"ckpt": args.ckpt, "K": args.num_samples, "method": args.method, "steps": args.steps,
              "fail_below": args.fail_below, "n_episodes": len(eps), "n_failures": int(labels.sum()),
              "auroc": {}}
    for key in ["disp_mean", "disp_max", "disp_median", "disp_early_mean"]:
        s = [e[key] for e in eps]
        report["auroc"][key] = {"auroc": auroc(s, labels), "ci95": bootstrap_ci(s, labels)}
    print(json.dumps(report["auroc"], indent=1))
    report["episodes"] = eps
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
