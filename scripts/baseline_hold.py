"""Reference floor: a 'hold position' policy (agent never moves) under the same protocol.
Because the score is the max coverage over the episode, it includes the coverage of the
initial state; this baseline shows how much score is obtained without any pushing.

    python scripts/baseline_hold.py --episodes 50
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dprepro.rollout import make_env  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--start-seed", type=int, default=100000)
    p.add_argument("--out", default="results/baseline_hold.json")
    a = p.parse_args()
    scores = []
    for i in range(a.episodes):
        env = make_env()
        obs, _ = env.reset(seed=a.start_seed + i)
        hold = obs[:2].astype(np.float32)
        best = 0.0
        for _ in range(300):
            obs, r, term, trunc, _ = env.step(hold)
            best = max(best, r)
            if term or trunc:
                break
        scores.append(float(best)); env.close()
    out = {"policy": "hold", "n_episodes": a.episodes, "mean_score": float(np.mean(scores)), "scores": scores}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"hold-position baseline mean score over {a.episodes} seeds: {out['mean_score']:.3f}")


if __name__ == "__main__":
    main()
