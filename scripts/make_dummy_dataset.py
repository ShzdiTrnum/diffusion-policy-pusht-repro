"""SMOKE TEST ONLY: writes a small zarr with the same schema as the official Push-T data, using
random-walk actions in gym-pusht. It contains no expert behavior; a policy trained on it is
meaningless. Use it to check that install, training, evaluation and plotting run end to end.

    python scripts/make_dummy_dataset.py --out data/dummy.zarr --episodes 4 --steps 60
"""
import argparse
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dprepro.rollout import make_env  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/dummy.zarr")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=60)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    states, actions, ends = [], [], []
    env = make_env()
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=ep)
        target = obs[:2].copy()
        for _ in range(a.steps):
            target = np.clip(target + rng.normal(0, 15, 2), 0, 512).astype(np.float32)
            states.append(obs.copy()); actions.append(target.copy())
            obs, *_ = env.step(target)
        ends.append(len(states))
    root = zarr.open(a.out, mode="w")
    root.create_group("data"); root.create_group("meta")
    root["data"]["state"] = np.asarray(states, np.float32)
    root["data"]["action"] = np.asarray(actions, np.float32)
    root["meta"]["episode_ends"] = np.asarray(ends, np.int64)
    print(f"wrote {a.out}: {len(states)} steps, {len(ends)} episodes (DUMMY DATA)")


if __name__ == "__main__":
    main()
