"""Closed-loop evaluation in gym-pusht, following the reference protocol:
receding-horizon control (observe To=2 frames, predict Tp=16 actions, execute Ta=8),
max 300 env steps, score = max reward over the episode, where
reward = clip(coverage / 0.95, 0, 1). Test seeds start at 100000 (as in the reference runner).
"""
import collections
import time

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import numpy as np


def make_env(render=False):
    kwargs = dict(obs_type="state", disable_env_checker=True)
    if render:
        kwargs["render_mode"] = "rgb_array"
    return gym.make("gym_pusht/PushT-v0", **kwargs)


def dispersion(samples: np.ndarray, start: int, n_exec: int) -> float:
    """Spread of K sampled action chunks over the executed window, in pixels:
    sqrt of the mean (over time) of the trace of the 2x2 sample covariance."""
    seg = samples[:, start:start + n_exec]  # (K, Ta, 2)
    if seg.shape[0] < 2:
        return float("nan")
    var = seg.var(axis=0, ddof=1).sum(-1)  # (Ta,)
    return float(np.sqrt(var.mean()))


def run_episode(policy, seed, method="ddpm", num_inference_steps=None, num_samples=1,
                max_steps=300, render=False, net=None, sample_seed=None):
    import torch
    cfg = policy.cfg
    To, Ta = cfg["obs_horizon"], cfg["action_horizon"]
    env = make_env(render)
    obs, info = env.reset(seed=seed)
    obs_deque = collections.deque([obs] * To, maxlen=To)
    gen = None
    if sample_seed is not None:
        gen = torch.Generator(device=policy.device).manual_seed(sample_seed)
    rewards, frames, disp, latencies, coverages = [], [], [], [], []
    if render:
        frames.append(env.render())
    steps, done = 0, False
    while not done:
        t0 = time.perf_counter()
        samples = policy.predict_action_samples(np.stack(obs_deque), num_samples=num_samples, method=method,
                                                num_inference_steps=num_inference_steps, net=net, generator=gen)
        latencies.append(time.perf_counter() - t0)
        start = To - 1
        disp.append(dispersion(samples, start, Ta))
        for a in samples[0, start:start + Ta]:  # execute sample 0
            obs, r, terminated, truncated, info = env.step(a.astype(np.float32))
            obs_deque.append(obs)
            rewards.append(float(r))
            coverages.append(float(info.get("coverage", np.nan)))
            if render:
                frames.append(env.render())
            steps += 1
            if terminated or truncated or steps >= max_steps:
                done = True
                break
    env.close()
    return {"seed": seed, "max_reward": float(max(rewards)), "steps": steps,
            "max_coverage": float(np.nanmax(coverages)),
            "dispersion_per_replan": disp, "latency_per_replan_s": latencies,
            "frames": frames}
