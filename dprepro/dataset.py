"""Push-T demonstration dataset (official zarr from the Diffusion Policy authors).

Expected layout (pusht_cchi_v7_replay.zarr):
    data/state   (N, 5)  agent_x, agent_y, block_x, block_y, block_angle
    data/action  (N, 2)  target agent position (pixels, 0..512)
    meta/episode_ends (E,) exclusive end index of each episode

Sampling follows the reference implementation: windows of length `pred_horizon`, padded by
repeating the first/last frame so every timestep can serve as the "current" observation.
"""
import numpy as np
import torch
import zarr


def get_val_mask(n_episodes, val_ratio, seed=0):
    """Same procedure as the official code: hold out round(n*val_ratio) episodes (at least 1)."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    """Same procedure as the official code: keep a random subset of max_n training episodes."""
    if max_n is None or mask.sum() <= max_n:
        return mask
    idxs = np.nonzero(mask)[0]
    rng = np.random.default_rng(seed=seed)
    keep = idxs[rng.choice(len(idxs), size=int(max_n), replace=False)]
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0, episode_mask=None):
    indices = []
    for i in range(len(episode_ends)):
        if episode_mask is not None and not episode_mask[i]:
            continue
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    return np.array(indices, dtype=np.int64)


def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = {}
    for key, arr in train_data.items():
        sample = arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            data = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class MinMaxNormalizer:
    """Per-dimension min-max scaling to [-1, 1], fitted on the training data."""

    def __init__(self, stats: dict):
        self.stats = {k: {kk: np.asarray(vv, dtype=np.float32) for kk, vv in v.items()} for k, v in stats.items()}

    @classmethod
    def fit(cls, data: dict):
        return cls({k: {"min": v.min(axis=0), "max": v.max(axis=0)} for k, v in data.items()})

    def normalize(self, key, x):
        s = self.stats[key]
        rng = np.maximum(s["max"] - s["min"], 1e-8)
        return ((x - s["min"]) / rng) * 2 - 1

    def unnormalize(self, key, x):
        s = self.stats[key]
        rng = np.maximum(s["max"] - s["min"], 1e-8)
        return (x + 1) / 2 * rng + s["min"]

    def state_dict(self):
        return {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in self.stats.items()}


class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path, pred_horizon=16, obs_horizon=2, action_horizon=8, max_episodes=None,
                 val_ratio=0.0, max_train_episodes=None, split_seed=42):
        """Default (val_ratio=0, max_train_episodes=None) trains on every episode.
        Official Push-T config: val_ratio=0.02, max_train_episodes=90, split_seed=42."""
        root = zarr.open(zarr_path, mode="r")
        episode_ends = root["meta"]["episode_ends"][:]
        if max_episodes is not None:  # data-efficiency ablations: first N episodes
            episode_ends = episode_ends[:max_episodes]
        n = int(episode_ends[-1])
        all_data = {
            "obs": root["data"]["state"][:n].astype(np.float32),
            "action": root["data"]["action"][:n].astype(np.float32),
        }
        val_mask = get_val_mask(len(episode_ends), val_ratio, seed=split_seed)
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed=split_seed)
        self.train_episode_ids = np.nonzero(train_mask)[0].tolist()
        self.val_episode_ids = np.nonzero(val_mask)[0].tolist()
        self.indices = create_sample_indices(episode_ends, pred_horizon, pad_before=obs_horizon - 1,
                                             pad_after=action_horizon - 1, episode_mask=train_mask)
        # normalizer fitted on the training episodes only
        starts = np.concatenate([[0], episode_ends[:-1]])
        rows = np.concatenate([np.arange(starts[i], episode_ends[i]) for i in self.train_episode_ids])
        self.normalizer = MinMaxNormalizer.fit({k: v[rows] for k, v in all_data.items()})
        self.data = {k: self.normalizer.normalize(k, v).astype(np.float32) for k, v in all_data.items()}
        self.n_episodes = len(self.train_episode_ids)
        self.pred_horizon, self.obs_horizon = pred_horizon, obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b0, b1, s0, s1 = self.indices[idx]
        s = sample_sequence(self.data, self.pred_horizon, b0, b1, s0, s1)
        return {"obs": torch.from_numpy(s["obs"][: self.obs_horizon]),
                "action": torch.from_numpy(s["action"])}
