"""Train a state-based CNN Diffusion Policy on Push-T.

Example:
    python scripts/train.py --data data/pusht/pusht_cchi_v7_replay.zarr --out runs/seed0 --seed 0
Smoke test (minutes on CPU):
    python scripts/train.py --data data/dummy.zarr --out runs/smoke --epochs 2 --small
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dprepro.dataset import PushTStateDataset  # noqa: E402
from dprepro.policy import EMA, DiffusionPolicy  # noqa: E402
from dprepro.rollout import run_episode  # noqa: E402


def cosine_with_warmup(optimizer, warmup, total):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "pusht_state.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    p.add_argument("--max-episodes", type=int, default=None, help="use only the first N demos")
    p.add_argument("--official-split", action="store_true",
                   help="match the official Push-T config: 2%% val episodes held out, 90 random train episodes (seed 42)")
    p.add_argument("--max-train-episodes", type=int, default=None)
    p.add_argument("--small", action="store_true", help="tiny network for smoke tests only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--keep-last", type=int, default=10,
                   help="keep only the newest N periodic checkpoints (EMA weights, ~260 MB each)")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.small:
        cfg.update(down_dims=[32, 64, 128], diffusion_step_embed_dim=32, batch_size=64,
                   checkpoint_every=1, eval_every=1, eval_episodes_during_training=1)
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    val_ratio, max_train = 0.0, args.max_train_episodes
    if args.official_split:
        val_ratio, max_train = 0.02, (args.max_train_episodes or 90)
    ds = PushTStateDataset(args.data, cfg["pred_horizon"], cfg["obs_horizon"], cfg["action_horizon"],
                           max_episodes=args.max_episodes, val_ratio=val_ratio, max_train_episodes=max_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                     num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                                     persistent_workers=args.workers > 0)
    print(f"episodes={ds.n_episodes} samples={len(ds)} iters/epoch={len(dl)} device={args.device}")

    policy = DiffusionPolicy(cfg, ds.normalizer, device=args.device)
    n_params = sum(p.numel() for p in policy.net.parameters())
    print(f"noise-prediction net parameters: {n_params/1e6:.2f}M")
    ema = EMA(policy.net)
    opt = torch.optim.AdamW(policy.net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = cosine_with_warmup(opt, cfg["lr_warmup_steps"], len(dl) * cfg["num_epochs"])
    json.dump({**cfg, "seed": args.seed, "data": args.data, "max_episodes": args.max_episodes,
               "val_ratio": val_ratio, "max_train_episodes": max_train,
               "train_episode_ids": ds.train_episode_ids, "val_episode_ids": ds.val_episode_ids,
               "n_params": n_params}, open(os.path.join(args.out, "config_used.json"), "w"), indent=2)

    log_path = os.path.join(args.out, "train_log.jsonl")
    log = open(log_path, "a")
    t_start = time.time()
    for epoch in range(1, cfg["num_epochs"] + 1):
        policy.net.train()
        losses = []
        for batch in dl:
            loss = policy.compute_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); ema.step(policy.net)
            losses.append(loss.item())
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               "elapsed_s": round(time.time() - t_start, 1)}
        if epoch % cfg["checkpoint_every"] == 0 or epoch == cfg["num_epochs"]:
            ckpt = os.path.join(args.out, f"ckpt_epoch{epoch:05d}.pt")
            policy.save(ckpt, ema_net=ema.averaged_model, extra={"epoch": epoch}, include_raw=False)
            old = sorted(f for f in os.listdir(args.out) if f.startswith("ckpt_epoch"))[:-args.keep_last]
            for f in old:
                os.remove(os.path.join(args.out, f))
        if epoch % cfg["eval_every"] == 0 and cfg["eval_episodes_during_training"] > 0:
            # quick monitor only; final numbers come from scripts/evaluate.py with 50 seeds
            scores = [run_episode(policy, seed=100000 + i, net=ema.averaged_model)["max_reward"]
                      for i in range(cfg["eval_episodes_during_training"])]
            rec["quick_eval_mean_score"] = float(np.mean(scores))
        print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
    log.close()


if __name__ == "__main__":
    main()
