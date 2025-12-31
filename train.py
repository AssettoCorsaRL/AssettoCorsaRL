import argparse
import time
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import create_gym_env
from sac import SACPolicy, SACConfig, get_device

from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--log-interval", type=int, default=1_000)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--load-replay",
        type=str,
        default="replay_buffer_init.pt",
        help="path to a saved LazyTensorStorage state dict (torch.save output) to load",
    )
    p.add_argument("--wandb-project", type=str, default="AssetoCorsaRL-CarRacing")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None, help="WandB run name")

    # exploration schedule (linear annealing from start to end over N steps)
    p.add_argument(
        "--explore-start",
        type=float,
        default=1.0,
        help="Starting probability of taking a random action",
    )
    p.add_argument(
        "--explore-end",
        type=float,
        default=0.0,
        help="Final probability of taking a random action",
    )
    p.add_argument(
        "--explore-steps",
        type=int,
        default=100_000,
        help="Number of steps over which to linearly anneal exploration",
    )

    return p.parse_args()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device() if args.device is None else torch.device(args.device)
    print("Using device:", device)

    cfg = SACConfig()

    try:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={"seed": args.seed, "total_steps": args.total_steps},
        )
        print("WandB initialized:", getattr(wandb.run, "name", None))
    except Exception as e:
        print("Warning: could not initialize WandB:", e)

    env = create_gym_env(device=device, num_envs=cfg.num_envs)
    td = env.reset()

    # ===== Agent =====
    agent = SACPolicy(env=env, num_cells=cfg.num_cells, device=device)
    modules = agent.modules()

    actor = modules["actor"]
    value = modules["value"]
    value_target = modules["value_target"]
    q1 = modules["q1"]
    q2 = modules["q2"]

    print("Initializing lazy modules...")
    with torch.no_grad():
        sample_pixels = td["pixels"][:1].to(device)  # take first env, single batch
        sample_action = torch.zeros(1, env.action_spec.shape[-1], device=device)

        actor_input = TensorDict({"pixels": sample_pixels}, batch_size=1)
        actor(actor_input)

        value(sample_pixels)

        q1(sample_pixels, sample_action)
        q2(sample_pixels, sample_action)

    print("Lazy modules initialized")

    modules["value_target"].load_state_dict(modules["value"].state_dict())

    print("Target network initialized")

    # ===== Optimizers =====
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr
    )
    value_opt = torch.optim.Adam(value.parameters(), lr=cfg.lr)

    print("using ReplayBuffer with LazyTensorStorage")
    storage = LazyTensorStorage(max_size=cfg.replay_size, device="cpu")
    if args.load_replay is not None:
        path = args.load_replay
        try:
            state = torch.load(path, map_location=device)
            storage.load_state_dict(state)
            print(f"Loaded replay storage from {path}")
        except Exception as e:
            print(f"Warning: could not load replay storage from {path}: {e}")
    rb = ReplayBuffer(storage=storage, batch_size=cfg.batch_size)

    from train_core import collect_initial_data, run_training_loop

    current_td, total_steps, episode_returns, current_episode_return = (
        collect_initial_data(env, rb, cfg, td, device)
    )

    try:
        torch.save(storage.state_dict(), "replay_buffer_init.pt")
        print("Replay buffer saved to replay_buffer_init.pt")
    except Exception:
        pass

    start_time = time.time()

    run_training_loop(
        env,
        rb,
        cfg,
        current_td,
        actor,
        value,
        value_target,
        q1,
        q2,
        actor_opt,
        critic_opt,
        value_opt,
        device,
        args,
        storage=storage,
        start_time=start_time,
        total_steps=0,
        episode_returns=episode_returns,
        current_episode_return=current_episode_return,
    )

    # Finish WandB run if active
    if args.wandb:
        try:
            import wandb

            wandb.finish()
            print("WandB finished")
        except Exception:
            pass


if __name__ == "__main__":
    train()
