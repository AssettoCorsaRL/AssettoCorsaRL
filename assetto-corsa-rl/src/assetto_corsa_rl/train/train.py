import argparse
import time
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import create_gym_env
from ..model.sac import SACPolicy, SACConfig, get_device

# configuration loader
import yaml
from types import SimpleNamespace
from pathlib import Path

from torchrl.data.replay_buffers import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    LazyTensorStorage,
)
from tensordict import TensorDict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--log-interval", type=int, default=1_000)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=67_41)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="AssetoCorsaRL-CarRacing")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None, help="WandB run name")

    return p.parse_args()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_cfg_from_yaml(root: Path = None):
    """Load `configs/env_config.yaml` and `configs/model_config.yaml` and merge them.

    Model keys override environment keys when names collide. Missing keys are filled
    from `SACConfig` defaults.
    """
    if root is None:
        # project root (assetto-corsa-rl)
        root = Path(__file__).resolve().parents[3]

    env_p = root / "configs" / "env_config.yaml"
    model_p = root / "configs" / "model_config.yaml"

    def _read(p):
        try:
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: could not read config {p}: {e}")
            return {}

    env = _read(env_p).get("environment", {})
    model = _read(model_p).get("model", {})

    cfg_dict = {}
    cfg_dict.update(model)
    cfg_dict.update(env)

    # Fill missing keys from SACConfig defaults
    defaults = {
        k: getattr(SACConfig, k)
        for k in dir(SACConfig)
        if not k.startswith("__") and not callable(getattr(SACConfig, k))
    }
    for k, v in defaults.items():
        if k not in cfg_dict:
            cfg_dict[k] = v

    cfg = SimpleNamespace(**cfg_dict)
    print(f"Loaded config from: {env_p}, {model_p}")
    return cfg


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device() if args.device is None else torch.device(args.device)
    print("Using device:", device)

    # Load configuration from YAML files (env + model) and fall back to SACConfig defaults
    cfg = load_cfg_from_yaml()

    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={"seed": args.seed, "total_steps": args.total_steps},
    )
    print("WandB initialized:", getattr(wandb.run, "name", None))
    args.wandb = True

    env = create_gym_env(device=device, num_envs=cfg.num_envs)
    td = env.reset()

    # ===== Agent =====
    agent = SACPolicy(
        env=env,
        num_cells=cfg.num_cells,
        device=device,
        use_noisy=cfg.use_noisy,
        noise_sigma=cfg.noise_sigma,
    )
    modules = agent.modules()

    if cfg.use_noisy:
        print(f"Using noisy networks for exploration (sigma={cfg.noise_sigma})")

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
        # Ensure the target network's lazy layers are initialized before loading state
        value_target(sample_pixels)

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

    log_alpha = nn.Parameter(torch.tensor(math.log(cfg.alpha), device=device))
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)

    target_entropy = -float(env.action_spec.shape[-1])  # -dim(A)
    print(f"Target entropy: {target_entropy}")

    print("using PrioritizedReplayBuffer with LazyTensorStorage")
    storage = LazyTensorStorage(max_size=cfg.replay_size, device="cpu")

    rb = PrioritizedReplayBuffer(
        alpha=cfg.per_alpha,
        beta=cfg.per_beta,
        storage=storage,
        batch_size=cfg.batch_size,
    )

    from train_core import run_training_loop

    current_td = td
    total_steps = 0
    episode_returns = []
    current_episode_return = torch.zeros(cfg.num_envs, device=device)

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
        log_alpha,
        alpha_opt,
        target_entropy,
        device,
        args,
        storage=storage,
        start_time=start_time,
        total_steps=total_steps,
        episode_returns=episode_returns,
        current_episode_return=current_episode_return,
    )

    # Finish WandB run if active
    if getattr(args, "wandb", False):
        try:
            import wandb

            wandb.finish()
            print("WandB finished")
        except Exception as e:
            print("Warning: could not finish WandB run:", e)


if __name__ == "__main__":
    train()
