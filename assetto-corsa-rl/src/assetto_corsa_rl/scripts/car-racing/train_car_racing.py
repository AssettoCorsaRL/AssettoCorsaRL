import time
import math
from collections import deque
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from assetto_corsa_rl.env_helper import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.env_helper import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore

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

try:
    from assetto_corsa_rl.cli_registry import cli_command, load_cfg_from_yaml
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, load_cfg_from_yaml  # type: ignore


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@cli_command(group="car-racing", name="train", help="Train SAC agent in CarRacing environment")
def train():
    cfg = load_cfg_from_yaml()

    torch.manual_seed(cfg.seed)

    device = get_device() if cfg.device is None else torch.device(cfg.device)
    print("Using device:", device)

    import wandb

    wandb_kwargs = {
        "project": cfg.wandb_project,
        "config": {"seed": cfg.seed, "total_steps": cfg.total_steps},
    }
    if getattr(cfg, "wandb_entity", None):
        wandb_kwargs["entity"] = cfg.wandb_entity
    if getattr(cfg, "wandb_name", None):
        wandb_kwargs["name"] = cfg.wandb_name

    wandb.init(**wandb_kwargs)
    print("WandB initialized:", getattr(wandb.run, "name", None))

    env = create_gym_env(
        device=device,
        num_envs=cfg.num_envs,
        fixed_track_seed=cfg.seed,
    )
    td = env.reset()

    # ===== Agent =====
    vae_path = getattr(cfg, "vae_checkpoint_path", None)
    agent = SACPolicy(
        env=env,
        num_cells=cfg.num_cells,
        device=device,
        use_noisy=cfg.use_noisy,
        noise_sigma=cfg.noise_sigma,
        vae_checkpoint_path=vae_path,
    )
    modules = agent.modules()

    if cfg.use_noisy:
        print(f"Using noisy networks for exploration (sigma={cfg.noise_sigma})")

    actor = modules["actor"]
    value = modules["value"]
    value_target = modules["value_target"]
    q1 = modules["q1"]
    q2 = modules["q2"]

    q1_target = modules["q1_target"]
    q2_target = modules["q2_target"]

    print("Networks initialized with explicit dimensions")

    # pretrained model if specified
    pretrained_path = getattr(cfg, "pretrained_model", None)
    if pretrained_path is not None and pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}...")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)

            if "actor_state" in checkpoint:
                actor.load_state_dict(checkpoint["actor_state"])
                print("Loaded actor state from pretrained model")

            if "q1_state" in checkpoint:
                q1.load_state_dict(checkpoint["q1_state"])
                print("Loaded Q1 state from pretrained model")
            if "q2_state" in checkpoint:
                q2.load_state_dict(checkpoint["q2_state"])
                print("Loaded Q2 state from pretrained model")

            if "value_state" in checkpoint:
                value.load_state_dict(checkpoint["value_state"])
                print("Loaded value state from pretrained model")

            # copy to target networks
            value_target.load_state_dict(value.state_dict())
            q1_target.load_state_dict(q1.state_dict())
            q2_target.load_state_dict(q2.state_dict())
            print("Copied states to target networks")

            print(f"Successfully loaded pretrained model from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained model: {e}")
            print("Continuing with randomly initialized networks")
            modules["value_target"].load_state_dict(modules["value"].state_dict())
    else:
        modules["value_target"].load_state_dict(modules["value"].state_dict())

    print("Target network initialized")

    # check if VAE encoders are trainable
    if vae_path:
        trainable_count = sum(1 for p in actor.parameters() if p.requires_grad)
        print(f"✓ VAE encoders are trainable with target networks for stability")
        print(f"  Actor has {trainable_count} trainable parameter groups")

    print("Networks:")
    for name, net in modules.items():
        print("=" * 40)
        print(f"{name}:")
        print(net)
        num_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params} (trainable: {trainable_params})")

    # ===== Optimizers =====
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr)

    log_alpha = nn.Parameter(torch.tensor(math.log(cfg.alpha), device=device))
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)

    target_entropy = -float(env.action_spec.shape[-1])
    print(f"Target entropy: {target_entropy}")

    print("using PrioritizedReplayBuffer with LazyTensorStorage")
    storage = LazyTensorStorage(max_size=cfg.replay_size, device="cpu")

    rb = PrioritizedReplayBuffer(
        alpha=cfg.per_alpha,
        beta=cfg.per_beta,
        storage=storage,
        batch_size=cfg.batch_size,
        dtype=torch.float64,
    )

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
        q1_target,
        q2_target,
        actor_opt,
        critic_opt,
        log_alpha,
        alpha_opt,
        target_entropy,
        device,
        storage=storage,
        start_time=start_time,
        total_steps=total_steps,
        episode_returns=episode_returns,
        current_episode_return=current_episode_return,
    )

    try:
        wandb.finish()
        print("WandB finished")
    except Exception as e:
        print("Warning: could not finish WandB run:", e)


if __name__ == "__main__":
    train()
