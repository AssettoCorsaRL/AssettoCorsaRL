import time
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from types import SimpleNamespace
import wandb
from tensordict import TensorDict

try:
    from assetto_corsa_rl.ac_env import create_transformed_env, get_device  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.ac_env import create_transformed_env, get_device  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore

from torchrl.data.replay_buffers import PrioritizedReplayBuffer, LazyTensorStorage

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


def load_cfg_from_yaml(root: Path = None):
    """Load configs/ac/env_config.yaml, model_config.yaml, train_config.yaml and merge."""
    if root is None:
        root = Path(__file__).resolve().parents[2]

    env_p = root / "configs" / "ac" / "env_config.yaml"
    model_p = root / "configs" / "ac" / "model_config.yaml"
    train_p = root / "configs" / "ac" / "train_config.yaml"

    def _read(p):
        try:
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: could not read config {p}: {e}")
            return {}

    env = _read(env_p).get("environment", {})
    model = _read(model_p).get("model", {})
    train_raw = _read(train_p)
    train = {}
    if isinstance(train_raw, dict):
        train.update(train_raw.get("train", {}))
        train.update(train_raw.get("training", {}))

    cfg_dict = {}
    cfg_dict.update(model)
    cfg_dict.update(env)
    cfg_dict.update(train)

    input_config = env.get("inputs", None)

    cfg_dict = {}
    cfg_dict.update(model)
    cfg_dict.update(env)
    cfg_dict.update(train)

    def _try_convert(x):
        if x is None or isinstance(x, bool):
            return x
        if isinstance(x, dict):
            return {k: _try_convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_try_convert(v) for v in x]
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return x
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("_", "")
            try:
                if "." not in s and "e" not in s.lower():
                    return int(s)
                return float(s)
            except Exception:
                return x
        return x

    converted = {k: _try_convert(v) for k, v in cfg_dict.items()}

    if isinstance(converted.get("wandb"), dict):
        wandb_dict = converted.pop("wandb")
        for k, v in wandb_dict.items():
            converted[f"wandb_{k}"] = v

    for k in ("wandb_project", "wandb_entity", "wandb_name", "wandb_enabled"):
        converted.setdefault(k, None)

    converted["num_envs"] = 1

    converted["input_config"] = input_config

    cfg = SimpleNamespace(**converted)
    print(f"Loaded config from: {env_p}, {model_p}, {train_p}")
    if input_config:
        enabled_count = sum(1 for v in input_config.values() if v)
        print(f"  Input config: {enabled_count}/{len(input_config)} inputs enabled")
    return cfg


@cli_command(group="ac", name="train", help="Train SAC agent in Assetto Corsa")
def train():
    cfg = load_cfg_from_yaml()

    torch.manual_seed(cfg.seed)
    device = get_device() if cfg.device is None else torch.device(cfg.device)
    print("Using device:", device)

    try:

        wandb_kwargs = {
            "project": cfg.wandb_project,
            "config": {"seed": cfg.seed, "total_steps": cfg.total_steps},
        }
        if getattr(cfg, "wandb_entity", None):
            wandb_kwargs["entity"] = cfg.wandb_entity
        if getattr(cfg, "wandb_name", None):
            wandb_kwargs["name"] = cfg.wandb_name
        wandb_run = wandb.init(**wandb_kwargs)
        print("WandB initialized:", getattr(wandb.run, "name", None))
    except Exception as e:
        print("Warning: WandB init failed, continuing without logging:", e)

    env = create_transformed_env(
        racing_line_path="racing_lines.json",
        device=device,
        image_shape=(84, 84),
        frame_stack=4,
        input_config=getattr(cfg, "input_config", None),
    )
    current_td = env.reset()

    input("press enter when ur sure the controller is connected n stuff")
    print(f"Initial pixels shape: {current_td.get('pixels').shape}")

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

    with torch.no_grad():
        dummy_pixels = current_td.get("pixels").unsqueeze(0).to(device)
        init_td = TensorDict({"pixels": dummy_pixels}, batch_size=[1])
        modules["actor"](init_td.clone())
        modules["value"](init_td.clone())

    if cfg.use_noisy:
        print(f"Using noisy networks for exploration (sigma={cfg.noise_sigma})")

    actor = modules["actor"]
    value = modules["value"]
    value_target = modules["value_target"]
    q1 = modules["q1"]
    q2 = modules["q2"]
    q1_target = modules["q1_target"]
    q2_target = modules["q2_target"]

    pretrained_path = getattr(cfg, "pretrained_model", None)
    bc_pretrained_path = getattr(cfg, "bc_pretrained_model", None)

    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}...")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if "actor_state" in checkpoint:
                actor.load_state_dict(checkpoint["actor_state"])
                print("Loaded actor state")
            if "q1_state" in checkpoint:
                q1.load_state_dict(checkpoint["q1_state"])
                print("Loaded Q1 state")
            if "q2_state" in checkpoint:
                q2.load_state_dict(checkpoint["q2_state"])
                print("Loaded Q2 state")
            if "value_state" in checkpoint:
                value.load_state_dict(checkpoint["value_state"])
                print("Loaded value state")
            value_target.load_state_dict(value.state_dict())
            q1_target.load_state_dict(q1.state_dict())
            q2_target.load_state_dict(q2.state_dict())
            print("Copied states to target networks")
        except Exception as e:
            print(f"Warning: Failed to load pretrained model: {e}")
            value_target.load_state_dict(value.state_dict())
    elif bc_pretrained_path:
        print(f"Loading BC-SAC pretrained model from {bc_pretrained_path}...")
        try:
            checkpoint = torch.load(bc_pretrained_path, map_location=device)
            if "actor_state" in checkpoint:
                actor.load_state_dict(checkpoint["actor_state"])
                print(
                    f"✓ Loaded BC-SAC pretrained actor (val_mse: {checkpoint.get('val_mse', 'N/A')})"
                )
            else:
                print("Warning: No actor_state found in BC-SAC checkpoint")

            # Load Q-networks if available (BC-SAC trains critics too)
            if "q1_state" in checkpoint:
                q1.load_state_dict(checkpoint["q1_state"])
                print("✓ Loaded BC-SAC pretrained Q1")
            if "q2_state" in checkpoint:
                q2.load_state_dict(checkpoint["q2_state"])
                print("✓ Loaded BC-SAC pretrained Q2")
            if "q1_target_state" in checkpoint:
                q1_target.load_state_dict(checkpoint["q1_target_state"])
                print("✓ Loaded BC-SAC pretrained Q1 target")
            if "q2_target_state" in checkpoint:
                q2_target.load_state_dict(checkpoint["q2_target_state"])
                print("✓ Loaded BC-SAC pretrained Q2 target")

            # Initialize value target from current value
            value_target.load_state_dict(value.state_dict())
        except Exception as e:
            print(f"Warning: Failed to load BC-SAC pretrained model: {e}")
            value_target.load_state_dict(value.state_dict())
    else:
        value_target.load_state_dict(value.state_dict())

    print("Target network initialized")

    actor_lr = getattr(cfg, "actor_lr", cfg.lr)
    critic_lr = getattr(cfg, "critic_lr", cfg.lr)
    value_lr = getattr(cfg, "value_lr", cfg.lr)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=critic_lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=value_lr)

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
    )

    total_steps = 0
    episode_returns = []
    current_episode_return = torch.zeros(1, device=device)
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
        value_opt,
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

    wandb.finish()
    print("WandB finished")


if __name__ == "__main__":
    train()
