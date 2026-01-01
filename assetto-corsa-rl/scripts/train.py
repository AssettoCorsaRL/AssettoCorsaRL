import time
import math
from collections import deque
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from assetto_corsa_rl.env import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.env import create_gym_env  # type: ignore
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_cfg_from_yaml(root: Path = None):
    """Load `configs/env_config.yaml` and `configs/model_config.yaml` and merge them.

    Model keys override environment keys when names collide. Missing keys are filled
    from `SACConfig` defaults.

    This version preserves integers and only converts string representations to appropriate types.
    """
    if root is None:
        # project root (assetto-corsa-rl)
        root = Path(__file__).resolve().parents[1]

    env_p = root / "configs" / "env_config.yaml"
    model_p = root / "configs" / "model_config.yaml"
    train_p = root / "configs" / "train_config.yaml"

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
    # Support both keys 'train' and 'training' to be robust to config variants
    if isinstance(train_raw, dict):
        train = {**train_raw.get("train", {}), **train_raw.get("training", {})}
    else:
        train = {}

    cfg_dict = {}
    cfg_dict.update(model)
    cfg_dict.update(env)
    cfg_dict.update(train)

    def _try_convert(x):
        # preserve booleans and None
        if x is None or isinstance(x, bool):
            return x
        # dict -> recurse
        if isinstance(x, dict):
            return {k: _try_convert(v) for k, v in x.items()}
        # list/tuple -> recurse and return list
        if isinstance(x, (list, tuple)):
            return [_try_convert(v) for v in x]
        # Keep integers as integers, floats as floats
        if isinstance(x, int):
            return x  # Don't convert to float!
        if isinstance(x, float):
            return x
        # strings -> try to parse as int first, then float
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("_", "")
            try:
                # Try int first (this handles cases like "4" or "1_000_000")
                if "." not in s and "e" not in s.lower():
                    return int(s)
                else:
                    return float(s)
            except Exception:
                return x
        # fallback: return original
        return x

    converted = {k: _try_convert(v) for k, v in cfg_dict.items()}

    # If train config contains a nested `wandb` dict, flatten it to top-level
    # keys with prefix `wandb_` so downstream code can access `cfg.wandb_project` etc.
    if isinstance(converted.get("wandb"), dict):
        wandb_dict = converted.pop("wandb")
        for k, v in wandb_dict.items():
            converted[f"wandb_{k}"] = v

    # Ensure common top-level wandb keys exist (default to None)
    for k in ("wandb_project", "wandb_entity", "wandb_name", "wandb_enabled"):
        converted.setdefault(k, None)

    cfg = SimpleNamespace(**converted)
    print(f"Loaded config from: {env_p}, {model_p}, {train_p}")
    return cfg


def train():

    cfg = load_cfg_from_yaml()

    torch.manual_seed(cfg.seed)

    device = get_device() if cfg.device is None else torch.device(cfg.device)
    print("Using device:", device)

    import wandb

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_name,
        config={"seed": cfg.seed, "total_steps": cfg.total_steps},
    )
    print("WandB initialized:", getattr(wandb.run, "name", None))

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

    q1_target = modules["q1_target"]
    q2_target = modules["q2_target"]

    # Network summary will be printed after lazy initialization so that
    # LazyModule parameters are initialized before we attempt to count them.
    # (See later in this function where we print the networks.)

    # def init_optimistic(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.constant_(m.bias, 1.0)  # Optimistic bias
    #         nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small weights

    # q1.apply(init_optimistic)
    # q2.apply(init_optimistic)

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

    # Now that LazyModules have been initialized, print the network summaries
    print("Networks:")
    for name, net in modules.items():
        print("=" * 40)
        print(f"{name}:")
        print(net)
        num_fn = getattr(net, "num_parameters", None)
        if callable(num_fn):
            nparams = num_fn()
        else:
            try:
                nparams = sum(p.numel() for p in net.parameters())
            except Exception:
                nparams = "uninitialized (LazyModule)"
        print(f"Number of parameters: {nparams}")

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

    try:
        wandb.finish()
        print("WandB finished")
    except Exception as e:
        print("Warning: could not finish WandB run:", e)


if __name__ == "__main__":
    train()
