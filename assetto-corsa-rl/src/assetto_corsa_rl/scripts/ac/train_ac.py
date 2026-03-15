# NOTE: all arguments for this script are the .yamls

import time
import math
import sys
import os
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from tensordict import TensorDict
import subprocess, time as _time  # noqa: E401

try:
    from assetto_corsa_rl.ac_env import create_transformed_env, get_device  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore
    from assetto_corsa_rl.train.logging_utils import print_banner, print_section_header, log_info, log_success, log_warning, log_error  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.ac_env import create_transformed_env, get_device  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
    from assetto_corsa_rl.train.train_core import run_training_loop  # type: ignore
    from assetto_corsa_rl.train.logging_utils import print_banner, print_section_header, log_info, log_success, log_warning, log_error  # type: ignore

from torchrl.data.replay_buffers import PrioritizedReplayBuffer, ListStorage

try:
    from assetto_corsa_rl.cli_registry import cli_command, load_cfg_from_yaml
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, load_cfg_from_yaml  # type: ignore


def _do_train():
    """Core training loop – called by ``train`` and by ``wandb.agent`` during sweeps."""
    log_success("Loading configuration...", bold=True)
    cfg = load_cfg_from_yaml()
    print(cfg)

    torch.manual_seed(cfg.seed)
    device = get_device() if cfg.device is None else torch.device(cfg.device)
    log_success(f"Device: {device}", bold=True)

    if getattr(cfg, "normalize_observations", False):
        log_success("Observation normalization enabled", bold=True)

    try:
        wandb_kwargs = {
            "project": cfg.wandb_project,
            "config": {"seed": cfg.seed, "total_steps": cfg.total_steps},
        }
        if getattr(cfg, "wandb_entity", None):
            wandb_kwargs["entity"] = cfg.wandb_entity
        if getattr(cfg, "wandb_name", None):
            wandb_kwargs["name"] = cfg.wandb_name
        wandb.init(**wandb_kwargs)
        is_sweep_run = bool(
            (wandb.run is not None and getattr(wandb.run, "sweep_id", None))
            or os.getenv("WANDB_SWEEP_ID")
        )
        if wandb.run is not None and is_sweep_run:
            for k, v in dict(wandb.config).items():
                if hasattr(cfg, k) and not k.startswith("_"):
                    setattr(cfg, k, v)
                    log_info(f"[sweep] cfg.{k} = {v}")
        log_success(f"WandB initialized: {getattr(wandb.run, 'name', None)}")
    except Exception as e:
        log_warning(f"WandB init failed, continuing without logging: {e}")

    env_kwargs = dict(
        racing_line_path=getattr(cfg, "racing_line_path", "racing_lines.json"),
        device=device,
        image_shape=(84, 84),
        frame_stack=3,
        input_config=getattr(cfg, "input_config", None),
        use_ac_ai_racer=False,
        normalize_observations=getattr(cfg, "normalize_observations", False),
        normalization_bounds=getattr(cfg, "normalization_bounds", None),
    )

    _proc_list = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq acs.exe"], capture_output=True, text=True
    )
    if "acs.exe" not in _proc_list.stdout.lower():
        log_info("Launching Assetto Corsa...")
        subprocess.Popen(
            [r"D:\Steam\steamapps\common\assettocorsa\acs.exe"],
            cwd=r"D:\Steam\steamapps\common\assettocorsa",
        )
        _time.sleep(10)
        from assetto_corsa_rl.train.train_utils import activate_ac_window

        activate_ac_window()
        log_success("Assetto Corsa launched.")
    else:
        log_success("Assetto Corsa is already running.")

    env = create_transformed_env(**env_kwargs)
    current_td = env.reset()

    log_info(f"Initial pixels shape: {current_td.get('pixels').shape}")

    vae_path = getattr(cfg, "vae_checkpoint_path", None)
    agent = SACPolicy(
        env=env,
        num_cells=cfg.num_cells,
        device=device,
        use_noisy=cfg.use_noisy,
        noise_sigma=cfg.noise_sigma,
        vae_checkpoint_path=vae_path,
        use_lstm=getattr(cfg, "use_lstm", False),
        lstm_hidden_size=getattr(cfg, "lstm_hidden_size", 256),
        lstm_layers=getattr(cfg, "lstm_layers", 1),
        stateful_inference=getattr(cfg, "stateful_inference", True),
    )
    modules = agent.modules()

    with torch.no_grad():
        dummy_pixels = current_td.get("pixels").unsqueeze(0).to(device)
        init_data = {"pixels": dummy_pixels}
        dummy_vector = current_td.get("vector", None)
        if dummy_vector is not None:
            init_data["vector"] = dummy_vector.unsqueeze(0).to(device)
        init_td = TensorDict(init_data, batch_size=[1])
        modules["actor"](init_td.clone())

    if cfg.use_noisy:
        log_info(f"Using noisy networks for exploration (sigma={cfg.noise_sigma})")

    actor = modules["actor"]
    q1 = modules["q1"]
    q2 = modules["q2"]
    q1_target = modules["q1_target"]
    q2_target = modules["q2_target"]

    pretrained_path = getattr(cfg, "pretrained_model", None)
    bc_pretrained_path = cfg.bc_pretrained_model
    log_info(f"BC pretrained model: {bc_pretrained_path}")

    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}...")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if "actor_state" in checkpoint:
                actor.load_state_dict(checkpoint["actor_state"])
                log_success("Loaded actor state")
            if "q1_state" in checkpoint:
                q1.load_state_dict(checkpoint["q1_state"])
                log_success("Loaded Q1 state")
            if "q2_state" in checkpoint:
                q2.load_state_dict(checkpoint["q2_state"])
                log_success("Loaded Q2 state")
            q1_target.load_state_dict(q1.state_dict())
            q2_target.load_state_dict(q2.state_dict())
            log_success("Copied states to target networks")
        except Exception as e:
            log_warning(f"Failed to load pretrained model: {e}")
    elif bc_pretrained_path:
        log_info(f"Loading BC-SAC pretrained model from {bc_pretrained_path}...")
        checkpoint = torch.load(bc_pretrained_path, map_location=device)

        # check if BC model was trained with different noisy setting
        bc_use_noisy = checkpoint.get("config", {}).get("use_noisy", False)
        current_use_noisy = cfg.use_noisy

        if bc_use_noisy != current_use_noisy:
            log_warning(
                f"BC model was trained with use_noisy={bc_use_noisy}, "
                f"current model has use_noisy={current_use_noisy}"
            )
            log_info("Loading with strict=False to handle architecture mismatch...")
            strict = False
        else:
            strict = True

        if "actor_state" in checkpoint:
            try:
                actor.load_state_dict(checkpoint["actor_state"], strict=strict)
                print(
                    f"✓ Loaded BC-SAC pretrained actor (val_mse: {checkpoint.get('val_mse', 'N/A')})"
                )
            except Exception as e:
                print(f"  Warning: Partial actor load: {e}")
        else:
            print("Warning: No actor_state found in BC-SAC checkpoint")

        if "q1_state" in checkpoint:
            try:
                q1.load_state_dict(checkpoint["q1_state"], strict=strict)
                log_success("Loaded BC-SAC pretrained Q1")
            except Exception as e:
                log_warning(f"Partial Q1 load: {e}")
        if "q2_state" in checkpoint:
            try:
                q2.load_state_dict(checkpoint["q2_state"], strict=strict)
                log_success("Loaded BC-SAC pretrained Q2")
            except Exception as e:
                log_warning(f"Partial Q2 load: {e}")
        if "q1_target_state" in checkpoint:
            try:
                q1_target.load_state_dict(checkpoint["q1_target_state"], strict=strict)
                log_success("Loaded BC-SAC pretrained Q1 target")
            except Exception as e:
                log_warning(f"Partial Q1 target load: {e}")
        if "q2_target_state" in checkpoint:
            try:
                q2_target.load_state_dict(checkpoint["q2_target_state"], strict=strict)
                log_success("Loaded BC-SAC pretrained Q2 target")
            except Exception as e:
                log_warning(f"Partial Q2 target load: {e}")

    log_success("Target network initialized")

    actor_lr = getattr(cfg, "actor_lr", cfg.lr)
    critic_lr = getattr(cfg, "critic_lr", cfg.lr)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=critic_lr)

    log_alpha = nn.Parameter(torch.tensor(math.log(cfg.alpha), device=device))
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
    target_entropy = -float(env.action_spec.shape[-1])
    log_info(f"Target entropy: {target_entropy}")

    def _collate_sequence_batch(batch):
        """Collate a list of sequence TensorDicts into a single batched TensorDict."""
        if isinstance(batch, TensorDict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            keys = batch[0].keys()
            result = {}
            for k in keys:
                result[k] = torch.stack([b[k] for b in batch], dim=0)
            return TensorDict(result, batch_size=[len(batch)])
        return batch

    log_info("Using PrioritizedReplayBuffer with ListStorage")
    storage = ListStorage(max_size=cfg.replay_size)
    rb = PrioritizedReplayBuffer(
        alpha=cfg.per_alpha,
        beta=cfg.per_beta,
        storage=storage,
        batch_size=cfg.batch_size,
        collate_fn=_collate_sequence_batch,
    )

    replay_buffer_path = getattr(cfg, "replay_buffer_path", None)
    if replay_buffer_path and Path(replay_buffer_path).exists():
        log_info(f"Loading replay buffer from {replay_buffer_path}...")
        try:
            import pickle

            # Support both .pt (torch) and .pkl (pickle) formats
            if replay_buffer_path.endswith(".pt"):
                rb_state = torch.load(replay_buffer_path, weights_only=False)
            else:
                with open(replay_buffer_path, "rb") as f:
                    rb_state = pickle.load(f)

            if "buffer" in rb_state:
                rb._storage._storage = rb_state["buffer"]
                log_success(
                    f"Loaded {rb_state.get('buffer_size', 'unknown')} transitions from replay buffer"
                )

            if "sampler_state" in rb_state:
                sampler_state = rb_state["sampler_state"]
                if sampler_state.get("alpha") is not None and hasattr(rb._sampler, "_alpha"):
                    rb._sampler._alpha = sampler_state["alpha"]
                if sampler_state.get("beta") is not None and hasattr(rb._sampler, "_beta"):
                    rb._sampler._beta = sampler_state["beta"]
                log_success(
                    f"Restored sampler state (alpha={sampler_state.get('alpha')}, beta={sampler_state.get('beta')})"
                )

            log_info(f"Replay buffer loaded from step {rb_state.get('total_steps', 'unknown')}")
        except Exception as e:
            log_warning(f"Failed to load replay buffer: {e}")
            log_info("Starting with empty replay buffer")
    elif replay_buffer_path:
        log_warning(f"Replay buffer path specified but file not found: {replay_buffer_path}")

    # ── Load expert demonstrations (if enabled) ────────────────────────
    # In async mode, demos are loaded inside the learner subprocess only.
    use_async = bool(getattr(cfg, "use_async", False))
    if getattr(cfg, "use_expert_demonstrations", False) and not use_async:
        log_warning(
            "Skipping expert demonstrations: current replay buffer stores sequence chunks "
            "(features/actions/...), while demo loader emits single-step transitions "
            "(pixels/next_pixels)."
        )
    elif getattr(cfg, "use_expert_demonstrations", False) and use_async:
        log_info("Expert demonstrations will be loaded by LearnerWorker (async mode)")

    total_steps = 0
    episode_returns = []
    current_episode_return = torch.zeros(1, device=device)

    start_time = time.time()

    print_banner("Training Started")
    print_section_header("SAC Training Loop")

    run_training_loop(
        env,
        rb,
        cfg,
        current_td,
        actor,
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
        env_kwargs=env_kwargs,
    )

    wandb.finish()
    log_success("WandB finished. Training complete!")


@cli_command(group="ac", name="train", help="Train SAC agent in Assetto Corsa")
def train():
    """Train SAC agent in Assetto Corsa."""
    _do_train()


if __name__ == "__main__":
    train()
