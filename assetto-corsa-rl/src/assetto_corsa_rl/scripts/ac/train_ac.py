import os
import psutil

p = psutil.Process(os.getpid())
p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)

try:
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == "acs.exe":
            psutil.Process(proc.info["pid"]).nice(psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass

# NOTE: all arguments for this script are the .yamls

import time
import sys
import os
from pathlib import Path

import torch
import wandb
from tensordict import TensorDict
import subprocess, time as _time  # noqa: E401
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.trainers.algorithms import SACTrainer

try:
    from assetto_corsa_rl.ac_env import create_transformed_env, get_device  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
    from assetto_corsa_rl.train.logging_utils import print_banner, print_section_header, log_info, log_success, log_warning, log_error  # type: ignore
    from assetto_corsa_rl.train.train_utils import activate_ac_window, kill_all_ac_instances  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.ac_env import create_transformed_env, get_device  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
    from assetto_corsa_rl.train.logging_utils import print_banner, print_section_header, log_info, log_success, log_warning, log_error  # type: ignore
    from assetto_corsa_rl.train.train_utils import activate_ac_window, kill_all_ac_instances  # type: ignore

from torchrl.data.replay_buffers import PrioritizedReplayBuffer, ReplayBuffer, LazyTensorStorage

try:
    from assetto_corsa_rl.cli_registry import cli_command, load_cfg_from_yaml
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, load_cfg_from_yaml  # type: ignore


def _build_sac_trainer(cfg, env, actor, q1, replay_buffer, device):
    total_frames = int(getattr(cfg, "total_steps", 1_000_000))
    frames_per_batch = max(1, int(getattr(cfg, "frames_per_batch", 1024)))
    updates_per_step = max(1, int(getattr(cfg, "updates_per_step", 1)))
    optim_steps_per_batch = max(
        1,
        int(getattr(cfg, "optim_steps_per_batch", frames_per_batch * updates_per_step)),
    )

    collector_device_cfg = getattr(cfg, "collector_device", None)
    collector_device = device if collector_device_cfg is None else torch.device(collector_device_cfg)

    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=collector_device,
        policy_device=device,
        storing_device=torch.device("cpu"),
        init_random_frames=max(0, int(getattr(cfg, "start_steps", 0))),
    )

    target_entropy = -float(env.action_spec.shape[-1])
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=q1,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=float(getattr(cfg, "alpha", 0.2)),
        target_entropy=target_entropy,
        fixed_alpha=False,
    ).to(device)
    loss_module.make_value_estimator(gamma=float(getattr(cfg, "gamma", 0.99)))

    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=float(getattr(cfg, "lr", 3e-4)),
        eps=1e-8,
    )

    tau = float(getattr(cfg, "tau", 0.01))
    soft_update_eps = float(max(0.0, min(1.0, 1.0 - tau)))
    target_net_updater = SoftUpdate(loss_module, eps=soft_update_eps)

    checkpoint_dir = Path(getattr(cfg, "checkpoint_dir", "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_state_path = checkpoint_dir / "sac_trainer_state.pt"

    trainer = SACTrainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=1,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        batch_size=int(getattr(cfg, "batch_size", 128)),
        clip_grad_norm=True,
        clip_norm=float(getattr(cfg, "max_grad_norm", 1.0)),
        progress_bar=True,
        seed=int(getattr(cfg, "seed", 0)),
        save_trainer_interval=max(1, int(getattr(cfg, "save_interval", 10_000))),
        log_interval=max(1, int(getattr(cfg, "log_interval", 10_000))),
        save_trainer_file=str(trainer_state_path),
        enable_logging=True,
        log_rewards=True,
        log_actions=True,
        log_observations=False,
        target_net_updater=target_net_updater,
    )

    return trainer, loss_module


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

        def _to_jsonable(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [_to_jsonable(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _to_jsonable(v) for k, v in value.items()}
            return str(value)

        wandb_cfg = {k: _to_jsonable(v) for k, v in vars(cfg).items() if not k.startswith("_")}

        wandb_kwargs = {
            "project": cfg.wandb_project,
            "config": wandb_cfg,
        }
        if getattr(cfg, "wandb_entity", None):
            wandb_kwargs["entity"] = cfg.wandb_entity
        if getattr(cfg, "wandb_name", None):
            wandb_kwargs["name"] = cfg.wandb_name
        wandb.init(**wandb_kwargs)
        run_sweep_id = getattr(wandb.run, "sweep_id", None) if wandb.run is not None else None
        env_sweep_id = os.getenv("WANDB_SWEEP_ID")
        is_sweep_run = bool(run_sweep_id)
        if env_sweep_id and not is_sweep_run:
            log_warning(
                "WANDB_SWEEP_ID is set but this run has no sweep_id; skipping sweep overrides."
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
        log_info("Cleaning up any existing Assetto Corsa instances...")
        kill_all_ac_instances()
        log_info("Launching Assetto Corsa...")
        subprocess.Popen(
            [r"D:\Steam\steamapps\common\assettocorsa\acs.exe"],
            cwd=r"D:\Steam\steamapps\common\assettocorsa",
        )
        _time.sleep(20)
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

        if "actor_state" in checkpoint:
            try:
                actor.load_state_dict(checkpoint["actor_state"], strict=True)
                print(
                    f"Loaded BC-SAC pretrained actor (val_mse: {checkpoint.get('val_mse', 'N/A')})"
                )
            except Exception as e:
                print(f"  Warning: Partial actor load: {e}")
        else:
            print("Warning: No actor_state found in BC-SAC checkpoint")

        if "q1_state" in checkpoint:
            try:
                q1.load_state_dict(checkpoint["q1_state"], strict=True)
                log_success("Loaded BC-SAC pretrained Q1")
            except Exception as e:
                log_warning(f"Partial Q1 load: {e}")
        if "q2_state" in checkpoint:
            try:
                q2.load_state_dict(checkpoint["q2_state"], strict=True)
                log_success("Loaded BC-SAC pretrained Q2")
            except Exception as e:
                log_warning(f"Partial Q2 load: {e}")
        if "q1_target_state" in checkpoint:
            try:
                q1_target.load_state_dict(checkpoint["q1_target_state"], strict=True)
                log_success("Loaded BC-SAC pretrained Q1 target")
            except Exception as e:
                log_warning(f"Partial Q1 target load: {e}")
        if "q2_target_state" in checkpoint:
            try:
                q2_target.load_state_dict(checkpoint["q2_target_state"], strict=True)
                log_success("Loaded BC-SAC pretrained Q2 target")
            except Exception as e:
                log_warning(f"Partial Q2 target load: {e}")

    log_success("Target network initialized")

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

    log_info("Creating replay buffer...")
    storage = LazyTensorStorage(max_size=cfg.replay_size)

    use_per = bool(getattr(cfg, "use_per", True))

    if use_per:
        log_info("Using PrioritizedReplayBuffer with LazyTensorStorage")
        rb = PrioritizedReplayBuffer(
            alpha=cfg.per_alpha,
            beta=cfg.per_beta,
            storage=storage,
            batch_size=cfg.batch_size,
            collate_fn=_collate_sequence_batch,
        )
    else:
        log_info("Using plain UniformReplayBuffer with LazyTensorStorage")
        rb = ReplayBuffer(
            storage=storage,
            batch_size=cfg.batch_size,
            collate_fn=_collate_sequence_batch,
        )

    replay_buffer_path = getattr(cfg, "replay_buffer_path", None)
    if replay_buffer_path and Path(replay_buffer_path).exists():
        log_info(f"Loading replay buffer from {replay_buffer_path}...")
        try:
            import pickle

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

    use_async = bool(getattr(cfg, "use_async", False))
    if use_async:
        log_warning("cfg.use_async=true is ignored when using TorchRL SACTrainer (sync collector path)")

    if getattr(cfg, "use_expert_demonstrations", False):
        log_warning(
            "Skipping expert demonstrations in SACTrainer path: current demo loader is wired "
            "for the legacy custom learner/replay schema."
        )

    if getattr(cfg, "save_interval_replaybuffer", None) is not None:
        log_warning("save_interval_replaybuffer is ignored in SACTrainer path")

    if bool(getattr(cfg, "ac_reset_interval_steps", 0)):
        log_warning("Periodic AC process restarts are not integrated in SACTrainer path")

    print_banner("Training Started")
    print_section_header("TorchRL SACTrainer Loop")

    trainer, loss_module = _build_sac_trainer(cfg, env, actor, q1, rb, device)
    trainer.train()

    save_dir = Path(getattr(cfg, "checkpoint_dir", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = {
        "actor_state": actor.state_dict(),
        "q1_state": q1.state_dict(),
        "q2_state": q2.state_dict(),
        "q1_target_state": q1_target.state_dict(),
        "q2_target_state": q2_target.state_dict(),
        "loss_module_state": loss_module.state_dict(),
        "steps": int(getattr(cfg, "total_steps", 0)),
    }
    torch.save(final_ckpt, save_dir / "sac_last.pt")
    log_success(f"Saved final checkpoint: {save_dir / 'sac_last.pt'}")

    try:
        if wandb.run is not None:
            wandb.finish()
            log_success("WandB finished. Training complete!")
        else:
            log_success("Training complete!")
    except Exception as e:
        log_warning(f"WandB finish failed (non-fatal): {e}")
        log_success("Training complete (W&B connection was already closed).")


@cli_command(group="ac", name="train", help="Train SAC agent in Assetto Corsa")
def train():
    """Train SAC agent in Assetto Corsa."""
    _do_train()


if __name__ == "__main__":
    train()
