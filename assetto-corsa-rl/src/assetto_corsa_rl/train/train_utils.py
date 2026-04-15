import math
import os
import subprocess
import time as _time

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.envs.transforms import Transform


class PixelCompressionTransform(Transform):
    """Replay transform that stores pixels as uint8 and decodes to float32 on sample."""

    def __init__(self, in_keys=None):
        if in_keys is None:
            # Support collector-style nested next pixels and legacy next_pixels layout.
            in_keys = ["pixels", ("next", "pixels"), "next_pixels"]
        super().__init__(
            in_keys=in_keys,
            out_keys=in_keys,
            in_keys_inv=in_keys,
            out_keys_inv=in_keys,
        )

    @staticmethod
    def _encode_pixels(obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            return obs
        if obs.dtype == torch.uint8:
            return obs
        if obs.dtype.is_floating_point:
            return (obs.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return obs.to(torch.uint8)

    @staticmethod
    def _decode_pixels(obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            return obs
        if obs.dtype == torch.uint8:
            return obs.to(torch.float32) / 255.0
        if obs.dtype.is_floating_point:
            # If already normalized, keep as-is to avoid accidental double division.
            if obs.numel() == 0:
                return obs.to(torch.float32)
            max_val = obs.detach().amax()
            if torch.isfinite(max_val) and max_val <= 1.0 + 1e-6:
                return obs.to(torch.float32)
            return obs.to(torch.float32) / 255.0
        return obs.to(torch.float32) / 255.0

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return self._encode_pixels(obs)

    def _inv_apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return self._decode_pixels(obs)

    def _call(self, tensordict):
        for key in self.in_keys:
            value = tensordict.get(key, None)
            if isinstance(value, torch.Tensor):
                tensordict.set(key, self._apply_transform(value))
        return tensordict

    def _inv_call(self, tensordict):
        for key in self.in_keys_inv:
            value = tensordict.get(key, None)
            if isinstance(value, torch.Tensor):
                tensordict.set(key, self._inv_apply_transform(value))
        return tensordict


class OrnsteinUhlenbeckNoise:
    """Temporally-correlated action noise for coherent exploration.

    Instead of independent random actions each step (which causes jittering),
    OU noise produces smooth trajectories — the agent commits to a maneuver
    for several steps before gradually drifting to another.

    The API is flexible: ``theta``, ``sigma`` and ``mu`` may be scalars or
    arrays/tensors with length equal to ``action_dim``.  If arrays are provided,
    they are broadcast across ``num_envs`` so each environment shares the same
    parameter vector.

    Equation:
        dx = theta * (mu - x) * dt + sigma * sqrt(dt) * N(0, 1)
    """

    def __init__(
        self,
        action_dim: int,
        num_envs: int = 1,
        theta: float | list[float] | torch.Tensor = 0.15,
        sigma: float | list[float] | torch.Tensor = 0.3,
        mu: float | list[float] | torch.Tensor | None = None,
        dt: float = 1.0,
        device=None,
    ):
        self.dt = dt
        self.device = device or torch.device("cpu")

        def _to_tensor(x, default):
            if isinstance(x, (list, tuple)):
                t = torch.tensor(x, dtype=torch.float32, device=self.device)
            elif isinstance(x, torch.Tensor):
                t = x.to(self.device).float()
            else:
                t = torch.tensor([x], dtype=torch.float32, device=self.device)

            if t.numel() > action_dim:
                t = t.flatten()[:action_dim]
            elif t.numel() < action_dim and t.numel() > 1:
                pad = torch.zeros(action_dim - t.numel(), device=self.device)
                t = torch.cat([t.flatten(), pad])

            # expand to (num_envs, action_dim) if needed
            if t.numel() == 1:
                t = t.expand(num_envs, action_dim)
            else:
                t = t.view(1, -1).expand(num_envs, -1)
            return t

        self.theta = _to_tensor(theta, 0.15)
        self.sigma = _to_tensor(sigma, 0.3)
        if mu is not None:
            self.mu = _to_tensor(mu, 0.0)
        else:
            self.mu = torch.zeros((num_envs, action_dim), device=self.device)
        self.state = self.mu.clone()

    def reset(self, env_indices=None):
        """Reset noise state (e.g. on episode end)."""
        if env_indices is None:
            self.state = self.mu.clone()
        else:
            self.state[env_indices] = self.mu[env_indices].clone()

    def sample(self) -> torch.Tensor:
        """Return next OU noise sample (num_envs, action_dim)."""
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * math.sqrt(
            self.dt
        ) * torch.randn_like(self.state)
        self.state = self.state + dx
        return self.state.clone()


def reduce_value_to_batch(x, batch_size):
    try:
        v = x.get("value") if hasattr(x, "get") else x
        if v.shape[0] == batch_size:
            if v.ndim == 1:
                return v.view(-1, 1)
            return v.flatten(1).mean(dim=1, keepdim=True)
        return v.view(batch_size, -1).mean(dim=1, keepdim=True)
    except Exception:
        return None


def sample_random_actions(num_envs, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.empty(num_envs, 2, device=device).uniform_(-1.0, 1.0)


def sample_random_action(n=1, dev=None):
    return sample_random_actions(n, device=dev)


def get_inner(td):
    return td["next"] if "next" in td.keys() else td


def extract_reward_and_done(td, num_envs, device):
    td = get_inner(td)
    if "reward" in td.keys():
        rewards = td["reward"].view(num_envs).to(device)
    elif "rewards" in td.keys():
        rewards = td["rewards"].view(num_envs).to(device)
    else:
        raise KeyError(f"Unexpected TensorDict structure. Keys: {td.keys()}")

    def _flag_or_zeros(key: str) -> torch.Tensor:
        if key in td.keys():
            return td[key].view(num_envs).to(device).to(torch.bool)
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
    dones |= _flag_or_zeros("done")
    dones |= _flag_or_zeros("terminated")
    dones |= _flag_or_zeros("truncated")
    return rewards, dones


def expand_actions_for_envs(actions, target_batch):
    if isinstance(target_batch, (tuple, list, torch.Size)) and len(target_batch) > 1:
        extra = target_batch[1:]
        new_shape = (actions.shape[0],) + (1,) * len(extra) + (actions.shape[1],)
        expand_shape = (actions.shape[0],) + tuple(extra) + (actions.shape[1],)
        return actions.view(new_shape).expand(expand_shape)
    return actions


def add_transition(rb, i, pixels, next_pixels, action, reward, done, vector=None, next_vector=None):
    """Add a single env transition to the replay buffer.

    Args:
        rb: replay buffer
        i: env index
        pixels: [num_envs, C, H, W] current pixels
        next_pixels: [num_envs, C, H, W] next pixels
        action: [num_envs, action_dim]
        reward: [num_envs]
        done: [num_envs]
        vector: optional [num_envs, obs_dim] telemetry vector
        next_vector: optional [num_envs, obs_dim] next telemetry vector
    """
    packed_pixels = pixels[i]
    packed_next = next_pixels[i]
    action_cpu = action[i].to(torch.float32).cpu()
    reward_cpu = reward[i].unsqueeze(0).cpu()
    done_cpu = done[i].unsqueeze(0).cpu()

    transition = TensorDict(
        {
            "pixels": packed_pixels,
            "action": action_cpu,
            "reward": reward_cpu,
            "next_pixels": packed_next,
            "done": done_cpu,
        },
        batch_size=[],
    )

    if vector is not None:
        transition["vector"] = vector[i].to(torch.float32).cpu()
    if next_vector is not None:
        transition["next_vector"] = next_vector[i].to(torch.float32).cpu()

    rb.add(transition)


def kill_all_ac_instances(max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """Kill all running Assetto Corsa instances.

    Uses taskkill to cleanly terminate all acs.exe processes.
    Retries up to max_retries times with delays to ensure termination.

    Args:
        max_retries: Maximum number of kill attempts
        retry_delay: Delay in seconds between retries

    Returns:
        True if all instances were successfully killed or none were running, False if some persisted
    """
    try:
        for attempt in range(max_retries):
            proc_list = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq acs.exe"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if "acs.exe" not in (proc_list.stdout or "").lower():
                if attempt > 0:
                    print("[AC] All Assetto Corsa instances terminated successfully.")
                return True

            print(
                f"[AC] Terminating all Assetto Corsa instances (attempt {attempt + 1}/{max_retries})..."
            )
            subprocess.run(["taskkill", "/IM", "acs.exe", "/F"], capture_output=True, timeout=5)

            if attempt < max_retries - 1:
                _time.sleep(retry_delay)

        proc_list = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq acs.exe"], capture_output=True, text=True, timeout=5
        )
        if "acs.exe" in (proc_list.stdout or "").lower():
            print("[AC] Warning: Could not fully terminate all Assetto Corsa instances.")
            return False
        return True

    except Exception as e:
        print(f"[AC] Error while killing AC instances: {e}")
        return False


def activate_ac_window(retries: int = 10, retry_delay: float = 2.0) -> bool:
    """Find the Assetto Corsa window and bring it to the foreground.

    Retries up to *retries* times, waiting *retry_delay* seconds between
    attempts, to give the process time to create its window after launch.
    Returns True if the window was found and activated, False otherwise.
    """
    import time as _time

    try:
        import win32gui
        import win32con
        import win32api
        import win32process
    except ImportError:
        print("[game] pywin32 not available – skipping window activation.")
        return False

    def _find():
        found = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Assetto Corsa" in title or "acs" in title.lower():
                    found.append(hwnd)
            return True

        win32gui.EnumWindows(_cb, None)
        return found[0] if found else None

    for attempt in range(retries):
        hwnd = _find()
        if hwnd:
            try:
                win32process.AttachThreadInput(
                    win32api.GetCurrentThreadId(),
                    win32process.GetWindowThreadProcessId(hwnd)[0],
                    True,
                )
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.BringWindowToTop(hwnd)
                try:
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass
                print("[game] Assetto Corsa window activated.")
                return True
            except Exception as e:
                print(f"[game] Window activation error: {e}")
                return False
        if attempt < retries - 1:
            print(f"[game] Waiting for AC window (attempt {attempt + 1}/{retries})...")
            _time.sleep(retry_delay)

    print("[game] Warning: Could not find Assetto Corsa window to activate.")
    return False


def _emit_log(message: str, log_fn=None):
    if log_fn is not None:
        try:
            log_fn(message)
            return
        except Exception:
            pass
    print(message)


def start_ac(
    ac_exe_path: str,
    startup_wait: float = 20.0,
    log_info_fn=None,
    log_success_fn=None,
    log_warning_fn=None,
) -> bool:
    """Ensure Assetto Corsa is running and focused."""
    proc_list = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq acs.exe"], capture_output=True, text=True
    )

    if "acs.exe" in (proc_list.stdout or "").lower():
        _emit_log("Assetto Corsa is already running.", log_success_fn)
        activate_ac_window()
        return True

    from pathlib import Path

    ac_path = Path(ac_exe_path)
    if not ac_path.exists():
        _emit_log(f"Assetto Corsa executable not found: {ac_exe_path}", log_warning_fn)
        return False

    _emit_log("Cleaning up any existing Assetto Corsa instances...", log_info_fn)
    kill_all_ac_instances()

    _emit_log("Launching Assetto Corsa...", log_info_fn)
    subprocess.Popen([str(ac_path)], cwd=str(ac_path.parent))
    _time.sleep(max(0.0, float(startup_wait)))
    activate_ac_window()
    _emit_log("Assetto Corsa launched.", log_success_fn)
    return True


def _to_jsonable(value):
    from pathlib import Path

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def init_wandb_trainer_logger(cfg, log_info_fn=None, log_warning_fn=None, log_success_fn=None):
    """Create TorchRL WandbLogger and apply sweep overrides when appropriate."""
    if not bool(getattr(cfg, "wandb_enabled", True)):
        _emit_log(
            "WandB disabled by config; SACTrainer metrics will only show in progress output",
            log_warning_fn,
        )
        return None

    import time
    from pathlib import Path
    from torchrl.record.loggers import WandbLogger

    wandb_cfg = {k: _to_jsonable(v) for k, v in vars(cfg).items() if not k.startswith("_")}
    exp_name = getattr(cfg, "wandb_name", None) or f"ac-sac-{int(time.time())}"
    wandb_dir = Path(getattr(cfg, "checkpoint_dir", "models")) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    trainer_logger = WandbLogger(
        exp_name=exp_name,
        project=getattr(cfg, "wandb_project", None),
        entity=getattr(cfg, "wandb_entity", None),
        config=wandb_cfg,
        save_dir=str(wandb_dir),
    )

    run = trainer_logger.experiment
    run_sweep_id = getattr(run, "sweep_id", None) if run is not None else None
    env_sweep_id = os.getenv("WANDB_SWEEP_ID")
    is_sweep_run = bool(run_sweep_id)

    if env_sweep_id and not is_sweep_run:
        _emit_log(
            "WANDB_SWEEP_ID is set but this run has no sweep_id; skipping sweep overrides.",
            log_warning_fn,
        )

    if run is not None and is_sweep_run:
        for k, v in dict(run.config).items():
            if hasattr(cfg, k) and not k.startswith("_"):
                setattr(cfg, k, v)
                _emit_log(f"[sweep] cfg.{k} = {v}", log_info_fn)

    _emit_log(f"WandB logger initialized: {getattr(run, 'name', None)}", log_success_fn)
    return trainer_logger


def finalize_wandb_run(trainer_logger, log_success_fn=None, log_warning_fn=None):
    """Close WandB run safely; never raise on shutdown."""
    try:
        if trainer_logger is not None and trainer_logger.experiment is not None:
            import wandb

            wandb.finish()
            _emit_log("WandB finished. Training complete!", log_success_fn)
        else:
            _emit_log("Training complete!", log_success_fn)
    except Exception as e:
        _emit_log(f"WandB finish failed (non-fatal): {e}", log_warning_fn)
        _emit_log("Training complete (W&B connection was already closed).", log_success_fn)


def collate_sequence_batch(batch):
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


def create_replay_buffer(cfg, log_info_fn=None, log_warning_fn=None):
    from pathlib import Path
    import shutil
    from torchrl.data.replay_buffers import PrioritizedReplayBuffer, ReplayBuffer, LazyTensorStorage

    try:
        from torchrl.data.replay_buffers import LazyMemmapStorage
    except Exception:
        LazyMemmapStorage = None

    replay_size = int(getattr(cfg, "replay_size", 100_000))
    compress_pixels = bool(getattr(cfg, "replay_compress_pixels", True))

    backend = str(getattr(cfg, "replay_storage_backend", "auto")).strip().lower()
    if backend not in {"auto", "tensor", "memmap"}:
        _emit_log(
            f"Unknown replay_storage_backend='{backend}', defaulting to 'auto'",
            log_warning_fn,
        )
        backend = "auto"

    if backend == "auto":
        memmap_threshold = max(1, int(getattr(cfg, "replay_memmap_threshold", 200_000)))
        backend = "memmap" if replay_size >= memmap_threshold else "tensor"

    frame_stack = max(1, int(getattr(cfg, "frame_stack", 3)))
    image_h = max(1, int(getattr(cfg, "image_height", 84)))
    image_w = max(1, int(getattr(cfg, "image_width", 84)))
    bytes_per_pixel = 1 if compress_pixels else 4
    approx_bytes = replay_size * frame_stack * image_h * image_w * bytes_per_pixel * 2
    approx_gb = approx_bytes / 1_000_000_000

    if compress_pixels:
        _emit_log(
            "Replay pixel compression enabled (store uint8, decode to float32 on sample).",
            log_info_fn,
        )

    storage_name = "LazyTensorStorage"
    if backend == "memmap" and LazyMemmapStorage is not None:
        memmap_dir_cfg = getattr(cfg, "replay_memmap_dir", None)
        checkpoint_dir = Path(getattr(cfg, "checkpoint_dir", "models"))
        memmap_dir = Path(memmap_dir_cfg) if memmap_dir_cfg else checkpoint_dir / "replay_memmap"
        memmap_dir.mkdir(parents=True, exist_ok=True)

        try:
            free_bytes = shutil.disk_usage(str(memmap_dir)).free
            if approx_bytes > free_bytes:
                _emit_log(
                    "Replay memmap estimate exceeds free disk space "
                    f"(need ~{approx_gb:.1f} GB, free ~{free_bytes / 1_000_000_000:.1f} GB). "
                    "Lower replay_size or point replay_memmap_dir to a drive with more space.",
                    log_warning_fn,
                )
        except Exception:
            pass

        storage = LazyMemmapStorage(max_size=replay_size, scratch_dir=str(memmap_dir))
        storage_name = "LazyMemmapStorage"
        _emit_log(
            f"Using {storage_name}(max_size={replay_size}, dir={memmap_dir}, est_pixels={approx_gb:.1f}GB)",
            log_info_fn,
        )
    else:
        if backend == "memmap" and LazyMemmapStorage is None:
            _emit_log(
                "LazyMemmapStorage unavailable in this torchrl version; falling back to LazyTensorStorage",
                log_warning_fn,
            )

        storage = LazyTensorStorage(max_size=replay_size)

        # Rough estimate for vision replay (pixels + next_pixels) to flag risky RAM setups.
        warn_threshold_gb = float(getattr(cfg, "replay_tensor_warning_gb", 16.0))
        if approx_gb >= warn_threshold_gb:
            _emit_log(
                "Replay buffer may require very high RAM "
                f"(~{approx_gb:.1f} GB for pixels/next_pixels estimate). "
                "Consider replay_storage_backend='memmap' or lowering replay_size.",
                log_warning_fn,
            )

    use_per = bool(getattr(cfg, "use_per", True))
    if use_per:
        _emit_log(f"Using PrioritizedReplayBuffer with {storage_name}", log_info_fn)
        rb = PrioritizedReplayBuffer(
            alpha=cfg.per_alpha,
            beta=cfg.per_beta,
            storage=storage,
            batch_size=cfg.batch_size,
            collate_fn=collate_sequence_batch,
        )
    else:
        _emit_log(f"Using plain UniformReplayBuffer with {storage_name}", log_info_fn)
        rb = ReplayBuffer(
            storage=storage,
            batch_size=cfg.batch_size,
            collate_fn=collate_sequence_batch,
        )

    if compress_pixels:
        rb.append_transform(PixelCompressionTransform(), invert=True)

    return rb


def maybe_restore_replay_buffer(
    rb,
    replay_buffer_path,
    log_info_fn=None,
    log_warning_fn=None,
    log_success_fn=None,
) -> bool:
    if not replay_buffer_path:
        return False

    from pathlib import Path
    import pickle

    path = Path(replay_buffer_path)
    if not path.exists():
        _emit_log(
            f"Replay buffer path specified but file not found: {replay_buffer_path}", log_warning_fn
        )
        return False

    _emit_log(f"Loading replay buffer from {replay_buffer_path}...", log_info_fn)
    try:
        if path.suffix == ".pt":
            rb_state = torch.load(str(path), weights_only=False)
        else:
            with path.open("rb") as f:
                rb_state = pickle.load(f)

        if "buffer" in rb_state:
            rb._storage._storage = rb_state["buffer"]
            _emit_log(
                f"Loaded {rb_state.get('buffer_size', 'unknown')} transitions from replay buffer",
                log_success_fn,
            )

        if "sampler_state" in rb_state:
            sampler_state = rb_state["sampler_state"]
            sampler = getattr(rb, "_sampler", None)
            if sampler_state.get("alpha") is not None and hasattr(sampler, "_alpha"):
                sampler._alpha = sampler_state["alpha"]
            if sampler_state.get("beta") is not None and hasattr(sampler, "_beta"):
                sampler._beta = sampler_state["beta"]
            _emit_log(
                f"Restored sampler state (alpha={sampler_state.get('alpha')}, beta={sampler_state.get('beta')})",
                log_success_fn,
            )

        _emit_log(
            f"Replay buffer loaded from step {rb_state.get('total_steps', 'unknown')}", log_info_fn
        )
        return True
    except Exception as e:
        _emit_log(f"Failed to load replay buffer: {e}", log_warning_fn)
        _emit_log("Starting with empty replay buffer", log_info_fn)
        return False


def warmup_actor_with_reset_td(actor, current_td, device):
    """Run one forward pass so lazy modules allocate parameters before training."""
    with torch.no_grad():
        dummy_pixels = current_td.get("pixels")
        if isinstance(dummy_pixels, torch.Tensor) and dummy_pixels.ndim == 3:
            dummy_pixels = dummy_pixels.unsqueeze(0)
        dummy_pixels = dummy_pixels.to(device)

        init_data = {"pixels": dummy_pixels}
        dummy_vector = current_td.get("vector", None)
        if dummy_vector is not None:
            if isinstance(dummy_vector, torch.Tensor) and dummy_vector.ndim == 1:
                dummy_vector = dummy_vector.unsqueeze(0)
            init_data["vector"] = dummy_vector.to(device)
        init_td = TensorDict(init_data, batch_size=[1])
        actor(init_td)


def load_policy_checkpoint(
    *,
    pretrained_path,
    bc_pretrained_path,
    actor,
    q1,
    q2,
    q1_target,
    q2_target,
    device,
    log_info_fn=None,
    log_success_fn=None,
    log_warning_fn=None,
):
    _emit_log(f"BC pretrained model: {bc_pretrained_path}", log_info_fn)

    if pretrained_path:
        _emit_log(f"Loading pretrained model from {pretrained_path}...", log_info_fn)
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if "actor_state" in checkpoint:
                actor.load_state_dict(checkpoint["actor_state"])
                _emit_log("Loaded actor state", log_success_fn)
            if "q1_state" in checkpoint:
                q1.load_state_dict(checkpoint["q1_state"])
                _emit_log("Loaded Q1 state", log_success_fn)
            if "q2_state" in checkpoint:
                q2.load_state_dict(checkpoint["q2_state"])
                _emit_log("Loaded Q2 state", log_success_fn)
            q1_target.load_state_dict(q1.state_dict())
            q2_target.load_state_dict(q2.state_dict())
            _emit_log("Copied states to target networks", log_success_fn)
            return
        except Exception as e:
            _emit_log(f"Failed to load pretrained model: {e}", log_warning_fn)
            return

    if not bc_pretrained_path:
        return

    _emit_log(f"Loading BC-SAC pretrained model from {bc_pretrained_path}...", log_info_fn)
    checkpoint = torch.load(bc_pretrained_path, map_location=device)

    if "actor_state" in checkpoint:
        try:
            actor.load_state_dict(checkpoint["actor_state"], strict=True)
            _emit_log(
                f"Loaded BC-SAC pretrained actor (val_mse: {checkpoint.get('val_mse', 'N/A')})",
                log_success_fn,
            )
        except Exception as e:
            _emit_log(f"Partial actor load: {e}", log_warning_fn)
    else:
        _emit_log("No actor_state found in BC-SAC checkpoint", log_warning_fn)

    if "q1_state" in checkpoint:
        try:
            q1.load_state_dict(checkpoint["q1_state"], strict=True)
            _emit_log("Loaded BC-SAC pretrained Q1", log_success_fn)
        except Exception as e:
            _emit_log(f"Partial Q1 load: {e}", log_warning_fn)

    if "q2_state" in checkpoint:
        try:
            q2.load_state_dict(checkpoint["q2_state"], strict=True)
            _emit_log("Loaded BC-SAC pretrained Q2", log_success_fn)
        except Exception as e:
            _emit_log(f"Partial Q2 load: {e}", log_warning_fn)

    if "q1_target_state" in checkpoint:
        try:
            q1_target.load_state_dict(checkpoint["q1_target_state"], strict=True)
            _emit_log("Loaded BC-SAC pretrained Q1 target", log_success_fn)
        except Exception as e:
            _emit_log(f"Partial Q1 target load: {e}", log_warning_fn)

    if "q2_target_state" in checkpoint:
        try:
            q2_target.load_state_dict(checkpoint["q2_target_state"], strict=True)
            _emit_log("Loaded BC-SAC pretrained Q2 target", log_success_fn)
        except Exception as e:
            _emit_log(f"Partial Q2 target load: {e}", log_warning_fn)


def warn_sactrainer_ignored_cfg(cfg, log_warning_fn=None):
    if bool(getattr(cfg, "use_async", False)):
        _emit_log(
            "cfg.use_async=true is ignored when using TorchRL SACTrainer (sync collector path)",
            log_warning_fn,
        )

    if getattr(cfg, "use_expert_demonstrations", False):
        _emit_log(
            "Skipping expert demonstrations in SACTrainer path: current demo loader is wired "
            "for the legacy custom learner/replay schema.",
            log_warning_fn,
        )

    if getattr(cfg, "save_interval_replaybuffer", None) is not None:
        _emit_log("save_interval_replaybuffer is ignored in SACTrainer path", log_warning_fn)

    if bool(getattr(cfg, "ac_reset_interval_steps", 0)):
        _emit_log(
            "Periodic AC process restarts are not integrated in SACTrainer path", log_warning_fn
        )


def fix_action_shape(a, batch_size, action_dim=None):
    if not isinstance(a, torch.Tensor):
        return a
    if a.ndim == 1:
        a = a.view(batch_size, -1)
    elif a.ndim > 2:
        a = a.view(batch_size, -1)
    if action_dim is None:
        return a
    L = a.shape[1]
    if L == action_dim:
        return a
    if L % action_dim == 0:
        return a.view(batch_size, L // action_dim, action_dim).mean(dim=1)
    return a[:, :action_dim]


def load_expert_demonstrations(
    rb,
    demo_dir: str,
    subsample: int = None,
    demo_epsilon: float = 1e-3,
    log_fn=None,
):
    """Load expert demonstrations from npz files into the replay buffer."""
    from pathlib import Path
    import numpy as np

    def _get_current_max_priority(replay_buffer) -> float:
        sampler_candidates = [
            replay_buffer,
            getattr(replay_buffer, "sampler", None),
            getattr(replay_buffer, "_sampler", None),
        ]
        attrs = ("max_priority", "_max_priority")

        for candidate in sampler_candidates:
            if candidate is None:
                continue
            for attr in attrs:
                value = getattr(candidate, attr, None)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if math.isfinite(value_f) and value_f > 0.0:
                    return value_f
        return 1.0

    def _emit(msg: str):
        if log_fn is not None:
            try:
                log_fn(msg)
                return
            except Exception:
                pass
        print(msg)

    demo_path = Path(demo_dir)
    if not demo_path.exists():
        _emit(f"[expert-demo] Directory not found: {demo_dir}")
        return 0

    demo_files = sorted(demo_path.glob("demo_batch_*.npz"))
    if not demo_files:
        _emit(f"[expert-demo] No demo_batch_*.npz files found in: {demo_dir}")
        return 0

    total_loaded = 0
    added_indices: list[int] = []
    for demo_file in demo_files:
        try:
            with np.load(demo_file, allow_pickle=True) as data:
                # Required keys
                required = ["frames", "actions", "rewards"]
                missing_required = [k for k in required if k not in data.files]
                if missing_required:
                    _emit(
                        f"[expert-demo] WARNING: {demo_file.name} missing required keys "
                        f"{missing_required}. Skipping file."
                    )
                    continue

                frames = data["frames"]
                actions = data["actions"]
                rewards = data["rewards"]

                observations = None
                if "observations" in data.files:
                    observations = data["observations"]
                else:
                    _emit(
                        f"[expert-demo] WARNING: {demo_file.name} has no 'observations'. "
                        f"Loading transitions without vector/next_vector."
                    )

                file_truncated = None
                if "truncated" in data.files:
                    file_truncated = data["truncated"]
                elif "truncateds" in data.files:
                    file_truncated = data["truncateds"]

            num_samples = len(frames)
            indices = list(range(num_samples))
            if subsample and subsample > 1:
                indices = indices[::subsample]

            for idx in indices:
                current_pixels = torch.from_numpy(frames[idx]).unsqueeze(0)
                if idx + 1 < len(frames):
                    next_pixels = torch.from_numpy(frames[idx + 1]).unsqueeze(0)
                else:
                    next_pixels = current_pixels.clone()

                action = torch.from_numpy(actions[idx]).unsqueeze(0)
                reward = torch.from_numpy(rewards[idx : idx + 1])
                done = torch.zeros(1, dtype=torch.bool)
                truncated = torch.zeros(1, dtype=torch.bool)
                if file_truncated is not None and idx < len(file_truncated):
                    truncated = torch.tensor([bool(file_truncated[idx])], dtype=torch.bool)

                transition = TensorDict(
                    {
                        "pixels": current_pixels[0],
                        "action": action[0].float().cpu(),
                        "reward": reward.float().cpu(),
                        "next_pixels": next_pixels[0],
                        "done": done.cpu(),
                        "truncated": truncated.cpu(),
                    },
                    batch_size=[],
                )

                if observations is not None:
                    transition["vector"] = torch.from_numpy(observations[idx]).float().cpu()
                    if idx + 1 < len(observations):
                        transition["next_vector"] = (
                            torch.from_numpy(observations[idx + 1]).float().cpu()
                        )
                    else:
                        transition["next_vector"] = transition["vector"].clone()

                added_idx = rb.add(transition)
                if added_idx is not None:
                    if isinstance(added_idx, torch.Tensor):
                        flat_idx = added_idx.detach().cpu().view(-1).tolist()
                        added_indices.extend(int(i) for i in flat_idx)
                    elif isinstance(added_idx, (list, tuple)):
                        added_indices.extend(int(i) for i in added_idx)
                    else:
                        try:
                            added_indices.append(int(added_idx))
                        except Exception:
                            pass
                total_loaded += 1

            # _emit(
            #     f"[expert-demo] Loaded {len(indices)} transitions from {demo_file.name}{subsample_info}"
            # )

        except Exception as e:
            _emit(f"[expert-demo] WARNING: Failed to load {demo_file.name}: {e}")
            continue

    _emit(f"[expert-demo] Total loaded transitions: {total_loaded}")

    if total_loaded > 0:
        try:
            eps = max(0.0, float(demo_epsilon))
            max_priority = _get_current_max_priority(rb)
            expert_priority = max_priority + eps

            if added_indices:
                expert_indices = torch.tensor(added_indices, dtype=torch.long)
            else:
                expert_indices = torch.arange(total_loaded, dtype=torch.long)

            expert_priorities = torch.full(
                (expert_indices.numel(),), expert_priority, dtype=torch.float32
            )
            rb.update_priority(expert_indices, expert_priorities.numpy())

            existing_expert_indices = getattr(rb, "_expert_indices", None)
            if (
                isinstance(existing_expert_indices, torch.Tensor)
                and existing_expert_indices.numel() > 0
            ):
                merged = torch.cat(
                    [existing_expert_indices.view(-1).to(torch.long), expert_indices.view(-1)]
                )
                rb._expert_indices = torch.unique(merged, sorted=True)
            else:
                rb._expert_indices = expert_indices.view(-1).to(torch.long)
            rb._expert_demo_epsilon = eps

            _emit(
                "[expert-demo] Set priority="
                f"{expert_priority:.6f} (max={max_priority:.6f} + eps={eps:.6f}) "
                f"for {expert_indices.numel()} expert transitions"
            )
        except Exception as e:
            _emit(f"[expert-demo] WARNING: Could not set priorities: {e}")

    return total_loaded
