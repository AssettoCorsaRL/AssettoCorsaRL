import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict


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


def pack_pixels(x):
    if not isinstance(x, torch.Tensor):
        return x
    return (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()


def unpack_pixels(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype == torch.uint8:
        return x.to(torch.float32) / 255.0
    if x.dtype == torch.int8:
        return x.to(torch.float32) / 127.0
    return x.to(torch.float32)


def sample_random_actions(num_envs, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    steer = torch.empty(num_envs, 1, device=device).uniform_(-1, 1)
    gas = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
    brake = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
    return torch.cat([steer, gas, brake], dim=-1)


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
    packed_pixels = pack_pixels(pixels[i])
    packed_next = pack_pixels(next_pixels[i])
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
        print(msg)
        if log_fn is not None:
            try:
                log_fn(msg)
            except Exception:
                pass

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
                        "pixels": pack_pixels(current_pixels[0]),
                        "action": action[0].float().cpu(),
                        "reward": reward.float().cpu(),
                        "next_pixels": pack_pixels(next_pixels[0]),
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
            _emit(
                "[expert-demo] Set priority="
                f"{expert_priority:.6f} (max={max_priority:.6f} + eps={eps:.6f}) "
                f"for {expert_indices.numel()} expert transitions"
            )
        except Exception as e:
            _emit(f"[expert-demo] WARNING: Could not set priorities: {e}")

    return total_loaded
