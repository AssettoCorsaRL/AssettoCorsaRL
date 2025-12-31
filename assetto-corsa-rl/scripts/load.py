import argparse
import glob
import os
import re
from pathlib import Path
import sys

import torch
from tensordict import TensorDict

# Allow running this file directly (not as a package).
# When executed as a script, the project `src` directory may not be on sys.path,
# so insert it if necessary and import the package normally.
try:
    from assetto_corsa_rl.env import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore
    from assetto_corsa_rl.train.train_utils import fix_action_shape  # type: ignore
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.env import create_gym_env  # type: ignore
    from assetto_corsa_rl.model.sac import SACPolicy, get_device  # type: ignore
    from assetto_corsa_rl.train.train_utils import fix_action_shape  # type: ignore


def find_latest_checkpoint(models_dir: str = "models") -> str:
    pattern = os.path.join(models_dir, "sac_checkpoint_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {models_dir}")

    def extract_step(p):
        m = re.search(r"sac_checkpoint_(\d+)\.pt", p)
        if m:
            return int(m.group(1))
        return 0

    files_sorted = sorted(files, key=lambda p: extract_step(p), reverse=True)
    return files_sorted[0]


def load_checkpoint(path: str, device: torch.device = None):
    map_loc = device if device is not None else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_loc)
    return ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., cpu, cuda:0). If omitted, `get_device()` is used.",
    )
    p.add_argument("--render", action="store_true", help="Render the environment")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions when possible",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing checkpoints",
    )
    return p.parse_args()


def build_env(device, render: bool = False):
    mode = "human" if render else None
    # single env for evaluation
    env = create_gym_env(device=device, num_envs=1, render_mode=mode)
    return env


def main():
    args = parse_args()

    # device
    device = torch.device(args.device) if args.device else get_device()
    print("Using device:", device)

    # checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        try:
            ckpt_path = find_latest_checkpoint(args.models_dir)
            print("Found latest checkpoint:", ckpt_path)
        except FileNotFoundError as e:
            print(e)
            return

    ckpt = load_checkpoint(ckpt_path, device=device)

    # Build env
    env = build_env(device, render=args.render)

    # Extract minimal model cfg from checkpoint if present
    # Fallback values chosen to match the training defaults used in this repo
    num_cells = ckpt.get("num_cells", 256)
    use_noisy = ckpt.get("use_noisy", True)
    noise_sigma = ckpt.get("noise_sigma", 0.5)

    # Build policy
    print("Constructing policy...")
    policy = SACPolicy(
        env=env,
        num_cells=num_cells,
        device=device,
        use_noisy=use_noisy,
        noise_sigma=noise_sigma,
    )

    modules = policy.modules()

    def _load_if_present(mod, key):
        if key in ckpt and hasattr(mod, "load_state_dict") and mod is not None:
            state = ckpt[key]
            # Try to load non-strictly first to tolerate missing/unexpected keys
            try:
                res = mod.load_state_dict(state, strict=False)
                mk = getattr(res, "missing_keys", None)
                uk = getattr(res, "unexpected_keys", None)
                if uk:
                    print(f"Warning: unexpected keys when loading {key}: {uk}")
                if mk:
                    print(f"Warning: missing keys when loading {key}: {mk}")
                print(f"Loaded {key} into module (strict=False)")
                return
            except Exception:
                pass

            # If the above fails, try stripping a leading "module." prefix from each key and retry
            if isinstance(state, dict):
                stripped = {re.sub(r"^module\.", "", k): v for k, v in state.items()}
                try:
                    res = mod.load_state_dict(stripped, strict=False)
                    mk = getattr(res, "missing_keys", None)
                    uk = getattr(res, "unexpected_keys", None)
                    if uk:
                        print(
                            f"Warning: unexpected keys when loading {key} after stripping prefix: {uk}"
                        )
                    if mk:
                        print(
                            f"Warning: missing keys when loading {key} after stripping prefix: {mk}"
                        )
                    print(
                        f"Loaded {key} into module after stripping 'module.' prefix (strict=False)"
                    )
                    return
                except Exception as e2:
                    print(f"Warning: failed to load {key}: {e2}")
            else:
                print(f"Warning: failed to load {key}: state dict is not a mapping")

    _load_if_present(modules.get("actor"), "actor_state")
    _load_if_present(modules.get("q1"), "q1_state")
    _load_if_present(modules.get("q2"), "q2_state")
    _load_if_present(modules.get("value"), "value_state")

    for m in modules.values():
        try:
            m.to(device)
        except Exception:
            pass
        try:
            m.eval()
        except Exception:
            pass

    # run episodes
    episodes = int(args.episodes)
    returns = []

    for ep in range(episodes):
        td = env.reset()
        inner = td["next"] if "next" in td.keys() else td
        pixels = inner["pixels"].to(device)

        done = torch.zeros(1, dtype=torch.bool)
        total_reward = 0.0

        step = 0
        while not done.item():
            # Build actor input
            actor_td = TensorDict({"pixels": pixels}, batch_size=[1])

            # If noisy nets used, resample noise per step for exploration-like behaviour
            if use_noisy:
                try:
                    for m in modules["actor"].modules():
                        if hasattr(m, "sample_noise"):
                            m.sample_noise()
                except Exception:
                    pass

            # Try deterministic if requested
            action = None
            try:
                if args.deterministic:
                    out = modules["actor"](actor_td, deterministic=True)
                else:
                    out = modules["actor"](actor_td)
            except TypeError:
                out = modules["actor"](actor_td)

            # Prefer explicit 'action' key
            if "action" in out.keys():
                action = out["action"]
            elif "loc" in out.keys():
                # use mean (loc) if available
                action = out["loc"]
            else:
                # last resort: try to sample using module's call
                try:
                    action = out
                except Exception:
                    raise RuntimeError(
                        "Actor output did not contain an action or loc key"
                    )

            action = fix_action_shape(
                action, batch_size=1, action_dim=env.action_spec.shape[-1]
            )
            # Ensure action on correct device
            if isinstance(action, torch.Tensor):
                action = action.to(device)

            action_step = action
            # Expand to env step shape expected by environment
            action_td = TensorDict({"action": action_step}, batch_size=[1])

            td_next = env.step(action_td)
            inner_next = td_next["next"] if "next" in td_next.keys() else td_next

            # gather reward & done info
            r = 0.0
            if "reward" in inner_next.keys():
                r = float(inner_next["reward"].view(-1).sum().item())
            elif "rewards" in inner_next.keys():
                r = float(inner_next["rewards"].view(-1).sum().item())

            total_reward += r

            # update for next step
            pixels = inner_next["pixels"].to(device)

            # assemble done
            done = torch.zeros(1, dtype=torch.bool)
            for key in ("done", "terminated", "truncated"):
                if key in inner_next.keys():
                    try:
                        done |= inner_next[key].view(1).to(torch.bool)
                    except Exception:
                        pass

            step += 1
            if step % 100 == 0:
                print(f"Episode {ep+1}, step {step}, reward so far {total_reward:.2f}")

        print(f"Episode {ep+1} finished in {step} steps, return: {total_reward:.2f}")
        returns.append(total_reward)

    if len(returns) > 0:
        print(
            f"Average return over {len(returns)} episodes: {sum(returns)/len(returns):.2f}"
        )


if __name__ == "__main__":
    main()
