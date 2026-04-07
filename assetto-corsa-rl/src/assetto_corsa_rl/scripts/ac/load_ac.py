"""
Usage:
    acrl ac test --checkpoint D:/acrl/AssetoCorsaRL/models/sac_last.pt --vae-checkpoint loss=0.1050.ckpt
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import time
import os
import subprocess
from pathlib import Path
import torch
from torch import multiprocessing
import click
import numpy as np
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.ac_env import create_transformed_env
from assetto_corsa_rl.model.sac import SACPolicy
from assetto_corsa_rl.train.train_utils import activate_ac_window, kill_all_ac_instances

from assetto_corsa_rl.cli_registry import cli_command, cli_option, load_cfg_from_yaml


def get_device():
    """Determine the appropriate device for training"""
    is_fork = multiprocessing.get_start_method() == "fork"
    if torch.cuda.is_available() and not is_fork:
        return torch.device(0)
    return torch.device("cpu")


def _is_ac_running() -> bool:
    try:
        proc_list = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq acs.exe"], capture_output=True, text=True
        )
        return "acs.exe" in (proc_list.stdout or "").lower()
    except Exception:
        return False


def _ensure_ac_running(auto_launch: bool, ac_exe_path: str | None, startup_wait: float) -> None:
    if _is_ac_running():
        print("Assetto Corsa is already running.")
        try:
            activate_ac_window()
            print("Activated Assetto Corsa window.")
        except Exception:
            pass
        return

    if not auto_launch:
        print("Assetto Corsa is not running. Start it manually or use --auto-launch.")
        return

    default_path = r"D:\Steam\steamapps\common\assettocorsa\acs.exe"
    resolved = ac_exe_path or os.getenv("ASSETTO_CORSA_EXE") or default_path
    exe_path = Path(resolved)

    if not exe_path.exists():
        print(
            f"Assetto Corsa not running and executable not found at: {exe_path}\n"
            "Set --ac-exe-path or ASSETTO_CORSA_EXE to your acs.exe path."
        )
        return

    # Kill any existing AC instances before launching
    print("[AC] Cleaning up any existing Assetto Corsa instances...")
    kill_all_ac_instances()

    print(f"Launching Assetto Corsa from: {exe_path}")
    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
    time.sleep(max(0.0, float(startup_wait)))
    try:
        activate_ac_window()
    except Exception:
        pass
    print("Assetto Corsa launch attempted.")


def _resolve_assetto_base_env(env):
    """Best-effort resolution of the underlying AssettoCorsa env from wrappers."""
    queue = [env]
    visited = set()

    while queue:
        node = queue.pop(0)
        if node is None:
            continue

        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        if hasattr(node, "controller") and hasattr(node, "telemetry"):
            return node

        for attr in ("base_env", "_env", "env"):
            child = getattr(node, attr, None)
            if child is not None:
                queue.append(child)

        base_envs = getattr(node, "base_envs", None)
        if isinstance(base_envs, (list, tuple)):
            queue.extend(base_envs)

    return None


def _extract_action_triplet(action_tensor):
    """Convert policy action [steer, accel] into steer/throttle/brake floats in [0,1]/[-1,1]."""
    flat = action_tensor.detach().to("cpu").float().view(-1)
    steer = float(np.clip(flat[0].item(), -1.0, 1.0))
    accel = float(np.clip(flat[1].item(), -1.0, 1.0))
    throttle = max(0.0, accel)
    brake = max(0.0, -accel)
    return steer, throttle, brake


def _get_live_inputs(assetto_env):
    """Read latest in-game control inputs from telemetry payload, if available."""
    if assetto_env is None:
        return None

    last_obs = getattr(assetto_env, "_last_obs", None)
    if not isinstance(last_obs, dict):
        return None

    inputs = last_obs.get("inputs", None)
    if not isinstance(inputs, dict):
        return None

    try:
        return {
            "gas": float(inputs.get("gas", 0.0)),
            "brake": float(inputs.get("brake", 0.0)),
            "steer": float(inputs.get("steer", 0.0)),
        }
    except Exception:
        return None


@cli_command(group="ac", name="test", help="Test trained BC-SAC policy in Assetto Corsa")
@cli_option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to BC-SAC checkpoint",
)
@cli_option(
    "--vae-checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to VAE checkpoint",
)
@cli_option("--max-steps", default=10000, help="Maximum steps per episode")
@cli_option("--episodes", default=5, help="Number of episodes to run")
@cli_option("--render", is_flag=True, help="Render the environment")
@cli_option(
    "--deterministic/--stochastic",
    default=True,
    help="Use deterministic policy actions (recommended for evaluation)",
)
@cli_option(
    "--show-critic/--no-show-critic",
    default=True,
    help="Print critic min-Q estimate (expected discounted return) during rollout",
)
@cli_option(
    "--low-speed-truncate-seconds",
    default=20.0,
    type=float,
    help="Seconds below low-speed threshold before truncating an episode",
)
@cli_option(
    "--auto-launch/--no-auto-launch",
    default=True,
    help="Auto-launch Assetto Corsa if acs.exe is not running",
)
@cli_option(
    "--ac-exe-path",
    default=None,
    help="Path to acs.exe (overrides ASSETTO_CORSA_EXE env var)",
)
@cli_option(
    "--startup-wait",
    default=20.0,
    type=float,
    help="Seconds to wait after launching Assetto Corsa",
)
@cli_option(
    "--refocus-every-steps",
    default=250,
    type=int,
    help="How often to bring Assetto Corsa window to foreground during evaluation (0=disabled)",
)
def test(
    checkpoint,
    vae_checkpoint,
    max_steps,
    episodes,
    render,
    deterministic,
    show_critic,
    low_speed_truncate_seconds,
    auto_launch,
    ac_exe_path,
    startup_wait,
    refocus_every_steps,
):
    """Load and test a trained BC-SAC policy in Assetto Corsa."""
    checkpoint = Path(checkpoint)
    vae_checkpoint = Path(vae_checkpoint)

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {checkpoint}...")
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config", {})
    num_cells = config.get("num_cells", 256)
    ckpt_use_noisy = bool(config.get("use_noisy", False))
    ckpt_noise_sigma = float(config.get("noise_sigma", 0.5))

    print(f"\nCheckpoint info:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Val MSE: {ckpt.get('val_mse', 'N/A')}")
    print(f"  Num cells: {num_cells}")
    print(f"  VAE checkpoint: {config.get('vae_checkpoint_path', 'N/A')}")
    print(f"  use_noisy: {ckpt_use_noisy}")
    print(f"  noise_sigma: {ckpt_noise_sigma}")

    cfg = load_cfg_from_yaml()
    print(cfg)

    _ensure_ac_running(auto_launch=auto_launch, ac_exe_path=ac_exe_path, startup_wait=startup_wait)

    print("\nCreating Assetto Corsa environment...")
    env = create_transformed_env(
        racing_line_path=getattr(cfg, "racing_line_path", "racing_lines.json"),
        device=device,
        image_shape=(84, 84),
        frame_stack=3,
        input_config=getattr(cfg, "input_config", None),
        use_ac_ai_racer=False,
        low_speed_truncate_seconds=float(low_speed_truncate_seconds),
        normalize_observations=getattr(cfg, "normalize_observations", False),
        normalization_bounds=getattr(cfg, "normalization_bounds", None),
    )
    assetto_env = _resolve_assetto_base_env(env)
    if assetto_env is None:
        print("[Control] Warning: could not resolve base AssettoCorsa env for live input checks.")

    print("\nInitializing policy...")
    agent = SACPolicy(
        env=env,
        num_cells=num_cells,
        device=device,
        use_noisy=ckpt_use_noisy,
        noise_sigma=ckpt_noise_sigma,
        actor_dropout=0.0,
        vae_checkpoint_path=str(vae_checkpoint),
    )

    modules = agent.modules()
    actor = modules["actor"]
    q1 = modules["q1"]
    q2 = modules["q2"]

    print("\nLoading actor weights...")
    actor.load_state_dict(ckpt["actor_state"], strict=False)

    print("Loading critic weights...")
    q1_loaded = False
    q2_loaded = False
    if "q1_state" in ckpt:
        q1.load_state_dict(ckpt["q1_state"], strict=False)
        q1_loaded = True
        print("  Loaded q1_state")
    else:
        print("  q1_state not found in checkpoint")

    if "q2_state" in ckpt:
        q2.load_state_dict(ckpt["q2_state"], strict=False)
        q2_loaded = True
        print("  Loaded q2_state")
    else:
        print("  q2_state not found in checkpoint")

    if show_critic and not (q1_loaded and q2_loaded):
        print(
            "[Critic] --show-critic requested, but q1_state/q2_state are missing. "
            "Disabling critic display."
        )
        show_critic = False
    elif show_critic:
        print("[Critic] Showing min(Q1, Q2) as expected discounted return estimate.")

    actor.eval()
    q1.eval()
    q2.eval()

    print(f"\n{'='*60}")
    print("Starting evaluation...")
    print(f"{'='*60}")
    print(f"Action mode: {'deterministic (mode)' if deterministic else 'stochastic (sampled)'}")
    print(f"Low-speed truncate window: {float(low_speed_truncate_seconds):.1f}s")
    print("\nWaiting for Assetto Corsa connection...")
    print("Make sure the AC_RL app is running in Assetto Corsa!")

    episode_rewards = []
    episode_lengths = []
    critic_values = []
    critic_runtime_enabled = show_critic

    td = env.reset()

    for episode in range(episodes):
        print(f"\n{'='*60}\nEpisode {episode + 1}/{episodes}\n{'='*60}")

        try:
            activate_ac_window()
        except Exception:
            pass

        td = env.reset()
        time.sleep(2)

        episode_reward = 0.0
        steps = 0
        done = False
        episode_critic_values = []
        control_mismatch_streak = 0
        last_live_inputs = None

        while not done and steps < max_steps:
            critic_q = None
            with torch.no_grad():
                was_unbatched = td.batch_size == torch.Size([])
                if was_unbatched:
                    curr_td = td.unsqueeze(0)
                else:
                    curr_td = td

                exploration_mode = (
                    ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
                )
                with set_exploration_type(exploration_mode):
                    actor_out = actor(curr_td)
                action_batched = actor_out["action"]

                if critic_runtime_enabled:
                    try:
                        curr_pixels = curr_td.get("pixels")
                        curr_vector = curr_td.get("vector") if "vector" in curr_td.keys() else None
                        q1_pred = q1.module(
                            pixels=curr_pixels, action=action_batched, vector=curr_vector
                        )
                        q2_pred = q2.module(
                            pixels=curr_pixels, action=action_batched, vector=curr_vector
                        )
                        critic_q = torch.min(q1_pred, q2_pred).mean().item()
                        episode_critic_values.append(critic_q)
                    except Exception as exc:
                        print(f"[Critic] Failed to compute Q estimate, disabling display: {exc}")
                        critic_runtime_enabled = False

                action = action_batched

                # Squeeze back to unbatched form if needed
                if was_unbatched:
                    action = action.squeeze(0)

            cmd_steer, cmd_throttle, cmd_brake = _extract_action_triplet(action)

            td.set("action", action)
            td = env.step(td)

            reward = td.get(("next", "reward")).item()
            done_flag = False
            terminated_flag = False
            truncated_flag = False
            try:
                done_flag = bool(td.get(("next", "done")).any().item())
            except Exception:
                done_flag = False
            try:
                terminated_flag = bool(td.get(("next", "terminated")).any().item())
            except Exception:
                terminated_flag = False
            try:
                truncated_flag = bool(td.get(("next", "truncated")).any().item())
            except Exception:
                truncated_flag = False

            done = bool(done_flag or terminated_flag or truncated_flag)
            episode_reward += reward
            steps += 1

            if refocus_every_steps > 0 and steps % refocus_every_steps == 0:
                try:
                    activate_ac_window()
                except Exception:
                    pass

            live_inputs = _get_live_inputs(assetto_env)
            if live_inputs is not None:
                last_live_inputs = live_inputs
                command_strength = max(abs(cmd_steer), cmd_throttle, cmd_brake)
                max_err = max(
                    abs(live_inputs["steer"] - cmd_steer),
                    abs(live_inputs["gas"] - cmd_throttle),
                    abs(live_inputs["brake"] - cmd_brake),
                )
                if command_strength > 0.20 and max_err > 0.35:
                    control_mismatch_streak += 1
                    if control_mismatch_streak == 30:
                        print(
                            "[Control] Command/input mismatch detected for 30 steps; "
                            "attempting to re-focus Assetto Corsa window."
                        )
                        try:
                            activate_ac_window()
                        except Exception:
                            pass
                else:
                    control_mismatch_streak = 0

            td = step_mdp(td)

            if steps % 100 == 0:
                cmd_text = (
                    f"cmd(gas={cmd_throttle:.2f}, brake={cmd_brake:.2f}, steer={cmd_steer:.2f})"
                )
                obs_text = ""
                if last_live_inputs is not None:
                    obs_text = (
                        " "
                        f"obs(gas={last_live_inputs['gas']:.2f}, "
                        f"brake={last_live_inputs['brake']:.2f}, "
                        f"steer={last_live_inputs['steer']:.2f})"
                    )
                if critic_runtime_enabled and critic_q is not None:
                    print(
                        f"  Step {steps}: reward={episode_reward:.2f} "
                        f"critic_min_q={critic_q:.3f} action={action} {cmd_text}{obs_text}"
                    )
                else:
                    print(
                        f"  Step {steps}: reward={episode_reward:.2f} "
                        f"action={action} {cmd_text}{obs_text}"
                    )

            if done and (terminated_flag or truncated_flag):
                if truncated_flag and not terminated_flag:
                    print(
                        "  Episode ended by truncation (likely low_speed, telemetry timeout, "
                        "lap complete, or time limit)."
                    )
                elif terminated_flag:
                    print("  Episode ended by termination.")

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if episode_critic_values:
            critic_values.extend(episode_critic_values)
        print(f"\nEpisode {episode + 1} finished: Reward: {episode_reward:.2f}, Steps: {steps}")
        if episode_critic_values:
            print(
                f"Episode {episode + 1} critic min-Q mean: {np.mean(episode_critic_values):.3f} "
                f"± {np.std(episode_critic_values):.3f}"
            )
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    if critic_values:
        print(f"Average critic min-Q: {np.mean(critic_values):.3f} ± {np.std(critic_values):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test()
