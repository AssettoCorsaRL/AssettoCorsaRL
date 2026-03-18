"""
Usage:
    acrl ac test --checkpoint models\sac_best.pt --vae-checkpoint loss=0.1050.ckpt
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
from torchrl.envs.utils import step_mdp

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.ac_env import create_transformed_env
from assetto_corsa_rl.model.sac import SACPolicy
from assetto_corsa_rl.train.train_utils import activate_ac_window

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

    print(f"Launching Assetto Corsa from: {exe_path}")
    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
    time.sleep(max(0.0, float(startup_wait)))
    try:
        activate_ac_window()
    except Exception:
        pass
    print("Assetto Corsa launch attempted.")


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
    default=10.0,
    type=float,
    help="Seconds to wait after launching Assetto Corsa",
)
def test(
    checkpoint,
    vae_checkpoint,
    max_steps,
    episodes,
    render,
    auto_launch,
    ac_exe_path,
    startup_wait,
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
        normalize_observations=getattr(cfg, "normalize_observations", False),
        normalization_bounds=getattr(cfg, "normalization_bounds", None),
    )

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
    print("\nLoading actor weights...")
    actor.load_state_dict(ckpt["actor_state"], strict=False)
    actor.eval()

    print(f"\n{'='*60}")
    print("Starting evaluation...")
    print(f"{'='*60}")
    print("\nWaiting for Assetto Corsa connection...")
    print("Make sure the AC_RL app is running in Assetto Corsa!")

    episode_rewards = []
    episode_lengths = []

    td = env.reset()

    input("Press Enter when in position...")
    for episode in range(episodes):
        print(f"\n{'='*60}\nEpisode {episode + 1}/{episodes}\n{'='*60}")

        td = env.reset()
        time.sleep(2)

        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            with torch.no_grad():
                if td.batch_size == torch.Size([]):
                    curr_td = td.unsqueeze(0)
                else:
                    curr_td = td

                actor_out = actor(curr_td)
                action = actor_out["loc"]

            td.set("action", action)
            td = env.step(td)

            reward = td.get(("next", "reward")).item()
            done = td.get(("next", "done")).any().item()
            episode_reward += reward
            steps += 1

            td = step_mdp(td)

            if steps % 100 == 0:
                print(f"  Step {steps}: reward={episode_reward:.2f} action={action}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"\nEpisode {episode + 1} finished: Reward: {episode_reward:.2f}, Steps: {steps}")
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test()
