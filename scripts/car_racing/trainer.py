"""Trainer for CarRacing using IQN agent.

This module contains the cleaned `train_cartpole` function (keeps original
name for compatibility) and delegates preprocessing / logging to helpers.
"""

from typing import Optional, Callable, Sequence, List, Dict

import os
import random

import gym
import numpy as np
import torch

from models.agent.iqn_agent import IQN_Agent
from config import Config
from .writer import WandbWriter
from .utils import to_chw, reset_env, init_frame_stacks


def train_gym(
    config: Optional[Config] = None,
    env_id: str = "CarRacing-v2",
    n_envs: Optional[int] = None,
    frame_stack: int = 4,
    max_episode_time: float = 10.0,
    network: str = "",
    batch_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    exploration: str = "exp",
    render: bool = False,
    log_interval: int = 20,
    wandb_project: str = "car_racing-iqn",
    env_kwargs: Optional[Dict] = None,
    env_factory: Optional[Callable[[int], gym.Env]] = None,
    load_path: Optional[str] = None,
):
    """Train IQN on an environment"""
    if config is None:
        config = Config()

    episodes = config.episodes
    batch_size = batch_size if batch_size is not None else config.batch_size
    buffer_size = buffer_size if buffer_size is not None else config.buffer_size
    lr = config.lr
    seed = seed if seed is not None else config.seed
    n_envs = n_envs if n_envs is not None else config.n_envs

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    wandb = None
    try:
        import wandb as _wandb

        wandb = _wandb
        try:
            wandb.init(
                project=wandb_project,
                config={
                    "episodes": episodes,
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "lr": lr,
                    "seed": seed,
                },
            )
        except Exception:
            pass
    except Exception:
        wandb = None

    # create environments (allow custom factory for testing/vectorized envs)
    env_kwargs = env_kwargs or {}
    if env_factory is not None:
        envs = [env_factory(i) for i in range(n_envs)]
    else:
        envs = [gym.make(env_id, **env_kwargs) for _ in range(n_envs)]

    obs_shape_hwc = envs[0].observation_space.shape
    obs_shape = (frame_stack, obs_shape_hwc[0], obs_shape_hwc[1])
    action_size = envs[0].action_space.n

    device = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        if wandb is not None:
            wandb.config.update({"device": device})
    except Exception:
        pass

    writer = WandbWriter(wandb) if wandb is not None else WandbWriter()

    try:
        writer.log_dict({"exploration/type": exploration})
    except Exception:
        pass

    agent = IQN_Agent(
        state_size=obs_shape,
        action_size=action_size,
        network=network,
        munchausen=False,
        layer_size=config.layer_size,
        n_step=config.n_step,
        BATCH_SIZE=batch_size,
        BUFFER_SIZE=buffer_size,
        LR=lr,
        TAU=config.tau,
        GAMMA=config.gamma,
        N=config.n,
        worker=n_envs,
        device=device,
        seed=seed,
    )

    # optionally load weights / optimizer / episode number from a checkpoint
    loaded_episode = 0
    if load_path:
        try:
            if os.path.exists(load_path):
                ckpt = torch.load(load_path, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                try:
                    agent.qnetwork_local.load_state_dict(state_dict)
                    try:
                        agent.qnetwork_target.load_state_dict(state_dict)
                    except Exception:
                        pass
                    print(f"Loaded model weights from {load_path}")
                except Exception as e:
                    print(f"Failed to load model weights: {e}")
                if (
                    "optimizer_state_dict" in ckpt
                    and hasattr(agent, "optimizer")
                    and getattr(agent, "optimizer") is not None
                ):
                    try:
                        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        print("Loaded optimizer state")
                    except Exception:
                        pass
                loaded_episode = int(ckpt.get("episode", 0))
            else:
                print(f"Checkpoint not found: {load_path}")
        except Exception as e:
            print(f"Error loading checkpoint {load_path}: {e}")

    frame_stacks, states, start_times = init_frame_stacks(envs, frame_stack)

    scores: List[float] = []
    scores_per_env = [0.0 for _ in range(n_envs)]
    completed_episodes = loaded_episode

    import time

    while completed_episodes < episodes:
        for i in range(n_envs):
            if completed_episodes >= episodes:
                break

            if render:
                try:
                    envs[i].render()
                except Exception:
                    pass

            state = states[i]
            action = agent.act(state)
            if isinstance(action, (list, tuple, np.ndarray)):
                a = int(action[0])
            else:
                a = int(action)

            step_ret = envs[i].step(a)
            if len(step_ret) == 5:
                next_state, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_ret

            elapsed = time.time() - start_times[i]
            if elapsed >= max_episode_time:
                done = True

            next_chw = to_chw(next_state)
            frame_stacks[i].append(next_chw)
            stacked_next = np.concatenate(list(frame_stacks[i]), axis=0)

            agent.step(state, a, reward, stacked_next, done, writer)
            states[i] = stacked_next
            scores_per_env[i] += reward

            if done:
                score = scores_per_env[i]
                scores.append(score)
                completed_episodes += 1

                recent = scores[-10:]
                avg10 = sum(recent) / len(recent)
                best10 = max(recent)

                writer.log_dict(
                    {
                        "pref/score": score,
                        "pref/avg10": avg10,
                        "pref/best10": best10,
                        "time/episode": completed_episodes,
                    },
                    step=completed_episodes,
                )
                try:
                    writer.add_scalar("exploration/eps", step=completed_episodes)
                except Exception:
                    pass

                if completed_episodes % log_interval == 1:
                    print()
                    print(
                        "{:<8} {:>8} {:>10} {:>8}".format(
                            "Episode", "Score", "Avg(10)", "Best10"
                        )
                    )
                    print("-" * 48)

                print(
                    "{:<8d} {:>8.2f} {:>10.2f} {:>8.2f}".format(
                        completed_episodes, score, avg10, best10
                    )
                )

                s = reset_env(envs[i])
                s_chw = to_chw(s)
                frame_stacks[i].clear()
                for _ in range(frame_stack):
                    frame_stacks[i].append(s_chw)
                states[i] = np.concatenate(list(frame_stacks[i]), axis=0)
                start_times[i] = time.time()
                scores_per_env[i] = 0.0

                # save checkpoint every 500 episodes
                if completed_episodes % 500 == 0:
                    ckpt_path = os.path.join(
                        os.getcwd(), f"iqn_agent_ep{completed_episodes}.pth"
                    )
                    try:
                        ckpt = {
                            "episode": completed_episodes,
                            "model_state_dict": agent.qnetwork_local.state_dict(),
                        }
                        opt = getattr(agent, "optimizer", None)
                        if opt is not None:
                            ckpt["optimizer_state_dict"] = opt.state_dict()
                        torch.save(ckpt, ckpt_path)
                        print(f"Checkpoint saved to {ckpt_path}")
                    except Exception as e:
                        print(
                            f"Failed to save checkpoint at episode {completed_episodes}: {e}"
                        )

    for e in envs:
        e.close()

    try:
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass

    model_save_path = os.path.join(os.getcwd(), "iqn_agent_final.pth")
    torch.save(agent.qnetwork_local.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_gym()
