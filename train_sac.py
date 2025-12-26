import itertools
from pathlib import Path
from getpass import getuser
from datetime import datetime
import warnings

import numpy as np
import torch
import wandb
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from sac.sac import SAC
from sac.replay_memory import ReplayMemory, PrioritizedReplayMemory
from perception.generate_AE_data import generate_action


torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


def train(
    seed: int = 69,
    batch_size: int = 256,
    num_steps: int = 5_000_000,
    updates_per_step: int = 1,
    start_steps: int = 100_000,
    replay_size: int = 5_000_000,
    eval: bool = True,
    eval_interval: int = 50,
    accelerated_exploration: bool = True,
    save_models: bool = True,
    load_models: bool = False,
    save_memory: bool = True,
    load_memory: bool = True,
    path_to_actor: str = "./models/sac_actor_carracer_klein_6_24_18.pt",
    path_to_critic: str = "./models/sac_critic_carracer_klein_6_24_18.pt",
    path_to_encoder: str = "./models/sac_encoder_carracer_klein_6_24_18.pt",
    path_to_buffer: str = "./memory/buffer_talk2_6h7jpbd_12_25_15.pkl",
    num_envs: int = 4,
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 1_000_000,
    per_eps: float = 1e-6,
):
    """
    ## The train function consist of:

    - Setting up the environment, agent and replay buffer
    - Logging hyperparameters and training results
    - Loading previously saved actor and critic models
    - Training loop
    - Evaluation (every *eval_interval* episodes)
    - Saving actor and critic models

    ## Parameters:

    - **seed** *(int)*: Seed value to generate random numbers.
    - **batch_size** *(int)*: Number of samples that will be propagated through the Q, V, and policy network.
    - **num_steps** *(int)*: Number of steps that the agent takes in the environment. Determines the training duration.
    - **updates_per_step** *(int)*: Number of network parameter updates per step in the environment.
    - **start_steps** *(int)*:  Number of steps for which a random action is sampled. After reaching *start_steps* an action
    according to the learned policy is chosen.
    - **replay_size** *(int)*: Size of the replay buffer.
    - **eval** *(bool)*:  If *True* the trained policy is evaluated every *eval_interval* episodes.
    - **eval_interval** *(int)*: Interval of episodes after which to evaluate the trained policy.
    - **accelerated_exploration** *(bool)*: If *True* an action with acceleration bias is sampled.
    - **save_memory** *(bool)*: If *True* the experience replay buffer is saved to the harddrive.
    - **save_models** *(bool)*: If *True* actor and critic models are saved to the harddrive.
    - **load_models** *(bool)*: If *True* actor and critic models are loaded from *path_to_actor* and *path_to_critic*.
    - **path_to_actor** *(str)*: Path to actor model.
    - **path_to_critic** *(str)*: Path to critic model.

    """

    def make_env_fn(rank):
        def _init():
            env_ = gym.make("CarRacing-v3")
            # env_.seed(seed + rank)

            return env_

        return _init

    # SyncVectorEnv auto-resets finished sub-environments by default
    envs = SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
    max_steps_per_env = envs.get_attr("_max_episode_steps")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # NOTE: ALWAYS CHECK PARAMETERS BEFORE TRAINING
    agent = SAC(
        envs.single_action_space,
        policy="Gaussian",
        gamma=0.99,
        lr=0.0001,  # Lower LR for stability with shared encoder
        alpha=0.8,
        automatic_temperature_tuning=True,
        batch_size=batch_size,
        hidden_size=512,
        target_update_interval=1,  # More frequent target updates
        input_dim=32,
    )

    def beta_by_frame(frame_idx: int):
        return min(
            1.0, per_beta_start + (1.0 - per_beta_start) * frame_idx / per_beta_frames
        )

    memory = (
        PrioritizedReplayMemory(replay_size, alpha=per_alpha, eps=per_eps)
        if use_per
        else ReplayMemory(replay_size)
    )

    if load_memory:
        # load memory and deactivate random exploration
        memory.load(path_to_buffer)

    if load_memory or load_models:
        start_steps = 0

    # Training Loop
    total_numsteps = 0
    updates = 0

    # Log Settings and training results
    date = datetime.now()
    log_dir = Path(
        f"runs/{date.year}_SAC_{date.month}_{date.day}_{date.hour}_{date.minute}_{getuser()}/"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="AssetoCorsaRL",
        name=f"SAC_{date.month}_{date.day}_{date.hour}_{date.minute}_{getuser()}",
        config={
            "seed": seed,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "updates_per_step": updates_per_step,
            "start_steps": start_steps,
            "replay_size": replay_size,
            "policy": agent.policy_type,
            "gamma": agent.gamma,
            "tau": agent.tau,
            "alpha": agent.alpha,
            "lr": agent.lr,
            "hidden_size": agent.hidden_size,
            "input_dim": agent.input_dim,
            "target_update_interval": agent.target_update_interval,
            "num_envs": num_envs,
            "use_per": use_per,
        },
    )

    settings_msg = (
        f"Training SAC for {num_steps} steps"
        "\n\nTRAINING SETTINGS:\n"
        f"Seed={seed}, Batch size: {batch_size}, Updates per step: {updates_per_step}\n"
        f"Accelerated exploration: {accelerated_exploration}, Start steps: {start_steps}, Replay size: {replay_size}"
        "\n\nALGORITHM SETTINGS:\n"
        f"Policy: {agent.policy_type}, Automatic temperature tuning: {agent.automatic_temperature_tuning}\n"
        f"Gamma: {agent.gamma}, Tau: {agent.tau}, Alpha: {agent.alpha}, LR: {agent.lr}\n"
        f"Target update interval: {agent.target_update_interval}, Latent dim: {agent.input_dim}, Hidden size: {agent.hidden_size}"
    )
    with open(log_dir / "settings.txt", "w") as file:
        file.write(settings_msg)

    if load_models:
        try:
            agent.load_model(path_to_actor, path_to_critic, path_to_encoder)
        except FileNotFoundError:
            warnings.warn(
                "Couldn't locate models in the specified paths. Training from scratch.",
                RuntimeWarning,
            )

    # Vectorized training loop
    # Initialize vector env states
    states, _ = envs.reset(seed=[seed + i for i in range(num_envs)])
    # Place cars at default starting positions for each env

    # Process initial states (use raw observations; encoder is inside SAC)
    processed_states = states.copy()

    episode_rewards = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs, dtype=int)
    episode_counts = np.zeros(num_envs, dtype=int)

    while total_numsteps < num_steps:
        # Select actions for each env
        actions = []
        for i in range(num_envs):
            if total_numsteps < start_steps and not load_models:
                if accelerated_exploration:
                    # sample random action then apply acceleration bias
                    a = envs.single_action_space.sample()
                    a = generate_action(a)
                else:
                    a = envs.single_action_space.sample()
            else:
                a = agent.select_action(processed_states[i])
            actions.append(a)
        actions = np.stack(actions)

        # Update networks if we have enough samples
        if len(memory) > batch_size:
            beta = beta_by_frame(total_numsteps) if use_per else None
            for _ in range(updates_per_step):
                if use_per:
                    (
                        batch_state,
                        batch_action,
                        batch_reward,
                        batch_next_state,
                        batch_done,
                        batch_weight,
                        batch_idx,
                    ) = memory.sample(batch_size, beta)
                else:
                    (
                        batch_state,
                        batch_action,
                        batch_reward,
                        batch_next_state,
                        batch_done,
                    ) = memory.sample(batch_size)
                    batch_weight, batch_idx = None, None

                (
                    critic_1_loss,
                    critic_2_loss,
                    policy_loss,
                    ent_loss,
                    alpha,
                    mean_log_pi,
                    mean_min_qf_pi,
                    mean_qf1,
                    mean_qf2,
                    td_error_mean,
                    critic_grad_norm,
                    policy_grad_norm,
                ) = agent.update_parameters(
                    memory,
                    batch_size,
                    updates,
                    batch=(
                        batch_state,
                        batch_action,
                        batch_reward,
                        batch_next_state,
                        batch_done,
                    ),
                    weights=batch_weight,
                    idxs=batch_idx,
                )

                wandb.log(
                    {
                        "loss/critic_1": critic_1_loss,
                        "loss/critic_2": critic_2_loss,
                        "loss/policy": policy_loss,
                        "loss/entropy_loss": ent_loss,
                        "entropy_temperature/alpha": alpha,
                        "policy/log_pi_mean": mean_log_pi,
                        "policy/min_qf_pi_mean": mean_min_qf_pi,
                        "policy/entropy_term": alpha * mean_log_pi,  # Add this
                        "policy/q_term": mean_min_qf_pi,  # Add this
                        "qf/mean_qf1": mean_qf1,
                        "qf/mean_qf2": mean_qf2,
                        "td_error/mean": td_error_mean,
                        "grad_norm/critic": critic_grad_norm,
                        "grad_norm/policy": policy_grad_norm,
                    },
                    step=total_numsteps,
                )
                updates += 1

        # Step all envs (with auto_reset enabled, finished envs are auto-reset)
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        done_mask = np.asarray(terminated) | np.asarray(truncated)

        # Handle individual env done/resets and process observations
        for i in range(num_envs):
            r = rewards[i]
            d = bool(done_mask[i])

            # When auto_reset is enabled, next_states contains the reset obs for done envs.
            # The actual terminal observation is stored in infos["final_observation"].
            if (
                d
                and "final_observation" in infos
                and infos["final_observation"][i] is not None
            ):
                ns = infos["final_observation"][i]  # Use actual terminal state
            else:
                ns = next_states[i]

            # Determine mask for episode termination (time horizon)
            max_steps = max_steps_per_env[i] if max_steps_per_env is not None else None
            ep_step = episode_steps[i] + 1
            done = float(done_mask[i])
            mask = 1.0 - done  # Simple: 0 if terminal, 1 otherwise

            # Scale reward to stabilize Q-values (only for replay buffer)
            # Note: episode_rewards tracks unscaled rewards for fair comparison with eval
            reward_scale = 0.1
            scaled_r = float(r) * reward_scale

            # push transition for this env (raw obs; encoder handled in SAC)
            memory.push(processed_states[i], actions[i], scaled_r, ns, mask)

            # update trackers (unscaled rewards for logging consistency with eval)
            episode_steps[i] = ep_step if not d else 0
            episode_rewards[i] += r  # Unscaled reward for logging
            total_numsteps += 1

            if d:
                episode_counts[i] += 1
                wandb.log(
                    {
                        f"reward/train_env_{i}": episode_rewards[i],
                        f"reward/train_env_{i}_scaled": episode_rewards[i]
                        * reward_scale,
                        "reward/train": float(np.mean(episode_rewards)),
                    },
                    step=total_numsteps,
                )

                print(
                    f"Env {i} Episode: {episode_counts[i]}, total numsteps: {total_numsteps}, episode steps: {ep_step}, reward: {round(episode_rewards[i],2)}"
                )

                # reset episode reward for this env
                episode_rewards[i] = 0.0

            # store new processed state (use next_states which has reset obs for done envs)
            processed_states[i] = next_states[i]

        # Evaluation and saving (outside per-env loop, based on total episodes)
        total_episodes = int(episode_counts.sum())
        if total_episodes > 0 and total_episodes % eval_interval == 0 and eval:
            # Check if we already evaluated at this episode count
            if (
                not hasattr(train, "_last_eval_episode")
                or train._last_eval_episode != total_episodes
            ):
                train._last_eval_episode = total_episodes

                avg_reward = 0.0
                episodes = 10
                video_frames = []

                for ep_idx in range(episodes):
                    # Record video only for the first episode
                    if ep_idx == 0:
                        eval_env = gym.make("CarRacing-v3", render_mode="rgb_array")
                    else:
                        eval_env = gym.make("CarRacing-v3")

                    s, _ = eval_env.reset()
                    done_eval = False
                    ep_r = 0

                    while not done_eval:
                        # Capture frame for first episode only
                        if ep_idx == 0:
                            frame = eval_env.render()
                            video_frames.append(frame)

                        a_eval = agent.select_action(s, eval=True)
                        s_next, r_eval, term_eval, trunc_eval, _ = eval_env.step(a_eval)
                        done_eval = term_eval or trunc_eval
                        s = s_next
                        ep_r += r_eval
                    avg_reward += ep_r
                    eval_env.close()
                avg_reward /= episodes

                # Log video to wandb (T, H, W, C) -> need to transpose for wandb
                if video_frames:
                    video_array = np.array(video_frames)  # (T, H, W, C)
                    video_array = video_array.transpose(0, 3, 1, 2)  # (T, C, H, W)
                    wandb.log(
                        {
                            "eval/video": wandb.Video(
                                video_array, fps=30, format="mp4"
                            ),
                            "avg_reward/test": avg_reward,
                        },
                        step=total_numsteps,
                    )
                else:
                    wandb.log({"avg_reward/test": avg_reward}, step=total_numsteps)

                print(
                    f"Evaluation over {episodes} episodes: avg_reward={avg_reward:.2f}"
                )

                if save_memory:
                    memory.save(
                        f"buffer_{getuser()}_{date.month}_{date.day}_{date.hour}"
                    )
                if save_models:
                    agent.save_model(
                        "carracer",
                        f"{getuser()}_{date.month}_{date.day}_{date.hour}",
                    )

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    train(
        batch_size=256,
        load_memory=False,
        eval_interval=50,
        load_models=False,
        save_memory=False,
        save_models=True,
        num_envs=4,
        use_per=False,  # Disable PER for stability
    )
