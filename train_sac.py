import itertools
from pathlib import Path
from getpass import getuser
from datetime import datetime
import warnings

import numpy as np
import torch
import wandb
from gymnasium.wrappers import (
    RecordVideo,
    GrayscaleObservation,
    FrameStackObservation,
    TransformObservation,
)
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from sac.sac import SAC
from sac.replay_memory import ReplayMemory, PrioritizedReplayMemory
from perception.generate_AE_data import generate_action


torch.backends.cudnn.benchmark = True
# Enable TF32 for faster computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


def train(
    seed: int = 69,
    batch_size: int = 512,  # Larger batch for better GPU utilization
    num_steps: int = 5_000_000,
    updates_per_step: int = 2,  # More updates per env step
    start_steps: int = 30_000,
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
    num_envs: int = 8,  # More parallel envs
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 1_000_000,
    per_eps: float = 1e-6,
    initial_step: int = 0,
    log_interval: int = 100,  # Log every N updates to reduce overhead
    use_async_envs: bool = True,  # Use async vectorized envs for parallelism
    prefetch_batches: int = 3,  # Number of batches to prefetch
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
            # Convert to grayscale (96, 96)
            env_ = GrayscaleObservation(env_, keep_dim=False)
            # Normalize pixel values to [0, 1] with proper observation space
            obs_space = env_.observation_space
            normalized_obs_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=obs_space.shape,
                dtype=np.float32,
            )
            env_ = TransformObservation(
                env_,
                lambda obs: (obs / 255.0).astype(np.float32),
                normalized_obs_space,
            )
            # Stack 3 frames -> (3, 96, 96)
            env_ = FrameStackObservation(env_, stack_size=3)
            return env_

        return _init

    # Use AsyncVectorEnv for parallel environment stepping (much faster)
    if use_async_envs:
        envs = AsyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
    else:
        envs = SyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
    max_steps_per_env = envs.get_attr("_max_episode_steps")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # NOTE: ALWAYS CHECK PARAMETERS BEFORE TRAINING
    # Frame stack of 3 grayscale frames -> in_channels=3
    frame_stack_size = 3
    agent = SAC(
        envs.single_action_space,
        policy="Gaussian",
        gamma=0.99,
        lr=0.0001,
        alpha=0.2,
        automatic_temperature_tuning=True,
        batch_size=batch_size,
        hidden_size=512,
        target_update_interval=1,
        input_dim=32,
        in_channels=frame_stack_size,
    )

    def beta_by_frame(frame_idx: int):
        return min(
            1.0, per_beta_start + (1.0 - per_beta_start) * frame_idx / per_beta_frames
        )

    memory = (
        PrioritizedReplayMemory(replay_size, alpha=per_alpha, eps=per_eps)
        if use_per
        else ReplayMemory(replay_size, prefetch_batches=prefetch_batches)
    )

    if load_memory:
        # load memory and deactivate random exploration
        memory.load(path_to_buffer)

    if load_memory or load_models:
        start_steps = 0

    # Training Loop
    total_numsteps = initial_step
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
            "updates_per_step": updates_per_step,
            "log_interval": log_interval,
            "use_async_envs": use_async_envs,
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
            # ADD THESE LINES - sync target networks
            from sac.utils import hard_update

            hard_update(agent.encoder_target, agent.encoder)
            hard_update(agent.critic_target, agent.critic)
            print("Target networks synchronized with BC weights")

            if agent.automatic_temperature_tuning:
                agent.log_alpha.data.fill_(np.log(0.05))  # Start with low alpha
                agent.alpha = 0.05
                print(f"Lowered alpha to {agent.alpha} for BC fine-tuning")

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
    best_eval_reward = float("-inf")  # Track best evaluation reward for video logging

    while total_numsteps < num_steps:
        if total_numsteps < start_steps and not load_models:
            if accelerated_exploration:
                actions = np.stack(
                    [
                        generate_action(envs.single_action_space.sample())
                        for _ in range(num_envs)
                    ]
                )
            else:
                actions = np.stack(
                    [envs.single_action_space.sample() for _ in range(num_envs)]
                )
        else:
            actions = agent.select_action_batch(processed_states)

        if len(memory) > batch_size:
            beta = beta_by_frame(total_numsteps) if use_per else None

            metrics_accum = {
                "loss/critic_1": 0,
                "loss/critic_2": 0,
                "loss/policy": 0,
                "loss/entropy_loss": 0,
                "entropy_temperature/alpha": 0,
                "policy/log_pi_mean": 0,
                "policy/min_qf_pi_mean": 0,
                "qf/mean_qf1": 0,
                "qf/mean_qf2": 0,
                "td_error/mean": 0,
                "grad_norm/critic": 0,
                "grad_norm/policy": 0,
            }

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

                # Accumulate metrics
                metrics_accum["loss/critic_1"] += critic_1_loss
                metrics_accum["loss/critic_2"] += critic_2_loss
                metrics_accum["loss/policy"] += policy_loss
                metrics_accum["loss/entropy_loss"] += ent_loss
                metrics_accum["entropy_temperature/alpha"] += alpha
                metrics_accum["policy/log_pi_mean"] += mean_log_pi
                metrics_accum["policy/min_qf_pi_mean"] += mean_min_qf_pi
                metrics_accum["qf/mean_qf1"] += mean_qf1
                metrics_accum["qf/mean_qf2"] += mean_qf2
                metrics_accum["td_error/mean"] += td_error_mean
                metrics_accum["grad_norm/critic"] += critic_grad_norm
                metrics_accum["grad_norm/policy"] += policy_grad_norm

                updates += 1

            # Log averaged metrics less frequently to reduce overhead
            if updates % log_interval == 0:
                avg_metrics = {
                    k: v / updates_per_step for k, v in metrics_accum.items()
                }
                avg_metrics["policy/entropy_term"] = (
                    avg_metrics["entropy_temperature/alpha"]
                    * avg_metrics["policy/log_pi_mean"]
                )
                avg_metrics["policy/q_term"] = avg_metrics["policy/min_qf_pi_mean"]
                wandb.log(avg_metrics, step=total_numsteps)

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

            max_steps = max_steps_per_env[i] if max_steps_per_env is not None else None
            ep_step = episode_steps[i] + 1
            done = float(done_mask[i])
            mask = 0.0 if terminated[i] else 1.0

            reward_scale = 1
            scaled_r = float(r) * reward_scale

            memory.push(processed_states[i], actions[i], scaled_r, ns, mask)

            episode_steps[i] = ep_step if not d else 0
            episode_rewards[i] += r
            total_numsteps += 1

            if d:
                episode_counts[i] += 1
                wandb.log(
                    {
                        f"reward/train_env_{i}": episode_rewards[i],
                        # f"reward/train_env_{i}_scaled": episode_rewards[i]
                        # * reward_scale,
                        "reward/train": float(np.mean(episode_rewards)),
                    },
                    step=total_numsteps,
                )

                print(
                    f"Env {i} Episode: {episode_counts[i]}, total numsteps: {total_numsteps}, episode steps: {ep_step}, reward: {round(episode_rewards[i],2)}"
                )

                episode_rewards[i] = 0.0

            processed_states[i] = next_states[i]

        total_episodes = int(episode_counts.sum())
        if total_episodes > 0 and total_episodes % eval_interval == 0 and eval:
            if (
                not hasattr(train, "_last_eval_episode")
                or train._last_eval_episode != total_episodes
            ):
                train._last_eval_episode = total_episodes

                avg_reward = 0.0
                episodes = 10

                # First pass: calculate avg_reward without video
                for ep_idx in range(episodes):
                    eval_env = gym.make("CarRacing-v3")
                    # Apply same preprocessing as training envs
                    eval_env = GrayscaleObservation(eval_env, keep_dim=False)
                    obs_space = eval_env.observation_space
                    normalized_obs_space = gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=obs_space.shape,
                        dtype=np.float32,
                    )
                    eval_env = TransformObservation(
                        eval_env,
                        lambda obs: (obs / 255.0).astype(np.float32),
                        normalized_obs_space,
                    )
                    eval_env = FrameStackObservation(eval_env, stack_size=3)

                    s, _ = eval_env.reset()
                    done_eval = False
                    ep_r = 0

                    while not done_eval:
                        a_eval = agent.select_action(s, eval=True)
                        s_next, r_eval, term_eval, trunc_eval, _ = eval_env.step(a_eval)
                        done_eval = term_eval or trunc_eval
                        s = s_next
                        ep_r += r_eval
                    avg_reward += ep_r
                    eval_env.close()
                avg_reward /= episodes

                # Record video only if new reward record
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    print(f"New reward record: {avg_reward:.2f}! Recording video...")

                    video_frames = []
                    eval_env = gym.make("CarRacing-v3", render_mode="rgb_array")
                    eval_env = GrayscaleObservation(eval_env, keep_dim=False)
                    obs_space = eval_env.observation_space
                    normalized_obs_space = gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=obs_space.shape,
                        dtype=np.float32,
                    )
                    eval_env = TransformObservation(
                        eval_env,
                        lambda obs: (obs / 255.0).astype(np.float32),
                        normalized_obs_space,
                    )
                    eval_env = FrameStackObservation(eval_env, stack_size=3)

                    s, _ = eval_env.reset()
                    done_eval = False
                    while not done_eval:
                        frame = eval_env.render()
                        video_frames.append(frame)
                        a_eval = agent.select_action(s, eval=True)
                        s_next, r_eval, term_eval, trunc_eval, _ = eval_env.step(a_eval)
                        done_eval = term_eval or trunc_eval
                        s = s_next
                    eval_env.close()

                    video_array = np.array(video_frames).transpose(0, 3, 1, 2)
                    wandb.log(
                        {
                            "eval/video": wandb.Video(
                                video_array, fps=30, format="mp4"
                            ),
                            "avg_reward/test": avg_reward,
                            "avg_reward/best": best_eval_reward,
                        },
                        step=total_numsteps,
                    )

                    # Save best model
                    if save_models:
                        agent.save_model("carracer", "best")
                else:
                    wandb.log(
                        {
                            "avg_reward/test": avg_reward,
                            "avg_reward/best": best_eval_reward,
                        },
                        step=total_numsteps,
                    )

                print(
                    f"Evaluation over {episodes} episodes: avg_reward={avg_reward:.2f}"
                )

                if save_memory:
                    memory.save(
                        f"buffer_{getuser()}_{date.month}_{date.day}_{date.hour}"
                    )
                # Always save last model checkpoint
                if save_models:
                    agent.save_model("carracer", "last")

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    train(
        batch_size=512,  # Larger batch = better GPU utilization
        load_memory=False,
        eval_interval=50,
        load_models=True,
        save_memory=False,
        save_models=True,
        num_envs=12,  # More parallel environments
        use_per=False,  # Disable PER for stability
        initial_step=10_000,
        updates_per_step=2,  # More updates per env step
        log_interval=100,  # Log less frequently
        use_async_envs=True,  # Async envs for parallelism
        prefetch_batches=3,  # Prefetch data batches
        path_to_buffer="memory\\demonstrations_talk2_6h7jpbd_12_27_14_34.pkl",
        path_to_actor="models\\sac_actor_carracer_bc_best.pt",
        path_to_critic="models\\sac_critic_carracer_bc_best.pt",
        path_to_encoder="models\\sac_encoder_carracer_bc_best.pt",
    )
