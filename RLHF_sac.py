"""
RLHF_sac.py - Record human demonstrations and pretrain SAC model

This script provides functionality for:
1. Recording human driving episodes using keyboard controls
2. Saving demonstrations to disk
3. Pretraining the SAC actor (policy) network using behavioral cloning
4. Fine-tuning with standard SAC training

Controls:
    - Arrow Up / W: Accelerate
    - Arrow Down / S: Brake
    - Arrow Left / A: Steer Left
    - Arrow Right / D: Steer Right
    - Space: Reset episode
    - Q: Quit recording
    - PS4 controller: L2 = Brake, R2 = Throttle
"""

import pickle
from pathlib import Path
from datetime import datetime
from getpass import getuser
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    FrameStackObservation,
    TransformObservation,
)

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Recording will use keyboard polling.")

from sac.sac import SAC
from sac.replay_memory import ReplayMemory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DemonstrationDataset(Dataset):
    """Dataset class for loading demonstration data."""

    def __init__(self, demonstrations: list):
        """
        Args:
            demonstrations: List of (state, action) tuples
        """
        self.states = []
        self.actions = []

        for demo in demonstrations:
            for state, action, _, _, _ in demo:
                self.states.append(state)
                self.actions.append(action)

        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)

        print(
            f"Loaded {len(self.states)} state-action pairs from {len(demonstrations)} episodes"
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class DemonstrationRecorder:
    """Records human demonstrations for the CarRacing environment."""

    def __init__(self, frame_stack_size: int = 3):
        self.frame_stack_size = frame_stack_size
        self.demonstrations = []
        self.current_episode = []

        # Initialize pygame for keyboard input
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.display.set_mode((400, 100))
            pygame.display.set_caption("CarRacing Controller - Press Q to quit")
            # Initialize joystick if present
            try:
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    self.joystick_available = True
                    print(f"Joystick detected: {self.joystick.get_name()}")
                else:
                    self.joystick = None
                    self.joystick_available = False
            except Exception:
                self.joystick = None
                self.joystick_available = False

    def _make_env(self, render_mode: str = "human"):
        """Create the CarRacing environment with preprocessing."""
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        # Convert to grayscale
        env = GrayscaleObservation(env, keep_dim=False)
        # Normalize pixel values
        obs_space = env.observation_space
        normalized_obs_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_space.shape,
            dtype=np.float32,
        )
        env = TransformObservation(
            env,
            lambda obs: (obs / 255.0).astype(np.float32),
            normalized_obs_space,
        )
        # Stack frames
        env = FrameStackObservation(env, stack_size=self.frame_stack_size)
        return env

    def _get_keyboard_action(self) -> tuple:
        """Get action from keyboard input using pygame."""
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [steering, gas, brake]
        quit_requested = False
        reset_requested = False
        if PYGAME_AVAILABLE:
            pygame.event.pump()

            # If a joystick is attached prefer controller input
            if getattr(self, "joystick_available", False) and self.joystick is not None:
                try:
                    # Steering: left stick X -> axis 0
                    num_axes = self.joystick.get_numaxes()
                    axes = [self.joystick.get_axis(i) for i in range(num_axes)]

                    # Steering (axis 0 if available)
                    if len(axes) > 0:
                        steer = float(axes[0]) / 5.0
                        # clamp to [-1,1]
                        steer = max(-1.0, min(1.0, steer))
                        action[0] = steer

                    # PS4 mapping: L2 -> brake (axis 4), R2 -> throttle (axis 5)
                    # Triggers report -1 (unpressed) to +1 (fully pressed)
                    l2_axis = 4
                    r2_axis = 5

                    def axis_to_trigger(v):
                        """Map trigger axis from [-1, 1] to [0, 1]."""
                        if v is None:
                            return 0.0
                        # Always map [-1, 1] -> [0, 1]
                        return max(0.0, min(1.0, (v + 1.0) / 2.0))

                    gas_val = 0.0
                    brake_val = 0.0

                    # Read R2 for throttle, L2 for brake
                    if r2_axis < len(axes):
                        gas_val = axis_to_trigger(axes[r2_axis])
                    if l2_axis < len(axes):
                        brake_val = axis_to_trigger(axes[l2_axis])

                    # Debug: print raw values occasionally (every ~30 frames)
                    if not hasattr(self, "_debug_counter"):
                        self._debug_counter = 0
                    self._debug_counter += 1
                    if self._debug_counter % 30 == 0:
                        raw_l2 = axes[l2_axis] if l2_axis < len(axes) else None
                        raw_r2 = axes[r2_axis] if r2_axis < len(axes) else None
                        print(
                            f"  [DEBUG] L2(raw={raw_l2:.2f}, brake={brake_val:.2f}) R2(raw={raw_r2:.2f}, gas={gas_val:.2f})"
                            if raw_l2 is not None
                            else f"  [DEBUG] axes={axes}"
                        )

                    # Clamp and assign
                    action[1] = float(max(0.0, min(1.0, gas_val)))
                    action[2] = float(max(0.0, min(1.0, brake_val)))

                    # Buttons for quit/reset if needed
                    if self.joystick.get_numbuttons() > 0:
                        # Common PS4: button 1 (Circle) or 2 (Square) can be used to quit/reset
                        # We map button 3 (Triangle) as reset and button 1 (Circle) as quit as a fallback
                        try:
                            if self.joystick.get_button(3):
                                reset_requested = True
                            if self.joystick.get_button(1):
                                quit_requested = True
                        except Exception:
                            pass
                except Exception:
                    # If joystick read fails, fall back to keyboard
                    pass
            else:
                keys = pygame.key.get_pressed()

                # Check for quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True

                if keys[pygame.K_q]:
                    quit_requested = True
                if keys[pygame.K_SPACE]:
                    reset_requested = True

                # Steering
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    action[0] = -1.0
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    action[0] = 1.0

                # Gas
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    action[1] = 1.0

                # Brake
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    action[2] = 0.8
        else:
            # Fallback: simple keyboard polling (less responsive)
            import msvcrt

            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b"q":
                    quit_requested = True
                elif key == b" ":
                    reset_requested = True
                elif key == b"w":
                    action[1] = 1.0
                elif key == b"s":
                    action[2] = 0.8
                elif key == b"a":
                    action[0] = -1.0
                elif key == b"d":
                    action[0] = 1.0

        return action, quit_requested, reset_requested

    def record_episodes(self, num_episodes: int = 10, min_reward: float = -100):
        """
        Record human driving episodes.

        Args:
            num_episodes: Target number of episodes to record
            min_reward: Minimum episode reward to keep the demonstration
        """
        print("\n" + "=" * 60)
        print("DEMONSTRATION RECORDING")
        print("=" * 60)
        print("\nControls:")
        print("  Arrow Keys / WASD: Drive the car")
        print("  Space: Reset current episode")
        print("  Q: Quit recording")
        print("  PS4 controller: L2 = Brake, R2 = Throttle")
        print("\nStarting in 3 seconds...")

        import time

        time.sleep(3)

        env = self._make_env(render_mode="human")
        episode_count = 0

        while episode_count < num_episodes:
            state, _ = env.reset()
            self.current_episode = []
            episode_reward = 0
            done = False
            step_count = 0

            print(f"\n--- Episode {episode_count + 1}/{num_episodes} ---")

            while not done:
                action, quit_requested, reset_requested = self._get_keyboard_action()

                if quit_requested:
                    print("\nQuitting recording...")
                    env.close()
                    if PYGAME_AVAILABLE:
                        pygame.quit()
                    return self.demonstrations

                if reset_requested:
                    print("Episode reset by user.")
                    break

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition (use mask=1 for not done, 0 for done)
                mask = 0.0 if terminated else 1.0
                self.current_episode.append((state, action, reward, next_state, mask))

                state = next_state
                episode_reward += reward
                step_count += 1

                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(
                        f"  Steps: {step_count}, Current Reward: {episode_reward:.2f}"
                    )

            # Check if episode should be saved
            if len(self.current_episode) > 0:
                if episode_reward >= min_reward:
                    self.demonstrations.append(self.current_episode)
                    episode_count += 1
                    print(
                        f"Episode saved! Reward: {episode_reward:.2f}, Steps: {step_count}"
                    )
                else:
                    print(
                        f"Episode discarded (reward {episode_reward:.2f} < {min_reward})"
                    )

        env.close()
        if PYGAME_AVAILABLE:
            pygame.quit()

        print(f"\n{'='*60}")
        print(f"Recording complete! Saved {len(self.demonstrations)} episodes")
        print(f"{'='*60}\n")

        return self.demonstrations

    def save_demonstrations(self, path: str = None):
        """Save recorded demonstrations to disk."""
        if path is None:
            date = datetime.now()
            path = f"demonstrations_{getuser()}_{date.month}_{date.day}_{date.hour}_{date.minute}.pkl"

        save_dir = Path("memory/")
        save_dir.mkdir(exist_ok=True)

        full_path = save_dir / path

        # Calculate statistics
        total_transitions = sum(len(ep) for ep in self.demonstrations)
        total_reward = sum(sum(t[2] for t in ep) for ep in self.demonstrations)

        payload = {
            "demonstrations": self.demonstrations,
            "num_episodes": len(self.demonstrations),
            "total_transitions": total_transitions,
            "avg_reward": (
                total_reward / len(self.demonstrations) if self.demonstrations else 0
            ),
            "timestamp": datetime.now().isoformat(),
        }

        with open(full_path, "wb") as f:
            pickle.dump(payload, f)

        print(
            f"Saved {len(self.demonstrations)} demonstrations ({total_transitions} transitions) to {full_path}"
        )
        return str(full_path)

    @staticmethod
    def load_demonstrations(path: str) -> list:
        """Load demonstrations from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        demos = payload.get("demonstrations", payload)  # Handle both old and new format

        if isinstance(payload, dict):
            print(
                f"Loaded {payload.get('num_episodes', len(demos))} episodes, "
                f"{payload.get('total_transitions', 'N/A')} transitions, "
                f"avg_reward: {payload.get('avg_reward', 'N/A'):.2f}"
            )

        return demos


class BehavioralCloning:
    """Pretrain SAC policy using behavioral cloning from demonstrations."""

    def __init__(
        self,
        agent: SAC,
        demonstrations: list,
        batch_size: int = 256,
        lr: float = 1e-4,
    ):
        self.agent = agent
        self.batch_size = batch_size
        self.demonstrations = demonstrations  # Store for critic pretraining

        # Create dataset and dataloader
        self.dataset = DemonstrationDataset(demonstrations)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # Use separate optimizer for BC training
        # Train both encoder and policy
        self.bc_optimizer = Adam(
            list(agent.encoder.parameters()) + list(agent.policy.parameters()),
            lr=lr,
        )

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        total_loss = 0
        total_mse = 0
        num_batches = 0

        for states, expert_actions in self.dataloader:
            states = states.to(self.agent.device)
            expert_actions = expert_actions.to(self.agent.device)

            # Encode states
            latent = self.agent._encode(states)

            # Get MEAN predicted actions from policy (deterministic for BC)
            _, _, pred_actions = self.agent.policy.sample(
                latent
            )  # Use mean (3rd output)

            # BC loss: MSE between predicted and expert actions
            loss = F.mse_loss(pred_actions, expert_actions)

            self.bc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.agent.encoder.parameters())
                + list(self.agent.policy.parameters()),
                max_norm=5.0,
            )
            self.bc_optimizer.step()

            total_loss += loss.item()
            total_mse += loss.item()
            num_batches += 1

        return {
            "bc_loss": total_loss / num_batches,
            "mse": total_mse / num_batches,
        }

    def pretrain_critic(self, num_epochs: int = 50, gamma: float = 0.99):
        """
        Pretrain critic using TD-learning from demonstrations with actor's policy.
        This helps the critic provide better Q-value estimates during RL fine-tuning.
        """
        print("\n" + "=" * 60)
        print("CRITIC PRETRAINING")
        print("=" * 60)
        print(f"Training critic for {num_epochs} epochs using actor's policy\n")

        # Build a dataset with (state, action, reward, next_state, mask) from demonstrations
        critic_data = []
        for demo in self.dataset.states, self.dataset.actions:
            pass  # We need the full transitions, not just state-action pairs

        # Rebuild transitions from original demonstrations
        transitions = []
        for demo in self.demonstrations:
            for state, action, reward, next_state, mask in demo:
                transitions.append((state, action, reward, next_state, mask))

        # Convert to tensors
        states = np.array([t[0] for t in transitions], dtype=np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.float32)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        masks = np.array([t[4] for t in transitions], dtype=np.float32)

        num_samples = len(transitions)

        for epoch in range(num_epochs):
            total_q_loss = 0
            num_batches = 0

            # Shuffle indices each epoch
            indices = np.random.permutation(num_samples)

            for batch_start in range(0, num_samples, self.batch_size):
                batch_indices = indices[batch_start : batch_start + self.batch_size]

                batch_states = torch.FloatTensor(states[batch_indices]).to(
                    self.agent.device
                )
                batch_actions = torch.FloatTensor(actions[batch_indices]).to(
                    self.agent.device
                )
                batch_rewards = (
                    torch.FloatTensor(rewards[batch_indices])
                    .to(self.agent.device)
                    .unsqueeze(1)
                )
                batch_next_states = torch.FloatTensor(next_states[batch_indices]).to(
                    self.agent.device
                )
                batch_masks = (
                    torch.FloatTensor(masks[batch_indices])
                    .to(self.agent.device)
                    .unsqueeze(1)
                )

                with torch.no_grad():
                    # Encode states
                    latent = self.agent._encode(batch_states)
                    next_latent = self.agent._encode(batch_next_states)

                    # Get actor's action for next state
                    next_action, next_log_pi, _ = self.agent.policy.sample(next_latent)

                    # Compute target Q-value using actor's policy
                    q1_next, q2_next = self.agent.critic_target(
                        next_latent, next_action
                    )
                    min_q_next = torch.min(q1_next, q2_next)

                    # Include entropy term for SAC-style targets
                    target_q = batch_rewards + batch_masks * gamma * (
                        min_q_next - self.agent.alpha * next_log_pi
                    )

                # Get current Q-values for the demonstration actions
                q1, q2 = self.agent.critic(latent.detach(), batch_actions)

                # MSE loss for both Q-networks
                q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                # Update critic only (not encoder)
                self.agent.critic_optim.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(), max_norm=5.0
                )
                self.agent.critic_optim.step()

                # Soft update target network
                from sac.utils import soft_update

                soft_update(self.agent.critic_target, self.agent.critic, self.agent.tau)

                total_q_loss += q_loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Critic Epoch {epoch+1}/{num_epochs} - Q Loss: {total_q_loss/num_batches:.6f}"
                )

        # Hard update target critic after pretraining
        from sac.utils import hard_update

        hard_update(self.agent.critic_target, self.agent.critic)
        print("Critic pretraining complete, target network synchronized\n")

    def pretrain(
        self,
        num_epochs: int = 100,
        eval_interval: int = 10,
        save_best: bool = True,
        pretrain_critic_epochs: int = 50,
    ) -> dict:
        """
        Pretrain the policy using behavioral cloning.

        Args:
            num_epochs: Number of training epochs
            eval_interval: Evaluate policy every N epochs
            save_best: Save best model based on evaluation reward
            pretrain_critic_epochs: Number of epochs to pretrain critic (0 to skip)

        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("BEHAVIORAL CLONING PRETRAINING")
        print("=" * 60)
        print(f"Dataset size: {len(self.dataset)} transitions")
        print(f"Batch size: {self.batch_size}")
        print(f"Training for {num_epochs} epochs\n")

        # Pretrain critic first if requested
        if pretrain_critic_epochs > 0:
            self.pretrain_critic(num_epochs=pretrain_critic_epochs)

        history = {
            "bc_loss": [],
            "eval_reward": [],
        }

        best_reward = float("-inf")

        for epoch in range(num_epochs):
            metrics = self.train_epoch()
            history["bc_loss"].append(metrics["bc_loss"])

            print(f"Epoch {epoch+1}/{num_epochs} - BC Loss: {metrics['bc_loss']:.6f}")

            if (epoch + 1) % eval_interval == 0:
                eval_reward = self._evaluate()
                history["eval_reward"].append(eval_reward)
                print(f"  Evaluation Reward: {eval_reward:.2f}")

                if save_best and eval_reward > best_reward:
                    best_reward = eval_reward
                    self.agent.save_model("carracer", "bc_best")
                    print(f"  New best model saved! (reward: {best_reward:.2f})")

        # Save final model
        self.agent.save_model("carracer", "bc_final")

        print(f"\n{'='*60}")
        print(f"Pretraining complete!")
        print(f"Final BC Loss: {history['bc_loss'][-1]:.6f}")
        if history["eval_reward"]:
            print(f"Best Evaluation Reward: {best_reward:.2f}")
        print(f"{'='*60}\n")

        return history

    def _evaluate(self, num_episodes: int = 10) -> float:
        """Evaluate the current policy."""
        env = gym.make("CarRacing-v3")
        env = GrayscaleObservation(env, keep_dim=False)
        obs_space = env.observation_space
        normalized_obs_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_space.shape, dtype=np.float32
        )
        env = TransformObservation(
            env,
            lambda obs: (obs / 255.0).astype(np.float32),
            normalized_obs_space,
        )
        env = FrameStackObservation(env, stack_size=3)

        total_reward = 0

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.select_action(state, eval=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_reward += episode_reward

        env.close()
        return total_reward / num_episodes


def fill_replay_buffer_from_demonstrations(
    memory: ReplayMemory,
    demonstrations: list,
    reward_scale: float = 1.0,
):
    """Fill a replay buffer with demonstration data."""
    count = 0
    for episode in demonstrations:
        for state, action, reward, next_state, mask in episode:
            memory.push(state, action, reward * reward_scale, next_state, mask)
            count += 1

    print(f"Added {count} transitions from demonstrations to replay buffer")
    return memory


def record_demonstrations(
    num_episodes: int = 10,
    min_reward: float = -100,
    save_path: str = None,
):
    """Convenience function to record and save demonstrations."""
    recorder = DemonstrationRecorder(frame_stack_size=3)
    demonstrations = recorder.record_episodes(num_episodes, min_reward)

    if demonstrations:
        path = recorder.save_demonstrations(save_path)
        return path
    else:
        print("No demonstrations recorded.")
        return None


def pretrain_from_demonstrations(
    path_to_demonstrations: str,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    eval_interval: int = 10,
    load_existing_model: bool = False,
    path_to_actor: str = None,
    path_to_critic: str = None,
    path_to_encoder: str = None,
    pretrain_critic_epochs: int = 50,
):
    """
    Pretrain SAC agent from recorded demonstrations.

    Args:
        path_to_demonstrations: Path to saved demonstrations file
        num_epochs: Number of BC training epochs
        batch_size: Training batch size
        lr: Learning rate for BC training
        eval_interval: Evaluate every N epochs
        load_existing_model: Whether to load an existing model before pretraining
        path_to_actor: Path to existing actor model
        path_to_critic: Path to existing critic model
        path_to_encoder: Path to existing encoder model
        pretrain_critic_epochs: Number of epochs to pretrain critic (0 to skip)
    """
    print("\n" + "=" * 60)
    print("RLHF SAC PRETRAINING")
    print("=" * 60)

    # Load demonstrations
    demonstrations = DemonstrationRecorder.load_demonstrations(path_to_demonstrations)

    # Create dummy env to get action space
    env = gym.make("CarRacing-v3")
    action_space = env.action_space
    env.close()

    # Create SAC agent
    agent = SAC(
        action_space,
        policy="Gaussian",
        gamma=0.99,
        lr=lr,
        alpha=0.2,
        automatic_temperature_tuning=True,
        batch_size=batch_size,
        hidden_size=512,
        target_update_interval=1,
        input_dim=32,
        in_channels=3,
    )

    # Optionally load existing model
    if load_existing_model and path_to_actor:
        agent.load_model(path_to_actor, path_to_critic, path_to_encoder)

    # Create BC trainer and pretrain
    bc_trainer = BehavioralCloning(
        agent=agent,
        demonstrations=demonstrations,
        batch_size=batch_size,
        lr=lr,
    )

    history = bc_trainer.pretrain(
        num_epochs=num_epochs,
        eval_interval=eval_interval,
        save_best=True,
        pretrain_critic_epochs=pretrain_critic_epochs,
    )

    return agent, history


def main():
    parser = argparse.ArgumentParser(
        description="RLHF SAC - Record and pretrain from demonstrations"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Record subcommand
    record_parser = subparsers.add_parser("record", help="Record human demonstrations")
    record_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to record"
    )
    record_parser.add_argument(
        "--min-reward", type=float, default=-100, help="Minimum reward to keep episode"
    )
    record_parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save demonstrations"
    )

    # Pretrain subcommand
    pretrain_parser = subparsers.add_parser(
        "pretrain", help="Pretrain from demonstrations"
    )
    pretrain_parser.add_argument(
        "--demos", type=str, required=True, help="Path to demonstrations file"
    )
    pretrain_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    pretrain_parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    pretrain_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    pretrain_parser.add_argument(
        "--eval-interval", type=int, default=10, help="Evaluation interval"
    )
    pretrain_parser.add_argument(
        "--critic-epochs",
        type=int,
        default=50,
        help="Critic pretraining epochs (0 to skip)",
    )
    pretrain_parser.add_argument(
        "--load-model", action="store_true", help="Load existing model"
    )
    pretrain_parser.add_argument(
        "--actor", type=str, default=None, help="Path to actor model"
    )
    pretrain_parser.add_argument(
        "--critic", type=str, default=None, help="Path to critic model"
    )
    pretrain_parser.add_argument(
        "--encoder", type=str, default=None, help="Path to encoder model"
    )

    # Both subcommand (record then pretrain)
    both_parser = subparsers.add_parser(
        "both", help="Record demonstrations then pretrain"
    )
    both_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to record"
    )
    both_parser.add_argument(
        "--min-reward", type=float, default=-100, help="Minimum reward to keep episode"
    )
    both_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    both_parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    both_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    if args.command == "record":
        record_demonstrations(
            num_episodes=args.episodes,
            min_reward=args.min_reward,
            save_path=args.save_path,
        )

    elif args.command == "pretrain":
        pretrain_from_demonstrations(
            path_to_demonstrations=args.demos,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_interval=args.eval_interval,
            pretrain_critic_epochs=args.critic_epochs,
            load_existing_model=args.load_model,
            path_to_actor=args.actor,
            path_to_critic=args.critic,
            path_to_encoder=args.encoder,
        )

    elif args.command == "both":
        # Record demonstrations
        demo_path = record_demonstrations(
            num_episodes=args.episodes,
            min_reward=args.min_reward,
        )

        if demo_path:
            # Pretrain from recorded demonstrations
            pretrain_from_demonstrations(
                path_to_demonstrations=demo_path,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
