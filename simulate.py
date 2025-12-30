import argparse
import time
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import create_gym_env
from sac import SACPolicy, SACConfig, get_device

from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict


class ExplorationSchedule:
    """Controls the probability of taking random actions vs policy actions.

    Supports multiple schedule types:
    - linear: linearly decay from 1.0 to min_epsilon
    - exponential: exponentially decay with given decay rate
    - cosine: cosine annealing schedule
    """

    def __init__(
        self,
        schedule_type="linear",
        total_steps=200_000,
        start_epsilon=1.0,
        min_epsilon=0.0,
        decay_rate=0.995,
    ):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.start_epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.current_step = 0

    def get_epsilon(self, step=None):
        """Get exploration probability for current or given step."""
        if step is None:
            step = self.current_step

        if self.schedule_type == "linear":
            progress = min(step / self.total_steps, 1.0)
            epsilon = self.start_epsilon - progress * (
                self.start_epsilon - self.min_epsilon
            )

        elif self.schedule_type == "exponential":
            epsilon = self.start_epsilon * (self.decay_rate**step)
            epsilon = max(epsilon, self.min_epsilon)

        elif self.schedule_type == "cosine":
            progress = min(step / self.total_steps, 1.0)
            epsilon = self.min_epsilon + 0.5 * (
                self.start_epsilon - self.min_epsilon
            ) * (1 + math.cos(math.pi * progress))

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return epsilon

    def step(self):
        """Increment the step counter."""
        self.current_step += 1

    def should_explore(self, step=None):
        """Randomly decide whether to explore based on current epsilon."""
        epsilon = self.get_epsilon(step)
        return torch.rand(1).item() < epsilon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--log-interval", type=int, default=1_000)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--load-replay",
        type=str,
        default="replay_buffer_init.pt",
        help="path to a saved LazyTensorStorage state dict (torch.save output) to load",
    )
    # Exploration schedule arguments
    p.add_argument(
        "--exploration-schedule",
        type=str,
        default="linear",
        choices=["linear", "exponential", "cosine"],
        help="Type of exploration schedule",
    )
    p.add_argument(
        "--start-epsilon",
        type=float,
        default=1.0,
        help="Initial exploration probability",
    )
    p.add_argument(
        "--min-epsilon", type=float, default=0.0, help="Minimum exploration probability"
    )
    p.add_argument(
        "--decay-rate",
        type=float,
        default=0.9999,
        help="Decay rate for exponential schedule",
    )
    p.add_argument(
        "--exploration-steps",
        type=int,
        default=None,
        help="Number of steps over which to decay exploration (defaults to total_steps)",
    )
    return p.parse_args()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device() if args.device is None else torch.device(args.device)
    print("Using device:", device)

    cfg = SACConfig()

    # Initialize exploration schedule
    exploration_steps = args.exploration_steps or args.total_steps
    exploration = ExplorationSchedule(
        schedule_type=args.exploration_schedule,
        total_steps=exploration_steps,
        start_epsilon=args.start_epsilon,
        min_epsilon=args.min_epsilon,
        decay_rate=args.decay_rate,
    )
    print(f"Using {args.exploration_schedule} exploration schedule:")
    print(f"  Start epsilon: {args.start_epsilon}")
    print(f"  Min epsilon: {args.min_epsilon}")
    print(f"  Exploration steps: {exploration_steps}")

    env = create_gym_env(device=device, num_envs=cfg.num_envs)
    td = env.reset()

    pixels = td["pixels"]

    # ===== Agent =====
    agent = SACPolicy(env=env, num_cells=cfg.num_cells, device=device)
    modules = agent.modules()

    actor = modules["actor"]
    value = modules["value"]
    q1 = modules["q1"]
    q2 = modules["q2"]

    print("Initializing lazy modules...")
    with torch.no_grad():
        sample_pixels = td["pixels"][:1].to(device)
        sample_action = torch.zeros(1, env.action_spec.shape[-1], device=device)

        actor_input = TensorDict({"pixels": sample_pixels}, batch_size=1)
        actor(actor_input)

        value(sample_pixels)

        q1(sample_pixels, sample_action)
        q2(sample_pixels, sample_action)

    print("Lazy modules initialized")

    # ===== Optimizers =====
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr
    )
    value_opt = torch.optim.Adam(value.parameters(), lr=cfg.lr)

    print("using ReplayBuffer with LazyTensorStorage")
    storage = LazyTensorStorage(max_size=cfg.replay_size, device="cpu")
    if args.load_replay is not None:
        path = args.load_replay
        try:
            state = torch.load(path, map_location=device)
            storage.load_state_dict(state)
            print(f"Loaded replay storage from {path}")
        except Exception as e:
            print(f"Warning: could not load replay storage from {path}: {e}")
    rb = ReplayBuffer(storage=storage, batch_size=cfg.batch_size)

    def _reduce_value_to_batch(x, batch_size):
        try:
            if isinstance(x, dict) or hasattr(x, "get"):
                v = x["value"] if "value" in x else x.get("value")
            else:
                v = x
            if not isinstance(v, torch.Tensor):
                return None
            if v.shape[0] == batch_size:
                if v.ndim == 1:
                    return v.view(-1, 1)
                return v.flatten(1).mean(dim=1, keepdim=True)
            return v.view(batch_size, -1).mean(dim=1, keepdim=True)
        except Exception:
            return None

    def _pack_pixels(x):
        if not isinstance(x, torch.Tensor):
            return x
        return (x.clamp(-1.0, 1.0) * 127.0).round().to(torch.int8).cpu()

    def _unpack_pixels(x):
        if not isinstance(x, torch.Tensor):
            return x
        return x.to(torch.float32) / 127.0

    def sample_random_actions(num_envs, device=None):
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        steer = torch.empty(num_envs, 1, device=device).uniform_(-1, 1)
        gas = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
        brake = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
        return torch.cat([steer, gas, brake], dim=-1)

    def sample_random_action(n=1, dev=None):
        return sample_random_actions(n, device=dev or device)

    def _get_inner(td):
        return td["next"] if "next" in td.keys() else td

    def _extract_reward_and_done(td, num_envs):
        td = _get_inner(td)
        if "reward" in td.keys():
            rewards = td["reward"].view(num_envs).to(device)
        elif "rewards" in td.keys():
            rewards = td["rewards"].view(num_envs).to(device)
        else:
            raise KeyError(f"Unexpected TensorDict structure. Keys: {td.keys()}")
        dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if "done" in td.keys():
            dones |= td["done"].view(num_envs).to(device).to(torch.bool)
        if "terminated" in td.keys():
            dones |= td["terminated"].view(num_envs).to(device).to(torch.bool)
        if "truncated" in td.keys():
            dones |= td["truncated"].view(num_envs).to(device).to(torch.bool)
        return rewards, dones

    def _expand_actions_for_envs(actions, target_batch):
        if (
            isinstance(target_batch, (tuple, list, torch.Size))
            and len(target_batch) > 1
        ):
            if actions.shape[0] != target_batch[0]:
                raise ValueError(
                    f"Action batch size ({actions.shape[0]}) does not match env batch ({target_batch[0]})"
                )
            extra = target_batch[1:]
            new_shape = (actions.shape[0],) + (1,) * len(extra) + (actions.shape[1],)
            expand_shape = (actions.shape[0],) + tuple(extra) + (actions.shape[1],)
            return actions.view(new_shape).expand(expand_shape)

        return actions

    def _add_transition(i, pixels, next_pixels, action, reward, done):
        packed_pixels = _pack_pixels(pixels[i])
        packed_next = _pack_pixels(next_pixels[i])
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
        rb.add(transition)

    total_steps = 0
    episode_returns = []
    current_episode_return = torch.zeros(cfg.num_envs, device=device)
    current_td = td

    # Track exploration statistics
    random_action_count = 0
    policy_action_count = 0

    print("Collecting initial random data...")
    while len(rb) < cfg.start_steps:
        actions = sample_random_action(cfg.num_envs)
        target_batch = current_td.batch_size
        actions_step = _expand_actions_for_envs(actions, target_batch)
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

        next_td = env.step(action_td)
        td_next = _get_inner(next_td)

        rewards, dones = _extract_reward_and_done(td_next, cfg.num_envs)
        next_pixels = td_next["pixels"]

        pixels = current_td["pixels"]
        for i in range(cfg.num_envs):
            _add_transition(i, pixels, next_pixels, actions, rewards, dones)

        current_episode_return += rewards
        for i, d in enumerate(dones):
            if d.item():
                episode_returns.append(current_episode_return[i].item())
                current_episode_return[i] = 0.0

        current_td = next_td
        if "next" in next_td.keys() and "pixels" in next_td["next"].keys():
            current_td = next_td["next"]

    print(f"Initialized replay buffer with {len(rb)} transitions")
    torch.save(storage.state_dict(), "replay_buffer_init.pt")
    print("Replay buffer saved to replay_buffer_init.pt")

    start_time = time.time()

    while total_steps < args.total_steps:
        for _ in range(cfg.frames_per_batch):
            target_batch = current_td.batch_size

            with torch.no_grad():
                inner_obs = _get_inner(current_td)
                pixels_only = inner_obs["pixels"]

                # Use exploration schedule to decide between random and policy actions
                if exploration.should_explore(total_steps):
                    actions = sample_random_action(cfg.num_envs, dev=device)
                    random_action_count += cfg.num_envs
                else:
                    actor_input = TensorDict(
                        {"pixels": pixels_only},
                        batch_size=[pixels_only.shape[0]],
                    )
                    actor_output = actor(actor_input)

                    if (
                        "action" in actor_output.keys()
                        and actor_output["action"].shape[-1] == 3
                    ):
                        actions = actor_output["action"]
                        policy_action_count += cfg.num_envs
                    else:
                        actions = sample_random_action(cfg.num_envs, dev=device)
                        random_action_count += cfg.num_envs

            actions_step = _expand_actions_for_envs(actions, target_batch)
            action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

            next_td = env.step(action_td)

            td_next = _get_inner(next_td)

            rewards, dones = _extract_reward_and_done(td_next, cfg.num_envs)

            pixels = current_td["pixels"]
            next_pixels = td_next["pixels"]

            for i in range(cfg.num_envs):
                _add_transition(i, pixels, next_pixels, actions, rewards, dones)

            current_episode_return += rewards
            for i, d in enumerate(dones):
                if d.item():
                    episode_returns.append(current_episode_return[i].item())
                    current_episode_return[i] = 0.0

            current_td = next_td
            if "next" in next_td.keys() and "pixels" in next_td["next"].keys():
                current_td = next_td["next"]

            total_steps += cfg.num_envs
            exploration.step()

        updates_per_batch = max(1, cfg.frames_per_batch // cfg.batch_size)
        for _ in range(updates_per_batch):
            if len(rb) < cfg.batch_size:
                continue

            batch = rb.sample(cfg.batch_size)

            pixels_b = batch["pixels"]
            if isinstance(pixels_b, torch.Tensor) and pixels_b.dtype == torch.int8:
                pixels_b = _unpack_pixels(pixels_b).to(device)
            else:
                pixels_b = pixels_b.to(device)

            actions_b = batch["action"].to(device).to(torch.float32)
            rewards_b = batch["reward"].to(device).view(-1, 1)

            next_pixels_b = batch["next_pixels"]
            if (
                isinstance(next_pixels_b, torch.Tensor)
                and next_pixels_b.dtype == torch.int8
            ):
                next_pixels_b = _unpack_pixels(next_pixels_b).to(device)
            else:
                next_pixels_b = next_pixels_b.to(device)

            dones_b = batch["done"].to(device).view(-1, 1).to(dtype=rewards_b.dtype)

            with torch.no_grad():
                next_v_raw = value(next_pixels_b)
                next_v = _reduce_value_to_batch(next_v_raw, next_pixels_b.shape[0])
                if next_v is None:
                    next_v = torch.zeros_like(rewards_b)

                q_target = rewards_b + cfg.gamma * (1.0 - dones_b) * next_v

            def _fix_action_shape(a, batch_size, action_dim=None):
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

            actions_b = _fix_action_shape(
                actions_b, pixels_b.shape[0], action_dim=env.action_spec.shape[-1]
            )
            q1_pred = q1(pixels_b, actions_b).view(-1, 1)
            q2_pred = q2(pixels_b, actions_b).view(-1, 1)

            # ===== critic loss =====
            q1_loss = F.mse_loss(q1_pred, q_target)
            q2_loss = F.mse_loss(q2_pred, q_target)
            critic_loss = q1_loss + q2_loss

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(q1.parameters()) + list(q2.parameters()), cfg.max_grad_norm
            )
            critic_opt.step()

            # ===== value loss =====
            td_in = TensorDict({"pixels": pixels_b}, batch_size=pixels_b.shape[0])
            out = actor(td_in)
            sampled_action = out["action"]
            sampled_action = _fix_action_shape(
                sampled_action,
                pixels_b.shape[0],
                action_dim=env.action_spec.shape[-1],
            )

            log_prob = out.get("log_prob", None)

            q1_for_v = q1(pixels_b, sampled_action).view(-1, 1)
            q2_for_v = q2(pixels_b, sampled_action).view(-1, 1)
            min_q = torch.min(q1_for_v, q2_for_v)

            value_pred_raw = value(pixels_b)
            value_pred = _reduce_value_to_batch(value_pred_raw, pixels_b.shape[0])
            if value_pred is None:
                value_pred = torch.zeros_like(min_q)

            if log_prob is not None and log_prob.ndim == 1:
                log_prob = log_prob.view(-1, 1)

            value_target = min_q - cfg.alpha * (
                log_prob if log_prob is not None else 0.0
            )
            value_loss = F.mse_loss(value_pred, value_target.detach())

            value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), cfg.max_grad_norm)
            value_opt.step()

            # ===== actor loss =====
            out = actor(td_in)
            new_actions = _fix_action_shape(
                out["action"],
                pixels_b.shape[0],
                action_dim=env.action_spec.shape[-1],
            )
            log_prob_new = out.get("log_prob")
            if log_prob_new is None:
                log_prob_new = torch.zeros((new_actions.shape[0], 1), device=device)

            q1_new = q1(pixels_b, new_actions).view(-1, 1)
            q2_new = q2(pixels_b, new_actions).view(-1, 1)
            min_q_new = torch.min(q1_new, q2_new)

            actor_loss = (cfg.alpha * log_prob_new - min_q_new).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

        if total_steps % args.log_interval < cfg.num_envs:
            elapsed = time.time() - start_time
            avg_return = sum(episode_returns[-100:]) / max(
                1, len(episode_returns[-100:])
            )
            current_epsilon = exploration.get_epsilon(total_steps)
            total_actions = random_action_count + policy_action_count
            random_pct = 100.0 * random_action_count / max(1, total_actions)

            print(
                f"Steps: {total_steps}, AvgReturn(100): {avg_return:.2f}, "
                f"Epsilon: {current_epsilon:.3f}, Random%: {random_pct:.1f}%, "
                f"Buffer: {len(rb)}, Time: {elapsed:.1f}s"
            )

        if total_steps % args.save_interval < cfg.num_envs:
            torch.save(
                {
                    "actor_state": actor.state_dict(),
                    "q1_state": q1.state_dict(),
                    "q2_state": q2.state_dict(),
                    "value_state": value.state_dict(),
                    "actor_opt": actor_opt.state_dict(),
                    "critic_opt": critic_opt.state_dict(),
                    "value_opt": value_opt.state_dict(),
                    "steps": total_steps,
                    "exploration_step": exploration.current_step,
                },
                f"sac_checkpoint_{total_steps}.pt",
            )
            print(f"Saved checkpoint at step {total_steps}")

    print("Training finished")
    print(
        f"Final exploration stats: {policy_action_count} policy actions, "
        f"{random_action_count} random actions"
    )


if __name__ == "__main__":
    train()
