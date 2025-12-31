"""Trainer class encapsulating the rollout and update loops.

This file provides a compact OOP wrapper while keeping the original
module-level functions `collect_initial_data` and `run_training_loop`
for backward compatibility.
"""

import time
import torch
import torch.nn.functional as F
from tensordict import TensorDict

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from train_utils import (
    add_transition,
    extract_reward_and_done,
    expand_actions_for_envs,
    fix_action_shape,
    get_inner,
    unpack_pixels,
    reduce_value_to_batch,
    sample_random_action,
)


class Trainer:
    """Encapsulates environment stepping, buffering, and learning updates."""

    def __init__(
        self,
        env,
        rb,
        cfg,
        td,
        actor,
        value,
        value_target,
        q1,
        q2,
        actor_opt,
        critic_opt,
        value_opt,
        device,
        args=None,
        storage=None,
    ):
        self.env = env
        self.rb = rb
        self.cfg = cfg
        self.current_td = td
        self.actor = actor
        self.value = value
        self.value_target = value_target
        self.q1 = q1
        self.q2 = q2
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.value_opt = value_opt

        self.device = device
        self.args = args
        self.storage = storage

        self.total_steps = 0
        self.episode_returns = []
        self.current_episode_return = torch.zeros(cfg.num_envs, device=device)
        self.start_time = time.time()

    # ===== Collection =====
    def collect_initial_data(self):
        print("Collecting initial random data...")
        while len(self.rb) < self.cfg.start_steps:
            self._step_random()
        print(f"Initialized replay buffer with {len(self.rb)} transitions")
        return (
            self.current_td,
            self.total_steps,
            self.episode_returns,
            self.current_episode_return,
        )

    def _step_random(self):
        actions = sample_random_action(self.cfg.num_envs)
        target_batch = self.current_td.batch_size
        actions_step = expand_actions_for_envs(actions, target_batch)
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

        next_td = self.env.step(action_td)
        td_next = get_inner(next_td)

        rewards, dones = extract_reward_and_done(
            td_next, self.cfg.num_envs, self.device
        )
        pixels = self.current_td["pixels"]
        next_pixels = td_next["pixels"]

        for i in range(self.cfg.num_envs):
            add_transition(self.rb, i, pixels, next_pixels, actions, rewards, dones)

        self._handle_episode_end(rewards, dones)
        self._maybe_reset(td_next, dones)

    def _exploration_epsilon(self):
        """Linearly annealed exploration epsilon between start and end over explore_steps."""
        if not self.args:
            return 0.0
        start = float(getattr(self.args, "explore_start", 1.0))
        end = float(getattr(self.args, "explore_end", 0.0))
        steps = int(getattr(self.args, "explore_steps", 100_000))
        if steps <= 0:
            return float(end)
        frac = min(1.0, float(self.total_steps) / float(steps))
        return float(start + (end - start) * frac)

    # ===== Main loop =====
    def run(self, total_steps=0):
        self.total_steps = total_steps
        while self.total_steps < self.args.total_steps:
            for _ in range(self.cfg.frames_per_batch):
                self._step_and_store()
            self._perform_updates()
            self._maybe_log_and_save()
        print("Training finished")

    def _step_and_store(self):
        target_batch = self.current_td.batch_size
        with torch.no_grad():
            inner_obs = get_inner(self.current_td)
            pixels_only = inner_obs["pixels"]
            actor_input = TensorDict(
                {"pixels": pixels_only}, batch_size=[pixels_only.shape[0]]
            )
            actor_output = self.actor(actor_input)
            has_actor_action = (
                "action" in actor_output.keys()
                and actor_output["action"].shape[-1] == self.env.action_spec.shape[-1]
            )
            actor_action = actor_output["action"] if has_actor_action else None

            eps = self._exploration_epsilon()

            if eps > 0.0:
                # per-env mask deciding whether to use random action
                mask = torch.rand(self.cfg.num_envs, device=self.device) < eps
                rand_actions = sample_random_action(self.cfg.num_envs, dev=self.device)
                if actor_action is None:
                    actions = rand_actions
                else:
                    rand_actions = rand_actions.to(actor_action.device)
                    actions = torch.where(mask.view(-1, 1), rand_actions, actor_action)
            else:
                if actor_action is None:
                    actions = sample_random_action(self.cfg.num_envs, dev=self.device)
                else:
                    actions = actor_action

        actions_step = expand_actions_for_envs(actions, target_batch)
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)
        next_td = self.env.step(action_td)
        td_next = get_inner(next_td)

        rewards, dones = extract_reward_and_done(
            td_next, self.cfg.num_envs, self.device
        )

        # if dones.any():
        #     self._debug_on_done(actions, actions_step, actor_output, td_next)

        pixels = self.current_td["pixels"]
        next_pixels = td_next["pixels"]
        for i in range(self.cfg.num_envs):
            add_transition(self.rb, i, pixels, next_pixels, actions, rewards, dones)

        self._handle_episode_end(rewards, dones)
        self._maybe_reset(td_next, dones)
        self.total_steps += self.cfg.num_envs

    def _debug_on_done(self, actions, actions_step, actor_output, td_next):
        print(
            "Action (per env):",
            (actions.tolist() if isinstance(actions, torch.Tensor) else actions),
        )

        print("actions_step shape:", getattr(actions_step, "shape", None))
        a_out = actor_output.get("action")
        print(
            "actor action stats: min, max, shape ->",
            a_out.min().item(),
            a_out.max().item(),
            a_out.shape,
        )

        print("next_td inner keys:", td_next.keys())
        for key in ("done", "terminated", "truncated"):
            if key in td_next.keys():
                try:
                    print(f"{key}:", td_next[key].tolist())
                except Exception:
                    print(f"{key}:", td_next[key])

    def _handle_episode_end(self, rewards, dones):
        # Ensure tensors are on the same device as current_episode_return
        rewards = rewards.to(self.current_episode_return.device)
        dones = dones.to(self.current_episode_return.device)
        self.current_episode_return += rewards
        for i, d in enumerate(dones):
            if d.item():
                self.episode_returns.append(self.current_episode_return[i].item())
                self.current_episode_return[i] = 0.0

    def _maybe_reset(self, td_next, dones):
        self.current_td = td_next
        if "next" in td_next.keys() and "pixels" in td_next["next"].keys():
            self.current_td = td_next["next"]
        if dones.any():
            try:
                reset_td = self.env.reset()
                self.current_td = (
                    reset_td["next"]
                    if (
                        "next" in reset_td.keys()
                        and "pixels" in reset_td["next"].keys()
                    )
                    else reset_td
                )
            except Exception:
                self.current_td = self.env.reset()
            try:
                idx = dones.to(self.current_episode_return.device)
                self.current_episode_return[idx] = 0.0
            except Exception:
                self.current_episode_return = torch.zeros_like(
                    self.current_episode_return
                )

    # ===== Updates =====
    def _perform_updates(self):
        updates_per_batch = max(1, self.cfg.frames_per_batch // self.cfg.batch_size)
        for _ in range(updates_per_batch):
            if len(self.rb) < self.cfg.batch_size:
                continue
            self._do_update()

    def _do_update(self):
        batch = self.rb.sample(self.cfg.batch_size)
        pixels_b = batch["pixels"]
        if isinstance(pixels_b, torch.Tensor) and pixels_b.dtype == torch.int8:
            pixels_b = unpack_pixels(pixels_b).to(self.device)
        else:
            pixels_b = pixels_b.to(self.device)

        actions_b = batch["action"].to(self.device).to(torch.float32)
        rewards_b = batch["reward"].to(self.device).view(-1, 1)

        next_pixels_b = batch["next_pixels"]
        if (
            isinstance(next_pixels_b, torch.Tensor)
            and next_pixels_b.dtype == torch.int8
        ):
            next_pixels_b = unpack_pixels(next_pixels_b).to(self.device)
        else:
            next_pixels_b = next_pixels_b.to(self.device)

        dones_b = batch["done"].to(self.device).view(-1, 1).to(dtype=rewards_b.dtype)

        with torch.no_grad():
            # USE TARGET NETWORK instead of regular value network
            next_v_raw = self.value_target(next_pixels_b)  # ← FIXED!
            next_v = reduce_value_to_batch(next_v_raw, next_pixels_b.shape[0])
            if next_v is None:
                next_v = torch.zeros_like(rewards_b)
            q_target = rewards_b + self.cfg.gamma * (1.0 - dones_b) * next_v

        actions_b = fix_action_shape(
            actions_b, pixels_b.shape[0], action_dim=self.env.action_spec.shape[-1]
        )
        q1_pred = self.q1(pixels_b, actions_b).view(-1, 1)
        q2_pred = self.q2(pixels_b, actions_b).view(-1, 1)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.max_grad_norm,
        )
        self.critic_opt.step()

        td_in = TensorDict({"pixels": pixels_b}, batch_size=pixels_b.shape[0])
        out = self.actor(td_in)
        sampled_action = out["action"]
        sampled_action = fix_action_shape(
            sampled_action, pixels_b.shape[0], action_dim=self.env.action_spec.shape[-1]
        )

        log_prob = out.get("log_prob", None)

        q1_for_v = self.q1(pixels_b, sampled_action).view(-1, 1)
        q2_for_v = self.q2(pixels_b, sampled_action).view(-1, 1)
        min_q = torch.min(q1_for_v, q2_for_v)

        value_pred_raw = self.value(pixels_b)
        value_pred = reduce_value_to_batch(value_pred_raw, pixels_b.shape[0])
        if value_pred is None:
            value_pred = torch.zeros_like(min_q)

        if log_prob is not None and log_prob.ndim == 1:
            log_prob = log_prob.view(-1, 1)

        value_target = min_q - self.cfg.alpha * (
            log_prob if log_prob is not None else 0.0
        )
        value_loss = F.mse_loss(value_pred, value_target.detach())

        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
        self.value_opt.step()

        out = self.actor(td_in)
        new_actions = fix_action_shape(
            out["action"], pixels_b.shape[0], action_dim=self.env.action_spec.shape[-1]
        )
        log_prob_new = out.get("log_prob")
        if log_prob_new is None:
            log_prob_new = torch.zeros((new_actions.shape[0], 1), device=self.device)

        q1_new = self.q1(pixels_b, new_actions).view(-1, 1)
        q2_new = self.q2(pixels_b, new_actions).view(-1, 1)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.cfg.alpha * log_prob_new - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        self._soft_update_target()

        # Log losses to WandB if enabled
        if self.args and getattr(self.args, "wandb", False) and WANDB_AVAILABLE:
            try:
                wandb.log(
                    {
                        "critic_loss": critic_loss.item(),
                        "value_loss": value_loss.item(),
                        "actor_loss": actor_loss.item(),
                    },
                    step=self.total_steps,
                )
            except Exception:
                pass

    def _soft_update_target(self):
        """Soft update of target network: θ_target = τ*θ + (1-τ)*θ_target"""
        tau = self.cfg.tau
        for param, target_param in zip(
            self.value.parameters(), self.value_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _maybe_log_and_save(self):
        if self.total_steps % self.args.log_interval < self.cfg.num_envs:
            elapsed = time.time() - self.start_time
            last = self.episode_returns[-100:]
            if len(last) > 0:
                avg_return = sum(last) / len(last)
            else:
                avg_return = float(self.current_episode_return.mean().item())
            eps = self._exploration_epsilon()
            print(
                f"Steps: {self.total_steps}, AvgReturn(100): {avg_return:.2f}, Buffer: {len(self.rb)}, Time: {elapsed:.1f}s, Eps: {eps:.3f}"
            )

            # Log to WandB if enabled
            if self.args and getattr(self.args, "wandb", False) and WANDB_AVAILABLE:
                try:
                    wandb.log(
                        {
                            "steps": self.total_steps,
                            "avg_return": avg_return,
                            "buffer": len(self.rb),
                            "time": elapsed,
                            "epsilon": eps,
                        },
                        step=self.total_steps,
                    )
                except Exception:
                    pass

        if self.total_steps % self.args.save_interval < self.cfg.num_envs:
            torch.save(
                {
                    "actor_state": self.actor.state_dict(),
                    "q1_state": self.q1.state_dict(),
                    "q2_state": self.q2.state_dict(),
                    "value_state": self.value.state_dict(),
                    "actor_opt": self.actor_opt.state_dict(),
                    "critic_opt": self.critic_opt.state_dict(),
                    "value_opt": self.value_opt.state_dict(),
                    "steps": self.total_steps,
                },
                f"sac_checkpoint_{self.total_steps}.pt",
            )
            print(f"Saved checkpoint at step {self.total_steps}")

            # Upload checkpoint to WandB if enabled
            if self.args and getattr(self.args, "wandb", False) and WANDB_AVAILABLE:
                try:
                    wandb.save(f"sac_checkpoint_{self.total_steps}.pt")
                    wandb.log(
                        {"checkpoint": f"sac_checkpoint_{self.total_steps}.pt"},
                        step=self.total_steps,
                    )
                except Exception:
                    pass

        if self.total_steps % self.args.save_interval < self.cfg.num_envs:
            torch.save(
                {
                    "actor_state": self.actor.state_dict(),
                    "q1_state": self.q1.state_dict(),
                    "q2_state": self.q2.state_dict(),
                    "value_state": self.value.state_dict(),
                    "actor_opt": self.actor_opt.state_dict(),
                    "critic_opt": self.critic_opt.state_dict(),
                    "value_opt": self.value_opt.state_dict(),
                    "steps": self.total_steps,
                },
                f"sac_checkpoint_{self.total_steps}.pt",
            )
            print(f"Saved checkpoint at step {self.total_steps}")


# Backwards-compatible wrappers
def collect_initial_data(env, rb, cfg, current_td, device):
    t = Trainer(
        env, rb, cfg, current_td, None, None, None, None, None, None, None, device, None
    )
    return t.collect_initial_data()


def run_training_loop(
    env,
    rb,
    cfg,
    current_td,
    actor,
    value,
    value_target,
    q1,
    q2,
    actor_opt,
    critic_opt,
    value_opt,
    device,
    args,
    storage=None,
    start_time=None,
    total_steps=0,
    episode_returns=None,
    current_episode_return=None,
):
    t = Trainer(
        env,
        rb,
        cfg,
        current_td,
        actor,
        value,
        value_target,
        q1,
        q2,
        actor_opt,
        critic_opt,
        value_opt,
        device,
        args,
        storage,
    )
    # restore passed-in state when available
    if episode_returns is not None:
        t.episode_returns = episode_returns
    if current_episode_return is not None:
        t.current_episode_return = current_episode_return
    if start_time is not None:
        t.start_time = start_time
    t.run(total_steps=total_steps)
