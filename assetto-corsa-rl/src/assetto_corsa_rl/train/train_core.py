"""Trainer class encapsulating the rollout and update loops.

This file provides a compact OOP wrapper while keeping the original
module-level functions `collect_initial_data` and `run_training_loop`
for backward compatibility.
"""

import math
import time
import torch
import torch.nn.functional as F
from tensordict import TensorDict
import wandb

from .train_utils import (
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
        q1,
        q2,
        q1_target,
        q2_target,
        actor_opt,
        critic_opt,
        log_alpha,
        alpha_opt,
        target_entropy,
        device,
        storage=None,
    ):
        self.env = env
        self.rb = rb
        self.cfg = cfg
        self.current_td = td
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt

        self.device = device
        self.storage = storage

        self.total_steps = 0
        self.episode_returns = []
        self.current_episode_return = torch.zeros(cfg.num_envs, device=device)
        self.start_time = time.time()

        self.log_alpha = log_alpha
        self.alpha_opt = alpha_opt
        self.target_entropy = target_entropy if target_entropy is not None else -3.0

        self._last_log_steps = 0
        self._last_save_steps = 0
        self._last_save_rb_steps = 0

        self._updates_count = 0
        self._updates_per_step = int(getattr(cfg, "updates_per_step", 1))
        self._actor_update_delay = getattr(cfg, "actor_update_delay", 1)
        self._freeze_encoder_steps = getattr(cfg, "freeze_encoder_steps", 0)
        self._encoder_frozen = False

    def _get_alpha_value(self):
        """Return current alpha value (prefer model parameter if present)."""
        return float(self.log_alpha.exp().item())

    def _set_encoder_requires_grad(self, requires_grad: bool):
        """Enable or disable gradients for visual encoder layers in all networks."""
        for net in [self.actor, self.q1, self.q2]:
            if net is None:
                continue
            for name, module in net.named_modules():
                if "cnn" in name.lower() or isinstance(module, torch.nn.Conv2d):
                    for param in module.parameters():
                        param.requires_grad = requires_grad

    def _step_random(self):
        actions = sample_random_action(self.cfg.num_envs)
        target_batch = self.current_td.batch_size
        actions_step = expand_actions_for_envs(actions, target_batch)
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

        next_td = self.env.step(action_td)
        td_next = get_inner(next_td)

        rewards, dones = extract_reward_and_done(td_next, self.cfg.num_envs, self.device)
        pixels = self.current_td["pixels"]
        next_pixels = td_next["pixels"]

        for i in range(self.cfg.num_envs):
            add_transition(self.rb, i, pixels, next_pixels, actions, rewards, dones)

        self._handle_episode_end(rewards, dones)
        self._maybe_reset(td_next, dones)

    def _exploration_epsilon(self):
        """Linearly annealed exploration epsilon between start and end over explore_steps."""
        if self.cfg.use_noisy:
            return 0.0

        start = float(getattr(self.cfg, "explore_start", 1.0))
        end = float(getattr(self.cfg, "explore_end", 0.0))
        steps = int(getattr(self.cfg, "explore_steps", 100_000))
        if steps <= 0:
            return float(end)
        frac = min(1.0, float(self.total_steps) / float(steps))
        return float(start + (end - start) * frac)

    def run(self, total_steps=0):
        self.total_steps = total_steps

        while self.total_steps < self.cfg.total_steps:
            for _ in range(self.cfg.frames_per_batch):
                self._step_and_store()

                if len(self.rb) >= self.cfg.batch_size:
                    for _ in range(self._updates_per_step):
                        self._do_update()

            self._maybe_log_and_save()
        print("Training finished")

    def _step_and_store(self):
        target_batch = self.current_td.batch_size
        with torch.no_grad():
            inner_obs = get_inner(self.current_td)
            pixels_only = inner_obs["pixels"]
            if pixels_only.dim() == 3:
                pixels_only = pixels_only.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            actor_input = TensorDict({"pixels": pixels_only}, batch_size=[pixels_only.shape[0]])
            use_noisy = getattr(self.cfg, "use_noisy", False)

            if use_noisy:
                for m in self.actor.modules():
                    if hasattr(m, "sample_noise"):
                        m.sample_noise()

            actor_output = self.actor(actor_input)
            has_actor_action = (
                "action" in actor_output.keys()
                and actor_output["action"].shape[-1] == self.env.action_spec.shape[-1]
            )
            actor_action = actor_output["action"] if has_actor_action else None

            if use_noisy:
                eps = 0.0
                if actor_action is None:
                    for m in self.actor.modules():
                        if hasattr(m, "sample_noise"):
                            m.sample_noise()
                    actor_output = self.actor(actor_input)
                    has_actor_action = (
                        "action" in actor_output.keys()
                        and actor_output["action"].shape[-1] == self.env.action_spec.shape[-1]
                    )
                    actor_action = actor_output["action"] if has_actor_action else None
            else:
                eps = self._exploration_epsilon()

            if eps > 0.0:
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

        rewards, dones = extract_reward_and_done(td_next, self.cfg.num_envs, self.device)

        pixels = self.current_td["pixels"]
        next_pixels = td_next["pixels"]

        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        if next_pixels.ndim == 3:
            next_pixels = next_pixels.unsqueeze(0)

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
        rewards = rewards.to(self.current_episode_return.device)
        dones = dones.to(self.current_episode_return.device)
        self.current_episode_return += rewards
        for i, d in enumerate(dones):
            if d.item():
                self.episode_returns.append(self.current_episode_return[i].item())
                self.current_episode_return[i] = 0.0
                episode_length = len(self.episode_returns)
                wandb.log({"episode_length": episode_length}, step=self.total_steps)

    def _maybe_reset(self, td_next, dones):
        self.current_td = td_next
        if "next" in td_next.keys() and "pixels" in td_next["next"].keys():
            self.current_td = td_next["next"]
        if dones.any():
            try:
                reset_td = self.env.reset()
                self.current_td = (
                    reset_td["next"]
                    if ("next" in reset_td.keys() and "pixels" in reset_td["next"].keys())
                    else reset_td
                )
            except Exception:
                self.current_td = self.env.reset()
            try:
                idx = dones.to(self.current_episode_return.device)
                self.current_episode_return[idx] = 0.0
            except Exception:
                self.current_episode_return = torch.zeros_like(self.current_episode_return)

    # ===== Updates =====
    def _do_update(self):
        if self._freeze_encoder_steps > 0:
            if self.total_steps < self._freeze_encoder_steps and not self._encoder_frozen:
                self._set_encoder_requires_grad(False)
                self._encoder_frozen = True
                print(
                    f"[INFO] Visual encoder frozen (step {self.total_steps} < {self._freeze_encoder_steps})"
                )
            elif self.total_steps >= self._freeze_encoder_steps and self._encoder_frozen:
                self._set_encoder_requires_grad(True)
                self._encoder_frozen = False
                print(
                    f"[INFO] Visual encoder unfrozen (step {self.total_steps} >= {self._freeze_encoder_steps})"
                )

        # sample with return_info=True to get indices for priority updates
        batch, info = self.rb.sample(self.cfg.batch_size, return_info=True)

        batch_indices = info.get("index", None)

        pixels_b = batch["pixels"]

        if isinstance(pixels_b, torch.Tensor) and not torch.is_floating_point(pixels_b):
            pixels_b = unpack_pixels(pixels_b).to(self.device)
        else:
            pixels_b = pixels_b.to(self.device)

        actions_b = batch["action"].to(self.device).to(torch.float32)
        rewards_b = batch["reward"].to(self.device).view(-1, 1)

        next_pixels_b = batch["next_pixels"]
        if isinstance(next_pixels_b, torch.Tensor) and not torch.is_floating_point(next_pixels_b):
            next_pixels_b = unpack_pixels(next_pixels_b).to(self.device)
        else:
            next_pixels_b = next_pixels_b.to(self.device)

        dones_b = batch["done"].to(self.device).view(-1, 1).to(dtype=rewards_b.dtype)

        # ===== Critic Update =====
        with torch.no_grad():
            next_td = TensorDict({"pixels": next_pixels_b}, batch_size=next_pixels_b.shape[0])
            next_out = self.actor(next_td)
            next_actions = fix_action_shape(
                next_out["action"],
                next_pixels_b.shape[0],
                action_dim=self.env.action_spec.shape[-1],
            )
            next_log_prob = next_out["action_log_prob"]
            next_log_prob = next_log_prob.view(-1, 1)

            next_q1 = self.q1_target(next_pixels_b, next_actions).view(-1, 1)
            next_q2 = self.q2_target(next_pixels_b, next_actions).view(-1, 1)
            next_min_q = torch.min(next_q1, next_q2)

            alpha = self.log_alpha.exp() if self.log_alpha is not None else self.cfg.alpha

            # V(s') = min_i Q_i(s', a') - α * log π(a'|s')
            next_v = next_min_q - alpha * (next_log_prob if next_log_prob is not None else 0.0)

            q_target = rewards_b + self.cfg.gamma * (1.0 - dones_b) * next_v

            min_q = float(getattr(self.cfg, "min_q_target", -1000.0))
            max_q = float(getattr(self.cfg, "max_q_target", 1000.0))
            q_target = torch.clamp(q_target, min=min_q, max=max_q)

        # current Q-values
        actions_b = fix_action_shape(
            actions_b, pixels_b.shape[0], action_dim=self.env.action_spec.shape[-1]
        )
        q1_pred = self.q1(pixels_b, actions_b).view(-1, 1)
        q2_pred = self.q2(pixels_b, actions_b).view(-1, 1)

        # compute TD errors for PER priority updates
        with torch.no_grad():
            td_error_1 = torch.abs(q1_pred - q_target)
            td_error_2 = torch.abs(q2_pred - q_target)
            td_errors = torch.max(td_error_1, td_error_2).squeeze(-1)

            if batch_indices is not None:
                priorities = td_errors.cpu().numpy()
                self.rb.update_priority(batch_indices, priorities)

        beta = min(
            1.0,
            self.cfg.per_beta
            + (1.0 - self.cfg.per_beta) * (self.total_steps / self.cfg.total_steps),
        )
        self.rb.beta = beta

        # critic loss: MSE
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

        # explained variance for Q-functions
        with torch.no_grad():
            q_var = torch.var(q_target)
            q1_explained_var = 1 - torch.var(q_target - q1_pred) / (q_var + 1e-8)
            q2_explained_var = 1 - torch.var(q_target - q2_pred) / (q_var + 1e-8)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.max_grad_norm,
        )
        self.critic_opt.step()

        # ===== Actor Update =====
        self._updates_count += 1
        if self._updates_count % self._actor_update_delay == 0:
            # sample actions from current policy
            td_in = TensorDict({"pixels": pixels_b}, batch_size=pixels_b.shape[0])
            out = self.actor(td_in)
            new_actions = fix_action_shape(
                out["action"],
                pixels_b.shape[0],
                action_dim=self.env.action_spec.shape[-1],
            )

            log_prob_new = out["action_log_prob"]
            if log_prob_new is None:
                print(f"WARNING: No log_prob returned! Actor output keys: {list(out.keys())}")
                log_prob_new = torch.zeros((new_actions.shape[0], 1), device=self.device)

            if log_prob_new.ndim == 1:
                log_prob_new = log_prob_new.view(-1, 1)

            wandb.log(
                {
                    "actor/loc_mean": out["loc"].mean().item(),
                    "actor/loc_std": out["loc"].std().item(),
                    "actor/loc_max": out["loc"].max().item(),
                    "actor/scale_mean": out["scale"].mean().item(),
                    "actor/scale_min": out["scale"].min().item(),
                    "actor/scale_max": out["scale"].max().item(),
                },
                step=self.total_steps,
            )

            # compute Q-values for new actions (clipped double-Q)
            q1_new = self.q1(pixels_b, new_actions).view(-1, 1)
            q2_new = self.q2(pixels_b, new_actions).view(-1, 1)
            min_q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q(s,a) - α * log π(a|s)
            actor_loss = (alpha.detach() * log_prob_new - min_q_new).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()

            # ===== Alpha (Temperature) Update =====
            alpha_loss = None
            if self.log_alpha is not None and self.alpha_opt is not None:
                with torch.no_grad():
                    entropy_error = log_prob_new + self.target_entropy
                alpha_loss = -(self.log_alpha * entropy_error).mean()

                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
                self.alpha_opt.step()
                alpha_min = float(getattr(self.cfg, "alpha_min", 0.01))
                with torch.no_grad():
                    self.log_alpha.clamp_(min=math.log(alpha_min))
        else:
            actor_loss = None
            alpha_loss = None
            log_prob_new = None
            min_q_new = None
            new_actions = None

        # ===== Soft Update Target Networks =====
        self._soft_update_target()

        # ===== Logging =====
        try:
            current_entropy = -log_prob_new.mean().item() if log_prob_new is not None else 0.0
            log_dict = {
                "loss/critic_loss": critic_loss.item(),
                "loss/q1_loss": q1_loss.item(),
                "loss/q2_loss": q2_loss.item(),
                "loss/actor_loss": (actor_loss.item() if actor_loss is not None else 0.0),
                #
                "critic/mean_q_value": (min_q_new.mean().item() if min_q_new is not None else 0.0),
                "critic/q_target_mean": q_target.mean().item(),
                "critic/next_v_mean": next_v.mean().item(),
                "critic/q1_explained_variance": q1_explained_var.item(),
                "critic/q2_explained_variance": q2_explained_var.item(),
                #
                "actor/mean_log_prob": (
                    log_prob_new.mean().item() if log_prob_new is not None else 0.0
                ),
                "actor/entropy": current_entropy,
                "actor/target_entropy": self.target_entropy,
                "actor/entropy_gap": current_entropy - self.target_entropy,
                "actor/alpha": alpha.item(),
                "actor/alpha_loss": (alpha_loss.item() if alpha_loss is not None else 0.0),
                #
                "reward/rewards_per_step_mean": rewards_b.mean().item(),
                "reward/rewards_per_step_max": rewards_b.max().item(),
                "reward/rewards_per_step_min": rewards_b.min().item(),
                #
                "actions/actions_std": actions_b.std().item(),
                "actions/sampled_action_mean": (
                    new_actions.mean().item() if new_actions is not None else 0.0
                ),
                "actions/sampled_action_std": (
                    new_actions.std().item() if new_actions is not None else 0.0
                ),
                #
                "per/td_error_mean": td_errors.mean().item(),
                "per/td_error_max": td_errors.max().item(),
                "per/td_error_min": td_errors.min().item(),
                "per/beta": beta,
            }

            wandb.log(log_dict, step=self.total_steps)
        except Exception as e:
            print("Warning: wandb logging failed (updates):", e)

    def _soft_update_target(self):
        """Soft update of target network: θ_target = τ*θ + (1-τ)*θ_target"""
        tau = self.cfg.tau

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _maybe_log_and_save(self):
        if self.total_steps - getattr(self, "_last_log_steps", 0) >= self.cfg.log_interval:
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

            try:
                wandb.log(
                    {
                        "steps": self.total_steps,
                        "reward/rewards_per_environment_mean": avg_return,
                        "buffer": len(self.rb),
                        "time": elapsed,
                        "epsilon": eps,
                    },
                    step=self.total_steps,
                )
            except Exception as e:
                print("Warning: wandb logging failed (_maybe_log_and_save):", e)

            self._last_log_steps = self.total_steps

        if self.total_steps - getattr(self, "_last_save_steps", 0) >= self.cfg.save_interval:
            torch.save(
                {
                    "actor_state": self.actor.state_dict(),
                    "q1_state": self.q1.state_dict(),
                    "q2_state": self.q2.state_dict(),
                    "actor_opt": self.actor_opt.state_dict(),
                    "critic_opt": self.critic_opt.state_dict(),
                    "steps": self.total_steps,
                    "config": {
                        "use_noisy": getattr(self.cfg, "use_noisy", False),
                        "num_cells": getattr(self.cfg, "num_cells", 256),
                        "vae_checkpoint_path": getattr(self.cfg, "vae_checkpoint_path", None),
                    },
                },
                f".\\models\\sac_checkpoint_{self.total_steps}.pt",
            )
            print(f"Saved checkpoint at step {self.total_steps}")
            self._last_save_steps = self.total_steps

        rb_save_interval = getattr(
            self.cfg, "save_interval_replaybuffer", self.cfg.save_interval * 2
        )
        if self.total_steps - getattr(self, "_last_save_rb_steps", 0) >= rb_save_interval:
            import pickle
            from pathlib import Path
            import shutil

            rb_dir = Path("./models")
            rb_dir.mkdir(parents=True, exist_ok=True)
            rb_path = rb_dir / f"replay_buffer_{self.total_steps}.pkl"

            try:
                min_free_space_gb = getattr(self.cfg, "min_free_space_gb", 10)
                min_free_space_bytes = min_free_space_gb * 1024 * 1024 * 1024

                stat = shutil.disk_usage(rb_dir)
                available_space = stat.free

                existing_buffers = sorted(rb_dir.glob("replay_buffer_*.pkl"))

                while existing_buffers and available_space < min_free_space_bytes:
                    oldest_buffer = existing_buffers.pop(0)
                    buffer_size = oldest_buffer.stat().st_size
                    oldest_buffer.unlink()
                    available_space += buffer_size
                    print(
                        f"Deleted old replay buffer: {oldest_buffer.name} (freed {buffer_size / (1024**3):.2f} GB)"
                    )

                stat = shutil.disk_usage(rb_dir)
                if stat.free < min_free_space_bytes:
                    print(
                        f"Warning: Low disk space ({stat.free / (1024**3):.2f} GB free). Skipping replay buffer save."
                    )
                else:
                    rb_state = {
                        "buffer": self.rb._storage._storage,
                        "sampler_state": {
                            "alpha": (
                                self.rb._sampler._alpha
                                if hasattr(self.rb._sampler, "_alpha")
                                else None
                            ),
                            "beta": (
                                self.rb._sampler._beta
                                if hasattr(self.rb._sampler, "_beta")
                                else None
                            ),
                        },
                        "total_steps": self.total_steps,
                        "buffer_size": len(self.rb),
                    }
                    with open(rb_path, "wb") as f:
                        pickle.dump(rb_state, f)

                    saved_size = rb_path.stat().st_size
                    remaining_space = shutil.disk_usage(rb_dir).free
                    print(
                        f"Saved replay buffer at step {self.total_steps} ({len(self.rb)} transitions, "
                        f"{saved_size / (1024**3):.2f} GB, {remaining_space / (1024**3):.2f} GB free)"
                    )
                    self._last_save_rb_steps = self.total_steps
            except Exception as e:
                print(f"Warning: Failed to save replay buffer: {e}")


def collect_initial_data(env, rb, cfg, current_td, device):
    t = Trainer(
        env,
        rb,
        cfg,
        current_td,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        device,
        None,
    )
    return t.collect_initial_data()


def run_training_loop(
    env,
    rb,
    cfg,
    current_td,
    actor,
    q1,
    q2,
    q1_target,
    q2_target,
    actor_opt,
    critic_opt,
    log_alpha,
    alpha_opt,
    target_entropy,
    device,
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
        q1,
        q2,
        q1_target,
        q2_target,
        actor_opt,
        critic_opt,
        log_alpha,
        alpha_opt,
        target_entropy,
        device,
        storage,
    )
    if episode_returns is not None:
        t.episode_returns = episode_returns
    if current_episode_return is not None:
        t.current_episode_return = current_episode_return
    if start_time is not None:
        t.start_time = start_time
    t.run(total_steps=total_steps)
