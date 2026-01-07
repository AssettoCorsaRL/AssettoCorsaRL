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
        value,
        value_target,
        q1,
        q2,
        q1_target,
        q2_target,
        actor_opt,
        critic_opt,
        value_opt,
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
        self.value = value
        self.value_target = value_target
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.value_opt = value_opt

        self.device = device
        self.storage = storage

        self.total_steps = 0
        self.episode_returns = []
        self.current_episode_return = torch.zeros(cfg.num_envs, device=device)
        self.start_time = time.time()

        self.log_alpha = log_alpha
        self.alpha_opt = alpha_opt
        self.target_entropy = target_entropy if target_entropy is not None else -3.0

        # Track last steps when logging and saving to ensure we log/save
        # at least every `log_interval` / `save_interval` regardless of batch increments.
        self._last_log_steps = 0
        self._last_save_steps = 0

        # Track update frequency for interleaved updates
        self._steps_since_update = 0
        # How many env steps between gradient updates (lower = more frequent updates)
        self._update_every = getattr(
            cfg, "update_every", cfg.num_envs
        )  # Default: update after each batch from all envs

    def _get_alpha_value(self):
        """Return current alpha value (prefer model parameter if present)."""
        return float(self.log_alpha.exp().item())

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
        """Linearly annealed exploration epsilon between start and end over explore_steps.

        Uses `cfg` values (source of truth). If noisy-network exploration is enabled in the
        config, epsilon-greedy is disabled (returns 0.0).
        """
        if self.cfg.use_noisy:
            return 0.0

        start = float(getattr(self.cfg, "explore_start", 1.0))
        end = float(getattr(self.cfg, "explore_end", 0.0))
        steps = int(getattr(self.cfg, "explore_steps", 100_000))
        if steps <= 0:
            return float(end)
        frac = min(1.0, float(self.total_steps) / float(steps))
        return float(start + (end - start) * frac)

    # ===== Main loop =====
    def run(self, total_steps=0):
        self.total_steps = total_steps

        while self.total_steps < self.cfg.total_steps:
            # Interleaved approach: spread updates throughout data collection
            for _ in range(self.cfg.frames_per_batch):
                self._step_and_store()

                # Perform update after collecting update_every steps
                self._steps_since_update += self.cfg.num_envs
                if self._steps_since_update >= self._update_every:
                    # Do one or more gradient updates
                    num_updates = max(1, self._steps_since_update // self._update_every)
                    for _ in range(num_updates):
                        if len(self.rb) >= self.cfg.batch_size:
                            self._do_update()
                    self._steps_since_update = 0

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
            use_noisy = getattr(self.cfg, "use_noisy", False)

            # if using noisy-net exploration, resample actor noise each step
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

            # If using noisy nets, disable epsilon-greedy entirely.
            # Also try one more actor call (resample noise) if no action was returned.
            if use_noisy:
                eps = 0.0
                if actor_action is None:
                    # try once more after resampling noise to obtain a policy action
                    for m in self.actor.modules():
                        if hasattr(m, "sample_noise"):
                            m.sample_noise()
                    actor_output = self.actor(actor_input)
                    has_actor_action = (
                        "action" in actor_output.keys()
                        and actor_output["action"].shape[-1]
                        == self.env.action_spec.shape[-1]
                    )
                    actor_action = actor_output["action"] if has_actor_action else None
            else:
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
    def _do_update(self):
        # Sample with return_info=True to get indices for priority updates
        batch, info = self.rb.sample(self.cfg.batch_size, return_info=True)

        # Extract indices from info dict
        batch_indices = info.get("index", None)

        pixels_b = batch["pixels"]

        # NOTE: Do NOT sample noise during gradient updates
        # Noisy networks should only resample during environment rollout
        # Sampling here adds unnecessary variance to gradients

        if isinstance(pixels_b, torch.Tensor) and not torch.is_floating_point(pixels_b):
            pixels_b = unpack_pixels(pixels_b).to(self.device)
        else:
            pixels_b = pixels_b.to(self.device)

        actions_b = batch["action"].to(self.device).to(torch.float32)
        rewards_b = batch["reward"].to(self.device).view(-1, 1)

        next_pixels_b = batch["next_pixels"]
        if isinstance(next_pixels_b, torch.Tensor) and not torch.is_floating_point(
            next_pixels_b
        ):
            next_pixels_b = unpack_pixels(next_pixels_b).to(self.device)
        else:
            next_pixels_b = next_pixels_b.to(self.device)

        dones_b = batch["done"].to(self.device).view(-1, 1).to(dtype=rewards_b.dtype)

        # ===== Critic Update =====
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_td = TensorDict(
                {"pixels": next_pixels_b}, batch_size=next_pixels_b.shape[0]
            )
            next_out = self.actor(next_td)
            next_actions = fix_action_shape(
                next_out["action"],
                next_pixels_b.shape[0],
                action_dim=self.env.action_spec.shape[-1],
            )
            next_log_prob = next_out["action_log_prob"]
            next_log_prob = next_log_prob.view(-1, 1)

            # Compute target Q-values using target networks (clipped double-Q)
            next_q1 = self.q1_target(next_pixels_b, next_actions).view(-1, 1)
            next_q2 = self.q2_target(next_pixels_b, next_actions).view(-1, 1)
            next_min_q = torch.min(next_q1, next_q2)

            alpha = (
                self.log_alpha.exp() if self.log_alpha is not None else self.cfg.alpha
            )

            # V(s') = min_i Q_i(s', a') - α * log π(a'|s')
            next_v = next_min_q - alpha * (
                next_log_prob if next_log_prob is not None else 0.0
            )

            q_target = rewards_b + self.cfg.gamma * (1.0 - dones_b) * next_v

            # clip Q-targets to prevent divergence (critical for stability)
            # Range based on expected discounted returns [-1000, 1000]
            min_q = float(getattr(self.cfg, "min_q_target", -1000.0))
            max_q = float(getattr(self.cfg, "max_q_target", 1000.0))
            q_target = torch.clamp(q_target, min=min_q, max=max_q)

        # Compute current Q-values
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

            # update priorities in PER
            if batch_indices is not None:
                # convert to numpy if needed
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
        # store old policy for KL penalty
        with torch.no_grad():
            td_in_old = TensorDict({"pixels": pixels_b}, batch_size=pixels_b.shape[0])
            out_old = self.actor(td_in_old)
            old_loc = out_old["loc"].detach()
            old_scale = out_old["scale"].detach()

        # sample actions from current policy
        td_in = TensorDict({"pixels": pixels_b}, batch_size=pixels_b.shape[0])
        out = self.actor(td_in)
        new_actions = fix_action_shape(
            out["action"], pixels_b.shape[0], action_dim=self.env.action_spec.shape[-1]
        )

        log_prob_new = out["action_log_prob"]
        if log_prob_new is None:
            print(
                f"WARNING: No log_prob returned! Actor output keys: {list(out.keys())}"
            )
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

        # compute KL divergence between old and new policy (Gaussian KL)
        # KL(N(μ1,σ1)||N(μ2,σ2)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        new_loc = out["loc"]
        new_scale = out["scale"]
        kl_div = (
            torch.log(old_scale / new_scale)
            + (new_scale.pow(2) + (new_loc - old_loc).pow(2)) / (2 * old_scale.pow(2))
            - 0.5
        ).sum(dim=-1, keepdim=True)

        # KL penalty coefficient (can be made configurable)
        beta_kl = getattr(self.cfg, "beta_kl", 0.01)
        kl_penalty = beta_kl * kl_div

        # Log KL divergence immediately after computation
        wandb.log(
            {
                "actor/kl_divergence_instant": kl_div.mean().item(),
                "actor/kl_divergence_max": kl_div.max().item(),
                "actor/kl_divergence_min": kl_div.min().item(),
                "actor/kl_penalty_instant": kl_penalty.mean().item(),
            },
            step=self.total_steps,
        )

        # Actor loss: maximize Q(s,a) - α * log π(a|s) - β * KL(π_old||π_new)
        # Equivalently: minimize α * log π(a|s) - Q(s,a) + β * KL(π_old||π_new)
        actor_loss = (alpha.detach() * log_prob_new - min_q_new + kl_penalty).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        # ===== Alpha (Temperature) Update =====
        alpha_loss = None
        if self.log_alpha is not None and self.alpha_opt is not None:
            # Tune alpha to maintain target entropy
            # We want: E[-log π(a|s)] ≈ target_entropy
            with torch.no_grad():
                entropy_error = log_prob_new + self.target_entropy
            alpha_loss = -(self.log_alpha * entropy_error).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_opt.step()

        # ===== Soft Update Target Networks =====
        self._soft_update_target()

        # ===== Logging =====
        try:
            current_entropy = -log_prob_new.mean().item()
            log_dict = {
                "loss/critic_loss": critic_loss.item(),
                "loss/q1_loss": q1_loss.item(),
                "loss/q2_loss": q2_loss.item(),
                "loss/actor_loss": actor_loss.item(),
                "loss/kl_penalty": kl_penalty.mean().item(),
                #
                "critic/mean_q_value": min_q_new.mean().item(),
                "critic/q_target_mean": q_target.mean().item(),
                "critic/next_v_mean": next_v.mean().item(),
                "critic/q1_explained_variance": q1_explained_var.item(),
                "critic/q2_explained_variance": q2_explained_var.item(),
                #
                "actor/mean_log_prob": log_prob_new.mean().item(),
                "actor/entropy": current_entropy,
                "actor/target_entropy": self.target_entropy,
                "actor/entropy_gap": current_entropy - self.target_entropy,
                "actor/alpha": alpha.item(),
                "actor/alpha_loss": (
                    alpha_loss.item() if alpha_loss is not None else 0.0
                ),
                "actor/kl_divergence": kl_div.mean().item(),
                "actor/beta_kl": beta_kl,
                #
                "reward/rewards_per_step_mean": rewards_b.mean().item(),
                "reward/rewards_per_step_max": rewards_b.max().item(),
                "reward/rewards_per_step_min": rewards_b.min().item(),
                #
                "actions/actions_std": actions_b.std().item(),
                "actions/sampled_action_mean": new_actions.mean().item(),
                "actions/sampled_action_std": new_actions.std().item(),
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

        # update value target
        for param, target_param in zip(
            self.value.parameters(), self.value_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # update Q1 target
        for param, target_param in zip(
            self.q1.parameters(), self.q1_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # update Q2 target
        for param, target_param in zip(
            self.q2.parameters(), self.q2_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _maybe_log_and_save(self):
        # Log if we've advanced at least `log_interval` steps since last log.
        if (
            self.total_steps - getattr(self, "_last_log_steps", 0)
            >= self.cfg.log_interval
        ):
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
                        # "alpha": (
                        #     alpha_val
                        #     if alpha_val is not None
                        #     else float(self.cfg.alpha)
                        # ),
                    },
                    step=self.total_steps,
                )
            except Exception as e:
                print("Warning: wandb logging failed (_maybe_log_and_save):", e)

            # record last log
            self._last_log_steps = self.total_steps

        # Save if we've advanced at least `save_interval` steps since last save.
        if (
            self.total_steps - getattr(self, "_last_save_steps", 0)
            >= self.cfg.save_interval
        ):
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
                    "config": {
                        "use_noisy": getattr(self.cfg, "use_noisy", False),
                        "num_cells": getattr(self.cfg, "num_cells", 256),
                        "vae_checkpoint_path": getattr(
                            self.cfg, "vae_checkpoint_path", None
                        ),
                    },
                },
                f".\\models\\sac_checkpoint_{self.total_steps}.pt",
            )
            print(f"Saved checkpoint at step {self.total_steps}")

            # record last save
            self._last_save_steps = self.total_steps

            # * i dont have enough wandb storage to do this sooo
            # if self.cfg and getattr(self.cfg, "wandb", False) and WANDB_AVAILABLE:
            #     try:
            #         wandb.save(f".\\models\\sac_checkpoint_{self.total_steps}.pt")
            #         wandb.log(
            #             {
            #                 "checkpoint": f".\\models\\sac_checkpoint_{self.total_steps}.pt"
            #             },
            #             step=self.total_steps,
            #         )
            #     except Exception:
            #         pass


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
    value,
    value_target,
    q1,
    q2,
    q1_target,
    q2_target,
    actor_opt,
    critic_opt,
    value_opt,
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
        value,
        value_target,
        q1,
        q2,
        q1_target,
        q2_target,
        actor_opt,
        critic_opt,
        value_opt,
        log_alpha,
        alpha_opt,
        target_entropy,
        device,
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
