import math
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
import wandb

from .train_utils import fix_action_shape, unpack_pixels
from .logging_utils import log_info, log_success, log_warning, log_error, log_metric


class LearnerWorker:
    def __init__(
        self,
        cfg,
        rb,
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
        transitions_queue,
        env,
        device,
        storage=None,
        # optional shared-memory weight broadcast (multi-process)
        shared_weights=None,
        weights_lock=None,
        weights_version=None,
        log_queue=None,
        stop_event=None,
    ):
        self.cfg = cfg
        self.rb = rb
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.log_alpha = log_alpha
        self.alpha_opt = alpha_opt
        self.target_entropy = target_entropy if target_entropy is not None else -3.0
        self.transitions_queue = transitions_queue
        self.env = env
        self.device = device
        self.storage = storage
        self.shared_weights = shared_weights
        self.weights_lock = weights_lock
        self.weights_version = weights_version
        self.log_queue = log_queue
        self.stop_event = stop_event

        self.total_steps = 0
        self.episode_returns: list[float] = []
        self.start_time = time.time()

        self._last_log_steps = 0
        self._last_save_steps = 0
        self._last_save_rb_steps = 0
        self._best_avg_return = -float("inf")

        self._updates_count = 0
        self._updates_per_step = int(getattr(cfg, "updates_per_step", 1))
        self._actor_update_delay = getattr(cfg, "actor_update_delay", 1)
        self._critic_updates_count = 0
        self._actor_updates_count = 0
        self._alpha_updates_count = 0

        self._cumul_used: int = int(getattr(cfg, "cumul_memories_used", 0))
        self._cumul_should: int = int(getattr(cfg, "cumul_memories_should_have_been_used", 0))
        self._freeze_encoder_steps = getattr(cfg, "freeze_encoder_steps", 0)
        self._encoder_frozen = False
        self._last_epsilon = 0.0  # updated from collector meta messages

        _default_log_updates = int(getattr(cfg, "updates_per_step", 1)) * int(
            getattr(cfg, "log_interval", 1000)
        )
        self._log_update_every: int = int(getattr(cfg, "log_update_every", _default_log_updates))

        if getattr(cfg, "compile_models", False):
            log_info("Applying torch.compile to actor / critic networks...")
            try:
                _mode = str(getattr(cfg, "compile_mode", "reduce-overhead"))
                actor.module.module = torch.compile(actor.module.module, mode=_mode)
                q1.module = torch.compile(q1.module, mode=_mode)
                q2.module = torch.compile(q2.module, mode=_mode)
                q1_target.module = torch.compile(q1_target.module, mode=_mode)
                q2_target.module = torch.compile(q2_target.module, mode=_mode)
                log_success(f"torch.compile applied (mode={_mode})")
            except Exception as _ce:
                log_warning(f"torch.compile failed (continuing without it): {_ce}")

        _total_updates = max(
            1,
            int(getattr(cfg, "total_steps", 1_000_000)) * int(getattr(cfg, "updates_per_step", 1)),
        )
        self._total_update_budget = _total_updates
        self.actor_scheduler = self._build_scheduler(self.actor_opt, _total_updates)
        self.critic_scheduler = self._build_scheduler(self.critic_opt, _total_updates)

        self._expert_priority_refresh_updates = int(
            getattr(cfg, "expert_priority_refresh_updates", 1000)
        )
        self._expert_priority_initial_bonus = float(getattr(cfg, "expert_demo_epsilon", 1e-3))
        self._last_expert_bonus = max(0.0, self._expert_priority_initial_bonus)
        self._last_expert_target_priority = 0.0
        self._last_expert_base_priority = 0.0

    def run(self):
        """Run until ``stop_event`` is set (multi-process usage).

        Gradient updates are paced to the data collection rate:
        for every new transition, we earn ``updates_per_step`` update credits.
        The learner only trains when it has credits, keeping the effective UTD
        ratio equal to ``updates_per_step`` regardless of GPU speed.
        """
        _weight_push_every = max(1, self._updates_per_step)
        _update_credit = 0
        # Allow a small burst when lots of transitions arrive at once, but
        # cap to keep the loop responsive to stop_event / logging.
        _max_updates_per_tick = self._updates_per_step * 64

        while self.stop_event is None or not self.stop_event.is_set():
            n = self._drain_transitions()
            self.total_steps += n
            _update_credit += n * self._updates_per_step

            trained_this_iter = False
            start_steps = int(getattr(self.cfg, "start_steps", 0))
            if (
                _update_credit > 0
                and self.total_steps >= start_steps
                and len(self.rb) >= self.cfg.batch_size
            ):
                k = min(_update_credit, _max_updates_per_tick)
                for _ in range(k):
                    self._do_update()
                _update_credit -= k
                if self._updates_count % _weight_push_every == 0:
                    self._push_weights()
                trained_this_iter = True

            if not trained_this_iter:
                time.sleep(0.001)
            self._maybe_log_and_save(epsilon=self._last_epsilon)

    def _set_encoder_requires_grad(self, requires_grad: bool):
        for net in [self.actor, self.q1, self.q2]:
            if net is None:
                continue
            for name, module in net.named_modules():
                if "cnn" in name.lower() or isinstance(module, torch.nn.Conv2d):
                    for param in module.parameters():
                        param.requires_grad = requires_grad

    # === transition ingestion =====================================================================

    def _drain_transitions(self, max_items: int = 8192) -> int:
        """Pull sequence chunks from the queue into the replay buffer.

        Each item is either:
        - A dict with "_meta" key (episode return, epsilon updates)
        - A sequence dict from EpisodeAccumulator with keys:
            features: (T+1, F)
            actions:  (T, A)
            rewards:  (T, 1)
            dones:    (T, 1)
            terminated: (T, 1)
            mask:     (T, 1)
            vector:   (T+1, O)  [optional]
        """
        count = 0
        for _ in range(max_items):
            try:
                item = self.transitions_queue.get_nowait()
            except Exception:
                break

            # Meta messages (episode returns, epsilon, etc.)
            if isinstance(item, dict) and item.get("_meta"):
                if "episode_return" in item:
                    self.episode_returns.append(float(item["episode_return"]))
                    if len(self.episode_returns) > 10_000:
                        del self.episode_returns[:1_000]
                if "epsilon" in item:
                    self._last_epsilon = float(item["epsilon"])
                continue

            # Sequence chunk from EpisodeAccumulator
            # Convert to TensorDict for the replay buffer
            td_data = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    td_data[k] = v.cpu()

            # Keep each sequence as a single replay item.
            # Time is a data dimension inside tensors (features=(T+1,F), actions=(T,A), ...),
            # not the TensorDict batch dimension.
            td = TensorDict(td_data, batch_size=[])
            self.rb.add(td)
            # Count by the number of real timesteps in the sequence
            mask = item.get("mask", None)
            if mask is not None:
                count += int(mask.sum().item())
            else:
                count += item["actions"].shape[0]

        return count

    # === weight broadcast (multi-process) ===================================================

    def _push_weights(self):
        if self.shared_weights is None:
            return
        with self.weights_lock:
            for k, v in self.actor.state_dict().items():
                self.shared_weights[k].copy_(v)
            self.weights_version.value += 1

    def _get_current_max_priority(self) -> float:
        sampler_candidates = [
            self.rb,
            getattr(self.rb, "sampler", None),
            getattr(self.rb, "_sampler", None),
        ]
        attrs = ("max_priority", "_max_priority")

        for candidate in sampler_candidates:
            if candidate is None:
                continue
            for attr in attrs:
                value = getattr(candidate, attr, None)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if math.isfinite(value_f) and value_f > 0.0:
                    return value_f
        return 1.0

    def _maybe_refresh_expert_priorities(self):
        if not bool(getattr(self.cfg, "use_expert_demonstrations", False)):
            return
        if self._expert_priority_refresh_updates <= 0:
            return
        if self._updates_count <= 0:
            return
        if self._updates_count % self._expert_priority_refresh_updates != 0:
            return

        expert_indices = getattr(self.rb, "_expert_indices", None)
        if not isinstance(expert_indices, torch.Tensor) or expert_indices.numel() == 0:
            return

        progress = min(1.0, float(self._updates_count) / float(max(1, self._total_update_budget)))
        annealed_bonus = max(0.0, self._expert_priority_initial_bonus * (1.0 - progress))

        current_max = self._get_current_max_priority()
        base_priority = max(1e-6, current_max - max(0.0, self._last_expert_bonus))
        target_priority = base_priority + annealed_bonus
        per_max_priority = float(getattr(self.cfg, "per_max_priority", 100.0))
        target_priority = float(np.clip(target_priority, 1e-6, per_max_priority))

        idx = expert_indices.view(-1).to(torch.long).cpu()
        priorities = np.full((idx.numel(),), target_priority, dtype=np.float32)
        self.rb.update_priority(idx, priorities)

        self._last_expert_bonus = annealed_bonus
        self._last_expert_base_priority = base_priority
        self._last_expert_target_priority = target_priority

    def _to_device_fast(self, x: torch.Tensor, dtype=None) -> torch.Tensor:
        """Move tensor to learner device, using pinned-memory async copies when possible."""
        if not isinstance(x, torch.Tensor):
            return x
        if dtype is not None:
            x = x.to(dtype=dtype)
        if x.device.type == "cpu":
            try:
                x = x.pin_memory()
            except Exception:
                pass
            return x.to(self.device, non_blocking=True)
        return x.to(self.device)

    def _unpack_batch(self, batch):
        """Convert sampled batch to a properly batched dict of tensors.

        Handles both ListStorage (returns list/stacked TensorDicts) and
        LazyTensorStorage (returns a single TensorDict).

        Args:
            batch: Either a TensorDict, list of TensorDicts, or similar

        Returns:
            Dict with keys mapping to tensors (already batched)
        """
        if isinstance(batch, (list, tuple)):
            # ListStorage: batch is a list of TensorDicts or dicts
            stacked = {}
            keys = batch[0].keys() if hasattr(batch[0], "keys") else batch[0].keys()
            for k in keys:
                tensors = [b[k] for b in batch]
                stacked[k] = torch.stack(tensors, dim=0)
            return stacked
        elif isinstance(batch, TensorDict):
            # Already batched TensorDict - convert to dict
            return {k: batch[k] for k in batch.keys()}
        else:
            # Assume it's already a dict
            return batch

    # === gradient update ===========================================================================

    def _do_update(self):
        """Perform one gradient update using sequence chunks from the replay buffer."""

        if self._freeze_encoder_steps > 0:
            if self.total_steps < self._freeze_encoder_steps and not self._encoder_frozen:
                self._set_encoder_requires_grad(False)
                self._encoder_frozen = True
            elif self.total_steps >= self._freeze_encoder_steps and self._encoder_frozen:
                self._set_encoder_requires_grad(True)
                self._encoder_frozen = False

        batch, info = self.rb.sample(self.cfg.batch_size, return_info=True)
        batch_indices = info.get("index", None)

        # ── Unpack sequence batch ──────────────────────────────────────
        # features: (B, T+1, F) - precomputed CNN features
        # actions:  (B, T, A)
        # rewards:  (B, T, 1)
        # dones:    (B, T, 1)
        # mask:     (B, T, 1) - 1 for real steps, 0 for padding
        # vector:   (B, T+1, O) [optional]

        features = self._to_device_fast(batch["features"], dtype=torch.float32)
        actions_b = self._to_device_fast(batch["actions"], dtype=torch.float32)
        rewards_b = self._to_device_fast(batch["rewards"], dtype=torch.float32)
        mask = self._to_device_fast(batch["mask"], dtype=torch.float32)

        if "terminated" in batch.keys():
            terminal_mask = self._to_device_fast(batch["terminated"]).to(dtype=torch.float32)
        else:
            terminal_mask = self._to_device_fast(batch["dones"]).to(dtype=torch.float32)

        vector_b = (
            self._to_device_fast(batch["vector"], dtype=torch.float32)
            if "vector" in batch.keys()
            else None
        )

        B, T_plus_1, F = features.shape
        T = T_plus_1 - 1
        A = actions_b.shape[-1]

        # Split features into current (0..T-1) and next (1..T)
        obs_features = features[:, :T, :]  # (B, T, F)
        next_obs_features = features[:, 1:, :]  # (B, T, F)

        # Split vector similarly if present
        obs_vector = vector_b[:, :T, :] if vector_b is not None else None
        next_obs_vector = vector_b[:, 1:, :] if vector_b is not None else None

        alpha = self.log_alpha.exp() if self.log_alpha is not None else self.cfg.alpha

        # ══════════════════════════════════════════════════════════════════
        # CRITIC UPDATE
        # ══════════════════════════════════════════════════════════════════

        # ── Compute target Q values ────────────────────────────────────
        with torch.no_grad():
            # 1) Get next actions from actor (need to run actor LSTM on obs sequence)
            actor_net = self.actor.module.module  # unwrap TensorDictModule + ProbabilisticActor
            actor_params, next_actor_state = actor_net.forward_sequence(
                next_obs_features, vector=next_obs_vector
            )
            # Sample actions from the distribution
            from torchrl.modules import TanhNormal

            next_dist = TanhNormal(
                loc=actor_params["loc"],
                scale=actor_params["scale"],
                low=torch.tensor([-1.0, 0.0, 0.0], device=self.device),
                high=torch.tensor([1.0, 1.0, 1.0], device=self.device),
            )
            next_actions = next_dist.rsample()  # (B, T, A)
            next_log_prob = next_dist.log_prob(next_actions)  # (B, T) or (B, T, A)
            if next_log_prob.dim() == 3:
                next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)  # (B, T, 1)
            elif next_log_prob.dim() == 2:
                next_log_prob = next_log_prob.unsqueeze(-1)  # (B, T, 1)

            # 2) Run target critic LSTMs on the NEXT observation sequence
            #    Key insight: target critics process obs-only through LSTM,
            #    then combine with actions in the Q-head
            q1_target_net = self.q1_target.module
            q2_target_net = self.q2_target.module

            next_q1, _ = q1_target_net.forward_sequence(
                next_obs_features, next_actions, vector=next_obs_vector
            )
            next_q2, _ = q2_target_net.forward_sequence(
                next_obs_features, next_actions, vector=next_obs_vector
            )
            next_min_q = torch.min(next_q1, next_q2)  # (B, T, 1)

            next_v = next_min_q - alpha * next_log_prob
            q_target = rewards_b + self.cfg.gamma * (1.0 - terminal_mask) * next_v

            min_q = float(getattr(self.cfg, "min_q_target", -1000.0))
            max_q = float(getattr(self.cfg, "max_q_target", 1000.0))
            q_target = torch.clamp(q_target, min=min_q, max=max_q)

        # ── Compute predicted Q values ─────────────────────────────────
        # Online critics share the same LSTM obs pass, diverge at Q-head
        q1_net = self.q1.module
        q2_net = self.q2.module

        q1_pred, _ = q1_net.forward_sequence(obs_features, actions_b, vector=obs_vector)
        q2_pred, _ = q2_net.forward_sequence(obs_features, actions_b, vector=obs_vector)

        # ── PER importance-sampling weights ────────────────────────────
        is_weights = info.get("_weight", info.get("weight", None))
        if is_weights is not None:
            is_weights = self._to_device_fast(is_weights).view(B, 1, 1).clamp(min=1e-4)
        else:
            is_weights = torch.ones(B, 1, 1, device=self.device)

        # ── Masked Huber loss ──────────────────────────────────────────
        q1_elementwise = F.smooth_l1_loss(q1_pred, q_target, reduction="none")  # (B,T,1)
        q2_elementwise = F.smooth_l1_loss(q2_pred, q_target, reduction="none")

        # Apply sequence mask: don't backprop through padded timesteps
        q1_masked = (q1_elementwise * mask * is_weights).sum() / mask.sum().clamp(min=1)
        q2_masked = (q2_elementwise * mask * is_weights).sum() / mask.sum().clamp(min=1)
        critic_loss = q1_masked + q2_masked

        # ── TD error for PER priorities ────────────────────────────────
        with torch.no_grad():
            td_error_1 = torch.abs(q1_pred - q_target)
            td_error_2 = torch.abs(q2_pred - q_target)
            # Per-sequence priority: mean over valid timesteps
            td_errors_seq = torch.max(td_error_1, td_error_2)  # (B, T, 1)
            # Masked mean per sequence
            td_errors_per_seq = (td_errors_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            td_errors = td_errors_per_seq.squeeze(-1)  # (B,)

            if batch_indices is not None:
                _max_priority = float(getattr(self.cfg, "per_max_priority", 100.0))
                new_priorities = torch.clamp(td_errors, min=1e-6, max=_max_priority).cpu().numpy()
                self.rb.update_priority(batch_indices, new_priorities)

        # ── Explained variance ─────────────────────────────────────────
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
        self._critic_updates_count += 1

        # Update PER beta
        beta = min(
            1.0,
            self.cfg.per_beta
            + (1.0 - self.cfg.per_beta) * (self.total_steps / self.cfg.total_steps),
        )
        self.rb.beta = beta

        # ══════════════════════════════════════════════════════════════════
        # ACTOR UPDATE (delayed)
        # ══════════════════════════════════════════════════════════════════
        self._updates_count += 1
        self._maybe_refresh_expert_priorities()

        if self._updates_count % self._actor_update_delay == 0:
            # Re-run actor LSTM to get fresh actions for the current obs sequence
            actor_params, _ = actor_net.forward_sequence(obs_features, vector=obs_vector)

            from torchrl.modules import TanhNormal

            dist = TanhNormal(
                loc=actor_params["loc"],
                scale=actor_params["scale"],
                low=torch.tensor([-1.0, 0.0, 0.0], device=self.device),
                high=torch.tensor([1.0, 1.0, 1.0], device=self.device),
            )
            new_actions = dist.rsample()  # (B, T, A)
            log_prob_new = dist.log_prob(new_actions)
            if log_prob_new.dim() == 3:
                log_prob_new = log_prob_new.sum(dim=-1, keepdim=True)
            elif log_prob_new.dim() == 2:
                log_prob_new = log_prob_new.unsqueeze(-1)

            # Q-values for new actions: reuse the same LSTM hidden states
            # by running forward_sequence with new actions
            # Note: we detach the critic LSTM to prevent actor gradients
            # flowing through critic parameters
            with torch.no_grad():
                q1_h, _ = q1_net.forward_lstm(obs_features, vector=obs_vector)
                q2_h, _ = q2_net.forward_lstm(obs_features, vector=obs_vector)

            q1_new = q1_net.forward_q(q1_h.detach(), new_actions)  # (B, T, 1)
            q2_new = q2_net.forward_q(q2_h.detach(), new_actions)
            min_q_new = torch.min(q1_new, q2_new)

            # Masked actor loss
            actor_loss_elementwise = alpha.detach() * log_prob_new - min_q_new  # (B, T, 1)
            actor_loss = (actor_loss_elementwise * mask).sum() / mask.sum().clamp(min=1)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()
            self._actor_updates_count += 1

            # ── Alpha update ───────────────────────────────────────────
            alpha_loss = None
            if self.log_alpha is not None and self.alpha_opt is not None:
                with torch.no_grad():
                    entropy_error = log_prob_new + self.target_entropy
                alpha_loss_elementwise = -(self.log_alpha * entropy_error)
                alpha_loss = (alpha_loss_elementwise * mask).sum() / mask.sum().clamp(min=1)

                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
                self.alpha_opt.step()
                self._alpha_updates_count += 1

                alpha_min = float(getattr(self.cfg, "alpha_min", 0.01))
                alpha_max = float(getattr(self.cfg, "alpha_max", 1.0))
                with torch.no_grad():
                    self.log_alpha.clamp_(min=math.log(alpha_min), max=math.log(alpha_max))

            # ── Logging ────────────────────────────────────────────────
            if self._updates_count % self._log_update_every == 0:
                try:
                    loc_scale_dict = {
                        "actor/loc_mean": actor_params["loc"].mean().item(),
                        "actor/loc_std": actor_params["loc"].std().item(),
                        "actor/scale_mean": actor_params["scale"].mean().item(),
                        "actor/scale_min": actor_params["scale"].min().item(),
                        "actor/scale_max": actor_params["scale"].max().item(),
                    }
                    self._log(loc_scale_dict)
                except Exception:
                    pass
        else:
            actor_loss = None
            alpha_loss = None
            log_prob_new = None
            min_q_new = None
            new_actions = None

        self._soft_update_target()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # ── Periodic detailed logging ──────────────────────────────────
        if self._updates_count % self._log_update_every != 0:
            return

        try:
            current_entropy = -log_prob_new.mean().item() if log_prob_new is not None else 0.0
            log_dict = {
                "loss/critic_loss": critic_loss.item(),
                "loss/q1_loss": q1_masked.item(),
                "loss/q2_loss": q2_masked.item(),
                "loss/actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
                "critic/q_target_mean": q_target.mean().item(),
                "critic/q1_explained_variance": q1_explained_var.item(),
                "critic/q2_explained_variance": q2_explained_var.item(),
                "actor/entropy": current_entropy,
                "actor/alpha": alpha.item(),
                "per/td_error_mean": td_errors.mean().item(),
                "per/beta": beta,
                "updates/critic_updates_count": self._critic_updates_count,
                "updates/actor_updates_count": self._actor_updates_count,
            }
            self._log(log_dict)
        except Exception as e:
            print(f"[ERROR] logging: {e}")

    def _log(self, data: dict):
        """Send metrics to the log queue or wandb directly."""
        if self.log_queue is not None:
            try:
                self.log_queue.put_nowait({"step": self.total_steps, "data": data})
            except Exception:
                try:
                    wandb.log(data, step=self.total_steps)
                except Exception:
                    pass
        else:
            try:
                wandb.log(data, step=self.total_steps)
            except Exception:
                pass

    def _collect_noisy_stats(self, net, prefix: str) -> dict:
        """Return sigma/mu stats for all FactorisedNoisyLayer modules in *net*."""
        from assetto_corsa_rl.model.noisy import FactorisedNoisyLayer  # type: ignore

        sigma_w_vals, mu_w_vals = [], []
        for m in net.modules():
            if isinstance(m, FactorisedNoisyLayer):
                sigma_w_vals.append(m.sigma_weight.abs().mean().item())
                mu_w_vals.append(m.mu_weight.abs().mean().item())
        if not sigma_w_vals:
            return {}
        mean_sigma = sum(sigma_w_vals) / len(sigma_w_vals)
        mean_mu = sum(mu_w_vals) / len(mu_w_vals)
        return {
            f"noisy/{prefix}_sigma_weight_mean": mean_sigma,
            f"noisy/{prefix}_mu_weight_mean": mean_mu,
            f"noisy/{prefix}_noise_to_signal": mean_sigma / (mean_mu + 1e-8),
        }

    def _soft_update_target(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.lerp_(p.data, tau)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.lerp_(p.data, tau)

    # === LR scheduler factory =============================================================

    def _build_scheduler(self, opt, total_updates: int):
        """Build a learning-rate scheduler for *opt*.

        Controlled by cfg keys:
          lr_scheduler    : "cosine" | "linear" | "none"  (default: "none")
          lr_warmup_steps : int  gradient-update steps of linear warm-up (default: 0)
          lr_min_factor   : float  minimum LR = initial_lr * factor  (default: 0.1)
        """
        sched_type = str(getattr(self.cfg, "lr_scheduler", "none")).lower().strip()
        if sched_type == "none":
            return None

        warmup = int(getattr(self.cfg, "lr_warmup_steps", 0))
        min_factor = float(getattr(self.cfg, "lr_min_factor", 0.1))
        main_steps = max(1, total_updates - warmup)

        # Initial LR is whatever the optimizer has right now (respects sweep overrides).
        initial_lr = opt.param_groups[0]["lr"]
        eta_min = initial_lr * min_factor

        if sched_type == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=main_steps, eta_min=eta_min
            )
        elif sched_type == "linear":
            main_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1.0, end_factor=min_factor, total_iters=main_steps
            )
        else:
            print(f"[scheduler] Unknown lr_scheduler='{sched_type}', disabling scheduler.")
            return None

        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-4, end_factor=1.0, total_iters=warmup
            )
            combined = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, main_sched], milestones=[warmup]
            )
            log_info(
                f"{sched_type} scheduler with warmup={warmup} steps, "
                f"T={main_steps}, lr {initial_lr:.2e} → {eta_min:.2e}"
            )
            return combined

        log_info(f"{sched_type} scheduler: T={main_steps}, lr {initial_lr:.2e} → {eta_min:.2e}")
        return main_sched

    # === periodic logging & checkpointing ================================================â”€

    def _maybe_log_and_save(self, epsilon: float = 0.0):
        last = self.episode_returns[-100:]
        avg_return = sum(last) / len(last) if last else 0.0

        if self.total_steps - self._last_log_steps >= self.cfg.log_interval:
            elapsed = time.time() - self.start_time
            print(
                f"Steps: {self.total_steps}, AvgReturn(100): {avg_return:.2f}, "
                f"Buffer: {len(self.rb)}, Time: {elapsed:.1f}s, Eps: {epsilon:.3f}"
            )
            try:
                stats_dict = {
                    "steps": self.total_steps,
                    "reward/rewards_per_environment_mean": avg_return,
                    "buffer": len(self.rb),
                    "time": elapsed,
                    "epsilon": epsilon,
                }
                if self.log_queue is not None:
                    try:
                        self.log_queue.put_nowait({"step": self.total_steps, "data": stats_dict})
                    except Exception as e:
                        print(f"[WARNING] Failed to log stats to queue: {e}")
                        # Fallback to direct wandb logging
                        try:
                            wandb.log(stats_dict, step=self.total_steps)
                        except Exception as e2:
                            print(f"[ERROR] Direct wandb.log also failed: {e2}")
                else:
                    wandb.log(stats_dict, step=self.total_steps)
            except Exception as e:
                print(f"[ERROR] Exception in _maybe_log_and_save: {e}")
            self._last_log_steps = self.total_steps

        if self.total_steps - self._last_save_steps >= self.cfg.save_interval:
            save_dir = Path("./models")
            save_dir.mkdir(parents=True, exist_ok=True)

            ckpt = {
                "actor_state": self.actor.state_dict(),
                "q1_state": self.q1.state_dict(),
                "q2_state": self.q2.state_dict(),
                "q1_target_state": self.q1_target.state_dict(),
                "q2_target_state": self.q2_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "steps": self.total_steps,
                "avg_return": avg_return,
                "config": {
                    "use_noisy": getattr(self.cfg, "use_noisy", False),
                    "num_cells": getattr(self.cfg, "num_cells", 256),
                    "vae_checkpoint_path": getattr(self.cfg, "vae_checkpoint_path", None),
                    "use_lstm": getattr(self.cfg, "use_lstm", False),
                    "lstm_hidden_size": getattr(self.cfg, "lstm_hidden_size", 256),
                    "lstm_layers": getattr(self.cfg, "lstm_layers", 1),
                    "stateful_inference": getattr(self.cfg, "stateful_inference", True),
                },
            }

            torch.save(ckpt, save_dir / "sac_last.pt")
            print(f"Saved sac_last.pt at step {self.total_steps} (avg_return={avg_return:.4f})")

            if avg_return > self._best_avg_return:
                self._best_avg_return = avg_return
                torch.save(ckpt, save_dir / "sac_best.pt")
                print(
                    f"Saved sac_best.pt at step {self.total_steps} (new best avg_return={avg_return:.4f})"
                )

            self._last_save_steps = self.total_steps

        rb_save_interval = getattr(
            self.cfg, "save_interval_replaybuffer", self.cfg.save_interval * 2
        )
        if self.total_steps - self._last_save_rb_steps >= rb_save_interval:
            rb_dir = Path("./models")
            rb_dir.mkdir(parents=True, exist_ok=True)
            rb_path = rb_dir / f"replay_buffer_{self.total_steps}.pt"
            try:
                min_free_space_gb = getattr(self.cfg, "min_free_space_gb", 10)
                min_free_space_bytes = min_free_space_gb * 1024 * 1024 * 1024
                stat = shutil.disk_usage(rb_dir)
                available_space = stat.free
                existing_buffers = sorted(rb_dir.glob("replay_buffer_*.pt"))
                while existing_buffers and available_space < min_free_space_bytes:
                    oldest_buffer = existing_buffers.pop(0)
                    buffer_size = oldest_buffer.stat().st_size
                    oldest_buffer.unlink()
                    available_space += buffer_size
                    print(
                        f"Deleted old replay buffer: {oldest_buffer.name} "
                        f"(freed {buffer_size / (1024**3):.2f} GB)"
                    )
                stat = shutil.disk_usage(rb_dir)
                if stat.free < min_free_space_bytes:
                    print(
                        f"Warning: Low disk space ({stat.free / (1024**3):.2f} GB free). "
                        "Skipping replay buffer save."
                    )
                else:
                    # Try torch.save first (better for torch tensors and LazyTensorStorage)
                    # Fallback to pickle if torch.save fails
                    try:
                        rb_state = {
                            "buffer": self.rb._storage._storage,
                            "sampler_state": {
                                "alpha": getattr(self.rb._sampler, "_alpha", None),
                                "beta": getattr(self.rb._sampler, "_beta", None),
                            },
                            "total_steps": self.total_steps,
                            "buffer_size": len(self.rb),
                        }
                        torch.save(rb_state, rb_path)
                    except (RuntimeError, ValueError, TypeError) as e:
                        # LazyTensorStorage may have closed file handles - try pickle
                        if "closed file" in str(e).lower():
                            # Silently skip if storage is in invalid state
                            return
                        with open(rb_path, "wb") as f:
                            pickle.dump(rb_state, f)

                    saved_size = rb_path.stat().st_size
                    remaining_space = shutil.disk_usage(rb_dir).free
                    print(
                        f"Saved replay buffer at step {self.total_steps} "
                        f"({len(self.rb)} transitions, "
                        f"{saved_size / (1024**3):.2f} GB, "
                        f"{remaining_space / (1024**3):.2f} GB free)"
                    )
                    self._last_save_rb_steps = self.total_steps
            except Exception as e:
                # Log but don't fail training if buffer save fails
                # (buffer can be restored from expert demos on restart)
                log_warning(f"Failed to save replay buffer: {e}")
