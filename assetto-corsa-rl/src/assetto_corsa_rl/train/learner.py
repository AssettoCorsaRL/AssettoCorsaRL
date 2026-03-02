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
        self._freeze_encoder_steps = getattr(cfg, "freeze_encoder_steps", 0)
        self._encoder_frozen = False
        self._last_epsilon = 0.0  # updated from collector meta messages

        # Fixed-size numpy array so:
        #   1. Index access is O(1) with no hash overhead
        #   2. Expired slots are reset to 0, so ListStorage slot reuse starts
        #      fresh (prevents stale counts from killing newly-added transitions)
        _rb_max = int(getattr(cfg, "replay_size", 1_000_000))
        self._sample_counts = np.zeros(_rb_max, dtype=np.int32)
        self._max_uses: int = int(
            getattr(cfg, "number_times_single_memory_is_used_before_discard", 32)
        )
        self._last_expired_count: int = 0
        # Log to wandb/queue every N gradient updates (not every single step)
        self._log_update_every: int = int(getattr(cfg, "log_update_every", 100))

        if getattr(cfg, "compile_models", False):
            print("[compile] Applying torch.compile to actor / critic networks...")
            try:
                _mode = str(getattr(cfg, "compile_mode", "reduce-overhead"))
                actor.module.module = torch.compile(actor.module.module, mode=_mode)
                q1.module = torch.compile(q1.module, mode=_mode)
                q2.module = torch.compile(q2.module, mode=_mode)
                q1_target.module = torch.compile(q1_target.module, mode=_mode)
                q2_target.module = torch.compile(q2_target.module, mode=_mode)
                print(f"[compile] Done (mode={_mode}).")
            except Exception as _ce:
                print(f"[compile] torch.compile failed (continuing without it): {_ce}")

        # ── LR schedulers ─────────────────────────────────────────────────────
        # Total gradient updates over the whole run (used as T_max / total_iters).
        _total_updates = max(
            1,
            int(getattr(cfg, "total_steps", 1_000_000)) * int(getattr(cfg, "updates_per_step", 1)),
        )
        self.actor_scheduler = self._build_scheduler(self.actor_opt, _total_updates)
        self.critic_scheduler = self._build_scheduler(self.critic_opt, _total_updates)

    def run(self):
        """Run until ``stop_event`` is set (multi-process usage)."""
        while self.stop_event is None or not self.stop_event.is_set():
            n = self._drain_transitions()
            self.total_steps += n
            if len(self.rb) >= self.cfg.batch_size and n > 0:
                for _ in range(self._updates_per_step):
                    self._do_update()
                self._push_weights()
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
        """Pull up to *max_items* transitions from the queue into the replay buffer.

        Returns the number of step transitions ingested (meta messages excluded).
        """
        count = 0
        for _ in range(max_items):
            try:
                item = self.transitions_queue.get_nowait()
            except Exception:
                break
            if isinstance(item, dict) and item.get("_meta"):
                if "episode_return" in item:
                    self.episode_returns.append(float(item["episode_return"]))
                    # Prevent unbounded growth — keep only the most recent 10 000 episodes
                    if len(self.episode_returns) > 10_000:
                        del self.episode_returns[:1_000]
                if "epsilon" in item:
                    self._last_epsilon = float(item["epsilon"])
                continue
            td_data = {
                "pixels": item["pixels"],
                "action": item["action"],
                "reward": item["reward"],
                "next_pixels": item["next_pixels"],
                "done": item["done"],
            }
            if "vector" in item:
                td_data["vector"] = item["vector"]
            if "next_vector" in item:
                td_data["next_vector"] = item["next_vector"]
            self.rb.add(TensorDict(td_data, batch_size=[]))
            count += 1
        return count

    # === weight broadcast (multi-process) ===================================================

    def _push_weights(self):
        if self.shared_weights is None:
            return
        with self.weights_lock:
            for k, v in self.actor.state_dict().items():
                self.shared_weights[k].copy_(v)
            self.weights_version.value += 1

    # === gradient update ===========================================================================â”€

    def _do_update(self):
        if self._freeze_encoder_steps > 0:
            if self.total_steps < self._freeze_encoder_steps and not self._encoder_frozen:
                self._set_encoder_requires_grad(False)
                self._encoder_frozen = True
                print(
                    f"[INFO] Visual encoder frozen "
                    f"(step {self.total_steps} < {self._freeze_encoder_steps})"
                )
            elif self.total_steps >= self._freeze_encoder_steps and self._encoder_frozen:
                self._set_encoder_requires_grad(True)
                self._encoder_frozen = False
                print(
                    f"[INFO] Visual encoder unfrozen "
                    f"(step {self.total_steps} >= {self._freeze_encoder_steps})"
                )

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

        # Extract optional telemetry vectors from the batch
        vector_b = batch["vector"].to(self.device) if "vector" in batch.keys() else None
        next_vector_b = (
            batch["next_vector"].to(self.device) if "next_vector" in batch.keys() else None
        )

        # ===== Critic Update =====
        with torch.no_grad():
            next_td_data = {"pixels": next_pixels_b}
            if next_vector_b is not None:
                next_td_data["vector"] = next_vector_b
            next_td = TensorDict(next_td_data, batch_size=next_pixels_b.shape[0])
            next_out = self.actor(next_td)
            next_actions = fix_action_shape(
                next_out["action"],
                next_pixels_b.shape[0],
                action_dim=self.env.action_spec.shape[-1],
            )
            next_log_prob = next_out["action_log_prob"].view(-1, 1)

            next_q1 = self.q1_target.module(next_pixels_b, next_actions, next_vector_b).view(-1, 1)
            next_q2 = self.q2_target.module(next_pixels_b, next_actions, next_vector_b).view(-1, 1)
            next_min_q = torch.min(next_q1, next_q2)

            alpha = self.log_alpha.exp() if self.log_alpha is not None else self.cfg.alpha
            next_v = next_min_q - alpha * next_log_prob
            q_target = rewards_b + self.cfg.gamma * (1.0 - dones_b) * next_v
            min_q = float(getattr(self.cfg, "min_q_target", -1000.0))
            max_q = float(getattr(self.cfg, "max_q_target", 1000.0))
            q_target = torch.clamp(q_target, min=min_q, max=max_q)

        actions_b = fix_action_shape(
            actions_b, pixels_b.shape[0], action_dim=self.env.action_spec.shape[-1]
        )
        q1_pred = self.q1.module(pixels_b, actions_b, vector_b).view(-1, 1)
        q2_pred = self.q2.module(pixels_b, actions_b, vector_b).view(-1, 1)

        with torch.no_grad():
            td_error_1 = torch.abs(q1_pred - q_target)
            td_error_2 = torch.abs(q2_pred - q_target)
            td_errors = torch.max(td_error_1, td_error_2).squeeze(-1)
            if batch_indices is not None:
                indices_arr = batch_indices.flatten().cpu().numpy()
                new_priorities = td_errors.cpu().numpy().copy()
                # Vectorised update — O(batch) instead of O(batch) Python loop
                self._sample_counts[indices_arr] += 1
                expired_mask = self._sample_counts[indices_arr] >= self._max_uses
                new_priorities[expired_mask] = 1e-8  # expired: never sample again
                # Reset counters so that when ListStorage REUSES a slot the new
                # transition starts with count=0 instead of inheriting stale count
                self._sample_counts[indices_arr[expired_mask]] = 0
                _expired = int(expired_mask.sum())
                self.rb.update_priority(batch_indices, new_priorities)
                self._last_expired_count = _expired

        beta = min(
            1.0,
            self.cfg.per_beta
            + (1.0 - self.cfg.per_beta) * (self.total_steps / self.cfg.total_steps),
        )
        self.rb.beta = beta

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

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
            td_in_data = {"pixels": pixels_b}
            if vector_b is not None:
                td_in_data["vector"] = vector_b
            td_in = TensorDict(td_in_data, batch_size=pixels_b.shape[0])
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

            if self._updates_count % self._log_update_every == 0:
                try:
                    loc_scale_dict = {
                        "actor/loc_mean": out["loc"].mean().item(),
                        "actor/loc_std": out["loc"].std().item(),
                        "actor/loc_max": out["loc"].max().item(),
                        "actor/scale_mean": out["scale"].mean().item(),
                        "actor/scale_min": out["scale"].min().item(),
                        "actor/scale_max": out["scale"].max().item(),
                    }
                    if self.log_queue is not None:
                        self.log_queue.put_nowait(
                            {"step": self.total_steps, "data": loc_scale_dict}
                        )
                    else:
                        wandb.log(loc_scale_dict, step=self.total_steps)
                except Exception:
                    pass

            q1_new = self.q1.module(pixels_b, new_actions, vector_b).view(-1, 1)
            q2_new = self.q2.module(pixels_b, new_actions, vector_b).view(-1, 1)
            min_q_new = torch.min(q1_new, q2_new)

            actor_loss = (alpha.detach() * log_prob_new - min_q_new).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()

            # ===== Alpha Update =====
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

        self._soft_update_target()

        # ── Step LR schedulers ────────────────────────────────────────────────
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # ===== Logging (throttled) =====
        if self._updates_count % self._log_update_every != 0:
            return
        try:
            current_entropy = -log_prob_new.mean().item() if log_prob_new is not None else 0.0
            log_dict = {
                "loss/critic_loss": critic_loss.item(),
                "loss/q1_loss": q1_loss.item(),
                "loss/q2_loss": q2_loss.item(),
                "loss/actor_loss": (actor_loss.item() if actor_loss is not None else 0.0),
                "critic/mean_q_value": (min_q_new.mean().item() if min_q_new is not None else 0.0),
                "critic/q_target_mean": q_target.mean().item(),
                "critic/next_v_mean": next_v.mean().item(),
                "critic/q1_explained_variance": q1_explained_var.item(),
                "critic/q2_explained_variance": q2_explained_var.item(),
                "actor/mean_log_prob": (
                    log_prob_new.mean().item() if log_prob_new is not None else 0.0
                ),
                "actor/entropy": current_entropy,
                "actor/target_entropy": self.target_entropy,
                "actor/entropy_gap": current_entropy - self.target_entropy,
                "actor/alpha": alpha.item(),
                "actor/alpha_loss": (alpha_loss.item() if alpha_loss is not None else 0.0),
                "reward/rewards_per_step_mean": rewards_b.mean().item(),
                "reward/rewards_per_step_max": rewards_b.max().item(),
                "reward/rewards_per_step_min": rewards_b.min().item(),
                "actions/actions_std": actions_b.std().item(),
                "actions/sampled_action_mean": (
                    new_actions.mean().item() if new_actions is not None else 0.0
                ),
                "actions/sampled_action_std": (
                    new_actions.std().item() if new_actions is not None else 0.0
                ),
                "per/td_error_mean": td_errors.mean().item(),
                "per/td_error_max": td_errors.max().item(),
                "per/td_error_min": td_errors.min().item(),
                "per/beta": beta,
                "buffer/expired_this_batch": self._last_expired_count,
                "buffer/expired_fraction": self._last_expired_count / max(1, self.cfg.batch_size),
                "lr/actor_lr": self.actor_opt.param_groups[0]["lr"],
                "lr/critic_lr": self.critic_opt.param_groups[0]["lr"],
            }
            if getattr(self.cfg, "use_noisy", False):
                log_dict.update(self._collect_noisy_stats(self.actor, "actor"))
                log_dict.update(self._collect_noisy_stats(self.q1, "q1"))
            if self.log_queue is not None:
                try:
                    self.log_queue.put_nowait({"step": self.total_steps, "data": log_dict})
                except Exception:
                    pass
            else:
                wandb.log(log_dict, step=self.total_steps)
        except Exception as e:
            print("Warning: wandb logging failed (updates):", e)

    # === noisy layer diagnostics ================================================================

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
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
            print(
                f"[scheduler] {sched_type} (warmup={warmup} steps, "
                f"T={main_steps}, lr {initial_lr:.2e} → {eta_min:.2e})"
            )
            return combined

        print(f"[scheduler] {sched_type} " f"(T={main_steps}, lr {initial_lr:.2e} → {eta_min:.2e})")
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
                    self.log_queue.put_nowait({"step": self.total_steps, "data": stats_dict})
                else:
                    wandb.log(stats_dict, step=self.total_steps)
            except Exception as e:
                print("Warning: wandb logging failed (_maybe_log_and_save):", e)
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
                },
            }

            # Always save sac_last.pt
            torch.save(ckpt, save_dir / "sac_last.pt")
            print(f"Saved sac_last.pt at step {self.total_steps} (avg_return={avg_return:.4f})")

            # Save sac_best.pt if this is a new best
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
                    rb_state = {
                        "buffer": self.rb._storage._storage,
                        "sampler_state": {
                            "alpha": getattr(self.rb._sampler, "_alpha", None),
                            "beta": getattr(self.rb._sampler, "_beta", None),
                        },
                        "total_steps": self.total_steps,
                        "buffer_size": len(self.rb),
                    }
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
                print(f"Warning: Failed to save replay buffer: {e}")
