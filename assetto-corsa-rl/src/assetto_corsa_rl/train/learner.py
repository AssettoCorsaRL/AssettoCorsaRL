import math
import pickle
import shutil
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
import wandb

from .train_utils import fix_action_shape, unpack_pixels
from .logging_utils import log_info, log_success, log_warning, log_error, log_metric
from torchrl.modules import TanhNormal


class PinnedMemoryCache:
    """Reuse pinned memory buffers to avoid unbounded allocation.

    PyTorch's tensor.pin_memory() allocates a new buffer each call.
    This cache pools buffers by shape/dtype to prevent memory leaks.
    """

    def __init__(self, max_buffers=32):
        self.cache = {}
        self.max_buffers = max_buffers
        self._access_order = []  # Track insertion order for FIFO eviction

    def get_pinned(self, tensor):
        """Get or create a pinned buffer matching tensor shape/dtype."""
        key = (tuple(tensor.shape), tensor.dtype)

        if key not in self.cache:
            # Evict oldest if cache is full
            if len(self.cache) >= self.max_buffers:
                oldest_key = self._access_order.pop(0)
                self.cache.pop(oldest_key, None)

            # Create new pinned buffer
            self.cache[key] = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            self._access_order.append(key)

        # Copy data into pinned buffer
        self.cache[key].copy_(tensor)
        return self.cache[key]


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
        self.target_entropy = target_entropy if target_entropy is not None else -2.0
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
        self.episode_returns: deque = deque(maxlen=10_000)  # Auto-evicts old entries
        self.start_time = time.time()

        self._last_log_steps = 0
        self._last_save_steps = 0
        self._last_save_rb_steps = 0
        self._best_avg_return = -float("inf")

        self._updates_count = 0
        self._updates_per_step = int(getattr(cfg, "updates_per_step", 1))
        self._actor_update_delay = getattr(cfg, "actor_update_delay", 1)
        _configured_max_updates = int(
            getattr(cfg, "max_updates_per_tick", self._updates_per_step * 16)
        )
        _responsive_cap = max(1, self._updates_per_step * 32)
        self._max_updates_per_tick = max(1, min(_configured_max_updates, _responsive_cap))
        self._max_update_credit = int(
            getattr(cfg, "max_update_credit", self._max_updates_per_tick * 4)
        )
        self._drain_max_items = max(1, int(getattr(cfg, "drain_max_items", 16384)))
        self._adaptive_updates = bool(getattr(cfg, "adaptive_updates", True))
        self._adaptive_queue_low = float(getattr(cfg, "adaptive_queue_low", 0.35))
        self._adaptive_queue_high = float(getattr(cfg, "adaptive_queue_high", 0.75))
        self._adaptive_scale_up = float(getattr(cfg, "adaptive_scale_up", 2.0))
        self._adaptive_scale_down = float(getattr(cfg, "adaptive_scale_down", 0.5))
        self._adaptive_max_updates_per_tick = max(
            self._max_updates_per_tick,
            int(getattr(cfg, "adaptive_max_updates_per_tick", self._max_updates_per_tick * 4)),
        )
        self._critic_updates_count = 0
        self._actor_updates_count = 0
        self._alpha_updates_count = 0

        self._cumul_used: int = int(getattr(cfg, "cumul_memories_used", 0))
        self._cumul_should: int = int(getattr(cfg, "cumul_memories_should_have_been_used", 0))
        self._freeze_encoder_steps = getattr(cfg, "freeze_encoder_steps", 0)
        self._encoder_frozen = False
        self._last_epsilon = 0.0  # updated from collector meta messages
        self._last_drain_count = 0
        self._last_train_batches = 0
        self._queue_size = -1
        self._queue_capacity = int(getattr(cfg, "queue_size", 0))
        self._collector_enqueue_full_count = 0
        self._collector_actor_infer_calls = 0
        self._last_effective_max_updates = float(self._max_updates_per_tick)
        self._last_update_credit = 0.0
        self._last_queue_fill_ratio = -1.0

        _default_log_updates = int(getattr(cfg, "updates_per_step", 1)) * int(
            getattr(cfg, "log_interval", 1000)
        )
        self._log_update_every: int = int(getattr(cfg, "log_update_every", _default_log_updates))
        log_info(
            "Learner schedule: updates_per_step=%s actor_update_delay=%s log_update_every=%s"
            % (self._updates_per_step, self._actor_update_delay, self._log_update_every)
        )

        if getattr(cfg, "compile_models", False):
            log_info("Applying torch.compile to actor / critic networks...")
            try:
                _mode = str(getattr(cfg, "compile_mode", "reduce-overhead"))
                actor_net = self._resolve_actor_net()
                compiled_actor = torch.compile(actor_net, mode=_mode)
                if not self._set_actor_net(compiled_actor):
                    log_warning(
                        "Could not replace wrapped actor net with compiled module; using eager actor"
                    )
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

        # Pinned memory cache for fast CPU→GPU transfers
        self._pinned_cache = PinnedMemoryCache()

    def _resolve_actor_net(self):
        """Return the underlying ActorNet exposing ``forward_features``.

        TorchRL wrappers can vary across versions, e.g.:
        - actor.module.module (TensorDictModule)
        - actor.module[0].module (ModuleList wrapper)
        - actor.module (already the raw net)
        """
        queue = [self.actor, getattr(self.actor, "module", None)]
        visited = set()

        while queue:
            node = queue.pop(0)
            if node is None:
                continue

            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)

            if hasattr(node, "forward_features"):
                return node

            inner = getattr(node, "module", None)
            if inner is not None:
                queue.append(inner)

            if isinstance(node, torch.nn.ModuleList):
                queue.extend(list(node))

            if isinstance(node, (list, tuple)):
                queue.extend(list(node))

        raise AttributeError("Could not resolve underlying actor network from actor wrappers")

    def _set_actor_net(self, new_actor_net) -> bool:
        """Best-effort replacement of the wrapped ActorNet (used for torch.compile)."""
        actor_module = getattr(self.actor, "module", None)
        if actor_module is None:
            return False

        if hasattr(actor_module, "module"):
            actor_module.module = new_actor_net
            return True

        if isinstance(actor_module, torch.nn.ModuleList) and len(actor_module) > 0:
            first = actor_module[0]
            if hasattr(first, "module"):
                first.module = new_actor_net
                return True
            if hasattr(first, "forward_features"):
                actor_module[0] = new_actor_net
                return True

        if hasattr(actor_module, "forward_features"):
            self.actor.module = new_actor_net
            return True

        return False

    def run(self):
        """Run until ``stop_event`` is set (multi-process usage).

        Simple credit-based training loop:
        - After warmup, earn ``updates_per_step`` credits per collected transition
        - Spend up to ``max_updates_per_tick`` credits per loop iteration
        - Cap accumulated credit to avoid long learner stalls
        """
        _weight_push_every = max(1, self._updates_per_step)
        _update_credit = 0
        start_steps = int(getattr(self.cfg, "start_steps", 0))

        while self.stop_event is None or not self.stop_event.is_set():
            # Pull newly collected transitions first.
            n = self._drain_transitions(max_items=self._drain_max_items)
            self._last_drain_count = n
            self.total_steps += n

            if self.total_steps < start_steps:
                _update_credit = 0

            queue_fill_ratio = -1.0
            if self._queue_size >= 0 and self._queue_capacity > 0:
                queue_fill_ratio = float(self._queue_size) / float(self._queue_capacity)
            self._last_queue_fill_ratio = queue_fill_ratio

            effective_max_updates = self._max_updates_per_tick
            if self._adaptive_updates and queue_fill_ratio >= 0.0:
                if queue_fill_ratio >= self._adaptive_queue_high:
                    boosted = int(max(1.0, effective_max_updates * self._adaptive_scale_up))
                    effective_max_updates = min(self._adaptive_max_updates_per_tick, boosted)
                elif queue_fill_ratio <= self._adaptive_queue_low:
                    reduced = int(max(1.0, effective_max_updates * self._adaptive_scale_down))
                    effective_max_updates = max(1, reduced)

            effective_credit_cap = max(self._max_update_credit, effective_max_updates * 4)

            has_enough_data = (
                self.total_steps >= start_steps and len(self.rb) >= self.cfg.batch_size
            )

            if has_enough_data and n > 0:
                _update_credit += n * self._updates_per_step
                _update_credit = min(_update_credit, effective_credit_cap)

            k = 0
            if has_enough_data and _update_credit > 0:
                k = int(min(_update_credit, effective_max_updates))

            if k > 0:
                for _ in range(k):
                    self._do_update()

                _update_credit -= k
                self._last_train_batches = k
                if self._updates_count % _weight_push_every == 0:
                    self._push_weights()
            else:
                self._last_train_batches = 0
                time.sleep(0.001)

            self._last_effective_max_updates = float(effective_max_updates)
            self._last_update_credit = float(_update_credit)

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
        """Pull single-step transitions from the queue into the replay buffer.

        Batches transitions before adding to avoid O(log n) sum-tree updates per item.

        Each item is either:
        - A dict with "_meta" key (episode return, epsilon updates)
        - A single-step transition dict with keys:
            pixels: (C, H, W)
            next_pixels: (C, H, W)
            action: (A)
            reward: (1)
            done: (1)
            terminated: (1)
            vector/next_vector: (O) [optional]
        """
        count = 0
        batch_transitions = []

        for _ in range(max_items):
            try:
                item = self.transitions_queue.get_nowait()
            except Exception:
                break

            # Meta messages (episode returns, epsilon, etc.)
            if isinstance(item, dict) and item.get("_meta"):
                # Flush accumulated batch before processing meta
                if batch_transitions:
                    self._add_transitions_batch(batch_transitions)
                    count += len(batch_transitions)
                    batch_transitions = []

                if "episode_return" in item:
                    self.episode_returns.append(float(item["episode_return"]))  # deque auto-evicts
                if "epsilon" in item:
                    self._last_epsilon = float(item["epsilon"])
                if "queue_size" in item and item["queue_size"] is not None:
                    self._queue_size = int(item["queue_size"])
                if "queue_capacity" in item and item["queue_capacity"] is not None:
                    self._queue_capacity = int(item["queue_capacity"])
                if "collector/enqueue_full_count" in item:
                    self._collector_enqueue_full_count = int(item["collector/enqueue_full_count"])
                if "collector/actor_infer_calls" in item:
                    self._collector_actor_infer_calls = int(item["collector/actor_infer_calls"])
                continue

            if isinstance(item, dict) and item.get("_batch"):
                transitions = item.get("transitions", [])
                if transitions:
                    self._add_transitions_batch(transitions)
                    count += len(transitions)
                continue

            if isinstance(item, (list, tuple)) and item and isinstance(item[0], dict):
                self._add_transitions_batch(list(item))
                count += len(item)
                continue

            # Accumulate transition for batch add
            batch_transitions.append(item)

        # Flush remaining batch
        if batch_transitions:
            self._add_transitions_batch(batch_transitions)
            count += len(batch_transitions)

        return count

    def _add_transitions_batch(self, transitions_list: list) -> None:
        """Add a batch of transitions efficiently by stacking before inserting.

        Reduces sum-tree update overhead by batching additions.
        """
        if not transitions_list:
            return

        # Stack all tensors from the batch
        stacked_data = {}
        for key in transitions_list[0].keys():
            values = [t[key] for t in transitions_list if key in t]
            if not values:
                continue

            # Stack the tensors
            if isinstance(values[0], torch.Tensor):
                stacked_data[key] = torch.stack(values, dim=0)
            else:
                # For non-tensors, convert to tensor first
                stacked_data[key] = torch.tensor(
                    [v if isinstance(v, (int, float, bool)) else v.item() for v in values]
                )

        # Create batched TensorDict and insert all rows as individual replay entries.
        # Using ``add`` with a batched TensorDict can store each batch as one item in
        # ListStorage, leading to variable leading dimensions at sample-time.
        td = TensorDict(stacked_data, batch_size=[len(transitions_list)])
        if hasattr(self.rb, "extend"):
            self.rb.extend(td)
        else:
            for i in range(len(transitions_list)):
                self.rb.add(td[i])

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
        if not bool(getattr(self.cfg, "use_per", True)):
            return  # Skip if not using PER
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
        if not isinstance(x, torch.Tensor):
            return x
        if dtype is not None:
            x = x.to(dtype=dtype)
        return x.to(self.device, non_blocking=True)

    def _extract_cnn_and_compute_features(
        self, pixels: torch.Tensor, actor_net=None
    ) -> torch.Tensor:
        """Extract CNN from actor and compute features from pixels.

        Args:
            pixels: Tensor of shape [B, C, H, W]
            actor_net: Pre-resolved actor network (optional, for caching)

        Returns:
            Features tensor of shape [B, F]
        """
        if actor_net is None:
            actor_net = self._resolve_actor_net()

        cnn = None
        for m in actor_net.modules():
            if hasattr(m, "cnn"):
                cnn = m.cnn
                break

        if cnn is None:
            raise RuntimeError("Could not find CNN in actor network")

        with torch.no_grad():
            features = cnn(pixels)
        return features

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

    @staticmethod
    def _as_batch_column(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Return tensor as shape [B, 1] by reducing non-batch dims with mean."""
        if x is None:
            return x
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 0:
            return x.view(1, 1).expand(batch_size, 1)
        if x.ndim == 1:
            return x.view(batch_size, 1)
        return x.reshape(batch_size, -1).mean(dim=1, keepdim=True)

    def _sample_squashed(self, loc, scale):
        """Numerically stable squashed-Gaussian sample + log_prob.

        Avoids atanh entirely by keeping the pre-tanh sample z.
        """
        low = torch.full_like(loc, -1.0)
        high = torch.full_like(loc, 1.0)

        normal = torch.distributions.Normal(loc, scale)
        z = normal.rsample()  # pre-tanh

        tanh_z = torch.tanh(z)  # in [-1, 1]

        # map [-1,1] → [low, high]
        actions = low + (tanh_z + 1.0) * 0.5 * (high - low)

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1.0 - tanh_z.pow(2) + 1e-6)  # tanh correction
        log_prob -= torch.log((high - low) * 0.5 + 1e-8)  # affine scaling
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, min=-100.0, max=10.0)  # safety clamp

        return actions, log_prob

    # === gradient update ===========================================================================

    def _do_update(self):
        """Perform one gradient update using single-step transitions from replay."""
        if self._freeze_encoder_steps > 0:
            if self.total_steps < self._freeze_encoder_steps and not self._encoder_frozen:
                self._set_encoder_requires_grad(False)
                self._encoder_frozen = True
            elif self.total_steps >= self._freeze_encoder_steps and self._encoder_frozen:
                self._set_encoder_requires_grad(True)
                self._encoder_frozen = False

        # Cache actor net resolution to avoid repeated traversals
        actor_net = self._resolve_actor_net()

        batch, info = self.rb.sample(self.cfg.batch_size, return_info=True)
        batch_indices = info.get("index", None)

        # Load raw pixels and compute features using CNN
        pixels = unpack_pixels(self._to_device_fast(batch["pixels"]))
        next_pixels = unpack_pixels(self._to_device_fast(batch["next_pixels"]))
        features = self._extract_cnn_and_compute_features(pixels, actor_net)
        next_features = self._extract_cnn_and_compute_features(next_pixels, actor_net)

        actions_b = self._to_device_fast(batch["action"], dtype=torch.float32)
        rewards_b = self._to_device_fast(batch["reward"], dtype=torch.float32)

        terminated_mask = None
        truncated_mask = None
        if "terminated" in batch.keys():
            terminated_mask = self._to_device_fast(batch["terminated"], dtype=torch.float32)
        if "truncated" in batch.keys():
            truncated_mask = self._to_device_fast(batch["truncated"], dtype=torch.float32)

        if terminated_mask is not None:
            done_mask = terminated_mask
            if (
                bool(getattr(self.cfg, "treat_truncated_as_done", False))
                and truncated_mask is not None
            ):
                done_mask = torch.maximum(done_mask, truncated_mask)
        elif "done" in batch.keys():
            done_mask = self._to_device_fast(batch["done"], dtype=torch.float32)
        else:
            done_mask = torch.zeros_like(rewards_b)

        obs_vector = (
            self._to_device_fast(batch["vector"], dtype=torch.float32)
            if "vector" in batch.keys()
            else None
        )
        next_obs_vector = (
            self._to_device_fast(batch["next_vector"], dtype=torch.float32)
            if "next_vector" in batch.keys()
            else None
        )
        B = features.shape[0]
        rewards_b = self._as_batch_column(rewards_b, B)
        done_mask = self._as_batch_column(done_mask, B)

        alpha = self.log_alpha.exp() if self.log_alpha is not None else self.cfg.alpha

        # ══════════════════════════════════════════════════════════════════
        # CRITIC UPDATE
        # ══════════════════════════════════════════════════════════════════

        with torch.no_grad():
            actor_params = actor_net.forward_features(next_features, vector=next_obs_vector)

            # TODO: add back tanhnormal if doesn't wokr
            # next_dist = TanhNormal(
            #     loc=actor_params["loc"],
            #     scale=actor_params["scale"],
            #     low=torch.tensor([-1.0, 0.0, 0.0], device=self.device),
            #     high=torch.tensor([1.0, 1.0, 1.0], device=self.device),
            # )
            # next_actions = next_dist.rsample()
            # next_log_prob = next_dist.log_prob(next_actions)

            next_actions, next_log_prob = self._sample_squashed(
                actor_params["loc"], actor_params["scale"]
            )
            next_log_prob = self._as_batch_column(next_log_prob, B)

            if next_log_prob.dim() >= 2:
                next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)
            next_log_prob = torch.clamp(next_log_prob, min=-100.0, max=10.0)  # ← ADD
            next_log_prob = self._as_batch_column(next_log_prob, B)

            next_q1 = self.q1_target.module(
                pixels=next_pixels,
                action=next_actions,
                vector=next_obs_vector,
            )
            next_q2 = self.q2_target.module(
                pixels=next_pixels,
                action=next_actions,
                vector=next_obs_vector,
            )
            next_min_q = torch.min(next_q1, next_q2)
            next_min_q = self._as_batch_column(next_min_q, B)

            next_v = next_min_q - alpha * next_log_prob
            q_target = rewards_b + self.cfg.gamma * (1.0 - done_mask) * next_v

            min_q = float(getattr(self.cfg, "min_q_target", -1000.0))
            max_q = float(getattr(self.cfg, "max_q_target", 1000.0))
            q_target = torch.clamp(q_target, min=min_q, max=max_q)

        q1_net = self.q1.module
        q2_net = self.q2.module
        q1_pred = q1_net(pixels=None, action=actions_b, vector=obs_vector, img_features=features)
        q2_pred = q2_net(pixels=None, action=actions_b, vector=obs_vector, img_features=features)
        q1_pred = self._as_batch_column(q1_pred, B)
        q2_pred = self._as_batch_column(q2_pred, B)
        q_target = self._as_batch_column(q_target, B)

        # ── PER importance-sampling weights ────────────────────────────
        is_weights = info.get("_weight", info.get("weight", None))
        if is_weights is not None:
            is_weights = self._to_device_fast(is_weights).view(B, 1).clamp(min=1e-4)
        else:
            is_weights = torch.ones(B, 1, device=self.device)

        q1_elementwise = F.smooth_l1_loss(q1_pred, q_target, reduction="none")
        q2_elementwise = F.smooth_l1_loss(q2_pred, q_target, reduction="none")
        weighted_sum = is_weights.sum().clamp(min=1e-6)
        q1_masked = (q1_elementwise * is_weights).sum() / weighted_sum
        q2_masked = (q2_elementwise * is_weights).sum() / weighted_sum
        critic_loss = q1_masked + q2_masked

        # ── TD error for PER priorities ────────────────────────────────
        use_per = bool(getattr(self.cfg, "use_per", True))

        with torch.no_grad():
            td_error_1 = torch.abs(q1_pred - q_target)
            td_error_2 = torch.abs(q2_pred - q_target)
            td_errors = torch.max(td_error_1, td_error_2).reshape(B, -1)

            if use_per:
                td_errors_for_per = td_errors.max(dim=1).values.cpu()
                if batch_indices is not None:
                    _max_priority = float(getattr(self.cfg, "per_max_priority", 100.0))
                    new_priorities = np.clip(td_errors_for_per.numpy(), 1e-6, _max_priority).astype(
                        np.float32
                    )
                    del td_errors_for_per  # Explicit cleanup
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

        beta = 0.0  # Default value when PER is disabled
        if use_per:
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
        actor_updated_this_step = False
        alpha_updated_this_step = False

        if self._updates_count % self._actor_update_delay == 0:
            actor_params = actor_net.forward_features(features, vector=obs_vector)

            # dist = TanhNormal(
            #     loc=actor_params["loc"],
            #     scale=actor_params["scale"],
            #     low=torch.tensor([-1.0, 0.0, 0.0], device=self.device),
            #     high=torch.tensor([1.0, 1.0, 1.0], device=self.device),
            # )
            # new_actions = dist.rsample()
            # log_prob_new = dist.log_prob(new_actions)

            new_actions, log_prob_new = self._sample_squashed(
                actor_params["loc"], actor_params["scale"]
            )
            log_prob_new = self._as_batch_column(log_prob_new, B)

            if log_prob_new.dim() >= 2:
                log_prob_new = log_prob_new.sum(dim=-1, keepdim=True)
            log_prob_new = torch.clamp(log_prob_new, min=-100.0, max=10.0)  # ← ADD
            log_prob_new = self._as_batch_column(log_prob_new, B)

            for p in list(q1_net.parameters()) + list(q2_net.parameters()):
                p.requires_grad_(False)

            q1_new = q1_net(
                pixels=None,
                action=new_actions,
                vector=obs_vector,
                img_features=features,
            )
            q2_new = q2_net(
                pixels=None,
                action=new_actions,
                vector=obs_vector,
                img_features=features,
            )
            q1_new = self._as_batch_column(q1_new, B)
            q2_new = self._as_batch_column(q2_new, B)

            for p in list(q1_net.parameters()) + list(q2_net.parameters()):
                p.requires_grad_(True)

            min_q_new = torch.min(q1_new, q2_new)

            actor_loss_elementwise = alpha.detach() * log_prob_new - min_q_new
            actor_loss = actor_loss_elementwise.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()
            self._actor_updates_count += 1
            actor_updated_this_step = True

            # ── Alpha update ───────────────────────────────────────────
            alpha_loss = None
            if self.log_alpha is not None and self.alpha_opt is not None:
                with torch.no_grad():
                    entropy_error = log_prob_new + self.target_entropy
                alpha_loss_elementwise = -(self.log_alpha * entropy_error)
                alpha_loss = alpha_loss_elementwise.mean()

                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
                self.alpha_opt.step()
                self._alpha_updates_count += 1
                alpha_updated_this_step = True

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
            current_entropy = (
                -log_prob_new.mean().item() if log_prob_new is not None else float("nan")
            )
            if new_actions is not None:
                new_actions_log = new_actions.reshape(-1, new_actions.shape[-1])
            else:
                new_actions_log = None

            # Compute rich statistics for logging
            log_dict = {
                # ── Loss metrics ──────────────────────────────────────
                "loss/critic_loss": critic_loss.item(),
                "loss/q1_loss": q1_masked.item(),
                "loss/q2_loss": q2_masked.item(),
                # Keep gaps when delayed actor/alpha updates are skipped, instead of fake zeros.
                "loss/actor_loss": (actor_loss.item() if actor_loss is not None else float("nan")),
                "loss/alpha_loss": (alpha_loss.item() if alpha_loss is not None else float("nan")),
                # ── Critic Q-value statistics ─────────────────────────
                "critic/q_target_mean": q_target.mean().item(),
                "critic/q_target_std": q_target.std().item(),
                "critic/q_target_min": q_target.min().item(),
                "critic/q_target_max": q_target.max().item(),
                "critic/q1_pred_mean": q1_pred.mean().item(),
                "critic/q1_pred_std": q1_pred.std().item(),
                "critic/q1_pred_min": q1_pred.min().item(),
                "critic/q1_pred_max": q1_pred.max().item(),
                "critic/q2_pred_mean": q2_pred.mean().item(),
                "critic/q2_pred_std": q2_pred.std().item(),
                "critic/q2_pred_min": q2_pred.min().item(),
                "critic/q2_pred_max": q2_pred.max().item(),
                "critic/td_error_mean": td_errors.mean().item(),
                "critic/td_error_std": td_errors.std().item() if td_errors.numel() > 1 else 0.0,
                "critic/td_error_min": td_errors.min().item(),
                "critic/td_error_max": td_errors.max().item(),
                "critic/q1_explained_variance": q1_explained_var.item(),
                "critic/q2_explained_variance": q2_explained_var.item(),
                # ── Next Q-value and advantage statistics ──────────────
                "critic/next_q_mean": next_min_q.mean().item(),
                "critic/next_q_std": next_min_q.std().item(),
                "critic/next_q_min": next_min_q.min().item(),
                "critic/next_q_max": next_min_q.max().item(),
                "critic/advantage_mean": (
                    (min_q_new.mean() - next_min_q.mean()).item() if min_q_new is not None else 0.0
                ),
                # ── Actor policy statistics ───────────────────────────
                "actor/entropy": current_entropy,
                "actor/entropy_loss_scalar": (
                    (alpha.detach() * current_entropy) if log_prob_new is not None else float("nan")
                ),
                "actor/alpha": alpha.item(),
                "actor/log_alpha": self.log_alpha.item() if self.log_alpha is not None else 0.0,
                # ── Action statistics (per action dim) ────────────────
                "action/steer_mean": (
                    new_actions_log[:, 0].mean().item()
                    if new_actions_log is not None
                    else float("nan")
                ),
                "action/steer_std": (
                    new_actions_log[:, 0].std().item()
                    if new_actions_log is not None
                    else float("nan")
                ),
                "action/accel_mean": (
                    new_actions_log[:, 1].mean().item()
                    if new_actions_log is not None
                    else float("nan")
                ),
                "action/accel_std": (
                    new_actions_log[:, 1].std().item()
                    if new_actions_log is not None
                    else float("nan")
                ),
                # ── Policy parameters (mu, sigma) ────────────────────
                "actor/loc_mean": (
                    actor_params["loc"].mean().item() if "loc" in actor_params else 0.0
                ),
                "actor/loc_std": actor_params["loc"].std().item() if "loc" in actor_params else 0.0,
                "actor/scale_mean": (
                    actor_params["scale"].mean().item() if "scale" in actor_params else 0.0
                ),
                "actor/scale_std": (
                    actor_params["scale"].std().item() if "scale" in actor_params else 0.0
                ),
                # ── Log probability (entropy from policy output) ───────
                "actor/log_prob_mean": (
                    log_prob_new.mean().item() if log_prob_new is not None else float("nan")
                ),
                "actor/log_prob_std": (
                    log_prob_new.std().item() if log_prob_new is not None else float("nan")
                ),
                "actor/log_prob_min": (
                    log_prob_new.min().item() if log_prob_new is not None else float("nan")
                ),
                # ── Reward statistics ─────────────────────────────────
                "reward/batch_mean": rewards_b.mean().item(),
                "reward/batch_std": rewards_b.std().item(),
                "reward/batch_min": rewards_b.min().item(),
                "reward/batch_max": rewards_b.max().item(),
                # ── Sequence masking statistics ───────────────────────
                "batch/size": float(B),
                # ── PER statistics ────────────────────────────────────
                "per/td_error_mean": td_errors.mean().item(),
                "per/beta": beta,
                # ── Update counts ─────────────────────────────────────
                "updates/critic_updates_count": self._critic_updates_count,
                "updates/actor_updates_count": self._actor_updates_count,
                "updates/alpha_updates_count": self._alpha_updates_count,
                "updates/actor_updated_this_log": float(actor_updated_this_step),
                "updates/alpha_updated_this_log": float(alpha_updated_this_step),
                "updates/total_env_steps": self.total_steps,
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

    def _get_grad_norms(self, net) -> dict:
        """Compute gradient statistics for a network."""
        total_norm = 0.0
        norms = []
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2).item()
                norms.append(param_norm)
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        if norms:
            return {
                "grad_norm_mean": sum(norms) / len(norms),
                "grad_norm_max": max(norms),
                "grad_norm_min": min(norms),
                "total_grad_norm": total_norm,
            }
        return {
            "grad_norm_mean": 0.0,
            "grad_norm_max": 0.0,
            "grad_norm_min": 0.0,
            "total_grad_norm": 0.0,
        }
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
        last = list(self.episode_returns)[-100:]  # deque slicing requires conversion
        avg_return = sum(last) / len(last) if last else 0.0

        if self.total_steps - self._last_log_steps >= self.cfg.log_interval:
            print(
                f"Steps: {self.total_steps}, AvgReturn(100): {avg_return:.2f}, "
                f"Buffer: {len(self.rb)}, Eps: {epsilon:.3f}"
            )
            try:
                # Compute episode statistics
                returns_list = list(self.episode_returns)  # Convert deque to list for slicing
                last_1 = returns_list[-1:] if returns_list else [0.0]
                last_10 = returns_list[-10:]
                last_100 = last

                def safe_std(vals):
                    if len(vals) < 2:
                        return 0.0
                    return (sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)) ** 0.5

                stats_dict = {
                    "steps": self.total_steps,
                    # ── Episode return statistics ──────────────────────
                    "episode/return_latest": last_1[0] if last_1 else 0.0,
                    "episode/return_mean_10": sum(last_10) / len(last_10) if last_10 else 0.0,
                    "episode/return_std_10": safe_std(last_10) if last_10 else 0.0,
                    "episode/return_min_10": min(last_10) if last_10 else 0.0,
                    "episode/return_max_10": max(last_10) if last_10 else 0.0,
                    "episode/return_mean_100": avg_return,
                    "episode/return_std_100": safe_std(last_100) if last_100 else 0.0,
                    "episode/return_min_100": min(last_100) if last_100 else 0.0,
                    "episode/return_max_100": max(last_100) if last_100 else 0.0,
                    # ── Buffer statistics ──────────────────────────────
                    "buffer/size": len(self.rb),
                    "buffer/capacity": getattr(self.cfg, "replay_size", 1000000),
                    "buffer/fill_ratio": len(self.rb) / getattr(self.cfg, "replay_size", 1000000),
                    # ── Scheduling ─────────────────────────────────────
                    "exploration/epsilon": epsilon,
                    # ── Async pacing / queue telemetry ────────────────
                    "pacing/drain_transitions_last_iter": float(self._last_drain_count),
                    "pacing/train_batches_last_iter": float(self._last_train_batches),
                    "pacing/effective_max_updates_per_tick": float(
                        self._last_effective_max_updates
                    ),
                    "pacing/update_credit": float(self._last_update_credit),
                    "pacing/drain_max_items": float(self._drain_max_items),
                    "queue/size": float(self._queue_size) if self._queue_size >= 0 else -1.0,
                    "queue/capacity": float(self._queue_capacity),
                    "queue/fill_ratio": (
                        float(self._queue_size) / float(self._queue_capacity)
                        if self._queue_size >= 0 and self._queue_capacity > 0
                        else -1.0
                    ),
                    "queue/fill_ratio_observed": float(self._last_queue_fill_ratio),
                    "collector/enqueue_full_count": float(self._collector_enqueue_full_count),
                    "collector/actor_infer_calls": float(self._collector_actor_infer_calls),
                    # ── Learning rate schedule (if available) ──────────
                    "schedule/actor_lr": self.actor_opt.param_groups[0].get("lr", 0.0),
                    "schedule/critic_lr": self.critic_opt.param_groups[0].get("lr", 0.0),
                    "schedule/alpha_lr": (
                        self.alpha_opt.param_groups[0].get("lr", 0.0) if self.alpha_opt else 0.0
                    ),
                }

                # ── Monitor queue health ──────────────────────────────────
                if self._queue_size >= 0 and self._queue_capacity > 0:
                    queue_usage = float(self._queue_size) / float(self._queue_capacity)
                    if queue_usage > 0.9:
                        log_warning(
                            f"Queue near full: {queue_usage:.1%} - learner may be bottlenecked"
                        )

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
        if (
            rb_save_interval is not None
            and self.total_steps - self._last_save_rb_steps >= rb_save_interval
        ):
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
                    try:
                        if hasattr(self.rb._storage, "_storage") and isinstance(
                            self.rb._storage._storage, (list, tuple)
                        ):
                            buffer_clones = [
                                t.clone() if isinstance(t, torch.Tensor) else t
                                for t in self.rb._storage._storage
                            ]
                        else:
                            buffer_clones = self.rb._storage._storage

                        rb_state = {
                            "buffer": buffer_clones,
                            "sampler_state": {
                                "alpha": getattr(self.rb._sampler, "_alpha", None),
                                "beta": getattr(self.rb._sampler, "_beta", None),
                            },
                            "total_steps": self.total_steps,
                            "buffer_size": len(self.rb),
                        }
                        torch.save(rb_state, rb_path)
                    except (RuntimeError, ValueError, TypeError) as e:
                        # If save fails due to closed files, log warning but continue training
                        if "closed file" in str(e).lower():
                            log_warning(f"Cannot save replay buffer (storage file closed): {e}")
                        else:
                            # Try pickle as fallback
                            try:
                                with open(rb_path, "wb") as f:
                                    pickle.dump(rb_state, f)
                            except Exception as pickle_err:
                                log_warning(
                                    f"Both torch.save and pickle failed for replay buffer: {pickle_err}"
                                )
                        return

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
