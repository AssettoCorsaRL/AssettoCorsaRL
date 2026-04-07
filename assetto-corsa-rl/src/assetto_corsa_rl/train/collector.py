import torch
import queue
import types
from tensordict import TensorDict

from .train_utils import (
    OrnsteinUhlenbeckNoise,
    expand_actions_for_envs,
    extract_reward_and_done,
    fix_action_shape,
    get_inner,
    pack_pixels,
    sample_random_action,
)
from .logging_utils import log_info
from .cpu_inference_utils import enable_cpu_optimizations
from ..model.sac_inference import SACInferenceEngine


class CollectorWorker:
    def __init__(
        self,
        cfg,
        env,
        actor,
        transitions_queue,
        stop_event=None,
        device=None,
        # optional shared-memory weight sync (multi-process only)
        shared_weights=None,
        weights_lock=None,
        weights_version=None,
    ):
        self.cfg = cfg
        self.env = env
        self.actor = actor
        self.transitions_queue = transitions_queue
        self.stop_event = stop_event
        self.device = device if device is not None else torch.device("cpu")
        self.shared_weights = shared_weights
        self.weights_lock = weights_lock
        self.weights_version = weights_version

        self.total_steps = 0
        self._local_version = -1
        self.current_td = self.env.reset()
        self.current_episode_return = torch.zeros(cfg.num_envs, device=self.device)
        self._queue_capacity = int(getattr(cfg, "queue_size", 0))
        self._queue_batch_size = max(1, int(getattr(cfg, "queue_batch_size", 64)))
        self._queue_batch_max_delay_steps = max(
            1, int(getattr(cfg, "queue_batch_max_delay_steps", 4))
        )
        self._pending_transitions = []
        self._steps_since_batch_flush = 0
        self._enqueue_full_count = 0
        self._actor_infer_calls = 0
        self._warned_action_shape_mismatch = False

        # self._weight_sync_thread = None
        # self._stop_weight_sync = threading.Event()

        # if self.weights_version is not None:
        #     self._weight_sync_thread = threading.Thread(target=self._weight_sync_loop, daemon=True)
        #     self._weight_sync_thread.start()

        self._start_steps_logged = False
        self._end_start_steps_logged = False
        start_steps = int(getattr(self.cfg, "start_steps", 0))
        if start_steps > 0:
            log_info(f"[COLLECTOR] Beginning random exploration phase: {start_steps:,} steps")

        action_dim = int(env.action_spec.shape[-1])
        ou_theta = getattr(cfg, "ou_theta", 0.15)
        ou_sigma = getattr(cfg, "ou_sigma", 0.3)
        ou_mu_cfg = getattr(cfg, "ou_mu", None)
        if ou_mu_cfg is None:
            ou_mu_default = torch.zeros(action_dim, device=self.device)
            if action_dim >= 2:
                ou_mu_default[1] = 0.4
        else:
            ou_mu_default = torch.tensor(ou_mu_cfg, dtype=torch.float32, device=self.device)
            if ou_mu_default.shape[0] > action_dim:
                ou_mu_default = ou_mu_default[:action_dim]

        self.ou_noise = OrnsteinUhlenbeckNoise(
            action_dim=action_dim,
            num_envs=cfg.num_envs,
            theta=ou_theta,
            sigma=ou_sigma,
            mu=ou_mu_default,
            device=self.device,
        )
        self._ou_noise_decay_steps = int(getattr(cfg, "ou_noise_decay_steps", 100_000))
        self._ou_noise_scale = float(getattr(cfg, "ou_noise_scale", 0.3))

        self._cpu_inference_engine = None

        self.actor.eval()

        if str(self.device) == "cpu":
            cpu_threads = int(getattr(cfg, "cpu_num_threads", 4))

            enable_cpu_optimizations(num_threads=cpu_threads)

            log_info(f"[COLLECTOR] CPU inference optimized: {cpu_threads} threads")

            use_noisy = bool(getattr(self.cfg, "use_noisy", False))
            if int(getattr(self.cfg, "num_envs", 1)) == 1 and not use_noisy:
                try:
                    backend = str(getattr(self.cfg, "cpu_inference_backend", "compile"))
                    policy_like = types.SimpleNamespace(actor=self.actor)
                    self._cpu_inference_engine = SACInferenceEngine(
                        policy_like,
                        backend=backend,
                        quantize=bool(getattr(self.cfg, "quantize_actor", False)),
                        channels_last=True,
                        num_threads=cpu_threads,
                        warmup=16,
                        benchmark_n=32,
                        deterministic=False,
                        copy=False,
                    )
                    log_info(
                        f"[COLLECTOR] SACInferenceEngine enabled for CPU actor inference (backend={backend})"
                    )
                except Exception as e:
                    self._cpu_inference_engine = None
                    log_info(
                        f"[COLLECTOR] SACInferenceEngine init failed, falling back to actor forward: {e}"
                    )
            elif use_noisy:
                log_info("[COLLECTOR] SACInferenceEngine disabled when use_noisy=true")
            else:
                log_info("[COLLECTOR] SACInferenceEngine disabled when num_envs>1")
        else:
            self._cpu_inference_engine = None

        self._reset_actor_context()

    def _reset_actor_context(self):
        for m in self.actor.modules():
            if hasattr(m, "reset_context"):
                m.reset_context()

    def stop(self):
        """Stop the weight synchronization thread."""
        # self._stop_weight_sync.set()
        # if self._weight_sync_thread is not None:
        #     self._weight_sync_thread.join()

        pass

    #! Async entry point

    def run(self):
        """Run until ``stop_event`` is set (multi-process usage)."""
        sync_every = int(getattr(self.cfg, "sync_every", 100))

        while self.stop_event is None or not self.stop_event.is_set():
            if self.total_steps % sync_every == 0:
                self._sync_weights()

            self._step_and_store()
            # broadcast epsilon so the learner can log it.
            if self.total_steps % sync_every == 0:
                self._flush_pending_transitions(force=True)
                eps = self._exploration_epsilon()
                queue_size = None
                try:
                    queue_size = int(self.transitions_queue.qsize())
                except Exception:
                    queue_size = None

                self._enqueue(
                    {
                        "_meta": True,
                        "epsilon": eps,
                        "queue_size": queue_size,
                        "queue_capacity": self._queue_capacity,
                        "collector/enqueue_full_count": self._enqueue_full_count,
                        "collector/actor_infer_calls": self._actor_infer_calls,
                    }
                )

                self._enqueue_full_count = 0
                self._actor_infer_calls = 0

        self._flush_pending_transitions(force=True)

    def _exploration_epsilon(self):
        """Linearly anneal epsilon from explore_start → explore_end over explore_steps.

        During start_steps, force epsilon=1.0 (purely random exploration).
        """
        start_steps = int(getattr(self.cfg, "start_steps", 0))
        if self.total_steps < start_steps:
            return 1.0

        if getattr(self.cfg, "use_noisy", False):
            return 0.0
        start = float(getattr(self.cfg, "explore_start", 1.0))
        end = float(getattr(self.cfg, "explore_end", 0.0))
        steps = int(getattr(self.cfg, "explore_steps", 100_000))
        if steps <= 0:
            return float(end)
        post_warmup_steps = max(0, self.total_steps - start_steps)
        frac = min(1.0, float(post_warmup_steps) / float(steps))
        return float(start + (end - start) * frac)

    def _step_and_store(self):
        """Take one step per env, accumulate into sequences, push complete sequences."""
        target_batch = self.current_td.batch_size
        action_dim = int(self.env.action_spec.shape[-1])
        start_steps = int(getattr(self.cfg, "start_steps", 0))
        in_random_phase = self.total_steps < start_steps

        with torch.no_grad():
            inner_obs = get_inner(self.current_td)
            pixels_only = inner_obs["pixels"]
            if pixels_only.dim() == 3:
                pixels_only = pixels_only.unsqueeze(0)
            vector_obs = inner_obs.get("vector", None)

            actor_input_data = {"pixels": pixels_only}
            if vector_obs is not None:
                if vector_obs.dim() == 1:
                    vector_obs = vector_obs.unsqueeze(0)
                actor_input_data["vector"] = vector_obs
            actor_input = TensorDict(actor_input_data, batch_size=[pixels_only.shape[0]])
            use_noisy = getattr(self.cfg, "use_noisy", False)

            actor_action = None
            if not in_random_phase:
                if use_noisy:
                    for m in self.actor.modules():
                        if hasattr(m, "sample_noise"):
                            m.sample_noise()

                if self._cpu_inference_engine is not None:
                    try:
                        vector_for_engine = None
                        if vector_obs is not None:
                            vector_for_engine = (
                                vector_obs[0] if vector_obs.dim() > 1 else vector_obs
                            )
                        action_1 = self._cpu_inference_engine.get_action(
                            pixels_only[0],
                            vector=vector_for_engine,
                        )
                        self._actor_infer_calls += 1
                        actor_action = fix_action_shape(
                            action_1.unsqueeze(0), batch_size=1, action_dim=action_dim
                        )
                    except Exception:
                        actor_action = None

                if actor_action is None:
                    actor_output = self.actor(actor_input)
                    self._actor_infer_calls += 1
                    if "action" in actor_output.keys():
                        raw_action = actor_output["action"]
                        if raw_action.ndim == 1:
                            raw_action = raw_action.unsqueeze(0)
                        if (
                            raw_action.shape[-1] != action_dim
                            and not self._warned_action_shape_mismatch
                        ):
                            log_info(
                                "[COLLECTOR] Actor action dim mismatch: got %s expected %s; applying fix_action_shape"
                                % (raw_action.shape[-1], action_dim)
                            )
                            self._warned_action_shape_mismatch = True
                        actor_action = fix_action_shape(
                            raw_action, batch_size=raw_action.shape[0], action_dim=action_dim
                        )
                    else:
                        actor_action = None

                if use_noisy and actor_action is None:
                    for m in self.actor.modules():
                        if hasattr(m, "sample_noise"):
                            m.sample_noise()
                    actor_output = self.actor(actor_input)
                    self._actor_infer_calls += 1
                    if "action" in actor_output.keys():
                        raw_action = actor_output["action"]
                        if raw_action.ndim == 1:
                            raw_action = raw_action.unsqueeze(0)
                        actor_action = fix_action_shape(
                            raw_action, batch_size=raw_action.shape[0], action_dim=action_dim
                        )
                    else:
                        actor_action = None

            if use_noisy:
                eps = 0.0
            else:
                eps = self._exploration_epsilon()

            if in_random_phase:
                ou_sample = self.ou_noise.sample()
                actions = ou_sample.clone()
                actions = actions.clamp(-1.0, 1.0)
            elif eps > 0.0:
                mask = torch.rand(self.cfg.num_envs, device=self.device) < eps
                rand_actions = sample_random_action(self.cfg.num_envs, dev=self.device)
                if actor_action is None:
                    actions = rand_actions
                else:
                    actions = torch.where(
                        mask.view(-1, 1), rand_actions.to(actor_action.device), actor_action
                    )
            else:
                actions = (
                    actor_action
                    if actor_action is not None
                    else sample_random_action(self.cfg.num_envs, dev=self.device)
                )

            if not in_random_phase:
                decay_frac = min(
                    1.0, float(self.total_steps - start_steps) / max(1, self._ou_noise_decay_steps)
                )
                noise_scale = self._ou_noise_scale * (1.0 - decay_frac)
                if noise_scale > 1e-6:
                    # Smoothly interpolate from OU-driven exploration to policy-driven control.
                    # At noise_scale=1.0 this matches pure OU (same behavior as start_steps phase).
                    ou_sample = self.ou_noise.sample()
                    actions = (1.0 - noise_scale) * actions.to(
                        ou_sample.device
                    ) + noise_scale * ou_sample
                    actions = actions.clamp(-1.0, 1.0)

        actions_step = expand_actions_for_envs(actions, target_batch)
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)
        next_td = self.env.step(action_td)
        td_next = get_inner(next_td)

        rewards, dones = extract_reward_and_done(td_next, self.cfg.num_envs, self.device)
        terminated = (
            td_next["terminated"].view(self.cfg.num_envs).to(self.device).to(torch.bool)
            if "terminated" in td_next.keys()
            else torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.device)
        )
        truncated = (
            td_next["truncated"].view(self.cfg.num_envs).to(self.device).to(torch.bool)
            if "truncated" in td_next.keys()
            else torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.device)
        )

        pixels = self.current_td["pixels"]
        next_pixels = td_next["pixels"]
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        if next_pixels.ndim == 3:
            next_pixels = next_pixels.unsqueeze(0)

        pixels = pixels.cpu()
        next_pixels = next_pixels.cpu()

        cur_vector = inner_obs.get("vector", None)
        next_vector = td_next.get("vector", None)
        transitions = []

        for i in range(self.cfg.num_envs):
            transition = {
                "pixels": pack_pixels(pixels[i]),  # uint8
                "next_pixels": pack_pixels(next_pixels[i]),  # uint8
                "action": actions[i].cpu().to(torch.float16),
                "reward": rewards[i].unsqueeze(0).cpu().to(torch.float16),
                "done": dones[i].unsqueeze(0).cpu(),  # bool is fine
                "terminated": terminated[i].unsqueeze(0).cpu(),
                "truncated": truncated[i].unsqueeze(0).cpu(),
            }
            if cur_vector is not None:
                v = cur_vector[i] if cur_vector.dim() > 1 else cur_vector
                transition["vector"] = v.cpu().float()
            if next_vector is not None:
                nv = next_vector[i] if next_vector.dim() > 1 else next_vector
                transition["next_vector"] = nv.cpu().float()

            transitions.append(transition)

        self._enqueue_transitions(transitions)

        self._handle_episode_end(rewards, dones)
        self._maybe_reset(td_next, dones)
        self.total_steps += self.cfg.num_envs

    def _handle_episode_end(self, rewards, dones):
        rewards = rewards.to(self.current_episode_return.device)
        dones = dones.to(self.current_episode_return.device)
        self.current_episode_return += rewards

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            self.ou_noise.reset(env_indices=done_indices)

        for i, d in enumerate(dones):
            if d.item():
                ep_ret = float(self.current_episode_return[i].item())
                self.current_episode_return[i] = 0.0
                self._enqueue({"_meta": True, "episode_return": ep_ret})

    def _maybe_reset(self, td_next, dones):
        self.current_td = td_next
        if "next" in td_next.keys() and "pixels" in td_next["next"].keys():
            self.current_td = td_next["next"]
        if dones.any():
            self._reset_actor_context()
            if self._cpu_inference_engine is not None:
                self._cpu_inference_engine.reset()
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

    def _enqueue(self, item):
        try:
            # Non-blocking put with immediate fallback for queue backpressure
            self.transitions_queue.put(item, timeout=0.001)
        except queue.Full:
            self._enqueue_full_count += 1
            # Drop oldest item to make room (backpressure handling)
            try:
                self.transitions_queue.get_nowait()
                self.transitions_queue.put_nowait(item)
            except queue.Empty:
                pass

    def _enqueue_transitions(self, transitions):
        if not transitions:
            return

        self._pending_transitions.extend(transitions)
        self._steps_since_batch_flush += 1

        should_flush = (
            len(self._pending_transitions) >= self._queue_batch_size
            or self._steps_since_batch_flush >= self._queue_batch_max_delay_steps
        )
        if should_flush:
            self._flush_pending_transitions(force=False)

    def _flush_pending_transitions(self, force: bool = False):
        if not self._pending_transitions:
            return

        if not force and len(self._pending_transitions) < self._queue_batch_size:
            return

        while self._pending_transitions:
            chunk = self._pending_transitions[: self._queue_batch_size]
            del self._pending_transitions[: self._queue_batch_size]
            self._enqueue({"_batch": True, "transitions": chunk})
            if not force:
                break

        if not self._pending_transitions:
            self._steps_since_batch_flush = 0

    def _sync_weights(self):
        if self.weights_version is None or self.weights_version.value == self._local_version:
            return
        with self.weights_lock:
            self.actor.load_state_dict(self.shared_weights, strict=False)
            self._local_version = self.weights_version.value
