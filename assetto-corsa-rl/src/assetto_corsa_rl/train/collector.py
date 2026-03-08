import torch
from tensordict import TensorDict

from .train_utils import (
    OrnsteinUhlenbeckNoise,
    expand_actions_for_envs,
    extract_reward_and_done,
    get_inner,
    pack_pixels,
    sample_random_action,
)
from .logging_utils import log_info


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

        # Track whether we've logged the start of random exploration
        self._start_steps_logged = False
        self._end_start_steps_logged = False
        start_steps = int(getattr(self.cfg, "start_steps", 0))
        if start_steps > 0:
            log_info(f"[COLLECTOR] Beginning random exploration phase: {start_steps:,} steps")

        # ── Ornstein-Uhlenbeck noise for coherent exploration ──────────
        action_dim = int(env.action_spec.shape[-1])
        ou_theta = getattr(cfg, "ou_theta", 0.15)
        ou_sigma = getattr(cfg, "ou_sigma", 0.3)
        # Slight forward-driving bias: [steer=0, gas=0.4, brake=0]
        ou_mu_cfg = getattr(cfg, "ou_mu", None)
        if ou_mu_cfg is None:
            ou_mu_default = torch.zeros(action_dim, device=self.device)
            if action_dim >= 2:
                ou_mu_default[1] = 0.4
        else:
            ou_mu_default = torch.tensor(ou_mu_cfg, dtype=torch.float32, device=self.device)
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

    # ── async entry-point (multi-process) ─────────────────────────────────

    def run(self):
        """Run until ``stop_event`` is set (multi-process usage)."""
        sync_every = int(getattr(self.cfg, "sync_every", 100))
        while self.stop_event is None or not self.stop_event.is_set():
            self._step_and_store()
            if self.weights_version is not None and self.total_steps % sync_every == 0:
                self._sync_weights()
            # Periodically broadcast epsilon so the learner can log it.
            if self.total_steps % sync_every == 0:
                eps = self._exploration_epsilon()
                try:
                    self.transitions_queue.put_nowait({"_meta": True, "epsilon": eps})
                except Exception:
                    pass

    # ── helpers ───────────────────────────────────────────────────────────

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
        frac = min(1.0, float(self.total_steps) / float(steps))
        return float(start + (end - start) * frac)

    # ── core step ─────────────────────────────────────────────────────────

    def _step_and_store(self):
        """Take one step per env, push transitions (and optional meta msgs) to the queue."""
        target_batch = self.current_td.batch_size
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

            if use_noisy:
                for m in self.actor.modules():
                    if hasattr(m, "sample_noise"):
                        m.sample_noise()

            actor_output = self.actor(actor_input)
            has_action = (
                "action" in actor_output.keys()
                and actor_output["action"].shape[-1] == self.env.action_spec.shape[-1]
            )
            actor_action = actor_output["action"] if has_action else None

            if use_noisy:
                eps = 0.0
                if actor_action is None:
                    for m in self.actor.modules():
                        if hasattr(m, "sample_noise"):
                            m.sample_noise()
                    actor_output = self.actor(actor_input)
                    has_action = (
                        "action" in actor_output.keys()
                        and actor_output["action"].shape[-1] == self.env.action_spec.shape[-1]
                    )
                    actor_action = actor_output["action"] if has_action else None
            else:
                eps = self._exploration_epsilon()

            start_steps = int(getattr(self.cfg, "start_steps", 0))
            in_random_phase = self.total_steps < start_steps

            if in_random_phase:
                # OU noise for coherent random exploration (car actually drives)
                ou_sample = self.ou_noise.sample()
                # Clip to action bounds: steer [-1,1], gas [0,1], brake [0,1]
                actions = ou_sample.clone()
                actions[:, 0] = actions[:, 0].clamp(-1.0, 1.0)
                if actions.shape[-1] >= 2:
                    actions[:, 1:] = actions[:, 1:].clamp(0.0, 1.0)
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

            # Additive OU noise on policy actions (decaying) after random phase
            if not in_random_phase and actor_action is not None:
                decay_frac = min(
                    1.0, float(self.total_steps - start_steps) / max(1, self._ou_noise_decay_steps)
                )
                noise_scale = self._ou_noise_scale * (1.0 - decay_frac)
                if noise_scale > 1e-6:
                    ou_delta = self.ou_noise.sample() - self.ou_noise.mu[0]
                    actions = actions + noise_scale * ou_delta
                    actions[:, 0] = actions[:, 0].clamp(-1.0, 1.0)
                    if actions.shape[-1] >= 2:
                        actions[:, 1:] = actions[:, 1:].clamp(0.0, 1.0)

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

        # Extract vector obs from current and next observations
        cur_vector = inner_obs.get("vector", None)
        next_vector = td_next.get("vector", None)

        for i in range(self.cfg.num_envs):
            transition = {
                "pixels": pack_pixels(pixels[i]),
                "action": actions[i].to(torch.float32).cpu(),
                "reward": rewards[i].unsqueeze(0).cpu(),
                "next_pixels": pack_pixels(next_pixels[i]),
                "done": dones[i].unsqueeze(0).cpu(),
            }
            if cur_vector is not None:
                v = cur_vector[i] if cur_vector.dim() > 1 else cur_vector
                transition["vector"] = v.to(torch.float32).cpu()
            if next_vector is not None:
                nv = next_vector[i] if next_vector.dim() > 1 else next_vector
                transition["next_vector"] = nv.to(torch.float32).cpu()
            try:
                self.transitions_queue.put_nowait(transition)
            except Exception:
                pass  # back-pressure: drop transition if queue is full

        self._handle_episode_end(rewards, dones)
        self._maybe_reset(td_next, dones)
        self.total_steps += self.cfg.num_envs

        # Log when exiting start_steps phase
        start_steps = int(getattr(self.cfg, "start_steps", 0))
        if not self._end_start_steps_logged and self.total_steps >= start_steps > 0:
            log_info(f"[COLLECTOR] Random exploration phase complete! ({self.total_steps:,} steps)")
            log_info(f"[COLLECTOR] Beginning policy learning...")
            self._end_start_steps_logged = True

    def _handle_episode_end(self, rewards, dones):
        rewards = rewards.to(self.current_episode_return.device)
        dones = dones.to(self.current_episode_return.device)
        self.current_episode_return += rewards
        # Reset OU noise on episode boundaries for fresh trajectories
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            self.ou_noise.reset(env_indices=done_indices)
        for i, d in enumerate(dones):
            if d.item():
                ep_ret = float(self.current_episode_return[i].item())
                self.current_episode_return[i] = 0.0
                try:
                    self.transitions_queue.put_nowait({"_meta": True, "episode_return": ep_ret})
                except Exception:
                    pass

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

    # ── weight sync (multi-process) ───────────────────────────────────────

    def _sync_weights(self):
        if self.weights_version is None or self.weights_version.value == self._local_version:
            return
        with self.weights_lock:
            self.actor.load_state_dict(self.shared_weights)
            self._local_version = self.weights_version.value
