import copy
import queue
import time
import types

import torch.multiprocessing as mp
import torch

from .collector import CollectorWorker
from .learner import LearnerWorker
from .logging_utils import log_info, log_success, log_warning, log_error


# ── subprocess entry-points ────────────────────────────────────────────────────


def _collector_process_fn(
    cfg,
    env_kwargs,
    actor,
    transitions_queue,
    stop_event,
    device,
    shared_weights,
    weights_lock,
    weights_version,
    initial_td,
):
    """Entry-point for the collector subprocess."""
    from assetto_corsa_rl.ac_env import create_transformed_env  # type: ignore

    env = create_transformed_env(**env_kwargs)
    actor = actor.to(device)
    worker = CollectorWorker(
        cfg=cfg,
        env=env,
        actor=actor,
        transitions_queue=transitions_queue,
        stop_event=stop_event,
        device=device,
        shared_weights=shared_weights,
        weights_lock=weights_lock,
        weights_version=weights_version,
    )
    if initial_td is not None:
        worker.current_td = initial_td
    worker.run()


def _learner_process_fn(
    cfg,
    rb_kwargs,
    actor,
    q1,
    q2,
    q1_target,
    q2_target,
    actor_lr,
    critic_lr,
    log_alpha_value,
    target_entropy,
    action_dim,
    transitions_queue,
    device,
    shared_weights,
    weights_lock,
    weights_version,
    log_queue,
    stop_event,
    total_steps,
    episode_returns,
    start_time,
):
    """Entry-point for the learner subprocess."""
    from torchrl.data.replay_buffers import PrioritizedReplayBuffer, LazyTensorStorage

    storage = LazyTensorStorage(max_size=rb_kwargs["max_size"])
    rb = PrioritizedReplayBuffer(
        alpha=rb_kwargs["alpha"],
        beta=rb_kwargs["beta"],
        storage=storage,
        batch_size=rb_kwargs["batch_size"],
    )

    log_alpha = torch.nn.Parameter(
        torch.tensor(log_alpha_value, dtype=torch.float32, device=device)
    )
    alpha_lr = float(getattr(cfg, "alpha_lr", 3e-4))
    alpha_opt = torch.optim.Adam([log_alpha], lr=alpha_lr)

    actor = actor.to(device)
    q1 = q1.to(device)
    q2 = q2.to(device)
    q1_target = q1_target.to(device)
    q2_target = q2_target.to(device)

    # Recreate optimizers with the deep-copied parameters that live in THIS
    # process.  The old optimizers from the main process referenced the
    # *original* (pre-deepcopy) parameters and would update dead tensors.
    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=critic_lr)

    _action_spec = types.SimpleNamespace(shape=(action_dim,))
    env = types.SimpleNamespace(action_spec=_action_spec)
    worker = LearnerWorker(
        cfg=cfg,
        rb=rb,
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        log_alpha=log_alpha,
        alpha_opt=alpha_opt,
        target_entropy=target_entropy,
        transitions_queue=transitions_queue,
        env=env,
        device=device,
        storage=storage,  # freshly created in this process
        shared_weights=shared_weights,
        weights_lock=weights_lock,
        weights_version=weights_version,
        log_queue=log_queue,
        stop_event=stop_event,
    )
    worker.total_steps = total_steps
    if episode_returns:
        worker.episode_returns = list(episode_returns)
    if start_time is not None:
        worker.start_time = start_time
    worker.run()


# ── Trainer ────────────────────────────────────────────────────────────────────


class Trainer:
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
        env_kwargs=None,
    ):
        self.env = env
        self.rb = rb
        self.cfg = cfg
        self.device = device
        self.storage = storage
        self.env_kwargs = env_kwargs or {}
        self.total_steps = 0
        self.start_time = time.time()

        self._queue: queue.Queue = queue.Queue()

        self.collector = CollectorWorker(
            cfg=cfg,
            env=env,
            actor=actor,
            transitions_queue=self._queue,
            device=device,
        )
        if td is not None:
            self.collector.current_td = td

        self.learner = LearnerWorker(
            cfg=cfg,
            rb=rb,
            actor=actor,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            log_alpha=log_alpha,
            alpha_opt=alpha_opt,
            target_entropy=target_entropy,
            transitions_queue=self._queue,
            env=env,
            device=device,
            storage=storage,
        )

    # ── backward-compat properties ─────────────────────────────────────────

    @property
    def episode_returns(self):
        return self.learner.episode_returns

    @episode_returns.setter
    def episode_returns(self, v):
        self.learner.episode_returns = v

    @property
    def current_episode_return(self):
        return self.collector.current_episode_return

    @current_episode_return.setter
    def current_episode_return(self, v):
        self.collector.current_episode_return = v

    # ── synchronous single-process training loop ───────────────────────────

    def run(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.collector.total_steps = total_steps
        self.learner.total_steps = total_steps
        self.learner.start_time = self.start_time

        while self.total_steps < self.cfg.total_steps:
            for _ in range(self.cfg.frames_per_batch):
                self.collector._step_and_store()
                self.total_steps = self.collector.total_steps
                self.learner.total_steps = self.total_steps

                self.learner._drain_transitions()

                if len(self.rb) >= self.cfg.batch_size:
                    for _ in range(self.learner._updates_per_step):
                        self.learner._do_update()

            self.learner._maybe_log_and_save(epsilon=self.collector._exploration_epsilon())

        print("Training finished")

    # ── async multiprocess training loop ──────────────────────────────────

    def run_async(self, total_steps: int = 0):
        """Spawn collector and learner as separate processes using
        torch.multiprocessing for asynchronous, decoupled training."""

        actor_state = {
            k: v.cpu().clone().share_memory_() for k, v in self.learner.actor.state_dict().items()
        }

        weights_lock = mp.Lock()
        weights_version = mp.Value("i", 0)

        queue_size = int(getattr(self.cfg, "queue_size", 65536))
        transitions_queue = mp.Queue(maxsize=queue_size)
        log_queue = mp.Queue(maxsize=10_000)

        stop_event = mp.Event()

        actor_for_learner = copy.deepcopy(self.learner.actor).cpu()
        actor_for_collector = copy.deepcopy(self.learner.actor).cpu()
        q1_mp = copy.deepcopy(self.learner.q1).cpu()
        q2_mp = copy.deepcopy(self.learner.q2).cpu()
        q1_target_mp = copy.deepcopy(self.learner.q1_target).cpu()
        q2_target_mp = copy.deepcopy(self.learner.q2_target).cpu()

        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass

        collector_proc = mp.Process(
            target=_collector_process_fn,
            kwargs=dict(
                cfg=self.cfg,
                env_kwargs=self.env_kwargs,
                actor=actor_for_collector,
                transitions_queue=transitions_queue,
                stop_event=stop_event,
                device=self.device,
                shared_weights=actor_state,
                weights_lock=weights_lock,
                weights_version=weights_version,
                initial_td=None,  # subprocess creates its own env and resets
            ),
            daemon=True,
            name="CollectorWorker",
        )

        rb_kwargs = dict(
            max_size=int(self.cfg.replay_size),
            alpha=float(self.cfg.per_alpha),
            beta=float(self.cfg.per_beta),
            batch_size=int(self.cfg.batch_size),
        )

        learner_proc = mp.Process(
            target=_learner_process_fn,
            kwargs=dict(
                cfg=self.cfg,
                rb_kwargs=rb_kwargs,
                actor=actor_for_learner,
                q1=q1_mp,
                q2=q2_mp,
                q1_target=q1_target_mp,
                q2_target=q2_target_mp,
                actor_lr=float(getattr(self.cfg, "actor_lr", getattr(self.cfg, "lr", 3e-4))),
                critic_lr=float(getattr(self.cfg, "critic_lr", getattr(self.cfg, "lr", 3e-4))),
                log_alpha_value=float(self.learner.log_alpha.data.cpu().item()),
                target_entropy=self.learner.target_entropy,
                action_dim=int(self.env.action_spec.shape[-1]),
                transitions_queue=transitions_queue,
                device=self.device,
                shared_weights=actor_state,
                weights_lock=weights_lock,
                weights_version=weights_version,
                log_queue=log_queue,
                stop_event=stop_event,
                total_steps=total_steps,
                episode_returns=list(self.learner.episode_returns),
                start_time=self.start_time,
            ),
            daemon=True,
            name="LearnerWorker",
        )

        log_info("Starting CollectorWorker process...")
        collector_proc.start()
        log_info("Starting LearnerWorker process...")
        learner_proc.start()

        # Main process: drain log_queue and forward to wandb so the learner
        # subprocess never has to call wandb.log() directly.
        try:
            import wandb

            while learner_proc.is_alive():
                _flush_log_queue(log_queue)
                time.sleep(0.05)
            # final drain after learner exits
            _flush_log_queue(log_queue)
        except KeyboardInterrupt:
            log_warning("KeyboardInterrupt — stopping workers...")
        finally:
            stop_event.set()
            collector_proc.join(timeout=10)
            learner_proc.join(timeout=30)
            if collector_proc.is_alive():
                log_warning("collector did not exit cleanly, terminating.")
                collector_proc.terminate()
            if learner_proc.is_alive():
                log_warning("learner did not exit cleanly, terminating.")
                learner_proc.terminate()
            log_success("All worker processes stopped.")


def _flush_log_queue(log_queue: mp.Queue) -> None:
    """Drain all pending log items from *log_queue* and forward to wandb."""
    import wandb

    while True:
        try:
            item = log_queue.get_nowait()
        except Exception:
            break
        if not isinstance(item, dict):
            continue
        step = item.pop("step", None)
        data = item.pop("data", item)  # support both {step, data} and flat dicts
        try:
            wandb.log(data, step=step)
        except Exception as e:
            print(f"Warning: wandb.log failed: {e}")


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
    env_kwargs=None,
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
        env_kwargs=env_kwargs,
    )
    if episode_returns is not None:
        t.episode_returns = episode_returns
    if current_episode_return is not None:
        t.current_episode_return = current_episode_return
    if start_time is not None:
        t.start_time = start_time

    use_async = bool(getattr(cfg, "use_async", False))
    if use_async:
        mp.set_start_method("spawn", force=True)
        t.run_async(total_steps=total_steps)
    else:
        t.run(total_steps=total_steps)
