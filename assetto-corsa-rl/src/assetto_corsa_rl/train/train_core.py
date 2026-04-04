import copy
import queue
import time
import types
import subprocess

import torch.multiprocessing as mp
import torch
from tensordict import TensorDict

from .collector import CollectorWorker
from .learner import LearnerWorker
from .logging_utils import log_info, log_success, log_warning, log_error


# ── Periodic AC reset utilities ────────────────────────────────────────────────


def _is_ac_running() -> bool:
    """Check if Assetto Corsa is currently running."""
    try:
        proc_list = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq acs.exe"], capture_output=True, text=True, timeout=5
        )
        return "acs.exe" in (proc_list.stdout or "").lower()
    except Exception:
        return False


def maybe_reset_ac(
    last_reset_step: int,
    current_step: int,
    reset_interval_steps: int,
    ac_exe_path: str = r"D:\Steam\steamapps\common\assettocorsa\acs.exe",
    startup_wait: float = 20.0,
) -> int:
    """Periodically kill and restart Assetto Corsa to prevent memory leaks/crashes.

    Args:
        last_reset_step: Step counter of the last reset (or 0 if never reset)
        current_step: Current step counter
        reset_interval_steps: Steps between resets (pass 0 to disable)
        ac_exe_path: Path to acs.exe executable
        startup_wait: Seconds to wait after relaunching AC

    Returns:
        Updated last_reset_step (current_step if reset occurred, otherwise last_reset_step)
    """
    if reset_interval_steps <= 0:
        return last_reset_step  # Disabled

    if current_step - last_reset_step < reset_interval_steps:
        return last_reset_step  # Not time yet

    log_warning(f"[AC Reset] Triggering periodic restart at step {current_step}")

    try:
        from .train_utils import kill_all_ac_instances, activate_ac_window
    except ImportError:
        log_warning("[AC Reset] Could not import AC control utilities")
        return last_reset_step

    # Kill all existing instances
    log_info("[AC Reset] Killing all Assetto Corsa instances...")
    kill_all_ac_instances(max_retries=3, retry_delay=1.0)

    # Relaunch
    log_info(f"[AC Reset] Relaunching from {ac_exe_path}")
    try:
        subprocess.Popen(
            [ac_exe_path],
            cwd=str(__import__("pathlib").Path(ac_exe_path).parent),
        )
        time.sleep(max(0.0, float(startup_wait)))
        activate_ac_window()
        log_success("[AC Reset] Assetto Corsa restarted successfully")
    except Exception as e:
        log_warning(f"[AC Reset] Error during restart: {e}")

    return current_step


# ── Collation function for transition batches ──────────────────────────────────


def _collate_sequence_batch(batch):
    """Collate a list of transition TensorDicts into a single batched TensorDict.

    When the replay buffer samples a batch of sequences (typically from ListStorage),
    each item is a dict/TensorDict with:
        features: (T+1, F)
        actions: (T, A)
        rewards: (T, 1)
        dones: (T, 1)
        terminated: (T, 1)
        mask: (T, 1)
        vector: (T+1, O) [optional]

    Handles variable-length sequences by padding to max length.

    This function stacks them along batch dimension to produce:
        features: (B, T_max+1, F)
        actions: (B, T_max, A)
        etc.

    Args:
        batch: Either a TensorDict (if already batched), list of TensorDicts, or list of dicts

    Returns:
        TensorDict with batched tensors (padded if sequences had different lengths)
    """
    if isinstance(batch, TensorDict):
        # Already batched
        return batch

    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        # Extract keys from first item
        first = batch[0]
        if isinstance(first, TensorDict):
            keys = first.keys()
        elif isinstance(first, dict):
            keys = first.keys()
        else:
            return batch  # Unknown format, return as-is

        first_key_set = set(keys)
        for i, item in enumerate(batch[1:], start=1):
            item_keys = set(item.keys()) if hasattr(item, "keys") else set()
            if item_keys != first_key_set:
                raise RuntimeError(
                    "Replay batch contains mixed schemas and cannot be collated. "
                    f"Sample[0] keys={sorted(first_key_set)}, sample[{i}] keys={sorted(item_keys)}. "
                    "This usually happens when expert transition data (pixels/next_pixels) "
                    "is mixed with sequence chunks (features/actions/...)."
                )

        # Check if sequences have variable lengths and determine max lengths
        def get_seq_length(item, key):
            """Get sequence length from first time dimension."""
            val = item.get(key) if isinstance(item, dict) else item[key]
            if isinstance(val, torch.Tensor) and val.ndim > 0:
                return val.shape[0]
            return None

        # Detect features/actions keys to determine sequence structure
        feature_keys = [
            k
            for k in keys
            if k in ("features", "actions", "rewards", "dones", "terminated", "mask")
        ]
        vector_keys = [k for k in keys if k in ("vector",)]

        # Get max sequence lengths for each key
        max_lens = {}
        has_variable_length = False
        for key in feature_keys:
            lens = [get_seq_length(b, key) for b in batch]
            if any(l is not None for l in lens):
                max_len = max(l for l in lens if l is not None)
                max_lens[key] = max_len
                if len(set(l for l in lens if l is not None)) > 1:
                    has_variable_length = True

        # Pad sequences if needed
        if has_variable_length:
            padded_batch = []
            for item in batch:
                padded_item = {}
                for k in keys:
                    val = item.get(k) if isinstance(item, dict) else item[k]
                    if not isinstance(val, torch.Tensor):
                        padded_item[k] = val
                        continue

                    # Determine if this tensor needs padding
                    if k in max_lens:
                        current_len = val.shape[0]
                        target_len = max_lens[k]

                        # Handle different sequence types
                        if k == "vector":
                            # vector is (T+1, D); pad along time dimension
                            if current_len < target_len + 1:
                                pad_amount = (target_len + 1) - current_len
                                padding = (0, 0, 0, pad_amount)  # pad time dim
                                padded_val = torch.nn.functional.pad(
                                    val, padding, mode="constant", value=0.0
                                )
                                padded_item[k] = padded_val
                            else:
                                padded_item[k] = val
                        elif k in ("features",):
                            # features is (T+1, F); pad along time dimension
                            if current_len < target_len + 1:
                                pad_amount = (target_len + 1) - current_len
                                padding = (0, 0, 0, pad_amount)  # pad time dim
                                padded_val = torch.nn.functional.pad(
                                    val, padding, mode="constant", value=0.0
                                )
                                padded_item[k] = padded_val
                            else:
                                padded_item[k] = val
                        else:
                            # actions, rewards, dones, terminated, mask are (T, ...)
                            if current_len < target_len:
                                pad_amount = target_len - current_len
                                padding = (0, 0, 0, pad_amount)  # pad time dim
                                padded_val = torch.nn.functional.pad(
                                    val, padding, mode="constant", value=0.0
                                )
                                padded_item[k] = padded_val
                            else:
                                padded_item[k] = val
                    else:
                        padded_item[k] = val

                padded_batch.append(padded_item)
            batch = padded_batch

        # Stack each key
        result_dict = {}
        for k in keys:
            tensors = [b[k] for b in batch]
            # Check if all tensors are torch.Tensor and have same shape
            if all(isinstance(t, torch.Tensor) for t in tensors):
                result_dict[k] = torch.stack(tensors, dim=0)
            else:
                result_dict[k] = tensors

        return TensorDict(result_dict, batch_size=[len(batch)])

    return batch


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
    from torchrl.data.replay_buffers import PrioritizedReplayBuffer, ReplayBuffer, LazyTensorStorage

    storage = LazyTensorStorage(max_size=rb_kwargs["max_size"])
    use_per = rb_kwargs.get("use_per", True)

    if use_per:
        rb = PrioritizedReplayBuffer(
            alpha=rb_kwargs["alpha"],
            beta=rb_kwargs["beta"],
            storage=storage,
            batch_size=rb_kwargs["batch_size"],
            collate_fn=_collate_sequence_batch,  # Custom collation for sequence batches
        )
    else:
        rb = ReplayBuffer(
            storage=storage,
            batch_size=rb_kwargs["batch_size"],
            collate_fn=_collate_sequence_batch,  # Custom collation for sequence batches
        )

    if getattr(cfg, "use_expert_demonstrations", False):
        from assetto_corsa_rl.train.logging_utils import log_warning

        log_warning(
            "[LEARNER] Skipping expert demonstrations: replay schema expects feature transitions, "
            "while demo loader emits raw pixel transitions."
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
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.log_alpha = log_alpha
        self.alpha_opt = alpha_opt
        self.target_entropy = target_entropy
        self.storage = storage
        self.env_kwargs = env_kwargs or {}
        self.total_steps = 0
        self.start_time = time.time()
        self._episode_returns = []
        self._current_episode_return = torch.zeros(1, device=device)

        # AC reset tracking
        self._last_ac_reset_step = 0
        self._ac_reset_interval_steps = int(getattr(cfg, "ac_reset_interval_steps", 0))
        self._ac_exe_path = getattr(
            cfg, "ac_exe_path", r"D:\Steam\steamapps\common\assettocorsa\acs.exe"
        )
        self._ac_startup_wait = float(getattr(cfg, "ac_startup_wait", 20.0))

        self._queue: queue.Queue = queue.Queue()

        use_async = bool(getattr(cfg, "use_async", False))
        if use_async:
            self.collector = None
            self.learner = None
            return

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
        if self.learner is not None:
            return self.learner.episode_returns
        return self._episode_returns

    @episode_returns.setter
    def episode_returns(self, v):
        if self.learner is not None:
            self.learner.episode_returns = v
        else:
            self._episode_returns = list(v)

    @property
    def current_episode_return(self):
        if self.collector is not None:
            return self.collector.current_episode_return
        return self._current_episode_return

    @current_episode_return.setter
    def current_episode_return(self, v):
        if self.collector is not None:
            self.collector.current_episode_return = v
        else:
            self._current_episode_return = v

    # ── synchronous single-process training loop ───────────────────────────

    def run(self, total_steps: int = 0):
        if self.collector is None or self.learner is None:
            raise RuntimeError("Synchronous run() requires non-async worker initialization")

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

            # Periodic AC reset
            if self._ac_reset_interval_steps > 0:
                self._last_ac_reset_step = maybe_reset_ac(
                    last_reset_step=self._last_ac_reset_step,
                    current_step=self.total_steps,
                    reset_interval_steps=self._ac_reset_interval_steps,
                    ac_exe_path=self._ac_exe_path,
                    startup_wait=self._ac_startup_wait,
                )

        print("Training finished")

    # ── async multiprocess training loop ──────────────────────────────────

    def run_async(self, total_steps: int = 0):
        """Spawn collector and learner as separate processes using
        torch.multiprocessing for asynchronous, decoupled training."""

        if self.actor is None or self.q1 is None or self.q2 is None:
            raise RuntimeError("Async run requires actor/critic modules")

        if self.log_alpha is None:
            raise RuntimeError("Async run requires log_alpha parameter")

        actor_state = {
            k: v.cpu().clone().share_memory_() for k, v in self.actor.state_dict().items()
        }

        collector_device_cfg = getattr(self.cfg, "collector_device", None)
        if collector_device_cfg is not None:
            collector_device = torch.device(str(collector_device_cfg))
        elif self.device.type == "cuda":
            collector_device = torch.device("cpu")
        else:
            collector_device = self.device

        collector_env_kwargs = dict(self.env_kwargs or {})
        collector_env_kwargs["device"] = collector_device
        log_info(f"Async devices: learner={self.device}, collector={collector_device}")

        weights_lock = mp.Lock()
        weights_version = mp.Value("i", 0)

        queue_size = int(getattr(self.cfg, "queue_size", 65536))
        transitions_queue = mp.Queue(maxsize=queue_size)
        log_queue = mp.Queue(maxsize=10_000)

        stop_event = mp.Event()

        actor_for_learner = copy.deepcopy(self.actor).cpu()
        actor_for_collector = copy.deepcopy(self.actor).cpu()
        q1_mp = copy.deepcopy(self.q1).cpu()
        q2_mp = copy.deepcopy(self.q2).cpu()
        q1_target_mp = copy.deepcopy(self.q1_target).cpu()
        q2_target_mp = copy.deepcopy(self.q2_target).cpu()

        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass

        collector_proc = mp.Process(
            target=_collector_process_fn,
            kwargs=dict(
                cfg=self.cfg,
                env_kwargs=collector_env_kwargs,
                actor=actor_for_collector,
                transitions_queue=transitions_queue,
                stop_event=stop_event,
                device=collector_device,
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
            use_per=bool(getattr(self.cfg, "use_per", True)),
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
                log_alpha_value=float(self.log_alpha.data.cpu().item()),
                target_entropy=self.target_entropy,
                action_dim=int(self.env.action_spec.shape[-1]),
                transitions_queue=transitions_queue,
                device=self.device,
                shared_weights=actor_state,
                weights_lock=weights_lock,
                weights_version=weights_version,
                log_queue=log_queue,
                stop_event=stop_event,
                total_steps=total_steps,
                episode_returns=list(self.episode_returns),
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
        except queue.Empty:
            break
        except Exception as e:
            print(f"[ERROR] Exception getting from log_queue: {type(e).__name__}: {e}")
            break
        if not isinstance(item, dict):
            print(f"[WARNING] Invalid log queue item type: {type(item)}")
            continue
        step = item.pop("step", None)
        data = item.pop("data", item)  # support both {step, data} and flat dicts
        try:
            wandb.log(data, step=step)
        except Exception as e:
            print(f"[ERROR] wandb.log failed: {type(e).__name__}: {e}")


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
