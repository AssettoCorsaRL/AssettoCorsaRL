import random
import numpy as np
import pickle
from pickle import UnpicklingError
from pathlib import Path
import torch
from threading import Thread
from queue import Queue


class ReplayMemory:
    """
    The SAC algorithm uses an experience replay buffer to learn from past samples. This class allows to add transitions
    to the replay buffer, save the replay buffer, and to sample past transitions from the replay buffer.

    **Parameters**:

    - **capacity** *(int)*:  Defines the size of the replay buffer.
    """

    def __init__(self, capacity, prefetch_batches: int = 2, device: str = "cuda"):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Prefetch queue for async data loading
        self.prefetch_batches = prefetch_batches
        self.prefetch_queue = (
            Queue(maxsize=prefetch_batches) if prefetch_batches > 0 else None
        )
        self.prefetch_thread = None
        self._stop_prefetch = False

    def push(self, state, action, reward, next_state, done):
        """
        Appends a transistion to the replay buffer.

        **Parameters**:

        - **state** *(array)*:  Contains thirty-two float32 values that represent the agent's observable environment before
        executing its action.
        - **action** *(array)*:  Contains three float32 values that allow to steer, accelerate and decelerate the vehicle.
        - **reward** *(float)*:  Represents the reward that the agent receives from the environment for its action.
        - **next_state** *(array)*:  Contains thirty-two float32 values that represent the agent's observable environment after
        executing its action.
        - **done** *(boolean)*:  Changes to *True* when the number of tiles that the agent visited euqals the current track's
        tile count or the agent goes to far astray and consequently aborts the current episode.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Push multiple transitions at once (more efficient for vectorized envs)."""
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.push(s, a, r, ns, d)

    def sample(self, batch_size):
        """
        Outputs samples contained in the replay buffer.

        **Parameters**:

        - **batch_size** *(int)*:  Defines how many samples to output.

        **Output**:

        - **state** *(array)*: Contains *batch_size* state arrays
        - **action** *(array)*: Contains *batch_size* action arrays
        - **reward** *(array)*: Contains *batch_size* reward values
        - **next_state** *(array)*: Contains *batch_size* next_state arrays
        - **done** *(array)*: Contains *batch_size* done values
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_tensors(self, batch_size):
        """Sample and return GPU tensors directly (faster for training)."""
        state, action, reward, next_state, done = self.sample(batch_size)
        # Use pin_memory for faster GPU transfer
        state_t = (
            torch.as_tensor(state, dtype=torch.float32)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        action_t = (
            torch.as_tensor(action, dtype=torch.float32)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        reward_t = (
            torch.as_tensor(reward, dtype=torch.float32)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        next_state_t = (
            torch.as_tensor(next_state, dtype=torch.float32)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        done_t = (
            torch.as_tensor(done, dtype=torch.float32)
            .pin_memory()
            .to(self.device, non_blocking=True)
        )
        return state_t, action_t, reward_t, next_state_t, done_t

    def start_prefetch(self, batch_size):
        """Start background thread to prefetch batches."""
        if self.prefetch_queue is None or self.prefetch_thread is not None:
            return
        self._stop_prefetch = False
        self.prefetch_thread = Thread(
            target=self._prefetch_worker, args=(batch_size,), daemon=True
        )
        self.prefetch_thread.start()

    def stop_prefetch(self):
        """Stop the prefetch thread."""
        self._stop_prefetch = True
        if self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=1.0)
            self.prefetch_thread = None

    def _prefetch_worker(self, batch_size):
        """Background worker that prefetches batches."""
        while not self._stop_prefetch:
            if len(self.buffer) >= batch_size:
                batch = self.sample_tensors(batch_size)
                try:
                    self.prefetch_queue.put(batch, timeout=0.1)
                except:
                    pass

    def get_prefetched_batch(self, batch_size):
        """Get a prefetched batch or sample a new one."""
        if self.prefetch_queue is not None and not self.prefetch_queue.empty():
            try:
                return self.prefetch_queue.get_nowait()
            except:
                pass
        return self.sample_tensors(batch_size)

    def save(self, path: str):
        """
        Option to save the replay buffer to allow to pick up model training in the future.
        The buffer is saved to path "/memory/buffer" as a Pickle file.

        **Parameters**:

        - **path** *(String)*:  Defined as "buffer" in the calling method train().
        """
        print(f"Saving replay buffer to {path}.")
        save_dir = Path("memory/")
        if not save_dir.exists():
            save_dir.mkdir()
        with open((save_dir / path).with_suffix(".pkl"), "wb") as fp:
            pickle.dump(self.buffer, fp)

    def load(self, path: str):
        """
        Option to load a previously saved replay buffer from a Pickle file.

        ## Parameters:

        - **path** *(String)*:  Path to Pickle file.
        """
        try:
            with open(path, "rb") as fp:
                mem = pickle.load(fp)
                assert (
                    len(mem) == self.capacity
                ), "Capacity of the replay buffer and length of the loaded memory don't match!"
                self.buffer = mem
            print(f"Loaded saved replay buffer from {path}.")
        except UnpicklingError:
            raise TypeError("This file doesn't contain a pickled list!")

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        max_prio = self.priorities.max() if self.buffer else 1.0
        max_prio = max(max_prio, self.eps)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta: float):
        prios = (
            self.priorities
            if len(self.buffer) == self.capacity
            else self.priorities[: self.position]
        )
        prios = np.maximum(prios, self.eps)
        probs = prios**self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full_like(probs, 1.0 / len(probs))
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return state, action, reward, next_state, done, weights, indices

    def save(self, path: str):
        """Save buffer plus priorities for later resume."""
        print(f"Saving prioritized replay buffer to {path}.")
        save_dir = Path("memory/")
        if not save_dir.exists():
            save_dir.mkdir()
        payload = {
            "buffer": self.buffer,
            "priorities": self.priorities,
            "position": self.position,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "eps": self.eps,
        }
        with open((save_dir / path).with_suffix(".pkl"), "wb") as fp:
            pickle.dump(payload, fp)

    def load(self, path: str):
        """Load buffer plus priorities from disk."""
        try:
            with open(path, "rb") as fp:
                payload = pickle.load(fp)
        except UnpicklingError:
            raise TypeError("This file doesn't contain a pickled buffer!")

        buf = payload.get("buffer")
        prios = payload.get("priorities")
        pos = payload.get("position", 0)

        if buf is None or prios is None:
            raise ValueError("Loaded payload missing buffer or priorities")
        if len(buf) > self.capacity:
            raise ValueError("Loaded buffer length exceeds capacity")

        self.buffer = buf
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        copy_len = min(len(prios), self.capacity)
        self.priorities[:copy_len] = prios[:copy_len]
        self.position = (
            pos % self.capacity
            if len(self.buffer) == self.capacity
            else len(self.buffer)
        )
        print(f"Loaded prioritized replay buffer from {path}.")

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(np.abs(prio) + self.eps)
