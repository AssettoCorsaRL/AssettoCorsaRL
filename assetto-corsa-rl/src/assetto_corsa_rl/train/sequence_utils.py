import torch


class EpisodeAccumulator:
    """Accumulates transitions, emits overlapping fixed-length sequence chunks."""

    def __init__(self, seq_len: int = 16, overlap: int = 8, burn_in: int = 0):
        self.seq_len = seq_len
        self.overlap = overlap
        self.stride = seq_len - overlap
        self.burn_in = burn_in
        self._buf: list[dict] = []
        self._emitted_first = False

    def add(self, transition: dict):
        self._buf.append(transition)

    def maybe_emit(self) -> list[dict]:
        out = []
        while len(self._buf) >= self.seq_len:
            # First chunk doesn't have burn-in, subsequent chunks use overlap as burn-in
            chunk_burn_in = self.burn_in if self._emitted_first else 0
            out.append(
                self._pack(
                    self._buf[: self.seq_len],
                    real_len=self.seq_len,
                    burn_in=chunk_burn_in,
                )
            )
            self._buf = self._buf[self.stride :]
            self._emitted_first = True
        return out

    def flush(self) -> list[dict]:
        out = []
        if len(self._buf) >= 4:
            real_len = len(self._buf)
            chunk = list(self._buf)
            while len(chunk) < self.seq_len:
                pad = {k: torch.zeros_like(v) for k, v in chunk[-1].items()}
                pad["done"] = torch.ones_like(pad["done"])
                pad["terminated"] = torch.ones_like(pad.get("terminated", pad["done"]))
                # next_features = last real next_features (frozen state)
                pad["next_features"] = chunk[-1]["next_features"].clone()
                pad["features"] = chunk[-1]["next_features"].clone()
                chunk.append(pad)
            out.append(self._pack(chunk[: self.seq_len], real_len=real_len, burn_in=0))
        self._buf.clear()
        return out

    @staticmethod
    def _pack(
        transitions: list[dict],
        real_len: int | None = None,
        burn_in: int = 0,
    ) -> dict:
        """Pack transitions into a sequence dict with optional burn-in masking.

        Args:
            transitions: List of transition dicts
            real_len: Number of real (non-padded) transitions. Defaults to len(transitions)
            burn_in: Number of initial steps to mask out (LSTM warm-up, no loss computed)

        Returns:
            Dict with stacked tensors and a mask indicating trainable steps
        """
        T = len(transitions)
        if real_len is None:
            real_len = T

        feats = [t["features"] for t in transitions] + [transitions[-1]["next_features"]]

        # Mask: 1 for trainable steps, 0 for padding AND burn-in
        mask = torch.ones(T, 1)

        # Zero out padded steps (beyond real_len)
        if real_len < T:
            mask[real_len:] = 0.0

        # Zero out burn-in steps (kept for LSTM warm-up but not trained)
        if burn_in > 0:
            mask[:burn_in] = 0.0

        result = {
            "features": torch.stack(feats),  # (T+1, F)
            "actions": torch.stack([t["action"] for t in transitions]),  # (T, A)
            "rewards": torch.stack([t["reward"] for t in transitions]),  # (T, 1)
            "dones": torch.stack([t["done"] for t in transitions]),  # (T, 1)
            "terminated": torch.stack([t["terminated"] for t in transitions]),  # (T, 1)
            "mask": mask,  # (T, 1) - 1 for trainable, 0 for padding/burn-in
        }
        if "vector" in transitions[0]:
            vecs = [t["vector"] for t in transitions]
            vecs.append(transitions[-1].get("next_vector", transitions[-1]["vector"]))
            result["vector"] = torch.stack(vecs)  # (T+1, O)
        return result


def collate_sequences(sequences: list[dict]) -> dict:
    """Stack a list of sequence dicts into a batched dict.

    Each sequence has shape:
        features: (T+1, F)
        actions:  (T, A)
        rewards:  (T, 1)
        dones:    (T, 1)
        terminated: (T, 1)
        mask:     (T, 1)
        vector:   (T+1, O)  [optional]

    This function stacks them along batch dimension to produce:
        features: (B, T+1, F)
        actions:  (B, T, A)
        rewards:  (B, T, 1)
        etc.

    Args:
        sequences: List of sequence dicts (typically from replay buffer sampling)

    Returns:
        Dict with same keys but batched along dim 0
    """
    if not sequences:
        return {}

    result = {}
    keys = sequences[0].keys()

    for k in keys:
        tensors = [s[k] for s in sequences]
        result[k] = torch.stack(tensors, dim=0)

    return result
