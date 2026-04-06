from __future__ import annotations

import os
import sys
import time
import warnings
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

warnings.filterwarnings("ignore")


def _resolve_actor_net(policy):
    """Resolve underlying actor network exposing ``cnn``/``mlp`` fields."""
    queue = [getattr(policy, "actor", None)]
    actor = getattr(policy, "actor", None)
    if actor is not None:
        queue.append(getattr(actor, "module", None))
    visited = set()

    while queue:
        node = queue.pop(0)
        if node is None:
            continue

        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        if all(hasattr(node, k) for k in ("cnn", "mlp", "obs_dim")):
            return node

        inner = getattr(node, "module", None)
        if inner is not None:
            queue.append(inner)

        if isinstance(node, nn.ModuleList):
            queue.extend(list(node))

        if isinstance(node, (list, tuple)):
            queue.extend(list(node))

    raise AttributeError("Could not resolve underlying actor network from policy.actor wrappers")


def configure_cpu(
    intra_threads: int = 8,
    inter_threads: int = 1,
    pin_p_cores: bool = True,
) -> None:
    """
    Tune every threading knob for minimum single-sample latency.

    For batch-size-1 on a hybrid CPU, using *only* the P-core count
    (8 on i7-14700F) beats saturating all 28 logical threads because
    the tiny matmuls are memory-bound and extra thread sync hurts.
    """
    torch.set_num_threads(intra_threads)
    try:  # may throw if already initialised
        torch.set_num_interop_threads(inter_threads)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("high")
    torch.backends.mkldnn.enabled = True

    env = {
        "OMP_NUM_THREADS": str(intra_threads),
        "MKL_NUM_THREADS": str(intra_threads),
        "OMP_WAIT_POLICY": "ACTIVE",  # spin-wait, never sleep
        "OMP_PROC_BIND": "CLOSE",
        "OMP_SCHEDULE": "STATIC",
        "KMP_BLOCKTIME": "0",
        "KMP_SETTINGS": "0",
        "MKL_ENABLE_INSTRUCTIONS": "AVX2",
        # DNNL (oneDNN / MKL-DNN) verbose off for perf
        "DNNL_VERBOSE": "0",
        "ONEDNN_VERBOSE": "0",
    }
    if pin_p_cores:
        ids = ",".join(map(str, range(intra_threads)))
        if sys.platform == "linux":
            env["KMP_AFFINITY"] = f"explicit,proclist=[{ids}],granularity=fine"
            env["GOMP_CPU_AFFINITY"] = ids
        else:
            env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

    os.environ.update(env)


class _ForwardCore(nn.Module):
    """
    Flat, JIT-trace-friendly forward:
        (pixels, vector) → (loc, scale)

    No Python dicts cross the module boundary; everything is tensors.
    """

    def __init__(
        self,
        cnn: nn.Module,
        mlp: nn.Module,
        obs_dim: int,
    ):
        super().__init__()
        self.cnn = cnn
        self.obs_dim = obs_dim
        # pull the final BoundedNormalParams out so we can return a tuple
        self.mlp_body = mlp[:-1]  # everything except last module
        self.param_head = mlp[-1]  # BoundedNormalParams

    def forward(
        self,
        pixels: Tensor,
        vector: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        feat = self.cnn(pixels)

        if self.obs_dim > 0:
            x = torch.cat([feat, vector], dim=-1)
        else:
            x = feat

        x = self.mlp_body(x)
        params = self.param_head(x)
        return params["loc"], params["scale"]


class SACInferenceEngine:
    """
    Zero-overhead inference wrapper.  Accepts raw tensors, returns raw
    tensors.  No TensorDict, no ProbabilisticActor, no distribution
    object creation — just math.
    """

    def __init__(
        self,
        policy,  # SACPolicy
        *,
        input_hw: Tuple[int, int] = (84, 84),
        backend: str = "compile",  # "compile" | "jit" | "eager"
        quantize: bool = True,
        channels_last: bool = True,
        num_threads: int = 8,  # = P-core count
        warmup: int = 120,
        benchmark_n: int = 300,
        deterministic: bool = False,
        copy: bool = True,
    ):

        self.deterministic = deterministic
        self._channels_last = channels_last
        self._backend = backend

        configure_cpu(intra_threads=num_threads)

        actor_net = _resolve_actor_net(policy)
        if copy:
            actor_net = deepcopy(actor_net)

        self.obs_dim: int = actor_net.obs_dim

        cnn = actor_net.cnn
        mlp = actor_net.mlp

        try:
            dk = policy.actor.distribution_kwargs
            self._lo = (
                dk["low"].cpu().float()
                if isinstance(dk["low"], Tensor)
                else torch.tensor(dk["low"], dtype=torch.float32)
            )
            self._hi = (
                dk["high"].cpu().float()
                if isinstance(dk["high"], Tensor)
                else torch.tensor(dk["high"], dtype=torch.float32)
            )
        except Exception:
            action_dim = actor_net.mlp[-2].out_features // 2  # derive from model
            self._lo = torch.full((action_dim,), -1.0)
            self._hi = torch.full((action_dim,), 1.0)
        self._range = self._hi - self._lo

        in_c = 3
        for m in cnn.modules():
            if isinstance(m, nn.Conv2d):
                in_c = m.in_channels
                break
        self._in_shape = (1, in_c, *input_hw)

        cnn.eval()
        mlp.eval()

        if channels_last:
            cnn = cnn.to(memory_format=torch.channels_last)

        if quantize:
            mlp = torch.quantization.quantize_dynamic(mlp, {nn.Linear}, dtype=torch.qint8)

        self._cnn = cnn
        self._mlp = mlp

        fmt = torch.channels_last if channels_last else torch.contiguous_format
        self._pix_buf = torch.empty(self._in_shape, dtype=torch.float32, memory_format=fmt)

        self._dummy_vec = torch.empty(1, max(self.obs_dim, 1))

        self._build_backend(cnn, mlp, backend)

        self._warmup_and_bench(warmup, benchmark_n)

    def _build_backend(self, cnn, mlp, backend: str):
        core = _ForwardCore(cnn, mlp, self.obs_dim)
        core.eval()

        if backend == "jit":
            self._fn = self._make_jit(core)
        elif backend == "compile":
            self._fn = self._make_compile(core)
        else:
            self._fn = core  # eager, already optimised via quant + channels_last

    def _make_compile(self, core: _ForwardCore):
        try:
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 8
            compiled = torch.compile(
                core,
                mode="max-autotune",
                backend="inductor",
                fullgraph=False,
                dynamic=False,
            )
            print("[engine] torch.compile  ✓  (mode=max-autotune, inductor)")
            return compiled
        except Exception as exc:
            print(f"[engine] torch.compile failed ({exc}); falling back to eager")
            return core

    def _make_jit(self, core: _ForwardCore):
        try:
            pix = torch.randn(self._in_shape)
            if self._channels_last:
                pix = pix.contiguous(memory_format=torch.channels_last)
            vec = torch.randn(1, max(self.obs_dim, 1))

            with torch.no_grad():
                traced = torch.jit.trace(core, (pix, vec), strict=False)
            frozen = torch.jit.freeze(traced)
            frozen = torch.jit.optimize_for_inference(frozen)
            # warm the fused graph
            for _ in range(10):
                frozen(pix, vec)
            print("[engine] TorchScript trace+freeze+optimize_for_inference  ✓")
            return frozen
        except Exception as exc:
            print(f"[engine] JIT trace failed ({exc}); falling back to eager")
            return core

    @torch.inference_mode()
    def _warmup_and_bench(self, warmup_n: int, bench_n: int):
        pix = torch.randn(self._in_shape, dtype=torch.float32)
        if self._channels_last:
            pix = pix.contiguous(memory_format=torch.channels_last)
        vec = torch.randn(1, self.obs_dim) if self.obs_dim > 0 else self._dummy_vec

        # warm-up  (compile / JIT caches)
        print(f"[engine] warming up ({warmup_n} iters) …")
        for _ in range(warmup_n):
            self._fn(pix, vec)

        # benchmark
        lat: list[float] = []
        for _ in range(bench_n):
            t0 = time.perf_counter_ns()
            self._fn(pix, vec)
            t1 = time.perf_counter_ns()
            lat.append((t1 - t0) / 1e6)

        lat.sort()
        trim = max(1, len(lat) // 20)
        body = lat[trim:-trim] if trim < len(lat) // 2 else lat
        avg = sum(body) / len(body)
        med = body[len(body) // 2]
        p99 = lat[int(len(lat) * 0.99)]
        best = lat[0]

        print(
            f"[engine] latency  →  avg {avg:.3f} ms │ "
            f"median {med:.3f} ms │ p99 {p99:.3f} ms │ best {best:.3f} ms\n"
            f"[engine] throughput  →  ~{1000 / avg:,.0f} inf/s"
        )

    @torch.inference_mode()
    def get_action(
        self,
        pixels: Tensor,
        vector: Optional[Tensor] = None,
        deterministic: Optional[bool] = None,
    ) -> Tensor:
        """
        pixels : (C,H,W) or (1,C,H,W)  float32
        vector : (D,) or (1,D)          float32, or None
        Returns: (action_dim,)          float32
        """
        det = deterministic if deterministic is not None else self.deterministic

        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)

        self._pix_buf.copy_(pixels)

        if self.obs_dim > 0:
            if vector is None:
                raise ValueError("Policy expects a vector observation")
            if vector.ndim == 1:
                vector = vector.unsqueeze(0)
        else:
            vector = self._dummy_vec

        loc, scale = self._fn(self._pix_buf, vector)

        if det:
            y = torch.tanh(loc)
        else:
            y = torch.tanh(loc + scale * torch.randn_like(scale))

        action = self._lo + (y + 1.0) * (0.5 * self._range)
        return action.squeeze(0)

    def reset(self) -> None:
        """No-op retained for API compatibility."""
        return None

    @staticmethod
    def benchmark(
        engine: "SACInferenceEngine",
        pixels: Tensor,
        vector: Optional[Tensor] = None,
        n: int = 2000,
    ) -> list[float]:
        """Full benchmark with percentile breakdown."""
        engine.reset()
        times: list[float] = []
        with torch.inference_mode():
            for _ in range(n):
                t0 = time.perf_counter_ns()
                engine.get_action(pixels, vector)
                t1 = time.perf_counter_ns()
                times.append((t1 - t0) / 1e6)
        engine.reset()
        times.sort()
        pct = lambda p: times[int(len(times) * p / 100)]
        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        print(f"\n{'═' * 56}")
        print(f"  SAC Inference Benchmark  ({n:,} iterations)")
        print(f"{'═' * 56}")
        print(f"  min    {times[0]:.3f} ms")
        print(f"  p1     {pct(1):.3f} ms")
        print(f"  p5     {pct(5):.3f} ms")
        print(f"  p25    {pct(25):.3f} ms")
        print(f"  p50    {pct(50):.3f} ms  (median)")
        print(f"  p75    {pct(75):.3f} ms")
        print(f"  p95    {pct(95):.3f} ms")
        print(f"  p99    {pct(99):.3f} ms")
        print(f"  max    {times[-1]:.3f} ms")
        print(f"  avg    {avg:.3f} ms  (σ={std:.3f})")
        print(f"  throughput  ~{1000 / avg:,.0f} inf/s")
        print(f"{'═' * 56}\n")
        return times
