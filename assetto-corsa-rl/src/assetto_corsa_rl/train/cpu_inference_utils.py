"""CPU inference optimizations: quantization, JIT compilation, threading."""

import torch
import torch.quantization as quant
import torch.nn as nn
from typing import Optional


def _safe_set_interop_threads(num_threads: int) -> None:
    """Best-effort inter-op thread configuration.

    PyTorch raises RuntimeError if inter-op threads are set after parallel work
    has started or if this setting was already configured. In those cases we
    keep running with the existing configuration.
    """
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass


def optimize_for_cpu_inference(
    model: torch.nn.Module,
    device: torch.device,
    quantize: bool = True,
    compile_with_jit: bool = False,
    num_threads: int = 4,
) -> torch.nn.Module:
    """Apply multiple optimizations for fast CPU inference.

    Args:
        model: Neural network model to optimize
        device: Target device (should be cpu)
        quantize: Whether to apply INT8 quantization (big speedup but less accurate)
        compile_with_jit: Whether to compile with TorchScript JIT (some speedup)
        num_threads: Number of threads for PyTorch ops (tune based on CPU cores)

    Returns:
        Optimized model ready for inference
    """
    model = model.to(device).eval()

    if str(device) == "cpu":
        torch.set_num_threads(num_threads)
        _safe_set_interop_threads(1)

    if quantize and str(device) == "cpu":
        try:
            model.qconfig = quant.get_default_qconfig("fbgemm")  # x86 CPUs
            quant.prepare(model, inplace=True)
            quant.convert(model, inplace=True)
            print(f"Quantized {type(model).__name__} to INT8")
        except Exception as e:
            print(f"Quantization failed (non-critical): {e}")
            model.eval()

    if compile_with_jit:
        try:
            model = torch.jit.script(model)
            print(f"Compiled {type(model).__name__} with TorchScript")
        except Exception as e:
            print(f"JIT compilation failed (non-critical): {e}")

    return model


def quantize_cnn_only(
    actor_model: torch.nn.Module,
    device: torch.device,
    num_threads: int = 4,
) -> torch.nn.Module:
    """Quantize just the CNN encoder for max compatibility.

    Quantizing the full actor can hurt policy quality due to action distribution,
    so this targets only the frozen CNN encoder which is safe to quantize.

    Args:
        actor_model: Full actor model
        device: Target device
        num_threads: PyTorch thread count

    Returns:
        Actor with quantized CNN (other parts unchanged)
    """
    if str(device) == "cpu":
        torch.set_num_threads(num_threads)

    for module in actor_model.modules():
        if hasattr(module, "cnn"):
            cnn = module.cnn
            try:
                cnn = cnn.eval()

                # NOTE:
                # Static conv quantization converts Conv2d -> quantized Conv2d and then
                # requires quantized (quint8) inputs at runtime. Our collector passes
                # float32 tensors directly to `shared_cnn(...)`, so static conversion can
                # crash with:
                #   Could not run 'quantized::conv2d.new' with arguments from 'CPU'
                # To keep this path safe, only apply dynamic quantization to modules
                # that support it (Linear/LSTM). This avoids quantized conv operators.
                quantized_cnn = torch.quantization.quantize_dynamic(
                    cnn,
                    {nn.Linear, nn.LSTM},
                    dtype=torch.qint8,
                )

                module.cnn = quantized_cnn
                print("[cpu] Applied safe dynamic INT8 quantization to CNN submodules")
            except Exception as e:
                print(f"[cpu][warn] CNN quantization failed: {e}")
            break

    return actor_model


def enable_cpu_optimizations(num_threads: Optional[int] = None):
    """Enable global CPU optimizations.

    Call this early in your training script.

    Args:
        num_threads: Number of threads (default: use all available)
    """
    if num_threads is None:
        num_threads = torch.get_num_threads()

    torch.set_num_threads(num_threads)
    _safe_set_interop_threads(1)

    # Disable openmp dynamic threads (more stable)
    torch.set_num_threads(num_threads)

    print(f"CPU optimizations enabled: {num_threads} threads, 1 inter-op thread")
