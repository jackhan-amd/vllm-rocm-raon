# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded HIP graph replay of the opt-in FlyDSL MoE path.

Do not relax the NVIDIA/TE gate in moe_outer_graph. This path captures the
already GPU-routed FlyDSL chain without padding every expert to 256 rows.
"""

import functools
import os
import threading
from typing import Any

import torch

_TLS = threading.local()
_ANNOUNCED = set()


@functools.cache
def _breakable_class():
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    cls: Any = BreakableCUDAGraphCapture
    # current() is None during replay, although capturing a new graph in an
    # eager callback in that window is unsafe. Track replay as well as capture.
    if not getattr(cls, "_raon_rocm_replay_guard", False):
        original = cls.replay

        @functools.wraps(original)
        def replay(self):
            depth = getattr(_TLS, "replay_depth", 0)
            _TLS.replay_depth = depth + 1
            try:
                return original(self)
            finally:
                _TLS.replay_depth = depth

        cls.replay = replay
        cls._raon_rocm_replay_guard = True
    return cls


def capture_permitted():
    cls = _breakable_class()
    return (
        cls.current() is None
        and not getattr(_TLS, "replay_depth", 0)
        and not torch.cuda.is_current_stream_capturing()
    )


def _announce(event, message):
    if event not in _ANNOUNCED:
        _ANNOUNCED.add(event)
        print(f"RAON_ROCM_MOE_OUTER_GRAPH {message}", flush=True)


def _stamp(module):
    from .mxfp8_flydsl import _identity, weight_stamp

    routed = module.experts.routed_experts
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    precision = getattr(module, "_raon_graph_precision", "mxfp8")
    quantized = (
        ()
        if precision == "mxfp4"
        else weight_stamp(routed.w13_weight, routed.w2_weight)
    )
    return tensors, (
        tuple(_identity(t) for t in tensors),
        quantized,
        os.environ.get("RAON_MXFP8_FLYDSL_FUSED_A2", "0")
        if precision == "mxfp8"
        else None,
    )


def try_forward(module, x, compute):
    from . import model

    precision = getattr(module, "_raon_graph_precision", "mxfp8")

    # Global graph capture can record the raw FlyDSL chain directly. A local
    # graph is only useful in the eager-break window, never nested capture.
    if torch.cuda.is_current_stream_capturing():
        return None
    if (
        model._MOE_ROUTE_CAPTURE
        or model.raon_phase.PREFILL_ALIGNED
        or model.moe_combine_ep_size()
        or os.environ.get("RAON_MOE_GROUPED_GRAPH", "1") != "1"
        or x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not 1
        <= x.shape[0]
        <= int(os.environ.get("RAON_MOE_OUTER_GRAPH_MAX_TOKENS", "128"))
    ):
        _announce(
            "config", "INERT unsupported shape, phase, or diagnostic configuration"
        )
        return None
    if precision == "mxfp8" and (
        not model.moe_trainer_realized() or not model.moe_expert_mxfp8_realized()
    ):
        _announce("precision", "INERT MXFP8 requires the trainer-realized expert chain")
        return None
    routed = getattr(module.experts, "routed_experts", None)
    router = getattr(module.experts, "router", None)
    if routed is None or getattr(router, "top_k", None) != 6 or x.shape[1] != 2560:
        _announce("layout", "INERT requires the TP1/DP1 Raon expert layout")
        return None
    if precision == "mxfp8" and (
        tuple(routed.w13_weight.shape) != (128, 2176, 2560)
        or tuple(routed.w2_weight.shape) != (128, 2560, 1088)
    ):
        _announce("mx8-layout", "INERT requires the compact TP1/DP1 MXFP8 layout")
        return None
    # Install the replay guard BEFORE a capture can retain our eager callback.
    permitted = capture_permitted()
    tensors, stamp = _stamp(module)
    cache = getattr(module, "_raon_moe_outer_cache", None)
    if cache is None or getattr(module, "_raon_moe_outer_stamp", None) != stamp:
        cache = module._raon_moe_outer_cache = {}
        module._raon_moe_outer_stamp = stamp
    key = (precision, tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
    entry = cache.get(key)
    if entry is None:
        if not permitted:
            _announce(
                "window",
                "INERT new shapes require warmup outside breakable capture/replay",
            )
            return None
        limit = int(os.environ.get("RAON_MOE_OUTER_GRAPH_SHAPES", "8"))
        if len(cache) >= limit:
            _announce("limit", f"INERT captured-shape limit reached limit={limit}")
            return None
        sx = torch.empty_like(x)
        sx.copy_(x)
        # Initialize native modules, packed weights and compiled kernels before
        # capture. MoE has no recurrent state, so both warmups are stateless.
        compute(module, sx)
        compute(module, sx)
        tensors, stamp = _stamp(module)
        module._raon_moe_outer_stamp = stamp
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = compute(module, sx)
        entry = (graph, sx, result, tensors)
        cache[key] = entry
        _announce(
            ("capture", precision, tuple(x.shape)),
            f"CAPTURED shape={tuple(x.shape)} precision={precision} backend=flydsl",
        )
    graph, sx, result, _ = entry
    sx.copy_(x)
    graph.replay()
    _announce(
        ("replay", precision, tuple(x.shape)),
        f"REPLAYED shape={tuple(x.shape)} precision={precision} backend=flydsl",
    )
    # The existing eager-break caller immediately copies this borrowed output.
    return result
