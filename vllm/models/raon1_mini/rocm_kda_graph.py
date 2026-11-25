# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in HIP graphs for the native ROCm AR and speculative RDA decode chain.

This wraps the existing AMD kernels; it does not claim a FlyDSL fused RDA kernel.
The initial warm execution is the current decode step. Never run the stateless
MoE graph warmup loop here: that would advance recurrent/conv caches twice.
"""

import copy
import os

import torch

from .rocm_moe_outer_graph import capture_permitted

_FIELDS = (
    "has_initial_state",
    "non_spec_query_start_loc",
    "non_spec_state_indices_tensor",
    "spec_token_indx",
    "non_spec_token_indx",
    "spec_state_indices_tensor",
    "spec_query_start_loc",
    "num_accepted_tokens",
)
_COUNTS = ("num_prefills", "num_decodes", "num_spec_decodes", "num_actual_tokens")
_ANNOUNCED = set()


def _signature(value):
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    return value


def _announce(event, text):
    if event not in _ANNOUNCED:
        _ANNOUNCED.add(event)
        print(f"RAON_ROCM_KDA_DECODE_GRAPH {text}", flush=True)


def _stamp(layer):
    params = tuple(layer.parameters())
    weights = []
    for param in params:
        try:
            version = param._version
        except RuntimeError:
            version = None
        weights.append((id(param), param.data_ptr(), version, _signature(param)))
    # Cache CONTENT changes on every token and must not invalidate the graph.
    # Cache replacement/layout changes must invalidate it.
    states = tuple((state.data_ptr(), _signature(state)) for state in layer.kv_cache)
    return params + tuple(layer.kv_cache), (
        tuple(weights),
        states,
        layer.gate_lower_bound,
        layer.kda_decode_backend,
        layer.o_norm.eps,
        getattr(layer, "rocm_aiter_fused_decode", False),
        getattr(layer, "rocm_decode_backend", "legacy"),
    )


def _copy_inputs(destinations, sources):
    for destination, source in zip(destinations, sources):
        destination.copy_(source)


def try_forward(layer, metadata, inputs, out):
    """Return True only after producing this step's output and state updates."""
    if not torch.version.hip or not out.is_cuda:
        return False
    if torch.cuda.is_current_stream_capturing():
        # An enclosing FULL graph owns these operations; do not nest a graph.
        return False
    plain = metadata.num_spec_decodes == 0 and metadata.num_decodes > 0
    speculative = metadata.num_spec_decodes > 0 and metadata.num_decodes == 0
    max_tokens = int(os.environ.get("RAON_KDA_DECODE_GRAPH_MAX", "128"))
    if (
        metadata.num_prefills != 0
        or not (plain or speculative)
        or not 1 <= metadata.num_actual_tokens <= max_tokens
        or layer.kda_decode_backend != "rocm_recurrent"
        or not getattr(layer, "_raon_decode_graph_local", False)
    ):
        _announce("shape", "INERT requires a bounded pure AR or pure SD decode batch")
        return False
    kind = "sd" if speculative else "ar"
    permitted = capture_permitted()
    tensors, stamp = _stamp(layer)
    cache = getattr(layer, "_raon_kda_decode_graph", None)
    if cache is None or getattr(layer, "_raon_kda_decode_graph_stamp", None) != stamp:
        cache = layer._raon_kda_decode_graph = {}
        layer._raon_kda_decode_graph_stamp = stamp
    values = tuple(getattr(metadata, name) for name in _FIELDS)
    key = (
        kind,
        tuple(getattr(metadata, name) for name in _COUNTS),
        tuple(_signature(value) for value in inputs),
        _signature(out),
        tuple(_signature(value) for value in values),
    )
    entry = cache.get(key)
    if entry is None:
        if not permitted:
            _announce(
                "window",
                "INERT new shapes need warmup outside breakable capture/replay",
            )
            return False
        limit = int(os.environ.get("RAON_KDA_DECODE_GRAPH_SHAPES", "8"))
        if len(cache) >= limit:
            _announce("limit", "INERT per-layer captured-shape limit reached")
            return False
        static_inputs = tuple(torch.empty_like(value) for value in inputs)
        _copy_inputs(static_inputs, inputs)
        static_metadata = copy.copy(metadata)
        metadata_inputs, metadata_sources = [], []
        for name, value in zip(_FIELDS, values):
            if isinstance(value, torch.Tensor):
                static_value = torch.empty_like(value)
                static_value.copy_(value)
                setattr(static_metadata, name, static_value)
                metadata_inputs.append(static_value)
                metadata_sources.append(value)
        static_out = torch.zeros_like(out)
        # Exactly ONE executed decode: compile kernels and advance the live
        # caches once. Stream capture below records GPU operations, not another
        # decode. Its output cannot be consumed until a subsequent replay.
        layer._forward_step(static_metadata, *static_inputs, static_out)
        first_output = static_out.clone()
        tensors, stamp = _stamp(layer)
        layer._raon_kda_decode_graph_stamp = stamp
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer._forward_step(static_metadata, *static_inputs, static_out)
        cache[key] = (
            graph,
            static_inputs,
            tuple(metadata_inputs),
            static_out,
            static_metadata,
            tensors,
        )
        out.copy_(first_output)
        _announce(
            ("capture", kind),
            f"CAPTURED kind={kind} tokens={metadata.num_actual_tokens} state_steps=1",
        )
        return True
    graph, static_inputs, metadata_inputs, static_out, _, _ = entry
    _copy_inputs(static_inputs, inputs)
    _copy_inputs(
        metadata_inputs, [value for value in values if isinstance(value, torch.Tensor)]
    )
    graph.replay()
    out.copy_(static_out)
    _announce(
        ("replay", kind),
        f"REPLAYED kind={kind} tokens={metadata.num_actual_tokens} "
        "metadata_refreshed=1",
    )
    return True
