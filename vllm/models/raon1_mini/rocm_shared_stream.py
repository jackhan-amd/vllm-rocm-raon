# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect the Raon stream switch to the native ROCm shared-expert runner.

MXFP8's custom MoE already owns a launch/join pair in model.py. Native MXFP4
instead delegates to SharedExperts; configure that owner, never run the shared
expert again outside it. Enabled only with the opt-in FlyDSL candidate.
"""

import os
import types

import torch


def configure_native(module, x):
    if (
        not torch.version.hip
        or os.environ.get("RAON_MOE_BACKEND", "legacy") != "flydsl"
    ):
        return
    requested = os.environ.get("RAON_MOE_SHARED_SIDE_STREAM")
    if requested is None:
        return
    if requested not in ("0", "1"):
        raise ValueError("RAON_MOE_SHARED_SIDE_STREAM must be 0 or 1")
    previous = getattr(module, "_raon_native_shared_stream_mode", None)
    if previous is not None:
        if previous != requested:
            raise RuntimeError(
                "Restart the server to change the shared-stream graph configuration"
            )
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Configure the native shared-expert stream before graph capture"
        )
    if not getattr(module, "_raon_local_graph_world", False):
        raise RuntimeError(
            "Raon native shared-stream candidate requires TP1/DP1 without EP"
        )
    shared = module.experts.shared_experts
    if shared is None:
        module._raon_native_shared_stream_mode = requested
        return
    if requested == "0":
        shared._stream = None
    else:
        from vllm import envs

        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            raise RuntimeError(
                "RAON_MOE_SHARED_SIDE_STREAM=1 conflicts with "
                "VLLM_DISABLE_SHARED_EXPERTS_STREAM=1"
            )
        # A separate stream avoids sharing MLA's auxiliary stream. The native
        # runner still owns event dependencies and the final shared addition.
        shared._stream = torch.cuda.Stream(device=x.device)
        shared._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
        shared._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
    module._raon_native_shared_stream_mode = requested
    print(
        f"RAON_ROCM_SHARED_STREAM CONFIGURED mode={requested} owner=native_moe_runner "
        "scheduling_constraints=preserved",
        flush=True,
    )
    if requested == "0":
        return

    # Observe the actual native enqueue/wait path, not just the env setting.
    # Do not alter SharedExpertsOrder, duplicate the expert or move arithmetic.
    original_maybe, original_wait = shared.maybe_forward_async, shared.wait
    pending, announced = set(), set()

    def maybe_forward_async(owner, inputs):
        queued = original_maybe(inputs)
        if queued:
            pending.add(owner._output_idx)
        elif "inert" not in announced:
            announced.add("inert")
            order = owner._determine_shared_experts_order(inputs)
            print(
                f"RAON_ROCM_SHARED_STREAM INERT native_order={order.name} "
                f"tokens={inputs.shape[0]}",
                flush=True,
            )
        return queued

    def wait(owner):
        original_wait()
        if owner._output_idx in pending:
            pending.remove(owner._output_idx)
            if "joined" not in announced:
                announced.add("joined")
                capturing = int(torch.cuda.is_current_stream_capturing())
                print(
                    f"RAON_ROCM_SHARED_STREAM JOIN_RECORDED owner=native_moe_runner "
                    f"capturing={capturing} enqueue_join_pair=1",
                    flush=True,
                )

    shared.maybe_forward_async = types.MethodType(maybe_forward_async, shared)
    shared.wait = types.MethodType(wait, shared)
