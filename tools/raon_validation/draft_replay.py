# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the first failing ROCm GQA request against dense masked attention."""

import torch

from .common import error, provenance


def run(args, result):
    from vllm.v1.attention.ops.paged_attn import PagedAttention
    from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

    dump = torch.load(args.dump, map_location="cpu", weights_only=True)
    a = dump["draft_attention"]
    m = a["metadata"]
    row = int(a["bad_rows"][0])
    req = int(
        torch.searchsorted(
            m["query_start_loc"][1:].long(), torch.tensor(row), right=True
        )
    )
    start, end = [int(x) for x in m["query_start_loc"][req : req + 2]]
    length, dim = int(m["seq_lens"][req]), args.dim
    q, k, v = [x[start:end].to("cuda").reshape(end - start, -1, dim) for x in a["qkv"]]
    assert q.shape[1] == args.heads and dim == 320, "Captured TP1 Raon draft only"
    pool = a["cache_blocks"].to("cuda")
    lookup = {int(x): i for i, x in enumerate(a["cache_block_ids"])}
    count = (length + pool.shape[2] - 1) // pool.shape[2]
    table = torch.tensor(
        [[lookup[int(x)] for x in m["block_table"][req, :count]]],
        device="cuda",
        dtype=torch.int32,
    )
    kc, vc = PagedAttention.split_kv_cache(pool.transpose(0, 1), k.shape[1], dim)
    kc, vc = kc.view(torch.float8_e4m3fn), vc.view(torch.float8_e4m3fn)
    out = torch.full_like(q, float("nan"))
    ks, vs = [a[name].to("cuda") for name in ["k_scale", "v_scale"]]
    causal = m["causal"] if isinstance(m["causal"], bool) else bool(m["causal"][req])
    context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=out,
        kv_cache_dtype="fp8_e4m3",
        k_cache=kc,
        v_cache=vc,
        b_loc=table,
        b_start_loc=torch.tensor([0, end - start], device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([length], device="cuda", dtype=torch.int32),
        max_seq_len=length,
        max_input_len=end - start,
        k_scale=ks,
        v_scale=vs,
        sliding_window=args.window,
        causal=causal,
    )
    # Convert the packed K and V independently. Read only the union of live
    # windows, so expired NaNs cannot contaminate the independent reference.
    dk = kc.permute(0, 3, 1, 2, 4).reshape(
        pool.shape[0], pool.shape[2], k.shape[1], dim
    )
    dv = vc.permute(0, 3, 1, 2).reshape_as(dk)
    logical_k = dk[table[0].long()].flatten(0, 1)
    logical_v = dv[table[0].long()].flatten(0, 1)
    context = length - (end - start)
    first = max(0, context - args.window + 1)
    refk = torch.cat([logical_k[first:context].float() * ks, k.float()])
    refv = torch.cat([logical_v[first:context].float() * vs, v.float()])
    factor = q.shape[1] // k.shape[1]
    refk, refv = [x.repeat_interleave(factor, dim=1) for x in [refk, refv]]
    scores = torch.einsum("qhd,khd->qhk", q.float(), refk) * dim**-0.5
    qp = context + torch.arange(end - start, device="cuda")
    kp = first + torch.arange(refk.shape[0], device="cuda")
    valid = kp[None, :] >= qp[:, None] - args.window + 1
    if causal:
        valid &= kp[None, :] <= qp[:, None]
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    expected = torch.einsum("qhk,khd->qhd", scores.softmax(-1), refv)
    result.update(
        source=provenance(context_attention_fwd),
        captured_stage=dump["stage"],
        original_nonfinite=not bool(
            torch.isfinite(dump["bad_tensor"][start:end]).all()
        ),
        query_length=end - start,
        sequence_length=length,
    )
    result["cases"].append({"output": error(out, expected)})
