# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a captured native MLA failure without loading model weights."""

import torch

from .common import error, write_json


def device(value):
    if isinstance(value, torch.Tensor):
        return value.to("cuda")
    return value


def run(args, result):
    import aiter
    from aiter.mla import mla_decode_fwd

    assert args.dump
    dump = torch.load(args.dump, map_location="cpu", weights_only=True)
    native = dump["native_mla"]
    old = native["arguments"]
    q, kv = old["q"].to("cuda"), native["gathered_kv"].to("cuda")
    assert q.ndim == 3 and q.shape[-1] == 576 and args.heads <= q.shape[1] <= 64
    assert old.get("page_size", 1) == 1 and old.get("nhead_kv", 1) == 1
    qo, indptr = old["qo_indptr"].to("cuda"), old["kv_indptr"].to("cuda")
    lengths = indptr.diff().tolist()
    query_lengths = qo.diff().tolist()
    assert all(0 < n <= k for n, k in zip(query_lengths, lengths))
    result.update(
        captured_stage=dump["stage"],
        q_shape=list(q.shape),
        cache_lengths=lengths,
        query_lengths=query_lengths,
        captured_cache_shape=native["cache_shape"],
        inputs_finite={
            "q": bool(torch.isfinite(q.float()).all()),
            "kv": bool(torch.isfinite(kv.float()).all()),
        },
    )
    # Captured work_meta_data embeds device addresses. It is not portable across
    # processes; only semantic CSR inputs may be reused. Save evidence before
    # invoking a native kernel, which can abort without a Python exception.
    write_json(args.output, result)
    assert all(result["inputs_finite"].values()), "Nonfinite captured MLA input"
    qscale, kscale = device(old.get("q_scale", 1.0)), device(old.get("kv_scale", 1.0))
    scale = old["sm_scale"]
    qd, kd = q.float() * qscale, kv.float() * kscale
    expected = torch.empty(q.shape[0], q.shape[1], 512, device="cuda")
    for i, length in enumerate(lengths):
        qs, qe = int(qo[i]), int(qo[i + 1])
        ks, ke = int(indptr[i]), int(indptr[i + 1])
        scores = torch.einsum("qhd,kd->qhk", qd[qs:qe], kd[ks:ke]) * scale
        if old.get("causal", True):
            future = torch.arange(length, device="cuda")[None, :] > (
                length - (qe - qs) + torch.arange(qe - qs, device="cuda")[:, None]
            )
            scores.masked_fill_(future[:, None, :], -torch.inf)
        expected[qs:qe] = torch.einsum(
            "qhk,kd->qhd", scores.softmax(-1), kd[ks:ke, :512]
        )
    # Give every sequence physically backed tail storage without changing logical CSR.
    storage_rows = sum(((n + 15) // 16) * 16 for n in lengths)
    cache = torch.zeros(storage_rows, kv.shape[-1], device="cuda", dtype=kv.dtype)
    index = torch.empty(kv.shape[0], device="cuda", dtype=torch.int32)
    offset = 0
    for i, length in enumerate(lengths):
        ks, ke = int(indptr[i]), int(indptr[i + 1])
        cache[offset : offset + length] = kv[ks:ke]
        index[ks:ke] = torch.arange(
            offset, offset + length, device="cuda", dtype=torch.int32
        )
        offset += ((length + 15) // 16) * 16
    max_q = max(query_lengths)
    for heads in dict.fromkeys([64, q.shape[1]]):
        query = (
            q
            if heads == q.shape[1]
            else q[:, : args.heads].repeat(1, 2, 1)[:, :heads].contiguous()
        )
        kw = {
            k: device(old[k])
            for k in ["sm_scale", "q_scale", "kv_scale", "causal"]
            if k in old
        }
        kw.update(
            qo_indptr=qo,
            kv_indptr=indptr,
            kv_last_page_lens=torch.ones(
                len(lengths), device="cuda", dtype=torch.int32
            ),
            max_seqlen_q=max_q,
        )
        work = [
            torch.empty(shape, device="cuda", dtype=dtype)
            for shape, dtype in aiter.get_mla_metadata_info_v1(
                len(lengths),
                max_q,
                heads,
                q.dtype,
                kv.dtype,
                is_sparse=False,
                fast_mode=True,
            )
        ]
        aiter.get_mla_metadata_v1(
            qo,
            indptr,
            kw["kv_last_page_lens"],
            heads,
            1,
            old.get("causal", True),
            work[0],
            work[2],
            work[1],
            work[3],
            work[4],
            work[5],
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=max_q,
            uni_seqlen_qo=max_q if len(set(query_lengths)) == 1 else -1,
            fast_mode=True,
            dtype_q=q.dtype,
            dtype_kv=kv.dtype,
        )
        kw.update(
            zip(
                [
                    "work_meta_data",
                    "work_indptr",
                    "work_info_set",
                    "reduce_indptr",
                    "reduce_final_map",
                    "reduce_partial_map",
                ],
                work,
            )
        )
        baseline = None
        for repeat in range(3):
            out = torch.full(
                (q.shape[0], heads, 512),
                float("nan"),
                device="cuda",
                dtype=old["o"].dtype,
            )
            mla_decode_fwd(
                q=query,
                kv_buffer=cache.view(-1, 1, 1, cache.shape[-1]),
                o=out,
                kv_indices=index,
                **kw,
            )
            result["cases"].append(
                {
                    "heads": heads,
                    "rebuild_metadata": True,
                    "repeat": repeat,
                    "output": error(
                        out[:, : args.heads], expected[:, : args.heads], 0.02, 0.05
                    ),
                    "repeat_stability": error(
                        out[:, : args.heads],
                        out[:, : args.heads] if baseline is None else baseline,
                        0.0,
                        0.0,
                        1e-6,
                    ),
                }
            )
            if baseline is None:
                baseline = out[:, : args.heads].clone()
            write_json(args.output, result)
    torch.save(
        {"reference": expected.cpu(), "last_output": out.cpu()},
        args.output + ".tensors.pt",
    )
