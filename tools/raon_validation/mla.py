# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify FP8 MLA blocks against dense FP32 attention of dequantized inputs."""

import inspect
from pathlib import Path

import torch

from .common import error, provenance, write_json


def run(args, result):
    import aiter
    from aiter.mla import mla_decode_fwd

    result["source"] = provenance(mla_decode_fwd)
    Path(args.output + ".native.py").write_text(inspect.getsource(mla_decode_fwd))
    for length in [17, 65, 257]:
        for query_length in [1, 2, 4, 8]:
            h = args.mla_heads
            q = (torch.randn(query_length, h, 576, device="cuda") * 0.25).to(
                torch.float8_e4m3fn
            )
            kv = (torch.randn(length, 576, device="cuda") * 0.25).to(
                torch.float8_e4m3fn
            )
            indices = torch.randperm(length, device="cuda").int()
            if args.kv_padding != "exact":
                capacity = ((length + 15) // 16) * 16
                padded = torch.empty(capacity, 576, device="cuda", dtype=kv.dtype)
                pad_value = {"zero": 0.0, "finite": 0.25, "nan": float("nan")}[
                    args.kv_padding
                ]
                padded.fill_(pad_value)
                padded[:length] = kv
                kv = padded
                all_indices = torch.arange(capacity, device="cuda", dtype=torch.int32)
                all_indices[:length] = indices
                indices = all_indices[:length]
            qo = torch.tensor([0, query_length], device="cuda", dtype=torch.int32)
            indptr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            last = torch.ones(1, device="cuda", dtype=torch.int32)
            work = [
                torch.empty(shape, device="cuda", dtype=dtype)
                for shape, dtype in aiter.get_mla_metadata_info_v1(
                    1,
                    query_length,
                    h,
                    q.dtype,
                    kv.dtype,
                    is_sparse=False,
                    fast_mode=True,
                )
            ]
            aiter.get_mla_metadata_v1(
                qo,
                indptr,
                last,
                h,
                1,
                True,
                work[0],
                work[2],
                work[1],
                work[3],
                work[4],
                work[5],
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=query_length,
                uni_seqlen_qo=query_length,
                fast_mode=True,
                dtype_q=q.dtype,
                dtype_kv=kv.dtype,
            )
            out = torch.empty(query_length, h, 512, device="cuda", dtype=torch.bfloat16)
            qs, ks = (
                torch.tensor([2.0], device="cuda"),
                torch.tensor([3.0], device="cuda"),
            )
            mla_decode_fwd(
                q,
                kv.view(-1, 1, 1, 576),
                out,
                qo,
                indptr,
                indices,
                last,
                query_length,
                sm_scale=192**-0.5,
                q_scale=qs,
                kv_scale=ks,
                causal=True,
                work_meta_data=work[0],
                work_indptr=work[1],
                work_info_set=work[2],
                reduce_indptr=work[3],
                reduce_final_map=work[4],
                reduce_partial_map=work[5],
            )
            scratch_check = None
            if args.scratch_poison:
                baseline = out.clone()
                original_empty = torch.empty

                def poisoned_empty(*sizes, _original=original_empty, **kwargs):
                    value = _original(*sizes, **kwargs)
                    if (
                        value.dtype == torch.float32
                        and value.ndim == 4
                        and value.shape[1] == 1
                        and value.shape[-1] in (1, 512)
                    ):
                        value.fill_(float("nan"))
                    return value

                torch.empty = poisoned_empty
                try:
                    mla_decode_fwd(
                        q,
                        kv.view(-1, 1, 1, 576),
                        out,
                        qo,
                        indptr,
                        indices,
                        last,
                        query_length,
                        sm_scale=192**-0.5,
                        q_scale=qs,
                        kv_scale=ks,
                        causal=True,
                        work_meta_data=work[0],
                        work_indptr=work[1],
                        work_info_set=work[2],
                        reduce_indptr=work[3],
                        reduce_final_map=work[4],
                        reduce_partial_map=work[5],
                    )
                finally:
                    torch.empty = original_empty
                # Only the model's first 40 heads are live; padded heads are
                # retained separately and must not explain a live-head failure.
                scratch_check = error(out[:, :40], baseline[:, :40], 0.0, 0.0, 1e-6)
                torch.save(
                    {
                        "q": q,
                        "kv": kv,
                        "indices": indices,
                        "baseline": baseline,
                        "poisoned": out.clone(),
                        "metadata": work,
                    },
                    args.output + f".scratch.L{length}.Q{query_length}.pt",
                )
            qd, kd = q.float() * qs, kv.float()[indices.long()] * ks
            scores = torch.einsum("qhd,kd->qhk", qd, kd) * 192**-0.5
            mask = torch.arange(length, device="cuda")[None, :] > (
                length
                - query_length
                + torch.arange(query_length, device="cuda")[:, None]
            )
            scores.masked_fill_(mask[:, None, :], -torch.inf)
            expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), kd[:, :512])
            diagnostics = []
            # Diagnostic only: isolate the native FP8 probability matmul rounding.
            # The dense-FP32 acceptance threshold above remains unchanged.
            global_max = scores.amax(-1, keepdim=True)
            total = (scores - global_max).exp().sum(-1, keepdim=True)
            for tile in [length, 16, 32, 64, 128]:
                for scale in [1.0, 256.0, 448.0]:
                    numerator = torch.zeros_like(expected)
                    for start in range(0, length, tile):
                        local = scores[..., start : start + tile]
                        maximum = local.amax(-1, keepdim=True)
                        unnorm = torch.where(
                            torch.isfinite(maximum), (local - maximum).exp(), 0.0
                        )
                        rounded = (unnorm * scale).to(
                            torch.float8_e4m3fn
                        ).float() / scale
                        weights = rounded * (maximum - global_max).exp()
                        numerator += torch.einsum(
                            "qhk,kd->qhd", weights, kd[start : start + tile, :512]
                        )
                    candidate = numerator / total
                    e = error(out[:, :40], candidate[:, :40], 0.02, 0.05)
                    diagnostics.append(
                        {
                            "tile": tile,
                            "probability_scale": scale,
                            "relative_rms": e.get("relative_rms"),
                        }
                    )
            fit = float(
                (out[:, :40].float() * expected[:, :40]).sum()
                / expected[:, :40].square().sum()
            )
            if not bool(torch.isfinite(out).all()):
                fit = None
            torch.save(
                {
                    "q": q,
                    "kv": kv,
                    "indices": indices,
                    "actual": out,
                    "expected": expected,
                    "q_scale": qs,
                    "kv_scale": ks,
                },
                args.output + f".L{length}.Q{query_length}.pt",
            )
            result["cases"].append(
                {
                    "kv_length": length,
                    "query_length": query_length,
                    "real_heads": 40,
                    "padded_heads": h,
                    "output": error(out[:, :40], expected[:, :40], 0.02, 0.05),
                    "best_fit_scale": fit,
                    **(
                        {"scratch_independence": scratch_check} if scratch_check else {}
                    ),
                    "probability_rounding_diagnostic": sorted(
                        diagnostics,
                        key=lambda row: row["relative_rms"]
                        if row["relative_rms"] is not None
                        else 1e30,
                    )[:3],
                }
            )
            write_json(args.output, result)
