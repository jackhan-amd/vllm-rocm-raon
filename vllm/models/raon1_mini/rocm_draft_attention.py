# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental noncausal DFlash attention: FlyDSL gather and native HIP Opus.

Each query aliases its exact sliding-window KV span through Opus's padded
sequence offsets. Q/K/V remain BF16; historical FP8 cache entries are dequantized
once, while the current K/V suffix retains its original BF16 precision.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _create(
    batch, capacity, tokens, left, page, table_stride, strides, new_strides, fp8
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.gdn_decode_verify import _load_vec, _store_vec
    from flydsl.expr import T, const_expr

    ks0, ks1, ks2, vs0, vs1, vs2 = strides
    key_stride, value_stride = new_strides

    @flyc.jit
    def load_cache(buf, offset):
        if const_expr(fp8):
            word = _load_vec(buf, offset, 4, fx.Uint8).bitcast(fx.Int32)[0]
            lo = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, word, False))
            hi = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, word, True))
            return lo.shuffle(hi, [0, 1, 2, 3])
        else:
            return _load_vec(buf, offset, 4, fx.BFloat16).to(fx.Float32)

    @flyc.kernel
    def metadata(
        cuq: fx.Tensor,
        lens: fx.Tensor,
        logical: fx.Tensor,
        physical: fx.Tensor,
        queries: fx.Tensor,
        sequence_logical: fx.Tensor,
        sequence_physical: fx.Tensor,
    ):
        tok = fx.Int64(fx.block_idx.x) * 64 + fx.Int64(fx.thread_idx.x)
        if tok <= tokens:
            prefix = fx.Int64(0)
            chosen_prefix = fx.Int64(0)
            chosen_start = fx.Int64(batch * capacity)
            found = fx.Int64(0)
            actual = fx.Int64(cuq[batch])
            kvprefix = fx.Int64(0)
            for b in range(fx.Int64(0), fx.Int64(batch), fx.Int64(1)):
                bos, eos = fx.Int64(cuq[b]), fx.Int64(cuq[b + 1])
                count = eos - bos
                size = fx.min(fx.Int64(lens[b]), left + count)
                size = (count > 0).select(size, 0)
                if tok == b:
                    sequence_logical[tok] = kvprefix.to(fx.Int32)
                kvprefix = kvprefix + size
                delta = size - count - left
                positive = fx.max(fx.Int64(0), count + delta - 1)
                total = count * size - positive * (positive + 1) // 2
                if (tok >= bos) & (tok < eos):
                    i = tok - bos
                    n = fx.max(fx.Int64(0), i + delta - 1)
                    chosen_prefix = prefix + i * size - n * (n + 1) // 2
                    chosen_start = b * capacity + fx.max(fx.Int64(0), delta + i)
                    found = fx.Int64(1)
                prefix = prefix + total
            if found == 0:
                chosen_prefix = prefix + tok - actual
            logical[tok] = chosen_prefix.to(fx.Int32)
            physical[tok] = chosen_start.to(fx.Int32)
            queries[tok] = tok.to(fx.Int32)
            if tok <= batch:
                sequence_physical[tok] = (tok * capacity).to(fx.Int32)
                if tok == batch:
                    sequence_logical[tok] = kvprefix.to(fx.Int32)

    tiles = (capacity * 640 + 511) // 512
    grid_tiles = min(tiles, max(64, 1024 // batch))

    @flyc.jit
    def gather_tile(
        kc: fx.Tensor,
        vc: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        table: fx.Tensor,
        cuq: fx.Tensor,
        lens: fx.Tensor,
        kscale: fx.Tensor,
        vscale: fx.Tensor,
        kout: fx.Tensor,
        vout: fx.Tensor,
        b: fx.Int64,
        element: fx.Int64,
    ):
        row, col = element // 640, element % 640
        kb, vb = fx.rocdl.make_buffer_tensor(kout), fx.rocdl.make_buffer_tensor(vout)
        if b < batch:
            bos, eos = fx.Int64(cuq[b]), fx.Int64(cuq[b + 1])
            count, size = eos - bos, fx.Int64(lens[b])
            first = fx.max(fx.Int64(0), size - count - left)
            if (count > 0) & (row < size - first):
                position = first + row
                offset = (b * capacity + row) * 640 + col
                if (position < size - count) | (count == 1):
                    block = fx.Int64(table[b * table_stride + position // page])
                    slot, head, dim = position % page, col // 128, col % 128
                    ck, cv = (
                        fx.rocdl.make_buffer_tensor(
                            fx.make_view(
                                fx.get_iter(kc) + block * ks0,
                                fx.make_layout((page - 1) * ks1 + 4 * ks2 + 128, 1),
                            )
                        ),
                        fx.rocdl.make_buffer_tensor(
                            fx.make_view(
                                fx.get_iter(vc) + block * vs0,
                                fx.make_layout((page - 1) * vs1 + 4 * vs2 + 128, 1),
                            )
                        ),
                    )
                    kval = load_cache(ck, slot * ks1 + head * ks2 + dim)
                    vval = load_cache(cv, slot * vs1 + head * vs2 + dim)
                    if const_expr(fp8):
                        kval = kval * fx.Float32(kscale[0])
                        vval = vval * fx.Float32(vscale[0])
                    _store_vec(kb, offset, kval.to(fx.BFloat16), fx.BFloat16)
                    _store_vec(vb, offset, vval.to(fx.BFloat16), fx.BFloat16)
                else:
                    token = bos + position - size + count
                    ksource = token * key_stride + col
                    vsource = token * value_stride + col
                    nk, nv = (
                        fx.rocdl.make_buffer_tensor(key),
                        fx.rocdl.make_buffer_tensor(value),
                    )
                    _store_vec(
                        kb, offset, _load_vec(nk, ksource, 4, fx.BFloat16), fx.BFloat16
                    )
                    _store_vec(
                        vb, offset, _load_vec(nv, vsource, 4, fx.BFloat16), fx.BFloat16
                    )
        elif row == 0:
            zero = fx.Vector.filled(4, 0.0, fx.BFloat16)
            _store_vec(kb, batch * capacity * 640 + col, zero, fx.BFloat16)
            _store_vec(vb, batch * capacity * 640 + col, zero, fx.BFloat16)

    @flyc.kernel
    def gather(
        kc: fx.Tensor,
        vc: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        table: fx.Tensor,
        cuq: fx.Tensor,
        lens: fx.Tensor,
        kscale: fx.Tensor,
        vscale: fx.Tensor,
        kout: fx.Tensor,
        vout: fx.Tensor,
    ):
        b = fx.Int64(fx.block_idx.x)
        lane = fx.Int64(fx.thread_idx.x) * 4
        if b < batch:
            count = fx.Int64(cuq[b + 1]) - fx.Int64(cuq[b])
            retained = fx.min(fx.Int64(lens[b]), left + count)
            live_tiles = (count > 0).select((retained * 640 + 511) // 512, 0)
            for tile in range(
                fx.Int64(fx.block_idx.y), live_tiles, fx.Int64(grid_tiles)
            ):
                gather_tile(
                    kc,
                    vc,
                    key,
                    value,
                    table,
                    cuq,
                    lens,
                    kscale,
                    vscale,
                    kout,
                    vout,
                    b,
                    tile * 512 + lane,
                )
        elif fx.block_idx.y == 0:
            gather_tile(
                kc,
                vc,
                key,
                value,
                table,
                cuq,
                lens,
                kscale,
                vscale,
                kout,
                vout,
                b,
                lane,
            )
            gather_tile(
                kc,
                vc,
                key,
                value,
                table,
                cuq,
                lens,
                kscale,
                vscale,
                kout,
                vout,
                b,
                512 + lane,
            )

    @flyc.jit
    def launch(
        kc: fx.Tensor,
        vc: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        table: fx.Tensor,
        cuq: fx.Tensor,
        lens: fx.Tensor,
        kscale: fx.Tensor,
        vscale: fx.Tensor,
        kout: fx.Tensor,
        vout: fx.Tensor,
        logical: fx.Tensor,
        physical: fx.Tensor,
        queries: fx.Tensor,
        sequence_logical: fx.Tensor,
        sequence_physical: fx.Tensor,
        stream: fx.Stream,
    ):
        metadata(
            cuq, lens, logical, physical, queries, sequence_logical, sequence_physical
        ).launch(grid=((tokens + 64) // 64,), block=(64,), stream=stream)
        gather(kc, vc, key, value, table, cuq, lens, kscale, vscale, kout, vout).launch(
            grid=(batch + 1, grid_tiles), block=(128,), stream=stream
        )

    return launch


def _flat_storage(view):
    span = 1 + sum((n - 1) * s for n, s in zip(view.shape, view.stride()))
    return view.as_strided((span,), (1,))


class DraftAttentionWorkspace:
    """Fixed-shape scratch; all sequence values are refreshed on graph replay."""

    def __init__(
        self, batch, tokens, max_query_len, device, window=2048, provider="opus"
    ):
        if provider not in ("opus", "ck"):
            raise ValueError("Native draft provider must be opus or ck")
        self.provider = provider
        self.max_query_len = max_query_len
        self.batch, self.tokens = batch, tokens
        self.left = window - 1
        self.capacity = self.left + max_query_len
        shape = (batch * self.capacity + 1, 5, 128)
        self.key = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.value = torch.empty_like(self.key)
        self.logical, self.physical, self.queries = [
            torch.empty(tokens + 1, dtype=torch.int32, device=device) for _ in range(3)
        ]
        self.sequence_logical, self.sequence_physical = [
            torch.empty(batch + 1, dtype=torch.int32, device=device) for _ in range(2)
        ]

    def __call__(
        self,
        query,
        key,
        value,
        kc,
        vc,
        table,
        cuq,
        lens,
        kscale,
        vscale,
        out,
        scale=None,
    ):
        if (
            query.shape != (self.tokens, 20, 128)
            or query.dtype != torch.bfloat16
            or query.stride()[1:] != (128, 1)
            or key.shape != (self.tokens, 5, 128)
            or value.shape != key.shape
            or key.stride()[1:] != (128, 1)
            or value.stride()[1:] != (128, 1)
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or kc.shape != vc.shape
            or kc.shape[2:] != (5, 128)
            or kc.stride(-1) != 1
            or vc.stride(-1) != 1
            or kc.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or kc.dtype != vc.dtype
            or not table.is_contiguous()
            or table.dtype != torch.int32
            or cuq.dtype != torch.int32
            or lens.dtype != torch.int32
            or lens.numel() != self.batch
            or cuq.numel() != self.batch + 1
        ):
            raise ValueError(
                "Native draft candidate requires BF16 20Q/5KV D128 and NHD cache"
            )
        import aiter
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        _run_compiled(
            _create(
                self.batch,
                self.capacity,
                self.tokens,
                self.left,
                kc.shape[1],
                table.stride(0),
                (*kc.stride()[:3], *vc.stride()[:3]),
                (key.stride(0), value.stride(0)),
                kc.dtype == torch.float8_e4m3fn,
            ),
            kc.view(torch.uint8) if kc.dtype == torch.float8_e4m3fn else kc,
            vc.view(torch.uint8) if vc.dtype == torch.float8_e4m3fn else vc,
            _flat_storage(key),
            _flat_storage(value),
            table.view(-1),
            cuq,
            lens,
            kscale.reshape(1),
            vscale.reshape(1),
            self.key.view(-1),
            self.value.view(-1),
            self.logical,
            self.physical,
            self.queries,
            self.sequence_logical,
            self.sequence_physical,
            torch.cuda.current_stream(query.device).cuda_stream,
        )
        if self.provider == "ck":
            return aiter.mha_varlen_fwd(
                query,
                self.key,
                self.value,
                cuq,
                self.sequence_logical,
                self.max_query_len,
                self.capacity,
                1,
                0.0,
                scale or 128**-0.5,
                0.0,
                False,
                False,
                self.left if self.capacity > self.left + 1 else -1,
                -1,
                0,
                False,
                False,
                out=out,
                cu_seqlens_k_padded=self.sequence_physical,
            )[0]
        return aiter.fmha_fwd_bf16_opus_varlen_fwd(
            query,
            self.key,
            self.value,
            scale or 128**-0.5,
            False,
            self.queries,
            self.logical,
            1,
            self.capacity,
            out=out,
            seqstart_k_pad=self.physical,
            window_size_left=-1,
        )
