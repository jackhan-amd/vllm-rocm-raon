# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Raon vector-decay RDA, adapted from AITER's FlyDSL GDN verify schedule.

See ROCM-DECODE-PROVENANCE.md for the source hash, changes and MIT license.
Conv uses AITER's FlyDSL API; this module owns recurrence and gated RMSNorm.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.gdn_decode_verify import _load_vec, _store_vec
from flydsl.expr import const_expr, range_constexpr


def _dtype(name):
    return {"torch.bfloat16": fx.BFloat16, "torch.float32": fx.Float32}[name]


@functools.lru_cache(maxsize=128)
def create_recurrence(
    heads,
    steps,
    speculative,
    state_stride,
    index_stride,
    index_step,
    beta_stride,
    ratio_stride,
    lower_bound,
):
    # 16 lanes cooperate on K=128; each block owns 16 independent V rows.
    # Four FP32 values per vector preserve AITER's 128-bit state accesses.
    ksplit, klocal, half, block = 16, 8, 4, 256
    vtile, d = block // ksplit, 128

    @flyc.jit
    def group_sum(value):
        value = value + fx.gpu.shuffle_xor(value, 1, 64)
        value = value + fx.gpu.shuffle_xor(value, 2, 64)
        value = value + fx.gpu.shuffle_xor(value, 4, 64)
        return value + fx.gpu.shuffle_xor(value, 8, 64)

    @flyc.jit
    def exp_vector(value):
        return fx.Vector.from_elements(
            [fx.exp(value[j]) for j in range_constexpr(half)], fx.Float32
        )

    @flyc.kernel
    def recurrence(
        qkv: fx.Tensor,
        gate: fx.Tensor,
        beta: fx.Tensor,
        ratio: fx.Tensor,
        a_log: fx.Tensor,
        bias: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        accepted: fx.Tensor,
        cu: fx.Tensor,
        out: fx.Tensor,
    ):
        tid = fx.Int64(fx.thread_idx.x)
        head = fx.Int64(fx.block_idx.x)
        vrow = fx.Int64(fx.block_idx.y) * vtile + tid // ksplit
        seq = fx.Int64(fx.block_idx.z)
        kpart = tid % ksplit
        kbase = kpart * klocal
        bos, eos = fx.Int64(cu[seq]), fx.Int64(cu[seq + 1])
        checkpoint = fx.Int64(0)
        acceptance_valid = bos < eos
        if const_expr(speculative):
            count = fx.Int64(accepted[seq])
            acceptance_valid = acceptance_valid & (count >= 0) & (count <= steps)
            checkpoint = ((count > 0) & (count <= steps)).select(count - 1, 0)
        # These offsets already include physical row/column strides. Tensor's
        # scalar logical coordinate would remap a rank-two SD index matrix.
        ib = fx.rocdl.make_buffer_tensor(indices)
        read_slot = fx.Int64(
            _load_vec(ib, seq * index_stride + checkpoint * index_step, 1, fx.Int32)[0]
        )

        def state_at(slot):
            row = fx.make_view(
                fx.get_iter(state) + slot * state_stride,
                fx.make_layout(heads * d * d, 1),
            )
            return fx.rocdl.make_buffer_tensor(row)

        qb = fx.rocdl.make_buffer_tensor(qkv)
        gb = fx.rocdl.make_buffer_tensor(gate)
        bb = fx.rocdl.make_buffer_tensor(beta)
        rb = fx.rocdl.make_buffer_tensor(ratio)
        db = fx.rocdl.make_buffer_tensor(bias)
        ob = fx.rocdl.make_buffer_tensor(out)

        def do_sequence():
            ss = state_at(read_slot)
            sbase = (head * d + vrow) * d + kbase
            h = [
                _load_vec(ss, sbase + j * half, half, fx.Float32)
                for j in range(klocal // half)
            ]
            exp_a = fx.exp(fx.Float32(a_log[head]))
            bias_vec = [
                _load_vec(db, head * d + kbase + j * half, half, fx.Float32)
                for j in range(klocal // half)
            ]
            live = acceptance_valid
            for t in range_constexpr(steps):
                tok = bos + t
                if const_expr(speculative):
                    write_slot = fx.Int64(
                        _load_vec(ib, seq * index_stride + t * index_step, 1, fx.Int32)[
                            0
                        ]
                    )
                else:
                    write_slot = read_slot
                live = live & (tok < eos) & (write_slot >= (1 if speculative else 0))
                # A NULL/PAD suffix must neither load past packed inputs nor
                # advance the held state. Token zero is live somewhere because
                # the host rejects an entirely empty packed batch.
                safe_tok = live.select(tok, 0)
                qbase = safe_tok * (3 * heads * d) + head * d + kbase
                gbase = (safe_tok * heads + head) * d + kbase
                q = [
                    _load_vec(qb, qbase + j * half, half, fx.BFloat16).to(fx.Float32)
                    for j in range(klocal // half)
                ]
                k = [
                    _load_vec(qb, qbase + heads * d + j * half, half, fx.BFloat16).to(
                        fx.Float32
                    )
                    for j in range(klocal // half)
                ]
                qsum = group_sum(
                    (q[0] * q[0] + q[1] * q[1]).reduce(
                        fx.ReductionOp.ADD, fx.Float32(0.0)
                    )
                )
                ksum = group_sum(
                    (k[0] * k[0] + k[1] * k[1]).reduce(
                        fx.ReductionOp.ADD, fx.Float32(0.0)
                    )
                )
                qnorm = (d**-0.5) / fx.sqrt(qsum + 1e-6)
                knorm = fx.Float32(1.0) / fx.sqrt(ksum + 1e-6)
                q = [x * qnorm for x in q]
                k = [x * knorm for x in k]
                decay = []
                for j in range_constexpr(klocal // half):
                    raw = _load_vec(gb, gbase + j * half, half, fx.BFloat16).to(
                        fx.Float32
                    )
                    z = (raw + bias_vec[j]) * exp_a
                    sigmoid = fx.Vector.filled(half, 1.0, fx.Float32) / (
                        exp_vector(-z) + 1.0
                    )
                    decay.append(exp_vector(sigmoid * lower_bound))
                hd = [h[j] * decay[j] for j in range(klocal // half)]
                predicted = group_sum(
                    (hd[0] * k[0] + hd[1] * k[1]).reduce(
                        fx.ReductionOp.ADD, fx.Float32(0.0)
                    )
                )
                raw_beta = fx.Float32(
                    _load_vec(bb, safe_tok * beta_stride + head, 1, fx.BFloat16)[0]
                )
                erase = fx.Float32(1.0) / (1.0 + fx.exp(-raw_beta))
                rv = fx.Float32(
                    _load_vec(rb, safe_tok * ratio_stride + head, 1, fx.Float32)[0]
                )
                v = fx.Float32(
                    _load_vec(
                        qb,
                        safe_tok * (3 * heads * d) + 2 * heads * d + head * d + vrow,
                        1,
                        fx.BFloat16,
                    )[0]
                )
                v = (v * rv).to(fx.BFloat16).to(fx.Float32)
                update = (v - predicted) * erase
                next_h = [hd[j] + k[j] * update for j in range(klocal // half)]
                h = [live.select(next_h[j], h[j]) for j in range(klocal // half)]
                value = group_sum(
                    (h[0] * q[0] + h[1] * q[1]).reduce(
                        fx.ReductionOp.ADD, fx.Float32(0.0)
                    )
                )

                def store_step(_slot=write_slot, _h=h, _tok=tok, _value=value):
                    dst = state_at(_slot)
                    for j in range_constexpr(klocal // half):
                        _store_vec(dst, sbase + j * half, _h[j], fx.Float32)

                    def store_output():
                        _store_vec(
                            ob,
                            (_tok * heads + head) * d + vrow,
                            fx.Vector.from_elements(
                                [_value.to(fx.BFloat16)], fx.BFloat16
                            ),
                            fx.BFloat16,
                        )

                    if kpart == 0:
                        store_output()

                if live:
                    store_step()

        if acceptance_valid & (read_slot >= (1 if speculative else 0)):
            do_sequence()

    @flyc.jit
    def launch(
        qkv: fx.Tensor,
        gate: fx.Tensor,
        beta: fx.Tensor,
        ratio: fx.Tensor,
        a_log: fx.Tensor,
        bias: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        accepted: fx.Tensor,
        cu: fx.Tensor,
        out: fx.Tensor,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        recurrence(
            qkv, gate, beta, ratio, a_log, bias, state, indices, accepted, cu, out
        ).launch(
            grid=(heads, d // vtile, batch),
            block=(block,),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=8)
def create_gated_norm(heads, eps, gate_stride, weight_dtype):
    wd = _dtype(weight_dtype)

    @flyc.kernel
    def norm(value: fx.Tensor, gate: fx.Tensor, weight: fx.Tensor, out: fx.Tensor):
        lane = fx.Int64(fx.thread_idx.x)
        row = fx.Int64(fx.block_idx.x)
        tok = row // heads
        head = row % heads
        offset = lane * 2
        vb = fx.rocdl.make_buffer_tensor(value)
        gb = fx.rocdl.make_buffer_tensor(gate)
        wb = fx.rocdl.make_buffer_tensor(weight)
        ob = fx.rocdl.make_buffer_tensor(out)
        x = _load_vec(vb, row * 128 + offset, 2, fx.BFloat16).to(fx.Float32)
        sq = (x * x).reduce(fx.ReductionOp.ADD, fx.Float32(0.0))
        sq = sq + fx.gpu.shuffle_xor(sq, 1, 64)
        sq = sq + fx.gpu.shuffle_xor(sq, 2, 64)
        sq = sq + fx.gpu.shuffle_xor(sq, 4, 64)
        sq = sq + fx.gpu.shuffle_xor(sq, 8, 64)
        sq = sq + fx.gpu.shuffle_xor(sq, 16, 64)
        sq = sq + fx.gpu.shuffle_xor(sq, 32, 64)
        inv = fx.Float32(1.0) / fx.sqrt(sq / 128.0 + eps)
        g = _load_vec(gb, tok * gate_stride + head * 128 + offset, 2, fx.BFloat16).to(
            fx.Float32
        )
        w = _load_vec(wb, offset, 2, wd).to(fx.Float32)
        sigmoid = fx.Vector.from_elements(
            [fx.Float32(1.0) / (1.0 + fx.exp(-g[j])) for j in range_constexpr(2)],
            fx.Float32,
        )
        _store_vec(
            ob, row * 128 + offset, (x * inv * w * sigmoid).to(fx.BFloat16), fx.BFloat16
        )

    @flyc.jit
    def launch(
        value: fx.Tensor,
        gate: fx.Tensor,
        weight: fx.Tensor,
        out: fx.Tensor,
        tokens: fx.Int32,
        stream: fx.Stream,
    ):
        norm(value, gate, weight, out).launch(
            grid=(tokens * heads,), block=(64,), stream=stream
        )

    return launch
