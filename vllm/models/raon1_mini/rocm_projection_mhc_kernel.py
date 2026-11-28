# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Raon attention projection plus four-stream mHC, using AITER's skinny MFMA.

Derived from aiter/ops/flydsl/kernels/gemm_a16w16_skinny_gfx950.py.
The DMA/LDS/MFMA schedule is retained; only the final store becomes the Raon
mHC epilogue. There are no cross-workgroup waits or global split-K atomics.
This is a local fusion building block, not a persistent full-layer monokernel.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import T

WAVE_SIZE = 64
DMA_VEC = 8
MFMA_K = 32
GFX950_NUM_XCDS = 8
REDUCE_ROW_PAD = 4


@functools.lru_cache(maxsize=256)
def _compile(
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    k_waves: int,
    mix_one: bool,
    device_index: int,
):
    TA, TN, KC, K_WAVES = block_m, block_n, block_k, k_waves
    # TA = A rows staged; TM = MFMA rows (TA rounded up to 16). MFMA rows in
    # [TA, TM) read other LDS rows, but they only feed C rows >= m, which the
    # C descriptor bounds drop.
    MF = (TA + 15) // 16
    TM = MF * 16
    NF = (TN + 15) // 16
    TNP = NF * 16
    N_BLOCKS = n // TN
    KS = k // K_WAVES
    CHUNKS = KS // KC
    ST = min(stages, CHUNKS)
    CPR = KC // DMA_VEC  # 16-byte chunks per staged row
    ROWS_PER_INST = WAVE_SIZE // CPR
    R = TA + TN  # rows per staged chunk: A rows, then W rows
    IPC = R // ROWS_PER_INST  # DMA instructions per chunk
    A_INSTS = TA // ROWS_PER_INST
    STEPS = KC // MFMA_K
    STAGE_ELEMS = R * KC
    WAVE_ELEMS = ST * STAGE_ELEMS
    BLOCK_THREADS = K_WAVES * WAVE_SIZE
    LDS_ELEMS = K_WAVES * WAVE_ELEMS
    # Operand-swapped MFMA (C^T tile) with accumulator-order partials: one MFMA
    # row tile only.
    SWAPPED = TA <= 16
    TNR = TNP + REDUCE_ROW_PAD
    # (mf, 4-column group, row) output slots of the swapped epilogue, one per thread
    EPI_SLOTS = MF * (TN // 4) * 16
    VEC_PER_ROW = TN // DMA_VEC
    OUT_VECS = TM * VEC_PER_ROW
    ROW_GROUP = max(1, 16 // CPR)  # rows sharing one 256-byte bank line
    XCD = GFX950_NUM_XCDS if N_BLOCKS % GFX950_NUM_XCDS == 0 else 0

    def swz(r, c):
        return c ^ ((r // ROW_GROUP) % CPR)

    @fx.struct
    class SkinnyLdsShared:
        buf: fx.Array[fx.BFloat16, LDS_ELEMS, 16]

    KERNEL_NAME = f"raon_projection_mhc_t{TA}x{TN}x{KC}x{ST}_w1x1x{K_WAVES}"

    def mfma(a_fr, b_fr, acc):
        # raw intrinsic: the fragments come straight out of the swizzled LDS
        # ring with no tensor/partition form for fx.make_mma_atom to take
        return rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_fr, b_fr, acc, 0, 0, 0])

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def skinny_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        S: fx.Pointer,
        H: fx.Pointer,
        m: fx.Int32,
    ):
        acc_init = fx.Vector.filled(4, 0.0, fx.Float32)
        a_bytes = fx.Int64(m) * (k * 2)
        c_bytes = fx.Int64(m) * (4 * n * 2)
        a_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(A)), num_records_bytes=a_bytes
        )
        c_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(C)), num_records_bytes=c_bytes
        )
        s_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(S)), num_records_bytes=c_bytes
        )
        h_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(H)), num_records_bytes=fx.Int64(m) * 20 * 4
        )

        def store_mix(total, row, col, width):
            # The separate GEMM writes BF16 before the mHC kernel reads it.
            branch = total.to(fx.BFloat16).to(fx.Float32)
            if row < m:
                streams = [
                    buffer_ops.buffer_load(
                        s_rsrc,
                        fx.Int32((row * 4 + j) * n + col),
                        vec_width=width,
                        dtype=T.bf16,
                    ).to(fx.Float32)
                    for j in range_constexpr(4)
                ]
                for i in range_constexpr(4):
                    h = [
                        fx.Float32(
                            buffer_ops.buffer_load(
                                h_rsrc,
                                fx.Int32(row * 20 + 4 + i * 4 + j),
                                vec_width=1,
                                dtype=T.f32,
                            )
                        )
                        for j in range_constexpr(4)
                    ]
                    h = [fx.Vector.filled(width, value, fx.Float32) for value in h]
                    value = fx.fma(
                        h[3],
                        streams[3],
                        fx.fma(
                            h[2],
                            streams[2],
                            fx.fma(h[1], streams[1], h[0] * streams[0]),
                        ),
                    )
                    if mix_one:
                        value = (
                            (h[0] * streams[0] + h[1] * streams[1]) + h[2] * streams[2]
                        ) + h[3] * streams[3]
                    post = fx.Float32(
                        buffer_ops.buffer_load(
                            h_rsrc,
                            fx.Int32(row * 20 + i),
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                    value = value + post * branch
                    buffer_ops.buffer_store(
                        value.to(fx.BFloat16),
                        c_rsrc,
                        fx.Int32((row * 4 + i) * n + col),
                    )

        lds = fx.SharedAllocator().allocate(SkinnyLdsShared).peek()
        buf_ptr = lds.buf.ptr
        red_ptr = fx.recast_iter(fx.Float32, buf_ptr)

        tid = fx.Index(fx.thread_idx.x)
        wid = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        ln = lane % 16
        lg = lane // 16
        if const_expr(XCD > 0):
            # XCD-contiguous slabs: workgroups are dealt round-robin over the
            # XCDs, so give each XCD a dense run of N slabs.
            bid = fx.Index(fx.block_idx.x)
            n0 = ((bid % XCD) * (N_BLOCKS // XCD) + bid // XCD) * TN
        else:
            n0 = fx.Index(fx.block_idx.x) * TN
        k0 = wid * KS

        # B is based at this workgroup's N slab, not at the tensor: the DMA's
        # voffset is one 32-bit VGPR, so a tensor-based descriptor wraps as soon
        # as n0 * k * 2 passes 2 GiB (the offset is cast to a SIGNED i32) and the
        # load reads whatever happens to be mapped there. Folding n0 into the
        # 64-bit base leaves the voffset at most TN * k * 2 bytes to span, which
        # no supported config comes near, and num_records then bounds the slab
        # exactly -- a future offset bug returns zeros instead of faulting.
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(B)) + fx.Int64(n0) * (k * 2),
            num_records_bytes=TN * k * 2,
        )

        buf_i64 = fx.Int64(fx.ptrtoint(buf_ptr))
        wave_off = rocdl.readfirstlane(
            T.i64, fx.Int64(wid * (WAVE_ELEMS * 2)).ir_value()
        )
        wave_lds = buffer_ops.get_element_ptr(
            create_llvm_ptr(buf_i64, fx.AddressSpace.Shared), wave_off
        )

        # per-lane global byte offset of chunk 0, one per DMA instruction
        def chunk0_offset(i):
            p = lane + i * WAVE_SIZE
            r = p // CPR
            c = swz(r, p % CPR)
            if const_expr(i < A_INSTS):
                g = (r * k + k0 + c * DMA_VEC) * 2
            else:
                # relative to b_rsrc, which is already based at slab n0
                g = ((r - TA) * k + k0 + c * DMA_VEC) * 2
            return fx.Int32(g)

        vofs = [None] * IPC
        chunk_soff = [fx.Int32(j * KC * 2) for j in range_constexpr(CHUNKS)]

        def issue_chunk(j):
            stage = j % ST
            for i in range_constexpr(IPC):
                lds_p = buffer_ops.get_element_ptr(
                    wave_lds,
                    static_byte_offset=stage * STAGE_ELEMS * 2 + i * WAVE_SIZE * 16,
                )
                rsrc = a_rsrc if const_expr(i < A_INSTS) else b_rsrc
                cpol = "" if const_expr(i < A_INSTS) else " nt"
                if const_expr(vofs[i] is None):
                    # each chunk-0 DMA goes out as soon as its own offset is
                    # ready; the sched_barrier below keeps the remaining
                    # address math from being hoisted in front of it
                    vofs[i] = chunk0_offset(i)
                # the chunk's K offset goes in soffset: an instruction offset
                # would also shift the LDS destination of the DMA.
                # llvm.inline_asm is a raw dialect op, so the typed operands
                # have to be unwrapped here.
                llvm.InlineAsmOp(
                    None,
                    [lds_p, vofs[i].ir_value(), rsrc, chunk_soff[j].ir_value()],
                    "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, $3 offen"
                    + cpol
                    + " lds",
                    "s,v,s,s",
                    has_side_effects=True,
                )
                if const_expr(j == 0):
                    rocdl.sched_barrier(0)

        def read_frag(stage, row, s):
            c = s * 4 + lg
            slot = row * CPR + swz(row, c)
            return fx.ptr_load(
                buf_ptr
                + fx.Int64(wid * WAVE_ELEMS + (stage * STAGE_ELEMS) + slot * DMA_VEC),
                result_type=fx.Vector.make_type(DMA_VEC, fx.BFloat16),
            )

        for j in range_constexpr(ST):
            issue_chunk(j)
        rocdl.sched_barrier(0)

        accs = [[acc_init] * NF for _ in range_constexpr(MF)]
        for j in range_constexpr(CHUNKS):
            stage = j % ST
            younger = min(CHUNKS, j + ST) - 1 - j
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt vmcnt({younger * IPC})", "", has_side_effects=True
            )
            rocdl.sched_barrier(0)
            frags = []
            for s in range_constexpr(STEPS):
                a_fr = [read_frag(stage, ln + mf * 16, s) for mf in range_constexpr(MF)]
                b_fr = [
                    read_frag(stage, ln + (TA + nf * 16), s)
                    for nf in range_constexpr(NF)
                ]
                frags.append((a_fr, b_fr))
            if const_expr(j + ST < CHUNKS):
                # refill the stage just read once its ds_reads have landed
                llvm.InlineAsmOp(
                    None, [], "s_waitcnt lgkmcnt(0)", "", has_side_effects=True
                )
                issue_chunk(j + ST)
            rocdl.sched_barrier(0)
            for s in range_constexpr(STEPS):
                a_fr, b_fr = frags[s]
                for mf in range_constexpr(MF):
                    for nf in range_constexpr(NF):
                        if const_expr(SWAPPED):
                            # C^T: four consecutive N columns of one M row.
                            accs[mf][nf] = mfma(b_fr[nf], a_fr[mf], accs[mf][nf])
                        else:
                            accs[mf][nf] = mfma(a_fr[mf], b_fr[nf], accs[mf][nf])
            rocdl.sched_barrier(0)

        red_wave0 = wid * (WAVE_ELEMS // 2)
        if const_expr(SWAPPED):
            # partials in accumulator order, one 16-byte LDS write per fragment:
            # red[w][mf*NF+nf][lane][i] = C[mf*16 + lane%16][nf*16 + 4*(lane//16) + i]
            for mf in range_constexpr(MF):
                for nf in range_constexpr(NF):
                    if const_expr(nf * 16 < TN):
                        fx.ptr_store(
                            accs[mf][nf],
                            red_ptr
                            + fx.Int64(
                                red_wave0 + (mf * NF + nf) * (WAVE_SIZE * 4) + lane * 4
                            ),
                        )
            gpu.barrier()
            for base in range_constexpr(0, EPI_SLOTS, BLOCK_THREADS):
                sl = tid + base
                mf_s = sl // ((TN // 4) * 16)
                rem = sl % ((TN // 4) * 16)
                cg = rem // 16  # 4-column group
                rr = rem % 16

                def reduce_store(mf_s, cg, rr):
                    src = (mf_s * NF + cg // 4) * (WAVE_SIZE * 4) + (
                        (cg % 4) * 16 + rr
                    ) * 4
                    tot = None
                    for w in range_constexpr(K_WAVES):
                        v = fx.ptr_load(
                            red_ptr + fx.Int64(src + w * (WAVE_ELEMS // 2)),
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )
                        tot = v if const_expr(w == 0) else tot + v
                    store_mix(tot, mf_s * 16 + rr, n0 + cg * 4, 4)

                if const_expr(base + BLOCK_THREADS <= EPI_SLOTS):
                    reduce_store(mf_s, cg, rr)
                else:
                    if sl < EPI_SLOTS:
                        reduce_store(mf_s, cg, rr)
            return

        # partials -> this wave's own LDS region, row-major fp32: red[w][row][col]
        for mf in range_constexpr(MF):
            for nf in range_constexpr(NF):
                acc_v = fx.Vector(accs[mf][nf])
                for i in range_constexpr(4):
                    row = lg * 4 + (mf * 16 + i)
                    col = ln + nf * 16
                    fx.ptr_store(
                        acc_v[i], red_ptr + fx.Int64(red_wave0 + row * TNR + col)
                    )
        gpu.barrier()

        if tid < OUT_VECS:
            r = tid // VEC_PER_ROW
            c = (tid % VEC_PER_ROW) * DMA_VEC
            rc_off = r * TNR + c
            tot = None
            for w in range_constexpr(K_WAVES):
                v = fx.ptr_load(
                    red_ptr + fx.Int64(rc_off + w * (WAVE_ELEMS // 2)),
                    result_type=fx.Vector.make_type(DMA_VEC, fx.Float32),
                )
                tot = v if const_expr(w == 0) else tot + v
            store_mix(tot, r, n0 + c, DMA_VEC)

    @flyc.jit
    def launch_skinny_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        S: fx.Pointer,
        H: fx.Pointer,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        skinny_kernel._func.__name__ = KERNEL_NAME
        skinny_kernel(C, A, B, S, H, m).launch(
            grid=(N_BLOCKS, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    # the ring-fill DMAs need the kernel args; preloaded into SGPRs they no
    # longer wait on the kernarg s_load round trip
    launch_skinny_kernel.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_skinny_kernel
