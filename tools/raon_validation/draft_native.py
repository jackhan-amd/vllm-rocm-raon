# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent Torch attention and cache-byte gold for native DFlash2 kernels."""

import torch

from vllm.models.raon1_mini.rocm_draft_attention import DraftAttentionWorkspace

from .common import error, provenance


def run(args, result):
    result["kernel_source"] = provenance(DraftAttentionWorkspace.__call__)
    if args.window != 2048:
        raise ValueError("Native draft fixtures use the checkpoint's 2048-token window")
    cases = [
        ([8], [4096], True, 0),
        ([1, 8, 3], [2048, 4096, 17], True, 0),
        ([0, 8, 0], [0, 4000, 0], True, 3),
        ([8, 8, 8], [8, 2055, 4097], False, 0),
        ([1, 1, 1], [2047, 2048, 4096], True, 0),
        ([16, 1], [8192, 4096], False, 7),
        ([3, 8], [47, 2049], True, 0),
        ([8] * 8, [4096] * 8, True, 0),
    ]
    qlens, lengths, fp8, pad = cases[args.draft_case]
    provider = args.draft_provider
    if provider == "auto":
        provider = "ck" if all(x in (0, 8) for x in qlens) else "opus"
    if provider == "ck" and any(x not in (0, 8) for x in qlens):
        raise ValueError("CK diagnostic supports active query lengths of eight only")
    batch = len(qlens)
    n = sum(qlens) + pad
    maxq = max(qlens)
    page = 1280 if args.draft_case == 6 else 32
    nb = sum((s + page - 1) // page for s in lengths) + 4
    if args.draft_case == 7:
        nb = 2048
    torch.manual_seed(249 + args.draft_case)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    if args.draft_case == 7:
        pool = torch.empty_strided(
            (nb, 5, page, 256),
            (2949120, page * 256, 256, 1),
            device="cuda",
            dtype=dtype,
        )
    else:
        pool = torch.empty((nb, 5, page, 256), device="cuda", dtype=dtype)
    kc, vc = pool.transpose(1, 2).split(128, dim=-1)
    ks = torch.tensor(0.3, device="cuda")
    vs = torch.tensor(1.7, device="cuda")
    for t in (kc, vc):
        t.copy_(torch.randn(t.shape, device="cuda", dtype=torch.bfloat16).to(dtype))
    perm = torch.randperm(nb, device="cuda", dtype=torch.int64)
    width = max((s + page - 1) // page for s in lengths)
    table = torch.zeros((batch, width), device="cuda", dtype=torch.int32)
    off = 0
    for b, size in enumerate(lengths):
        count = (size + page - 1) // page
        table[b, :count] = perm[off : off + count].int()
        off += count
    cu = torch.tensor([0] + qlens, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    sl = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    q = torch.randn(n, 20, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn(n, 5, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k)
    if args.draft_case in (6, 7):
        packed = torch.cat((q, k, v), dim=1)
        q, k, v = packed.split((20, 5, 5), dim=1)
    # Native HIP writer must respect the strided hybrid NHD pool.
    from vllm import _custom_ops as ops

    slots = []
    for b, (count, size) in enumerate(zip(qlens, lengths)):
        for pos in range(size - count, size):
            slots.append(table[b, pos // page].long() * page + pos % page)
    slot_map = (
        torch.stack(slots).to(torch.int64)
        if slots
        else torch.empty(0, device="cuda", dtype=torch.int64)
    )
    slot_map = torch.cat(
        (slot_map, torch.full((pad,), -1, device="cuda", dtype=torch.int64))
    )
    before = pool.clone()
    ops.reshape_and_cache_flash(
        k, v, kc, vc, slot_map, "fp8_e4m3" if fp8 else "auto", ks, vs
    )
    for source, cache, scale in ((k, kc, ks), (v, vc, vs)):
        ids = slot_map[: sum(qlens)]
        expected = (
            (source[: sum(qlens)].float() / scale).to(dtype)
            if fp8
            else source[: sum(qlens)]
        )
        actual = cache[ids // page, ids % page]
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            "Native cache writer differs from torch quantizer"
        )
    written = torch.zeros(nb * page, device="cuda", dtype=torch.bool)
    written[slot_map[slot_map >= 0]] = True
    # Pool is HND; compare untouched token slots after restoring logical NHD order.
    assert torch.equal(
        pool.transpose(1, 2).reshape(nb * page, 5, 256)[~written].view(torch.uint8),
        before.transpose(1, 2).reshape(nb * page, 5, 256)[~written].view(torch.uint8),
    ), "Writer touched an unrelated slot"
    clean_pool = pool.clone()
    # Poison expired positions: the independent reference never includes them.
    for b, (count, size) in enumerate(zip(qlens, lengths)):
        first = max(0, size - count - 2047) if count else 0
        expired = torch.arange(first, device="cuda")
        ids = table[b, expired // page].long()
        offsets = expired % page
        if fp8:
            kc.view(torch.uint8)[ids, offsets] = 127
            vc.view(torch.uint8)[ids, offsets] = 127
        else:
            kc[ids, offsets] = float("nan")
            vc[ids, offsets] = float("nan")
    out = torch.full(q.shape, float("nan"), device=q.device, dtype=q.dtype)
    ws = DraftAttentionWorkspace(batch, n, maxq, q.device, provider=provider)

    def reference():
        refs = []
        begin = 0
        for b, (count, size) in enumerate(zip(qlens, lengths)):
            if not count:
                continue
            first = max(0, size - count - 2047)
            positions = torch.arange(first, size, device="cuda")
            ids = table[b, positions // page].long()
            slots = positions % page
            rk, rv = [
                (c[ids, slots].float() * (scale if fp8 else 1)).bfloat16()
                for c, scale in ((kc, ks), (vc, vs))
            ]
            if count > 1:
                rk[-count:] = k[begin : begin + count]
                rv[-count:] = v[begin : begin + count]
            rk, rv = [x.float().repeat_interleave(4, 1) for x in (rk, rv)]
            scores = (
                torch.einsum("qhd,khd->qhk", q[begin : begin + count].float(), rk)
                * 128**-0.5
            )
            qp = size - count + torch.arange(count, device="cuda")
            valid = positions[None, :] >= qp[:, None] - 2047
            scores.masked_fill_(~valid[:, None, :], -torch.inf)
            refs.append(torch.einsum("qhk,khd->qhd", scores.softmax(-1), rv))
            begin += count
        return torch.cat(refs)

    result.update(
        draft_case=args.draft_case,
        provider=provider,
        query_lengths=list(qlens),
        sequence_lengths=list(lengths),
        fp8_cache=fp8,
        padding=pad,
    )
    result["cases"].append(
        {
            "native_cache_write": {
                "pass": True,
                "torch_quantizer_bytes_equal": True,
                "unrelated_slots_untouched": True,
            }
        }
    )
    call = lambda: ws(q, k, v, kc, vc, table, cu, sl, ks, vs, out)
    call()
    torch.accelerator.synchronize()
    gold = reference()
    result["cases"].append({"initial": error(out[: sum(qlens)], gold)})
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    # Mutate live cache, query, block mappings and descales after capture.
    pool.copy_(clean_pool)
    q.mul_(0.7)
    k.mul_(1.1)
    v.mul_(0.3)
    ks.fill_(0.5)
    vs.fill_(1.3)
    table.copy_(table.flip(1).contiguous())
    lengths[:] = [
        max(count, size - 17) if count else 0 for count, size in zip(qlens, lengths)
    ]
    if batch > 1 and qlens[-1] > 0 and any(qlens[:-1]):
        qlens[-1] = 0
    sl.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
    cu.copy_(
        torch.tensor([0] + qlens, device="cuda", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
    )
    # Reversal may select unassigned but valid pages, which remain initialized.
    graph.replay()
    torch.accelerator.synchronize()
    result["cases"].append({"replay": error(out[: sum(qlens)], reference())})
